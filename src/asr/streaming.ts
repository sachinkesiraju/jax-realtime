// Streaming transcription on top of SpeechRecognizer. Runs Whisper passes
// back-to-back over the growing utterance window and applies a LocalAgreement-2
// commit policy so the UI can show stable "committed" text plus a live
// "tentative" tail. Also filters the assistant's own TTS out of the hypothesis
// (self-echo) so barge-in detection only fires on real user speech.

import type {
  ASRConfidence,
  ASRResult,
  SpeechRecognizer,
} from "../pipeline";
import { TUNABLES } from "../tunables";

const SAMPLE_RATE = 16_000;

export type StreamingUpdate = {
  committed: string;
  tentative: string;
  lastChangeAt: number;
};

export type StreamingOptions = {
  /** Minimum time between the starts of two transcribe passes. */
  minPassIntervalMs?: number;
  /** Max utterance window fed to Whisper, in seconds. */
  maxWindowSec?: number;
  /** Below this much audio (seconds) a pass is skipped. */
  minWindowSec?: number;
  /** When this returns true, skip passes (e.g. while the assistant speaks, to
   *  keep ASR off the GPU so TTS generation stays smooth). */
  pauseWhile?: () => boolean;
  /** Commit the whole hypothesis on a single high-confidence pass instead of
   *  requiring two-pass LocalAgreement-2. Defaults to TUNABLES.asrFastCommit. */
  fastCommit?: boolean;
  /** avgLogProb floor for fastCommit. Defaults to TUNABLES.asrFastCommitThreshold. */
  fastCommitThreshold?: number;
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Lowercase, punctuation-stripped word list. */
function normalizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/** Word tokens preserving original casing/punctuation for display. */
function displayWords(text: string): string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}

function commonPrefixLen(a: string[], b: string[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (
    i < n &&
    a[i].toLowerCase().replace(/[^\p{L}\p{N}']/gu, "") ===
      b[i].toLowerCase().replace(/[^\p{L}\p{N}']/gu, "")
  ) {
    i++;
  }
  return i;
}

export class StreamingTranscriber {
  private active = false;
  private loopPromise: Promise<void> | null = null;
  private prevWords: string[] = [];
  private committedWords: string[] = [];
  private tentativeText = "";
  private latestConfidence: ASRConfidence | null = null;
  private lastChangeAt = 0;
  // Bumped by reset(); passes started under an older generation are discarded
  // so a stale hypothesis can't repopulate freshly-cleared commit state.
  private generation = 0;

  constructor(
    private asr: SpeechRecognizer,
    /** Returns the current utterance PCM (16 kHz mono, since utterance start). */
    private getSamples: () => Float32Array,
    /** Called after every pass with the current committed/tentative split. */
    private onUpdate: (update: StreamingUpdate) => void,
    /** Current assistant TTS line while it's speaking, else null (echo filter). */
    private getAssistantUtterance: () => string | null,
    private opts: StreamingOptions = {},
  ) {
    this.lastChangeAt = performance.now();
  }

  get committed(): string {
    return this.committedWords.join(" ");
  }

  get committedWordCount(): number {
    return this.committedWords.length;
  }

  get tentative(): string {
    return this.tentativeText;
  }

  /** Best streaming hypothesis paired with the decoder scores that produced it. */
  bestResult(): { text: string; confidence: ASRConfidence | null } {
    return {
      text: `${this.committed} ${this.tentativeText}`.trim(),
      confidence: this.latestConfidence,
    };
  }

  get lastChange(): number {
    return this.lastChangeAt;
  }

  start(): void {
    if (this.active) return;
    this.active = true;
    this.reset();
    this.loopPromise = this.loop();
  }

  async stop(): Promise<void> {
    this.active = false;
    if (this.loopPromise) {
      try {
        await this.loopPromise;
      } catch {
        // Swallow; stopping.
      }
      this.loopPromise = null;
    }
  }

  /** Clear commit state for a fresh utterance (loop keeps running). */
  reset(): void {
    this.generation++;
    this.prevWords = [];
    this.committedWords = [];
    this.tentativeText = "";
    this.latestConfidence = null;
    this.lastChangeAt = performance.now();
  }

  /** One final pass over the whole current window. */
  async finalize(): Promise<{
    text: string;
    confidence: ASRConfidence | null;
  }> {
    const fallback = this.bestResult();
    const samples = this.windowSamples();
    if (samples.length < SAMPLE_RATE * (this.opts.minWindowSec ?? 0.2)) {
      return fallback;
    }
    // Wait out any in-flight streaming pass so the final pass actually runs
    // (the recognizer allows one transcription at a time).
    for (let waited = 0; this.asr.isBusy && waited < 2_000; waited += 50) {
      await sleep(50);
    }
    let result: ASRResult;
    try {
      result = await this.asr.transcribeWithConfidence(
        samples,
        samples.length / SAMPLE_RATE,
      );
    } catch {
      return fallback;
    }
    const filtered = this.filterEcho(result.text).trim();
    return filtered ? { text: filtered, confidence: result.confidence } : fallback;
  }

  private windowSamples(): Float32Array {
    const maxWindow =
      (this.opts.maxWindowSec ?? TUNABLES.asrMaxWindowSec) * SAMPLE_RATE;
    const samples = this.getSamples();
    return samples.length > maxWindow
      ? samples.subarray(samples.length - maxWindow)
      : samples;
  }

  private async loop(): Promise<void> {
    const minWindow = (this.opts.minWindowSec ?? 0.2) * SAMPLE_RATE;

    while (this.active) {
      // Read per-iteration so the bench can tune it live.
      const minInterval = this.opts.minPassIntervalMs ?? TUNABLES.asrPassIntervalMs;
      const passStart = performance.now();
      const passGeneration = this.generation;
      const samples = this.windowSamples();

      if (
        samples.length < minWindow ||
        this.asr.isBusy ||
        this.opts.pauseWhile?.()
      ) {
        await sleep(minInterval);
        continue;
      }

      let result: ASRResult;
      try {
        result = await this.asr.transcribeWithConfidence(
          samples,
          samples.length / SAMPLE_RATE,
        );
      } catch {
        // Busy or transient error; back off and retry.
        await sleep(minInterval);
        continue;
      }
      if (!this.active) break;
      // Discard results from a pass over a window that has since been reset.
      if (passGeneration !== this.generation) continue;

      this.ingest(result);

      const elapsed = performance.now() - passStart;
      if (elapsed < minInterval) await sleep(minInterval - elapsed);
    }
  }

  private ingest(result: ASRResult): void {
    const filtered = this.filterEcho(result.text);
    const words = displayWords(filtered);

    // Options win when set; otherwise fall back to the live TUNABLES value.
    const fastCommit = this.opts.fastCommit ?? TUNABLES.asrFastCommit;
    const fastCommitThreshold =
      this.opts.fastCommitThreshold ?? TUNABLES.asrFastCommitThreshold;
    const avgLogProb = result.confidence?.avgLogProb ?? null;

    let committedWords: string[];
    let tentative: string;
    if (
      fastCommit &&
      avgLogProb !== null &&
      avgLogProb >= fastCommitThreshold
    ) {
      // Fast commit: this pass is confident enough to commit the whole
      // hypothesis now — no empty tail, no second-pass agreement required.
      committedWords = words;
      tentative = "";
    } else {
      // LocalAgreement-2: the longest common word-prefix of the previous and
      // current hypotheses is committed; the newest hypothesis's tail is tentative.
      const commonLen = commonPrefixLen(this.prevWords, words);
      committedWords = words.slice(0, commonLen);
      tentative = words.slice(commonLen).join(" ");
    }

    const prevCommitted = this.committedWords.join(" ");
    const nextCommitted = committedWords.join(" ");
    if (nextCommitted !== prevCommitted || tentative !== this.tentativeText) {
      this.lastChangeAt = performance.now();
    }

    this.committedWords = committedWords;
    this.tentativeText = tentative;
    this.latestConfidence = result.confidence;
    this.prevWords = words;

    this.onUpdate({
      committed: nextCommitted,
      tentative,
      lastChangeAt: this.lastChangeAt,
    });
  }

  /**
   * If the assistant is speaking and ≥70% of the hypothesis words appear in its
   * current TTS line, treat the whole hypothesis as self-echo and drop it.
   */
  private filterEcho(hypothesis: string): string {
    const assistant = this.getAssistantUtterance();
    if (!assistant) return hypothesis;
    const hypWords = normalizeWords(hypothesis);
    if (hypWords.length === 0) return hypothesis;
    const ttsWords = new Set(normalizeWords(assistant));
    let present = 0;
    for (const word of hypWords) if (ttsWords.has(word)) present++;
    return present / hypWords.length >= 0.7 ? "" : hypothesis;
  }
}
