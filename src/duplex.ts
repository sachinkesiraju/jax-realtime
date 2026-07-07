// Full-duplex micro-turn engine. The mic is always capturing; a ~150 ms tick
// loop applies a deterministic, priority-ordered policy — barge-in, user turn
// end (adaptive endpointing), backchannel, time-awareness, idle — over a
// continuously-running streaming ASR lane. TTS/LLM run on WebGPU; ASR runs on
// wasm (when available) so it can transcribe while the assistant speaks.

import { StreamingTranscriber, type StreamingUpdate } from "./asr/streaming";
import { VoiceCapture } from "./mic";
import { analyserLevel } from "./orb";
import type {
  ChatMessage,
  ChatModel,
  TTSVoice,
  VoicePipeline,
} from "./pipeline";

const TICK_MS = 150;
const START_LEVEL = 0.07; // speech onset / barge-in threshold
const STOP_LEVEL = 0.04; // below this counts as silence
const MIN_SPEECH_MS = 350; // ignore sub-blip "utterances"
const ENDPOINT_PUNCT_MS = 450; // silence to end a turn that ends in . ! ?
const ENDPOINT_SILENCE_MS = 800; // silence to end a turn otherwise
const MAX_UTTERANCE_MS = 28_000;
const BARGE_TICKS = 2; // sustained loud ticks required for barge-in
const BARGE_MIN_WORDS = 2; // committed (echo-filtered) words required
const BACKCHANNEL_MIN_MS = 2_000; // utterance length before a backchannel
const BACKCHANNEL_PAUSE_MIN = 450;
const BACKCHANNEL_PAUSE_MAX = 800;

const TERMINAL_PUNCT = /[.!?…]\s*$/;
const TIMER_UNIT = /(\d+)\s*(seconds?|secs?|minutes?|mins?)/i;
const TIMER_VERB = /(timer|remind|tell me|let me know|when|after|in)\b/i;

export type DuplexMetrics = {
  turnLatencyMs?: number;
  interruptStopMs?: number;
  asrLagMs?: number;
};

export interface DuplexCallbacks {
  /** Live captions for the in-progress user utterance. */
  onCaptions(committed: string, tentative: string): void;
  /** A user turn was finalized (commit a user transcript bubble). */
  onUserTurn(text: string): void;
  /** The assistant began a new reply (open an assistant bubble). */
  onAssistantStart(): void;
  /** The assistant's reply text grew. */
  onAssistantPartial(text: string): void;
  /** The assistant reply ended (interrupted keeps spoken-so-far + " —"). */
  onAssistantEnd(text: string, interrupted: boolean): void;
  /** One-line policy event for the ticker. */
  onEvent(text: string): void;
  /** Rolling latency metrics for the rail. */
  onMetric(metric: DuplexMetrics): void;
  /** A stage started/stopped actually computing (activity dots). */
  onStageActivity(stage: "asr" | "llm" | "tts", active: boolean): void;
  onError(error: unknown): void;
}

export interface DuplexConfig {
  pipeline: VoicePipeline;
  capture: VoiceCapture;
  getVoice: () => TTSVoice;
  getModel: () => ChatModel;
  callbacks: DuplexCallbacks;
}

type Phase = "listening" | "responding";

type AssistantState = {
  controller: AbortController;
  fullText: string; // everything the LLM has produced so far
  spoken: string; // sentences actually handed to TTS
};

type PendingTimer = {
  fireAt: number;
  amount: number;
  unit: string;
  fired: boolean;
};

export class DuplexSession {
  private readonly pipeline: VoicePipeline;
  private readonly capture: VoiceCapture;
  private readonly getVoice: () => TTSVoice;
  private readonly getModel: () => ChatModel;
  private readonly cb: DuplexCallbacks;

  private transcriber: StreamingTranscriber | null = null;
  private tick: ReturnType<typeof setInterval> | null = null;
  private running = false;

  readonly history: ChatMessage[] = [];

  private phase: Phase = "listening";
  private sessionStart = 0;

  // User-speech tracking (listening phase).
  private speechStartAt = 0;
  private silenceStart = 0;
  private aboveTicks = 0;
  private backchannelUsed = false;

  // Assistant / response tracking.
  private assistant: AssistantState | null = null;
  private respondPromise: Promise<void> | null = null;
  private proactiveSpeaking = false;
  private currentTtsText: string | null = null;
  private bargeAt = 0;

  // TTS analyser for the duplex orb core.
  private ttsAnalyser: AnalyserNode | null = null;
  private analyserBuffer = new Float32Array(1024);

  private timers: PendingTimer[] = [];

  constructor(config: DuplexConfig) {
    this.pipeline = config.pipeline;
    this.capture = config.capture;
    this.getVoice = config.getVoice;
    this.getModel = config.getModel;
    this.cb = config.callbacks;
  }

  // --- Orb level sources -------------------------------------------------

  micLevel(): number {
    return this.capture.level();
  }

  ttsLevel(): number {
    if (!this.ttsAnalyser) return 0;
    return analyserLevel(this.ttsAnalyser, this.analyserBuffer);
  }

  // --- Lifecycle ---------------------------------------------------------

  async start(): Promise<void> {
    if (this.running) return;
    await this.capture.start();
    this.capture.clear();
    await this.capture.resume();

    this.transcriber = new StreamingTranscriber(
      this.pipeline.asr,
      () => this.capture.samples(),
      (update) => this.onTranscript(update),
      () => (this.isAssistantAudible() ? this.currentTtsText : null),
      { minPassIntervalMs: 400, maxWindowSec: 28 },
    );
    this.transcriber.start();

    this.running = true;
    this.phase = "listening";
    this.sessionStart = performance.now();
    this.resetUtterance();
    this.tick = setInterval(() => this.onTick(), TICK_MS);
  }

  async stop(): Promise<void> {
    if (!this.running) return;
    this.running = false;
    if (this.tick !== null) {
      clearInterval(this.tick);
      this.tick = null;
    }
    // Abort any in-flight assistant response and wait for teardown.
    this.assistant?.controller.abort();
    if (this.respondPromise) {
      try {
        await this.respondPromise;
      } catch {
        // ignore
      }
    }
    await this.transcriber?.stop();
    this.transcriber = null;
    await this.capture.close();
    this.ttsAnalyser = null;
    this.currentTtsText = null;
    this.assistant = null;
    this.proactiveSpeaking = false;
  }

  // --- ASR callback ------------------------------------------------------

  private onTranscript(update: StreamingUpdate): void {
    // Only surface captions while the user (not the assistant echo) is talking.
    if (this.phase === "listening" && this.speechStartAt) {
      this.cb.onCaptions(update.committed, update.tentative);
    }
  }

  // --- Tick policy -------------------------------------------------------

  private onTick(): void {
    if (!this.running || !this.transcriber) return;
    const now = performance.now();
    const level = this.capture.level();

    if (level > START_LEVEL) this.aboveTicks++;
    else this.aboveTicks = 0;

    // 1. Barge-in.
    if (this.phase === "responding" && this.isAssistantAudible()) {
      if (
        this.aboveTicks >= BARGE_TICKS &&
        this.transcriber.committedWordCount >= BARGE_MIN_WORDS
      ) {
        this.handleBargeIn(now);
        return;
      }
    }

    if (this.phase === "responding") {
      // Nothing else to do while responding (barge-in handled above).
      return;
    }

    // Listening phase: track speech onset and silence.
    if (level > START_LEVEL) {
      if (!this.speechStartAt) this.speechStartAt = now;
      this.silenceStart = 0;
    } else if (level < STOP_LEVEL) {
      if (this.speechStartAt && !this.silenceStart) this.silenceStart = now;
    }

    // 4. Time awareness (independent of speech state).
    this.checkTimers(now);

    if (!this.speechStartAt) return; // idle

    const speechMs = now - this.speechStartAt;
    const trailingSilence = this.silenceStart ? now - this.silenceStart : 0;
    const committed = this.transcriber.committed;
    const endsTerminal = TERMINAL_PUNCT.test(committed);

    // 2. User turn end (adaptive endpointing).
    const endByPunct =
      endsTerminal && trailingSilence >= ENDPOINT_PUNCT_MS;
    const endBySilence = trailingSilence >= ENDPOINT_SILENCE_MS;
    const endByMax = speechMs >= MAX_UTTERANCE_MS;
    if (speechMs >= MIN_SPEECH_MS && (endByPunct || endBySilence || endByMax)) {
      void this.endUserTurn(this.silenceStart || now);
      return;
    }

    // 3. Backchannel (mid-utterance pause; does not end the turn).
    if (
      !this.backchannelUsed &&
      speechMs >= BACKCHANNEL_MIN_MS &&
      trailingSilence >= BACKCHANNEL_PAUSE_MIN &&
      trailingSilence < BACKCHANNEL_PAUSE_MAX &&
      !this.isAssistantAudible()
    ) {
      this.backchannelUsed = true;
      this.pipeline.tts.playBackchannel();
      this.cb.onEvent("backchannel");
    }
  }

  // --- User turn end -----------------------------------------------------

  private async endUserTurn(endOfSpeechAt: number): Promise<void> {
    const transcriber = this.transcriber;
    if (!transcriber) return;
    // Switch to responding immediately so the tick loop stops re-entering.
    this.phase = "responding";
    this.cb.onStageActivity("asr", true);

    let text = "";
    try {
      text = await transcriber.finalize();
    } catch (error) {
      this.cb.onError(error);
    } finally {
      this.cb.onStageActivity("asr", false);
    }
    const asrLagMs = performance.now() - endOfSpeechAt;

    if (!this.running) return;

    if (!text.trim()) {
      // Empty/noise turn: discard and go back to listening.
      this.startFreshListening();
      return;
    }

    this.cb.onMetric({ asrLagMs });
    this.cb.onUserTurn(text);
    this.history.push({ role: "user", content: text, t: this.elapsed() });
    this.maybeScheduleTimer(text);

    this.respondPromise = this.respond(endOfSpeechAt);
  }

  // --- Assistant response ------------------------------------------------

  private async respond(endOfSpeechAt: number): Promise<void> {
    const controller = new AbortController();
    const state: AssistantState = {
      controller,
      fullText: "",
      spoken: "",
    };
    this.assistant = state;
    this.currentTtsText = "";
    this.cb.onAssistantStart();
    this.cb.onStageActivity("llm", true);

    let firstAudioAt = 0;
    const speakStart = performance.now();

    try {
      const model = this.getModel();
      const stream = model.generateStream(this.history);
      const sentences = this.sentenceStream(stream, state);

      this.cb.onStageActivity("tts", true);
      const stats = await this.pipeline.tts.speakStream(
        this.getVoice(),
        sentences,
        {
          signal: controller.signal,
          onAnalyser: (analyser) => {
            this.ttsAnalyser = analyser;
          },
        },
      );
      if (stats.firstAudioMs > 0) firstAudioAt = speakStart + stats.firstAudioMs;
    } catch (error) {
      if (!controller.signal.aborted) this.cb.onError(error);
    } finally {
      this.cb.onStageActivity("llm", false);
      this.cb.onStageActivity("tts", false);
      this.ttsAnalyser = null;
      this.currentTtsText = null;
    }

    const interrupted = controller.signal.aborted;
    const finalText = interrupted
      ? `${state.spoken.trim()} [interrupted]`.trim()
      : state.fullText.trim();

    if (finalText.replace("[interrupted]", "").trim()) {
      this.history.push({
        role: "assistant",
        content: finalText,
        t: this.elapsed(),
      });
    }
    this.cb.onAssistantEnd(state.spoken.trim() || state.fullText.trim(), interrupted);

    if (interrupted) {
      if (this.bargeAt) {
        this.cb.onMetric({ interruptStopMs: performance.now() - this.bargeAt });
        this.bargeAt = 0;
      }
      // The user's interruption is already the next utterance; keep listening.
    } else {
      if (firstAudioAt > 0) {
        this.cb.onMetric({ turnLatencyMs: firstAudioAt - endOfSpeechAt });
      }
      this.startFreshListening();
    }

    this.assistant = null;
  }

  /**
   * Split the LLM delta stream into sentences, flushing on terminal punctuation
   * followed by whitespace or at ≥120 chars, and update the assistant bubble.
   */
  private async *sentenceStream(
    stream: AsyncGenerator<string, unknown, void>,
    state: AssistantState,
  ): AsyncGenerator<string, void, void> {
    let buffer = "";
    let result = await stream.next();
    while (!result.done) {
      if (state.controller.signal.aborted) {
        await stream.return?.(undefined);
        return;
      }
      const delta = result.value;
      buffer += delta;
      state.fullText += delta;
      this.cb.onAssistantPartial(state.fullText);

      let idx: number;
      while ((idx = findSentenceEnd(buffer)) !== -1) {
        const sentence = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx);
        if (sentence) yield this.trackSpoken(state, sentence);
      }
      if (buffer.length >= 120) {
        const sentence = buffer.trim();
        buffer = "";
        if (sentence) yield this.trackSpoken(state, sentence);
      }

      result = await stream.next();
    }

    if (!state.controller.signal.aborted) {
      const tail = buffer.trim();
      if (tail) yield this.trackSpoken(state, tail);
    }
  }

  private trackSpoken(state: AssistantState, sentence: string): string {
    state.spoken += (state.spoken ? " " : "") + sentence;
    this.currentTtsText = state.spoken;
    return sentence;
  }

  // --- Barge-in ----------------------------------------------------------

  private handleBargeIn(now: number): void {
    this.bargeAt = now;
    this.assistant?.controller.abort();
    this.cb.onEvent("barge-in · stopping");
    // The interrupting speech is already being captured & transcribed; treat it
    // as the start of a fresh user utterance.
    this.phase = "listening";
    this.speechStartAt = now - BARGE_TICKS * TICK_MS;
    this.silenceStart = 0;
    this.backchannelUsed = false;
    this.aboveTicks = 0;
  }

  // --- Time awareness ----------------------------------------------------

  private maybeScheduleTimer(text: string): void {
    if (!TIMER_VERB.test(text)) return;
    const match = TIMER_UNIT.exec(text);
    if (!match) return;
    const amount = parseInt(match[1], 10);
    if (!Number.isFinite(amount) || amount <= 0) return;
    const unit = /min/i.test(match[2]) ? "minute" : "second";
    const ms = unit === "minute" ? amount * 60_000 : amount * 1_000;
    this.timers.push({
      fireAt: performance.now() + ms,
      amount,
      unit,
      fired: false,
    });
    this.cb.onEvent(`timer set (${amount} ${unit}${amount === 1 ? "" : "s"})`);
  }

  private checkTimers(now: number): void {
    for (const timer of this.timers) {
      if (timer.fired || now < timer.fireAt) continue;
      // Fire only when nobody is talking.
      if (this.speechStartAt || this.isAssistantAudible() || this.responding()) {
        continue;
      }
      timer.fired = true;
      const unit = `${timer.unit}${timer.amount === 1 ? "" : "s"}`;
      const line = `That's ${timer.amount} ${unit} — time's up.`;
      this.cb.onEvent(`timer fired (${timer.amount} ${timer.unit})`);
      void this.speakProactive(line);
    }
    this.timers = this.timers.filter((t) => !t.fired);
  }

  private async speakProactive(text: string): Promise<void> {
    if (this.responding() || this.proactiveSpeaking) return;
    this.proactiveSpeaking = true;
    this.currentTtsText = text;
    this.cb.onAssistantStart();
    this.cb.onAssistantPartial(text);
    this.cb.onStageActivity("tts", true);
    try {
      await this.pipeline.tts.speak(this.getVoice(), text, {
        onAnalyser: (analyser) => {
          this.ttsAnalyser = analyser;
        },
      });
      this.history.push({
        role: "assistant",
        content: text,
        t: this.elapsed(),
      });
      this.cb.onAssistantEnd(text, false);
    } catch (error) {
      this.cb.onError(error);
    } finally {
      this.cb.onStageActivity("tts", false);
      this.ttsAnalyser = null;
      this.currentTtsText = null;
      this.proactiveSpeaking = false;
    }
  }

  // --- Helpers -----------------------------------------------------------

  private responding(): boolean {
    return this.phase === "responding" || this.assistant !== null;
  }

  private isAssistantAudible(): boolean {
    return this.assistant !== null || this.proactiveSpeaking;
  }

  private elapsed(): number {
    return (performance.now() - this.sessionStart) / 1000;
  }

  private resetUtterance(): void {
    this.speechStartAt = 0;
    this.silenceStart = 0;
    this.aboveTicks = 0;
    this.backchannelUsed = false;
  }

  /** Fresh listening window: drop buffered audio and reset ASR commit state. */
  private startFreshListening(): void {
    this.phase = "listening";
    this.capture.clear();
    this.transcriber?.reset();
    this.resetUtterance();
    this.cb.onCaptions("", "");
  }
}

/**
 * Index just after a sentence-ending punctuation that is followed by
 * whitespace, or -1. Requiring a trailing whitespace avoids flushing on
 * decimals like "3.14" mid-stream; the end-of-stream tail is flushed separately.
 */
function findSentenceEnd(buffer: string): number {
  for (let i = 0; i < buffer.length - 1; i++) {
    const c = buffer[i];
    if ((c === "." || c === "!" || c === "?" || c === "…") && /\s/.test(buffer[i + 1])) {
      return i + 1;
    }
  }
  return -1;
}
