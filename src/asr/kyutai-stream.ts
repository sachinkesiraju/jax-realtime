// Streaming Kyutai "ear": drives the parity-verified Mimi encoder
// (mimi-encode.ts) + stt-1b decoder (kyutai-stt.ts) over the live mic buffer,
// implementing the same Transcriber contract duplex.ts consumes from the
// Whisper StreamingTranscriber — so the duplex engine only chooses which one
// to construct (TUNABLES.asrEngine) and everything downstream is unchanged.
//
// How this lane differs from the Whisper lane, and why:
//   * Whisper re-transcribes the GROWING window every pass and needs
//     LocalAgreement-2 to stabilize a committed prefix (hypotheses churn).
//     Kyutai's delayed-streams model is genuinely streaming: each 80 ms frame
//     is encoded to 32 Mimi codes and decoded to ONE text token exactly once,
//     and tokens are never revised. So ALL decoded text is "committed" and
//     `tentative` is always "" — there is no second hypothesis to disagree
//     with. The commit/agreement machinery has nothing to do here.
//   * The model's text lags its audio by a trained-in ~0.5 s (≈6 frames).
//     Nothing implements that delay — it's baked into the weights — but it
//     shapes the turn-end story: by the time the duplex endpoint fires
//     (≥380–620 ms of trailing silence, which the mic captured as REAL room
//     silence and this loop already processed), most of the text has flushed.
//     finalize() drains whatever frames remain plus ~1.2 s of synthetic
//     silence to push out the tail (the HF feature extractor pads
//     audio_delay_seconds + 1.0 s = 1.5 s; we spend a bit less and accept the
//     tradeoff, see FLUSH_FRAMES).
//   * Self-echo: the Whisper lane word-filters the assistant's own TTS out of
//     hypotheses (currentTtsText overlap). This lane deliberately SKIPS that
//     for the first integration and relies on (a) pauseWhile() — no frames are
//     processed while the assistant is audible, exactly like the Whisper loop
//     pauses — and (b) the duplex energy barge-in, which never needed ASR.
//     Risk accepted: after a barge-in the backlog includes a user+echo mix
//     that gets transcribed verbatim; echoCancellation on the mic stream is
//     what keeps that mostly-user. Documented tradeoff, revisit if the live
//     bench shows echo text leaking into turns.
//   * The mic captures 16 kHz PCM (mic.ts); Mimi wants 24 kHz. The ratio is a
//     clean 2/3, so a linear resampler (same interpolation the app already
//     uses in features.ts resampleMono, reimplemented here over absolute
//     positions so per-frame chunks can't drift) feeds exact 1920-sample
//     frames from a running frame cursor.
//
// Concurrency model: encoder and decoder states are single-stream and
// mutated in place, so ALL GPU work (loop batches, finalize, state rebuilds)
// is serialized on one promise chain (`enqueue`). reset() must take effect
// synchronously for the duplex tick (it reads `committed` right after), so it
// clears the JS-side text/cursor immediately, bumps a generation counter that
// makes any in-flight batch abandon its results, and queues the state rebuild
// on the chain where it can't race a dispatch in flight.

import { tree } from "@jax-js/jax";

import type { KyutaiRecognizer } from "../pipeline";
import {
  createSttState,
  decodeSttTokens,
  STT_CONFIG,
  type SttState,
  sttStep,
} from "./kyutai-stt";
import {
  createMimiEncodeState,
  encodeFrame,
  MIMI_CONFIG,
  type MimiEncodeState,
} from "./mimi-encode";
import type { StreamingUpdate, Transcriber } from "./transcriber";

const MIC_RATE = 16_000;
const FRAME_OUT = MIMI_CONFIG.frameSize; // 1920 samples @ 24 kHz = 80 ms
// One 24 kHz frame consumes exactly 1280 mic samples (16k * 0.08) — the 2/3
// rate ratio keeps the frame cursor integer-exact with no drift.
const RATE_RATIO = MIC_RATE / MIMI_CONFIG.sampleRate; // 2/3

// Loop cadence: ~2 Mimi frames of audio accumulate per wake. Each frame costs
// ~45 ms of GPU (encode ~20 + decode ~23), so a wake's batch (~2 frames)
// finishes well inside the interval and the lane stays caught up while
// remaining bursty enough for the vision stage to interleave frames.
const TICK_MS = 160;
// Backlog batch cap per wake (≈2 s of audio, ~1.1 s of GPU). After a pause
// (assistant speaking) the buffered audio can be long; chunking the catch-up
// keeps each chain link bounded so a new pauseWhile()/reset() takes effect
// within one cap instead of after the whole backlog.
const MAX_FRAMES_PER_BATCH = 25;
// finalize() tail flush: 15 frames = 1.2 s of synthetic silence, ≈0.7 s of
// GPU at ~45 ms/frame. The reference pads 1.5 s; 1.2 s keeps the worst-case
// finalize under ~1 s of compute while comfortably covering the ~0.5 s
// trained-in text delay (+ safety margin). If live benching shows even this
// too slow, 0.75 s (9-10 frames) still clears the delay with less margin.
const FLUSH_FRAMES = 15;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export type KyutaiStreamingOptions = {
  /** When this returns true, skip processing (e.g. while the assistant
   *  speaks, to keep ASR off the GPU so TTS generation stays smooth). The
   *  backlog is processed on resume — at ~45 ms of GPU per 80 ms frame the
   *  lane catches up faster than real time. */
  pauseWhile?: () => boolean;
};

export class KyutaiStreamingTranscriber implements Transcriber {
  private active = false;
  private loopPromise: Promise<void> | null = null;
  // Serializes ALL model access (batches, finalize, rebuilds) — see module
  // comment. Every link swallows its error so one failure can't wedge the
  // chain forever.
  private chain: Promise<void> = Promise.resolve();
  // Bumped by reset(); work started under an older generation abandons its
  // results so a stale frame can't repopulate freshly-cleared text.
  private generation = 0;

  // Model streaming state, (re)built lazily on the serialized chain.
  private encState: MimiEncodeState | null = null;
  private sttState: SttState | null = null;
  private bosStepDone = false; // decoder step 0 (bos frame) run this utterance
  private framesConsumed = 0; // frame cursor into the utterance buffer

  private tokens: number[] = [];
  private committedText = "";
  private lastChangeAt = 0;
  // Set by finalize(): the utterance's decoder state has consumed synthetic
  // silence and MUST NOT eat more real frames; every endUserTurn path resets
  // shortly after finalize, this just closes the one-tick race window.
  private finalized = false;

  /** Wall-clock of the last finalize() (backlog + silence flush), for the
   *  bench. TURN_LOG's transcriptReady-fired covers it too when used. */
  lastFinalizeMs = 0;

  constructor(
    private asr: KyutaiRecognizer,
    /** Returns the current utterance PCM (16 kHz mono, since utterance start
     *  — reset() is always paired with capture.clear(), see Transcriber). */
    private getSamples: () => Float32Array,
    /** Called whenever the committed text grows. */
    private onUpdate: (update: StreamingUpdate) => void,
    private opts: KyutaiStreamingOptions = {},
  ) {
    this.lastChangeAt = performance.now();
  }

  get committed(): string {
    return this.committedText;
  }

  /** Always "" — the model streams monotonically; there is no unstable
   *  hypothesis tail (see module comment). */
  get tentative(): string {
    return "";
  }

  get lastChange(): number {
    return this.lastChangeAt;
  }

  /** Everything decoded so far (no extra model work — the loop has already
   *  processed the trailing real silence by endpoint time). */
  bestText(): string {
    return this.committedText.trim();
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
    // Release the encoder/decoder state after any in-flight work drains.
    await this.enqueue(async () => this.disposeStates());
  }

  /**
   * Fresh utterance: clear the transcript + frame cursor NOW (the duplex tick
   * reads `committed` synchronously right after) and rebuild the
   * encoder+decoder streaming state on the serialized chain (a fresh KV/conv
   * state mirrors capture.clear()'s fresh audio window — the model must not
   * attend across utterance boundaries it will never see audio for).
   */
  reset(): void {
    this.generation++;
    this.tokens = [];
    this.committedText = "";
    this.framesConsumed = 0;
    this.bosStepDone = false;
    this.finalized = false;
    this.lastChangeAt = performance.now();
    void this.enqueue(async () => {
      this.disposeStates();
    });
  }

  /**
   * Turn-end flush: drain any remaining whole frames from the live buffer,
   * pad the partial tail frame with zeros, then run FLUSH_FRAMES of synthetic
   * silence through encoder+decoder so the trained-in ~0.5 s text delay
   * flushes the last words out. Bounded: backlog is normally ≤2 frames while
   * listening (the loop keeps up), so the cost is ≈(1 + FLUSH_FRAMES) × 45 ms
   * ≈ 0.7 s of GPU; `lastFinalizeMs` records the real number per call.
   */
  async finalize(): Promise<string> {
    const gen = this.generation;
    const t0 = performance.now();
    await this.enqueue(async () => {
      if (gen !== this.generation || this.finalized) return;
      this.asr.markBusy(true);
      try {
        await this.ensureStates(gen);
        // 1. Remaining whole frames of real audio.
        await this.processAvailable(gen, Number.POSITIVE_INFINITY, true);
        if (gen !== this.generation) return;
        // 2. The partial tail frame (real samples + zero padding), if any new
        //    audio is left, then pure-silence flush frames. Both go through
        //    resampleFrame, which zero-fills past the end of the buffer —
        //    beyond the tail frame every frame is exactly zeros.
        const samples = this.getSamples();
        const consumedSrc = this.framesConsumed * FRAME_OUT * RATE_RATIO;
        const hasTail = samples.length > consumedSrc;
        const flushTotal = FLUSH_FRAMES + (hasTail ? 1 : 0);
        for (let i = 0; i < flushTotal; i++) {
          if (gen !== this.generation) return;
          await this.stepFrame(
            resampleFrame(samples, this.framesConsumed),
            gen,
          );
          this.framesConsumed++;
        }
        this.finalized = true;
      } finally {
        this.asr.markBusy(false);
      }
    });
    this.lastFinalizeMs = performance.now() - t0;
    return this.committedText.trim();
  }

  // --- Internals ---------------------------------------------------------

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.chain.then(fn);
    this.chain = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  private async loop(): Promise<void> {
    while (this.active) {
      if (!this.opts.pauseWhile?.() && !this.finalized) {
        await this.enqueue(async () => {
          const gen = this.generation;
          if (!this.active || this.finalized || gen !== this.generation) return;
          this.asr.markBusy(true);
          try {
            await this.ensureStates(gen);
            await this.processAvailable(gen, MAX_FRAMES_PER_BATCH);
          } finally {
            this.asr.markBusy(false);
          }
        }).catch(() => {
          // Transient GPU error: back off one tick and try again — matching
          // the Whisper loop's swallow-and-retry posture.
        });
      }
      await sleep(TICK_MS);
    }
  }

  /** Encode+decode every WHOLE frame currently available (up to `cap`),
   *  bailing between frames on reset/pause/stop so a long backlog can't hold
   *  the chain (and the GPU) hostage. finalize() sets `isFinalize` and must
   *  not stop for a pause or a session stop mid-drain — the assistant isn't
   *  audible during endUserTurn, and its result is awaited either way. */
  private async processAvailable(
    gen: number,
    cap: number,
    isFinalize = false,
  ): Promise<void> {
    for (let n = 0; n < cap; n++) {
      if (gen !== this.generation) return;
      // Mid-batch bail: a reply starting mid-catch-up should pause the lane
      // within one frame (~45 ms), not one batch (~1.1 s).
      if (!isFinalize && (!this.active || this.opts.pauseWhile?.())) return;
      const samples = this.getSamples();
      if (availableFrames(samples.length) <= this.framesConsumed) return;
      await this.stepFrame(resampleFrame(samples, this.framesConsumed), gen);
      if (gen !== this.generation) return;
      this.framesConsumed++;
    }
  }

  /** One frame through encoder → decoder → token ingest. Assumes states
   *  exist (ensureStates ran on this chain link). */
  private async stepFrame(
    pcm: Float32Array<ArrayBuffer>,
    gen: number,
  ): Promise<void> {
    const { codes, preQuant } = encodeFrame(this.asr.mimi, this.encState!, pcm);
    preQuant.dispose();
    const ids = await codes.data();
    // A reset may have landed during the readback; the stale states get
    // rebuilt on the chain right after this link, so just drop the frame.
    if (gen !== this.generation) return;
    const { tokenId, hidden } = await sttStep(
      this.asr.stt,
      this.sttState!,
      ids as ArrayLike<number>,
    );
    hidden.dispose();
    if (gen !== this.generation) return;
    this.ingestToken(tokenId);
  }

  private ingestToken(tokenId: number): void {
    this.tokens.push(tokenId);
    // Most frames emit <pad> (no text); only re-decode when the transcript
    // could have changed. decodeSttTokens drops specials/8000 itself, but the
    // cheap JS-side check keeps the common no-op path allocation-free.
    if (tokenId <= STT_CONFIG.padTokenId || tokenId >= 8000) return;
    const text = decodeSttTokens(this.asr.tokenizer, this.tokens);
    if (text === this.committedText) return;
    this.committedText = text;
    this.lastChangeAt = performance.now();
    this.onUpdate({
      committed: text,
      tentative: "",
      lastChangeAt: this.lastChangeAt,
    });
  }

  /** Build fresh streaming states + run the decoder's bos step (step 0
   *  consumes the audio-bos frame, not audio — see kyutai-stt.ts). */
  private async ensureStates(gen: number): Promise<void> {
    if (!this.encState) this.encState = createMimiEncodeState(this.asr.mimi);
    if (!this.sttState) this.sttState = createSttState(this.asr.stt);
    if (!this.bosStepDone) {
      const bosFrame = new Int32Array(STT_CONFIG.numCodebooks).fill(
        STT_CONFIG.audioBosTokenId,
      );
      const { tokenId, hidden } = await sttStep(
        this.asr.stt,
        this.sttState,
        bosFrame,
      );
      hidden.dispose();
      if (gen !== this.generation) return;
      this.bosStepDone = true;
      this.ingestToken(tokenId);
    }
  }

  /** Dispose GPU-side state (null-safe: downsampleState is null before the
   *  first frame, so a generic tree.dispose over the struct would throw). */
  private disposeStates(): void {
    if (this.encState) {
      tree.dispose([this.encState.convStates, this.encState.kvCaches]);
      this.encState.downsampleState?.dispose();
      this.encState = null;
    }
    if (this.sttState) {
      tree.dispose(this.sttState.caches);
      this.sttState = null;
    }
  }
}

/** Whole 24 kHz frames fully covered by `srcLen` 16 kHz samples: output index
 *  i needs source position 2i/3 with a RIGHT neighbor for interpolation, so
 *  the last safe index is floor(1.5·(srcLen−2)). */
function availableFrames(srcLen: number): number {
  if (srcLen < 2) return 0;
  const outSamples = Math.floor(1.5 * (srcLen - 2)) + 1;
  return Math.floor(outSamples / FRAME_OUT);
}

/**
 * Linearly resample frame `frameIdx` (1920 samples @ 24 kHz) out of the 16 kHz
 * utterance buffer. Positions are ABSOLUTE (i = frameIdx·1920 + j, source pos
 * 2i/3) so consecutive frames are phase-exact with no cumulative drift —
 * unlike resampling per-chunk. Positions past the end of the buffer produce
 * zeros (only finalize() ever reads there: the padded tail frame and the
 * synthetic-silence flush frames).
 */
function resampleFrame(
  src: Float32Array,
  frameIdx: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(FRAME_OUT);
  const base = frameIdx * FRAME_OUT;
  for (let j = 0; j < FRAME_OUT; j++) {
    const pos = (base + j) * RATE_RATIO;
    const left = Math.floor(pos);
    if (left >= src.length) break; // rest stays zero (silence padding)
    const right = Math.min(left + 1, src.length - 1);
    const mix = pos - left;
    out[j] = src[left] * (1 - mix) + src[right] * mix;
  }
  return out;
}
