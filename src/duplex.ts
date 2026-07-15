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
import {
  detectTool,
  type ToolCall,
  type ToolKind,
  type UiCard,
} from "./tools/tools";
import { TUNABLES, TURN_LOG, type TurnRecord } from "./tunables";
import type { VisionSession } from "./vision/vision";

// Latency-critical knobs (tick, endpoint windows, min speech) live in
// TUNABLES so the optimization bench can vary them at runtime.
const START_LEVEL = 0.05; // speech onset / barge-in threshold
// Phantom-turn guard: Whisper hallucinates text on near-silence ("Thank you."
// is its most famous), and a brief blip above START_LEVEL is enough to latch
// an "utterance" that then endpoints into a fake turn. Signal-based on purpose
// — no phrase blocklists, so a genuine quiet "thank you" passes.
// Evidence design (three attempts taught this): anything derived from the
// tick-sampled level meter is a coin flip — the meter decays between words
// and GPU work janks the tick cadence, so real 2 s speech and an ambient blip
// measure alike. Instead the guard reads the CAPTURED PCM at endpoint time and
// measures actual voiced duration (30 ms windows above an RMS floor) + peak
// amplitude. Deterministic, no sampling artifacts. Sub-threshold "utterances"
// are discarded before Whisper ever sees them; anything loud-but-short that
// slips through still hits the empty-transcript discard after transcription.
const GUARD_WINDOW = 160; // 10 ms at 16 kHz (fine enough to resolve a keystroke)
const GUARD_RMS_FLOOR = 0.02; // absolute floor for a voiced window
const GUARD_RMS_CEIL = 0.055; // never demand more than soft speech delivers
// The core discriminator is SUSTAIN, not loudness or total energy. A spoken
// word carries a continuous voiced run (the vowel) of ~100 ms+; a keystroke is
// a ~5-30 ms transient, and typing is a train of such transients separated by
// gaps — so its LONGEST run of consecutive voiced windows stays short even
// though individual clicks are loud and the total voiced time can add up.
// Requiring a minimum contiguous run rejects keyboard noise (which was being
// transcribed into invented conversations) while still admitting a snappy
// one-word reply. 80 ms sits in the wide gap between typing (~10-40 ms runs)
// and the shortest real word (~120-300 ms).
const MIN_VOICED_RUN_MS = 80;
// Peak amplitude a turn must reach to be real (ambient/HVAC swells peak
// ~0.03-0.06; real speech 0.2-0.7). A loud-but-transient click clears this, so
// peak alone can't gate it — the run length does.
const MIN_PEAK_ABS = 0.09;

/**
 * Peak amplitude and the LONGEST contiguous voiced run of a PCM buffer (see
 * phantom-turn guard). The voiced threshold is ADAPTIVE: real mics with
 * auto-gain boost quiet rooms until the ambient tone sits near any fixed floor,
 * so a fixed threshold counts room tone as speech. Speech clears the room tone
 * by a large ratio regardless of gain, so the threshold is 2× the buffer's
 * 20th-percentile window RMS (an ambient estimate — the buffer starts at
 * listening-start, so it holds pre-speech ambient), clamped to a floor/ceiling.
 * The longest run of consecutive above-threshold windows is what separates
 * sustained speech from staccato keyboard transients.
 */
function voicedStats(samples: Float32Array): { maxRunMs: number; peak: number } {
  const windowRms: number[] = [];
  let peak = 0;
  for (let start = 0; start + GUARD_WINDOW <= samples.length; start += GUARD_WINDOW) {
    let energy = 0;
    for (let i = start; i < start + GUARD_WINDOW; i++) {
      const s = samples[i];
      energy += s * s;
      const a = Math.abs(s);
      if (a > peak) peak = a;
    }
    windowRms.push(Math.sqrt(energy / GUARD_WINDOW));
  }
  const sorted = [...windowRms].sort((a, b) => a - b);
  const ambient = sorted.length ? sorted[Math.floor(sorted.length * 0.2)] : 0;
  const threshold = Math.min(
    GUARD_RMS_CEIL,
    Math.max(GUARD_RMS_FLOOR, ambient * 2),
  );
  const windowMs = (GUARD_WINDOW / 16_000) * 1000;
  let run = 0;
  let maxRun = 0;
  for (const rms of windowRms) {
    if (rms > threshold) {
      run++;
      if (run > maxRun) maxRun = run;
    } else {
      run = 0;
    }
  }
  return { maxRunMs: maxRun * windowMs, peak };
}
const MAX_UTTERANCE_MS = 28_000;
// Hard cap on how long the engine may stay in "responding". A real reply
// (generate + speak, even a long one with a background tool) finishes well
// inside this; exceeding it means the response path wedged, and the watchdog
// force-recovers to listening so the session can't die silently.
const RESPONDING_MAX_MS = 30_000;
// Cap on the rolling per-turn bench log so a long live session can't grow it
// (and its retained strings) without bound.
const TURN_LOG_MAX = 500;
const BARGE_TICKS = 2; // sustained loud ticks required for the ASR barge path
const BARGE_MIN_WORDS = 1; // one echo-filtered committed word + loud = the user
// Energy barge-in: with echoCancellation on, sustained loud mic input during
// the assistant's reply is the user talking over it (not our own playback), so
// interrupt on energy alone — the ASR path often misses it because the mic
// picks up a mix of user + assistant and the self-echo filter drops it.
// Adaptive energy barge-in. The threshold is the per-reply echo floor (the
// loudest the mic hears during the calibration window, when only our own
// playback is audible) times a ratio, with an absolute minimum so a silent
// echo floor doesn't make a whisper trigger. Fewer sustained ticks than before
// so short interjections ("wait", "stop") interrupt.
const BARGE_FLOOR_CALIB_TICKS = 3; // ~450 ms to estimate the echo floor
const BARGE_ENERGY_RATIO = 1.8; // user must clear the echo floor by this much
const BARGE_ENERGY_MIN = 0.05; // absolute floor (level units, min·4 RMS)
const BARGE_ENERGY_TICKS = 2; // ~300 ms above threshold → interrupt
// Post-fire continuation-merge (TUNABLES.continuationMerge): sustained ticks
// above START_LEVEL, inside the merge window, that count as "the user's
// speech resumed". Same shape as the listening-phase onset detector: one tick
// is a blip (a chair creak, a breath), two consecutive ticks (~300 ms) is
// speech. Deliberately NOT the energy-barge threshold — the merge window is
// silent (no reply audio yet), so there is no echo floor to clear and
// START_LEVEL is the right gate, exactly as in the listening phase.
const MERGE_RESUME_TICKS = 2;
const BACKCHANNEL_MIN_MS = 2_000; // utterance length before a backchannel
// Backchannel pause window. It sits BELOW the earliest endpoint threshold
// (endpointPunctMs = 380 ms) on purpose: the endpoint checks run first in the
// tick with early returns, so any window at/above 380 ms is shadowed — a
// punct-terminal turn endpoints at 380 ms and a plain turn at 620 ms before a
// [450,800) backchannel could ever fire. Placing it at [250,380) means a genuine
// mid-utterance pause is acknowledged before either endpoint fires, adding zero
// turn latency (the block never returns early / touches the endpoint logic).
const BACKCHANNEL_PAUSE_MIN = 250;
const BACKCHANNEL_PAUSE_MAX = 380;

const TERMINAL_PUNCT = /[.!?…]\s*$/;
const TIMER_UNIT = /(\d+)\s*(seconds?|secs?|minutes?|mins?)/i;
const TIMER_VERB = /(timer|remind|tell me|let me know|when|after|in)\b/i;

// Vision "Eye" proactive interjections. Persistence is time-based (~1.2 s per
// detector frame); a single 8 s cooldown gates all of them so they never
// dogpile. Lines kept as a small rule table per the spec.
const VISION_COOLDOWN_MS = 8_000;
const VISION_PERSON_PRESENT_MS = 3_600; // person established (≥3 frames)
const VISION_PERSON_GONE_MS = 3_600; // then absent (≥3 frames)
const VISION_PHONE_MS = 3_600; // phone persists (≥3 frames)
const VISION_SLOUCH_MS = 4_800; // slouch persists (≥4 frames)
const VISION_LINES = {
  stepAway: "Did you step away? I'll be here when you're back.",
  phone: "Phone again? I can wait.",
  slouch: "Hey — sit up straight.",
} as const;

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
  /** A background tool task was kicked off (render an in-progress chip). */
  onToolCall(kind: ToolKind, query: string): void;
  /** A background tool task resolved with a card to render. */
  onToolResult(card: UiCard): void;
  /** Background-activity indicator: pulses while a tool task runs. */
  onBackground(active: boolean): void;
  onError(error: unknown): void;
}

export interface DuplexConfig {
  pipeline: VoicePipeline;
  capture: VoiceCapture;
  getVoice: () => TTSVoice;
  getModel: () => ChatModel;
  callbacks: DuplexCallbacks;
  /** Optional webcam "Eye" stage for scene grounding + proactive lines. */
  vision?: VisionSession | null;
}

type Phase = "listening" | "responding";

type AssistantState = {
  controller: AbortController;
  fullText: string; // everything the LLM has produced so far
  spoken: string; // sentences actually handed to TTS
  // Continuation-merge (TUNABLES.continuationMerge) bookkeeping:
  /** While the merge window is open this reply is PROVISIONAL — the assistant
   *  bubble (onAssistantStart/Partial) is held back so a merged-away reply
   *  leaves zero UI trace. Cleared (and the bubble revealed) when the window
   *  closes at first audible audio. */
  uiDeferred: boolean;
  /** Set by handleContinuationMerge before aborting: tells respond()'s
   *  teardown that this reply was rescinded — no history entry, no bubble,
   *  no TURN_LOG record. The turn never happened. */
  merged: boolean;
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
  private vision: VisionSession | null;

  private transcriber: StreamingTranscriber | null = null;
  private tick: ReturnType<typeof setInterval> | null = null;
  private running = false;

  readonly history: ChatMessage[] = [];

  private phase: Phase = "listening";
  private respondingSince = 0; // perf.now() when the current reply began (watchdog)
  private sessionStart = 0;

  // User-speech tracking (listening phase).
  private speechStartAt = 0;
  private silenceStart = 0;
  private aboveTicks = 0;
  private aboveBargeTicks = 0; // consecutive ticks above the energy-barge level
  // Adaptive barge-in echo-floor calibration (reset each reply).
  private bargeFloor = 0;
  private bargeFloorTicks = 0;
  private backchannelUsed = false;
  // A barge-in continuation skips the phantom-turn guard (its sustained energy
  // already proved itself to the barge detector).
  private bargeContinuation = false;

  // Post-fire continuation-merge (TUNABLES.continuationMerge) state.
  // Window = [turn fired & respond() dispatched, first AUDIBLE reply chunk).
  // While it is open, the fired turn's undoable side effects are deferred
  // (pendingTurn below) and sustained mic energy rescinds the reply instead
  // of barging in — there is nothing audible to barge in ON yet.
  private mergeWindowOpen = false;
  private mergeResumeTicks = 0; // consecutive loud ticks inside the window
  // The fired turn's deferred side effects. `msg` is the history entry that
  // COULD NOT be deferred (the LLM prompt is built from history at
  // generateStream() time, so it must be in place before the reply starts) —
  // it is the one thing a merge has to UNDO (splice back out). Everything
  // else — the user bubble (onUserTurn), the timer parse, capture.clear() /
  // transcriber.reset() / resetUtterance() — is DEFERRED to window close
  // instead, because deferral is strictly simpler than undo: there is no
  // "remove bubble" callback surface to invent, no un-schedule-timer path,
  // and the capture/ASR state must stay alive anyway for the utterance to be
  // append-able. Cost of deferral: on a normal (unmerged) turn the user
  // bubble/timer land ~0.5–1.2 s later, at first audible audio — invisible
  // next to the reply latency it rides on.
  private pendingTurn: { text: string; msg: ChatMessage } | null = null;
  // Merges absorbed by the utterance currently in progress; the eventual
  // completed turn logs `merged: true` in its TurnRecord and resets this.
  private pendingMerges = 0;

  // Assistant / response tracking.
  private assistant: AssistantState | null = null;
  private respondPromise: Promise<void> | null = null;
  private proactiveSpeaking = false;
  private currentTtsText: string | null = null;
  // Onset filler text for the current reply (e.g. "So,"). JUDGMENT CALL on the
  // self-echo filter: the filler is audio-only — never shown in the transcript
  // or pushed to history — but it IS real sound from the speakers, and echo
  // cancellation is imperfect (see the adaptive barge-floor comments). If the
  // mic picks up "so"/"right"/"okay" while the assistant is audible and those
  // words aren't in currentTtsText, filterEcho's ≥70%-overlap test can pass
  // them through as user speech, feeding the ASR barge path a phantom word.
  // So the filler words are FOLDED INTO currentTtsText (via trackSpoken's
  // prefix below) but kept out of state.spoken/fullText — echo filtering sees
  // them, the UI and history never do. The over-filter risk is negligible:
  // a genuine interruption composed ≥70% of "so/right/okay" is exactly the
  // ambiguous double-talk the energy barge path (not ASR) is there to catch.
  private onsetPrefix = "";
  private bargeAt = 0;

  // TTS analyser for the duplex orb core.
  private ttsAnalyser: AnalyserNode | null = null;
  private analyserBuffer = new Float32Array(1024);

  // Bench instrumentation: stage marks for the turn currently being answered;
  // pushed to TURN_LOG when the response ends.
  private turnMarks: Partial<TurnRecord> = {};
  private lastEndCause: "punct" | "silence" | "max" = "punct";

  private timers: PendingTimer[] = [];

  // Two-tier tool use: one background fetch at a time; resolved speech is queued
  // and delivered on the next silence so it never talks over the user.
  private backgroundTask: Promise<void> | null = null;
  private pendingToolSpeech: string[] = [];

  // Vision proactive-interjection tracking (time-based persistence).
  private visionCooldownUntil = 0;
  private personSeen = false; // a person has been established this session
  private personPresentSince = 0;
  private personAbsentSince = 0;
  private stepAwayAnnounced = false;
  private phonePresentSince = 0;
  private phoneGoneSince = 0;
  private phoneAnnounced = false;
  private slouchSince = 0;
  private slouchAnnounced = false;

  constructor(config: DuplexConfig) {
    this.pipeline = config.pipeline;
    this.capture = config.capture;
    this.getVoice = config.getVoice;
    this.getModel = config.getModel;
    this.cb = config.callbacks;
    this.vision = config.vision ?? null;
  }

  /** Attach or detach the webcam "Eye" stage on a running session. */
  setVision(vision: VisionSession | null): void {
    this.vision = vision;
    // Reset proactive tracking so a freshly-attached camera starts clean.
    this.personSeen = false;
    this.personPresentSince = 0;
    this.personAbsentSince = 0;
    this.stepAwayAnnounced = false;
    this.phonePresentSince = 0;
    this.phoneAnnounced = false;
    this.slouchSince = 0;
    this.slouchAnnounced = false;
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
      {
        // Interval/window come from TUNABLES (read live by the loop). Pause
        // while the assistant is speaking so ASR doesn't steal the GPU from TTS.
        pauseWhile: () => this.isAssistantAudible(),
      },
    );
    this.transcriber.start();

    this.running = true;
    this.phase = "listening";
    this.sessionStart = performance.now();
    this.resetUtterance();
    this.tick = setInterval(() => this.onTick(), TUNABLES.tickMs);
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
    // Drop any queued tool speech; the background fetch (if any) resolves into
    // a no-op since the queue is cleared and the session is no longer running.
    this.pendingToolSpeech = [];
    this.backgroundTask = null;
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

    // 1. Barge-in: stop the assistant the moment the user talks over it.
    //    Energy-based, and ADAPTIVE — a fixed threshold failed in the wild: on
    //    real hardware the assistant's own playback leaks into the mic (echo
    //    cancellation is imperfect) and, worse, the mic's AGC ducks the user
    //    during double-talk, so the absolute level of a genuine interruption
    //    varies wildly by device. Instead we calibrate the echo/ambient floor
    //    over the reply's first few ticks (before the user could react) and
    //    fire when the level clears that floor by a ratio. The ASR path can't
    //    help here — it's paused during assistant speech to free the GPU.
    if (this.phase === "responding" && this.assistant) {
      // 1a. Continuation-merge check FIRST (cycle-5 law: in a lagging cascade
      //     ASR there is NO reliable pre-fire continuation signal, so patience
      //     is bought post-fire). While the merge window is open the reply has
      //     produced NO audible audio yet, so sustained mic energy can only be
      //     the user resuming the utterance — there is nothing to barge in on.
      //     Same onset detector as the listening phase (level > START_LEVEL,
      //     sustained), on a dedicated counter so a stale aboveTicks streak
      //     from before the endpoint can never trip it.
      if (this.mergeWindowOpen) {
        if (level > START_LEVEL) this.mergeResumeTicks++;
        else this.mergeResumeTicks = 0;
        if (this.mergeResumeTicks >= MERGE_RESUME_TICKS) {
          this.handleContinuationMerge();
          return;
        }
        // Barge-in echo-floor calibration is DEFERRED while the window is
        // open: the floor must measure the reply's playback echo, and nothing
        // is audible yet — calibrating on window silence would bake in a
        // too-low floor. bargeFloorTicks starts counting the tick after the
        // window closes, i.e. from the first audible audio, which is exactly
        // when the echo exists. (With continuationMerge off the window never
        // opens and calibration starts at respond() dispatch, as shipped.)
      } else if (this.bargeFloorTicks < BARGE_FLOOR_CALIB_TICKS) {
        // Calibration window: the loudest thing the mic hears now is our own
        // echo, so take the max as the floor to beat.
        this.bargeFloor = Math.max(this.bargeFloor, level);
        this.bargeFloorTicks++;
      } else {
        const threshold = Math.max(
          BARGE_ENERGY_MIN,
          this.bargeFloor * BARGE_ENERGY_RATIO,
        );
        if (level > threshold) this.aboveBargeTicks++;
        else this.aboveBargeTicks = 0;
        if (this.aboveBargeTicks >= BARGE_ENERGY_TICKS) {
          this.handleBargeIn(now);
          return;
        }
      }
    }

    if (this.phase === "responding") {
      // Watchdog: a reply should never take this long. If we're still
      // "responding" past the cap, something wedged (an unhandled error in the
      // response path, a stalled generation) and the session would otherwise be
      // dead forever — every tick early-returns here. Force-recover to
      // listening so the user isn't stuck talking to a frozen assistant.
      if (this.respondingSince && now - this.respondingSince > RESPONDING_MAX_MS) {
        this.cb.onEvent("recovered · response stalled");
        this.assistant?.controller.abort();
        this.assistant = null;
        this.proactiveSpeaking = false;
        this.startFreshListening();
      }
      // Nothing else to do while responding (barge-in handled above).
      return;
    }

    // Don't endpoint a user turn while a proactive line is still playing;
    // starting a second TTS stream would overlap audio on the same GPU lane.
    if (this.proactiveSpeaking) return;

    // Listening phase: track speech onset and silence. Anything below the
    // speech threshold counts toward silence — using a lower "stop" gate here
    // created a dead-band where a quiet-but-nonzero signal latched neither
    // state and the turn never endpointed. Brief within-word dips are absorbed
    // by the endpoint silence windows (450–800 ms), not this per-tick gate.
    if (level > START_LEVEL) {
      if (!this.speechStartAt) this.speechStartAt = now;
      this.silenceStart = 0;
    } else if (this.speechStartAt && !this.silenceStart) {
      this.silenceStart = now;
    }

    // 4. Time awareness (independent of speech state).
    this.checkTimers(now);

    // 4b. Vision "Eye" proactive interjections — only while idle (user not
    // speaking; assistant/responding already gated above).
    if (this.vision?.active && !this.speechStartAt) this.evaluateVision(now);

    // 4c. Two-tier tool result: deliver a queued background result on silence.
    if (
      this.pendingToolSpeech.length &&
      !this.speechStartAt &&
      !this.responding() &&
      !this.proactiveSpeaking
    ) {
      const line = this.pendingToolSpeech.shift()!;
      void this.speakProactive(line);
    }

    if (!this.speechStartAt) return; // idle

    const speechMs = now - this.speechStartAt;
    const trailingSilence = this.silenceStart ? now - this.silenceStart : 0;
    const committed = this.transcriber.committed;
    const endsTerminal = TERMINAL_PUNCT.test(committed);

    // 2. User turn end (adaptive endpointing). Patience modes widen the
    //    SILENCE window when the utterance doesn't look finished, so a
    //    mid-thought pause isn't mistaken for the end of the turn; the punct
    //    fast-path is never affected.
    // NOTE (cycle-5 campaign, all candidates rejected): "patience" endpointing
    // — extending these windows when the utterance looks unfinished — cannot
    // work pre-fire in this cascade. At a mid-clause pause the committed text
    // ends at the last complete sentence (terminal punct = false end-of-turn
    // signal), and the tentative tail that knows better lags the audio by more
    // than the punct window. The viable design is post-fire continuation-merge
    // (abort the reply if speech resumes before first audio); see BENCHMARKS.
    const endByPunct =
      endsTerminal && trailingSilence >= TUNABLES.endpointPunctMs;
    const endBySilence = trailingSilence >= TUNABLES.endpointSilenceMs;
    const endByMax = speechMs >= MAX_UTTERANCE_MS;
    if (speechMs >= TUNABLES.minSpeechMs && (endByPunct || endBySilence || endByMax)) {
      this.lastEndCause = endByPunct ? "punct" : endBySilence ? "silence" : "max";
      void this.endUserTurn(this.silenceStart || now);
      return;
    }

    // 3. Backchannel (mid-utterance pause; does not end the turn). Only when the
    //    committed text does NOT look turn-final — a terminal-punct tail means
    //    the user is finishing, not pausing mid-thought, and that turn is about
    //    to endpoint anyway.
    if (
      !this.backchannelUsed &&
      !endsTerminal &&
      speechMs >= BACKCHANNEL_MIN_MS &&
      trailingSilence >= BACKCHANNEL_PAUSE_MIN &&
      trailingSilence < BACKCHANNEL_PAUSE_MAX &&
      !this.isAssistantAudible() &&
      // Same voiced-evidence test the phantom-turn guard applies at endpoint,
      // applied BEFORE humming at the user: the adversarial bench caught the
      // engine backchanneling at keyboard noise (typing sustains the level
      // meter past BACKCHANNEL_MIN_MS, but its longest voiced run stays far
      // under a spoken word's). Runs at most once per utterance (the flag is
      // set regardless) so the PCM scan cost isn't paid every tick.
      this.utteranceSoundsVoiced()
    ) {
      this.pipeline.tts.playBackchannel();
      this.cb.onEvent("backchannel");
    }
  }

  /** Voiced-evidence check over the captured utterance so far (see the
   *  phantom-turn guard); marks the backchannel as used either way. */
  private utteranceSoundsVoiced(): boolean {
    this.backchannelUsed = true;
    const { maxRunMs, peak } = voicedStats(this.capture.samples());
    return maxRunMs >= MIN_VOICED_RUN_MS && peak >= MIN_PEAK_ABS;
  }

  // --- User turn end -----------------------------------------------------

  private async endUserTurn(endOfSpeechAt: number): Promise<void> {
    const transcriber = this.transcriber;
    if (!transcriber) return;

    // Phantom-turn guard: without enough voiced evidence in the captured PCM
    // this "utterance" was ambient noise, and transcribing near-silence makes
    // Whisper hallucinate ("Thank you." etc.). Discard before transcription.
    // Barge-in continuations skip the check — their sustained energy already
    // proved itself to the barge detector.
    if (!this.bargeContinuation) {
      const stats = voicedStats(this.capture.samples());
      // Reject anything that isn't a sustained voiced sound. Too quiet → always
      // ambient. Too short a contiguous voiced RUN → a transient: a keystroke,
      // a click, or a train of them from typing (each loud, but none sustained)
      // — exactly the noise that was getting transcribed into invented convos.
      // A real word, even a one-syllable "what?", carries a run well past the
      // bar, so genuine speech (including snappy replies) still passes.
      const tooQuiet = stats.peak < MIN_PEAK_ABS;
      const notSustained = stats.maxRunMs < MIN_VOICED_RUN_MS;
      if (tooQuiet || notSustained) {
        this.cb.onEvent("noise · discarded");
        this.startFreshListening();
        return;
      }
    }
    this.bargeContinuation = false;

    // Switch to responding immediately so the tick loop stops re-entering.
    this.phase = "responding";
    this.respondingSince = performance.now(); // watchdog clock

    // Bench instrumentation: per-turn stage marks (see tunables.ts).
    this.turnMarks = {
      endOfSpeech: endOfSpeechAt,
      fired: performance.now(),
      usedBestText: true,
      endCause: this.lastEndCause,
    };

    // Latency hillclimb: the streaming loop has already transcribed this
    // utterance incrementally, so prefer its result and skip the extra
    // multi-second Whisper finalize pass that used to dominate turn latency.
    // Only fall back to finalize() when streaming hasn't caught up (short/empty).
    let text = transcriber.bestText();
    if (displayWordCount(text) < 3) {
      this.turnMarks.usedBestText = false;
      this.cb.onStageActivity("asr", true);
      try {
        text = await transcriber.finalize();
      } catch (error) {
        this.cb.onError(error);
      } finally {
        this.cb.onStageActivity("asr", false);
      }
    }
    this.turnMarks.transcriptReady = performance.now();
    this.turnMarks.transcript = text;
    const asrLagMs = performance.now() - endOfSpeechAt;

    if (!this.running) return;

    if (!text.trim()) {
      // Empty/noise turn: discard and go back to listening.
      this.startFreshListening();
      return;
    }
    if (isDegenerateTranscript(text)) {
      // Whisper repetition loop ("All in all. All in all. …") — a decoder
      // artifact, not something the user said. Never answer it.
      this.cb.onEvent("asr · garbled, discarded");
      this.startFreshListening();
      return;
    }

    this.cb.onMetric({ asrLagMs });

    // Vision: the detector only *measures* (objects + colours). Precise factual
    // questions (count, colour, "what do you see") are answered directly from
    // those measurements — the small local model deflects on them. Broader /
    // interpretive visual questions ("what am I doing", "does my room look
    // tidy") are handed to the LLM with the measured scene as grounding, so the
    // model reasons rather than us templating a reply.
    // (No continuation-merge on this path: the answer is proactive speech, not
    // a respond() reply, so there is no pre-audio window to watch — shipped
    // immediate bookkeeping kept.)
    if (this.vision?.active && this.vision.matchesQuestion(text)) {
      const reply = this.vision.answer(text);
      this.cb.onEvent("eye · answering from the camera");
      this.cb.onUserTurn(text);
      this.history.push({ role: "user", content: text, t: this.elapsed() });
      this.capture.clear();
      transcriber.reset();
      this.resetUtterance();
      this.startFreshListening();
      void this.speakProactive(reply);
      return;
    }

    let content = text;
    if (this.vision?.active && this.vision.referencesVision(text)) {
      const facts = this.vision.sceneFacts();
      content = facts ? `[scene: ${facts}] ${text}` : text;
      this.cb.onEvent("eye · grounding from the camera");
    }
    // The history push happens NOW even when continuation-merge will defer the
    // rest: respond() builds the LLM prompt from history, so the user message
    // must be in place before the reply starts. A merge undoes exactly this
    // one push (see handleContinuationMerge).
    const userMsg: ChatMessage = { role: "user", content, t: this.elapsed() };
    this.history.push(userMsg);

    // Two-tier path: a tool intent runs as a background task while the fast
    // model stays present. Only one background task at a time; if one is
    // already running, fall through to a normal reply.
    let tool = this.backgroundTask ? null : detectTool(text);
    // Eye-as-oracle for lookups: when the lookup subject names something the
    // camera can currently see ("what is the person doing"), the turn is about
    // the scene, not the web — drop the tool so the scene-grounded LLM answers.
    // (Direct "tell me about the person" turns never reach here; matchesQuestion
    // answers them from measurements above.)
    if (
      tool?.kind === "lookup" &&
      this.vision?.active &&
      this.vision.seesSubject(tool.query)
    ) {
      this.cb.onEvent("eye · lookup subject is in frame, answering from scene");
      tool = null;
    }
    if (tool) {
      // Tool turns keep the shipped immediate bookkeeping (no merge window):
      // the holding line is proactive speech via startToolTask, not a
      // respond() reply, so there is no abortable pre-audio pipeline to
      // rescind — a resume during the holding line is just the user talking,
      // handled by the ordinary listening phase.
      this.cb.onUserTurn(text);
      this.maybeScheduleTimer(text);
      this.capture.clear();
      transcriber.reset();
      this.resetUtterance();
      this.startToolTask(tool);
      return;
    }

    if (TUNABLES.continuationMerge && this.lastEndCause !== "max") {
      // POST-FIRE CONTINUATION-MERGE window opens. Defer every side effect a
      // merge would otherwise need to undo (see the pendingTurn field comment
      // for the defer-vs-undo reasoning). Crucially capture.clear() /
      // transcriber.reset() are deferred too: the whisper streaming lane
      // happily keeps transcribing the still-growing buffer, so if the user
      // resumes we re-open the SAME utterance — audio and committed text
      // intact — instead of re-seeding anything. The response-phase contract
      // those calls used to serve ("barge-in ASR must see only NEW words")
      // is re-established at window close, which is exactly when barge-in
      // becomes possible (the reply is audible from then on).
      //
      // "max"-fired turns are EXEMPT: at a max fire the user is typically
      // still talking (the cap cut them off deliberately, to bound utterance
      // growth), so the resume detector would trip instantly and re-open the
      // just-capped utterance in an endless fire→merge→fire loop. The cap
      // must stay a hard stop.
      this.pendingTurn = { text, msg: userMsg };
      this.mergeWindowOpen = true;
      this.mergeResumeTicks = 0;
    } else {
      this.cb.onUserTurn(text);
      this.maybeScheduleTimer(text);
      // Fresh audio window for the response phase: barge-in detection must see
      // only NEW committed words (spoken over the reply), not this turn's.
      this.capture.clear();
      transcriber.reset();
      this.resetUtterance();
    }

    this.respondPromise = this.respond(endOfSpeechAt);
  }

  // --- Two-tier tool use -------------------------------------------------

  private startToolTask(tool: ToolCall): void {
    this.cb.onToolCall(tool.kind, tool.query);
    this.cb.onBackground(true);
    // Return to listening first (so speakProactive doesn't bail on the
    // "responding" phase), then speak the holding line while the fetch runs —
    // the user can keep talking / interrupt / be backchanneled meanwhile.
    this.startFreshListening();
    // Instant tools (calc/convert/clock) have no holding line — their result
    // arrives immediately and is spoken from the pending queue instead.
    if (tool.holding) void this.speakProactive(tool.holding);
    this.backgroundTask = tool
      .run()
      .then((result) => {
        // Render the card now; queue the spoken line for the next silence.
        this.cb.onToolResult(result.card);
        this.pendingToolSpeech.push(result.speech);
      })
      .catch((error) => {
        this.cb.onError(error);
      })
      .finally(() => {
        this.backgroundTask = null;
        this.cb.onBackground(false);
      });
  }

  // --- Assistant response ------------------------------------------------

  private async respond(endOfSpeechAt: number): Promise<void> {
    const controller = new AbortController();
    const state: AssistantState = {
      controller,
      fullText: "",
      spoken: "",
      // Merge window open ⇒ this reply is provisional: hold the assistant
      // bubble back so a merged-away reply leaves no UI trace ("the turn
      // never happened"). Revealed by closeMergeWindow() at first audio.
      uiDeferred: this.mergeWindowOpen,
      merged: false,
    };
    this.assistant = state;
    this.currentTtsText = "";
    this.onsetPrefix = "";
    // Recalibrate the adaptive barge-in echo floor for this reply.
    this.bargeFloor = 0;
    this.bargeFloorTicks = 0;
    this.aboveBargeTicks = 0;
    if (!state.uiDeferred) this.cb.onAssistantStart();
    this.cb.onStageActivity("llm", true);

    let firstAudioAt = 0;
    let onsetAudioAt = 0;
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
          // The onset filler is audio-only: it must reach the ASR self-echo
          // filter (via currentTtsText) but never the transcript/history —
          // see the onsetPrefix field comment for the full reasoning.
          onOnset: (text) => {
            this.onsetPrefix = text;
            this.currentTtsText = text;
            // MERGE WINDOW CLOSES AT THE ONSET, not at first synth audio.
            // The spec's window is "before the reply speaks" — and the onset
            // IS the reply speaking: real sound from the speakers (~0.5 s
            // in), so a resume after the user heard "So," is the user
            // talking OVER the assistant, i.e. a normal barge-in. Yes, that
            // makes the window short (~0.5 s) on onset-filler turns — that's
            // the honest reading of the spec; the alternative (merging after
            // an audible "So,") would rescind a reply the user already
            // responded to. The cycle-3 one-stream/one-clock law guarantees
            // onset ≤ first synth audio, so closing here is always the
            // earlier of the two. currentTtsText is set above BEFORE closing
            // so the echo filter is armed the instant closeMergeWindow's
            // transcriber.reset() opens the fresh window.
            this.closeMergeWindow();
          },
          // No-onset replies (onsetFiller off / no cached voice): the window
          // closes at the first synthesized chunk instead — the first moment
          // the reply is audible.
          onFirstAudio: () => this.closeMergeWindow(),
        },
      );
      if (stats.firstAudioMs > 0) firstAudioAt = speakStart + stats.firstAudioMs;
      if (stats.onsetAudioMs > 0) onsetAudioAt = speakStart + stats.onsetAudioMs;
    } catch (error) {
      if (!controller.signal.aborted) this.cb.onError(error);
    } finally {
      this.cb.onStageActivity("llm", false);
      this.cb.onStageActivity("tts", false);
      this.ttsAnalyser = null;
      this.currentTtsText = null;
    }

    const interrupted = controller.signal.aborted;

    if (state.merged) {
      // CONTINUATION MERGE: this reply was rescinded before it ever became
      // audible — the turn NEVER HAPPENED. So, unlike a barge-in abort:
      // - no assistant history entry and no interrupted bubble (the bubble
      //   was never shown: uiDeferred held it back for the whole window);
      // - no TURN_LOG record (turnMarks dropped) — the eventual turn over
      //   the full merged utterance logs instead, with merged: true;
      // - no startFreshListening — handleContinuationMerge already put the
      //   engine back in listening with the utterance still open (capture
      //   buffer and streaming-ASR commit state intact, speechStartAt
      //   continuing), so clearing anything here would destroy the merge.
      this.turnMarks = {};
      if (this.assistant === state) this.assistant = null;
      return;
    }

    // A reply that died with the merge window still open (stop()/watchdog
    // abort or a TTS/LLM failure before any audio) still owes the deferred
    // user-turn side effects — flush them so the history entry that already
    // exists isn't invisible in the UI. No-op when the window closed normally.
    this.closeMergeWindow();

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

    // Bench instrumentation: complete and log this turn's stage record.
    if (this.turnMarks.endOfSpeech) {
      TURN_LOG.push({
        endOfSpeech: this.turnMarks.endOfSpeech,
        fired: this.turnMarks.fired ?? 0,
        transcriptReady: this.turnMarks.transcriptReady ?? 0,
        usedBestText: this.turnMarks.usedBestText ?? true,
        firstDelta: this.turnMarks.firstDelta ?? 0,
        firstSentence: this.turnMarks.firstSentence ?? 0,
        firstAudio: firstAudioAt,
        // Absent (not 0) when no filler played, so the bench can distinguish
        // "onsetFiller off / no cached PCM" from a degenerate timestamp.
        onsetAudio: onsetAudioAt > 0 ? onsetAudioAt : undefined,
        endCause: this.turnMarks.endCause,
        transcript: this.turnMarks.transcript ?? "",
        reply: finalText,
        interrupted,
        // Absent (not false) when no merge happened, mirroring onsetAudio's
        // convention — the bench counts turns with merged === true.
        merged: this.pendingMerges > 0 ? true : undefined,
      });
      // The merges are consumed by the turn that finally completed over the
      // whole utterance; a fresh utterance starts its count at zero.
      this.pendingMerges = 0;
      // Bounded ring: the bench only ever inspects recent turns, and a long
      // live session would otherwise grow this array (and its retained
      // transcript/reply strings) without limit.
      if (TURN_LOG.length > TURN_LOG_MAX) TURN_LOG.shift();
      this.turnMarks = {};
    }

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

    // Only clear if this is still the active response: after a barge-in the
    // user may already have started the next turn (a new respond() with its own
    // state), and nulling it here would break that turn's barge-in + teardown.
    if (this.assistant === state) this.assistant = null;
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
    let firstEmitted = false;
    let result = await stream.next();
    while (!result.done) {
      if (state.controller.signal.aborted) {
        await stream.return?.(undefined);
        return;
      }
      const delta = result.value;
      if (!this.turnMarks.firstDelta) this.turnMarks.firstDelta = performance.now();
      buffer += delta;
      state.fullText += delta;
      // While the merge window is open the reply is provisional — deltas
      // accumulate in state.fullText but stay off-screen so a merged-away
      // reply leaves no trace; closeMergeWindow() replays the accumulated
      // text when the window closes.
      if (!state.uiDeferred) this.cb.onAssistantPartial(state.fullText);

      // Fastest first audio: flush the first clause as soon as a comma/colon/
      // semicolon appears (once there's enough to sound natural), so speech
      // starts after "The weather in Tokyo," instead of the whole sentence.
      // If no punctuation shows up, flush at a WORD BOUNDARY once ~2× the
      // clause minimum has accumulated — otherwise the reply text is fully
      // written on screen while the voice still waits for the first sentence
      // to complete before it can even start synthesizing.
      if (!firstEmitted) {
        let clauseIdx = findClauseEnd(buffer);
        if (clauseIdx === -1 && buffer.length >= TUNABLES.firstClauseMinChars * 2) {
          const lastSpace = buffer.lastIndexOf(" ");
          if (lastSpace >= TUNABLES.firstClauseMinChars) clauseIdx = lastSpace + 1;
        }
        if (clauseIdx !== -1) {
          const clause = buffer.slice(0, clauseIdx).trim();
          buffer = buffer.slice(clauseIdx);
          if (clause) {
            firstEmitted = true;
            yield this.trackSpoken(state, clause);
          }
        }
      }

      let idx: number;
      while ((idx = findSentenceEnd(buffer)) !== -1) {
        const sentence = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx);
        if (sentence) {
          firstEmitted = true;
          yield this.trackSpoken(state, sentence);
        }
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
    if (!this.turnMarks.firstSentence) {
      this.turnMarks.firstSentence = performance.now();
    }
    state.spoken += (state.spoken ? " " : "") + sentence;
    // Echo filter sees filler + reply (both are audible from the speakers);
    // state.spoken stays filler-free so the UI/history never show the onset.
    this.currentTtsText = this.onsetPrefix
      ? `${this.onsetPrefix} ${state.spoken}`
      : state.spoken;
    return sentence;
  }

  // --- Continuation merge (TUNABLES.continuationMerge) --------------------

  /**
   * Close the merge window and flush the fired turn's deferred side effects.
   * Called (a) from respond()'s speakStream callbacks at the first AUDIBLE
   * chunk — the onset filler when one plays, else the first synthesized chunk
   * (from then on resumed speech is a normal barge-in), and (b) from
   * respond()'s teardown as a safety flush for replies that die before any
   * audio. Idempotent: no-op once the window is closed (or was never opened,
   * e.g. continuationMerge off).
   */
  private closeMergeWindow(): void {
    if (!this.mergeWindowOpen) return;
    this.mergeWindowOpen = false;
    this.mergeResumeTicks = 0;

    // Deferred user-turn side effects: the turn is now definitive (the reply
    // is audible), so show the user bubble and parse the timer intent.
    const pending = this.pendingTurn;
    this.pendingTurn = null;
    if (pending) {
      this.cb.onUserTurn(pending.text);
      this.maybeScheduleTimer(pending.text);
    }

    // Deferred fresh-audio window (see endUserTurn): from here on, committed
    // ASR words must be NEW speech only — the response-phase contract that
    // barge-in relies on. Doing this at window close instead of at fire is
    // what made the utterance merge-able (the buffer stayed alive).
    this.capture.clear();
    this.transcriber?.reset();
    this.resetUtterance();

    // Reveal the deferred assistant bubble with everything the LLM has
    // produced so far; subsequent deltas stream into it normally.
    const state = this.assistant;
    if (state?.uiDeferred) {
      state.uiDeferred = false;
      this.cb.onAssistantStart();
      if (state.fullText) this.cb.onAssistantPartial(state.fullText);
    }
  }

  /**
   * The user's speech resumed inside the merge window: rescind the in-flight
   * reply and re-open the utterance so it APPENDS (never restarts). Abort
   * mechanics are exactly barge-in's (controller.abort); the bookkeeping is
   * deliberately different — see respond()'s state.merged teardown branch.
   */
  private handleContinuationMerge(): void {
    this.mergeWindowOpen = false;
    this.mergeResumeTicks = 0;
    this.pendingMerges++;

    // Undo the one side effect that couldn't be deferred: the user-turn
    // history push (the LLM prompt needed it). Splice by identity — a
    // proactive line or tool result could in principle have appended entries
    // after it, so "pop the last entry" would be wrong.
    const pending = this.pendingTurn;
    this.pendingTurn = null;
    if (pending) {
      const idx = this.history.lastIndexOf(pending.msg);
      if (idx !== -1) this.history.splice(idx, 1);
    }

    // Abort exactly like barge-in; state.merged tells respond()'s teardown
    // to suppress history/UI/TURN_LOG (the turn never happened).
    const state = this.assistant;
    if (state) {
      state.merged = true;
      state.controller.abort();
    }
    this.cb.onEvent("continuation · merged");

    // Re-open the utterance. speechStartAt was never reset (resetUtterance
    // was deferred), so it CONTINUES from the original onset — endpointing,
    // the backchannel gate and the MAX_UTTERANCE_MS cap all measure the whole
    // merged utterance, and the capture buffer + streaming-ASR commit state
    // are untouched, so the eventual transcript covers all of it. Only the
    // trailing-silence clock resets: the resume that got us here is speech.
    this.phase = "listening";
    this.respondingSince = 0;
    this.silenceStart = 0;
    this.aboveBargeTicks = 0;
  }

  // --- Barge-in ----------------------------------------------------------

  private handleBargeIn(now: number): void {
    this.bargeAt = now;
    this.assistant?.controller.abort();
    this.cb.onEvent("barge-in · stopping");
    // The interrupting speech is already being captured & transcribed (echo
    // cancellation keeps our own playback out of the buffer), so treat it as a
    // fresh user utterance already in progress — don't clear it.
    // BUFFER HYGIENE (TUNABLES.bargePreRollMs, 0 = off): "don't clear" must
    // not mean "keep everything". The buffer at this moment spans the whole
    // reply period — imperfectly-cancelled reply echo plus ambient noise that
    // PRECEDE the user's barge words — and the first post-barge ASR passes
    // run after respond()'s teardown nulls currentTtsText, so the ≥70%
    // echo filter is disarmed and the stale audio transcribes as user speech.
    // Keep only the last ~1.2 s of PCM (the energy detector needed ~2 loud
    // ticks ≈ 300 ms to fire, so the pre-roll comfortably covers the user's
    // first words) and reset the streaming lane's commit state so any text
    // committed from reply-period audio — or an in-flight pass over the
    // untrimmed window (generation-checked and discarded by reset) — can't
    // leak into this turn's transcript. Echo filtering of whatever reply
    // audio remains inside the pre-roll is unchanged (filterEcho still runs
    // while the assistant is audible).
    if (TUNABLES.bargePreRollMs > 0) {
      this.capture.trimToLast(TUNABLES.bargePreRollMs);
      this.transcriber?.reset();
    }
    this.phase = "listening";
    this.speechStartAt = now - BARGE_ENERGY_TICKS * TUNABLES.tickMs;
    this.silenceStart = 0;
    this.backchannelUsed = false;
    this.aboveTicks = 0;
    this.aboveBargeTicks = 0;
    // The interrupting speech already proved itself (sustained energy above the
    // barge threshold) — mark it so the phantom-turn guard doesn't second-guess
    // a genuine barge-in utterance.
    this.bargeContinuation = true;
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

  // --- Vision proactive interjections ------------------------------------

  private evaluateVision(now: number): void {
    const vision = this.vision;
    if (!vision) return;
    if (this.responding() || this.proactiveSpeaking) return;

    const { personPresent, phonePresent, slouching } = vision.state;

    // Track person presence/absence streaks.
    if (personPresent) {
      if (!this.personPresentSince) this.personPresentSince = now;
      this.personAbsentSince = 0;
      if (now - this.personPresentSince >= VISION_PERSON_PRESENT_MS) {
        this.personSeen = true;
        this.stepAwayAnnounced = false;
      }
    } else {
      if (!this.personAbsentSince) this.personAbsentSince = now;
      this.personPresentSince = 0;
    }

    // Track phone streak. Re-arm the announcement only after the phone has
    // been genuinely gone for a while — detection-score jitter around the
    // threshold must not make the line repeat while the phone sits in frame.
    if (phonePresent) {
      if (!this.phonePresentSince) this.phonePresentSince = now;
      this.phoneGoneSince = 0;
    } else {
      this.phonePresentSince = 0;
      if (!this.phoneGoneSince) this.phoneGoneSince = now;
      if (this.phoneAnnounced && now - this.phoneGoneSince > 60_000) {
        this.phoneAnnounced = false;
      }
    }

    // Track slouch streak.
    if (slouching) {
      if (!this.slouchSince) this.slouchSince = now;
    } else {
      this.slouchSince = 0;
      this.slouchAnnounced = false;
    }

    if (now < this.visionCooldownUntil) return;

    // Rule 1: person was established, now gone for ≥3 frames.
    if (
      this.personSeen &&
      !this.stepAwayAnnounced &&
      this.personAbsentSince &&
      now - this.personAbsentSince >= VISION_PERSON_GONE_MS
    ) {
      this.stepAwayAnnounced = true;
      this.personSeen = false;
      this.fireVision(now, "eye · stepped away", VISION_LINES.stepAway);
      return;
    }

    // Rule 2: phone newly appears and persists ≥3 frames.
    if (
      !this.phoneAnnounced &&
      this.phonePresentSince &&
      now - this.phonePresentSince >= VISION_PHONE_MS
    ) {
      this.phoneAnnounced = true;
      this.fireVision(now, "eye · phone spotted", VISION_LINES.phone);
      return;
    }

    // Rule 3: slouch persists ≥4 frames.
    if (
      !this.slouchAnnounced &&
      this.slouchSince &&
      now - this.slouchSince >= VISION_SLOUCH_MS
    ) {
      this.slouchAnnounced = true;
      this.fireVision(now, "eye · slouching", VISION_LINES.slouch);
    }
  }

  private fireVision(now: number, event: string, line: string): void {
    this.visionCooldownUntil = now + VISION_COOLDOWN_MS;
    this.cb.onEvent(event);
    void this.speakProactive(line);
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

  /** True when the audio pipeline is using the GPU (an ASR pass is in flight or
   *  the assistant is speaking) — the vision stage checks this to yield. */
  audioActive(): boolean {
    return this.isAssistantAudible() || this.pipeline.asr.isBusy;
  }

  private elapsed(): number {
    return (performance.now() - this.sessionStart) / 1000;
  }

  private resetUtterance(): void {
    this.speechStartAt = 0;
    this.silenceStart = 0;
    this.aboveTicks = 0;
    this.aboveBargeTicks = 0;
    this.backchannelUsed = false;
    this.bargeContinuation = false;
  }

  /** Fresh listening window: drop buffered audio and reset ASR commit state. */
  private startFreshListening(): void {
    this.phase = "listening";
    this.respondingSince = 0;
    this.capture.clear();
    this.transcriber?.reset();
    this.resetUtterance();
    // A merged-into utterance that ends up discarded here (phantom guard,
    // empty/degenerate transcript) takes its absorbed merges with it — they
    // must not tag some later, unrelated turn as merged.
    this.pendingMerges = 0;
    this.cb.onCaptions("", "");
  }
}

/**
 * Index just after a sentence-ending punctuation that is followed by
 * whitespace, or -1. Requiring a trailing whitespace avoids flushing on
 * decimals like "3.14" mid-stream; the end-of-stream tail is flushed separately.
 */
/** Count whitespace-separated word tokens in a transcript string. */
function displayWordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

/**
 * Whisper's repetition-loop failure mode: on ambiguous audio the decoder can
 * emit one phrase over and over ("All in all. All in all. All in all…"). That
 * is a decode artifact, not speech — a real utterance of ≥6 words has far more
 * lexical variety. Signal-based (a repetition statistic), no phrase lists.
 */
function isDegenerateTranscript(text: string): boolean {
  const tokens = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .split(/\s+/)
    .filter(Boolean);
  if (tokens.length < 6) return false;
  const unique = new Set(tokens).size;
  return unique / tokens.length < 0.4;
}

function findSentenceEnd(buffer: string): number {
  for (let i = 0; i < buffer.length - 1; i++) {
    const c = buffer[i];
    if ((c === "." || c === "!" || c === "?" || c === "…") && /\s/.test(buffer[i + 1])) {
      return i + 1;
    }
  }
  return -1;
}

/**
 * Index just after the first clause break (comma/colon/semicolon or sentence
 * end) followed by whitespace, but only once ≥18 chars have accumulated so the
 * first spoken fragment isn't a choppy one-word stub. Used only for the very
 * first chunk, to minimize time-to-first-audio on longer opening sentences.
 */
function findClauseEnd(buffer: string): number {
  const MIN = TUNABLES.firstClauseMinChars;
  for (let i = 0; i < buffer.length - 1; i++) {
    const c = buffer[i];
    const isBreak =
      c === "," || c === ";" || c === ":" || c === "." || c === "!" || c === "?" || c === "…";
    if (isBreak && i + 1 >= MIN && /\s/.test(buffer[i + 1])) {
      return i + 1;
    }
  }
  return -1;
}
