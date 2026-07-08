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
const GUARD_WINDOW = 480; // 30 ms at 16 kHz
const GUARD_RMS_FLOOR = 0.02; // window counts as voiced above this RMS
const MIN_VOICED_MS = 250; // real speech easily exceeds this; blips don't
const MIN_PEAK_ABS = 0.04; // near-silence hallucination cutoff

/** Voiced duration + peak of a PCM buffer (see phantom-turn guard). */
function voicedStats(samples: Float32Array): { voicedMs: number; peak: number } {
  let voiced = 0;
  let peak = 0;
  for (let start = 0; start + GUARD_WINDOW <= samples.length; start += GUARD_WINDOW) {
    let energy = 0;
    for (let i = start; i < start + GUARD_WINDOW; i++) {
      const s = samples[i];
      energy += s * s;
      const a = Math.abs(s);
      if (a > peak) peak = a;
    }
    if (Math.sqrt(energy / GUARD_WINDOW) > GUARD_RMS_FLOOR) voiced++;
  }
  return { voicedMs: voiced * 30, peak };
}
const MAX_UTTERANCE_MS = 28_000;
const BARGE_TICKS = 2; // sustained loud ticks required for the ASR barge path
const BARGE_MIN_WORDS = 1; // one echo-filtered committed word + loud = the user
// Energy barge-in: with echoCancellation on, sustained loud mic input during
// the assistant's reply is the user talking over it (not our own playback), so
// interrupt on energy alone — the ASR path often misses it because the mic
// picks up a mix of user + assistant and the self-echo filter drops it.
const BARGE_ENERGY_LEVEL = 0.08;
const BARGE_ENERGY_TICKS = 3; // ~450 ms sustained → interrupt
const BACKCHANNEL_MIN_MS = 2_000; // utterance length before a backchannel
const BACKCHANNEL_PAUSE_MIN = 450;
const BACKCHANNEL_PAUSE_MAX = 800;

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
  private sessionStart = 0;

  // User-speech tracking (listening phase).
  private speechStartAt = 0;
  private silenceStart = 0;
  private aboveTicks = 0;
  private aboveBargeTicks = 0; // consecutive ticks above the energy-barge level
  private backchannelUsed = false;
  // A barge-in continuation skips the phantom-turn guard (its sustained energy
  // already proved itself to the barge detector).
  private bargeContinuation = false;

  // Assistant / response tracking.
  private assistant: AssistantState | null = null;
  private respondPromise: Promise<void> | null = null;
  private proactiveSpeaking = false;
  private currentTtsText: string | null = null;
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
    if (level > BARGE_ENERGY_LEVEL) this.aboveBargeTicks++;
    else this.aboveBargeTicks = 0;

    // 1. Barge-in: stop the assistant the moment the user talks over it. Two
    //    independent triggers — sustained mic energy (robust: the ASR path
    //    frequently misses talk-over because the self-echo filter discards the
    //    user's words mixed with our playback), or a freshly-committed word.
    if (this.phase === "responding" && this.assistant) {
      const energyBarge = this.aboveBargeTicks >= BARGE_ENERGY_TICKS;
      const asrBarge =
        this.aboveTicks >= BARGE_TICKS &&
        this.transcriber.committedWordCount >= BARGE_MIN_WORDS;
      if (energyBarge || asrBarge) {
        this.handleBargeIn(now);
        return;
      }
    }

    if (this.phase === "responding") {
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

    // Phantom-turn guard: without enough voiced evidence in the captured PCM
    // this "utterance" was ambient noise, and transcribing near-silence makes
    // Whisper hallucinate ("Thank you." etc.). Discard before transcription.
    // Barge-in continuations skip the check — their sustained energy already
    // proved itself to the barge detector.
    if (!this.bargeContinuation) {
      const stats = voicedStats(this.capture.samples());
      if (stats.voicedMs < MIN_VOICED_MS || stats.peak < MIN_PEAK_ABS) {
        this.cb.onEvent("noise · discarded");
        this.startFreshListening();
        return;
      }
    }
    this.bargeContinuation = false;

    // Switch to responding immediately so the tick loop stops re-entering.
    this.phase = "responding";

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

    this.cb.onMetric({ asrLagMs });
    this.cb.onUserTurn(text);

    // Vision: the detector only *measures* (objects + colours). Precise factual
    // questions (count, colour, "what do you see") are answered directly from
    // those measurements — the 270M model deflects on them. Broader /
    // interpretive visual questions ("what am I doing", "does my room look
    // tidy") are handed to the LLM with the measured scene as grounding, so the
    // model reasons rather than us templating a reply.
    if (this.vision?.active && this.vision.matchesQuestion(text)) {
      const reply = this.vision.answer(text);
      this.cb.onEvent("eye · answering from the camera");
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
      content = facts
        ? `(Right now through the camera you can see ${facts}. You can't tell colours of clothing or fine detail beyond that.) ${text}`
        : text;
      this.cb.onEvent("eye · grounding from the camera");
    }
    this.history.push({ role: "user", content, t: this.elapsed() });
    this.maybeScheduleTimer(text);

    // Fresh audio window for the response phase: barge-in detection must see
    // only NEW committed words (spoken over the reply), not this turn's.
    this.capture.clear();
    transcriber.reset();
    this.resetUtterance();

    // Two-tier path: a tool intent runs as a background task while the fast
    // model stays present. Only one background task at a time; if one is
    // already running, fall through to a normal reply.
    const tool = this.backgroundTask ? null : detectTool(text);
    if (tool) {
      this.startToolTask(tool);
      return;
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
        endCause: this.turnMarks.endCause,
        transcript: this.turnMarks.transcript ?? "",
        reply: finalText,
        interrupted,
      });
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
      this.cb.onAssistantPartial(state.fullText);

      // Fastest first audio: flush the first clause as soon as a comma/colon/
      // semicolon appears (once there's enough to sound natural), so speech
      // starts after "The weather in Tokyo," instead of the whole sentence.
      if (!firstEmitted) {
        const clauseIdx = findClauseEnd(buffer);
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
    this.currentTtsText = state.spoken;
    return sentence;
  }

  // --- Barge-in ----------------------------------------------------------

  private handleBargeIn(now: number): void {
    this.bargeAt = now;
    this.assistant?.controller.abort();
    this.cb.onEvent("barge-in · stopping");
    // The interrupting speech is already being captured & transcribed (echo
    // cancellation keeps our own playback out of the buffer), so treat it as a
    // fresh user utterance already in progress — don't clear it.
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
/** Count whitespace-separated word tokens in a transcript string. */
function displayWordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
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
