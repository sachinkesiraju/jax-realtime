// Runtime-tunable latency/performance knobs, grouped by "region" for the
// map-reduce optimization harness (docs/BENCHMARKS.md). Values are read at
// use-time (not captured at construction) so the in-browser bench can change
// them between sessions without a rebuild. Defaults are the shipped values.

export const TUNABLES = {
  // region: engine
  /** Micro-turn policy tick. Read at session start. */
  tickMs: 150,

  // region: endpoint
  /** Silence to end a turn whose committed text ends in . ! ? */
  endpointPunctMs: 380,
  /** Silence to end a turn otherwise. */
  endpointSilenceMs: 620,
  /** Ignore sub-blip "utterances" shorter than this. */
  minSpeechMs: 350,

  // region: asr
  /** Minimum time between the starts of two streaming Whisper passes. */
  asrPassIntervalMs: 150,
  /** Max utterance window fed to Whisper, seconds. */
  asrMaxWindowSec: 28,
  /**
   * ASR decode sampler. "js" reads the full ~51k-vocab logits back and applies
   * the timestamp/text gating on the CPU (shipped, and the only path that
   * preserves the gate semantics exactly). "gpu" is reserved for a device-side
   * reduction path; it is NOT exactly equivalent (the forceTimestamp decision
   * compares a logSumExp over the timestamp range against a masked max over the
   * text range, and a topK candidate list cannot guarantee the masked argmax
   * winner is present), so it currently routes to the "js" implementation —
   * gating is never approximated silently.
   */
  asrSampler: "js" as "js" | "gpu",

  // region: llm
  /** Cap on generated tokens per reply (keeps spoken replies short). */
  llmMaxNewTokens: 96,
  /**
   * Bucket size (tokens) for padding the SmolLM full prefill; 0 = off
   * (shipped behavior). jax-js trace caches key on avals (shapes), and every
   * conversation turn has a NEW prompt length T — so all 32 layer prefill
   * jits re-trace + recompile EVERY turn (measured: a 250-token prefill is
   * ~334 ms warm vs ~1004 ms on first encounter of a length; turn-latency
   * benches show llmFirst growing 219→2129 ms as history grows). Padding the
   * prompt up to the next multiple of the bucket makes trace shapes repeat
   * across turns, so a warm bucket costs only the (small) extra FLOPs of the
   * pad tokens. Exact by construction: pads sit at the end, logits are read
   * at the last REAL token, and pad KV slots are overwritten before they can
   * ever be attended (see runSmolLmPrefill / runBucketedPrefill comments).
   * Shipped 64 (cycle 6): MAP llmFirst median 674–1002 → 347 ms and turn
   * latency 1815–1965 → 1366 ms; holdout (unseen clip) confirmed with no
   * reversal (turn 1713 → 1191 ms fused, llmFirst flat ~250 ms instead of
   * growing past 1 s). Equivalence-gated on-device: bucketed vs unbucketed
   * logits argmax identical, max |Δ| 3.6e-5 (fp16 reduction-order noise) —
   * benchPrefillEquivalence(250, 64).
   */
  llmPrefillBucket: 64,
  /**
   * Cap on the number of chat messages kept when formatting the LLM prompt
   * (whole user/assistant pairs). 0 = unlimited. Shipped at 16 (8 exchanges):
   * unbounded history let the brain's prefill grow every turn until a long session
   * crawled (and risked a capacity throw that wedged the response path); a
   * rolling window keeps recent context while bounding per-turn cost.
   */
  llmMaxHistoryTurns: 16,

  // region: tts-split
  /** Min chars before the first clause is flushed to TTS early. */
  firstClauseMinChars: 18,

  // region: tts-generation
  /**
   * Fuse the per-frame Pocket TTS decode into as few jitted dispatches as
   * possible, the same lever as the LLM's fused decode step. When on:
   *   - the flow-LM decode step (input proj → 6 streaming-transformer layers →
   *     out-norm → EOS head → LSD/flow decode) collapses from ~8 jit dispatches
   *     plus ~10 eager ops into ONE jitted dispatch (`runFlowLMStepFused`), and
   *   - the Mimi decode (quantizer conv → upsample → 2 decoder-transformer
   *     layers → SEANet decoder) collapses from ~3 jit dispatches plus ~12 eager
   *     ops into ONE jitted dispatch (`runMimiDecodeFused`).
   * The per-frame command-buffer submit count drops from ~35 to ~2 (plus the
   * two readbacks playTTS already does). Trace caches key on avals, so per-frame
   * dynamics (position/offset/kv-length, noise) are passed as np.Array inputs
   * and the trace is reused across frames (it only re-traces on the stepwise
   * KV-cache capacity growth, exactly as the unfused path does).
   *
   * Numerically the fused path runs the identical math in the identical order
   * (the inline helpers are verbatim copies of the jitted originals with the
   * inner jit calls inlined to avoid nested-jit boundaries); the flow-LM prefill
   * (step 0) still uses the unfused path (fuse-decode-only, like the LLM). A/B with `window.__pipeline().tts.benchSynth(sentence, {fused})`.
   * Shipped on: bench (fixed seed) showed 1291→1010 ms gen (~22%), realtime
   * factor 0.40→0.31, with identical frame count both paths (same EOS decision,
   * same audio duration) — equivalent output, fewer dispatches.
   */
  ttsFusedStep: true,
  /**
   * Bucket size (tokens) for padding the Pocket TTS flow-LM prefill text;
   * 0 = off (shipped). Same disease as llmPrefillBucket, different patient:
   * playTTS's step-0 prefill (always the UNFUSED runFlowLMStep — see
   * inference.ts `fuseFlow = ttsFusedStep && step > 0`) pushes a
   * [voiceLen + textLen + 1, 1024] activation through the 6 jitted
   * streaming-transformer layers plus the jitted out-norm, and textLen is the
   * sentence's token count — so every NEW sentence length re-traces and
   * recompiles those jits on the critical path to first audio (the 90–380 ms
   * per-reply variance in TTS first-audio). Bucketing textLen makes the
   * prefill trace shapes repeat across sentences.
   *
   * Unlike the LLM case there is no logits-gather to fix: the flow-LM reads
   * its output (and the EOS logit) at the LAST position, which is the BOS
   * latent concatenated AFTER the text embeds, so padding can never shift the
   * readout. Pad = LEADING spaces prepended to the prepared text and
   * re-encoded through the real tokenizer (SpeechSynthesizer's
   * encodeTextBucketed), mirroring the 8-leading-space pad prepareTextPrompt
   * already applies to every <5-word phrase — the one padding this exact
   * model is known to treat as neutral (it is Kyutai's own reference
   * behavior). End-side padding is deliberately avoided: spaces between the
   * sentence-final "." and the latent positions are an arrangement the model
   * never saw in training and risk shifting EOS timing / trailing artifacts.
   *
   * Default 0 until the bench proves it AND a listen confirms large pads add
   * no audible leading silence. A/B: benchTtsPrefill() (prefill-only, shows
   * re-trace vs warm per length) and benchSynth(text, { bucket }).
   */
  ttsPrefillBucket: 0,

  // region: tts-onset
  /**
   * Speak a short pre-rendered onset filler ("So," / "Right," / "Okay, so")
   * the instant a reply's TTS stream opens, to mask the ~1.3–1.8 s real turn
   * latency that the single-GPU serialization law says we cannot lower. The
   * fillers are synthesized to PCM once at load (zero runtime GPU cost, same
   * machinery as the backchannels) and are audio-only — never shown in the
   * transcript or stored in history, because they are a vocal gesture, not
   * content.
   *
   * Cycle-3 law (docs/BENCHMARKS.md, Campaign A — SHIPPED then REVERTED): the
   * first attempt played the filler through a SEPARATE short-lived
   * AudioContext, so the real reply began over/into the filler's tail and the
   * hand-off stop() clipped it mid-word — it sounded broken even though every
   * timing metric looked good. The recorded law: filler and reply must be ONE
   * gapless stream on ONE clock. This redo schedules the cached onset PCM as
   * the first chunk of the SAME streaming player the reply uses, so overlap
   * is structurally impossible (the player's nextStartTime clock serializes
   * every chunk; worst case is dead air between filler and reply, never a
   * collision). Default false: ships only if the bench AND a human listen
   * pass. Bench half passed (cycle 6): first sound at ~510–805 ms after end
   * of user speech vs ~1.2–1.8 s for the real reply, with real-reply latency
   * unchanged within noise. The EARS half of the gate is still open — flip
   * this on and listen for whether "So, … <pause> …reply" beats silence
   * before it defaults on.
   */
  onsetFiller: false,

  // region: tools (campaign 2 — delegation)
  /**
   * Tool routing breadth. "conservative" is the shipped behavior (explicit
   * lookup phrases + weather only). "broad" adds instant deterministic tools
   * (calculator, unit conversion, clock/date), wh-question lookup routing, and
   * weather-query cleanup — the GPT-Live-style "delegate what the small model
   * can't answer" posture. Small-talk protection (stoplist + validQuery) is
   * unchanged in both modes. Shipped "broad": QA accuracy 42% → 83% on MAP,
   * 88% on holdout, zero small-talk false triggers in both splits.
   */
  toolRouting: "broad" as "conservative" | "broad",
};

export type Tunables = typeof TUNABLES;

/** One completed turn's stage timing breakdown (all ms, absolute perf.now). */
export type TurnRecord = {
  /** performance.now() when trailing silence began (end of user speech). */
  endOfSpeech: number;
  /** Endpoint decision fired. endpointWait = fired - endOfSpeech. */
  fired: number;
  /** Transcript ready (bestText or finalize done). */
  transcriptReady: number;
  /** Whether the fast bestText path was used (vs a full finalize pass). */
  usedBestText: boolean;
  /** Which endpoint rule fired the turn (campaign-1 diagnosis). */
  endCause?: "punct" | "silence" | "max";
  /** First LLM text delta arrived. */
  firstDelta: number;
  /** First sentence/clause handed to TTS. */
  firstSentence: number;
  /** First TTS audio chunk scheduled. Always the first SYNTHESIZED reply
   *  chunk — the onset filler (below) never counts, so this stat keeps its
   *  meaning across onsetFiller on/off runs. */
  firstAudio: number;
  /** Pre-rendered onset filler chunk scheduled (absent = no filler played).
   *  Kept separate from firstAudio deliberately: cycle 3's single first-sound
   *  metric rewarded ANY sound, including one that stepped on the real reply.
   *  The bench computes onset = onsetAudio - endOfSpeech. */
  onsetAudio?: number;
  transcript: string;
  reply: string;
  interrupted: boolean;
};

/** Rolling log of completed turns, for the bench (exposed on window in DEV). */
export const TURN_LOG: TurnRecord[] = [];
