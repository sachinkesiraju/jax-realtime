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
  /**
   * Cap on generated tokens per reply (keeps spoken replies short). 96 -> 64
   * in cycle 8: the quality bench's shortSpoken axis kept failing on 60+
   * word open-ended answers ("low quality convo" reports — a voice reply
   * should be a breath or two, and long tails also delay the next turn).
   * 64 tokens ≈ 45 spoken words; replies usually hit their stop token well
   * before it, and sentenceStream only speaks complete sentences either way.
   */
  llmMaxNewTokens: 64,
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
   * benchPrefillEquivalence(250, 64). Raised 64 -> 256 in the cycle-8
   * delay fix: 64-token buckets still churned a new prefill shape every
   * couple of turns as history grew (each first encounter re-traces 32
   * layer jits, ~0.5-1 s on that turn); 256 gives at most ~5 shapes per
   * session and warmup() pre-traces the first three, so conversations run
   * re-trace-free. The pad FLOPs are trivial next to the dispatch overhead.
   */
  llmPrefillBucket: 256,
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
  // NOTE (cycle 7): a ttsPrefillBucket lever (pad the flow-LM prefill text to
  // 16-token multiples, llmPrefillBucket's twin) lived here and was REJECTED
  // at MAP: the padded prompt made warm prefills SLOWER (~150 ms vs ~30-60 ms)
  // and the turn bench's tts stage regressed 115 -> 221 ms median. The step-0
  // re-trace variance it targeted is real (benchTtsPrefill shows it); the
  // open lever is fusing the step-0 prefill, not padding it.

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
   * before it defaults on. Flipped ON in cycle 8 ("voice responses don't
   * feel realtime"): first sound lands ~0.5-0.8 s after end of speech with
   * real-reply latency unchanged. The ears gate is now live-in-production —
   * if the filler-then-pause cadence sounds worse than silence, flip this
   * back off; the bench numbers will not miss it.
   */
  onsetFiller: true,

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

  // region: quality (campaign 3 — conversation quality)
  //
  // Three INDEPENDENTLY-gated candidates targeting observed live failures of
  // the SmolLM2-360M brain. Each defaults to the SHIPPED behavior so the
  // external eval (bench/quality.mjs) can A/B every knob in isolation. The
  // eval scores: asksClarify on garbled input, noFalseClarify on clean input,
  // notVerbatim on repeat requests, shortSpoken on open-ended prompts, and
  // noMarkdown/noPlaceholder on everything.
  /**
   * Token-level hard ban on formatting junk in SmolLM sampling. Observed live
   * failure: replies containing markdown/template artifacts like "I'm really
   * into [activity]s" or "**great**" — a VOICE assistant never needs [ ] * #
   * or backtick in its output, so any token whose decoded text contains one
   * of those characters gets its logit set to -Infinity before the
   * temperature/top-p draw (see SmolLmChatModel.maskFormatTokens). The ban
   * set is built once, lazily, by scanning the tokenizer's full id→bytes
   * decoder map (BPE tokens are multi-char, so encoding "[" alone would miss
   * merged tokens like " [" or "]("). The system-prompt instruction alone
   * ("no lists ... or markdown") demonstrably does not stop a 360M model;
   * this makes the junk unsampleable instead of merely discouraged. Safe for
   * the "[scene: …]" tag because that appears only in PROMPT text (user
   * content), never in generated output. SHIPPED ON (cycle 7): zero
   * regressions across every eval axis on MAP + holdout, and it converts the
   * prompt's "no markdown" request from a suggestion into a guarantee (239 of
   * 49k vocab ids banned; none contain a letter, so no word is affected).
   */
  qualityBanFormatTokens: true,
  /**
   * Append one clarify-on-garble sentence to the SmolLM system prompt.
   * Observed live failure: ASR-garbled input ("whazzit fmm the uh...") gets a
   * confident confabulated answer instead of a clarifying question. SmolLM2
   * honors a real ChatML system role, but the clause ALONE was inert at MAP
   * (0/6 asks-to-clarify — a 360M model doesn't follow rules), so the flag
   * also injects ONE few-shot exemplar exchange demonstrating the behavior
   * (SMOLLM_GARBLE_EXEMPLAR; the conversation-quality diagnosis's Tier-2
   * design — open roadmap in docs/BENCHMARKS.md). Read at
   * generation time (encodePrompt rebuilds the system turn every call) so the
   * bench can flip it without a reload. SHIPPED ON (cycle 7): clause+exemplar
   * took asksClarify 0/6 → 5/6 on MAP and 0/4 → 3/4 on holdout with zero
   * false clarifies on clean input and other axes flat. Cost: ~35 extra
   * prompt tokens per turn (bucket-padded prefill absorbs it).
   */
  qualityGarbleClause: true,
  /**
   * SmolLM sampling temperature (was hardcoded 0.7 in generateStream).
   * Observed live failure: occasional rambling / off-prompt drift, which
   * lower temperature plausibly tames on a 360M model — but too low risks
   * verbatim repeats (the repetition penalty only covers the previous reply)
   * and duller open-ended answers. 0.5 was A/B'd in cycle 7: it won brevity
   * on MAP (shortSpoken 2/6 → 4/6) but FUSED with the garble exemplar it
   * reversed on holdout (asksClarify 3/4 → 1/4, factual misses) — cooler
   * sampling fights the few-shot behavior. Stays 0.7 (the garble win is
   * worth more than the brevity win); recorded so 0.5 isn't retried blind.
   */
  qualityTemperature: 0.7,
  qualityVlmTemperature: 0.5,
  qualityTypedMemory: true,
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
