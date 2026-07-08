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
   * Fuse the whole Gemma decode step (embedding → 18 layers → norm → LM head)
   * into a single jitted dispatch instead of ~21 separate jit calls. The fused
   * path is numerically identical (same math, same order; greedy-equivalence
   * verified) with far fewer command-buffer submits per token — shipped on
   * after the cycle-4 bench (46.9 → 34.2 ms/token combined with "topk").
   */
  llmFusedStep: true,
  /**
   * Decode-time sampler. "js" reads the full 262k-vocab logits back and scans
   * them on the CPU (shipped). "topk" runs `lax.topK(logits, 64)` on the GPU
   * and reads back only the 64 candidates, then finishes top-p/temperature
   * sampling in JS over those 64 — bit-identical selection to "js" (both pick
   * the same top-64 set), just a ~1 MB smaller readback per token. Shipped on
   * (cycle-4 bench, greedy-equivalence verified).
   */
  llmSampler: "topk" as "js" | "topk",
  /**
   * Fold `lax.topK(logits, 64)` INTO the fused Gemma decode jit (only takes
   * effect when llmFusedStep is on). The fused step then emits the 64 top-k
   * (value, index) pairs packed into one fp32 readback array instead of the
   * full 262k-vocab logits — the topK reduction rides the same dispatch as the
   * decode step (saving one dispatch + sync per token) and the readback is one
   * `.data()` of 128 floats instead of two. Selection is identical to the
   * llmSampler:"topk" path (same top-64 set, same values). Shipped on:
   * greedy-equivalence verified across all five configs in-browser; paired
   * speed runs were consistently faster (~42.7→38.0, ~45.1→44.0 ms/token —
   * small, but direction-consistent and zero-risk by construction).
   */
  llmTopkInFused: true,
  /**
   * Reuse the KV cache across conversation turns. When on, a new turn whose
   * tokenized prompt shares a prefix with the cached sequence only prefills the
   * differing suffix instead of the whole prompt. Default false rebuilds the
   * cache every turn (shipped behavior). Prefix comparison is on token IDs, so
   * any divergence (e.g. retokenized history) safely falls back to a rebuild.
   */
  llmKvReuse: false,
  /**
   * Cap on the number of chat messages kept when formatting the LLM prompt
   * (whole user/assistant pairs). 0 = unlimited. Shipped at 16 (8 exchanges):
   * unbounded history let Gemma's prefill grow every turn until a long session
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
   * possible, mirroring the Gemma `llmFusedStep` lever. When on:
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
   * (step 0) still uses the unfused path, matching Gemma's fuse-decode-only
   * split. A/B with `window.__pipeline().tts.benchSynth(sentence, {fused})`.
   * Shipped on: bench (fixed seed) showed 1291→1010 ms gen (~22%), realtime
   * factor 0.40→0.31, with identical frame count both paths (same EOS decision,
   * same audio duration) — equivalent output, fewer dispatches.
   */
  ttsFusedStep: true,

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
  /** First TTS audio chunk scheduled. */
  firstAudio: number;
  transcript: string;
  reply: string;
  interrupted: boolean;
};

/** Rolling log of completed turns, for the bench (exposed on window in DEV). */
export const TURN_LOG: TurnRecord[] = [];
