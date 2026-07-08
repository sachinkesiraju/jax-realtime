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
   * Reuse the KV cache across conversation turns. When on, a new turn whose
   * tokenized prompt shares a prefix with the cached sequence only prefills the
   * differing suffix instead of the whole prompt. Default false rebuilds the
   * cache every turn (shipped behavior). Prefix comparison is on token IDs, so
   * any divergence (e.g. retokenized history) safely falls back to a rebuild.
   */
  llmKvReuse: false,
  /**
   * Cap on the number of chat messages kept when formatting the LLM prompt
   * (whole user/assistant pairs). 0 = unlimited (shipped behavior).
   */
  llmMaxHistoryTurns: 0,

  // region: tts-split
  /** Min chars before the first clause is flushed to TTS early. */
  firstClauseMinChars: 18,
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
