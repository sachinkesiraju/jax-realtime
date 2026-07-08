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

  // region: llm
  /** Cap on generated tokens per reply (keeps spoken replies short). */
  llmMaxNewTokens: 96,

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
