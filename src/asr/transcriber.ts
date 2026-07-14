// The minimal streaming-transcriber contract the duplex engine consumes —
// exactly the surface duplex.ts touches on StreamingTranscriber, extracted so
// the Whisper LocalAgreement lane (streaming.ts) and the Kyutai delayed-streams
// lane (kyutai-stream.ts) are interchangeable behind TUNABLES.asrEngine.
// Kept in its own module (rather than in streaming.ts) so kyutai-stream.ts can
// import the type without pulling in the Whisper machinery, and vice versa.

export type StreamingUpdate = {
  committed: string;
  tentative: string;
  lastChangeAt: number;
};

export interface Transcriber {
  /**
   * Stable transcript of the current utterance. For the Whisper lane this is
   * the LocalAgreement-2 committed prefix; for the Kyutai lane the model
   * streams monotonically (tokens are never revised), so EVERYTHING decoded so
   * far is committed. The duplex tick reads this for punctuation endpointing
   * and the caption/barge paths read it via onUpdate.
   */
  readonly committed: string;
  /** Unstable hypothesis tail. Always "" for the Kyutai lane (no re-decoded
   *  hypothesis exists — see `committed`). */
  readonly tentative: string;
  /** Best transcript available right now WITHOUT extra model work (the
   *  turn-end fast path — see duplex.endUserTurn). */
  bestText(): string;

  /**
   * Optional: wait for the committed stream to SETTLE before reading
   * bestText(). Only meaningful for delayed-stream engines (kyutai): their
   * text lags the audio by ~0.5 s, so an endpoint that fires the moment the
   * user finishes (semantic VAD) lands before the last words have flushed —
   * settle() polls until the text stops growing (or maxMs), closing the
   * truncation gap the mini bench caught ("...hobby to do on the").
   */
  settle?(maxMs: number): Promise<void>;
  /**
   * OPTIONAL semantic-VAD hook (Kyutai lane only): latest frame's model-
   * predicted probability that the user is done talking, or null when
   * unavailable (Whisper lane doesn't implement it; the Kyutai lane returns
   * null until vad-capable weights are loaded and ≥2 frames of the current
   * utterance have decoded). The duplex tick uses it — when the
   * kyutaiVadEndpoint tunable is on — INSTEAD of the punct/silence timers.
   */
  pauseProb?(): number | null;
  /**
   * Turn-end slow path: produce the best final transcript, doing extra model
   * work if needed (Whisper: one full-window pass; Kyutai: drain the frame
   * backlog + flush the trained-in 0.5 s text delay with synthetic silence).
   */
  finalize(): Promise<string>;
  /** Clear per-utterance state for a fresh utterance. Callers always pair this
   *  with capture.clear() (see duplex.startFreshListening), so implementations
   *  may assume the sample buffer restarts at zero. */
  reset(): void;
  /** Start the background streaming loop (idempotent). */
  start(): void;
  /** Stop the loop and release per-utterance resources. */
  stop(): Promise<void>;
}
