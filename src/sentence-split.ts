// Pure TTS text chunker. Splits an LLM token/delta stream into the strings
// handed to speech synthesis. Kept dependency-free (no WebGPU/DOM/model
// imports) so it can be unit-tested in plain Node.

/** Sentence-ending punctuation followed by whitespace; skips decimals. */
export function findSentenceEnd(buffer: string): number {
  for (let i = 0; i < buffer.length - 1; i++) {
    const c = buffer[i];
    if ((c === "." || c === "!" || c === "?" || c === "…") && /\s/.test(buffer[i + 1])) {
      return i + 1;
    }
  }
  return -1;
}

/**
 * Index just after a clause break (comma/colon/semicolon or sentence end)
 * followed by whitespace, but only once ≥`min` chars have accumulated so the
 * spoken fragment isn't a choppy one-word stub. Used to minimize
 * time-to-first-audio on longer opening sentences (and, when clause streaming
 * is enabled, on subsequent clauses too).
 */
export function findClauseEnd(buffer: string, min: number): number {
  for (let i = 0; i < buffer.length - 1; i++) {
    const c = buffer[i];
    const isBreak =
      c === "," || c === ";" || c === ":" || c === "." || c === "!" || c === "?" || c === "…";
    if (isBreak && i + 1 >= min && /\s/.test(buffer[i + 1])) {
      return i + 1;
    }
  }
  return -1;
}

export interface SplitSpeechChunksOptions {
  /** Min chars before the first (and, when streaming clauses, each) clause flushes. */
  firstClauseMinChars: number;
  /**
   * When true, flush on clause boundaries (comma/colon/semicolon) after the
   * first clause too, not just on sentence ends. When false, only the first
   * clause is flushed early; the rest waits for sentence-end punctuation or the
   * hard cap.
   */
  streamFlushClauses: boolean;
  /** Hard cap: flush whatever is buffered once it reaches this many chars. */
  hardCapChars?: number;
  /** Predicate gating the final tail flush (e.g. skip when the turn aborted). */
  flushTail?: () => boolean;
}

/**
 * Pure TTS chunker: consumes an async stream of LLM text deltas and yields the
 * strings that should be handed to speech synthesis. Splitting rules:
 *   - flush the FIRST clause early (comma/colon/semicolon, or a word boundary
 *     once ~2× the clause minimum has accumulated with no punctuation);
 *   - flush on every sentence end (terminal punctuation + whitespace);
 *   - when `streamFlushClauses`, additionally flush on subsequent clause
 *     boundaries after the first clause;
 *   - flush whatever is buffered once it reaches the hard cap;
 *   - flush the trailing tail at end of stream unless `flushTail` says not to.
 */
export async function* splitSpeechChunks(
  deltas: AsyncIterable<string>,
  opts: SplitSpeechChunksOptions,
): AsyncGenerator<string, void, void> {
  const hardCap = opts.hardCapChars ?? 120;
  let buffer = "";
  let firstEmitted = false;

  for await (const delta of deltas) {
    buffer += delta;

    // Fastest first audio: flush the first clause as soon as a comma/colon/
    // semicolon appears (once there's enough to sound natural), so speech
    // starts after "The weather in Tokyo," instead of the whole sentence.
    // If no punctuation shows up, flush at a WORD BOUNDARY once ~2× the
    // clause minimum has accumulated — otherwise the reply text is fully
    // written on screen while the voice still waits for the first sentence
    // to complete before it can even start synthesizing.
    if (!firstEmitted) {
      let clauseIdx = findClauseEnd(buffer, opts.firstClauseMinChars);
      if (clauseIdx === -1 && buffer.length >= opts.firstClauseMinChars * 2) {
        const lastSpace = buffer.lastIndexOf(" ");
        if (lastSpace >= opts.firstClauseMinChars) clauseIdx = lastSpace + 1;
      }
      if (clauseIdx !== -1) {
        const clause = buffer.slice(0, clauseIdx).trim();
        buffer = buffer.slice(clauseIdx);
        if (clause) {
          firstEmitted = true;
          yield clause;
        }
      }
    }

    let idx: number;
    while ((idx = findSentenceEnd(buffer)) !== -1) {
      const sentence = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx);
      if (sentence) {
        firstEmitted = true;
        yield sentence;
      }
    }

    // When enabled, keep flushing on clause boundaries (comma/colon/semicolon)
    // after the first clause too, so TTS starts on smaller mid-utterance
    // chunks rather than waiting for full sentence-end punctuation.
    if (opts.streamFlushClauses && firstEmitted) {
      let clauseIdx: number;
      while ((clauseIdx = findClauseEnd(buffer, opts.firstClauseMinChars)) !== -1) {
        const clause = buffer.slice(0, clauseIdx).trim();
        buffer = buffer.slice(clauseIdx);
        if (clause) yield clause;
      }
    }

    if (buffer.length >= hardCap) {
      const sentence = buffer.trim();
      buffer = "";
      if (sentence) yield sentence;
    }
  }

  if (!opts.flushTail || opts.flushTail()) {
    const tail = buffer.trim();
    if (tail) yield tail;
  }
}
