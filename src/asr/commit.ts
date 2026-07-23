// Pure commit-policy logic for the streaming transcriber, split out so it can
// be unit-tested directly (no async pass loop, no timers). Given the previous
// and current hypotheses plus this pass's decoder confidence, it decides which
// words are committed and which stay tentative.

export type CommitPolicy = {
  /** Commit the whole hypothesis on one high-confidence pass. */
  fastCommit: boolean;
  /** avgLogProb floor for fastCommit. */
  fastCommitThreshold: number;
};

export type CommitDecision = {
  committedWords: string[];
  tentative: string;
};

/** Length of the longest word-prefix shared by `a` and `b`, comparing words
 *  case- and punctuation-insensitively. */
export function commonPrefixLen(a: string[], b: string[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (
    i < n &&
    a[i].toLowerCase().replace(/[^\p{L}\p{N}']/gu, "") ===
      b[i].toLowerCase().replace(/[^\p{L}\p{N}']/gu, "")
  ) {
    i++;
  }
  return i;
}

/**
 * Decide the committed/tentative split for a streaming pass.
 *
 * Fast-commit path: when enabled AND this pass's confidence
 * (`avgLogProb`) is at or above the threshold, the entire current
 * hypothesis is committed immediately with an empty tentative tail.
 *
 * Otherwise LocalAgreement-2: commit the longest common word-prefix of the
 * previous and current hypotheses; the newest hypothesis's tail is tentative.
 * A null confidence (unknown score) always takes this safe path.
 */
export function decideCommit(
  prevWords: string[],
  words: string[],
  avgLogProb: number | null,
  policy: CommitPolicy,
): CommitDecision {
  if (
    policy.fastCommit &&
    avgLogProb !== null &&
    avgLogProb >= policy.fastCommitThreshold
  ) {
    return { committedWords: words, tentative: "" };
  }
  const commonLen = commonPrefixLen(prevWords, words);
  return {
    committedWords: words.slice(0, commonLen),
    tentative: words.slice(commonLen).join(" "),
  };
}
