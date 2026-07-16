import { numpy as np } from "@jax-js/jax";

import { TUNABLES } from "../tunables";
import type { WhisperConfig } from "./model";
import { WhisperTokenizer } from "./tokenizer";

export type TranscriptSegment = {
  text: string;
  start: number;
  end: number;
  row: number;
};

type Gate = {
  tokens: readonly number[];
  duration: number;
  forceTimestamp: boolean;
};

type DecoderSample = {
  token: number;
  /** Selected-token log probability normalized over the top two candidates. */
  logProb: number;
};

/** Greedy sample plus its top-two normalized decoder log probability. */
export async function sampleGreedyWithScore(
  logits: np.Array,
  tokens: readonly number[],
  duration: number,
  config: WhisperConfig,
): Promise<DecoderSample> {
  // TUNABLES.asrSampler exists for the bench harness, but only the "js" path
  // preserves the gate semantics exactly. The gating needs two full-vocab
  // reductions (a logSumExp over the admissible timestamp range vs a masked max
  // over the admissible text range) whose comparison decides `forceTimestamp`,
  // and then a masked argmax over a set that spans both ranges. A device-side
  // topK candidate list can miss the masked argmax winner when the largest raw
  // logits are all suppressed tokens, and a device reduction is not guaranteed
  // bit-identical to the sequential JS reduction — either would change the
  // sampled token. So we never approximate the gate: "gpu" falls through to the
  // exact JS implementation below.
  void TUNABLES.asrSampler;
  const values = (await logits.data()) as ArrayLike<number>;
  const gate = { tokens, duration, forceTimestamp: false };
  const timestamps = timestampCandidates(values, gate, config);
  const text = textCandidates(values, gate, config);
  gate.forceTimestamp = timestamps.logSumExp > text.bestValue;

  // If timestamp mass beats the best text logit, the gate admits timestamps
  // only. Otherwise no individual timestamp can beat the text winner because
  // each is <= their logSumExp. Reusing those reduction winners avoids the old
  // redundant full-vocabulary argmax scan while preserving its exact choice.
  const selected = gate.forceTimestamp ? timestamps : text;
  const otherRunner = gate.forceTimestamp
    ? -Infinity
    : timestamps.bestValue;
  const secondValue = Math.max(selected.secondValue, otherRunner);
  // Top-two normalization is monotonic with the decoder margin and uses only
  // actual admissible-token logits. Full softmax regressed ASR latency >5%.
  const logProb = secondValue === -Infinity
    ? 0
    : -Math.log1p(Math.exp(secondValue - selected.bestValue));
  return { token: selected.bestToken, logProb };
}

export function decodeTranscriptTokens(
  tokens: readonly number[],
  tokenizer: WhisperTokenizer,
  config: WhisperConfig,
): string {
  return tokenizer.decode(
    tokens.filter((token) => !isTimestampToken(token, config)),
  );
}

export function buildTimestampSegments(
  tokens: readonly number[],
  tokenizer: WhisperTokenizer,
  totalDuration: number,
  config: WhisperConfig,
): TranscriptSegment[] {
  const groups: Omit<TranscriptSegment, "row">[] = [];
  let start: number | null = null;
  let textTokens: number[] = [];

  for (const token of tokens) {
    if (isTimestampToken(token, config)) {
      flushSegment(
        groups,
        tokenizer,
        textTokens,
        start,
        timestampSeconds(token, totalDuration, config),
        config,
      );
      textTokens = [];
      start = timestampSeconds(token, totalDuration, config);
    } else if (token < config.eosToken) {
      textTokens.push(token);
    }
  }
  flushSegment(
    groups,
    tokenizer,
    textTokens,
    start ?? 0,
    Math.min(totalDuration, 30),
    config,
  );

  return groups.map((group) => ({ ...group, row: 0 }));
}

function canSample(token: number, gate: Gate, config: WhisperConfig): boolean {
  if (gate.forceTimestamp && !isTimestampToken(token, config)) return false;
  if (isTimestampToken(token, config)) return canTimestamp(token, gate, config);
  if (
    config.suppressTokens.includes(token) ||
    token === config.noTimestampsToken
  )
    return false;
  if (token > config.eosToken && token < config.timestampBeginToken)
    return false;
  if (gate.tokens.length === 0 && config.beginSuppressTokens.includes(token))
    return false;
  if (timestampPhase(gate.tokens, config) === "timestamp-or-eos") {
    return token === config.eosToken;
  }
  return token <= config.eosToken;
}

function canTimestamp(
  token: number,
  { tokens, duration }: Gate,
  config: WhisperConfig,
): boolean {
  if (timestampPhase(tokens, config) === "text") return false;
  const last = findLastTimestampToken(tokens, config);
  const maxTimestamp =
    config.timestampBeginToken +
    Math.ceil(
      Math.min(Math.max(duration, 0.02), 30) / config.timestampStepSeconds,
    ) +
    2;
  return (last === null || token >= last) && token <= maxTimestamp;
}

function timestampPhase(tokens: readonly number[], config: WhisperConfig) {
  const last = tokens[tokens.length - 1];
  if (!isTimestampToken(last, config)) return "any";
  return tokens.length < 2 ||
    isTimestampToken(tokens[tokens.length - 2], config)
    ? "text"
    : "timestamp-or-eos";
}

type CandidateSet = {
  bestToken: number;
  bestValue: number;
  secondValue: number;
};

function timestampCandidates(
  values: ArrayLike<number>,
  gate: Gate,
  config: WhisperConfig,
): CandidateSet & { logSumExp: number } {
  let bestToken = config.timestampBeginToken;
  let bestValue = -Infinity;
  let secondValue = -Infinity;
  for (let token = config.timestampBeginToken; token < values.length; token++) {
    if (!canTimestamp(token, gate, config)) continue;
    const value = values[token];
    if (value > bestValue) {
      secondValue = bestValue;
      bestToken = token;
      bestValue = value;
    } else if (value > secondValue) {
      secondValue = value;
    }
  }
  if (bestValue === -Infinity) {
    return { bestToken, bestValue, secondValue, logSumExp: -Infinity };
  }
  let sum = 0;
  for (let token = config.timestampBeginToken; token < values.length; token++) {
    if (canTimestamp(token, gate, config)) {
      sum += Math.exp(values[token] - bestValue);
    }
  }
  return {
    bestToken,
    bestValue,
    secondValue,
    logSumExp: bestValue + Math.log(sum),
  };
}

function textCandidates(
  values: ArrayLike<number>,
  gate: Gate,
  config: WhisperConfig,
): CandidateSet {
  let bestToken = config.eosToken;
  let bestValue = -Infinity;
  let secondValue = -Infinity;
  for (let token = 0; token < config.timestampBeginToken; token++) {
    if (!canSample(token, gate, config)) continue;
    const value = values[token];
    if (value > bestValue) {
      secondValue = bestValue;
      bestToken = token;
      bestValue = value;
    } else if (value > secondValue) {
      secondValue = value;
    }
  }
  return { bestToken, bestValue, secondValue };
}

function isTimestampToken(
  token: number | undefined,
  config: WhisperConfig,
): boolean {
  return (
    token !== undefined &&
    token >= config.timestampBeginToken &&
    token < config.vocabSize
  );
}

function findLastTimestampToken(
  tokens: readonly number[],
  config: WhisperConfig,
): number | null {
  for (let i = tokens.length - 1; i >= 0; i--) {
    const token = tokens[i];
    if (isTimestampToken(token, config)) return token;
  }
  return null;
}

function timestampSeconds(
  token: number,
  totalDuration: number,
  config: WhisperConfig,
): number {
  const seconds =
    (token - config.timestampBeginToken) * config.timestampStepSeconds;
  return Math.max(0, Math.min(Math.min(totalDuration, 30), seconds));
}

function flushSegment(
  groups: Omit<TranscriptSegment, "row">[],
  tokenizer: WhisperTokenizer,
  textTokens: number[],
  start: number | null,
  end: number,
  config: WhisperConfig,
) {
  const text = tokenizer.decode(textTokens).trim();
  if (!text) return;
  const from = start ?? 0;
  groups.push({
    text,
    start: from,
    end: Math.max(from + config.timestampStepSeconds, end),
  });
}
