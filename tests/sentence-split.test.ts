// Unit tests for the pure TTS chunker. Run with:
//   node --experimental-strip-types --test tests/sentence-split.test.ts
// (or `npx tsx --test tests/sentence-split.test.ts`).

import assert from "node:assert/strict";
import { test } from "node:test";

import { splitSpeechChunks, type SplitSpeechChunksOptions } from "../src/sentence-split.ts";

/** Mock the LLM delta stream: yield `text` split into `chunkSize`-char deltas. */
async function* mockStream(text: string, chunkSize = text.length): AsyncGenerator<string> {
  for (let i = 0; i < text.length; i += chunkSize) {
    yield text.slice(i, i + chunkSize);
  }
}

async function collect(
  text: string,
  opts: SplitSpeechChunksOptions,
  chunkSize?: number,
): Promise<string[]> {
  const out: string[] = [];
  for await (const chunk of splitSpeechChunks(mockStream(text, chunkSize), opts)) {
    out.push(chunk);
  }
  return out;
}

const BASE: SplitSpeechChunksOptions = {
  firstClauseMinChars: 18,
  streamFlushClauses: false,
};

const SENTENCE = "I checked the calendar for you, and the meeting is at noon, so please be ready.";

test("streamFlushClauses=false: only first clause flushes early; rest waits for sentence end", async () => {
  const chunks = await collect(SENTENCE, { ...BASE, streamFlushClauses: false });
  assert.deepEqual(chunks, [
    "I checked the calendar for you,",
    "and the meeting is at noon, so please be ready.",
  ]);
});

test("streamFlushClauses=true: subsequent clause boundaries flush too", async () => {
  const chunks = await collect(SENTENCE, { ...BASE, streamFlushClauses: true });
  assert.deepEqual(chunks, [
    "I checked the calendar for you,",
    "and the meeting is at noon,",
    "so please be ready.",
  ]);
});

test("streamFlushClauses=true is delta-granularity independent (char-by-char == whole)", async () => {
  const whole = await collect(SENTENCE, { ...BASE, streamFlushClauses: true });
  const charByChar = await collect(SENTENCE, { ...BASE, streamFlushClauses: true }, 1);
  assert.deepEqual(charByChar, whole);
});

test("false mode is also delta-granularity independent", async () => {
  const whole = await collect(SENTENCE, { ...BASE, streamFlushClauses: false });
  const charByChar = await collect(SENTENCE, { ...BASE, streamFlushClauses: false }, 1);
  assert.deepEqual(charByChar, whole);
});

test("full sentence stream flushes on sentence ends and tail", async () => {
  // The first flush uses the early-clause rule (>=18 chars), so the short
  // leading "Hello there." rides along into the first chunk; the rest splits on
  // sentence ends, with the unterminated remainder emitted as the tail.
  const text = "Hello there. How are you today? Now I am fine and ready.";
  const chunks = await collect(text, { ...BASE, streamFlushClauses: false });
  assert.deepEqual(chunks, [
    "Hello there. How are you today?",
    "Now I am fine and ready.",
  ]);
});

test("hard cap flushes buffered text with no punctuation", async () => {
  const chunks = await collect("abcdefghijklmnop", {
    firstClauseMinChars: 1000,
    streamFlushClauses: false,
    hardCapChars: 10,
  });
  assert.deepEqual(chunks, ["abcdefghijklmnop"]);
});

test("firstWordFlushChars lowers the no-punctuation first flush (cycle 23)", async () => {
  // "The weather in Tokyo is" (23 chars, no punctuation): with the default 2×
  // fallback (36) nothing flushes until more text arrives; with
  // firstWordFlushChars 20 the first words flush at the last word boundary.
  const text = "The weather in Tokyo is mild today with light wind.";
  const eager = await collect(text, {
    firstClauseMinChars: 18,
    firstWordFlushChars: 20,
    streamFlushClauses: false,
  }, 24);
  assert.equal(eager[0], "The weather in Tokyo is");
  assert.equal(eager.join(" ").replace(/\s+/g, " "), text);

  // Default behavior unchanged when the option is omitted.
  const classic = await collect(text, {
    firstClauseMinChars: 18,
    streamFlushClauses: false,
  }, 24);
  assert.equal(classic[0], "The weather in Tokyo is mild today with light");
});

test("flushTail predicate suppresses the trailing tail", async () => {
  const kept = await collect("hello world", {
    firstClauseMinChars: 1000,
    streamFlushClauses: false,
  });
  assert.deepEqual(kept, ["hello world"]);

  const dropped = await collect("hello world", {
    firstClauseMinChars: 1000,
    streamFlushClauses: false,
    flushTail: () => false,
  });
  assert.deepEqual(dropped, []);
});
