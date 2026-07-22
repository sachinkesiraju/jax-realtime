// Standalone test for StreamingTranscriber's commit policy. Run with:
//   npm run test:asr
// It runs through Node's TypeScript transform + a tiny extensionless-import
// resolver hook (scripts/) so no build step or test-runner dependency is
// needed. Kept free of node: builtins so it type-checks under the app's
// DOM-only tsconfig; a thrown assertion rejects the top-level await, which
// makes Node print it and exit non-zero.

import type { ASRResult, SpeechRecognizer } from "../pipeline.ts";
import { StreamingTranscriber, type StreamingOptions } from "./streaming.ts";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assertEqual<T>(actual: T, expected: T, message: string): void {
  if (actual !== expected) {
    throw new Error(
      `${message}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`,
    );
  }
}

/** A SpeechRecognizer stand-in that hands back a fixed queue of ASR results,
 *  then throws (like a busy/transient error) so the loop idles without
 *  ingesting an empty tail that would clobber the state we're asserting on. */
function mockRecognizer(results: ASRResult[]): {
  asr: SpeechRecognizer;
  drained: () => boolean;
} {
  const queue = [...results];
  const asr = {
    isBusy: false,
    async transcribeWithConfidence(): Promise<ASRResult> {
      const next = queue.shift();
      if (!next) throw new Error("drained");
      return next;
    },
  };
  return {
    asr: asr as unknown as SpeechRecognizer,
    drained: () => queue.length === 0,
  };
}

function result(text: string, avgLogProb: number | null): ASRResult {
  return { text, confidence: { avgLogProb } };
}

async function drive(
  results: ASRResult[],
  opts: StreamingOptions,
): Promise<StreamingTranscriber> {
  const { asr, drained } = mockRecognizer(results);
  const transcriber = new StreamingTranscriber(
    asr,
    () => new Float32Array(16_000), // 1 s window, clears minWindow
    () => {},
    () => null, // no assistant TTS => no echo filtering
    opts,
  );
  transcriber.start();
  for (let i = 0; i < 200 && !drained(); i++) await sleep(10);
  await sleep(50); // let the final ingest settle
  await transcriber.stop();
  return transcriber;
}

const FAST: StreamingOptions = {
  minPassIntervalMs: 5,
  fastCommit: true,
  fastCommitThreshold: -0.3,
};

async function main(): Promise<void> {
  // 1. Fast commit fires above threshold: the whole hypothesis commits on ONE
  //    pass, with an empty tentative tail.
  {
    const t = await drive([result("hello world", -0.1)], FAST);
    assertEqual(t.committed, "hello world", "fast-commit should commit all words");
    assertEqual(t.tentative, "", "fast-commit should leave no tentative tail");
  }

  // 2. Below threshold, fast commit does NOT fire: falls back to
  //    LocalAgreement-2, so a lone first pass stays fully tentative.
  {
    const t = await drive([result("hello world", -0.5)], FAST);
    assertEqual(t.committed, "", "below threshold must not fast-commit");
    assertEqual(t.tentative, "hello world", "below threshold stays tentative");
  }

  // 3. LocalAgreement-2 still works on the below-threshold fallback: two
  //    agreeing passes commit their common prefix.
  {
    const t = await drive(
      [result("hello world", -0.5), result("hello world there", -0.5)],
      FAST,
    );
    assertEqual(t.committed, "hello world", "agreed prefix should commit");
    assertEqual(t.tentative, "there", "divergent tail stays tentative");
  }

  // 4. Default path (fastCommit unset -> TUNABLES.asrFastCommit === false):
  //    even a high-confidence pass takes the two-pass LocalAgreement-2 route.
  {
    const t = await drive([result("hello world", -0.1)], {
      minPassIntervalMs: 5,
    });
    assertEqual(t.committed, "", "default keeps LocalAgreement-2 (no fast-commit)");
    assertEqual(t.tentative, "hello world", "default leaves single pass tentative");
  }
}

await main();
console.log("streaming.test.ts: all assertions passed");
