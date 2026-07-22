// Deterministic unit test for the streaming commit policy. Run with:
//   npm run test:asr
// Pure and synchronous — no async pass loop, no timers, no test-runner
// dependency. Executes under `node --experimental-strip-types` because
// commit.ts has only type-only imports (erased at runtime).

import { decideCommit } from "./commit.ts";

let failures = 0;

function check(name: string, cond: boolean, detail?: string): void {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.error(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

function words(text: string): string[] {
  return text.split(" ").filter(Boolean);
}

const FAST = { fastCommit: true, fastCommitThreshold: -0.3 };
const DEFAULT = { fastCommit: false, fastCommitThreshold: -0.3 };

// 1. Fast commit fires at/above threshold: whole hypothesis commits, empty tail.
{
  const d = decideCommit([], words("hello world"), -0.1, FAST);
  check(
    "fast-commit above threshold commits all words with empty tail",
    d.committedWords.join(" ") === "hello world" && d.tentative === "",
    JSON.stringify(d),
  );
}

// 2. Exactly at the threshold still fast-commits (>= boundary).
{
  const d = decideCommit([], words("hello world"), -0.3, FAST);
  check(
    "fast-commit at exact threshold commits",
    d.committedWords.join(" ") === "hello world" && d.tentative === "",
    JSON.stringify(d),
  );
}

// 3. Below threshold does NOT fast-commit: falls back to LocalAgreement-2,
//    so a lone first pass (no prior agreement) stays fully tentative.
{
  const d = decideCommit([], words("hello world"), -0.5, FAST);
  check(
    "below threshold stays tentative (LocalAgreement-2)",
    d.committedWords.length === 0 && d.tentative === "hello world",
    JSON.stringify(d),
  );
}

// 4. Null confidence never fast-commits even when the flag is on.
{
  const d = decideCommit([], words("hello world"), null, FAST);
  check(
    "null confidence takes the safe LocalAgreement-2 path",
    d.committedWords.length === 0 && d.tentative === "hello world",
    JSON.stringify(d),
  );
}

// 5. LocalAgreement-2 commits the agreed common prefix across two passes.
{
  const d = decideCommit(words("hello world"), words("hello world there"), -0.5, FAST);
  check(
    "LocalAgreement-2 commits agreed prefix, keeps divergent tail tentative",
    d.committedWords.join(" ") === "hello world" && d.tentative === "there",
    JSON.stringify(d),
  );
}

// 6. Default policy (fastCommit off) never fast-commits even on high confidence.
{
  const d = decideCommit([], words("hello world"), -0.01, DEFAULT);
  check(
    "default policy keeps LocalAgreement-2 regardless of confidence",
    d.committedWords.length === 0 && d.tentative === "hello world",
    JSON.stringify(d),
  );
}

if (failures > 0) {
  throw new Error(`commit.test.ts: ${failures} assertion(s) failed`);
}
console.log("commit.test.ts: all assertions passed");
