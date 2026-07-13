// Conversation-quality eval: drives the LLM directly with scripted chat
// histories (no audio path) and scores replies with deterministic checks.
// Buckets mirror the observed live failures:
//   garble    — ASR-garbled input: should ask to clarify, not confabulate
//   clean     — normal input: must NOT ask to clarify (false-clarify guard)
//   repeat    — same question twice: second reply must not echo the first
//   brevity   — open-ended asks: spoken style, <= 2 sentences, no lists
//   format    — never emit markdown/brackets/placeholder junk ("[activity]")
//   factual   — simple factual one-liners must still be answered (no regression)
//
//   node bench/quality.mjs --label baseline --tunables '{}' [--runs 2]
//
// Writes bench/results/quality-<label>-<ts>.json and prints per-bucket rates.
// Each item is scored on EVERY run (sampling is stochastic; more runs = less
// noise). The same seed items are used for every condition (paired eval).

import { launch } from "puppeteer-core";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, all) =>
    a.startsWith("--") ? [a.slice(2), all[i + 1]] : null,
  ).filter(Boolean),
);
const label = args.label ?? "quality";
const tunables = JSON.parse(args.tunables ?? "{}");
const runs = Number(args.runs ?? 2);

const u = (content) => ({ role: "user", content, t: 0 });
const a = (content) => ({ role: "assistant", content, t: 0 });

// MAP set. Holdout lives in quality-holdout list below (--holdout flag).
const MAP_ITEMS = [
  // garble: real ASR garbles observed in this project's logs + synthetic
  { id: "g1", bucket: "garble", history: [u("to Now I actually Actually the answering Cisco")] },
  { id: "g2", bucket: "garble", history: [u("All in all. All in all the weather brick soon")] },
  { id: "g3", bucket: "garble", history: [u("can you the with about it lamp for Tuesday")] },
  // clean: must not trigger clarify
  { id: "c1", bucket: "clean", history: [u("How are you doing today?")] },
  { id: "c2", bucket: "clean", history: [u("Tell me a fun fact about octopuses.")] },
  { id: "c3", bucket: "clean", history: [u("What should I make for dinner tonight?")] },
  // repeat: second answer must differ
  { id: "r1", bucket: "repeat", history: [u("Tell me a joke."), a("Why don't scientists trust atoms? Because they make up everything!"), u("Tell me another joke.")] },
  { id: "r2", bucket: "repeat", history: [u("Give me a productivity tip."), a("Try time-boxing: give each task a fixed window and stop when it ends."), u("Give me a different one.")] },
  // brevity/format: open-ended asks that used to ramble or emit lists
  { id: "b1", bucket: "brevity", history: [u("What are some good weekend hobbies?")] },
  { id: "b2", bucket: "brevity", history: [u("How do I stay focused while working?")] },
  { id: "b3", bucket: "brevity", history: [u("Describe your ideal morning routine.")] },
  // factual guard (things the tool router leaves to the LLM)
  { id: "f1", bucket: "factual", history: [u("What color is the sky on a clear day?")], expect: /blue/i },
  { id: "f2", bucket: "factual", history: [u("How many days are in a week?")], expect: /seven|7/i },
  // scene grounding must keep working
  { id: "s1", bucket: "factual", history: [u("[scene: a person and a couch] What am I sitting on?")], expect: /couch|sofa/i },
];

const HOLDOUT_ITEMS = [
  { id: "hg1", bucket: "garble", history: [u("the it about when purple engine forgot to")] },
  { id: "hg2", bucket: "garble", history: [u("I was thinking maybe the the it's fine tomorrow gravel")] },
  { id: "hc1", bucket: "clean", history: [u("What's your favorite season and why?")] },
  { id: "hc2", bucket: "clean", history: [u("Any tips for learning to cook?")] },
  { id: "hr1", bucket: "repeat", history: [u("Say something encouraging."), a("You're doing better than you think — keep going, one step at a time."), u("Say something else encouraging.")] },
  { id: "hb1", bucket: "brevity", history: [u("What makes a good friendship?")] },
  { id: "hb2", bucket: "brevity", history: [u("Tell me about the ocean.")] },
  { id: "hf1", bucket: "factual", history: [u("How many legs does a spider have?")], expect: /eight|8/i },
  { id: "hs1", bucket: "factual", history: [u("[scene: two people and a laptop] How many people can you see?")], expect: /two|2/i },
];

const ITEMS = args.holdout != null ? HOLDOUT_ITEMS : MAP_ITEMS;

// --- Deterministic checks ---------------------------------------------------
const CLARIFY_RE =
  /\b(didn'?t (quite )?(catch|get|hear)|say (that|it) again|could you repeat|not sure (i|I) (understood|heard)|rephrase|come again|missed that)\b/i;
const sentences = (t) => (t.match(/[^.!?]+[.!?]+/g) ?? [t]).length;
const words = (t) => t.split(/\s+/).filter(Boolean);

function score(item, reply) {
  const r = reply.trim();
  const checks = {};
  // format checks apply to every bucket
  checks.noMarkdown = !/[*#`_]|^\s*[-•]\s/m.test(r);
  checks.noPlaceholder = !/\[[^\]]*\]/.test(r);
  checks.nonEmpty = words(r).length >= 2;
  if (item.bucket === "garble") checks.asksClarify = CLARIFY_RE.test(r);
  if (item.bucket === "clean") checks.noFalseClarify = !CLARIFY_RE.test(r);
  if (item.bucket === "repeat") {
    const prev = item.history.filter((m) => m.role === "assistant").at(-1).content;
    const pw = new Set(words(prev.toLowerCase()));
    const rw = words(r.toLowerCase());
    const overlap = rw.filter((w) => pw.has(w)).length / Math.max(1, rw.length);
    checks.notVerbatim = overlap < 0.6;
  }
  if (item.bucket === "brevity") checks.shortSpoken = sentences(r) <= 3 && words(r).length <= 60;
  if (item.expect) checks.correct = item.expect.test(r);
  return checks;
}

// --- Drive the page ----------------------------------------------------------
const browser = await launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: false,
  protocolTimeout: 0,
  userDataDir: resolve(ROOT, "bench/.profile"),
  args: ["--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--mute-audio"],
});
try {
  const page = (await browser.pages())[0] ?? (await browser.newPage());
  await page.goto("http://localhost:5173", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#load-btn:not([disabled])", { timeout: 30_000 });
  await page.click("#load-btn");
  await page.waitForSelector("#orb-btn:not([disabled])", { timeout: 30 * 60_000 });
  await page.evaluate((over) => Object.assign(window.__tunables, over), tunables);

  const rows = [];
  for (let run = 0; run < runs; run++) {
    for (const item of ITEMS) {
      const reply = await page.evaluate(async (history) => {
        const { text } = await window.__pipeline().llm.generate(history, () => {});
        return text;
      }, item.history);
      const checks = score(item, reply);
      rows.push({ run, id: item.id, bucket: item.bucket, reply, checks });
      const failed = Object.entries(checks).filter(([, v]) => !v).map(([k]) => k);
      console.log(
        `[q] ${item.id} run${run} ${failed.length ? "FAIL " + failed.join(",") : "pass"} | ${reply.slice(0, 90)}`,
      );
    }
  }

  // Aggregate pass rates per check name.
  const agg = {};
  for (const row of rows) {
    for (const [k, v] of Object.entries(row.checks)) {
      agg[k] ??= { pass: 0, total: 0 };
      agg[k].total++;
      if (v) agg[k].pass++;
    }
  }
  const summary = Object.fromEntries(
    Object.entries(agg).map(([k, { pass, total }]) => [k, `${pass}/${total}`]),
  );
  mkdirSync(resolve(ROOT, "bench/results"), { recursive: true });
  const file = resolve(ROOT, `bench/results/quality-${label}-${Date.now()}.json`);
  writeFileSync(file, JSON.stringify({ label, tunables, holdout: args.holdout != null, summary, rows }, null, 2));
  console.log("\n[quality]", label, JSON.stringify(summary, null, 1));
  console.log("[quality] wrote", file);
} finally {
  await browser.close();
}
