// Repeatable fake-mic latency bench (map-reduce harness).
//
// Drives the dev app in real Chrome with a fake mic clip and samples the
// per-turn stage marks from window.__turnLog (DEV hooks). One process = one
// condition = one page session; conditions differ only by TUNABLES overrides,
// so a single build serves every candidate.
//
//   node bench/run.mjs --clip bench/clips/map_a.wav --turns 5 \
//     --label baseline --tunables '{"llmKvReuse":false}'
//
// Writes bench/results/<label>-<timestamp>.json and prints a summary table.
// The Chrome profile persists in bench/.profile so OPFS-cached weights are
// downloaded once. The Eye is force-disabled (it interjects mid-bench).

import { launch } from "puppeteer-core";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME =
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, all) =>
    a.startsWith("--") ? [a.slice(2), all[i + 1]] : null,
  ).filter(Boolean),
);
const clip = resolve(ROOT, args.clip ?? "bench/clips/map_a.wav");
const turns = Number(args.turns ?? 5);
const label = args.label ?? "run";
const tunables = JSON.parse(args.tunables ?? "{}");
const url = args.url ?? "http://localhost:5173";
// Per-turn budget: clip loop is ~16.4 s; allow slack for cold turn 1.
const timeoutMs = Number(args.timeout ?? (turns * 16_400 + 120_000));

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};

const browser = await launch({
  executablePath: CHROME,
  headless: false,
  // Model load blocks the page main thread for long stretches (wasm compile,
  // GPU warmup), which stalls CDP round-trips — disable the protocol timeout.
  protocolTimeout: 0,
  userDataDir: resolve(ROOT, "bench/.profile"),
  args: [
    "--no-sandbox", // fake-device file capture cannot read the WAV under the sandbox
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    `--use-file-for-fake-audio-capture=${clip}`,
    "--autoplay-policy=no-user-gesture-required",
    "--mute-audio",
    "--window-size=1200,900",
  ],
});

try {
  const page = (await browser.pages())[0] ?? (await browser.newPage());
  page.on("console", (m) => {
    const t = m.text();
    if (/error|failed/i.test(t)) console.error("[page]", t.slice(0, 200));
  });
  await page.goto(url, { waitUntil: "domcontentloaded" });

  // Load models (OPFS-cached after the first run; first run downloads ~680MB).
  await page.waitForSelector("#load-btn:not([disabled])", { timeout: 30_000 });
  console.log("[bench] loading models…");
  await page.click("#load-btn");
  await page.waitForSelector("#orb-btn:not([disabled])", {
    timeout: 30 * 60_000,
  });
  console.log("[bench] models ready");

  // Kill the Eye before the session starts (it interjects mid-bench).
  await page.evaluate(() => {
    const t = document.querySelector("#eye-toggle");
    if (t?.checked) t.click();
    window.__vision?.()?.stop?.();
  });

  // Apply the condition's tunable overrides.
  await page.evaluate((over) => {
    Object.assign(window.__tunables, over);
  }, tunables);

  const before = await page.evaluate(() => window.__turnLog.length);
  await page.click("#orb-btn");
  console.log(`[bench] session started; waiting for ${turns} turns…`);

  const deadline = Date.now() + timeoutMs;
  let count = before;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2_000));
    const n = await page.evaluate(() => window.__turnLog.length);
    if (n !== count) {
      count = n;
      const last = await page.evaluate(() => {
        const t = window.__turnLog[window.__turnLog.length - 1];
        return {
          lat: Math.round(t.firstAudio - t.endOfSpeech),
          transcript: t.transcript,
          reply: (t.reply ?? "").slice(0, 60),
        };
      });
      console.log(
        `[bench] turn ${n - before}/${turns}: ${last.lat} ms | "${last.transcript}" -> "${last.reply}"`,
      );
    }
    if (n - before >= turns) break;
  }

  await page.click("#orb-btn"); // stop session
  const records = await page.evaluate(
    (b) => window.__turnLog.slice(b),
    before,
  );

  const rows = records.map((t) => ({
    turnLat: Math.round(t.firstAudio - t.endOfSpeech),
    endpoint: Math.round(t.fired - t.endOfSpeech),
    asr: Math.round(t.transcriptReady - t.fired),
    llmFirst: Math.round(t.firstDelta - t.transcriptReady),
    sentence: Math.round(t.firstSentence - t.firstDelta),
    tts: Math.round(t.firstAudio - t.firstSentence),
    onset: t.onsetAudio ? Math.round(t.onsetAudio - t.endOfSpeech) : null,
    endCause: t.endCause,
    interrupted: t.interrupted,
    transcript: t.transcript,
    reply: t.reply,
  }));
  // Turn 1 is cold (jit warmup); report separately, exclude from medians.
  // Interrupted turns stay in: barge-in happens AFTER firstAudio, so the
  // latency marks are complete (the fake clip's next loop can cut a long
  // reply short — that's a reply-length artifact, not a latency one).
  const warm = rows.slice(1);
  const summary = {
    label,
    clip,
    tunables,
    turns: rows.length,
    warmMedians: {
      turnLat: median(warm.map((r) => r.turnLat)),
      endpoint: median(warm.map((r) => r.endpoint)),
      asr: median(warm.map((r) => r.asr)),
      llmFirst: median(warm.map((r) => r.llmFirst)),
      sentence: median(warm.map((r) => r.sentence)),
      tts: median(warm.map((r) => r.tts)),
      onset: median(warm.map((r) => r.onset).filter((x) => x != null)),
    },
    rows,
  };

  mkdirSync(resolve(ROOT, "bench/results"), { recursive: true });
  const file = resolve(
    ROOT,
    `bench/results/${label}-${Date.now()}.json`,
  );
  writeFileSync(file, JSON.stringify(summary, null, 2));
  console.log("\n[bench] label:", label);
  console.table(rows.map(({ transcript, reply, ...r }) => r));
  console.log("[bench] warm medians:", summary.warmMedians);
  console.log("[bench] wrote", file);
} finally {
  await browser.close();
}
