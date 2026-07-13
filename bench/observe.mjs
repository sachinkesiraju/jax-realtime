// Adversarial detection bench: play a clip (noise / quiet speech / normal
// speech) through the fake mic for a fixed duration and report what the
// micro-turn engine DID — turns fired (TURN_LOG), transcripts, and the
// policy-event ticker history (phantom discards, endpoint causes).
//
//   node bench/observe.mjs --clip bench/clips/noise_typing.wav --seconds 60 \
//     --label typing-baseline --tunables '{}'
//
// Success criteria live with the caller: a noise clip should fire ZERO turns;
// a quiet-speech clip should fire one turn per ~16s loop.

import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { launchBench, ROOT } from "./launch.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, all) =>
    a.startsWith("--") ? [a.slice(2), all[i + 1]] : null,
  ).filter(Boolean),
);
const clip = resolve(ROOT, args.clip ?? "bench/clips/noise_typing.wav");
const seconds = Number(args.seconds ?? 60);
const label = args.label ?? "observe";
const tunables = JSON.parse(args.tunables ?? "{}");

const browser = await launchBench([
  `--use-file-for-fake-audio-capture=${clip}`,
  "--window-size=1200,900",
]);
try {
  const page = (await browser.pages())[0] ?? (await browser.newPage());
  await page.goto(args.url ?? "http://localhost:5173", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForSelector("#load-btn:not([disabled])", { timeout: 30_000 });
  // Eye off before load (lazy detector never loads).
  await page.evaluate(() => {
    const t = document.querySelector("#eye-toggle");
    if (t?.checked) t.click();
  });
  await page.click("#load-btn");
  await page.waitForSelector("#orb-btn:not([disabled])", {
    timeout: 30 * 60_000,
  });
  await page.evaluate((over) => Object.assign(window.__tunables, over), tunables);

  // Record every ticker event (the ticker DOM shows only the latest, so
  // observe mutations for the session's full policy-event history).
  await page.evaluate(() => {
    window.__events = [];
    const el = document.querySelector("#ticker");
    new MutationObserver(() => {
      const t = el.textContent?.trim();
      if (t) window.__events.push(t);
    }).observe(el, { childList: true, characterData: true, subtree: true });
  });

  const before = await page.evaluate(() => window.__turnLog.length);
  await page.click("#orb-btn");
  console.log(`[observe] session up; watching for ${seconds}s…`);
  await new Promise((r) => setTimeout(r, seconds * 1000));
  await page.click("#orb-btn");

  const report = await page.evaluate((b) => {
    const turns = window.__turnLog.slice(b).map((t) => ({
      transcript: t.transcript,
      reply: (t.reply ?? "").slice(0, 60),
      endCause: t.endCause,
    }));
    const events = window.__events;
    const count = (re) => events.filter((e) => re.test(e)).length;
    return {
      turnsFired: turns.length,
      turns,
      eventCounts: {
        phantomDiscards: count(/discarded|garbled/i),
        backchannels: count(/backchannel/i),
        total: events.length,
      },
      events: events.slice(-30),
    };
  }, before);

  mkdirSync(resolve(ROOT, "bench/results"), { recursive: true });
  const file = resolve(ROOT, `bench/results/observe-${label}-${Date.now()}.json`);
  writeFileSync(file, JSON.stringify({ label, clip, seconds, tunables, ...report }, null, 2));
  console.log(`[observe] ${label}: ${report.turnsFired} turns fired in ${seconds}s`);
  for (const t of report.turns) console.log(`  - "${t.transcript}" -> "${t.reply}"`);
  console.log("[observe] events:", JSON.stringify(report.eventCounts));
  console.log("[observe] wrote", file);
} finally {
  await browser.close();
}
