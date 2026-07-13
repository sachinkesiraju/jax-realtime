// Memory probe: measures JS heap (CDP) + Chrome helper-process RSS (GPU +
// renderer, via ps — on Apple Silicon unified memory the GPU process footprint
// is the closest observable proxy for WebGPU buffer residency) at three
// points: page loaded, models loaded, after N scripted LLM turns.
//
//   node bench/memory.mjs --label baseline --tunables '{}' [--eye on|off]
//
// Writes bench/results/memory-<label>-<ts>.json.

import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { launchBench, ROOT } from "./launch.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, all) =>
    a.startsWith("--") ? [a.slice(2), all[i + 1]] : null,
  ).filter(Boolean),
);
const label = args.label ?? "memory";
const tunables = JSON.parse(args.tunables ?? "{}");
const eye = args.eye ?? "off";

// RSS (MB) of this Chrome instance's helper processes, keyed by type. The
// bench profile dir makes our instance's processes identifiable.
function helperRss() {
  const out = execSync("ps -axo rss=,command=", { maxBuffer: 64e6 }).toString();
  const rows = out.split("\n").filter((l) => l.includes("bench/.profile"));
  const sum = (re) =>
    Math.round(
      rows.filter((l) => re.test(l)).reduce((a, l) => a + Number(l.trim().split(/\s+/)[0]), 0) / 1024,
    );
  return {
    gpuMB: sum(/--type=gpu-process/),
    rendererMB: sum(/--type=renderer/),
    totalMB: sum(/./),
  };
}

const browser = await launchBench();
try {
  const page = (await browser.pages())[0] ?? (await browser.newPage());
  const snap = async (phase) => {
    // Nudge GC via CDP so paired snapshots compare live sets, not GC timing.
    try {
      const client = await page.createCDPSession();
      await client.send("HeapProfiler.collectGarbage");
      await client.detach();
    } catch {}
    await new Promise((r) => setTimeout(r, 1500));
    const m = await page.metrics();
    return { phase, jsHeapMB: Math.round(m.JSHeapUsedSize / 1048576), ...helperRss() };
  };

  const snaps = [];
  await page.goto(args.url ?? "http://localhost:5173", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#load-btn:not([disabled])", { timeout: 30_000 });
  snaps.push(await snap("page-loaded"));

  await page.evaluate((over) => Object.assign(window.__tunables, over), tunables);
  if (eye === "off") {
    // Uncheck BEFORE load: with lazy D-FINE this must prevent the download +
    // GPU residency entirely (on the old eager code it preloads regardless).
    await page.evaluate(() => {
      const t = document.querySelector("#eye-toggle");
      if (t?.checked) t.click();
    });
  }
  await page.click("#load-btn");
  await page.waitForSelector("#orb-btn:not([disabled])", { timeout: 30 * 60_000 });
  if (eye === "off") {
    await page.evaluate(() => {
      const t = document.querySelector("#eye-toggle");
      if (t?.checked) t.click();
      window.__vision?.()?.stop?.();
    });
  }
  await new Promise((r) => setTimeout(r, 5000)); // let any post-ready lazy work settle
  snaps.push(await snap("models-loaded"));

  // A few scripted turns straight through the LLM (per-turn allocations,
  // KV growth, log strings) — no audio needed.
  await page.evaluate(async () => {
    const histories = [
      [{ role: "user", content: "Tell me about the ocean and its creatures in detail.", t: 0 }],
      [{ role: "user", content: "What are some good habits for staying healthy?", t: 0 }],
      [{ role: "user", content: "Describe a perfect weekend day from start to finish.", t: 0 }],
    ];
    for (const h of histories) await window.__pipeline().llm.generate(h, () => {});
  });
  snaps.push(await snap("after-3-turns"));

  mkdirSync(resolve(ROOT, "bench/results"), { recursive: true });
  const file = resolve(ROOT, `bench/results/memory-${label}-${Date.now()}.json`);
  writeFileSync(file, JSON.stringify({ label, tunables, eye, snaps }, null, 2));
  console.table(snaps);
  console.log("[memory] wrote", file);
} finally {
  await browser.close();
}
