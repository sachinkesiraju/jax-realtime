// One-off console probe: load models, then evaluate an expression against the
// dev hooks. Usage: node bench/probe.mjs '<js expr returning a promise>'
import { resolve } from "node:path";
import { launchBench, ROOT } from "./launch.mjs";

const expr = process.argv[2];

const browser = await launchBench();
try {
  const page = (await browser.pages())[0] ?? (await browser.newPage());
  await page.goto("http://localhost:5173", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#load-btn:not([disabled])", { timeout: 30_000 });
  await page.click("#load-btn");
  await page.waitForSelector("#orb-btn:not([disabled])", { timeout: 10 * 60_000 });
  const result = await page.evaluate(expr);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
