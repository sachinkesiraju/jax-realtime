// One-off console probe: load models, then evaluate an expression against the
// dev hooks. Usage: node bench/probe.mjs '<js expr returning a promise>'
import { launch } from "puppeteer-core";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const expr = process.argv[2];

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
  await page.waitForSelector("#orb-btn:not([disabled])", { timeout: 10 * 60_000 });
  const result = await page.evaluate(expr);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
