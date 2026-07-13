// Shared Chrome launcher for the bench scripts. Centralizes the hygiene that
// keeps repeated bench runs from bloating or crashing Chrome:
//   - sweeps stale Chrome processes still holding bench/.profile (an
//     interrupted run's zombie otherwise lingers and the next launch fights
//     it for the profile lock),
//   - caps the HTTP disk cache at 1 MB — the ~790 MB of model weights live in
//     the profile's OPFS store (that cache is the POINT of the persistent
//     profile; delete bench/.profile to reclaim it), and without the cap the
//     HTTP cache holds a redundant second copy of every safetensors download,
//   - closes the browser on SIGINT/SIGTERM/uncaught errors so a Ctrl-C mid-run
//     can't leave a zombie.
import { launch } from "puppeteer-core";
import { execSync } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PROFILE = resolve(ROOT, "bench/.profile");

export async function launchBench(extraArgs = []) {
  // Sweep zombies from interrupted runs (pgrep -f matches the profile path in
  // the command line; our own launch hasn't happened yet so this only hits
  // stale instances).
  try {
    execSync(`pkill -f "user-data-dir=${PROFILE}"`, { stdio: "ignore" });
    await new Promise((r) => setTimeout(r, 1500));
  } catch {
    // pkill exits 1 when nothing matched — the common, good case.
  }

  const browser = await launch({
    executablePath: CHROME,
    headless: false,
    // Model load blocks the page main thread for long stretches (wasm
    // compile, GPU warmup) which stalls CDP round-trips — no protocol timeout.
    protocolTimeout: 0,
    userDataDir: PROFILE,
    args: [
      "--no-sandbox", // fake-device file capture cannot read the WAV under the sandbox
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required",
      "--mute-audio",
      "--disk-cache-size=1048576",
      ...extraArgs,
    ],
  });

  // Interrupt-safe teardown: an uncaught rejection or Ctrl-C must not leave a
  // Chrome zombie holding the profile (and ~1 GB of model memory).
  const closeAndExit = (code) => {
    browser.close().catch(() => {}).finally(() => process.exit(code));
  };
  process.on("SIGINT", () => closeAndExit(130));
  process.on("SIGTERM", () => closeAndExit(143));
  process.on("uncaughtException", (err) => {
    console.error(err);
    closeAndExit(1);
  });
  process.on("unhandledRejection", (err) => {
    console.error(err);
    closeAndExit(1);
  });

  return browser;
}
