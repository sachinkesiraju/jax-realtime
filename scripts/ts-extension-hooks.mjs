// Module-resolution hook so `node --experimental-transform-types` can run the
// app's source directly: the codebase uses extensionless relative imports
// (bundler resolution, e.g. `import { TUNABLES } from "../tunables"`), which
// Node's ESM loader does not resolve on its own. This appends `.ts` to any
// extensionless relative specifier that maps to an existing .ts file.
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

export async function resolve(specifier, context, nextResolve) {
  if (
    context.parentURL &&
    (specifier.startsWith("./") || specifier.startsWith("../")) &&
    !/\.[cm]?[jt]sx?$/.test(specifier)
  ) {
    const candidate = new URL(`${specifier}.ts`, context.parentURL);
    if (existsSync(fileURLToPath(candidate))) {
      return nextResolve(`${specifier}.ts`, context);
    }
  }
  return nextResolve(specifier, context);
}
