// Registers the extensionless-.ts resolver hook. Loaded via `node --import`
// before the main test module (see `npm run test:asr`) so the hook applies to
// the app source it pulls in.
import { register } from "node:module";

register("./ts-extension-hooks.mjs", import.meta.url);
