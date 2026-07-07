import { defineConfig } from "vite";

// Cross-origin isolation headers. jax-js's wasm backend multithreads via
// SharedArrayBuffer (see src/backend/wasm/parallel.ts in the jax-js repo),
// which browsers only expose when the document is cross-origin isolated.
// HuggingFace weight downloads are CORS-enabled, so they keep working under
// require-corp.
const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  build: {
    chunkSizeWarningLimit: 4000,
  },
  server: {
    headers: crossOriginIsolation,
  },
  preview: {
    headers: crossOriginIsolation,
  },
});
