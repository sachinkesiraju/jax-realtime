# Eye (vision) — design notes & the VLM investigation

The Eye is a **low-priority** background signal for the realtime voice agent. It
runs D-FINE object detection (COCO-80) on the WebGPU lane, yielding to
ASR/LLM/TTS, at ~2.2 s intervals.

## What the Eye answers, and how

| Query type | Method | Cost |
| --- | --- | --- |
| "what do you see", "what's behind me", "how many people" | D-FINE detections (temporally stable: a label must appear in ≥2 of the last 3 frames, so single-frame false positives like a phantom "cat" are dropped) | free (already running) |
| "what colour is my chair / what am I wearing" | average the pixels inside the object's box (person → torso region) and name the dominant colour via HSL bucketing | ~free (canvas pixels) |
| "what am I doing" (activity) | **honest, grounded** — report the objects actually in frame and state plainly we can't read the action. We do NOT fabricate an activity from a hardcoded object→activity table. | free |
| proactive (stepped away / phone / slouch) | D-FINE + a coarse posture proxy, rule table with cooldown | free |

Scene questions are answered **directly from the detector**, not routed through
Gemma 270M (which fumbles fuzzy scene text).

## Cheap-VLM investigation (why there's no VLM here yet)

We investigated adding a small vision-language model so the Eye could answer
open-vocab / attribute / activity questions without hardcoding.

Findings (July 2026):
- **MobileCLIP2-S0** (Apple; jax-js already downloads this checkpoint and
  implements its *text* encoder, `visual: any // TODO`). ONNX export at
  `plhery/mobileclip2-onnx`: **vision 45 MB, text 254 MB**, 256×256 input,
  512-d embeddings, WebGPU via transformers.js.
- **SmolVLM-256M/500M** — real generative VQA/captioning, ONNX + WebGPU via
  transformers.js; ~150–300 MB, autoregressive (slower per query).
- The **elegant path** would be: run only the 45 MB vision encoder via
  `@jax-js/onnx` (no second runtime, no 254 MB text model), pair it with a
  **precomputed prompt-embedding bank** (~hundreds of KB, made offline with the
  text encoder over a curated vocabulary), and do zero-shot cosine matching at
  query time — with a "Gemma proposes candidates → CLIP verifies against pixels"
  loop so nothing is hardcoded.
- **Blocker (verified by browser probe):** `@jax-js/onnx` loads the vision ONNX
  but **cannot execute its graph** — it hits an unsupported op lowering
  (`Invalid type for full`, a dynamic-shape/ConstantOfShape-style op) under both
  `jit` and eager. D-FINE runs because its ops are simpler.

## Decision

Do **not** add a VLM now. The high-value Eye queries are already served cheaply
(D-FINE + pixel colour); the only gap is the least-used, interpretive activity
query on a low-priority channel — not worth a permanent second inference runtime
(onnxruntime-web via transformers.js). Instead we removed the brittle hardcoded
activity heuristics and answer honestly.

## Upgrade path (if open-vocab vision / richer proactive cues become a priority)

1. Add `@huggingface/transformers` (onnxruntime-web). Serve its wasm under the
   existing COOP/COEP headers.
2. Load **only** `plhery/mobileclip2-onnx` **vision** (45 MB) on WebGPU.
3. Offline, once, run the text encoder over a curated prompt vocabulary
   (activities, attributes, open-vocab objects, proactive cues like "a person
   slouching / looking distressed") → save `bank.json` (~N×512 floats).
4. At query time (on-demand, low priority): one vision forward pass on the
   webcam frame → L2-normalize → cosine vs the bank → top matches. Optionally
   have Gemma generate the candidate set per question for open-ended coverage.
5. This also upgrades the proactive interjections (TML slouching/danger style)
   beyond D-FINE's 80 classes.
