# Spec v2: Close the gap with Thinking Machines' interaction-model demo

The full-duplex core (v1, see DUPLEX_SPEC.md) is built and working: 150 ms
micro-turn engine (`src/duplex.ts`), always-on mic, dual compute lanes (Whisper
ASR on **wasm**, Gemma LLM + Pocket TTS on **webgpu**), barge-in, backchannels,
adaptive endpointing, time-awareness timers, live captions, duplex orb. ASR is
warmed at load (fixed a ~40 s cold-start). `npx tsc --noEmit` and `npm run build`
are clean. Dev server runs cross-origin-isolated (COOP/COEP in vite.config.ts).

This spec adds four capabilities to get closer to
https://thinkingmachines.ai/blog/interaction-models/. Build IN THIS ORDER
(each is independently shippable; do not let a later one break an earlier one):

Reference jax-js clone (API patterns, demos to port):
`/private/tmp/claude-501/-Users-sachinkesiraju-Downloads-jax-realtime/ab484e12-d6f3-4126-829e-4773697d2671/scratchpad/jax-js`

Read the existing `src/**` first and match its style + jax-js refcounting
(`.ref` / `.dispose()`, `tree.ref`, `tree.dispose`; jit fns consume inputs).
Constraints that bit us before, still true:
- NEVER `AudioContext.suspend()` the capture context (kills the mic feed).
- `playTTS` disposes the model refs it receives → always pass `tree.ref(model)`.
- One transcribe pass per SpeechRecognizer at a time (guarded by `isBusy`).
- Whisper on wasm = fp32; fp16 is webgpu-only.
- Keep the Cerebras path compiling.
- `npx tsc --noEmit` + `npm run build` must stay green. Don't run the browser or
  commit; the reviewer (Fable) drives Chrome and finishes verification.

---

## Feature 1 — Latency hillclimb (do FIRST; foundational)

Today `endUserTurn` calls `transcriber.finalize()` — a full extra wasm Whisper
pass (~3.6 s warm) on the whole utterance — before the LLM even starts. That
pass dominates turn latency. The streaming loop has ALREADY transcribed the
utterance incrementally; use that.

In `src/asr/streaming.ts`:
- Add `bestText(): string` returning `(committed + " " + tentative).trim()`.

In `src/duplex.ts` `endUserTurn`:
- Replace the blocking `finalize()` with: `const text = transcriber.bestText()`.
  If `text` has ≥ 3 words, use it directly and DO NOT finalize (save the pass).
  If it's shorter/empty (streaming hadn't caught up), fall back to `finalize()`.
- Optionally kick a background `finalize()` that, if it returns materially more
  text than `bestText()` before the LLM has emitted its first token, replaces
  the user bubble text — but only if trivial to do safely; otherwise skip.
- Metric: this should drop end-of-speech→first-audio substantially; keep
  reporting `turnLatencyMs` (already wired to the rail).

Acceptance: warm turn latency (end of speech → first assistant audio) should be
roughly LLM-first-token + TTS-first-audio, with no separate multi-second ASR
finalize on the critical path. Live captions must still show during speech.

---

## Feature 2 — Vision "Eye" stage (proactive visual interjections)

Their signature (SLOUCHING, DANGER, ProactiveVideoQA). Port jax-js's **D-FINE**
object detector (webcam, COCO-80), which runs on jax-js's own backends via
`@jax-js/onnx` (NOT onnxruntime). Reference:
`website/src/routes/d-fine/+page.svelte` and `website/src/routes/detr-resnet-50/coco.ts`
in the clone.

Setup:
- `npm i @jax-js/onnx onnx-buf` (onnx-buf is a peer of the loader path used by
  the demo; add whatever the D-FINE page imports).
- Copy `coco.ts` (COCO_CLASSES) into `src/vision/coco.ts`.
- Model URL (COCO): `https://huggingface.co/bukuroo/D-FINE-ONNX/resolve/main/dfine_s_obj2coco.onnx`.

`src/vision/detector.ts`:
- `class ObjectDetector` wrapping `ONNXModel` (see D-FINE page for load + the
  exact input tensor prep: letterbox resize to the model's expected size,
  RGB float normalization, NCHW; and output post-processing → boxes, scores,
  labels; threshold ~0.4). Expose:
  - `static load(onProgress): Promise<ObjectDetector>` (fetch weights via the
    existing DownloadManager-style progress) and a `warmup()` (one dummy frame).
  - `detect(source: HTMLVideoElement | HTMLCanvasElement): Promise<Detection[]>`
    where `Detection = { label: string; score: number; box: [x,y,w,h] }`
    (box in source-pixel coords). One detection at a time (guard `isBusy`).
- Run on webgpu by default (shares the lane; 1 fps is light). If that visibly
  stalls TTS in the reviewer's tests, allow wasm via a device param.

`src/vision/vision.ts`:
- `class VisionSession`:
  - `start()`: `getUserMedia({ video: { facingMode: "user" } })`, attach to an
    off-DOM (or preview) `<video>`, and loop: every ~1200 ms grab the current
    frame and run `detector.detect`. Store `latest: Detection[]`.
  - Derived scene state getters: `personCount`, `personPresent`,
    `phonePresent` (COCO "cell phone"), and a coarse posture proxy from the
    largest person's box (e.g. `slouching` when the box top drops and height
    shrinks vs. a short rolling baseline — best-effort, label it approximate).
  - `describe(): string` → compact scene string for the LLM, e.g.
    `"1 person, holding a cell phone"` or `"no one in frame"`.
  - `stop()`, and expose `latest` + the video element for the preview.

Wire into `DuplexSession` (constructor takes an optional `vision`):
- Proactive interjections (only when user NOT speaking, assistant NOT speaking,
  not responding; throttle: one per state-change, ≥8 s cooldown, via the same
  `speakProactive()` path already used by timers):
  - person was present ≥3 frames, now absent ≥3 frames → "Did you step away?
    I'll be here when you're back."
  - "cell phone" newly appears and persists ≥3 frames → "Phone again? I can
    wait."
  - posture `slouching` persists ≥4 frames → "Hey — sit up straight."
  Keep these as a small rule table; make the lines a const array.
- Scene grounding for Q&A (ProactiveVideoQA-lite): when a user turn is finalized
  and vision is on, prepend `[scene: <describe()>]` to that user message
  (like the `[t+Ns]` tag) so "what do you see?" / "how many people?" work. Do
  NOT read the tag aloud (system hint already covers bracket tags).

UI (`main.ts`, `style.css`):
- Add a 4th rail stage card **Eye — D-FINE** with an activity dot; only shows
  active state when vision is enabled.
- Webcam preview thumbnail in the console (corner PiP like the TML demos) with
  optional detection boxes drawn on a canvas overlay.
- Dock toggle "Eye · webcam" (off by default). Enabling loads+warms the model
  (progress in the downloads panel), starts VisionSession, and passes it to the
  active/next DuplexSession. Disabling stops the camera.

Skip rep-*counting* of motion (unreliable); object counting ("how many people")
is fine and covered by `describe()`.

---

## Feature 3 — Two-tier architecture + concurrent tool use + generative UI

Their headline (UBER, SEARCH, the real-time + async background model). The fast
local model stays present and talking while a **background async task** does the
slow work, then results are woven back + rendered as a generative UI card.

`src/tools/tools.ts`:
- Intent detection on a finalized user turn (regex/keywords), returning a
  `ToolCall | null`:
  - `weather`: /weather|temperature|forecast/ + a location → open-meteo
    (CORS-open, no key): geocode `https://geocoding-api.open-meteo.com/v1/search?name=<q>&count=1`
    then `https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&current=temperature_2m,weather_code,wind_speed_10m`.
  - `lookup` (web search stand-in): /search|look up|who is|what is|tell me about/
    + an entity → Wikipedia REST summary (CORS-open):
    `https://en.wikipedia.org/api/rest_v1/page/summary/<Title>` → title + extract.
  - (timers already handled by the duplex time-awareness path — leave as is.)
- Each ToolCall has `{ kind, query, run(): Promise<ToolResult> }` where
  `ToolResult = { speech: string; card: UiCard }` and `UiCard` is a small tagged
  union (`weather`, `factcard`, `list`) with plain data for rendering.

Two-tier flow in `DuplexSession`:
- When a finalized user turn produces a ToolCall, DO NOT run the normal LLM
  reply path. Instead:
  1. Immediately speak a holding line via `speakProactive`-style TTS
     ("Let me look that up.") and emit a **tool-call chip** event (a new
     callback `onToolCall(kind, query)`), then return the tick loop to
     listening — the user can still talk / interrupt / be backchanneled while
     the background task runs (this IS the two-tier "stays present" behavior).
  2. Run `toolCall.run()` as a background promise (NOT awaited in the tick).
  3. On resolve: if the user isn't mid-utterance, speak `result.speech` and fire
     `onToolResult(card)`; push a synthetic assistant history entry
     (`content: result.speech`) so context stays coherent. If the user is
     speaking, queue it and deliver at the next silence.
- Guard against overlap: only one background tool task at a time; if another
  intent arrives, replace/queue sensibly.

Generative UI (`main.ts`, `style.css`):
- A "canvas" panel (right of / below the transcript, or a slide-in) that renders
  the latest `UiCard`:
  - weather: location, big temp, condition text/emoji, wind.
  - factcard: title, extract (clamped), "via Wikipedia" source line.
  - list: title + bullet items.
- Tool-call chip in the transcript stream (small monospace pill like the TML
  `web_search` chip) shown while the task runs, resolving to done.
- A background-activity indicator on the rail (e.g. a "Background" pill that
  pulses while a tool task runs) to make the two-tier concurrency legible.

Everything CORS-open and keyless so it works out of the box. If a Cerebras key
is present, you MAY route the final summary phrasing through it, but the tool
fetch itself must be keyless.

---

## Feature 4 — Simultaneous speech / streaming response (do LAST; best-effort)

Their ANGER demo: user and model speak concurrently (e.g. live translation).
True same-stream concurrency is the hardest; our two lanes (ASR wasm, TTS
webgpu) already run concurrently, and we have a text self-echo filter. Build a
**"Simultaneous" mode** toggle (off by default):

- When ON, do NOT wait for end-of-turn. As the streaming transcriber COMMITS
  each new sentence (terminal punct in committed text), immediately process that
  sentence while the user keeps talking:
  - Translation sub-mode (a small target-language `<select>`: Spanish/French/
    German): translate the committed sentence via the LLM
    (`"Translate to <lang>, output only the translation: <sentence>"`) and speak
    it. NOTE + document honestly: Kyutai Pocket TTS is English-only, so a
    non-English translation is spoken with an English voice reading foreign text
    — the point demonstrated is the *simultaneity* (speak while listening), not
    TTS quality. Cerebras key (if set) gives much better translations.
  - The self-echo filter must keep our TTS out of the ASR commit stream; verify
    the filter also works when languages differ (word-overlap will be low, which
    is fine — it just won't false-trigger).
- Concurrency requirement: the ASR streaming loop keeps running and committing
  new sentences WHILE `speakStream` is playing the previous sentence's
  translation. Use a queue: committed sentences enqueue; a single consumer
  translates+speaks them in order on one shared player; the mic never pauses.
- Barge-in/normal endpointing are disabled in this mode (it's a continuous
  interpreter cadence). Exiting the mode returns to normal duplex.
- If multilingual INPUT is needed later, note that `src/asr/model.ts` already
  has a multilingual Whisper config path; for this pass assume English input →
  target-language output.

Acceptance: with Simultaneous mode on, speaking several sentences continuously
produces spoken output for earlier sentences while later ones are still being
transcribed — ASR (wasm) and TTS (webgpu) demonstrably overlap. Honestly
document the English-TTS limitation in the README.

---

## Cross-cutting

- Keep the tagline/README honest: everything runs in-browser on jax-js; call out
  what's emulated vs. native (cascade vs. TML's single model), and the
  English-TTS translation caveat.
- Update README with the new modes and a one-line "how close to TML" table.
- New deps: `@jax-js/onnx`, `onnx-buf` (vision). No other runtime deps
  (tools use fetch; UI is hand-rendered).
- Reviewer will verify in Chrome with a fake mic + (for vision) may need to
  point the camera or use a canvas-fed fake video; make vision degrade
  gracefully if no camera.

## Final report (agent → reviewer)

Per-feature: what you built, deviations + why, files touched, what compiles,
what you could NOT finish, and the top things for the reviewer to check in
Chrome (with any device/perf risks, esp. webgpu contention from vision).
