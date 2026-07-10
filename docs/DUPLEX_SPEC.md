# Spec: Full-duplex "interaction model" mode (Thinking Machines-style)

Goal: evolve this app from a turn-based voice assistant into an emulation of
Thinking Machines' interaction models (https://thinkingmachines.ai/blog/interaction-models/)
— continuous micro-turn processing, no hard turn boundaries, barge-in
interruption, backchanneling, time awareness, proactive interjections, vision,
concurrent tool use, and simultaneous speech — all in-browser with jax-js. We
keep the cascaded pipeline (Whisper → local LLM → Pocket TTS) but restructure the
harness into a 150–200 ms tick loop.

---

# Part I — Full-duplex core (built)

## 1. Two compute lanes (critical)

ASR moves to the **wasm backend** so Whisper passes run continuously without
stalling TTS/LLM on WebGPU:

- `vite.config.ts`: add dev+preview headers
  `Cross-Origin-Opener-Policy: same-origin`,
  `Cross-Origin-Embedder-Policy: require-corp` (required for
  SharedArrayBuffer multithreading in jax-js's wasm backend — see
  `src/backend/wasm/parallel.ts` in the clone).
- Because of COEP, replace the Google Fonts `<link>` in `index.html` with
  self-hosted fonts: `npm i @fontsource-variable/instrument-sans
  @fontsource/jetbrains-mono` and import them in `src/main.ts` (or style.css).
  HF weight downloads are CORS-enabled and keep working.
- `await init("webgpu", "wasm")` at startup; require webgpu, warn-and-degrade
  if wasm missing (then ASR shares webgpu as today).
- Pin ASR to wasm via **explicit device placement** — never flip
  `defaultDevice` (races with concurrent WebGPU work). In `src/asr/model.ts`
  and the ASR call path, every array creation gets `{ device }`:
  weights in `tensorToArray`/`fromSafetensors` (thread a `device` param
  through), `np.zeros` in `createWhisperState`, token/position arrays in
  `runWhisperDecoderStep`, and the features array in the transcribe path.
  Whisper on wasm must use **fp32** (`np.float32`) — fp16 is webgpu-only.
  Ops inherit device from operands; jit re-specializes per device. Verify
  with a one-off test that a wasm transcribe pass and a webgpu TTS call can
  run concurrently (Promise.all) without errors.

## 2. Streaming ASR (`src/asr/streaming.ts`)

Class `StreamingTranscriber` wrapping SpeechRecognizer:

- Owns a loop: while active, transcribe the current utterance window
  (capture buffer since utterance start, max 28 s) back-to-back (each pass is
  its own async call; no timer needed — start next pass when previous ends,
  but throttle to ≥400 ms between pass starts).
- **LocalAgreement-2 commit policy**: keep the previous hypothesis; the
  longest common word-prefix of the current and previous hypothesis becomes
  `committed`; the remainder of the newest hypothesis is `tentative`.
- Emits `{ committed, tentative, lastChangeAt }` via a callback each pass.
- Method `finalize()`: one last pass over the full utterance, returns final
  text (used when the policy closes a user turn).
- Self-echo filter: constructor takes `getAssistantUtterance(): string | null`;
  when the assistant is speaking, drop hypothesis words that fuzzy-match the
  current TTS text (lowercase, strip punctuation, ≥70 % of words present in
  the TTS line ⇒ treat as echo, not user speech).

## 3. Abortable, streaming TTS

- `src/tts/inference.ts` `playTTS`: add `signal?: AbortSignal` in options;
  check `signal.aborted` at the top of each generation step; on abort, stop
  cleanly (dispose per existing finally block) and call `player.stop()`.
- `src/tts/audio.ts` streaming player: track live `AudioBufferSourceNode`s;
  add `stop()` that stops/disconnects all scheduled sources immediately.
  Keep the analyser tap.
- `SpeechSynthesizer.speak(voice, text, { signal, onAnalyser })` returns
  `{ firstAudioMs, totalMs, aborted: boolean }`.
- New `SpeechSynthesizer.speakStream(voice, sentences: AsyncIterable<string>, opts)`:
  synthesizes sentence-by-sentence sequentially on one shared player/abort;
  speech starts after the first sentence. Sentence splitter: accumulate LLM
  token stream, flush on `.  !  ?  …` followed by space/end, or ≥120 chars.
- **Pre-synthesized backchannels**: at load time (after models ready),
  synthesize "Mm-hmm.", "Right.", "Got it." once each, capture PCM via the
  player's `toWav`-style chunk collection (add a `collectPcm` option or use
  a non-playing player), cache as Float32Arrays. `playBackchannel()`: plays
  a random cached clip instantly through a short-lived AudioContext — never
  hits the GPU. If pre-synthesis is awkward, acceptable fallback: synthesize
  each backchannel live once on first use and cache the PCM then.

## 4. LLM

Keep `LocalChatModel.generate(history, onToken)` but expose the raw token
stream as an async iterator (`generateStream`) so main.ts can feed the
sentence splitter → `speakStream` as tokens arrive. Add to the SYSTEM_HINT:
answers must be 1–3 short spoken sentences; if the user message ends with
"[interrupted]", acknowledge briefly and yield. Keep CerebrasChatModel
working (non-streaming is fine there; wrap its full reply as a one-item
stream).

## 5. Micro-turn engine (`src/duplex.ts` — the heart)

A `DuplexSession` class driven by a ~150 ms `setInterval` tick. Mic is
**always capturing** (never paused, even while the assistant speaks).

State: `assistantSpeaking` (with current utterance text + abort controller),
`userSpeaking` (energy above threshold recently), utterance buffer bounds,
timers.

Policy per tick (deterministic, in priority order):

1. **Barge-in**: if `assistantSpeaking` and (sustained level > startLevel for
   ≥2 ticks) and streaming ASR yields ≥2 committed words that fail the
   self-echo filter → abort TTS + LLM stream, mark the assistant history
   entry `"…" + " [interrupted]"`, transition to user turn. Target: audio
   stops ≤200 ms after user starts talking over.
2. **User turn end**: user was speaking, now silent. Adaptive endpointing:
   silence ≥ 450 ms AND committed text ends in terminal punctuation → end
   turn; otherwise silence ≥ 800 ms → end turn; max utterance 28 s.
   On end: finalize the user turn (see Part II Feature 1 for the fast path),
   push history, start LLM stream → sentence splitter → `speakStream`. Record
   turn latency (end-of-speech → first TTS audio) for the metrics rail.
3. **Backchannel**: user mid-utterance pause 450–800 ms, utterance ≥ 2 s so
   far, at most one per utterance, assistant not speaking → `playBackchannel()`
   (do NOT treat as a turn end; ASR keeps accumulating the same utterance).
4. **Time awareness**: session clock. If the latest user turn matched
   /(\d+)\s*(second|sec|minute|min)/ + verbs like "timer|remind|tell me|when",
   schedule a proactive event; when it fires and the user isn't speaking,
   speak "That's ⟨N⟩ ⟨unit⟩ — time's up." directly via TTS (no LLM), and add
   it to history. Also prefix each user message in the prompt with
   `[t+⟨seconds⟩s]` so the LLM sees elapsed time.
5. **Idle**: nothing.

Session start/stop stays on the orb click. On stop: abort everything, close
mic.

## 6. UI (`src/main.ts`, `src/orb.ts`, `src/style.css`)

- Orb shows **both parties at once**: user level drives an outer ring
  (red) while assistant TTS analyser drives the core (lime) — full-duplex
  made visible. Add `orb.setDuplex(userLevel: () => number, ttsLevel: (() => number) | null)`
  or equivalent.
- Live caption strip under the orb: committed text solid, tentative text
  dimmed italic, replacing the old is-live bubble mechanics (bubbles are
  created/committed on turn end; an interrupted assistant bubble keeps the
  spoken-so-far text + " —").
- Event ticker (small monospace line under the captions): last policy event,
  e.g. `barge-in · stopped in 180 ms`, `backchannel`, `timer set (30 s)`,
  `turn latency 1.1 s`.
- Metrics rail: replace ASR/LLM/TTS values with rolling
  `turn latency` (end-of-speech→first-audio), `interrupt stop time`, `ASR lag`
  (audio-end→committed-text). Keep the Ear/Brain/Voice stage cards, add
  per-stage activity dots that light while each stage is actually computing
  (ASR can be active while Voice is speaking — show that).

---

# Part II — Closing the gap with the TML demo (roadmap)

Build IN THIS ORDER; each is independently shippable.

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

# Constraints & gotchas (learned the hard way)

- NEVER `AudioContext.suspend()` on the capture context (breaks the mic feed).
- `playTTS` disposes the model refs it receives — always pass `tree.ref(model)`.
- jax-js arrays are refcounted: match every `.ref` / `.dispose()`; jit fns
  consume their inputs. Follow the existing patterns exactly.
- Whisper on wasm = fp32 (fp16 is webgpu-only; double weight RAM — tiny.en is
  74 MB fp16 → ~150 MB fp32, fine).
- One transcribe pass at a time per SpeechRecognizer instance (guarded by
  `isBusy`).
- Keep the Cerebras path compiling and working.
- `npx tsc --noEmit` + `npm run build` must stay green.
- Do NOT run the browser or commit; the reviewer (Fable) drives Chrome and
  finishes verification. You may run `npx tsc --noEmit` and `npm run build`.
  Leave the working tree dirty.

## New deps

- Part II Feature 2 (vision): `@jax-js/onnx`, `onnx-buf`.
- No other runtime deps (tools use fetch; UI is hand-rendered).

---

# Acceptance checklists

## Part I — core (reviewer verifies in Chrome)

- [ ] Models load; conversation starts with one orb press; mic stays hot.
- [ ] Speaking produces live committed/tentative captions (<1.5 s lag).
- [ ] Turn ends ~0.5–0.8 s after speech stops; first TTS audio ≤ ~1.5 s
      after end of speech (warm).
- [ ] Talking over the assistant stops its audio ≤ ~300 ms, transcript marks
      "[interrupted]", and the user's interruption becomes the next turn.
- [ ] A long user utterance with a mid-pause triggers exactly one backchannel
      and does not split the turn.
- [ ] "Tell me when 15 seconds have passed" → proactive spoken reminder ~15 s
      later without user input.
- [ ] ASR (wasm) demonstrably runs while TTS (webgpu) is speaking (activity
      dots / no audio stutter).
- [ ] Session ends cleanly on second orb press; models stay loaded; a new
      session works.

## Part II — roadmap features

- Feature 1: warm turn latency ≈ LLM-first-token + TTS-first-audio, no
  multi-second ASR finalize on the critical path; live captions still show.
- Feature 2: Eye toggle loads D-FINE; scene state feeds proactive lines and
  `[scene:]` grounding for "what do you see?"; degrades gracefully with no
  camera.
- Feature 3: a tool-triggering turn speaks a holding line, keeps listening
  while the background task runs, then speaks the result and renders a UiCard;
  only one background task at a time.
- Feature 4: Simultaneous mode overlaps ASR (wasm) and TTS (webgpu) across
  sentences; English-TTS translation caveat documented.

---

# Cross-cutting

- Keep the tagline/README honest: everything runs in-browser on jax-js; call out
  what's emulated vs. native (cascade vs. TML's single model), and the
  English-TTS translation caveat.
- Update README with the new modes and a one-line "how close to TML" table.
- Reviewer will verify in Chrome with a fake mic + (for vision) may need to
  point the camera or use a canvas-fed fake video; make vision degrade
  gracefully if no camera.

## Final report (agent → reviewer)

Per-feature: what you built, deviations + why, files touched, what compiles,
what you could NOT finish, and the top things for the reviewer to check in
Chrome (with any device/perf risks, esp. webgpu contention from vision).
