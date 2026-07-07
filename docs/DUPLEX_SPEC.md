# Spec: Full-duplex "interaction model" mode (Thinking Machines-style)

Goal: evolve this app from a turn-based voice assistant into an emulation of
Thinking Machines' interaction models (https://thinkingmachines.ai/blog/interaction-models/)
— continuous micro-turn processing, no hard turn boundaries, barge-in
interruption, backchanneling, time awareness, proactive interjections — all
in-browser with jax-js. We keep the cascaded pipeline (Whisper → Gemma →
Pocket TTS) but restructure the harness into a 150–200 ms tick loop.

Reference source for all jax-js patterns: a clone of ekzhang/jax-js lives at
`/private/tmp/claude-501/-Users-sachinkesiraju-Downloads-jax-realtime/ab484e12-d6f3-4126-829e-4773697d2671/scratchpad/jax-js`
(especially `website/src/routes/{whisper,chat,tts,mobileclip}` and
`src/backend/wasm/parallel.ts`).

## Current state (all working, verified end to end)

- `src/pipeline.ts` — SpeechRecognizer (Whisper tiny.en fp16 WebGPU),
  LocalChatModel (Gemma 3 270M fp16 WebGPU, streaming tokens),
  SpeechSynthesizer (Pocket TTS fp16 WebGPU, streaming playback),
  CerebrasChatModel (optional cloud LLM). Weights cached in OPFS.
- `src/mic.ts` — VoiceCapture: AudioWorklet 16 kHz PCM ring, `level()`,
  pause/resume by dropping samples (never suspend the AudioContext — Chrome
  kills MediaStream feeds after suspend/resume).
- `src/main.ts` — hands-free session: energy VAD, incremental Whisper partials
  every 1.5 s, orb UI (`src/orb.ts`), transcript bubbles, per-stage metrics.
- `npx tsc --noEmit` is clean; `npm run dev` on port 5173.

## Architecture changes

### 1. Two compute lanes (critical)

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

### 2. Streaming ASR (`src/asr/streaming.ts`, new)

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

### 3. Abortable, streaming TTS

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

### 4. LLM

Keep `LocalChatModel.generate(history, onToken)` but expose the raw token
stream as an async iterator (`generateStream`) so main.ts can feed the
sentence splitter → `speakStream` as tokens arrive. Add to the SYSTEM_HINT:
answers must be 1–3 short spoken sentences; if the user message ends with
"[interrupted]", acknowledge briefly and yield. Keep CerebrasChatModel
working (non-streaming is fine there; wrap its full reply as a one-item
stream).

### 5. Micro-turn engine (`src/duplex.ts`, new — the heart)

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
   On end: `finalize()` ASR, push history, start LLM stream → sentence
   splitter → `speakStream`. Record turn latency (end-of-speech →
   first TTS audio) for the metrics rail.
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

### 6. UI (`src/main.ts`, `src/orb.ts`, `src/style.css`)

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

### 7. Vision "Eye" stage — stretch goal, implement LAST

Port MobileCLIP from the clone (`website/src/routes/mobileclip`) into
`src/vision/`. Webcam via getUserMedia video, one frame/second onto WebGPU
(or wasm if webgpu contended), scored against prompt embeddings:
"a person slouching", "a person sitting up straight", "a person waving at
the camera", "an empty chair". If a non-neutral state wins for 3 consecutive
ticks and the user isn't speaking, proactive line via TTS ("Hey — sit up
straight!", "Hello! I see you waving.", "Did you leave? I'll wait.").
Toggle in the dock ("Eye · webcam"), off by default. Camera preview thumb in
the corner like the TML demos. If time or complexity blows up, skip — do not
let this break the core.

## Constraints & gotchas (learned the hard way)

- NEVER `AudioContext.suspend()` on the capture context (breaks the feed).
- `playTTS` disposes the model refs passed in — always pass `tree.ref(model)`.
- jax-js arrays are refcounted: match every `.ref` / `.dispose()`; jit fns
  consume their inputs. Follow the existing patterns exactly.
- Whisper wasm = fp32 (double weight RAM; tiny.en is 74 MB fp16 → ~150 MB
  fp32, fine).
- One transcribe pass at a time per SpeechRecognizer instance (guard flag).
- Keep the Cerebras path compiling and working.
- `npx tsc --noEmit` must stay clean. `npm run build` must succeed.
- Do NOT run browser tests; the reviewer will drive Chrome. You may run
  `npx tsc --noEmit` and `npm run build`.
- Do not commit; leave the working tree dirty.

## Acceptance checklist (reviewer will verify in Chrome)

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
