# jax-realtime

A real-time speech-to-speech voice assistant that runs **entirely in your
browser** on WebGPU, built with [jax-js](https://github.com/ekzhang/jax-js).
It recreates the architecture of the
[Hugging Face × Cerebras voice AI demo](https://huggingface.co/blog/cerebras-gemma4-voice-ai)
— speech → ASR → LLM → TTS → speech — but with every stage running locally:

| Stage | Model | Runs on |
| --- | --- | --- |
| Ear (ASR) | Whisper tiny.en (fp32) | **wasm** lane via jax-js (fp16 on WebGPU if wasm is unavailable) |
| Brain (LLM) | Gemma 3 270M instruction-tuned (fp16) | WebGPU via jax-js |
| Voice (TTS) | Kyutai Pocket TTS + Mimi codec (fp16) | WebGPU via jax-js |
| Eye (vision, optional) | D-FINE small (COCO-80) | WebGPU via `@jax-js/onnx` |

ASR runs on the **wasm** compute lane so Whisper passes keep transcribing while
TTS/LLM occupy WebGPU — the two lanes run concurrently (that's what makes
barge-in and live captions work while the assistant is talking).

Optionally, the LLM stage can be routed through the Cerebras cloud API
(paste an API key under "LLM backend"), matching the original blog setup.

## Interaction-model modes

Beyond the basic cascade, the app emulates several behaviours from Thinking
Machines' [interaction models](https://thinkingmachines.ai/blog/interaction-models/)
demo — everything still fully in-browser:

- **Full-duplex micro-turns** — a ~150 ms tick loop with barge-in (talk over
  the assistant and its audio stops in ~200 ms), mid-utterance backchannels,
  adaptive endpointing, and time-awareness timers.
- **Low-latency turns** — the turn-end path reuses the streaming transcript
  instead of running a second multi-second Whisper "finalize" pass, so latency
  is roughly LLM-first-token + TTS-first-audio.
- **Eye (vision)** — toggle *Eye · webcam* to load D-FINE and run object
  detection at ~1 fps. The assistant makes proactive interjections (you stepped
  away, phone spotted, slouching — best-effort/approximate) and grounds
  "what do you see?" / "how many people?" with a `[scene: …]` tag. The webcam
  frame shows as a corner PiP with detection boxes.
- **Two-tier tools** — ask for the weather or a fact ("what's the weather in
  Tokyo", "who is Ada Lovelace") and the fast local model speaks a holding line
  and stays present while a background task fetches
  [open-meteo](https://open-meteo.com/) / Wikipedia (both keyless + CORS-open);
  the result is spoken on the next silence and rendered as a generative-UI card.
- **Simultaneous mode** — toggle *Simul* to live-translate each committed
  sentence while you keep talking (ASR on wasm + TTS on WebGPU overlap).

### Honesty / how close to TML

| TML capability | Here |
| --- | --- |
| Single end-to-end interaction model | Emulated with a Whisper→Gemma→TTS **cascade** on two lanes |
| Proactive vision (SLOUCHING/DANGER) | D-FINE object detection + coarse rule table (posture is an approximate box heuristic, not pose estimation) |
| Async background tool use + generative UI | Real: keyless weather + Wikipedia, background task, card render |
| Simultaneous speech / translation | Demonstrates the *simultaneity* (speak while listening). **Caveat:** Kyutai Pocket TTS is English-only, so a non-English translation is spoken by an English voice reading foreign text — the point is the overlap, not TTS quality. A Cerebras key gives much better translations. |

## Run it

```sh
npm install
npm run dev
```

Open http://localhost:5173 in a WebGPU-capable browser (Chrome/Edge on
desktop, Safari 26+). Click **Load models** (~750 MB on first run, cached in
OPFS afterwards), then press the orb once and just talk — the conversation is
hands-free, like the [HF Space demo](https://huggingface.co/spaces/amir-tfrere/minimal-conversation-app-s2s-backend-websocket):
turn ends are detected by silence (energy-based VAD), your words stream into
the transcript live while you speak (incremental Whisper passes), and the
assistant answers out loud, then resumes listening. Press the orb again to
end the conversation.

The orb reacts in real time: it breathes when idle, swells with your voice
while listening (red), shimmers while the model thinks (amber), and pulses
with the synthesized speech while answering (lime). Per-stage latencies are
shown in the pipeline rail at the top.

The server-side agent tools from the Space demo are approximated locally: a
keyless `web_search`/weather background task (Wikipedia + open-meteo) and a
`camera_snapshot`-style vision stage (D-FINE) — see *Interaction-model modes*
above. These are lighter than a large tool-calling, vision-capable model, but
they run entirely in the tab.

Measured on an Apple-silicon MacBook after warmup: with the streaming-transcript
turn-end path, latency is roughly LLM-first-token + TTS-first-audio (~1 s to
first spoken audio) rather than the old ~2 s that included a separate Whisper
finalize pass.

## How it works

- `src/asr/` — Whisper encoder/decoder, log-mel features, greedy timestamp
  decoding (ported from the jax-js website demo).
- `src/llm/gemma.ts` — Gemma 3 forward pass with KV cache.
- `src/tts/` — Pocket TTS flow-matching LM + Mimi streaming decoder, with a
  streaming `AudioContext` player.
- `src/pipeline.ts` — loads weights from Hugging Face (cached via OPFS) and
  orchestrates the three stages; also contains the optional Cerebras client.
- `src/mic.ts` — 16 kHz PCM capture via AudioWorklet, used for VAD and
  incremental transcription (`src/main.ts` holds the hands-free loop).
- `src/orb.ts` — the audio-reactive orb (canvas 2D, driven by mic/TTS
  `AnalyserNode` levels).
- `src/asr/streaming.ts` — LocalAgreement-2 streaming transcriber (committed +
  tentative text, self-echo filter, `bestText()` for the low-latency turn end).
- `src/duplex.ts` — the full-duplex micro-turn engine (barge-in, backchannels,
  timers, vision interjections, two-tier tool tasks, simultaneous mode).
- `src/vision/` — D-FINE detector on `@jax-js/onnx`, webcam VisionSession, COCO.
- `src/tools/tools.ts` — keyless intent detection + weather/Wikipedia fetches.
- `src/main.ts` — UI and wiring.

Model inference code is adapted from the demos in the
[jax-js repository](https://github.com/ekzhang/jax-js/tree/main/website/src/routes)
by Eric Zhang (MIT licensed).
