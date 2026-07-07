# jax-realtime

A real-time speech-to-speech voice assistant that runs **entirely in your
browser** on WebGPU, built with [jax-js](https://github.com/ekzhang/jax-js).
It recreates the architecture of the
[Hugging Face × Cerebras voice AI demo](https://huggingface.co/blog/cerebras-gemma4-voice-ai)
— speech → ASR → LLM → TTS → speech — but with every stage running locally:

| Stage | Model | Runs on |
| --- | --- | --- |
| Ear (ASR) | Whisper tiny.en (fp16) | WebGPU via jax-js |
| Brain (LLM) | Gemma 3 270M instruction-tuned (fp16) | WebGPU via jax-js |
| Voice (TTS) | Kyutai Pocket TTS + Mimi codec (fp16) | WebGPU via jax-js |

Optionally, the LLM stage can be routed through the Cerebras cloud API
(paste an API key under "LLM backend"), matching the original blog setup.

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

Not replicated from the Space demo: the server-side agent tools
(`web_search`, `camera_snapshot`) — those need a larger tool-calling,
vision-capable model than the 270M Gemma that fits in a browser tab.

Measured on an Apple-silicon MacBook after warmup: ~0.5 s ASR, ~0.5–1.5 s
LLM, ~0.8 s to first TTS audio — roughly 2 s from end of speech to the
start of the spoken reply.

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
- `src/main.ts` — UI and conversation loop.

Model inference code is adapted from the demos in the
[jax-js repository](https://github.com/ekzhang/jax-js/tree/main/website/src/routes)
by Eric Zhang (MIT licensed).
