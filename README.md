# jax-realtime

A real-time, full-duplex voice assistant that runs **entirely in your browser**
on WebGPU, built with [jax-js](https://github.com/ekzhang/jax-js).

Every stage — speech → ASR → LLM → TTS → speech, plus optional vision — runs
locally in the tab; nothing is sent to a server.

<img src="docs/assets/orb.png" alt="jax-realtime — the idle orb, ready to talk" width="75%" />

It's inspired by the Thinking Machines
[interaction model](https://thinkingmachines.ai/blog/interaction-models/) and
[GPT-Live](https://openai.com/index/introducing-gpt-live/): the goal is a
conversation that *feels* live — you can interrupt it mid-sentence, pause to
think, and it backchannels while you talk — reproduced as a small-model cascade
that fits in a browser.

| Stage | Model | Runs on |
| --- | --- | --- |
| Ear (ASR) | Whisper base.en (fp16) | WebGPU via jax-js |
| Brain (LLM) | SmolLM2-360M-Instruct (int8 download, fp16 runtime) | WebGPU via jax-js |
| Voice (TTS) | Kyutai Pocket TTS + Mimi codec (fp16) | WebGPU via jax-js |
| Eye (vision) | D-FINE small (COCO-80) | WebGPU via `@jax-js/onnx` |

Everything shares the single WebGPU device. The streaming ASR lane is paused
while the assistant speaks so it doesn't contend with TTS for the GPU; barge-in
is therefore energy-based (see below), and captions resume the moment the
assistant stops.

## Interaction

- **Full-duplex micro-turns** — a ~150 ms tick loop drives a deterministic,
  priority-ordered policy: adaptive **barge-in** (talk over the assistant and
  its audio cuts in ~300 ms; the threshold auto-calibrates to the echo floor of
  each reply), mid-utterance **backchannels**, adaptive **endpointing**, and
  time-awareness timers. A watchdog force-recovers the session if a reply ever
  stalls, so it can't wedge.
- **Phantom-turn guard** — near-silence and ambient swells are rejected from the
  captured PCM (voiced-duration + peak, over an adaptive noise floor) before
  they reach Whisper, and a repetition-degeneracy gate drops decoder loops — so
  the assistant doesn't answer "thank you"s you never said. Snappy one-word
  replies ("what?", "no") still get through.
- **Eye (vision)** — optional and off by default, so its 42 MB model and GPU
  residency stay out of the normal voice path. Enable it with one click when
  you want webcam context. D-FINE runs low-priority object detection (it yields
  the GPU to audio), smooths the person count, and answers
  "what do you see?" / "how many people?" / "tell me about the person"
  directly from the measurements. Proactive interjections (stepped away,
  phone spotted, slouching) are best-effort rule heuristics. The webcam shows
  as a corner PiP with detection boxes.
- **Voice-clean replies** — markdown/bracket tokens are unsampleable at the
  logit level (a voice never needs "**" or "[placeholder]"), and structurally
  garbled or low-confidence transcripts get a deterministic request to repeat
  instead of a confidently wrong answer.
- **Typed conversation memory** — bounded facts the user explicitly states
  (name, trip, pet, favorite, plans, relationships) are retained and injected
  only when relevant; exact recall bypasses small-model guessing.
- **Two-tier tools** — factual asks are delegated so the small on-device model
  isn't left guessing: weather ("what's the weather in Tokyo" → [open-meteo](https://open-meteo.com/),
  in °F/mph), facts ("who is Ada Lovelace" → Wikipedia), plus instant offline
  **calculator** and **clock/date**. Web lookups speak a holding line and fetch
  in the background, then answer on the next silence and render a card; the
  card clears when the conversation moves on.

## Performance (all on jax-js / WebGPU)

The turn-latency floor is set by the single GPU, so the work went into cutting
GPU cost per token/frame rather than overlapping stages (which a single device
can't do — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the full
map-reduce campaign log, including the negative results):

- **Fused decode** — the LLM decode step is fused from dozens of per-layer jit
  dispatches into one, and Pocket TTS from ~11 into two, cutting the
  command-buffer submit overhead that dominated per-step cost (~22% each).
- **GPU top-k sampling** — the LLM samples from a device-side top-64 (one small
  readback) instead of transferring the full vocab logits every token,
  folded into the fused step's single dispatch.
- **Bucket-padded prefill** — jax-js re-traces its jits for every new tensor
  shape, and every conversation turn has a new prompt length; padding the
  prompt to 64-token buckets makes traces repeat, holding LLM first-token
  flat (~250–350 ms) instead of growing past 1 s as history accumulates
  (−30% turn latency on the holdout bench, exactness verified on-device).
- **Smaller download** — the LLM and Whisper ship per-row int8 (363 MB instead
  of 724 MB and 73 MB instead of 144 MB), while the TTS checkpoint omits 35 MB
  of audio-encoder weights never used for synthesis. The quantized artifacts
  are dequantized at load, so runtime kernels stay unchanged.

Runtime behaviour is tunable at `src/tunables.ts` (read live, so A/B
experiments don't need a rebuild).

## Run it

```sh
npm install
npm run dev
```

Open http://localhost:5173 in a WebGPU-capable browser (Chrome/Edge on desktop,
Safari 26+). Click **Load models** (~640 MB on first run — SmolLM 363 +
Pocket TTS 201 + Whisper 73, all cached in OPFS afterwards). The optional Eye is
off by default; enabling it adds the 42 MB D-FINE model and requests camera
access. Then press the orb once and just talk — hands-free: turn ends are
detected by silence, your words stream into the transcript live, the assistant
answers out loud and resumes listening. Press the orb again to end.

> **Smaller model downloads.** By default the app fetches per-row int8 builds of
> SmolLM2-360M and Whisper base.en from
> [Hugging Face](https://huggingface.co/sachink98/jax-realtime-weights), then
> dequantizes them to the normal runtime dtype during load. The LLM measured
> perplexity +0.7%; Whisper produced identical paired MAP and holdout transcripts.
> Either artifact falls back to its full fp16 file if the compact download is
> unreachable.

The orb reacts in real time: it breathes when idle, swells with your voice while
listening, shimmers while the model thinks, and pulses with the synthesized
speech while answering. Per-stage latencies and the active GPU are shown in the
pipeline rail and footer.

## How it works

The pipeline stages, from microphone to speaker:

| Path | What's there |
| --- | --- |
| `src/mic.ts` | 16 kHz PCM capture via an AudioWorklet. |
| `src/asr/` | Whisper encoder/decoder, log-mel features, greedy timestamp decoding. `streaming.ts` transcribes live using LocalAgreement-2: it locks in words once two passes agree, filters out the assistant's own voice, and exposes a best-guess transcript the moment your turn ends. |
| `src/llm/smollm.ts` | SmolLM2-360M (Llama architecture) forward pass with a KV cache — the brain. Each token is generated in a single fused GPU dispatch, and the prompt prefill is bucket-padded so jit traces are reused across turns. Chosen via a blind-judged model shootout against same-size and larger alternatives. |
| `src/memory.ts` | Bounded extraction, relevance filtering, and deterministic recall for facts the user explicitly shared. |
| `src/tts/` | Pocket TTS flow-matching LM + Kyutai's [Mimi](https://github.com/kyutai-labs/moshi) streaming neural codec (reimplemented on jax-js, with the fused per-frame decode) and a streaming `AudioContext` player. |
| `src/vision/` | D-FINE detector on `@jax-js/onnx`, webcam `VisionSession`, COCO labels, box-dedupe and person-count smoothing. |
| `src/tools/tools.ts` | Keyless intent detection → weather / Wikipedia / calc / clock. |

## License

[MIT](LICENSE). Model inference code is adapted from the
[jax-js repository](https://github.com/ekzhang/jax-js/tree/main/website/src/routes)
by Eric Zhang (MIT licensed); model weights remain under their respective
licenses.
