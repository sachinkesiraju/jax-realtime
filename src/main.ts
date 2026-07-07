import "./style.css";

import { VoiceCapture } from "./mic";
import { analyserLevel, Orb } from "./orb";
import {
  CerebrasChatModel,
  type ChatMessage,
  type DownloadProgress,
  loadPipeline,
  TTS_VOICES,
  type TTSVoice,
  type VoicePipeline,
} from "./pipeline";

type Stage = "asr" | "llm" | "tts";

// Voice-activity detection tuning (levels are VoiceCapture.level() units).
const VAD = {
  startLevel: 0.07,
  stopLevel: 0.04,
  silenceMs: 900,
  minSpeechMs: 500,
  maxUtteranceMs: 28_000,
  partialIntervalMs: 1_500,
};

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <main class="shell">
    <header class="masthead">
      <div>
        <h1>jax&#8209;realtime</h1>
        <p class="tagline">
          A speech&#8209;to&#8209;speech voice assistant running entirely in your browser
          with <a href="https://github.com/ekzhang/jax-js" target="_blank">jax&#8209;js</a> on WebGPU
          &mdash; the same ASR&nbsp;&rarr;&nbsp;LLM&nbsp;&rarr;&nbsp;TTS pipeline as the
          <a href="https://huggingface.co/blog/cerebras-gemma4-voice-ai" target="_blank">HF &times; Cerebras voice AI demo</a>.
        </p>
      </div>
    </header>

    <section class="rail" aria-label="Pipeline stages">
      <div class="stage" id="stage-asr">
        <span class="stage-role">Ear</span>
        <span class="stage-model">Whisper tiny.en</span>
        <span class="stage-metric" id="metric-asr">&ndash;</span>
      </div>
      <span class="rail-arrow">&rarr;</span>
      <div class="stage" id="stage-llm">
        <span class="stage-role">Brain</span>
        <span class="stage-model" id="llm-label">Gemma 3 270M</span>
        <span class="stage-metric" id="metric-llm">&ndash;</span>
      </div>
      <span class="rail-arrow">&rarr;</span>
      <div class="stage" id="stage-tts">
        <span class="stage-role">Voice</span>
        <span class="stage-model">Kyutai Pocket TTS</span>
        <span class="stage-metric" id="metric-tts">&ndash;</span>
      </div>
    </section>

    <section class="console">
      <div class="console-status">
        <span class="status-dot" id="status-dot"></span>
        <span id="status-text">standing by</span>
      </div>

      <div class="orb-area">
        <button id="orb-btn" class="orb-btn" aria-label="Start conversation" disabled>
          <canvas id="orb-canvas"></canvas>
        </button>
        <p class="orb-hint" id="orb-hint">
          Load the models, then press the orb and just talk &mdash; turns are
          detected automatically.<br />
          The first load downloads ~750&nbsp;MB of weights; cached afterwards.
        </p>
      </div>

      <div class="transcript" id="transcript"></div>

      <div class="dock">
        <button id="load-btn" class="load-btn">Load models</button>
        <div class="dock-side">
          <label class="field">
            <span>Voice</span>
            <select id="voice-select"></select>
          </label>
          <details class="field cerebras">
            <summary>LLM backend</summary>
            <div class="cerebras-body">
              <p>Default is Gemma running locally in your browser. Paste a
              Cerebras API key to route the LLM stage through Cerebras inference
              instead, as in the blog post.</p>
              <input id="cerebras-key" type="password" placeholder="Cerebras API key (optional)" />
              <input id="cerebras-model" type="text" value="gemma-4-31b" placeholder="Model name" />
            </div>
          </details>
        </div>
      </div>

      <div class="downloads" id="downloads" hidden></div>
    </section>

    <footer class="colophon">
      <span>Models: <a href="https://huggingface.co/mlx-community/whisper-tiny.en-asr-fp16" target="_blank">Whisper tiny.en</a>,
      <a href="https://huggingface.co/ekzhang/jax-js-models" target="_blank">Gemma 3 270M</a>,
      <a href="https://huggingface.co/kyutai/pocket-tts-without-voice-cloning" target="_blank">Kyutai Pocket TTS</a></span>
      <span id="backend-chip" class="backend-chip">WebGPU</span>
    </footer>
  </main>
`;

const el = {
  loadBtn: document.querySelector<HTMLButtonElement>("#load-btn")!,
  orbBtn: document.querySelector<HTMLButtonElement>("#orb-btn")!,
  orbCanvas: document.querySelector<HTMLCanvasElement>("#orb-canvas")!,
  orbHint: document.querySelector<HTMLParagraphElement>("#orb-hint")!,
  statusDot: document.querySelector<HTMLSpanElement>("#status-dot")!,
  statusText: document.querySelector<HTMLSpanElement>("#status-text")!,
  transcript: document.querySelector<HTMLDivElement>("#transcript")!,
  downloads: document.querySelector<HTMLDivElement>("#downloads")!,
  voiceSelect: document.querySelector<HTMLSelectElement>("#voice-select")!,
  cerebrasKey: document.querySelector<HTMLInputElement>("#cerebras-key")!,
  cerebrasModel: document.querySelector<HTMLInputElement>("#cerebras-model")!,
  llmLabel: document.querySelector<HTMLSpanElement>("#llm-label")!,
  metrics: {
    asr: document.querySelector<HTMLSpanElement>("#metric-asr")!,
    llm: document.querySelector<HTMLSpanElement>("#metric-llm")!,
    tts: document.querySelector<HTMLSpanElement>("#metric-tts")!,
  },
  stages: {
    asr: document.querySelector<HTMLDivElement>("#stage-asr")!,
    llm: document.querySelector<HTMLDivElement>("#stage-llm")!,
    tts: document.querySelector<HTMLDivElement>("#stage-tts")!,
  },
};

for (const voice of TTS_VOICES) {
  const option = document.createElement("option");
  option.value = voice;
  option.textContent = voice.charAt(0).toUpperCase() + voice.slice(1);
  el.voiceSelect.appendChild(option);
}
el.voiceSelect.value = "azelma";

const orb = new Orb(el.orbCanvas);
const capture = new VoiceCapture();
if (import.meta.env.DEV) {
  (window as unknown as Record<string, unknown>).__capture = capture;
}
let pipeline: VoicePipeline | null = null;
let session = false;
let sessionEnding = false;
const history: ChatMessage[] = [];

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(text: string, mode: "idle" | "live" | "busy" | "error" = "idle") {
  el.statusText.textContent = text;
  el.statusDot.dataset.mode = mode;
}

function setHint(text: string) {
  el.orbHint.innerHTML = text;
}

function setActiveStage(stage: Stage | null) {
  for (const [name, node] of Object.entries(el.stages)) {
    node.classList.toggle("is-active", name === stage);
  }
}

function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function addBubble(role: "user" | "assistant"): HTMLDivElement {
  const bubble = document.createElement("div");
  bubble.className = `bubble bubble-${role}`;
  el.transcript.appendChild(bubble);
  el.transcript.scrollTop = el.transcript.scrollHeight;
  return bubble;
}

const downloadRows = new Map<string, HTMLDivElement>();

function onDownloadProgress(progress: DownloadProgress) {
  el.downloads.hidden = false;
  let row = downloadRows.get(progress.name);
  if (!row) {
    row = document.createElement("div");
    row.className = "download-row";
    row.innerHTML = `
      <span class="download-name"></span>
      <span class="download-size"></span>
      <div class="download-bar"><div class="download-fill"></div></div>
    `;
    row.querySelector(".download-name")!.textContent = progress.name;
    el.downloads.appendChild(row);
    downloadRows.set(progress.name, row);
  }
  const fill = row.querySelector<HTMLDivElement>(".download-fill")!;
  const size = row.querySelector<HTMLSpanElement>(".download-size")!;
  const mb = (progress.loadedBytes / 1e6).toFixed(0);
  if (progress.done) {
    fill.style.width = "100%";
    row.classList.add("is-done");
    size.textContent = `${mb} MB`;
  } else if (progress.totalBytes) {
    fill.style.width = `${(100 * progress.loadedBytes) / progress.totalBytes}%`;
    size.textContent = `${mb} / ${(progress.totalBytes / 1e6).toFixed(0)} MB`;
  } else {
    size.textContent = `${mb} MB`;
  }
}

async function handleLoad() {
  el.loadBtn.disabled = true;
  setStatus("downloading models", "busy");
  try {
    pipeline = await loadPipeline(onDownloadProgress);
    el.loadBtn.hidden = true;
    el.orbBtn.disabled = false;
    orb.setState("idle");
    setStatus("ready", "idle");
    setHint("Press the orb and just talk &mdash; turns are detected automatically.");
    setTimeout(() => {
      el.downloads.hidden = true;
    }, 1500);
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), "error");
    el.loadBtn.disabled = false;
  }
}

type Utterance = {
  samples: Float32Array;
  duration: number;
  liveBubble: HTMLDivElement | null;
};

/**
 * Wait for one utterance: speech onset, then trailing silence. While the user
 * is talking, run incremental Whisper passes to stream a live transcript.
 */
async function listenForUtterance(): Promise<Utterance | null> {
  capture.clear();
  await capture.resume();
  orb.setState("listening", () => capture.level());
  setStatus("listening — just talk", "live");

  let speechStart = 0;
  let silenceStart = 0;
  let lastPartial = 0;
  let partialBusy = false;
  let partialChain: Promise<void> = Promise.resolve();
  const live: { bubble: HTMLDivElement | null } = { bubble: null };

  while (session) {
    await sleep(100);
    const level = capture.level();
    const now = performance.now();

    if (!speechStart) {
      if (level > VAD.startLevel) speechStart = now;
      continue;
    }

    if (level < VAD.stopLevel) {
      if (!silenceStart) silenceStart = now;
    } else {
      silenceStart = 0;
    }

    const speechMs = now - speechStart;
    const trailingSilence = silenceStart ? now - silenceStart : 0;
    if (
      (trailingSilence > VAD.silenceMs && speechMs > VAD.minSpeechMs) ||
      speechMs > VAD.maxUtteranceMs
    ) {
      break;
    }

    // Live transcript: incremental Whisper pass on the buffer so far.
    if (
      pipeline &&
      !partialBusy &&
      now - lastPartial > VAD.partialIntervalMs &&
      speechMs > 600
    ) {
      partialBusy = true;
      lastPartial = now;
      const snapshot = capture.samples();
      partialChain = pipeline.asr
        .transcribe(snapshot, snapshot.length / 16_000)
        .then((text) => {
          if (text && session) {
            live.bubble ??= addBubble("user");
            live.bubble.classList.add("is-live");
            live.bubble.textContent = text;
            el.transcript.scrollTop = el.transcript.scrollHeight;
          }
        })
        .catch((error) => console.warn("partial ASR failed", error))
        .finally(() => {
          partialBusy = false;
        });
    }
  }

  await partialChain;
  if (!session) {
    live.bubble?.remove();
    return null;
  }
  await capture.pause();
  return {
    samples: capture.samples(),
    duration: capture.durationSeconds(),
    liveBubble: live.bubble,
  };
}

async function processTurn(utterance: Utterance): Promise<void> {
  if (!pipeline) return;
  orb.setState("thinking");

  // 1. ASR (final pass over the full utterance)
  setActiveStage("asr");
  setStatus("transcribing", "busy");
  const asrStart = performance.now();
  const text = await pipeline.asr.transcribe(
    utterance.samples,
    utterance.duration,
  );
  el.metrics.asr.textContent = formatMs(performance.now() - asrStart);

  if (!text) {
    utterance.liveBubble?.remove();
    setActiveStage(null);
    return;
  }
  const userBubble = utterance.liveBubble ?? addBubble("user");
  userBubble.classList.remove("is-live");
  userBubble.textContent = text;
  history.push({ role: "user", content: text });

  // 2. LLM
  setActiveStage("llm");
  setStatus("thinking", "busy");
  setHint("Thinking&hellip;");
  const bubble = addBubble("assistant");
  const cerebrasKey = el.cerebrasKey.value.trim();
  const llm = cerebrasKey
    ? new CerebrasChatModel(cerebrasKey, el.cerebrasModel.value.trim())
    : pipeline.llm;
  el.llmLabel.textContent = cerebrasKey
    ? `Cerebras · ${el.cerebrasModel.value.trim()}`
    : "Gemma 3 270M";
  const { text: reply, stats } = await llm.generate(history, (partial) => {
    bubble.textContent = partial;
    el.transcript.scrollTop = el.transcript.scrollHeight;
  });
  el.metrics.llm.textContent = `${formatMs(stats.totalMs)} · ${stats.newTokens} tok`;

  if (!reply) {
    bubble.textContent = "(no reply)";
    setActiveStage(null);
    return;
  }
  bubble.textContent = reply;
  history.push({ role: "assistant", content: reply });

  // 3. TTS (mic is paused, so the assistant doesn't hear itself)
  setActiveStage("tts");
  setStatus("speaking", "live");
  setHint("Speaking&hellip; the orb keeps listening after.");
  const speakable = reply.replace(/[*_`#>|]/g, "").replace(/\s+/g, " ");
  const ttsStats = await pipeline.tts.speak(
    el.voiceSelect.value as TTSVoice,
    speakable,
    (analyser) => {
      const buffer = new Float32Array(analyser.fftSize);
      orb.setState("speaking", () => analyserLevel(analyser, buffer));
    },
  );
  el.metrics.tts.textContent = `${formatMs(ttsStats.firstAudioMs)} to audio`;
  setActiveStage(null);
  setHint("Just talk. Press the orb again to end the conversation.");
}

async function conversationLoop() {
  try {
    await capture.start();
    el.orbBtn.classList.add("is-recording");
    el.orbBtn.setAttribute("aria-label", "End conversation");
    setHint("Just talk. Press the orb again to end the conversation.");

    while (session) {
      const utterance = await listenForUtterance();
      if (!utterance || !session) break;
      await processTurn(utterance);
    }
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    session = false;
    sessionEnding = false;
    await capture.close();
    el.orbBtn.classList.remove("is-recording");
    el.orbBtn.setAttribute("aria-label", "Start conversation");
    el.orbBtn.disabled = false;
    setActiveStage(null);
    orb.setState("idle");
    if (el.statusDot.dataset.mode !== "error") setStatus("ready", "idle");
    setHint("Press the orb and just talk &mdash; turns are detected automatically.");
  }
}

function handleOrb() {
  if (!pipeline || sessionEnding) return;
  if (session) {
    // End the conversation; the loop notices and cleans up.
    session = false;
    sessionEnding = true;
    el.orbBtn.disabled = true;
    setStatus("ending conversation", "busy");
    return;
  }
  session = true;
  void conversationLoop();
}

el.loadBtn.addEventListener("click", () => void handleLoad());
el.orbBtn.addEventListener("click", handleOrb);
