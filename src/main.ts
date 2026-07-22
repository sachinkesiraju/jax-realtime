import "./float16-polyfill";
import "@fontsource-variable/instrument-sans";
import "@fontsource/jetbrains-mono/latin-400.css";
import "@fontsource/jetbrains-mono/latin-600.css";
import "./style.css";

import { DuplexSession, isGarbledTranscript } from "./duplex";
import {
  type ConversationalMemory,
  injectMemoryTag,
  relevantMemoryFacts,
  rememberUserFacts,
} from "./memory";
import { VoiceCapture } from "./mic";
import { TUNABLES, TURN_LOG } from "./tunables";
import { Orb } from "./orb";
import { detectTool, type ToolKind, type UiCard } from "./tools/tools";
import { ObjectDetector } from "./vision/detector";
import { VisionSession } from "./vision/vision";
import {
  type ChatModel,
  type DownloadProgress,
  loadPipeline,
  TTS_VOICES,
  type TTSVoice,
  type VoicePipeline,
} from "./pipeline";

type Stage = "asr" | "llm" | "tts";

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <main class="shell">
    <header class="masthead">
      <div>
        <h1>jax&#8209;realtime</h1>
        <p class="tagline">
          A full&#8209;duplex voice assistant running entirely in your browser with
          <a href="https://github.com/ekzhang/jax-js" target="_blank">jax&#8209;js</a>.
          <br />
          It listens while it talks, you can interrupt it mid&#8209;sentence,
          and it backchannels while you speak.
          <br />
          Inspired by the
          <a href="https://thinkingmachines.ai/blog/interaction-models/" target="_blank">Thinking Machines interaction model</a>
          and <a href="https://openai.com/index/introducing-gpt-live/" target="_blank">GPT&#8209;Live</a>.
        </p>
      </div>
    </header>

    <section class="rail" aria-label="Pipeline stages">
      <div class="stage" id="stage-asr">
        <span class="stage-role"><span class="stage-dot" id="dot-asr"></span>Ear <span class="stage-lane" id="lane-asr">webgpu</span></span>
        <span class="stage-model"><a href="https://huggingface.co/mlx-community/whisper-tiny.en-asr-fp16" target="_blank">Whisper tiny.en</a></span>
        <span class="stage-metric" id="metric-asr">&ndash;</span>
      </div>
      <span class="rail-arrow">+</span>
      <div class="stage" id="stage-llm">
        <span class="stage-role"><span class="stage-dot" id="dot-llm"></span>Brain <span class="stage-lane">webgpu</span></span>
        <span class="stage-model" id="llm-label"><a href="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct" target="_blank">SmolLM2 360M</a></span>
        <span class="stage-metric" id="metric-llm">&ndash;</span>
      </div>
      <span class="rail-arrow">+</span>
      <div class="stage" id="stage-tts">
        <span class="stage-role"><span class="stage-dot" id="dot-tts"></span>Voice <span class="stage-lane">webgpu</span></span>
        <span class="stage-model"><a href="https://huggingface.co/kyutai/pocket-tts-without-voice-cloning" target="_blank">Kyutai Pocket TTS</a></span>
        <span class="stage-metric" id="metric-tts">&ndash;</span>
      </div>
      <span class="rail-arrow rail-arrow-eye">+</span>
      <div class="stage stage-eye" id="stage-eye">
        <span class="stage-role"><span class="stage-dot" id="dot-eye"></span>Eye <span class="stage-lane">webgpu</span></span>
        <span class="stage-model"><a href="https://huggingface.co/bukuroo/D-FINE-ONNX" target="_blank">D&#8209;FINE</a></span>
        <span class="stage-metric" id="metric-eye">off</span>
      </div>
    </section>

    <section class="console">
      <div class="console-status">
        <span class="status-dot" id="status-dot"></span>
        <span id="status-text">standing by</span>
        <span class="bg-pill" id="bg-pill" hidden>background</span>
      </div>

      <div class="pip" id="pip" hidden>
        <canvas id="pip-overlay"></canvas>
        <span class="pip-label">eye · D-FINE</span>
      </div>

      <div class="orb-area">
        <button id="orb-btn" class="orb-btn" aria-label="Start conversation" disabled>
          <canvas id="orb-canvas"></canvas>
        </button>
        <p class="captions" id="captions" hidden>
          <span id="cap-committed"></span><span id="cap-tentative" class="cap-tentative"></span>
        </p>
        <p class="orb-hint" id="orb-hint">
          Load the models, then press the orb once and just talk &mdash; no
          buttons between turns. Talk over it to interrupt.<br />
          The first load downloads ~1.0&nbsp;GB of weights; cached afterwards.
        </p>
        <p class="ticker" id="ticker"></p>
      </div>

      <div class="transcript" id="transcript"></div>

      <div class="canvas-panel" id="canvas-panel" hidden></div>

      <div class="dock">
        <button id="load-btn" class="load-btn">Load models</button>
        <div class="dock-side">
          <span id="backend-chip" class="backend-chip">WebGPU</span>
          <label class="field eye-toggle" title="Webcam object detection (D-FINE). On by default.">
            <!-- The toggle is usable before "Load models" because D-FINE is
                 loaded lazily. Enabling it opts into the separate 42 MB model;
                 leaving it off avoids that download and GPU residency. -->
            <input type="checkbox" id="eye-toggle" checked />
            <span>Eye &middot; webcam</span>
          </label>
          <label class="field">
            <span>Voice</span>
            <select id="voice-select"></select>
          </label>
        </div>
      </div>

      <div class="downloads" id="downloads" hidden></div>
    </section>

    <footer class="colophon">
      <span>A project by <a href="https://sachinkesiraju.com" target="_blank">Sachin Kesiraju</a></span>
    </footer>
  </main>
`;

const el = {
  loadBtn: document.querySelector<HTMLButtonElement>("#load-btn")!,
  orbBtn: document.querySelector<HTMLButtonElement>("#orb-btn")!,
  orbCanvas: document.querySelector<HTMLCanvasElement>("#orb-canvas")!,
  orbHint: document.querySelector<HTMLParagraphElement>("#orb-hint")!,
  captions: document.querySelector<HTMLParagraphElement>("#captions")!,
  capCommitted: document.querySelector<HTMLSpanElement>("#cap-committed")!,
  capTentative: document.querySelector<HTMLSpanElement>("#cap-tentative")!,
  ticker: document.querySelector<HTMLParagraphElement>("#ticker")!,
  statusDot: document.querySelector<HTMLSpanElement>("#status-dot")!,
  statusText: document.querySelector<HTMLSpanElement>("#status-text")!,
  transcript: document.querySelector<HTMLDivElement>("#transcript")!,
  downloads: document.querySelector<HTMLDivElement>("#downloads")!,
  voiceSelect: document.querySelector<HTMLSelectElement>("#voice-select")!,
  llmLabel: document.querySelector<HTMLSpanElement>("#llm-label")!,
  backendChip: document.querySelector<HTMLSpanElement>("#backend-chip")!,
  laneAsr: document.querySelector<HTMLSpanElement>("#lane-asr")!,
  eyeToggle: document.querySelector<HTMLInputElement>("#eye-toggle")!,
  stageEye: document.querySelector<HTMLDivElement>("#stage-eye")!,
  metricEye: document.querySelector<HTMLSpanElement>("#metric-eye")!,
  dotEye: document.querySelector<HTMLSpanElement>("#dot-eye")!,
  pip: document.querySelector<HTMLDivElement>("#pip")!,
  pipOverlay: document.querySelector<HTMLCanvasElement>("#pip-overlay")!,
  bgPill: document.querySelector<HTMLSpanElement>("#bg-pill")!,
  canvasPanel: document.querySelector<HTMLDivElement>("#canvas-panel")!,
  metrics: {
    asr: document.querySelector<HTMLSpanElement>("#metric-asr")!,
    llm: document.querySelector<HTMLSpanElement>("#metric-llm")!,
    tts: document.querySelector<HTMLSpanElement>("#metric-tts")!,
  },
  dots: {
    asr: document.querySelector<HTMLSpanElement>("#dot-asr")!,
    llm: document.querySelector<HTMLSpanElement>("#dot-llm")!,
    tts: document.querySelector<HTMLSpanElement>("#dot-tts")!,
  },
  stages: {
    asr: document.querySelector<HTMLDivElement>("#stage-asr")!,
    llm: document.querySelector<HTMLDivElement>("#stage-llm")!,
    tts: document.querySelector<HTMLDivElement>("#stage-tts")!,
    eye: document.querySelector<HTMLDivElement>("#stage-eye")!,
  },
};

// The per-card lanes already say "webgpu"; make the footer chip earn its place
// by naming the actual GPU. Set at page init (not load time) so it's always
// accurate; Chrome populates adapter-level info, not GPUDevice.adapterInfo.
void (async () => {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    const info = adapter?.info;
    const gpu =
      info?.description ||
      [info?.vendor, info?.architecture].filter(Boolean).join(" ");
    if (gpu) el.backendChip.textContent = `WebGPU · ${gpu}`;
  } catch {
    // Leave the static "WebGPU" label.
  }
})();

for (const voice of TTS_VOICES) {
  const option = document.createElement("option");
  option.value = voice;
  option.textContent = voice.charAt(0).toUpperCase() + voice.slice(1);
  el.voiceSelect.appendChild(option);
}
el.voiceSelect.value = "azelma";

const orb = new Orb(el.orbCanvas);
const capture = new VoiceCapture();
let pipeline: VoicePipeline | null = null;
let duplex: DuplexSession | null = null;
let sessionActive = false;
let toggling = false;
let assistantBubble: HTMLDivElement | null = null;
let toolChip: HTMLDivElement | null = null;

// Vision "Eye" stage (D-FINE). Loaded lazily when the dock toggle is enabled.
let detector: ObjectDetector | null = null;
let detectorPromise: Promise<ObjectDetector> | null = null;
let vision: VisionSession | null = null;
let visionRaf: number | null = null;
let visionBusy = false;

if (import.meta.env.DEV) {
  const dev = window as unknown as Record<string, unknown>;
  dev.__capture = capture;
  dev.__getDuplex = () => duplex;
  dev.__vision = () => vision;
  dev.__detector = () => detector;
  dev.__tunables = TUNABLES;
  dev.__turnLog = TURN_LOG;
  dev.__pipeline = () => pipeline;
  dev.__detectTool = detectTool;
  dev.__garbleProbe = isGarbledTranscript;
  dev.__memoryProbe = (turns: string[], query: string) => {
    let memory: ConversationalMemory = {};
    turns.forEach((turn, index) => {
      memory = rememberUserFacts(memory, turn, index + 1);
    });
    const facts = relevantMemoryFacts(memory, query, turns.length + 1);
    return { prompt: injectMemoryTag(query, facts) };
  };
}

function setStatus(text: string, mode: "idle" | "live" | "busy" | "error" = "idle") {
  el.statusText.textContent = text;
  el.statusDot.dataset.mode = mode;
}

function setHint(text: string) {
  el.orbHint.innerHTML = text;
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

function tick(text: string) {
  el.ticker.textContent = text;
}

// A monospace chip in the transcript stream, shown while a background tool task
// runs and resolved to "done" when the result lands (like the TML web_search chip).
function addToolChip(kind: ToolKind, query: string): HTMLDivElement {
  const chip = document.createElement("div");
  chip.className = "tool-chip";
  // Label the chip by the actual tool: the instant offline tools (calc/clock)
  // aren't web searches, so don't mislabel them as one.
  const verb =
    kind === "lookup" ? "web_search" : kind === "weather" ? "weather" : kind;
  const dot = document.createElement("span");
  dot.className = "chip-dot";
  const label = document.createElement("span");
  label.textContent = `${verb} · ${query}`;
  chip.append(dot, label);
  el.transcript.appendChild(chip);
  el.transcript.scrollTop = el.transcript.scrollHeight;
  return chip;
}

// Generative UI: render the latest tool result as a data card in the canvas panel.
function renderCard(card: UiCard) {
  const panel = el.canvasPanel;
  panel.hidden = false;
  panel.replaceChildren();

  const make = (cls: string, text?: string): HTMLDivElement => {
    const d = document.createElement("div");
    d.className = cls;
    if (text !== undefined) d.textContent = text;
    return d;
  };

  if (card.kind === "weather") {
    panel.classList.add("is-weather");
    const head = make("card-head", card.location);
    const big = make("card-temp");
    big.textContent = `${card.emoji} ${card.temperature}°`;
    const cond = make("card-sub", card.condition);
    const wind = make("card-meta", `wind ${card.wind}`);
    panel.append(head, big, cond, wind);
  } else if (card.kind === "factcard") {
    panel.classList.remove("is-weather");
    const head = make("card-head", card.title);
    const body = make("card-body", card.extract);
    const src = make("card-meta", `via ${card.source}`);
    panel.append(head, body, src);
  } else {
    panel.classList.remove("is-weather");
    const head = make("card-head", card.title);
    panel.append(head);
    const ul = document.createElement("ul");
    ul.className = "card-list";
    for (const item of card.items) {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    }
    panel.append(ul);
  }
}

function currentModel(): ChatModel {
  // Everything runs locally on jax-js; the Brain is always the local SmolLM.
  return pipeline!.llm;
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
    // Respect a pre-load Eye opt-in after WebGPU initialization.
    if (el.eyeToggle.checked) void toggleVision(true);
    el.laneAsr.textContent = pipeline.asrDevice;
    setStatus("preparing backchannels", "busy");
    await pipeline.tts.prepareBackchannels(el.voiceSelect.value as TTSVoice);
    // Pre-trace the flow-LM step-0 prefill for common sentence lengths so the
    // first reply doesn't pay the on-turn JIT re-trace (gated on
    // TUNABLES.ttsWarmup; no-op when off). No audio is produced.
    setStatus("warming up TTS", "busy");
    await pipeline.tts.warmup(el.voiceSelect.value as TTSVoice);
    el.loadBtn.hidden = true;
    el.orbBtn.disabled = false;
    orb.setState("idle");
    setStatus("ready", "idle");
    setHint(
      "Press the orb once and just talk &mdash; interrupt it, pause, ask it to " +
        "time things. Press again to end.",
    );
    setTimeout(() => {
      el.downloads.hidden = true;
    }, 1500);
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), "error");
    el.loadBtn.disabled = false;
  }
}

el.voiceSelect.addEventListener("change", () => {
  // Re-synthesize backchannel clips + re-warm the flow-LM prefill in the new
  // voice (background, best-effort).
  void pipeline?.tts
    .prepareBackchannels(el.voiceSelect.value as TTSVoice)
    .then(() => pipeline?.tts.warmup(el.voiceSelect.value as TTSVoice))
    .catch(() => {});
});

function stageActivity(stage: Stage, active: boolean) {
  el.dots[stage].classList.toggle("is-on", active);
  el.stages[stage].classList.toggle("is-active", active);
}

// --- Vision "Eye" stage -------------------------------------------------

function drawPreview() {
  visionRaf = requestAnimationFrame(drawPreview);
  if (!vision) return;
  // Scene metrics first — they must not depend on the pip canvas being laid
  // out (the early returns below skip frames whenever it isn't, which used to
  // leave the person count frozen at "no people").
  const people = vision.personCount;
  el.metricEye.textContent =
    people === 0 ? "no people" : `${people} ${people === 1 ? "person" : "people"}`;
  el.dotEye.classList.toggle("is-on", vision.active);
  el.stages.eye.classList.toggle("is-active", vision.active);
  const video = vision.video;
  const overlay = el.pipOverlay;
  const cssSize = overlay.clientWidth;
  if (!cssSize || !video.videoWidth) return;
  const ratio = window.devicePixelRatio || 1;
  const px = Math.floor(cssSize * ratio);
  if (overlay.width !== px || overlay.height !== px) {
    overlay.width = px;
    overlay.height = px;
  }
  const ctx = overlay.getContext("2d")!;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssSize, cssSize);

  // The pip shows the center square (object-fit: cover on a square box), which
  // is exactly the region the detector center-crops, so detection boxes (in
  // source pixels) map linearly onto it.
  const crop = Math.min(video.videoWidth, video.videoHeight);
  const sx = (video.videoWidth - crop) / 2;
  const sy = (video.videoHeight - crop) / 2;
  const scale = cssSize / crop;

  ctx.lineWidth = 2;
  ctx.font = "600 10px " + getComputedStyle(document.body).fontFamily;
  for (const det of vision.latest) {
    const x = (det.box[0] - sx) * scale;
    const y = (det.box[1] - sy) * scale;
    const w = det.box[2] * scale;
    const h = det.box[3] * scale;
    ctx.strokeStyle = "oklch(90% 0.19 118)";
    ctx.strokeRect(x, y, w, h);
    const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
    ctx.fillStyle = "oklch(90% 0.19 118 / 0.85)";
    const tw = ctx.measureText(label).width + 6;
    ctx.fillRect(x, Math.max(0, y - 13), tw, 13);
    ctx.fillStyle = "oklch(20% 0.05 118)";
    ctx.fillText(label, x + 3, Math.max(10, y - 3));
  }
}

async function enableVision(): Promise<void> {
  if (!pipeline) return;
  setStatus("loading D-FINE detector", "busy");
  if (!detector) {
    // Camera permission FIRST, download second: if the user denies the camera
    // there can be no Eye session, so the 42 MB D-FINE download + its GPU
    // residency should never happen. The probe stream is stopped immediately;
    // VisionSession.start() re-acquires its own stream in a moment (the
    // permission is granted by then, so no second prompt). A denial throws
    // here — before any bytes are fetched — and toggleVision's catch unchecks
    // the toggle. Only the first enable pays this probe: once `detector` is
    // cached, vision.start() below is the sole camera acquisition, as before.
    const probe = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
    });
    for (const track of probe.getTracks()) track.stop();
    // Sole D-FINE load path (handleLoad's unconditional preload was removed —
    // see the M2 note there): the first enable, including the auto-enable
    // right after load, downloads + JIT-warms the detector here with the same
    // progress rows. Cached across toggles, so a mid-session re-enable reuses
    // the warm detector instantly.
    detectorPromise ??= ObjectDetector.load(onDownloadProgress).then(
      async (d) => {
        await d.warmup();
        return d;
      },
    );
    try {
      detector = await detectorPromise;
    } catch (error) {
      // A rejected load must NOT stay cached, or every later toggle awaits
      // the same dead promise and the Eye can never recover. Clear it so a
      // subsequent enable retries the load from scratch.
      detectorPromise = null;
      throw error;
    }
    setTimeout(() => (el.downloads.hidden = true), 1500);
  }
  vision = new VisionSession(detector);
  // Vision is the lowest-priority stage: skip a detection frame whenever the
  // audio pipeline is using the GPU (an ASR pass or the assistant speaking).
  vision.pauseWhile = () =>
    duplex ? duplex.audioActive() : (pipeline?.asr.isBusy ?? false);
  // Mount the session's video element into the pip (behind the overlay).
  vision.video.className = "pip-video";
  el.pip.insertBefore(vision.video, el.pipOverlay);
  await vision.start();
  el.pip.hidden = false;
  el.stageEye.classList.add("is-active");
  if (visionRaf === null) drawPreview();
  duplex?.setVision(vision);
  tick("eye · webcam on (D-FINE on webgpu)");
  setStatus(sessionActive ? "listening — just talk" : "ready", sessionActive ? "live" : "idle");
}

function disableVision(): void {
  duplex?.setVision(null);
  vision?.stop();
  if (vision?.video.parentElement) vision.video.remove();
  vision = null;
  if (visionRaf !== null) {
    cancelAnimationFrame(visionRaf);
    visionRaf = null;
  }
  el.pip.hidden = true;
  el.stageEye.classList.remove("is-active");
  el.dotEye.classList.remove("is-on");
  el.stages.eye.classList.remove("is-active");
  el.metricEye.textContent = "off";
}

async function toggleVision(enabled: boolean): Promise<void> {
  if (visionBusy) return;
  visionBusy = true;
  el.eyeToggle.disabled = true;
  try {
    if (enabled) await enableVision();
    else disableVision();
  } catch (error) {
    console.error(error);
    tick(error instanceof Error ? error.message : String(error));
    el.eyeToggle.checked = false;
    disableVision();
  } finally {
    el.eyeToggle.disabled = false;
    visionBusy = false;
  }
}

function buildSession(pipe: VoicePipeline): DuplexSession {
  return new DuplexSession({
    pipeline: pipe,
    capture,
    getVoice: () => el.voiceSelect.value as TTSVoice,
    getModel: currentModel,
    vision,
    callbacks: {
      onCaptions(committed, tentative) {
        const any = committed || tentative;
        el.captions.hidden = !any;
        el.capCommitted.textContent = committed ? committed + " " : "";
        el.capTentative.textContent = tentative;
      },
      onUserTurn(text) {
        el.captions.hidden = true;
        // A tool card belongs to the exchange that produced it; once a new user
        // turn starts, that answer is stale — clear it so it doesn't linger.
        el.canvasPanel.hidden = true;
        el.canvasPanel.replaceChildren();
        addBubble("user").textContent = text;
      },
      onAssistantStart() {
        assistantBubble = addBubble("assistant");
        setStatus("responding", "busy");
      },
      onAssistantPartial(text) {
        if (assistantBubble) {
          assistantBubble.textContent = text;
          el.transcript.scrollTop = el.transcript.scrollHeight;
        }
      },
      onAssistantEnd(text, interrupted) {
        if (assistantBubble) {
          const shown = text.trim();
          if (!shown) assistantBubble.remove();
          else assistantBubble.textContent = shown + (interrupted ? " —" : "");
        }
        assistantBubble = null;
        if (interrupted) tick("barge-in · assistant yielded");
        if (sessionActive) setStatus("listening — just talk", "live");
      },
      onEvent(text) {
        tick(text);
      },
      onMetric(metric) {
        if (metric.asrLagMs !== undefined) {
          el.metrics.asr.textContent = `lag ${formatMs(metric.asrLagMs)}`;
        }
        if (metric.turnLatencyMs !== undefined) {
          el.metrics.llm.textContent = `turn ${formatMs(metric.turnLatencyMs)}`;
          tick(`turn latency ${formatMs(metric.turnLatencyMs)}`);
        }
        if (metric.interruptStopMs !== undefined) {
          el.metrics.tts.textContent = `stop ${formatMs(metric.interruptStopMs)}`;
          tick(`barge-in · stopped in ${formatMs(metric.interruptStopMs)}`);
        }
      },
      onStageActivity: stageActivity,
      onToolCall(kind, query) {
        toolChip = addToolChip(kind, query);
        tick(`${kind} · looking up "${query}"`);
      },
      onToolResult(card) {
        if (toolChip) {
          toolChip.classList.add("is-done");
          toolChip = null;
        }
        renderCard(card);
      },
      onBackground(active) {
        el.bgPill.hidden = !active;
      },
      onError(error) {
        console.error(error);
        tick(error instanceof Error ? error.message : String(error));
      },
    },
  });
}

async function handleOrb() {
  if (!pipeline || toggling) return;
  toggling = true;
  try {
    if (sessionActive && duplex) {
      setStatus("ending conversation", "busy");
      el.orbBtn.disabled = true;
      await duplex.stop();
      duplex = null;
      sessionActive = false;
      el.orbBtn.disabled = false;
      el.orbBtn.classList.remove("is-recording");
      el.orbBtn.setAttribute("aria-label", "Start conversation");
      orb.setState("idle");
      el.captions.hidden = true;
      for (const stage of ["asr", "llm", "tts"] as Stage[]) {
        stageActivity(stage, false);
      }
      setStatus("ready", "idle");
      setHint(
        "Press the orb once and just talk &mdash; interrupt it, pause, ask it " +
          "to time things. Press again to end.",
      );
      return;
    }

    duplex = buildSession(pipeline);
    await duplex.start();
    sessionActive = true;
    const session = duplex;
    orb.setDuplex(
      () => session.micLevel(),
      () => session.ttsLevel(),
    );
    el.orbBtn.classList.add("is-recording");
    el.orbBtn.setAttribute("aria-label", "End conversation");
    setStatus("listening — just talk", "live");
    setHint("Live. Talk naturally; talk over it to interrupt. Press the orb to end.");
    tick("duplex · live");
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), "error");
    sessionActive = false;
    duplex = null;
  } finally {
    toggling = false;
  }
}

el.loadBtn.addEventListener("click", () => void handleLoad());
el.orbBtn.addEventListener("click", () => void handleOrb());
el.eyeToggle.addEventListener("change", () =>
  void toggleVision(el.eyeToggle.checked),
);
