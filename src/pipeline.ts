// Speech-to-speech pipeline: Whisper ASR -> Gemma LLM -> Kyutai Pocket TTS.
// All three stages run locally in the browser on WebGPU via jax-js, mirroring
// the architecture of the HF/Cerebras real-time voice AI demo.

import {
  blockUntilReady,
  defaultDevice,
  type Device,
  init,
  lax,
  numpy as np,
  tree,
} from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";

import { TUNABLES } from "./tunables";

import {
  decodeTranscriptTokens,
  sampleGreedy,
} from "./asr/decoding";
import { whisperLogMel } from "./asr/features";
import {
  createWhisperState,
  fromSafetensors as whisperFromSafetensors,
  type KVCache,
  prepareWhisperCrossKV,
  runWhisperDecoderStep,
  runWhisperEncoder,
  WHISPER_MODELS,
  type WhisperConfig,
  type WhisperModel,
  type WhisperState,
} from "./asr/model";
import { WhisperTokenizer } from "./asr/tokenizer";
import {
  createGemmaState,
  ensureGemmaStateCapacity,
  fromSafetensors as gemmaFromSafetensors,
  type GemmaModel,
  type GemmaState,
  GEMMA_TOPK,
  runGemmaPrefill,
  runGemmaStep,
  runGemmaStepFused,
  runGemmaStepFusedTopK,
} from "./llm/gemma";
import {
  createSmolLmState,
  ensureSmolLmStateCapacity,
  fromSafetensors as smolLmFromSafetensors,
  runSmolLmPrefill,
  runSmolLmStepFusedTopK,
  SMOLLM_TOPK,
  type SmolLmModel,
  type SmolLmState,
} from "./llm/smollm";
import { type AudioPlayer, createStreamingPlayer } from "./tts/audio";
import { playTTS } from "./tts/inference";
import {
  fromSafetensors as ttsFromSafetensors,
  type PocketTTS,
} from "./tts/pocket-tts";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  /** Seconds since session start when the message was spoken (for time
   *  awareness). Prefixed onto the prompt as `[t+Ns]`; not shown to the user. */
  t?: number;
};

export type DownloadProgress = {
  name: string;
  loadedBytes: number;
  totalBytes?: number;
  done: boolean;
};

export type ProgressFn = (progress: DownloadProgress) => void;

// base.en over tiny.en: tiny is the most hallucination-prone Whisper size (it
// invents "thank you" / repeats on near-silence and garbles fast speech), and
// base is markedly more accurate for ~+70 MB. On WebGPU a pass is still well
// under real-time, so captions keep up.
const WHISPER_CONFIG: WhisperConfig = WHISPER_MODELS.find(
  (m) => m.id === "base.en",
)!;
const ASR_MAX_NEW_TOKENS = 96;

const GEMMA_BASE =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m";
const GEMMA_START_OF_TURN = 105;
const GEMMA_END_OF_TURN = 106;

const TTS_WEIGHTS_URL =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";
// Gemma build with the tied embedding table quantized to int8 (dequantized to
// fp16 at load, so runtime is unchanged). This is the default download: 369 MB
// vs. 536 MB for the fp16 file. GEMMA_Q8_LOCAL is checked first so a build that
// vendors the file under public/weights/ serves it without a network hop; the
// HF-hosted copy is the fresh-clone default; the fp16 file is the last resort.
const GEMMA_Q8_LOCAL = "/weights/gemma-it-q8e.safetensors";
const GEMMA_Q8_URL =
  "https://huggingface.co/sachink98/jax-realtime-weights/resolve/main/gemma-it-q8e.safetensors";
const TTS_HF_PREFIX =
  "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280";

export const TTS_VOICES = [
  "alba",
  "azelma",
  "cosette",
  "eponine",
  "fantine",
  "javert",
  "jean",
  "marius",
] as const;
export type TTSVoice = (typeof TTS_VOICES)[number];

// Gemma 3 270M has no system role; fold instructions into the first user turn.
// Keep this SHORT and POSITIVE: a 270M model can't follow long instructions
// or negation (naming a phrase to avoid just primes it to say that phrase).
const SYSTEM_HINT =
  "You are a warm, helpful voice assistant. Answer the user's question or " +
  "message directly in one or two short spoken sentences. A [scene: …] tag " +
  "tells you what the camera sees. Do not read any bracketed tag aloud.";

export async function fetchWithProgress(
  name: string,
  url: string,
  onProgress: ProgressFn,
): Promise<Uint8Array<ArrayBuffer>> {
  onProgress({ name, loadedBytes: 0, done: false });
  const data = await cachedFetch(url, {}, (p) => {
    onProgress({
      name,
      loadedBytes: p.loadedBytes,
      totalBytes: p.totalBytes,
      done: false,
    });
  });
  onProgress({ name, loadedBytes: data.byteLength, done: true });
  return data as Uint8Array<ArrayBuffer>;
}

function hfWhisperUrl(file: string): string {
  return `https://huggingface.co/${WHISPER_CONFIG.repo}/resolve/main/${file}`;
}

export class SpeechRecognizer {
  private busy = false;

  private constructor(
    private model: WhisperModel,
    private tokenizer: WhisperTokenizer,
    private dtype: np.DType,
    private device: Device | undefined,
  ) {}

  static async load(
    onProgress: ProgressFn,
    { device, dtype }: { device?: Device; dtype?: np.DType } = {},
  ): Promise<SpeechRecognizer> {
    // Whisper on the wasm backend must run in fp32 (fp16 is WebGPU-only).
    const resolvedDtype = dtype ?? (device === "wasm" ? np.float32 : np.float16);
    const vocab = await fetchWithProgress(
      "Whisper tokenizer",
      hfWhisperUrl("vocab.json"),
      onProgress,
    );
    const tokenizer = WhisperTokenizer.fromVocabBytes(vocab);
    const data = await fetchWithProgress(
      `Whisper ${WHISPER_CONFIG.label} weights`,
      hfWhisperUrl("model.safetensors"),
      onProgress,
    );
    const weights = safetensors.parse(data);
    const model = await whisperFromSafetensors(
      weights,
      resolvedDtype,
      WHISPER_CONFIG,
      device,
    );
    return new SpeechRecognizer(model, tokenizer, resolvedDtype, device);
  }

  /** True while a transcribe pass is in flight (one at a time per instance). */
  get isBusy(): boolean {
    return this.busy;
  }

  /**
   * Run one throwaway pass to trigger backend kernel compilation (JIT). On the
   * wasm lane the first Whisper pass compiles kernels and can take tens of
   * seconds; doing it here moves that cost into the loading screen instead of
   * the user's first turn.
   */
  async warmup(): Promise<void> {
    try {
      await this.transcribe(new Float32Array(16_000), 1);
    } catch {
      // Warmup is best-effort; a failure here just means the first real pass
      // pays the compilation cost.
    }
  }

  /**
   * DEV diagnostic: per-token Whisper decoder-step cost. The decoder step is
   * ALREADY a single fused jit (see asr/model.ts `decoderStepJit`: embedding →
   * all layers → norm → logits in one dispatch), so unlike Gemma there is no
   * per-layer "unfused" path to A/B against — this just reports the ms/token of
   * the shipped fused step (including the realistic full-vocab readback the gate
   * needs), so the orchestrator can confirm the ASR pass cost. Not used by the
   * app.
   */
  async benchDecodeStep(n = 24): Promise<Record<string, unknown>> {
    const config = WHISPER_CONFIG;
    const device = this.device;
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

    const features = whisperLogMel(new Float32Array(16_000 * 3));
    let inputFeatures = np.array(features.data as Float32Array<ArrayBuffer>, {
      shape: [1, features.mels, features.frames],
      dtype: np.float32,
      device,
    });
    if (this.dtype !== np.float32) inputFeatures = inputFeatures.astype(this.dtype);
    const encoded = runWhisperEncoder(this.model.encoder, inputFeatures, config);
    const crossKV = prepareWhisperCrossKV(this.model.decoder, encoded, config);
    const state = createWhisperState(
      ASR_MAX_NEW_TOKENS + config.promptTokens.length + 8,
      this.dtype,
      config,
      device,
    );

    let logits: np.Array | null = null;
    for (const token of config.promptTokens) {
      logits?.dispose();
      logits = runWhisperDecoderStep(
        this.model.decoder,
        crossKV,
        state,
        token,
        config,
        device,
      );
    }

    const dispatchMs: number[] = [];
    const fixedToken = config.timestampBeginToken; // any valid decode token
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      await logits!.data(); // consumes logits (realistic full-vocab readback)
      const d0 = performance.now();
      logits = runWhisperDecoderStep(
        this.model.decoder,
        crossKV,
        state,
        fixedToken,
        config,
        device,
      );
      dispatchMs.push(performance.now() - d0);
    }
    await blockUntilReady(logits!.ref);
    const msPerTok = (performance.now() - t0) / n;
    logits!.dispose();
    tree.dispose(crossKV);
    tree.dispose(state);

    return {
      nTokens: n,
      msPerTok: +msPerTok.toFixed(2),
      fused: true,
      note: "decoder step is already single-jit fused; no unfused path to A/B",
      dispatchSyncMs: {
        first5: dispatchMs.slice(0, 5).map((x) => +x.toFixed(2)),
        mean: +mean(dispatchMs).toFixed(2),
        max: +Math.max(...dispatchMs).toFixed(2),
      },
    };
  }

  /** Transcribe 16 kHz mono PCM samples to text. */
  async transcribe(samples: Float32Array, duration: number): Promise<string> {
    if (this.busy) {
      throw new Error("SpeechRecognizer is already transcribing");
    }
    this.busy = true;
    const config = WHISPER_CONFIG;
    const device = this.device;
    const features = whisperLogMel(samples);

    let inputFeatures: np.Array | null = null;
    let encoded: np.Array | null = null;
    let crossKV: KVCache[] | null = null;
    let state: WhisperState | null = null;
    let logits: np.Array | null = null;

    try {
      inputFeatures = np.array(features.data as Float32Array<ArrayBuffer>, {
        shape: [1, features.mels, features.frames],
        dtype: np.float32,
        device,
      });
      if (this.dtype !== np.float32) {
        inputFeatures = inputFeatures.astype(this.dtype);
      }
      encoded = runWhisperEncoder(this.model.encoder, inputFeatures.ref, config);
      crossKV = prepareWhisperCrossKV(this.model.decoder, encoded, config);
      encoded = null;

      state = createWhisperState(
        ASR_MAX_NEW_TOKENS + config.promptTokens.length + 8,
        this.dtype,
        config,
        device,
      );
      for (const token of config.promptTokens) {
        logits?.dispose();
        logits = runWhisperDecoderStep(
          this.model.decoder,
          crossKV,
          state,
          token,
          config,
          device,
        );
      }

      const generated: number[] = [];
      for (let i = 0; i < ASR_MAX_NEW_TOKENS; i++) {
        const sampledLogits = logits;
        if (!sampledLogits) throw new Error("Decoder logits were not ready");
        logits = null;
        const next = await sampleGreedy(
          sampledLogits,
          generated,
          duration,
          config,
        );
        if (next === config.eosToken) break;
        generated.push(next);
        logits = runWhisperDecoderStep(
          this.model.decoder,
          crossKV,
          state,
          next,
          config,
          device,
        );
      }

      return decodeTranscriptTokens(generated, this.tokenizer, config).trim();
    } finally {
      logits?.dispose();
      encoded?.dispose();
      inputFeatures?.dispose();
      if (crossKV) tree.dispose(crossKV);
      if (state) tree.dispose(state);
      this.busy = false;
    }
  }
}

/** Length of the longest shared prefix of two token-id sequences. */
function commonPrefixLen(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

type Candidate = { id: number; logit: number };

/** Insert `{id, logit}` into a descending-sorted top-`k` candidate list. */
function insertCandidate(
  candidates: Candidate[],
  id: number,
  logit: number,
  k: number,
): void {
  if (Number.isNaN(logit)) return;
  if (
    candidates.length < k ||
    logit > candidates[candidates.length - 1].logit
  ) {
    let insertIndex = 0;
    while (
      insertIndex < candidates.length &&
      logit < candidates[insertIndex].logit
    ) {
      insertIndex++;
    }
    candidates.splice(insertIndex, 0, { id, logit });
    if (candidates.length > k) candidates.pop();
  }
}

/** Temperature / top-p nucleus draw over an already-sorted candidate list. */
function sampleFromCandidates(
  candidates: Candidate[],
  opts: { temperature: number; topP: number },
): number {
  if (candidates.length === 0) {
    throw new Error("Model returned all-NaN logits.");
  }
  if (opts.temperature <= 0) return candidates[0].id;

  const maxLogit = candidates[0].logit;
  if (!Number.isFinite(maxLogit)) return candidates[0].id;

  const probs = candidates.map((c) =>
    Math.exp((c.logit - maxLogit) / opts.temperature),
  );
  const total = probs.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(total) || total <= 0) return candidates[0].id;

  let keptTotal = 0;
  let kept = 0;
  for (; kept < candidates.length; kept++) {
    keptTotal += probs[kept];
    if (keptTotal / total >= opts.topP) {
      kept++;
      break;
    }
  }
  if (kept === 0) kept = 1;

  let r = Math.random() * keptTotal;
  for (let i = 0; i < kept; i++) {
    r -= probs[i];
    if (r <= 0) return candidates[i].id;
  }
  return candidates[kept - 1].id;
}

function sampleLogits(
  logits: Float32Array,
  opts: { temperature: number; topK: number; topP: number },
): number {
  const k = Math.max(1, Math.min(opts.topK, logits.length));
  const candidates: Candidate[] = [];
  for (let id = 0; id < logits.length; id++) {
    insertCandidate(candidates, id, logits[id], k);
  }
  return sampleFromCandidates(candidates, opts);
}

/**
 * Equivalent to `sampleLogits`, but over the top-k `(value, index)` pairs read
 * back from a device-side `lax.topK(logits, k)` instead of the full vocab. The
 * pairs are fed through the *same* `insertCandidate` logic in ascending-id
 * order (topK returns them value-descending) so the resulting sorted candidate
 * list — and hence the sampled token for any RNG state — is identical to the
 * full-vocab path. For distinct logits (the norm) the top-k set is identical;
 * an exact tie at the k/k+1 boundary is the only way the two sets could differ,
 * which is vanishingly unlikely with fp32 logits and cannot affect the
 * temperature-0 argmax.
 */
function sampleTopKPairs(
  values: ArrayLike<number>,
  indices: ArrayLike<number>,
  opts: { temperature: number; topK: number; topP: number },
): number {
  const n = Math.min(values.length, indices.length);
  const k = Math.max(1, Math.min(opts.topK, n));
  const pairs: Candidate[] = [];
  for (let j = 0; j < n; j++) pairs.push({ id: indices[j], logit: values[j] });
  pairs.sort((a, b) => a.id - b.id);
  const candidates: Candidate[] = [];
  for (const p of pairs) insertCandidate(candidates, p.id, p.logit, k);
  return sampleFromCandidates(candidates, opts);
}

export type GenerateStats = {
  promptTokens: number;
  newTokens: number;
  firstTokenMs: number;
  totalMs: number;
};

/**
 * Common LLM interface. `generateStream` yields incremental text deltas as
 * tokens arrive and returns final stats; `generate` is the buffered form.
 */
export interface ChatModel {
  generate(
    history: ChatMessage[],
    onText: (partial: string) => void,
  ): Promise<{ text: string; stats: GenerateStats }>;
  generateStream(
    history: ChatMessage[],
  ): AsyncGenerator<string, GenerateStats, void>;
}

/**
 * Cap chat history to the last `TUNABLES.llmMaxHistoryTurns` messages (whole
 * user/assistant pairs). Shared by both local brains so every backend windows
 * identically instead of iterating the full transcript each turn.
 */
function windowHistory(history: ChatMessage[]): ChatMessage[] {
  const n = TUNABLES.llmMaxHistoryTurns;
  if (n <= 0 || history.length <= n) return history;
  // Keep the last N messages, but never start on an orphaned assistant reply
  // — drop it so the window always begins on a user turn (whole pairs).
  let sliced = history.slice(history.length - n);
  if (sliced[0]?.role === "assistant") sliced = sliced.slice(1);
  return sliced;
}

export class LocalChatModel implements ChatModel {
  /** Persistent KV-cache reused across turns when TUNABLES.llmKvReuse is on. */
  private kvState: GemmaState | null = null;
  /** Token IDs currently resident in `kvState` (prompt + fed generations). */
  private kvTokens: number[] = [];

  private constructor(
    private model: GemmaModel,
    private tokenizer: tokenizers.SentencePiece,
  ) {}

  static async load(onProgress: ProgressFn): Promise<LocalChatModel> {
    const tokData = await fetchWithProgress(
      "Gemma tokenizer",
      `${GEMMA_BASE}/tokenizer.model`,
      onProgress,
    );
    const tokenizer = tokenizers.SentencePiece.fromBinary(tokData);
    // Default to the int8-embedding build (536 → 369 MB download; dequantized
    // to fp16 at load, so runtime is unchanged). Prefer a locally-vendored copy
    // under public/weights/, else the HF-hosted default, else the fp16 file.
    // The local HEAD probe guards against the dev server's SPA fallback
    // answering 200 with index.html for a missing file (hence the size gate).
    let weightsUrl = GEMMA_Q8_URL;
    let weightsLabel = "Gemma 3 270M weights (int8 embed)";
    try {
      const head = await fetch(GEMMA_Q8_LOCAL, { method: "HEAD" });
      const size = Number(head.headers.get("content-length") ?? 0);
      if (head.ok && size > 100_000_000) {
        weightsUrl = GEMMA_Q8_LOCAL;
      }
    } catch {
      // No locally-vendored build; use the HF-hosted int8 default.
    }
    let data: Uint8Array<ArrayBuffer>;
    try {
      data = await fetchWithProgress(weightsLabel, weightsUrl, onProgress);
    } catch (err) {
      // int8 download unreachable — fall back to the full fp16 file so the app
      // still loads.
      console.warn("int8 Gemma download failed, falling back to fp16", err);
      data = await fetchWithProgress(
        "Gemma 3 270M weights",
        `${GEMMA_BASE}/model-it-fp16.safetensors`,
        onProgress,
      );
    }
    const weights = safetensors.parse(data);
    const model = await gemmaFromSafetensors(weights, np.float16);
    return new LocalChatModel(model, tokenizer);
  }

  /**
   * Generate a couple of throwaway tokens so the Gemma prefill + decode kernels
   * JIT-compile at load time instead of on the user's first turn (that cold
   * pass added ~1.7 s to turn 1).
   */
  async warmup(): Promise<void> {
    try {
      const stream = this.generateStream([{ role: "user", content: "Hi" }], 2);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // drain
      }
    } catch {
      // Best-effort; a failure just means the first real turn pays the cost.
    }
  }

  /**
   * DEV diagnostic (cycle-4 map-reduce): per-token decode cost split. Measures
   * the shipped path (full-vocab readback + JS scan per token) against a
   * no-readback path, the JS scan alone, and a GPU `lax.topK(64)` path, plus
   * the synchronous dispatch cost of each `runGemmaStep` call (re-trace check).
   * Throwaway measurement code — not used by the app.
   */
  private benchPrompt(): number[] {
    return [
      this.tokenizer.bosToken,
      ...this.tokenizer.encode(
        "<start_of_turn>user\nTell me about the ocean.<end_of_turn>\n<start_of_turn>model\n",
      ),
    ];
  }

  async benchDecode(
    n = 24,
    opts?: { fused?: boolean; sampler?: "js" | "topk"; topkInFused?: boolean },
  ): Promise<Record<string, unknown>> {
    // When a specific configuration is requested, measure just that config's
    // ms/token and per-call sync dispatch cost (used by the map-reduce run).
    if (opts) return this.benchConfig(n, opts);
    const prompt = this.benchPrompt();
    const FIXED_TOKEN = 108;
    const mkState = () => {
      const state = createGemmaState({ dtype: np.float16 });
      const ids = np.array(prompt, { dtype: np.uint32 });
      return { state, logits: runGemmaPrefill(tree.ref(this.model), ids, state) };
    };
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

    // A. Shipped path: await data() (full 262k fp32 readback) + JS scan + step.
    let s = mkState();
    let logits = s.logits;
    const dispatchMs: number[] = [];
    let tokSample = 0;
    const tA = performance.now();
    for (let i = 0; i < n; i++) {
      const data = (await logits.data()) as Float32Array;
      tokSample = sampleLogits(data, { temperature: 0.7, topK: 64, topP: 0.95 });
      const d0 = performance.now();
      logits = runGemmaStep(tree.ref(this.model), tokSample, s.state);
      dispatchMs.push(performance.now() - d0);
    }
    await blockUntilReady(logits.ref);
    const fullMsPerTok = (performance.now() - tA) / n;

    // E. Readback alone on a settled queue (map + 1MB transfer, no GPU wait).
    const tE = performance.now();
    const settled = (await logits.data()) as Float32Array; // consumes logits
    const readbackMs = performance.now() - tE;
    tree.dispose(s.state);

    // C. JS scan alone on the CPU-side array.
    const tC = performance.now();
    for (let i = 0; i < n; i++) {
      sampleLogits(settled, { temperature: 0.7, topK: 64, topP: 0.95 });
    }
    const scanMsPerTok = (performance.now() - tC) / n;

    // B. No-readback path: steps enqueued back-to-back with a fixed token; one
    // sync at the end. Isolates GPU compute + dispatch from readback/sync.
    s = mkState();
    logits = s.logits;
    const tB = performance.now();
    for (let i = 0; i < n; i++) {
      logits.dispose();
      logits = runGemmaStep(tree.ref(this.model), FIXED_TOKEN, s.state);
    }
    await blockUntilReady(logits.ref);
    const noReadbackMsPerTok = (performance.now() - tB) / n;
    logits.dispose();
    tree.dispose(s.state);

    // D. Candidate topk-gpu path: lax.topK(64) on device, read back 64 pairs.
    s = mkState();
    logits = s.logits;
    const tD = performance.now();
    for (let i = 0; i < n; i++) {
      const [vals, idx] = lax.topK(logits, 64); // consumes logits
      await vals.data();
      await idx.data();
      logits = runGemmaStep(tree.ref(this.model), FIXED_TOKEN, s.state);
    }
    await blockUntilReady(logits.ref);
    const topkMsPerTok = (performance.now() - tD) / n;
    logits.dispose();
    tree.dispose(s.state);

    return {
      nTokens: n,
      fullMsPerTok: +fullMsPerTok.toFixed(2),
      noReadbackMsPerTok: +noReadbackMsPerTok.toFixed(2),
      readbackPlusSyncMsPerTok: +(fullMsPerTok - noReadbackMsPerTok).toFixed(2),
      readbackAloneMs: +readbackMs.toFixed(2),
      scanMsPerTok: +scanMsPerTok.toFixed(2),
      topkMsPerTok: +topkMsPerTok.toFixed(2),
      dispatchSyncMs: {
        first5: dispatchMs.slice(0, 5).map((x) => +x.toFixed(2)),
        mean: +mean(dispatchMs).toFixed(2),
        max: +Math.max(...dispatchMs).toFixed(2),
      },
    };
  }

  /**
   * Measure ms/token for one decode configuration (fused step on/off, GPU
   * top-k sampler on/off) with real sampled feedback tokens, plus the mean/max
   * synchronous dispatch cost of each step call. Same measurement style as
   * `benchDecode` above; used by the map-reduce harness to compare configs.
   */
  private async benchConfig(
    n: number,
    {
      fused = false,
      sampler = "js" as "js" | "topk",
      topkInFused = false,
    },
  ): Promise<Record<string, unknown>> {
    const prompt = this.benchPrompt();
    const stepFn = fused ? runGemmaStepFused : runGemmaStep;
    const opts = { temperature: 0.7, topK: GEMMA_TOPK, topP: 0.95 };
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

    const state = createGemmaState({ dtype: np.float16 });
    // `out` is the current sampleable: full-vocab logits, or the packed top-k
    // array once the folded step is producing them (isCombined tracks which).
    let out = runGemmaPrefill(
      tree.ref(this.model),
      np.array(prompt, { dtype: np.uint32 }),
      state,
    );
    let isCombined = false;

    const sampleCombined = (packed: ArrayLike<number>): number => {
      const values = new Array(GEMMA_TOPK);
      const indices = new Array(GEMMA_TOPK);
      for (let j = 0; j < GEMMA_TOPK; j++) {
        values[j] = packed[j];
        indices[j] = packed[GEMMA_TOPK + j];
      }
      return sampleTopKPairs(values, indices, opts);
    };

    const dispatchMs: number[] = [];
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      let tok: number;
      if (isCombined) {
        tok = sampleCombined((await out.data()) as ArrayLike<number>);
      } else if (sampler === "topk") {
        const [vals, idx] = lax.topK(out, 64); // consumes out
        const v = (await vals.data()) as ArrayLike<number>;
        const ix = (await idx.data()) as ArrayLike<number>;
        tok = sampleTopKPairs(v, ix, opts);
      } else {
        tok = sampleLogits((await out.data()) as Float32Array, opts);
      }
      const d0 = performance.now();
      if (topkInFused) {
        out = runGemmaStepFusedTopK(tree.ref(this.model), tok, state);
        isCombined = true;
      } else {
        out = stepFn(tree.ref(this.model), tok, state);
        isCombined = false;
      }
      dispatchMs.push(performance.now() - d0);
    }
    await blockUntilReady(out.ref);
    const msPerTok = (performance.now() - t0) / n;
    out.dispose();
    tree.dispose(state);

    return {
      config: { fused, sampler, topkInFused },
      nTokens: n,
      msPerTok: +msPerTok.toFixed(2),
      dispatchSyncMs: {
        first5: dispatchMs.slice(0, 5).map((x) => +x.toFixed(2)),
        mean: +mean(dispatchMs).toFixed(2),
        max: +Math.max(...dispatchMs).toFixed(2),
      },
    };
  }

  /**
   * Greedy (temperature 0) generation of `nTokens` from a fixed prompt under
   * each of the four {fused, sampler} configurations, returning the four
   * token-id arrays. The orchestrator asserts they are identical, proving the
   * fused step and the GPU top-k sampler preserve the shipped path's output.
   */
  async benchEquivalence(nTokens = 32): Promise<Record<string, number[]>> {
    const prompt = this.benchPrompt();
    const greedy = { temperature: 0, topK: 64, topP: 0.95 };

    const run = async (
      fused: boolean,
      sampler: "js" | "topk",
      topkInFused = false,
    ): Promise<number[]> => {
      const stepFn = fused ? runGemmaStepFused : runGemmaStep;
      const state = createGemmaState({ dtype: np.float16 });
      let cur = runGemmaPrefill(
        tree.ref(this.model),
        np.array(prompt, { dtype: np.uint32 }),
        state,
      );
      let isCombined = false;
      const out: number[] = [];
      try {
        for (let i = 0; i < nTokens; i++) {
          let tok: number;
          if (isCombined) {
            const packed = (await cur.data()) as ArrayLike<number>;
            const values = new Array(GEMMA_TOPK);
            const indices = new Array(GEMMA_TOPK);
            for (let j = 0; j < GEMMA_TOPK; j++) {
              values[j] = packed[j];
              indices[j] = packed[GEMMA_TOPK + j];
            }
            tok = sampleTopKPairs(values, indices, greedy);
          } else if (sampler === "topk") {
            const [vals, idx] = lax.topK(cur, 64); // consumes cur
            const v = (await vals.data()) as ArrayLike<number>;
            const ix = (await idx.data()) as ArrayLike<number>;
            tok = sampleTopKPairs(v, ix, greedy);
          } else {
            tok = sampleLogits((await cur.data()) as Float32Array, greedy);
          }
          out.push(tok);
          if (topkInFused) {
            cur = runGemmaStepFusedTopK(tree.ref(this.model), tok, state);
            isCombined = true;
          } else {
            cur = stepFn(tree.ref(this.model), tok, state);
            isCombined = false;
          }
        }
      } finally {
        cur.dispose();
        tree.dispose(state);
      }
      return out;
    };

    return {
      jsUnfused: await run(false, "js"),
      jsFused: await run(true, "js"),
      topkUnfused: await run(false, "topk"),
      topkFused: await run(true, "topk"),
      topkInFused: await run(true, "topk", true),
    };
  }

  /**
   * Sample the next token from on-device logits, honoring TUNABLES.llmSampler.
   * Both paths consume `logits`; "topk" reads back only 64 candidates instead
   * of the full 262k-vocab, with an identical selection (see sampleTopKPairs).
   */
  private async sampleNext(
    logits: np.Array,
    temperature: number,
  ): Promise<number> {
    const opts = { temperature, topK: 64, topP: 0.95 };
    if (TUNABLES.llmSampler === "topk") {
      const [vals, idx] = lax.topK(logits, 64); // consumes logits
      const v = (await vals.data()) as ArrayLike<number>;
      const ix = (await idx.data()) as ArrayLike<number>;
      return sampleTopKPairs(v, ix, opts);
    }
    return sampleLogits((await logits.data()) as Float32Array, opts);
  }

  /**
   * Sample from the packed top-k array emitted by the topk-in-fused decode step
   * (see runGemmaDecodeStepFusedTopK): one readback of [values ..k.., indices
   * ..k..] fp32, split back into pairs and fed through the same selection as
   * `sampleTopKPairs` — identical semantics, one `.data()` instead of two.
   */
  private async sampleNextFromCombined(
    combined: np.Array,
    temperature: number,
  ): Promise<number> {
    const opts = { temperature, topK: GEMMA_TOPK, topP: 0.95 };
    const packed = (await combined.data()) as ArrayLike<number>; // consumes
    const values = new Array(GEMMA_TOPK);
    const indices = new Array(GEMMA_TOPK);
    for (let i = 0; i < GEMMA_TOPK; i++) {
      values[i] = packed[i];
      indices[i] = packed[GEMMA_TOPK + i];
    }
    return sampleTopKPairs(values, indices, opts);
  }

  /**
   * Prepare decode state for a turn. Without KV reuse: a fresh full prefill
   * (shipped behavior), disposed after the stream. With reuse: re-use the
   * persistent cache, prefilling only the suffix of `promptTokens` past the
   * longest shared prefix; if there is no usable prefix (or the prompt exceeds
   * the cap) the persistent cache is rebuilt.
   */
  private prepareState(promptTokens: number[]): {
    state: GemmaState;
    fedTokens: number[];
    logits: np.Array;
    persistent: boolean;
  } {
    const MAX_CACHED = 2048;

    if (TUNABLES.llmKvReuse) {
      const prev = this.kvState;
      const p = prev ? commonPrefixLen(this.kvTokens, promptTokens) : 0;
      if (
        prev &&
        p > 0 &&
        p < promptTokens.length &&
        promptTokens.length <= MAX_CACHED
      ) {
        // Roll the cache back to the shared prefix and re-feed the diverging
        // suffix. Stale KV slots beyond `p` are overwritten before they can be
        // attended to (each step writes its own slot and attends only slots
        // <= its position), so the result is exact.
        ensureGemmaStateCapacity(prev, promptTokens.length);
        prev.position = p;
        const fedTokens = promptTokens.slice(0, p);
        let logits: np.Array | null = null;
        for (let j = p; j < promptTokens.length; j++) {
          logits?.dispose();
          logits = runGemmaStep(tree.ref(this.model), promptTokens[j], prev);
          fedTokens.push(promptTokens[j]);
        }
        return { state: prev, fedTokens, logits: logits!, persistent: true };
      }
      if (prev) tree.dispose(prev);
      this.kvState = null;
      this.kvTokens = [];
      const state = createGemmaState({ dtype: np.float16 });
      const logits = runGemmaPrefill(
        tree.ref(this.model),
        np.array(promptTokens, { dtype: np.uint32 }),
        state,
      );
      return { state, fedTokens: promptTokens.slice(), logits, persistent: true };
    }

    const state = createGemmaState({ dtype: np.float16 });
    const logits = runGemmaPrefill(
      tree.ref(this.model),
      np.array(promptTokens, { dtype: np.uint32 }),
      state,
    );
    return { state, fedTokens: promptTokens.slice(), logits, persistent: false };
  }

  /** Persist (or drop, if over the cap) the reused KV state after a turn. */
  private commitState(state: GemmaState, fedTokens: number[]): void {
    const MAX_CACHED = 2048;
    if (fedTokens.length > MAX_CACHED) {
      tree.dispose(state);
      this.kvState = null;
      this.kvTokens = [];
      return;
    }
    this.kvState = state;
    this.kvTokens = fedTokens;
  }

  private formatPrompt(history: ChatMessage[]): string {
    // Gemma 3 270M follows the plain chat template far better than an
    // instruction preamble — folding a system prompt or per-turn [t+Ns] tag in
    // here made the tiny model echo the instructions instead of answering. So
    // the local prompt is just the raw conversation (the SYSTEM_HINT is used
    // only by the larger Cerebras model). A [scene: …] tag, when present, is
    // already baked into the message content by the duplex layer.
    history = windowHistory(history);
    let text = "";
    for (const message of history) {
      const content = message.content.trim();
      if (content === "") continue;
      const role = message.role === "assistant" ? "model" : "user";
      text += `<start_of_turn>${role}\n${content}<end_of_turn>\n`;
    }
    text += "<start_of_turn>model\n";
    return text;
  }

  /**
   * Yield decoded text deltas as tokens are sampled, returning final stats.
   * Consuming a suffix of this generator (e.g. stopping early on barge-in)
   * runs the `finally` block and releases the KV-cache state.
   */
  async *generateStream(
    history: ChatMessage[],
    // Kept short so spoken replies stay to a couple of sentences — without a
    // length instruction (which the 270M model echoes) this cap is what keeps
    // it from monologuing.
    maxNewTokens = TUNABLES.llmMaxNewTokens,
  ): AsyncGenerator<string, GenerateStats, void> {
    const promptTokens = [
      this.tokenizer.bosToken,
      ...this.tokenizer.encode(this.formatPrompt(history)),
    ];
    const generatedTokens: number[] = [];
    // Fused single-dispatch decode vs the shipped per-layer path; same signature
    // and side effects, so it swaps in transparently.
    const stepFn = TUNABLES.llmFusedStep ? runGemmaStepFused : runGemmaStep;
    // When on, the fused step folds topK into its jit and returns the packed
    // top-k array instead of full logits (only meaningful with the fused step).
    const topkInFused = TUNABLES.llmFusedStep && TUNABLES.llmTopkInFused;
    const startTime = performance.now();
    let firstTokenMs = 0;
    let emitted = "";

    const prepared = this.prepareState(promptTokens);
    const { state, fedTokens, persistent } = prepared;
    // `pending` is the next thing to sample from: the prefill's full-vocab
    // logits for token 0, then either full logits or a packed top-k array from
    // each step (pendingIsCombined tracks which, so the right sampler is used).
    let pending: np.Array | null = prepared.logits;
    let pendingIsCombined = false;

    try {
      const stopTokens = [this.tokenizer.eosToken, GEMMA_END_OF_TURN];

      for (let i = 0; i < maxNewTokens; i++) {
        const sampleable = pending!;
        pending = null;
        // The plain chat template (no instruction preamble) responds best at a
        // natural sampling temperature, matching the jax-js chat demo.
        const nextToken = pendingIsCombined
          ? await this.sampleNextFromCombined(sampleable, 0.7)
          : await this.sampleNext(sampleable, 0.7);
        if (i === 0) firstTokenMs = performance.now() - startTime;
        if (stopTokens.includes(nextToken)) break;

        generatedTokens.push(nextToken);
        const full = this.decodeVisible(generatedTokens);
        const delta = full.startsWith(emitted) ? full.slice(emitted.length) : full;
        emitted = full;
        if (delta) yield delta;

        if (i === maxNewTokens - 1) break;
        if (topkInFused) {
          pending = runGemmaStepFusedTopK(tree.ref(this.model), nextToken, state);
          pendingIsCombined = true;
        } else {
          pending = stepFn(tree.ref(this.model), nextToken, state);
          pendingIsCombined = false;
        }
        fedTokens.push(nextToken);
      }

      return {
        promptTokens: promptTokens.length,
        newTokens: generatedTokens.length,
        firstTokenMs,
        totalMs: performance.now() - startTime,
      };
    } finally {
      pending?.dispose();
      // With KV reuse the state is persisted for the next turn; otherwise it is
      // released here (shipped behavior).
      if (persistent) this.commitState(state, fedTokens);
      else tree.dispose(state);
    }
  }

  async generate(
    history: ChatMessage[],
    onText: (partial: string) => void,
    maxNewTokens = TUNABLES.llmMaxNewTokens,
  ): Promise<{ text: string; stats: GenerateStats }> {
    let text = "";
    const stream = this.generateStream(history, maxNewTokens);
    let result = await stream.next();
    while (!result.done) {
      text += result.value;
      onText(text);
      result = await stream.next();
    }
    return { text: text.trim(), stats: result.value };
  }

  private decodeVisible(tokens: number[]): string {
    const visible = tokens.filter(
      (t) =>
        t !== 0 &&
        t !== this.tokenizer.bosToken &&
        t !== this.tokenizer.eosToken &&
        t !== GEMMA_START_OF_TURN &&
        t !== GEMMA_END_OF_TURN,
    );
    return this.tokenizer.decode(visible);
  }
}

// SmolLM2-360M-Instruct brain. A blind-judged model shootout put it ~+0.9 (of 5)
// over Gemma 3 270M at essentially the same size, winning every dimension across
// three runs (see docs/BRAIN.md). Weights (fp16) + a precomputed tiktoken-style
// tokenizer artifact are hosted on Hugging Face (CORS-open).
const SMOLLM_BASE =
  "https://huggingface.co/sachink98/jax-realtime-weights/resolve/main";
const SMOLLM_WEIGHTS_URL = `${SMOLLM_BASE}/smollm2-360m-it-fp16.safetensors`;
// Per-row symmetric int8 build (363 MB vs 724 MB fp16), dequantized to fp16 at
// load so runtime is unchanged. Campaign-validated near-lossless (ppl +0.7%).
const SMOLLM_Q8_URL = `${SMOLLM_BASE}/smollm2-360m-it-q8r.safetensors`;
const SMOLLM_TOKENIZER_URL = `${SMOLLM_BASE}/smollm2-360m-tokenizer.json`;
const SMOLLM_REPEAT_PENALTY = 1.3;
const EMPTY_SET: ReadonlySet<number> = new Set<number>();
const SMOLLM_IM_START = 1; // <|im_start|>
const SMOLLM_IM_END = 2; // <|im_end|> — ends the assistant turn
const SMOLLM_EOS = 0; // <|endoftext|>
// SmolLM2 honors a system role (Gemma 3 270M didn't). A spoken-format prompt.
const SMOLLM_SYSTEM =
  "You are a warm, helpful voice assistant. Answer directly in a natural, " +
  "spoken style — a sentence or two, no lists, bullet points, or markdown. A " +
  "[scene: …] tag tells you what the camera sees; never read a bracketed tag " +
  "aloud.";

type SmolLmTokenizerData = {
  encoder: Record<string, number>;
  special: Record<string, number>;
  pattern: string;
};

/**
 * SmolLM2 brain implementing the same ChatModel interface as LocalChatModel, so
 * the duplex engine is agnostic to which model backs it. Uses the fused
 * single-dispatch decode + GPU top-k path for parity with the Gemma perf work.
 */
export class SmolLmChatModel implements ChatModel {
  /**
   * Persistent KV-cache reused across turns when TUNABLES.llmKvReuse is on
   * (same flag as LocalChatModel). Detached (nulled) while a turn is running so
   * a mid-turn throw can never leave a half-updated cache installed; it is only
   * re-installed by `commitState` after the turn ends cleanly.
   */
  private kvState: SmolLmState | null = null;
  /** Token IDs currently resident in `kvState` (prompt + fed generations). */
  private kvTokens: number[] = [];

  private constructor(
    private model: SmolLmModel,
    private tokenizer: tokenizers.BpeEncoding,
    private specialIds: Set<number>,
  ) {}

  static async load(onProgress: ProgressFn): Promise<SmolLmChatModel> {
    const tokData = await fetchWithProgress(
      "SmolLM tokenizer",
      SMOLLM_TOKENIZER_URL,
      onProgress,
    );
    const spec = JSON.parse(
      new TextDecoder().decode(tokData),
    ) as SmolLmTokenizerData;
    const tokenizer = new tokenizers.BpeEncoding(
      new Map(Object.entries(spec.encoder)),
      spec.special,
      new RegExp(spec.pattern, "gu"),
    );
    const specialIds = new Set(Object.values(spec.special));

    let data: Uint8Array<ArrayBuffer>;
    try {
      data = await fetchWithProgress(
        "SmolLM2 360M weights (int8)",
        SMOLLM_Q8_URL,
        onProgress,
      );
    } catch (err) {
      // int8 download unreachable — fall back to the full fp16 file so the app
      // still loads.
      console.warn("int8 SmolLM download failed, falling back to fp16", err);
      data = await fetchWithProgress(
        "SmolLM2 360M weights",
        SMOLLM_WEIGHTS_URL,
        onProgress,
      );
    }
    const weights = safetensors.parse(data);
    const model = await smolLmFromSafetensors(weights, np.float16);
    return new SmolLmChatModel(model, tokenizer, specialIds);
  }

  async warmup(): Promise<void> {
    try {
      const stream = this.generateStream([{ role: "user", content: "Hi" }], 2);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        /* drain */
      }
    } catch {
      // Best-effort; a failure just means the first real turn pays JIT cost.
    }
  }

  // ChatML prompt tokens, assembled manually. We can't use
  // tokenizer.encodeWithSpecialTokens: @jax-js/loaders' BpeEncoding.encode
  // mis-advances past special tokens (uses RegExpExecArray.length === 1 instead
  // of the matched string length), re-tokenizing them as ordinary text. Instead
  // we encode special-token-free segments and splice the known ids in.
  private encodePrompt(history: ChatMessage[]): number[] {
    const enc = (s: string) => this.tokenizer.encode(s);
    const nl = enc("\n");
    const tokens: number[] = [];
    const turn = (role: string, content: string) => {
      tokens.push(SMOLLM_IM_START, ...enc(`${role}\n${content}`), SMOLLM_IM_END, ...nl);
    };
    turn("system", SMOLLM_SYSTEM);
    for (const message of windowHistory(history)) {
      const content = message.content.trim();
      if (content === "") continue;
      turn(message.role === "assistant" ? "assistant" : "user", content);
    }
    tokens.push(SMOLLM_IM_START, ...enc("assistant\n"));
    return tokens;
  }

  // HF-style repetition penalty over the top-k candidates: divide (or, for
  // negative logits, multiply) the value of any candidate whose token id is in
  // `penalize`. Seeded with the previous assistant turn + this turn's output, it
  // breaks the verbatim-repeat loops a 360M model falls into ("give me a
  // different joke" → the same joke).
  private penalize(
    values: number[],
    indices: ArrayLike<number>,
    penalize: ReadonlySet<number>,
  ): number[] {
    if (penalize.size === 0) return values;
    return values.map((v, j) =>
      penalize.has(indices[j])
        ? v > 0
          ? v / SMOLLM_REPEAT_PENALTY
          : v * SMOLLM_REPEAT_PENALTY
        : v,
    );
  }

  private async sampleFromCombined(
    combined: np.Array,
    temperature: number,
    penalize: ReadonlySet<number> = EMPTY_SET,
  ): Promise<number> {
    const opts = { temperature, topK: SMOLLM_TOPK, topP: 0.95 };
    const packed = (await combined.data()) as ArrayLike<number>; // consumes
    const values = new Array(SMOLLM_TOPK);
    const indices = new Array(SMOLLM_TOPK);
    for (let i = 0; i < SMOLLM_TOPK; i++) {
      values[i] = packed[i];
      indices[i] = packed[SMOLLM_TOPK + i];
    }
    return sampleTopKPairs(this.penalize(values, indices, penalize), indices, opts);
  }

  private async sampleFromLogits(
    logits: np.Array,
    temperature: number,
    penalize: ReadonlySet<number> = EMPTY_SET,
  ): Promise<number> {
    const [vals, idx] = lax.topK(logits, SMOLLM_TOPK); // consumes logits
    const v = Array.from((await vals.data()) as ArrayLike<number>);
    const ix = (await idx.data()) as ArrayLike<number>;
    return sampleTopKPairs(this.penalize(v, ix, penalize), ix, {
      temperature,
      topK: SMOLLM_TOPK,
      topP: 0.95,
    });
  }

  private decodeVisible(tokens: number[]): string {
    return this.tokenizer.decode(tokens.filter((t) => !this.specialIds.has(t)));
  }

  /**
   * Prepare decode state for a turn (SmolLM port of LocalChatModel.prepareState).
   * Without KV reuse: a fresh full prefill (shipped behavior), disposed after
   * the stream. With reuse: roll the persistent cache back to the longest
   * shared token-id prefix and feed only the suffix. The suffix is fed
   * token-by-token through the SAME fused topk decode jit the generation loop
   * uses (already traced/warm, one dispatch per token) — runSmolLmPrefill
   * cannot start at an offset: it rebuilds every layer cache from scratch with
   * RoPE offset 0 and a self-only causal mask, so extending it would mean a
   * prefix-aware attention mask + cache concat for marginal gain on the short
   * (~new-user-utterance-sized) suffixes this path sees.
   *
   * Safety: the persistent state is DETACHED from `this` here. If anything
   * throws mid-turn the caller disposes it and the next turn rebuilds; only a
   * cleanly-finished turn re-installs it via commitState. (This deliberately
   * avoids the known Gemma-path bug where a mid-loop throw commits a
   * partially-fed cache — see docs/BENCHMARKS.md.)
   */
  private prepareState(promptTokens: number[]): {
    state: SmolLmState;
    fedTokens: number[];
    pending: np.Array;
    // Whether `pending` is the packed [topk values, topk indices] array from
    // the fused step (suffix feed) or full-vocab logits (full prefill).
    pendingIsCombined: boolean;
    persistent: boolean;
  } {
    const MAX_CACHED = 2048;

    if (TUNABLES.llmKvReuse) {
      const prev = this.kvState;
      // Detach immediately: from here on, `this` never references a state that
      // an exception could leave half-updated.
      this.kvState = null;
      const cachedTokens = this.kvTokens;
      this.kvTokens = [];
      const p = prev ? commonPrefixLen(cachedTokens, promptTokens) : 0;
      // One line per turn so the bench can verify reuse actually fires.
      console.debug(
        `[smollm kv] prefixLen=${p} promptLen=${promptTokens.length} cached=${cachedTokens.length}`,
      );
      if (
        prev &&
        p > 0 &&
        p < promptTokens.length &&
        promptTokens.length <= MAX_CACHED
      ) {
        // Roll the cache back to the shared prefix and re-feed the diverging
        // suffix. Stale KV slots beyond `p` are overwritten before they can be
        // attended to (each fused step writes slot `position` and its
        // validMask admits only slots <= position), so the result is exact.
        try {
          ensureSmolLmStateCapacity(prev, promptTokens.length);
          prev.position = p;
          const fedTokens = promptTokens.slice(0, p);
          let pending: np.Array | null = null;
          for (let j = p; j < promptTokens.length; j++) {
            pending?.dispose();
            pending = runSmolLmStepFusedTopK(
              tree.ref(this.model),
              promptTokens[j],
              prev,
            );
            fedTokens.push(promptTokens[j]);
          }
          return {
            state: prev,
            fedTokens,
            pending: pending!,
            pendingIsCombined: true,
            persistent: true,
          };
        } catch (err) {
          // A throw mid-suffix-feed leaves `prev` inconsistent; drop it so the
          // rebuild below (next turn) starts clean.
          tree.dispose(prev);
          throw err;
        }
      }
      if (prev) tree.dispose(prev);
      const state = createSmolLmState({ dtype: np.float16 });
      const pending = runSmolLmPrefill(
        tree.ref(this.model),
        np.array(promptTokens, { dtype: np.uint32 }),
        state,
      );
      return {
        state,
        fedTokens: promptTokens.slice(),
        pending,
        pendingIsCombined: false,
        persistent: true,
      };
    }

    const state = createSmolLmState({ dtype: np.float16 });
    const pending = runSmolLmPrefill(
      tree.ref(this.model),
      np.array(promptTokens, { dtype: np.uint32 }),
      state,
    );
    return {
      state,
      fedTokens: promptTokens.slice(),
      pending,
      pendingIsCombined: false,
      persistent: false,
    };
  }

  /** Persist (or drop, if over the cap) the reused KV state after a turn. */
  private commitState(state: SmolLmState, fedTokens: number[]): void {
    const MAX_CACHED = 2048;
    if (fedTokens.length > MAX_CACHED) {
      tree.dispose(state);
      this.kvState = null;
      this.kvTokens = [];
      return;
    }
    this.kvState = state;
    this.kvTokens = fedTokens;
  }

  /**
   * Measure a full prefill of a synthetic ~voice-turn prompt (median over
   * `runs` fresh states), callable from the browser console to quantify the
   * KV-reuse payoff (compare against the per-turn suffix feed). The first run
   * includes jit trace/compile for this prompt length — the prefill jit
   * specializes on T — so it is reported separately from the median.
   */
  async benchPrefill(nTokens = 250, runs = 5): Promise<Record<string, unknown>> {
    // Real ChatML head + filler user text repeated to the target length, so the
    // measured shape matches a real turn's prompt rather than random ids.
    const tokens = this.encodePrompt([{ role: "user", content: "Hi" }]);
    const filler = this.tokenizer.encode(
      " tell me a little more about the ocean and the weather today",
    );
    while (tokens.length < nTokens) tokens.push(...filler);
    tokens.length = nTokens;

    const times: number[] = [];
    for (let r = 0; r < runs; r++) {
      const state = createSmolLmState({ dtype: np.float16 });
      const t0 = performance.now();
      const logits = runSmolLmPrefill(
        tree.ref(this.model),
        np.array(tokens, { dtype: np.uint32 }),
        state,
      );
      await blockUntilReady(logits.ref);
      times.push(performance.now() - t0);
      logits.dispose();
      tree.dispose(state);
    }
    const sorted = [...times].sort((a, b) => a - b);
    return {
      nTokens,
      runs,
      firstMs: +times[0].toFixed(1),
      medianMs: +sorted[Math.floor(sorted.length / 2)].toFixed(1),
      allMs: times.map((t) => +t.toFixed(1)),
    };
  }

  async *generateStream(
    history: ChatMessage[],
    maxNewTokens = TUNABLES.llmMaxNewTokens,
  ): AsyncGenerator<string, GenerateStats, void> {
    const promptTokens = this.encodePrompt(history);
    const generatedTokens: number[] = [];
    const temperature = 0.7;
    // Seed the repetition penalty with the previous assistant turn's tokens so a
    // "say it differently" follow-up can't echo the same reply verbatim; each
    // freshly generated token is added below.
    const lastReply = [...history]
      .reverse()
      .find((m) => m.role === "assistant");
    const penalize = new Set<number>(
      lastReply ? this.tokenizer.encode(lastReply.content) : [],
    );
    const startTime = performance.now();
    let firstTokenMs = 0;
    let emitted = "";

    // With KV reuse (TUNABLES.llmKvReuse) this only feeds the suffix past the
    // cached prefix; otherwise it's the shipped full prefill on a fresh state.
    const prepared = this.prepareState(promptTokens);
    const { state, fedTokens, persistent } = prepared;
    let pending: np.Array | null = prepared.pending;
    let pendingIsCombined = prepared.pendingIsCombined;
    // Commit-on-throw safety: only a cleanly-ended turn (return OR an early
    // generator.return from barge-in, where every fed token's step has fully
    // run) may persist the cache. A throw can leave `state` half-updated
    // relative to `fedTokens`, so it must be dropped, never installed.
    let failed = false;

    try {
      const stopTokens = [SMOLLM_IM_END, SMOLLM_EOS];
      for (let i = 0; i < maxNewTokens; i++) {
        const sampleable = pending!;
        pending = null;
        const nextToken = pendingIsCombined
          ? await this.sampleFromCombined(sampleable, temperature, penalize)
          : await this.sampleFromLogits(sampleable, temperature, penalize);
        if (i === 0) firstTokenMs = performance.now() - startTime;
        if (stopTokens.includes(nextToken)) break;

        generatedTokens.push(nextToken);
        penalize.add(nextToken);
        const full = this.decodeVisible(generatedTokens);
        const delta = full.startsWith(emitted)
          ? full.slice(emitted.length)
          : full;
        emitted = full;
        if (delta) yield delta;

        if (i === maxNewTokens - 1) break;
        pending = runSmolLmStepFusedTopK(tree.ref(this.model), nextToken, state);
        pendingIsCombined = true;
        fedTokens.push(nextToken);
      }

      return {
        promptTokens: promptTokens.length,
        newTokens: generatedTokens.length,
        firstTokenMs,
        totalMs: performance.now() - startTime,
      };
    } catch (err) {
      failed = true;
      throw err;
    } finally {
      pending?.dispose();
      // Without reuse the state is released here (shipped behavior). With
      // reuse it is persisted for the next turn — unless the turn threw, in
      // which case it is dropped so the next turn rebuilds from scratch.
      if (!persistent) tree.dispose(state);
      else if (failed) tree.dispose(state);
      else this.commitState(state, fedTokens);
    }
  }

  async generate(
    history: ChatMessage[],
    onText: (partial: string) => void,
    maxNewTokens = TUNABLES.llmMaxNewTokens,
  ): Promise<{ text: string; stats: GenerateStats }> {
    let text = "";
    const stream = this.generateStream(history, maxNewTokens);
    let result = await stream.next();
    while (!result.done) {
      text += result.value;
      onText(text);
      result = await stream.next();
    }
    return { text: text.trim(), stats: result.value };
  }

}

/** LLM stage backed by the Cerebras cloud API (as in the original blog post). */
export class CerebrasChatModel implements ChatModel {
  constructor(
    private apiKey: string,
    private modelName: string,
  ) {}

  private async fetchReply(
    history: ChatMessage[],
  ): Promise<{ text: string; stats: GenerateStats }> {
    const startTime = performance.now();
    const response = await fetch(
      "https://api.cerebras.ai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.modelName,
          messages: [
            { role: "system", content: SYSTEM_HINT },
            ...history.map((m) => ({
              role: m.role,
              content:
                m.role === "user" && m.t !== undefined
                  ? `[t+${Math.round(m.t)}s] ${m.content}`
                  : m.content,
            })),
          ],
          max_tokens: 200,
        }),
      },
    );
    if (!response.ok) {
      throw new Error(
        `Cerebras API error ${response.status}: ${await response.text()}`,
      );
    }
    const result = await response.json();
    const text: string = result.choices?.[0]?.message?.content ?? "";
    return {
      text,
      stats: {
        promptTokens: result.usage?.prompt_tokens ?? 0,
        newTokens: result.usage?.completion_tokens ?? 0,
        firstTokenMs: performance.now() - startTime,
        totalMs: performance.now() - startTime,
      },
    };
  }

  async generate(
    history: ChatMessage[],
    onText: (partial: string) => void,
  ): Promise<{ text: string; stats: GenerateStats }> {
    const reply = await this.fetchReply(history);
    onText(reply.text);
    return reply;
  }

  // Cerebras is non-streaming here; wrap the full reply as a one-item stream.
  async *generateStream(
    history: ChatMessage[],
  ): AsyncGenerator<string, GenerateStats, void> {
    const reply = await this.fetchReply(history);
    if (reply.text) yield reply.text;
    return reply.stats;
  }
}

export type SpeakStats = {
  firstAudioMs: number;
  totalMs: number;
  aborted: boolean;
};

export type SpeakOptions = {
  signal?: AbortSignal;
  onAnalyser?: (analyser: AnalyserNode) => void;
};

const TTS_SAMPLE_RATE = 24_000; // Mimi codec output rate.
const BACKCHANNEL_PHRASES = ["Mm-hmm.", "Right.", "Got it."] as const;

export class SpeechSynthesizer {
  private voiceEmbeds = new Map<TTSVoice, np.Array>();
  private backchannels: Float32Array[] = [];
  private backchannelVoice: TTSVoice | null = null;

  private constructor(
    private model: PocketTTS,
    private tokenizer: tokenizers.SentencePiece,
  ) {}

  static async load(onProgress: ProgressFn): Promise<SpeechSynthesizer> {
    const tokData = await fetchWithProgress(
      "Pocket TTS tokenizer",
      `${TTS_HF_PREFIX}/tokenizer.model`,
      onProgress,
    );
    const tokenizer = tokenizers.SentencePiece.fromBinary(tokData);
    const data = await fetchWithProgress(
      "Pocket TTS weights",
      TTS_WEIGHTS_URL,
      onProgress,
    );
    const weights = safetensors.parse(data);
    const model = ttsFromSafetensors(weights);
    return new SpeechSynthesizer(model, tokenizer);
  }

  private async getVoiceEmbed(voice: TTSVoice): Promise<np.Array> {
    let embed = this.voiceEmbeds.get(voice);
    if (!embed) {
      const data = (await cachedFetch(
        `${TTS_HF_PREFIX}/embeddings/${voice}.safetensors`,
      )) as Uint8Array<ArrayBuffer>;
      const audioPrompt = safetensors.parse(data).tensors.audio_prompt;
      embed = np
        .array(audioPrompt.data as Float32Array<ArrayBuffer>, {
          shape: audioPrompt.shape,
          dtype: np.float32,
        })
        .slice(0)
        .astype(np.float16);
      this.voiceEmbeds.set(voice, embed);
    }
    return embed;
  }

  /** Ported from the Python repository (see jax-js tts demo). */
  private prepareTextPrompt(text: string): [string, number] {
    text = text.trim();
    if (text === "") throw new Error("TTS prompt cannot be empty");
    text = text.replace(/\s+/g, " ");
    const numberOfWords = text.split(" ").length;
    const framesAfterEosGuess = numberOfWords <= 4 ? 5 : 3;
    text = text.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());
    if (/[\p{L}\p{N}]$/u.test(text)) text = text + ".";
    if (text.split(" ").length < 5) text = " ".repeat(8) + text;
    return [text, framesAfterEosGuess];
  }

  /** Synthesize one line of text into an existing player (no close). */
  private async synthOne(
    voice: TTSVoice,
    text: string,
    player: AudioPlayer,
    signal: AbortSignal | null,
  ): Promise<void> {
    const [prepared, framesAfterEos] = this.prepareTextPrompt(text);
    const tokens = this.tokenizer.encode(prepared);
    const voiceEmbed = await this.getVoiceEmbed(voice);

    const tokensAr = np.array(tokens, { dtype: np.uint32 });
    let embeds = this.model.flowLM.conditionerEmbed.ref.slice(tokensAr);
    embeds = np.concatenate([voiceEmbed.ref, embeds]);

    // playTTS disposes the model + embeds it receives, so hand it a ref.
    await playTTS(player, tree.ref(this.model), embeds, {
      framesAfterEos,
      seed: null,
      temperature: 0.7,
      lsdDecodeSteps: 1,
      signal,
    });
  }

  /** Synthesize and play `text`, resolving once playback has finished. */
  async speak(
    voice: TTSVoice,
    text: string,
    { signal, onAnalyser }: SpeakOptions = {},
  ): Promise<SpeakStats> {
    const startTime = performance.now();
    let firstAudioMs = 0;
    const inner = createStreamingPlayer();
    onAnalyser?.(inner.analyser);
    const player = withFirstAudio(inner, () => {
      if (firstAudioMs === 0) firstAudioMs = performance.now() - startTime;
    });

    // Mirror speakStream: if an abort lands outside playTTS's frame loop, cut
    // scheduled audio immediately instead of letting it drain via close().
    const onAbort = () => inner.stop();
    signal?.addEventListener("abort", onAbort);

    try {
      await this.synthOne(voice, text, player, signal ?? null);
    } finally {
      if (signal?.aborted) inner.stop();
      // Keep the listener armed WHILE close() drains the scheduled tail — a
      // barge-in during the drain must still cut the audio, not play it out.
      await player.close();
      signal?.removeEventListener("abort", onAbort);
    }
    return {
      firstAudioMs,
      totalMs: performance.now() - startTime,
      aborted: !!signal?.aborted,
    };
  }

  /**
   * Synthesize a stream of sentences sequentially on one shared player/abort,
   * so speech starts right after the first sentence and continues gaplessly.
   */
  async speakStream(
    voice: TTSVoice,
    sentences: AsyncIterable<string>,
    { signal, onAnalyser }: SpeakOptions = {},
  ): Promise<SpeakStats> {
    const startTime = performance.now();
    let firstAudioMs = 0;
    const inner = createStreamingPlayer();
    onAnalyser?.(inner.analyser);
    const player = withFirstAudio(inner, () => {
      if (firstAudioMs === 0) firstAudioMs = performance.now() - startTime;
    });

    // Barge-in can fire while we're between sentences (awaiting the next LLM
    // sentence), where playTTS isn't running to notice the signal — without
    // this listener the already-scheduled audio keeps draining and overlaps
    // the next reply. Cut it the instant the abort lands.
    const onAbort = () => inner.stop();
    signal?.addEventListener("abort", onAbort);

    try {
      for await (const sentence of sentences) {
        if (signal?.aborted) break;
        const line = sentence.trim();
        if (!line) continue;
        await this.synthOne(voice, line, player, signal ?? null);
      }
    } finally {
      if (signal?.aborted) inner.stop();
      // Keep the listener armed WHILE close() drains: the loop exits as soon
      // as the LLM finishes, but seconds of scheduled audio may still be
      // playing — a barge-in during that drain must cut it, not talk over it.
      await player.close();
      signal?.removeEventListener("abort", onAbort);
    }
    return {
      firstAudioMs,
      totalMs: performance.now() - startTime,
      aborted: !!signal?.aborted,
    };
  }

  /**
   * DEV A/B bench hook: synthesize `sentence` once (off the audio graph) and
   * measure pure generation cost, so the harness can compare the fused
   * per-frame decode (TUNABLES.ttsFusedStep) against the shipped path on an
   * identical sentence. A fixed seed makes fused-vs-unfused frame counts (and
   * thus audio duration) comparable. Because the player collects PCM without
   * touching the speakers, `genMs` is the wall-clock of generation alone (GPU
   * dispatch + per-frame EOS readback), not real-time playback.
   *
   * `realtimeFactor = genMs / audioDurationMs`; < 1 means we generate faster
   * than real time (the goal). A `warmup` run (default on) is done first so the
   * timed run does not eat the one-time JIT compile of the fused/unfused traces.
   *
   * Usage: `await window.__pipeline().tts.benchSynth("some sentence.", { fused: true })`
   */
  async benchSynth(
    sentence: string,
    {
      fused = false,
      voice = TTS_VOICES[0],
      seed = 1234,
      warmup = true,
    }: {
      fused?: boolean;
      voice?: TTSVoice;
      seed?: number;
      warmup?: boolean;
    } = {},
  ): Promise<{
    genMs: number;
    firstAudioMs: number;
    audioDurationMs: number;
    realtimeFactor: number;
  }> {
    const [prepared, framesAfterEos] = this.prepareTextPrompt(sentence);
    const tokens = this.tokenizer.encode(prepared);
    const voiceEmbed = await this.getVoiceEmbed(voice);

    const runOnce = async (): Promise<{
      genMs: number;
      firstAudioMs: number;
      audioDurationMs: number;
    }> => {
      const tokensAr = np.array(tokens, { dtype: np.uint32 });
      let embeds = this.model.flowLM.conditionerEmbed.ref.slice(tokensAr);
      embeds = np.concatenate([voiceEmbed.ref, embeds]);

      const inner = createStreamingPlayer({ collectPcm: true });
      let firstAudioMs = 0;
      const start = performance.now();
      const player = withFirstAudio(inner, () => {
        if (firstAudioMs === 0) firstAudioMs = performance.now() - start;
      });
      try {
        await playTTS(player, tree.ref(this.model), embeds, {
          framesAfterEos,
          seed,
          temperature: 0.7,
          lsdDecodeSteps: 1,
          signal: null,
        });
      } finally {
        // playTTS consumed the embeds ref; drain any pending PCM copy.
      }
      const genMs = performance.now() - start;
      const samples = inner.pcm().length;
      await inner.close();
      return {
        genMs,
        firstAudioMs,
        audioDurationMs: (samples / TTS_SAMPLE_RATE) * 1000,
      };
    };

    const prev = TUNABLES.ttsFusedStep;
    TUNABLES.ttsFusedStep = fused;
    try {
      if (warmup) await runOnce();
      const { genMs, firstAudioMs, audioDurationMs } = await runOnce();
      return {
        genMs,
        firstAudioMs,
        audioDurationMs,
        realtimeFactor: audioDurationMs > 0 ? genMs / audioDurationMs : NaN,
      };
    } finally {
      TUNABLES.ttsFusedStep = prev;
    }
  }

  /**
   * Pre-synthesize the backchannel phrases once (off the audio graph) and cache
   * their PCM, so `playBackchannel()` never touches the GPU mid-conversation.
   */
  async prepareBackchannels(voice: TTSVoice): Promise<void> {
    if (this.backchannels.length && this.backchannelVoice === voice) return;
    const clips: Float32Array[] = [];
    for (const phrase of BACKCHANNEL_PHRASES) {
      const collector = createStreamingPlayer({ collectPcm: true });
      try {
        await this.synthOne(voice, phrase, collector, null);
        clips.push(collector.pcm());
      } finally {
        await collector.close();
      }
    }
    this.backchannels = clips;
    this.backchannelVoice = voice;
  }

  /** Play a random cached backchannel instantly through a short-lived context. */
  playBackchannel(): void {
    if (!this.backchannels.length) return;
    const pcm =
      this.backchannels[Math.floor(Math.random() * this.backchannels.length)];
    if (!pcm.length) return;
    const ctx = new AudioContext({ sampleRate: TTS_SAMPLE_RATE });
    const buffer = ctx.createBuffer(1, pcm.length, TTS_SAMPLE_RATE);
    buffer.getChannelData(0).set(pcm);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => void ctx.close().catch(() => {});
    source.start();
  }
}

/** Wrap a player so the first played chunk fires `onFirst()` (for timing). */
function withFirstAudio(inner: AudioPlayer, onFirst: () => void): AudioPlayer {
  return {
    async playChunk(samples) {
      onFirst();
      await inner.playChunk(samples);
    },
    stop: () => inner.stop(),
    close: () => inner.close(),
    toWav: () => inner.toWav(),
    pcm: () => inner.pcm(),
    get context() {
      return inner.context;
    },
    get analyser() {
      return inner.analyser;
    },
  };
}

export type VoicePipeline = {
  asr: SpeechRecognizer;
  llm: ChatModel;
  tts: SpeechSynthesizer;
  /** Compute device the ASR lane runs on ("wasm" when available, else "webgpu"). */
  asrDevice: Device;
  /** True if ASR runs on wasm, i.e. concurrently with WebGPU TTS/LLM. */
  dualLane: boolean;
};

export type DeviceSetup = {
  devices: Device[];
  asrDevice: Device;
  asrDtype: np.DType;
};

/**
 * Initialize backends. WebGPU is required (all stages). ASR runs on WebGPU too
 * (fp16): a Whisper pass there is ~250 ms vs ~1.6 s on the wasm CPU lane, so
 * live captions actually keep up. The streaming ASR loop is paused while the
 * assistant speaks (barge-in is energy-based, not ASR-based), so it never
 * contends with TTS generation on the GPU.
 */
export async function initDevice(): Promise<DeviceSetup> {
  let devices: Device[];
  try {
    devices = await init("webgpu", "wasm");
  } catch {
    devices = await init("webgpu");
  }
  if (!devices.includes("webgpu")) {
    throw new Error(
      "WebGPU is not available in this browser; it is required to run the models.",
    );
  }
  defaultDevice("webgpu");
  return { devices, asrDevice: "webgpu", asrDtype: np.float16 };
}

export async function loadPipeline(
  onProgress: ProgressFn,
): Promise<VoicePipeline> {
  const setup = await initDevice();
  // All three weight downloads run in parallel (they were sequential; HTTP/2
  // multiplexes them over one connection, so wall-clock ≈ the largest file).
  const [asr, llm, tts] = await Promise.all([
    SpeechRecognizer.load(onProgress, {
      device: setup.asrDevice,
      dtype: setup.asrDtype,
    }),
    SmolLmChatModel.load(onProgress),
    SpeechSynthesizer.load(onProgress),
  ]);
  // Compile the ASR + LLM kernels now (esp. slow to JIT on wasm) so the first
  // real turn isn't hit with a multi-second cold start. (TTS is warmed via
  // prepareBackchannels, which the UI calls right after load.)
  onProgress({ name: "Warming up models", loadedBytes: 0, done: false });
  await asr.warmup();
  await llm.warmup();
  onProgress({ name: "Warming up models", loadedBytes: 1, done: true });
  return {
    asr,
    llm,
    tts,
    asrDevice: setup.asrDevice,
    dualLane: setup.asrDevice === "wasm",
  };
}
