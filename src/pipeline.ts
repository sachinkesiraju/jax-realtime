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
  runGemmaPrefill,
  runGemmaStep,
  runGemmaStepFused,
} from "./llm/gemma";
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

const WHISPER_CONFIG: WhisperConfig = WHISPER_MODELS.find(
  (m) => m.id === "tiny.en",
)!;
const ASR_MAX_NEW_TOKENS = 96;

const GEMMA_BASE =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m";
const GEMMA_START_OF_TURN = 105;
const GEMMA_END_OF_TURN = 106;

const TTS_WEIGHTS_URL =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";
// Locally-hosted Gemma build with the tied embedding table quantized to int8
// (dequantized to fp16 at load). Served from public/; absent in fresh clones,
// in which case the loader falls back to the fp16 HF file above.
const GEMMA_Q8_URL = "/weights/gemma-it-q8e.safetensors";
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
    // Prefer the locally-hosted int8-embedding build (536 → 369 MB download;
    // dequantized to fp16 at load, so runtime is unchanged). Fall back to the
    // fp16 HF file when it isn't hosted. The HEAD probe guards against the dev
    // server's SPA fallback answering 200 with index.html for a missing file.
    let weightsUrl = `${GEMMA_BASE}/model-it-fp16.safetensors`;
    let weightsLabel = "Gemma 3 270M weights";
    try {
      const head = await fetch(GEMMA_Q8_URL, { method: "HEAD" });
      const size = Number(head.headers.get("content-length") ?? 0);
      if (head.ok && size > 100_000_000) {
        weightsUrl = GEMMA_Q8_URL;
        weightsLabel = "Gemma 3 270M weights (int8 embed)";
      }
    } catch {
      // No local quantized build; use the fp16 original.
    }
    const data = await fetchWithProgress(weightsLabel, weightsUrl, onProgress);
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
    opts?: { fused?: boolean; sampler?: "js" | "topk" },
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
    { fused = false, sampler = "js" as "js" | "topk" },
  ): Promise<Record<string, unknown>> {
    const prompt = this.benchPrompt();
    const stepFn = fused ? runGemmaStepFused : runGemmaStep;
    const opts = { temperature: 0.7, topK: 64, topP: 0.95 };
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

    const state = createGemmaState({ dtype: np.float16 });
    let logits = runGemmaPrefill(
      tree.ref(this.model),
      np.array(prompt, { dtype: np.uint32 }),
      state,
    );

    const dispatchMs: number[] = [];
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      let tok: number;
      if (sampler === "topk") {
        const [vals, idx] = lax.topK(logits, 64); // consumes logits
        const v = (await vals.data()) as ArrayLike<number>;
        const ix = (await idx.data()) as ArrayLike<number>;
        tok = sampleTopKPairs(v, ix, opts);
      } else {
        tok = sampleLogits((await logits.data()) as Float32Array, opts);
      }
      const d0 = performance.now();
      logits = stepFn(tree.ref(this.model), tok, state);
      dispatchMs.push(performance.now() - d0);
    }
    await blockUntilReady(logits.ref);
    const msPerTok = (performance.now() - t0) / n;
    logits.dispose();
    tree.dispose(state);

    return {
      config: { fused, sampler },
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
    ): Promise<number[]> => {
      const stepFn = fused ? runGemmaStepFused : runGemmaStep;
      const state = createGemmaState({ dtype: np.float16 });
      let logits = runGemmaPrefill(
        tree.ref(this.model),
        np.array(prompt, { dtype: np.uint32 }),
        state,
      );
      const out: number[] = [];
      try {
        for (let i = 0; i < nTokens; i++) {
          let tok: number;
          if (sampler === "topk") {
            const [vals, idx] = lax.topK(logits, 64); // consumes logits
            const v = (await vals.data()) as ArrayLike<number>;
            const ix = (await idx.data()) as ArrayLike<number>;
            tok = sampleTopKPairs(v, ix, greedy);
          } else {
            tok = sampleLogits((await logits.data()) as Float32Array, greedy);
          }
          out.push(tok);
          logits = stepFn(tree.ref(this.model), tok, state);
        }
      } finally {
        logits.dispose();
        tree.dispose(state);
      }
      return out;
    };

    return {
      jsUnfused: await run(false, "js"),
      jsFused: await run(true, "js"),
      topkUnfused: await run(false, "topk"),
      topkFused: await run(true, "topk"),
    };
  }

  private windowHistory(history: ChatMessage[]): ChatMessage[] {
    const n = TUNABLES.llmMaxHistoryTurns;
    if (n <= 0 || history.length <= n) return history;
    // Keep the last N messages, but never start on an orphaned assistant reply
    // — drop it so the window always begins on a user turn (whole pairs).
    let sliced = history.slice(history.length - n);
    if (sliced[0]?.role === "assistant") sliced = sliced.slice(1);
    return sliced;
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
    history = this.windowHistory(history);
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
    const startTime = performance.now();
    let firstTokenMs = 0;
    let emitted = "";

    const prepared = this.prepareState(promptTokens);
    const { state, fedTokens, persistent } = prepared;
    let logits: np.Array | null = prepared.logits;

    try {
      const stopTokens = [this.tokenizer.eosToken, GEMMA_END_OF_TURN];

      for (let i = 0; i < maxNewTokens; i++) {
        const sampledLogits = logits!;
        logits = null;
        // The plain chat template (no instruction preamble) responds best at a
        // natural sampling temperature, matching the jax-js chat demo.
        const nextToken = await this.sampleNext(sampledLogits, 0.7);
        if (i === 0) firstTokenMs = performance.now() - startTime;
        if (stopTokens.includes(nextToken)) break;

        generatedTokens.push(nextToken);
        const full = this.decodeVisible(generatedTokens);
        const delta = full.startsWith(emitted) ? full.slice(emitted.length) : full;
        emitted = full;
        if (delta) yield delta;

        if (i === maxNewTokens - 1) break;
        logits = stepFn(tree.ref(this.model), nextToken, state);
        fedTokens.push(nextToken);
      }

      return {
        promptTokens: promptTokens.length,
        newTokens: generatedTokens.length,
        firstTokenMs,
        totalMs: performance.now() - startTime,
      };
    } finally {
      logits?.dispose();
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

/** The TTS surface the app consumes (duplex engine + UI). */
export type TTSStage = Pick<
  SpeechSynthesizer,
  "speak" | "speakStream" | "prepareBackchannels" | "playBackchannel"
>;

export type VoicePipeline = {
  asr: SpeechRecognizer;
  llm: LocalChatModel;
  tts: TTSStage;
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

/**
 * Deferred TTS handle: the ~236 MB voice model is the biggest single chunk of
 * the download but isn't needed to *start talking* — only for the first reply.
 * The pipeline flips to ready once ASR+LLM are warm; this wrapper transparently
 * awaits the still-downloading TTS on first use, and backchannels simply no-op
 * until it has landed.
 */
class DeferredTTS implements TTSStage {
  private real: SpeechSynthesizer | null = null;

  constructor(private readonly loading: Promise<SpeechSynthesizer>) {
    void loading.then((tts) => {
      this.real = tts;
    });
  }

  async speak(
    ...args: Parameters<SpeechSynthesizer["speak"]>
  ): ReturnType<SpeechSynthesizer["speak"]> {
    return (this.real ?? (await this.loading)).speak(...args);
  }

  async speakStream(
    ...args: Parameters<SpeechSynthesizer["speakStream"]>
  ): ReturnType<SpeechSynthesizer["speakStream"]> {
    return (this.real ?? (await this.loading)).speakStream(...args);
  }

  async prepareBackchannels(voice: TTSVoice): Promise<void> {
    return (await this.loading).prepareBackchannels(voice);
  }

  playBackchannel(): void {
    // Silent no-op while the voice model is still downloading.
    this.real?.playBackchannel();
  }
}

export async function loadPipeline(
  onProgress: ProgressFn,
): Promise<VoicePipeline> {
  const setup = await initDevice();
  // All three weight downloads run in parallel (they were sequential; HTTP/2
  // multiplexes them over one connection, so wall-clock ≈ the largest file).
  const asrP = SpeechRecognizer.load(onProgress, {
    device: setup.asrDevice,
    dtype: setup.asrDtype,
  });
  const llmP = LocalChatModel.load(onProgress);
  const ttsP = SpeechSynthesizer.load(onProgress);
  const [asr, llm] = await Promise.all([asrP, llmP]);
  // Compile the ASR + LLM kernels now (esp. slow to JIT on wasm) so the first
  // real turn isn't hit with a multi-second cold start. TTS is deliberately
  // NOT awaited: readiness only needs ears + brain, and DeferredTTS awaits the
  // voice on first reply (prepareBackchannels doubles as its warmup once it
  // lands — main.ts fires that in the background).
  onProgress({ name: "Warming up models", loadedBytes: 0, done: false });
  await asr.warmup();
  await llm.warmup();
  onProgress({ name: "Warming up models", loadedBytes: 1, done: true });
  return {
    asr,
    llm,
    tts: new DeferredTTS(ttsP),
    asrDevice: setup.asrDevice,
    dualLane: setup.asrDevice === "wasm",
  };
}
