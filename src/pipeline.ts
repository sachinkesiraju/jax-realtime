// Speech-to-speech pipeline: Whisper ASR -> SmolLM2 LLM -> Kyutai Pocket TTS.
// All three stages run locally in the browser on WebGPU via jax-js, mirroring
// the architecture of the HF/Cerebras real-time voice AI demo.

import {
  blockUntilReady,
  defaultDevice,
  type Device,
  init,
  lax,
  numpy as np,
  random,
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
  createSmolLmState,
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
  createFlowLMState,
  fromSafetensors as ttsFromSafetensors,
  type PocketTTS,
  runFlowLMStep,
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

const TTS_WEIGHTS_URL =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";
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

// System instructions for the cloud (Cerebras) brain; the local SmolLM brain
// carries its own SMOLLM_SYSTEM. Keep this SHORT and POSITIVE: small models
// can't follow long instructions or negation (naming a phrase to avoid just
// primes them to say that phrase).
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
    // Memory hygiene: the download (~150 MB) and the parsed File are only
    // needed until the tensors are uploaded to the device. safetensors.parse
    // returns typed-array VIEWS into the download's ArrayBuffer (zero-copy),
    // and whisperFromSafetensors copies every tensor into backend memory
    // EAGERLY (np.array → backend.malloc, which writeBuffer/memcpy's at call
    // time) before its first await — so once it returns, nothing lazy points
    // at the download. Null both locals so this async frame (long-lived: the
    // three model loads run concurrently under Promise.all) can't keep the
    // buffer reachable a moment longer than model construction.
    let data: Uint8Array<ArrayBuffer> | null = await fetchWithProgress(
      `Whisper ${WHISPER_CONFIG.label} weights`,
      hfWhisperUrl("model.safetensors"),
      onProgress,
    );
    let weights: safetensors.File | null = safetensors.parse(data);
    data = null; // views inside `weights` keep the buffer alive until upload
    const model = await whisperFromSafetensors(
      weights,
      resolvedDtype,
      WHISPER_CONFIG,
      device,
    );
    weights = null; // last reference to the download buffer's views
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
   * all layers → norm → logits in one dispatch), so there is no per-layer
   * "unfused" path to A/B against — this just reports the ms/token of
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

// SmolLM2-360M-Instruct brain, chosen via a blind-judged model shootout over
// same-size and larger alternatives (winning every dimension across three
// runs). Weights + a precomputed tiktoken-style tokenizer artifact are hosted
// on Hugging Face (CORS-open).
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
// SmolLM2 honors a real ChatML system role. A spoken-format prompt.
const SMOLLM_SYSTEM =
  "You are a warm, helpful voice assistant. Answer directly in a natural, " +
  "spoken style — a sentence or two, no lists, bullet points, or markdown. A " +
  "[scene: …] tag tells you what the camera sees; never read a bracketed tag " +
  "aloud.";
// Optional clarify-on-garble instruction (TUNABLES.qualityGarbleClause).
// Observed live failure: speech recognition sometimes hands the brain
// gibberish ("whazzit fmm the uh...") and a 360M model answers it CONFIDENTLY
// instead of asking for a repeat. One sentence, positive phrasing ("say X"
// rather than "don't guess" — naming a behavior to avoid primes small models
// to do it), appended to SMOLLM_SYSTEM at prompt-encode time so the bench can
// flip the tunable between sessions without a reload.
const SMOLLM_GARBLE_CLAUSE =
  "If the user's message is garbled or doesn't make sense, say you didn't " +
  "catch that and ask them to say it again.";

type SmolLmTokenizerData = {
  encoder: Record<string, number>;
  special: Record<string, number>;
  pattern: string;
};

/**
 * SmolLM2 brain implementing the ChatModel interface, so the duplex engine is
 * agnostic to which model backs it. Each decode step is a fused single-dispatch
 * jit with the top-k reduction folded in (one small readback per token).
 */
export class SmolLmChatModel implements ChatModel {
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

    let data: Uint8Array<ArrayBuffer> | null;
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
    // Memory hygiene: `data` is 363 MB (int8) / 724 MB (fp16) and the parsed
    // File's tensors are zero-copy views into it. smolLmFromSafetensors
    // materializes every weight EAGERLY before its first await — int8 tensors
    // are dequantized into a fresh Float16Array and fp16 tensors are copied
    // into backend memory by np.array (backend.malloc copies at call time) —
    // so after it resolves the download buffer backs nothing. Null the locals
    // so this frame (alive throughout loadPipeline's Promise.all) doesn't pin
    // an extra copy of the largest download next to the GPU-resident weights.
    let weights: safetensors.File | null = safetensors.parse(data);
    data = null; // views inside `weights` keep the buffer alive until upload
    const model = await smolLmFromSafetensors(weights, np.float16);
    weights = null; // last reference to the download buffer's views
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
    // System string is assembled HERE (not at module scope) so the garble
    // clause reflects the tunable's value at generation time — the quality
    // bench flips it per-session without reloading the model.
    turn(
      "system",
      TUNABLES.qualityGarbleClause
        ? `${SMOLLM_SYSTEM} ${SMOLLM_GARBLE_CLAUSE}`
        : SMOLLM_SYSTEM,
    );
    for (const message of windowHistory(history)) {
      const content = message.content.trim();
      if (content === "") continue;
      turn(message.role === "assistant" ? "assistant" : "user", content);
    }
    tokens.push(SMOLLM_IM_START, ...enc("assistant\n"));
    return tokens;
  }

  // Lazily-built set of token ids whose decoded text contains formatting junk
  // a voice assistant must never SPEAK: [ ] * # ` (markdown emphasis/headers/
  // code fences and the "[activity]" template-placeholder pattern seen live).
  // Built by scanning the FULL vocabulary — BPE merges mean the junk hides in
  // multi-char tokens (" [", "](", "**", "###"), so encoding "[" alone would
  // miss most of them. `tokenizer.decoder` is BpeEncoding's id→bytes map (the
  // exact non-special vocab, ~49k entries), and decode([id]) turns each entry
  // into text; the banned characters are all single-byte ASCII, so they
  // survive decoding even inside tokens that are otherwise partial UTF-8.
  // One pass at first use (~49k tiny decodes), then cached for the session.
  // Measured on the shipped artifact: 239 of 49135 vocab ids banned, e.g.
  // "**", " [", "](", "###", "``" — and ZERO of them contain a letter, so no
  // ordinary word is collateral damage (verified offline against
  // smollm2-360m-tokenizer.json). Deliberately NOT banned: apostrophes/
  // quotes/parens/dashes — legitimate in speech — and special ids
  // (<|im_end|> must stay sampleable as the stop token).
  private bannedFormatIds: Set<number> | null = null;

  private getBannedFormatIds(): Set<number> {
    if (this.bannedFormatIds) return this.bannedFormatIds;
    const junk = /[[\]*#`]/;
    const banned = new Set<number>();
    for (const id of this.tokenizer.decoder.keys()) {
      if (junk.test(this.tokenizer.decode([id]))) banned.add(id);
    }
    this.bannedFormatIds = banned;
    return banned;
  }

  // Hard mask (TUNABLES.qualityBanFormatTokens): set banned candidates' logits
  // to -Infinity so the temperature/top-p draw can NEVER pick them — the
  // system prompt already says "no lists ... or markdown" and the model emits
  // them anyway, so this failure needs a mechanism, not more instructions.
  // exp(-Inf - max) = 0, so a banned candidate gets zero probability mass and
  // the nucleus renormalizes over what remains. Safe for the prompt-side
  // "[scene: …]" tag: the mask applies only to SAMPLED output tokens. In the
  // degenerate case where every top-k candidate is banned (never observed;
  // k = 64), sampleFromCandidates falls back to the argmax — a banned token —
  // rather than throwing, which is the right failure mode mid-conversation.
  private maskFormatTokens(
    values: number[],
    indices: ArrayLike<number>,
  ): number[] {
    if (!TUNABLES.qualityBanFormatTokens) return values;
    const banned = this.getBannedFormatIds();
    for (let j = 0; j < values.length; j++) {
      if (banned.has(indices[j])) values[j] = -Infinity;
    }
    return values;
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
    // Ban mask AFTER the repetition penalty: -Inf is a fixed point of the
    // penalty math, but ordering it last makes "banned means banned" obvious.
    return sampleTopKPairs(
      this.maskFormatTokens(this.penalize(values, indices, penalize), indices),
      indices,
      opts,
    );
  }

  private async sampleFromLogits(
    logits: np.Array,
    temperature: number,
    penalize: ReadonlySet<number> = EMPTY_SET,
  ): Promise<number> {
    const [vals, idx] = lax.topK(logits, SMOLLM_TOPK); // consumes logits
    const v = Array.from((await vals.data()) as ArrayLike<number>);
    const ix = (await idx.data()) as ArrayLike<number>;
    return sampleTopKPairs(
      this.maskFormatTokens(this.penalize(v, ix, penalize), ix),
      ix,
      {
        temperature,
        topK: SMOLLM_TOPK,
        topP: 0.95,
      },
    );
  }

  private decodeVisible(tokens: number[]): string {
    return this.tokenizer.decode(tokens.filter((t) => !this.specialIds.has(t)));
  }

  /**
   * Full prefill, optionally bucket-padded (TUNABLES.llmPrefillBucket > 0):
   * pad the prompt UP to the next multiple of `bucket` so the prefill jits'
   * trace shapes (keyed on T) repeat across turns instead of re-tracing all
   * 32 layers for every new prompt length. Exactness argument (verified
   * against runSmolLmStep/runSmolLmDecodeStepFused): pads sit at the END, so
   * every real token's causal attention sees only real tokens; the logits are
   * gathered at realLength-1 (a real token). Pad queries produce garbage
   * outputs (discarded) and garbage KV in slots [realLength, paddedLen) — but
   * runSmolLmPrefill sets state.position = realLength, every decode step's
   * validMask admits only slots < position+1, and each step overwrites
   * slot == position before position advances past it, so a garbage slot is
   * always overwritten before it becomes attendable. Wrong RoPE angles on pad
   * positions are irrelevant for the same reason. Pad id: <|im_end|> (any
   * valid embedding row works; this is SmolLM2's eos/pad token).
   *
   * `promptTokens` is never mutated — callers keep it as the REAL token list
   * (fedTokens/kvTokens must never contain padding).
   */
  private runBucketedPrefill(
    promptTokens: number[],
    state: SmolLmState,
    bucket = TUNABLES.llmPrefillBucket,
  ): np.Array {
    const realLength = promptTokens.length;
    let ids = promptTokens;
    if (bucket > 0 && realLength % bucket !== 0) {
      const paddedLen = Math.ceil(realLength / bucket) * bucket;
      ids = promptTokens.concat(
        new Array<number>(paddedLen - realLength).fill(SMOLLM_IM_END),
      );
    }
    return runSmolLmPrefill(
      tree.ref(this.model),
      np.array(ids, { dtype: np.uint32 }),
      state,
      realLength,
    );
  }

  // NOTE (cycle 6): cross-turn KV-cache reuse was built here (prefix-match +
  // token-by-token suffix feed through the fused decode step) and REJECTED at
  // MAP: every fused step
  // is a full GPU submit→execute→sync roundtrip (~40-60 ms), so re-feeding a
  // reply-sized suffix (30-100 tokens) costs seconds — measured llmFirst
  // medians went 674-1002 ms → 4786 ms, with a 16 s spike when a KV-capacity
  // grow re-traced the fused jit on the critical path. The family is removed,
  // removed rather than flag-gated (the repo's rule for proven dead-ends): it
  // cannot pay until a batched offset-prefill (prefix-aware attention mask in
  // smollm.ts) exists, and bucketed prefill (llmPrefillBucket) already cut the
  // cost it targeted.

  /**
   * Measure a full prefill of a synthetic ~voice-turn prompt (median over
   * `runs` fresh states), callable from the browser console. The first run
   * includes jit trace/compile for this prompt length — the prefill jit
   * specializes on T — so it is reported separately from the median.
   *
   * `bucket` (default: TUNABLES.llmPrefillBucket) exercises the bucket-padded
   * path: compare firstMs at, e.g., nTokens=250 vs 251 with bucket=64 (same
   * padded shape → the second length's first run is already warm) against
   * bucket=0 (every new length pays the ~700 ms re-trace).
   */
  async benchPrefill(
    nTokens = 250,
    runs = 5,
    opts: { bucket?: number } = {},
  ): Promise<Record<string, unknown>> {
    const bucket = opts.bucket ?? TUNABLES.llmPrefillBucket;
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
      const logits = this.runBucketedPrefill(tokens, state, bucket);
      await blockUntilReady(logits.ref);
      times.push(performance.now() - t0);
      logits.dispose();
      tree.dispose(state);
    }
    const sorted = [...times].sort((a, b) => a - b);
    return {
      nTokens,
      runs,
      bucket,
      firstMs: +times[0].toFixed(1),
      medianMs: +sorted[Math.floor(sorted.length / 2)].toFixed(1),
      allMs: times.map((t) => +t.toFixed(1)),
    };
  }

  /**
   * Equivalence gate for the bucketed prefill: run the same prompt through the
   * unbucketed (shipped) and bucketed paths and compare full-logits argmax and
   * max |Δ|. The bucketing argument (pad tokens never attendable, logits read
   * at the last REAL token) predicts bit-identical output — this verifies it
   * on-device before the tunable is allowed to default on.
   */
  async benchPrefillEquivalence(
    nTokens = 250,
    bucket = 64,
  ): Promise<Record<string, unknown>> {
    const tokens = this.encodePrompt([{ role: "user", content: "Hi" }]);
    const filler = this.tokenizer.encode(
      " tell me a little more about the ocean and the weather today",
    );
    while (tokens.length < nTokens) tokens.push(...filler);
    tokens.length = nTokens;

    const readLogits = async (b: number): Promise<Float32Array> => {
      const state = createSmolLmState({ dtype: np.float16 });
      const logits = this.runBucketedPrefill(tokens, state, b);
      // astype consumes `logits` (move semantics); data() drains the result.
      const data = new Float32Array(await logits.astype(np.float32).data());
      tree.dispose(state);
      return data;
    };
    const base = await readLogits(0);
    const bucketed = await readLogits(bucket);
    let maxAbsDiff = 0;
    let argmaxBase = 0;
    let argmaxBucketed = 0;
    for (let i = 0; i < base.length; i++) {
      const d = Math.abs(base[i] - bucketed[i]);
      if (d > maxAbsDiff) maxAbsDiff = d;
      if (base[i] > base[argmaxBase]) argmaxBase = i;
      if (bucketed[i] > bucketed[argmaxBucketed]) argmaxBucketed = i;
    }
    return {
      nTokens,
      bucket,
      argmaxMatch: argmaxBase === argmaxBucketed,
      argmaxBase,
      argmaxBucketed,
      maxAbsDiff,
    };
  }

  async *generateStream(
    history: ChatMessage[],
    maxNewTokens = TUNABLES.llmMaxNewTokens,
  ): AsyncGenerator<string, GenerateStats, void> {
    const promptTokens = this.encodePrompt(history);
    const generatedTokens: number[] = [];
    // Read per-generation (not hoisted to a const) so the quality bench can
    // A/B temperature between turns without reloading the model. Shipped 0.7;
    // lower values are a candidate fix for observed rambling, traded against
    // verbatim-repeat and dull-answer risk (see tunables.ts).
    const temperature = TUNABLES.qualityTemperature;
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

    const state = createSmolLmState({ dtype: np.float16 });
    let pending: np.Array | null = this.runBucketedPrefill(promptTokens, state);
    let pendingIsCombined = false;

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
      }

      return {
        promptTokens: promptTokens.length,
        newTokens: generatedTokens.length,
        firstTokenMs,
        totalMs: performance.now() - startTime,
      };
    } finally {
      pending?.dispose();
      tree.dispose(state);
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
  /** First SYNTHESIZED chunk scheduled (the onset filler never counts here —
   *  the bench's firstAudio stat must keep meaning real reply audio, or an
   *  onsetFiller run would silently inflate its own latency numbers). */
  firstAudioMs: number;
  /** Pre-rendered onset filler chunk scheduled (0 = none played). */
  onsetAudioMs: number;
  totalMs: number;
  aborted: boolean;
};

export type SpeakOptions = {
  signal?: AbortSignal;
  onAnalyser?: (analyser: AnalyserNode) => void;
  /** Fired when an onset filler is scheduled, with its phrase text, so the
   *  duplex layer can fold the words into the ASR self-echo filter (the
   *  filler is audible speech the mic may pick up, even though it is never
   *  part of the reply text/history). */
  onOnset?: (text: string) => void;
};

const TTS_SAMPLE_RATE = 24_000; // Mimi codec output rate.
const BACKCHANNEL_PHRASES = ["Mm-hmm.", "Right.", "Got it."] as const;
// Onset fillers: spoken at reply start to mask real turn latency (see
// TUNABLES.onsetFiller). Deliberately open-ended lead-ins (trailing comma /
// "so") rather than complete words like the backchannels — they must sound
// like the start of the sentence that follows, not a finished acknowledgment.
const ONSET_PHRASES = ["So,", "Right,", "Okay, so"] as const;

export class SpeechSynthesizer {
  private voiceEmbeds = new Map<TTSVoice, np.Array>();
  private backchannels: Float32Array[] = [];
  private backchannelVoice: TTSVoice | null = null;
  // Onset fillers cached per-voice exactly like the backchannels: PCM only,
  // pre-rendered at load so speakStream never touches the GPU for them (the
  // GPU is busy with the LLM prefill at exactly the moment the onset plays).
  private onsets: { text: string; pcm: Float32Array }[] = [];
  private onsetVoice: TTSVoice | null = null;
  // Rotate phrases so consecutive replies don't all open with the same word
  // (a fixed "So," on every turn reads as a tic, not a natural lead-in).
  private lastOnsetIdx = -1;

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
    // Memory hygiene (same pattern as the ASR/LLM loaders): the parsed File's
    // tensors are zero-copy views into the download buffer, and the TTS
    // fromSafetensors is fully SYNCHRONOUS — every np.array copies into
    // backend memory before it returns — so the moment `model` exists the
    // download backs nothing. Null the locals so this async frame (concurrent
    // with the other two loads under Promise.all) releases the buffer instead
    // of holding it next to the GPU-resident weights.
    let data: Uint8Array<ArrayBuffer> | null = await fetchWithProgress(
      "Pocket TTS weights",
      TTS_WEIGHTS_URL,
      onProgress,
    );
    let weights: safetensors.File | null = safetensors.parse(data);
    data = null; // views inside `weights` keep the buffer alive until upload
    const model = ttsFromSafetensors(weights);
    weights = null; // last reference to the download buffer's views
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

  /**
   * Encode `prepared` (output of prepareTextPrompt), padded with LEADING
   * spaces until the token count reaches the next multiple of `bucket`
   * (TUNABLES.ttsPrefillBucket; <= 0 or already aligned = plain encode). WHY:
   * jax-js trace caches key on avals (shapes), and the flow-LM's step-0
   * prefill (always the unfused runFlowLMStep — inference.ts keeps step 0 off
   * the fused path) runs its 6 jitted streaming-transformer layers + the
   * jitted out-norm over a [voiceLen + textLen + 1, 1024] activation. A new
   * sentence token count means re-tracing them all before the reply's first
   * audio chunk; bucketing textLen makes warm shapes repeat across sentences
   * (the LLM's llmPrefillBucket lever, ported to the TTS prefill).
   *
   * Pad choice — real spaces through the real tokenizer, at the START:
   *   - Spaces, because prepareTextPrompt already fronts every <5-word phrase
   *     with 8 spaces (Kyutai's reference behavior), so leading whitespace is
   *     the one padding this model demonstrably treats as neutral — it adds
   *     no leading silence beyond what shipped synthesis already produces.
   *   - Start-side, because the flow-LM conditions its EOS decision and
   *     sentence-final prosody on the END of the text (the tokens adjacent to
   *     the latent positions, which follow the embeds in the prefill).
   *     Trailing spaces after the final "." are an arrangement the model
   *     never saw and risk trailing artifacts / shifted EOS timing; leading
   *     spaces before a capitalized sentence are exactly the training-time
   *     arrangement.
   *   - Re-encoding the space-padded TEXT (instead of splicing a pad token id
   *     into the token array) keeps the sequence exactly what the tokenizer
   *     itself produces for a space-padded sentence — in-distribution
   *     segmentation, no assumptions about which "▁" pieces exist.
   * The readout cannot move: runFlowLMStep slices position -1, the BOS latent
   * appended AFTER the text embeds, so padding never changes which position
   * the latent/EOS are read from — only prepends neutral context.
   *
   * SentencePiece merges make token count non-additive in the space count, so
   * we re-encode and adjust; if the exact target is unreachable (e.g. a
   * tokenizer that collapses whitespace) we fall back to the unpadded
   * encoding with a warning — correctness over trace warmth, never a silent
   * behavior change.
   */
  private encodeTextBucketed(
    prepared: string,
    bucket = TUNABLES.ttsPrefillBucket,
  ): number[] {
    const tokens = this.tokenizer.encode(prepared);
    if (bucket <= 0 || tokens.length % bucket === 0) return tokens;
    let target = Math.ceil(tokens.length / bucket) * bucket;
    let spaces = target - tokens.length;
    for (let tries = 0; tries < 8 && spaces > 0; tries++) {
      const padded = this.tokenizer.encode(" ".repeat(spaces) + prepared);
      if (padded.length === target) return padded;
      spaces += target - padded.length;
      if (spaces <= 0) {
        // Merged space pieces overshot the target for every smaller space
        // count; aim one bucket higher rather than under-pad (an unaligned
        // length would silently defeat the bucket).
        target += bucket;
        spaces += bucket;
      }
    }
    console.warn(
      `ttsPrefillBucket: could not pad TTS prompt to ${target} tokens; ` +
        "synthesizing unpadded (re-trace possible)",
    );
    return tokens;
  }

  /** Synthesize one line of text into an existing player (no close). */
  private async synthOne(
    voice: TTSVoice,
    text: string,
    player: AudioPlayer,
    signal: AbortSignal | null,
  ): Promise<void> {
    const [prepared, framesAfterEos] = this.prepareTextPrompt(text);
    const tokens = this.encodeTextBucketed(prepared);
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
      onsetAudioMs: 0, // onsets are for real replies only, never speak()
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
    { signal, onAnalyser, onOnset }: SpeakOptions = {},
  ): Promise<SpeakStats> {
    const startTime = performance.now();
    let firstAudioMs = 0;
    let onsetAudioMs = 0;
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
      // Onset filler (cycle-3 law: ONE gapless stream on ONE clock). The
      // cached PCM is scheduled through THE SAME player as the reply, so the
      // first synthesized chunk appends on the player's nextStartTime clock —
      // dead air between filler and reply is possible (acceptable), audible
      // overlap/clipping is structurally impossible (the failure mode of the
      // reverted two-context attempt, docs/BENCHMARKS.md Campaign A). Played
      // via `inner`, NOT the withFirstAudio wrapper: firstAudioMs must keep
      // meaning the first SYNTHESIZED chunk for the bench. Registered abort
      // listener above already covers it: inner.stop() cuts every live
      // source, onset included, so a barge-in mid-filler goes silent too.
      if (
        TUNABLES.onsetFiller &&
        this.onsetVoice === voice &&
        this.onsets.length > 0 &&
        !signal?.aborted
      ) {
        const pick = this.pickOnset();
        await inner.playChunk(pick.pcm);
        onsetAudioMs = performance.now() - startTime;
        onOnset?.(pick.text);
      }

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
      onsetAudioMs,
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
      bucket = TUNABLES.ttsPrefillBucket,
    }: {
      fused?: boolean;
      voice?: TTSVoice;
      seed?: number;
      warmup?: boolean;
      /** ttsPrefillBucket override for A/B (0 = unpadded, shipped). */
      bucket?: number;
    } = {},
  ): Promise<{
    genMs: number;
    firstAudioMs: number;
    audioDurationMs: number;
    realtimeFactor: number;
    /** Trace-cold numbers from the warmup run (absent when warmup=false).
     *  cold − warm firstAudio at a NEW sentence length is exactly the step-0
     *  re-trace cost that ttsPrefillBucket targets, so the A/B can show it
     *  without a page reload. */
    coldGenMs?: number;
    coldFirstAudioMs?: number;
    textTokens: number;
    paddedTokens: number;
    bucket: number;
  }> {
    const [prepared, framesAfterEos] = this.prepareTextPrompt(sentence);
    const tokens = this.encodeTextBucketed(prepared, bucket);
    const textTokens = this.tokenizer.encode(prepared).length;
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
      const cold = warmup ? await runOnce() : undefined;
      const { genMs, firstAudioMs, audioDurationMs } = await runOnce();
      return {
        genMs,
        firstAudioMs,
        audioDurationMs,
        realtimeFactor: audioDurationMs > 0 ? genMs / audioDurationMs : NaN,
        coldGenMs: cold?.genMs,
        coldFirstAudioMs: cold?.firstAudioMs,
        textTokens,
        paddedTokens: tokens.length,
        bucket,
      };
    } finally {
      TUNABLES.ttsFusedStep = prev;
    }
  }

  /**
   * DEV probe for TUNABLES.ttsPrefillBucket: time ONLY the flow-LM's step-0
   * prefill — the sole part of TTS whose jit traces key on the sentence's
   * token count — for each sentence, `repeats` times on a fresh state each.
   * The first run of a NEW prefill shape pays trace+compile for the 6
   * streaming-transformer layers + out-norm; later runs are warm. With
   * bucket > 0, sentences whose padded lengths land in the same bucket share
   * one shape, so every sentence after the first should open warm — with
   * bucket = 0 each distinct length shows the cold spike (the 90–380 ms
   * first-audio variance this tunable targets). Mirrors synthOne's embed
   * construction exactly (same conditionerEmbed gather + voice concat), so
   * the measured shapes are the real ones.
   *
   * Usage: `await window.__pipeline().tts.benchTtsPrefill()` (shipped bucket)
   *        `await window.__pipeline().tts.benchTtsPrefill(undefined, { bucket: 16 })`
   */
  async benchTtsPrefill(
    sentences: string[] = [
      // Deliberately different token counts that a 16/32 bucket folds
      // together — with bucket=0 each pays its own re-trace.
      "Tell me a little more about the ocean.",
      "Tell me a little more about the weather today.",
      "Tell me a little more about the weather and the tides tomorrow.",
    ],
    opts: { bucket?: number; voice?: TTSVoice; repeats?: number } = {},
  ): Promise<Record<string, unknown>> {
    const bucket = opts.bucket ?? TUNABLES.ttsPrefillBucket;
    const repeats = opts.repeats ?? 2;
    const voiceEmbed = await this.getVoiceEmbed(opts.voice ?? TTS_VOICES[0]);
    const results: Record<string, unknown>[] = [];
    for (const sentence of sentences) {
      const [prepared] = this.prepareTextPrompt(sentence);
      const tokens = this.encodeTextBucketed(prepared, bucket);
      const runsMs: number[] = [];
      for (let r = 0; r < repeats; r++) {
        const tokensAr = np.array(tokens, { dtype: np.uint32 });
        let embeds = this.model.flowLM.conditionerEmbed.ref.slice(tokensAr);
        embeds = np.concatenate([voiceEmbed.ref, embeds]);
        // Fresh state per run, exactly like a real sentence's step 0: empty
        // KV caches, offset 0, BOS latent as the sequence. runFlowLMStep
        // consumes the refs/embeds and disposes the empty input caches.
        const state = createFlowLMState(this.model.flowLM);
        const t0 = performance.now();
        const {
          latent,
          isEos,
          state: newState,
        } = runFlowLMStep(
          tree.ref(this.model.flowLM),
          state,
          random.key(0),
          this.model.flowLM.bosEmb.ref.reshape([1, -1]),
          embeds,
          0, // step-0 prefill always starts at position 0
        );
        await blockUntilReady([latent.ref, isEos.ref]);
        runsMs.push(performance.now() - t0);
        latent.dispose();
        isEos.dispose();
        tree.dispose(newState.kvCaches);
      }
      results.push({
        sentence,
        textTokens: this.tokenizer.encode(prepared).length,
        paddedTokens: tokens.length,
        // The actual jit trace-key length: voice frames + text tokens + BOS.
        prefillLen: voiceEmbed.shape[0] + tokens.length + 1,
        runsMs: runsMs.map((t) => +t.toFixed(1)),
      });
    }
    return { bucket, repeats, results };
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

  /**
   * Pre-synthesize the onset filler phrases once (off the audio graph) and
   * cache their PCM, mirroring prepareBackchannels: speakStream can then
   * schedule an onset with zero GPU work at the exact moment the GPU is busy
   * with the LLM prefill. Prepared unconditionally (not gated on
   * TUNABLES.onsetFiller) so the in-browser bench can flip the tunable
   * between sessions without a reload — three sub-second phrases add only a
   * couple seconds to the one-time load, same order as the backchannels.
   */
  async prepareOnsets(voice: TTSVoice): Promise<void> {
    if (this.onsets.length && this.onsetVoice === voice) return;
    const clips: { text: string; pcm: Float32Array }[] = [];
    for (const phrase of ONSET_PHRASES) {
      const collector = createStreamingPlayer({ collectPcm: true });
      try {
        await this.synthOne(voice, phrase, collector, null);
        // Trim the silent tail Pocket TTS emits after short phrases (the
        // framesAfterEos padding). The backchannels don't bother — they play
        // in isolation — but here every trailing silent sample directly delays
        // the reply's first synthesized chunk on the shared clock, turning
        // "So, <reply>" into "So, ... <reply>".
        clips.push({ text: phrase, pcm: trimTrailingSilence(collector.pcm()) });
      } finally {
        await collector.close();
      }
    }
    this.onsets = clips;
    this.onsetVoice = voice;
    this.lastOnsetIdx = -1;
  }

  /** Pick a random onset, avoiding an immediate repeat of the last one. */
  private pickOnset(): { text: string; pcm: Float32Array } {
    let idx = Math.floor(Math.random() * this.onsets.length);
    if (this.onsets.length > 1 && idx === this.lastOnsetIdx) {
      idx = (idx + 1) % this.onsets.length;
    }
    this.lastOnsetIdx = idx;
    return this.onsets[idx];
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

/**
 * Drop trailing near-silence from a PCM clip: scan backwards in 10 ms windows
 * and cut everything after the last window whose RMS clears ~1e-3 (well below
 * audible speech, above float noise). Used on the pre-rendered onset fillers,
 * where Pocket TTS's post-EOS padding frames would otherwise sit between the
 * filler and the reply's first chunk on the shared player clock as dead air.
 * Windowed RMS (not per-sample) so a single stray sample can't defeat the trim.
 */
function trimTrailingSilence(
  pcm: Float32Array,
  sampleRate = TTS_SAMPLE_RATE,
): Float32Array {
  const win = Math.max(1, Math.round(sampleRate * 0.01)); // 10 ms
  let end = pcm.length;
  while (end > 0) {
    const start = Math.max(0, end - win);
    let sumSq = 0;
    for (let i = start; i < end; i++) sumSq += pcm[i] * pcm[i];
    if (Math.sqrt(sumSq / (end - start)) >= 1e-3) break;
    end = start;
  }
  return end === pcm.length ? pcm : pcm.slice(0, end);
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
