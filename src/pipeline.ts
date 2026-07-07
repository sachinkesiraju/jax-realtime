// Speech-to-speech pipeline: Whisper ASR -> Gemma LLM -> Kyutai Pocket TTS.
// All three stages run locally in the browser on WebGPU via jax-js, mirroring
// the architecture of the HF/Cerebras real-time voice AI demo.

import {
  defaultDevice,
  type Device,
  init,
  numpy as np,
  tree,
} from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";

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
  fromSafetensors as gemmaFromSafetensors,
  type GemmaModel,
  runGemmaPrefill,
  runGemmaStep,
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
const SYSTEM_HINT =
  "You are a friendly, quick voice assistant in a live full-duplex " +
  "conversation. Answer in one to three short, plain spoken-English " +
  "sentences. No markdown, no lists, no emoji. Timestamps like [t+12s] mark " +
  "when each turn was spoken — use them for time awareness but never read " +
  "them aloud. If a previous assistant turn is marked [interrupted], the user " +
  "cut you off: acknowledge briefly and address what they just said.";

async function fetchWithProgress(
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

function sampleLogits(
  logits: Float32Array,
  opts: { temperature: number; topK: number; topP: number },
): number {
  const k = Math.max(1, Math.min(opts.topK, logits.length));
  const candidates: { id: number; logit: number }[] = [];

  for (let id = 0; id < logits.length; id++) {
    const logit = logits[id];
    if (Number.isNaN(logit)) continue;
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
    const data = await fetchWithProgress(
      "Gemma 3 270M weights",
      `${GEMMA_BASE}/model-it-fp16.safetensors`,
      onProgress,
    );
    const weights = safetensors.parse(data);
    const model = await gemmaFromSafetensors(weights, np.float16);
    return new LocalChatModel(model, tokenizer);
  }

  private formatPrompt(history: ChatMessage[]): string {
    let text = "";
    let firstUser = true;
    for (const message of history) {
      let content = message.content.trim();
      if (content === "") continue;
      const role = message.role === "assistant" ? "model" : "user";
      if (message.role === "user" && message.t !== undefined) {
        content = `[t+${Math.round(message.t)}s] ${content}`;
      }
      if (message.role === "user" && firstUser) {
        content = `${SYSTEM_HINT}\n\n${content}`;
        firstUser = false;
      }
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
    maxNewTokens = 160,
  ): AsyncGenerator<string, GenerateStats, void> {
    const promptTokens = [
      this.tokenizer.bosToken,
      ...this.tokenizer.encode(this.formatPrompt(history)),
    ];
    const generatedTokens: number[] = [];
    const state = createGemmaState({ dtype: np.float16 });
    const inputIds = np.array(promptTokens, { dtype: np.uint32 });
    let logits: np.Array | null = null;
    const startTime = performance.now();
    let firstTokenMs = 0;
    let emitted = "";

    try {
      logits = runGemmaPrefill(tree.ref(this.model), inputIds, state);
      const stopTokens = [this.tokenizer.eosToken, GEMMA_END_OF_TURN];

      for (let i = 0; i < maxNewTokens; i++) {
        const sampledLogits = logits;
        logits = null;
        const data = (await sampledLogits.data()) as Float32Array;
        const nextToken = sampleLogits(data, {
          temperature: 0.8,
          topK: 64,
          topP: 0.95,
        });
        if (i === 0) firstTokenMs = performance.now() - startTime;
        if (stopTokens.includes(nextToken)) break;

        generatedTokens.push(nextToken);
        const full = this.decodeVisible(generatedTokens);
        const delta = full.startsWith(emitted) ? full.slice(emitted.length) : full;
        emitted = full;
        if (delta) yield delta;

        if (i === maxNewTokens - 1) break;
        logits = runGemmaStep(tree.ref(this.model), nextToken, state);
      }

      return {
        promptTokens: promptTokens.length,
        newTokens: generatedTokens.length,
        firstTokenMs,
        totalMs: performance.now() - startTime,
      };
    } finally {
      logits?.dispose();
      tree.dispose(state);
    }
  }

  async generate(
    history: ChatMessage[],
    onText: (partial: string) => void,
    maxNewTokens = 160,
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

    try {
      await this.synthOne(voice, text, player, signal ?? null);
    } finally {
      await player.close();
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

    try {
      for await (const sentence of sentences) {
        if (signal?.aborted) break;
        const line = sentence.trim();
        if (!line) continue;
        await this.synthOne(voice, line, player, signal ?? null);
      }
    } finally {
      await player.close();
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

export type VoicePipeline = {
  asr: SpeechRecognizer;
  llm: LocalChatModel;
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
 * Initialize backends. WebGPU is required (TTS/LLM lane). wasm is preferred for
 * the ASR lane so Whisper passes run concurrently without stalling WebGPU; if
 * wasm can't initialize (e.g. not cross-origin isolated), ASR degrades onto
 * WebGPU and shares it as before.
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
      "WebGPU is not available in this browser; it is required for the TTS stage.",
    );
  }
  defaultDevice("webgpu");
  const wasmReady = devices.includes("wasm");
  if (!wasmReady) {
    console.warn(
      "wasm backend unavailable (cross-origin isolation?); ASR will share WebGPU.",
    );
  }
  return {
    devices,
    asrDevice: wasmReady ? "wasm" : "webgpu",
    asrDtype: wasmReady ? np.float32 : np.float16,
  };
}

export async function loadPipeline(
  onProgress: ProgressFn,
): Promise<VoicePipeline> {
  const setup = await initDevice();
  const asr = await SpeechRecognizer.load(onProgress, {
    device: setup.asrDevice,
    dtype: setup.asrDtype,
  });
  const llm = await LocalChatModel.load(onProgress);
  const tts = await SpeechSynthesizer.load(onProgress);
  return {
    asr,
    llm,
    tts,
    asrDevice: setup.asrDevice,
    dualLane: setup.asrDevice === "wasm",
  };
}
