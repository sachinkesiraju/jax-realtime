// Streaming Mimi ENCODER (kyutai/mimi) on jax-js — 24 kHz PCM in, 32 RVQ
// codebook ids per 12.5 Hz frame out. This is the front half of a future
// kyutai/stt-1b port: the STT temporal transformer consumes exactly these ids.
//
// This is a PARALLEL implementation to the Mimi decoder in src/tts/pocket-tts.ts
// (it borrows that file's streaming-conv/KV-cache patterns) rather than an
// extension of it, because the Pocket TTS checkpoint ships a TRIMMED Mimi
// variant (2-layer transformers, 3 SEANet stages, a DummyQuantizer instead of
// the RVQ) while STT needs the REAL kyutai/mimi encoder. Verified against
// transformers 5.13.1 `MimiModel.from_pretrained("kyutai/mimi")`:
//
//   * SEANet encoder: initConv (1->64, k7) -> 4x [ResnetBlock, ELU,
//     strided down-conv] with strides [4, 5, 6, 8] (reversed decoder ratios,
//     kernel = 2*stride) -> ELU -> finalConv (1024->512, k3). All convs are
//     CAUSAL with zero ("constant") left-padding of kernel-stride — which is
//     exactly the streaming conv-state pattern from pocket-tts (state carries
//     the last kernel-stride inputs; zeros initially). 24 kHz -> 25 Hz.
//     The checkpoint stores plain conv weights (no weight_norm split), so the
//     loader does no weight-norm folding.
//   * Encoder transformer: 8 layers, d=512, 8 heads (no GQA), head_dim 64,
//     MLP 2048 with EXACT (erf) gelu, LayerNorm(eps 1e-5) with bias, per-layer
//     layer scales, sliding-window causal attention (window 250 INCLUDING
//     self), RoPE theta 10 000 in the HF *rotate_half* convention — NOT the
//     interleaved-pair convention pocket-tts's runRope implements (the HF
//     conversion permuted q/k weights), so the RoPE here mirrors
//     src/llm/smollm.ts instead.
//   * Downsample: causal conv (512->512, k4, s2, bias-free), 25 Hz -> 12.5 Hz,
//     with REPLICATE padding: the initial stream state is the first input
//     column repeated, not zeros (transformers MimiConv1dPaddingCache
//     _cache_init) — handled by a first-frame trace variant below.
//   * RVQ (MimiSplitResidualVectorQuantizer): SPLIT into 1 semantic + 31
//     acoustic quantizers. Each split has its OWN 512->256 input projection
//     (conv k=1, bias-free), and both consume the SAME downsample output (the
//     acoustic residual does NOT start from the semantic residual!). Each
//     level: nearest codebook entry by Euclidean distance over 2048 entries of
//     dim 256, subtract the chosen embedding, next level. Codebooks are
//     materialized as embed_sum / clamp(cluster_usage, 1e-5) at export time
//     (bench/mimi/export_weights.py).
//
// Golden parity fixtures + the fp16 weight export live under bench/mimi/
// (gitignored); `mimiEncodeParity()` below is the DEV probe that checks
// per-codebook exact-match rates in the browser against transformers output.
//
// Everything runs in fp32 (weights are stored fp16 and upcast at load):
// argmin-over-distances is exact-match brittle, the whole encoder is only
// ~56 M params, and a frame is T=2 tokens — bandwidth is not the bottleneck.
//
// The per-frame step is fused into ONE jitted dispatch (same rationale and
// inline style as pocket-tts's runMimiDecodeFused / smollm's fused decode
// step): per-frame position scalars are traced np.Arrays so a single trace is
// reused for every steady-state frame.
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  nn,
  numpy as np,
  tree,
} from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";

export const MIMI_CONFIG = {
  sampleRate: 24_000,
  frameSize: 1920, // 24 kHz / 12.5 Hz
  ratios: [4, 5, 6, 8], // SEANet encoder downsampling strides, in order
  hiddenSize: 512,
  numHeads: 8,
  headDim: 64,
  context: 250, // sliding attention window (includes self)
  ropeTheta: 10_000,
  normEps: 1e-5,
  numQuantizers: 32, // 1 semantic + 31 acoustic
  codebookSize: 2048,
  codebookDim: 256,
} as const;

// KV-cache capacity. Must be >= context + tokensPerFrame and divisible by
// tokensPerFrame (the tile-write hack below assumes it); 256 leaves 6 slots of
// slack over the 250-token window with 2 new tokens per frame.
const KV_CAPACITY = 256;
const TOKENS_PER_FRAME = 2; // 80 ms frame -> 2 tokens at 25 Hz pre-downsample

export type Linear = { weight: np.Array }; // [out, in], never biased here
export type Conv1d = { weight: np.Array; bias?: np.Array }; // [C_out, C_in, K]
export type LayerNorm = { weight: np.Array; bias: np.Array };

export type MimiTransformerLayer = {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  oProj: Linear;
  fc1: Linear;
  fc2: Linear;
  norm1: LayerNorm; // input_layernorm
  norm2: LayerNorm; // post_attention_layernorm
  layerScale1: np.Array; // [512] residual scale after attention
  layerScale2: np.Array; // [512] residual scale after MLP
};

export type SEANetEncoderModel = {
  initConv: Conv1d; // 1 -> 64, k7
  blocks: {
    res: { conv1: Conv1d; conv2: Conv1d }; // k3 then k1, hidden = dim/2
    down: Conv1d; // k = 2*stride
  }[];
  finalConv: Conv1d; // 1024 -> 512, k3
};

export type MimiEncoderModel = {
  encoder: SEANetEncoderModel;
  transformer: MimiTransformerLayer[]; // 8 layers
  downsample: { weight: np.Array }; // [512, 512, 4], stride 2, no bias
  rvq: {
    semanticInputProj: Linear; // [256, 512]
    acousticInputProj: Linear; // [256, 512]
    semanticCodebook: np.Array; // [2048, 256]
    acousticCodebooks: np.Array[]; // 31 x [2048, 256]
  };
};

export type KVCache = {
  key: np.Array; // [KV_CAPACITY, H, D]
  value: np.Array; // [KV_CAPACITY, H, D]
};

export type MimiEncodeState = {
  // Streaming conv tails, in SEANet layer order: initConv, then per block
  // [res.conv1, res.conv2, down], then finalConv — 14 entries.
  convStates: np.Array[];
  // Downsample conv tail. null until the first frame: its replicate padding
  // must be initialized from the first frame's own 25 Hz embedding (which
  // only exists inside the trace), so the first dispatch uses a separate
  // trace variant that synthesizes it and returns the real state.
  downsampleState: np.Array | null;
  kvCaches: KVCache[];
  kvCacheLen: number; // valid tokens in cache (always a multiple of 2)
  offset: number; // absolute 25 Hz token position of the next frame
};

// ---------------------------------------------------------------------------
// Building blocks. Plain (non-jitted) so they inline into the single fused
// per-frame trace — same style as the fused paths in pocket-tts.ts/smollm.ts.

function runLinear({ weight }: Linear, x: np.Array): np.Array {
  return np.dot(x, weight.transpose());
}

function layerNormInline(
  { weight, bias }: LayerNorm,
  x: np.Array,
  eps: number = MIMI_CONFIG.normEps,
): np.Array {
  // Everything is fp32 already; keep the explicit high-precision structure
  // anyway so this stays correct if activations ever move to fp16.
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const mean = x.ref.mean(-1, { keepdims: true });
  const var_ = np.var_(x.ref, -1, {
    mean: mean.ref,
    correction: 0,
    keepdims: true,
  });
  x = x.sub(mean).div(np.sqrt(var_.add(eps)));
  return x.mul(weight).add(bias).astype(dtype);
}

/** Streaming conv state: the last (kernel - stride) input columns, zeros
 * initially (== causal zero padding of padding_total). Same math as
 * pocket-tts's createConv1dState, but fp32. */
function createConvState(weight: np.Array, stride: number = 1): np.Array {
  return np.zeros([1, weight.shape[1], weight.shape[2] - stride], {
    dtype: np.float32,
  });
}

/** Streaming causal conv: prepend the carried tail, VALID conv, carry the new
 * tail. Identical to pocket-tts's runConv1d (which validated this pattern
 * against the same MimiConv1d semantics on the decoder side). */
function convInline(
  { weight, bias }: Conv1d,
  state: np.Array,
  x: np.Array, // [1, C_in, T]
  stride: number = 1,
): [np.Array, np.Array] {
  x = np.concatenate([state, x], 2);
  const newState = x.ref.slice([], [], [x.shape[2] - (weight.shape[2] - stride)]);
  let y = lax.conv(x, weight, [stride], "VALID");
  if (bias) y = y.add(np.expandDims(bias, -1));
  return [y, newState];
}

/** SEANet encoder for one frame: [1, 1, 1920] -> [1, 512, 2] (25 Hz), threading
 * the per-conv streaming states (consumed and replaced). */
function seanetEncoderInline(
  { initConv, blocks, finalConv }: SEANetEncoderModel,
  states: np.Array[],
  x: np.Array, // [1, 1, 1920]
): [np.Array, np.Array[]] {
  const newStates: np.Array[] = [];
  const step = (conv: Conv1d, v: np.Array, stride = 1): np.Array => {
    const [y, s] = convInline(conv, states[newStates.length], v, stride);
    newStates.push(s);
    return y;
  };

  x = step(initConv, x);
  for (let i = 0; i < blocks.length; i++) {
    // ResnetBlock = [ELU, conv k3, ELU, conv k1] + identity shortcut
    // (use_conv_shortcut=false). NOTE: the block itself begins with ELU;
    // there is no extra ELU between the previous conv and the block.
    let v = step(blocks[i].res.conv1, nn.elu(x.ref));
    v = step(blocks[i].res.conv2, nn.elu(v));
    x = x.add(v);
    // ELU + strided downsampling conv (kernel = 2*stride).
    x = step(blocks[i].down, nn.elu(x), MIMI_CONFIG.ratios[i]);
  }
  x = step(finalConv, nn.elu(x));
  return [x, newStates];
}

/** HF rotate_half RoPE (see src/llm/smollm.ts) — NOT pocket-tts's interleaved
 * runRope; the HF Mimi conversion permuted q/k weights into this convention. */
function ropeInline(
  q: np.Array, // [T, H, D]
  k: np.Array,
  offset: np.Array, // int32 scalar: absolute position of token 0
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;

  const rotateHalf = (x: np.Array): np.Array => {
    const [x1, x2] = np.split(x, 2, -1);
    return np.concatenate([x2.mul(-1), x1], -1);
  };

  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(MIMI_CONFIG.ropeTheta) * 2) / D));
  const positions = np
    .arange(T)
    .add(offset)
    .astype(np.float32)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq); // [T, D/2]

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

/** One transformer layer with a linear (shift-managed-in-JS) KV cache and the
 * sliding-window causal mask. Cache-write trick is pocket-tts's: keep slots
 * < kvCacheLen, tile-write the T new tokens everywhere else (KV_CAPACITY % T
 * == 0 and kvCacheLen % T == 0 make the tiling line up at the target slots;
 * the garbage beyond is masked out now and overwritten before it can age in).
 */
function transformerLayerInline(
  layer: MimiTransformerLayer,
  kvCache: KVCache,
  x: np.Array, // [T, 512]
  offset: np.Array, // int32 scalar
  kvCacheLen: np.Array, // int32 scalar
): [np.Array, KVCache] {
  const { numHeads, headDim, context } = MIMI_CONFIG;
  const T = x.shape[0];

  // --- Attention block (pre-norm, layer-scaled residual).
  const xOrig = x.ref;
  x = layerNormInline(layer.norm1, x);
  let q = runLinear(layer.qProj, x.ref).reshape([T, numHeads, headDim]);
  let k = runLinear(layer.kProj, x.ref).reshape([T, numHeads, headDim]);
  const v = runLinear(layer.vProj, x).reshape([T, numHeads, headDim]);
  [q, k] = ropeInline(q, k, offset);

  const cacheMask = np
    .arange(KV_CAPACITY)
    .reshape([-1, 1, 1])
    .less(kvCacheLen.ref);
  const key = np.where(
    cacheMask.ref,
    kvCache.key,
    np.tile(k, [KV_CAPACITY / T, 1, 1]),
  );
  const value = np.where(cacheMask, kvCache.value, np.tile(v, [KV_CAPACITY / T, 1, 1]));

  // delta = (key absolute pos) - (query absolute pos); attend iff
  // -context < delta <= 0 (window includes self + context-1 past tokens,
  // matching transformers' create_sliding_window_causal_mask).
  const maskDelta = np
    .arange(KV_CAPACITY)
    .sub(np.arange(T).reshape([T, 1]))
    .sub(kvCacheLen); // [T, KV_CAPACITY]
  const mask = maskDelta.ref.lessEqual(0).mul(maskDelta.greater(-context));

  let attn = nn.dotProductAttention(q, key.ref, value.ref, {
    mask,
    scale: 1 / Math.sqrt(headDim),
  });
  attn = runLinear(layer.oProj, attn.reshape([T, numHeads * headDim]));
  x = xOrig.add(attn.mul(layer.layerScale1));

  // --- MLP block (pre-norm, exact gelu, layer-scaled residual).
  const xOrig2 = x.ref;
  x = layerNormInline(layer.norm2, x);
  x = runLinear(layer.fc1, x);
  x = nn.gelu(x, { approximate: false });
  x = runLinear(layer.fc2, x);
  x = xOrig2.add(x.mul(layer.layerScale2));

  return [x, { key, value }];
}

/** One residual-VQ split: project to 256-d, then per level pick the nearest
 * codebook entry (Euclidean) and subtract it from the running residual.
 * argmin ||r - c||^2 == argmax (2 r.c - ||c||^2) — the ||r||^2 term is
 * constant across entries, so a [1, 2048] matmul + argmax suffices (same
 * reduction ops smollm's fused top-k path leans on). */
function rvqSplitInline(
  inputProj: Linear,
  codebooks: np.Array[],
  x: np.Array, // [1, 512] downsample output
): np.Array[] {
  let residual = runLinear(inputProj, x); // [1, 256]
  const indices: np.Array[] = [];
  for (const cb of codebooks) {
    const scores = np.dot(residual.ref, cb.ref.transpose()); // [1, 2048]
    const normSq = cb.ref.mul(cb.ref).sum(-1); // [2048]
    const idx = np.argmax(scores.mul(2).sub(normSq), -1).astype(np.uint32); // [1]
    const chosen = cb.slice(idx.ref); // [1, 256] embedding gather
    residual = residual.sub(chosen);
    indices.push(idx);
  }
  residual.dispose();
  return indices;
}

// ---------------------------------------------------------------------------
// Fused per-frame step: SEANet -> transformer -> downsample -> RVQ in a single
// jitted dispatch. `firstFrame` is a static arg because the downsample conv's
// REPLICATE padding state must be synthesized from in-trace data on frame 0
// (two traces total: frame 0 and steady state — position scalars are traced,
// so the steady-state trace is reused for every subsequent frame).
const runMimiEncodeStepFused = jit(
  function runMimiEncodeStepFused(
    model: MimiEncoderModel,
    convStates: np.Array[],
    downsampleState: np.Array,
    kvCaches: KVCache[],
    offset: np.Array, // int32 scalar
    kvCacheLen: np.Array, // int32 scalar
    pcm: np.Array, // [1, 1, 1920]
    firstFrame: boolean,
  ): [np.Array, np.Array, np.Array[], np.Array, KVCache[]] {
    // SEANet encoder: [1, 1, 1920] -> [1, 512, 2].
    let x: np.Array;
    [x, convStates] = seanetEncoderInline(model.encoder, convStates, pcm);

    // Encoder transformer over the frame's 2 tokens at 25 Hz.
    x = x.slice(0).transpose([1, 0]); // [2, 512]
    for (let i = 0; i < model.transformer.length; i++) {
      [x, kvCaches[i]] = transformerLayerInline(
        model.transformer[i],
        kvCaches[i],
        x,
        offset.ref,
        kvCacheLen.ref,
      );
    }
    offset.dispose();
    kvCacheLen.dispose();
    x = np.expandDims(x.transpose([1, 0]), 0); // [1, 512, 2]

    // Downsample 25 Hz -> 12.5 Hz. Replicate padding: on the first frame the
    // carried tail is the frame's own first column repeated (transformers
    // MimiConv1dPaddingCache._cache_init for pad_mode="replicate").
    if (firstFrame) {
      downsampleState.dispose();
      downsampleState = np.tile(x.ref.slice([], [], [0, 1]), [1, 1, 2]);
    }
    [x, downsampleState] = convInline(
      { weight: model.downsample.weight },
      downsampleState,
      x,
      2,
    ); // [1, 512, 1]

    // Split RVQ: semantic level + 31 acoustic levels, BOTH starting from the
    // same embedding (separate input projections).
    const emb = x.slice(0).transpose([1, 0]); // [1, 512]
    const semantic = rvqSplitInline(
      model.rvq.semanticInputProj,
      [model.rvq.semanticCodebook],
      emb.ref,
    );
    const acoustic = rvqSplitInline(
      model.rvq.acousticInputProj,
      model.rvq.acousticCodebooks,
      emb.ref,
    );
    const codes = np
      .concatenate([...semantic, ...acoustic])
      .astype(np.int32); // [32]

    // Also expose the pre-quantization embedding — the parity probe compares
    // it against the transformers fixture to localize errors. [512], cheap.
    const preQuant = emb.reshape([MIMI_CONFIG.hiddenSize]);

    return [codes, preQuant, convStates, downsampleState, kvCaches];
  },
  { staticArgnums: [7] },
);

// ---------------------------------------------------------------------------
// Public API.

export function createMimiEncodeState(model: MimiEncoderModel): MimiEncodeState {
  const { encoder, transformer } = model;
  const convStates: np.Array[] = [createConvState(encoder.initConv.weight.ref)];
  for (let i = 0; i < encoder.blocks.length; i++) {
    convStates.push(createConvState(encoder.blocks[i].res.conv1.weight.ref));
    convStates.push(createConvState(encoder.blocks[i].res.conv2.weight.ref));
    convStates.push(
      createConvState(encoder.blocks[i].down.weight.ref, MIMI_CONFIG.ratios[i]),
    );
  }
  convStates.push(createConvState(encoder.finalConv.weight.ref));
  return {
    convStates,
    downsampleState: null,
    kvCaches: transformer.map(() => ({
      key: np.zeros([KV_CAPACITY, MIMI_CONFIG.numHeads, MIMI_CONFIG.headDim], {
        dtype: np.float32,
      }),
      value: np.zeros([KV_CAPACITY, MIMI_CONFIG.numHeads, MIMI_CONFIG.headDim], {
        dtype: np.float32,
      }),
    })),
    kvCacheLen: 0,
    offset: 0,
  };
}

/**
 * Encode one 80 ms frame (1920 samples of 24 kHz mono PCM) to its 32 RVQ code
 * ids. Returns the codes as an int32 np.Array of shape [32] (await
 * `.data()` for the ids) plus the pre-quantization embedding for debugging;
 * advances `state` in place.
 */
export function encodeFrame(
  model: MimiEncoderModel,
  state: MimiEncodeState,
  pcm: Float32Array<ArrayBuffer>,
): { codes: np.Array; preQuant: np.Array } {
  if (pcm.length !== MIMI_CONFIG.frameSize) {
    throw new Error(`Expected ${MIMI_CONFIG.frameSize} samples, got ${pcm.length}`);
  }

  // Sliding-window cache maintenance (shape ops stay in JS, mirroring
  // pocket-tts's decoder): when the linear cache is full, drop the 2 oldest
  // slots. The retained 254 tokens still cover the 250-token window.
  if (state.kvCacheLen + TOKENS_PER_FRAME > KV_CAPACITY) {
    state.kvCacheLen -= TOKENS_PER_FRAME;
    for (const c of state.kvCaches) {
      c.key = np.pad(c.key.slice([TOKENS_PER_FRAME]), {
        0: [0, TOKENS_PER_FRAME],
      });
      c.value = np.pad(c.value.slice([TOKENS_PER_FRAME]), {
        0: [0, TOKENS_PER_FRAME],
      });
    }
  }

  const firstFrame = state.downsampleState === null;
  const downsampleState =
    state.downsampleState ??
    // Placeholder with the real shape so both traces see identical avals;
    // the first-frame trace ignores and disposes it.
    np.zeros([1, MIMI_CONFIG.hiddenSize, 2], { dtype: np.float32 });

  const offsetArr = np.array(state.offset, { dtype: np.int32 });
  const kvCacheLenArr = np.array(state.kvCacheLen, { dtype: np.int32 });
  const x = np.array(pcm, {
    dtype: np.float32,
    shape: [1, 1, MIMI_CONFIG.frameSize],
  });

  let codes: np.Array;
  let preQuant: np.Array;
  [codes, preQuant, state.convStates, state.downsampleState, state.kvCaches] =
    runMimiEncodeStepFused(
      tree.ref(model),
      state.convStates,
      downsampleState,
      state.kvCaches,
      offsetArr,
      kvCacheLenArr,
      x,
      firstFrame,
    );
  state.kvCacheLen += TOKENS_PER_FRAME;
  state.offset += TOKENS_PER_FRAME;
  return { codes, preQuant };
}

/** Batch helper for parity testing: encode a whole clip, returning one
 * Int32Array of 32 ids per full frame (a trailing partial frame is dropped). */
export async function encodeWav(
  model: MimiEncoderModel,
  pcm: Float32Array<ArrayBuffer>,
): Promise<Int32Array[]> {
  const state = createMimiEncodeState(model);
  const frames: Int32Array[] = [];
  const numFrames = Math.floor(pcm.length / MIMI_CONFIG.frameSize);
  for (let i = 0; i < numFrames; i++) {
    const { codes, preQuant } = encodeFrame(
      model,
      state,
      pcm.subarray(i * MIMI_CONFIG.frameSize, (i + 1) * MIMI_CONFIG.frameSize),
    );
    preQuant.dispose();
    // data() consumes `codes` (move semantics) — no dispose needed.
    frames.push(new Int32Array((await codes.data()) as ArrayLike<number>));
  }
  tree.dispose(state);
  return frames;
}

/** Hydrate the encoder from the fp16 safetensors produced by
 * bench/mimi/export_weights.py (clean names — no WeightMapper needed). All
 * weights are upcast to fp32; see the module comment for why. */
export function fromSafetensors(file: safetensors.File): MimiEncoderModel {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    if (tensor.dtype !== "F16") {
      throw new Error(`Expected fp16 Mimi weights, got ${tensor.dtype} for ${key}`);
    }
    hydrated[key] = np.array(
      new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
      { shape: tensor.shape, dtype: np.float32 },
    );
  }
  return safetensors.toNested(hydrated) as MimiEncoderModel;
}

// ---------------------------------------------------------------------------
// DEV parity probe. Run via a bare vite page (no app model load needed):
//   import("/src/asr/mimi-encode.ts").then((m) => m.mimiEncodeParity())
// against fixtures generated by bench/mimi/make_fixtures.py. Reports
// per-codebook exact-match rates vs transformers' golden codes, plus
// pre-quantization embedding error and per-frame encode latency.

type ParityFixture = {
  transformers: string;
  pcm: number[];
  codes: number[][]; // [T][32]
  preQuantStream: number[][]; // [<=8][512]
};

export async function mimiEncodeParity(
  fixtureUrl: string = "/bench/mimi/fixture_sine.json",
  weightsUrl: string = "/bench/mimi/mimi-encoder.fp16.safetensors",
) {
  await init("webgpu");
  defaultDevice("webgpu");

  const [fixture, weightsData] = await Promise.all([
    fetch(fixtureUrl).then((r) => r.json()) as Promise<ParityFixture>,
    fetch(weightsUrl)
      .then((r) => r.arrayBuffer())
      .then((b) => new Uint8Array(b)),
  ]);
  const model = fromSafetensors(safetensors.parse(weightsData));
  await blockUntilReady(model);

  const pcm = new Float32Array(fixture.pcm);
  const state = createMimiEncodeState(model);
  const numFrames = Math.floor(pcm.length / MIMI_CONFIG.frameSize);

  const got: Int32Array[] = [];
  let preQuantErr = 0;
  const frameMs: number[] = [];
  for (let i = 0; i < numFrames; i++) {
    const t0 = performance.now();
    const { codes, preQuant } = encodeFrame(
      model,
      state,
      pcm.subarray(i * MIMI_CONFIG.frameSize, (i + 1) * MIMI_CONFIG.frameSize),
    );
    // data() consumes the arrays (move semantics) — dispose only skipped ones.
    got.push(new Int32Array((await codes.data()) as ArrayLike<number>));
    frameMs.push(performance.now() - t0);
    if (i < fixture.preQuantStream.length) {
      const pq = (await preQuant.data()) as Float32Array;
      const ref = fixture.preQuantStream[i];
      let se = 0;
      for (let j = 0; j < ref.length; j++) se += (pq[j] - ref[j]) ** 2;
      preQuantErr = Math.max(preQuantErr, Math.sqrt(se / ref.length));
    } else {
      preQuant.dispose();
    }
  }
  tree.dispose(state);

  // Amortized throughput over ~50 frames, syncing every 4th frame: the
  // per-frame loop above pays a full GPU sync + readback per frame (that is
  // the LATENCY number); a live pipeline can let a few frames' dispatches
  // queue before reading codes back. (Syncing only once at the very end is
  // counterproductively SLOWER — a 50-frame dependency chain of deferred
  // dispatches — so a small sync stride is the realistic steady-state cost.)
  const benchFrames = Math.min(50, numFrames);
  const benchState = createMimiEncodeState(model);
  const t0 = performance.now();
  for (let i = 0; i < benchFrames; i++) {
    const { codes, preQuant } = encodeFrame(
      model,
      benchState,
      pcm.subarray(i * MIMI_CONFIG.frameSize, (i + 1) * MIMI_CONFIG.frameSize),
    );
    preQuant.dispose();
    if (i % 4 === 3 || i === benchFrames - 1) await codes.data();
    else codes.dispose();
  }
  const pipelinedMs = (performance.now() - t0) / benchFrames;
  tree.dispose([benchState, model]);

  // Per-codebook exact-match rate vs the golden (streaming-path) codes.
  const perCodebook = new Array(MIMI_CONFIG.numQuantizers).fill(0);
  for (let t = 0; t < numFrames; t++) {
    for (let q = 0; q < MIMI_CONFIG.numQuantizers; q++) {
      if (got[t][q] === fixture.codes[t][q]) perCodebook[q]++;
    }
  }
  const rates = perCodebook.map((c) => c / numFrames);
  // Steady-state latency: skip the first 2 frames (trace compilation).
  const steady = frameMs.slice(2);
  return {
    transformers: fixture.transformers,
    frames: numFrames,
    semanticMatch: rates[0],
    acousticMatch: rates.slice(1).reduce((a, b) => a + b, 0) / 31,
    averageMatch: rates.reduce((a, b) => a + b, 0) / rates.length,
    perCodebook: rates.map((r) => Math.round(r * 1000) / 1000),
    preQuantRmseMax: preQuantErr,
    encodeMsMean: steady.reduce((a, b) => a + b, 0) / steady.length,
    encodeMsP50: steady.slice().sort((a, b) => a - b)[Math.floor(steady.length / 2)],
    encodeMsPipelined: pipelinedMs,
  };
}
