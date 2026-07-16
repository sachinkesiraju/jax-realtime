// Kyutai STT decoder (kyutai/stt-1b-en_fr) on jax-js — the back half of the
// speech-to-text pipeline: per 80 ms frame it consumes the 32 Mimi RVQ codes
// (from src/asr/mimi-encode.ts) plus its own previous text token and emits one
// text token by greedy argmax over an 8001-way SentencePiece vocab.
//
// This is the "delayed streams" temporal transformer of the Moshi family.
// Verified against transformers 5.13.1
// `KyutaiSpeechToTextForConditionalGeneration` (kyutai/stt-1b-en_fr-trfs):
//
//   * 16 pre-norm decoder layers, hidden 2048, 16 heads x head_dim 128 (no
//     GQA), all projections bias-free. RMSNorm eps 1e-8 computed in fp32
//     (output = norm(x) * weight, no offset). RoPE theta 100 000 in the HF
//     rotate_half convention (same as src/llm/smollm.ts).
//   * GATED MLP, but NOT the Llama 3-matrix layout: ONE fc1 (2048 -> 11264)
//     whose output is viewed as [2, 5632] — the FIRST half is the gate, the
//     second the up projection — then silu(gate) * up -> fc2 (5632 -> 2048)
//     (KyutaiSpeechToTextGatingMLP; config.hidden_act = "silu").
//   * ONE flat embedding table of 8001 text + 32*2049 audio + 1 = 73 570 rows:
//     per step the 33 input ids [text, code_0..code_31] are offset-indexed
//     (audio code q -> 8001 + q*2049 + code) and their embeddings are SUMMED
//     (KyutaiSpeechToTextEmbeddings). The lm_head (8001 x 2048) is NOT tied —
//     it's a separate tensor (config.tie_word_embeddings = false).
//   * Stream layout at inference (from prepare_inputs_for_generation): step 0
//     feeds text bos 48000 (a raw index into the flat table — deliberately
//     outside the 8001 text rows) + the audio bos code 2048 in all 32
//     codebooks; step s >= 1 feeds the PREVIOUS generated text token + the
//     mimi codes of audio frame s-1. So a clip of N frames yields N+1 steps
//     whose LAST audio frame is never consumed — the reference caps
//     max_new_tokens at the frame count. The audio pad id (69569) only occurs
//     in training; it never appears at inference and is not handled here.
//   * The 0.5 s text-vs-audio delay is IMPLICIT (trained-in): the token for a
//     word appears ~6 frames after its audio. Nothing to implement — but
//     callers must keep feeding ~6 frames (the HF feature extractor pads
//     audio_delay_seconds + 1.0 = 1.5 s of trailing silence) or the last
//     words never flush.
//   * Sliding-window attention, window 375 INCLUDING self: the reference
//     generates with a sliding KV cache that retains the last 374 past tokens
//     (DynamicSlidingWindowLayer keeps sliding_window - 1), while RoPE keeps
//     using absolute positions. Mirrored here with the linear
//     shift-managed-in-JS cache + delta-window mask that
//     src/asr/mimi-encode.ts already validated (there window=250, shift 2;
//     here window=375, shift 9).
//
// Weights ship fp16 (bench/stt/export_weights.py; parity gate) or per-row
// symmetric int8 (~half the download; same `<name>.scale` scheme as
// src/llm/smollm.ts). The residual stream and KV cache run fp32 — with only
// one token per step, bandwidth is dominated by the fp16 weights themselves.
//
// The per-frame step is ONE jitted dispatch (same rationale as smollm's
// runSmolLmDecodeStepFusedTopK / mimi-encode's fused step): embedding sum ->
// 16 layers -> norm -> lm_head -> argmax on GPU, and only the winning token
// id is read back. Position scalars are traced np.Arrays so a single trace
// serves every frame.
//
// SEMANTIC-VAD EXTRA HEADS (vadProbs). kyutai/stt-1b-en_fr also ships 4 tiny
// "extra heads" trained to predict the probability that the user is done
// talking (kyutai.org/stt) — implemented upstream only in Kyutai's Rust
// server. Reverse-engineered spec (moshi rust/moshi-core/src/lm.rs + asr.rs,
// delayed-streams-modeling configs/config-stt-en_fr-hf.toml, and unmute's
// consumer unmute/stt/speech_to_text.py):
//   * Weights: extra_heads.{0..3}.weight, each [6, 2048], bias-free, in the
//     CANDLE checkpoint only (kyutai/stt-1b-en_fr-candle — absent from the
//     HF transformers export; base weights of the two are bit-identical, so
//     the heads apply directly to our hidden state). Exported here as
//     extraHeads.{i}.weight by bench/stt/export_weights.py.
//   * Math: logits_i = W_i @ ys where ys is the POST-final-norm hidden state
//     (lm.rs forward_cond returns ys = out_norm(transformer_out) — the same
//     tensor that feeds text_linear/lm_head, i.e. exactly the `hidden` this
//     fused step already computes); p_i = softmax(logits_i)[0] is the
//     probability the user has finished (asr.rs takes element 0 of each
//     head; the other 5 classes are unused by every consumer).
//   * Head i predicts "finished" at horizon [0.5, 1, 2, 3][i] seconds
//     (delayed-streams-modeling issue #8). Unmute's decision rule pauses on
//     prs[2] (the 2 s horizon) > 0.6 after a near-instant EMA
//     (attack/release 0.01 s at 12.5 Hz ⇒ alpha ≈ 0.996, effectively raw),
//     ignoring the first 12 steps ("all over the place in the first few
//     steps"), and treats prs[2] < 0.4 as "user speaking" for barge-in.
// The heads fold into the same fused dispatch (one extra [4·6, 2048] matmul,
// near-zero cost); loading tolerates weight files without them (vadProbs
// null → callers fall back to timer endpointing) so old cached weights work.
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
  tree,
} from "@jax-js/jax";
import { safetensors, tokenizers } from "@jax-js/loaders";

import {
  createMimiEncodeState,
  encodeFrame,
  fromSafetensors as mimiFromSafetensors,
  MIMI_CONFIG,
} from "./mimi-encode";

export const STT_CONFIG = {
  textVocabSize: 8001,
  numCodebooks: 32,
  codebookVocabSize: 2049,
  bosTokenId: 48_000, // raw flat-table index; NOT a text-vocab id
  audioBosTokenId: 2048,
  padTokenId: 3, // ids 0..3 are specials (<unk> <s> </s> <pad>) — not text
  hiddenSize: 2048,
  numLayers: 16,
  numHeads: 16,
  headDim: 128,
  ffnDim: 11_264, // fc1 output; gate/up halves of 5632 each
  slidingWindow: 375, // attention window, INCLUDING self
  ropeTheta: 100_000,
  rmsNormEps: 1e-8,
  frameRate: 12.5, // Hz; one text token per 80 ms frame
  // Semantic-VAD "extra heads" (see the vadProbs section of the module
  // comment): 4 heads, one per end-of-turn prediction horizon, softmax over
  // 6 classes each. numVadHeads matches config-stt-en_fr-hf.toml
  // ([modules.asr.model.extra_heads] num_heads = 4, dim = 6).
  numVadHeads: 4,
  vadHeadDim: 6,
  vadHorizonsSec: [0.5, 1, 2, 3], // head i predicts "done" at this horizon
} as const;

// KV-cache capacity. Window is 375 including self, so a step needs at most
// 374 past slots + 1 new; when the linear cache would overflow, JS drops the
// KV_SHIFT oldest slots (the retained 375 still cover every attendable
// position). 384 gives 9 frames of slack so the shift runs ~once per 0.7 s.
const KV_CAPACITY = 384;
const KV_SHIFT = KV_CAPACITY - STT_CONFIG.slidingWindow; // 9

const ATTENTION_SCALE = 1 / Math.sqrt(STT_CONFIG.headDim);

export type Linear = { weight: np.Array }; // [out, in], always bias-free here
export type RMSNorm = { weight: np.Array };

export type SttLayer = {
  inputLayernorm: RMSNorm;
  postAttentionLayernorm: RMSNorm;
  selfAttn: { qProj: Linear; kProj: Linear; vProj: Linear; oProj: Linear };
  mlp: { fc1: Linear; fc2: Linear }; // fused-gate MLP, see module comment
};

export type SttModel = {
  embed: Linear; // [73570, 2048] flat text+audio table
  lmHead: Linear; // [8001, 2048], untied
  layers: SttLayer[]; // 16
  norm: RMSNorm;
  /** Semantic-VAD extra heads, [6, 2048] each (see module comment). Absent
   *  when the weight file predates the VAD export — vadProbs is then null. */
  extraHeads?: Linear[];
};

export type SttKVCache = {
  key: np.Array; // [KV_CAPACITY, H, D] fp32
  value: np.Array;
};

export type SttState = {
  caches: SttKVCache[];
  kvCacheLen: number; // valid (compacted-to-front) slots in the linear cache
  position: number; // absolute step index (RoPE position of the next step)
  prevToken: number; // text token fed to the next step (bos initially)
};

// ---------------------------------------------------------------------------
// Building blocks. Plain (non-jitted) so they inline into the single fused
// per-frame trace — same style as mimi-encode.ts / smollm.ts's fused paths.

function linearInline({ weight }: Linear, x: np.Array): np.Array {
  return np.dot(x, weight.transpose());
}

function rmsNormInline({ weight }: RMSNorm, x: np.Array): np.Array {
  // Reference computes in fp32 (mandatory here: eps 1e-8 underflows fp16).
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  x = x.div(np.sqrt(rms.add(STT_CONFIG.rmsNormEps)));
  return x.mul(weight.astype(np.float32)).astype(dtype);
}

/** Fused-gate MLP: fc1 packs [gate, up] as the two halves of its output
 * (torch `.view(.., 2, ffn/2)` puts the gate FIRST), then silu(gate) * up. */
function mlpInline({ fc1, fc2 }: SttLayer["mlp"], x: np.Array): np.Array {
  const y = linearInline(fc1, x); // [1, 11264]
  const [gate, up] = np.split(y, 2, -1); // [1, 5632] each
  return linearInline(fc2, nn.silu(gate).mul(up));
}

function rotateHalf(x: np.Array): np.Array {
  const [x1, x2] = np.split(x, 2, -1);
  return np.concatenate([x2.mul(-1), x1], -1);
}

/** HF rotate_half RoPE at an absolute (traced) position — same convention as
 * smollm.ts / mimi-encode.ts (NOT the interleaved-pair variant). */
function ropeInline(
  q: np.Array, // [1, H, D]
  k: np.Array,
  position: np.Array, // int32 scalar
): [np.Array, np.Array] {
  const D = STT_CONFIG.headDim;
  const dim = np.arange(D / 2, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(STT_CONFIG.ropeTheta) * 2) / D));
  const freqs = position.astype(np.float32).reshape([1, 1]).mul(invFreq); // [1, D/2]

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([1, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([1, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

/** One decoder layer over the step's single token, with the linear KV cache +
 * sliding-window mask validated in mimi-encode.ts: keep slots < kvCacheLen,
 * tile-write the new token everywhere else (only slot kvCacheLen is attendable
 * now; the garbage beyond is masked out and overwritten before it ages in). */
function layerInline(
  layer: SttLayer,
  cache: SttKVCache,
  x: np.Array, // [1, 2048]
  position: np.Array, // int32 scalar
  kvCacheLen: np.Array, // int32 scalar
): [np.Array, SttKVCache] {
  const { numHeads, headDim, slidingWindow } = STT_CONFIG;

  const residual = x.ref;
  x = rmsNormInline(layer.inputLayernorm, x);
  let q = linearInline(layer.selfAttn.qProj, x.ref).reshape([1, numHeads, headDim]);
  let k = linearInline(layer.selfAttn.kProj, x.ref).reshape([1, numHeads, headDim]);
  const v = linearInline(layer.selfAttn.vProj, x).reshape([1, numHeads, headDim]);
  [q, k] = ropeInline(q, k, position);

  const cacheMask = np
    .arange(KV_CAPACITY)
    .reshape([-1, 1, 1])
    .less(kvCacheLen.ref);
  const key = np.where(cacheMask.ref, cache.key, np.tile(k, [KV_CAPACITY, 1, 1]));
  const value = np.where(cacheMask, cache.value, np.tile(v, [KV_CAPACITY, 1, 1]));

  // delta = (key absolute pos) - (query absolute pos): slot i holds position
  // position - kvCacheLen + i. Attend iff -window < delta <= 0 — the window
  // includes self + 374 past tokens, matching the reference's sliding cache
  // (which retains sliding_window - 1 = 374 past keys).
  const maskDelta = np.arange(KV_CAPACITY).reshape([1, -1]).sub(kvCacheLen); // [1, CAP]
  const mask = maskDelta.ref.lessEqual(0).mul(maskDelta.greater(-slidingWindow));

  let attn = nn.dotProductAttention(q, key.ref, value.ref, {
    mask,
    scale: ATTENTION_SCALE,
  });
  attn = linearInline(
    layer.selfAttn.oProj,
    attn.reshape([1, numHeads * headDim]),
  );
  x = residual.add(attn);

  const residual2 = x.ref;
  x = rmsNormInline(layer.postAttentionLayernorm, x);
  x = mlpInline(layer.mlp, x);
  x = residual2.add(x);

  return [x, { key, value }];
}

// ---------------------------------------------------------------------------
// Fused per-frame step: embedding sum -> 16 layers -> norm -> lm_head ->
// argmax (+ the semantic-VAD extra heads when loaded), one jitted dispatch.
// `ids` are the 33 PRE-OFFSET flat-table row indices (computed on the JS
// side — cheaper than in-trace offset math and keeps the trace input a
// single array). `vadProbs` is a 0-or-1-element list (a [4] array of each
// head's softmax[0] when the model has extraHeads, else empty) — the LIST
// keeps one jit signature serving both model structures (jit re-traces per
// input tree structure, so each variant gets its own cached trace).
const runSttStepFused = jit(function runSttStepFused(
  model: SttModel,
  caches: SttKVCache[],
  ids: np.Array, // [33] uint32: [text row, audio rows...]
  position: np.Array, // int32 scalar
  kvCacheLen: np.Array, // int32 scalar
): [np.Array, np.Array, np.Array[], SttKVCache[]] {
  // Summed embedding of the text token + all 32 audio codes. fp32 residual
  // stream (the fp16 table rows are upcast before the 33-way sum).
  let x = model.embed.weight.slice(ids).astype(np.float32).sum(0, {
    keepdims: true,
  }); // [1, 2048]

  const newCaches: SttKVCache[] = [];
  for (let i = 0; i < STT_CONFIG.numLayers; i++) {
    let cache: SttKVCache;
    [x, cache] = layerInline(
      model.layers[i],
      caches[i],
      x,
      position.ref,
      kvCacheLen.ref,
    );
    newCaches.push(cache);
  }
  position.dispose();
  kvCacheLen.dispose();

  x = rmsNormInline(model.norm, x);
  const logits = linearInline(model.lmHead, x.ref); // [1, 8001]
  const tokenId = np.argmax(logits, -1).astype(np.int32); // [1]

  // Semantic-VAD extra heads on the same post-final-norm hidden state (see
  // module comment): per head, softmax over 6 classes, element 0 = P(user
  // done talking). All four 6×2048 matmuls fuse into this dispatch.
  const vadProbs: np.Array[] = [];
  if (model.extraHeads) {
    const probs = model.extraHeads.map((head) =>
      nn.softmax(linearInline(head, x.ref).astype(np.float32), -1).slice(0, 0),
    ); // 4 × [] scalars (softmax[0] of each head)
    vadProbs.push(np.stack(probs)); // [4]
  }

  // Also expose the final-norm hidden state — the parity probe compares it
  // against the transformers fixture to localize errors. [2048], cheap.
  const hidden = x.reshape([STT_CONFIG.hiddenSize]);
  return [tokenId, hidden, vadProbs, newCaches];
});

// ---------------------------------------------------------------------------
// Public API.

export function createSttState(model: SttModel): SttState {
  return {
    caches: model.layers.map(() => ({
      key: np.zeros([KV_CAPACITY, STT_CONFIG.numHeads, STT_CONFIG.headDim], {
        dtype: np.float32,
      }),
      value: np.zeros([KV_CAPACITY, STT_CONFIG.numHeads, STT_CONFIG.headDim], {
        dtype: np.float32,
      }),
    })),
    kvCacheLen: 0,
    position: 0,
    prevToken: STT_CONFIG.bosTokenId,
  };
}

/**
 * Run one decoder step on a frame's 32 Mimi codes (pass all
 * `STT_CONFIG.audioBosTokenId` for the initial bos frame). Autoregressive:
 * awaits the argmax readback and feeds it to the next step via
 * `state.prevToken`. Returns the text token id, the final-norm hidden state
 * (for parity debugging; dispose it if unused), and `vadProbs` — the four
 * semantic-VAD "user is done talking" probabilities (one per horizon
 * `STT_CONFIG.vadHorizonsSec`, already read back; see module comment) or
 * null when the loaded weights lack the extra heads.
 */
export async function sttStep(
  model: SttModel,
  state: SttState,
  codes: ArrayLike<number>,
): Promise<{ tokenId: number; hidden: np.Array; vadProbs: Float32Array | null }> {
  const { numCodebooks, textVocabSize, codebookVocabSize } = STT_CONFIG;
  if (codes.length !== numCodebooks) {
    throw new Error(`Expected ${numCodebooks} codes, got ${codes.length}`);
  }

  // Sliding-window cache maintenance (shape ops stay in JS, mirroring
  // mimi-encode.ts): when the linear cache is full, drop the KV_SHIFT oldest
  // slots — the retained 375 still cover the whole 375-token window.
  if (state.kvCacheLen + 1 > KV_CAPACITY) {
    state.kvCacheLen -= KV_SHIFT;
    for (const c of state.caches) {
      c.key = np.pad(c.key.slice([KV_SHIFT]), { 0: [0, KV_SHIFT] });
      c.value = np.pad(c.value.slice([KV_SHIFT]), { 0: [0, KV_SHIFT] });
    }
  }

  // Flat-table row ids: text token raw, audio code q at 8001 + q*2049 + code.
  const ids = new Uint32Array(1 + numCodebooks);
  ids[0] = state.prevToken;
  for (let q = 0; q < numCodebooks; q++) {
    ids[1 + q] = textVocabSize + q * codebookVocabSize + Number(codes[q]);
  }

  let tokenArr: np.Array;
  let hidden: np.Array;
  let vadArrs: np.Array[];
  [tokenArr, hidden, vadArrs, state.caches] = runSttStepFused(
    tree.ref(model),
    state.caches,
    np.array(ids, { dtype: np.uint32 }),
    np.array(state.position, { dtype: np.int32 }),
    np.array(state.kvCacheLen, { dtype: np.int32 }),
  );
  state.position++;
  state.kvCacheLen++;

  // data() consumes tokenArr / the vad array (move semantics) — no dispose
  // needed. The GPU work is one dispatch; the second readback rides the same
  // sync point as the first.
  const tokenId = Number((await tokenArr.data())[0]);
  const vadProbs = vadArrs.length
    ? ((await vadArrs[0].data()) as Float32Array)
    : null;
  state.prevToken = tokenId;
  return { tokenId, hidden, vadProbs };
}

/** Decode text tokens to a transcript. Ids 0..3 are specials (pad included —
 * the model emits <pad> for frames with no text, i.e. most of them) and 8000
 * has no SentencePiece piece; both are skipped, matching the reference's
 * `batch_decode(..., skip_special_tokens=True)`. */
export function decodeSttTokens(
  sp: tokenizers.SentencePiece,
  tokens: number[],
): string {
  return sp.decode(
    tokens.filter((t) => t > STT_CONFIG.padTokenId && t < 8000),
  );
}

/**
 * Batch helper: transcribe a whole clip's per-frame Mimi codes. Runs the
 * reference stream layout — one bos step, then one step per frame — and
 * returns the per-STEP text tokens (frames.length + 1 of them) plus the
 * decoded transcript if a tokenizer is given. NOTE: for the trailing words to
 * flush, the codes should include ~1.5 s of encoded silence at the end (the
 * reference feature extractor pads audio_delay_seconds + 1.0 s).
 */
export async function sttFromCodes(
  model: SttModel,
  codesPerFrame: ArrayLike<number>[],
  sp?: tokenizers.SentencePiece,
): Promise<{ tokens: number[]; transcript: string | null }> {
  const state = createSttState(model);
  const tokens: number[] = [];

  const bosFrame = new Int32Array(STT_CONFIG.numCodebooks).fill(
    STT_CONFIG.audioBosTokenId,
  );
  for (let s = 0; s <= codesPerFrame.length; s++) {
    const { tokenId, hidden } = await sttStep(
      model,
      state,
      s === 0 ? bosFrame : codesPerFrame[s - 1],
    );
    hidden.dispose();
    tokens.push(tokenId);
  }
  tree.dispose(state);
  return { tokens, transcript: sp ? decodeSttTokens(sp, tokens) : null };
}

// ---------------------------------------------------------------------------
// Loading. bench/stt/export_weights.py writes clean names (embed.weight,
// lmHead.weight, layers.N..., norm.weight) so no WeightMapper is needed.

function tensorToArray(tensor: safetensors.Tensor, dtype: np.DType): np.Array {
  if (tensor.dtype !== "F16") {
    throw new Error(`Expected fp16 STT weights, got ${tensor.dtype}`);
  }
  switch (dtype) {
    case np.float16:
      return np.array(tensor.data as Float16Array<ArrayBuffer>, {
        shape: tensor.shape,
        dtype: np.float16,
      });
    case np.float32:
      return np.array(
        new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
        { shape: tensor.shape, dtype: np.float32 },
      );
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

// Per-row symmetric int8 dequant (companion `<name>.scale` F32, one scale per
// row) — the exact scheme src/llm/smollm.ts ships, so the ~2 GB fp16 download
// can halve to ~1 GB.
function dequantizeI8(
  tensor: safetensors.Tensor,
  scaleTensor: safetensors.Tensor,
  dtype: np.DType,
): np.Array {
  if (tensor.shape.length !== 2) {
    throw new Error(`Expected 2-D quantized tensor, got [${tensor.shape}]`);
  }
  const [rows, cols] = tensor.shape as [number, number];
  const q = tensor.data as Int8Array;
  const scales = scaleTensor.data as Float32Array;
  if (scales.length !== rows) {
    throw new Error(`Quantization scale length ${scales.length} != rows ${rows}`);
  }
  const out =
    dtype === np.float32
      ? new Float32Array(rows * cols)
      : new Float16Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const s = scales[r];
    const base = r * cols;
    for (let c = 0; c < cols; c++) out[base + c] = q[base + c] * s;
  }
  return np.array(out as Float16Array<ArrayBuffer>, {
    shape: tensor.shape,
    dtype,
  });
}

/** Hydrate the decoder from bench/stt/export_weights.py's fp16 or int8
 * safetensors. Weights stay fp16 on the GPU (the fp32 residual stream upcasts
 * activations, not weights — same recipe as smollm.ts). */
export async function fromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<SttModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    if (key.endsWith(".scale")) continue; // companion of a quantized tensor
    if (tensor.dtype === "I8") {
      const scale = file.tensors[`${key}.scale`];
      if (!scale) throw new Error(`Quantized tensor ${key} is missing its .scale`);
      hydrated[key] = dequantizeI8(tensor, scale, dtype);
      continue;
    }
    hydrated[key] = tensorToArray(tensor, dtype);
  }
  const model = safetensors.toNested(hydrated) as SttModel;
  if (model.layers.length !== STT_CONFIG.numLayers) {
    throw new Error(
      `Expected ${STT_CONFIG.numLayers} STT layers, found ${model.layers.length}`,
    );
  }
  // Semantic-VAD heads are OPTIONAL: older exported/cached weight files lack
  // them (vadProbs stays null and callers fall back to timer endpointing).
  // But a PARTIAL set means a corrupted export — fail loudly.
  if (model.extraHeads && model.extraHeads.length !== STT_CONFIG.numVadHeads) {
    throw new Error(
      `Expected ${STT_CONFIG.numVadHeads} VAD extra heads, found ${model.extraHeads.length}`,
    );
  }
  return blockUntilReady(model);
}

// ---------------------------------------------------------------------------
// DEV parity probe. Run via a bare vite page (no app model load needed):
//   import("/src/asr/kyutai-stt.ts").then((m) => m.sttParity())
// against the fixture from bench/stt/make_fixture.py. Reports per-frame text
// token exact-match rate vs the transformers golden run, the decoded
// transcript vs the reference transcript, final-norm hidden-state error on
// the first steps, and per-frame latency.

type SttFixture = {
  transformers: string;
  bosTokenId: number;
  audioBosTokenId: number;
  codes: number[][]; // [steps-1][32] real frames (bos frame synthesized here)
  tokens: number[]; // [steps] golden per-step text tokens
  transcript: string;
  spmTranscript: string;
  hidden: number[][]; // [<=8][2048] final-norm hidden states
};

export async function sttParity(
  fixtureUrl: string = "/bench/stt/fixture_stt.json",
  weightsUrl: string = "/bench/stt/kyutai-stt.fp16.safetensors",
  tokenizerUrl: string = "/bench/stt/tokenizer_en_fr_audio_8000.model",
) {
  await init("webgpu");
  defaultDevice("webgpu");

  const [fixture, weightsData, spData] = await Promise.all([
    fetch(fixtureUrl).then((r) => r.json()) as Promise<SttFixture>,
    fetch(weightsUrl)
      .then((r) => r.arrayBuffer())
      .then((b) => new Uint8Array(b)),
    fetch(tokenizerUrl)
      .then((r) => r.arrayBuffer())
      .then((b) => new Uint8Array(b)),
  ]);
  const model = await fromSafetensors(safetensors.parse(weightsData));
  const sp = tokenizers.SentencePiece.fromBinary(spData);

  const state = createSttState(model);
  const numSteps = fixture.codes.length + 1;
  const bosFrame = new Int32Array(STT_CONFIG.numCodebooks).fill(
    STT_CONFIG.audioBosTokenId,
  );

  const got: number[] = [];
  const frameMs: number[] = [];
  let hiddenRmseMax = 0;
  let matches = 0;
  let firstMismatch = -1;
  for (let s = 0; s < numSteps; s++) {
    // GOLDEN-INPUT teacher forcing is deliberately NOT used: the run is fully
    // autoregressive (prevToken feeds back), exactly like the reference.
    const t0 = performance.now();
    const { tokenId, hidden } = await sttStep(
      model,
      state,
      s === 0 ? bosFrame : fixture.codes[s - 1],
    );
    frameMs.push(performance.now() - t0);
    got.push(tokenId);
    if (tokenId === fixture.tokens[s]) matches++;
    else if (firstMismatch < 0) firstMismatch = s;
    if (s < fixture.hidden.length) {
      const h = (await hidden.data()) as Float32Array;
      const ref = fixture.hidden[s];
      let se = 0;
      for (let j = 0; j < ref.length; j++) se += (h[j] - ref[j]) ** 2;
      hiddenRmseMax = Math.max(hiddenRmseMax, Math.sqrt(se / ref.length));
    } else {
      hidden.dispose();
    }
  }
  tree.dispose([state, model]);

  const transcript = decodeSttTokens(sp, got);
  const refTranscript = fixture.transcript;
  // Steady-state latency: skip the first 2 frames (trace compilation).
  const steady = frameMs.slice(2).sort((a, b) => a - b);
  return {
    transformers: fixture.transformers,
    steps: numSteps,
    tokenMatch: matches / numSteps,
    firstMismatch,
    transcript,
    refTranscript,
    transcriptMatch: transcript.trim() === refTranscript.trim(),
    hiddenRmseMax,
    frameMsP50: steady[Math.floor(steady.length / 2)],
    frameMsMean: steady.reduce((a, b) => a + b, 0) / steady.length,
    weightsMB: Math.round(weightsData.length / 1e6),
  };
}

// ---------------------------------------------------------------------------
// DEV semantic-VAD behavioral probe. Run via a bare vite page:
//   import("/src/asr/kyutai-stt.ts").then((m) => m.sttVadProbe())          // "speech"
//   import("/src/asr/kyutai-stt.ts").then((m) => m.sttVadProbe("midpause"))
//
// HONESTY NOTE: no reference implementation of the extra heads exists
// outside Kyutai's Rust server, so unlike the token/hidden parity gates this
// is a BEHAVIORAL check, not golden parity. It runs the speech fixture
// (bench/mimi/fixture_speech.json: real speech ending ~2.4 s in, then padded
// silence) through the full encoder+decoder and dumps the per-frame
// P(user done talking) series (prs, one per horizon 0.5/1/2/3 s). Expected
// if the port is right: prs[2] LOW (<~0.2) while speech is ongoing, rising
// sharply within a few frames of the real end of speech and staying high
// through the silence. Mode "midpause" concatenates fixture speech + 0.6 s
// silence + the same speech again + trailing silence. MEASURED CAVEAT: the
// fixture utterance is a COMPLETE question ("What is your favorite hobby to
// do on the weekend?"), so the model legitimately calls a pause after it an
// end of turn (that is the correct semantic read); what "midpause" actually
// demonstrates is instant revocation — the probability collapses to ~0.01
// the frame speech resumes. The true semantic-vs-energy differentiator is
// mode "midclause": the same 0.64 s of silence spliced in MID-SENTENCE
// (after ~1.2 s of speech) — an energy VAD sees the identical pause, but
// the semantic head should stay low because the clause is unfinished.
type VadProbeFixture = { pcm: number[] };

export async function sttVadProbe(
  mode: "speech" | "midpause" | "midclause" = "speech",
  fixtureUrl: string = "/bench/mimi/fixture_speech.json",
  weightsUrl: string = "/bench/stt/kyutai-stt.fp16.safetensors",
  mimiWeightsUrl: string = "/bench/mimi/mimi-encoder.fp16.safetensors",
) {
  await init("webgpu");
  defaultDevice("webgpu");

  const [fixture, sttData, mimiData] = await Promise.all([
    fetch(fixtureUrl).then((r) => r.json()) as Promise<VadProbeFixture>,
    fetch(weightsUrl)
      .then((r) => r.arrayBuffer())
      .then((b) => new Uint8Array(b)),
    fetch(mimiWeightsUrl)
      .then((r) => r.arrayBuffer())
      .then((b) => new Uint8Array(b)),
  ]);
  const model = await fromSafetensors(safetensors.parse(sttData));
  if (!model.extraHeads) {
    tree.dispose(model);
    return { error: "weights have no extraHeads — re-run bench/stt/export_weights.py" };
  }
  const mimi = mimiFromSafetensors(safetensors.parse(mimiData));

  const F = MIMI_CONFIG.frameSize;
  const src = Float32Array.from(fixture.pcm);
  let pcm: Float32Array<ArrayBuffer>;
  // The fixture's real speech ends ~2.4 s in (frame 30); the rest is padded
  // silence.
  const SPEECH_END_FRAME = 30;
  if (mode === "speech") {
    // Speech + 4 s of the fixture's own trailing silence (frames past ~2.4 s).
    pcm = src.slice(0, (SPEECH_END_FRAME + 50) * F);
  } else if (mode === "midpause") {
    // Fixture speech (incl. a couple frames of its natural trailing silence),
    // a 0.64 s fake pause, the same speech again, then 2.4 s true silence.
    const seg = src.slice(0, 32 * F);
    const pauseFrames = 8; // 0.64 s
    pcm = new Float32Array((32 + pauseFrames + 32 + 30) * F);
    pcm.set(seg, 0);
    pcm.set(seg, (32 + pauseFrames) * F);
  } else {
    // Mid-CLAUSE pause: first 1.2 s of the utterance (mid-sentence), 0.64 s
    // of silence, then the remainder of the utterance, then true silence.
    const cut = 15; // frames; ~1.2 s
    const pauseFrames = 8; // 0.64 s
    const rest = 32 - cut;
    pcm = new Float32Array((cut + pauseFrames + rest + 30) * F);
    pcm.set(src.slice(0, cut * F), 0);
    pcm.set(src.slice(cut * F, 32 * F), (cut + pauseFrames) * F);
  }

  const encState = createMimiEncodeState(mimi);
  const state = createSttState(model);
  const bosFrame = new Int32Array(STT_CONFIG.numCodebooks).fill(
    STT_CONFIG.audioBosTokenId,
  );
  const numFrames = Math.floor(pcm.length / F);

  // series[s] = prs for the step that CONSUMED audio frame s (decoder step
  // s+1 — step 0 is the bos frame and consumes no audio).
  const series: number[][] = [];
  const tokens: number[] = [];
  (await sttStep(model, state, bosFrame)).hidden.dispose();
  for (let i = 0; i < numFrames; i++) {
    const { codes, preQuant } = encodeFrame(
      mimi,
      encState,
      pcm.slice(i * F, (i + 1) * F),
    );
    preQuant.dispose();
    const ids = await codes.data();
    const { tokenId, hidden, vadProbs } = await sttStep(
      model,
      state,
      ids as ArrayLike<number>,
    );
    hidden.dispose();
    tokens.push(tokenId);
    series.push([...vadProbs!].map((p) => Math.round(p * 1000) / 1000));
  }
  tree.dispose([state, model]);
  tree.dispose([encState.convStates, encState.kvCaches, mimi]);
  encState.downsampleState?.dispose();

  // Summary over prs[2] (the 2 s horizon head — the one unmute thresholds).
  const prs2 = series.map((p) => p[2]);
  const firstAbove = (from: number, thresh: number) => {
    for (let i = from; i < prs2.length; i++) if (prs2[i] > thresh) return i;
    return -1;
  };
  const maxIn = (a: number, b: number) =>
    Math.max(...prs2.slice(Math.max(0, a), Math.min(prs2.length, b)));
  const summary =
    mode === "speech"
      ? {
          // Skip the first 12 frames (unstable, per unmute) for the speech max.
          maxDuringSpeech: maxIn(12, SPEECH_END_FRAME),
          firstFrameAbove0_6: firstAbove(SPEECH_END_FRAME - 2, 0.6),
          framesFromSpeechEndTo0_6:
            firstAbove(SPEECH_END_FRAME - 2, 0.6) - SPEECH_END_FRAME,
          minAfterRise: Math.min(
            ...prs2.slice(Math.max(SPEECH_END_FRAME + 8, 0)),
          ),
        }
      : mode === "midpause"
        ? {
            // Fake pause spans frames 32..39; second speech 40..71; true end ~70.
            maxAtFakePause: maxIn(30, 42),
            maxDuringSecondSpeech: maxIn(46, 68),
            maxAfterTrueEnd: maxIn(72, prs2.length),
            firstAbove0_6AfterTrueEnd: firstAbove(70, 0.6),
          }
        : {
            // Mid-clause pause spans frames 15..22; speech resumes 23..40;
            // true end ~38 (= 30 - 15 + 23).
            maxAtMidClausePause: maxIn(14, 25),
            maxAfterTrueEnd: maxIn(40, prs2.length),
            firstAbove0_6AfterTrueEnd: firstAbove(36, 0.6),
          };
  return { mode, frames: numFrames, summary, prs2, series, tokens };
}
