// SmolLM2-360M-Instruct forward pass on jax-js — the brain, chosen via a
// blind-judged model shootout in which it won every dimension (helpfulness,
// coherence, spoken-naturalness, smarts) across three runs against same-size
// and larger alternatives.
//
// It's a plain Llama architecture:
//   * pre-norm only: 2 norms/layer (input, post-attention), applied to each
//     sublayer INPUT; the raw sublayer output is added to the residual.
//   * RMSNorm scale is the raw weight (output = norm(x) * weight).
//   * no query/key norm; embeddings are not scaled by sqrt(hidden).
//   * a single RoPE theta for every layer (no local/global interleave), and
//     full causal attention on every layer (no sliding window).
//   * SiLU MLP; attention scale is 1/sqrt(head_dim).
//   * tied embeddings (lm_head == embed_tokens).
import { blockUntilReady, jit, lax, nn, numpy as np, tree } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

export const SMOLLM_CONFIG = {
  bosTokenId: 1, // <|im_start|>
  eosTokenId: 2, // <|im_end|>
  padTokenId: 2,
  vocabSize: 49_152,
  hiddenSize: 960,
  intermediateSize: 2560,
  numHiddenLayers: 32,
  numAttentionHeads: 15,
  numKeyValueHeads: 5,
  headDim: 64,
  rmsNormEps: 1e-5,
  ropeTheta: 100_000,
} as const;

export type Linear = {
  weight: np.Array;
  bias?: np.Array;
};

export type RMSNorm = {
  weight: np.Array;
};

export type SmolLmMLP = {
  gateProj: Linear;
  upProj: Linear;
  downProj: Linear;
};

export type SmolLmAttention = {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  oProj: Linear;
};

export type SmolLmDecoderLayer = {
  inputLayernorm: RMSNorm;
  postAttentionLayernorm: RMSNorm;
  selfAttn: SmolLmAttention;
  mlp: SmolLmMLP;
};

export type SmolLmModel = {
  embedTokens: Linear;
  layers: SmolLmDecoderLayer[];
  norm: RMSNorm;
};

export type SmolLmKVCache = {
  key: np.Array; // [capacity, num_key_value_heads, head_dim]
  value: np.Array; // [capacity, num_key_value_heads, head_dim]
};

export type SmolLmState = {
  caches: SmolLmKVCache[];
  position: number;
  capacity: number;
};

const ATTENTION_SCALE = 1 / Math.sqrt(SMOLLM_CONFIG.headDim);
const KV_CACHE_BLOCK_SIZE = 512;

const KV_GROUPS =
  SMOLLM_CONFIG.numAttentionHeads / SMOLLM_CONFIG.numKeyValueHeads; // 15 / 5 = 3

// Repeat_interleave KV heads to match the query-head count (HF Llama GQA).
// Input [S, K, H] -> output [S, K*KV_GROUPS, H] with head order
// [k0,k0,k0, k1,k1,k1, ...]. jax-js dotProductAttention would otherwise TILE
// (block-repeat) the KV heads, which pairs query heads with the wrong KV head
// for K>1.
function repeatKvHeads(x: np.Array): np.Array {
  const [S, K, H] = x.shape;
  return np
    .tile(x.reshape([S, K, 1, H]), [1, 1, KV_GROUPS, 1])
    .reshape([S, K * KV_GROUPS, H]);
}

const runLinear = jit(function runLinear(
  { weight, bias }: Linear,
  x: np.Array,
): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
});

const runEmbedding = jit(function runEmbedding(
  { weight }: Linear,
  tokenIds: np.Array,
): np.Array {
  // Trained for bf16 activations; keep the residual stream in fp32 so fp16
  // residuals don't overflow in this browser implementation.
  // SmolLM2/Llama does NOT scale embeddings by sqrt(hidden).
  return weight.slice(tokenIds).astype(np.float32);
});

const runSmolLmRMSNorm = jit(
  function runSmolLmRMSNorm(
    { weight }: RMSNorm,
    x: np.Array,
    eps: number = SMOLLM_CONFIG.rmsNormEps,
  ): np.Array {
    const dtype = x.dtype;
    x = x.astype(np.float32);
    const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
    x = x.div(np.sqrt(rms.add(eps)));
    // Llama RMSNorm: output = norm(x) * weight (no zero-centered offset).
    return x.mul(weight.astype(np.float32)).astype(dtype);
  },
  { staticArgnums: [2] },
);

function rotateHalf(x: np.Array): np.Array {
  const [x1, x2] = np.split(x, 2, -1);
  return np.concatenate([x2.mul(-1), x1], -1);
}

function applySmolLmRoPE(
  q: np.Array, // [T, num_heads, head_dim]
  k: np.Array, // [T, num_key_value_heads, head_dim]
  offset: number | np.Array,
  theta: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;

  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(theta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq); // [T, head_dim / 2]

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

function runMLP({ gateProj, upProj, downProj }: SmolLmMLP, x: np.Array) {
  const gate = nn.silu(runLinear(gateProj, x.ref));
  const up = runLinear(upProj, x);
  return runLinear(downProj, gate.mul(up));
}

function runAttentionPrefill(
  { qProj, kProj, vProj, oProj }: SmolLmAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  let q = runLinear(qProj, x.ref).reshape([
    T,
    SMOLLM_CONFIG.numAttentionHeads,
    SMOLLM_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    SMOLLM_CONFIG.numKeyValueHeads,
    SMOLLM_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    SMOLLM_CONFIG.numKeyValueHeads,
    SMOLLM_CONFIG.headDim,
  ]);

  [q, k] = applySmolLmRoPE(q, k, 0, SMOLLM_CONFIG.ropeTheta);

  const attn = nn.dotProductAttention(q, repeatKvHeads(k.ref), repeatKvHeads(v.ref), {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, SMOLLM_CONFIG.numAttentionHeads * SMOLLM_CONFIG.headDim]),
  );
  return { output, key: k, value: v };
}

function runAttentionStep(
  { qProj, kProj, vProj, oProj }: SmolLmAttention,
  cache: SmolLmKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: SmolLmKVCache } {
  const T = 1;
  let q = runLinear(qProj, x.ref).reshape([
    T,
    SMOLLM_CONFIG.numAttentionHeads,
    SMOLLM_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    SMOLLM_CONFIG.numKeyValueHeads,
    SMOLLM_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    SMOLLM_CONFIG.numKeyValueHeads,
    SMOLLM_CONFIG.headDim,
  ]);

  [q, k] = applySmolLmRoPE(q, k, position, SMOLLM_CONFIG.ropeTheta);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);

  const validMask = np.arange(capacity).less(validLength);
  const attn = nn.dotProductAttention(q, repeatKvHeads(key.ref), repeatKvHeads(value.ref), {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, SMOLLM_CONFIG.numAttentionHeads * SMOLLM_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}

function padCache(
  key: np.Array,
  value: np.Array,
  capacity: number,
): SmolLmKVCache {
  const T = key.shape[0];
  if (T > capacity) {
    throw new Error(`Prompt length ${T} exceeds cache capacity ${capacity}`);
  }
  if (T === capacity) return { key, value };
  return {
    key: np.pad(key, { 0: [0, capacity - T] }),
    value: np.pad(value, { 0: [0, capacity - T] }),
  };
}

export const runSmolLmDecoderLayerPrefill = jit(
  function runSmolLmDecoderLayerPrefill(
    { inputLayernorm, postAttentionLayernorm, selfAttn, mlp }: SmolLmDecoderLayer,
    x: np.Array,
    capacity: number,
  ): [np.Array, SmolLmKVCache] {
    // Pre-norm: residual + sublayer(norm(x)).
    const residual = x.ref;
    x = runSmolLmRMSNorm(inputLayernorm, x);
    const { output: attnOut, key, value } = runAttentionPrefill(selfAttn, x);
    x = residual.add(attnOut);

    const residual2 = x.ref;
    x = runSmolLmRMSNorm(postAttentionLayernorm, x);
    x = runMLP(mlp, x);
    x = residual2.add(x);

    return [x, padCache(key, value, capacity)];
  },
  { staticArgnums: [2] },
);

export const runSmolLmDecoderLayerStep = jit(
  function runSmolLmDecoderLayerStep(
    { inputLayernorm, postAttentionLayernorm, selfAttn, mlp }: SmolLmDecoderLayer,
    cache: SmolLmKVCache,
    x: np.Array,
    position: number,
    slot: number,
    validLength: number,
  ): [np.Array, SmolLmKVCache] {
    const residual = x.ref;
    x = runSmolLmRMSNorm(inputLayernorm, x);
    const { output: attnOut, cache: updatedCache } = runAttentionStep(
      selfAttn,
      cache,
      x,
      position,
      slot,
      validLength,
    );
    x = residual.add(attnOut);

    const residual2 = x.ref;
    x = runSmolLmRMSNorm(postAttentionLayernorm, x);
    x = runMLP(mlp, x);
    x = residual2.add(x);

    return [x, updatedCache];
  },
);

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createSmolLmState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): SmolLmState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: SMOLLM_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros(
        [capacity, SMOLLM_CONFIG.numKeyValueHeads, SMOLLM_CONFIG.headDim],
        { dtype },
      ),
      value: np.zeros(
        [capacity, SMOLLM_CONFIG.numKeyValueHeads, SMOLLM_CONFIG.headDim],
        { dtype },
      ),
    })),
  };
}

export function ensureSmolLmStateCapacity(
  state: SmolLmState,
  requiredCapacity: number,
) {
  if (state.capacity >= requiredCapacity) return;

  const oldCapacity = state.capacity;
  const newCapacity = roundCacheCapacity(requiredCapacity);
  for (const cache of state.caches) {
    cache.key = np.pad(cache.key, { 0: [0, newCapacity - oldCapacity] });
    cache.value = np.pad(cache.value, { 0: [0, newCapacity - oldCapacity] });
  }
  state.capacity = newCapacity;
}

// `realLength` supports bucket-padded prefills (TUNABLES.llmPrefillBucket):
// tokenIds may be padded past the real prompt so the per-layer prefill jit's
// trace (keyed on avals, i.e. on T) is reused across turns instead of
// re-tracing all 32 layers for every new prompt length. Exactness: pad tokens
// sit at the END, real-token queries are causal (attend only to j <= i, all
// real), and the logits are gathered at realLength-1 (the last REAL token) —
// so the returned distribution is identical to an unpadded prefill. The pad
// rows do write garbage KV (wrong RoPE positions, pad-token content) into
// slots [realLength, tokenIds.shape[0]), but those slots are never attended
// later: state.position is set to realLength, every decode step's validMask
// admits only slots < position+1, and each step overwrites slot == position
// with real KV before position advances past it — garbage is always
// overwritten before it becomes attendable.
export function runSmolLmPrefill(
  model: SmolLmModel,
  tokenIds: np.Array,
  state: SmolLmState,
  realLength: number = tokenIds.shape[0],
): np.Array {
  // Capacity must cover the PADDED length — the pad rows' KV slots are
  // written (then later overwritten) even though they are never attended.
  ensureSmolLmStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < SMOLLM_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runSmolLmDecoderLayerPrefill(
      model.layers[i],
      x,
      state.capacity,
    );
  }

  x = runSmolLmRMSNorm(model.norm, x);
  x = x.slice([realLength - 1, realLength]);
  const logits = runLinear(model.embedTokens, x).reshape([
    SMOLLM_CONFIG.vocabSize,
  ]);
  state.position = realLength;
  return logits;
}

export function runSmolLmStep(
  model: SmolLmModel,
  tokenId: number,
  state: SmolLmState,
): np.Array {
  ensureSmolLmStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < SMOLLM_CONFIG.numHiddenLayers; i++) {
    [x, state.caches[i]] = runSmolLmDecoderLayerStep(
      model.layers[i],
      state.caches[i],
      x,
      position,
      slot,
      validLength,
    );
  }

  x = runSmolLmRMSNorm(model.norm, x);
  const logits = runLinear(model.embedTokens, x).reshape([
    SMOLLM_CONFIG.vocabSize,
  ]);
  state.position++;
  return logits;
}

// ---------------------------------------------------------------------------
// Fused single-dispatch decode step — inlines the whole step into one jitted
// function so it traces to a single dispatch group. The helpers below are
// plain, non-jitted inline copies of the jitted originals (nested-jit
// boundaries would split the dispatch group), and `position` is passed as a
// traced np.Array so one trace is shared across every token/step.

function embedInline(weight: np.Array, tokenIds: np.Array): np.Array {
  return weight.slice(tokenIds).astype(np.float32);
}

function linearInline({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
}

function rmsNormInline({ weight }: RMSNorm, x: np.Array): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  x = x.div(np.sqrt(rms.add(SMOLLM_CONFIG.rmsNormEps)));
  return x.mul(weight.astype(np.float32)).astype(dtype);
}

function mlpInline({ gateProj, upProj, downProj }: SmolLmMLP, x: np.Array) {
  const gate = nn.silu(linearInline(gateProj, x.ref));
  const up = linearInline(upProj, x);
  return linearInline(downProj, gate.mul(up));
}

function ropeInline(
  q: np.Array,
  k: np.Array,
  offset: np.Array,
  theta: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;

  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(theta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq);

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

function attentionStepInline(
  { qProj, kProj, vProj, oProj }: SmolLmAttention,
  cache: SmolLmKVCache,
  x: np.Array,
  position: np.Array,
  slotMask: np.Array,
  validMask: np.Array,
): { output: np.Array; cache: SmolLmKVCache } {
  const T = 1;
  let q = linearInline(qProj, x.ref).reshape([
    T,
    SMOLLM_CONFIG.numAttentionHeads,
    SMOLLM_CONFIG.headDim,
  ]);
  let k = linearInline(kProj, x.ref).reshape([
    T,
    SMOLLM_CONFIG.numKeyValueHeads,
    SMOLLM_CONFIG.headDim,
  ]);
  const v = linearInline(vProj, x).reshape([
    T,
    SMOLLM_CONFIG.numKeyValueHeads,
    SMOLLM_CONFIG.headDim,
  ]);

  [q, k] = ropeInline(q, k, position, SMOLLM_CONFIG.ropeTheta);

  const capacity = cache.key.shape[0];
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);

  const attn = nn.dotProductAttention(q, repeatKvHeads(key.ref), repeatKvHeads(value.ref), {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = linearInline(
    oProj,
    attn.reshape([T, SMOLLM_CONFIG.numAttentionHeads * SMOLLM_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}

function smolLmDecodeStepBody(
  model: SmolLmModel,
  caches: SmolLmKVCache[],
  tokenId: np.Array,
  position: np.Array,
): [np.Array, SmolLmKVCache[]] {
  const capacity = caches[0].key.shape[0];

  const slotMask = np
    .arange(capacity)
    .equal(position.ref)
    .reshape([capacity, 1, 1]);
  const validMask = np.arange(capacity).lessEqual(position.ref);

  let x = embedInline(model.embedTokens.weight.ref, tokenId);

  const newCaches: SmolLmKVCache[] = [];
  for (let i = 0; i < SMOLLM_CONFIG.numHiddenLayers; i++) {
    const layer = model.layers[i];

    const residual = x.ref;
    x = rmsNormInline(layer.inputLayernorm, x);
    const { output: attnOut, cache: newCache } = attentionStepInline(
      layer.selfAttn,
      caches[i],
      x,
      position.ref,
      slotMask.ref,
      validMask.ref,
    );
    x = residual.add(attnOut);

    const residual2 = x.ref;
    x = rmsNormInline(layer.postAttentionLayernorm, x);
    x = mlpInline(layer.mlp, x);
    x = residual2.add(x);

    newCaches.push(newCache);
  }

  slotMask.dispose();
  validMask.dispose();
  position.dispose();

  x = rmsNormInline(model.norm, x);
  const logits = linearInline(model.embedTokens, x).reshape([
    SMOLLM_CONFIG.vocabSize,
  ]);
  return [logits, newCaches];
}

export const runSmolLmDecodeStepFused = jit(function runSmolLmDecodeStepFused(
  model: SmolLmModel,
  caches: SmolLmKVCache[],
  tokenId: np.Array,
  position: np.Array,
): [np.Array, SmolLmKVCache[]] {
  return smolLmDecodeStepBody(model, caches, tokenId, position);
});

export const SMOLLM_TOPK = 64;

export const runSmolLmDecodeStepFusedTopK = jit(
  function runSmolLmDecodeStepFusedTopK(
    model: SmolLmModel,
    caches: SmolLmKVCache[],
    tokenId: np.Array,
    position: np.Array,
  ): [np.Array, SmolLmKVCache[]] {
    const [logits, newCaches] = smolLmDecodeStepBody(
      model,
      caches,
      tokenId,
      position,
    );
    const [vals, idx] = lax.topK(logits, SMOLLM_TOPK); // consumes logits
    const combined = np.concatenate([
      vals.astype(np.float32),
      idx.astype(np.float32),
    ]);
    return [combined, newCaches];
  },
);

export function runSmolLmStepFused(
  model: SmolLmModel,
  tokenId: number,
  state: SmolLmState,
): np.Array {
  ensureSmolLmStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  const posArr = np.array([state.position], { dtype: np.int32 });
  let logits: np.Array;
  [logits, state.caches] = runSmolLmDecodeStepFused(
    tree.ref(model),
    state.caches,
    tokenIds,
    posArr,
  );
  state.position++;
  return logits;
}

export function runSmolLmStepFusedTopK(
  model: SmolLmModel,
  tokenId: number,
  state: SmolLmState,
): np.Array {
  ensureSmolLmStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  const posArr = np.array([state.position], { dtype: np.int32 });
  let combined: np.Array;
  [combined, state.caches] = runSmolLmDecodeStepFusedTopK(
    tree.ref(model),
    state.caches,
    tokenIds,
    posArr,
  );
  state.position++;
  return combined;
}

const mapper = new WeightMapper({
  prefix: {
    "model.": "",
  },
  substring: {
    embed_tokens: "embedTokens",
    input_layernorm: "inputLayernorm",
    post_attention_layernorm: "postAttentionLayernorm",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    o_proj: "oProj",
    gate_proj: "gateProj",
    up_proj: "upProj",
    down_proj: "downProj",
  },
});

function tensorToArray(
  tensor: safetensors.Tensor,
  dtype: np.DType = np.float16,
): np.Array {
  if (tensor.dtype !== "F16") {
    throw new Error(
      `Expected fp16 SmolLM weights, but tensor has dtype ${tensor.dtype}.`,
    );
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
// row), so the weights — including the tied embedding table — can ship int8.
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
    throw new Error(
      `Quantization scale length ${scales.length} != rows ${rows}`,
    );
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
  return np.array(out as Float16Array<ArrayBuffer>, { shape: tensor.shape, dtype });
}

export async function fromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<SmolLmModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    if (key.endsWith(".scale")) continue; // companion of a quantized tensor
    // lm_head is tied to embed_tokens; ignore a materialized copy if present.
    if (key === "lm_head.weight") continue;
    if (tensor.dtype === "I8") {
      const scale = file.tensors[`${key}.scale`];
      if (!scale) {
        throw new Error(`Quantized tensor ${key} is missing its .scale`);
      }
      hydrated[mapper.mapKey(key)] = dequantizeI8(tensor, scale, dtype);
      continue;
    }
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }

  const model = safetensors.toNested(hydrated) as SmolLmModel;
  if (model.layers.length !== SMOLLM_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${SMOLLM_CONFIG.numHiddenLayers} SmolLM layers, ` +
        `found ${model.layers.length}`,
    );
  }
  return blockUntilReady(model);
}
