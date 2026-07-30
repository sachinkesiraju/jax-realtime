// Silero VAD v5, ported to pure TypeScript. ~200k parameters and ~750k
// MACs per 32 ms frame, so it runs on plain CPU JS — no GPU and no
// onnxruntime dependency. The LSTM cell (the op @jax-js/onnx lacks) is two
// matmuls and four gates, written out below.
//
// Architecture (decoded from silero_vad.onnx's 16 kHz branch; verified
// bit-close against an onnxruntime reference, see docs/BENCHMARKS.md cycle 22):
//   input [576] = 64-sample rolling context ‖ 512-sample chunk @16 kHz
//     (the context is the official wrapper's contract — without it the STFT
//     frames misalign and speech scores collapse to ~0.16 on clean speech)
//   → reflect-pad right by 64 → [640]
//   → STFT-as-conv: fixed basis [258,1,256], stride 128 → [258,4]
//     (rows 0..129 = real, 129..258 = imag) → magnitude [129,4]
//   → Conv1d(129→128, k3, p1) + ReLU
//   → Conv1d(128→64, k3, p1, s2) + ReLU
//   → Conv1d(64→64, k3, p1, s2) + ReLU
//   → Conv1d(64→128, k3, p1) + ReLU        → [128, T]
//   → LSTM cell over the T time steps (hidden 128, PyTorch gate order ifgo)
//   → ReLU(h_final) → Conv1x1(128→1) + bias → sigmoid → P(speech)
//
// Weights ship as public/vad/silero-vad-v5.bin: [u32 manifest length]
// [manifest JSON][f32 tensor data...] (see extract-silero.mjs).

export type VadWeights = {
  stftBasis: Float32Array; // [130][128]
  enc: { w: Float32Array; b: Float32Array; cIn: number; cOut: number; stride: number }[];
  lstmWih: Float32Array; // [512][128]
  lstmWhh: Float32Array; // [512][128]
  lstmBih: Float32Array; // [512]
  lstmBhh: Float32Array; // [512]
  outW: Float32Array; // [128]
  outB: number;
};

const CHUNK = 512;
const CONTEXT = 64;
const PAD_RIGHT = 64;
const STFT_K = 256;
const STFT_HOP = 128;
const STFT_BINS = 129;
const HIDDEN = 128;

export function parseVadWeights(buf: ArrayBuffer): VadWeights {
  const view = new DataView(buf);
  const manifestLen = view.getUint32(0, true);
  const manifest = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, manifestLen)),
  ) as {
    tensors: Record<string, { dims: number[]; offset: number; length: number }>;
    constants: Record<string, { values: number[] }>;
  };
  const base = 4 + manifestLen;
  const f32 = (name: string): Float32Array => {
    const t = manifest.tensors[name];
    if (!t) throw new Error(`VAD weights missing tensor ${name}`);
    // The f32 region may not be 4-byte aligned after the JSON header; copy.
    return new Float32Array(buf.slice(base + t.offset * 4, base + (t.offset + t.length) * 4));
  };
  const encSpec = [
    { name: "encoder.0", cIn: STFT_BINS, cOut: 128, stride: 1 },
    { name: "encoder.1", cIn: 128, cOut: 64, stride: 2 },
    { name: "encoder.2", cIn: 64, cOut: 64, stride: 2 },
    { name: "encoder.3", cIn: 64, cOut: 128, stride: 1 },
  ];
  return {
    stftBasis: f32("stft.forward_basis_buffer"),
    enc: encSpec.map((s) => ({
      w: f32(`${s.name}.reparam_conv.weight`),
      b: f32(`${s.name}.reparam_conv.bias`),
      cIn: s.cIn,
      cOut: s.cOut,
      stride: s.stride,
    })),
    lstmWih: f32("decoder.rnn.weight_ih"),
    lstmWhh: f32("decoder.rnn.weight_hh"),
    lstmBih: f32("decoder.rnn.bias_ih"),
    lstmBhh: f32("decoder.rnn.bias_hh"),
    outW: f32("decoder.decoder.2.weight"),
    outB: manifest.constants["decoder.decoder.2.bias"]?.values[0] ?? 0,
  };
}

/** Conv1d with zero padding 1, kernel 3 (the encoder shape). */
function conv1dK3(
  x: Float32Array,
  tIn: number,
  cIn: number,
  w: Float32Array,
  b: Float32Array,
  cOut: number,
  stride: number,
): { y: Float32Array; tOut: number } {
  const tOut = Math.floor((tIn + 2 - 3) / stride) + 1;
  const y = new Float32Array(cOut * tOut);
  for (let oc = 0; oc < cOut; oc++) {
    const wBase = oc * cIn * 3;
    for (let ot = 0; ot < tOut; ot++) {
      let acc = b[oc];
      const t0 = ot * stride - 1; // pad 1
      for (let ic = 0; ic < cIn; ic++) {
        const xBase = ic * tIn;
        const wRow = wBase + ic * 3;
        for (let k = 0; k < 3; k++) {
          const t = t0 + k;
          if (t >= 0 && t < tIn) acc += w[wRow + k] * x[xBase + t];
        }
      }
      y[oc * tOut + ot] = acc;
    }
  }
  return { y, tOut };
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/**
 * Stateful frame-level VAD. Feed sequential 512-sample chunks of 16 kHz mono
 * PCM (32 ms each); returns P(speech) per chunk. `reset()` between separate
 * audio streams.
 */
export class SileroVad {
  private h = new Float32Array(HIDDEN);
  private c = new Float32Array(HIDDEN);
  private context = new Float32Array(CONTEXT);
  private w: VadWeights;

  // Plain assignment (not a parameter property) so the node strip-types
  // parity harness can import this file directly, like mic.ts's probe.
  constructor(weights: VadWeights) {
    this.w = weights;
  }

  reset(): void {
    this.h.fill(0);
    this.c.fill(0);
    this.context.fill(0);
  }

  /** State snapshot (h then c), for parity tests. */
  state(): Float32Array {
    const out = new Float32Array(HIDDEN * 2);
    out.set(this.h, 0);
    out.set(this.c, HIDDEN);
    return out;
  }

  process(chunk: Float32Array): number {
    if (chunk.length !== CHUNK) {
      throw new Error(`SileroVad.process expects ${CHUNK} samples, got ${chunk.length}`);
    }
    const w = this.w;

    // context ‖ chunk, then reflect-pad right by 64: x[640]
    const full = CONTEXT + CHUNK;
    const padded = new Float32Array(full + PAD_RIGHT);
    padded.set(this.context, 0);
    padded.set(chunk, CONTEXT);
    for (let i = 0; i < PAD_RIGHT; i++) {
      padded[full + i] = padded[full - 2 - i];
    }
    this.context.set(chunk.subarray(CHUNK - CONTEXT));

    // STFT as conv: [130, frames], then magnitude [65, frames].
    const frames = Math.floor((padded.length - STFT_K) / STFT_HOP) + 1;
    const mag = new Float32Array(STFT_BINS * frames);
    for (let bin = 0; bin < STFT_BINS; bin++) {
      const reBase = bin * STFT_K;
      const imBase = (bin + STFT_BINS) * STFT_K;
      for (let f = 0; f < frames; f++) {
        const off = f * STFT_HOP;
        let re = 0;
        let im = 0;
        for (let k = 0; k < STFT_K; k++) {
          const s = padded[off + k];
          re += w.stftBasis[reBase + k] * s;
          im += w.stftBasis[imBase + k] * s;
        }
        mag[bin * frames + f] = Math.sqrt(re * re + im * im);
      }
    }

    // Encoder: 4 × (conv k3 p1 + ReLU).
    let x: Float32Array = mag;
    let t = frames;
    for (const layer of w.enc) {
      const { y, tOut } = conv1dK3(x, t, layer.cIn, layer.w, layer.b, layer.cOut, layer.stride);
      for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
      x = y;
      t = tOut;
    }

    // LSTM cell over the T time steps (input 128, hidden 128, gates i,f,g,o).
    // The head (ReLU → 1x1 conv → sigmoid) is applied to EVERY step's hidden
    // state and the per-step probabilities are averaged (the graph's trailing
    // ReduceMean) — not just to the final state.
    const { h, c } = this;
    const gates = new Float32Array(4 * HIDDEN);
    let probSum = 0;
    for (let step = 0; step < t; step++) {
      for (let g = 0; g < 4 * HIDDEN; g++) {
        let acc = w.lstmBih[g] + w.lstmBhh[g];
        const ihRow = g * HIDDEN;
        for (let i = 0; i < HIDDEN; i++) {
          acc += w.lstmWih[ihRow + i] * x[i * t + step];
          acc += w.lstmWhh[ihRow + i] * h[i];
        }
        gates[g] = acc;
      }
      for (let i = 0; i < HIDDEN; i++) {
        const ig = sigmoid(gates[i]);
        const fg = sigmoid(gates[HIDDEN + i]);
        const gg = Math.tanh(gates[2 * HIDDEN + i]);
        const og = sigmoid(gates[3 * HIDDEN + i]);
        c[i] = fg * c[i] + ig * gg;
        h[i] = og * Math.tanh(c[i]);
      }
      let acc = w.outB;
      for (let i = 0; i < HIDDEN; i++) {
        const v = h[i] > 0 ? h[i] : 0;
        acc += w.outW[i] * v;
      }
      probSum += sigmoid(acc);
    }
    return probSum / t;
  }
}
