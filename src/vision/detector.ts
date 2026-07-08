// D-FINE object detector on jax-js. Wraps `@jax-js/onnx`'s ONNXModel (which runs
// on jax-js's own backends — NOT onnxruntime) and exposes a small detect() API
// over a webcam frame. Ported from the jax-js `d-fine` demo page; input prep is
// a 640×640 center-crop letterbox, RGB /255, NCHW, plus an `orig_target_sizes`
// tensor so the model rescales boxes into that 640-space, which we then map back
// to source pixels.

import {
  blockUntilReady,
  type Device,
  jit,
  numpy as np,
  tree,
} from "@jax-js/jax";
import { ONNXModel } from "@jax-js/onnx";

import { fetchWithProgress, type ProgressFn } from "../pipeline";
import { COCO_80_CLASSES, COCO_CLASSES } from "./coco";

const MODEL_SIZE = 640;
const DEFAULT_MODEL_URL =
  "https://huggingface.co/bukuroo/D-FINE-ONNX/resolve/main/dfine_s_obj2coco.onnx";
const DEFAULT_THRESHOLD = 0.45;

export type Detection = {
  label: string;
  score: number;
  /** [x, y, w, h] in source-pixel coordinates. */
  box: [number, number, number, number];
  /** Dominant colour name, filled in by the vision layer (cheap pixel average
   *  over the box) — the detector itself doesn't compute it. */
  color?: string;
};

type OnnxRun = (
  inputs: Record<string, np.Array>,
  options?: unknown,
) => Record<string, np.Array>;

type FrameSource = HTMLVideoElement | HTMLCanvasElement;

function sourceSize(source: FrameSource): { w: number; h: number } {
  if (source instanceof HTMLVideoElement) {
    return { w: source.videoWidth, h: source.videoHeight };
  }
  return { w: source.width, h: source.height };
}

function labelFor(id: number): string {
  // D-FINE obj2coco emits labels in the dense 0–79 space; fall back to the
  // sparse 0–90 COCO space if an out-of-range id shows up.
  return COCO_80_CLASSES[id] ?? COCO_CLASSES[id] ?? `class ${id}`;
}

export class ObjectDetector {
  private busy = false;
  private warmed = false;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  private constructor(
    private model: ONNXModel,
    private run: OnnxRun,
    private threshold: number,
    private device: Device | undefined,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = MODEL_SIZE;
    this.canvas.height = MODEL_SIZE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  }

  static async load(
    onProgress: ProgressFn,
    {
      device,
      threshold = DEFAULT_THRESHOLD,
      url = DEFAULT_MODEL_URL,
    }: { device?: Device; threshold?: number; url?: string } = {},
  ): Promise<ObjectDetector> {
    const bytes = await fetchWithProgress("D-FINE detector weights", url, onProgress);
    const model = new ONNXModel(bytes);
    // Second (options) arg is static, exactly as the jax-js d-fine demo does it.
    const run = jit(model.run, { staticArgnums: [1] }) as OnnxRun;
    return new ObjectDetector(model, run, threshold, device);
  }

  get isBusy(): boolean {
    return this.busy;
  }

  /** Run one throwaway pass so kernel compilation happens off the hot path. */
  async warmup(): Promise<void> {
    if (this.warmed) return;
    try {
      this.ctx.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
      const { images, sizes } = this.prepareInputs();
      const outputs = this.run({ images, orig_target_sizes: sizes });
      await blockUntilReady(outputs);
      tree.dispose(outputs);
      this.warmed = true;
    } catch {
      // Best-effort; the first real detect pass just pays the compile cost.
    }
  }

  /** Detect COCO objects in one frame. One pass at a time (guarded). */
  async detect(source: FrameSource): Promise<Detection[]> {
    if (this.busy) return [];
    const { w: srcW, h: srcH } = sourceSize(source);
    if (!srcW || !srcH) return [];
    this.busy = true;

    // Center-crop the largest square from the source and scale to 640×640.
    const crop = Math.min(srcW, srcH);
    const sx = (srcW - crop) / 2;
    const sy = (srcH - crop) / 2;
    this.ctx.drawImage(source, sx, sy, crop, crop, 0, 0, MODEL_SIZE, MODEL_SIZE);

    let outputs: Record<string, np.Array> | null = null;
    try {
      const { images, sizes } = this.prepareInputs();
      outputs = this.run({ images, orig_target_sizes: sizes });
      await blockUntilReady(outputs);
      // `.data()` consumes (frees) the array — read through `.ref` so the
      // originals survive for `tree.dispose(outputs)` below. Reading them
      // directly freed the tensors, then the dispose double-freed and threw,
      // which left `this.busy` stuck true and silently killed all detection.
      const labels = await outputs.labels.ref.data();
      const boxes = await outputs.boxes.ref.data();
      const scores = await outputs.scores.ref.data();
      return this.parse(labels, boxes, scores, sx, sy, crop / MODEL_SIZE);
    } catch {
      return [];
    } finally {
      this.busy = false;
      if (outputs) {
        try {
          tree.dispose(outputs);
        } catch {
          // Already released; never let cleanup wedge the busy flag.
        }
      }
    }
  }

  dispose(): void {
    this.model.dispose();
  }

  private prepareInputs(): { images: np.Array; sizes: np.Array } {
    const pixels = this.ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const images = np
      .array(new Float32Array(new Uint8Array(pixels.buffer)), {
        shape: [MODEL_SIZE, MODEL_SIZE, 4],
        device: this.device,
      })
      .slice([], [], [0, 3])
      .mul(1 / 255)
      .transpose([2, 0, 1])
      .reshape([1, 3, MODEL_SIZE, MODEL_SIZE]);
    const sizes = np.array(new Int32Array([MODEL_SIZE, MODEL_SIZE]), {
      shape: [1, 2],
      dtype: np.int32,
      device: this.device,
    });
    return { images, sizes };
  }

  private parse(
    labels: ArrayLike<number>,
    boxes: ArrayLike<number>,
    scores: ArrayLike<number>,
    sx: number,
    sy: number,
    scale: number, // source-pixels per model-pixel (crop / 640)
  ): Detection[] {
    const count = Math.min(
      labels.length,
      scores.length,
      Math.floor(boxes.length / 4),
    );
    const items: Detection[] = [];
    for (let i = 0; i < count; i++) {
      const score = Number(scores[i]);
      if (score < this.threshold) continue;
      const o = i * 4;
      let x1 = Number(boxes[o]);
      let y1 = Number(boxes[o + 1]);
      let x2 = Number(boxes[o + 2]);
      let y2 = Number(boxes[o + 3]);
      // Some exports emit normalized [0,1] boxes; scale those into 640-space.
      if (Math.max(x1, y1, x2, y2) <= 1.5) {
        x1 *= MODEL_SIZE;
        y1 *= MODEL_SIZE;
        x2 *= MODEL_SIZE;
        y2 *= MODEL_SIZE;
      }
      const x = sx + x1 * scale;
      const y = sy + y1 * scale;
      const w = (x2 - x1) * scale;
      const h = (y2 - y1) * scale;
      if (w < 1 || h < 1) continue;
      items.push({ label: labelFor(Number(labels[i])), score, box: [x, y, w, h] });
    }
    return items.sort((a, b) => b.score - a.score).slice(0, 40);
  }
}
