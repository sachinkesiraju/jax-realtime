import { type Device, jit, numpy as np } from "@jax-js/jax";
import { ONNXModel } from "@jax-js/onnx";

const IMAGE_SIZE = 512;

type OnnxRun = (inputs: Record<string, np.Array>) => Record<string, np.Array>;
type ImageSource = HTMLVideoElement | HTMLCanvasElement | HTMLImageElement | ImageBitmap;

function dimensions(source: ImageSource): [number, number] {
  if (source instanceof HTMLVideoElement) return [source.videoWidth, source.videoHeight];
  if (source instanceof HTMLImageElement) return [source.naturalWidth, source.naturalHeight];
  return [source.width, source.height];
}

export class SmolVlmVisionEncoder {
  private readonly canvas = document.createElement("canvas");
  private readonly ctx: CanvasRenderingContext2D;

  private constructor(
    private model: ONNXModel,
    private run: OnnxRun,
    private device: Device | undefined,
  ) {
    this.canvas.width = IMAGE_SIZE;
    this.canvas.height = IMAGE_SIZE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  }

  static fromBytes(
    bytes: Uint8Array<ArrayBuffer>,
    device?: Device,
  ): SmolVlmVisionEncoder {
    const model = new ONNXModel(bytes);
    return new SmolVlmVisionEncoder(model, jit(model.run) as OnnxRun, device);
  }

  encode(source: ImageSource): np.Array {
    const [width, height] = dimensions(source);
    if (!width || !height) throw new Error("The visual source is not ready.");
    const crop = Math.min(width, height);
    this.ctx.drawImage(
      source,
      (width - crop) / 2,
      (height - crop) / 2,
      crop,
      crop,
      0,
      0,
      IMAGE_SIZE,
      IMAGE_SIZE,
    );
    const rgba = this.ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
    const rgb = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
    const plane = IMAGE_SIZE * IMAGE_SIZE;
    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
      rgb[p] = rgba[i] / 127.5 - 1;
      rgb[plane + p] = rgba[i + 1] / 127.5 - 1;
      rgb[2 * plane + p] = rgba[i + 2] / 127.5 - 1;
    }
    const output = this.run({
      pixel_values: np.array(rgb, {
        shape: [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        dtype: np.float32,
        device: this.device,
      }),
    });
    return output.image_features.reshape([64, 960]);
  }

  dispose(): void {
    this.model.dispose();
  }
}

export type { ImageSource as SmolVlmImageSource };
