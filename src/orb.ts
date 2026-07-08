// Audio-reactive orb, in the spirit of the HF realtime voice demo: a soft
// blob that breathes when idle, swells with your voice while listening,
// shimmers while the model thinks, and pulses with the synthesized speech.

export type OrbState =
  | "off"
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "duplex";

type Palette = {
  core: string;
  mid: string;
  edge: string;
  glow: string;
};

const PALETTES: Record<OrbState, Palette> = {
  off: {
    core: "oklch(55% 0.02 220)",
    mid: "oklch(40% 0.02 220)",
    edge: "oklch(30% 0.02 220 / 0)",
    glow: "oklch(45% 0.02 220 / 0.25)",
  },
  idle: {
    core: "oklch(95% 0.16 118)",
    mid: "oklch(75% 0.16 118)",
    edge: "oklch(58% 0.13 118 / 0)",
    glow: "oklch(88% 0.19 118 / 0.4)",
  },
  listening: {
    core: "oklch(90% 0.12 25)",
    mid: "oklch(68% 0.19 25)",
    edge: "oklch(55% 0.19 25 / 0)",
    glow: "oklch(68% 0.21 25 / 0.5)",
  },
  thinking: {
    core: "oklch(94% 0.12 85)",
    mid: "oklch(78% 0.15 85)",
    edge: "oklch(62% 0.14 85 / 0)",
    glow: "oklch(84% 0.16 85 / 0.45)",
  },
  speaking: {
    core: "oklch(97% 0.17 118)",
    mid: "oklch(80% 0.19 118)",
    edge: "oklch(60% 0.14 118 / 0)",
    glow: "oklch(90% 0.21 118 / 0.55)",
  },
  // Duplex draws two palettes at once (red ring = user, lime core = assistant);
  // this entry is only a base tint for the idle core.
  duplex: {
    core: "oklch(95% 0.16 118)",
    mid: "oklch(72% 0.16 118)",
    edge: "oklch(58% 0.13 118 / 0)",
    glow: "oklch(88% 0.19 118 / 0.35)",
  },
};

export class Orb {
  private ctx: CanvasRenderingContext2D;
  private frame: number | null = null;
  private state: OrbState = "off";
  private levelSource: (() => number) | null = null;
  private smoothLevel = 0;
  private userLevelSource: (() => number) | null = null;
  private ttsLevelSource: (() => number) | null = null;
  private smoothUser = 0;
  private smoothTts = 0;
  private reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.start();
  }

  setState(state: OrbState, levelSource: (() => number) | null = null) {
    this.state = state;
    this.levelSource = levelSource;
    this.userLevelSource = null;
    this.ttsLevelSource = null;
  }

  /**
   * Full-duplex view: the user's mic level drives a red outer ring while the
   * assistant's TTS level drives the lime core — both parties visible at once.
   */
  setDuplex(
    userLevel: () => number,
    ttsLevel: (() => number) | null = null,
  ) {
    this.state = "duplex";
    this.levelSource = null;
    this.userLevelSource = userLevel;
    this.ttsLevelSource = ttsLevel;
  }

  private start() {
    const tick = (time: number) => {
      if (this.state === "duplex") this.drawDuplex(time / 1000);
      else this.draw(time / 1000);
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  destroy() {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
  }

  private draw(t: number) {
    const { canvas, ctx } = this;
    const ratio = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    if (size === 0) return;
    const px = Math.floor(size * ratio);
    if (canvas.width !== px || canvas.height !== px) {
      canvas.width = px;
      canvas.height = px;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);

    // Smoothed audio level: fast attack, slow release.
    const raw = this.levelSource ? this.levelSource() : 0;
    this.smoothLevel =
      raw > this.smoothLevel
        ? this.smoothLevel * 0.4 + raw * 0.6
        : this.smoothLevel * 0.88 + raw * 0.12;
    const level = this.smoothLevel;

    const cx = size / 2;
    const cy = size / 2;
    const baseR = size * 0.3;

    // Idle breathing / thinking pulse when no audio drives the orb.
    let breathe = 0;
    if (!this.reducedMotion) {
      if (this.state === "thinking") breathe = 0.05 * Math.sin(t * 5.2);
      else if (this.state !== "off") breathe = 0.035 * Math.sin(t * 1.4);
    }
    const r = baseR * (1 + breathe + level * 0.42);

    const palette = PALETTES[this.state];
    const wobbleAmp = this.reducedMotion
      ? 0
      : (this.state === "off" ? 0.008 : 0.02) + level * 0.06;
    const speed = this.state === "thinking" ? 2.4 : 1;

    // Outer glow halo.
    const halo = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r * 1.75);
    halo.addColorStop(0, palette.glow);
    halo.addColorStop(1, "transparent");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.75, 0, Math.PI * 2);
    ctx.fill();

    // Main blob with a gently wobbling outline.
    ctx.beginPath();
    const steps = 96;
    for (let i = 0; i <= steps; i++) {
      const theta = (i / steps) * Math.PI * 2;
      const wobble =
        1 +
        wobbleAmp * Math.sin(3 * theta + t * 1.9 * speed) +
        wobbleAmp * 0.7 * Math.sin(5 * theta - t * 2.6 * speed) +
        wobbleAmp * 0.5 * Math.sin(8 * theta + t * 3.7 * speed);
      const rr = r * wobble;
      const x = cx + rr * Math.cos(theta);
      const y = cy + rr * Math.sin(theta);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    const body = ctx.createRadialGradient(
      cx - r * 0.28,
      cy - r * 0.32,
      r * 0.1,
      cx,
      cy,
      r * 1.05,
    );
    body.addColorStop(0, palette.core);
    body.addColorStop(0.55, palette.mid);
    body.addColorStop(1, palette.edge);
    ctx.fillStyle = body;
    ctx.fill();

    // Specular highlight for depth.
    const spec = ctx.createRadialGradient(
      cx - r * 0.35,
      cy - r * 0.42,
      0,
      cx - r * 0.35,
      cy - r * 0.42,
      r * 0.55,
    );
    spec.addColorStop(0, "oklch(100% 0 0 / 0.5)");
    spec.addColorStop(1, "transparent");
    ctx.fillStyle = spec;
    ctx.beginPath();
    ctx.arc(cx - r * 0.35, cy - r * 0.42, r * 0.55, 0, Math.PI * 2);
    ctx.fill();
  }

  /** Two-party full-duplex render: red user ring + lime assistant core. */
  private drawDuplex(t: number) {
    const { canvas, ctx } = this;
    const ratio = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    if (size === 0) return;
    const px = Math.floor(size * ratio);
    if (canvas.width !== px || canvas.height !== px) {
      canvas.width = px;
      canvas.height = px;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const rawUser = this.userLevelSource ? this.userLevelSource() : 0;
    const rawTts = this.ttsLevelSource ? this.ttsLevelSource() : 0;
    this.smoothUser =
      rawUser > this.smoothUser
        ? this.smoothUser * 0.4 + rawUser * 0.6
        : this.smoothUser * 0.86 + rawUser * 0.14;
    this.smoothTts =
      rawTts > this.smoothTts
        ? this.smoothTts * 0.4 + rawTts * 0.6
        : this.smoothTts * 0.86 + rawTts * 0.14;

    const cx = size / 2;
    const cy = size / 2;
    const baseR = size * 0.3;
    const breathe = this.reducedMotion ? 0 : 0.03 * Math.sin(t * 1.4);

    // Lime core driven by the assistant's TTS output.
    const coreR = baseR * (0.82 + breathe + this.smoothTts * 0.4);
    const speaking = PALETTES.speaking;
    const idleCore = PALETTES.duplex;
    const halo = ctx.createRadialGradient(
      cx,
      cy,
      coreR * 0.5,
      cx,
      cy,
      coreR * 1.7,
    );
    halo.addColorStop(0, this.smoothTts > 0.02 ? speaking.glow : idleCore.glow);
    halo.addColorStop(1, "transparent");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, coreR * 1.7, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
    const body = ctx.createRadialGradient(
      cx - coreR * 0.28,
      cy - coreR * 0.32,
      coreR * 0.1,
      cx,
      cy,
      coreR * 1.05,
    );
    const palette = this.smoothTts > 0.02 ? speaking : idleCore;
    body.addColorStop(0, palette.core);
    body.addColorStop(0.55, palette.mid);
    body.addColorStop(1, palette.edge);
    ctx.fillStyle = body;
    ctx.fill();

    // The user's mic level shows as a soft, on-palette halo that expands with
    // your voice — cohesive with the lime orb rather than a hard red ring.
    const userR = baseR * (1.15 + this.smoothUser * 0.55);
    const intensity = Math.min(1, this.smoothUser * 1.3);
    if (intensity > 0.02) {
      const halo = ctx.createRadialGradient(
        cx,
        cy,
        userR * 0.72,
        cx,
        cy,
        userR * 1.15,
      );
      halo.addColorStop(0, "transparent");
      halo.addColorStop(
        0.6,
        `oklch(86% 0.15 118 / ${(0.14 + intensity * 0.34).toFixed(3)})`,
      );
      halo.addColorStop(1, "transparent");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(cx, cy, userR * 1.15, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** RMS level in [0, 1] from an AnalyserNode. */
export function analyserLevel(
  analyser: AnalyserNode,
  buffer: Float32Array<ArrayBuffer>,
): number {
  analyser.getFloatTimeDomainData(buffer);
  let sum = 0;
  for (const v of buffer) sum += v * v;
  return Math.min(1, Math.sqrt(sum / buffer.length) * 4);
}
