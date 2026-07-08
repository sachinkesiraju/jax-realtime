// Webcam "Eye" stage. Owns the camera + a ~1.2 s detection loop over the
// ObjectDetector, tracks a small amount of derived scene state (people, phones,
// a coarse posture proxy) and produces a compact scene string for the LLM. The
// DuplexSession reads these getters to drive proactive interjections and to
// ground "what do you see?" turns.

import type { Detection, ObjectDetector } from "./detector";

// Vision is the lowest-priority stage — scene context is used far less than
// ASR/LLM/TTS — so it detects infrequently and yields the GPU to the audio
// pipeline (see `pauseWhile`).
const FRAME_INTERVAL_MS = 2200;
// Retry cadence after a tick was skipped for priority reasons — cheap (no GPU
// work happened) and keeps the Eye from starving while audio stays busy.
const SKIP_RETRY_MS = 250;
const POSTURE_BASELINE = 6; // rolling frames used for the slouch baseline
// A label must appear in ≥2 of the last STABILITY_FRAMES detector frames to be
// reported — this filters flickering false positives (a "cat" that D-FINE only
// hallucinates for a single frame) out of what we tell the user.
const STABILITY_FRAMES = 3;

export type SceneState = {
  personCount: number;
  personPresent: boolean;
  phonePresent: boolean;
  slouching: boolean;
};

function largestPerson(detections: Detection[]): Detection | null {
  let best: Detection | null = null;
  let bestArea = 0;
  for (const d of detections) {
    if (d.label !== "person") continue;
    const area = d.box[2] * d.box[3];
    if (area > bestArea) {
      bestArea = area;
      best = d;
    }
  }
  return best;
}

export class VisionSession {
  latest: Detection[] = [];
  readonly video: HTMLVideoElement;

  private stream: MediaStream | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;

  // Coarse posture proxy: keep a short rolling baseline of the largest person's
  // box top (y) and height; "slouching" ≈ top drops AND height shrinks vs. it.
  private topHistory: number[] = [];
  private heightHistory: number[] = [];
  private slouchingNow = false;

  // Recent per-frame label sets + the labels currently considered stable.
  private recentLabels: string[][] = [];
  private stableSet = new Set<string>();
  private colorCanvas: HTMLCanvasElement | null = null;

  /** When set and it returns true, a detection frame is skipped so the audio
   *  pipeline (ASR/LLM/TTS) keeps the GPU to itself. */
  pauseWhile: (() => boolean) | null = null;

  constructor(private detector: ObjectDetector) {
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.setAttribute("playsinline", "");
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.running = true;
    this.loop();
  }

  stop(): void {
    this.running = false;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.latest = [];
    this.topHistory = [];
    this.heightHistory = [];
    this.slouchingNow = false;
    this.recentLabels = [];
    this.stableSet = new Set();
  }

  /** Detections in the latest frame limited to temporally-stable labels. */
  private stableDetections(): Detection[] {
    return this.latest.filter((d) => this.stableSet.has(d.label));
  }

  /**
   * Cheap colour enrichment: average the pixels inside each detection box (for a
   * person, its torso region ≈ clothing) and name the dominant colour. No model
   * — just canvas pixels — so it rides along in the low-priority Eye loop and
   * lets the agent answer "what colour is my chair" / "what am I wearing".
   */
  private sampleColors(detections: Detection[]): void {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh || detections.length === 0) return;
    const cw = 200;
    const ch = Math.max(1, Math.round((vh * cw) / vw));
    const canvas = (this.colorCanvas ??= document.createElement("canvas"));
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    try {
      ctx.drawImage(this.video, 0, 0, cw, ch);
    } catch {
      return;
    }
    const sx = cw / vw;
    const sy = ch / vh;
    for (const d of detections) {
      let [x, y, w, h] = d.box;
      if (d.label === "person") {
        // Torso ≈ clothing: middle horizontally, upper-mid vertically.
        x += w * 0.3;
        w *= 0.4;
        y += h * 0.42;
        h *= 0.28;
      } else {
        x += w * 0.25;
        y += h * 0.25;
        w *= 0.5;
        h *= 0.5;
      }
      const rx = Math.max(0, Math.round(x * sx));
      const ry = Math.max(0, Math.round(y * sy));
      const rw = Math.max(1, Math.min(Math.round(w * sx), cw - rx));
      const rh = Math.max(1, Math.min(Math.round(h * sy), ch - ry));
      try {
        const data = ctx.getImageData(rx, ry, rw, rh).data;
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let i = 0; i < data.length; i += 4) {
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          n++;
        }
        if (n) d.color = colorName(r / n, g / n, b / n);
      } catch {
        // Ignore a bad region; the object just won't have a colour this frame.
      }
    }
  }

  /** Find the stable detection a colour/appearance question refers to. */
  private targetObject(text: string): Detection | null {
    const stable = this.stableDetections();
    let best: Detection | null = null;
    for (const d of stable) {
      if (text.includes(d.label) || text.includes(d.label.split(" ").pop()!)) {
        if (!best || d.box[2] * d.box[3] > best.box[2] * best.box[3]) best = d;
      }
    }
    return best;
  }

  /**
   * The one source of truth for what's in frame: stable detections rendered as
   * a plain natural list with measured colours and counts, e.g. "a blue chair,
   * a person, and a cell phone". This is pure measurement — no interpretation —
   * and it's what both the direct answers and the LLM grounding are built from.
   */
  sceneFacts(): string {
    const byLabel = new Map<string, Detection[]>();
    for (const d of this.stableDetections()) {
      (byLabel.get(d.label) ?? byLabel.set(d.label, []).get(d.label)!).push(d);
    }
    if (byLabel.size === 0) return "";
    const parts = [...byLabel.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 6)
      .map(([label, dets]) => {
        const n = dets.length;
        if (n === 1) {
          const color = dets[0].color;
          const desc = color ? `${color} ${label}` : label;
          return `${/^[aeiou]/.test(color ?? label) ? "an" : "a"} ${desc}`;
        }
        return `${numWord(n)} ${pluralize(label)}`;
      });
    return joinList(parts);
  }

  private updateStability(detections: Detection[]): void {
    this.recentLabels.push([...new Set(detections.map((d) => d.label))]);
    if (this.recentLabels.length > STABILITY_FRAMES) this.recentLabels.shift();
    const counts = new Map<string, number>();
    for (const labels of this.recentLabels) {
      for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    // During warmup (<2 frames) trust the latest frame; otherwise require ≥2.
    const need = this.recentLabels.length < 2 ? 1 : 2;
    this.stableSet = new Set(
      [...counts.entries()].filter(([, c]) => c >= need).map(([l]) => l),
    );
  }

  get active(): boolean {
    return this.running;
  }

  // --- Derived scene state ------------------------------------------------

  get personCount(): number {
    // Only count people once "person" is stable, and don't blink to 0 on a
    // single dropped frame — report at least 1 while it stays stable.
    if (!this.stableSet.has("person")) return 0;
    return Math.max(1, this.latest.filter((d) => d.label === "person").length);
  }

  get personPresent(): boolean {
    return this.personCount > 0;
  }

  get phonePresent(): boolean {
    return this.stableSet.has("cell phone");
  }

  get slouching(): boolean {
    return this.slouchingNow;
  }

  get state(): SceneState {
    return {
      personCount: this.personCount,
      personPresent: this.personPresent,
      phonePresent: this.phonePresent,
      slouching: this.slouching,
    };
  }

  /**
   * A precise factual question we answer directly from measurements (the 270M
   * model deflects on these, and exactness matters): count, colour, or a
   * straight "what do you see". Broader/interpretive visual turns are handled by
   * `referencesVision` + the LLM instead.
   */
  matchesQuestion(text: string): boolean {
    const s = text.toLowerCase();
    return (
      /\bwhat( do| can| are)? you see\b/.test(s) ||
      /\bwhat('?s| is)?\s*(in )?my (background|room|surroundings)\b/.test(s) ||
      /\bwhat('?s| is| are)?\b.*\b(in (view|frame|the (background|room|picture|shot|scene))|behind me|around me|in front of me)\b/.test(
        s,
      ) ||
      /\bhow many (people|persons|faces|chairs|things|objects)\b/.test(s) ||
      /\bwhat colou?r\b/.test(s) ||
      /\b(am i|are we)\b.*\bwearing\b/.test(s) ||
      /\bwhat am i wearing\b/.test(s)
    );
  }

  /**
   * A looser "this turn is about what you can see" test — used to decide when to
   * hand the LLM the grounded scene facts so it can answer open-ended /
   * interpretive visual questions ("what am I doing", "does my room look tidy")
   * itself, rather than us templating a reply.
   */
  referencesVision(text: string): boolean {
    return /\b(see|seeing|look|looking|camera|webcam|frame|view|room|background|surroundings|around me|behind me|in front of me|wearing|holding|doing|on my phone)\b/i.test(
      text,
    );
  }

  /** Answer a precise factual visual question directly from measurements. */
  answer(text: string): string {
    const s = text.toLowerCase();

    // What are you wearing → the person's measured torso colour.
    if (/\bwearing\b|\bshirt\b|\btop\b|\boutfit\b|\bclothes\b/.test(s)) {
      const person = largestPerson(this.stableDetections());
      if (!person) return "I can't see you clearly enough to tell what you're wearing.";
      return person.color
        ? `Looks like you're wearing something ${person.color}.`
        : "I can see you, but can't quite pick out the colour.";
    }

    // Colour of a specific object → its measured colour.
    if (/\bcolou?r\b/.test(s)) {
      const target = this.targetObject(s);
      if (target) {
        return target.color
          ? `Your ${target.label} looks ${target.color}.`
          : `I can see a ${target.label}, but can't pin down its colour.`;
      }
    }

    // Count of people.
    if (/\bhow many\b/.test(s) && /\b(people|person|face|faces)\b/.test(s)) {
      const n = this.personCount;
      return n === 0
        ? "I don't see anyone in frame right now."
        : `I can see ${numWord(n)} ${n === 1 ? "person" : "people"}.`;
    }

    // Otherwise: just render the measured scene.
    const facts = this.sceneFacts();
    if (!facts) return "I can't make out anything specific in the frame right now.";
    return `I can see ${facts}.`;
  }

  // --- Detection loop -----------------------------------------------------

  private loop(): void {
    // Single recursive scheduler: a COMPLETED frame waits the full low-priority
    // interval, but a SKIPPED tick (detector busy / yielding to ASR-TTS, which
    // is most tick moments while a session runs) retries quickly — otherwise
    // every skip burned a whole 2.2 s and the first detection could take 5-10 s
    // to appear after enabling the Eye.
    void this.tick().then((ran) => {
      if (!this.running) return;
      const delay = ran ? FRAME_INTERVAL_MS : SKIP_RETRY_MS;
      this.timer = setTimeout(() => this.loop(), delay);
    });
  }

  /** Returns true when a detector frame actually ran (vs a priority skip). */
  private async tick(): Promise<boolean> {
    if (!this.running || this.detector.isBusy) return false;
    // Yield to the higher-priority audio pipeline: skip this frame while ASR is
    // transcribing or the assistant is speaking. The scheduler retries soon.
    if (this.pauseWhile?.()) return false;
    try {
      const detections = await this.detector.detect(this.video);
      if (!this.running) return true;
      this.sampleColors(detections);
      this.latest = detections;
      this.updateStability(detections);
      this.updatePosture(detections);
      return true;
    } catch {
      // Transient; the next frame retries.
      return true;
    }
  }

  /**
   * Best-effort, explicitly approximate posture proxy. We compare the current
   * largest person's box top and height against a short rolling baseline: when
   * the top drops (person sinks down) AND the box gets shorter, we call it
   * slouching. This is a heuristic, not a real pose estimate.
   */
  private updatePosture(detections: Detection[]): void {
    const person = largestPerson(detections);
    if (!person) {
      this.slouchingNow = false;
      return;
    }
    const top = person.box[1];
    const height = person.box[3];

    if (this.topHistory.length >= POSTURE_BASELINE) {
      const baseTop = avg(this.topHistory);
      const baseHeight = avg(this.heightHistory);
      // Top lower on screen by >8% of frame height and box shorter by >12%.
      this.slouchingNow =
        top > baseTop + baseHeight * 0.08 && height < baseHeight * 0.88;
    } else {
      this.slouchingNow = false;
    }

    this.topHistory.push(top);
    this.heightHistory.push(height);
    if (this.topHistory.length > POSTURE_BASELINE) this.topHistory.shift();
    if (this.heightHistory.length > POSTURE_BASELINE) this.heightHistory.shift();
  }
}

function avg(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

const NUM_WORDS = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
];

function numWord(n: number): string {
  return NUM_WORDS[n] ?? String(n);
}

function pluralize(label: string): string {
  if (label === "person") return "people";
  if (/(s|sh|ch|x|z)$/.test(label)) return label + "es";
  return label + "s";
}

/** Map an average RGB to a rough colour name via HSL bucketing. */
function colorName(r: number, g: number, b: number): string {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2 / 255;
  const delta = (max - min) / 255;
  const sat = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
  let h = 0;
  if (delta !== 0) {
    const dn = delta;
    if (max === r) h = ((g - b) / 255 / dn) % 6;
    else if (max === g) h = (b - r) / 255 / dn + 2;
    else h = (r - g) / 255 / dn + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  if (l < 0.13) return "black";
  if (l > 0.9 && sat < 0.15) return "white";
  if (sat < 0.12) return l < 0.4 ? "dark gray" : l > 0.7 ? "light gray" : "gray";
  // Low-lightness orange reads as brown.
  if (h < 45 && l < 0.4 && sat > 0.2) return "brown";
  const shade = l < 0.32 ? "dark " : l > 0.78 ? "light " : "";
  const hue =
    h < 15 || h >= 345
      ? "red"
      : h < 45
        ? "orange"
        : h < 65
          ? "yellow"
          : h < 170
            ? "green"
            : h < 200
              ? "teal"
              : h < 255
                ? "blue"
                : h < 290
                  ? "purple"
                  : "pink";
  return shade + hue;
}

function joinList(parts: string[]): string {
  if (parts.length === 1) return parts[0];
  if (parts.length === 2) return `${parts[0]} and ${parts[1]}`;
  return `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
}
