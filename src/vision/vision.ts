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
// While the scene isn't yet established (no stable person), poll faster so a
// person who just sat down registers within ~1 s instead of up to one full
// low-priority interval. Once a person is stable we relax to FRAME_INTERVAL_MS.
const SEARCH_INTERVAL_MS = 700;
// Retry cadence after a tick was skipped for priority reasons — cheap (no GPU
// work happened) and keeps the Eye from starving while audio stays busy.
const SKIP_RETRY_MS = 250;
// Confidence floor for a detection to be reported in the scene description /
// answers (the overlay still draws everything above the detector threshold).
// D-FINE-S (COCO) confidently confuses similar furniture (bed/couch/chair) in
// the 0.55-0.7 band on a webcam; 0.62 trims the worst of it. Genuine confusions
// above this are a small-detector accuracy limit, not something the app can fix.
const SCENE_MIN_SCORE = 0.62;
// Cap the scene description to its most confident objects — a long tail of
// low-ish detections reads as noise even when each clears the floor.
const SCENE_MAX_OBJECTS = 4;
const POSTURE_BASELINE = 6; // rolling frames used for the slouch baseline
// A label must appear in ≥2 of the last STABILITY_FRAMES detector frames to be
// reported — this filters flickering false positives (a "cat" that D-FINE only
// hallucinates for a single frame) out of what we tell the user.
const STABILITY_FRAMES = 3;
// Min score for a person box to COUNT toward the announced number (the overlay
// still draws everything above the detector's own display threshold).
const PERSON_COUNT_MIN_SCORE = 0.6;
// "Tell me about / describe X" phrasing. When X names something the eye can
// currently see, the eye — not a web lookup — is the authority on it, so the
// turn is answered from measurements. Before this, "tell me about the person"
// matched the lookup tool's trigger and web-searched "the person".
const DESCRIBE_RE =
  /\b(tell me (?:more )?about|describe|what about|who(?:'s| is| are))\b/i;

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
  private recentPersonCounts: number[] = [];
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
    this.recentPersonCounts = [];
    this.stableSet = new Set();
  }

  /** Detections in the latest frame limited to temporally-stable labels. */
  private stableDetections(): Detection[] {
    // Confidence floor on top of stability: D-FINE confidently flickers
    // background furniture (a "couch"/"bed" at ~0.5) in a webcam scene, and
    // reporting those as fact reads as hallucination. Real, salient objects
    // (the person, a held phone) sit well above this.
    return this.latest.filter(
      (d) => d.score >= SCENE_MIN_SCORE && this.stableSet.has(d.label),
    );
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
        // Clothing band: LOWER-middle of the box. A seated webcam framing gives
        // a head-and-shoulders box where 40-70% height is still face/neck —
        // sampling there described skin ("orange person", "dark red shirt" on a
        // navy polo). 58-90% height hits the chest/shirt in that framing, and
        // still lands on clothing (pants) for a full-body box.
        x += w * 0.3;
        w *= 0.4;
        y += h * 0.58;
        h *= 0.32;
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

  /** Find the stable detection a colour/appearance/describe question refers to. */
  private targetObject(text: string): Detection | null {
    // Word-boundary matching (not substring) so a "tie" detection can't fire
    // inside "patience"; "people" normalizes to the COCO label "person". COCO
    // labels are plain words, so no regex escaping is needed.
    const s = text.toLowerCase().replace(/\bpeople\b/g, "person");
    const stable = this.stableDetections();
    let best: Detection | null = null;
    for (const d of stable) {
      const last = d.label.split(" ").pop()!;
      const re = new RegExp(`\\b(?:${d.label}|${last})(?:e?s)?\\b`);
      if (re.test(s)) {
        if (!best || d.box[2] * d.box[3] > best.box[2] * best.box[3]) best = d;
      }
    }
    return best;
  }

  /** True when the text names something the eye can currently see. */
  seesSubject(text: string): boolean {
    return this.targetObject(text) !== null;
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
      .slice(0, SCENE_MAX_OBJECTS)
      .map(([label, dets]) => {
        const n = dets.length;
        // Colours are deliberately NOT volunteered here: the cheap box-average
        // sampler is unreliable under webcam lighting (it has called a navy
        // polo "dark red" and a person "orange"), so stating a colour unasked
        // reads as hallucination. A direct "what colour is X" still answers
        // from the measurement in answer(), where it's explicitly requested.
        if (n === 1) return `${/^[aeiou]/.test(label) ? "an" : "a"} ${label}`;
        return `${numWord(n)} ${pluralize(label)}`;
      });
    return joinList(parts);
  }

  private updateStability(detections: Detection[]): void {
    this.recentLabels.push([...new Set(detections.map((d) => d.label))]);
    if (this.recentLabels.length > STABILITY_FRAMES) this.recentLabels.shift();
    // Per-frame person counts feed a median in `personCount`, so one dropped
    // frame (yielded to audio) or one ghost box can't flap the reported number.
    // Counting demands higher confidence than displaying: a tentative 0.5 box
    // is worth drawing on the overlay, but announcing "2 people" is a claim —
    // real people measure 0.74-0.93 here; posters/reflections sit near 0.5.
    this.recentPersonCounts.push(
      detections.filter(
        (d) => d.label === "person" && d.score >= PERSON_COUNT_MIN_SCORE,
      ).length,
    );
    if (this.recentPersonCounts.length > STABILITY_FRAMES) {
      this.recentPersonCounts.shift();
    }
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
    // Only count people once "person" is stable, then report the MEDIAN count
    // over the stability window (never below 1 while stable) — the instant
    // latest-frame count flapped 0→2 on dropped frames and duplicate boxes.
    if (!this.stableSet.has("person")) return 0;
    const sorted = [...this.recentPersonCounts].sort((a, b) => a - b);
    const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 1;
    return Math.max(1, median);
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
      /\bwhat am i wearing\b/.test(s) ||
      // "Tell me about the person / describe the couch" — describing something
      // in frame is the eye's job, not Wikipedia's.
      (DESCRIBE_RE.test(s) && this.targetObject(s) !== null)
    );
  }

  /**
   * A looser "this turn is about what you can see" test — used to decide when to
   * hand the LLM the grounded scene facts so it can answer open-ended /
   * interpretive visual questions ("what am I doing", "does my room look tidy")
   * itself, rather than us templating a reply.
   */
  referencesVision(text: string): boolean {
    if (
      /\b(see|seeing|look|looking|camera|webcam|frame|view|room|background|surroundings|around me|behind me|in front of me|wearing|holding|doing|on my phone)\b/i.test(
        text,
      )
    ) {
      return true;
    }
    // Naming something the eye can currently see ("is the couch big?") makes
    // the turn visual even without a see/look verb.
    return this.targetObject(text) !== null;
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

    // Describe something in frame ("tell me about the person", "describe the
    // couch") — measurements only: count, rough distance, sampled colour.
    if (DESCRIBE_RE.test(s)) {
      const target = this.targetObject(s);
      if (target) return this.describeTarget(target);
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

  /**
   * Measurement-grounded description of one in-frame object: person count, a
   * rough distance from the box's frame fraction, and the sampled colour —
   * hedged, never invented. Left/right is deliberately omitted: the preview is
   * mirrored (style.css scaleX(-1)) while boxes are in raw video coordinates,
   * so a spoken "on the left" would contradict what the user sees on screen.
   * The closing line states the detector's ceiling honestly instead of letting
   * the reply trail off as if more detail were withheld.
   */
  private describeTarget(d: Detection): string {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const frac = vw && vh ? (d.box[2] * d.box[3]) / (vw * vh) : 0;
    const distance =
      frac > 0.3 ? "close to the camera" : frac > 0.08 ? "a bit further back" : "off in the background";
    const ceiling = "That's about all the detail my eye picks out.";
    if (d.label === "person") {
      const n = this.personCount;
      const dress = d.color ? `, wearing something ${d.color}` : "";
      if (n > 1) {
        return `I can see ${numWord(n)} people; the nearest is ${distance}${dress}. ${ceiling}`;
      }
      return `I can see one person ${distance}${dress}. ${ceiling}`;
    }
    const colour = d.color ? ` — it looks ${d.color}` : "";
    const article = /^[aeiou]/.test(d.label) ? "an" : "a";
    return `I can see ${article} ${d.label} ${distance}${colour}. ${ceiling}`;
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
      // Completed frame: fast search cadence until a person is established,
      // then relax to the low-priority interval. Skipped frame: retry soon.
      const delay = ran
        ? this.personCount > 0
          ? FRAME_INTERVAL_MS
          : SEARCH_INTERVAL_MS
        : SKIP_RETRY_MS;
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
      // Dedupe at the source: DETR-family models emit a duplicate overlapping
      // box for the same object near the score threshold (e.g. one person as a
      // 0.93 box plus a ~0.5 ghost), which inflated counts ("2 people") and
      // drew double boxes on the overlay.
      const detections = dedupeDetections(await this.detector.detect(this.video));
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

/**
 * Overlap of two [x, y, w, h] boxes, normalized by the SMALLER box's area
 * (intersection-over-minimum). This catches both classic duplicates (high IoU)
 * and contained ghosts — e.g. a face-only "person" box inside the full-body
 * box, whose IoU is small because the areas differ hugely.
 */
function overlapOverMin(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const minArea = Math.min(a[2] * a[3], b[2] * b[3]);
  return minArea > 0 ? inter / minArea : 0;
}

/**
 * Drop same-label detections mostly covered by a stronger one — duplicates or
 * parts of a single object, not a second object. Distinct objects of the same
 * class (two chairs side by side) overlap little and are kept.
 */
function dedupeDetections(detections: Detection[]): Detection[] {
  const kept: Detection[] = [];
  for (const d of [...detections].sort((a, b) => b.score - a.score)) {
    const dup = kept.some(
      (k) => k.label === d.label && overlapOverMin(k.box, d.box) > 0.7,
    );
    if (!dup) kept.push(d);
  }
  return kept;
}
