// Microphone capture as raw 16 kHz PCM via an AudioWorklet, so the app can
// run voice-activity detection and incremental Whisper passes on the live
// buffer (hands-free mode, like the HF speech-to-speech Space).

const SAMPLE_RATE = 16_000;
const MAX_CAPTURE_SAMPLES = SAMPLE_RATE * 32;

/**
 * Chunk-friendly PCM queue with a strict sample bound. Shrinking the bound
 * trims through the oldest chunk when necessary, rather than retaining a
 * whole extra worklet block. That exact bound matters for barge-in pre-roll:
 * old reply-period audio must not survive merely because it shares a chunk
 * with the retained tail.
 */
export class BoundedPcmBuffer {
  private chunks: Float32Array[] = [];
  private totalSamples = 0;
  private maxSamples: number;

  constructor(maxSamples: number) {
    this.maxSamples = sampleLimit(maxSamples);
  }

  get length(): number {
    return this.totalSamples;
  }

  append(samples: Float32Array): void {
    if (samples.length === 0) return;
    this.chunks.push(samples);
    this.totalSamples += samples.length;
    this.trimTo(this.maxSamples);
  }

  clear(): void {
    this.chunks = [];
    this.totalSamples = 0;
  }

  /** Change the rolling bound, immediately dropping any excess oldest PCM. */
  setMaxSamples(maxSamples: number): void {
    this.maxSamples = sampleLimit(maxSamples);
    this.trimTo(this.maxSamples);
  }

  /** Copy the retained contiguous PCM tail. */
  samples(): Float32Array {
    const out = new Float32Array(this.totalSamples);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  private trimTo(keepSamples: number): void {
    let drop = this.totalSamples - keepSamples;
    while (drop > 0 && this.chunks.length > 0) {
      const first = this.chunks[0];
      if (drop >= first.length) {
        this.chunks.shift();
        this.totalSamples -= first.length;
        drop -= first.length;
      } else {
        // Copy the partial tail so a tiny pre-roll cannot retain a large source
        // ArrayBuffer through a subarray view.
        this.chunks[0] = first.slice(drop);
        this.totalSamples -= drop;
        drop = 0;
      }
    }
  }
}

function sampleLimit(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("PCM sample limit must be a finite non-negative number");
  }
  return Math.floor(value);
}

export type PcmBufferProbeResult = {
  passed: boolean;
  rollingTail: number[];
  zeroBoundLength: number;
  unboundedTail: number[];
  handoffTail: number[];
};

/**
 * Deterministic, device-free probe for the exact-bound and pre-roll handoff
 * invariants. Exposed as window.__pcmBufferProbe in DEV so it can run without
 * microphone permission, model downloads, timing, or WebAudio scheduling.
 */
export function runPcmBufferProbe(): PcmBufferProbeResult {
  const buffer = new BoundedPcmBuffer(5);
  buffer.append(Float32Array.from([0, 1, 2, 3]));
  buffer.append(Float32Array.from([4, 5, 6, 7]));
  const rollingTail = Array.from(buffer.samples());

  buffer.setMaxSamples(0);
  const zeroBoundLength = buffer.length;

  // Paired model of the reply-to-user handoff. 10/11 are old reply-period
  // markers; 12..17 are the interruption. The old path retains both groups.
  const unbounded = new BoundedPcmBuffer(32);
  unbounded.append(Float32Array.from([10, 11, 12, 13, 14, 15, 16, 17]));
  const unboundedTail = Array.from(unbounded.samples());

  // The bounded path rolls off 10/11 during the reply, then restores the normal
  // cap before appending the rest of the interruption.
  buffer.setMaxSamples(4);
  buffer.append(Float32Array.from([10, 11, 12]));
  buffer.append(Float32Array.from([13, 14, 15]));
  buffer.setMaxSamples(32);
  buffer.append(Float32Array.from([16, 17]));
  const handoffTail = Array.from(buffer.samples());

  const passed =
    rollingTail.join(",") === "3,4,5,6,7" &&
    zeroBoundLength === 0 &&
    unboundedTail.join(",") === "10,11,12,13,14,15,16,17" &&
    handoffTail.join(",") === "12,13,14,15,16,17";
  if (!passed) throw new Error("bounded PCM buffer probe failed");
  return { passed, rollingTail, zeroBoundLength, unboundedTail, handoffTail };
}

const WORKLET_CODE = `
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("pcm-capture", PCMCapture);
`;

export class VoiceCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  // The normal 32 s cap exceeds the 28 s max utterance window. During an
  // assistant reply startPreRoll() temporarily lowers it; handoffPreRoll()
  // restores this cap without clearing the user's interruption.
  private pcm = new BoundedPcmBuffer(MAX_CAPTURE_SAMPLES);
  private recentLevel = 0;
  private capturing = true;

  get open(): boolean {
    return this.context !== null;
  }

  /** Request the microphone and start collecting PCM. */
  async start(): Promise<void> {
    if (this.context) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone capture is not available in this browser");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const context = new AudioContext({ sampleRate: SAMPLE_RATE });
    const workletUrl = URL.createObjectURL(
      new Blob([WORKLET_CODE], { type: "application/javascript" }),
    );
    try {
      await context.audioWorklet.addModule(workletUrl);
    } finally {
      URL.revokeObjectURL(workletUrl);
    }
    const source = context.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(context, "pcm-capture");
    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
      if (!this.capturing) return;
      const block = event.data;
      this.pcm.append(block);
      let sum = 0;
      for (const v of block) sum += v * v;
      const rms = Math.sqrt(sum / block.length);
      // Smoothed over roughly the last ~100 ms of blocks.
      this.recentLevel = this.recentLevel * 0.7 + rms * 0.3;
    };
    source.connect(node);
    // Keep the graph alive without echoing the mic to the speakers.
    const sink = context.createGain();
    sink.gain.value = 0;
    node.connect(sink);
    sink.connect(context.destination);
    this.context = context;
    this.clear();
  }

  /** Discard buffered audio and restore the normal utterance-size cap. */
  clear(): void {
    this.pcm.clear();
    this.pcm.setMaxSamples(MAX_CAPTURE_SAMPLES);
  }

  /**
   * Start a fresh, strictly bounded rolling pre-roll window. Used while the
   * assistant replies: old reply-period capture falls off continuously, so a
   * barge-in never has to trim a large buffer after detection.
   */
  startPreRoll(ms: number): void {
    const maxSamples = sampleLimit((ms / 1000) * SAMPLE_RATE);
    this.pcm.clear();
    this.pcm.setMaxSamples(maxSamples);
  }

  /**
   * Hand the retained pre-roll to a normal growing utterance buffer. Existing
   * PCM is preserved exactly; subsequent interruption audio appends under the
   * normal 32 s cap.
   */
  handoffPreRoll(): void {
    this.pcm.setMaxSamples(MAX_CAPTURE_SAMPLES);
  }

  /** Copy of all currently retained PCM, at 16 kHz mono. */
  samples(): Float32Array {
    return this.pcm.samples();
  }

  durationSeconds(): number {
    return this.pcm.length / SAMPLE_RATE;
  }

  /** Smoothed RMS input level, scaled to roughly [0, 1]. */
  level(): number {
    return Math.min(1, this.recentLevel * 4);
  }

  /**
   * Stop buffering but keep the audio graph running. Suspending the
   * AudioContext instead looks equivalent, but in Chrome the MediaStream
   * source can stay silent after resume — so we just drop samples.
   */
  async pause(): Promise<void> {
    this.capturing = false;
    this.recentLevel = 0;
  }

  async resume(): Promise<void> {
    this.capturing = true;
    // Recover if the context was suspended by the browser (e.g. tab idle).
    if (this.context?.state === "suspended") await this.context.resume();
  }

  /** Release the microphone entirely. */
  async close(): Promise<void> {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.context) {
      await this.context.close().catch(() => {});
      this.context = null;
    }
    this.clear();
    this.recentLevel = 0;
  }
}
