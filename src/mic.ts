// Microphone capture as raw 16 kHz PCM via an AudioWorklet, so the app can
// run voice-activity detection and incremental Whisper passes on the live
// buffer (hands-free mode, like the HF speech-to-speech Space).

const SAMPLE_RATE = 16_000;

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
  private chunks: Float32Array[] = [];
  private totalSamples = 0;
  private recentLevel = 0;
  private capturing = true;
  // Rolling cap so the always-on duplex buffer can't grow without bound between
  // utterance clears (32 s > the 28 s max utterance window).
  private maxSamples = SAMPLE_RATE * 32;

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
      this.chunks.push(block);
      this.totalSamples += block.length;
      // Drop oldest chunks past the rolling cap.
      while (this.totalSamples > this.maxSamples && this.chunks.length > 1) {
        this.totalSamples -= this.chunks.shift()!.length;
      }
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
    this.chunks = [];
    this.totalSamples = 0;
  }

  /** Discard buffered audio (start of a new utterance). */
  clear(): void {
    this.chunks = [];
    this.totalSamples = 0;
  }

  /**
   * Drop all but the most recent `ms` of buffered PCM (barge-in buffer
   * hygiene — see TUNABLES.bargePreRollMs). Implemented on the same
   * oldest-chunks-first discipline as the rolling `maxSamples` cap above, so
   * the samples()/durationSeconds() contract is untouched: the buffer is
   * still a contiguous most-recent tail of the capture, just a shorter one.
   * Chunk-granular on purpose: worklet blocks are 128 samples (8 ms at
   * 16 kHz), so the kept tail can exceed `ms` by at most one block — we KEEP
   * the extra rather than slicing a chunk, because trimming must never risk
   * cutting into the audio the caller wants (the user's barge words).
   * level() is untouched (recentLevel is a separate smoothed meter, not
   * derived from the buffer).
   */
  trimToLast(ms: number): void {
    const keep = Math.max(0, Math.round((ms / 1000) * SAMPLE_RATE));
    while (
      this.chunks.length > 0 &&
      this.totalSamples - this.chunks[0].length >= keep
    ) {
      this.totalSamples -= this.chunks.shift()!.length;
    }
  }

  /** Copy of all PCM captured since the last clear(), at 16 kHz mono. */
  samples(): Float32Array {
    const out = new Float32Array(this.totalSamples);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  durationSeconds(): number {
    return this.totalSamples / SAMPLE_RATE;
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
    this.chunks = [];
    this.totalSamples = 0;
    this.recentLevel = 0;
  }
}
