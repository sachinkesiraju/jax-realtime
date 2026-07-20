// Audio playback utilities for TTS output.

const SAMPLE_RATE = 24000; // 24kHz sample rate for Mimi codec

export interface AudioPlayer {
  /** Play a chunk of PCM samples (Float32Array in range [-1, 1]). */
  playChunk(samples: Float32Array): Promise<void>;

  /**
   * Stop all currently scheduled audio immediately (barge-in). Disconnects
   * every live source; the context stays open so the player can be reused or
   * closed cleanly afterwards.
   */
  stop(): void;

  /** Wait for all queued audio to finish, then close the audio context. */
  close(): Promise<void>;

  /** Get all played audio as a WAV blob. */
  toWav(): Blob;

  /** Concatenation of every chunk passed to playChunk (Float32 PCM). */
  pcm(): Float32Array;

  /** Get the underlying AudioContext. */
  readonly context: AudioContext;

  /** Analyser tapping the played audio, for visualizations. */
  readonly analyser: AnalyserNode;
}

/**
 * Creates a streaming audio player for playing PCM chunks as they're generated.
 * Each chunk is scheduled to play immediately after the previous one.
 */
/**
 * Converts PCM samples (Float32Array in range [-1, 1]) to a WAV file Blob.
 */
export function samplesToWav(
  samples: Float32Array,
  sampleRate = SAMPLE_RATE,
): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;

  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  // fmt subchunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // subchunk size
  view.setUint16(20, 1, true); // audio format (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data subchunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Write PCM samples as 16-bit integers
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export function createStreamingPlayer(
  { collectPcm = false }: { collectPcm?: boolean } = {},
): AudioPlayer {
  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 1024;
  let nextStartTime = audioCtx.currentTime;
  let lastEndedPromise: Promise<void> = Promise.resolve();
  const chunks: Float32Array[] = [];
  const liveSources = new Set<AudioBufferSourceNode>();
  let stopped = false;

  function concatChunks(): Float32Array {
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return combined;
  }

  return {
    async playChunk(samples: Float32Array) {
      // In collect-only mode we accumulate PCM without ever touching the
      // speakers — used to pre-synthesize backchannels off the audio graph.
      if (collectPcm) {
        chunks.push(samples.slice());
        return;
      }
      if (stopped) return;

      // Resume audio context if suspended, which is common on mobile after idle
      // periods or keyboard use.
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      if (stopped) return;

      const buffer = audioCtx.createBuffer(1, samples.length, SAMPLE_RATE);
      buffer.getChannelData(0).set(samples);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      source.connect(analyserNode);

      // Schedule this chunk right after the previous one
      const startTime = Math.max(nextStartTime, audioCtx.currentTime);
      source.start(startTime);
      nextStartTime = startTime + buffer.duration;

      // Track live sources so a barge-in can cut them all off immediately.
      liveSources.add(source);
      lastEndedPromise = new Promise((resolve) => {
        source.onended = () => {
          liveSources.delete(source);
          resolve();
        };
      });
    },

    stop() {
      stopped = true;
      for (const source of liveSources) {
        // Deliberately keep source.onended attached: stop() fires the "ended"
        // event, which resolves the per-chunk promise a concurrent close() may
        // already be awaiting — nulling it would leave close() hanging forever.
        try {
          source.stop();
        } catch {
          // Already stopped or never started; ignore.
        }
        try {
          source.disconnect();
        } catch {
          // Ignore.
        }
      }
      liveSources.clear();
      nextStartTime = audioCtx.currentTime;
      lastEndedPromise = Promise.resolve();
    },

    async close() {
      await lastEndedPromise;
      await audioCtx.close();
    },

    toWav() {
      return samplesToWav(concatChunks());
    },

    pcm() {
      return concatChunks();
    },

    get context() {
      return audioCtx;
    },

    get analyser() {
      return analyserNode;
    },
  };
}
