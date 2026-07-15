import { blockUntilReady, numpy as np, random, tree } from "@jax-js/jax";

import { TUNABLES } from "../tunables";
import type { AudioPlayer } from "./audio";
import {
  createFlowLMState,
  createMimiDecodeState,
  type PocketTTS,
  runFlowLMPrefillFused,
  runFlowLMStep,
  runFlowLMStepFused,
  runMimiDecode,
  runMimiDecodeFused,
} from "./pocket-tts";

export interface PlayTTSOptions {
  framesAfterEos: number;
  seed: number | null;
  lsdDecodeSteps: number;
  temperature: number;
  noiseClamp: number | null;
  /** Abort generation mid-stream (barge-in). Playback is cut immediately. */
  signal: AbortSignal | null;
}

export async function playTTS(
  player: AudioPlayer,
  model: PocketTTS,
  embeds: np.Array,
  {
    framesAfterEos = 0,
    seed = null,
    lsdDecodeSteps = 1,
    temperature = 0.7,
    noiseClamp = null,
    signal = null,
  }: Partial<PlayTTSOptions> = {},
): Promise<void> {
  let lastLatent = model.flowLM.bosEmb.ref.reshape([1, -1]); // [1, 32]
  let audioPromise: Promise<void> = Promise.resolve();

  if (seed === null) seed = Math.floor(Math.random() * 2 ** 32);
  let key = random.key(seed);

  try {
    let flowLMState = createFlowLMState(model.flowLM);
    let mimiState = createMimiDecodeState(model.mimi);
    let eosStep: number | null = null;

    console.log("Starting TTS generation...");
    let lastTimestamp = performance.now();

    for (let step = 0; step < 1000; step++) {
      // Barge-in: cut playback and stop generating the moment we're aborted.
      if (signal?.aborted) {
        player.stop();
        break;
      }

      let stepKey: np.Array;
      [key, stepKey] = random.split(key);
      // Two independently-gated fusion levers, one per step kind:
      //   - step 0 (the prefill: conditioning embeds concatenated onto the
      //     projected BOS latent, empty KV caches) fuses into one dispatch
      //     behind TUNABLES.ttsFusedPrefill — built as the cycle-7 open
      //     lever but OFF by default: measured, its one-time compile at a
      //     NEW sentence length (~+90–160 ms) costs MORE than the unfused
      //     path's residual cold cost (~+15–50 ms; the 6 layer calls share
      //     one jit cache entry), with warm at parity (see the tunable);
      //   - steps > 0 (the steady-state T=1 decode) fuse behind
      //     TUNABLES.ttsFusedStep, the shipped ~22% win.
      const {
        latent,
        isEos,
        state: newFlowLMState,
      } = step === 0
        ? TUNABLES.ttsFusedPrefill
          ? runFlowLMPrefillFused(
              tree.ref(model.flowLM),
              flowLMState,
              stepKey,
              lastLatent.ref,
              embeds.ref,
              flowLMState.kvCacheLen, // same as offset
              lsdDecodeSteps,
              temperature,
              noiseClamp,
            )
          : runFlowLMStep(
              tree.ref(model.flowLM),
              flowLMState,
              stepKey,
              lastLatent.ref,
              embeds.ref,
              flowLMState.kvCacheLen, // same as offset
              lsdDecodeSteps,
              temperature,
              noiseClamp,
            )
        : TUNABLES.ttsFusedStep
          ? runFlowLMStepFused(
              tree.ref(model.flowLM),
              flowLMState,
              stepKey,
              lastLatent.ref,
              flowLMState.kvCacheLen, // same as offset
              lsdDecodeSteps,
              temperature,
              noiseClamp,
            )
          : runFlowLMStep(
              tree.ref(model.flowLM),
              flowLMState,
              stepKey,
              lastLatent.ref,
              null,
              flowLMState.kvCacheLen, // same as offset
              lsdDecodeSteps,
              temperature,
              noiseClamp,
            );
      flowLMState = newFlowLMState;

      const isEosData = await isEos.data();
      if (isEosData[0] && eosStep === null) {
        console.log(`🛑 EOS at step ${step}!`);
        eosStep = step;
      }
      if (eosStep !== null && step >= eosStep + framesAfterEos) {
        console.log(
          `Generation ended at step ${step}, ${framesAfterEos} frames after EOS.`,
        );
        latent.dispose();
        break;
      }

      const prevLatent = lastLatent;
      lastLatent = latent;
      prevLatent.dispose();

      const timestamp = performance.now();
      console.log(
        `Generated step ${step} in ${(timestamp - lastTimestamp).toFixed(1)} ms`,
      );
      lastTimestamp = timestamp;

      const mimiInput = latent.ref
        .mul(model.flowLM.embStd.ref)
        .add(model.flowLM.embMean.ref);

      // The Mimi decode fuses both its first-frame prefill and steady-state
      // decode into one dispatch, so it can gate on the flag alone.
      const [audio, newMimiState] = TUNABLES.ttsFusedStep
        ? runMimiDecodeFused(tree.ref(model.mimi), mimiState, mimiInput)
        : runMimiDecode(tree.ref(model.mimi), mimiState, mimiInput);
      mimiState = newMimiState;

      const lastAudioPromise = audioPromise;
      audioPromise = (async () => {
        const audioPcm = (await np
          .clip(audio.slice(0), -1, 1)
          .astype(np.float32)
          .data()) as Float32Array;
        if (audioPcm.length !== 1920) {
          throw new Error(
            `expected 1920 audio samples, got ${audioPcm.length}`,
          );
        }
        await lastAudioPromise;
        await player.playChunk(audioPcm);
      })();
    }
  } finally {
    lastLatent.dispose();
    tree.dispose([model, embeds]);
    await audioPromise;
  }
}

/**
 * Minimal surface of pipeline.ts's SpeechSynthesizer this file's DEV bench
 * needs. Duck-typed (not imported) to keep tts/ free of a pipeline import
 * cycle; at runtime `window.__pipeline().tts` satisfies it.
 */
type SynthLike = {
  model: PocketTTS;
  tokenizer: { encode(text: string): number[] };
  getVoiceEmbed(voice: string): Promise<np.Array>;
  prepareTextPrompt(text: string): [string, number];
};

/**
 * DEV equivalence + timing gate for the fused flow-LM step-0 prefill
 * (`runFlowLMPrefillFused`, TUNABLES.ttsFusedPrefill), the hard gate every
 * fusion before it passed. For each sentence it synthesizes the SAME text
 * with a FIXED seed four times — prefill-unfused cold, unfused warm, fused
 * cold, fused warm (cold = first encounter of that sentence's prefill length
 * in this page session; only the prefill lever flips, the decode fusion stays
 * at its shipped setting) — and then asserts the codebase's equivalence
 * criterion: identical frame count on both paths (same EOS decision, same
 * audio duration) plus a max-|Δ| comparison of the first audio frame's PCM.
 * The generation loop below mirrors playTTS exactly (same dispatch gating,
 * same EOS bookkeeping, same RNG stream) minus the audio player; `prefillMs`
 * is the step-0 flow-LM wall time (blockUntilReady on latent+isEos).
 *
 * Usage (probe): `const m = await import('/src/tts/inference.ts');
 *   await m.benchPrefillFusedEquivalence(window.__pipeline().tts)`
 */
export async function benchPrefillFusedEquivalence(
  synth: SynthLike,
  sentences: string[] = [
    // Deliberately different token counts (same set as benchTtsPrefill) —
    // each distinct length pays its own one-time trace on both paths.
    "Tell me a little more about the ocean.",
    "Tell me a little more about the weather today.",
    "Tell me a little more about the weather and the tides tomorrow.",
  ],
  {
    voice = "azelma",
    seed = 1234,
    // Which path pays its cold trace first per sentence — flip to check that
    // the first-run path isn't warming shared GPU pipelines for the second.
    order = "unfusedFirst" as "unfusedFirst" | "fusedFirst",
  }: {
    voice?: string;
    seed?: number;
    order?: "unfusedFirst" | "fusedFirst";
  } = {},
): Promise<Record<string, unknown>> {
  const model = synth.model;
  const voiceEmbed = await synth.getVoiceEmbed(voice); // cached — do not dispose

  type RunResult = {
    prefillMs: number;
    totalMs: number;
    frames: number;
    firstAudio: Float32Array;
  };

  const runOnce = async (
    fused: boolean,
    tokens: number[],
    framesAfterEos: number,
  ): Promise<RunResult> => {
    // Flip ONLY the prefill lever; the decode fusion stays at its shipped
    // setting so the A/B isolates the step-0 change.
    const prevFlag = TUNABLES.ttsFusedPrefill;
    TUNABLES.ttsFusedPrefill = fused;
    try {
      // Embed construction mirrors synthOne exactly, so the measured prefill
      // shapes are the real ones: [voiceLen + textLen + 1, 1024].
      const tokensAr = np.array(tokens, { dtype: np.uint32 });
      let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr);
      embeds = np.concatenate([voiceEmbed.ref, embeds]);

      let lastLatent = model.flowLM.bosEmb.ref.reshape([1, -1]);
      let key = random.key(seed);
      let flowLMState = createFlowLMState(model.flowLM);
      let mimiState = createMimiDecodeState(model.mimi);
      let eosStep: number | null = null;
      let frames = 0;
      let firstAudio: Float32Array | null = null;
      let prefillMs = 0;
      const tStart = performance.now();
      for (let step = 0; step < 1000; step++) {
        let stepKey: np.Array;
        [key, stepKey] = random.split(key);
        const t0 = performance.now();
        // Same gating as playTTS.
        const {
          latent,
          isEos,
          state: newFlowLMState,
        } = step === 0
          ? TUNABLES.ttsFusedPrefill
            ? runFlowLMPrefillFused(
                tree.ref(model.flowLM),
                flowLMState,
                stepKey,
                lastLatent.ref,
                embeds.ref,
                flowLMState.kvCacheLen,
              )
            : runFlowLMStep(
                tree.ref(model.flowLM),
                flowLMState,
                stepKey,
                lastLatent.ref,
                embeds.ref,
                flowLMState.kvCacheLen,
              )
          : TUNABLES.ttsFusedStep
            ? runFlowLMStepFused(
                tree.ref(model.flowLM),
                flowLMState,
                stepKey,
                lastLatent.ref,
                flowLMState.kvCacheLen,
              )
            : runFlowLMStep(
                tree.ref(model.flowLM),
                flowLMState,
                stepKey,
                lastLatent.ref,
                null,
                flowLMState.kvCacheLen,
              );
        flowLMState = newFlowLMState;
        if (step === 0) {
          await blockUntilReady([latent.ref, isEos.ref]);
          prefillMs = performance.now() - t0;
        }

        const isEosData = await isEos.data();
        if (isEosData[0] && eosStep === null) eosStep = step;
        if (eosStep !== null && step >= eosStep + framesAfterEos) {
          latent.dispose();
          break;
        }

        const prevLatent = lastLatent;
        lastLatent = latent;
        prevLatent.dispose();

        const mimiInput = latent.ref
          .mul(model.flowLM.embStd.ref)
          .add(model.flowLM.embMean.ref);
        const [audio, newMimiState] = TUNABLES.ttsFusedStep
          ? runMimiDecodeFused(tree.ref(model.mimi), mimiState, mimiInput)
          : runMimiDecode(tree.ref(model.mimi), mimiState, mimiInput);
        mimiState = newMimiState;

        const audioPcm = (await np
          .clip(audio.slice(0), -1, 1)
          .astype(np.float32)
          .data()) as Float32Array;
        frames++;
        firstAudio ??= audioPcm;
      }
      key.dispose();
      lastLatent.dispose();
      embeds.dispose();
      tree.dispose([flowLMState.kvCaches, mimiState]);
      return {
        prefillMs,
        totalMs: performance.now() - tStart,
        frames,
        firstAudio: firstAudio ?? new Float32Array(0),
      };
    } finally {
      TUNABLES.ttsFusedStep = prevFlag;
    }
  };

  const results: Record<string, unknown>[] = [];
  let allPass = true;
  for (const sentence of sentences) {
    const [prepared, framesAfterEos] = synth.prepareTextPrompt(sentence);
    const tokens = synth.tokenizer.encode(prepared);

    let unfusedCold: RunResult, unfusedWarm: RunResult;
    let fusedCold: RunResult, fusedWarm: RunResult;
    if (order === "unfusedFirst") {
      unfusedCold = await runOnce(false, tokens, framesAfterEos);
      unfusedWarm = await runOnce(false, tokens, framesAfterEos);
      fusedCold = await runOnce(true, tokens, framesAfterEos);
      fusedWarm = await runOnce(true, tokens, framesAfterEos);
    } else {
      fusedCold = await runOnce(true, tokens, framesAfterEos);
      fusedWarm = await runOnce(true, tokens, framesAfterEos);
      unfusedCold = await runOnce(false, tokens, framesAfterEos);
      unfusedWarm = await runOnce(false, tokens, framesAfterEos);
    }

    // Equivalence gate: identical frame count both paths (same EOS decision,
    // same audio duration) — the shipped fusions' criterion — plus first-audio
    // content compared sample-by-sample.
    const framesEqual =
      unfusedCold.frames === unfusedWarm.frames &&
      unfusedWarm.frames === fusedCold.frames &&
      fusedCold.frames === fusedWarm.frames;
    let firstAudioMaxAbsDiff = Infinity;
    const a = unfusedWarm.firstAudio;
    const b = fusedWarm.firstAudio;
    if (a.length === b.length) {
      firstAudioMaxAbsDiff = 0;
      for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (d > firstAudioMaxAbsDiff) firstAudioMaxAbsDiff = d;
      }
    }
    // fp16 tolerance: PCM is in [-1, 1]; a couple of fp16 ULPs at unit scale
    // (~1e-3) covers reduction-order noise between differently-fused kernels,
    // same order as the LLM bucketed-prefill gate's recorded 3.6e-5.
    const pass = framesEqual && firstAudioMaxAbsDiff <= 1e-3;
    allPass &&= pass;

    const fmt = (r: RunResult) => ({
      prefillMs: +r.prefillMs.toFixed(1),
      totalMs: +r.totalMs.toFixed(1),
      frames: r.frames,
    });
    results.push({
      sentence,
      textTokens: tokens.length,
      prefillLen: voiceEmbed.shape[0] + tokens.length + 1,
      unfusedCold: fmt(unfusedCold),
      unfusedWarm: fmt(unfusedWarm),
      fusedCold: fmt(fusedCold),
      fusedWarm: fmt(fusedWarm),
      framesEqual,
      firstAudioMaxAbsDiff,
      pass,
    });
  }
  return { seed, allPass, results };
}
