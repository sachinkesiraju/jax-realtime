import { numpy as np, random, tree } from "@jax-js/jax";

import { TUNABLES } from "../tunables";
import type { AudioPlayer } from "./audio";
import {
  createFlowLMState,
  createMimiDecodeState,
  type PocketTTS,
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

    for (let step = 0; step < 1000; step++) {
      // Barge-in: cut playback and stop generating the moment we're aborted.
      if (signal?.aborted) {
        player.stop();
        break;
      }

      let stepKey: np.Array;
      [key, stepKey] = random.split(key);
      // Fuse the steady-state flow-LM decode into one dispatch when enabled.
      // Step 0 is the prefill (conditioning embeds concatenated), which stays on
      // the unfused path — the same fuse-decode-only split as the LLM.
      const fuseFlow = TUNABLES.ttsFusedStep && step > 0;
      const {
        latent,
        isEos,
        state: newFlowLMState,
      } = fuseFlow
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
            step === 0 ? embeds.ref : null,
            flowLMState.kvCacheLen, // same as offset
            lsdDecodeSteps,
            temperature,
            noiseClamp,
          );
      flowLMState = newFlowLMState;

      const isEosData = await isEos.data();
      if (isEosData[0] && eosStep === null) {
        eosStep = step;
      }
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
