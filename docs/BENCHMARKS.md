# Benchmarks & latency hill-climb

## What's actually comparable to TML

Thinking Machines' headline numbers are mostly *quality* benchmarks we cannot
run (Audio MultiChallenge 43.4, FD-bench 77.8, TimeSpeak 64.7%, CueSpeak 81.7%,
RepCount-A 35.4%, ProactiveVideoQA 33.5) — they require their eval harnesses and
grading. The one directly comparable, measurable target is:

- **Turn-taking latency: TML 0.40 s** (end of user speech → start of model
  speech, warm). This is our hill-climb target.

We also track internal stage latencies (not TML-published, but the levers):
ASR pass time, LLM first-token + tok/s, TTS first-audio + realtime factor,
barge-in stop time. And qualitative pass/fail on the interaction behaviors.

## Method (repeatable, fake-mic)

Measured warm (after load warmup), Chrome, Apple-silicon, dual-lane
(ASR wasm / TTS+LLM webgpu). Drive with a fixed fake-mic clip so utterances are
identical across runs. Metrics come from the app's own instrumentation
(`DuplexMetrics`) plus a console harness (`scratchpad/bench.js`) that samples the
dev hooks. Report median of ≥3 turns; note cold-start separately.

Turn latency defined exactly as: `firstAssistantAudioAt - endOfUserSpeechAt`,
both `performance.now()` timestamps inside DuplexSession.

## Results log

Fill in as measured. `–` = not yet measured.

| Build | Turn latency (warm) | ASR lag (end→committed) | TTS first-audio | Notes |
|---|---|---|---|---|
| v1 (finalize on critical path) | ~5–9 s | ~3.6 s | ~0.8 s | full Whisper finalize dominated |
| v2 (skip finalize) | **~2.0 s** (1.37 / 2.20 / 2.48) | 0.45–1.0 s | ~0.8 s | committed streaming text reused; measured 3-turn, Apple silicon, dialogue clip |
| v2.1 (prompt fix + early-clause TTS + tighter endpoint) | ~1.7 s | 0.4–0.9 s | earlier on long replies | endpoint 800→620 ms; first clause flushed on comma |
| v2.2 (LLM warmup at load) | **turn1 1.94 s, steady 1.2–1.4 s** | — | — | measured E2E, 3-turn, dialogue clip: turn1 1.94 / 1.21 / 1.37 s. Warming Gemma at load cut turn-1 cold start from 3.23→1.94 s |
| **TML target** | **0.40 s** | — | — | their published turn-taking |

End-to-end verified (v2.2): ASR transcribes correctly, replies are sensible/
empathetic, TTS audio confirmed playing (analyser level ~0.5–0.7) every turn,
multi-turn stable. Quality is good; remaining nit is occasional boilerplate /
repetition on repeated identical input (a test artifact + 270M variance).

Critical path after skip-finalize = endpoint wait (~0.4–0.6 s) + LLM first-token
(~0.6–0.8 s, Gemma 270M prefill) + TTS first-audio (~0.6–0.8 s). We're ~4–5× off
TML's 0.40 s. The obvious lever — **speculative LLM prefill during the user's
pause** — was built and benched in cycle 2 (below) and *falsified*: on a single
WebGPU device the speculation contends with the streaming ASR, so it doesn't
pre-pay first-token and even regresses. The real ceiling is the single GPU.

### Response quality note
Gemma 270M is the local ceiling. A too-long/negative system prompt was priming
the model to answer every turn with "Okay, I understand" — fixed by a short,
positive prompt (factual Qs like "capital of Japan?" → "Tokyo" now work) and
cooler sampling (temp 0.8→0.45). Open-ended prompts still degrade; the honest
fixes are tool-grounding (weather/Wikipedia return real facts) or the Cerebras
key for a larger Brain.

## Map-reduce campaign — cycle 1 (timing knobs)

Harness: `src/tunables.ts` (runtime knobs, region-grouped) + per-turn stage
marks pushed to `TURN_LOG` (endpoint wait / ASR / LLM-first-token / sentence /
TTS-first-audio). Bench drives a fixed fake-mic clip; medians over K turns.

**Noise calibration** (baseline run twice, n=4, same clip): 1538 vs 1614 ms →
~76 ms band on turn latency. Deltas under ~150 ms are not signal.

**MAP** (n=4 each, 2.2 s clip that ends in a period):

| cond | turn lat | Δ vs base | endpoint | note |
| --- | --- | --- | --- | --- |
| baseline | ~1576 ms | — | ~490 ms | — |
| E1 endpointPunctMs 380→280 | 1419 | −157 | 443 | ~noise |
| E2 endpointSilenceMs 620→480 | 1342 | −234 | 455 | borderline |
| K1 tickMs 150→100 | 1465 | −111 | 451 | noise |
| A1 asrPassIntervalMs 150→90 | 1212 | −364 | 452 | best on MAP |

**HOLDOUT** (unseen clip, K=6, fused A1+E2): **reversed** — fused median
2573 ms (n=2 valid) vs baseline 2094 ms; endpoint wait *rose* to ~630–744 ms.

**Verdict — negative result (kept as a design law):** timing-knob tuning is
not where turn latency lives. The endpoint wait is floored by *committed-text
settle*, not the timer (cutting the punct/silence windows barely moved it), and
it's utterance-dependent (clean-punct clip ~450 ms, run-on clip ~750 ms). The
MAP win for A1 came mostly from the clip ending in a period; it did not
generalize. **No knob change shipped** — defaults retained.

**Diagnosed budget:** endpoint-settle (~450–750 ms) + LLM first-token
(~400–600 ms) + sentence/TTS (~200–400 ms). Cycle 2 must attack these
structurally, not with knobs:
1. **Speculative LLM prefill during the pause** — start Gemma prefill on the
   stable committed prefix before the endpoint fires, so first-token is
   pre-paid. Biggest lever; needs false-start guarding.
2. **Faster commit** — endpoint on a stability signal over tentative text
   rather than waiting for LocalAgreement to promote it to committed.

## Map-reduce campaign — cycle 2 (speculative LLM prefill)

Cycle 1 diagnosed the budget as endpoint-settle + **LLM first-token**, and named
speculative prefill as the biggest structural lever: start Gemma on the stable
committed prefix *during the user's pause*, so first-token is pre-paid by the
time the endpoint fires. Cycle 2 built it (behind `TUNABLES.speculativePrefill`)
and benched it — a producer/consumer `DeltaStream` lets an in-flight speculation
be adopted by `respond()` if the finalized turn text matches, else aborted.

**Instrumentation confirmed it works as designed:** on the MAP clip the
speculation *fired and was adopted on 3 of 4 turns* (~75 %). So the mechanism is
sound; the question was only whether it pays.

**MAP (n=4 each, `dialogue.wav[0:2.6s]`), turn latency = firstAudio − endOfSpeech:**

| cond | turn lat (med) | endpoint→first-delta (med) | note |
| --- | --- | --- | --- |
| baseline (OFF) | **1731 ms** | 435 ms | — |
| speculative ON (run 1) | 2012 ms | 1054 ms | worse |
| speculative ON (run 2) | 2425 ms | 756 ms | 3/4 adopted, still worse |

**Verdict — negative result, and the most important design law of the two
cycles.** Speculative prefill is consistently *slower*, and the tell is
`endpoint→first-delta` rising from 435 ms to 756–1054 ms. It should have
*dropped* toward zero if the head start were real (deltas buffered before the
endpoint). It rose because **ASR and the LLM share the single WebGPU device.**
Running Gemma prefill during the pause time-slices the GPU with the still-running
streaming-ASR passes, so:
1. the speculative generation is GPU-starved — it hasn't produced even its first
   token by the time the endpoint fires (nothing buffered to adopt), and
2. stealing GPU from ASR delays committed-text settle, pushing the *endpoint
   itself* later (endpoint median 749 → 830 ms).

Net: the "overlap" is fictional on a single-GPU pipeline — the two stages
serialize on the device regardless of how we schedule them. Rejected at MAP (no
holdout needed; there was no win to confirm). **Code reverted; default stays
OFF — the whole family is removed, not flag-gated, because it's a proven
dead-end on this architecture.**

**What this means for the ceiling.** Both cycles now point to the same wall:
turn latency on jax-realtime is **serialized by the single WebGPU device**, not
by our policy timers or our scheduling. You cannot make it faster by overlapping
stages (cycle 2) or shaving endpoint windows (cycle 1). The only levers that can
actually move it *reduce total GPU work on the critical path*:
- a smaller / faster LLM, or fewer first-token FLOPs (shorter prompt → cheaper
  prefill; the dominant first-token cost is Gemma prefill);
- fewer tokens before first audio (already flushing on the first clause);
- a faster TTS first-frame.
TML's 0.40 s almost certainly comes from dedicated per-stage compute (separate
accelerators) that a single browser GPU context doesn't offer. Our realistic
floor with a cascade on one WebGPU device is the ~1.3–1.7 s warm turn we already
have; further wins require a cheaper model, not cleverer scheduling.

## Hill-climb levers (ordered by expected payoff)

Critical path after skip-finalize ≈ **LLM first-token + TTS first-audio**
(ASR is now off the path). So:

1. ~~**Speculative LLM prefill during user speech**~~ — *tried and rejected in
   cycle 2.* It fires and adopts ~75 % of the time but regresses turn latency:
   the speculation contends with streaming ASR on the single WebGPU device, so
   first-token isn't pre-paid and the endpoint is pushed later. Removed. The
   takeaway reorders this list: on one GPU you cannot buy latency by overlapping
   stages — only by cutting total GPU work (items 3–4).
2. **TTS first-audio** — Pocket TTS `framesAfterEos`/`lsdDecodeSteps=1` already
   minimal; ensure the first *sentence* is as short as safely possible (the
   sentence splitter already flushes on first terminal punct). Consider emitting
   TTS on the first *clause* (comma) for very long first sentences.
3. **Shorter replies** — cap first-sentence tokens; the SYSTEM_HINT already asks
   for 1–3 short sentences. A tighter "lead with one short sentence" nudge lowers
   time-to-first-audio variance.
4. **LLM prefill speed** — Gemma prefill is the dominant first-token cost; batch
   the prompt prefill (already single-shot) and keep prompts short (trim history
   / summarize old turns) so prefill stays cheap as the session grows.
5. **ASR commit latency** (for caption freshness / earlier turn-end confidence),
   not turn latency: shorten `minPassIntervalMs`, cap the streaming window to the
   last ~8 s so passes stay fast, tune LocalAgreement so committed text stabilizes
   sooner → lets endpointing fire on `ENDPOINT_PUNCT_MS` more often.
6. **Endpointing** — already adaptive (450 ms after terminal punct); the punct
   signal comes from committed ASR, so (5) feeds this.

Non-levers / ceilings: wasm CPU Whisper pass time (~seconds) is fine now that
it's off the critical path; won't chase it. We will not hit 0.40 s with a
cascade, but LLM-first-token + TTS-first-audio in the ~0.8–1.5 s range is a
realistic floor to push toward.
