# Benchmarks & latency hill-climb

> **Historical note.** This file is an append-only campaign log. Cycles 1–5
> and the original weights-reduction work were measured on the earlier
> **Gemma 3 270M** brain, which has since been replaced by **SmolLM2-360M**
> (the Gemma code path is deleted from the tree). Gemma mentions below are
> preserved as the record of those measurements — every *shipped default*
> described here applies to the SmolLM2 pipeline, and cycles 6+ plus the
> branch-vs-main head-to-head were measured on it directly.

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

## Map-reduce campaign — cycle 3 (perceived latency + endpointing)

After two negative cycles established that turn latency is serialized by the
single WebGPU device, cycle 3 split into two campaigns aimed at the two things
that *can* still move: **hide** the gap we can't lower (A), and shave the one fat
budget that isn't GPU-bound — the endpoint wait (B). New instrumentation:
`onsetAudio`/`onsetDurMs` on `TurnRecord` → time-to-first-sound and the
filler→reply gap; a synthetic mid-pause clip (0.5 s inserted silence) for the
false-cut cost guard.

### Campaign A — onset fillers (SHIPPED, then REVERTED)
Pre-render short lead-ins ("So,"/"Right,"/"Okay,") to PCM at load (zero runtime
GPU, like the backchannels) and play one instantly at endpoint while the real
reply generates behind it.

**MAP (4 turns each, `dialogue.wav[0:2.6s]`), time-to-first-sound = first audible
sound − endOfSpeech:**

| cond | first-sound (med) | real reply (turnLat) | note |
| --- | --- | --- | --- |
| baseline (off) | ~1.3–1.8 s | ~1.3–1.8 s | silence until the reply |
| onset "ack" | **~0.46 s** | ~1.3–1.8 s | reply unchanged |
| onset "think" | ~0.45 s | ~1.7–2.5 s | same first-sound; phrase is a quality choice, not latency |

**HOLDOUT (`dialogue.wav[2.6:5.2s]`, unseen):** first-sound ~0.45–0.75 s on the
*metric* — the timing win held. **But the metric lied about the experience.**

**Reverted after listening.** The bench measured time-to-first-sound and the
filler→reply gap, and both looked good — but neither captures the actual defect:
the onset (a separate short-lived AudioContext) and the real reply (the TTS
player) are two independent audio streams with no shared clock, so the reply
begins *over/into* the tail of the onset and the hand-off `stop()` clips the
filler mid-word. It sounds broken — a stutter, not a smooth "So, … answer." The
two "fixes" (hand-off cut, silence-trim + 0.6 s cap) reduced the numeric overlap
but not the audible collision. Removed entirely.

**Lesson (recorded as a law):** perceived-audio quality is not capturable by the
deterministic timing proxies we had. A first-sound metric rewards *any* sound,
including one that steps on the real reply. Masking the gap needs the filler and
the reply to be **one gapless stream on one clock** (append the onset as the
first TTS sentence, or gate the reply until the filler's buffer drains) — not two
overlapping contexts. Until that's built, no onset ships. Do not re-attempt with
a two-context design.

### Campaign B — earlier endpointing (REJECTED)
Two candidate modes vs the committed-text baseline: `tentativePunct` (fire on
terminal punct in the *tentative* tail) and `tentativeStable` (fire when the full
hypothesis is unchanged for 2 ticks + `endpointPunctMs` silence).

| cond | endpoint (med) | false-cuts (pause clip, 3×) |
| --- | --- | --- |
| committed (baseline) | ~452–460 ms | 0 |
| tentativePunct | ~456 ms | — |
| tentativeStable | ~451–453 ms | 0 |

**Verdict — no candidate beat baseline.** On clean utterances the committed
path already fires at ~450 ms (the ASR commits the period fast enough), so the
tentative modes have no headroom — endpoint moved <10 ms, deep inside noise.
tentativeStable didn't false-cut on the 0.5 s-pause clip, but it also bought
nothing, and pushing the silence floor down only trades UX safety for a
non-existent gain. This re-confirms cycle 1's law: **endpoint is settle-bound and
already near its safe floor.** Code removed (net diff is docs only for B); the
negative result is recorded here so the family isn't retried.

**FUSE:** A and B touch disjoint regions, but B has no winner — fusion is just A.

**Net cycle 3: no shippable win.** Onset masking looked like a ~1.3 s perceived
win on the timing proxy but sounded broken on ears (two un-synced audio streams
colliding) and was reverted; the endpoint modes bought nothing. The lasting
output is the law above — perceived-audio quality needs an ears-in-the-loop gate,
and gap-masking must be a single-stream design — plus confirmation (again) that
the GPU floor is the real ceiling.

## Map-reduce campaign — cycle 4 (jax-js compute: fused decode + GPU sampling)

The first campaign aimed at the GPU floor itself. DIAGNOSE (in-app
`benchDecode`) measured the 46.7 ms/token decode budget: readback transfer only
0.7 ms, JS scan 1.2 ms, **~4.4 ms/token of synchronous CPU dispatch across the
~21 separate jit calls** per step, ASR pass cost flat vs window length (kills
the window-cap idea). Verified from jax-js source: number args do NOT re-trace
(trace cache keys on avals), so the dispatch cost is genuinely ~21
command-buffer submits.

Implementation (Opus subagent, audited line-by-line): single-jit fused decode
step (`runGemmaDecodeStepFused`, token/position as np.Array inputs so the trace
caches across tokens), GPU `lax.topK(64)` sampler with bit-identical selection,
KV-cache reuse across turns, history windowing — all behind TUNABLES. The ASR
GPU sampler was **declined honestly**: the timestamp gate needs full-vocab
reductions whose exactness a topK candidate list can't guarantee.

**Equivalence gate (hard):** 4-way greedy 32-token identity across
{fused}×{sampler} — **all identical**. Zero quality risk by construction.

**MAP (n=32 ×2 runs each):**

| config | ms/token | Δ | dispatch-sync |
| --- | --- | --- | --- |
| base (js, unfused) | 46.9 / 40.6 | — | 4.3–4.6 ms |
| fused | 39.1 / 33.3 | −17 % | 2.6–2.9 ms |
| topk | 35.4 / 36.8 | −17 % | 3.7 ms |
| **fused+topk (shipped)** | **34.7 / 33.6** | **−22 %** | 2.4 ms |

**Shipped:** `llmFusedStep: true`, `llmSampler: "topk"` → ~22 → ~29 tok/s.
KV reuse + windowing stay off by default (correct but unproven payoff for short
demo sessions; available as tunables). Honest residual: ~34 ms/token floor is
the per-token GPU submit→execute→map roundtrip — decode must sync every token
to sample, and that roundtrip is the next (hard) wall.

## Weights-download reduction (shipped)

Measured reality: initial download was **846 MB** (Gemma 536 + TTS 236 +
Whisper 74), of which ONE tensor — Gemma's tied embedding table
(262,144×640 fp16 = 335 MB) — is 40 %. CDN serves uncompressed (confirmed) and
fp16 is high-entropy, so transit compression is dead. jax-js has no int8
compute dtype, but @jax-js/loaders parses I8 — so: **quantize the embedding to
per-row int8 offline, dequantize to fp16 at load**. Compute unchanged.

Quality gates:
1. Reconstruction: worst-row cosine 0.9986, median 0.99991, max-abs-err 0.0037.
2. Greedy equivalence vs fp16: identical for the first 31 tokens, then ONE
   near-tie flip ("adapting"→"influenced") forked the sequence — both branches
   fully coherent. (The pre-registered "≥95 % positional agreement" gate read
   67 %, which is the wrong metric after a greedy fork; recorded as a
   gate-design lesson, not a quality failure.)
3. Live E2E on the real mic: correct transcript, coherent reply, turn ~1.3 s.

Also shipped: **parallel weight fetches** (they were strictly sequential), so
wall-clock ≈ the largest file instead of the sum. A deferred-TTS variant
(ready after ASR+LLM, voice downloads in background) was built and then
removed at the owner's call — one "ready" that means fully ready is simpler
than a partial state whose first reply may stall on a download.

| | before | after |
| --- | --- | --- |
| total download | 846 MB | **610 MB** |
| fetch schedule | sequential | parallel |

The quantized file is served from `public/weights/` (gitignored); fresh clones
fall back to the fp16 HF file automatically.

*(Historical — these totals are for the Gemma-era pipeline. The shipped
pipeline today is SmolLM2-360M int8 363 + Pocket TTS 236 + Whisper base.en
144 + D-FINE 42 ≈ **790 MB**, still fetched in parallel and OPFS-cached; the
same per-row int8 scheme carried over to the SmolLM weights.)*

## Map-reduce campaign — cycle 5 (GPT-Live-inspired: patience + delegation)

Source: OpenAI's GPT-Live launch (July 2026). Its two transferable ideas for a
cascade: turn-taking *patience* ("waits while you think" — their stated fix for
silence-based endpointing) and *delegation* (decouple the interaction model
from intelligence; hand hard questions to a background brain).

### Campaign 1 — patience endpointing (ALL CANDIDATES REJECTED, law recorded)

Predicted: baseline false-cuts ~100% on mid-clause pauses ≥0.7 s; candidates
→ ~0. Baseline confirmed exactly (3/3 plays truncated at the pause). But the
candidates went 0-for-5:

- P1(static)/P2(cue)/P3(nonterminal)/P4(tentative-activity): **inert.** The
  diagnose showed every false-cut fires via the PUNCT path — at a mid-clause
  pause, LocalAgreement's committed text ends at the last complete sentence
  (terminal punct), so committed-text gating never engages.
- P5(hypothesis — hold both windows while committed+tentative ends
  non-terminal): **rejected, 3/3 false-cuts, identical to baseline.** The
  mechanism is absent at decision time: streaming-ASR passes lag the audio by
  more than the 380 ms punct window, so the tentative "unfinished" tail does
  not exist yet when the endpoint decides.

**Law: in a lagging cascade ASR there is NO reliable pre-fire continuation
signal.** Patience cannot be bought before the endpoint fires without slowing
every turn (the forbidden cycle-1/3 trade). The design that fits the physics is
post-fire **continuation-merge**: fire the turn, and if speech resumes during
the ~1.2 s before first reply audio, silently abort the response and re-open
the utterance (append, don't restart). Specced for a future cycle; not built.

Bench side-finds fixed on the way: the phantom-turn guard required ≥2 voiced
ticks, but the level meter decays between words so real 2 s utterances measured
1-2 ticks — a coin flip that silently discarded genuine turns (relaxed to ≥1
tick + the peak test, which is the reliable half); the Eye must be disabled
during benches (it spotted a phone mid-run and talked about it).

### Campaign 2 — delegation deepening (SHIPPED: toolRouting "broad")

Diagnose on a 12-question factual QA set + 6 small-talk controls: conservative
routing fired on only 3/12; routing misses (5) and missing tools (4) dominated,
as predicted. Gemma-270M alone is fine on encyclopedic one-liners but
catastrophically wrong on numbers (17×23=411; 15% of 80=1200; 26 mi=26,000 ft;
today="July 21 2024").

Candidates shipped as "broad": D1 wh-question → Wikipedia lookup routing,
D2 weather-query cleanup ("in Paris right now" → "Paris"), D3 instant offline
tools (calculator, unit conversion, clock/date; no holding line — the result is
immediate).

| | conservative (shipped before) | broad |
| --- | --- | --- |
| MAP QA accuracy (12 q) | 5/12 (42 %) | **10/12 (83 %)** |
| HOLDOUT (8 unseen q) | ~5-6/8 | **7/8 (88 %) — no reversal** |
| small-talk false triggers | 0/6, 0/4 | **0/6, 0/4** |

Residuals (recorded, not shipped): phrase queries that aren't Wikipedia titles
("tallest mountain in the world") miss honestly; a title-search fallback was
tried and REJECTED — it converted honest misses into confidently irrelevant
answers ("how far is the moon" → a 2007 film). Summary-lacks-the-number
(population questions) needs snippet retrieval, not title summaries.

Prediction audit: C1 predicted a win and delivered a rejection (the pre-fire
signal doesn't exist — worth knowing); C2 predicted 55-75% and delivered 83/88%
with the 0-false-trigger guard intact.

## Audit cycle (post-cycle-5): bug audit + topK-in-jit

Opus audit over the recent-change surface, Fable-validated line-by-line, all
claims re-benched in-browser.

**Bugs fixed:** detector-preload rejection stayed cached (Eye could never
recover from one failed load — cleared on rejection now); TURN_LOG grew without
bound (ring-capped at 500); calc/convert/clock chips mislabeled "web_search";
[validator] the convert regex table's ungrouped alternations (`/\bmeters?|m\b/`
matches "war**m**") — replaced wholesale, see below.

**Design ruling (owner call): the unit-conversion tool is DELETED.** It went
pairwise-regex-table → generic lexicon in one day, but the owner's challenge
held up: conversion factors are stored knowledge — a hand-maintained
mini-almanac with no principled stopping point (currencies? time zones?), and
the query class was on the QA set because the eval author put it there
(eval-driven feature invention — a campaign-2 methodology flaw, recorded).
The line now drawn: instant tools must be CAPABILITIES (calculator = pure
arithmetic, clock = device state), never knowledge. Conversion queries fall
through to Gemma, which will sometimes be wrong — an honest demo of why
delegation matters rather than a curated illusion. QA scores restated:
MAP 9/12, holdout 6/8 (conversion items ceded).

**Perf:** the brief's Whisper-fusion premise was STALE — the decoder step is
already a single fused jit (measured 9.7 ms/token; ASR pass cost is encoder +
overhead, not decode). The real win: `lax.topK` folded INTO the fused Gemma
jit, emitting a packed [values..64, indices..64] fp32 array — one dispatch and
ONE readback per token instead of step+topK dispatches and two `.data()`
round-trips (indices < 2^24 are exact in fp32). **Greedy-equivalence passed
across all five sampler/fusion configs**; paired speed runs 42.7→38.0 and
45.1→44.0 ms/token (small, direction-consistent, zero-risk). Shipped on
(`llmTopkInFused`).

**Reported, not fixed (recorded):** calc handles one binary op per utterance;
CLOCK_RE misses some phrasings (expanding risks small-talk false positives);
KV-reuse mid-loop-throw leaves inconsistent cache state (feature is off by
default — must be fixed before ever enabling it).

## TTS decode fusion (shipped)

Pocket TTS generation was the pipeline bottleneck (~1.1x realtime under GPU
contention -> longer replies stuttered). Same lever as the Gemma fusion: the
per-frame decode dispatched ~11 separate jits (+ ~25 eager ops); an Opus pass
fused it to 2 (flow-LM step -> 1, Mimi decode -> 1) via verbatim inline copies
(no nested-jit boundaries), per-frame dynamics as np.Array trace inputs. Bench
(fixed seed): gen 1291->1010 ms (~22%), realtime factor 0.40->0.31, IDENTICAL
frame count both paths (same EOS decision, same 3200 ms audio) = equivalent
output. E2E verified (62 audio chunks, coherent reply). Shipped on
(ttsFusedStep). Prefill step 0 stays unfused (fuse-decode-only, like Gemma).

## Map-reduce campaign — cycle 6 (turn-latency: prefill re-trace + onset redo)

Target: the user-felt "delay in speaking after the message is sent" —
`firstAudio − endOfSpeech` — which regressed with the SmolLM2-360M brain swap.
New infra: a repeatable Node harness (`bench/run.mjs`, puppeteer-core driving
real Chrome with `--use-fake-device-for-media-stream` +
`--use-file-for-fake-audio-capture`; `--no-sandbox` is required or the fake
device silently reads nothing) that loads the app, disables the Eye, applies
per-condition TUNABLES overrides, and samples `window.__turnLog`. Clips are
`say`-synthesized WAVs with 14 s silence pads (one loop = one turn) plus the
adversarial noise clips described below — local artifacts under
`bench/clips/` (gitignored), trivially re-made with `say`/`ffmpeg`;
`bench/probe.mjs` runs console-level micro-benches.

**DIAGNOSE (measured, not assumed):** stage medians put the fat in
`llmFirst` (endpoint→first LLM delta): 639–1208 ms and *growing per turn*,
vs endpoint ~450–600 (settle floor, cycle-1 law), sentence ~250–330, TTS
~90–380 ms. `benchPrefill(250)` found the smoking gun: **334 ms warm vs
1004 ms on first encounter of a length** — jax-js trace caches key on shapes,
and every turn has a new prompt length, so all 32 SmolLM prefill layer jits
re-trace + recompile EVERY turn. The prefill cost was compile time, not GPU
math (250-token × 360M prefill is ~50 ms of FLOPs).

**MAP** (n=6 turns/cond, warm medians of turns 2–6, same clip, paired
overrides `llmMaxHistoryTurns:4, llmMaxNewTokens:48` to bound history-growth
noise; baseline noise band ±150 ms):

| cond | turnLat | llmFirst | onset (first sound) |
| --- | --- | --- | --- |
| baseline ×2 | 1815 / 1965 | 674 / 1002 | — |
| **kv-reuse (SmolLM port)** | **5810 — REJECTED** | 4786 (+16 s re-trace spike) | — |
| **bucket64 prefill** | **1366** | **347, flat per-turn** | — |
| onset filler (same-clock) | 1707 (≈noise) | 495 | **659** |

- **KV reuse rejected, family removed** (the cycle-2 rule: proven dead-ends
  are deleted, not flag-gated). The port was mechanically correct
  (prefix-match + throw-safe commit), but its suffix feed pays one fused-step
  GPU sync roundtrip per token — a reply-sized suffix (30–100 tokens) costs
  seconds, and `ensureSmolLmStateCapacity` growth re-traced the fused jit ON
  the critical path (the 16 s spike). **Law: on jax-js, token-by-token
  feeding can never beat a batched prefill for suffixes longer than ~5
  tokens; KV reuse pays only with a batched offset-prefill (prefix-aware
  attention mask), which doesn't exist yet.**
- **bucket64 shipped** (`llmPrefillBucket: 64`): pad the prompt to the next
  64-token multiple (pad id = eos) so prefill trace shapes repeat. Exact by
  construction — pads sit at the end, logits read at the last REAL token,
  pad KV slots are overwritten before ever becoming attendable — and
  equivalence-gated on-device: argmax identical, max |Δ| 3.6e-5
  (`benchPrefillEquivalence`).
- **Onset filler** (cycle-3 Campaign A redo, this time obeying the law): the
  pre-rendered "So,"/"Right,"/"Okay, so" PCM is scheduled as the first chunk
  of the SAME streaming player the reply uses — one stream, one clock;
  overlap is structurally impossible, worst case is dead air. First sound at
  ~510–805 ms with real-reply latency unchanged. `onsetFiller` stays
  **default off**: the cycle-3 law demands an ears gate, and only the bench
  half has passed. Flip it on and listen before shipping it.

**FUSE + HOLDOUT** (bucket64 + onset, unseen clip, n=6):

| cond | turnLat | llmFirst | onset |
| --- | --- | --- | --- |
| baseline (holdout clip) | 1713 | 817 (growing, max 1231) | — |
| **fused (holdout clip)** | **1191 (−30 %)** | **250, flat** | 510 |

No reversal. `llmFirst` stops growing with session length entirely — the
per-turn re-trace is gone (each new 64-bucket boundary pays one warm-up
re-trace, then every turn in that bucket is warm).

Residual (recorded): the shipped-defaults confirmation run (no bench caps)
was blocked by a post-sleep CoreAudio wedge on the bench machine
(`AudioQueueStart failed -66681` — even `afplay` fails; `sudo killall
coreaudiod` fixes it). Holdout numbers stand; re-run
`node bench/run.mjs --clip bench/clips/map_a.wav --turns 6 --label defaults`
after the audio daemon restart for the record.

## Map-reduce campaign — cycle 7 (three-front: perf residual, memory, conversation quality)

Three parallel diagnoses, one candidate family per front, everything gated by
paired evals. New infra: `bench/quality.mjs` (scripted chat histories driven
straight through `__pipeline().llm.generate` — no audio path — scored by
deterministic checks: asksClarify on garbled input, noFalseClarify on clean,
notVerbatim on repeats, shortSpoken on open-ended, noMarkdown/noPlaceholder
everywhere), `bench/memory.mjs` (JS heap via CDP + per-process RSS as a GPU
proxy), and `bench/launch.mjs` (shared launcher: zombie sweep, 1 MB HTTP disk
cache so the OPFS model store isn't duplicated, interrupt-safe teardown).

### Perf — ttsPrefillBucket (REJECTED at MAP; law recorded)

Diagnosis was correct: the flow-LM's step-0 prefill (always unfused) re-traces
its 6 jitted layers per NEW sentence token length — `benchTtsPrefill` measured
~320–550 ms first-encounter vs ~30–60 ms warm, the source of the 90–380 ms
tts-stage variance. But the fix that worked for the LLM (bucket-pad to
16-token multiples, leading spaces through the real tokenizer) made even WARM
prefills slower (~150–165 ms) and regressed the turn bench's tts median
115 → 221 ms. **Law: padding pays only when the padded shape's warm cost is
unchanged — extra flow-LM prefill tokens are not free the way extra SmolLM
prefill tokens are.** Removed per the dead-end rule; the open lever for the
variance is fusing the step-0 prefill, not padding it. (`benchTtsPrefill`
stays as the diagnostic.)

### Memory (SHIPPED: hygiene + lazy Eye)

- Loaders now release the downloaded weight buffers + safetensors views right
  after the GPU copies exist (verified eager: `np.array` copies at call time),
  so the `Promise.all` load can't pin up to 724 MB of JS heap next to the GPU
  weights. Steady-state heap was already fine (~30 MB); this trims the peak.
- D-FINE is now lazy: the eye toggle drives the existing on-demand load path
  instead of an unconditional preload, so an unchecked Eye (or denied camera)
  never downloads (42 MB) or holds GPU residency; "ready" no longer waits on
  the detector warmup. Verified: eye-off run fetches no D-FINE, eye-on session
  loads and detects normally.
- Measurement honesty: macOS RSS of Chrome's GPU process proved too noisy
  across runs (±100 MB) to headline a number; the deterministic claims above
  are what shipped. The fp16 GPU weight residency (~1.1 GB across the three
  models) is the floor until jax-js grows an int8 compute path.

### Conversation quality (SHIPPED: format-token ban + garble clause/exemplar)

Baseline (n=2 runs × 14 items): asksClarify **0/6** — the model confabulates
on every garbled input; shortSpoken 2/6; format axes clean on the eval set
(the live "[activity]" failures are conversation-pressure artifacts).

| cond (MAP) | asksClarify | noFalseClarify | shortSpoken | correct |
| --- | --- | --- | --- | --- |
| baseline ×2 | 0/6 | 6/6 | 2/6 | 5–6/6 |
| ban-format-tokens | 0/6 | 6/6 | 3/6 | 6/6 |
| garble clause alone | **0/6 — inert** | 6/6 | 2/6 | 5/6 |
| garble clause + 1 exemplar | **5/6** | 6/6 | 4/6 | 5/6 |
| temperature 0.5 | 0/6 | 6/6 | **4/6** | 6/6 |

- **Ban (shipped on):** 239 of 49k vocab ids whose decoded text contains
  `[ ] * # \`` get -Infinity at sampling; none contain a letter, so no word is
  affected — the prompt's "no markdown" request becomes a guarantee. Zero
  regressions anywhere. Residual: numbered lists ("1. …") are digits and
  can't be token-banned (seen ~1/40 replies).
- **Garble clause (shipped on, WITH exemplar):** the instruction alone is
  inert at 360M; one demonstrated clarify exchange in the prompt is what
  teaches it (docs/CONVERSATION.md's Tier-2 design, confirmed). Iteration 2:
  scene-tagged turns skip the exemplar (it leaked a clarify onto "What am I
  sitting on?"). Final confirmation (n=3 runs): asksClarify 0 → **6/9 MAP,
  3/6 holdout**, noFalseClarify 8/9 / 6/6, other axes flat. Cost ~35 prompt
  tokens/turn. Known mild residual: ~1-in-9 clean turns gets a clarify —
  the recovery (user repeats) is graceful vs the confabulation it replaced.
- **Temperature 0.5 (NOT shipped, law recorded):** wins brevity alone, but
  FUSED with the exemplar it reversed on holdout (asksClarify 3/4 → 1/4,
  factual misses) — cooler sampling fights few-shot imitation. Stays 0.7.

Fusion lesson (again): the shippable fusion was ban+garble only — the
three-way fusion's holdout reversal is exactly why the HOLDOUT gate exists.

## Parallel-VAD investigation (NOT ADOPTED — baseline has no headroom)

Question: would a parallel VAD (frame-level speech-probability model) beside
the micro-turn engine improve detection accuracy, serve as a fallback, or
simplify the energy-heuristic stack without adding latency?

**Method:** adversarial fake-mic bench (`bench/observe.mjs` + new clips) that
plays hostile audio and counts what the engine DID — turns fired (the
user-visible failure), phantom discards, backchannels:

| clip | want | baseline result |
| --- | --- | --- |
| `noise_typing.wav` (60 s of irregular 15 ms clicks) | 0 turns | **0 turns** — all 16 latched "utterances" discarded by the phantom-turn guard; blemish: 2 backchannels hummed at the keyboard |
| `noise_ambient.wav` (60 s pink noise) | 0 turns | **0 turns** (1 discard) |
| `quiet_speech.wav` (speech at 0.15× amplitude) | 1 turn/loop | **3/3 turns, transcripts correct** (mic AGC + the guard's adaptive ambient floor absorb the gain) |

**Verdict: no VAD.** The failure buckets a VAD targets (keyboard transients,
ambient swells, quiet-speaker misses) are EMPTY at baseline — the
voiced-run/peak guard plus the downstream transcript gates already reject
noise at 100 % on this suite while passing 0.15× speech. A second detector
would add tuning surface, not accuracy. Integration reality reinforces it:
Silero VAD needs LSTM ops `@jax-js/onnx` doesn't implement; `vad-web` brings
onnxruntime-web (~24 MB) and wants to own the mic; and the GPU law forbids
putting it on the WebGPU device. Recorded so the idea isn't retried without
new evidence of a failing bucket (a real-world garble corpus would be that
evidence).

The one measured leak WAS fixed classically: the backchannel fired on typing
noise because it ran before the endpoint-time guard — it now requires the
same voiced-evidence test (`utteranceSoundsVoiced`, duplex.ts) before humming.
Confirmed post-fix: 60 s typing clip → 0 turns, **0 backchannels** (was 2),
19 phantom discards (non-vacuous — the mic heard every click train), and a
4-turn speech run stayed healthy (turnLat 1195–1535 ms, endpoint ~449 ms,
zero behavioral change on real speech).

Bench-host note (recurring): the Chrome fake-mic goes silent whenever this
Mac sleeps — CoreAudio wedges (`AudioQueueStart -66681`, even `afplay`
fails) and `sudo killall coreaudiod` revives it. A wedged run is detectable
by zero turns AND zero discards on a clip that should produce either.

## Branch-vs-main head-to-head (pre-merge verification)

Shipped-vs-shipped comparison of this branch against `main` (Gemma-era,
2987a8b), run back-to-back on the same machine, same clips, same eval items,
same tunables caps; main served from a worktree on its own port with its own
OPFS cache.

**Turn latency** (map_a, 6 turns, warm medians):

| | main (Gemma 270M) | branch (SmolLM2-360M) |
| --- | --- | --- |
| turn latency | 1566 ms | **1269 ms (−19 %)** |
| LLM first-token | 566 ms | **323 ms** |
| TTS first-audio | 216 ms | 144 ms |

A **35 % larger model answering 19 % faster** — the bucketed-prefill work more
than pays for the brain upgrade.

**Conversation quality** (deterministic axes, n=3 runs, MAP + holdout):

| axis | main MAP / holdout | branch MAP / holdout |
| --- | --- | --- |
| noMarkdown | 28/42 · **14/27** | **42/42 · 27/27** (token ban) |
| asksClarify (garbled) | 0/9 · 0/6 | **6/9 · 2/6** |
| shortSpoken | **0/9** · 2/6 | 3/9 · 3/6 |
| correct (factual/scene) | 8/9 · **0/6** | 8/9 · **5/6** |
| noFalseClarify | 9/9 · 6/6 | 9/9 · 6/6 |

Gemma emits markdown/emoji in a third to half of voice replies ("😊" gets
handed to TTS), rambles on every open-ended MAP prompt, confabulates on 100 %
of garbled input, and went 0-for-6 on the holdout factual/scene items. The
branch's only regression axis: GPU weight residency (SmolLM fp16 724 MB vs
Gemma 536 MB, +188 MB) — the cost of the bigger brain.

## Hill-climb levers (ordered by expected payoff)

Critical path after skip-finalize ≈ **LLM first-token + TTS first-audio**
(ASR is now off the path). So:

1. ~~**Speculative LLM prefill during user speech**~~ — *tried and rejected in
   cycle 2.* It fires and adopts ~75 % of the time but regresses turn latency:
   the speculation contends with streaming ASR on the single WebGPU device, so
   first-token isn't pre-paid and the endpoint is pushed later. Removed. The
   takeaway reorders this list: on one GPU you cannot buy latency by overlapping
   stages — only by cutting total GPU work (items 3–4).
2. **Perceived first-audio** — *tried in cycle 3 (onset filler), reverted.* The
   two-AudioContext design collided with the real reply and sounded broken. A
   redo must make the filler + reply one gapless stream on one clock (prepend the
   onset as the first TTS sentence). Real TTS first-audio itself
   (`framesAfterEos`/`lsdDecodeSteps=1`) is already minimal; the first clause
   already flushes on a comma.
3. **Shorter replies** — cap first-sentence tokens; the SYSTEM_HINT already asks
   for 1–3 short sentences. A tighter "lead with one short sentence" nudge lowers
   time-to-first-audio variance.
4. ~~**LLM prefill speed**~~ — *fixed in cycle 6*: the dominant cost was
   per-prompt-length jit re-tracing, not FLOPs; `llmPrefillBucket: 64` makes
   trace shapes repeat and holds llmFirst flat (~250-350 ms) as the session
   grows. KV reuse remains a dead-end until a batched offset-prefill exists.
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
