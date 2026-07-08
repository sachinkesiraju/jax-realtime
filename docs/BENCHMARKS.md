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
TML's 0.40 s; the remaining big lever is **speculative LLM prefill during the
user's pause** (start Gemma on the stable committed prefix so first-token is
pre-paid) — not yet implemented (risk of false starts).

### Response quality note
Gemma 270M is the local ceiling. A too-long/negative system prompt was priming
the model to answer every turn with "Okay, I understand" — fixed by a short,
positive prompt (factual Qs like "capital of Japan?" → "Tokyo" now work) and
cooler sampling (temp 0.8→0.45). Open-ended prompts still degrade; the honest
fixes are tool-grounding (weather/Wikipedia return real facts) or the Cerebras
key for a larger Brain.

## Hill-climb levers (ordered by expected payoff)

Critical path after skip-finalize ≈ **LLM first-token + TTS first-audio**
(ASR is now off the path). So:

1. **Speculative LLM prefill during user speech** — when committed text is stable
   for ~1 tick and the user is mid-pause, start Gemma prefill on the committed
   prefix; if the turn then ends with that prefix unchanged, first-token is
   already paid. Biggest single win; guard against wasted prefill on false ends.
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
