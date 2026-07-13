# Conversation-quality campaign — diagnosis & roadmap

Five parallel diagnosis agents (hearing / brain / flow / grounding / repair) read the
code and decomposed the live-session failures. This file is the ranked roadmap after dedupe + cross-validation; Tier 1 is implemented, and the Tier-2
"repair clause + few-shot exemplar" shipped in cycle 7 (docs/BENCHMARKS.md).
(Historical: the one Gemma mention below records that brain's config at
diagnosis time; the shipped brain is SmolLM2-360M.)

## Cross-agent convergence (the strongest findings)

The "answering Cisco" turn decomposed into THREE stacked failures, found
independently by three agents:
1. **Buffer contamination** (flow C1): at barge-in the capture buffer is kept and
   spans the whole reply period; the echo filter is off by transcription time
   (duplex.ts:846, streaming.ts:241) → reply words + ambient leak into the turn
   ("to Now I actually Actually…" prefix).
2. **Acoustic miss** (hearing): base.en greedily mishears "San Francisco" →
   "answering Cisco"; turn text is literally the newest single hypothesis
   (streaming.ts:103) — no orchestration fix can recover it.
3. **Confabulation** (brain/repair): the LLM has no instruction for unclear input
   (SMOLLM_SYSTEM says nothing) → "glad you're answering Cisco correctly!". The
   three duplex guards are all silent discards; the system can never SAY "I
   didn't catch that".

## Tier 1 — confirmed structural bugs, low effort (do immediately)

| Fix | Source | Impact/Effort | Evidence |
|---|---|---|---|
| Port `windowHistory` to SmolLmChatModel (and consider KV reuse later) | brain | H/L | encodePrompt iterates FULL history (pipeline.ts:1154); Gemma capped at 16 turns; prefill cost + drift grow per turn |
| Fix scene-format mismatch | brain+grounding (both found it) | M/L | SMOLLM_SYSTEM briefs a `[scene: …]` tag nothing emits; duplex sends `"(Right now through the camera you can see …)"` (duplex.ts:605 vs pipeline.ts:1090) |
| Humanize failure lines; never speak the raw query | repair | M/trivial | tools.ts:168/219 echo garble aloud ("…a place called San Francisco instead..") → "I couldn't find that place — which city was it?" |
| Calc hyphen-range false-positive | grounding | M/L | "3-4 people" → "3 minus 4 is -1"; broad mode runs calc FIRST |
| "forecast for tomorrow" geocodes "tomorrow" | grounding | M/L | time-words after "for" reach the geocoder as a place |
| Backchannel window is unreachable | flow C7 | M/L | endpoints fire at 380/620ms before the [450,800) backchannel window is checked → backchannels effectively dead |

## Tier 2 — the garble-repair stack (worst live failure; measurable)

| Candidate | Source | Impact/Effort | How to measure |
|---|---|---|---|
| System-prompt repair clause + 1 few-shot exemplar | brain+repair | H/L | **Phase-2 A/B running**: binary asks-to-clarify vs confabulates on garbled items; false-clarify rate on clean items. Binary metric sidesteps the ±0.85 judge noise floor |
| ASR confidence gate (avg logprob, nearly free — logits already on CPU in decoding.ts:37) → spoken "sorry, could you say that again?" | hearing+repair | H/M | threshold sweep on scripted clean vs garbled PCM (fake-mic); ship at <5% false-gate |
| Barge-in buffer hygiene (trim to ~1.2s pre-roll; keep echo-filtering aborted reply text) | flow C1 | H/M | fake-mic WAV interrupting a reply; assert zero reply words in barge turn via __turnLog |
| Post-fire continuation-merge (speech resumes <700ms before first TTS audio → abort reply, re-endpoint same buffer) | flow C2 | H/M-H | scripted WAV with mid-sentence 700ms pause → one merged turn |
| Correction-turn tagging ("No, …" rapid post-reply → prefix "(the user is correcting your previous answer)") | flow+repair | M-H/L | scripted correction convo, judge reply relevance |

## Tier 3 — capability upgrades (bigger, gated)

| Candidate | Source | Impact/Effort | Gate |
|---|---|---|---|
| **Whisper small.en** (144→481 MB, ~3× FLOPs, est. 500-900ms/pass) | hearing | H/M | fake-mic WER fixture incl. "San Francisco". **Synergy: quant campaign frees 360 MB** — int8 brain (363) + small.en (481) = 844 MB ≈ today's 868 MB total, with better ears |
| LLM-emitted tool markers (function-calling-lite) replacing lookup/weather regexes | grounding C1 | H/M | build routing test-set (C5, ~50-80 labeled utterances) FIRST; ship only at marker P≥0.9/R≥0.8 |
| Tool-context memory (lastTool for 2 turns → "what about tomorrow?", "and in London?"; feed card extract into next LLM turn) | grounding C2 | H/M | follow-up section of routing test-set |
| Richer-but-hedged scene facts (positions/colors exist in vision.ts but are withheld) + honest ceiling line | grounding C4 | M/M | groundedness judge on scene items |
| Fresh-tail reconciliation at turn end (one bounded extra ASR pass when hypothesis is stale) | hearing #3 | M/L-M | last-word error rate on fixtures |
| Scope/decay the repetition penalty (currently the whole previous reply incl. function words, never decays — plausible cause of nonsense jokes) | brain 4th | M/L | joke-coherence judge A/B |

## Negative results / rejected
- Whisper decode prompt-biasing: mechanically possible but sot_prev conditioning is
  a hallucination amplifier, and WhisperTokenizer is decode-only (no encoder). Skip.
- Pre-fire endpoint patience: cycle-5 already proved it can't work (latency);
  post-fire merge (flow C2) is the viable form.

## Verdict
Conversation quality is NOT one lever. The single worst live failure (garbled turn
→ confident confabulation) is a three-layer stack needing hearing+flow+brain fixes
that reinforce: buffer hygiene reduces garble frequency, the confidence gate
catches what remains, and the repair prompt makes the model graceful when both
miss. Tier 1 is a half-day of near-free confirmed-bug fixes; Phase-2 A/B
(repair prompt matrix) is running headlessly now.
