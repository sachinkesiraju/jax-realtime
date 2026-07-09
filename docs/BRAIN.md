# "Smarter brain" map-reduce campaign — final

**Question:** the agent's LLM replies feel amateur / not intelligent. Can prompting,
sampling, or a different model make the brain smarter — inside a browser voice
demo running on jax-js?

**Harness:** candidate models run headlessly via `transformers` (float32/MPS),
matching the browser weights. 3 independent **blind** LLM judges (config identity
hidden behind per-item randomized letters) score each reply 1–5 on `helpful`,
`coherent` (incl. factual accuracy), `spoken` (natural aloud, no markdown), `smart`.
OVERALL = mean of the four; judges averaged. Eval = 12 dev + 8 holdout + 10 extra
realistic voice turns, disjoint.

## Noise floor (measured 4×)

Rerunning the identical baseline with a different seed swung OVERALL by ~1.0 at
n=12 and ~0.84 at n=30, at both temp 0.7 and 0.5. **Run-to-run + judge noise ≈
±0.85–1.0.** Any single-config OVERALL delta inside that band is not signal.

## Prompt / sampling on Gemma 270M (rounds 1–3) — net wash

- Sampling (temp 0.4–0.7, top-p) alone: nothing beyond noise.
- Few-shot exemplars: didn't fix formatting, hurt helpfulness.
- **System prompt: reliably trades dimensions** — +0.8–1.1 spoken, +0.5 coherent
  (kills markdown/monologue/garbage), but −0.5 helpful, −0.5 smart (concise
  instruction → intro-and-stop / persona echo). Net ~0 on OVERALL.
- Best prompt = `P2_fmtbrev` ("answer naturally and completely in a sentence or
  two of plain spoken language — no lists/markdown"): spoken 2.60→3.69, coherent
  2.59→3.20, and keeps helpful highest of the prompts (2.21). Fixes the amateur
  *look*; does not add intelligence.

**Conclusion: you cannot make the 270M smarter via prompt/sampling. Ceiling = model.**

## Model shootout — the real lever (raw, no system prompt, temp 0.7)

MAP set (dev+extra, n=22):

| model | helpful | coherent | spoken | smart | OVERALL | vs 270M |
|---|---|---|---|---|---|---|
| **SmolLM2-360M** | **3.50** | **3.64** | **3.76** | **2.82** | **3.43** | **+0.92** |
| Qwen2.5-1.5B | 3.17 | 2.86 | 2.71 | 2.58 | 2.83 | +0.32 |
| Gemma 3 1B | 3.09 | 2.20 | 2.36 | 2.70 | 2.59 | +0.08 |
| Gemma 3 270M (shipped) | 2.52 | 2.50 | 3.09 | 1.94 | 2.51 | — |
| Qwen2.5-0.5B | 2.35 | 2.17 | 2.29 | 1.86 | 2.17 | −0.34 |

Holdout (8 unseen items) — **validates the winner**:

| model | helpful | coherent | spoken | smart | OVERALL | vs 270M |
|---|---|---|---|---|---|---|
| **SmolLM2-360M** | 3.33 | 2.96 | 3.54 | 2.71 | **3.14** | **+0.94** |
| Qwen2.5-1.5B | 2.71 | 2.17 | 2.33 | 2.42 | 2.41 | +0.21 |
| Gemma 3 270M | 2.46 | 1.96 | 2.21 | 2.17 | 2.20 | — |

## Verdict

1. **SmolLM2-360M-Instruct is the smarter brain.** Nearly the same size as Gemma
   270M (~360M; 693 MB fp16 vs 544 MB), yet **+0.9 OVERALL, winning on EVERY
   dimension**, replicated on holdout (+0.92 → +0.94). No garbage-token injection.
   This dwarfs every prompt/sampling effect and is the only change that adds
   actual intelligence at ~no size/latency cost.
2. Bigger Gemma (1B) is NOT worth it: +0.08, and ~1.7 GB + ~4× latency.
   Qwen2.5-0.5B is worse than Gemma 270M; Qwen2.5-1.5B is better but big + verbose.
3. The `P2_fmtbrev` system prompt is a cheap interim win on the *current* Gemma
   (fixes the amateur look), and would also help SmolLM2.

## Verification round (n=30, more same-size + larger competitors) — confirms winner

| model | helpful | coherent | spoken | smart | OVERALL | vs 270M |
|---|---|---|---|---|---|---|
| **SmolLM2-360M** | 3.19 | 3.42 | 3.41 | 2.58 | **3.15** | **+0.92** |
| Llama-3.2-1B | 3.16 | 2.60 | 2.08 | 2.81 | 2.66 | +0.44 |
| Qwen3-0.6B | 2.70 | 2.47 | 2.40 | 2.22 | 2.45 | +0.22 |
| Gemma 3 270M | 2.29 | 2.26 | 2.47 | 1.89 | 2.23 | — |
| SmolLM2-135M | 1.98 | 2.02 | 2.39 | 1.62 | 2.00 | −0.22 |

SmolLM2-360M wins a **third** independent judged run (+0.92, all four dims), and
beats even the 3× larger Llama-3.2-1B and Qwen2.5-1.5B. 135M is too small.
**Final pick: SmolLM2-360M-Instruct.** Bonus: it honors a real system role, so it
also takes the P2 spoken-format prompt (Gemma 270M couldn't).

## To ship SmolLM2 in jax-js

SmolLM2 = Llama architecture (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings) —
*simpler* than the existing Gemma 3 impl (no QK-norm, no sliding-window interleave,
no logit softcap). Needs: a new `llama`-style forward pass in jax-js mirroring
`gemma.ts` (attention + KV cache + fused decode + GPU top-k), a safetensors
weight loader for the Llama key layout, and hosting the ~360M weights (int8-embed
to match the current download budget). Same size ⇒ comparable in-browser latency.

## The port (src/llm/smollm.ts)

SmolLM2 is a Llama-architecture model, so `smollm.ts` is a Llama sibling of
`gemma.ts` (simpler: pre-norm with 2 norms/layer, no QK-norm, single RoPE theta,
SiLU MLP, no embedding scale, RMSNorm scale is the raw weight). Same fused
single-dispatch decode + GPU top-k path. Tokenizer is a precomputed
tiktoken-style artifact (the GPT2/BPE `BpeEncoding` in `@jax-js/loaders`).

Two bugs were found and fixed during the port, each validated by comparing the
browser's greedy token ids against HuggingFace `transformers` (now **bit-exact**):

1. **Prompt tokenization** — `BpeEncoding.encode` in `@jax-js/loaders` advances
   only 1 char past each special token (uses `RegExpExecArray.length`, always 1,
   instead of the matched string's length), re-tokenizing `<|im_start|>` etc. as
   ordinary text. Worked around by assembling the ChatML prompt token array
   manually (encode special-free segments, splice known special ids in).
2. **GQA head grouping** — jax-js `dotProductAttention` expands KV heads with
   `tile` (block-repeat) rather than HF's `repeat_interleave`, so with 15 query /
   5 KV heads the query heads paired with the wrong KV heads. (Gemma has 1 KV
   head, where the two are identical, so it never surfaced.) Fixed by
   repeat-interleaving K/V to 15 heads in our code before the attention call.

Validated in-browser: greedy decoding is token-for-token identical to HF; decode
~60 ms/token, turn latency ~1.2–2.8 s — comparable to the Gemma brain.
