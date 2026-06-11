# DiffusionGemma vs Gemma-4 AR on Apple Silicon (M3 Max) — Results

**Question:** Does the block-diffusion decode speedup actually materialize on M3 Max,
or does the "compute-bound" design backfire on compute-poor hardware?

**Answer: NEGATIVE.** On M3 Max (4-bit), the diffusion path is **slower than
autoregressive at every prompt size**, and the gap *widens* with length. Google's
own "compute-bound" framing predicts Apple Silicon as the worst case — confirmed.

## Setup (apples-to-apples)
- Same machine: Apple M3 Max, 128 GB, macOS 26.4.
- Same family / size / precision: Gemma-4 **26B-A4B-it**, **4-bit MLX, group_size 64** on both sides.
  - Diffusion: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
  - AR baseline: `lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit`
- Same sampler path: mlx-vlm `generate()` (diffusion sampler `mlx_vlm/generate/diffusion.py`,
  not reimplemented). mlx-vlm 0.6.3 @ git main, mlx 0.31.2.
- Same prompts, temperature 0.0. Coherence verified before trusting any tok/s (PHASE 1 passed:
  Tokyo / 80 km·h⁻¹ / correct palindrome; long-form essay & story coherent).
- **Precision caveat:** brief asked BF16; only ~26 GB disk free forced 4-bit on both sides and
  sequential runs. Numbers characterize the *quantized* regime (see env.md).

## Per-prompt comparison (decode throughput)

| category | prompt | denoise steps | eff tok/fwd | DIFF tok/s | AR tok/s | tok/s ratio | DIFF wall | AR wall | DIFF mem | AR mem |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| short_structured | short_code | 3 | 7.3 | 42.5 | 94.8 | **0.45×** | 0.71s | 1.34s | 17.8GB | 15.7GB |
| short_structured | short_json | 3 | 10.0 | 58.4 | 95.2 | **0.61×** | 0.62s | 0.49s | 17.8GB | 15.7GB |
| medium_chat | medium_chat | 48 | 10.7 | 32.5 | 91.1 | **0.36×** | 15.82s | 5.70s | 18.3GB | 15.8GB |
| medium_chat | medium_advice | 58 | 8.8 | 24.3 | 90.2 | **0.27×** | 21.17s | 5.78s | 18.3GB | 15.8GB |
| long_form | long_essay | 113 | 6.7 | 16.0 | 89.5 | **0.18×** | 47.28s | 10.09s | 18.4GB | 15.9GB |
| long_form | long_story | 93 | 6.2 | 15.5 | 89.9 | **0.17×** | 37.27s | 9.38s | 18.4GB | 15.9GB |

### Aggregates (decode tok/s, diffusion ÷ AR)
- short structured: 50.5 vs 95.0 → **0.53×**, ~3 denoising steps
- medium chat:      28.4 vs 90.7 → **0.31×**, ~53 steps
- long-form:        15.7 vs 89.7 → **0.18×**, ~103 steps
- **overall mean ratio: 0.34×** (diffusion is ~3× slower)

## Findings

1. **AR wins everywhere.** AR holds a flat ~90–95 tok/s regardless of prompt;
   diffusion ranges 15–58 tok/s and never reaches AR. Best diffusion case
   (short_json, 58 tok/s) still loses to AR (95 tok/s).

2. **The adaptive-compute claim is real — but it's a liability here, not a win.**
   Denoising steps scale cleanly with prompt complexity (3 → ~53 → ~103). The arch
   *does* spend less compute on simple prompts. But every denoising step is a
   **full-canvas forward pass**, and on compute-bound M3 Max that compute cannot be
   hidden: as steps grow, throughput collapses (0.53× → 0.18×) and the AR gap widens.

3. **Effective tokens/forward (~6–11) never pays for itself.** Parallel multi-token
   decode per step does not amortize the per-step full-canvas compute on this hardware.
   AR's one-token-per-forward, memory-bandwidth-bound decode is simply a better fit for
   Apple Silicon's compute/bandwidth balance.

4. **Diffusion also costs more memory:** ~17.8–18.4 GB peak vs ~15.7–15.9 GB for AR
   (the denoising canvas), so there isn't even a memory-for-speed trade to fall back on.

5. **Wall-clock caveat:** diffusion and AR emit different output lengths, so wall-clock
   isn't perfectly normalized; tok/s is the fair metric. Wall-clock still corroborates
   (long-form diffusion 4–5× slower end-to-end).

## Bottom line
On M3 Max in the 4-bit regime, **block-diffusion decode for Gemma-4 26B-A4B is a net
regression vs autoregressive** — ~3× slower on average, up to ~5× on long-form, with
higher memory. The compute-bound design backfires on Apple Silicon exactly as the
architecture's framing predicts. Worth re-checking at BF16 on a machine with disk
headroom, but 4-bit *helps* the memory-bound AR path more than the compute-bound
diffusion path, so BF16 is unlikely to reverse the verdict.
