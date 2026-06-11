# gemma4-diffusion

Rigorous Apple Silicon benchmark: **does DiffusionGemma's block-diffusion decode
speedup actually materialize on an M3 Max — or does the "compute-bound" design
backfire on compute-poor hardware?**

**TL;DR — it backfires.** At 4-bit on M3 Max, the diffusion path is ~3× slower than
autoregressive on average (up to ~5× on long-form) and uses more memory. Adaptive
compute is real (denoising steps scale 3→113 with prompt complexity) but it's a
liability here: each step is a full-canvas forward that M3 Max compute can't hide.

See **[RESULTS.md](RESULTS.md)** for the full table and methodology, **[env.md](env.md)**
for the exact setup, **[NOTES.md](NOTES.md)** for blockers/deviations.

Apples-to-apples: same Gemma-4 26B-A4B-it, same 4-bit MLX (group_size 64), same
mlx-vlm `generate()` sampler (not reimplemented), same prompts, temp 0. Coherence
verified before trusting any tok/s.

```
python bench.py mlx-community/diffusiongemma-26B-A4B-it-4bit  sanity   # coherence + path gate
python bench.py mlx-community/diffusiongemma-26B-A4B-it-4bit  matrix   # diffusion metrics
python bench.py lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit matrix  # AR baseline
```
