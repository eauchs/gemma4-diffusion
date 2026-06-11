# Environment — gemma4-diffusion benchmark

## Hardware
- Apple M3 Max, 128 GB unified memory
- macOS 26.4
- **Free disk: ~26 GB** (hard constraint — see precision decision)

## Software (captured 2026-06-11)
- python      3.12.12  (uv venv `.venv`; system python is 3.9, too old — mlx-vlm main needs >=3.10)
- mlx_vlm     0.6.3    (installed from git+https://github.com/Blaizzy/mlx-vlm.git @ 5a4222a "Fix DiffusionGemma long-context prefill (#1348)", 2026-06-10)
- mlx         0.31.2
- mlx_lm      0.31.3
- transformers 5.11.0
- Diffusion sampler path: mlx_vlm/generate/diffusion.py  (verified importable)

## Models
- Diffusion: mlx-community/diffusiongemma-26B-A4B-it-4bit  (15.7 GB)
- AR baseline: lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit (15.6 GB)

## Precision decision (deviation from brief)
Brief asked BF16 first (51.7 GB for diffusion). Only ~26 GB disk free, and even two
4-bit models (~31 GB) cannot coexist. Decision:
- Use standard MLX **4-bit** on BOTH sides (same quant scheme) → comparison stays apples-to-apples.
- Run **sequentially**: download diffusion -> bench -> purge -> download AR -> bench.
- CAVEAT: results characterize the *quantized* regime. 4-bit shifts the compute/memory
  balance vs BF16, so the "compute-bound on Apple Silicon" question is answered for 4-bit,
  not BF16. Flagged for write-up.
- AVOIDED: mxfp4 (known crash, issue #1350), nvfp4 (different scheme), OptiQ-4bit /
  heretic-4bit (different quantizer / abliterated weights — not a clean AR baseline).
