# NOTES — blockers, bugs, deviations

## 2026-06-11 PHASE 0
- System python 3.9.6 too old for mlx-vlm main (requires >=3.10). Created uv venv py3.12. RESOLVED.
- PyPI mlx-vlm 0.1.15 has NO diffusion path. Installed from git main (0.6.3) which has
  mlx_vlm/generate/diffusion.py + models/diffusion_gemma/. RESOLVED.
- Disk: only ~26 GB free -> forced 4-bit + sequential runs (see env.md). DEVIATION from BF16 brief.
- Known upstream crash to watch: mxfp4 dequantize mode (issue #1350) — not using mxfp4 weights.
