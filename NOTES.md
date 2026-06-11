# NOTES — blockers, bugs, deviations

## 2026-06-11 PHASE 0
- System python 3.9.6 too old for mlx-vlm main (requires >=3.10). Created uv venv py3.12. RESOLVED.
- PyPI mlx-vlm 0.1.15 has NO diffusion path. Installed from git main (0.6.3) which has
  mlx_vlm/generate/diffusion.py + models/diffusion_gemma/. RESOLVED.
- Disk: only ~26 GB free -> forced 4-bit + sequential runs (see env.md). DEVIATION from BF16 brief.
- Known upstream crash to watch: mxfp4 dequantize mode (issue #1350) — not using mxfp4 weights.

## 2026-06-12 PHASES 1-3 outcome
- PHASE 1 sanity: diffusion path active 3/3, output coherent. PASS.
- AR baseline quant verified: bits=4, group_size=64, model_type=gemma4 -> matches diffusion 4-bit scheme.
- The "AR/fallback" flag on the AR baseline is EXPECTED (it IS the AR model); not a bug.
- PHASE 3: see RESULTS.md. NEGATIVE result — diffusion ~3x slower than AR on M3 Max @4-bit,
  gap widens with length (0.53x short -> 0.18x long), diffusion uses more memory.
- No crashes hit (avoided mxfp4 weights tied to issue #1350). Did not need to file a bug.
- LIMITATION: 4-bit only (disk-constrained), single-run per prompt (no averaging over N repeats),
  temperature 0.0. For a publishable claim: rerun at BF16 w/ disk headroom + N>=5 repeats + warmup.
