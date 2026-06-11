"""Benchmark harness: DiffusionGemma (diffusion path) vs Gemma-4 AR baseline.

Uses mlx-vlm's existing unified generate() — we do NOT reimplement the sampler.
generate() dispatches to the diffusion or AR path based on the loaded model and
returns a GenerationResult whose fields already expose every metric PHASE 2 asks
for (generation_tps, prompt_tps, peak_memory, diffusion_denoising_steps,
diffusion_canvas_tokens, diffusion_work_tokens, diffusion_*_tps).

Usage:
    python bench.py <hf_repo> <sanity|matrix> [--max-denoising-steps N] [--out results.json]

Coherence + fallback gate: for a diffusion model we assert diffusion fields are
populated. If they are zero, the run silently behaved like AR -> the comparison
would be invalid, so we flag it loudly.
"""
import argparse
import gc
import json
import sys
import time

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

import prompts as P

METRIC_FIELDS = [
    "prompt_tokens", "generation_tokens", "total_tokens",
    "prompt_tps", "generation_tps", "peak_memory", "finish_reason",
    "diffusion_canvas_tokens", "diffusion_denoising_steps",
    "diffusion_work_tokens", "diffusion_canvas_tps", "diffusion_work_tps",
]


def run_one(model, processor, item, diff_kwargs):
    formatted = apply_chat_template(processor, model.config, item["prompt"])
    mx.clear_cache()
    t0 = time.perf_counter()
    res = generate(
        model, processor, formatted,
        max_tokens=item["max_tokens"],
        verbose=False,
        temperature=0.0,
        **diff_kwargs,
    )
    wall = time.perf_counter() - t0
    rec = {"id": item["id"], "category": item.get("category"),
           "max_tokens": item["max_tokens"], "wall_s": round(wall, 3)}
    for f in METRIC_FIELDS:
        v = getattr(res, f, None)
        rec[f] = round(v, 4) if isinstance(v, float) else v
    rec["text"] = res.text
    # effective tokens per forward pass: how much real text each denoising step yields
    steps = rec.get("diffusion_denoising_steps") or 0
    rec["is_diffusion_run"] = bool(steps > 0 or (rec.get("diffusion_canvas_tokens") or 0) > 0)
    rec["eff_tokens_per_forward"] = (
        round(rec["generation_tokens"] / steps, 3) if steps > 0 else None
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("phase", choices=["sanity", "matrix"])
    ap.add_argument("--max-denoising-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diff_kwargs = {}
    if args.max_denoising_steps is not None:
        diff_kwargs["max_denoising_steps"] = args.max_denoising_steps

    print(f"[load] {args.repo}", flush=True)
    t0 = time.perf_counter()
    model, processor = load(args.repo)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s  model_type={model.config.model_type}", flush=True)

    items = P.SANITY if args.phase == "sanity" else P.MATRIX
    records = []
    for it in items:
        print(f"[run] {it['id']} ...", flush=True)
        rec = run_one(model, processor, it, diff_kwargs)
        flag = "DIFFUSION" if rec["is_diffusion_run"] else "AR/fallback"
        print(f"      {flag}  gen_tps={rec['generation_tps']}  "
              f"steps={rec['diffusion_denoising_steps']}  "
              f"eff_tok/fwd={rec['eff_tokens_per_forward']}  "
              f"peak={rec['peak_memory']}GB  wall={rec['wall_s']}s", flush=True)
        records.append(rec)

    out = {
        "repo": args.repo,
        "phase": args.phase,
        "model_type": model.config.model_type,
        "diff_kwargs": diff_kwargs,
        "records": records,
    }
    path = args.out or f"results_{args.phase}_{args.repo.split('/')[-1]}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}", flush=True)

    # PHASE 1 gate summary
    if args.phase == "sanity":
        n_diff = sum(r["is_diffusion_run"] for r in records)
        print(f"\n[gate] diffusion-path runs: {n_diff}/{len(records)}")
        if n_diff < len(records):
            print("[gate] WARNING: some runs did NOT exercise the diffusion path "
                  "(diffusion metrics are zero). Comparison may be invalid — investigate.")


if __name__ == "__main__":
    main()
