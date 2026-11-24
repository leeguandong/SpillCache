#!/usr/bin/env python3
"""Capture dense edit/no-op trajectories and compare fixed-budget spatial reconstruction.

This is an offline fidelity experiment. It does not establish closed-loop sparse
inference, a universal interpolation guarantee, or an end-to-end speedup.
"""

from __future__ import annotations

import argparse
import os
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(os.environ.get("SPILLCACHE_MODEL_PATH", "models/Qwen-Image-Edit-2511"))
DEFAULT_DIFFUSERS = Path(os.environ.get("SPILLCACHE_DIFFUSERS_PATH", "external/diffusers"))
DEFAULT_MANIFEST = PROJECT_ROOT / "data/pilot_manifest.jsonl"
REFERENCE_INSTRUCTION = "Keep the image exactly the same."


def parse_args(argv: Any = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--diffusers-path", type=Path, default=DEFAULT_DIFFUSERS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--budgets",
        default="0.25",
        help="comma-separated active-token ratios K",
    )
    return p.parse_args(argv)


def load_manifest_sample(manifest: Path, sample_id: str) -> dict:
    with open(manifest) as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] == sample_id:
                return obj
    raise KeyError(f"sample {sample_id} not found in {manifest}")


def import_pipeline(diffusers_path: Path) -> Any:
    source_root = diffusers_path / "src"
    import_root = source_root if source_root.is_dir() else diffusers_path
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    module = importlib.import_module("diffusers")
    return module.QwenImageEditPlusPipeline


class _VelocityCapture:
    """Wrap scheduler.step to capture (velocity, latent) at each denoising step."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self._orig_step = pipe.scheduler.step
        self.velocities: list[torch.Tensor] = []
        self.latents: list[torch.Tensor] = []

    def __enter__(self) -> "_VelocityCapture":
        orig = self._orig_step

        def wrapped(model_output, timestep, sample, *args, **kwargs):
            # model_output is the CFG-combined velocity on generated tokens
            self.velocities.append(model_output.detach().to("cpu", torch.float32).clone())
            self.latents.append(sample.detach().to("cpu", torch.float32).clone())
            return orig(model_output, timestep, sample, *args, **kwargs)

        self.pipe.scheduler.step = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        self.pipe.scheduler.step = self._orig_step


def run_trajectory(
    pipe: Any,
    image: Any,
    instruction: str,
    *,
    steps: int,
    cfg: float,
    seed: int,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], Any]:
    """Run one edit and capture per-step velocity + latent."""
    generator = torch.Generator(device=device).manual_seed(seed)
    with _VelocityCapture(pipe) as cap:
        out = pipe(
            image=[image],
            prompt=instruction,
            negative_prompt=" ",
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            generator=generator,
            num_images_per_prompt=1,
        )
    return cap.velocities, cap.latents, out.images[0]


def make_active_set(
    edit_deviation: torch.Tensor, grid_h: int, grid_w: int, budget: int, strategy: str
) -> torch.Tensor:
    """Build an active-token index set of exactly `budget` tokens.

    Strategies:
    - "edit": top-budget by edit deviation magnitude (upper-bound oracle-ish).
    - "checkerboard": JiT-style regular sparse grid.
    """
    n = grid_h * grid_w
    if strategy == "edit":
        _, idx = torch.topk(edit_deviation, k=budget, largest=True)
        return idx.sort().values
    if strategy == "checkerboard":
        stride = max(1, int((n / budget) ** 0.5))
        yy, xx = torch.meshgrid(
            torch.arange(grid_h), torch.arange(grid_w), indexing="ij"
        )
        mask = ((yy % stride == 0) & (xx % stride == 0)).flatten()
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() >= budget:
            return idx[:budget].sort().values
        # pad with remaining tokens
        rest = torch.nonzero(~mask, as_tuple=False).flatten()
        pad = rest[: budget - idx.numel()]
        return torch.cat([idx, pad]).sort().values
    raise ValueError(f"unknown strategy {strategy}")


def absolute_lift(
    v_active: torch.Tensor, active_idx: torch.Tensor, grid_h: int, grid_w: int
) -> torch.Tensor:
    """JiT-style: NN gather + masked Gaussian blur on absolute velocity.

    v_active: [M, D]; returns [N, D] full velocity estimate.
    """
    import torch.nn.functional as F
    from torchvision.transforms.functional import gaussian_blur

    n = grid_h * grid_w
    d = v_active.shape[-1]
    device = v_active.device

    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()  # [N,2]
    coords_active = coords[active_idx]  # [M,2]
    dist = torch.cdist(coords, coords_active)  # [N,M]
    nearest = dist.argmin(dim=-1)  # [N]
    full_nn = v_active[nearest]  # [N,D]

    sparsity = active_idx.numel() / n
    char_dist = 1.0 / (sparsity ** 0.5)
    sigma = max(0.5, 0.4 * char_dist)
    ksize = int(2 * round(3 * sigma) + 1)
    v2d = full_nn.reshape(grid_h, grid_w, d).permute(2, 0, 1).unsqueeze(0)
    v_blur = gaussian_blur(v2d, kernel_size=[ksize, ksize], sigma=[sigma, sigma])
    v_blur = v_blur.squeeze(0).permute(1, 2, 0).reshape(n, d)

    active_mask = torch.zeros(n, dtype=torch.bool, device=device)
    active_mask[active_idx] = True
    out = torch.where(active_mask.unsqueeze(-1), full_nn, v_blur)
    return out


def deviation_lift(
    v_edit_active: torch.Tensor,
    v_ref_full: torch.Tensor,
    active_idx: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Ours: lift residual (v_edit - v_ref) on active tokens, add back v_ref.

    v_edit_active: [M, D]; v_ref_full: [N, D]; returns [N, D].
    """
    delta_active = v_edit_active - v_ref_full[active_idx]  # [M, D]
    delta_full = absolute_lift(delta_active, active_idx, grid_h, grid_w)  # reuse spatial op
    return v_ref_full + delta_full


def region_error(
    est: torch.Tensor, true: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """RMSE and relative error on a token region."""
    if mask.sum() == 0:
        return {"rmse": float("nan"), "rel": float("nan"), "count": 0}
    diff = (est[mask] - true[mask]).float()
    rmse = float(torch.sqrt((diff ** 2).mean()))
    rel = float(diff.norm() / (true[mask].float().norm() + 1e-8))
    return {"rmse": rmse, "rel": rel, "count": int(mask.sum())}


def main(argv: Any = None) -> int:
    args = parse_args(argv)
    budgets = [float(x) for x in args.budgets.split(",")]
    sample = load_manifest_sample(args.manifest, args.sample_id)
    args.output.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    pipeline_class = import_pipeline(args.diffusers_path)
    print(f"[load] loading Qwen-2511 on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    pipe = pipeline_class.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize(args.device)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    source = Image.open(sample["source_path"]).convert("RGB")

    # 1. Edit trajectory
    print(f"[edit] running edit: {sample['instruction']!r}", flush=True)
    v_edit, x_edit, img_edit = run_trajectory(
        pipe, source, sample["instruction"],
        steps=args.steps, cfg=args.cfg, seed=args.seed, device=args.device,
    )
    print(f"[edit] captured {len(v_edit)} steps, shape {tuple(v_edit[0].shape)}", flush=True)

    # 2. Reference trajectory ("keep unchanged")
    print(f"[ref] running reference: {REFERENCE_INSTRUCTION!r}", flush=True)
    v_ref, x_ref, img_ref = run_trajectory(
        pipe, source, REFERENCE_INSTRUCTION,
        steps=args.steps, cfg=args.cfg, seed=args.seed, device=args.device,
    )
    print(f"[ref] captured {len(v_ref)} steps", flush=True)

    # Grid dims: 1024 -> packed 64x64 = 4096 tokens
    n_tokens = v_edit[0].shape[1]
    grid = int(round(n_tokens ** 0.5))
    assert grid * grid == n_tokens, f"non-square grid: {n_tokens}"
    print(f"[grid] {grid}x{grid} = {n_tokens} tokens", flush=True)

    results = []
    for step in range(len(v_edit)):
        ve = v_edit[step].squeeze(0)  # [N, D]
        vr = v_ref[step].squeeze(0)   # [N, D] "keep unchanged" reference
        edit_dev = (ve - vr).norm(dim=-1)  # per-token deviation magnitude
        # Temporal reference: previous step's edit velocity (zero extra inference).
        # Step 0 has no previous step -> temporal == absolute (no reference).
        vt = v_edit[step - 1].squeeze(0) if step > 0 else None

        for K in budgets:
            budget = int(K * n_tokens)
            for strategy in ("edit", "checkerboard"):
                active_idx = make_active_set(edit_dev, grid, grid, budget, strategy)
                inactive = torch.ones(n_tokens, dtype=torch.bool)
                inactive[active_idx] = False

                abs_est = absolute_lift(ve[active_idx], active_idx, grid, grid)
                dev_est = deviation_lift(ve[active_idx], vr, active_idx, grid, grid)
                abs_err = region_error(abs_est, ve, inactive)
                dev_err = region_error(dev_est, ve, inactive)

                # Temporal-reference deviation lift (zero-extra-cost variant)
                if vt is not None:
                    tmp_est = deviation_lift(ve[active_idx], vt, active_idx, grid, grid)
                    tmp_err = region_error(tmp_est, ve, inactive)
                else:
                    tmp_err = abs_err

                results.append({
                    "step": step, "K": K, "strategy": strategy,
                    "budget": budget,
                    "absolute_lift": abs_err,
                    "deviation_lift": dev_err,
                    "temporal_lift": tmp_err,
                    "improvement_rel": (
                        (abs_err["rel"] - dev_err["rel"]) / (abs_err["rel"] + 1e-8)
                    ),
                })
        if step % 5 == 0:
            print(f"[recon] step {step}/{len(v_edit)} done", flush=True)

    report = {
        "sample_id": args.sample_id,
        "instruction": sample["instruction"],
        "reference_instruction": REFERENCE_INSTRUCTION,
        "n_tokens": n_tokens, "grid": grid, "steps": args.steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_json = args.output / "fidelity.json"
    out_json.write_text(json.dumps(report, indent=2))
    img_edit.save(args.output / "edit.png")
    img_ref.save(args.output / "reference.png")

    # Aggregate summary
    print("\n" + "=" * 70)
    print(f"LIFTING FIDELITY SUMMARY — {args.sample_id}")
    print("=" * 70)
    for K in budgets:
        for strategy in ("edit", "checkerboard"):
            subset = [r for r in results if r["K"] == K and r["strategy"] == strategy]
            abs_rel = sum(r["absolute_lift"]["rel"] for r in subset) / len(subset)
            dev_rel = sum(r["deviation_lift"]["rel"] for r in subset) / len(subset)
            impr = (abs_rel - dev_rel) / abs_rel * 100
            print(f"  K={K:.2f} {strategy:12s} | abs_rel={abs_rel:.4f} "
                  f"dev_rel={dev_rel:.4f} | deviation-lift better by {impr:+.1f}%")
    print("=" * 70)
    print(f"[done] report saved to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
