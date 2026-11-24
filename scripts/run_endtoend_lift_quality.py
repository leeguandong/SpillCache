#!/usr/bin/env python3
"""Compare decoded-image fidelity under captured dense-trajectory replay.

The historical filename uses endtoend for decoding the replayed latent into an
image. No sparse transformer is evaluated at replayed latent states, so these
measurements are not closed-loop inference or a speedup benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts.run_lifting_fidelity import (
    import_pipeline,
    load_manifest_sample,
    make_active_set,
    absolute_lift,
    deviation_lift,
    run_trajectory,
    DEFAULT_MODEL,
    DEFAULT_DIFFUSERS,
    DEFAULT_MANIFEST,
    REFERENCE_INSTRUCTION,
)


def parse_args(argv=None):
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
    p.add_argument("--budgets", default="0.25")
    p.add_argument("--dense-only-last", type=int, default=1,
                   help="keep the final N steps fully dense (step-27 divergence fix)")
    return p.parse_args(argv)


class _TrajCapture:
    """Capture (velocity, latent_in, sigma, sigma_next) per step for replay."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self._orig = pipe.scheduler.step
        self.velocities: list[torch.Tensor] = []
        self.latents_in: list[torch.Tensor] = []
        self.sigmas: list[float] = []
        self.sigmas_next: list[float] = []

    def __enter__(self):
        orig = self._orig
        sched = self.pipe.scheduler

        def wrapped(model_output, timestep, sample, *a, **k):
            idx = sched.step_index
            if idx is None:
                sched._init_step_index(timestep)
                idx = sched.step_index
            sig = float(sched.sigmas[idx])
            sig_next = float(sched.sigmas[idx + 1])
            self.velocities.append(model_output.detach().to("cpu", torch.float32).clone())
            self.latents_in.append(sample.detach().to("cpu", torch.float32).clone())
            self.sigmas.append(sig)
            self.sigmas_next.append(sig_next)
            return orig(model_output, timestep, sample, *a, **k)

        sched.step = wrapped
        return self

    def __exit__(self, *exc):
        self.pipe.scheduler.step = self._orig


def replay_with_lift(
    velocities, latents_in, sigmas, sigmas_next, *, grid, budget, strategy,
    v_ref_full=None, dense_last=1,
):
    """Replay the flow-match Euler ODE using LIFTED velocity at each step.

    FlowMatchEuler update: x_{next} = x + (sigma_next - sigma) * v.
    We start from the true initial latent and chain estimated velocities so
    lifting error accumulates realistically.
    """
    x = latents_in[0].clone()  # [1, N, D]
    n_steps = len(velocities)
    for t in range(n_steps):
        ve = velocities[t].squeeze(0)  # [N, D] true edit velocity
        if t >= n_steps - dense_last:
            v_use = ve  # keep final steps dense (divergence fix)
        else:
            edit_dev = (ve - v_ref_full[t].squeeze(0)).norm(dim=-1) if v_ref_full is not None \
                else ve.norm(dim=-1)
            active_idx = make_active_set(edit_dev, grid, grid, budget, strategy)
            if v_ref_full is None:
                v_est = absolute_lift(ve[active_idx], active_idx, grid, grid)
            else:
                v_est = deviation_lift(ve[active_idx], v_ref_full[t].squeeze(0),
                                       active_idx, grid, grid)
            v_use = v_est
        x = x + (sigmas_next[t] - sigmas[t]) * v_use.unsqueeze(0)
    return x  # final latent [1, N, D]


def decode_latent(pipe, latent, sample, device):
    """Unpack + VAE-decode a packed latent to a PIL image."""
    height = sample.get("height", 1024)
    width = sample.get("width", 1024)
    lat = latent.to(device=device, dtype=pipe.vae.dtype)
    lat = pipe._unpack_latents(lat, height, width, pipe.vae_scale_factor)
    lmean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, lat.dtype)
    lstd = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, lat.dtype)
    lat = lat / lstd + lmean
    with torch.no_grad():
        img = pipe.vae.decode(lat, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


def capture_traj(pipe, image, instruction, *, steps, cfg, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    with _TrajCapture(pipe) as cap:
        out = pipe(image=[image], prompt=instruction, negative_prompt=" ",
                   num_inference_steps=steps, true_cfg_scale=cfg,
                   generator=generator, num_images_per_prompt=1)
    return cap, out.images[0]


def main(argv=None):
    args = parse_args(argv)
    budgets = [float(x) for x in args.budgets.split(",")]
    sample = load_manifest_sample(args.manifest, args.sample_id)
    args.output.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from spillcache.fidelity import FidelityEvaluator

    pipeline_class = import_pipeline(args.diffusers_path)
    print(f"[load] loading on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    pipe = pipeline_class.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize(args.device)
    print(f"[load] done {time.perf_counter()-t0:.1f}s", flush=True)

    source = Image.open(sample["source_path"]).convert("RGB")

    print("[edit] capturing edit trajectory", flush=True)
    edit_cap, img_edit = capture_traj(pipe, source, sample["instruction"],
                                      steps=args.steps, cfg=args.cfg, seed=args.seed, device=args.device)
    print("[ref] capturing reference trajectory", flush=True)
    ref_cap, _ = capture_traj(pipe, source, REFERENCE_INSTRUCTION,
                              steps=args.steps, cfg=args.cfg, seed=args.seed, device=args.device)

    n_tokens = edit_cap.velocities[0].shape[1]
    grid = int(round(n_tokens ** 0.5))
    v_ref_full = ref_cap.velocities  # list of [1,N,D]

    # Baseline: decode the captured edit trajectory's final latent as our "dense" anchor.
    # This isolates lifting error from any decode mismatch vs the on-disk reference.
    dense_final = edit_cap.latents_in[-1]  # last latent_in is pre-final-step; replay full for true final
    # Recompute the true final latent by replaying with all-true velocity:
    x_true = edit_cap.latents_in[0].clone()
    for t in range(len(edit_cap.velocities)):
        x_true = x_true + (edit_cap.sigmas_next[t] - edit_cap.sigmas[t]) * edit_cap.velocities[t]
    img_dense = decode_latent(pipe, x_true, sample, args.device)
    dense_path = args.output / "dense_replay.png"
    img_dense.save(dense_path)
    img_edit.save(args.output / "edit_pipeline.png")

    evaluator = FidelityEvaluator(include_lpips=True, device=args.device)
    results = []
    for K in budgets:
        budget = int(K * n_tokens)
        for strategy in ("edit", "checkerboard"):
            for lift_name, vref in (("absolute", None), ("deviation", v_ref_full)):
                x_final = replay_with_lift(
                    edit_cap.velocities, edit_cap.latents_in,
                    edit_cap.sigmas, edit_cap.sigmas_next,
                    grid=grid, budget=budget, strategy=strategy,
                    v_ref_full=vref, dense_last=args.dense_only_last,
                )
                img = decode_latent(pipe, x_final, sample, args.device)
                out_png = args.output / f"K{K}_{strategy}_{lift_name}.png"
                img.save(out_png)
                m = evaluator.evaluate(out_png, dense_path)
                results.append({"K": K, "strategy": strategy, "lift": lift_name,
                                **m.to_dict()})
                print(f"  K={K} {strategy:12s} {lift_name:9s} "
                      f"LPIPS={m.lpips:.4f} PSNR={m.psnr_db:.2f} SSIM={m.ssim:.4f}", flush=True)

    report = {"sample_id": args.sample_id, "instruction": sample["instruction"],
              "n_tokens": n_tokens, "grid": grid, "steps": args.steps,
              "dense_last": args.dense_only_last,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "results": results}
    (args.output / "quality.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print(f"END-TO-END LIFT QUALITY — {args.sample_id} (vs dense replay)")
    print("=" * 72)
    for K in budgets:
        for strategy in ("edit", "checkerboard"):
            a = next(r for r in results if r["K"] == K and r["strategy"] == strategy and r["lift"] == "absolute")
            d = next(r for r in results if r["K"] == K and r["strategy"] == strategy and r["lift"] == "deviation")
            impr = (a["lpips"] - d["lpips"]) / a["lpips"] * 100
            print(f"  K={K:.2f} {strategy:12s} | LPIPS abs={a['lpips']:.4f} "
                  f"dev={d['lpips']:.4f} | deviation better by {impr:+.1f}%")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
