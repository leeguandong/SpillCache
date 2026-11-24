#!/usr/bin/env python3
"""Aggregate end-to-end lift quality (LPIPS/PSNR/SSIM) across samples.

Reads outputs/lift_quality/*/quality.json and reports, per (K, strategy):
- mean LPIPS/PSNR/SSIM for absolute-lift vs deviation-lift
- deviation-lift win rate (lower LPIPS)
- paired bootstrap 95% CI on LPIPS reduction

Produces the paper's image-level headline table.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/lift_quality"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


def paired_bootstrap(deltas, n_boot, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.root.glob("*/quality.json"))
    if not files:
        raise SystemExit(f"no quality.json under {args.root}")

    samples = {}
    for f in files:
        data = json.loads(f.read_text())
        samples[data["sample_id"]] = data["results"]
    print(f"Loaded {len(samples)} samples: {', '.join(sorted(samples))}\n")

    budgets = sorted({r["K"] for r in next(iter(samples.values()))})
    strategies = sorted({r["strategy"] for r in next(iter(samples.values()))})

    report = {"n_samples": len(samples), "samples": sorted(samples), "table": []}
    header = (f"{'K':>5} {'strategy':>12} | {'LPIPS abs':>9} {'LPIPS dev':>9} {'impr%':>6} "
              f"{'win':>6} | {'PSNR abs':>8} {'PSNR dev':>8} | {'SSIM abs':>8} {'SSIM dev':>8}")
    print(header)
    print("-" * len(header))

    for K in budgets:
        for strategy in strategies:
            rows = []
            for sid, results in samples.items():
                a = next((r for r in results if r["K"] == K and r["strategy"] == strategy and r["lift"] == "absolute"), None)
                d = next((r for r in results if r["K"] == K and r["strategy"] == strategy and r["lift"] == "deviation"), None)
                if a and d and a["lpips"] is not None and d["lpips"] is not None:
                    rows.append((a, d))
            if not rows:
                continue
            n = len(rows)
            la = sum(a["lpips"] for a, _ in rows) / n
            ld = sum(d["lpips"] for _, d in rows) / n
            pa = sum(a["psnr_db"] for a, _ in rows if a["psnr_db"]) / n
            pd = sum(d["psnr_db"] for _, d in rows if d["psnr_db"]) / n
            sa = sum(a["ssim"] for a, _ in rows) / n
            sd = sum(d["ssim"] for _, d in rows) / n
            deltas = [a["lpips"] - d["lpips"] for a, d in rows]
            wins = sum(1 for x in deltas if x > 0)
            lo, hi = paired_bootstrap(deltas, args.bootstrap)
            impr = (la - ld) / la * 100
            print(f"{K:>5.2f} {strategy:>12} | {la:>9.4f} {ld:>9.4f} {impr:>5.1f}% "
                  f"{wins}/{n:>2} | {pa:>8.2f} {pd:>8.2f} | {sa:>8.4f} {sd:>8.4f}")
            report["table"].append({
                "K": K, "strategy": strategy, "n": n,
                "lpips_abs": la, "lpips_dev": ld, "lpips_impr_pct": impr,
                "psnr_abs": pa, "psnr_dev": pd, "ssim_abs": sa, "ssim_dev": sd,
                "win_rate": wins / n, "lpips_reduction_ci95": [lo, hi],
            })

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
