#!/usr/bin/env python3
"""Aggregate lifting fidelity results across samples with paired statistics.

Reads outputs/lifting_fidelity/*/fidelity.json and reports, per (K, strategy):
- mean relative error for absolute-lift vs deviation-lift
- per-sample win rate for deviation-lift
- paired bootstrap 95% CI on the improvement

This produces the paper table for the deviation-lifting fidelity claim.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/lifting_fidelity"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--exclude-last-step", action="store_true",
                   help="drop the final denoising step (reference/edit fully diverged)")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


def sample_mean_rel(results, K, strategy, exclude_last):
    """Mean relative error per lift method over steps for one sample."""
    steps = [r for r in results if r["K"] == K and r["strategy"] == strategy]
    if exclude_last:
        max_step = max(r["step"] for r in steps)
        steps = [r for r in steps if r["step"] != max_step]
    if not steps:
        return None
    abs_rel = sum(r["absolute_lift"]["rel"] for r in steps) / len(steps)
    dev_rel = sum(r["deviation_lift"]["rel"] for r in steps) / len(steps)
    return abs_rel, dev_rel


def paired_bootstrap(deltas, n_boot, seed=0):
    """Bootstrap 95% CI on the mean of paired deltas (abs - dev)."""
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.root.glob("*/fidelity.json"))
    if not files:
        raise SystemExit(f"no fidelity.json under {args.root}")

    samples = {}
    for f in files:
        data = json.loads(f.read_text())
        samples[data["sample_id"]] = data["results"]
    print(f"Loaded {len(samples)} samples: {', '.join(sorted(samples))}\n")

    budgets = sorted({r["K"] for r in next(iter(samples.values()))})
    strategies = sorted({r["strategy"] for r in next(iter(samples.values()))})

    report = {"n_samples": len(samples), "samples": sorted(samples),
              "exclude_last_step": args.exclude_last_step, "table": []}

    header = f"{'K':>5} {'strategy':>12} {'abs_rel':>9} {'dev_rel':>9} {'impr%':>7} {'win':>7} {'95% CI':>18}"
    print(header)
    print("-" * len(header))
    for K in budgets:
        for strategy in strategies:
            abs_list, dev_list, deltas = [], [], []
            for sid, results in samples.items():
                pair = sample_mean_rel(results, K, strategy, args.exclude_last_step)
                if pair is None:
                    continue
                a, d = pair
                abs_list.append(a); dev_list.append(d); deltas.append(a - d)
            n = len(deltas)
            mean_abs = sum(abs_list) / n
            mean_dev = sum(dev_list) / n
            impr = (mean_abs - mean_dev) / mean_abs * 100
            wins = sum(1 for x in deltas if x > 0)
            lo, hi = paired_bootstrap(deltas, args.bootstrap)
            print(f"{K:>5.2f} {strategy:>12} {mean_abs:>9.4f} {mean_dev:>9.4f} "
                  f"{impr:>6.1f}% {wins}/{n:>3} [{lo:>7.4f},{hi:>7.4f}]")
            report["table"].append({
                "K": K, "strategy": strategy, "n": n,
                "mean_abs_rel": mean_abs, "mean_dev_rel": mean_dev,
                "improvement_pct": impr, "win_rate": wins / n,
                "ci95": [lo, hi],
            })

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
