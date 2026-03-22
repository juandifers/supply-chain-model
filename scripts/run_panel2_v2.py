#!/usr/bin/env python
"""
Panel 2 V2 — Mechanism Analysis (Shock Architecture)

Purpose: understand *why* policies win — shock severity and duration effects.

Grid:
  magnitude_mean: [0.50, 0.70, 0.85]
  duration_mean:  [5, 15, 25]
  × 2 representative regimes = 18 shocked configs + baselines

Two representative regimes:
  Regime A (local, no budget):    ds=100, local shocks, eb=0
  Regime B (regional, budget):    ds=100, regional shocks, eb=50000

Usage:
    python scripts/run_panel2_v2.py --seeds 20 --workers 4
    python scripts/run_panel2_v2.py --output-dir artifacts/experiments/v2_benchmark_YYYYMMDD_HHMM
    python scripts/run_panel2_v2.py --dry-run
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.experiment_utils import (
    V2_DEFAULTS, ALL_POLICIES, MIP_AVAILABLE, GIT_COMMIT,
    run_single_experiment_v2, save_result, load_completed, exp_key,
    compute_delta_metrics, estimate_runtime,
)

PANEL_NAME = "panel2_v2"

MAGNITUDE_MEANS = [0.50, 0.70, 0.85]
DURATION_MEANS  = [5, 15, 25]

REGIME_A = dict(default_supply=100, event_type="local",    expedite_budget=0,      label="local_eb0")
REGIME_B = dict(default_supply=100, event_type="regional", expedite_budget=50_000, label="regional_eb50k")


def _regime_id(label, mm, dm):
    return f"{label}_mm{mm}_dm{dm}"


def build_experiment_list(args) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for base in [REGIME_A, REGIME_B]:
        label = base["label"]
        for mm in MAGNITUDE_MEANS:
            for dm in DURATION_MEANS:
                cfg = dict(V2_DEFAULTS, **base, magnitude_mean=mm, duration_mean=dm)
                rid = _regime_id(label, mm, dm)

                for policy in policies:
                    for seed in range(args.seeds):
                        experiments.append(dict(
                            policy=policy, seed=seed,
                            config=cfg, regime_id=rid,
                            panel_name=PANEL_NAME,
                        ))

                # Matched baseline
                bl_cfg = dict(cfg, is_baseline_run=True)
                for seed in range(args.seeds):
                    experiments.append(dict(
                        policy="no_intervention", seed=seed,
                        config=bl_cfg, regime_id=rid,
                        panel_name=PANEL_NAME,
                    ))
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Panel 2 V2 — Mechanism (Shock Architecture)")
    parser.add_argument("--seeds",          type=int,  default=20)
    parser.add_argument("--workers",        type=int,  default=4)
    parser.add_argument("--output-dir",     type=str,  default=None)
    parser.add_argument("--no-mip",         action="store_true")
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--skip-policies",  type=str, nargs="*", default=None)
    args = parser.parse_args()

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_root = Path(f"artifacts/experiments/v2_benchmark_{ts}")
    panel_dir = out_root / PANEL_NAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    (panel_dir / "plots").mkdir(exist_ok=True)
    output_path = panel_dir / "per_run_results.csv"
    error_log = panel_dir / "errors.log"

    experiments = build_experiment_list(args)

    if args.dry_run:
        est = estimate_runtime(experiments, args.workers)
        print(f"[Panel 2 V2] Dry run — {len(experiments)} total experiments")
        print(f"  magnitude_mean: {MAGNITUDE_MEANS}")
        print(f"  duration_mean:  {DURATION_MEANS}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        return

    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 2 V2] Running {len(remaining)} experiments ({len(completed)} skipped)")

    if remaining:
        t_start = time.time()
        bar_fmt = "[Panel 2 V2] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        with tqdm(total=len(remaining), bar_format=bar_fmt) as pbar:
            if args.workers == 1:
                for e in remaining:
                    try:
                        result = run_single_experiment_v2(e)
                        save_result(result, output_path)
                        pbar.set_postfix(auc=f"{result['backlog_auc']:.0f}", t=f"{result['runtime_s']:.1f}s")
                    except Exception as ex:
                        with open(error_log, "a") as f:
                            f.write(f"FAILED {e['regime_id']} {e['policy']} seed={e['seed']}: {ex}\n")
                            f.write(traceback.format_exc() + "\n")
                    pbar.update(1)
            else:
                with multiprocessing.Pool(args.workers) as pool:
                    for result in pool.imap_unordered(run_single_experiment_v2, remaining):
                        try:
                            save_result(result, output_path)
                            pbar.set_postfix(regime=result["regime_id"][:20], auc=f"{result['backlog_auc']:.0f}")
                        except Exception as ex:
                            with open(error_log, "a") as f:
                                f.write(f"SAVE ERROR: {ex}\n")
                        pbar.update(1)

        print(f"\n[Panel 2 V2] Complete in {(time.time()-t_start)/60:.1f} min")

    if not output_path.exists():
        return

    print("[Panel 2 V2] Computing delta metrics...")
    df_raw = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)
    print(f"  Wrote {len(df_delta)} rows → aggregated_results.csv")
    print(f"\n[Panel 2 V2] Done. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/run_panel3_v2.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
