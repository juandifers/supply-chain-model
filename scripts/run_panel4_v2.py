#!/usr/bin/env python
"""
Panel 4 V2 — Budget Frontier (Shock Architecture)

Purpose: diminishing returns from expedite budget.

Grid:
  expedite_budget: [0, 5000, 20000, 50000, 100000]
  × 2 representative regimes × 2 event types = 4 base configs

Usage:
    python scripts/run_panel4_v2.py --seeds 20 --workers 4
    python scripts/run_panel4_v2.py --output-dir artifacts/experiments/v2_benchmark_YYYYMMDD_HHMM
    python scripts/run_panel4_v2.py --dry-run
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

PANEL_NAME = "panel4_v2"

EXPEDITE_BUDGETS = [0, 5_000, 20_000, 50_000, 100_000]

REGIME_A = dict(default_supply=100, event_type="local",    label="local")
REGIME_B = dict(default_supply=100, event_type="regional", label="regional")


def _regime_id(label, eb):
    return f"{label}_eb{eb}"


def build_experiment_list(args) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for base in [REGIME_A, REGIME_B]:
        label = base["label"]
        for eb in EXPEDITE_BUDGETS:
            cfg = dict(V2_DEFAULTS, **base, expedite_budget=eb)
            rid = _regime_id(label, eb)

            for policy in policies:
                for seed in range(args.seeds):
                    experiments.append(dict(
                        policy=policy, seed=seed,
                        config=cfg, regime_id=rid,
                        panel_name=PANEL_NAME,
                    ))

            bl_cfg = dict(cfg, is_baseline_run=True)
            for seed in range(args.seeds):
                experiments.append(dict(
                    policy="no_intervention", seed=seed,
                    config=bl_cfg, regime_id=rid,
                    panel_name=PANEL_NAME,
                ))
    return experiments


def compute_budget_frontier(df: pd.DataFrame) -> pd.DataFrame:
    if "expedite_budget" not in df.columns:
        return pd.DataFrame()

    rows = []
    for label in ["local", "regional"]:
        label_df = df[df["regime_id"].str.startswith(label)].copy()
        if label_df.empty:
            continue
        for eb in sorted(label_df["expedite_budget"].unique()):
            eb_df = label_df[label_df["expedite_budget"] == eb]
            for pol in eb_df["policy"].unique():
                pol_df = eb_df[eb_df["policy"] == pol]
                pct_mean = pol_df["policy_gain_pct"].mean() if "policy_gain_pct" in pol_df.columns else np.nan
                pct_std = pol_df["policy_gain_pct"].std() if "policy_gain_pct" in pol_df.columns else np.nan
                rows.append(dict(
                    regime_label=label,
                    expedite_budget=eb,
                    policy=pol,
                    policy_gain_pct_mean=pct_mean,
                    policy_gain_pct_std=pct_std,
                    n_seeds=len(pol_df),
                ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Panel 4 V2 — Budget Frontier (Shock Architecture)")
    parser.add_argument("--seeds",          type=int,  default=10)
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
        print(f"[Panel 4 V2] Dry run — {len(experiments)} total experiments")
        print(f"  expedite_budget: {EXPEDITE_BUDGETS}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        return

    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 4 V2] Running {len(remaining)} experiments ({len(completed)} skipped)")

    if remaining:
        t_start = time.time()
        bar_fmt = "[Panel 4 V2] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        with tqdm(total=len(remaining), bar_format=bar_fmt) as pbar:
            if args.workers == 1:
                for e in remaining:
                    try:
                        result = run_single_experiment_v2(e)
                        save_result(result, output_path)
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
                        except Exception as ex:
                            with open(error_log, "a") as f:
                                f.write(f"SAVE ERROR: {ex}\n")
                        pbar.update(1)

        print(f"\n[Panel 4 V2] Complete in {(time.time()-t_start)/60:.1f} min")

    if not output_path.exists():
        return

    print("[Panel 4 V2] Computing delta metrics and budget frontier...")
    df_raw = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)

    frontier = compute_budget_frontier(df_delta)
    if not frontier.empty:
        frontier.to_csv(panel_dir / "budget_frontier.csv", index=False)
        print(frontier[frontier["policy"].isin(["graph_informed", "backlog_greedy"])].to_string(index=False))

    print(f"\n[Panel 4 V2] Done. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/analyze_and_plot_v2.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
