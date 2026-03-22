#!/usr/bin/env python
"""
Panel 3 V2 — Robustness (Shock Architecture)

Purpose: do policy rankings survive changes in disruption frequency?

Grid:
  shock_prob: [0.10, 0.20, 0.30]
  × 2 representative regimes

Usage:
    python scripts/run_panel3_v2.py --seeds 20 --workers 4
    python scripts/run_panel3_v2.py --output-dir artifacts/experiments/v2_benchmark_YYYYMMDD_HHMM
    python scripts/run_panel3_v2.py --dry-run
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

PANEL_NAME = "panel3_v2"

SHOCK_PROBS = [0.10, 0.20, 0.30]

REGIME_A = dict(default_supply=100, event_type="local",    expedite_budget=0,      label="local_eb0")
REGIME_B = dict(default_supply=100, event_type="regional", expedite_budget=50_000, label="regional_eb50k")


def _regime_id(label, sp):
    return f"{label}_sp{sp}"


def build_experiment_list(args) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for base in [REGIME_A, REGIME_B]:
        label = base["label"]
        for sp in SHOCK_PROBS:
            cfg = dict(V2_DEFAULTS, **base, shock_prob=sp)
            rid = _regime_id(label, sp)

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


def check_ranking_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["local_eb0", "regional_eb50k"]:
        label_df = df[df["regime_id"].str.startswith(label)].copy()
        if label_df.empty or "shock_prob" not in label_df.columns:
            continue

        rank_tables = {}
        for sp in sorted(label_df["shock_prob"].unique()):
            sp_df = label_df[label_df["shock_prob"] == sp]
            means = sp_df.groupby("policy")["backlog_auc"].mean().sort_values()
            rank_tables[sp] = means.rank().to_dict()

        sp_levels = sorted(rank_tables.keys())
        for i in range(len(sp_levels) - 1):
            sp1, sp2 = sp_levels[i], sp_levels[i + 1]
            r1, r2 = rank_tables[sp1], rank_tables[sp2]
            all_pols = sorted(set(r1.keys()) & set(r2.keys()))
            rank_changes = [p for p in all_pols if abs(r1.get(p, 0) - r2.get(p, 0)) >= 2.0]
            gi_better_sp1 = r1.get("graph_informed", 99) < r1.get("backlog_greedy", 99)
            gi_better_sp2 = r2.get("graph_informed", 99) < r2.get("backlog_greedy", 99)

            rows.append(dict(
                regime_label=label,
                shock_prob_from=sp1,
                shock_prob_to=sp2,
                rank_changes=", ".join(rank_changes) if rank_changes else "none",
                n_rank_changes=len(rank_changes),
                gi_advantage_stable=gi_better_sp1 == gi_better_sp2,
            ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Panel 3 V2 — Robustness (Shock Architecture)")
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
        print(f"[Panel 3 V2] Dry run — {len(experiments)} total experiments")
        print(f"  shock_prob: {SHOCK_PROBS}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        return

    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 3 V2] Running {len(remaining)} experiments ({len(completed)} skipped)")

    if remaining:
        t_start = time.time()
        bar_fmt = "[Panel 3 V2] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

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

        print(f"\n[Panel 3 V2] Complete in {(time.time()-t_start)/60:.1f} min")

    if not output_path.exists():
        return

    print("[Panel 3 V2] Computing delta metrics...")
    df_raw = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)

    print("[Panel 3 V2] Checking ranking stability...")
    stab_df = check_ranking_stability(df_delta)
    if not stab_df.empty:
        stab_df.to_csv(panel_dir / "ranking_stability.csv", index=False)
        print(stab_df.to_string(index=False))

    print(f"\n[Panel 3 V2] Done. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/run_panel4_v2.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
