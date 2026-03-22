#!/usr/bin/env python
"""
Panel 3 — Robustness

Purpose: do policy rankings survive changes in disruption frequency?

Grid:
  shock_prob: [0.05, 0.15, 0.25]
  × same two representative regimes as Panel 2

Fixed: shock_magnitude=0.70, recovery_rate=1.05
Seeds: 10 minimum (increase to 20 if rankings unstable)

Usage:
    python scripts/run_panel3.py --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM
    python scripts/run_panel3.py --dry-run --seeds 20
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
    DEFAULTS, ALL_POLICIES, MIP_AVAILABLE, GIT_COMMIT,
    run_single_experiment, save_result, load_completed, exp_key,
    compute_delta_metrics, estimate_runtime,
)

PANEL_NAME = "panel3"

# ── Panel 3 grid ──────────────────────────────────────────────────────────────
SHOCK_PROBS     = [0.05, 0.15, 0.25]
FIXED_SM        = 0.70
FIXED_RR        = 1.05

DEFAULT_REGIME_A = dict(
    default_supply=100, firm_shock_fraction=0.3, expedite_budget=0,
    label="regime_A",
)
DEFAULT_REGIME_B = dict(
    default_supply=100, firm_shock_fraction=1.0, expedite_budget=50_000,
    label="regime_B",
)


def _regime_id(label, sp):
    return f"{label}_sp{sp}"


def build_experiment_list(args, regime_a: dict, regime_b: dict) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for base_regime in [regime_a, regime_b]:
        label = base_regime["label"]
        for sp in SHOCK_PROBS:
            cfg = dict(
                DEFAULTS,
                default_supply      = base_regime["default_supply"],
                firm_shock_fraction = base_regime["firm_shock_fraction"],
                expedite_budget     = base_regime["expedite_budget"],
                shock_prob          = sp,
                shock_magnitude     = FIXED_SM,
                recovery_rate       = FIXED_RR,
            )
            rid = _regime_id(label, sp)

            # Shocked runs
            for policy in policies:
                for seed in range(args.seeds):
                    experiments.append(dict(
                        policy=policy, seed=seed,
                        config=cfg, regime_id=rid,
                        panel_name=PANEL_NAME,
                    ))

            # Matched no-shock baseline
            bl_cfg = dict(cfg, shock_prob=0.0, shock_magnitude=0.0)
            for seed in range(args.seeds):
                experiments.append(dict(
                    policy="no_intervention", seed=seed,
                    config=bl_cfg, regime_id=rid,
                    panel_name=PANEL_NAME,
                ))
    return experiments


def check_ranking_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether policy rankings are stable across shock_prob levels.
    Returns a DataFrame noting any rank changes.
    """
    rows = []
    for label in ["regime_A", "regime_B"]:
        label_df = df[df["regime_id"].str.startswith(label)].copy()
        if label_df.empty:
            continue

        # Compute mean rank per (shock_prob, policy)
        if "shock_prob" not in label_df.columns:
            continue

        rank_tables = {}
        for sp in sorted(label_df["shock_prob"].unique()):
            sp_df = label_df[label_df["shock_prob"] == sp]
            means = sp_df.groupby("policy")["backlog_auc"].mean().sort_values()
            rank_tables[sp] = means.rank().to_dict()

        # Compare adjacent shock_prob levels
        sp_levels = sorted(rank_tables.keys())
        for i in range(len(sp_levels) - 1):
            sp1, sp2 = sp_levels[i], sp_levels[i + 1]
            r1, r2   = rank_tables[sp1], rank_tables[sp2]
            all_pols = sorted(set(r1.keys()) & set(r2.keys()))
            rank_changes = [
                p for p in all_pols
                if abs(r1.get(p, 0) - r2.get(p, 0)) >= 2.0
            ]
            # GI vs BG comparison
            gi_better_sp1 = r1.get("graph_informed", 99) < r1.get("backlog_greedy", 99)
            gi_better_sp2 = r2.get("graph_informed", 99) < r2.get("backlog_greedy", 99)
            gi_advantage_stable = gi_better_sp1 == gi_better_sp2

            rows.append(dict(
                regime_label        = label,
                shock_prob_from     = sp1,
                shock_prob_to       = sp2,
                rank_changes        = ", ".join(rank_changes) if rank_changes else "none",
                n_rank_changes      = len(rank_changes),
                gi_advantage_stable = gi_advantage_stable,
            ))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Panel 3 — Robustness")
    parser.add_argument("--seeds",          type=int,  default=10)
    parser.add_argument("--workers",        type=int,  default=4)
    parser.add_argument("--output-dir",     type=str,  default=None)
    parser.add_argument("--no-mip",         action="store_true")
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--skip-policies",  type=str, nargs="*", default=None)
    parser.add_argument("--regime-a-json",  type=str, default=None)
    parser.add_argument("--regime-b-json",  type=str, default=None)
    args = parser.parse_args()

    def _load_regime(arg, default):
        if arg is None:
            return default
        if os.path.exists(arg):
            with open(arg) as f:
                return json.load(f)
        return json.loads(arg)

    regime_a = _load_regime(args.regime_a_json, DEFAULT_REGIME_A)
    regime_b = _load_regime(args.regime_b_json, DEFAULT_REGIME_B)
    regime_a.setdefault("label", "regime_A")
    regime_b.setdefault("label", "regime_B")

    # ── Output paths ─────────────────────────────────────────────────────────
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M")
        out_root = Path(f"artifacts/experiments/rework_benchmark_{ts}")
    panel_dir   = out_root / PANEL_NAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    (panel_dir / "plots").mkdir(exist_ok=True)
    output_path = panel_dir / "per_run_results.csv"
    error_log   = panel_dir / "errors.log"

    # ── Build experiments ─────────────────────────────────────────────────────
    experiments = build_experiment_list(args, regime_a, regime_b)

    if args.dry_run:
        est = estimate_runtime(experiments, args.workers)
        print(f"[Panel 3] Dry run — {len(experiments)} total experiments")
        print(f"  shock_prob levels: {SHOCK_PROBS}")
        print(f"  Seeds: {args.seeds}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        return

    # ── Resume / overwrite ────────────────────────────────────────────────────
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    if args.resume:
        print(f"[Panel 3] Resuming — {len(completed)} runs already completed")
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 3] Running {len(remaining)} experiments ({len(completed)} skipped)")

    if not remaining:
        print("[Panel 3] Nothing to run.")
    else:
        t_start = time.time()
        bar_fmt = "[Panel 3] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        def _on_result(result, e):
            save_result(result, output_path)
            desc = (f"[Panel 3] regime={e['regime_id'][:18]}  "
                    f"seed={e['seed']}  policy={e['policy']}")
            pf = dict(auc=f"{result['backlog_auc']:.0f}",
                      fill=f"{result['fill_rate']:.2f}",
                      elapsed=f"{result['runtime_s']:.1f}s")
            if e["policy"] == "mip" and result["mean_mip_step_s"] > 0:
                pf["step_solve"] = f"{result['mean_mip_step_s']:.2f}s"
            return desc, pf

        with tqdm(total=len(remaining), bar_format=bar_fmt) as pbar:
            if args.workers == 1:
                for e in remaining:
                    try:
                        result = run_single_experiment(e)
                        desc, pf = _on_result(result, e)
                        pbar.set_description(desc)
                        pbar.set_postfix(pf)
                    except Exception as ex:
                        with open(error_log, "a") as f:
                            f.write(f"FAILED {e['regime_id']} {e['policy']} "
                                    f"seed={e['seed']}: {ex}\n")
                            f.write(traceback.format_exc() + "\n")
                        pbar.set_postfix({"ERROR": str(ex)[:40]})
                    pbar.update(1)
            else:
                with multiprocessing.Pool(args.workers) as pool:
                    for result in pool.imap_unordered(run_single_experiment, remaining):
                        e = next(
                            (x for x in remaining
                             if x["policy"] == result["policy"]
                             and x["seed"] == result["seed"]
                             and x["regime_id"] == result["regime_id"]),
                            remaining[0]
                        )
                        try:
                            desc, pf = _on_result(result, e)
                            pbar.set_description(desc)
                            pbar.set_postfix(pf)
                        except Exception as ex:
                            with open(error_log, "a") as f:
                                f.write(f"SAVE ERROR: {ex}\n")
                        pbar.update(1)

        print(f"\n[Panel 3] Complete in {(time.time()-t_start)/60:.1f} min → {output_path}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if not output_path.exists():
        print("[Panel 3] No results — skipping analysis.")
        return

    print("[Panel 3] Computing delta metrics...")
    df_raw   = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)

    # Ranking stability report
    print("[Panel 3] Checking ranking stability...")
    stab_df = check_ranking_stability(df_delta)
    if not stab_df.empty:
        stab_df.to_csv(panel_dir / "ranking_stability.csv", index=False)
        print(stab_df.to_string(index=False))
        rank_changes = stab_df[stab_df["n_rank_changes"] > 0]
        if not rank_changes.empty:
            print(f"\n  [NOTE] Rank changes detected in {len(rank_changes)} transitions:")
            for _, row in rank_changes.iterrows():
                print(f"    {row['regime_label']} sp={row['shock_prob_from']}→{row['shock_prob_to']}: "
                      f"{row['rank_changes']}")
        gi_unstable = stab_df[~stab_df["gi_advantage_stable"]]
        if not gi_unstable.empty:
            print(f"  [NOTE] graph_informed vs backlog_greedy ranking flips at: "
                  f"{gi_unstable[['regime_label','shock_prob_from','shock_prob_to']].to_string(index=False)}")

    print(f"\n[Panel 3] Done. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/run_panel4.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
