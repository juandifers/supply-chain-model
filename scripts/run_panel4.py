#!/usr/bin/env python
"""
Panel 4 — Budget Frontier

Purpose: diminishing returns from expedite budget.

Grid:
  expedite_budget: [0, 1000, 5000, 20000, 50000]
  × same two representative regimes as Panels 2 and 3

Fixed: shock_prob=0.15, shock_magnitude=0.70, recovery_rate=1.05
Seeds: 10

Usage:
    python scripts/run_panel4.py --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM
    python scripts/run_panel4.py --dry-run
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

PANEL_NAME = "panel4"

# ── Panel 4 grid ──────────────────────────────────────────────────────────────
EXPEDITE_BUDGETS = [0, 1_000, 5_000, 20_000, 50_000]
FIXED_SP         = 0.15
FIXED_SM         = 0.70
FIXED_RR         = 1.05

DEFAULT_REGIME_A = dict(
    default_supply=100, firm_shock_fraction=0.3, expedite_budget=0,
    label="regime_A",
)
DEFAULT_REGIME_B = dict(
    default_supply=100, firm_shock_fraction=1.0, expedite_budget=50_000,
    label="regime_B",
)


def _regime_id(label, eb):
    return f"{label}_eb{eb}"


def build_experiment_list(args, regime_a: dict, regime_b: dict) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for base_regime in [regime_a, regime_b]:
        label = base_regime["label"]
        for eb in EXPEDITE_BUDGETS:
            cfg = dict(
                DEFAULTS,
                default_supply      = base_regime["default_supply"],
                firm_shock_fraction = base_regime["firm_shock_fraction"],
                expedite_budget     = eb,
                shock_prob          = FIXED_SP,
                shock_magnitude     = FIXED_SM,
                recovery_rate       = FIXED_RR,
            )
            rid = _regime_id(label, eb)

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


def compute_budget_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute policy_gain_pct and mip/gi ratio across expedite_budget levels.
    Returns a summary DataFrame for the budget frontier plot.
    """
    if "expedite_budget" not in df.columns:
        return pd.DataFrame()

    rows = []
    for label in ["regime_A", "regime_B"]:
        label_df = df[df["regime_id"].str.startswith(label)].copy()
        if label_df.empty:
            continue
        for eb in sorted(label_df["expedite_budget"].unique()):
            eb_df = label_df[label_df["expedite_budget"] == eb]
            for pol in eb_df["policy"].unique():
                pol_df = eb_df[eb_df["policy"] == pol]
                pct_mean = pol_df["policy_gain_pct"].mean() if "policy_gain_pct" in pol_df.columns else np.nan
                pct_std  = pol_df["policy_gain_pct"].std()  if "policy_gain_pct" in pol_df.columns else np.nan
                rows.append(dict(
                    regime_label   = label,
                    expedite_budget = eb,
                    policy         = pol,
                    policy_gain_pct_mean = pct_mean,
                    policy_gain_pct_std  = pct_std,
                    n_seeds        = len(pol_df),
                ))

    frontier = pd.DataFrame(rows)

    # Add mip_gain / gi_gain ratio column
    if not frontier.empty:
        gi_lookup = (
            frontier[frontier["policy"] == "graph_informed"]
            .set_index(["regime_label", "expedite_budget"])["policy_gain_pct_mean"]
        )
        mip_lookup = (
            frontier[frontier["policy"] == "mip"]
            .set_index(["regime_label", "expedite_budget"])["policy_gain_pct_mean"]
        )
        def _ratio(row):
            if row["policy"] != "mip":
                return np.nan
            key = (row["regime_label"], row["expedite_budget"])
            mip_g = mip_lookup.get(key, np.nan)
            gi_g  = gi_lookup.get(key, np.nan)
            if np.isnan(gi_g) or gi_g == 0:
                return np.nan
            return mip_g / gi_g
        frontier["mip_gain_over_gi_ratio"] = frontier.apply(_ratio, axis=1)

    return frontier


def main():
    parser = argparse.ArgumentParser(description="Panel 4 — Budget Frontier")
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
        print(f"[Panel 4] Dry run — {len(experiments)} total experiments")
        print(f"  expedite_budget levels: {EXPEDITE_BUDGETS}")
        print(f"  Seeds: {args.seeds}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        return

    # ── Resume / overwrite ────────────────────────────────────────────────────
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    if args.resume:
        print(f"[Panel 4] Resuming — {len(completed)} runs already completed")
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 4] Running {len(remaining)} experiments ({len(completed)} skipped)")

    if not remaining:
        print("[Panel 4] Nothing to run.")
    else:
        t_start = time.time()
        bar_fmt = "[Panel 4] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        def _on_result(result, e):
            save_result(result, output_path)
            desc = (f"[Panel 4] regime={e['regime_id'][:18]}  "
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

        print(f"\n[Panel 4] Complete in {(time.time()-t_start)/60:.1f} min → {output_path}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if not output_path.exists():
        print("[Panel 4] No results — skipping analysis.")
        return

    print("[Panel 4] Computing delta metrics and budget frontier...")
    df_raw   = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)

    frontier = compute_budget_frontier(df_delta)
    if not frontier.empty:
        frontier.to_csv(panel_dir / "budget_frontier.csv", index=False)
        print(f"  Budget frontier saved → {panel_dir}/budget_frontier.csv")
        print(frontier[frontier["policy"].isin(["graph_informed", "mip"])].to_string(index=False))

    print(f"\n[Panel 4] Done. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/analyze_and_plot.py --output-dir {out_root}")
    print(f"  Then: python scripts/generate_report.py  --output-dir {out_root}")


if __name__ == "__main__":
    main()
