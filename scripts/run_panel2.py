#!/usr/bin/env python
"""
Panel 2 — Mechanism

Purpose: understand *why* policies win — shock severity and recovery speed.

Grid:
  shock_magnitude: [0.50, 0.70, 0.85]
  recovery_rate:   [1.02, 1.05, 1.25]
  → 9 configs × 2 representative regimes = 18 shocked configs

Two representative regimes (override with --regime-a-json / --regime-b-json):
  Regime A (reroute-friendly):   default_supply=100, firm_shock_fraction=0.3, expedite_budget=0
  Regime B (systemic+budget):    default_supply=100, firm_shock_fraction=1.0, expedite_budget=50000

Seeds: 20 (auto-reduced to 10 if total runtime would exceed 2h)

Usage:
    python scripts/run_panel2.py --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM
    python scripts/run_panel2.py --output-dir <dir> --seeds 10 --no-mip --resume
    python scripts/run_panel2.py --dry-run
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

PANEL_NAME = "panel2"

# ── Panel 2 grid ──────────────────────────────────────────────────────────────
SHOCK_MAGNITUDES = [0.50, 0.70, 0.85]
RECOVERY_RATES   = [1.02, 1.05, 1.25]
FIXED_SP         = 0.15   # shock_prob

# Default representative regimes
DEFAULT_REGIME_A = dict(
    default_supply=100, firm_shock_fraction=0.3, expedite_budget=0,
    label="regime_A",
)
DEFAULT_REGIME_B = dict(
    default_supply=100, firm_shock_fraction=1.0, expedite_budget=50_000,
    label="regime_B",
)


def _regime_id(label, sm, rr):
    return f"{label}_sm{sm}_rr{rr}"


def build_experiment_list(args, regime_a: dict, regime_b: dict) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for base_regime in [regime_a, regime_b]:
        label = base_regime["label"]
        for sm in SHOCK_MAGNITUDES:
            for rr in RECOVERY_RATES:
                cfg = dict(
                    DEFAULTS,
                    default_supply      = base_regime["default_supply"],
                    firm_shock_fraction = base_regime["firm_shock_fraction"],
                    expedite_budget     = base_regime["expedite_budget"],
                    shock_prob          = FIXED_SP,
                    shock_magnitude     = sm,
                    recovery_rate       = rr,
                )
                rid = _regime_id(label, sm, rr)

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


def save_interpretation_md(df: pd.DataFrame, panel_dir: Path) -> None:
    """
    Auto-generate panel2/interpretation.md with placeholder analysis.
    Replace the placeholder paragraphs with actual findings after running.
    """
    lines = [
        "# Panel 2 — Mechanism: Interpretation",
        "",
        "> **Note:** This file is auto-generated from panel2 results.",
        "> Replace placeholder paragraphs with actual findings.",
        "",
    ]

    # Summarise shock_magnitude effect
    agg_sm = (
        df.groupby(["shock_magnitude", "policy"])["policy_gain_pct"]
        .mean().reset_index()
        if "policy_gain_pct" in df.columns else pd.DataFrame()
    )
    lines += [
        "## Plot 4a — shock_magnitude vs policy_gain_pct",
        "",
        "**What the plot shows:** Policy gain (% reduction in backlog AUC vs no_intervention)",
        "as shock_magnitude increases from 0.50 → 0.85.",
        "",
    ]
    if not agg_sm.empty:
        for pol in ["graph_informed", "backlog_greedy", "mip"]:
            sub = agg_sm[agg_sm["policy"] == pol]
            if not sub.empty:
                gains = sub.sort_values("shock_magnitude")["policy_gain_pct"].values
                trend = "increases" if len(gains) > 1 and gains[-1] > gains[0] else "decreases"
                lines.append(
                    f"- **{pol}**: gain {trend} with shock_magnitude "
                    f"({', '.join(f'{g:.1%}' for g in gains)})"
                )
    lines += [
        "",
        "**Interpretation placeholder:** [Fill in after reviewing the plot.]",
        "Key question: Does the graph_informed advantage over backlog_greedy",
        "widen as shock_magnitude increases?",
        "",
    ]

    # Summarise recovery_rate effect
    agg_rr = (
        df.groupby(["recovery_rate", "policy"])["policy_gain_pct"]
        .mean().reset_index()
        if "policy_gain_pct" in df.columns else pd.DataFrame()
    )
    lines += [
        "## Plot 4b — recovery_rate vs policy_gain_pct",
        "",
        "**What the plot shows:** Policy gain as recovery_rate varies from 1.02 → 1.25.",
        "(Higher recovery_rate = faster natural recovery.)",
        "",
    ]
    if not agg_rr.empty:
        for pol in ["graph_informed", "backlog_greedy"]:
            sub = agg_rr[agg_rr["policy"] == pol]
            if not sub.empty:
                gains = sub.sort_values("recovery_rate")["policy_gain_pct"].values
                trend = "decreases" if len(gains) > 1 and gains[-1] < gains[0] else "increases"
                lines.append(
                    f"- **{pol}**: gain {trend} as recovery_rate increases "
                    f"({', '.join(f'{g:.1%}' for g in gains)})"
                )
    lines += [
        "",
        "**Interpretation placeholder:** [Fill in after reviewing the plot.]",
        "Key question: When recovery is slow (rr=1.02), is intervention more valuable?",
        "",
        "## MIP vs graph_informed gap across sweep",
        "",
    ]

    # MIP gap
    if "mip_vs_graph_informed_gap_pct" in df.columns:
        mip_df = df[df["policy"] == "mip"][
            ["shock_magnitude", "recovery_rate", "mip_vs_graph_informed_gap_pct"]
        ]
        if not mip_df.empty:
            mean_gap = float(mip_df["mip_vs_graph_informed_gap_pct"].mean())
            max_gap  = float(mip_df["mip_vs_graph_informed_gap_pct"].max())
            lines += [
                f"- Mean MIP vs graph_informed gap: {mean_gap:.2%}",
                f"- Max gap: {max_gap:.2%}",
                "",
                "**Interpretation placeholder:** In which (shock_magnitude, recovery_rate)",
                "combinations does MIP provide the most extra value over graph_informed?",
            ]

    (panel_dir / "interpretation.md").write_text("\n".join(lines))
    print(f"  [Panel 2] interpretation.md → {panel_dir}/interpretation.md")


def main():
    parser = argparse.ArgumentParser(description="Panel 2 — Mechanism")
    parser.add_argument("--seeds",           type=int,  default=20)
    parser.add_argument("--workers",         type=int,  default=4)
    parser.add_argument("--output-dir",      type=str,  default=None)
    parser.add_argument("--no-mip",          action="store_true")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--overwrite",       action="store_true")
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--skip-policies",   type=str, nargs="*", default=None)
    parser.add_argument("--regime-a-json",   type=str, default=None,
                        help="JSON string or file path overriding Regime A config")
    parser.add_argument("--regime-b-json",   type=str, default=None,
                        help="JSON string or file path overriding Regime B config")
    args = parser.parse_args()

    # ── Load regime overrides ─────────────────────────────────────────────────
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
        print(f"[Panel 2] Dry run — {len(experiments)} total experiments")
        print(f"  Regime A: {regime_a}")
        print(f"  Regime B: {regime_b}")
        print(f"  shock_magnitude: {SHOCK_MAGNITUDES}")
        print(f"  recovery_rate:   {RECOVERY_RATES}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        if est["est_hours_1worker"] > 2.0 and args.seeds > 10:
            print(f"  [HINT] Total estimate > 2h. Consider --seeds 10")
        return

    # Auto-reduce seeds if estimate > 2h
    est = estimate_runtime(experiments, args.workers)
    if est["est_hours_1worker"] / args.workers > 2.0 and args.seeds > 10:
        print(f"[Panel 2] Estimated runtime > 2h — reducing seeds from {args.seeds} to 10.")
        args.seeds = 10
        experiments = build_experiment_list(args, regime_a, regime_b)

    # ── Resume / overwrite ────────────────────────────────────────────────────
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    if args.resume:
        print(f"[Panel 2] Resuming — {len(completed)} runs already completed")
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 2] Running {len(remaining)} experiments ({len(completed)} skipped)")

    if not remaining:
        print("[Panel 2] Nothing to run.")
    else:
        t_start = time.time()
        bar_fmt = "[Panel 2] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        def _on_result(result, e):
            save_result(result, output_path)
            cfg  = e["config"]
            desc = (f"[Panel 2] regime={e['regime_id'][:18]}  "
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

        print(f"\n[Panel 2] Complete in {(time.time()-t_start)/60:.1f} min → {output_path}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if not output_path.exists():
        print("[Panel 2] No results — skipping analysis.")
        return

    print("[Panel 2] Computing delta metrics...")
    df_raw   = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)

    # Annotate with shock_magnitude and recovery_rate (for plotting convenience)
    for col in ["shock_magnitude", "recovery_rate"]:
        if col not in df_delta.columns and col in df_raw.columns:
            df_delta[col] = df_raw[col].values

    save_interpretation_md(df_delta, panel_dir)

    print(f"\n[Panel 2] Done. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/run_panel3.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
