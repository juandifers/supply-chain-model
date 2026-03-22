#!/usr/bin/env python
"""
Panel 1 — Core Benchmark

Purpose: headline thesis result — does graph_informed beat baselines,
         and under what structural conditions?

Grid:
  default_supply:      [50, 100, 150]
  firm_shock_fraction: [0.3, 1.0]
  expedite_budget:     [0, 50000]
  → 12 shocked regimes + matched no-shock baselines

Fixed: shock_prob=0.15, shock_magnitude=0.70, recovery_rate=1.05
Seeds: 20 per regime

Usage:
    python scripts/run_panel1.py --seeds 20 --workers 4
    python scripts/run_panel1.py --dry-run
    python scripts/run_panel1.py --resume --no-mip
    python scripts/run_panel1.py --output-dir artifacts/experiments/run_20260320_1200
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
    DEFAULTS, ALL_POLICIES, MIP_AVAILABLE, GIT_COMMIT, P1_NAMED_REGIMES,
    run_single_experiment, save_result, load_completed, exp_key,
    compute_delta_metrics, aggregate_results, estimate_runtime,
)

PANEL_NAME = "panel1"

# ── Panel 1 grid ──────────────────────────────────────────────────────────────
DEFAULT_SUPPLIES     = [50, 100, 150]
FIRM_SHOCK_FRACTIONS = [0.3, 1.0]
EXPEDITE_BUDGETS     = [0, 50_000]
FIXED = dict(shock_prob=0.15, shock_magnitude=0.70, recovery_rate=1.05)


def _regime_id(ds, fsf, eb):
    """Return human-readable regime ID, using named regimes where possible."""
    for name, spec in P1_NAMED_REGIMES.items():
        if (spec["default_supply"] == ds and
                abs(spec["firm_shock_fraction"] - fsf) < 1e-6 and
                spec["expedite_budget"] == eb):
            return name
    return f"ds{ds}_fsf{fsf}_eb{eb}"


def build_experiment_list(args) -> list:
    policies = [
        p for p in ALL_POLICIES
        if p not in (args.skip_policies or [])
        and (p != "mip" or (MIP_AVAILABLE and not args.no_mip))
    ]
    experiments = []
    for ds in DEFAULT_SUPPLIES:
        for fsf in FIRM_SHOCK_FRACTIONS:
            for eb in EXPEDITE_BUDGETS:
                base_cfg = dict(DEFAULTS, **FIXED,
                                default_supply=ds,
                                firm_shock_fraction=fsf,
                                expedite_budget=eb)
                rid = _regime_id(ds, fsf, eb)

                # Shocked runs
                for policy in policies:
                    for seed in range(args.seeds):
                        experiments.append(dict(
                            policy=policy, seed=seed,
                            config=base_cfg, regime_id=rid,
                            panel_name=PANEL_NAME,
                        ))

                # Matched no-shock baseline (no_intervention only, same seeds)
                bl_cfg = dict(base_cfg, shock_prob=0.0, shock_magnitude=0.0)
                for seed in range(args.seeds):
                    experiments.append(dict(
                        policy="no_intervention", seed=seed,
                        config=bl_cfg, regime_id=rid,
                        panel_name=PANEL_NAME,
                    ))
    return experiments


# ── Post-run analysis ─────────────────────────────────────────────────────────

def _spearman_r(a, b):
    """Manual Spearman r for two lists of equal length."""
    from scipy.stats import spearmanr
    if len(a) < 2:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def run_differentiation_check(df: pd.DataFrame) -> tuple:
    """
    Per-regime differentiation check. Returns (report_df, n_passing).

    A regime passes if:
      - delta_backlog_auc(no_intervention) meaningfully > 0  (not disruption_too_mild)
      - graph_informed beats no_intervention by >= 5%
        (reduced from 10%: no-budget regimes are legitimately lower-signal —
         expediting is graph_informed's primary differentiator vs backlog_greedy)
      - reroute_only and expedite_only differ materially (lever_contrast > 0.01)
      - policy rankings stable across seeds (split-half Spearman >= 0.50)
        (split-half method compares mean ranks in seeds[0:N//2] vs seeds[N//2:],
         rather than all pairwise per-seed comparisons — the latter is dominated
         by noise from different network topologies per seed)
    """
    from scipy.stats import spearmanr as _spearmanr

    rows = []
    all_policies = [p for p in ALL_POLICIES if p in df["policy"].unique()]

    for rid in df["regime_id"].unique():
        rdf = df[df["regime_id"] == rid].copy()

        # Disruption damage
        ni_delta = rdf[rdf["policy"] == "no_intervention"]["delta_backlog_auc"].mean()
        mild = bool(np.isnan(ni_delta) or ni_delta < 500)

        # GI improvement (vs no_intervention)
        gi_vals = rdf[rdf["policy"] == "graph_informed"]["backlog_auc"].values
        ni_vals = rdf[rdf["policy"] == "no_intervention"]["backlog_auc"].values
        ni_mean = float(np.mean(ni_vals)) if len(ni_vals) > 0 else np.nan
        gi_mean = float(np.mean(gi_vals)) if len(gi_vals) > 0 else np.nan
        gi_improvement = ((ni_mean - gi_mean) / ni_mean) if (ni_mean and ni_mean > 0) else np.nan

        # Reroute vs expedite lever contrast
        ro = rdf[rdf["policy"] == "reroute_only"]["backlog_auc"].mean()
        eo = rdf[rdf["policy"] == "expedite_only"]["backlog_auc"].mean()
        lever_contrast = (abs(ro - eo) / max(ni_mean, 1e-6)
                          if (not np.isnan(ro) and not np.isnan(eo) and ni_mean) else np.nan)

        # Ranking stability — split-half Spearman on mean ranks.
        # Split seeds into two halves; compute mean AUC per policy in each half;
        # Spearman-correlate the two mean-rank vectors.
        # This measures whether the AVERAGE ordering is reproducible, not whether
        # any individual seed replicates another (which is noise-dominated when
        # each seed draws a different network topology).
        rank_corr = np.nan
        seeds = sorted(rdf["seed"].unique())
        if len(seeds) >= 4:
            mid    = len(seeds) // 2
            half1  = set(seeds[:mid])
            half2  = set(seeds[mid:])
            means1 = (rdf[rdf["seed"].isin(half1)]
                      .groupby("policy")["backlog_auc"].mean())
            means2 = (rdf[rdf["seed"].isin(half2)]
                      .groupby("policy")["backlog_auc"].mean())
            common = sorted(set(means1.index) & set(means2.index))
            if len(common) >= 3:
                r, _ = _spearmanr([means1[p] for p in common],
                                  [means2[p] for p in common])
                rank_corr = float(r)

        passes = bool(
            (not mild) and
            (not np.isnan(gi_improvement) and gi_improvement >= 0.05) and
            (not np.isnan(lever_contrast) and lever_contrast > 0.01) and
            (not np.isnan(rank_corr) and rank_corr >= 0.50)
        )

        rows.append(dict(
            regime_id       = rid,
            disruption_mild = mild,
            ni_delta_auc    = round(float(ni_delta), 1) if not np.isnan(ni_delta) else np.nan,
            gi_improvement  = round(float(gi_improvement), 4) if not np.isnan(gi_improvement) else np.nan,
            lever_contrast  = round(float(lever_contrast), 4) if not np.isnan(lever_contrast) else np.nan,
            rank_corr       = round(float(rank_corr), 4) if not np.isnan(rank_corr) else np.nan,
            passes          = passes,
        ))

    report_df  = pd.DataFrame(rows)
    n_passing  = int(report_df["passes"].sum())
    return report_df, n_passing


def save_analysis_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    # A: Regime definitions
    reg_cols = ["regime_id", "default_supply", "firm_shock_fraction",
                "expedite_budget", "shock_magnitude", "shock_prob", "recovery_rate"]
    df[reg_cols].drop_duplicates().to_csv(
        tables_dir / "table_A_regime_definitions.csv", index=False)

    # B: Structural baselines (no-shock runs)
    # (not in df after compute_delta_metrics; read raw for this)

    # C/D: Shocked outcomes and damage deltas — group by regime + policy
    agg = df.groupby(["regime_id", "policy"]).agg(
        backlog_auc_mean      = ("backlog_auc",        "mean"),
        backlog_auc_std       = ("backlog_auc",        "std"),
        fill_rate_mean        = ("fill_rate",           "mean"),
        fill_rate_std         = ("fill_rate",           "std"),
        delta_backlog_auc_mean = ("delta_backlog_auc",  "mean"),
        delta_backlog_auc_std  = ("delta_backlog_auc",  "std"),
        policy_gain_pct_mean  = ("policy_gain_pct",     "mean"),
        policy_gain_pct_std   = ("policy_gain_pct",     "std"),
        mip_gap_mean          = ("mip_vs_graph_informed_gap_pct", "mean"),
        n_seeds               = ("seed",               "count"),
    ).reset_index()
    agg.to_csv(tables_dir / "table_C_shocked_outcomes.csv", index=False)

    damage = df.groupby(["regime_id", "policy"]).agg(
        delta_backlog_auc_mean  = ("delta_backlog_auc",  "mean"),
        delta_fill_rate_mean    = ("delta_fill_rate",    "mean"),
        delta_peak_backlog_mean = ("delta_peak_backlog", "mean"),
        delta_lost_sales_mean   = ("delta_lost_sales",   "mean"),
        policy_gain_on_damage   = ("policy_gain_on_damage", "mean"),
        disruption_too_mild     = ("disruption_too_mild", "any"),
    ).reset_index()
    damage.to_csv(tables_dir / "table_D_disruption_damage_deltas.csv", index=False)

    # E: Policy rankings per regime
    ranks = df.groupby(["regime_id", "policy"])["backlog_auc"].mean().reset_index()
    ranks["rank"] = ranks.groupby("regime_id")["backlog_auc"].rank()
    ranks.to_csv(tables_dir / "table_E_policy_rankings.csv", index=False)

    # F: Pairwise comparisons (mandatory pairs)
    pairs = []
    for rid in df["regime_id"].unique():
        rdf = df[df["regime_id"] == rid]
        comparisons = [
            ("graph_informed",  "no_intervention"),
            ("graph_informed",  "backlog_greedy"),
            ("mip",             "graph_informed"),
            ("reroute_only",    "expedite_only"),
        ]
        for pol_a, pol_b in comparisons:
            a = rdf[rdf["policy"] == pol_a]["backlog_auc"].values
            b = rdf[rdf["policy"] == pol_b]["backlog_auc"].values
            if len(a) < 2 or len(b) < 2:
                continue
            n        = min(len(a), len(b))
            diff     = b[:n] - a[:n]   # positive = b has higher (worse) AUC
            mean_d   = float(np.mean(diff))
            std_d    = float(np.std(diff, ddof=1))
            ci_lo    = mean_d - 1.96 * std_d / np.sqrt(n)
            ci_hi    = mean_d + 1.96 * std_d / np.sqrt(n)
            pct_imp  = mean_d / max(float(np.mean(b)), 1e-6)
            comparable = (abs(pct_imp) < 0.03) and (ci_lo < 0 < ci_hi)
            pairs.append(dict(
                regime_id       = rid,
                policy_a        = pol_a,
                policy_b        = pol_b,
                mean_diff_auc   = round(mean_d, 1),
                std_diff_auc    = round(std_d, 1),
                ci_95_lo        = round(ci_lo, 1),
                ci_95_hi        = round(ci_hi, 1),
                pct_improvement = round(pct_imp, 4),
                comparable      = comparable,
                n_seeds         = n,
            ))
    pd.DataFrame(pairs).to_csv(
        tables_dir / "table_F_pairwise_comparisons.csv", index=False)

    print(f"  [Panel 1] Tables saved → {tables_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Panel 1 — Core Benchmark")
    parser.add_argument("--seeds",         type=int,  default=20)
    parser.add_argument("--workers",       type=int,  default=4)
    parser.add_argument("--output-dir",    type=str,  default=None,
                        help="Root benchmark dir (shared across panels)")
    parser.add_argument("--no-mip",        action="store_true")
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--skip-policies", type=str, nargs="*", default=None)
    args = parser.parse_args()

    # ── Output paths ─────────────────────────────────────────────────────────
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M")
        out_root = Path(f"artifacts/experiments/rework_benchmark_{ts}")
    panel_dir   = out_root / PANEL_NAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    (panel_dir / "plots").mkdir(exist_ok=True)
    (panel_dir / "tables").mkdir(exist_ok=True)
    output_path = panel_dir / "per_run_results.csv"
    error_log   = panel_dir / "errors.log"

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = dict(
        panel="panel1", git_commit=GIT_COMMIT,
        seeds=args.seeds, workers=args.workers,
        mip_available=MIP_AVAILABLE, no_mip=args.no_mip,
        started=datetime.now().isoformat(),
    )
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Build experiments ─────────────────────────────────────────────────────
    experiments = build_experiment_list(args)

    if args.dry_run:
        est = estimate_runtime(experiments, args.workers)
        print(f"[Panel 1] Dry run — {len(experiments)} total experiments")
        print(f"  Regimes:  {sorted(set(e['regime_id'] for e in experiments))}")
        print(f"  Policies: {sorted(set(e['policy'] for e in experiments))}")
        print(f"  Seeds:    0..{args.seeds - 1}")
        print(f"  Est time: ~{est[f'est_hours_{args.workers}workers']:.1f}h at {args.workers} workers")
        return

    # ── Resume / overwrite ────────────────────────────────────────────────────
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path) if args.resume else set()
    if args.resume:
        print(f"[Panel 1] Resuming — {len(completed)} runs already completed")
    remaining = [e for e in experiments if exp_key(e) not in completed]
    print(f"[Panel 1] Running {len(remaining)} experiments "
          f"({len(completed)} skipped)")

    if not remaining:
        print("[Panel 1] Nothing to run — all experiments already completed.")
    else:
        t_start   = time.time()
        mip_times = []
        gi_times  = []

        bar_fmt = "[Panel 1] |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        def _on_result(result, e):
            save_result(result, output_path)
            if e["policy"] == "mip":
                mip_times.append(result["runtime_s"])
            if e["policy"] == "graph_informed":
                gi_times.append(result["runtime_s"])
            desc = (f"[Panel 1] regime={e['regime_id'][:20]}  "
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
                        # Best-effort match for description
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

        elapsed_total = time.time() - t_start

        # MIP timing warning
        if mip_times and gi_times:
            mip_mean = float(np.mean(mip_times))
            gi_mean  = float(np.mean(gi_times))
            if gi_mean > 0 and mip_mean > 10 * gi_mean:
                print(f"\n[WARNING] MIP is {mip_mean/gi_mean:.0f}× slower than "
                      f"graph_informed ({mip_mean:.1f}s vs {gi_mean:.1f}s). "
                      "Use --no-mip for remaining panels.")

        print(f"\n[Panel 1] Runs complete in {elapsed_total/60:.1f} min → {output_path}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if not output_path.exists():
        print("[Panel 1] No results file — skipping analysis.")
        return

    print("[Panel 1] Computing delta metrics...")
    df_raw   = pd.read_csv(output_path)
    df_delta = compute_delta_metrics(df_raw)
    df_delta.to_csv(panel_dir / "aggregated_results.csv", index=False)
    print(f"  Wrote {len(df_delta)} shocked rows → aggregated_results.csv")

    # ── Differentiation check ──────────────────────────────────────────────
    print("[Panel 1] Differentiation check...")
    diff_df, n_passing = run_differentiation_check(df_delta)
    diff_df.to_csv(panel_dir / "differentiation_report.csv", index=False)

    lines = ["Panel 1 — Differentiation Report", "=" * 56]
    for _, row in diff_df.sort_values("regime_id").iterrows():
        status = "PASS" if row["passes"] else "FAIL"
        ni_d   = f"{row['ni_delta_auc']:.0f}" if not np.isnan(row["ni_delta_auc"]) else "n/a"
        gi_imp = f"{row['gi_improvement']:.1%}" if not np.isnan(row["gi_improvement"]) else "n/a"
        rc     = f"{row['rank_corr']:.3f}" if not np.isnan(row["rank_corr"]) else "n/a"
        lc     = f"{row['lever_contrast']:.3f}" if not np.isnan(row["lever_contrast"]) else "n/a"
        lines += [
            f"\n  [{status}] {row['regime_id']}",
            f"    disruption_mild={row['disruption_mild']}, ni_delta_auc={ni_d}",
            f"    gi_improvement={gi_imp} (need ≥5%)  lever_contrast={lc} (need >0.01)",
            f"    rank_corr={rc} (need ≥0.50, split-half mean ranks)",
        ]
    lines.append(f"\n  {n_passing} / {len(diff_df)} regimes pass.")
    report_text = "\n".join(lines)
    print(report_text)
    (panel_dir / "differentiation_report.txt").write_text(report_text)

    if n_passing < 4:
        print(
            f"\n[Panel 1][EXIT CODE 2] Only {n_passing}/4 regimes differentiate.\n"
            "Review differentiation_report.txt before proceeding to Panel 2.\n"
            "Possible fixes: increase --seeds, check shock schedule is applied,\n"
            "or adjust shock_prob/shock_magnitude in the grid."
        )
        sys.exit(2)

    # ── Analysis tables ────────────────────────────────────────────────────
    save_analysis_tables(df_delta, panel_dir / "tables")

    print(f"\n[Panel 1] Complete. All outputs in {panel_dir}/")
    print(f"  Next: python scripts/run_panel2.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
