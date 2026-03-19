"""
Analyze experiment results and generate plots.

Usage:
    python scripts/analyze_results.py --experiment-dir artifacts/experiments/smoke_test_v2
    python scripts/analyze_results.py --experiment-dir artifacts/experiments/smoke_test_v2 --output-dir artifacts/figures
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_summary(experiment_dir: str) -> pd.DataFrame:
    path = os.path.join(experiment_dir, "summary.csv")
    if not os.path.exists(path):
        path = os.path.join(experiment_dir, "sweep_summary.csv")
    return pd.read_csv(path)


def load_kpi_histories(experiment_dir: str) -> Dict[str, pd.DataFrame]:
    kpi_dir = os.path.join(experiment_dir, "kpi_histories")
    if not os.path.isdir(kpi_dir):
        return {}
    result = {}
    for fname in sorted(os.listdir(kpi_dir)):
        if fname.endswith(".csv"):
            key = fname.replace(".csv", "")
            result[key] = pd.read_csv(os.path.join(kpi_dir, fname))
    return result


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_aggregate_stats(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, 95% CI for each metric per policy."""
    metrics = ["backlog_auc", "peak_backlog", "total_expedite_spend",
               "total_reroutes", "final_fill_rate", "mean_fill_rate", "time_to_recovery"]
    available = [m for m in metrics if m in summary.columns]

    rows = []
    for policy in sorted(summary["policy_name"].unique()):
        policy_data = summary[summary["policy_name"] == policy]
        n = len(policy_data)
        row = {"policy_name": policy, "n_seeds": n}
        for m in available:
            vals = policy_data[m].dropna()
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std())
            ci95 = 1.96 * std / np.sqrt(max(len(vals), 1))
            row[f"{m}_mean"] = round(mean, 2)
            row[f"{m}_std"] = round(std, 2)
            row[f"{m}_ci95"] = round(ci95, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_deltas_vs_baseline(summary: pd.DataFrame, baseline: str = "no_intervention") -> pd.DataFrame:
    """Compute improvement of each policy vs no-intervention baseline."""
    baseline_data = summary[summary["policy_name"] == baseline]
    if len(baseline_data) == 0:
        return pd.DataFrame()

    baseline_means = baseline_data[["backlog_auc", "peak_backlog"]].mean()
    rows = []
    for policy in sorted(summary["policy_name"].unique()):
        if policy == baseline:
            continue
        policy_data = summary[summary["policy_name"] == policy]
        policy_means = policy_data[["backlog_auc", "peak_backlog"]].mean()

        backlog_delta_pct = (baseline_means["backlog_auc"] - policy_means["backlog_auc"]) / max(baseline_means["backlog_auc"], 1) * 100
        peak_delta_pct = (baseline_means["peak_backlog"] - policy_means["peak_backlog"]) / max(baseline_means["peak_backlog"], 1) * 100

        # Paired test (Wilcoxon if enough samples, else t-test)
        seeds = sorted(set(baseline_data["seed"]) & set(policy_data["seed"]))
        if len(seeds) >= 3:
            bl_vals = baseline_data.set_index("seed").loc[seeds, "backlog_auc"].values
            po_vals = policy_data.set_index("seed").loc[seeds, "backlog_auc"].values
            diffs = bl_vals - po_vals
            if np.all(diffs == 0):
                p_value = 1.0
            elif len(seeds) >= 6:
                _, p_value = stats.wilcoxon(diffs)
            else:
                _, p_value = stats.ttest_rel(bl_vals, po_vals)
        else:
            p_value = None

        rows.append({
            "policy_name": policy,
            "backlog_auc_reduction_pct": round(backlog_delta_pct, 2),
            "peak_backlog_reduction_pct": round(peak_delta_pct, 2),
            "p_value": round(p_value, 4) if p_value is not None else None,
            "significant_at_005": p_value < 0.05 if p_value is not None else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_backlog_auc_bar(summary: pd.DataFrame, output_path: str):
    """Bar chart: Backlog AUC by policy with error bars."""
    if not HAS_MPL:
        return
    agg = summary.groupby("policy_name")["backlog_auc"].agg(["mean", "std"]).sort_values("mean")
    n_per = summary.groupby("policy_name")["backlog_auc"].count()
    ci95 = 1.96 * agg["std"] / np.sqrt(n_per)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(agg.index, agg["mean"], xerr=ci95, capsize=4, color="steelblue", edgecolor="black")
    ax.set_xlabel("Backlog AUC (lower is better)")
    ax.set_title("Backlog AUC by Policy")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_backlog_timeseries(kpi_histories: Dict[str, pd.DataFrame], output_path: str):
    """Time series: Backlog trajectory averaged across seeds, per policy."""
    if not HAS_MPL:
        return

    # Group by policy (strip _seedN suffix)
    policy_series = {}
    for key, df in kpi_histories.items():
        parts = key.rsplit("_seed", 1)
        policy_name = parts[0]
        if policy_name not in policy_series:
            policy_series[policy_name] = []
        policy_series[policy_name].append(df.set_index("t")["consumer_backlog_units"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for policy_name, series_list in sorted(policy_series.items()):
        combined = pd.concat(series_list, axis=1)
        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        ax.plot(mean.index, mean, label=policy_name, linewidth=1.5)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Consumer Backlog (units)")
    ax.set_title("Backlog Trajectory by Policy (mean ± 1 std)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto(summary: pd.DataFrame, output_path: str, x_col: str = "backlog_auc",
                y_col: str = "total_expedite_spend"):
    """Scatter: Backlog AUC vs Spend per policy."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for policy in sorted(summary["policy_name"].unique()):
        data = summary[summary["policy_name"] == policy]
        ax.scatter(data[x_col], data[y_col], label=policy, s=40, alpha=0.7)
        # Draw mean point
        ax.scatter(data[x_col].mean(), data[y_col].mean(), marker="X", s=120,
                   edgecolors="black", linewidths=1, zorder=5)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(f"{x_col} vs {y_col}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_sweep(sweep_summary: pd.DataFrame, output_path: str):
    """Parameter sweep curves: Backlog AUC vs swept parameter."""
    if not HAS_MPL or "sweep_param" not in sweep_summary.columns:
        return

    param_name = sweep_summary["sweep_param"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6))

    for policy in sorted(sweep_summary["policy_name"].unique()):
        data = sweep_summary[sweep_summary["policy_name"] == policy]
        agg = data.groupby("sweep_value")["backlog_auc"].agg(["mean", "std"])
        n_per = data.groupby("sweep_value")["backlog_auc"].count()
        ci95 = 1.96 * agg["std"] / np.sqrt(n_per)
        ax.errorbar(agg.index, agg["mean"], yerr=ci95, label=policy,
                    capsize=3, marker="o", markersize=4)

    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel("Backlog AUC")
    ax.set_title(f"Backlog AUC vs {param_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment output dir")
    parser.add_argument("--output-dir", default=None, help="Where to save figures (default: artifacts/figures)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(ROOT, "artifacts", "figures")
    os.makedirs(output_dir, exist_ok=True)

    summary = load_summary(args.experiment_dir)
    kpi_histories = load_kpi_histories(args.experiment_dir)

    # Aggregate stats
    print("=== AGGREGATE STATISTICS ===")
    agg = compute_aggregate_stats(summary)
    print(agg.to_string(index=False))

    # Deltas vs baseline
    print("\n=== IMPROVEMENT vs NO_INTERVENTION ===")
    deltas = compute_deltas_vs_baseline(summary)
    if len(deltas) > 0:
        print(deltas.to_string(index=False))

    # Save analysis CSVs
    agg.to_csv(os.path.join(output_dir, "aggregate_stats.csv"), index=False)
    if len(deltas) > 0:
        deltas.to_csv(os.path.join(output_dir, "deltas_vs_baseline.csv"), index=False)

    # Generate plots
    if HAS_MPL:
        plot_backlog_auc_bar(summary, os.path.join(output_dir, "backlog_auc_bar.png"))
        if kpi_histories:
            plot_backlog_timeseries(kpi_histories, os.path.join(output_dir, "backlog_timeseries.png"))
        plot_pareto(summary, os.path.join(output_dir, "pareto_backlog_vs_spend.png"))

        if "sweep_param" in summary.columns:
            plot_sweep(summary, os.path.join(output_dir, "sweep_curves.png"))
    else:
        print("\nmatplotlib not available — skipping plots")


if __name__ == "__main__":
    main()
