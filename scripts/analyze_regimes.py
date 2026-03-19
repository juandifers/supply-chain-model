"""
STANDALONE SCRIPT — fast, can be run by Cursor or manually.
Expected runtime: < 30 seconds.

Analyzes regime experiment results and generates figures.

Usage:
    python scripts/analyze_regimes.py \
        --input artifacts/experiments/regime_mapping/all_results.csv \
        --output-dir artifacts/figures/

Produces:
    - artifacts/figures/heatmap_intervention_value.png
    - artifacts/figures/heatmap_graph_signal_value.png
    - artifacts/figures/heatmap_reroute_vs_expedite.png
    - artifacts/figures/policy_ranking_by_regime.png
    - artifacts/figures/algorithm_comparison.png
    - artifacts/experiments/regime_mapping/summary.txt
    - artifacts/experiments/regime_mapping/statistical_tests.csv
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — skipping plots")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_regime_stats(df):
    """Compute per-regime, per-policy aggregate stats."""
    group_cols = ["config_name", "firm_shock_fraction", "shock_prob", "policy"]
    agg = df.groupby(group_cols).agg(
        backlog_auc_mean=("backlog_auc", "mean"),
        backlog_auc_std=("backlog_auc", "std"),
        backlog_auc_n=("backlog_auc", "count"),
        peak_backlog_mean=("peak_backlog", "mean"),
        fill_rate_mean=("final_fill_rate", "mean"),
        spend_mean=("total_expedite_spend", "mean"),
        reroutes_mean=("total_reroutes", "mean"),
    ).reset_index()
    agg["backlog_auc_ci95"] = 1.96 * agg["backlog_auc_std"] / np.sqrt(agg["backlog_auc_n"])
    return agg


def compute_pairwise_deltas(df, baseline="no_intervention"):
    """For each regime, compute improvement of each policy vs baseline."""
    rows = []
    for config_name in df["config_name"].unique():
        config_data = df[df["config_name"] == config_name]
        bl_data = config_data[config_data["policy"] == baseline]
        if len(bl_data) == 0:
            continue
        bl_mean = bl_data["backlog_auc"].mean()

        for policy in config_data["policy"].unique():
            if policy == baseline:
                continue
            pol_data = config_data[config_data["policy"] == policy]
            pol_mean = pol_data["backlog_auc"].mean()
            delta_pct = (bl_mean - pol_mean) / max(bl_mean, 1) * 100

            # Paired test
            seeds = sorted(set(bl_data["seed"]) & set(pol_data["seed"]))
            if len(seeds) >= 3:
                bl_vals = bl_data.set_index("seed").loc[seeds, "backlog_auc"].values
                po_vals = pol_data.set_index("seed").loc[seeds, "backlog_auc"].values
                diffs = bl_vals - po_vals
                if np.all(diffs == 0):
                    p_val = 1.0
                elif len(seeds) >= 6:
                    _, p_val = stats.wilcoxon(diffs)
                else:
                    _, p_val = stats.ttest_rel(bl_vals, po_vals)
            else:
                p_val = None

            fsf = config_data["firm_shock_fraction"].iloc[0]
            sp = config_data["shock_prob"].iloc[0]
            rows.append({
                "config_name": config_name,
                "firm_shock_fraction": fsf,
                "shock_prob": sp,
                "policy": policy,
                "baseline": baseline,
                "baseline_auc_mean": round(bl_mean, 2),
                "policy_auc_mean": round(pol_mean, 2),
                "delta_pct": round(delta_pct, 2),
                "p_value": round(p_val, 4) if p_val is not None else None,
                "significant_005": p_val < 0.05 if p_val is not None else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_heatmap(pivot_data, title, output_path, cmap="RdYlGn", fmt=".1f",
                  vmin=None, vmax=None, center=None):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_data.values, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot_data.columns])
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot_data.index])
    ax.set_xlabel("shock_prob")
    ax.set_ylabel("firm_shock_fraction")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        color="black", fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_intervention_value_heatmap(deltas_df, output_dir):
    """Heatmap: graph_informed AUC reduction vs no_intervention."""
    gi = deltas_df[deltas_df["policy"] == "graph_informed"]
    if len(gi) == 0:
        return
    pivot = gi.pivot_table(values="delta_pct", index="firm_shock_fraction",
                           columns="shock_prob", aggfunc="mean")
    _make_heatmap(pivot, "Intervention Value: graph_informed AUC reduction vs no_intervention (%)",
                  os.path.join(output_dir, "heatmap_intervention_value.png"),
                  cmap="RdYlGn", fmt=".1f")


def plot_graph_signal_value_heatmap(deltas_df, output_dir):
    """Heatmap: graph_informed improvement over backlog_only."""
    gi = deltas_df[deltas_df["policy"] == "graph_informed"]
    bl = deltas_df[deltas_df["policy"] == "backlog_only"]
    if len(gi) == 0 or len(bl) == 0:
        return

    # Compute delta between graph_informed and backlog_only
    merged = gi.merge(bl, on=["config_name", "firm_shock_fraction", "shock_prob"],
                      suffixes=("_gi", "_bl"))
    merged["graph_signal_value"] = merged["delta_pct_gi"] - merged["delta_pct_bl"]

    pivot = merged.pivot_table(values="graph_signal_value", index="firm_shock_fraction",
                               columns="shock_prob", aggfunc="mean")
    _make_heatmap(pivot, "Graph Signal Value: graph_informed improvement over backlog_only (pp)",
                  os.path.join(output_dir, "heatmap_graph_signal_value.png"),
                  cmap="RdYlGn", fmt=".1f")


def plot_reroute_vs_expedite_heatmap(deltas_df, output_dir):
    """Heatmap: reroute_only vs expedite_only relative performance."""
    rr = deltas_df[deltas_df["policy"] == "reroute_only"]
    eo = deltas_df[deltas_df["policy"] == "expedite_only"]
    if len(rr) == 0 or len(eo) == 0:
        return

    merged = rr.merge(eo, on=["config_name", "firm_shock_fraction", "shock_prob"],
                      suffixes=("_rr", "_eo"))
    merged["reroute_advantage"] = merged["delta_pct_rr"] - merged["delta_pct_eo"]

    pivot = merged.pivot_table(values="reroute_advantage", index="firm_shock_fraction",
                               columns="shock_prob", aggfunc="mean")
    _make_heatmap(pivot, "Reroute vs Expedite Advantage (pp, positive = reroute better)",
                  os.path.join(output_dir, "heatmap_reroute_vs_expedite.png"),
                  cmap="RdYlBu", fmt=".1f")


def plot_algorithm_comparison(deltas_df, output_dir):
    """Heatmaps comparing greedy vs MIP and threshold vs graph_informed."""
    # MIP vs Greedy
    gi = deltas_df[deltas_df["policy"] == "graph_informed"]
    mip = deltas_df[deltas_df["policy"] == "mip"]
    if len(gi) > 0 and len(mip) > 0:
        merged = gi.merge(mip, on=["config_name", "firm_shock_fraction", "shock_prob"],
                          suffixes=("_gi", "_mip"))
        merged["mip_advantage"] = merged["delta_pct_mip"] - merged["delta_pct_gi"]
        pivot = merged.pivot_table(values="mip_advantage", index="firm_shock_fraction",
                                   columns="shock_prob", aggfunc="mean")
        _make_heatmap(pivot, "MIP vs Greedy Advantage (pp, positive = MIP better)",
                      os.path.join(output_dir, "heatmap_mip_vs_greedy.png"),
                      cmap="RdYlBu", fmt=".1f")

    # Threshold vs Graph-Informed (optimization premium)
    th = deltas_df[deltas_df["policy"] == "threshold"]
    if len(gi) > 0 and len(th) > 0:
        merged = gi.merge(th, on=["config_name", "firm_shock_fraction", "shock_prob"],
                          suffixes=("_gi", "_th"))
        merged["optimization_premium"] = merged["delta_pct_gi"] - merged["delta_pct_th"]
        pivot = merged.pivot_table(values="optimization_premium", index="firm_shock_fraction",
                                   columns="shock_prob", aggfunc="mean")
        _make_heatmap(pivot, "Optimization Premium: graph_informed over threshold (pp)",
                      os.path.join(output_dir, "heatmap_optimization_premium.png"),
                      cmap="RdYlGn", fmt=".1f")


def plot_policy_ranking_lines(df, output_dir):
    """Line plots: X=shock parameter, Y=mean AUC, one line per policy."""
    if not HAS_MPL:
        return

    for x_param, x_label in [("firm_shock_fraction", "Firm Shock Fraction"),
                              ("shock_prob", "Shock Probability")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy in sorted(df["policy"].unique()):
            pol_data = df[df["policy"] == policy]
            agg = pol_data.groupby(x_param)["backlog_auc"].agg(["mean", "std"])
            n_per = pol_data.groupby(x_param)["backlog_auc"].count()
            ci95 = 1.96 * agg["std"] / np.sqrt(n_per)
            ax.errorbar(agg.index, agg["mean"], yerr=ci95, label=policy,
                        capsize=3, marker="o", markersize=4)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean Backlog AUC (lower is better)")
        ax.set_title(f"Policy Ranking by {x_label}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"policy_ranking_by_{x_param}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(output_dir, fname)}")


def plot_bar_overall(df, output_dir):
    """Overall bar chart: mean AUC across all regimes."""
    if not HAS_MPL:
        return
    agg = df.groupby("policy")["backlog_auc"].agg(["mean", "std"]).sort_values("mean")
    n_per = df.groupby("policy")["backlog_auc"].count()
    ci95 = 1.96 * agg["std"] / np.sqrt(n_per)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(agg.index, agg["mean"], xerr=ci95, capsize=4, color="steelblue", edgecolor="black")
    ax.set_xlabel("Backlog AUC (lower is better)")
    ax.set_title("Overall Backlog AUC by Policy (all regimes)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_overall_auc.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'bar_overall_auc.png')}")


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def write_summary(df, deltas_df, regime_stats, output_path):
    """Write human-readable summary."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("REGIME MAPPING EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Overall stats
        f.write("OVERALL POLICY COMPARISON (all regimes averaged)\n")
        f.write("-" * 60 + "\n")
        overall = df.groupby("policy")["backlog_auc"].agg(["mean", "std", "count"])
        overall["ci95"] = 1.96 * overall["std"] / np.sqrt(overall["count"])
        overall = overall.sort_values("mean")
        f.write(overall.to_string() + "\n\n")

        # Best policy per regime
        f.write("BEST POLICY PER REGIME\n")
        f.write("-" * 60 + "\n")
        for config in sorted(df["config_name"].unique()):
            config_data = df[df["config_name"] == config]
            best = config_data.groupby("policy")["backlog_auc"].mean().idxmin()
            best_auc = config_data.groupby("policy")["backlog_auc"].mean().min()
            f.write(f"  {config:25s}: {best:20s} (AUC={best_auc:.0f})\n")
        f.write("\n")

        # Significant results
        if len(deltas_df) > 0 and "significant_005" in deltas_df.columns:
            sig = deltas_df[deltas_df["significant_005"] == True]
            f.write(f"STATISTICALLY SIGNIFICANT IMPROVEMENTS (p<0.05): {len(sig)} / {len(deltas_df)}\n")
            f.write("-" * 60 + "\n")
            if len(sig) > 0:
                for _, row in sig.sort_values("delta_pct", ascending=False).head(20).iterrows():
                    f.write(f"  {row['config_name']:25s} {row['policy']:20s}: {row['delta_pct']:+.1f}% (p={row['p_value']:.4f})\n")
            f.write("\n")

        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 60 + "\n")

        # Graph signal value
        if "graph_informed" in df["policy"].unique() and "backlog_only" in df["policy"].unique():
            gi_mean = df[df["policy"] == "graph_informed"]["backlog_auc"].mean()
            bl_mean = df[df["policy"] == "backlog_only"]["backlog_auc"].mean()
            gsv = (bl_mean - gi_mean) / max(bl_mean, 1) * 100
            f.write(f"  Graph signal value (gi vs backlog_only): {gsv:+.1f}%\n")

        # MIP vs greedy
        if "mip" in df["policy"].unique():
            mip_mean = df[df["policy"] == "mip"]["backlog_auc"].mean()
            gi_mean = df[df["policy"] == "graph_informed"]["backlog_auc"].mean()
            mv = (gi_mean - mip_mean) / max(gi_mean, 1) * 100
            f.write(f"  MIP vs greedy: {mv:+.1f}% (positive = MIP better)\n")

        # Optimization premium
        if "threshold" in df["policy"].unique():
            th_mean = df[df["policy"] == "threshold"]["backlog_auc"].mean()
            gi_mean = df[df["policy"] == "graph_informed"]["backlog_auc"].mean()
            op = (th_mean - gi_mean) / max(th_mean, 1) * 100
            f.write(f"  Optimization premium (gi vs threshold): {op:+.1f}%\n")

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze regime experiment results")
    parser.add_argument("--input", required=True, help="Path to all_results.csv")
    parser.add_argument("--output-dir", default=None, help="Where to save figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} results from {args.input}")
    print(f"Policies: {sorted(df['policy'].unique())}")
    print(f"Configs: {sorted(df['config_name'].unique())}")

    output_dir = args.output_dir or os.path.join(ROOT, "artifacts", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Compute stats
    regime_stats = compute_regime_stats(df)
    deltas_df = compute_pairwise_deltas(df)

    # Save stats
    input_dir = str(Path(args.input).parent)
    regime_stats.to_csv(os.path.join(input_dir, "regime_stats.csv"), index=False)
    if len(deltas_df) > 0:
        deltas_df.to_csv(os.path.join(input_dir, "statistical_tests.csv"), index=False)

    # Generate figures
    if HAS_MPL:
        plot_bar_overall(df, output_dir)
        plot_policy_ranking_lines(df, output_dir)
        plot_intervention_value_heatmap(deltas_df, output_dir)
        plot_graph_signal_value_heatmap(deltas_df, output_dir)
        plot_reroute_vs_expedite_heatmap(deltas_df, output_dir)
        plot_algorithm_comparison(deltas_df, output_dir)

    # Write summary
    write_summary(df, deltas_df, regime_stats, os.path.join(input_dir, "summary.txt"))

    print(f"\nDone! Figures in {output_dir}")


if __name__ == "__main__":
    main()
