#!/usr/bin/env python
"""
Analyze and Plot V2

Reads all V2 panel outputs and generates thesis-ready plots + report.
Runnable after any subset of panels — skips missing panels gracefully.

Produces:
  Plot 1 — Heatmap: GI separation across V2 regimes (Panel 1)
  Plot 2 — Bar chart: all policies for 8 regimes (Panel 1)
  Plot 3 — Mechanism: magnitude_mean and duration_mean effects (Panel 2)
  Plot 4 — Robustness: shock_prob vs policy_gain_pct (Panel 3)
  Plot 5 — Budget frontier (Panel 4)

Usage:
    python scripts/analyze_and_plot_v2.py --output-dir artifacts/experiments/v2_benchmark_YYYYMMDD_HHMM
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

POLICY_COLORS = {
    "no_intervention": "#bdbdbd",
    "random_reroute":  "#fdae61",
    "reroute_only":    "#2b83ba",
    "expedite_only":   "#abdda4",
    "backlog_greedy":  "#d7191c",
    "graph_informed":  "#1a9641",
    "mip":             "#7b2d8b",
}
POLICY_LABELS = {
    "no_intervention": "No Intervention",
    "random_reroute":  "Random Reroute",
    "reroute_only":    "Reroute Only",
    "expedite_only":   "Expedite Only",
    "backlog_greedy":  "Backlog Greedy",
    "graph_informed":  "Graph Informed",
    "mip":             "MIP",
}

V2_REGIME_ORDER = [
    "tight_local_eb0",     "tight_local_eb50k",
    "tight_regional_eb0",  "tight_regional_eb50k",
    "loose_local_eb0",     "loose_local_eb50k",
    "loose_regional_eb0",  "loose_regional_eb50k",
]


def _load_panel(out_root: Path, panel: str) -> pd.DataFrame | None:
    f = out_root / panel / "aggregated_results.csv"
    if not f.exists():
        print(f"  [SKIP] {panel}: {f} not found")
        return None
    df = pd.read_csv(f)
    if "is_baseline_run" in df.columns:
        df["is_baseline_run"] = df["is_baseline_run"].map(
            lambda x: str(x).lower() in ("true", "1", "yes"))
    print(f"  [LOAD] {panel}: {len(df)} rows")
    return df


def _policy_order(df):
    from scripts.experiment_utils import ALL_POLICIES
    present = set(df["policy"].unique()) if "policy" in df.columns else set()
    return [p for p in ALL_POLICIES if p in present]


# ── Plot 1: Heatmap ─────────────────────────────────────────────────────────

def plot1_heatmap(df: pd.DataFrame, plots_dir: Path):
    if "policy_gain_pct" not in df.columns:
        return

    regimes = [r for r in V2_REGIME_ORDER if r in df["regime_id"].unique()]
    if not regimes:
        return

    policies = ["graph_informed", "backlog_greedy", "reroute_only", "expedite_only"]
    policies = [p for p in policies if p in df["policy"].unique()]

    fig, ax = plt.subplots(figsize=(max(8, len(regimes) * 0.9), len(policies) * 1.2 + 1))
    fig.suptitle("Policy Gain vs No-Intervention (%)", fontsize=13, fontweight="bold")

    data = np.full((len(policies), len(regimes)), np.nan)
    for i, pol in enumerate(policies):
        for j, rid in enumerate(regimes):
            sub = df[(df["policy"] == pol) & (df["regime_id"] == rid)]
            if not sub.empty:
                data[i, j] = sub["policy_gain_pct"].mean() * 100

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=25)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=7)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([POLICY_LABELS.get(p, p) for p in policies])

    for i in range(len(policies)):
        for j in range(len(regimes)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < 15 else "white")

    plt.colorbar(im, ax=ax, label="policy gain (%)")
    plt.tight_layout()
    out = plots_dir / "plot1_v2_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 2: Bar chart ───────────────────────────────────────────────────────

def plot2_bar_chart(df: pd.DataFrame, plots_dir: Path):
    regimes = [r for r in V2_REGIME_ORDER if r in df["regime_id"].unique()]
    if not regimes:
        return

    policies = _policy_order(df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Panel 1 V2 — Core Benchmark (8 Regimes)", fontsize=13, fontweight="bold")

    x = np.arange(len(regimes))
    bar_width = 0.8 / max(len(policies), 1)

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("backlog_auc",     "Backlog AUC",     "Backlog AUC by Policy"),
        ("policy_gain_pct", "Policy Gain (%)", "Policy Gain vs No-Intervention"),
    ]):
        ax = axes[ax_idx]
        for pol_i, pol in enumerate(policies):
            means, stds = [], []
            for rid in regimes:
                sub = df[(df["regime_id"] == rid) & (df["policy"] == pol)]
                means.append(sub[metric].mean() if not sub.empty and metric in sub else np.nan)
                stds.append(sub[metric].std() if not sub.empty and metric in sub else 0)
            offset = (pol_i - len(policies) / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width,
                   label=POLICY_LABELS.get(pol, pol),
                   color=POLICY_COLORS.get(pol, "#888"),
                   yerr=stds, capsize=2, error_kw={"elinewidth": 0.6},
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ax_idx == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=7)
        if metric == "policy_gain_pct":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    out = plots_dir / "plot2_v2_bar_regimes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 3: Mechanism (Panel 2) ─────────────────────────────────────────────

def plot3_mechanism(df: pd.DataFrame, plots_dir: Path):
    if "policy_gain_pct" not in df.columns:
        return

    policies = [p for p in _policy_order(df) if p != "no_intervention"]
    regime_labels = sorted(set(
        rid.rsplit("_mm", 1)[0] for rid in df["regime_id"].unique() if "_mm" in rid
    ))
    if not regime_labels:
        return

    fig, axes = plt.subplots(len(regime_labels), 2,
                             figsize=(12, 5 * len(regime_labels)), squeeze=False)
    fig.suptitle("Panel 2 V2 — Mechanism Analysis", fontsize=13, fontweight="bold")

    for row_i, label in enumerate(regime_labels):
        label_df = df[df["regime_id"].str.startswith(label + "_mm")].copy()
        if label_df.empty:
            continue

        # Extract magnitude_mean and duration_mean from regime_id
        import re
        label_df["_mm"] = label_df["regime_id"].apply(
            lambda x: float(re.search(r"mm([\d.]+)", x).group(1)) if re.search(r"mm([\d.]+)", x) else np.nan)
        label_df["_dm"] = label_df["regime_id"].apply(
            lambda x: float(re.search(r"dm([\d.]+)", x).group(1)) if re.search(r"dm([\d.]+)", x) else np.nan)

        for col_i, (x_col, x_label) in enumerate([
            ("_mm", "Magnitude Mean"),
            ("_dm", "Duration Mean"),
        ]):
            ax = axes[row_i][col_i]
            xvals = sorted(label_df[x_col].dropna().unique())
            for pol in policies:
                pol_df = label_df[label_df["policy"] == pol]
                if pol_df.empty:
                    continue
                means = [pol_df[pol_df[x_col] == x]["policy_gain_pct"].mean() for x in xvals]
                ax.plot(xvals, [m * 100 if not np.isnan(m) else np.nan for m in means],
                        marker="o", linewidth=2,
                        color=POLICY_COLORS.get(pol, "#888"),
                        label=POLICY_LABELS.get(pol, pol))

            ax.set_xlabel(x_label)
            ax.set_ylabel("Policy Gain (%)")
            ax.set_title(f"{label} — {x_label}")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    out = plots_dir / "plot3_v2_mechanism.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 4: Robustness (Panel 3) ────────────────────────────────────────────

def plot4_robustness(df: pd.DataFrame, plots_dir: Path):
    if "policy_gain_pct" not in df.columns or "shock_prob" not in df.columns:
        return

    policies = [p for p in _policy_order(df) if p != "no_intervention"]
    regime_labels = sorted(set(
        rid.rsplit("_sp", 1)[0] for rid in df["regime_id"].unique() if "_sp" in rid
    ))
    if not regime_labels:
        return

    fig, axes = plt.subplots(1, len(regime_labels),
                             figsize=(7 * len(regime_labels), 5), squeeze=False)
    fig.suptitle("Panel 3 V2 — Robustness: Ranking Stability", fontsize=13, fontweight="bold")

    for col_i, label in enumerate(regime_labels):
        ax = axes[0][col_i]
        label_df = df[df["regime_id"].str.startswith(label + "_sp")].copy()
        if label_df.empty:
            continue
        xvals = sorted(label_df["shock_prob"].unique())

        for pol in policies:
            means = [
                label_df[(label_df["policy"] == pol) & (label_df["shock_prob"] == sp)]["policy_gain_pct"].mean()
                for sp in xvals
            ]
            ax.plot(xvals, [m * 100 if not np.isnan(m) else np.nan for m in means],
                    marker="o", linewidth=2,
                    color=POLICY_COLORS.get(pol, "#888"),
                    label=POLICY_LABELS.get(pol, pol))

        ax.set_xlabel("Shock Probability")
        ax.set_ylabel("Policy Gain (%)")
        ax.set_title(f"{label}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = plots_dir / "plot4_v2_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 5: Budget frontier (Panel 4) ───────────────────────────────────────

def plot5_budget_frontier(df: pd.DataFrame, plots_dir: Path):
    if "policy_gain_pct" not in df.columns or "expedite_budget" not in df.columns:
        return

    regime_labels = sorted(set(
        rid.rsplit("_eb", 1)[0] for rid in df["regime_id"].unique() if "_eb" in rid
    ))
    if not regime_labels:
        return

    fig, axes = plt.subplots(1, len(regime_labels),
                             figsize=(8 * len(regime_labels), 5), squeeze=False)
    fig.suptitle("Panel 4 V2 — Budget Frontier", fontsize=13, fontweight="bold")

    for col_i, label in enumerate(regime_labels):
        ax = axes[0][col_i]
        label_df = df[df["regime_id"].str.startswith(label + "_eb")].copy()
        if label_df.empty:
            continue
        xvals = sorted(label_df["expedite_budget"].unique())

        for pol in [p for p in _policy_order(df) if p != "no_intervention"]:
            means = [
                label_df[(label_df["policy"] == pol) & (label_df["expedite_budget"] == eb)]["policy_gain_pct"].mean()
                for eb in xvals
            ]
            ax.plot(xvals, [m * 100 if not np.isnan(m) else np.nan for m in means],
                    marker="o", linewidth=2,
                    color=POLICY_COLORS.get(pol, "#888"),
                    label=POLICY_LABELS.get(pol, pol))

        ax.set_xlabel("Expedite Budget")
        ax.set_ylabel("Policy Gain (%)")
        ax.set_title(f"Budget Frontier — {label}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out = plots_dir / "plot5_v2_budget_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(out_root: Path, df1, df2, df3, df4):
    def _pct(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.1f}%"

    def _fmt(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}"

    lines = [
        "# SupplySim V2 Shock Architecture — Experiment Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Architecture: V2 (four-layer shock architecture)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    if df1 is not None:
        gi_gains = df1[df1["policy"] == "graph_informed"].groupby("regime_id")["policy_gain_pct"].mean()
        bg_gains = df1[df1["policy"] == "backlog_greedy"].groupby("regime_id")["policy_gain_pct"].mean()

        best_regime = gi_gains.idxmax() if not gi_gains.empty else "unknown"
        best_gain = gi_gains.max() if not gi_gains.empty else np.nan
        lines.append(f"- **graph_informed** peaks at `{best_regime}` with {_pct(best_gain)} gain vs no-intervention")

        mean_gi = gi_gains.mean()
        mean_bg = bg_gains.mean() if not bg_gains.empty else np.nan
        lines.append(f"- Mean GI gain across regimes: {_pct(mean_gi)} | Mean BG gain: {_pct(mean_bg)}")

        ro_auc = df1[df1["policy"] == "reroute_only"]["backlog_auc"].mean()
        eo_auc = df1[df1["policy"] == "expedite_only"]["backlog_auc"].mean()
        dominant = "rerouting" if ro_auc < eo_auc else "expediting"
        lines.append(f"- Primary lever: **{dominant}** (reroute_only AUC={_fmt(ro_auc)}, expedite_only AUC={_fmt(eo_auc)})")

        # eb=0 separation
        eb0_regimes = [r for r in df1["regime_id"].unique() if "eb0" in r]
        if eb0_regimes:
            eb0_gi = df1[(df1["policy"] == "graph_informed") & (df1["regime_id"].isin(eb0_regimes))]["policy_gain_pct"].mean()
            lines.append(f"- GI gain at **eb=0** (rerouting only): {_pct(eb0_gi)}")

    lines += ["", "---", ""]

    # Panel 1 detail
    if df1 is not None:
        lines += ["## Panel 1 — Core Benchmark", ""]
        lines.append("| Regime | GI Gain | BG Gain | RO Gain | EO Gain |")
        lines.append("|--------|---------|---------|---------|---------|")
        for rid in V2_REGIME_ORDER:
            if rid not in df1["regime_id"].values:
                continue
            rdf = df1[df1["regime_id"] == rid]
            gi = rdf[rdf["policy"] == "graph_informed"]["policy_gain_pct"].mean()
            bg = rdf[rdf["policy"] == "backlog_greedy"]["policy_gain_pct"].mean()
            ro = rdf[rdf["policy"] == "reroute_only"]["policy_gain_pct"].mean()
            eo = rdf[rdf["policy"] == "expedite_only"]["policy_gain_pct"].mean()
            lines.append(f"| {rid} | {_pct(gi)} | {_pct(bg)} | {_pct(ro)} | {_pct(eo)} |")
        lines += ["", "---", ""]

    # Panel 2 detail
    if df2 is not None:
        lines += ["## Panel 2 — Mechanism Analysis", ""]
        lines.append("See `panel2_v2/aggregated_results.csv` and `plots/plot3_v2_mechanism.png`")
        lines += ["", "---", ""]

    # Panel 3 detail
    if df3 is not None:
        lines += ["## Panel 3 — Robustness", ""]
        stab_path = out_root / "panel3_v2" / "ranking_stability.csv"
        if stab_path.exists():
            stab = pd.read_csv(stab_path)
            lines.append(stab.to_markdown(index=False))
        lines += ["", "---", ""]

    # Panel 4 detail
    if df4 is not None:
        lines += ["## Panel 4 — Budget Frontier", ""]
        frontier_path = out_root / "panel4_v2" / "budget_frontier.csv"
        if frontier_path.exists():
            frontier = pd.read_csv(frontier_path)
            key = frontier[frontier["policy"].isin(["graph_informed", "backlog_greedy"])]
            if not key.empty:
                lines.append(key.to_markdown(index=False))
        lines += ["", "---", ""]

    report_path = out_root / "report_v2.md"
    report_path.write_text("\n".join(lines))
    print(f"  [SAVE] {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot all V2 panels")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading V2 results from {out_root}/")
    df1 = _load_panel(out_root, "panel1_v2")
    df2 = _load_panel(out_root, "panel2_v2")
    df3 = _load_panel(out_root, "panel3_v2")
    df4 = _load_panel(out_root, "panel4_v2")

    print(f"\nGenerating plots → {plots_dir}/")

    if df1 is not None:
        plot1_heatmap(df1, plots_dir)
        plot2_bar_chart(df1, plots_dir)

    if df2 is not None:
        plot3_mechanism(df2, plots_dir)

    if df3 is not None:
        plot4_robustness(df3, plots_dir)

    if df4 is not None:
        plot5_budget_frontier(df4, plots_dir)

    # Generate report
    generate_report(out_root, df1, df2, df3, df4)

    generated = list(plots_dir.glob("*_v2_*.png"))
    print(f"\nGenerated {len(generated)} V2 plot(s)")
    for f in sorted(generated):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
