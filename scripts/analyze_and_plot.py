#!/usr/bin/env python
"""
Analyze and Plot

Reads all panel outputs and generates thesis-ready plots.
Runnable after any subset of panels — skips missing panels gracefully.

Produces:
  Plot 1 — Heatmap: graph_informed improvement (Panel 1)
  Plot 2 — Bar chart: all 7 policies for 4 named regimes (Panel 1)
  Plot 3 — Delta damage chart (Panel 1)
  Plot 4 — Mechanism: shock_magnitude and recovery_rate effects (Panel 2)
  Plot 5 — Robustness: shock_prob vs policy_gain_pct (Panel 3)
  Plot 6 — Budget frontier (Panel 4)

Usage:
    python scripts/analyze_and_plot.py --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM
"""
import argparse
import os
import sys
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
from matplotlib.gridspec import GridSpec

# ── Plotting style ────────────────────────────────────────────────────────────
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

NAMED_REGIME_ORDER = [
    "tight_local_nobudget",
    "tight_systemic_budget",
    "loose_local_nobudget",
    "loose_systemic_budget",
]


def _load_panel(out_root: Path, panel: str) -> pd.DataFrame | None:
    f = out_root / panel / "aggregated_results.csv"
    if not f.exists():
        print(f"  [SKIP] {panel}: {f} not found")
        return None
    df = pd.read_csv(f)
    # Normalise bool column
    if "is_baseline_run" in df.columns:
        df["is_baseline_run"] = df["is_baseline_run"].map(
            lambda x: str(x).lower() in ("true", "1", "yes")
        )
    print(f"  [LOAD] {panel}: {len(df)} rows")
    return df


def _policy_order(df: pd.DataFrame) -> list:
    from scripts.experiment_utils import ALL_POLICIES
    present = set(df["policy"].unique()) if "policy" in df.columns else set()
    return [p for p in ALL_POLICIES if p in present]


# ── Plot 1: Heatmap ───────────────────────────────────────────────────────────

def plot1_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Heatmap of graph_informed improvement vs no_intervention.
    Axes: default_supply × firm_shock_fraction
    Separate subplots for expedite_budget=0 and =50000.
    Second pair of subplots: MIP improvement and MIP vs GI gap.
    """
    if "policy_gain_pct" not in df.columns:
        print("  [SKIP plot1] policy_gain_pct not in data")
        return

    budgets = sorted(df["expedite_budget"].unique()) if "expedite_budget" in df.columns else [None]
    n_cols  = max(len(budgets), 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9), squeeze=False)
    fig.suptitle("Policy Gain vs No-Intervention (%)", fontsize=13, fontweight="bold")

    def _make_pivot(sub_df, policy, metric):
        pol_df = sub_df[sub_df["policy"] == policy].copy()
        if pol_df.empty or "default_supply" not in pol_df or "firm_shock_fraction" not in pol_df:
            return None
        return pol_df.pivot_table(
            index="default_supply", columns="firm_shock_fraction",
            values=metric, aggfunc="mean"
        )

    for col_i, eb in enumerate(budgets):
        eb_df = df[df["expedite_budget"] == eb] if eb is not None else df
        eb_label = f"budget={int(eb):,}" if eb is not None else ""

        for row_i, (policy, title_suffix) in enumerate([
            ("graph_informed", "Graph Informed"),
            ("mip",            "MIP"),
        ]):
            ax  = axes[row_i][col_i]
            piv = _make_pivot(eb_df, policy, "policy_gain_pct")
            if piv is None or piv.empty:
                ax.set_title(f"{title_suffix}\n{eb_label}\n(no data)")
                ax.axis("off")
                continue
            im = ax.imshow(
                piv.values * 100,
                cmap="RdYlGn", aspect="auto",
                vmin=-5, vmax=25,
            )
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"fsf={c:.1f}" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"ds={r}" for r in piv.index])
            for i in range(len(piv.index)):
                for j in range(len(piv.columns)):
                    val = piv.values[i, j]
                    ax.text(j, i, f"{val*100:.1f}%",
                            ha="center", va="center", fontsize=8,
                            color="black" if abs(val) < 0.15 else "white")
            plt.colorbar(im, ax=ax, label="policy gain (%)")
            ax.set_title(f"{title_suffix}\n{eb_label}")
            ax.set_xlabel("firm_shock_fraction")
            ax.set_ylabel("default_supply")

    plt.tight_layout()
    out = plots_dir / "plot1_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 2: Bar chart (4 named regimes) ───────────────────────────────────────

def plot2_bar_chart(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Bar chart of all policies for the 4 named regimes.
    Error bars = 1 std across seeds. Secondary panel: policy_gain_pct.
    """
    named = df[df["regime_id"].isin(NAMED_REGIME_ORDER)].copy()
    if named.empty:
        print("  [SKIP plot2] No named regime data found")
        return

    policies    = _policy_order(named)
    n_regimes   = len(NAMED_REGIME_ORDER)
    fig, axes   = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Panel 1 — Core Benchmark (4 Named Regimes)", fontsize=13, fontweight="bold")

    x         = np.arange(n_regimes)
    bar_width  = 0.8 / max(len(policies), 1)

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("backlog_auc",     "Backlog AUC",       "Backlog AUC by Policy"),
        ("policy_gain_pct", "Policy Gain (%)",   "Policy Gain vs No-Intervention"),
    ]):
        ax = axes[ax_idx]
        for pol_i, pol in enumerate(policies):
            means, stds = [], []
            for rid in NAMED_REGIME_ORDER:
                sub = named[(named["regime_id"] == rid) & (named["policy"] == pol)]
                means.append(sub[metric].mean() if not sub.empty and metric in sub else np.nan)
                stds.append(sub[metric].std()  if not sub.empty and metric in sub else 0)
            offset = (pol_i - len(policies) / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                label=POLICY_LABELS.get(pol, pol),
                color=POLICY_COLORS.get(pol, "#888888"),
                yerr=stds, capsize=3, error_kw={"elinewidth": 0.8},
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in NAMED_REGIME_ORDER],
            fontsize=8
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ax_idx == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=8)
        if metric == "policy_gain_pct":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    out = plots_dir / "plot2_bar_named_regimes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 3: Delta damage chart ────────────────────────────────────────────────

def plot3_delta_damage(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Fraction of disruption-induced damage removed by each policy.
    Grouped bars for 4 named regimes; MIP as visible upper bound.
    """
    if "policy_gain_on_damage" not in df.columns:
        print("  [SKIP plot3] policy_gain_on_damage not in data")
        return
    named = df[df["regime_id"].isin(NAMED_REGIME_ORDER)].copy()
    if named.empty:
        print("  [SKIP plot3] No named regime data")
        return

    policies   = [p for p in _policy_order(named) if p != "no_intervention"]
    n_regimes  = len(NAMED_REGIME_ORDER)
    fig, ax    = plt.subplots(figsize=(12, 5))
    x          = np.arange(n_regimes)
    bar_width  = 0.8 / max(len(policies), 1)

    for pol_i, pol in enumerate(policies):
        means = []
        for rid in NAMED_REGIME_ORDER:
            sub = named[(named["regime_id"] == rid) & (named["policy"] == pol)]
            means.append(sub["policy_gain_on_damage"].mean()
                         if not sub.empty else np.nan)
        offset = (pol_i - len(policies) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            label=POLICY_LABELS.get(pol, pol),
            color=POLICY_COLORS.get(pol, "#888888"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in NAMED_REGIME_ORDER], fontsize=8)
    ax.set_ylabel("Fraction of Disruption Damage Removed")
    ax.set_title("Panel 1 — Disruption Damage Removed by Each Policy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    out = plots_dir / "plot3_delta_damage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 4: Mechanism (Panel 2) ───────────────────────────────────────────────

def plot4_mechanism(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    shock_magnitude vs policy_gain_pct and recovery_rate vs policy_gain_pct.
    One line per policy, per regime (A and B).
    """
    if "policy_gain_pct" not in df.columns:
        print("  [SKIP plot4] policy_gain_pct not in data")
        return
    if "shock_magnitude" not in df.columns or "recovery_rate" not in df.columns:
        print("  [SKIP plot4] shock_magnitude / recovery_rate columns missing")
        return

    policies   = [p for p in _policy_order(df) if p != "no_intervention"]
    regime_labels = sorted(set(
        rid.split("_sm")[0].split("_sp")[0].split("_eb")[0]
        for rid in df["regime_id"].unique()
    ))

    fig, axes = plt.subplots(
        len(regime_labels), 2,
        figsize=(12, 5 * len(regime_labels)),
        squeeze=False,
    )
    fig.suptitle("Panel 2 — Mechanism Analysis", fontsize=13, fontweight="bold")

    for row_i, label in enumerate(regime_labels):
        label_df = df[df["regime_id"].str.startswith(label)].copy()
        if label_df.empty:
            continue

        for col_i, (x_col, x_label) in enumerate([
            ("shock_magnitude", "Shock Magnitude"),
            ("recovery_rate",   "Recovery Rate"),
        ]):
            ax   = axes[row_i][col_i]
            xval = sorted(label_df[x_col].unique())
            for pol in policies:
                pol_df = label_df[label_df["policy"] == pol]
                if pol_df.empty:
                    continue
                means = [pol_df[pol_df[x_col] == x]["policy_gain_pct"].mean() for x in xval]
                ax.plot(
                    xval, [m * 100 if not np.isnan(m) else np.nan for m in means],
                    marker="o", linewidth=2,
                    color=POLICY_COLORS.get(pol, "#888888"),
                    label=POLICY_LABELS.get(pol, pol),
                )

                # Annotate MIP vs GI gap
                if pol == "mip" and "mip_vs_graph_informed_gap_pct" in label_df.columns:
                    for x_val in xval:
                        gap_mean = label_df[
                            (label_df["policy"] == "mip") & (label_df[x_col] == x_val)
                        ]["mip_vs_graph_informed_gap_pct"].mean()
                        if not np.isnan(gap_mean):
                            ax.annotate(
                                f"gap={gap_mean*100:.1f}%",
                                xy=(x_val, means[xval.index(x_val)] * 100),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=7, color=POLICY_COLORS["mip"],
                            )

            ax.set_xlabel(x_label)
            ax.set_ylabel("Policy Gain (%)")
            ax.set_title(f"{label} — {x_label}")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    out = plots_dir / "plot4_mechanism.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 5: Robustness (Panel 3) ──────────────────────────────────────────────

def plot5_robustness(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    shock_prob vs policy_gain_pct per regime. Highlight rank changes.
    """
    if "policy_gain_pct" not in df.columns:
        print("  [SKIP plot5] policy_gain_pct not in data")
        return
    if "shock_prob" not in df.columns:
        print("  [SKIP plot5] shock_prob column missing")
        return

    policies      = [p for p in _policy_order(df) if p != "no_intervention"]
    regime_labels = sorted(set(
        rid.split("_sp")[0] for rid in df["regime_id"].unique()
    ))

    fig, axes = plt.subplots(1, len(regime_labels),
                             figsize=(7 * len(regime_labels), 5),
                             squeeze=False)
    fig.suptitle("Panel 3 — Robustness: Ranking Stability Across Disruption Frequency",
                 fontsize=13, fontweight="bold")

    for col_i, label in enumerate(regime_labels):
        ax       = axes[0][col_i]
        label_df = df[df["regime_id"].str.startswith(label)].copy()
        if label_df.empty:
            ax.set_title(f"{label}\n(no data)")
            continue
        xvals = sorted(label_df["shock_prob"].unique())

        for pol in policies:
            means = [
                label_df[(label_df["policy"] == pol) &
                         (label_df["shock_prob"] == sp)]["policy_gain_pct"].mean()
                for sp in xvals
            ]
            ax.plot(
                xvals, [m * 100 if not np.isnan(m) else np.nan for m in means],
                marker="o", linewidth=2,
                color=POLICY_COLORS.get(pol, "#888888"),
                label=POLICY_LABELS.get(pol, pol),
            )

        # Highlight rank changes between adjacent shock_prob levels
        for i in range(len(xvals) - 1):
            sp1, sp2 = xvals[i], xvals[i + 1]
            ranks1 = {
                pol: label_df[(label_df["policy"] == pol) &
                              (label_df["shock_prob"] == sp1)]["backlog_auc"].mean()
                for pol in policies
            }
            ranks2 = {
                pol: label_df[(label_df["policy"] == pol) &
                              (label_df["shock_prob"] == sp2)]["backlog_auc"].mean()
                for pol in policies
            }
            # Rank by ascending AUC (lower is better)
            r1 = {p: sorted(ranks1, key=ranks1.get).index(p) for p in ranks1}
            r2 = {p: sorted(ranks2, key=ranks2.get).index(p) for p in ranks2}
            changed = [p for p in policies if abs(r1.get(p, 0) - r2.get(p, 0)) >= 2]
            if changed:
                ax.axvline(
                    (sp1 + sp2) / 2, color="red", linewidth=1,
                    linestyle=":", alpha=0.6,
                )
                ax.text(
                    (sp1 + sp2) / 2, ax.get_ylim()[1] * 0.95,
                    "rank\nchange", color="red", fontsize=7, ha="center",
                )

        ax.set_xlabel("Shock Probability")
        ax.set_ylabel("Policy Gain (%)")
        ax.set_title(f"{label}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = plots_dir / "plot5_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Plot 6: Budget frontier (Panel 4) ─────────────────────────────────────────

def plot6_budget_frontier(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    expedite_budget vs policy_gain_pct. Secondary axis: mip_gain / gi_gain ratio.
    """
    if "policy_gain_pct" not in df.columns:
        print("  [SKIP plot6] policy_gain_pct not in data")
        return
    if "expedite_budget" not in df.columns:
        print("  [SKIP plot6] expedite_budget column missing")
        return

    # Try to load pre-computed frontier
    frontier_path = Path(str(plots_dir).replace("/plots", "")) / "budget_frontier.csv"
    if frontier_path.exists():
        frontier = pd.read_csv(frontier_path)
    else:
        # Recompute from df
        rows = []
        for label in sorted(set(
            rid.split("_eb")[0] for rid in df["regime_id"].unique()
        )):
            label_df = df[df["regime_id"].str.startswith(label)]
            for eb in sorted(label_df["expedite_budget"].unique()):
                for pol in label_df["policy"].unique():
                    sub = label_df[(label_df["expedite_budget"] == eb) &
                                   (label_df["policy"] == pol)]
                    rows.append(dict(
                        regime_label=label, expedite_budget=eb, policy=pol,
                        policy_gain_pct_mean=sub["policy_gain_pct"].mean(),
                        policy_gain_pct_std=sub["policy_gain_pct"].std(),
                    ))
        frontier = pd.DataFrame(rows)

    if frontier.empty:
        print("  [SKIP plot6] empty frontier data")
        return

    regime_labels = sorted(frontier["regime_label"].unique()) if "regime_label" in frontier.columns else []
    if not regime_labels:
        print("  [SKIP plot6] no regime_label column")
        return

    fig, axes = plt.subplots(1, len(regime_labels),
                             figsize=(8 * len(regime_labels), 5),
                             squeeze=False)
    fig.suptitle("Panel 4 — Budget Frontier", fontsize=13, fontweight="bold")

    for col_i, label in enumerate(regime_labels):
        ax    = axes[0][col_i]
        ax2   = ax.twinx()
        ldf   = frontier[frontier["regime_label"] == label]
        xvals = sorted(ldf["expedite_budget"].unique())

        for pol in [p for p in [
            "no_intervention", "random_reroute", "reroute_only", "expedite_only",
            "backlog_greedy", "graph_informed", "mip"
        ] if p in ldf["policy"].unique()]:
            means = [
                ldf[(ldf["policy"] == pol) & (ldf["expedite_budget"] == eb)]["policy_gain_pct_mean"].mean()
                for eb in xvals
            ]
            ax.plot(
                xvals, [m * 100 if not np.isnan(m) else np.nan for m in means],
                marker="o", linewidth=2,
                color=POLICY_COLORS.get(pol, "#888888"),
                label=POLICY_LABELS.get(pol, pol),
            )

        # Secondary axis: mip/gi ratio
        if "mip_gain_over_gi_ratio" in ldf.columns:
            mip_ldf = ldf[ldf["policy"] == "mip"].copy()
            if not mip_ldf.empty:
                ratio_x = [eb for eb in xvals if not mip_ldf[
                    mip_ldf["expedite_budget"] == eb]["mip_gain_over_gi_ratio"].isna().all()
                ]
                ratio_y = [float(mip_ldf[mip_ldf["expedite_budget"] == eb]["mip_gain_over_gi_ratio"].mean())
                           for eb in ratio_x]
                ax2.plot(ratio_x, ratio_y, "k--", linewidth=1.2, alpha=0.6,
                         label="MIP/GI ratio (right)")
                ax2.set_ylabel("MIP gain / GI gain ratio", color="gray", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_xlabel("Expedite Budget")
        ax.set_ylabel("Policy Gain (%)")
        ax.set_title(f"Budget Frontier — {label}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out = plots_dir / "plot6_budget_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot all panels")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root benchmark directory (contains panel1/, panel2/, ...)")
    args = parser.parse_args()

    out_root  = Path(args.output_dir)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {out_root}/")
    df1 = _load_panel(out_root, "panel1")
    df2 = _load_panel(out_root, "panel2")
    df3 = _load_panel(out_root, "panel3")
    df4 = _load_panel(out_root, "panel4")

    print(f"\nGenerating plots → {plots_dir}/")

    # Panel 1 plots
    if df1 is not None:
        plot1_heatmap(df1, plots_dir)
        plot2_bar_chart(df1, plots_dir)
        plot3_delta_damage(df1, plots_dir)
        # Also copy plots to panel1/plots/
        p1_plots = out_root / "panel1" / "plots"
        p1_plots.mkdir(exist_ok=True)
        for f in plots_dir.glob("plot[123]_*.png"):
            import shutil
            shutil.copy(f, p1_plots / f.name)

    # Panel 2 plots
    if df2 is not None:
        plot4_mechanism(df2, plots_dir)
        p2_plots = out_root / "panel2" / "plots"
        p2_plots.mkdir(exist_ok=True)
        for f in plots_dir.glob("plot4_*.png"):
            import shutil
            shutil.copy(f, p2_plots / f.name)

    # Panel 3 plots
    if df3 is not None:
        plot5_robustness(df3, plots_dir)
        p3_plots = out_root / "panel3" / "plots"
        p3_plots.mkdir(exist_ok=True)
        for f in plots_dir.glob("plot5_*.png"):
            import shutil
            shutil.copy(f, p3_plots / f.name)

    # Panel 4 plots
    if df4 is not None:
        plot6_budget_frontier(df4, plots_dir)
        p4_plots = out_root / "panel4" / "plots"
        p4_plots.mkdir(exist_ok=True)
        for f in plots_dir.glob("plot6_*.png"):
            import shutil
            shutil.copy(f, p4_plots / f.name)

    generated = list(plots_dir.glob("*.png"))
    print(f"\nGenerated {len(generated)} plot(s) in {plots_dir}/")
    for f in sorted(generated):
        print(f"  {f.name}")
    print(f"\nNext: python scripts/generate_report.py --output-dir {out_root}")


if __name__ == "__main__":
    main()
