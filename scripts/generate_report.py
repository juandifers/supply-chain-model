#!/usr/bin/env python
"""
Generate Report

Reads all panel outputs and produces report.md with thesis-ready findings.
Runnable after any subset of panels — skips missing sections gracefully.

Usage:
    python scripts/generate_report.py --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM
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


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _pct(v, decimals=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def _fmt(v, decimals=0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:,.{decimals}f}"


def _ci95(values) -> str:
    """Return mean ± 95% CI string."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "n/a"
    mean = np.mean(arr)
    se   = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
    return f"{mean:.1f} ± {1.96 * se:.1f} (n={len(arr)})"


def _comparable(pct_diff: float, ci_lo: float, ci_hi: float) -> bool:
    return abs(pct_diff) < 0.03 and ci_lo < 0 < ci_hi


def _make_policy_summary(df: pd.DataFrame, regime_id: str) -> str:
    """One-line per policy for a given regime."""
    rdf = df[df["regime_id"] == regime_id]
    if rdf.empty:
        return "  (no data)\n"
    lines = []
    for pol in ["no_intervention", "random_reroute", "reroute_only", "expedite_only",
                "backlog_greedy", "graph_informed", "mip"]:
        sub = rdf[rdf["policy"] == pol]
        if sub.empty:
            continue
        auc_mean = sub["backlog_auc"].mean()
        gain_pct = sub["policy_gain_pct"].mean() if "policy_gain_pct" in sub.columns else np.nan
        mip_gap  = sub["mip_vs_graph_informed_gap_pct"].mean() \
            if "mip_vs_graph_informed_gap_pct" in sub.columns and pol == "mip" else np.nan
        note = ""
        if not np.isnan(mip_gap):
            note = f" [MIP vs GI gap: {_pct(mip_gap)}]"
        gain_str = f"+{_pct(gain_pct)}" if not np.isnan(gain_pct) and gain_pct > 0 \
            else (_pct(gain_pct) if not np.isnan(gain_pct) else "n/a")
        lines.append(f"  | {pol:<20} | AUC={_fmt(auc_mean)} | gain={gain_str}{note}")
    return "\n".join(lines) + "\n"


def generate_report(out_root: Path) -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    df1    = _load(out_root / "panel1" / "aggregated_results.csv")
    df2    = _load(out_root / "panel2" / "aggregated_results.csv")
    df3    = _load(out_root / "panel3" / "aggregated_results.csv")
    df4    = _load(out_root / "panel4" / "aggregated_results.csv")
    diff_r = _load(out_root / "panel1" / "differentiation_report.csv")
    pf_f   = _load(out_root / "panel1" / "tables" / "table_F_pairwise_comparisons.csv")
    bfront = _load(out_root / "panel4" / "budget_frontier.csv")

    manifest = {}
    if (out_root / "manifest.json").exists():
        with open(out_root / "manifest.json") as f:
            manifest = json.load(f)

    git_commit  = manifest.get("git_commit", "unknown")
    seeds_p1    = manifest.get("seeds", "?")
    mip_avail   = manifest.get("mip_available", "unknown")
    no_mip      = manifest.get("no_mip", False)
    started     = manifest.get("started", "unknown")

    lines = []

    # ── Executive summary ─────────────────────────────────────────────────────
    lines += [
        "# SupplySim Disruption-Response Experiment Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> Git commit: `{git_commit}`  ",
        f"> Benchmark directory: `{out_root}`",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report summarises a four-panel experiment evaluating seven disruption-response",
        "policies on a reworked supply-chain simulator. The key findings are:",
        "",
    ]

    # Auto-fill summary bullets from data
    summary_bullets = []
    if df1 is not None:
        # Find best regime for GI
        gi_by_regime = df1.groupby("regime_id")["policy_gain_pct"].apply(
            lambda x: x[df1.loc[x.index, "policy"] == "graph_informed"].mean()
            if "policy_gain_pct" in df1.columns else np.nan
        )
        best_regime = gi_by_regime.idxmax() if not gi_by_regime.empty else "unknown"
        best_gain   = gi_by_regime.max() if not gi_by_regime.empty else np.nan
        summary_bullets.append(
            f"1. **graph_informed** achieves its strongest performance in regime "
            f"`{best_regime}` with a mean policy gain of {_pct(best_gain)} vs no-intervention."
        )
        # GI vs BG comparison across all regimes
        gi_auc = df1[df1["policy"] == "graph_informed"]["backlog_auc"].mean()
        bg_auc = df1[df1["policy"] == "backlog_greedy"]["backlog_auc"].mean()
        if not np.isnan(gi_auc) and not np.isnan(bg_auc) and bg_auc > 0:
            gi_vs_bg = (bg_auc - gi_auc) / bg_auc
            summary_bullets.append(
                f"2. **graph_informed vs backlog_greedy**: mean improvement of "
                f"{_pct(gi_vs_bg)} across all Panel 1 regimes."
            )
        # Reroute vs expedite
        ro_auc = df1[df1["policy"] == "reroute_only"]["backlog_auc"].mean()
        eo_auc = df1[df1["policy"] == "expedite_only"]["backlog_auc"].mean()
        dominant = "rerouting" if ro_auc < eo_auc else "expediting"
        summary_bullets.append(
            f"3. **Primary lever**: {dominant} is the dominant driver of improvement "
            f"(reroute_only AUC={_fmt(ro_auc)}, expedite_only AUC={_fmt(eo_auc)})."
        )
        # MIP gap
        if "mip_vs_graph_informed_gap_pct" in df1.columns:
            mip_gap = df1[df1["policy"] == "mip"]["mip_vs_graph_informed_gap_pct"].mean()
            if not np.isnan(mip_gap):
                comparable_note = (
                    " The greedy approach closes the optimality gap in this regime."
                    if abs(mip_gap) < 0.03 else ""
                )
                summary_bullets.append(
                    f"4. **MIP vs graph_informed** mean gap: {_pct(mip_gap)}.{comparable_note}"
                )

    for b in summary_bullets:
        lines.append(b)
        lines.append("")

    lines += [
        "**Best figure for thesis**: `plots/plot2_bar_named_regimes.png`",
        "(Bar chart comparing all 7 policies across 4 interpretable structural regimes.)",
        "",
        "---",
        "",
    ]

    # ── Section 1: What was run ───────────────────────────────────────────────
    lines += [
        "## 1. What Was Run",
        "",
        f"- **Started**: {started}",
        f"- **Git commit**: `{git_commit}`",
        f"- **Seeds (Panel 1)**: {seeds_p1}",
        f"- **MIP available**: {mip_avail}  |  no-mip flag: {no_mip}",
        "",
        "| Panel | Purpose | Status |",
        "|-------|---------|--------|",
    ]
    for panel, purpose in [
        ("panel1", "Core benchmark — 12 regimes × 7 policies × 20 seeds"),
        ("panel2", "Mechanism — shock_magnitude × recovery_rate sweep"),
        ("panel3", "Robustness — shock_prob sweep"),
        ("panel4", "Budget frontier — expedite_budget sweep"),
    ]:
        status = "complete" if _load(out_root / panel / "aggregated_results.csv") is not None else "missing"
        lines.append(f"| {panel} | {purpose} | {status} |")

    lines += ["", "---", ""]

    # ── Section 2: MIP availability ───────────────────────────────────────────
    lines += [
        "## 2. MIP Availability and Performance",
        "",
        f"MIP (pulp/CBC) was {'available and run' if mip_avail and not no_mip else 'NOT run'} "
        f"for this experiment.",
    ]
    if df1 is not None and "mip_fallback_steps" in df1.columns:
        total_fallback = int(df1[df1["policy"] == "mip"]["mip_fallback_steps"].sum())
        if total_fallback > 0:
            lines.append(
                f"\n**Fallback steps**: MIP fell back to no-intervention on "
                f"{total_fallback} step(s) across all Panel 1 runs."
            )
    if df1 is not None and "mean_mip_step_s" in df1.columns:
        mip_step_t = df1[df1["policy"] == "mip"]["mean_mip_step_s"].mean()
        gi_step_t  = df1[df1["policy"] == "graph_informed"]["runtime_s"].mean()
        if not np.isnan(mip_step_t) and not np.isnan(gi_step_t) and gi_step_t > 0:
            ratio = mip_step_t / (gi_step_t / 90)  # approx per-step
            lines.append(
                f"\nMIP mean step solve time: {mip_step_t:.3f}s. "
                f"Approx {ratio:.0f}× slower than graph_informed per episode."
            )
    lines += ["", "---", ""]

    # ── Section 3: Preflight fixes ────────────────────────────────────────────
    lines += [
        "## 3. Preflight Fixes",
        "",
        "_Any fixes applied before running experiments should be noted here._",
        "Run `python scripts/preflight_check.py` to verify no fixes are needed.",
        "",
        "---",
        "",
    ]

    # ── Section 4: Final regime grid ─────────────────────────────────────────
    lines += ["## 4. Final Regime Grid", ""]
    if df1 is not None:
        reg_df = df1[["regime_id", "default_supply", "firm_shock_fraction",
                      "expedite_budget", "shock_magnitude", "shock_prob",
                      "recovery_rate"]].drop_duplicates().sort_values("regime_id")
        lines.append(reg_df.to_markdown(index=False))
    else:
        lines.append("_(Panel 1 not available)_")
    lines += ["", "---", ""]

    # ── Section 5: Structural baseline summary ────────────────────────────────
    lines += ["## 5. Structural Baseline Summary (No-Shock)", ""]
    for df, label in [(df1, "Panel 1"), (df2, "Panel 2"), (df3, "Panel 3"), (df4, "Panel 4")]:
        if df is None:
            continue
        bl_cols = [c for c in ["backlog_auc_base", "fill_rate_base"] if c in df.columns]
        if not bl_cols:
            continue
        lines.append(f"**{label}** — mean structural baseline (no-shock, no-intervention):")
        auc_b  = df["backlog_auc_base"].mean() if "backlog_auc_base" in df.columns else np.nan
        fill_b = df["fill_rate_base"].mean()   if "fill_rate_base"   in df.columns else np.nan
        lines.append(f"  - Backlog AUC (base): {_fmt(auc_b)}")
        lines.append(f"  - Fill rate (base):   {_pct(fill_b)}")
        lines.append("")
    lines += ["---", ""]

    # ── Section 6: Disruption damage summary ─────────────────────────────────
    lines += ["## 6. Disruption Damage Summary", ""]
    if df1 is not None and "delta_backlog_auc" in df1.columns:
        ni_df = df1[df1["policy"] == "no_intervention"].copy()
        if not ni_df.empty:
            lines.append("Mean delta_backlog_auc (no_intervention, shocked vs baseline):")
            lines.append("")
            agg = ni_df.groupby("regime_id")["delta_backlog_auc"].agg(["mean", "std"]).reset_index()
            agg.columns = ["regime_id", "delta_auc_mean", "delta_auc_std"]
            agg["disruption_too_mild"] = agg["delta_auc_mean"] < 500
            lines.append(agg.to_markdown(index=False))
            n_mild = int(agg["disruption_too_mild"].sum())
            if n_mild > 0:
                lines.append(f"\n**{n_mild} regime(s) flagged as disruption_too_mild** "
                              f"(delta_backlog_auc < 500).")
    lines += ["", "---", ""]

    # ── Section 7: Core policy comparison (Panel 1) ───────────────────────────
    lines += ["## 7. Core Policy Comparison (Panel 1)", ""]
    NAMED_REGIMES = [
        "tight_local_nobudget", "tight_systemic_budget",
        "loose_local_nobudget", "loose_systemic_budget",
    ]
    if df1 is not None:
        for rid in NAMED_REGIMES:
            if rid not in df1["regime_id"].values:
                continue
            lines.append(f"### Regime: `{rid}`")
            lines.append("")
            lines.append(_make_policy_summary(df1, rid))

        # Pairwise table
        if pf_f is not None:
            lines.append("### Pairwise Comparisons (mandatory pairs)")
            lines.append("")
            key_pairs = pf_f[pf_f["policy_a"].isin(["graph_informed", "mip"])].copy()
            if not key_pairs.empty:
                lines.append(key_pairs[
                    ["regime_id", "policy_a", "policy_b",
                     "pct_improvement", "comparable", "n_seeds"]
                ].to_markdown(index=False))
    else:
        lines.append("_(Panel 1 not available)_")
    lines += ["", "---", ""]

    # ── Section 8: Mechanism findings (Panel 2) ───────────────────────────────
    lines += ["## 8. Mechanism Findings (Panel 2)", ""]
    if df2 is not None and "policy_gain_pct" in df2.columns:
        # shock_magnitude effect on GI advantage
        gi_bg = df2[df2["policy"].isin(["graph_informed", "backlog_greedy"])].copy()
        if "shock_magnitude" in gi_bg.columns and not gi_bg.empty:
            pivot = gi_bg.pivot_table(
                index="shock_magnitude", columns="policy",
                values="policy_gain_pct", aggfunc="mean"
            )
            lines.append("**Policy gain vs shock_magnitude** (mean over seeds and regimes):")
            lines.append("")
            lines.append(pivot.to_markdown())
            lines.append("")
            # Does GI advantage grow with severity?
            if "graph_informed" in pivot and "backlog_greedy" in pivot:
                gi_vals = pivot["graph_informed"].values
                bg_vals = pivot["backlog_greedy"].values
                gi_grows = len(gi_vals) > 1 and gi_vals[-1] > gi_vals[0]
                bg_grows = len(bg_vals) > 1 and bg_vals[-1] > bg_vals[0]
                diff_low  = gi_vals[0]  - bg_vals[0]  if len(gi_vals) > 0 else np.nan
                diff_high = gi_vals[-1] - bg_vals[-1] if len(gi_vals) > 0 else np.nan
                lines.append(
                    f"graph_informed advantage over backlog_greedy: "
                    f"{_pct(diff_low)} at lowest shock_magnitude, "
                    f"{_pct(diff_high)} at highest — "
                    f"{'widens' if diff_high > diff_low else 'narrows'} with severity."
                )
        # Check interpretation.md
        interp_path = out_root / "panel2" / "interpretation.md"
        if interp_path.exists():
            lines.append("\n_See `panel2/interpretation.md` for plot-level interpretation._")
    else:
        lines.append("_(Panel 2 not available)_")
    lines += ["", "---", ""]

    # ── Section 9: Robustness findings (Panel 3) ──────────────────────────────
    lines += ["## 9. Robustness Findings (Panel 3)", ""]
    stab_path = out_root / "panel3" / "ranking_stability.csv"
    if stab_path.exists():
        stab = pd.read_csv(stab_path)
        lines.append("**Ranking stability across shock_prob levels:**")
        lines.append("")
        lines.append(stab.to_markdown(index=False))
        n_changes = int(stab["n_rank_changes"].sum()) if "n_rank_changes" in stab else 0
        gi_stable = stab["gi_advantage_stable"].all() if "gi_advantage_stable" in stab else None
        lines += [
            "",
            f"- Total rank-change transitions: {n_changes}",
            f"- graph_informed advantage over backlog_greedy stable: {gi_stable}",
        ]
    elif df3 is not None:
        lines.append("_(Ranking stability CSV not found; Panel 3 results available)_")
    else:
        lines.append("_(Panel 3 not available)_")
    lines += ["", "---", ""]

    # ── Section 10: Budget frontier (Panel 4) ────────────────────────────────
    lines += ["## 10. Budget Frontier Findings (Panel 4)", ""]
    if bfront is not None:
        lines.append("**Policy gain vs expedite_budget (mean across seeds):**")
        lines.append("")
        key = bfront[bfront["policy"].isin(["graph_informed", "mip", "expedite_only"])].copy()
        if not key.empty:
            lines.append(key[
                ["regime_label", "expedite_budget", "policy",
                 "policy_gain_pct_mean", "mip_gain_over_gi_ratio"]
            ].to_markdown(index=False))
    elif df4 is not None:
        lines.append("_(budget_frontier.csv not found; Panel 4 results available)_")
    else:
        lines.append("_(Panel 4 not available)_")
    lines += ["", "---", ""]

    # ── Section 11: Key thesis takeaways ─────────────────────────────────────
    lines += [
        "## 11. Key Thesis-Friendly Takeaways",
        "",
        "The report must answer each of these questions explicitly:",
        "",
    ]

    takeaways = {
        "When do policies differentiate most?":
            "_(Fill in from Panel 1 differentiation report and Panel 2 mechanism plots.)_",
        "When is rerouting the dominant lever?":
            "_(Compare reroute_only vs expedite_only AUC across Panel 1 regimes.)_",
        "When is expediting the dominant lever?":
            "_(Note regimes where expedite_only outperforms reroute_only.)_",
        "When does graph_informed provide extra value beyond backlog_greedy?":
            "_(From Panel 2: does the gap widen with shock_magnitude?)_",
        "How close does graph_informed come to MIP optimality?":
            "_(From mip_vs_graph_informed_gap_pct across regimes; note any ≤3% gap.)_",
        "Which regime family for headline thesis results?":
            "_(Recommend the regime where: disruption is severe, rankings are stable,"
            " GI beats BG, and MIP gap is meaningful but not overwhelming.)_",
    }

    if df1 is not None:
        # Auto-answer some questions from data
        ro_auc = df1[df1["policy"] == "reroute_only"]["backlog_auc"].mean()
        eo_auc = df1[df1["policy"] == "expedite_only"]["backlog_auc"].mean()
        if not np.isnan(ro_auc) and not np.isnan(eo_auc):
            dominant = "rerouting" if ro_auc < eo_auc else "expediting"
            takeaways["When is rerouting the dominant lever?"] = (
                f"Rerouting is dominant overall (reroute_only AUC={_fmt(ro_auc)} vs "
                f"expedite_only AUC={_fmt(eo_auc)}). "
                f"{'Rerouting' if dominant == 'rerouting' else 'Expediting'} is the primary lever."
            )

        # Headline regime recommendation
        if diff_r is not None and "passes" in diff_r.columns:
            passing = diff_r[diff_r["passes"] == True]["regime_id"].tolist()
            if passing:
                # Pick regime with highest ni_delta_auc among passing
                best_cand = diff_r[diff_r["passes"] == True].nlargest(1, "ni_delta_auc")
                best_rid  = best_cand.iloc[0]["regime_id"]
                takeaways["Which regime family for headline thesis results?"] = (
                    f"Recommended: **`{best_rid}`** — highest disruption damage "
                    f"({_fmt(best_cand.iloc[0]['ni_delta_auc'])} AUC increase) "
                    f"among the {len(passing)} passing regimes."
                )

    for q, a in takeaways.items():
        lines += [f"### {q}", "", a, ""]

    lines += ["---", ""]

    # ── Section 12: Limitations ───────────────────────────────────────────────
    lines += [
        "## 12. Limitations",
        "",
        "- Results depend on the calibrated simulator parameters; findings may not"
        " generalise to real supply chains.",
        "- MIP is a one-step lookahead only and does not account for future shocks.",
        "- graph_informed uses greedy selection; optimality gap vs true multi-period"
        " MIP is not measured.",
        "- All policies use the same random seed for the environment, ensuring"
        " matched comparisons, but results may vary across different network structures.",
        "- The `disruption_too_mild` flag excludes some regimes from damage-gain"
        " reporting; these are listed in the regime table above.",
        "",
        "---",
        "",
    ]

    # ── Section 13: Recommended regime family ─────────────────────────────────
    lines += [
        "## 13. Recommended Regime Family for Headline Thesis Results",
        "",
        "_(Filled in automatically in Section 11 above; see 'Which regime family' takeaway.)_",
        "",
        "---",
        "",
    ]

    # ── Section 14: One-page executive summary ────────────────────────────────
    lines += [
        "## 14. One-Page Executive Summary",
        "",
        "### 3-4 Key Findings",
        "",
    ]
    for i, b in enumerate(summary_bullets, 1):
        lines.append(f"{i}. {b.lstrip('0123456789. ')}")
    lines += [
        "",
        "### Single Best Figure for Thesis",
        "",
        "**`plots/plot2_bar_named_regimes.png`** — Bar chart of all 7 policies across",
        "the 4 interpretable named regimes, with error bars and a secondary panel",
        "showing policy_gain_pct. This single figure captures the headline result:",
        "which policies win, by how much, and under what structural conditions.",
        "",
    ]

    # ── Write report ──────────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    report_path = out_root / "report.md"
    report_path.write_text(report_text)
    print(f"[Report] Written → {report_path}")
    print(f"  Sections: {report_text.count('## ')}")
    wc = len(report_text.split())
    print(f"  Word count: ~{wc}")


def main():
    parser = argparse.ArgumentParser(description="Generate thesis report from panel results")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    generate_report(Path(args.output_dir))


if __name__ == "__main__":
    main()
