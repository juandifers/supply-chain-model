#!/usr/bin/env python3
"""
Head-to-head comparison between Arch A (original) and Arch B (new shock architecture).

Runs:
- 4 regimes (tight/loose × local/regional)
- Both budget levels (eb=0, eb=50000)
- All 6 policies
- 10 seeds

Produces comparison table and arch_comparison.md report.

Usage:
    python scripts/compare_architectures.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.shock_architecture import ArchitectureConfig
from scripts.supplysim_env import SupplySimEnv
from scripts.supplysim_env_v2 import SupplySimEnvV2
from scripts.validate_shock_architecture import (
    get_policies,
    compute_separation_scores,
    make_base_config,
    WARMUP_STEPS,
    TRIVIAL_BACKLOG_THRESHOLD,
)

OUTPUT_DIR = os.path.join(ROOT, "artifacts", "validation", "shock_architecture")
COMPARISON_SEEDS = 10
COMPARISON_T = 80
DEFAULT_SUPPLY = 100

REGIMES = {
    "tight_local": {
        "default_supply": 50,
        "shock_prob": 0.25,
        "shock_magnitude": 0.85,
    },
    "tight_regional": {
        "default_supply": 50,
        "shock_prob": 0.20,
        "shock_magnitude": 0.85,
    },
    "loose_local": {
        "default_supply": 100,
        "shock_prob": 0.20,
        "shock_magnitude": 0.85,
    },
    "loose_regional": {
        "default_supply": 100,
        "shock_prob": 0.15,
        "shock_magnitude": 0.85,
    },
}


def run_arch_a_episode(
    seed: int, T: int, policy_fn,
    default_supply: float, shock_prob: float,
    expedite_budget: float,
) -> Dict[str, Any]:
    """Run one episode using Arch A (original supplysim_env)."""
    env = SupplySimEnv(
        seed=seed, T=T, gamma=0.8, log_kpis=False,
        expedite_budget=expedite_budget,
    )
    shock_supply = default_supply * (1 - 0.7)  # 70% magnitude
    obs = env.reset(
        default_supply=default_supply,
        shock_prob=shock_prob,
        shock_supply=shock_supply,
        recovery_rate=1.05,
    )

    t0 = time.time()
    done = False
    while not done:
        action, _ = policy_fn(obs, env.t, env)
        obs, reward, done, info = env.step(action)
    elapsed = time.time() - t0

    kpi_df = env.get_kpi_history()
    if kpi_df.empty:
        return {"backlog_auc": 0.0, "fill_rate": 0.0, "runtime": elapsed}

    backlog_auc = float(kpi_df["consumer_backlog_units"].sum())
    post_warmup = kpi_df[kpi_df["t"] >= WARMUP_STEPS]
    fill_rate = float(post_warmup["consumer_cumulative_fill_rate"].iloc[-1]) if len(post_warmup) > 0 else 0.0
    return {"backlog_auc": backlog_auc, "fill_rate": fill_rate, "runtime": elapsed}


def run_arch_b_episode(
    seed: int, T: int, policy_fn,
    arch_config: ArchitectureConfig,
    default_supply: float,
    expedite_budget: float,
) -> Dict[str, Any]:
    """Run one episode using Arch B (new shock architecture)."""
    env = SupplySimEnvV2(
        seed=seed, T=T, gamma=0.8, log_kpis=False,
        arch_config=arch_config,
        expedite_budget=expedite_budget,
        shock_architecture_enabled=True,
    )
    obs = env.reset(default_supply=default_supply, warmup_steps=WARMUP_STEPS)

    t0 = time.time()
    done = False
    while not done:
        action, _ = policy_fn(obs, env.t, env)
        obs, reward, done, info = env.step(action)
    elapsed = time.time() - t0

    kpi_df = env.get_kpi_history()
    if kpi_df.empty:
        return {"backlog_auc": 0.0, "fill_rate": 0.0, "runtime": elapsed}

    backlog_auc = float(kpi_df["consumer_backlog_units"].sum())
    post_warmup = kpi_df[kpi_df["t"] >= WARMUP_STEPS]
    fill_rate = float(post_warmup["consumer_cumulative_fill_rate"].iloc[-1]) if len(post_warmup) > 0 else 0.0
    return {"backlog_auc": backlog_auc, "fill_rate": fill_rate, "runtime": elapsed}


def load_calibrated_config() -> ArchitectureConfig:
    """Load calibrated config or use defaults."""
    config_path = os.path.join(OUTPUT_DIR, "calibrated_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return ArchitectureConfig.from_dict(json.load(f))
    return make_base_config()


def run_comparison() -> pd.DataFrame:
    """Run full comparison between Arch A and Arch B."""
    seeds = list(range(COMPARISON_SEEDS))
    T = COMPARISON_T
    policies = get_policies()
    arch_b_config = load_calibrated_config()

    rows = []
    total_runs = len(REGIMES) * 2 * len(policies) * len(seeds) * 2  # 2 archs, 2 budgets
    run_count = 0

    for regime_name, regime_params in REGIMES.items():
        ds = regime_params["default_supply"]
        sp = regime_params["shock_prob"]

        for eb_label, eb in [("eb0", 0), ("eb50k", 50_000)]:
            for policy_name, policy_fn in policies.items():
                for seed in seeds:
                    # Arch A
                    res_a = run_arch_a_episode(seed, T, policy_fn, ds, sp, eb)
                    rows.append({
                        "arch": "A",
                        "regime": regime_name,
                        "budget": eb_label,
                        "policy": policy_name,
                        "seed": seed,
                        "backlog_auc": res_a["backlog_auc"],
                        "fill_rate": res_a["fill_rate"],
                        "runtime": res_a["runtime"],
                    })
                    run_count += 1

                    # Arch B
                    arch_b = ArchitectureConfig.from_dict(arch_b_config.to_dict())
                    if "regional" in regime_name:
                        arch_b.shock_generation.event_type_probs = {
                            "localized": 0.2, "regional": 0.6, "cascade": 0.2
                        }
                    else:
                        arch_b.shock_generation.event_type_probs = {
                            "localized": 0.8, "regional": 0.1, "cascade": 0.1
                        }

                    res_b = run_arch_b_episode(seed, T, policy_fn, arch_b, ds, eb)
                    rows.append({
                        "arch": "B",
                        "regime": regime_name,
                        "budget": eb_label,
                        "policy": policy_name,
                        "seed": seed,
                        "backlog_auc": res_b["backlog_auc"],
                        "fill_rate": res_b["fill_rate"],
                        "runtime": res_b["runtime"],
                    })
                    run_count += 1

                    if run_count % 50 == 0:
                        print(f"  Progress: {run_count}/{total_runs} runs")

    return pd.DataFrame(rows)


def compute_comparison_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per (arch, regime, budget) separation scores and metrics."""
    results = []

    for (arch, regime, budget), group in df.groupby(["arch", "regime", "budget"]):
        scores = compute_separation_scores(group)

        no_int_auc = scores.get("_baseline_auc", 0)
        gi_sep = scores.get("graph_informed", 0)
        bg_sep = scores.get("backlog_greedy", 0)
        rr_sep = scores.get("reroute_only", 0)
        exp_sep = scores.get("expedite_only", 0)

        # Lever contrast: reroute_only vs expedite_only gap
        lever_contrast = rr_sep - exp_sep

        # Rank stability: Spearman correlation of per-seed policy rankings
        # (simplified: check if ordering is consistent across seeds)
        policy_aucs_per_seed = group.pivot_table(
            index="seed", columns="policy", values="backlog_auc"
        )
        if len(policy_aucs_per_seed) >= 3:
            from scipy.stats import spearmanr
            # Rank each seed, compute mean pairwise Spearman
            ranks = policy_aucs_per_seed.rank(axis=1)
            correlations = []
            seed_list = list(ranks.index)
            for i in range(len(seed_list)):
                for j in range(i + 1, len(seed_list)):
                    r, _ = spearmanr(ranks.loc[seed_list[i]], ranks.loc[seed_list[j]])
                    if not np.isnan(r):
                        correlations.append(r)
            rank_stability = float(np.mean(correlations)) if correlations else 0.0
        else:
            rank_stability = 0.0

        # Mean runtime
        mean_runtime = group["runtime"].mean()

        results.append({
            "arch": arch,
            "regime": regime,
            "budget": budget,
            "baseline_auc": no_int_auc,
            "gi_separation": gi_sep,
            "bg_separation": bg_sep,
            "rr_separation": rr_sep,
            "exp_separation": exp_sep,
            "lever_contrast": lever_contrast,
            "rank_stability": rank_stability,
            "mean_runtime": mean_runtime,
        })

    return pd.DataFrame(results)


def generate_comparison_report(metrics_df: pd.DataFrame, raw_df: pd.DataFrame):
    """Generate arch_comparison.md."""
    lines = ["# Architecture Comparison Report\n"]
    lines.append("## Arch A (Original) vs Arch B (New Shock Architecture)\n")

    # Summary table
    lines.append("### Comparison Table\n")
    lines.append("| Arch | Regime | Budget | GI Sep | BG Sep | RR Sep | Exp Sep | Lever Contrast | Rank Stability | Runtime (s) |")
    lines.append("|------|--------|--------|--------|--------|--------|---------|----------------|----------------|-------------|")

    for _, row in metrics_df.sort_values(["arch", "regime", "budget"]).iterrows():
        lines.append(
            f"| {row['arch']} | {row['regime']} | {row['budget']} | "
            f"{row['gi_separation']:.4f} | {row['bg_separation']:.4f} | "
            f"{row['rr_separation']:.4f} | {row['exp_separation']:.4f} | "
            f"{row['lever_contrast']:.4f} | {row['rank_stability']:.3f} | "
            f"{row['mean_runtime']:.2f} |"
        )

    # Answer the four questions
    lines.append("\n### Q1: Does Arch B produce meaningful separation at eb=0 where Arch A failed?\n")
    a_eb0 = metrics_df[(metrics_df["arch"] == "A") & (metrics_df["budget"] == "eb0")]
    b_eb0 = metrics_df[(metrics_df["arch"] == "B") & (metrics_df["budget"] == "eb0")]
    a_mean_sep = a_eb0["gi_separation"].mean() if len(a_eb0) > 0 else 0
    b_mean_sep = b_eb0["gi_separation"].mean() if len(b_eb0) > 0 else 0
    lines.append(f"- Arch A mean GI separation at eb=0: {a_mean_sep:.4f}")
    lines.append(f"- Arch B mean GI separation at eb=0: {b_mean_sep:.4f}")
    if b_mean_sep > a_mean_sep + 0.05:
        lines.append(f"- **Yes**: Arch B improves separation at eb=0 by {b_mean_sep - a_mean_sep:.4f}")
    else:
        lines.append(f"- **No significant improvement**: delta = {b_mean_sep - a_mean_sep:.4f}")

    lines.append("\n### Q2: Does Arch B preserve or improve separation at eb=50000?\n")
    a_eb50k = metrics_df[(metrics_df["arch"] == "A") & (metrics_df["budget"] == "eb50k")]
    b_eb50k = metrics_df[(metrics_df["arch"] == "B") & (metrics_df["budget"] == "eb50k")]
    a_mean_50k = a_eb50k["gi_separation"].mean() if len(a_eb50k) > 0 else 0
    b_mean_50k = b_eb50k["gi_separation"].mean() if len(b_eb50k) > 0 else 0
    lines.append(f"- Arch A mean GI separation at eb=50k: {a_mean_50k:.4f}")
    lines.append(f"- Arch B mean GI separation at eb=50k: {b_mean_50k:.4f}")

    lines.append("\n### Q3: What is the runtime cost of the new architecture?\n")
    a_rt = metrics_df[metrics_df["arch"] == "A"]["mean_runtime"].mean()
    b_rt = metrics_df[metrics_df["arch"] == "B"]["mean_runtime"].mean()
    lines.append(f"- Arch A mean runtime per episode: {a_rt:.2f}s")
    lines.append(f"- Arch B mean runtime per episode: {b_rt:.2f}s")
    overhead = (b_rt - a_rt) / max(a_rt, 0.01) * 100
    lines.append(f"- Overhead: {overhead:.1f}%")

    lines.append("\n### Q4: Which layer contributes most to improvement?\n")
    lines.append("(Full ablation requires separate runs; this report uses heuristic analysis)")
    lines.append("")

    # Compare lever contrast
    a_lc = metrics_df[metrics_df["arch"] == "A"]["lever_contrast"].mean()
    b_lc = metrics_df[metrics_df["arch"] == "B"]["lever_contrast"].mean()
    lines.append(f"- Arch A mean lever contrast (reroute-expedite gap): {a_lc:.4f}")
    lines.append(f"- Arch B mean lever contrast (reroute-expedite gap): {b_lc:.4f}")
    if b_lc > a_lc:
        lines.append("- Arch B shows higher reroute value relative to expedite, suggesting topology/dynamics layers drive improvement")
    else:
        lines.append("- Interface layer (cost structure) may be the primary driver")

    report_path = os.path.join(OUTPUT_DIR, "arch_comparison.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Architecture Comparison: A (Original) vs B (New Shock Architecture)")
    print("=" * 70)

    print(f"\nRunning {COMPARISON_SEEDS} seeds × 4 regimes × 2 budgets × 6 policies × 2 archs...")
    raw_df = run_comparison()

    # Save raw results
    raw_path = os.path.join(OUTPUT_DIR, "arch_comparison_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    # Compute metrics
    metrics_df = compute_comparison_metrics(raw_df)
    metrics_path = os.path.join(OUTPUT_DIR, "arch_comparison_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Generate report
    generate_comparison_report(metrics_df, raw_df)

    print("\n=== Comparison complete ===")


if __name__ == "__main__":
    main()
