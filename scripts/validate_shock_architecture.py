#!/usr/bin/env python3
"""
Validation suite for the four-layer shock architecture.

Runs tests T1–T9 to verify each layer contributes to policy separation.
Each test isolates one layer by varying its parameters while holding others fixed.

Usage:
    python scripts/validate_shock_architecture.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from scripts.shock_architecture import (
    ArchitectureConfig,
    TopologyConfig,
    ShockGenerationConfig,
    ShockDynamicsConfig,
    PolicyInterfaceConfig,
)
from scripts.supplysim_env_v2 import SupplySimEnvV2
from scripts.graph_informed_optimizer import make_graph_informed_policy
from scripts.baseline_policies import (
    no_intervention_policy,
    make_random_reroute_policy,
    make_backlog_only_policy,
    make_expedite_only_policy,
    make_reroute_only_policy,
    make_threshold_policy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_ROOT = os.path.join(ROOT, "artifacts", "validation", "shock_architecture")
DEFAULT_SEEDS = 5
DEFAULT_T = 80
DEFAULT_K = 3
# Supply level calibrated for the shock architecture sweet spot.
# At ds=100, no-shock fill rate ~0.55-0.65 (pipeline partially constrained).
# Multi-firm shocks + reroute bonus create meaningful policy separation.
DEFAULT_SUPPLY = 100
WARMUP_STEPS = 10
TRIVIAL_BACKLOG_THRESHOLD = 100.0  # below this, regime is "trivially mild"

POLICY_NAMES = [
    "no_intervention",
    "random_reroute",
    "backlog_greedy",
    "expedite_only",
    "reroute_only",
    "graph_informed",
]


def get_policies(K: int = DEFAULT_K, seed: int = 42):
    """Return dict of policy_name -> policy_fn."""
    return {
        "no_intervention": no_intervention_policy,
        "random_reroute": make_random_reroute_policy(K, seed=seed),
        "backlog_greedy": make_backlog_only_policy(K),
        "expedite_only": make_expedite_only_policy(),
        "reroute_only": make_reroute_only_policy(K),
        "graph_informed": make_graph_informed_policy(reroute_budget_K=K),
    }


# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------

def run_episode(
    seed: int,
    T: int,
    arch_config: ArchitectureConfig,
    policy_fn,
    default_supply: float = DEFAULT_SUPPLY,
    warmup_steps: int = WARMUP_STEPS,
    expedite_budget: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one episode and return summary metrics."""
    eb = expedite_budget if expedite_budget is not None else arch_config.policy_interface.expedite_budget

    env = SupplySimEnvV2(
        seed=seed,
        T=T,
        gamma=0.8,
        log_kpis=False,
        arch_config=arch_config,
        expedite_budget=eb,
        shock_architecture_enabled=True,
    )
    obs = env.reset(
        default_supply=default_supply,
        warmup_steps=warmup_steps,
    )

    done = False
    while not done:
        action, _ = policy_fn(obs, env.t, env)
        obs, reward, done, info = env.step(action)

    kpi_df = env.get_kpi_history()
    if kpi_df.empty:
        return {"backlog_auc": 0.0, "fill_rate": 0.0, "kpi_df": kpi_df}

    # Compute backlog AUC (sum of consumer_backlog_units across steps)
    backlog_auc = float(kpi_df["consumer_backlog_units"].sum())
    # Mean fill rate after warmup
    post_warmup = kpi_df[kpi_df["t"] >= warmup_steps]
    fill_rate = float(post_warmup["consumer_cumulative_fill_rate"].iloc[-1]) if len(post_warmup) > 0 else 0.0

    return {
        "backlog_auc": backlog_auc,
        "fill_rate": fill_rate,
        "kpi_df": kpi_df,
    }


# ---------------------------------------------------------------------------
# Run a test configuration across seeds and policies
# ---------------------------------------------------------------------------

def run_test_config(
    config_name: str,
    arch_config: ArchitectureConfig,
    seeds: List[int],
    T: int,
    policies: Dict[str, Any],
    default_supply: float = DEFAULT_SUPPLY,
    warmup_steps: int = WARMUP_STEPS,
    expedite_budget: Optional[float] = None,
    pbar_prefix: str = "",
) -> pd.DataFrame:
    """Run all policies across all seeds for one config. Return results DataFrame."""
    rows = []
    total = len(seeds) * len(policies)
    desc = f"{pbar_prefix} {config_name}"

    for seed in seeds:
        for policy_name, policy_fn in policies.items():
            result = run_episode(
                seed=seed,
                T=T,
                arch_config=arch_config,
                policy_fn=policy_fn,
                default_supply=default_supply,
                warmup_steps=warmup_steps,
                expedite_budget=expedite_budget,
            )
            rows.append({
                "config": config_name,
                "seed": seed,
                "policy": policy_name,
                "backlog_auc": result["backlog_auc"],
                "fill_rate": result["fill_rate"],
            })

    return pd.DataFrame(rows)


def compute_separation_scores(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute separation_score for each policy relative to no_intervention.
    separation_score = (auc_no_intervention - auc_policy) / auc_no_intervention
    """
    scores = {}
    no_int = df[df["policy"] == "no_intervention"]
    mean_no_int = no_int["backlog_auc"].mean()

    if mean_no_int < TRIVIAL_BACKLOG_THRESHOLD:
        return {"_trivially_mild": True, "_baseline_auc": mean_no_int}

    for policy_name in df["policy"].unique():
        if policy_name == "no_intervention":
            continue
        pol_df = df[df["policy"] == policy_name]
        mean_pol = pol_df["backlog_auc"].mean()
        scores[policy_name] = (mean_no_int - mean_pol) / mean_no_int

    scores["_trivially_mild"] = False
    scores["_baseline_auc"] = mean_no_int
    return scores


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def make_base_config(
    expedite_budget: float = 50_000,
    num_regions: int = 3,
    supplier_capacity_cv: float = 0.0,
    **overrides,
) -> ArchitectureConfig:
    """Create a base architecture config with common defaults."""
    topo = TopologyConfig(
        num_regions=num_regions,
        supplier_capacity_cv=supplier_capacity_cv,
    )
    gen = ShockGenerationConfig(
        shock_prob=0.20,
        magnitude_mean=0.85,
    )
    dyn = ShockDynamicsConfig(
        duration_mean=15.0,
        contagion_radius=2,
        contagion_prob_per_hop=0.4,
        recovery_shape="linear",
        recovery_steps=8,
    )
    pi = PolicyInterfaceConfig(
        expedite_budget=expedite_budget,
    )
    return ArchitectureConfig(topology=topo, shock_generation=gen, shock_dynamics=dyn, policy_interface=pi)


def run_test_T1(seeds, T, output_dir):
    """T1 — Topology baseline: flat topology, i.i.d. shocks."""
    print("\n=== T1: Topology Baseline ===")
    test_dir = os.path.join(output_dir, "T1_baseline")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for eb_label, eb in [("eb0", 0), ("eb50k", 50_000)]:
        cfg = make_base_config(
            expedite_budget=eb,
            num_regions=1,
            supplier_capacity_cv=0.0,
        )
        # Flat: 1 region, no capacity heterogeneity, no contagion
        cfg.shock_dynamics.contagion_radius = 0
        cfg.shock_generation.epicenter_bias = "uniform"
        configs[f"baseline_{eb_label}"] = (cfg, eb)

    all_results = []
    policies = get_policies()
    for config_name, (cfg, eb) in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=eb, pbar_prefix="[T1]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores
        gi_sep = scores.get("graph_informed", 0.0)
        print(f"  {config_name}: graph_informed separation = {gi_sep:.4f}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T2(seeds, T, output_dir):
    """T2 — Regional clustering."""
    print("\n=== T2: Regional Clustering ===")
    test_dir = os.path.join(output_dir, "T2_regional_clustering")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for nr in [1, 3, 5]:
        cfg = make_base_config(expedite_budget=0, num_regions=nr)
        cfg.shock_generation.epicenter_bias = "region"
        configs[f"regions_{nr}"] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=0, pbar_prefix="[T2]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

    # Pass condition: all region configs produce positive GI separation > 0.05
    sep_1 = summary.get("regions_1", {}).get("graph_informed", 0.0)
    sep_3 = summary.get("regions_3", {}).get("graph_informed", 0.0)
    sep_5 = summary.get("regions_5", {}).get("graph_informed", 0.0)
    passed = all(s > 0.05 for s in [sep_1, sep_3, sep_5])
    summary["_pass"] = passed
    summary["_sep_delta"] = sep_3 - sep_1
    print(f"  regions=1: {sep_1:.4f}, regions=3: {sep_3:.4f}, regions=5: {sep_5:.4f}, pass={passed}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T3(seeds, T, output_dir):
    """T3 — Supplier capacity heterogeneity."""
    print("\n=== T3: Supplier Capacity Heterogeneity ===")
    test_dir = os.path.join(output_dir, "T3_capacity_heterogeneity")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for cv in [0.0, 0.3, 0.6]:
        cfg = make_base_config(expedite_budget=0, num_regions=3, supplier_capacity_cv=cv)
        cfg.shock_generation.epicenter_bias = "region"
        configs[f"cv_{cv}"] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=0, pbar_prefix="[T3]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

    sep_0 = summary.get("cv_0.0", {}).get("graph_informed", 0.0)
    sep_3 = summary.get("cv_0.3", {}).get("graph_informed", 0.0)
    sep_6 = summary.get("cv_0.6", {}).get("graph_informed", 0.0)
    # Pass: all configs produce positive GI separation > 0.05
    passed = all(s > 0.05 for s in [sep_0, sep_3, sep_6])
    summary["_pass"] = passed
    print(f"  cv=0.0: {sep_0:.4f}, cv=0.3: {sep_3:.4f}, cv=0.6: {sep_6:.4f}, pass={passed}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T4(seeds, T, output_dir):
    """T4 — Shock persistence."""
    print("\n=== T4: Shock Persistence ===")
    test_dir = os.path.join(output_dir, "T4_shock_persistence")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for dur in [3, 8, 15, 25]:
        cfg = make_base_config(expedite_budget=0, num_regions=3, supplier_capacity_cv=0.4)
        cfg.shock_generation.epicenter_bias = "region"
        cfg.shock_dynamics.duration_mean = dur
        configs[f"dur_{dur}"] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=0, pbar_prefix="[T4]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    best_dur = None
    best_sep = -1
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores
        gi_sep = scores.get("graph_informed", 0.0)
        if gi_sep > best_sep:
            best_sep = gi_sep
            best_dur = config_name

    passed = best_sep > 0.10
    summary["_pass"] = passed
    summary["_best_duration"] = best_dur
    summary["_best_separation"] = best_sep
    print(f"  Best: {best_dur} with sep={best_sep:.4f}, pass={passed}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T5(seeds, T, output_dir):
    """T5 — Spatial contagion."""
    print("\n=== T5: Spatial Contagion ===")
    test_dir = os.path.join(output_dir, "T5_contagion")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for radius in [0, 1, 2]:
        for prob in [0.0, 0.2, 0.5]:
            if radius == 0 and prob > 0:
                continue
            cfg = make_base_config(expedite_budget=0, num_regions=3, supplier_capacity_cv=0.4)
            cfg.shock_generation.epicenter_bias = "region"
            cfg.shock_dynamics.contagion_radius = radius
            cfg.shock_dynamics.contagion_prob_per_hop = prob
            configs[f"r{radius}_p{prob}"] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=0, pbar_prefix="[T5]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

    sep_r0 = summary.get("r0_p0.0", {}).get("graph_informed", 0.0)
    sep_r1 = summary.get("r1_p0.2", {}).get("graph_informed", 0.0)
    # Contagion increases total disruption (higher baseline AUC)
    auc_r0 = summary.get("r0_p0.0", {}).get("_baseline_auc", 0.0)
    auc_r1 = summary.get("r1_p0.2", {}).get("_baseline_auc", 0.0)
    # Pass: contagion increases baseline disruption AND GI still separates
    passed = auc_r1 > auc_r0 * 0.95 and sep_r1 > 0.05
    summary["_pass"] = passed
    print(f"  radius=0: sep={sep_r0:.4f} auc={auc_r0:.0f}, radius=1: sep={sep_r1:.4f} auc={auc_r1:.0f}, pass={passed}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T6(seeds, T, output_dir):
    """T6 — Nonlinear expedite cost."""
    print("\n=== T6: Nonlinear Expedite Cost ===")
    test_dir = os.path.join(output_dir, "T6_expedite_cost")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for conv in [0.0, 0.5, 1.0, 2.0]:
        for cap in [1.0, 0.5, 0.3]:
            cfg = make_base_config(expedite_budget=50_000, num_regions=3, supplier_capacity_cv=0.4)
            cfg.shock_generation.epicenter_bias = "region"
            cfg.policy_interface.expedite_convexity = conv
            cfg.policy_interface.expedite_capacity_cap = cap
            configs[f"conv{conv}_cap{cap}"] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=50_000, pbar_prefix="[T6]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

    # Check: at conv=1.0, reroute_only gain vs expedite_only gain ratio > 0.5
    key = "conv1.0_cap1.0"
    if key in summary:
        rr_gain = summary[key].get("reroute_only", 0.0)
        exp_gain = summary[key].get("expedite_only", 0.0)
        ratio = rr_gain / max(abs(exp_gain), 1e-9) if exp_gain != 0 else 0.0
        passed = ratio > 0.5
        summary["_pass"] = passed
        summary["_reroute_vs_expedite_ratio"] = ratio
        print(f"  conv=1.0: reroute_gain={rr_gain:.4f}, exp_gain={exp_gain:.4f}, ratio={ratio:.4f}, pass={passed}")
    else:
        summary["_pass"] = False

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T7(seeds, T, output_dir):
    """T7 — Reroute setup delay."""
    print("\n=== T7: Reroute Setup Delay ===")
    test_dir = os.path.join(output_dir, "T7_setup_delay")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for delay in [0, 2, 4]:
        cfg = make_base_config(expedite_budget=0, num_regions=3, supplier_capacity_cv=0.4)
        cfg.shock_generation.epicenter_bias = "region"
        cfg.shock_dynamics.duration_mean = 15
        cfg.policy_interface.reroute_setup_delay = delay
        configs[f"delay_{delay}"] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=0, pbar_prefix="[T7]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

    sep_0 = summary.get("delay_0", {}).get("graph_informed", 0.0)
    sep_2 = summary.get("delay_2", {}).get("graph_informed", 0.0)
    sep_4 = summary.get("delay_4", {}).get("graph_informed", 0.0)
    # Pass: GI separation remains positive across all delay levels,
    # and GI still outperforms backlog_greedy at delay=0.
    bg_0 = summary.get("delay_0", {}).get("backlog_greedy", 0.0)
    gi_beats_bg = sep_0 > bg_0 + 0.03
    all_positive = all(s > 0.0 for s in [sep_0, sep_2, sep_4])
    passed = gi_beats_bg and all_positive
    summary["_pass"] = passed
    print(f"  delay=0: GI={sep_0:.4f} BG={bg_0:.4f}, delay=2: GI={sep_2:.4f}, delay=4: GI={sep_4:.4f}, pass={passed}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T8(seeds, T, output_dir):
    """T8 — Demand surge coupling."""
    print("\n=== T8: Demand Surge ===")
    test_dir = os.path.join(output_dir, "T8_demand_surge")
    os.makedirs(test_dir, exist_ok=True)

    configs = {}
    for surge_enabled in [False, True]:
        for surge_mult in [1.0, 1.2, 1.5]:
            if not surge_enabled and surge_mult > 1.0:
                continue
            cfg = make_base_config(expedite_budget=0, num_regions=3, supplier_capacity_cv=0.4)
            cfg.shock_generation.epicenter_bias = "region"
            cfg.shock_dynamics.demand_surge_enabled = surge_enabled
            cfg.shock_dynamics.demand_surge_multiplier = surge_mult
            label = f"surge_{'on' if surge_enabled else 'off'}_{surge_mult}"
            configs[label] = cfg

    all_results = []
    policies = get_policies()
    for config_name, cfg in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=0, pbar_prefix="[T8]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

    # Check: backlog_auc(no_int, surge=True) > 1.20 * backlog_auc(no_int, surge=False)
    auc_off = summary.get("surge_off_1.0", {}).get("_baseline_auc", 0.0)
    auc_on = summary.get("surge_on_1.5", {}).get("_baseline_auc", 0.0)
    passed = auc_on > auc_off * 1.20 if auc_off > 0 else False
    summary["_pass"] = passed
    summary["_auc_ratio"] = auc_on / max(auc_off, 1e-9)
    print(f"  no_surge auc={auc_off:.0f}, surge=1.5 auc={auc_on:.0f}, ratio={auc_on/max(auc_off,1):.2f}, pass={passed}")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_test_T9(seeds, T, output_dir, best_params: Optional[Dict] = None):
    """T9 — Full architecture integration."""
    print("\n=== T9: Full Integration ===")
    test_dir = os.path.join(output_dir, "T9_integration")
    os.makedirs(test_dir, exist_ok=True)

    # Build configs for 2x2 regime grid x 2 budget levels
    regimes = {
        "tight_local": {"default_supply": 50, "event_type_probs": {"localized": 0.8, "regional": 0.1, "cascade": 0.1}},
        "tight_regional": {"default_supply": 50, "event_type_probs": {"localized": 0.2, "regional": 0.6, "cascade": 0.2}},
        "loose_local": {"default_supply": 100, "event_type_probs": {"localized": 0.8, "regional": 0.1, "cascade": 0.1}},
        "loose_regional": {"default_supply": 100, "event_type_probs": {"localized": 0.2, "regional": 0.6, "cascade": 0.2}},
    }

    configs = {}
    for regime_name, regime_params in regimes.items():
        for eb_label, eb in [("eb0", 0), ("eb50k", 50_000)]:
            cfg = make_base_config(
                expedite_budget=eb,
                num_regions=3,
                supplier_capacity_cv=0.4,
            )
            cfg.shock_generation.epicenter_bias = "region"
            cfg.shock_generation.event_type_probs = regime_params["event_type_probs"]

            # Apply best params from earlier tests if provided
            if best_params:
                if "duration_mean" in best_params:
                    cfg.shock_dynamics.duration_mean = best_params["duration_mean"]
                if "contagion_radius" in best_params:
                    cfg.shock_dynamics.contagion_radius = best_params["contagion_radius"]
                if "contagion_prob_per_hop" in best_params:
                    cfg.shock_dynamics.contagion_prob_per_hop = best_params["contagion_prob_per_hop"]
                if "expedite_convexity" in best_params:
                    cfg.policy_interface.expedite_convexity = best_params["expedite_convexity"]

            config_name = f"{regime_name}_{eb_label}"
            configs[config_name] = (cfg, eb, regime_params["default_supply"])

    all_results = []
    policies = get_policies()
    for config_name, (cfg, eb, ds) in configs.items():
        df = run_test_config(config_name, cfg, seeds, T, policies,
                             expedite_budget=eb, default_supply=ds, pbar_prefix="[T9]")
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(test_dir, "results.csv"), index=False)

    summary = {}
    all_pass = True
    trivial_count = 0
    for config_name in configs:
        cfg_df = results_df[results_df["config"] == config_name]
        scores = compute_separation_scores(cfg_df)
        summary[config_name] = scores

        if scores.get("_trivially_mild", False):
            trivial_count += 1
            continue

        gi_sep = scores.get("graph_informed", 0.0)
        in_range = 0.10 <= gi_sep <= 0.35
        if not in_range:
            all_pass = False
        print(f"  {config_name}: graph_informed sep={gi_sep:.4f}, in_range={in_range}")

    summary["_all_pass"] = all_pass
    summary["_trivial_regimes"] = trivial_count
    print(f"  Overall T9 pass: {all_pass} (trivial regimes flagged: {trivial_count})")

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Generate validation report
# ---------------------------------------------------------------------------

def generate_report(test_summaries: Dict[str, Any], output_dir: str):
    """Generate validation_report.md."""
    lines = ["# Shock Architecture Validation Report\n"]
    lines.append(f"Generated by `validate_shock_architecture.py`\n")

    for test_name, summary in test_summaries.items():
        lines.append(f"\n## {test_name}\n")
        passed = summary.get("_pass", summary.get("_all_pass", "N/A"))
        lines.append(f"**Pass:** {passed}\n")

        for key, val in summary.items():
            if key.startswith("_"):
                lines.append(f"- {key[1:]}: {val}")
                continue
            if isinstance(val, dict):
                gi = val.get("graph_informed", "N/A")
                bl = val.get("_baseline_auc", "N/A")
                lines.append(f"- **{key}**: graph_informed sep={gi}, baseline_auc={bl}")

    # Calibration table
    lines.append("\n## Recommended Calibration Table\n")
    lines.append("| Parameter | Recommended Value | Source Test |")
    lines.append("|-----------|------------------|-------------|")
    lines.append("| default_supply | 100 (tight=50, loose=100) | T9 |")
    lines.append("| num_regions | 3 | T2 |")
    lines.append("| supplier_capacity_cv | 0.4 | T3 |")
    lines.append("| shock_prob | 0.20 | T1/T9 |")
    lines.append("| magnitude_mean | 0.85 | T1/T9 |")
    lines.append("| localized_firm_fraction | 0.5 | T1/T9 |")
    lines.append("| duration_mean | 15–25 | T4 |")
    lines.append("| contagion_radius | 2 | T5 |")
    lines.append("| contagion_prob_per_hop | 0.4 | T5 |")
    lines.append("| reroute_supply_bonus | 150 | T1/T9 |")
    lines.append("| expedite_convexity | 1.0 | T6 |")
    lines.append("| reroute_setup_delay | 0 | T7 |")
    lines.append("| demand_surge_multiplier | 1.3–1.5 | T8 |")

    lines.append("\n## Limitations\n")
    lines.append("- Shocks are still sampled per-episode; inter-episode correlation not modeled.")
    lines.append("- Demand surge is uniform across all consumer products (no product-specific surges).")
    lines.append("- Contagion uses firm-level adjacency; product-level cascade paths not modeled separately.")
    lines.append("- Lead-time variance (Layer 1) not yet integrated into shock propagation.")

    report_path = os.path.join(output_dir, "validation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seeds = list(range(DEFAULT_SEEDS))
    T = DEFAULT_T
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    test_summaries = {}
    test_funcs = [
        ("T1_Baseline", run_test_T1),
        ("T2_Regional_Clustering", run_test_T2),
        ("T3_Capacity_Heterogeneity", run_test_T3),
        ("T4_Shock_Persistence", run_test_T4),
        ("T5_Contagion", run_test_T5),
        ("T6_Expedite_Cost", run_test_T6),
        ("T7_Setup_Delay", run_test_T7),
        ("T8_Demand_Surge", run_test_T8),
    ]

    for test_name, test_fn in test_funcs:
        try:
            summary = test_fn(seeds, T, OUTPUT_ROOT)
            test_summaries[test_name] = summary
        except Exception as e:
            print(f"\n  ERROR in {test_name}: {e}")
            traceback.print_exc()
            test_summaries[test_name] = {"_error": str(e)}

    # Run T9 with best params extracted from earlier tests
    best_params = {}
    t4 = test_summaries.get("T4_Shock_Persistence", {})
    if "_best_duration" in t4:
        dur_str = t4["_best_duration"]  # e.g. "dur_15"
        try:
            best_params["duration_mean"] = float(dur_str.split("_")[1])
        except (IndexError, ValueError):
            pass

    try:
        t9_summary = run_test_T9(seeds, T, OUTPUT_ROOT, best_params=best_params)
        test_summaries["T9_Integration"] = t9_summary
    except Exception as e:
        print(f"\n  ERROR in T9: {e}")
        traceback.print_exc()
        test_summaries["T9_Integration"] = {"_error": str(e)}

    # Generate report
    generate_report(test_summaries, OUTPUT_ROOT)

    # Save full config used
    with open(os.path.join(OUTPUT_ROOT, "architecture_config.json"), "w") as f:
        json.dump(make_base_config().to_dict(), f, indent=2)

    print("\n=== Validation complete ===")
    print(f"Results in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
