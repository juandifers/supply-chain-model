#!/usr/bin/env python3
"""
Sequential calibration search for the shock architecture.

Strategy:
1. Load validation results from T2–T8 (winners per parameter axis)
2. Combine winners into a candidate config
3. Run T9 — if it passes, stop
4. Otherwise, local grid search (vary one param at a time from winner, ±1 step)

Acceptance criteria:
- T9 passes: all 4 regimes show separation_score in [0.10, 0.35] at both budget levels
- reroute_only > 0.05 at eb=0 in >= 2 regimes
- expedite_only > 0.05 at eb=50k in >= 2 regimes
- graph_informed beats backlog_greedy by > 0.03 in >= 2 regimes
- No regime shows separation > 0.45

Usage:
    python scripts/calibrate_shock_architecture.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.shock_architecture import (
    ArchitectureConfig,
    TopologyConfig,
    ShockGenerationConfig,
    ShockDynamicsConfig,
    PolicyInterfaceConfig,
)
from scripts.validate_shock_architecture import (
    OUTPUT_ROOT as VALIDATION_ROOT,
    run_test_T9,
    compute_separation_scores,
    get_policies,
    run_test_config,
    DEFAULT_SEEDS,
    DEFAULT_T,
    DEFAULT_SUPPLY,
    WARMUP_STEPS,
    make_base_config,
)

OUTPUT_DIR = os.path.join(ROOT, "artifacts", "validation", "shock_architecture")
CALIBRATION_CSV = os.path.join(OUTPUT_DIR, "calibration_search.csv")

# Penalty on std for calibration score
PENALTY = 0.5

# Calibration grid (focused, not exhaustive)
CALIBRATION_GRID = {
    "num_regions": [2, 3, 5],
    "supplier_capacity_cv": [0.0, 0.3, 0.5],
    "chokepoint_fraction": [0.0, 0.2],
    "duration_mean": [5, 10, 15],
    "contagion_radius": [0, 1, 2],
    "contagion_prob_per_hop": [0.2, 0.4],
    "recovery_shape": ["instant", "linear"],
    "reroute_setup_delay": [0, 2],
    "reroute_capacity_fraction": [1.0, 0.5],
    "expedite_convexity": [0.0, 1.0],
    "expedite_capacity_cap": [1.0, 0.4],
}


def load_validation_winners() -> Dict[str, Any]:
    """Load best parameter values from T2–T8 validation results."""
    winners = {
        "num_regions": 3,
        "supplier_capacity_cv": 0.4,
        "duration_mean": 15,
        "contagion_radius": 2,
        "contagion_prob_per_hop": 0.4,
        "recovery_shape": "linear",
        "reroute_setup_delay": 0,
        "reroute_capacity_fraction": 1.0,
        "expedite_convexity": 0.0,
        "expedite_capacity_cap": 1.0,
        "chokepoint_fraction": 0.2,
        "reroute_supply_bonus": 150.0,
    }

    # Try to load from validation summaries
    test_param_map = {
        "T2_regional_clustering": "num_regions",
        "T3_capacity_heterogeneity": "supplier_capacity_cv",
        "T4_shock_persistence": "duration_mean",
        "T5_contagion": "contagion_radius",
        "T6_expedite_cost": "expedite_convexity",
        "T7_setup_delay": "reroute_setup_delay",
    }

    for test_dir, param_name in test_param_map.items():
        summary_path = os.path.join(VALIDATION_ROOT, test_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            # Find config with best graph_informed separation
            best_val = None
            best_sep = -1
            for key, val in summary.items():
                if key.startswith("_"):
                    continue
                if isinstance(val, dict):
                    gi_sep = val.get("graph_informed", 0)
                    if isinstance(gi_sep, (int, float)) and gi_sep > best_sep:
                        best_sep = gi_sep
                        best_val = key
            if best_val is not None:
                # Extract parameter value from config name
                parts = best_val.split("_")
                try:
                    if param_name in ("num_regions", "contagion_radius", "reroute_setup_delay"):
                        winners[param_name] = int(parts[-1])
                    elif param_name in ("supplier_capacity_cv", "duration_mean",
                                        "expedite_convexity"):
                        winners[param_name] = float(parts[-1])
                except (ValueError, IndexError):
                    pass
        except Exception:
            pass

    return winners


def build_config_from_params(params: Dict[str, Any], expedite_budget: float = 50_000) -> ArchitectureConfig:
    """Build ArchitectureConfig from a flat parameter dict."""
    topo = TopologyConfig(
        num_regions=params.get("num_regions", 3),
        supplier_capacity_cv=params.get("supplier_capacity_cv", 0.4),
        chokepoint_fraction=params.get("chokepoint_fraction", 0.2),
    )
    gen = ShockGenerationConfig(
        shock_prob=0.20,
        magnitude_mean=0.85,
        epicenter_bias="region",
    )
    dyn = ShockDynamicsConfig(
        duration_mean=params.get("duration_mean", 15),
        contagion_radius=params.get("contagion_radius", 2),
        contagion_prob_per_hop=params.get("contagion_prob_per_hop", 0.4),
        recovery_shape=params.get("recovery_shape", "linear"),
        recovery_steps=8,
    )
    pi = PolicyInterfaceConfig(
        expedite_budget=expedite_budget,
        reroute_setup_delay=params.get("reroute_setup_delay", 0),
        reroute_capacity_fraction=params.get("reroute_capacity_fraction", 1.0),
        expedite_convexity=params.get("expedite_convexity", 0.0),
        expedite_capacity_cap=params.get("expedite_capacity_cap", 1.0),
        reroute_supply_bonus=params.get("reroute_supply_bonus", 150.0),
    )
    return ArchitectureConfig(topology=topo, shock_generation=gen, shock_dynamics=dyn, policy_interface=pi)


def evaluate_t9(params: Dict[str, Any], seeds: List[int], T: int) -> Dict[str, Any]:
    """Run T9-style evaluation and return calibration metrics."""
    regimes = {
        "tight_local": {"default_supply": 50, "event_type_probs": {"localized": 0.8, "regional": 0.1, "cascade": 0.1}},
        "tight_regional": {"default_supply": 50, "event_type_probs": {"localized": 0.2, "regional": 0.6, "cascade": 0.2}},
        "loose_local": {"default_supply": 100, "event_type_probs": {"localized": 0.8, "regional": 0.1, "cascade": 0.1}},
        "loose_regional": {"default_supply": 100, "event_type_probs": {"localized": 0.2, "regional": 0.6, "cascade": 0.2}},
    }

    policies = get_policies()
    all_sep_scores = []
    regime_results = {}

    for regime_name, regime_params in regimes.items():
        for eb_label, eb in [("eb0", 0), ("eb50k", 50_000)]:
            cfg = build_config_from_params(params, expedite_budget=eb)
            cfg.shock_generation.event_type_probs = regime_params["event_type_probs"]
            config_name = f"{regime_name}_{eb_label}"

            df = run_test_config(config_name, cfg, seeds, T, policies,
                                 expedite_budget=eb,
                                 default_supply=regime_params["default_supply"])
            scores = compute_separation_scores(df)
            regime_results[config_name] = scores

            gi_sep = scores.get("graph_informed", 0.0)
            if not scores.get("_trivially_mild", False):
                all_sep_scores.append(gi_sep)

    if len(all_sep_scores) == 0:
        return {"calibration_score": -1.0, "regime_results": regime_results, "params": params}

    mean_sep = float(np.mean(all_sep_scores))
    std_sep = float(np.std(all_sep_scores))
    cal_score = mean_sep - PENALTY * std_sep

    # Check acceptance criteria
    criteria = check_acceptance(regime_results)

    return {
        "calibration_score": cal_score,
        "mean_separation": mean_sep,
        "std_separation": std_sep,
        "regime_results": regime_results,
        "criteria": criteria,
        "params": params,
    }


def check_acceptance(regime_results: Dict[str, Dict]) -> Dict[str, bool]:
    """Check all T9 acceptance criteria."""
    criteria = {}

    # Criterion 1: all regimes in [0.10, 0.35]
    all_in_range = True
    for name, scores in regime_results.items():
        if scores.get("_trivially_mild", False):
            continue
        gi_sep = scores.get("graph_informed", 0.0)
        if not (0.10 <= gi_sep <= 0.35):
            all_in_range = False
    criteria["all_regimes_in_range"] = all_in_range

    # Criterion 2: reroute_only > 0.05 at eb=0 in >= 2 regimes
    reroute_pass_count = 0
    for name, scores in regime_results.items():
        if "eb0" not in name:
            continue
        if scores.get("_trivially_mild", False):
            continue
        if scores.get("reroute_only", 0.0) > 0.05:
            reroute_pass_count += 1
    criteria["reroute_only_passes"] = reroute_pass_count >= 2

    # Criterion 3: expedite_only > 0.05 at eb=50k in >= 2 regimes
    expedite_pass_count = 0
    for name, scores in regime_results.items():
        if "eb50k" not in name:
            continue
        if scores.get("_trivially_mild", False):
            continue
        if scores.get("expedite_only", 0.0) > 0.05:
            expedite_pass_count += 1
    criteria["expedite_only_passes"] = expedite_pass_count >= 2

    # Criterion 4: graph_informed beats backlog_greedy by > 0.03 in >= 2 regimes
    gi_beats_bg_count = 0
    for name, scores in regime_results.items():
        if scores.get("_trivially_mild", False):
            continue
        gi = scores.get("graph_informed", 0.0)
        bg = scores.get("backlog_greedy", 0.0)
        if gi - bg > 0.03:
            gi_beats_bg_count += 1
    criteria["gi_beats_bg"] = gi_beats_bg_count >= 2

    # Criterion 5: no regime > 0.45
    no_trivial_domination = True
    for name, scores in regime_results.items():
        if scores.get("_trivially_mild", False):
            continue
        gi = scores.get("graph_informed", 0.0)
        if gi > 0.45:
            no_trivial_domination = False
    criteria["no_trivial_domination"] = no_trivial_domination

    criteria["all_pass"] = all(criteria.values())
    return criteria


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seeds = list(range(DEFAULT_SEEDS))
    T = DEFAULT_T

    print("=" * 70)
    print("Shock Architecture Calibration Search")
    print("=" * 70)

    # Step 1: Load validation winners
    print("\n--- Step 1: Loading validation winners ---")
    winners = load_validation_winners()
    print(f"  Winner params: {json.dumps(winners, indent=2)}")

    # Step 2: Evaluate winner candidate
    print("\n--- Step 2: Evaluating winner candidate ---")
    result = evaluate_t9(winners, seeds, T)
    cal_score = result["calibration_score"]
    criteria = result["criteria"]
    print(f"  Calibration score: {cal_score:.4f}")
    print(f"  Criteria: {json.dumps(criteria, indent=2)}")

    search_log = [{"iteration": 0, "params": winners, "cal_score": cal_score,
                    "criteria": criteria}]

    if criteria.get("all_pass", False):
        print("\n  Winner passes all criteria! No further search needed.")
    else:
        # Step 3: Local grid search
        print("\n--- Step 3: Local grid search ---")
        best_params = dict(winners)
        best_score = cal_score
        iteration = 1

        for param_name, grid_values in CALIBRATION_GRID.items():
            current_val = best_params.get(param_name)
            # Find neighbors in grid
            try:
                idx = grid_values.index(current_val)
            except ValueError:
                idx = 0

            neighbors = set()
            if idx > 0:
                neighbors.add(grid_values[idx - 1])
            if idx < len(grid_values) - 1:
                neighbors.add(grid_values[idx + 1])
            # Also try all grid values if there are only 2-3
            if len(grid_values) <= 3:
                neighbors = set(grid_values) - {current_val}

            for new_val in neighbors:
                candidate = dict(best_params)
                candidate[param_name] = new_val

                print(f"\n  Iteration {iteration}: {param_name}={new_val}")
                result = evaluate_t9(candidate, seeds, T)
                cs = result["calibration_score"]
                crit = result["criteria"]

                search_log.append({
                    "iteration": iteration,
                    "params": candidate,
                    "cal_score": cs,
                    "criteria": crit,
                })

                print(f"    cal_score={cs:.4f}, all_pass={crit.get('all_pass', False)}")

                if crit.get("all_pass", False):
                    print(f"\n  Found passing config at iteration {iteration}!")
                    best_params = candidate
                    best_score = cs
                    break
                elif cs > best_score:
                    best_params = candidate
                    best_score = cs

                iteration += 1

            # Early exit if we found a passing config
            if any(log["criteria"].get("all_pass", False) for log in search_log):
                break

    # Save search log
    log_df = pd.DataFrame([
        {"iteration": log["iteration"], "cal_score": log["cal_score"],
         **{f"param_{k}": v for k, v in log["params"].items()},
         **{f"crit_{k}": v for k, v in log["criteria"].items()}}
        for log in search_log
    ])
    log_df.to_csv(CALIBRATION_CSV, index=False)
    print(f"\nSearch log saved to {CALIBRATION_CSV}")

    # Save best config
    best_cfg = build_config_from_params(
        search_log[-1]["params"] if search_log[-1]["criteria"].get("all_pass") else
        max(search_log, key=lambda x: x["cal_score"])["params"]
    )
    config_path = os.path.join(OUTPUT_DIR, "calibrated_config.json")
    with open(config_path, "w") as f:
        json.dump(best_cfg.to_dict(), f, indent=2)
    print(f"Best config saved to {config_path}")

    # Diagnostic if no config passes
    any_pass = any(log["criteria"].get("all_pass", False) for log in search_log)
    if not any_pass:
        print("\n=== DIAGNOSTIC: No config passed all criteria ===")
        # Find hardest criterion
        crit_fail_counts = {}
        for log in search_log:
            for k, v in log["criteria"].items():
                if k == "all_pass":
                    continue
                if not v:
                    crit_fail_counts[k] = crit_fail_counts.get(k, 0) + 1
        if crit_fail_counts:
            hardest = max(crit_fail_counts, key=crit_fail_counts.get)
            print(f"  Hardest criterion: {hardest} (failed {crit_fail_counts[hardest]}/{len(search_log)} times)")
        print("  Suggestions:")
        print("  - Increase shock_prob or magnitude_mean for more disruption")
        print("  - Increase duration_mean for longer shocks")
        print("  - Try more extreme capacity_cv values")
        print("  - Expand grid to include magnitude_mean as a search axis")

    print("\n=== Calibration search complete ===")


if __name__ == "__main__":
    main()
