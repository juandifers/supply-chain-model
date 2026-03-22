"""
Run 4 policies x 3 seeds x 1 config to verify policy differentiation.
Also times each run to estimate full experiment duration.
Should complete in under 2 minutes.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env
from scripts.baseline_policies import (
    no_intervention_policy,
    make_backlog_only_policy,
    make_threshold_policy,
    make_mip_policy,
)
from scripts.graph_informed_optimizer import make_graph_informed_policy
import numpy as np


# --- Calibrated config from quick_calibrate.py ---
CONFIG = {
    "default_supply": 1e6,
    "shock_magnitude": 0.7,
    "firm_shock_fraction": 0.7,
    "shock_prob": 0.1,
    "recovery_rate": 1.25,
    "warmup_steps": 15,
    "init_inv": 0,
    "T": 60,
    "expedite_budget": 5000,
    "expedite_m_max": 3.0,
}

K = 10  # reroute budget

PILOT_POLICIES = {
    "no_intervention": no_intervention_policy,
    "threshold": make_threshold_policy(reroute_budget_K=K),
    "backlog_only": make_backlog_only_policy(reroute_budget_K=K),
    "graph_informed": make_graph_informed_policy(reroute_budget_K=K),
    "mip": make_mip_policy(reroute_budget_K=K),
}

PILOT_SEEDS = 3


def run_one(policy_fn, seed, config):
    env, obs, _ = create_calibrated_env(
        seed=seed,
        T=config["T"],
        default_supply=config["default_supply"],
        shock_magnitude=config["shock_magnitude"],
        shock_prob=config["shock_prob"],
        recovery_rate=config["recovery_rate"],
        firm_shock_fraction=config["firm_shock_fraction"],
        warmup_steps=config["warmup_steps"],
        init_inv=config["init_inv"],
        expedite_budget=config["expedite_budget"],
        expedite_m_max=config["expedite_m_max"],
    )
    done = False
    while not done:
        action, _ = policy_fn(obs, env.t, env)
        obs, _, done, _ = env.step(action)

    kpi = env.get_kpi_history()
    return {
        "fill": float(kpi["consumer_cumulative_fill_rate"].iloc[-1]),
        "auc": float(kpi["consumer_backlog_units"].sum()),
        "peak": float(kpi["consumer_backlog_units"].max()),
        "spend": float(kpi["expedite_cost_cum"].iloc[-1]),
        "reroutes": int(kpi["reroutes_cumulative"].iloc[-1]),
    }


def main():
    print("=" * 90)
    print("MICRO-PILOT: 5 policies x 3 seeds")
    print("=" * 90)
    print(f"Config: {CONFIG}")
    print(f"Reroute budget K={K}")
    print()

    results = {}  # {policy: [result_dicts]}
    timings = {}  # {policy: [seconds]}

    for policy_name, policy_fn in PILOT_POLICIES.items():
        results[policy_name] = []
        timings[policy_name] = []
        for seed in range(PILOT_SEEDS):
            t0 = time.time()
            r = run_one(policy_fn, seed, CONFIG)
            elapsed = time.time() - t0
            results[policy_name].append(r)
            timings[policy_name].append(elapsed)
            print(f"  {policy_name:20s} seed={seed}  auc={r['auc']:10.0f}  fill={r['fill']:.3f}  "
                  f"spend={r['spend']:.0f}  reroutes={r['reroutes']:3d}  ({elapsed:.2f}s)")

    # --- Aggregate results ---
    print(f"\n{'=' * 90}")
    print("AGGREGATE RESULTS (mean over 3 seeds)")
    print(f"{'=' * 90}")
    print(f"{'policy':>20s}  {'auc':>10s}  {'fill':>6s}  {'spend':>8s}  {'reroutes':>8s}  {'time_s':>7s}")
    print("-" * 70)

    policy_means = {}
    for policy_name in PILOT_POLICIES:
        aucs = [r["auc"] for r in results[policy_name]]
        fills = [r["fill"] for r in results[policy_name]]
        spends = [r["spend"] for r in results[policy_name]]
        reroutes = [r["reroutes"] for r in results[policy_name]]
        avg_time = np.mean(timings[policy_name])

        mean_auc = np.mean(aucs)
        policy_means[policy_name] = mean_auc

        print(f"{policy_name:>20s}  {mean_auc:10.0f}  {np.mean(fills):6.3f}  "
              f"{np.mean(spends):8.0f}  {np.mean(reroutes):8.0f}  {avg_time:7.2f}s")

    # --- Differentiation check ---
    print(f"\n{'=' * 90}")
    print("DIFFERENTIATION CHECK")
    print(f"{'=' * 90}")

    ni_auc = policy_means["no_intervention"]
    for name in ["threshold", "backlog_only", "graph_informed", "mip"]:
        delta = (ni_auc - policy_means[name]) / max(ni_auc, 1) * 100
        print(f"  {name:20s} vs no_intervention: {delta:+.1f}%")

    gi_auc = policy_means["graph_informed"]
    for name in ["threshold", "backlog_only", "mip"]:
        delta = (gi_auc - policy_means[name]) / max(gi_auc, 1) * 100
        sign = "better" if delta > 0 else "worse"
        print(f"  {name:20s} vs graph_informed:   {delta:+.1f}% ({sign})")

    # Gate check
    gi_delta = (ni_auc - gi_auc) / max(ni_auc, 1) * 100
    if gi_delta > 10:
        print(f"\n  GATE 1 PASSED: graph_informed beats no_intervention by {gi_delta:.1f}% (>10%)")
    else:
        print(f"\n  GATE 1 FAILED: graph_informed only beats no_intervention by {gi_delta:.1f}% (need >10%)")

    # --- Compute estimation ---
    print(f"\n{'=' * 90}")
    print("COMPUTE ESTIMATION (based on pilot timings)")
    print(f"{'=' * 90}")

    avg_time_per_run = {p: np.mean(t) for p, t in timings.items()}

    estimated_per_run = {
        "no_intervention": avg_time_per_run["no_intervention"],
        "random_reroute": avg_time_per_run["no_intervention"] * 1.2,
        "threshold": avg_time_per_run["threshold"],
        "backlog_only": avg_time_per_run["backlog_only"],
        "graph_informed": avg_time_per_run["graph_informed"],
        "mip": avg_time_per_run["mip"],
        "reroute_only": avg_time_per_run["graph_informed"],
        "expedite_only": avg_time_per_run["graph_informed"],
    }

    FULL_POLICIES = 8
    FULL_SEEDS = 20
    # Regime grid: firm_shock_fraction x shock_fraction x structural configs
    FIRM_SHOCK_FRACTIONS = [0.3, 0.5, 0.7, 1.0]
    SHOCK_PROBS = [0.05, 0.1, 0.15, 0.2]
    STRUCTURAL_CONFIGS = 1  # single graph structure for now
    FULL_CONFIGS = len(FIRM_SHOCK_FRACTIONS) * len(SHOCK_PROBS) * STRUCTURAL_CONFIGS

    total_runs = FULL_POLICIES * FULL_SEEDS * FULL_CONFIGS
    total_seconds_serial = sum(
        estimated_per_run[p] * FULL_SEEDS * FULL_CONFIGS
        for p in estimated_per_run
    )

    print(f"\nTotal experiment runs: {total_runs}")
    print(f"Grid: {len(FIRM_SHOCK_FRACTIONS)} firm_shock_fracs x {len(SHOCK_PROBS)} shock_probs x {STRUCTURAL_CONFIGS} structs = {FULL_CONFIGS} configs")
    print(f"\nAverage time per run by policy:")
    for policy, t in sorted(estimated_per_run.items(), key=lambda x: x[1]):
        print(f"  {policy:25s}: {t:.2f}s")

    print(f"\nEstimated total wall-clock time:")
    for workers in [1, 2, 4, 8]:
        hours = total_seconds_serial / workers / 3600
        print(f"  {workers} workers: ~{hours:.1f} hours")

    # Gate check
    hours_4w = total_seconds_serial / 4 / 3600
    if hours_4w > 12:
        print(f"\n  GATE 2 WARNING: {hours_4w:.1f} hours with 4 workers exceeds 12h limit.")
        print(f"  Consider: fewer seeds, skip MIP, or reduce grid.")
    else:
        print(f"\n  GATE 2 PASSED: {hours_4w:.1f} hours with 4 workers (< 12h)")

    print(f"\nRecommendation: use --workers 4 and run overnight.")
    if avg_time_per_run.get("mip", 0) > 2.0:
        print(f"MIP is slow ({avg_time_per_run['mip']:.1f}s/run). Consider --skip-policies mip for fast first pass.")


if __name__ == "__main__":
    main()
