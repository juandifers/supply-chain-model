"""
Calibration sweep: finds simulation parameters that produce meaningful policy differentiation.

Steps:
1. Find default_supply that gives fill_rate ~0.9 with NO shocks
2. Verify shocks create real shortages
3. Verify rerouting helps (firm-level shocks)
4. Verify expediting helps
5. Verify policy ordering

Usage:
    ./venv/bin/python scripts/calibration_sweep.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env
from scripts.baseline_policies import (
    no_intervention_policy,
    make_random_reroute_policy,
    make_backlog_only_policy,
    make_expedite_only_policy,
    make_reroute_only_policy,
)
from scripts.graph_informed_optimizer import make_graph_informed_policy


def run_calibrated_policy(policy_fn, policy_name, seed, T=60, **cal_kwargs):
    """Run a single episode with calibrated scenario. Returns summary dict."""
    env, obs, shock_log = create_calibrated_env(seed=seed, T=T, **cal_kwargs)

    done = False
    t = 0
    while not done:
        action, explanation = policy_fn(obs, t, env)
        obs, reward, done, info = env.step(action)
        t += 1

    kpi_df = env.get_kpi_history()
    if len(kpi_df) == 0:
        return {"policy_name": policy_name, "seed": seed, "backlog_auc": 0, "fill_rate": 0, "mean_fill_rate": 0}

    backlog_auc = float(kpi_df["consumer_backlog_units"].sum())
    final_fill = float(kpi_df["consumer_cumulative_fill_rate"].iloc[-1])
    mean_fill = float(kpi_df["consumer_new_demand_fill_rate"].mean())
    total_spend = float(kpi_df["expedite_cost_cum"].iloc[-1])
    total_reroutes = int(kpi_df["reroutes_cumulative"].iloc[-1])

    return {
        "policy_name": policy_name,
        "seed": seed,
        "backlog_auc": backlog_auc,
        "peak_backlog": float(kpi_df["consumer_backlog_units"].max()),
        "fill_rate": final_fill,
        "mean_fill_rate": mean_fill,
        "total_spend": total_spend,
        "total_reroutes": total_reroutes,
        "num_shocks": len(shock_log),
        "kpi_df": kpi_df,
    }


def step1_find_default_supply(T=60, warmup_steps=15, seed=42):
    """Binary search for default_supply that gives fill_rate ~0.9 with NO shocks."""
    print("\n" + "=" * 70)
    print("STEP 1: Finding default_supply for fill_rate ~0.85-0.95 (no shocks)")
    print("=" * 70)

    candidates = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    results = []

    for ds in candidates:
        result = run_calibrated_policy(
            no_intervention_policy, "no_intervention", seed=seed,
            T=T, default_supply=ds, shock_prob=0.0, warmup_steps=warmup_steps,
        )
        results.append((ds, result["fill_rate"], result["mean_fill_rate"], result["backlog_auc"]))
        print(f"  default_supply={ds:>10,}  fill_rate={result['fill_rate']:.4f}  "
              f"mean_fill={result['mean_fill_rate']:.4f}  backlog_AUC={result['backlog_auc']:,.0f}")

    # Find the best candidate near 0.9
    best = None
    best_dist = float("inf")
    for ds, fr, mfr, auc in results:
        dist = abs(mfr - 0.90)
        if dist < best_dist:
            best_dist = dist
            best = ds

    print(f"\n  >> Best candidate: default_supply={best:,}")

    # Refine with binary search if needed
    # Find the two candidates that bracket 0.9
    below = [(ds, mfr) for ds, fr, mfr, auc in results if mfr < 0.90]
    above = [(ds, mfr) for ds, fr, mfr, auc in results if mfr >= 0.90]

    if below and above:
        lo = max(below, key=lambda x: x[1])[0]
        hi = min(above, key=lambda x: x[1])[0]
        print(f"  Refining between {lo:,} and {hi:,}...")

        for _ in range(5):
            mid = int((lo + hi) / 2)
            result = run_calibrated_policy(
                no_intervention_policy, "no_intervention", seed=seed,
                T=T, default_supply=mid, shock_prob=0.0, warmup_steps=warmup_steps,
            )
            mfr = result["mean_fill_rate"]
            print(f"  default_supply={mid:>10,}  mean_fill={mfr:.4f}")

            if mfr < 0.87:
                lo = mid
            elif mfr > 0.93:
                hi = mid
            else:
                best = mid
                break

            if abs(mfr - 0.90) < abs(best_dist):
                best = mid
                best_dist = abs(mfr - 0.90)

    print(f"\n  >> CALIBRATED default_supply = {best:,}")
    return best


def step2_verify_shocks(default_supply, T=60, warmup_steps=15, seed=42):
    """Verify shocks create real shortages at various shock_magnitude values."""
    print("\n" + "=" * 70)
    print("STEP 2: Verifying shocks create real shortages")
    print("=" * 70)

    results = []
    for sm in [0.1, 0.2, 0.3, 0.5, 0.7]:
        result = run_calibrated_policy(
            no_intervention_policy, "no_intervention", seed=seed,
            T=T, default_supply=default_supply, shock_magnitude=sm,
            shock_prob=0.15, firm_shock_fraction=0.5, warmup_steps=warmup_steps,
        )
        results.append((sf, result))
        print(f"  shock_magnitude={sm:.1f}  fill_rate={result['fill_rate']:.4f}  "
              f"mean_fill={result['mean_fill_rate']:.4f}  backlog_AUC={result['backlog_auc']:,.0f}  "
              f"shocks={result['num_shocks']}")

    return results


def step3_verify_rerouting(default_supply, T=60, warmup_steps=15, seed=42):
    """Verify rerouting helps under firm-level shocks."""
    print("\n" + "=" * 70)
    print("STEP 3: Verifying rerouting helps (firm-level shocks)")
    print("=" * 70)

    reroute_policy = make_reroute_only_policy(reroute_budget_K=3)

    for fsf in [0.3, 0.5, 0.7, 1.0]:
        cal_kwargs = dict(
            T=T, default_supply=default_supply, shock_magnitude=0.7,
            shock_prob=0.15, firm_shock_fraction=fsf, warmup_steps=warmup_steps,
        )
        no_int = run_calibrated_policy(
            no_intervention_policy, "no_intervention", seed=seed, **cal_kwargs
        )
        reroute = run_calibrated_policy(
            reroute_policy, "reroute_only", seed=seed, **cal_kwargs
        )

        if no_int["backlog_auc"] > 0:
            improvement = (no_int["backlog_auc"] - reroute["backlog_auc"]) / no_int["backlog_auc"] * 100
        else:
            improvement = 0.0

        print(f"  firm_shock_frac={fsf:.1f}  "
              f"no_int_auc={no_int['backlog_auc']:,.0f}  "
              f"reroute_auc={reroute['backlog_auc']:,.0f}  "
              f"improvement={improvement:+.1f}%")


def step4_verify_expediting(default_supply, T=60, warmup_steps=15, seed=42):
    """Verify expediting helps with various budget levels."""
    print("\n" + "=" * 70)
    print("STEP 4: Verifying expediting helps")
    print("=" * 70)

    cal_kwargs = dict(
        default_supply=default_supply, shock_magnitude=0.7,
        shock_prob=0.15, firm_shock_fraction=0.5, warmup_steps=warmup_steps,
    )

    # First measure the deficit with no intervention
    no_int = run_calibrated_policy(
        no_intervention_policy, "no_intervention", seed=seed, T=T, **cal_kwargs
    )
    baseline_auc = no_int["backlog_auc"]
    print(f"  No-intervention backlog AUC: {baseline_auc:,.0f}")

    for budget in [50, 200, 500, 1000, 5000, 10000, 50000]:
        exp_policy = make_expedite_only_policy()
        result = run_calibrated_policy(
            exp_policy, "expedite_only", seed=seed, T=T,
            expedite_budget=budget, **cal_kwargs,
        )
        if baseline_auc > 0:
            improvement = (baseline_auc - result["backlog_auc"]) / baseline_auc * 100
        else:
            improvement = 0.0
        print(f"  budget={budget:>8,}  "
              f"auc={result['backlog_auc']:,.0f}  "
              f"spend={result['total_spend']:,.0f}  "
              f"improvement={improvement:+.1f}%")


def step5_verify_policy_ordering(default_supply, T=60, warmup_steps=15,
                                  num_seeds=10, expedite_budget=500):
    """Run all policies across seeds and verify ordering."""
    print("\n" + "=" * 70)
    print(f"STEP 5: Verifying policy ordering ({num_seeds} seeds)")
    print("=" * 70)

    cal_kwargs = dict(
        default_supply=default_supply, shock_magnitude=0.7,
        shock_prob=0.15, firm_shock_fraction=0.5, warmup_steps=warmup_steps,
        expedite_budget=expedite_budget,
    )

    policies = {
        "no_intervention": no_intervention_policy,
        "random_reroute": make_random_reroute_policy(reroute_budget_K=3, seed=999),
        "backlog_only": make_backlog_only_policy(reroute_budget_K=3),
        "expedite_only": make_expedite_only_policy(),
        "reroute_only": make_reroute_only_policy(reroute_budget_K=3),
        "graph_informed": make_graph_informed_policy(reroute_budget_K=3),
    }

    all_results = {name: [] for name in policies}
    total = len(policies) * num_seeds
    i = 0

    for name, policy_fn in policies.items():
        for seed in range(num_seeds):
            i += 1
            t0 = time.time()
            result = run_calibrated_policy(
                policy_fn, name, seed=seed, T=T, **cal_kwargs
            )
            elapsed = time.time() - t0
            all_results[name].append(result)
            print(f"  [{i}/{total}] {name:25s} seed={seed}  "
                  f"auc={result['backlog_auc']:,.0f}  fill={result['fill_rate']:.3f}  "
                  f"({elapsed:.1f}s)")

    # Summary
    print("\n" + "-" * 70)
    print(f"{'Policy':25s}  {'Backlog AUC':>20s}  {'Fill Rate':>12s}")
    print("-" * 70)

    summary = {}
    for name in policies:
        aucs = [r["backlog_auc"] for r in all_results[name]]
        fills = [r["fill_rate"] for r in all_results[name]]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ci95 = 1.96 * std_auc / np.sqrt(len(aucs))
        mean_fill = np.mean(fills)
        summary[name] = {"mean_auc": mean_auc, "std_auc": std_auc, "ci95": ci95, "mean_fill": mean_fill}

    # Sort by mean AUC ascending (lower is better)
    no_int_auc = summary["no_intervention"]["mean_auc"]
    for name, stats in sorted(summary.items(), key=lambda x: x[1]["mean_auc"]):
        if no_int_auc > 0:
            pct = (no_int_auc - stats["mean_auc"]) / no_int_auc * 100
        else:
            pct = 0.0
        print(f"  {name:25s}  {stats['mean_auc']:10,.0f} ± {stats['ci95']:6,.0f}  "
              f"fill={stats['mean_fill']:.3f}  ({pct:+.1f}% vs none)")

    # Check key conditions
    print("\n" + "-" * 70)
    print("KEY CHECKS:")
    gi = summary.get("graph_informed", {})
    bo = summary.get("backlog_only", {})
    ni = summary.get("no_intervention", {})

    if ni["mean_auc"] > 0:
        sep = (ni["mean_auc"] - gi.get("mean_auc", ni["mean_auc"])) / ni["mean_auc"] * 100
        print(f"  graph_informed vs no_intervention: {sep:+.1f}% (target: ≥10%)")

        sep2 = (bo.get("mean_auc", 0) - gi.get("mean_auc", 0)) / bo.get("mean_auc", 1) * 100
        print(f"  graph_informed vs backlog_only:    {sep2:+.1f}% (target: ≥5%)")

    return summary, all_results


def main():
    T = 60
    warmup_steps = 15

    # Step 1
    default_supply = step1_find_default_supply(T=T, warmup_steps=warmup_steps)

    # Step 2
    step2_verify_shocks(default_supply, T=T, warmup_steps=warmup_steps)

    # Step 3
    step3_verify_rerouting(default_supply, T=T, warmup_steps=warmup_steps)

    # Step 4
    step4_verify_expediting(default_supply, T=T, warmup_steps=warmup_steps)

    # Step 5
    step5_verify_policy_ordering(default_supply, T=T, warmup_steps=warmup_steps, num_seeds=10)

    print("\n" + "=" * 70)
    print(f"CALIBRATION COMPLETE")
    print(f"Recommended: default_supply={default_supply:,}")
    print("=" * 70)
    print("\nReview results above before proceeding to full experiments.")


if __name__ == "__main__":
    main()
