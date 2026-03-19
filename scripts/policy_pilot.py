"""
Phase 4: Policy pilot with calibrated parameters.
4 key policies × 3 seeds × 2 regimes.
Using: ds=500K, recovery_rate=1.05 (from escalation tests).
"""
import numpy as np
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
    make_reroute_only_policy,
    make_expedite_only_policy,
)
from scripts.graph_informed_optimizer import make_graph_informed_policy

CALIBRATED_SUPPLY = 500_000
RECOVERY_RATE = 1.05
WARMUP = 10
T = 60
REROUTE_K = 3
EXPEDITE_BUDGET = 50_000.0


def run_policy_episode(policy_fn, default_supply, shock_prob, shock_fraction,
                       firm_shock_fraction, seed, recovery_rate=RECOVERY_RATE):
    """Run a single episode with a given policy."""
    env, obs, shock_log = create_calibrated_env(
        seed=seed,
        default_supply=default_supply,
        shock_fraction=shock_fraction,
        shock_prob=shock_prob,
        firm_shock_fraction=firm_shock_fraction,
        warmup_steps=WARMUP,
        recovery_rate=recovery_rate,
        T=T,
        init_inv=0,
        init_supply=100,
        init_demand=1,
        expedite_budget=EXPEDITE_BUDGET,
    )

    for t_step in range(T):
        action, explanation = policy_fn(obs, t_step, env)
        obs, _, done, info = env.step(action)
        if done:
            break

    kpi_df = env.get_kpi_history()
    return {
        "fill_rate": kpi_df["consumer_cumulative_fill_rate"].iloc[-1],
        "backlog_auc": kpi_df["consumer_backlog_units"].sum(),
        "expedite_cost": kpi_df["expedite_cost_cum"].iloc[-1],
        "reroutes": kpi_df["reroutes_cumulative"].iloc[-1],
        "num_shocks": len(shock_log),
    }


def make_policies():
    """Create all policy functions."""
    return {
        "no_intervention": no_intervention_policy,
        "backlog_greedy": make_backlog_only_policy(reroute_budget_K=REROUTE_K),
        "graph_informed": make_graph_informed_policy(reroute_budget_K=REROUTE_K),
        "reroute_only": make_reroute_only_policy(reroute_budget_K=REROUTE_K),
        "expedite_only": make_expedite_only_policy(severity_threshold=0.1),
    }


def main():
    REGIMES = [
        {"name": "mild", "shock_prob": 0.10, "shock_fraction": 0.5, "firm_shock_fraction": 0.5},
        {"name": "severe", "shock_prob": 0.20, "shock_fraction": 0.3, "firm_shock_fraction": 0.5},
    ]
    SEEDS = [0, 1, 2]
    policies = make_policies()

    results = []
    total_runs = len(REGIMES) * len(policies) * len(SEEDS)
    run_count = 0

    for regime in REGIMES:
        for policy_name, policy_fn in policies.items():
            aucs = []
            fills = []
            costs = []
            for seed in SEEDS:
                run_count += 1
                r = run_policy_episode(
                    policy_fn,
                    default_supply=CALIBRATED_SUPPLY,
                    shock_prob=regime["shock_prob"],
                    shock_fraction=regime["shock_fraction"],
                    firm_shock_fraction=regime["firm_shock_fraction"],
                    seed=seed,
                )
                aucs.append(r["backlog_auc"])
                fills.append(r["fill_rate"])
                costs.append(r["expedite_cost"])
                if run_count % 5 == 0:
                    print(f"  [{run_count}/{total_runs}] done...")

            results.append({
                "regime": regime["name"],
                "policy": policy_name,
                "mean_auc": np.mean(aucs),
                "std_auc": np.std(aucs),
                "mean_fill": np.mean(fills),
                "mean_cost": np.mean(costs),
                "aucs": aucs,
            })

    # Print comparison table
    print("\n" + "=" * 90)
    print("POLICY PILOT RESULTS")
    print(f"ds={CALIBRATED_SUPPLY:,}, rr={RECOVERY_RATE}, warmup={WARMUP}, T={T}, K={REROUTE_K}, budget={EXPEDITE_BUDGET:,.0f}")
    print("=" * 90)
    print(f"{'Regime':<10} {'Policy':<20} {'Mean AUC':>12} {'Std AUC':>10} {'Mean Fill':>10} {'Exp Cost':>10}")
    print("-" * 76)
    for r in results:
        print(f"{r['regime']:<10} {r['policy']:<20} {r['mean_auc']:>12,.0f} {r['std_auc']:>10,.0f} {r['mean_fill']:>10.4f} {r['mean_cost']:>10,.0f}")

    # Key comparisons
    print("\n" + "=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)

    for regime_name in ["mild", "severe"]:
        ni = [r for r in results if r['regime'] == regime_name and r['policy'] == 'no_intervention'][0]
        print(f"\n--- {regime_name.upper()} regime ---")
        print(f"  no_intervention baseline AUC: {ni['mean_auc']:,.0f}")
        for r in results:
            if r['regime'] != regime_name or r['policy'] == 'no_intervention':
                continue
            delta = (ni['mean_auc'] - r['mean_auc']) / ni['mean_auc'] * 100
            print(f"  {r['policy']:<20}: {r['mean_auc']:>10,.0f}  ({delta:+.1f}% vs baseline)")

    # Regime effect
    ni_mild = [r for r in results if r['regime'] == 'mild' and r['policy'] == 'no_intervention'][0]
    ni_severe = [r for r in results if r['regime'] == 'severe' and r['policy'] == 'no_intervention'][0]
    regime_effect = (ni_severe['mean_auc'] - ni_mild['mean_auc']) / ni_mild['mean_auc'] * 100
    print(f"\nRegime effect on baseline: severe vs mild = {regime_effect:+.1f}%")
    print("(Must be positive and >10% for shocks to matter)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
