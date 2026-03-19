"""
Test supply levels and parameters to find a good operating point.
Should complete in under 30 seconds.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env
from scripts.baseline_policies import no_intervention_policy
from scripts.graph_informed_optimizer import make_graph_informed_policy
import numpy as np


def run_config(seed, T, default_supply, shock_prob, shock_fraction, firm_shock_fraction,
               warmup_steps, init_inv, expedite_budget, K, policy_fn=None):
    env, obs, shock_log = create_calibrated_env(
        seed=seed, T=T, default_supply=default_supply,
        shock_fraction=shock_fraction, shock_prob=shock_prob,
        recovery_rate=1.25, firm_shock_fraction=firm_shock_fraction,
        warmup_steps=warmup_steps, init_inv=init_inv,
        expedite_budget=expedite_budget, expedite_m_max=3.0,
    )
    if policy_fn is None:
        policy_fn = no_intervention_policy

    done = False
    while not done:
        action, _ = policy_fn(obs, env.t, env)
        obs, _, done, _ = env.step(action)

    kpi = env.get_kpi_history()
    return {
        "fill": float(kpi["consumer_cumulative_fill_rate"].iloc[-1]),
        "auc": float(kpi["consumer_backlog_units"].sum()),
        "peak": float(kpi["consumer_backlog_units"].max()),
        "n_shocks": len(shock_log),
    }


def main():
    print("=" * 90)
    print("MICRO-CALIBRATION: Parameter sweep for policy differentiation")
    print("=" * 90)

    # Fixed params
    seed = 0
    T = 60
    default_supply = 1e6
    warmup = 15
    init_inv = 0

    print("\n--- Sweep: firm_shock_fraction x shock_prob x K x expedite_budget ---")
    print(f"{'firm_sf':>8s}  {'sp':>5s}  {'K':>3s}  {'eb':>6s}  "
          f"{'no_int_auc':>11s}  {'gi_auc':>10s}  {'delta%':>8s}  {'gi_fill':>8s}")
    print("-" * 80)

    configs = [
        (0.5, 0.10, 10, 5000),
        (0.5, 0.15, 10, 5000),
        (0.7, 0.10, 10, 5000),
        (0.5, 0.10,  5, 5000),
        (0.5, 0.10, 10, 2000),
        (0.3, 0.15, 10, 5000),
    ]

    best = None
    best_delta = 0

    for fsf, sp, K, eb in configs:
        r_ni = run_config(seed, T, default_supply, sp, 0.3, fsf, warmup, init_inv, eb, K)
        gi = make_graph_informed_policy(reroute_budget_K=K)
        r_gi = run_config(seed, T, default_supply, sp, 0.3, fsf, warmup, init_inv, eb, K, policy_fn=gi)

        delta = (r_ni["auc"] - r_gi["auc"]) / max(r_ni["auc"], 1) * 100
        print(f"{fsf:8.1f}  {sp:5.2f}  {K:3d}  {eb:6.0f}  "
              f"{r_ni['auc']:11.0f}  {r_gi['auc']:10.0f}  {delta:+7.1f}%  {r_gi['fill']:8.3f}")

        if delta > best_delta:
            best_delta = delta
            best = (fsf, sp, K, eb)

    if best:
        fsf, sp, K, eb = best
        print(f"\nBEST CONFIG: firm_shock_fraction={fsf}, shock_prob={sp}, K={K}, expedite_budget={eb}")
        print(f"             graph_informed vs no_intervention: {best_delta:+.1f}%")
    else:
        print("\nWARNING: No config showed meaningful differentiation")

    print(f"\n{'=' * 90}")
    print("RECOMMENDED DEFAULTS:")
    print(f"  default_supply = {default_supply:.0f}")
    print(f"  shock_fraction = 0.3")
    print(f"  firm_shock_fraction = {best[0] if best else 0.5}")
    print(f"  shock_prob = {best[1] if best else 0.1}")
    print(f"  reroute_budget_K = {best[2] if best else 10}")
    print(f"  expedite_budget = {best[3] if best else 5000}")
    print(f"  warmup_steps = {warmup}")
    print(f"  init_inv = {init_inv}")
    print(f"  T = {T}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
