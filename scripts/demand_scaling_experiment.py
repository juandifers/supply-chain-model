"""
Experiment: demand multiplier sweep at [0.5, 0.65, 0.8, 1.0].
Measures no-shock fill rate, shock sensitivity, and policy differentiation.
Uses calibrated shock settings: ds=500K, rr=1.05, warmup=10, T=60.
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
)
from scripts.graph_informed_optimizer import make_graph_informed_policy

DS = 500_000
RR = 1.05
WARMUP = 10
T = 60
K = 3
BUDGET = 50_000.0
DEMAND_MULTIPLIERS = [0.5, 0.65, 0.8, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def run_episode(policy_fn, seed, demand_multiplier, shock_prob=0.0,
                shock_magnitude=0.0, firm_shock_fraction=1.0):
    env, obs, shock_log = create_calibrated_env(
        seed=seed, default_supply=DS, shock_magnitude=shock_magnitude,
        shock_prob=shock_prob, firm_shock_fraction=firm_shock_fraction,
        recovery_rate=RR, warmup_steps=WARMUP, T=T, init_inv=0,
        expedite_budget=BUDGET, demand_multiplier=demand_multiplier,
    )
    for t_step in range(T):
        action, _ = policy_fn(obs, t_step, env)
        obs, _, done, info = env.step(action)
        if done:
            break
    kpi = env.get_kpi_history()
    return {
        "fill": kpi["consumer_cumulative_fill_rate"].iloc[-1],
        "auc": kpi["consumer_backlog_units"].sum(),
        "shocks": len(shock_log),
    }


def section1_fill_rate_ceiling():
    print("=" * 75)
    print("SECTION 1: No-shock fill rate ceiling by demand multiplier")
    print("=" * 75)
    print(f"{'dm':>6} {'seed':>5} {'fill':>8} {'auc':>12}")
    print("-" * 35)

    summary = {}
    for dm in DEMAND_MULTIPLIERS:
        fills, aucs = [], []
        for seed in SEEDS:
            r = run_episode(no_intervention_policy, seed, dm)
            fills.append(r["fill"])
            aucs.append(r["auc"])
            print(f"{dm:>6.2f} {seed:>5} {r['fill']:>8.4f} {r['auc']:>12,.0f}")
        summary[dm] = {"mean_fill": np.mean(fills), "std_fill": np.std(fills),
                       "mean_auc": np.mean(aucs), "std_auc": np.std(aucs)}

    print(f"\n{'dm':>6} {'mean_fill':>10} {'std_fill':>10} {'mean_auc':>12} {'std_auc':>10}")
    print("-" * 52)
    for dm, s in summary.items():
        print(f"{dm:>6.2f} {s['mean_fill']:>10.4f} {s['std_fill']:>10.4f} {s['mean_auc']:>12,.0f} {s['std_auc']:>10,.0f}")
    return summary


def section2_shock_sensitivity():
    print("\n" + "=" * 75)
    print("SECTION 2: Shock sensitivity by demand multiplier")
    print("  (sp=0.15, sf=0.3, fsf=0.5 vs no-shock baseline, 5 seeds)")
    print("=" * 75)
    print(f"{'dm':>6} {'baseline_auc':>14} {'shocked_auc':>14} {'delta%':>8} {'fill_ns':>8} {'fill_sh':>8}")
    print("-" * 72)

    summary = {}
    for dm in DEMAND_MULTIPLIERS:
        ns_aucs, sh_aucs, ns_fills, sh_fills = [], [], [], []
        for seed in SEEDS:
            r0 = run_episode(no_intervention_policy, seed, dm,
                           shock_prob=0.0)
            r1 = run_episode(no_intervention_policy, seed, dm,
                           shock_prob=0.15, shock_magnitude=0.7, firm_shock_fraction=0.5)
            ns_aucs.append(r0["auc"])
            sh_aucs.append(r1["auc"])
            ns_fills.append(r0["fill"])
            sh_fills.append(r1["fill"])

        mean_ns = np.mean(ns_aucs)
        mean_sh = np.mean(sh_aucs)
        delta = (mean_sh - mean_ns) / max(mean_ns, 1) * 100
        print(f"{dm:>6.2f} {mean_ns:>14,.0f} {mean_sh:>14,.0f} {delta:>+8.1f}% "
              f"{np.mean(ns_fills):>8.4f} {np.mean(sh_fills):>8.4f}")
        summary[dm] = {"baseline_auc": mean_ns, "shocked_auc": mean_sh,
                       "delta_pct": delta, "fill_ns": np.mean(ns_fills),
                       "fill_sh": np.mean(sh_fills)}

    # Also test severity sweep at each dm
    print(f"\n{'dm':>6} {'sm':>5} {'mean_auc':>12} {'delta%':>8}")
    print("-" * 35)
    for dm in DEMAND_MULTIPLIERS:
        r0 = np.mean([run_episode(no_intervention_policy, s, dm, shock_prob=0.0)["auc"]
                       for s in SEEDS])
        for sm in [0.1, 0.3, 0.5, 0.7]:
            r1 = np.mean([run_episode(no_intervention_policy, s, dm,
                         shock_prob=0.15, shock_magnitude=sm,
                         firm_shock_fraction=0.5)["auc"] for s in SEEDS])
            d = (r1 - r0) / max(r0, 1) * 100
            print(f"{dm:>6.2f} {sm:>5.1f} {r1:>12,.0f} {d:>+8.1f}%")
        print()

    return summary


def section3_policy_differentiation():
    print("=" * 75)
    print("SECTION 3: Policy differentiation by demand multiplier")
    print("  (severe regime: sp=0.20, sf=0.3, fsf=0.5, 5 seeds)")
    print("=" * 75)

    policies = {
        "no_intervention": no_intervention_policy,
        "backlog_greedy": make_backlog_only_policy(reroute_budget_K=K),
        "graph_informed": make_graph_informed_policy(reroute_budget_K=K),
        "reroute_only": make_reroute_only_policy(reroute_budget_K=K),
    }

    all_results = {}
    for dm in DEMAND_MULTIPLIERS:
        print(f"\n--- demand_multiplier = {dm} ---")
        print(f"  {'policy':<20} {'mean_auc':>12} {'std':>10} {'mean_fill':>10} {'vs NI':>8}")
        print("  " + "-" * 64)

        dm_results = {}
        ni_auc = None
        for pname, pfn in policies.items():
            aucs, fills = [], []
            for seed in SEEDS:
                r = run_episode(pfn, seed, dm,
                              shock_prob=0.20, shock_magnitude=0.7, firm_shock_fraction=0.5)
                aucs.append(r["auc"])
                fills.append(r["fill"])
            mean_auc = np.mean(aucs)
            if pname == "no_intervention":
                ni_auc = mean_auc
            delta = (ni_auc - mean_auc) / ni_auc * 100 if ni_auc else 0
            print(f"  {pname:<20} {mean_auc:>12,.0f} {np.std(aucs):>10,.0f} "
                  f"{np.mean(fills):>10.4f} {delta:>+8.1f}%")
            dm_results[pname] = {"mean_auc": mean_auc, "std_auc": np.std(aucs),
                                "mean_fill": np.mean(fills), "delta_vs_ni": delta}
        all_results[dm] = dm_results

    # Summary table
    print("\n" + "=" * 75)
    print("SUMMARY: graph_informed improvement vs no_intervention")
    print("=" * 75)
    print(f"{'dm':>6} {'NI_fill':>8} {'NI_auc':>12} {'GI_auc':>12} {'GI_delta':>10} {'BG_auc':>12} {'BG_delta':>10}")
    print("-" * 74)
    for dm in DEMAND_MULTIPLIERS:
        ni = all_results[dm]["no_intervention"]
        gi = all_results[dm]["graph_informed"]
        bg = all_results[dm]["backlog_greedy"]
        print(f"{dm:>6.2f} {ni['mean_fill']:>8.4f} {ni['mean_auc']:>12,.0f} "
              f"{gi['mean_auc']:>12,.0f} {gi['delta_vs_ni']:>+10.1f}% "
              f"{bg['mean_auc']:>12,.0f} {bg['delta_vs_ni']:>+10.1f}%")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    s1 = section1_fill_rate_ceiling()
    s2 = section2_shock_sensitivity()
    s3 = section3_policy_differentiation()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
