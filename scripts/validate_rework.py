#!/usr/bin/env python
"""
Validation script for the simulator rework (6 priority fixes).
Runs V1-V6 from the prompt and prints results.
"""
import os
import sys
import time
import io
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv
from scripts.calibrated_scenario import create_calibrated_env
from scripts.graph_informed_optimizer import make_graph_informed_policy
from scripts.baseline_policies import (
    no_intervention_policy,
    make_backlog_only_policy,
    make_reroute_only_policy,
    make_expedite_only_policy,
)

PASS = "PASS"
FAIL = "FAIL"

# Calibrated parameters for the reworked simulator
CALIBRATED = dict(
    num_inner_layers=2,
    num_per_layer=10,
    min_num_suppliers=2,
    max_num_suppliers=3,
    min_inputs=1,
    max_inputs=2,
    min_units=1,
    max_units=1,
    max_order_age=10,
    kpi_start_step=4,
    init_demand=10,
    default_supply=100,
    recovery_rate=1.05,
    warmup_steps=10,
)


def quiet(func):
    """Run func with stdout suppressed."""
    f = io.StringIO()
    with redirect_stdout(f):
        return func()


def v1_reroute_mechanism():
    """V1: Reroute Mechanism Works"""
    print("\n" + "=" * 60)
    print("V1: REROUTE MECHANISM")
    print("=" * 60)

    def _run():
        env = SupplySimEnv(seed=42, T=30, **{k: v for k, v in CALIBRATED.items()
                                               if k not in ('init_demand', 'default_supply',
                                                            'recovery_rate', 'warmup_steps')})
        env.reset(init_demand=10, shock_prob=0.0, default_supply=100)

        for _ in range(5):
            env.step()

        for (buyer, product), supplier in env.inputs2supplier.items():
            old_key = (supplier, product)
            orders = env.curr_orders.get(old_key, [])
            buyer_orders = [o for o in orders if o[0] == buyer]
            alternatives = [s for s in env.prod2firms[product] if s != supplier]
            if buyer_orders and alternatives:
                old_sup, new_sup = supplier, alternatives[0]
                n_before = len(buyer_orders)

                # Snapshot orders BEFORE reroute
                action = {'reroute': [(buyer, product, new_sup)]}
                env.step(action)

                old_remaining = [o for o in env.curr_orders.get((old_sup, product), []) if o[0] == buyer]

                # After 3 more steps, new supplier should get orders from buyer
                for _ in range(3):
                    env.step()

                # Check supplier mapping is updated
                current_supplier = env.inputs2supplier.get((buyer, product))

                return {
                    "orders_transferred": len(old_remaining) == 0,
                    "supplier_updated": current_supplier == new_sup,
                    "n_before": n_before,
                }
        return None

    result = quiet(_run)
    if result is None:
        print("  Could not find suitable reroute test case")
        return FAIL

    print(f"  Orders transferred from old supplier: {result['orders_transferred']}")
    print(f"  Supplier mapping updated: {result['supplier_updated']}")
    print(f"  Orders before reroute: {result['n_before']}")
    status = PASS if result["orders_transferred"] and result["supplier_updated"] else FAIL
    print(f"  Result: {status}")
    return status


def v2_supply_demand_balance():
    """V2: Supply-Demand Balance"""
    print("\n" + "=" * 60)
    print("V2: SUPPLY-DEMAND BALANCE")
    print("=" * 60)

    BOM = {k: v for k, v in CALIBRATED.items()
           if k not in ('init_demand', 'default_supply', 'recovery_rate', 'warmup_steps')}

    # No-shock fill rate
    def _run():
        env = SupplySimEnv(seed=42, T=60, **BOM)
        env.reset(init_demand=CALIBRATED['init_demand'], shock_prob=0.0,
                  default_supply=CALIBRATED['default_supply'])
        for t in range(60):
            env.step()
        return env.get_kpi_history()

    kpis = quiet(_run)
    fill_rate = float(kpis["consumer_cumulative_fill_rate"].iloc[-1])
    backlog = float(kpis["consumer_backlog_units"].iloc[-1])

    print(f"  No-shock fill rate (post-warmup): {fill_rate:.3f}")
    print(f"  Final backlog: {backlog:.0f}")
    print(f"  Default supply: {CALIBRATED['default_supply']}")
    print(f"  Init demand: {CALIBRATED['init_demand']}")

    passed = 0.80 <= fill_rate <= 0.99
    status = PASS if passed else FAIL
    print(f"  Result: {status}")
    return status


def v3_shock_sensitivity():
    """V3: Shock Sensitivity"""
    print("\n" + "=" * 60)
    print("V3: SHOCK SENSITIVITY")
    print("=" * 60)

    def _run(shock_prob):
        aucs, fills = [], []
        for seed in range(3):
            def go(s=seed, sp=shock_prob):
                env, obs, logs = create_calibrated_env(
                    seed=s, T=60, shock_magnitude=0.7, shock_prob=sp,
                    expedite_budget=50000, **CALIBRATED)
                done = False
                while not done:
                    obs, _, done, _ = env.step()
                kpis = env.get_kpi_history()
                return (float(kpis["consumer_backlog_units"].sum()),
                        float(kpis["consumer_cumulative_fill_rate"].iloc[-1]))
            auc, fill = quiet(go)
            aucs.append(auc)
            fills.append(fill)
        return np.mean(aucs), np.mean(fills)

    auc_no, fill_no = _run(0.0)
    auc_sh, fill_sh = _run(0.15)

    fill_drop = fill_no - fill_sh
    auc_increase = (auc_sh - auc_no) / max(auc_no, 1) * 100

    print(f"  No-shock:  fill={fill_no:.3f}, backlog_auc={auc_no:.0f}")
    print(f"  Shocked:   fill={fill_sh:.3f}, backlog_auc={auc_sh:.0f}")
    print(f"  Fill rate drop: {fill_drop:.3f} ({fill_drop / max(fill_no, 0.001) * 100:.1f}%)")
    print(f"  Backlog AUC increase: {auc_increase:.1f}%")

    passed = auc_increase > 10
    status = PASS if passed else FAIL
    print(f"  Result: {status}")
    return status


def v4_policy_differentiation():
    """V4: Policy Differentiation"""
    print("\n" + "=" * 60)
    print("V4: POLICY DIFFERENTIATION")
    print("=" * 60)

    policies = {
        "no_intervention": no_intervention_policy,
        "graph_informed": make_graph_informed_policy(reroute_budget_K=3),
        "backlog_greedy": make_backlog_only_policy(reroute_budget_K=3),
        "reroute_only": make_reroute_only_policy(reroute_budget_K=3),
        "expedite_only": make_expedite_only_policy(),
    }

    results = {}
    for regime_name, sp, sm in [("mild", 0.05, 0.7), ("severe", 0.15, 0.7)]:
        results[regime_name] = {}
        for policy_name, policy_fn in policies.items():
            aucs, fills = [], []
            for seed in range(3):
                def _run(s=seed, pf=policy_fn, sp_=sp, sm_=sm):
                    env, obs, logs = create_calibrated_env(
                        seed=s, T=60, shock_magnitude=sm_, shock_prob=sp_,
                        expedite_budget=50000, **CALIBRATED)
                    done = False
                    while not done:
                        action, _ = pf(obs, env.t, env)
                        obs, _, done, _ = env.step(action)
                    kpis = env.get_kpi_history()
                    return (float(kpis["consumer_backlog_units"].sum()),
                            float(kpis["consumer_cumulative_fill_rate"].iloc[-1]))
                auc, fill = quiet(_run)
                aucs.append(auc)
                fills.append(fill)
            results[regime_name][policy_name] = {
                "mean_auc": np.mean(aucs), "mean_fill": np.mean(fills),
            }

    for regime in ["mild", "severe"]:
        print(f"\n  {regime.upper()} regime:")
        ni_auc = results[regime]["no_intervention"]["mean_auc"]
        for pname in policies:
            r = results[regime][pname]
            delta = (ni_auc - r["mean_auc"]) / max(ni_auc, 1) * 100
            print(f"    {pname:20s}: AUC={r['mean_auc']:8.0f}, fill={r['mean_fill']:.3f}, vs NI={delta:+.1f}%")

    # Checks
    severe_gi = results["severe"]["graph_informed"]["mean_auc"]
    severe_ni = results["severe"]["no_intervention"]["mean_auc"]
    diff_gi = (severe_ni - severe_gi) / max(severe_ni, 1) * 100

    severe_ro = results["severe"]["reroute_only"]["mean_auc"]
    severe_bg = results["severe"]["backlog_greedy"]["mean_auc"]
    ro_bg_diff = abs(severe_ro - severe_bg) / max(severe_bg, 1) * 100

    print(f"\n  graph_informed vs no_intervention (severe): {diff_gi:.1f}%")
    print(f"  reroute_only vs backlog_greedy diff: {ro_bg_diff:.1f}%")

    passed = diff_gi > 5
    status = PASS if passed else FAIL
    print(f"  Result: {status}")
    return status


def v5_order_expiry():
    """V5: Order Expiry Effect"""
    print("\n" + "=" * 60)
    print("V5: ORDER EXPIRY EFFECT")
    print("=" * 60)

    def _run(max_age):
        params = dict(CALIBRATED)
        params['max_order_age'] = max_age
        env, obs, logs = create_calibrated_env(
            seed=42, T=60, shock_magnitude=0.7, shock_prob=0.15,
            expedite_budget=50000, **params)
        done = False
        while not done:
            obs, _, done, _ = env.step()
        kpis = env.get_kpi_history()
        return list(kpis["consumer_backlog_units"])

    backlogs_with = quiet(lambda: _run(10))
    backlogs_without = quiet(lambda: _run(999))

    final_with = backlogs_with[-1]
    final_without = backlogs_without[-1]

    print(f"  With expiry (age=10):     final_backlog={final_with:.0f}")
    print(f"  Without expiry (age=999): final_backlog={final_without:.0f}")
    print(f"  Expiry reduces final backlog: {final_with < final_without}")

    passed = final_with <= final_without
    status = PASS if passed else FAIL
    print(f"  Result: {status}")
    return status


def v6_regime_variation():
    """V6: Regime Variation"""
    print("\n" + "=" * 60)
    print("V6: REGIME VARIATION (monotonic backlog increase with shock_prob)")
    print("=" * 60)

    shock_probs = [0.0, 0.05, 0.10, 0.15, 0.20]
    aucs = []

    for sp in shock_probs:
        def _run(sp_=sp):
            env, obs, logs = create_calibrated_env(
                seed=42, T=60, shock_magnitude=0.7, shock_prob=sp_,
                expedite_budget=50000, **CALIBRATED)
            done = False
            while not done:
                obs, _, done, _ = env.step()
            return float(env.get_kpi_history()["consumer_backlog_units"].sum())
        auc = quiet(_run)
        aucs.append(auc)
        print(f"  shock_prob={sp:.2f}: backlog_auc={auc:.0f}")

    monotonic = all(aucs[i] <= aucs[i + 1] for i in range(len(aucs) - 1))
    range_pct = (aucs[-1] - aucs[0]) / max(aucs[0], 1) * 100

    print(f"  Monotonic increase: {monotonic}")
    print(f"  Range: {range_pct:.1f}% ({aucs[0]:.0f} -> {aucs[-1]:.0f})")

    nearly_monotonic = sum(1 for i in range(len(aucs) - 1) if aucs[i] > aucs[i + 1] * 1.05) == 0
    passed = nearly_monotonic and range_pct > 5
    status = PASS if passed else FAIL
    print(f"  Result: {status}")
    return status


def main():
    print("=" * 60)
    print("SIMULATOR REWORK VALIDATION")
    print("=" * 60)

    t0 = time.time()
    results = {}

    results["V1_reroute"] = v1_reroute_mechanism()
    results["V2_supply_demand"] = v2_supply_demand_balance()
    results["V3_shock_sensitivity"] = v3_shock_sensitivity()
    results["V4_policy_differentiation"] = v4_policy_differentiation()
    results["V5_order_expiry"] = v5_order_expiry()
    results["V6_regime_variation"] = v6_regime_variation()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\n  Total time: {elapsed:.1f}s")

    all_passed = all(v == PASS for v in results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    print("\n" + "=" * 60)
    print("CALIBRATED PARAMETERS")
    print("=" * 60)
    for k, v in sorted(CALIBRATED.items()):
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
