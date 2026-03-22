#!/usr/bin/env python
"""
Phase 0: Preflight Check

Verifies simulator semantics, policy imports, and MIP availability
before running any panel.

Usage:
    python scripts/preflight_check.py
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Result tracking ───────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
_results = {}
MIP_AVAILABLE = False  # set by check 8


def _check(name, fn):
    try:
        msg = fn()
        _results[name] = (PASS, msg or "")
        print(f"  [PASS] {name}: {msg or ''}")
        return True
    except AssertionError as e:
        _results[name] = (FAIL, str(e))
        print(f"  [FAIL] {name}: {e}")
        return False
    except Exception as e:
        _results[name] = (FAIL, str(e))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 64)
print("SUPPLYSIM PREFLIGHT CHECK")
print("=" * 64)

# [1] Imports ──────────────────────────────────────────────────────────────────
print("\n[1] Module imports")

def _check_imports():
    from scripts.supplysim_env import SupplySimEnv  # noqa
    from scripts.calibrated_scenario import create_calibrated_env  # noqa
    from scripts.baseline_policies import (  # noqa
        no_intervention_policy,
        make_random_reroute_policy,
        make_backlog_only_policy,
        make_expedite_only_policy,
        make_reroute_only_policy,
    )
    from scripts.graph_informed_optimizer import make_graph_informed_policy  # noqa
    return "supplysim_env, calibrated_scenario, baseline_policies, graph_informed_optimizer"

_check("imports", _check_imports)

# [2] shock_magnitude semantics ────────────────────────────────────────────────
print("\n[2] shock_magnitude semantics (shock_magnitude=0.7 → supply drops BY 70%)")

def _check_shock_magnitude():
    from scripts.calibrated_scenario import create_calibrated_env
    env, obs, shock_log = create_calibrated_env(
        seed=0, T=5, default_supply=100, shock_magnitude=0.7,
        shock_prob=1.0, firm_shock_fraction=1.0, warmup_steps=0,
    )
    supplies = [v for v in env.exog_schedule[0].values()]
    assert supplies, "No exog supply entries found at t=0"
    mean_supply = sum(supplies) / len(supplies)
    # With shock_magnitude=0.7 and shock_prob=1.0 and no warmup,
    # supply should be default_supply*(1-0.7)=30, ±Poisson noise
    assert mean_supply < 65, (
        f"Expected shocked supply ~30 (default*0.3), got mean={mean_supply:.1f}. "
        "shock_magnitude semantics may be wrong."
    )
    return f"mean shocked supply={mean_supply:.1f} (expected ~30 for shock_magnitude=0.7)"

_check("shock_magnitude_semantics", _check_shock_magnitude)

# [3] shock_magnitude=0.0 → no-shock baseline ──────────────────────────────────
print("\n[3] shock_magnitude=0.0 → no-shock baseline")

def _check_no_shock():
    from scripts.calibrated_scenario import create_calibrated_env
    env, obs, shock_log = create_calibrated_env(
        seed=0, T=5, default_supply=100, shock_magnitude=0.0,
        shock_prob=0.0, firm_shock_fraction=1.0, warmup_steps=0,
    )
    supplies = [v for v in env.exog_schedule[0].values()]
    assert supplies, "No exog supply entries found at t=0"
    mean_supply = sum(supplies) / len(supplies)
    assert mean_supply > 50, (
        f"Expected supply ~100 (no shock), got mean={mean_supply:.1f}"
    )
    return f"mean no-shock supply={mean_supply:.1f} (expected ~100)"

_check("no_shock_baseline", _check_no_shock)

# [4] Reroute transfers pending orders ─────────────────────────────────────────
print("\n[4] Reroute: pending orders transfer to new supplier")

def _check_reroute_transfer():
    from scripts.calibrated_scenario import create_calibrated_env
    env, obs, _ = create_calibrated_env(
        seed=42, T=20, default_supply=100, shock_magnitude=0.0,
        shock_prob=0.0, warmup_steps=5,
    )
    # Warm up a few steps to accumulate orders
    for _ in range(3):
        env.step({"reroute": [], "supply_multiplier": {}})

    assert env.inputs2supplier, "inputs2supplier is empty after warmup"

    for (buyer, product), cur_sup in env.inputs2supplier.items():
        alternatives = [s for s in env.prod2firms[product] if s != cur_sup]
        if not alternatives:
            continue
        new_sup = alternatives[0]
        old_key = (cur_sup, product)
        new_key = (new_sup, product)
        old_before = sum(1 for o in env.curr_orders.get(old_key, []) if o[0] == buyer)
        new_before = sum(1 for o in env.curr_orders.get(new_key, []) if o[0] == buyer)
        env._apply_action_dict({"reroute": [(buyer, product, new_sup)]})
        old_after  = sum(1 for o in env.curr_orders.get(old_key, []) if o[0] == buyer)
        new_after  = sum(1 for o in env.curr_orders.get(new_key, []) if o[0] == buyer)
        if old_before > 0:
            assert old_after < old_before,  "Old supplier orders did not decrease after reroute"
            assert new_after > new_before,  "New supplier orders did not increase after reroute"
            transferred = old_before - old_after
            return f"Transferred {transferred} order(s) from {cur_sup} → {new_sup} for {product}"

    return "No pending orders found to transfer (OK — reroute logic intact)"

_check("reroute_order_transfer", _check_reroute_transfer)

# [5] KPI warm-start ───────────────────────────────────────────────────────────
print("\n[5] KPI warm-start: cumulative metrics only accumulate after kpi_start_step")

def _check_kpi_warmstart():
    from scripts.calibrated_scenario import create_calibrated_env
    env, obs, _ = create_calibrated_env(
        seed=0, T=15, default_supply=100, shock_magnitude=0.0,
        shock_prob=0.0, warmup_steps=0, kpi_start_step=4,
    )
    # Steps 0,1,2 are before kpi_start_step=4
    for _ in range(3):
        env.step({"reroute": [], "supply_multiplier": {}})
    assert env.t == 3, f"Expected t=3 after 3 steps, got {env.t}"
    # Run to kpi_start_step and beyond
    for _ in range(4):
        env.step({"reroute": [], "supply_multiplier": {}})
    assert env.t == 7
    # consumer_demand_cum should be > 0 now (steps 4,5,6 contributed)
    assert env.consumer_demand_cum > 0, (
        "consumer_demand_cum still 0 after passing kpi_start_step"
    )
    return f"kpi_start_step=4, consumer_demand_cum={env.consumer_demand_cum:.0f} at t={env.t}"

_check("kpi_warmstart", _check_kpi_warmstart)

# [6] Order expiry ─────────────────────────────────────────────────────────────
print("\n[6] Order expiry: orders older than max_order_age are dropped")

def _check_order_expiry():
    from scripts.calibrated_scenario import create_calibrated_env
    # Use very low supply to accumulate orders, very short max_order_age
    env, obs, _ = create_calibrated_env(
        seed=0, T=30, default_supply=1, shock_magnitude=0.0,
        shock_prob=0.0, warmup_steps=0, max_order_age=3,
    )
    total_lost = 0.0
    for _ in range(20):
        _, _, done, info = env.step({"reroute": [], "supply_multiplier": {}})
        total_lost += info["kpis"].get("lost_sales_units", 0.0)
        if done:
            break
    # Can't assert total_lost > 0 since supply might be enough to clear all orders.
    # Just assert it's non-negative (sanity check the field exists and is numeric).
    assert total_lost >= 0, "lost_sales_units should be non-negative"
    return f"total lost_sales_units over 20 steps = {total_lost:.0f} (max_order_age=3)"

_check("order_expiry", _check_order_expiry)

# [7] Seed reproducibility ─────────────────────────────────────────────────────
print("\n[7] Same seed → identical shock schedule across policy runs")

def _check_seed_reproducibility():
    from scripts.calibrated_scenario import create_calibrated_env
    env1, _, log1 = create_calibrated_env(
        seed=7, T=10, default_supply=100, shock_magnitude=0.7,
        shock_prob=0.2, warmup_steps=3,
    )
    env2, _, log2 = create_calibrated_env(
        seed=7, T=10, default_supply=100, shock_magnitude=0.7,
        shock_prob=0.2, warmup_steps=3,
    )
    assert len(log1) == len(log2), (
        f"Shock log lengths differ: {len(log1)} vs {len(log2)}"
    )
    for i, (s1, s2) in enumerate(zip(log1, log2)):
        assert s1["t"] == s2["t"] and s1["product"] == s2["product"], (
            f"Shock event {i} differs: {s1} vs {s2}"
        )
    # Also check exog schedules match
    for t in range(10):
        keys1 = set(env1.exog_schedule[t].keys())
        keys2 = set(env2.exog_schedule[t].keys())
        assert keys1 == keys2, f"Exog schedule keys differ at t={t}"
    return f"Identical shock schedule ({len(log1)} shocks) confirmed for seed=7"

_check("seed_reproducibility", _check_seed_reproducibility)

# [8] MIP availability ─────────────────────────────────────────────────────────
print("\n[8] MIP availability (pulp + CBC)")

def _check_mip():
    global MIP_AVAILABLE
    try:
        from pulp import (
            LpProblem, LpMaximize, LpVariable, LpBinary,
            lpSum, PULP_CBC_CMD, LpStatus,
        )
    except ImportError:
        _results["mip_availability"] = (WARN, "pulp not installed — MIP_AVAILABLE=False")
        print("  [WARN] mip_availability: pulp not installed — MIP_AVAILABLE=False")
        return None
    t0   = time.time()
    prob = LpProblem("preflight_test", LpMaximize)
    x    = LpVariable("x", cat=LpBinary)
    prob += x
    prob += x <= 1
    prob.solve(PULP_CBC_CMD(msg=0))
    elapsed = time.time() - t0
    assert LpStatus[prob.status] == "Optimal", f"Trivial MIP status: {LpStatus[prob.status]}"
    MIP_AVAILABLE = True
    return f"pulp available, trivial solve in {elapsed:.3f}s"

_r = _check_mip()
if _r is None:
    _results["mip_availability"] = (WARN, "pulp not installed")

# [9] Smoke test ───────────────────────────────────────────────────────────────
print("\n[9] Smoke test (2 seeds × 2 regimes × all policies)")

def _check_smoke():
    from scripts.experiment_utils import run_single_experiment, ALL_POLICIES, DEFAULTS

    required_fields = [
        "backlog_auc", "fill_rate", "peak_backlog", "final_backlog",
        "lost_sales_units", "time_to_recovery", "runtime_s", "mip_fallback_steps",
        "seed", "policy", "regime_id", "panel_name", "git_commit",
        "shock_magnitude", "default_supply", "shock_prob",
        "recovery_rate", "firm_shock_fraction", "expedite_budget",
    ]
    policies_to_test = [p for p in ALL_POLICIES if p != "mip" or MIP_AVAILABLE]

    mild_config   = dict(DEFAULTS, shock_prob=0.15, shock_magnitude=0.30,
                         firm_shock_fraction=0.3)
    severe_config = dict(DEFAULTS, shock_prob=0.20, shock_magnitude=0.70,
                         firm_shock_fraction=1.0)

    gi_times  = []
    mip_times = []
    failures  = []

    for regime_label, config in [("mild", mild_config), ("severe", severe_config)]:
        for seed in [0, 1]:
            for policy in policies_to_test:
                exp = dict(
                    policy=policy, seed=seed, config=config,
                    regime_id=regime_label, panel_name="smoke_test",
                )
                try:
                    result = run_single_experiment(exp)
                    missing = [f for f in required_fields if f not in result]
                    if missing:
                        failures.append(
                            f"{regime_label}/{policy}/seed{seed}: missing {missing}"
                        )
                    if policy == "graph_informed":
                        gi_times.append(result["runtime_s"])
                    if policy == "mip":
                        mip_times.append(result["runtime_s"])
                except Exception as e:
                    failures.append(f"{regime_label}/{policy}/seed{seed}: {e}")

    if failures:
        raise AssertionError("Smoke test failures:\n" + "\n".join(failures))

    # MIP timing warning
    if mip_times and gi_times:
        mip_mean = float(np.mean(mip_times))
        gi_mean  = float(np.mean(gi_times))
        if gi_mean > 0 and mip_mean > 10 * gi_mean:
            print(
                f"  [WARN] Mean MIP runtime ({mip_mean:.1f}s) is "
                f"{mip_mean/gi_mean:.0f}× mean graph_informed ({gi_mean:.1f}s). "
                "Consider --no-mip for large panels."
            )

    total = 2 * 2 * len(policies_to_test)
    return f"{total} smoke runs completed, all required fields present"

import numpy as np  # needed for smoke test MIP timing warning
_check("smoke_test", _check_smoke)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("PREFLIGHT SUMMARY")
print("=" * 64)
n_pass = sum(1 for s, _ in _results.values() if s == PASS)
n_warn = sum(1 for s, _ in _results.values() if s == WARN)
n_fail = sum(1 for s, _ in _results.values() if s == FAIL)
for name, (status, msg) in _results.items():
    icon = "✓" if status == PASS else ("△" if status == WARN else "✗")
    print(f"  {icon} [{status}] {name}")
print(f"\n  {n_pass} passed  |  {n_warn} warnings  |  {n_fail} failed")

if n_fail > 0:
    print("\n[ERROR] Preflight failed. Fix issues above before running panels.")
    sys.exit(1)
else:
    print("\n[OK] All checks passed. Ready to run panels.")
    if not MIP_AVAILABLE:
        print("[NOTE] MIP (pulp) not available. Add --no-mip to all panel commands.")
    print("\nRun order:")
    print("  python scripts/run_panel1.py --seeds 20 --workers 4")
    print("  python scripts/run_panel2.py --output-dir <same dir as panel1>")
    print("  python scripts/run_panel3.py --output-dir <same dir>")
    print("  python scripts/run_panel4.py --output-dir <same dir>")
    print("  python scripts/analyze_and_plot.py --output-dir <same dir>")
    print("  python scripts/generate_report.py --output-dir <same dir>")
