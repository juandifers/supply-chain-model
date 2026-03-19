"""
Phase 2+3: Fine-grained supply sweep and shock sensitivity validation.
Find the supply level where shocks create meaningful backlog differences.
"""
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env


def run_episode(default_supply, shock_prob=0.0, shock_fraction=1.0,
                firm_shock_fraction=1.0, warmup_steps=15, T=60, seed=42,
                recovery_rate=1.25, init_inv=0):
    """Run a single episode and return summary stats."""
    env, obs, shock_log = create_calibrated_env(
        seed=seed,
        default_supply=default_supply,
        shock_fraction=shock_fraction,
        shock_prob=shock_prob,
        firm_shock_fraction=firm_shock_fraction,
        warmup_steps=warmup_steps,
        recovery_rate=recovery_rate,
        T=T,
        init_inv=init_inv,
        init_supply=100,
        init_demand=1,
    )
    for t in range(T):
        obs, _, done, info = env.step({"reroute": [], "supply_multiplier": {}})
        if done:
            break
    kpi_df = env.get_kpi_history()
    fill = kpi_df["consumer_cumulative_fill_rate"].iloc[-1]
    backlog_auc = kpi_df["consumer_backlog_units"].sum()
    total_txn = kpi_df["transaction_units"].sum()
    total_inv = np.sum(env.inventories)
    return {
        "fill_rate": fill,
        "backlog_auc": backlog_auc,
        "total_txn": total_txn,
        "total_inv": total_inv,
        "num_shocks": len(shock_log),
    }


def phase2_fine_sweep():
    """Fine-grained supply sweep around the transition zone (200K-600K)."""
    print("=" * 70)
    print("PHASE 2: Fine-grained supply sweep (no shocks)")
    print("=" * 70)
    print(f"{'default_supply':>14} {'fill_rate':>10} {'backlog_auc':>12} {'total_txn':>14} {'total_inv':>12}")
    print("-" * 66)

    for ds in [250_000, 300_000, 350_000, 400_000, 450_000, 500_000,
               600_000, 750_000, 1_000_000]:
        r = run_episode(ds, shock_prob=0.0, warmup_steps=0, init_inv=0)
        print(f"{ds:>14,} {r['fill_rate']:>10.4f} {r['backlog_auc']:>12,.0f} {r['total_txn']:>14,.0f} {r['total_inv']:>12,.0f}")


def phase3_shock_sensitivity():
    """Test if shocks matter at various supply levels."""
    print("\n" + "=" * 70)
    print("PHASE 3: Shock sensitivity tests")
    print("=" * 70)

    # Test a few candidate supply levels
    for ds in [300_000, 400_000, 500_000, 750_000]:
        print(f"\n--- default_supply = {ds:,} ---")
        print(f"{'shock_prob':>10} {'sf':>5} {'fill_rate':>10} {'backlog_auc':>12} {'#shocks':>8}")
        print("-" * 50)

        # No shocks baseline
        r = run_episode(ds, shock_prob=0.0, shock_fraction=1.0, warmup_steps=10)
        print(f"{'0.00':>10} {'1.0':>5} {r['fill_rate']:>10.4f} {r['backlog_auc']:>12,.0f} {r['num_shocks']:>8}")
        baseline_auc = r["backlog_auc"]

        # Varying shock_prob with sf=0.3
        for sp in [0.05, 0.10, 0.15, 0.20]:
            r = run_episode(ds, shock_prob=sp, shock_fraction=0.3,
                           firm_shock_fraction=0.5, warmup_steps=10)
            delta = (r["backlog_auc"] - baseline_auc) / max(baseline_auc, 1) * 100
            print(f"{sp:>10.2f} {'0.3':>5} {r['fill_rate']:>10.4f} {r['backlog_auc']:>12,.0f} {r['num_shocks']:>8}  ({delta:+.1f}%)")

        # Varying shock severity with sp=0.15
        for sf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            r = run_episode(ds, shock_prob=0.15, shock_fraction=sf,
                           firm_shock_fraction=0.5, warmup_steps=10)
            delta = (r["backlog_auc"] - baseline_auc) / max(baseline_auc, 1) * 100
            print(f"{'0.15':>10} {sf:>5.1f} {r['fill_rate']:>10.4f} {r['backlog_auc']:>12,.0f} {r['num_shocks']:>8}  ({delta:+.1f}%)")


def phase3_inventory_tracking():
    """Track inventory buildup to understand buffering behavior."""
    print("\n" + "=" * 70)
    print("PHASE 3b: Inventory tracking over time (ds=400K, no shocks)")
    print("=" * 70)

    env, obs, _ = create_calibrated_env(
        seed=42, default_supply=400_000, shock_prob=0.0,
        shock_fraction=1.0, warmup_steps=0, T=60,
        init_inv=0, init_supply=100, init_demand=1,
    )
    print(f"{'t':>4} {'total_inv':>12} {'exog_inv':>12} {'backlog':>12} {'fill_rate':>10}")
    print("-" * 54)
    for t in range(60):
        obs, _, done, info = env.step({"reroute": [], "supply_multiplier": {}})
        total_inv = np.sum(env.inventories)
        # Exogenous product inventory
        exog_inv = 0.0
        for p in env.exog_prods:
            p_idx = env.prod2idx[p]
            exog_inv += np.sum(env.inventories[:, p_idx])
        kpis = info["kpis"]
        if t % 5 == 0 or t < 10:
            print(f"{t:>4} {total_inv:>12,.0f} {exog_inv:>12,.0f} {kpis['consumer_backlog_units']:>12,.1f} {kpis['consumer_cumulative_fill_rate']:>10.4f}")
        if done:
            break


def phase3_multi_seed():
    """Check consistency across seeds."""
    print("\n" + "=" * 70)
    print("PHASE 3c: Multi-seed consistency (ds=400K)")
    print("=" * 70)
    print(f"{'seed':>5} {'no-shock AUC':>14} {'sp=0.15 sf=0.3':>16} {'delta%':>8}")
    print("-" * 48)

    for seed in range(5):
        r0 = run_episode(400_000, shock_prob=0.0, warmup_steps=10, seed=seed)
        r1 = run_episode(400_000, shock_prob=0.15, shock_fraction=0.3,
                        firm_shock_fraction=0.5, warmup_steps=10, seed=seed)
        delta = (r1["backlog_auc"] - r0["backlog_auc"]) / max(r0["backlog_auc"], 1) * 100
        print(f"{seed:>5} {r0['backlog_auc']:>14,.0f} {r1['backlog_auc']:>16,.0f} {delta:>8.1f}%")


if __name__ == "__main__":
    phase2_fine_sweep()
    phase3_shock_sensitivity()
    phase3_inventory_tracking()
    phase3_multi_seed()
