"""
GOAL: Determine the actual exogenous supply consumption rate of the pipeline.
Run ONE episode with UNLIMITED supply and NO shocks, measure how much exogenous
supply is actually consumed per timestep.

Exogenous supply is a per-timestep throughput cap (not a stockpile).
simulate_actions_for_firm serves orders FIFO from exog_supp[(f,p)] until exhausted.
Consumption = total transaction volume for exogenous products.
"""
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env


def measure_pipeline():
    UNLIMITED = 10_000_000
    T = 60
    SEED = 42

    env, obs, _ = create_calibrated_env(
        seed=SEED,
        default_supply=UNLIMITED,
        shock_fraction=1.0,  # No shock effect
        shock_prob=0.0,      # No shocks
        firm_shock_fraction=1.0,
        warmup_steps=0,
        T=T,
        init_inv=0,
        init_supply=100,
        init_demand=1,
    )

    # Identify exogenous products and their (firm, product) pairs
    exog_pairs = set()
    for t_idx in range(T):
        for key in env.exog_schedule[t_idx]:
            f, p = key
            if p in env.exog_prods:
                exog_pairs.add(key)

    print(f"\nExogenous products: {env.exog_prods}")
    print(f"Number of exog (firm, product) pairs: {len(exog_pairs)}")

    consumption_by_step = []

    for t in range(T):
        # Record available supply before step
        total_available = 0.0
        supply_before = {}
        for key in exog_pairs:
            amt = env.exog_schedule[t].get(key, 0)
            supply_before[key] = amt
            total_available += amt

        # Step with no intervention
        obs, reward, done, info = env.step({"reroute": [], "supply_multiplier": {}})
        txns_df = info["transactions"]

        # Measure consumption: sum of transaction amounts for exogenous products
        total_consumed = 0.0
        if len(txns_df) > 0:
            # product_id is an index into env.products
            for _, row in txns_df.iterrows():
                prod = env.products[int(row["product_id"])]
                if prod in env.exog_prods:
                    total_consumed += float(row["amount"])

        consumption_by_step.append({
            't': t,
            'total_available': total_available,
            'total_consumed': total_consumed,
            'utilization': total_consumed / total_available if total_available > 0 else 0,
        })

        if done:
            break

    # Print results
    print(f"\n=== PIPELINE CONSUMPTION PROFILE ===")
    print(f"{'Step':>4} {'Available':>14} {'Consumed':>12} {'Utilization':>10}")
    print("-" * 44)
    for row in consumption_by_step:
        print(f"{row['t']:>4} {row['total_available']:>14,.0f} {row['total_consumed']:>12,.0f} {row['utilization']:>10.6f}")

    # Steady-state consumption (steps 20-50)
    steady_state = consumption_by_step[20:50]
    if len(steady_state) == 0:
        print("ERROR: Not enough steps for steady state analysis")
        return

    avg_consumption = np.mean([r['total_consumed'] for r in steady_state])
    max_consumption = np.max([r['total_consumed'] for r in steady_state])
    std_consumption = np.std([r['total_consumed'] for r in steady_state])
    cv = std_consumption / avg_consumption if avg_consumption > 0 else float('inf')

    print(f"\n=== CALIBRATION ANCHOR ===")
    print(f"Steady-state avg consumption: {avg_consumption:,.0f} units/step")
    print(f"Steady-state max consumption: {max_consumption:,.0f} units/step")
    print(f"Consumption CV (coefficient of variation): {cv:.2%}")
    print(f"Number of exogenous (firm, product) pairs: {len(exog_pairs)}")

    num_exog_pairs = len(exog_pairs)
    per_pair_consumption = avg_consumption / max(num_exog_pairs, 1)
    print(f"Avg consumption per (firm, product): {per_pair_consumption:,.0f}")

    # Sanity checks
    avg_util = np.mean([r['utilization'] for r in steady_state])
    print(f"\n=== SANITY CHECKS ===")
    print(f"Avg utilization: {avg_util:.6%}")
    if avg_util > 0.01:
        print("WARNING: Utilization too high — supply may not have been effectively unlimited")
    else:
        print("OK: Supply was effectively unlimited (utilization << 1%)")

    if cv > 0.5:
        print(f"WARNING: Consumption is highly variable (CV={cv:.2%})")
    else:
        print(f"OK: Consumption is stable (CV={cv:.2%})")

    if avg_consumption < 1:
        print("ERROR: Pipeline not consuming any exogenous supply")
    else:
        print(f"OK: Pipeline is active (avg={avg_consumption:,.0f} units/step)")

    # Also track inventory growth
    total_inv = np.sum(env.inventories)
    print(f"\nFinal total inventory: {total_inv:,.0f}")

    # CALIBRATION RECOMMENDATIONS
    print(f"\n=== RECOMMENDED SUPPLY LEVELS ===")
    print(f"(per (firm, product) pair, based on avg consumption = {per_pair_consumption:,.0f})")
    print(f"{'Overcapacity':>12} {'default_supply':>15} | {'sf=0.3':>12} {'sf=0.5':>12} {'sf=0.7':>12}")
    print("-" * 72)
    for overcapacity in [1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.00]:
        ds = int(per_pair_consumption * overcapacity)
        parts = []
        for sf in [0.3, 0.5, 0.7]:
            shocked = int(ds * sf)
            ratio = shocked / per_pair_consumption if per_pair_consumption > 0 else 0
            status = "SHORT" if ratio < 0.95 else "OK" if ratio < 1.05 else "SURP"
            parts.append(f"{shocked:>6} ({ratio:.0%} {status})")
        print(f"  {overcapacity:>10.0%} {ds:>15,} | {'  '.join(parts)}")


def binary_search_supply():
    """Fallback: find supply level where fill rate transitions."""
    print("\n\n=== BINARY SEARCH: Supply level vs Fill Rate ===")
    T = 60
    SEED = 42

    print(f"{'default_supply':>14} {'fill_rate':>10} {'backlog_auc':>12} {'total_txn_units':>15}")
    print("-" * 55)

    for ds in [10_000_000, 1_000_000, 500_000, 200_000, 100_000, 50_000,
               20_000, 10_000, 5_000, 2_000, 1_000, 500, 200, 100, 50]:
        env, obs, _ = create_calibrated_env(
            seed=SEED,
            default_supply=ds,
            shock_fraction=1.0,
            shock_prob=0.0,
            firm_shock_fraction=1.0,
            warmup_steps=0,
            T=T,
            init_inv=0,
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
        print(f"{ds:>14,} {fill:>10.4f} {backlog_auc:>12,.0f} {total_txn:>15,.0f}")


if __name__ == "__main__":
    measure_pipeline()
    binary_search_supply()
