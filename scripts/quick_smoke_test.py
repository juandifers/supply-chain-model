"""Minimal smoke test: import env, run reset/step, print state and KPI fields."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv


def main():
    env = SupplySimEnv(seed=0, T=10, gamma=0.8, expedite_budget=100.0,
                       expedite_c0=1.0, expedite_alpha=0.5, expedite_m_max=3.0)
    obs = env.reset(init_inv=0, init_supply=100, init_demand=1, shock_prob=0.01)

    print("=== RESET ===")
    print(f"obs keys: {list(obs.keys())}")
    print(f"inventories shape: {obs['inventories'].shape}")
    print(f"num firms: {len(env.firms)}, num products: {len(env.products)}")
    print(f"exog products: {env.exog_prods}")
    print(f"consumer products: {env.consumer_prods}")
    print(f"inputs2supplier entries: {len(env.inputs2supplier)}")
    print(f"prod2firms sample: {list(env.prod2firms.items())[:3]}")
    print(f"expedite_budget_remaining: {env.expedite_budget_remaining}")

    # Step with no action
    print("\n=== STEP (no action) ===")
    obs, reward, done, info = env.step(action=None)
    kpis = info["kpis"]
    print(f"KPI fields: {list(kpis.keys())}")
    print(f"t={kpis['t']}, txns={kpis['transactions']}, backlog={kpis['consumer_backlog_units']}")
    print(f"shock_exposure={kpis['shock_exposure']}, worst={kpis['worst_shocked_product']}")

    # Step with a reroute action
    print("\n=== STEP (reroute) ===")
    # Find a reroutable (buyer, product) pair
    sample_key = list(env.inputs2supplier.keys())[0]
    buyer, product = sample_key
    current_supplier = env.inputs2supplier[sample_key]
    alt_suppliers = [s for s in env.prod2firms[product] if s != current_supplier]
    if alt_suppliers:
        action = {"reroute": [(buyer, product, alt_suppliers[0])]}
        obs, reward, done, info = env.step(action=action)
        print(f"Rerouted ({buyer}, {product}): {current_supplier} -> {alt_suppliers[0]}")
        print(f"reroutes_applied={info['kpis']['reroutes_applied']}")
    else:
        print(f"No alternative suppliers for ({buyer}, {product})")

    # Step with expedite action
    print("\n=== STEP (expedite) ===")
    exog_key = list(env.exog_schedule[env.t].keys())[0]
    action = {"supply_multiplier": {exog_key: 2.0}}
    obs, reward, done, info = env.step(action=action)
    print(f"Expedited {exog_key} with multiplier 2.0")
    print(f"expedite_cost_t={info['kpis']['expedite_cost_t']}")
    print(f"expedite_budget_remaining={info['kpis']['expedite_budget_remaining']}")

    # Run remaining steps
    print("\n=== REMAINING STEPS ===")
    while not done:
        obs, reward, done, info = env.step(action=None)
    print(f"Episode done at t={info['kpis']['t']}")
    print(f"Final backlog={info['kpis']['consumer_backlog_units']}")
    print(f"Cumulative fill rate={info['kpis']['consumer_cumulative_fill_rate']}")
    print(f"Total expedite cost={info['kpis']['expedite_cost_cum']}")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
