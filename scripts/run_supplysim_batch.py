# scripts/run_supplysim_batch.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from TGB.modules import synthetic_data as sd

def run(seed=0, T=50, gamma=0.8, debug=False):
    firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier = sd.generate_static_graphs(seed=seed)

    firm2idx = {f: i for i, f in enumerate(firms)}
    prod2idx = {p: i for i, p in enumerate(products)}

    inventories, curr_orders, exog_supp = sd.generate_initial_conditions(
        firms=firms,
        products=products,
        prod_graph=prod_graph,
        prod2firms=prod2firms,
        init_inv=0,
        init_supply=100,
        init_demand=1,
    )

    demand_schedule = sd.generate_demand_schedule(
        num_timesteps=T,
        prod_graph=prod_graph,
        prod2firms=prod2firms,
        seed=seed,
        min_demand=0.5,
        init_demand=2,
    )

    exog_schedule = sd.generate_exog_schedule_with_shocks(
        num_timesteps=T,
        prod_graph=prod_graph,
        prod2firms=prod2firms,
        seed=seed,
        default_supply=1e6,
        shock_supply=1000,
        shock_prob=0.001,
        recovery_rate=1.25,
    )

    txns = sd.generate_transactions(
        num_timesteps=T,
        inventories=inventories,
        curr_orders=curr_orders,
        exog_supp=exog_supp,
        firms=firms,
        products=products,
        firm2idx=firm2idx,
        prod2idx=prod2idx,
        prod_graph=prod_graph,
        firm2prods=firm2prods,
        prod2firms=prod2firms,
        inputs2supplier=inputs2supplier,
        exog_schedule=exog_schedule,
        demand_schedule=demand_schedule,
        gamma=gamma,
        seed=seed,
        debug=debug,
    )

    print(txns.head())
    print("rows:", len(txns), "t range:", txns["time"].min(), "->", txns["time"].max())
    return txns

if __name__ == "__main__":
    run()
