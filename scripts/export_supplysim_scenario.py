import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard.backend.app.scenario_schema import SCHEMA_VERSION, validate_scenario_dir
from scripts.supplysim_env import SupplySimEnv


def _slugify_float(value: float) -> str:
    return str(value).replace(".", "p")


def _default_scenario_id(seed: int, T: int, gamma: float, shock_prob: float) -> str:
    return f"seed{seed}_T{T}_gamma{_slugify_float(gamma)}_shock{_slugify_float(shock_prob)}"


def _parse_product_cost_overrides(raw: str | None) -> Dict[str, float]:
    if raw is None:
        return {}
    text = raw.strip()
    if text == "":
        return {}

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--expedite-cost-overrides must be a JSON object: {\"product_name\": cost}")

    out: Dict[str, float] = {}
    for product, cost in payload.items():
        if not isinstance(product, str):
            raise ValueError("expedite-cost-overrides keys must be product name strings")
        out[product] = float(cost)
    return out


def _build_static_tables(env: SupplySimEnv) -> Dict[str, pd.DataFrame]:
    exog_set = set(env.exog_prods)
    consumer_set = set(env.consumer_prods)

    firm_nodes = pd.DataFrame(
        [
            {"firm_id": env.firm2idx[f], "firm_name": f}
            for f in env.firms
        ]
    ).sort_values("firm_id")

    product_nodes = pd.DataFrame(
        [
            {
                "product_id": env.prod2idx[p],
                "product_name": p,
                "is_exogenous": int(p in exog_set),
                "is_consumer": int(p in consumer_set),
            }
            for p in env.products
        ]
    ).sort_values("product_id")

    product_graph = env.prod_graph.copy()
    product_graph["source_id"] = product_graph["source"].map(env.prod2idx)
    product_graph["dest_id"] = product_graph["dest"].map(env.prod2idx)
    product_graph = product_graph[["source_id", "source", "dest_id", "dest", "units", "layer"]]

    edges: List[Dict[str, object]] = []
    for (buyer, product), supplier in env.inputs2supplier.items():
        edges.append(
            {
                "supplier_id": env.firm2idx[supplier],
                "supplier_name": supplier,
                "buyer_id": env.firm2idx[buyer],
                "buyer_name": buyer,
                "product_id": env.prod2idx[product],
                "product_name": product,
            }
        )
    firm_supplier_edges = pd.DataFrame(edges)
    if len(firm_supplier_edges) > 0:
        firm_supplier_edges = firm_supplier_edges.sort_values(["supplier_id", "buyer_id", "product_id"])

    demand_rows: List[Dict[str, object]] = []
    if env.demand_schedule is not None:
        for t in range(env.T):
            demand_t = env.demand_schedule[t]
            by_product: Dict[str, float] = {p: 0.0 for p in env.products}
            for (firm, p), amount in demand_t.items():
                del firm
                by_product[p] += float(amount)
            for p in env.products:
                demand_rows.append(
                    {
                        "t": t,
                        "product_id": env.prod2idx[p],
                        "product_name": p,
                        "demand_units": by_product[p],
                    }
                )
    demand_timeseries = pd.DataFrame(demand_rows)

    exog_rows: List[Dict[str, object]] = []
    if env.exog_schedule is not None:
        for t in range(env.T):
            exog_t = env.exog_schedule[t]
            by_product: Dict[str, float] = {p: 0.0 for p in env.exog_prods}
            for (firm, p), amount in exog_t.items():
                del firm
                if p in by_product:
                    by_product[p] += float(amount)
            for p in env.exog_prods:
                baseline = float(env.exog_baseline_supply.get(p, max(1.0, by_product[p])))
                severity = max(0.0, min(1.0, 1.0 - (by_product[p] / max(1.0, baseline))))
                exog_rows.append(
                    {
                        "t": t,
                        "product_id": env.prod2idx[p],
                        "product_name": p,
                        "total_supply": by_product[p],
                        "baseline_supply": baseline,
                        "shock_severity": np.round(severity, 6),
                    }
                )
    exog_supply_timeseries = pd.DataFrame(exog_rows)

    return {
        "firm_nodes": firm_nodes,
        "product_nodes": product_nodes,
        "product_graph": product_graph,
        "firm_supplier_edges": firm_supplier_edges,
        "demand_timeseries": demand_timeseries,
        "exog_supply_timeseries": exog_supply_timeseries,
    }


def _compute_open_orders_snapshots(env: SupplySimEnv) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    per_product_open_orders: Dict[int, float] = {env.prod2idx[p]: 0.0 for p in env.products}
    per_product_consumer_backlog: Dict[int, float] = {env.prod2idx[p]: 0.0 for p in env.products}
    per_firm_open_orders: Dict[int, float] = {env.firm2idx[f]: 0.0 for f in env.firms}

    for (supplier_name, product_name), orders in env.curr_orders.items():
        p_idx = env.prod2idx[product_name]
        s_idx = env.firm2idx[supplier_name]
        per_product_open_orders[p_idx] += float(len(orders))
        per_firm_open_orders[s_idx] += float(len(orders))

        backlog = 0.0
        for buyer, amount, *_ in orders:
            if buyer == "consumer":
                backlog += float(amount)
        per_product_consumer_backlog[p_idx] += backlog

    return per_product_open_orders, per_product_consumer_backlog, per_firm_open_orders


def _export_scenario(
    scenario_dir: Path,
    seed: int,
    T: int,
    gamma: float,
    shock_prob: float,
    init_inv: float,
    init_supply: float,
    init_demand: float,
    description: str,
    baseline_scenario_id: str | None,
    expedite_budget: float | None,
    expedite_c0: float,
    expedite_alpha: float,
    expedite_m_max: float | None,
    expedite_cost_default: float,
    expedite_cost_overrides: Dict[str, float],
) -> None:
    env = SupplySimEnv(
        seed=seed,
        T=T,
        gamma=gamma,
        log_kpis=False,
        expedite_budget=expedite_budget,
        expedite_c0=expedite_c0,
        expedite_alpha=expedite_alpha,
        expedite_m_max=expedite_m_max,
        expedite_cost_default=expedite_cost_default,
        expedite_cost_per_unit=expedite_cost_overrides,
    )
    env.reset(
        init_inv=init_inv,
        init_supply=init_supply,
        init_demand=init_demand,
        use_demand_schedule=True,
        use_exog_schedule=True,
        shock_prob=shock_prob,
    )

    static_tables = _build_static_tables(env)

    all_transactions: List[pd.DataFrame] = []
    per_product_rows: List[Dict[str, object]] = []
    per_firm_rows: List[Dict[str, object]] = []

    done = False
    while not done:
        _, _, done, info = env.step(action=None, debug=False, log_kpis=False)
        t = int(info["kpis"]["t"])
        txns_t = info["transactions"].copy()
        if len(txns_t) > 0:
            all_transactions.append(txns_t)

        tx_units_per_product: Dict[int, float] = {env.prod2idx[p]: 0.0 for p in env.products}
        inbound_per_firm: Dict[int, float] = {env.firm2idx[f]: 0.0 for f in env.firms}
        outbound_per_firm: Dict[int, float] = {env.firm2idx[f]: 0.0 for f in env.firms}

        if len(txns_t) > 0:
            grouped_prod = txns_t.groupby("product_id")["amount"].sum().to_dict()
            for p_idx, amount in grouped_prod.items():
                tx_units_per_product[int(p_idx)] = float(amount)

            grouped_in = txns_t.groupby("buyer_id")["amount"].sum().to_dict()
            grouped_out = txns_t.groupby("supplier_id")["amount"].sum().to_dict()
            for firm_idx, amount in grouped_in.items():
                inbound_per_firm[int(firm_idx)] = float(amount)
            for firm_idx, amount in grouped_out.items():
                outbound_per_firm[int(firm_idx)] = float(amount)

        exog_shock_by_product: Dict[int, float] = {env.prod2idx[p]: 0.0 for p in env.products}
        if env.exog_schedule is not None:
            exog_t = env.exog_schedule[t]
            supply_by_product: Dict[str, float] = {p: 0.0 for p in env.exog_prods}
            for (firm, p), amount in exog_t.items():
                del firm
                supply_by_product[p] += float(amount)
            for p in env.exog_prods:
                baseline = float(env.exog_baseline_supply.get(p, max(1.0, supply_by_product[p])))
                severity = max(0.0, min(1.0, 1.0 - supply_by_product[p] / max(1.0, baseline)))
                exog_shock_by_product[env.prod2idx[p]] = np.round(severity, 6)

        per_product_open_orders, per_product_consumer_backlog, per_firm_open_orders = _compute_open_orders_snapshots(env)

        for p in env.products:
            p_idx = env.prod2idx[p]
            per_product_rows.append(
                {
                    "t": t,
                    "product_id": p_idx,
                    "product_name": p,
                    "tx_units": np.round(tx_units_per_product[p_idx], 5),
                    "open_orders": int(per_product_open_orders[p_idx]),
                    "backlog_units": np.round(per_product_consumer_backlog[p_idx], 5),
                    "shock_severity": exog_shock_by_product[p_idx],
                }
            )

        for f in env.firms:
            f_idx = env.firm2idx[f]
            per_firm_rows.append(
                {
                    "t": t,
                    "firm_id": f_idx,
                    "firm_name": f,
                    "inbound_units": np.round(inbound_per_firm[f_idx], 5),
                    "outbound_units": np.round(outbound_per_firm[f_idx], 5),
                    "open_orders": int(per_firm_open_orders[f_idx]),
                }
            )

    kpi_history = env.get_kpi_history()
    transactions = (
        pd.concat(all_transactions, ignore_index=True)
        if len(all_transactions) > 0
        else pd.DataFrame(columns=["supplier_id", "buyer_id", "product_id", "amount", "time"])
    )
    transactions = transactions.sort_values(["time", "supplier_id", "buyer_id", "product_id"]).reset_index(drop=True)

    per_product_df = pd.DataFrame(per_product_rows)
    per_firm_df = pd.DataFrame(per_firm_rows)

    scenario_dir.mkdir(parents=True, exist_ok=True)
    kpi_history.to_csv(scenario_dir / "kpi_history.csv", index=False)
    transactions.to_csv(scenario_dir / "transactions.csv", index=False)
    static_tables["product_graph"].to_csv(scenario_dir / "product_graph.csv", index=False)
    static_tables["firm_nodes"].to_csv(scenario_dir / "firm_nodes.csv", index=False)
    static_tables["product_nodes"].to_csv(scenario_dir / "product_nodes.csv", index=False)
    static_tables["firm_supplier_edges"].to_csv(scenario_dir / "firm_supplier_edges.csv", index=False)
    static_tables["exog_supply_timeseries"].to_csv(scenario_dir / "exog_supply_timeseries.csv", index=False)
    static_tables["demand_timeseries"].to_csv(scenario_dir / "demand_timeseries.csv", index=False)
    per_product_df.to_csv(scenario_dir / "per_product_timeseries.csv", index=False)
    per_firm_df.to_csv(scenario_dir / "per_firm_timeseries.csv", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sim_config": {
            "seed": seed,
            "T": T,
            "gamma": gamma,
            "shock_prob": shock_prob,
            "init_inv": init_inv,
            "init_supply": init_supply,
            "init_demand": init_demand,
            "expedite_budget": expedite_budget,
            "expedite_c0": expedite_c0,
            "expedite_alpha": expedite_alpha,
            "expedite_m_max": expedite_m_max,
            "expedite_cost_default": expedite_cost_default,
            "expedite_cost_overrides": expedite_cost_overrides,
        },
        "description": description,
        "baseline_scenario_id": baseline_scenario_id,
    }
    with (scenario_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    is_valid, errors = validate_scenario_dir(scenario_dir)
    if not is_valid:
        raise RuntimeError(f"Scenario package failed validation: {errors}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SupplySim run into a versioned scenario package")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--shock-prob", type=float, default=0.001)
    parser.add_argument("--init-inv", type=float, default=0.0)
    parser.add_argument("--init-supply", type=float, default=100.0)
    parser.add_argument("--init-demand", type=float, default=1.0)
    parser.add_argument("--scenario-id", type=str, default=None)
    parser.add_argument("--description", type=str, default="SupplySim scenario export")
    parser.add_argument("--baseline-scenario-id", type=str, default=None)
    parser.add_argument(
        "--expedite-budget",
        type=float,
        default=None,
        help="Episode expedite budget. Omit for unlimited.",
    )
    parser.add_argument(
        "--expedite-c0",
        type=float,
        default=1.0,
        help="Base per-unit expedite cost at depth 0.",
    )
    parser.add_argument(
        "--expedite-alpha",
        type=float,
        default=0.5,
        help="Tier premium factor in c_p = c0 * (1 + alpha * depth).",
    )
    parser.add_argument(
        "--expedite-m-max",
        type=float,
        default=3.0,
        help="Clamp for supply multipliers m in actions; set <=0 to disable expedite boost.",
    )
    parser.add_argument(
        "--expedite-cost-default",
        type=float,
        default=1.0,
        help="Fallback per-unit cost if product cost is missing.",
    )
    parser.add_argument(
        "--expedite-cost-overrides",
        type=str,
        default=None,
        help="JSON dict of per-product unit costs, e.g. '{\"product4\": 2.5, \"product7\": 1.2}'",
    )
    parser.add_argument("--output-root", type=str, default=os.path.join(ROOT, "artifacts", "scenarios"))
    return parser.parse_args()


def main() -> None:
    args = get_args()
    expedite_cost_overrides = _parse_product_cost_overrides(args.expedite_cost_overrides)
    scenario_id = args.scenario_id or _default_scenario_id(args.seed, args.T, args.gamma, args.shock_prob)
    scenario_dir = Path(args.output_root) / scenario_id

    _export_scenario(
        scenario_dir=scenario_dir,
        seed=args.seed,
        T=args.T,
        gamma=args.gamma,
        shock_prob=args.shock_prob,
        init_inv=args.init_inv,
        init_supply=args.init_supply,
        init_demand=args.init_demand,
        description=args.description,
        baseline_scenario_id=args.baseline_scenario_id,
        expedite_budget=args.expedite_budget,
        expedite_c0=args.expedite_c0,
        expedite_alpha=args.expedite_alpha,
        expedite_m_max=args.expedite_m_max,
        expedite_cost_default=args.expedite_cost_default,
        expedite_cost_overrides=expedite_cost_overrides,
    )
    print(f"Exported scenario: {scenario_id}")
    print(f"Path: {scenario_dir}")


if __name__ == "__main__":
    main()
