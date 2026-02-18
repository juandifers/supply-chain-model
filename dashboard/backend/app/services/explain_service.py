from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .scenario_store import ScenarioData


def _normalize(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    max_v = float(series.max())
    min_v = float(series.min())
    if np.isclose(max_v, min_v):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def _rolling_baseline(series: pd.Series, t: int, window: int = 8) -> float:
    hist = series[series.index < t]
    if len(hist) == 0:
        return float(series.mean()) if len(series) > 0 else 0.0
    return float(hist.tail(window).mean())


def _build_product_graph(data: ScenarioData) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in data.product_nodes.itertuples(index=False):
        g.add_node(int(row.product_id), name=row.product_name, is_consumer=bool(row.is_consumer), is_exogenous=bool(row.is_exogenous))
    for row in data.product_graph.itertuples(index=False):
        g.add_edge(int(row.source_id), int(row.dest_id), units=float(row.units))
    return g


def _product_distance_map(g: nx.DiGraph, source_products: List[int]) -> Dict[int, int]:
    if len(source_products) == 0:
        return {}

    dist: Dict[int, int] = {}
    for source in source_products:
        lengths = nx.single_source_shortest_path_length(g, source)
        for node, d in lengths.items():
            if node not in dist:
                dist[node] = d
            else:
                dist[node] = min(dist[node], d)
    return dist


def _ripple_products(data: ScenarioData, t: int, top_k: int, active_shocked_products: List[int]) -> List[Dict]:
    product_ts = data.per_product_timeseries
    rows_t = product_ts[product_ts["t"] == t].copy()
    if len(rows_t) == 0:
        return []

    g = _build_product_graph(data)
    dist_map = _product_distance_map(g, active_shocked_products)

    tx_by_product = product_ts.pivot(index="t", columns="product_id", values="tx_units").fillna(0)
    backlog_by_product = product_ts.pivot(index="t", columns="product_id", values="backlog_units").fillna(0)

    scores = []
    for row in rows_t.itertuples(index=False):
        p = int(row.product_id)
        curr_tx = float(row.tx_units)
        curr_backlog = float(row.backlog_units)

        tx_series = tx_by_product[p] if p in tx_by_product.columns else pd.Series(dtype=float)
        backlog_series = backlog_by_product[p] if p in backlog_by_product.columns else pd.Series(dtype=float)

        tx_baseline = _rolling_baseline(tx_series, t)
        backlog_baseline = _rolling_baseline(backlog_series, t)

        tx_drop_ratio = max(0.0, (tx_baseline - curr_tx) / max(1.0, tx_baseline))
        backlog_increase_ratio = max(0.0, (curr_backlog - backlog_baseline) / max(1.0, backlog_baseline + 1.0))

        if p in dist_map:
            proximity = 1.0 / (1.0 + float(dist_map[p]))
        else:
            proximity = 0.0

        impact_score = 0.5 * tx_drop_ratio + 0.3 * backlog_increase_ratio + 0.2 * proximity
        scores.append(
            {
                "product_id": p,
                "product_name": row.product_name,
                "impact_score": float(np.round(impact_score, 6)),
                "tx_drop_ratio": float(np.round(tx_drop_ratio, 6)),
                "backlog_increase_ratio": float(np.round(backlog_increase_ratio, 6)),
                "shock_proximity_score": float(np.round(proximity, 6)),
                "tx_units": float(curr_tx),
                "backlog_units": float(curr_backlog),
            }
        )

    scores = sorted(scores, key=lambda x: x["impact_score"], reverse=True)
    return scores[:top_k]


def _firm_chokepoints(data: ScenarioData, t: int, top_k: int) -> List[Dict]:
    edges = data.firm_supplier_edges[["supplier_id", "buyer_id", "product_id"]].drop_duplicates()
    if len(edges) == 0:
        return []

    unique_pairs = edges[["supplier_id", "buyer_id"]].drop_duplicates()
    out_degree = unique_pairs.groupby("supplier_id").size().rename("out_degree")

    product_graph = _build_product_graph(data)
    consumer_products = set(
        data.product_nodes[data.product_nodes["is_consumer"] == 1]["product_id"].astype(int).tolist()
    )

    downstream_count_by_product: Dict[int, int] = {}
    for p in data.product_nodes["product_id"].astype(int).tolist():
        reachable = nx.descendants(product_graph, p)
        downstream_count_by_product[p] = len(reachable.intersection(consumer_products))

    supplied_products_by_firm: Dict[int, set] = defaultdict(set)
    for row in edges.itertuples(index=False):
        supplied_products_by_firm[int(row.supplier_id)].add(int(row.product_id))

    coverage_by_firm: Dict[int, float] = {}
    for firm_id, products in supplied_products_by_firm.items():
        coverage_by_firm[firm_id] = float(np.sum([downstream_count_by_product[p] for p in products]))

    firm_rows_t = data.per_firm_timeseries[data.per_firm_timeseries["t"] == t].copy()
    if len(firm_rows_t) == 0:
        return []

    firm_rows_t["constrained_flow_share"] = (
        firm_rows_t["open_orders"] / (firm_rows_t["open_orders"] + firm_rows_t["outbound_units"] + 1e-6)
    )
    firm_rows_t["out_degree"] = firm_rows_t["firm_id"].map(out_degree).fillna(0.0)
    firm_rows_t["downstream_coverage"] = firm_rows_t["firm_id"].map(coverage_by_firm).fillna(0.0)

    firm_rows_t["out_degree_norm"] = _normalize(firm_rows_t["out_degree"]).fillna(0.0)
    firm_rows_t["coverage_norm"] = _normalize(firm_rows_t["downstream_coverage"]).fillna(0.0)
    firm_rows_t["constrained_norm"] = _normalize(firm_rows_t["constrained_flow_share"]).fillna(0.0)

    firm_rows_t["criticality_score"] = (
        0.4 * firm_rows_t["out_degree_norm"]
        + 0.3 * firm_rows_t["coverage_norm"]
        + 0.3 * firm_rows_t["constrained_norm"]
    )

    top = (
        firm_rows_t.sort_values("criticality_score", ascending=False)
        .head(top_k)
        .loc[
            :,
            [
                "firm_id",
                "firm_name",
                "criticality_score",
                "out_degree",
                "downstream_coverage",
                "constrained_flow_share",
                "open_orders",
                "outbound_units",
            ],
        ]
    )

    rows = []
    for row in top.itertuples(index=False):
        rows.append(
            {
                "firm_id": int(row.firm_id),
                "firm_name": row.firm_name,
                "criticality_score": float(np.round(row.criticality_score, 6)),
                "out_degree": float(np.round(row.out_degree, 3)),
                "downstream_coverage": float(np.round(row.downstream_coverage, 3)),
                "constrained_flow_share": float(np.round(row.constrained_flow_share, 6)),
                "open_orders": float(np.round(row.open_orders, 3)),
                "outbound_units": float(np.round(row.outbound_units, 3)),
            }
        )
    return rows


def _shock_paths(
    data: ScenarioData,
    t: int,
    top_k: int,
    active_shocked_products: List[Dict],
    ripple_products: List[Dict],
) -> List[Dict]:
    if len(active_shocked_products) == 0:
        return []

    g = _build_product_graph(data)
    consumer_products = set(
        data.product_nodes[data.product_nodes["is_consumer"] == 1]["product_id"].astype(int).tolist()
    )

    ripple_score_lookup = {int(r["product_id"]): float(r["impact_score"]) for r in ripple_products}

    candidate_paths: List[Dict] = []
    for shock in active_shocked_products:
        source = int(shock["product_id"])
        shock_severity = float(shock["shock_severity"])

        lengths = nx.single_source_shortest_path_length(g, source)
        for target, dist in lengths.items():
            if target not in consumer_products:
                continue
            try:
                path = nx.shortest_path(g, source=source, target=target)
            except nx.NetworkXNoPath:
                continue
            ripple_component = ripple_score_lookup.get(int(target), 0.0)
            score = (shock_severity * (0.6 + ripple_component)) / (1.0 + dist)

            path_nodes = []
            for p_id in path:
                node_row = data.product_nodes[data.product_nodes["product_id"] == p_id].head(1)
                name = node_row["product_name"].iloc[0] if len(node_row) > 0 else str(p_id)
                path_nodes.append({"product_id": int(p_id), "product_name": str(name)})

            candidate_paths.append(
                {
                    "source_product_id": source,
                    "source_product_name": shock["product_name"],
                    "target_consumer_product_id": int(target),
                    "target_consumer_product_name": path_nodes[-1]["product_name"],
                    "path_length": int(dist),
                    "path_score": float(np.round(score, 6)),
                    "source_shock_severity": float(np.round(shock_severity, 6)),
                    "path": path_nodes,
                }
            )

    candidate_paths = sorted(candidate_paths, key=lambda x: x["path_score"], reverse=True)
    deduped: List[Dict] = []
    seen_targets = set()
    for item in candidate_paths:
        tgt = item["target_consumer_product_id"]
        if tgt in seen_targets:
            continue
        seen_targets.add(tgt)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped


def explain_timestep(data: ScenarioData, t: int, top_k: int = 10) -> Dict:
    top_k = max(1, min(int(top_k), 50))
    kpi_row = data.kpi_history[data.kpi_history["t"] == t]
    if len(kpi_row) == 0:
        raise ValueError(f"No KPI row found for timestep t={t}")
    kpi = kpi_row.iloc[0].to_dict()

    exog_t = data.exog_supply_timeseries[data.exog_supply_timeseries["t"] == t].copy()
    active_shocks = exog_t[exog_t["shock_severity"] >= 0.5].copy()
    if len(active_shocks) == 0 and len(exog_t) > 0:
        active_shocks = exog_t.sort_values("shock_severity", ascending=False).head(1)

    active_shock_products = [
        {
            "product_id": int(row.product_id),
            "product_name": row.product_name,
            "shock_severity": float(np.round(row.shock_severity, 6)),
            "total_supply": float(np.round(row.total_supply, 3)),
            "baseline_supply": float(np.round(row.baseline_supply, 3)),
        }
        for row in active_shocks.itertuples(index=False)
    ]

    active_shock_ids = [x["product_id"] for x in active_shock_products]
    ripple_products = _ripple_products(data=data, t=t, top_k=top_k, active_shocked_products=active_shock_ids)
    critical_firms = _firm_chokepoints(data=data, t=t, top_k=top_k)
    paths = _shock_paths(data=data, t=t, top_k=top_k, active_shocked_products=active_shock_products, ripple_products=ripple_products)

    shock_summary = {
        "t": int(t),
        "active_shocked_products": active_shock_products,
        "worst_shocked_product": kpi.get("worst_shocked_product"),
        "worst_shock_severity": float(np.round(float(kpi.get("worst_shock_severity", 0.0)), 6)),
        "shock_exposure": float(np.round(float(kpi.get("shock_exposure", 0.0)), 6)),
        "active_exogenous_shocks": int(kpi.get("active_exogenous_shocks", 0)),
    }

    return {
        "t": int(t),
        "shock_summary": shock_summary,
        "ripple_products_top_k": ripple_products,
        "critical_firms_top_k": critical_firms,
        "paths": paths,
    }
