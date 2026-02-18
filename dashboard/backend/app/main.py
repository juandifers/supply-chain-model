from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import SCENARIO_ROOT
from .models import (
    CompareResponse,
    ExplainResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    KPIsResponse,
    ScenarioMeta,
    ScenarioSummary,
    ScenariosResponse,
)
from .services.compare_service import compare_scenarios
from .services.explain_service import explain_timestep
from .services.scenario_index import list_scenarios
from .services.scenario_store import (
    ScenarioValidationError,
    filter_by_t,
    get_timestep_bounds,
    load_scenario_data,
)


app = FastAPI(
    title="SupplySim Dashboard API",
    version="1.0.0",
    description="Scenario replay and explainability backend for SupplySim",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _load_or_404(scenario_id: str):
    try:
        return load_scenario_data(scenario_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ScenarioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Failed to load scenario '{scenario_id}': {exc}") from exc


def _resolve_timestep(data, t: Optional[int]) -> int:
    t_min, t_max = get_timestep_bounds(data)
    t = t_max if t is None else int(t)
    if t < t_min or t > t_max:
        raise HTTPException(status_code=400, detail=f"Invalid timestep t={t}; expected [{t_min}, {t_max}]")
    return t


def _parse_product_filter(product_filter: Optional[str], data) -> Optional[Set[int]]:
    if product_filter is None:
        return None

    raw = product_filter.strip()
    if raw == "":
        return None

    if raw.isdigit():
        return {int(raw)}

    tokens = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if len(tokens) == 0:
        return None

    product_df = data.product_nodes.copy()
    product_df["product_name_lower"] = product_df["product_name"].astype(str).str.lower()
    matches = set()
    for token in tokens:
        mask = product_df["product_name_lower"].str.contains(token)
        matches.update(product_df[mask]["product_id"].astype(int).tolist())
    if len(matches) == 0:
        raise HTTPException(status_code=400, detail=f"product_filter='{product_filter}' matched no products")
    return matches


@app.get("/api/v1/scenarios", response_model=ScenariosResponse)
def get_scenarios() -> ScenariosResponse:
    rows = list_scenarios()
    if hasattr(ScenarioMeta, "model_fields"):  # pydantic v2
        allowed = set(ScenarioMeta.model_fields.keys())
    else:  # pydantic v1
        allowed = set(ScenarioMeta.__fields__.keys())
    scenarios = [ScenarioMeta(**{k: row[k] for k in allowed if k in row}) for row in rows]
    return ScenariosResponse(scenario_root=str(SCENARIO_ROOT), count=len(scenarios), scenarios=scenarios)


@app.get("/api/v1/scenarios/{scenario_id}/summary", response_model=ScenarioSummary)
def get_scenario_summary(scenario_id: str) -> ScenarioSummary:
    data = _load_or_404(scenario_id)
    kpi = data.kpi_history

    t_min, t_max = get_timestep_bounds(data)
    peak_open_orders_t = None
    peak_open_orders = None
    peak_backlog_t = None
    peak_backlog_units = None
    final_fill = None

    if len(kpi) > 0:
        open_idx = int(kpi["open_orders"].idxmax())
        peak_open_orders_t = int(kpi.loc[open_idx, "t"])
        peak_open_orders = float(kpi.loc[open_idx, "open_orders"])

        backlog_idx = int(kpi["consumer_backlog_units"].idxmax())
        peak_backlog_t = int(kpi.loc[backlog_idx, "t"])
        peak_backlog_units = float(kpi.loc[backlog_idx, "consumer_backlog_units"])

        final_fill = float(kpi.iloc[-1]["consumer_cumulative_fill_rate"])

    return ScenarioSummary(
        scenario_id=scenario_id,
        manifest=data.manifest,
        t_min=t_min,
        t_max=t_max,
        num_timesteps=int(max(0, t_max - t_min + 1)),
        num_transactions=int(len(data.transactions)),
        peak_open_orders_t=peak_open_orders_t,
        peak_open_orders=peak_open_orders,
        peak_backlog_t=peak_backlog_t,
        peak_backlog_units=peak_backlog_units,
        final_cumulative_fill_rate=final_fill,
    )


@app.get("/api/v1/scenarios/{scenario_id}/kpis", response_model=KPIsResponse)
def get_kpis(
    scenario_id: str,
    start_t: Optional[int] = Query(default=None),
    end_t: Optional[int] = Query(default=None),
) -> KPIsResponse:
    data = _load_or_404(scenario_id)

    t_min, t_max = get_timestep_bounds(data)
    if start_t is not None and start_t < t_min:
        raise HTTPException(status_code=400, detail=f"start_t must be >= {t_min}")
    if end_t is not None and end_t > t_max:
        raise HTTPException(status_code=400, detail=f"end_t must be <= {t_max}")
    if start_t is not None and end_t is not None and start_t > end_t:
        raise HTTPException(status_code=400, detail="start_t must be <= end_t")

    rows = filter_by_t(data.kpi_history, start_t=start_t, end_t=end_t)
    return KPIsResponse(
        scenario_id=scenario_id,
        t_min=t_min,
        t_max=t_max,
        rows=rows.to_dict(orient="records"),
    )


@app.get("/api/v1/scenarios/{scenario_id}/graph", response_model=GraphResponse)
def get_graph(
    scenario_id: str,
    t: Optional[int] = Query(default=None),
    product_filter: Optional[str] = Query(default=None),
    min_flow: float = Query(default=0.0, ge=0.0),
) -> GraphResponse:
    data = _load_or_404(scenario_id)
    t = _resolve_timestep(data, t)

    product_filter_ids = _parse_product_filter(product_filter=product_filter, data=data)

    tx_all = data.transactions.copy()
    tx_t = tx_all[tx_all["time"] == t].copy()

    if product_filter_ids is not None:
        tx_all = tx_all[tx_all["product_id"].isin(product_filter_ids)]
        tx_t = tx_t[tx_t["product_id"].isin(product_filter_ids)]

    flow_all = (
        tx_all.groupby(["supplier_id", "buyer_id", "product_id"])["amount"].sum().to_dict()
        if len(tx_all) > 0
        else {}
    )
    flow_t = (
        tx_t.groupby(["supplier_id", "buyer_id", "product_id"])["amount"].sum().to_dict()
        if len(tx_t) > 0
        else {}
    )

    candidate_edges = data.firm_supplier_edges.copy()
    if product_filter_ids is not None:
        candidate_edges = candidate_edges[candidate_edges["product_id"].isin(product_filter_ids)]

    per_firm_t = data.per_firm_timeseries[data.per_firm_timeseries["t"] == t].copy()
    per_firm_lookup = {
        int(row.firm_id): {
            "inbound_units": float(row.inbound_units),
            "outbound_units": float(row.outbound_units),
            "open_orders": float(row.open_orders),
        }
        for row in per_firm_t.itertuples(index=False)
    }

    edges: List[GraphEdge] = []
    node_ids_in_edges: Set[int] = set()

    for row in candidate_edges.itertuples(index=False):
        key = (int(row.supplier_id), int(row.buyer_id), int(row.product_id))
        total_flow = float(flow_all.get(key, 0.0))
        flow_at_t = float(flow_t.get(key, 0.0))
        if max(total_flow, flow_at_t) < min_flow:
            continue

        edge_id = f"{row.supplier_id}-{row.buyer_id}-{row.product_id}"
        edges.append(
            GraphEdge(
                id=edge_id,
                source=f"firm:{int(row.supplier_id)}",
                target=f"firm:{int(row.buyer_id)}",
                product_id=int(row.product_id),
                product_name=str(row.product_name),
                flow_at_t=float(np.round(flow_at_t, 6)),
                total_flow=float(np.round(total_flow, 6)),
            )
        )
        node_ids_in_edges.add(int(row.supplier_id))
        node_ids_in_edges.add(int(row.buyer_id))

    nodes: List[GraphNode] = []
    firm_df = data.firm_nodes
    if len(node_ids_in_edges) == 0:
        node_ids_in_edges = set(firm_df["firm_id"].astype(int).tolist())

    for row in firm_df.itertuples(index=False):
        firm_id = int(row.firm_id)
        if firm_id not in node_ids_in_edges:
            continue
        nodes.append(
            GraphNode(
                id=f"firm:{firm_id}",
                label=str(row.firm_name),
                type="firm",
                metrics=per_firm_lookup.get(firm_id, {"inbound_units": 0.0, "outbound_units": 0.0, "open_orders": 0.0}),
            )
        )

    return GraphResponse(
        scenario_id=scenario_id,
        t=t,
        product_filter=product_filter,
        min_flow=min_flow,
        nodes=nodes,
        edges=edges,
    )


@app.get("/api/v1/scenarios/{scenario_id}/explain", response_model=ExplainResponse)
def get_explain(
    scenario_id: str,
    t: Optional[int] = Query(default=None),
    top_k: int = Query(default=10, ge=1, le=50),
) -> ExplainResponse:
    data = _load_or_404(scenario_id)
    t = _resolve_timestep(data, t)

    try:
        explain = explain_timestep(data=data, t=t, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExplainResponse(scenario_id=scenario_id, t=t, explain=explain)


@app.get("/api/v1/compare", response_model=CompareResponse)
def get_compare(
    baseline_id: str = Query(...),
    scenario_id: str = Query(...),
    start_t: Optional[int] = Query(default=None),
    end_t: Optional[int] = Query(default=None),
) -> CompareResponse:
    baseline_data = _load_or_404(baseline_id)
    scenario_data = _load_or_404(scenario_id)

    try:
        payload = compare_scenarios(
            baseline_data=baseline_data,
            scenario_data=scenario_data,
            start_t=start_t,
            end_t=end_t,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CompareResponse(**payload)
