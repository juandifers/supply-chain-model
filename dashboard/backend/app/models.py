from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScenarioMeta(BaseModel):
    scenario_id: str
    created_at: Optional[str] = None
    description: Optional[str] = None
    baseline_scenario_id: Optional[str] = None
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    sim_config: Dict[str, Any] = Field(default_factory=dict)


class ScenariosResponse(BaseModel):
    scenario_root: str
    count: int
    scenarios: List[ScenarioMeta]


class ScenarioSummary(BaseModel):
    scenario_id: str
    manifest: Dict[str, Any]
    t_min: int
    t_max: int
    num_timesteps: int
    num_transactions: int
    peak_open_orders_t: Optional[int] = None
    peak_open_orders: Optional[float] = None
    peak_backlog_t: Optional[int] = None
    peak_backlog_units: Optional[float] = None
    final_cumulative_fill_rate: Optional[float] = None


class KPIsResponse(BaseModel):
    scenario_id: str
    t_min: int
    t_max: int
    rows: List[Dict[str, Any]]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    product_id: int
    product_name: str
    flow_at_t: float
    total_flow: float


class GraphResponse(BaseModel):
    scenario_id: str
    t: int
    product_filter: Optional[str] = None
    min_flow: float
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ExplainResponse(BaseModel):
    scenario_id: str
    t: int
    explain: Dict[str, Any]


class CompareResponse(BaseModel):
    baseline_id: str
    scenario_id: str
    summary: Dict[str, Any]
    series: List[Dict[str, Any]]
