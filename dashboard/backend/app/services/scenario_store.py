from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..scenario_schema import load_manifest, validate_scenario_dir
from .scenario_index import resolve_scenario_dir


class ScenarioValidationError(Exception):
    pass


@dataclass(frozen=True)
class ScenarioData:
    scenario_id: str
    root: Path
    manifest: Dict
    kpi_history: pd.DataFrame
    transactions: pd.DataFrame
    product_graph: pd.DataFrame
    firm_nodes: pd.DataFrame
    product_nodes: pd.DataFrame
    firm_supplier_edges: pd.DataFrame
    exog_supply_timeseries: pd.DataFrame
    demand_timeseries: pd.DataFrame
    per_product_timeseries: pd.DataFrame
    per_firm_timeseries: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=16)
def load_scenario_data(scenario_id: str) -> ScenarioData:
    scenario_dir = resolve_scenario_dir(scenario_id)
    is_valid, errors = validate_scenario_dir(scenario_dir)
    if not is_valid:
        raise ScenarioValidationError(
            f"Scenario '{scenario_id}' failed validation: {'; '.join(errors)}"
        )

    manifest = load_manifest(scenario_dir)

    data = ScenarioData(
        scenario_id=scenario_id,
        root=scenario_dir,
        manifest=manifest,
        kpi_history=_read_csv(scenario_dir / "kpi_history.csv"),
        transactions=_read_csv(scenario_dir / "transactions.csv"),
        product_graph=_read_csv(scenario_dir / "product_graph.csv"),
        firm_nodes=_read_csv(scenario_dir / "firm_nodes.csv"),
        product_nodes=_read_csv(scenario_dir / "product_nodes.csv"),
        firm_supplier_edges=_read_csv(scenario_dir / "firm_supplier_edges.csv"),
        exog_supply_timeseries=_read_csv(scenario_dir / "exog_supply_timeseries.csv"),
        demand_timeseries=_read_csv(scenario_dir / "demand_timeseries.csv"),
        per_product_timeseries=_read_csv(scenario_dir / "per_product_timeseries.csv"),
        per_firm_timeseries=_read_csv(scenario_dir / "per_firm_timeseries.csv"),
    )
    return data


def clear_cache() -> None:
    load_scenario_data.cache_clear()


def get_timestep_bounds(data: ScenarioData) -> tuple[int, int]:
    if len(data.kpi_history) == 0:
        return 0, 0
    return int(data.kpi_history["t"].min()), int(data.kpi_history["t"].max())


def filter_by_t(df: pd.DataFrame, start_t: Optional[int], end_t: Optional[int]) -> pd.DataFrame:
    out = df
    if start_t is not None:
        out = out[out["t"] >= start_t]
    if end_t is not None:
        out = out[out["t"] <= end_t]
    return out
