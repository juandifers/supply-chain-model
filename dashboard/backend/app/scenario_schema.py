from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "1.0"
REQUIRED_SCENARIO_FILES = [
    "manifest.json",
    "kpi_history.csv",
    "transactions.csv",
    "product_graph.csv",
    "firm_nodes.csv",
    "product_nodes.csv",
    "firm_supplier_edges.csv",
    "exog_supply_timeseries.csv",
    "demand_timeseries.csv",
    "per_product_timeseries.csv",
    "per_firm_timeseries.csv",
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    required_top_level = [
        "schema_version",
        "scenario_id",
        "created_at",
        "sim_config",
        "description",
    ]
    for key in required_top_level:
        if key not in manifest:
            errors.append(f"Missing manifest field: {key}")

    if "schema_version" in manifest and str(manifest["schema_version"]) != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version '{manifest['schema_version']}', expected '{SCHEMA_VERSION}'"
        )

    sim_config = manifest.get("sim_config", {})
    for key in ["seed", "T", "gamma", "shock_prob"]:
        if key not in sim_config:
            errors.append(f"Missing sim_config field: {key}")

    if "seed" in sim_config and not isinstance(sim_config["seed"], int):
        errors.append("sim_config.seed must be an integer")
    if "T" in sim_config and not isinstance(sim_config["T"], int):
        errors.append("sim_config.T must be an integer")
    if "gamma" in sim_config and not _is_number(sim_config["gamma"]):
        errors.append("sim_config.gamma must be numeric")
    if "shock_prob" in sim_config and not _is_number(sim_config["shock_prob"]):
        errors.append("sim_config.shock_prob must be numeric")
    if "expedite_budget" in sim_config and sim_config["expedite_budget"] is not None and not _is_number(sim_config["expedite_budget"]):
        errors.append("sim_config.expedite_budget must be numeric or null")
    if "expedite_c0" in sim_config and not _is_number(sim_config["expedite_c0"]):
        errors.append("sim_config.expedite_c0 must be numeric")
    if "expedite_alpha" in sim_config and not _is_number(sim_config["expedite_alpha"]):
        errors.append("sim_config.expedite_alpha must be numeric")
    if "expedite_m_max" in sim_config and sim_config["expedite_m_max"] is not None and not _is_number(sim_config["expedite_m_max"]):
        errors.append("sim_config.expedite_m_max must be numeric or null")
    if "expedite_cost_default" in sim_config and not _is_number(sim_config["expedite_cost_default"]):
        errors.append("sim_config.expedite_cost_default must be numeric")
    if "expedite_cost_overrides" in sim_config and not isinstance(sim_config["expedite_cost_overrides"], dict):
        errors.append("sim_config.expedite_cost_overrides must be a dict")

    baseline_id = manifest.get("baseline_scenario_id")
    if baseline_id is not None and not isinstance(baseline_id, str):
        errors.append("baseline_scenario_id must be a string or null")

    return errors


def load_manifest(scenario_dir: Path) -> Dict[str, Any]:
    manifest_path = scenario_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_scenario_dir(scenario_dir: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not scenario_dir.exists() or not scenario_dir.is_dir():
        return False, [f"Scenario path does not exist: {scenario_dir}"]

    for filename in REQUIRED_SCENARIO_FILES:
        file_path = scenario_dir / filename
        if not file_path.exists():
            errors.append(f"Missing required file: {filename}")

    manifest_path = scenario_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = load_manifest(scenario_dir)
            errors.extend(validate_manifest(manifest))
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"Invalid manifest.json: {exc}")

    return len(errors) == 0, errors
