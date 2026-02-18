from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..config import SCENARIO_ROOT
from ..scenario_schema import load_manifest, validate_scenario_dir


def _iter_scenario_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_scenarios() -> List[Dict]:
    scenarios: List[Dict] = []
    for scenario_dir in _iter_scenario_dirs(SCENARIO_ROOT):
        is_valid, errors = validate_scenario_dir(scenario_dir)
        manifest = {}
        if (scenario_dir / "manifest.json").exists():
            try:
                manifest = load_manifest(scenario_dir)
            except Exception as exc:  # pylint: disable=broad-except
                errors = errors + [f"Failed to parse manifest: {exc}"]
                is_valid = False

        scenarios.append(
            {
                "scenario_id": scenario_dir.name,
                "path": str(scenario_dir),
                "is_valid": is_valid,
                "errors": errors,
                "created_at": manifest.get("created_at"),
                "description": manifest.get("description"),
                "baseline_scenario_id": manifest.get("baseline_scenario_id"),
                "sim_config": manifest.get("sim_config", {}),
            }
        )

    scenarios.sort(key=lambda x: (x.get("created_at") or "", x["scenario_id"]), reverse=True)
    return scenarios


def resolve_scenario_dir(scenario_id: str) -> Path:
    scenario_dir = SCENARIO_ROOT / scenario_id
    if not scenario_dir.exists() or not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario '{scenario_id}' does not exist under {SCENARIO_ROOT}")
    return scenario_dir
