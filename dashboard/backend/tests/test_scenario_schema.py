from __future__ import annotations

import json
from pathlib import Path

from dashboard.backend.app.scenario_schema import SCHEMA_VERSION, validate_manifest, validate_scenario_dir


def test_manifest_validation_happy_path():
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "x",
        "created_at": "2026-01-01T00:00:00Z",
        "sim_config": {"seed": 0, "T": 20, "gamma": 0.8, "shock_prob": 0.001},
        "description": "fixture",
        "baseline_scenario_id": None,
    }
    errors = validate_manifest(manifest)
    assert errors == []


def test_manifest_validation_rejects_missing_fields():
    manifest = {"schema_version": SCHEMA_VERSION}
    errors = validate_manifest(manifest)
    assert any("Missing manifest field" in e for e in errors)


def test_scenario_dir_validation_reports_missing_files(tmp_path: Path):
    scenario_dir = tmp_path / "bad_scenario"
    scenario_dir.mkdir(parents=True)
    with (scenario_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"schema_version": "1.0"}, f)

    ok, errors = validate_scenario_dir(scenario_dir)
    assert not ok
    assert any("Missing required file" in e for e in errors)
