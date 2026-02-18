from __future__ import annotations

def test_compare_service_basic(client):
    from dashboard.backend.app.services.compare_service import compare_scenarios
    from dashboard.backend.app.services.scenario_store import load_scenario_data

    baseline = load_scenario_data("baseline_seed0")
    scenario = load_scenario_data("scenario_seed1")
    payload = compare_scenarios(baseline, scenario, start_t=0, end_t=10)

    assert payload["baseline_id"] == "baseline_seed0"
    assert payload["scenario_id"] == "scenario_seed1"
    assert payload["summary"]["num_points"] > 0
    assert len(payload["series"]) == payload["summary"]["num_points"]
