from __future__ import annotations


def test_scenarios_endpoint(client):
    resp = client.get("/api/v1/scenarios")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] >= 2
    ids = [x["scenario_id"] for x in payload["scenarios"]]
    assert "baseline_seed0" in ids


def test_summary_endpoint(client):
    resp = client.get("/api/v1/scenarios/baseline_seed0/summary")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scenario_id"] == "baseline_seed0"
    assert payload["num_timesteps"] > 0


def test_kpis_endpoint_filters(client):
    resp = client.get("/api/v1/scenarios/baseline_seed0/kpis", params={"start_t": 3, "end_t": 7})
    assert resp.status_code == 200
    payload = resp.json()
    rows = payload["rows"]
    assert len(rows) > 0
    assert min(r["t"] for r in rows) >= 3
    assert max(r["t"] for r in rows) <= 7


def test_graph_endpoint(client):
    resp = client.get("/api/v1/scenarios/baseline_seed0/graph", params={"t": 5, "min_flow": 0.0})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scenario_id"] == "baseline_seed0"
    assert payload["t"] == 5
    assert "nodes" in payload and "edges" in payload


def test_explain_endpoint(client):
    resp = client.get("/api/v1/scenarios/baseline_seed0/explain", params={"t": 10, "top_k": 5})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scenario_id"] == "baseline_seed0"
    assert "shock_summary" in payload["explain"]
    assert "ripple_products_top_k" in payload["explain"]


def test_compare_endpoint(client):
    resp = client.get(
        "/api/v1/compare",
        params={"baseline_id": "baseline_seed0", "scenario_id": "scenario_seed1", "start_t": 2, "end_t": 15},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["baseline_id"] == "baseline_seed0"
    assert payload["scenario_id"] == "scenario_seed1"
    assert payload["summary"]["num_points"] > 0


def test_missing_scenario_returns_404(client):
    resp = client.get("/api/v1/scenarios/does_not_exist/summary")
    assert resp.status_code == 404
