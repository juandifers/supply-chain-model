# SupplySim Explainable Dashboard

This folder contains the V1 dashboard stack:
- `backend/`: FastAPI API for scenario replay, explainability, and comparison.
- `frontend/`: React + TypeScript UI with graph, timeline, and explainability panels.

## 1) Export scenarios

```bash
./venv/bin/python scripts/export_supplysim_scenario.py --seed 0 --T 50 --scenario-id baseline_seed0
./venv/bin/python scripts/export_supplysim_scenario.py --seed 1 --T 50 --gamma 0.7 --shock-prob 0.003 --scenario-id scenario_seed1 --baseline-scenario-id baseline_seed0
./venv/bin/python scripts/export_supplysim_scenario.py --seed 2 --T 50 --scenario-id scenario_seed2_expedite --baseline-scenario-id baseline_seed0 --expedite-budget 12000 --expedite-c0 1.0 --expedite-alpha 0.5 --expedite-m-max 3.0 --expedite-cost-default 1.0 --expedite-cost-overrides '{"product4":2.5,"product7":1.8}'
```

Default output path:
`artifacts/scenarios/{scenario_id}`

## 2) Start backend

```bash
cd dashboard/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 3) Start frontend

```bash
cd dashboard/frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.
