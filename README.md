# SupplySim Thesis Workspace

This repository started from the codebase for our AAAI 2025 paper, ["Learning Production Functions for Supply Chains with Graph Neural Networks"](https://arxiv.org/abs/2407.18772), and vendors a fork of [TGB](https://github.com/shenyangHuang/TGB) plus related baseline code. At the current thesis stage, the active project is the explainable SupplySim replay dashboard and scenario-export pipeline. Most day-to-day work should happen outside the upstream `TGB/` tree.

## Agent Orientation

If you are a new agent or collaborator, use these defaults:

- Start with `scripts/`, `dashboard/backend/`, `dashboard/frontend/`, and `artifacts/scenarios/`.
- Assume the current thesis focus is scenario replay, explainability, and baseline comparison, not reproducing the full paper training stack.
- Only work inside `TGB/` when the task is explicitly about simulator internals or the original paper experiments.
- Treat `dashboard/frontend/node_modules/`, `dashboard/frontend/dist/`, `__pycache__/`, and ad hoc exports in `artifacts/` or the repo root as generated artifacts, not primary source files.

## Current Thesis Focus

The active workflow in this repo is:

1. Run SupplySim through a stepwise wrapper.
2. Export a versioned scenario package.
3. Load that package in the backend.
4. Explore it in the frontend replay and comparison UI.

The key files and folders for that workflow are:

- `scripts/supplysim_env.py`: step-by-step wrapper around `TGB/modules/synthetic_data.py` with KPI tracking and intervention hooks.
- `scripts/export_supplysim_scenario.py`: exports a simulation run into `artifacts/scenarios/{scenario_id}`.
- `artifacts/scenarios/`: versioned scenario packages used by the backend and frontend. Checked-in examples currently include `baseline_seed0`, `scenario_seed1`, `demo_baseline`, and `demo_scenario`.
- `dashboard/backend/`: FastAPI API for scenario discovery, summaries, KPI slices, graph replay, explainability, and baseline comparison.
- `dashboard/frontend/`: React + TypeScript control-tower UI for scenario replay and comparison.

Unless a task explicitly mentions model training, dataset registration, or paper reproduction, this is the part of the repo you probably want.

## Active Vs Legacy Areas

### Active now

- `scripts/`
  - Runtime wrappers and scenario export utilities for the thesis dashboard workflow.
- `artifacts/scenarios/`
  - Scenario packages loaded by the backend by default.
- `dashboard/backend/app/`
  - API entry points and services.
- `dashboard/backend/tests/`
  - Backend tests for scenario loading, schema validation, explainability, and comparison.
- `dashboard/frontend/src/`
  - Replay UI, comparison page, charts, and graph/explain panels.

### Kept mainly for upstream or paper-reference purposes

- `TGB/`
  - Vendored upstream research code. The main currently reused dependency is `TGB/modules/synthetic_data.py`.
- `TGB/examples/linkproppred/general/`
  - Original experiment runners for link prediction and model evaluation.
- `TGB/tgb/`
  - Dataset, evaluation, and utility code from the original research stack.
- `register_data/`
  - Paper-era preprocessing and hypergraph registration scripts for proprietary real-world datasets.
- `synthetic_data/`
  - Released synthetic CSV / pickle artifacts from the paper workflow. Useful as reference, but not required for the dashboard scenario pipeline.

## Scenario Package Contract

Each scenario under `artifacts/scenarios/{scenario_id}` is a self-contained package that the backend validates and serves. In practice, each package contains:

- `manifest.json`
  - Scenario metadata, schema version, simulation config, description, and optional `baseline_scenario_id`.
- `kpi_history.csv`
  - Timestep KPI history for replay and charting.
- `transactions.csv`
  - Transaction-level flow data over time.
- `product_graph.csv`, `product_nodes.csv`, `firm_nodes.csv`, `firm_supplier_edges.csv`
  - Static network structure used for graph views and lookup tables.
- `demand_timeseries.csv`, `exog_supply_timeseries.csv`
  - Demand and exogenous supply traces.
- `per_product_timeseries.csv`, `per_firm_timeseries.csv`
  - Derived replay/explainability views.

The schema is enforced in `dashboard/backend/app/scenario_schema.py`.

## Environment And Local Runbook

### Python environment

Install the base Python dependencies from the repo root:

```bash
pip install -r requirements.txt
pip install -e ./TGB/
```

If you are using the local repo virtualenv, existing commands in this project often use `./venv/bin/python ...`.

`requirements-cu117.txt` is only needed for the older CUDA 11.7 training stack and is not required for the dashboard workflow.

### Export a scenario

```bash
./venv/bin/python scripts/export_supplysim_scenario.py --seed 0 --T 50 --scenario-id baseline_seed0
./venv/bin/python scripts/export_supplysim_scenario.py --seed 1 --T 50 --gamma 0.7 --shock-prob 0.003 --scenario-id scenario_seed1 --baseline-scenario-id baseline_seed0
```

By default, exports land in:

```bash
artifacts/scenarios/{scenario_id}
```

### Run the backend

```bash
cd dashboard/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend reads scenarios from `artifacts/scenarios/` by default. Override with:

```bash
export SUPPLYSIM_SCENARIO_ROOT=/absolute/path/to/artifacts/scenarios
```

### Run the frontend

```bash
cd dashboard/frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:8000` unless `VITE_API_BASE_URL` is set.

### Tests

Backend:

```bash
pytest dashboard/backend/tests -q
```

Frontend:

```bash
cd dashboard/frontend
npm run test
npm run test:e2e
```

## Paper And Upstream Context

This repo still contains the original paper-oriented structure and can still be used for that work when needed:

- Synthetic simulator core: `TGB/modules/synthetic_data.py`
- Model training / experiment entry points: `TGB/examples/linkproppred/general/`
- Data preprocessing / hypergraph registration: `register_data/`

We ran the original research on two proprietary real-world datasets plus synthetic SupplySim variants. The real-world datasets are not included here.

## Citation

If you use this work, please cite:

```tex
@inproceedings{chang2025supplychain,
  author  = {Serina Chang and Zhiyin Lin and Benjamin Yan and Swapnil Bembde and Qi Xiu and Chi Heem Wong and Yu Qin and Frank Kloster and Alex Luo and Raj Palleti and Jure Leskovec},
  title   = {Learning Production Functions for Supply Chains with Graph Neural Networks},
  booktitle = {Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence},
  year    = {2025},
}
```
