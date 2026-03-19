# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SupplySim is a thesis-stage supply chain simulation with an explainable replay dashboard and a graph-informed greedy optimizer for budget-constrained disruption response. It originated from an AAAI 2025 paper (Chang et al.).

Two active workflows:
1. **Dashboard**: run simulation → export scenario package → serve via FastAPI → explore in React dashboard
2. **Optimizer experiments**: run policies across seeds → compare backlog AUC, fill rates, spend → generate plots

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ./TGB/

# Export scenarios (local venv)
./venv/bin/python scripts/export_supplysim_scenario.py --seed 0 --T 50 --scenario-id baseline_seed0
./venv/bin/python scripts/export_supplysim_scenario.py --seed 1 --T 50 --gamma 0.7 --shock-prob 0.003 \
  --scenario-id scenario_seed1 --baseline-scenario-id baseline_seed0

# Backend (FastAPI)
cd dashboard/backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (React + TypeScript)
cd dashboard/frontend && npm install && npm run dev

# Tests
pytest dashboard/backend/tests -q                    # backend API tests
./venv/bin/python -m pytest tests/ -v                 # optimizer unit + integration tests
./venv/bin/python -m pytest tests/test_optimizer.py -v -k "test_severity"  # single test
cd dashboard/frontend && npm run test
cd dashboard/frontend && npm run test:e2e

# Optimizer experiments
./venv/bin/python scripts/run_experiments.py --seeds 30 --T 50 --shock-prob 0.01
./venv/bin/python scripts/run_experiments.py --seeds 5 --T 30 --shock-prob 0.02 --experiment-id quick_test

# Parameter sweep
./venv/bin/python scripts/run_experiments.py --seeds 10 --T 50 --sweep shock_prob --sweep-values 0.005,0.01,0.02,0.05

# Analyze results and generate plots
./venv/bin/python scripts/analyze_results.py --experiment-dir artifacts/experiments/<experiment_id>

# Smoke test (quick env validation)
./venv/bin/python scripts/quick_smoke_test.py
```

Override scenario root: `export SUPPLYSIM_SCENARIO_ROOT=/path/to/scenarios`
Frontend API override: set `VITE_API_BASE_URL` (defaults to `http://localhost:8000`).

## Architecture

### Simulation Core

`TGB/modules/synthetic_data.py` — the agent-based simulator engine. Generates static graphs (firms, products, supplier edges), initial conditions, demand/exogenous-supply schedules with shocks, and runs per-firm order processing each timestep.

`scripts/supplysim_env.py` — gym-like wrapper (`SupplySimEnv`) around synthetic_data. Key API:

- `env.reset(init_inv, init_supply, init_demand, shock_prob, ...)` → returns obs dict `{t, inventories, pending, num_open_orders, last_kpis}`
- `env.step(action)` → returns `(obs, reward, done, info)` where `info = {transactions: DataFrame, num_orders, kpis: dict}`
- Action format (dict): `{"reroute": [(buyer, product, new_supplier), ...], "supply_multiplier": {(firm, product): multiplier, ...}}`
- Reroutes are free; expedites cost `c0 * (1 + alpha * depth(product))` per unit added
- Budget tracking: `expedite_budget_remaining` decremented each step; over-budget requests are proportionally downscaled (`alpha_scale`)
- `expedite_m_max` clamps multipliers; reductions (m < 1) bypass budget

### Scenario Export

`scripts/export_supplysim_scenario.py` — runs a full episode and serializes to `artifacts/scenarios/{id}/`. Each package contains `manifest.json`, `kpi_history.csv`, `transactions.csv`, product/firm graph CSVs, demand/supply timeseries, and derived per-product/per-firm timeseries.

Schema enforced in `dashboard/backend/app/scenario_schema.py`.

### Dashboard Backend

`dashboard/backend/app/main.py` — FastAPI endpoints: `/scenarios`, `/scenarios/{id}/kpis`, `/scenarios/{id}/graph`, `/scenarios/{id}/explain`, `/scenarios/{id}/compare`.

`dashboard/backend/app/services/explain_service.py` — computes explainability signals **post-hoc from exported CSVs** (not at runtime):
- **Ripple impact**: `0.5 * tx_drop_ratio + 0.3 * backlog_increase_ratio + 0.2 * proximity` (proximity = 1/(1+dist) from shocked product in product DAG)
- **Chokepoint criticality**: `0.4 * out_degree_norm + 0.3 * coverage_norm + 0.3 * constrained_norm` (constrained = open_orders / (open_orders + outbound + ε))
- **Shock paths**: shortest paths from shocked exogenous products to consumers, scored by `severity * (0.6 + ripple_component) / (1 + dist)`

### Graph-Informed Optimizer

`scripts/graph_informed_optimizer.py` — rolling greedy optimizer (the core thesis contribution). At each timestep:
- `SignalComputer` computes runtime versions of explainability signals (shock severity, ripple impact, chokepoint criticality, path scores) directly from env state — these are runtime equivalents of the post-hoc signals in `explain_service.py`
- `CandidateEnumerator` lists all feasible reroutes and expedites
- Greedy selection picks top-K reroutes by value score, then expedites by value-per-cost
- Returns `(action_dict, explanation_dict)` where action is directly passable to `env.step()`

`scripts/baseline_policies.py` — 5 comparison policies (no-intervention, random reroute, backlog-only greedy, expedite-only, reroute-only). All share the same signature: `policy_fn(obs, t, env) -> (action, explanation)`.

`scripts/run_experiments.py` — runs all policies across seeds with `run_policy()`, supports parameter sweeps, exports to `artifacts/experiments/`.

`scripts/analyze_results.py` — loads experiment CSVs, computes aggregate stats with CIs, paired significance tests (Wilcoxon/t-test), and generates bar charts, timeseries, and Pareto plots to `artifacts/figures/`.

### Dashboard Frontend

`dashboard/frontend/src/` — React + TypeScript control-tower UI with timeline slider, graph visualization, explain panel, and scenario comparison.

## Key Internal Data Structures

- `inputs2supplier`: dict mapping `(buyer_firm, product) → supplier_firm` — reroutes mutate this
- `exog_schedule`: list of dicts (one per timestep), each mapping `(firm, product) → supply_amount` — expedites mutate current timestep entry
- `prod_graph`: DataFrame with columns `source, dest, units, layer` — the product dependency DAG
- `firm2prods` / `prod2firms`: mappings between firms and products they produce/use
- `inventories`: numpy array shaped `(num_firms, num_products)`
- `curr_orders`: dict of `(supplier, product) → [(buyer, amount), ...]`

## Boundaries

- **Active code**: `scripts/`, `dashboard/backend/app/`, `dashboard/frontend/src/`, `artifacts/scenarios/`
- **Upstream/legacy** (avoid modifying): `TGB/`, `register_data/`, `synthetic_data/`
- **Generated artifacts** (not source): `node_modules/`, `dist/`, `__pycache__/`, ad hoc files in repo root
- `requirements-cu117.txt` is for the old CUDA training stack, not needed for dashboard work

## Calibration Results (2026-03-19)

### Exogenous Supply Flow
Exogenous supply is a **per-timestep throughput cap**, not a stockpile. Each timestep, `exog_supp[(f,p)]` units are available for FIFO order fulfillment. Realized via Poisson noise around the configured mean.

### Pipeline Characteristics (seed=42, 5 exog products, 31 exog firm-product pairs)
- Steady-state consumption: ~2.6M units/step total, ~84,531 per (firm, product) pair
- Structural fill rate ceiling: **~0.59** even with unlimited supply (pipeline delay across 6 tier levels)
- Supply transition zone: fill rate crashes between ds=200K (0.002) and ds=500K (0.563) per pair

### Calibrated Parameters
```
default_supply = 500,000    # Per (firm, product), ~6x avg consumption
recovery_rate  = 1.05       # Slow recovery amplifies shock persistence (was 1.25)
warmup_steps   = 10         # Shock-free warmup
shock_fraction = 0.3        # 70% supply reduction during shock (primary severity axis)
T              = 60
K              = 3          # Reroute budget per step
expedite_budget = 50,000
```

### Calibration Validation
| Config                          | Backlog AUC | Fill Rate | Delta vs baseline |
|---------------------------------|-------------|-----------|-------------------|
| No shocks (baseline)            | 91,005      | 0.563     | —                 |
| sp=0.15, sf=0.3, rr=1.05       | 118,282     | 0.216     | +30.0%            |
| sp=0.15, sf=0.1, rr=1.05       | 140,982     | 0.061     | +54.9%            |
| sp=0.15, sf=0.5, rr=1.05       | 106,454     | 0.353     | +17.0%            |

### Policy Pilot (3 seeds, calibrated params)
| Regime  | Policy           | Mean AUC | vs no_intervention |
|---------|------------------|----------|--------------------|
| mild    | no_intervention  | 98,197   | —                  |
| mild    | graph_informed   | 95,194   | +3.1%              |
| mild    | backlog_greedy   | 92,804   | +5.5%              |
| severe  | no_intervention  | 122,173  | —                  |
| severe  | graph_informed   | 111,902  | +8.4%              |
| severe  | backlog_greedy   | 114,025  | +6.7%              |
| **Regime effect** | severe vs mild baseline | | **+24.4%** |

Key finding: graph_informed's advantage over backlog_greedy **increases with disruption severity** (graph-informed outperforms by 1.7pp in severe vs underperforming by 2.4pp in mild). Expediting alone has zero effect — all improvement comes from rerouting.

### Running Full Experiments
```bash
# Calibration sweep (generates calibration_table.json)
./venv/bin/python scripts/run_calibration.py --output artifacts/experiments/calibration_table.json --seeds 10

# Full regime experiment (uses calibrated defaults)
./venv/bin/python scripts/run_regime_experiment.py --seeds 20 --workers 4

# Quick pilot (3 seeds, fast)
./venv/bin/python scripts/policy_pilot.py
```

## Conventions

- All numeric rounding uses `np.round(..., num_decimals)` with `num_decimals=5`
- Product/firm names follow `product0`, `firm1` convention
- Firms process orders in random order each timestep (seeded RNG)
- Exogenous products = sources in product DAG (no predecessors); consumer products = sinks (no successors)
- Shock severity: `clip(1 - current_supply / baseline_supply, 0, 1)` where baseline = 90th percentile of planned supply
