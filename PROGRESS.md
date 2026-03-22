# SupplySim — Project Progress

## What Is This

SupplySim is a thesis-stage supply chain simulation with an explainable replay dashboard and a graph-informed greedy optimizer for budget-constrained disruption response. It originated from an AAAI 2025 paper (Chang et al.) on learning production functions with GNNs.

Two active workflows:
1. **Dashboard**: simulation → scenario export → FastAPI backend → React frontend for interactive replay
2. **Optimizer experiments**: run policies across seeds/regimes → compare backlog AUC, fill rates, spend → generate analysis

---

## Timeline

### Phase 1: Foundation (commits up to `6b0a254`)
- Forked TGB codebase with synthetic data generation (`TGB/modules/synthetic_data.py`)
- Agent-based simulator: firms process orders FIFO each timestep, exogenous supply feeds the pipeline, demand arrives at consumer products
- Product DAG (exog → inner layers → consumer), firm-product assignment, supplier-buyer graph
- Transaction generation, temporal analysis, network statistics

### Phase 2: Environment Wrapper + Dashboard (`0508e7d` — `d3900a8`)
- **`scripts/supplysim_env.py`**: Gym-like `SupplySimEnv` wrapping the simulator
  - `reset()` → obs dict; `step(action)` → (obs, reward, done, info)
  - Action format: `{"reroute": [...], "supply_multiplier": {...}}`
  - Full KPI tracking: backlog, fill rate, transactions, reroutes, expedite costs
- **`scripts/export_supplysim_scenario.py`**: Serializes episodes to scenario packages (manifest, CSVs, timeseries)
- **Dashboard backend** (`dashboard/backend/app/`): FastAPI with endpoints for scenarios, KPIs, graph, explain, compare
- **Explain service**: Post-hoc explainability signals — ripple impact, chokepoint criticality, shock paths
- **Dashboard frontend** (`dashboard/frontend/src/`): React + TypeScript with timeline slider, graph viz, explain panel, scenario comparison
- 4 demo scenarios exported to `artifacts/scenarios/`

### Phase 3: Optimizer + Baselines (`7c44762`)
- **`scripts/graph_informed_optimizer.py`** — core thesis contribution:
  - `SignalComputer`: runtime shock severity, ripple impact, chokepoint criticality, path scores
  - `CandidateEnumerator`: enumerates feasible reroutes and expedites
  - Rolling greedy: top-K reroutes by value score, then expedites by value-per-cost
- **`scripts/baseline_policies.py`** — 8 comparison policies:
  - no_intervention, random_reroute, threshold, backlog_only (greedy), graph_informed, mip (PuLP), reroute_only, expedite_only
- **`scripts/calibrated_scenario.py`**: Proportional shocks (`shock_supply = default_supply × shock_fraction`), firm-level shocks, warm-up period
- **Experiment infrastructure**:
  - `run_regime_experiment.py` — full grid runner (multiprocessing, resume, dry-run)
  - `run_experiments.py` — single-config runner with sweep support
  - `analyze_regimes.py` — statistical analysis, heatmaps, Pareto plots
  - `policy_pilot.py` — quick 3-seed pilot
- **Calibration** (`run_calibration.py`, `calibration_sweep.py`) → `artifacts/experiments/calibration_table.json`
- **Tests**: 23 unit + integration tests covering signals, budget feasibility, determinism, all policies

### Phase 4: Simulator Rework (2026-03-19) — current

Diagnosed three structural problems making the simulator unsuitable for intervention benchmarking:

1. **Rerouting was broken.** Pending orders at the old supplier suppressed new orders via `clip(needs - inv - pending, 0)`. Reroutes were decorative.
2. **Supply dwarfed demand ~500,000x.** System was throughput-constrained by pipeline batching, never supply-constrained. Shocks had marginal effect.
3. **BOM amplification was ~400x.** With 4 inner layers and 2-4 inputs at 1-4 units each, demand amplified enormously upstream. Combined with all-or-nothing FIFO, interventions at the source couldn't reach consumers before shocks self-healed.

**6 changes implemented:**

| # | Change | File(s) |
|---|--------|---------|
| 1 | **Transfer pending orders on reroute** — move buyer's orders from old to new supplier queue | `supplysim_env.py` |
| 2 | **2 inner layers** (was 4) — pipeline latency ~3 steps instead of ~5 | `synthetic_data.py`, `supplysim_env.py` |
| 3 | **2-3 suppliers/product** (was 4-8) — reroute decisions are consequential | `synthetic_data.py`, `supplysim_env.py` |
| 4 | **Right-sized supply/demand** — reduced BOM fan-in (1-2 inputs, 1 unit each), `init_demand=10`, `default_supply=100` | `synthetic_data.py`, `supplysim_env.py`, `calibrated_scenario.py` |
| 5 | **Order expiry** (10 steps) — bounded backlog, `(buyer, amt, t_placed)` 3-tuple format | `synthetic_data.py`, `supplysim_env.py`, all policy files |
| 6 | **KPI warm-start** — only accumulate metrics after `kpi_start_step=4` | `supplysim_env.py` |

---

## Current Calibrated Parameters

```
num_inner_layers    = 2          # 4-tier pipeline (exog → 2 inner → consumer)
num_per_layer       = 10
min_num_suppliers   = 2          # Consequential reroute decisions
max_num_suppliers   = 3
min_inputs          = 1          # Reduced BOM fan-in
max_inputs          = 2
min_units           = 1          # No multiplicative amplification
max_units           = 1
init_demand         = 10         # Consumer demand per (firm, product) per step
default_supply      = 100        # Per exog (firm, product) — right-sized to throughput
recovery_rate       = 1.05       # Slow recovery amplifies shock persistence
warmup_steps        = 10         # Shock-free warmup
max_order_age       = 10         # Order expiry
kpi_start_step      = 4          # Pipeline fill before measuring
T                   = 60         # Episode length
K                   = 3          # Reroute budget per step
expedite_budget     = 50,000
```

---

## Validation Results (V1–V6)

All pass. Run with `./venv/bin/python scripts/validate_rework.py`.

| Test | Result | Key metric |
|------|--------|------------|
| V1: Reroute mechanism | PASS | Orders transferred, supplier mapping updated |
| V2: Supply-demand balance | PASS | No-shock fill rate: **0.938** |
| V3: Shock sensitivity | PASS | +30.7% backlog AUC, -21.2% fill drop under shocks |
| V4: Policy differentiation | PASS | graph_informed **+22.2%** vs no_intervention (severe) |
| V5: Order expiry | PASS | Final backlog 404 vs 526 without expiry |
| V6: Regime variation | PASS | 10.7% backlog range across shock_prob sweep |

---

## Policy Differentiation (Post-Rework)

### Mild Regime (shock_prob=0.05, shock_fraction=0.3)

| Policy | Backlog AUC | Fill Rate | vs no_intervention |
|--------|-------------|-----------|-------------------|
| no_intervention | 23,326 | 0.808 | — |
| graph_informed | 19,724 | 0.924 | **+15.4%** |
| backlog_greedy | 19,276 | 0.922 | +17.4% |
| expedite_only | 19,466 | 0.947 | +16.5% |
| reroute_only | 24,029 | 0.741 | -3.0% |

### Severe Regime (shock_prob=0.15, shock_fraction=0.3)

| Policy | Backlog AUC | Fill Rate | vs no_intervention |
|--------|-------------|-----------|-------------------|
| no_intervention | 26,858 | 0.708 | — |
| graph_informed | 20,888 | 0.879 | **+22.2%** |
| backlog_greedy | 20,817 | 0.853 | +22.5% |
| expedite_only | 21,453 | 0.888 | +20.1% |
| reroute_only | 26,015 | 0.692 | +3.1% |

**Key findings:**
- graph_informed advantage **increases with disruption severity** (15.4% → 22.2%)
- reroute_only now **differs** from backlog_greedy (25% AUC gap — reroutes actually work)
- expedite_only shows **real effect** (supply is now genuinely scarce)

---

## Regime Axis Probes (Pre-Experiment Planning)

Tested which axes create meaningful **policy separation** (not just level shift):

| Axis | Range tested | Effect on GI advantage | Verdict |
|------|-------------|----------------------|---------|
| `default_supply` (tightness) | 50–200 | 5% at 200 → 22% at 100 | **Strong.** Loose supply collapses all policies. |
| `recovery_rate` (persistence) | 1.02–1.50 | 22% at 1.02 → 9% at 1.50 | **Moderate.** Fast recovery makes interventions moot. |
| `shock_fraction` (severity) | 0.1–0.7 | Non-monotonic, peaks at 0.3 | **Strong.** Sweet spot where shocks matter but are fixable. |
| `firm_shock_fraction` (breadth) | 0.3–1.0 | 18–22%, shifts reroute vs expedite tradeoff | **Moderate.** Controls whether rerouting is viable. |
| `expedite_budget` | 0–50K | 8% at 0 → 22% at 50K | **Very strong.** eb=0 isolates reroute value; eb=50K tests joint optimization. |

### Proposed Full Experiment Grid

**4 axes:**

| Axis | Values | What it tests |
|------|--------|---------------|
| Supply tightness (`default_supply`) | 50, 75, 100, 150 | How scarce are resources? |
| Shock severity (`shock_fraction`) | 0.15, 0.3, 0.5 | How bad are individual disruptions? |
| Recovery speed (`recovery_rate`) | 1.02, 1.05, 1.25 | Do interventions beat waiting? |
| Firm breadth (`firm_shock_fraction`) | 0.3, 0.5, 1.0 | Can you reroute around shocks? |

**Plus budget contrast:** `expedite_budget` = 0, 5000, 50000

**Fixed:** `shock_prob=0.15`, `K=3`, all calibrated BOM params.

**Scale:** 4 × 3 × 3 × 3 = 108 configs × 3 budgets × 8 policies × 20 seeds = **51,840 runs** (~4–8 hours with 4 workers).

---

## File Inventory

### Active Scripts

| Script | Lines | Purpose |
|--------|-------|---------|
| `supplysim_env.py` | ~640 | Gym wrapper, all rework changes |
| `graph_informed_optimizer.py` | ~630 | Core thesis optimizer |
| `baseline_policies.py` | ~620 | 8 comparison policies |
| `calibrated_scenario.py` | ~250 | Calibrated env factory |
| `export_supplysim_scenario.py` | ~410 | Scenario serialization |
| `run_regime_experiment.py` | ~340 | Full experiment grid |
| `run_experiments.py` | ~350 | Legacy experiment runner |
| `analyze_regimes.py` | ~380 | Statistical analysis + plots |
| `validate_rework.py` | ~360 | V1–V6 validation suite |
| `policy_pilot.py` | ~160 | Quick pilot |

### Dashboard

| Component | Location | Tech |
|-----------|----------|------|
| Backend | `dashboard/backend/app/` | FastAPI, 5 endpoints |
| Explain service | `dashboard/backend/app/services/explain_service.py` | Ripple, chokepoint, shock paths |
| Frontend | `dashboard/frontend/src/` | React + TypeScript + Vite |
| Tests | `dashboard/backend/tests/`, `dashboard/frontend/src/test/` | pytest, vitest |

### Tests

- `tests/test_optimizer.py` + `tests/test_integration.py` — 23 tests, all passing
- `dashboard/backend/tests/` — API + schema + compare service tests

---

## Artifacts

```
artifacts/
├── experiments/
│   ├── calibration_table.json       # 16-config calibration (pre-rework)
│   ├── smoke_test/                  # 6 policies × 3 seeds (T=30)
│   └── smoke_test_v2/              # Same, updated
├── figures/
│   ├── backlog_auc_bar.png
│   ├── backlog_timeseries.png
│   └── pareto_backlog_vs_spend.png
└── scenarios/
    ├── baseline_seed0/              # 13 files each
    ├── scenario_seed1/
    ├── demo_baseline/
    └── demo_scenario/
```

---

## How to Run

```bash
# Validate rework
./venv/bin/python scripts/validate_rework.py

# Quick pilot (3 seeds, ~30s)
./venv/bin/python scripts/policy_pilot.py

# Full experiment (20 seeds, ~4-8 hours)
./venv/bin/python scripts/run_regime_experiment.py --seeds 20 --workers 4

# Analyze results
./venv/bin/python scripts/analyze_regimes.py --input artifacts/experiments/regime_mapping/all_results.csv

# Dashboard
cd dashboard/backend && uvicorn app.main:app --reload --port 8000
cd dashboard/frontend && npm run dev

# Tests
./venv/bin/python -m pytest tests/ -v
pytest dashboard/backend/tests -q
```

---

## What's Next

1. **Finalize regime grid** — lock down the 4-axis experimental design from the probe results above
2. **Run full experiment** — 108+ configs × 8 policies × 20 seeds
3. **Statistical analysis** — paired tests, confidence intervals, regime-conditional rankings
4. **Thesis figures** — heatmaps of "what works where", Pareto frontiers, regime transition plots
5. **Dashboard scenarios** — export representative episodes for each regime for interactive exploration
