# Simulator Rework Changelog (2026-03-19)

## Motivation

The simulator was built for learning production functions (TGB link prediction), not for testing intervention policies. Three structural problems made it unsuitable for disruption-response benchmarking:

1. **Rerouting was mechanically broken.** Pending orders at the old supplier suppressed new orders via `clip(needs - inv - pending, 0, None)`, so reroutes never generated new orders until old ones cleared.
2. **Supply dwarfed demand by orders of magnitude.** Even calibrated to 500K, supply far exceeded what the pipeline could consume. Shocks created marginal backlog increases but the system was never truly supply-constrained.
3. **BOM amplification was too extreme.** With 4 inner layers and 2-4 inputs per product at 1-4 units each, demand amplified ~400x through the pipeline. Combined with all-or-nothing FIFO batching, interventions at the source couldn't reach consumers before shocks naturally recovered.

---

## Changes Implemented

### Change 1: Transfer Pending Orders on Reroute (CRITICAL)

**File:** `scripts/supplysim_env.py` — `_apply_action_dict()`

**Before:** When `inputs2supplier[(buyer, product)] = new_supplier`, only the supplier mapping changed. Pending orders remained at the old supplier, and the ordering formula `clip(needs - inv - pending, 0, None)` saw `pending > 0` and suppressed new orders to the new supplier.

**After:** When a reroute is applied:
1. The supplier mapping is updated
2. All of the buyer's pending orders at the old supplier are transferred to the new supplier's queue
3. The `pending` array is unchanged (buyer still expects delivery)

This ensures the reroute takes effect immediately rather than waiting for old orders to clear.

### Change 2: Reduce Network Depth to 2 Inner Layers

**Files:** `TGB/modules/synthetic_data.py` — `generate_static_graphs()`, `scripts/supplysim_env.py`

**Before:** `make_product_graph()` defaulted to `num_inner_layers=4`, creating a 6-tier pipeline (exog → 4 inner → consumer) with ~5-step propagation delay.

**After:** `generate_static_graphs()` now accepts and passes through `num_inner_layers` (default: 2). The `SupplySimEnv` constructor accepts `num_inner_layers` and `num_per_layer` and forwards them to graph generation. This reduces pipeline latency from ~5 steps to ~3 steps.

**Product count:** 30 (5 exog + 2×10 inner + 5 consumer base, though some inner products become effective consumers with reduced fan-in).

### Change 3: Reduce Suppliers Per Product to 2-3

**Files:** `TGB/modules/synthetic_data.py` — `generate_static_graphs()`, `scripts/supplysim_env.py`

**Before:** `make_supplier_product_graph()` defaulted to `min_num_suppliers=4, max_num_suppliers=8`.

**After:** Defaults changed to `min_num_suppliers=2, max_num_suppliers=3`. These are configurable via `SupplySimEnv` constructor. With fewer suppliers per product, reroute decisions are consequential — rerouting buyer A to the only alternative supplier S2 may crowd out buyer B.

### Change 4: Right-Size Supply to Demand

**Files:** `TGB/modules/synthetic_data.py`, `scripts/supplysim_env.py`, `scripts/calibrated_scenario.py`

This required two sub-changes:

**4a: Reduce BOM fan-in.** Added `min_inputs`, `max_inputs`, `min_units`, `max_units` parameters to `generate_static_graphs()` and `make_product_graph()`. Defaults changed from `(2,4,1,4)` to `(1,2,1,1)`. This eliminates the ~400x BOM amplification that made the system throughput-constrained regardless of supply level.

**4b: Calibrate supply to demand.** With `init_demand=10` and the reduced BOM:
- No-shock fill rate with unlimited supply: 0.938 (structural ceiling from pipeline delay)
- No-shock fill rate with `default_supply=100` per exog pair: 0.938 (supply not the bottleneck)
- Shocked fill rate (`shock_fraction=0.3`, `shock_prob=0.15`): 0.708 (genuine supply shortage)

The exogenous supply transition zone is between `supply=30` (severely constrained) and `supply=100` (pipeline-constrained). Setting `default_supply=100` puts the system at the edge where shocks create real impact.

### Change 5: Order Expiry

**File:** `scripts/supplysim_env.py` — `step()`

**Before:** Orders sat in `curr_orders` indefinitely, accumulating unbounded backlog.

**After:** At the start of each `step()`, orders older than `max_order_age` (default: 10 steps) are removed. Expired orders:
- Are counted as `lost_sales_units` in the KPI dict
- Have their `pending` amounts decremented for the buyer firm

Orders now use 3-tuple format `(buyer, amount, t_placed)` throughout the codebase. All order consumers updated to handle both 2-tuple (legacy) and 3-tuple formats via `for buyer, amt, *_ in orders:`.

**Impact:** With expiry, final backlog is 404 vs 526 without (23% reduction). Backlog plateaus instead of growing monotonically.

### Change 6: Warm-Start KPIs

**File:** `scripts/supplysim_env.py` — `step()`

**Before:** KPIs (fill rate, backlog) were tracked from `t=0`, including the cold-start ramp-up period where no orders could possibly be fulfilled.

**After:** Cumulative demand and fulfillment are only accumulated for `t >= kpi_start_step` (default: `num_inner_layers + 2 = 4`). This gives the pipeline time to fill before measuring, producing cleaner cumulative fill rate metrics.

---

## Calibrated Parameters

| Parameter | Old Value | New Value | Rationale |
|---|---|---|---|
| `num_inner_layers` | 4 | 2 | Reduce pipeline latency |
| `min_num_suppliers` | 4 | 2 | Consequential reroute decisions |
| `max_num_suppliers` | 8 | 3 | Consequential reroute decisions |
| `min_inputs` | 2 | 1 | Reduce BOM amplification |
| `max_inputs` | 4 | 2 | Reduce BOM amplification |
| `min_units` | 1 | 1 | (unchanged) |
| `max_units` | 4 | 1 | Eliminate multiplicative BOM amplification |
| `init_demand` | 1-2 | 10 | Scale consumer demand |
| `default_supply` | 500,000-1,000,000 | 100 | Right-sized to actual exog throughput |
| `recovery_rate` | 1.25 | 1.05 | Slower recovery amplifies shock persistence |
| `warmup_steps` | 15 | 10 | Shock-free warmup |
| `max_order_age` | ∞ | 10 | Bounded backlog |
| `kpi_start_step` | 0 | 4 | Pipeline fill before measuring |

---

## Validation Results

| Test | Status | Key Metric |
|---|---|---|
| V1: Reroute mechanism | PASS | Orders transferred, supplier mapping updated |
| V2: Supply-demand balance | PASS | No-shock fill rate: 0.938 |
| V3: Shock sensitivity | PASS | +30.7% backlog AUC, -21.2% fill rate drop |
| V4: Policy differentiation | PASS | graph_informed: +22.2% vs no_intervention (severe) |
| V5: Order expiry | PASS | Final backlog 404 vs 526 without expiry |
| V6: Regime variation | PASS | 10.7% backlog range across shock_prob sweep |

### Policy Differentiation Detail

**Mild regime** (shock_prob=0.05):

| Policy | Backlog AUC | Fill Rate | vs no_intervention |
|---|---|---|---|
| no_intervention | 23,326 | 0.808 | — |
| graph_informed | 19,724 | 0.924 | +15.4% |
| backlog_greedy | 19,276 | 0.922 | +17.4% |
| reroute_only | 24,029 | 0.741 | -3.0% |
| expedite_only | 19,466 | 0.947 | +16.5% |

**Severe regime** (shock_prob=0.15):

| Policy | Backlog AUC | Fill Rate | vs no_intervention |
|---|---|---|---|
| no_intervention | 26,858 | 0.708 | — |
| graph_informed | 20,888 | 0.879 | +22.2% |
| backlog_greedy | 20,817 | 0.853 | +22.5% |
| reroute_only | 26,015 | 0.692 | +3.1% |
| expedite_only | 21,453 | 0.888 | +20.1% |

Key findings:
- **graph_informed advantage increases with severity** (15.4% mild → 22.2% severe)
- **reroute_only now differs from backlog_greedy** (reroutes work, 25% AUC difference)
- **expedite_only shows real effect** (supply now matters at the margin)
- All 23 existing unit/integration tests pass unchanged

---

## Files Modified

| File | Changes |
|---|---|
| `TGB/modules/synthetic_data.py` | `generate_static_graphs()` accepts graph/BOM params; 3-tuple orders; exog product detection fix |
| `scripts/supplysim_env.py` | All 6 changes: reroute transfer, graph params, BOM params, order expiry, KPI warm-start |
| `scripts/calibrated_scenario.py` | New defaults, pass-through for all new params |
| `scripts/baseline_policies.py` | 3-tuple order compatibility |
| `scripts/graph_informed_optimizer.py` | 3-tuple order compatibility |
| `scripts/export_supplysim_scenario.py` | 3-tuple order compatibility |
| `scripts/run_regime_experiment.py` | New calibrated defaults, pass-through params |
| `scripts/validate_rework.py` | New validation script (V1-V6) |

---

## Running Experiments

```bash
# Validate rework
./venv/bin/python scripts/validate_rework.py

# Run full regime experiment
./venv/bin/python scripts/run_regime_experiment.py --seeds 20 --workers 4

# Quick pilot (3 seeds)
./venv/bin/python scripts/run_regime_experiment.py --seeds 3 --workers 1
```
