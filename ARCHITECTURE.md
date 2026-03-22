# SupplySim Shock Architecture (V2)

## Overview

SupplySim V2 replaces the original flat shock model with a four-layer architecture that controls how supply chain disruptions are generated, propagate, recover, and interact with policy decisions. The goal: make policies genuinely compete — including at zero expedite budget — by producing asymmetric shocks where some firms are disrupted while others remain viable rerouting targets.

The architecture is implemented across three files:

| File | Role |
|------|------|
| `scripts/shock_architecture.py` | Layer configs, ShockEngine, PolicyInterfaceLayer, SupplyGraph |
| `scripts/supplysim_env_v2.py` | V2 environment wrapper (drop-in replacement for V1) |
| `scripts/experiment_utils.py` | Experiment runner with `run_single_experiment_v2()` |

## The Four Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Topology         — network structure              │
│  Layer 2: Shock Generation — how disruptions are seeded     │
│  Layer 3: Shock Dynamics   — how shocks evolve over time    │
│  Layer 4: Policy Interface — friction and constraints       │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1 — Network Topology (`TopologyConfig`)

Controls the static structure of the supply network. Built once per episode in `SupplyGraph`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_regions` | 3 | Number of geographic regions firms are assigned to |
| `region_assignment` | `"random"` | How firms are assigned to regions (`random`, `clustered`, `tiered`) |
| `chokepoint_fraction` | 0.2 | Fraction of firms marked as chokepoints (highest connectivity) |
| `supplier_capacity_cv` | 0.4 | Coefficient of variation for supplier capacity (lognormal) |
| `min_num_suppliers` | 2 | Minimum alternative suppliers per (buyer, product) |
| `max_num_suppliers` | 3 | Maximum alternative suppliers per (buyer, product) |
| `num_inner_layers` | 2 | Number of intermediate layers in the product DAG |
| `num_per_layer` | 10 | Firms per layer |

**Key structures built by `SupplyGraph`:**

- **`firm_region`**: Maps each firm to a region ID. Regions determine which firms are hit by regional shocks and which remain unaffected (enabling rerouting).
- **`is_chokepoint`**: Flags high-connectivity firms. Chokepoints have disproportionate impact when disrupted.
- **`supplier_capacity`**: Per-firm capacity multiplier drawn from a lognormal distribution. Creates heterogeneity in how much each firm can absorb redirected orders.
- **`firm_neighbors`**: Adjacency graph for contagion propagation. Two firms are neighbors if they supply the same product or trade directly.

### Layer 2 — Shock Generation (`ShockGenerationConfig`)

Controls how new disruption events are created each timestep.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shock_prob` | 0.20 | Per-timestep probability of a new shock |
| `magnitude_mean` | 0.85 | Mean supply reduction (85% lost during shock) |
| `magnitude_cv` | 0.20 | Magnitude variability (beta distribution) |
| `magnitude_dist` | `"beta"` | Distribution for magnitude sampling (`fixed`, `uniform`, `beta`) |
| `epicenter_bias` | `"region"` | How epicenters are chosen (`uniform`, `centrality`, `region`) |
| `localized_firm_fraction` | 0.5 | Fraction of firms hit in a localized shock |
| `max_concurrent_shocks` | 3 | Cap on simultaneously active shocks |
| `shock_overlap_prob` | 0.3 | Probability of allowing overlapping shocks |

**Three shock event types:**

| Type | Behavior | Rerouting Value |
|------|----------|-----------------|
| **Localized** | Hits ~50% of firms for one product. Prefers same-region firms. Always leaves at least 1 firm unaffected. | High — unaffected firms in other regions are reroute targets |
| **Regional** | Hits ALL firms in one region across all their exogenous products. | High — firms in other regions are completely unaffected |
| **Cascade** | Starts like localized, but contagion spreads more aggressively downstream. | Moderate — depends on contagion reach |

**Event type probability presets:**

```python
"local":    {"localized": 0.8, "regional": 0.1, "cascade": 0.1}
"regional": {"localized": 0.2, "regional": 0.6, "cascade": 0.2}
```

**Why asymmetric shocks matter:** In V1, shocks hit all firms equally (firm_shock_fraction=1.0 by default), making rerouting pointless — every alternative supplier was just as disrupted. V2 shocks hit a subset of firms, creating information asymmetry that rewards policies capable of identifying and routing to unaffected suppliers.

### Layer 3 — Shock Dynamics (`ShockDynamicsConfig`)

Controls how shocks evolve after onset: duration, contagion spread, and recovery.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration_mean` | 15.0 | Mean shock duration (geometric distribution) |
| `duration_dist` | `"geometric"` | Duration sampling distribution |
| `contagion_radius` | 2 | Max hops a shock can spread in the product DAG |
| `contagion_prob_per_hop` | 0.4 | Per-hop probability of spreading to a downstream product |
| `recovery_shape` | `"linear"` | How supply recovers after shock ends (`instant`, `linear`, `concave`) |
| `recovery_steps` | 8 | Number of timesteps for full recovery |
| `demand_surge_enabled` | `false` | Whether shocks trigger demand spikes |
| `demand_surge_multiplier` | 1.3 | Demand multiplier during surge |

**Shock lifecycle:**

```
Onset ──▶ Active (duration_mean steps) ──▶ Recovering (recovery_steps) ──▶ Ended
  │                    │
  │                    ├── Contagion spreads downstream each step
  │                    │   (only to firms in AFFECTED regions)
  │                    │
  │                    └── Supply multiplied by (1 - magnitude)
  │
  └── Supply multipliers compound multiplicatively for overlapping shocks
```

**Asymmetric contagion:** When a shock spreads to a downstream product, only firms in the *same regions as already-affected firms* are added. Firms in unaffected regions remain clean. This preserves rerouting value even as shocks cascade through the product DAG.

**Recovery model (linear):**

```
supply_retained = (1 - magnitude) + magnitude × (steps_into_recovery / recovery_steps)
```

At recovery start, supply is at `(1 - magnitude)` of baseline. It linearly returns to 100% over `recovery_steps` timesteps.

### Layer 4 — Policy Interface (`PolicyInterfaceConfig`)

Controls the friction and constraints that policies face when responding to disruptions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reroute_setup_delay` | 0 | Timesteps before a reroute takes effect |
| `reroute_capacity_fraction` | 1.0 | Max load a supplier can absorb from reroutes (fraction of capacity) |
| `reroute_supply_bonus` | 150.0 | Bonus supply units when rerouting away from a shocked supplier |
| `expedite_budget` | 50,000 | Total expedite budget per episode |
| `expedite_convexity` | 0.0 | Exponent for convex cost model (0 = linear) |
| `expedite_capacity_cap` | 1.0 | Max fraction of requested expedite units actually granted |
| `observability` | `"full"` | What policies can see (`full`, `delayed`, `noisy`) |

**Reroute supply bonus — the key mechanism:**

When a policy reroutes orders away from a shocked supplier to an unshocked one, the new supplier receives bonus supply units injected directly into the exogenous schedule:

```python
severity = max(0, 1 - old_supplier_supply / per_firm_baseline)
if severity > 0.1:
    bonus = reroute_supply_bonus × severity  # e.g., 150 × 0.85 = 127 units
    exog_schedule[t][(new_supplier, product)] += bonus
```

**Why this exists:** In the real world, rerouting during a crisis involves emergency contracts, buffer stock activation, and expedited production — not just redirecting existing orders. Without this mechanic, each reroute only redirects ~2% of buyer-product edges (K=3 out of ~82 edges), which is too small to measurably reduce backlog. The bonus amplifies each reroute's impact to model the real-world phenomenon of emergency supplier activation.

**Reroute setup delay:** When `reroute_setup_delay > 0`, reroutes don't take effect immediately. The policy must initiate the reroute, wait N timesteps, then the reroute is applied. This penalizes reactive policies.

## Environment: `SupplySimEnvV2`

Drop-in replacement for V1 with the same API:

```python
from scripts.supplysim_env_v2 import SupplySimEnvV2
from scripts.shock_architecture import ArchitectureConfig

config = ArchitectureConfig()
env = SupplySimEnvV2(seed=0, T=80, arch_config=config, shock_architecture_enabled=True)
obs = env.reset(default_supply=100, warmup_steps=10)

done = False
while not done:
    action, explanation = policy_fn(obs, env.t, env)
    obs, reward, done, info = env.step(action)
```

**Step loop (per timestep):**

1. **ShockEngine.step(t)** — Ages existing shocks, applies contagion, samples new events, computes supply multipliers
2. **Apply multipliers** — Multiply exog_schedule entries for affected (firm, product) pairs
3. **Apply demand surge** — If enabled, multiply demand schedule
4. **Expire old orders** — Orders older than `max_order_age` become lost sales
5. **Apply action** — Process reroutes (with PolicyInterfaceLayer constraints) and expedites
6. **Add new demand** — Consumer orders placed
7. **Simulate firms** — FIFO order processing per firm (random order, seeded)
8. **Update inventories** — Transaction settlement
9. **Place new orders** — Firms order inputs from suppliers (gamma-weighted preferred supplier)
10. **Record KPIs** — Backlog, fill rate, shock exposure, etc.

**Observation dict** (same as V1):
```python
{
    "t": int,
    "inventories": np.ndarray,  # (num_firms, num_products)
    "pending": np.ndarray,       # (num_firms, num_products)
    "num_open_orders": int,
    "last_kpis": dict,
}
```

**Action dict** (same as V1):
```python
{
    "reroute": [(buyer, product, new_supplier), ...],  # up to K per step
    "supply_multiplier": {(firm, product): float, ...}  # expedite multipliers
}
```

## Calibrated Parameters

These were validated through a 9-test suite (T1–T9) producing separation scores in the [0.10, 0.35] range across all 8 regime × budget combinations.

| Parameter | Value | Source |
|-----------|-------|--------|
| `default_supply` | 100 (tight=50, loose=100) | Supply sweep + T9 |
| `shock_prob` | 0.20 | T1/T9 |
| `magnitude_mean` | 0.85 | T1/T9 |
| `localized_firm_fraction` | 0.5 | Asymmetry calibration |
| `duration_mean` | 15 | T4 |
| `contagion_radius` | 2 | T5 |
| `contagion_prob_per_hop` | 0.4 | T5 |
| `recovery_shape` | `"linear"` | T4/T6 |
| `reroute_supply_bonus` | 150 | Mechanism calibration |
| `reroute_setup_delay` | 0 | T7 |
| `num_regions` | 3 | T2 |
| `supplier_capacity_cv` | 0.4 | T3 |
| `expedite_budget` | 50,000 | Policy pilot |
| `K` (reroute budget) | 3 | Policy pilot |
| `gamma` (supplier stickiness) | 0.8 | Default |
| `T` (episode length) | 80 | T9 |
| `warmup_steps` | 10 | Default |

## Experiment Regimes

Eight regimes defined by crossing supply tightness, shock geography, and budget:

| Regime | `default_supply` | Event Type | `expedite_budget` |
|--------|-----------------|------------|-------------------|
| `tight_local_eb0` | 50 | local | 0 |
| `tight_local_eb50k` | 50 | local | 50,000 |
| `tight_regional_eb0` | 50 | regional | 0 |
| `tight_regional_eb50k` | 50 | regional | 50,000 |
| `loose_local_eb0` | 100 | local | 0 |
| `loose_local_eb50k` | 100 | local | 50,000 |
| `loose_regional_eb0` | 100 | regional | 0 |
| `loose_regional_eb50k` | 100 | regional | 50,000 |

**Tight vs loose:** At `ds=50`, the system is supply-constrained even without shocks (fill rate ~0.25). At `ds=100`, the system operates near capacity (fill rate ~0.55–0.65), so shocks cause meaningful degradation.

**Local vs regional:** Local regimes (80% localized shocks) create single-product disruptions. Regional regimes (60% regional shocks) create multi-product, multi-firm disruptions that test contagion and cross-product reasoning.

**eb=0 vs eb=50k:** Zero-budget regimes isolate rerouting value. Budget regimes test the interplay between rerouting and expediting.

## Policy Comparison

Seven policies compete across all regimes:

| Policy | Mechanism | Decision Logic |
|--------|-----------|---------------|
| `no_intervention` | Baseline | No action taken |
| `random_reroute` | Reroute only | Randomly reassign K buyer-product edges |
| `reroute_only` | Reroute only | Same as backlog_greedy's rerouting (congestion-based) |
| `expedite_only` | Expedite only | Boost supply for high-severity products (multiplier 2.0) |
| `backlog_greedy` | Reroute + expedite | Reroute by congestion (min open orders), expedite by severity |
| `graph_informed` | Reroute + expedite | Uses shock severity, ripple impact, chokepoint criticality, path scores |
| `mip` | Reroute + expedite | Mixed-integer program (optimal but slow) |

**Why `graph_informed` wins:** It uses supply-side signals (shock severity, ripple propagation) that detect disruptions *before* backlogs accumulate. `backlog_greedy` reacts to congestion (open order counts), which is a lagging indicator. This information advantage is amplified by the reroute supply bonus — `graph_informed` routes to the right unshocked suppliers, triggering maximum bonus.

## Validation Results

All 9 validation tests pass. T9 (integration) results:

| Regime | GI Separation |
|--------|--------------|
| `tight_local_eb0` | 0.131 |
| `tight_local_eb50k` | 0.180 |
| `tight_regional_eb0` | 0.141 |
| `tight_regional_eb50k` | 0.215 |
| `loose_local_eb0` | 0.127 |
| `loose_local_eb50k` | 0.153 |
| `loose_regional_eb0` | 0.172 |
| `loose_regional_eb50k` | 0.219 |

Separation score = `(backlog_auc(no_intervention) - backlog_auc(policy)) / backlog_auc(no_intervention)`. All values in the target [0.10, 0.35] range.

## V1 vs V2 Comparison

| Aspect | V1 (`SupplySimEnv` + `calibrated_scenario.py`) | V2 (`SupplySimEnvV2` + `shock_architecture.py`) |
|--------|------|------|
| Shock model | Flat: per-product per-timestep Bernoulli, all firms equally affected | Structured: 3 event types, asymmetric firm targeting, lifecycle management |
| Rerouting value | Near-zero at eb=0 (all suppliers equally hit) | 10–22% backlog reduction at eb=0 |
| Contagion | None | Region-aware DAG propagation |
| Recovery | Multiplicative per-firm (`supply *= recovery_rate`) | Shaped (linear/concave) with configurable duration |
| Configuration | Flat kwargs scattered across functions | Bundled `ArchitectureConfig` with 4 typed layers |
| Demand surge | Not modeled | Optional multiplier tied to shock activity |
| Reroute constraints | None | Setup delay, capacity fraction |
| Expedite cost | Linear only | Linear, convex, or step function |
| Observability | Full | Full, delayed, or noisy |

## File Map

```
scripts/
├── shock_architecture.py          # Layer configs, ShockEngine, PolicyInterfaceLayer, SupplyGraph
├── supplysim_env_v2.py            # V2 environment (uses shock architecture)
├── supplysim_env.py               # V1 environment (legacy, still used by some scripts)
├── calibrated_scenario.py         # V1 calibrated scenario generator
├── validate_shock_architecture.py # 9-test validation suite (T1–T9)
├── calibrate_shock_architecture.py# Calibration search
├── compare_architectures.py       # Head-to-head V1 vs V2 comparison
├── experiment_utils.py            # Shared experiment infrastructure (V1 + V2 runners)
├── run_panel1_v2.py               # Panel 1: Core benchmark (V2)
├── graph_informed_optimizer.py    # Graph-informed greedy policy
├── baseline_policies.py           # 5 baseline policies
└── analyze_and_plot.py            # Analysis and plotting
```
