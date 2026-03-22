# Claude Code: Design and Validate a Defensible Shock Simulation Architecture for SupplySim

## CONTEXT

SupplySim is a thesis-stage supply chain simulator used to benchmark disruption-response
policies. The existing reworked simulator (scripts/supplysim_env.py) has been validated
for basic mechanics but Panel 1 experiments revealed a structural weakness:

  expedite_budget=0  → 0/6 regimes pass differentiation
  expedite_budget=50000 → 6/6 regimes pass

This means "inject more supply" trivially dominates. Rerouting adds no value because
shocks are i.i.d. across suppliers — all alternatives are equally depleted. The graph
topology carries no information because disruption is symmetric.

The goal of this task is to design, implement, and validate a new shock architecture
that makes policies genuinely compete — including without expedite budget — by making
network topology and shock dynamics load-bearing.

**Do not run experiments. Write scripts and architecture only.**
**Do not modify supplysim_env.py or synthetic_data.py directly.**
**All new mechanics go into new files or clearly isolated extension classes.**

---

## THE FOUR-LAYER ARCHITECTURE

Structure the new architecture as four independent, composable layers.
Each layer has its own parameter class and can be varied independently.

### Layer 1 — Network topology
Defines the structural properties of the supply graph.
```python
@dataclass
class TopologyConfig:
    num_inner_layers: int = 2
    num_per_layer: int = 10

    # Regional clustering: suppliers assigned to geographic/categorical regions
    # Shocks will target regions, making some alternatives viable and others not
    num_regions: int = 3                  # how many supplier clusters exist
    region_assignment: str = "random"     # "random" | "clustered" | "tiered"

    # Chokepoint structure: some inner nodes have higher in-degree
    # High-centrality nodes create cascade amplification
    chokepoint_fraction: float = 0.2      # fraction of inner nodes that are chokepoints
    chokepoint_in_degree_multiplier: float = 2.5  # chokepoints have 2.5x normal in-degree

    # Supplier capacity heterogeneity
    # Uniform capacity = rerouting is always feasible at full volume
    # Heterogeneous = some alternatives are bottlenecked
    supplier_capacity_cv: float = 0.4     # coefficient of variation across suppliers
    # capacity = default_supply * Lognormal(1, cv)

    # Path redundancy: how many viable alternative paths exist per consumer product
    # Low redundancy = rerouting is scarce and consequential
    min_num_suppliers: int = 2
    max_num_suppliers: int = 3

    # Lead-time variance: stochastic pipeline latency
    lead_time_mean: int = 2               # steps
    lead_time_cv: float = 0.0            # 0 = deterministic, 0.3 = noisy pipeline
```

### Layer 2 — Shock generation
Defines how disruption events are seeded.
```python
@dataclass
class ShockGenerationConfig:
    shock_prob: float = 0.15              # probability of a new shock event per step

    # Event types — each has its own magnitude and spread profile
    # Probabilities must sum to 1.0
    event_type_probs: dict = field(default_factory=lambda: {
        "localized":   0.5,   # single supplier, random location
        "regional":    0.3,   # all suppliers in one region simultaneously
        "cascade":     0.2,   # starts at one node, spreads to topological neighbors
    })

    # Magnitude: fraction of supply REMOVED (shock_magnitude semantics)
    # shock_supply = default_supply * (1 - magnitude)
    magnitude_dist: str = "beta"          # "fixed" | "uniform" | "beta"
    magnitude_mean: float = 0.70
    magnitude_cv: float = 0.25            # spread around mean

    # Epicenter selection bias
    # "uniform" = any supplier equally likely
    # "centrality" = high-centrality nodes more likely to be epicenter
    # "region" = shocks cluster within regions
    epicenter_bias: str = "region"

    # Overlapping shocks: can multiple shocks be active simultaneously?
    max_concurrent_shocks: int = 3
    shock_overlap_prob: float = 0.2       # probability a new shock overlaps an existing one
```

### Layer 3 — Shock propagation and dynamics
Defines how a shock evolves over time once seeded.
```python
@dataclass
class ShockDynamicsConfig:
    # Persistence: how long does a shock last?
    # "geometric" = memoryless duration, mean = 1 / (1 - persistence_prob)
    duration_dist: str = "geometric"
    duration_mean: float = 12.0           # steps — longer = interventions beat waiting

    # Spatial contagion: does a shock spread to neighboring suppliers?
    contagion_radius: int = 1             # how many hops a shock can spread
    contagion_prob_per_hop: float = 0.3   # probability of spreading to each neighbor

    # Recovery shape: how does supply return after shock ends?
    # "instant" = full recovery at end of duration (current model)
    # "linear"  = gradual ramp back to baseline over recovery_steps
    # "concave" = fast initial recovery, slow tail (realistic)
    recovery_shape: str = "linear"
    recovery_steps: int = 8               # steps to full recovery after shock ends

    # Recovery rate multiplier (legacy parameter — maps to recovery_steps)
    # recovery_steps = ceil(log(0.05) / log(recovery_rate))
    recovery_rate: float = 1.05

    # Demand surge: correlated demand spike during supply shock
    # Models panic-buying, substitution demand, buffer stock building
    demand_surge_enabled: bool = False
    demand_surge_multiplier: float = 1.3  # demand × this during active shocks
    demand_surge_lag: int = 2             # steps after shock onset before surge appears
```

### Layer 4 — Policy interface (competitive pressure)
Defines the friction and constraints policies face.
```python
@dataclass
class PolicyInterfaceConfig:
    # Rerouting constraints
    reroute_setup_delay: int = 0          # steps before new supplier starts fulfilling
    reroute_capacity_fraction: float = 1.0  # how much of a supplier's capacity
                                             # can be absorbed by diverted orders
                                             # 1.0 = unlimited (current), 0.5 = half

    # Expedite cost structure
    expedite_budget: float = 50_000
    expedite_cost_model: str = "linear"   # "linear" | "convex" | "step"
    expedite_convexity: float = 0.0       # 0 = linear, 1 = quadratic, 2 = cubic
    expedite_capacity_cap: float = 1.0    # max expedite units as fraction of default_supply

    # Observability: what can the policy see?
    # "full" = current model (full state visible)
    # "delayed" = shock state revealed with N-step lag
    # "noisy" = supply observations have noise
    observability: str = "full"
    observation_delay: int = 0
    observation_noise_cv: float = 0.0

    # Order expiry
    max_order_age: int = 10
```

---

## TASK 1: IMPLEMENT THE ARCHITECTURE

Write `scripts/shock_architecture.py`.

This file defines the four config dataclasses above and implements:

### ShockState
A runtime object tracking all active shocks:
```python
@dataclass
class ShockState:
    shock_id: int
    event_type: str           # "localized" | "regional" | "cascade"
    epicenter_firm: int
    epicenter_product: int
    affected_firms: List[int] # all firms currently under this shock
    magnitude: float          # fraction of supply removed
    onset_step: int
    expected_duration: int
    contagion_front: Set[int] # firms at the spreading edge
    recovering: bool = False
    recovery_step_start: int = 0
```

### ShockEngine
Manages shock lifecycle each timestep:
```python
class ShockEngine:
    def __init__(self, gen_config: ShockGenerationConfig,
                       dyn_config: ShockDynamicsConfig,
                       topology: SupplyGraph,
                       rng: np.random.Generator):
        ...

    def step(self, t: int) -> Tuple[Dict[Tuple[int,int], float], List[ShockEvent]]:
        """
        Returns:
          supply_multipliers: {(firm, product): multiplier} for all affected nodes
          new_events: list of shock events that started or ended this step
        """
        # 1. Age existing shocks — transition to recovering when duration expires
        # 2. Apply contagion — spread active shocks to neighbors
        # 3. Sample new shock events if rand < shock_prob
        # 4. Compute supply_multipliers from all active shocks
        #    (overlapping shocks compound multiplicatively)
        # 5. Return multipliers and event log
```

### PolicyInterfaceLayer
Wraps the env step to enforce rerouting constraints and nonlinear costs:
```python
class PolicyInterfaceLayer:
    def __init__(self, config: PolicyInterfaceConfig):
        ...

    def apply_reroute_constraints(self, proposed_reroutes, supplier_loads):
        """Filter reroutes that exceed supplier capacity fraction."""

    def compute_expedite_cost(self, units_requested, current_step_spend):
        """Apply convex cost model. Return actual units granted and cost."""

    def observe(self, true_state, t):
        """Apply observation delay or noise. Return policy-visible state."""
```

All four configs must be serializable to JSON for experiment logging.
Provide `to_dict()` and `from_dict()` on each config class.

---

## TASK 2: VALIDATION SUITE

Write `scripts/validate_shock_architecture.py`.

This is the core diagnostic tool. It runs a structured battery of tests to check
whether each layer is doing real work — contributing to policy separation — and
whether the architecture is calibrated correctly.

### Philosophy
Each test isolates one layer by varying only its parameters while holding others fixed.
A layer "does work" if varying it changes the differentiation metric meaningfully.
A layer that doesn't do work is either miscalibrated or redundant.

The differentiation metric used throughout:
```
separation_score = (backlog_auc(no_intervention) - backlog_auc(graph_informed))
                   / backlog_auc(no_intervention)
```
Target: separation_score ∈ [0.10, 0.35] across most regimes.
Below 0.05: policies are indistinguishable — regime is uninformative.
Above 0.40: graph_informed dominates trivially — regime is too easy.

### Test battery

Run each test with: 5 seeds, T=80, all 6 policies (no MIP for speed).

---

**Test T1 — Topology baseline**
Purpose: establish what separation looks like with current flat topology.
Config: current defaults, i.i.d. shocks, no capacity constraints, linear expedite cost.
Expected: separation_score ≈ 0.15–0.20 with budget, ≈ 0.02–0.05 without.
This is your baseline — all other tests are measured relative to it.

---

**Test T2 — Regional clustering**
Purpose: does adding regional structure to shocks make rerouting informative?
Vary: TopologyConfig.num_regions ∈ [1, 3, 5], ShockGenerationConfig.epicenter_bias = "region"
Hold: PolicyInterfaceConfig.expedite_budget = 0 (isolate rerouting value)
Expected: separation_score should rise as num_regions increases.
Pass condition: separation_score(num_regions=3, eb=0) > separation_score(num_regions=1, eb=0) + 0.05
Failure mode: if no improvement, shocks are still too symmetric — increase contagion_radius.

---

**Test T3 — Supplier capacity heterogeneity**
Purpose: do capacity constraints make the rerouting allocation decision non-trivial?
Vary: TopologyConfig.supplier_capacity_cv ∈ [0.0, 0.3, 0.6]
Hold: expedite_budget=0, regional shocks active
Expected: higher CV → more consequential reroute choices → higher separation.
Pass condition: separation_score(cv=0.6) > separation_score(cv=0.0) + 0.05
Failure mode: if flat, rerouting destinations are not binding — increase shock magnitude.

---

**Test T4 — Shock persistence**
Purpose: does longer shock duration make early intervention more valuable?
Vary: ShockDynamicsConfig.duration_mean ∈ [3, 8, 15, 25]
Hold: expedite_budget=0, regional shocks, capacity constraints active
Expected: separation_score peaks at some intermediate duration — too short = shocks
self-resolve before policy acts; too long = system collapses regardless.
Pass condition: peak separation_score > 0.10 and occurs at duration_mean ≥ 8.
Failure mode: monotone decrease = pipeline latency is too long relative to shock duration.
Fix: reduce lead_time_mean or increase T.

---

**Test T5 — Spatial contagion**
Purpose: does shock spreading make topology signals more valuable?
Vary: ShockDynamicsConfig.contagion_radius ∈ [0, 1, 2]
       ShockDynamicsConfig.contagion_prob_per_hop ∈ [0.0, 0.2, 0.5]
Hold: regional shocks, capacity constraints, expedite_budget=0
Expected: at radius=1, separation_score rises because cascade prediction becomes valuable.
At radius=2, may decrease if shocks become too widespread.
Pass condition: separation_score(radius=1) > separation_score(radius=0) + 0.03
Failure mode: if contagion makes things worse, chokepoint structure is too sparse.

---

**Test T6 — Nonlinear expedite cost**
Purpose: does convex expedite cost prevent budget from trivially dominating?
Vary: PolicyInterfaceConfig.expedite_convexity ∈ [0.0, 0.5, 1.0, 2.0]
       PolicyInterfaceConfig.expedite_capacity_cap ∈ [1.0, 0.5, 0.3]
Hold: all other layers at calibrated values, expedite_budget=50000
Expected: as convexity increases, rerouting becomes relatively more valuable.
Pass condition: at convexity=1.0, separation_score(reroute_only) rises relative to
separation_score(expedite_only). The ratio reroute_gain/expedite_gain should exceed 0.5.
Failure mode: if expedite_only still dominates at high convexity, the cap is too loose.

---

**Test T7 — Reroute setup delay**
Purpose: does adding setup delay create a meaningful early-vs-late tradeoff?
Vary: PolicyInterfaceConfig.reroute_setup_delay ∈ [0, 2, 4]
Hold: persistent regional shocks, capacity constraints, no expedite budget
Expected: with delay=0, rerouting is always safe to attempt. With delay=2+,
early rerouting under uncertainty becomes risky — graph signals about cascade
likelihood become valuable for deciding whether to commit.
Pass condition: separation_score(delay=2) > separation_score(delay=0) + 0.03
Failure mode: if delay collapses separation (all policies hurt equally), shocks
are too short for delay to matter — increase duration_mean.

---

**Test T8 — Demand surge coupling**
Purpose: does correlated demand surge close the "wait and recover" escape hatch?
Vary: ShockDynamicsConfig.demand_surge_enabled ∈ [False, True]
       ShockDynamicsConfig.demand_surge_multiplier ∈ [1.0, 1.2, 1.5]
Hold: regional shocks, persistence, no expedite budget
Expected: with surge enabled, no_intervention backlog rises significantly.
graph_informed advantage should increase because passive waiting is now costly.
Pass condition: backlog_auc(no_intervention, surge=True) >
               backlog_auc(no_intervention, surge=False) × 1.20
Failure mode: if surge has no effect, demand is not binding — check init_demand calibration.

---

**Test T9 — Full architecture integration**
Purpose: with all layers combined at calibrated values, do policies compete meaningfully
across regimes INCLUDING at expedite_budget=0?
Config: best parameters found from T1–T8
Regimes: 2×2 grid: {tight, loose} supply × {local, regional} shocks
Expected: separation_score ∈ [0.10, 0.35] for ALL 4 regimes, with AND without budget.
Pass condition: all 4 regimes pass at both budget levels.
This is the acceptance test. If it passes, the architecture is thesis-defensible.

---

### Output format

For each test, save:
```
artifacts/validation/shock_architecture/
├── T1_baseline/
│   ├── results.csv          # per (config, seed, policy): backlog_auc, fill_rate, separation_score
│   ├── summary.json         # mean separation_score per config, pass/fail verdict
│   └── plot_separation.png  # separation_score vs varied parameter, line per policy
├── T2_regional_clustering/
│   └── ...
...
├── T9_integration/
│   └── ...
└── validation_report.md
```

`validation_report.md` must contain:
- One paragraph per test: what was varied, what was expected, what was observed, pass/fail
- A calibration table: recommended parameter values derived from each test
- A final section: "Recommended architecture configuration" — the specific parameter
  set that passes T9, with justification for each choice
- A "limitations" section: what the architecture still cannot model

---

## TASK 3: CALIBRATION SEARCH

Write `scripts/calibrate_shock_architecture.py`.

After the validation suite identifies which parameters matter, this script
runs a focused calibration search to find the parameter combination that maximizes
policy differentiation across regimes while keeping the architecture interpretable.

### Calibration objective

Maximize:
```
calibration_score = mean(separation_score across regimes) 
                  - penalty × std(separation_score across regimes)
```

The penalty on std penalizes configurations where one regime is highly differentiated
but others are flat. We want consistent, interpretable separation everywhere.

`penalty = 0.5` (tunable).

### Search space (focused, not exhaustive)

Fix: T=80, seeds=5 for calibration speed. Re-run winners at seeds=20.
```python
CALIBRATION_GRID = {
    # Topology
    "num_regions":               [2, 3, 5],
    "supplier_capacity_cv":      [0.0, 0.3, 0.5],
    "chokepoint_fraction":       [0.0, 0.2],

    # Shock dynamics
    "duration_mean":             [5, 10, 15],
    "contagion_radius":          [0, 1, 2],
    "contagion_prob_per_hop":    [0.2, 0.4],
    "recovery_shape":            ["instant", "linear"],

    # Policy interface
    "reroute_setup_delay":       [0, 2],
    "reroute_capacity_fraction": [1.0, 0.5],
    "expedite_convexity":        [0.0, 1.0],
    "expedite_capacity_cap":     [1.0, 0.4],
}
```

Do NOT run a full factorial (that's 3×3×2×3×3×2×2×2×2×2×2 = 5184 configs).

Instead use a sequential search:
1. Run T2–T8 validation tests first (they already sweep the key axes individually)
2. From each test, pick the parameter value that maximized separation_score
3. Combine the winners into a candidate config
4. Run T9 with the candidate — if it passes, stop
5. If T9 fails, run a local grid search around the candidate (vary one parameter
   at a time from the winner, ±1 step) until T9 passes or search exhausts

Save all candidate configs and their T9 scores to:
`artifacts/validation/shock_architecture/calibration_search.csv`

### Acceptance criteria

The final calibrated config must satisfy all of:
- T9 passes: all 4 regimes show separation_score ∈ [0.10, 0.35] at both budget levels
- reroute_only shows separation_score > 0.05 at expedite_budget=0 in at least 2 regimes
- expedite_only shows separation_score > 0.05 at expedite_budget=50000 in at least 2 regimes
- graph_informed beats backlog_greedy by > 0.03 in at least 2 regimes
- No regime shows separation_score > 0.45 (trivial domination)

If no config passes after exhausting the search, print a diagnostic identifying
which criterion is hardest to satisfy, and suggest parameter ranges to expand.

---

## TASK 4: ARCHITECTURE COMPARISON REPORT

Write `scripts/compare_architectures.py`.

This script runs a head-to-head comparison between:
- **Arch A**: the original reworked architecture (current supplysim_env.py defaults)
- **Arch B**: the new shock architecture at calibrated values

For each architecture, run:
- 4 regimes (tight/loose × local/regional)
- Both budget levels (eb=0, eb=50000)
- All 6 policies
- 10 seeds

Produce a comparison table showing for each (arch, regime, budget):
- separation_score per policy pair
- lever_contrast (reroute_only vs expedite_only gap)
- rank_stability (Spearman across seeds)
- mean runtime per episode

Save to `artifacts/validation/shock_architecture/arch_comparison.md`.

The report must answer:
1. Does Arch B produce meaningful separation at eb=0 where Arch A failed?
2. Does Arch B preserve or improve separation at eb=50000?
3. What is the runtime cost of the new architecture?
4. Is the improvement in separation due to topology, dynamics, or interface changes?
   (Use ablation: A + topology only, A + dynamics only, A + interface only, full B)

---

## TASK 5: INTEGRATION WRAPPER

Write `scripts/supplysim_env_v2.py`.

A drop-in replacement for `supplysim_env.py` that uses the new shock architecture
while preserving the same external API:
- `reset()` → obs dict
- `step(action)` → (obs, reward, done, info)
- `get_kpi_history()` → DataFrame
- Same action format: `{"reroute": [...], "supply_multiplier": {...}}`

The wrapper injects the `ShockEngine` and `PolicyInterfaceLayer` into the existing
env step loop. It must:
- Accept an `ArchitectureConfig` dataclass that bundles all four layer configs
- Be backward compatible: `ArchitectureConfig()` with all defaults reproduces
  the current behavior exactly (verified by comparing T1 results)
- Log per-step shock state to `info["shock_state"]` for explainability

---

## DELIVERABLES

Write exactly these files:
```
scripts/shock_architecture.py          # four config classes + ShockEngine + PolicyInterfaceLayer
scripts/validate_shock_architecture.py # T1–T9 test battery
scripts/calibrate_shock_architecture.py # sequential calibration search
scripts/compare_architectures.py        # Arch A vs Arch B comparison
scripts/supplysim_env_v2.py            # drop-in env wrapper using new architecture
```

Do not modify:
- scripts/supplysim_env.py
- scripts/synthetic_data.py
- scripts/calibrated_scenario.py
- scripts/baseline_policies.py
- scripts/graph_informed_optimizer.py

---

## IMPLEMENTATION NOTES

### shock_magnitude semantics (mandatory throughout)
```
shock_supply = default_supply * (1 - shock_magnitude)
shock_magnitude=0.7 → 70% supply drop, 30% retained
```
Never use shock_fraction. All configs, logs, and plots use shock_magnitude.

### Reproducibility
Every stochastic element must use a seeded `np.random.Generator` passed at construction.
The same seed must produce identical shock sequences regardless of policy.
Shock sequences are generated at episode start and stored — policies cannot
influence shock generation.

### Progress bars
All scripts that run simulations use tqdm with per-run postfix showing:
```
[T4] duration=12  seed=3  policy=graph_informed | sep=0.18
```

### Config serialization
Every ArchitectureConfig must round-trip through JSON without loss.
Save the full config alongside every result file.

### No silent failures
If a validation test cannot be run (missing dependency, import error),
print a clear diagnostic and skip that test — do not silently pass it.

### Separation score edge cases
If backlog_auc(no_intervention) < 100, the regime has negligible disruption.
Flag it as "trivially mild" and exclude from separation_score computation.
Report how many regimes were flagged.

---

## EXECUTION ORDER

The researcher will run these manually in order:
```bash
# 1. Run validation battery (tests T1–T9)
python scripts/validate_shock_architecture.py

# 2. If T9 fails, run calibration search
python scripts/calibrate_shock_architecture.py

# 3. Compare old vs new architecture
python scripts/compare_architectures.py

# 4. Review artifacts/validation/shock_architecture/validation_report.md
#    and arch_comparison.md before proceeding to full experiments
```

When all scripts are written, print:
- List of files created
- Which validation tests are expected to pass immediately vs require calibration
- One paragraph: what a passing T9 means for the thesis — why it makes the
  policy comparison results defensible to a committee