# Claude Code: Definitive Calibration Fix for SupplySim

## CONTEXT: Why We're Here

The simulation produces 4-8% improvement from graph-informed optimizer, but this improvement is **structural routing optimization, not disruption response**. The proof: no-intervention backlog AUC is IDENTICAL across all shock_prob values (108,456 for every shock_prob from 0.05 to 0.20). Shocks don't stress the system because supply far exceeds what the production pipeline can actually consume.

Previous calibration attempts failed because they guessed supply levels. This time we calibrate from first principles: **measure what the pipeline actually consumes, then set supply relative to that.**

## PREREQUISITE: Read the Codebase

Before any code changes:

```bash
cat CLAUDE.md
cat scripts/calibrated_scenario.py
cat scripts/supplysim_env.py
# Understand how exog_schedule is built and consumed:
grep -n "exog_schedule\|exog_supply\|supply_level\|consume\|fulfill\|produce" TGB/modules/synthetic_data.py | head -60
grep -n "exog_schedule\|supply_mult\|exog" scripts/supplysim_env.py | head -40
```

Understand:
- How does the simulator consume exogenous supply each timestep?
- Is consumption = min(demand, available_supply)? Or is it drawn from a distribution?
- Where in the step() function does exogenous supply get used?
- What is the actual flow: exog_supply → firm inventory → production → orders → consumer?

Update CLAUDE.md with your findings before proceeding.

---

## PHASE 1: Measure the Pipeline (< 30 seconds)

Create and run `scripts/measure_pipeline.py`:

```python
"""
GOAL: Determine the actual exogenous supply consumption rate of the pipeline.
This is the calibration anchor — everything else is set relative to this number.

Run ONE episode with UNLIMITED supply (default_supply=10,000,000) and NO shocks.
At each timestep, measure:
1. How much exogenous supply was AVAILABLE per (firm, product)
2. How much was actually CONSUMED (drawn into inventory/production)
3. The difference (= wasted oversupply)

The steady-state consumption rate (after warmup) is our calibration target.
"""
import numpy as np

def measure_pipeline():
    # Create env with massive supply, no shocks
    env, obs, _ = create_calibrated_env(
        seed=42,
        default_supply=10_000_000,  # Effectively unlimited
        shock_fraction=1.0,         # No shock effect
        shock_prob=0.0,             # No shocks
        firm_shock_fraction=1.0,
        warmup_steps=0,             # We want to see the full ramp-up
        T=60,
        # ... other standard params
    )
    
    consumption_by_step = []
    
    for t in range(60):
        # BEFORE step: record exogenous supply available
        # You need to find where this is stored — likely env.exog_schedule[t]
        # or similar. Read the env code to find exact variable.
        supply_before = {}  # {(firm, product): available_units}
        
        # Step with no intervention
        obs, reward, done, info = env.step({"reroute": [], "supply_multiplier": {}})
        
        # AFTER step: measure what was consumed
        # This might be: supply_before - remaining_supply
        # Or it might be logged in transactions/info
        # Or you might need to compute: inventory_increase + units_shipped_out
        
        # Record total exogenous consumption this step
        total_consumed = ...  # Sum across all exogenous (firm, product) pairs
        total_available = ... # Sum of what was in exog_schedule[t]
        
        consumption_by_step.append({
            't': t,
            'total_available': total_available,
            'total_consumed': total_consumed,
            'utilization': total_consumed / total_available if total_available > 0 else 0,
        })
        
        if done:
            break
    
    # Print results
    print("\n=== PIPELINE CONSUMPTION PROFILE ===")
    print(f"{'Step':>4} {'Available':>12} {'Consumed':>12} {'Utilization':>10}")
    print("-" * 42)
    for row in consumption_by_step:
        print(f"{row['t']:>4} {row['total_available']:>12,.0f} {row['total_consumed']:>12,.0f} {row['utilization']:>10.4f}")
    
    # Steady-state consumption (average over steps 20-50, after ramp-up)
    steady_state = consumption_by_step[20:50]
    avg_consumption = np.mean([r['total_consumed'] for r in steady_state])
    max_consumption = np.max([r['total_consumed'] for r in steady_state])
    
    print(f"\n=== CALIBRATION ANCHOR ===")
    print(f"Steady-state avg consumption: {avg_consumption:,.0f} units/step")
    print(f"Steady-state max consumption: {max_consumption:,.0f} units/step")
    print(f"Number of exogenous (firm, product) pairs: {len(supply_before)}")
    print(f"Avg consumption per (firm, product): {avg_consumption / max(len(supply_before), 1):,.0f}")
    
    # CALIBRATION RECOMMENDATIONS
    # default_supply should be per (firm, product), so divide by num pairs
    num_exog_pairs = len(supply_before)
    per_pair_consumption = avg_consumption / max(num_exog_pairs, 1)
    
    print(f"\n=== RECOMMENDED SUPPLY LEVELS ===")
    for overcapacity in [1.1, 1.2, 1.3, 1.5, 2.0]:
        ds = int(per_pair_consumption * overcapacity)
        print(f"  {overcapacity:.0%} overcapacity: default_supply = {ds:,}")
        for sf in [0.3, 0.5, 0.7]:
            shocked = int(ds * sf)
            ratio = shocked / per_pair_consumption
            status = "SHORTAGE" if ratio < 0.95 else "OK" if ratio < 1.05 else "SURPLUS"
            print(f"    shock_fraction={sf}: shocked_supply={shocked:,} ({ratio:.1%} of need) [{status}]")

if __name__ == "__main__":
    measure_pipeline()
```

**CRITICAL: The exact implementation depends on how the env stores and consumes supply.** You MUST read `supplysim_env.py` and `TGB/modules/synthetic_data.py` to find:
- Where exogenous supply is stored (likely `env.exog_schedule`)
- How it's consumed during `step()` (look for inventory updates, production logic)
- Whether consumption is deterministic or stochastic

If you cannot directly measure consumption from env internals, use an INDIRECT approach:
```python
# Alternative: measure via inventory changes
# If exogenous products go from supply → firm inventory → consumed by production
# Then consumption ≈ supply_injected - inventory_increase_of_exog_products
# Or consumption ≈ total_units_in_transactions for exogenous products
```

**STOP AFTER THIS PHASE.** Print results. Do not proceed until you have a consumption number you trust.

---

## PHASE 2: Validate the Consumption Measurement (< 30 seconds)

Before using the consumption number for calibration, validate it makes sense:

```python
"""
Sanity checks on the consumption measurement:
1. Consumption should be roughly stable after warmup (not growing unboundedly)
2. Consumption should be << available supply (since we gave unlimited supply)
3. Consumption should be > 0 (pipeline is actually running)
4. Consumption × T should be roughly proportional to total units in transactions
"""

# Check 1: Stability
std_consumption = np.std([r['total_consumed'] for r in steady_state])
cv = std_consumption / avg_consumption
print(f"Consumption coefficient of variation: {cv:.2%}")
if cv > 0.5:
    print("WARNING: Consumption is highly variable — steady state may not exist")
    print("Try increasing warmup or T")

# Check 2: Utilization << 1 (confirming supply was truly unlimited)
avg_util = np.mean([r['utilization'] for r in steady_state])
print(f"Avg utilization: {avg_util:.4%}")
if avg_util > 0.01:
    print("WARNING: Utilization is too high — supply may not have been effectively unlimited")
    print("Increase default_supply and re-run")

# Check 3: Non-zero
if avg_consumption < 1:
    print("ERROR: Pipeline is not consuming any exogenous supply")
    print("Check if production logic is actually running")

# Check 4: Cross-reference with transactions
total_tx_units = info.get('total_transaction_units', 'N/A')  # or compute from logs
print(f"Total transaction units in episode: {total_tx_units}")
print(f"Total consumption over episode: {sum(r['total_consumed'] for r in consumption_by_step):,.0f}")
```

**IF CONSUMPTION MEASUREMENT FAILS** (can't measure it directly), use this fallback:

```python
"""
FALLBACK: Binary search for the supply level where fill rate transitions.
If we can't measure consumption directly, we find it indirectly:
the supply level where fill rate drops from ~1.0 to < 0.9 IS the consumption rate.
"""
# Run no-shock episodes with decreasing supply until fill rate drops
for ds in [10_000_000, 1_000_000, 500_000, 200_000, 100_000, 50_000, 20_000, 10_000, 5_000, 2_000, 1_000, 500]:
    result = run_episode(default_supply=ds, shock_prob=0.0, T=60, seed=42)
    fill = result['mean_fill_rate']  # or however fill rate is computed
    auc = result['backlog_auc']
    print(f"default_supply={ds:>12,}  fill_rate={fill:.4f}  backlog_auc={auc:,.0f}")

# The supply level where fill_rate first drops below 0.95 ≈ steady-state consumption per pair
# Fine-tune with binary search around that level
```

---

## PHASE 3: Calibrate and Validate (< 2 minutes)

Using the consumption measurement, set supply levels and verify the system behaves correctly:

```python
"""
Set default_supply to ~1.2x steady-state consumption (slight overcapacity).
Verify:
1. No-shock fill rate is 0.85-0.95 (near capacity, not stressed)
2. Shock at fraction=0.3 drops fill rate significantly (to 0.4-0.7)
3. No-intervention backlog AUC INCREASES as shock_prob increases (shocks matter!)
4. No-intervention backlog AUC INCREASES as shock_fraction decreases (severity matters!)
"""

# Use calibrated supply level from Phase 1
CALIBRATED_SUPPLY = ...  # from Phase 1 recommendations, use 1.2x overcapacity

# Test 1: No-shock fill rate
result_noshock = run_episode(
    default_supply=CALIBRATED_SUPPLY, shock_prob=0.0, 
    warmup_steps=15, T=60, seed=42
)
print(f"No-shock fill rate: {result_noshock['mean_fill_rate']:.3f}")
# TARGET: 0.85-0.95

# Test 2: Shocked fill rate at different severities
for sf in [0.3, 0.5, 0.7]:
    result = run_episode(
        default_supply=CALIBRATED_SUPPLY, shock_prob=0.15,
        shock_fraction=sf, firm_shock_fraction=0.5,
        warmup_steps=15, T=60, seed=42
    )
    print(f"shock_fraction={sf}: fill_rate={result['mean_fill_rate']:.3f}, backlog_auc={result['backlog_auc']:,.0f}")
# TARGET: fill rate drops with lower shock_fraction

# Test 3: THE CRITICAL TEST — does backlog change with shock_prob?
for sp in [0.0, 0.05, 0.10, 0.15, 0.20]:
    result = run_episode(
        default_supply=CALIBRATED_SUPPLY, shock_prob=sp,
        shock_fraction=0.3, firm_shock_fraction=0.5,
        warmup_steps=15, T=60, seed=42
    )
    print(f"shock_prob={sp}: backlog_auc={result['backlog_auc']:,.0f}, fill_rate={result['mean_fill_rate']:.3f}")
# TARGET: backlog_auc should INCREASE with shock_prob. If it's flat, calibration failed.

# Test 4: Does shock severity matter?
for sf in [0.1, 0.3, 0.5, 0.7, 0.9]:
    result = run_episode(
        default_supply=CALIBRATED_SUPPLY, shock_prob=0.15,
        shock_fraction=sf, firm_shock_fraction=0.5,
        warmup_steps=15, T=60, seed=42
    )
    print(f"shock_fraction={sf}: backlog_auc={result['backlog_auc']:,.0f}")
# TARGET: backlog_auc should DECREASE with higher shock_fraction (milder shocks)
```

**PRINT ALL RESULTS AS A CLEAR TABLE. STOP AND EVALUATE BEFORE PROCEEDING.**

---

## DECISION GATE: Did Calibration Work?

After Phase 3, evaluate results against these criteria:

### Gate A: SUCCESS — Proceed to Phase 4
All of these must be true:
- [ ] No-shock fill rate is 0.80-0.95
- [ ] Backlog AUC increases monotonically with shock_prob (at least 20% increase from sp=0.0 to sp=0.20)
- [ ] Backlog AUC increases monotonically with shock severity (lower shock_fraction = higher backlog)
- [ ] Shocked fill rate at shock_fraction=0.3 is noticeably lower than no-shock fill rate

If all pass → proceed to Phase 4 (pilot with policies).

### Gate B: PARTIAL — Fill rate OK but shocks don't matter
If no-shock fill rate is in range BUT backlog doesn't change with shock_prob:

**Diagnosis:** Supply is still too high even at 1.2x. The pipeline adapts by building inventory buffers during no-shock periods that absorb the shocks.

**Fix:** Reduce overcapacity factor. Try:
```python
for overcapacity in [1.05, 1.10, 1.15, 1.20]:
    ds = int(per_pair_consumption * overcapacity)
    # Re-run Tests 3 and 4
```

The system needs to be running with very thin margins so that any supply reduction creates immediate backlog. If 1.05x still doesn't show shock sensitivity, move to Escalation Path 1.

### Gate C: FAILURE — Can't find working supply level
If no supply level produces both reasonable fill rate AND shock sensitivity:

**Escalation Path 1: Reduce inventory buffering**

The pipeline may be buffering supply into inventory during good periods, absorbing shocks. Check:
```python
# Measure inventory levels over time
# If inventories grow during no-shock periods, the system is self-buffering
for t in range(60):
    obs, _, _, info = env.step(no_action)
    total_inv = np.sum(env.inventories)  # or however inventories are accessed
    print(f"t={t}: total_inventory={total_inv:,.0f}")
```

If inventories are growing, consider:
- Setting `init_inv=0` AND using a shorter warmup (so less time to build buffer)
- Or: reduce the warmup_steps to 5-8 instead of 15, so the system barely reaches steady state before shocks hit

**Escalation Path 2: Make shocks longer/harder to recover from**

If individual shock events are too brief to matter:
```python
# Reduce recovery_rate (slower recovery = longer disruptions)
for rr in [0.5, 0.3, 0.1]:  # Lower = slower recovery
    result = run_episode(
        default_supply=CALIBRATED_SUPPLY, shock_prob=0.15,
        shock_fraction=0.3, recovery_rate=rr, ...
    )
    print(f"recovery_rate={rr}: backlog_auc={result['backlog_auc']:,.0f}")
```

Slower recovery means each shock persists for more timesteps, amplifying its impact.

**Escalation Path 3: Increase demand relative to supply**

Instead of reducing supply, increase demand:
```python
# Check if demand is configurable
grep -n "demand\|init_demand\|consumer_demand\|poisson" TGB/modules/synthetic_data.py scripts/*.py
```

If demand can be scaled up (e.g., multiplying the demand schedule by 2x), this effectively halves the supply-to-demand ratio without changing supply levels.

**Escalation Path 4: Smaller network**

If the amplification factor (2000-5000x from consumer to exogenous) is the core problem, use a smaller/shallower production graph:
```python
# Fewer layers = less amplification = tighter supply-demand coupling
# Try: num_inner_layers=1 or 2 instead of default
# Try: fewer products per layer
```

This is a last resort because it changes the network structure, but a smaller network with clear shock sensitivity is better than a large network where shocks don't matter.

**Escalation Path 5: Accept structural routing story**

If after ALL escalation paths, shocks still don't create measurable backlog differences, then the simulator's architecture fundamentally prevents supply-limited operation at steady state. This is a valid finding. Reframe the thesis as graph-informed routing optimization on multi-tier supply networks, drop the disruption response framing, and use the existing 4-8% routing improvement results.

---

## PHASE 4: Policy Pilot with Calibrated Params (< 2 minutes)

Only run this if Gate A passed.

```python
"""
Quick pilot: 4 key policies × 3 seeds × 2 contrasting regimes.
Verify policy differentiation under proper calibration.
"""
REGIMES = [
    {"name": "mild", "shock_prob": 0.10, "shock_fraction": 0.5, "firm_shock_fraction": 0.5},
    {"name": "severe", "shock_prob": 0.20, "shock_fraction": 0.3, "firm_shock_fraction": 0.5},
]
POLICIES = ["no_intervention", "backlog_only_greedy", "graph_informed", "mip"]
SEEDS = [0, 1, 2]

results = []
for regime in REGIMES:
    for policy in POLICIES:
        aucs = []
        for seed in SEEDS:
            result = run_policy(policy, {**regime, "default_supply": CALIBRATED_SUPPLY, ...}, seed)
            aucs.append(result['backlog_auc'])
        mean_auc = np.mean(aucs)
        results.append({"regime": regime['name'], "policy": policy, "mean_auc": mean_auc, "std": np.std(aucs)})

# Print comparison table
print("\n=== POLICY PILOT RESULTS ===")
print(f"{'Regime':<10} {'Policy':<25} {'Mean AUC':>12} {'Std':>10}")
for r in results:
    print(f"{r['regime']:<10} {r['policy']:<25} {r['mean_auc']:>12,.0f} {r['std']:>10,.0f}")

# Key comparisons
for regime_name in ["mild", "severe"]:
    ni = [r for r in results if r['regime']==regime_name and r['policy']=='no_intervention'][0]
    gi = [r for r in results if r['regime']==regime_name and r['policy']=='graph_informed'][0]
    delta = (ni['mean_auc'] - gi['mean_auc']) / ni['mean_auc'] * 100
    print(f"\n{regime_name}: graph_informed vs no_intervention = {delta:+.1f}%")

# CRITICAL: Compare mild vs severe for no_intervention
ni_mild = [r for r in results if r['regime']=='mild' and r['policy']=='no_intervention'][0]
ni_severe = [r for r in results if r['regime']=='severe' and r['policy']=='no_intervention'][0]
regime_effect = (ni_severe['mean_auc'] - ni_mild['mean_auc']) / ni_mild['mean_auc'] * 100
print(f"\nRegime effect on baseline: severe vs mild = {regime_effect:+.1f}%")
print("(This MUST be positive and >10% — if not, shocks still don't matter)")
```

**Gate check:** 
- Regime effect on baseline > 10%? → Shocks matter. Proceed to full experiment.
- graph_informed beats no_intervention by > 10% in severe regime? → Optimizer works under stress. Proceed.
- If either fails → go back to escalation paths.

---

## PHASE 5: Update Experiment Scripts (DO NOT RUN)

If Phase 4 passes, update `run_regime_experiment.py` and `run_calibration.py` with the new calibrated `default_supply` value. Update the regime grid to use `shock_fraction` as a primary axis (since it now produces actual variation in baseline backlog):

```python
# Updated regime grid
FIRM_SHOCK_FRACTIONS = [0.3, 0.5, 0.7, 1.0]
SHOCK_FRACTIONS = [0.2, 0.4, 0.6, 0.8]  # Now these create real variation
# shock_prob can be fixed at 0.15 or swept as secondary
```

Print the user instructions for running the full experiment.

---

## PHASE 6: Print Final Calibration Report

Regardless of which gate/path was taken, print a complete calibration report:

```
============================================================
CALIBRATION REPORT
============================================================
Pipeline steady-state consumption: X units/step
Calibrated default_supply: Y per (firm, product)
Overcapacity factor: Z

No-shock fill rate: 0.XX
Shocked fill rate (sf=0.3): 0.XX
Shocked fill rate (sf=0.7): 0.XX

Baseline backlog AUC (no shocks): X,XXX
Baseline backlog AUC (sp=0.15, sf=0.3): X,XXX  (+XX%)
Baseline backlog AUC (sp=0.15, sf=0.7): X,XXX  (+XX%)

Policy differentiation (sp=0.15, sf=0.3, fsf=0.5):
  no_intervention:    X,XXX
  graph_informed:     X,XXX  (-XX% vs baseline)
  backlog_only:       X,XXX  (-XX% vs baseline)
  mip:                X,XXX  (-XX% vs baseline)

VERDICT: [SUCCESS / PARTIAL / REFRAMED]
RECOMMENDED REGIME GRID: [parameters]
ESTIMATED FULL EXPERIMENT TIME: X hours with 4 workers

Next steps:
  [specific commands to run]
============================================================
```

Update CLAUDE.md with the calibration results and recommended parameters.