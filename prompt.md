# Claude Code: Build SupplySim Disruption-Response Experiment Scripts

## CONTEXT

The simulator has already been reworked and validated. Do not modify simulator mechanics.

Validated changes already in place:
- reroute transfers pending orders to the new supplier immediately
- network depth reduced to 2 inner layers
- suppliers per product reduced to 2–3
- BOM fan-in / unit amplification reduced
- supply-demand balance reworked
- order expiry added
- KPI warm-start added

Current calibrated defaults:
```
num_inner_layers    = 2
min_num_suppliers   = 2
max_num_suppliers   = 3
min_inputs          = 1
max_inputs          = 2
min_units           = 1
max_units           = 1
init_demand         = 10
default_supply      = 100
recovery_rate       = 1.05
warmup_steps        = 10
max_order_age       = 10
kpi_start_step      = 4
shock_prob          = 0.20
```

### shock_magnitude semantics (CRITICAL)

`shock_fraction` has been renamed and its semantics inverted:
```
Old: shock_supply = default_supply * shock_fraction
     shock_fraction=0.3 → supply drops TO 30% (70% reduction)

New: shock_supply = default_supply * (1 - shock_magnitude)
     shock_magnitude=0.7 → supply drops BY 70% (30% retained)
```

Value conversions (new = 1 − old):
```
shock_fraction=0.3  →  shock_magnitude=0.7   # same 70% reduction
shock_fraction=0.5  →  shock_magnitude=0.5   # same 50% reduction
shock_fraction=1.0  →  shock_magnitude=0.0   # no-shock baseline
```

Apply this consistently across all scripts, configs, JSON outputs, plot labels, and report text.
If any existing script still uses `shock_fraction`, support it as a backward-compatible alias
internally but always output `shock_magnitude` in results and plots.

---

## EXISTING CODEBASE TO REUSE

The file `scripts/run_regime_experiment.py` already exists and contains a working pattern.
**Do not rewrite it. Inherit its structure for all new panel scripts.**

Specifically, reuse and preserve:
- `run_single_experiment(exp)` — the per-run worker function (picklable for multiprocessing)
- `build_policy(name, K)` — policy factory
- `save_result(result, output_path)` — append-to-CSV pattern (resumable)
- `main()` structure — argparse, tqdm loop, multiprocessing pool, error log
- CLI flags: `--seeds`, `--workers`, `--dry-run`, `--resume`, `--skip-policies`, `--output-dir`

Extract the shared logic into `scripts/experiment_utils.py`:
- `run_single_experiment`
- `build_policy`
- `save_result`
- `DEFAULTS` dict
- any other helpers used across panel scripts

Each panel script then imports from `experiment_utils` and only defines:
- its own config grid
- its own `build_experiment_list`
- its own post-run checks

**Do not duplicate `run_single_experiment` or `build_policy` across files.**

---

## GOAL

Write a self-contained experiment script suite that a researcher can run manually,
one script at a time, to produce thesis-ready results.

**Do not run the experiments. Write the scripts only.**

Each script must:
- be runnable standalone from the project root
- print a `tqdm` progress bar showing panel, regime, seed, and policy
- save all outputs to the correct folder on completion
- be safe to re-run (`--resume` skips completed runs, `--overwrite` forces redo)
- print a clear summary at the end: configs run, time elapsed, output location

---

## DELIVERABLES

Write exactly these files:
```
scripts/experiment_utils.py      # shared logic extracted from run_regime_experiment.py
scripts/preflight_check.py
scripts/run_panel1.py
scripts/run_panel2.py
scripts/run_panel3.py
scripts/run_panel4.py            # optional panel, still write the script
scripts/analyze_and_plot.py
scripts/generate_report.py
```

Do not modify:
- `scripts/run_regime_experiment.py`
- `scripts/supplysim_env.py`
- `scripts/synthetic_data.py`
- `scripts/calibrated_scenario.py`

You may add `random_reroute` to `scripts/baseline_policies.py` if it is not present.
Document any such addition clearly at the top of that file.

---

## PHASE 0: PREFLIGHT SCRIPT

Write `scripts/preflight_check.py`.

Checks to perform (in order):

1. Import all modules: `supplysim_env`, `calibrated_scenario`, `baseline_policies`,
   `graph_informed_optimizer`. Report PASS/FAIL per import.

2. Verify `shock_magnitude` semantics:
   - Instantiate env with `shock_magnitude=0.7`
   - Confirm shocked supply = `default_supply * (1 - 0.7)` = 30% of baseline
   - FAIL loudly if not

3. Verify `shock_magnitude=0.0` produces a no-shock baseline (supply unchanged).

4. Verify reroute transfer: confirm that rerouting moves pending orders
   from old to new supplier.

5. Verify KPI warm-start: confirm metrics only accumulate after `kpi_start_step`.

6. Verify order expiry: confirm orders older than `max_order_age` are dropped.

7. Verify matched scenarios: confirm same seed produces identical graphs,
   shocks, and demand schedules across policy runs.

8. Verify MIP: check it is importable and solves a trivial 1-step instance.
   Record solve time. If unavailable, set `MIP_AVAILABLE=False` and continue —
   all panel scripts must skip MIP gracefully when this flag is False.

9. Smoke test: 2 seeds × 2 regimes × all 7 policies.
   Confirm all required output fields are present, including delta metrics.

10. Print final PASS/FAIL summary. Exit with code 1 if any check fails.

---

## POLICIES

Run this comparator set in all panels:

1. `no_intervention`
2. `random_reroute` — uniform random selection from feasible reroute candidates.
   If candidates are empty, do nothing. Add to `baseline_policies.py` if absent.
3. `reroute_only`
4. `expedite_only`
5. `backlog_greedy`
6. `graph_informed`
7. `mip` — run on the full main grid. See MIP notes below.

MIP notes:
- If mean episode runtime > 10× mean `graph_informed` runtime on the smoke test,
  emit a prominent warning but continue.
- If MIP solve fails on any step, fall back to `no_intervention` for that step,
  record the failure count in `mip_fallback_steps`, and continue rather than crashing.
- If `MIP_AVAILABLE=False`, skip MIP in all panels and note it in `manifest.json`
  and `report.md`.
- All panel scripts accept `--no-mip` to skip MIP regardless of availability.

---

## METRICS

Record for every (regime, seed, policy) triple:
```
backlog_auc
fill_rate
peak_backlog
final_backlog
lost_sales_units
time_to_recovery     # steps until backlog <= baseline level; episode length if never
runtime_s            # wall time for that episode
mip_fallback_steps   # integer; 0 for non-MIP policies
seed
policy
regime_id
panel_name
git_commit           # from `git rev-parse --short HEAD`, else "unknown"
```

Plus all regime parameters using `shock_magnitude`, never `shock_fraction`.

### Delta metrics (MANDATORY)

For every shocked regime, also run a matched no-shock baseline
(`shock_prob=0`, `shock_magnitude=0.0`) using the same seed.

Compute and save:
```python
# Structural baseline (no-shock run)
backlog_auc_base
fill_rate_base
peak_backlog_base
lost_sales_base

# Disruption damage
delta_backlog_auc   = backlog_auc_shocked  - backlog_auc_base
delta_fill_rate     = fill_rate_shocked    - fill_rate_base
delta_peak_backlog  = peak_backlog_shocked - peak_backlog_base
delta_lost_sales    = lost_sales_shocked   - lost_sales_base

# Policy gain within shocked regime
policy_gain_total   = backlog_auc(no_intervention) - backlog_auc(policy)
policy_gain_pct     = policy_gain_total / backlog_auc(no_intervention)

# Policy gain on disruption damage specifically
policy_gain_on_damage = (
    delta_backlog_auc(no_intervention) - delta_backlog_auc(policy)
) / max(delta_backlog_auc(no_intervention), 1e-6)
```

Flag any regime where `delta_backlog_auc(no_intervention) < 500` as
`disruption_too_mild = True`. Exclude those regimes from `policy_gain_on_damage`
reporting and count how many were flagged.

For MIP specifically, compute and save:
```
mip_vs_graph_informed_gap_pct = (
    backlog_auc(mip) - backlog_auc(graph_informed)
) / backlog_auc(no_intervention)
```

---

## OUTPUT STRUCTURE
```
artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM/
├── manifest.json
├── config_grid.json
├── panel1/
│   ├── per_run_results.csv
│   ├── aggregated_results.csv
│   ├── differentiation_report.txt
│   ├── plots/
│   └── tables/
├── panel2/
│   ├── per_run_results.csv
│   ├── aggregated_results.csv
│   ├── interpretation.md
│   └── plots/
├── panel3/
│   ├── per_run_results.csv
│   ├── aggregated_results.csv
│   └── plots/
├── panel4/
│   ├── per_run_results.csv
│   ├── aggregated_results.csv
│   └── plots/
└── report.md
```

Every row in `per_run_results.csv` must contain all regime parameters,
all metrics, all delta metrics, and `git_commit`.

All panel scripts accept `--output-dir` to override the default timestamped folder.

---

## PANEL 1 — CORE BENCHMARK

**Purpose:** headline thesis result — does graph_informed beat baselines,
and under what structural conditions?

Write `scripts/run_panel1.py`.

### Grid
```
default_supply:       [50, 100, 150]
firm_shock_fraction:  [0.3, 1.0]
expedite_budget:      [0, 50000]
```

Fixed:
```
shock_prob       = 0.15
shock_magnitude  = 0.70
recovery_rate    = 1.05
```

3 × 2 × 2 = 12 shocked regimes.
For each, run a matched no-shock baseline (`shock_prob=0`, `shock_magnitude=0.0`).

### Seeds
20 per regime.

### Progress bar format
```
[Panel 1] regime=tight_local_nobudget  seed=7  policy=graph_informed
```
For MIP runs add step solve time:
```
[Panel 1] regime=tight_systemic_budget  seed=3  policy=mip  | step_solve=0.43s
```

### Post-run differentiation check

After aggregation, evaluate each regime automatically. A regime passes if:
- `delta_backlog_auc(no_intervention)` meaningfully > 0 (not disruption_too_mild)
- `graph_informed` beats `no_intervention` by ≥ 10%
- `reroute_only` and `expedite_only` differ materially from each other
- policy rankings stable across seeds (Spearman rank correlation ≥ 0.8)

Print per-regime results to stdout and save to `panel1/differentiation_report.txt`.

If fewer than 4 regimes pass: print a diagnostic, save it, and exit with code 2.
Do not proceed to further panels until the researcher reviews.

MIP stop condition: if mean MIP runtime > 10× mean `graph_informed` runtime,
print a prominent warning suggesting `--no-mip` for remaining panels.

### Required analysis outputs

For each regime, compute and save:
- policy ranking by mean `backlog_auc`
- `graph_informed` vs `no_intervention`: mean ± std improvement
- `graph_informed` vs `backlog_greedy`: mean ± std improvement
- `mip` vs `graph_informed` gap (`mip_vs_graph_informed_gap_pct`)
- `reroute_only` vs `expedite_only` contrast
- fraction of disruption-induced damage removed by each policy

Label four named regimes in all outputs:
```
tight_local_nobudget:    default_supply=50,  firm_shock_fraction=0.3, expedite_budget=0
tight_systemic_budget:   default_supply=50,  firm_shock_fraction=1.0, expedite_budget=50000
loose_local_nobudget:    default_supply=150, firm_shock_fraction=0.3, expedite_budget=0
loose_systemic_budget:   default_supply=150, firm_shock_fraction=1.0, expedite_budget=50000
```

---

## PANEL 2 — MECHANISM

**Purpose:** understand *why* policies win — shock severity and recovery speed.

Write `scripts/run_panel2.py`.

### Representative regimes

Accept `--regime-a` and `--regime-b` CLI arguments (regime_ids from Panel 1 output).

Defaults if not provided:
```
Regime A (reroute-friendly):
  default_supply=100, firm_shock_fraction=0.3, expedite_budget=0

Regime B (systemic + joint-control):
  default_supply=100, firm_shock_fraction=1.0, expedite_budget=50000
```

### Grid
```
shock_magnitude:  [0.50, 0.70, 0.85]
recovery_rate:    [1.02, 1.05, 1.25]
```

Fixed: `shock_prob=0.15`

2 regimes × 3 × 3 = 18 configs.

### Seeds
20 (reduce to 10 only if total runtime exceeds 2 hours).

### Progress bar format
```
[Panel 2] regime=A  shock_mag=0.70  rr=1.05  seed=4  policy=graph_informed
```

### Required analysis outputs

- `shock_magnitude` vs `policy_gain_pct` (one line per policy, per regime)
- `recovery_rate` vs `policy_gain_pct` (one line per policy, per regime)
- `mip` vs `graph_informed` gap across the sweep
- `panel2/interpretation.md`: one paragraph per plot answering the mechanism question

---

## PANEL 3 — ROBUSTNESS

**Purpose:** do rankings survive changes in disruption frequency?

Write `scripts/run_panel3.py`.

### Uses same two representative regimes as Panel 2.

### Grid
```
shock_prob: [0.05, 0.15, 0.25]
```

Fixed:
```
shock_magnitude  = 0.70
recovery_rate    = 1.05
```

### Seeds
10 minimum; increase to 20 if rankings are unstable across seeds.

### Progress bar format
```
[Panel 3] regime=A  shock_prob=0.15  seed=2  policy=reroute_only
```

### Required analysis outputs

- Policy ranking stability across `shock_prob` levels
- Does `graph_informed` retain its advantage over `backlog_greedy`?
- Does MIP maintain its advantage at higher disruption frequency?
- Flag any rank changes and note them in the output

---

## PANEL 4 — BUDGET FRONTIER

**Purpose:** diminishing returns from expedite budget.

Write `scripts/run_panel4.py`.

### Uses same two representative regimes as Panels 2 and 3.

### Grid
```
expedite_budget: [0, 1000, 5000, 20000, 50000]
```

Fixed:
```
shock_prob       = 0.15
shock_magnitude  = 0.70
recovery_rate    = 1.05
```

### Seeds
10.

### Progress bar format
```
[Panel 4] regime=B  budget=5000  seed=1  policy=mip  | step_solve=0.61s
```

### Key MIP analysis for this panel

Compute and plot `policy_gain_pct` vs `expedite_budget` for MIP and `graph_informed`.
Add a secondary axis showing `mip_gain / graph_informed_gain` ratio.
This answers: does MIP extract more value per unit of budget?

---

## ANALYSIS AND PLOTTING SCRIPT

Write `scripts/analyze_and_plot.py`.

Reads all panel outputs. Skips missing panels gracefully.
Runnable after any subset of panels.

### Plots

**Plot 1 — Heatmap (Panel 1)**
- `graph_informed` improvement vs `no_intervention`
- axes: `default_supply` × `firm_shock_fraction`
- separate heatmaps for `expedite_budget=0` and `expedite_budget=50000`
- second set: `mip` improvement and `mip vs graph_informed` gap

**Plot 2 — Bar chart (Panel 1, 4 named regimes)**
- all 7 policies, error bars = 1 std across seeds
- secondary panel: `policy_gain_pct`

**Plot 3 — Delta damage chart (Panel 1)**
- disruption-induced damage removed by each policy
- grouped bars for 4 named regimes
- MIP bar as visible upper bound

**Plot 4 — Mechanism plots (Panel 2)**
- `shock_magnitude` vs `policy_gain_pct`, one line per policy, per regime
- `recovery_rate` vs `policy_gain_pct`, one line per policy, per regime
- annotate `mip vs graph_informed` gap at each point

**Plot 5 — Robustness plot (Panel 3)**
- `shock_prob` vs `policy_gain_pct` per regime
- highlight any rank changes

**Plot 6 — Budget frontier (Panel 4)**
- `expedite_budget` vs `policy_gain_pct`
- secondary axis: `mip_gain / graph_informed_gain` ratio

### Tables

Save to `panel1/tables/`:
```
table_A_regime_definitions.csv
table_B_structural_baselines.csv
table_C_shocked_outcomes.csv
table_D_disruption_damage_deltas.csv
table_E_policy_rankings.csv
table_F_pairwise_comparisons.csv   # mip vs graph_informed is a mandatory pair
```

All headers use `shock_magnitude`, never `shock_fraction`.

---

## REPORT SCRIPT

Write `scripts/generate_report.py`.

Produces `report.md` with these sections:

1. What was run (panels, policies, seeds, git commit)
2. MIP availability and performance note
3. Preflight fixes (any fixes applied before running — separate from results)
4. Final regime grid
5. Structural baseline summary
6. Disruption damage summary
7. Core policy comparison results — include `mip vs graph_informed` gap
8. Mechanism findings
9. Robustness findings
10. Budget frontier findings (if Panel 4 was run)
11. Key thesis-friendly takeaways
12. Limitations
13. Recommended regime family for headline thesis results
14. one-page executive summary with the 3-4 key findings and the single best figure for the thesis.

### Mandatory takeaways — the report must explicitly answer:

- When do policies differentiate most?
- When is rerouting the dominant lever?
- When is expediting the dominant lever?
- When does `graph_informed` provide extra value beyond `backlog_greedy`?
- How close does `graph_informed` come to MIP optimality, and in which regimes
  does the gap widen?
- Which regime family should be used for the headline thesis results?

---

## STATISTICAL REQUIREMENTS

- Use the same seeds across all policies within the same regime
- Report means and standard deviations
- Report 95% confidence intervals for main comparisons (bootstrap or t-interval)
- Use paired differences across seeds for policy comparisons
- If two policies differ by < 3% and CIs overlap, describe them as comparable —
  do not overclaim
- If MIP and `graph_informed` are within 3% on a regime, state that the greedy
  approach closes the optimality gap in that regime

---

## PROGRESS BAR REQUIREMENTS

Every panel script must use `tqdm`. Requirements:

- One bar over the full run list for that panel
- `set_description` updates with panel name, regime id, seed, policy
- `set_postfix` updates with live metrics after each completed run:
```
  auc=12453  fill=0.87  elapsed=2.3s
```
- For MIP runs, also show `step_solve=Xs` in postfix
- Bar format:
```
  [Panel N] |████████░░░░░░░░| 312/960 [04:12<08:33, 1.24run/s]
```

---

## IMPLEMENTATION NOTES

- Reuse `run_regime_experiment.py` structure — do not rewrite from scratch
- Extract shared logic to `experiment_utils.py` — no duplication across panel scripts
- Do not modify simulator mechanics
- Any fix discovered during preflight must be documented in `report.md`
  under "Preflight fixes" — separate from experiment results
- All file I/O must be robust: create output dirs if missing, handle partial runs
- Each panel script accepts `--output-dir`, `--no-mip`, `--resume`, `--overwrite`
- Do not expand into a full factorial — keep design focused and interpretable