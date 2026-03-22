  ---
  How to run the experiments

  Step 0 — Preflight check

  ./venv/bin/python scripts/preflight_check.py
  This verifies shock semantics, reroute transfer, KPI warm-start, order expiry, seed reproducibility, and MIP availability. Fix any [FAIL] items before
  proceeding. If MIP is unavailable, add --no-mip to all subsequent commands.

  Step 1 — Panel 1 (Core Benchmark) — ~2–4h at 4 workers

  ./venv/bin/python scripts/run_panel1.py \
      --seeds 20 --workers 4 \
      --output-dir artifacts/experiments/rework_benchmark_$(date +%Y%m%d_%H%M)
  Note the output directory printed — use it for all subsequent panels. Panel 1 exits with code 2 if fewer than 4 regimes pass the differentiation check. Review
  panel1/differentiation_report.txt if that happens.

  Step 2 — Panel 2 (Mechanism)

  ./venv/bin/python scripts/run_panel2.py \
      --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM \
      --seeds 20 --workers 4

  Step 3 — Panel 3 (Robustness)

  ./venv/bin/python scripts/run_panel3.py \
      --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM \
      --seeds 10 --workers 4

  Step 4 — Panel 4 (Budget Frontier) — optional

  ./venv/bin/python scripts/run_panel4.py \
      --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM \
      --seeds 10 --workers 4

  Step 5 — Plots

  ./venv/bin/python scripts/analyze_and_plot.py \
      --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM

  Step 6 — Report

  ./venv/bin/python scripts/generate_report.py \
      --output-dir artifacts/experiments/rework_benchmark_YYYYMMDD_HHMM

  ---
  Common flags (all panel scripts)

  ┌─────────────────────┬────────────────────────────────────────────────────────────────┐
  │        Flag         │                            Purpose                             │
  ├─────────────────────┼────────────────────────────────────────────────────────────────┤
  │ --dry-run           │ Print experiment count + time estimate, don't run              │
  ├─────────────────────┼────────────────────────────────────────────────────────────────┤
  │ --resume            │ Skip already-completed runs (safe to re-run after crash)       │
  ├─────────────────────┼────────────────────────────────────────────────────────────────┤
  │ --overwrite         │ Delete existing results and rerun from scratch                 │
  ├─────────────────────┼────────────────────────────────────────────────────────────────┤
  │ --no-mip            │ Skip MIP policy (use if pulp not installed or too slow)        │
  ├─────────────────────┼────────────────────────────────────────────────────────────────┤
  │ --workers N         │ Parallel workers (1 = no multiprocessing, safer for debugging) │
  ├─────────────────────┼────────────────────────────────────────────────────────────────┤
  │ --skip-policies X Y │ Skip named policies (e.g. --skip-policies mip threshold)       │
  └─────────────────────┴────────────────────────────────────────────────────────────────┘

  Files created

  ┌─────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │            File             │                               Description                               │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/experiment_utils.py │ Shared core: run_single_experiment, build_policy, compute_delta_metrics │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/preflight_check.py  │ 9-check preflight verification                                          │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/run_panel1.py       │ Core benchmark (12 regimes × 7 policies × 20 seeds)                     │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/run_panel2.py       │ Mechanism sweep (shock_magnitude × recovery_rate)                       │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/run_panel3.py       │ Robustness sweep (shock_prob)                                           │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/run_panel4.py       │ Budget frontier (expedite_budget)                                       │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/analyze_and_plot.py │ 6 thesis-ready plots                                                    │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ scripts/generate_report.py  │ report.md with all 14 sections                                          │
  └─────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘