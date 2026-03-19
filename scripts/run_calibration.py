"""
STANDALONE SCRIPT — run manually in terminal.
Expected runtime: 10-15 minutes.

Full calibration across structural configs and supply/shock parameters.
Finds the operating point for each config where the system has meaningful
differentiation potential.

Usage:
    python scripts/run_calibration.py --output artifacts/experiments/calibration_table.json
    python scripts/run_calibration.py --output artifacts/experiments/calibration_table.json --seeds 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm

from scripts.calibrated_scenario import create_calibrated_env
from scripts.baseline_policies import no_intervention_policy
from scripts.graph_informed_optimizer import make_graph_informed_policy


# --- Default grid ---
FIRM_SHOCK_FRACTIONS = [0.3, 0.5, 0.7, 1.0]
SHOCK_FRACTIONS = [0.2, 0.4, 0.6, 0.8]  # Primary severity axis
SHOCK_PROBS = [0.05, 0.1, 0.15, 0.2]
DEFAULT_SUPPLY = 500_000  # Calibrated: ~6x per-pair consumption (84K), transition zone
SHOCK_FRACTION = 0.3
RECOVERY_RATE = 1.05     # Calibrated: slower recovery amplifies shock persistence
WARMUP_STEPS = 10
INIT_INV = 0
T = 60
EXPEDITE_BUDGET = 50_000
EXPEDITE_M_MAX = 3.0
K = 3


def run_calibration_point(seed, firm_shock_fraction, shock_prob):
    """Run no_intervention and graph_informed for one (fsf, sp, seed) combo."""
    common = dict(
        seed=seed, T=T, default_supply=DEFAULT_SUPPLY,
        shock_fraction=SHOCK_FRACTION, shock_prob=shock_prob,
        recovery_rate=RECOVERY_RATE, firm_shock_fraction=firm_shock_fraction,
        warmup_steps=WARMUP_STEPS, init_inv=INIT_INV,
        expedite_budget=EXPEDITE_BUDGET, expedite_m_max=EXPEDITE_M_MAX,
    )

    # No intervention
    env, obs, shock_log = create_calibrated_env(**common)
    done = False
    while not done:
        action, _ = no_intervention_policy(obs, env.t, env)
        obs, _, done, _ = env.step(action)
    kpi_ni = env.get_kpi_history()
    auc_ni = float(kpi_ni["consumer_backlog_units"].sum())
    fill_ni = float(kpi_ni["consumer_cumulative_fill_rate"].iloc[-1])

    # Graph-informed
    env, obs, _ = create_calibrated_env(**common)
    gi = make_graph_informed_policy(reroute_budget_K=K)
    done = False
    while not done:
        action, _ = gi(obs, env.t, env)
        obs, _, done, _ = env.step(action)
    kpi_gi = env.get_kpi_history()
    auc_gi = float(kpi_gi["consumer_backlog_units"].sum())
    fill_gi = float(kpi_gi["consumer_cumulative_fill_rate"].iloc[-1])

    return {
        "seed": seed,
        "firm_shock_fraction": firm_shock_fraction,
        "shock_prob": shock_prob,
        "auc_ni": auc_ni,
        "auc_gi": auc_gi,
        "fill_ni": fill_ni,
        "fill_gi": fill_gi,
        "n_shocks": len(shock_log),
        "delta_pct": (auc_ni - auc_gi) / max(auc_ni, 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibration sweep")
    parser.add_argument("--output", type=str, default="artifacts/experiments/calibration_table.json")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build experiment list
    experiments = []
    for fsf in FIRM_SHOCK_FRACTIONS:
        for sp in SHOCK_PROBS:
            for seed in range(args.seeds):
                experiments.append({"fsf": fsf, "sp": sp, "seed": seed})

    print(f"Calibration sweep: {len(experiments)} runs")
    print(f"  {len(FIRM_SHOCK_FRACTIONS)} firm_shock_fractions x {len(SHOCK_PROBS)} shock_probs x {args.seeds} seeds")

    results = []
    errors = []

    with tqdm(total=len(experiments), desc="Calibrating",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for exp in experiments:
            try:
                r = run_calibration_point(exp["seed"], exp["fsf"], exp["sp"])
                results.append(r)
                pbar.set_postfix({
                    "fsf": exp["fsf"],
                    "sp": exp["sp"],
                    "delta": f"{r['delta_pct']:+.1f}%",
                })
            except Exception as e:
                errors.append({"exp": exp, "error": str(e)})
                pbar.set_postfix({"status": "ERROR"})
            pbar.update(1)

    # Aggregate by (fsf, sp)
    print(f"\n{'=' * 80}")
    print("CALIBRATION TABLE")
    print(f"{'=' * 80}")
    print(f"{'fsf':>5s}  {'sp':>5s}  {'mean_auc_ni':>12s}  {'mean_auc_gi':>12s}  {'mean_delta%':>12s}  {'mean_fill_ni':>12s}")
    print("-" * 70)

    table = {}
    for fsf in FIRM_SHOCK_FRACTIONS:
        for sp in SHOCK_PROBS:
            matching = [r for r in results if r["firm_shock_fraction"] == fsf and r["shock_prob"] == sp]
            if not matching:
                continue
            mean_auc_ni = np.mean([r["auc_ni"] for r in matching])
            mean_auc_gi = np.mean([r["auc_gi"] for r in matching])
            mean_delta = np.mean([r["delta_pct"] for r in matching])
            mean_fill_ni = np.mean([r["fill_ni"] for r in matching])
            mean_fill_gi = np.mean([r["fill_gi"] for r in matching])

            print(f"{fsf:5.1f}  {sp:5.2f}  {mean_auc_ni:12.0f}  {mean_auc_gi:12.0f}  {mean_delta:+11.1f}%  {mean_fill_ni:12.3f}")

            table[f"{fsf}_{sp}"] = {
                "firm_shock_fraction": fsf,
                "shock_prob": sp,
                "default_supply": DEFAULT_SUPPLY,
                "shock_fraction": SHOCK_FRACTION,
                "K": K,
                "expedite_budget": EXPEDITE_BUDGET,
                "warmup_steps": WARMUP_STEPS,
                "init_inv": INIT_INV,
                "T": T,
                "mean_auc_ni": round(mean_auc_ni, 2),
                "mean_auc_gi": round(mean_auc_gi, 2),
                "mean_delta_pct": round(mean_delta, 2),
                "mean_fill_ni": round(mean_fill_ni, 4),
                "mean_fill_gi": round(mean_fill_gi, 4),
            }

    # Save
    output = {
        "calibration_table": table,
        "fixed_params": {
            "default_supply": DEFAULT_SUPPLY,
            "shock_fraction": SHOCK_FRACTION,
            "recovery_rate": RECOVERY_RATE,
            "warmup_steps": WARMUP_STEPS,
            "init_inv": INIT_INV,
            "T": T,
            "expedite_budget": EXPEDITE_BUDGET,
            "expedite_m_max": EXPEDITE_M_MAX,
            "K": K,
        },
        "raw_results": results,
        "errors": errors,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")
    if errors:
        print(f"WARNING: {len(errors)} errors occurred")


if __name__ == "__main__":
    main()
