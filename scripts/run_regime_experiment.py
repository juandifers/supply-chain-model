"""
STANDALONE SCRIPT — run manually in terminal.

Full experimental grid: 8 policies x N seeds x M regime configs.
Supports multiprocessing, resume, dry-run, and skip-policies.

Usage:
    python scripts/run_regime_experiment.py \
        --calibration artifacts/experiments/calibration_table.json \
        --output-dir artifacts/experiments/regime_mapping/ \
        --seeds 20 \
        --workers 4

    # Dry run (see what will be run without running it):
    python scripts/run_regime_experiment.py --dry-run

    # Skip slow policies for a fast first pass:
    python scripts/run_regime_experiment.py --skip-policies mip

    # Resume after crash (skips already-completed runs):
    python scripts/run_regime_experiment.py --resume
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm


# --- Default regime grid (calibrated 2026-03-19) ---
FIRM_SHOCK_FRACTIONS = [0.3, 0.5, 0.7, 1.0]
SHOCK_MAGNITUDES = [0.2, 0.4, 0.6, 0.8]  # Primary severity axis (0.2 = mild 20% drop, 0.8 = severe 80% drop)
SHOCK_PROBS = [0.05, 0.1, 0.15, 0.2]

# Fixed params (can be overridden by calibration file)
DEFAULTS = {
    "default_supply": 100,       # Calibrated: right-sized to demand with reduced BOM
    "shock_magnitude": 0.7,
    "recovery_rate": 1.05,       # Calibrated: slower recovery amplifies shock persistence
    "warmup_steps": 10,
    "init_inv": 0,
    "init_demand": 10,           # Calibrated: with min_units=1, max_units=1 BOM
    "T": 90,
    "expedite_budget": 50_000,
    "expedite_m_max": 3.0,
    "K": 3,
    # Graph structure
    "num_inner_layers": 2,
    "num_per_layer": 10,
    "min_num_suppliers": 2,
    "max_num_suppliers": 3,
    "min_inputs": 1,
    "max_inputs": 2,
    "min_units": 1,
    "max_units": 1,
    # Order expiry + KPI warm-start
    "max_order_age": 10,
    "kpi_start_step": 4,
}


def build_policy(name, K):
    """Build a policy function by name. Must be importable in worker processes."""
    from scripts.baseline_policies import (
        no_intervention_policy,
        make_random_reroute_policy,
        make_backlog_only_policy,
        make_expedite_only_policy,
        make_reroute_only_policy,
        make_threshold_policy,
        make_mip_policy,
    )
    from scripts.graph_informed_optimizer import make_graph_informed_policy

    if name == "no_intervention":
        return no_intervention_policy
    elif name == "random_reroute":
        return make_random_reroute_policy(reroute_budget_K=K, seed=999)
    elif name == "threshold":
        return make_threshold_policy(reroute_budget_K=K)
    elif name == "backlog_only":
        return make_backlog_only_policy(reroute_budget_K=K)
    elif name == "graph_informed":
        return make_graph_informed_policy(reroute_budget_K=K)
    elif name == "mip":
        return make_mip_policy(reroute_budget_K=K)
    elif name == "reroute_only":
        return make_reroute_only_policy(reroute_budget_K=K)
    elif name == "expedite_only":
        return make_expedite_only_policy()
    else:
        raise ValueError(f"Unknown policy: {name}")


ALL_POLICIES = [
    "no_intervention", "random_reroute", "threshold", "backlog_only",
    "graph_informed", "mip", "reroute_only", "expedite_only",
]


def run_single_experiment(exp):
    """Run a single (policy, config, seed) experiment. Picklable for multiprocessing."""
    from scripts.calibrated_scenario import create_calibrated_env

    policy_name = exp["policy"]
    seed = exp["seed"]
    config = exp["config"]
    K = config.get("K", DEFAULTS["K"])

    t0 = time.time()
    policy_fn = build_policy(policy_name, K)

    env, obs, shock_log = create_calibrated_env(
        seed=seed,
        T=config.get("T", DEFAULTS["T"]),
        default_supply=config.get("default_supply", DEFAULTS["default_supply"]),
        shock_magnitude=config.get("shock_magnitude", DEFAULTS["shock_magnitude"]),
        shock_prob=config["shock_prob"],
        recovery_rate=config.get("recovery_rate", DEFAULTS["recovery_rate"]),
        firm_shock_fraction=config["firm_shock_fraction"],
        warmup_steps=config.get("warmup_steps", DEFAULTS["warmup_steps"]),
        init_inv=config.get("init_inv", DEFAULTS["init_inv"]),
        init_demand=config.get("init_demand", DEFAULTS["init_demand"]),
        expedite_budget=config.get("expedite_budget", DEFAULTS["expedite_budget"]),
        expedite_m_max=config.get("expedite_m_max", DEFAULTS["expedite_m_max"]),
        num_inner_layers=config.get("num_inner_layers", DEFAULTS["num_inner_layers"]),
        num_per_layer=config.get("num_per_layer", DEFAULTS["num_per_layer"]),
        min_num_suppliers=config.get("min_num_suppliers", DEFAULTS["min_num_suppliers"]),
        max_num_suppliers=config.get("max_num_suppliers", DEFAULTS["max_num_suppliers"]),
        min_inputs=config.get("min_inputs", DEFAULTS["min_inputs"]),
        max_inputs=config.get("max_inputs", DEFAULTS["max_inputs"]),
        min_units=config.get("min_units", DEFAULTS["min_units"]),
        max_units=config.get("max_units", DEFAULTS["max_units"]),
        max_order_age=config.get("max_order_age", DEFAULTS["max_order_age"]),
        kpi_start_step=config.get("kpi_start_step", DEFAULTS["kpi_start_step"]),
    )

    done = False
    while not done:
        action, explanation = policy_fn(obs, env.t, env)
        obs, _, done, info = env.step(action)

    kpi_df = env.get_kpi_history()
    elapsed = time.time() - t0

    backlog_auc = float(kpi_df["consumer_backlog_units"].sum())
    peak_backlog = float(kpi_df["consumer_backlog_units"].max())
    final_fill = float(kpi_df["consumer_cumulative_fill_rate"].iloc[-1])
    mean_fill = float(kpi_df["consumer_new_demand_fill_rate"].mean())
    total_spend = float(kpi_df["expedite_cost_cum"].iloc[-1])
    total_reroutes = int(kpi_df["reroutes_cumulative"].iloc[-1])

    # Time to recovery
    if len(kpi_df) > 1:
        pre_shock = float(kpi_df["consumer_backlog_units"].iloc[0])
        peak_t = int(kpi_df["consumer_backlog_units"].idxmax())
        post_peak = kpi_df.iloc[peak_t:]
        recovered = post_peak[post_peak["consumer_backlog_units"] <= pre_shock * 1.1]
        ttr = int(recovered.iloc[0]["t"] - peak_t) if len(recovered) > 0 else config.get("T", DEFAULTS["T"])
    else:
        ttr = 0

    return {
        "config_name": exp["config_name"],
        "policy": policy_name,
        "seed": seed,
        "firm_shock_fraction": config["firm_shock_fraction"],
        "shock_prob": config["shock_prob"],
        "backlog_auc": backlog_auc,
        "peak_backlog": peak_backlog,
        "final_fill_rate": final_fill,
        "mean_fill_rate": mean_fill,
        "total_expedite_spend": total_spend,
        "total_reroutes": total_reroutes,
        "time_to_recovery": ttr,
        "n_shocks": len(shock_log),
        "elapsed_s": round(elapsed, 2),
    }


def build_experiment_list(args, calibration=None):
    """Build list of all experiments to run."""
    experiments = []
    policies = [p for p in ALL_POLICIES if p not in (args.skip_policies or [])]

    for fsf in FIRM_SHOCK_FRACTIONS:
        for sp in SHOCK_PROBS:
            config_name = f"fsf{fsf}_sp{sp}"

            # Get config from calibration or use defaults
            config = dict(DEFAULTS)
            config["firm_shock_fraction"] = fsf
            config["shock_prob"] = sp

            if calibration:
                cal_key = f"{fsf}_{sp}"
                if cal_key in calibration.get("calibration_table", {}):
                    cal_entry = calibration["calibration_table"][cal_key]
                    for k in ["default_supply", "shock_magnitude", "K", "expedite_budget",
                              "warmup_steps", "init_inv", "T"]:
                        if k in cal_entry:
                            config[k] = cal_entry[k]

                # Also apply fixed_params from calibration
                for k, v in calibration.get("fixed_params", {}).items():
                    if k not in config:
                        config[k] = v

            for policy in policies:
                for seed in range(args.seeds):
                    experiments.append({
                        "config_name": config_name,
                        "policy": policy,
                        "seed": seed,
                        "config": config,
                    })

    return experiments


def save_result(result, output_path):
    """Append a single result row to CSV (resumable)."""
    df = pd.DataFrame([result])
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False)


def main():
    parser = argparse.ArgumentParser(description="Full regime experiment grid")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to calibration_table.json")
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments/regime_mapping")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-policies", type=str, nargs="*", default=None,
                        help="Policies to skip (e.g., mip)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment count without running")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed runs")
    args = parser.parse_args()

    # Load calibration if provided
    calibration = None
    if args.calibration and os.path.exists(args.calibration):
        with open(args.calibration) as f:
            calibration = json.load(f)
        print(f"Loaded calibration from {args.calibration}")

    experiments = build_experiment_list(args, calibration)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_results.csv"
    error_log = output_dir / "errors.log"

    if args.dry_run:
        policies = sorted(set(e["policy"] for e in experiments))
        configs = sorted(set(e["config_name"] for e in experiments))
        print(f"Would run {len(experiments)} experiments")
        print(f"Policies ({len(policies)}): {policies}")
        print(f"Configs ({len(configs)}): {configs}")
        print(f"Seeds: {args.seeds}")

        # Rough time estimate from pilot (if available)
        fast_policies = {"no_intervention", "random_reroute", "threshold", "backlog_only"}
        n_fast = sum(1 for e in experiments if e["policy"] in fast_policies)
        n_graph = sum(1 for e in experiments if e["policy"] in {"graph_informed", "reroute_only", "expedite_only"})
        n_mip = sum(1 for e in experiments if e["policy"] == "mip")
        est_seconds = n_fast * 0.7 + n_graph * 5.0 + n_mip * 10.0
        for w in [1, 2, 4, 8]:
            print(f"  {w} workers: ~{est_seconds / w / 3600:.1f} hours (rough estimate)")
        return

    # Handle resume
    completed = set()
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        completed = set(zip(existing["config_name"], existing["policy"], existing["seed"]))
        print(f"Resuming: {len(completed)} already completed")

    remaining = [e for e in experiments
                 if (e["config_name"], e["policy"], e["seed"]) not in completed]
    print(f"Running {len(remaining)} experiments ({len(completed)} already done)")

    if len(remaining) == 0:
        print("Nothing to do!")
        return

    with tqdm(total=len(remaining), desc="Regime Experiments",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        if args.workers == 1:
            for exp in remaining:
                try:
                    result = run_single_experiment(exp)
                    save_result(result, output_path)
                    pbar.set_postfix({
                        "policy": exp["policy"][:12],
                        "config": exp["config_name"][:15],
                        "auc": f"{result['backlog_auc']:.0f}",
                    })
                except Exception as e:
                    with open(error_log, "a") as f:
                        f.write(f"FAILED: {exp['config_name']} {exp['policy']} seed={exp['seed']} — {e}\n")
                        f.write(traceback.format_exc() + "\n")
                    pbar.set_postfix({"status": "ERROR", "policy": exp["policy"]})
                pbar.update(1)
        else:
            with multiprocessing.Pool(args.workers) as pool:
                for result in pool.imap_unordered(run_single_experiment, remaining):
                    try:
                        save_result(result, output_path)
                        pbar.set_postfix({
                            "policy": result["policy"][:12],
                            "auc": f"{result['backlog_auc']:.0f}",
                        })
                    except Exception as e:
                        with open(error_log, "a") as f:
                            f.write(f"SAVE ERROR: {e}\n")
                    pbar.update(1)

    print(f"\nComplete! Results saved to {output_path}")
    print(f"Total runs: {len(remaining) + len(completed)}")
    print(f"Run: python scripts/analyze_regimes.py --input {output_path}")


if __name__ == "__main__":
    main()
