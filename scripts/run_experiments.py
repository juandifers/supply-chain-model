"""
Experiment runner: runs all policies across multiple seeds and parameter configurations.
Exports results to artifacts/experiments/ as CSVs.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv
from scripts.graph_informed_optimizer import make_graph_informed_policy
from scripts.baseline_policies import (
    no_intervention_policy,
    make_random_reroute_policy,
    make_backlog_only_policy,
    make_expedite_only_policy,
    make_reroute_only_policy,
)


# ---------------------------------------------------------------------------
# Policy runner
# ---------------------------------------------------------------------------

def run_policy(
    policy_fn: Callable,
    policy_name: str,
    seed: int,
    T: int = 50,
    gamma: float = 0.8,
    shock_prob: float = 0.01,
    expedite_budget: Optional[float] = 200.0,
    expedite_c0: float = 1.0,
    expedite_alpha: float = 0.5,
    expedite_m_max: float = 3.0,
    init_inv: float = 0,
    init_supply: float = 100,
    init_demand: float = 1,
    default_supply: Optional[float] = None,
    shock_supply: Optional[float] = None,
    recovery_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a single episode with the given policy. Returns summary + per-step KPIs."""
    env = SupplySimEnv(
        seed=seed, T=T, gamma=gamma,
        expedite_budget=expedite_budget,
        expedite_c0=expedite_c0,
        expedite_alpha=expedite_alpha,
        expedite_m_max=expedite_m_max,
    )
    reset_kwargs = dict(
        init_inv=init_inv, init_supply=init_supply,
        init_demand=init_demand, shock_prob=shock_prob,
    )
    if default_supply is not None:
        reset_kwargs["default_supply"] = default_supply
    if shock_supply is not None:
        reset_kwargs["shock_supply"] = shock_supply
    if recovery_rate is not None:
        reset_kwargs["recovery_rate"] = recovery_rate
    obs = env.reset(**reset_kwargs)

    action_log = []
    done = False
    t = 0
    while not done:
        action, explanation = policy_fn(obs, t, env)
        obs, reward, done, info = env.step(action)
        action_log.append({
            "t": t,
            "reroutes": len(action.get("reroute", [])),
            "expedites": len(action.get("supply_multiplier", {})),
            "explanation_summary": explanation.get("policy_name", policy_name),
        })
        t += 1

    kpi_df = env.get_kpi_history()

    # Compute aggregate metrics
    backlog_auc = float(kpi_df["consumer_backlog_units"].sum())
    peak_backlog = float(kpi_df["consumer_backlog_units"].max())
    total_expedite_spend = float(kpi_df["expedite_cost_cum"].iloc[-1]) if len(kpi_df) > 0 else 0.0
    total_reroutes = int(kpi_df["reroutes_cumulative"].iloc[-1]) if len(kpi_df) > 0 else 0
    final_fill_rate = float(kpi_df["consumer_cumulative_fill_rate"].iloc[-1]) if len(kpi_df) > 0 else 0.0
    mean_fill_rate = float(kpi_df["consumer_new_demand_fill_rate"].mean()) if len(kpi_df) > 0 else 0.0

    # Time to recovery: first timestep after peak where backlog returns to pre-shock level
    if len(kpi_df) > 1:
        pre_shock_backlog = float(kpi_df["consumer_backlog_units"].iloc[0])
        peak_t = int(kpi_df["consumer_backlog_units"].idxmax())
        post_peak = kpi_df.iloc[peak_t:]
        recovered = post_peak[post_peak["consumer_backlog_units"] <= pre_shock_backlog * 1.1]
        time_to_recovery = int(recovered.iloc[0]["t"] - peak_t) if len(recovered) > 0 else T
    else:
        time_to_recovery = 0

    return {
        "policy_name": policy_name,
        "seed": seed,
        "T": T,
        "gamma": gamma,
        "shock_prob": shock_prob,
        "expedite_budget": expedite_budget,
        "reroute_budget_K": explanation.get("reroute_budget_K", None),
        "backlog_auc": backlog_auc,
        "peak_backlog": peak_backlog,
        "total_expedite_spend": total_expedite_spend,
        "total_reroutes": total_reroutes,
        "final_fill_rate": final_fill_rate,
        "mean_fill_rate": mean_fill_rate,
        "time_to_recovery": time_to_recovery,
        "kpi_history": kpi_df,
        "action_log": action_log,
    }


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

def build_default_policies(reroute_budget_K: int = 3) -> Dict[str, Callable]:
    """Build the standard set of policies for comparison."""
    return {
        "no_intervention": no_intervention_policy,
        "random_reroute": make_random_reroute_policy(reroute_budget_K=reroute_budget_K, seed=999),
        "backlog_only": make_backlog_only_policy(reroute_budget_K=reroute_budget_K),
        "expedite_only": make_expedite_only_policy(),
        "reroute_only": make_reroute_only_policy(reroute_budget_K=reroute_budget_K),
        "graph_informed": make_graph_informed_policy(reroute_budget_K=reroute_budget_K),
    }


def run_experiment(
    policies: Dict[str, Callable],
    seeds: List[int],
    T: int = 50,
    gamma: float = 0.8,
    shock_prob: float = 0.01,
    expedite_budget: Optional[float] = 200.0,
    expedite_c0: float = 1.0,
    expedite_alpha: float = 0.5,
    expedite_m_max: float = 3.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run all policies across all seeds. Returns summary DataFrame and full results list."""
    all_summaries = []
    all_results = []

    total_runs = len(policies) * len(seeds)
    run_idx = 0

    for policy_name, policy_fn in policies.items():
        for seed in seeds:
            run_idx += 1
            if verbose:
                print(f"[{run_idx}/{total_runs}] {policy_name} seed={seed} ...", end=" ", flush=True)

            t0 = time.time()
            result = run_policy(
                policy_fn=policy_fn,
                policy_name=policy_name,
                seed=seed,
                T=T, gamma=gamma, shock_prob=shock_prob,
                expedite_budget=expedite_budget,
                expedite_c0=expedite_c0, expedite_alpha=expedite_alpha,
                expedite_m_max=expedite_m_max,
            )
            elapsed = time.time() - t0

            summary = {k: v for k, v in result.items() if k not in ("kpi_history", "action_log")}
            summary["elapsed_s"] = round(elapsed, 2)
            all_summaries.append(summary)
            all_results.append(result)

            if verbose:
                print(f"backlog_auc={result['backlog_auc']:.0f} "
                      f"fill={result['final_fill_rate']:.3f} "
                      f"spend={result['total_expedite_spend']:.1f} "
                      f"({elapsed:.1f}s)")

    summary_df = pd.DataFrame(all_summaries)
    return summary_df, all_results


def run_parameter_sweep(
    sweep_param: str,
    sweep_values: List,
    seeds: List[int],
    base_config: Dict[str, Any],
    reroute_budget_K: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep one parameter across values, running all policies for each."""
    all_summaries = []

    for val in sweep_values:
        if verbose:
            print(f"\n=== {sweep_param}={val} ===")

        config = dict(base_config)
        if sweep_param == "reroute_budget_K":
            policies = build_default_policies(reroute_budget_K=val)
        else:
            policies = build_default_policies(reroute_budget_K=reroute_budget_K)
            config[sweep_param] = val

        summary_df, _ = run_experiment(
            policies=policies, seeds=seeds, verbose=verbose, **config
        )
        summary_df["sweep_param"] = sweep_param
        summary_df["sweep_value"] = val
        all_summaries.append(summary_df)

    return pd.concat(all_summaries, ignore_index=True)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    summary_df: pd.DataFrame,
    all_results: List[Dict],
    experiment_id: str,
    output_dir: str = None,
):
    """Save experiment results to artifacts/experiments/{experiment_id}/."""
    if output_dir is None:
        output_dir = os.path.join(ROOT, "artifacts", "experiments", experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Summary CSV
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Per-policy KPI histories
    kpi_dir = os.path.join(output_dir, "kpi_histories")
    os.makedirs(kpi_dir, exist_ok=True)
    for result in all_results:
        fname = f"{result['policy_name']}_seed{result['seed']}.csv"
        result["kpi_history"].to_csv(os.path.join(kpi_dir, fname), index=False)

    # Action logs
    action_dir = os.path.join(output_dir, "action_logs")
    os.makedirs(action_dir, exist_ok=True)
    for result in all_results:
        fname = f"{result['policy_name']}_seed{result['seed']}.json"
        with open(os.path.join(action_dir, fname), "w") as f:
            json.dump(result["action_log"], f, indent=2)

    # Manifest
    manifest = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_policies": len(summary_df["policy_name"].unique()),
        "num_seeds": len(summary_df["seed"].unique()),
        "policies": list(summary_df["policy_name"].unique()),
        "seeds": sorted(summary_df["seed"].unique().tolist()),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Results exported to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run supply chain optimizer experiments")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default 5)")
    parser.add_argument("--T", type=int, default=50, help="Episode length")
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--shock-prob", type=float, default=0.01)
    parser.add_argument("--expedite-budget", type=float, default=200.0)
    parser.add_argument("--reroute-budget-K", type=int, default=3)
    parser.add_argument("--default-supply", type=float, default=None,
                        help="Exogenous supply level (default 1e6 in simulator)")
    parser.add_argument("--shock-supply", type=float, default=None,
                        help="Supply level during shock (default 1000 in simulator)")
    parser.add_argument("--recovery-rate", type=float, default=None,
                        help="Multiplicative recovery rate per timestep (default 1.25)")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--sweep", type=str, default=None,
                        help="Parameter to sweep (e.g., shock_prob, expedite_budget)")
    parser.add_argument("--sweep-values", type=str, default=None,
                        help="Comma-separated sweep values")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    experiment_id = args.experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    base_config = {
        "T": args.T,
        "gamma": args.gamma,
        "shock_prob": args.shock_prob,
        "expedite_budget": args.expedite_budget,
    }
    if args.default_supply is not None:
        base_config["default_supply"] = args.default_supply
    if args.shock_supply is not None:
        base_config["shock_supply"] = args.shock_supply
    if args.recovery_rate is not None:
        base_config["recovery_rate"] = args.recovery_rate

    if args.sweep and args.sweep_values:
        sweep_values = [float(v) if "." in v else int(v) for v in args.sweep_values.split(",")]
        summary_df = run_parameter_sweep(
            sweep_param=args.sweep,
            sweep_values=sweep_values,
            seeds=seeds,
            base_config=base_config,
            reroute_budget_K=args.reroute_budget_K,
        )
        output_dir = os.path.join(ROOT, "artifacts", "experiments", experiment_id)
        os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(output_dir, "sweep_summary.csv"), index=False)
        print(f"Sweep results exported to {output_dir}")
    else:
        policies = build_default_policies(reroute_budget_K=args.reroute_budget_K)
        summary_df, all_results = run_experiment(
            policies=policies, seeds=seeds, **base_config
        )
        export_results(summary_df, all_results, experiment_id)

        # Print aggregate comparison
        print("\n=== AGGREGATE RESULTS ===")
        agg = summary_df.groupby("policy_name").agg(
            backlog_auc_mean=("backlog_auc", "mean"),
            backlog_auc_std=("backlog_auc", "std"),
            peak_backlog_mean=("peak_backlog", "mean"),
            fill_rate_mean=("final_fill_rate", "mean"),
            spend_mean=("total_expedite_spend", "mean"),
            reroutes_mean=("total_reroutes", "mean"),
        ).round(2)
        print(agg.to_string())


if __name__ == "__main__":
    main()
