"""
Shared utilities for all panel experiment scripts.
Extracted and extended from scripts/run_regime_experiment.py.

DO NOT modify scripts/run_regime_experiment.py — this file supersedes it
for the new panel suite. Single source of truth for:
  - run_single_experiment / run_single_experiment_v2
  - build_policy
  - save_result
  - DEFAULTS / ALL_POLICIES
  - delta metric computation
"""
import json
import os
import sys
import time
import subprocess
import traceback
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

# ── MIP availability ──────────────────────────────────────────────────────────
try:
    import pulp  # noqa: F401
    MIP_AVAILABLE = True
except ImportError:
    MIP_AVAILABLE = False

# ── Calibrated defaults (reworked simulator, 2026-03-20) ─────────────────────
DEFAULTS = {
    "default_supply":      100,
    "shock_magnitude":     0.70,
    "shock_prob":          0.15,
    "recovery_rate":       1.05,
    "firm_shock_fraction": 0.5,
    "warmup_steps":        10,
    "init_inv":            0,
    "init_demand":         10,
    "T":                   90,
    "expedite_budget":     50_000,
    "expedite_m_max":      3.0,
    "K":                   3,
    "num_inner_layers":    2,
    "num_per_layer":       10,
    "min_num_suppliers":   2,
    "max_num_suppliers":   3,
    "min_inputs":          1,
    "max_inputs":          2,
    "min_units":           1,
    "max_units":           1,
    "max_order_age":       10,
    "kpi_start_step":      4,
}

# ── V2 shock architecture defaults (calibrated 2026-03-22) ───────────────────
V2_DEFAULTS = {
    "default_supply":          100,
    "T":                       80,
    "warmup_steps":            10,
    "expedite_budget":         50_000,
    "expedite_m_max":          3.0,
    "K":                       3,
    # Shock generation
    "shock_prob":              0.20,
    "magnitude_mean":          0.85,
    "localized_firm_fraction": 0.5,
    # Shock dynamics
    "duration_mean":           15,
    "contagion_radius":        2,
    "contagion_prob_per_hop":  0.4,
    "recovery_shape":          "linear",
    # Policy interface
    "reroute_supply_bonus":    150.0,
    "reroute_setup_delay":     0,
    "reroute_capacity_fraction": 1.0,
    "expedite_convexity":      0.0,
    "expedite_capacity_cap":   1.0,
    # Topology
    "num_regions":             3,
    "supplier_capacity_cv":    0.4,
    "chokepoint_fraction":     0.2,
}

# Canonical 7-policy comparator set for all panels
ALL_POLICIES = [
    "no_intervention",
    "random_reroute",
    "reroute_only",
    "expedite_only",
    "backlog_greedy",
    "graph_informed",
    "mip",
]

# Panel 1 named regimes (V1)
P1_NAMED_REGIMES = {
    "tight_local_nobudget":  {"default_supply": 50,  "firm_shock_fraction": 0.3, "expedite_budget": 0},
    "tight_systemic_budget": {"default_supply": 50,  "firm_shock_fraction": 1.0, "expedite_budget": 50_000},
    "loose_local_nobudget":  {"default_supply": 150, "firm_shock_fraction": 0.3, "expedite_budget": 0},
    "loose_systemic_budget": {"default_supply": 150, "firm_shock_fraction": 1.0, "expedite_budget": 50_000},
}

# Panel 1 V2 regimes — use shock architecture event types instead of firm_shock_fraction
P1_V2_REGIMES = {
    "tight_local_eb0":     {"default_supply": 50,  "event_type": "local",    "expedite_budget": 0},
    "tight_local_eb50k":   {"default_supply": 50,  "event_type": "local",    "expedite_budget": 50_000},
    "tight_regional_eb0":  {"default_supply": 50,  "event_type": "regional", "expedite_budget": 0},
    "tight_regional_eb50k":{"default_supply": 50,  "event_type": "regional", "expedite_budget": 50_000},
    "loose_local_eb0":     {"default_supply": 100, "event_type": "local",    "expedite_budget": 0},
    "loose_local_eb50k":   {"default_supply": 100, "event_type": "local",    "expedite_budget": 50_000},
    "loose_regional_eb0":  {"default_supply": 100, "event_type": "regional", "expedite_budget": 0},
    "loose_regional_eb50k":{"default_supply": 100, "event_type": "regional", "expedite_budget": 50_000},
}

# Event type probability presets
EVENT_TYPE_PROBS = {
    "local":    {"localized": 0.8, "regional": 0.1, "cascade": 0.1},
    "regional": {"localized": 0.2, "regional": 0.6, "cascade": 0.2},
}


def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


GIT_COMMIT = get_git_commit()


def make_regime_id(config: dict) -> str:
    ds  = config.get("default_supply",      DEFAULTS["default_supply"])
    fsf = config.get("firm_shock_fraction", DEFAULTS["firm_shock_fraction"])
    sp  = config.get("shock_prob",          DEFAULTS["shock_prob"])
    sm  = config.get("shock_magnitude",     DEFAULTS["shock_magnitude"])
    rr  = config.get("recovery_rate",       DEFAULTS["recovery_rate"])
    eb  = config.get("expedite_budget",     DEFAULTS["expedite_budget"])
    return f"ds{ds}_fsf{fsf}_sp{sp}_sm{sm}_rr{rr}_eb{eb}"


def build_policy(name: str, K: int):
    """
    Build a policy function by name. Picklable — safe for multiprocessing.
    Supports: no_intervention, random_reroute, reroute_only, expedite_only,
              backlog_greedy (alias: backlog_only), graph_informed, mip
    """
    from scripts.baseline_policies import (
        no_intervention_policy,
        make_random_reroute_policy,
        make_backlog_only_policy,
        make_expedite_only_policy,
        make_reroute_only_policy,
        make_mip_policy,
    )
    from scripts.graph_informed_optimizer import make_graph_informed_policy

    if name == "no_intervention":
        return no_intervention_policy
    elif name == "random_reroute":
        return make_random_reroute_policy(reroute_budget_K=K, seed=999)
    elif name in ("backlog_greedy", "backlog_only"):
        return make_backlog_only_policy(reroute_budget_K=K)
    elif name == "reroute_only":
        return make_reroute_only_policy(reroute_budget_K=K)
    elif name == "expedite_only":
        return make_expedite_only_policy()
    elif name == "graph_informed":
        return make_graph_informed_policy(reroute_budget_K=K)
    elif name == "mip":
        if not MIP_AVAILABLE:
            raise ValueError("MIP requested but 'pulp' is not installed. Use --no-mip.")
        return make_mip_policy(reroute_budget_K=K)
    else:
        raise ValueError(f"Unknown policy: {name!r}")


def run_single_experiment(exp: dict) -> dict:
    """
    Run one (policy, config, seed) triple end-to-end. Picklable for multiprocessing.

    exp keys:
      policy     : str  — policy name
      seed       : int
      config     : dict — full parameter config
      regime_id  : str  — label for this regime
      panel_name : str
    """
    from scripts.calibrated_scenario import create_calibrated_env

    policy_name = exp["policy"]
    seed        = exp["seed"]
    config      = exp["config"]
    regime_id   = exp.get("regime_id",  "unknown")
    panel_name  = exp.get("panel_name", "unknown")
    K           = int(config.get("K", DEFAULTS["K"]))

    # shock_fraction backward-compat alias (shock_fraction=0.3 → shock_magnitude=0.7)
    if "shock_fraction" in config and "shock_magnitude" not in config:
        config = dict(config)
        config["shock_magnitude"] = round(1.0 - float(config["shock_fraction"]), 6)

    t0        = time.time()
    policy_fn = build_policy(policy_name, K)

    env, obs, shock_log = create_calibrated_env(
        seed                 = seed,
        T                    = config.get("T",                    DEFAULTS["T"]),
        default_supply       = config.get("default_supply",       DEFAULTS["default_supply"]),
        shock_magnitude      = config.get("shock_magnitude",      DEFAULTS["shock_magnitude"]),
        shock_prob           = config.get("shock_prob",           DEFAULTS["shock_prob"]),
        recovery_rate        = config.get("recovery_rate",        DEFAULTS["recovery_rate"]),
        firm_shock_fraction  = config.get("firm_shock_fraction",  DEFAULTS["firm_shock_fraction"]),
        warmup_steps         = config.get("warmup_steps",         DEFAULTS["warmup_steps"]),
        init_inv             = config.get("init_inv",             DEFAULTS["init_inv"]),
        init_demand          = config.get("init_demand",          DEFAULTS["init_demand"]),
        expedite_budget      = config.get("expedite_budget",      DEFAULTS["expedite_budget"]),
        expedite_m_max       = config.get("expedite_m_max",       DEFAULTS["expedite_m_max"]),
        num_inner_layers     = config.get("num_inner_layers",     DEFAULTS["num_inner_layers"]),
        num_per_layer        = config.get("num_per_layer",        DEFAULTS["num_per_layer"]),
        min_num_suppliers    = config.get("min_num_suppliers",    DEFAULTS["min_num_suppliers"]),
        max_num_suppliers    = config.get("max_num_suppliers",    DEFAULTS["max_num_suppliers"]),
        min_inputs           = config.get("min_inputs",           DEFAULTS["min_inputs"]),
        max_inputs           = config.get("max_inputs",           DEFAULTS["max_inputs"]),
        min_units            = config.get("min_units",            DEFAULTS["min_units"]),
        max_units            = config.get("max_units",            DEFAULTS["max_units"]),
        max_order_age        = config.get("max_order_age",        DEFAULTS["max_order_age"]),
        kpi_start_step       = config.get("kpi_start_step",      DEFAULTS["kpi_start_step"]),
    )

    # ── Episode loop ──────────────────────────────────────────────────────────
    done               = False
    mip_fallback_steps = 0
    mip_step_times     = []

    while not done:
        if policy_name == "mip":
            step_t0 = time.time()
            try:
                action, _expl = policy_fn(obs, env.t, env)
            except Exception:
                mip_fallback_steps += 1
                action = {"reroute": [], "supply_multiplier": {}}
            mip_step_times.append(time.time() - step_t0)
        else:
            action, _expl = policy_fn(obs, env.t, env)
        obs, _reward, done, _info = env.step(action)

    kpi_df  = env.get_kpi_history()
    elapsed = time.time() - t0

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    backlog_series   = kpi_df["consumer_backlog_units"]
    backlog_auc      = float(backlog_series.sum())
    peak_backlog     = float(backlog_series.max())
    final_backlog    = float(backlog_series.iloc[-1])
    fill_rate        = float(kpi_df["consumer_cumulative_fill_rate"].iloc[-1])
    lost_sales_units = (float(kpi_df["lost_sales_units"].sum())
                        if "lost_sales_units" in kpi_df.columns else 0.0)

    # Time to recovery: steps from peak until backlog ≤ pre-peak level
    T_len = int(config.get("T", DEFAULTS["T"]))
    ttr   = T_len
    if len(kpi_df) > 1:
        pre_level  = float(backlog_series.iloc[0])
        threshold  = max(pre_level * 1.1, pre_level + 1.0, 1.0)
        peak_pos   = int(backlog_series.values.argmax())
        post_vals  = backlog_series.values[peak_pos:]
        recovered  = np.where(post_vals <= threshold)[0]
        ttr = int(recovered[0]) if len(recovered) > 0 else T_len

    mean_mip_step_s = float(np.mean(mip_step_times)) if mip_step_times else 0.0
    is_baseline_run = (
        float(config.get("shock_prob",      1.0)) == 0.0 and
        float(config.get("shock_magnitude", 1.0)) == 0.0
    )

    return {
        "panel_name":           panel_name,
        "regime_id":            regime_id,
        "policy":               policy_name,
        "seed":                 seed,
        "git_commit":           GIT_COMMIT,
        "is_baseline_run":      is_baseline_run,
        # Regime params — always shock_magnitude, never shock_fraction
        "default_supply":       config.get("default_supply",       DEFAULTS["default_supply"]),
        "shock_magnitude":      config.get("shock_magnitude",      DEFAULTS["shock_magnitude"]),
        "shock_prob":           config.get("shock_prob",           DEFAULTS["shock_prob"]),
        "recovery_rate":        config.get("recovery_rate",        DEFAULTS["recovery_rate"]),
        "firm_shock_fraction":  config.get("firm_shock_fraction",  DEFAULTS["firm_shock_fraction"]),
        "expedite_budget":      config.get("expedite_budget",      DEFAULTS["expedite_budget"]),
        "K":                    K,
        # Core metrics
        "backlog_auc":          backlog_auc,
        "fill_rate":            fill_rate,
        "peak_backlog":         peak_backlog,
        "final_backlog":        final_backlog,
        "lost_sales_units":     lost_sales_units,
        "time_to_recovery":     ttr,
        "runtime_s":            round(elapsed, 3),
        "mip_fallback_steps":   mip_fallback_steps,
        "mean_mip_step_s":      round(mean_mip_step_s, 4),
        "n_shocks":             len(shock_log),
    }


def _build_arch_config(config: dict):
    """Build an ArchitectureConfig from a flat config dict."""
    from scripts.shock_architecture import (
        ArchitectureConfig, TopologyConfig, ShockGenerationConfig,
        ShockDynamicsConfig, PolicyInterfaceConfig,
    )

    event_type = config.get("event_type", "local")
    event_probs = EVENT_TYPE_PROBS.get(event_type, EVENT_TYPE_PROBS["local"])
    if "event_type_probs" in config:
        event_probs = config["event_type_probs"]

    topo = TopologyConfig(
        num_regions=config.get("num_regions", V2_DEFAULTS["num_regions"]),
        supplier_capacity_cv=config.get("supplier_capacity_cv", V2_DEFAULTS["supplier_capacity_cv"]),
        chokepoint_fraction=config.get("chokepoint_fraction", V2_DEFAULTS["chokepoint_fraction"]),
    )
    gen = ShockGenerationConfig(
        shock_prob=config.get("shock_prob", V2_DEFAULTS["shock_prob"]),
        magnitude_mean=config.get("magnitude_mean", V2_DEFAULTS["magnitude_mean"]),
        epicenter_bias="region",
        event_type_probs=event_probs,
        localized_firm_fraction=config.get("localized_firm_fraction", V2_DEFAULTS["localized_firm_fraction"]),
    )
    dyn = ShockDynamicsConfig(
        duration_mean=config.get("duration_mean", V2_DEFAULTS["duration_mean"]),
        contagion_radius=config.get("contagion_radius", V2_DEFAULTS["contagion_radius"]),
        contagion_prob_per_hop=config.get("contagion_prob_per_hop", V2_DEFAULTS["contagion_prob_per_hop"]),
        recovery_shape=config.get("recovery_shape", V2_DEFAULTS["recovery_shape"]),
        recovery_steps=8,
    )
    pi = PolicyInterfaceConfig(
        expedite_budget=config.get("expedite_budget", V2_DEFAULTS["expedite_budget"]),
        reroute_setup_delay=config.get("reroute_setup_delay", V2_DEFAULTS["reroute_setup_delay"]),
        reroute_capacity_fraction=config.get("reroute_capacity_fraction", V2_DEFAULTS["reroute_capacity_fraction"]),
        expedite_convexity=config.get("expedite_convexity", V2_DEFAULTS["expedite_convexity"]),
        expedite_capacity_cap=config.get("expedite_capacity_cap", V2_DEFAULTS["expedite_capacity_cap"]),
        reroute_supply_bonus=config.get("reroute_supply_bonus", V2_DEFAULTS["reroute_supply_bonus"]),
    )
    return ArchitectureConfig(topology=topo, shock_generation=gen, shock_dynamics=dyn, policy_interface=pi)


def run_single_experiment_v2(exp: dict) -> dict:
    """
    Run one (policy, config, seed) triple using the V2 shock architecture.

    exp keys:
      policy     : str  — policy name
      seed       : int
      config     : dict — full parameter config
      regime_id  : str  — label for this regime
      panel_name : str
    """
    from scripts.supplysim_env_v2 import SupplySimEnvV2

    policy_name = exp["policy"]
    seed        = exp["seed"]
    config      = exp["config"]
    regime_id   = exp.get("regime_id",  "unknown")
    panel_name  = exp.get("panel_name", "unknown")
    K           = int(config.get("K", V2_DEFAULTS["K"]))

    t0        = time.time()
    policy_fn = build_policy(policy_name, K)

    # Build architecture config
    is_baseline_run = config.get("is_baseline_run", False)
    arch_config = _build_arch_config(config)

    # For baseline runs, disable shocks
    if is_baseline_run:
        arch_config.shock_generation.shock_prob = 0.0

    eb = config.get("expedite_budget", V2_DEFAULTS["expedite_budget"])

    env = SupplySimEnvV2(
        seed=seed,
        T=config.get("T", V2_DEFAULTS["T"]),
        gamma=0.8,
        log_kpis=False,
        arch_config=arch_config,
        expedite_budget=eb,
        expedite_m_max=config.get("expedite_m_max", V2_DEFAULTS["expedite_m_max"]),
        shock_architecture_enabled=True,
    )

    ds = config.get("default_supply", V2_DEFAULTS["default_supply"])
    warmup = config.get("warmup_steps", V2_DEFAULTS["warmup_steps"])
    obs = env.reset(default_supply=ds, warmup_steps=warmup)

    # ── Episode loop ──────────────────────────────────────────────────────────
    done               = False
    mip_fallback_steps = 0
    mip_step_times     = []

    while not done:
        if policy_name == "mip":
            step_t0 = time.time()
            try:
                action, _expl = policy_fn(obs, env.t, env)
            except Exception:
                mip_fallback_steps += 1
                action = {"reroute": [], "supply_multiplier": {}}
            mip_step_times.append(time.time() - step_t0)
        else:
            action, _expl = policy_fn(obs, env.t, env)
        obs, _reward, done, _info = env.step(action)

    kpi_df  = env.get_kpi_history()
    elapsed = time.time() - t0

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    backlog_series   = kpi_df["consumer_backlog_units"]
    backlog_auc      = float(backlog_series.sum())
    peak_backlog     = float(backlog_series.max())
    final_backlog    = float(backlog_series.iloc[-1])
    fill_rate        = float(kpi_df["consumer_cumulative_fill_rate"].iloc[-1])
    lost_sales_units = (float(kpi_df["lost_sales_units"].sum())
                        if "lost_sales_units" in kpi_df.columns else 0.0)

    T_len = int(config.get("T", V2_DEFAULTS["T"]))
    ttr   = T_len
    if len(kpi_df) > 1:
        pre_level  = float(backlog_series.iloc[0])
        threshold  = max(pre_level * 1.1, pre_level + 1.0, 1.0)
        peak_pos   = int(backlog_series.values.argmax())
        post_vals  = backlog_series.values[peak_pos:]
        recovered  = np.where(post_vals <= threshold)[0]
        ttr = int(recovered[0]) if len(recovered) > 0 else T_len

    mean_mip_step_s = float(np.mean(mip_step_times)) if mip_step_times else 0.0
    n_shocks = len(env._shock_events_log) if hasattr(env, '_shock_events_log') else 0

    return {
        "panel_name":           panel_name,
        "regime_id":            regime_id,
        "policy":               policy_name,
        "seed":                 seed,
        "git_commit":           GIT_COMMIT,
        "is_baseline_run":      is_baseline_run,
        "arch":                 "V2",
        # Regime params
        "default_supply":       ds,
        "event_type":           config.get("event_type", "local"),
        "shock_prob":           config.get("shock_prob", V2_DEFAULTS["shock_prob"]),
        "magnitude_mean":       config.get("magnitude_mean", V2_DEFAULTS["magnitude_mean"]),
        "expedite_budget":      eb,
        "K":                    K,
        # Core metrics
        "backlog_auc":          backlog_auc,
        "fill_rate":            fill_rate,
        "peak_backlog":         peak_backlog,
        "final_backlog":        final_backlog,
        "lost_sales_units":     lost_sales_units,
        "time_to_recovery":     ttr,
        "runtime_s":            round(elapsed, 3),
        "mip_fallback_steps":   mip_fallback_steps,
        "mean_mip_step_s":      round(mean_mip_step_s, 4),
        "n_shocks":             n_shocks,
    }


def save_result(result: dict, output_path: Path) -> None:
    """Append one result row to CSV. Creates with header on first write."""
    df     = pd.DataFrame([result])
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False)


def load_completed(output_path: Path) -> set:
    """Return set of (regime_id, policy, seed, is_baseline_run) already done."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(
            output_path, usecols=["regime_id", "policy", "seed", "is_baseline_run"]
        )
        return set(zip(
            existing["regime_id"].astype(str),
            existing["policy"].astype(str),
            existing["seed"].astype(int),
            existing["is_baseline_run"].map(lambda x: str(x).lower() == "true"),
        ))
    except Exception:
        return set()


def exp_key(e: dict) -> tuple:
    """Unique key for an experiment dict (for resume deduplication)."""
    is_bl = (
        float(e["config"].get("shock_prob",      1.0)) == 0.0 and
        float(e["config"].get("shock_magnitude", 1.0)) == 0.0
    )
    return (str(e["regime_id"]), str(e["policy"]), int(e["seed"]), bool(is_bl))


def compute_delta_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-run DataFrame (shocked + baseline rows), compute delta metrics.

    Baseline rows: is_baseline_run == True  (shock_prob=0, shock_magnitude=0)
    Shocked  rows: is_baseline_run == False

    Returns enriched shocked-only DataFrame with these columns added:
      backlog_auc_base, fill_rate_base, peak_backlog_base, lost_sales_base,
      delta_backlog_auc, delta_fill_rate, delta_peak_backlog, delta_lost_sales,
      policy_gain_total, policy_gain_pct, policy_gain_on_damage,
      disruption_too_mild, mip_vs_graph_informed_gap_pct
    """
    if df.empty:
        return df.copy()

    # Normalise the bool column (CSV may store as "True"/"False" strings)
    df = df.copy()
    df["is_baseline_run"] = df["is_baseline_run"].map(
        lambda x: str(x).lower() in ("true", "1", "yes")
    )

    baseline = df[df["is_baseline_run"]].copy()
    shocked  = df[~df["is_baseline_run"]].copy()

    nan_cols = [
        "backlog_auc_base", "fill_rate_base", "peak_backlog_base", "lost_sales_base",
        "delta_backlog_auc", "delta_fill_rate", "delta_peak_backlog", "delta_lost_sales",
        "policy_gain_total", "policy_gain_pct", "policy_gain_on_damage",
        "disruption_too_mild", "mip_vs_graph_informed_gap_pct",
    ]
    if baseline.empty or shocked.empty:
        for c in nan_cols:
            shocked[c] = np.nan
        return shocked.reset_index(drop=True)

    # Baseline lookup: prefer no_intervention rows; fall back to any
    bl_src = baseline[baseline["policy"] == "no_intervention"]
    if bl_src.empty:
        bl_src = baseline
    bl_lookup = (
        bl_src
        .groupby(["regime_id", "seed"])
        [["backlog_auc", "fill_rate", "peak_backlog", "lost_sales_units"]]
        .first()
        .rename(columns={
            "backlog_auc":       "backlog_auc_base",
            "fill_rate":         "fill_rate_base",
            "peak_backlog":      "peak_backlog_base",
            "lost_sales_units":  "lost_sales_base",
        })
    )
    shocked = shocked.join(bl_lookup, on=["regime_id", "seed"], how="left")

    # Disruption damage
    shocked["delta_backlog_auc"]  = shocked["backlog_auc"]        - shocked["backlog_auc_base"]
    shocked["delta_fill_rate"]    = shocked["fill_rate"]           - shocked["fill_rate_base"]
    shocked["delta_peak_backlog"] = shocked["peak_backlog"]        - shocked["peak_backlog_base"]
    shocked["delta_lost_sales"]   = shocked["lost_sales_units"]    - shocked["lost_sales_base"]

    # No-intervention benchmark per (regime_id, seed)
    ni = (
        shocked[shocked["policy"] == "no_intervention"]
        .groupby(["regime_id", "seed"])
        [["backlog_auc", "delta_backlog_auc"]]
        .first()
        .rename(columns={
            "backlog_auc":       "_ni_auc",
            "delta_backlog_auc": "_ni_delta_auc",
        })
    )
    shocked = shocked.join(ni, on=["regime_id", "seed"], how="left")

    # Policy gain within shocked regime
    shocked["policy_gain_total"] = shocked["_ni_auc"] - shocked["backlog_auc"]
    shocked["policy_gain_pct"]   = (
        shocked["policy_gain_total"] / shocked["_ni_auc"].clip(lower=1e-6)
    )

    # Policy gain on disruption damage
    denom = shocked["_ni_delta_auc"].clip(lower=1e-6)
    shocked["policy_gain_on_damage"] = (
        (shocked["_ni_delta_auc"] - shocked["delta_backlog_auc"]) / denom
    )

    # disruption_too_mild flag
    mild_mask = shocked["_ni_delta_auc"].fillna(0) < 500
    shocked["disruption_too_mild"] = mild_mask
    shocked.loc[mild_mask, "policy_gain_on_damage"] = np.nan

    # MIP vs graph_informed gap
    gi_auc = (
        shocked[shocked["policy"] == "graph_informed"]
        .groupby(["regime_id", "seed"])["backlog_auc"]
        .first()
        .rename("_gi_auc")
    )
    shocked = shocked.join(gi_auc, on=["regime_id", "seed"], how="left")
    shocked["mip_vs_graph_informed_gap_pct"] = np.where(
        shocked["policy"] == "mip",
        (shocked["backlog_auc"] - shocked["_gi_auc"]) / shocked["_ni_auc"].clip(lower=1e-6),
        np.nan,
    )

    shocked = shocked.drop(columns=["_ni_auc", "_ni_delta_auc", "_gi_auc"], errors="ignore")
    return shocked.reset_index(drop=True)


def aggregate_results(df: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """Compute mean ± std over seeds for each (regime_id, policy)."""
    if group_cols is None:
        group_cols = ["regime_id", "policy"]
    metric_cols = [
        "backlog_auc", "fill_rate", "peak_backlog", "final_backlog",
        "lost_sales_units", "time_to_recovery", "runtime_s",
        "policy_gain_pct", "policy_gain_on_damage",
        "mip_vs_graph_informed_gap_pct",
        "delta_backlog_auc", "delta_fill_rate", "delta_peak_backlog",
        "mip_fallback_steps",
    ]
    existing = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(group_cols)[existing].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def estimate_runtime(experiments: list, workers: int = 1) -> dict:
    fast = {"no_intervention", "random_reroute", "expedite_only"}
    mid  = {"reroute_only", "backlog_greedy", "backlog_only"}
    slow = {"graph_informed"}
    mip  = {"mip"}
    n_fast = sum(1 for e in experiments if e["policy"] in fast)
    n_mid  = sum(1 for e in experiments if e["policy"] in mid)
    n_slow = sum(1 for e in experiments if e["policy"] in slow)
    n_mip  = sum(1 for e in experiments if e["policy"] in mip)
    est = n_fast * 0.5 + n_mid * 1.5 + n_slow * 5.0 + n_mip * 15.0
    return {
        "total_runs": len(experiments),
        "est_seconds_1worker": est,
        "est_hours_1worker":   round(est / 3600, 2),
        f"est_hours_{workers}workers": round(est / max(workers, 1) / 3600, 2),
    }
