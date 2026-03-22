"""
Calibrated scenario generator for meaningful policy differentiation.

Three key modifications over the default generator:
1. Proportional shocks: shock_supply = default_supply * (1 - shock_magnitude) (not absolute)
2. Firm-level shocks: only a fraction of firms are affected per shocked product
3. Warm-up period: first warmup_steps timesteps are shock-free

Does NOT modify TGB/modules/synthetic_data.py — generates a replacement exog_schedule.
"""
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TGB.modules import synthetic_data as sd


def generate_calibrated_exog_schedule(
    prod_graph,
    prod2firms,
    num_timesteps: int,
    seed: int = 0,
    default_supply: float = 1e6,
    shock_magnitude: float = 0.7,
    shock_prob: float = 0.15,
    recovery_rate: float = 1.25,
    firm_shock_fraction: float = 0.5,
    warmup_steps: int = 15,
):
    """
    Generate a calibrated exogenous supply schedule.

    Parameters
    ----------
    prod_graph : DataFrame
        Product dependency DAG with columns [source, dest, units, layer].
    prod2firms : dict
        Mapping product -> list of firms that supply it.
    num_timesteps : int
        Total simulation length.
    seed : int
        RNG seed for reproducibility.
    default_supply : float
        Baseline supply level per firm per product.
    shock_magnitude : float
        Fraction of default_supply lost during shock (0.7 = 70% reduction).
    shock_prob : float
        Per-product per-timestep probability of shock (only after warmup).
    recovery_rate : float
        Multiplicative recovery per timestep.
    firm_shock_fraction : float
        Fraction of firms affected per shocked product (< 1.0 enables rerouting).
    warmup_steps : int
        Number of initial shock-free timesteps.

    Returns
    -------
    exog_schedule : dict
        {t: {(firm, product): supply_amount, ...}, ...}
    shock_log : list of dict
        Log of shock events for debugging/analysis.
    """
    rng = np.random.RandomState(seed)

    exog_prods = sorted(
        set(prod_graph.source.values) - set(prod_graph.dest.values)
    )
    shock_supply = default_supply * (1 - shock_magnitude)

    # Recovery math
    if shock_supply > 0 and recovery_rate > 1:
        time_to_recovery = (
            (np.log(default_supply) - np.log(shock_supply)) / np.log(recovery_rate)
        )
    else:
        time_to_recovery = float("inf")

    active_steps = max(0, num_timesteps - warmup_steps)
    expected_shocks = len(exog_prods) * active_steps * shock_prob
    print(
        f"[calibrated] {len(exog_prods)} exog products, "
        f"warmup={warmup_steps}, shock_prob={shock_prob}, "
        f"expected shocks={expected_shocks:.1f}, "
        f"default_supply={default_supply}, shock_supply={shock_supply:.0f}, "
        f"firm_shock_frac={firm_shock_fraction}, "
        f"recovery ~{time_to_recovery:.1f} steps"
    )

    # Track per-product, per-firm supply state
    # Each firm can be independently shocked or not
    # prod_firm_supply[(product, firm)] = current supply level for that firm
    prod_firm_supply = {}
    for p in exog_prods:
        for f in prod2firms[p]:
            prod_firm_supply[(p, f)] = default_supply

    exog_schedule = {}
    shock_log = []

    for t in range(num_timesteps):
        exog_supp_t = {}

        for p in exog_prods:
            firms_for_p = prod2firms[p]

            # Determine if a new shock occurs for this product
            is_warmup = t < warmup_steps
            new_shock = (not is_warmup) and (rng.rand() < shock_prob)

            if new_shock:
                # Select which firms get shocked
                num_to_shock = max(1, int(round(len(firms_for_p) * firm_shock_fraction)))
                num_to_shock = min(num_to_shock, len(firms_for_p))
                shocked_firms = set(
                    rng.choice(firms_for_p, size=num_to_shock, replace=False)
                )

                shock_log.append({
                    "t": t,
                    "product": p,
                    "num_firms_total": len(firms_for_p),
                    "num_firms_shocked": len(shocked_firms),
                    "shocked_firms": list(shocked_firms),
                })

                for f in firms_for_p:
                    if f in shocked_firms:
                        prod_firm_supply[(p, f)] = shock_supply
                    # Unshocked firms keep their current level (may already be recovering)

            else:
                # Recovery: each firm recovers independently toward default_supply
                for f in firms_for_p:
                    curr = prod_firm_supply[(p, f)]
                    if curr < default_supply:
                        prod_firm_supply[(p, f)] = min(
                            default_supply, curr * recovery_rate
                        )

            # Realize supply with Poisson noise (matches original generator)
            for f in firms_for_p:
                realized = rng.poisson(prod_firm_supply[(p, f)])
                exog_supp_t[(f, p)] = int(realized)

        exog_schedule[t] = exog_supp_t

    return exog_schedule, shock_log


def create_calibrated_env(
    seed: int = 0,
    T: int = 60,
    gamma: float = 0.8,
    # Calibrated disruption params
    default_supply: float = 100,
    shock_magnitude: float = 0.7,
    shock_prob: float = 0.15,
    recovery_rate: float = 1.05,
    firm_shock_fraction: float = 0.5,
    warmup_steps: int = 10,
    # Init params
    init_inv: float = 0,
    init_supply: float = 100,
    init_demand: float = 10,
    # Policy params
    expedite_budget: float = None,
    expedite_c0: float = 1.0,
    expedite_alpha: float = 0.5,
    expedite_m_max: float = 3.0,
    # Demand scaling
    demand_multiplier: float = 1.0,
    # Graph structure params
    num_inner_layers: int = 2,
    num_per_layer: int = 10,
    min_num_suppliers: int = 2,
    max_num_suppliers: int = 3,
    # BOM params
    min_inputs: int = 1,
    max_inputs: int = 2,
    min_units: int = 1,
    max_units: int = 2,
    # Order expiry
    max_order_age: int = 10,
    # KPI warm-start
    kpi_start_step: int = None,
):
    """
    Create a SupplySimEnv with a calibrated exogenous supply schedule.

    Returns (env, obs, shock_log).
    """
    from scripts.supplysim_env import SupplySimEnv

    env = SupplySimEnv(
        seed=seed,
        T=T,
        gamma=gamma,
        expedite_budget=expedite_budget,
        expedite_c0=expedite_c0,
        expedite_alpha=expedite_alpha,
        expedite_m_max=expedite_m_max,
        num_inner_layers=num_inner_layers,
        num_per_layer=num_per_layer,
        min_num_suppliers=min_num_suppliers,
        max_num_suppliers=max_num_suppliers,
        max_order_age=max_order_age,
        kpi_start_step=kpi_start_step,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_units=min_units,
        max_units=max_units,
    )

    # Reset with shock_prob=0 to build the graph and initial conditions
    # without generating any shocks in the default schedule
    obs = env.reset(
        init_inv=init_inv,
        init_supply=init_supply,
        init_demand=init_demand,
        shock_prob=0.0,  # No shocks from default generator
        default_supply=default_supply,
    )

    # Replace with calibrated schedule
    exog_schedule, shock_log = generate_calibrated_exog_schedule(
        prod_graph=env.prod_graph,
        prod2firms=env.prod2firms,
        num_timesteps=T,
        seed=seed,
        default_supply=default_supply,
        shock_magnitude=shock_magnitude,
        shock_prob=shock_prob,
        recovery_rate=recovery_rate,
        firm_shock_fraction=firm_shock_fraction,
        warmup_steps=warmup_steps,
    )
    env.exog_schedule = exog_schedule

    # Recompute baselines with the new schedule
    env.exog_baseline_supply = env._compute_exog_baselines()

    # Scale demand schedule if multiplier != 1.0
    if demand_multiplier != 1.0 and env.demand_schedule is not None:
        for t in range(T):
            for key in env.demand_schedule[t]:
                env.demand_schedule[t][key] = max(
                    0, int(round(env.demand_schedule[t][key] * demand_multiplier))
                )

    return env, obs, shock_log
