"""
Baseline policies for comparison against the graph-informed greedy optimizer.

Each factory returns a closure with signature:
    policy_fn(obs, t, env) -> (action_dict, explanation_dict)

Action format:
    {"reroute": [(buyer, product, new_supplier), ...],
     "supply_multiplier": {(firm, product): multiplier, ...}}
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feasible_reroutes(env):
    """
    Return list of (buyer, product, current_supplier, [alternative_suppliers])
    for every (buyer, product) pair that has at least one alternative supplier.
    """
    feasible = []
    for (buyer, product), current_supplier in env.inputs2supplier.items():
        alternatives = [s for s in env.prod2firms[product] if s != current_supplier]
        if alternatives:
            feasible.append((buyer, product, current_supplier, alternatives))
    return feasible


def _count_open_orders(env, supplier, product):
    """Count the number of open order lines for (supplier, product)."""
    key = (supplier, product)
    if key in env.curr_orders:
        return len(env.curr_orders[key])
    return 0


def _count_open_order_units(env, supplier, product):
    """Sum the units of open orders for (supplier, product)."""
    key = (supplier, product)
    if key not in env.curr_orders:
        return 0.0
    return sum(float(amt) for _, amt, *_ in env.curr_orders[key])


def _get_shock_severities(env, t):
    """
    Compute shock severity for each exogenous product at timestep t.
    severity = clip(1 - current_supply / baseline, 0, 1)
    Returns dict: product -> severity (float in [0, 1]).
    """
    if env.exog_schedule is None or not env.exog_baseline_supply:
        return {}

    severities = {}
    exog_t = env.exog_schedule[t]
    for p in env.exog_prods:
        curr_supply = 0.0
        for (firm, prod), amount in exog_t.items():
            if prod == p:
                curr_supply += float(amount)
        baseline = env.exog_baseline_supply.get(p, 1.0)
        sev = max(0.0, min(1.0, 1.0 - curr_supply / baseline))
        severities[p] = sev
    return severities


def _get_exog_firms_for_product(env, product, t):
    """Return list of (firm, product) keys in the exog schedule for a given product at time t."""
    if env.exog_schedule is None:
        return []
    exog_t = env.exog_schedule[t]
    return [(firm, prod) for (firm, prod) in exog_t if prod == product]


def _count_consumer_orders_for_product(env, product):
    """Count consumer order lines for a specific consumer product."""
    count = 0
    for f in env.prod2firms.get(product, []):
        key = (f, product)
        if key in env.curr_orders:
            for buyer, _, *_ in env.curr_orders[key]:
                if buyer == "consumer":
                    count += 1
    return count


# ---------------------------------------------------------------------------
# 1. No Intervention
# ---------------------------------------------------------------------------

def no_intervention_policy(obs, t, env):
    """Returns empty action every step. No reroutes, no expedites."""
    action = {"reroute": [], "supply_multiplier": {}}
    explanation = {
        "policy_name": "no_intervention",
        "description": "Passive baseline; no actions taken.",
    }
    return action, explanation


# ---------------------------------------------------------------------------
# 2. Random Reroute
# ---------------------------------------------------------------------------

def make_random_reroute_policy(reroute_budget_K, seed=42):
    """
    Factory: randomly selects up to K reroutes from the feasible set each step.

    Parameters
    ----------
    reroute_budget_K : int
        Maximum number of reroutes per step.
    seed : int
        Seed for the policy's own RNG (does not interfere with env RNG).
    """
    rng = np.random.RandomState(seed)

    def random_reroute_policy(obs, t, env):
        feasible = _get_feasible_reroutes(env)
        if not feasible:
            action = {"reroute": [], "supply_multiplier": {}}
            explanation = {
                "policy_name": "random_reroute",
                "K": reroute_budget_K,
                "seed": seed,
                "num_feasible": 0,
                "reroutes_selected": [],
            }
            return action, explanation

        # Sample up to K feasible reroutes without replacement
        num_to_select = min(reroute_budget_K, len(feasible))
        indices = rng.choice(len(feasible), size=num_to_select, replace=False)
        reroutes = []
        reroute_details = []
        for idx in indices:
            buyer, product, current_supplier, alternatives = feasible[idx]
            new_supplier = alternatives[rng.randint(len(alternatives))]
            reroutes.append((buyer, product, new_supplier))
            reroute_details.append({
                "buyer": buyer,
                "product": product,
                "from": current_supplier,
                "to": new_supplier,
            })

        action = {"reroute": reroutes, "supply_multiplier": {}}
        explanation = {
            "policy_name": "random_reroute",
            "K": reroute_budget_K,
            "seed": seed,
            "num_feasible": len(feasible),
            "reroutes_selected": reroute_details,
        }
        return action, explanation

    return random_reroute_policy


# ---------------------------------------------------------------------------
# 3. Backlog-Only Greedy (key ablation comparator)
# ---------------------------------------------------------------------------

def make_backlog_only_policy(reroute_budget_K):
    """
    Factory: scores interventions using ONLY current backlog/order levels,
    no graph signals.

    Reroute score  = number of open orders for the current supplier's
                     (supplier, product) pair  (higher = more congested).
    Expedite score = consumer orders downstream of exogenous product / cost.

    Parameters
    ----------
    reroute_budget_K : int
        Maximum number of reroutes per step.
    """

    def backlog_only_greedy_policy(obs, t, env):
        # --- Reroutes: score by current supplier congestion ---
        feasible = _get_feasible_reroutes(env)
        scored_reroutes = []
        for buyer, product, current_supplier, alternatives in feasible:
            score = _count_open_orders(env, current_supplier, product)
            # Pick the alternative with the fewest open orders
            best_alt = min(alternatives, key=lambda s: _count_open_orders(env, s, product))
            scored_reroutes.append((score, buyer, product, current_supplier, best_alt))

        # Sort descending by congestion score, take top K
        scored_reroutes.sort(key=lambda x: -x[0])
        selected = scored_reroutes[:reroute_budget_K]
        reroutes = [(buyer, product, new_sup) for _, buyer, product, _, new_sup in selected]
        reroute_details = [
            {"buyer": buyer, "product": product, "from": cur, "to": new, "score": score}
            for score, buyer, product, cur, new in selected
        ]

        # --- Expedites: boost shocked exogenous products ---
        severities = _get_shock_severities(env, t)
        supply_multiplier = {}
        expedite_details = []

        # For each shocked exogenous product, compute a downstream consumer demand score
        for p_exog, sev in severities.items():
            if sev < 0.1:
                continue  # not significantly shocked
            # Score: count consumer orders for all consumer products
            # (simple proxy -- the backlog-only policy doesn't use graph structure
            #  to trace which consumer products are downstream)
            consumer_order_count = 0
            for cp in env.consumer_prods:
                consumer_order_count += _count_consumer_orders_for_product(env, cp)

            cost = env.expedite_cost_per_unit.get(p_exog, 1.0)
            if cost <= 0:
                score = float("inf") if consumer_order_count > 0 else 0.0
            else:
                score = consumer_order_count / cost

            # Apply multiplier of 2.0 for shocked products
            exog_keys = _get_exog_firms_for_product(env, p_exog, t)
            for key in exog_keys:
                supply_multiplier[key] = 2.0

            expedite_details.append({
                "product": p_exog,
                "severity": round(sev, 4),
                "consumer_order_count": consumer_order_count,
                "cost_per_unit": cost,
                "score": round(score, 4) if score != float("inf") else "inf",
                "multiplier": 2.0,
            })

        action = {"reroute": reroutes, "supply_multiplier": supply_multiplier}
        explanation = {
            "policy_name": "backlog_only_greedy",
            "K": reroute_budget_K,
            "num_feasible_reroutes": len(feasible),
            "reroutes_selected": reroute_details,
            "expedite_details": expedite_details,
        }
        return action, explanation

    return backlog_only_greedy_policy


# ---------------------------------------------------------------------------
# 4. Expedite-Only
# ---------------------------------------------------------------------------

def make_expedite_only_policy(severity_threshold=0.1):
    """
    Factory: only uses expediting (multiplier 2.0 for all shocked exogenous
    products with severity > threshold), no reroutes. Greedy by severity.

    Parameters
    ----------
    severity_threshold : float
        Minimum shock severity to trigger expediting (default 0.1).
    """

    def expedite_only_policy(obs, t, env):
        severities = _get_shock_severities(env, t)
        supply_multiplier = {}
        expedite_details = []

        # Sort by severity descending (greedy by severity)
        sorted_shocks = sorted(severities.items(), key=lambda x: -x[1])

        for p_exog, sev in sorted_shocks:
            if sev <= severity_threshold:
                continue
            exog_keys = _get_exog_firms_for_product(env, p_exog, t)
            for key in exog_keys:
                supply_multiplier[key] = 2.0
            expedite_details.append({
                "product": p_exog,
                "severity": round(sev, 4),
                "multiplier": 2.0,
            })

        action = {"reroute": [], "supply_multiplier": supply_multiplier}
        explanation = {
            "policy_name": "expedite_only",
            "severity_threshold": severity_threshold,
            "num_shocked": len(expedite_details),
            "expedite_details": expedite_details,
        }
        return action, explanation

    return expedite_only_policy


# ---------------------------------------------------------------------------
# 5. Reroute-Only
# ---------------------------------------------------------------------------

def make_reroute_only_policy(reroute_budget_K):
    """
    Factory: only uses rerouting, no expediting. Reroutes away from suppliers
    with the most open orders.

    Parameters
    ----------
    reroute_budget_K : int
        Maximum number of reroutes per step.
    """

    def reroute_only_policy(obs, t, env):
        feasible = _get_feasible_reroutes(env)
        scored = []
        for buyer, product, current_supplier, alternatives in feasible:
            score = _count_open_orders(env, current_supplier, product)
            # Pick the alternative with the fewest open orders
            best_alt = min(alternatives, key=lambda s: _count_open_orders(env, s, product))
            scored.append((score, buyer, product, current_supplier, best_alt))

        # Sort descending by congestion, take top K
        scored.sort(key=lambda x: -x[0])
        selected = scored[:reroute_budget_K]
        reroutes = [(buyer, product, new_sup) for _, buyer, product, _, new_sup in selected]
        reroute_details = [
            {"buyer": buyer, "product": product, "from": cur, "to": new, "score": score}
            for score, buyer, product, cur, new in selected
        ]

        action = {"reroute": reroutes, "supply_multiplier": {}}
        explanation = {
            "policy_name": "reroute_only",
            "K": reroute_budget_K,
            "num_feasible": len(feasible),
            "reroutes_selected": reroute_details,
        }
        return action, explanation

    return reroute_only_policy


# ---------------------------------------------------------------------------
# 6. Threshold Policy (Reactive Rules — Practitioner Baseline)
# ---------------------------------------------------------------------------

def make_threshold_policy(
    reroute_budget_K: int = 3,
    severity_threshold: float = 0.5,
    expedite_threshold: float = 0.3,
):
    """
    Factory: simple reactive rules mimicking a human planner.

    1. If current supplier severity > θ_severity, reroute to the least-affected alternative
    2. If exogenous product severity > θ_expedite, apply max expedite multiplier

    Parameters
    ----------
    reroute_budget_K : int
        Maximum reroutes per step.
    severity_threshold : float
        Minimum severity to trigger a reroute.
    expedite_threshold : float
        Minimum severity to trigger an expedite.
    """

    def threshold_policy(obs, t, env):
        severities = _get_shock_severities(env, t)
        reroutes = []
        reroute_details = []

        # --- Rerouting: find buyer-product pairs with severely shocked suppliers ---
        # Build supplier -> severity lookup for exog products
        # For non-exog products, use open-order congestion as a proxy for "severity"
        supplier_product_severity = {}
        for (buyer, product), current_supplier in env.inputs2supplier.items():
            if product in severities:
                supplier_product_severity[(current_supplier, product)] = severities[product]
            else:
                # Use congestion ratio as a proxy
                open_units = _count_open_order_units(env, current_supplier, product)
                f_idx = env.firm2idx.get(current_supplier)
                p_idx = env.prod2idx.get(product)
                inv = 0.0
                if f_idx is not None and p_idx is not None:
                    inv = float(env.inventories[f_idx, p_idx])
                total = open_units + inv + 1e-6
                proxy_sev = open_units / total
                supplier_product_severity[(current_supplier, product)] = proxy_sev

        for (buyer, product), current_supplier in env.inputs2supplier.items():
            if len(reroutes) >= reroute_budget_K:
                break
            sev = supplier_product_severity.get((current_supplier, product), 0.0)
            if sev <= severity_threshold:
                continue

            alternatives = [s for s in env.prod2firms.get(product, []) if s != current_supplier]
            if not alternatives:
                continue

            # Pick the first alternative with lower severity
            best_alt = None
            best_alt_sev = sev
            for alt in alternatives:
                alt_sev = supplier_product_severity.get((alt, product), 0.0)
                if alt_sev < best_alt_sev:
                    best_alt = alt
                    best_alt_sev = alt_sev

            if best_alt is not None:
                reroutes.append((buyer, product, best_alt))
                reroute_details.append({
                    "buyer": buyer,
                    "product": product,
                    "from": current_supplier,
                    "to": best_alt,
                    "severity": round(sev, 4),
                    "alt_severity": round(best_alt_sev, 4),
                })

        # --- Expediting: for shocked exogenous products above threshold ---
        supply_multiplier = {}
        expedite_details = []
        m_max = env.expedite_m_max if env.expedite_m_max is not None else 3.0

        for p_exog, sev in sorted(severities.items(), key=lambda x: -x[1]):
            if sev <= expedite_threshold:
                continue
            exog_keys = _get_exog_firms_for_product(env, p_exog, t)
            for key in exog_keys:
                supply_multiplier[key] = m_max
            expedite_details.append({
                "product": p_exog,
                "severity": round(sev, 4),
                "multiplier": m_max,
            })

        action = {"reroute": reroutes, "supply_multiplier": supply_multiplier}
        explanation = {
            "policy_name": "threshold",
            "K": reroute_budget_K,
            "severity_threshold": severity_threshold,
            "expedite_threshold": expedite_threshold,
            "reroutes_selected": reroute_details,
            "expedite_details": expedite_details,
        }
        return action, explanation

    return threshold_policy


# ---------------------------------------------------------------------------
# 7. Rolling-Horizon MIP
# ---------------------------------------------------------------------------

def make_mip_policy(reroute_budget_K: int = 3):
    """
    Factory: one-step-lookahead MIP that jointly optimizes all interventions.

    Uses the same scoring signals as graph_informed_optimizer but solves a
    small integer program instead of greedy selection.

    Parameters
    ----------
    reroute_budget_K : int
        Maximum number of reroutes per step.
    """
    from pulp import (
        LpProblem, LpMaximize, LpVariable, LpBinary,
        lpSum, PULP_CBC_CMD, LpStatus,
    )

    def mip_policy(obs, t, env):
        from scripts.graph_informed_optimizer import SignalComputer, CandidateEnumerator

        signals = SignalComputer(env)
        enumerator = CandidateEnumerator()

        reroute_candidates = enumerator.enumerate_reroutes(env, signals)
        expedite_candidates = enumerator.enumerate_expedites(env, signals)

        # Deduplicate expedites: keep best per (firm, product)
        seen_exp = set()
        deduped_expedites = []
        for cand in expedite_candidates:
            key = (cand["firm"], cand["product"])
            if key not in seen_exp:
                seen_exp.add(key)
                deduped_expedites.append(cand)

        # If no candidates at all, return empty action
        if not reroute_candidates and not deduped_expedites:
            action = {"reroute": [], "supply_multiplier": {}}
            explanation = {
                "policy_name": "mip",
                "K": reroute_budget_K,
                "status": "no_candidates",
                "reroutes_selected": [],
                "expedite_details": [],
            }
            return action, explanation

        # --- Build MIP ---
        prob = LpProblem("disruption_response", LpMaximize)

        # Reroute binary variables
        reroute_vars = {}
        reroute_scores = {}
        for i, cand in enumerate(reroute_candidates):
            var_name = f"r_{i}"
            reroute_vars[i] = LpVariable(var_name, cat=LpBinary)
            reroute_scores[i] = cand["value_score"]

        # Expedite continuous variables (units to inject)
        expedite_vars = {}
        expedite_value_per_unit = {}
        expedite_cost_per_unit = {}
        for i, cand in enumerate(deduped_expedites):
            var_name = f"e_{i}"
            max_units = cand["added_units"]
            expedite_vars[i] = LpVariable(var_name, lowBound=0, upBound=max(0, max_units))
            if cand["cost"] > 0:
                expedite_value_per_unit[i] = cand["value"] / max(cand["added_units"], 1e-9)
                expedite_cost_per_unit[i] = cand["cost"] / max(cand["added_units"], 1e-9)
            else:
                expedite_value_per_unit[i] = 0.0
                expedite_cost_per_unit[i] = 0.0

        # Objective: maximize total value
        obj_terms = []
        for i in reroute_vars:
            obj_terms.append(reroute_scores[i] * reroute_vars[i])
        for i in expedite_vars:
            obj_terms.append(expedite_value_per_unit[i] * expedite_vars[i])

        if obj_terms:
            prob += lpSum(obj_terms)

        # Constraint: at most K reroutes
        if reroute_vars:
            prob += lpSum(reroute_vars.values()) <= reroute_budget_K

        # Constraint: at most one reroute per (buyer, product)
        bp_groups = defaultdict(list)
        for i, cand in enumerate(reroute_candidates):
            bp_groups[(cand["buyer"], cand["product"])].append(i)
        for bp_key, indices in bp_groups.items():
            if len(indices) > 1:
                prob += lpSum(reroute_vars[i] for i in indices) <= 1

        # Constraint: expedite budget
        budget_remaining = env.expedite_budget_remaining
        if budget_remaining is not None and expedite_vars:
            prob += lpSum(
                expedite_cost_per_unit[i] * expedite_vars[i]
                for i in expedite_vars
            ) <= budget_remaining

        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        status = LpStatus[prob.status]

        # Extract solution
        selected_reroutes = []
        for i, cand in enumerate(reroute_candidates):
            if i in reroute_vars and reroute_vars[i].varValue is not None and reroute_vars[i].varValue > 0.5:
                selected_reroutes.append(cand)

        selected_expedites = []
        for i, cand in enumerate(deduped_expedites):
            if i in expedite_vars and expedite_vars[i].varValue is not None and expedite_vars[i].varValue > 0.01:
                # Compute the multiplier from the selected units
                units_selected = expedite_vars[i].varValue
                if cand["added_units"] > 0:
                    frac = units_selected / cand["added_units"]
                    effective_m = 1.0 + (cand["multiplier"] - 1.0) * frac
                else:
                    effective_m = cand["multiplier"]
                selected_expedites.append({**cand, "effective_multiplier": effective_m})

        # Format action
        reroute_tuples = [
            (r["buyer"], r["product"], r["new_supplier"])
            for r in selected_reroutes
        ]
        supply_multiplier = {}
        for e in selected_expedites:
            exog_keys = _get_exog_firms_for_product(env, e["product"], t)
            m = e.get("effective_multiplier", e["multiplier"])
            for key in exog_keys:
                supply_multiplier[key] = m

        action = {"reroute": reroute_tuples, "supply_multiplier": supply_multiplier}
        explanation = {
            "policy_name": "mip",
            "K": reroute_budget_K,
            "status": status,
            "num_reroute_candidates": len(reroute_candidates),
            "num_expedite_candidates": len(deduped_expedites),
            "reroutes_selected": [
                {"buyer": r["buyer"], "product": r["product"],
                 "from": r["current_supplier"], "to": r["new_supplier"],
                 "value_score": r["value_score"]}
                for r in selected_reroutes
            ],
            "expedite_details": [
                {"product": e["product"], "multiplier": e.get("effective_multiplier", e["multiplier"]),
                 "cost": e["cost"], "value": e["value"]}
                for e in selected_expedites
            ],
        }
        return action, explanation

    return mip_policy
