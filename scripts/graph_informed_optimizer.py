# scripts/graph_informed_optimizer.py
"""
Graph-informed greedy optimizer for supply chain disruption response.

At each simulation timestep this module:
1. Computes explainability signals from the live environment state
2. Enumerates all candidate interventions (reroutes + expedites)
3. Scores them using graph-informed value functions
4. Greedily selects the best feasible set under budget constraints

Usage
-----
    from scripts.graph_informed_optimizer import make_graph_informed_policy

    policy = make_graph_informed_policy(reroute_budget_K=3)
    obs = env.reset()
    done = False
    while not done:
        action, explanation = policy(obs, env.t, env)
        obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Signal computer -- runtime equivalents of the post-hoc explain_service
# ---------------------------------------------------------------------------

class SignalComputer:
    """Computes explainability signals from live env state at runtime."""

    def __init__(self, env):
        self.env = env

        # --- build networkx product DAG (static, cacheable) ---
        self._graph = nx.DiGraph()
        for row in env.prod_graph.itertuples(index=False):
            self._graph.add_edge(row.source, row.dest, units=float(row.units))

        # --- cache static graph properties ---
        self._consumer_set = set(env.consumer_prods)
        self._exog_set = set(env.exog_prods)

        # product depths (longest-path distance from any exog source)
        self._depths: Dict[str, int] = {}
        for p in env.products:
            if p not in self._graph:
                self._depths[p] = 0  # isolated product (no edges in DAG)
                continue
            preds = list(self._graph.predecessors(p))
            if len(preds) == 0:
                self._depths[p] = 0
            # filled lazily below via topological pass

        for p in nx.topological_sort(self._graph):
            preds = list(self._graph.predecessors(p))
            if len(preds) == 0:
                self._depths[p] = 0
            else:
                self._depths[p] = 1 + max(self._depths.get(pr, 0) for pr in preds)

        self._max_depth = max(self._depths.values()) if self._depths else 1

        # shortest distance from each product to the nearest consumer product
        # (reverse BFS from consumer products in reversed graph)
        rev = self._graph.reverse(copy=False)
        self._dist_to_consumer: Dict[str, int] = {}
        for cp in env.consumer_prods:
            lengths = nx.single_source_shortest_path_length(rev, cp)
            for node, d in lengths.items():
                if node not in self._dist_to_consumer or d < self._dist_to_consumer[node]:
                    self._dist_to_consumer[node] = d

        # downstream consumer coverage per product
        self._downstream_consumers: Dict[str, set] = {}
        for p in env.products:
            if p not in self._graph:
                self._downstream_consumers[p] = set()
                continue
            reachable = nx.descendants(self._graph, p)
            self._downstream_consumers[p] = reachable.intersection(self._consumer_set)

        # firm → set of distinct buyer firms (outdegree proxy)
        self._firm_buyers: Dict[str, set] = defaultdict(set)
        for (buyer, product), supplier in env.inputs2supplier.items():
            self._firm_buyers[supplier].add(buyer)

        # firm → downstream consumer coverage
        self._firm_downstream_cov: Dict[str, int] = {}
        for firm in env.firms:
            covered = set()
            for p in env.firm2prods.get(firm, []):
                covered |= self._downstream_consumers.get(p, set())
            self._firm_downstream_cov[firm] = len(covered)

        # normalisation helpers
        all_outdeg = [len(self._firm_buyers[f]) for f in env.firms]
        self._max_outdeg = max(1, max(all_outdeg)) if all_outdeg else 1

        all_downcov = [self._firm_downstream_cov.get(f, 0) for f in env.firms]
        self._max_downcov = max(1, max(all_downcov)) if all_downcov else 1

    # -- shock severity -------------------------------------------------------

    def shock_severity(self, product: str, t: int) -> float:
        """severity(p, t) = clip(1 - S(p,t) / S_bar(p), 0, 1)"""
        env = self.env
        if env.exog_schedule is None or product not in self._exog_set:
            return 0.0
        baseline = env.exog_baseline_supply.get(product, 1.0)
        if baseline <= 0:
            return 0.0
        current_supply = 0.0
        for (firm, prod), amount in env.exog_schedule[t].items():
            if prod == product:
                current_supply += float(amount)
        return float(np.clip(1.0 - current_supply / baseline, 0.0, 1.0))

    def _shocked_exog_products(self, t: int, threshold: float = 0.0) -> Dict[str, float]:
        """Return {product: severity} for all exog products with severity > threshold."""
        result = {}
        for p in self.env.exog_prods:
            s = self.shock_severity(p, t)
            if s > threshold:
                result[p] = s
        return result

    # -- ripple impact ---------------------------------------------------------

    def ripple_impact(self, product: str, t: int) -> Tuple[float, Dict[str, float]]:
        """
        Runtime approximation of ripple impact score.

        Returns (impact_score, breakdown_dict) both in [0, 1].
        """
        env = self.env
        p_idx = env.prod2idx.get(product)
        if p_idx is None:
            return 0.0, {"tx_drop": 0.0, "backlog_inc": 0.0, "shock_prox": 0.0}

        # --- tx_drop proxy: high pending orders + low inventory ≈ likely tx drop
        total_inv = 0.0
        total_pending = 0.0
        for f in env.firms:
            f_idx = env.firm2idx[f]
            total_inv += float(env.inventories[f_idx, p_idx])
            total_pending += float(env.pending[f_idx, p_idx])

        # orders waiting for this product
        total_open_orders = 0.0
        for (supplier, prod), orders in env.curr_orders.items():
            if prod == product:
                total_open_orders += sum(float(amt) for _, amt, *_ in orders)

        demand_proxy = total_open_orders + total_pending
        supply_proxy = total_inv + 1e-6
        tx_drop = float(np.clip(1.0 - supply_proxy / max(demand_proxy, 1e-6), 0.0, 1.0))

        # --- backlog_inc proxy: consumer orders still pending downstream
        downstream_consumer_backlog = 0.0
        downstream_prods = self._downstream_consumers.get(product, set()) | {product}
        for (supplier, prod), orders in env.curr_orders.items():
            if prod in downstream_prods:
                for buyer, amt, *_ in orders:
                    if buyer == "consumer":
                        downstream_consumer_backlog += float(amt)
        # normalise by total consumer backlog
        total_consumer_backlog = 0.0
        for orders in env.curr_orders.values():
            for buyer, amt, *_ in orders:
                if buyer == "consumer":
                    total_consumer_backlog += float(amt)
        backlog_inc = float(np.clip(
            downstream_consumer_backlog / max(total_consumer_backlog, 1e-6), 0.0, 1.0
        ))

        # --- shock_prox: 1 / (1 + BFS distance to nearest shocked exog product)
        shocked = self._shocked_exog_products(t)
        if len(shocked) == 0:
            shock_prox = 0.0
        else:
            min_dist = float("inf")
            for exog_p in shocked:
                if exog_p == product:
                    min_dist = 0
                    break
                try:
                    d = nx.shortest_path_length(self._graph, exog_p, product)
                    min_dist = min(min_dist, d)
                except nx.NetworkXNoPath:
                    pass
            shock_prox = 1.0 / (1.0 + min_dist) if min_dist < float("inf") else 0.0

        impact = 0.5 * tx_drop + 0.3 * backlog_inc + 0.2 * shock_prox
        breakdown = {
            "tx_drop": round(tx_drop, 6),
            "backlog_inc": round(backlog_inc, 6),
            "shock_prox": round(shock_prox, 6),
        }
        return float(np.clip(impact, 0.0, 1.0)), breakdown

    # -- chokepoint criticality ------------------------------------------------

    def chokepoint_criticality(self, firm: str, t: int) -> Tuple[float, Dict[str, float]]:
        """
        criticality(f, t) in [0, 1].

        Returns (criticality_score, breakdown_dict).
        """
        env = self.env

        # outdeg normalised
        outdeg = len(self._firm_buyers.get(firm, set()))
        outdeg_norm = outdeg / self._max_outdeg

        # downstream consumer coverage normalised
        downcov = self._firm_downstream_cov.get(firm, 0)
        downcov_norm = downcov / self._max_downcov

        # constrained flow share: open_orders / (open_orders + fulfilled + eps)
        open_units = 0.0
        for (supplier, prod), orders in env.curr_orders.items():
            if supplier == firm:
                open_units += sum(float(amt) for _, amt, *_ in orders)
        # fulfilled proxy: inventory of products this firm supplies
        fulfilled_proxy = 0.0
        f_idx = env.firm2idx.get(firm)
        if f_idx is not None:
            for p in env.firm2prods.get(firm, []):
                p_idx = env.prod2idx[p]
                fulfilled_proxy += float(env.inventories[f_idx, p_idx])
        constr = open_units / (open_units + fulfilled_proxy + 1e-6)
        constr_norm = float(np.clip(constr, 0.0, 1.0))

        criticality = 0.4 * outdeg_norm + 0.3 * downcov_norm + 0.3 * constr_norm
        breakdown = {
            "outdeg_norm": round(outdeg_norm, 6),
            "downcov_norm": round(downcov_norm, 6),
            "constr_norm": round(constr_norm, 6),
        }
        return float(np.clip(criticality, 0.0, 1.0)), breakdown

    # -- path score ------------------------------------------------------------

    def path_score(self, source_product: str, t: int) -> float:
        """path_score(π, t) = severity(source, t) / (1 + |π|) for shortest path to consumer."""
        severity = self.shock_severity(source_product, t)
        if severity <= 0:
            return 0.0
        dist = self._dist_to_consumer.get(source_product)
        if dist is None:
            return 0.0
        return float(np.clip(severity / (1.0 + dist), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Candidate enumerator
# ---------------------------------------------------------------------------

class CandidateEnumerator:
    """Enumerates all legal reroute and expedite interventions."""

    def enumerate_reroutes(
        self,
        env,
        signals: SignalComputer,
    ) -> List[Dict[str, Any]]:
        """
        For each (buyer, product) where the current supplier is degraded,
        list feasible alternative suppliers.

        Returns list of candidate dicts sorted by value_score descending.
        """
        t = env.t
        candidates: List[Dict[str, Any]] = []

        # precompute per-product shock severity and ripple impact
        severity_cache: Dict[str, float] = {}
        ripple_cache: Dict[str, Tuple[float, dict]] = {}

        for (buyer, product), current_supplier in env.inputs2supplier.items():
            # only consider rerouting away from suppliers whose products
            # have nonzero shock severity or high ripple impact
            if product not in severity_cache:
                severity_cache[product] = signals.shock_severity(product, t) if product in signals._exog_set else 0.0
            if product not in ripple_cache:
                ripple_cache[product] = signals.ripple_impact(product, t)

            sev = severity_cache[product]
            ripple_val, ripple_bd = ripple_cache[product]

            # also check chokepoint criticality of current supplier
            crit, crit_bd = signals.chokepoint_criticality(current_supplier, t)

            # skip if the current supplier is doing fine
            if sev < 0.01 and ripple_val < 0.05 and crit < 0.1:
                continue

            # enumerate alternatives
            alt_suppliers = [s for s in env.prod2firms.get(product, []) if s != current_supplier]
            if len(alt_suppliers) == 0:
                continue

            ps = signals.path_score(product, t)

            for new_supplier in alt_suppliers:
                # score the alternative: prefer less-critical new suppliers
                new_crit, new_crit_bd = signals.chokepoint_criticality(new_supplier, t)
                # severity relief: moving away from a critical, shock-exposed supplier
                severity_relief = sev * max(0.0, crit - new_crit)

                value_score = float(np.clip(
                    0.4 * severity_relief + 0.3 * ripple_val + 0.2 * crit + 0.1 * ps,
                    0.0, 1.0,
                ))

                candidates.append({
                    "buyer": buyer,
                    "product": product,
                    "current_supplier": current_supplier,
                    "new_supplier": new_supplier,
                    "value_score": round(value_score, 6),
                    "score_breakdown": {
                        "severity_relief": round(severity_relief, 6),
                        "ripple_impact": round(ripple_val, 6),
                        "current_supplier_criticality": round(crit, 6),
                        "new_supplier_criticality": round(new_crit, 6),
                        "path_score": round(ps, 6),
                    },
                })

        candidates.sort(key=lambda c: c["value_score"], reverse=True)
        return candidates

    def enumerate_expedites(
        self,
        env,
        signals: SignalComputer,
        multiplier_levels: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        For each (firm, product) in the current exog schedule with severity > 0,
        enumerate multiplier levels and compute cost/value.

        Returns list of candidate dicts sorted by value_per_cost descending.
        """
        if multiplier_levels is None:
            multiplier_levels = [1.5, 2.0]

        t = env.t
        if env.exog_schedule is None:
            return []

        # skip if budget is explicitly zero
        if env.expedite_budget_remaining is not None and env.expedite_budget_remaining <= 0:
            return []

        exog_t = env.exog_schedule[t]
        candidates: List[Dict[str, Any]] = []

        for (firm, product), base_amount in exog_t.items():
            base_amount = float(base_amount)
            if base_amount <= 0:
                continue

            sev = signals.shock_severity(product, t)
            if sev < 0.01:
                continue

            ripple_val, ripple_bd = signals.ripple_impact(product, t)
            ps = signals.path_score(product, t)
            unit_cost = float(env.expedite_cost_per_unit.get(product, env.expedite_cost_default))

            for m in multiplier_levels:
                # respect m_max
                effective_m = m
                if env.expedite_m_max is not None:
                    effective_m = min(m, env.expedite_m_max)
                if effective_m <= 1.0:
                    continue

                added_units = base_amount * (effective_m - 1.0)
                cost = added_units * unit_cost
                if cost <= 0:
                    continue

                # value signal
                value = 0.4 * sev + 0.3 * ripple_val + 0.3 * ps
                value_per_cost = value / max(cost, 1e-9)

                candidates.append({
                    "firm": firm,
                    "product": product,
                    "multiplier": round(effective_m, 4),
                    "added_units": round(added_units, 4),
                    "cost": round(cost, 4),
                    "value": round(float(np.clip(value, 0.0, 1.0)), 6),
                    "value_per_cost": round(value_per_cost, 6),
                    "score_breakdown": {
                        "severity": round(sev, 6),
                        "ripple_impact": round(ripple_val, 6),
                        "path_score": round(ps, 6),
                        "unit_cost": round(unit_cost, 6),
                    },
                })

        candidates.sort(key=lambda c: c["value_per_cost"], reverse=True)
        return candidates


# ---------------------------------------------------------------------------
# Main policy function
# ---------------------------------------------------------------------------

def graph_informed_policy(
    obs: Dict,
    t: int,
    env,
    reroute_budget_K: int = 3,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict, Dict]:
    """
    Policy function: (obs, t, env) -> (action_dict, explanation_dict).

    Parameters
    ----------
    obs : dict
        Observation returned by env (unused by this policy but kept for
        signature compatibility).
    t : int
        Current timestep.
    env : SupplySimEnv
        Live environment instance.
    reroute_budget_K : int
        Maximum number of reroutes to apply per timestep.
    weights : dict, optional
        Signal weights with keys w_severity, w_ripple, w_chokepoint, w_path.

    Returns
    -------
    action : dict
        ``{"reroute": [...], "supply_multiplier": {...}}`` passable to
        ``env.step()``.
    explanation : dict
        Human-readable breakdown of chosen interventions and signal summaries.
    """
    if weights is None:
        weights = {
            "w_severity": 0.4,
            "w_ripple": 0.3,
            "w_chokepoint": 0.2,
            "w_path": 0.1,
        }

    signals = SignalComputer(env)
    enumerator = CandidateEnumerator()

    # ---- 1. Enumerate candidates ------------------------------------------
    reroute_candidates = enumerator.enumerate_reroutes(env, signals)
    expedite_candidates = enumerator.enumerate_expedites(env, signals)

    # ---- 2. Greedy selection under budget ----------------------------------

    # -- reroutes: pick top-K by value_score (free, no budget cost) --
    selected_reroutes: List[Dict] = []
    # deduplicate: only one reroute per (buyer, product)
    seen_buyer_product = set()
    for cand in reroute_candidates:
        key = (cand["buyer"], cand["product"])
        if key in seen_buyer_product:
            continue
        seen_buyer_product.add(key)
        selected_reroutes.append(cand)
        if len(selected_reroutes) >= reroute_budget_K:
            break

    # -- expedites: pick by value_per_cost, submit all and let env downscale --
    # The env proportionally downscales expedites when total cost exceeds
    # remaining budget, so we don't need to pre-filter by cost.
    selected_expedites: List[Dict] = []
    budget_left = env.expedite_budget_remaining  # may be None (unlimited)

    if budget_left is None or budget_left > 0:
        # deduplicate: only one multiplier per (firm, product); pick best
        seen_firm_product = set()
        for cand in expedite_candidates:
            key = (cand["firm"], cand["product"])
            if key in seen_firm_product:
                continue
            seen_firm_product.add(key)
            selected_expedites.append(cand)

    # ---- 3. Format action dict --------------------------------------------
    reroute_tuples = [
        (r["buyer"], r["product"], r["new_supplier"])
        for r in selected_reroutes
    ]
    supply_multiplier = {
        (e["firm"], e["product"]): e["multiplier"]
        for e in selected_expedites
    }

    action: Dict[str, Any] = {
        "reroute": reroute_tuples,
        "supply_multiplier": supply_multiplier,
    }

    # ---- 4. Build explanation ---------------------------------------------
    # summarise shock landscape
    shocked = signals._shocked_exog_products(t)
    shock_summary = {
        "num_shocked": len(shocked),
        "products": {p: round(s, 4) for p, s in sorted(shocked.items(), key=lambda x: -x[1])},
    }

    explanation: Dict[str, Any] = {
        "t": t,
        "reroutes": [
            {
                "buyer": r["buyer"],
                "product": r["product"],
                "from": r["current_supplier"],
                "to": r["new_supplier"],
                "value_score": r["value_score"],
                "breakdown": r["score_breakdown"],
            }
            for r in selected_reroutes
        ],
        "expedites": [
            {
                "firm": e["firm"],
                "product": e["product"],
                "multiplier": e["multiplier"],
                "cost": e["cost"],
                "value_per_cost": e["value_per_cost"],
                "breakdown": e["score_breakdown"],
            }
            for e in selected_expedites
        ],
        "signals_summary": {
            "shock": shock_summary,
            "weights_used": weights,
            "reroute_candidates_total": len(reroute_candidates),
            "expedite_candidates_total": len(expedite_candidates),
            "reroutes_selected": len(selected_reroutes),
            "expedites_selected": len(selected_expedites),
        },
    }

    return action, explanation


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_graph_informed_policy(
    reroute_budget_K: int = 3,
    weights: Optional[Dict[str, float]] = None,
):
    """
    Return a closure ``policy_fn(obs, t, env) -> (action, explanation)``
    with the given hyper-parameters baked in.

    Usage::

        policy = make_graph_informed_policy(reroute_budget_K=5)
        obs = env.reset()
        done = False
        while not done:
            action, explanation = policy(obs, env.t, env)
            obs, reward, done, info = env.step(action)
    """

    def _policy(obs, t, env):
        return graph_informed_policy(
            obs, t, env,
            reroute_budget_K=reroute_budget_K,
            weights=weights,
        )

    return _policy


# ---------------------------------------------------------------------------
# Quick smoke test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from scripts.supplysim_env import SupplySimEnv

    env = SupplySimEnv(
        seed=0,
        T=30,
        gamma=0.8,
        log_kpis=False,
        expedite_budget=500.0,
        expedite_m_max=3.0,
    )
    obs = env.reset(shock_prob=0.05)

    policy = make_graph_informed_policy(reroute_budget_K=3)

    print("=" * 90)
    print("Graph-informed policy smoke test")
    print("=" * 90)

    done = False
    while not done:
        action, explanation = policy(obs, env.t, env)
        obs, reward, done, info = env.step(action)
        k = info["kpis"]
        n_rr = len(explanation["reroutes"])
        n_ex = len(explanation["expedites"])
        print(
            f"t={k['t']:03d}  txns={k['transactions']:4d}  "
            f"open={k['open_orders']:5d}  "
            f"backlog_u={k['consumer_backlog_units']:10.1f}  "
            f"fill_cum={k['consumer_cumulative_fill_rate']:.3f}  "
            f"shock={k['shock_exposure']:.2f}  "
            f"reroutes={n_rr}  expedites={n_ex}  "
            f"exp_cost_t={k['expedite_cost_t']:.1f}  "
            f"budget_left={k['expedite_budget_remaining']}"
        )

    print("\nDone.")
