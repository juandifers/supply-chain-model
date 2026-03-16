# scripts/supplysim_env.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TGB.modules import synthetic_data as sd

class SupplySimEnv:
    """
    Step-by-step wrapper around synthetic_data.py.
    Observation is intentionally simple at first.
    Action hook lets you intervene before a timestep (reroute, cap supply, etc.).
    """

    def __init__(
        self,
        seed=0,
        T=100,
        gamma=0.8,
        num_decimals=5,
        log_kpis=False,
        expedite_budget=None,
        expedite_c0=1.0,
        expedite_alpha=0.5,
        expedite_m_max=3.0,
        expedite_cost_per_unit=None,
        expedite_cost_default=1.0,
        expedite_default_cost=None,
    ):
        self.seed = seed
        self.T = T
        self.gamma = gamma
        self.num_decimals = num_decimals
        self.log_kpis = log_kpis
        self.rng = np.random.RandomState(seed)
        self.expedite_budget = None if expedite_budget is None else float(expedite_budget)
        self.expedite_budget_remaining = self.expedite_budget
        self.expedite_c0 = float(expedite_c0)
        self.expedite_alpha = float(expedite_alpha)
        self.expedite_m_max = None if expedite_m_max is None else float(expedite_m_max)
        self._expedite_cost_overrides = dict(expedite_cost_per_unit or {})
        if expedite_default_cost is not None:
            expedite_cost_default = expedite_default_cost
        self.expedite_cost_default = float(expedite_cost_default)
        self.expedite_cost_per_unit = {}
        self.expedite_cost_t = 0.0
        self.expedite_units_added_t = 0.0
        self.expedite_cost_cum = 0.0

        # filled in reset()
        self.firms = None
        self.products = None
        self.prod_graph = None
        self.firm2prods = None
        self.prod2firms = None
        self.inputs2supplier = None
        self.firm2idx = None
        self.prod2idx = None
        self.prod_mat = None

        self.inventories = None
        self.curr_orders = None
        self.exog_supp = None
        self.pending = None

        self.demand_schedule = None
        self.exog_schedule = None
        self.consumer_prods = None
        self.exog_prods = None
        self.exog_baseline_supply = None

        self.t = 0
        self.kpi_history = []
        self.last_kpis = {}
        self.consumer_demand_cum = 0.0
        self.consumer_fulfilled_cum = 0.0
        self.reroutes_cum = 0

    def reset(self, init_inv=0, init_supply=100, init_demand=1,
              use_demand_schedule=True, use_exog_schedule=True,
              shock_prob=0.001):
        # static graphs
        (self.firms, self.products, self.prod_graph,
         self.firm2prods, self.prod2firms, self.inputs2supplier) = sd.generate_static_graphs(seed=self.seed)

        self.firm2idx = {f: i for i, f in enumerate(self.firms)}
        self.prod2idx = {p: i for i, p in enumerate(self.products)}
        self.prod_mat = sd.get_prod_mat(self.prod_graph, self.prod2idx)

        # initial conditions
        self.inventories, self.curr_orders, self.exog_supp = sd.generate_initial_conditions(
            firms=self.firms,
            products=self.products,
            prod_graph=self.prod_graph,
            prod2firms=self.prod2firms,
            init_inv=init_inv,
            init_supply=init_supply,
            init_demand=init_demand,
        )
        self.pending = np.zeros_like(self.inventories)

        # schedules
        self.demand_schedule = None
        if use_demand_schedule:
            self.demand_schedule = sd.generate_demand_schedule(
                num_timesteps=self.T,
                prod_graph=self.prod_graph,
                prod2firms=self.prod2firms,
                seed=self.seed,
            )

        self.exog_schedule = None
        if use_exog_schedule:
            self.exog_schedule = sd.generate_exog_schedule_with_shocks(
                num_timesteps=self.T,
                prod_graph=self.prod_graph,
                prod2firms=self.prod2firms,
                seed=self.seed,
                shock_prob=shock_prob,
            )

        self.consumer_prods = sorted(set(self.prod_graph.dest.values) - set(self.prod_graph.source.values))
        self.exog_prods = sorted(set(self.prod_graph.source.values) - set(self.prod_graph.dest.values))
        self.exog_baseline_supply = self._compute_exog_baselines()
        self.expedite_cost_per_unit = self._compute_expedite_cost_per_unit()

        self.t = 0
        self.kpi_history = []
        self.last_kpis = {}
        self.reroutes_cum = 0
        self.expedite_budget_remaining = self.expedite_budget
        self.expedite_cost_t = 0.0
        self.expedite_units_added_t = 0.0
        self.expedite_cost_cum = 0.0
        _, init_consumer_backlog_units = self._consumer_order_stats()
        self.consumer_demand_cum = init_consumer_backlog_units
        self.consumer_fulfilled_cum = 0.0
        return self._obs()

    def _obs(self):
        # Start simple: you can enrich later (KPIs, graph embeddings, etc.)
        return {
            "t": self.t,
            "inventories": self.inventories,
            "pending": self.pending,
            "num_open_orders": sum(len(v) for v in self.curr_orders.values()),
            "last_kpis": self.last_kpis,
        }

    def _consumer_order_stats(self):
        consumer_lines = 0
        consumer_units = 0.0
        for orders in self.curr_orders.values():
            for buyer, amount in orders:
                if buyer == "consumer":
                    consumer_lines += 1
                    consumer_units += float(amount)
        return consumer_lines, np.round(consumer_units, self.num_decimals)

    def _all_order_stats(self):
        order_lines = 0
        order_units = 0.0
        for orders in self.curr_orders.values():
            order_lines += len(orders)
            for _, amount in orders:
                order_units += float(amount)
        return order_lines, np.round(order_units, self.num_decimals)

    def _compute_exog_baselines(self):
        if self.exog_schedule is None:
            return {}
        baselines = {}
        for p in self.exog_prods:
            supply_ts = []
            for t in range(self.T):
                total_supply_p = 0.0
                for (firm, prod), amount in self.exog_schedule[t].items():
                    if prod == p:
                        total_supply_p += float(amount)
                supply_ts.append(total_supply_p)
            baselines[p] = max(1.0, float(np.percentile(supply_ts, 90)))
        return baselines

    def _compute_product_depths(self):
        products = list(self.products)
        predecessors = {p: [] for p in products}
        for row in self.prod_graph.itertuples(index=False):
            predecessors[row.dest].append(row.source)

        state = {p: 0 for p in products}
        depth = {}

        def _dfs(product):
            if state[product] == 2:
                return depth[product]
            if state[product] == 1:
                raise ValueError("Product graph contains a cycle; expected DAG for depth computation")
            state[product] = 1
            preds = predecessors.get(product, [])
            if len(preds) == 0:
                value = 0
            else:
                value = 1 + max(_dfs(parent) for parent in preds)
            depth[product] = int(value)
            state[product] = 2
            return depth[product]

        for p in products:
            if state[p] == 0:
                _dfs(p)
        return depth

    def _compute_expedite_cost_per_unit(self):
        depths = self._compute_product_depths()
        costs = {}
        for p in self.products:
            tier_cost = self.expedite_c0 * (1.0 + self.expedite_alpha * float(depths.get(p, 0)))
            costs[p] = float(np.round(max(0.0, tier_cost), self.num_decimals))
        for p, override in self._expedite_cost_overrides.items():
            costs[p] = float(np.round(max(0.0, float(override)), self.num_decimals))
        return costs

    def _get_shock_stats(self, t):
        if self.exog_schedule is None or len(self.exog_baseline_supply) == 0:
            return {
                "shock_exposure": 0.0,
                "active_exogenous_shocks": 0,
                "worst_shocked_product": None,
                "worst_shock_severity": 0.0,
            }

        severity = {}
        for p in self.exog_prods:
            curr_supply_p = 0.0
            for (firm, prod), amount in self.exog_schedule[t].items():
                if prod == p:
                    curr_supply_p += float(amount)
            baseline = self.exog_baseline_supply[p]
            severity_p = max(0.0, min(1.0, 1.0 - curr_supply_p / baseline))
            severity[p] = severity_p

        worst_product = max(severity, key=severity.get)
        worst_severity = severity[worst_product]
        active_shocks = int(np.sum([1 if v >= 0.5 else 0 for v in severity.values()]))
        return {
            "shock_exposure": float(np.mean(list(severity.values()))),
            "active_exogenous_shocks": active_shocks,
            "worst_shocked_product": worst_product,
            "worst_shock_severity": float(worst_severity),
        }

    def _print_kpi_dashboard_line(self, kpis):
        worst_prod = kpis["worst_shocked_product"] if kpis["worst_shocked_product"] is not None else "-"
        print(
            f"[t={kpis['t']:03d}] txns={kpis['transactions']:4d} "
            f"open={kpis['open_orders']:5d} backlog_u={kpis['consumer_backlog_units']:10.1f} "
            f"fill(new/cum)={kpis['consumer_new_demand_fill_rate']:.2f}/{kpis['consumer_cumulative_fill_rate']:.2f} "
            f"reroutes={kpis['reroutes_applied']:3d} "
            f"shock={kpis['shock_exposure']:.2f} active={kpis['active_exogenous_shocks']} "
            f"worst={worst_prod}:{kpis['worst_shock_severity']:.2f}"
        )

    def get_kpi_history(self):
        if len(self.kpi_history) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.kpi_history).copy()

    def step(self, action=None, debug=False, log_kpis=None):
        """
        action: optional callable(env) or dict specifying interventions.
        returns: obs, reward, done, info
        """
        if self.t >= self.T:
            return self._obs(), 0.0, True, {"transactions": pd.DataFrame()}

        should_log_kpis = self.log_kpis if log_kpis is None else bool(log_kpis)

        # --- 0) apply action hook BEFORE the timestep evolves ---
        # Keep this flexible: you can pass a function or a dict.
        reroutes_applied = 0
        self.expedite_cost_t = 0.0
        self.expedite_units_added_t = 0.0
        if action is not None:
            prev_supplier_map = dict(self.inputs2supplier)
            if callable(action):
                action(self)
            elif isinstance(action, dict):
                self._apply_action_dict(action)
            changed_keys = set(prev_supplier_map.keys()).union(set(self.inputs2supplier.keys()))
            reroutes_applied = np.sum([1 if prev_supplier_map.get(k) != self.inputs2supplier.get(k) else 0
                                       for k in changed_keys])
            self.reroutes_cum += int(reroutes_applied)

        consumer_backlog_orders_start, consumer_backlog_units_start = self._consumer_order_stats()

        # --- 1) add new demand from consumers ---
        demand_added_units = 0.0
        demand_added_orders = 0
        if self.demand_schedule is not None:
            demand_t = self.demand_schedule[self.t]
            for p in self.consumer_prods:
                for f in self.prod2firms[p]:
                    demand_val = float(demand_t[(f, p)])
                    self.curr_orders[(f, p)].append(("consumer", demand_val))
                    demand_added_units += demand_val
                    demand_added_orders += 1
        self.consumer_demand_cum += demand_added_units

        consumer_backlog_orders_after_demand, consumer_backlog_units_after_demand = self._consumer_order_stats()

        # --- 2) simulate firms in random order ---
        transactions_t = []
        all_inputs_needed = np.zeros_like(self.inventories)

        firm_order = np.array(self.firms)[self.rng.choice(len(self.firms), replace=False, size=len(self.firms))]

        for f in firm_order:
            exog_supp_t = self.exog_supp if self.exog_schedule is None else self.exog_schedule[self.t]
            self.inventories, self.curr_orders, txns_completed, inputs_needed = sd.simulate_actions_for_firm(
                f,
                self.inventories,
                self.curr_orders,
                exog_supp_t,
                self.firms,
                self.products,
                self.firm2idx,
                self.prod2idx,
                self.prod_mat,
                self.firm2prods,
                self.prod2firms,
                self.inputs2supplier,
                debug=debug,
            )
            transactions_t += txns_completed
            all_inputs_needed[self.firm2idx[f]] = inputs_needed

        # --- 3) update inventories and pending based on completed transactions ---
        txns_df = pd.DataFrame(columns=["supplier_id", "buyer_id", "product_id", "amount", "time"])
        if len(transactions_t) > 0:
            s_idxs, b_idxs, p_idxs, amts = list(zip(*transactions_t))
            buyer_product_mat = csr_matrix((amts, (b_idxs, p_idxs)),
                                           shape=(len(self.firms), len(self.products))).toarray()

            self.inventories = np.round(self.inventories + buyer_product_mat, self.num_decimals)
            self.pending = np.round(self.pending - buyer_product_mat, self.num_decimals)

            txns_df = pd.DataFrame(transactions_t, columns=["supplier_id", "buyer_id", "product_id", "amount"])
            txns_df["time"] = self.t
            txns_df = txns_df.sample(replace=False, frac=1.0, random_state=self.seed)
            txns_units = float(np.sum(amts))
        else:
            txns_units = 0.0

        # --- 4) place new orders based on unmet inputs (needs - inv - pending) ---
        all_inputs_needed = np.clip(all_inputs_needed - self.inventories - self.pending, 0, None)
        all_inputs_needed = np.round(all_inputs_needed, self.num_decimals)

        for f in firm_order:
            f_idx = self.firm2idx[f]
            inputs_needed = all_inputs_needed[f_idx]
            for p_idx in inputs_needed.nonzero()[0]:
                p = self.products[p_idx]
                suppliers = self.prod2firms[p]

                if self.gamma < 1:
                    uni_prob = (1 - self.gamma) / len(suppliers)
                    probs = [self.gamma + uni_prob if s == self.inputs2supplier[(f, p)] else uni_prob
                             for s in suppliers]
                    s = self.rng.choice(suppliers, p=np.array(probs) / np.sum(probs))
                else:
                    s = self.inputs2supplier[(f, p)]

                self.curr_orders[(s, p)].append((f, inputs_needed[p_idx]))
                self.pending[f_idx, p_idx] += inputs_needed[p_idx]

        num_orders, open_order_units = self._all_order_stats()

        consumer_backlog_orders_end, consumer_backlog_units_end = self._consumer_order_stats()
        consumer_fulfilled_units = np.round(
            max(0.0, consumer_backlog_units_after_demand - consumer_backlog_units_end),
            self.num_decimals,
        )
        consumer_fulfilled_orders = int(max(0, consumer_backlog_orders_after_demand - consumer_backlog_orders_end))
        self.consumer_fulfilled_cum += consumer_fulfilled_units

        if demand_added_units > 0:
            consumer_new_demand_fill_rate = min(1.0, consumer_fulfilled_units / demand_added_units)
        else:
            consumer_new_demand_fill_rate = 1.0
        if consumer_backlog_units_after_demand > 0:
            consumer_queue_clearance_rate = min(1.0, consumer_fulfilled_units / consumer_backlog_units_after_demand)
        else:
            consumer_queue_clearance_rate = 1.0
        if self.consumer_demand_cum > 0:
            consumer_cumulative_fill_rate = min(1.0, self.consumer_fulfilled_cum / self.consumer_demand_cum)
        else:
            consumer_cumulative_fill_rate = 1.0

        shock_stats = self._get_shock_stats(self.t)

        kpis = {
            "t": self.t,
            "transactions": int(len(transactions_t)),
            "transaction_units": float(np.round(txns_units, self.num_decimals)),
            "open_orders": int(num_orders),
            "open_order_units": float(open_order_units),
            "consumer_demand_added_orders": int(demand_added_orders),
            "consumer_demand_added_units": float(np.round(demand_added_units, self.num_decimals)),
            "consumer_fulfilled_orders": consumer_fulfilled_orders,
            "consumer_fulfilled_units": float(consumer_fulfilled_units),
            "consumer_backlog_orders": int(consumer_backlog_orders_end),
            "consumer_backlog_units": float(consumer_backlog_units_end),
            "consumer_backlog_delta_units": float(np.round(consumer_backlog_units_end - consumer_backlog_units_start, self.num_decimals)),
            "consumer_new_demand_fill_rate": float(np.round(consumer_new_demand_fill_rate, 6)),
            "consumer_queue_clearance_rate": float(np.round(consumer_queue_clearance_rate, 6)),
            "consumer_cumulative_fill_rate": float(np.round(consumer_cumulative_fill_rate, 6)),
            "reroutes_applied": int(reroutes_applied),
            "reroutes_cumulative": int(self.reroutes_cum),
            "expedite_cost_t": float(np.round(self.expedite_cost_t, self.num_decimals)),
            "expedite_cost_cum": float(np.round(self.expedite_cost_cum, self.num_decimals)),
            "expedite_units_added_t": float(np.round(self.expedite_units_added_t, self.num_decimals)),
            "expedite_budget_remaining": (
                None
                if self.expedite_budget_remaining is None
                else float(np.round(self.expedite_budget_remaining, self.num_decimals))
            ),
            "shock_exposure": float(np.round(shock_stats["shock_exposure"], 6)),
            "active_exogenous_shocks": int(shock_stats["active_exogenous_shocks"]),
            "worst_shocked_product": shock_stats["worst_shocked_product"],
            "worst_shock_severity": float(np.round(shock_stats["worst_shock_severity"], 6)),
        }
        self.kpi_history.append(kpis)
        self.last_kpis = kpis
        if should_log_kpis:
            self._print_kpi_dashboard_line(kpis)

        # --- reward: placeholder (you’ll define KPIs) ---
        # e.g. penalize open orders or stockouts
        reward = -float(num_orders)

        self.t += 1
        done = (self.t >= self.T) or (num_orders == 0)

        info = {
            "transactions": txns_df,
            "num_orders": num_orders,
            "kpis": kpis,
        }
        return self._obs(), reward, done, info

    def _apply_action_dict(self, action):
        """
        Example supported actions (easy to extend):
        - {"reroute": [("buyerFirmName","productX","newSupplierFirmName"), ...]}
        - {"supply_multiplier": {(firm, product): 0.5, ...}} applied to exog_schedule at current t
        """
        if "reroute" in action:
            for buyer, product, new_supplier in action["reroute"]:
                self.inputs2supplier[(buyer, product)] = new_supplier

        if "supply_multiplier" in action and self.exog_schedule is not None:
            mults = action["supply_multiplier"]
            exog_t = self.exog_schedule[self.t]
            reduction_plans = []
            expedite_plans = []

            for (firm, product), m in mults.items():
                key = (firm, product)
                if key not in exog_t:
                    continue
                base_supply = int(exog_t[key])
                m = float(m)
                if self.expedite_m_max is not None:
                    m = min(m, self.expedite_m_max)
                m = max(0.0, m)
                requested_supply = max(0, int(base_supply * m))
                requested_delta = int(requested_supply - base_supply)
                unit_cost = float(self.expedite_cost_per_unit.get(product, self.expedite_cost_default))
                unit_cost = max(0.0, unit_cost)
                if requested_delta <= 0:
                    reduction_plans.append(
                        {
                            "key": key,
                            "requested_supply": requested_supply,
                        }
                    )
                else:
                    requested_cost = float(requested_delta * unit_cost)
                    expedite_plans.append(
                        {
                            "key": key,
                            "base_supply": base_supply,
                            "requested_supply": requested_supply,
                            "requested_delta": requested_delta,
                            "unit_cost": unit_cost,
                            "requested_cost": requested_cost,
                        }
                    )

            for plan in reduction_plans:
                exog_t[plan["key"]] = int(plan["requested_supply"])

            positive_cost_plans = [p for p in expedite_plans if p["unit_cost"] > 0]
            requested_total_cost = float(np.sum([p["requested_cost"] for p in positive_cost_plans]))
            remaining_budget = None if self.expedite_budget_remaining is None else max(0.0, float(self.expedite_budget_remaining))
            alpha_scale = 1.0
            if remaining_budget is not None and requested_total_cost > remaining_budget and requested_total_cost > 0:
                alpha_scale = remaining_budget / requested_total_cost

            step_cost = 0.0
            step_units = 0.0

            for plan in expedite_plans:
                base_supply = plan["base_supply"]
                requested_supply = plan["requested_supply"]
                requested_delta = plan["requested_delta"]
                unit_cost = plan["unit_cost"]
                requested_cost = plan["requested_cost"]

                if unit_cost == 0:
                    # Free expedite entries bypass the budget cap by design.
                    applied_supply = requested_supply
                    applied_cost = 0.0
                    applied_delta = requested_delta
                elif alpha_scale < 1.0:
                    budget_i = alpha_scale * requested_cost
                    applied_delta = int(np.floor(budget_i / unit_cost))
                    applied_delta = int(max(0, min(requested_delta, applied_delta)))
                    applied_supply = base_supply + applied_delta
                    applied_cost = float(applied_delta * unit_cost)
                else:
                    applied_supply = requested_supply
                    applied_cost = requested_cost
                    applied_delta = requested_delta

                exog_t[plan["key"]] = int(max(0, applied_supply))
                if applied_delta > 0:
                    step_units += float(applied_delta)
                    step_cost += float(applied_cost)

            self.expedite_units_added_t += float(np.round(step_units, self.num_decimals))
            self.expedite_cost_t += float(np.round(step_cost, self.num_decimals))
            self.expedite_cost_cum += float(np.round(step_cost, self.num_decimals))

            if self.expedite_budget_remaining is not None:
                self.expedite_budget_remaining = max(0.0, float(self.expedite_budget_remaining - step_cost))
