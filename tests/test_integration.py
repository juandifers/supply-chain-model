"""Integration tests for the full experiment pipeline."""
import os
import sys

import numpy as np
import pytest

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
from scripts.run_experiments import run_policy


ALL_POLICIES = {
    "no_intervention": no_intervention_policy,
    "random_reroute": make_random_reroute_policy(reroute_budget_K=3, seed=999),
    "backlog_only": make_backlog_only_policy(reroute_budget_K=3),
    "expedite_only": make_expedite_only_policy(),
    "reroute_only": make_reroute_only_policy(reroute_budget_K=3),
    "graph_informed": make_graph_informed_policy(reroute_budget_K=3),
}

EXPECTED_KPI_FIELDS = [
    "t", "transactions", "transaction_units", "open_orders",
    "consumer_backlog_units", "consumer_cumulative_fill_rate",
    "reroutes_applied", "reroutes_cumulative",
    "expedite_cost_t", "expedite_cost_cum",
    "shock_exposure", "worst_shocked_product",
]


class TestFullEpisode:
    def test_graph_informed_full_episode(self):
        """Run graph-informed policy for full episode without crashes."""
        result = run_policy(
            policy_fn=make_graph_informed_policy(reroute_budget_K=3),
            policy_name="graph_informed",
            seed=0, T=30, shock_prob=0.02, expedite_budget=200.0,
        )
        assert result["backlog_auc"] > 0
        assert len(result["kpi_history"]) == 30
        assert len(result["action_log"]) == 30

    def test_all_policies_complete(self):
        """All policies run to completion on same scenario."""
        for name, policy_fn in ALL_POLICIES.items():
            result = run_policy(
                policy_fn=policy_fn, policy_name=name,
                seed=0, T=20, shock_prob=0.02, expedite_budget=200.0,
            )
            assert len(result["kpi_history"]) > 0, f"{name} produced no KPIs"
            assert result["backlog_auc"] >= 0, f"{name} has negative backlog_auc"


class TestKPILogging:
    def test_kpi_fields_present(self):
        result = run_policy(
            policy_fn=no_intervention_policy,
            policy_name="no_intervention",
            seed=0, T=10, shock_prob=0.01,
        )
        kpi_df = result["kpi_history"]
        for field in EXPECTED_KPI_FIELDS:
            assert field in kpi_df.columns, f"Missing KPI field: {field}"


class TestBudgetFeasibility:
    @pytest.mark.parametrize("policy_name,policy_fn", [
        ("backlog_only", make_backlog_only_policy(reroute_budget_K=3)),
        ("graph_informed", make_graph_informed_policy(reroute_budget_K=3)),
        ("expedite_only", make_expedite_only_policy()),
    ])
    def test_budget_not_exceeded(self, policy_name, policy_fn):
        """Verify expedite budget constraints hold."""
        budget = 100.0
        for seed in range(3):
            result = run_policy(
                policy_fn=policy_fn, policy_name=policy_name,
                seed=seed, T=30, shock_prob=0.03,
                expedite_budget=budget,
            )
            final_spend = result["total_expedite_spend"]
            assert final_spend <= budget + 0.01, (
                f"{policy_name} seed={seed}: spend={final_spend} > budget={budget}"
            )

    @pytest.mark.parametrize("policy_name,policy_fn", [
        ("reroute_only", make_reroute_only_policy(reroute_budget_K=3)),
        ("graph_informed", make_graph_informed_policy(reroute_budget_K=3)),
        ("backlog_only", make_backlog_only_policy(reroute_budget_K=3)),
    ])
    def test_reroute_budget_per_step(self, policy_name, policy_fn):
        """Verify reroute count per step never exceeds K."""
        K = 3
        env = SupplySimEnv(seed=0, T=20, gamma=0.8, expedite_budget=200.0)
        obs = env.reset(shock_prob=0.02)
        done = False
        while not done:
            action, _ = policy_fn(obs, env.t, env)
            assert len(action.get("reroute", [])) <= K, (
                f"{policy_name} t={env.t}: {len(action['reroute'])} reroutes > K={K}"
            )
            obs, _, done, info = env.step(action)
