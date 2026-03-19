"""Unit tests for the graph-informed optimizer."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv
from scripts.graph_informed_optimizer import (
    SignalComputer,
    CandidateEnumerator,
    graph_informed_policy,
    make_graph_informed_policy,
)
from scripts.baseline_policies import (
    no_intervention_policy,
    make_random_reroute_policy,
    make_backlog_only_policy,
    make_expedite_only_policy,
    make_reroute_only_policy,
)


class TestSignalComputation:
    def test_severity_in_range(self, env_with_shocks):
        signals = SignalComputer(env_with_shocks)
        for p in env_with_shocks.exog_prods:
            sev = signals.shock_severity(p, 0)
            assert 0.0 <= sev <= 1.0, f"severity({p})={sev} out of range"

    def test_severity_non_exog_is_zero(self, env_with_shocks):
        signals = SignalComputer(env_with_shocks)
        non_exog = [p for p in env_with_shocks.products if p not in env_with_shocks.exog_prods]
        for p in non_exog[:5]:
            assert signals.shock_severity(p, 0) == 0.0

    def test_ripple_impact_in_range(self, env_with_shocks):
        signals = SignalComputer(env_with_shocks)
        for p in env_with_shocks.products[:10]:
            score, breakdown = signals.ripple_impact(p, 0)
            assert 0.0 <= score <= 1.0, f"ripple({p})={score}"
            for k, v in breakdown.items():
                assert 0.0 <= v <= 1.0, f"ripple breakdown {k}={v}"

    def test_criticality_in_range(self, env_with_shocks):
        signals = SignalComputer(env_with_shocks)
        for f in env_with_shocks.firms[:10]:
            crit, breakdown = signals.chokepoint_criticality(f, 0)
            assert 0.0 <= crit <= 1.0, f"criticality({f})={crit}"

    def test_path_score_in_range(self, env_with_shocks):
        signals = SignalComputer(env_with_shocks)
        for p in env_with_shocks.exog_prods:
            ps = signals.path_score(p, 0)
            assert 0.0 <= ps <= 1.0, f"path_score({p})={ps}"


class TestGreedySelection:
    def test_respects_reroute_budget(self, env_with_shocks):
        for K in [1, 2, 5]:
            obs = {"t": 0, "inventories": env_with_shocks.inventories,
                   "pending": env_with_shocks.pending, "num_open_orders": 0, "last_kpis": {}}
            action, _ = graph_informed_policy(obs, 0, env_with_shocks, reroute_budget_K=K)
            assert len(action["reroute"]) <= K, f"Got {len(action['reroute'])} reroutes with K={K}"

    def test_respects_expedite_budget(self):
        """Cumulative expedite spend never exceeds budget (env handles clamping)."""
        env = SupplySimEnv(seed=0, T=30, gamma=0.8, expedite_budget=100.0,
                           expedite_c0=1.0, expedite_alpha=0.5, expedite_m_max=3.0)
        obs = env.reset(shock_prob=0.05)
        policy = make_graph_informed_policy(reroute_budget_K=3)

        done = False
        while not done:
            action, _ = policy(obs, env.t, env)
            obs, _, done, info = env.step(action)

        assert info["kpis"]["expedite_cost_cum"] <= 100.0 + 0.01

    def test_no_intervention_baseline(self, env_with_shocks):
        action, explanation = no_intervention_policy(
            {"t": 0}, 0, env_with_shocks
        )
        assert action["reroute"] == []
        assert action["supply_multiplier"] == {}

    def test_action_format_matches_env(self, env_with_shocks):
        """Actions produced by optimizer are accepted by env.step() without error."""
        obs = {"t": 0, "inventories": env_with_shocks.inventories,
               "pending": env_with_shocks.pending, "num_open_orders": 0, "last_kpis": {}}
        action, _ = graph_informed_policy(obs, 0, env_with_shocks, reroute_budget_K=3)
        # Should not raise
        obs2, reward, done, info = env_with_shocks.step(action)
        assert "kpis" in info
        assert isinstance(reward, float)


class TestDeterminism:
    def test_deterministic_with_seed(self):
        """Same seed + same policy = same KPI trajectory."""
        results = []
        for _ in range(2):
            env = SupplySimEnv(seed=42, T=15, gamma=0.8, expedite_budget=200.0)
            obs = env.reset(shock_prob=0.02)
            policy = make_graph_informed_policy(reroute_budget_K=3)

            done = False
            while not done:
                action, _ = policy(obs, env.t, env)
                obs, _, done, info = env.step(action)

            kpi_df = env.get_kpi_history()
            results.append(kpi_df["consumer_backlog_units"].tolist())

        assert results[0] == results[1], "Non-deterministic results with same seed!"


class TestBaselinePolicies:
    def test_random_reroute_respects_K(self, env_with_shocks):
        policy = make_random_reroute_policy(reroute_budget_K=2, seed=42)
        obs = {"t": 0}
        action, _ = policy(obs, 0, env_with_shocks)
        assert len(action["reroute"]) <= 2

    def test_backlog_only_produces_actions(self, env_with_shocks):
        policy = make_backlog_only_policy(reroute_budget_K=3)
        obs = {"t": 0}
        action, _ = policy(obs, 0, env_with_shocks)
        assert "reroute" in action
        assert "supply_multiplier" in action

    def test_expedite_only_no_reroutes(self, env_with_shocks):
        policy = make_expedite_only_policy()
        obs = {"t": 0}
        action, _ = policy(obs, 0, env_with_shocks)
        assert action["reroute"] == []

    def test_reroute_only_no_expedites(self, env_with_shocks):
        policy = make_reroute_only_policy(reroute_budget_K=3)
        obs = {"t": 0}
        action, _ = policy(obs, 0, env_with_shocks)
        assert action["supply_multiplier"] == {}
