"""Shared fixtures for optimizer tests."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv


@pytest.fixture
def env_with_shocks():
    """Environment configured with moderate shocks and expedite budget."""
    env = SupplySimEnv(
        seed=0, T=30, gamma=0.8,
        expedite_budget=500.0,
        expedite_c0=1.0, expedite_alpha=0.5, expedite_m_max=3.0,
    )
    env.reset(init_inv=0, init_supply=100, init_demand=1, shock_prob=0.05)
    return env


@pytest.fixture
def env_no_shocks():
    """Environment with no shocks for baseline testing."""
    env = SupplySimEnv(
        seed=0, T=20, gamma=0.8,
        expedite_budget=200.0,
    )
    env.reset(init_inv=0, init_supply=100, init_demand=1, shock_prob=0.0)
    return env
