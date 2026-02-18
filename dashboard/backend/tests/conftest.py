from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.export_supplysim_scenario import _export_scenario


@pytest.fixture(scope="session")
def scenario_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("supplysim_scenarios") / "scenarios"
    root.mkdir(parents=True, exist_ok=True)

    _export_scenario(
        scenario_dir=root / "baseline_seed0",
        seed=0,
        T=20,
        gamma=0.8,
        shock_prob=0.001,
        init_inv=0.0,
        init_supply=100.0,
        init_demand=1.0,
        description="baseline fixture",
        baseline_scenario_id=None,
    )

    _export_scenario(
        scenario_dir=root / "scenario_seed1",
        seed=1,
        T=20,
        gamma=0.7,
        shock_prob=0.003,
        init_inv=0.0,
        init_supply=100.0,
        init_demand=1.0,
        description="comparison fixture",
        baseline_scenario_id="baseline_seed0",
    )

    return root


@pytest.fixture()
def client(scenario_root: Path) -> TestClient:
    os.environ["SUPPLYSIM_SCENARIO_ROOT"] = str(scenario_root)

    import dashboard.backend.app.config as config_module
    import dashboard.backend.app.main as main_module
    import dashboard.backend.app.services.scenario_store as scenario_store

    importlib.reload(config_module)
    importlib.reload(scenario_store)
    importlib.reload(main_module)

    return TestClient(main_module.app)
