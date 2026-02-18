from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIO_ROOT = ROOT / "artifacts" / "scenarios"
SCENARIO_ROOT = Path(os.getenv("SUPPLYSIM_SCENARIO_ROOT", str(DEFAULT_SCENARIO_ROOT))).resolve()
