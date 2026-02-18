from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .scenario_store import ScenarioData, filter_by_t


DEFAULT_COMPARE_METRICS = [
    "transactions",
    "open_orders",
    "consumer_backlog_units",
    "consumer_cumulative_fill_rate",
    "shock_exposure",
]


def _align_kpis(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    metrics: List[str],
) -> pd.DataFrame:
    keep_cols = ["t"] + metrics
    b = baseline[keep_cols].copy()
    s = scenario[keep_cols].copy()
    merged = pd.merge(b, s, on="t", how="inner", suffixes=("_baseline", "_scenario"))
    merged = merged.sort_values("t").reset_index(drop=True)

    for m in metrics:
        merged[f"{m}_delta"] = merged[f"{m}_scenario"] - merged[f"{m}_baseline"]
    return merged


def compare_scenarios(
    baseline_data: ScenarioData,
    scenario_data: ScenarioData,
    start_t: Optional[int] = None,
    end_t: Optional[int] = None,
    metrics: Optional[List[str]] = None,
) -> Dict:
    metrics = metrics or DEFAULT_COMPARE_METRICS

    b = filter_by_t(baseline_data.kpi_history, start_t=start_t, end_t=end_t)
    s = filter_by_t(scenario_data.kpi_history, start_t=start_t, end_t=end_t)
    aligned = _align_kpis(baseline=b, scenario=s, metrics=metrics)

    if len(aligned) == 0:
        raise ValueError("No overlapping timestep range found between baseline and scenario")

    summary = {
        "t_start": int(aligned["t"].min()),
        "t_end": int(aligned["t"].max()),
        "num_points": int(len(aligned)),
        "metrics": {},
    }

    for m in metrics:
        delta_col = f"{m}_delta"
        baseline_col = f"{m}_baseline"
        scenario_col = f"{m}_scenario"

        final_baseline = float(aligned.iloc[-1][baseline_col])
        final_scenario = float(aligned.iloc[-1][scenario_col])
        final_delta = float(aligned.iloc[-1][delta_col])

        peak_abs_idx = int(aligned[delta_col].abs().idxmax())
        peak_abs_row = aligned.loc[peak_abs_idx]

        summary["metrics"][m] = {
            "final_baseline": float(np.round(final_baseline, 6)),
            "final_scenario": float(np.round(final_scenario, 6)),
            "final_delta": float(np.round(final_delta, 6)),
            "mean_delta": float(np.round(aligned[delta_col].mean(), 6)),
            "peak_abs_delta": float(np.round(float(peak_abs_row[delta_col]), 6)),
            "peak_abs_delta_t": int(peak_abs_row["t"]),
        }

    series = []
    for row in aligned.itertuples(index=False):
        item = {"t": int(row.t)}
        for m in metrics:
            item[m] = {
                "baseline": float(getattr(row, f"{m}_baseline")),
                "scenario": float(getattr(row, f"{m}_scenario")),
                "delta": float(getattr(row, f"{m}_delta")),
            }
        series.append(item)

    return {
        "baseline_id": baseline_data.scenario_id,
        "scenario_id": scenario_data.scenario_id,
        "summary": summary,
        "series": series,
    }
