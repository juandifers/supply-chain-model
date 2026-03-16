import { useMemo } from 'react'
import { useScenarioUiStore } from '../lib/store'
import type { KPIRow } from '../types'

type ReplayControlsProps = {
  kpis: KPIRow[]
}

export function ReplayControls({ kpis }: ReplayControlsProps) {
  const {
    t,
    tMin,
    tMax,
    isPlaying,
    productFilter,
    minFlow,
    topK,
    setT,
    setPlaying,
    stepForward,
    stepBackward,
    setProductFilter,
    setMinFlow,
    setTopK,
  } = useScenarioUiStore()

  const shockPeakT = useMemo(() => {
    if (kpis.length === 0) return tMin
    return kpis.reduce((best, row) => (row.shock_exposure > best.shock_exposure ? row : best), kpis[0]).t
  }, [kpis, tMin])

  const activeKpi = useMemo(() => kpis.find((row) => row.t === t), [kpis, t])

  const expediteCostT =
    activeKpi && typeof activeKpi.expedite_cost_t === 'number' ? activeKpi.expedite_cost_t : 0
  const expediteCostCum =
    activeKpi && typeof activeKpi.expedite_cost_cum === 'number' ? activeKpi.expedite_cost_cum : 0
  const expediteUnitsT =
    activeKpi && typeof activeKpi.expedite_units_added_t === 'number' ? activeKpi.expedite_units_added_t : 0
  const budgetRemaining =
    activeKpi && typeof activeKpi.expedite_budget_remaining === 'number'
      ? activeKpi.expedite_budget_remaining
      : null

  return (
    <section className="panel controls-panel">
      <div className="controls-top">
        <button onClick={() => setPlaying(!isPlaying)}>{isPlaying ? 'Pause' : 'Play'}</button>
        <button onClick={() => stepBackward()}>Step -1</button>
        <button onClick={() => stepForward()}>Step +1</button>
        <button onClick={() => setT(shockPeakT)}>Jump To Shock Peak</button>
        <span className="pill">t={t}</span>
      </div>

      <div className="controls-kpi-strip">
        <span className="pill">Expedite Cost t: {expediteCostT.toFixed(1)}</span>
        <span className="pill">Expedite Units t: {expediteUnitsT.toFixed(1)}</span>
        <span className="pill">Expedite Cost Cum: {expediteCostCum.toFixed(1)}</span>
        <span className="pill">
          Budget Remaining: {budgetRemaining === null ? 'unlimited' : budgetRemaining.toFixed(1)}
        </span>
      </div>

      <div className="slider-row">
        <span>{tMin}</span>
        <input type="range" min={tMin} max={tMax} value={t} onChange={(e) => setT(Number(e.target.value))} />
        <span>{tMax}</span>
      </div>

      <div className="control-grid">
        <label>
          Product Filter
          <input
            value={productFilter}
            placeholder="product name,product id"
            onChange={(e) => setProductFilter(e.target.value)}
          />
        </label>

        <label>
          Min Flow
          <input
            type="number"
            min={0}
            step={1}
            value={minFlow}
            onChange={(e) => setMinFlow(Number(e.target.value))}
          />
        </label>

        <label>
          Explain Top-K
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </label>
      </div>
    </section>
  )
}
