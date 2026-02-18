import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { Link, useSearchParams } from 'react-router-dom'
import { fetchCompare, fetchScenarios } from '../lib/api'

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const baselineId = searchParams.get('baseline') ?? ''
  const scenarioId = searchParams.get('scenario') ?? ''

  const scenariosQuery = useQuery({
    queryKey: ['scenarios'],
    queryFn: fetchScenarios,
  })

  const compareQuery = useQuery({
    queryKey: ['compare', baselineId, scenarioId],
    queryFn: () => fetchCompare({ baselineId, scenarioId }),
    enabled: Boolean(baselineId && scenarioId),
  })

  const chartOption = useMemo(() => {
    const payload = compareQuery.data
    if (!payload) return {}

    const t = payload.series.map((x) => x.t)
    const backlogB = payload.series.map((x) => Number((x.consumer_backlog_units as { baseline: number }).baseline))
    const backlogS = payload.series.map((x) => Number((x.consumer_backlog_units as { scenario: number }).scenario))
    const fillB = payload.series.map((x) => Number((x.consumer_cumulative_fill_rate as { baseline: number }).baseline))
    const fillS = payload.series.map((x) => Number((x.consumer_cumulative_fill_rate as { scenario: number }).scenario))

    return {
      tooltip: { trigger: 'axis' },
      legend: { top: 4, textStyle: { color: '#ece6d7' } },
      grid: [
        { left: 56, right: 20, top: 35, height: '45%' },
        { left: 56, right: 20, top: '60%', height: '30%' },
      ],
      xAxis: [
        { type: 'category', data: t, gridIndex: 0, axisLabel: { color: '#d8d3c5' } },
        { type: 'category', data: t, gridIndex: 1, axisLabel: { color: '#d8d3c5' } },
      ],
      yAxis: [
        { type: 'value', gridIndex: 0, axisLabel: { color: '#d8d3c5' } },
        { type: 'value', min: 0, max: 1, gridIndex: 1, axisLabel: { color: '#d8d3c5' } },
      ],
      series: [
        { name: 'Backlog Baseline', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: backlogB, lineStyle: { color: '#ff9f7f' } },
        { name: 'Backlog Scenario', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: backlogS, lineStyle: { color: '#ff5f6d' } },
        { name: 'Fill Baseline', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: fillB, lineStyle: { color: '#8dc9ff' } },
        { name: 'Fill Scenario', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: fillS, lineStyle: { color: '#7b8cff' } },
      ],
    }
  }, [compareQuery.data])

  const metrics = compareQuery.data?.summary.metrics ?? {}

  return (
    <main className="page-shell">
      <header className="top-row">
        <div>
          <h1>Baseline vs Scenario Comparison</h1>
          <p>Overlay KPI trajectories and inspect metric deltas at final and peak divergence points.</p>
        </div>
        <Link className="btn secondary" to="/">
          Back
        </Link>
      </header>

      <section className="panel controls-panel compare-controls">
        <label>
          Baseline
          <select
            value={baselineId}
            onChange={(e) =>
              setSearchParams((prev) => {
                const next = new URLSearchParams(prev)
                next.set('baseline', e.target.value)
                return next
              })
            }
          >
            <option value="">Select baseline...</option>
            {(scenariosQuery.data ?? []).map((s) => (
              <option key={s.scenario_id} value={s.scenario_id}>
                {s.scenario_id}
              </option>
            ))}
          </select>
        </label>

        <label>
          Scenario
          <select
            value={scenarioId}
            onChange={(e) =>
              setSearchParams((prev) => {
                const next = new URLSearchParams(prev)
                next.set('scenario', e.target.value)
                return next
              })
            }
          >
            <option value="">Select scenario...</option>
            {(scenariosQuery.data ?? []).map((s) => (
              <option key={s.scenario_id} value={s.scenario_id}>
                {s.scenario_id}
              </option>
            ))}
          </select>
        </label>
      </section>

      {compareQuery.data && (
        <>
          <section className="metric-grid">
            {Object.entries(metrics).map(([name, m]) => (
              <article key={name} className="panel metric-card">
                <h3>{name}</h3>
                <div className="metric-row">
                  <span>Final Delta</span>
                  <strong>{m.final_delta.toFixed(3)}</strong>
                </div>
                <div className="metric-row">
                  <span>Mean Delta</span>
                  <strong>{m.mean_delta.toFixed(3)}</strong>
                </div>
                <div className="metric-row">
                  <span>Peak |delta| @ t={m.peak_abs_delta_t}</span>
                  <strong>{m.peak_abs_delta.toFixed(3)}</strong>
                </div>
              </article>
            ))}
          </section>

          <section className="panel chart-panel">
            <h3>KPI Overlay</h3>
            <ReactECharts option={chartOption} style={{ width: '100%', height: 360 }} />
          </section>
        </>
      )}

      {compareQuery.isLoading && <section className="panel muted">Loading comparison…</section>}
      {compareQuery.isError && <section className="panel error">Comparison failed. Check scenario IDs and backend logs.</section>}
    </main>
  )
}
