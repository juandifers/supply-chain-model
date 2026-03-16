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
    const readMetricSide = (row: (typeof payload.series)[number], metric: string, side: 'baseline' | 'scenario') => {
      const value = row[metric]
      if (typeof value === 'object' && value !== null && side in value) {
        const raw = (value as { baseline?: unknown; scenario?: unknown })[side]
        const n = Number(raw)
        return Number.isFinite(n) ? n : 0
      }
      return 0
    }

    const backlogB = payload.series.map((x) => readMetricSide(x, 'consumer_backlog_units', 'baseline'))
    const backlogS = payload.series.map((x) => readMetricSide(x, 'consumer_backlog_units', 'scenario'))
    const fillB = payload.series.map((x) => readMetricSide(x, 'consumer_cumulative_fill_rate', 'baseline'))
    const fillS = payload.series.map((x) => readMetricSide(x, 'consumer_cumulative_fill_rate', 'scenario'))
    const expediteCumB = payload.series.map((x) => readMetricSide(x, 'expedite_cost_cum', 'baseline'))
    const expediteCumS = payload.series.map((x) => readMetricSide(x, 'expedite_cost_cum', 'scenario'))

    return {
      tooltip: { trigger: 'axis' },
      legend: { top: 4, textStyle: { color: '#ece6d7' } },
      grid: [
        { left: 56, right: 20, top: 35, height: '25%' },
        { left: 56, right: 20, top: '38%', height: '22%' },
        { left: 56, right: 20, top: '67%', height: '24%' },
      ],
      xAxis: [
        { type: 'category', data: t, gridIndex: 0, axisLabel: { color: '#d8d3c5' } },
        { type: 'category', data: t, gridIndex: 1, axisLabel: { color: '#d8d3c5' } },
        { type: 'category', data: t, gridIndex: 2, axisLabel: { color: '#d8d3c5' } },
      ],
      yAxis: [
        { type: 'value', gridIndex: 0, axisLabel: { color: '#d8d3c5' } },
        { type: 'value', min: 0, max: 1, gridIndex: 1, axisLabel: { color: '#d8d3c5' } },
        { type: 'value', gridIndex: 2, axisLabel: { color: '#d8d3c5' } },
      ],
      series: [
        { name: 'Backlog Baseline', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: backlogB, lineStyle: { color: '#ff9f7f' } },
        { name: 'Backlog Scenario', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: backlogS, lineStyle: { color: '#ff5f6d' } },
        { name: 'Fill Baseline', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: fillB, lineStyle: { color: '#8dc9ff' } },
        { name: 'Fill Scenario', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: fillS, lineStyle: { color: '#7b8cff' } },
        {
          name: 'Expedite Cum Cost Baseline',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: expediteCumB,
          lineStyle: { color: '#ffd166' },
        },
        {
          name: 'Expedite Cum Cost Scenario',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: expediteCumS,
          lineStyle: { color: '#42e2b8' },
        },
      ],
    }
  }, [compareQuery.data])

  const metrics = compareQuery.data?.summary.metrics ?? {}
  const metricLabels: Record<string, string> = {
    transactions: 'Transactions',
    open_orders: 'Open Orders',
    consumer_backlog_units: 'Consumer Backlog Units',
    consumer_cumulative_fill_rate: 'Consumer Cumulative Fill Rate',
    shock_exposure: 'Shock Exposure',
    expedite_cost_t: 'Expedite Cost (t)',
    expedite_cost_cum: 'Expedite Cost (Cumulative)',
    expedite_units_added_t: 'Expedite Units Added (t)',
  }

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
                <h3>{metricLabels[name] ?? name}</h3>
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
            <ReactECharts option={chartOption} style={{ width: '100%', height: 440 }} />
          </section>
        </>
      )}

      {compareQuery.isLoading && <section className="panel muted">Loading comparison…</section>}
      {compareQuery.isError && <section className="panel error">Comparison failed. Check scenario IDs and backend logs.</section>}
    </main>
  )
}
