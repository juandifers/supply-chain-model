import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { fetchExplain, fetchGraph, fetchKpis, fetchScenarioSummary } from '../lib/api'
import { useScenarioUiStore } from '../lib/store'
import { ExplainPanel } from '../components/ExplainPanel'
import { GraphPanel } from '../components/GraphPanel'
import { KpiTimelineChart } from '../components/KpiTimelineChart'
import { ReplayControls } from '../components/ReplayControls'

export function ScenarioPage() {
  const { id } = useParams<{ id: string }>()
  const scenarioId = id ?? ''

  const {
    t,
    tMin,
    tMax,
    isPlaying,
    productFilter,
    minFlow,
    topK,
    setBounds,
    stepForward,
    setPlaying,
    setProductFilter,
  } = useScenarioUiStore()

  const summaryQuery = useQuery({
    queryKey: ['scenario-summary', scenarioId],
    queryFn: () => fetchScenarioSummary(scenarioId),
    enabled: Boolean(scenarioId),
  })

  const kpiQuery = useQuery({
    queryKey: ['scenario-kpis', scenarioId],
    queryFn: () => fetchKpis(scenarioId),
    enabled: Boolean(scenarioId),
  })

  useEffect(() => {
    if (summaryQuery.data) {
      setBounds(summaryQuery.data.t_min, summaryQuery.data.t_max)
    }
  }, [summaryQuery.data, setBounds])

  useEffect(() => {
    if (!isPlaying) return
    if (t >= tMax) {
      setPlaying(false)
      return
    }
    const timer = window.setInterval(() => {
      stepForward()
    }, 800)
    return () => window.clearInterval(timer)
  }, [isPlaying, t, tMax, stepForward, setPlaying])

  const graphQuery = useQuery({
    queryKey: ['scenario-graph', scenarioId, t, productFilter, minFlow],
    queryFn: () => fetchGraph({ scenarioId, t, productFilter: productFilter || undefined, minFlow }),
    enabled: Boolean(scenarioId),
    placeholderData: (previousData) => previousData,
  })

  const explainQuery = useQuery({
    queryKey: ['scenario-explain', scenarioId, t, topK],
    queryFn: () => fetchExplain({ scenarioId, t, topK }),
    enabled: Boolean(scenarioId),
  })

  const kpis = kpiQuery.data ?? []
  const baselineScenarioId =
    typeof summaryQuery.data?.manifest?.baseline_scenario_id === 'string'
      ? summaryQuery.data.manifest.baseline_scenario_id
      : undefined

  return (
    <main className="page-shell">
      <header className="top-row">
        <div>
          <h1>Scenario Replay: {scenarioId}</h1>
          <p>
            Use time controls to inspect graph flow, shock exposure, and explainability evidence as the network evolves.
          </p>
        </div>
        <div className="top-actions">
          <Link className="btn secondary" to="/">
            Back
          </Link>
          {baselineScenarioId && (
            <Link
              className="btn"
              to={`/compare?baseline=${baselineScenarioId}&scenario=${scenarioId}`}
            >
              Compare To Baseline
            </Link>
          )}
        </div>
      </header>

      <ReplayControls kpis={kpis} />

      <section className="scenario-layout">
        <div className="center-column">
          <GraphPanel graph={graphQuery.data} />
          <KpiTimelineChart kpis={kpis} activeT={t} />
        </div>

        <ExplainPanel
          explain={explainQuery.data}
          onSelectPathProduct={(productId) => {
            setProductFilter(String(productId))
          }}
        />
      </section>

      {(summaryQuery.isError || kpiQuery.isError || graphQuery.isError || explainQuery.isError) && (
        <div className="error panel">One or more dashboard panels failed to load. Check backend logs and scenario validity.</div>
      )}

      <footer className="muted footer-line">
        t range: {tMin} -&gt; {tMax} | Graph {graphQuery.isFetching ? 'refreshing…' : 'synced'} | Explain{' '}
        {explainQuery.isFetching ? 'refreshing…' : 'synced'}
      </footer>
    </main>
  )
}
