import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchScenarios } from '../lib/api'

export function HomePage() {
  const scenariosQuery = useQuery({
    queryKey: ['scenarios'],
    queryFn: fetchScenarios,
  })

  const scenarios = scenariosQuery.data ?? []

  return (
    <main className="page-shell">
      <header className="hero">
        <h1>SupplySim Control Tower</h1>
        <p>
          Replay scenarios, inspect shock propagation across supplier networks, and explain backlog with product ripple
          and chokepoint evidence.
        </p>
      </header>

      <section className="panel home-panel">
        <div className="panel-title-row">
          <h2>Scenarios</h2>
          <span className="pill">{scenarios.length} loaded</span>
        </div>

        {scenariosQuery.isLoading && <div className="muted">Loading scenarios…</div>}
        {scenariosQuery.isError && <div className="error">Failed to load scenarios.</div>}

        <div className="scenario-grid">
          {scenarios.map((scenario) => (
            <article key={scenario.scenario_id} className={`scenario-card ${scenario.is_valid ? '' : 'invalid'}`}>
              <h3>{scenario.scenario_id}</h3>
              <p>{scenario.description || 'No description'}</p>
              <div className="config-line">
                seed: {String((scenario.sim_config as Record<string, unknown>).seed ?? '-')}, T:{' '}
                {String((scenario.sim_config as Record<string, unknown>).T ?? '-')}, gamma:{' '}
                {String((scenario.sim_config as Record<string, unknown>).gamma ?? '-')}
              </div>
              {!scenario.is_valid && <div className="error small">Invalid: {scenario.errors.join('; ')}</div>}

              <div className="card-actions">
                <Link className="btn" to={`/scenario/${scenario.scenario_id}`}>
                  Open Replay
                </Link>

                {scenario.baseline_scenario_id && (
                  <Link
                    className="btn secondary"
                    to={`/compare?baseline=${scenario.baseline_scenario_id}&scenario=${scenario.scenario_id}`}
                  >
                    Compare To Baseline
                  </Link>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  )
}
