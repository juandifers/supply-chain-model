import type { ExplainResponse } from '../types'

type ExplainPanelProps = {
  explain?: ExplainResponse
  onSelectPathProduct: (productId: number) => void
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function ExplainPanel({ explain, onSelectPathProduct }: ExplainPanelProps) {
  const payload = explain?.explain

  return (
    <section className="panel explain-panel">
      <h3>Why Is Backlog Rising?</h3>

      {!payload && <div className="muted">Loading explainability…</div>}

      {payload && (
        <>
          <div className="shock-card">
            <div className="metric-row">
              <span>Shock Exposure</span>
              <strong>{formatPct(payload.shock_summary.shock_exposure)}</strong>
            </div>
            <div className="metric-row">
              <span>Active Exogenous Shocks</span>
              <strong>{payload.shock_summary.active_exogenous_shocks}</strong>
            </div>
            <div className="metric-row">
              <span>Worst Shock Product</span>
              <strong>{payload.shock_summary.worst_shocked_product ?? '-'}</strong>
            </div>
          </div>

          <div className="explain-block">
            <h4>Ripple Products</h4>
            {payload.ripple_products_top_k.slice(0, 6).map((row) => (
              <div className="rank-item" key={row.product_id}>
                <div>
                  <div className="rank-name">{row.product_name}</div>
                  <div className="rank-sub">
                    score {row.impact_score.toFixed(3)} | tx drop {formatPct(row.tx_drop_ratio)} | backlog +
                    {formatPct(row.backlog_increase_ratio)}
                  </div>
                </div>
                <button className="chip" onClick={() => onSelectPathProduct(row.product_id)}>
                  Filter Graph
                </button>
              </div>
            ))}
          </div>

          <div className="explain-block">
            <h4>Chokepoint Firms</h4>
            {payload.critical_firms_top_k.slice(0, 6).map((row) => (
              <div className="rank-item" key={row.firm_id}>
                <div>
                  <div className="rank-name">{row.firm_name}</div>
                  <div className="rank-sub">
                    criticality {row.criticality_score.toFixed(3)} | open orders {row.open_orders.toFixed(0)}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="explain-block">
            <h4>Shock To Consumer Paths</h4>
            {payload.paths.slice(0, 5).map((path, idx) => (
              <div className="path-item" key={`${path.source_product_id}-${path.target_consumer_product_id}-${idx}`}>
                <div className="path-head">
                  <span>
                    {path.source_product_name} {'->'} {path.target_consumer_product_name}
                  </span>
                  <button className="chip" onClick={() => onSelectPathProduct(path.target_consumer_product_id)}>
                    Highlight
                  </button>
                </div>
                <div className="path-flow">{path.path.map((p) => p.product_name).join(' -> ')}</div>
                <div className="path-strip">
                  <div className="path-strip-fill" style={{ width: `${Math.min(100, Math.max(8, path.path_score * 100))}%` }} />
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}
