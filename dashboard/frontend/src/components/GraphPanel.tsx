import { useEffect, useMemo, useRef, useState } from 'react'
import type cytoscape from 'cytoscape'
import CytoscapeComponent from 'react-cytoscapejs'
import type { GraphResponse } from '../types'

type GraphPanelProps = {
  graph?: GraphResponse
}

export function GraphPanel({ graph }: GraphPanelProps) {
  const [layout, setLayout] = useState<'cose' | 'breadthfirst'>('breadthfirst')
  const cyRef = useRef<cytoscape.Core | null>(null)

  const elements = useMemo(() => {
    if (!graph) return []

    const nodeElements = graph.nodes.map((n) => ({
      data: {
        id: n.id,
        label: n.label,
        openOrders: n.metrics.open_orders,
        inbound: n.metrics.inbound_units,
        outbound: n.metrics.outbound_units,
      },
      classes: n.metrics.open_orders > 50 ? 'node-hot' : 'node-cool',
    }))

    const edgeElements = graph.edges.map((e) => ({
      data: {
        id: e.id,
        source: e.source,
        target: e.target,
        label: `${e.product_name} | t:${e.flow_at_t.toFixed(1)} total:${e.total_flow.toFixed(1)}`,
        flow: e.flow_at_t,
      },
      classes: e.flow_at_t > 0 ? 'edge-active' : 'edge-idle',
    }))

    return [...nodeElements, ...edgeElements]
  }, [graph])

  const layoutConfig = useMemo(
    () =>
      layout === 'breadthfirst'
        ? {
            name: 'breadthfirst',
            fit: true,
            directed: true,
            circle: false,
            padding: 20,
            spacingFactor: 1.15,
            avoidOverlap: true,
            animate: true,
          }
        : {
            name: 'cose',
            fit: true,
            padding: 20,
            animate: true,
          },
    [layout]
  )

  useEffect(() => {
    const cy = cyRef.current
    if (!cy || !graph) return

    const rafId = window.requestAnimationFrame(() => {
      cy.layout(layoutConfig as cytoscape.LayoutOptions).run()
    })
    return () => window.cancelAnimationFrame(rafId)
  }, [graph?.t, layoutConfig, graph])

  return (
    <section className="panel graph-panel">
      <div className="panel-title-row">
        <h3>Firm-Supplier Network</h3>
        <div className="layout-toggle">
          <button onClick={() => setLayout('breadthfirst')} className={layout === 'breadthfirst' ? 'active' : ''}>
            DAG
          </button>
          <button onClick={() => setLayout('cose')} className={layout === 'cose' ? 'active' : ''}>
            Organic
          </button>
        </div>
      </div>

      <div className="graph-caption">
        {graph ? `${graph.nodes.length} firms | ${graph.edges.length} supplier links` : 'Loading graph...'}
      </div>

      <CytoscapeComponent
        elements={elements}
        style={{ width: '100%', height: '100%' }}
        layout={layoutConfig}
        cy={(cy: cytoscape.Core) => {
          cyRef.current = cy
        }}
        stylesheet={[
          {
            selector: 'node',
            style: {
              label: 'data(label)',
              color: '#111',
              'font-size': 10,
              'text-wrap': 'wrap',
              'text-max-width': 75,
              'background-color': '#a6d8ff',
              width: 'mapData(openOrders, 0, 200, 18, 44)',
              height: 'mapData(openOrders, 0, 200, 18, 44)',
              'border-width': 1,
              'border-color': '#183047',
            },
          },
          {
            selector: 'node.node-hot',
            style: {
              'background-color': '#ffad8a',
            },
          },
          {
            selector: 'edge',
            style: {
              width: 'mapData(flow, 0, 250, 1, 6)',
              'line-color': '#6a7f8f',
              'target-arrow-color': '#6a7f8f',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              opacity: 0.8,
              label: 'data(label)',
              'font-size': 8,
              'text-background-color': 'rgba(7,10,12,0.85)',
              'text-background-opacity': 1,
              'text-background-padding': 2,
              color: '#e9f1f6',
            },
          },
          {
            selector: 'edge.edge-active',
            style: {
              'line-color': '#ffd166',
              'target-arrow-color': '#ffd166',
            },
          },
        ]}
      />
    </section>
  )
}
