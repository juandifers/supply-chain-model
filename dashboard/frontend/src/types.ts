export type ScenarioMeta = {
  scenario_id: string
  created_at?: string | null
  description?: string | null
  baseline_scenario_id?: string | null
  is_valid: boolean
  errors: string[]
  sim_config: Record<string, unknown>
}

export type ScenarioSummary = {
  scenario_id: string
  manifest: Record<string, unknown>
  t_min: number
  t_max: number
  num_timesteps: number
  num_transactions: number
  peak_open_orders_t?: number | null
  peak_open_orders?: number | null
  peak_backlog_t?: number | null
  peak_backlog_units?: number | null
  final_cumulative_fill_rate?: number | null
}

export type KPIRow = {
  t: number
  transactions: number
  open_orders: number
  consumer_backlog_units: number
  consumer_cumulative_fill_rate: number
  shock_exposure: number
  active_exogenous_shocks: number
  expedite_cost_t?: number | null
  expedite_cost_cum?: number | null
  expedite_units_added_t?: number | null
  expedite_budget_remaining?: number | null
  [key: string]: string | number | null | undefined
}

export type GraphNode = {
  id: string
  label: string
  type: string
  metrics: {
    inbound_units: number
    outbound_units: number
    open_orders: number
  }
}

export type GraphEdge = {
  id: string
  source: string
  target: string
  product_id: number
  product_name: string
  flow_at_t: number
  total_flow: number
}

export type GraphResponse = {
  scenario_id: string
  t: number
  product_filter?: string | null
  min_flow: number
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export type ExplainResponse = {
  scenario_id: string
  t: number
  explain: {
    shock_summary: {
      t: number
      active_shocked_products: Array<{
        product_id: number
        product_name: string
        shock_severity: number
        total_supply: number
        baseline_supply: number
      }>
      worst_shocked_product: string | null
      worst_shock_severity: number
      shock_exposure: number
      active_exogenous_shocks: number
    }
    expedite_summary?: {
      expedite_cost_t: number
      expedite_cost_cum: number
      expedite_units_added_t: number
      expedite_budget_remaining: number | null
    }
    ripple_products_top_k: Array<{
      product_id: number
      product_name: string
      impact_score: number
      tx_drop_ratio: number
      backlog_increase_ratio: number
      shock_proximity_score: number
      tx_units: number
      backlog_units: number
    }>
    critical_firms_top_k: Array<{
      firm_id: number
      firm_name: string
      criticality_score: number
      out_degree: number
      downstream_coverage: number
      constrained_flow_share: number
      open_orders: number
      outbound_units: number
    }>
    paths: Array<{
      source_product_id: number
      source_product_name: string
      target_consumer_product_id: number
      target_consumer_product_name: string
      path_length: number
      path_score: number
      source_shock_severity: number
      path: Array<{ product_id: number; product_name: string }>
    }>
  }
}

export type CompareResponse = {
  baseline_id: string
  scenario_id: string
  summary: {
    t_start: number
    t_end: number
    num_points: number
    metrics: Record<
      string,
      {
        final_baseline: number
        final_scenario: number
        final_delta: number
        mean_delta: number
        peak_abs_delta: number
        peak_abs_delta_t: number
      }
    >
  }
  series: Array<{
    t: number
    [metric: string]:
      | number
      | {
          baseline: number
          scenario: number
          delta: number
        }
  }>
}
