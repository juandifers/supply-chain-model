import axios from 'axios'
import type { CompareResponse, ExplainResponse, GraphResponse, KPIRow, ScenarioMeta, ScenarioSummary } from '../types'

const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export const api = axios.create({
  baseURL: apiBase,
  timeout: 30000,
})

export async function fetchScenarios(): Promise<ScenarioMeta[]> {
  const { data } = await api.get<{ scenarios: ScenarioMeta[] }>('/api/v1/scenarios')
  return data.scenarios
}

export async function fetchScenarioSummary(scenarioId: string): Promise<ScenarioSummary> {
  const { data } = await api.get<ScenarioSummary>(`/api/v1/scenarios/${scenarioId}/summary`)
  return data
}

export async function fetchKpis(scenarioId: string, startT?: number, endT?: number): Promise<KPIRow[]> {
  const { data } = await api.get<{ rows: KPIRow[] }>(`/api/v1/scenarios/${scenarioId}/kpis`, {
    params: {
      start_t: startT,
      end_t: endT,
    },
  })
  return data.rows
}

export async function fetchGraph(params: {
  scenarioId: string
  t: number
  productFilter?: string
  minFlow?: number
}): Promise<GraphResponse> {
  const { data } = await api.get<GraphResponse>(`/api/v1/scenarios/${params.scenarioId}/graph`, {
    params: {
      t: params.t,
      product_filter: params.productFilter,
      min_flow: params.minFlow ?? 0,
    },
  })
  return data
}

export async function fetchExplain(params: {
  scenarioId: string
  t: number
  topK?: number
}): Promise<ExplainResponse> {
  const { data } = await api.get<ExplainResponse>(`/api/v1/scenarios/${params.scenarioId}/explain`, {
    params: {
      t: params.t,
      top_k: params.topK ?? 10,
    },
  })
  return data
}

export async function fetchCompare(params: {
  baselineId: string
  scenarioId: string
  startT?: number
  endT?: number
}): Promise<CompareResponse> {
  const { data } = await api.get<CompareResponse>('/api/v1/compare', {
    params: {
      baseline_id: params.baselineId,
      scenario_id: params.scenarioId,
      start_t: params.startT,
      end_t: params.endT,
    },
  })
  return data
}
