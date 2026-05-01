import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 60000 })

export interface FillModelRow {
  id: string
  family: 'cox_ph' | 'kaplan_meier' | string
  strata_key: string
  trained_at: string | null
  training_window_start: string | null
  training_window_end: string | null
  n_events: number
  n_observations: number
  concordance_index: number | null
  brier_score: number | null
  log_likelihood: number | null
  coefficients: Record<string, number>
  baseline_survival: Record<string, number>
  feature_means: Record<string, number>
  feature_stds: Record<string, number>
  config: Record<string, unknown>
  promoted_at: string | null
  active: boolean
  notes: string
}

export interface EmpiricalConstantsResponse {
  measured: boolean
  sample_count: number
  measured_at_epoch: number
  notes: string
  values: {
    displayed_depth_factor: number
    maker_queue_ahead_fraction: number
    maker_trade_flow_multiplier: number
    adverse_selection_multiplier: number
    stale_depth_decay: number
    min_depth_factor: number
  }
  overrides: Record<string, number>
}

export interface LatencyDistribution {
  p50_ms: number
  p95_ms: number
  p99_ms: number
  sample_count: number
  pessimistic_ms: number
  realistic_ms: number
  optimistic_ms: number
  fallback_p50_ms?: number
  fallback_p95_ms?: number
  fallback_p99_ms?: number
}

export interface DecompositionSummary {
  window_hours: number
  trade_count: number
  cancel_count: number
  trade_size: number
  cancel_size: number
  trade_count_pct: number | null
  trade_size_pct: number | null
}

export interface TriangulationResponse {
  strategy_slug: string
  window_days: number
  modes: Record<
    string,
    {
      orders: number
      filled: number
      filled_notional_usd: number
      realized_pnl_usd: number
    }
  >
}

export async function getActiveFillModel(strataKey = 'pooled'): Promise<FillModelRow | null> {
  try {
    const { data } = await api.get<FillModelRow>('/fill-model/active', {
      params: { strata_key: strataKey },
    })
    return data
  } catch (err: unknown) {
    if ((err as { response?: { status?: number } }).response?.status === 404) {
      return null
    }
    throw err
  }
}

export async function getFillModelHistory(
  strataKey?: string,
  limit = 20,
): Promise<FillModelRow[]> {
  const params: Record<string, string | number> = { limit }
  if (strataKey) params.strata_key = strataKey
  const { data } = await api.get<{ models: FillModelRow[] }>('/fill-model/history', { params })
  return data.models ?? []
}

export async function triggerRetrain(windowDays = 30): Promise<{ trained: Array<Record<string, unknown>>; window_days: number }> {
  const { data } = await api.post('/fill-model/retrain', null, { params: { window_days: windowDays } })
  return data
}

export async function promoteModel(modelId: string): Promise<{ promoted: boolean; model_id: string }> {
  const { data } = await api.post(`/fill-model/promote/${encodeURIComponent(modelId)}`)
  return data
}

export async function getEmpiricalConstants(): Promise<EmpiricalConstantsResponse> {
  const { data } = await api.get<EmpiricalConstantsResponse>('/fill-model/empirical-constants')
  return data
}

export async function setEmpiricalOverrides(
  overrides: Partial<EmpiricalConstantsResponse['values']>,
): Promise<EmpiricalConstantsResponse> {
  const { data } = await api.put<EmpiricalConstantsResponse>('/fill-model/empirical-constants', overrides)
  return data
}

export async function getLatencyDistribution(): Promise<LatencyDistribution> {
  const { data } = await api.get<LatencyDistribution>('/fill-model/latency')
  return data
}

export async function updateLatencyFallbacks(
  values: { p50_ms?: number | null; p95_ms?: number | null; p99_ms?: number | null },
): Promise<{ p50_ms: number; p95_ms: number; p99_ms: number }> {
  const params: Record<string, number> = {}
  if (values.p50_ms != null) params.p50_ms = values.p50_ms
  if (values.p95_ms != null) params.p95_ms = values.p95_ms
  if (values.p99_ms != null) params.p99_ms = values.p99_ms
  const { data } = await api.put<{ p50_ms: number; p95_ms: number; p99_ms: number }>(
    '/fill-model/latency/fallbacks',
    null,
    { params },
  )
  return data
}

export async function getDecompositionSummary(hours = 24): Promise<DecompositionSummary> {
  const { data } = await api.get<DecompositionSummary>('/fill-model/decomposition-summary', {
    params: { hours },
  })
  return data
}

export async function getTriangulation(strategySlug: string, days = 30): Promise<TriangulationResponse> {
  const { data } = await api.get<TriangulationResponse>(`/fill-model/triangulation/${encodeURIComponent(strategySlug)}`, {
    params: { days },
  })
  return data
}

export interface MLCapabilitySummary {
  task_key: string
  label: string
  description: string
  allowed_assets: string[]
  allowed_timeframes: string[]
  default_lookback: number
  owner_strategy_slug: string | null
  feature_names: string[]
}

export async function listMLCapabilities(): Promise<MLCapabilitySummary[]> {
  const { data } = await api.get<{ tasks: MLCapabilitySummary[] }>('/ml/tasks')
  return data.tasks ?? []
}
