import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 600000 })

export interface MetricCI {
  value: number
  ci_low: number | null
  ci_high: number | null
}

export interface ExecutionResult {
  success: boolean
  strategy_slug: string
  strategy_name: string
  class_name: string
  initial_capital_usd: number
  start_iso: string
  end_iso: string
  n_intents: number
  n_snapshots: number
  final_equity_usd: number
  total_return_pct: number
  annualized_return_pct: number
  sharpe: MetricCI
  sortino: MetricCI
  calmar: MetricCI
  max_drawdown_pct: number
  max_drawdown_usd: number
  drawdown_duration_seconds: number
  hit_rate: MetricCI
  profit_factor: MetricCI
  expectancy_usd: MetricCI
  avg_win_usd: number
  avg_loss_usd: number
  trade_count: number
  fees_paid_usd: number
  fees_per_fill_usd: number
  fees_resolution_usd: number
  total_fills: number
  rejected_orders: number
  cancelled_orders: number
  closed_position_count: number
  open_position_count: number
  fills_sample: Array<Record<string, unknown>>
  equity_curve_sample: Array<{ timestamp?: string; equity_usd?: number }>
  load_time_ms: number
  data_fetch_time_ms: number
  run_time_ms: number
  total_time_ms: number
  validation_errors: string[]
  validation_warnings: string[]
  runtime_error: string | null
  runtime_traceback: string | null
}

export interface FillModelInfo {
  loaded: boolean
  family?: string
  strata_key?: string
  n_events?: number
  concordance_index?: number | null
  trained_at_epoch?: number
  promoted_at_epoch?: number | null
  coefficients?: Record<string, number>
  feature_means?: Record<string, number>
  feature_stds?: Record<string, number>
  baseline_survival_points?: Array<{ t_seconds: number; survival: number }>
  notes?: string
}

export interface EmpiricalConstants {
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
}

export interface LatencyDistribution {
  p50_ms: number
  p95_ms: number
  p99_ms: number
  sample_count: number
  pessimistic_ms: number
  realistic_ms: number
  optimistic_ms: number
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

export interface CounterfactualReplayResult {
  filled_shares: number
  average_fill_price: number | null
  time_to_fill_seconds: number | null
  final_queue_ahead: number
  cancels_ahead_observed: number
  trades_ahead_observed: number
  events_processed: number
  expired: boolean
  notes: string
}

export interface CounterfactualEntry {
  fill: {
    token_id: string
    side: string
    price: number
    size: number
    placed_at: string
  }
  result: CounterfactualReplayResult
}

export interface EnsembleBandEntry {
  fill_id: string
  pessimistic: number
  realistic: number
  optimistic: number
  cox_loaded: boolean
}

export interface UnifiedBacktestResult {
  run_id: string
  started_at: string
  completed_at: string
  total_time_ms: number
  strategy_slug: string
  strategy_name: string | null
  execution: ExecutionResult
  fill_model: FillModelInfo
  empirical_constants: EmpiricalConstants
  latency: LatencyDistribution
  decomposition: DecompositionSummary
  counterfactuals: CounterfactualEntry[]
  ensemble_band: EnsembleBandEntry[]
}

export interface BacktestRunSummary {
  run_id: string
  strategy_slug: string | null
  strategy_name: string | null
  started_at: string
  completed_at: string
  total_time_ms: number
  status: 'ok' | 'failed'
  trade_count: number
  total_return_pct: number
}

export interface RunBacktestRequest {
  source_code: string
  slug?: string
  config?: Record<string, unknown>
  token_ids?: string[]
  start?: string
  end?: string
  initial_capital_usd?: number
  submit_p50_ms?: number
  submit_p95_ms?: number
  cancel_p50_ms?: number
  cancel_p95_ms?: number
  seed?: number
  counterfactual_sample_size?: number
  ensemble_sample_size?: number
}

export async function runUnifiedBacktest(req: RunBacktestRequest): Promise<UnifiedBacktestResult> {
  const { data } = await api.post<UnifiedBacktestResult>('/backtest/run', req)
  return data
}

export async function listBacktestRuns(): Promise<BacktestRunSummary[]> {
  const { data } = await api.get<{ runs: BacktestRunSummary[] }>('/backtest/runs')
  return data.runs ?? []
}

export async function getBacktestRun(runId: string): Promise<UnifiedBacktestResult> {
  const { data } = await api.get<UnifiedBacktestResult>(`/backtest/runs/${encodeURIComponent(runId)}`)
  return data
}
