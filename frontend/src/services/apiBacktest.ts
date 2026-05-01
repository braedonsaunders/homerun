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

export interface CalibrationBin {
  bin: number
  n: number
  predicted_mean: number
  observed_rate: number
  predicted_min: number
  predicted_max: number
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
  calibration_bins?: CalibrationBin[] | null
  notes?: string
}

export interface DeflatedSharpeResult {
  observed_sharpe: number
  sr_zero: number
  probabilistic_sharpe: number
  deflated_sharpe: number
  n_observations: number
  n_trials: number
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

export interface RegimeBucket {
  bucket: string
  n: number
  wins: number
  total_pnl_usd: number
  win_rate: number
  mean_pnl_usd: number
}

export interface RegimeBreakdown {
  by_hour: RegimeBucket[]
  by_dow: RegimeBucket[]
  by_ttr: RegimeBucket[]
  by_size: RegimeBucket[]
}

export interface PartialFillAggregates {
  n_orders: number
  n_instant_fills: number
  n_partial_fills: number
  instant_fill_rate: number
  mean_children_per_order: number
  max_children_per_order: number
  mean_intra_order_seconds: number
  mean_vwap_dispersion_bps: number
  child_count_distribution: Array<{ children: number; n_orders: number }>
}

export interface UnifiedBacktestResult {
  run_id: string
  started_at: string
  completed_at: string
  total_time_ms: number
  strategy_slug: string
  strategy_name: string | null
  execution: ExecutionResult
  deflated_sharpe: DeflatedSharpeResult
  regime_breakdown: RegimeBreakdown
  partial_fills: PartialFillAggregates
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

export interface WalkForwardWindow {
  index: number
  train_start_iso: string
  train_end_iso: string
  test_start_iso: string
  test_end_iso: string
  success: boolean
  runtime_error: string | null
  initial_capital_usd: number
  final_equity_usd: number
  total_return_pct: number
  sharpe: number | null
  sortino: number | null
  hit_rate: number | null
  trade_count: number
  total_fills: number
  rejected_orders: number
  cancelled_orders: number
}

export interface WalkForwardSummary {
  n_windows_run: number
  n_windows_succeeded: number
  mean_return_pct: number
  min_return_pct: number
  max_return_pct: number
  stable_window_pct: number
  mean_sharpe: number | null
  min_sharpe: number | null
  max_sharpe: number | null
}

export interface WalkForwardResult {
  mode: 'anchored' | 'rolling'
  n_windows_run: number
  overall_start_iso: string
  overall_end_iso: string
  windows: WalkForwardWindow[]
  summary: WalkForwardSummary
}

export interface WalkForwardRequest {
  source_code: string
  slug?: string
  config?: Record<string, unknown>
  token_ids?: string[]
  start: string
  end: string
  initial_capital_usd?: number
  mode?: 'anchored' | 'rolling'
  n_folds?: number
  train_ratio?: number
  embargo_seconds?: number
  submit_p50_ms?: number
  submit_p95_ms?: number
  cancel_p50_ms?: number
  cancel_p95_ms?: number
  seed?: number
  concurrency?: number
}

export async function runWalkForward(req: WalkForwardRequest): Promise<WalkForwardResult> {
  const { data } = await api.post<WalkForwardResult>('/backtest/walk-forward', req)
  return data
}

export interface PortfolioCorrelationResult {
  window_days: number
  strategies: string[]
  pnl_series_by_strategy: Record<string, Record<string, number>>
  correlation_matrix: number[][]
  summary: {
    n_strategies: number
    n_days: number
    mean_pairwise_correlation: number
    mean_abs_pairwise_correlation: number
    max_pairwise_correlation: number
    min_pairwise_correlation: number
    diversification_ratio: number
  }
}

export async function getPortfolioCorrelation(
  windowDays = 30,
  minStrategyTrades = 5,
): Promise<PortfolioCorrelationResult> {
  const { data } = await api.get<PortfolioCorrelationResult>('/backtest/portfolio-correlation', {
    params: { window_days: windowDays, min_strategy_trades: minStrategyTrades },
  })
  return data
}
