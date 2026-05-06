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
  expected_shortfall_5pct: MetricCI
  expected_shortfall_1pct: MetricCI
  tail_ratio: MetricCI
  gain_to_pain: MetricCI
  fills_sample: Array<Record<string, unknown>>
  equity_curve_sample: Array<{ timestamp?: string; equity_usd?: number }>
  positions_summary: Array<Record<string, unknown>>
  load_time_ms: number
  data_fetch_time_ms: number
  run_time_ms: number
  total_time_ms: number
  validation_errors: string[]
  validation_warnings: string[]
  runtime_error: string | null
  runtime_traceback: string | null
  /** Which replay source the engine ran against (live-parity if "deltas*"). */
  replay_source?: ReplaySource
  /** Pre-flight data coverage stats — same struct as the top-level field. */
  data_coverage?: DataCoverageStats
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

export interface DataQualityStats {
  accepted_books: number
  total_attempts: number
  accept_rate: number | null
  rejects_by_reason: Record<string, number>
  sequence_gaps_observed: number
  tokens_tracked: number
  queue_dropped: number
}

export interface OutcomeNettingReport {
  gross_exposure_usd: number
  net_exposure_usd: number
  rebate_estimate_usd: number
  capital_efficiency_pct: number | null
  locked_capital_usd: number
  open_positions: number
  outcome_groups: {
    full_coverage: number
    partial: number
    single: number
    total: number
  }
  by_outcome_count: Record<string, number>
  avg_lockup_seconds: number | null
  max_lockup_seconds: number | null
  n_lockup_samples: number
}

export interface TradeOrderMonteCarlo {
  n_resamples: number
  realized_sharpe?: number
  n_trades?: number
  skipped_reason?: string
  sharpe_distribution: {
    mean?: number
    stdev?: number
    p5?: number
    p25?: number
    p50?: number
    p75?: number
    p95?: number
    min?: number
    max?: number
  }
  observed_vs_distribution: {
    position_pct: number
    z_score: number
    interpretation: 'sequence-driven' | 'robust-to-sequence'
  } | null
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
  data_quality?: DataQualityStats
  outcome_netting?: OutcomeNettingReport
  trade_order_monte_carlo?: TradeOrderMonteCarlo
  /**
   * Pre-flight historical-data coverage stats for the run's opp universe.
   * "0 trades" with low fidelity is a data problem, not a strategy
   * problem — surface this prominently before the trade-count headline.
   */
  data_coverage?: DataCoverageStats
}

/**
 * Backtest data coverage / fidelity rating.
 *
 * The matching engine reads from ``market_microstructure_snapshots``;
 * the live system writes deltas to ``book_delta_events``.  When the
 * snapshot table is sparse (no recorder ever ran) but deltas are
 * dense, fidelity comes back "low" / "none" with a recommendation
 * pointing to the backfill or provider-import path.
 */
export interface DataCoverageStats {
  opp_tokens: number
  tokens_with_snapshots: number
  tokens_with_deltas: number
  snapshots_total: number
  deltas_total: number
  median_snaps_per_token_per_hour: number
  p10_snaps_per_token_per_hour: number
  median_deltas_per_token_per_hour: number
  fidelity_rating: 'high' | 'medium' | 'low' | 'none' | 'unknown'
  recommended_action: string
  error?: string
}

/**
 * Which book-replay source the matching engine ran against.
 *   - "snapshots"     — BookReplay reading market_microstructure_snapshots
 *   - "deltas"        — BookDeltaReplay reading book_delta_events (live-parity)
 *   - "deltas+anchor" — BookDeltaReplay seeded from mms anchors + replayed
 */
export type ReplaySource = 'snapshots' | 'deltas' | 'deltas+anchor' | ''

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
  // Up-to-16-point equity-curve % drift from the run's start.  Empty
  // when the run produced no equity samples (e.g. instant failure).
  sparkline_pct?: number[]
}

export interface RunBacktestRequest {
  source_code: string
  slug?: string
  config?: Record<string, unknown>
  token_ids?: string[]
  start?: string
  end?: string
  /** Recording session ID — overrides token_ids/start/end if set. */
  session_id?: string | null
  /** Imported provider dataset IDs — resolves to union of token_ids and
   *  [min start, max end].  Mutually exclusive with session_id; session
   *  wins when both are present. */
  provider_dataset_ids?: string[] | null
  initial_capital_usd?: number
  submit_p50_ms?: number
  submit_p95_ms?: number
  cancel_p50_ms?: number
  cancel_p95_ms?: number
  seed?: number
  counterfactual_sample_size?: number
  ensemble_sample_size?: number
  // Phase 12f/g operator overrides
  impact_strength_bps?: number
  impact_capacity_threshold?: number
  impact_capacity_exponent?: number
  maker_rebate_bps?: number
  maker_rebate_max_spread_bps?: number
  latency_correlation_window_ms?: number
}

/**
 * Legacy synchronous run.  Returns the full result blob; blocks the
 * API process for the entire engine wall time.  Kept for back-compat;
 * the BacktestStudio now uses ``enqueueBacktest`` + polling.
 */
export async function runUnifiedBacktest(req: RunBacktestRequest): Promise<UnifiedBacktestResult> {
  const { data } = await api.post<UnifiedBacktestResult>('/backtest/run', req)
  return data
}

/**
 * Async-by-default backtest enqueue.  Returns immediately with the
 * allocated ``run_id``; the engine runs in the dedicated worker
 * process (services/backtest/job_runner.py).  Operator polls
 * ``getBacktestRunStatus(runId)`` for progress.
 */
export interface EnqueueBacktestResponse {
  run_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'ok'
  message: string | null
  created_at: string | null
}

export async function enqueueBacktest(req: RunBacktestRequest): Promise<EnqueueBacktestResponse> {
  const { data } = await api.post<EnqueueBacktestResponse>('/backtest/runs/enqueue', req)
  return data
}

/**
 * Lightweight poll-friendly status.  Distinct from ``getBacktestRun``
 * which returns the heavy result blob (only meaningful once the run
 * is complete).
 *
 * ``progress`` is 0.0-1.0 when ``snapshots_total_estimate`` is set;
 * otherwise stays at 0 and the operator reads ``snapshots_processed``
 * + ``message`` for an indeterminate-style indicator.
 */
export interface BacktestRunStatus {
  run_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'ok'
  progress: number
  message: string | null
  snapshots_processed: number
  snapshots_total_estimate: number | null
  trade_count: number
  total_return_pct: number
  error: string | null
  claimed_at: string | null
  completed_at: string | null
  worker_id: string | null
  cancel_requested: boolean
}

export async function getBacktestRunStatus(runId: string): Promise<BacktestRunStatus> {
  const { data } = await api.get<BacktestRunStatus>(
    `/backtest/runs/${encodeURIComponent(runId)}/status`,
  )
  return data
}

export async function cancelBacktestRun(runId: string): Promise<{ run_id: string; cancel_requested: boolean }> {
  const { data } = await api.post<{ run_id: string; cancel_requested: boolean }>(
    `/backtest/runs/${encodeURIComponent(runId)}/cancel`,
  )
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

// ─── CPCV (Combinatorial Purged Cross-Validation) ───────────────────────────

export interface CPCVPath {
  path_index: number
  test_fold_indices: number[]
  test_start_iso: string
  test_end_iso: string
  n_intents: number
  trade_count: number
  total_fills: number
  success: boolean
  runtime_error: string | null
  total_return_pct: number
  sharpe: number | null
  sortino: number | null
  max_drawdown_pct: number
}

export interface CPCVSummary {
  n_paths_run: number
  n_paths_succeeded: number
  sharpe_mean: number | null
  sharpe_median: number | null
  sharpe_p10: number | null
  sharpe_p90: number | null
  sharpe_min: number | null
  sharpe_max: number | null
  return_mean_pct: number | null
  return_min_pct: number | null
  return_max_pct: number | null
  max_dd_mean_pct: number | null
  max_dd_worst_pct: number | null
  pbo: number | null
  stable_path_pct: number
}

export interface CPCVResult {
  n_folds: number
  k_test_folds: number
  embargo_seconds: number
  n_paths: number
  paths: CPCVPath[]
  summary: CPCVSummary
}

export interface CPCVRequest {
  source_code: string
  slug?: string
  config?: Record<string, unknown>
  token_ids?: string[]
  start: string
  end: string
  initial_capital_usd?: number
  n_folds?: number
  k_test_folds?: number
  embargo_seconds?: number
  submit_p50_ms?: number
  submit_p95_ms?: number
  cancel_p50_ms?: number
  cancel_p95_ms?: number
  seed?: number
  concurrency?: number
  max_paths?: number
}

export async function runCPCV(req: CPCVRequest): Promise<CPCVResult> {
  const { data } = await api.post<CPCVResult>('/backtest/cpcv', req)
  return data
}

// ─── Monte Carlo latency perturbation ───────────────────────────────────────

export interface LatencyPerturbationRun {
  p95_multiplier: number
  submit_p50_ms: number
  submit_p95_ms: number
  cancel_p50_ms: number
  cancel_p95_ms: number
  success: boolean
  runtime_error: string | null
  trade_count: number
  total_return_pct: number
  sharpe: number | null
  max_drawdown_pct: number
  fees_paid_usd: number
}

export interface MonteCarloLatencySummary {
  n_runs: number
  n_runs_succeeded: number
  sharpe_slope_per_x_latency?: number
  sharpe_at_baseline?: number | null
  sharpe_at_worst_latency?: number
  sharpe_at_best_latency?: number
  sharpe_range?: number
}

export interface MonteCarloLatencyResult {
  base_submit_p50_ms: number
  base_submit_p95_ms: number
  base_cancel_p50_ms: number
  base_cancel_p95_ms: number
  runs: LatencyPerturbationRun[]
  summary: MonteCarloLatencySummary
}

export interface MonteCarloLatencyRequest {
  source_code: string
  slug?: string
  config?: Record<string, unknown>
  token_ids?: string[]
  start: string
  end: string
  initial_capital_usd?: number
  base_submit_p50_ms?: number
  base_submit_p95_ms?: number
  base_cancel_p50_ms?: number
  base_cancel_p95_ms?: number
  multipliers?: number[]
  seed?: number
  concurrency?: number
}

export async function runMonteCarloLatency(
  req: MonteCarloLatencyRequest,
): Promise<MonteCarloLatencyResult> {
  const { data } = await api.post<MonteCarloLatencyResult>('/backtest/monte-carlo-latency', req)
  return data
}

// ─── Live-vs-backtest drift ─────────────────────────────────────────────────

export interface StrategyDriftReport {
  strategy_slug: string
  strategy_name: string | null
  severity: 'stable' | 'degraded' | 'improved' | 'stale'
  reason: string
  backtest_run_id: string | null
  backtest_completed_at: string | null
  backtest_window_days: number | null
  backtest_trade_count: number
  backtest_sharpe: number | null
  backtest_total_return_pct: number | null
  backtest_trades_per_day: number | null
  live_window_days: number
  live_trade_count: number
  live_sharpe: number | null
  live_total_pnl_usd: number
  live_hit_rate: number | null
  live_trades_per_day: number | null
  sharpe_delta: number | null
  trade_rate_ratio: number | null
}

export interface DriftMonitorResult {
  window_days: number
  generated_at: string
  strategies: StrategyDriftReport[]
  summary: {
    n_strategies: number
    by_severity: Record<string, number>
    worst_offender: {
      strategy_slug: string
      sharpe_delta: number | null
      reason: string
    } | null
  }
}

export async function getDriftMonitor(windowDays = 30): Promise<DriftMonitorResult> {
  const { data } = await api.get<DriftMonitorResult>('/backtest/drift', {
    params: { window_days: windowDays },
  })
  return data
}
