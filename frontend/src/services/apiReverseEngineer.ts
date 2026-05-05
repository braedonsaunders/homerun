/**
 * Client for /api/strategy-reverse-engineer.
 *
 * Mirrors backend/api/routes_strategy_reverse_engineer.py.  Powers the
 * Strategy Research → Reverse Engineer tab and the WalletAnalysisPanel
 * action link.
 */
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 600_000 })

export type ReverseEngineerJobStatus =
  | 'queued'
  | 'profiling'
  | 'importing_data'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type DataSourceKind = 'auto' | 'recording_session' | 'provider_dataset' | 'live'
export type ReportMode = 'report' | 'strategy_seed'

export interface WalletProfileSummary {
  trade_count: number
  unique_markets: number
  revisited_markets: number
  side_distribution: Record<string, number>
  outcome_distribution: Record<string, number>
  notional: SeriesStats
  size: SeriesStats
  price: SeriesStats
  inter_trade_seconds: SeriesStats
  hour_of_day_top: { hour: number; count: number }[]
  day_of_week_top: { dow: number; count: number }[]
}

export interface SeriesStats {
  count: number
  min: number | null
  max: number | null
  mean: number | null
  median: number | null
  p95: number | null
}

export interface WalletProfileMarket {
  market_id: string | null
  title: string | null
  trade_count: number
  total_notional_usd: number
  first_trade: string | null
  last_trade: string | null
}

export interface WalletProfile {
  address: string
  fetched_count: number
  window_start: string | null
  window_end: string | null
  summary: WalletProfileSummary
  markets: WalletProfileMarket[]
  sample_trades: Array<Record<string, unknown>>
}

export interface ReverseEngineerJob {
  id: string
  wallet_address: string
  label: string | null
  report_mode: ReportMode
  data_source_kind: DataSourceKind
  recording_session_ids: string[]
  provider_dataset_ids: string[]
  llm_model: string | null
  max_iterations: number
  target_score: number
  max_cost_usd: number | null
  max_wallet_trades: number
  status: ReverseEngineerJobStatus
  progress: number
  current_iteration: number
  activity: string | null
  error: string | null
  wallet_profile: WalletProfile | null
  wallet_trade_count: number
  wallet_window_start: string | null
  wallet_window_end: string | null
  best_iteration_id: string | null
  best_score: number | null
  best_strategy_class: string | null
  best_strategy_code: string | null
  best_backtest_run_id: string | null
  total_input_tokens: number
  total_output_tokens: number
  total_cost_usd: number
  promoted_strategy_id: string | null
  created_at: string | null
  started_at: string | null
  finished_at: string | null
}

export interface ScoreBreakdown {
  composite: number
  trade_overlap_pct: number
  side_agreement_pct: number
  pnl_correlation: number
  frequency_match: number
  timing_mae_seconds: number | null
  matched_trades: number
  backtest_trade_count: number
  wallet_trade_count: number
  weights: Record<string, number>
  notes: string[]
}

export interface ReverseEngineerIteration {
  id: string
  job_id: string
  iteration: number
  status: 'running' | 'completed' | 'failed'
  strategy_class: string | null
  strategy_code: string | null
  backtest_run_id: string | null
  score: number | null
  score_breakdown: ScoreBreakdown | null
  divergence_summary: string | null
  llm_critique: string | null
  notes: string | null
  error: string | null
  input_tokens: number
  output_tokens: number
  cost_usd: number
  duration_ms: number
  created_at: string | null
  completed_at: string | null
}

export interface CreateReverseEngineerJobRequest {
  wallet_address: string
  label?: string | null
  /** 'report' (default) → multi-section analytical PDF report;
   *  'strategy_seed' → LLM agent loop synthesizes a BaseStrategy candidate. */
  report_mode?: ReportMode
  data_source_kind?: DataSourceKind
  recording_session_ids?: string[]
  provider_dataset_ids?: string[]
  llm_model?: string | null
  max_iterations?: number | null
  target_score?: number | null
  max_cost_usd?: number | null
  max_wallet_trades?: number | null
}

export async function createReverseEngineerJob(
  req: CreateReverseEngineerJobRequest,
): Promise<ReverseEngineerJob> {
  const { data } = await api.post<ReverseEngineerJob>('/strategy-reverse-engineer/jobs', req)
  return data
}

export async function listReverseEngineerJobs(params?: {
  wallet_address?: string
  status?: ReverseEngineerJobStatus
  limit?: number
}): Promise<ReverseEngineerJob[]> {
  const { data } = await api.get<{ jobs: ReverseEngineerJob[] }>(
    '/strategy-reverse-engineer/jobs',
    { params },
  )
  return data.jobs ?? []
}

export async function getReverseEngineerJob(jobId: string): Promise<ReverseEngineerJob> {
  const { data } = await api.get<ReverseEngineerJob>(
    `/strategy-reverse-engineer/jobs/${encodeURIComponent(jobId)}`,
  )
  return data
}

export async function cancelReverseEngineerJob(
  jobId: string,
): Promise<{ cancelled: boolean; id: string }> {
  const { data } = await api.post(
    `/strategy-reverse-engineer/jobs/${encodeURIComponent(jobId)}/cancel`,
  )
  return data
}

export async function listReverseEngineerIterations(
  jobId: string,
): Promise<ReverseEngineerIteration[]> {
  const { data } = await api.get<{ iterations: ReverseEngineerIteration[] }>(
    `/strategy-reverse-engineer/jobs/${encodeURIComponent(jobId)}/iterations`,
  )
  return data.iterations ?? []
}

export async function promoteReverseEngineerJob(
  jobId: string,
  body: { name: string; slug: string; description?: string; enabled?: boolean },
): Promise<{ strategy_id: string; slug: string; name: string; enabled: boolean }> {
  const { data } = await api.post(
    `/strategy-reverse-engineer/jobs/${encodeURIComponent(jobId)}/promote`,
    body,
  )
  return data
}

export function reverseEngineerPdfUrl(jobId: string): string {
  return `/api/strategy-reverse-engineer/jobs/${encodeURIComponent(jobId)}/report.pdf`
}
