/**
 * Client for /api/providers — external market-data provider integration.
 *
 * Mirrors backend/api/routes_providers.py.  Powers Data Lab → Providers
 * tab and the Backtest Studio dataset picker.
 */
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 600_000 })

// ─── Provider catalog ─────────────────────────────────────────────────

export interface ProviderHealth {
  configured: boolean
  ok?: boolean
  status_code?: number
  elapsed_ms?: number | null
  error?: string
}

export interface ProviderInfo {
  key: string
  label: string
  description: string
  homepage: string
  docs_url: string
  asset_classes: string[]
  supported_coins: string[]
  configured: boolean
  health: ProviderHealth
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const { data } = await api.get<{ providers: ProviderInfo[] }>('/providers')
  return data.providers ?? []
}

// ─── Polybacktest market browser ─────────────────────────────────────

export interface PolybacktestMarket {
  market_id: string
  slug: string | null
  /** Synthesized human title — always populated.
   *  e.g. "BTC Up/Down · 5m · 2026-05-04 12:30 UTC (open $80,149.13)". */
  title: string
  market_type: string | null
  start_time: string | null
  end_time: string | null
  winner: 'Up' | 'Down' | null
  final_volume: number | null
  final_liquidity: number | null
  coin_price_start: number | null
  coin_price_end: number | null
}

export interface PolybacktestMarketsPage {
  coin: string
  total: number
  limit: number
  offset: number
  markets: PolybacktestMarket[]
}

export async function listPolybacktestMarkets(params: {
  coin: string
  offset?: number
  search?: string
  market_type?: '5m' | '15m' | '1h' | '4h' | '24h'
  resolved?: boolean
  limit?: number
}): Promise<PolybacktestMarketsPage> {
  const { data } = await api.get<PolybacktestMarketsPage>(
    '/providers/polybacktest/markets',
    { params },
  )
  return data
}

// ─── Imports ──────────────────────────────────────────────────────────

export type ImportJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface ImportJob {
  id: string
  provider: string
  status: ImportJobStatus
  progress: number
  message: string | null
  payload: Record<string, unknown>
  result: Record<string, unknown> | null
  error: string | null
  snapshots_fetched: number
  snapshots_inserted: number
  trades_fetched: number
  api_calls: number
  bytes_downloaded: number
  created_at: string | null
  started_at: string | null
  finished_at: string | null
}

export interface PolybacktestImportRequest {
  coin: string
  market_ids: string[]
  start: string
  end: string
}

export async function importPolybacktest(req: PolybacktestImportRequest): Promise<ImportJob> {
  const { data } = await api.post<ImportJob>('/providers/polybacktest/import', req)
  return data
}

export async function listImportJobs(params?: {
  provider?: string
  status?: ImportJobStatus
  limit?: number
}): Promise<ImportJob[]> {
  const { data } = await api.get<{ jobs: ImportJob[] }>('/providers/import', { params })
  return data.jobs ?? []
}

export async function getImportJob(jobId: string): Promise<ImportJob> {
  const { data } = await api.get<ImportJob>(`/providers/import/${encodeURIComponent(jobId)}`)
  return data
}

export async function cancelImportJob(jobId: string): Promise<{ cancelled: boolean; id: string }> {
  const { data } = await api.post(`/providers/import/${encodeURIComponent(jobId)}/cancel`)
  return data
}

// ─── Imported datasets catalog ───────────────────────────────────────

export interface ProviderDataset {
  id: string
  provider: string
  coin: string | null
  external_id: string
  external_slug: string | null
  title: string | null
  asset_class: string
  token_ids: string[]
  start_ts: string | null
  end_ts: string | null
  snapshot_count: number
  trade_count: number
  last_imported_at: string | null
  last_import_job_id: string | null
  created_at: string | null
  updated_at: string | null
  payload?: Record<string, unknown>
}

export async function listProviderDatasets(params?: {
  provider?: string
  coin?: string
  limit?: number
}): Promise<ProviderDataset[]> {
  const { data } = await api.get<{ datasets: ProviderDataset[] }>('/providers/datasets', { params })
  return data.datasets ?? []
}

export async function getProviderDataset(id: string): Promise<ProviderDataset> {
  const { data } = await api.get<ProviderDataset>(`/providers/datasets/${encodeURIComponent(id)}`)
  return data
}

export async function deleteProviderDataset(id: string): Promise<{ deleted: boolean; id: string }> {
  const { data } = await api.delete(`/providers/datasets/${encodeURIComponent(id)}`)
  return data
}

export interface ProviderDatasetScope {
  dataset_ids: string[]
  labels: (string | null)[]
  token_ids: string[]
  start: string | null
  end: string | null
}

export async function resolveProviderDatasetScope(
  datasetIds: string[],
): Promise<ProviderDatasetScope> {
  const { data } = await api.post<ProviderDatasetScope>('/providers/datasets/scope', {
    dataset_ids: datasetIds,
  })
  return data
}

// ─── Provider settings (API key + reverse-engineer defaults) ─────────

export interface ProviderSettings {
  polybacktest_api_key_set: boolean
  polybacktest_base_url: string | null
  // Default LLM model for the reverse-engineer agent lives in the
  // canonical AI → Models view (llm_model_assignments['strategy_reverse_engineer']).
  reverse_engineer_max_iterations: number | null
  reverse_engineer_target_score: number | null
  reverse_engineer_max_cost_usd: number | null
  reverse_engineer_max_wallet_trades: number | null
}

export interface ProviderSettingsUpdate {
  polybacktest_api_key?: string | null
  polybacktest_base_url?: string | null
  reverse_engineer_max_iterations?: number | null
  reverse_engineer_target_score?: number | null
  reverse_engineer_max_cost_usd?: number | null
  reverse_engineer_max_wallet_trades?: number | null
}

export async function getProviderSettings(): Promise<ProviderSettings> {
  const { data } = await api.get<ProviderSettings>('/providers/settings')
  return data
}

export async function updateProviderSettings(
  body: ProviderSettingsUpdate,
): Promise<{ ok: boolean }> {
  const { data } = await api.put<{ ok: boolean }>('/providers/settings', body)
  return data
}


// ─── Parquet datasets (operator-supplied vendor data) ────────────────

export interface ParquetRoot {
  root: string
  exists: boolean
  env_var: string
}

export interface ParquetDataset {
  id: string
  provider: string
  coin: string | null
  title: string | null
  start_ts: string | null
  end_ts: string | null
  token_count: number
  snapshot_count: number
  trade_count: number
  storage_uri: string
  last_imported_at: string | null
}

export interface ParquetRescanResult {
  provider?: string
  coin?: string
  window?: string
  id?: string
  tokens?: number
  snapshot_files?: number
  delta_files?: number
  snapshot_rows?: number
  delta_rows?: number
  errors?: string[]
  skipped?: boolean
  reason?: string
  error?: string
}

export interface ParquetRescanReport {
  root: string
  groups_seen: number
  results: ParquetRescanResult[]
  elapsed_ms: number
  scanned_at_epoch: number
}

export async function getParquetRoot(): Promise<ParquetRoot> {
  const { data } = await api.get<ParquetRoot>('/providers/parquet/root')
  return data
}

export async function listParquetDatasets(): Promise<ParquetDataset[]> {
  const { data } = await api.get<{ count: number; datasets: ParquetDataset[] }>(
    '/providers/parquet/datasets',
  )
  return data.datasets ?? []
}

export async function rescanParquetRoot(): Promise<ParquetRescanReport> {
  const { data } = await api.post<ParquetRescanReport>('/providers/parquet/rescan')
  return data
}
