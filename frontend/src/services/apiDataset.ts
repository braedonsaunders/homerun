import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 600000 })

export type DatasetColumnType = 'string' | 'int' | 'float' | 'datetime' | 'json' | 'enum'

export interface DatasetColumn {
  key: string
  label: string
  type: DatasetColumnType
  sortable: boolean
  default_visible: boolean
  enum_values: string[] | null
  description: string
}

export type DatasetFilterKind =
  | 'eq'
  | 'contains'
  | 'time_range_start'
  | 'time_range_end'
  | 'enum_in'

export interface DatasetFilter {
  key: string
  column: string
  label: string
  kind: DatasetFilterKind
  description: string
}

export interface DatasetSummary {
  name: string
  label: string
  description: string
  row_count: number
  default_sort: string
  default_sort_dir: 'asc' | 'desc'
  columns: DatasetColumn[]
  filters: DatasetFilter[]
}

export interface DatasetQueryResult {
  dataset: string
  label: string
  total: number
  limit: number
  offset: number
  order_by: string
  order_dir: 'asc' | 'desc'
  columns: DatasetColumn[]
  filters: DatasetFilter[]
  rows: Array<Record<string, unknown>>
}

export type DatasetFilterValues = Record<string, string | string[] | undefined>

export interface DatasetQueryParams {
  limit?: number
  offset?: number
  order_by?: string
  order_dir?: 'asc' | 'desc'
  filters?: DatasetFilterValues
}

export async function listDatasets(): Promise<DatasetSummary[]> {
  const { data } = await api.get<{ datasets: DatasetSummary[] }>('/dataset')
  return data.datasets ?? []
}

function buildParams(p: DatasetQueryParams): Record<string, string | number> {
  const out: Record<string, string | number> = {}
  if (p.limit != null) out.limit = p.limit
  if (p.offset != null) out.offset = p.offset
  if (p.order_by) out.order_by = p.order_by
  if (p.order_dir) out.order_dir = p.order_dir
  if (p.filters) {
    for (const [k, v] of Object.entries(p.filters)) {
      if (v == null || v === '') continue
      if (Array.isArray(v)) {
        if (v.length > 0) out[k] = v.join(',')
      } else {
        out[k] = v
      }
    }
  }
  return out
}

export async function queryDataset(
  name: string,
  params: DatasetQueryParams = {},
): Promise<DatasetQueryResult> {
  const { data } = await api.get<DatasetQueryResult>(`/dataset/${encodeURIComponent(name)}`, {
    params: buildParams(params),
  })
  return data
}

// ─── Recording sessions ─────────────────────────────────────────────────

export type RecordingSessionStatus =
  | 'pending'
  | 'scheduled'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type RecordingTargetKind = 'token' | 'condition' | 'event'
export type RecordingCaptureType = 'book' | 'trade' | 'delta'

export interface RecordingSession {
  id: string
  name: string
  description: string | null
  status: RecordingSessionStatus
  platform: string
  target_kind: RecordingTargetKind
  target_values: string[]
  target_token_ids: string[]
  capture_types: RecordingCaptureType[]
  tick_interval_ms: number
  retention_days: number | null
  scheduled_start_at: string | null
  scheduled_end_at: string | null
  max_duration_seconds: number | null
  started_at: string | null
  ended_at: string | null
  rows_captured: number
  last_capture_at: string | null
  error: string | null
  config: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface CreateRecordingSessionPayload {
  name: string
  description?: string
  platform?: string
  target_kind?: RecordingTargetKind
  target_values: string[]
  capture_types?: RecordingCaptureType[]
  tick_interval_ms?: number
  retention_days?: number | null
  scheduled_start_at?: string | null
  scheduled_end_at?: string | null
  max_duration_seconds?: number | null
  config?: Record<string, unknown> | null
}

export async function listRecordingSessions(
  statuses?: RecordingSessionStatus[],
  limit = 100,
): Promise<RecordingSession[]> {
  const params: Record<string, string | number> = { limit }
  if (statuses && statuses.length > 0) params.statuses = statuses.join(',')
  const { data } = await api.get<{ sessions: RecordingSession[] }>('/dataset/sessions', { params })
  return data.sessions ?? []
}

export async function createRecordingSession(
  payload: CreateRecordingSessionPayload,
): Promise<RecordingSession> {
  const { data } = await api.post<RecordingSession>('/dataset/sessions', payload)
  return data
}

export async function getRecordingSession(id: string): Promise<RecordingSession> {
  const { data } = await api.get<RecordingSession>(`/dataset/sessions/${encodeURIComponent(id)}`)
  return data
}

export async function startRecordingSession(id: string): Promise<RecordingSession> {
  const { data } = await api.post<RecordingSession>(`/dataset/sessions/${encodeURIComponent(id)}/start`)
  return data
}

export async function stopRecordingSession(id: string): Promise<RecordingSession> {
  const { data } = await api.post<RecordingSession>(`/dataset/sessions/${encodeURIComponent(id)}/stop`)
  return data
}

export async function cancelRecordingSession(id: string): Promise<RecordingSession> {
  const { data } = await api.post<RecordingSession>(`/dataset/sessions/${encodeURIComponent(id)}/cancel`)
  return data
}

export async function deleteRecordingSession(id: string): Promise<void> {
  await api.delete(`/dataset/sessions/${encodeURIComponent(id)}`)
}

// ─── Manual REST backfill ──────────────────────────────────────────────

export type BackfillScope = 'token' | 'strategy' | 'session' | 'catalog_top_liquid'

export interface BackfillRequest {
  scope: BackfillScope
  target_values?: string[]
  strategy_slug?: string
  session_id?: string
  start?: string
  end?: string
  interval?: string
  fidelity_minutes?: number
  synthetic_spread_bps?: number
  catalog_max_tokens?: number
  catalog_min_liquidity_usd?: number
  concurrency?: number
  max_tokens?: number
}

export interface BackfillTokenResult {
  token_id: string
  rows_inserted: number
  skipped_existing: number
  points_fetched: number
  error: string | null
}

export interface BackfillResult {
  job_id: string
  scope: BackfillScope
  started_at: string
  completed_at: string
  duration_seconds: number
  target_token_count: number
  tokens_with_data: number
  tokens_with_errors: number
  rows_inserted_total: number
  points_fetched_total: number
  skipped_existing_total: number
  interval: string
  fidelity_minutes: number | null
  start: string | null
  end: string | null
  synthetic_spread_bps: number
  per_token: BackfillTokenResult[]
  error: string | null
}

export async function runRecorderBackfill(req: BackfillRequest): Promise<BackfillResult> {
  const { data } = await api.post<BackfillResult>('/dataset/recorder/backfill', req)
  return data
}


export interface ProactiveSubscriptionStatus {
  max_tokens: number
  min_liquidity_usd: number
  loop_interval_seconds: number
  last_run_at_epoch: number | null
  last_run_age_seconds: number | null
  last_run_duration_ms: number
  last_run_subscribed_count: number
  last_run_target_count: number
  last_run_catalog_market_count: number
  last_run_catalog_token_count: number
  last_run_dropped_low_liquidity: number
  last_run_dropped_over_cap: number
  last_run_already_subscribed: number
  last_error: string | null
  total_runs: number
}

export async function getProactiveSubscriptionStatus(): Promise<ProactiveSubscriptionStatus> {
  const { data } = await api.get<ProactiveSubscriptionStatus>(
    '/dataset/recorder/proactive-subscription',
  )
  return data
}

export interface MicrostructureRecorderStatus {
  running: boolean
  tokens_tracked: number
  accepted_books: number
  total_attempts: number
  accept_rate: number | null
  rejects_by_reason: Record<string, number>
  sequence_gaps_observed: number
  /** Aggregate of snapshot + delta queue drops (kept for backwards-compat). */
  queue_dropped: number
  /** Snapshot persistence queue drops (full L2 books → mms). */
  snapshot_queue_dropped?: number
  /** Delta persistence queue drops (per-level changes → bde). */
  delta_queue_dropped?: number
  /** Persistence-task flush latency p50 (ms). */
  flush_latency_ms_p50?: number | null
  /** Persistence-task flush latency p95 (ms). */
  flush_latency_ms_p95?: number | null
  error?: string
}

export async function getMicrostructureRecorderStatus(): Promise<MicrostructureRecorderStatus> {
  const { data } = await api.get<MicrostructureRecorderStatus>('/dataset/recorder/microstructure')
  return data
}

export interface DatasetStorageRow {
  name: string
  label: string
  table_name: string
  row_count: number
  size_bytes: number | null
  oldest_at: string | null
  newest_at: string | null
}

export interface DatasetStorageSummary {
  tables: DatasetStorageRow[]
  total_rows: number
  total_bytes: number | null
}

export async function getDatasetStorageSummary(): Promise<DatasetStorageSummary> {
  const { data } = await api.get<DatasetStorageSummary>('/dataset/storage/summary')
  return data
}

export async function getDatasetDistinct(
  name: string,
  column: string,
  limit = 200,
): Promise<string[]> {
  const { data } = await api.get<{ column: string; values: unknown[] }>(
    `/dataset/${encodeURIComponent(name)}/distinct/${encodeURIComponent(column)}`,
    { params: { limit } },
  )
  return (data.values ?? []).map((v) => String(v))
}

/** Build a CSV-export URL the browser can navigate to (triggers download). */
export function datasetCsvUrl(
  name: string,
  params: DatasetQueryParams & { columns?: string[]; max_rows?: number } = {},
): string {
  const q = new URLSearchParams()
  const built = buildParams(params)
  for (const [k, v] of Object.entries(built)) {
    q.set(k, String(v))
  }
  if (params.columns && params.columns.length > 0) {
    q.set('columns', params.columns.join(','))
  }
  if (params.max_rows != null) {
    q.set('max_rows', String(params.max_rows))
  }
  return `/api/dataset/${encodeURIComponent(name)}/csv?${q.toString()}`
}
