/**
 * Client for /api/topics — the recorded-event-bus topic catalog.
 *
 * Mirrors backend/api/routes_topic_catalog.py.  Powers:
 *  • Data Lab → Topics panel (canonical "what data is in the system" view).
 *  • Backtest Studio data picker (replace per-source pickers with topic picker).
 *  • Strategy editor "subscriptions" autocomplete.
 */
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 60_000 })

export type StorageKind = 'parquet' | 'external_parquet' | 'sql_table' | 'memory'

export interface TopicSpec {
  slug: string
  title: string
  description: string | null
  storage_kind: StorageKind
  storage_uri: string | null
  schema_version: number
  retention_days: number | null
  max_bytes: number | null
  enabled: boolean
  is_replayable: boolean
  publishers: string[]
  subscribers: string[]
  last_published_at: string | null
  last_replayed_at: string | null
  event_count: number
  bytes_on_disk: number
}

export interface ListTopicsParams {
  storage_kind?: StorageKind
  enabled_only?: boolean
  replayable_only?: boolean
}

export async function listTopics(params: ListTopicsParams = {}): Promise<TopicSpec[]> {
  const r = await api.get<TopicSpec[]>('/topics', { params })
  return r.data
}

export async function getTopic(slug: string): Promise<TopicSpec | null> {
  try {
    const r = await api.get<TopicSpec>(`/topics/${encodeURIComponent(slug)}`)
    return r.data
  } catch (e: any) {
    if (e?.response?.status === 404) return null
    throw e
  }
}

export interface RegisterTopicRequest {
  slug: string
  title: string
  storage_kind: StorageKind
  storage_uri?: string | null
  description?: string | null
  payload_schema?: Record<string, unknown> | null
  schema_version?: number
  retention_days?: number | null
  publishers?: string[]
  subscribers?: string[]
  enabled?: boolean
  is_replayable?: boolean
}

export async function registerTopic(req: RegisterTopicRequest): Promise<TopicSpec> {
  const r = await api.post<TopicSpec>('/topics', req)
  return r.data
}

export async function deleteTopic(slug: string): Promise<void> {
  await api.delete(`/topics/${encodeURIComponent(slug)}`)
}

// ─── Per-topic patch (enable / retention / size cap) ─────────────────

export interface PatchTopicRequest {
  enabled?: boolean
  is_replayable?: boolean
  retention_days?: number  // 0 → clear cap
  max_bytes?: number       // 0 → clear cap
  description?: string
}

export async function patchTopic(slug: string, req: PatchTopicRequest): Promise<TopicSpec> {
  const r = await api.patch<TopicSpec>(`/topics/${encodeURIComponent(slug)}`, req)
  return r.data
}

// ─── Global rotation controls ────────────────────────────────────────

export interface BusRotationSettings {
  pruner_enabled: boolean
  global_max_bytes: number | null
  total_bytes_on_disk: number
  n_parquet_topics: number
}

export async function getBusRotationSettings(): Promise<BusRotationSettings> {
  const r = await api.get<BusRotationSettings>('/topics/settings/rotation')
  return r.data
}

export async function updateBusRotationSettings(
  req: { pruner_enabled?: boolean; global_max_bytes?: number },
): Promise<BusRotationSettings> {
  const r = await api.patch<BusRotationSettings>('/topics/settings/rotation', req)
  return r.data
}

export interface PruneReport {
  scanned_topics?: number
  freed_bytes_age?: number
  freed_bytes_per_topic?: number
  freed_bytes_global?: number
  global_cap?: number | null
  skipped?: string
}

export async function triggerPruneNow(): Promise<PruneReport> {
  const r = await api.post<PruneReport>('/topics/settings/prune-now')
  return r.data
}

export interface ReplayPreviewRequest {
  topics: string[]
  start_us: number
  end_us: number
  limit?: number
  time_field?: 'observed_at_us' | 'ingested_at_us'
}

export interface ReplayPreviewSample {
  topic: string
  entity_id: string
  observed_at_us: number
  ingested_at_us: number
  source: string
  sequence: number | null
  schema_version: number
  payload: Record<string, unknown>
}

export interface ReplayPreviewResponse {
  n_seen: number
  truncated: boolean
  samples: ReplayPreviewSample[]
}

export async function replayPreview(req: ReplayPreviewRequest): Promise<ReplayPreviewResponse> {
  const r = await api.post<ReplayPreviewResponse>('/topics/replay/preview', req)
  return r.data
}

// ─── Helpers ──────────────────────────────────────────────────────────

export function formatTopicVolume(t: TopicSpec): string {
  if (t.event_count <= 0 && t.bytes_on_disk <= 0) return '—'
  const events = t.event_count >= 1_000_000
    ? `${(t.event_count / 1_000_000).toFixed(1)}M events`
    : t.event_count >= 1_000
    ? `${(t.event_count / 1_000).toFixed(1)}K events`
    : `${t.event_count} events`
  const sizeMB = t.bytes_on_disk / (1024 * 1024)
  if (sizeMB <= 0) return events
  if (sizeMB >= 1024) return `${events} · ${(sizeMB / 1024).toFixed(2)} GB`
  if (sizeMB >= 1) return `${events} · ${sizeMB.toFixed(1)} MB`
  return `${events} · ${(sizeMB * 1024).toFixed(0)} KB`
}

export function topicStorageBadgeColor(kind: StorageKind): string {
  switch (kind) {
    case 'parquet':   return '#16a34a'  // green — operator-managed canonical store
    case 'sql_table': return '#0891b2'  // teal  — wraps existing table
    case 'memory':    return '#a855f7'  // purple — live-only
    default:          return '#64748b'
  }
}
