import axios from 'axios'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'
import { normalizeCountryCode } from '../lib/worldCountries'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

api.interceptors.response.use(
  (response) => {
    normalizeUtcTimestampsInPlace(response.data)
    return response
  },
  (error) => Promise.reject(error)
)

// ==================== TYPES ====================

export interface WorldSignal {
  signal_id: string
  signal_type: 'conflict' | 'tension' | 'instability' | 'convergence' | 'anomaly' | 'military' | 'infrastructure'
  severity: number
  country: string | null
  country_iso3?: string | null
  country_name?: string | null
  latitude: number | null
  longitude: number | null
  title: string
  description: string
  source: string
  detected_at: string
  metadata: Record<string, any> | null
  related_market_ids: string[]
  market_relevance_score: number | null
}

export interface InstabilityScore {
  country: string
  iso3: string
  country_name?: string | null
  score: number
  trend: 'rising' | 'falling' | 'stable'
  change_24h: number | null
  change_7d: number | null
  components: Record<string, number>
  contributing_signals: Array<Record<string, any>>
  last_updated: string | null
}

export interface TensionPair {
  country_a: string
  country_b: string
  country_a_iso3?: string | null
  country_b_iso3?: string | null
  country_a_name?: string | null
  country_b_name?: string | null
  tension_score: number
  event_count: number
  avg_goldstein_scale: number | null
  trend: 'rising' | 'falling' | 'stable'
  top_event_types: string[]
  last_updated: string | null
}

export interface ConvergenceZone {
  grid_key: string
  latitude: number
  longitude: number
  signal_types: string[]
  signal_count: number
  urgency_score: number
  country: string
  nearby_markets: string[]
  detected_at: string | null
}

export interface WorldRegionHotspot {
  id: string
  name: string
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
  event_count?: number
  last_detected_at?: string | null
  activity_types?: string[]
}

export interface WorldRegionChokepoint {
  id: string
  name: string
  latitude: number
  longitude: number
  risk_score?: number
  nearby_signal_count?: number
  signal_breakdown?: Record<string, number>
  source?: string
  chokepoint_source?: string
  risk_method?: string
  last_updated?: string | null
  daily_metrics_date?: string | null
  daily_dataset_updated_at?: string | null
  daily_transit_total?: number | null
  daily_capacity_estimate?: number | null
  baseline_vessel_count_total?: number | null
}

export interface WorldRegionsResponse {
  version: number
  updated_at: string | null
  hotspots: WorldRegionHotspot[]
  chokepoints: WorldRegionChokepoint[]
}

export interface TemporalAnomaly {
  signal_type: string
  country: string
  z_score: number
  severity: 'normal' | 'medium' | 'high' | 'critical'
  current_value: number
  baseline_mean: number
  baseline_std: number
  description: string
  detected_at: string | null
}

export interface WorldIntelligenceSummary {
  signal_summary: Record<string, any>
  critical_countries: Array<{ country: string; iso3: string; score: number; trend: string }>
  high_tensions: Array<{ pair: string; score: number; trend: string }>
  critical_anomalies: number
  active_convergences: number
  last_collection: string | null
}

export interface WorldIntelligenceStatus {
  status: Record<string, any>
  stats: Record<string, any>
  updated_at: string | null
}

export interface WorldSourceStatusResponse {
  sources: Record<string, any>
  errors: string[]
  updated_at: string | null
}

// ==================== API FUNCTIONS ====================

function timestampValue(value: string | null | undefined): number {
  if (!value) return Number.NEGATIVE_INFINITY
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY
}

function dedupeInstabilityScores(scores: InstabilityScore[]): InstabilityScore[] {
  const byIso3 = new Map<string, InstabilityScore>()
  for (const score of scores) {
    const canonicalIso3 =
      normalizeCountryCode(score.iso3 || score.country_name || score.country)
      || String(score.iso3 || '').trim().toUpperCase()
    if (!canonicalIso3) continue
    const normalized: InstabilityScore = {
      ...score,
      iso3: canonicalIso3,
    }
    const existing = byIso3.get(canonicalIso3)
    if (!existing) {
      byIso3.set(canonicalIso3, normalized)
      continue
    }
    const existingTs = timestampValue(existing.last_updated)
    const incomingTs = timestampValue(normalized.last_updated)
    if (incomingTs > existingTs) {
      byIso3.set(canonicalIso3, normalized)
      continue
    }
    if (incomingTs === existingTs && Number(normalized.score || 0) > Number(existing.score || 0)) {
      byIso3.set(canonicalIso3, normalized)
    }
  }
  return [...byIso3.values()].sort((a, b) => Number(b.score || 0) - Number(a.score || 0))
}

export async function getWorldSignals(params?: {
  signal_type?: string
  country?: string
  min_severity?: number
  limit?: number
  offset?: number
}): Promise<{
  signals: WorldSignal[]
  total: number
  offset?: number
  limit?: number
  has_more?: boolean
  next_offset?: number | null
  last_collection: string | null
}> {
  const { data } = await api.get('/world-intelligence/signals', { params })
  return data
}

export async function getInstabilityScores(params?: {
  country?: string
  min_score?: number
  limit?: number
}): Promise<{ scores: InstabilityScore[]; total: number }> {
  const { data } = await api.get('/world-intelligence/instability', { params })
  const rawScores = Array.isArray(data?.scores) ? data.scores as InstabilityScore[] : []
  const scores = dedupeInstabilityScores(rawScores)
  return {
    ...data,
    scores,
    total: scores.length,
  }
}

export async function getTensionPairs(params?: {
  min_tension?: number
  limit?: number
}): Promise<{ tensions: TensionPair[]; total: number }> {
  const { data } = await api.get('/world-intelligence/tensions', { params })
  return data
}

export async function getConvergenceZones(): Promise<{ zones: ConvergenceZone[]; total: number }> {
  const { data } = await api.get('/world-intelligence/convergences')
  return data
}

export async function getTemporalAnomalies(params?: {
  min_severity?: string
}): Promise<{ anomalies: TemporalAnomaly[]; total: number }> {
  const { data } = await api.get('/world-intelligence/anomalies', { params })
  return data
}

export async function getWorldRegions(): Promise<WorldRegionsResponse> {
  const { data } = await api.get('/world-intelligence/regions')
  return data
}

export async function getWorldIntelligenceSummary(): Promise<WorldIntelligenceSummary> {
  const { data } = await api.get('/world-intelligence/summary')
  return data
}

export async function getWorldIntelligenceStatus(): Promise<WorldIntelligenceStatus> {
  const { data } = await api.get('/world-intelligence/status')
  return data
}

export async function getWorldSourceStatus(): Promise<WorldSourceStatusResponse> {
  const { data } = await api.get('/world-intelligence/sources')
  return data
}

export async function getMilitaryActivity(): Promise<Record<string, any>> {
  const { data } = await api.get('/world-intelligence/military')
  return data
}

export async function getInfrastructureEvents(): Promise<Record<string, any>> {
  const { data } = await api.get('/world-intelligence/infrastructure')
  return data
}
