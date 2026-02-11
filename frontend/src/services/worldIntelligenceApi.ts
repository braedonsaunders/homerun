import axios from 'axios'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'

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

// ==================== API FUNCTIONS ====================

export async function getWorldSignals(params?: {
  signal_type?: string
  country?: string
  min_severity?: number
  limit?: number
}): Promise<{ signals: WorldSignal[]; total: number; last_collection: string | null }> {
  const { data } = await api.get('/world-intelligence/signals', { params })
  return data
}

export async function getInstabilityScores(params?: {
  country?: string
  min_score?: number
  limit?: number
}): Promise<{ scores: InstabilityScore[]; total: number }> {
  const { data } = await api.get('/world-intelligence/instability', { params })
  return data
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

export async function getWorldIntelligenceSummary(): Promise<WorldIntelligenceSummary> {
  const { data } = await api.get('/world-intelligence/summary')
  return data
}

export async function getWorldIntelligenceStatus(): Promise<WorldIntelligenceStatus> {
  const { data } = await api.get('/world-intelligence/status')
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
