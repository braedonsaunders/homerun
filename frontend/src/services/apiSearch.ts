import { api, unwrapApiData } from './apiClient'

// ==================== GLOBAL SEARCH ====================
//
// World-class FTS over the entire system, served by Postgres
// GIN/tsvector + pg_trgm.  See backend/services/search/service.py
// for the SQL — composite ranking blends BM25-style FTS rank with
// trigram similarity, liquidity / volume signals, and recency.
//
// One row per searchable entity, grouped by ``entity_type`` so the
// command-palette UI can render section headers per type.

export type SearchEntityType =
  | 'market'
  | 'event'
  | 'category'
  | 'strategy'
  | 'data_source'
  | 'trader'
  | 'wallet'
  | 'news'
  | 'alert'
  | 'research'
  | 'opportunity'

export interface SearchResultItem {
  entity_type: SearchEntityType | string
  entity_id: string
  title: string
  subtitle: string | null
  category: string | null
  tags: string[]
  metadata: Record<string, any>
  liquidity: number | null
  volume: number | null
  recency: string | null
  /** Server-rendered HTML snippet with <mark>matching tokens</mark>. */
  snippet: string
  /** Composite rank score; higher = better. Useful for client-side debug. */
  score: number
  fts_rank: number
  trigram_similarity: number
}

export interface SearchGlobalResponse {
  query: string
  results: SearchResultItem[]
  total: number
  groups: Record<string, SearchResultItem[]>
  latency_ms: number
}

export const searchGlobal = async (
  q: string,
  opts: { limit?: number; types?: string[] } = {}
): Promise<SearchGlobalResponse> => {
  const { limit = 30, types } = opts
  const params: Record<string, string | number> = { q, limit }
  if (types && types.length > 0) {
    params.types = types.join(',')
  }
  const { data } = await api.get('/search/global', { params })
  return unwrapApiData(data)
}

export interface RecentSearchEntry {
  query: string
  last_at: string | null
  result_count: number
}

export const getRecentSearches = async (limit = 10): Promise<{ queries: RecentSearchEntry[]; total: number }> => {
  const { data } = await api.get('/search/recent', { params: { limit } })
  return unwrapApiData(data)
}

export interface SearchStatsByType {
  entity_type: string
  count: number
  last_updated: string | null
}

export interface SearchStatsResponse {
  total: number
  by_type: SearchStatsByType[]
}

export const getSearchStats = async (): Promise<SearchStatsResponse> => {
  const { data } = await api.get('/search/stats')
  return unwrapApiData(data)
}

export const reindexSearch = async (entityType?: string): Promise<any> => {
  const { data } = await api.post('/search/reindex', entityType ? { entity_type: entityType } : {})
  return unwrapApiData(data)
}

export const getSearchTypes = async (): Promise<{ types: string[] }> => {
  const { data } = await api.get('/search/types')
  return unwrapApiData(data)
}
