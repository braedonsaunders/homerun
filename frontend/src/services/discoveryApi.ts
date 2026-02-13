import axios from 'axios'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'

const API_BASE = '/api/discovery'
const discoveryHttp = axios.create({
  timeout: 60000,
})

discoveryHttp.interceptors.response.use((response) => {
  normalizeUtcTimestampsInPlace(response.data)
  return response
})

export interface DiscoveredWallet {
  address: string
  username: string | null
  total_trades: number
  wins: number
  losses: number
  win_rate: number
  total_pnl: number
  avg_roi: number
  sharpe_ratio: number | null
  sortino_ratio: number | null
  max_drawdown: number | null
  profit_factor: number | null
  calmar_ratio: number | null
  rolling_pnl: Record<string, number> | null
  rolling_roi: Record<string, number> | null
  rolling_win_rate: Record<string, number> | null
  rolling_trade_count: Record<string, number> | null
  rolling_sharpe: Record<string, number | null> | null
  anomaly_score: number
  is_bot: boolean
  is_profitable: boolean
  recommendation: string
  rank_score: number
  rank_position: number | null
  quality_score?: number
  activity_score?: number
  stability_score?: number
  composite_score?: number
  last_trade_at?: string | null
  trades_1h?: number
  trades_24h?: number
  unique_markets_24h?: number
  in_top_pool?: boolean
  pool_tier?: string | null
  pool_membership_reason?: string | null
  source_flags?: Record<string, boolean>
  market_categories?: string[]
  insider_score?: number
  insider_confidence?: number
  insider_sample_size?: number
  insider_last_scored_at?: string | null
  insider_metrics?: Record<string, unknown> | null
  insider_reasons?: string[]
  tags: string[]
  cluster_id: string | null
  strategies_detected: string[]
  days_active: number
  trades_per_day: number
  unique_markets: number
  last_analyzed_at: string | null
  // Period-specific metrics (populated when time_period filter is active)
  period_pnl?: number
  period_roi?: number
  period_win_rate?: number
  period_trades?: number
  period_sharpe?: number | null
}

export interface ConfluenceSignal {
  id: string
  market_id: string
  market_question: string | null
  market_slug: string | null
  yes_price?: number | null
  no_price?: number | null
  price_history?: Array<{
    t: number
    yes: number
    no: number
  }>
  signal_type: string
  strength: number
  conviction_score?: number
  tier?: string
  window_minutes?: number
  wallet_count: number
  cluster_adjusted_wallet_count?: number
  unique_core_wallets?: number
  weighted_wallet_score?: number
  wallets: string[]
  outcome: string | null
  avg_entry_price: number | null
  total_size: number | null
  avg_wallet_rank: number | null
  net_notional?: number | null
  conflicting_notional?: number | null
  market_liquidity?: number | null
  market_volume_24h?: number | null
  is_active: boolean
  first_seen_at?: string | null
  last_seen_at?: string | null
  detected_at: string
}

export interface WalletCluster {
  id: string
  label: string | null
  confidence: number
  total_wallets: number
  combined_pnl: number
  combined_trades: number
  avg_win_rate: number
  wallets: DiscoveredWallet[]
}

export interface TagInfo {
  name: string
  display_name: string
  description: string
  category: string
  color: string
  wallet_count: number
}

export interface DiscoveryStats {
  total_discovered: number
  total_profitable: number
  total_copy_candidates: number
  last_run_at: string | null
  wallets_discovered_last_run?: number
  wallets_analyzed_last_run?: number
  is_running: boolean
  current_activity?: string | null
  interval_minutes?: number
  paused?: boolean
  requested_run_at?: string | null
}

export interface PoolStats {
  target_pool_size: number
  min_pool_size: number
  max_pool_size: number
  pool_size: number
  active_1h: number
  active_24h: number
  active_1h_pct: number
  active_24h_pct: number
  churn_rate: number
  last_pool_recompute_at: string | null
  freshest_trade_at: string | null
  stale_floor_trade_at: string | null
}

export interface TrackedTraderOpportunity extends ConfluenceSignal {
  top_wallets?: Array<{
    address: string
    username: string | null
    rank_score: number
    composite_score: number
    quality_score: number
    activity_score: number
  }>
  outcome_labels?: string[]
  yes_label?: string | null
  no_label?: string | null
  current_yes_price?: number | null
  current_no_price?: number | null
  source_flags?: {
    from_pool?: boolean
    from_tracked_traders?: boolean
    from_trader_groups?: boolean
    qualified?: boolean
  }
  source_breakdown?: {
    wallets_considered?: number
    pool_wallets?: number
    tracked_wallets?: number
    group_wallets?: number
    group_count?: number
    group_ids?: string[]
  }
  validation?: {
    is_valid?: boolean
    is_actionable?: boolean
    is_tradeable?: boolean
    checks?: Record<string, boolean>
    reasons?: string[]
  }
  is_valid?: boolean
  is_actionable?: boolean
  is_tradeable?: boolean
  validation_reasons?: string[]
}

export interface InsiderOpportunity {
  id: string
  signal_key?: string | null
  market_id: string
  market_question: string
  market_slug?: string | null
  yes_price?: number | null
  no_price?: number | null
  price_history?: Array<{
    t: number
    yes: number
    no: number
  }>
  direction: 'buy_yes' | 'buy_no' | string
  entry_price?: number | null
  edge_percent?: number | null
  confidence: number
  insider_score?: number | null
  wallet_addresses: string[]
  wallet_count: number
  cluster_count: number
  pre_news_lead_minutes?: number
  timing_alpha_short?: number
  market_liquidity?: number
  suggested_size_usd?: number | null
  status?: string
  freshness_minutes?: number
  top_wallet?: {
    address: string
    username?: string | null
    insider_score?: number
    insider_confidence?: number
  } | null
  wallets?: Array<{
    address: string
    username?: string | null
    insider_score?: number
    insider_confidence?: number
  }>
  created_at?: string | null
  outcome_labels?: string[]
  yes_label?: string | null
  no_label?: string | null
  current_yes_price?: number | null
  current_no_price?: number | null
  source_flags?: {
    from_pool?: boolean
    from_tracked_traders?: boolean
    from_trader_groups?: boolean
    qualified?: boolean
  }
  source_breakdown?: {
    wallets_considered?: number
    pool_wallets?: number
    tracked_wallets?: number
    group_wallets?: number
    group_count?: number
    group_ids?: string[]
  }
  validation?: {
    is_valid?: boolean
    is_actionable?: boolean
    is_tradeable?: boolean
    checks?: Record<string, boolean>
    reasons?: string[]
  }
  is_valid?: boolean
  is_actionable?: boolean
  is_tradeable?: boolean
  validation_reasons?: string[]
}

export interface InsiderIntent {
  id: string
  signal_key?: string | null
  market_id: string
  market_question: string
  direction: string
  entry_price?: number | null
  edge_percent?: number | null
  confidence?: number | null
  insider_score?: number | null
  wallet_addresses?: string[]
  suggested_size_usd?: number | null
  metadata?: Record<string, unknown>
  status: 'pending' | 'submitted' | 'executed' | 'skipped' | 'expired' | string
  created_at?: string | null
  consumed_at?: string | null
}

export interface TraderGroupMember {
  id: string
  wallet_address: string
  source: string
  confidence?: number | null
  notes?: string | null
  added_at?: string | null
  username?: string | null
  composite_score?: number | null
  quality_score?: number | null
  activity_score?: number | null
  pool_tier?: string | null
}

export interface TraderGroup {
  id: string
  name: string
  description?: string | null
  source_type: string
  suggestion_key?: string | null
  criteria?: Record<string, unknown>
  auto_track_members?: boolean
  member_count: number
  members?: TraderGroupMember[]
  created_at?: string | null
  updated_at?: string | null
}

export interface TraderGroupSuggestion {
  id: string
  kind: 'cluster' | 'pool_tier' | 'tag' | string
  name: string
  description: string
  wallet_count: number
  wallet_addresses: string[]
  avg_composite_score?: number
  already_exists?: boolean
  criteria?: Record<string, unknown>
  sample_wallets?: Array<{
    address: string
    username?: string | null
    composite_score?: number | null
    pool_tier?: string | null
  }>
}

export interface TradersOverview {
  tracked: {
    wallets: Array<{
      address: string
      label?: string | null
      username?: string | null
      recent_trade_count: number
      latest_trade_at?: string | null
      open_positions: number
    }>
    total_wallets: number
    hours_window: number
    recent_trade_count: number
  }
  groups: {
    items: TraderGroup[]
    total_groups: number
    total_members: number
  }
  confluence: {
    signals: TrackedTraderOpportunity[]
    total_signals: number
    min_tier: string
  }
}

export interface PoolMember {
  address: string
  username?: string | null
  display_name?: string | null
  name_source?: 'username' | 'tracked_label' | 'cluster_label' | 'unresolved' | string
  tracked_label?: string | null
  cluster_label?: string | null
  in_top_pool: boolean
  pool_tier?: string | null
  pool_membership_reason?: string | null
  rank_score: number
  composite_score: number
  quality_score: number
  activity_score: number
  stability_score?: number
  selection_score?: number
  selection_rank?: number | null
  selection_percentile?: number | null
  selection_reasons?: Array<{
    code: string
    label: string
    detail?: string
  }>
  selection_breakdown?: Record<string, number>
  selection_updated_at?: string | null
  trades_1h: number
  trades_24h: number
  last_trade_at?: string | null
  total_trades: number
  total_pnl: number
  win_rate: number
  tags: string[]
  strategies_detected: string[]
  market_categories: string[]
  tracked_wallet: boolean
  pool_flags: {
    manual_include: boolean
    manual_exclude: boolean
    blacklisted: boolean
  }
}

export interface PoolMembersResponse {
  total: number
  offset: number
  limit: number
  members: PoolMember[]
  stats: {
    pool_members: number
    blacklisted: number
    manual_included: number
    manual_excluded: number
    tracked_in_pool: number
    tracked_total: number
  }
}

export const discoveryApi = {
  getLeaderboard: async (params: {
    limit?: number
    offset?: number
    min_trades?: number
    min_pnl?: number
    insider_only?: boolean
    min_insider_score?: number
    sort_by?: string
    sort_dir?: string
    tags?: string
    recommendation?: string
    time_period?: string
    active_within_hours?: number
    min_activity_score?: number
    pool_only?: boolean
    tier?: string
    search?: string
    unique_entities_only?: boolean
    market_category?: string
  } = {}) => {
    const { data } = await discoveryHttp.get(`${API_BASE}/leaderboard`, { params })
    return data
  },

  getDiscoveryStats: async (): Promise<DiscoveryStats> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/leaderboard/stats`)
    return data
  },

  getWalletProfile: async (address: string) => {
    const { data } = await discoveryHttp.get(`${API_BASE}/wallet/${address}/profile`)
    return data
  },

  triggerDiscovery: async (maxMarkets = 50, maxWalletsPerMarket = 30) => {
    const { data } = await discoveryHttp.post(`${API_BASE}/run`, null, {
      params: { max_markets: maxMarkets, max_wallets_per_market: maxWalletsPerMarket },
    })
    return data
  },

  refreshLeaderboard: async () => {
    const { data } = await discoveryHttp.post(`${API_BASE}/refresh-leaderboard`)
    return data
  },

  getConfluenceSignals: async (minStrength = 0, limit = 50): Promise<ConfluenceSignal[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/confluence`, {
      params: { min_strength: minStrength, limit, min_tier: 'WATCH' },
    })
    return data.signals || []
  },

  triggerConfluenceScan: async () => {
    const { data } = await discoveryHttp.post(`${API_BASE}/confluence/scan`)
    return data
  },

  getPoolStats: async (): Promise<PoolStats> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/pool/stats`)
    return data
  },

  getPoolMembers: async (params: {
    limit?: number
    offset?: number
    pool_only?: boolean
    include_blacklisted?: boolean
    tier?: 'core' | 'rising'
    search?: string
    min_win_rate?: number
    sort_by?: 'selection_score' | 'composite_score' | 'quality_score' | 'activity_score' | 'trades_24h' | 'trades_1h' | 'total_trades' | 'total_pnl' | 'win_rate' | 'last_trade_at' | 'rank_score'
    sort_dir?: 'asc' | 'desc'
  } = {}): Promise<PoolMembersResponse> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/pool/members`, { params })
    return data
  },

  poolManualInclude: async (address: string, reason?: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/pool/members/${address}/manual-include`, reason ? { reason } : undefined)
    return data
  },

  clearPoolManualInclude: async (address: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.delete(`${API_BASE}/pool/members/${address}/manual-include`)
    return data
  },

  poolManualExclude: async (address: string, reason?: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/pool/members/${address}/manual-exclude`, reason ? { reason } : undefined)
    return data
  },

  clearPoolManualExclude: async (address: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.delete(`${API_BASE}/pool/members/${address}/manual-exclude`)
    return data
  },

  blacklistPoolWallet: async (address: string, reason?: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/pool/members/${address}/blacklist`, reason ? { reason } : undefined)
    return data
  },

  unblacklistPoolWallet: async (address: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.delete(`${API_BASE}/pool/members/${address}/blacklist`)
    return data
  },

  deletePoolWallet: async (address: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.delete(`${API_BASE}/pool/members/${address}`)
    return data
  },

  promoteTrackedWalletsToPool: async (limit = 300): Promise<{ status: string; promoted: number; created: number; updated: number }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/pool/actions/promote-tracked`, null, {
      params: { limit },
    })
    return data
  },

  getTrackedTraderOpportunities: async (
    limit = 50,
    minTier: 'WATCH' | 'HIGH' | 'EXTREME' = 'WATCH'
  ): Promise<TrackedTraderOpportunity[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/opportunities/tracked-traders`, {
      params: { limit, min_tier: minTier },
    })
    return data.opportunities || []
  },

  getInsiderOpportunities: async (params: {
    limit?: number
    offset?: number
    min_confidence?: number
    direction?: 'buy_yes' | 'buy_no'
    max_age_minutes?: number
  } = {}): Promise<{ total: number; offset: number; limit: number; opportunities: InsiderOpportunity[] }> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/opportunities/insider`, { params })
    return {
      total: data.total || 0,
      offset: data.offset || 0,
      limit: data.limit || params.limit || 50,
      opportunities: data.opportunities || [],
    }
  },

  getInsiderIntents: async (
    status_filter?: InsiderIntent['status'],
    limit = 100,
  ): Promise<InsiderIntent[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/insider/intents`, {
      params: { status_filter, limit },
    })
    return data.intents || []
  },

  getTradersOverview: async (params: {
    tracked_limit?: number
    confluence_limit?: number
    min_tier?: 'WATCH' | 'HIGH' | 'EXTREME'
    hours?: number
  } = {}): Promise<TradersOverview> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/traders`, { params })
    return data
  },

  getTraderGroups: async (includeMembers = false, memberLimit = 25): Promise<TraderGroup[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/groups`, {
      params: { include_members: includeMembers, member_limit: memberLimit },
    })
    return data.groups || []
  },

  createTraderGroup: async (payload: {
    name: string
    description?: string
    wallet_addresses?: string[]
    source_type?: 'manual' | 'suggested_cluster' | 'suggested_tag' | 'suggested_pool'
    suggestion_key?: string
    criteria?: Record<string, unknown>
    auto_track_members?: boolean
    source_label?: string
  }): Promise<{ status: string; group: TraderGroup | null; tracked_members: number }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/groups`, payload)
    return data
  },

  addTraderGroupMembers: async (
    groupId: string,
    payload: {
      wallet_addresses: string[]
      add_to_tracking?: boolean
      source_label?: string
    },
  ): Promise<{ status: string; added_members: number; tracked_members: number; group: TraderGroup | null }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/groups/${groupId}/members`, payload)
    return data
  },

  removeTraderGroupMember: async (groupId: string, walletAddress: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.delete(`${API_BASE}/groups/${groupId}/members/${walletAddress}`)
    return data
  },

  deleteTraderGroup: async (groupId: string): Promise<{ status: string }> => {
    const { data } = await discoveryHttp.delete(`${API_BASE}/groups/${groupId}`)
    return data
  },

  trackTraderGroupMembers: async (groupId: string): Promise<{ status: string; tracked_members: number }> => {
    const { data } = await discoveryHttp.post(`${API_BASE}/groups/${groupId}/track`)
    return data
  },

  getTraderGroupSuggestions: async (params: {
    min_group_size?: number
    max_suggestions?: number
    min_composite_score?: number
  } = {}): Promise<TraderGroupSuggestion[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/groups/suggestions`, { params })
    return data.suggestions || []
  },

  getClusters: async (minWallets = 2): Promise<WalletCluster[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/clusters`, {
      params: { min_wallets: minWallets },
    })
    return data.clusters || []
  },

  getTags: async (): Promise<TagInfo[]> => {
    const { data } = await discoveryHttp.get(`${API_BASE}/tags`)
    return data.tags || []
  },

  getWalletsByTag: async (tagName: string, limit = 100) => {
    const { data } = await discoveryHttp.get(`${API_BASE}/tags/${tagName}/wallets`, {
      params: { limit },
    })
    return data
  },

  getCrossPlatformEntities: async (limit = 50) => {
    const { data } = await discoveryHttp.get(`${API_BASE}/cross-platform`, { params: { limit } })
    return data
  },
}
