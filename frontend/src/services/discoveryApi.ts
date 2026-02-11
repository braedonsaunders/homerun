import axios from 'axios'

const API_BASE = '/api/discovery'

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
  is_running: boolean
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
}

export const discoveryApi = {
  getLeaderboard: async (params: {
    limit?: number
    offset?: number
    min_trades?: number
    min_pnl?: number
    sort_by?: string
    sort_dir?: string
    tags?: string
    recommendation?: string
    time_period?: string
    active_within_hours?: number
    min_activity_score?: number
    pool_only?: boolean
    tier?: string
  } = {}) => {
    const { data } = await axios.get(`${API_BASE}/leaderboard`, { params })
    return data
  },

  getDiscoveryStats: async (): Promise<DiscoveryStats> => {
    const { data } = await axios.get(`${API_BASE}/leaderboard/stats`)
    return data
  },

  getWalletProfile: async (address: string) => {
    const { data } = await axios.get(`${API_BASE}/wallet/${address}/profile`)
    return data
  },

  triggerDiscovery: async (maxMarkets = 50, maxWalletsPerMarket = 30) => {
    const { data } = await axios.post(`${API_BASE}/run`, null, {
      params: { max_markets: maxMarkets, max_wallets_per_market: maxWalletsPerMarket },
    })
    return data
  },

  refreshLeaderboard: async () => {
    const { data } = await axios.post(`${API_BASE}/refresh-leaderboard`)
    return data
  },

  getConfluenceSignals: async (minStrength = 0, limit = 50): Promise<ConfluenceSignal[]> => {
    const { data } = await axios.get(`${API_BASE}/confluence`, {
      params: { min_strength: minStrength, limit, min_tier: 'WATCH' },
    })
    return data.signals || []
  },

  triggerConfluenceScan: async () => {
    const { data } = await axios.post(`${API_BASE}/confluence/scan`)
    return data
  },

  getPoolStats: async (): Promise<PoolStats> => {
    const { data } = await axios.get(`${API_BASE}/pool/stats`)
    return data
  },

  getTrackedTraderOpportunities: async (
    limit = 50,
    minTier: 'WATCH' | 'HIGH' | 'EXTREME' = 'WATCH'
  ): Promise<TrackedTraderOpportunity[]> => {
    const { data } = await axios.get(`${API_BASE}/opportunities/tracked-traders`, {
      params: { limit, min_tier: minTier },
    })
    return data.opportunities || []
  },

  getClusters: async (minWallets = 2): Promise<WalletCluster[]> => {
    const { data } = await axios.get(`${API_BASE}/clusters`, {
      params: { min_wallets: minWallets },
    })
    return data.clusters || []
  },

  getTags: async (): Promise<TagInfo[]> => {
    const { data } = await axios.get(`${API_BASE}/tags`)
    return data.tags || []
  },

  getWalletsByTag: async (tagName: string, limit = 100) => {
    const { data } = await axios.get(`${API_BASE}/tags/${tagName}/wallets`, {
      params: { limit },
    })
    return data
  },

  getCrossPlatformEntities: async (limit = 50) => {
    const { data } = await axios.get(`${API_BASE}/cross-platform`, { params: { limit } })
    return data
  },
}
