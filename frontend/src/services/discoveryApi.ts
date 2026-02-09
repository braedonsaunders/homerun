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
  anomaly_score: number
  is_bot: boolean
  is_profitable: boolean
  recommendation: string
  rank_score: number
  rank_position: number | null
  tags: string[]
  cluster_id: string | null
  strategies_detected: string[]
  days_active: number
  trades_per_day: number
  unique_markets: number
  last_analyzed_at: string | null
}

export interface ConfluenceSignal {
  id: string
  market_id: string
  market_question: string | null
  signal_type: string
  strength: number
  wallet_count: number
  wallets: string[]
  outcome: string | null
  avg_entry_price: number | null
  total_size: number | null
  is_active: boolean
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
      params: { min_strength: minStrength, limit },
    })
    return data.signals || []
  },

  triggerConfluenceScan: async () => {
    const { data } = await axios.post(`${API_BASE}/confluence/scan`)
    return data
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
