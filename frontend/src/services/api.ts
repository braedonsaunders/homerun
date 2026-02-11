import axios from 'axios'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // 60s so heavy backend work (discovery, scans) doesn't starve requests
})

// Debug interceptor — logs every response so issues are visible in browser console
api.interceptors.response.use(
  (response) => {
    normalizeUtcTimestampsInPlace(response.data)
    const count = Array.isArray(response.data) ? response.data.length : '?'
    console.debug(`[API] ${response.config.method?.toUpperCase()} ${response.config.url} → ${response.status} (${count} items)`)
    return response
  },
  (error) => {
    console.error(`[API] ${error.config?.method?.toUpperCase()} ${error.config?.url} → ${error.response?.status || error.message}`, error.response?.data)
    return Promise.reject(error)
  }
)

// ==================== TYPES ====================

export interface AIAnalysis {
  overall_score: number
  profit_viability: number
  resolution_safety: number
  execution_feasibility: number
  market_efficiency: number
  recommendation: string
  reasoning: string | null
  risk_factors: string[]
  judged_at: string | null
  resolution_analyses: Array<{
    market_id: string
    clarity_score: number
    risk_score: number
    confidence: number
    recommendation: string
    summary: string
    ambiguities: string[]
    edge_cases: string[]
  }>
}

export interface Opportunity {
  id: string
  stable_id: string
  strategy: string
  title: string
  description: string
  total_cost: number
  expected_payout: number
  gross_profit: number
  fee: number
  net_profit: number
  roi_percent: number
  risk_score: number
  risk_factors: string[]
  markets: Market[]
  event_id?: string
  event_slug?: string
  event_title?: string
  category?: string
  min_liquidity: number
  volume?: number
  max_position_size: number
  detected_at: string
  resolution_date?: string
  positions_to_take: Position[]
  ai_analysis: AIAnalysis | null
}

export interface Market {
  id: string
  slug?: string
  event_slug?: string
  question: string
  yes_price: number
  no_price: number
  liquidity: number
  platform?: string  // "polymarket" | "kalshi"
  price_history?: Array<{
    t: number
    yes: number
    no: number
  }>
}

export interface Position {
  action: string
  outcome: string
  market: string
  price: number
  token_id?: string
  platform?: string  // "polymarket" | "kalshi"
  ticker?: string    // Kalshi market ticker
  market_id?: string
}

export interface ScannerStatus {
  running: boolean
  enabled: boolean
  interval_seconds: number
  last_scan: string | null
  opportunities_count: number
  current_activity?: string
  strategies: Strategy[]
}

export interface Strategy {
  type: string
  name: string
  description: string
  is_plugin?: boolean
  plugin_id?: string
  plugin_slug?: string
  status?: string  // For plugins: loaded, error, unloaded
}

export interface Wallet {
  address: string
  label: string
  username?: string
  positions: any[]
  recent_trades: any[]
}

export interface SimulationAccount {
  id: string
  name: string
  initial_capital: number
  current_capital: number
  total_pnl: number
  roi_percent: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  open_positions: number
  unrealized_pnl: number
  book_value: number
  market_value: number
  created_at: string | null
}

export interface TradingPosition {
  token_id: string
  market_id: string
  market_slug?: string
  event_slug?: string
  market_question: string
  outcome: string
  size: number
  average_cost: number
  current_price: number
  unrealized_pnl: number
}

export interface SimulationPosition {
  id: string
  market_id: string
  market_slug?: string
  event_slug?: string
  market_question: string
  token_id: string
  side: string
  quantity: number
  entry_price: number
  entry_cost: number
  current_price: number | null
  unrealized_pnl: number
  opened_at: string
  resolution_date: string | null
  status: string
  take_profit_price: number | null
  stop_loss_price: number | null
}

export interface EquityPoint {
  date: string
  equity: number
  pnl: number
  cumulative_pnl: number
  trade_count: number
  trade_id?: string
  status?: string
}

export interface EquityHistorySummary {
  total_trades: number
  winning_trades: number
  losing_trades: number
  open_trades: number
  total_invested: number
  total_returned: number
  realized_pnl: number
  unrealized_pnl: number
  total_pnl: number
  book_value: number
  market_value: number
  max_drawdown: number
  max_drawdown_pct: number
  profit_factor: number
  best_trade: number
  worst_trade: number
  avg_win: number
  avg_loss: number
  win_rate: number
  roi_percent: number
}

export interface EquityHistoryResponse {
  account_id: string
  initial_capital: number
  current_capital: number
  equity_points: EquityPoint[]
  summary: EquityHistorySummary
}

export interface SimulationTrade {
  id: string
  opportunity_id: string
  strategy_type: string
  total_cost: number
  expected_profit: number
  slippage: number
  status: string
  actual_payout?: number
  actual_pnl?: number
  fees_paid: number
  executed_at: string
  resolved_at?: string
  copied_from?: string
}

export interface CopyConfig {
  id: string
  source_wallet: string
  account_id: string
  enabled: boolean
  copy_mode: string
  settings: {
    min_roi_threshold: number
    max_position_size: number
    copy_delay_seconds: number
    slippage_tolerance: number
    proportional_sizing: boolean
    proportional_multiplier: number
    copy_buys: boolean
    copy_sells: boolean
    market_categories: string[]
  }
  stats: {
    total_copied: number
    successful_copies: number
    failed_copies: number
    total_pnl: number
    total_buys_copied: number
    total_sells_copied: number
  }
}

export interface CopiedTrade {
  id: string
  config_id: string
  source_trade_id: string
  source_wallet: string
  market_id: string
  market_question: string | null
  token_id: string | null
  side: string
  outcome: string | null
  source_price: number
  source_size: number
  executed_price: number | null
  executed_size: number | null
  status: string
  execution_mode: string
  error_message: string | null
  source_timestamp: string | null
  copied_at: string
  executed_at: string | null
  realized_pnl: number | null
}

export interface CopyTradingStatus {
  service_running: boolean
  poll_interval_seconds: number
  total_configs: number
  enabled_configs: number
  tracked_wallets: string[]
  configs_summary: Array<{
    id: string
    source_wallet: string
    copy_mode: string
    enabled: boolean
    total_copied: number
    successful_copies: number
  }>
}

export interface WalletAnalysis {
  wallet: string
  stats: {
    total_trades: number
    win_rate: number
    total_pnl: number
    avg_roi: number
    max_roi: number
    avg_hold_time_hours?: number
    trade_frequency_per_day?: number
    markets_traded?: number
  }
  strategies_detected: string[]
  anomaly_score: number
  anomalies: Anomaly[]
  is_profitable_pattern: boolean
  recommendation: string
}

export interface Anomaly {
  type: string
  severity: string
  score: number
  description: string
  evidence: Record<string, any>
}

export interface WalletTrade {
  id: string
  market: string
  market_slug: string
  market_title: string
  event_slug: string
  outcome: string
  side: string
  size: number
  price: number
  cost: number
  timestamp: string
  transaction_hash: string
}

export interface WalletPosition {
  market: string
  title: string
  market_slug: string
  event_slug?: string
  outcome: string
  size: number
  avg_price: number
  current_price: number
  cost_basis: number
  current_value: number
  unrealized_pnl: number
  roi_percent: number
}

export interface WalletSummary {
  wallet: string
  summary: {
    total_trades: number
    buys: number
    sells: number
    open_positions: number
    total_invested: number
    total_returned: number
    position_value: number
    realized_pnl: number
    unrealized_pnl: number
    total_pnl: number
    roi_percent: number
  }
}

// ==================== OPPORTUNITIES ====================

export interface OpportunitiesResponse {
  opportunities: Opportunity[]
  total: number
}

export const getOpportunities = async (params?: {
  min_profit?: number
  max_risk?: number
  strategy?: string
  min_liquidity?: number
  search?: string
  category?: string
  sort_by?: string
  sort_dir?: string
  exclude_strategy?: string
  limit?: number
  offset?: number
}): Promise<OpportunitiesResponse> => {
  const response = await api.get('/opportunities', { params })
  const total = parseInt(response.headers['x-total-count'] || '0', 10)
  return {
    opportunities: response.data,
    total
  }
}

// ==================== CRYPTO MARKETS (independent infrastructure) ====================

export interface CryptoMarketUpcoming {
  id: string
  slug: string
  event_title: string
  start_time: string | null
  end_time: string | null
  up_price: number | null
  down_price: number | null
  best_bid: number | null
  best_ask: number | null
  liquidity: number
  volume: number
}

export interface CryptoMarket {
  id: string
  condition_id: string
  slug: string
  question: string
  asset: string
  timeframe: string
  start_time: string | null
  end_time: string | null
  seconds_left: number | null
  is_live: boolean
  is_current: boolean
  up_price: number | null
  down_price: number | null
  best_bid: number | null
  best_ask: number | null
  spread: number | null
  combined: number | null
  liquidity: number
  volume: number
  volume_24h: number
  series_volume_24h: number
  series_liquidity: number
  last_trade_price: number | null
  clob_token_ids: string[]
  fees_enabled: boolean
  event_slug: string
  event_title: string
  upcoming_markets: CryptoMarketUpcoming[]
  // Attached by API
  oracle_price: number | null
  oracle_updated_at_ms: number | null
  oracle_age_seconds: number | null
  price_to_beat: number | null
  oracle_history: { t: number; p: number }[]
}

export const getCryptoMarkets = async (): Promise<CryptoMarket[]> => {
  const { data } = await api.get('/crypto/markets')
  return data
}

// ==================== OPPORTUNITY COUNTS ====================

export interface OpportunityCounts {
  strategies: Record<string, number>
  categories: Record<string, number>
}

export const getOpportunityCounts = async (params?: {
  min_profit?: number
  max_risk?: number
  min_liquidity?: number
  search?: string
}): Promise<OpportunityCounts> => {
  const { data } = await api.get('/opportunities/counts', { params })
  return data
}

export const searchPolymarketOpportunities = async (params: {
  q: string
  limit?: number
}): Promise<OpportunitiesResponse> => {
  const response = await api.get('/opportunities/search-polymarket', { params, timeout: 60_000 })
  const total = parseInt(response.headers['x-total-count'] || '0', 10)
  return {
    opportunities: response.data,
    total
  }
}

export const evaluateSearchResults = async (conditionIds: string[]): Promise<{ status: string; count: number; message: string }> => {
  const { data } = await api.post('/opportunities/search-polymarket/evaluate', { condition_ids: conditionIds })
  return data
}

export const triggerScan = async () => {
  const { data } = await api.post('/scan')
  return data
}

// ==================== SCANNER ====================

export const getScannerStatus = async (): Promise<ScannerStatus> => {
  const { data } = await api.get('/scanner/status')
  return data
}

export const startScanner = async (): Promise<ScannerStatus> => {
  const { data } = await api.post('/scanner/start')
  return data
}

export const pauseScanner = async (): Promise<ScannerStatus> => {
  const { data } = await api.post('/scanner/pause')
  return data
}

export const setScannerInterval = async (intervalSeconds: number): Promise<ScannerStatus> => {
  const { data } = await api.post('/scanner/interval', null, {
    params: { interval_seconds: intervalSeconds }
  })
  return data
}

export const getStrategies = async (): Promise<Strategy[]> => {
  const { data } = await api.get('/strategies')
  return data
}

// ==================== PLUGINS ====================

export interface PluginRuntime {
  slug: string
  class_name: string
  name: string
  description: string
  loaded_at: string
  source_hash: string
  run_count: number
  error_count: number
  total_opportunities: number
  last_run: string | null
  last_error: string | null
}

export interface StrategyPlugin {
  id: string
  slug: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  enabled: boolean
  status: 'unloaded' | 'loaded' | 'error'
  error_message: string | null
  config: Record<string, unknown>
  version: number
  sort_order: number
  created_at: string | null
  updated_at: string | null
  runtime: PluginRuntime | null
}

export interface PluginValidation {
  valid: boolean
  class_name: string | null
  strategy_name: string | null
  strategy_description: string | null
  errors: string[]
  warnings: string[]
}

export const getPlugins = async (): Promise<StrategyPlugin[]> => {
  const { data } = await api.get('/plugins')
  return data
}

export const createPlugin = async (plugin: {
  slug: string
  source_code: string
  config?: Record<string, unknown>
  enabled?: boolean
}): Promise<StrategyPlugin> => {
  const { data } = await api.post('/plugins', plugin)
  return data
}

export const updatePlugin = async (
  id: string,
  updates: Partial<{
    source_code: string
    config: Record<string, unknown>
    enabled: boolean
    name: string
    description: string
  }>
): Promise<StrategyPlugin> => {
  const { data } = await api.put(`/plugins/${id}`, updates)
  return data
}

export const deletePlugin = async (id: string): Promise<void> => {
  await api.delete(`/plugins/${id}`)
}

export const validatePlugin = async (source_code: string): Promise<PluginValidation> => {
  const { data } = await api.post('/plugins/validate', { source_code })
  return data
}

export const getPluginTemplate = async (): Promise<{
  template: string
  instructions: string
  available_imports: string[]
}> => {
  const { data } = await api.get('/plugins/template')
  return data
}

export const reloadPlugin = async (id: string): Promise<{
  status: string
  message: string
  runtime: PluginRuntime | null
}> => {
  const { data } = await api.post(`/plugins/${id}/reload`)
  return data
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const getPluginDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/plugins/docs')
  return data
}

// ==================== WALLETS ====================

export const getWallets = async (): Promise<Wallet[]> => {
  const { data } = await api.get('/wallets')
  return data
}

export const addWallet = async (address: string, label?: string): Promise<{ status: string; address: string; label: string | null }> => {
  const { data } = await api.post('/wallets', null, {
    params: { address, label }
  })
  return data
}

export const removeWallet = async (address: string): Promise<{ status: string; message: string }> => {
  const { data } = await api.delete(`/wallets/${address}`)
  return data
}

// Recent trades from tracked wallets
export interface RecentTradeFromWallet {
  id?: string
  market?: string
  market_title?: string
  market_slug?: string
  event_slug?: string
  outcome?: string
  side?: string
  size?: number
  price?: number
  cost?: number
  timestamp?: string
  timestamp_iso?: string
  time?: string
  match_time?: string
  created_at?: string
  transaction_hash?: string
  asset_id?: string
  wallet_address: string
  wallet_label: string
  wallet_username?: string
}

export interface RecentTradesResponse {
  trades: RecentTradeFromWallet[]
  total: number
  tracked_wallets: number
  hours_window: number
}

export interface TrackedTraderOpportunityDTO {
  id: string
  market_id: string
  market_question: string | null
  market_slug: string | null
  signal_type: string
  strength: number
  conviction_score: number
  tier: 'WATCH' | 'HIGH' | 'EXTREME' | string
  window_minutes: number
  wallet_count: number
  cluster_adjusted_wallet_count: number
  unique_core_wallets: number
  weighted_wallet_score: number
  wallets: string[]
  outcome: string | null
  avg_entry_price: number | null
  total_size: number | null
  net_notional: number | null
  conflicting_notional: number | null
  market_liquidity: number | null
  market_volume_24h: number | null
  first_seen_at: string | null
  last_seen_at: string | null
  detected_at: string | null
  is_active: boolean
  top_wallets?: Array<{
    address: string
    username: string | null
    rank_score: number
    composite_score: number
    quality_score: number
    activity_score: number
  }>
}

export const getRecentTradesFromWallets = async (params?: {
  limit?: number
  hours?: number
}): Promise<RecentTradesResponse> => {
  const { data } = await api.get('/wallets/recent-trades/all', { params })
  return data
}

export const getTrackedTraderOpportunities = async (params?: {
  limit?: number
  min_tier?: 'WATCH' | 'HIGH' | 'EXTREME'
}): Promise<{ opportunities: TrackedTraderOpportunityDTO[]; total: number }> => {
  const { data } = await api.get('/discovery/opportunities/tracked-traders', {
    params,
  })
  return data
}

export const getWalletPositions = async (address: string): Promise<{ address: string; positions: WalletPosition[] }> => {
  const { data } = await api.get(`/wallets/${address}/positions`)
  return data
}

export const getWalletTrades = async (address: string, limit = 100): Promise<{ address: string; trades: WalletTrade[] }> => {
  const { data } = await api.get(`/wallets/${address}/trades`, {
    params: { limit }
  })
  return data
}

// ==================== MARKETS ====================

export const getMarkets = async (params?: {
  active?: boolean
  limit?: number
  offset?: number
}): Promise<Market[]> => {
  const { data } = await api.get('/markets', { params })
  return data
}

export const getEvents = async (params?: {
  closed?: boolean
  limit?: number
  offset?: number
}): Promise<Record<string, any>[]> => {
  const { data } = await api.get('/events', { params })
  return data
}

// ==================== SIMULATION ====================

export const createSimulationAccount = async (params: {
  name: string
  initial_capital?: number
  max_position_pct?: number
  max_positions?: number
}): Promise<{ account_id: string; name: string; initial_capital: number; message: string }> => {
  const { data } = await api.post('/simulation/accounts', params)
  return data
}

export const getSimulationAccounts = async (): Promise<SimulationAccount[]> => {
  const { data } = await api.get('/simulation/accounts')
  return data
}

export const getSimulationAccount = async (accountId: string): Promise<SimulationAccount> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}`)
  return data
}

export const deleteSimulationAccount = async (accountId: string): Promise<{ message: string; account_id: string }> => {
  const { data } = await api.delete(`/simulation/accounts/${accountId}`)
  return data
}

export const getAccountPositions = async (accountId: string): Promise<SimulationPosition[]> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/positions`)
  return data
}

export const getAccountTrades = async (accountId: string, limit = 50): Promise<SimulationTrade[]> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/trades`, { params: { limit } })
  return data
}

export const executeOpportunity = async (
  accountId: string,
  opportunityId: string,
  positionSize?: number,
  takeProfitPrice?: number,
  stopLossPrice?: number
): Promise<{ trade_id: string; status: string; total_cost: number; expected_profit: number; slippage: number; message: string }> => {
  const { data } = await api.post(`/simulation/accounts/${accountId}/execute`, {
    opportunity_id: opportunityId,
    position_size: positionSize,
    take_profit_price: takeProfitPrice,
    stop_loss_price: stopLossPrice
  })
  return data
}

export const getAccountPerformance = async (accountId: string): Promise<Record<string, any>> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/performance`)
  return data
}

export const getAccountEquityHistory = async (accountId: string): Promise<EquityHistoryResponse> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/equity-history`)
  return data
}

// ==================== COPY TRADING ====================

export const getCopyConfigs = async (accountId?: string): Promise<CopyConfig[]> => {
  const { data } = await api.get('/copy-trading/configs', { params: { account_id: accountId } })
  return data
}

export const createCopyConfig = async (params: {
  source_wallet: string
  account_id: string
  copy_mode?: string
  min_roi_threshold?: number
  max_position_size?: number
  copy_delay_seconds?: number
  slippage_tolerance?: number
  proportional_sizing?: boolean
  proportional_multiplier?: number
  copy_buys?: boolean
  copy_sells?: boolean
}): Promise<{ config_id: string; source_wallet: string; account_id: string; enabled: boolean; copy_mode: string; message: string }> => {
  const { data } = await api.post('/copy-trading/configs', params)
  return data
}

export const updateCopyConfig = async (configId: string, params: {
  enabled?: boolean
  copy_mode?: string
  min_roi_threshold?: number
  max_position_size?: number
  copy_delay_seconds?: number
  slippage_tolerance?: number
  proportional_sizing?: boolean
  proportional_multiplier?: number
  copy_buys?: boolean
  copy_sells?: boolean
}): Promise<{ message: string; config_id: string }> => {
  const { data } = await api.patch(`/copy-trading/configs/${configId}`, params)
  return data
}

export const deleteCopyConfig = async (configId: string): Promise<{ message: string; config_id: string }> => {
  const { data } = await api.delete(`/copy-trading/configs/${configId}`)
  return data
}

export const enableCopyConfig = async (configId: string): Promise<{ message: string; config_id: string }> => {
  const { data } = await api.post(`/copy-trading/configs/${configId}/enable`)
  return data
}

export const disableCopyConfig = async (configId: string): Promise<{ message: string; config_id: string }> => {
  const { data } = await api.post(`/copy-trading/configs/${configId}/disable`)
  return data
}

export const forceSyncCopyConfig = async (configId: string): Promise<Record<string, any>> => {
  const { data } = await api.post(`/copy-trading/configs/${configId}/sync`)
  return data
}

export const getCopyTrades = async (params?: {
  config_id?: string
  status?: string
  limit?: number
  offset?: number
}): Promise<CopiedTrade[]> => {
  const { data } = await api.get('/copy-trading/trades', { params })
  return data
}

export const getCopyTradingStatus = async (): Promise<CopyTradingStatus> => {
  const { data } = await api.get('/copy-trading/status')
  return data
}

// ==================== ANOMALY DETECTION ====================

export const analyzeWallet = async (address: string): Promise<WalletAnalysis> => {
  const { data } = await api.get(`/anomaly/analyze/${address}`)
  return data
}

export const findProfitableWallets = async (params?: {
  min_trades?: number
  min_win_rate?: number
  min_pnl?: number
  max_anomaly_score?: number
}): Promise<{ count: number; wallets: WalletAnalysis[] }> => {
  const { data } = await api.post('/anomaly/find-profitable', params || {})
  return data
}

export const getAnomalies = async (params?: {
  severity?: string
  anomaly_type?: string
  limit?: number
}): Promise<{ count: number; anomalies: Anomaly[] }> => {
  const { data } = await api.get('/anomaly/anomalies', { params })
  return data
}

export const quickCheckWallet = async (address: string): Promise<{ wallet: string; is_suspicious: boolean; anomaly_score: number; critical_anomalies: number; win_rate: number; total_pnl: number; verdict: string; summary: string }> => {
  const { data } = await api.get(`/anomaly/check/${address}`)
  return data
}

export const getWalletTradesAnalysis = async (address: string, limit = 100): Promise<{ wallet: string; total: number; trades: WalletTrade[] }> => {
  const { data } = await api.get(`/anomaly/wallet/${address}/trades`, { params: { limit } })
  return data
}

export const getWalletPositionsAnalysis = async (address: string): Promise<{ wallet: string; total_positions: number; total_value: number; total_unrealized_pnl: number; positions: WalletPosition[] }> => {
  const { data } = await api.get(`/anomaly/wallet/${address}/positions`)
  return data
}

export const getWalletSummary = async (address: string): Promise<WalletSummary> => {
  const { data } = await api.get(`/anomaly/wallet/${address}/summary`)
  return data
}

// ==================== HEALTH ====================

export const getHealthStatus = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/health/detailed')
  return data
}

// ==================== TRADER DISCOVERY ====================

export interface DiscoveredTrader {
  address: string
  username?: string
  trades: number
  volume: number
  pnl?: number
  rank?: number
  buys: number
  sells: number
  win_rate?: number
  wins?: number
  losses?: number
  total_markets?: number
  trade_count?: number
}

export type TimePeriod = 'DAY' | 'WEEK' | 'MONTH' | 'ALL'
export type OrderBy = 'PNL' | 'VOL'
export type Category = 'OVERALL' | 'POLITICS' | 'SPORTS' | 'CRYPTO' | 'CULTURE' | 'WEATHER' | 'ECONOMICS' | 'TECH' | 'FINANCE'

export interface LeaderboardFilters {
  limit?: number
  time_period?: TimePeriod
  order_by?: OrderBy
  category?: Category
}

export interface WinRateFilters {
  min_win_rate?: number
  min_trades?: number
  limit?: number
  time_period?: TimePeriod
  category?: Category
  min_volume?: number
  max_volume?: number
  scan_count?: number
}

export interface WalletWinRate {
  address: string
  win_rate: number
  wins: number
  losses: number
  total_markets: number
  trade_count: number
  error?: string
}

export interface WalletPnL {
  address: string
  total_trades: number
  open_positions: number
  total_invested: number
  total_returned: number
  position_value: number
  realized_pnl: number
  unrealized_pnl: number
  total_pnl: number
  roi_percent: number
  error?: string
}

export const getLeaderboard = async (filters?: LeaderboardFilters) => {
  const { data } = await api.get('/discover/leaderboard', { params: filters })
  return data
}

export const discoverTopTraders = async (
  limit = 50,
  minTrades = 10,
  filters?: Omit<LeaderboardFilters, 'limit'>
): Promise<DiscoveredTrader[]> => {
  const { data } = await api.get('/discover/top-traders', {
    params: {
      limit,
      min_trades: minTrades,
      ...filters
    }
  })
  return data
}

export const discoverByWinRate = async (filters?: WinRateFilters): Promise<DiscoveredTrader[]> => {
  const { data } = await api.get('/discover/by-win-rate', { params: filters })
  return data
}

export const getWalletWinRate = async (address: string, timePeriod?: TimePeriod): Promise<WalletWinRate> => {
  const { data } = await api.get(`/discover/wallet/${address}/win-rate`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return data
}

export const analyzeWalletPnL = async (address: string, timePeriod?: TimePeriod): Promise<WalletPnL> => {
  const { data } = await api.get(`/discover/wallet/${address}`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return data
}

export interface WalletProfile {
  username: string | null
  address: string
  pnl?: number
  volume?: number
  rank?: number
}

export const getWalletProfile = async (address: string): Promise<WalletProfile> => {
  const { data } = await api.get(`/wallets/${address}/profile`)
  return data
}

// Add time-filtered wallet PnL (for future time filter support)
export const analyzeWalletPnLWithFilter = async (address: string, timePeriod?: TimePeriod): Promise<WalletPnL> => {
  const { data } = await api.get(`/discover/wallet/${address}`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return data
}

export const analyzeAndTrackWallet = async (params: {
  address: string
  label?: string
  auto_copy?: boolean
  simulation_account_id?: string
}) => {
  const { data } = await api.post('/discover/analyze-and-track', null, { params })
  return data
}

// ==================== TRADING ====================

export interface TradingStatus {
  enabled: boolean
  initialized: boolean
  wallet_address: string | null
  stats: {
    total_trades: number
    winning_trades: number
    losing_trades: number
    total_volume: number
    total_pnl: number
    daily_volume: number
    daily_pnl: number
    open_positions: number
    last_trade_at: string | null
  }
  limits: {
    max_trade_size_usd: number
    max_daily_volume: number
    max_open_positions: number
    min_order_size_usd: number
    max_slippage_percent: number
  }
}

export interface Order {
  id: string
  token_id: string
  side: string
  price: number
  size: number
  order_type: string
  status: string
  filled_size: number
  clob_order_id: string | null
  error_message: string | null
  market_question: string | null
  created_at: string
}

export const getTradingStatus = async (): Promise<TradingStatus> => {
  const { data } = await api.get('/trading/status')
  return data
}

export const initializeTrading = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/trading/initialize')
  return data
}

export const placeOrder = async (params: {
  token_id: string
  side: string
  price: number
  size: number
  order_type?: string
  market_question?: string
}): Promise<Order> => {
  const { data } = await api.post('/trading/orders', params)
  return data
}

export const getOrders = async (limit = 100, status?: string): Promise<Order[]> => {
  const { data } = await api.get('/trading/orders', { params: { limit, status } })
  return data
}

export const getOpenOrders = async (): Promise<Order[]> => {
  const { data } = await api.get('/trading/orders/open')
  return data
}

export const cancelOrder = async (orderId: string): Promise<{ status: string; order_id: string }> => {
  const { data } = await api.delete(`/trading/orders/${orderId}`)
  return data
}

export const cancelAllOrders = async (): Promise<{ status: string; cancelled_count: number }> => {
  const { data } = await api.delete('/trading/orders')
  return data
}

export const getTradingPositions = async (): Promise<TradingPosition[]> => {
  const { data } = await api.get('/trading/positions')
  return data
}

export const getTradingBalance = async (): Promise<{ balance: number; available: number; reserved: number; currency: string; timestamp: string }> => {
  const { data } = await api.get('/trading/balance')
  return data
}

export const executeOpportunityLive = async (params: {
  opportunity_id: string
  positions: any[]
  size_usd: number
}): Promise<{ status: string; orders: Order[]; message?: string }> => {
  const { data } = await api.post('/trading/execute-opportunity', params)
  return data
}

export const emergencyStopTrading = async (): Promise<{ status: string; cancelled_orders: number; message: string }> => {
  const { data } = await api.post('/trading/emergency-stop')
  return data
}

// ==================== AUTO TRADER ====================

export interface AutoTraderConfig {
  mode: string
  enabled_strategies: string[]
  min_roi_percent: number
  max_risk_score: number
  min_liquidity_usd: number
  base_position_size_usd: number
  max_position_size_usd: number
  max_daily_trades: number
  max_daily_loss_usd: number
  max_concurrent_positions: number
  execution_delay_seconds: number
  circuit_breaker_losses: number
  circuit_breaker_duration_minutes: number
  auto_retry_failed: boolean
  require_confirmation: boolean
  paper_account_capital?: number
  min_guaranteed_profit: number
  use_profit_guarantee: boolean
  max_end_date_days: number | null
  min_end_date_days: number | null
  prefer_near_settlement: boolean
  priority_method: string
  settlement_weight: number
  roi_weight: number
  liquidity_weight: number
  risk_weight: number
  max_trades_per_event: number
  max_exposure_per_event_usd: number
  excluded_categories: string[]
  excluded_keywords: string[]
  excluded_event_slugs: string[]
  min_volume_usd: number
  take_profit_pct: number
  stop_loss_pct: number
  enable_spread_exits: boolean
  ai_resolution_gate: boolean
  ai_max_resolution_risk: number
  ai_min_resolution_clarity: number
  ai_resolution_block_avoid: boolean
  ai_resolution_model: string | null
  ai_skip_on_analysis_failure: boolean
  ai_position_sizing: boolean
  ai_min_score_to_trade: number
  ai_score_size_multiplier: boolean
  ai_score_boost_threshold: number
  ai_score_boost_multiplier: number
  ai_judge_model: string | null
  news_workflow_enabled: boolean
  news_workflow_min_edge: number
  news_workflow_max_age_minutes: number
  weather_workflow_enabled: boolean
  weather_workflow_min_edge: number
  weather_workflow_max_age_minutes: number
  llm_verify_trades: boolean
  llm_verify_strategies: string[]
  auto_ai_scoring: boolean
}

export interface AutoTraderStatus {
  mode: string
  running: boolean
  trading_active: boolean
  worker_running: boolean
  control: {
    is_enabled: boolean
    is_paused: boolean
    kill_switch: boolean
    requested_run_at: string | null
    run_interval_seconds: number
    updated_at: string | null
  }
  snapshot: {
    running: boolean
    enabled: boolean
    current_activity: string | null
    interval_seconds: number
    last_run_at: string | null
    last_error: string | null
    updated_at: string | null
  }
  config: AutoTraderConfig
  stats: {
    total_trades: number
    winning_trades: number
    losing_trades: number
    win_rate: number
    total_profit: number
    total_invested: number
    roi_percent: number
    daily_trades: number
    daily_profit: number
    consecutive_losses: number
    circuit_breaker_active: boolean
    last_trade_at: string | null
    opportunities_seen: number
    opportunities_executed: number
    opportunities_skipped: number
  }
}

export interface AutoTraderSourcePolicy {
  enabled: boolean
  weight: number
  daily_budget_usd: number
  max_open_positions: number
  min_signal_score: number
  size_multiplier: number
  cooldown_seconds: number
  max_daily_loss?: number | null
  max_total_open_positions?: number | null
  max_per_market_exposure?: number | null
  max_per_event_exposure?: number | null
  kill_switch?: boolean | null
  metadata?: Record<string, any>
}

export interface WorkerStatus {
  worker_name: string
  running: boolean
  enabled: boolean
  current_activity: string | null
  interval_seconds: number
  last_run_at: string | null
  lag_seconds: number | null
  last_error: string | null
  stats: Record<string, any>
  updated_at: string | null
  control?: Record<string, any>
}

export interface TradeSignal {
  id: string
  source: string
  source_item_id: string | null
  signal_type: string
  strategy_type: string | null
  market_id: string
  market_question: string | null
  direction: string | null
  entry_price: number | null
  effective_price: number | null
  edge_percent: number | null
  confidence: number | null
  liquidity: number | null
  expires_at: string | null
  status: string
  payload: Record<string, any> | null
  dedupe_key: string
  created_at: string | null
  updated_at: string | null
}

export interface AutoTraderTrade {
  id: string
  opportunity_id: string
  strategy: string
  executed_at: string
  total_cost: number
  expected_profit: number
  actual_profit: number | null
  status: string
  mode: string
  source?: string
  market_id?: string
  market_question?: string | null
  direction?: string | null
  reason?: string | null
  created_at?: string | null
}

export interface AutoTraderExposure {
  as_of: string | null
  global: {
    daily_budget_usd: number
    budget_used_usd: number
    budget_remaining_usd: number
    budget_utilization_pct: number
    max_total_open_positions: number
    open_positions: number
    max_per_market_exposure: number
    max_per_event_exposure: number
    kill_switch: boolean
  }
  sources: Array<{
    source: string
    enabled: boolean
    daily_budget_usd: number
    budget_used_usd: number
    budget_remaining_usd: number
    budget_utilization_pct: number
    max_open_positions: number
    open_positions: number
    open_position_utilization_pct: number
    weight: number
    min_signal_score: number
    size_multiplier: number
    cooldown_seconds: number
    metadata: Record<string, any>
  }>
  markets: Array<{
    market_id: string
    notional_usd: number
    open_positions: number
    directions: string[]
  }>
  events: Array<{
    event_key: string
    notional_usd: number
    open_positions: number
  }>
}

export interface AutoTraderSourceMetrics {
  source: string
  policy_enabled: boolean
  pending_signals: number
  decisions_total: number
  selected: number
  skipped: number
  executed: number
  submitted: number
  failed: number
  skip_rate: number
  success_rate: number
  trades: Record<string, number>
  avg_decision_score: number
  decisions_last_hour: number
  throughput_per_minute: number
  avg_decision_to_trade_latency_seconds: number
  top_skip_reasons: Array<{ reason: string; count: number }>
  last_decision_at: string | null
  last_trade_at: string | null
  last_pending_signal_at: string | null
}

export interface AutoTraderMetrics {
  as_of: string | null
  summary: {
    sources_tracked: number
    active_sources: number
    decisions_last_hour: number
  }
  decision_funnel: {
    seen: number
    pending: number
    selected: number
    skipped: number
    submitted: number
    executed: number
    failed: number
  }
  skip_reasons: Array<{ reason: string; count: number }>
  sources: AutoTraderSourceMetrics[]
}

export interface AutoTraderDecision {
  id: string
  signal_id: string | null
  source: string
  decision: string
  reason: string | null
  score: number | null
  policy_snapshot: Record<string, any>
  risk_snapshot: Record<string, any>
  payload: Record<string, any>
  created_at: string | null
}

export const getAutoTraderStatus = async (): Promise<AutoTraderStatus> => {
  const { data } = await api.get('/auto-trader/status')
  const control = data.control || {}
  const snapshot = data.snapshot || {}
  const policies = data.policies || {}
  const sources = policies.sources || {}
  const tradingActive = Boolean(control.is_enabled) && !Boolean(control.is_paused) && !Boolean(control.kill_switch)
  const workerRunning = Boolean(snapshot.running)

  // Backward-compatible shape for existing UI components.
  return {
    mode: control.mode || 'paper',
    running: tradingActive,
    trading_active: tradingActive,
    worker_running: workerRunning,
    control: {
      is_enabled: Boolean(control.is_enabled),
      is_paused: Boolean(control.is_paused),
      kill_switch: Boolean(control.kill_switch),
      requested_run_at: control.requested_run_at || null,
      run_interval_seconds: control.run_interval_seconds || 2,
      updated_at: control.updated_at || null,
    },
    snapshot: {
      running: workerRunning,
      enabled: Boolean(snapshot.enabled),
      current_activity: snapshot.current_activity || null,
      interval_seconds: snapshot.interval_seconds || 2,
      last_run_at: snapshot.last_run_at || null,
      last_error: snapshot.last_error || null,
      updated_at: snapshot.updated_at || null,
    },
    config: {
      mode: control.mode || 'paper',
      enabled_strategies: [],
      min_roi_percent: 0,
      max_risk_score: 1,
      min_liquidity_usd: 0,
      base_position_size_usd: 10,
      max_position_size_usd: 100,
      max_daily_trades: 0,
      max_daily_loss_usd: policies.global?.max_daily_loss ?? 0,
      max_concurrent_positions: policies.global?.max_total_open_positions ?? 0,
      execution_delay_seconds: 0,
      require_confirmation: false,
      auto_retry_failed: false,
      circuit_breaker_losses: 0,
      circuit_breaker_duration_minutes: 0,
      min_guaranteed_profit: 0,
      use_profit_guarantee: false,
      max_end_date_days: null,
      min_end_date_days: null,
      prefer_near_settlement: true,
      priority_method: 'composite',
      settlement_weight: 0,
      roi_weight: 0,
      liquidity_weight: 0,
      risk_weight: 0,
      max_trades_per_event: 0,
      max_exposure_per_event_usd: policies.global?.max_per_event_exposure ?? 0,
      excluded_categories: [],
      excluded_keywords: [],
      excluded_event_slugs: [],
      min_volume_usd: 0,
      take_profit_pct: 0,
      stop_loss_pct: 0,
      enable_spread_exits: false,
      ai_resolution_gate: false,
      ai_max_resolution_risk: 0,
      ai_min_resolution_clarity: 0,
      ai_resolution_block_avoid: false,
      ai_resolution_model: null,
      ai_skip_on_analysis_failure: false,
      ai_position_sizing: false,
      ai_min_score_to_trade: 0,
      ai_score_size_multiplier: false,
      ai_score_boost_threshold: 0,
      ai_score_boost_multiplier: 1,
      ai_judge_model: null,
      news_workflow_enabled: Boolean(sources.news?.enabled ?? true),
      news_workflow_min_edge: 0,
      news_workflow_max_age_minutes: 0,
      weather_workflow_enabled: Boolean(sources.weather?.enabled ?? true),
      weather_workflow_min_edge: 0,
      weather_workflow_max_age_minutes: 0,
      llm_verify_trades: false,
      llm_verify_strategies: [],
      auto_ai_scoring: false,
    },
    stats: {
      total_trades: snapshot.trades_count || 0,
      winning_trades: 0,
      losing_trades: 0,
      win_rate: 0,
      total_profit: snapshot.daily_pnl || 0,
      total_invested: 0,
      roi_percent: 0,
      daily_trades: snapshot.trades_count || 0,
      daily_profit: snapshot.daily_pnl || 0,
      consecutive_losses: 0,
      circuit_breaker_active: Boolean(control.kill_switch),
      last_trade_at: snapshot.last_run_at || null,
      opportunities_seen: snapshot.signals_seen || 0,
      opportunities_executed: snapshot.signals_selected || 0,
      opportunities_skipped: Math.max(
        0,
        (snapshot.signals_seen || 0) - (snapshot.signals_selected || 0)
      ),
    },
  }
}

export const startAutoTrader = async (mode?: string, accountId?: string): Promise<{ status: string; mode: string; message: string }> => {
  const { data } = await api.post('/auto-trader/start', null, { params: { mode, account_id: accountId } })
  return {
    status: data.status || 'started',
    mode: data.control?.mode || mode || 'paper',
    message: 'Auto-trader started from manual launch state.',
  }
}

export const stopAutoTrader = async (): Promise<{ status: string }> => {
  const { data } = await api.post('/auto-trader/stop')
  return { status: data.status || 'stopped' }
}

export const updateAutoTraderConfig = async (config: Partial<AutoTraderConfig>): Promise<{ status: string; config: Record<string, any> }> => {
  const controlUpdates: Record<string, any> = {}
  const policyUpdates: Record<string, any> = { sources: {} }

  if (config.mode) controlUpdates.mode = config.mode
  if (typeof config.max_daily_loss_usd === 'number') {
    policyUpdates.global = { max_daily_loss: config.max_daily_loss_usd }
  }
  if (typeof config.max_concurrent_positions === 'number') {
    policyUpdates.global = {
      ...(policyUpdates.global || {}),
      max_total_open_positions: config.max_concurrent_positions,
    }
  }
  if (typeof config.news_workflow_enabled === 'boolean') {
    policyUpdates.sources.news = {
      ...(policyUpdates.sources.news || {}),
      enabled: config.news_workflow_enabled,
    }
  }
  if (typeof config.weather_workflow_enabled === 'boolean') {
    policyUpdates.sources.weather = {
      ...(policyUpdates.sources.weather || {}),
      enabled: config.weather_workflow_enabled,
    }
  }

  if (Object.keys(controlUpdates).length > 0) {
    await api.put('/auto-trader/control', controlUpdates)
  }
  if (
    policyUpdates.global ||
    Object.keys(policyUpdates.sources as Record<string, any>).length > 0
  ) {
    await api.put('/auto-trader/policies', policyUpdates)
  }

  return { status: 'updated', config: config as Record<string, any> }
}

export const getAutoTraderTrades = async (limit = 100, status?: string): Promise<AutoTraderTrade[]> => {
  const { data } = await api.get('/auto-trader/trades', { params: { limit, status } })
  const trades = data.trades || []
  return trades.map((t: any) => ({
    id: t.id,
    opportunity_id: t.signal_id,
    strategy: t.source,
    executed_at: t.executed_at || t.created_at,
    total_cost: t.notional_usd || 0,
    expected_profit: 0,
    actual_profit: null,
    status: t.status,
    mode: t.mode,
    source: t.source,
    market_id: t.market_id,
    market_question: t.market_question,
    direction: t.direction,
    reason: t.reason,
    created_at: t.created_at,
  }))
}

export const getAutoTraderStats = async (): Promise<AutoTraderStatus['stats']> => {
  const status = await getAutoTraderStatus()
  return status.stats
}

export const resetAutoTraderStats = async (): Promise<{ status: string; message: string }> => {
  return { status: 'noop', message: 'Auto-trader stats are persisted and audit-backed.' }
}

export const resetCircuitBreaker = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/auto-trader/kill-switch', null, {
    params: { enabled: false },
  })
  return { status: data.status || 'updated', message: 'Kill switch disabled' }
}

export const enableLiveTrading = async (maxDailyLoss = 100): Promise<{ status: string; warning: string; max_daily_loss: number; config: Record<string, any> }> => {
  await api.put('/auto-trader/policies', {
    global: { max_daily_loss: maxDailyLoss },
  })
  const { data } = await api.post('/auto-trader/start', null, { params: { mode: 'live' } })
  return {
    status: data.status || 'started',
    warning: 'Auto trader is now configured for live mode.',
    max_daily_loss: maxDailyLoss,
    config: data.control || {},
  }
}

export const emergencyStopAutoTrader = async (): Promise<{ status: string; auto_trader: string; mode: string; cancelled_orders: number; message: string }> => {
  const { data } = await api.post('/auto-trader/kill-switch', null, {
    params: { enabled: true },
  })
  return {
    status: data.status || 'updated',
    auto_trader: 'paused',
    mode: 'kill_switch',
    cancelled_orders: 0,
    message: 'Kill switch enabled; autotrader worker will stop selecting trades.',
  }
}

export const getAutoTraderDecisions = async (params?: {
  source?: string
  decision?: string
  limit?: number
}): Promise<{ total: number; decisions: AutoTraderDecision[] }> => {
  const { data } = await api.get('/auto-trader/decisions', { params })
  return data
}

export const getAutoTraderPolicies = async (): Promise<{
  global: AutoTraderSourcePolicy
  sources: Record<string, AutoTraderSourcePolicy>
}> => {
  const { data } = await api.get('/auto-trader/policies')
  return data
}

export const getAutoTraderExposure = async (): Promise<AutoTraderExposure> => {
  const { data } = await api.get('/auto-trader/exposure')
  return data
}

export const getAutoTraderMetrics = async (): Promise<AutoTraderMetrics> => {
  const { data } = await api.get('/auto-trader/metrics')
  return data
}

export const updateAutoTraderPolicies = async (payload: {
  global?: Partial<AutoTraderSourcePolicy>
  sources?: Record<string, Partial<AutoTraderSourcePolicy>>
}) => {
  const { data } = await api.put('/auto-trader/policies', payload)
  return data
}

export const getSignals = async (params?: {
  source?: string
  status?: string
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; signals: TradeSignal[] }> => {
  const { data } = await api.get('/signals', { params })
  return data
}

export const getSignalStats = async (): Promise<{
  totals: Record<string, number>
  sources: Array<Record<string, any>>
}> => {
  const { data } = await api.get('/signals/stats')
  return data
}

export const getWorkersStatus = async (): Promise<{ workers: WorkerStatus[] }> => {
  const { data } = await api.get('/workers/status')
  return data
}

export const startWorker = async (worker: string) => {
  const { data } = await api.post(`/workers/${worker}/start`)
  return data
}

export const pauseWorker = async (worker: string) => {
  const { data } = await api.post(`/workers/${worker}/pause`)
  return data
}

export const runWorkerOnce = async (worker: string) => {
  const { data } = await api.post(`/workers/${worker}/run-once`)
  return data
}

export const setWorkerInterval = async (worker: string, intervalSeconds: number) => {
  const { data } = await api.post(`/workers/${worker}/interval`, null, {
    params: { interval_seconds: intervalSeconds },
  })
  return data
}

// ==================== SETTINGS ====================

export interface PolymarketSettings {
  api_key: string | null
  api_secret: string | null
  api_passphrase: string | null
  private_key: string | null
}

export interface KalshiSettings {
  email: string | null
  password: string | null
  api_key: string | null
}

export interface LLMSettings {
  provider: string
  openai_api_key: string | null
  anthropic_api_key: string | null
  google_api_key: string | null
  xai_api_key: string | null
  deepseek_api_key: string | null
  ollama_api_key: string | null
  ollama_base_url: string | null
  lmstudio_api_key: string | null
  lmstudio_base_url: string | null
  model: string | null
  max_monthly_spend: number | null
}

export interface NotificationSettings {
  enabled: boolean
  telegram_bot_token: string | null
  telegram_chat_id: string | null
  notify_on_opportunity: boolean
  notify_on_trade: boolean
  notify_min_roi: number
}

export interface ScannerSettings {
  scan_interval_seconds: number
  min_profit_threshold: number
  max_markets_to_scan: number
  min_liquidity: number
}

export interface TradingSettingsConfig {
  trading_enabled: boolean
  max_trade_size_usd: number
  max_daily_trade_volume: number
  max_open_positions: number
  max_slippage_percent: number
}

export interface MaintenanceSettings {
  auto_cleanup_enabled: boolean
  cleanup_interval_hours: number
  cleanup_resolved_trade_days: number
}

export interface TradingProxySettings {
  enabled: boolean
  proxy_url: string | null
  verify_ssl: boolean
  timeout: number
  require_vpn: boolean
}

export interface SearchFilterSettings {
  // Hard rejection filters
  min_liquidity_hard: number
  min_position_size: number
  min_absolute_profit: number
  min_annualized_roi: number
  max_resolution_months: number
  max_plausible_roi: number
  max_trade_legs: number
  // NegRisk
  negrisk_min_total_yes: number
  negrisk_warn_total_yes: number
  negrisk_election_min_total_yes: number
  negrisk_max_resolution_spread_days: number
  // Settlement lag
  settlement_lag_max_days_to_resolution: number
  settlement_lag_near_zero: number
  settlement_lag_near_one: number
  settlement_lag_min_sum_deviation: number
  // Risk scoring
  risk_very_short_days: number
  risk_short_days: number
  risk_long_lockup_days: number
  risk_extended_lockup_days: number
  risk_low_liquidity: number
  risk_moderate_liquidity: number
  risk_complex_legs: number
  risk_multiple_legs: number
  // BTC/ETH high-frequency
  btc_eth_hf_enabled: boolean
  btc_eth_hf_series_btc_15m: string
  btc_eth_hf_series_eth_15m: string
  btc_eth_hf_series_sol_15m: string
  btc_eth_hf_series_xrp_15m: string
  btc_eth_pure_arb_max_combined: number
  btc_eth_dump_hedge_drop_pct: number
  btc_eth_thin_liquidity_usd: number
  // Miracle strategy
  miracle_min_no_price: number
  miracle_max_no_price: number
  miracle_min_impossibility_score: number
  // Cross-platform
  cross_platform_enabled: boolean
  // Combinatorial
  combinatorial_min_confidence: number
  combinatorial_high_confidence: number
  // Bayesian cascade
  bayesian_cascade_enabled: boolean
  bayesian_min_edge_percent: number
  bayesian_propagation_depth: number
  // Liquidity vacuum
  liquidity_vacuum_enabled: boolean
  liquidity_vacuum_min_imbalance_ratio: number
  liquidity_vacuum_min_depth_usd: number
  // Entropy arb
  entropy_arb_enabled: boolean
  entropy_arb_min_deviation: number
  // Event-driven
  event_driven_enabled: boolean
  // Temporal decay
  temporal_decay_enabled: boolean
  // Correlation arb
  correlation_arb_enabled: boolean
  correlation_arb_min_correlation: number
  correlation_arb_min_divergence: number
  // Market making
  market_making_enabled: boolean
  market_making_spread_bps: number
  market_making_max_inventory_usd: number
  // Statistical arb
  stat_arb_enabled: boolean
  stat_arb_min_edge: number
}

export interface AllSettings {
  polymarket: PolymarketSettings
  kalshi: KalshiSettings
  llm: LLMSettings
  notifications: NotificationSettings
  scanner: ScannerSettings
  trading: TradingSettingsConfig
  maintenance: MaintenanceSettings
  trading_proxy: TradingProxySettings
  search_filters: SearchFilterSettings
  updated_at: string | null
}

export interface UpdateSettingsRequest {
  polymarket?: Partial<PolymarketSettings>
  kalshi?: Partial<KalshiSettings>
  llm?: Partial<LLMSettings>
  notifications?: Partial<NotificationSettings>
  scanner?: Partial<ScannerSettings>
  trading?: Partial<TradingSettingsConfig>
  maintenance?: Partial<MaintenanceSettings>
  trading_proxy?: Partial<TradingProxySettings>
  search_filters?: Partial<SearchFilterSettings>
}

export const getSettings = async (): Promise<AllSettings> => {
  const { data } = await api.get('/settings')
  return data
}

export const updateSettings = async (settings: UpdateSettingsRequest): Promise<{ status: string; message: string; updated_at: string }> => {
  const { data } = await api.put('/settings', settings)
  return data
}

export const updatePolymarketSettings = async (settings: Partial<PolymarketSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/polymarket', settings)
  return data
}

export const updateLLMSettings = async (settings: Partial<LLMSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/llm', settings)
  return data
}

export const updateNotificationSettings = async (settings: Partial<NotificationSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/notifications', settings)
  return data
}

export const updateScannerSettings = async (settings: Partial<ScannerSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/scanner', settings)
  return data
}

export const updateTradingSettings = async (settings: Partial<TradingSettingsConfig>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/trading', settings)
  return data
}

export const updateMaintenanceSettings = async (settings: Partial<MaintenanceSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/maintenance', settings)
  return data
}

export interface LLMModelOption {
  id: string
  name: string
}

export interface LLMModelsResponse {
  models: Record<string, LLMModelOption[]>
}

export interface RefreshModelsResponse {
  status: string
  message: string
  models: Record<string, LLMModelOption[]>
}

export const getLLMModels = async (provider?: string): Promise<LLMModelsResponse> => {
  const { data } = await api.get('/settings/llm/models', { params: provider ? { provider } : undefined })
  return data
}

export const refreshLLMModels = async (provider?: string): Promise<RefreshModelsResponse> => {
  const { data } = await api.post('/settings/llm/models/refresh', null, { params: provider ? { provider } : undefined })
  return data
}

export const testPolymarketConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/polymarket')
  return data
}

export const testTelegramConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/telegram')
  return data
}

export const testTradingProxy = async (): Promise<{ status: string; message: string; proxy_enabled?: boolean; proxy_ip?: string; direct_ip?: string; vpn_active?: boolean }> => {
  const { data } = await api.post('/settings/test/trading-proxy')
  return data
}

// ==================== VALIDATION / BACKTESTING ====================

export interface ValidationSummaryMetric {
  sample_size?: number
  expected_roi_mean?: number | null
  actual_roi_mean?: number | null
  mae_roi?: number | null
  rmse_roi?: number | null
  directional_accuracy?: number | null
  optimism_bias_roi?: number | null
}

export interface ValidationOverview {
  current_params: Record<string, unknown>
  active_parameter_set: Record<string, unknown> | null
  parameter_spec_count: number
  parameter_set_count: number
  latest_optimization: Record<string, unknown> | null
  opportunity_stats: Record<string, unknown>
  strategy_accuracy: Record<string, unknown>
  roi_30d: Record<string, unknown>
  decay_30d: Record<string, unknown>
  calibration_90d: {
    window_days: number
    sample_size: number
    overall: ValidationSummaryMetric
    by_strategy: Record<string, ValidationSummaryMetric>
  }
  calibration_trend_90d: Array<{
    bucket_start: string
    sample_size: number
    mae_roi: number
    directional_accuracy: number
  }>
  combinatorial_validation: Record<string, unknown>
  strategy_health: ValidationStrategyHealth[]
  guardrail_config: ValidationGuardrailConfig
  jobs: ValidationJob[]
}

export interface BacktestRequest {
  params?: Record<string, unknown>
  save_parameter_set?: boolean
  parameter_set_name?: string
  activate_saved_set?: boolean
}

export interface OptimizationRequest {
  method?: 'grid' | 'random'
  param_ranges?: Record<string, unknown>
  n_random_samples?: number
  random_seed?: number
  walk_forward?: boolean
  n_windows?: number
  train_ratio?: number
  top_k?: number
  save_best_as_active?: boolean
  best_set_name?: string
}

export interface ValidationJob {
  id: string
  job_type: 'backtest' | 'optimize' | string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string
  payload?: Record<string, unknown>
  result?: Record<string, unknown>
  error?: string | null
  progress?: number
  message?: string | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
}

export interface ValidationGuardrailConfig {
  enabled: boolean
  min_samples: number
  min_directional_accuracy: number
  max_mae_roi: number
  lookback_days: number
  auto_promote: boolean
}

export interface ValidationStrategyHealth {
  strategy_type: string
  status: 'active' | 'demoted' | string
  sample_size: number
  directional_accuracy?: number | null
  mae_roi?: number | null
  rmse_roi?: number | null
  optimism_bias_roi?: number | null
  last_reason?: string | null
  manual_override?: boolean
  manual_override_note?: string | null
  demoted_at?: string | null
  restored_at?: string | null
  updated_at?: string | null
}

export const getValidationOverview = async (): Promise<ValidationOverview> => {
  const { data } = await api.get('/validation/overview')
  return data
}

export const runValidationBacktest = async (payload?: BacktestRequest): Promise<{
  status: string
  job_id: string
}> => {
  const { data } = await api.post('/validation/jobs/backtest', payload || {})
  return data
}

export const runValidationOptimization = async (payload?: OptimizationRequest): Promise<{
  status: string
  job_id: string
}> => {
  const { data } = await api.post('/validation/jobs/optimize', payload || {})
  return data
}

export const getValidationJobs = async (limit = 50): Promise<{ jobs: ValidationJob[] }> => {
  const { data } = await api.get('/validation/jobs', { params: { limit } })
  return data
}

export const getValidationJob = async (jobId: string): Promise<ValidationJob> => {
  const { data } = await api.get(`/validation/jobs/${jobId}`)
  return data
}

export const cancelValidationJob = async (jobId: string): Promise<{ status: string; job_id: string }> => {
  const { data } = await api.post(`/validation/jobs/${jobId}/cancel`)
  return data
}

export const getValidationGuardrailConfig = async (): Promise<ValidationGuardrailConfig> => {
  const { data } = await api.get('/validation/guardrails/config')
  return data
}

export const updateValidationGuardrailConfig = async (patch: Partial<ValidationGuardrailConfig>): Promise<ValidationGuardrailConfig> => {
  const { data } = await api.put('/validation/guardrails/config', patch)
  return data
}

export const evaluateValidationGuardrails = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/validation/guardrails/evaluate')
  return data
}

export const getValidationStrategyHealth = async (): Promise<{ strategy_health: ValidationStrategyHealth[] }> => {
  const { data } = await api.get('/validation/strategy-health')
  return data
}

export const overrideValidationStrategy = async (
  strategyType: string,
  status: 'active' | 'demoted',
  note?: string
): Promise<Record<string, unknown>> => {
  const { data } = await api.post(`/validation/strategy-health/${strategyType}/override`, null, {
    params: { status, note }
  })
  return data
}

export const clearValidationStrategyOverride = async (strategyType: string): Promise<Record<string, unknown>> => {
  const { data } = await api.delete(`/validation/strategy-health/${strategyType}/override`)
  return data
}

export const getOptimizationResults = async (topK = 50): Promise<{
  count: number
  results: Array<Record<string, unknown>>
}> => {
  const { data } = await api.get('/validation/optimization-results', { params: { top_k: topK } })
  return data
}

export const getValidationParameterSets = async (): Promise<{
  count: number
  parameter_sets: Array<Record<string, unknown>>
}> => {
  const { data } = await api.get('/validation/parameter-sets')
  return data
}

export const activateValidationParameterSet = async (setId: string): Promise<{
  status: string
  active_set_id: string
}> => {
  const { data } = await api.post(`/validation/parameter-sets/${setId}/activate`)
  return data
}

// ==================== AI INTELLIGENCE ====================

// AI endpoints that invoke LLM calls need a longer timeout than the default 15s
const AI_TIMEOUT = { timeout: 120_000 }

export const getAIStatus = () => api.get('/ai/status')
export const analyzeResolution = (data: any) => api.post('/ai/resolution/analyze', data, AI_TIMEOUT)
export const getResolutionAnalysis = (marketId: string) => api.get(`/ai/resolution/${marketId}`)
export const judgeOpportunity = (data: any) => api.post('/ai/judge/opportunity', data, AI_TIMEOUT)
export const judgeOpportunitiesBulk = (data?: { opportunity_ids?: string[]; force?: boolean }) =>
  api.post('/ai/judge/opportunities/bulk', data || {}, AI_TIMEOUT)
export const getJudgmentHistory = (params?: any) => api.get('/ai/judge/history', { params })
export const getAgreementStats = () => api.get('/ai/judge/agreement-stats')
export const analyzeMarket = (data: any) => api.post('/ai/market/analyze', data, AI_TIMEOUT)
export const analyzeNewsSentiment = (data: any) => api.post('/ai/news/sentiment', data, AI_TIMEOUT)

// ==================== NEWS INTELLIGENCE ====================

export interface NewsArticle {
  article_id: string
  title: string
  source: string
  feed_source: string
  url: string
  published: string | null
  category: string
  summary: string
  has_embedding: boolean
  fetched_at: string
}

export interface NewsFeedStatus {
  article_count: number
  sources: Record<string, number>
  running: boolean
  matcher: {
    initialized: boolean
    mode: string
    articles_embedded: number
    markets_indexed: number
  }
}

export interface NewsMatch {
  article_id: string
  article_title: string
  article_source: string
  article_url: string
  market_question: string
  market_id: string
  market_price: number
  similarity: number
  match_method: string
}

export interface NewsEdge {
  article_title: string
  article_source: string
  article_url: string
  market_question: string
  market_id: string
  market_price: number
  model_probability: number
  edge_percent: number
  direction: string
  confidence: number
  reasoning: string
  similarity: number
  estimated_at: string
}

export interface NewsEdgesResponse {
  total_articles: number
  total_markets: number
  total_matches: number
  total_edges: number
  edges: NewsEdge[]
}

export interface NewsMatchResponse {
  total_articles: number
  total_markets: number
  total_matches: number
  matcher_mode: string
  matches: NewsMatch[]
}

export interface ForecastAgent {
  name: string
  probability: number
  confidence: number
  reasoning: string
  model: string
}

export interface ForecastResult {
  market_question: string
  market_price: number
  final_probability: number
  edge_percent: number
  direction: string
  confidence: number
  aggregation_method: string
  news_context: string
  analyzed_at: string
  agents: ForecastAgent[]
}

export const getNewsFeedStatus = async (): Promise<NewsFeedStatus> => {
  const { data } = await api.get('/news/feed/status')
  return data
}

export const triggerNewsFetch = async (): Promise<{ new_articles: number; total_articles: number; articles: Array<{ title: string; source: string; feed_source: string; url: string; published: string | null; category: string }> }> => {
  const { data } = await api.post('/news/feed/fetch')
  return data
}

export const getNewsArticles = async (params?: {
  max_age_hours?: number
  source?: string
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; has_more: boolean; articles: NewsArticle[] }> => {
  const { data } = await api.get('/news/feed/articles', { params })
  return data
}

export const searchNewsArticles = async (params: {
  q: string
  max_age_hours?: number
  limit?: number
}): Promise<{ query: string; total: number; articles: NewsArticle[] }> => {
  const { data } = await api.get('/news/feed/search', { params })
  return data
}

export const clearNewsArticles = async (): Promise<{ cleared: number }> => {
  const { data } = await api.delete('/news/feed/clear')
  return data
}

export const runNewsMatching = async (params?: {
  max_age_hours?: number
  top_k?: number
  threshold?: number
}): Promise<NewsMatchResponse> => {
  const { data } = await api.post('/news/match', params || {}, { timeout: 120_000 })
  return data
}

export const getNewsEdgesCached = async (): Promise<{ total_edges: number; edges: NewsEdge[] }> => {
  const { data } = await api.get('/news/edges')
  return data
}

export const detectNewsEdges = async (params?: {
  max_age_hours?: number
  top_k?: number
  threshold?: number
  model?: string
}): Promise<NewsEdgesResponse> => {
  const { data } = await api.post('/news/edges', params || {})
  return data
}

export const analyzeNewsEdgeSingle = async (params: {
  article_id: string
  market_id: string
  model?: string
}): Promise<{ edge: NewsEdge | null; message?: string }> => {
  const { data } = await api.post('/news/edges/single', params)
  return data
}

export const runForecastCommittee = async (params: {
  market_question: string
  market_price: number
  news_context?: string
  event_title?: string
  category?: string
  model?: string
}): Promise<ForecastResult> => {
  const { data } = await api.post('/news/forecast', params)
  return data
}

export const forecastMarketById = async (params: {
  market_id: string
  model?: string
  include_news?: boolean
  max_articles?: number
}): Promise<ForecastResult> => {
  const { data } = await api.post('/news/forecast/market', params, AI_TIMEOUT)
  return data
}
export const listSkills = () => api.get('/ai/skills')
export const executeSkill = (data: any) => api.post('/ai/skills/execute', data, AI_TIMEOUT)
export const getResearchSessions = (params?: any) => api.get('/ai/sessions', { params })
export const getResearchSession = (sessionId: string) => api.get(`/ai/sessions/${sessionId}`)
export const getAIUsage = () => api.get('/ai/usage')

// ==================== AI DEEP INTEGRATION ====================

export interface MarketSearchResult {
  market_id: string
  question: string
  yes_price: number | null
  no_price: number | null
  liquidity: number | null
  event_title: string | null
  category: string | null
}

export const searchMarkets = async (q: string, limit = 10): Promise<{ results: MarketSearchResult[]; total: number }> => {
  const { data } = await api.get('/ai/markets/search', { params: { q, limit } })
  return data
}

export interface OpportunityAISummary {
  opportunity_id: string
  judgment: {
    overall_score: number
    profit_viability: number
    resolution_safety: number
    execution_feasibility: number
    market_efficiency: number
    recommendation: string
    reasoning: string
  } | null
  resolution_analyses: Array<{
    market_id: string
    clarity_score: number
    risk_score: number
    confidence: number
    recommendation: string
    summary: string
    ambiguities: string[]
    edge_cases: string[]
  }>
}

export const getOpportunityAISummary = async (opportunityId: string): Promise<OpportunityAISummary> => {
  const { data } = await api.get(`/ai/opportunity/${opportunityId}/summary`)
  return data
}

export interface AIChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AIChatResponse {
  session_id: string
  response: string
  model: string
  tokens_used: Record<string, number>
}

export const sendAIChat = async (params: {
  message: string
  session_id?: string
  context_type?: string
  context_id?: string
  history?: AIChatMessage[]
}): Promise<AIChatResponse> => {
  const { data } = await api.post('/ai/chat', params, AI_TIMEOUT)
  return data
}

export interface AIChatSession {
  session_id: string
  context_type: string | null
  context_id: string | null
  title: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AIChatSessionDetail extends AIChatSession {
  messages: Array<{
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    created_at: string | null
  }>
}

export const listAIChatSessions = async (params?: {
  context_type?: string
  context_id?: string
  limit?: number
}): Promise<{ sessions: AIChatSession[]; total: number }> => {
  const { data } = await api.get('/ai/chat/sessions', { params })
  return data
}

export const getAIChatSession = async (sessionId: string): Promise<AIChatSessionDetail> => {
  const { data } = await api.get(`/ai/chat/sessions/${sessionId}`)
  return data
}

export const archiveAIChatSession = async (sessionId: string): Promise<{ status: string; session_id: string }> => {
  const { data } = await api.delete(`/ai/chat/sessions/${sessionId}`)
  return data
}

// ==================== KALSHI ACCOUNT ====================

export interface KalshiAccountStatus {
  platform: string
  authenticated: boolean
  member_id: string | null
  email: string | null
  balance: {
    balance: number
    payout: number
    available: number
    reserved: number
    currency: string
  } | null
  positions_count: number
}

export interface KalshiPosition {
  token_id: string
  market_id: string
  event_slug?: string
  market_question: string
  outcome: string
  size: number
  average_cost: number
  current_price: number
  unrealized_pnl: number
  platform: string
}

export const getKalshiStatus = async (): Promise<KalshiAccountStatus> => {
  const { data } = await api.get('/kalshi/status')
  return data
}

export const loginKalshi = async (params: {
  email?: string
  password?: string
  api_key?: string
}): Promise<{ status: string; message: string; authenticated: boolean; member_id?: string }> => {
  const { data } = await api.post('/kalshi/login', params)
  return data
}

export const logoutKalshi = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/kalshi/logout')
  return data
}

export const getKalshiBalance = async (): Promise<{ balance: number; available: number; reserved: number; currency: string }> => {
  const { data } = await api.get('/kalshi/balance')
  return data
}

export const getKalshiPositions = async (): Promise<KalshiPosition[]> => {
  const { data } = await api.get('/kalshi/positions')
  return data
}

export const updateKalshiSettings = async (settings: Partial<KalshiSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/kalshi', settings)
  return data
}

// ==================== NEWS WORKFLOW (Independent Pipeline) ====================

export interface NewsWorkflowFinding {
  id: string
  article_id: string
  market_id: string
  article_title: string
  article_source: string
  article_url: string
  signal_key?: string | null
  cache_key?: string | null
  market_question: string
  market_price: number
  model_probability: number
  edge_percent: number
  direction: string
  confidence: number
  retrieval_score: number
  semantic_score: number
  keyword_score: number
  event_score: number
  rerank_score: number
  event_graph: Record<string, unknown>
  evidence: Record<string, unknown>
  reasoning: string
  actionable: boolean
  consumed_by_auto_trader: boolean
  created_at: string
}

export interface NewsTradeIntent {
  id: string
  signal_key?: string | null
  finding_id: string
  market_id: string
  market_question: string
  direction: string
  entry_price: number
  model_probability: number
  edge_percent: number
  confidence: number
  suggested_size_usd: number
  metadata?: {
    market?: {
      id?: string
      slug?: string
      event_slug?: string
      event_title?: string
      liquidity?: number
      yes_price?: number
      no_price?: number
      token_ids?: string[]
    }
    finding?: {
      article_id?: string
      signal_key?: string
      cache_key?: string
    }
  }
  status: string
  created_at: string
  consumed_at: string | null
}

export interface NewsWorkflowStatus {
  running: boolean
  enabled: boolean
  paused: boolean
  interval_seconds: number
  last_scan: string | null
  next_scan: string | null
  current_activity: string | null
  last_error: string | null
  degraded_mode: boolean
  budget_remaining: number | null
  pending_intents: number
  requested_scan_at: string | null
  stats: Record<string, unknown>
}

export interface NewsWorkflowSettings {
  enabled: boolean
  auto_run: boolean
  scan_interval_seconds: number
  top_k: number
  rerank_top_n: number
  similarity_threshold: number
  keyword_weight: number
  semantic_weight: number
  event_weight: number
  require_verifier: boolean
  market_min_liquidity: number
  market_max_days_to_resolution: number
  min_keyword_signal: number
  min_semantic_signal: number
  min_edge_percent: number
  min_confidence: number
  require_second_source: boolean
  auto_trader_enabled: boolean
  auto_trader_min_edge: number
  auto_trader_max_age_minutes: number
  cycle_spend_cap_usd: number
  hourly_spend_cap_usd: number
  cycle_llm_call_cap: number
  cache_ttl_minutes: number
  max_edge_evals_per_article: number
  model: string | null
}

export const getNewsWorkflowStatus = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.get('/news-workflow/status')
  return data
}

export const runNewsWorkflow = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/news-workflow/run')
  return data
}

export const startNewsWorkflow = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/start')
  return data
}

export const pauseNewsWorkflow = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/pause')
  return data
}

export const setNewsWorkflowInterval = async (intervalSeconds: number): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/interval', null, {
    params: { interval_seconds: intervalSeconds },
  })
  return data
}

export const getNewsWorkflowFindings = async (params?: {
  min_edge?: number
  actionable_only?: boolean
  include_debug_rejections?: boolean
  max_age_hours?: number
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; findings: NewsWorkflowFinding[] }> => {
  const { data } = await api.get('/news-workflow/findings', { params })
  return data
}

export const getNewsWorkflowIntents = async (params?: {
  status_filter?: string
  limit?: number
}): Promise<{ total: number; intents: NewsTradeIntent[] }> => {
  const { data } = await api.get('/news-workflow/intents', { params })
  return data
}

export const skipNewsWorkflowIntent = async (intentId: string): Promise<{ status: string; intent_id: string }> => {
  const { data } = await api.post(`/news-workflow/intents/${intentId}/skip`)
  return data
}

export const getNewsWorkflowSettings = async (): Promise<NewsWorkflowSettings> => {
  const { data } = await api.get('/news-workflow/settings')
  return data
}

export const updateNewsWorkflowSettings = async (
  settings: Partial<NewsWorkflowSettings>
): Promise<{ status: string; settings: NewsWorkflowSettings }> => {
  const { data } = await api.put('/news-workflow/settings', settings)
  return data
}

// ==================== WEATHER WORKFLOW (Independent Pipeline) ====================

export interface WeatherWorkflowStatus {
  running: boolean
  enabled: boolean
  interval_seconds: number
  last_scan: string | null
  opportunities_count: number
  current_activity: string | null
  stats: Record<string, unknown>
  pending_intents: number
  paused: boolean
  requested_scan_at: string | null
}

export interface WeatherWorkflowSettings {
  enabled: boolean
  auto_run: boolean
  scan_interval_seconds: number
  entry_max_price: number
  take_profit_price: number
  stop_loss_pct: number
  min_edge_percent: number
  min_confidence: number
  min_model_agreement: number
  min_liquidity: number
  max_markets_per_scan: number
  auto_trader_enabled: boolean
  auto_trader_min_edge: number
  auto_trader_max_age_minutes: number
  default_size_usd: number
  max_size_usd: number
  model: string | null
}

export interface WeatherTradeIntent {
  id: string
  market_id: string
  market_question: string
  direction: string
  entry_price: number | null
  take_profit_price: number | null
  stop_loss_pct: number | null
  model_probability: number | null
  edge_percent: number | null
  confidence: number | null
  model_agreement: number | null
  suggested_size_usd: number | null
  metadata: Record<string, unknown> | null
  status: string
  created_at: string | null
  consumed_at: string | null
}

export interface WeatherWorkflowPerformance {
  lookback_days: number
  trades_total: number
  trades_resolved: number
  wins: number
  losses: number
  win_rate: number
  total_pnl: number
  intents_total: number
  pending_intents: number
  executed_intents: number
}

export const getWeatherWorkflowStatus = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.get('/weather-workflow/status')
  return data
}

export const runWeatherWorkflow = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/weather-workflow/run')
  return data
}

export const startWeatherWorkflow = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/start')
  return data
}

export const pauseWeatherWorkflow = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/pause')
  return data
}

export const setWeatherWorkflowInterval = async (
  intervalSeconds: number
): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/interval', null, {
    params: { interval_seconds: intervalSeconds }
  })
  return data
}

export const getWeatherWorkflowOpportunities = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; opportunities: Opportunity[] }> => {
  const { data } = await api.get('/weather-workflow/opportunities', { params })
  return data
}

export const getWeatherWorkflowIntents = async (params?: {
  status_filter?: string
  limit?: number
}): Promise<{ total: number; intents: WeatherTradeIntent[] }> => {
  const { data } = await api.get('/weather-workflow/intents', { params })
  return data
}

export const skipWeatherWorkflowIntent = async (intentId: string): Promise<{ status: string; intent_id: string }> => {
  const { data } = await api.post(`/weather-workflow/intents/${intentId}/skip`)
  return data
}

export const getWeatherWorkflowSettings = async (): Promise<WeatherWorkflowSettings> => {
  const { data } = await api.get('/weather-workflow/settings')
  return data
}

export const updateWeatherWorkflowSettings = async (
  settings: Partial<WeatherWorkflowSettings>
): Promise<{ status: string; settings: WeatherWorkflowSettings }> => {
  const { data } = await api.put('/weather-workflow/settings', settings)
  return data
}

export const getWeatherWorkflowPerformance = async (
  lookbackDays = 90
): Promise<WeatherWorkflowPerformance> => {
  const { data } = await api.get('/weather-workflow/performance', {
    params: { lookback_days: lookbackDays }
  })
  return data
}

export default api
