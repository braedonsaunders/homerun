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
  volume?: number
  platform?: string  // "polymarket" | "kalshi"
  weather?: WeatherMarketDetails
  price_history?: Array<{
    t: number
    yes: number
    no: number
  }>
}

export interface WeatherForecastSource {
  source_id: string
  provider: string
  model: string
  value_c: number | null
  value_f: number | null
  probability: number | null
  weight: number | null
  target_time: string | null
}

export interface WeatherMarketDetails {
  location?: string
  metric?: string
  operator?: string
  threshold_c?: number | null
  threshold_c_low?: number | null
  threshold_c_high?: number | null
  raw_threshold?: number | null
  raw_threshold_low?: number | null
  raw_threshold_high?: number | null
  raw_unit?: string | null
  target_time?: string | null
  gfs_probability?: number | null
  ecmwf_probability?: number | null
  gfs_value?: number | null
  ecmwf_value?: number | null
  forecast_sources?: WeatherForecastSource[]
  source_weights?: Record<string, number> | null
  source_count?: number | null
  source_spread_c?: number | null
  source_spread_f?: number | null
  consensus_probability?: number | null
  consensus_temp_c?: number | null
  consensus_temp_f?: number | null
  market_probability?: number | null
  market_implied_temp_c?: number | null
  market_implied_temp_f?: number | null
  agreement?: number | null
  model_confidence?: number | null
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
  domain?: 'event_markets' | 'crypto' | string
  timeframe?: string
  sources?: string[]
  validation_status?: 'active' | 'demoted' | 'unknown' | string
  validation_sample_size?: number
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
  token_id?: string | null
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
  oracle_source: string | null
  oracle_updated_at_ms: number | null
  oracle_age_seconds: number | null
  oracle_prices_by_source?: Record<string, {
    source: string
    price: number | null
    updated_at_ms: number | null
    age_seconds: number | null
  }>
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

// ==================== TRADER ORCHESTRATOR ====================

export interface TraderOrchestratorConfig {
  mode: string
  kill_switch: boolean
  run_interval_seconds: number
  global_risk: {
    max_gross_exposure_usd: number
    max_daily_loss_usd: number
    max_orders_per_cycle: number
  }
  trading_domains: string[]
  enabled_strategies: string[]
  llm_verify_trades: boolean
  paper_account_id?: string | null
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

export interface TraderDecisionCheck {
  id: string
  check_key: string
  check_label: string
  passed: boolean
  score: number | null
  detail: string | null
  payload: Record<string, any>
  created_at: string | null
}

export interface TraderOrchestratorOverview {
  control: {
    is_enabled: boolean
    is_paused: boolean
    mode: string
    run_interval_seconds: number
    requested_run_at: string | null
    kill_switch: boolean
    settings: Record<string, any>
    updated_at: string | null
  }
  worker: {
    running: boolean
    enabled: boolean
    current_activity: string | null
    interval_seconds: number
    signals_seen: number
    signals_selected: number
    decisions_count: number
    trades_count: number
    open_positions: number
    daily_pnl: number
    last_run_at: string | null
    last_error: string | null
    updated_at: string | null
    stats: Record<string, any>
  }
  config: TraderOrchestratorConfig
  metrics: {
    traders_total: number
    traders_running: number
    decisions_count: number
    orders_count: number
    open_orders: number
    gross_exposure_usd: number
    daily_pnl: number
  }
  traders: Trader[]
}

export interface TraderOrchestratorStatus {
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
    signals_seen: number
    signals_selected: number
    trades_count: number
    daily_pnl: number
  }
  config: TraderOrchestratorConfig
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

export interface TraderOrchestratorLivePreflightResponse {
  status: 'passed' | 'failed' | string
  preflight_id: string
  requested_mode: string
  checks: Array<Record<string, any>>
  failed_checks: Array<Record<string, any>>
}

export interface TraderOrchestratorLiveArmResponse {
  status: 'armed' | string
  preflight_id: string
  arm_token: string
  expires_at: string
  ttl_seconds: number
}

const mapOverviewToStatus = (overview: TraderOrchestratorOverview): TraderOrchestratorStatus => {
  const control = overview.control || ({} as TraderOrchestratorOverview['control'])
  const worker = overview.worker || ({} as TraderOrchestratorOverview['worker'])
  const metrics = overview.metrics || ({} as TraderOrchestratorOverview['metrics'])
  const tradingActive = Boolean(control.is_enabled) && !Boolean(control.is_paused) && !Boolean(control.kill_switch)
  const totalTrades = Number(metrics.orders_count || 0)

  return {
    mode: control.mode || 'paper',
    running: tradingActive,
    trading_active: tradingActive,
    worker_running: Boolean(worker.running),
    control: {
      is_enabled: Boolean(control.is_enabled),
      is_paused: Boolean(control.is_paused),
      kill_switch: Boolean(control.kill_switch),
      requested_run_at: control.requested_run_at || null,
      run_interval_seconds: Number(control.run_interval_seconds || 2),
      updated_at: control.updated_at || null,
    },
    snapshot: {
      running: Boolean(worker.running),
      enabled: Boolean(worker.enabled),
      current_activity: worker.current_activity || null,
      interval_seconds: Number(worker.interval_seconds || 2),
      last_run_at: worker.last_run_at || null,
      last_error: worker.last_error || null,
      updated_at: worker.updated_at || null,
      signals_seen: Number(worker.signals_seen || 0),
      signals_selected: Number(worker.signals_selected || 0),
      trades_count: Number(worker.trades_count || 0),
      daily_pnl: Number(worker.daily_pnl || 0),
    },
    config: overview.config,
    stats: {
      total_trades: totalTrades,
      winning_trades: 0,
      losing_trades: 0,
      win_rate: 0,
      total_profit: Number(metrics.daily_pnl || 0),
      total_invested: Number(metrics.gross_exposure_usd || 0),
      roi_percent: 0,
      daily_trades: Number(metrics.orders_count || 0),
      daily_profit: Number(metrics.daily_pnl || 0),
      consecutive_losses: 0,
      circuit_breaker_active: Boolean(control.kill_switch),
      last_trade_at: worker.last_run_at || null,
      opportunities_seen: Number(metrics.decisions_count || 0),
      opportunities_executed: Number(metrics.orders_count || 0),
      opportunities_skipped: Math.max(0, Number(metrics.decisions_count || 0) - Number(metrics.orders_count || 0)),
    },
  }
}

export const getTraderOrchestratorOverview = async (): Promise<TraderOrchestratorOverview> => {
  const { data } = await api.get('/trader-orchestrator/overview')
  return data
}

export const getTraderOrchestratorStatus = async (): Promise<TraderOrchestratorStatus> => {
  const overview = await getTraderOrchestratorOverview()
  return mapOverviewToStatus(overview)
}

export const startTraderOrchestrator = async (payload?: {
  mode?: string
  paper_account_id?: string
  requested_by?: string
}): Promise<{ status: string; mode: string; message: string }> => {
  const { data } = await api.post('/trader-orchestrator/start', {
    mode: payload?.mode || 'paper',
    paper_account_id: payload?.paper_account_id,
    requested_by: payload?.requested_by,
  })
  return {
    status: data.status || 'started',
    mode: data.control?.mode || payload?.mode || 'paper',
    message: 'Trader orchestrator start command submitted.',
  }
}

export const stopTraderOrchestrator = async (): Promise<{ status: string }> => {
  const { data } = await api.post('/trader-orchestrator/stop')
  return { status: data.status || 'stopped' }
}

export const runTraderOrchestratorLivePreflight = async (payload?: {
  mode?: string
  requested_by?: string
}): Promise<TraderOrchestratorLivePreflightResponse> => {
  const { data } = await api.post('/trader-orchestrator/live/preflight', {
    mode: payload?.mode || 'live',
    requested_by: payload?.requested_by,
  })
  return data
}

export const armTraderOrchestratorLiveStart = async (payload: {
  preflight_id: string
  ttl_seconds?: number
  requested_by?: string
}): Promise<TraderOrchestratorLiveArmResponse> => {
  const { data } = await api.post('/trader-orchestrator/live/arm', {
    preflight_id: payload.preflight_id,
    ttl_seconds: payload.ttl_seconds ?? 300,
    requested_by: payload.requested_by,
  })
  return data
}

export const startTraderOrchestratorLive = async (payload: {
  arm_token: string
  mode?: string
  requested_by?: string
}) => {
  const { data } = await api.post('/trader-orchestrator/live/start', {
    arm_token: payload.arm_token,
    mode: payload.mode || 'live',
    requested_by: payload.requested_by,
  })
  return data
}

export const stopTraderOrchestratorLive = async (requestedBy?: string) => {
  const { data } = await api.post('/trader-orchestrator/live/stop', {
    requested_by: requestedBy,
  })
  return data
}

export const setTraderOrchestratorLiveKillSwitch = async (enabled: boolean, requestedBy?: string) => {
  const { data } = await api.post('/trader-orchestrator/kill-switch', {
    enabled,
    requested_by: requestedBy,
  })
  return data
}

export interface Trader {
  id: string
  name: string
  description?: string | null
  strategy_key: string
  strategy_version: string
  sources: string[]
  params: Record<string, any>
  risk_limits: Record<string, any>
  metadata: Record<string, any>
  is_enabled: boolean
  is_paused: boolean
  interval_seconds: number
  requested_run_at: string | null
  last_run_at: string | null
  next_run_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface TraderTemplate {
  id: string
  name: string
  description?: string | null
  strategy_key: string
  sources: string[]
  interval_seconds: number
  params: Record<string, any>
  risk_limits: Record<string, any>
}

export interface TraderDecision {
  id: string
  trader_id: string
  signal_id: string | null
  source: string
  strategy_key: string
  decision: string
  reason: string | null
  score: number | null
  event_id: string | null
  trace_id: string | null
  checks_summary: Record<string, any>
  risk_snapshot: Record<string, any>
  payload: Record<string, any>
  created_at: string | null
}

export interface TraderOrder {
  id: string
  trader_id: string
  signal_id: string | null
  decision_id: string | null
  source: string
  market_id: string
  market_question: string | null
  direction: string | null
  mode: string
  status: string
  notional_usd: number | null
  entry_price: number | null
  effective_price: number | null
  edge_percent: number | null
  confidence: number | null
  actual_profit: number | null
  reason: string | null
  payload: Record<string, any>
  error_message: string | null
  event_id: string | null
  trace_id: string | null
  created_at: string | null
  executed_at: string | null
  updated_at: string | null
}

export interface TraderEvent {
  id: string
  trader_id: string | null
  event_type: string
  severity: string
  source: string | null
  operator: string | null
  message: string | null
  trace_id: string | null
  payload: Record<string, any>
  created_at: string | null
}

export interface TraderDecisionDetail {
  decision: TraderDecision
  checks: TraderDecisionCheck[]
  orders: TraderOrder[]
}

export interface TraderSource {
  key: string
  label: string
  description: string
  domains: string[]
  signal_types: string[]
}

export const getTraders = async (): Promise<Trader[]> => {
  const { data } = await api.get('/traders')
  return data.traders || []
}

export const createTrader = async (payload: Record<string, any>): Promise<Trader> => {
  const { data } = await api.post('/traders', payload)
  return data
}

export const getTrader = async (traderId: string): Promise<Trader> => {
  const { data } = await api.get(`/traders/${traderId}`)
  return data
}

export const updateTrader = async (traderId: string, payload: Record<string, any>): Promise<Trader> => {
  const { data } = await api.put(`/traders/${traderId}`, payload)
  return data
}

export const deleteTrader = async (traderId: string): Promise<{ status: string; trader_id: string }> => {
  const { data } = await api.delete(`/traders/${traderId}`)
  return data
}

export const startTrader = async (traderId: string): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/start`)
  return data
}

export const pauseTrader = async (traderId: string): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/pause`)
  return data
}

export const runTraderOnce = async (traderId: string): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/run-once`)
  return data
}

export const getTraderDecisions = async (
  traderId: string,
  params?: { decision?: string; limit?: number }
): Promise<TraderDecision[]> => {
  const { data } = await api.get(`/traders/${traderId}/decisions`, { params })
  return data.decisions || []
}

export const getTraderOrders = async (
  traderId: string,
  params?: { status?: string; limit?: number }
): Promise<TraderOrder[]> => {
  const { data } = await api.get(`/traders/${traderId}/orders`, { params })
  return data.orders || []
}

export const getAllTraderOrders = async (limitPerTrader = 150): Promise<TraderOrder[]> => {
  const traders = await getTraders()
  const rows = await Promise.all(
    traders.map((trader) => getTraderOrders(trader.id, { limit: limitPerTrader }))
  )
  return rows
    .flat()
    .sort((a, b) => {
      const left = new Date(a.created_at || 0).getTime()
      const right = new Date(b.created_at || 0).getTime()
      return right - left
    })
}

export const getTraderEvents = async (
  traderId: string,
  params?: { cursor?: string; limit?: number; types?: string[] }
): Promise<{ events: TraderEvent[]; next_cursor: string | null }> => {
  const { data } = await api.get(`/traders/${traderId}/events`, {
    params: {
      cursor: params?.cursor,
      limit: params?.limit ?? 200,
      types: params?.types?.join(','),
    },
  })
  return {
    events: data.events || [],
    next_cursor: data.next_cursor || null,
  }
}

export const getTraderDecisionDetail = async (decisionId: string): Promise<TraderDecisionDetail> => {
  const { data } = await api.get(`/traders/decisions/${decisionId}`)
  return data
}

export const getTraderTemplates = async (): Promise<TraderTemplate[]> => {
  const { data } = await api.get('/traders/templates')
  return data.templates || []
}

export const createTraderFromTemplate = async (payload: {
  template_id: string
  overrides?: Record<string, any>
  requested_by?: string
}): Promise<Trader> => {
  const { data } = await api.post('/traders/from-template', payload)
  return data
}

export const getTraderSources = async (): Promise<TraderSource[]> => {
  const { data } = await api.get('/trader-sources')
  return data.sources || []
}

export const getTraderOrchestratorStats = async (): Promise<TraderOrchestratorStatus['stats']> => {
  const overview = await getTraderOrchestratorOverview()
  const metrics: any = overview?.metrics || {}
  const worker: any = overview?.worker || {}
  const control: any = overview?.control || {}
  return {
    total_trades: Number(metrics.orders_count || 0),
    winning_trades: 0,
    losing_trades: 0,
    win_rate: 0,
    total_profit: Number(metrics.daily_pnl || 0),
    total_invested: Number(metrics.gross_exposure_usd || 0),
    roi_percent: 0,
    daily_trades: Number(metrics.orders_count || 0),
    daily_profit: Number(metrics.daily_pnl || 0),
    consecutive_losses: 0,
    circuit_breaker_active: Boolean(control.kill_switch),
    last_trade_at: worker.last_run_at || null,
    opportunities_seen: Number(metrics.decisions_count || 0),
    opportunities_executed: Number(metrics.orders_count || 0),
    opportunities_skipped: Math.max(0, Number(metrics.decisions_count || 0) - Number(metrics.orders_count || 0)),
  }
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

export const pauseAllWorkers = async (): Promise<{ status: string; workers: WorkerStatus[] }> => {
  const { data } = await api.post('/workers/pause-all')
  return data
}

export const resumeAllWorkers = async (): Promise<{ status: string; workers: WorkerStatus[] }> => {
  const { data } = await api.post('/workers/resume-all')
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

export type DatabaseFlushTarget = 'scanner' | 'weather' | 'news' | 'trader_orchestrator' | 'all'

export interface DatabaseFlushResponse {
  status: string
  target: DatabaseFlushTarget
  timestamp: string
  flushed: Record<string, Record<string, number>>
  protected_datasets: string[]
  message: string
}

export const flushDatabaseData = async (target: DatabaseFlushTarget): Promise<DatabaseFlushResponse> => {
  const { data } = await api.post('/maintenance/flush', {
    target,
    confirm: true,
  })
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
  notify_autotrader_orders: boolean
  notify_autotrader_issues: boolean
  notify_autotrader_timeline: boolean
  notify_autotrader_summary_interval_minutes: number
  notify_autotrader_summary_per_trader: boolean
}

export interface ScannerSettings {
  scan_interval_seconds: number
  min_profit_threshold: number
  max_markets_to_scan: number
  min_liquidity: number
}

export interface DiscoverySettings {
  max_discovered_wallets: number
  maintenance_enabled: boolean
  keep_recent_trade_days: number
  keep_new_discoveries_days: number
  maintenance_batch: number
  stale_analysis_hours: number
  analysis_priority_batch_limit: number
  delay_between_markets: number
  delay_between_wallets: number
  max_markets_per_run: number
  max_wallets_per_market: number
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
  market_cache_hygiene_enabled: boolean
  market_cache_hygiene_interval_hours: number
  market_cache_retention_days: number
  market_cache_reference_lookback_days: number
  market_cache_weak_entry_grace_days: number
  market_cache_max_entries_per_slug: number
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
  btc_eth_hf_series_btc_5m: string
  btc_eth_hf_series_eth_5m: string
  btc_eth_hf_series_sol_5m: string
  btc_eth_hf_series_xrp_5m: string
  btc_eth_hf_series_btc_1h: string
  btc_eth_hf_series_eth_1h: string
  btc_eth_hf_series_sol_1h: string
  btc_eth_hf_series_xrp_1h: string
  btc_eth_hf_series_btc_4h: string
  btc_eth_hf_series_eth_4h: string
  btc_eth_hf_series_sol_4h: string
  btc_eth_hf_series_xrp_4h: string
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
  discovery: DiscoverySettings
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
  discovery?: Partial<DiscoverySettings>
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

export const getDiscoverySettings = async (): Promise<DiscoverySettings> => {
  const { data } = await api.get('/settings/discovery')
  return data
}

export const updateDiscoverySettings = async (settings: Partial<DiscoverySettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/discovery', settings)
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
  trader_orchestrator_execution_30d?: {
    window_days: number
    sample_size: number
    executed_or_open: number
    failed: number
    failure_rate: number
    notional_total_usd: number
    realized_pnl_total: number
    by_source: Array<Record<string, unknown>>
    by_strategy: Array<Record<string, unknown>>
  }
  world_intel_resolver_7d?: {
    window_days: number
    signals_sampled: number
    candidates: number
    tradable: number
    tradable_rate: number
    by_signal_type: Array<Record<string, unknown>>
  }
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
  consumed_by_orchestrator: boolean
  supporting_articles?: NewsSupportingArticle[]
  supporting_article_count?: number
  created_at: string
}

export interface NewsSupportingArticle {
  article_id: string
  title: string
  url: string
  source: string
  published?: string | null
  fetched_at?: string | null
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
    supporting_articles?: NewsSupportingArticle[]
    supporting_article_count?: number
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

export interface NewsRssFeedConfig {
  id: string
  name: string
  url: string
  enabled: boolean
  category: string
}

export interface NewsGovRssFeedConfig {
  id: string
  agency: string
  name: string
  url: string
  priority: 'critical' | 'high' | 'medium' | 'low'
  country_iso3: string
  enabled: boolean
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
  orchestrator_enabled: boolean
  orchestrator_min_edge: number
  orchestrator_max_age_minutes: number
  cycle_spend_cap_usd: number
  hourly_spend_cap_usd: number
  cycle_llm_call_cap: number
  cache_ttl_minutes: number
  max_edge_evals_per_article: number
  rss_feeds: NewsRssFeedConfig[]
  gov_rss_enabled: boolean
  gov_rss_feeds: NewsGovRssFeedConfig[]
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
  orchestrator_enabled: boolean
  orchestrator_min_edge: number
  orchestrator_max_age_minutes: number
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
