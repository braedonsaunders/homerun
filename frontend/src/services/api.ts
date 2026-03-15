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
    const status = error?.response?.status
    if (status === 423 && typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('ui-lock-required'))
    }
    if (status !== 423) {
      console.error(
        `[API] ${error.config?.method?.toUpperCase()} ${error.config?.url} → ${status || error.message}`,
        error.response?.data
      )
    }
    return Promise.reject(error)
  }
)

function unwrapApiData(data: any): any {
  if (Array.isArray(data)) {
    return data
  }
  if (data && typeof data === 'object' && 'items' in data) {
    return data.items
  }
  return data
}

function unwrapStrategyManagerPayload(data: any): any {
  const first = unwrapApiData(data)
  if (first && typeof first === 'object' && !Array.isArray(first) && 'data' in first) {
    return unwrapApiData((first as any).data)
  }
  return first
}

function getStrategyManagerItems(data: any): any[] {
  const payload = unwrapStrategyManagerPayload(data)
  if (Array.isArray(payload)) {
    return payload
  }
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

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
  strategy_subtype?: string | null
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
  confidence?: number
  markets: Market[]
  polymarket_url?: string | null
  kalshi_url?: string | null
  event_id?: string
  event_slug?: string
  event_title?: string
  category?: string
  min_liquidity: number
  volume?: number
  max_position_size: number
  detected_at: string
  first_detected_at?: string | null
  last_detected_at?: string | null
  last_priced_at?: string | null
  last_seen_at?: string | null
  resolution_date?: string
  positions_to_take: Position[]
  strategy_context?: Record<string, unknown> | null
  revision?: number | null
  ai_analysis: AIAnalysis | null
}

export interface Market {
  id: string
  condition_id?: string
  conditionId?: string
  slug?: string
  event_slug?: string
  event_ticker?: string
  eventTicker?: string
  url?: string
  market_url?: string
  question: string
  yes_price: number
  no_price: number
  current_yes_price?: number | null
  current_no_price?: number | null
  price_updated_at?: string | null
  price_age_seconds?: number | null
  is_price_fresh?: boolean
  outcome_labels?: string[]
  outcomes?: unknown[]
  outcome_prices?: number[]
  tokens?: Array<{
    token_id?: string
    outcome?: string
    name?: string
    label?: string
    price?: number | null
  }>
  liquidity: number
  volume?: number
  platform?: string  // "polymarket" | "kalshi"
  weather?: WeatherMarketDetails
  price_history?: Array<Record<string, unknown> | unknown[]>
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
  last_fast_scan?: string | null
  last_heavy_scan?: string | null
  opportunities_count: number
  current_activity?: string | null
  lane_watchdogs?: Record<string, unknown> | null
  strategies: Strategy[]
}

export interface Strategy {
  type: string
  name: string
  description: string
  is_plugin?: boolean
  is_system?: boolean
  plugin_id?: string
  plugin_slug?: string
  source_key?: string
  enabled?: boolean
  status?: string  // For plugins: loaded, error, unloaded
  error_message?: string | null
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

export interface OpportunityIdsResponse {
  total: number
  offset: number
  limit: number
  ids: string[]
}

export const getOpportunities = async (params?: {
  source?: 'markets' | 'traders' | 'all'
  min_profit?: number
  max_risk?: number
  strategy?: string
  sub_strategy?: string
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
  const payload = unwrapApiData(response.data)
  const opportunities = Array.isArray(payload)
    ? payload
    : (payload && typeof payload === 'object' && Array.isArray((payload as { opportunities?: unknown }).opportunities)
      ? (payload as { opportunities: Opportunity[] }).opportunities
      : [])
  const headerTotal = parseInt(response.headers['x-total-count'] || '', 10)
  const payloadTotal = payload && typeof payload === 'object'
    ? Number((payload as { total?: unknown }).total)
    : Number.NaN
  const total = Number.isFinite(headerTotal)
    ? headerTotal
    : (Number.isFinite(payloadTotal) ? payloadTotal : opportunities.length)
  return {
    opportunities,
    total
  }
}

export const getOpportunityIds = async (params?: {
  source?: 'markets' | 'traders' | 'all'
  min_profit?: number
  max_risk?: number
  strategy?: string
  sub_strategy?: string
  min_liquidity?: number
  search?: string
  category?: string
  sort_by?: string
  sort_dir?: string
  exclude_strategy?: string
  limit?: number
  offset?: number
}): Promise<OpportunityIdsResponse> => {
  const { data } = await api.get('/opportunities/ids', { params })
  return unwrapApiData(data)
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
  strategy_sdk?: string | null
  strategy_key?: string | null
  strategy?: string | null
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

export const getCryptoMarkets = async (params?: { viewer_active?: boolean }): Promise<CryptoMarket[]> => {
  const { data } = await api.get('/crypto/markets', { params })
  return unwrapApiData(data)
}

// ==================== OPPORTUNITY COUNTS ====================

export interface OpportunityCounts {
  strategies: Record<string, number>
  categories: Record<string, number>
  sub_strategies?: Record<string, number>
}

export const getOpportunityCounts = async (params?: {
  source?: 'markets' | 'traders' | 'all'
  min_profit?: number
  max_risk?: number
  min_liquidity?: number
  search?: string
  strategy?: string
  sub_strategy?: string
  category?: string
}): Promise<OpportunityCounts> => {
  const { data } = await api.get('/opportunities/counts', { params })
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

export const triggerScan = async () => {
  const { data } = await api.post('/scan')
  return unwrapApiData(data)
}

export const clearOpportunities = async () => {
  const { data } = await api.delete('/opportunities')
  return unwrapApiData(data)
}

// ==================== SCANNER ====================

export const getScannerStatus = async (): Promise<ScannerStatus> => {
  const { data } = await api.get('/scanner/status')
  return unwrapApiData(data)
}

export const startScanner = async (): Promise<ScannerStatus> => {
  const { data } = await api.post('/scanner/start')
  return unwrapApiData(data)
}

export const pauseScanner = async (): Promise<ScannerStatus> => {
  const { data } = await api.post('/scanner/pause')
  return unwrapApiData(data)
}

export const setScannerInterval = async (intervalSeconds: number): Promise<ScannerStatus> => {
  const { data } = await api.post('/scanner/interval', null, {
    params: { interval_seconds: intervalSeconds }
  })
  return unwrapApiData(data)
}

export const getStrategies = async (): Promise<Strategy[]> => {
  const { data } = await api.get('/strategies')
  return unwrapApiData(data)
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
  source_key: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  is_system: boolean
  enabled: boolean
  status: 'unloaded' | 'loaded' | 'error'
  error_message: string | null
  config: Record<string, unknown>
  config_schema: { param_fields: { key: string; label: string; type: string; min?: number; max?: number; options?: string[] }[] } | null
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
  const { data } = await api.get('/strategy-manager')
  return getStrategyManagerItems(data)
}

export const createPlugin = async (plugin: {
  slug: string
  source_key?: string
  source_code: string
  config?: Record<string, unknown>
  name?: string
  description?: string
  enabled?: boolean
}): Promise<StrategyPlugin> => {
  const { data } = await api.post('/strategy-manager', plugin)
  return unwrapApiData(data)
}

export const updatePlugin = async (
  id: string,
  updates: Partial<{
    slug: string
    source_code: string
    config: Record<string, unknown>
    enabled: boolean
    source_key: string
    name: string
    description: string
    unlock_system: boolean
  }>
): Promise<StrategyPlugin> => {
  const { data } = await api.put(`/strategy-manager/${id}`, updates)
  return unwrapApiData(data)
}

export const deletePlugin = async (id: string): Promise<void> => {
  await api.delete(`/strategy-manager/${id}`)
}

export const validatePlugin = async (source_code: string): Promise<PluginValidation> => {
  const { data } = await api.post('/strategy-manager/validate', { source_code })
  return unwrapApiData(data)
}

export const getPluginTemplate = async (): Promise<{
  template: string
  instructions: string
  available_imports: string[]
}> => {
  const { data } = await api.get('/strategy-manager/template')
  return unwrapApiData(data)
}

export const reloadPlugin = async (id: string): Promise<{
  status: string
  message: string
  runtime: PluginRuntime | null
}> => {
  const { data } = await api.post(`/strategy-manager/${id}/reload`)
  return unwrapApiData(data)
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const getPluginDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/strategy-manager/docs')
  if (Array.isArray(data)) {
    return {}
  }
  return (data?.items && !Array.isArray(data.items) ? data.items : data) ?? {}
}

export type OpportunityStrategyDefinition = StrategyPlugin

export const getOpportunityStrategies = async (): Promise<OpportunityStrategyDefinition[]> => {
  return getPlugins()
}

export const createOpportunityStrategy = async (payload: {
  slug: string
  source_key: string
  source_code: string
  config?: Record<string, unknown>
  name?: string
  description?: string
  enabled?: boolean
}): Promise<OpportunityStrategyDefinition> => {
  return createPlugin(payload)
}

export const updateOpportunityStrategy = async (
  id: string,
  payload: Partial<{
    slug: string
    source_code: string
    config: Record<string, unknown>
    enabled: boolean
    source_key: string
    name: string
    description: string
  }>
): Promise<OpportunityStrategyDefinition> => {
  return updatePlugin(id, payload)
}

export const validateOpportunityStrategy = async (source_code: string): Promise<PluginValidation> => {
  return validatePlugin(source_code)
}

export const reloadOpportunityStrategy = async (id: string): Promise<{
  status: string
  message: string
  runtime: PluginRuntime | null
}> => {
  return reloadPlugin(id)
}

export const deleteOpportunityStrategy = async (id: string): Promise<void> => {
  return deletePlugin(id)
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const getTraderStrategyDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/strategy-manager/docs')
  const payload = unwrapStrategyManagerPayload(data)
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return {}
  }
  return payload
}

// ==================== WALLETS ====================

export const getWallets = async (): Promise<Wallet[]> => {
  const { data } = await api.get('/wallets')
  return unwrapApiData(data)
}

export const addWallet = async (address: string, label?: string): Promise<{ status: string; address: string; label: string | null }> => {
  const { data } = await api.post('/wallets', null, {
    params: { address, label }
  })
  return unwrapApiData(data)
}

export const removeWallet = async (address: string): Promise<{ status: string; message: string }> => {
  const { data } = await api.delete(`/wallets/${address}`)
  return unwrapApiData(data)
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

export const getRecentTradesFromWallets = async (params?: {
  limit?: number
  hours?: number
}): Promise<RecentTradesResponse> => {
  const { data } = await api.get('/wallets/recent-trades/all', { params })
  return unwrapApiData(data)
}

export const getWalletPositions = async (address: string): Promise<{ address: string; positions: WalletPosition[] }> => {
  const { data } = await api.get(`/wallets/${address}/positions`)
  return unwrapApiData(data)
}

export const getWalletTrades = async (address: string, limit = 100): Promise<{ address: string; trades: WalletTrade[] }> => {
  const { data } = await api.get(`/wallets/${address}/trades`, {
    params: { limit }
  })
  return unwrapApiData(data)
}

// ==================== MARKETS ====================

export const getMarkets = async (params?: {
  active?: boolean
  limit?: number
  offset?: number
}): Promise<Market[]> => {
  const { data } = await api.get('/markets', { params })
  return unwrapApiData(data)
}

export const getEvents = async (params?: {
  closed?: boolean
  limit?: number
  offset?: number
}): Promise<Record<string, any>[]> => {
  const { data } = await api.get('/events', { params })
  return unwrapApiData(data)
}

// ==================== SIMULATION ====================

export const createSimulationAccount = async (params: {
  name: string
  initial_capital?: number
  max_position_pct?: number
  max_positions?: number
}): Promise<{ account_id: string; name: string; initial_capital: number; message: string }> => {
  const { data } = await api.post('/simulation/accounts', params, { timeout: 20_000 })
  return unwrapApiData(data)
}

export const getSimulationAccounts = async (): Promise<SimulationAccount[]> => {
  const { data } = await api.get('/simulation/accounts')
  return unwrapApiData(data)
}

export const getSimulationAccount = async (accountId: string): Promise<SimulationAccount> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}`)
  return unwrapApiData(data)
}

export const deleteSimulationAccount = async (accountId: string): Promise<{ message: string; account_id: string }> => {
  const { data } = await api.delete(`/simulation/accounts/${accountId}`)
  return unwrapApiData(data)
}

export const getAccountPositions = async (accountId: string): Promise<SimulationPosition[]> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/positions`)
  return unwrapApiData(data)
}

export const getAccountTrades = async (accountId: string, limit = 50): Promise<SimulationTrade[]> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/trades`, { params: { limit } })
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

export const getAccountPerformance = async (accountId: string): Promise<Record<string, any>> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/performance`)
  return unwrapApiData(data)
}

export const getAccountEquityHistory = async (accountId: string): Promise<EquityHistoryResponse> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/equity-history`)
  return unwrapApiData(data)
}

// ==================== ANOMALY DETECTION ====================

export const analyzeWallet = async (address: string): Promise<WalletAnalysis> => {
  const { data } = await api.get(`/anomaly/analyze/${address}`)
  return unwrapApiData(data)
}

export const findProfitableWallets = async (params?: {
  min_trades?: number
  min_win_rate?: number
  min_pnl?: number
  max_anomaly_score?: number
}): Promise<{ count: number; wallets: WalletAnalysis[] }> => {
  const { data } = await api.post('/anomaly/find-profitable', params || {})
  return unwrapApiData(data)
}

export const getAnomalies = async (params?: {
  severity?: string
  anomaly_type?: string
  limit?: number
}): Promise<{ count: number; anomalies: Anomaly[] }> => {
  const { data } = await api.get('/anomaly/anomalies', { params })
  return unwrapApiData(data)
}

export const quickCheckWallet = async (address: string): Promise<{ wallet: string; is_suspicious: boolean; anomaly_score: number; critical_anomalies: number; win_rate: number; total_pnl: number; verdict: string; summary: string }> => {
  const { data } = await api.get(`/anomaly/check/${address}`)
  return unwrapApiData(data)
}

export const getWalletTradesAnalysis = async (address: string, limit = 100): Promise<{ wallet: string; total: number; trades: WalletTrade[] }> => {
  const { data } = await api.get(`/anomaly/wallet/${address}/trades`, { params: { limit } })
  return unwrapApiData(data)
}

export const getWalletPositionsAnalysis = async (address: string): Promise<{ wallet: string; total_positions: number; total_value: number; total_unrealized_pnl: number; positions: WalletPosition[] }> => {
  const { data } = await api.get(`/anomaly/wallet/${address}/positions`)
  return unwrapApiData(data)
}

export const getWalletSummary = async (address: string): Promise<WalletSummary> => {
  const { data } = await api.get(`/anomaly/wallet/${address}/summary`)
  return unwrapApiData(data)
}

// ==================== HEALTH ====================

export const getHealthStatus = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/health/detailed')
  return unwrapApiData(data)
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

function toDiscoveryTimePeriod(value?: TimePeriod): string | undefined {
  switch (value) {
    case 'DAY':
      return '24h'
    case 'WEEK':
      return '7d'
    case 'MONTH':
      return '30d'
    case 'ALL':
      return 'all'
    default:
      return undefined
  }
}

function toDiscoveryCategory(value?: Category): string | undefined {
  const normalized = String(value || '').trim().toLowerCase()
  if (!normalized || normalized === 'overall') return undefined
  return normalized
}

function normalizeRateValue(value: unknown): number | undefined {
  if (value == null) return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return undefined
  return parsed <= 1 ? parsed * 100 : parsed
}

function toDiscoveredTraderRows(rows: unknown[]): DiscoveredTrader[] {
  const payloadRows = Array.isArray(rows) ? rows : []
  return payloadRows
    .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === 'object')
    .map((row) => {
      const totalTrades = Number(row.total_trades ?? row.trade_count ?? 0)
      const totalReturned = Number(row.total_returned ?? row.total_invested ?? 0)
      const winRate = normalizeRateValue(row.win_rate)
      const wins = Number(row.wins ?? 0)
      const losses = Number(row.losses ?? 0)
      const rank = Number(row.rank_position ?? row.rank ?? 0)
      return {
        address: String(row.address || ''),
        username: row.username ? String(row.username) : undefined,
        trades: Number.isFinite(totalTrades) ? totalTrades : 0,
        volume: Number.isFinite(totalReturned) ? totalReturned : 0,
        pnl: Number(row.total_pnl ?? 0),
        rank: Number.isFinite(rank) && rank > 0 ? rank : undefined,
        buys: 0,
        sells: 0,
        win_rate: winRate,
        wins: Number.isFinite(wins) ? wins : undefined,
        losses: Number.isFinite(losses) ? losses : undefined,
        total_markets: Number(row.unique_markets ?? 0) || undefined,
        trade_count: Number.isFinite(totalTrades) ? totalTrades : undefined,
      }
    })
    .filter((row) => row.address.length > 0)
}

export const getLeaderboard = async (filters?: LeaderboardFilters) => {
  const params = {
    limit: filters?.limit,
    min_trades: 0,
    sort_by: filters?.order_by === 'VOL' ? 'total_returned' : 'total_pnl',
    sort_dir: 'desc',
    time_period: toDiscoveryTimePeriod(filters?.time_period),
    market_category: toDiscoveryCategory(filters?.category),
  }
  const { data } = await api.get('/discovery/leaderboard', { params })
  return unwrapApiData(data)
}

export const discoverTopTraders = async (
  limit = 50,
  minTrades = 10,
  filters?: Omit<LeaderboardFilters, 'limit'>
): Promise<DiscoveredTrader[]> => {
  const { data } = await api.get('/discovery/leaderboard', {
    params: {
      limit,
      min_trades: minTrades,
      sort_by: filters?.order_by === 'VOL' ? 'total_returned' : 'total_pnl',
      sort_dir: 'desc',
      time_period: toDiscoveryTimePeriod(filters?.time_period),
      market_category: toDiscoveryCategory(filters?.category),
    }
  })
  const rows = unwrapApiData(data)?.wallets || []
  return toDiscoveredTraderRows(rows)
}

export const discoverByWinRate = async (filters?: WinRateFilters): Promise<DiscoveredTrader[]> => {
  const limit = Math.max(1, Math.min(500, Number(filters?.scan_count || filters?.limit || 200)))
  const { data } = await api.get('/discovery/leaderboard', {
    params: {
      limit,
      min_trades: filters?.min_trades ?? 0,
      sort_by: 'win_rate',
      sort_dir: 'desc',
      time_period: toDiscoveryTimePeriod(filters?.time_period),
      market_category: toDiscoveryCategory(filters?.category),
    },
  })
  const rows = toDiscoveredTraderRows(unwrapApiData(data)?.wallets || [])
  const minWinRate = Number(filters?.min_win_rate ?? 0)
  const minVolume = Number(filters?.min_volume ?? 0)
  const maxVolume = Number(filters?.max_volume ?? 0)
  const requestedLimit = Math.max(1, Number(filters?.limit || 50))

  const filtered = rows.filter((row) => {
    const rowWinRate = Number(row.win_rate ?? 0)
    const rowVolume = Number(row.volume ?? 0)
    if (minWinRate > 0 && rowWinRate < minWinRate) return false
    if (minVolume > 0 && rowVolume < minVolume) return false
    if (maxVolume > 0 && rowVolume > maxVolume) return false
    return true
  })
  return filtered.slice(0, requestedLimit)
}

export const getWalletWinRate = async (address: string, timePeriod?: TimePeriod): Promise<WalletWinRate> => {
  const { data } = await api.get(`/discover/wallet/${address}/win-rate`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return unwrapApiData(data)
}

export const analyzeWalletPnL = async (address: string, timePeriod?: TimePeriod): Promise<WalletPnL> => {
  const { data } = await api.get(`/discover/wallet/${address}`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

// Add time-filtered wallet PnL (for future time filter support)
export const analyzeWalletPnLWithFilter = async (address: string, timePeriod?: TimePeriod): Promise<WalletPnL> => {
  const { data } = await api.get(`/discover/wallet/${address}`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return unwrapApiData(data)
}

export const analyzeAndTrackWallet = async (params: {
  address: string
  label?: string
}) => {
  const { data } = await api.post('/discover/analyze-and-track', null, { params })
  return unwrapApiData(data)
}

// ==================== TRADING ====================

export interface TradingStatus {
  initialized: boolean
  authenticated: boolean
  credentials_configured: boolean
  wallet_address: string | null
  execution_wallet_address?: string | null
  eoa_wallet_address?: string | null
  proxy_funder_wallet?: string | null
  auth_error?: string | null
  native_gas?: {
    wallet_address: string | null
    affordable_for_approval: boolean
    balance_wei: number
    balance_native: number
    gas_price_wei: number
    required_wei_for_approval: number
    required_native_for_approval: number
    error?: string | null
  } | null
  execution_paths?: {
    normal_trading: string
    direct_ctf_actions: string
  } | null
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
    min_order_size_usd: number
    max_slippage_percent: number
  }
}

export interface TradingVpnStatus {
  proxy_enabled: boolean
  proxy_url?: string | null
  proxy_reachable: boolean
  direct_ip?: string | null
  proxy_ip?: string | null
  vpn_active: boolean
  error?: string
  direct_ip_error?: string
  proxy_ip_error?: string
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

export interface CancelAllOrdersResult {
  status: 'success' | 'partial_failure' | 'failed'
  requested_count: number
  cancelled_count: number
  failed_count: number
  failed_order_ids: string[]
  message: string
}

export interface EmergencyStopTradingResult {
  status: string
  cancelled_orders: number
  message: string
  cancel_result: CancelAllOrdersResult
}

export const getTradingStatus = async (): Promise<TradingStatus> => {
  const { data } = await api.get('/trader-orchestrator/live/status')
  return unwrapApiData(data)
}

export const getTradingVpnStatus = async (): Promise<TradingVpnStatus> => {
  const { data } = await api.get('/trader-orchestrator/live/vpn-status')
  return unwrapApiData(data)
}

export const initializeTrading = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/trader-orchestrator/live/initialize')
  return unwrapApiData(data)
}

export const placeOrder = async (params: {
  token_id: string
  side: string
  price: number
  size: number
  order_type?: string
  market_question?: string
}): Promise<Order> => {
  const { data } = await api.post('/trader-orchestrator/live/orders', params)
  return unwrapApiData(data)
}

export const getOrders = async (limit = 100, status?: string): Promise<Order[]> => {
  const { data } = await api.get('/trader-orchestrator/live/orders', { params: { limit, status } })
  return unwrapApiData(data)
}

export const getOpenOrders = async (): Promise<Order[]> => {
  const { data } = await api.get('/trader-orchestrator/live/orders/open')
  return unwrapApiData(data)
}

export const cancelOrder = async (orderId: string): Promise<{ status: string; order_id: string }> => {
  const { data } = await api.delete(`/trader-orchestrator/live/orders/${orderId}`)
  return unwrapApiData(data)
}

export const cancelAllOrders = async (): Promise<CancelAllOrdersResult> => {
  const { data } = await api.delete('/trader-orchestrator/live/orders')
  return unwrapApiData(data)
}

export const getTradingPositions = async (): Promise<TradingPosition[]> => {
  const { data } = await api.get('/trader-orchestrator/live/positions')
  return unwrapApiData(data)
}

export const getTradingBalance = async (): Promise<{ balance: number; available: number; reserved: number; currency: string; timestamp: string }> => {
  const { data } = await api.get('/trader-orchestrator/live/balance')
  return unwrapApiData(data)
}

export interface LiveWalletFill {
  id: string
  side: 'buy' | 'sell' | string
  condition_id: string
  token_id: string
  market_title: string
  market_slug: string | null
  event_slug: string | null
  category: string | null
  outcome: string | null
  size: number
  price: number
  notional: number
  timestamp: string
  transaction_hash: string | null
}

export interface LiveWalletRoundTrip {
  id: string
  condition_id: string
  token_id: string
  market_title: string
  market_slug: string | null
  event_slug: string | null
  category: string | null
  outcome: string | null
  quantity: number
  avg_buy_price: number
  avg_sell_price: number
  buy_notional: number
  sell_notional: number
  realized_pnl: number
  roi_percent: number
  opened_at: string
  closed_at: string
  hold_minutes: number
  lots_matched: number
}

export interface LiveWalletOpenLot {
  token_id: string
  condition_id: string
  market_title: string
  market_slug: string | null
  event_slug: string | null
  category: string | null
  outcome: string | null
  remaining_size: number
  avg_cost: number
  cost_basis: number
  opened_at: string
}

export interface LiveWalletPerformance {
  wallet_address: string
  generated_at: string
  fills: LiveWalletFill[]
  round_trips: LiveWalletRoundTrip[]
  open_lots: LiveWalletOpenLot[]
  summary: {
    total_fills: number
    buy_fills: number
    sell_fills: number
    total_round_trips: number
    winning_round_trips: number
    losing_round_trips: number
    win_rate_percent: number
    gross_profit: number
    gross_loss: number
    net_realized_pnl: number
    total_buy_notional: number
    total_sell_notional: number
    avg_roi_percent: number
    avg_hold_minutes: number
    profit_factor: number
    unmatched_sell_size: number
    open_inventory_notional: number
    open_inventory_size: number
    open_lot_count: number
  }
}

export const getLiveWalletPerformance = async (limit = 1000): Promise<LiveWalletPerformance> => {
  const { data } = await api.get('/trader-orchestrator/live/performance', { params: { limit } })
  return unwrapApiData(data)
}

export const executeOpportunityLive = async (params: {
  opportunity_id: string
  positions: any[]
  size_usd: number
}): Promise<{ status: string; orders: Order[]; message?: string }> => {
  const { data } = await api.post('/trader-orchestrator/live/execute-opportunity', params)
  return unwrapApiData(data)
}

export const emergencyStopTrading = async (): Promise<EmergencyStopTradingResult> => {
  const { data } = await api.post('/trader-orchestrator/live/emergency-stop')
  return unwrapApiData(data)
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
  global_runtime: {
    pending_live_exit_guard: {
      max_pending_exits: number
      identity_guard_enabled: boolean
      terminal_statuses: string[]
    }
    live_risk_clamps: {
      enforce_allow_averaging_off: boolean
      min_cooldown_seconds: number
      max_consecutive_losses_cap: number
      max_open_orders_cap: number
      max_open_positions_cap: number
      max_trade_notional_usd_cap: number
      max_orders_per_cycle_cap: number
      enforce_halt_on_consecutive_losses: boolean
    }
    live_market_context: {
      enabled: boolean
      history_window_seconds: number
      history_fidelity_seconds: number
      max_history_points: number
      timeout_seconds: number
    }
    live_provider_health: {
      window_seconds: number
      min_errors: number
      block_seconds: number
    }
    trader_cycle_timeout_seconds: number | null
  }
}

export interface TraderOrchestratorSettingsUpdatePayload {
  run_interval_seconds?: number
  global_risk?: TraderOrchestratorConfig['global_risk']
  global_runtime?: TraderOrchestratorConfig['global_runtime']
  requested_by?: string
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
  check_name?: string
  check_label: string
  passed: boolean
  score: number | null
  detail: string | null
  message?: string | null
  payload: Record<string, any>
  created_at: string | null
}

export interface TraderDecisionFailedCheck {
  check_key: string
  check_label: string
  detail: string | null
  score: number | null
  payload: Record<string, any>
  created_at: string | null
}

export interface ExecutionLatencyPercentiles {
  p50: number | null
  p95: number | null
  p99: number | null
}

export interface ExecutionLatencyBucket {
  count: number
  armed_to_ws_release_ms: ExecutionLatencyPercentiles
  emit_to_queue_wake_ms: ExecutionLatencyPercentiles
  ws_release_to_decision_ms: ExecutionLatencyPercentiles
  ws_release_to_submit_start_ms: ExecutionLatencyPercentiles
  wake_to_context_ready_ms: ExecutionLatencyPercentiles
  context_ready_to_decision_ms: ExecutionLatencyPercentiles
  decision_to_submit_start_ms: ExecutionLatencyPercentiles
  submit_start_to_provider_ack_ms: ExecutionLatencyPercentiles
  emit_to_submit_start_ms: ExecutionLatencyPercentiles
}

export interface ExecutionLatencySummary {
  internal_sla_definition: string
  internal_sla_target_ms: number
  rolling_window_seconds: number
  sample_count: number
  overall: ExecutionLatencyBucket
  by_source: Record<string, ExecutionLatencyBucket>
  by_strategy: Record<string, ExecutionLatencyBucket>
  by_trader: Record<string, ExecutionLatencyBucket>
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
  reconciliation_worker?: {
    running: boolean
    enabled: boolean
    current_activity: string | null
    interval_seconds: number
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
    execution_latency?: ExecutionLatencySummary
  }
  traders?: Trader[]
}

export interface TraderOrchestratorControlResponse {
  status: string
  control?: Partial<TraderOrchestratorOverview['control']>
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
  const workerRunning = Boolean(worker.running)
  const workerActivity = worker.current_activity || null
  const workerLastRunAt = worker.last_run_at || null
  const workerLastError = worker.last_error || null
  const workerUpdatedAt = worker.updated_at || null
  const totalTrades = Number(metrics.orders_count || 0)

  return {
    mode: control.mode || 'shadow',
    running: tradingActive && workerRunning,
    trading_active: tradingActive,
    worker_running: workerRunning,
    control: {
      is_enabled: Boolean(control.is_enabled),
      is_paused: Boolean(control.is_paused),
      kill_switch: Boolean(control.kill_switch),
      requested_run_at: control.requested_run_at || null,
      run_interval_seconds: Number(control.run_interval_seconds || 2),
      updated_at: control.updated_at || null,
    },
    snapshot: {
      running: workerRunning,
      enabled: Boolean(worker.enabled),
      current_activity: workerActivity,
      interval_seconds: Number(worker.interval_seconds || 2),
      last_run_at: workerLastRunAt,
      last_error: workerLastError,
      updated_at: workerUpdatedAt,
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
      last_trade_at: workerLastRunAt,
      opportunities_seen: Number(metrics.decisions_count || 0),
      opportunities_executed: Number(metrics.orders_count || 0),
      opportunities_skipped: Math.max(0, Number(metrics.decisions_count || 0) - Number(metrics.orders_count || 0)),
    },
  }
}

export const getTraderOrchestratorOverview = async (): Promise<TraderOrchestratorOverview> => {
  const { data } = await api.get('/trader-orchestrator/overview')
  return unwrapApiData(data)
}

export const getTraderOrchestratorStatus = async (): Promise<TraderOrchestratorStatus> => {
  const overview = await getTraderOrchestratorOverview()
  return mapOverviewToStatus(overview)
}

export const startTraderOrchestrator = async (payload?: {
  mode?: string
  requested_by?: string
}): Promise<TraderOrchestratorControlResponse> => {
  const { data } = await api.post('/trader-orchestrator/start', {
    mode: payload?.mode || 'shadow',
    requested_by: payload?.requested_by,
  })
  return unwrapApiData(data)
}

export const stopTraderOrchestrator = async (): Promise<TraderOrchestratorControlResponse> => {
  const { data } = await api.post('/trader-orchestrator/stop')
  return unwrapApiData(data)
}

export const runTraderOrchestratorLivePreflight = async (payload?: {
  mode?: string
  requested_by?: string
}): Promise<TraderOrchestratorLivePreflightResponse> => {
  const { data } = await api.post('/trader-orchestrator/live/preflight', {
    mode: payload?.mode || 'live',
    requested_by: payload?.requested_by,
  })
  return unwrapApiData(data)
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
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

export const stopTraderOrchestratorLive = async (requestedBy?: string) => {
  const { data } = await api.post('/trader-orchestrator/live/stop', {
    requested_by: requestedBy,
  })
  return unwrapApiData(data)
}

export const setTraderOrchestratorLiveKillSwitch = async (enabled: boolean, requestedBy?: string) => {
  const { data } = await api.post('/trader-orchestrator/kill-switch', {
    enabled,
    requested_by: requestedBy,
  })
  return unwrapApiData(data)
}

export const updateTraderOrchestratorSettings = async (payload: TraderOrchestratorSettingsUpdatePayload) => {
  const { data } = await api.put('/trader-orchestrator/settings', payload)
  return unwrapApiData(data)
}

export type TraderSourceKey = 'scanner' | 'crypto' | 'news' | 'weather' | 'traders' | 'manual'

export interface TraderSourceConfig {
  source_key: TraderSourceKey | string
  strategy_key: string
  strategy_version?: number | null
  strategy_params: Record<string, any>
}

export interface Trader {
  id: string
  name: string
  description?: string | null
  mode: 'shadow' | 'live'
  source_configs: TraderSourceConfig[]
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
  source_configs: TraderSourceConfig[]
  interval_seconds: number
  risk_limits: Record<string, any>
}

export interface TraderDecision {
  id: string
  trader_id: string
  signal_id: string | null
  source: string
  strategy_key: string
  strategy_version?: number | null
  decision: string
  reason: string | null
  score: number | null
  market_id: string | null
  market_question: string | null
  direction: string | null
  direction_side?: string | null
  direction_label?: string | null
  yes_label?: string | null
  no_label?: string | null
  market_price: number | null
  model_probability: number | null
  edge_percent: number | null
  confidence: number | null
  signal_score: number | null
  signal_payload?: Record<string, any>
  signal_strategy_context?: Record<string, any>
  event_id: string | null
  trace_id: string | null
  checks_summary: Record<string, any>
  risk_snapshot: Record<string, any>
  payload: Record<string, any>
  failed_checks?: TraderDecisionFailedCheck[]
  failed_check_count?: number
  created_at: string | null
}

export interface TraderOrder {
  id: string
  trader_id: string
  signal_id: string | null
  decision_id: string | null
  source: string
  strategy_key?: string | null
  strategy_version?: number | null
  market_id: string
  market_question: string | null
  direction: string | null
  direction_side?: string | null
  direction_label?: string | null
  yes_label?: string | null
  no_label?: string | null
  mode: string
  status: string
  notional_usd: number | null
  entry_price: number | null
  effective_price: number | null
  provider_order_id: string
  provider_clob_order_id: string
  provider_snapshot_status: string
  filled_shares: number | null
  filled_notional_usd: number | null
  average_fill_price: number | null
  current_price: number | null
  mark_source: string
  mark_updated_at: string
  unrealized_pnl: number | null
  edge_percent: number | null
  confidence: number | null
  actual_profit: number | null
  reason: string | null
  close_trigger?: string | null
  close_reason?: string | null
  payload: Record<string, any>
  copy_attribution?: {
    source_wallet?: string
    source_tx_hash?: string
    source_order_hash?: string
    side?: string
    source_price?: number
    follower_effective_price?: number
    source_size?: number
    source_notional_usd?: number
    copy_latency_ms?: number
    source_detection_latency_ms?: number
    slippage_bps?: number
    adverse_slippage_bps?: number
    detected_at?: string
  } | null
  error_message: string | null
  event_id: string | null
  trace_id: string | null
  created_at: string | null
  executed_at: string | null
  updated_at: string | null
}

export interface TraderLiveWalletPosition {
  token_id: string
  market_id: string
  condition_id: string | null
  market_question: string | null
  outcome: string | null
  direction: string | null
  size: number
  avg_price: number | null
  current_price: number | null
  initial_value: number | null
  current_value: number | null
  unrealized_pnl: number | null
  yes_token_id: string | null
  no_token_id: string | null
  yes_price: number | null
  no_price: number | null
  market_slug: string | null
  event_slug: string | null
  market_url: string | null
  is_managed: boolean
  managed_order_id: string | null
}

export interface TraderLiveWalletPositionsPayload {
  trader_id: string
  wallet_address: string | null
  positions: TraderLiveWalletPosition[]
  managed_token_ids: string[]
  managed_order_ids: string[]
  summary: {
    total_positions: number
    managed_positions: number
    unmanaged_positions: number
    returned_positions: number
  }
}

export interface TraderAdoptLiveWalletPositionPayload {
  token_id: string
  reason?: string
  requested_by?: string
}

export interface TraderAdoptLiveWalletPositionResult {
  status: string
  trader_id: string
  wallet_address: string
  strategy_key?: string
  token_id: string
  market_id: string
  direction: string
  order: TraderOrder
  position_inventory: {
    trader_id: string
    mode: string
    open_positions: number
    inserts: number
    updates: number
    closures: number
  }
}

export interface TraderMarketHistoryPoint {
  t: number
  yes: number
  no: number
  idx_0: number
  idx_1: number
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

export interface TraderCopyAnalyticsLeader {
  source_wallet: string
  orders: number
  active_orders: number
  realized_orders: number
  realized_pnl_usd: number
  gross_notional_usd: number
  open_exposure_usd: number
  avg_slippage_bps: number | null
  avg_adverse_slippage_bps: number | null
  avg_copy_latency_ms: number | null
}

export interface TraderCopyAnalytics {
  trader_id: string
  mode: string
  as_of: string | null
  sample_size: number
  scanned_orders: number
  summary: {
    total_orders: number
    resolved_orders: number
    win_orders: number
    loss_orders: number
    win_rate_pct: number | null
    realized_pnl_usd: number
    gross_notional_usd: number
    open_exposure_usd: number
    avg_slippage_bps: number | null
    avg_adverse_slippage_bps: number | null
    avg_copy_latency_ms: number | null
    distinct_leaders: number
  }
  leaders: TraderCopyAnalyticsLeader[]
}

export interface TraderSource {
  key: string
  label: string
  description: string
  domains: string[]
  signal_types: string[]
  aliases?: string[]
  default_strategy_key?: string
  strategy_options?: TraderSourceStrategyOption[]
  default_config?: TraderSourceConfig
}

export interface TraderSourceStrategyOption {
  key: string
  label: string
  description: string
  default_params: Record<string, any>
  param_fields: Array<Record<string, any>>
  version?: number
  latest_version?: number
  versions?: number[]
}

export interface TraderConfigSchema {
  version: string
  sources: TraderSource[]
  default_strategy_key?: string
  strategies?: Array<Record<string, any>>
  shared_risk_fields: Array<Record<string, any>>
  shared_risk_defaults?: Record<string, any>
  shared_exit_fields?: Array<Record<string, any>>
  runtime_fields: Array<Record<string, any>>
  default_runtime_metadata?: Record<string, any>
}

export interface TraderStrategyDefinition {
  id: string
  strategy_key: string
  slug?: string
  source_key: string
  label: string
  name?: string
  description: string | null
  class_name: string
  source_code: string
  default_params_json: Record<string, any>
  config?: Record<string, any>
  param_schema_json: Record<string, any>
  config_schema?: Record<string, any>
  aliases_json: string[]
  is_system: boolean
  enabled: boolean
  status: string
  error_message: string | null
  version: number
  created_at: string | null
  updated_at: string | null
  runtime?: Record<string, any> | null
}

export interface TraderStrategyValidationResult {
  valid: boolean
  class_name: string | null
  errors: string[]
  warnings: string[]
}

const DEFAULT_STRATEGY_BY_SOURCE: Record<string, string> = {
  scanner: 'basic',
  crypto: 'btc_eth_highfreq',
  manual: 'manual_wallet_position',
  news: 'news_edge',
  weather: 'weather_distribution',
  traders: 'traders_confluence',
}

const DEFAULT_STRATEGY_KEY = 'btc_eth_highfreq'

function normalizeTraderSourceKey(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

function normalizeTraderStrategyKey(value: unknown): string {
  const key = String(value || '').trim().toLowerCase()
  return key || DEFAULT_STRATEGY_KEY
}

function normalizeStrategyVersionValue(value: unknown): number | null {
  if (value === null || value === undefined) return null
  const raw = String(value || '').trim().toLowerCase()
  if (!raw || raw === 'latest') return null
  const parsed = Number(raw.startsWith('v') ? raw.slice(1) : raw)
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return Math.trunc(parsed)
}

function normalizeTraderMode(value: unknown): 'shadow' | 'live' {
  const mode = String(value || '').trim().toLowerCase()
  if (mode === 'live') return 'live'
  return 'shadow'
}

function normalizeTraderStrategyKeyForSource(sourceKey: string, value: unknown): string {
  const key = normalizeTraderStrategyKey(value)
  const normalizedSource = normalizeTraderSourceKey(sourceKey)
  if (key) {
    return key
  }
  return DEFAULT_STRATEGY_BY_SOURCE[normalizedSource] || DEFAULT_STRATEGY_KEY
}

function normalizeTraderFields(raw: any): Trader {
  const sourceConfigs = Array.isArray(raw?.source_configs) ? raw.source_configs : []
  const normalizedSourceConfigs = sourceConfigs
    .filter((item: any) => item && typeof item === 'object')
    .map((item: any) => {
      const sourceKey = normalizeTraderSourceKey(item.source_key)
      return {
        source_key: sourceKey,
        strategy_key: normalizeTraderStrategyKeyForSource(sourceKey, item.strategy_key),
        strategy_version: normalizeStrategyVersionValue(item.strategy_version),
        strategy_params: item.strategy_params && typeof item.strategy_params === 'object' ? item.strategy_params : {},
      }
    })

  return {
    ...raw,
    mode: normalizeTraderMode(raw?.mode),
    source_configs: normalizedSourceConfigs,
  }
}

export const getTraders = async (params?: { mode?: 'shadow' | 'live' }): Promise<Trader[]> => {
  const { data } = await api.get('/traders', { params })
  return (data.traders || []).map(normalizeTraderFields)
}

export const createTrader = async (payload: Record<string, any>): Promise<Trader> => {
  const { data } = await api.post('/traders', payload)
  return normalizeTraderFields(data)
}

export const getTrader = async (traderId: string): Promise<Trader> => {
  const { data } = await api.get(`/traders/${traderId}`)
  return normalizeTraderFields(data)
}

export const updateTrader = async (traderId: string, payload: Record<string, any>): Promise<Trader> => {
  const { data } = await api.put(`/traders/${traderId}`, payload)
  return normalizeTraderFields(data)
}

export interface TraderStartPayload {
  copy_existing_positions?: boolean | null
  requested_by?: string
}

export type TraderStopLifecycleMode = 'keep_positions' | 'close_shadow_positions' | 'close_all_positions'

export interface TraderStopPayload {
  stop_lifecycle?: TraderStopLifecycleMode
  confirm_live?: boolean
  requested_by?: string
  reason?: string
}

export interface TraderManualSellOrderResponse {
  status: string
  trader_id: string
  order_id: string
  mode: string
  result: {
    order_id: string
    mode: string
    matched: number
    closed: number
    state_updates: number
    would_close: number
    details: Array<Record<string, any>>
  }
}

export const startTrader = async (traderId: string, payload?: TraderStartPayload): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/start`, payload || {})
  return normalizeTraderFields(data)
}

export const stopTrader = async (traderId: string, payload?: TraderStopPayload): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/stop`, payload || {})
  return normalizeTraderFields(data)
}

export const sellTraderOrderNow = async (
  traderId: string,
  orderId: string,
  payload?: { requested_by?: string; reason?: string }
): Promise<TraderManualSellOrderResponse> => {
  const { data } = await api.post(`/traders/${traderId}/orders/${orderId}/sell`, payload || {})
  return unwrapApiData(data)
}

export const activateTrader = async (traderId: string, payload?: { requested_by?: string; reason?: string }): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/activate`, payload || {})
  return normalizeTraderFields(data)
}

export const deactivateTrader = async (traderId: string, payload?: { requested_by?: string; reason?: string }): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/deactivate`, payload || {})
  return normalizeTraderFields(data)
}

export type TraderDeleteAction = 'block' | 'disable' | 'force_delete'

export const deleteTrader = async (
  traderId: string,
  options?: { action?: TraderDeleteAction }
): Promise<{
  status: string
  trader_id: string
  action?: TraderDeleteAction
  open_live_positions?: number
  open_shadow_positions?: number
  open_other_positions?: number
  open_live_orders?: number
  open_shadow_orders?: number
  open_other_orders?: number
  open_total_positions?: number
  open_total_orders?: number
  message?: string
}> => {
  const { data } = await api.delete(`/traders/${traderId}`, { params: options })
  return unwrapApiData(data)
}

export const runTraderOnce = async (traderId: string): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/run-once`)
  return unwrapApiData(data)
}

export const getTraderDecisions = async (
  traderId: string,
  params?: { decision?: string; limit?: number }
): Promise<TraderDecision[]> => {
  const { data } = await api.get(`/traders/${traderId}/decisions`, { params })
  const rows = Array.isArray(data.decisions) ? data.decisions : []
  return rows.map((row: any) => ({
    ...row,
    failed_checks: Array.isArray(row?.failed_checks) ? row.failed_checks : [],
    failed_check_count: Number(row?.failed_check_count || 0),
  }))
}

export const getAllTraderDecisions = async (
  traderIds: string[],
  params?: { decision?: string; limit?: number; per_trader_limit?: number }
): Promise<TraderDecision[]> => {
  const normalizedTraderIds = Array.from(
    new Set(
      traderIds
        .map((value) => String(value || '').trim())
        .filter(Boolean)
    )
  )
  if (normalizedTraderIds.length === 0) return []
  const { data } = await api.get('/traders/decisions/all', {
    params: {
      decision: params?.decision,
      trader_ids: normalizedTraderIds.join(','),
      limit: Math.max(1, Math.trunc(Number(params?.limit) || 2000)),
      per_trader_limit: Math.max(1, Math.trunc(Number(params?.per_trader_limit) || 160)),
    },
  })
  const rows = Array.isArray(data.decisions) ? data.decisions : []
  return rows.map((row: any) => ({
    ...row,
    failed_checks: Array.isArray(row?.failed_checks) ? row.failed_checks : [],
    failed_check_count: Number(row?.failed_check_count || 0),
  }))
}

export const getTraderOrders = async (
  traderId: string,
  params?: { status?: string; limit?: number }
): Promise<TraderOrder[]> => {
  const { data } = await api.get(`/traders/${traderId}/orders`, { params })
  return data.orders || []
}

export const getTraderCopyAnalytics = async (
  traderId: string,
  params?: { mode?: 'shadow' | 'live'; limit?: number; leader_limit?: number }
): Promise<TraderCopyAnalytics> => {
  const { data } = await api.get(`/traders/${traderId}/copy-analytics`, { params })
  return unwrapApiData(data)
}

export const getTraderLiveWalletPositions = async (
  traderId: string,
  params?: { include_managed?: boolean },
): Promise<TraderLiveWalletPositionsPayload> => {
  const { data } = await api.get(`/traders/${traderId}/positions/live-wallet`, { params })
  return unwrapApiData(data)
}

export const adoptTraderLiveWalletPosition = async (
  traderId: string,
  payload: TraderAdoptLiveWalletPositionPayload,
): Promise<TraderAdoptLiveWalletPositionResult> => {
  const { data } = await api.post(`/traders/${traderId}/positions/live-wallet/adopt`, payload)
  return unwrapApiData(data)
}

export const getAllTraderOrders = async (limit = 2000): Promise<TraderOrder[]> => {
  const { data } = await api.get('/traders/orders/all', {
    params: { limit: Math.max(1, Math.trunc(Number(limit) || 2000)) },
  })
  return data.orders || []
}

export const getTraderMarketHistory = async (
  marketIds: string[],
  limit = 120
): Promise<Record<string, TraderMarketHistoryPoint[]>> => {
  const normalizedIds = Array.from(
    new Set(
      marketIds
        .map((value) => String(value || '').trim().toLowerCase())
        .filter(Boolean)
    )
  )
  if (normalizedIds.length === 0) return {}
  const { data } = await api.get('/traders/market-history', {
    params: {
      market_ids: normalizedIds.join(','),
      limit: Math.max(2, Math.min(5000, Math.trunc(Number(limit) || 120))),
    },
  })
  const payload = unwrapApiData(data)
  return (payload?.histories && typeof payload.histories === 'object')
    ? payload.histories as Record<string, TraderMarketHistoryPoint[]>
    : {}
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
  const normalized = unwrapApiData(data)
  const checks = Array.isArray(normalized.checks) ? normalized.checks : []
  normalized.checks = checks.map((check: any) => {
    const label = String(check?.check_label || check?.check_name || check?.label || check?.check_key || 'Check')
    const detail = check?.detail ?? check?.message ?? null
    return {
      ...check,
      check_name: String(check?.check_name || label),
      check_label: label,
      detail,
      message: detail,
      payload: check && typeof check.payload === 'object' && check.payload !== null ? check.payload : {},
    }
  })
  return normalized
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
  return unwrapApiData(data)
}

export const getTraderSources = async (): Promise<TraderSource[]> => {
  const { data } = await api.get('/trader-sources')
  return data.sources || []
}

export const getTraderConfigSchema = async (): Promise<TraderConfigSchema> => {
  const { data } = await api.get('/trader-sources/schema')
  return unwrapApiData(data)
}

// Trader strategy functions now proxy to the unified /strategy-manager API.

export const getTraderStrategies = async (params?: {
  source_key?: string
  enabled?: boolean
  status?: string
}): Promise<TraderStrategyDefinition[]> => {
  const { data } = await api.get('/strategy-manager', { params })
  const items = getStrategyManagerItems(data)
  return items.map((s: any) => ({
    ...s,
    strategy_key: s.strategy_key || s.slug,
    label: s.label || s.name,
    default_params_json: s.default_params_json || s.config || {},
    param_schema_json: s.param_schema_json || s.config_schema || {},
    aliases_json: [],
  }))
}

export const getTraderStrategy = async (id: string): Promise<TraderStrategyDefinition> => {
  const { data } = await api.get(`/strategy-manager/${id}`)
  const strategy = unwrapStrategyManagerPayload(data) || {}
  return {
    ...strategy,
    strategy_key: strategy.strategy_key || strategy.slug,
    label: strategy.label || strategy.name,
    default_params_json: strategy.default_params_json || strategy.config || {},
    param_schema_json: strategy.param_schema_json || strategy.config_schema || {},
    aliases_json: [],
  }
}

export const createTraderStrategy = async (payload: {
  strategy_key: string
  source_key: string
  label: string
  description?: string | null
  source_code: string
  default_params_json?: Record<string, any>
  param_schema_json?: Record<string, any>
  enabled?: boolean
}): Promise<TraderStrategyDefinition> => {
  const { data } = await api.post('/strategy-manager', {
    slug: payload.strategy_key,
    source_key: payload.source_key,
    name: payload.label,
    description: payload.description,
    source_code: payload.source_code,
    config: payload.default_params_json || {},
    config_schema: payload.param_schema_json || {},
    enabled: payload.enabled ?? true,
  })
  return unwrapApiData(data)
}

export const updateTraderStrategy = async (
  id: string,
  payload: Partial<{
    strategy_key: string
    source_key: string
    label: string
    description: string | null
    source_code: string
    default_params_json: Record<string, any>
    param_schema_json: Record<string, any>
    enabled: boolean
    unlock_system: boolean
  }>
): Promise<TraderStrategyDefinition> => {
  const unified: Record<string, any> = {}
  if (payload.strategy_key !== undefined) unified.slug = payload.strategy_key
  if (payload.source_key !== undefined) unified.source_key = payload.source_key
  if (payload.label !== undefined) unified.name = payload.label
  if (payload.description !== undefined) unified.description = payload.description
  if (payload.source_code !== undefined) unified.source_code = payload.source_code
  if (payload.default_params_json !== undefined) unified.config = payload.default_params_json
  if (payload.param_schema_json !== undefined) unified.config_schema = payload.param_schema_json
  if (payload.enabled !== undefined) unified.enabled = payload.enabled
  if (payload.unlock_system !== undefined) unified.unlock_system = payload.unlock_system
  const { data } = await api.put(`/strategy-manager/${id}`, unified)
  return unwrapApiData(data)
}

export const validateTraderStrategy = async (
  _id: string,
  payload?: {
    source_code?: string
  }
): Promise<TraderStrategyValidationResult> => {
  const { data } = await api.post('/strategy-manager/validate', {
    source_code: payload?.source_code || '',
  })
  return {
    valid: Boolean(data.valid),
    class_name: data.class_name || null,
    errors: Array.isArray(data.errors) ? data.errors : [],
    warnings: Array.isArray(data.warnings) ? data.warnings : [],
  }
}

export const reloadTraderStrategy = async (id: string): Promise<{
  status: string
  reload: Record<string, any>
  strategy: TraderStrategyDefinition
}> => {
  const { data } = await api.post(`/strategy-manager/${id}/reload`)
  return unwrapApiData(data)
}

export const cloneTraderStrategy = async (
  id: string,
  payload?: { strategy_key?: string; label?: string; enabled?: boolean }
): Promise<TraderStrategyDefinition> => {
  // Clone: create a new strategy based on the existing one
  const original = await getTraderStrategy(id)
  const newSlug = payload?.strategy_key || `${original.strategy_key || original.slug}_clone`
  const newLabel = payload?.label || `${original.label || original.name} (Clone)`
  return createTraderStrategy({
    strategy_key: newSlug,
    source_key: original.source_key,
    label: newLabel,
    description: original.description,
    source_code: original.source_code,
    default_params_json: original.default_params_json || original.config || {},
    param_schema_json: original.param_schema_json || original.config_schema || {},
    enabled: payload?.enabled ?? false,
  })
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
  return unwrapApiData(data)
}

export const getSignalStats = async (): Promise<{
  totals: Record<string, number>
  sources: Array<Record<string, any>>
}> => {
  const { data } = await api.get('/signals/stats')
  return unwrapApiData(data)
}

export const getWorkersStatus = async (): Promise<{ workers: WorkerStatus[] }> => {
  const { data } = await api.get('/workers/status')
  return unwrapApiData(data)
}

export const pauseAllWorkers = async (): Promise<{ status: string; workers: WorkerStatus[] }> => {
  const { data } = await api.post('/workers/pause-all')
  return unwrapApiData(data)
}

export const resumeAllWorkers = async (): Promise<{ status: string; workers: WorkerStatus[] }> => {
  const { data } = await api.post('/workers/resume-all')
  return unwrapApiData(data)
}

export const startWorker = async (worker: string) => {
  const { data } = await api.post(`/workers/${worker}/start`)
  return unwrapApiData(data)
}

export const pauseWorker = async (worker: string) => {
  const { data } = await api.post(`/workers/${worker}/pause`)
  return unwrapApiData(data)
}

export const runWorkerOnce = async (worker: string) => {
  const { data } = await api.post(`/workers/${worker}/run-once`)
  return unwrapApiData(data)
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

export interface TableBloatEntry {
  table: string
  live_tuples: number
  dead_tuples: number
  dead_pct: number
  total_bytes: number
  table_bytes: number
  index_bytes: number
  last_vacuum: string | null
  last_autovacuum: string | null
  last_analyze: string | null
  last_autoanalyze: string | null
}

export interface DatabaseMaintenanceStats {
  db_size_bytes: number | null
  total_rows: number | null
  estimated_total_rows?: number | null
  simulation_trades: {
    total: number
    by_status: Record<string, number>
  }
  simulation_positions: {
    total: number
    open: number
  }
  wallet_trades: number
  wallet_activity_rollups: number
  trade_signal_emissions: number
  opportunity_history: number
  anomalies: {
    total: number
    resolved: number
  }
  date_range: {
    oldest_trade: string | null
    newest_trade: string | null
  }
  table_bloat?: TableBloatEntry[] | null
}

export const flushDatabaseData = async (target: DatabaseFlushTarget): Promise<DatabaseFlushResponse> => {
  const { data } = await api.post('/maintenance/flush', {
    target,
    confirm: true,
  })
  return unwrapApiData(data)
}

export const getDatabaseMaintenanceStats = async (): Promise<DatabaseMaintenanceStats> => {
  const { data } = await api.get('/maintenance/stats')
  const payload = unwrapApiData(data)
  return payload.stats as DatabaseMaintenanceStats
}

export const runVacuumAnalyze = async (full: boolean = false): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/maintenance/vacuum', null, { params: { full } })
  return unwrapApiData(data)
}

export const runReindexTables = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/maintenance/reindex')
  return unwrapApiData(data)
}

export const setWorkerInterval = async (worker: string, intervalSeconds: number) => {
  const { data } = await api.post(`/workers/${worker}/interval`, null, {
    params: { interval_seconds: intervalSeconds },
  })
  return unwrapApiData(data)
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
  notify_autotrader_closes: boolean
  notify_autotrader_issues: boolean
  notify_autotrader_timeline: boolean
  notify_autotrader_summary_interval_minutes: number
  notify_autotrader_summary_per_trader: boolean
}

export interface ScannerSettings {
  scan_interval_seconds: number
  min_profit_threshold: number
  max_markets_to_scan: number
  max_events_to_scan: number
  market_fetch_page_size: number
  market_fetch_order: string
  min_liquidity: number
  max_opportunities_total: number
  max_opportunities_per_strategy: number
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
  trader_opps_source_filter: 'all' | 'tracked' | 'pool'
  trader_opps_min_tier: 'WATCH' | 'HIGH' | 'EXTREME'
  trader_opps_side_filter: 'all' | 'buy' | 'sell'
  trader_opps_confluence_limit: number
  trader_opps_insider_limit: number
  trader_opps_insider_min_confidence: number
  trader_opps_insider_max_age_minutes: number
  pool_recompute_mode: 'quality_only' | 'balanced'
  pool_target_size: number
  pool_min_size: number
  pool_max_size: number
  pool_active_window_hours: number
  pool_inactive_rising_retention_hours: number
  pool_selection_score_floor: number
  pool_max_hourly_replacement_rate: number
  pool_replacement_score_cutoff: number
  pool_max_cluster_share: number
  pool_high_conviction_threshold: number
  pool_insider_priority_threshold: number
  pool_min_eligible_trades: number
  pool_max_eligible_anomaly: number
  pool_core_min_win_rate: number
  pool_core_min_sharpe: number
  pool_core_min_profit_factor: number
  pool_rising_min_win_rate: number
  pool_slo_min_analyzed_pct: number
  pool_slo_min_profitable_pct: number
  pool_leaderboard_wallet_trade_sample: number
  pool_incremental_wallet_trade_sample: number
  pool_full_sweep_interval_seconds: number
  pool_incremental_refresh_interval_seconds: number
  pool_activity_reconciliation_interval_seconds: number
  pool_recompute_interval_seconds: number
}

export interface LiveExecutionSettingsConfig {
  max_trade_size_usd: number
  max_daily_trade_volume: number
  max_slippage_percent: number
  min_account_balance_usd: number
}

export interface MaintenanceSettings {
  auto_cleanup_enabled: boolean
  cleanup_interval_hours: number
  cleanup_resolved_trade_days: number
  cleanup_trade_signal_emission_days: number
  cleanup_trade_signal_update_days: number
  cleanup_wallet_activity_rollup_days: number
  cleanup_wallet_activity_dedupe_enabled: boolean
  llm_usage_retention_days: number
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

export interface UILockSettings {
  enabled: boolean
  idle_timeout_minutes: number
  has_password: boolean
  password?: string | null
  clear_password?: boolean
}

export interface EventsSettings {
  enabled: boolean
  interval_seconds: number
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
  min_liquidity_per_leg: number
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
  btc_eth_hf_maker_mode: boolean
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
}

export interface NetworkSettings {
  allow_network_access: boolean
}

export interface AllSettings {
  polymarket: PolymarketSettings
  kalshi: KalshiSettings
  llm: LLMSettings
  notifications: NotificationSettings
  scanner: ScannerSettings
  live_execution: LiveExecutionSettingsConfig
  maintenance: MaintenanceSettings
  discovery: DiscoverySettings
  trading_proxy: TradingProxySettings
  ui_lock: UILockSettings
  network: NetworkSettings
  events: EventsSettings
  search_filters: SearchFilterSettings
  updated_at: string | null
}

export interface UpdateSettingsRequest {
  polymarket?: Partial<PolymarketSettings>
  kalshi?: Partial<KalshiSettings>
  llm?: Partial<LLMSettings>
  notifications?: Partial<NotificationSettings>
  scanner?: Partial<ScannerSettings>
  live_execution?: Partial<LiveExecutionSettingsConfig>
  maintenance?: Partial<MaintenanceSettings>
  discovery?: Partial<DiscoverySettings>
  trading_proxy?: Partial<TradingProxySettings>
  ui_lock?: Partial<UILockSettings>
  network?: Partial<NetworkSettings>
  events?: Partial<EventsSettings>
  search_filters?: Partial<SearchFilterSettings>
}

export type SettingsTransferCategory =
  | 'bot_traders'
  | 'strategies'
  | 'data_sources'
  | 'market_credentials'
  | 'vpn_configuration'
  | 'llm_configuration'
  | 'telegram_configuration'

export interface BotTraderLiveWalletStateBundle {
  runtime_states?: Array<Record<string, unknown>>
  orders?: Array<Record<string, unknown>>
  positions?: Array<Record<string, unknown>>
}

export interface BotTraderTradeStateBundle {
  orders: Array<Record<string, unknown>>
  live_wallet_state?: BotTraderLiveWalletStateBundle
}

export interface BotTradersBundle {
  traders: Array<Record<string, unknown>>
  orchestrator?: Record<string, unknown> | null
  trade_state?: BotTraderTradeStateBundle
}

export interface SettingsExportBundle {
  schema_version: number
  exported_at: string
  categories: SettingsTransferCategory[]
  bot_traders?: BotTradersBundle
  strategies?: Array<Record<string, unknown>>
  data_sources?: Array<Record<string, unknown>>
  market_credentials?: Record<string, unknown>
  vpn_configuration?: Record<string, unknown>
  llm_configuration?: Record<string, unknown>
  telegram_configuration?: Record<string, unknown>
}

export interface ExportSettingsRequest {
  include_categories?: SettingsTransferCategory[]
}

export interface ExportSettingsResponse {
  status: string
  bundle: SettingsExportBundle
  counts: Record<string, number>
}

export interface ImportSettingsRequest {
  bundle: Record<string, unknown>
  include_categories?: SettingsTransferCategory[]
}

export interface ImportSettingsResponse {
  status: string
  imported_categories: SettingsTransferCategory[]
  results: Record<string, Record<string, number>>
  imported_at: string
}

export const getSettings = async (): Promise<AllSettings> => {
  const { data } = await api.get('/settings')
  return unwrapApiData(data)
}

export const updateSettings = async (settings: UpdateSettingsRequest): Promise<{ status: string; message: string; updated_at: string }> => {
  const { data } = await api.put('/settings', settings)
  return unwrapApiData(data)
}

export const exportSettingsBundle = async (request: ExportSettingsRequest): Promise<ExportSettingsResponse> => {
  const { data } = await api.post('/settings/export', request)
  return unwrapApiData(data)
}

export const importSettingsBundle = async (request: ImportSettingsRequest): Promise<ImportSettingsResponse> => {
  const { data } = await api.post('/settings/import', request)
  return unwrapApiData(data)
}

export interface UILockStatus {
  enabled: boolean
  locked: boolean
  idle_timeout_minutes: number
  has_password: boolean
}

export const getUILockStatus = async (): Promise<UILockStatus> => {
  const { data } = await api.get('/ui-lock/status')
  return unwrapApiData(data)
}

export const unlockUILock = async (password: string): Promise<{ status: string; message: string; locked: boolean }> => {
  const { data } = await api.post('/ui-lock/unlock', { password })
  return unwrapApiData(data)
}

export const lockUILock = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/ui-lock/lock')
  return unwrapApiData(data)
}

export const sendUILockActivity = async (): Promise<{ status: string }> => {
  const { data } = await api.post('/ui-lock/activity')
  return unwrapApiData(data)
}

export const updatePolymarketSettings = async (settings: Partial<PolymarketSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/polymarket', settings)
  return unwrapApiData(data)
}

export const updateLLMSettings = async (settings: Partial<LLMSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/llm', settings)
  return unwrapApiData(data)
}

export const updateNotificationSettings = async (settings: Partial<NotificationSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/notifications', settings)
  return unwrapApiData(data)
}

export const updateScannerSettings = async (settings: Partial<ScannerSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/scanner', settings)
  return unwrapApiData(data)
}

export const updateLiveExecutionSettings = async (
  settings: Partial<LiveExecutionSettingsConfig>
): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/live-execution', settings)
  return unwrapApiData(data)
}

export const updateMaintenanceSettings = async (settings: Partial<MaintenanceSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/maintenance', settings)
  return unwrapApiData(data)
}

export const getDiscoverySettings = async (): Promise<DiscoverySettings> => {
  const { data } = await api.get('/settings/discovery')
  return unwrapApiData(data)
}

export const updateDiscoverySettings = async (settings: Partial<DiscoverySettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/discovery', settings)
  return unwrapApiData(data)
}

export interface LLMModelOption {
  id: string
  name: string
}

export interface LLMModelsResponse {
  models: Record<string, LLMModelOption[]>
}

export interface LLMTestResponse {
  status: string
  message: string
  provider?: string
  model_count?: number
}

export interface RefreshModelsResponse {
  status: string
  message: string
  models: Record<string, LLMModelOption[]>
}

export const getLLMModels = async (provider?: string): Promise<LLMModelsResponse> => {
  const { data } = await api.get('/settings/llm/models', { params: provider ? { provider } : undefined })
  return unwrapApiData(data)
}

export const refreshLLMModels = async (provider?: string): Promise<RefreshModelsResponse> => {
  const { data } = await api.post('/settings/llm/models/refresh', null, { params: provider ? { provider } : undefined })
  return unwrapApiData(data)
}

export const testLLMConnection = async (provider?: string): Promise<LLMTestResponse> => {
  const { data } = await api.post('/settings/test/llm', null, { params: provider ? { provider } : undefined })
  return unwrapApiData(data)
}

export const testPolymarketConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/polymarket')
  return unwrapApiData(data)
}

export const testKalshiConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/kalshi')
  return unwrapApiData(data)
}

export const testTelegramConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/telegram')
  return unwrapApiData(data)
}

export const testTradingProxy = async (): Promise<{ status: string; message: string; proxy_enabled?: boolean; proxy_ip?: string; direct_ip?: string; vpn_active?: boolean }> => {
  const { data } = await api.post('/settings/test/trading-proxy')
  return unwrapApiData(data)
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
  events_resolver_7d?: {
    window_days: number
    signals_sampled: number
    candidates: number
    tradable: number
    tradable_rate: number
    by_signal_type: Array<Record<string, unknown>>
  }
  jobs: ValidationJob[]
}

export interface ValidationJob {
  id: string
  job_type: 'backtest' | 'optimize' | 'execution_simulation' | 'live_truth_monitor' | string
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

export interface LiveTruthMonitorJobRequest {
  trader_id?: string
  trader_name?: string
  duration_seconds?: number
  poll_seconds?: number
  run_llm_analysis?: boolean
  llm_model?: string
  include_strategy_source?: boolean
  max_alerts_for_llm?: number
  enable_provider_checks?: boolean
}

export interface LiveTruthMonitorReportPayload {
  path: string
  line_count: number
  alert_count: number
  heartbeat_count: number
  transition_count: number
  alerts_by_rule: Record<string, number>
  alert_samples: Array<Record<string, unknown>>
}

export interface LiveTruthMonitorRawResponse {
  job_id: string
  job_status: ValidationJob['status']
  payload: Record<string, unknown>
  monitor: {
    summary: Record<string, unknown>
    summary_path: string
    report_path: string
    stdout_events: Array<Record<string, unknown>>
    report: LiveTruthMonitorReportPayload
  }
  llm_analysis: Record<string, unknown>
}

export type LiveTruthMonitorArtifact = 'summary_json' | 'report_jsonl' | 'llm_analysis_json' | 'bundle_json'

export interface ValidationJobEnqueueResponse {
  status: string
  job_id: string
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
  return unwrapApiData(data)
}

export const enqueueLiveTruthMonitorJob = async (
  payload: LiveTruthMonitorJobRequest
): Promise<ValidationJobEnqueueResponse> => {
  const { data } = await api.post('/validation/jobs/live-truth-monitor', payload)
  return unwrapApiData(data)
}

export const getValidationJob = async (jobId: string): Promise<ValidationJob> => {
  const { data } = await api.get(`/validation/jobs/${jobId}`)
  return unwrapApiData(data)
}

export const getLiveTruthMonitorRaw = async (
  jobId: string,
  params?: { max_alerts?: number }
): Promise<LiveTruthMonitorRawResponse> => {
  const { data } = await api.get(`/validation/jobs/${jobId}/live-truth-monitor/raw`, { params })
  return unwrapApiData(data)
}

function resolveExportFilename(contentDisposition: unknown, fallback: string): string {
  const header = String(contentDisposition || '').trim()
  if (!header) return fallback
  const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i)
  if (utf8Match && utf8Match[1]) {
    try {
      return decodeURIComponent(utf8Match[1].trim().replace(/^"|"$/g, '')) || fallback
    } catch {
      return utf8Match[1].trim().replace(/^"|"$/g, '') || fallback
    }
  }
  const plainMatch = header.match(/filename=([^;]+)/i)
  if (!plainMatch || !plainMatch[1]) return fallback
  return plainMatch[1].trim().replace(/^"|"$/g, '') || fallback
}

export const exportLiveTruthMonitorArtifact = async (
  jobId: string,
  artifact: LiveTruthMonitorArtifact
): Promise<{ filename: string; mediaType: string; blob: Blob }> => {
  const response = await api.get(`/validation/jobs/${jobId}/live-truth-monitor/export`, {
    params: { artifact },
    responseType: 'blob',
  })
  const mediaType = String(response.headers['content-type'] || 'application/octet-stream')
  const fallback = `live_truth_monitor_${jobId.slice(0, 8)}_${artifact}.json`
  const filename = resolveExportFilename(response.headers['content-disposition'], fallback)
  return {
    filename,
    mediaType,
    blob: response.data as Blob,
  }
}

export const overrideValidationStrategy = async (
  strategyType: string,
  status: 'active' | 'demoted',
  note?: string
): Promise<Record<string, unknown>> => {
  const { data } = await api.post(`/validation/strategy-health/${strategyType}/override`, null, {
    params: { status, note }
  })
  return unwrapApiData(data)
}

export const clearValidationStrategyOverride = async (strategyType: string): Promise<Record<string, unknown>> => {
  const { data } = await api.delete(`/validation/strategy-health/${strategyType}/override`)
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

export const triggerNewsFetch = async (): Promise<{ new_articles: number; total_articles: number; articles: Array<{ title: string; source: string; feed_source: string; url: string; published: string | null; category: string }> }> => {
  const { data } = await api.post('/news/feed/fetch')
  return unwrapApiData(data)
}

export const getNewsArticles = async (params?: {
  max_age_hours?: number
  source?: string
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; has_more: boolean; articles: NewsArticle[] }> => {
  const { data } = await api.get('/news/feed/articles', { params })
  return unwrapApiData(data)
}

export const searchNewsArticles = async (params: {
  q: string
  max_age_hours?: number
  limit?: number
}): Promise<{ query: string; total: number; articles: NewsArticle[] }> => {
  const { data } = await api.get('/news/feed/search', { params })
  return unwrapApiData(data)
}

export const clearNewsArticles = async (): Promise<{ cleared: number }> => {
  const { data } = await api.delete('/news/feed/clear')
  return unwrapApiData(data)
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
  return unwrapApiData(data)
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
  return unwrapApiData(data)
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
  actions_applied?: Array<{
    type: string
    strategy_id?: string
    data_source_id?: string
    slug?: string
    version?: number
    status?: string
  }>
  action_errors?: Array<{
    index?: number
    type?: string
    error: string
  }>
}

export const sendAIChat = async (params: {
  message: string
  session_id?: string
  context_type?: string
  context_id?: string
  history?: AIChatMessage[]
  allow_actions?: boolean
}): Promise<AIChatResponse> => {
  const { data } = await api.post('/ai/chat', params, AI_TIMEOUT)
  return unwrapApiData(data)
}

export interface AIStrategyDraftGenerationResponse {
  name: string
  slug: string
  source_key: string
  description: string
  source_code: string
  class_name: string
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  aliases: string[]
  validation: {
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
    capabilities: Record<string, boolean>
    inferred_type: string
  }
  used_repair_pass: boolean
  model: string
  tokens_used: Record<string, number>
}

export const generateAIStrategyDraft = async (params: {
  description: string
  source_key?: string
  model?: string
}): Promise<AIStrategyDraftGenerationResponse> => {
  const { data } = await api.post('/ai/generate/strategy-draft', params, AI_TIMEOUT)
  return unwrapApiData(data)
}

export interface AIDataSourceDraftGenerationResponse {
  name: string
  slug: string
  source_key: string
  source_kind: string
  description: string
  source_code: string
  class_name: string
  retention: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  validation: {
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
    capabilities: Record<string, boolean>
  }
  used_repair_pass: boolean
  model: string
  tokens_used: Record<string, number>
}

export const generateAIDataSourceDraft = async (params: {
  description: string
  source_key?: string
  source_kind?: string
  model?: string
}): Promise<AIDataSourceDraftGenerationResponse> => {
  const { data } = await api.post('/ai/generate/data-source-draft', params, AI_TIMEOUT)
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

export const getAIChatSession = async (sessionId: string): Promise<AIChatSessionDetail> => {
  const { data } = await api.get(`/ai/chat/sessions/${sessionId}`)
  return unwrapApiData(data)
}

export const archiveAIChatSession = async (sessionId: string): Promise<{ status: string; session_id: string }> => {
  const { data } = await api.delete(`/ai/chat/sessions/${sessionId}`)
  return unwrapApiData(data)
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
  return unwrapApiData(data)
}

export const loginKalshi = async (params: {
  email?: string
  password?: string
  api_key?: string
}): Promise<{ status: string; message: string; authenticated: boolean; member_id?: string }> => {
  const { data } = await api.post('/kalshi/login', params)
  return unwrapApiData(data)
}

export const logoutKalshi = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/kalshi/logout')
  return unwrapApiData(data)
}

export const getKalshiBalance = async (): Promise<{ balance: number; available: number; reserved: number; currency: string }> => {
  const { data } = await api.get('/kalshi/balance')
  return unwrapApiData(data)
}

export const getKalshiPositions = async (): Promise<KalshiPosition[]> => {
  const { data } = await api.get('/kalshi/positions')
  return unwrapApiData(data)
}

export const updateKalshiSettings = async (settings: Partial<KalshiSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/kalshi', settings)
  return unwrapApiData(data)
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
  strategy_sdk?: string | null
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
  price_history?: Array<Record<string, unknown> | unknown[]>
  outcome_labels?: string[]
  outcome_prices?: number[]
  market_token_ids?: string[]
  yes_price?: number | null
  no_price?: number | null
  current_yes_price?: number | null
  current_no_price?: number | null
  market_platform?: string | null
  market_slug?: string | null
  market_event_slug?: string | null
  market_event_ticker?: string | null
  market_url?: string | null
  polymarket_url?: string | null
  kalshi_url?: string | null
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
  orchestrator_enabled: boolean
  orchestrator_min_edge: number
  orchestrator_max_age_minutes: number
  model: string | null
  article_max_age_hours: number
  cycle_spend_cap_usd: number
  hourly_spend_cap_usd: number
  cycle_llm_call_cap: number
  cache_ttl_minutes: number
  max_edge_evals_per_article: number
}

type NewsWorkflowSettingsUpdate = Omit<NewsWorkflowSettings, 'article_max_age_hours'>

export const getNewsWorkflowStatus = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.get('/news-workflow/status')
  return unwrapApiData(data)
}

export const runNewsWorkflow = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/news-workflow/run')
  return unwrapApiData(data)
}

export const startNewsWorkflow = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/start')
  return unwrapApiData(data)
}

export const pauseNewsWorkflow = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/pause')
  return unwrapApiData(data)
}

export const setNewsWorkflowInterval = async (intervalSeconds: number): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/interval', null, {
    params: { interval_seconds: intervalSeconds },
  })
  return unwrapApiData(data)
}

export const getNewsWorkflowSettings = async (): Promise<NewsWorkflowSettings> => {
  const { data } = await api.get('/news-workflow/settings')
  return unwrapApiData(data)
}

export const updateNewsWorkflowSettings = async (
  settings: Partial<NewsWorkflowSettingsUpdate>
): Promise<{ status: string; settings: NewsWorkflowSettings }> => {
  const { data } = await api.put('/news-workflow/settings', settings)
  return unwrapApiData(data)
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
  return unwrapApiData(data)
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
  default_size_usd: number
  max_size_usd: number
  model: string | null
  temperature_unit: 'F' | 'C'
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

export interface WeatherOpportunityDateBucket {
  date: string
  count: number
}

export interface WeatherWorkflowOpportunityIdsResponse {
  total: number
  offset: number
  limit: number
  ids: string[]
}

export const getWeatherWorkflowStatus = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.get('/weather-workflow/status')
  return unwrapApiData(data)
}

export const runWeatherWorkflow = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/weather-workflow/run')
  return unwrapApiData(data)
}

export const startWeatherWorkflow = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/start')
  return unwrapApiData(data)
}

export const pauseWeatherWorkflow = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/pause')
  return unwrapApiData(data)
}

export const setWeatherWorkflowInterval = async (
  intervalSeconds: number
): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/interval', null, {
    params: { interval_seconds: intervalSeconds }
  })
  return unwrapApiData(data)
}

export const getWeatherWorkflowOpportunities = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  target_date?: string
  include_report_only?: boolean
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; opportunities: Opportunity[] }> => {
  const { data } = await api.get('/weather-workflow/opportunities', { params })
  return unwrapApiData(data)
}

export const getWeatherWorkflowOpportunityDates = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  include_report_only?: boolean
}): Promise<{ total_dates: number; dates: WeatherOpportunityDateBucket[] }> => {
  const { data } = await api.get('/weather-workflow/opportunity-dates', { params })
  return unwrapApiData(data)
}

export const getWeatherWorkflowOpportunityIds = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  target_date?: string
  include_report_only?: boolean
  limit?: number
  offset?: number
}): Promise<WeatherWorkflowOpportunityIdsResponse> => {
  const { data } = await api.get('/weather-workflow/opportunity-ids', { params })
  return unwrapApiData(data)
}

export const getWeatherWorkflowIntents = async (params?: {
  status_filter?: string
  limit?: number
}): Promise<{ total: number; intents: WeatherTradeIntent[] }> => {
  const { data } = await api.get('/weather-workflow/intents', { params })
  return unwrapApiData(data)
}

export const skipWeatherWorkflowIntent = async (intentId: string): Promise<{ status: string; intent_id: string }> => {
  const { data } = await api.post(`/weather-workflow/intents/${intentId}/skip`)
  return unwrapApiData(data)
}

export const getWeatherWorkflowSettings = async (): Promise<WeatherWorkflowSettings> => {
  const { data } = await api.get('/weather-workflow/settings')
  return unwrapApiData(data)
}

export const updateWeatherWorkflowSettings = async (
  settings: Partial<WeatherWorkflowSettings>
): Promise<{ status: string; settings: WeatherWorkflowSettings }> => {
  const { data } = await api.put('/weather-workflow/settings', settings)
  return unwrapApiData(data)
}

export const getWeatherWorkflowPerformance = async (
  lookbackDays = 90
): Promise<WeatherWorkflowPerformance> => {
  const { data } = await api.get('/weather-workflow/performance', {
    params: { lookback_days: lookbackDays }
  })
  return unwrapApiData(data)
}

export default api

// ==================== UNIFIED STRATEGY API ====================

export interface UnifiedStrategy {
  id: string
  slug: string
  source_key: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  is_system: boolean
  enabled: boolean
  status: string
  error_message: string | null
  version: number
  config: Record<string, unknown>
  config_schema: Record<string, unknown> | null
  aliases: string[]
  sort_order: number
  created_at: string | null
  updated_at: string | null
  capabilities: {
    has_detect: boolean
    has_detect_async: boolean
    has_evaluate: boolean
    has_should_exit: boolean
  }
  strategy_type: string  // 'detect' | 'execute' | 'unified'
  runtime: Record<string, any> | null
}

export const getUnifiedStrategies = async (params?: {
  type?: string
  source_key?: string
  enabled?: boolean
}): Promise<UnifiedStrategy[]> => {
  const { data } = await api.get('/strategy-manager', { params })
  return getStrategyManagerItems(data)
}

export const getUnifiedStrategy = async (id: string): Promise<UnifiedStrategy> => {
  const { data } = await api.get(`/strategy-manager/${id}`)
  return unwrapStrategyManagerPayload(data)
}

export const createUnifiedStrategy = async (payload: {
  slug: string
  source_key?: string
  name?: string
  description?: string
  source_code: string
  class_name?: string
  config?: Record<string, unknown>
  config_schema?: Record<string, unknown>
  aliases?: string[]
  enabled?: boolean
}): Promise<UnifiedStrategy> => {
  const { data } = await api.post('/strategy-manager', payload)
  return unwrapApiData(data)
}

export const updateUnifiedStrategy = async (
  id: string,
  payload: Partial<{
    slug: string
    source_key: string
    name: string
    description: string
    source_code: string
    class_name: string
    config: Record<string, unknown>
    config_schema: Record<string, unknown>
    aliases: string[]
    enabled: boolean
    unlock_system: boolean
  }>
): Promise<UnifiedStrategy> => {
  const { data } = await api.put(`/strategy-manager/${id}`, payload)
  return unwrapApiData(data)
}

export const deleteUnifiedStrategy = async (id: string): Promise<void> => {
  await api.delete(`/strategy-manager/${id}`)
}

export const validateUnifiedStrategy = async (source_code: string, class_name?: string): Promise<{
  valid: boolean
  inferred_type: string
  capabilities: Record<string, boolean>
  class_name: string | null
  errors: string[]
  warnings: string[]
}> => {
  const { data } = await api.post('/strategy-manager/validate', { source_code, class_name })
  return unwrapApiData(data)
}

export const reloadUnifiedStrategy = async (id: string): Promise<{
  status: string
  message?: string
  runtime?: Record<string, any>
}> => {
  const { data } = await api.post(`/strategy-manager/${id}/reload`)
  return unwrapApiData(data)
}

export const getUnifiedStrategyTemplate = async (): Promise<{
  template: string
  instructions: string
  available_imports: string[]
}> => {
  const { data } = await api.get('/strategy-manager/template')
  return unwrapApiData(data)
}

export const getUnifiedStrategyDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/strategy-manager/docs')
  const payload = unwrapStrategyManagerPayload(data)
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return {}
  }
  return payload
}

export interface UnifiedStrategyVersion {
  id: string
  strategy_id: string
  strategy_slug: string
  source_key: string
  version: number
  is_latest: boolean
  name: string
  description: string | null
  source_code?: string
  class_name: string | null
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  aliases: string[]
  enabled: boolean
  is_system: boolean
  sort_order: number
  parent_version: number | null
  created_by: string | null
  reason: string | null
  created_at: string | null
}

export const getUnifiedStrategyVersions = async (
  strategyId: string,
  params?: { limit?: number; include_source?: boolean }
): Promise<UnifiedStrategyVersion[]> => {
  const { data } = await api.get(`/strategy-manager/${strategyId}/versions`, { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

export const getUnifiedStrategyVersion = async (
  strategyId: string,
  version: number | string
): Promise<UnifiedStrategyVersion> => {
  const { data } = await api.get(`/strategy-manager/${strategyId}/versions/${version}`)
  return unwrapApiData(data)
}

export const restoreUnifiedStrategyVersion = async (
  strategyId: string,
  version: number | string,
  payload?: { reason?: string; created_by?: string }
): Promise<{
  status: string
  strategy: UnifiedStrategy
  restored_snapshot: UnifiedStrategyVersion
  source_snapshot: UnifiedStrategyVersion
}> => {
  const { data } = await api.post(`/strategy-manager/${strategyId}/versions/${version}/restore`, payload || {})
  return unwrapApiData(data)
}

export interface StrategyExperiment {
  id: string
  name: string
  source_key: string
  strategy_key: string
  control_version: number
  candidate_version: number
  candidate_allocation_pct: number
  scope: Record<string, unknown>
  status: 'active' | 'paused' | 'completed' | 'archived' | string
  created_by: string | null
  notes: string | null
  metadata: Record<string, unknown>
  promoted_version: number | null
  ended_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface StrategyExperimentAssignment {
  id: string
  experiment_id: string
  trader_id: string | null
  signal_id: string | null
  decision_id: string | null
  order_id: string | null
  source_key: string
  strategy_key: string
  strategy_version: number
  assignment_group: string
  payload: Record<string, unknown>
  created_at: string | null
  updated_at: string | null
}

export const listStrategyExperiments = async (params?: {
  source_key?: string
  strategy_key?: string
  status?: string
  limit?: number
}): Promise<StrategyExperiment[]> => {
  const { data } = await api.get('/strategy-manager/experiments', { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

export const createStrategyExperiment = async (payload: {
  name: string
  source_key: string
  strategy_key: string
  control_version: number
  candidate_version: number
  candidate_allocation_pct?: number
  scope?: Record<string, unknown>
  notes?: string
  created_by?: string
}): Promise<StrategyExperiment> => {
  const { data } = await api.post('/strategy-manager/experiments', payload)
  return unwrapApiData(data)
}

export const setStrategyExperimentStatus = async (
  experimentId: string,
  status: string
): Promise<StrategyExperiment> => {
  const { data } = await api.post(`/strategy-manager/experiments/${experimentId}/status`, { status })
  return unwrapApiData(data)
}

export const promoteStrategyExperiment = async (
  experimentId: string,
  payload?: { promoted_version?: number; notes?: string }
): Promise<StrategyExperiment> => {
  const { data } = await api.post(`/strategy-manager/experiments/${experimentId}/promote`, payload || {})
  return unwrapApiData(data)
}

export const listStrategyExperimentAssignments = async (
  experimentId: string,
  params?: { limit?: number }
): Promise<StrategyExperimentAssignment[]> => {
  const { data } = await api.get(`/strategy-manager/experiments/${experimentId}/assignments`, { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

export interface TraderTuneAgentRequest {
  prompt: string
  model?: string
  max_iterations?: number
  monitor_job_id?: string
}

export interface TraderTuneAgentResponse {
  session_id: string
  answer: string
  parsed: Record<string, unknown> | null
  applied_param_updates: Array<{
    source_key: string
    strategy_key: string
    changed_keys: string[]
    params_patch: Record<string, unknown>
    reason: string
  }>
  applied_param_update_count: number
  updated_trader: Trader | null
  raw: Record<string, unknown>
}

export const runTraderTuneIteration = async (
  traderId: string,
  payload: TraderTuneAgentRequest
): Promise<TraderTuneAgentResponse> => {
  const { data } = await api.post(`/traders/${traderId}/tune/iterate`, payload, { timeout: 240_000 })
  return unwrapApiData(data)
}

// ==================== UNIFIED DATA SOURCE API ====================

export interface UnifiedDataSource {
  id: string
  slug: string
  source_key: string
  source_kind: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  is_system: boolean
  enabled: boolean
  status: string
  error_message: string | null
  version: number
  retention: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown> | null
  sort_order: number
  created_at: string | null
  updated_at: string | null
  capabilities: {
    has_fetch: boolean
    has_fetch_async: boolean
    has_transform: boolean
  }
  runtime: Record<string, any> | null
}

export interface UnifiedDataSourceRun {
  id: string
  status: string
  fetched_count: number
  transformed_count: number
  upserted_count: number
  skipped_count: number
  error_message: string | null
  metadata: Record<string, unknown>
  started_at: string | null
  completed_at: string | null
  duration_ms: number | null
}

export interface UnifiedDataSourceRecord {
  id: string
  external_id: string | null
  title: string | null
  summary: string | null
  category: string | null
  source: string | null
  url: string | null
  geotagged: boolean
  country_iso3: string | null
  latitude: number | null
  longitude: number | null
  observed_at: string | null
  ingested_at: string | null
  payload: Record<string, unknown>
  transformed: Record<string, unknown>
  tags: string[]
}

export interface UnifiedDataSourceTemplatePreset {
  id: string
  slug_prefix?: string
  name: string
  description: string
  source_key: string
  source_kind: string
  source_code: string
  retention?: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  is_system_seed?: boolean
}

export interface UnifiedDataSourceTemplateResponse {
  template: string
  default_preset?: string
  presets?: UnifiedDataSourceTemplatePreset[]
  instructions: string
  available_imports: string[]
}

export const getUnifiedDataSources = async (params?: {
  source_key?: string
  enabled?: boolean
  include_code?: boolean
}): Promise<UnifiedDataSource[]> => {
  const { data } = await api.get('/data-sources', { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (Array.isArray(payload?.items)) return payload.items
  return []
}

export const getUnifiedDataSource = async (id: string): Promise<UnifiedDataSource> => {
  const { data } = await api.get(`/data-sources/${id}`)
  return unwrapApiData(data)
}

export const createUnifiedDataSource = async (payload: {
  slug: string
  source_key?: string
  source_kind?: string
  name?: string
  description?: string
  source_code: string
  retention?: {
    max_records?: number
    max_age_days?: number
  }
  config?: Record<string, unknown>
  config_schema?: Record<string, unknown>
  enabled?: boolean
}): Promise<UnifiedDataSource> => {
  const { data } = await api.post('/data-sources', payload)
  return unwrapApiData(data)
}

export const updateUnifiedDataSource = async (
  id: string,
  payload: Partial<{
    slug: string
    source_key: string
    source_kind: string
    name: string
    description: string
    source_code: string
    retention: {
      max_records?: number
      max_age_days?: number
    }
    config: Record<string, unknown>
    config_schema: Record<string, unknown>
    enabled: boolean
    unlock_system: boolean
  }>
): Promise<UnifiedDataSource> => {
  const { data } = await api.put(`/data-sources/${id}`, payload)
  return unwrapApiData(data)
}

export const deleteUnifiedDataSource = async (id: string): Promise<void> => {
  await api.delete(`/data-sources/${id}`)
}

export const validateUnifiedDataSource = async (source_code: string, class_name?: string): Promise<{
  valid: boolean
  class_name: string | null
  source_name: string | null
  source_description: string | null
  capabilities: Record<string, boolean>
  errors: string[]
  warnings: string[]
}> => {
  const { data } = await api.post('/data-sources/validate', { source_code, class_name })
  return unwrapApiData(data)
}

export const reloadUnifiedDataSource = async (id: string): Promise<{
  status: string
  message?: string
  runtime?: Record<string, any> | null
}> => {
  const { data } = await api.post(`/data-sources/${id}/reload`)
  return unwrapApiData(data)
}

export const runUnifiedDataSource = async (
  id: string,
  payload?: { max_records?: number }
): Promise<{
  run_id: string
  source_slug: string
  status: string
  fetched_count: number
  transformed_count: number
  upserted_count: number
  skipped_count: number
  error_message: string | null
  duration_ms: number | null
}> => {
  const { data } = await api.post(`/data-sources/${id}/run`, payload || {})
  return unwrapApiData(data)
}

export const getUnifiedDataSourceRuns = async (
  id: string,
  params?: { limit?: number }
): Promise<{
  source_id: string
  source_slug: string
  runs: UnifiedDataSourceRun[]
}> => {
  const { data } = await api.get(`/data-sources/${id}/runs`, { params })
  return unwrapApiData(data)
}

export const getUnifiedDataSourceRecords = async (
  id: string,
  params?: { limit?: number; offset?: number; geotagged?: boolean }
): Promise<{
  source_id: string
  source_slug: string
  total: number
  offset: number
  limit: number
  records: UnifiedDataSourceRecord[]
}> => {
  const { data } = await api.get(`/data-sources/${id}/records`, { params })
  return unwrapApiData(data)
}

export const previewUnifiedDataSource = async (
  id: string,
  payload?: { max_records?: number }
): Promise<{
  source_id: string
  source_slug: string
  total_fetched: number
  duration_ms: number
  records: UnifiedDataSourceRecord[]
}> => {
  const { data } = await api.post(`/data-sources/${id}/preview`, payload || {})
  return unwrapApiData(data)
}

export const getUnifiedDataSourceTemplate = async (): Promise<UnifiedDataSourceTemplateResponse> => {
  const { data } = await api.get('/data-sources/template')
  return unwrapApiData(data)
}

export const getUnifiedDataSourceDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/data-sources/docs')
  return unwrapApiData(data)
}


// ==================== ML Training Pipeline ====================

export interface MLRecorderConfig {
  is_recording: boolean
  interval_seconds: number
  retention_days: number
  assets: string[]
  timeframes: string[]
  schedule_enabled: boolean
  schedule_start_utc: string | null
  schedule_end_utc: string | null
}

export interface MLRecorderStats {
  config: MLRecorderConfig
  total_snapshots: number
  oldest_snapshot: string | null
  newest_snapshot: string | null
  snapshots_by_asset: Record<string, number>
}

export interface MLTrainingJob {
  id: string
  status: string
  model_type: string
  assets: string[]
  timeframes: string[]
  progress: number
  message: string | null
  error: string | null
  trained_model_id: string | null
  result_summary: Record<string, any> | null
  created_at: string | null
  started_at: string | null
  finished_at: string | null
}

export interface MLTrainedModel {
  id: string
  name: string
  model_type: string
  version: number
  status: string
  assets: string[]
  timeframes: string[]
  train_accuracy: number | null
  test_accuracy: number | null
  test_auc: number | null
  feature_importance: Record<string, number> | null
  train_samples: number
  test_samples: number
  training_date_range: { start: string | null; end: string | null } | null
  walkforward_results: any[] | null
  created_at: string | null
  promoted_at: string | null
  notes: string | null
}

export interface MLDataStats {
  total_snapshots: number
  oldest: string | null
  newest: string | null
  groups: Array<{
    asset: string
    timeframe: string
    count: number
    oldest: string | null
    newest: string | null
  }>
}

// Recorder
export const getMLRecorderConfig = async (): Promise<MLRecorderStats> => {
  const { data } = await api.get('/ml/recorder/config')
  return data
}

export const updateMLRecorderConfig = async (config: Partial<MLRecorderConfig>): Promise<MLRecorderStats> => {
  const { data } = await api.put('/ml/recorder/config', config)
  return data
}

// Data
export const getMLDataStats = async (): Promise<MLDataStats> => {
  const { data } = await api.get('/ml/data/stats')
  return data
}

export const pruneMLData = async (olderThanDays?: number, confirm = false) => {
  const { data } = await api.post('/ml/data/prune', { older_than_days: olderThanDays, confirm })
  return data
}

export const deleteAllMLData = async () => {
  const { data } = await api.delete('/ml/data/all', { params: { confirm: true } })
  return data
}

// Training
export const startMLTraining = async (params: {
  model_type?: string
  assets?: string[]
  timeframes?: string[]
  days_lookback?: number
  hyperparams?: Record<string, any>
}): Promise<{ job_id: string; status: string }> => {
  const { data } = await api.post('/ml/train', params)
  return data
}

export const getMLTrainingJobs = async (status?: string): Promise<MLTrainingJob[]> => {
  const { data } = await api.get('/ml/train/jobs', { params: { status } })
  return Array.isArray(data) ? data : []
}

export const getMLTrainingJob = async (jobId: string): Promise<MLTrainingJob> => {
  const { data } = await api.get(`/ml/train/jobs/${jobId}`)
  return data
}

// Models
export const getMLModels = async (status?: string): Promise<MLTrainedModel[]> => {
  const { data } = await api.get('/ml/models', { params: { status } })
  return Array.isArray(data) ? data : []
}

export const getMLModel = async (modelId: string): Promise<MLTrainedModel> => {
  const { data } = await api.get(`/ml/models/${modelId}`)
  return data
}

export const promoteMLModel = async (modelId: string) => {
  const { data } = await api.post(`/ml/models/${modelId}/promote`)
  return data
}

export const archiveMLModel = async (modelId: string) => {
  const { data } = await api.post(`/ml/models/${modelId}/archive`)
  return data
}

export const deleteMLModel = async (modelId: string) => {
  const { data } = await api.delete(`/ml/models/${modelId}`, { params: { confirm: true } })
  return data
}

export const getActiveMLModel = async (modelType?: string) => {
  const { data } = await api.get('/ml/active-model', { params: { model_type: modelType } })
  return data
}
