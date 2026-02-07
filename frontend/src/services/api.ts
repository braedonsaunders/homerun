import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

// ==================== TYPES ====================

export interface Opportunity {
  id: string
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
  event_title?: string
  category?: string
  min_liquidity: number
  max_position_size: number
  detected_at: string
  resolution_date?: string
  positions_to_take: Position[]
}

export interface Market {
  id: string
  question: string
  yes_price: number
  no_price: number
  liquidity: number
}

export interface Position {
  action: string
  outcome: string
  market: string
  price: number
  token_id?: string
}

export interface ScannerStatus {
  running: boolean
  enabled: boolean
  interval_seconds: number
  last_scan: string | null
  opportunities_count: number
  strategies: Strategy[]
}

export interface Strategy {
  type: string
  name: string
  description: string
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
  settings: {
    min_roi_threshold: number
    max_position_size: number
    copy_delay_seconds: number
    slippage_tolerance: number
  }
  stats: {
    total_copied: number
    successful_copies: number
    failed_copies: number
    total_pnl: number
  }
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

// ==================== WALLETS ====================

export const getWallets = async (): Promise<Wallet[]> => {
  const { data } = await api.get('/wallets')
  return data
}

export const addWallet = async (address: string, label?: string) => {
  const { data } = await api.post('/wallets', null, {
    params: { address, label }
  })
  return data
}

export const removeWallet = async (address: string) => {
  const { data } = await api.delete(`/wallets/${address}`)
  return data
}

// Recent trades from tracked wallets
export interface RecentTradeFromWallet {
  id?: string
  market?: string
  market_title?: string
  market_slug?: string
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
  return data
}

export const getWalletPositions = async (address: string) => {
  const { data } = await api.get(`/wallets/${address}/positions`)
  return data
}

export const getWalletTrades = async (address: string, limit = 100) => {
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
}) => {
  const { data } = await api.get('/markets', { params })
  return data
}

export const getEvents = async (params?: {
  closed?: boolean
  limit?: number
  offset?: number
}) => {
  const { data } = await api.get('/events', { params })
  return data
}

// ==================== SIMULATION ====================

export const createSimulationAccount = async (params: {
  name: string
  initial_capital?: number
  max_position_pct?: number
  max_positions?: number
}) => {
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

export const deleteSimulationAccount = async (accountId: string) => {
  const { data } = await api.delete(`/simulation/accounts/${accountId}`)
  return data
}

export const getAccountPositions = async (accountId: string) => {
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
) => {
  const { data } = await api.post(`/simulation/accounts/${accountId}/execute`, {
    opportunity_id: opportunityId,
    position_size: positionSize,
    take_profit_price: takeProfitPrice,
    stop_loss_price: stopLossPrice
  })
  return data
}

export const getAccountPerformance = async (accountId: string) => {
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
  min_roi_threshold?: number
  max_position_size?: number
  copy_delay_seconds?: number
  slippage_tolerance?: number
}) => {
  const { data } = await api.post('/copy-trading/configs', params)
  return data
}

export const deleteCopyConfig = async (configId: string) => {
  const { data } = await api.delete(`/copy-trading/configs/${configId}`)
  return data
}

export const enableCopyConfig = async (configId: string) => {
  const { data } = await api.post(`/copy-trading/configs/${configId}/enable`)
  return data
}

export const disableCopyConfig = async (configId: string) => {
  const { data } = await api.post(`/copy-trading/configs/${configId}/disable`)
  return data
}

export const getCopyTradingStatus = async () => {
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
}) => {
  const { data } = await api.post('/anomaly/find-profitable', params || {})
  return data
}

export const getAnomalies = async (params?: {
  severity?: string
  anomaly_type?: string
  limit?: number
}) => {
  const { data } = await api.get('/anomaly/anomalies', { params })
  return data
}

export const quickCheckWallet = async (address: string) => {
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

export const getHealthStatus = async () => {
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

export const initializeTrading = async () => {
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
}) => {
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

export const cancelOrder = async (orderId: string) => {
  const { data } = await api.delete(`/trading/orders/${orderId}`)
  return data
}

export const cancelAllOrders = async () => {
  const { data } = await api.delete('/trading/orders')
  return data
}

export const getTradingPositions = async () => {
  const { data } = await api.get('/trading/positions')
  return data
}

export const getTradingBalance = async () => {
  const { data } = await api.get('/trading/balance')
  return data
}

export const executeOpportunityLive = async (params: {
  opportunity_id: string
  positions: any[]
  size_usd: number
}) => {
  const { data } = await api.post('/trading/execute-opportunity', params)
  return data
}

export const emergencyStopTrading = async () => {
  const { data } = await api.post('/trading/emergency-stop')
  return data
}

// ==================== AUTO TRADER ====================

export interface AutoTraderStatus {
  mode: string
  running: boolean
  config: {
    mode: string
    enabled_strategies: string[]
    min_roi_percent: number
    max_risk_score: number
    min_liquidity_usd: number
    base_position_size_usd: number
    max_position_size_usd: number
    max_daily_trades: number
    max_daily_loss_usd: number
    circuit_breaker_losses: number
    require_confirmation: boolean
    paper_account_capital?: number
  }
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
}

export const getAutoTraderStatus = async (): Promise<AutoTraderStatus> => {
  const { data } = await api.get('/auto-trader/status')
  return data
}

export const startAutoTrader = async (mode?: string) => {
  const { data } = await api.post('/auto-trader/start', null, { params: { mode } })
  return data
}

export const stopAutoTrader = async () => {
  const { data } = await api.post('/auto-trader/stop')
  return data
}

export const updateAutoTraderConfig = async (config: Partial<{
  mode: string
  enabled_strategies: string[]
  min_roi_percent: number
  max_risk_score: number
  min_liquidity_usd: number
  base_position_size_usd: number
  max_position_size_usd: number
  max_daily_trades: number
  max_daily_loss_usd: number
  require_confirmation: boolean
}>) => {
  const { data } = await api.put('/auto-trader/config', config)
  return data
}

export const getAutoTraderTrades = async (limit = 100, status?: string): Promise<AutoTraderTrade[]> => {
  const { data } = await api.get('/auto-trader/trades', { params: { limit, status } })
  return data
}

export const getAutoTraderStats = async () => {
  const { data } = await api.get('/auto-trader/stats')
  return data
}

export const resetAutoTraderStats = async () => {
  const { data } = await api.post('/auto-trader/reset-stats')
  return data
}

export const resetCircuitBreaker = async () => {
  const { data } = await api.post('/auto-trader/reset-circuit-breaker')
  return data
}

export const enableLiveTrading = async (maxDailyLoss = 100) => {
  const { data } = await api.post('/auto-trader/enable-live-trading', null, {
    params: { confirm: true, max_daily_loss: maxDailyLoss }
  })
  return data
}

export const emergencyStopAutoTrader = async () => {
  const { data } = await api.post('/auto-trader/emergency-stop')
  return data
}

// ==================== SETTINGS ====================

export interface PolymarketSettings {
  api_key: string | null
  api_secret: string | null
  api_passphrase: string | null
  private_key: string | null
}

export interface LLMSettings {
  provider: string
  openai_api_key: string | null
  anthropic_api_key: string | null
  google_api_key: string | null
  xai_api_key: string | null
  deepseek_api_key: string | null
  model: string | null
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

export interface AllSettings {
  polymarket: PolymarketSettings
  llm: LLMSettings
  notifications: NotificationSettings
  scanner: ScannerSettings
  trading: TradingSettingsConfig
  maintenance: MaintenanceSettings
  updated_at: string | null
}

export interface UpdateSettingsRequest {
  polymarket?: Partial<PolymarketSettings>
  llm?: Partial<LLMSettings>
  notifications?: Partial<NotificationSettings>
  scanner?: Partial<ScannerSettings>
  trading?: Partial<TradingSettingsConfig>
  maintenance?: Partial<MaintenanceSettings>
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

// ==================== AI INTELLIGENCE ====================

export const getAIStatus = () => api.get('/ai/status')
export const analyzeResolution = (data: any) => api.post('/ai/resolution/analyze', data)
export const getResolutionAnalysis = (marketId: string) => api.get(`/ai/resolution/${marketId}`)
export const judgeOpportunity = (data: any) => api.post('/ai/judge/opportunity', data)
export const getJudgmentHistory = (params?: any) => api.get('/ai/judge/history', { params })
export const getAgreementStats = () => api.get('/ai/judge/agreement-stats')
export const analyzeMarket = (data: any) => api.post('/ai/market/analyze', data)
export const analyzeNewsSentiment = (data: any) => api.post('/ai/news/sentiment', data)
export const listSkills = () => api.get('/ai/skills')
export const executeSkill = (data: any) => api.post('/ai/skills/execute', data)
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
  response: string
  model: string
  tokens_used: Record<string, number>
}

export const sendAIChat = async (params: {
  message: string
  context_type?: string
  context_id?: string
  history?: AIChatMessage[]
}): Promise<AIChatResponse> => {
  const { data } = await api.post('/ai/chat', params)
  return data
}

export default api
