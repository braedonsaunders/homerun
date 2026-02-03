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

// ==================== OPPORTUNITIES ====================

export const getOpportunities = async (params?: {
  min_profit?: number
  max_risk?: number
  strategy?: string
  min_liquidity?: number
  limit?: number
}): Promise<Opportunity[]> => {
  const { data } = await api.get('/opportunities', { params })
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

export const getAccountPositions = async (accountId: string) => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/positions`)
  return data
}

export const getAccountTrades = async (accountId: string, limit = 50): Promise<SimulationTrade[]> => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/trades`, { params: { limit } })
  return data
}

export const executeOpportunity = async (accountId: string, opportunityId: string, positionSize?: number) => {
  const { data } = await api.post(`/simulation/accounts/${accountId}/execute`, {
    opportunity_id: opportunityId,
    position_size: positionSize
  })
  return data
}

export const getAccountPerformance = async (accountId: string) => {
  const { data } = await api.get(`/simulation/accounts/${accountId}/performance`)
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

// ==================== HEALTH ====================

export const getHealthStatus = async () => {
  const { data } = await api.get('/health/detailed')
  return data
}

export default api
