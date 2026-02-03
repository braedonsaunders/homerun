import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

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

// Opportunities
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

// Scanner
export const getScannerStatus = async (): Promise<ScannerStatus> => {
  const { data } = await api.get('/scanner/status')
  return data
}

export const getStrategies = async (): Promise<Strategy[]> => {
  const { data } = await api.get('/strategies')
  return data
}

// Wallets
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

// Markets
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

export default api
