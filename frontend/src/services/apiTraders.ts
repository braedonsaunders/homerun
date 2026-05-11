import { api, getStrategyManagerItems, unwrapApiData, unwrapStrategyManagerPayload } from './apiClient'
import type { TradingPosition } from './apiCore'

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
  const { data } = await api.get(`/discovery/wallet/${address}/win-rate`, {
    params: timePeriod ? { time_period: timePeriod } : undefined
  })
  return unwrapApiData(data)
}

export const analyzeWalletPnL = async (address: string, timePeriod?: TimePeriod): Promise<WalletPnL> => {
  const { data } = await api.get(`/discovery/wallet/${address}/pnl`, {
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

export const analyzeAndTrackWallet = async (params: {
  address: string
  label?: string
}) => {
  const { data } = await api.post(
    `/discovery/wallet/${params.address}/track`,
    params.label ? { label: params.label } : {}
  )
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

/** Snapshot of the trader-orchestrator process — the worker that
 *  actually ticks live traders.  ``running=true`` means the
 *  orchestrator loop is alive and ``traders_running`` reflects how
 *  many traders it's actively processing this tick.  Distinguishes
 *  "trader configured live in DB but orchestrator process down" from
 *  "trader is actually tickling Polymarket right now". */
export interface OrchestratorSnapshot {
  id?: string
  updated_at?: string | null
  last_run_at?: string | null
  running: boolean
  enabled: boolean
  current_activity?: string | null
  interval_seconds?: number
  traders_total: number
  traders_running: number
  decisions_count: number
  orders_count: number
  open_orders: number
  gross_exposure_usd: number
  daily_pnl: number
  last_error?: string | null
  stats?: Record<string, unknown>
}

export interface OrchestratorStatusResponse {
  control: {
    is_enabled: boolean
    is_paused: boolean
    kill_switch: boolean
    mode?: string | null
    [k: string]: unknown
  }
  snapshot: OrchestratorSnapshot
  config?: Record<string, unknown>
}

/** Pulls /trader-orchestrator/status — ``snapshot.running`` tells
 *  callers whether the worker process is actually alive (vs. just
 *  having traders configured in the DB). */
export const getOrchestratorStatus = async (): Promise<OrchestratorStatusResponse> => {
  const { data } = await api.get<OrchestratorStatusResponse>('/trader-orchestrator/status')
  return data
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
      enforce_allow_averaging_off?: boolean | null
      min_cooldown_seconds?: number | null
      max_consecutive_losses_cap?: number | null
      max_open_orders_cap?: number | null
      max_open_positions_cap?: number | null
      max_trade_notional_usd_cap?: number | null
      max_orders_per_cycle_cap?: number | null
      enforce_halt_on_consecutive_losses?: boolean | null
      max_daily_spend_usd_cap?: number | null
      max_spread_bps_cap?: number | null
      slippage_bps_cap?: number | null
      retry_limit_cap?: number | null
      retry_backoff_ms_cap?: number | null
      order_ttl_seconds_cap?: number | null
    }
    live_market_context: {
      enabled: boolean
      history_window_seconds: number
      history_fidelity_seconds: number
      max_history_points: number
      timeout_seconds: number
      strict_ws_pricing_only: boolean
      max_market_data_age_ms: number
    }
    live_provider_health: {
      window_seconds: number
      min_errors: number
      block_seconds: number
    }
    trader_cycle_timeout_seconds: number | null
    runtime_trigger_cycle_timeout_seconds: number | null
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
  submit_round_trip_ms: ExecutionLatencyPercentiles
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
  selected_account_id: string
  requested_by?: string
}): Promise<TraderOrchestratorControlResponse> => {
  const { data } = await api.post('/trader-orchestrator/start', {
    mode: payload?.mode || 'shadow',
    selected_account_id: payload?.selected_account_id,
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
  selected_account_id: string
  requested_by?: string
}) => {
  const { data } = await api.post('/trader-orchestrator/live/start', {
    arm_token: payload.arm_token,
    mode: payload.mode || 'live',
    selected_account_id: payload.selected_account_id,
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

export type TraderLatencyClass = 'fast' | 'normal' | 'slow'

export interface Trader {
  id: string
  name: string
  description?: string | null
  mode: 'shadow' | 'live'
  latency_class: TraderLatencyClass
  source_configs: TraderSourceConfig[]
  risk_limits: Record<string, any>
  metadata: Record<string, any>
  is_enabled: boolean
  is_paused: boolean
  block_new_orders?: boolean
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
  execution_wallet_address: string | null
  provider_order_id: string
  provider_clob_order_id: string
  verification_status: string
  verification_source: string | null
  verification_reason: string | null
  verification_tx_hash: string | null
  verification_disputed: boolean
  verified_at: string | null
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
  trade_bundle?: TraderOrderTradeBundle | null
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

export interface TraderOrderTradeBundleLeg {
  leg_index: number
  leg_id: string | null
  market_id: string | null
  market_question: string | null
  token_id: string | null
  side: string | null
  outcome: string | null
  limit_price: number | null
  notional_weight: number | null
  condition_id: string | null
}

export interface TraderOrderTradeBundle {
  bundle_id: string
  plan_id: string | null
  kind: string
  label: string
  leg_count: number
  is_guaranteed: boolean
  signal_is_guaranteed: boolean
  guarantee_proven: boolean
  guarantee_reason: string | null
  roi_type: string | null
  mispricing_type: string | null
  total_cost: number | null
  expected_payout: number | null
  gross_profit: number | null
  net_profit: number | null
  roi_percent: number | null
  planned_market_count: number
  market_roster_count: number | null
  current_leg_id: string | null
  current_leg_index: number | null
  current_leg_token_id: string | null
  legs: TraderOrderTradeBundleLeg[]
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
  redeemable: boolean
  counts_as_open: boolean
  end_date: string | null
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
    open_positions: number
    managed_positions: number
    managed_open_positions: number
    unmanaged_positions: number
    unmanaged_open_positions: number
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
  /**
   * Firehose volume tier — orthogonal to severity.  Only set for
   * events emitted by the strategy firehose; null/undefined for the
   * existing trader-orchestrator event stream.
   *
   * Tiers (lowest → loudest): whisper < murmur < voice < shout.
   * The Terminal UI's volume dial shows everything at-or-louder than
   * the selected tier, plus all warnings/errors regardless.
   */
  verbosity?: 'whisper' | 'murmur' | 'voice' | 'shout' | null
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
  crypto: 'btc_eth_maker_quote',
  manual: 'manual_wallet_position',
  news: 'news_edge',
  weather: 'weather_distribution',
  traders: 'traders_confluence',
}

const DEFAULT_STRATEGY_KEY = 'btc_eth_maker_quote'

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

export interface ReconcileOrderResponse {
  status: string
  trader_id: string
  order_id: string
  before: { notional_usd: number; effective_price: number }
  after: { notional_usd: number; effective_price: number }
  polymarket: { size: number; avg_price: number; current_value: number }
}

export const reconcileTraderOrder = async (
  traderId: string,
  orderId: string,
  payload?: { requested_by?: string }
): Promise<ReconcileOrderResponse> => {
  const { data } = await api.post(`/traders/${traderId}/orders/${orderId}/reconcile`, payload || {})
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

export const setTraderBlockNewOrders = async (
  traderId: string,
  enabled: boolean,
  payload?: { requested_by?: string; reason?: string },
): Promise<Trader> => {
  const { data } = await api.post(`/traders/${traderId}/block-new-orders`, {
    enabled,
    ...(payload || {}),
  })
  return normalizeTraderFields(data)
}

export type TraderDeleteAction = 'block' | 'disable' | 'force_delete' | 'transfer_delete'

export const deleteTrader = async (
  traderId: string,
  options?: { action?: TraderDeleteAction; transfer_to_trader_id?: string }
): Promise<{
  status: string
  trader_id: string
  action?: TraderDeleteAction
  transfer_to_trader_id?: string
  orders_transferred?: number
  positions_transferred?: number
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

export interface TraderManualBuyPosition {
  token_id: string
  side: string
  price: number
  market_id?: string
  market_question?: string
  outcome?: string
}

export interface TraderManualBuyResponse {
  status: string
  trader_id: string
  mode: string
  orders: Array<{
    order_id: string
    market_id: string
    market_question: string
    direction: string
    status: string
    notional_usd: number
    entry_price: number
    size_shares: number
    error: string | null
  }>
  message: string
}

export const traderManualBuy = async (
  traderId: string,
  params: {
    positions: TraderManualBuyPosition[]
    size_usd: number
    opportunity_id?: string
  }
): Promise<TraderManualBuyResponse> => {
  const { data } = await api.post(`/traders/${traderId}/manual-buy`, params)
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
  params?: { status?: string; mode?: string; limit?: number }
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

export const getAllTraderOrders = async (limit = 500, offset = 0): Promise<TraderOrder[]> => {
  const { data } = await api.get('/traders/orders/all', {
    params: {
      limit: Math.max(1, Math.trunc(Number(limit) || 500)),
      offset: Math.max(0, Math.trunc(Number(offset) || 0)),
    },
  })
  return data.orders || []
}

export interface TraderOrdersSummary {
  total_count: number
  open: number
  resolved: number
  failed: number
  total_trades: number
  open_trades: number
  resolved_trades: number
  failed_trades: number
  partial_open_bundles: number
  resolved_pnl: number
  wins: number
  losses: number
  total_notional: number
  win_rate: number
  avg_edge: number
  avg_confidence: number
  by_trader: Array<{
    trader_id: string
    orders: number
    open: number
    resolved: number
    trade_count: number
    open_trades: number
    resolved_trades: number
    failed_trades: number
    partial_open_bundles: number
    pnl: number
    notional: number
    wins: number
    losses: number
    latest_activity_ts: string | null
  }>
  by_source: Array<{
    source: string
    orders: number
    resolved: number
    pnl: number
    notional: number
    wins: number
    losses: number
  }>
}

export const getTraderOrdersSummary = async (mode?: string): Promise<TraderOrdersSummary> => {
  const { data } = await api.get('/traders/orders/summary', {
    params: mode ? { mode } : undefined,
  })
  return data
}

export const getAllTraderEventsBulk = async (
  traderIds: string[],
  params?: { limit?: number; types?: string[] }
): Promise<TraderEvent[]> => {
  const normalizedIds = Array.from(new Set(traderIds.map((id) => String(id).trim()).filter(Boolean)))
  if (normalizedIds.length === 0) return []
  const { data } = await api.get('/traders/events/all', {
    params: {
      trader_ids: normalizedIds.join(','),
      limit: params?.limit ?? 500,
      types: params?.types?.join(',') || undefined,
    },
  })
  return data.events || []
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
  const { data } = await api.post('/maintenance/vacuum', null, { params: { full }, timeout: 300_000 })
  return unwrapApiData(data)
}

export const runReindexTables = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/maintenance/reindex', null, { timeout: 300_000 })
  return unwrapApiData(data)
}

export const setWorkerInterval = async (worker: string, intervalSeconds: number) => {
  const { data } = await api.post(`/workers/${worker}/interval`, null, {
    params: { interval_seconds: intervalSeconds },
  })
  return unwrapApiData(data)
}
