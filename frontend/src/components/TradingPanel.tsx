import { Fragment, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom, useAtomValue } from 'jotai'
import { Liveline } from 'liveline'
import type { LivelinePoint, LivelineSeries } from 'liveline'
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  ExternalLink,
  Loader2,
  PieChart,
  Play,
  Plus,
  Settings,
  ShieldAlert,
  Sparkles,
  Square,
  Trophy,
  TrendingUp,
  Zap,
} from 'lucide-react'
import {
  activateTrader,
  armTraderOrchestratorLiveStart,
  createTrader,
  deactivateTrader,
  deleteTrader,
  getAllTraderDecisions,
  getAllTraderOrders,
  getSettings,
  getTraderMarketHistory,
  getCryptoMarkets,
  type CryptoMarket,
  getTraderDecisionDetail,
  getTraderEvents,
  getTraderConfigSchema,
  getTraderOrchestratorOverview,
  getTraderSources,
  getSimulationAccounts,
  getWallets,
  getTraders,
  runTraderOnce,
  runTraderOrchestratorLivePreflight,
  setTraderOrchestratorLiveKillSwitch,
  startTrader,
  startTraderOrchestrator,
  startTraderOrchestratorLive,
  stopTrader,
  stopTraderOrchestrator,
  stopTraderOrchestratorLive,
  runTraderTuneIteration,
  sellTraderOrderNow,
  type ExecutionLatencySummary,
  type Trader,
  type TraderConfigSchema,
  type TraderEvent,
  type TraderOrder,
  type TraderStopPayload,
  type TraderStopLifecycleMode,
  type TraderSourceConfig,
  type TraderSource,
  type TraderTuneAgentResponse,
  updateSettings,
  updateTrader,
  type TraderOrchestratorConfig,
  updateTraderOrchestratorSettings,
} from '../services/api'
import { discoveryApi } from '../services/discoveryApi'
import { cn } from '../lib/utils'
import { getTraderOrderPlatformLinks } from '../lib/marketUrls'
import { selectedAccountIdAtom, themeAtom } from '../store/atoms'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { Switch } from './ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import { FlashNumber } from './AnimatedNumber'
import { toTimeValueSeries } from '../lib/priceHistory'
import StrategyConfigForm from './StrategyConfigForm'

type FeedFilter = 'all' | 'decision' | 'order' | 'event'
type TradeStatusFilter = 'all' | 'open_resolved' | 'open' | 'resolved' | 'failed'
type DecisionOutcomeFilter = 'all' | 'selected' | 'blocked' | 'skipped'
type AllBotsTab = 'overview' | 'trades' | 'positions'
type TradeAction = 'BUY' | 'SELL'
type DirectionSide = 'YES' | 'NO'
type PositionDirectionFilter = 'all' | 'yes' | 'no'
type PositionSortField = 'exposure' | 'updated' | 'edge' | 'confidence' | 'unrealized'
type PositionSortDirection = 'asc' | 'desc'
type BotRosterSort = 'name_asc' | 'name_desc' | 'pnl_desc' | 'pnl_asc' | 'open_desc' | 'activity_desc'
type BotRosterGroupBy = 'none' | 'status' | 'source'
type TerminalDensity = 'compact' | 'expanded'
type TraderToggleAction = 'start' | 'stop' | 'activate' | 'deactivate'
type ExecutionLatencyStageKey =
  | 'armed_to_ws_release_ms'
  | 'emit_to_queue_wake_ms'
  | 'ws_release_to_decision_ms'
  | 'ws_release_to_submit_start_ms'
  | 'wake_to_context_ready_ms'
  | 'context_ready_to_decision_ms'
  | 'decision_to_submit_start_ms'
  | 'submit_start_to_provider_ack_ms'
  | 'emit_to_submit_start_ms'

const TRADE_STATUS_FILTER_OPTIONS: Array<{ value: TradeStatusFilter; label: string }> = [
  { value: 'all', label: 'all' },
  { value: 'open_resolved', label: 'open+resolved' },
  { value: 'open', label: 'open' },
  { value: 'resolved', label: 'resolved' },
  { value: 'failed', label: 'failed' },
]

type TerminalLeg = {
  action: TradeAction | null
  outcome: 'YES' | 'NO' | null
  marketId: string | null
  marketQuestion: string | null
  price: number | null
}

type ActivityRow = {
  kind: 'decision' | 'order' | 'event'
  id: string
  ts: string | null
  traderId: string | null
  title: string
  detail: string
  action: TradeAction | null
  tone: 'neutral' | 'positive' | 'negative' | 'warning'
}

type OverviewTrendBucket = {
  key: string
  label: string
  orders: number
  selected: number
  resolvedPnl: number
  failed: number
  warnings: number
  cumulativeResolvedPnl: number
}

type PositionBookRow = {
  key: string
  traderId: string
  traderName: string
  marketId: string
  marketAliases: string[]
  marketQuestion: string
  sourceSummary: string
  executionSummary: string
  direction: string
  directionSide: DirectionSide | null
  exposureUsd: number
  averagePrice: number | null
  markPrice: number | null
  markUpdatedAt: string | null
  markFresh: boolean
  unrealizedPnl: number | null
  weightedEdge: number | null
  weightedConfidence: number | null
  orderCount: number
  liveOrderCount: number
  shadowOrderCount: number
  lastUpdated: string | null
  statusSummary: string
  links: {
    polymarket: string | null
    kalshi: string | null
  }
}

type BotMarketModalKind = 'trade' | 'position'

type BotMarketModalScope = {
  kind: BotMarketModalKind
  traderId: string | null
  traderName: string
  marketId: string
  marketIds: string[]
  marketQuestion: string
  directionSide: DirectionSide | null
  directionLabel: string
  yesLabel: string | null
  noLabel: string | null
  anchorOrderId: string | null
  sourceSummary: string
  statusSummary: string
  modeSummary: string
  executionSummary: string
  outcomeSummary: string | null
  links: {
    polymarket: string | null
    kalshi: string | null
  }
}

type BotMarketModalState = {
  market: CryptoMarket | null
  scope: BotMarketModalScope
}

type TraderRuntimeStatus = 'running' | 'engine_stopped' | 'bot_stopped' | 'inactive'

type TraderStatusPresentation = {
  key: TraderRuntimeStatus
  label: string
  dotClassName: string
  badgeVariant: 'default' | 'secondary' | 'outline'
  badgeClassName: string
}

type PerformanceBucketRow = {
  key: string
  label: string
  orders: number
  open: number
  resolved: number
  wins: number
  losses: number
  failed: number
  resolvedNotional: number
  pnl: number
  roiPercent: number
  fullLosses: number
}

type PerformanceSubview = 'latency' | 'configuration'

type PerformanceSection = {
  sectionKey: string
  sectionLabel: string
  sourceKey: string
  sourceLabel: string
  strategyKey: string
  strategyLabel: string
  strategyVersion: number | null
  strategyVersionLabel: string
  groups: StrategyParamGroup[]
  fieldKeys: string[]
  paramFields: Array<Record<string, unknown>>
  values: Record<string, unknown>
}

type PerformanceOrderSnapshot = {
  order: TraderOrder
  sourceKey: string
  sourceLabel: string
  strategyKey: string
  strategyLabel: string
  strategyVersion: number | null
  strategyVersionLabel: string
  sectionKey: string
  sectionLabel: string
  params: Record<string, unknown>
  usedCurrentConfigFallback: boolean
}

type PerformanceConfigurationRow = PerformanceBucketRow & {
  sectionKey: string
  sectionLabel: string
  sourceLabel: string
  strategyLabel: string
  strategyVersionLabel: string
}

type PerformanceParamSummaryRow = {
  key: string
  label: string
  currentValueLabel: string
  observedValueCount: number
  currentResolved: number
  currentPnl: number
  currentRoiPercent: number
  hasVariation: boolean
}

type PerformanceParamValueRow = PerformanceBucketRow & {
  valueLabel: string
  isCurrent: boolean
  isMissing: boolean
}

type LatencyStageRow = {
  key: ExecutionLatencyStageKey
  label: string
  traderLatencyLabel: string
  overallLatencyLabel: string
}

type LatencyGroupRow = {
  key: string
  label: string
  count: number
  latencyLabel: string
}

type StrategyParamGroupKey = 'scope' | 'timing' | 'entry' | 'sizing' | 'exit' | 'risk' | 'advanced'

type StrategyParamGroup = {
  key: StrategyParamGroupKey
  label: string
  fields: Array<Record<string, unknown>>
}

type DynamicStrategyParamSection = {
  sectionKey: string
  sourceLabel: string
  strategyLabel: string
  groups: StrategyParamGroup[]
  fieldKeys: string[]
  values: Record<string, unknown>
}

type TuneRevertSnapshot = {
  traderId: string
  sourceConfigs: TraderSourceConfig[]
  capturedAt: string
}

const TERMINAL_ACTIVITY_MAX_ROWS = 320
const TERMINAL_SELECTED_MAX_ROWS = 220
const TERMINAL_ALL_BOTS_MAX_ROWS = 120
const TERMINAL_COMPACT_ROW_HEIGHT = 34
const TERMINAL_COMPACT_OVERSCAN = 16

const CRYPTO_STRATEGY_OPTIONS = [
  { key: 'btc_eth_highfreq', label: 'Crypto High Frequency' },
  { key: 'crypto_spike_reversion', label: 'Crypto Spike Reversion' },
] as const
const LATENCY_STAGE_OPTIONS: Array<{ key: ExecutionLatencyStageKey; label: string }> = [
  { key: 'armed_to_ws_release_ms', label: 'Armed -> WS Release' },
  { key: 'emit_to_queue_wake_ms', label: 'Emit -> Queue Wake' },
  { key: 'wake_to_context_ready_ms', label: 'Wake -> Context Ready' },
  { key: 'context_ready_to_decision_ms', label: 'Context Ready -> Decision' },
  { key: 'ws_release_to_decision_ms', label: 'WS Release -> Decision' },
  { key: 'decision_to_submit_start_ms', label: 'Decision -> Submit Start' },
  { key: 'submit_start_to_provider_ack_ms', label: 'Submit Start -> Provider Ack' },
  { key: 'ws_release_to_submit_start_ms', label: 'WS Release -> Submit Start' },
  { key: 'emit_to_submit_start_ms', label: 'Emit -> Submit Start' },
]
const PERFORMANCE_TIMEFRAME_ORDER: Record<string, number> = { '5m': 0, '15m': 1, '1h': 2, '4h': 3 }
const PERFORMANCE_MODE_ORDER: Record<string, number> = {
  auto: 0,
  directional: 1,
  pure_arb: 2,
  rebalance: 3,
  dump_hedge: 4,
  pre_placed_limits: 5,
  directional_edge: 6,
}
const STRATEGY_PARAM_GROUP_ORDER = [
  'scope',
  'timing',
  'entry',
  'sizing',
  'exit',
  'risk',
  'advanced',
] as const
const STRATEGY_PARAM_GROUP_LABELS: Record<StrategyParamGroupKey, string> = {
  scope: 'Scope & Modes',
  timing: 'Timing & Freshness',
  entry: 'Entry Filters',
  sizing: 'Sizing',
  exit: 'Exit Controls',
  risk: 'Risk Guards',
  advanced: 'Advanced',
}
type TradersScopeMode = 'tracked' | 'pool' | 'individual' | 'group'
type TradingScheduleDay = 'mon' | 'tue' | 'wed' | 'thu' | 'fri' | 'sat' | 'sun'

type TradingScheduleDraft = {
  enabled: boolean
  days: TradingScheduleDay[]
  startTimeUtc: string
  endTimeUtc: string
  startDateUtc: string
  endDateUtc: string
  endAtUtc: string
}

type GlobalSettingsDraft = {
  runIntervalSeconds: string
  maxGrossExposureUsd: string
  maxDailyLossUsd: string
  maxOrdersPerCycle: string
  maxTradeSizeUsd: string
  maxDailyTradeVolumeUsd: string
  minAccountBalanceUsd: string
  maxOpenPositions: string
  maxSlippagePercent: string
  pendingExitMaxAllowed: string
  pendingExitIdentityGuardEnabled: boolean
  pendingExitTerminalStatuses: string
  enforceAllowAveragingOff: boolean
  minCooldownSeconds: string
  maxConsecutiveLossesCap: string
  maxOpenOrdersCap: string
  maxTradeNotionalUsdCap: string
  maxOrdersPerCycleCap: string
  enforceHaltOnConsecutiveLosses: boolean
  liveMarketContextEnabled: boolean
  liveMarketHistoryWindowSeconds: string
  liveMarketHistoryFidelitySeconds: string
  liveMarketHistoryMaxPoints: string
  liveMarketContextTimeoutSeconds: string
  liveProviderHealthWindowSeconds: string
  liveProviderHealthMinErrors: string
  liveProviderHealthBlockSeconds: string
  traderCycleTimeoutSeconds: string
}

const DEFAULT_ORCHESTRATOR_GLOBAL_RISK = {
  max_gross_exposure_usd: 5000,
  max_daily_loss_usd: 500,
  max_orders_per_cycle: 50,
} as const
const DEFAULT_LIVE_EXECUTION_LIMITS = {
  max_trade_size_usd: 100,
  max_daily_trade_volume: 1000,
  max_slippage_percent: 2,
  min_account_balance_usd: 0,
} as const
const DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME = {
  pending_live_exit_guard: {
    max_pending_exits: 0,
    identity_guard_enabled: true,
    terminal_statuses: ['filled', 'superseded_resolution', 'superseded_external', 'cancelled'],
  },
  live_risk_clamps: {
    enforce_allow_averaging_off: true,
    min_cooldown_seconds: 90,
    max_consecutive_losses_cap: 3,
    max_open_orders_cap: 6,
    max_open_positions_cap: 4,
    max_trade_notional_usd_cap: 200,
    max_orders_per_cycle_cap: 4,
    enforce_halt_on_consecutive_losses: true,
  },
  live_market_context: {
    enabled: true,
    history_window_seconds: 7200,
    history_fidelity_seconds: 300,
    max_history_points: 120,
    timeout_seconds: 4,
  },
  live_provider_health: {
    window_seconds: 180,
    min_errors: 2,
    block_seconds: 120,
  },
  trader_cycle_timeout_seconds: null as number | null,
} as const
const TRADING_SCHEDULE_DAYS: TradingScheduleDay[] = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
const TRADING_SCHEDULE_WEEKDAYS: TradingScheduleDay[] = ['mon', 'tue', 'wed', 'thu', 'fri']
const TRADING_SCHEDULE_WEEKENDS: TradingScheduleDay[] = ['sat', 'sun']
const TRADING_SCHEDULE_DAY_LABEL: Record<TradingScheduleDay, string> = {
  mon: 'Mon',
  tue: 'Tue',
  wed: 'Wed',
  thu: 'Thu',
  fri: 'Fri',
  sat: 'Sat',
  sun: 'Sun',
}

const OPEN_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])
const RESOLVED_ORDER_STATUSES = new Set([
  'resolved',
  'resolved_win',
  'resolved_loss',
  'closed_win',
  'closed_loss',
  'win',
  'loss',
])
const FAILED_ORDER_STATUSES = new Set(['failed', 'rejected', 'error', 'cancelled'])

const FALLBACK_TRADER_SOURCES: TraderSource[] = [
  {
    key: 'crypto',
    label: 'Crypto Markets',
    description: 'Crypto microstructure signals.',
    domains: ['crypto'],
    signal_types: ['crypto_market'],
    strategy_options: CRYPTO_STRATEGY_OPTIONS.map((item) => ({
      key: item.key,
      label: item.label,
      description: `${item.label} strategy`,
      default_params: {},
      param_fields: [],
    })),
  },
  {
    key: 'manual',
    label: 'Manual Positions',
    description: 'Manually adopted live positions managed without new entries.',
    domains: ['event_markets'],
    signal_types: ['manual_position'],
    strategy_options: [
      {
        key: 'manual_wallet_position',
        label: 'Manual Manage Hold',
        description: '',
        default_params: {},
        param_fields: [],
      },
    ],
  },
  {
    key: 'news',
    label: 'News Workflow',
    description: 'News-driven intents and event reactions.',
    domains: ['event_markets'],
    signal_types: ['news_intent'],
    strategy_options: [{ key: 'news_edge', label: 'News Edge', description: '', default_params: {}, param_fields: [] }],
  },
  {
    key: 'scanner',
    label: 'General Opportunities',
    description: 'Scanner-originated arbitrage opportunities.',
    domains: ['event_markets'],
    signal_types: ['opportunity'],
    strategy_options: [{ key: 'basic', label: 'Opportunity General', description: '', default_params: {}, param_fields: [] }],
  },
  {
    key: 'traders',
    label: 'Traders',
    description: 'Tracked/pool/individual/group trader activity signals.',
    domains: ['event_markets'],
    signal_types: ['confluence'],
    strategy_options: [{ key: 'traders_confluence', label: 'Traders Confluence', description: '', default_params: {}, param_fields: [] }],
  },
  {
    key: 'weather',
    label: 'Weather Workflow',
    description: 'Weather forecast probability dislocations.',
    domains: ['event_markets'],
    signal_types: ['weather_intent'],
    strategy_options: [{ key: 'weather_distribution', label: 'Weather Distribution', description: '', default_params: {}, param_fields: [] }],
  },
]

const STRATEGY_LABELS: Record<string, string> = {
  basic: 'Opportunity General',
  btc_eth_highfreq: 'Crypto High-Frequency',
  crypto_spike_reversion: 'Crypto Spike Reversion',
  manual_wallet_position: 'Manual Manage Hold',
  news_edge: 'News Reaction',
  weather_distribution: 'Weather Distribution',
  traders_confluence: 'Traders Confluence',
  flash_crash_reversion: 'Opportunity Flash Reversion',
  tail_end_carry: 'Opportunity Tail Carry',
}

const DEFAULT_STRATEGY_KEY = 'btc_eth_highfreq'
const DEFAULT_STRATEGY_BY_SOURCE: Record<string, string> = {
  crypto: 'btc_eth_highfreq',
  manual: 'manual_wallet_position',
  scanner: 'basic',
  news: 'news_edge',
  weather: 'weather_distribution',
  traders: 'traders_confluence',
}

type StrategyOption = {
  key: string
  label: string
}

type StrategyOptionDetail = {
  key: string
  label: string
  defaultParams: Record<string, unknown>
  paramFields: Array<Record<string, unknown>>
  version: number | null
  latestVersion: number | null
  versions: number[]
}

const STABLE_OUTCOME_LABELS_BY_MARKET_SIDE = new Map<string, string>()


function toNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function clampNumber(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback
  return Math.max(min, Math.min(max, value))
}

function normalizePendingExitTerminalStatusesCsv(value: string): string[] {
  const seen = new Set<string>()
  const rows = value
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean)
  const out: string[] = []
  for (const status of rows) {
    if (seen.has(status)) continue
    seen.add(status)
    out.push(status)
  }
  return out.length > 0
    ? out
    : [...DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard.terminal_statuses]
}

function buildGlobalSettingsDraft(
  config: TraderOrchestratorConfig | null | undefined,
  liveExecutionSettings: {
    max_trade_size_usd?: number | null
    max_daily_trade_volume?: number | null
    max_slippage_percent?: number | null
    min_account_balance_usd?: number | null
  } | null | undefined,
): GlobalSettingsDraft {
  const globalRisk = config?.global_risk || DEFAULT_ORCHESTRATOR_GLOBAL_RISK
  const runtime = config?.global_runtime || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME
  const pending = runtime.pending_live_exit_guard || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard
  const clamps = runtime.live_risk_clamps || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps
  const marketContext = runtime.live_market_context || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context
  const providerHealth = runtime.live_provider_health || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health
  const pendingTerminalStatuses = toStringList(pending.terminal_statuses)
  const maxOpenPositions = Math.trunc(
    clampNumber(
      toNumber(clamps.max_open_positions_cap ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_positions_cap),
      1,
      1000,
      DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_positions_cap,
    )
  )
  return {
    runIntervalSeconds: String(config?.run_interval_seconds ?? 5),
    maxGrossExposureUsd: String(globalRisk.max_gross_exposure_usd ?? DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_gross_exposure_usd),
    maxDailyLossUsd: String(globalRisk.max_daily_loss_usd ?? DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_daily_loss_usd),
    maxOrdersPerCycle: String(globalRisk.max_orders_per_cycle ?? DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_orders_per_cycle),
    maxTradeSizeUsd: String(liveExecutionSettings?.max_trade_size_usd ?? DEFAULT_LIVE_EXECUTION_LIMITS.max_trade_size_usd),
    maxDailyTradeVolumeUsd: String(
      liveExecutionSettings?.max_daily_trade_volume ?? DEFAULT_LIVE_EXECUTION_LIMITS.max_daily_trade_volume
    ),
    minAccountBalanceUsd: String(
      liveExecutionSettings?.min_account_balance_usd ?? DEFAULT_LIVE_EXECUTION_LIMITS.min_account_balance_usd
    ),
    maxOpenPositions: String(maxOpenPositions),
    maxSlippagePercent: String(
      liveExecutionSettings?.max_slippage_percent ?? DEFAULT_LIVE_EXECUTION_LIMITS.max_slippage_percent
    ),
    pendingExitMaxAllowed: String(pending.max_pending_exits ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard.max_pending_exits),
    pendingExitIdentityGuardEnabled: Boolean(
      pending.identity_guard_enabled ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard.identity_guard_enabled
    ),
    pendingExitTerminalStatuses: (
      pendingTerminalStatuses.length > 0
        ? pendingTerminalStatuses
        : [...DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard.terminal_statuses]
    ).join(', '),
    enforceAllowAveragingOff: Boolean(
      clamps.enforce_allow_averaging_off ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.enforce_allow_averaging_off
    ),
    minCooldownSeconds: String(clamps.min_cooldown_seconds ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.min_cooldown_seconds),
    maxConsecutiveLossesCap: String(
      clamps.max_consecutive_losses_cap ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_consecutive_losses_cap
    ),
    maxOpenOrdersCap: String(clamps.max_open_orders_cap ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_orders_cap),
    maxTradeNotionalUsdCap: String(
      clamps.max_trade_notional_usd_cap ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_trade_notional_usd_cap
    ),
    maxOrdersPerCycleCap: String(
      clamps.max_orders_per_cycle_cap ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_orders_per_cycle_cap
    ),
    enforceHaltOnConsecutiveLosses: Boolean(
      clamps.enforce_halt_on_consecutive_losses
      ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.enforce_halt_on_consecutive_losses
    ),
    liveMarketContextEnabled: Boolean(
      marketContext.enabled ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.enabled
    ),
    liveMarketHistoryWindowSeconds: String(
      marketContext.history_window_seconds
      ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.history_window_seconds
    ),
    liveMarketHistoryFidelitySeconds: String(
      marketContext.history_fidelity_seconds
      ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.history_fidelity_seconds
    ),
    liveMarketHistoryMaxPoints: String(
      marketContext.max_history_points ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.max_history_points
    ),
    liveMarketContextTimeoutSeconds: String(
      marketContext.timeout_seconds ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.timeout_seconds
    ),
    liveProviderHealthWindowSeconds: String(
      providerHealth.window_seconds ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health.window_seconds
    ),
    liveProviderHealthMinErrors: String(
      providerHealth.min_errors ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health.min_errors
    ),
    liveProviderHealthBlockSeconds: String(
      providerHealth.block_seconds ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health.block_seconds
    ),
    traderCycleTimeoutSeconds: runtime.trader_cycle_timeout_seconds === null
      ? ''
      : String(runtime.trader_cycle_timeout_seconds),
  }
}

function toStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean)
  }
  if (typeof value === 'string') {
    return csvToList(value)
  }
  return []
}

function normalizeCryptoTimeframe(value: unknown): string | null {
  const tf = String(value || '').trim().toLowerCase()
  if (!tf) return null
  if (tf === '5m' || tf === '5min' || tf === '5') return '5m'
  if (tf === '15m' || tf === '15min' || tf === '15') return '15m'
  if (tf === '1h' || tf === '1hr' || tf === '60m' || tf === '60min') return '1h'
  if (tf === '4h' || tf === '4hr' || tf === '240m' || tf === '240min') return '4h'
  return null
}

function normalizeStatus(value: string | null | undefined): string {
  return String(value || 'unknown').trim().toLowerCase()
}

function toTs(value: string | null | undefined): number {
  if (!value) return 0
  const ts = new Date(value).getTime()
  return Number.isFinite(ts) ? ts : 0
}

function latestTimestampValue(...values: Array<string | null | undefined>): string {
  let bestValue = ''
  let bestTs = 0
  for (const rawValue of values) {
    const value = String(rawValue || '').trim()
    if (!value) continue
    const ts = toTs(value)
    if (ts > bestTs) {
      bestTs = ts
      bestValue = value
    }
  }
  if (bestValue) return bestValue
  for (const rawValue of values) {
    const value = String(rawValue || '').trim()
    if (value) return value
  }
  return ''
}

function utcDayKeyFromTs(ts: number): string | null {
  if (!(ts > 0)) return null
  return new Date(ts).toISOString().slice(0, 10)
}

function formatDayKeyLabel(dayKey: string): string {
  const ts = Date.parse(`${dayKey}T00:00:00Z`)
  if (!Number.isFinite(ts)) return dayKey
  return new Date(ts).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  })
}

function formatCurrency(value: number, compact = false): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 1 : 2,
  }).format(value)
}

function formatPercent(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`
}

function formatSignedCurrency(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—'
  const magnitude = formatCurrency(Math.abs(value))
  if (value > 0) return `+${magnitude}`
  if (value < 0) return `-${magnitude}`
  return magnitude
}

function formatSignedPercent(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return '—'
  const magnitude = `${Math.abs(value).toFixed(digits)}%`
  if (value > 0) return `+${magnitude}`
  if (value < 0) return `-${magnitude}`
  return magnitude
}

function toUnixSeconds(value: number): number {
  if (value > 1_000_000_000_000) return Math.floor(value / 1000)
  if (value > 10_000_000_000) return Math.floor(value / 1000)
  return Math.floor(value)
}

function timeframeChartWindowSeconds(timeframe: string | null | undefined): number {
  const normalized = normalizeCryptoTimeframe(timeframe)
  if (normalized === '5m') return 300
  if (normalized === '15m') return 900
  if (normalized === '1h') return 3600
  if (normalized === '4h') return 14_400
  return 900
}

const BOT_MODAL_SERIES_COLORS_DARK = ['#38bdf8', '#a78bfa', '#f59e0b', '#22d3ee', '#fb923c']
const BOT_MODAL_SERIES_COLORS_LIGHT = ['#0284c7', '#7c3aed', '#d97706', '#0e7490', '#c2410c']

function formatSeriesLabel(value: string): string {
  return String(value || '')
    .trim()
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase()) || 'Series'
}

function buildFlatLivelineSeries(
  value: number,
  startTime: number,
  endTime: number,
): LivelinePoint[] {
  const start = Math.max(1, Math.floor(startTime))
  const end = Math.max(start + 1, Math.floor(endTime))
  return [
    { time: start, value },
    { time: end, value },
  ]
}

function historyPointTimestampSeconds(point: Record<string, unknown> | unknown[]): number | null {
  if (Array.isArray(point)) {
    const rawTime = toFiniteNumber(point[0])
    if (rawTime === null) return null
    return Math.max(1, toUnixSeconds(rawTime))
  }
  const raw = point.t ?? point.ts ?? point.time ?? point.timestamp ?? point.date ?? point.created_at ?? point.updated_at
  const numeric = toFiniteNumber(raw)
  if (numeric !== null) return Math.max(1, toUnixSeconds(numeric))
  const isoTs = toTs(typeof raw === 'string' ? raw : null)
  if (isoTs <= 0) return null
  return Math.max(1, toUnixSeconds(isoTs))
}

function historyPointBinaryPrice(
  point: Record<string, unknown> | unknown[],
  directionSide: DirectionSide | null
): number | null {
  if (Array.isArray(point)) {
    const yes = toFiniteNumber(point[1] ?? point[0])
    const no = toFiniteNumber(point[2])
    if (directionSide === 'YES') return yes ?? (no !== null ? Math.max(0, Math.min(1, 1 - no)) : null)
    if (directionSide === 'NO') return no ?? (yes !== null ? Math.max(0, Math.min(1, 1 - yes)) : null)
    return yes ?? no
  }

  const yes = toFiniteNumber(point.yes ?? point.y ?? point.idx_0 ?? point.up ?? point.up_price)
  const no = toFiniteNumber(point.no ?? point.n ?? point.idx_1 ?? point.down ?? point.down_price)
  const mid = toFiniteNumber(point.p ?? point.price ?? point.mid ?? point.value)

  if (directionSide === 'YES') return yes ?? mid ?? (no !== null ? Math.max(0, Math.min(1, 1 - no)) : null)
  if (directionSide === 'NO') return no ?? mid ?? (yes !== null ? Math.max(0, Math.min(1, 1 - yes)) : null)
  return yes ?? mid ?? no
}

function extractLivelinePointsFromSharedHistory(
  history: unknown[],
  directionSide: DirectionSide | null,
): LivelinePoint[] {
  const points: LivelinePoint[] = []
  for (const entry of history) {
    if (!Array.isArray(entry) && !isRecord(entry)) continue
    const ts = historyPointTimestampSeconds(entry)
    const value = historyPointBinaryPrice(entry, directionSide)
    if (ts === null || value === null) continue
    points.push({ time: ts, value })
  }
  points.sort((left, right) => left.time - right.time)
  return points
}

function extractLivelinePointsFromOrders(
  orders: TraderOrder[],
  directionSide: DirectionSide | null,
): LivelinePoint[] {
  const points: LivelinePoint[] = []
  for (const order of orders) {
    const orderSide = resolveOrderDirectionPresentation(order).side || directionSide
    const payload = isRecord(order.payload) ? order.payload : {}
    const liveMarket = isRecord(payload.live_market) ? payload.live_market : {}
    const historyCandidates = [
      liveMarket.history_tail,
      payload.history_tail,
      payload.price_history,
    ]
    for (const history of historyCandidates) {
      if (!Array.isArray(history)) continue
      for (const entry of history) {
        if (!Array.isArray(entry) && !isRecord(entry)) continue
        const ts = historyPointTimestampSeconds(entry as Record<string, unknown> | unknown[])
        const value = historyPointBinaryPrice(entry as Record<string, unknown> | unknown[], orderSide)
        if (ts === null || value === null) continue
        points.push({ time: ts, value })
      }
    }

    const snapshot = resolveOrderModalSnapshot(order)
    const value = toFiniteNumber(snapshot.markPrice ?? snapshot.entryPrice ?? order.current_price)
    if (value === null) continue
    const tsRaw = latestTimestampValue(
      order.mark_updated_at,
      snapshot.updatedAt,
      order.updated_at,
      order.executed_at,
      order.created_at
    )
    const tsMs = toTs(tsRaw)
    if (tsMs <= 0) continue
    points.push({ time: Math.max(1, toUnixSeconds(tsMs)), value })
  }
  points.sort((left, right) => left.time - right.time)
  return points
}

interface BotLivelineResult {
  primary: LivelinePoint[]
  complement: LivelinePoint[]
}

function buildBotMarketLivelineSeries(params: {
  sharedHistory: unknown[]
  historyOrders: TraderOrder[]
  directionSide: DirectionSide | null
  markPrice: number | null
  entryPrice: number | null
  openedAt: string | null
  updatedAt: string | null
}): BotLivelineResult {
  const {
    sharedHistory,
    historyOrders,
    directionSide,
    markPrice,
    entryPrice,
    openedAt,
    updatedAt,
  } = params

  const complementSide: DirectionSide | null =
    directionSide === 'YES' ? 'NO' : directionSide === 'NO' ? 'YES' : null

  const buildSide = (side: DirectionSide | null): LivelinePoint[] => {
    const normalized = [
      ...extractLivelinePointsFromSharedHistory(sharedHistory, side),
      ...extractLivelinePointsFromOrders(historyOrders, side),
    ].sort((left, right) => left.time - right.time)

    const deduped: LivelinePoint[] = []
    for (const point of normalized) {
      const previous = deduped[deduped.length - 1]
      if (previous && previous.time === point.time) {
        deduped[deduped.length - 1] = point
        continue
      }
      deduped.push(point)
    }
    return deduped
  }

  const deduped = buildSide(directionSide)
  const complement = complementSide ? buildSide(complementSide) : []

  const livePrice = toFiniteNumber(markPrice ?? entryPrice)
  const nowSec = Math.floor(Date.now() / 1000)
  const openedSec = toTs(openedAt) > 0 ? Math.floor(toTs(openedAt) / 1000) : Math.max(1, nowSec - 120)
  const updatedSec = toTs(updatedAt) > 0 ? Math.floor(toTs(updatedAt) / 1000) : nowSec

  if (deduped.length === 0) {
    const basePrice = livePrice ?? 0.5
    const startTime = Math.max(1, Math.min(openedSec, updatedSec - 1))
    deduped.push({ time: startTime, value: entryPrice ?? basePrice })
    deduped.push({ time: Math.max(startTime + 1, updatedSec), value: basePrice })
  }

  if (deduped.length === 1) {
    const only = deduped[0]
    deduped.push({ time: only.time + 1, value: only.value })
  }

  if (livePrice !== null) {
    const previous = deduped[deduped.length - 1]
    const liveTime = Math.max(updatedSec, previous.time)
    if (liveTime > previous.time) {
      deduped.push({ time: liveTime, value: livePrice })
    } else if (Math.abs(previous.value - livePrice) > 1e-9) {
      deduped[deduped.length - 1] = { time: previous.time, value: livePrice }
    }
  }

  const cap = (arr: LivelinePoint[]) => arr.length <= 800 ? arr : arr.slice(arr.length - 800)
  return { primary: cap(deduped), complement: cap(complement) }
}

function marketMatchesCryptoIdentity(value: string | null | undefined, market: CryptoMarket | null): boolean {
  if (!market) return false
  const key = String(value || '').trim().toLowerCase()
  if (!key) return false
  const candidates = [market.id, market.condition_id, market.slug, market.event_slug]
    .map((candidate) => String(candidate || '').trim().toLowerCase())
    .filter(Boolean)
  return candidates.includes(key)
}

type OrderModalSnapshot = {
  status: string
  notionalUsd: number
  filledNotionalUsd: number
  filledShares: number
  entryPrice: number | null
  markPrice: number | null
  unrealizedPnl: number | null
  realizedPnl: number
  edgePercent: number | null
  confidencePercent: number | null
  source: string
  mode: string
  updatedAt: string | null
  createdAt: string | null
}

function resolveOrderModalSnapshot(order: TraderOrder): OrderModalSnapshot {
  const payload = isRecord(order.payload) ? order.payload : {}
  const providerReconciliation = isRecord(payload.provider_reconciliation) ? payload.provider_reconciliation : {}
  const providerSnapshot = isRecord(providerReconciliation.snapshot) ? providerReconciliation.snapshot : {}
  const positionState = isRecord(payload.position_state) ? payload.position_state : {}
  const status = normalizeStatus(order.status)
  const notionalUsd = Math.abs(toNumber(order.notional_usd))
  const filledNotionalUsd = Math.abs(
    toNumber(
      order.filled_notional_usd
      ?? providerReconciliation.filled_notional_usd
      ?? providerSnapshot.filled_notional_usd
      ?? order.notional_usd
    )
  )
  const filledShares = Math.max(
    0,
    toNumber(
      order.filled_shares
      ?? providerReconciliation.filled_size
      ?? providerSnapshot.filled_size
      ?? payload.filled_size
    )
  )
  const entryPrice = toFiniteNumber(
    order.average_fill_price
    ?? providerReconciliation.average_fill_price
    ?? providerSnapshot.average_fill_price
    ?? order.effective_price
    ?? order.entry_price
  )
  const markPrice = toFiniteNumber(
    order.current_price
    ?? positionState.last_mark_price
    ?? payload.market_price
    ?? payload.resolved_price
  )
  let unrealizedPnl = toFiniteNumber(order.unrealized_pnl)
  if (unrealizedPnl === null && markPrice !== null && filledShares > 0 && filledNotionalUsd > 0) {
    unrealizedPnl = (markPrice * filledShares) - filledNotionalUsd
  }
  return {
    status,
    notionalUsd,
    filledNotionalUsd,
    filledShares,
    entryPrice,
    markPrice,
    unrealizedPnl,
    realizedPnl: toNumber(order.actual_profit),
    edgePercent: toFiniteNumber(order.edge_percent),
    confidencePercent: toFiniteNumber(order.confidence),
    source: String(order.source || '').trim().toUpperCase() || 'UNKNOWN',
    mode: String(order.mode || '').trim().toUpperCase() || 'N/A',
    updatedAt: resolveOrderMarketUpdateTimestamp(order, payload),
    createdAt: cleanText(order.created_at) || cleanText(order.executed_at),
  }
}

function normalizeConfidencePercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (Math.abs(value) <= 1) return value * 100
  return value
}

function normalizeEdgePercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (Math.abs(value) <= 1) return value * 100
  if (Math.abs(value) > 200) return value / 100
  return value
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return 'n/a'
  const ts = new Date(value)
  if (Number.isNaN(ts.getTime())) return 'n/a'
  return ts.toLocaleString()
}

function formatShortDate(value: string | null | undefined): string {
  if (!value) return 'n/a'
  const ts = new Date(value)
  if (Number.isNaN(ts.getTime())) return 'n/a'
  return ts.toLocaleString()
}

function formatRelativeAge(value: string | null | undefined): string {
  const ts = toTs(value)
  if (ts <= 0) return '—'
  const ageMs = Math.max(0, Date.now() - ts)
  if (ageMs < 60_000) return `${Math.round(ageMs / 1000)}s`
  if (ageMs < 3_600_000) return `${Math.round(ageMs / 60_000)}m`
  if (ageMs < 86_400_000) return `${Math.round(ageMs / 3_600_000)}h`
  return `${Math.round(ageMs / 86_400_000)}d`
}

function computeOrderDynamicEdgePercent(params: {
  status: string
  edgePercent: number
  unrealizedPnl: number | null
  realizedPnl: number
  filledNotional: number
}): number {
  const {
    status,
    edgePercent,
    unrealizedPnl,
    realizedPnl,
    filledNotional,
  } = params
  if (OPEN_ORDER_STATUSES.has(status) && unrealizedPnl !== null && filledNotional > 0) {
    return (unrealizedPnl / filledNotional) * 100
  }
  if (RESOLVED_ORDER_STATUSES.has(status) && filledNotional > 0) {
    return (realizedPnl / filledNotional) * 100
  }
  return normalizeEdgePercent(edgePercent)
}

function computeOrderFillProgressPercent(
  payload: Record<string, unknown>,
  params: {
    filledSize: number
    filledNotional: number
    requestedNotionalFallback: number
  }
): number | null {
  const {
    filledSize,
    filledNotional,
    requestedNotionalFallback,
  } = params
  const requestedSize = (
    toFiniteNumber(payload.requested_shares)
    ?? toFiniteNumber(payload.requested_size)
    ?? toFiniteNumber(payload.shares)
  )
  if (requestedSize !== null && requestedSize > 0) {
    return clamp((Math.max(0, filledSize) / requestedSize) * 100, 0, 100)
  }
  const requestedNotional = (
    toFiniteNumber(payload.requested_notional_usd)
    ?? toFiniteNumber(payload.effective_notional_usd)
    ?? (requestedNotionalFallback > 0 ? requestedNotionalFallback : null)
  )
  if (requestedNotional !== null && requestedNotional > 0) {
    return clamp((Math.max(0, filledNotional) / requestedNotional) * 100, 0, 100)
  }
  return null
}

function computePendingExitProgressPercent(pendingExit: Record<string, unknown>): number | null {
  const fillRatio = toFiniteNumber(pendingExit.fill_ratio)
  if (fillRatio !== null && fillRatio >= 0) {
    return clamp(fillRatio * 100, 0, 100)
  }
  const filledSize = toFiniteNumber(pendingExit.filled_size)
  const exitSize = toFiniteNumber(pendingExit.exit_size)
  if (filledSize !== null && exitSize !== null && exitSize > 0) {
    return clamp((Math.max(0, filledSize) / exitSize) * 100, 0, 100)
  }
  return null
}

function shortId(value: string | null | undefined): string {
  if (!value) return 'n/a'
  return value.length <= 12 ? value : `${value.slice(0, 6)}...${value.slice(-4)}`
}

function normalizeMarketAlias(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

function collectMarketAliases(values: unknown[]): string[] {
  const seen = new Set<string>()
  const aliases: string[] = []
  for (const value of values) {
    const normalized = normalizeMarketAlias(value)
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    aliases.push(normalized)
  }
  return aliases
}

function collectOrderMarketAliasIds(order: TraderOrder): string[] {
  const payload = isRecord(order.payload) ? order.payload : {}
  const liveMarket = isRecord(payload.live_market) ? payload.live_market : {}
  const executionPlan = isRecord(payload.execution_plan) ? payload.execution_plan : {}
  const legs = Array.isArray(executionPlan.legs) ? executionPlan.legs : []
  const aliases = collectMarketAliases([
    order.market_id,
    payload.market_id,
    payload.marketId,
    payload.condition_id,
    payload.conditionId,
    payload.slug,
    payload.market_slug,
    payload.marketSlug,
    payload.event_slug,
    payload.eventSlug,
    payload.ticker,
    payload.event_ticker,
    payload.eventTicker,
    liveMarket.id,
    liveMarket.market_id,
    liveMarket.condition_id,
    liveMarket.conditionId,
    liveMarket.slug,
    liveMarket.market_slug,
    liveMarket.marketSlug,
    liveMarket.event_slug,
    liveMarket.eventSlug,
    liveMarket.ticker,
    liveMarket.event_ticker,
    liveMarket.eventTicker,
  ])
  for (const rawLeg of legs) {
    if (!isRecord(rawLeg)) continue
    for (const alias of collectMarketAliases([
      rawLeg.market_id,
      rawLeg.marketId,
      rawLeg.condition_id,
      rawLeg.conditionId,
      rawLeg.slug,
      rawLeg.market_slug,
      rawLeg.marketSlug,
      rawLeg.event_slug,
      rawLeg.eventSlug,
      rawLeg.ticker,
      rawLeg.event_ticker,
      rawLeg.eventTicker,
    ])) {
      if (!aliases.includes(alias)) aliases.push(alias)
    }
  }
  return aliases
}

function compactText(value: string | null | undefined, maxChars = 96): string {
  const text = cleanText(value)
  if (!text) return 'No reason provided'
  if (text.length <= maxChars) return text
  return `${text.slice(0, Math.max(1, maxChars - 1)).trimEnd()}…`
}

function buildOrderMarketLinks(
  order: TraderOrder,
  payload: Record<string, unknown>,
  signalPayload: Record<string, unknown> | null = null
): { polymarket: string | null; kalshi: string | null } {
  const mergedPayload = signalPayload ? { ...signalPayload, ...payload } : payload
  const links = getTraderOrderPlatformLinks({
    source: order.source,
    marketId: order.market_id,
    marketQuestion: order.market_question,
    payload: mergedPayload,
  })
  return {
    polymarket: links.polymarketUrl,
    kalshi: links.kalshiUrl,
  }
}

function isTraderExecutionEnabled(
  trader: Pick<Trader, 'is_enabled' | 'is_paused'> | null | undefined
): boolean {
  return Boolean(trader?.is_enabled) && !Boolean(trader?.is_paused)
}

function isTraderActive(
  trader: Pick<Trader, 'is_enabled'> | null | undefined
): boolean {
  return Boolean(trader?.is_enabled)
}

function resolveTraderStatusPresentation(
  trader: Pick<Trader, 'is_enabled' | 'is_paused'> | null | undefined,
  orchestratorExecutionActive: boolean
): TraderStatusPresentation {
  if (!isTraderActive(trader)) {
    return {
      key: 'inactive',
      label: 'Inactive',
      dotClassName: 'bg-zinc-500',
      badgeVariant: 'outline',
      badgeClassName: '',
    }
  }

  if (Boolean(trader?.is_paused)) {
    return {
      key: 'bot_stopped',
      label: 'Bot Stopped',
      dotClassName: 'bg-slate-400',
      badgeVariant: 'outline',
      badgeClassName: 'border-slate-400/35 bg-slate-500/10 text-slate-300',
    }
  }

  if (!orchestratorExecutionActive) {
    return {
      key: 'engine_stopped',
      label: 'Engine Stopped',
      dotClassName: 'bg-amber-400',
      badgeVariant: 'outline',
      badgeClassName: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
    }
  }

  return {
    key: 'running',
    label: 'Running',
    dotClassName: 'bg-emerald-500',
    badgeVariant: 'default',
    badgeClassName: '',
  }
}

function titleCaseStatusLabel(value: string): string {
  const normalized = String(value || '').trim().toLowerCase()
  if (!normalized) return 'Unknown'
  return normalized
    .split('_')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}

function resolveOrderLifecycleLabel(status: string): string {
  if (status === 'submitted' || status === 'pending' || status === 'queued') return 'Submitted'
  if (status === 'open') return 'Working'
  if (status === 'executed') return 'Filled'
  if (status === 'cancelled') return 'Canceled'
  if (status === 'rejected') return 'Rejected'
  if (status === 'failed' || status === 'error') return 'Failed'
  if (status === 'resolved_win') return 'Settled (Profit)'
  if (status === 'resolved_loss') return 'Settled (Loss)'
  if (status === 'closed_win') return 'Closed (Profit)'
  if (status === 'closed_loss') return 'Closed (Loss)'
  if (status === 'win') return 'Settled (Profit)'
  if (status === 'loss') return 'Settled (Loss)'
  if (status === 'resolved') return 'Settled'
  return titleCaseStatusLabel(status)
}

function resolveVenueStatusPresentation(providerSnapshotStatus: string): {
  label: string
  detail: string
  className: string
} {
  const key = normalizeStatus(providerSnapshotStatus)
  if (key === 'filled') {
    return {
      label: 'Filled',
      detail: 'Venue reports the order as filled.',
      className: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/12 dark:text-emerald-200',
    }
  }
  if (key === 'partially_filled') {
    return {
      label: 'Partial',
      detail: 'Venue reports a partial fill.',
      className: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/12 dark:text-sky-200',
    }
  }
  if (key === 'open') {
    return {
      label: 'Working',
      detail: 'Venue order remains working on book.',
      className: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/12 dark:text-sky-200',
    }
  }
  if (key === 'pending') {
    return {
      label: 'Pending',
      detail: 'Venue has accepted but not yet worked the order.',
      className: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/12 dark:text-amber-200',
    }
  }
  if (key === 'cancelled' || key === 'expired') {
    return {
      label: 'Canceled',
      detail: 'Venue confirms cancellation/expiry.',
      className: 'border-zinc-300 bg-zinc-100 text-zinc-900 dark:border-zinc-400/45 dark:bg-zinc-500/12 dark:text-zinc-200',
    }
  }
  if (key === 'failed' || key === 'rejected') {
    return {
      label: 'Rejected',
      detail: 'Venue reports failed/rejected execution.',
      className: 'border-red-300 bg-red-100 text-red-900 dark:border-red-400/45 dark:bg-red-500/12 dark:text-red-200',
    }
  }
  return {
    label: '\u2014',
    detail: 'No venue status snapshot available.',
    className: 'border-border bg-muted/50 text-muted-foreground',
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function cleanText(value: unknown): string | null {
  const text = String(value || '').trim()
  return text ? text : null
}

function resolveOrderMarketUpdateTimestamp(
  order: TraderOrder,
  payloadInput?: Record<string, unknown>,
): string {
  const payload = payloadInput ?? (isRecord(order.payload) ? order.payload : {})
  const providerReconciliation = isRecord(payload.provider_reconciliation) ? payload.provider_reconciliation : {}
  const providerSnapshot = isRecord(providerReconciliation.snapshot) ? providerReconciliation.snapshot : {}
  const positionState = isRecord(payload.position_state) ? payload.position_state : {}
  const liveMarket = isRecord(payload.live_market) ? payload.live_market : {}
  return latestTimestampValue(
    cleanText(order.mark_updated_at),
    cleanText(positionState.last_marked_at),
    cleanText(providerSnapshot.updated_at),
    cleanText(providerSnapshot.updatedAt),
    cleanText(liveMarket.live_market_fetched_at),
    cleanText(liveMarket.fetched_at)
  )
}

function resolveOrderExitEvaluationTimestamp(
  order: TraderOrder,
  payloadInput?: Record<string, unknown>,
): string {
  const payload = payloadInput ?? (isRecord(order.payload) ? order.payload : {})
  const positionState = isRecord(payload.position_state) ? payload.position_state : {}
  const pendingExit = isRecord(payload.pending_live_exit) ? payload.pending_live_exit : {}
  return latestTimestampValue(
    cleanText(positionState.last_exit_evaluated_at),
    cleanText(pendingExit.last_attempt_at),
    cleanText(pendingExit.triggered_at)
  )
}

function normalizeDecisionOutcome(value: unknown): Exclude<DecisionOutcomeFilter, 'all'> {
  const outcome = String(value || '').trim().toLowerCase()
  if (outcome === 'selected') return 'selected'
  if (outcome === 'blocked') return 'blocked'
  return 'skipped'
}

function resolveDecisionMarketLabel(decision: {
  market_id: string | null
  market_question: string | null
  signal_payload?: Record<string, unknown>
}): string {
  const marketId = cleanText(decision.market_id)
  const normalizedMarketId = marketId ? marketId.toLowerCase() : null
  const signalPayload = isRecord(decision.signal_payload) ? decision.signal_payload : null
  if (signalPayload) {
    const markets = Array.isArray(signalPayload.markets) ? signalPayload.markets : []
    let firstQuestion: string | null = null
    for (const rawMarket of markets) {
      if (!isRecord(rawMarket)) continue
      const question = cleanText(rawMarket.question)
      if (!firstQuestion && question) firstQuestion = question
      const candidateId = cleanText(rawMarket.id)
      if (
        question &&
        normalizedMarketId &&
        candidateId &&
        candidateId.toLowerCase() === normalizedMarketId
      ) {
        return question
      }
    }
    if (firstQuestion) return firstQuestion
  }

  const marketQuestion = cleanText(decision.market_question)
  if (!marketQuestion) return shortId(marketId)
  const withoutMoreSuffix = marketQuestion.replace(/\s+\+\d+\s+more$/i, '').trim()
  const firstSegment = withoutMoreSuffix.split(' | ')[0]?.trim() || withoutMoreSuffix
  const withoutActionPrefix = firstSegment.replace(/^(buy|sell)\s+(yes|no)\s+/i, '').trim()
  const withoutPriceSuffix = withoutActionPrefix.replace(/\s+@\d+(?:\.\d+)?$/, '').trim()
  if (withoutPriceSuffix) return withoutPriceSuffix
  return shortId(marketId)
}

function toFiniteNumber(value: unknown): number | null {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function normalizeTradeAction(value: unknown): TradeAction | null {
  const key = String(value || '').trim().toLowerCase()
  if (!key) return null
  if (
    key === 'sell'
    || key.startsWith('sell_')
    || key.endsWith('_sell')
    || key === 'short'
    || key === 'close'
    || key === 'exit'
  ) {
    return 'SELL'
  }
  if (
    key === 'buy'
    || key.startsWith('buy')
    || key === 'long'
    || key === 'open'
    || key === 'yes'
    || key === 'no'
  ) {
    return 'BUY'
  }
  return null
}

function normalizeOutcome(value: unknown): 'YES' | 'NO' | null {
  const key = String(value || '').trim().toLowerCase()
  if (!key) return null
  if (
    key === 'yes'
    || key === 'buy_yes'
    || key === 'sell_yes'
    || key.endsWith('_yes')
    || key.startsWith('yes_')
    || key === 'long'
    || key === 'up'
  ) {
    return 'YES'
  }
  if (
    key === 'no'
    || key === 'buy_no'
    || key === 'sell_no'
    || key.endsWith('_no')
    || key.startsWith('no_')
    || key === 'short'
    || key === 'down'
  ) {
    return 'NO'
  }
  return null
}

function normalizeDirectionSide(value: unknown): DirectionSide | null {
  const key = String(value || '').trim().toUpperCase().replace(/[\s-]+/g, '_')
  if (!key) return null
  if (key === 'YES' || key === 'BUY_YES' || key === 'SELL_YES' || key === 'BUY' || key === 'LONG' || key === 'UP') {
    return 'YES'
  }
  if (key === 'NO' || key === 'BUY_NO' || key === 'SELL_NO' || key === 'SELL' || key === 'SHORT' || key === 'DOWN') {
    return 'NO'
  }
  return null
}

function isGenericDirectionLabel(value: string | null | undefined): boolean {
  const key = String(value || '').trim().toUpperCase()
  return key === 'YES' || key === 'NO'
}

function resolveOrderBinaryOutcomeLabels(
  order: TraderOrder,
): { yesLabel: string | null; noLabel: string | null } {
  let yesLabel = cleanText(order.yes_label)
  let noLabel = cleanText(order.no_label)
  const side = normalizeDirectionSide(order.direction_side ?? order.direction)
  const explicitLabel = cleanText(order.direction_label)
  if (explicitLabel && !isGenericDirectionLabel(explicitLabel)) {
    if (side === 'YES' && !yesLabel) yesLabel = explicitLabel
    if (side === 'NO' && !noLabel) noLabel = explicitLabel
  }

  return {
    yesLabel: yesLabel || null,
    noLabel: noLabel || null,
  }
}

function resolveOrderDirectionPresentation(
  order: TraderOrder,
): {
  side: DirectionSide | null
  label: string
  yesLabel: string | null
  noLabel: string | null
} {
  const side = normalizeDirectionSide(order.direction_side ?? order.direction)
  const binaryLabels = resolveOrderBinaryOutcomeLabels(order)
  const explicitLabel = cleanText(order.direction_label)
  const marketAliases = collectOrderMarketAliasIds(order)
  const stableMarketKey = marketAliases[0] || normalizeMarketAlias(order.market_id)
  const stableSideKey = side && stableMarketKey ? `${stableMarketKey}:${side}` : null
  const explicitIsGeneric = isGenericDirectionLabel(explicitLabel)

  if (stableSideKey && side === 'YES' && binaryLabels.yesLabel) {
    STABLE_OUTCOME_LABELS_BY_MARKET_SIDE.set(stableSideKey, binaryLabels.yesLabel)
  }
  if (stableSideKey && side === 'NO' && binaryLabels.noLabel) {
    STABLE_OUTCOME_LABELS_BY_MARKET_SIDE.set(stableSideKey, binaryLabels.noLabel)
  }
  if (stableSideKey && explicitLabel && !explicitIsGeneric) {
    STABLE_OUTCOME_LABELS_BY_MARKET_SIDE.set(stableSideKey, explicitLabel)
  }

  const stableSideLabel = stableSideKey
    ? STABLE_OUTCOME_LABELS_BY_MARKET_SIDE.get(stableSideKey) || null
    : null
  const sideLabel = (
    side === 'YES'
      ? (binaryLabels.yesLabel || stableSideLabel)
      : side === 'NO'
        ? (binaryLabels.noLabel || stableSideLabel)
        : null
  )
  const label = (
    sideLabel
    || (explicitLabel && !isGenericDirectionLabel(explicitLabel) ? explicitLabel : null)
    || explicitLabel
    || side
    || String(order.direction || '').trim().toUpperCase()
    || 'N/A'
  )
  return {
    side,
    label,
    yesLabel: binaryLabels.yesLabel,
    noLabel: binaryLabels.noLabel,
  }
}

function resolveDecisionDirectionPresentation(decision: {
  direction: string | null
  direction_side?: string | null
  direction_label?: string | null
}): { side: DirectionSide | null; label: string } {
  const side = normalizeDirectionSide(decision.direction_side ?? decision.direction)
  const label = cleanText(decision.direction_label) || side || String(decision.direction || '').trim().toUpperCase() || 'N/A'
  return {
    side,
    label,
  }
}

function terminalLegFromExecutionPlanLeg(leg: Record<string, unknown>): TerminalLeg | null {
  const marketId = cleanText(leg.market_id)
  const marketQuestion = cleanText(leg.market_question)
  const action = normalizeTradeAction(leg.side ?? leg.action)
  const outcome = normalizeOutcome(leg.outcome ?? leg.direction)
  const price = toFiniteNumber(leg.limit_price ?? leg.target_price ?? leg.price)
  if (!marketId && !marketQuestion && !action && !outcome && price === null) return null
  return {
    action,
    outcome,
    marketId,
    marketQuestion,
    price,
  }
}

function collectExecutionPlanLegs(payload: Record<string, unknown> | null): TerminalLeg[] {
  if (!payload) return []
  const executionPlan = isRecord(payload.execution_plan) ? payload.execution_plan : null
  if (!executionPlan) return []
  const rawLegs = Array.isArray(executionPlan.legs) ? executionPlan.legs : []
  const legs: TerminalLeg[] = []
  for (const rawLeg of rawLegs) {
    if (!isRecord(rawLeg)) continue
    const leg = terminalLegFromExecutionPlanLeg(rawLeg)
    if (leg) legs.push(leg)
  }
  return legs
}

function collectSignalPositionLegs(
  signalPayload: Record<string, unknown> | null,
  fallbackDirection: unknown
): TerminalLeg[] {
  if (!signalPayload) return []
  const positions = Array.isArray(signalPayload.positions_to_take) ? signalPayload.positions_to_take : []
  if (positions.length === 0) return []
  const marketById = new Map<string, string>()
  const markets = Array.isArray(signalPayload.markets) ? signalPayload.markets : []
  for (const rawMarket of markets) {
    if (!isRecord(rawMarket)) continue
    const marketId = cleanText(rawMarket.id)
    const marketQuestion = cleanText(rawMarket.question)
    if (marketId && marketQuestion) {
      marketById.set(marketId, marketQuestion)
    }
  }
  const legs: TerminalLeg[] = []
  for (const rawPosition of positions) {
    if (!isRecord(rawPosition)) continue
    const marketId = cleanText(rawPosition.market_id ?? rawPosition.id ?? rawPosition.market)
    const marketQuestion = cleanText(rawPosition.market_question) || (marketId ? marketById.get(marketId) || null : null)
    const action = normalizeTradeAction(rawPosition.action ?? rawPosition.side ?? fallbackDirection)
    const outcome = normalizeOutcome(rawPosition.outcome ?? fallbackDirection)
    const price = toFiniteNumber(rawPosition.price)
    legs.push({
      action,
      outcome,
      marketId,
      marketQuestion,
      price,
    })
  }
  return legs
}

function collectDecisionLegs(decision: {
  direction: string | null
  market_id: string | null
  market_question: string | null
  market_price: number | null
  payload: Record<string, unknown>
  signal_payload?: Record<string, unknown>
}): TerminalLeg[] {
  const decisionPayload = isRecord(decision.payload) ? decision.payload : null
  const strategyPayload = decisionPayload && isRecord(decisionPayload.strategy_payload)
    ? decisionPayload.strategy_payload
    : null
  const signalPayload = isRecord(decision.signal_payload) ? decision.signal_payload : null

  const candidates = [decisionPayload, strategyPayload, signalPayload]
  for (const candidate of candidates) {
    const legs = collectExecutionPlanLegs(candidate)
    if (legs.length > 0) return legs
  }

  const fallbackFromPositions = collectSignalPositionLegs(signalPayload, decision.direction)
  if (fallbackFromPositions.length > 0) return fallbackFromPositions

  const fallbackMarketId = cleanText(decision.market_id)
  const fallbackMarketQuestion = cleanText(decision.market_question)
  if (!fallbackMarketId && !fallbackMarketQuestion) return []
  return [
    {
      action: normalizeTradeAction(decision.direction),
      outcome: normalizeOutcome(decision.direction),
      marketId: fallbackMarketId,
      marketQuestion: fallbackMarketQuestion,
      price: toFiniteNumber(decision.market_price),
    },
  ]
}

function collectOrderLeg(order: TraderOrder): TerminalLeg {
  const orderPayload = isRecord(order.payload) ? order.payload : null
  const legPayload = orderPayload && isRecord(orderPayload.leg) ? orderPayload.leg : null
  return {
    action: normalizeTradeAction(
      (legPayload ? legPayload.side : null)
      ?? (orderPayload ? orderPayload.side : null)
      ?? (orderPayload ? orderPayload.action : null)
      ?? order.direction
    ),
    outcome: normalizeOutcome(
      (legPayload ? legPayload.outcome : null)
      ?? (orderPayload ? orderPayload.outcome : null)
      ?? order.direction
    ),
    marketId: cleanText((legPayload ? legPayload.market_id : null) ?? order.market_id),
    marketQuestion: cleanText((legPayload ? legPayload.market_question : null) ?? order.market_question ?? order.market_id),
    price: toFiniteNumber(order.effective_price ?? (legPayload ? legPayload.limit_price : null) ?? order.entry_price),
  }
}

function primaryMarketLabel(legs: TerminalLeg[], fallback: string | null): string {
  const labels = new Set<string>()
  for (const leg of legs) {
    const label = cleanText(leg.marketQuestion) || (leg.marketId ? shortId(leg.marketId) : null)
    if (label) labels.add(label)
  }
  const uniqueLabels = Array.from(labels)
  if (uniqueLabels.length === 0) return fallback || 'n/a'
  if (uniqueLabels.length === 1) return uniqueLabels[0]
  return `${uniqueLabels[0]} +${uniqueLabels.length - 1} more`
}

function renderLegLabel(leg: TerminalLeg): string {
  const actionPart = leg.action || 'HOLD'
  const outcomePart = leg.outcome ? ` ${leg.outcome}` : ''
  const marketPart = cleanText(leg.marketQuestion) || (leg.marketId ? shortId(leg.marketId) : 'n/a')
  const pricePart = leg.price !== null ? ` @${leg.price.toFixed(3)}` : ''
  return `${actionPart}${outcomePart} ${marketPart}${pricePart}`
}

function renderMarketsDetail(legs: TerminalLeg[], fallback: string | null): string {
  if (legs.length === 0) return fallback || 'n/a'
  const rendered = legs.slice(0, 3).map((leg) => renderLegLabel(leg))
  if (legs.length > 3) {
    rendered.push(`+${legs.length - 3} more`)
  }
  return rendered.join(' | ')
}

function normalizeActivityText(value: string | null): string {
  return String(value || '').trim().toLowerCase().replace(/\s+/g, ' ')
}

function activityDuplicateFingerprint(
  traderId: string | null,
  timestamp: string | null,
  reason: string,
  market: string | null,
): string {
  const bucketSeconds = Math.floor(toTs(timestamp) / 1000)
  return [
    normalizeActivityText(traderId),
    String(bucketSeconds),
    normalizeActivityText(reason),
    normalizeActivityText(market),
  ].join('|')
}

function areReasonsEquivalent(left: string, right: string): boolean {
  const a = normalizeActivityText(left)
  const b = normalizeActivityText(right)
  if (!a || !b) return false
  return a === b || a.includes(b) || b.includes(a)
}

function isGenericDecisionReason(reason: string | null): boolean {
  const normalized = normalizeActivityText(reason)
  if (!normalized) return false
  return normalized.includes('crypto worker filters not met') || normalized.includes('filters not met')
}

function decisionReasonDetail(decision: {
  reason: string | null
  payload: Record<string, unknown>
  failed_checks?: Array<unknown>
}): string {
  const payload = isRecord(decision.payload) ? decision.payload : null
  const strategyDecision = payload && isRecord(payload.strategy_decision) ? payload.strategy_decision : null
  const platformGates = payload && Array.isArray(payload.platform_gates) ? payload.platform_gates : []
  const failedChecks = Array.isArray(decision.failed_checks) ? decision.failed_checks : []
  let failedGateReason: string | null = null
  for (const rawGate of platformGates) {
    if (!isRecord(rawGate)) continue
    if (rawGate.passed === false) {
      failedGateReason = cleanText(rawGate.detail) || cleanText(rawGate.reason)
      if (failedGateReason) break
    }
  }
  let failedCheckReason: string | null = null
  for (const rawCheck of failedChecks) {
    if (!isRecord(rawCheck)) continue
    failedCheckReason = cleanText(rawCheck.detail)
    if (failedCheckReason) break
  }
  const reason = cleanText(decision.reason)
  const strategyReason = strategyDecision ? cleanText(strategyDecision.reason) : null
  const bestFallback = failedCheckReason || failedGateReason || strategyReason
  if (reason && isGenericDecisionReason(reason) && bestFallback) {
    return `${reason} | ${bestFallback}`
  }
  return reason || strategyReason || failedCheckReason || failedGateReason || 'No reason provided'
}

function decisionFailedChecksDetail(decision: {
  failed_checks?: Array<unknown>
}): string | null {
  const failedChecks = Array.isArray(decision.failed_checks) ? decision.failed_checks : []
  if (failedChecks.length === 0) return null
  const rendered: string[] = []
  for (const rawCheck of failedChecks.slice(0, 4)) {
    if (!isRecord(rawCheck)) continue
    const label = cleanText(rawCheck.check_label) || cleanText(rawCheck.check_key) || 'check'
    const detail = cleanText(rawCheck.detail)
    const score = toFiniteNumber(rawCheck.score)
    const scoreText = score !== null ? ` (score=${score.toFixed(3)})` : ''
    rendered.push(detail ? `${label}: ${detail}${scoreText}` : `${label}${scoreText}`)
  }
  if (rendered.length === 0) return null
  if (failedChecks.length > rendered.length) {
    rendered.push(`+${failedChecks.length - rendered.length} more`)
  }
  return rendered.join(' | ')
}

function orderCloseLifecycleReason(order: TraderOrder): string | null {
  const payload = isRecord(order.payload) ? order.payload : null
  if (!payload) return null
  const positionClose = isRecord(payload.position_close) ? payload.position_close : null
  const pendingExit = isRecord(payload.pending_live_exit) ? payload.pending_live_exit : null
  const closeTrigger = cleanText(order.close_trigger) || cleanText(positionClose ? positionClose.close_trigger : null)
  const closeReason = cleanText(order.close_reason) || cleanText(positionClose ? positionClose.reason : null)
  const pendingTrigger = cleanText(pendingExit ? pendingExit.close_trigger : null)
  const pendingReason = cleanText(pendingExit ? pendingExit.reason : null)
  const primary = closeTrigger || pendingTrigger || closeReason || pendingReason
  if (!primary) return null
  const secondary = closeReason || pendingReason
  if (
    secondary
    && primary.toLowerCase() !== secondary.toLowerCase()
    && !primary.toLowerCase().includes(secondary.toLowerCase())
  ) {
    return `${primary} • ${secondary}`
  }
  return primary
}

function orderReasonDetail(order: TraderOrder): string {
  const status = normalizeStatus(order.status)
  const closeLifecycleReason = orderCloseLifecycleReason(order)
  if ((RESOLVED_ORDER_STATUSES.has(status) || FAILED_ORDER_STATUSES.has(status)) && closeLifecycleReason) {
    return closeLifecycleReason
  }
  const payload = isRecord(order.payload) ? order.payload : null
  const legPayload = payload && isRecord(payload.leg) ? payload.leg : null
  return (
    cleanText(order.error_message)
    || cleanText(order.reason)
    || (payload ? cleanText(payload.error_message) : null)
    || (payload ? cleanText(payload.reason) : null)
    || (payload ? cleanText(payload.message) : null)
    || (legPayload ? cleanText(legPayload.reason) : null)
    || closeLifecycleReason
    || (status === 'executed' ? 'Execution filled' : 'No reason provided')
  )
}

function orderCloseHeadlineFromReason(reason: string): string {
  const normalizedReason = reason.toLowerCase()
  if (normalizedReason.includes('stop loss')) return 'Stop loss'
  if (normalizedReason.includes('take profit')) return 'Take profit'
  if (normalizedReason.includes('trailing stop')) return 'Trailing stop'
  if (normalizedReason.includes('max hold')) return 'Max hold'
  if (normalizedReason.includes('market inactive')) return 'Market inactive'
  if (normalizedReason.includes('external_wallet_flatten') || normalizedReason.includes('external wallet')) return 'External flatten'
  if (normalizedReason.includes('resolution')) return 'Resolution'
  return 'Closed'
}

function isAllowanceErrorText(raw: string): boolean {
  const text = String(raw || '').toLowerCase()
  if (!text) return false
  return (
    text.includes('not enough balance / allowance')
    || text.includes('balance/allowance')
    || text.includes('conditional token balance/allowance')
    || (text.includes('allowance') && text.includes('not enough'))
  )
}

function isGasErrorText(raw: string): boolean {
  const text = String(raw || '').toLowerCase()
  if (!text) return false
  if (text.includes('not enough gas')) return true
  if (text.includes('insufficient funds for gas')) return true
  if (text.includes('out of gas')) return true
  if (text.includes('intrinsic gas too low')) return true
  if (text.includes('base fee') && text.includes('gas')) return true
  if (text.includes('gas required exceeds allowance')) return true
  if (text.includes('insufficient') && (text.includes('matic') || text.includes('polygon') || text.includes('native token'))) return true
  if (text.includes('insufficient') && text.includes('gas')) return true
  return false
}

function orderFailureHeadline(order: TraderOrder): string {
  const status = normalizeStatus(order.status)
  const reason = orderReasonDetail(order)
  const normalizedReason = reason.toLowerCase()
  const payload = isRecord(order.payload) ? order.payload : null
  const submission = payload ? cleanText(payload.submission)?.toLowerCase() || '' : ''
  const priceResolution = payload ? cleanText(payload.price_resolution)?.toLowerCase() || '' : ''
  const resolvedPrice = payload ? toFiniteNumber(payload.resolved_price) : null

  if (status === 'cancelled') {
    if (normalizedReason.includes('cleanup:max_open_order_timeout')) {
      return 'Unfilled timeout cancel'
    }
    if (normalizedReason.includes('session:expired') || normalizedReason.includes('session timed out')) {
      return 'Session expired'
    }
    if (normalizedReason.includes('cleanup:')) {
      return 'Cleanup cancel'
    }
    return 'Canceled'
  }

  if (
    normalizedReason.includes('could not resolve a valid live price')
    || (
      submission === 'rejected'
      && priceResolution === 'live_quote'
      && resolvedPrice !== null
      && resolvedPrice <= 0
    )
  ) {
    return 'No live quote'
  }
  if (normalizedReason.includes('maximum open positions')) {
    return 'Position cap hit'
  }
  if (isGasErrorText(normalizedReason)) {
    return 'Insufficient gas'
  }
  if (isAllowanceErrorText(normalizedReason) || normalizedReason.includes('not enough balance')) {
    return 'Balance/allowance'
  }
  if (normalizedReason.includes('invalid signature')) {
    return 'Signature invalid'
  }
  if (
    normalizedReason.includes('below minimum')
    || normalizedReason.includes('min order')
    || normalizedReason.includes('exit_notional_below_min')
  ) {
    return 'Below minimum size'
  }
  if (normalizedReason.includes('global pause')) {
    return 'Global pause'
  }
  if (normalizedReason.includes('kill switch')) {
    return 'Kill switch active'
  }
  return 'Execution rejected'
}

function orderOutcomeSummary(order: TraderOrder): { headline: string; detail: string } {
  const status = normalizeStatus(order.status)
  const reason = orderReasonDetail(order)
  if (FAILED_ORDER_STATUSES.has(status)) {
    return {
      headline: orderFailureHeadline(order),
      detail: reason,
    }
  }
  if (RESOLVED_ORDER_STATUSES.has(status)) {
    return {
      headline: orderCloseHeadlineFromReason(reason),
      detail: reason,
    }
  }
  if (OPEN_ORDER_STATUSES.has(status)) {
    return {
      headline: 'Working',
      detail: reason,
    }
  }
  return {
    headline: status.toUpperCase(),
    detail: reason,
  }
}

type TradeLifecycleStageTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger'
type TradeLifecycleStageState = 'done' | 'current' | 'future'

type TradeLifecycleStage = {
  key: string
  label: string
  tone: TradeLifecycleStageTone
  state: TradeLifecycleStageState
}

function buildTradeLifecycleStages(args: {
  status: string
  outcomeHeadline: string
  reasonDetail: string
  closeTrigger: string | null
}): TradeLifecycleStage[] {
  const status = normalizeStatus(args.status)
  const reason = String(args.reasonDetail || '').toLowerCase()
  const closeTriggerLabel = args.closeTrigger ? `Exit (${compactText(args.closeTrigger, 20)})` : 'Exit'
  const finalLabel = compactText(args.outcomeHeadline || resolveOrderLifecycleLabel(status), 28)
  const stages: TradeLifecycleStage[] = [
    { key: 'signal', label: 'Signal', tone: 'neutral', state: 'done' },
    { key: 'submitted', label: 'Submitted', tone: 'info', state: 'future' },
    { key: 'working', label: 'Working', tone: 'info', state: 'future' },
    { key: 'exit', label: closeTriggerLabel, tone: 'warning', state: 'future' },
    { key: 'outcome', label: finalLabel, tone: 'neutral', state: 'future' },
  ]

  if (status === 'submitted' || status === 'pending' || status === 'queued') {
    stages[1].state = 'current'
    return stages
  }
  if (status === 'open') {
    stages[1].state = 'done'
    stages[2].state = 'current'
    return stages
  }
  if (status === 'executed') {
    stages[1].state = 'done'
    stages[2].state = 'current'
    stages[2].label = 'Filled'
    stages[2].tone = 'success'
    return stages
  }
  if (RESOLVED_ORDER_STATUSES.has(status)) {
    stages[1].state = 'done'
    stages[2].state = 'done'
    stages[2].label = 'Filled'
    stages[2].tone = 'success'
    stages[3].state = 'done'
    stages[4].state = 'current'
    if (status.includes('loss') || status === 'loss') {
      stages[4].tone = 'danger'
    } else if (status.includes('win') || status === 'win') {
      stages[4].tone = 'success'
    } else {
      stages[4].tone = 'info'
    }
    return stages
  }
  if (status === 'cancelled') {
    stages[1].state = 'done'
    stages[2].state = 'done'
    stages[3].state = 'done'
    stages[4].state = 'current'
    stages[4].tone = reason.includes('session:expired') ? 'warning' : 'neutral'
    if (reason.includes('cleanup:max_open_order_timeout')) {
      stages[4].label = 'Timeout cancel'
      stages[4].tone = 'warning'
    } else if (reason.includes('session:expired') || reason.includes('session timed out')) {
      stages[4].label = 'Session expired'
      stages[4].tone = 'warning'
    } else {
      stages[4].label = 'Canceled'
    }
    return stages
  }
  if (status === 'failed' || status === 'rejected' || status === 'error') {
    stages[1].state = 'done'
    stages[4].state = 'current'
    stages[4].tone = 'danger'
    return stages
  }

  stages[4].state = 'current'
  return stages
}

function tradeLifecycleStageClassName(stage: TradeLifecycleStage, pulseCurrentStage: boolean): string {
  const base = 'inline-flex h-4 items-center rounded-full border px-1.5 text-[8px] font-semibold whitespace-nowrap'
  if (stage.state === 'future') {
    return `${base} border-border/60 bg-background/50 text-muted-foreground/65`
  }
  if (stage.state === 'current') {
    const pulseClass = pulseCurrentStage ? ' animate-pulse' : ''
    if (stage.tone === 'success') {
      return `${base} border-emerald-300 bg-emerald-100 text-emerald-900 ring-1 ring-emerald-300/60${pulseClass} dark:border-emerald-400/45 dark:bg-emerald-500/15 dark:text-emerald-200`
    }
    if (stage.tone === 'warning') {
      return `${base} border-amber-300 bg-amber-100 text-amber-900 ring-1 ring-amber-300/60${pulseClass} dark:border-amber-400/45 dark:bg-amber-500/15 dark:text-amber-200`
    }
    if (stage.tone === 'danger') {
      return `${base} border-red-300 bg-red-100 text-red-900 ring-1 ring-red-300/60${pulseClass} dark:border-red-400/45 dark:bg-red-500/15 dark:text-red-200`
    }
    if (stage.tone === 'info') {
      return `${base} border-sky-300 bg-sky-100 text-sky-900 ring-1 ring-sky-300/60${pulseClass} dark:border-sky-400/45 dark:bg-sky-500/15 dark:text-sky-200`
    }
    return `${base} border-border bg-muted/70 text-foreground ring-1 ring-border/70${pulseClass}`
  }
  if (stage.tone === 'success') {
    return `${base} border-emerald-300/80 bg-emerald-100/70 text-emerald-900 dark:border-emerald-400/40 dark:bg-emerald-500/12 dark:text-emerald-200`
  }
  if (stage.tone === 'warning') {
    return `${base} border-amber-300/80 bg-amber-100/70 text-amber-900 dark:border-amber-400/40 dark:bg-amber-500/12 dark:text-amber-200`
  }
  if (stage.tone === 'danger') {
    return `${base} border-red-300/80 bg-red-100/70 text-red-900 dark:border-red-400/40 dark:bg-red-500/12 dark:text-red-200`
  }
  if (stage.tone === 'info') {
    return `${base} border-sky-300/80 bg-sky-100/70 text-sky-900 dark:border-sky-400/40 dark:bg-sky-500/12 dark:text-sky-200`
  }
  return `${base} border-border/70 bg-muted/60 text-foreground/80`
}

function renderTradeLifecycleFlow(args: {
  status: string
  outcomeHeadline: string
  outcomeDetail: string
  executionSummary: string
  venueLabel: string
  closeTrigger: string | null
  pendingExitLabel?: string | null
  pendingExitTone?: 'neutral' | 'warning'
  pulseCurrentStage?: boolean
}): ReactNode {
  const stages = buildTradeLifecycleStages({
    status: args.status,
    outcomeHeadline: args.outcomeHeadline,
    reasonDetail: args.outcomeDetail,
    closeTrigger: args.closeTrigger,
  })
  const compactReason = compactText(args.outcomeDetail || 'No reason provided', 180)
  const metaParts: string[] = []
  if (args.executionSummary && args.executionSummary !== '—') metaParts.push(args.executionSummary)
  if (args.venueLabel && args.venueLabel !== '—') metaParts.push(`Venue ${args.venueLabel}`)
  if (args.pendingExitLabel) metaParts.push(args.pendingExitLabel)
  const pulseCurrentStage = Boolean(args.pulseCurrentStage)

  return (
    <div className="w-full px-2 py-0.5">
      <div className="flex min-w-0 items-center gap-1.5 overflow-hidden">
        {stages.map((stage, index) => (
          <div key={stage.key} className="flex items-center gap-1">
            <span className={tradeLifecycleStageClassName(stage, pulseCurrentStage)}>{stage.label}</span>
            {index < stages.length - 1 && <ChevronRight className="h-3 w-3 text-muted-foreground/65" />}
          </div>
        ))}
        <span className="h-3 w-px shrink-0 bg-border/50" />
        <span className="min-w-0 flex-1 truncate text-[9px] text-foreground/90" title={args.outcomeDetail || 'No reason provided'}>
          <span className="mr-1 text-muted-foreground">Reason:</span>
          {compactReason}
        </span>
        {metaParts.length > 0 && (
          <span
            className={cn(
              'shrink-0 truncate text-[8px] text-muted-foreground',
              args.pendingExitTone === 'warning' && 'text-amber-700 dark:text-amber-300'
            )}
            title={metaParts.join(' • ')}
          >
            {metaParts.join(' • ')}
          </span>
        )}
      </div>
    </div>
  )
}

function normalizeExecutionToken(value: unknown): string | null {
  const text = cleanText(value)
  if (!text) return null
  return text.replace(/[\s-]+/g, '_').toLowerCase()
}

function orderExecutionTypeSummary(order: TraderOrder): string {
  const payload = isRecord(order.payload) ? order.payload : null
  const legPayload = payload && isRecord(payload.leg) ? payload.leg : null
  const paperSimulation = payload && isRecord(payload.paper_simulation) ? payload.paper_simulation : null

  const pricePolicy = normalizeExecutionToken(
    (legPayload ? legPayload.price_policy : null)
    ?? (paperSimulation ? paperSimulation.price_policy : null)
    ?? (payload ? payload.price_policy : null)
  )
  const timeInForceRaw = cleanText(
    (legPayload ? legPayload.time_in_force : null)
    ?? (paperSimulation ? paperSimulation.time_in_force : null)
    ?? (payload ? payload.time_in_force : null)
    ?? (payload ? payload.order_type : null)
  )
  const timeInForce = timeInForceRaw ? timeInForceRaw.replace(/\s+/g, '').toUpperCase() : null
  const postOnly = Boolean(
    (legPayload ? legPayload.post_only : null)
    ?? (paperSimulation ? paperSimulation.post_only : null)
    ?? (payload ? payload.post_only : null)
  )
  const priceResolution = normalizeExecutionToken(payload ? payload.price_resolution : null)

  let executionMode: 'LIMIT' | 'MARKET' | null = null
  let liquidityRole: 'MAKER' | 'TAKER' | null = null

  if (pricePolicy === 'market' || pricePolicy === 'marketable' || pricePolicy === 'aggressive') {
    executionMode = 'MARKET'
    liquidityRole = 'TAKER'
  } else if (pricePolicy === 'taker_limit' || pricePolicy === 'taker') {
    executionMode = 'LIMIT'
    liquidityRole = 'TAKER'
  } else if (pricePolicy === 'maker_limit' || pricePolicy === 'maker' || pricePolicy === 'post_only') {
    executionMode = 'LIMIT'
    liquidityRole = 'MAKER'
  } else if (pricePolicy) {
    executionMode = 'LIMIT'
  } else if (priceResolution === 'explicit_limit' || timeInForce) {
    executionMode = 'LIMIT'
  }

  if (!executionMode) return '—'

  const parts: string[] = [executionMode]
  if (liquidityRole) parts.push(liquidityRole)
  if (postOnly) parts.push('POST')
  if (timeInForce) parts.push(timeInForce)
  return parts.join(' · ')
}

function summarizeExecutionTypes(labels: Iterable<string>): string {
  const unique = Array.from(
    new Set(Array.from(labels).filter((label) => Boolean(label) && label !== '—'))
  )
  if (unique.length === 0) return '—'
  if (unique.length <= 2) return unique.join(' | ')
  return `${unique.slice(0, 2).join(' | ')} +${unique.length - 2}`
}

function eventReasonDetail(event: TraderEvent): string {
  const payload = isRecord(event.payload) ? event.payload : null
  return (
    cleanText(event.message)
    || (payload ? cleanText(payload.reason) : null)
    || (payload ? cleanText(payload.error_message) : null)
    || (payload ? cleanText(payload.message) : null)
    || 'No message provided'
  )
}

function formatLatencyMs(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return `${Math.max(0, Math.round(value))}ms`
}

function eventLatencyDetail(event: TraderEvent): string | null {
  if (String(event.event_type || '').trim().toLowerCase() !== 'execution_latency') return null
  const payload = isRecord(event.payload) ? event.payload : null
  const latency = payload && isRecord(payload.latency) ? payload.latency : null
  if (!latency) return null
  const parts = [
    formatLatencyMs(latency.emit_to_now_ms),
    formatLatencyMs(latency.ingest_to_now_ms),
    formatLatencyMs(latency.decision_to_submit_ms),
  ]
  const labels = ['emit', 'ingest', 'submit']
  const detail = parts
    .map((part, index) => (part ? `${labels[index]}=${part}` : null))
    .filter((part): part is string => Boolean(part))
  return detail.length > 0 ? detail.join(' | ') : null
}

function latencyStagePercentiles(
  bucket: unknown,
  stageKey: ExecutionLatencyStageKey
): { p95: number | null; p99: number | null } {
  const stage = isRecord(bucket) && isRecord(bucket[stageKey]) ? bucket[stageKey] : null
  return {
    p95: stage ? toNumber(stage.p95) : null,
    p99: stage ? toNumber(stage.p99) : null,
  }
}

function formatLatencyPercentilePair(bucket: unknown, stageKey: ExecutionLatencyStageKey): string {
  const { p95, p99 } = latencyStagePercentiles(bucket, stageKey)
  const p95Label = formatLatencyMs(p95)
  const p99Label = formatLatencyMs(p99)
  if (!p95Label && !p99Label) return '—'
  if (!p99Label) return `p95 ${p95Label}`
  if (!p95Label) return `p99 ${p99Label}`
  return `${p95Label}/${p99Label}`
}

function formatLatencyWindow(seconds: number | null | undefined): string {
  const value = toNumber(seconds)
  if (value === null || value <= 0) return '—'
  if (value % 3600 === 0) return `${value / 3600}h`
  if (value % 60 === 0) return `${value / 60}m`
  return `${value}s`
}

function worstLatencyGroup(
  summary: ExecutionLatencySummary | null | undefined,
  groupKey: 'by_source' | 'by_strategy' | 'by_trader',
  stageKey: ExecutionLatencyStageKey
): { label: string | null; p95: number | null; p99: number | null } {
  const groups = summary?.[groupKey]
  if (!groups || typeof groups !== 'object') {
    return { label: null, p95: null, p99: null }
  }

  let bestLabel: string | null = null
  let bestP95: number | null = null
  let bestP99: number | null = null
  for (const [label, bucket] of Object.entries(groups)) {
    const { p95, p99 } = latencyStagePercentiles(bucket, stageKey)
    if (p95 === null || (bestP95 !== null && p95 <= bestP95)) continue
    bestLabel = label
    bestP95 = p95
    bestP99 = p99
  }
  return { label: bestLabel, p95: bestP95, p99: bestP99 }
}

function formatStrategyVersionLabel(value: number | null): string {
  return value === null ? 'Latest' : `v${value}`
}

function buildPerformanceSectionKey(
  sourceKey: string,
  strategyKey: string,
  strategyVersion: number | null,
): string {
  return `${normalizeSourceKey(sourceKey)}:${normalizeStrategyKey(strategyKey)}:${strategyVersion === null ? 'latest' : `v${strategyVersion}`}`
}

function buildPerformanceSectionLabel(
  sourceLabel: string,
  strategyLabel: string,
  strategyVersionLabel: string,
): string {
  return `${sourceLabel} · ${strategyLabel} · ${strategyVersionLabel}`
}

function buildLatencyGroupRows(
  summary: ExecutionLatencySummary | null | undefined,
  groupKey: 'by_source' | 'by_strategy',
  stageKey: ExecutionLatencyStageKey,
  sourceCatalog: TraderSource[],
): LatencyGroupRow[] {
  const groups = summary?.[groupKey]
  if (!groups || typeof groups !== 'object') return []

  return Object.entries(groups)
    .map(([key, bucket]) => {
      const { p95 } = latencyStagePercentiles(bucket, stageKey)
      const sourceLabel = sourceCatalog.find((item) => normalizeSourceKey(item.key) === normalizeSourceKey(key))?.label
      return {
        key,
        label: groupKey === 'by_strategy'
          ? strategyLabelForKey(key, sourceCatalog)
          : sourceLabel || key.toUpperCase(),
        count: Math.max(0, Math.trunc(toNumber(isRecord(bucket) ? bucket.count : 0))),
        latencyLabel: formatLatencyPercentilePair(bucket, stageKey),
        p95: p95 ?? -1,
      }
    })
    .sort((left, right) => {
      if (left.p95 !== right.p95) return right.p95 - left.p95
      if (left.count !== right.count) return right.count - left.count
      return left.label.localeCompare(right.label)
    })
    .map(({ p95: _p95, ...row }) => row)
}

function parseJsonObject(text: string): { value: Record<string, unknown> | null; error: string | null } {
  try {
    const parsed: unknown = JSON.parse(text || '{}')
    if (!isRecord(parsed)) {
      return {
        value: null,
        error: 'Must be a JSON object.',
      }
    }
    return { value: parsed, error: null }
  } catch (error) {
    return { value: null, error: error instanceof Error ? error.message : 'Invalid JSON' }
  }
}

function parseTraderDeleteLiveExposure(error: unknown): { message: string; summary: string } | null {
  if (typeof error !== 'object' || error === null || !('response' in error)) return null
  const maybeResponse = (error as { response?: { data?: unknown } }).response
  if (!maybeResponse || typeof maybeResponse !== 'object') return null
  const responseData = maybeResponse.data
  if (!isRecord(responseData)) return null
  const detail = responseData.detail
  if (!isRecord(detail)) return null
  if (String(detail.code || '') !== 'open_live_exposure') return null

  const message = cleanText(detail.message) || 'Trader has live exposure.'
  const livePositions = Math.max(0, Math.trunc(toNumber(detail.open_live_positions)))
  const shadowPositions = Math.max(0, Math.trunc(toNumber(detail.open_shadow_positions)))
  const liveOrders = Math.max(0, Math.trunc(toNumber(detail.open_live_orders)))
  const shadowOrders = Math.max(0, Math.trunc(toNumber(detail.open_shadow_orders)))
  const otherPositions = Math.max(0, Math.trunc(toNumber(detail.open_other_positions)))
  const otherOrders = Math.max(0, Math.trunc(toNumber(detail.open_other_orders)))
  const parts: string[] = []
  if (livePositions > 0) parts.push(`${livePositions} live position(s)`)
  if (shadowPositions > 0) parts.push(`${shadowPositions} shadow position(s)`)
  if (liveOrders > 0) parts.push(`${liveOrders} live active order(s)`)
  if (shadowOrders > 0) parts.push(`${shadowOrders} shadow active order(s)`)
  if (otherPositions > 0) parts.push(`${otherPositions} unknown position(s)`)
  if (otherOrders > 0) parts.push(`${otherOrders} unknown active order(s)`)
  return {
    message,
    summary: parts.join(' • '),
  }
}

function errorMessage(error: unknown, fallback: string): string {
  if (typeof error === 'object' && error !== null && 'response' in error) {
    const maybeResponse = (error as { response?: { data?: unknown } }).response
    const data = maybeResponse?.data
    if (typeof data === 'string') return data
    if (typeof data === 'object' && data !== null) {
      const detail = (data as { detail?: unknown }).detail
      if (typeof detail === 'string') return detail
      if (Array.isArray(detail)) {
        const messages = detail
          .map((item) => {
            if (typeof item === 'string') return item
            if (typeof item !== 'object' || item === null) return ''
            const msg = (item as { msg?: unknown }).msg
            return typeof msg === 'string' ? msg : ''
          })
          .filter((item) => item.length > 0)
        if (messages.length > 0) return messages.join('; ')
      }
      if (typeof detail === 'object' && detail !== null && 'message' in detail) {
        const message = (detail as { message?: unknown }).message
        if (typeof message === 'string') return message
      }
    }
  }
  if (error instanceof Error) return error.message || fallback
  return fallback
}

function toBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase()
    if (lowered === 'true' || lowered === '1' || lowered === 'yes') return true
    if (lowered === 'false' || lowered === '0' || lowered === 'no') return false
  }
  return fallback
}

function csvToList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function upsertTraderRows(rows: Trader[] | undefined, trader: Trader): Trader[] {
  const current = Array.isArray(rows) ? rows : []
  const next = current.filter((row) => row.id !== trader.id)
  next.push(trader)
  next.sort((left, right) => left.name.localeCompare(right.name))
  return next
}

function normalizeTuneList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item || '').trim())
      .filter((item) => item.length > 0)
  }
  if (typeof value !== 'string') return []

  const compact = value.trim()
  if (!compact) return []

  const parts = compact
    .split(/\r?\n+/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.replace(/^\d+[\).]\s*/, '').replace(/^[-*]\s*/, '').trim())
    .filter((line) => line.length > 0)
  return parts.length > 0 ? parts : [compact]
}

function normalizeSourceKey(value: string): string {
  const key = String(value || '').trim().toLowerCase()
  return key
}

function uniqueSourceList(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const trimmed = String(value || '').trim()
    const normalized = normalizeSourceKey(trimmed)
    if (!trimmed || !normalized || seen.has(normalized)) continue
    seen.add(normalized)
    out.push(trimmed)
  }
  return out
}

function normalizeStrategyKey(value: unknown): string {
  const key = String(value || '').trim().toLowerCase()
  return key || DEFAULT_STRATEGY_KEY
}

function normalizeStrategyKeyForSource(sourceKey: string, value: unknown): string {
  const normalizedSource = normalizeSourceKey(sourceKey)
  const key = normalizeStrategyKey(value)
  return key || DEFAULT_STRATEGY_BY_SOURCE[normalizedSource] || DEFAULT_STRATEGY_KEY
}

function normalizeStrategyVersion(value: unknown): number | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return Math.trunc(value)
  }
  const raw = String(value || '').trim().toLowerCase()
  if (!raw || raw === 'latest') return null
  const parsed = Number(raw.startsWith('v') ? raw.slice(1) : raw)
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return Math.trunc(parsed)
}

function normalizeVersionList(value: unknown): number[] {
  const raw = Array.isArray(value) ? value : []
  const seen = new Set<number>()
  const out: number[] = []
  for (const item of raw) {
    const normalized = normalizeStrategyVersion(item)
    if (normalized == null || seen.has(normalized)) continue
    seen.add(normalized)
    out.push(normalized)
  }
  out.sort((left, right) => right - left)
  return out
}

function normalizeTradingScheduleDay(value: unknown): TradingScheduleDay | null {
  const token = String(value || '').trim().toLowerCase()
  if (token.startsWith('mon')) return 'mon'
  if (token.startsWith('tue')) return 'tue'
  if (token.startsWith('wed')) return 'wed'
  if (token.startsWith('thu')) return 'thu'
  if (token.startsWith('fri')) return 'fri'
  if (token.startsWith('sat')) return 'sat'
  if (token.startsWith('sun')) return 'sun'
  return null
}

function normalizeTradingScheduleDays(value: unknown): TradingScheduleDay[] {
  const raw = Array.isArray(value) ? value : []
  const next: TradingScheduleDay[] = []
  const seen = new Set<TradingScheduleDay>()
  for (const item of raw) {
    const normalized = normalizeTradingScheduleDay(item)
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    next.push(normalized)
  }
  return next.length > 0 ? next : [...TRADING_SCHEDULE_DAYS]
}

function normalizeTradingScheduleDraft(value: unknown): TradingScheduleDraft {
  const raw = isRecord(value) ? value : {}
  const normalizeDate = (input: unknown): string => {
    const text = String(input || '').trim()
    if (!text) return ''
    const datePart = text.includes('T') ? text.slice(0, 10) : text
    const parsed = new Date(`${datePart}T00:00:00Z`)
    if (Number.isNaN(parsed.getTime())) return ''
    return parsed.toISOString().slice(0, 10)
  }
  return {
    enabled: toBoolean(raw.enabled, false),
    days: normalizeTradingScheduleDays(raw.days),
    startTimeUtc: String(raw.start_time || '00:00'),
    endTimeUtc: String(raw.end_time || '23:59'),
    startDateUtc: normalizeDate(raw.start_date),
    endDateUtc: normalizeDate(raw.end_date),
    endAtUtc: String(raw.end_at || '').trim(),
  }
}

function buildTradingScheduleMetadata(schedule: TradingScheduleDraft): Record<string, unknown> {
  return {
    enabled: Boolean(schedule.enabled),
    days: normalizeTradingScheduleDays(schedule.days),
    start_time: String(schedule.startTimeUtc || '00:00'),
    end_time: String(schedule.endTimeUtc || '23:59'),
    start_date: schedule.startDateUtc || null,
    end_date: schedule.endDateUtc || null,
    end_at: schedule.endAtUtc || null,
  }
}

function strategyLabelForKey(key: string, sourceCatalog: TraderSource[] = []): string {
  const normalized = String(key || '').trim().toLowerCase()
  for (const source of sourceCatalog) {
    const option = (source.strategy_options || [])
      .find((item) => String(item.key || '').trim().toLowerCase() === normalized)
    if (option?.label) {
      return String(option.label)
    }
  }
  return STRATEGY_LABELS[normalized] || normalized || key
}

function isCryptoSourceKey(key: string): boolean {
  const k = normalizeSourceKey(key)
  return k === 'crypto'
}

function sourceStrategyDetails(source: TraderSource): StrategyOptionDetail[] {
  const options = (source.strategy_options || [])
    .filter((item) => item && typeof item === 'object')
    .map((item) => {
      const key = String(item.key || '').trim().toLowerCase()
      const version = normalizeStrategyVersion(item.version)
      const latestVersion = normalizeStrategyVersion(item.latest_version) ?? version
      const versions = normalizeVersionList(item.versions)
      if (latestVersion != null && !versions.includes(latestVersion)) {
        versions.unshift(latestVersion)
      }
      return {
        key,
        label: String(item.label || strategyLabelForKey(key)),
        defaultParams: isRecord(item.default_params) ? { ...item.default_params } : {},
        paramFields: Array.isArray(item.param_fields)
          ? item.param_fields.filter((field): field is Record<string, unknown> => isRecord(field))
          : [],
        version,
        latestVersion,
        versions,
      }
    })
    .filter((item) => item.key)
  if (options.length > 0) return options
  const fallback =
    source.default_strategy_key ||
    DEFAULT_STRATEGY_BY_SOURCE[normalizeSourceKey(source.key)]
  if (fallback) {
    const normalizedFallback = normalizeStrategyKeyForSource(source.key, fallback)
    return [{
      key: normalizedFallback,
      label: strategyLabelForKey(normalizedFallback, [source]),
      defaultParams: {},
      paramFields: [],
      version: 1,
      latestVersion: 1,
      versions: [1],
    }]
  }
  return []
}

function sourceStrategyOptions(source: TraderSource): StrategyOption[] {
  return sourceStrategyDetails(source).map((item) => ({ key: item.key, label: item.label }))
}

function defaultStrategyForSource(sourceKey: string, sourceCatalog: TraderSource[]): string {
  const normalized = normalizeSourceKey(sourceKey)
  const source = sourceCatalog.find((item) => normalizeSourceKey(item.key) === normalized)
  const options = source ? sourceStrategyOptions(source) : []
  const preferred = source ? normalizeStrategyKeyForSource(normalized, source.default_strategy_key) : ''
  if (preferred && options.some((option) => option.key === preferred)) {
    return preferred
  }
  if (options.length > 0) return options[0].key
  return DEFAULT_STRATEGY_BY_SOURCE[normalized] || DEFAULT_STRATEGY_KEY
}

function normalizeTradersScopeConfig(value: unknown): {
  modes: TradersScopeMode[]
  individual_wallets: string[]
  group_ids: string[]
} {
  const raw = isRecord(value) ? value : {}
  const modes: TradersScopeMode[] = []
  const seenModes = new Set<TradersScopeMode>()
  for (const rawMode of toStringList(raw.modes)) {
    const mode = String(rawMode || '').trim().toLowerCase()
    if (mode !== 'tracked' && mode !== 'pool' && mode !== 'individual' && mode !== 'group') continue
    if (seenModes.has(mode)) continue
    seenModes.add(mode)
    modes.push(mode)
  }
  const individual_wallets: string[] = []
  const seenWallets = new Set<string>()
  for (const rawWallet of toStringList(raw.individual_wallets)) {
    const wallet = String(rawWallet || '').trim().toLowerCase()
    if (!wallet || seenWallets.has(wallet)) continue
    seenWallets.add(wallet)
    individual_wallets.push(wallet)
  }
  const group_ids: string[] = []
  const seenGroups = new Set<string>()
  for (const rawGroupId of toStringList(raw.group_ids)) {
    const groupId = String(rawGroupId || '').trim()
    if (!groupId || seenGroups.has(groupId)) continue
    seenGroups.add(groupId)
    group_ids.push(groupId)
  }
  return {
    modes: modes.length > 0 ? modes : ['tracked', 'pool'],
    individual_wallets,
    group_ids,
  }
}

function buildSourceStrategyParams(
  raw: Record<string, unknown>,
  sourceKey: string,
  strategyDetail: StrategyOptionDetail | null
): Record<string, unknown> {
  const strategyDefaults = isRecord(strategyDetail?.defaultParams)
    ? (strategyDetail.defaultParams as Record<string, unknown>)
    : {}
  const next: Record<string, unknown> = { ...strategyDefaults, ...raw }
  if (normalizeSourceKey(sourceKey) === 'traders') {
    next.traders_scope = normalizeTradersScopeConfig(next.traders_scope)
    return next
  }
  delete next.traders_scope
  return next
}

function traderSourceKeys(trader: Trader): string[] {
  if (Array.isArray(trader.source_configs) && trader.source_configs.length > 0) {
    const seen = new Set<string>()
    const out: string[] = []
    for (const sourceConfig of trader.source_configs) {
      const sourceKey = normalizeSourceKey(String(sourceConfig.source_key || ''))
      if (!sourceKey || seen.has(sourceKey)) continue
      seen.add(sourceKey)
      out.push(sourceKey)
    }
    return out
  }
  return []
}

function isTradersCopyTradeSourceConfig(sourceConfig: TraderSourceConfig | null | undefined): boolean {
  if (!sourceConfig) return false
  const sourceKey = normalizeSourceKey(String(sourceConfig.source_key || ''))
  const strategyKey = String(sourceConfig.strategy_key || '').trim().toLowerCase()
  return sourceKey === 'traders' && strategyKey === 'traders_copy_trade'
}

function traderHasCopyTradeSource(trader: Trader | null | undefined): boolean {
  if (!trader || !Array.isArray(trader.source_configs)) return false
  return trader.source_configs.some((sourceConfig) => isTradersCopyTradeSourceConfig(sourceConfig))
}

function traderCopyExistingOnStartDefault(trader: Trader | null | undefined): boolean {
  if (!trader || !Array.isArray(trader.source_configs)) return false
  for (const sourceConfig of trader.source_configs) {
    if (!isTradersCopyTradeSourceConfig(sourceConfig)) continue
    const params = isRecord(sourceConfig.strategy_params)
      ? (sourceConfig.strategy_params as Record<string, unknown>)
      : {}
    if (toBoolean(params.copy_existing_positions_on_start, false)) {
      return true
    }
  }
  return false
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function buildPositionBookRows(
  orders: TraderOrder[],
  traderNameById: Record<string, string>,
  decisionSignalPayloadByDecisionId: Map<string, Record<string, unknown>>
): PositionBookRow[] {
  const buckets = new Map<string, {
    traderId: string
    traderName: string
    marketId: string
    marketAliases: Set<string>
    marketQuestion: string
    sources: Set<string>
    executionTypes: Set<string>
    direction: string
    directionSide: DirectionSide | null
    exposureUsd: number
    weightedPrice: number
    weightedMark: number
    markWeight: number
    weightedEdge: number
    edgeWeight: number
    weightedConfidence: number
    confidenceWeight: number
    unrealizedPnl: number
    hasUnrealizedPnl: boolean
    orderCount: number
    liveOrderCount: number
    shadowOrderCount: number
    markUpdatedAt: string | null
    lastUpdated: string | null
    statuses: Set<string>
    polymarketLink: string | null
    kalshiLink: string | null
  }>()

  for (const order of orders) {
    const status = normalizeStatus(order.status)
    if (!OPEN_ORDER_STATUSES.has(status)) continue

    const traderId = String(order.trader_id || 'unknown')
    const marketId = String(order.market_id || 'unknown')
    const orderPayload = isRecord(order.payload) ? order.payload : {}
    const linkedDecisionId = String(order.decision_id || '').trim()
    const signalPayload = linkedDecisionId
      ? decisionSignalPayloadByDecisionId.get(linkedDecisionId) || null
      : null
    const directionPresentation = resolveOrderDirectionPresentation(order)
    const directionKey = directionPresentation.side || directionPresentation.label.toUpperCase() || 'UNKNOWN'
    const key = `${traderId}:${marketId}:${directionKey}`
    const orderAliases = collectOrderMarketAliasIds(order)
    const positionState = isRecord(orderPayload.position_state) ? orderPayload.position_state : {}
    const markPrice = toNumber(
      order.current_price
      ?? positionState.last_mark_price
      ?? orderPayload.market_price
      ?? orderPayload.resolved_price
    )
    const filledSize = toNumber(
      order.filled_shares
      ?? orderPayload.filled_size
      ?? positionState.filled_size
    )
    const filledNotional = toNumber(
      order.filled_notional_usd
      ?? order.notional_usd
    )
    let unrealizedPnl: number | null = null
    if (order.unrealized_pnl !== null && order.unrealized_pnl !== undefined) {
      unrealizedPnl = toNumber(order.unrealized_pnl)
    } else if (markPrice > 0 && filledSize > 0 && filledNotional > 0) {
      unrealizedPnl = (markPrice * filledSize) - filledNotional
    }
    const notional = Math.abs(toNumber(order.notional_usd))
    const px = toNumber(order.effective_price ?? order.entry_price)
    const pnl = toNumber(order.actual_profit)
    const edge = computeOrderDynamicEdgePercent({
      status,
      edgePercent: toNumber(order.edge_percent),
      unrealizedPnl: unrealizedPnl,
      realizedPnl: pnl,
      filledNotional,
    })
    const confidence = toNumber(order.confidence)
    const traderName = traderNameById[traderId] || shortId(traderId)
    const mode = normalizeStatus(order.mode)
    const sourceLabel = cleanText(order.source)?.toUpperCase() || 'UNKNOWN'
    const executionSummary = orderExecutionTypeSummary(order)
    const links = buildOrderMarketLinks(order, orderPayload, signalPayload)
    const markUpdatedAt = cleanText(resolveOrderMarketUpdateTimestamp(order, orderPayload))

    if (!buckets.has(key)) {
      buckets.set(key, {
        traderId,
        traderName,
        marketId,
        marketAliases: new Set<string>(),
        marketQuestion: String(order.market_question || order.market_id || 'Unknown market'),
        sources: new Set<string>(),
        executionTypes: new Set<string>(),
        direction: directionPresentation.label,
        directionSide: directionPresentation.side,
        exposureUsd: 0,
        weightedPrice: 0,
        weightedMark: 0,
        markWeight: 0,
        weightedEdge: 0,
        edgeWeight: 0,
        weightedConfidence: 0,
        confidenceWeight: 0,
        unrealizedPnl: 0,
        hasUnrealizedPnl: false,
        orderCount: 0,
        liveOrderCount: 0,
        shadowOrderCount: 0,
        markUpdatedAt: null,
        lastUpdated: null,
        statuses: new Set<string>(),
        polymarketLink: null,
        kalshiLink: null,
      })
    }

    const bucket = buckets.get(key)
    if (!bucket) continue
    for (const alias of orderAliases) {
      bucket.marketAliases.add(alias)
    }

    bucket.exposureUsd += notional
    bucket.weightedPrice += px > 0 && notional > 0 ? px * notional : 0
    bucket.weightedMark += markPrice > 0 && notional > 0 ? markPrice * notional : 0
    bucket.markWeight += markPrice > 0 && notional > 0 ? notional : 0
    bucket.weightedEdge += edge !== 0 && notional > 0 ? edge * notional : 0
    bucket.edgeWeight += edge !== 0 && notional > 0 ? notional : 0
    bucket.weightedConfidence += confidence !== 0 && notional > 0 ? confidence * notional : 0
    bucket.confidenceWeight += confidence !== 0 && notional > 0 ? notional : 0
    if (unrealizedPnl !== null) {
      bucket.unrealizedPnl += unrealizedPnl
      bucket.hasUnrealizedPnl = true
    }
    bucket.orderCount += 1
    if (mode === 'live') {
      bucket.liveOrderCount += 1
    } else if (mode === 'shadow') {
      bucket.shadowOrderCount += 1
    }
    bucket.sources.add(sourceLabel)
    if (executionSummary !== '—') {
      bucket.executionTypes.add(executionSummary)
    }
    bucket.lastUpdated = toTs(markUpdatedAt) > toTs(bucket.lastUpdated)
      ? markUpdatedAt
      : bucket.lastUpdated
    bucket.markUpdatedAt = toTs(markUpdatedAt) > toTs(bucket.markUpdatedAt)
      ? markUpdatedAt
      : bucket.markUpdatedAt
    bucket.statuses.add(status)
    if (!bucket.directionSide && directionPresentation.side) {
      bucket.directionSide = directionPresentation.side
    }
    if (
      directionPresentation.label
      && bucket.direction === (bucket.directionSide || '')
      && directionPresentation.label !== (bucket.directionSide || '')
    ) {
      bucket.direction = directionPresentation.label
    }
    if (!bucket.polymarketLink && links.polymarket) {
      bucket.polymarketLink = links.polymarket
    }
    if (!bucket.kalshiLink && links.kalshi) {
      bucket.kalshiLink = links.kalshi
    }
  }

  return Array.from(buckets.entries())
    .map((entry) => {
      const [key, bucket] = entry
      const markFresh = toTs(bucket.markUpdatedAt) > 0 && (Date.now() - toTs(bucket.markUpdatedAt)) <= 15_000
      return {
        key,
        traderId: bucket.traderId,
        traderName: bucket.traderName,
        marketId: bucket.marketId,
        marketAliases: Array.from(bucket.marketAliases),
        marketQuestion: bucket.marketQuestion,
        sourceSummary: Array.from(bucket.sources).join(', '),
        executionSummary: summarizeExecutionTypes(bucket.executionTypes),
        direction: bucket.direction,
        directionSide: bucket.directionSide,
        exposureUsd: bucket.exposureUsd,
        averagePrice: bucket.exposureUsd > 0 ? bucket.weightedPrice / bucket.exposureUsd : null,
        markPrice: bucket.markWeight > 0 ? bucket.weightedMark / bucket.markWeight : null,
        markUpdatedAt: bucket.markUpdatedAt,
        markFresh,
        unrealizedPnl: bucket.hasUnrealizedPnl ? bucket.unrealizedPnl : null,
        weightedEdge: bucket.edgeWeight > 0 ? bucket.weightedEdge / bucket.edgeWeight : null,
        weightedConfidence: bucket.confidenceWeight > 0 ? bucket.weightedConfidence / bucket.confidenceWeight : null,
        orderCount: bucket.orderCount,
        liveOrderCount: bucket.liveOrderCount,
        shadowOrderCount: bucket.shadowOrderCount,
        lastUpdated: bucket.lastUpdated,
        statusSummary: Array.from(bucket.statuses).join(', '),
        links: {
          polymarket: bucket.polymarketLink,
          kalshi: bucket.kalshiLink,
        },
      }
    })
    .sort((a, b) => b.exposureUsd - a.exposureUsd)
}

function isYesDirection(value: unknown): boolean {
  return normalizeDirectionSide(value) === 'YES'
}

function isNoDirection(value: unknown): boolean {
  return normalizeDirectionSide(value) === 'NO'
}

function compareNullableNumber(
  left: number | null,
  right: number | null,
  sortDirection: PositionSortDirection
): number {
  if (left === null && right === null) return 0
  if (left === null) return 1
  if (right === null) return -1
  return sortDirection === 'asc' ? left - right : right - left
}

function sortPositionRows(
  rows: PositionBookRow[],
  sortField: PositionSortField,
  sortDirection: PositionSortDirection
): PositionBookRow[] {
  const sorted = [...rows]
  sorted.sort((left, right) => {
    if (sortField === 'exposure') {
      return sortDirection === 'asc'
        ? left.exposureUsd - right.exposureUsd
        : right.exposureUsd - left.exposureUsd
    }

    if (sortField === 'updated') {
      const leftTs = toTs(left.lastUpdated || left.markUpdatedAt)
      const rightTs = toTs(right.lastUpdated || right.markUpdatedAt)
      return sortDirection === 'asc' ? leftTs - rightTs : rightTs - leftTs
    }

    if (sortField === 'edge') {
      const delta = compareNullableNumber(left.weightedEdge, right.weightedEdge, sortDirection)
      if (delta !== 0) return delta
      return right.exposureUsd - left.exposureUsd
    }

    if (sortField === 'confidence') {
      const delta = compareNullableNumber(left.weightedConfidence, right.weightedConfidence, sortDirection)
      if (delta !== 0) return delta
      return right.exposureUsd - left.exposureUsd
    }

    const delta = compareNullableNumber(left.unrealizedPnl, right.unrealizedPnl, sortDirection)
    if (delta !== 0) return delta
    return right.exposureUsd - left.exposureUsd
  })
  return sorted
}

function summarizePositionRows(rows: PositionBookRow[]): {
  totalRows: number
  yesRows: number
  noRows: number
  totalExposure: number
  totalUnrealizedPnl: number
  rowsWithUnrealized: number
  avgEdge: number
  avgConfidence: number
  liveOrders: number
  shadowOrders: number
  markedRows: number
  freshMarks: number
} {
  let yesRows = 0
  let noRows = 0
  let totalExposure = 0
  let totalUnrealizedPnl = 0
  let rowsWithUnrealized = 0
  let edgeWeighted = 0
  let edgeWeight = 0
  let confidenceWeighted = 0
  let confidenceWeight = 0
  let liveOrders = 0
  let shadowOrders = 0
  let markedRows = 0
  let freshMarks = 0

  for (const row of rows) {
    totalExposure += row.exposureUsd
    if (isYesDirection(row.directionSide || row.direction)) yesRows += 1
    if (isNoDirection(row.directionSide || row.direction)) noRows += 1
    if (row.unrealizedPnl !== null) {
      totalUnrealizedPnl += row.unrealizedPnl
      rowsWithUnrealized += 1
    }
    if (row.weightedEdge !== null && row.exposureUsd > 0) {
      edgeWeighted += row.weightedEdge * row.exposureUsd
      edgeWeight += row.exposureUsd
    }
    if (row.weightedConfidence !== null && row.exposureUsd > 0) {
      confidenceWeighted += row.weightedConfidence * row.exposureUsd
      confidenceWeight += row.exposureUsd
    }
    liveOrders += row.liveOrderCount
    shadowOrders += row.shadowOrderCount
    if (row.markPrice !== null) markedRows += 1
    if (row.markFresh) freshMarks += 1
  }

  return {
    totalRows: rows.length,
    yesRows,
    noRows,
    totalExposure,
    totalUnrealizedPnl,
    rowsWithUnrealized,
    avgEdge: edgeWeight > 0 ? edgeWeighted / edgeWeight : 0,
    avgConfidence: confidenceWeight > 0 ? confidenceWeighted / confidenceWeight : 0,
    liveOrders,
    shadowOrders,
    markedRows,
    freshMarks,
  }
}

function positionMetaLine(row: PositionBookRow): string {
  const sourceOrStatus = cleanText(row.sourceSummary) || cleanText(row.statusSummary) || 'n/a'
  if (row.executionSummary === '—') return sourceOrStatus
  return `${sourceOrStatus} • ${row.executionSummary}`
}

function BotTradePositionModal({
  market,
  sharedHistory,
  sharedHistoryLoading,
  scope,
  orders,
  themeMode,
  onSell,
  sellPendingOrderId,
  sellError,
  sellSuccess,
  onClose,
}: {
  market: CryptoMarket | null
  sharedHistory: unknown[]
  sharedHistoryLoading: boolean
  scope: BotMarketModalScope
  orders: TraderOrder[]
  themeMode: 'dark' | 'light'
  onSell: (order: TraderOrder) => void
  sellPendingOrderId: string | null
  sellError: string | null
  sellSuccess: string | null
  onClose: () => void
}) {
  const scopeMarketIds = useMemo(
    () => new Set(
      collectMarketAliases([
        scope.marketId,
        ...(Array.isArray(scope.marketIds) ? scope.marketIds : []),
      ])
    ),
    [scope.marketId, scope.marketIds]
  )

  const relatedOrders = useMemo(() => {
    const filtered = orders.filter((order) => {
      if (scope.traderId && String(order.trader_id || '') !== scope.traderId) return false
      const matchesScopeIds = collectOrderMarketAliasIds(order).some((alias) => scopeMarketIds.has(alias))
      if (
        !marketMatchesCryptoIdentity(order.market_id, market)
        && !matchesScopeIds
      ) {
        return false
      }
      if (!scope.directionSide) return true
      const side = resolveOrderDirectionPresentation(order).side
      return !side || side === scope.directionSide
    })
    filtered.sort((left, right) => {
      const leftTs = Math.max(toTs(left.updated_at), toTs(left.executed_at), toTs(left.created_at))
      const rightTs = Math.max(toTs(right.updated_at), toTs(right.executed_at), toTs(right.created_at))
      return rightTs - leftTs
    })
    return filtered
  }, [
    market,
    orders,
    scope.directionSide,
    scope.marketId,
    scope.marketIds,
    scope.traderId,
    scopeMarketIds,
  ])

  const anchorOrder = useMemo(() => {
    if (!scope.anchorOrderId) return relatedOrders[0] || null
    return relatedOrders.find((order) => order.id === scope.anchorOrderId) || relatedOrders[0] || null
  }, [relatedOrders, scope.anchorOrderId])

  const scopedOrders = scope.kind === 'trade' && anchorOrder ? [anchorOrder] : relatedOrders
  const anchorSnapshot = useMemo(
    () => (anchorOrder ? resolveOrderModalSnapshot(anchorOrder) : null),
    [anchorOrder]
  )
  const canSellAnchorOrder = Boolean(
    scope.kind === 'trade'
    && anchorOrder
    && scope.traderId
    && anchorSnapshot
    && OPEN_ORDER_STATUSES.has(anchorSnapshot.status)
  )

  const metrics = useMemo(() => {
    const snapshots = scopedOrders.map((order) => resolveOrderModalSnapshot(order))
    let totalExposure = 0
    let openExposure = 0
    let resolvedExposure = 0
    let openFilledNotional = 0
    let resolvedFilledNotional = 0
    let livePnl = 0
    let realizedPnl = 0
    let openCount = 0
    let resolvedCount = 0
    let failedCount = 0
    let liveOrderCount = 0
    let shadowOrderCount = 0
    let weightedEntry = 0
    let weightedEntryWeight = 0
    let weightedMark = 0
    let weightedMarkWeight = 0
    let weightedEdge = 0
    let edgeWeight = 0
    let weightedConfidence = 0
    let confidenceWeight = 0
    let openedAt: string | null = null
    let updatedAt: string | null = null
    const sourceSet = new Set<string>()
    const modeSet = new Set<string>()
    const statusSet = new Set<string>()

    for (const snapshot of snapshots) {
      const basis = snapshot.filledNotionalUsd > 0 ? snapshot.filledNotionalUsd : snapshot.notionalUsd
      const status = snapshot.status
      totalExposure += snapshot.notionalUsd
      sourceSet.add(snapshot.source)
      modeSet.add(snapshot.mode)
      statusSet.add(status)

      if (snapshot.mode.toLowerCase() === 'live') liveOrderCount += 1
      if (snapshot.mode.toLowerCase() === 'shadow') shadowOrderCount += 1

      if (basis > 0 && snapshot.entryPrice !== null) {
        weightedEntry += snapshot.entryPrice * basis
        weightedEntryWeight += basis
      }
      if (basis > 0 && snapshot.markPrice !== null) {
        weightedMark += snapshot.markPrice * basis
        weightedMarkWeight += basis
      }
      if (snapshot.edgePercent !== null && basis > 0) {
        weightedEdge += snapshot.edgePercent * basis
        edgeWeight += basis
      }
      if (snapshot.confidencePercent !== null && basis > 0) {
        weightedConfidence += snapshot.confidencePercent * basis
        confidenceWeight += basis
      }

      if (OPEN_ORDER_STATUSES.has(status)) {
        openCount += 1
        openExposure += snapshot.notionalUsd
        openFilledNotional += basis
        if (snapshot.unrealizedPnl !== null) livePnl += snapshot.unrealizedPnl
      } else if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolvedCount += 1
        resolvedExposure += snapshot.notionalUsd
        resolvedFilledNotional += basis
        realizedPnl += snapshot.realizedPnl
      } else if (FAILED_ORDER_STATUSES.has(status)) {
        failedCount += 1
      }

      const candidateOpenedAt = snapshot.createdAt
      if (candidateOpenedAt && (toTs(candidateOpenedAt) < toTs(openedAt) || !openedAt)) {
        openedAt = candidateOpenedAt
      }
      const candidateUpdatedAt = snapshot.updatedAt
      if (candidateUpdatedAt && (toTs(candidateUpdatedAt) > toTs(updatedAt) || !updatedAt)) {
        updatedAt = candidateUpdatedAt
      }
    }

    const hasLiveExposure = openCount > 0
    const activePnl = hasLiveExposure
      ? livePnl
      : (resolvedCount > 0 ? realizedPnl : null)
    const returnBasis = hasLiveExposure
      ? (openFilledNotional > 0 ? openFilledNotional : openExposure)
      : (resolvedFilledNotional > 0 ? resolvedFilledNotional : resolvedExposure)
    const returnPercent = activePnl !== null && returnBasis > 0
      ? (activePnl / returnBasis) * 100
      : null

    return {
      orderCount: snapshots.length,
      openCount,
      resolvedCount,
      failedCount,
      liveOrderCount,
      shadowOrderCount,
      exposureUsd: totalExposure,
      entryPrice: weightedEntryWeight > 0 ? weightedEntry / weightedEntryWeight : null,
      markPrice: weightedMarkWeight > 0 ? weightedMark / weightedMarkWeight : null,
      activePnl,
      returnPercent,
      avgEdgePercent: edgeWeight > 0 ? normalizeEdgePercent(weightedEdge / edgeWeight) : null,
      avgConfidencePercent: confidenceWeight > 0 ? normalizeConfidencePercent(weightedConfidence / confidenceWeight) : null,
      sourceSummary: sourceSet.size > 0 ? Array.from(sourceSet).join(', ') : scope.sourceSummary,
      modeSummary: modeSet.size > 0 ? Array.from(modeSet).join(' / ') : scope.modeSummary,
      statusSummary: statusSet.size > 0 ? Array.from(statusSet).join(', ') : scope.statusSummary,
      openedAt,
      updatedAt,
    }
  }, [
    scope.modeSummary,
    scope.sourceSummary,
    scope.statusSummary,
    scopedOrders,
  ])

  const livelineResult = useMemo(
    () => buildBotMarketLivelineSeries({
      sharedHistory,
      historyOrders: relatedOrders,
      directionSide: scope.directionSide,
      markPrice: metrics.markPrice,
      entryPrice: metrics.entryPrice,
      openedAt: metrics.openedAt,
      updatedAt: metrics.updatedAt,
    }),
    [
      metrics.entryPrice,
      metrics.markPrice,
      metrics.openedAt,
      metrics.updatedAt,
      relatedOrders,
      scope.directionSide,
      sharedHistory,
    ]
  )
  const oracleHistoryData = useMemo<LivelinePoint[]>(() => {
    const raw = Array.isArray(market?.oracle_history) ? market.oracle_history : []
    const normalized = raw
      .map((point) => {
        if (!point || typeof point !== 'object') return null
        const row = point as Record<string, unknown>
        const rawTime = toFiniteNumber(row.t ?? row.time)
        const rawValue = toFiniteNumber(row.p ?? row.price)
        if (rawTime === null || rawValue === null) return null
        return {
          time: Math.max(1, toUnixSeconds(rawTime)),
          value: rawValue,
        }
      })
      .filter((point): point is LivelinePoint => point !== null)
      .sort((left, right) => left.time - right.time)

    const deduped: LivelinePoint[] = []
    for (const point of normalized) {
      const previous = deduped[deduped.length - 1]
      if (previous && previous.time === point.time) {
        deduped[deduped.length - 1] = point
      } else {
        deduped.push(point)
      }
    }

    const oracleValue = toFiniteNumber(market?.oracle_price)
    if (oracleValue !== null) {
      const currentRawTime = toFiniteNumber(market?.oracle_updated_at_ms)
      const fallbackTime = currentRawTime !== null ? toUnixSeconds(currentRawTime) : Math.floor(Date.now() / 1000)
      if (deduped.length === 0) {
        deduped.push({ time: Math.max(1, fallbackTime - 1), value: oracleValue })
        deduped.push({ time: Math.max(2, fallbackTime), value: oracleValue })
      } else {
        const last = deduped[deduped.length - 1]
        const pointTime = Math.max(last.time, fallbackTime)
        if (pointTime > last.time) {
          deduped.push({ time: pointTime, value: oracleValue })
        } else if (Math.abs(last.value - oracleValue) > 1e-9) {
          deduped[deduped.length - 1] = { time: last.time, value: oracleValue }
        }
      }
    }

    if (deduped.length < 2) return []
    return deduped.length <= 600 ? deduped : deduped.slice(deduped.length - 600)
  }, [market?.oracle_history, market?.oracle_price, market?.oracle_updated_at_ms])
  const useOracleSeries = livelineResult.primary.length < 2 && oracleHistoryData.length >= 2
  const livelineData = useOracleSeries ? oracleHistoryData : livelineResult.primary
  const oracleValue = toFiniteNumber(market?.oracle_price)
  const livelineValue = useOracleSeries
    ? (
      oracleValue
      ?? oracleHistoryData[oracleHistoryData.length - 1]?.value
      ?? 0
    )
    : (
      toFiniteNumber(metrics.markPrice ?? metrics.entryPrice)
      ?? livelineData[livelineData.length - 1]?.value
      ?? 0
    )
  const isDark = themeMode === 'dark'
  const yesSeriesLabel = scope.yesLabel || 'Yes'
  const noSeriesLabel = scope.noLabel || 'No'
  const priceToBeat = toFiniteNumber(market?.price_to_beat)
  const pnlPositive = (metrics.activePnl ?? 0) >= 0
  const colorByPriceToBeat = priceToBeat !== null && oracleValue !== null
  const lineColor = (
    colorByPriceToBeat
      ? (
        oracleValue >= priceToBeat
          ? (isDark ? '#22c55e' : '#16a34a')
          : (isDark ? '#f87171' : '#dc2626')
      )
      : (
        pnlPositive
          ? (isDark ? '#22c55e' : '#16a34a')
          : (isDark ? '#f87171' : '#dc2626')
      )
  )
  const complementColor = isDark ? '#64748b' : '#94a3b8'
  const complementValue = livelineResult.complement.length > 0
    ? livelineResult.complement[livelineResult.complement.length - 1].value
    : 0
  const oracleSourceSeries = useMemo<LivelineSeries[]>(() => {
    if (!useOracleSeries) return []
    const sourceMap = market?.oracle_prices_by_source
    if (!sourceMap || typeof sourceMap !== 'object') return []
    if (livelineData.length < 2) return []
    const entries = Object.entries(sourceMap)
      .map(([sourceKey, rawSnapshot]) => {
        if (!rawSnapshot || typeof rawSnapshot !== 'object') return null
        const snapshot = rawSnapshot as Record<string, unknown>
        const resolvedSource = String(snapshot.source || sourceKey || '').trim()
        const value = toFiniteNumber(snapshot.price)
        if (!resolvedSource || value === null) return null
        return {
          key: resolvedSource.toLowerCase(),
          label: formatSeriesLabel(resolvedSource),
          value,
        }
      })
      .filter((row): row is { key: string; label: string; value: number } => row !== null)

    if (entries.length < 2) return []

    const primarySourceKey = String(market?.oracle_source || '').trim().toLowerCase()
    const filteredEntries = entries
      .filter((entry) => !primarySourceKey || entry.key !== primarySourceKey)
      .sort((left, right) => left.label.localeCompare(right.label))
      .slice(0, 4)

    if (filteredEntries.length === 0) return []

    const startTime = livelineData[0]?.time || Math.floor(Date.now() / 1000) - 60
    const endTime = livelineData[livelineData.length - 1]?.time || startTime + 60
    const palette = isDark ? BOT_MODAL_SERIES_COLORS_DARK : BOT_MODAL_SERIES_COLORS_LIGHT

    return filteredEntries.map((entry, index) => ({
      id: `oracle-source-${entry.key}`,
      data: buildFlatLivelineSeries(entry.value, startTime, endTime),
      value: entry.value,
      color: palette[index % palette.length],
      label: entry.label,
    }))
  }, [
    isDark,
    livelineData,
    market?.oracle_prices_by_source,
    market?.oracle_source,
    useOracleSeries,
  ])
  const livelineSeries = useMemo<LivelineSeries[]>(() => {
    const series: LivelineSeries[] = []
    if (livelineData.length >= 2) {
      const primaryLabel = (
        useOracleSeries
          ? formatSeriesLabel(String(market?.oracle_source || 'oracle'))
          : scope.directionSide === 'YES'
            ? yesSeriesLabel
            : scope.directionSide === 'NO'
              ? noSeriesLabel
              : 'Primary'
      )
      series.push({
        id: 'primary',
        data: livelineData,
        value: livelineValue,
        color: lineColor,
        label: primaryLabel,
      })
    }
    if (!useOracleSeries && livelineResult.complement.length >= 2) {
      const complementLabel = scope.directionSide === 'YES' ? noSeriesLabel : yesSeriesLabel
      series.push({
        id: 'complement',
        data: livelineResult.complement,
        value: complementValue,
        color: complementColor,
        label: complementLabel,
      })
    }
    if (oracleSourceSeries.length > 0) {
      series.push(...oracleSourceSeries)
    }
    return series
  }, [
    complementColor,
    complementValue,
    lineColor,
    livelineData,
    livelineResult.complement,
    livelineValue,
    oracleSourceSeries,
    market?.oracle_source,
    noSeriesLabel,
    scope.directionSide,
    yesSeriesLabel,
    useOracleSeries,
  ])
  const referencePrice = priceToBeat ?? metrics.entryPrice
  const referenceLabel = (
    priceToBeat !== null
      ? 'Price to beat'
      : metrics.entryPrice !== null
        ? 'Entry'
        : null
  )
  const livelineWindow = Math.max(
    timeframeChartWindowSeconds(market?.timeframe),
    livelineData.length > 1
      ? livelineData[livelineData.length - 1].time - livelineData[0].time
      : 0
  )
  const entryMarkLabel = useOracleSeries ? 'Oracle / Price to beat' : 'Entry / Mark'
  const entryValue = useOracleSeries ? oracleValue : metrics.entryPrice
  const markValue = useOracleSeries ? priceToBeat : metrics.markPrice
  const markUpdateLabel = useOracleSeries ? 'oracle update' : 'mark update'
  const pnlLabel = metrics.openCount > 0 ? 'Live P&L' : metrics.resolvedCount > 0 ? 'Realized P&L' : 'P&L'
  const returnLabel = metrics.openCount > 0 ? 'Live Return' : 'Return'
  const oracleAgeSeconds = toFiniteNumber(market?.oracle_age_seconds)
  const markUpdatedAge = formatRelativeAge(metrics.updatedAt)

  return (
    <Card className="w-[min(1150px,calc(100vw-2rem))] max-h-[90vh] overflow-hidden rounded-2xl border-border/70 bg-background shadow-[0_40px_120px_rgba(0,0,0,0.55)]">
      <div className="border-b border-border/60 px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-1.5">
              <h3 className="text-sm font-semibold truncate max-w-[620px]" title={scope.marketQuestion}>
                {scope.marketQuestion}
              </h3>
              <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                {scope.kind === 'trade' ? 'Trade' : 'Position'}
              </Badge>
              <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                {scope.directionLabel || 'N/A'}
              </Badge>
              <Badge variant="outline" className="h-5 px-1.5 text-[10px] border-border/80 bg-muted/60 text-muted-foreground">
                {scope.traderName}
              </Badge>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {String(market?.asset || 'N/A').toUpperCase()} · {String(market?.timeframe || 'n/a').toUpperCase()} · {metrics.sourceSummary || 'n/a'} · {metrics.modeSummary || 'n/a'}
            </p>
          </div>
          <div className="flex items-center gap-1">
            {canSellAnchorOrder && anchorOrder ? (
              <Button
                type="button"
                size="sm"
                variant="destructive"
                className="h-7 px-2 text-[11px]"
                onClick={() => onSell(anchorOrder)}
                disabled={sellPendingOrderId === String(anchorOrder.id || '')}
              >
                {sellPendingOrderId === String(anchorOrder.id || '') ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                Sell Now
              </Button>
            ) : null}
            {scope.links.polymarket && (
              <a
                href={scope.links.polymarket}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border/60 text-muted-foreground transition-colors hover:text-foreground hover:bg-muted/60"
                title="Open Polymarket market"
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
            {scope.links.kalshi && (
              <a
                href={scope.links.kalshi}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border/60 text-muted-foreground transition-colors hover:text-foreground hover:bg-muted/60"
                title="Open Kalshi market"
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
            <Button type="button" size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </div>

      <div className="max-h-[calc(90vh-72px)] overflow-y-auto px-4 py-3 space-y-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{pnlLabel}</p>
            <p className={cn('text-sm font-mono', (metrics.activePnl ?? 0) > 0 ? 'text-emerald-500' : (metrics.activePnl ?? 0) < 0 ? 'text-red-500' : '')}>
              {formatSignedCurrency(metrics.activePnl)}
            </p>
            <p className="text-[10px] text-muted-foreground">{returnLabel}: {formatSignedPercent(metrics.returnPercent, 2)}</p>
          </div>
          <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Exposure</p>
            <p className="text-sm font-mono">{formatCurrency(metrics.exposureUsd, true)}</p>
            <p className="text-[10px] text-muted-foreground">{metrics.openCount} open · {metrics.resolvedCount} resolved · {metrics.failedCount} failed</p>
          </div>
          <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{entryMarkLabel}</p>
            <p className="text-sm font-mono">
              {entryValue !== null ? entryValue.toFixed(3) : '—'}
              <span className="mx-1 text-muted-foreground">→</span>
              {markValue !== null ? markValue.toFixed(3) : '—'}
            </p>
            <p className="text-[10px] text-muted-foreground">{markUpdateLabel} {markUpdatedAge}</p>
          </div>
          <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Edge / Confidence</p>
            <p className="text-sm font-mono">
              {formatSignedPercent(metrics.avgEdgePercent, 2)}
              <span className="mx-1 text-muted-foreground">·</span>
              {formatSignedPercent(metrics.avgConfidencePercent, 1)}
            </p>
            <p className="text-[10px] text-muted-foreground">{metrics.liveOrderCount} live · {metrics.shadowOrderCount} shadow</p>
          </div>
        </div>

        <div className={cn(
          'rounded-lg border overflow-hidden',
          isDark
            ? 'border-slate-700/40 bg-gradient-to-b from-slate-900/75 via-slate-950/80 to-black/90'
            : 'border-slate-200/90 bg-gradient-to-b from-white via-slate-50 to-slate-100/70',
        )}>
          {sellError ? (
            <div className="border-b border-red-500/30 bg-red-500/10 px-3 py-2 text-[11px] text-red-700 dark:text-red-100">
              {sellError}
            </div>
          ) : null}
          {sellSuccess ? (
            <div className="border-b border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-700 dark:text-emerald-100">
              {sellSuccess}
            </div>
          ) : null}
          {livelineData.length >= 2 ? (
            <Liveline
              data={livelineData}
              value={livelineValue}
              series={livelineSeries.length > 1 ? livelineSeries : undefined}
              color={lineColor}
              theme={isDark ? 'dark' : 'light'}
              showValue
              valueMomentumColor
              grid
              badge
              pulse
              fill={livelineSeries.length <= 1}
              seriesToggleCompact={livelineSeries.length > 1}
              window={livelineWindow > 0 ? livelineWindow : undefined}
              lerpSpeed={0.1}
              padding={{ top: 8, right: 80, bottom: 24, left: 14 }}
              tooltipOutline={isDark}
              formatValue={(value) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
              referenceLine={referencePrice !== null && referenceLabel ? { value: referencePrice, label: referenceLabel } : undefined}
              style={{ height: 280 }}
            />
          ) : (
            <div className="h-[280px] flex items-center justify-center text-xs text-muted-foreground">
              {sharedHistoryLoading ? 'Hydrating shared price history backfill...' : 'Waiting for live price history...'}
            </div>
          )}
        </div>

        <div className="grid gap-2 lg:grid-cols-2">
          <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Lifecycle</p>
            <p className="text-xs mt-0.5">{scope.executionSummary || '—'}</p>
            <p className="text-[10px] text-muted-foreground mt-1">{scope.outcomeSummary || scope.statusSummary || metrics.statusSummary || 'n/a'}</p>
          </div>
          <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Timing & Feed</p>
            <p className="text-xs mt-0.5">
              Opened: {formatTimestamp(metrics.openedAt)}
              <span className="mx-1 text-muted-foreground">·</span>
              Updated: {formatTimestamp(metrics.updatedAt)}
            </p>
            <p className="text-[10px] text-muted-foreground mt-1">
              Oracle age: {oracleAgeSeconds !== null ? `${Math.round(oracleAgeSeconds)}s` : 'n/a'}
            </p>
          </div>
        </div>

        {scopedOrders.length === 0 && (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-[11px] text-amber-700 dark:text-amber-100">
            No matching order rows were found in the current order window for this market/direction scope.
          </div>
        )}
      </div>
    </Card>
  )
}

function normalizePerformanceTimeframe(value: unknown): string | null {
  const normalized = normalizeCryptoTimeframe(value)
  if (normalized) return normalized
  const text = String(value || '').trim().toLowerCase()
  if (!text) return null
  if (text === '5min' || text === '5m' || text === '5') return '5m'
  if (text === '15min' || text === '15m' || text === '15') return '15m'
  if (text === '1hr' || text === '1h' || text === '60m' || text === '60min') return '1h'
  if (text === '4hr' || text === '4h' || text === '240m' || text === '240min') return '4h'
  return text
}

function normalizePerformanceMode(value: unknown): string | null {
  const text = String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_')
  if (!text) return null
  if (text === 'purearb') return 'pure_arb'
  if (text === 'dumphedge') return 'dump_hedge'
  if (text === 'preplacedlimits' || text === 'preplaced') return 'pre_placed_limits'
  if (text === 'directionaledge') return 'directional_edge'
  return text
}

function humanizeStrategyParamLabel(key: string): string {
  return key
    .split('_')
    .filter(Boolean)
    .map((part) => (part.length <= 3 ? part.toUpperCase() : `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`))
    .join(' ')
}

function inferStrategyParamField(key: string, value: unknown): Record<string, unknown> | null {
  const cleanKey = String(key || '').trim()
  if (!cleanKey || cleanKey === '_schema') return null

  let type = 'string'
  if (typeof value === 'boolean') {
    type = 'boolean'
  } else if (typeof value === 'number') {
    type = Number.isInteger(value) ? 'integer' : 'number'
  } else if (Array.isArray(value)) {
    type = 'list'
  } else if (isRecord(value)) {
    type = 'json'
  }

  return {
    key: cleanKey,
    label: humanizeStrategyParamLabel(cleanKey),
    type,
  }
}

function dedupeStrategyParamFields(fields: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = []
  const seen = new Set<string>()
  for (const field of fields) {
    if (!isRecord(field)) continue
    const key = String(field.key || '').trim()
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(field)
  }
  return out
}

function stableSerializePerformanceValue(value: unknown): string {
  if (value === undefined) return 'undefined'
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value.trim())
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'nan'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableSerializePerformanceValue(item)).join(',')}]`
  }
  if (isRecord(value)) {
    const parts = Object.keys(value)
      .sort((left, right) => left.localeCompare(right))
      .map((key) => `${JSON.stringify(key)}:${stableSerializePerformanceValue(value[key])}`)
    return `{${parts.join(',')}}`
  }
  return JSON.stringify(value)
}

function formatPerformanceParamValue(value: unknown): string {
  if (value === undefined || value === null) return 'Not recorded'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'Not recorded'
    if (Math.abs(value) >= 1000) {
      return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
    }
    const rounded = value.toFixed(Math.abs(value) >= 1 ? 2 : 4)
    return rounded.includes('.') ? rounded.replace(/\.?0+$/, '') : rounded
  }
  if (typeof value === 'string') {
    const text = value.trim()
    return text || 'Blank'
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const preview = value.slice(0, 3).map((item) => formatPerformanceParamValue(item)).join(', ')
    return value.length > 3 ? `${preview} +${value.length - 3}` : preview
  }
  if (isRecord(value)) {
    const keys = Object.keys(value)
    if (keys.length === 0) return '{}'
    const preview = keys
      .slice(0, 2)
      .map((key) => `${key}=${formatPerformanceParamValue(value[key])}`)
      .join(', ')
    return keys.length > 2 ? `${preview} +${keys.length - 2}` : preview
  }
  return String(value)
}

function performanceParamBucketMeta(value: unknown): { key: string; label: string; isMissing: boolean } {
  if (value === undefined || value === null) {
    return {
      key: 'not_recorded',
      label: 'Not recorded',
      isMissing: true,
    }
  }
  return {
    key: stableSerializePerformanceValue(value),
    label: formatPerformanceParamValue(value),
    isMissing: false,
  }
}

function mergePerformanceParamRecord(
  target: Record<string, unknown>,
  value: unknown,
): boolean {
  if (!isRecord(value)) return false
  let merged = false
  for (const [rawKey, rawValue] of Object.entries(value)) {
    const key = String(rawKey || '').trim()
    if (!key || key === '_schema') continue
    target[key] = rawValue
    merged = true
  }
  return merged
}

function mergePerformanceParamChildren(
  target: Record<string, unknown>,
  value: unknown,
): boolean {
  if (!isRecord(value)) return false
  let merged = false
  for (const nestedKey of ['strategy_params', 'sub_strategy_params', 'params', 'parameters', 'config', 'effective_strategy_params']) {
    merged = mergePerformanceParamRecord(target, value[nestedKey]) || merged
  }
  return merged
}

function extractOrderPerformanceParams(
  order: TraderOrder,
  decision: Record<string, unknown> | null,
  fallbackParams: Record<string, unknown> | null,
): { params: Record<string, unknown>; usedCurrentConfigFallback: boolean } {
  const params: Record<string, unknown> = {}
  let recorded = false
  const payload = isRecord(order.payload) ? order.payload : null

  if (payload) {
    recorded = mergePerformanceParamRecord(params, payload.strategy_params) || recorded
    recorded = mergePerformanceParamRecord(params, payload.strategy_exit_config) || recorded
    recorded = mergePerformanceParamChildren(params, payload.strategy_context) || recorded
    recorded = mergePerformanceParamChildren(params, payload.signal_strategy_context) || recorded
    recorded = mergePerformanceParamChildren(params, payload.signal_payload) || recorded
    recorded = mergePerformanceParamChildren(params, payload.position_state) || recorded
    recorded = mergePerformanceParamChildren(params, payload.position_close) || recorded
  }

  if (decision) {
    recorded = mergePerformanceParamRecord(params, decision.strategy_params) || recorded
    recorded = mergePerformanceParamChildren(params, decision.payload) || recorded
    recorded = mergePerformanceParamChildren(params, decision.signal_strategy_context) || recorded
    recorded = mergePerformanceParamChildren(params, decision.signal_payload) || recorded
  }

  if (!recorded && fallbackParams && Object.keys(fallbackParams).length > 0) {
    return {
      params: { ...fallbackParams },
      usedCurrentConfigFallback: true,
    }
  }

  return {
    params,
    usedCurrentConfigFallback: false,
  }
}

function _pushRecord(
  target: Array<Record<string, unknown>>,
  value: unknown,
) {
  if (isRecord(value)) target.push(value)
}

function orderPerformanceContexts(
  order: TraderOrder,
  decision: Record<string, unknown> | null,
): Array<Record<string, unknown>> {
  const contexts: Array<Record<string, unknown>> = []
  _pushRecord(contexts, order.payload)

  const payload = isRecord(order.payload) ? order.payload : null
  if (payload) {
    _pushRecord(contexts, payload.strategy_context)
    _pushRecord(contexts, payload.signal_payload)
    _pushRecord(contexts, payload.signal_strategy_context)
    _pushRecord(contexts, payload.live_market)
    _pushRecord(contexts, payload.position_state)
    _pushRecord(contexts, payload.strategy_params)
    _pushRecord(contexts, payload.strategy_exit_config)
    _pushRecord(contexts, payload.position_close)
    if (isRecord(payload.strategy_context)) {
      _pushRecord(contexts, payload.strategy_context.sub_strategy_params)
      _pushRecord(contexts, payload.strategy_context.params)
      _pushRecord(contexts, payload.strategy_context.parameters)
    }
    if (isRecord(payload.signal_strategy_context)) {
      _pushRecord(contexts, payload.signal_strategy_context.sub_strategy_params)
      _pushRecord(contexts, payload.signal_strategy_context.params)
      _pushRecord(contexts, payload.signal_strategy_context.parameters)
    }
  }

  if (decision) {
    _pushRecord(contexts, decision)
    _pushRecord(contexts, decision.signal_payload)
    _pushRecord(contexts, decision.signal_strategy_context)
    _pushRecord(contexts, decision.payload)
    if (isRecord(decision.signal_strategy_context)) {
      _pushRecord(contexts, decision.signal_strategy_context.sub_strategy_params)
      _pushRecord(contexts, decision.signal_strategy_context.params)
      _pushRecord(contexts, decision.signal_strategy_context.parameters)
    }
    if (isRecord(decision.payload)) {
      _pushRecord(contexts, decision.payload.strategy_params)
    }
  }

  return contexts
}

function readPerformanceContextValue(
  contexts: Array<Record<string, unknown>>,
  keys: readonly string[],
  normalize: (value: unknown) => string | null = cleanText,
): string | null {
  for (const context of contexts) {
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(context, key)) continue
      const normalized = normalize(context[key])
      if (normalized) return normalized
    }
  }
  return null
}

function extractOrderPerformanceDimensions(
  order: TraderOrder,
  decision: Record<string, unknown> | null,
): {
  strategyKey: string
  timeframe: string
  mode: string
  subStrategy: string
} {
  const contexts = orderPerformanceContexts(order, decision)
  const strategyKey = (
    cleanText(decision?.strategy_key)
    || readPerformanceContextValue(contexts, ['strategy_key', 'strategy_slug', 'strategy_type', 'strategy'], cleanText)
    || cleanText(order.source)
    || 'unknown'
  ).toLowerCase()
  const timeframe = readPerformanceContextValue(
    contexts,
    ['timeframe', 'cadence', 'interval', 'window'],
    normalizePerformanceTimeframe,
  ) || 'unclassified'
  const mode = readPerformanceContextValue(
    contexts,
    ['active_mode', 'requested_mode', 'strategy_mode', 'mode', 'dominant_mode', 'dominant_strategy'],
    normalizePerformanceMode,
  ) || 'unclassified'
  const subStrategy = readPerformanceContextValue(
    contexts,
    ['sub_strategy', 'dominant_strategy', 'strategy_variant', 'variant'],
    normalizePerformanceMode,
  ) || 'unclassified'
  return {
    strategyKey,
    timeframe,
    mode,
    subStrategy,
  }
}

function performanceBucketSort(left: PerformanceBucketRow, right: PerformanceBucketRow): number {
  if (Math.abs(left.pnl) !== Math.abs(right.pnl)) return Math.abs(right.pnl) - Math.abs(left.pnl)
  if (left.orders !== right.orders) return right.orders - left.orders
  return left.label.localeCompare(right.label)
}

function buildPerformanceBuckets(
  orders: TraderOrder[],
  bucketKeyForOrder: (order: TraderOrder, index: number) => { key: string; label: string },
): PerformanceBucketRow[] {
  const byBucket = new Map<string, PerformanceBucketRow>()
  for (let index = 0; index < orders.length; index += 1) {
    const order = orders[index]
    const bucketMeta = bucketKeyForOrder(order, index)
    const bucketKey = cleanText(bucketMeta.key) || 'unclassified'
    const bucketLabel = cleanText(bucketMeta.label) || bucketKey
    if (!byBucket.has(bucketKey)) {
      byBucket.set(bucketKey, {
        key: bucketKey,
        label: bucketLabel,
        orders: 0,
        open: 0,
        resolved: 0,
        wins: 0,
        losses: 0,
        failed: 0,
        resolvedNotional: 0,
        pnl: 0,
        roiPercent: 0,
        fullLosses: 0,
      })
    }
    const bucket = byBucket.get(bucketKey)
    if (!bucket) continue

    const status = normalizeStatus(order.status)
    const notional = Math.abs(toNumber(order.notional_usd))
    const pnl = toNumber(order.actual_profit)
    bucket.orders += 1

    if (OPEN_ORDER_STATUSES.has(status)) bucket.open += 1
    if (FAILED_ORDER_STATUSES.has(status)) bucket.failed += 1
    if (RESOLVED_ORDER_STATUSES.has(status)) {
      bucket.resolved += 1
      bucket.pnl += pnl
      bucket.resolvedNotional += notional
      if (pnl > 0) bucket.wins += 1
      if (pnl < 0) bucket.losses += 1
      if (pnl < 0 && notional > 0 && Math.abs(pnl) >= notional * 0.98) {
        bucket.fullLosses += 1
      }
    }
  }

  const rows = Array.from(byBucket.values())
  for (const row of rows) {
    row.roiPercent = row.resolvedNotional > 0 ? (row.pnl / row.resolvedNotional) * 100 : 0
  }
  return rows
}

function classifyStrategyParamGroup(fieldKey: string): StrategyParamGroupKey {
  const key = String(fieldKey || '').trim().toLowerCase()
  if (!key) return 'advanced'
  if (
    key.startsWith('strategy_mode')
    || key === 'mode'
    || key === 'traders_scope'
    || key.startsWith('include_')
    || key.startsWith('exclude_')
    || key === 'enabled_sub_strategies'
    || key.includes('sub_strategy')
  ) {
    return 'scope'
  }
  if (
    key.includes('signal_age')
    || key.includes('market_data_age')
    || key.includes('live_context_age')
    || key.includes('oracle_age')
    || key.includes('seconds_left')
    || key.includes('reentry_cooldown')
    || key.includes('freshness')
    || key.includes('timeout')
  ) {
    return 'timing'
  }
  if (
    key.includes('edge')
    || key.includes('confidence')
    || key.includes('liquidity')
    || key.includes('spread')
    || key.includes('imbalance')
    || key.includes('entry_price')
    || key.includes('entry_executable')
    || key.includes('opening_')
    || key.includes('guardrail')
    || key.includes('require_oracle')
  ) {
    return 'entry'
  }
  if (
    key.includes('size')
    || key.includes('sizing')
    || key.includes('notional')
    || key.includes('position')
    || key.includes('multiplier')
    || key.includes('kelly')
    || key.includes('capital')
  ) {
    return 'sizing'
  }
  if (
    key.includes('take_profit')
    || key.includes('stop_loss')
    || key.includes('trailing')
    || key.includes('min_hold')
    || key.includes('max_hold')
    || key.startsWith('rapid_')
    || key.startsWith('reverse_')
    || key.startsWith('underwater_')
    || key.startsWith('force_flatten')
    || key.includes('close_on_inactive')
    || key.includes('resolve_only')
    || key.includes('preplace_take_profit')
    || key.includes('enforce_min_exit_notional')
  ) {
    return 'exit'
  }
  if (key.startsWith('risk') || key.startsWith('max_risk') || key.startsWith('resolution_risk')) {
    return 'risk'
  }
  return 'advanced'
}

function groupStrategyParamFields(fields: Array<Record<string, unknown>>): StrategyParamGroup[] {
  const grouped = new Map<StrategyParamGroupKey, Array<Record<string, unknown>>>()
  for (const field of fields) {
    const fieldKey = String(field.key || '').trim()
    if (!fieldKey) continue
    const groupKey = classifyStrategyParamGroup(fieldKey)
    const current = grouped.get(groupKey) || []
    current.push(field)
    grouped.set(groupKey, current)
  }
  const orderedGroups: StrategyParamGroup[] = []
  for (const groupKey of STRATEGY_PARAM_GROUP_ORDER) {
    const fieldsForGroup = grouped.get(groupKey)
    if (!fieldsForGroup || fieldsForGroup.length === 0) continue
    orderedGroups.push({
      key: groupKey,
      label: STRATEGY_PARAM_GROUP_LABELS[groupKey],
      fields: fieldsForGroup,
    })
  }
  return orderedGroups
}

function FlyoutSection({
  title,
  subtitle,
  icon: Icon,
  count,
  defaultOpen = true,
  iconClassName = 'text-orange-500',
  tone = 'default',
  children,
}: {
  title: string
  subtitle?: string
  icon: any
  count?: string
  defaultOpen?: boolean
  iconClassName?: string
  tone?: 'default' | 'danger'
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <Card
      className={cn(
        'rounded-xl shadow-none overflow-hidden',
        tone === 'danger' ? 'bg-red-500/5 border-red-500/25' : 'bg-card/40 border-border/40'
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className={cn(
          'w-full flex items-center justify-between gap-2 px-3 py-2 transition-colors border-b',
          tone === 'danger'
            ? 'border-red-500/20 hover:bg-red-500/10'
            : 'border-border/40 hover:bg-muted/25'
        )}
      >
        <div className="flex items-center gap-1.5">
          <Icon className={cn('w-3.5 h-3.5', iconClassName)} />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">{title}</h4>
        </div>
        <div className="flex items-center gap-1.5">
          {count ? (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/60 text-muted-foreground">
              {count}
            </span>
          ) : null}
          {open ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />}
        </div>
      </button>
      {open ? (
        <div className="px-3 py-3 space-y-3">
          {subtitle ? <p className="text-[10px] text-muted-foreground/70 -mt-0.5">{subtitle}</p> : null}
          {children}
        </div>
      ) : null}
    </Card>
  )
}

type TradingPanelProps = {
  isConnected?: boolean
}

export default function TradingPanel({ isConnected = false }: TradingPanelProps = {}) {
  const queryClient = useQueryClient()
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const selectedAccountIsLive = Boolean(selectedAccountId?.startsWith('live:'))
  const selectedAccountMode: 'shadow' | 'live' = selectedAccountIsLive ? 'live' : 'shadow'
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const [traderFeedFilter, setTraderFeedFilter] = useState<FeedFilter>('all')
  const [terminalDensity, setTerminalDensity] = useState<TerminalDensity>('compact')
  const [terminalScrollTop, setTerminalScrollTop] = useState(0)
  const [terminalViewportHeight, setTerminalViewportHeight] = useState(0)
  const [tradeStatusFilter, setTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [tradeSearch, setTradeSearch] = useState('')
  const [decisionSearch, setDecisionSearch] = useState('')
  const [decisionOutcomeFilter, setDecisionOutcomeFilter] = useState<DecisionOutcomeFilter>('all')
  const [botRosterSearch, setBotRosterSearch] = useState('')
  const [botRosterHideInactive, setBotRosterHideInactive] = useState(true)
  const [botRosterSort, setBotRosterSort] = useState<BotRosterSort>('name_asc')
  const [botRosterGroupBy, setBotRosterGroupBy] = useState<BotRosterGroupBy>('status')
  const [confirmLiveStartOpen, setConfirmLiveStartOpen] = useState(false)
  const [confirmTraderStartOpen, setConfirmTraderStartOpen] = useState(false)
  const [confirmTraderStopOpen, setConfirmTraderStopOpen] = useState(false)
  const [enableCopyExistingPositions, setEnableCopyExistingPositions] = useState(false)
  const [stopLifecycleMode, setStopLifecycleMode] = useState<TraderStopLifecycleMode>('keep_positions')
  const [stopConfirmLiveClose, setStopConfirmLiveClose] = useState(false)
  const [globalSettingsFlyoutOpen, setGlobalSettingsFlyoutOpen] = useState(false)
  const [globalSettingsSaveError, setGlobalSettingsSaveError] = useState<string | null>(null)
  const [controlActionError, setControlActionError] = useState<string | null>(null)
  const [globalSettingsDraft, setGlobalSettingsDraft] = useState<GlobalSettingsDraft>(() => buildGlobalSettingsDraft(null, null))
  const [workTab, setWorkTab] = useState<'trades' | 'terminal' | 'tune' | 'decisions' | 'performance'>('trades')
  const [performanceSubview, setPerformanceSubview] = useState<PerformanceSubview>('latency')
  const [performanceSectionKey, setPerformanceSectionKey] = useState('')
  const [performanceParamKey, setPerformanceParamKey] = useState('')
  const [allBotsTab, setAllBotsTab] = useState<AllBotsTab>('overview')
  const [allBotsTradeStatusFilter, setAllBotsTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [allBotsTradeSearch, setAllBotsTradeSearch] = useState('')
  const [allBotsPositionSearch, setAllBotsPositionSearch] = useState('')
  const [allBotsPositionDirectionFilter, setAllBotsPositionDirectionFilter] = useState<PositionDirectionFilter>('all')
  const [allBotsPositionSortField, setAllBotsPositionSortField] = useState<PositionSortField>('exposure')
  const [allBotsPositionSortDirection, setAllBotsPositionSortDirection] = useState<PositionSortDirection>('desc')
  const terminalViewportRef = useRef<HTMLDivElement | null>(null)

  const [traderFlyoutOpen, setTraderFlyoutOpen] = useState(false)
  const [traderFlyoutMode, setTraderFlyoutMode] = useState<'create' | 'edit'>('create')
  const [draftName, setDraftName] = useState('')
  const [draftDescription, setDraftDescription] = useState('')
  const [, setDraftStrategyKey] = useState(DEFAULT_STRATEGY_KEY)
  const [draftSourceStrategies, setDraftSourceStrategies] = useState<Record<string, string>>({})
  const [draftSourceStrategyVersions, setDraftSourceStrategyVersions] = useState<Record<string, number | null>>({})
  const [draftInterval, setDraftInterval] = useState('60')
  const [draftSources, setDraftSources] = useState('')
  const [draftParams, setDraftParams] = useState('{}')
  const [draftRisk, setDraftRisk] = useState('{}')
  const [draftMetadata, setDraftMetadata] = useState('{}')
  const [draftMode, setDraftMode] = useState<'shadow' | 'live'>('shadow')
  const [draftCopyFromTraderId, setDraftCopyFromTraderId] = useState('')
  const [draftCopyFromMode, setDraftCopyFromMode] = useState<'shadow' | 'live'>('shadow')
  const [creatingTraderPreview, setCreatingTraderPreview] = useState<{
    name: string
    mode: 'shadow' | 'live'
  } | null>(null)
  const [traderTogglePendingById, setTraderTogglePendingById] = useState<Record<string, TraderToggleAction>>({})
  const [saveError, setSaveError] = useState<string | null>(null)
  const [deleteAction, setDeleteAction] = useState<'block' | 'disable' | 'force_delete'>('disable')
  const [deleteForceConfirm, setDeleteForceConfirm] = useState(false)
  const [tuneDraftTraderId, setTuneDraftTraderId] = useState<string | null>(null)
  const [tuneDraftDirty, setTuneDraftDirty] = useState(false)
  const [tuneSaveError, setTuneSaveError] = useState<string | null>(null)
  const [tuneIteratePrompt, setTuneIteratePrompt] = useState(
    'Analyze recent trader performance and optimize source strategy parameters for higher risk-adjusted PnL. Apply only high-confidence parameter updates.'
  )
  const [tuneIterateModel, setTuneIterateModel] = useState('')
  const [tuneIterateMaxIterations, setTuneIterateMaxIterations] = useState('12')
  const [tuneIterateError, setTuneIterateError] = useState<string | null>(null)
  const [tuneIterateResponse, setTuneIterateResponse] = useState<TraderTuneAgentResponse | null>(null)
  const [tuneAutoEnabled, setTuneAutoEnabled] = useState(false)
  const [tuneAutoIntervalMinutes, setTuneAutoIntervalMinutes] = useState('15')
  const [tuneAutoLastRunAt, setTuneAutoLastRunAt] = useState<number | null>(null)
  const [tuneRevertSnapshot, setTuneRevertSnapshot] = useState<TuneRevertSnapshot | null>(null)
  const [tuneRevertError, setTuneRevertError] = useState<string | null>(null)
  const [tuneParamSectionTab, setTuneParamSectionTab] = useState('')

  const overviewQuery = useQuery({
    queryKey: ['trader-orchestrator-overview'],
    queryFn: getTraderOrchestratorOverview,
    refetchInterval: isConnected ? 10000 : 20000,
  })

  const settingsQuery = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
    refetchInterval: isConnected ? 15000 : 30000,
  })
  const liveExecutionSettings = settingsQuery.data?.live_execution ?? null

  const tradersQuery = useQuery({
    queryKey: ['traders-list', selectedAccountMode],
    queryFn: () => getTraders({ mode: selectedAccountMode }),
    refetchInterval: isConnected ? 15000 : 30000,
  })

  const allTradersQuery = useQuery({
    queryKey: ['traders-list', 'all'],
    queryFn: () => getTraders(),
    refetchInterval: isConnected ? 15000 : 30000,
  })

  const traderSourcesQuery = useQuery({
    queryKey: ['trader-sources'],
    queryFn: getTraderSources,
    staleTime: 300000,
  })

  const traderConfigSchemaQuery = useQuery({
    queryKey: ['trader-config-schema'],
    queryFn: getTraderConfigSchema,
    staleTime: 300000,
  })

  const simulationAccountsQuery = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    staleTime: 30000,
  })
  const trackedWalletsQuery = useQuery({
    queryKey: ['wallets'],
    queryFn: getWallets,
    staleTime: 15000,
  })
  const tradersScopePoolMembersQuery = useQuery({
    queryKey: ['traders-scope-pool-members'],
    queryFn: () => discoveryApi.getPoolMembers({
      limit: 500,
      offset: 0,
      pool_only: true,
      include_blacklisted: false,
      sort_by: 'selection_score',
      sort_dir: 'desc',
    }),
    staleTime: 15000,
  })
  const tradersScopeGroupsQuery = useQuery({
    queryKey: ['traders-scope-groups'],
    queryFn: () => discoveryApi.getTraderGroups(false, 200),
    staleTime: 15000,
  })

  const cryptoMarketsQuery = useQuery({
    queryKey: ['crypto-markets'],
    queryFn: () => getCryptoMarkets(),
    refetchInterval: isConnected ? 5000 : 20000,
  })
  const cryptoMarkets = useMemo(
    () => (Array.isArray(cryptoMarketsQuery.data) ? (cryptoMarketsQuery.data as CryptoMarket[]) : []),
    [cryptoMarketsQuery.data]
  )
  const cryptoMarketById = useMemo(() => {
    const map = new Map<string, CryptoMarket>()
    const register = (value: unknown, market: CryptoMarket) => {
      const key = String(value || '').trim()
      if (!key) return
      map.set(key, market)
      map.set(key.toLowerCase(), market)
    }
    for (const m of cryptoMarkets) {
      register(m.id, m)
      register(m.condition_id, m)
      register(m.slug, m)
      register(m.event_slug, m)
    }
    return map
  }, [cryptoMarkets])
  const resolveCryptoMarket = (value: string | null | undefined): CryptoMarket | null => {
    const key = String(value || '').trim()
    if (!key) return null
    return cryptoMarketById.get(key) || cryptoMarketById.get(key.toLowerCase()) || null
  }
  const resolveCryptoMarketFromAliases = (values: unknown[]): CryptoMarket | null => {
    const aliases = collectMarketAliases(values)
    for (const alias of aliases) {
      const market = resolveCryptoMarket(alias)
      if (market) return market
    }
    return null
  }
  const resolveOrderRealtimeCryptoSnapshot = (
    order: TraderOrder,
    side: DirectionSide | null,
  ): { updatedAt: string | null; markPrice: number | null } => {
    const market = resolveCryptoMarketFromAliases(collectOrderMarketAliasIds(order))
    if (!market) {
      return { updatedAt: null, markPrice: null }
    }

    let latestUpdateMs = toFiniteNumber(market.oracle_updated_at_ms)
    if (latestUpdateMs !== null && latestUpdateMs > 0 && latestUpdateMs < 1_000_000_000_000) {
      latestUpdateMs *= 1000
    }

    const sourceMap = market.oracle_prices_by_source
    if (sourceMap && typeof sourceMap === 'object') {
      for (const rawSnapshot of Object.values(sourceMap)) {
        if (!rawSnapshot || typeof rawSnapshot !== 'object') continue
        let sourceUpdatedMs = toFiniteNumber((rawSnapshot as Record<string, unknown>).updated_at_ms)
        if (sourceUpdatedMs !== null && sourceUpdatedMs > 0 && sourceUpdatedMs < 1_000_000_000_000) {
          sourceUpdatedMs *= 1000
        }
        if (sourceUpdatedMs !== null && sourceUpdatedMs > (latestUpdateMs || 0)) {
          latestUpdateMs = sourceUpdatedMs
        }
      }
    }

    const updatedAt = latestUpdateMs && latestUpdateMs > 0
      ? new Date(latestUpdateMs).toISOString()
      : null

    const upPrice = toFiniteNumber(market.up_price)
    const downPrice = toFiniteNumber(market.down_price)
    const oraclePrice = toFiniteNumber(market.oracle_price)
    const lastTradePrice = toFiniteNumber(market.last_trade_price)

    let markPrice: number | null = null
    if (side === 'YES') {
      markPrice = upPrice
        ?? (downPrice !== null ? Math.max(0, Math.min(1, 1 - downPrice)) : null)
        ?? oraclePrice
        ?? lastTradePrice
    } else if (side === 'NO') {
      markPrice = downPrice
        ?? (upPrice !== null ? Math.max(0, Math.min(1, 1 - upPrice)) : null)
        ?? oraclePrice
        ?? lastTradePrice
    } else {
      markPrice = lastTradePrice ?? oraclePrice ?? upPrice ?? downPrice
    }

    return {
      updatedAt,
      markPrice,
    }
  }
  const [marketModalState, setMarketModalState] = useState<BotMarketModalState | null>(null)
  const [marketModalSellError, setMarketModalSellError] = useState<string | null>(null)
  const [marketModalSellSuccess, setMarketModalSellSuccess] = useState<string | null>(null)
  const marketModalMarket = marketModalState ? marketModalState.market : null
  const themeMode = useAtomValue(themeAtom)

  useEffect(() => {
    if (!marketModalState) return
    document.body.style.overflow = 'hidden'
    const handleEscape = (e: KeyboardEvent) => { if (e.key === 'Escape') setMarketModalState(null) }
    window.addEventListener('keydown', handleEscape)
    return () => { document.body.style.overflow = ''; window.removeEventListener('keydown', handleEscape) }
  }, [marketModalState])

  const traderConfigSchema: TraderConfigSchema | null = traderConfigSchemaQuery.data ?? null
  const traders = tradersQuery.data || []
  const allTraders = allTradersQuery.data || []
  const simulationAccounts = simulationAccountsQuery.data || []
  const trackedWallets = trackedWalletsQuery.data || []
  const tradersScopePoolMembers = tradersScopePoolMembersQuery.data?.members || []
  const tradersScopeGroups = tradersScopeGroupsQuery.data || []
  const selectedSandboxAccount = simulationAccounts.find((account) => account.id === selectedAccountId)
  const selectedAccountValid = selectedAccountIsLive || Boolean(selectedSandboxAccount)
  const sourceCatalog = traderConfigSchema?.sources?.length
    ? traderConfigSchema.sources
    : traderSourcesQuery.data?.length
      ? traderSourcesQuery.data
      : FALLBACK_TRADER_SOURCES
  const defaultSourceKeys = useMemo(
    () => uniqueSourceList(sourceCatalog.map((source) => source.key)),
    [sourceCatalog]
  )
  const defaultSourceCsv = useMemo(() => defaultSourceKeys.join(', '), [defaultSourceKeys])

  const traderIds = useMemo(() => traders.map((trader) => trader.id), [traders])
  const traderIdSet = useMemo(() => new Set(traderIds), [traderIds])
  const traderIdsKey = useMemo(() => traderIds.join('|'), [traderIds])

  const allOrdersQuery = useQuery({
    queryKey: ['trader-orders-all'],
    queryFn: () => getAllTraderOrders(5000),
    enabled: traderIds.length > 0,
    refetchInterval: isConnected ? 2000 : 10000,
    staleTime: 0,
    refetchOnMount: 'always',
  })

  const allDecisionsQuery = useQuery({
    queryKey: ['trader-decisions-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: isConnected ? 30000 : 30000,
    staleTime: 0,
    refetchOnMount: 'always',
    queryFn: () => getAllTraderDecisions(traderIds, {
      limit: Math.min(5000, Math.max(200, traderIds.length * 160)),
      per_trader_limit: 160,
    }),
  })

  const allEventsQuery = useQuery({
    queryKey: ['trader-events-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: isConnected ? 30000 : 30000,
    queryFn: async () => {
      const grouped = await Promise.all(
        traderIds.map((traderId) => getTraderEvents(traderId, { limit: 80 }))
      )
      return grouped
        .flatMap((group) => group.events)
        .sort((a, b) => toTs(b.created_at) - toTs(a.created_at))
    },
  })

  const allOrders = useMemo(
    () => (allOrdersQuery.data || []).filter((order) => {
      if (!traderIdSet.has(String(order.trader_id || ''))) return false
      const orderMode = String(order.mode || '').trim().toLowerCase()
      if (orderMode === 'live' || orderMode === 'shadow') return orderMode === selectedAccountMode
      return selectedAccountMode === 'shadow'
    }),
    [allOrdersQuery.data, selectedAccountMode, traderIdSet]
  )
  const marketModalMarketIds = useMemo(
    () => collectMarketAliases([
      marketModalState?.scope.marketId,
      ...(marketModalState?.scope.marketIds || []),
    ]),
    [marketModalState]
  )
  const marketModalMarketIdsKey = marketModalMarketIds.join('|')
  const marketHistoryQuery = useQuery({
    queryKey: ['trader-market-history', marketModalMarketIdsKey],
    enabled: Boolean(marketModalState) && marketModalMarketIds.length > 0,
    refetchInterval: marketModalState ? 1000 : false,
    staleTime: 0,
    refetchOnMount: 'always',
    queryFn: async () => {
      if (marketModalMarketIds.length === 0) return {}
      return getTraderMarketHistory(marketModalMarketIds, 600)
    },
  })
  useEffect(() => {
    if (!marketModalState || marketModalMarketIds.length === 0) return
    void marketHistoryQuery.refetch()
  }, [marketModalState, marketModalMarketIdsKey])

  const sellTradeNowMutation = useMutation({
    mutationFn: async (params: { traderId: string; orderId: string }) => {
      return sellTraderOrderNow(params.traderId, params.orderId, {
        requested_by: 'trading_panel_modal',
        reason: 'manual_trade_sell_modal',
      })
    },
    onMutate: () => {
      setMarketModalSellError(null)
      setMarketModalSellSuccess(null)
    },
    onSuccess: async (result) => {
      setMarketModalSellSuccess(
        result.mode === 'live'
          ? 'Sell request submitted. Exit execution is now in-flight.'
          : 'Trade sold and marked to market.'
      )
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['trader-orders-all'] }),
        queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] }),
      ])
      void marketHistoryQuery.refetch()
    },
    onError: (error: unknown) => {
      setMarketModalSellError(errorMessage(error, 'Failed to sell trade immediately'))
    },
  })
  const modalSharedHistory = useMemo(
    () => {
      const byMarket = marketHistoryQuery.data || {}
      let bestHistory: unknown[] = []
      for (const marketId of marketModalMarketIds) {
        const history = byMarket[marketId]
        if (Array.isArray(history) && history.length >= 2 && history.length > bestHistory.length) {
          bestHistory = history
        }
      }
      return bestHistory.length >= 2 ? bestHistory : []
    },
    [marketHistoryQuery.data, marketModalMarketIds]
  )
  const allDecisions = allDecisionsQuery.data || []
  const allEvents = allEventsQuery.data || []
  const decisionSignalPayloadByDecisionId = useMemo(() => {
    const byDecisionId = new Map<string, Record<string, unknown>>()
    for (const decision of allDecisions) {
      const decisionId = String(decision.id || '').trim()
      if (!decisionId) continue
      if (!isRecord(decision.signal_payload)) continue
      byDecisionId.set(decisionId, decision.signal_payload)
    }
    return byDecisionId
  }, [allDecisions])

  const selectedTrader = useMemo(
    () => traders.find((trader) => trader.id === selectedTraderId) || null,
    [traders, selectedTraderId]
  )
  useEffect(() => {
    setDeleteForceConfirm(false)
  }, [selectedTraderId])
  const selectedTraderSourceConfigs = useMemo(
    () => (Array.isArray(selectedTrader?.source_configs) ? selectedTrader.source_configs : []),
    [selectedTrader]
  )
  const selectedTraderHasCopySource = useMemo(
    () => traderHasCopyTradeSource(selectedTrader),
    [selectedTrader]
  )
  const selectedTraderCopyExistingOnStartDefault = useMemo(
    () => traderCopyExistingOnStartDefault(selectedTrader),
    [selectedTrader]
  )

  useEffect(() => {
    setSelectedTraderId((current) => {
      if (!current) return current
      return traders.some((trader) => trader.id === current) ? current : null
    })
  }, [traders])

  useEffect(() => {
    if (Object.keys(traderTogglePendingById).length === 0) return
    const tradersById = new Map(traders.map((trader) => [trader.id, trader]))
    setTraderTogglePendingById((current) => {
      let changed = false
      const next: Record<string, TraderToggleAction> = {}
      for (const [traderId, action] of Object.entries(current)) {
        const trader = tradersById.get(traderId)
        const settled =
          action === 'start'
            ? Boolean(trader?.is_enabled) && !Boolean(trader?.is_paused)
            : action === 'stop'
              ? !Boolean(trader?.is_enabled) || Boolean(trader?.is_paused)
              : action === 'activate'
                ? Boolean(trader?.is_enabled)
                : !Boolean(trader?.is_enabled)
        if (settled) {
          changed = true
          continue
        }
        next[traderId] = action
      }
      return changed ? next : current
    })
  }, [traderTogglePendingById, traders])

  const selectedOrders = useMemo(
    () => (selectedTraderId ? allOrders.filter((order) => order.trader_id === selectedTraderId) : []),
    [allOrders, selectedTraderId]
  )

  const selectedDecisions = useMemo(
    () => allDecisions.filter((decision) => decision.trader_id === selectedTraderId),
    [allDecisions, selectedTraderId]
  )

  const selectedEvents = useMemo(
    () => allEvents.filter((event) => event.trader_id === selectedTraderId),
    [allEvents, selectedTraderId]
  )
  const selectedDecisionById = useMemo(() => {
    const byId = new Map<string, Record<string, unknown>>()
    for (const decision of selectedDecisions) {
      const decisionId = cleanText(decision.id)
      if (!decisionId) continue
      byId.set(decisionId, decision as unknown as Record<string, unknown>)
    }
    return byId
  }, [selectedDecisions])

  const sourceCards = useMemo(() => {
    return uniqueSourceList(sourceCatalog.map((source) => source.key))
      .map((key) => sourceCatalog.find((source) => normalizeSourceKey(source.key) === normalizeSourceKey(key)))
      .filter((source): source is TraderSource => Boolean(source))
      .map((source) => ({
        ...source,
        isLegacy: false,
      }))
  }, [sourceCatalog])

  const copySourceTraders = useMemo(
    () => allTraders
      .filter((trader) => trader.mode === draftCopyFromMode)
      .slice()
      .sort((left, right) => left.name.localeCompare(right.name)),
    [allTraders, draftCopyFromMode]
  )

  const selectedSourceKeySet = useMemo(
    () => new Set(csvToList(draftSources).map((sourceKey) => normalizeSourceKey(sourceKey))),
    [draftSources]
  )

  const selectedSourceCount = useMemo(
    () => sourceCards.filter((source) => selectedSourceKeySet.has(normalizeSourceKey(source.key))).length,
    [selectedSourceKeySet, sourceCards]
  )

  const effectiveDraftSources = useMemo(() => {
    const out: string[] = []
    const seen = new Set<string>()
    for (const sourceKey of csvToList(draftSources)) {
      const normalized = normalizeSourceKey(sourceKey)
      if (!normalized || seen.has(normalized)) continue
      seen.add(normalized)
      out.push(normalized)
    }
    return out
  }, [draftSources])

  const sourceStrategyDetailsByKey = useMemo(() => {
    const out: Record<string, StrategyOptionDetail[]> = {}
    for (const source of sourceCards) {
      out[normalizeSourceKey(source.key)] = sourceStrategyDetails(source)
    }
    return out
  }, [sourceCards])

  const sourceStrategyOptionsByKey = useMemo(() => {
    const out: Record<string, StrategyOption[]> = {}
    for (const [sourceKey, details] of Object.entries(sourceStrategyDetailsByKey)) {
      out[sourceKey] = details.map((item) => ({ key: item.key, label: item.label }))
    }
    return out
  }, [sourceStrategyDetailsByKey])

  const sourceStrategyDetailsLookup = useMemo(() => {
    const out: Record<string, Record<string, StrategyOptionDetail>> = {}
    for (const [sourceKey, details] of Object.entries(sourceStrategyDetailsByKey)) {
      out[sourceKey] = {}
      for (const detail of details) {
        out[sourceKey][detail.key] = detail
      }
    }
    return out
  }, [sourceStrategyDetailsByKey])

  const effectiveSourceStrategies = useMemo(() => {
    const out: Record<string, string> = {}
    for (const sourceKey of effectiveDraftSources) {
      const options = sourceStrategyOptionsByKey[sourceKey] || []
      const configured = normalizeStrategyKey(draftSourceStrategies[sourceKey] || '')
      const configuredExists = options.some((option) => option.key === configured)
      out[sourceKey] = configuredExists
        ? configured
        : defaultStrategyForSource(sourceKey, sourceCards)
    }
    return out
  }, [draftSourceStrategies, effectiveDraftSources, sourceCards, sourceStrategyOptionsByKey])
  const effectiveSourceStrategyVersions = useMemo(() => {
    const out: Record<string, number | null> = {}
    for (const sourceKey of effectiveDraftSources) {
      const strategyKey = effectiveSourceStrategies[sourceKey]
      const detail = sourceStrategyDetailsLookup[sourceKey]?.[strategyKey] || null
      const configured = normalizeStrategyVersion(draftSourceStrategyVersions[sourceKey])
      if (!detail) {
        out[sourceKey] = configured
        continue
      }
      const knownVersions = detail.versions
      if (configured == null) {
        out[sourceKey] = null
        continue
      }
      out[sourceKey] = knownVersions.includes(configured) ? configured : null
    }
    return out
  }, [
    draftSourceStrategyVersions,
    effectiveDraftSources,
    effectiveSourceStrategies,
    sourceStrategyDetailsLookup,
  ])

  const cryptoStrategyKeyDraft = useMemo(
    () => effectiveSourceStrategies.crypto || DEFAULT_STRATEGY_KEY,
    [effectiveSourceStrategies]
  )
  const isCryptoStrategyDraft = useMemo(
    () => selectedSourceKeySet.has('crypto'),
    [selectedSourceKeySet]
  )
  const riskFormSchema = useMemo(
    () => ({
      param_fields: Array.isArray(traderConfigSchema?.shared_risk_fields) ? traderConfigSchema.shared_risk_fields : [],
    }),
    [traderConfigSchema]
  )
  const tradersScopeWalletOptions = useMemo(() => {
    const byAddress = new Map<string, { label: string; tags: Set<string> }>()

    const upsert = (rawAddress: unknown, rawLabel: unknown, tag: 'tracked' | 'pool') => {
      const address = String(rawAddress || '').trim().toLowerCase()
      if (!address) return
      const fallback = shortId(address)
      const preferredLabel = String(rawLabel || '').trim() || fallback
      const existing = byAddress.get(address)
      if (!existing) {
        byAddress.set(address, {
          label: preferredLabel,
          tags: new Set([tag]),
        })
        return
      }
      existing.tags.add(tag)
      if (!existing.label || existing.label === fallback) {
        existing.label = preferredLabel
      }
    }

    for (const wallet of trackedWallets) {
      upsert(
        wallet.address,
        String(wallet.username || '').trim() || String(wallet.label || '').trim() || wallet.address,
        'tracked'
      )
    }
    for (const wallet of tradersScopePoolMembers) {
      upsert(
        wallet.address,
        String(wallet.display_name || '').trim() || String(wallet.username || '').trim() || wallet.address,
        'pool'
      )
    }

    return Array.from(byAddress.entries())
      .map(([address, row]) => {
        const tags = Array.from(row.tags.values()).sort()
        const suffix = tags.length > 0 ? ` · ${tags.join('/')}` : ''
        return {
          value: address,
          label: `${row.label} (${shortId(address)})${suffix}`,
        }
      })
      .sort((left, right) => left.label.localeCompare(right.label))
  }, [trackedWallets, tradersScopePoolMembers])
  const tradersScopeGroupOptions = useMemo(
    () =>
      tradersScopeGroups
        .map((group) => {
          const value = String(group.id || '').trim()
          if (!value) return null
          const label = String(group.name || value).trim() || value
          const memberCount = Math.max(0, Math.trunc(Number(group.member_count || 0)))
          return {
            value,
            label: `${label} (${memberCount})`,
          }
        })
        .filter((option): option is { value: string; label: string } => Boolean(option))
        .sort((left, right) => left.label.localeCompare(right.label)),
    [tradersScopeGroups]
  )
  const parsedDraftParams = useMemo(() => parseJsonObject(draftParams || '{}'), [draftParams])
  const parsedDraftRisk = useMemo(() => parseJsonObject(draftRisk || '{}'), [draftRisk])
  const parsedDraftMetadata = useMemo(() => parseJsonObject(draftMetadata || '{}'), [draftMetadata])
  const dynamicStrategyParamSections = useMemo(() => {
    const sharedParams = isRecord(parsedDraftParams.value) ? parsedDraftParams.value : {}
    const sections: DynamicStrategyParamSection[] = []

    for (const sourceKey of effectiveDraftSources) {
      const strategyKey = normalizeStrategyKey(
        normalizeStrategyKeyForSource(
          sourceKey,
          effectiveSourceStrategies[sourceKey] || defaultStrategyForSource(sourceKey, sourceCards),
        ),
      )
      const strategyDetail = sourceStrategyDetailsLookup[sourceKey]?.[strategyKey] || null
      if (!strategyDetail || !Array.isArray(strategyDetail.paramFields)) continue

      const decoratedParamFields = strategyDetail.paramFields.map((field) => {
        if (!isRecord(field)) return field
        if (sourceKey !== 'traders' || strategyKey !== 'traders_copy_trade') return field
        const fieldKey = String(field.key || '').trim()
        if (fieldKey !== 'traders_scope') return field
        const properties = Array.isArray(field.properties) ? field.properties : []
        const nextProperties = properties.map((property) => {
          if (!isRecord(property)) return property
          const propertyKey = String(property.key || '').trim()
          if (propertyKey === 'individual_wallets' && tradersScopeWalletOptions.length > 0) {
            return {
              ...property,
              options: tradersScopeWalletOptions,
            }
          }
          if (propertyKey === 'group_ids' && tradersScopeGroupOptions.length > 0) {
            return {
              ...property,
              options: tradersScopeGroupOptions,
            }
          }
          return property
        })
        return {
          ...field,
          properties: nextProperties,
        }
      })

      const filteredFields = decoratedParamFields.filter((field): field is Record<string, unknown> => {
        if (!isRecord(field)) return false
        const key = String(field.key || '').trim()
        return Boolean(key)
      })
      if (filteredFields.length === 0) continue

      const fieldKeys = filteredFields
        .map((field) => String(field.key || '').trim())
        .filter(Boolean)
      if (fieldKeys.length === 0) continue

      const defaults = isRecord(strategyDetail.defaultParams) ? strategyDetail.defaultParams : {}
      const merged = { ...defaults, ...sharedParams }
      const values: Record<string, unknown> = {}
      for (const fieldKey of fieldKeys) {
        if (Object.prototype.hasOwnProperty.call(merged, fieldKey)) {
          values[fieldKey] = merged[fieldKey]
        }
      }

      const sourceLabel = sourceCards.find((source) => normalizeSourceKey(source.key) === sourceKey)?.label || sourceKey.toUpperCase()
      sections.push({
        sectionKey: `${sourceKey}:${strategyKey}`,
        sourceLabel,
        strategyLabel: strategyDetail.label || strategyLabelForKey(strategyKey, sourceCards),
        groups: groupStrategyParamFields(filteredFields),
        fieldKeys,
        values,
      })
    }

    return sections
  }, [
    effectiveDraftSources,
    effectiveSourceStrategies,
    parsedDraftParams.value,
    sourceCards,
    sourceStrategyDetailsLookup,
    tradersScopeGroupOptions,
    tradersScopeWalletOptions,
  ])
  useEffect(() => {
    if (dynamicStrategyParamSections.length === 0) {
      if (tuneParamSectionTab !== '') setTuneParamSectionTab('')
      return
    }
    if (dynamicStrategyParamSections.some((section) => section.sectionKey === tuneParamSectionTab)) return
    setTuneParamSectionTab(dynamicStrategyParamSections[0].sectionKey)
  }, [dynamicStrategyParamSections, tuneParamSectionTab])
  const tradingScheduleDraft = useMemo(
    () => normalizeTradingScheduleDraft(parsedDraftMetadata.value?.trading_schedule_utc),
    [parsedDraftMetadata.value]
  )
  const riskFormValues = useMemo(
    () => (isRecord(parsedDraftRisk.value) ? parsedDraftRisk.value : {}),
    [parsedDraftRisk.value]
  )
  const setSourceStrategy = (sourceKey: string, strategyKey: string) => {
    const normalizedSource = normalizeSourceKey(sourceKey)
    const normalizedStrategy = normalizeStrategyKeyForSource(normalizedSource, strategyKey)
    setDraftSourceStrategies((current) => ({
      ...current,
      [normalizedSource]: normalizedStrategy,
    }))
    setDraftSourceStrategyVersions((current) => ({
      ...current,
      [normalizedSource]: null,
    }))
    if (normalizedSource === 'crypto') {
      setDraftStrategyKey(normalizedStrategy)
    }
  }

  const setSourceStrategyVersion = (sourceKey: string, versionValue: string) => {
    const normalizedSource = normalizeSourceKey(sourceKey)
    const parsedVersion = normalizeStrategyVersion(versionValue)
    setDraftSourceStrategyVersions((current) => ({
      ...current,
      [normalizedSource]: parsedVersion,
    }))
  }

  const toggleDraftSource = (sourceKey: string) => {
    setDraftSources((current) => {
      const currentList = csvToList(current)
      const normalizedTarget = normalizeSourceKey(sourceKey)
      const hasTarget = currentList.some((item) => normalizeSourceKey(item) === normalizedTarget)
      const next = hasTarget
        ? currentList.filter((item) => normalizeSourceKey(item) !== normalizedTarget)
        : [...currentList, sourceKey]
      return uniqueSourceList(next).join(', ')
    })
    const normalizedTarget = normalizeSourceKey(sourceKey)
    const hasTarget = effectiveDraftSources.includes(normalizedTarget)
    if (hasTarget) {
      setDraftSourceStrategies((current) => {
        const next = { ...current }
        delete next[normalizedTarget]
        return next
      })
      setDraftSourceStrategyVersions((current) => {
        const next = { ...current }
        delete next[normalizedTarget]
        return next
      })
    } else {
      const defaultStrategy = defaultStrategyForSource(normalizedTarget, sourceCards)
      setDraftSourceStrategies((current) => ({ ...current, [normalizedTarget]: defaultStrategy }))
      setDraftSourceStrategyVersions((current) => ({ ...current, [normalizedTarget]: null }))
    }
  }

  const enableAllSourceCards = () => {
    setDraftSources(uniqueSourceList(sourceCards.map((source) => source.key)).join(', '))
    const next: Record<string, string> = {}
    const nextVersions: Record<string, number | null> = {}
    for (const source of sourceCards) {
      const sourceKey = normalizeSourceKey(source.key)
      next[sourceKey] = defaultStrategyForSource(sourceKey, sourceCards)
      nextVersions[sourceKey] = null
    }
    setDraftSourceStrategies(next)
    setDraftSourceStrategyVersions(nextVersions)
  }

  const disableAllSourceCards = () => {
    setDraftSources('')
    setDraftSourceStrategies({})
    setDraftSourceStrategyVersions({})
  }

  useEffect(() => {
    if (traderFlyoutMode !== 'create') return
    setDraftMode(selectedAccountMode)
  }, [selectedAccountMode, traderFlyoutMode])

  useEffect(() => {
    if (traderFlyoutMode !== 'create') return
    if (!draftCopyFromTraderId) return
    const match = allTraders.find((trader) => trader.id === draftCopyFromTraderId)
    if (!match || match.mode !== draftCopyFromMode) {
      setDraftCopyFromTraderId('')
    }
  }, [allTraders, draftCopyFromMode, draftCopyFromTraderId, traderFlyoutMode])

  const decisionDetailQuery = useQuery({
    queryKey: ['trader-decision-detail', selectedDecisionId],
    queryFn: () => getTraderDecisionDetail(String(selectedDecisionId)),
    enabled: Boolean(selectedDecisionId),
    refetchInterval: 7000,
  })

  const tuneIterateParsed = useMemo(
    () => (isRecord(tuneIterateResponse?.parsed) ? tuneIterateResponse?.parsed : null),
    [tuneIterateResponse]
  )
  const tuneIterateActions = useMemo(() => {
    if (!tuneIterateParsed) return [] as string[]
    return normalizeTuneList(tuneIterateParsed.actions_taken)
  }, [tuneIterateParsed])
  const tuneIterateNextSteps = useMemo(() => {
    if (!tuneIterateParsed) return [] as string[]
    return normalizeTuneList(tuneIterateParsed.suggested_next_steps)
  }, [tuneIterateParsed])
  const tuneIterateIssues = useMemo(() => {
    if (!tuneIterateParsed) return [] as string[]
    const issueCandidates = [
      tuneIterateParsed.issues_identified,
      tuneIterateParsed.issues,
      tuneIterateParsed.problem_analysis,
      tuneIterateParsed.risk_findings,
    ]
    for (const candidate of issueCandidates) {
      const issues = normalizeTuneList(candidate)
      if (issues.length > 0) return issues
    }
    return [] as string[]
  }, [tuneIterateParsed])
  const tuneIterateAppliedUpdates = useMemo(
    () => (Array.isArray(tuneIterateResponse?.applied_param_updates) ? tuneIterateResponse.applied_param_updates : []),
    [tuneIterateResponse]
  )

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
    queryClient.invalidateQueries({ queryKey: ['traders-list'] })
    queryClient.invalidateQueries({ queryKey: ['trader-orders-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-orders'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decisions-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decision-detail'] })
    queryClient.invalidateQueries({ queryKey: ['wallets'] })
    queryClient.invalidateQueries({ queryKey: ['traders-scope-pool-members'] })
    queryClient.invalidateQueries({ queryKey: ['traders-scope-groups'] })
    queryClient.invalidateQueries({ queryKey: ['trader-config-schema'] })
    queryClient.invalidateQueries({ queryKey: ['trader-sources'] })
    queryClient.invalidateQueries({ queryKey: ['unified-strategies'] })
    queryClient.invalidateQueries({ queryKey: ['unified-strategy-versions'] })
    queryClient.invalidateQueries({ queryKey: ['settings'] })
  }

  const upsertTraderInCache = (trader: Trader) => {
    const normalizedMode: 'shadow' | 'live' = trader.mode === 'live' ? 'live' : 'shadow'
    const otherMode: 'shadow' | 'live' = normalizedMode === 'live' ? 'shadow' : 'live'
    queryClient.setQueryData<Trader[]>(['traders-list', 'all'], (current) => upsertTraderRows(current, trader))
    queryClient.setQueryData<Trader[]>(['traders-list', normalizedMode], (current) => upsertTraderRows(current, trader))
    queryClient.setQueryData<Trader[]>(['traders-list', otherMode], (current) => {
      if (!Array.isArray(current)) return current
      return current.filter((row) => row.id !== trader.id)
    })
  }

  const applyTraderDraftSettings = (
    trader: Trader,
    options: { preserveName?: boolean; preserveCopyFrom?: boolean; preserveMode?: boolean } = {}
  ) => {
    const traderSourceConfigs = Array.isArray(trader.source_configs) ? trader.source_configs : []
    const normalizedSourceKeys = uniqueSourceList(
      traderSourceConfigs.map((config) => normalizeSourceKey(String(config.source_key || '')))
    )
    const sourceStrategyMap: Record<string, string> = {}
    const sourceVersionMap: Record<string, number | null> = {}
    for (const config of traderSourceConfigs) {
      const sourceKey = normalizeSourceKey(String(config.source_key || ''))
      if (!sourceKey) continue
      sourceStrategyMap[sourceKey] = normalizeStrategyKeyForSource(
        sourceKey,
        config.strategy_key || defaultStrategyForSource(sourceKey, sourceCards)
      )
      sourceVersionMap[sourceKey] = normalizeStrategyVersion(config.strategy_version)
    }
    const primaryParams = (traderSourceConfigs[0]?.strategy_params || {}) as Record<string, unknown>
    const tradersScope = traderSourceConfigs.find((config) => normalizeSourceKey(String(config.source_key || '')) === 'traders')
      ?.strategy_params?.traders_scope
    const mergedDraftParams: Record<string, unknown> = { ...primaryParams }
    if (tradersScope !== undefined) {
      mergedDraftParams.traders_scope = normalizeTradersScopeConfig(tradersScope)
    }

    if (!options.preserveName) {
      setDraftName(trader.name)
    }
    setDraftDescription(trader.description || '')
    if (!options.preserveMode) {
      setDraftMode(trader.mode === 'live' ? 'live' : 'shadow')
    }
    setDraftStrategyKey(normalizeStrategyKey(sourceStrategyMap.crypto || DEFAULT_STRATEGY_KEY))
    setDraftSourceStrategies(sourceStrategyMap)
    setDraftSourceStrategyVersions(sourceVersionMap)
    setDraftInterval(String(trader.interval_seconds || 60))
    setDraftSources(normalizedSourceKeys.join(', ') || defaultSourceCsv)
    const risk = trader.risk_limits || {}
    const metadata = trader.metadata || {}
    setDraftParams(JSON.stringify(mergedDraftParams, null, 2))
    setDraftRisk(JSON.stringify(risk, null, 2))
    setDraftMetadata(JSON.stringify(metadata, null, 2))
    if (!options.preserveCopyFrom) {
      setDraftCopyFromTraderId('')
    }
  }

  useEffect(() => {
    if (traderFlyoutOpen) return
    if (creatingTraderPreview) return
    if (!selectedTrader) {
      if (tuneDraftTraderId !== null) setTuneDraftTraderId(null)
      if (tuneDraftDirty) setTuneDraftDirty(false)
      return
    }
    if (tuneDraftTraderId === selectedTrader.id) return

    applyTraderDraftSettings(selectedTrader)
    setTuneDraftTraderId(selectedTrader.id)
    setTuneDraftDirty(false)
    setTuneSaveError(null)
    setTuneRevertError(null)
  }, [
    creatingTraderPreview,
    selectedTrader,
    traderFlyoutOpen,
    tuneDraftDirty,
    tuneDraftTraderId,
  ])

  const applyCreateCopyFromSelection = (value: string) => {
    const sourceTraderId = value === '__none__' ? '' : String(value || '').trim()
    setDraftCopyFromTraderId(sourceTraderId)
    if (!sourceTraderId) {
      setSaveError(null)
      return
    }

    const sourceTrader = allTraders.find((trader) => trader.id === sourceTraderId)
    if (!sourceTrader) {
      setSaveError('Selected copy source bot was not found. Refresh and try again.')
      return
    }

    applyTraderDraftSettings(sourceTrader, { preserveName: true, preserveCopyFrom: true, preserveMode: true })
    setSaveError(null)
  }

  const openCreateTraderFlyout = () => {
    const defaultSources = defaultSourceKeys.length > 0 ? defaultSourceKeys.map((key) => normalizeSourceKey(key)) : ['crypto']
    const defaultStrategies = Object.fromEntries(
      defaultSources.map((sourceKey) => [sourceKey, defaultStrategyForSource(sourceKey, sourceCards)])
    ) as Record<string, string>
    const defaultStrategyVersions = Object.fromEntries(
      defaultSources.map((sourceKey) => [sourceKey, null])
    ) as Record<string, number | null>
    setTraderFlyoutMode('create')
    setDraftName('')
    setDraftDescription('')
    setDraftStrategyKey(normalizeStrategyKey(defaultStrategies.crypto || DEFAULT_STRATEGY_KEY))
    setDraftSourceStrategies(defaultStrategies)
    setDraftSourceStrategyVersions(defaultStrategyVersions)
    setDraftInterval('5')
    setDraftSources(defaultSources.join(', '))
    setDraftParams('{}')
    setDraftRisk(JSON.stringify(isRecord(traderConfigSchema?.shared_risk_defaults) ? traderConfigSchema.shared_risk_defaults : {}, null, 2))
    setDraftMetadata('{}')
    setDraftMode(selectedAccountMode)
    setDraftCopyFromTraderId('')
    setDraftCopyFromMode(selectedAccountMode)
    setDeleteAction('disable')
    setDeleteForceConfirm(false)
    setSaveError(null)
    setTuneDraftTraderId(null)
    setTuneDraftDirty(false)
    setTuneSaveError(null)
    setTuneIteratePrompt(
      'Analyze recent trader performance and optimize source strategy parameters for higher risk-adjusted PnL. Apply only high-confidence parameter updates.'
    )
    setTuneIterateModel('')
    setTuneIterateMaxIterations('12')
    setTuneIterateError(null)
    setTuneIterateResponse(null)
    setTuneAutoEnabled(false)
    setTuneAutoLastRunAt(null)
    setTuneRevertSnapshot(null)
    setTuneRevertError(null)
    setTraderFlyoutOpen(true)
  }

  const openEditTraderFlyout = (trader: Trader) => {
    setSelectedTraderId(trader.id)
    setTraderFlyoutMode('edit')
    applyTraderDraftSettings(trader)
    setDraftCopyFromMode(trader.mode === 'live' ? 'live' : 'shadow')
    setDeleteAction('disable')
    setDeleteForceConfirm(false)
    setSaveError(null)
    setTuneDraftTraderId(trader.id)
    setTuneDraftDirty(false)
    setTuneSaveError(null)
    setTuneIteratePrompt(
      'Analyze this trader performance and optimize source strategy parameters for measurable, risk-adjusted PnL improvement.'
    )
    setTuneIterateModel('')
    setTuneIterateMaxIterations('12')
    setTuneIterateError(null)
    setTuneIterateResponse(null)
    setTuneAutoEnabled(false)
    setTuneAutoLastRunAt(null)
    setTuneRevertSnapshot(null)
    setTuneRevertError(null)
    setTraderFlyoutOpen(true)
  }

  const applyDynamicStrategyFormValues = (
    fieldKeys: string[],
    values: Record<string, unknown>,
  ) => {
    setTuneDraftDirty(true)
    setTuneSaveError(null)
    setTuneRevertError(null)
    setDraftParams((current) => {
      const parsed = parseJsonObject(current || '{}')
      const nextValues = isRecord(parsed.value) ? { ...parsed.value } : {}
      for (const key of fieldKeys) {
        if (!Object.prototype.hasOwnProperty.call(values, key)) continue
        const value = values[key]
        if (value === undefined) {
          delete nextValues[key]
          continue
        }
        if (key === 'traders_scope') {
          nextValues[key] = normalizeTradersScopeConfig(value)
        } else {
          nextValues[key] = value
        }
      }
      return JSON.stringify(nextValues, null, 2)
    })
  }

  const setDraftRiskFromForm = (values: Record<string, unknown>) => {
    setDraftRisk(JSON.stringify(values, null, 2))
  }

  const setDraftTradingSchedule = (
    patch:
      | Partial<TradingScheduleDraft>
      | ((current: TradingScheduleDraft) => Partial<TradingScheduleDraft>)
  ) => {
    const metadataBase: Record<string, unknown> = parsedDraftMetadata.value || {}
    const current = normalizeTradingScheduleDraft(metadataBase.trading_schedule_utc)
    const resolvedPatch = typeof patch === 'function' ? patch(current) : patch
    const next = normalizeTradingScheduleDraft({
      enabled: resolvedPatch.enabled ?? current.enabled,
      days: resolvedPatch.days ?? current.days,
      start_time: resolvedPatch.startTimeUtc ?? current.startTimeUtc,
      end_time: resolvedPatch.endTimeUtc ?? current.endTimeUtc,
      start_date: resolvedPatch.startDateUtc ?? current.startDateUtc,
      end_date: resolvedPatch.endDateUtc ?? current.endDateUtc,
      end_at: resolvedPatch.endAtUtc ?? current.endAtUtc,
    })
    const nextMetadata = {
      ...metadataBase,
      trading_schedule_utc: buildTradingScheduleMetadata(next),
    }
    setDraftMetadata(JSON.stringify(nextMetadata, null, 2))
  }

  const toggleTradingScheduleDay = (day: TradingScheduleDay) => {
    setDraftTradingSchedule((current) => {
      const exists = current.days.includes(day)
      const nextDays = exists
        ? current.days.filter((item) => item !== day)
        : [...current.days, day]
      return { days: normalizeTradingScheduleDays(nextDays) }
    })
  }

  const buildDraftSourceConfigs = (rawStrategyParams: Record<string, unknown>): TraderSourceConfig[] => {
    const configs: TraderSourceConfig[] = []
    for (const sourceKey of effectiveDraftSources) {
      const strategyKey = normalizeStrategyKey(
        normalizeStrategyKeyForSource(
          sourceKey,
          effectiveSourceStrategies[sourceKey] || defaultStrategyForSource(sourceKey, sourceCards)
        )
      )
      const strategyDetail = sourceStrategyDetailsLookup[sourceKey]?.[strategyKey] || null
      const strategyVersion = normalizeStrategyVersion(effectiveSourceStrategyVersions[sourceKey])
      const nextConfig: TraderSourceConfig = {
        source_key: sourceKey,
        strategy_key: strategyKey,
        strategy_version: strategyVersion,
        strategy_params: buildSourceStrategyParams(rawStrategyParams, sourceKey, strategyDetail),
      }
      configs.push(nextConfig)
    }
    return configs
  }

  const cloneSourceConfigsForTuneSnapshot = (configs: TraderSourceConfig[]): TraderSourceConfig[] => {
    return configs.map((config) => {
      let strategyParams: Record<string, unknown> = {}
      try {
        strategyParams = JSON.parse(JSON.stringify(config.strategy_params || {})) as Record<string, unknown>
      } catch {
        strategyParams = {}
      }
      return {
        source_key: String(config.source_key || ''),
        strategy_key: String(config.strategy_key || ''),
        strategy_version: normalizeStrategyVersion(config.strategy_version),
        strategy_params: strategyParams,
      }
    })
  }

  const captureTuneRevertSnapshot = (): TuneRevertSnapshot | null => {
    if (!selectedTrader) return null
    return {
      traderId: selectedTrader.id,
      sourceConfigs: cloneSourceConfigsForTuneSnapshot(selectedTraderSourceConfigs),
      capturedAt: new Date().toISOString(),
    }
  }

  const startBySelectedAccountMutation = useMutation({
    mutationFn: async () => {
      if (!selectedAccountId || !selectedAccountValid) {
        throw new Error('Select a valid global account in the top control bar.')
      }
      if (selectedAccountIsLive) {
        const preflight = await runTraderOrchestratorLivePreflight({ mode: 'live' })
        if (preflight.status !== 'passed') {
          throw new Error('Live preflight did not pass. Review checks before live launch.')
        }
        const armed = await armTraderOrchestratorLiveStart({ preflight_id: preflight.preflight_id })
        return startTraderOrchestratorLive({ arm_token: armed.arm_token, mode: 'live' })
      }
      if (!selectedSandboxAccount?.id) {
        throw new Error('No sandbox account is selected for shadow mode.')
      }
      return startTraderOrchestrator({ mode: 'shadow' })
    },
    onMutate: () => {
      setControlActionError(null)
    },
    onSuccess: (result: any) => {
      const responseControl = result?.control && typeof result.control === 'object' ? result.control : {}
      const startMode = String(responseControl.mode || '').trim().toLowerCase()
      queryClient.setQueryData(['trader-orchestrator-overview'], (current: any) => {
        if (!current || typeof current !== 'object') {
          return current
        }
        const currentControl = current.control && typeof current.control === 'object' ? current.control : {}
        const currentWorker = current.worker && typeof current.worker === 'object' ? current.worker : {}
        return {
          ...current,
          control: {
            ...currentControl,
            ...responseControl,
          },
          worker: {
            ...currentWorker,
            running: false,
            enabled: true,
            current_activity: startMode === 'live' ? 'Live start command queued' : 'Start command queued',
            interval_seconds: Number(
              responseControl.run_interval_seconds
              || currentWorker.interval_seconds
              || 2
            ),
            last_error: null,
          },
        }
      })
      refreshAll()
    },
    onError: (error: unknown) => {
      setControlActionError(errorMessage(error, 'Failed to start orchestrator'))
    },
  })

  const stopByModeMutation = useMutation({
    mutationFn: async () => {
      const mode = String(overviewQuery.data?.control?.mode || 'shadow').toLowerCase()
      if (mode === 'live') {
        return { response: await stopTraderOrchestratorLive(), mode }
      }
      return { response: await stopTraderOrchestrator(), mode }
    },
    onMutate: () => {
      setControlActionError(null)
    },
    onSuccess: (result: { response: any; mode: string }) => {
      const responseControl = result?.response?.control && typeof result.response.control === 'object'
        ? result.response.control
        : {}
      queryClient.setQueryData(['trader-orchestrator-overview'], (current: any) => {
        if (!current || typeof current !== 'object') {
          return current
        }
        const currentControl = current.control && typeof current.control === 'object' ? current.control : {}
        const currentWorker = current.worker && typeof current.worker === 'object' ? current.worker : {}
        return {
          ...current,
          control: {
            ...currentControl,
            ...responseControl,
          },
          worker: {
            ...currentWorker,
            running: false,
            enabled: false,
            current_activity: result.mode === 'live' ? 'Live stop requested' : 'Manual stop requested',
            interval_seconds: Number(
              responseControl.run_interval_seconds
              || currentWorker.interval_seconds
              || 2
            ),
            last_error: null,
          },
        }
      })
      refreshAll()
    },
    onError: (error: unknown) => {
      setControlActionError(errorMessage(error, 'Failed to stop orchestrator'))
    },
  })

  const killSwitchMutation = useMutation({
    mutationFn: (enabled: boolean) => setTraderOrchestratorLiveKillSwitch(enabled),
    onMutate: async (enabled: boolean) => {
      setControlActionError(null)
      await queryClient.cancelQueries({ queryKey: ['trader-orchestrator-overview'] })
      const previousOverview = queryClient.getQueryData(['trader-orchestrator-overview'])
      queryClient.setQueryData(['trader-orchestrator-overview'], (current: any) => {
        if (!current || typeof current !== 'object') {
          return current
        }
        const currentControl = current.control && typeof current.control === 'object' ? current.control : {}
        const currentConfig = current.config && typeof current.config === 'object' ? current.config : {}
        return {
          ...current,
          control: {
            ...currentControl,
            kill_switch: enabled,
          },
          config: {
            ...currentConfig,
            kill_switch: enabled,
          },
        }
      })
      return { previousOverview }
    },
    onSuccess: (result: any, enabled: boolean) => {
      queryClient.setQueryData(['trader-orchestrator-overview'], (current: any) => {
        if (!current || typeof current !== 'object') {
          return current
        }
        const currentControl = current.control && typeof current.control === 'object' ? current.control : {}
        const currentConfig = current.config && typeof current.config === 'object' ? current.config : {}
        const responseControl = result?.control && typeof result.control === 'object' ? result.control : {}
        const killSwitchValue = Boolean(result?.kill_switch ?? responseControl.kill_switch ?? enabled)
        return {
          ...current,
          control: {
            ...currentControl,
            ...responseControl,
            kill_switch: killSwitchValue,
          },
          config: {
            ...currentConfig,
            kill_switch: killSwitchValue,
          },
        }
      })
      refreshAll()
    },
    onError: (error: unknown, _enabled: boolean, context: { previousOverview: unknown } | undefined) => {
      if (context) {
        queryClient.setQueryData(['trader-orchestrator-overview'], context.previousOverview)
      }
      setControlActionError(errorMessage(error, 'Failed to update Block new orders'))
    },
  })

  const updateGlobalSettingsMutation = useMutation({
    mutationFn: async () => {
      const runIntervalSeconds = Math.trunc(clampNumber(toNumber(globalSettingsDraft.runIntervalSeconds), 1, 300, 5))
      const maxGrossExposureUsd = clampNumber(
        toNumber(globalSettingsDraft.maxGrossExposureUsd),
        1,
        1_000_000,
        DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_gross_exposure_usd,
      )
      const maxDailyLossUsd = clampNumber(
        toNumber(globalSettingsDraft.maxDailyLossUsd),
        0,
        1_000_000,
        DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_daily_loss_usd,
      )
      const maxOrdersPerCycle = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.maxOrdersPerCycle),
          1,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_orders_per_cycle,
        )
      )
      const pendingExitMaxAllowed = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.pendingExitMaxAllowed),
          0,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard.max_pending_exits,
        )
      )
      const minCooldownSeconds = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.minCooldownSeconds),
          0,
          86400,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.min_cooldown_seconds,
        )
      )
      const maxConsecutiveLossesCap = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.maxConsecutiveLossesCap),
          1,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_consecutive_losses_cap,
        )
      )
      const maxOpenOrdersCap = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.maxOpenOrdersCap),
          1,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_orders_cap,
        )
      )
      const maxTradeNotionalUsdCap = clampNumber(
        toNumber(globalSettingsDraft.maxTradeNotionalUsdCap),
        1,
        1_000_000,
        DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_trade_notional_usd_cap,
      )
      const maxOrdersPerCycleCap = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.maxOrdersPerCycleCap),
          1,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_orders_per_cycle_cap,
        )
      )
      const liveMarketHistoryWindowSeconds = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.liveMarketHistoryWindowSeconds),
          300,
          21600,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.history_window_seconds,
        )
      )
      const liveMarketHistoryFidelitySeconds = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.liveMarketHistoryFidelitySeconds),
          30,
          1800,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.history_fidelity_seconds,
        )
      )
      const liveMarketHistoryMaxPoints = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.liveMarketHistoryMaxPoints),
          20,
          240,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.max_history_points,
        )
      )
      const liveMarketContextTimeoutSeconds = clampNumber(
        toNumber(globalSettingsDraft.liveMarketContextTimeoutSeconds),
        1,
        12,
        DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context.timeout_seconds,
      )
      const liveProviderHealthWindowSeconds = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.liveProviderHealthWindowSeconds),
          30,
          900,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health.window_seconds,
        )
      )
      const liveProviderHealthMinErrors = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.liveProviderHealthMinErrors),
          1,
          20,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health.min_errors,
        )
      )
      const liveProviderHealthBlockSeconds = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.liveProviderHealthBlockSeconds),
          15,
          3600,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health.block_seconds,
        )
      )
      const traderCycleTimeoutRaw = globalSettingsDraft.traderCycleTimeoutSeconds.trim()
      const traderCycleTimeoutSeconds = traderCycleTimeoutRaw
        ? clampNumber(toNumber(traderCycleTimeoutRaw), 3, 120, 0)
        : null
      const maxTradeSizeUsd = clampNumber(
        toNumber(globalSettingsDraft.maxTradeSizeUsd),
        1,
        100_000,
        DEFAULT_LIVE_EXECUTION_LIMITS.max_trade_size_usd,
      )
      const maxDailyTradeVolume = clampNumber(
        toNumber(globalSettingsDraft.maxDailyTradeVolumeUsd),
        10,
        10_000_000,
        DEFAULT_LIVE_EXECUTION_LIMITS.max_daily_trade_volume,
      )
      const minAccountBalanceUsd = clampNumber(
        toNumber(globalSettingsDraft.minAccountBalanceUsd),
        0,
        1_000_000,
        DEFAULT_LIVE_EXECUTION_LIMITS.min_account_balance_usd,
      )
      const maxOpenPositions = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.maxOpenPositions),
          1,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_positions_cap,
        )
      )
      const maxSlippagePercent = clampNumber(
        toNumber(globalSettingsDraft.maxSlippagePercent),
        0.1,
        10,
        DEFAULT_LIVE_EXECUTION_LIMITS.max_slippage_percent,
      )

      const orchestratorPayload = {
        run_interval_seconds: runIntervalSeconds,
        global_risk: {
          max_gross_exposure_usd: maxGrossExposureUsd,
          max_daily_loss_usd: maxDailyLossUsd,
          max_orders_per_cycle: maxOrdersPerCycle,
        },
        global_runtime: {
          pending_live_exit_guard: {
            max_pending_exits: pendingExitMaxAllowed,
            identity_guard_enabled: globalSettingsDraft.pendingExitIdentityGuardEnabled,
            terminal_statuses: normalizePendingExitTerminalStatusesCsv(globalSettingsDraft.pendingExitTerminalStatuses),
          },
          live_risk_clamps: {
            enforce_allow_averaging_off: globalSettingsDraft.enforceAllowAveragingOff,
            min_cooldown_seconds: minCooldownSeconds,
            max_consecutive_losses_cap: maxConsecutiveLossesCap,
            max_open_orders_cap: maxOpenOrdersCap,
            max_open_positions_cap: maxOpenPositions,
            max_trade_notional_usd_cap: maxTradeNotionalUsdCap,
            max_orders_per_cycle_cap: maxOrdersPerCycleCap,
            enforce_halt_on_consecutive_losses: globalSettingsDraft.enforceHaltOnConsecutiveLosses,
          },
          live_market_context: {
            enabled: globalSettingsDraft.liveMarketContextEnabled,
            history_window_seconds: liveMarketHistoryWindowSeconds,
            history_fidelity_seconds: liveMarketHistoryFidelitySeconds,
            max_history_points: liveMarketHistoryMaxPoints,
            timeout_seconds: liveMarketContextTimeoutSeconds,
          },
          live_provider_health: {
            window_seconds: liveProviderHealthWindowSeconds,
            min_errors: liveProviderHealthMinErrors,
            block_seconds: liveProviderHealthBlockSeconds,
          },
          trader_cycle_timeout_seconds: traderCycleTimeoutSeconds,
        },
      }

      await Promise.all([
        updateTraderOrchestratorSettings(orchestratorPayload),
        updateSettings({
          live_execution: {
            max_trade_size_usd: maxTradeSizeUsd,
            max_daily_trade_volume: maxDailyTradeVolume,
            max_slippage_percent: maxSlippagePercent,
            min_account_balance_usd: minAccountBalanceUsd,
          },
        }),
      ])

      return { status: 'ok' }
    },
    onSuccess: () => {
      setGlobalSettingsSaveError(null)
      setGlobalSettingsFlyoutOpen(false)
      refreshAll()
    },
    onError: (error: unknown) => {
      setGlobalSettingsSaveError(errorMessage(error, 'Failed to update global orchestrator settings'))
    },
  })

  const traderStartMutation = useMutation({
    mutationFn: ({ traderId, copyExistingPositions }: { traderId: string; copyExistingPositions?: boolean }) =>
      startTrader(traderId, { copy_existing_positions: copyExistingPositions }),
    onMutate: ({ traderId }: { traderId: string; copyExistingPositions?: boolean }) => {
      setSaveError(null)
      setTraderTogglePendingById((current) => ({
        ...current,
        [traderId]: 'start',
      }))
    },
    onSuccess: (updatedTrader) => {
      queryClient.setQueriesData({ queryKey: ['traders-list'] }, (current: unknown) => {
        if (!Array.isArray(current)) return current
        return current.map((candidate) => {
          if (!candidate || typeof candidate !== 'object') return candidate
          const trader = candidate as Trader
          if (trader.id !== updatedTrader.id) return candidate
          return updatedTrader
        })
      })
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to start bot'))
    },
    onSettled: (_data, _error, variables) => {
      const traderId = variables?.traderId
      if (!traderId) return
      setTraderTogglePendingById((current) => {
        if (!(traderId in current)) return current
        const next = { ...current }
        delete next[traderId]
        return next
      })
    },
  })

  const traderStopMutation = useMutation({
    mutationFn: ({ traderId, payload }: { traderId: string; payload: TraderStopPayload }) =>
      stopTrader(traderId, payload),
    onMutate: ({ traderId }: { traderId: string; payload: TraderStopPayload }) => {
      setSaveError(null)
      setTraderTogglePendingById((current) => ({
        ...current,
        [traderId]: 'stop',
      }))
    },
    onSuccess: (updatedTrader) => {
      queryClient.setQueriesData({ queryKey: ['traders-list'] }, (current: unknown) => {
        if (!Array.isArray(current)) return current
        return current.map((candidate) => {
          if (!candidate || typeof candidate !== 'object') return candidate
          const trader = candidate as Trader
          if (trader.id !== updatedTrader.id) return candidate
          return updatedTrader
        })
      })
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to stop bot'))
    },
    onSettled: (_data, _error, variables) => {
      const traderId = variables?.traderId
      if (!traderId) return
      setTraderTogglePendingById((current) => {
        if (!(traderId in current)) return current
        const next = { ...current }
        delete next[traderId]
        return next
      })
    },
  })

  const traderActivateMutation = useMutation({
    mutationFn: ({ traderId }: { traderId: string }) => activateTrader(traderId, {}),
    onMutate: ({ traderId }: { traderId: string }) => {
      setSaveError(null)
      setTraderTogglePendingById((current) => ({
        ...current,
        [traderId]: 'activate',
      }))
    },
    onSuccess: (updatedTrader) => {
      queryClient.setQueriesData({ queryKey: ['traders-list'] }, (current: unknown) => {
        if (!Array.isArray(current)) return current
        return current.map((candidate) => {
          if (!candidate || typeof candidate !== 'object') return candidate
          const trader = candidate as Trader
          if (trader.id !== updatedTrader.id) return candidate
          return updatedTrader
        })
      })
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to activate bot'))
    },
    onSettled: (_data, _error, variables) => {
      const traderId = variables?.traderId
      if (!traderId) return
      setTraderTogglePendingById((current) => {
        if (!(traderId in current)) return current
        const next = { ...current }
        delete next[traderId]
        return next
      })
    },
  })

  const traderDeactivateMutation = useMutation({
    mutationFn: ({ traderId }: { traderId: string }) => deactivateTrader(traderId, {}),
    onMutate: ({ traderId }: { traderId: string }) => {
      setSaveError(null)
      setTraderTogglePendingById((current) => ({
        ...current,
        [traderId]: 'deactivate',
      }))
    },
    onSuccess: (updatedTrader) => {
      queryClient.setQueriesData({ queryKey: ['traders-list'] }, (current: unknown) => {
        if (!Array.isArray(current)) return current
        return current.map((candidate) => {
          if (!candidate || typeof candidate !== 'object') return candidate
          const trader = candidate as Trader
          if (trader.id !== updatedTrader.id) return candidate
          return updatedTrader
        })
      })
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to set bot inactive'))
    },
    onSettled: (_data, _error, variables) => {
      const traderId = variables?.traderId
      if (!traderId) return
      setTraderTogglePendingById((current) => {
        if (!(traderId in current)) return current
        const next = { ...current }
        delete next[traderId]
        return next
      })
    },
  })

  const traderRunOnceMutation = useMutation({
    mutationFn: (traderId: string) => runTraderOnce(traderId),
    onSuccess: refreshAll,
  })

  const saveTuneParametersMutation = useMutation({
    mutationFn: async () => {
      if (!selectedTrader) {
        throw new Error('Select a bot before saving tune parameters.')
      }
      const parsedParams = parseJsonObject(draftParams || '{}')
      if (!parsedParams.value) {
        throw new Error(`Strategy params JSON error: ${parsedParams.error || 'invalid object'}`)
      }
      if (effectiveDraftSources.length === 0) {
        throw new Error('Enable at least one source.')
      }
      const tradersEnabled = effectiveDraftSources.includes('traders')
      const tradersScope = normalizeTradersScopeConfig(parsedParams.value.traders_scope)
      if (tradersEnabled && tradersScope.modes.includes('individual') && tradersScope.individual_wallets.length === 0) {
        throw new Error('Select at least one individual wallet for wallet scope.')
      }
      if (tradersEnabled && tradersScope.modes.includes('group') && tradersScope.group_ids.length === 0) {
        throw new Error('Select at least one group for wallet scope.')
      }
      return updateTrader(selectedTrader.id, {
        source_configs: buildDraftSourceConfigs(parsedParams.value),
      })
    },
    onMutate: () => {
      const snapshot = captureTuneRevertSnapshot()
      if (snapshot) setTuneRevertSnapshot(snapshot)
      setTuneSaveError(null)
      setTuneRevertError(null)
    },
    onSuccess: (trader) => {
      setTuneSaveError(null)
      setTuneRevertError(null)
      setTuneDraftDirty(false)
      setTuneDraftTraderId(trader.id)
      applyTraderDraftSettings(trader)
      refreshAll()
    },
    onError: (error: unknown) => {
      setTuneSaveError(errorMessage(error, 'Failed to save tune parameters'))
    },
  })

  const revertTuneParametersMutation = useMutation({
    mutationFn: async () => {
      if (!selectedTrader) {
        throw new Error('Select a bot before reverting tune parameters.')
      }
      if (!tuneRevertSnapshot || tuneRevertSnapshot.traderId !== selectedTrader.id) {
        throw new Error('No tune snapshot is available to revert.')
      }
      const snapshot = tuneRevertSnapshot
      const trader = await updateTrader(selectedTrader.id, {
        source_configs: cloneSourceConfigsForTuneSnapshot(snapshot.sourceConfigs),
      })
      return { trader, snapshot }
    },
    onMutate: () => {
      setTuneRevertError(null)
      setTuneSaveError(null)
    },
    onSuccess: ({ trader }) => {
      setTuneRevertError(null)
      setTuneSaveError(null)
      setTuneDraftDirty(false)
      setTuneDraftTraderId(trader.id)
      setTuneRevertSnapshot(null)
      applyTraderDraftSettings(trader)
      refreshAll()
    },
    onError: (error: unknown) => {
      setTuneRevertError(errorMessage(error, 'Failed to revert tune parameters'))
    },
  })

  const runTuneIterateMutation = useMutation({
    mutationFn: async ({ trigger }: { trigger: 'manual' | 'auto' }) => {
      if (!selectedTrader) {
        throw new Error('Select a bot before running agent.')
      }
      if (trigger !== 'manual' && trigger !== 'auto') {
        throw new Error('Invalid tune trigger.')
      }
      const prompt = tuneIteratePrompt.trim()
      if (!prompt) {
        throw new Error('Enter an agent prompt.')
      }
      const maxIterations = Math.max(1, Math.min(24, Math.trunc(toNumber(tuneIterateMaxIterations || 12))))
      return runTraderTuneIteration(selectedTrader.id, {
        prompt,
        max_iterations: maxIterations,
        ...(tuneIterateModel.trim() ? { model: tuneIterateModel.trim() } : {}),
      })
    },
    onMutate: () => {
      const snapshot = captureTuneRevertSnapshot()
      if (snapshot) setTuneRevertSnapshot(snapshot)
      setTuneAutoLastRunAt(Date.now())
      setTuneIterateError(null)
      setTuneSaveError(null)
      setTuneRevertError(null)
    },
    onSuccess: (result) => {
      setTuneIterateError(null)
      setTuneIterateResponse(result)
      if (result.updated_trader) {
        applyTraderDraftSettings(result.updated_trader)
        setTuneDraftDirty(false)
        setTuneDraftTraderId(result.updated_trader.id)
      }
      refreshAll()
    },
    onError: (error: unknown, variables) => {
      if (variables.trigger === 'manual') {
        setTuneIterateError(errorMessage(error, 'Failed to run agent'))
      }
    },
  })

  const createTraderMutation = useMutation({
    mutationFn: async () => {
      const copyFromTraderId = String(draftCopyFromTraderId || '').trim()
      const parsedParams = parseJsonObject(draftParams || '{}')
      if (!parsedParams.value) {
        throw new Error(`Strategy params JSON error: ${parsedParams.error || 'invalid object'}`)
      }

      const parsedRisk = parseJsonObject(draftRisk || '{}')
      if (!parsedRisk.value) {
        throw new Error(`Risk limits JSON error: ${parsedRisk.error || 'invalid object'}`)
      }

      const parsedMetadata = parseJsonObject(draftMetadata || '{}')
      if (!parsedMetadata.value) {
        throw new Error(`Metadata JSON error: ${parsedMetadata.error || 'invalid object'}`)
      }

      if (effectiveDraftSources.length === 0) {
        throw new Error('Enable at least one source.')
      }
      const tradersEnabled = effectiveDraftSources.includes('traders')
      const tradersScope = normalizeTradersScopeConfig(parsedParams.value.traders_scope)
      if (tradersEnabled && tradersScope.modes.includes('individual') && tradersScope.individual_wallets.length === 0) {
        throw new Error('Select at least one individual wallet for wallet scope.')
      }
      if (tradersEnabled && tradersScope.modes.includes('group') && tradersScope.group_ids.length === 0) {
        throw new Error('Select at least one group for wallet scope.')
      }

      const payload: Record<string, unknown> = {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        mode: draftMode,
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        source_configs: buildDraftSourceConfigs(parsedParams.value),
        risk_limits: parsedRisk.value,
        metadata: parsedMetadata.value,
        is_enabled: true,
        is_paused: false,
      }

      if (copyFromTraderId) {
        payload.copy_from_trader_id = copyFromTraderId
      }

      return createTrader(payload)
    },
    onMutate: () => {
      const previewName = draftName.trim() || 'Creating bot...'
      setCreatingTraderPreview({
        name: previewName,
        mode: draftMode,
      })
      setTraderFlyoutOpen(false)
      setSaveError(null)
    },
    onSuccess: (trader) => {
      setCreatingTraderPreview(null)
      upsertTraderInCache(trader)
      setSaveError(null)
      setTraderFlyoutOpen(false)
      setSelectedTraderId(trader.id)
      refreshAll()
    },
    onError: (error: unknown) => {
      const message = errorMessage(error, 'Failed to create bot')
      setTraderFlyoutOpen(true)
      setSaveError(message)
      setCreatingTraderPreview(null)
    },
  })

  const saveTraderMutation = useMutation({
    mutationFn: async (traderId: string) => {
      const parsedParams = parseJsonObject(draftParams || '{}')
      if (!parsedParams.value) {
        throw new Error(`Strategy params JSON error: ${parsedParams.error || 'invalid object'}`)
      }

      const parsedRisk = parseJsonObject(draftRisk || '{}')
      if (!parsedRisk.value) {
        throw new Error(`Risk limits JSON error: ${parsedRisk.error || 'invalid object'}`)
      }

      const parsedMetadata = parseJsonObject(draftMetadata || '{}')
      if (!parsedMetadata.value) {
        throw new Error(`Metadata JSON error: ${parsedMetadata.error || 'invalid object'}`)
      }

      if (effectiveDraftSources.length === 0) {
        throw new Error('Enable at least one source.')
      }
      const tradersEnabled = effectiveDraftSources.includes('traders')
      const tradersScope = normalizeTradersScopeConfig(parsedParams.value.traders_scope)
      if (tradersEnabled && tradersScope.modes.includes('individual') && tradersScope.individual_wallets.length === 0) {
        throw new Error('Select at least one individual wallet for wallet scope.')
      }
      if (tradersEnabled && tradersScope.modes.includes('group') && tradersScope.group_ids.length === 0) {
        throw new Error('Select at least one group for wallet scope.')
      }

      return updateTrader(traderId, {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        mode: draftMode,
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        source_configs: buildDraftSourceConfigs(parsedParams.value),
        risk_limits: parsedRisk.value,
        metadata: parsedMetadata.value,
      })
    },
    onSuccess: () => {
      setSaveError(null)
      setTraderFlyoutOpen(false)
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to save bot'))
    },
  })

  const deleteTraderMutation = useMutation({
    mutationFn: async ({ traderId, action }: { traderId: string; action: 'block' | 'disable' | 'force_delete' }) => {
      return deleteTrader(traderId, { action })
    },
    onSuccess: (result, variables) => {
      setSaveError(null)
      setDeleteForceConfirm(false)
      if (result.status === 'deleted') {
        if (selectedTraderId === variables.traderId) {
          const fallback = traders.find((row) => row.id !== variables.traderId)
          setSelectedTraderId(fallback?.id || null)
        }
        setTraderFlyoutOpen(false)
      }
      if (result.status === 'disabled') {
        setDeleteAction('block')
      }
      refreshAll()
    },
    onError: (error: unknown) => {
      const liveExposure = parseTraderDeleteLiveExposure(error)
      if (liveExposure) {
        setDeleteAction('force_delete')
        setDeleteForceConfirm(false)
        setSaveError(
          [
            liveExposure.message,
            liveExposure.summary ? `Current exposure: ${liveExposure.summary}.` : null,
            'Select Force Delete and confirm the override to permanently delete now.',
          ]
            .filter(Boolean)
            .join(' ')
        )
        return
      }
      setSaveError(errorMessage(error, 'Failed to delete or disable bot'))
    },
  })

  const worker = overviewQuery.data?.worker
  const orchestratorControl = overviewQuery.data?.control
  const orchestratorConfig = overviewQuery.data?.config || null
  const metrics = overviewQuery.data?.metrics
  const executionLatency = metrics?.execution_latency || null
  const executionLatencyWindowLabel = formatLatencyWindow(executionLatency?.rolling_window_seconds)
  const executionLatencySampleCount = toNumber(executionLatency?.sample_count)
  const executionLatencyOverall = executionLatency?.overall || null
  const executionLatencyOverallLabel = formatLatencyPercentilePair(
    executionLatencyOverall,
    'ws_release_to_submit_start_ms'
  )
  const executionLatencyTargetMs = toNumber(executionLatency?.internal_sla_target_ms)
  const executionLatencyOverallP95 = latencyStagePercentiles(
    executionLatencyOverall,
    'ws_release_to_submit_start_ms'
  ).p95
  const executionLatencySlaBreached = Boolean(
    executionLatencyTargetMs !== null &&
    executionLatencyOverallP95 !== null &&
    executionLatencyOverallP95 > executionLatencyTargetMs
  )
  const worstLatencySource = worstLatencyGroup(executionLatency, 'by_source', 'ws_release_to_submit_start_ms')
  const worstLatencyStrategy = worstLatencyGroup(executionLatency, 'by_strategy', 'ws_release_to_submit_start_ms')
  const worstLatencySourceLabel = worstLatencySource.label
    ? `${worstLatencySource.label} ${formatLatencyPercentilePair({ ws_release_to_submit_start_ms: worstLatencySource }, 'ws_release_to_submit_start_ms')}`
    : '—'
  const worstLatencyStrategyLabel = worstLatencyStrategy.label
    ? `${worstLatencyStrategy.label} ${formatLatencyPercentilePair({ ws_release_to_submit_start_ms: worstLatencyStrategy }, 'ws_release_to_submit_start_ms')}`
    : '—'
  const selectedTraderLatencyBucket = selectedTrader ? executionLatency?.by_trader?.[selectedTrader.id] || null : null
  const selectedTraderLatencyLabel = formatLatencyPercentilePair(
    selectedTraderLatencyBucket,
    'ws_release_to_submit_start_ms'
  )
  const selectedTraderLatencyP95 = latencyStagePercentiles(
    selectedTraderLatencyBucket,
    'ws_release_to_submit_start_ms'
  ).p95
  const selectedTraderArmedToReleaseLabel = formatLatencyPercentilePair(
    selectedTraderLatencyBucket,
    'armed_to_ws_release_ms'
  )
  const selectedTraderReleaseToDecisionLabel = formatLatencyPercentilePair(
    selectedTraderLatencyBucket,
    'ws_release_to_decision_ms'
  )
  const selectedTraderLatencySlaBreached = Boolean(
    executionLatencyTargetMs !== null &&
    selectedTraderLatencyP95 !== null &&
    selectedTraderLatencyP95 > executionLatencyTargetMs
  )
  const killSwitchOn = Boolean(orchestratorControl?.kill_switch)
  const killSwitchSwitchValue = killSwitchMutation.isPending && typeof killSwitchMutation.variables === 'boolean'
    ? killSwitchMutation.variables
    : killSwitchOn
  const killSwitchStatusLabel = killSwitchMutation.isPending
    ? killSwitchSwitchValue ? 'BLOCKING...' : 'OPENING...'
    : killSwitchOn ? 'BLOCKED' : 'OPEN'
  const orchestratorEnabled = Boolean(orchestratorControl?.is_enabled) && !Boolean(orchestratorControl?.is_paused)
  const workerActivity = String(worker?.current_activity || '').trim().toLowerCase()
  const orchestratorWorkerRunning = Boolean(worker?.running)
  const orchestratorRunning = orchestratorEnabled && orchestratorWorkerRunning
  const orchestratorControlMismatch = orchestratorWorkerRunning && !orchestratorEnabled
  const orchestratorStartStopActive = orchestratorEnabled
  const orchestratorBlocked = orchestratorEnabled && !orchestratorWorkerRunning && workerActivity.startsWith('blocked')
  const orchestratorStatusLabel = orchestratorBlocked
    ? 'BLOCKED'
    : orchestratorWorkerRunning
      ? 'RUNNING'
      : 'STOPPED'
  const orchestratorStartRequestPending =
    startBySelectedAccountMutation.isPending &&
    !orchestratorEnabled &&
    !orchestratorWorkerRunning
  const orchestratorStopRequestPending =
    stopByModeMutation.isPending &&
    (orchestratorEnabled || orchestratorWorkerRunning)

  const controlBusy =
    orchestratorStartRequestPending ||
    orchestratorStopRequestPending ||
    killSwitchMutation.isPending
  const traderFlyoutBusy =
    createTraderMutation.isPending ||
    saveTraderMutation.isPending ||
    deleteTraderMutation.isPending
  const globalSettingsBusy = updateGlobalSettingsMutation.isPending

  useEffect(() => {
    if (globalSettingsFlyoutOpen) return
    setGlobalSettingsDraft(buildGlobalSettingsDraft(orchestratorConfig, liveExecutionSettings))
  }, [globalSettingsFlyoutOpen, orchestratorConfig, liveExecutionSettings])

  const traderNameById = useMemo(
    () => Object.fromEntries(traders.map((trader) => [trader.id, trader.name])) as Record<string, string>,
    [traders]
  )
  const closeMarketModal = () => {
    setMarketModalState(null)
    setMarketModalSellError(null)
    setMarketModalSellSuccess(null)
  }

  const openTradeMarketModal = (params: {
    market: CryptoMarket | null
    order: TraderOrder
    directionSide: DirectionSide | null
    directionLabel: string
    yesLabel: string | null
    noLabel: string | null
    statusSummary: string
    executionSummary: string
    outcomeSummary: string | null
    links: {
      polymarket: string | null
      kalshi: string | null
    }
  }) => {
    setMarketModalSellError(null)
    setMarketModalSellSuccess(null)
    const {
      market,
      order,
      directionSide,
      directionLabel,
      yesLabel,
      noLabel,
      statusSummary,
      executionSummary,
      outcomeSummary,
      links,
    } = params
    const marketAliases = collectOrderMarketAliasIds(order)
    const resolvedMarket = market || resolveCryptoMarketFromAliases([
      order.market_id,
      ...marketAliases,
    ])
    const traderId = cleanText(order.trader_id) || null
    const traderName = traderId ? (traderNameById[traderId] || shortId(traderId)) : 'All Bots'
    const modeSummary = String(order.mode || '').trim().toUpperCase() || 'N/A'
    setMarketModalState({
      market: resolvedMarket,
      scope: {
        kind: 'trade',
        traderId,
        traderName,
        marketId: String(order.market_id || ''),
        marketIds: marketAliases,
        marketQuestion: String(order.market_question || order.market_id || resolvedMarket?.question || 'Unknown market'),
        directionSide,
        directionLabel,
        yesLabel,
        noLabel,
        anchorOrderId: String(order.id || ''),
        sourceSummary: String(order.source || '').trim().toUpperCase() || 'UNKNOWN',
        statusSummary,
        modeSummary,
        executionSummary,
        outcomeSummary,
        links,
      },
    })
  }

  const openPositionMarketModal = (params: {
    market: CryptoMarket | null
    row: PositionBookRow
    traderId?: string | null
    traderName?: string
  }) => {
    setMarketModalSellError(null)
    setMarketModalSellSuccess(null)
    const { market, row } = params
    const marketAliases = row.marketAliases.length > 0 ? row.marketAliases : [normalizeMarketAlias(row.marketId)]
    const resolvedMarket = market || resolveCryptoMarketFromAliases([row.marketId, ...marketAliases])
    const traderId = params.traderId ?? row.traderId ?? null
    const traderName = params.traderName || row.traderName || (traderId ? (traderNameById[traderId] || shortId(traderId)) : 'All Bots')
    setMarketModalState({
      market: resolvedMarket,
      scope: {
        kind: 'position',
        traderId,
        traderName,
        marketId: row.marketId,
        marketIds: marketAliases,
        marketQuestion: row.marketQuestion,
        directionSide: row.directionSide,
        directionLabel: row.direction,
        yesLabel: row.directionSide === 'YES' && !isGenericDirectionLabel(row.direction) ? row.direction : null,
        noLabel: row.directionSide === 'NO' && !isGenericDirectionLabel(row.direction) ? row.direction : null,
        anchorOrderId: null,
        sourceSummary: row.sourceSummary,
        statusSummary: row.statusSummary,
        modeSummary: `${row.liveOrderCount}L/${row.shadowOrderCount}S`,
        executionSummary: row.executionSummary,
        outcomeSummary: null,
        links: row.links,
      },
    })
  }

  const handleSellModalOrder = (order: TraderOrder) => {
    if (!marketModalState?.scope.traderId) {
      setMarketModalSellError('This trade is not attached to a specific bot and cannot be sold from this view.')
      return
    }
    const orderId = String(order.id || '').trim()
    if (!orderId) {
      setMarketModalSellError('Trade is missing an order id and cannot be sold.')
      return
    }
    sellTradeNowMutation.mutate({
      traderId: marketModalState.scope.traderId,
      orderId,
    })
  }

  const globalSummary = useMemo(() => {
    let resolved = 0
    let wins = 0
    let losses = 0
    let failed = 0
    let open = 0
    let totalNotional = 0
    let resolvedPnl = 0
    let edgeSum = 0
    let edgeCount = 0
    let confidenceSum = 0
    let confidenceCount = 0

    const byTrader = new Map<string, {
      traderId: string
      traderName: string
      orders: number
      open: number
      resolved: number
      pnl: number
      notional: number
      wins: number
      losses: number
      latest_activity_ts: number
    }>()

    const bySource = new Map<string, {
      source: string
      orders: number
      resolved: number
      pnl: number
      notional: number
      wins: number
      losses: number
    }>()

    for (const order of allOrders) {
      const status = normalizeStatus(order.status)
      const notional = Math.abs(toNumber(order.notional_usd))
      const pnl = toNumber(order.actual_profit)
      const edge = toNumber(order.edge_percent)
      const confidence = toNumber(order.confidence)
      const traderId = String(order.trader_id || 'unknown')
      const traderName = traderNameById[traderId] || shortId(traderId)
      const source = String(order.source || 'unknown')
      const orderTs = Math.max(
        toTs(order.updated_at),
        toTs(order.executed_at),
        toTs(order.created_at)
      )

      totalNotional += notional
      if (edge !== 0) {
        edgeSum += edge
        edgeCount += 1
      }
      if (confidence !== 0) {
        confidenceSum += confidence
        confidenceCount += 1
      }

      if (!byTrader.has(traderId)) {
        byTrader.set(traderId, {
          traderId,
          traderName,
          orders: 0,
          open: 0,
          resolved: 0,
          pnl: 0,
          notional: 0,
          wins: 0,
          losses: 0,
          latest_activity_ts: 0,
        })
      }

      if (!bySource.has(source)) {
        bySource.set(source, {
          source,
          orders: 0,
          resolved: 0,
          pnl: 0,
          notional: 0,
          wins: 0,
          losses: 0,
        })
      }

      const traderRow = byTrader.get(traderId)
      const sourceRow = bySource.get(source)
      if (!traderRow || !sourceRow) continue

      traderRow.orders += 1
      traderRow.notional += notional
      traderRow.latest_activity_ts = Math.max(traderRow.latest_activity_ts, orderTs)
      sourceRow.orders += 1
      sourceRow.notional += notional

      if (OPEN_ORDER_STATUSES.has(status)) {
        open += 1
        traderRow.open += 1
      }

      if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolved += 1
        resolvedPnl += pnl
        traderRow.resolved += 1
        sourceRow.resolved += 1
        traderRow.pnl += pnl
        sourceRow.pnl += pnl

        if (pnl > 0) {
          wins += 1
          traderRow.wins += 1
          sourceRow.wins += 1
        }
        if (pnl < 0) {
          losses += 1
          traderRow.losses += 1
          sourceRow.losses += 1
        }
      }

      if (FAILED_ORDER_STATUSES.has(status)) {
        failed += 1
      }
    }

    const winRate = resolved > 0 ? (wins / resolved) * 100 : 0
    const traderRows = Array.from(byTrader.values())
      .sort((a, b) => {
        if (a.pnl !== b.pnl) return b.pnl - a.pnl
        if (a.open !== b.open) return b.open - a.open
        return a.traderName.localeCompare(b.traderName)
      })

    const sourceRows = Array.from(bySource.values())
      .sort((a, b) => b.orders - a.orders)
      .slice(0, 8)

    return {
      open,
      resolved,
      wins,
      losses,
      failed,
      totalNotional,
      resolvedPnl,
      winRate,
      avgEdge: edgeCount > 0 ? edgeSum / edgeCount : 0,
      avgConfidence: confidenceCount > 0 ? confidenceSum / confidenceCount : 0,
      traderRows,
      sourceRows,
    }
  }, [allOrders, traderNameById])

  const globalPositionBook = useMemo(
    () => buildPositionBookRows(allOrders, traderNameById, decisionSignalPayloadByDecisionId),
    [allOrders, traderNameById, decisionSignalPayloadByDecisionId]
  )

  const selectedPositionBook = useMemo(
    () => globalPositionBook.filter((row) => row.traderId === selectedTraderId),
    [globalPositionBook, selectedTraderId]
  )

  const selectedTraderSummary = useMemo(() => {
    let resolved = 0
    let wins = 0
    let losses = 0
    let failed = 0
    let open = 0
    let pnl = 0
    let notional = 0
    let edgeSum = 0
    let edgeCount = 0
    let confidenceSum = 0
    let confidenceCount = 0

    for (const order of selectedOrders) {
      const status = normalizeStatus(order.status)
      const orderPnl = toNumber(order.actual_profit)
      const orderNotional = Math.abs(toNumber(order.notional_usd))
      const edge = toNumber(order.edge_percent)
      const confidence = toNumber(order.confidence)
      notional += orderNotional

      if (OPEN_ORDER_STATUSES.has(status)) {
        open += 1
      }
      if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolved += 1
        pnl += orderPnl
        if (orderPnl > 0) wins += 1
        if (orderPnl < 0) losses += 1
      }
      if (FAILED_ORDER_STATUSES.has(status)) {
        failed += 1
      }
      if (edge !== 0) {
        edgeSum += edge
        edgeCount += 1
      }
      if (confidence !== 0) {
        confidenceSum += confidence
        confidenceCount += 1
      }
    }

    const decisions = selectedDecisions.length
    const selectedDecisionsCount = selectedDecisions.filter(
      (decision) => String(decision.decision).toLowerCase() === 'selected'
    ).length

    return {
      resolved,
      wins,
      losses,
      failed,
      open,
      pnl,
      notional,
      winRate: resolved > 0 ? (wins / resolved) * 100 : 0,
      decisions,
      selectedDecisions: selectedDecisionsCount,
      events: selectedEvents.length,
      conversion: decisions > 0 ? (selectedOrders.length / decisions) * 100 : 0,
      selectionRate: decisions > 0 ? (selectedDecisionsCount / decisions) * 100 : 0,
      avgEdge: edgeCount > 0 ? edgeSum / edgeCount : 0,
      avgConfidence: confidenceCount > 0 ? confidenceSum / confidenceCount : 0,
    }
  }, [selectedOrders, selectedDecisions, selectedEvents.length])

  const selectedPerformance = useMemo(() => {
    const dimensions = selectedOrders.map((order) => {
      const decisionId = cleanText(order.decision_id)
      const decision = decisionId ? selectedDecisionById.get(decisionId) || null : null
      return extractOrderPerformanceDimensions(order, decision)
    })

    const timeframeRows = buildPerformanceBuckets(selectedOrders, (_order, index) => {
      const timeframe = dimensions[index]?.timeframe || 'unclassified'
      return { key: timeframe, label: timeframe }
    }).sort((left, right) => {
      const leftRank = PERFORMANCE_TIMEFRAME_ORDER[left.key] ?? 99
      const rightRank = PERFORMANCE_TIMEFRAME_ORDER[right.key] ?? 99
      if (leftRank !== rightRank) return leftRank - rightRank
      return performanceBucketSort(left, right)
    })

    const modeRows = buildPerformanceBuckets(selectedOrders, (_order, index) => {
      const mode = dimensions[index]?.mode || 'unclassified'
      return { key: mode, label: mode }
    }).sort((left, right) => {
      const leftRank = PERFORMANCE_MODE_ORDER[left.key] ?? 99
      const rightRank = PERFORMANCE_MODE_ORDER[right.key] ?? 99
      if (leftRank !== rightRank) return leftRank - rightRank
      return performanceBucketSort(left, right)
    })

    const subStrategyRows = buildPerformanceBuckets(selectedOrders, (_order, index) => {
      const subStrategy = dimensions[index]?.subStrategy || 'unclassified'
      return { key: subStrategy, label: subStrategy }
    }).sort(performanceBucketSort)

    const timeframeModeRows = buildPerformanceBuckets(selectedOrders, (_order, index) => {
      const timeframe = dimensions[index]?.timeframe || 'unclassified'
      const mode = dimensions[index]?.mode || 'unclassified'
      return {
        key: `${mode}|${timeframe}`,
        label: `${mode} + ${timeframe}`,
      }
    }).sort(performanceBucketSort)

    const strategyRows = buildPerformanceBuckets(selectedOrders, (_order, index) => {
      const strategyKey = dimensions[index]?.strategyKey || 'unknown'
      return {
        key: strategyKey,
        label: strategyLabelForKey(strategyKey, sourceCatalog),
      }
    }).sort(performanceBucketSort)

    const sourceRows = buildPerformanceBuckets(selectedOrders, (order) => {
      const sourceKey = normalizeSourceKey(String(order.source || '')) || 'unknown'
      const sourceLabel = sourceCatalog.find((item) => normalizeSourceKey(item.key) === sourceKey)?.label || sourceKey.toUpperCase()
      return {
        key: sourceKey,
        label: sourceLabel,
      }
    }).sort(performanceBucketSort)

    let resolvedPnl = 0
    let resolvedNotional = 0
    let resolved = 0
    let wins = 0
    let losses = 0
    let failed = 0
    let open = 0
    let allowanceErrorCount = 0
    let gasErrorCount = 0

    for (const order of selectedOrders) {
      const status = normalizeStatus(order.status)
      const pnl = toNumber(order.actual_profit)
      const notional = Math.abs(toNumber(order.notional_usd))
      if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolved += 1
        resolvedPnl += pnl
        resolvedNotional += notional
        if (pnl > 0) wins += 1
        if (pnl < 0) losses += 1
      }
      if (FAILED_ORDER_STATUSES.has(status)) failed += 1
      if (OPEN_ORDER_STATUSES.has(status)) open += 1

      const payload = isRecord(order.payload) ? order.payload : {}
      const pendingExit = isRecord(payload.pending_live_exit) ? payload.pending_live_exit : {}
      const allowanceText = [
        cleanText(order.error_message),
        cleanText(payload.error_message),
        cleanText(payload.error),
        cleanText(pendingExit.last_error),
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
      if (isAllowanceErrorText(allowanceText) || allowanceText.includes('not enough balance / allowance')) {
        allowanceErrorCount += 1
      }
      if (isGasErrorText(allowanceText)) {
        gasErrorCount += 1
      }
    }

    return {
      timeframeRows,
      modeRows,
      subStrategyRows,
      timeframeModeRows,
      sourceRows,
      strategyRows,
      resolved,
      wins,
      losses,
      failed,
      open,
      resolvedPnl,
      resolvedNotional,
      roiPercent: resolvedNotional > 0 ? (resolvedPnl / resolvedNotional) * 100 : 0,
      allowanceErrorCount,
      gasErrorCount,
    }
  }, [selectedDecisionById, selectedOrders, sourceCatalog])

  const selectedPerformanceConfig = useMemo(() => {
    type MutablePerformanceSection = PerformanceSection & { sortIndex: number }

    const sourceLabelByKey = new Map<string, string>()
    for (const source of sourceCards) {
      sourceLabelByKey.set(normalizeSourceKey(source.key), source.label)
    }

    const currentConfigs = selectedTraderSourceConfigs.map((sourceConfig, index) => {
      const sourceKey = normalizeSourceKey(String(sourceConfig.source_key || ''))
      const strategyKey = normalizeStrategyKeyForSource(sourceKey, sourceConfig.strategy_key)
      const strategyVersion = normalizeStrategyVersion(sourceConfig.strategy_version)
      const detail = sourceStrategyDetailsLookup[sourceKey]?.[strategyKey] || null
      const sourceLabel = sourceLabelByKey.get(sourceKey) || sourceKey.toUpperCase() || 'UNKNOWN'
      const strategyLabel = detail?.label || strategyLabelForKey(strategyKey, sourceCards)
      const strategyVersionLabel = formatStrategyVersionLabel(strategyVersion)
      const defaultParams = isRecord(detail?.defaultParams) ? detail.defaultParams : {}
      const currentValues = {
        ...defaultParams,
        ...(isRecord(sourceConfig.strategy_params) ? sourceConfig.strategy_params : {}),
      }
      const observedFieldCandidates = Object.entries(currentValues)
        .map(([key, value]) => inferStrategyParamField(key, value))
        .filter((field): field is Record<string, unknown> => Boolean(field))
      const paramFields = dedupeStrategyParamFields([
        ...(Array.isArray(detail?.paramFields) ? detail.paramFields : []),
        ...observedFieldCandidates,
      ])

      return {
        sortIndex: index,
        sourceKey,
        sourceLabel,
        strategyKey,
        strategyLabel,
        strategyVersion,
        strategyVersionLabel,
        sectionKey: buildPerformanceSectionKey(sourceKey, strategyKey, strategyVersion),
        sectionLabel: buildPerformanceSectionLabel(sourceLabel, strategyLabel, strategyVersionLabel),
        paramFields,
        values: currentValues,
      }
    })

    const sectionsByKey = new Map<string, MutablePerformanceSection>()
    const upsertSection = (payload: {
      sortIndex: number
      sectionKey: string
      sectionLabel: string
      sourceKey: string
      sourceLabel: string
      strategyKey: string
      strategyLabel: string
      strategyVersion: number | null
      strategyVersionLabel: string
      values: Record<string, unknown>
      paramFields: Array<Record<string, unknown>>
    }) => {
      const existing = sectionsByKey.get(payload.sectionKey)
      const mergedFields = dedupeStrategyParamFields([
        ...(existing?.paramFields || []),
        ...payload.paramFields,
      ])
      const mergedValues = existing
        ? { ...existing.values, ...payload.values }
        : { ...payload.values }
      sectionsByKey.set(payload.sectionKey, {
        sectionKey: payload.sectionKey,
        sectionLabel: payload.sectionLabel,
        sourceKey: payload.sourceKey,
        sourceLabel: payload.sourceLabel,
        strategyKey: payload.strategyKey,
        strategyLabel: payload.strategyLabel,
        strategyVersion: payload.strategyVersion,
        strategyVersionLabel: payload.strategyVersionLabel,
        paramFields: mergedFields,
        groups: groupStrategyParamFields(mergedFields),
        fieldKeys: mergedFields
          .map((field) => String(field.key || '').trim())
          .filter(Boolean),
        values: mergedValues,
        sortIndex: existing ? Math.min(existing.sortIndex, payload.sortIndex) : payload.sortIndex,
      })
    }

    for (const config of currentConfigs) {
      upsertSection(config)
    }

    let fallbackOrderCount = 0
    const snapshots: PerformanceOrderSnapshot[] = []

    for (const order of selectedOrders) {
      const decisionId = cleanText(order.decision_id)
      const decision = decisionId ? selectedDecisionById.get(decisionId) || null : null
      const sourceKey = normalizeSourceKey(String(order.source || ''))
      const dimensionMeta = extractOrderPerformanceDimensions(order, decision)
      const strategyKey = normalizeStrategyKeyForSource(
        sourceKey,
        cleanText(order.strategy_key) || cleanText(decision?.strategy_key) || dimensionMeta.strategyKey,
      )
      const explicitVersion = normalizeStrategyVersion(order.strategy_version)
      const matchingCurrent =
        currentConfigs.find((config) =>
          config.sourceKey === sourceKey
          && config.strategyKey === strategyKey
          && config.strategyVersion === explicitVersion
        )
        || (explicitVersion === null
          ? currentConfigs.find((config) => config.sourceKey === sourceKey && config.strategyKey === strategyKey)
          : currentConfigs.find((config) =>
            config.sourceKey === sourceKey
            && config.strategyKey === strategyKey
            && config.strategyVersion === null
          ))
        || null
      const strategyVersion = explicitVersion ?? matchingCurrent?.strategyVersion ?? null
      const detail = sourceStrategyDetailsLookup[sourceKey]?.[strategyKey] || null
      const sourceLabel = sourceLabelByKey.get(sourceKey) || sourceKey.toUpperCase() || 'UNKNOWN'
      const strategyLabel = detail?.label || matchingCurrent?.strategyLabel || strategyLabelForKey(strategyKey, sourceCards)
      const strategyVersionLabel = formatStrategyVersionLabel(strategyVersion)
      const sectionKey = buildPerformanceSectionKey(sourceKey, strategyKey, strategyVersion)
      const sectionLabel = buildPerformanceSectionLabel(sourceLabel, strategyLabel, strategyVersionLabel)
      const extracted = extractOrderPerformanceParams(
        order,
        decision,
        matchingCurrent?.values || null,
      )
      if (extracted.usedCurrentConfigFallback) {
        fallbackOrderCount += 1
      }

      const inferredFields = Object.entries(extracted.params)
        .map(([key, value]) => inferStrategyParamField(key, value))
        .filter((field): field is Record<string, unknown> => Boolean(field))
      upsertSection({
        sortIndex: matchingCurrent?.sortIndex ?? 10_000 + snapshots.length,
        sectionKey,
        sectionLabel,
        sourceKey,
        sourceLabel,
        strategyKey,
        strategyLabel,
        strategyVersion,
        strategyVersionLabel,
        values: matchingCurrent?.values || {},
        paramFields: [
          ...(Array.isArray(detail?.paramFields) ? detail.paramFields : []),
          ...inferredFields,
        ],
      })

      snapshots.push({
        order,
        sourceKey,
        sourceLabel,
        strategyKey,
        strategyLabel,
        strategyVersion,
        strategyVersionLabel,
        sectionKey,
        sectionLabel,
        params: extracted.params,
        usedCurrentConfigFallback: extracted.usedCurrentConfigFallback,
      })
    }

    const sections = Array.from(sectionsByKey.values())
      .sort((left, right) => {
        if (left.sortIndex !== right.sortIndex) return left.sortIndex - right.sortIndex
        return left.sectionLabel.localeCompare(right.sectionLabel)
      })
      .map(({ sortIndex: _sortIndex, ...section }) => section)

    const configurationBuckets = buildPerformanceBuckets(
      snapshots.map((snapshot) => snapshot.order),
      (_order, index) => ({
        key: snapshots[index]?.sectionKey || 'unknown',
        label: snapshots[index]?.sectionLabel || 'Unknown configuration',
      }),
    )
    const configurationBucketByKey = new Map(configurationBuckets.map((row) => [row.key, row]))
    const configurationRows: PerformanceConfigurationRow[] = sections
      .map((section) => {
        const bucket = configurationBucketByKey.get(section.sectionKey)
        return {
          sectionKey: section.sectionKey,
          sectionLabel: section.sectionLabel,
          sourceLabel: section.sourceLabel,
          strategyLabel: section.strategyLabel,
          strategyVersionLabel: section.strategyVersionLabel,
          key: section.sectionKey,
          label: section.sectionLabel,
          orders: bucket?.orders || 0,
          open: bucket?.open || 0,
          resolved: bucket?.resolved || 0,
          wins: bucket?.wins || 0,
          losses: bucket?.losses || 0,
          failed: bucket?.failed || 0,
          resolvedNotional: bucket?.resolvedNotional || 0,
          pnl: bucket?.pnl || 0,
          roiPercent: bucket?.roiPercent || 0,
          fullLosses: bucket?.fullLosses || 0,
        }
      })
      .sort((left, right) => {
        if (Math.abs(left.pnl) !== Math.abs(right.pnl)) return Math.abs(right.pnl) - Math.abs(left.pnl)
        if (left.orders !== right.orders) return right.orders - left.orders
        return left.sectionLabel.localeCompare(right.sectionLabel)
      })

    const paramSummaryBySection: Record<string, PerformanceParamSummaryRow[]> = {}
    const paramBucketsBySection: Record<string, Record<string, PerformanceParamValueRow[]>> = {}

    for (const section of sections) {
      const sectionSnapshots = snapshots.filter((snapshot) => snapshot.sectionKey === section.sectionKey)
      const summaryRows: PerformanceParamSummaryRow[] = []
      const bucketsForSection: Record<string, PerformanceParamValueRow[]> = {}

      for (const fieldKey of section.fieldKeys) {
        const currentValueMeta = performanceParamBucketMeta(section.values[fieldKey])
        const bucketRows = buildPerformanceBuckets(
          sectionSnapshots.map((snapshot) => snapshot.order),
          (_order, index) => {
            const bucketMeta = performanceParamBucketMeta(sectionSnapshots[index]?.params[fieldKey])
            return { key: bucketMeta.key, label: bucketMeta.label }
          },
        )
          .map((row) => ({
            ...row,
            valueLabel: row.label,
            isCurrent: row.key === currentValueMeta.key,
            isMissing: row.key === 'not_recorded',
          }))
          .sort((left, right) => {
            if (left.isCurrent !== right.isCurrent) return left.isCurrent ? -1 : 1
            return performanceBucketSort(left, right)
          })

        const currentBucket = bucketRows.find((row) => row.isCurrent) || null
        const matchingField = section.paramFields.find((field) => String(field.key || '').trim() === fieldKey) || null
        summaryRows.push({
          key: fieldKey,
          label: String(matchingField?.label || humanizeStrategyParamLabel(fieldKey)),
          currentValueLabel: currentValueMeta.label,
          observedValueCount: bucketRows.length,
          currentResolved: currentBucket?.resolved || 0,
          currentPnl: currentBucket?.pnl || 0,
          currentRoiPercent: currentBucket?.roiPercent || 0,
          hasVariation: bucketRows.length > 1,
        })
        bucketsForSection[fieldKey] = bucketRows
      }

      summaryRows.sort((left, right) => {
        if (left.hasVariation !== right.hasVariation) return left.hasVariation ? -1 : 1
        if (left.observedValueCount !== right.observedValueCount) return right.observedValueCount - left.observedValueCount
        return left.label.localeCompare(right.label)
      })
      paramSummaryBySection[section.sectionKey] = summaryRows
      paramBucketsBySection[section.sectionKey] = bucketsForSection
    }

    return {
      sections,
      snapshots,
      configurationRows,
      fallbackOrderCount,
      paramSummaryBySection,
      paramBucketsBySection,
    }
  }, [
    selectedDecisionById,
    selectedOrders,
    selectedTraderSourceConfigs,
    sourceCards,
    sourceStrategyDetailsLookup,
  ])

  const latencyStageRows = useMemo<LatencyStageRow[]>(() => {
    return LATENCY_STAGE_OPTIONS.map((stage) => ({
      key: stage.key,
      label: stage.label,
      traderLatencyLabel: formatLatencyPercentilePair(selectedTraderLatencyBucket, stage.key),
      overallLatencyLabel: formatLatencyPercentilePair(executionLatencyOverall, stage.key),
    }))
  }, [executionLatencyOverall, selectedTraderLatencyBucket])

  const latencySourceRows = useMemo(
    () => buildLatencyGroupRows(executionLatency, 'by_source', 'ws_release_to_submit_start_ms', sourceCards).slice(0, 10),
    [executionLatency, sourceCards]
  )
  const latencyStrategyRows = useMemo(
    () => buildLatencyGroupRows(executionLatency, 'by_strategy', 'ws_release_to_submit_start_ms', sourceCards).slice(0, 10),
    [executionLatency, sourceCards]
  )

  const activePerformanceSection = useMemo(
    () => selectedPerformanceConfig.sections.find((section) => section.sectionKey === performanceSectionKey) || selectedPerformanceConfig.sections[0] || null,
    [performanceSectionKey, selectedPerformanceConfig.sections]
  )
  const activePerformanceParamSummaryRows = useMemo(
    () => activePerformanceSection ? (selectedPerformanceConfig.paramSummaryBySection[activePerformanceSection.sectionKey] || []) : [],
    [activePerformanceSection, selectedPerformanceConfig.paramSummaryBySection]
  )
  const activePerformanceParamRows = useMemo(
    () => {
      if (!activePerformanceSection || !performanceParamKey) return []
      return selectedPerformanceConfig.paramBucketsBySection[activePerformanceSection.sectionKey]?.[performanceParamKey] || []
    },
    [activePerformanceSection, performanceParamKey, selectedPerformanceConfig.paramBucketsBySection]
  )
  const activePerformanceParamSummaryByKey = useMemo(() => {
    const out = new Map<string, PerformanceParamSummaryRow>()
    for (const row of activePerformanceParamSummaryRows) {
      out.set(row.key, row)
    }
    return out
  }, [activePerformanceParamSummaryRows])

  useEffect(() => {
    if (selectedPerformanceConfig.sections.length === 0) {
      setPerformanceSectionKey('')
      return
    }
    setPerformanceSectionKey((current) => {
      if (current && selectedPerformanceConfig.sections.some((section) => section.sectionKey === current)) {
        return current
      }
      return selectedPerformanceConfig.sections[0].sectionKey
    })
  }, [selectedPerformanceConfig.sections])

  useEffect(() => {
    if (!activePerformanceSection) {
      setPerformanceParamKey('')
      return
    }
    const summaryRows = selectedPerformanceConfig.paramSummaryBySection[activePerformanceSection.sectionKey] || []
    setPerformanceParamKey((current) => {
      if (current && summaryRows.some((row) => row.key === current)) {
        return current
      }
      return summaryRows.find((row) => row.hasVariation)?.key || summaryRows[0]?.key || ''
    })
  }, [activePerformanceSection, selectedPerformanceConfig.paramSummaryBySection])

  const filteredDecisions = useMemo(() => {
    const q = decisionSearch.trim().toLowerCase()
    return selectedDecisions
      .filter((decision) => {
        const outcome = normalizeDecisionOutcome(decision.decision)
        if (decisionOutcomeFilter !== 'all' && outcome !== decisionOutcomeFilter) return false
        if (!q) return true
        const marketLabel = resolveDecisionMarketLabel(decision)
        const haystack = `${marketLabel} ${decision.source} ${decision.strategy_key} ${decision.reason || ''} ${decision.decision}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 200)
  }, [selectedDecisions, decisionSearch, decisionOutcomeFilter])

  useEffect(() => {
    if (filteredDecisions.length === 0) {
      setSelectedDecisionId(null)
      return
    }
    setSelectedDecisionId((current) => {
      if (current && filteredDecisions.some((decision) => decision.id === current)) {
        return current
      }
      return filteredDecisions[0].id
    })
  }, [filteredDecisions])

  const filteredTradeHistoryFull = useMemo(() => {
    const q = tradeSearch.trim().toLowerCase()
    return selectedOrders
      .filter((order) => {
        const status = normalizeStatus(order.status)
        const matchesStatus =
          tradeStatusFilter === 'all' ||
          (tradeStatusFilter === 'open_resolved' && (OPEN_ORDER_STATUSES.has(status) || RESOLVED_ORDER_STATUSES.has(status))) ||
          (tradeStatusFilter === 'open' && OPEN_ORDER_STATUSES.has(status)) ||
          (tradeStatusFilter === 'resolved' && RESOLVED_ORDER_STATUSES.has(status)) ||
          (tradeStatusFilter === 'failed' && FAILED_ORDER_STATUSES.has(status))

        if (!matchesStatus) return false
        if (!q) return true

        const haystack = `${order.market_question || ''} ${order.market_id || ''} ${order.source || ''} ${order.direction || ''} ${order.direction_label || ''} ${order.direction_side || ''} ${orderExecutionTypeSummary(order)}`.toLowerCase()
        return haystack.includes(q)
      })
  }, [selectedOrders, tradeSearch, tradeStatusFilter])

  const filteredTradeHistory = useMemo(
    () => filteredTradeHistoryFull.slice(0, 250),
    [filteredTradeHistoryFull]
  )

  const selectedTradeRows = useMemo(() => {
    return filteredTradeHistory.map((order) => {
      const status = normalizeStatus(order.status)
      const lifecycleLabel = resolveOrderLifecycleLabel(status)
      const pnl = toNumber(order.actual_profit)
      const directionPresentation = resolveOrderDirectionPresentation(order)
      const orderPayload = order.payload && typeof order.payload === 'object' ? order.payload : {}
      const providerReconciliation = orderPayload.provider_reconciliation && typeof orderPayload.provider_reconciliation === 'object'
        ? orderPayload.provider_reconciliation
        : {}
      const providerSnapshot = providerReconciliation.snapshot && typeof providerReconciliation.snapshot === 'object'
        ? providerReconciliation.snapshot
        : {}
      const positionState = orderPayload.position_state && typeof orderPayload.position_state === 'object'
        ? orderPayload.position_state
        : {}
      const pendingExit = orderPayload.pending_live_exit && typeof orderPayload.pending_live_exit === 'object'
        ? orderPayload.pending_live_exit
        : {}
      const positionClose = orderPayload.position_close && typeof orderPayload.position_close === 'object'
        ? orderPayload.position_close
        : {}
      const pendingExitStatus = normalizeStatus(String(pendingExit.status || ''))
      const closeTrigger = cleanText(
        order.close_trigger
        || positionClose.close_trigger
        || pendingExit.close_trigger
      )
      const fillPx = toNumber(
        order.average_fill_price
        ?? providerReconciliation.average_fill_price
        ?? providerSnapshot.average_fill_price
        ?? order.effective_price
        ?? order.entry_price
      )
      const realtimeCrypto = resolveOrderRealtimeCryptoSnapshot(order, directionPresentation.side)
      const markPx = toNumber(
        realtimeCrypto.markPrice
        ?? order.current_price
        ?? positionState.last_mark_price
        ?? orderPayload.market_price
        ?? orderPayload.resolved_price
      )
      const filledSize = toNumber(
        order.filled_shares
        ?? providerReconciliation.filled_size
        ?? providerSnapshot.filled_size
        ?? orderPayload.filled_size
      )
      const filledNotional = toNumber(
        order.filled_notional_usd
        ?? providerReconciliation.filled_notional_usd
        ?? providerSnapshot.filled_notional_usd
        ?? order.notional_usd
      )
      let unrealized = toNumber(order.unrealized_pnl)
      if ((order.unrealized_pnl === null || order.unrealized_pnl === undefined) && markPx > 0 && filledSize > 0 && filledNotional > 0) {
        unrealized = (markPx * filledSize) - filledNotional
      }
      const fillProgressPercent = computeOrderFillProgressPercent(
        orderPayload as Record<string, unknown>,
        {
          filledSize,
          filledNotional,
          requestedNotionalFallback: Math.abs(toNumber(order.notional_usd)),
        }
      )
      const dynamicEdgePercent = computeOrderDynamicEdgePercent({
        status,
        edgePercent: toNumber(order.edge_percent),
        unrealizedPnl: unrealized,
        realizedPnl: pnl,
        filledNotional,
      })
      const exitProgressPercent = computePendingExitProgressPercent(pendingExit as Record<string, unknown>)
      const markUpdatedAt = latestTimestampValue(
        realtimeCrypto.updatedAt,
        resolveOrderMarketUpdateTimestamp(order, orderPayload),
      )
      const exitEvaluatedAt = resolveOrderExitEvaluationTimestamp(order, orderPayload)
      const markUpdatedAtRaw = markUpdatedAt
      const markUpdatedTs = toTs(markUpdatedAtRaw)
      const markFresh = markUpdatedTs > 0 && (Date.now() - markUpdatedTs) <= 15_000
      const providerSnapshotStatus = normalizeStatus(
        String(
          order.provider_snapshot_status
          || providerReconciliation.snapshot_status
          || providerSnapshot.normalized_status
          || providerSnapshot.status
          || ''
        )
      )
      const linkedDecisionId = String(order.decision_id || '').trim()
      const signalPayload = linkedDecisionId
        ? decisionSignalPayloadByDecisionId.get(linkedDecisionId) || null
        : null
      const links = buildOrderMarketLinks(order, orderPayload, signalPayload)
      const outcome = orderOutcomeSummary(order)
      const executionSummary = orderExecutionTypeSummary(order)
      const venuePresentation = resolveVenueStatusPresentation(providerSnapshotStatus)
      return {
        order,
        status,
        lifecycleLabel,
        pnl,
        fillPx,
        markPx,
        filledSize,
        filledNotional,
        unrealized,
        fillProgressPercent,
        dynamicEdgePercent,
        exitProgressPercent,
        markUpdatedAt,
        exitEvaluatedAt,
        providerSnapshotStatus,
        pendingExitStatus,
        closeTrigger,
        pendingExit,
        markFresh,
        links,
        directionSide: directionPresentation.side,
        directionLabel: directionPresentation.label,
        yesLabel: directionPresentation.yesLabel,
        noLabel: directionPresentation.noLabel,
        executionSummary,
        outcomeHeadline: outcome.headline,
        outcomeDetail: outcome.detail,
        venuePresentation,
      }
    })
  }, [cryptoMarkets, decisionSignalPayloadByDecisionId, filteredTradeHistory])

  const selectedTradeTotals = useMemo(() => {
    let total = 0
    let open = 0
    let resolved = 0
    let wins = 0
    let losses = 0
    let failed = 0
    let totalNotional = 0
    let realizedPnl = 0
    let unrealizedPnl = 0
    for (const order of filteredTradeHistoryFull) {
      const status = normalizeStatus(order.status)
      const pnl = toNumber(order.actual_profit)
      total += 1
      totalNotional += Math.abs(toNumber(order.notional_usd))
      if (OPEN_ORDER_STATUSES.has(status)) {
        open += 1
        unrealizedPnl += toNumber(order.unrealized_pnl)
      }
      if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolved += 1
        realizedPnl += pnl
        if (pnl > 0) wins += 1
        if (pnl < 0) losses += 1
      }
      if (FAILED_ORDER_STATUSES.has(status)) {
        failed += 1
      }
    }
    return {
      total,
      open,
      resolved,
      wins,
      losses,
      failed,
      totalNotional,
      realizedPnl,
      unrealizedPnl,
      winRate: resolved > 0 ? (wins / resolved) * 100 : 0,
    }
  }, [filteredTradeHistoryFull])

  const filteredAllTradeHistory = useMemo(() => {
    const q = allBotsTradeSearch.trim().toLowerCase()
    return allOrders
      .filter((order) => {
        const status = normalizeStatus(order.status)
        const matchesStatus =
          allBotsTradeStatusFilter === 'all' ||
          (allBotsTradeStatusFilter === 'open_resolved' && (OPEN_ORDER_STATUSES.has(status) || RESOLVED_ORDER_STATUSES.has(status))) ||
          (allBotsTradeStatusFilter === 'open' && OPEN_ORDER_STATUSES.has(status)) ||
          (allBotsTradeStatusFilter === 'resolved' && RESOLVED_ORDER_STATUSES.has(status)) ||
          (allBotsTradeStatusFilter === 'failed' && FAILED_ORDER_STATUSES.has(status))

        if (!matchesStatus) return false
        if (!q) return true

        const haystack = `${order.market_question || ''} ${order.market_id || ''} ${order.source || ''} ${order.direction || ''} ${order.direction_label || ''} ${order.direction_side || ''} ${traderNameById[String(order.trader_id || '')] || ''} ${orderExecutionTypeSummary(order)}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 300)
  }, [allBotsTradeSearch, allBotsTradeStatusFilter, allOrders, traderNameById])

  const filteredAllPositionBook = useMemo(() => {
    const query = allBotsPositionSearch.trim().toLowerCase()
    const rows = globalPositionBook.filter((row) => {
      if (allBotsPositionDirectionFilter === 'yes' && !isYesDirection(row.directionSide || row.direction)) return false
      if (allBotsPositionDirectionFilter === 'no' && !isNoDirection(row.directionSide || row.direction)) return false
      if (!query) return true
      const haystack = `${row.traderName} ${row.marketQuestion} ${row.marketId} ${row.sourceSummary} ${row.statusSummary} ${row.direction} ${row.directionSide || ''} ${row.executionSummary}`.toLowerCase()
      return haystack.includes(query)
    })
    return sortPositionRows(rows, allBotsPositionSortField, allBotsPositionSortDirection)
  }, [
    allBotsPositionDirectionFilter,
    allBotsPositionSearch,
    allBotsPositionSortDirection,
    allBotsPositionSortField,
    globalPositionBook,
  ])

  const allBotsPositionSummary = useMemo(
    () => summarizePositionRows(filteredAllPositionBook),
    [filteredAllPositionBook]
  )

  const activityRows = useMemo(() => {
    const decisionsById = new Map(allDecisions.map((decision) => [decision.id, decision]))
    const latestOrderByDecisionId = new Map<string, TraderOrder>()
    const decisionEchoFingerprints = new Set<string>()
    const decisionReasonById = new Map<string, string>()
    for (const order of allOrders) {
      const decisionId = cleanText(order.decision_id)
      if (!decisionId || latestOrderByDecisionId.has(decisionId)) continue
      latestOrderByDecisionId.set(decisionId, order)
    }

    const decisionRows: ActivityRow[] = allDecisions.map((decision) => {
      const decisionKey = String(decision.decision || '').trim().toLowerCase()
      const legs = collectDecisionLegs(decision)
      const fallbackMarket = cleanText(decision.market_question) || cleanText(decision.market_id)
      const marketLabel = primaryMarketLabel(legs, fallbackMarket)
      const reason = decisionReasonDetail(decision)
      const failedChecksDetail = decisionFailedChecksDetail(decision)
      const action = legs.find((leg) => leg.action)?.action || normalizeTradeAction(decision.direction)
      const tone: ActivityRow['tone'] =
        decisionKey === 'selected' ? 'positive' :
        decisionKey === 'failed' || decisionKey === 'blocked' ? 'negative' :
        decisionKey === 'skipped' ? 'warning' :
        'neutral'
      decisionReasonById.set(decision.id, reason)
      decisionEchoFingerprints.add(
        activityDuplicateFingerprint(
          cleanText(decision.trader_id),
          decision.created_at,
          reason,
          fallbackMarket || marketLabel,
        )
      )

      return {
        kind: 'decision',
        id: decision.id,
        ts: decision.created_at,
        traderId: decision.trader_id,
        title: `${String(decision.decision).toUpperCase()} • ${String(decision.source || 'unknown').toUpperCase()} • ${marketLabel}`,
        detail: [
          `Markets: ${renderMarketsDetail(legs, fallbackMarket)}`,
          `Reason: ${reason}`,
          failedChecksDetail ? `Failed checks: ${failedChecksDetail}` : null,
        ]
          .filter(Boolean)
          .join(' • '),
        action,
        tone,
      }
    })

    const orderRows: ActivityRow[] = allOrders.map((order) => {
      const status = normalizeStatus(order.status)
      const pnl = toNumber(order.actual_profit)
      const tone: ActivityRow['tone'] =
        FAILED_ORDER_STATUSES.has(status) ? 'negative' :
        RESOLVED_ORDER_STATUSES.has(status) && pnl > 0 ? 'positive' :
        RESOLVED_ORDER_STATUSES.has(status) && pnl < 0 ? 'negative' : 'neutral'

      const leg = collectOrderLeg(order)
      const fallbackMarket = cleanText(order.market_question) || cleanText(order.market_id)
      const marketLabel = primaryMarketLabel([leg], fallbackMarket)
      const reason = orderReasonDetail(order)
      const outcome = orderOutcomeSummary(order)
      const detailParts = [
        `Markets: ${renderMarketsDetail([leg], fallbackMarket)}`,
        `Notional: ${formatCurrency(toNumber(order.notional_usd))}`,
        `Mode: ${String(order.mode || '').toUpperCase() || 'N/A'}`,
        `Outcome: ${outcome.headline}`,
        `Reason: ${reason}`,
      ]
      if (RESOLVED_ORDER_STATUSES.has(status)) {
        detailParts.push(`P&L: ${formatCurrency(pnl)}`)
      }

      return {
        kind: 'order',
        id: order.id,
        ts: order.created_at,
        traderId: order.trader_id,
        title: `${status.toUpperCase()}${leg.action ? ` • ${leg.action}` : ''} • ${marketLabel}`,
        detail: detailParts.join(' • '),
        action: leg.action,
        tone,
      }
    })

    const eventRows: ActivityRow[] = []
    for (const event of allEvents) {
      const payload = isRecord(event.payload) ? event.payload : null
      const linkedDecisionId = payload ? cleanText(payload.decision_id) : null
      const linkedDecision = linkedDecisionId ? decisionsById.get(linkedDecisionId) || null : null
      const linkedOrder = linkedDecisionId ? latestOrderByDecisionId.get(linkedDecisionId) || null : null
      const linkedOrderLeg = linkedOrder ? collectOrderLeg(linkedOrder) : null
      const linkedLegs = linkedDecision
        ? collectDecisionLegs(linkedDecision)
        : linkedOrderLeg
          ? [linkedOrderLeg]
          : []
      const payloadMarket = payload ? (cleanText(payload.market_question) || cleanText(payload.market_id)) : null
      const fallbackMarket = payloadMarket
        || (linkedDecision ? cleanText(linkedDecision.market_question) || cleanText(linkedDecision.market_id) : null)
        || (linkedOrder ? cleanText(linkedOrder.market_question) || cleanText(linkedOrder.market_id) : null)
      const marketLabel = primaryMarketLabel(linkedLegs, fallbackMarket)
      const action = (
        normalizeTradeAction(payload ? (payload.action ?? payload.side ?? payload.direction) : null)
        || (linkedOrderLeg ? linkedOrderLeg.action : null)
        || (linkedDecision ? normalizeTradeAction(linkedDecision.direction) : null)
      )
      const reason = eventReasonDetail(event)
      const latencyDetail = eventLatencyDetail(event)
      const severity = String(event.severity || '').trim().toLowerCase()
      const eventType = String(event.event_type || '').trim().toLowerCase()
      const linkedDecisionReason = linkedDecision ? decisionReasonDetail(linkedDecision) : ''
      const resolvedReason =
        eventType === 'decision' && isGenericDecisionReason(reason) && linkedDecisionReason
          ? linkedDecisionReason
          : reason
      const decisionFingerprint = activityDuplicateFingerprint(
        cleanText(event.trader_id),
        event.created_at,
        resolvedReason,
        fallbackMarket || marketLabel,
      )
      const linkedDecisionRowReason = linkedDecisionId ? decisionReasonById.get(linkedDecisionId) || '' : ''
      if (
        eventType === 'decision'
        && (
          (linkedDecision && areReasonsEquivalent(resolvedReason, linkedDecisionRowReason))
          || decisionEchoFingerprints.has(decisionFingerprint)
        )
      ) {
        continue
      }
      const tone: ActivityRow['tone'] =
        severity === 'warn' || severity === 'warning' ? 'warning' :
        severity === 'error' || severity === 'failed' ? 'negative' :
        'neutral'

      eventRows.push({
        kind: 'event',
        id: event.id,
        ts: event.created_at,
        traderId: event.trader_id,
        title: `${String(event.event_type || 'event').toUpperCase()} • ${String(event.severity || 'info').toUpperCase()} • ${marketLabel}`,
        detail: `Markets: ${renderMarketsDetail(linkedLegs, fallbackMarket)} :: Reason: ${resolvedReason}${latencyDetail ? ` :: Latency: ${latencyDetail}` : ''}`,
        action,
        tone,
      })
    }

    return [...decisionRows, ...orderRows, ...eventRows]
      .sort((a, b) => toTs(b.ts) - toTs(a.ts))
      .slice(0, TERMINAL_ACTIVITY_MAX_ROWS)
  }, [allDecisions, allOrders, allEvents])

  const selectedTraderActivityRows = useMemo(
    () => activityRows.filter((row) => row.traderId === selectedTraderId).slice(0, TERMINAL_SELECTED_MAX_ROWS),
    [activityRows, selectedTraderId]
  )

  const filteredTraderActivityRows = useMemo(() => {
    return selectedTraderActivityRows
      .filter((row) => traderFeedFilter === 'all' || row.kind === traderFeedFilter)
  }, [selectedTraderActivityRows, traderFeedFilter])

  useEffect(() => {
    setTerminalScrollTop(0)
    if (terminalViewportRef.current) {
      terminalViewportRef.current.scrollTop = 0
    }
  }, [selectedTraderId, traderFeedFilter, terminalDensity])

  useEffect(() => {
    const viewport = terminalViewportRef.current
    if (!viewport) return
    const sync = () => setTerminalViewportHeight(viewport.clientHeight)
    sync()
    const observer = new ResizeObserver(sync)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [terminalDensity, workTab])

  const compactTerminalWindow = useMemo(() => {
    const total = filteredTraderActivityRows.length
    if (terminalDensity !== 'compact') {
      return {
        rows: filteredTraderActivityRows,
        topPad: 0,
        bottomPad: 0,
        total,
      }
    }
    const visibleRows = Math.max(
      1,
      Math.ceil((terminalViewportHeight || TERMINAL_COMPACT_ROW_HEIGHT) / TERMINAL_COMPACT_ROW_HEIGHT)
    )
    const start = Math.max(0, Math.floor(terminalScrollTop / TERMINAL_COMPACT_ROW_HEIGHT) - TERMINAL_COMPACT_OVERSCAN)
    const end = Math.min(total, start + visibleRows + TERMINAL_COMPACT_OVERSCAN * 2)
    return {
      rows: filteredTraderActivityRows.slice(start, end),
      topPad: start * TERMINAL_COMPACT_ROW_HEIGHT,
      bottomPad: Math.max(0, (total - end) * TERMINAL_COMPACT_ROW_HEIGHT),
      total,
    }
  }, [filteredTraderActivityRows, terminalDensity, terminalScrollTop, terminalViewportHeight])

  const riskActivityRows = useMemo(
    () => activityRows.filter((row) => row.tone === 'negative' || row.tone === 'warning').slice(0, 240),
    [activityRows]
  )

  const recentSelectedDecisions = useMemo(
    () => allDecisions.filter((decision) => String(decision.decision).toLowerCase() === 'selected').slice(0, 24),
    [allDecisions]
  )

  const selectedDecisionCountAllBots = useMemo(
    () => allDecisions.filter((decision) => String(decision.decision).toLowerCase() === 'selected').length,
    [allDecisions]
  )

  const allBotsActivityRows = useMemo(
    () => activityRows.slice(0, TERMINAL_ALL_BOTS_MAX_ROWS),
    [activityRows]
  )

  const traderPerformanceById = useMemo(
    () => new Map(globalSummary.traderRows.map((row) => [row.traderId, row])),
    [globalSummary.traderRows]
  )

  const sourceLabelByKey = useMemo(() => {
    const labels: Record<string, string> = {}
    for (const source of sourceCards) {
      const key = normalizeSourceKey(source.key)
      if (!key) continue
      labels[key] = String(source.label || source.key).trim() || source.key.toUpperCase()
    }
    return labels
  }, [sourceCards])

  const sourceGroupOrderByKey = useMemo(() => {
    const order = new Map<string, number>()
    sourceCards.forEach((source, index) => {
      const key = normalizeSourceKey(source.key)
      if (!key) return
      order.set(key, index)
    })
    return order
  }, [sourceCards])

  const botRosterRows = useMemo(() => {
    return traders.map((trader) => {
      const status = resolveTraderStatusPresentation(trader, orchestratorRunning)
      const sourceKeys = traderSourceKeys(trader)
      const sourceLabels = sourceKeys.map((sourceKey) => sourceLabelByKey[sourceKey] || sourceKey.toUpperCase())
      const performance = traderPerformanceById.get(trader.id)
      const latestActivityTs = Math.max(
        toTs(trader.last_run_at),
        toNumber(performance?.latest_activity_ts)
      )
      return {
        trader,
        status,
        sourceKeys,
        sourceLabels,
        primarySourceKey: sourceKeys.length === 1 ? sourceKeys[0] : sourceKeys.length > 1 ? 'multi' : 'unknown',
        open: toNumber(performance?.open),
        resolved: toNumber(performance?.resolved),
        pnl: toNumber(performance?.pnl),
        latestActivityTs,
        isInactive: !isTraderActive(trader),
      }
    })
  }, [orchestratorRunning, sourceLabelByKey, traderPerformanceById, traders])

  const filteredBotRosterRows = useMemo(() => {
    const query = botRosterSearch.trim().toLowerCase()
    const rows = botRosterRows.filter((row) => {
      if (botRosterHideInactive && row.isInactive) return false
      if (!query) return true
      const haystack = `${row.trader.name} ${row.trader.id} ${row.status.label} ${row.sourceLabels.join(' ')}`.toLowerCase()
      return haystack.includes(query)
    })
    rows.sort((left, right) => {
      if (botRosterSort === 'name_asc') {
        return left.trader.name.localeCompare(right.trader.name)
      }
      if (botRosterSort === 'name_desc') {
        return right.trader.name.localeCompare(left.trader.name)
      }
      if (botRosterSort === 'pnl_desc') {
        if (left.pnl !== right.pnl) return right.pnl - left.pnl
      }
      if (botRosterSort === 'pnl_asc') {
        if (left.pnl !== right.pnl) return left.pnl - right.pnl
      }
      if (botRosterSort === 'open_desc') {
        if (left.open !== right.open) return right.open - left.open
      }
      if (botRosterSort === 'activity_desc') {
        if (left.latestActivityTs !== right.latestActivityTs) return right.latestActivityTs - left.latestActivityTs
      }
      if (left.pnl !== right.pnl) return right.pnl - left.pnl
      return left.trader.name.localeCompare(right.trader.name)
    })
    return rows
  }, [botRosterHideInactive, botRosterRows, botRosterSearch, botRosterSort])

  const groupedBotRosterRows = useMemo(() => {
    const groups = new Map<string, { key: string; label: string; order: number; rows: typeof filteredBotRosterRows }>()

    if (botRosterGroupBy === 'none') {
      return [{
        key: 'all',
        label: 'All Bots',
        order: 0,
        rows: filteredBotRosterRows,
      }]
    }

    for (const row of filteredBotRosterRows) {
      let key = ''
      let label = ''
      let order = 99
      if (botRosterGroupBy === 'status') {
        key = row.status.key
        if (row.status.key === 'running') {
          label = 'Running'
          order = 0
        } else if (row.status.key === 'engine_stopped') {
          label = 'Engine Stopped'
          order = 1
        } else if (row.status.key === 'bot_stopped') {
          label = 'Bot Stopped'
          order = 2
        } else {
          label = 'Inactive'
          order = 3
        }
      } else {
        key = row.primarySourceKey
        if (key === 'multi') {
          label = 'Multi-source'
          order = sourceGroupOrderByKey.size + 1
        } else if (key === 'unknown') {
          label = 'Unassigned'
          order = sourceGroupOrderByKey.size + 2
        } else {
          label = sourceLabelByKey[key] || key.toUpperCase()
          order = sourceGroupOrderByKey.get(key) ?? sourceGroupOrderByKey.size
        }
      }

      if (!groups.has(key)) {
        groups.set(key, { key, label, order, rows: [] })
      }
      const group = groups.get(key)
      if (!group) continue
      group.rows.push(row)
    }

    return Array.from(groups.values()).sort((a, b) => {
      if (a.order !== b.order) return a.order - b.order
      return a.label.localeCompare(b.label)
    })
  }, [botRosterGroupBy, filteredBotRosterRows, sourceGroupOrderByKey, sourceLabelByKey])
  const showCreatingTraderSkeleton = Boolean(
    creatingTraderPreview
    && createTraderMutation.isPending
    && creatingTraderPreview.mode === selectedAccountMode
  )

  const allBotsLeaderboardRows = useMemo(() => {
    return traders
      .map((trader) => {
        const row = traderPerformanceById.get(trader.id)
        const resolved = toNumber(row?.resolved)
        const wins = toNumber(row?.wins)
        return {
          trader,
          orders: toNumber(row?.orders),
          open: toNumber(row?.open),
          resolved,
          pnl: toNumber(row?.pnl),
          notional: toNumber(row?.notional),
          wins,
          losses: toNumber(row?.losses),
          winRate: resolved > 0 ? (wins / resolved) * 100 : 0,
        }
      })
      .sort((a, b) => {
        if (a.pnl !== b.pnl) return b.pnl - a.pnl
        if (a.open !== b.open) return b.open - a.open
        return a.trader.name.localeCompare(b.trader.name)
      })
      .slice(0, 12)
  }, [traderPerformanceById, traders])

  const allBotsOverviewBuckets = useMemo(() => {
    const dayWindow = 14
    const todayUtc = new Date()
    todayUtc.setUTCHours(0, 0, 0, 0)
    const rows: OverviewTrendBucket[] = []
    for (let offset = dayWindow - 1; offset >= 0; offset -= 1) {
      const day = new Date(todayUtc)
      day.setUTCDate(todayUtc.getUTCDate() - offset)
      const key = day.toISOString().slice(0, 10)
      rows.push({
        key,
        label: formatDayKeyLabel(key),
        orders: 0,
        selected: 0,
        resolvedPnl: 0,
        failed: 0,
        warnings: 0,
        cumulativeResolvedPnl: 0,
      })
    }

    const bucketIndexByKey = new Map(rows.map((row, index) => [row.key, index]))

    for (const order of allOrders) {
      const ts = Math.max(toTs(order.executed_at), toTs(order.updated_at), toTs(order.created_at))
      const dayKey = utcDayKeyFromTs(ts)
      if (!dayKey) continue
      const bucketIndex = bucketIndexByKey.get(dayKey)
      if (bucketIndex === undefined) continue
      const bucket = rows[bucketIndex]
      const status = normalizeStatus(order.status)

      bucket.orders += 1
      if (RESOLVED_ORDER_STATUSES.has(status)) {
        bucket.resolvedPnl += toNumber(order.actual_profit)
      }
      if (FAILED_ORDER_STATUSES.has(status)) {
        bucket.failed += 1
        bucket.warnings += 1
      }
    }

    for (const decision of allDecisions) {
      const ts = toTs(decision.created_at)
      const dayKey = utcDayKeyFromTs(ts)
      if (!dayKey) continue
      const bucketIndex = bucketIndexByKey.get(dayKey)
      if (bucketIndex === undefined) continue
      const bucket = rows[bucketIndex]
      const outcome = String(decision.decision || '').trim().toLowerCase()
      if (outcome === 'selected') {
        bucket.selected += 1
      } else if (outcome === 'blocked' || outcome === 'failed') {
        bucket.warnings += 1
      }
    }

    for (const event of allEvents) {
      const ts = toTs(event.created_at)
      const dayKey = utcDayKeyFromTs(ts)
      if (!dayKey) continue
      const bucketIndex = bucketIndexByKey.get(dayKey)
      if (bucketIndex === undefined) continue
      const bucket = rows[bucketIndex]
      const severity = String(event.severity || '').trim().toLowerCase()
      if (severity === 'warn' || severity === 'warning' || severity === 'error' || severity === 'failed') {
        bucket.warnings += 1
      }
    }

    let cumulativeResolvedPnl = 0
    for (const row of rows) {
      cumulativeResolvedPnl += row.resolvedPnl
      row.cumulativeResolvedPnl = cumulativeResolvedPnl
    }

    return rows
  }, [allDecisions, allEvents, allOrders])

  const allBotsSourceMixChart = useMemo(() => {
    const palette = ['#22c55e', '#38bdf8', '#f59e0b', '#a78bfa', '#f97316', '#14b8a6']
    const topRows = globalSummary.sourceRows.slice(0, 5)
    if (globalSummary.sourceRows.length > 5) {
      const remainder = globalSummary.sourceRows.slice(5).reduce(
        (accumulator, row) => ({
          source: 'OTHER',
          orders: accumulator.orders + row.orders,
          resolved: accumulator.resolved + row.resolved,
          pnl: accumulator.pnl + row.pnl,
          notional: accumulator.notional + row.notional,
          wins: accumulator.wins + row.wins,
          losses: accumulator.losses + row.losses,
        }),
        {
          source: 'OTHER',
          orders: 0,
          resolved: 0,
          pnl: 0,
          notional: 0,
          wins: 0,
          losses: 0,
        }
      )
      if (remainder.orders > 0) topRows.push(remainder)
    }
    const totalOrders = topRows.reduce((sum, row) => sum + row.orders, 0)
    const totalPnl = topRows.reduce((sum, row) => sum + row.pnl, 0)

    let cursor = 0
    const slices = topRows.map((row, index) => {
      const percent = totalOrders > 0 ? (row.orders / totalOrders) * 100 : 0
      const span = (percent / 100) * 360
      const start = cursor
      cursor += span
      const end = cursor
      const color = palette[index % palette.length]
      return {
        key: String(row.source || 'unknown').toLowerCase(),
        label: String(row.source || 'UNKNOWN').toUpperCase(),
        orders: row.orders,
        pnl: row.pnl,
        percent,
        color,
        segment: `${color} ${start.toFixed(2)}deg ${end.toFixed(2)}deg`,
      }
    })

    return {
      totalOrders,
      totalPnl,
      slices,
      gradient: slices.length > 0
        ? `conic-gradient(${slices.map((slice) => slice.segment).join(', ')})`
        : 'conic-gradient(#334155 0deg 360deg)',
    }
  }, [globalSummary.sourceRows])

  const allBotsLifecycleMixChart = useMemo(() => {
    const rows = [
      { key: 'open', label: 'Open', value: globalSummary.open, color: '#38bdf8' },
      { key: 'resolved', label: 'Resolved', value: globalSummary.resolved, color: '#22c55e' },
      { key: 'failed', label: 'Failed', value: globalSummary.failed, color: '#ef4444' },
    ]
    const total = rows.reduce((sum, row) => sum + row.value, 0)
    let cursor = 0
    const slices = rows.map((row) => {
      const percent = total > 0 ? (row.value / total) * 100 : 0
      const span = (percent / 100) * 360
      const start = cursor
      cursor += span
      const end = cursor
      return {
        ...row,
        percent,
        segment: `${row.color} ${start.toFixed(2)}deg ${end.toFixed(2)}deg`,
      }
    })

    return {
      total,
      slices,
      gradient: total > 0
        ? `conic-gradient(${slices.map((slice) => slice.segment).join(', ')})`
        : 'conic-gradient(#334155 0deg 360deg)',
    }
  }, [globalSummary.failed, globalSummary.open, globalSummary.resolved])

  const allBotsLeaderboardWithTrend = useMemo(() => {
    const bucketKeys = allBotsOverviewBuckets.map((bucket) => bucket.key)
    const bucketIndexByKey = new Map(bucketKeys.map((key, index) => [key, index]))
    const trendByTraderId = new Map<string, number[]>(
      allBotsLeaderboardRows.map((row) => [row.trader.id, bucketKeys.map(() => 0)])
    )

    for (const order of allOrders) {
      const traderId = String(order.trader_id || '')
      const trend = trendByTraderId.get(traderId)
      if (!trend) continue
      if (!RESOLVED_ORDER_STATUSES.has(normalizeStatus(order.status))) continue
      const ts = Math.max(toTs(order.executed_at), toTs(order.updated_at), toTs(order.created_at))
      const dayKey = utcDayKeyFromTs(ts)
      if (!dayKey) continue
      const bucketIndex = bucketIndexByKey.get(dayKey)
      if (bucketIndex === undefined) continue
      trend[bucketIndex] += toNumber(order.actual_profit)
    }

    let topAbsPnl = 0
    for (const row of allBotsLeaderboardRows) {
      topAbsPnl = Math.max(topAbsPnl, Math.abs(row.pnl))
    }
    const denominator = topAbsPnl > 0 ? topAbsPnl : 1

    return allBotsLeaderboardRows.map((row, index) => {
      const dailyTrend = trendByTraderId.get(row.trader.id) || bucketKeys.map(() => 0)
      const cumulativeTrend: number[] = []
      let running = 0
      for (const value of dailyTrend) {
        running += value
        cumulativeTrend.push(running)
      }
      return {
        ...row,
        rank: index + 1,
        trend: cumulativeTrend,
        pnlBarPercent: Math.max(10, Math.min(100, (Math.abs(row.pnl) / denominator) * 100)),
      }
    })
  }, [allBotsLeaderboardRows, allBotsOverviewBuckets, allOrders])

  const activeTraderCount = useMemo(
    () => traders.filter((trader) => isTraderActive(trader)).length,
    [traders]
  )

  const startedTraderCount = useMemo(
    () => traders.filter((trader) => isTraderExecutionEnabled(trader)).length,
    [traders]
  )
  const activeCryptoTraderCount = useMemo(
    () =>
      traders.filter((trader) => {
        if (!isTraderExecutionEnabled(trader)) return false
        return traderSourceKeys(trader).some((sourceKey) => isCryptoSourceKey(sourceKey))
      }).length,
    [traders]
  )
  const highFrequencyCryptoLoopActive = orchestratorEnabled && !killSwitchOn && activeCryptoTraderCount > 0
  const effectiveGlobalLoopLabel = highFrequencyCryptoLoopActive
    ? 'REAL-TIME (event-driven + high-frequency monitor)'
    : 'REAL-TIME (event-driven)'
  const effectiveGlobalLoopDetail = highFrequencyCryptoLoopActive
    ? 'Signal events wake matching bots immediately, with high-frequency monitoring active for crypto bots.'
    : 'Signal events wake matching bots immediately; fallback intervals apply when no new events arrive.'

  const inactiveTraderCount = useMemo(
    () => Math.max(0, traders.length - activeTraderCount),
    [traders.length, activeTraderCount]
  )

  const runningTraderCount = useMemo(
    () => (orchestratorRunning ? startedTraderCount : 0),
    [orchestratorRunning, startedTraderCount]
  )


  const selectedTraderExposure = useMemo(
    () => selectedPositionBook.reduce((sum, row) => sum + row.exposureUsd, 0),
    [selectedPositionBook]
  )

  const selectedTraderOpenLivePositions = useMemo(
    () => selectedPositionBook.filter((row) => row.liveOrderCount > 0).length,
    [selectedPositionBook]
  )

  const selectedTraderOpenShadowPositions = useMemo(
    () => selectedPositionBook.filter((row) => row.shadowOrderCount > 0).length,
    [selectedPositionBook]
  )

  const selectedTraderOpenLiveOrders = useMemo(
    () => selectedOrders.filter((order) => OPEN_ORDER_STATUSES.has(normalizeStatus(order.status)) && String(order.mode || '').toLowerCase() === 'live').length,
    [selectedOrders]
  )

  const selectedTraderOpenShadowOrders = useMemo(
    () => selectedOrders.filter((order) => OPEN_ORDER_STATUSES.has(normalizeStatus(order.status)) && String(order.mode || '').toLowerCase() === 'shadow').length,
    [selectedOrders]
  )

  const selectedTraderHasAnyDeleteExposure = (
    selectedTraderOpenLivePositions > 0
    || selectedTraderOpenShadowPositions > 0
    || selectedTraderOpenLiveOrders > 0
    || selectedTraderOpenShadowOrders > 0
  )

  const selectedTraderHasLiveDeleteExposure = selectedTraderOpenLivePositions > 0 || selectedTraderOpenLiveOrders > 0

  const selectedTraderDeleteExposureSummary = [
    selectedTraderOpenLivePositions > 0 ? `${selectedTraderOpenLivePositions} live position(s)` : null,
    selectedTraderOpenShadowPositions > 0 ? `${selectedTraderOpenShadowPositions} shadow position(s)` : null,
    selectedTraderOpenLiveOrders > 0 ? `${selectedTraderOpenLiveOrders} live open order(s)` : null,
    selectedTraderOpenShadowOrders > 0 ? `${selectedTraderOpenShadowOrders} shadow open order(s)` : null,
  ]
    .filter(Boolean)
    .join(' • ')

  const selectedDecision = useMemo(
    () => selectedDecisions.find((decision) => decision.id === selectedDecisionId) || null,
    [selectedDecisions, selectedDecisionId]
  )
  const selectedDecisionDirection = useMemo(
    () => (selectedDecision ? resolveDecisionDirectionPresentation(selectedDecision) : { side: null, label: 'N/A' }),
    [selectedDecision]
  )
  const decisionDetailLoading = decisionDetailQuery.isPending || (decisionDetailQuery.isFetching && !decisionDetailQuery.data)
  const decisionChecks = decisionDetailQuery.data?.checks || []
  const decisionOrders = decisionDetailQuery.data?.orders || []
  const decisionOutcomeSummary = useMemo(() => {
    let selected = 0
    let blocked = 0
    let skipped = 0
    for (const decision of selectedDecisions) {
      const outcome = normalizeDecisionOutcome(decision.decision)
      if (outcome === 'selected') selected += 1
      else if (outcome === 'blocked') blocked += 1
      else skipped += 1
    }
    return {
      selected,
      blocked,
      skipped,
    }
  }, [selectedDecisions])
  const decisionPassCount = decisionChecks.filter((check) => check.passed).length
  const decisionFailCount = decisionChecks.length - decisionPassCount
  const riskChecks = Array.isArray(selectedDecision?.risk_snapshot?.checks)
    ? selectedDecision?.risk_snapshot?.checks
    : []
  const riskAllowed = selectedDecision ? toBoolean(selectedDecision.risk_snapshot?.allowed, false) : false
  const latestSelectedTraderActivityTs = selectedTraderActivityRows.length > 0
    ? toTs(selectedTraderActivityRows[0].ts)
    : 0
  const latestSelectedTraderRunTs = toTs(selectedTrader?.last_run_at || worker?.last_run_at)
  const selectedTraderNoNewRows = Boolean(
    selectedTrader &&
    orchestratorRunning &&
    latestSelectedTraderRunTs > (latestSelectedTraderActivityTs + 1000)
  )

  const tradersRunningDisplay = orchestratorRunning ? toNumber(metrics?.traders_running) : 0
  const displayAvgEdge = normalizeEdgePercent(globalSummary.avgEdge)
  const selectedTraderStatus = resolveTraderStatusPresentation(selectedTrader, orchestratorRunning)
  const selectedTraderPendingAction = selectedTrader
    ? traderTogglePendingById[selectedTrader.id] || null
    : null
  const selectedTraderExecutionEnabled = isTraderExecutionEnabled(selectedTrader)
  const selectedTraderIsActive = Boolean(selectedTrader?.is_enabled)
  const selectedTraderIsStopped = Boolean(selectedTrader?.is_paused)
  const selectedTraderCanStart = Boolean(
    selectedTrader
    && selectedTraderIsActive
    && selectedTraderIsStopped
    && selectedTraderPendingAction !== 'start'
  )
  const selectedTraderCanStop = Boolean(
    selectedTrader
    && selectedTraderIsActive
    && !selectedTraderIsStopped
    && selectedTraderPendingAction !== 'stop'
  )
  const selectedTraderCanActivate = Boolean(
    selectedTrader
    && !selectedTraderIsActive
    && selectedTraderPendingAction !== 'activate'
  )
  const selectedTraderCanDeactivate = Boolean(
    selectedTrader
    && selectedTraderIsActive
    && selectedTraderPendingAction !== 'deactivate'
  )
  const selectedTraderControlPending = selectedTraderPendingAction !== null
  const stopLifecycleNeedsLiveConfirm = Boolean(
    selectedTrader
    && stopLifecycleMode === 'close_all_positions'
  )

  const requestStartTrader = () => {
    if (!selectedTrader || selectedTraderControlPending) return
    if (!selectedTraderIsActive) {
      setSaveError('Activate this bot before starting it.')
      return
    }
    if (selectedTraderHasCopySource) {
      setEnableCopyExistingPositions(selectedTraderCopyExistingOnStartDefault)
      setConfirmTraderStartOpen(true)
      return
    }
    traderStartMutation.mutate({ traderId: selectedTrader.id })
  }

  const confirmStartTrader = () => {
    if (!selectedTrader) return
    setConfirmTraderStartOpen(false)
    traderStartMutation.mutate({
      traderId: selectedTrader.id,
      copyExistingPositions: selectedTraderHasCopySource ? enableCopyExistingPositions : undefined,
    })
  }

  const requestStopTrader = () => {
    if (!selectedTrader || selectedTraderControlPending) return
    if (!selectedTraderIsActive) {
      setSaveError('This bot is inactive and already not running.')
      return
    }
    setStopLifecycleMode('keep_positions')
    setStopConfirmLiveClose(false)
    setConfirmTraderStopOpen(true)
  }

  const confirmStopTrader = () => {
    if (!selectedTrader) return
    if (stopLifecycleNeedsLiveConfirm && !stopConfirmLiveClose) {
      setSaveError('Enable "confirm live close" before requesting live position cleanup.')
      return
    }
    setConfirmTraderStopOpen(false)
    traderStopMutation.mutate({
      traderId: selectedTrader.id,
      payload: {
        stop_lifecycle: stopLifecycleMode,
        confirm_live: stopLifecycleNeedsLiveConfirm ? stopConfirmLiveClose : undefined,
      },
    })
  }

  const requestActivateTrader = () => {
    if (!selectedTrader || selectedTraderControlPending) return
    traderActivateMutation.mutate({ traderId: selectedTrader.id })
  }

  const requestDeactivateTrader = () => {
    if (!selectedTrader || selectedTraderControlPending) return
    traderDeactivateMutation.mutate({ traderId: selectedTrader.id })
  }

  useEffect(() => {
    if (!tuneAutoEnabled) return
    if (!selectedTrader || !selectedTraderExecutionEnabled) return
    if (runTuneIterateMutation.isPending) return

    const intervalMinutes = Math.max(1, Math.min(360, Math.trunc(toNumber(tuneAutoIntervalMinutes || 15) || 15)))
    const intervalMs = Math.max(
      60_000,
      Math.min(360 * 60_000, intervalMinutes * 60_000)
    )
    const baseRunAt = tuneAutoLastRunAt ?? Date.now()
    const dueInMs = Math.max(0, (baseRunAt + intervalMs) - Date.now())
    const timeoutId = window.setTimeout(() => {
      if (runTuneIterateMutation.isPending) return
      runTuneIterateMutation.mutate({ trigger: 'auto' })
    }, dueInMs)
    return () => window.clearTimeout(timeoutId)
  }, [
    runTuneIterateMutation,
    selectedTrader,
    selectedTraderExecutionEnabled,
    tuneAutoEnabled,
    tuneAutoIntervalMinutes,
    tuneAutoLastRunAt,
  ])

  const showingAllBotsDashboard = !selectedTraderId
  const overviewStartLabel = allBotsOverviewBuckets[0]?.label || 'n/a'
  const overviewEndLabel = allBotsOverviewBuckets[allBotsOverviewBuckets.length - 1]?.label || 'n/a'
  const overviewLatestBucket = allBotsOverviewBuckets[allBotsOverviewBuckets.length - 1] || null
  const overviewPreviousBucket = allBotsOverviewBuckets[allBotsOverviewBuckets.length - 2] || null
  const overviewPnlSeries = allBotsOverviewBuckets.map((bucket) => bucket.cumulativeResolvedPnl)
  const overviewOrdersSeries = allBotsOverviewBuckets.map((bucket) => bucket.orders)
  const overviewSelectedSeries = allBotsOverviewBuckets.map((bucket) => bucket.selected)
  const overviewRiskSeries = allBotsOverviewBuckets.map((bucket) => bucket.warnings)

  const requestOrchestratorStart = () => {
    if (selectedAccountIsLive) {
      setConfirmLiveStartOpen(true)
      return
    }
    startBySelectedAccountMutation.mutate()
  }

  const confirmLiveStart = () => {
    setConfirmLiveStartOpen(false)
    startBySelectedAccountMutation.mutate()
  }

  const setGlobalSettingsField = <K extends keyof GlobalSettingsDraft,>(
    key: K,
    value: GlobalSettingsDraft[K]
  ) => {
    setGlobalSettingsDraft((current) => ({
      ...current,
      [key]: value,
    }))
  }

  const openGlobalSettingsFlyout = () => {
    setGlobalSettingsDraft(buildGlobalSettingsDraft(orchestratorConfig, liveExecutionSettings))
    setGlobalSettingsSaveError(null)
    setGlobalSettingsFlyoutOpen(true)
  }

  const resetGlobalSettingsDraft = () => {
    setGlobalSettingsDraft(buildGlobalSettingsDraft(orchestratorConfig, liveExecutionSettings))
    setGlobalSettingsSaveError(null)
  }

  const saveGlobalSettings = () => {
    setGlobalSettingsSaveError(null)
    updateGlobalSettingsMutation.mutate()
  }

  const canStartOrchestrator =
    !controlBusy &&
    !orchestratorStartStopActive &&
    Boolean(selectedAccountId) &&
    selectedAccountValid &&
    !(selectedAccountIsLive && killSwitchOn)
  const canStopOrchestrator = !controlBusy && orchestratorStartStopActive
  const startStopIsConfigured = orchestratorStartStopActive
  const startStopIsRunning = orchestratorWorkerRunning
  const startStopIsStarting =
    orchestratorStartRequestPending ||
    (orchestratorEnabled && !startStopIsRunning && workerActivity.includes('start command queued'))
  const startStopIsStopping = orchestratorStopRequestPending
  const startStopPending = startStopIsStarting || startStopIsStopping
  const startStopDisabled = startStopPending || (startStopIsConfigured ? !canStopOrchestrator : !canStartOrchestrator)

  const runStartStopCommand = () => {
    if (startStopIsConfigured) {
      if (!canStopOrchestrator) return
      stopByModeMutation.mutate()
      return
    }
    if (!canStartOrchestrator) return
    requestOrchestratorStart()
  }

  const shellLoading = overviewQuery.isLoading || tradersQuery.isLoading

  if (shellLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 flex items-center justify-center gap-3 text-sm text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading orchestrator control plane...
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-1.5">
      {/* ── Hub Strip ── */}
      <div className="shrink-0 rounded-lg border border-cyan-500/30 bg-card px-3 py-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
        <div className="flex items-center gap-1.5">
          <Button
            onClick={runStartStopCommand}
            disabled={startStopDisabled}
            className="h-7 min-w-[140px] text-[11px]"
            variant={startStopIsConfigured ? 'secondary' : 'default'}
            size="sm"
          >
            {startStopPending ? (
              <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />
            ) : startStopIsConfigured ? (
              <Square className="w-3.5 h-3.5 mr-1" />
            ) : selectedAccountIsLive ? (
              <Zap className="w-3.5 h-3.5 mr-1" />
            ) : (
              <Play className="w-3.5 h-3.5 mr-1" />
            )}
            {startStopIsStarting
              ? 'Starting...'
              : startStopIsStopping
                ? 'Stopping...'
                : startStopIsConfigured
                  ? 'Stop'
                  : selectedAccountMode.toUpperCase()}
          </Button>
          <div className="flex items-center gap-1.5 rounded border border-red-500/30 bg-red-500/5 px-1.5 py-0.5">
            <ShieldAlert className="w-3 h-3 text-red-400" />
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <Switch
                    checked={killSwitchSwitchValue}
                    onCheckedChange={(enabled) => killSwitchMutation.mutate(enabled)}
                    disabled={controlBusy}
                    className="scale-[0.8]"
                  />
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-[320px] text-xs leading-snug">
                Blocks new entry orders only. Bots stay running in manage-only mode so existing positions and orders can
                still be monitored, sold, and reconciled.
              </TooltipContent>
            </Tooltip>
            {killSwitchMutation.isPending ? (
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-red-300">
                <Loader2 className="w-3 h-3 animate-spin" />
                {killSwitchSwitchValue ? 'Blocking...' : 'Opening...'}
              </span>
            ) : null}
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span
            className={cn(
              'w-1.5 h-1.5 rounded-full',
              worker?.last_error
                ? 'bg-amber-400'
                : orchestratorWorkerRunning
                  ? 'bg-emerald-500'
                  : 'bg-amber-400'
            )}
          />
          <Clock3 className="w-3 h-3" />
          {formatTimestamp(worker?.last_run_at)}
        </div>

        <div className="flex items-center gap-1.5">
          <Badge
            className="h-5 px-1.5 text-[10px]"
            variant={orchestratorBlocked ? 'destructive' : orchestratorWorkerRunning ? 'default' : 'secondary'}
          >
            {orchestratorStatusLabel}
          </Badge>
          {orchestratorControlMismatch ? (
            <Badge className="h-5 px-1.5 text-[10px]" variant="destructive">
              DESYNC
            </Badge>
          ) : null}
          <Badge className="h-5 px-1.5 text-[10px]" variant={selectedAccountMode === 'live' ? 'destructive' : 'outline'}>
            {selectedAccountMode.toUpperCase()}
          </Badge>
          <Badge
            className="h-5 px-1.5 text-[10px]"
            variant={killSwitchMutation.isPending ? 'secondary' : killSwitchOn ? 'destructive' : 'outline'}
          >
            {killSwitchStatusLabel}
          </Badge>
        </div>

        <div className="hidden lg:flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
          <span>Bots {tradersRunningDisplay}/{toNumber(metrics?.traders_total)}</span>
          <span className="text-border">|</span>
          <span className={toNumber(metrics?.daily_pnl) >= 0 ? 'text-emerald-500' : 'text-red-500'}>
            {formatCurrency(toNumber(metrics?.daily_pnl))}
          </span>
          <span className="text-border">|</span>
          <span>Exp {formatCurrency(toNumber(metrics?.gross_exposure_usd), true)}</span>
          <span className="text-border">|</span>
          <span>{globalSummary.open} open</span>
          <span className="text-border">|</span>
          <span>WR {formatPercent(globalSummary.winRate)}</span>
          <span className="text-border">|</span>
          <span>Edge {formatPercent(displayAvgEdge)}</span>
        </div>

        <div className="ml-auto flex items-center">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-6 px-2 text-[10px]"
            onClick={openGlobalSettingsFlyout}
            disabled={globalSettingsBusy}
          >
            {globalSettingsBusy ? (
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            ) : (
              <Settings className="w-3 h-3 mr-1" />
            )}
            Settings
          </Button>
        </div>
      </div>
      {controlActionError ? (
        <div className="shrink-0 rounded-md border border-red-500/35 bg-red-500/10 px-2 py-1 text-[11px] text-red-300">
          {controlActionError}
        </div>
      ) : null}

      {/* ── Main: Roster Rail + Work Area ── */}
      <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[240px_minmax(0,1fr)]">
        {/* Left rail — Bot Roster */}
        <div className="hidden xl:flex flex-col min-h-0 rounded-lg border border-border/70 bg-card overflow-hidden">
          <div className="shrink-0 px-2.5 py-2 border-b border-border/50 flex items-center justify-between gap-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Bots</span>
            <Button size="sm" className="h-6 w-6 p-0" variant="outline" onClick={openCreateTraderFlyout} title="New bot">
              <Plus className="w-3.5 h-3.5" />
            </Button>
          </div>
          <div className="shrink-0 px-1.5 py-1.5 border-b border-border/50 space-y-1">
            <Input
              value={botRosterSearch}
              onChange={(event) => setBotRosterSearch(event.target.value)}
              placeholder="Search bots, source, status..."
              className="h-6 text-[10px]"
            />
            <div className="grid grid-cols-2 gap-1">
              <Select value={botRosterSort} onValueChange={(value) => setBotRosterSort(value as BotRosterSort)}>
                <SelectTrigger className="h-6 text-[10px] px-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="name_asc">Name A-Z</SelectItem>
                  <SelectItem value="name_desc">Name Z-A</SelectItem>
                  <SelectItem value="pnl_desc">P&amp;L High-Low</SelectItem>
                  <SelectItem value="pnl_asc">P&amp;L Low-High</SelectItem>
                  <SelectItem value="open_desc">Open Orders</SelectItem>
                  <SelectItem value="activity_desc">Latest Activity</SelectItem>
                </SelectContent>
              </Select>
              <Select value={botRosterGroupBy} onValueChange={(value) => setBotRosterGroupBy(value as BotRosterGroupBy)}>
                <SelectTrigger className="h-6 text-[10px] px-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="status">Group: Status</SelectItem>
                  <SelectItem value="source">Group: Source</SelectItem>
                  <SelectItem value="none">Group: None</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <label className="h-6 rounded border border-border/50 bg-background/50 px-2 flex items-center gap-1.5">
              <Switch
                checked={botRosterHideInactive}
                onCheckedChange={setBotRosterHideInactive}
                className="scale-[0.75]"
              />
              <span className="text-[10px] text-muted-foreground">Hide inactive</span>
              <span className="ml-auto text-[10px] font-mono text-muted-foreground">
                {filteredBotRosterRows.length}/{traders.length}
              </span>
            </label>
          </div>
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-1.5 space-y-1.5">
              {/* "All Bots" aggregate */}
              <button
                type="button"
                onClick={() => setSelectedTraderId(null)}
                className={cn(
                  'w-full text-left rounded-md px-2 py-1.5 text-[11px] transition-colors',
                  !selectedTraderId
                    ? 'bg-cyan-500/15 text-foreground font-medium'
                    : 'text-muted-foreground hover:bg-muted/40'
                )}
              >
                <div className="flex min-w-0 items-center justify-between gap-2">
                  <span className="truncate">All Bots</span>
                  <span className={cn('shrink-0 font-mono text-[10px]', globalSummary.resolvedPnl > 0 ? 'text-emerald-500' : globalSummary.resolvedPnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                    {formatCurrency(globalSummary.resolvedPnl, true)}
                  </span>
                </div>
                <div className="mt-0.5 truncate text-[9px] text-muted-foreground">
                  Showing {filteredBotRosterRows.length}/{traders.length}
                </div>
              </button>
              {showCreatingTraderSkeleton ? (
                <div className="w-full rounded-md border border-cyan-500/35 bg-cyan-500/10 px-2 py-1.5 animate-pulse">
                  <div className="flex items-center justify-between gap-1.5">
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
                        <span className="text-[11px] font-medium truncate leading-tight text-foreground">
                          {creatingTraderPreview?.name || 'Creating bot...'}
                        </span>
                      </div>
                      <div className="pl-3 mt-0.5 text-[9px] text-muted-foreground">
                        Provisioning {String(creatingTraderPreview?.mode || selectedAccountMode).toUpperCase()} bot...
                      </div>
                    </div>
                    <Loader2 className="w-3.5 h-3.5 text-cyan-300 animate-spin shrink-0" />
                  </div>
                </div>
              ) : null}
              {groupedBotRosterRows.length === 0 ? (
                <p className="py-6 text-center text-[11px] text-muted-foreground">No bots match these filters.</p>
              ) : (
                groupedBotRosterRows.map((group) => {
                  const groupPnl = group.rows.reduce((sum, row) => sum + row.pnl, 0)
                  return (
                    <div key={group.key} className="space-y-0.5">
                      {botRosterGroupBy !== 'none' ? (
                        <div className="px-1 py-0.5 flex items-center justify-between">
                          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{group.label}</span>
                          <span className={cn('text-[9px] font-mono', groupPnl > 0 ? 'text-emerald-500' : groupPnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                            {group.rows.length} · {formatCurrency(groupPnl, true)}
                          </span>
                        </div>
                      ) : null}
                      {group.rows.map((row) => {
                        const isActive = row.trader.id === selectedTraderId
                        const pendingToggleAction = traderTogglePendingById[row.trader.id] || null
                        const rowTogglePending = pendingToggleAction !== null
                        return (
                          <button
                            key={row.trader.id}
                            type="button"
                            onClick={() => setSelectedTraderId(row.trader.id)}
                            disabled={rowTogglePending}
                            className={cn(
                              'w-full text-left rounded-md px-2 py-1.5 transition-colors group',
                              isActive
                                ? 'bg-cyan-500/15 text-foreground'
                                : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground',
                              rowTogglePending && 'cursor-wait border border-cyan-500/35 bg-cyan-500/10 text-foreground/85'
                            )}
                          >
                            {rowTogglePending ? (
                              <div className="flex items-center justify-between gap-2">
                                <div className="min-w-0 flex-1 animate-pulse space-y-1">
                                  <div className="flex items-center gap-1.5">
                                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-300/80 shrink-0" />
                                    <div className="h-2.5 w-24 max-w-full rounded bg-muted/70" />
                                  </div>
                                  <div className="pl-3 h-2 w-28 max-w-full rounded bg-muted/60" />
                                  <div className="pl-3 flex gap-1">
                                    <span className="h-2 w-10 rounded bg-muted/60" />
                                    <span className="h-2 w-12 rounded bg-muted/50" />
                                  </div>
                                </div>
                                <div className="shrink-0 flex items-center gap-1 text-[9px] text-cyan-300">
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                  {pendingToggleAction === 'start'
                                    ? 'Starting...'
                                    : pendingToggleAction === 'stop'
                                      ? 'Stopping...'
                                      : pendingToggleAction === 'activate'
                                        ? 'Activating...'
                                        : 'Deactivating...'}
                                </div>
                              </div>
                            ) : (
                              <>
                                <div className="flex items-center justify-between gap-1.5">
                                  <div className="min-w-0">
                                    <div className="flex items-center gap-1.5">
                                      <span className={cn(
                                        'w-1.5 h-1.5 rounded-full shrink-0',
                                        row.status.dotClassName
                                      )} />
                                      <span className="text-[11px] font-medium truncate leading-tight">{row.trader.name}</span>
                                    </div>
                                    <div className="pl-3 mt-0.5 text-[9px] text-muted-foreground">
                                      {row.open} open · {row.resolved} resolved
                                    </div>
                                  </div>
                                  <span className={cn('shrink-0 text-[10px] font-mono', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                                    {formatCurrency(row.pnl, true)}
                                  </span>
                                </div>
                                <div className="flex flex-wrap gap-0.5 mt-0.5 pl-3">
                                  {row.sourceLabels.length > 0 ? row.sourceLabels.map((sourceLabel) => (
                                    <span key={`${row.trader.id}:${sourceLabel}`} className="px-1 py-0 text-[8px] rounded bg-muted/60 text-muted-foreground leading-relaxed">{sourceLabel}</span>
                                  )) : (
                                    <span className="px-1 py-0 text-[8px] rounded bg-muted/60 text-muted-foreground leading-relaxed">Unassigned</span>
                                  )}
                                </div>
                              </>
                            )}
                          </button>
                        )
                      })}
                    </div>
                  )
                })
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Right — Work Area */}
        <div className="flex flex-col min-h-0 min-w-0 gap-1.5">
          {showingAllBotsDashboard ? (
            <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-cyan-500/25 bg-card">
              <div className="h-full min-h-0 flex flex-col">
                <div className="shrink-0 px-2 py-1.5">
                  <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-4">
                    <div className="rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Running Bots</p>
                      <p className="text-sm font-mono">{runningTraderCount}/{activeTraderCount}</p>
                      <p className="text-[10px] text-muted-foreground">Inactive {inactiveTraderCount}</p>
                    </div>
                    <div className="rounded-md border border-cyan-500/25 bg-cyan-500/10 px-2.5 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Selected Signals</p>
                      <p className="text-sm font-mono">{selectedDecisionCountAllBots}</p>
                      <p className="text-[10px] text-muted-foreground">Latest {recentSelectedDecisions.length}</p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Resolved P&amp;L</p>
                      <p className={cn('text-sm font-mono', globalSummary.resolvedPnl >= 0 ? 'text-emerald-500' : 'text-red-500')}>
                        {formatCurrency(globalSummary.resolvedPnl)}
                      </p>
                      <p className="text-[10px] text-muted-foreground">WR {formatPercent(globalSummary.winRate)}</p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Exposure</p>
                      <p className="text-sm font-mono">{formatCurrency(toNumber(metrics?.gross_exposure_usd), true)}</p>
                      <p className="text-[10px] text-muted-foreground">{globalSummary.open} open orders</p>
                    </div>
                  </div>
                </div>

                <Tabs
                  value={allBotsTab}
                  onValueChange={(value) => setAllBotsTab(value as AllBotsTab)}
                  className="flex-1 min-h-0 flex flex-col overflow-hidden px-2 pb-2"
                >
                  <div className="shrink-0">
                    <TabsList className="h-auto justify-start gap-1 rounded-lg border border-border/60 bg-card/70 p-1">
                      <TabsTrigger value="overview" className="h-7 px-2.5 text-[11px]">Overview</TabsTrigger>
                      <TabsTrigger value="trades" className="h-7 px-2.5 text-[11px]">All Trades</TabsTrigger>
                      <TabsTrigger value="positions" className="h-7 px-2.5 text-[11px]">All Positions</TabsTrigger>
                    </TabsList>
                  </div>

                  <TabsContent value="overview" className="mt-2 flex-1 min-h-0 overflow-hidden">
                    <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
                      <div className="min-h-0 flex flex-col gap-2">
                        <div className="grid gap-2 sm:grid-cols-2">
                          <div className="rounded-md border border-emerald-500/25 bg-emerald-500/10 p-2.5">
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5">
                                <TrendingUp className="w-3.5 h-3.5 text-emerald-500" />
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Resolved P&amp;L</span>
                              </div>
                              <span className="text-[9px] font-mono text-muted-foreground">{overviewStartLabel} - {overviewEndLabel}</span>
                            </div>
                            <p className={cn('mt-1 text-sm font-mono', globalSummary.resolvedPnl >= 0 ? 'text-emerald-500' : 'text-red-500')}>
                              {formatCurrency(globalSummary.resolvedPnl)}
                            </p>
                            <p className="text-[10px] text-muted-foreground">
                              Latest day {formatSignedCurrency(overviewLatestBucket?.resolvedPnl ?? 0)}
                            </p>
                            <div className="mt-1.5 h-8">
                              {overviewPnlSeries.length >= 2 && (
                                <Liveline
                                  data={toTimeValueSeries(overviewPnlSeries)}
                                  value={overviewPnlSeries[overviewPnlSeries.length - 1] ?? 0}
                                  color={globalSummary.resolvedPnl >= 0 ? '#22c55e' : '#ef4444'}
                                  theme={themeMode}
                                  window={(overviewPnlSeries.length - 1) * 60}
                                  paused
                                  grid={false}
                                  badge={false}
                                  fill
                                  pulse={false}
                                  momentum={false}
                                  scrub={false}
                                  lerpSpeed={0.2}
                                  padding={{ top: 2, right: 2, bottom: 2, left: 2 }}
                                  style={{ height: 30 }}
                                />
                              )}
                            </div>
                          </div>

                          <div className="rounded-md border border-cyan-500/25 bg-cyan-500/10 p-2.5">
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5">
                                <BarChart3 className="w-3.5 h-3.5 text-cyan-500" />
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Order Throughput</span>
                              </div>
                              <span className="text-[9px] font-mono text-muted-foreground">14d</span>
                            </div>
                            <p className="mt-1 text-sm font-mono">{overviewLatestBucket?.orders ?? 0} orders</p>
                            <p className="text-[10px] text-muted-foreground">
                              Prev {(overviewPreviousBucket?.orders ?? 0)} · Failed {(overviewLatestBucket?.failed ?? 0)}
                            </p>
                            <div className="mt-1.5 h-8">
                              {overviewOrdersSeries.length >= 2 && (
                                <Liveline
                                  data={toTimeValueSeries(overviewOrdersSeries)}
                                  value={overviewOrdersSeries[overviewOrdersSeries.length - 1] ?? 0}
                                  color="#06b6d4"
                                  theme={themeMode}
                                  window={(overviewOrdersSeries.length - 1) * 60}
                                  paused
                                  grid={false}
                                  badge={false}
                                  fill
                                  pulse={false}
                                  momentum={false}
                                  scrub={false}
                                  lerpSpeed={0.2}
                                  padding={{ top: 2, right: 2, bottom: 2, left: 2 }}
                                  style={{ height: 30 }}
                                />
                              )}
                            </div>
                          </div>

                          <div className="rounded-md border border-violet-500/25 bg-violet-500/10 p-2.5">
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5">
                                <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Selected Signals</span>
                              </div>
                              <span className="text-[9px] font-mono text-muted-foreground">{recentSelectedDecisions.length} recent</span>
                            </div>
                            <p className="mt-1 text-sm font-mono">{selectedDecisionCountAllBots}</p>
                            <p className="text-[10px] text-muted-foreground">
                              Latest day {(overviewLatestBucket?.selected ?? 0)} · WR {formatPercent(globalSummary.winRate)}
                            </p>
                            <div className="mt-1.5 h-8">
                              {overviewSelectedSeries.length >= 2 && (
                                <Liveline
                                  data={toTimeValueSeries(overviewSelectedSeries)}
                                  value={overviewSelectedSeries[overviewSelectedSeries.length - 1] ?? 0}
                                  color="#a78bfa"
                                  theme={themeMode}
                                  window={(overviewSelectedSeries.length - 1) * 60}
                                  paused
                                  grid={false}
                                  badge={false}
                                  fill
                                  pulse={false}
                                  momentum={false}
                                  scrub={false}
                                  lerpSpeed={0.2}
                                  padding={{ top: 2, right: 2, bottom: 2, left: 2 }}
                                  style={{ height: 30 }}
                                />
                              )}
                            </div>
                          </div>

                          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5">
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5">
                                <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Risk Pressure</span>
                              </div>
                              <span className="text-[9px] font-mono text-muted-foreground">{riskActivityRows.length} alerts</span>
                            </div>
                            <p className="mt-1 text-sm font-mono">{overviewLatestBucket?.warnings ?? 0} today</p>
                            <p className="text-[10px] text-muted-foreground">
                              Prev {overviewPreviousBucket?.warnings ?? 0}
                            </p>
                            <div className="mt-1.5 h-8">
                              {overviewRiskSeries.length >= 2 && (
                                <Liveline
                                  data={toTimeValueSeries(overviewRiskSeries)}
                                  value={overviewRiskSeries[overviewRiskSeries.length - 1] ?? 0}
                                  color="#f59e0b"
                                  theme={themeMode}
                                  window={(overviewRiskSeries.length - 1) * 60}
                                  paused
                                  grid={false}
                                  badge={false}
                                  fill
                                  pulse={false}
                                  momentum={false}
                                  scrub={false}
                                  lerpSpeed={0.2}
                                  padding={{ top: 2, right: 2, bottom: 2, left: 2 }}
                                  style={{ height: 30 }}
                                />
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="min-h-0 flex-1 rounded-md border border-border/60 bg-card/80 overflow-hidden">
                          <div className="px-2.5 py-2 border-b border-border/40 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <Clock3 className="w-3.5 h-3.5 text-cyan-500" />
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Live Pulse Feed</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground">{allBotsActivityRows.length} events</span>
                          </div>
                          <ScrollArea className="h-[260px] xl:h-full">
                            <div className="space-y-1.5 p-2 text-[11px]">
                              {allBotsActivityRows.length === 0 ? (
                                <p className="py-10 text-center text-muted-foreground text-xs">No activity captured yet.</p>
                              ) : (
                                allBotsActivityRows.slice(0, 40).map((row) => (
                                  <div
                                    key={`${row.kind}:${row.id}`}
                                    className={cn(
                                      'rounded-md border px-2.5 py-2',
                                      row.tone === 'positive' && 'border-emerald-500/25 bg-emerald-500/5',
                                      row.tone === 'negative' && 'border-red-500/30 bg-red-500/5',
                                      row.tone === 'warning' && 'border-amber-500/30 bg-amber-500/5',
                                      row.tone === 'neutral' && 'border-border/50 bg-background/40'
                                    )}
                                  >
                                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] text-muted-foreground">
                                      <span className="font-mono">{formatTimestamp(row.ts)}</span>
                                      <span className="uppercase">{row.kind}</span>
                                      {row.action ? (
                                        <span className={cn('uppercase font-semibold', row.action === 'BUY' ? 'text-emerald-500' : 'text-red-500')}>
                                          {row.action}
                                        </span>
                                      ) : null}
                                      <span>{traderNameById[String(row.traderId || '')] || shortId(row.traderId || '')}</span>
                                    </div>
                                    <p className="mt-0.5 font-medium break-words">{row.title}</p>
                                    <p className="mt-0.5 text-[10px] text-muted-foreground break-words">{row.detail}</p>
                                  </div>
                                ))
                              )}
                            </div>
                          </ScrollArea>
                        </div>
                      </div>

                      <div className="min-h-0 flex flex-col gap-2">
                        <div className="rounded-md border border-border/60 bg-card/80 p-2.5">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <PieChart className="w-3.5 h-3.5 text-cyan-500" />
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Execution Mix</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground">{allBotsSourceMixChart.totalOrders} orders</span>
                          </div>
                          <div className="mt-2 grid gap-2 md:grid-cols-2">
                            <div className="rounded-md border border-border/50 bg-background/40 p-2">
                              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Sources</p>
                              <div className="mt-2 grid grid-cols-[96px_minmax(0,1fr)] gap-2 items-center">
                                <div className="relative h-24 w-24 rounded-full border border-border/50" style={{ background: allBotsSourceMixChart.gradient }}>
                                  <div className="absolute inset-[18px] rounded-full border border-border/60 bg-card" />
                                  <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                                    <span className="text-sm font-mono">{allBotsSourceMixChart.totalOrders}</span>
                                    <span className="text-[9px] uppercase tracking-wider text-muted-foreground">orders</span>
                                  </div>
                                </div>
                                <div className="space-y-1">
                                  {allBotsSourceMixChart.slices.length === 0 ? (
                                    <p className="text-[10px] text-muted-foreground">No source activity yet.</p>
                                  ) : (
                                    allBotsSourceMixChart.slices.map((slice) => (
                                      <div key={slice.key} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1.5 text-[10px]">
                                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: slice.color }} />
                                        <span className="truncate">{slice.label}</span>
                                        <span className="font-mono text-muted-foreground">{slice.percent.toFixed(0)}%</span>
                                      </div>
                                    ))
                                  )}
                                </div>
                              </div>
                              <p className={cn('mt-2 text-[10px] font-mono', allBotsSourceMixChart.totalPnl > 0 ? 'text-emerald-500' : allBotsSourceMixChart.totalPnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                                Mix P&amp;L {formatCurrency(allBotsSourceMixChart.totalPnl)}
                              </p>
                            </div>

                            <div className="rounded-md border border-border/50 bg-background/40 p-2">
                              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Lifecycle</p>
                              <div className="mt-2 grid grid-cols-[96px_minmax(0,1fr)] gap-2 items-center">
                                <div className="relative h-24 w-24 rounded-full border border-border/50" style={{ background: allBotsLifecycleMixChart.gradient }}>
                                  <div className="absolute inset-[18px] rounded-full border border-border/60 bg-card" />
                                  <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                                    <span className="text-sm font-mono">{allBotsLifecycleMixChart.total}</span>
                                    <span className="text-[9px] uppercase tracking-wider text-muted-foreground">orders</span>
                                  </div>
                                </div>
                                <div className="space-y-1">
                                  {allBotsLifecycleMixChart.slices.map((slice) => (
                                    <div key={slice.key} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1.5 text-[10px]">
                                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: slice.color }} />
                                      <span>{slice.label}</span>
                                      <span className="font-mono text-muted-foreground">{slice.value}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              <div className="mt-2 space-y-1">
                                {riskActivityRows.length === 0 ? (
                                  <p className="text-[10px] text-muted-foreground">No active risk alerts.</p>
                                ) : (
                                  riskActivityRows.slice(0, 3).map((row) => (
                                    <p key={`${row.kind}:${row.id}`} className="truncate text-[10px] text-muted-foreground" title={row.title}>
                                      {formatTimestamp(row.ts)} · {row.title}
                                    </p>
                                  ))
                                )}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="min-h-0 flex-1 rounded-md border border-border/60 bg-card/80 overflow-hidden">
                          <div className="px-2.5 py-2 border-b border-border/40 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <Trophy className="w-3.5 h-3.5 text-cyan-500" />
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Bot Leaderboard</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground">Top {allBotsLeaderboardWithTrend.length}</span>
                          </div>
                          <ScrollArea className="h-[280px] xl:h-full">
                            <div className="space-y-1.5 p-2">
                              {allBotsLeaderboardWithTrend.length === 0 ? (
                                <p className="py-8 text-center text-[11px] text-muted-foreground">No bot performance data yet.</p>
                              ) : (
                                allBotsLeaderboardWithTrend.map((row) => {
                                  const traderStatus = resolveTraderStatusPresentation(row.trader, orchestratorRunning)
                                  return (
                                    <button
                                      key={row.trader.id}
                                      type="button"
                                      onClick={() => setSelectedTraderId(row.trader.id)}
                                      className={cn(
                                        'w-full rounded-md border border-border/50 bg-background/40 px-2.5 py-2 text-left transition-colors hover:border-cyan-500/40 hover:bg-cyan-500/5',
                                        selectedTraderId === row.trader.id && 'border-cyan-500/50 bg-cyan-500/10'
                                      )}
                                    >
                                      <div className="flex items-center gap-2">
                                        <span className="w-5 shrink-0 text-center text-[10px] font-mono text-muted-foreground">
                                          #{row.rank}
                                        </span>
                                        <div className="min-w-0 flex-1">
                                          <div className="flex items-center gap-1.5">
                                            <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', traderStatus.dotClassName)} />
                                            <span className="truncate text-[11px] font-medium" title={row.trader.name}>{row.trader.name}</span>
                                          </div>
                                          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted/70">
                                            <div
                                              className={cn('h-full rounded-full', row.pnl >= 0 ? 'bg-emerald-500/80' : 'bg-red-500/80')}
                                              style={{ width: `${row.pnlBarPercent}%` }}
                                            />
                                          </div>
                                        </div>
                                        <div className="shrink-0 text-right">
                                          <p className={cn('text-[11px] font-mono', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                                            {formatCurrency(row.pnl, true)}
                                          </p>
                                          <p className="text-[9px] text-muted-foreground">WR {formatPercent(row.winRate)}</p>
                                        </div>
                                      </div>
                                      <div className="mt-1.5 flex items-center justify-between gap-2">
                                        <span className="text-[9px] text-muted-foreground">
                                          {row.open} open · {row.resolved} resolved
                                        </span>
                                        {row.trend.length >= 2 && (
                                          <Liveline
                                            data={toTimeValueSeries(row.trend)}
                                            value={row.trend[row.trend.length - 1] ?? 0}
                                            color={row.pnl >= 0 ? '#22c55e' : '#ef4444'}
                                            theme={themeMode}
                                            window={(row.trend.length - 1) * 60}
                                            paused
                                            grid={false}
                                            badge={false}
                                            fill
                                            pulse={false}
                                            momentum={false}
                                            scrub={false}
                                            lerpSpeed={0.2}
                                            padding={{ top: 2, right: 2, bottom: 2, left: 2 }}
                                            style={{ height: 24, width: 96 }}
                                          />
                                        )}
                                      </div>
                                    </button>
                                  )
                                })
                              )}
                            </div>
                          </ScrollArea>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="trades" className="mt-2 flex-1 min-h-0 overflow-hidden">
                    <div className="h-full flex flex-col min-h-0 gap-1.5">
                      <div className="shrink-0 flex flex-wrap items-center gap-1">
                        <Input
                          value={allBotsTradeSearch}
                          onChange={(event) => setAllBotsTradeSearch(event.target.value)}
                          placeholder="Search bot, market, source..."
                          className="h-6 w-56 text-[11px]"
                        />
                        {TRADE_STATUS_FILTER_OPTIONS.map((statusOption) => (
                          <Button
                            key={statusOption.value}
                            size="sm"
                            variant={allBotsTradeStatusFilter === statusOption.value ? 'default' : 'outline'}
                            onClick={() => setAllBotsTradeStatusFilter(statusOption.value)}
                            className="h-5 px-2 text-[10px]"
                          >
                            {statusOption.label}
                          </Button>
                        ))}
                        <span className="ml-auto text-[10px] font-mono text-muted-foreground">{filteredAllTradeHistory.length} rows</span>
                      </div>
                      <div className="flex-1 min-h-0 flex flex-col rounded-md border border-border/60 bg-card/80">
                        {filteredAllTradeHistory.length === 0 ? (
                          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No trades matching filters.</div>
                        ) : (
                          <>
                            <div className="shrink-0 border-b border-border/60">
                              <table className="w-full table-fixed text-sm">
                                <thead>
                                  <tr className="border-b">
                                    <th className="w-[32%] h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[10px]">Market</th>
                                    <th className="w-[6%] h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[10px]">Dir</th>
                                    <th className="w-[8%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">Notional</th>
                                    <th className="w-[6%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">Fill</th>
                                    <th className="w-[6%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">Fill Progress</th>
                                    <th className="w-[6%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">Mark</th>
                                    <th className="w-[8%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">U-P&amp;L</th>
                                    <th className="w-[7%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">Edge Δ</th>
                                    <th className="w-[8%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">R-P&amp;L</th>
                                    <th className="w-[8%] h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[10px]">Venue</th>
                                    <th className="w-[6%] h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[10px]">Exit %</th>
                                    <th className="w-[5%] h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[10px]">Mark Age</th>
                                    <th className="w-[5%] h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[10px]">Eval Age</th>
                                  </tr>
                                </thead>
                              </table>
                            </div>
                            <div className="flex-1 min-h-0 overflow-auto">
                              <Table className="w-full table-fixed">
                                <TableBody>
                                {filteredAllTradeHistory.map((order) => {
                                  const status = normalizeStatus(order.status)
                                  const lifecycleLabel = resolveOrderLifecycleLabel(status)
                                  const pnl = toNumber(order.actual_profit)
                                  const orderPayload = order.payload && typeof order.payload === 'object' ? order.payload : {}
                                  const providerReconciliation = orderPayload.provider_reconciliation && typeof orderPayload.provider_reconciliation === 'object'
                                    ? orderPayload.provider_reconciliation
                                    : {}
                                  const providerSnapshot = providerReconciliation.snapshot && typeof providerReconciliation.snapshot === 'object'
                                    ? providerReconciliation.snapshot
                                    : {}
                                  const positionState = orderPayload.position_state && typeof orderPayload.position_state === 'object'
                                    ? orderPayload.position_state
                                    : {}
                                  const pendingExit = orderPayload.pending_live_exit && typeof orderPayload.pending_live_exit === 'object'
                                    ? orderPayload.pending_live_exit
                                    : {}
                                  const providerSnapshotStatus = normalizeStatus(
                                    String(
                                      order.provider_snapshot_status
                                      || providerReconciliation.snapshot_status
                                      || providerSnapshot.normalized_status
                                      || providerSnapshot.status
                                      || ''
                                    )
                                  )
                                  const fillPx = toNumber(
                                    order.average_fill_price
                                    ?? providerReconciliation.average_fill_price
                                    ?? providerSnapshot.average_fill_price
                                    ?? order.effective_price
                                    ?? order.entry_price
                                  )
                                  const directionPresentation = resolveOrderDirectionPresentation(order)
                                  const realtimeCrypto = resolveOrderRealtimeCryptoSnapshot(order, directionPresentation.side)
                                  const markPx = toNumber(
                                    realtimeCrypto.markPrice
                                    ?? order.current_price
                                    ?? positionState.last_mark_price
                                    ?? orderPayload.market_price
                                    ?? orderPayload.resolved_price
                                  )
                                  const filledSize = toNumber(
                                    order.filled_shares
                                    ?? providerReconciliation.filled_size
                                    ?? providerSnapshot.filled_size
                                    ?? orderPayload.filled_size
                                  )
                                  const filledNotional = toNumber(
                                    order.filled_notional_usd
                                    ?? providerReconciliation.filled_notional_usd
                                    ?? providerSnapshot.filled_notional_usd
                                    ?? order.notional_usd
                                  )
                                  let unrealized = toNumber(order.unrealized_pnl)
                                  if ((order.unrealized_pnl === null || order.unrealized_pnl === undefined) && markPx > 0 && filledSize > 0 && filledNotional > 0) {
                                    unrealized = (markPx * filledSize) - filledNotional
                                  }
                                  const fillProgressPercent = computeOrderFillProgressPercent(
                                    orderPayload as Record<string, unknown>,
                                    {
                                      filledSize,
                                      filledNotional,
                                      requestedNotionalFallback: Math.abs(toNumber(order.notional_usd)),
                                    }
                                  )
                                   const dynamicEdgePercent = computeOrderDynamicEdgePercent({
                                     status,
                                     edgePercent: toNumber(order.edge_percent),
                                     unrealizedPnl: unrealized,
                                     realizedPnl: pnl,
                                     filledNotional,
                                   })
                                  const exitProgressPercent = computePendingExitProgressPercent(pendingExit as Record<string, unknown>)
                                  const linkedDecisionId = String(order.decision_id || '').trim()
                                  const signalPayload = linkedDecisionId
                                    ? decisionSignalPayloadByDecisionId.get(linkedDecisionId) || null
                                    : null
                                  const links = buildOrderMarketLinks(order, orderPayload, signalPayload)
                                  const primaryMarketLink = links.polymarket || links.kalshi
                                  const markUpdatedAt = latestTimestampValue(
                                    realtimeCrypto.updatedAt,
                                    resolveOrderMarketUpdateTimestamp(order, orderPayload),
                                  )
                                  const exitEvaluatedAt = resolveOrderExitEvaluationTimestamp(order, orderPayload)
                                  const positionClose = orderPayload.position_close && typeof orderPayload.position_close === 'object'
                                    ? orderPayload.position_close
                                    : {}
                                  const closeTrigger = cleanText(
                                    order.close_trigger
                                    || (positionClose as Record<string, unknown>).close_trigger
                                    || (pendingExit as Record<string, unknown>).close_trigger
                                  )
                                  const pendingExitStatus = normalizeStatus(String((pendingExit as Record<string, unknown>).status || ''))
                                  const pendingExitLabel = pendingExitStatus && pendingExitStatus !== 'unknown'
                                    ? `Exit:${pendingExitStatus.slice(0, 4).toUpperCase()}`
                                    : null
                                  const pendingExitTone: 'neutral' | 'warning' =
                                    pendingExitStatus === 'failed' && OPEN_ORDER_STATUSES.has(status)
                                      ? 'warning'
                                      : 'neutral'
                                  const outcome = orderOutcomeSummary(order)
                                  const executionSummary = orderExecutionTypeSummary(order)
                                  const venuePresentation = resolveVenueStatusPresentation(providerSnapshotStatus)
                                  const traderLabel = traderNameById[String(order.trader_id || '')] || shortId(order.trader_id)
                                  const marketForModal = resolveCryptoMarketFromAliases(collectOrderMarketAliasIds(order))
                                  const openModal = () => {
                                    openTradeMarketModal({
                                      market: marketForModal,
                                      order,
                                      directionSide: directionPresentation.side,
                                      directionLabel: directionPresentation.label,
                                      yesLabel: directionPresentation.yesLabel,
                                      noLabel: directionPresentation.noLabel,
                                      statusSummary: lifecycleLabel,
                                      executionSummary,
                                      outcomeSummary: outcome.detail,
                                      links,
                                    })
                                  }
                                  return (
                                    <Fragment key={order.id}>
                                      <TableRow
                                        className="border-b-0 bg-muted/[0.08] text-[11px] leading-tight cursor-pointer hover:bg-muted/[0.16] [&>td]:border-t [&>td]:border-border/70 [&>td:first-child]:border-l [&>td:last-child]:border-r"
                                        onClick={openModal}
                                      >
                                        <TableCell className="max-w-[260px] py-0.5" title={order.market_question || order.market_id}>
                                          <div className="flex min-w-0 items-center gap-1">
                                            <div className="flex shrink-0 items-center gap-0.5">
                                              {links.polymarket && (
                                                <a
                                                  href={links.polymarket}
                                                  target="_blank"
                                                  rel="noopener noreferrer"
                                                  onClick={(event) => event.stopPropagation()}
                                                  className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                                  title="Open Polymarket market"
                                                >
                                                  <ExternalLink className="h-3 w-3" />
                                                </a>
                                              )}
                                              {links.kalshi && (
                                                <a
                                                  href={links.kalshi}
                                                  target="_blank"
                                                  rel="noopener noreferrer"
                                                  onClick={(event) => event.stopPropagation()}
                                                  className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                                  title="Open Kalshi market"
                                                >
                                                  <ExternalLink className="h-3 w-3" />
                                                </a>
                                              )}
                                            </div>
                                            {primaryMarketLink ? (
                                              <a
                                                href={primaryMarketLink}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                onClick={(event) => event.stopPropagation()}
                                                className="truncate hover:underline underline-offset-2"
                                                title="Open market"
                                              >
                                                {order.market_question || shortId(order.market_id)}
                                              </a>
                                            ) : (
                                              <span className="truncate">
                                                {order.market_question || shortId(order.market_id)}
                                              </span>
                                            )}
                                          </div>
                                          <p className="truncate text-[9px] leading-none text-muted-foreground" title={traderLabel}>
                                            {traderLabel}
                                          </p>
                                        </TableCell>
                                        <TableCell className="py-0.5">
                                          <Badge
                                            variant="outline"
                                            className="h-4 max-w-[120px] truncate border-border/80 bg-muted/60 px-1 text-[9px] text-muted-foreground"
                                            title={directionPresentation.label}
                                          >
                                            {directionPresentation.label}
                                          </Badge>
                                        </TableCell>
                                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatCurrency(toNumber(order.notional_usd), true)}</TableCell>
                                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{fillPx > 0 ? fillPx.toFixed(3) : '\u2014'}</TableCell>
                                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{fillProgressPercent !== null ? formatPercent(fillProgressPercent, 0) : '\u2014'}</TableCell>
                                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{markPx > 0 ? markPx.toFixed(3) : '\u2014'}</TableCell>
                                        <TableCell className={cn('text-right font-mono py-0.5 text-[10px] font-semibold', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}>
                                          {OPEN_ORDER_STATUSES.has(status) ? formatCurrency(unrealized) : '\u2014'}
                                        </TableCell>
                                        <TableCell className={cn('text-right font-mono py-0.5 text-[10px] font-semibold', dynamicEdgePercent > 0 ? 'text-emerald-500' : dynamicEdgePercent < 0 ? 'text-red-500' : '')}>{formatPercent(dynamicEdgePercent)}</TableCell>
                                        <TableCell className={cn('text-right font-mono py-0.5 text-[10px] font-semibold', pnl > 0 ? 'text-emerald-500' : pnl < 0 ? 'text-red-500' : '')}>
                                          {RESOLVED_ORDER_STATUSES.has(status) ? formatCurrency(pnl) : '\u2014'}
                                        </TableCell>
                                        <TableCell className="py-0.5">
                                          <Badge
                                            variant="outline"
                                            title={`Venue: ${venuePresentation.detail}${providerSnapshotStatus ? ` • provider:${providerSnapshotStatus}` : ''}`}
                                            className={cn('h-4 max-w-[120px] truncate px-1 text-[9px] font-semibold', venuePresentation.className)}
                                          >
                                            {venuePresentation.label}
                                          </Badge>
                                        </TableCell>
                                        <TableCell className="text-right font-mono py-0.5 text-[10px]">
                                          {exitProgressPercent !== null ? formatPercent(exitProgressPercent, 0) : '\u2014'}
                                        </TableCell>
                                      <TableCell className="py-0.5 text-[9px] text-muted-foreground">
                                        <span title={`${String(order.mode || '').toUpperCase()} • mark:${formatTimestamp(markUpdatedAt)} • created:${formatTimestamp(order.created_at)}`}>
                                          {formatRelativeAge(markUpdatedAt)}
                                        </span>
                                      </TableCell>
                                      <TableCell className="py-0.5 text-[9px] text-muted-foreground">
                                        <span title={`${String(order.mode || '').toUpperCase()} • exit eval:${formatTimestamp(exitEvaluatedAt)} • updated:${formatTimestamp(order.updated_at)}`}>
                                          {formatRelativeAge(exitEvaluatedAt)}
                                        </span>
                                      </TableCell>
                                      </TableRow>
                                      <TableRow className="cursor-pointer bg-muted/[0.08] hover:bg-muted/[0.16]" onClick={openModal}>
                                        <TableCell colSpan={13} className="border-b-2 border-l border-r border-border/80 px-0 py-0.5">
                                          {renderTradeLifecycleFlow({
                                            status,
                                            outcomeHeadline: outcome.headline,
                                            outcomeDetail: outcome.detail,
                                            executionSummary,
                                            venueLabel: venuePresentation.label,
                                            closeTrigger,
                                            pendingExitLabel,
                                            pendingExitTone,
                                            pulseCurrentStage: OPEN_ORDER_STATUSES.has(status) && String(order.mode || '').toLowerCase() === 'live',
                                          })}
                                        </TableCell>
                                      </TableRow>
                                    </Fragment>
                                  )
                                })}
                              </TableBody>
                            </Table>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="positions" className="mt-2 flex-1 min-h-0 overflow-hidden">
                    <div className="h-full flex flex-col min-h-0 gap-1.5">
                      <div className="shrink-0 flex flex-wrap items-center gap-1">
                        <Input
                          value={allBotsPositionSearch}
                          onChange={(event) => setAllBotsPositionSearch(event.target.value)}
                          placeholder="Search bot, market, source..."
                          className="h-6 w-56 text-[11px]"
                        />
                        {(['all', 'yes', 'no'] as PositionDirectionFilter[]).map((direction) => (
                          <Button
                            key={direction}
                            size="sm"
                            variant={allBotsPositionDirectionFilter === direction ? 'default' : 'outline'}
                            onClick={() => setAllBotsPositionDirectionFilter(direction)}
                            className="h-5 px-2 text-[10px]"
                          >
                            {direction}
                          </Button>
                        ))}
                        <Select
                          value={allBotsPositionSortField}
                          onValueChange={(value) => setAllBotsPositionSortField(value as PositionSortField)}
                        >
                          <SelectTrigger className="h-6 w-[132px] text-[11px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="exposure">Exposure</SelectItem>
                            <SelectItem value="unrealized">U-P&L</SelectItem>
                            <SelectItem value="edge">Edge</SelectItem>
                            <SelectItem value="confidence">Confidence</SelectItem>
                            <SelectItem value="updated">Updated</SelectItem>
                          </SelectContent>
                        </Select>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setAllBotsPositionSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'))}
                          className="h-5 px-2 text-[10px]"
                        >
                          {allBotsPositionSortDirection === 'desc' ? 'desc' : 'asc'}
                        </Button>
                        <span className="ml-auto text-[10px] font-mono text-muted-foreground">{filteredAllPositionBook.length} rows</span>
                      </div>
                      <div className="shrink-0 grid grid-cols-2 gap-1 sm:grid-cols-4 lg:grid-cols-8">
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Positions</p>
                          <p className="text-xs font-mono">{allBotsPositionSummary.totalRows}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">YES / NO</p>
                          <p className="text-xs font-mono">{allBotsPositionSummary.yesRows} / {allBotsPositionSummary.noRows}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Exposure</p>
                          <p className="text-xs font-mono">{formatCurrency(allBotsPositionSummary.totalExposure, true)}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">U-P&amp;L</p>
                          <p className={cn(
                            'text-xs font-mono',
                            allBotsPositionSummary.totalUnrealizedPnl > 0 ? 'text-emerald-500' : allBotsPositionSummary.totalUnrealizedPnl < 0 ? 'text-red-500' : ''
                          )}
                          >
                            {allBotsPositionSummary.rowsWithUnrealized > 0
                              ? formatCurrency(allBotsPositionSummary.totalUnrealizedPnl, true)
                              : '—'}
                          </p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Avg Edge</p>
                          <p className="text-xs font-mono">{formatPercent(allBotsPositionSummary.avgEdge)}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Avg Conf</p>
                          <p className="text-xs font-mono">{formatPercent(normalizeConfidencePercent(allBotsPositionSummary.avgConfidence))}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Live / Shadow</p>
                          <p className="text-xs font-mono">{allBotsPositionSummary.liveOrders} / {allBotsPositionSummary.shadowOrders}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Marks</p>
                          <p className="text-xs font-mono">{allBotsPositionSummary.freshMarks} / {allBotsPositionSummary.markedRows}</p>
                        </div>
                      </div>
                      <div className="flex-1 min-h-0 flex flex-col rounded-md border border-border/60 bg-card/80">
                        {filteredAllPositionBook.length === 0 ? (
                          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No positions matching filters.</div>
                        ) : (
                          <>
                            <div className="shrink-0 border-b border-border/60 overflow-x-auto">
                              <table className="min-w-[1240px] w-full text-sm">
                                <thead>
                                  <tr className="border-b">
                                    <th className="h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[11px]">Market</th>
                                    <th className="h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[11px]">L</th>
                                    <th className="h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[11px]">Dir</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Exposure</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Avg Px</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Mark</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">U-P&amp;L</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Edge</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Conf</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Orders</th>
                                    <th className="h-8 px-4 text-right align-middle font-medium text-muted-foreground text-[11px]">Mode</th>
                                    <th className="h-8 px-4 text-left align-middle font-medium text-muted-foreground text-[11px]">Updated</th>
                                  </tr>
                                </thead>
                              </table>
                            </div>
                            <div className="flex-1 min-h-0 overflow-auto">
                              <Table className="min-w-[1240px]">
                                <TableBody>
                                {filteredAllPositionBook.map((row) => {
                                  const marketForModal = resolveCryptoMarketFromAliases([row.marketId, ...row.marketAliases])
                                  return (
                                  <TableRow
                                    key={row.key}
                                    className="text-xs cursor-pointer hover:bg-muted/30"
                                    onClick={() => {
                                      openPositionMarketModal({
                                        market: marketForModal,
                                        row,
                                      })
                                    }}
                                  >
                                    <TableCell className="max-w-[280px] truncate py-1" title={row.marketQuestion}>
                                      <p className="truncate">{row.marketQuestion}</p>
                                      <p className="text-[10px] text-muted-foreground truncate" title={positionMetaLine(row)}>
                                        {row.traderName} • {positionMetaLine(row)}
                                      </p>
                                    </TableCell>
                                    <TableCell className="py-1">
                                      <div className="flex items-center gap-1">
                                        {row.links.polymarket && (
                                          <a
                                            href={row.links.polymarket}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            onClick={(event) => event.stopPropagation()}
                                            className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                            title="Open Polymarket market"
                                          >
                                            <ExternalLink className="h-3 w-3" />
                                          </a>
                                        )}
                                        {row.links.kalshi && (
                                          <a
                                            href={row.links.kalshi}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            onClick={(event) => event.stopPropagation()}
                                            className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                            title="Open Kalshi market"
                                          >
                                            <ExternalLink className="h-3 w-3" />
                                          </a>
                                        )}
                                        {!row.links.polymarket && !row.links.kalshi && (
                                          <span className="text-[9px] text-muted-foreground">—</span>
                                        )}
                                      </div>
                                  </TableCell>
                                  <TableCell className="py-1">
                                      <Badge variant="outline" className="h-5 max-w-[140px] truncate border-border/80 bg-muted/60 px-1.5 text-[10px] text-muted-foreground" title={row.direction}>
                                        {row.direction}
                                      </Badge>
                                  </TableCell>
                                    <TableCell className="text-right font-mono py-1">{formatCurrency(row.exposureUsd)}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.averagePrice !== null ? row.averagePrice.toFixed(3) : '—'}</TableCell>
                                    <TableCell className={cn('text-right font-mono py-1', row.markFresh && 'text-sky-300')}>
                                      {row.markPrice !== null ? (
                                        <FlashNumber
                                          value={row.markPrice}
                                          decimals={3}
                                          className={cn('font-mono text-xs', row.markFresh ? 'data-glow-blue' : '')}
                                          positiveClass="data-glow-green"
                                          negativeClass="data-glow-red"
                                        />
                                      ) : '—'}
                                    </TableCell>
                                    <TableCell className={cn('text-right font-mono py-1', (row.unrealizedPnl || 0) > 0 ? 'text-emerald-500' : (row.unrealizedPnl || 0) < 0 ? 'text-red-500' : '')}>
                                      {row.unrealizedPnl !== null ? formatCurrency(row.unrealizedPnl) : '—'}
                                    </TableCell>
                                    <TableCell className={cn('text-right font-mono py-1 font-semibold', (row.weightedEdge || 0) > 0 ? 'text-emerald-500' : (row.weightedEdge || 0) < 0 ? 'text-red-500' : '')}>{row.weightedEdge !== null ? formatPercent(row.weightedEdge) : '—'}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.weightedConfidence !== null ? formatPercent(normalizeConfidencePercent(row.weightedConfidence)) : '—'}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.orderCount}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.liveOrderCount}L/{row.shadowOrderCount}S</TableCell>
                                    <TableCell className="py-1 text-[10px] text-muted-foreground">{formatShortDate(row.lastUpdated || row.markUpdatedAt)}</TableCell>
                                  </TableRow>
                                  )
                                })}
                              </TableBody>
                            </Table>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </TabsContent>
	                </Tabs>
	              </div>
	            </div>
	          ) : (
            <>
              {selectedTrader && (
                <div className="shrink-0 rounded-lg border border-border/70 bg-card px-3 py-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="text-sm font-semibold">{selectedTrader.name}</span>
                  <Badge
                    className={cn('h-5 px-1.5 text-[10px]', selectedTraderStatus.badgeClassName)}
                    variant={selectedTraderStatus.badgeVariant}
                  >
                    {selectedTraderPendingAction === 'start'
                      ? 'Starting...'
                      : selectedTraderPendingAction === 'stop'
                        ? 'Stopping...'
                        : selectedTraderPendingAction === 'activate'
                          ? 'Activating...'
                          : selectedTraderPendingAction === 'deactivate'
                            ? 'Deactivating...'
                        : selectedTraderStatus.label}
                  </Badge>
                  <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-muted-foreground">
                    <span className={selectedTraderSummary.pnl >= 0 ? 'text-emerald-500' : 'text-red-500'}>{formatCurrency(selectedTraderSummary.pnl)}</span>
                    <span className="text-border">|</span>
                    <span>WR {formatPercent(selectedTraderSummary.winRate)}</span>
                    <span className="text-border">|</span>
                    <span>{selectedOrders.length} orders</span>
                    <span className="text-border">|</span>
                    <span>Exp {formatCurrency(selectedTraderExposure, true)}</span>
                    <span className="text-border">|</span>
                    <span>Edge {formatPercent(normalizeEdgePercent(selectedTraderSummary.avgEdge))}</span>
                  </div>
                  <div className="ml-auto flex items-center gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[10px]"
                      disabled={selectedTraderControlPending || !selectedTraderCanStart}
                      onClick={requestStartTrader}
                    >
                      {selectedTraderPendingAction === 'start' ? (
                        <>
                          <Loader2 className="w-3 h-3 mr-0.5 animate-spin" /> Starting...
                        </>
                      ) : (
                        <>
                          <Play className="w-3 h-3 mr-0.5" /> Start
                        </>
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[10px]"
                      disabled={selectedTraderControlPending || !selectedTraderCanStop}
                      onClick={requestStopTrader}
                    >
                      {selectedTraderPendingAction === 'stop' ? (
                        <>
                          <Loader2 className="w-3 h-3 mr-0.5 animate-spin" /> Stopping...
                        </>
                      ) : (
                        <>
                          <Square className="w-3 h-3 mr-0.5" /> Stop
                        </>
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant={selectedTraderIsActive ? 'secondary' : 'outline'}
                      className="h-6 px-2 text-[10px]"
                      disabled={
                        selectedTraderControlPending
                        || (selectedTraderIsActive ? !selectedTraderCanDeactivate : !selectedTraderCanActivate)
                      }
                      onClick={selectedTraderIsActive ? requestDeactivateTrader : requestActivateTrader}
                    >
                      {selectedTraderPendingAction === 'activate' ? (
                        <>
                          <Loader2 className="w-3 h-3 mr-0.5 animate-spin" /> Activating...
                        </>
                      ) : selectedTraderPendingAction === 'deactivate' ? (
                        <>
                          <Loader2 className="w-3 h-3 mr-0.5 animate-spin" /> Deactivating...
                        </>
                      ) : selectedTraderIsActive ? (
                        <>
                          <Square className="w-3 h-3 mr-0.5" /> Deactivate
                        </>
                      ) : (
                        <>
                          <Play className="w-3 h-3 mr-0.5" /> Activate
                        </>
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[10px]"
                      onClick={() => traderRunOnceMutation.mutate(selectedTrader.id)}
                      disabled={traderRunOnceMutation.isPending || selectedTraderControlPending}
                    >
                      <Zap className="w-3 h-3 mr-0.5" /> Once
                    </Button>
                    <Button size="sm" variant="outline" className="h-6 px-2 text-[10px]" onClick={() => openEditTraderFlyout(selectedTrader)}>
                      <Settings className="w-3 h-3 mr-0.5" /> Config
                    </Button>
                  </div>
                </div>
              )}

              <div className="shrink-0 flex items-center gap-0.5 border-b border-border/50 px-1">
                {([
                  { key: 'trades' as const, label: 'Trades' },
                  { key: 'terminal' as const, label: 'Terminal' },
                  { key: 'tune' as const, label: 'Tune' },
                  { key: 'decisions' as const, label: 'Decisions' },
                  { key: 'performance' as const, label: 'Performance' },
                ]).map((tab) => (
                  <button
                    key={tab.key}
                    type="button"
                    onClick={() => setWorkTab(tab.key)}
                    className={cn(
                      'px-3 py-1.5 text-[11px] font-medium transition-colors border-b-2 -mb-[1px]',
                      workTab === tab.key
                        ? 'border-cyan-500 text-foreground'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="flex-1 min-h-0 overflow-hidden">
                {workTab === 'terminal' && (
                  <div className="h-full flex flex-col min-h-0 gap-1.5">
                    <div className="shrink-0 flex flex-wrap items-center gap-1 px-1">
                      {(['all', 'decision', 'order', 'event'] as FeedFilter[]).map((kind) => (
                        <Button key={kind} size="sm" variant={traderFeedFilter === kind ? 'default' : 'outline'} onClick={() => setTraderFeedFilter(kind)} className="h-5 px-2 text-[10px]">
                          {kind}
                        </Button>
                      ))}
                      <div className="ml-1 inline-flex items-center gap-1">
                        <Button size="sm" variant={terminalDensity === 'compact' ? 'default' : 'outline'} onClick={() => setTerminalDensity('compact')} className="h-5 px-2 text-[10px]">
                          compact
                        </Button>
                        <Button size="sm" variant={terminalDensity === 'expanded' ? 'default' : 'outline'} onClick={() => setTerminalDensity('expanded')} className="h-5 px-2 text-[10px]">
                          expanded
                        </Button>
                      </div>
                      <span className="text-[10px] text-muted-foreground ml-1">Auto-truncate: latest {TERMINAL_SELECTED_MAX_ROWS}</span>
                      {terminalDensity === 'compact' && (
                        <span className="text-[10px] text-muted-foreground">Rendering {compactTerminalWindow.rows.length}/{compactTerminalWindow.total}</span>
                      )}
                    </div>
                    {selectedTraderNoNewRows && (
                      <div className="shrink-0 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-700 dark:text-amber-100 mx-1">
                        No new rows since last cycle ({formatTimestamp(selectedTrader?.last_run_at || worker?.last_run_at)}).
                      </div>
                    )}
                    <div
                      ref={terminalViewportRef}
                      onScroll={(event) => {
                        if (terminalDensity !== 'compact') return
                        setTerminalScrollTop(event.currentTarget.scrollTop)
                      }}
                      className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10 mx-1 overflow-auto"
                    >
                      {filteredTraderActivityRows.length === 0 ? (
                        <div className="py-8 text-center text-muted-foreground text-xs">No events matching filters.</div>
                      ) : terminalDensity === 'compact' ? (
                        <div className="p-1.5 font-mono text-[11px]">
                          <div style={{ height: compactTerminalWindow.topPad }} />
                          <div className="space-y-0.5">
                            {compactTerminalWindow.rows.map((row) => (
                              <div
                                key={`${row.kind}:${row.id}`}
                                style={{ minHeight: TERMINAL_COMPACT_ROW_HEIGHT }}
                                className={cn(
                                  'rounded border px-2 py-1 flex items-center gap-1.5 whitespace-nowrap',
                                  row.tone === 'positive' && 'border-emerald-500/25 text-emerald-700 dark:text-emerald-100',
                                  row.tone === 'negative' && 'border-red-500/30 text-red-700 dark:text-red-100',
                                  row.tone === 'warning' && 'border-amber-500/30 text-amber-700 dark:text-amber-100',
                                  row.tone === 'neutral' && row.action === 'BUY' && 'border-emerald-500/25 bg-emerald-500/5 text-emerald-700 dark:text-emerald-100',
                                  row.tone === 'neutral' && row.action === 'SELL' && 'border-red-500/30 bg-red-500/5 text-red-700 dark:text-red-100',
                                  row.tone === 'neutral' && !row.action && 'border-border/50 text-foreground'
                                )}
                              >
                                <span className="text-muted-foreground shrink-0">[{formatTimestamp(row.ts)}]</span>
                                <span className="uppercase text-[10px] shrink-0">{row.kind}</span>
                                {row.action && (
                                  <span
                                    className={cn(
                                      'uppercase text-[10px] font-semibold shrink-0',
                                      row.action === 'BUY' ? 'text-emerald-500' : 'text-red-500'
                                    )}
                                  >
                                    {row.action}
                                  </span>
                                )}
                                <span className="font-medium truncate">{row.title}</span>
                                <span className="text-muted-foreground truncate">{row.detail}</span>
                              </div>
                            ))}
                          </div>
                          <div style={{ height: compactTerminalWindow.bottomPad }} />
                        </div>
                      ) : (
                        <div className="space-y-0.5 p-1.5 font-mono text-[11px] leading-relaxed">
                          {filteredTraderActivityRows.map((row) => (
                            <div
                              key={`${row.kind}:${row.id}`}
                              className={cn(
                                'rounded border px-2 py-1',
                                row.tone === 'positive' && 'border-emerald-500/25 text-emerald-700 dark:text-emerald-100',
                                row.tone === 'negative' && 'border-red-500/30 text-red-700 dark:text-red-100',
                                row.tone === 'warning' && 'border-amber-500/30 text-amber-700 dark:text-amber-100',
                                row.tone === 'neutral' && row.action === 'BUY' && 'border-emerald-500/25 bg-emerald-500/5 text-emerald-700 dark:text-emerald-100',
                                row.tone === 'neutral' && row.action === 'SELL' && 'border-red-500/30 bg-red-500/5 text-red-700 dark:text-red-100',
                                row.tone === 'neutral' && !row.action && 'border-border/50 text-foreground'
                              )}
                            >
                              <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
                                <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>
                                <span className="uppercase text-[10px]">{row.kind}</span>
                                {row.action && (
                                  <span className={cn(
                                    'uppercase text-[10px] font-semibold',
                                    row.action === 'BUY' ? 'text-emerald-500' : 'text-red-500'
                                  )}
                                  >
                                    {row.action}
                                  </span>
                                )}
                                <span className="font-medium">{row.title}</span>
                              </div>
                              <div className="text-[10px] leading-relaxed text-muted-foreground mt-0.5 break-words">{row.detail}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {workTab === 'trades' && (
                  <div className="h-full flex flex-col min-h-0 gap-1.5">
                    <div className="shrink-0 flex flex-wrap items-center gap-1 px-1">
                      <Input value={tradeSearch} onChange={(event) => setTradeSearch(event.target.value)} placeholder="Search..." className="h-6 w-36 text-[11px]" />
                      {TRADE_STATUS_FILTER_OPTIONS.map((statusOption) => (
                        <Button
                          key={statusOption.value}
                          size="sm"
                          variant={tradeStatusFilter === statusOption.value ? 'default' : 'outline'}
                          onClick={() => setTradeStatusFilter(statusOption.value)}
                          className="h-5 px-2 text-[10px]"
                        >
                          {statusOption.label}
                        </Button>
                      ))}
                    </div>
                    <div className="shrink-0 grid grid-cols-2 gap-1 px-1 sm:grid-cols-4 lg:grid-cols-8">
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Orders</p>
                        <p className="text-xs font-mono">{selectedTradeTotals.total}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Open</p>
                        <p className="text-xs font-mono">{selectedTradeTotals.open}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Win / Loss</p>
                        <p className="text-xs font-mono">{selectedTradeTotals.wins} / {selectedTradeTotals.losses}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Win Rate</p>
                        <p className="text-xs font-mono">{formatPercent(selectedTradeTotals.winRate)}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Failed</p>
                        <p className="text-xs font-mono">{selectedTradeTotals.failed}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Notional</p>
                        <p className="text-xs font-mono">{formatCurrency(selectedTradeTotals.totalNotional, true)}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">R-P&amp;L</p>
                        <p className={cn('text-xs font-mono', selectedTradeTotals.realizedPnl > 0 ? 'text-emerald-500' : selectedTradeTotals.realizedPnl < 0 ? 'text-red-500' : '')}>
                          {formatCurrency(selectedTradeTotals.realizedPnl, true)}
                        </p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">U-P&amp;L</p>
                        <p className={cn('text-xs font-mono', selectedTradeTotals.unrealizedPnl > 0 ? 'text-emerald-500' : selectedTradeTotals.unrealizedPnl < 0 ? 'text-red-500' : '')}>
                          {formatCurrency(selectedTradeTotals.unrealizedPnl, true)}
                        </p>
                      </div>
                    </div>
                    <div className="flex-1 min-h-0 overflow-hidden px-1">
                      {selectedTradeRows.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No trades matching filters.</div>
                      ) : (
                        <div className="h-full min-h-0 overflow-auto">
                          <Table className="w-full table-fixed">
                            <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
                              <TableRow>
                                <TableHead className="w-[32%] text-[10px]">Market</TableHead>
                                <TableHead className="w-[6%] text-[10px]">Dir</TableHead>
                                <TableHead className="w-[8%] text-[10px] text-right">Notional</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Fill</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Fill Progress</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Mark</TableHead>
                                <TableHead className="w-[8%] text-[10px] text-right">U-P&amp;L</TableHead>
                                <TableHead className="w-[7%] text-[10px] text-right">Edge Δ</TableHead>
                                <TableHead className="w-[8%] text-[10px] text-right">R-P&amp;L</TableHead>
                                <TableHead className="w-[8%] text-[10px]">Venue</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Exit %</TableHead>
                                <TableHead className="w-[5%] text-[10px]">Mark Age</TableHead>
                                <TableHead className="w-[5%] text-[10px]">Eval Age</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedTradeRows.map((row) => {
                                const {
                                  order,
                                  status,
                                  pnl,
                                  lifecycleLabel,
                                  fillPx,
                                  markPx,
                                  unrealized,
                                  fillProgressPercent,
                                  dynamicEdgePercent,
                                  exitProgressPercent,
                                  markUpdatedAt,
                                  exitEvaluatedAt,
                                  providerSnapshotStatus,
                                  pendingExitStatus,
                                  closeTrigger,
                                  markFresh,
                                  links,
                                  directionSide,
                                  directionLabel,
                                  yesLabel,
                                  noLabel,
                                  executionSummary,
                                  outcomeHeadline,
                                  outcomeDetail,
                                  venuePresentation,
                                } = row
                                const pendingExitLabel = pendingExitStatus && pendingExitStatus !== 'unknown'
                                  ? (pendingExitStatus === 'failed' && OPEN_ORDER_STATUSES.has(status)
                                    ? 'Exit:RETRY'
                                    : `Exit:${pendingExitStatus.slice(0, 4).toUpperCase()}`)
                                  : null
                                const pendingExitTone: 'neutral' | 'warning' =
                                  pendingExitStatus === 'failed' && OPEN_ORDER_STATUSES.has(status)
                                    ? 'warning'
                                    : 'neutral'
                                const marketForModal = resolveCryptoMarketFromAliases(collectOrderMarketAliasIds(order))
                                const openModal = () => {
                                  openTradeMarketModal({
                                    market: marketForModal,
                                    order,
                                    directionSide,
                                    directionLabel,
                                    yesLabel,
                                    noLabel,
                                    statusSummary: lifecycleLabel,
                                    executionSummary,
                                    outcomeSummary: outcomeDetail,
                                    links,
                                  })
                                }
                                const primaryMarketLink = links.polymarket || links.kalshi
                                return (
                                  <Fragment key={order.id}>
                                    <TableRow
                                      className="border-b-0 bg-muted/[0.08] text-[11px] leading-tight cursor-pointer hover:bg-muted/[0.16] [&>td]:border-t [&>td]:border-border/70 [&>td:first-child]:border-l [&>td:last-child]:border-r"
                                      onClick={openModal}
                                    >
                                      <TableCell className="max-w-[260px] py-0.5" title={order.market_question || order.market_id}>
                                        <div className="flex min-w-0 items-center gap-1">
                                          <div className="flex shrink-0 items-center gap-0.5">
                                            {links.polymarket && (
                                              <a
                                                href={links.polymarket}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                onClick={(event) => event.stopPropagation()}
                                                className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                                title="Open Polymarket market"
                                              >
                                                <ExternalLink className="h-3 w-3" />
                                              </a>
                                            )}
                                            {links.kalshi && (
                                              <a
                                                href={links.kalshi}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                onClick={(event) => event.stopPropagation()}
                                                className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                                title="Open Kalshi market"
                                              >
                                                <ExternalLink className="h-3 w-3" />
                                              </a>
                                            )}
                                          </div>
                                          {primaryMarketLink ? (
                                            <a
                                              href={primaryMarketLink}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              onClick={(event) => event.stopPropagation()}
                                              className="truncate hover:underline underline-offset-2"
                                              title="Open market"
                                            >
                                              {order.market_question || shortId(order.market_id)}
                                            </a>
                                          ) : (
                                            <span className="truncate">{order.market_question || shortId(order.market_id)}</span>
                                          )}
                                        </div>
                                      </TableCell>
                                      <TableCell className="py-0.5">
                                        <Badge
                                          variant="outline"
                                          className="h-4 max-w-[120px] truncate border-border/80 bg-muted/60 px-1 text-[9px] text-muted-foreground"
                                          title={directionLabel}
                                        >
                                          {directionLabel}
                                        </Badge>
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatCurrency(toNumber(order.notional_usd), true)}</TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{fillPx > 0 ? fillPx.toFixed(3) : '\u2014'}</TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{fillProgressPercent !== null ? formatPercent(fillProgressPercent, 0) : '\u2014'}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', markFresh && 'text-sky-300')}>
                                        {markPx > 0 ? (
                                          <FlashNumber
                                            value={markPx}
                                            decimals={3}
                                            className={cn('font-mono text-[10px]', markFresh ? 'data-glow-blue' : '')}
                                            positiveClass="data-glow-green"
                                            negativeClass="data-glow-red"
                                          />
                                        ) : '\u2014'}
                                      </TableCell>
                                      <TableCell className={cn('text-right font-mono py-0.5 text-[10px] font-semibold', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}>
                                        {OPEN_ORDER_STATUSES.has(status) ? (
                                          <FlashNumber
                                            value={unrealized}
                                            decimals={2}
                                            prefix="$"
                                            className={cn('font-mono text-[10px] font-semibold', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}
                                            positiveClass="data-glow-green"
                                            negativeClass="data-glow-red"
                                          />
                                        ) : '\u2014'}
                                      </TableCell>
                                      <TableCell className={cn('text-right font-mono py-0.5 text-[10px] font-semibold', dynamicEdgePercent > 0 ? 'text-emerald-500' : dynamicEdgePercent < 0 ? 'text-red-500' : '')}>{formatPercent(dynamicEdgePercent)}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-0.5 text-[10px] font-semibold', pnl > 0 ? 'text-emerald-500' : pnl < 0 ? 'text-red-500' : '')}>
                                        {RESOLVED_ORDER_STATUSES.has(status) ? formatCurrency(pnl, true) : '\u2014'}
                                      </TableCell>
                                      <TableCell className="py-0.5">
                                        <Badge
                                          variant="outline"
                                          title={
                                            `${venuePresentation.detail}`
                                            + (providerSnapshotStatus ? ` • provider:${providerSnapshotStatus}` : '')
                                            + (
                                              order.provider_clob_order_id || order.provider_order_id
                                                ? ` • order_ref:${order.provider_clob_order_id || order.provider_order_id}`
                                                : ''
                                            )
                                          }
                                          className={cn('h-4 max-w-[120px] truncate px-1 text-[9px] font-semibold', venuePresentation.className)}
                                        >
                                          {venuePresentation.label}
                                        </Badge>
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">
                                        {exitProgressPercent !== null ? formatPercent(exitProgressPercent, 0) : '\u2014'}
                                      </TableCell>
                                      <TableCell className="py-0.5 text-[9px] text-muted-foreground">
                                        <span title={`${String(order.mode || '').toUpperCase()} • mark:${formatTimestamp(markUpdatedAt)} • created:${formatTimestamp(order.created_at)}`}>
                                          {formatRelativeAge(markUpdatedAt)}
                                        </span>
                                      </TableCell>
                                      <TableCell className="py-0.5 text-[9px] text-muted-foreground">
                                        <span title={`${String(order.mode || '').toUpperCase()} • exit eval:${formatTimestamp(exitEvaluatedAt)} • updated:${formatTimestamp(order.updated_at)}`}>
                                          {formatRelativeAge(exitEvaluatedAt)}
                                        </span>
                                      </TableCell>
                                    </TableRow>
                                    <TableRow className="cursor-pointer bg-muted/[0.08] hover:bg-muted/[0.16]" onClick={openModal}>
                                      <TableCell colSpan={13} className="border-b-2 border-l border-r border-border/80 px-0 py-0.5">
                                        {renderTradeLifecycleFlow({
                                          status,
                                          outcomeHeadline,
                                          outcomeDetail,
                                          executionSummary,
                                          venueLabel: venuePresentation.label,
                                          closeTrigger,
                                          pendingExitLabel,
                                          pendingExitTone,
                                          pulseCurrentStage: OPEN_ORDER_STATUSES.has(status) && String(order.mode || '').toLowerCase() === 'live',
                                        })}
                                      </TableCell>
                                    </TableRow>
                                  </Fragment>
                                )
                              })}
                            </TableBody>
                          </Table>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {workTab === 'tune' && (
                  <div className="h-full min-h-0 overflow-hidden px-1">
                    <div className="h-full min-h-0 rounded-md border border-border/50 bg-muted/10 p-2">
                      {!selectedTrader ? (
                        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-700 dark:text-amber-100">
                          Select a bot to use the agent.
                        </div>
                      ) : (
                        <div
                          className="grid h-full min-h-0 gap-2"
                          style={{ gridTemplateRows: 'minmax(0, calc(33.333% + 20px)) minmax(0, calc(66.667% - 20px))' }}
                        >
                          <div className="min-h-0 overflow-hidden rounded-md border border-cyan-500/30 bg-cyan-500/5 p-2.5">
                            <div className="flex h-full min-h-0 flex-col gap-2">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-1.5">
                                  <p className="text-[11px] font-medium">Agent</p>
                                  <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                                    {tuneIterateResponse?.session_id ? shortId(tuneIterateResponse.session_id) : 'idle'}
                                  </Badge>
                                  {runTuneIterateMutation.isPending ? (
                                    <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-amber-500">
                                      RUNNING
                                    </span>
                                  ) : null}
                                  {tuneAutoEnabled ? (
                                    <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-500">
                                      AUTO
                                    </span>
                                  ) : null}
                                </div>
                                <p className="text-[10px] text-muted-foreground">
                                  Bot: <span className="font-mono text-foreground/85">{selectedTrader.name}</span>
                                </p>
                              </div>
                              <p className="text-[10px] text-muted-foreground/80">
                                Run the agent against the latest trader context. High-confidence parameter updates apply immediately.
                              </p>
                              <div
                                className={cn(
                                  'grid min-h-0 flex-1 gap-2 overflow-hidden',
                                  tuneIterateResponse && !runTuneIterateMutation.isPending
                                    ? 'xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]'
                                    : ''
                                )}
                              >
                                <div className="min-h-0 flex flex-col gap-1.5">
                                  <details className="rounded-md border border-border/60 bg-background/80 px-2 py-1">
                                    <summary className="cursor-pointer text-[10px] font-medium text-muted-foreground">
                                      Agent Prompt
                                    </summary>
                                    <textarea
                                      value={tuneIteratePrompt}
                                      onChange={(event) => setTuneIteratePrompt(event.target.value)}
                                      className="mt-1 min-h-[72px] max-h-[96px] w-full rounded-md border border-border/60 bg-background px-2 py-1.5 text-xs leading-relaxed"
                                      placeholder="Describe optimization goal, constraints, and risk preferences..."
                                    />
                                  </details>
                                  <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-4">
                                    <div>
                                      <Label className="text-[11px] text-muted-foreground">Model Override</Label>
                                      <Input
                                        value={tuneIterateModel}
                                        onChange={(event) => setTuneIterateModel(event.target.value)}
                                        placeholder="app default"
                                        className="mt-1 h-8 text-xs font-mono"
                                      />
                                    </div>
                                    <div>
                                      <Label className="text-[11px] text-muted-foreground">Max Iterations</Label>
                                      <Input
                                        type="number"
                                        min={1}
                                        max={24}
                                        value={tuneIterateMaxIterations}
                                        onChange={(event) => setTuneIterateMaxIterations(event.target.value)}
                                        className="mt-1 h-8 text-xs font-mono"
                                      />
                                    </div>
                                    <div>
                                      <Label className="text-[11px] text-muted-foreground">Auto Agent</Label>
                                      <div className="mt-1 flex h-8 items-center justify-between rounded-md border border-border/60 bg-background px-2">
                                        <span className="text-[10px] text-muted-foreground">
                                          {selectedTraderExecutionEnabled ? 'while trading' : 'bot stopped/inactive'}
                                        </span>
                                        <Switch
                                          checked={tuneAutoEnabled}
                                          onCheckedChange={(checked) => {
                                            setTuneAutoEnabled(checked)
                                            setTuneAutoLastRunAt(checked ? Date.now() : null)
                                          }}
                                        />
                                      </div>
                                    </div>
                                    <div>
                                      <Label className="text-[11px] text-muted-foreground">Auto Interval (min)</Label>
                                      <Input
                                        type="number"
                                        min={1}
                                        max={360}
                                        value={tuneAutoIntervalMinutes}
                                        onChange={(event) => setTuneAutoIntervalMinutes(event.target.value)}
                                        className="mt-1 h-8 text-xs font-mono"
                                        disabled={!tuneAutoEnabled}
                                      />
                                    </div>
                                  </div>
                                  {tuneAutoEnabled ? (
                                    <p className="text-[10px] text-muted-foreground/80">
                                      Agent runs every {Math.max(1, Math.min(360, Math.trunc(toNumber(tuneAutoIntervalMinutes || 15) || 15)))} minute(s) while this bot is enabled.
                                      {tuneAutoLastRunAt ? ` Last run: ${formatTimestamp(new Date(tuneAutoLastRunAt).toISOString())}.` : ''}
                                    </p>
                                  ) : null}
                                  {tuneIterateError ? (
                                    <p className="text-[10px] text-red-500">{tuneIterateError}</p>
                                  ) : null}
                                  <Button
                                    size="sm"
                                    className="h-8 shrink-0 text-xs"
                                    onClick={() => runTuneIterateMutation.mutate({ trigger: 'manual' })}
                                    disabled={runTuneIterateMutation.isPending || !selectedTrader}
                                  >
                                    {runTuneIterateMutation.isPending ? (
                                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                                    ) : (
                                      <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                                    )}
                                    {runTuneIterateMutation.isPending ? 'Running...' : 'Run Agent'}
                                  </Button>
                                </div>

                                {tuneIterateResponse && !runTuneIterateMutation.isPending ? (
                                  <div className="min-h-0 rounded-md border border-border/60 bg-background/70 p-2">
                                    <div className="flex h-full min-h-0 flex-col gap-1.5">
                                      <div className="flex items-center justify-between gap-2">
                                        <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Latest Agent Result</p>
                                        <Badge variant="outline" className="h-4 px-1 text-[9px] font-mono">
                                          {tuneIterateAppliedUpdates.length} updates
                                        </Badge>
                                      </div>
                                      <div className="space-y-1.5">
                                        <div className="space-y-1 rounded border border-border/50 bg-background/60 px-2 py-1.5">
                                          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Summary</p>
                                          {tuneIterateParsed && typeof tuneIterateParsed.summary === 'string' ? (
                                            <p className="line-clamp-3 text-[10px] text-foreground/90">
                                              {String(tuneIterateParsed.summary)}
                                            </p>
                                          ) : (
                                            <p className="text-[10px] text-muted-foreground/80">No summary returned by model.</p>
                                          )}
                                        </div>
                                        <div className="grid grid-cols-3 gap-1">
                                          <div className="rounded border border-border/50 bg-background/60 px-2 py-1">
                                            <p className="text-[9px] uppercase text-muted-foreground">Issues</p>
                                            <p className="text-[10px] font-mono">{tuneIterateIssues.length}</p>
                                          </div>
                                          <div className="rounded border border-border/50 bg-background/60 px-2 py-1">
                                            <p className="text-[9px] uppercase text-muted-foreground">Actions</p>
                                            <p className="text-[10px] font-mono">{tuneIterateActions.length}</p>
                                          </div>
                                          <div className="rounded border border-border/50 bg-background/60 px-2 py-1">
                                            <p className="text-[9px] uppercase text-muted-foreground">Next</p>
                                            <p className="text-[10px] font-mono">{tuneIterateNextSteps.length}</p>
                                          </div>
                                        </div>
                                        {tuneIterateAppliedUpdates.length > 0 ? (
                                          <p className="line-clamp-2 text-[10px] text-muted-foreground/90">
                                            Latest: {tuneIterateAppliedUpdates[0].source_key}:{tuneIterateAppliedUpdates[0].strategy_key}{' -> '}
                                            {tuneIterateAppliedUpdates[0].changed_keys.join(', ')}
                                          </p>
                                        ) : (
                                          <p className="text-[10px] text-muted-foreground/80">No parameter changes were applied.</p>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                ) : null}
                              </div>
                            </div>
                          </div>

                          <div className="min-h-0 rounded-md border border-border/60 bg-background/70 p-2.5">
                            <div className="flex h-full min-h-0 flex-col gap-2">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-1.5">
                                  <p className="text-[11px] font-medium">Parameter Workspace</p>
                                  <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                                    {dynamicStrategyParamSections.reduce((sum, section) => sum + section.fieldKeys.length, 0)} fields
                                  </Badge>
                                  {tuneDraftDirty ? (
                                    <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-amber-500">
                                      UNSAVED
                                    </span>
                                  ) : null}
                                </div>
                                <div className="flex flex-wrap items-center gap-1.5">
                                  <Button
                                    type="button"
                                    size="sm"
                                    variant="outline"
                                    className="h-6 px-2 text-[10px]"
                                    onClick={() => {
                                      if (!selectedTrader) return
                                      applyTraderDraftSettings(selectedTrader)
                                      setTuneDraftDirty(false)
                                      setTuneSaveError(null)
                                      setTuneRevertError(null)
                                    }}
                                  >
                                    Discard Edits
                                  </Button>
                                  <Button
                                    type="button"
                                    size="sm"
                                    variant="outline"
                                    className="h-6 px-2 text-[10px]"
                                    onClick={() => revertTuneParametersMutation.mutate()}
                                    disabled={
                                      revertTuneParametersMutation.isPending
                                      || !tuneRevertSnapshot
                                      || tuneRevertSnapshot.traderId !== selectedTrader.id
                                    }
                                  >
                                    {revertTuneParametersMutation.isPending ? (
                                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                    ) : null}
                                    Revert Last Applied
                                  </Button>
                                  <Button
                                    type="button"
                                    size="sm"
                                    className="h-6 px-2 text-[10px]"
                                    onClick={() => saveTuneParametersMutation.mutate()}
                                    disabled={saveTuneParametersMutation.isPending || !tuneDraftDirty}
                                  >
                                    {saveTuneParametersMutation.isPending ? (
                                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                    ) : null}
                                    Save Parameters
                                  </Button>
                                </div>
                              </div>
                              <p className="text-[10px] text-muted-foreground/80">
                                Review and edit grouped strategy controls before committing.
                              </p>
                              {tuneRevertSnapshot && tuneRevertSnapshot.traderId === selectedTrader.id ? (
                                <p className="text-[10px] text-muted-foreground/80">
                                  Revert snapshot captured at {formatTimestamp(tuneRevertSnapshot.capturedAt)}.
                                </p>
                              ) : null}
                              {tuneSaveError ? (
                                <p className="text-[10px] text-red-500">{tuneSaveError}</p>
                              ) : null}
                              {tuneRevertError ? (
                                <p className="text-[10px] text-red-500">{tuneRevertError}</p>
                              ) : null}
                              {dynamicStrategyParamSections.length === 0 ? (
                                <p className="text-[10px] text-muted-foreground/80">No dynamic parameter fields available for this bot.</p>
                              ) : (
                                <Tabs
                                  value={tuneParamSectionTab}
                                  onValueChange={setTuneParamSectionTab}
                                  className="flex min-h-0 flex-1 flex-col overflow-hidden"
                                >
                                  <div className="shrink-0 overflow-x-auto pb-1">
                                    <TabsList className="h-auto w-max min-w-full justify-start gap-1 rounded-md border border-border/50 bg-background/60 p-1">
                                      {dynamicStrategyParamSections.map((section) => (
                                        <TabsTrigger key={section.sectionKey} value={section.sectionKey} className="h-6 gap-1 px-2 text-[10px]">
                                          <span className="max-w-[220px] truncate">{section.sourceLabel} · {section.strategyLabel}</span>
                                          <span className="text-[9px] text-muted-foreground">{section.fieldKeys.length}</span>
                                        </TabsTrigger>
                                      ))}
                                    </TabsList>
                                  </div>

                                  {dynamicStrategyParamSections.map((section) => (
                                    <TabsContent key={section.sectionKey} value={section.sectionKey} className="mt-0 flex min-h-0 flex-1 flex-col overflow-hidden">
                                      {section.groups.length === 0 ? (
                                        <p className="text-[10px] text-muted-foreground/80">No grouped parameter fields are available for this strategy.</p>
                                      ) : (
                                        <Tabs defaultValue={section.groups[0].key} className="flex min-h-0 flex-1 flex-col overflow-hidden">
                                          <div className="shrink-0 overflow-x-auto pb-1">
                                            <TabsList className="h-auto w-max min-w-full justify-start gap-1 rounded-md border border-border/50 bg-background/50 p-1">
                                              {section.groups.map((group) => (
                                                <TabsTrigger key={`${section.sectionKey}:${group.key}`} value={group.key} className="h-6 gap-1 px-2 text-[10px]">
                                                  <span>{group.label}</span>
                                                  <span className="text-[9px] text-muted-foreground">{group.fields.length}</span>
                                                </TabsTrigger>
                                              ))}
                                            </TabsList>
                                          </div>

                                          {section.groups.map((group) => (
                                            <TabsContent
                                              key={`${section.sectionKey}:panel:${group.key}`}
                                              value={group.key}
                                              className="mt-0 min-h-0 flex-1 overflow-auto rounded-md border border-border/50 bg-background/65 p-2"
                                            >
                                              <StrategyConfigForm
                                                schema={{ param_fields: group.fields as any[] }}
                                                values={section.values}
                                                onChange={(nextValues) => applyDynamicStrategyFormValues(section.fieldKeys, nextValues)}
                                              />
                                            </TabsContent>
                                          ))}
                                        </Tabs>
                                      )}
                                    </TabsContent>
                                  ))}
                                </Tabs>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {workTab === 'decisions' && (
                  <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] px-1">
                    <div className="flex min-w-0 flex-col gap-1.5 min-h-0 overflow-hidden">
                      <Input value={decisionSearch} onChange={(event) => setDecisionSearch(event.target.value)} placeholder="Search decisions..." className="h-6 text-[11px] shrink-0" />
                      <div className="shrink-0 flex items-center justify-between gap-2">
                        <p className="text-[10px] text-muted-foreground">Showing {filteredDecisions.length}/{selectedDecisions.length}</p>
                        <Button
                          type="button"
                          size="sm"
                          variant={decisionOutcomeFilter === 'all' ? 'default' : 'outline'}
                          className="h-5 px-2 text-[10px]"
                          onClick={() => setDecisionOutcomeFilter('all')}
                        >
                          all
                        </Button>
                      </div>
                      <div className="shrink-0 grid gap-1 grid-cols-3">
                        <button
                          type="button"
                          onClick={() => setDecisionOutcomeFilter((current) => (current === 'selected' ? 'all' : 'selected'))}
                          className={cn(
                            'rounded border px-2 py-1 text-center transition-colors',
                            decisionOutcomeFilter === 'selected'
                              ? 'border-cyan-500/50 bg-cyan-500/10'
                              : 'border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10'
                          )}
                        >
                          <p className="text-[9px] uppercase text-muted-foreground">Selected</p>
                          <p className="text-xs font-mono text-emerald-500">{decisionOutcomeSummary.selected}</p>
                        </button>
                        <button
                          type="button"
                          onClick={() => setDecisionOutcomeFilter((current) => (current === 'blocked' ? 'all' : 'blocked'))}
                          className={cn(
                            'rounded border px-2 py-1 text-center transition-colors',
                            decisionOutcomeFilter === 'blocked'
                              ? 'border-cyan-500/50 bg-cyan-500/10'
                              : 'border-red-500/30 bg-red-500/5 hover:bg-red-500/10'
                          )}
                        >
                          <p className="text-[9px] uppercase text-muted-foreground">Blocked</p>
                          <p className="text-xs font-mono text-red-500">{decisionOutcomeSummary.blocked}</p>
                        </button>
                        <button
                          type="button"
                          onClick={() => setDecisionOutcomeFilter((current) => (current === 'skipped' ? 'all' : 'skipped'))}
                          className={cn(
                            'rounded border px-2 py-1 text-center transition-colors',
                            decisionOutcomeFilter === 'skipped'
                              ? 'border-cyan-500/50 bg-cyan-500/10'
                              : 'border-border/70 bg-background/70 hover:bg-muted/40'
                          )}
                        >
                          <p className="text-[9px] uppercase text-muted-foreground">Skipped</p>
                          <p className="text-xs font-mono">{decisionOutcomeSummary.skipped}</p>
                        </button>
                      </div>
                      <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10">
                        <div className="space-y-0.5 p-1.5 pr-2 text-xs">
                          {filteredDecisions.length === 0 ? (
                            <p className="py-4 text-center text-muted-foreground">No decisions.</p>
                          ) : (
                            filteredDecisions.map((decision) => {
                              const isActive = decision.id === selectedDecisionId
                              const outcome = normalizeDecisionOutcome(decision.decision)
                              const marketLabel = resolveDecisionMarketLabel(decision)
                              return (
                                <button
                                  key={decision.id}
                                  type="button"
                                  onClick={() => setSelectedDecisionId(decision.id)}
                                  className={cn(
                                    'w-full min-w-0 text-left rounded border px-2 py-1 transition-colors',
                                    isActive ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-border/50 hover:bg-muted/40',
                                    outcome === 'selected' && !isActive ? 'border-emerald-500/25' :
                                    outcome === 'blocked' && !isActive ? 'border-red-500/25' : ''
                                  )}
                                >
                                  <div className="flex min-w-0 items-center justify-between gap-2 font-mono">
                                    <span className="min-w-0 flex-1 truncate" title={marketLabel}>{marketLabel}</span>
                                    <Badge variant={outcome === 'selected' ? 'default' : outcome === 'blocked' ? 'destructive' : 'outline'} className="text-[9px] h-4 px-1 shrink-0">{outcome}</Badge>
                                  </div>
                                  <p className="min-w-0 text-[10px] text-muted-foreground truncate">{decision.reason || decision.strategy_key}</p>
                                </button>
                              )
                            })
                          )}
                        </div>
                      </ScrollArea>
                    </div>

                    <div className="flex min-w-0 flex-col gap-1.5 min-h-0 overflow-hidden">
                      {selectedDecision ? (
                        <>
                          <div className="shrink-0 rounded-md border border-border p-2 text-xs space-y-1">
                            <p className="font-medium">{resolveDecisionMarketLabel(selectedDecision)}</p>
                            <div className="grid gap-1 text-[11px] text-muted-foreground sm:grid-cols-2">
                              <span>Source: {selectedDecision.source}</span>
                              <span>Strategy: {selectedDecision.strategy_key}</span>
                              <span>Direction: {selectedDecisionDirection.label}</span>
                              <span>Price: {toNumber(selectedDecision.market_price).toFixed(3)}</span>
                              <span>Model: {toNumber(selectedDecision.model_probability).toFixed(3)}</span>
                              <span>Edge: {formatPercent(toNumber(selectedDecision.edge_percent))}</span>
                              <span>Confidence: {formatPercent(normalizeConfidencePercent(toNumber(selectedDecision.confidence)))}</span>
                              <span>Score: {toNumber(selectedDecision.signal_score).toFixed(3)}</span>
                            </div>
                            <p className="text-[10px]">Reason: {selectedDecision.reason || 'n/a'}</p>
                          </div>

                          {decisionDetailLoading ? (
                            <div className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10 p-2">
                              <div className="space-y-1.5 animate-pulse">
                                <div className="h-2.5 w-40 rounded bg-muted/60" />
                                {Array.from({ length: 6 }).map((_, index) => (
                                  <div key={`decision-check-skeleton-${index}`} className="rounded border border-border/40 bg-background/35 px-2 py-1.5">
                                    <div className="h-2.5 w-44 rounded bg-muted/55" />
                                    <div className="mt-1.5 h-2 w-[92%] rounded bg-muted/50" />
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : decisionChecks.length > 0 ? (
                            <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10">
                              <div className="space-y-1 p-2 text-xs">
                                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Checks ({decisionPassCount} pass / {decisionFailCount} fail)</p>
                                {decisionChecks.map((check, i) => (
                                  <div key={i} className={cn('rounded border px-2 py-1', check.passed ? 'border-emerald-500/25' : 'border-red-500/25')}>
                                    <div className="flex items-center gap-1">
                                      {check.passed ? <CheckCircle2 className="w-3 h-3 text-emerald-500" /> : <AlertTriangle className="w-3 h-3 text-red-500" />}
                                      <span className="font-medium">{check.check_label || check.check_name || check.check_key || 'Check'}</span>
                                    </div>
                                    {(check.detail || check.message) ? <p className="text-[10px] text-muted-foreground mt-0.5 pl-4">{check.detail || check.message}</p> : null}
                                  </div>
                                ))}
                                {riskChecks && riskChecks.length > 0 ? (
                                  <>
                                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground mt-2 mb-1">Risk Checks — {riskAllowed ? 'Allowed' : 'Blocked'}</p>
                                    {riskChecks.map((check: any, i: number) => (
                                      <div key={`risk-${i}`} className={cn('rounded border px-2 py-1', check.passed ? 'border-emerald-500/25' : 'border-red-500/25')}>
                                        <div className="flex items-center gap-1">
                                          {check.passed ? <CheckCircle2 className="w-3 h-3 text-emerald-500" /> : <AlertTriangle className="w-3 h-3 text-red-500" />}
                                          <span className="font-medium">{check.check_name || check.name}</span>
                                        </div>
                                        {check.message ? <p className="text-[10px] text-muted-foreground mt-0.5 pl-4">{check.message}</p> : null}
                                      </div>
                                    ))}
                                  </>
                                ) : null}
                                {decisionOrders.length > 0 ? (
                                  <>
                                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground mt-2 mb-1">Linked Orders ({decisionOrders.length})</p>
                                    {decisionOrders.map((order: any) => {
                                      const directionPresentation = resolveOrderDirectionPresentation(order as TraderOrder)
                                      return (
                                        <div key={order.id} className="rounded border border-border px-2 py-1 font-mono text-[10px]">
                                          {normalizeStatus(order.status).toUpperCase()} {'\u2022'} {formatCurrency(toNumber(order.notional_usd))} {'\u2022'} {directionPresentation.label}
                                        </div>
                                      )
                                    })}
                                  </>
                                ) : null}
                              </div>
                            </ScrollArea>
                          ) : (
                            <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">No checks data.</div>
                          )}
                        </>
                      ) : (
                        <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">Select a decision to view details.</div>
                      )}
                    </div>
                  </div>
                )}

                {workTab === 'performance' && (
                  <div className="h-full min-h-0 flex flex-col gap-2 px-1">
                    <div className="shrink-0 grid gap-1 sm:grid-cols-2 lg:grid-cols-6">
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Realized P&amp;L</p>
                        <p className={cn('text-xs font-mono', selectedPerformance.resolvedPnl > 0 ? 'text-emerald-500' : selectedPerformance.resolvedPnl < 0 ? 'text-red-500' : '')}>
                          {formatCurrency(selectedPerformance.resolvedPnl)}
                        </p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Resolved ROI</p>
                        <p className={cn('text-xs font-mono', selectedPerformance.roiPercent > 0 ? 'text-emerald-500' : selectedPerformance.roiPercent < 0 ? 'text-red-500' : '')}>
                          {selectedPerformance.roiPercent > 0 ? '+' : ''}{formatPercent(selectedPerformance.roiPercent, 2)}
                        </p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Resolved</p>
                        <p className="text-xs font-mono">{selectedPerformance.resolved}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Win / Loss</p>
                        <p className="text-xs font-mono">{selectedPerformance.wins} / {selectedPerformance.losses}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Open</p>
                        <p className="text-xs font-mono">{selectedPerformance.open}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Failed</p>
                        <p className="text-xs font-mono">{selectedPerformance.failed}</p>
                      </div>
                    </div>
                    <Tabs
                      value={performanceSubview}
                      onValueChange={(value) => setPerformanceSubview(value as PerformanceSubview)}
                      className="flex min-h-0 flex-1 flex-col overflow-hidden"
                    >
                      <div className="shrink-0 flex items-center justify-between gap-2 overflow-x-auto pb-1">
                        <TabsList className="h-auto justify-start gap-1 rounded-lg border border-border/60 bg-card/70 p-1">
                          <TabsTrigger value="latency" className="h-7 px-2.5 text-[11px]">Latency</TabsTrigger>
                          <TabsTrigger value="configuration" className="h-7 px-2.5 text-[11px]">Configuration</TabsTrigger>
                        </TabsList>
                        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                          <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5 font-mono">
                            {selectedPerformanceConfig.sections.length} configs
                          </span>
                          <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5 font-mono">
                            {selectedPerformanceConfig.snapshots.length} orders
                          </span>
                        </div>
                      </div>

                      {performanceSubview === 'latency' ? (
                        <TabsContent value="latency" className="mt-0 flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
                        {executionLatency ? (
                          <div className="shrink-0 grid gap-1 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-8">
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Internal SLA</p>
                              <p className="text-xs font-mono">{formatLatencyMs(executionLatencyTargetMs) || '—'}</p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Rolling Window</p>
                              <p className="text-xs font-mono">
                                {executionLatencyWindowLabel}{executionLatencySampleCount !== null ? ` · n=${executionLatencySampleCount}` : ''}
                              </p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Trader Release to Submit</p>
                              <p className={cn('text-xs font-mono', selectedTraderLatencySlaBreached ? 'text-amber-400' : 'text-cyan-400')}>
                                {selectedTraderLatencyLabel}
                              </p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Overall Release to Submit</p>
                              <p className={cn('text-xs font-mono', executionLatencySlaBreached ? 'text-amber-400' : 'text-cyan-400')}>
                                {executionLatencyOverallLabel}
                              </p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Trader Armed to Release</p>
                              <p className="text-xs font-mono">{selectedTraderArmedToReleaseLabel}</p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Trader Release to Decision</p>
                              <p className="text-xs font-mono">{selectedTraderReleaseToDecisionLabel}</p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Worst Source</p>
                              <p className="text-xs font-mono">{worstLatencySourceLabel}</p>
                            </div>
                            <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                              <p className="text-[9px] uppercase text-muted-foreground">Worst Strategy</p>
                              <p className="text-xs font-mono">{worstLatencyStrategyLabel}</p>
                            </div>
                          </div>
                        ) : (
                          <div className="shrink-0 rounded-md border border-border/60 bg-background/70 px-3 py-2 text-[11px] text-muted-foreground">
                            No execution latency samples are available yet for the current rolling window.
                          </div>
                        )}

                        {selectedPerformance.allowanceErrorCount > 0 ? (
                          <div className="shrink-0 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-700 dark:text-amber-100">
                            Found {selectedPerformance.allowanceErrorCount} orders with `not enough balance / allowance` in execution payloads.
                          </div>
                        ) : null}
                        {selectedPerformance.gasErrorCount > 0 ? (
                          <div className="shrink-0 rounded-md border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-[11px] text-orange-700 dark:text-orange-100">
                            Found {selectedPerformance.gasErrorCount} orders with `not enough gas` / native-token gas funding errors.
                          </div>
                        ) : null}

                        <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
                          <div className="min-h-0 rounded-md border border-border/60 bg-card/60 flex flex-col">
                            <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Stage Breakdown</div>
                            <div className="flex-1 min-h-0 overflow-auto">
                              <Table>
                                <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
                                  <TableRow>
                                    <TableHead className="text-[10px]">Stage</TableHead>
                                    <TableHead className="text-[10px] text-right">Trader</TableHead>
                                    <TableHead className="text-[10px] text-right">Overall</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {latencyStageRows.map((row) => (
                                    <TableRow key={`latency-stage-${row.key}`} className="text-xs">
                                      <TableCell className="py-1">{row.label}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{row.traderLatencyLabel}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{row.overallLatencyLabel}</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          </div>

                          <div className="min-h-0 grid gap-2 xl:grid-rows-2">
                            <div className="min-h-0 rounded-md border border-border/60 bg-card/60 flex flex-col">
                              <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Slowest Sources</div>
                              <div className="flex-1 min-h-0 overflow-auto">
                                <Table>
                                  <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
                                    <TableRow>
                                      <TableHead className="text-[10px]">Source</TableHead>
                                      <TableHead className="text-[10px] text-right">Release to Submit</TableHead>
                                      <TableHead className="text-[10px] text-right">Samples</TableHead>
                                    </TableRow>
                                  </TableHeader>
                                  <TableBody>
                                    {latencySourceRows.length === 0 ? (
                                      <TableRow>
                                        <TableCell colSpan={3} className="py-6 text-center text-[11px] text-muted-foreground">
                                          No per-source latency samples yet.
                                        </TableCell>
                                      </TableRow>
                                    ) : (
                                      latencySourceRows.map((row) => (
                                        <TableRow key={`latency-source-${row.key}`} className="text-xs">
                                          <TableCell className="py-1">{row.label}</TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.latencyLabel}</TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.count}</TableCell>
                                        </TableRow>
                                      ))
                                    )}
                                  </TableBody>
                                </Table>
                              </div>
                            </div>

                            <div className="min-h-0 rounded-md border border-border/60 bg-card/60 flex flex-col">
                              <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Slowest Strategies</div>
                              <div className="flex-1 min-h-0 overflow-auto">
                                <Table>
                                  <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
                                    <TableRow>
                                      <TableHead className="text-[10px]">Strategy</TableHead>
                                      <TableHead className="text-[10px] text-right">Release to Submit</TableHead>
                                      <TableHead className="text-[10px] text-right">Samples</TableHead>
                                    </TableRow>
                                  </TableHeader>
                                  <TableBody>
                                    {latencyStrategyRows.length === 0 ? (
                                      <TableRow>
                                        <TableCell colSpan={3} className="py-6 text-center text-[11px] text-muted-foreground">
                                          No per-strategy latency samples yet.
                                        </TableCell>
                                      </TableRow>
                                    ) : (
                                      latencyStrategyRows.map((row) => (
                                        <TableRow key={`latency-strategy-${row.key}`} className="text-xs">
                                          <TableCell className="py-1">{row.label}</TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.latencyLabel}</TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.count}</TableCell>
                                        </TableRow>
                                      ))
                                    )}
                                  </TableBody>
                                </Table>
                              </div>
                            </div>
                          </div>
                        </div>
                        </TabsContent>
                      ) : (
                        <TabsContent value="configuration" className="mt-0 flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
                        <div className="shrink-0 flex flex-wrap items-end gap-2">
                          <div className="min-w-[220px] max-w-[320px] flex-1">
                            <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">Strategy Section</Label>
                            {selectedPerformanceConfig.sections.length > 1 ? (
                              <Select value={activePerformanceSection?.sectionKey || ''} onValueChange={setPerformanceSectionKey}>
                                <SelectTrigger className="mt-1 h-8 text-[11px]">
                                  <SelectValue placeholder="Select configuration" />
                                </SelectTrigger>
                                <SelectContent>
                                  {selectedPerformanceConfig.sections.map((section) => (
                                    <SelectItem key={section.sectionKey} value={section.sectionKey}>
                                      {section.sectionLabel}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            ) : (
                              <div className="mt-1 rounded-md border border-border/60 bg-background/70 px-2 py-1.5 text-[11px]">
                                {activePerformanceSection?.sectionLabel || 'No configured strategy sections'}
                              </div>
                            )}
                          </div>
                          <div className="min-w-[220px] max-w-[320px] flex-1">
                            <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">Parameter</Label>
                            {activePerformanceParamSummaryRows.length > 0 ? (
                              <Select value={performanceParamKey} onValueChange={setPerformanceParamKey}>
                                <SelectTrigger className="mt-1 h-8 text-[11px]">
                                  <SelectValue placeholder="Select parameter" />
                                </SelectTrigger>
                                <SelectContent>
                                  {activePerformanceParamSummaryRows.map((row) => (
                                    <SelectItem key={row.key} value={row.key}>
                                      {row.label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            ) : (
                              <div className="mt-1 rounded-md border border-border/60 bg-background/70 px-2 py-1.5 text-[11px] text-muted-foreground">
                                No parameter fields recorded for this strategy.
                              </div>
                            )}
                          </div>
                        </div>

                        {selectedPerformanceConfig.fallbackOrderCount > 0 ? (
                          <div className="shrink-0 rounded-md border border-blue-500/25 bg-blue-500/10 px-2 py-1 text-[11px] text-blue-700 dark:text-blue-100">
                            {selectedPerformanceConfig.fallbackOrderCount} historical orders are using the trader&apos;s current config as a fallback because those order rows did not persist `strategy_params`.
                          </div>
                        ) : null}

                        <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
                          <div className="min-h-0 rounded-md border border-border/60 bg-card/60 flex flex-col">
                            <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Observed Configurations</div>
                            <div className="flex-1 min-h-0 overflow-auto">
                              <Table>
                                <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
                                  <TableRow>
                                    <TableHead className="text-[10px]">Config</TableHead>
                                    <TableHead className="text-[10px] text-right">Orders</TableHead>
                                    <TableHead className="text-[10px] text-right">Resolved</TableHead>
                                    <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                    <TableHead className="text-[10px] text-right">ROI</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {selectedPerformanceConfig.configurationRows.length === 0 ? (
                                    <TableRow>
                                      <TableCell colSpan={5} className="py-6 text-center text-[11px] text-muted-foreground">
                                        No configuration-linked orders yet.
                                      </TableCell>
                                    </TableRow>
                                  ) : (
                                    selectedPerformanceConfig.configurationRows.map((row) => (
                                      <TableRow
                                        key={`config-row-${row.sectionKey}`}
                                        onClick={() => setPerformanceSectionKey(row.sectionKey)}
                                        className={cn(
                                          'text-xs cursor-pointer',
                                          activePerformanceSection?.sectionKey === row.sectionKey
                                            ? 'bg-cyan-500/5'
                                            : 'hover:bg-muted/30'
                                        )}
                                      >
                                        <TableCell className="py-1">
                                          <div className="font-medium">{row.strategyLabel}</div>
                                          <div className="text-[9px] text-muted-foreground">{row.sourceLabel} · {row.strategyVersionLabel}</div>
                                        </TableCell>
                                        <TableCell className="text-right font-mono py-1">{row.orders}</TableCell>
                                        <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                        <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>
                                          {formatCurrency(row.pnl)}
                                        </TableCell>
                                        <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                          {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                        </TableCell>
                                      </TableRow>
                                    ))
                                  )}
                                </TableBody>
                              </Table>
                            </div>
                          </div>

                          <div className="min-h-0 flex flex-col gap-2">
                            <div className="min-h-0 rounded-md border border-border/60 bg-card/60 flex flex-col">
                              <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Parameter Performance</div>
                              <div className="shrink-0 grid gap-1 border-b border-border/40 px-2 py-1 sm:grid-cols-3">
                                <div>
                                  <p className="text-[9px] uppercase text-muted-foreground">Current Value</p>
                                  <p className="text-[11px] font-mono">
                                    {activePerformanceParamSummaryByKey.get(performanceParamKey)?.currentValueLabel || '—'}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-[9px] uppercase text-muted-foreground">Observed Buckets</p>
                                  <p className="text-[11px] font-mono">
                                    {activePerformanceParamSummaryByKey.get(performanceParamKey)?.observedValueCount || 0}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-[9px] uppercase text-muted-foreground">Resolved At Current</p>
                                  <p className="text-[11px] font-mono">
                                    {activePerformanceParamSummaryByKey.get(performanceParamKey)?.currentResolved || 0}
                                  </p>
                                </div>
                              </div>
                              <div className="flex-1 min-h-[170px] overflow-auto">
                                <Table>
                                  <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
                                    <TableRow>
                                      <TableHead className="text-[10px]">Value</TableHead>
                                      <TableHead className="text-[10px] text-right">Orders</TableHead>
                                      <TableHead className="text-[10px] text-right">Resolved</TableHead>
                                      <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                      <TableHead className="text-[10px] text-right">ROI</TableHead>
                                      <TableHead className="text-[10px] text-right">W/L</TableHead>
                                    </TableRow>
                                  </TableHeader>
                                  <TableBody>
                                    {activePerformanceParamRows.length === 0 ? (
                                      <TableRow>
                                        <TableCell colSpan={6} className="py-6 text-center text-[11px] text-muted-foreground">
                                          Select a parameter with recorded values to compare its buckets.
                                        </TableCell>
                                      </TableRow>
                                    ) : (
                                      activePerformanceParamRows.map((row) => (
                                        <TableRow key={`param-row-${performanceParamKey}-${row.key}`} className={cn('text-xs', row.isCurrent && 'bg-cyan-500/5')}>
                                          <TableCell className="py-1">
                                            <div className="font-medium">{row.valueLabel}</div>
                                            <div className="text-[9px] text-muted-foreground">{row.isCurrent ? 'current value' : row.isMissing ? 'missing from snapshot' : 'historical bucket'}</div>
                                          </TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.orders}</TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>
                                            {formatCurrency(row.pnl)}
                                          </TableCell>
                                          <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                            {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                          </TableCell>
                                          <TableCell className="text-right font-mono py-1">{row.wins}/{row.losses}</TableCell>
                                        </TableRow>
                                      ))
                                    )}
                                  </TableBody>
                                </Table>
                              </div>
                              <div className="px-2 py-1 border-y border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Observed Runtime Context</div>
                              <div className="flex-1 min-h-0 grid gap-2 p-2 xl:grid-cols-2">
                                <div className="min-h-0 rounded-md border border-border/50 bg-background/50 flex flex-col">
                                  <div className="px-2 py-1 border-b border-border/40 text-[10px] uppercase tracking-wider text-muted-foreground">Timeframe / Mode</div>
                                  <div className="flex-1 min-h-0 overflow-auto">
                                    <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">Timeframe</div>
                                    <Table>
                                      <TableBody>
                                        {selectedPerformance.timeframeRows.map((row) => (
                                          <TableRow key={`tf-runtime-${row.key}`} className="text-xs">
                                            <TableCell className="font-mono py-1">{row.label}</TableCell>
                                            <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                            <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                    <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">Mode</div>
                                    <Table>
                                      <TableBody>
                                        {selectedPerformance.modeRows.map((row) => (
                                          <TableRow key={`mode-runtime-${row.key}`} className="text-xs">
                                            <TableCell className="font-mono py-1">{row.label}</TableCell>
                                            <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                            <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                    <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">Mode + Timeframe</div>
                                    <Table>
                                      <TableBody>
                                        {selectedPerformance.timeframeModeRows.slice(0, 16).map((row) => (
                                          <TableRow key={`combo-runtime-${row.key}`} className="text-xs">
                                            <TableCell className="font-mono py-1">{row.label}</TableCell>
                                            <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                            <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                  </div>
                                </div>

                                <div className="min-h-0 rounded-md border border-border/50 bg-background/50 flex flex-col">
                                  <div className="px-2 py-1 border-b border-border/40 text-[10px] uppercase tracking-wider text-muted-foreground">Source / Strategy / Variant</div>
                                  <div className="flex-1 min-h-0 overflow-auto">
                                    <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">Source</div>
                                    <Table>
                                      <TableBody>
                                        {selectedPerformance.sourceRows.map((row) => (
                                          <TableRow key={`source-runtime-${row.key}`} className="text-xs">
                                            <TableCell className="py-1">{row.label}</TableCell>
                                            <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                            <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                    <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">Strategy</div>
                                    <Table>
                                      <TableBody>
                                        {selectedPerformance.strategyRows.map((row) => (
                                          <TableRow key={`strategy-runtime-${row.key}`} className="text-xs">
                                            <TableCell className="py-1">
                                              <div className="font-medium">{row.label}</div>
                                              <div className="text-[9px] font-mono text-muted-foreground">{row.key}</div>
                                            </TableCell>
                                            <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                            <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                    <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">Sub-Strategy</div>
                                    <Table>
                                      <TableBody>
                                        {selectedPerformance.subStrategyRows.slice(0, 16).map((row) => (
                                          <TableRow key={`sub-runtime-${row.key}`} className="text-xs">
                                            <TableCell className="font-mono py-1">{row.label}</TableCell>
                                            <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                            <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        </TabsContent>
                      )}
                    </Tabs>
                  </div>
                )}

              </div>
            </>
          )}
        </div>
	      </div>

	      {createPortal(
	        <AnimatePresence>
	          {marketModalState && (
	            <motion.div
	              key="trading-market-modal"
	              initial={{ opacity: 0 }}
	              animate={{ opacity: 1 }}
	              exit={{ opacity: 0 }}
	              transition={{ duration: 0.18 }}
	              className="fixed inset-0 z-[120] flex items-center justify-center"
	              onClick={closeMarketModal}
	            >
	              <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
	              <motion.div
	                initial={{ scale: 0.94, y: 22, opacity: 0 }}
	                animate={{ scale: 1, y: 0, opacity: 1 }}
	                exit={{ scale: 0.94, y: 22, opacity: 0 }}
	                transition={{ type: 'spring', damping: 28, stiffness: 340, mass: 0.85 }}
	                className="relative w-[96vw] max-w-[1180px] max-h-[92vh]"
	                onClick={(event) => event.stopPropagation()}
	              >
	                <BotTradePositionModal
	                  market={marketModalMarket}
	                  sharedHistory={modalSharedHistory}
	                  sharedHistoryLoading={marketHistoryQuery.isFetching}
	                  scope={marketModalState.scope}
	                  orders={allOrders}
	                  themeMode={themeMode}
	                  onSell={handleSellModalOrder}
	                  sellPendingOrderId={sellTradeNowMutation.isPending ? String(sellTradeNowMutation.variables?.orderId || '') : null}
	                  sellError={marketModalSellError}
	                  sellSuccess={marketModalSellSuccess}
	                  onClose={closeMarketModal}
	                />
	              </motion.div>
	            </motion.div>
	          )}
	        </AnimatePresence>,
	        document.body
	      )}

	      <Dialog open={confirmLiveStartOpen} onOpenChange={setConfirmLiveStartOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Live Trading Start</DialogTitle>
            <DialogDescription>
              This will start the orchestrator in LIVE mode against your globally selected live account.
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-100">
            Live trading can place real orders. Confirm only if preflight checks and risk controls are intentionally set.
          </div>
          <div className="grid gap-1 rounded-md border border-border p-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Account mode</span>
              <span className="font-mono">LIVE</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Block new orders</span>
              <span
                className={cn(
                  'font-mono',
                  killSwitchMutation.isPending
                    ? 'text-amber-500'
                    : killSwitchOn
                      ? 'text-red-500'
                      : 'text-emerald-600'
                )}
              >
                {killSwitchMutation.isPending ? 'UPDATING' : killSwitchOn ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmLiveStartOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmLiveStart}
              disabled={startBySelectedAccountMutation.isPending || killSwitchOn || !selectedAccountIsLive}
            >
              {startBySelectedAccountMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Confirm Start Live
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={confirmTraderStartOpen} onOpenChange={setConfirmTraderStartOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Start Trader</DialogTitle>
            <DialogDescription>
              Start this active trader and optionally seed copy signals from currently open source-wallet positions.
            </DialogDescription>
          </DialogHeader>
          {selectedTraderHasCopySource ? (
            <div className="rounded-md border border-border p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <p className="text-xs font-medium">Copy existing open positions on start</p>
                  <p className="text-[11px] text-muted-foreground">
                    Generates startup copy signals for current source-wallet positions in configured scope.
                  </p>
                </div>
                <Switch
                  checked={enableCopyExistingPositions}
                  onCheckedChange={setEnableCopyExistingPositions}
                />
              </div>
              <p className="text-[11px] text-muted-foreground">
                Strategy default: {selectedTraderCopyExistingOnStartDefault ? 'enabled' : 'disabled'}.
              </p>
            </div>
          ) : null}
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmTraderStartOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={confirmStartTrader}
              disabled={traderStartMutation.isPending || !selectedTrader}
            >
              {traderStartMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Start Trader
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={confirmTraderStopOpen} onOpenChange={setConfirmTraderStopOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Stop Trader</DialogTitle>
            <DialogDescription>
              Stop this trader and choose how existing positions/orders should be handled.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Stop lifecycle</Label>
              <Select
                value={stopLifecycleMode}
                onValueChange={(value) => {
                  setStopLifecycleMode(value as TraderStopLifecycleMode)
                  setStopConfirmLiveClose(false)
                }}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="keep_positions">Keep existing positions</SelectItem>
                  <SelectItem value="close_shadow_positions">Close shadow positions</SelectItem>
                  <SelectItem value="close_all_positions">Close live + shadow positions</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {stopLifecycleNeedsLiveConfirm ? (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="text-xs font-medium text-amber-700 dark:text-amber-100">Confirm live close</p>
                    <p className="text-[11px] text-amber-700/90 dark:text-amber-100/90">
                      This action can close live positions and cancel live open orders.
                    </p>
                  </div>
                  <Switch checked={stopConfirmLiveClose} onCheckedChange={setStopConfirmLiveClose} />
                </div>
              </div>
            ) : null}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmTraderStopOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmStopTrader}
              disabled={
                traderStopMutation.isPending
                || !selectedTrader
                || (stopLifecycleNeedsLiveConfirm && !stopConfirmLiveClose)
              }
            >
              {traderStopMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Stop Trader
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Sheet
        open={globalSettingsFlyoutOpen}
        onOpenChange={(open) => {
          setGlobalSettingsFlyoutOpen(open)
          if (!open) {
            setGlobalSettingsSaveError(null)
          }
        }}
      >
        <SheetContent side="right" className="w-full sm:max-w-xl p-0">
          <div className="h-full min-h-0 flex flex-col">
            <div className="border-b border-border px-4 py-3">
              <SheetHeader className="space-y-1 text-left">
                <SheetTitle className="text-base">Global Settings</SheetTitle>
                <SheetDescription>
                  Configure orchestrator-wide live/shadow runtime controls, risk clamps, and pending-exit behavior.
                </SheetDescription>
              </SheetHeader>
            </div>

            <ScrollArea className="flex-1 min-h-0 px-4 py-3">
              <div className="space-y-3 pb-2">
                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Loop</p>
                  <div>
                    <Label>Run Interval (seconds)</Label>
                    <Input
                      type="number"
                      min={1}
                      max={300}
                      value={globalSettingsDraft.runIntervalSeconds}
                      onChange={(event) => setGlobalSettingsField('runIntervalSeconds', event.target.value)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>Trader Cycle Timeout (seconds, blank = auto)</Label>
                    <Input
                      type="number"
                      min={3}
                      max={120}
                      placeholder="auto"
                      value={globalSettingsDraft.traderCycleTimeoutSeconds}
                      onChange={(event) => setGlobalSettingsField('traderCycleTimeoutSeconds', event.target.value)}
                      className="mt-1"
                    />
                  </div>
                </div>

                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Global Risk</p>
                  <div className="grid gap-2 sm:grid-cols-3">
                    <div>
                      <Label>Max Gross Exposure (USD)</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxGrossExposureUsd}
                        onChange={(event) => setGlobalSettingsField('maxGrossExposureUsd', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Daily Loss (USD)</Label>
                      <Input
                        type="number"
                        min={0}
                        value={globalSettingsDraft.maxDailyLossUsd}
                        onChange={(event) => setGlobalSettingsField('maxDailyLossUsd', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Orders / Cycle</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxOrdersPerCycle}
                        onChange={(event) => setGlobalSettingsField('maxOrdersPerCycle', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live Execution Limits</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div>
                      <Label>Max Trade Size (USD)</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxTradeSizeUsd}
                        onChange={(event) => setGlobalSettingsField('maxTradeSizeUsd', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Daily Trade Volume (USD)</Label>
                      <Input
                        type="number"
                        min={10}
                        value={globalSettingsDraft.maxDailyTradeVolumeUsd}
                        onChange={(event) => setGlobalSettingsField('maxDailyTradeVolumeUsd', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Minimum Account Balance (USD)</Label>
                      <Input
                        type="number"
                        min={0}
                        value={globalSettingsDraft.minAccountBalanceUsd}
                        onChange={(event) => setGlobalSettingsField('minAccountBalanceUsd', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Open Positions</Label>
                      <Input
                        type="number"
                        min={1}
                        max={100}
                        value={globalSettingsDraft.maxOpenPositions}
                        onChange={(event) => setGlobalSettingsField('maxOpenPositions', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Slippage (%)</Label>
                      <Input
                        type="number"
                        min={0.1}
                        max={10}
                        step={0.1}
                        value={globalSettingsDraft.maxSlippagePercent}
                        onChange={(event) => setGlobalSettingsField('maxSlippagePercent', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Pending Live Exits</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div>
                      <Label>Max Pending Exits Allowed</Label>
                      <Input
                        type="number"
                        min={0}
                        value={globalSettingsDraft.pendingExitMaxAllowed}
                        onChange={(event) => setGlobalSettingsField('pendingExitMaxAllowed', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <label className="rounded-md border border-border/60 bg-muted/15 px-2.5 py-2 flex items-center justify-between gap-2">
                      <span className="text-xs text-muted-foreground">Identity Guard Enabled</span>
                      <Switch
                        checked={globalSettingsDraft.pendingExitIdentityGuardEnabled}
                        onCheckedChange={(checked) => setGlobalSettingsField('pendingExitIdentityGuardEnabled', checked)}
                      />
                    </label>
                  </div>
                  <div>
                    <Label>Terminal Pending-Exit Statuses (comma-separated)</Label>
                    <Input
                      value={globalSettingsDraft.pendingExitTerminalStatuses}
                      onChange={(event) => setGlobalSettingsField('pendingExitTerminalStatuses', event.target.value)}
                      className="mt-1 font-mono text-xs"
                    />
                  </div>
                </div>

                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live Risk Clamps</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <label className="rounded-md border border-border/60 bg-muted/15 px-2.5 py-2 flex items-center justify-between gap-2">
                      <span className="text-xs text-muted-foreground">Force Averaging Off</span>
                      <Switch
                        checked={globalSettingsDraft.enforceAllowAveragingOff}
                        onCheckedChange={(checked) => setGlobalSettingsField('enforceAllowAveragingOff', checked)}
                      />
                    </label>
                    <label className="rounded-md border border-border/60 bg-muted/15 px-2.5 py-2 flex items-center justify-between gap-2">
                      <span className="text-xs text-muted-foreground">Force Halt on Loss Streak</span>
                      <Switch
                        checked={globalSettingsDraft.enforceHaltOnConsecutiveLosses}
                        onCheckedChange={(checked) => setGlobalSettingsField('enforceHaltOnConsecutiveLosses', checked)}
                      />
                    </label>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div>
                      <Label>Min Cooldown (seconds)</Label>
                      <Input
                        type="number"
                        min={0}
                        value={globalSettingsDraft.minCooldownSeconds}
                        onChange={(event) => setGlobalSettingsField('minCooldownSeconds', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Consecutive Losses Cap</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxConsecutiveLossesCap}
                        onChange={(event) => setGlobalSettingsField('maxConsecutiveLossesCap', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Open Orders Cap</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxOpenOrdersCap}
                        onChange={(event) => setGlobalSettingsField('maxOpenOrdersCap', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Trade Notional Cap (USD)</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxTradeNotionalUsdCap}
                        onChange={(event) => setGlobalSettingsField('maxTradeNotionalUsdCap', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max Orders / Cycle Cap</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxOrdersPerCycleCap}
                        onChange={(event) => setGlobalSettingsField('maxOrdersPerCycleCap', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live Market Context</p>
                  <label className="rounded-md border border-border/60 bg-muted/15 px-2.5 py-2 flex items-center justify-between gap-2">
                    <span className="text-xs text-muted-foreground">Enabled</span>
                    <Switch
                      checked={globalSettingsDraft.liveMarketContextEnabled}
                      onCheckedChange={(checked) => setGlobalSettingsField('liveMarketContextEnabled', checked)}
                    />
                  </label>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div>
                      <Label>History Window (seconds)</Label>
                      <Input
                        type="number"
                        min={300}
                        value={globalSettingsDraft.liveMarketHistoryWindowSeconds}
                        onChange={(event) => setGlobalSettingsField('liveMarketHistoryWindowSeconds', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>History Fidelity (seconds)</Label>
                      <Input
                        type="number"
                        min={30}
                        value={globalSettingsDraft.liveMarketHistoryFidelitySeconds}
                        onChange={(event) => setGlobalSettingsField('liveMarketHistoryFidelitySeconds', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Max History Points</Label>
                      <Input
                        type="number"
                        min={20}
                        value={globalSettingsDraft.liveMarketHistoryMaxPoints}
                        onChange={(event) => setGlobalSettingsField('liveMarketHistoryMaxPoints', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Context Timeout (seconds)</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.liveMarketContextTimeoutSeconds}
                        onChange={(event) => setGlobalSettingsField('liveMarketContextTimeoutSeconds', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-border p-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live Provider Health Guard</p>
                  <div className="grid gap-2 sm:grid-cols-3">
                    <div>
                      <Label>Window (seconds)</Label>
                      <Input
                        type="number"
                        min={30}
                        value={globalSettingsDraft.liveProviderHealthWindowSeconds}
                        onChange={(event) => setGlobalSettingsField('liveProviderHealthWindowSeconds', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Min Errors</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.liveProviderHealthMinErrors}
                        onChange={(event) => setGlobalSettingsField('liveProviderHealthMinErrors', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Block (seconds)</Label>
                      <Input
                        type="number"
                        min={15}
                        value={globalSettingsDraft.liveProviderHealthBlockSeconds}
                        onChange={(event) => setGlobalSettingsField('liveProviderHealthBlockSeconds', event.target.value)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </ScrollArea>

            <div className="border-t border-border px-4 py-3 flex flex-wrap items-center justify-end gap-2">
              {globalSettingsSaveError ? (
                <div className="mr-auto text-xs text-red-500 max-w-[65%] break-words leading-tight" title={globalSettingsSaveError}>
                  {globalSettingsSaveError}
                </div>
              ) : null}
              <Button
                type="button"
                variant="outline"
                onClick={resetGlobalSettingsDraft}
                disabled={globalSettingsBusy}
              >
                Reset
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => setGlobalSettingsFlyoutOpen(false)}
                disabled={globalSettingsBusy}
              >
                Close
              </Button>
              <Button
                type="button"
                onClick={saveGlobalSettings}
                disabled={globalSettingsBusy}
              >
                {globalSettingsBusy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                Save Settings
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <Sheet
        open={traderFlyoutOpen}
        onOpenChange={(open) => {
          setTraderFlyoutOpen(open)
          if (!open) {
            setSaveError(null)
            setDeleteForceConfirm(false)
            setTuneSaveError(null)
            setTuneIterateError(null)
            setTuneRevertError(null)
          }
        }}
      >
        <SheetContent side="right" className="w-full sm:max-w-3xl p-0">
          <div className="h-full min-h-0 flex flex-col">
            <div className="border-b border-border px-4 py-3">
              <SheetHeader className="space-y-1 text-left">
                <SheetTitle className="text-base">
                  {traderFlyoutMode === 'create' ? 'Create Auto Bot' : 'Edit Auto Bot'}
                </SheetTitle>
                <SheetDescription>
                  {traderFlyoutMode === 'create'
                    ? 'Configure a new bot profile with explicit strategy, source, risk, and schedule controls.'
                    : 'Update strategy, source, risk, and schedule settings for this bot.'}
                </SheetDescription>
              </SheetHeader>
            </div>

            <ScrollArea className="flex-1 min-h-0 px-4 py-3">
              <div className="space-y-3 pb-2">
                <FlyoutSection
                  title="Bot Profile"
                  icon={Sparkles}
                  subtitle="Name this bot and configure source-specific execution strategies below."
                >
                  <div className="rounded-md border border-border/60 bg-muted/15 px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Bot Mode</p>
                      <Badge className="h-5 px-1.5 text-[10px]" variant={draftMode === 'live' ? 'destructive' : 'outline'}>
                        {draftMode.toUpperCase()}
                      </Badge>
                    </div>
                    <p className="mt-1 text-[10px] text-muted-foreground/75">
                      This bot is scoped to {draftMode === 'live' ? 'live' : 'sandbox'} execution only.
                    </p>
                  </div>

                  <div>
                    <Label>Name</Label>
                    <Input value={draftName} onChange={(event) => setDraftName(event.target.value)} className="mt-1" />
                  </div>

                  <div>
                    <Label>Description</Label>
                    <Input value={draftDescription} onChange={(event) => setDraftDescription(event.target.value)} className="mt-1" />
                  </div>

                  {traderFlyoutMode === 'create' ? (
                    <div>
                      <Label>Copy Settings From Existing Bot (Optional)</Label>
                      <div className="mt-1 flex items-center gap-1">
                        <Button
                          type="button"
                          size="sm"
                          variant={draftCopyFromMode === 'shadow' ? 'default' : 'outline'}
                          className="h-6 px-2 text-[10px]"
                          onClick={() => setDraftCopyFromMode('shadow')}
                        >
                          Sandbox Bots
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant={draftCopyFromMode === 'live' ? 'default' : 'outline'}
                          className="h-6 px-2 text-[10px]"
                          onClick={() => setDraftCopyFromMode('live')}
                        >
                          Live Bots
                        </Button>
                      </div>
                      <Select
                        value={draftCopyFromTraderId || '__none__'}
                        onValueChange={applyCreateCopyFromSelection}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Start from scratch" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__none__">Start from scratch</SelectItem>
                          {copySourceTraders.map((trader) => (
                            <SelectItem key={trader.id} value={trader.id}>
                              {trader.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {copySourceTraders.length === 0 ? (
                        <p className="mt-1 text-[10px] text-muted-foreground/75 leading-tight">
                          No {draftCopyFromMode === 'live' ? 'live' : 'sandbox'} bots available to copy from.
                        </p>
                      ) : null}
                      <p className="mt-1 text-[10px] text-muted-foreground/75 leading-tight">
                        Creates a new bot with copied source config, strategy params, interval, risk limits, and schedule settings
                        from the selected bot. Trades, decisions, orders, and events are never copied.
                      </p>
                    </div>
                  ) : null}
                </FlyoutSection>

                <FlyoutSection
                  title="Signal Sources"
                  icon={Zap}
                  count={`${selectedSourceCount}/${sourceCards.length || sourceCatalog.length} enabled`}
                  subtitle="Toggle signal sources this bot should consume."
                >
                  <div className="flex flex-wrap items-center gap-1.5 mb-2">
                    <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={enableAllSourceCards}>
                      Enable all
                    </Button>
                    <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={disableAllSourceCards}>
                      Disable all
                    </Button>
                  </div>

                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {sourceCards.map((source) => {
                      const sourceKey = normalizeSourceKey(source.key)
                      const isEnabled = selectedSourceKeySet.has(sourceKey)
                      const strategyOptions = sourceStrategyOptionsByKey[sourceKey] || []
                      const selectedStrategy = effectiveSourceStrategies[sourceKey] || defaultStrategyForSource(sourceKey, sourceCards)
                      const strategyDetail = sourceStrategyDetailsLookup[sourceKey]?.[selectedStrategy] || null
                      const latestVersion = strategyDetail?.latestVersion ?? strategyDetail?.version ?? null
                      const selectedVersion = effectiveSourceStrategyVersions[sourceKey]
                      const selectedVersionToken = selectedVersion == null ? 'latest' : `v${selectedVersion}`
                      const availableVersions = (() => {
                        const rows = normalizeVersionList(strategyDetail?.versions || [])
                        if (latestVersion != null && !rows.includes(latestVersion)) {
                          rows.unshift(latestVersion)
                        }
                        if (selectedVersion != null && !rows.includes(selectedVersion)) {
                          rows.unshift(selectedVersion)
                        }
                        return rows
                      })()
                      return (
                        <div
                          key={source.key}
                          className={cn(
                            'rounded-lg border px-2.5 py-2 text-left transition-colors',
                            isEnabled
                              ? 'border-emerald-500/40 bg-emerald-500/10'
                              : 'border-border/70 bg-background hover:border-emerald-500/30 hover:bg-muted/40'
                          )}
                        >
                          <button
                            type="button"
                            onClick={() => toggleDraftSource(source.key)}
                            className="w-full text-left"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <p className="text-xs font-medium leading-tight">{source.label}</p>
                              <span
                                className={cn(
                                  'rounded-full px-1.5 py-0.5 text-[9px] font-semibold',
                                  isEnabled ? 'bg-emerald-500/20 text-emerald-600' : 'bg-muted text-muted-foreground'
                                )}
                              >
                                {isEnabled ? 'ON' : 'OFF'}
                              </span>
                            </div>
                            <p className="mt-1 text-[10px] leading-tight text-muted-foreground/75">
                              {source.description}
                            </p>
                          </button>
                          {isEnabled && strategyOptions.length > 0 ? (
                            <div className="mt-2">
                              <Label className="text-[10px] text-muted-foreground uppercase tracking-wide">Strategy</Label>
                              <Select
                                value={selectedStrategy}
                                onValueChange={(value) => setSourceStrategy(sourceKey, value)}
                              >
                                <SelectTrigger className="mt-1 h-7 text-xs">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {strategyOptions.map((option) => (
                                    <SelectItem key={option.key} value={option.key}>
                                      {option.label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              <div className="mt-1.5 flex min-w-0 items-center gap-1.5">
                                <Badge
                                  variant="outline"
                                  className="h-4 min-w-0 flex-1 truncate px-1.5 text-[9px] font-mono border-emerald-500/30 text-emerald-300 bg-emerald-500/10"
                                >
                                  {latestVersion != null ? `Latest v${latestVersion}` : 'Latest'}
                                </Badge>
                                <Select
                                  value={selectedVersionToken}
                                  onValueChange={(value) => setSourceStrategyVersion(sourceKey, value)}
                                >
                                  <SelectTrigger className="h-7 w-[142px] shrink-0 text-[10px] font-mono">
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="latest">
                                      {latestVersion != null ? `latest (v${latestVersion})` : 'latest'}
                                    </SelectItem>
                                    {availableVersions.map((version) => (
                                      <SelectItem key={`${sourceKey}-${selectedStrategy}-v${version}`} value={`v${version}`}>
                                        {`v${version}`}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      )
                    })}
                  </div>

                </FlyoutSection>

                <FlyoutSection
                  title="Fallback Interval"
                  icon={Clock3}
                  iconClassName="text-sky-500"
                  count={`${Number(draftInterval || 0)}s`}
                  subtitle="Idle-sweep fallback cadence. Real-time signal events can trigger this bot sooner."
                >
                  <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                    <div>
                      <Label>Fallback Interval Seconds</Label>
                      <Input
                        type="number"
                        min={1}
                        value={draftInterval}
                        onChange={(event) => setDraftInterval(event.target.value)}
                        className="mt-1"
                      />
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        Saves to this bot&apos;s <span className="font-mono">interval_seconds</span>.
                      </p>
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        Real-time trade-signal events wake matching bots immediately. If no new events arrive, this bot still runs when this fallback interval elapses.
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/15 px-3 py-2">
                      <p className="text-[11px] font-medium">Global Orchestrator Runtime</p>
                      <p className="mt-2 text-[11px] font-medium">Execution Mode</p>
                      <p className="mt-1 text-sm font-mono">{effectiveGlobalLoopLabel}</p>
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        {effectiveGlobalLoopDetail}
                      </p>
                      {highFrequencyCryptoLoopActive ? (
                        <p className="mt-1 text-[10px] text-emerald-600 dark:text-emerald-300">
                          High-frequency mode active ({activeCryptoTraderCount} crypto bot
                          {activeCryptoTraderCount === 1 ? '' : 's'} enabled).
                        </p>
                      ) : (
                        <p className="mt-1 text-[10px] text-amber-600 dark:text-amber-300">
                          Enable an active crypto bot to activate high-frequency monitoring.
                        </p>
                      )}
                    </div>
                  </div>
                  {isCryptoStrategyDraft && cryptoStrategyKeyDraft === 'btc_eth_highfreq' && Number(draftInterval || 0) >= 60 ? (
                    <p className="text-xs text-amber-700 dark:text-amber-100">
                      60s is too slow for short-horizon crypto execution. Recommended cadence is 2s to 10s.
                    </p>
                  ) : null}
                </FlyoutSection>

                <FlyoutSection
                  title="Trading Schedule"
                  icon={Clock3}
                  iconClassName="text-cyan-500"
                  count={tradingScheduleDraft.enabled ? 'active' : 'always-on'}
                  subtitle="UTC-only bot gate by days, time window, and optional date bounds."
                >
                  <div className="rounded-md border border-border p-3 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium">Enable Schedule Gate</p>
                        <p className="text-[10px] text-muted-foreground">
                          When off, this bot can trade any time.
                        </p>
                      </div>
                      <Switch
                        checked={tradingScheduleDraft.enabled}
                        onCheckedChange={(checked) => setDraftTradingSchedule({ enabled: checked })}
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-1.5 md:grid-cols-6">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() =>
                          setDraftTradingSchedule({
                            enabled: false,
                          })
                        }
                      >
                        Always On
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() =>
                          setDraftTradingSchedule({
                            enabled: true,
                            days: [...TRADING_SCHEDULE_DAYS],
                            startTimeUtc: '00:00',
                            endTimeUtc: '23:59',
                          })
                        }
                      >
                        24/7
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() =>
                          setDraftTradingSchedule({
                            enabled: true,
                            days: [...TRADING_SCHEDULE_WEEKDAYS],
                            startTimeUtc: '00:00',
                            endTimeUtc: '23:59',
                          })
                        }
                      >
                        Weekdays
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() =>
                          setDraftTradingSchedule({
                            enabled: true,
                            days: [...TRADING_SCHEDULE_WEEKENDS],
                            startTimeUtc: '00:00',
                            endTimeUtc: '23:59',
                          })
                        }
                      >
                        Weekends
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() =>
                          setDraftTradingSchedule({
                            enabled: true,
                            startTimeUtc: '00:00',
                            endTimeUtc: '23:59',
                          })
                        }
                      >
                        Reset Window
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() =>
                          setDraftTradingSchedule({
                            startDateUtc: '',
                            endDateUtc: '',
                            endAtUtc: '',
                          })
                        }
                      >
                        Clear Bounds
                      </Button>
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">Days (UTC)</Label>
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        {TRADING_SCHEDULE_DAYS.map((day) => {
                          const selected = tradingScheduleDraft.days.includes(day)
                          return (
                            <Button
                              key={day}
                              type="button"
                              size="sm"
                              variant={selected ? 'default' : 'outline'}
                              className="h-6 px-2 text-[11px]"
                              onClick={() => toggleTradingScheduleDay(day)}
                              disabled={!tradingScheduleDraft.enabled}
                            >
                              {TRADING_SCHEDULE_DAY_LABEL[day]}
                            </Button>
                          )
                        })}
                      </div>
                    </div>

                    <div className="grid gap-2 md:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Start Time (UTC)</Label>
                        <Input
                          type="time"
                          value={tradingScheduleDraft.startTimeUtc}
                          onChange={(event) => setDraftTradingSchedule({ startTimeUtc: event.target.value })}
                          className="mt-1 h-8 text-xs font-mono"
                          disabled={!tradingScheduleDraft.enabled}
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">End Time (UTC)</Label>
                        <Input
                          type="time"
                          value={tradingScheduleDraft.endTimeUtc}
                          onChange={(event) => setDraftTradingSchedule({ endTimeUtc: event.target.value })}
                          className="mt-1 h-8 text-xs font-mono"
                          disabled={!tradingScheduleDraft.enabled}
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Start Date (UTC)</Label>
                        <Input
                          type="date"
                          value={tradingScheduleDraft.startDateUtc}
                          onChange={(event) => setDraftTradingSchedule({ startDateUtc: event.target.value })}
                          className="mt-1 h-8 text-xs font-mono"
                          disabled={!tradingScheduleDraft.enabled}
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">End Date (UTC)</Label>
                        <Input
                          type="date"
                          value={tradingScheduleDraft.endDateUtc}
                          onChange={(event) => setDraftTradingSchedule({ endDateUtc: event.target.value })}
                          className="mt-1 h-8 text-xs font-mono"
                          disabled={!tradingScheduleDraft.enabled}
                        />
                      </div>
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">Hard End Timestamp (UTC ISO, optional)</Label>
                      <Input
                        value={tradingScheduleDraft.endAtUtc}
                        onChange={(event) => setDraftTradingSchedule({ endAtUtc: event.target.value })}
                        placeholder="2026-02-28T23:59:59Z"
                        className="mt-1 h-8 text-xs font-mono"
                        disabled={!tradingScheduleDraft.enabled}
                      />
                    </div>
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Risk Limits"
                  icon={AlertTriangle}
                  iconClassName="text-rose-500"
                  count={`${riskFormSchema.param_fields.length} fields`}
                  defaultOpen={false}
                  subtitle="Rendered directly from StrategySDK risk limit schema."
                >
                  {riskFormSchema.param_fields.length > 0 ? (
                    <StrategyConfigForm
                      schema={riskFormSchema as { param_fields: any[] }}
                      values={riskFormValues}
                      onChange={setDraftRiskFromForm}
                    />
                  ) : (
                    <p className="text-[10px] text-muted-foreground/80">No risk schema available.</p>
                  )}
                </FlyoutSection>

                <FlyoutSection
                  title="Advanced JSON Editors"
                  icon={Square}
                  iconClassName="text-slate-500"
                  count="2 editors"
                  defaultOpen={false}
                >
                  <details className="rounded-md border border-border p-2">
                    <summary className="cursor-pointer text-xs font-medium">Strategy Params JSON</summary>
                    <textarea
                      className="mt-2 w-full min-h-[190px] rounded-md border bg-background p-2 text-xs font-mono"
                      value={draftParams}
                      onChange={(event) => setDraftParams(event.target.value)}
                    />
                  </details>

                  <details className="rounded-md border border-border p-2">
                    <summary className="cursor-pointer text-xs font-medium">Risk Limits JSON</summary>
                    <textarea
                      className="mt-2 w-full min-h-[190px] rounded-md border bg-background p-2 text-xs font-mono"
                      value={draftRisk}
                      onChange={(event) => setDraftRisk(event.target.value)}
                    />
                  </details>
                </FlyoutSection>

                {traderFlyoutMode === 'edit' && selectedTrader ? (
                  <FlyoutSection
                    title="Delete / Disable Bot"
                    icon={AlertTriangle}
                    iconClassName="text-red-500"
                    tone="danger"
                    count={`${selectedTraderOpenLivePositions + selectedTraderOpenShadowPositions} positions • ${selectedTraderOpenLiveOrders + selectedTraderOpenShadowOrders} open orders`}
                    defaultOpen={false}
                  >
                    <p className="text-xs text-muted-foreground">
                      Open live positions: {selectedTraderOpenLivePositions} • Open shadow positions: {selectedTraderOpenShadowPositions}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Open live orders: {selectedTraderOpenLiveOrders} • Open shadow orders: {selectedTraderOpenShadowOrders}
                    </p>
                    <Select
                      value={deleteAction}
                      onValueChange={(value) => {
                        setDeleteAction(value as 'block' | 'disable' | 'force_delete')
                        setDeleteForceConfirm(false)
                      }}
                    >
                      <SelectTrigger className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="disable">Disable (Recommended)</SelectItem>
                        <SelectItem value="block">Delete (No Open Positions)</SelectItem>
                        <SelectItem value="force_delete">Force Delete (Override Live Checks)</SelectItem>
                      </SelectContent>
                    </Select>
                    {deleteAction === 'force_delete' ? (
                      <div className="space-y-2 rounded-md border border-red-500/40 bg-red-500/10 p-3">
                        <p className="text-xs font-medium text-red-700 dark:text-red-100">
                          {selectedTraderHasLiveDeleteExposure ? 'Force delete with live exposure' : 'Confirm force delete'}
                        </p>
                        <p className="text-[11px] text-red-700/90 dark:text-red-100/90">
                          {selectedTraderHasLiveDeleteExposure
                            ? `This permanently deletes ${selectedTrader.name} while live exposure is still open. Only continue if those positions or orders were already flattened outside Homerun.`
                            : `This permanently deletes ${selectedTrader.name} and bypasses the normal exposure safety check.`}
                        </p>
                        {selectedTraderHasAnyDeleteExposure ? (
                          <p className="text-[11px] text-red-700/90 dark:text-red-100/90">
                            Current exposure: {selectedTraderDeleteExposureSummary}.
                          </p>
                        ) : null}
                        <div className="flex items-center justify-between gap-3 rounded-md border border-red-500/30 bg-background/70 px-3 py-2">
                          <div className="space-y-0.5">
                            <Label className="text-xs text-foreground">Confirm permanent deletion</Label>
                            <p className="text-[11px] text-muted-foreground">
                              {selectedTraderHasLiveDeleteExposure
                                ? 'I understand this can orphan live exposure if it is still open.'
                                : 'I understand this permanently deletes the bot.'}
                            </p>
                          </div>
                          <Switch checked={deleteForceConfirm} onCheckedChange={setDeleteForceConfirm} />
                        </div>
                      </div>
                    ) : null}
                    <Button
                      variant="destructive"
                      className="h-8 text-xs"
                      disabled={
                        deleteTraderMutation.isPending ||
                        (deleteAction === 'force_delete' && !deleteForceConfirm)
                      }
                      onClick={() => deleteTraderMutation.mutate({ traderId: selectedTrader.id, action: deleteAction })}
                    >
                      {deleteTraderMutation.isPending
                        ? 'Processing...'
                        : deleteAction === 'disable'
                          ? 'Disable Bot'
                          : deleteAction === 'force_delete'
                            ? 'Force Delete Bot'
                            : 'Delete Bot'}
                    </Button>
                  </FlyoutSection>
                ) : null}

              </div>
            </ScrollArea>

            <div className="border-t border-border px-4 py-3 flex flex-wrap items-center justify-end gap-2">
              {saveError ? (
                <div className="mr-auto text-xs text-red-500 max-w-[65%] break-words leading-tight" title={saveError}>
                  {saveError}
                </div>
              ) : null}
              <Button variant="outline" onClick={() => setTraderFlyoutOpen(false)} disabled={traderFlyoutBusy}>
                Close
              </Button>
              <Button
                onClick={() => {
                  if (traderFlyoutMode === 'create') {
                    createTraderMutation.mutate()
                    return
                  }
                  if (selectedTrader) {
                    saveTraderMutation.mutate(selectedTrader.id)
                  }
                }}
                disabled={
                  traderFlyoutBusy ||
                  !draftName.trim() ||
                  (traderFlyoutMode === 'create' && !draftCopyFromTraderId && effectiveDraftSources.length === 0)
                }
              >
                {traderFlyoutMode === 'create' ? 'Create Bot' : 'Save Bot'}
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}
