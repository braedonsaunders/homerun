import { type ReactNode, useEffect, useMemo, useState } from 'react'
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
  armTraderOrchestratorLiveStart,
  createTrader,
  deleteTrader,
  getAllTraderOrders,
  getTraderMarketHistory,
  getCryptoMarkets,
  type CryptoMarket,
  getTraderDecisionDetail,
  getTraderDecisions,
  getTraderEvents,
  getValidationJob,
  getLiveTruthMonitorRaw,
  getTraderConfigSchema,
  getTraderOrchestratorOverview,
  getTraderSources,
  getSimulationAccounts,
  getTraders,
  createCopyConfig,
  disableCopyConfig,
  runTraderOnce,
  runTraderOrchestratorLivePreflight,
  setTraderOrchestratorLiveKillSwitch,
  setStrategyExperimentStatus,
  startTraderOrchestrator,
  startTraderOrchestratorLive,
  stopTraderOrchestrator,
  stopTraderOrchestratorLive,
  enqueueLiveTruthMonitorJob,
  exportLiveTruthMonitorArtifact,
  createStrategyExperiment,
  listStrategyExperiments,
  listStrategyExperimentAssignments,
  promoteStrategyExperiment,
  runTraderMonitorIteration,
  type Trader,
  type TraderConfigSchema,
  type TraderEvent,
  type TraderOrder,
  type TraderSourceConfig,
  type TraderSource,
  type StrategyExperimentAssignment,
  type TraderMonitorAgentResponse,
  type ValidationJob,
  type LiveTruthMonitorArtifact,
  type LiveTruthMonitorRawResponse,
  updateCopyConfig,
  updateDiscoverySettings,
  updateTrader,
  getActiveCopyMode,
  getDiscoverySettings,
  type ActiveCopyMode,
  type CopySourceType,
  type DiscoverySettings,
  type TraderOrchestratorConfig,
  updateTraderOrchestratorSettings,
} from '../services/api'
import { cn } from '../lib/utils'
import { getOpportunityPlatformLinks, getTraderOrderPlatformLinks } from '../lib/marketUrls'
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
type TradeStatusFilter = 'all' | 'open' | 'resolved' | 'failed'
type DecisionOutcomeFilter = 'all' | 'selected' | 'blocked' | 'skipped'
type AllBotsTab = 'overview' | 'trades' | 'positions'
type TradeAction = 'BUY' | 'SELL'
type DirectionSide = 'YES' | 'NO'
type PositionDirectionFilter = 'all' | 'yes' | 'no'
type PositionSortField = 'exposure' | 'updated' | 'edge' | 'confidence' | 'unrealized'
type PositionSortDirection = 'asc' | 'desc'
type BotRosterSort = 'name_asc' | 'name_desc' | 'pnl_desc' | 'pnl_asc' | 'open_desc' | 'activity_desc'
type BotRosterGroupBy = 'none' | 'status' | 'source'

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
  paperOrderCount: number
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

type TraderRuntimeStatus = 'running' | 'paused_engine' | 'disabled'

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

type StrategyParamGroupKey = 'scope' | 'timing' | 'entry' | 'sizing' | 'exit' | 'risk' | 'advanced'

type StrategyParamGroup = {
  key: StrategyParamGroupKey
  label: string
  fields: Array<Record<string, unknown>>
}

type DynamicStrategyParamSectionKind = 'strategy' | 'signal_filters' | 'copy_trading'

type DynamicStrategyParamSection = {
  sectionKey: string
  sourceLabel: string
  strategyLabel: string
  groups: StrategyParamGroup[]
  fieldKeys: string[]
  values: Record<string, unknown>
  kind: DynamicStrategyParamSectionKind
}

const CRYPTO_STRATEGY_OPTIONS = [
  { key: 'btc_eth_highfreq', label: 'Crypto High Frequency' },
  { key: 'crypto_spike_reversion', label: 'Crypto Spike Reversion' },
] as const
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

type CopyTradingMode = 'disabled' | CopySourceType

type CopyTradingFormState = {
  copy_mode_type: CopyTradingMode
  individual_wallet: string
  account_id: string
  copy_trade_mode: 'all_trades' | 'arb_only'
  max_position_size: number
  proportional_sizing: boolean
  proportional_multiplier: number
  copy_buys: boolean
  copy_sells: boolean
  copy_delay_seconds: number
  slippage_tolerance: number
  min_roi_threshold: number
}

type TraderSignalFilters = {
  source_filter: 'all' | 'tracked' | 'pool'
  min_tier: 'WATCH' | 'HIGH' | 'EXTREME'
  side_filter: 'all' | 'buy' | 'sell'
  confluence_limit: number
  individual_trade_limit: number
  individual_trade_min_confidence: number
  individual_trade_max_age_minutes: number
}

type GlobalSettingsDraft = {
  runIntervalSeconds: string
  maxGrossExposureUsd: string
  maxDailyLossUsd: string
  maxOrdersPerCycle: string
  pendingExitMaxAllowed: string
  pendingExitIdentityGuardEnabled: boolean
  pendingExitTerminalStatuses: string
  enforceAllowAveragingOff: boolean
  minCooldownSeconds: string
  maxConsecutiveLossesCap: string
  maxOpenOrdersCap: string
  maxOpenPositionsCap: string
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

const DEFAULT_COPY_TRADING: CopyTradingFormState = {
  copy_mode_type: 'disabled',
  individual_wallet: '',
  account_id: '',
  copy_trade_mode: 'all_trades',
  max_position_size: 1000,
  proportional_sizing: false,
  proportional_multiplier: 1.0,
  copy_buys: true,
  copy_sells: true,
  copy_delay_seconds: 5,
  slippage_tolerance: 1.0,
  min_roi_threshold: 2.5,
}

const DEFAULT_SIGNAL_FILTERS: TraderSignalFilters = {
  source_filter: 'all',
  min_tier: 'WATCH',
  side_filter: 'all',
  confluence_limit: 50,
  individual_trade_limit: 40,
  individual_trade_min_confidence: 0.62,
  individual_trade_max_age_minutes: 180,
}
const DEFAULT_ORCHESTRATOR_GLOBAL_RISK = {
  max_gross_exposure_usd: 5000,
  max_daily_loss_usd: 500,
  max_orders_per_cycle: 50,
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
const WIN_ORDER_STATUSES = new Set(['resolved_win', 'closed_win', 'win'])
const LOSS_ORDER_STATUSES = new Set(['resolved_loss', 'closed_loss', 'loss'])

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
    label: 'Wallet Signals',
    description: 'Tracked/pool/individual/group wallet activity signals.',
    domains: ['event_markets'],
    signal_types: ['confluence'],
    strategy_options: [{ key: 'traders_confluence', label: 'Wallet Confluence', description: '', default_params: {}, param_fields: [] }],
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
  news_edge: 'News Reaction',
  weather_distribution: 'Weather Distribution',
  traders_confluence: 'Wallet Confluence',
  flash_crash_reversion: 'Opportunity Flash Reversion',
  tail_end_carry: 'Opportunity Tail Carry',
}

const DEFAULT_STRATEGY_KEY = 'btc_eth_highfreq'
const DEFAULT_STRATEGY_BY_SOURCE: Record<string, string> = {
  crypto: 'btc_eth_highfreq',
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

function buildGlobalSettingsDraft(config: TraderOrchestratorConfig | null | undefined): GlobalSettingsDraft {
  const globalRisk = config?.global_risk || DEFAULT_ORCHESTRATOR_GLOBAL_RISK
  const runtime = config?.global_runtime || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME
  const pending = runtime.pending_live_exit_guard || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.pending_live_exit_guard
  const clamps = runtime.live_risk_clamps || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps
  const marketContext = runtime.live_market_context || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_market_context
  const providerHealth = runtime.live_provider_health || DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_provider_health
  const pendingTerminalStatuses = toStringList(pending.terminal_statuses)
  return {
    runIntervalSeconds: String(config?.run_interval_seconds ?? 5),
    maxGrossExposureUsd: String(globalRisk.max_gross_exposure_usd ?? DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_gross_exposure_usd),
    maxDailyLossUsd: String(globalRisk.max_daily_loss_usd ?? DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_daily_loss_usd),
    maxOrdersPerCycle: String(globalRisk.max_orders_per_cycle ?? DEFAULT_ORCHESTRATOR_GLOBAL_RISK.max_orders_per_cycle),
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
    maxOpenPositionsCap: String(
      clamps.max_open_positions_cap ?? DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_positions_cap
    ),
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
    updatedAt: latestTimestampValue(
      cleanText(order.updated_at),
      cleanText(order.mark_updated_at),
      cleanText(positionState.last_marked_at),
      cleanText(order.executed_at),
      cleanText(order.created_at)
    ),
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
  fillPrice: number
  markPrice: number
  realizedPnl: number
  filledNotional: number
}): number {
  const {
    status,
    edgePercent,
    fillPrice,
    markPrice,
    realizedPnl,
    filledNotional,
  } = params
  if (OPEN_ORDER_STATUSES.has(status) && fillPrice > 0 && markPrice > 0) {
    return ((markPrice - fillPrice) / fillPrice) * 100
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
  if (signalPayload) {
    const rawMarkets = Array.isArray(signalPayload.markets) ? signalPayload.markets : []
    const signalMarkets = rawMarkets.filter((market): market is Record<string, unknown> => isRecord(market))
    const marketId = String(order.market_id || '').trim()
    const normalizedMarketId = marketId.toLowerCase()
    const matchedMarkets = marketId
      ? signalMarkets.filter((market) => {
          const candidates = [
            cleanText(market.id),
            cleanText(market.market_id),
            cleanText(market.slug),
            cleanText(market.market_slug),
            cleanText(market.ticker),
            cleanText(market.condition_id),
            cleanText(market.conditionId),
          ]
            .filter((value): value is string => Boolean(value))
            .map((value) => value.toLowerCase())
          return candidates.includes(normalizedMarketId)
        })
      : []
    const opportunityLinks = getOpportunityPlatformLinks({
      ...signalPayload,
      markets: matchedMarkets.length > 0 ? matchedMarkets : signalMarkets,
    } as any)
    if (opportunityLinks.polymarketUrl || opportunityLinks.kalshiUrl) {
      return {
        polymarket: opportunityLinks.polymarketUrl,
        kalshi: opportunityLinks.kalshiUrl,
      }
    }
  }

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

function resolveTraderStatusPresentation(
  trader: Pick<Trader, 'is_enabled' | 'is_paused'> | null | undefined,
  orchestratorExecutionActive: boolean
): TraderStatusPresentation {
  if (!isTraderExecutionEnabled(trader)) {
    return {
      key: 'disabled',
      label: 'Disabled',
      dotClassName: 'bg-zinc-500',
      badgeVariant: 'outline',
      badgeClassName: '',
    }
  }

  if (!orchestratorExecutionActive) {
    return {
      key: 'paused_engine',
      label: 'Paused (Engine)',
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

function resolveOrderStatusBadgePresentation(
  status: string,
  pnl: number
): { variant: 'default' | 'secondary' | 'destructive' | 'outline'; className: string } {
  if (status === 'cancelled') {
    return {
      variant: 'outline',
      className: 'border-zinc-300 bg-zinc-100 text-zinc-900 dark:border-zinc-400/45 dark:bg-zinc-500/20 dark:text-zinc-100',
    }
  }

  if (status === 'open') {
    return {
      variant: 'outline',
      className: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/20 dark:text-sky-100',
    }
  }

  if (status === 'executed') {
    return {
      variant: 'outline',
      className: 'border-indigo-300 bg-indigo-100 text-indigo-900 dark:border-indigo-400/45 dark:bg-indigo-500/20 dark:text-indigo-100',
    }
  }

  if (status === 'submitted' || status === 'pending' || status === 'queued') {
    return {
      variant: 'outline',
      className: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/20 dark:text-amber-100',
    }
  }

  if (WIN_ORDER_STATUSES.has(status)) {
    return {
      variant: 'outline',
      className: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/20 dark:text-emerald-100',
    }
  }

  if (LOSS_ORDER_STATUSES.has(status) || FAILED_ORDER_STATUSES.has(status)) {
    return {
      variant: 'outline',
      className: 'border-red-300 bg-red-100 text-red-900 dark:border-red-400/45 dark:bg-red-500/20 dark:text-red-100',
    }
  }

  if (status === 'resolved') {
    return {
      variant: 'outline',
      className: 'border-slate-300 bg-slate-100 text-slate-900 dark:border-slate-400/45 dark:bg-slate-500/20 dark:text-slate-100',
    }
  }

  if (RESOLVED_ORDER_STATUSES.has(status)) {
    if (pnl > 0) {
      return {
        variant: 'outline',
        className: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/20 dark:text-emerald-100',
      }
    }
    if (pnl < 0) {
      return {
        variant: 'outline',
        className: 'border-red-300 bg-red-100 text-red-900 dark:border-red-400/45 dark:bg-red-500/20 dark:text-red-100',
      }
    }
    return {
      variant: 'outline',
      className: 'border-slate-300 bg-slate-100 text-slate-900 dark:border-slate-400/45 dark:bg-slate-500/20 dark:text-slate-100',
    }
  }

  return {
    variant: 'outline',
    className: 'border-border bg-muted/50 text-foreground',
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

function resolveOrderOutcomeBadgeClassName(status: string): string {
  if (status === 'cancelled') {
    return 'border-zinc-300/90 bg-zinc-100/80 text-zinc-900 dark:border-zinc-400/45 dark:bg-zinc-500/12 dark:text-zinc-200'
  }
  if (FAILED_ORDER_STATUSES.has(status)) {
    return 'border-red-300/90 bg-red-100/80 text-red-900 dark:border-red-400/45 dark:bg-red-500/12 dark:text-red-200'
  }
  if (RESOLVED_ORDER_STATUSES.has(status)) {
    return 'border-emerald-300/90 bg-emerald-100/80 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/12 dark:text-emerald-200'
  }
  return 'border-zinc-300/90 bg-zinc-100/80 text-zinc-800 dark:border-zinc-500/45 dark:bg-zinc-500/12 dark:text-zinc-200'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function cleanText(value: unknown): string | null {
  const text = String(value || '').trim()
  return text ? text : null
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
  return (
    cleanText(decision.reason)
    || (strategyDecision ? cleanText(strategyDecision.reason) : null)
    || failedCheckReason
    || failedGateReason
    || 'No reason provided'
  )
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
  const reason = orderReasonDetail(order)
  const normalizedReason = reason.toLowerCase()
  const payload = isRecord(order.payload) ? order.payload : null
  const submission = payload ? cleanText(payload.submission)?.toLowerCase() || '' : ''
  const priceResolution = payload ? cleanText(payload.price_resolution)?.toLowerCase() || '' : ''
  const resolvedPrice = payload ? toFiniteNumber(payload.resolved_price) : null

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
  const liveOrders = Math.max(0, Math.trunc(toNumber(detail.open_live_orders)))
  const otherPositions = Math.max(0, Math.trunc(toNumber(detail.open_other_positions)))
  const otherOrders = Math.max(0, Math.trunc(toNumber(detail.open_other_orders)))
  const parts: string[] = []
  if (livePositions > 0) parts.push(`${livePositions} live position(s)`)
  if (liveOrders > 0) parts.push(`${liveOrders} live active order(s)`)
  if (otherPositions > 0) parts.push(`${otherPositions} unknown position(s)`)
  if (otherOrders > 0) parts.push(`${otherOrders} unknown active order(s)`)
  return {
    message,
    summary: parts.join(' • '),
  }
}

function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error) return error.message || fallback
  if (typeof error === 'object' && error !== null && 'response' in error) {
    const maybeResponse = (error as { response?: { data?: unknown } }).response
    const data = maybeResponse?.data
    if (typeof data === 'string') return data
    if (typeof data === 'object' && data !== null) {
      const detail = (data as { detail?: unknown }).detail
      if (typeof detail === 'string') return detail
      if (typeof detail === 'object' && detail !== null && 'message' in detail) {
        const message = (detail as { message?: unknown }).message
        if (typeof message === 'string') return message
      }
    }
  }
  return fallback
}

function downloadBlobFile(filename: string, blob: Blob): void {
  const link = document.createElement('a')
  const objectUrl = window.URL.createObjectURL(blob)
  link.href = objectUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(objectUrl)
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

function asSignalSideFilter(value: unknown): TraderSignalFilters['side_filter'] {
  const side = String(value || '').trim().toLowerCase()
  if (side === 'buy') return 'buy'
  if (side === 'sell') return 'sell'
  return 'all'
}

function asSignalSourceFilter(value: unknown): TraderSignalFilters['source_filter'] {
  const source = String(value || '').trim().toLowerCase()
  if (source === 'tracked') return 'tracked'
  if (source === 'pool') return 'pool'
  return 'all'
}

function signalFiltersFromDiscoverySettings(
  discovery: DiscoverySettings | undefined
): TraderSignalFilters {
  if (!discovery) return { ...DEFAULT_SIGNAL_FILTERS }
  return {
    source_filter: asSignalSourceFilter(discovery.trader_opps_source_filter),
    min_tier: discovery.trader_opps_min_tier || DEFAULT_SIGNAL_FILTERS.min_tier,
    side_filter: asSignalSideFilter(discovery.trader_opps_side_filter),
    confluence_limit: Math.round(clamp(Number(discovery.trader_opps_confluence_limit || 50), 1, 200)),
    individual_trade_limit: Math.round(clamp(Number(discovery.trader_opps_insider_limit || 40), 1, 500)),
    individual_trade_min_confidence: clamp(Number(discovery.trader_opps_insider_min_confidence || 0.62), 0, 1),
    individual_trade_max_age_minutes: Math.round(clamp(Number(discovery.trader_opps_insider_max_age_minutes || 180), 1, 1440)),
  }
}

function normalizeSignalFiltersConfig(value: Record<string, unknown> | null | undefined): TraderSignalFilters {
  const raw = value || {}
  return {
    source_filter: asSignalSourceFilter(raw.source_filter),
    min_tier:
      String(raw.min_tier || '').trim().toUpperCase() === 'HIGH'
        ? 'HIGH'
        : String(raw.min_tier || '').trim().toUpperCase() === 'EXTREME'
          ? 'EXTREME'
          : 'WATCH',
    side_filter: asSignalSideFilter(raw.side_filter),
    confluence_limit: Math.round(clamp(Number(raw.confluence_limit ?? DEFAULT_SIGNAL_FILTERS.confluence_limit), 1, 200)),
    individual_trade_limit: Math.round(
      clamp(Number(raw.individual_trade_limit ?? DEFAULT_SIGNAL_FILTERS.individual_trade_limit), 1, 500)
    ),
    individual_trade_min_confidence: clamp(
      Number(raw.individual_trade_min_confidence ?? DEFAULT_SIGNAL_FILTERS.individual_trade_min_confidence),
      0,
      1
    ),
    individual_trade_max_age_minutes: Math.round(
      clamp(
        Number(raw.individual_trade_max_age_minutes ?? DEFAULT_SIGNAL_FILTERS.individual_trade_max_age_minutes),
        1,
        1440
      )
    ),
  }
}

function normalizeCopyTradingConfig(value: Record<string, unknown> | null | undefined): CopyTradingFormState {
  const raw = value || {}
  const modeRaw = String(raw.copy_mode_type || '').trim().toLowerCase()
  const copyModeType: CopyTradingMode =
    modeRaw === 'individual' || modeRaw === 'pool' || modeRaw === 'tracked_group'
      ? (modeRaw as CopyTradingMode)
      : 'disabled'
  const tradeModeRaw = String(raw.copy_trade_mode || '').trim().toLowerCase()
  return {
    copy_mode_type: copyModeType,
    individual_wallet: String(raw.individual_wallet || ''),
    account_id: String(raw.account_id || ''),
    copy_trade_mode: tradeModeRaw === 'arb_only' ? 'arb_only' : 'all_trades',
    max_position_size: clamp(Number(raw.max_position_size ?? DEFAULT_COPY_TRADING.max_position_size), 10, 1_000_000),
    proportional_sizing: toBoolean(raw.proportional_sizing, DEFAULT_COPY_TRADING.proportional_sizing),
    proportional_multiplier: clamp(
      Number(raw.proportional_multiplier ?? DEFAULT_COPY_TRADING.proportional_multiplier),
      0.01,
      100
    ),
    copy_buys: toBoolean(raw.copy_buys, DEFAULT_COPY_TRADING.copy_buys),
    copy_sells: toBoolean(raw.copy_sells, DEFAULT_COPY_TRADING.copy_sells),
    copy_delay_seconds: Math.round(clamp(Number(raw.copy_delay_seconds ?? DEFAULT_COPY_TRADING.copy_delay_seconds), 0, 300)),
    slippage_tolerance: clamp(Number(raw.slippage_tolerance ?? DEFAULT_COPY_TRADING.slippage_tolerance), 0, 10),
    min_roi_threshold: clamp(Number(raw.min_roi_threshold ?? DEFAULT_COPY_TRADING.min_roi_threshold), 0, 100),
  }
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
  return uniqueSourceList((trader.sources || []).map((source) => normalizeSourceKey(source)))
}

function copyTradingFromActiveMode(active: ActiveCopyMode): CopyTradingFormState {
  if (active.mode === 'disabled' || !active.config_id) return { ...DEFAULT_COPY_TRADING }
  return normalizeCopyTradingConfig({
    copy_mode_type: active.mode,
    individual_wallet: active.source_wallet || '',
    account_id: active.account_id || '',
    copy_trade_mode: active.copy_mode || 'all_trades',
    max_position_size: active.settings?.max_position_size ?? DEFAULT_COPY_TRADING.max_position_size,
    proportional_sizing: active.settings?.proportional_sizing ?? DEFAULT_COPY_TRADING.proportional_sizing,
    proportional_multiplier: active.settings?.proportional_multiplier ?? DEFAULT_COPY_TRADING.proportional_multiplier,
    copy_buys: active.settings?.copy_buys ?? DEFAULT_COPY_TRADING.copy_buys,
    copy_sells: active.settings?.copy_sells ?? DEFAULT_COPY_TRADING.copy_sells,
    copy_delay_seconds: active.settings?.copy_delay_seconds ?? DEFAULT_COPY_TRADING.copy_delay_seconds,
    slippage_tolerance: active.settings?.slippage_tolerance ?? DEFAULT_COPY_TRADING.slippage_tolerance,
    min_roi_threshold: active.settings?.min_roi_threshold ?? DEFAULT_COPY_TRADING.min_roi_threshold,
  })
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
    paperOrderCount: number
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
    const edge = toNumber(order.edge_percent)
    const confidence = toNumber(order.confidence)
    const traderName = traderNameById[traderId] || shortId(traderId)
    const mode = normalizeStatus(order.mode)
    const sourceLabel = cleanText(order.source)?.toUpperCase() || 'UNKNOWN'
    const executionSummary = orderExecutionTypeSummary(order)
    const links = buildOrderMarketLinks(order, orderPayload, signalPayload)
    const markUpdatedAt = cleanText(order.mark_updated_at) || cleanText(positionState.last_marked_at)

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
        paperOrderCount: 0,
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
    } else if (mode === 'paper') {
      bucket.paperOrderCount += 1
    }
    bucket.sources.add(sourceLabel)
    if (executionSummary !== '—') {
      bucket.executionTypes.add(executionSummary)
    }
    bucket.lastUpdated = toTs(order.updated_at) > toTs(bucket.lastUpdated)
      ? (order.updated_at || order.executed_at || order.created_at || null)
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
        paperOrderCount: bucket.paperOrderCount,
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
  paperOrders: number
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
  let paperOrders = 0
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
    paperOrders += row.paperOrderCount
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
    paperOrders,
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
  scope,
  orders,
  themeMode,
  onClose,
}: {
  market: CryptoMarket | null
  sharedHistory: unknown[]
  scope: BotMarketModalScope
  orders: TraderOrder[]
  themeMode: 'dark' | 'light'
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
    let paperOrderCount = 0
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
      if (snapshot.mode.toLowerCase() === 'paper') paperOrderCount += 1

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
      paperOrderCount,
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
  const useOracleSeries = oracleHistoryData.length >= 2
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
            <p className="text-[10px] text-muted-foreground">{metrics.liveOrderCount} live · {metrics.paperOrderCount} paper</p>
          </div>
        </div>

        <div className={cn(
          'rounded-lg border overflow-hidden',
          isDark
            ? 'border-slate-700/40 bg-gradient-to-b from-slate-900/75 via-slate-950/80 to-black/90'
            : 'border-slate-200/90 bg-gradient-to-b from-white via-slate-50 to-slate-100/70',
        )}>
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
              Waiting for live price history...
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
    _pushRecord(contexts, payload.position_close)
  }

  if (decision) {
    _pushRecord(contexts, decision)
    _pushRecord(contexts, decision.signal_payload)
    _pushRecord(contexts, decision.signal_strategy_context)
    _pushRecord(contexts, decision.payload)
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
  const selectedAccountMode: 'paper' | 'live' = selectedAccountIsLive ? 'live' : 'paper'
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const [traderFeedFilter, setTraderFeedFilter] = useState<FeedFilter>('all')
  const [tradeStatusFilter, setTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [tradeSearch, setTradeSearch] = useState('')
  const [decisionSearch, setDecisionSearch] = useState('')
  const [decisionOutcomeFilter, setDecisionOutcomeFilter] = useState<DecisionOutcomeFilter>('all')
  const [botRosterSearch, setBotRosterSearch] = useState('')
  const [botRosterHideInactive, setBotRosterHideInactive] = useState(false)
  const [botRosterSort, setBotRosterSort] = useState<BotRosterSort>('name_asc')
  const [botRosterGroupBy, setBotRosterGroupBy] = useState<BotRosterGroupBy>('status')
  const [confirmLiveStartOpen, setConfirmLiveStartOpen] = useState(false)
  const [globalSettingsFlyoutOpen, setGlobalSettingsFlyoutOpen] = useState(false)
  const [globalSettingsSaveError, setGlobalSettingsSaveError] = useState<string | null>(null)
  const [globalSettingsDraft, setGlobalSettingsDraft] = useState<GlobalSettingsDraft>(() => buildGlobalSettingsDraft(null))
  const [workTab, setWorkTab] = useState<'trades' | 'terminal' | 'monitor' | 'decisions' | 'performance'>('trades')
  const [allBotsTab, setAllBotsTab] = useState<AllBotsTab>('overview')
  const [allBotsTradeStatusFilter, setAllBotsTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [allBotsTradeSearch, setAllBotsTradeSearch] = useState('')
  const [allBotsPositionSearch, setAllBotsPositionSearch] = useState('')
  const [allBotsPositionDirectionFilter, setAllBotsPositionDirectionFilter] = useState<PositionDirectionFilter>('all')
  const [allBotsPositionSortField, setAllBotsPositionSortField] = useState<PositionSortField>('exposure')
  const [allBotsPositionSortDirection, setAllBotsPositionSortDirection] = useState<PositionSortDirection>('desc')

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
  const [draftMode, setDraftMode] = useState<'paper' | 'live'>('paper')
  const [draftCopyFromTraderId, setDraftCopyFromTraderId] = useState('')
  const [draftCopyFromMode, setDraftCopyFromMode] = useState<'paper' | 'live'>('paper')
  const [saveError, setSaveError] = useState<string | null>(null)
  const [deleteAction, setDeleteAction] = useState<'block' | 'disable' | 'force_delete'>('disable')
  const [deleteConfirmName, setDeleteConfirmName] = useState('')
  const [draftCopyTrading, setDraftCopyTrading] = useState<CopyTradingFormState>(DEFAULT_COPY_TRADING)
  const [draftSignalFilters, setDraftSignalFilters] = useState<TraderSignalFilters>(DEFAULT_SIGNAL_FILTERS)
  const [liveTruthDurationSeconds, setLiveTruthDurationSeconds] = useState('300')
  const [liveTruthPollSeconds, setLiveTruthPollSeconds] = useState('1.0')
  const [liveTruthRunLlm, setLiveTruthRunLlm] = useState(false)
  const [liveTruthEnableProviderChecks, setLiveTruthEnableProviderChecks] = useState(false)
  const [liveTruthIncludeStrategySource, setLiveTruthIncludeStrategySource] = useState(true)
  const [liveTruthModel, setLiveTruthModel] = useState('')
  const [liveTruthMaxAlertsForLlm, setLiveTruthMaxAlertsForLlm] = useState('80')
  const [liveTruthJobId, setLiveTruthJobId] = useState<string | null>(null)
  const [liveTruthError, setLiveTruthError] = useState<string | null>(null)
  const [monitorIteratePrompt, setMonitorIteratePrompt] = useState('')
  const [monitorIterateModel, setMonitorIterateModel] = useState('')
  const [monitorIterateMaxIterations, setMonitorIterateMaxIterations] = useState('12')
  const [monitorIterateMonitorJobId, setMonitorIterateMonitorJobId] = useState('')
  const [monitorIterateError, setMonitorIterateError] = useState<string | null>(null)
  const [monitorIterateResponse, setMonitorIterateResponse] = useState<TraderMonitorAgentResponse | null>(null)
  const [experimentFilterStatus, setExperimentFilterStatus] = useState('all')
  const [experimentDraftSourceKey, setExperimentDraftSourceKey] = useState('')
  const [experimentDraftStrategyKey, setExperimentDraftStrategyKey] = useState('')
  const [experimentDraftControlVersion, setExperimentDraftControlVersion] = useState('')
  const [experimentDraftCandidateVersion, setExperimentDraftCandidateVersion] = useState('')
  const [experimentDraftAllocationPct, setExperimentDraftAllocationPct] = useState('50')
  const [experimentDraftName, setExperimentDraftName] = useState('')
  const [experimentDraftNotes, setExperimentDraftNotes] = useState('')
  const [selectedExperimentId, setSelectedExperimentId] = useState('')
  const [experimentActionError, setExperimentActionError] = useState<string | null>(null)

  const overviewQuery = useQuery({
    queryKey: ['trader-orchestrator-overview'],
    queryFn: getTraderOrchestratorOverview,
    refetchInterval: 4000,
  })

  const tradersQuery = useQuery({
    queryKey: ['traders-list', selectedAccountMode],
    queryFn: () => getTraders({ mode: selectedAccountMode }),
    refetchInterval: 5000,
  })

  const allTradersQuery = useQuery({
    queryKey: ['traders-list', 'all'],
    queryFn: () => getTraders(),
    refetchInterval: 5000,
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

  const activeCopyModeQuery = useQuery({
    queryKey: ['copy-trading-active-mode'],
    queryFn: getActiveCopyMode,
    staleTime: 15000,
  })

  const discoverySettingsQuery = useQuery({
    queryKey: ['settings-discovery'],
    queryFn: getDiscoverySettings,
    staleTime: 15000,
  })

  const cryptoMarketsQuery = useQuery({
    queryKey: ['crypto-markets'],
    queryFn: () => getCryptoMarkets(),
    refetchInterval: 5000,
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
  const [marketModalState, setMarketModalState] = useState<BotMarketModalState | null>(null)
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
  const activeCopyMode = activeCopyModeQuery.data ?? null
  const traders = tradersQuery.data || []
  const allTraders = allTradersQuery.data || []
  const simulationAccounts = simulationAccountsQuery.data || []
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
    refetchInterval: isConnected ? 2000 : 1000,
    staleTime: 0,
    refetchOnMount: 'always',
  })

  const allDecisionsQuery = useQuery({
    queryKey: ['trader-decisions-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: isConnected ? 15000 : 1000,
    staleTime: 0,
    refetchOnMount: 'always',
    queryFn: async () => {
      const grouped = await Promise.all(
        traderIds.map((traderId) => getTraderDecisions(traderId, { limit: 160 }))
      )
      return grouped
        .flat()
        .sort((a, b) => toTs(b.created_at) - toTs(a.created_at))
    },
  })

  const allEventsQuery = useQuery({
    queryKey: ['trader-events-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: false,
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
      if (orderMode === 'live' || orderMode === 'paper') return orderMode === selectedAccountMode
      return selectedAccountMode === 'paper'
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
    enabled: marketModalMarketIds.length > 0,
    refetchInterval: marketModalState ? 1000 : false,
    staleTime: 0,
    refetchOnMount: 'always',
    queryFn: async () => {
      if (marketModalMarketIds.length === 0) return {}
      return getTraderMarketHistory(marketModalMarketIds, 600)
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
  const selectedTraderSourceConfigs = useMemo(
    () => (Array.isArray(selectedTrader?.source_configs) ? selectedTrader.source_configs : []),
    [selectedTrader]
  )

  useEffect(() => {
    setSelectedTraderId((current) => {
      if (!current) return current
      return traders.some((trader) => trader.id === current) ? current : null
    })
  }, [traders])

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
  const tradersSourceEnabled = useMemo(
    () => selectedSourceKeySet.has('traders'),
    [selectedSourceKeySet]
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
  const traderOpportunityFilterSchema = useMemo(
    () => ({
      param_fields: Array.isArray(traderConfigSchema?.trader_opportunity_filters_schema?.param_fields)
        ? traderConfigSchema.trader_opportunity_filters_schema.param_fields
        : [],
    }),
    [traderConfigSchema]
  )
  const copyTradingSchema = useMemo(() => {
    const baseFields = Array.isArray(traderConfigSchema?.copy_trading_schema?.param_fields)
      ? traderConfigSchema.copy_trading_schema.param_fields
      : []
    const accountOptions = simulationAccounts
      .map((account) => {
        const value = String(account.id || '').trim()
        if (!value) return null
        const label = String(account.name || account.id || '').trim() || value
        return { value, label }
      })
      .filter((option): option is { value: string; label: string } => Boolean(option))
    const existingAccountId = String(draftCopyTrading.account_id || '').trim()
    if (existingAccountId && !accountOptions.some((option) => option.value === existingAccountId)) {
      accountOptions.unshift({ value: existingAccountId, label: existingAccountId })
    }
    const mode = draftCopyTrading.copy_mode_type
    const tradeMode = draftCopyTrading.copy_trade_mode
    const proportionalSizing = draftCopyTrading.proportional_sizing
    const filteredFields = baseFields.filter((field: Record<string, unknown>) => {
      const key = String(field.key || '').trim()
      if (!key) return false
      if (mode === 'disabled') {
        return key === 'copy_mode_type'
      }
      if (mode !== 'individual' && key === 'individual_wallet') return false
      if (tradeMode !== 'arb_only' && key === 'min_roi_threshold') return false
      if (!proportionalSizing && key === 'proportional_multiplier') return false
      return true
    })
    const accountField = accountOptions.length > 0
      ? [{ key: 'account_id', label: 'Simulation Account', type: 'enum', options: accountOptions }]
      : []
    return {
      param_fields: [...accountField, ...filteredFields],
    }
  }, [
    draftCopyTrading.account_id,
    draftCopyTrading.copy_mode_type,
    draftCopyTrading.copy_trade_mode,
    draftCopyTrading.proportional_sizing,
    simulationAccounts,
    traderConfigSchema,
  ])
  const parsedDraftParams = useMemo(() => parseJsonObject(draftParams || '{}'), [draftParams])
  const parsedDraftRisk = useMemo(() => parseJsonObject(draftRisk || '{}'), [draftRisk])
  const parsedDraftMetadata = useMemo(() => parseJsonObject(draftMetadata || '{}'), [draftMetadata])
  const defaultSignalFilters = useMemo(
    () =>
      normalizeSignalFiltersConfig(
        isRecord(traderConfigSchema?.trader_opportunity_filters_defaults)
          ? (traderConfigSchema.trader_opportunity_filters_defaults as Record<string, unknown>)
          : DEFAULT_SIGNAL_FILTERS
      ),
    [traderConfigSchema]
  )
  const copyTradingDefaults = useMemo(
    () =>
      normalizeCopyTradingConfig(
        isRecord(traderConfigSchema?.copy_trading_defaults)
          ? (traderConfigSchema.copy_trading_defaults as Record<string, unknown>)
          : DEFAULT_COPY_TRADING
      ),
    [traderConfigSchema]
  )
  const defaultCopyTradingAccountId = useMemo(() => {
    if (activeCopyMode?.account_id) return String(activeCopyMode.account_id)
    if (selectedSandboxAccount?.id) return String(selectedSandboxAccount.id)
    return simulationAccounts[0]?.id || ''
  }, [activeCopyMode, selectedSandboxAccount, simulationAccounts])
  const copyTradingFormValues = useMemo(
    () => ({
      ...copyTradingDefaults,
      ...draftCopyTrading,
      account_id: draftCopyTrading.account_id || defaultCopyTradingAccountId,
    }),
    [copyTradingDefaults, defaultCopyTradingAccountId, draftCopyTrading]
  )
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

      const filteredFields = strategyDetail.paramFields.filter((field): field is Record<string, unknown> => {
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
        kind: 'strategy',
      })
    }

    if (tradersSourceEnabled) {
      const signalFields = traderOpportunityFilterSchema.param_fields.filter((field): field is Record<string, unknown> => {
        if (!isRecord(field)) return false
        const key = String(field.key || '').trim()
        return Boolean(key)
      })
      if (signalFields.length > 0) {
        const fieldKeys = signalFields
          .map((field) => String(field.key || '').trim())
          .filter(Boolean)
        if (fieldKeys.length > 0) {
          sections.push({
            sectionKey: 'traders:signal_filters',
            sourceLabel: 'Wallet Signals',
            strategyLabel: 'Wallet Signal Filters (Global)',
            groups: groupStrategyParamFields(signalFields),
            fieldKeys,
            values: draftSignalFilters as Record<string, unknown>,
            kind: 'signal_filters',
          })
        }
      }

      const copyFields = copyTradingSchema.param_fields.filter((field): field is Record<string, unknown> => {
        if (!isRecord(field)) return false
        const key = String(field.key || '').trim()
        return Boolean(key)
      })
      if (copyFields.length > 0) {
        const fieldKeys = copyFields
          .map((field) => String(field.key || '').trim())
          .filter(Boolean)
        if (fieldKeys.length > 0) {
          sections.push({
            sectionKey: 'traders:copy_trading',
            sourceLabel: 'Wallet Signals',
            strategyLabel: 'Copy Trading',
            groups: groupStrategyParamFields(copyFields),
            fieldKeys,
            values: copyTradingFormValues,
            kind: 'copy_trading',
          })
        }
      }
    }

    return sections
  }, [
    copyTradingFormValues,
    copyTradingSchema,
    draftSignalFilters,
    effectiveDraftSources,
    effectiveSourceStrategies,
    parsedDraftParams.value,
    sourceCards,
    sourceStrategyDetailsLookup,
    traderOpportunityFilterSchema,
    tradersSourceEnabled,
  ])
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

  const liveTruthJobQuery = useQuery({
    queryKey: ['validation-job-live-truth', liveTruthJobId],
    queryFn: () => getValidationJob(String(liveTruthJobId)),
    enabled: Boolean(liveTruthJobId),
    refetchInterval: (query) => {
      const status = normalizeStatus((query.state.data as ValidationJob | undefined)?.status)
      return status === 'queued' || status === 'running' ? 1500 : false
    },
  })

  const liveTruthJobStatus = normalizeStatus(liveTruthJobQuery.data?.status)
  const liveTruthJobRunning = liveTruthJobStatus === 'queued' || liveTruthJobStatus === 'running'
  const liveTruthJobCompleted = liveTruthJobStatus === 'completed'
  const liveTruthJobFailed = liveTruthJobStatus === 'failed'
  const liveTruthJobProgressPct = Math.round(Math.max(0, Math.min(1, toNumber(liveTruthJobQuery.data?.progress))) * 100)

  const liveTruthRawQuery = useQuery({
    queryKey: ['validation-live-truth-raw', liveTruthJobId],
    queryFn: () => getLiveTruthMonitorRaw(String(liveTruthJobId), { max_alerts: 250 }),
    enabled: Boolean(liveTruthJobId) && liveTruthJobCompleted,
    staleTime: 0,
    refetchOnMount: 'always',
  })

  const liveTruthRaw = liveTruthRawQuery.data as LiveTruthMonitorRawResponse | undefined
  const liveTruthReport = liveTruthRaw?.monitor?.report
  const liveTruthSummary = liveTruthRaw?.monitor?.summary || {}
  const liveTruthAlertsByRule = liveTruthReport?.alerts_by_rule || {}
  const liveTruthAlertRuleRows = useMemo(
    () => Object.entries(liveTruthAlertsByRule).sort((a, b) => Number(b[1]) - Number(a[1])),
    [liveTruthAlertsByRule]
  )
  const liveTruthLlm = liveTruthRaw?.llm_analysis || {}
  const liveTruthHasLlm = isRecord(liveTruthLlm) && Object.keys(liveTruthLlm).length > 0
  const liveTruthLlmAnalysis = isRecord(liveTruthLlm.analysis) ? liveTruthLlm.analysis : null
  const liveTruthLlmStrategyChanges = Array.isArray(liveTruthLlmAnalysis?.strategy_changes)
    ? liveTruthLlmAnalysis.strategy_changes.filter((item): item is Record<string, unknown> => isRecord(item))
    : []
  const strategyExperimentsQuery = useQuery({
    queryKey: ['strategy-experiments', selectedTraderId],
    enabled: Boolean(selectedTraderId),
    refetchInterval: selectedTraderId ? 5000 : false,
    queryFn: () => listStrategyExperiments({ limit: 250 }),
  })
  const selectedTraderSourceKeySet = useMemo(() => {
    const out = new Set<string>()
    for (const config of selectedTraderSourceConfigs) {
      const sourceKey = normalizeSourceKey(String(config.source_key || ''))
      if (!sourceKey) continue
      out.add(sourceKey)
    }
    return out
  }, [selectedTraderSourceConfigs])
  const filteredStrategyExperiments = useMemo(() => {
    const rows = Array.isArray(strategyExperimentsQuery.data) ? strategyExperimentsQuery.data : []
    let out = rows
    if (selectedTraderSourceKeySet.size > 0) {
      out = out.filter((row) => selectedTraderSourceKeySet.has(normalizeSourceKey(row.source_key)))
    }
    if (experimentFilterStatus !== 'all') {
      out = out.filter((row) => String(row.status || '').trim().toLowerCase() === experimentFilterStatus)
    }
    return out
  }, [strategyExperimentsQuery.data, selectedTraderSourceKeySet, experimentFilterStatus])
  const selectedStrategyExperiment = useMemo(
    () => filteredStrategyExperiments.find((row) => row.id === selectedExperimentId) || null,
    [filteredStrategyExperiments, selectedExperimentId]
  )
  const experimentAssignmentsQuery = useQuery({
    queryKey: ['strategy-experiment-assignments', selectedExperimentId],
    enabled: Boolean(selectedExperimentId),
    refetchInterval: selectedExperimentId ? 5000 : false,
    queryFn: () => listStrategyExperimentAssignments(String(selectedExperimentId), { limit: 200 }),
  })
  const selectedExperimentAssignments = useMemo<StrategyExperimentAssignment[]>(
    () => (Array.isArray(experimentAssignmentsQuery.data) ? experimentAssignmentsQuery.data : []),
    [experimentAssignmentsQuery.data]
  )
  const experimentSourceChoices = useMemo(() => {
    const out: string[] = []
    const seen = new Set<string>()
    for (const config of selectedTraderSourceConfigs) {
      const sourceKey = normalizeSourceKey(String(config.source_key || ''))
      if (!sourceKey || seen.has(sourceKey)) continue
      seen.add(sourceKey)
      out.push(sourceKey)
    }
    return out
  }, [selectedTraderSourceConfigs])
  const experimentStrategyChoices = useMemo(() => {
    const sourceKey = normalizeSourceKey(experimentDraftSourceKey)
    if (!sourceKey) return [] as StrategyOption[]
    const fromCatalog = sourceStrategyOptionsByKey[sourceKey] || []
    if (fromCatalog.length > 0) return fromCatalog
    const seen = new Set<string>()
    const out: StrategyOption[] = []
    for (const config of selectedTraderSourceConfigs) {
      if (normalizeSourceKey(String(config.source_key || '')) !== sourceKey) continue
      const strategyKey = normalizeStrategyKeyForSource(sourceKey, config.strategy_key)
      if (!strategyKey || seen.has(strategyKey)) continue
      seen.add(strategyKey)
      out.push({ key: strategyKey, label: strategyLabelForKey(strategyKey, sourceCards) })
    }
    return out
  }, [experimentDraftSourceKey, selectedTraderSourceConfigs, sourceCards, sourceStrategyOptionsByKey])
  const experimentStrategyDetail = useMemo(() => {
    const sourceKey = normalizeSourceKey(experimentDraftSourceKey)
    const strategyKey = String(experimentDraftStrategyKey || '').trim().toLowerCase()
    if (!sourceKey || !strategyKey) return null
    return sourceStrategyDetailsLookup[sourceKey]?.[strategyKey] || null
  }, [experimentDraftSourceKey, experimentDraftStrategyKey, sourceStrategyDetailsLookup])
  const experimentVersionChoices = useMemo(() => {
    const rows = normalizeVersionList(experimentStrategyDetail?.versions || [])
    const latestVersion = normalizeStrategyVersion(experimentStrategyDetail?.latestVersion)
    if (latestVersion != null && !rows.includes(latestVersion)) {
      rows.unshift(latestVersion)
    }
    const configuredVersions = selectedTraderSourceConfigs
      .filter((item) => {
        const sourceMatches = normalizeSourceKey(String(item.source_key || '')) === normalizeSourceKey(experimentDraftSourceKey)
        const selectedStrategyKey = String(experimentDraftStrategyKey || '').trim().toLowerCase()
        const strategyMatches =
          selectedStrategyKey
          && normalizeStrategyKeyForSource(String(item.source_key || ''), item.strategy_key) === selectedStrategyKey
        return sourceMatches && strategyMatches
      })
      .map((item) => normalizeStrategyVersion(item.strategy_version))
      .filter((value): value is number => value != null)
    for (const configured of configuredVersions) {
      if (!rows.includes(configured)) rows.unshift(configured)
    }
    const controlVersion = normalizeStrategyVersion(experimentDraftControlVersion)
    if (controlVersion != null && !rows.includes(controlVersion)) rows.unshift(controlVersion)
    const candidateVersion = normalizeStrategyVersion(experimentDraftCandidateVersion)
    if (candidateVersion != null && !rows.includes(candidateVersion)) rows.unshift(candidateVersion)
    rows.sort((left, right) => right - left)
    return rows
  }, [
    experimentDraftCandidateVersion,
    experimentDraftControlVersion,
    experimentDraftSourceKey,
    experimentDraftStrategyKey,
    experimentStrategyDetail,
    selectedTraderSourceConfigs,
  ])
  const monitorIterateParsed = useMemo(
    () => (isRecord(monitorIterateResponse?.parsed) ? monitorIterateResponse?.parsed : null),
    [monitorIterateResponse]
  )
  const monitorIterateActions = useMemo(() => {
    if (!monitorIterateParsed) return [] as string[]
    const raw = monitorIterateParsed.actions_taken
    if (!Array.isArray(raw)) return [] as string[]
    return raw.map((item) => String(item || '').trim()).filter(Boolean)
  }, [monitorIterateParsed])
  const monitorIterateNextSteps = useMemo(() => {
    if (!monitorIterateParsed) return [] as string[]
    const raw = monitorIterateParsed.suggested_next_steps
    if (!Array.isArray(raw)) return [] as string[]
    return raw.map((item) => String(item || '').trim()).filter(Boolean)
  }, [monitorIterateParsed])
  const selectedExperimentAssignmentSummary = useMemo(() => {
    let control = 0
    let candidate = 0
    let other = 0
    for (const row of selectedExperimentAssignments) {
      const group = String(row.assignment_group || '').trim().toLowerCase()
      if (group === 'control') {
        control += 1
      } else if (group === 'candidate') {
        candidate += 1
      } else {
        other += 1
      }
    }
    return { control, candidate, other, total: selectedExperimentAssignments.length }
  }, [selectedExperimentAssignments])

  useEffect(() => {
    if (!liveTruthJobId) return
    if (monitorIterateMonitorJobId.trim()) return
    setMonitorIterateMonitorJobId(String(liveTruthJobId))
  }, [liveTruthJobId, monitorIterateMonitorJobId])

  useEffect(() => {
    if (!selectedTrader) {
      setExperimentDraftSourceKey('')
      return
    }
    if (experimentSourceChoices.length === 0) {
      setExperimentDraftSourceKey('')
      return
    }
    const normalizedCurrent = normalizeSourceKey(experimentDraftSourceKey)
    if (!normalizedCurrent || !experimentSourceChoices.includes(normalizedCurrent)) {
      setExperimentDraftSourceKey(experimentSourceChoices[0])
    }
  }, [selectedTrader, experimentDraftSourceKey, experimentSourceChoices])

  useEffect(() => {
    if (!experimentDraftSourceKey) {
      setExperimentDraftStrategyKey('')
      return
    }
    if (experimentStrategyChoices.length === 0) {
      setExperimentDraftStrategyKey('')
      return
    }
    const normalizedCurrent = String(experimentDraftStrategyKey || '').trim().toLowerCase()
    if (!normalizedCurrent || !experimentStrategyChoices.some((option) => option.key === normalizedCurrent)) {
      setExperimentDraftStrategyKey(experimentStrategyChoices[0].key)
    }
  }, [experimentDraftSourceKey, experimentDraftStrategyKey, experimentStrategyChoices])

  useEffect(() => {
    if (experimentVersionChoices.length === 0) {
      if (experimentDraftControlVersion) setExperimentDraftControlVersion('')
      if (experimentDraftCandidateVersion) setExperimentDraftCandidateVersion('')
      return
    }
    const controlVersion = normalizeStrategyVersion(experimentDraftControlVersion)
    if (controlVersion == null || !experimentVersionChoices.includes(controlVersion)) {
      setExperimentDraftControlVersion(String(experimentVersionChoices[0]))
    }
    const candidateVersion = normalizeStrategyVersion(experimentDraftCandidateVersion)
    if (candidateVersion != null && experimentVersionChoices.includes(candidateVersion)) {
      return
    }
    const fallbackCandidate = experimentVersionChoices.find((version) => version !== (controlVersion ?? experimentVersionChoices[0]))
      || experimentVersionChoices[0]
    setExperimentDraftCandidateVersion(String(fallbackCandidate))
  }, [
    experimentDraftCandidateVersion,
    experimentDraftControlVersion,
    experimentVersionChoices,
  ])

  useEffect(() => {
    if (filteredStrategyExperiments.length === 0) {
      if (selectedExperimentId) {
        setSelectedExperimentId('')
      }
      return
    }
    if (filteredStrategyExperiments.some((row) => row.id === selectedExperimentId)) {
      return
    }
    setSelectedExperimentId(filteredStrategyExperiments[0].id)
  }, [filteredStrategyExperiments, selectedExperimentId])

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
    queryClient.invalidateQueries({ queryKey: ['traders-list'] })
    queryClient.invalidateQueries({ queryKey: ['trader-orders-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-orders'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decisions-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decision-detail'] })
    queryClient.invalidateQueries({ queryKey: ['copy-trading-active-mode'] })
    queryClient.invalidateQueries({ queryKey: ['settings-discovery'] })
    queryClient.invalidateQueries({ queryKey: ['strategy-experiments'] })
    queryClient.invalidateQueries({ queryKey: ['strategy-experiment-assignments'] })
    queryClient.invalidateQueries({ queryKey: ['trader-config-schema'] })
    queryClient.invalidateQueries({ queryKey: ['trader-sources'] })
    queryClient.invalidateQueries({ queryKey: ['unified-strategies'] })
    queryClient.invalidateQueries({ queryKey: ['unified-strategy-versions'] })
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
      setDraftMode(trader.mode === 'live' ? 'live' : 'paper')
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
    setDeleteConfirmName('')
    setSaveError(null)
    setLiveTruthError(null)
    setLiveTruthJobId(null)
    setLiveTruthDurationSeconds('300')
    setLiveTruthPollSeconds('1.0')
    setLiveTruthRunLlm(false)
    setLiveTruthEnableProviderChecks(false)
    setLiveTruthIncludeStrategySource(true)
    setLiveTruthModel('')
    setLiveTruthMaxAlertsForLlm('80')
    setMonitorIteratePrompt(
      'Review recent trades and decisions, tune parameters for risk-adjusted PnL improvement, and propose an A/B test between current and candidate strategy versions.'
    )
    setMonitorIterateModel('')
    setMonitorIterateMaxIterations('12')
    setMonitorIterateMonitorJobId('')
    setMonitorIterateError(null)
    setMonitorIterateResponse(null)
    setExperimentFilterStatus('all')
    setExperimentDraftSourceKey(defaultSources[0] || '')
    setExperimentDraftStrategyKey(defaultStrategies[defaultSources[0] || ''] || '')
    setExperimentDraftControlVersion('')
    setExperimentDraftCandidateVersion('')
    setExperimentDraftAllocationPct('50')
    setExperimentDraftName('')
    setExperimentDraftNotes('')
    setSelectedExperimentId('')
    setExperimentActionError(null)
    const resolvedCopy = activeCopyMode && activeCopyMode.mode !== 'disabled'
      ? copyTradingFromActiveMode(activeCopyMode)
      : copyTradingDefaults
    setDraftCopyTrading({
      ...resolvedCopy,
      account_id: resolvedCopy.account_id || defaultCopyTradingAccountId,
    })
    setDraftSignalFilters(
      normalizeSignalFiltersConfig({
        ...defaultSignalFilters,
        ...signalFiltersFromDiscoverySettings(discoverySettingsQuery.data),
      })
    )
    setTraderFlyoutOpen(true)
  }

  const openEditTraderFlyout = (trader: Trader) => {
    setSelectedTraderId(trader.id)
    setTraderFlyoutMode('edit')
    applyTraderDraftSettings(trader)
    setDraftCopyFromMode(trader.mode === 'live' ? 'live' : 'paper')
    setDeleteAction('disable')
    setDeleteConfirmName('')
    setSaveError(null)
    setLiveTruthError(null)
    setLiveTruthJobId(null)
    setLiveTruthDurationSeconds('300')
    setLiveTruthPollSeconds('1.0')
    setLiveTruthRunLlm(false)
    setLiveTruthEnableProviderChecks(false)
    setLiveTruthIncludeStrategySource(true)
    setLiveTruthModel('')
    setLiveTruthMaxAlertsForLlm('80')
    setMonitorIteratePrompt(
      'Analyze this trader performance, tune strategy parameters or source code where justified, and create an A/B experiment for measurable improvement.'
    )
    setMonitorIterateModel('')
    setMonitorIterateMaxIterations('12')
    setMonitorIterateMonitorJobId('')
    setMonitorIterateError(null)
    setMonitorIterateResponse(null)
    setExperimentFilterStatus('all')
    setExperimentDraftSourceKey('')
    setExperimentDraftStrategyKey('')
    setExperimentDraftControlVersion('')
    setExperimentDraftCandidateVersion('')
    setExperimentDraftAllocationPct('50')
    setExperimentDraftName('')
    setExperimentDraftNotes('')
    setSelectedExperimentId('')
    setExperimentActionError(null)
    const resolvedCopy = activeCopyMode && activeCopyMode.mode !== 'disabled'
      ? copyTradingFromActiveMode(activeCopyMode)
      : copyTradingDefaults
    setDraftCopyTrading({
      ...resolvedCopy,
      account_id: resolvedCopy.account_id || defaultCopyTradingAccountId,
    })
    setDraftSignalFilters(
      normalizeSignalFiltersConfig({
        ...defaultSignalFilters,
        ...signalFiltersFromDiscoverySettings(discoverySettingsQuery.data),
      })
    )
    setTraderFlyoutOpen(true)
  }

  const applyDynamicStrategyFormValues = (
    fieldKeys: string[],
    values: Record<string, unknown>,
  ) => {
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

  const applySignalFilterFormValues = (values: Record<string, unknown>) => {
    setDraftSignalFilters(normalizeSignalFiltersConfig(values))
  }

  const applyCopyTradingFormValues = (values: Record<string, unknown>) => {
    setDraftCopyTrading((previous) => {
      const hasAccountId = Object.prototype.hasOwnProperty.call(values, 'account_id')
      const nextAccountIdRaw = hasAccountId
        ? String(values.account_id || '').trim()
        : String(previous.account_id || '').trim()
      const nextAccountId = nextAccountIdRaw === '__none__' ? '' : nextAccountIdRaw
      return normalizeCopyTradingConfig({
        ...previous,
        ...values,
        account_id: nextAccountId || defaultCopyTradingAccountId,
      })
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

  const persistGlobalWalletSettings = async () => {
    const discoveryBase =
      discoverySettingsQuery.data ||
      (await getDiscoverySettings())
    await updateDiscoverySettings({
      ...discoveryBase,
      trader_opps_source_filter: draftSignalFilters.source_filter,
      trader_opps_min_tier: draftSignalFilters.min_tier,
      trader_opps_side_filter: draftSignalFilters.side_filter,
      trader_opps_confluence_limit: draftSignalFilters.confluence_limit,
      trader_opps_insider_limit: draftSignalFilters.individual_trade_limit,
      trader_opps_insider_min_confidence: draftSignalFilters.individual_trade_min_confidence,
      trader_opps_insider_max_age_minutes: draftSignalFilters.individual_trade_max_age_minutes,
    })

    const targetCopy = normalizeCopyTradingConfig(copyTradingFormValues)
    if (targetCopy.copy_mode_type === 'disabled') {
      if (activeCopyMode?.config_id) {
        await disableCopyConfig(activeCopyMode.config_id)
      }
      return
    }

    const targetAccountId = String(targetCopy.account_id || defaultCopyTradingAccountId || '').trim()
    if (!targetAccountId) {
      throw new Error('Copy trading requires a simulation account.')
    }
    const targetWallet = String(targetCopy.individual_wallet || '').trim().toLowerCase()
    if (targetCopy.copy_mode_type === 'individual' && !targetWallet) {
      throw new Error('Copy trading individual mode requires a wallet address.')
    }

    const copyPayload = {
      copy_mode: targetCopy.copy_trade_mode,
      min_roi_threshold: targetCopy.min_roi_threshold,
      max_position_size: targetCopy.max_position_size,
      copy_delay_seconds: targetCopy.copy_delay_seconds,
      slippage_tolerance: targetCopy.slippage_tolerance,
      proportional_sizing: targetCopy.proportional_sizing,
      proportional_multiplier: targetCopy.proportional_multiplier,
      copy_buys: targetCopy.copy_buys,
      copy_sells: targetCopy.copy_sells,
    }

    const activeMode = activeCopyMode?.mode || 'disabled'
    const activeWallet = String(activeCopyMode?.source_wallet || '').trim().toLowerCase()
    const activeAccountId = String(activeCopyMode?.account_id || '').trim()
    const needsNewConfig =
      !activeCopyMode?.config_id ||
      activeMode !== targetCopy.copy_mode_type ||
      activeAccountId !== targetAccountId ||
      (targetCopy.copy_mode_type === 'individual' && activeWallet !== targetWallet)

    if (needsNewConfig) {
      if (activeCopyMode?.config_id) {
        await disableCopyConfig(activeCopyMode.config_id)
      }
      await createCopyConfig({
        source_wallet: targetCopy.copy_mode_type === 'individual' ? targetWallet : null,
        account_id: targetAccountId,
        source_type: targetCopy.copy_mode_type as CopySourceType,
        ...copyPayload,
      })
      return
    }

    if (!activeCopyMode?.config_id) {
      throw new Error('Active copy trading configuration was not found.')
    }
    await updateCopyConfig(activeCopyMode.config_id, {
      enabled: true,
      ...copyPayload,
    })
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
        throw new Error('No sandbox account is selected for paper mode.')
      }
      return startTraderOrchestrator({ mode: 'paper', paper_account_id: selectedSandboxAccount.id })
    },
    onSuccess: refreshAll,
  })

  const stopByModeMutation = useMutation({
    mutationFn: async () => {
      const mode = String(overviewQuery.data?.control?.mode || 'paper').toLowerCase()
      if (mode === 'live') {
        return stopTraderOrchestratorLive()
      }
      return stopTraderOrchestrator()
    },
    onSuccess: refreshAll,
  })

  const killSwitchMutation = useMutation({
    mutationFn: (enabled: boolean) => setTraderOrchestratorLiveKillSwitch(enabled),
    onSuccess: refreshAll,
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
      const maxOpenPositionsCap = Math.trunc(
        clampNumber(
          toNumber(globalSettingsDraft.maxOpenPositionsCap),
          1,
          1000,
          DEFAULT_ORCHESTRATOR_GLOBAL_RUNTIME.live_risk_clamps.max_open_positions_cap,
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
      return updateTraderOrchestratorSettings({
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
            max_open_positions_cap: maxOpenPositionsCap,
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
      })
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

  const traderEnableMutation = useMutation({
    mutationFn: (traderId: string) => updateTrader(traderId, { is_enabled: true, is_paused: false }),
    onSuccess: refreshAll,
  })

  const traderDisableMutation = useMutation({
    mutationFn: (traderId: string) => updateTrader(traderId, { is_enabled: false, is_paused: false }),
    onSuccess: refreshAll,
  })

  const traderRunOnceMutation = useMutation({
    mutationFn: (traderId: string) => runTraderOnce(traderId),
    onSuccess: refreshAll,
  })

  const runLiveTruthMonitorMutation = useMutation({
    mutationFn: async () => {
      if (!selectedTrader) {
        throw new Error('Select a bot before running monitor diagnostics.')
      }
      const durationSeconds = Math.max(10, Math.min(7200, Math.trunc(toNumber(liveTruthDurationSeconds || 300))))
      const pollSeconds = Math.max(0.2, Math.min(10, toNumber(liveTruthPollSeconds || 1.0)))
      const maxAlertsForLlm = Math.max(1, Math.min(400, Math.trunc(toNumber(liveTruthMaxAlertsForLlm || 80))))
      const request = {
        trader_id: selectedTrader.id,
        duration_seconds: durationSeconds,
        poll_seconds: pollSeconds,
        run_llm_analysis: liveTruthRunLlm,
        enable_provider_checks: liveTruthEnableProviderChecks,
        include_strategy_source: liveTruthIncludeStrategySource,
        max_alerts_for_llm: maxAlertsForLlm,
        ...(liveTruthModel.trim() ? { llm_model: liveTruthModel.trim() } : {}),
      }
      return enqueueLiveTruthMonitorJob(request)
    },
    onMutate: () => {
      setLiveTruthError(null)
      setLiveTruthJobId(null)
    },
    onSuccess: (result) => {
      setLiveTruthError(null)
      setLiveTruthJobId(result.job_id)
    },
    onError: (error: unknown) => {
      setLiveTruthError(errorMessage(error, 'Failed to enqueue live truth monitor job'))
    },
  })

  const exportLiveTruthArtifactMutation = useMutation({
    mutationFn: async (artifact: LiveTruthMonitorArtifact) => {
      if (!liveTruthJobId) {
        throw new Error('Run monitor diagnostics first.')
      }
      return exportLiveTruthMonitorArtifact(liveTruthJobId, artifact)
    },
    onSuccess: ({ filename, blob }) => {
      setLiveTruthError(null)
      downloadBlobFile(filename, blob)
    },
    onError: (error: unknown) => {
      setLiveTruthError(errorMessage(error, 'Failed to export monitor artifact'))
    },
  })

  const runMonitorIterateMutation = useMutation({
    mutationFn: async () => {
      if (!selectedTrader) {
        throw new Error('Select a bot before running monitor iterate.')
      }
      const prompt = monitorIteratePrompt.trim()
      if (!prompt) {
        throw new Error('Enter an iterate prompt.')
      }
      const maxIterations = Math.max(1, Math.min(24, Math.trunc(toNumber(monitorIterateMaxIterations || 12))))
      const monitorJobId = monitorIterateMonitorJobId.trim() || (liveTruthJobCompleted ? String(liveTruthJobId || '') : '')
      return runTraderMonitorIteration(selectedTrader.id, {
        prompt,
        max_iterations: maxIterations,
        ...(monitorIterateModel.trim() ? { model: monitorIterateModel.trim() } : {}),
        ...(monitorJobId ? { monitor_job_id: monitorJobId } : {}),
      })
    },
    onMutate: () => {
      setMonitorIterateError(null)
      setExperimentActionError(null)
    },
    onSuccess: (result) => {
      setMonitorIterateError(null)
      setMonitorIterateResponse(result)
      refreshAll()
    },
    onError: (error: unknown) => {
      setMonitorIterateError(errorMessage(error, 'Failed to run monitor iterate'))
    },
  })

  const createStrategyExperimentMutation = useMutation({
    mutationFn: async () => {
      if (!selectedTrader) {
        throw new Error('Select a bot before creating an experiment.')
      }
      const sourceKey = normalizeSourceKey(experimentDraftSourceKey)
      const strategyKey = String(experimentDraftStrategyKey || '').trim().toLowerCase()
      const controlVersion = normalizeStrategyVersion(experimentDraftControlVersion)
      const candidateVersion = normalizeStrategyVersion(experimentDraftCandidateVersion)
      if (!sourceKey) throw new Error('Select a source for the experiment.')
      if (!strategyKey) throw new Error('Select a strategy for the experiment.')
      if (controlVersion == null || candidateVersion == null) {
        throw new Error('Select both control and candidate versions.')
      }
      if (controlVersion === candidateVersion) {
        throw new Error('Control and candidate versions must be different.')
      }
      const allocationPct = Math.max(0.1, Math.min(99.9, toNumber(experimentDraftAllocationPct || 50)))
      const autoName = `${strategyLabelForKey(strategyKey, sourceCards)} v${controlVersion} vs v${candidateVersion}`
      return createStrategyExperiment({
        name: experimentDraftName.trim() || autoName,
        source_key: sourceKey,
        strategy_key: strategyKey,
        control_version: controlVersion,
        candidate_version: candidateVersion,
        candidate_allocation_pct: allocationPct,
        scope: {
          trader_id: selectedTrader.id,
          trader_mode: selectedTrader.mode,
        },
        notes: experimentDraftNotes.trim() || undefined,
        created_by: 'trading_panel',
      })
    },
    onMutate: () => {
      setExperimentActionError(null)
    },
    onSuccess: (result) => {
      setExperimentActionError(null)
      setSelectedExperimentId(result.id)
      setExperimentDraftName('')
      queryClient.invalidateQueries({ queryKey: ['strategy-experiments'] })
      refreshAll()
    },
    onError: (error: unknown) => {
      setExperimentActionError(errorMessage(error, 'Failed to create A/B experiment'))
    },
  })

  const setExperimentStatusMutation = useMutation({
    mutationFn: async ({ experimentId, status }: { experimentId: string; status: string }) => {
      return setStrategyExperimentStatus(experimentId, status)
    },
    onMutate: () => {
      setExperimentActionError(null)
    },
    onSuccess: (result) => {
      setExperimentActionError(null)
      setSelectedExperimentId(result.id)
      queryClient.invalidateQueries({ queryKey: ['strategy-experiments'] })
      refreshAll()
    },
    onError: (error: unknown) => {
      setExperimentActionError(errorMessage(error, 'Failed to update experiment status'))
    },
  })

  const promoteStrategyExperimentMutation = useMutation({
    mutationFn: async ({ experimentId, promotedVersion }: { experimentId: string; promotedVersion?: number }) => {
      return promoteStrategyExperiment(
        experimentId,
        promotedVersion != null ? { promoted_version: promotedVersion } : undefined
      )
    },
    onMutate: () => {
      setExperimentActionError(null)
    },
    onSuccess: (result) => {
      setExperimentActionError(null)
      setSelectedExperimentId(result.id)
      queryClient.invalidateQueries({ queryKey: ['strategy-experiments'] })
      refreshAll()
    },
    onError: (error: unknown) => {
      setExperimentActionError(errorMessage(error, 'Failed to promote experiment'))
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

      await persistGlobalWalletSettings()

      const payload: Record<string, unknown> = {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        mode: draftMode,
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        source_configs: buildDraftSourceConfigs(parsedParams.value),
        risk_limits: parsedRisk.value,
        metadata: parsedMetadata.value,
        is_enabled: false,
        is_paused: false,
      }

      if (copyFromTraderId) {
        payload.copy_from_trader_id = copyFromTraderId
      }

      return createTrader(payload)
    },
    onSuccess: (trader) => {
      setSaveError(null)
      setTraderFlyoutOpen(false)
      setSelectedTraderId(trader.id)
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to create bot'))
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

      await persistGlobalWalletSettings()

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
        setDeleteConfirmName('')
        setSaveError(
          [
            liveExposure.message,
            liveExposure.summary ? `Current exposure: ${liveExposure.summary}.` : null,
            'Select Force Delete and type the bot name to confirm permanent deletion.',
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
  const orchestratorConfig = overviewQuery.data?.config || null
  const metrics = overviewQuery.data?.metrics
  const killSwitchOn = Boolean(overviewQuery.data?.control?.kill_switch)
  const orchestratorEnabled = Boolean(overviewQuery.data?.control?.is_enabled) && !Boolean(overviewQuery.data?.control?.is_paused)
  const workerActivity = String(worker?.current_activity || '').trim().toLowerCase()
  const orchestratorWorkerRunning = Boolean(worker?.running)
  const orchestratorRunning = orchestratorEnabled && orchestratorWorkerRunning
  const orchestratorBlocked = orchestratorEnabled && !orchestratorWorkerRunning && workerActivity.startsWith('blocked')
  const orchestratorStatusLabel = orchestratorBlocked ? 'BLOCKED' : orchestratorRunning ? 'RUNNING' : 'STOPPED'

  const controlBusy =
    startBySelectedAccountMutation.isPending ||
    stopByModeMutation.isPending ||
    killSwitchMutation.isPending
  const traderFlyoutBusy =
    createTraderMutation.isPending ||
    saveTraderMutation.isPending ||
    deleteTraderMutation.isPending
  const globalSettingsBusy = updateGlobalSettingsMutation.isPending

  useEffect(() => {
    if (globalSettingsFlyoutOpen) return
    setGlobalSettingsDraft(buildGlobalSettingsDraft(orchestratorConfig))
  }, [globalSettingsFlyoutOpen, orchestratorConfig])

  const traderNameById = useMemo(
    () => Object.fromEntries(traders.map((trader) => [trader.id, trader.name])) as Record<string, string>,
    [traders]
  )
  const closeMarketModal = () => setMarketModalState(null)

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
        modeSummary: `${row.liveOrderCount}L/${row.paperOrderCount}P`,
        executionSummary: row.executionSummary,
        outcomeSummary: null,
        links: row.links,
      },
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

  const filteredTradeHistory = useMemo(() => {
    const q = tradeSearch.trim().toLowerCase()
    return selectedOrders
      .filter((order) => {
        const status = normalizeStatus(order.status)
        const matchesStatus =
          tradeStatusFilter === 'all' ||
          (tradeStatusFilter === 'open' && OPEN_ORDER_STATUSES.has(status)) ||
          (tradeStatusFilter === 'resolved' && RESOLVED_ORDER_STATUSES.has(status)) ||
          (tradeStatusFilter === 'failed' && FAILED_ORDER_STATUSES.has(status))

        if (!matchesStatus) return false
        if (!q) return true

        const haystack = `${order.market_question || ''} ${order.market_id || ''} ${order.source || ''} ${order.direction || ''} ${order.direction_label || ''} ${order.direction_side || ''} ${orderExecutionTypeSummary(order)}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 250)
  }, [selectedOrders, tradeSearch, tradeStatusFilter])

  const selectedTradeRows = useMemo(() => {
    return filteredTradeHistory.map((order) => {
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
      const markPx = toNumber(
        order.current_price
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
        fillPrice: fillPx,
        markPrice: markPx,
        realizedPnl: pnl,
        filledNotional,
      })
      const exitProgressPercent = computePendingExitProgressPercent(pendingExit as Record<string, unknown>)
      const updatedAt = latestTimestampValue(
        order.mark_updated_at,
        positionState.last_marked_at,
        order.updated_at,
        order.executed_at,
        order.created_at
      )
      const markUpdatedAtRaw = String(order.mark_updated_at || positionState.last_marked_at || '')
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
      const directionPresentation = resolveOrderDirectionPresentation(order)
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
        updatedAt,
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
        outcomeDetailCompact: compactText(outcome.detail, 160),
        venuePresentation,
      }
    })
  }, [decisionSignalPayloadByDecisionId, filteredTradeHistory])

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
    for (const row of selectedTradeRows) {
      total += 1
      totalNotional += Math.abs(toNumber(row.order.notional_usd))
      if (OPEN_ORDER_STATUSES.has(row.status)) {
        open += 1
        unrealizedPnl += row.unrealized
      }
      if (RESOLVED_ORDER_STATUSES.has(row.status)) {
        resolved += 1
        realizedPnl += row.pnl
        if (row.pnl > 0) wins += 1
        if (row.pnl < 0) losses += 1
      }
      if (FAILED_ORDER_STATUSES.has(row.status)) {
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
  }, [selectedTradeRows])

  const filteredAllTradeHistory = useMemo(() => {
    const q = allBotsTradeSearch.trim().toLowerCase()
    return allOrders
      .filter((order) => {
        const status = normalizeStatus(order.status)
        const matchesStatus =
          allBotsTradeStatusFilter === 'all' ||
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

    const eventRows: ActivityRow[] = allEvents.map((event) => {
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
      const severity = String(event.severity || '').trim().toLowerCase()
      const tone: ActivityRow['tone'] =
        severity === 'warn' || severity === 'warning' ? 'warning' :
        severity === 'error' || severity === 'failed' ? 'negative' :
        'neutral'

      return {
        kind: 'event',
        id: event.id,
        ts: event.created_at,
        traderId: event.trader_id,
        title: `${String(event.event_type || 'event').toUpperCase()} • ${String(event.severity || 'info').toUpperCase()} • ${marketLabel}`,
        detail: `Markets: ${renderMarketsDetail(linkedLegs, fallbackMarket)} :: Reason: ${reason}`,
        action,
        tone,
      }
    })

    return [...decisionRows, ...orderRows, ...eventRows]
      .sort((a, b) => toTs(b.ts) - toTs(a.ts))
      .slice(0, 350)
  }, [allDecisions, allOrders, allEvents])

  const selectedTraderActivityRows = useMemo(
    () => activityRows.filter((row) => row.traderId === selectedTraderId),
    [activityRows, selectedTraderId]
  )

  const filteredTraderActivityRows = useMemo(() => {
    return selectedTraderActivityRows
      .filter((row) => traderFeedFilter === 'all' || row.kind === traderFeedFilter)
      .slice(0, 240)
  }, [selectedTraderActivityRows, traderFeedFilter])

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
    () => activityRows.slice(0, 120),
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
        isInactive: !isTraderExecutionEnabled(trader),
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
        } else if (row.status.key === 'paused_engine') {
          label = 'Paused (Engine)'
          order = 1
        } else {
          label = 'Inactive'
          order = 2
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

  const enabledTraderCount = useMemo(
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

  const disabledTraderCount = useMemo(
    () => Math.max(0, traders.length - enabledTraderCount),
    [traders.length, enabledTraderCount]
  )

  const runningTraderCount = useMemo(
    () => (orchestratorRunning ? enabledTraderCount : 0),
    [orchestratorRunning, enabledTraderCount]
  )


  const selectedTraderExposure = useMemo(
    () => selectedPositionBook.reduce((sum, row) => sum + row.exposureUsd, 0),
    [selectedPositionBook]
  )

  const selectedTraderOpenLiveOrders = useMemo(
    () => selectedOrders.filter((order) => OPEN_ORDER_STATUSES.has(normalizeStatus(order.status)) && String(order.mode || '').toLowerCase() === 'live').length,
    [selectedOrders]
  )

  const selectedTraderOpenPaperOrders = useMemo(
    () => selectedOrders.filter((order) => OPEN_ORDER_STATUSES.has(normalizeStatus(order.status)) && String(order.mode || '').toLowerCase() === 'paper').length,
    [selectedOrders]
  )

  const selectedDecision = useMemo(
    () => selectedDecisions.find((decision) => decision.id === selectedDecisionId) || null,
    [selectedDecisions, selectedDecisionId]
  )
  const selectedDecisionDirection = useMemo(
    () => (selectedDecision ? resolveDecisionDirectionPresentation(selectedDecision) : { side: null, label: 'N/A' }),
    [selectedDecision]
  )
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
  const selectedTraderExecutionEnabled = isTraderExecutionEnabled(selectedTrader)
  const selectedTraderCanEnable = Boolean(selectedTrader && !selectedTraderExecutionEnabled)
  const selectedTraderCanDisable = Boolean(selectedTrader && selectedTraderExecutionEnabled)
  const selectedTraderControlPending = traderEnableMutation.isPending || traderDisableMutation.isPending
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
    setGlobalSettingsDraft(buildGlobalSettingsDraft(orchestratorConfig))
    setGlobalSettingsSaveError(null)
    setGlobalSettingsFlyoutOpen(true)
  }

  const resetGlobalSettingsDraft = () => {
    setGlobalSettingsDraft(buildGlobalSettingsDraft(orchestratorConfig))
    setGlobalSettingsSaveError(null)
  }

  const saveGlobalSettings = () => {
    setGlobalSettingsSaveError(null)
    updateGlobalSettingsMutation.mutate()
  }

  const canStartOrchestrator =
    !controlBusy &&
    !orchestratorEnabled &&
    Boolean(selectedAccountId) &&
    selectedAccountValid &&
    !(selectedAccountIsLive && killSwitchOn)
  const canStopOrchestrator = !controlBusy && orchestratorEnabled
  const startStopIsConfigured = orchestratorEnabled
  const startStopIsRunning = orchestratorRunning
  const startStopIsStarting =
    startBySelectedAccountMutation.isPending ||
    (startStopIsConfigured && !startStopIsRunning && workerActivity.includes('start command queued'))
  const startStopIsStopping = stopByModeMutation.isPending
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
            ) : startStopIsConfigured && startStopIsRunning ? (
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
                  ? (startStopIsRunning ? 'Stop' : 'Start')
                  : selectedAccountMode.toUpperCase()}
          </Button>
          <div className="flex items-center gap-1.5 rounded border border-red-500/30 bg-red-500/5 px-1.5 py-0.5">
            <ShieldAlert className="w-3 h-3 text-red-400" />
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <Switch
                    checked={killSwitchOn}
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
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span
            className={cn(
              'w-1.5 h-1.5 rounded-full',
              worker?.last_error
                ? 'bg-amber-400'
                : orchestratorRunning
                  ? 'bg-emerald-500'
                  : 'bg-amber-400'
            )}
          />
          <Clock3 className="w-3 h-3" />
          {formatTimestamp(worker?.last_run_at)}
        </div>

        <div className="flex items-center gap-1.5">
          <Badge className="h-5 px-1.5 text-[10px]" variant={orchestratorBlocked ? 'destructive' : orchestratorRunning ? 'default' : 'secondary'}>
            {orchestratorStatusLabel}
          </Badge>
          <Badge className="h-5 px-1.5 text-[10px]" variant={selectedAccountMode === 'live' ? 'destructive' : 'outline'}>
            {selectedAccountMode.toUpperCase()}
          </Badge>
          <Badge className="h-5 px-1.5 text-[10px]" variant={killSwitchOn ? 'destructive' : 'outline'}>
            {killSwitchOn ? 'BLOCKED' : 'OPEN'}
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
                        return (
                          <button
                            key={row.trader.id}
                            type="button"
                            onClick={() => setSelectedTraderId(row.trader.id)}
                            className={cn(
                              'w-full text-left rounded-md px-2 py-1.5 transition-colors group',
                              isActive
                                ? 'bg-cyan-500/15 text-foreground'
                                : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
                            )}
                          >
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
                      <p className="text-sm font-mono">{runningTraderCount}/{enabledTraderCount}</p>
                      <p className="text-[10px] text-muted-foreground">Disabled {disabledTraderCount}</p>
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
                        {(['all', 'open', 'resolved', 'failed'] as TradeStatusFilter[]).map((status) => (
                          <Button
                            key={status}
                            size="sm"
                            variant={allBotsTradeStatusFilter === status ? 'default' : 'outline'}
                            onClick={() => setAllBotsTradeStatusFilter(status)}
                            className="h-5 px-2 text-[10px]"
                          >
                            {status}
                          </Button>
                        ))}
                        <span className="ml-auto text-[10px] font-mono text-muted-foreground">{filteredAllTradeHistory.length} rows</span>
                      </div>
                      <div className="flex-1 min-h-0 overflow-hidden">
                        {filteredAllTradeHistory.length === 0 ? (
                          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No trades matching filters.</div>
                        ) : (
                          <ScrollArea className="h-full min-h-0 rounded-md border border-border/60 bg-card/80">
                            <div className="w-full overflow-x-auto">
                            <Table className="w-full table-fixed">
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="w-[26%] text-[10px]">Market</TableHead>
                                  <TableHead className="w-[6%] text-[10px]">Dir</TableHead>
                                  <TableHead className="w-[24%] text-[10px]">Lifecycle / Outcome</TableHead>
                                  <TableHead className="w-[8%] text-[10px] text-right">Notional</TableHead>
                                  <TableHead className="w-[6%] text-[10px] text-right">Fill</TableHead>
                                  <TableHead className="w-[6%] text-[10px] text-right">Fill Progress</TableHead>
                                  <TableHead className="w-[6%] text-[10px] text-right">Mark</TableHead>
                                  <TableHead className="w-[8%] text-[10px] text-right">U-P&amp;L</TableHead>
                                  <TableHead className="w-[7%] text-[10px] text-right">Edge Δ</TableHead>
                                  <TableHead className="w-[8%] text-[10px] text-right">R-P&amp;L</TableHead>
                                  <TableHead className="w-[8%] text-[10px]">Venue</TableHead>
                                  <TableHead className="w-[6%] text-[10px] text-right">Exit %</TableHead>
                                  <TableHead className="w-[5%] text-[10px]">Age</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {filteredAllTradeHistory.map((order) => {
                                  const status = normalizeStatus(order.status)
                                  const lifecycleLabel = resolveOrderLifecycleLabel(status)
                                  const pnl = toNumber(order.actual_profit)
                                  const statusBadge = resolveOrderStatusBadgePresentation(status, pnl)
                                  const outcomeBadgeClassName = resolveOrderOutcomeBadgeClassName(status)
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
                                  const markPx = toNumber(
                                    order.current_price
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
                                    fillPrice: fillPx,
                                    markPrice: markPx,
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
                                  const updatedAt = latestTimestampValue(
                                    order.mark_updated_at,
                                    positionState.last_marked_at,
                                    order.updated_at,
                                    order.executed_at,
                                    order.created_at
                                  )
                                  const outcome = orderOutcomeSummary(order)
                                  const outcomeDetailCompact = compactText(outcome.detail, 120)
                                  const executionSummary = orderExecutionTypeSummary(order)
                                  const venuePresentation = resolveVenueStatusPresentation(providerSnapshotStatus)
                                  const directionPresentation = resolveOrderDirectionPresentation(order)
                                  const traderLabel = traderNameById[String(order.trader_id || '')] || shortId(order.trader_id)
                                  const marketForModal = resolveCryptoMarketFromAliases(collectOrderMarketAliasIds(order))
                                  return (
                                  <TableRow
                                    key={order.id}
                                    className="text-[11px] leading-tight cursor-pointer hover:bg-muted/30"
                                    onClick={() => {
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
                                    }}
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
                                      <TableCell className="py-0.5 min-w-[260px]">
                                        <div className="min-w-0 space-y-0">
                                          <div className="flex min-w-0 items-center gap-0.5 overflow-hidden">
                                            <Badge
                                              variant={statusBadge.variant}
                                              title={`Raw status: ${status}`}
                                              className={cn('h-4 shrink-0 whitespace-nowrap px-1 text-[9px] font-semibold', statusBadge.className)}
                                            >
                                              {lifecycleLabel}
                                            </Badge>
                                            <Badge
                                              variant="outline"
                                              title={outcome.detail}
                                              className={cn(
                                                'h-4 max-w-[180px] truncate px-1 text-[9px] font-medium',
                                                outcomeBadgeClassName
                                              )}
                                            >
                                              {outcome.headline}
                                            </Badge>
                                            {executionSummary !== '—' && (
                                              <Badge
                                                variant="outline"
                                                title={`Execution: ${executionSummary}`}
                                                className="h-4 max-w-[150px] truncate border-border/80 bg-background/80 px-1 text-[9px] font-medium text-foreground/85"
                                              >
                                                {executionSummary}
                                              </Badge>
                                            )}
                                          </div>
                                          <p className="truncate text-[9px] leading-none text-foreground/85" title={outcome.detail}>
                                            <span className="font-medium text-muted-foreground">Reason:</span> {outcomeDetailCompact}
                                          </p>
                                        </div>
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatCurrency(toNumber(order.notional_usd), true)}</TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{fillPx > 0 ? fillPx.toFixed(3) : '\u2014'}</TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{fillProgressPercent !== null ? formatPercent(fillProgressPercent, 0) : '\u2014'}</TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{markPx > 0 ? markPx.toFixed(3) : '\u2014'}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}>
                                        {OPEN_ORDER_STATUSES.has(status) ? formatCurrency(unrealized) : '\u2014'}
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatPercent(dynamicEdgePercent)}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', pnl > 0 ? 'text-emerald-500' : pnl < 0 ? 'text-red-500' : '')}>
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
                                        <span title={`${String(order.mode || '').toUpperCase()} • created:${formatTimestamp(order.created_at)} • updated:${formatTimestamp(updatedAt)}`}>
                                          {formatRelativeAge(updatedAt)}
                                        </span>
                                      </TableCell>
                                    </TableRow>
                                  )
                                })}
                              </TableBody>
                            </Table>
                            </div>
                          </ScrollArea>
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
                          <p className="text-[9px] uppercase text-muted-foreground">Live / Paper</p>
                          <p className="text-xs font-mono">{allBotsPositionSummary.liveOrders} / {allBotsPositionSummary.paperOrders}</p>
                        </div>
                        <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                          <p className="text-[9px] uppercase text-muted-foreground">Marks</p>
                          <p className="text-xs font-mono">{allBotsPositionSummary.freshMarks} / {allBotsPositionSummary.markedRows}</p>
                        </div>
                      </div>
                      <div className="flex-1 min-h-0 overflow-hidden">
                        {filteredAllPositionBook.length === 0 ? (
                          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No positions matching filters.</div>
                        ) : (
                          <ScrollArea className="h-full min-h-0 rounded-md border border-border/60 bg-card/80">
                            <div className="w-full overflow-x-auto">
                            <Table className="min-w-[1240px]">
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="text-[11px]">Market</TableHead>
                                  <TableHead className="text-[11px]">L</TableHead>
                                  <TableHead className="text-[11px]">Dir</TableHead>
                                  <TableHead className="text-[11px] text-right">Exposure</TableHead>
                                  <TableHead className="text-[11px] text-right">Avg Px</TableHead>
                                  <TableHead className="text-[11px] text-right">Mark</TableHead>
                                  <TableHead className="text-[11px] text-right">U-P&amp;L</TableHead>
                                  <TableHead className="text-[11px] text-right">Edge</TableHead>
                                  <TableHead className="text-[11px] text-right">Conf</TableHead>
                                  <TableHead className="text-[11px] text-right">Orders</TableHead>
                                  <TableHead className="text-[11px] text-right">Mode</TableHead>
                                  <TableHead className="text-[11px]">Updated</TableHead>
                                </TableRow>
                              </TableHeader>
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
                                    <TableCell className="text-right font-mono py-1">{row.weightedEdge !== null ? formatPercent(row.weightedEdge) : '—'}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.weightedConfidence !== null ? formatPercent(normalizeConfidencePercent(row.weightedConfidence)) : '—'}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.orderCount}</TableCell>
                                    <TableCell className="text-right font-mono py-1">{row.liveOrderCount}L/{row.paperOrderCount}P</TableCell>
                                    <TableCell className="py-1 text-[10px] text-muted-foreground">{formatShortDate(row.lastUpdated || row.markUpdatedAt)}</TableCell>
                                  </TableRow>
                                  )
                                })}
                              </TableBody>
                            </Table>
                            </div>
                          </ScrollArea>
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
                    {selectedTraderStatus.label}
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
                      disabled={selectedTraderControlPending || !selectedTraderCanEnable}
                      onClick={() => traderEnableMutation.mutate(selectedTrader.id)}
                    >
                      <Play className="w-3 h-3 mr-0.5" /> Enable
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[10px]"
                      disabled={selectedTraderControlPending || !selectedTraderCanDisable}
                      onClick={() => traderDisableMutation.mutate(selectedTrader.id)}
                    >
                      <Square className="w-3 h-3 mr-0.5" /> Disable
                    </Button>
                    <Button size="sm" variant="outline" className="h-6 px-2 text-[10px]" onClick={() => traderRunOnceMutation.mutate(selectedTrader.id)} disabled={traderRunOnceMutation.isPending}>
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
                  { key: 'monitor' as const, label: 'Monitor' },
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
                    </div>
                    {selectedTraderNoNewRows && (
                      <div className="shrink-0 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-700 dark:text-amber-100 mx-1">
                        No new rows since last cycle ({formatTimestamp(selectedTrader?.last_run_at || worker?.last_run_at)}).
                      </div>
                    )}
                    <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10 mx-1">
                      <div className="space-y-0.5 p-1.5 font-mono text-[11px] leading-relaxed">
                        {filteredTraderActivityRows.length === 0 ? (
                          <div className="py-8 text-center text-muted-foreground text-xs">No events matching filters.</div>
                        ) : (
                          filteredTraderActivityRows.map((row) => (
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
                          ))
                        )}
                      </div>
                    </ScrollArea>
                  </div>
                )}

                {workTab === 'trades' && (
                  <div className="h-full flex flex-col min-h-0 gap-1.5">
                    <div className="shrink-0 flex flex-wrap items-center gap-1 px-1">
                      <Input value={tradeSearch} onChange={(event) => setTradeSearch(event.target.value)} placeholder="Search..." className="h-6 w-36 text-[11px]" />
                      {(['all', 'open', 'resolved', 'failed'] as TradeStatusFilter[]).map((status) => (
                        <Button key={status} size="sm" variant={tradeStatusFilter === status ? 'default' : 'outline'} onClick={() => setTradeStatusFilter(status)} className="h-5 px-2 text-[10px]">{status}</Button>
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
                        <ScrollArea className="h-full min-h-0">
                          <div className="w-full overflow-x-auto">
                          <Table className="w-full table-fixed">
                            <TableHeader>
                              <TableRow>
                                <TableHead className="w-[26%] text-[10px]">Market</TableHead>
                                <TableHead className="w-[6%] text-[10px]">Dir</TableHead>
                                <TableHead className="w-[24%] text-[10px]">Lifecycle / Outcome</TableHead>
                                <TableHead className="w-[8%] text-[10px] text-right">Notional</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Fill</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Fill Progress</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Mark</TableHead>
                                <TableHead className="w-[8%] text-[10px] text-right">U-P&amp;L</TableHead>
                                <TableHead className="w-[7%] text-[10px] text-right">Edge Δ</TableHead>
                                <TableHead className="w-[8%] text-[10px] text-right">R-P&amp;L</TableHead>
                                <TableHead className="w-[8%] text-[10px]">Venue</TableHead>
                                <TableHead className="w-[6%] text-[10px] text-right">Exit %</TableHead>
                                <TableHead className="w-[5%] text-[10px]">Age</TableHead>
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
                                  updatedAt,
                                  providerSnapshotStatus,
                                  pendingExitStatus,
                                  closeTrigger,
                                  pendingExit,
                                  markFresh,
                                  links,
                                  directionSide,
                                  directionLabel,
                                  yesLabel,
                                  noLabel,
                                  executionSummary,
                                  outcomeHeadline,
                                  outcomeDetail,
                                  outcomeDetailCompact,
                                  venuePresentation,
                                } = row
                                const pendingExitError = pendingExit && typeof pendingExit === 'object'
                                  ? String((pendingExit as Record<string, unknown>).last_error || (pendingExit as Record<string, unknown>).error || '').trim()
                                  : ''
                                const pendingExitNextRetry = pendingExit && typeof pendingExit === 'object'
                                  ? String((pendingExit as Record<string, unknown>).next_retry_at || '').trim()
                                  : ''
                                const pendingExitLabel = pendingExitStatus === 'failed' && OPEN_ORDER_STATUSES.has(status)
                                  ? 'E:RETRY'
                                  : `E:${pendingExitStatus.slice(0, 4).toUpperCase()}`
                                const statusBadge = resolveOrderStatusBadgePresentation(status, pnl)
                                const outcomeBadgeClassName = resolveOrderOutcomeBadgeClassName(status)
                                const marketForModal = resolveCryptoMarketFromAliases(collectOrderMarketAliasIds(order))
                                return (
                                  <TableRow
                                    key={order.id}
                                    className="text-[11px] leading-tight cursor-pointer hover:bg-muted/30"
                                    onClick={() => {
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
                                    }}
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
                                        <span className="truncate">{order.market_question || shortId(order.market_id)}</span>
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
                                    <TableCell className="py-0.5 min-w-[260px]">
                                        <div className="min-w-0 space-y-0">
                                          <div className="flex min-w-0 items-center gap-0.5 overflow-hidden">
                                          <Badge
                                            variant={statusBadge.variant}
                                            title={`Raw status: ${status}`}
                                            className={cn('h-4 shrink-0 whitespace-nowrap px-1 text-[9px] font-semibold', statusBadge.className)}
                                          >
                                            {lifecycleLabel}
                                          </Badge>
                                          <Badge
                                            variant="outline"
                                            title={outcomeDetail}
                                            className={cn(
                                              'h-4 max-w-[180px] truncate px-1 text-[9px] font-medium',
                                              outcomeBadgeClassName
                                            )}
                                          >
                                            {outcomeHeadline}
                                          </Badge>
                                          {executionSummary !== '—' && (
                                            <Badge variant="outline" title={`Execution: ${executionSummary}`} className="h-4 max-w-[150px] truncate border-border/80 bg-background/80 px-1 text-[9px] font-medium text-foreground/85">
                                              {executionSummary}
                                            </Badge>
                                          )}
                                          {closeTrigger && (
                                            <Badge
                                              variant="outline"
                                              title={`Close trigger: ${closeTrigger}`}
                                              className="h-4 max-w-[150px] truncate border-sky-300 bg-sky-100 px-1 text-[9px] font-medium text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/12 dark:text-sky-200"
                                            >
                                              {`T:${closeTrigger}`}
                                            </Badge>
                                          )}
                                          {pendingExitStatus && pendingExitStatus !== 'unknown' && (
                                            <Badge
                                              variant="outline"
                                              title={
                                                `exit:${pendingExitStatus}`
                                                + (pendingExitError ? ` • ${pendingExitError}` : '')
                                                + (pendingExitNextRetry ? ` • next_retry:${formatTimestamp(pendingExitNextRetry)}` : '')
                                              }
                                              className={cn(
                                                'h-4 px-1 text-[9px] font-semibold',
                                                pendingExitStatus === 'failed'
                                                  ? 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/15 dark:text-amber-200'
                                                  : 'border-border/80 bg-background/80 text-foreground/80'
                                              )}
                                            >
                                              {pendingExitLabel}
                                            </Badge>
                                          )}
                                        </div>
                                        <p className="truncate text-[9px] leading-none text-foreground/85" title={outcomeDetail}>
                                          <span className="font-medium text-muted-foreground">Reason:</span> {outcomeDetailCompact}
                                        </p>
                                      </div>
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
                                    <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}>
                                      {OPEN_ORDER_STATUSES.has(status) ? (
                                        <FlashNumber
                                          value={unrealized}
                                          decimals={2}
                                          prefix="$"
                                          className={cn('font-mono text-[10px]', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}
                                          positiveClass="data-glow-green"
                                          negativeClass="data-glow-red"
                                        />
                                      ) : '\u2014'}
                                    </TableCell>
                                    <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatPercent(dynamicEdgePercent)}</TableCell>
                                    <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', pnl > 0 ? 'text-emerald-500' : pnl < 0 ? 'text-red-500' : '')}>{RESOLVED_ORDER_STATUSES.has(status) ? formatCurrency(pnl, true) : '\u2014'}</TableCell>
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
                                      <span title={`${String(order.mode || '').toUpperCase()} • created:${formatTimestamp(order.created_at)} • updated:${formatTimestamp(updatedAt)}`}>
                                        {formatRelativeAge(updatedAt)}
                                      </span>
                                    </TableCell>
                                  </TableRow>
                                )
                              })}
                            </TableBody>
                          </Table>
                          </div>
                        </ScrollArea>
                      )}
                    </div>
                  </div>
                )}

                {workTab === 'monitor' && (
                  <div className="h-full min-h-0 overflow-hidden px-1">
                    <ScrollArea className="h-full min-h-0 rounded-md border border-border/50 bg-muted/10">
                      <div className="space-y-2 p-2">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <p className="text-[11px] font-medium">Live Truth Monitor</p>
                          <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                            {liveTruthJobId ? shortId(liveTruthJobId) : 'idle'}
                          </Badge>
                          {liveTruthJobId ? (
                            <>
                              <span
                                className={cn(
                                  'rounded px-1.5 py-0.5 text-[9px] font-semibold',
                                  liveTruthJobFailed
                                    ? 'bg-red-500/15 text-red-500'
                                    : liveTruthJobCompleted
                                      ? 'bg-emerald-500/15 text-emerald-500'
                                      : 'bg-amber-500/15 text-amber-500'
                                )}
                              >
                                {liveTruthJobStatus.toUpperCase()}
                              </span>
                              <span className="text-[9px] text-muted-foreground">{liveTruthJobProgressPct}%</span>
                            </>
                          ) : null}
                        </div>

                        <p className="text-[10px] text-muted-foreground/80">
                          Run live-trading lifecycle diagnostics with optional provider checks and optional LLM analysis.
                        </p>

                        {!selectedTrader ? (
                          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-700 dark:text-amber-100">
                            Select a bot to run monitor diagnostics.
                          </div>
                        ) : null}

                        <div className="grid gap-2 md:grid-cols-3">
                          <div>
                            <Label className="text-[11px] text-muted-foreground">Duration (seconds)</Label>
                            <Input
                              type="number"
                              min={10}
                              max={7200}
                              value={liveTruthDurationSeconds}
                              onChange={(event) => setLiveTruthDurationSeconds(event.target.value)}
                              className="mt-1 h-8 text-xs font-mono"
                            />
                          </div>
                          <div>
                            <Label className="text-[11px] text-muted-foreground">Poll (seconds)</Label>
                            <Input
                              type="number"
                              min={0.2}
                              max={10}
                              step={0.1}
                              value={liveTruthPollSeconds}
                              onChange={(event) => setLiveTruthPollSeconds(event.target.value)}
                              className="mt-1 h-8 text-xs font-mono"
                            />
                          </div>
                          <div>
                            <Label className="text-[11px] text-muted-foreground">Max Alerts To LLM</Label>
                            <Input
                              type="number"
                              min={1}
                              max={400}
                              value={liveTruthMaxAlertsForLlm}
                              onChange={(event) => setLiveTruthMaxAlertsForLlm(event.target.value)}
                              className="mt-1 h-8 text-xs font-mono"
                              disabled={!liveTruthRunLlm}
                            />
                          </div>
                        </div>

                        <div className="grid gap-2 md:grid-cols-3">
                          <div className="rounded-md border border-border/60 p-2">
                            <div className="flex items-center justify-between">
                              <p className="text-xs font-medium">Provider Deep Checks</p>
                              <Switch checked={liveTruthEnableProviderChecks} onCheckedChange={setLiveTruthEnableProviderChecks} />
                            </div>
                            <p className="mt-1 text-[10px] text-muted-foreground/80">
                              Enables wallet/provider API reconciliation checks. Leave off to avoid impacting active trading.
                            </p>
                          </div>
                          <div className="rounded-md border border-border/60 p-2">
                            <div className="flex items-center justify-between">
                              <p className="text-xs font-medium">Analyze with LLM</p>
                              <Switch checked={liveTruthRunLlm} onCheckedChange={setLiveTruthRunLlm} />
                            </div>
                            <p className="mt-1 text-[10px] text-muted-foreground/80">
                              Sends monitor summary + alert samples to configured LLM provider.
                            </p>
                          </div>
                          <div className="rounded-md border border-border/60 p-2">
                            <div className="flex items-center justify-between">
                              <p className="text-xs font-medium">Include Strategy Source</p>
                              <Switch
                                checked={liveTruthIncludeStrategySource}
                                onCheckedChange={setLiveTruthIncludeStrategySource}
                                disabled={!liveTruthRunLlm}
                              />
                            </div>
                            <p className="mt-1 text-[10px] text-muted-foreground/80">
                              Adds active strategy code to LLM context for file-level change suggestions.
                            </p>
                          </div>
                        </div>

                        <div>
                          <Label className="text-[11px] text-muted-foreground">LLM Model (optional override)</Label>
                          <Input
                            value={liveTruthModel}
                            onChange={(event) => setLiveTruthModel(event.target.value)}
                            placeholder="Use app default model"
                            className="mt-1 h-8 text-xs font-mono"
                            disabled={!liveTruthRunLlm}
                          />
                        </div>

                        <Button
                          size="sm"
                          className="h-8 text-xs"
                          onClick={() => runLiveTruthMonitorMutation.mutate()}
                          disabled={runLiveTruthMonitorMutation.isPending || liveTruthJobRunning || !selectedTrader}
                        >
                          {runLiveTruthMonitorMutation.isPending || liveTruthJobRunning ? (
                            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          ) : (
                            <Play className="w-3.5 h-3.5 mr-1.5" />
                          )}
                          {liveTruthJobRunning ? 'Monitoring...' : 'Run Monitor'}
                        </Button>

                        {liveTruthJobId ? (
                          <div className="rounded-md border border-border/60 bg-muted/20 p-2">
                            <p className="text-[10px] text-muted-foreground/80">
                              {String(liveTruthJobQuery.data?.message || 'No job status message')}
                            </p>
                            {liveTruthJobQuery.data?.error ? (
                              <p className="mt-1 text-[10px] text-red-500">{String(liveTruthJobQuery.data.error)}</p>
                            ) : null}
                          </div>
                        ) : null}

                        {liveTruthError ? (
                          <p className="text-[10px] text-red-500">{liveTruthError}</p>
                        ) : null}

                        {liveTruthRawQuery.isPending ? (
                          <p className="text-[10px] text-muted-foreground/80">Loading monitor artifacts...</p>
                        ) : null}
                        {liveTruthRawQuery.error ? (
                          <p className="text-[10px] text-red-500">
                            {errorMessage(liveTruthRawQuery.error, 'Failed to load monitor artifacts')}
                          </p>
                        ) : null}

                        {liveTruthReport ? (
                          <div className="rounded-md border border-emerald-500/25 bg-emerald-500/5 p-2 space-y-2">
                            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
                              <span>Alerts: {Math.trunc(toNumber(liveTruthReport.alert_count))}</span>
                              <span>Heartbeats: {Math.trunc(toNumber(liveTruthReport.heartbeat_count))}</span>
                              <span>Transitions: {Math.trunc(toNumber(liveTruthReport.transition_count))}</span>
                              <span>Lines: {Math.trunc(toNumber(liveTruthReport.line_count))}</span>
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {liveTruthAlertRuleRows.slice(0, 8).map(([rule, count]) => (
                                <span key={rule} className="rounded-full bg-background/70 border border-border/70 px-1.5 py-0.5 text-[10px]">
                                  {rule}: {Math.trunc(toNumber(count))}
                                </span>
                              ))}
                              {liveTruthAlertRuleRows.length === 0 ? (
                                <span className="text-[10px] text-muted-foreground/80">No alert rules triggered.</span>
                              ) : null}
                            </div>
                            <p className="text-[10px] text-muted-foreground/80 font-mono break-all">
                              {String(liveTruthRaw?.monitor?.summary_path || liveTruthRaw?.monitor?.report_path || '')}
                            </p>
                            <p className="text-[10px] text-muted-foreground/80">
                              Trader: {String(liveTruthSummary.target_trader_name || selectedTrader?.name || 'n/a')} ({String(liveTruthSummary.target_trader_id || selectedTrader?.id || 'n/a')})
                            </p>
                          </div>
                        ) : null}

                        {liveTruthHasLlm ? (
                          <div className="rounded-md border border-sky-500/25 bg-sky-500/5 p-2 space-y-1.5">
                            <p className="text-[11px] font-medium">
                              LLM: {String(liveTruthLlm.status || 'unknown')}
                              {liveTruthLlm.requested_model ? ` (${String(liveTruthLlm.requested_model)})` : ''}
                            </p>
                            {liveTruthLlm.error ? (
                              <p className="text-[10px] text-red-500">{String(liveTruthLlm.error)}</p>
                            ) : null}
                            {liveTruthLlmAnalysis && typeof liveTruthLlmAnalysis.assessment === 'string' ? (
                              <p className="text-[10px] text-muted-foreground/85">{liveTruthLlmAnalysis.assessment}</p>
                            ) : null}
                            {liveTruthLlmStrategyChanges.length > 0 ? (
                              <div className="space-y-1">
                                {liveTruthLlmStrategyChanges.slice(0, 3).map((change, index) => (
                                  <div key={`${index}-${String(change.title || '')}`} className="rounded border border-border/60 bg-background/70 px-2 py-1">
                                    <p className="text-[10px] font-medium">{String(change.title || 'Strategy change')}</p>
                                    <p className="text-[10px] text-muted-foreground/80">{String(change.target || '')}</p>
                                  </div>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        ) : null}

                        {liveTruthJobCompleted && liveTruthJobId ? (
                          <div className="flex flex-wrap gap-1.5">
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className="h-6 px-2 text-[10px]"
                              onClick={() => exportLiveTruthArtifactMutation.mutate('summary_json')}
                              disabled={exportLiveTruthArtifactMutation.isPending}
                            >
                              Export Summary
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className="h-6 px-2 text-[10px]"
                              onClick={() => exportLiveTruthArtifactMutation.mutate('report_jsonl')}
                              disabled={exportLiveTruthArtifactMutation.isPending}
                            >
                              Export Report
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className="h-6 px-2 text-[10px]"
                              onClick={() => exportLiveTruthArtifactMutation.mutate('llm_analysis_json')}
                              disabled={exportLiveTruthArtifactMutation.isPending}
                            >
                              Export LLM
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className="h-6 px-2 text-[10px]"
                              onClick={() => exportLiveTruthArtifactMutation.mutate('bundle_json')}
                              disabled={exportLiveTruthArtifactMutation.isPending}
                            >
                              Export Bundle
                            </Button>
                          </div>
                        ) : null}

                        <div className="rounded-md border border-cyan-500/30 bg-cyan-500/5 p-2.5 space-y-2">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <p className="text-[11px] font-medium">Iterate Agent</p>
                            <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                              {monitorIterateResponse?.session_id ? shortId(monitorIterateResponse.session_id) : 'idle'}
                            </Badge>
                            {runMonitorIterateMutation.isPending ? (
                              <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold bg-amber-500/15 text-amber-500">
                                RUNNING
                              </span>
                            ) : null}
                          </div>
                          <p className="text-[10px] text-muted-foreground/80">
                            Tool-based LLM loop with direct access to trader decisions, strategy versions, parameter tuning, source edits, and A/B experiment creation.
                          </p>
                          <textarea
                            value={monitorIteratePrompt}
                            onChange={(event) => setMonitorIteratePrompt(event.target.value)}
                            className="w-full min-h-[96px] rounded-md border border-border/60 bg-background px-2 py-1.5 text-xs leading-relaxed"
                            placeholder="Describe what should be improved, what to optimize, and any constraints..."
                          />
                          <div className="grid gap-2 md:grid-cols-3">
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Model Override</Label>
                              <Input
                                value={monitorIterateModel}
                                onChange={(event) => setMonitorIterateModel(event.target.value)}
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
                                value={monitorIterateMaxIterations}
                                onChange={(event) => setMonitorIterateMaxIterations(event.target.value)}
                                className="mt-1 h-8 text-xs font-mono"
                              />
                            </div>
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Monitor Job Context</Label>
                              <Input
                                value={monitorIterateMonitorJobId}
                                onChange={(event) => setMonitorIterateMonitorJobId(event.target.value)}
                                placeholder={liveTruthJobId || 'optional job id'}
                                className="mt-1 h-8 text-xs font-mono"
                              />
                            </div>
                          </div>
                          <Button
                            size="sm"
                            className="h-8 text-xs"
                            onClick={() => runMonitorIterateMutation.mutate()}
                            disabled={runMonitorIterateMutation.isPending || !selectedTrader}
                          >
                            {runMonitorIterateMutation.isPending ? (
                              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                            ) : (
                              <Sparkles className="w-3.5 h-3.5 mr-1.5" />
                            )}
                            {runMonitorIterateMutation.isPending ? 'Iterating...' : 'Run Iterate'}
                          </Button>
                          {monitorIterateError ? (
                            <p className="text-[10px] text-red-500">{monitorIterateError}</p>
                          ) : null}
                          {monitorIterateResponse ? (
                            <div className="rounded-md border border-border/60 bg-background/70 p-2 space-y-1.5">
                              {monitorIterateParsed && typeof monitorIterateParsed.summary === 'string' ? (
                                <p className="text-[10px] text-muted-foreground/90">
                                  {String(monitorIterateParsed.summary)}
                                </p>
                              ) : null}
                              {monitorIterateActions.length > 0 ? (
                                <div className="space-y-0.5">
                                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Actions Taken</p>
                                  {monitorIterateActions.slice(0, 8).map((action, index) => (
                                    <p key={`${index}-${action}`} className="text-[10px] text-foreground/90">
                                      {index + 1}. {action}
                                    </p>
                                  ))}
                                </div>
                              ) : null}
                              {monitorIterateNextSteps.length > 0 ? (
                                <div className="space-y-0.5">
                                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Suggested Next</p>
                                  {monitorIterateNextSteps.slice(0, 8).map((step, index) => (
                                    <p key={`${index}-${step}`} className="text-[10px] text-muted-foreground/90">
                                      {index + 1}. {step}
                                    </p>
                                  ))}
                                </div>
                              ) : null}
                              <details className="rounded border border-border/50 bg-background/50 px-2 py-1">
                                <summary className="cursor-pointer text-[10px] text-muted-foreground">Raw agent response</summary>
                                <pre className="mt-1 whitespace-pre-wrap break-words text-[10px] font-mono text-muted-foreground/90">
                                  {monitorIterateResponse.answer || JSON.stringify(monitorIterateResponse.raw, null, 2)}
                                </pre>
                              </details>
                            </div>
                          ) : null}
                        </div>

                        <div className="rounded-md border border-violet-500/30 bg-violet-500/5 p-2.5 space-y-2">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <p className="text-[11px] font-medium">A/B Version Lab</p>
                              <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                                {filteredStrategyExperiments.length} experiments
                              </Badge>
                            </div>
                            <Select value={experimentFilterStatus} onValueChange={setExperimentFilterStatus}>
                              <SelectTrigger className="h-7 w-[130px] text-[10px]">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="all">All Statuses</SelectItem>
                                <SelectItem value="active">Active</SelectItem>
                                <SelectItem value="paused">Paused</SelectItem>
                                <SelectItem value="completed">Completed</SelectItem>
                                <SelectItem value="archived">Archived</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          <div className="grid gap-2 md:grid-cols-2">
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Source</Label>
                              <Select value={experimentDraftSourceKey || '__none__'} onValueChange={(value) => setExperimentDraftSourceKey(value === '__none__' ? '' : value)}>
                                <SelectTrigger className="mt-1 h-8 text-xs">
                                  <SelectValue placeholder="Select source" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="__none__">Select source</SelectItem>
                                  {experimentSourceChoices.map((sourceKey) => (
                                    <SelectItem key={`exp-source-${sourceKey}`} value={sourceKey}>
                                      {sourceCards.find((source) => normalizeSourceKey(source.key) === sourceKey)?.label || sourceKey}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Strategy</Label>
                              <Select value={experimentDraftStrategyKey || '__none__'} onValueChange={(value) => setExperimentDraftStrategyKey(value === '__none__' ? '' : value)}>
                                <SelectTrigger className="mt-1 h-8 text-xs">
                                  <SelectValue placeholder="Select strategy" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="__none__">Select strategy</SelectItem>
                                  {experimentStrategyChoices.map((option) => (
                                    <SelectItem key={`exp-strategy-${option.key}`} value={option.key}>
                                      {option.label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          </div>

                          <div className="grid gap-2 md:grid-cols-3">
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Control Version</Label>
                              <Select value={experimentDraftControlVersion || '__none__'} onValueChange={(value) => setExperimentDraftControlVersion(value === '__none__' ? '' : value)}>
                                <SelectTrigger className="mt-1 h-8 text-xs font-mono">
                                  <SelectValue placeholder="v?" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="__none__">Select</SelectItem>
                                  {experimentVersionChoices.map((version) => (
                                    <SelectItem key={`exp-control-v${version}`} value={String(version)}>
                                      {`v${version}`}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Candidate Version</Label>
                              <Select value={experimentDraftCandidateVersion || '__none__'} onValueChange={(value) => setExperimentDraftCandidateVersion(value === '__none__' ? '' : value)}>
                                <SelectTrigger className="mt-1 h-8 text-xs font-mono">
                                  <SelectValue placeholder="v?" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="__none__">Select</SelectItem>
                                  {experimentVersionChoices.map((version) => (
                                    <SelectItem key={`exp-candidate-v${version}`} value={String(version)}>
                                      {`v${version}`}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Candidate %</Label>
                              <Input
                                type="number"
                                min={0.1}
                                max={99.9}
                                step={0.1}
                                value={experimentDraftAllocationPct}
                                onChange={(event) => setExperimentDraftAllocationPct(event.target.value)}
                                className="mt-1 h-8 text-xs font-mono"
                              />
                            </div>
                          </div>

                          <div className="grid gap-2 md:grid-cols-2">
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Experiment Name</Label>
                              <Input
                                value={experimentDraftName}
                                onChange={(event) => setExperimentDraftName(event.target.value)}
                                placeholder="auto-generated if blank"
                                className="mt-1 h-8 text-xs"
                              />
                            </div>
                            <div>
                              <Label className="text-[11px] text-muted-foreground">Notes</Label>
                              <Input
                                value={experimentDraftNotes}
                                onChange={(event) => setExperimentDraftNotes(event.target.value)}
                                placeholder="optional"
                                className="mt-1 h-8 text-xs"
                              />
                            </div>
                          </div>

                          <Button
                            size="sm"
                            variant="outline"
                            className="h-8 text-xs"
                            onClick={() => createStrategyExperimentMutation.mutate()}
                            disabled={!selectedTrader || createStrategyExperimentMutation.isPending}
                          >
                            {createStrategyExperimentMutation.isPending ? (
                              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                            ) : (
                              <Plus className="w-3.5 h-3.5 mr-1.5" />
                            )}
                            Create Experiment
                          </Button>

                          {experimentActionError ? (
                            <p className="text-[10px] text-red-500">{experimentActionError}</p>
                          ) : null}

                          {strategyExperimentsQuery.isPending ? (
                            <p className="text-[10px] text-muted-foreground/80">Loading experiments...</p>
                          ) : null}
                          {strategyExperimentsQuery.error ? (
                            <p className="text-[10px] text-red-500">
                              {errorMessage(strategyExperimentsQuery.error, 'Failed to load experiments')}
                            </p>
                          ) : null}

                          <div className="space-y-1">
                            {filteredStrategyExperiments.length === 0 ? (
                              <p className="text-[10px] text-muted-foreground/80">No experiments for this bot sources yet.</p>
                            ) : (
                              filteredStrategyExperiments.slice(0, 12).map((row) => {
                                const status = String(row.status || '').trim().toLowerCase()
                                const isSelected = row.id === selectedExperimentId
                                return (
                                  <div
                                    key={row.id}
                                    className={cn(
                                      'rounded border px-2 py-1.5',
                                      isSelected ? 'border-cyan-500/45 bg-cyan-500/10' : 'border-border/60 bg-background/60'
                                    )}
                                  >
                                    <div className="flex flex-wrap items-center justify-between gap-1">
                                      <div className="min-w-0">
                                        <p className="truncate text-[10px] font-medium">
                                          {row.name}
                                        </p>
                                        <p className="text-[10px] text-muted-foreground/85 font-mono">
                                          {`${row.source_key}:${row.strategy_key} • v${row.control_version} vs v${row.candidate_version} • ${row.candidate_allocation_pct.toFixed(1)}%`}
                                        </p>
                                      </div>
                                      <Badge
                                        variant="outline"
                                        className={cn(
                                          'h-4 px-1.5 text-[9px]',
                                          status === 'active'
                                            ? 'border-emerald-500/35 text-emerald-300 bg-emerald-500/10'
                                            : status === 'paused'
                                              ? 'border-amber-500/35 text-amber-300 bg-amber-500/10'
                                              : 'border-border/60 text-muted-foreground bg-background/70'
                                        )}
                                      >
                                        {status || 'unknown'}
                                      </Badge>
                                    </div>
                                    <div className="mt-1 flex flex-wrap gap-1">
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        className="h-5 px-1.5 text-[9px]"
                                        onClick={() => setSelectedExperimentId(row.id)}
                                      >
                                        Inspect
                                      </Button>
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        className="h-5 px-1.5 text-[9px]"
                                        disabled={setExperimentStatusMutation.isPending}
                                        onClick={() =>
                                          setExperimentStatusMutation.mutate({
                                            experimentId: row.id,
                                            status: status === 'active' ? 'paused' : 'active',
                                          })
                                        }
                                      >
                                        {status === 'active' ? 'Pause' : 'Activate'}
                                      </Button>
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        className="h-5 px-1.5 text-[9px]"
                                        disabled={setExperimentStatusMutation.isPending || status === 'completed' || status === 'archived'}
                                        onClick={() =>
                                          setExperimentStatusMutation.mutate({
                                            experimentId: row.id,
                                            status: 'completed',
                                          })
                                        }
                                      >
                                        Complete
                                      </Button>
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        className="h-5 px-1.5 text-[9px]"
                                        disabled={promoteStrategyExperimentMutation.isPending}
                                        onClick={() =>
                                          promoteStrategyExperimentMutation.mutate({
                                            experimentId: row.id,
                                          })
                                        }
                                      >
                                        Promote Candidate
                                      </Button>
                                    </div>
                                  </div>
                                )
                              })
                            )}
                          </div>

                          {selectedStrategyExperiment ? (
                            <div className="rounded border border-border/60 bg-background/70 p-2 space-y-1">
                              <p className="text-[10px] font-medium">
                                Assignments: {selectedStrategyExperiment.name}
                              </p>
                              <p className="text-[10px] text-muted-foreground/85 font-mono">
                                control:{selectedExperimentAssignmentSummary.control}
                                {' • '}
                                candidate:{selectedExperimentAssignmentSummary.candidate}
                                {selectedExperimentAssignmentSummary.other > 0
                                  ? ` • other:${selectedExperimentAssignmentSummary.other}`
                                  : ''}
                                {' • '}
                                total:{selectedExperimentAssignmentSummary.total}
                              </p>
                              {experimentAssignmentsQuery.isPending ? (
                                <p className="text-[10px] text-muted-foreground/80">Loading assignments...</p>
                              ) : null}
                              {experimentAssignmentsQuery.error ? (
                                <p className="text-[10px] text-red-500">
                                  {errorMessage(experimentAssignmentsQuery.error, 'Failed to load assignments')}
                                </p>
                              ) : null}
                              {selectedExperimentAssignments.slice(0, 10).map((assignment) => (
                                <div key={assignment.id} className="rounded border border-border/50 bg-background/80 px-1.5 py-1 text-[10px] font-mono">
                                  {`${assignment.assignment_group} • v${assignment.strategy_version} • trader:${shortId(assignment.trader_id)} • signal:${shortId(assignment.signal_id)}`}
                                </div>
                              ))}
                              {selectedExperimentAssignments.length === 0 && !experimentAssignmentsQuery.isPending ? (
                                <p className="text-[10px] text-muted-foreground/80">No assignments yet.</p>
                              ) : null}
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </ScrollArea>
                  </div>
                )}

                {workTab === 'decisions' && (
                  <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] px-1">
                    <div className="flex flex-col gap-1.5 min-h-0 overflow-hidden">
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
                        <div className="space-y-0.5 p-1.5 text-xs">
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
                                    'w-full text-left rounded border px-2 py-1 transition-colors',
                                    isActive ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-border/50 hover:bg-muted/40',
                                    outcome === 'selected' && !isActive ? 'border-emerald-500/25' :
                                    outcome === 'blocked' && !isActive ? 'border-red-500/25' : ''
                                  )}
                                >
                                  <div className="flex items-center justify-between gap-2 font-mono">
                                    <span className="truncate max-w-[180px]" title={marketLabel}>{marketLabel}</span>
                                    <Badge variant={outcome === 'selected' ? 'default' : outcome === 'blocked' ? 'destructive' : 'outline'} className="text-[9px] h-4 px-1 shrink-0">{outcome}</Badge>
                                  </div>
                                  <p className="text-[10px] text-muted-foreground truncate">{decision.reason || decision.strategy_key}</p>
                                </button>
                              )
                            })
                          )}
                        </div>
                      </ScrollArea>
                    </div>

                    <div className="flex flex-col gap-1.5 min-h-0 overflow-hidden">
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

                          {decisionChecks.length > 0 ? (
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

                    <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-2">
                      <div className="min-h-0 rounded-md border border-border/60 bg-card/60">
                        <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Timeframe Performance</div>
                        <ScrollArea className="h-[32%] min-h-[150px]">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="text-[10px]">Timeframe</TableHead>
                                <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                <TableHead className="text-[10px] text-right">ROI</TableHead>
                                <TableHead className="text-[10px] text-right">W/L</TableHead>
                                <TableHead className="text-[10px] text-right">Full-Loss</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedPerformance.timeframeRows.map((row) => (
                                <TableRow key={`tf-${row.key}`} className="text-xs">
                                  <TableCell className="font-mono py-1">{row.label}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                    {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                  </TableCell>
                                  <TableCell className="text-right font-mono py-1">{row.wins}/{row.losses}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.fullLosses > 0 && 'text-red-500')}>
                                    {row.fullLosses > 0 ? `${row.fullLosses}/${row.resolved}` : '—'}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                        <div className="px-2 py-1 border-y border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Mode Performance</div>
                        <ScrollArea className="h-[32%] min-h-[150px]">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="text-[10px]">Mode</TableHead>
                                <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                <TableHead className="text-[10px] text-right">ROI</TableHead>
                                <TableHead className="text-[10px] text-right">Resolved</TableHead>
                                <TableHead className="text-[10px] text-right">Full-Loss</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedPerformance.modeRows.map((row) => (
                                <TableRow key={`mode-${row.key}`} className="text-xs">
                                  <TableCell className="font-mono py-1">{row.label}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                    {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                  </TableCell>
                                  <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.fullLosses > 0 && 'text-red-500')}>
                                    {row.fullLosses > 0 ? `${row.fullLosses}/${row.resolved}` : '—'}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                        <div className="px-2 py-1 border-y border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Mode + Timeframe</div>
                        <ScrollArea className="h-[32%] min-h-[150px]">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="text-[10px]">Bucket</TableHead>
                                <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                <TableHead className="text-[10px] text-right">ROI</TableHead>
                                <TableHead className="text-[10px] text-right">Resolved</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedPerformance.timeframeModeRows.slice(0, 24).map((row) => (
                                <TableRow key={`combo-${row.key}`} className="text-xs">
                                  <TableCell className="font-mono py-1">{row.label}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                    {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                  </TableCell>
                                  <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      </div>

                      <div className="min-h-0 rounded-md border border-border/60 bg-card/60">
                        <div className="px-2 py-1 border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Strategy Performance</div>
                        <ScrollArea className="h-[49%] min-h-[220px]">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="text-[10px]">Strategy</TableHead>
                                <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                <TableHead className="text-[10px] text-right">ROI</TableHead>
                                <TableHead className="text-[10px] text-right">Resolved</TableHead>
                                <TableHead className="text-[10px] text-right">W/L</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedPerformance.strategyRows.map((row) => (
                                <TableRow key={`strategy-${row.key}`} className="text-xs">
                                  <TableCell className="py-1">
                                    <div className="font-medium">{row.label}</div>
                                    <div className="text-[9px] font-mono text-muted-foreground">{row.key}</div>
                                  </TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                    {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                  </TableCell>
                                  <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                  <TableCell className="text-right font-mono py-1">{row.wins}/{row.losses}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                        <div className="px-2 py-1 border-y border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">Sub-Strategy Performance</div>
                        <ScrollArea className="h-[49%] min-h-[220px]">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="text-[10px]">Sub-Strategy</TableHead>
                                <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                                <TableHead className="text-[10px] text-right">ROI</TableHead>
                                <TableHead className="text-[10px] text-right">Resolved</TableHead>
                                <TableHead className="text-[10px] text-right">Full-Loss</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedPerformance.subStrategyRows.slice(0, 24).map((row) => (
                                <TableRow key={`sub-${row.key}`} className="text-xs">
                                  <TableCell className="font-mono py-1">{row.label}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>{formatCurrency(row.pnl)}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.roiPercent > 0 ? 'text-emerald-500' : row.roiPercent < 0 ? 'text-red-500' : '')}>
                                    {row.roiPercent > 0 ? '+' : ''}{formatPercent(row.roiPercent, 2)}
                                  </TableCell>
                                  <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                  <TableCell className={cn('text-right font-mono py-1', row.fullLosses > 0 && 'text-red-500')}>
                                    {row.fullLosses > 0 ? `${row.fullLosses}/${row.resolved}` : '—'}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      </div>
                    </div>
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
	                  scope={marketModalState.scope}
	                  orders={allOrders}
	                  themeMode={themeMode}
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
              <span className={cn('font-mono', killSwitchOn ? 'text-red-500' : 'text-emerald-600')}>
                {killSwitchOn ? 'ON' : 'OFF'}
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
                  Configure orchestrator-wide live/paper runtime controls, risk clamps, and pending-exit behavior.
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
                      <Label>Max Open Positions Cap</Label>
                      <Input
                        type="number"
                        min={1}
                        value={globalSettingsDraft.maxOpenPositionsCap}
                        onChange={(event) => setGlobalSettingsField('maxOpenPositionsCap', event.target.value)}
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
            setDeleteConfirmName('')
            setLiveTruthError(null)
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
                          variant={draftCopyFromMode === 'paper' ? 'default' : 'outline'}
                          className="h-6 px-2 text-[10px]"
                          onClick={() => setDraftCopyFromMode('paper')}
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
                              <div className="mt-1.5 flex items-center justify-between gap-1.5">
                                <Badge
                                  variant="outline"
                                  className="h-4 px-1.5 text-[9px] font-mono border-emerald-500/30 text-emerald-300 bg-emerald-500/10"
                                >
                                  {latestVersion != null ? `Latest v${latestVersion}` : 'Latest'}
                                </Badge>
                                <Select
                                  value={selectedVersionToken}
                                  onValueChange={(value) => setSourceStrategyVersion(sourceKey, value)}
                                >
                                  <SelectTrigger className="h-7 w-[142px] text-[10px] font-mono">
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

                  {dynamicStrategyParamSections.length > 0 && (
                    <div className="rounded-lg border border-cyan-500/25 bg-cyan-500/5 p-2.5 space-y-2 mt-2">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-cyan-400">Dynamic Parameters</p>
                        <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300">
                          {dynamicStrategyParamSections.reduce((sum, section) => sum + section.fieldKeys.length, 0)} fields
                        </span>
                      </div>
                      <p className="text-[10px] text-muted-foreground/75">
                        Strategy and wallet-signal controls are grouped automatically based on active source selections.
                      </p>
                      <div className="space-y-2">
                        {dynamicStrategyParamSections.map((section) => (
                          <details key={section.sectionKey} className="rounded-md border border-border/60 bg-background/50 p-2">
                            <summary className="cursor-pointer text-[11px] font-medium flex items-center justify-between">
                              <span>{section.sourceLabel} · {section.strategyLabel}</span>
                              <span className="text-[9px] text-muted-foreground">{section.fieldKeys.length} fields</span>
                            </summary>
                            <div className="mt-2 space-y-2">
                              {section.groups.map((group) => (
                                <details key={`${section.sectionKey}:${group.key}`} className="rounded-md border border-border/50 bg-background/70 p-2" open={group.key === 'scope'}>
                                  <summary className="cursor-pointer text-[10px] font-medium flex items-center justify-between">
                                    <span>{group.label}</span>
                                    <span className="text-[9px] text-muted-foreground">{group.fields.length}</span>
                                  </summary>
                                  <div className="mt-2">
                                    <StrategyConfigForm
                                      schema={{ param_fields: group.fields as any[] }}
                                      values={section.values}
                                      onChange={(nextValues) => {
                                        if (section.kind === 'strategy') {
                                          applyDynamicStrategyFormValues(section.fieldKeys, nextValues)
                                          return
                                        }
                                        if (section.kind === 'signal_filters') {
                                          applySignalFilterFormValues(nextValues)
                                          return
                                        }
                                        applyCopyTradingFormValues(nextValues)
                                      }}
                                    />
                                  </div>
                                </details>
                              ))}
                              {section.kind === 'copy_trading' && activeCopyMode && activeCopyMode.stats && activeCopyMode.mode !== 'disabled' && (
                                <div className="grid grid-cols-4 gap-2 pt-1">
                                  <div className="text-center">
                                    <p className="text-[10px] text-muted-foreground">Copied</p>
                                    <p className="text-xs font-medium">{activeCopyMode.stats.total_copied}</p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-[10px] text-muted-foreground">Success</p>
                                    <p className="text-xs font-medium text-green-400">{activeCopyMode.stats.successful_copies}</p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-[10px] text-muted-foreground">Failed</p>
                                    <p className="text-xs font-medium text-red-400">{activeCopyMode.stats.failed_copies}</p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-[10px] text-muted-foreground">PnL</p>
                                    <p className={cn('text-xs font-medium', activeCopyMode.stats.total_pnl >= 0 ? 'text-green-400' : 'text-red-400')}>
                                      ${activeCopyMode.stats.total_pnl.toFixed(2)}
                                    </p>
                                  </div>
                                </div>
                              )}
                            </div>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}

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
                    count={`${selectedTraderOpenLiveOrders + selectedTraderOpenPaperOrders} open orders`}
                    defaultOpen={false}
                  >
                    <p className="text-xs text-muted-foreground">
                      Open live orders: {selectedTraderOpenLiveOrders} • Open paper orders: {selectedTraderOpenPaperOrders}
                    </p>
                    <Select
                      value={deleteAction}
                      onValueChange={(value) => setDeleteAction(value as 'block' | 'disable' | 'force_delete')}
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
                      <div>
                        <p className="text-[11px] text-amber-500/90 mb-1">
                          Override mode: use when live orders were already closed manually on Polymarket.
                        </p>
                        <Label className="text-xs">
                          Type bot name to confirm force delete: <span className="font-mono">{selectedTrader.name}</span>
                        </Label>
                        <Input
                          value={deleteConfirmName}
                          onChange={(event) => setDeleteConfirmName(event.target.value)}
                          className="mt-1"
                        />
                      </div>
                    ) : null}
                    <Button
                      variant="destructive"
                      className="h-8 text-xs"
                      disabled={
                        deleteTraderMutation.isPending ||
                        (deleteAction === 'force_delete' && deleteConfirmName !== selectedTrader.name)
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
