import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  Copy,
  ExternalLink,
  Filter,
  Loader2,
  Play,
  Plus,
  Settings,
  ShieldAlert,
  Sparkles,
  Square,
  Zap,
} from 'lucide-react'
import {
  armTraderOrchestratorLiveStart,
  createTrader,
  deleteTrader,
  getAllTraderOrders,
  getTraderOrders,
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
  getWallets,
  createCopyConfig,
  disableCopyConfig,
  runTraderOnce,
  runTraderOrchestratorLivePreflight,
  setTraderOrchestratorLiveKillSwitch,
  startTraderOrchestrator,
  startTraderOrchestratorLive,
  stopTraderOrchestrator,
  stopTraderOrchestratorLive,
  enqueueLiveTruthMonitorJob,
  exportLiveTruthMonitorArtifact,
  type Trader,
  type TraderConfigSchema,
  type TraderEvent,
  type TraderOrder,
  type TraderSourceConfig,
  type TraderSource,
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
} from '../services/api'
import { discoveryApi } from '../services/discoveryApi'
import { cn } from '../lib/utils'
import { getOpportunityPlatformLinks, getTraderOrderPlatformLinks } from '../lib/marketUrls'
import { selectedAccountIdAtom } from '../store/atoms'
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
import StrategyConfigForm from './StrategyConfigForm'

type FeedFilter = 'all' | 'decision' | 'order' | 'event'
type TradeStatusFilter = 'all' | 'open' | 'resolved' | 'failed'
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

type PositionBookRow = {
  key: string
  traderId: string
  traderName: string
  marketId: string
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

type TraderRuntimeStatus = 'running' | 'paused_engine' | 'disabled'

type TraderStatusPresentation = {
  key: TraderRuntimeStatus
  label: string
  dotClassName: string
  badgeVariant: 'default' | 'secondary' | 'outline'
  badgeClassName: string
}

const CRYPTO_STRATEGY_MODES = ['auto', 'directional', 'pure_arb', 'rebalance'] as const
const CRYPTO_ASSET_OPTIONS = ['BTC', 'ETH', 'SOL', 'XRP'] as const
const CRYPTO_STRATEGY_OPTIONS = [
  { key: 'btc_eth_highfreq', label: 'Crypto High Frequency', timeframe: null },
  { key: 'crypto_spike_reversion', label: 'Crypto Spike Reversion', timeframe: null },
] as const
const CRYPTO_TIMEFRAME_OPTIONS = ['5m', '15m', '1h', '4h'] as const
const CRYPTO_MODE_PARAM_KEYS = ['strategy_mode', 'mode'] as const
const CRYPTO_INCLUDE_ASSET_PARAM_KEYS = ['include_assets'] as const
const CRYPTO_EXCLUDE_ASSET_PARAM_KEYS = ['exclude_assets'] as const
const CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS = ['include_timeframes'] as const
const CRYPTO_EXCLUDE_TIMEFRAME_PARAM_KEYS = ['exclude_timeframes'] as const
type TradersScopeMode = 'tracked' | 'pool' | 'individual' | 'group'
type TradingScheduleDay = 'mon' | 'tue' | 'wed' | 'thu' | 'fri' | 'sat' | 'sun'

type TraderAdvancedConfig = {
  strategyMode: string
  cryptoAssetsCsv: string
  cryptoExcludedAssetsCsv: string
  cryptoIncludedTimeframesCsv: string
  cryptoExcludedTimeframesCsv: string
}

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
}


function toNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function isFixedTimeframeCryptoStrategyKey(value: string): boolean {
  const key = String(value || '').trim().toLowerCase()
  return CRYPTO_STRATEGY_OPTIONS.some((option) => option.key === key && Boolean(option.timeframe))
}

function normalizeCryptoStrategyMode(
  value: unknown,
  allowedModes: string[] = [...CRYPTO_STRATEGY_MODES],
  fallback = 'auto'
): string {
  const normalizedAllowed: string[] = []
  const seen = new Set<string>()
  for (const raw of allowedModes) {
    const candidate = String(raw || '').trim().toLowerCase()
    if (!candidate || seen.has(candidate)) continue
    seen.add(candidate)
    normalizedAllowed.push(candidate)
  }
  const mode = String(value || '').trim().toLowerCase()
  if (mode && normalizedAllowed.includes(mode)) {
    return mode
  }
  const normalizedFallback = String(fallback || '').trim().toLowerCase()
  if (normalizedFallback && normalizedAllowed.includes(normalizedFallback)) {
    return normalizedFallback
  }
  if (normalizedAllowed.length > 0) {
    return normalizedAllowed[0]
  }
  return mode || normalizedFallback || 'auto'
}

function normalizeCryptoAsset(value: unknown): string | null {
  const asset = String(value || '').trim().toUpperCase()
  if (!asset) return null
  if (asset === 'XBT') return 'BTC'
  return asset
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

function toStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean)
  }
  if (typeof value === 'string') {
    return csvToList(value)
  }
  return []
}

function normalizeCryptoAssetList(value: unknown, allowedAssets?: string[]): string[] {
  const selected = new Set(
    toStringList(value)
      .map((item) => normalizeCryptoAsset(item))
      .filter((item): item is string => Boolean(item))
  )
  const allowed = new Set(
    (allowedAssets || [])
      .map((item) => normalizeCryptoAsset(item))
      .filter((item): item is string => Boolean(item))
  )
  if (allowed.size > 0) {
    return Array.from(allowed).filter((item) => selected.has(item))
  }
  return Array.from(selected)
}

function normalizeCryptoTimeframeList(value: unknown): string[] {
  const selected = new Set(
    toStringList(value)
      .map((item) => normalizeCryptoTimeframe(item))
      .filter((item): item is string => Boolean(item))
  )
  return CRYPTO_TIMEFRAME_OPTIONS.filter((item) => selected.has(item))
}

function normalizeStatus(value: string | null | undefined): string {
  return String(value || 'unknown').trim().toLowerCase()
}

function toTs(value: string | null | undefined): number {
  if (!value) return 0
  const ts = new Date(value).getTime()
  return Number.isFinite(ts) ? ts : 0
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

function resolveOrderDirectionPresentation(
  order: TraderOrder,
): { side: DirectionSide | null; label: string } {
  const side = normalizeDirectionSide(order.direction_side ?? order.direction)
  const label = cleanText(order.direction_label) || side || String(order.direction || '').trim().toUpperCase() || 'N/A'
  return {
    side,
    label,
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
  if (normalizedReason.includes('allowance') || normalizedReason.includes('not enough balance')) {
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

function isTraderSourceKey(key: string): boolean {
  const k = normalizeSourceKey(key)
  return k === 'traders'
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
      return {
        key,
        label: String(item.label || strategyLabelForKey(key)),
        defaultParams: isRecord(item.default_params) ? { ...item.default_params } : {},
        paramFields: Array.isArray(item.param_fields)
          ? item.param_fields.filter((field): field is Record<string, unknown> => isRecord(field))
          : [],
      }
    })
    .filter((item) => item.key)
  if (options.length > 0) return options
  const fallback =
    source.default_strategy_key ||
    DEFAULT_STRATEGY_BY_SOURCE[normalizeSourceKey(source.key)]
  if (fallback) {
    const normalizedFallback = normalizeStrategyKeyForSource(source.key, fallback)
    return [{ key: normalizedFallback, label: strategyLabelForKey(normalizedFallback, [source]), defaultParams: {}, paramFields: [] }]
  }
  return []
}

function sourceStrategyOptions(source: TraderSource): StrategyOption[] {
  return sourceStrategyDetails(source).map((item) => ({ key: item.key, label: item.label }))
}

function strategyParamKey(
  strategy: StrategyOptionDetail | null | undefined,
  candidateKeys: readonly string[]
): string | null {
  if (!strategy) return null
  for (const candidate of candidateKeys) {
    if (strategy.paramFields.some((field) => String(field.key || '').trim().toLowerCase() === candidate)) {
      return candidate
    }
  }
  for (const candidate of candidateKeys) {
    if (Object.prototype.hasOwnProperty.call(strategy.defaultParams, candidate)) {
      return candidate
    }
  }
  return null
}

function strategyParamOptions(
  strategy: StrategyOptionDetail | null | undefined,
  paramKey: string | null
): string[] {
  if (!strategy || !paramKey) return []
  const field = strategy.paramFields.find(
    (item) => String(item.key || '').trim().toLowerCase() === paramKey
  )
  const rawOptions = Array.isArray(field?.options) ? field.options : []
  const out: string[] = []
  const seen = new Set<string>()
  for (const rawOption of rawOptions) {
    let normalized = ''
    if (typeof rawOption === 'string') {
      normalized = rawOption.trim()
    } else if (isRecord(rawOption)) {
      if (typeof rawOption.value === 'string') normalized = rawOption.value.trim()
      else if (typeof rawOption.label === 'string') normalized = rawOption.label.trim()
    }
    if (!normalized) continue
    const dedupeKey = normalized.toLowerCase()
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)
    out.push(normalized)
  }
  return out
}

function strategyDefaultValuesForKeys(
  strategy: StrategyOptionDetail | null | undefined,
  candidateKeys: readonly string[]
): string[] {
  if (!strategy) return []
  for (const candidate of candidateKeys) {
    const values = toStringList(strategy.defaultParams[candidate])
    if (values.length > 0) return values
  }
  return []
}

function cryptoModeOptionsForStrategy(strategy: StrategyOptionDetail | null | undefined): string[] {
  const paramKey = strategyParamKey(strategy, CRYPTO_MODE_PARAM_KEYS)
  const options = strategyParamOptions(strategy, paramKey)
    .map((item) => normalizeCryptoStrategyMode(item))
    .filter(Boolean)
  const out: string[] = []
  const seen = new Set<string>()
  for (const option of options) {
    const key = option.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(option)
  }
  return out
}

function cryptoAssetOptionsForStrategy(strategy: StrategyOptionDetail | null | undefined): string[] {
  const paramKey = strategyParamKey(strategy, CRYPTO_INCLUDE_ASSET_PARAM_KEYS)
  const fieldOptions = normalizeCryptoAssetList(strategyParamOptions(strategy, paramKey))
  if (fieldOptions.length > 0) return fieldOptions
  return normalizeCryptoAssetList(strategyDefaultValuesForKeys(strategy, CRYPTO_INCLUDE_ASSET_PARAM_KEYS))
}

function cryptoTimeframeOptionsForStrategy(strategy: StrategyOptionDetail | null | undefined): string[] {
  const paramKey = strategyParamKey(strategy, CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS)
  const fieldOptions = normalizeCryptoTimeframeList(strategyParamOptions(strategy, paramKey))
  if (fieldOptions.length > 0) return fieldOptions
  return normalizeCryptoTimeframeList(strategyDefaultValuesForKeys(strategy, CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS))
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

function cryptoTimeframeForStrategyKey(strategyKey: string): string {
  const normalized = String(strategyKey || '').trim().toLowerCase()
  const timeframe = CRYPTO_STRATEGY_OPTIONS.find((item) => item.key === normalized)?.timeframe
  return typeof timeframe === 'string' ? timeframe : ''
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

function defaultAdvancedConfig(): TraderAdvancedConfig {
  return {
    strategyMode: 'auto',
    cryptoAssetsCsv: CRYPTO_ASSET_OPTIONS.join(', '),
    cryptoExcludedAssetsCsv: '',
    cryptoIncludedTimeframesCsv: CRYPTO_TIMEFRAME_OPTIONS.join(', '),
    cryptoExcludedTimeframesCsv: '',
  }
}

function computeAdvancedConfig(
  params: Record<string, unknown>
): TraderAdvancedConfig {
  const defaults = defaultAdvancedConfig()
  const configuredCryptoAssets = normalizeCryptoAssetList(params.include_assets)
  const excludedCryptoAssets = normalizeCryptoAssetList(params.exclude_assets)
  const configuredCryptoTimeframes = normalizeCryptoTimeframeList(params.include_timeframes)
  const excludedCryptoTimeframes = normalizeCryptoTimeframeList(params.exclude_timeframes)

  return {
    strategyMode: normalizeCryptoStrategyMode(params.strategy_mode ?? params.mode ?? defaults.strategyMode),
    cryptoAssetsCsv: (configuredCryptoAssets.length > 0 ? configuredCryptoAssets : [...CRYPTO_ASSET_OPTIONS]).join(', '),
    cryptoExcludedAssetsCsv: excludedCryptoAssets.join(', '),
    cryptoIncludedTimeframesCsv: (configuredCryptoTimeframes.length > 0 ? configuredCryptoTimeframes : [...CRYPTO_TIMEFRAME_OPTIONS]).join(', '),
    cryptoExcludedTimeframesCsv: excludedCryptoTimeframes.join(', '),
  }
}

function withConfiguredParams(
  raw: Record<string, unknown>,
  config: TraderAdvancedConfig,
  sourceKey: string,
  strategyKey: string,
  strategyDetail: StrategyOptionDetail | null
): Record<string, unknown> {
  const next: Record<string, unknown> = { ...raw }

  if (normalizeSourceKey(sourceKey) !== 'crypto') {
    for (const key of CRYPTO_MODE_PARAM_KEYS) delete next[key]
    for (const key of CRYPTO_INCLUDE_ASSET_PARAM_KEYS) delete next[key]
    for (const key of CRYPTO_EXCLUDE_ASSET_PARAM_KEYS) delete next[key]
    for (const key of CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS) delete next[key]
    for (const key of CRYPTO_EXCLUDE_TIMEFRAME_PARAM_KEYS) delete next[key]
    return next
  }

  const supportsFixedTimeframe = isFixedTimeframeCryptoStrategyKey(strategyKey)
  const modeParamKey = strategyParamKey(strategyDetail, CRYPTO_MODE_PARAM_KEYS)
  const modeOptions = cryptoModeOptionsForStrategy(strategyDetail)
  if (modeParamKey || supportsFixedTimeframe) {
    const targetKey = modeParamKey || 'strategy_mode'
    const allowedModes = modeOptions.length > 0 ? modeOptions : [...CRYPTO_STRATEGY_MODES]
    next[targetKey] = normalizeCryptoStrategyMode(config.strategyMode, allowedModes, 'auto')
  } else {
    for (const key of CRYPTO_MODE_PARAM_KEYS) delete next[key]
  }

  const includeAssetParamKey = strategyParamKey(strategyDetail, CRYPTO_INCLUDE_ASSET_PARAM_KEYS)
  const excludeAssetParamKey = strategyParamKey(strategyDetail, CRYPTO_EXCLUDE_ASSET_PARAM_KEYS)
  const schemaAssets = cryptoAssetOptionsForStrategy(strategyDetail)
  const allowedAssets = schemaAssets.length > 0 ? schemaAssets : [...CRYPTO_ASSET_OPTIONS]
  if (includeAssetParamKey) {
    const targetKey = includeAssetParamKey
    const targetAssets = normalizeCryptoAssetList(config.cryptoAssetsCsv, allowedAssets)
    next[targetKey] = targetAssets.length > 0 ? targetAssets : allowedAssets
  } else {
    for (const key of CRYPTO_INCLUDE_ASSET_PARAM_KEYS) delete next[key]
  }
  if (excludeAssetParamKey) {
    const targetKey = excludeAssetParamKey
    const excludedAssets = normalizeCryptoAssetList(config.cryptoExcludedAssetsCsv, allowedAssets)
    next[targetKey] = excludedAssets
  } else {
    for (const key of CRYPTO_EXCLUDE_ASSET_PARAM_KEYS) delete next[key]
  }

  const includeTimeframeParamKey = strategyParamKey(strategyDetail, CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS)
  const excludeTimeframeParamKey = strategyParamKey(strategyDetail, CRYPTO_EXCLUDE_TIMEFRAME_PARAM_KEYS)
  const schemaTimeframes = cryptoTimeframeOptionsForStrategy(strategyDetail)
  const allowedTimeframes = schemaTimeframes.length > 0 ? schemaTimeframes : [...CRYPTO_TIMEFRAME_OPTIONS]
  const fixedTimeframe = cryptoTimeframeForStrategyKey(strategyKey)
  if (includeTimeframeParamKey || fixedTimeframe) {
    const targetKey = includeTimeframeParamKey || 'include_timeframes'
    if (fixedTimeframe) {
      next[targetKey] = [fixedTimeframe]
    } else {
      const targetTimeframes = normalizeCryptoTimeframeList(config.cryptoIncludedTimeframesCsv)
      next[targetKey] = targetTimeframes.length > 0 ? targetTimeframes : allowedTimeframes
    }
  } else {
    for (const key of CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS) delete next[key]
  }

  if (excludeTimeframeParamKey) {
    const targetKey = excludeTimeframeParamKey
    if (fixedTimeframe) {
      next[targetKey] = []
    } else {
      next[targetKey] = normalizeCryptoTimeframeList(config.cryptoExcludedTimeframesCsv)
    }
  } else {
    for (const key of CRYPTO_EXCLUDE_TIMEFRAME_PARAM_KEYS) delete next[key]
  }
  return next
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
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const [traderFeedFilter, setTraderFeedFilter] = useState<FeedFilter>('all')
  const [tradeStatusFilter, setTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [tradeSearch, setTradeSearch] = useState('')
  const [decisionSearch, setDecisionSearch] = useState('')
  const [botRosterSearch, setBotRosterSearch] = useState('')
  const [botRosterHideInactive, setBotRosterHideInactive] = useState(false)
  const [botRosterSort, setBotRosterSort] = useState<BotRosterSort>('name_asc')
  const [botRosterGroupBy, setBotRosterGroupBy] = useState<BotRosterGroupBy>('status')
  const [confirmLiveStartOpen, setConfirmLiveStartOpen] = useState(false)
  const [workTab, setWorkTab] = useState<'terminal' | 'positions' | 'trades' | 'monitor' | 'decisions' | 'risk'>('terminal')
  const [allBotsTab, setAllBotsTab] = useState<AllBotsTab>('overview')
  const [allBotsTradeStatusFilter, setAllBotsTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [allBotsTradeSearch, setAllBotsTradeSearch] = useState('')
  const [positionSearch, setPositionSearch] = useState('')
  const [positionDirectionFilter, setPositionDirectionFilter] = useState<PositionDirectionFilter>('all')
  const [positionSortField, setPositionSortField] = useState<PositionSortField>('exposure')
  const [positionSortDirection, setPositionSortDirection] = useState<PositionSortDirection>('desc')
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
  const [draftInterval, setDraftInterval] = useState('60')
  const [draftSources, setDraftSources] = useState('')
  const [draftTradersScopeModes, setDraftTradersScopeModes] = useState<TradersScopeMode[]>(['tracked', 'pool'])
  const [draftTradersIndividualWallets, setDraftTradersIndividualWallets] = useState<string[]>([])
  const [draftTradersGroupIds, setDraftTradersGroupIds] = useState<string[]>([])
  const [draftParams, setDraftParams] = useState('{}')
  const [draftRisk, setDraftRisk] = useState('{}')
  const [draftMetadata, setDraftMetadata] = useState('{}')
  const [draftCopyFromTraderId, setDraftCopyFromTraderId] = useState('')
  const [advancedConfig, setAdvancedConfig] = useState<TraderAdvancedConfig>(defaultAdvancedConfig())
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

  const overviewQuery = useQuery({
    queryKey: ['trader-orchestrator-overview'],
    queryFn: getTraderOrchestratorOverview,
    refetchInterval: 4000,
  })

  const tradersQuery = useQuery({
    queryKey: ['traders-list'],
    queryFn: getTraders,
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

  const walletsQuery = useQuery({
    queryKey: ['tracked-wallets'],
    queryFn: getWallets,
    staleTime: 30000,
  })

  const traderGroupsQuery = useQuery({
    queryKey: ['discovery-trader-groups'],
    queryFn: () => discoveryApi.getTraderGroups(false, 200),
    staleTime: 30000,
  })

  const traderConfigSchema: TraderConfigSchema | null = traderConfigSchemaQuery.data ?? null
  const activeCopyMode = activeCopyModeQuery.data ?? null
  const traders = tradersQuery.data || []
  const simulationAccounts = simulationAccountsQuery.data || []
  const selectedSandboxAccount = simulationAccounts.find((account) => account.id === selectedAccountId)
  const selectedAccountIsLive = Boolean(selectedAccountId?.startsWith('live:'))
  const selectedAccountValid = selectedAccountIsLive || Boolean(selectedSandboxAccount)
  const selectedAccountMode = selectedAccountIsLive ? 'live' : 'paper'
  const trackedWallets = walletsQuery.data || []
  const traderGroups = traderGroupsQuery.data || []
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
  const traderIdsKey = useMemo(() => traderIds.join('|'), [traderIds])

  const allOrdersQuery = useQuery({
    queryKey: ['trader-orders-all'],
    queryFn: () => getAllTraderOrders(5000),
    enabled: traderIds.length > 0,
    refetchInterval: isConnected ? 15000 : 1000,
    staleTime: 0,
    refetchOnMount: 'always',
  })

  const allDecisionsQuery = useQuery({
    queryKey: ['trader-decisions-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: false,
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

  const allOrders = allOrdersQuery.data || []
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

  const selectedOrdersQuery = useQuery({
    queryKey: ['trader-orders', selectedTraderId],
    enabled: Boolean(selectedTraderId),
    refetchInterval: selectedTraderId ? (isConnected ? 15000 : 1000) : false,
    queryFn: () => getTraderOrders(String(selectedTraderId), { limit: 5000 }),
    staleTime: 0,
    refetchOnMount: 'always',
  })

  const selectedOrders = useMemo(
    () => {
      if (!selectedTraderId) return []
      const queryRows = selectedOrdersQuery.data
      if (Array.isArray(queryRows)) {
        return queryRows
      }
      return allOrders.filter((order) => order.trader_id === selectedTraderId)
    },
    [allOrders, selectedOrdersQuery.data, selectedTraderId]
  )

  const selectedDecisions = useMemo(
    () => allDecisions.filter((decision) => decision.trader_id === selectedTraderId),
    [allDecisions, selectedTraderId]
  )

  const selectedEvents = useMemo(
    () => allEvents.filter((event) => event.trader_id === selectedTraderId),
    [allEvents, selectedTraderId]
  )

  const sourceCards = useMemo(() => {
    return uniqueSourceList(sourceCatalog.map((source) => source.key))
      .map((key) => sourceCatalog.find((source) => normalizeSourceKey(source.key) === normalizeSourceKey(key)))
      .filter((source): source is TraderSource => Boolean(source))
      .map((source) => ({
        ...source,
        isLegacy: false,
      }))
  }, [sourceCatalog])

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

  const cryptoStrategyKeyDraft = useMemo(
    () => effectiveSourceStrategies.crypto || DEFAULT_STRATEGY_KEY,
    [effectiveSourceStrategies]
  )
  const isCryptoStrategyDraft = useMemo(
    () => selectedSourceKeySet.has('crypto'),
    [selectedSourceKeySet]
  )
  const selectedCryptoStrategyDetail = useMemo(
    () => sourceStrategyDetailsLookup.crypto?.[cryptoStrategyKeyDraft] || null,
    [cryptoStrategyKeyDraft, sourceStrategyDetailsLookup]
  )
  const tradersStrategyKeyDraft = useMemo(
    () => effectiveSourceStrategies.traders || defaultStrategyForSource('traders', sourceCards),
    [effectiveSourceStrategies, sourceCards]
  )
  const selectedTradersStrategyDetail = useMemo(
    () => sourceStrategyDetailsLookup.traders?.[tradersStrategyKeyDraft] || null,
    [sourceStrategyDetailsLookup, tradersStrategyKeyDraft]
  )
  const cryptoModeParamKey = useMemo(
    () => strategyParamKey(selectedCryptoStrategyDetail, CRYPTO_MODE_PARAM_KEYS),
    [selectedCryptoStrategyDetail]
  )
  const cryptoStrategyModeOptions = useMemo(() => {
    const options = cryptoModeOptionsForStrategy(selectedCryptoStrategyDetail)
    if (options.length > 0) return options
    if (isFixedTimeframeCryptoStrategyKey(cryptoStrategyKeyDraft)) return [...CRYPTO_STRATEGY_MODES]
    return []
  }, [cryptoStrategyKeyDraft, selectedCryptoStrategyDetail])
  const cryptoIncludeAssetParamKey = useMemo(
    () => strategyParamKey(selectedCryptoStrategyDetail, CRYPTO_INCLUDE_ASSET_PARAM_KEYS),
    [selectedCryptoStrategyDetail]
  )
  const cryptoExcludeAssetParamKey = useMemo(
    () => strategyParamKey(selectedCryptoStrategyDetail, CRYPTO_EXCLUDE_ASSET_PARAM_KEYS),
    [selectedCryptoStrategyDetail]
  )
  const cryptoIncludeTimeframeParamKey = useMemo(
    () => strategyParamKey(selectedCryptoStrategyDetail, CRYPTO_INCLUDE_TIMEFRAME_PARAM_KEYS),
    [selectedCryptoStrategyDetail]
  )
  const cryptoExcludeTimeframeParamKey = useMemo(
    () => strategyParamKey(selectedCryptoStrategyDetail, CRYPTO_EXCLUDE_TIMEFRAME_PARAM_KEYS),
    [selectedCryptoStrategyDetail]
  )
  const cryptoAssetOptions = useMemo(() => {
    const options = cryptoAssetOptionsForStrategy(selectedCryptoStrategyDetail)
    if (options.length > 0) return options
    if (isFixedTimeframeCryptoStrategyKey(cryptoStrategyKeyDraft)) return [...CRYPTO_ASSET_OPTIONS]
    return []
  }, [cryptoStrategyKeyDraft, selectedCryptoStrategyDetail])
  const cryptoTimeframeOptions = useMemo(() => {
    const options = cryptoTimeframeOptionsForStrategy(selectedCryptoStrategyDetail)
    if (options.length > 0) return options
    if (isFixedTimeframeCryptoStrategyKey(cryptoStrategyKeyDraft)) return [...CRYPTO_TIMEFRAME_OPTIONS]
    return []
  }, [cryptoStrategyKeyDraft, selectedCryptoStrategyDetail])
  const cryptoTimeframeDraft = useMemo(
    () => cryptoTimeframeForStrategyKey(cryptoStrategyKeyDraft),
    [cryptoStrategyKeyDraft]
  )
  const selectedCryptoAssets = useMemo(
    () =>
      new Set(
        normalizeCryptoAssetList(
          advancedConfig.cryptoAssetsCsv,
          cryptoAssetOptions.length > 0 ? cryptoAssetOptions : [...CRYPTO_ASSET_OPTIONS]
        )
      ),
    [advancedConfig.cryptoAssetsCsv, cryptoAssetOptions]
  )
  const selectedExcludedCryptoAssets = useMemo(
    () =>
      new Set(
        normalizeCryptoAssetList(
          advancedConfig.cryptoExcludedAssetsCsv,
          cryptoAssetOptions.length > 0 ? cryptoAssetOptions : [...CRYPTO_ASSET_OPTIONS]
        )
      ),
    [advancedConfig.cryptoExcludedAssetsCsv, cryptoAssetOptions]
  )
  const selectedCryptoIncludedTimeframes = useMemo(
    () =>
      new Set(
        normalizeCryptoTimeframeList(advancedConfig.cryptoIncludedTimeframesCsv)
      ),
    [advancedConfig.cryptoIncludedTimeframesCsv]
  )
  const selectedCryptoExcludedTimeframes = useMemo(
    () =>
      new Set(
        normalizeCryptoTimeframeList(advancedConfig.cryptoExcludedTimeframesCsv)
      ),
    [advancedConfig.cryptoExcludedTimeframesCsv]
  )
  const tradersStrategyFormSchema = useMemo(
    () => ({
      param_fields: Array.isArray(selectedTradersStrategyDetail?.paramFields) ? selectedTradersStrategyDetail.paramFields : [],
    }),
    [selectedTradersStrategyDetail]
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
    const mode = draftCopyTrading.copy_mode_type
    const tradeMode = draftCopyTrading.copy_trade_mode
    const proportionalSizing = draftCopyTrading.proportional_sizing
    return {
      param_fields: baseFields.filter((field: Record<string, unknown>) => {
        const key = String(field.key || '').trim()
        if (!key) return false
        if (mode === 'disabled') {
          return key === 'copy_mode_type'
        }
        if (mode !== 'individual' && key === 'individual_wallet') return false
        if (tradeMode !== 'arb_only' && key === 'min_roi_threshold') return false
        if (!proportionalSizing && key === 'proportional_multiplier') return false
        return true
      }),
    }
  }, [draftCopyTrading.copy_mode_type, draftCopyTrading.copy_trade_mode, draftCopyTrading.proportional_sizing, traderConfigSchema])
  const parsedDraftParams = useMemo(() => parseJsonObject(draftParams || '{}'), [draftParams])
  const parsedDraftRisk = useMemo(() => parseJsonObject(draftRisk || '{}'), [draftRisk])
  const parsedDraftMetadata = useMemo(() => parseJsonObject(draftMetadata || '{}'), [draftMetadata])
  const tradingScheduleDraft = useMemo(
    () => normalizeTradingScheduleDraft(parsedDraftMetadata.value?.trading_schedule_utc),
    [parsedDraftMetadata.value]
  )
  const riskFormValues = useMemo(
    () => ({
      ...(isRecord(traderConfigSchema?.shared_risk_defaults) ? traderConfigSchema.shared_risk_defaults : {}),
      ...(parsedDraftRisk.value || {}),
    }),
    [parsedDraftRisk.value, traderConfigSchema]
  )
  const tradersStrategyFormValues = useMemo(
    () => ({
      ...(isRecord(selectedTradersStrategyDetail?.defaultParams) ? selectedTradersStrategyDetail?.defaultParams : {}),
      ...(parsedDraftParams.value || {}),
      traders_scope: {
        modes: draftTradersScopeModes,
        individual_wallets: draftTradersIndividualWallets,
        group_ids: draftTradersGroupIds,
      },
    }),
    [
      draftTradersGroupIds,
      draftTradersIndividualWallets,
      draftTradersScopeModes,
      parsedDraftParams.value,
      selectedTradersStrategyDetail?.defaultParams,
    ]
  )
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

  const setSourceStrategy = (sourceKey: string, strategyKey: string) => {
    const normalizedSource = normalizeSourceKey(sourceKey)
    const normalizedStrategy = normalizeStrategyKeyForSource(normalizedSource, strategyKey)
    setDraftSourceStrategies((current) => ({
      ...current,
      [normalizedSource]: normalizedStrategy,
    }))
    if (normalizedSource === 'crypto') {
      setDraftStrategyKey(normalizedStrategy)
    }
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
    } else {
      const defaultStrategy = defaultStrategyForSource(normalizedTarget, sourceCards)
      setDraftSourceStrategies((current) => ({ ...current, [normalizedTarget]: defaultStrategy }))
    }
  }

  const enableAllSourceCards = () => {
    setDraftSources(uniqueSourceList(sourceCards.map((source) => source.key)).join(', '))
    const next: Record<string, string> = {}
    for (const source of sourceCards) {
      const sourceKey = normalizeSourceKey(source.key)
      next[sourceKey] = defaultStrategyForSource(sourceKey, sourceCards)
    }
    setDraftSourceStrategies(next)
  }

  const disableAllSourceCards = () => {
    setDraftSources('')
    setDraftSourceStrategies({})
  }

  const toggleCryptoAssetTarget = (asset: string) => {
    const availableAssets = cryptoAssetOptions.length > 0 ? cryptoAssetOptions : [...CRYPTO_ASSET_OPTIONS]
    const next = new Set(normalizeCryptoAssetList(advancedConfig.cryptoAssetsCsv, availableAssets))
    if (next.has(asset)) {
      next.delete(asset)
    } else {
      next.add(asset)
    }
    setAdvancedValue('cryptoAssetsCsv', availableAssets.filter((item) => next.has(item)).join(', '))
  }

  const toggleExcludedCryptoAssetTarget = (asset: string) => {
    const availableAssets = cryptoAssetOptions.length > 0 ? cryptoAssetOptions : [...CRYPTO_ASSET_OPTIONS]
    const next = new Set(normalizeCryptoAssetList(advancedConfig.cryptoExcludedAssetsCsv, availableAssets))
    if (next.has(asset)) {
      next.delete(asset)
    } else {
      next.add(asset)
    }
    setAdvancedValue('cryptoExcludedAssetsCsv', availableAssets.filter((item) => next.has(item)).join(', '))
  }

  const toggleCryptoTimeframeTarget = (timeframe: string) => {
    const availableTimeframes = cryptoTimeframeOptions.length > 0 ? cryptoTimeframeOptions : [...CRYPTO_TIMEFRAME_OPTIONS]
    const next = new Set(normalizeCryptoTimeframeList(advancedConfig.cryptoIncludedTimeframesCsv))
    if (next.has(timeframe)) {
      next.delete(timeframe)
    } else {
      next.add(timeframe)
    }
    setAdvancedValue('cryptoIncludedTimeframesCsv', availableTimeframes.filter((item) => next.has(item)).join(', '))
  }

  const toggleExcludedCryptoTimeframeTarget = (timeframe: string) => {
    const availableTimeframes = cryptoTimeframeOptions.length > 0 ? cryptoTimeframeOptions : [...CRYPTO_TIMEFRAME_OPTIONS]
    const next = new Set(normalizeCryptoTimeframeList(advancedConfig.cryptoExcludedTimeframesCsv))
    if (next.has(timeframe)) {
      next.delete(timeframe)
    } else {
      next.add(timeframe)
    }
    setAdvancedValue('cryptoExcludedTimeframesCsv', availableTimeframes.filter((item) => next.has(item)).join(', '))
  }

  const enableAllCryptoTargets = () => {
    const availableAssets = cryptoAssetOptions.length > 0 ? cryptoAssetOptions : [...CRYPTO_ASSET_OPTIONS]
    const availableTimeframes = cryptoTimeframeOptions.length > 0 ? cryptoTimeframeOptions : [...CRYPTO_TIMEFRAME_OPTIONS]
    if (availableAssets.length > 0) {
      setAdvancedValue('cryptoAssetsCsv', availableAssets.join(', '))
    }
    setAdvancedValue('cryptoExcludedAssetsCsv', '')
    if (cryptoTimeframeDraft) {
      setAdvancedValue('cryptoIncludedTimeframesCsv', cryptoTimeframeDraft)
      setAdvancedValue('cryptoExcludedTimeframesCsv', '')
      return
    }
    if (availableTimeframes.length > 0) {
      setAdvancedValue('cryptoIncludedTimeframesCsv', availableTimeframes.join(', '))
    }
    setAdvancedValue('cryptoExcludedTimeframesCsv', '')
  }

  useEffect(() => {
    if (!isCryptoStrategyDraft || cryptoStrategyModeOptions.length === 0) return
    const normalizedMode = normalizeCryptoStrategyMode(
      advancedConfig.strategyMode,
      cryptoStrategyModeOptions,
      cryptoStrategyModeOptions[0]
    )
    if (normalizedMode === advancedConfig.strategyMode) return
    setAdvancedConfig((current) => ({ ...current, strategyMode: normalizedMode }))
  }, [advancedConfig.strategyMode, cryptoStrategyModeOptions, isCryptoStrategyDraft])

  useEffect(() => {
    if (!isCryptoStrategyDraft || cryptoAssetOptions.length === 0) return
    const normalizedAssets = normalizeCryptoAssetList(
      advancedConfig.cryptoAssetsCsv,
      cryptoAssetOptions
    ).join(', ')
    if (normalizedAssets === advancedConfig.cryptoAssetsCsv) return
    setAdvancedConfig((current) => ({ ...current, cryptoAssetsCsv: normalizedAssets }))
  }, [advancedConfig.cryptoAssetsCsv, cryptoAssetOptions, isCryptoStrategyDraft])

  useEffect(() => {
    if (!isCryptoStrategyDraft || cryptoAssetOptions.length === 0) return
    const normalizedAssets = normalizeCryptoAssetList(
      advancedConfig.cryptoExcludedAssetsCsv,
      cryptoAssetOptions
    ).join(', ')
    if (normalizedAssets === advancedConfig.cryptoExcludedAssetsCsv) return
    setAdvancedConfig((current) => ({ ...current, cryptoExcludedAssetsCsv: normalizedAssets }))
  }, [advancedConfig.cryptoExcludedAssetsCsv, cryptoAssetOptions, isCryptoStrategyDraft])

  useEffect(() => {
    if (!isCryptoStrategyDraft || cryptoTimeframeOptions.length === 0) return
    const allowed = new Set(cryptoTimeframeOptions)
    const normalizedTimeframes = normalizeCryptoTimeframeList(advancedConfig.cryptoIncludedTimeframesCsv)
      .filter((item) => allowed.has(item))
      .join(', ')
    if (normalizedTimeframes === advancedConfig.cryptoIncludedTimeframesCsv) return
    setAdvancedConfig((current) => ({ ...current, cryptoIncludedTimeframesCsv: normalizedTimeframes }))
  }, [advancedConfig.cryptoIncludedTimeframesCsv, cryptoTimeframeOptions, isCryptoStrategyDraft])

  useEffect(() => {
    if (!isCryptoStrategyDraft || cryptoTimeframeOptions.length === 0) return
    const allowed = new Set(cryptoTimeframeOptions)
    const normalizedTimeframes = normalizeCryptoTimeframeList(advancedConfig.cryptoExcludedTimeframesCsv)
      .filter((item) => allowed.has(item))
      .join(', ')
    if (normalizedTimeframes === advancedConfig.cryptoExcludedTimeframesCsv) return
    setAdvancedConfig((current) => ({ ...current, cryptoExcludedTimeframesCsv: normalizedTimeframes }))
  }, [advancedConfig.cryptoExcludedTimeframesCsv, cryptoTimeframeOptions, isCryptoStrategyDraft])

  useEffect(() => {
    if (selectedDecisions.length === 0) {
      setSelectedDecisionId(null)
      return
    }

    setSelectedDecisionId((current) => {
      if (current && selectedDecisions.some((decision) => decision.id === current)) {
        return current
      }
      return selectedDecisions[0].id
    })
  }, [selectedDecisions])

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
  }

  const applyTraderDraftSettings = (
    trader: Trader,
    options: { preserveName?: boolean; preserveCopyFrom?: boolean } = {}
  ) => {
    const traderSourceConfigs = Array.isArray(trader.source_configs) ? trader.source_configs : []
    const normalizedSourceKeys = uniqueSourceList(
      traderSourceConfigs.map((config) => normalizeSourceKey(String(config.source_key || '')))
    )
    const sourceStrategyMap: Record<string, string> = {}
    for (const config of traderSourceConfigs) {
      const sourceKey = normalizeSourceKey(String(config.source_key || ''))
      if (!sourceKey) continue
      sourceStrategyMap[sourceKey] = normalizeStrategyKeyForSource(
        sourceKey,
        config.strategy_key || defaultStrategyForSource(sourceKey, sourceCards)
      )
    }
    const primaryParams = (traderSourceConfigs[0]?.strategy_params || {}) as Record<string, unknown>
    const cryptoParams =
      (traderSourceConfigs.find((config) => normalizeSourceKey(String(config.source_key || '')) === 'crypto')?.strategy_params ||
        primaryParams) as Record<string, unknown>
    const tradersScope = traderSourceConfigs.find((config) => normalizeSourceKey(String(config.source_key || '')) === 'traders')
      ?.strategy_params?.traders_scope

    if (!options.preserveName) {
      setDraftName(trader.name)
    }
    setDraftDescription(trader.description || '')
    setDraftStrategyKey(normalizeStrategyKey(sourceStrategyMap.crypto || DEFAULT_STRATEGY_KEY))
    setDraftSourceStrategies(sourceStrategyMap)
    setDraftInterval(String(trader.interval_seconds || 60))
    setDraftSources(normalizedSourceKeys.join(', ') || defaultSourceCsv)
    const risk = trader.risk_limits || {}
    const metadata = trader.metadata || {}
    setDraftParams(JSON.stringify(primaryParams, null, 2))
    setDraftRisk(JSON.stringify(risk, null, 2))
    setDraftMetadata(JSON.stringify(metadata, null, 2))
    if (!options.preserveCopyFrom) {
      setDraftCopyFromTraderId('')
    }
    setAdvancedConfig(computeAdvancedConfig(cryptoParams))
    const tradersScopeRecord = isRecord(tradersScope) ? tradersScope : {}
    setDraftTradersScopeModes(
      (toStringList(tradersScopeRecord.modes).length > 0 ? toStringList(tradersScopeRecord.modes) : ['tracked', 'pool'])
        .map((mode) => String(mode || '').toLowerCase())
        .filter((mode): mode is TradersScopeMode =>
          mode === 'tracked' || mode === 'pool' || mode === 'individual' || mode === 'group'
        )
    )
    setDraftTradersIndividualWallets(
      toStringList(tradersScopeRecord.individual_wallets)
        .map((wallet) => String(wallet || '').trim().toLowerCase())
        .filter(Boolean)
    )
    setDraftTradersGroupIds(
      toStringList(tradersScopeRecord.group_ids)
        .map((groupId) => String(groupId || '').trim())
        .filter(Boolean)
    )
  }

  const applyCreateCopyFromSelection = (value: string) => {
    const sourceTraderId = value === '__none__' ? '' : String(value || '').trim()
    setDraftCopyFromTraderId(sourceTraderId)
    if (!sourceTraderId) {
      setSaveError(null)
      return
    }

    const sourceTrader = traders.find((trader) => trader.id === sourceTraderId)
    if (!sourceTrader) {
      setSaveError('Selected copy source bot was not found. Refresh and try again.')
      return
    }

    applyTraderDraftSettings(sourceTrader, { preserveName: true, preserveCopyFrom: true })
    setSaveError(null)
  }

  const openCreateTraderFlyout = () => {
    const defaultSources = defaultSourceKeys.length > 0 ? defaultSourceKeys.map((key) => normalizeSourceKey(key)) : ['crypto']
    const defaultStrategies = Object.fromEntries(
      defaultSources.map((sourceKey) => [sourceKey, defaultStrategyForSource(sourceKey, sourceCards)])
    ) as Record<string, string>
    setTraderFlyoutMode('create')
    setDraftName('')
    setDraftDescription('')
    setDraftStrategyKey(normalizeStrategyKey(defaultStrategies.crypto || DEFAULT_STRATEGY_KEY))
    setDraftSourceStrategies(defaultStrategies)
    setDraftInterval('5')
    setDraftSources(defaultSources.join(', '))
    setDraftTradersScopeModes(['tracked', 'pool'])
    setDraftTradersIndividualWallets([])
    setDraftTradersGroupIds([])
    setDraftParams('{}')
    setDraftRisk(JSON.stringify(isRecord(traderConfigSchema?.shared_risk_defaults) ? traderConfigSchema.shared_risk_defaults : {}, null, 2))
    setDraftMetadata('{}')
    setDraftCopyFromTraderId('')
    setAdvancedConfig(defaultAdvancedConfig())
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

  const setCt = (partial: Partial<CopyTradingFormState>) =>
    setDraftCopyTrading((prev) => normalizeCopyTradingConfig({ ...prev, ...partial }))

  const setAdvancedValue = <K extends keyof TraderAdvancedConfig>(key: K, value: TraderAdvancedConfig[K]) => {
    setAdvancedConfig((current) => ({ ...current, [key]: value }))
  }

  const applyTradersStrategyFormValues = (values: Record<string, unknown>) => {
    const nextValues = { ...values }
    const rawScope = isRecord(nextValues.traders_scope) ? nextValues.traders_scope : {}
    const scopeModes = toStringList(rawScope.modes)
      .map((mode) => String(mode || '').trim().toLowerCase())
      .filter((mode): mode is TradersScopeMode =>
        mode === 'tracked' || mode === 'pool' || mode === 'individual' || mode === 'group'
      )
    const normalizedScopeModes: TradersScopeMode[] = scopeModes.length > 0 ? scopeModes : ['tracked']
    const normalizedScope = {
      modes: normalizedScopeModes,
      individual_wallets: toStringList(rawScope.individual_wallets).map((wallet) => wallet.toLowerCase()),
      group_ids: toStringList(rawScope.group_ids),
    }
    setDraftTradersScopeModes(normalizedScope.modes)
    setDraftTradersIndividualWallets(normalizedScope.individual_wallets)
    setDraftTradersGroupIds(normalizedScope.group_ids)
    nextValues.traders_scope = normalizedScope
    setDraftParams(JSON.stringify(nextValues, null, 2))
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
    setDraftCopyTrading((previous) =>
      normalizeCopyTradingConfig({
        ...previous,
        ...values,
        account_id: previous.account_id || defaultCopyTradingAccountId,
      })
    )
  }

  const toggleIndividualWallet = (wallet: string) => {
    const normalized = String(wallet || '').trim().toLowerCase()
    if (!normalized) return
    setDraftTradersIndividualWallets((current) =>
      current.includes(normalized) ? current.filter((item) => item !== normalized) : [...current, normalized]
    )
  }

  const toggleTradersGroup = (groupId: string) => {
    const normalized = String(groupId || '').trim()
    if (!normalized) return
    setDraftTradersGroupIds((current) =>
      current.includes(normalized) ? current.filter((item) => item !== normalized) : [...current, normalized]
    )
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
      const nextConfig: TraderSourceConfig = {
        source_key: sourceKey,
        strategy_key: strategyKey,
        strategy_params: withConfiguredParams(rawStrategyParams, advancedConfig, sourceKey, strategyKey, strategyDetail),
      }
      if (sourceKey === 'traders') {
        nextConfig.strategy_params = {
          ...nextConfig.strategy_params,
          traders_scope: {
            modes: draftTradersScopeModes,
            individual_wallets: draftTradersIndividualWallets,
            group_ids: draftTradersGroupIds,
          },
        }
      } else if (isRecord(nextConfig.strategy_params)) {
        delete nextConfig.strategy_params.traders_scope
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
      if (tradersEnabled && draftTradersScopeModes.includes('individual') && draftTradersIndividualWallets.length === 0) {
        throw new Error('Select at least one individual wallet for wallet scope.')
      }
      if (tradersEnabled && draftTradersScopeModes.includes('group') && draftTradersGroupIds.length === 0) {
        throw new Error('Select at least one group for wallet scope.')
      }

      await persistGlobalWalletSettings()

      const payload: Record<string, unknown> = {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
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
      if (tradersEnabled && draftTradersScopeModes.includes('individual') && draftTradersIndividualWallets.length === 0) {
        throw new Error('Select at least one individual wallet for wallet scope.')
      }
      if (tradersEnabled && draftTradersScopeModes.includes('group') && draftTradersGroupIds.length === 0) {
        throw new Error('Select at least one group for wallet scope.')
      }

      await persistGlobalWalletSettings()

      return updateTrader(traderId, {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
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
  const metrics = overviewQuery.data?.metrics
  const killSwitchOn = Boolean(overviewQuery.data?.control?.kill_switch)
  const globalMode = String(overviewQuery.data?.control?.mode || 'paper').toLowerCase()
  const orchestratorEnabled = Boolean(overviewQuery.data?.control?.is_enabled) && !Boolean(overviewQuery.data?.control?.is_paused)
  const workerActivity = String(worker?.current_activity || '').trim().toLowerCase()
  const orchestratorWorkerRunning = Boolean(worker?.running)
  const orchestratorRunning = orchestratorEnabled && orchestratorWorkerRunning
  const orchestratorBlocked = orchestratorEnabled && !orchestratorWorkerRunning && workerActivity.startsWith('blocked')
  const orchestratorStatusLabel = orchestratorBlocked ? 'BLOCKED' : orchestratorRunning ? 'RUNNING' : 'STOPPED'

  const modeMismatch = selectedAccountValid && orchestratorEnabled && globalMode !== selectedAccountMode

  const controlBusy =
    startBySelectedAccountMutation.isPending ||
    stopByModeMutation.isPending ||
    killSwitchMutation.isPending
  const traderFlyoutBusy =
    createTraderMutation.isPending ||
    saveTraderMutation.isPending ||
    deleteTraderMutation.isPending

  const traderNameById = useMemo(
    () => Object.fromEntries(traders.map((trader) => [trader.id, trader.name])) as Record<string, string>,
    [traders]
  )

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

  const filteredSelectedPositionBook = useMemo(() => {
    const query = positionSearch.trim().toLowerCase()
    const rows = selectedPositionBook.filter((row) => {
      if (positionDirectionFilter === 'yes' && !isYesDirection(row.directionSide || row.direction)) return false
      if (positionDirectionFilter === 'no' && !isNoDirection(row.directionSide || row.direction)) return false
      if (!query) return true
      const haystack = `${row.marketQuestion} ${row.marketId} ${row.sourceSummary} ${row.statusSummary} ${row.direction} ${row.directionSide || ''} ${row.executionSummary}`.toLowerCase()
      return haystack.includes(query)
    })
    return sortPositionRows(rows, positionSortField, positionSortDirection)
  }, [
    positionDirectionFilter,
    positionSearch,
    positionSortDirection,
    positionSortField,
    selectedPositionBook,
  ])

  const selectedPositionSummary = useMemo(
    () => summarizePositionRows(filteredSelectedPositionBook),
    [filteredSelectedPositionBook]
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

  const filteredDecisions = useMemo(() => {
    const q = decisionSearch.trim().toLowerCase()
    return selectedDecisions
      .filter((decision) => {
        if (!q) return true
        const haystack = `${decision.source} ${decision.strategy_key} ${decision.reason || ''} ${decision.decision}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 200)
  }, [selectedDecisions, decisionSearch])

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
      const updatedAt = String(order.updated_at || order.mark_updated_at || order.created_at || '')
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
    ? 'Signal events wake the orchestrator immediately with high-frequency monitoring active.'
    : 'Signal events wake the orchestrator immediately.'

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

  const failedOrders = useMemo(
    () => allOrders.filter((order) => FAILED_ORDER_STATUSES.has(normalizeStatus(order.status))).slice(0, 80),
    [allOrders]
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
      const outcome = String(decision.decision || '').toLowerCase()
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

        <div className="flex items-center gap-1.5">
          <Badge className="h-5 px-1.5 text-[10px]" variant={orchestratorBlocked ? 'destructive' : orchestratorRunning ? 'default' : 'secondary'}>
            {orchestratorStatusLabel}
          </Badge>
          <Badge className="h-5 px-1.5 text-[10px]" variant={globalMode === 'live' ? 'destructive' : 'outline'}>
            {globalMode.toUpperCase()}
          </Badge>
          <Badge className="h-5 px-1.5 text-[10px]" variant={killSwitchOn ? 'destructive' : 'outline'}>
            {killSwitchOn ? 'BLOCKED' : 'OPEN'}
          </Badge>
        </div>

        <div className="hidden lg:flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
          <span>{tradersRunningDisplay}/{toNumber(metrics?.traders_total)}</span>
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

        <div className="ml-auto flex items-center gap-1 text-[10px] text-muted-foreground">
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
                <div className="flex items-center justify-between gap-2">
                  <span>All Bots</span>
                  <span className={cn('font-mono text-[10px]', globalSummary.resolvedPnl > 0 ? 'text-emerald-500' : globalSummary.resolvedPnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                    {formatCurrency(globalSummary.resolvedPnl, true)}
                  </span>
                </div>
                <div className="mt-0.5 text-[9px] text-muted-foreground">
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
        <div className="flex flex-col min-h-0 gap-1.5">
          {showingAllBotsDashboard ? (
            <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-cyan-500/25 bg-card">
              <div className="h-full min-h-0 flex flex-col">
                <div className="shrink-0 border-b border-cyan-500/20 px-3 py-2">
                  <div className="flex flex-wrap items-center justify-end gap-1.5">
                    <div className="flex items-center gap-1.5">
                      <Badge className="h-5 px-1.5 text-[10px]" variant={orchestratorRunning ? 'default' : 'secondary'}>
                        Engine {orchestratorStatusLabel}
                      </Badge>
                      <Badge className="h-5 px-1.5 text-[10px]" variant={globalMode === 'live' ? 'destructive' : 'outline'}>
                        {globalMode.toUpperCase()}
                      </Badge>
                      <Badge className="h-5 px-1.5 text-[10px]" variant={killSwitchOn ? 'destructive' : 'outline'}>
                        {killSwitchOn ? 'BLOCKED' : 'OPEN'}
                      </Badge>
                    </div>
                  </div>
                </div>

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
                    <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
                      <div className="min-h-0 flex flex-col gap-2">
                        <div className="flex-1 min-h-0 rounded-md border border-border/60 bg-card/80 overflow-hidden">
                          <div className="px-2.5 py-2 border-b border-border/40 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <Clock3 className="w-3.5 h-3.5 text-cyan-500" />
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Activity</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground">{allBotsActivityRows.length} rows</span>
                          </div>
                          <ScrollArea className="h-[240px] xl:h-full">
                            <div className="space-y-0.5 p-1.5 font-mono text-[11px]">
                              {allBotsActivityRows.length === 0 ? (
                                <p className="py-8 text-center text-muted-foreground text-xs">No activity captured yet.</p>
                              ) : (
                                allBotsActivityRows.slice(0, 60).map((row) => (
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
                                      {row.action ? (
                                        <span className={cn('uppercase text-[10px] font-semibold', row.action === 'BUY' ? 'text-emerald-500' : 'text-red-500')}>
                                          {row.action}
                                        </span>
                                      ) : null}
                                      <span className="text-muted-foreground">
                                        {traderNameById[String(row.traderId || '')] || shortId(row.traderId || '')}
                                      </span>
                                      <span className="font-medium">{row.title}</span>
                                    </div>
                                    <div className="text-[10px] leading-relaxed text-muted-foreground mt-0.5 break-words">{row.detail}</div>
                                  </div>
                                ))
                              )}
                            </div>
                          </ScrollArea>
                        </div>

                        <div className="rounded-md border border-border/60 bg-card/80 overflow-hidden">
                          <div className="px-2.5 py-2 border-b border-border/40 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <Sparkles className="w-3.5 h-3.5 text-cyan-500" />
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Recent Selections</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground">{recentSelectedDecisions.length} entries</span>
                          </div>
                          <ScrollArea className="h-[180px]">
                            <div className="space-y-1 p-2">
                              {recentSelectedDecisions.length === 0 ? (
                                <p className="text-[11px] text-muted-foreground">No recent selections.</p>
                              ) : (
                                recentSelectedDecisions.slice(0, 16).map((decision) => (
                                  <div key={decision.id} className="rounded border border-border/50 px-2 py-1">
                                    <div className="flex items-start justify-between gap-2">
                                      <div className="min-w-0">
                                        <p className="text-[11px] leading-tight truncate" title={decision.market_question || ''}>
                                          {decision.market_question || shortId(decision.market_id)}
                                        </p>
                                        <p className="text-[10px] text-muted-foreground mt-0.5 truncate">
                                          {traderNameById[String(decision.trader_id || '')] || shortId(decision.trader_id || '')}
                                        </p>
                                      </div>
                                      <div className="flex items-center gap-1.5 shrink-0">
                                        <Badge variant="outline" className="h-4 px-1 text-[9px]">
                                          {decision.source}
                                        </Badge>
                                        <span className="text-[10px] font-mono text-muted-foreground">
                                          {formatPercent(normalizeEdgePercent(toNumber(decision.edge_percent)))}
                                        </span>
                                      </div>
                                    </div>
                                    <p className="text-[10px] text-muted-foreground mt-0.5">{formatTimestamp(decision.created_at)}</p>
                                  </div>
                                ))
                              )}
                            </div>
                          </ScrollArea>
                        </div>
                      </div>

                      <div className="min-h-0 flex flex-col gap-2">
                        <div className="rounded-md border border-border/60 bg-card/80 overflow-hidden">
                          <div className="px-2.5 py-2 border-b border-border/40 flex items-center justify-between gap-2">
                            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Bot Leaderboard</span>
                            <span className="text-[10px] font-mono text-muted-foreground">Top {allBotsLeaderboardRows.length}</span>
                          </div>
                          <ScrollArea className="h-[240px]">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="text-[11px]">Bot</TableHead>
                                  <TableHead className="text-[11px] text-right">Open</TableHead>
                                  <TableHead className="text-[11px] text-right">Resolved</TableHead>
                                  <TableHead className="text-[11px] text-right">WR</TableHead>
                                  <TableHead className="text-[11px] text-right">P&amp;L</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {allBotsLeaderboardRows.map((row) => {
                                  const traderStatus = resolveTraderStatusPresentation(row.trader, orchestratorRunning)
                                  return (
                                    <TableRow key={row.trader.id} className="text-xs">
                                      <TableCell className="py-1">
                                        <div className="flex items-center gap-1.5">
                                          <span className={cn(
                                            'w-1.5 h-1.5 rounded-full shrink-0',
                                            traderStatus.dotClassName
                                          )} />
                                          <span className="truncate max-w-[170px]" title={row.trader.name}>{row.trader.name}</span>
                                        </div>
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-1">{row.open}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{row.resolved}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{formatPercent(row.winRate)}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-1', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : '')}>
                                        {formatCurrency(row.pnl)}
                                      </TableCell>
                                    </TableRow>
                                  )
                                })}
                              </TableBody>
                            </Table>
                          </ScrollArea>
                        </div>

                        <div className="grid gap-2 xl:grid-cols-1 2xl:grid-cols-2">
                          <div className="rounded-md border border-border/60 bg-card/80 overflow-hidden">
                            <div className="px-2.5 py-2 border-b border-border/40">
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Source Mix</span>
                            </div>
                            <div className="space-y-1 p-2">
                              {globalSummary.sourceRows.length === 0 ? (
                                <p className="text-[11px] text-muted-foreground">No source activity yet.</p>
                              ) : (
                                globalSummary.sourceRows.slice(0, 6).map((row) => (
                                  <div key={row.source} className="rounded border border-border/50 px-2 py-1">
                                    <div className="flex items-center justify-between gap-2">
                                      <span className="text-[11px] uppercase">{row.source}</span>
                                      <span className="text-[10px] font-mono text-muted-foreground">{row.orders} orders</span>
                                    </div>
                                    <div className="flex items-center justify-between gap-2 text-[10px] mt-0.5">
                                      <span className="text-muted-foreground">Resolved {row.resolved}</span>
                                      <span className={cn('font-mono', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                                        {formatCurrency(row.pnl)}
                                      </span>
                                    </div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>

                          <div className="rounded-md border border-border/60 bg-card/80 overflow-hidden">
                            <div className="px-2.5 py-2 border-b border-border/40 flex items-center justify-between">
                              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Risk Watch</span>
                              <span className="text-[10px] font-mono text-muted-foreground">{riskActivityRows.length} alerts</span>
                            </div>
                            <ScrollArea className="h-[188px]">
                              <div className="space-y-1 p-2">
                                {riskActivityRows.length === 0 ? (
                                  <p className="text-[11px] text-muted-foreground">No active risk alerts.</p>
                                ) : (
                                  riskActivityRows.slice(0, 10).map((row) => (
                                    <div key={`${row.kind}:${row.id}`} className={cn('rounded border px-2 py-1', row.tone === 'negative' ? 'border-red-500/30' : 'border-amber-500/30')}>
                                      <p className="text-[11px] truncate" title={row.title}>{row.title}</p>
                                      <p className="text-[10px] text-muted-foreground">{formatTimestamp(row.ts)}</p>
                                    </div>
                                  ))
                                )}
                              </div>
                            </ScrollArea>
                          </div>
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
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="text-[11px]">Bot</TableHead>
                                  <TableHead className="text-[11px]">Market</TableHead>
                                  <TableHead className="text-[11px]">Dir</TableHead>
                                  <TableHead className="text-[11px]">Lifecycle / Outcome</TableHead>
                                  <TableHead className="text-[11px] text-right">Notional</TableHead>
                                  <TableHead className="text-[11px] text-right">Fill Px</TableHead>
                                  <TableHead className="text-[11px] text-right">Fill Progress</TableHead>
                                  <TableHead className="text-[11px] text-right">Mark Px</TableHead>
                                  <TableHead className="text-[11px] text-right">U-P&amp;L</TableHead>
                                  <TableHead className="text-[11px] text-right">Edge Δ</TableHead>
                                  <TableHead className="text-[11px] text-right">R-P&amp;L</TableHead>
                                  <TableHead className="text-[11px]">Mode</TableHead>
                                  <TableHead className="text-[11px]">Created</TableHead>
                                  <TableHead className="text-[11px]">Updated</TableHead>
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
                                  const linkedDecisionId = String(order.decision_id || '').trim()
                                  const signalPayload = linkedDecisionId
                                    ? decisionSignalPayloadByDecisionId.get(linkedDecisionId) || null
                                    : null
                                  const links = buildOrderMarketLinks(order, orderPayload, signalPayload)
                                  const primaryMarketLink = links.polymarket || links.kalshi
                                  const updatedAt = String(order.updated_at || order.mark_updated_at || order.created_at || '')
                                  const outcome = orderOutcomeSummary(order)
                                  const outcomeDetailCompact = compactText(outcome.detail, 120)
                                  const executionSummary = orderExecutionTypeSummary(order)
                                  const venuePresentation = resolveVenueStatusPresentation(providerSnapshotStatus)
                                  const directionPresentation = resolveOrderDirectionPresentation(order)
                                  return (
                                    <TableRow key={order.id} className="text-xs">
                                      <TableCell className="py-1 max-w-[140px] truncate" title={traderNameById[String(order.trader_id || '')] || shortId(order.trader_id)}>
                                        {traderNameById[String(order.trader_id || '')] || shortId(order.trader_id)}
                                      </TableCell>
                                      <TableCell className="max-w-[260px] truncate py-1" title={order.market_question || order.market_id}>
                                        <div className="flex min-w-0 items-center gap-1">
                                          <div className="flex shrink-0 items-center gap-0.5">
                                            {links.polymarket && (
                                              <a
                                                href={links.polymarket}
                                                target="_blank"
                                                rel="noopener noreferrer"
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
                                      </TableCell>
                                      <TableCell className="py-1">
                                        <Badge
                                          variant="outline"
                                          className="h-5 max-w-[140px] truncate border-border/80 bg-muted/60 px-1.5 text-[10px] text-muted-foreground"
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
                                            {venuePresentation.label !== '\u2014' && (
                                              <Badge
                                                variant="outline"
                                                title={`Venue: ${venuePresentation.detail} • raw:${providerSnapshotStatus}`}
                                                className={cn(
                                                  'h-4 max-w-[120px] truncate px-1 text-[9px] font-medium',
                                                  venuePresentation.className
                                                )}
                                              >
                                                {`V:${venuePresentation.label}`}
                                              </Badge>
                                            )}
                                          </div>
                                          <p className="truncate text-[9px] leading-none text-foreground/85" title={outcome.detail}>
                                            <span className="font-medium text-muted-foreground">Reason:</span> {outcomeDetailCompact}
                                          </p>
                                        </div>
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-1">{formatCurrency(toNumber(order.notional_usd))}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{fillPx > 0 ? fillPx.toFixed(3) : '\u2014'}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{fillProgressPercent !== null ? formatPercent(fillProgressPercent, 0) : '\u2014'}</TableCell>
                                      <TableCell className="text-right font-mono py-1">{markPx > 0 ? markPx.toFixed(3) : '\u2014'}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-1', unrealized > 0 ? 'text-emerald-500' : unrealized < 0 ? 'text-red-500' : '')}>
                                        {OPEN_ORDER_STATUSES.has(status) ? formatCurrency(unrealized) : '\u2014'}
                                      </TableCell>
                                      <TableCell className="text-right font-mono py-1">{formatPercent(dynamicEdgePercent)}</TableCell>
                                      <TableCell className={cn('text-right font-mono py-1', pnl > 0 ? 'text-emerald-500' : pnl < 0 ? 'text-red-500' : '')}>
                                        {RESOLVED_ORDER_STATUSES.has(status) ? formatCurrency(pnl) : '\u2014'}
                                      </TableCell>
                                      <TableCell className="py-1 uppercase text-[10px]">{String(order.mode || '').toUpperCase()}</TableCell>
                                      <TableCell className="py-1 text-[10px] text-muted-foreground">{formatShortDate(order.created_at)}</TableCell>
                                      <TableCell className="py-1 text-[10px] text-muted-foreground" title={formatTimestamp(updatedAt)}>{formatRelativeAge(updatedAt)}</TableCell>
                                    </TableRow>
                                  )
                                })}
                              </TableBody>
                            </Table>
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
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="text-[11px]">Bot</TableHead>
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
                                {filteredAllPositionBook.map((row) => (
                                  <TableRow key={row.key} className="text-xs">
                                    <TableCell className="py-1 max-w-[140px] truncate" title={row.traderName}>
                                      {row.traderName}
                                    </TableCell>
                                    <TableCell className="max-w-[280px] truncate py-1" title={row.marketQuestion}>
                                      <p className="truncate">{row.marketQuestion}</p>
                                      <p className="text-[10px] text-muted-foreground truncate" title={positionMetaLine(row)}>
                                        {positionMetaLine(row)}
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
                                ))}
                              </TableBody>
                            </Table>
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
                  { key: 'terminal' as const, label: 'Terminal' },
                  { key: 'positions' as const, label: 'Positions' },
                  { key: 'trades' as const, label: 'Trades' },
                  { key: 'monitor' as const, label: 'Monitor' },
                  { key: 'decisions' as const, label: 'Decisions' },
                  { key: 'risk' as const, label: 'Risk' },
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

                {workTab === 'positions' && (
                  <div className="h-full flex flex-col min-h-0 gap-1.5">
                    <div className="shrink-0 flex flex-wrap items-center gap-1 px-1">
                      <Input
                        value={positionSearch}
                        onChange={(event) => setPositionSearch(event.target.value)}
                        placeholder="Search market, source..."
                        className="h-6 w-44 text-[11px]"
                      />
                      {(['all', 'yes', 'no'] as PositionDirectionFilter[]).map((direction) => (
                        <Button
                          key={direction}
                          size="sm"
                          variant={positionDirectionFilter === direction ? 'default' : 'outline'}
                          onClick={() => setPositionDirectionFilter(direction)}
                          className="h-5 px-2 text-[10px]"
                        >
                          {direction}
                        </Button>
                      ))}
                      <Select
                        value={positionSortField}
                        onValueChange={(value) => setPositionSortField(value as PositionSortField)}
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
                        onClick={() => setPositionSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'))}
                        className="h-5 px-2 text-[10px]"
                      >
                        {positionSortDirection === 'desc' ? 'desc' : 'asc'}
                      </Button>
                      <span className="ml-auto text-[10px] font-mono text-muted-foreground">{filteredSelectedPositionBook.length} rows</span>
                    </div>
                    <div className="shrink-0 grid grid-cols-2 gap-1 px-1 sm:grid-cols-4 lg:grid-cols-8">
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Positions</p>
                        <p className="text-xs font-mono">{selectedPositionSummary.totalRows}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">YES / NO</p>
                        <p className="text-xs font-mono">{selectedPositionSummary.yesRows} / {selectedPositionSummary.noRows}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Exposure</p>
                        <p className="text-xs font-mono">{formatCurrency(selectedPositionSummary.totalExposure, true)}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">U-P&amp;L</p>
                        <p className={cn(
                          'text-xs font-mono',
                          selectedPositionSummary.totalUnrealizedPnl > 0 ? 'text-emerald-500' : selectedPositionSummary.totalUnrealizedPnl < 0 ? 'text-red-500' : ''
                        )}
                        >
                          {selectedPositionSummary.rowsWithUnrealized > 0
                            ? formatCurrency(selectedPositionSummary.totalUnrealizedPnl, true)
                            : '—'}
                        </p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Avg Edge</p>
                        <p className="text-xs font-mono">{formatPercent(selectedPositionSummary.avgEdge)}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Avg Conf</p>
                        <p className="text-xs font-mono">{formatPercent(normalizeConfidencePercent(selectedPositionSummary.avgConfidence))}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Live / Paper</p>
                        <p className="text-xs font-mono">{selectedPositionSummary.liveOrders} / {selectedPositionSummary.paperOrders}</p>
                      </div>
                      <div className="rounded border border-border/60 bg-background/70 px-2 py-1">
                        <p className="text-[9px] uppercase text-muted-foreground">Marks</p>
                        <p className="text-xs font-mono">{selectedPositionSummary.freshMarks} / {selectedPositionSummary.markedRows}</p>
                      </div>
                    </div>
                    <div className="flex-1 min-h-0 overflow-hidden px-1">
                      {filteredSelectedPositionBook.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No positions matching filters.</div>
                      ) : (
                        <ScrollArea className="h-full min-h-0 rounded-md border border-border/60 bg-card/60">
                          <Table>
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
                              {filteredSelectedPositionBook.map((row) => (
                                <TableRow key={row.key} className="text-xs">
                                  <TableCell className="max-w-[260px] truncate py-1" title={row.marketQuestion}>
                                    <p className="truncate">{row.marketQuestion}</p>
                                    <p className="text-[10px] text-muted-foreground truncate" title={positionMetaLine(row)}>
                                      {positionMetaLine(row)}
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
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      )}
                    </div>
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
                                  directionLabel,
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
                                return (
                                  <TableRow key={order.id} className="text-[11px] leading-tight">
                                    <TableCell className="max-w-[260px] py-0.5" title={order.market_question || order.market_id}>
                                      <div className="flex min-w-0 items-center gap-1">
                                        <div className="flex shrink-0 items-center gap-0.5">
                                          {links.polymarket && (
                                            <a
                                              href={links.polymarket}
                                              target="_blank"
                                              rel="noopener noreferrer"
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
                      </div>
                    </ScrollArea>
                  </div>
                )}

                {workTab === 'decisions' && (
                  <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] px-1">
                    <div className="flex flex-col gap-1.5 min-h-0 overflow-hidden">
                      <Input value={decisionSearch} onChange={(event) => setDecisionSearch(event.target.value)} placeholder="Search decisions..." className="h-6 text-[11px] shrink-0" />
                      <div className="shrink-0 grid gap-1 grid-cols-3">
                        <div className="rounded border border-emerald-500/30 bg-emerald-500/5 px-2 py-1 text-center">
                          <p className="text-[9px] uppercase text-muted-foreground">Selected</p>
                          <p className="text-xs font-mono text-emerald-500">{decisionOutcomeSummary.selected}</p>
                        </div>
                        <div className="rounded border border-red-500/30 bg-red-500/5 px-2 py-1 text-center">
                          <p className="text-[9px] uppercase text-muted-foreground">Blocked</p>
                          <p className="text-xs font-mono text-red-500">{decisionOutcomeSummary.blocked}</p>
                        </div>
                        <div className="rounded border border-border/70 bg-background/70 px-2 py-1 text-center">
                          <p className="text-[9px] uppercase text-muted-foreground">Skipped</p>
                          <p className="text-xs font-mono">{decisionOutcomeSummary.skipped}</p>
                        </div>
                      </div>
                      <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10">
                        <div className="space-y-0.5 p-1.5 text-xs">
                          {filteredDecisions.length === 0 ? (
                            <p className="py-4 text-center text-muted-foreground">No decisions.</p>
                          ) : (
                            filteredDecisions.map((decision) => {
                              const isActive = decision.id === selectedDecisionId
                              const outcome = String(decision.decision || '').toLowerCase()
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
                                    <span className="truncate max-w-[180px]" title={decision.market_question || decision.market_id || ''}>{decision.market_question || shortId(decision.market_id)}</span>
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
                            <p className="font-medium">{selectedDecision.market_question || shortId(selectedDecision.market_id)}</p>
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

                {workTab === 'risk' && (
                  <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] px-1">
                    <div className="flex flex-col min-h-0 gap-1.5 overflow-hidden">
                      <div className="shrink-0 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground px-1">Governance</div>
                      <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10">
                        <div className="space-y-2 p-2">
                          <div className="rounded border border-border/70 p-2 text-xs space-y-1.5">
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Control State</p>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Mode</span><Badge variant={globalMode === 'live' ? 'destructive' : 'outline'}>{globalMode.toUpperCase()}</Badge></div>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Engine</span><span className={cn('font-medium', orchestratorBlocked ? 'text-amber-400' : '')}>{orchestratorStatusLabel}</span></div>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Block orders</span><span className={cn('font-medium', killSwitchOn ? 'text-red-400' : 'text-emerald-400')}>{killSwitchOn ? 'Active' : 'Inactive'}</span></div>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Run interval</span><span className="font-mono">{toNumber(overviewQuery.data?.control?.run_interval_seconds)}s</span></div>
                          </div>
                          <div className={cn('rounded border p-2 text-xs space-y-1.5', !selectedAccountValid || modeMismatch ? 'border-amber-500/35 bg-amber-500/5' : 'border-border/70')}>
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Account Governance</p>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Account selected</span><span className="font-medium">{selectedAccountValid ? 'Yes' : 'No'}</span></div>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Mode sync</span><span className="font-medium">{modeMismatch ? 'No' : 'Yes'}</span></div>
                          </div>
                          <div className="rounded border border-border/70 p-2 text-xs space-y-1.5">
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Live Guardrails</p>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Preflight + arm</span><Badge variant="outline" className="text-[9px] h-4">Enforced</Badge></div>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Kill switch guard</span><Badge variant="outline" className="text-[9px] h-4">Enforced</Badge></div>
                            <div className="flex items-center justify-between"><span className="text-muted-foreground">Worker pause override</span><Badge variant="outline" className="text-[9px] h-4">Enforced</Badge></div>
                          </div>
                        </div>
                      </ScrollArea>
                    </div>

                    <div className="flex flex-col min-h-0 gap-1.5 overflow-hidden">
                      <div className="shrink-0 flex items-center gap-3 px-1">
                        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Risk Terminal</span>
                        <div className="flex items-center gap-2 text-[11px] font-mono text-muted-foreground">
                          <span>{riskActivityRows.length} warnings</span>
                          <span className="text-border">|</span>
                          <span>{failedOrders.length} failed</span>
                        </div>
                      </div>
                      <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10">
                        <div className="space-y-0.5 p-1.5 font-mono text-[11px] leading-relaxed">
                          {riskActivityRows.length === 0 ? (
                            <div className="rounded border border-emerald-500/25 px-2 py-1 text-emerald-700 dark:text-emerald-100">[HEALTHY] no warning or failure events captured.</div>
                          ) : (
                            riskActivityRows.map((row) => (
                              <div key={`${row.kind}:${row.id}`} className={cn('rounded border px-2 py-1', row.tone === 'negative' ? 'border-red-500/30 text-red-700 dark:text-red-100' : 'border-amber-500/30 text-amber-700 dark:text-amber-100')}>
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
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

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
                      <Select
                        value={draftCopyFromTraderId || '__none__'}
                        onValueChange={applyCreateCopyFromSelection}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Start from scratch" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__none__">Start from scratch</SelectItem>
                          {traders.map((trader) => (
                            <SelectItem key={trader.id} value={trader.id}>
                              {trader.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
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
                            </div>
                          ) : null}
                        </div>
                      )
                    })}
                  </div>

                  {/* Crypto inline config — shown when any crypto source is enabled */}
                  {sourceCards.some((s) => isCryptoSourceKey(s.key) && selectedSourceKeySet.has(normalizeSourceKey(s.key))) && (
                    <div className="rounded-lg border border-sky-500/30 bg-sky-500/5 p-2.5 space-y-2 mt-2">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-sky-400">Crypto Settings</p>
                        {(Boolean(cryptoIncludeAssetParamKey) ||
                          Boolean(cryptoIncludeTimeframeParamKey) ||
                          cryptoAssetOptions.length > 0 ||
                          cryptoTimeframeOptions.length > 0) && (
                          <Button type="button" size="sm" variant="outline" className="h-5 px-2 text-[10px]" onClick={enableAllCryptoTargets}>
                            Use all
                          </Button>
                        )}
                      </div>
                      {(Boolean(cryptoIncludeAssetParamKey) || cryptoAssetOptions.length > 0) && (
                        <div className="space-y-1.5">
                          <p className="text-[11px] text-muted-foreground/80">Include Assets</p>
                          {cryptoAssetOptions.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {cryptoAssetOptions.map((asset) => (
                                <Button
                                  key={asset}
                                  type="button"
                                  size="sm"
                                  variant={selectedCryptoAssets.has(asset) ? 'default' : 'outline'}
                                  className="h-6 px-2 text-[11px]"
                                  onClick={() => toggleCryptoAssetTarget(asset)}
                                >
                                  {asset}
                                </Button>
                              ))}
                            </div>
                          ) : (
                            <p className="text-[10px] text-muted-foreground/75">No asset options exposed by this strategy schema.</p>
                          )}
                        </div>
                      )}

                      {Boolean(cryptoExcludeAssetParamKey) && (
                        <div className="space-y-1.5">
                          <p className="text-[11px] text-muted-foreground/80">Exclude Assets</p>
                          {cryptoAssetOptions.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {cryptoAssetOptions.map((asset) => (
                                <Button
                                  key={`exclude-${asset}`}
                                  type="button"
                                  size="sm"
                                  variant={selectedExcludedCryptoAssets.has(asset) ? 'destructive' : 'outline'}
                                  className="h-6 px-2 text-[11px]"
                                  onClick={() => toggleExcludedCryptoAssetTarget(asset)}
                                >
                                  {asset}
                                </Button>
                              ))}
                            </div>
                          ) : (
                            <p className="text-[10px] text-muted-foreground/75">No asset options exposed by this strategy schema.</p>
                          )}
                        </div>
                      )}

                      {cryptoTimeframeDraft ? (
                        <div className="space-y-1.5">
                          <p className="text-[11px] text-muted-foreground/80">
                            Timeframe is fixed by selected crypto strategy ({cryptoTimeframeDraft}).
                          </p>
                        </div>
                      ) : null}

                      {(Boolean(cryptoIncludeTimeframeParamKey) || cryptoTimeframeOptions.length > 0) && !cryptoTimeframeDraft && (
                        <div className="space-y-1.5">
                          <p className="text-[11px] text-muted-foreground/80">Include Timeframes</p>
                          {cryptoTimeframeOptions.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {cryptoTimeframeOptions.map((timeframe) => (
                                <Button
                                  key={timeframe}
                                  type="button"
                                  size="sm"
                                  variant={selectedCryptoIncludedTimeframes.has(timeframe) ? 'default' : 'outline'}
                                  className="h-6 px-2 text-[11px]"
                                  onClick={() => toggleCryptoTimeframeTarget(timeframe)}
                                >
                                  {timeframe}
                                </Button>
                              ))}
                            </div>
                          ) : (
                            <p className="text-[10px] text-muted-foreground/75">No timeframe options exposed by this strategy schema.</p>
                          )}
                        </div>
                      )}

                      {Boolean(cryptoExcludeTimeframeParamKey) && !cryptoTimeframeDraft && (
                        <div className="space-y-1.5">
                          <p className="text-[11px] text-muted-foreground/80">Exclude Timeframes</p>
                          {cryptoTimeframeOptions.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {cryptoTimeframeOptions.map((timeframe) => (
                                <Button
                                  key={`exclude-${timeframe}`}
                                  type="button"
                                  size="sm"
                                  variant={selectedCryptoExcludedTimeframes.has(timeframe) ? 'destructive' : 'outline'}
                                  className="h-6 px-2 text-[11px]"
                                  onClick={() => toggleExcludedCryptoTimeframeTarget(timeframe)}
                                >
                                  {timeframe}
                                </Button>
                              ))}
                            </div>
                          ) : (
                            <p className="text-[10px] text-muted-foreground/75">No timeframe options exposed by this strategy schema.</p>
                          )}
                        </div>
                      )}

                      {(Boolean(cryptoModeParamKey) || cryptoStrategyModeOptions.length > 0) && (
                        <div className="mt-1">
                          <Label className="text-[11px] text-muted-foreground">Strategy Mode</Label>
                          <Select
                            value={normalizeCryptoStrategyMode(
                              advancedConfig.strategyMode,
                              cryptoStrategyModeOptions,
                              cryptoStrategyModeOptions[0] || 'auto'
                            )}
                            onValueChange={(value) =>
                              setAdvancedValue(
                                'strategyMode',
                                normalizeCryptoStrategyMode(
                                  value,
                                  cryptoStrategyModeOptions,
                                  cryptoStrategyModeOptions[0] || 'auto'
                                )
                              )
                            }
                          >
                            <SelectTrigger className="h-7 text-xs mt-0.5">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {cryptoStrategyModeOptions.map((mode) => (
                                <SelectItem key={mode} value={mode} className="text-xs">{mode}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                      {!cryptoTimeframeDraft &&
                        !(Boolean(cryptoIncludeAssetParamKey) || cryptoAssetOptions.length > 0) &&
                        !Boolean(cryptoExcludeAssetParamKey) &&
                        !(Boolean(cryptoIncludeTimeframeParamKey) || cryptoTimeframeOptions.length > 0) &&
                        !Boolean(cryptoExcludeTimeframeParamKey) &&
                        !(Boolean(cryptoModeParamKey) || cryptoStrategyModeOptions.length > 0) && (
                        <p className="text-[10px] text-muted-foreground/75">
                          Selected strategy does not expose dedicated crypto controls in its schema.
                        </p>
                      )}
                    </div>
                  )}

                  {/* Wallet-signal inline config — shown when traders source is enabled */}
                  {sourceCards.some((s) => isTraderSourceKey(s.key) && selectedSourceKeySet.has(normalizeSourceKey(s.key))) && (
                    <div className="rounded-lg border border-orange-500/30 bg-orange-500/5 p-2.5 space-y-3 mt-2">
                      <div className="border border-orange-500/30 bg-background/60 rounded-md p-2.5 space-y-2">
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-orange-400">Strategy Parameters</p>
                        {tradersStrategyFormSchema.param_fields.length > 0 ? (
                          <StrategyConfigForm
                            schema={tradersStrategyFormSchema as { param_fields: any[] }}
                            values={tradersStrategyFormValues}
                            onChange={applyTradersStrategyFormValues}
                          />
                        ) : (
                          <p className="text-[10px] text-muted-foreground/80">No strategy schema available for selected wallet source strategy.</p>
                        )}
                        {draftTradersScopeModes.includes('individual') ? (
                          <div className="space-y-1.5">
                            <p className="text-[11px] text-muted-foreground leading-tight">Individual Wallets</p>
                            <div className="max-h-28 overflow-auto rounded border border-border/70 p-1.5 space-y-1">
                              {trackedWallets.length === 0 ? (
                                <p className="text-[10px] text-muted-foreground">No tracked wallets found.</p>
                              ) : (
                                trackedWallets.map((wallet) => {
                                  const address = String(wallet.address || '').toLowerCase()
                                  const selected = draftTradersIndividualWallets.includes(address)
                                  return (
                                    <label key={wallet.address} className="flex items-center gap-1.5 text-[10px]">
                                      <input
                                        type="checkbox"
                                        checked={selected}
                                        onChange={() => toggleIndividualWallet(address)}
                                        className="accent-orange-500"
                                      />
                                      <span className="font-mono">{wallet.label ? `${wallet.label} (${wallet.address})` : wallet.address}</span>
                                    </label>
                                  )
                                })
                              )}
                            </div>
                          </div>
                        ) : null}
                        {draftTradersScopeModes.includes('group') ? (
                          <div className="space-y-1.5">
                            <p className="text-[11px] text-muted-foreground leading-tight">Discovery Groups</p>
                            <div className="max-h-28 overflow-auto rounded border border-border/70 p-1.5 space-y-1">
                              {traderGroups.length === 0 ? (
                                <p className="text-[10px] text-muted-foreground">No groups found.</p>
                              ) : (
                                traderGroups.map((group) => {
                                  const selected = draftTradersGroupIds.includes(group.id)
                                  return (
                                    <label key={group.id} className="flex items-center gap-1.5 text-[10px]">
                                      <input
                                        type="checkbox"
                                        checked={selected}
                                        onChange={() => toggleTradersGroup(group.id)}
                                        className="accent-orange-500"
                                      />
                                      <span>{group.name}</span>
                                    </label>
                                  )
                                })
                              )}
                            </div>
                          </div>
                        ) : null}
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-1.5">
                          <Filter className="w-3.5 h-3.5 text-orange-400" />
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-orange-400">Wallet Signal Filters (Global)</p>
                        </div>
                        <p className="text-[10px] text-muted-foreground/80">
                          These filters are shared with wallet-opportunity discovery settings.
                        </p>
                        {traderOpportunityFilterSchema.param_fields.length > 0 ? (
                          <StrategyConfigForm
                            schema={traderOpportunityFilterSchema as { param_fields: any[] }}
                            values={draftSignalFilters as Record<string, unknown>}
                            onChange={applySignalFilterFormValues}
                          />
                        ) : (
                          <p className="text-[10px] text-muted-foreground/80">No trader-opportunity filter schema available.</p>
                        )}
                      </div>

                      <div className="border-t border-orange-500/20 pt-3 space-y-2.5">
                        <div className="flex items-center gap-1.5">
                          <Copy className="w-3.5 h-3.5 text-green-400" />
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-green-400">Copy Trading</p>
                          {draftCopyTrading.copy_mode_type !== 'disabled' && (
                            <span className="ml-auto text-[9px] px-1.5 py-0.5 rounded-full bg-green-500/20 text-green-400 font-medium">ACTIVE</span>
                          )}
                        </div>
                        <div className="grid gap-2.5 md:grid-cols-2">
                          <div>
                            <Label className="text-[11px] text-muted-foreground leading-tight">Simulation Account</Label>
                            <Select
                              value={copyTradingFormValues.account_id ? String(copyTradingFormValues.account_id) : '__none__'}
                              onValueChange={(value) => setCt({ account_id: value === '__none__' ? '' : value })}
                            >
                              <SelectTrigger className="mt-0.5 h-7 text-xs">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="__none__">Unassigned</SelectItem>
                                {simulationAccounts.map((account) => (
                                  <SelectItem key={account.id} value={account.id}>
                                    {account.name || account.id}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          {draftCopyTrading.copy_mode_type === 'individual' ? (
                            <div>
                              <Label className="text-[11px] text-muted-foreground leading-tight">Wallet Address</Label>
                              <Input
                                type="text"
                                value={draftCopyTrading.individual_wallet}
                                onChange={(event) => setCt({ individual_wallet: event.target.value })}
                                placeholder="0x..."
                                className="mt-0.5 text-xs h-7 font-mono"
                              />
                            </div>
                          ) : null}
                        </div>
                        {copyTradingSchema.param_fields.length > 0 ? (
                          <StrategyConfigForm
                            schema={copyTradingSchema as { param_fields: any[] }}
                            values={copyTradingFormValues}
                            onChange={applyCopyTradingFormValues}
                          />
                        ) : (
                          <p className="text-[10px] text-muted-foreground/80">No copy-trading schema available.</p>
                        )}
                        {activeCopyMode && activeCopyMode.stats && activeCopyMode.mode !== 'disabled' && (
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
                    </div>
                  )}

                </FlyoutSection>

                <FlyoutSection
                  title="Run Cadence"
                  icon={Clock3}
                  iconClassName="text-sky-500"
                  count={`${Number(draftInterval || 0)}s`}
                  subtitle="Scheduling only. Strategy/selection logic is configured in source strategy schemas."
                >
                  <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                    <div>
                      <Label>Bot Interval Seconds</Label>
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
