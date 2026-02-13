import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  Loader2,
  Play,
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
  getTraderDecisionDetail,
  getTraderDecisions,
  getTraderEvents,
  getTraderOrchestratorOverview,
  getTraderSources,
  getSimulationAccounts,
  getTraderTemplates,
  getTraders,
  pauseTrader,
  runTraderOnce,
  runTraderOrchestratorLivePreflight,
  setTraderOrchestratorLiveKillSwitch,
  startTrader,
  startTraderOrchestrator,
  startTraderOrchestratorLive,
  stopTraderOrchestrator,
  stopTraderOrchestratorLive,
  type Trader,
  type TraderOrder,
  type TraderSource,
  updateTrader,
} from '../services/api'
import { cn } from '../lib/utils'
import { selectedAccountIdAtom } from '../store/atoms'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { Switch } from './ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

type FeedFilter = 'all' | 'decision' | 'order' | 'event'
type ScopeFilter = 'all' | 'selected'
type TradeStatusFilter = 'all' | 'open' | 'resolved' | 'failed'

type ActivityRow = {
  kind: 'decision' | 'order' | 'event'
  id: string
  ts: string | null
  traderId: string | null
  title: string
  detail: string
  tone: 'neutral' | 'positive' | 'negative' | 'warning'
}

type PositionBookRow = {
  key: string
  traderId: string
  traderName: string
  marketId: string
  marketQuestion: string
  direction: string
  exposureUsd: number
  averagePrice: number | null
  weightedEdge: number | null
  weightedConfidence: number | null
  orderCount: number
  lastUpdated: string | null
  statusSummary: string
}

const CRYPTO_STRATEGY_MODES = ['auto', 'directional', 'pure_arb', 'rebalance'] as const
const CRYPTO_ASSET_OPTIONS = ['BTC', 'ETH', 'SOL', 'XRP'] as const
const CRYPTO_TIMEFRAME_OPTIONS = ['5m', '15m', '1h', '4h'] as const
type CryptoStrategyMode = (typeof CRYPTO_STRATEGY_MODES)[number]

type TraderAdvancedConfig = {
  cadenceProfile: string
  strategyMode: CryptoStrategyMode
  cryptoAssetsCsv: string
  cryptoTimeframesCsv: string
  minSignalScore: number
  minEdgePercent: number
  minConfidence: number
  lookbackMinutes: number
  scanBatchSize: number
  maxSignalsPerCycle: number
  requireSecondSource: boolean
  sourcePriorityCsv: string
  blockedKeywordsCsv: string
  maxOrdersPerCycle: number
  maxOpenOrders: number
  maxOpenPositions: number
  maxPositionNotionalUsd: number
  maxGrossExposureUsd: number
  maxTradeNotionalUsd: number
  maxDailyLossUsd: number
  maxDailySpendUsd: number
  cooldownSeconds: number
  orderTtlSeconds: number
  slippageBps: number
  maxSpreadBps: number
  retryLimit: number
  retryBackoffMs: number
  allowAveraging: boolean
  useDynamicSizing: boolean
  haltOnConsecutiveLosses: boolean
  maxConsecutiveLosses: number
  circuitBreakerDrawdownPct: number
  tradingWindowStartUtc: string
  tradingWindowEndUtc: string
  tagsCsv: string
  notes: string
}

type TraderSourceGroupKey = 'markets' | 'crypto' | 'pool_watched' | 'other'

type TraderSourceGroupMeta = {
  key: TraderSourceGroupKey
  label: string
  subtitle: string
}

const OPEN_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])
const RESOLVED_ORDER_STATUSES = new Set(['resolved', 'resolved_win', 'resolved_loss', 'win', 'loss'])
const FAILED_ORDER_STATUSES = new Set(['failed', 'rejected', 'error', 'cancelled'])

const FALLBACK_TRADER_SOURCES: TraderSource[] = [
  {
    key: 'crypto',
    label: 'Crypto Markets',
    description: 'Crypto microstructure and 5m/15m market signals.',
    domains: ['crypto'],
    signal_types: ['crypto_market'],
  },
  {
    key: 'insider',
    label: 'Insider Signals',
    description: 'Insider and smart-wallet behavior intents.',
    domains: ['event_markets'],
    signal_types: ['insider_intent'],
  },
  {
    key: 'news',
    label: 'News Workflow',
    description: 'News-driven intents and event reactions.',
    domains: ['event_markets'],
    signal_types: ['news_intent'],
  },
  {
    key: 'scanner',
    label: 'General Opportunities',
    description: 'Scanner-originated arbitrage opportunities.',
    domains: ['event_markets'],
    signal_types: ['opportunity'],
  },
  {
    key: 'tracked_traders',
    label: 'Tracked Traders',
    description: 'Signals synthesized from tracked trader activity.',
    domains: ['event_markets'],
    signal_types: ['tracked_trader'],
  },
  {
    key: 'weather',
    label: 'Weather Workflow',
    description: 'Weather forecast probability dislocations.',
    domains: ['event_markets'],
    signal_types: ['weather_intent'],
  },
  {
    key: 'world_intelligence',
    label: 'World Intelligence',
    description: 'Geopolitical conflict and tension opportunity signals.',
    domains: ['event_markets'],
    signal_types: ['world_intelligence'],
  },
]

const TRADER_SOURCE_GROUPS: TraderSourceGroupMeta[] = [
  {
    key: 'markets',
    label: 'Markets',
    subtitle: 'General event and workflow opportunities',
  },
  {
    key: 'crypto',
    label: 'Crypto',
    subtitle: 'Fast crypto market signals',
  },
  {
    key: 'pool_watched',
    label: 'Pool / Watched Traders',
    subtitle: 'Insider and tracked trader signal pools',
  },
  {
    key: 'other',
    label: 'Other',
    subtitle: 'Custom or uncategorized adapters',
  },
]

function toNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function isCryptoStrategyKey(value: string): boolean {
  const key = String(value || '').trim().toLowerCase()
  return key === 'crypto_15m' || key.includes('crypto') || key.includes('btc')
}

function normalizeCryptoStrategyMode(value: unknown): CryptoStrategyMode {
  const mode = String(value || '').trim().toLowerCase()
  if (CRYPTO_STRATEGY_MODES.includes(mode as CryptoStrategyMode)) {
    return mode as CryptoStrategyMode
  }
  return 'auto'
}

function normalizeCryptoAsset(value: unknown): string | null {
  const asset = String(value || '').trim().toUpperCase()
  if (!asset) return null
  if (asset === 'XBT') return 'BTC'
  return CRYPTO_ASSET_OPTIONS.includes(asset as (typeof CRYPTO_ASSET_OPTIONS)[number]) ? asset : null
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

function normalizeCryptoAssetList(value: unknown): string[] {
  const selected = new Set(
    toStringList(value)
      .map((item) => normalizeCryptoAsset(item))
      .filter((item): item is string => Boolean(item))
  )
  return CRYPTO_ASSET_OPTIONS.filter((item) => selected.has(item))
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

function confidencePercentToFraction(value: number): number {
  if (!Number.isFinite(value)) return 0
  const normalized = Math.abs(value) <= 1 ? value : value / 100
  return Math.max(0, Math.min(1, normalized))
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
  return ts.toLocaleDateString()
}

function shortId(value: string | null | undefined): string {
  if (!value) return 'n/a'
  return value.length <= 12 ? value : `${value.slice(0, 6)}...${value.slice(-4)}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
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

function toCsv(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean).join(', ')
  }
  if (typeof value === 'string') return value
  return ''
}

function csvToList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function normalizeSourceKey(value: string): string {
  return String(value || '').trim().toLowerCase()
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

function classifyTraderSource(source: Pick<TraderSource, 'key' | 'domains'>): TraderSourceGroupKey {
  const sourceKey = normalizeSourceKey(source.key)
  const domains = (source.domains || []).map((item) => normalizeSourceKey(item))
  if (sourceKey === 'crypto' || domains.some((item) => item.includes('crypto'))) {
    return 'crypto'
  }
  if (
    sourceKey.includes('tracked') ||
    sourceKey.includes('watch') ||
    sourceKey.includes('pool') ||
    sourceKey.includes('insider')
  ) {
    return 'pool_watched'
  }
  if (
    domains.some((item) => item.includes('market')) ||
    sourceKey.includes('scanner') ||
    sourceKey.includes('news') ||
    sourceKey.includes('weather') ||
    sourceKey.includes('world')
  ) {
    return 'markets'
  }
  return 'other'
}

function defaultAdvancedConfig(): TraderAdvancedConfig {
  return {
    cadenceProfile: 'custom',
    strategyMode: 'auto',
    cryptoAssetsCsv: CRYPTO_ASSET_OPTIONS.join(', '),
    cryptoTimeframesCsv: CRYPTO_TIMEFRAME_OPTIONS.join(', '),
    minSignalScore: 0.1,
    minEdgePercent: 8,
    minConfidence: 60,
    lookbackMinutes: 240,
    scanBatchSize: 240,
    maxSignalsPerCycle: 24,
    requireSecondSource: false,
    sourcePriorityCsv: '',
    blockedKeywordsCsv: '',
    maxOrdersPerCycle: 6,
    maxOpenOrders: 20,
    maxOpenPositions: 12,
    maxPositionNotionalUsd: 350,
    maxGrossExposureUsd: 2000,
    maxTradeNotionalUsd: 350,
    maxDailyLossUsd: 300,
    maxDailySpendUsd: 2000,
    cooldownSeconds: 0,
    orderTtlSeconds: 1200,
    slippageBps: 35,
    maxSpreadBps: 75,
    retryLimit: 2,
    retryBackoffMs: 250,
    allowAveraging: false,
    useDynamicSizing: true,
    haltOnConsecutiveLosses: true,
    maxConsecutiveLosses: 4,
    circuitBreakerDrawdownPct: 12,
    tradingWindowStartUtc: '00:00',
    tradingWindowEndUtc: '23:59',
    tagsCsv: '',
    notes: '',
  }
}

function computeAdvancedConfig(
  params: Record<string, unknown>,
  risk: Record<string, unknown>,
  metadata: Record<string, unknown>
): TraderAdvancedConfig {
  const defaults = defaultAdvancedConfig()
  const tradingWindow = isRecord(metadata.trading_window_utc) ? metadata.trading_window_utc : {}
  const configuredCryptoAssets = normalizeCryptoAssetList(
    params.target_assets ?? params.allowed_assets ?? params.assets ?? params.coins
  )
  const configuredCryptoTimeframes = normalizeCryptoTimeframeList(
    params.target_timeframes ?? params.allowed_timeframes ?? params.timeframes ?? params.cadence
  )

  return {
    cadenceProfile: String(metadata.cadence_profile || defaults.cadenceProfile),
    strategyMode: normalizeCryptoStrategyMode(params.strategy_mode ?? params.mode ?? defaults.strategyMode),
    cryptoAssetsCsv: (configuredCryptoAssets.length > 0 ? configuredCryptoAssets : [...CRYPTO_ASSET_OPTIONS]).join(', '),
    cryptoTimeframesCsv: (configuredCryptoTimeframes.length > 0 ? configuredCryptoTimeframes : [...CRYPTO_TIMEFRAME_OPTIONS]).join(', '),
    minSignalScore: toNumber(params.min_signal_score ?? defaults.minSignalScore),
    minEdgePercent: toNumber(params.min_edge_percent ?? defaults.minEdgePercent),
    minConfidence: normalizeConfidencePercent(toNumber(params.min_confidence ?? defaults.minConfidence)),
    lookbackMinutes: toNumber(params.lookback_minutes ?? defaults.lookbackMinutes),
    scanBatchSize: toNumber(params.scan_batch_size ?? defaults.scanBatchSize),
    maxSignalsPerCycle: toNumber(params.max_signals_per_cycle ?? defaults.maxSignalsPerCycle),
    requireSecondSource: toBoolean(params.require_second_source, defaults.requireSecondSource),
    sourcePriorityCsv: toCsv(params.source_priority ?? defaults.sourcePriorityCsv),
    blockedKeywordsCsv: toCsv(params.blocked_market_keywords ?? defaults.blockedKeywordsCsv),
    maxOrdersPerCycle: toNumber(risk.max_orders_per_cycle ?? defaults.maxOrdersPerCycle),
    maxOpenOrders: toNumber(risk.max_open_orders ?? defaults.maxOpenOrders),
    maxOpenPositions: toNumber(risk.max_open_positions ?? defaults.maxOpenPositions),
    maxPositionNotionalUsd: toNumber(risk.max_position_notional_usd ?? defaults.maxPositionNotionalUsd),
    maxGrossExposureUsd: toNumber(risk.max_gross_exposure_usd ?? defaults.maxGrossExposureUsd),
    maxTradeNotionalUsd: toNumber(risk.max_trade_notional_usd ?? defaults.maxTradeNotionalUsd),
    maxDailyLossUsd: toNumber(risk.max_daily_loss_usd ?? defaults.maxDailyLossUsd),
    maxDailySpendUsd: toNumber(risk.max_daily_spend_usd ?? defaults.maxDailySpendUsd),
    cooldownSeconds: toNumber(risk.cooldown_seconds ?? defaults.cooldownSeconds),
    orderTtlSeconds: toNumber(risk.order_ttl_seconds ?? defaults.orderTtlSeconds),
    slippageBps: toNumber(risk.slippage_bps ?? defaults.slippageBps),
    maxSpreadBps: toNumber(risk.max_spread_bps ?? defaults.maxSpreadBps),
    retryLimit: toNumber(risk.retry_limit ?? defaults.retryLimit),
    retryBackoffMs: toNumber(risk.retry_backoff_ms ?? defaults.retryBackoffMs),
    allowAveraging: toBoolean(risk.allow_averaging, defaults.allowAveraging),
    useDynamicSizing: toBoolean(risk.use_dynamic_sizing, defaults.useDynamicSizing),
    haltOnConsecutiveLosses: toBoolean(risk.halt_on_consecutive_losses, defaults.haltOnConsecutiveLosses),
    maxConsecutiveLosses: toNumber(risk.max_consecutive_losses ?? defaults.maxConsecutiveLosses),
    circuitBreakerDrawdownPct: toNumber(risk.circuit_breaker_drawdown_pct ?? defaults.circuitBreakerDrawdownPct),
    tradingWindowStartUtc: String(tradingWindow.start || defaults.tradingWindowStartUtc),
    tradingWindowEndUtc: String(tradingWindow.end || defaults.tradingWindowEndUtc),
    tagsCsv: toCsv(metadata.tags ?? defaults.tagsCsv),
    notes: String(metadata.notes || defaults.notes),
  }
}

function withConfiguredParams(
  raw: Record<string, unknown>,
  config: TraderAdvancedConfig,
  strategyKey: string
): Record<string, unknown> {
  const targetAssets = normalizeCryptoAssetList(config.cryptoAssetsCsv)
  const targetTimeframes = normalizeCryptoTimeframeList(config.cryptoTimeframesCsv)
  const next: Record<string, unknown> = {
    ...raw,
    min_signal_score: config.minSignalScore,
    min_edge_percent: config.minEdgePercent,
    min_confidence: confidencePercentToFraction(config.minConfidence),
    lookback_minutes: config.lookbackMinutes,
    scan_batch_size: config.scanBatchSize,
    max_signals_per_cycle: config.maxSignalsPerCycle,
    require_second_source: config.requireSecondSource,
    source_priority: csvToList(config.sourcePriorityCsv),
    blocked_market_keywords: csvToList(config.blockedKeywordsCsv),
  }
  if (isCryptoStrategyKey(strategyKey)) {
    next.strategy_mode = config.strategyMode
    next.target_assets = targetAssets.length > 0 ? targetAssets : [...CRYPTO_ASSET_OPTIONS]
    next.target_timeframes = targetTimeframes.length > 0 ? targetTimeframes : [...CRYPTO_TIMEFRAME_OPTIONS]
  } else {
    delete next.strategy_mode
    delete next.target_assets
    delete next.target_timeframes
  }
  return next
}

function withConfiguredRiskLimits(
  raw: Record<string, unknown>,
  config: TraderAdvancedConfig
): Record<string, unknown> {
  return {
    ...raw,
    max_orders_per_cycle: config.maxOrdersPerCycle,
    max_open_orders: config.maxOpenOrders,
    max_open_positions: config.maxOpenPositions,
    max_position_notional_usd: config.maxPositionNotionalUsd,
    max_gross_exposure_usd: config.maxGrossExposureUsd,
    max_trade_notional_usd: config.maxTradeNotionalUsd,
    max_daily_loss_usd: config.maxDailyLossUsd,
    max_daily_spend_usd: config.maxDailySpendUsd,
    cooldown_seconds: config.cooldownSeconds,
    order_ttl_seconds: config.orderTtlSeconds,
    slippage_bps: config.slippageBps,
    max_spread_bps: config.maxSpreadBps,
    retry_limit: config.retryLimit,
    retry_backoff_ms: config.retryBackoffMs,
    allow_averaging: config.allowAveraging,
    use_dynamic_sizing: config.useDynamicSizing,
    halt_on_consecutive_losses: config.haltOnConsecutiveLosses,
    max_consecutive_losses: config.maxConsecutiveLosses,
    circuit_breaker_drawdown_pct: config.circuitBreakerDrawdownPct,
  }
}

function withConfiguredMetadata(
  raw: Record<string, unknown>,
  config: TraderAdvancedConfig
): Record<string, unknown> {
  return {
    ...raw,
    cadence_profile: config.cadenceProfile,
    trading_window_utc: {
      start: config.tradingWindowStartUtc,
      end: config.tradingWindowEndUtc,
    },
    tags: csvToList(config.tagsCsv),
    notes: config.notes,
  }
}

function cadenceProfileForInterval(seconds: number): string {
  if (seconds === 2) return 'ultra_fast'
  if (seconds === 5) return 'fast'
  if (seconds === 10) return 'balanced'
  if (seconds === 30) return 'slow'
  return 'custom'
}

function buildPositionBookRows(orders: TraderOrder[], traderNameById: Record<string, string>): PositionBookRow[] {
  const buckets = new Map<string, {
    traderId: string
    traderName: string
    marketId: string
    marketQuestion: string
    direction: string
    exposureUsd: number
    weightedPrice: number
    weightedEdge: number
    edgeWeight: number
    weightedConfidence: number
    confidenceWeight: number
    orderCount: number
    lastUpdated: string | null
    statuses: Set<string>
  }>()

  for (const order of orders) {
    const status = normalizeStatus(order.status)
    if (!OPEN_ORDER_STATUSES.has(status)) continue

    const traderId = String(order.trader_id || 'unknown')
    const marketId = String(order.market_id || 'unknown')
    const direction = String(order.direction || 'flat').toUpperCase()
    const key = `${traderId}:${marketId}:${direction}`
    const notional = Math.abs(toNumber(order.notional_usd))
    const px = toNumber(order.effective_price ?? order.entry_price)
    const edge = toNumber(order.edge_percent)
    const confidence = toNumber(order.confidence)
    const traderName = traderNameById[traderId] || shortId(traderId)

    if (!buckets.has(key)) {
      buckets.set(key, {
        traderId,
        traderName,
        marketId,
        marketQuestion: String(order.market_question || order.market_id || 'Unknown market'),
        direction,
        exposureUsd: 0,
        weightedPrice: 0,
        weightedEdge: 0,
        edgeWeight: 0,
        weightedConfidence: 0,
        confidenceWeight: 0,
        orderCount: 0,
        lastUpdated: null,
        statuses: new Set<string>(),
      })
    }

    const bucket = buckets.get(key)
    if (!bucket) continue

    bucket.exposureUsd += notional
    bucket.weightedPrice += px > 0 && notional > 0 ? px * notional : 0
    bucket.weightedEdge += edge !== 0 && notional > 0 ? edge * notional : 0
    bucket.edgeWeight += edge !== 0 && notional > 0 ? notional : 0
    bucket.weightedConfidence += confidence !== 0 && notional > 0 ? confidence * notional : 0
    bucket.confidenceWeight += confidence !== 0 && notional > 0 ? notional : 0
    bucket.orderCount += 1
    bucket.lastUpdated = toTs(order.updated_at) > toTs(bucket.lastUpdated)
      ? (order.updated_at || order.executed_at || order.created_at || null)
      : bucket.lastUpdated
    bucket.statuses.add(status)
  }

  return Array.from(buckets.entries())
    .map(([key, bucket]) => ({
      key,
      traderId: bucket.traderId,
      traderName: bucket.traderName,
      marketId: bucket.marketId,
      marketQuestion: bucket.marketQuestion,
      direction: bucket.direction,
      exposureUsd: bucket.exposureUsd,
      averagePrice: bucket.exposureUsd > 0 ? bucket.weightedPrice / bucket.exposureUsd : null,
      weightedEdge: bucket.edgeWeight > 0 ? bucket.weightedEdge / bucket.edgeWeight : null,
      weightedConfidence: bucket.confidenceWeight > 0 ? bucket.weightedConfidence / bucket.confidenceWeight : null,
      orderCount: bucket.orderCount,
      lastUpdated: bucket.lastUpdated,
      statusSummary: Array.from(bucket.statuses).join(', '),
    }))
    .sort((a, b) => b.exposureUsd - a.exposureUsd)
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

export default function TradingPanel() {
  const queryClient = useQueryClient()
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const [feedFilter, setFeedFilter] = useState<FeedFilter>('all')
  const [traderFeedFilter, setTraderFeedFilter] = useState<FeedFilter>('all')
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('all')
  const [tradeStatusFilter, setTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [tradeSearch, setTradeSearch] = useState('')
  const [decisionSearch, setDecisionSearch] = useState('')
  const [confirmLiveStartOpen, setConfirmLiveStartOpen] = useState(false)

  const [traderFlyoutOpen, setTraderFlyoutOpen] = useState(false)
  const [traderFlyoutMode, setTraderFlyoutMode] = useState<'create' | 'edit'>('create')
  const [templateSelection, setTemplateSelection] = useState<string>('none')
  const [draftName, setDraftName] = useState('')
  const [draftDescription, setDraftDescription] = useState('')
  const [draftStrategyKey, setDraftStrategyKey] = useState('')
  const [draftInterval, setDraftInterval] = useState('60')
  const [draftSources, setDraftSources] = useState('')
  const [draftEnabled, setDraftEnabled] = useState(true)
  const [draftPaused, setDraftPaused] = useState(false)
  const [draftParams, setDraftParams] = useState('{}')
  const [draftRisk, setDraftRisk] = useState('{}')
  const [draftMetadata, setDraftMetadata] = useState('{}')
  const [advancedConfig, setAdvancedConfig] = useState<TraderAdvancedConfig>(defaultAdvancedConfig())
  const [saveError, setSaveError] = useState<string | null>(null)
  const [deleteAction, setDeleteAction] = useState<'block' | 'disable' | 'force_delete'>('disable')
  const [deleteConfirmName, setDeleteConfirmName] = useState('')

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

  const templatesQuery = useQuery({
    queryKey: ['traders-templates'],
    queryFn: getTraderTemplates,
    staleTime: 60000,
  })

  const traderSourcesQuery = useQuery({
    queryKey: ['trader-sources'],
    queryFn: getTraderSources,
    staleTime: 300000,
  })

  const simulationAccountsQuery = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    staleTime: 30000,
  })

  const traders = tradersQuery.data || []
  const templates = templatesQuery.data || []
  const sourceCatalog = traderSourcesQuery.data?.length ? traderSourcesQuery.data : FALLBACK_TRADER_SOURCES
  const defaultSourceKeys = useMemo(
    () => uniqueSourceList(sourceCatalog.map((source) => source.key)),
    [sourceCatalog]
  )
  const defaultSourceCsv = useMemo(() => defaultSourceKeys.join(', '), [defaultSourceKeys])

  const traderIds = useMemo(() => traders.map((trader) => trader.id), [traders])
  const traderIdsKey = useMemo(() => traderIds.join('|'), [traderIds])

  const allOrdersQuery = useQuery({
    queryKey: ['trader-orders-all'],
    queryFn: () => getAllTraderOrders(220),
    enabled: traderIds.length > 0,
    refetchInterval: 5000,
  })

  const allDecisionsQuery = useQuery({
    queryKey: ['trader-decisions-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: 5000,
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
    refetchInterval: 5000,
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

  const selectedTrader = useMemo(
    () => traders.find((trader) => trader.id === selectedTraderId) || null,
    [traders, selectedTraderId]
  )

  const selectedOrders = useMemo(
    () => allOrders.filter((order) => order.trader_id === selectedTraderId),
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

  const sourceCards = useMemo(() => {
    const known = uniqueSourceList(sourceCatalog.map((source) => source.key))
      .map((key) => sourceCatalog.find((source) => normalizeSourceKey(source.key) === normalizeSourceKey(key)))
      .filter((source): source is TraderSource => Boolean(source))
      .map((source) => ({
        ...source,
        isLegacy: false,
      }))

    const knownKeys = new Set(known.map((source) => normalizeSourceKey(source.key)))
    const selected = csvToList(draftSources)
    const legacy = selected
      .filter((sourceKey) => !knownKeys.has(normalizeSourceKey(sourceKey)))
      .map((sourceKey) => ({
        key: sourceKey,
        label: sourceKey,
        description: 'Custom/legacy adapter key.',
        domains: ['legacy'],
        signal_types: [],
        isLegacy: true,
      }))

    return [...known, ...legacy]
  }, [draftSources, sourceCatalog])

  const selectedSourceKeySet = useMemo(
    () => new Set(csvToList(draftSources).map((sourceKey) => normalizeSourceKey(sourceKey))),
    [draftSources]
  )

  const sourceCardsByGroup = useMemo(() => {
    const grouped: Record<TraderSourceGroupKey, Array<TraderSource & { isLegacy: boolean }>> = {
      markets: [],
      crypto: [],
      pool_watched: [],
      other: [],
    }
    for (const source of sourceCards) {
      grouped[classifyTraderSource(source)].push(source)
    }
    return grouped
  }, [sourceCards])

  const selectedSourceCount = useMemo(
    () => sourceCards.filter((source) => selectedSourceKeySet.has(normalizeSourceKey(source.key))).length,
    [selectedSourceKeySet, sourceCards]
  )

  const effectiveDraftSources = useMemo(() => {
    const explicit = uniqueSourceList(csvToList(draftSources))
    if (explicit.length > 0) return explicit
    return defaultSourceKeys
  }, [defaultSourceKeys, draftSources])

  const isCryptoStrategyDraft = useMemo(
    () => isCryptoStrategyKey(draftStrategyKey),
    [draftStrategyKey]
  )
  const selectedCryptoAssets = useMemo(
    () => new Set(normalizeCryptoAssetList(advancedConfig.cryptoAssetsCsv)),
    [advancedConfig.cryptoAssetsCsv]
  )
  const selectedCryptoTimeframes = useMemo(
    () => new Set(normalizeCryptoTimeframeList(advancedConfig.cryptoTimeframesCsv)),
    [advancedConfig.cryptoTimeframesCsv]
  )

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
  }

  const enableAllSourceCards = () => {
    setDraftSources(uniqueSourceList(sourceCards.map((source) => source.key)).join(', '))
  }

  const toggleCryptoAssetTarget = (asset: (typeof CRYPTO_ASSET_OPTIONS)[number]) => {
    const next = new Set(normalizeCryptoAssetList(advancedConfig.cryptoAssetsCsv))
    if (next.has(asset)) {
      next.delete(asset)
    } else {
      next.add(asset)
    }
    setAdvancedValue('cryptoAssetsCsv', CRYPTO_ASSET_OPTIONS.filter((item) => next.has(item)).join(', '))
  }

  const toggleCryptoTimeframeTarget = (timeframe: (typeof CRYPTO_TIMEFRAME_OPTIONS)[number]) => {
    const next = new Set(normalizeCryptoTimeframeList(advancedConfig.cryptoTimeframesCsv))
    if (next.has(timeframe)) {
      next.delete(timeframe)
    } else {
      next.add(timeframe)
    }
    setAdvancedValue('cryptoTimeframesCsv', CRYPTO_TIMEFRAME_OPTIONS.filter((item) => next.has(item)).join(', '))
  }

  const enableAllCryptoTargets = () => {
    setAdvancedValue('cryptoAssetsCsv', CRYPTO_ASSET_OPTIONS.join(', '))
    setAdvancedValue('cryptoTimeframesCsv', CRYPTO_TIMEFRAME_OPTIONS.join(', '))
  }

  useEffect(() => {
    if (!selectedTraderId && traders.length > 0) {
      setSelectedTraderId(traders[0].id)
    }
  }, [selectedTraderId, traders])

  useEffect(() => {
    if (!selectedTrader) return
    setDraftName(selectedTrader.name)
    setDraftDescription(selectedTrader.description || '')
    setDraftStrategyKey(selectedTrader.strategy_key)
    setDraftInterval(String(selectedTrader.interval_seconds || 60))
    setDraftSources(uniqueSourceList(selectedTrader.sources || []).join(', ') || defaultSourceCsv)
    setDraftEnabled(Boolean(selectedTrader.is_enabled))
    setDraftPaused(Boolean(selectedTrader.is_paused))
    const params = selectedTrader.params || {}
    const risk = selectedTrader.risk_limits || {}
    const metadata = selectedTrader.metadata || {}
    setDraftParams(JSON.stringify(params, null, 2))
    setDraftRisk(JSON.stringify(risk, null, 2))
    setDraftMetadata(JSON.stringify(metadata, null, 2))
    setAdvancedConfig(computeAdvancedConfig(params, risk, metadata))
    setSaveError(null)
  }, [defaultSourceCsv, selectedTrader])

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

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
    queryClient.invalidateQueries({ queryKey: ['traders-list'] })
    queryClient.invalidateQueries({ queryKey: ['trader-orders-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decisions-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decision-detail'] })
  }

  const hydrateDraftFromTemplate = (templateId: string) => {
    const template = templates.find((row) => row.id === templateId)
    if (!template) return
    const suggestedInterval = template.strategy_key.toLowerCase().includes('crypto') && toNumber(template.interval_seconds) >= 60
      ? 5
      : toNumber(template.interval_seconds || 60)
    const params = template.params || {}
    const risk = template.risk_limits || {}
    const metadata: Record<string, unknown> = { template_id: template.id }
    setDraftName(template.name)
    setDraftDescription(template.description || '')
    setDraftStrategyKey(template.strategy_key)
    setDraftInterval(String(suggestedInterval))
    setDraftSources(uniqueSourceList(template.sources || []).join(', ') || defaultSourceCsv)
    setDraftEnabled(true)
    setDraftPaused(false)
    setDraftParams(JSON.stringify(params, null, 2))
    setDraftRisk(JSON.stringify(risk, null, 2))
    setDraftMetadata(JSON.stringify(metadata, null, 2))
    setAdvancedConfig(computeAdvancedConfig(params, risk, metadata))
  }

  const openCreateTraderFlyout = () => {
    setTraderFlyoutMode('create')
    setTemplateSelection('none')
    setDraftName('')
    setDraftDescription('')
    setDraftStrategyKey('strategy.default')
    setDraftInterval('5')
    setDraftSources(defaultSourceCsv)
    setDraftEnabled(true)
    setDraftPaused(false)
    setDraftParams('{}')
    setDraftRisk('{}')
    setDraftMetadata('{}')
    setAdvancedConfig(defaultAdvancedConfig())
    setDeleteAction('disable')
    setDeleteConfirmName('')
    setSaveError(null)
    setTraderFlyoutOpen(true)
  }

  const openEditTraderFlyout = (trader: Trader) => {
    setSelectedTraderId(trader.id)
    setTraderFlyoutMode('edit')
    setTemplateSelection('none')
    setDraftName(trader.name)
    setDraftDescription(trader.description || '')
    setDraftStrategyKey(trader.strategy_key)
    setDraftInterval(String(trader.interval_seconds || 60))
    setDraftSources(uniqueSourceList(trader.sources || []).join(', ') || defaultSourceCsv)
    setDraftEnabled(Boolean(trader.is_enabled))
    setDraftPaused(Boolean(trader.is_paused))
    const params = trader.params || {}
    const risk = trader.risk_limits || {}
    const metadata = trader.metadata || {}
    setDraftParams(JSON.stringify(params, null, 2))
    setDraftRisk(JSON.stringify(risk, null, 2))
    setDraftMetadata(JSON.stringify(metadata, null, 2))
    setAdvancedConfig(computeAdvancedConfig(params, risk, metadata))
    setDeleteAction('disable')
    setDeleteConfirmName('')
    setSaveError(null)
    setTraderFlyoutOpen(true)
  }

  const setAdvancedValue = <K extends keyof TraderAdvancedConfig>(key: K, value: TraderAdvancedConfig[K]) => {
    setAdvancedConfig((current) => ({ ...current, [key]: value }))
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

  const traderStartMutation = useMutation({
    mutationFn: (traderId: string) => startTrader(traderId),
    onSuccess: refreshAll,
  })

  const traderPauseMutation = useMutation({
    mutationFn: (traderId: string) => pauseTrader(traderId),
    onSuccess: refreshAll,
  })

  const traderRunOnceMutation = useMutation({
    mutationFn: (traderId: string) => runTraderOnce(traderId),
    onSuccess: refreshAll,
  })

  const createTraderMutation = useMutation({
    mutationFn: async () => {
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

      return createTrader({
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        strategy_key: draftStrategyKey.trim(),
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        sources: effectiveDraftSources,
        params: withConfiguredParams(parsedParams.value, advancedConfig, draftStrategyKey),
        risk_limits: withConfiguredRiskLimits(parsedRisk.value, advancedConfig),
        metadata: withConfiguredMetadata(parsedMetadata.value, advancedConfig),
        is_enabled: draftEnabled,
        is_paused: draftPaused,
      })
    },
    onSuccess: (trader) => {
      setSaveError(null)
      setTraderFlyoutOpen(false)
      setSelectedTraderId(trader.id)
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to create trader'))
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

      return updateTrader(traderId, {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        strategy_key: draftStrategyKey.trim(),
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        sources: effectiveDraftSources,
        params: withConfiguredParams(parsedParams.value, advancedConfig, draftStrategyKey),
        risk_limits: withConfiguredRiskLimits(parsedRisk.value, advancedConfig),
        metadata: withConfiguredMetadata(parsedMetadata.value, advancedConfig),
        is_enabled: draftEnabled,
        is_paused: draftPaused,
      })
    },
    onSuccess: () => {
      setSaveError(null)
      setTraderFlyoutOpen(false)
      refreshAll()
    },
    onError: (error: unknown) => {
      setSaveError(errorMessage(error, 'Failed to save trader'))
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
      setSaveError(errorMessage(error, 'Failed to delete or disable trader'))
    },
  })

  const worker = overviewQuery.data?.worker
  const metrics = overviewQuery.data?.metrics
  const killSwitchOn = Boolean(overviewQuery.data?.control?.kill_switch)
  const globalMode = String(overviewQuery.data?.control?.mode || 'paper').toLowerCase()
  const orchestratorEnabled = Boolean(overviewQuery.data?.control?.is_enabled) && !Boolean(overviewQuery.data?.control?.is_paused)
  const orchestratorRunning = Boolean(worker?.running)
  const workerActivity = String(worker?.current_activity || '').toLowerCase()
  const orchestratorBlocked = orchestratorEnabled && !orchestratorRunning && workerActivity.startsWith('blocked')
  const orchestratorStatusLabel = orchestratorBlocked ? 'BLOCKED' : orchestratorRunning ? 'RUNNING' : 'STOPPED'

  const simulationAccounts = simulationAccountsQuery.data || []
  const selectedSandboxAccount = simulationAccounts.find((account) => account.id === selectedAccountId)
  const selectedAccountIsLive = Boolean(selectedAccountId?.startsWith('live:'))
  const selectedAccountValid = selectedAccountIsLive || Boolean(selectedSandboxAccount)
  const selectedAccountMode = selectedAccountIsLive ? 'live' : 'paper'
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
    const topTraderRows = Array.from(byTrader.values())
      .sort((a, b) => b.pnl - a.pnl)
      .slice(0, 8)

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
      topTraderRows,
      sourceRows,
    }
  }, [allOrders, traderNameById])

  const globalPositionBook = useMemo(
    () => buildPositionBookRows(allOrders, traderNameById),
    [allOrders, traderNameById]
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

        const haystack = `${order.market_question || ''} ${order.market_id || ''} ${order.source || ''} ${order.direction || ''}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 250)
  }, [selectedOrders, tradeSearch, tradeStatusFilter])

  const activityRows = useMemo(() => {
    const decisionRows: ActivityRow[] = allDecisions.map((decision) => ({
      kind: 'decision',
      id: decision.id,
      ts: decision.created_at,
      traderId: decision.trader_id,
      title: `${String(decision.decision).toUpperCase()} • ${decision.source}`,
      detail: decision.reason || decision.strategy_key,
      tone: String(decision.decision).toLowerCase() === 'selected' ? 'positive' : 'neutral',
    }))

    const orderRows: ActivityRow[] = allOrders.map((order) => {
      const status = normalizeStatus(order.status)
      const pnl = toNumber(order.actual_profit)
      const tone: ActivityRow['tone'] =
        FAILED_ORDER_STATUSES.has(status) ? 'negative' :
        RESOLVED_ORDER_STATUSES.has(status) && pnl > 0 ? 'positive' :
        RESOLVED_ORDER_STATUSES.has(status) && pnl < 0 ? 'negative' : 'neutral'

      return {
        kind: 'order',
        id: order.id,
        ts: order.created_at,
        traderId: order.trader_id,
        title: `${status.toUpperCase()} • ${order.market_question || order.market_id}`,
        detail: `${formatCurrency(toNumber(order.notional_usd))} • ${String(order.mode || '').toUpperCase()}`,
        tone,
      }
    })

    const eventRows: ActivityRow[] = allEvents.map((event) => ({
      kind: 'event',
      id: event.id,
      ts: event.created_at,
      traderId: event.trader_id,
      title: `${event.event_type} • ${event.severity}`,
      detail: event.message || 'No message provided',
      tone: String(event.severity || '').toLowerCase() === 'warn' ? 'warning' : 'neutral',
    }))

    return [...decisionRows, ...orderRows, ...eventRows]
      .sort((a, b) => toTs(b.ts) - toTs(a.ts))
      .slice(0, 350)
  }, [allDecisions, allOrders, allEvents])

  const filteredActivityRows = useMemo(() => {
    return activityRows.filter((row) => {
      const kindMatches = feedFilter === 'all' || row.kind === feedFilter
      const scopeMatches = scopeFilter === 'all' || row.traderId === selectedTraderId
      return kindMatches && scopeMatches
    })
  }, [activityRows, feedFilter, scopeFilter, selectedTraderId])

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

  const selectedDecisionTotal = useMemo(
    () => allDecisions.filter((decision) => String(decision.decision).toLowerCase() === 'selected').length,
    [allDecisions]
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
  const lastCycleDecisions = toNumber(worker?.stats?.decisions_last_cycle)
  const lastCycleOrders = toNumber(worker?.stats?.orders_last_cycle)
  const latestSelectedTraderActivityTs = selectedTraderActivityRows.length > 0
    ? toTs(selectedTraderActivityRows[0].ts)
    : 0
  const latestSelectedTraderRunTs = toTs(selectedTrader?.last_run_at || worker?.last_run_at)
  const selectedTraderNoNewRows = Boolean(
    selectedTrader &&
    orchestratorRunning &&
    latestSelectedTraderRunTs > (latestSelectedTraderActivityTs + 1000)
  )

  const runRate = toNumber(metrics?.decisions_count) > 0
    ? (toNumber(metrics?.orders_count) / toNumber(metrics?.decisions_count)) * 100
    : 0
  const tradersRunningDisplay = orchestratorRunning ? toNumber(metrics?.traders_running) : 0
  const displayAvgEdge = normalizeEdgePercent(globalSummary.avgEdge)
  const displayAvgConfidence = normalizeConfidencePercent(globalSummary.avgConfidence)
  const selectedTraderStatusLabel = !orchestratorRunning
    ? 'Engine Off'
    : !selectedTrader?.is_enabled
      ? 'Disabled'
      : selectedTrader?.is_paused
        ? 'Paused'
        : 'Running'
  const selectedTraderCanResume = Boolean(selectedTrader?.is_enabled && selectedTrader?.is_paused)
  const selectedTraderCanPause = Boolean(selectedTrader?.is_enabled && !selectedTrader?.is_paused)

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
  const startStopIsRunning = orchestratorEnabled
  const startStopDisabled = startStopIsRunning ? !canStopOrchestrator : !canStartOrchestrator
  const startStopPending = startStopIsRunning ? stopByModeMutation.isPending : startBySelectedAccountMutation.isPending

  const runStartStopCommand = () => {
    if (startStopIsRunning) {
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
    <div className="h-full min-h-0 flex flex-col gap-3">
      <Card className="shrink-0 border-cyan-500/30 bg-gradient-to-br from-cyan-500/[0.08] via-card to-emerald-500/[0.08]">
        <CardContent className="px-3 py-2 space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-300" />
            <span className="text-sm font-semibold leading-none">Autotrading Orchestration Hub</span>
            <Badge
              className="h-5 px-1.5 text-[10px]"
              variant={orchestratorBlocked ? 'destructive' : orchestratorRunning ? 'default' : 'secondary'}
            >
              {orchestratorStatusLabel}
            </Badge>
            <Badge className="h-5 px-1.5 text-[10px]" variant={globalMode === 'live' ? 'destructive' : 'outline'}>
              {globalMode.toUpperCase()}
            </Badge>
            <Badge className="h-5 px-1.5 text-[10px]" variant={killSwitchOn ? 'destructive' : 'outline'}>
              {killSwitchOn ? 'ORDERS BLOCKED' : 'ORDERS OPEN'}
            </Badge>
            <div className="ml-auto text-[11px] text-muted-foreground flex items-center gap-1">
              <Clock3 className="w-3 h-3" />
              {formatTimestamp(worker?.last_run_at)}
            </div>
          </div>

          <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5 flex flex-wrap items-center gap-1.5">
            <Button
              onClick={runStartStopCommand}
              disabled={startStopDisabled}
              className="h-7 min-w-[164px] text-[11px]"
              variant={startStopIsRunning ? 'secondary' : 'default'}
            >
              {startStopPending ? (
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              ) : startStopIsRunning ? (
                <Square className="w-3.5 h-3.5 mr-1.5" />
              ) : selectedAccountIsLive ? (
                <Zap className="w-3.5 h-3.5 mr-1.5" />
              ) : (
                <Play className="w-3.5 h-3.5 mr-1.5" />
              )}
              {startStopIsRunning ? 'Stop Engine' : `Start Engine (${selectedAccountMode.toUpperCase()})`}
            </Button>
            <div className="flex items-center gap-2 rounded-md border border-red-500/35 bg-red-500/10 px-2 py-1">
              <ShieldAlert className="w-3.5 h-3.5 text-red-700 dark:text-red-200" />
              <span className="text-[11px] text-red-700 dark:text-red-200">Block New Orders</span>
              <Switch
                checked={killSwitchOn}
                onCheckedChange={(enabled) => killSwitchMutation.mutate(enabled)}
                disabled={controlBusy}
              />
            </div>
            <span className="ml-auto text-[11px] text-muted-foreground hidden xl:block">
              {!selectedAccountValid
                ? 'Select a global account in the top control panel to start the engine.'
                : modeMismatch
                  ? 'Account mode and orchestrator mode are not aligned.'
                  : 'Start/Stop controls the global engine. Trader controls below only pause/resume individual traders.'}
            </span>
          </div>

          <div className="grid gap-1.5 grid-cols-2 sm:grid-cols-4 xl:grid-cols-8">
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Worker</p>
              <p className="text-[11px] truncate" title={worker?.current_activity || 'Idle'}>
                {worker?.current_activity || 'Idle'}
              </p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Health</p>
              <p className="text-[11px] flex items-center gap-1">
                {worker?.last_error ? <AlertTriangle className="w-3 h-3 text-amber-400" /> : <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
                {worker?.last_error ? 'Degraded' : 'Healthy'}
              </p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Traders</p>
              <p className="text-[11px]">{tradersRunningDisplay} / {toNumber(metrics?.traders_total)} active</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Capture</p>
              <p className="text-[11px]">{formatPercent(runRate)}</p>
            </div>
            <div className={cn(
              'rounded-md border px-2 py-1',
              toNumber(metrics?.daily_pnl) >= 0 ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'
            )}>
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Daily PnL</p>
              <p className="text-[11px] font-mono">{formatCurrency(toNumber(metrics?.daily_pnl))}</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Exposure</p>
              <p className="text-[11px] font-mono">{formatCurrency(toNumber(metrics?.gross_exposure_usd), true)}</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Open / Orders</p>
              <p className="text-[11px] font-mono">{globalSummary.open} / {allOrders.length}</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Win / Edge / Conf</p>
              <p className="text-[11px] font-mono">
                {formatPercent(globalSummary.winRate)} / {formatPercent(displayAvgEdge)} / {formatPercent(displayAvgConfidence)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="command" className="flex-1 min-h-0 flex flex-col gap-2">
        <TabsList className="w-full justify-start overflow-auto shrink-0 h-9">
          <TabsTrigger value="command">Command Center</TabsTrigger>
          <TabsTrigger value="traders">Auto Traders</TabsTrigger>
          <TabsTrigger value="governance">Governance</TabsTrigger>
        </TabsList>

        <TabsContent value="command" className="mt-0 flex-1 min-h-0 overflow-hidden">
          <div className="h-full min-h-0 grid gap-3 grid-rows-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="grid gap-3 xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)] min-h-0">
              <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                    <span>Live Global Terminal</span>
                    <Badge variant="outline">{filteredActivityRows.length} rows</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 overflow-hidden">
                  <div className="flex flex-wrap items-center gap-1 mb-2">
                    {(['all', 'decision', 'order', 'event'] as FeedFilter[]).map((kind) => (
                      <Button
                        key={kind}
                        size="sm"
                        variant={feedFilter === kind ? 'default' : 'outline'}
                        onClick={() => setFeedFilter(kind)}
                        className="h-6 px-2 text-[11px]"
                      >
                        {kind}
                      </Button>
                    ))}
                    <div className="ml-auto flex items-center gap-1">
                      <Button
                        size="sm"
                        variant={scopeFilter === 'all' ? 'default' : 'outline'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setScopeFilter('all')}
                      >
                        all
                      </Button>
                      <Button
                        size="sm"
                        variant={scopeFilter === 'selected' ? 'default' : 'outline'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setScopeFilter('selected')}
                        disabled={!selectedTraderId}
                      >
                        selected
                      </Button>
                    </div>
                  </div>

                  {filteredActivityRows.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No events matching filters.</div>
                  ) : (
                    <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 bg-muted/20">
                      <div className="space-y-1 p-2 font-mono text-[11px] leading-relaxed">
                        {filteredActivityRows.map((row) => (
                          <div
                            key={`${row.kind}:${row.id}`}
                            className={cn(
                              'rounded border px-2 py-1',
                              row.tone === 'positive' && 'border-emerald-500/30 text-emerald-700 dark:text-emerald-100',
                              row.tone === 'negative' && 'border-red-500/35 text-red-700 dark:text-red-100',
                              row.tone === 'warning' && 'border-amber-500/35 text-amber-700 dark:text-amber-100',
                              row.tone === 'neutral' && 'border-border text-foreground'
                            )}
                          >
                            <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>{' '}
                            <span className="uppercase">{row.kind}</span>{' '}
                            <span className="text-muted-foreground">{traderNameById[String(row.traderId || '')] || shortId(row.traderId || '')}</span>{' '}
                            <span>{row.title}</span>
                            <span className="text-muted-foreground"> :: {row.detail}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>

              <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm">Execution Radar</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 overflow-hidden space-y-2">
                  <div className="grid gap-2 grid-cols-2">
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Decisions</p>
                      <p className="text-sm font-mono">{allDecisions.length}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Orders</p>
                      <p className="text-sm font-mono">{allOrders.length}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Conversion</p>
                      <p className="text-sm font-mono">{formatPercent(runRate)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Worker State</p>
                      <p className="text-sm">{worker?.last_error ? 'Degraded' : 'Healthy'}</p>
                    </div>
                  </div>

                  <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 p-2">
                    <div className="space-y-3 pr-2">
                      <div>
                        <p className="text-[11px] uppercase tracking-widest text-muted-foreground mb-1">Source Leaders</p>
                        {globalSummary.sourceRows.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No source-level data.</p>
                        ) : (
                          <div className="space-y-1.5">
                            {globalSummary.sourceRows.slice(0, 6).map((row) => (
                              <div key={row.source} className="rounded-md border border-border/70 px-2 py-1.5">
                                <div className="flex items-center justify-between text-xs">
                                  <span>{row.source}</span>
                                  <span className="font-mono">{row.orders}</span>
                                </div>
                                <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                                  <span>{formatCurrency(row.notional, true)}</span>
                                  <span className={cn(row.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>{formatCurrency(row.pnl)}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      <div>
                        <p className="text-[11px] uppercase tracking-widest text-muted-foreground mb-1">Trader Leaders</p>
                        {globalSummary.topTraderRows.slice(0, 6).map((row) => (
                          <div key={row.traderId} className="rounded-md border border-border/70 px-2 py-1.5 mb-1.5">
                            <div className="flex items-center justify-between text-xs">
                              <span className="truncate">{row.traderName}</span>
                              <span className={cn('font-mono', row.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                {formatCurrency(row.pnl)}
                              </span>
                            </div>
                            <div className="mt-1 text-[11px] text-muted-foreground">
                              {row.orders} orders • {row.resolved} resolved
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-3 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] min-h-0">
              <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm">Open Position Book</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 overflow-hidden">
                  {globalPositionBook.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No live positions are currently held by orchestrator traders.</div>
                  ) : (
                    <ScrollArea className="h-full min-h-0">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Trader</TableHead>
                            <TableHead>Market</TableHead>
                            <TableHead>Dir</TableHead>
                            <TableHead className="text-right">Exposure</TableHead>
                            <TableHead className="text-right">Avg Px</TableHead>
                            <TableHead className="text-right">Edge</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {globalPositionBook.slice(0, 150).map((row) => (
                            <TableRow key={row.key}>
                              <TableCell className="font-medium">{row.traderName}</TableCell>
                              <TableCell>
                                <div className="max-w-[320px] truncate" title={row.marketQuestion}>{row.marketQuestion}</div>
                                <div className="text-[11px] text-muted-foreground">{shortId(row.marketId)}</div>
                              </TableCell>
                              <TableCell>
                                <Badge variant={row.direction === 'BUY' ? 'default' : row.direction === 'SELL' ? 'secondary' : 'outline'}>{row.direction}</Badge>
                              </TableCell>
                              <TableCell className="text-right font-mono">{formatCurrency(row.exposureUsd)}</TableCell>
                              <TableCell className="text-right font-mono">{row.averagePrice !== null ? row.averagePrice.toFixed(3) : 'n/a'}</TableCell>
                              <TableCell className="text-right font-mono">{row.weightedEdge !== null ? formatPercent(normalizeEdgePercent(row.weightedEdge)) : 'n/a'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>

              <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm">Decision Funnel + Selection Queue</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 overflow-hidden space-y-2">
                  <div className="grid gap-2 grid-cols-2">
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Observed</p>
                      <p className="text-sm font-mono">{allDecisions.length}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Selected</p>
                      <p className="text-sm font-mono">{selectedDecisionTotal}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Orders Emitted</p>
                      <p className="text-sm font-mono">{toNumber(metrics?.orders_count)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Failed Orders</p>
                      <p className="text-sm font-mono">{globalSummary.failed}</p>
                    </div>
                  </div>

                  <div className={cn(
                    'rounded-md border px-2 py-1.5 text-xs',
                    worker?.last_error ? 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-100' : 'border-border/70 bg-background/70 text-muted-foreground'
                  )}>
                    {worker?.last_error ? `Worker alert: ${worker.last_error}` : 'No worker alerts. Decision and order pipeline healthy.'}
                  </div>

                  <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 bg-muted/20">
                    <div className="space-y-1 p-2">
                      {recentSelectedDecisions.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No selected decisions yet.</p>
                      ) : (
                        recentSelectedDecisions.map((decision) => (
                          <div key={decision.id} className="rounded-md border border-border/70 px-2 py-1.5">
                            <div className="flex items-center justify-between text-[11px]">
                              <span className="font-mono text-muted-foreground">{formatTimestamp(decision.created_at)}</span>
                              <span className="font-mono">{decision.score !== null ? decision.score.toFixed(2) : 'n/a'}</span>
                            </div>
                            <p className="text-xs mt-1 truncate" title={`${decision.source} • ${decision.strategy_key}`}>
                              {decision.source} • {decision.strategy_key}
                            </p>
                            <p className="text-[11px] text-muted-foreground truncate" title={decision.reason || ''}>
                              {decision.reason || 'No decision rationale recorded.'}
                            </p>
                          </div>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="traders" className="mt-0 flex-1 min-h-0 overflow-hidden">
          <div className="h-full min-h-0 grid gap-3 xl:grid-cols-[320px_minmax(0,1fr)]">
            <Card className="h-full flex flex-col min-h-0 overflow-hidden">
              <CardHeader className="py-2">
                <CardTitle className="text-sm flex items-center justify-between gap-2">
                  <span>Trader Roster</span>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{traders.length}</Badge>
                    <Button size="sm" className="h-7 px-2 text-[11px]" onClick={openCreateTraderFlyout}>
                      Add Trader
                    </Button>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 overflow-hidden">
                <ScrollArea className="h-full min-h-0">
                  <div className="space-y-2 pr-2">
                    {traders.map((trader) => {
                      const isSelected = selectedTraderId === trader.id
                      const traderSummary = globalSummary.topTraderRows.find((row) => row.traderId === trader.id)
                      const traderPnl = traderSummary?.pnl || 0
                      const traderStatus = !orchestratorRunning
                        ? 'Engine Off'
                        : !trader.is_enabled
                          ? 'Disabled'
                          : trader.is_paused
                            ? 'Paused'
                            : 'Running'

                      return (
                        <div
                          key={trader.id}
                          className={cn(
                            'rounded-md border p-2 transition-colors',
                            isSelected ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-border hover:bg-muted/40'
                          )}
                        >
                          <button className="w-full text-left" onClick={() => setSelectedTraderId(trader.id)}>
                            <div className="flex items-center justify-between gap-2">
                              <p className="font-medium text-sm truncate" title={trader.name}>{trader.name}</p>
                              <Badge variant={traderStatus === 'Running' ? 'default' : 'secondary'}>
                                {traderStatus}
                              </Badge>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1 truncate" title={trader.strategy_key}>
                              {trader.strategy_key} • {trader.interval_seconds}s
                            </p>
                            <div className="mt-1 flex items-center justify-between text-[11px]">
                              <span className="text-muted-foreground">{(trader.sources || []).join(', ') || 'No sources'}</span>
                              <span className={cn('font-mono', traderPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                {formatCurrency(traderPnl)}
                              </span>
                            </div>
                          </button>

                          <div className="mt-2 flex flex-wrap gap-1">
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-6 px-2 text-[11px]"
                              onClick={() => openEditTraderFlyout(trader)}
                            >
                              Edit
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-6 px-2 text-[11px]"
                              onClick={() => {
                                setSelectedTraderId(trader.id)
                                openEditTraderFlyout(trader)
                              }}
                            >
                              Delete
                            </Button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <div className="h-full min-h-0 flex flex-col gap-3 min-w-0">
              <Card className="shrink-0">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                    <span>{selectedTrader?.name || 'Trader Console'}</span>
                    <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                      <Badge variant={selectedTraderStatusLabel === 'Running' ? 'default' : 'secondary'}>
                        {selectedTraderStatusLabel}
                      </Badge>
                      <span>{formatTimestamp(selectedTrader?.last_run_at)}</span>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex flex-wrap gap-1.5">
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-[11px]"
                      onClick={() => selectedTrader && traderStartMutation.mutate(selectedTrader.id)}
                      disabled={!selectedTraderCanResume || traderStartMutation.isPending}
                    >
                      Resume Trader
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-[11px]"
                      onClick={() => selectedTrader && traderPauseMutation.mutate(selectedTrader.id)}
                      disabled={!selectedTraderCanPause || traderPauseMutation.isPending}
                    >
                      Pause Trader
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-[11px]"
                      onClick={() => selectedTrader && traderRunOnceMutation.mutate(selectedTrader.id)}
                      disabled={!selectedTrader || !selectedTrader.is_enabled || traderRunOnceMutation.isPending}
                    >
                      Run Once
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-[11px]"
                      onClick={() => selectedTrader && openEditTraderFlyout(selectedTrader)}
                      disabled={!selectedTrader}
                    >
                      Configure
                    </Button>
                  </div>
                  <p className="text-[11px] text-muted-foreground">
                    Engine start/stop is global. Trader controls here only change this trader&apos;s runtime state.
                  </p>

                  <div className="grid gap-1.5 grid-cols-2 md:grid-cols-4 xl:grid-cols-8">
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Open</p>
                      <p className="text-[11px] font-mono">{selectedTraderSummary.open}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Exposure</p>
                      <p className="text-[11px] font-mono">{formatCurrency(selectedTraderExposure, true)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Decisions</p>
                      <p className="text-[11px] font-mono">{selectedTraderSummary.decisions}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Selected</p>
                      <p className="text-[11px] font-mono">{selectedTraderSummary.selectedDecisions}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Conversion</p>
                      <p className="text-[11px] font-mono">{formatPercent(selectedTraderSummary.conversion)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Win Rate</p>
                      <p className="text-[11px] font-mono">{formatPercent(selectedTraderSummary.winRate)}</p>
                    </div>
                    <div className={cn(
                      'rounded-md border px-2 py-1',
                      selectedTraderSummary.pnl >= 0 ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'
                    )}>
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Realized PnL</p>
                      <p className="text-[11px] font-mono">{formatCurrency(selectedTraderSummary.pnl)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Edge / Conf</p>
                      <p className="text-[11px] font-mono">
                        {formatPercent(normalizeEdgePercent(selectedTraderSummary.avgEdge))} / {formatPercent(normalizeConfidencePercent(selectedTraderSummary.avgConfidence))}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Tabs defaultValue="terminal" className="flex-1 min-h-0 flex flex-col gap-2">
                <TabsList className="w-full justify-start overflow-auto shrink-0 h-8">
                  <TabsTrigger value="terminal">Live Terminal</TabsTrigger>
                  <TabsTrigger value="positions">Positions</TabsTrigger>
                  <TabsTrigger value="decisions">Decisions</TabsTrigger>
                  <TabsTrigger value="trades">Trades</TabsTrigger>
                </TabsList>

                <TabsContent value="terminal" className="mt-0 flex-1 min-h-0">
                  <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                    <CardHeader className="py-2">
                      <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                        <span>Trader Activity Terminal</span>
                        <div className="flex items-center gap-1">
                          {(['all', 'decision', 'order', 'event'] as FeedFilter[]).map((kind) => (
                            <Button
                              key={kind}
                              size="sm"
                              variant={traderFeedFilter === kind ? 'default' : 'outline'}
                              className="h-6 px-2 text-[11px]"
                              onClick={() => setTraderFeedFilter(kind)}
                            >
                              {kind}
                            </Button>
                          ))}
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0 overflow-hidden space-y-2">
                      {selectedTrader ? (
                        <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5 text-[11px]">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="font-medium">Engine heartbeat</span>
                            <span className="font-mono text-muted-foreground">{formatTimestamp(selectedTrader?.last_run_at || worker?.last_run_at)}</span>
                          </div>
                          <p className="mt-1 text-muted-foreground">
                            Last cycle: decisions={lastCycleDecisions} orders={lastCycleOrders}
                            {selectedTraderNoNewRows ? ' • no new qualifying signals for this trader yet.' : ''}
                          </p>
                        </div>
                      ) : null}
                      {!selectedTrader ? (
                        <div className="h-full flex items-center justify-center text-sm text-muted-foreground">Select a trader from the roster.</div>
                      ) : filteredTraderActivityRows.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center gap-1 text-sm text-muted-foreground">
                          <p>No terminal rows for this trader yet.</p>
                          {orchestratorRunning ? (
                            <p className="text-[11px]">Worker is running; rows appear when new decisions, orders, or events are created.</p>
                          ) : null}
                        </div>
                      ) : (
                        <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 bg-muted/20">
                          <div className="space-y-1 p-2 font-mono text-[11px] leading-relaxed">
                            {filteredTraderActivityRows.map((row) => (
                              <div
                                key={`${row.kind}:${row.id}`}
                                className={cn(
                                  'rounded border px-2 py-1',
                                  row.tone === 'positive' && 'border-emerald-500/30 text-emerald-700 dark:text-emerald-100',
                                  row.tone === 'negative' && 'border-red-500/35 text-red-700 dark:text-red-100',
                                  row.tone === 'warning' && 'border-amber-500/35 text-amber-700 dark:text-amber-100',
                                  row.tone === 'neutral' && 'border-border text-foreground'
                                )}
                              >
                                <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>{' '}
                                <span className="uppercase">{row.kind}</span>{' '}
                                <span>{row.title}</span>
                                <span className="text-muted-foreground"> :: {row.detail}</span>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="positions" className="mt-0 flex-1 min-h-0">
                  <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                    <CardHeader className="py-2">
                      <CardTitle className="text-sm">Positions Held by Trader</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0 overflow-hidden">
                      {selectedPositionBook.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No open positions held by this trader.</div>
                      ) : (
                        <ScrollArea className="h-full min-h-0 rounded-md border border-border/80">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Market</TableHead>
                                <TableHead>Dir</TableHead>
                                <TableHead className="text-right">Exposure</TableHead>
                                <TableHead className="text-right">Avg Px</TableHead>
                                <TableHead className="text-right">Edge</TableHead>
                                <TableHead className="text-right">Conf</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedPositionBook.map((row) => (
                                <TableRow key={row.key}>
                                  <TableCell>
                                    <div className="max-w-[380px] truncate" title={row.marketQuestion}>{row.marketQuestion}</div>
                                    <div className="text-[11px] text-muted-foreground">{shortId(row.marketId)}</div>
                                  </TableCell>
                                  <TableCell>
                                    <Badge variant={row.direction === 'BUY' ? 'default' : row.direction === 'SELL' ? 'secondary' : 'outline'}>{row.direction}</Badge>
                                  </TableCell>
                                  <TableCell className="text-right font-mono">{formatCurrency(row.exposureUsd)}</TableCell>
                                  <TableCell className="text-right font-mono">{row.averagePrice !== null ? row.averagePrice.toFixed(3) : 'n/a'}</TableCell>
                                  <TableCell className="text-right font-mono">{row.weightedEdge !== null ? formatPercent(normalizeEdgePercent(row.weightedEdge)) : 'n/a'}</TableCell>
                                  <TableCell className="text-right font-mono">{row.weightedConfidence !== null ? formatPercent(normalizeConfidencePercent(row.weightedConfidence)) : 'n/a'}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="decisions" className="mt-0 flex-1 min-h-0 overflow-hidden">
                  <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                    <CardHeader className="py-2">
                      <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                        <span>Decision Inspector</span>
                        <Badge variant="outline">{filteredDecisions.length} rows</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0 overflow-hidden grid gap-3 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
                      <div className="min-h-0 flex flex-col gap-2 overflow-hidden">
                        <Input
                          value={decisionSearch}
                          onChange={(event) => setDecisionSearch(event.target.value)}
                          placeholder="Filter by source, strategy, reason..."
                        />
                        <div className="grid gap-2 grid-cols-3">
                          <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Selected</p>
                            <p className="text-[11px] font-mono">{decisionOutcomeSummary.selected}</p>
                          </div>
                          <div className="rounded-md border border-amber-500/30 bg-amber-500/5 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Blocked</p>
                            <p className="text-[11px] font-mono">{decisionOutcomeSummary.blocked}</p>
                          </div>
                          <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Other</p>
                            <p className="text-[11px] font-mono">{decisionOutcomeSummary.skipped}</p>
                          </div>
                        </div>
                        <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/80 p-2">
                          <div className="space-y-2 pr-2">
                            {filteredDecisions.map((decision) => (
                              <button
                                key={decision.id}
                                onClick={() => setSelectedDecisionId(decision.id)}
                                className={cn(
                                  'w-full text-left rounded-md border p-2 transition-colors',
                                  selectedDecisionId === decision.id ? 'border-cyan-500/40 bg-cyan-500/10' : 'border-border hover:bg-muted/30'
                                )}
                              >
                                <div className="flex items-center justify-between gap-2 text-[11px]">
                                  <span className="font-mono text-muted-foreground">{formatTimestamp(decision.created_at)}</span>
                                  <Badge variant={String(decision.decision).toLowerCase() === 'selected' ? 'default' : String(decision.decision).toLowerCase() === 'blocked' ? 'destructive' : 'outline'}>
                                    {String(decision.decision).toUpperCase()}
                                  </Badge>
                                </div>
                                <div className="mt-1 flex items-center justify-between gap-2">
                                  <p className="text-xs font-medium truncate">{decision.source} • {decision.strategy_key}</p>
                                  <span className="text-[11px] font-mono text-muted-foreground">
                                    {decision.score !== null ? decision.score.toFixed(2) : 'n/a'}
                                  </span>
                                </div>
                                <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">{decision.reason || 'No reason captured.'}</p>
                              </button>
                            ))}
                            {filteredDecisions.length === 0 ? <p className="text-sm text-muted-foreground">No decisions matching filter.</p> : null}
                          </div>
                        </ScrollArea>
                      </div>

                      <div className="min-h-0 flex flex-col gap-2 overflow-hidden">
                        {!selectedDecision ? (
                          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                            Select a decision from the left panel to inspect signal, risk, checks, and order linkage.
                          </div>
                        ) : (
                          <>
                            <div className="rounded-md border border-border p-2.5 space-y-2">
                              <div className="flex items-center justify-between gap-2">
                                <p className="font-medium text-sm truncate">{selectedDecision.strategy_key}</p>
                                <Badge variant={String(selectedDecision.decision).toLowerCase() === 'selected' ? 'default' : String(selectedDecision.decision).toLowerCase() === 'blocked' ? 'destructive' : 'outline'}>
                                  {String(selectedDecision.decision).toUpperCase()}
                                </Badge>
                              </div>
                              <div className="grid gap-1 text-[11px] text-muted-foreground sm:grid-cols-2">
                                <span>Score: {selectedDecision.score !== null ? selectedDecision.score.toFixed(2) : 'n/a'}</span>
                                <span>Timestamp: {formatTimestamp(selectedDecision.created_at)}</span>
                                <span className="truncate">Source: {selectedDecision.source}</span>
                                <span className="truncate">Signal: {shortId(selectedDecision.signal_id)}</span>
                              </div>
                              <p className="text-[11px] text-muted-foreground">{selectedDecision.reason || 'No reason text.'}</p>
                            </div>

                            <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/80 p-2">
                              <div className="space-y-2 pr-2">
                                <div className={cn(
                                  'rounded-md border p-2',
                                  riskAllowed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-amber-500/30 bg-amber-500/10'
                                )}>
                                  <div className="flex items-center justify-between gap-2">
                                    <p className="text-xs font-semibold">Risk Gate</p>
                                    <Badge variant={riskAllowed ? 'default' : 'destructive'}>
                                      {riskAllowed ? 'ALLOW' : 'BLOCK'}
                                    </Badge>
                                  </div>
                                  {riskChecks.length === 0 ? (
                                    <p className="mt-1 text-[11px] text-muted-foreground">No risk checks captured.</p>
                                  ) : (
                                    <div className="mt-2 space-y-1.5">
                                      {riskChecks.map((check, index) => {
                                        const passed = toBoolean(check?.passed, false)
                                        const label = String(check?.check_label || check?.check_key || `risk_check_${index + 1}`)
                                        const detail = String(check?.detail || '')
                                        const score = check?.score
                                        return (
                                          <div key={`${label}:${index}`} className="rounded border border-border/60 px-2 py-1.5 text-[11px]">
                                            <div className="flex items-center justify-between gap-2">
                                              <span className="truncate">{label}</span>
                                              <Badge variant={passed ? 'default' : 'destructive'}>{passed ? 'PASS' : 'FAIL'}</Badge>
                                            </div>
                                            <p className="mt-1 text-muted-foreground">{detail || 'No detail.'}</p>
                                            {score !== null && score !== undefined ? (
                                              <p className="mt-1 text-muted-foreground font-mono">score={toNumber(score).toFixed(2)}</p>
                                            ) : null}
                                          </div>
                                        )
                                      })}
                                    </div>
                                  )}
                                </div>

                                <div className="rounded-md border border-border p-2">
                                  <div className="flex items-center justify-between gap-2">
                                    <p className="text-xs font-semibold">Strategy Checks</p>
                                    <span className="text-[11px] text-muted-foreground font-mono">
                                      pass={decisionPassCount} fail={decisionFailCount}
                                    </span>
                                  </div>
                                  {decisionDetailQuery.isLoading ? (
                                    <div className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
                                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                      Loading check records...
                                    </div>
                                  ) : decisionChecks.length === 0 ? (
                                    <p className="mt-2 text-[11px] text-muted-foreground">No check records for this decision.</p>
                                  ) : (
                                    <div className="mt-2 space-y-1.5">
                                      {decisionChecks.map((check) => (
                                        <div key={check.id} className={cn(
                                          'rounded-md border p-2',
                                          check.passed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'
                                        )}>
                                          <div className="flex items-center justify-between gap-2">
                                            <p className="text-xs font-medium truncate">{check.check_label}</p>
                                            <Badge variant={check.passed ? 'default' : 'destructive'}>{check.passed ? 'PASS' : 'FAIL'}</Badge>
                                          </div>
                                          <p className="mt-1 text-[11px] text-muted-foreground">{check.detail || 'No detail provided.'}</p>
                                          {check.score !== null ? (
                                            <p className="mt-1 text-[11px] text-muted-foreground font-mono">score={check.score.toFixed(2)}</p>
                                          ) : null}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>

                                <div className="rounded-md border border-border p-2">
                                  <div className="flex items-center justify-between gap-2">
                                    <p className="text-xs font-semibold">Linked Orders</p>
                                    <span className="text-[11px] text-muted-foreground font-mono">{decisionOrders.length}</span>
                                  </div>
                                  {decisionOrders.length === 0 ? (
                                    <p className="mt-2 text-[11px] text-muted-foreground">No order linked to this decision.</p>
                                  ) : (
                                    <div className="mt-2 space-y-1.5">
                                      {decisionOrders.map((order) => (
                                        <div key={order.id} className="rounded border border-border/60 px-2 py-1.5 text-[11px]">
                                          <div className="flex items-center justify-between gap-2">
                                            <span className="truncate">{order.market_question || shortId(order.market_id)}</span>
                                            <Badge variant={FAILED_ORDER_STATUSES.has(normalizeStatus(order.status)) ? 'destructive' : OPEN_ORDER_STATUSES.has(normalizeStatus(order.status)) ? 'outline' : 'default'}>
                                              {normalizeStatus(order.status)}
                                            </Badge>
                                          </div>
                                          <p className="mt-1 text-muted-foreground font-mono">
                                            {formatCurrency(toNumber(order.notional_usd))} • {String(order.mode || 'n/a').toUpperCase()} • {String(order.direction || 'n/a').toUpperCase()}
                                          </p>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </ScrollArea>
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="trades" className="mt-0 flex-1 min-h-0">
                  <Card className="h-full flex flex-col min-h-0 overflow-hidden">
                    <CardHeader className="py-2">
                      <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                        <span>Trade History + Outcomes</span>
                        <div className="flex items-center gap-1 flex-wrap">
                          {(['all', 'open', 'resolved', 'failed'] as TradeStatusFilter[]).map((filter) => (
                            <Button
                              key={filter}
                              size="sm"
                              variant={tradeStatusFilter === filter ? 'default' : 'outline'}
                              className="h-6 px-2 text-[11px]"
                              onClick={() => setTradeStatusFilter(filter)}
                            >
                              {filter}
                            </Button>
                          ))}
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0 overflow-hidden space-y-2">
                      <Input
                        value={tradeSearch}
                        onChange={(event) => setTradeSearch(event.target.value)}
                        placeholder="Search market, source, direction..."
                      />

                      <ScrollArea className="h-full min-h-0 rounded-md border border-border/80">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Market</TableHead>
                              <TableHead>Status</TableHead>
                              <TableHead>Dir</TableHead>
                              <TableHead className="text-right">Notional</TableHead>
                              <TableHead className="text-right">Edge</TableHead>
                              <TableHead className="text-right">P/L</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {filteredTradeHistory.map((order) => {
                              const status = normalizeStatus(order.status)
                              const pnl = toNumber(order.actual_profit)
                              return (
                                <TableRow key={order.id}>
                                  <TableCell>
                                    <div className="max-w-[380px] truncate" title={order.market_question || order.market_id}>{order.market_question || order.market_id}</div>
                                    <div className="text-[11px] text-muted-foreground">{formatShortDate(order.executed_at || order.created_at)}</div>
                                  </TableCell>
                                  <TableCell>
                                    <Badge
                                      variant={
                                        FAILED_ORDER_STATUSES.has(status) ? 'destructive' :
                                          RESOLVED_ORDER_STATUSES.has(status) ? 'default' :
                                            'outline'
                                      }
                                    >
                                      {status}
                                    </Badge>
                                  </TableCell>
                                  <TableCell>{String(order.direction || 'n/a').toUpperCase()}</TableCell>
                                  <TableCell className="text-right font-mono">{formatCurrency(toNumber(order.notional_usd))}</TableCell>
                                  <TableCell className="text-right font-mono">{order.edge_percent !== null ? formatPercent(normalizeEdgePercent(toNumber(order.edge_percent))) : 'n/a'}</TableCell>
                                  <TableCell className={cn('text-right font-mono', pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                    {order.actual_profit !== null ? formatCurrency(pnl) : 'n/a'}
                                  </TableCell>
                                </TableRow>
                              )
                            })}
                            {filteredTradeHistory.length === 0 ? (
                              <TableRow>
                                <TableCell colSpan={6} className="text-center text-muted-foreground">No trades match the selected filters.</TableCell>
                              </TableRow>
                            ) : null}
                          </TableBody>
                        </Table>
                      </ScrollArea>
                    </CardContent>
                  </Card>
                </TabsContent>

              </Tabs>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="governance" className="mt-0 flex-1 min-h-0 overflow-hidden">
          <div className="h-full min-h-0 grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
            <Card className="h-full flex flex-col min-h-0 overflow-hidden">
              <CardHeader className="py-2">
                <CardTitle className="text-sm">Governance Controls + Guardrails</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 overflow-hidden">
                <ScrollArea className="h-full min-h-0 pr-2">
                  <div className="space-y-3 pb-2">
                    <div className="rounded-md border border-border p-3">
                      <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Control state</p>
                      <div className="mt-2 space-y-2 text-sm">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Mode</span>
                          <Badge variant={globalMode === 'live' ? 'destructive' : 'outline'}>{globalMode.toUpperCase()}</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Engine</span>
                          <span className={cn('font-medium', orchestratorBlocked ? 'text-amber-600 dark:text-amber-200' : '')}>
                            {orchestratorStatusLabel}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Block new orders</span>
                          <span className={cn('font-medium', killSwitchOn ? 'text-red-400' : 'text-emerald-400')}>{killSwitchOn ? 'Active' : 'Inactive'}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Run interval</span>
                          <span className="font-medium font-mono">{toNumber(overviewQuery.data?.control?.run_interval_seconds)}s</span>
                        </div>
                      </div>
                    </div>

                    <div className={cn(
                      'rounded-md border p-3',
                      !selectedAccountValid || modeMismatch ? 'border-amber-500/40 bg-amber-500/10' : 'border-border'
                    )}>
                      <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Account + Mode Governance</p>
                      <div className="mt-2 text-sm space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Global account selected</span>
                          <span className="font-medium">{selectedAccountValid ? 'Yes' : 'No'}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Mode synchronized</span>
                          <span className="font-medium">{modeMismatch ? 'No' : 'Yes'}</span>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Start commands are blocked until a valid global account is selected and mode alignment is satisfied.
                        </div>
                      </div>
                    </div>

                    <div className="rounded-md border border-border p-3">
                      <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Live guardrails</p>
                      <div className="mt-2 space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Live requires preflight pass + arm token</span>
                          <Badge variant="outline">Enforced</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Block New Orders guard blocks order creation</span>
                          <Badge variant="outline">Enforced</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Worker pause overrides start</span>
                          <Badge variant="outline">Enforced</Badge>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-md border border-border p-3">
                      <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Emergency controls</p>
                      <div className="mt-2 rounded-md border border-border/70 bg-muted/20 px-2 py-1.5 text-xs text-muted-foreground">
                        Header controls are intentionally split by function:
                        <span className="font-medium text-foreground"> Start/Stop </span>
                        changes engine state,
                        <span className="font-medium text-foreground"> Block New Orders </span>
                        keeps engine running but blocks new selected orders.
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <Card className="h-full flex flex-col min-h-0 overflow-hidden">
              <CardHeader className="py-2">
                <CardTitle className="text-sm">Risk + Failure Terminal</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 overflow-hidden space-y-2">
                <div className="grid gap-2 grid-cols-2">
                  <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Risk rows</p>
                    <p className="text-sm font-mono">{riskActivityRows.length}</p>
                  </div>
                  <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Failed orders</p>
                    <p className="text-sm font-mono">{failedOrders.length}</p>
                  </div>
                </div>

                <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 bg-muted/20">
                  <div className="space-y-1 p-2 font-mono text-[11px] leading-relaxed">
                    {riskActivityRows.length === 0 ? (
                      <div className="rounded border border-emerald-500/30 px-2 py-1 text-emerald-700 dark:text-emerald-100">
                        [HEALTHY] no warning or failure events captured.
                      </div>
                    ) : (
                      riskActivityRows.map((row) => (
                        <div
                          key={`${row.kind}:${row.id}`}
                          className={cn(
                            'rounded border px-2 py-1',
                            row.tone === 'negative' ? 'border-red-500/35 text-red-700 dark:text-red-100' : 'border-amber-500/35 text-amber-700 dark:text-amber-100'
                          )}
                        >
                          <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>{' '}
                          <span className="uppercase">{row.kind}</span>{' '}
                          <span>{row.title}</span>
                          <span className="text-muted-foreground"> :: {row.detail}</span>
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

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
          }
        }}
      >
        <SheetContent side="right" className="w-full sm:max-w-3xl p-0">
          <div className="h-full min-h-0 flex flex-col">
            <div className="border-b border-border px-4 py-3">
              <SheetHeader className="space-y-1 text-left">
                <SheetTitle className="text-base">
                  {traderFlyoutMode === 'create' ? 'Create Auto Trader' : 'Edit Auto Trader'}
                </SheetTitle>
                <SheetDescription>
                  {traderFlyoutMode === 'create'
                    ? 'Configure a new trader profile. Templates can preload strategy and risk defaults.'
                    : 'Update runtime configuration and lifecycle state for this trader.'}
                </SheetDescription>
              </SheetHeader>
            </div>

            <ScrollArea className="flex-1 min-h-0 px-4 py-3">
              <div className="space-y-3 pb-2">
                <FlyoutSection
                  title="Trader Profile"
                  icon={Sparkles}
                  subtitle="Core identity and strategy metadata used by the orchestrator."
                >
                  {traderFlyoutMode === 'create' ? (
                    <div>
                      <Label>Template</Label>
                      <Select
                        value={templateSelection}
                        onValueChange={(value) => {
                          setTemplateSelection(value)
                          if (value !== 'none') {
                            hydrateDraftFromTemplate(value)
                          }
                        }}
                      >
                        <SelectTrigger className="mt-1 h-9">
                          <SelectValue placeholder="Start from blank trader" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">Blank Trader</SelectItem>
                          {templates.map((template) => (
                            <SelectItem key={template.id} value={template.id}>
                              {template.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ) : null}

                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <Label>Name</Label>
                      <Input value={draftName} onChange={(event) => setDraftName(event.target.value)} className="mt-1" />
                    </div>
                    <div>
                      <Label>Strategy Key</Label>
                      <Input value={draftStrategyKey} onChange={(event) => setDraftStrategyKey(event.target.value)} className="mt-1 font-mono" />
                    </div>
                  </div>

                  <div>
                    <Label>Description</Label>
                    <Input value={draftDescription} onChange={(event) => setDraftDescription(event.target.value)} className="mt-1" />
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Signal Sources"
                  icon={Zap}
                  count={`${selectedSourceCount}/${sourceCards.length || sourceCatalog.length} enabled`}
                  subtitle="Pick the signal sources this trader should consume."
                >
                  <div className="rounded-md border border-border/60 bg-muted/15 px-3 py-2">
                    <p className="text-[11px] font-medium">Source Controls</p>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={enableAllSourceCards}
                      >
                        Enable all
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setDraftSources(defaultSourceCsv)}
                      >
                        Use default
                      </Button>
                    </div>
                    <p className="mt-1 text-[10px] text-muted-foreground/70">
                      3-4 cards per row on desktop for quick source toggling.
                    </p>
                  </div>

                  {TRADER_SOURCE_GROUPS.map((group) => {
                    const groupSources = sourceCardsByGroup[group.key]
                    if (!groupSources || groupSources.length === 0) return null
                    const enabledCount = groupSources.filter((source) => selectedSourceKeySet.has(normalizeSourceKey(source.key))).length
                    return (
                      <div key={group.key} className="rounded-lg border border-border/60 bg-muted/10 p-2.5 space-y-2">
                        <div className="flex items-center justify-between gap-2">
                          <div>
                            <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{group.label}</p>
                            <p className="text-[10px] text-muted-foreground/70">{group.subtitle}</p>
                          </div>
                          <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/70 text-muted-foreground">
                            {enabledCount}/{groupSources.length}
                          </span>
                        </div>
                        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                          {groupSources.map((source) => {
                            const isEnabled = selectedSourceKeySet.has(normalizeSourceKey(source.key))
                            const descriptor = source.signal_types?.slice(0, 2).join(' • ') || source.key
                            return (
                              <button
                                key={source.key}
                                type="button"
                                onClick={() => toggleDraftSource(source.key)}
                                className={cn(
                                  'rounded-lg border px-2.5 py-2 text-left transition-colors',
                                  isEnabled
                                    ? 'border-emerald-500/40 bg-emerald-500/10'
                                    : 'border-border/70 bg-background hover:border-emerald-500/30 hover:bg-muted/40'
                                )}
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
                                <p className="mt-1 text-[9px] uppercase tracking-wide text-muted-foreground/70">
                                  {descriptor}
                                </p>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}

                  <details className="rounded-md border border-border/60 bg-muted/10 p-2.5">
                    <summary className="cursor-pointer text-xs font-medium">Custom source keys (optional)</summary>
                    <div className="mt-2 space-y-1">
                      <Input value={draftSources} onChange={(event) => setDraftSources(event.target.value)} className="h-8 font-mono text-xs" />
                      <p className="text-[10px] text-muted-foreground/70">
                        Comma-separated keys are supported for legacy/custom source adapters.
                      </p>
                    </div>
                  </details>
                </FlyoutSection>

                <FlyoutSection
                  title="Execution Cadence"
                  icon={Clock3}
                  iconClassName="text-sky-500"
                  count={`${Number(draftInterval || 0)}s`}
                  subtitle="Set how often this trader is eligible to run."
                >
                  <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                    <div>
                      <Label>Trader Interval Seconds</Label>
                      <Input
                        type="number"
                        min={1}
                        value={draftInterval}
                        onChange={(event) => {
                          setDraftInterval(event.target.value)
                          setAdvancedValue('cadenceProfile', cadenceProfileForInterval(Math.max(1, Number(event.target.value) || 60)))
                        }}
                        className="mt-1"
                      />
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        Saves to this trader&apos;s <span className="font-mono">interval_seconds</span>.
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/15 px-3 py-2">
                      <p className="text-[11px] font-medium">Global Orchestrator Loop</p>
                      <p className="mt-1 text-sm font-mono">{toNumber(overviewQuery.data?.control?.run_interval_seconds)}s</p>
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        Separate worker-level cadence. Traders run only when due on both schedules.
                      </p>
                    </div>
                  </div>
                  {isCryptoStrategyDraft && Number(draftInterval || 0) >= 60 ? (
                    <p className="text-xs text-amber-700 dark:text-amber-100">
                      60s is too slow for most BTC 15m execution loops. Recommended cadence is 2s to 10s.
                    </p>
                  ) : null}
                </FlyoutSection>

                <FlyoutSection
                  title="Runtime State"
                  icon={Play}
                  iconClassName="text-emerald-500"
                  count={`${draftEnabled ? 'enabled' : 'disabled'} / ${draftPaused ? 'paused' : 'active'}`}
                  defaultOpen={false}
                  subtitle="Lifecycle controls applied when this trader is loaded by the orchestrator."
                >
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="rounded-md border border-border p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Enabled</span>
                        <Switch checked={draftEnabled} onCheckedChange={(checked) => setDraftEnabled(checked)} />
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">Disabled traders are excluded from orchestrator cycles.</p>
                    </div>
                    <div className="rounded-md border border-border p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Paused</span>
                        <Switch checked={draftPaused} onCheckedChange={(checked) => setDraftPaused(checked)} />
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">Paused traders stay loaded but do not execute decisions.</p>
                    </div>
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Signal Gating + Selection"
                  icon={ShieldAlert}
                  iconClassName="text-amber-500"
                  count={isCryptoStrategyDraft ? '10 controls' : '7 controls'}
                  defaultOpen={false}
                >
                  {isCryptoStrategyDraft ? (
                    <div className="rounded-md border border-border/60 bg-muted/10 p-2.5 space-y-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Crypto Market Targets</p>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          className="h-6 px-2 text-[11px]"
                          onClick={enableAllCryptoTargets}
                        >
                          Use all
                        </Button>
                      </div>
                      <div className="space-y-1.5">
                        <p className="text-[11px] text-muted-foreground/80">Coins</p>
                        <div className="flex flex-wrap gap-1.5">
                          {CRYPTO_ASSET_OPTIONS.map((asset) => (
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
                      </div>
                      <div className="space-y-1.5">
                        <p className="text-[11px] text-muted-foreground/80">Market Cadence</p>
                        <div className="flex flex-wrap gap-1.5">
                          {CRYPTO_TIMEFRAME_OPTIONS.map((timeframe) => (
                            <Button
                              key={timeframe}
                              type="button"
                              size="sm"
                              variant={selectedCryptoTimeframes.has(timeframe) ? 'default' : 'outline'}
                              className="h-6 px-2 text-[11px]"
                              onClick={() => toggleCryptoTimeframeTarget(timeframe)}
                            >
                              {timeframe}
                            </Button>
                          ))}
                        </div>
                      </div>
                      {selectedCryptoAssets.size === 0 || selectedCryptoTimeframes.size === 0 ? (
                        <p className="text-[11px] text-amber-700 dark:text-amber-100">
                          Empty target lists default to all configured crypto assets and timeframes.
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                  <div className={cn('grid gap-3 md:grid-cols-3', isCryptoStrategyDraft ? 'lg:grid-cols-4' : '')}>
                    {isCryptoStrategyDraft ? (
                      <div>
                        <Label>Crypto Strategy Mode</Label>
                        <Select
                          value={advancedConfig.strategyMode}
                          onValueChange={(value) => setAdvancedValue('strategyMode', normalizeCryptoStrategyMode(value))}
                        >
                          <SelectTrigger className="mt-1">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="auto">Auto (Regime)</SelectItem>
                            <SelectItem value="directional">Directional</SelectItem>
                            <SelectItem value="pure_arb">Pure Arb</SelectItem>
                            <SelectItem value="rebalance">Rebalance</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    ) : null}
                    <div>
                      <Label>Min Signal Score</Label>
                      <Input type="number" value={advancedConfig.minSignalScore} onChange={(event) => setAdvancedValue('minSignalScore', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Min Edge (%)</Label>
                      <Input type="number" value={advancedConfig.minEdgePercent} onChange={(event) => setAdvancedValue('minEdgePercent', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Min Confidence (%)</Label>
                      <Input type="number" value={advancedConfig.minConfidence} onChange={(event) => setAdvancedValue('minConfidence', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Lookback (minutes)</Label>
                      <Input type="number" value={advancedConfig.lookbackMinutes} onChange={(event) => setAdvancedValue('lookbackMinutes', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Scan Batch Size</Label>
                      <Input type="number" value={advancedConfig.scanBatchSize} onChange={(event) => setAdvancedValue('scanBatchSize', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Signals / Cycle</Label>
                      <Input type="number" value={advancedConfig.maxSignalsPerCycle} onChange={(event) => setAdvancedValue('maxSignalsPerCycle', toNumber(event.target.value))} className="mt-1" />
                    </div>
                  </div>
                  {isCryptoStrategyDraft ? (
                    <p className="text-[11px] text-muted-foreground/80">
                      Auto switches between directional, pure-arb, and rebalance by market regime.
                    </p>
                  ) : null}
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <Label>Source Priority (comma separated)</Label>
                      <Input value={advancedConfig.sourcePriorityCsv} onChange={(event) => setAdvancedValue('sourcePriorityCsv', event.target.value)} className="mt-1" />
                    </div>
                    <div>
                      <Label>Blocked Market Keywords</Label>
                      <Input value={advancedConfig.blockedKeywordsCsv} onChange={(event) => setAdvancedValue('blockedKeywordsCsv', event.target.value)} className="mt-1" />
                    </div>
                  </div>
                  <div className="rounded-md border border-border p-2 flex items-center justify-between">
                    <span className="text-sm">Require Second Source Confirmation</span>
                    <Switch checked={advancedConfig.requireSecondSource} onCheckedChange={(checked) => setAdvancedValue('requireSecondSource', checked)} />
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Risk Envelope"
                  icon={AlertTriangle}
                  iconClassName="text-rose-500"
                  count="9 limits"
                  defaultOpen={false}
                >
                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <Label>Max Orders / Cycle</Label>
                      <Input type="number" value={advancedConfig.maxOrdersPerCycle} onChange={(event) => setAdvancedValue('maxOrdersPerCycle', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Open Orders</Label>
                      <Input type="number" value={advancedConfig.maxOpenOrders} onChange={(event) => setAdvancedValue('maxOpenOrders', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Open Positions</Label>
                      <Input type="number" value={advancedConfig.maxOpenPositions} onChange={(event) => setAdvancedValue('maxOpenPositions', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Position Notional (USD)</Label>
                      <Input type="number" value={advancedConfig.maxPositionNotionalUsd} onChange={(event) => setAdvancedValue('maxPositionNotionalUsd', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Gross Exposure (USD)</Label>
                      <Input type="number" value={advancedConfig.maxGrossExposureUsd} onChange={(event) => setAdvancedValue('maxGrossExposureUsd', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Trade Notional (USD)</Label>
                      <Input type="number" value={advancedConfig.maxTradeNotionalUsd} onChange={(event) => setAdvancedValue('maxTradeNotionalUsd', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Daily Loss (USD)</Label>
                      <Input type="number" value={advancedConfig.maxDailyLossUsd} onChange={(event) => setAdvancedValue('maxDailyLossUsd', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Daily Spend (USD)</Label>
                      <Input type="number" value={advancedConfig.maxDailySpendUsd} onChange={(event) => setAdvancedValue('maxDailySpendUsd', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Cooldown (seconds)</Label>
                      <Input type="number" value={advancedConfig.cooldownSeconds} onChange={(event) => setAdvancedValue('cooldownSeconds', toNumber(event.target.value))} className="mt-1" />
                    </div>
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Execution Quality + Circuit Breakers"
                  icon={CheckCircle2}
                  iconClassName="text-cyan-500"
                  count="10 controls"
                  defaultOpen={false}
                >
                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <Label>Order TTL (seconds)</Label>
                      <Input type="number" value={advancedConfig.orderTtlSeconds} onChange={(event) => setAdvancedValue('orderTtlSeconds', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Slippage Guard (bps)</Label>
                      <Input type="number" value={advancedConfig.slippageBps} onChange={(event) => setAdvancedValue('slippageBps', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Spread (bps)</Label>
                      <Input type="number" value={advancedConfig.maxSpreadBps} onChange={(event) => setAdvancedValue('maxSpreadBps', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Retry Limit</Label>
                      <Input type="number" value={advancedConfig.retryLimit} onChange={(event) => setAdvancedValue('retryLimit', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Retry Backoff (ms)</Label>
                      <Input type="number" value={advancedConfig.retryBackoffMs} onChange={(event) => setAdvancedValue('retryBackoffMs', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Max Consecutive Losses</Label>
                      <Input type="number" value={advancedConfig.maxConsecutiveLosses} onChange={(event) => setAdvancedValue('maxConsecutiveLosses', toNumber(event.target.value))} className="mt-1" />
                    </div>
                    <div>
                      <Label>Circuit Breaker Drawdown (%)</Label>
                      <Input type="number" value={advancedConfig.circuitBreakerDrawdownPct} onChange={(event) => setAdvancedValue('circuitBreakerDrawdownPct', toNumber(event.target.value))} className="mt-1" />
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="rounded-md border border-border p-2 flex items-center justify-between">
                      <span className="text-xs">Allow Averaging</span>
                      <Switch checked={advancedConfig.allowAveraging} onCheckedChange={(checked) => setAdvancedValue('allowAveraging', checked)} />
                    </div>
                    <div className="rounded-md border border-border p-2 flex items-center justify-between">
                      <span className="text-xs">Dynamic Position Sizing</span>
                      <Switch checked={advancedConfig.useDynamicSizing} onCheckedChange={(checked) => setAdvancedValue('useDynamicSizing', checked)} />
                    </div>
                    <div className="rounded-md border border-border p-2 flex items-center justify-between">
                      <span className="text-xs">Halt on Loss Streak</span>
                      <Switch checked={advancedConfig.haltOnConsecutiveLosses} onCheckedChange={(checked) => setAdvancedValue('haltOnConsecutiveLosses', checked)} />
                    </div>
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Session Window + Metadata"
                  icon={Sparkles}
                  iconClassName="text-blue-500"
                  count="4 fields"
                  defaultOpen={false}
                >
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <Label>Trading Window Start (UTC HH:MM)</Label>
                      <Input value={advancedConfig.tradingWindowStartUtc} onChange={(event) => setAdvancedValue('tradingWindowStartUtc', event.target.value)} className="mt-1 font-mono" />
                    </div>
                    <div>
                      <Label>Trading Window End (UTC HH:MM)</Label>
                      <Input value={advancedConfig.tradingWindowEndUtc} onChange={(event) => setAdvancedValue('tradingWindowEndUtc', event.target.value)} className="mt-1 font-mono" />
                    </div>
                    <div>
                      <Label>Tags (comma separated)</Label>
                      <Input value={advancedConfig.tagsCsv} onChange={(event) => setAdvancedValue('tagsCsv', event.target.value)} className="mt-1" />
                    </div>
                    <div>
                      <Label>Operator Notes</Label>
                      <Input value={advancedConfig.notes} onChange={(event) => setAdvancedValue('notes', event.target.value)} className="mt-1" />
                    </div>
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Advanced JSON Editors"
                  icon={Square}
                  iconClassName="text-slate-500"
                  count="3 editors"
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

                  <details className="rounded-md border border-border p-2">
                    <summary className="cursor-pointer text-xs font-medium">Metadata JSON</summary>
                    <textarea
                      className="mt-2 w-full min-h-[160px] rounded-md border bg-background p-2 text-xs font-mono"
                      value={draftMetadata}
                      onChange={(event) => setDraftMetadata(event.target.value)}
                    />
                  </details>
                </FlyoutSection>

                {traderFlyoutMode === 'edit' && selectedTrader ? (
                  <FlyoutSection
                    title="Delete / Disable Trader"
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
                        <SelectItem value="disable">Disable + Pause (Recommended)</SelectItem>
                        <SelectItem value="block">Delete (No Open Positions)</SelectItem>
                        <SelectItem value="force_delete">Force Delete (Danger)</SelectItem>
                      </SelectContent>
                    </Select>
                    {deleteAction === 'force_delete' ? (
                      <div>
                        <Label className="text-xs">
                          Type trader name to confirm force delete: <span className="font-mono">{selectedTrader.name}</span>
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
                      {deleteAction === 'disable' ? 'Disable Trader' : 'Delete Trader'}
                    </Button>
                  </FlyoutSection>
                ) : null}

                {saveError ? <div className="text-xs text-red-500">{saveError}</div> : null}
              </div>
            </ScrollArea>

            <div className="border-t border-border px-4 py-3 flex flex-wrap items-center justify-end gap-2">
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
                  !draftStrategyKey.trim()
                }
              >
                {traderFlyoutMode === 'create' ? 'Create Trader' : 'Save Trader'}
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}
