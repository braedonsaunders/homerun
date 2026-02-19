import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Clock3,
  Copy,
  Filter,
  Loader2,
  Pause,
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
  getTraderDecisionDetail,
  getTraderDecisions,
  getTraderEvents,
  getTraderConfigSchema,
  getTraderOrchestratorOverview,
  getTraderSources,
  getSimulationAccounts,
  getTraders,
  getWallets,
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
  type TraderConfigSchema,
  type TraderOrder,
  type TraderSourceConfig,
  type TraderSource,
  updateTrader,
  getActiveCopyMode,
  getDiscoverySettings,
  type ActiveCopyMode,
  type CopySourceType,
  type DiscoverySettings,
} from '../services/api'
import { discoveryApi } from '../services/discoveryApi'
import { cn } from '../lib/utils'
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
const RESUME_POLICY_OPTIONS = ['resume_full', 'manage_only', 'flatten_then_start'] as const
type ResumePolicy = (typeof RESUME_POLICY_OPTIONS)[number]
type TradersScopeMode = 'tracked' | 'pool' | 'individual' | 'group'
const TRADERS_SCOPE_OPTIONS: Array<{ key: TradersScopeMode; label: string }> = [
  { key: 'tracked', label: 'Tracked' },
  { key: 'pool', label: 'Pool' },
  { key: 'individual', label: 'Individual' },
  { key: 'group', label: 'Group' },
]

type TraderAdvancedConfig = {
  cadenceProfile: string
  strategyMode: string
  cryptoAssetsCsv: string
  cryptoExcludedAssetsCsv: string
  cryptoIncludedTimeframesCsv: string
  cryptoExcludedTimeframesCsv: string
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
  resumePolicy: ResumePolicy
  tradingWindowStartUtc: string
  tradingWindowEndUtc: string
  tagsCsv: string
  notes: string
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
  side_filter: 'all' | 'BUY' | 'SELL'
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

const OPEN_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])
const RESOLVED_ORDER_STATUSES = new Set(['resolved', 'resolved_win', 'resolved_loss', 'win', 'loss'])
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

function normalizeResumePolicy(value: unknown): ResumePolicy {
  const policy = String(value || '').trim().toLowerCase()
  if (RESUME_POLICY_OPTIONS.includes(policy as ResumePolicy)) {
    return policy as ResumePolicy
  }
  return 'resume_full'
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
  if (side === 'buy') return 'BUY'
  if (side === 'sell') return 'SELL'
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

function strategyHasParamKey(
  strategy: StrategyOptionDetail | null | undefined,
  paramKey: string
): boolean {
  if (!strategy || !paramKey) return false
  if (strategy.paramFields.some((field) => String(field.key || '').trim().toLowerCase() === paramKey)) {
    return true
  }
  return Object.prototype.hasOwnProperty.call(strategy.defaultParams, paramKey)
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
  if (active.mode === 'disabled' || !active.config_id) {
    return DEFAULT_COPY_TRADING
  }
  return {
    copy_mode_type: active.mode as CopyTradingMode,
    individual_wallet: active.source_wallet || '',
    account_id: active.account_id || '',
    copy_trade_mode: (active.copy_mode || 'all_trades') as 'all_trades' | 'arb_only',
    max_position_size: active.settings?.max_position_size ?? 1000,
    proportional_sizing: active.settings?.proportional_sizing ?? false,
    proportional_multiplier: active.settings?.proportional_multiplier ?? 1.0,
    copy_buys: active.settings?.copy_buys ?? true,
    copy_sells: active.settings?.copy_sells ?? true,
    copy_delay_seconds: active.settings?.copy_delay_seconds ?? 5,
    slippage_tolerance: active.settings?.slippage_tolerance ?? 1.0,
    min_roi_threshold: active.settings?.min_roi_threshold ?? 2.5,
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function defaultAdvancedConfig(): TraderAdvancedConfig {
  return {
    cadenceProfile: 'custom',
    strategyMode: 'auto',
    cryptoAssetsCsv: CRYPTO_ASSET_OPTIONS.join(', '),
    cryptoExcludedAssetsCsv: '',
    cryptoIncludedTimeframesCsv: CRYPTO_TIMEFRAME_OPTIONS.join(', '),
    cryptoExcludedTimeframesCsv: '',
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
    resumePolicy: 'resume_full',
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
  const configuredCryptoAssets = normalizeCryptoAssetList(params.include_assets)
  const excludedCryptoAssets = normalizeCryptoAssetList(params.exclude_assets)
  const configuredCryptoTimeframes = normalizeCryptoTimeframeList(params.include_timeframes)
  const excludedCryptoTimeframes = normalizeCryptoTimeframeList(params.exclude_timeframes)

  return {
    cadenceProfile: String(metadata.cadence_profile || defaults.cadenceProfile),
    strategyMode: normalizeCryptoStrategyMode(params.strategy_mode ?? params.mode ?? defaults.strategyMode),
    cryptoAssetsCsv: (configuredCryptoAssets.length > 0 ? configuredCryptoAssets : [...CRYPTO_ASSET_OPTIONS]).join(', '),
    cryptoExcludedAssetsCsv: excludedCryptoAssets.join(', '),
    cryptoIncludedTimeframesCsv: (configuredCryptoTimeframes.length > 0 ? configuredCryptoTimeframes : [...CRYPTO_TIMEFRAME_OPTIONS]).join(', '),
    cryptoExcludedTimeframesCsv: excludedCryptoTimeframes.join(', '),
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
    resumePolicy: normalizeResumePolicy(metadata.resume_policy ?? defaults.resumePolicy),
    tradingWindowStartUtc: String(tradingWindow.start || defaults.tradingWindowStartUtc),
    tradingWindowEndUtc: String(tradingWindow.end || defaults.tradingWindowEndUtc),
    tagsCsv: toCsv(metadata.tags ?? defaults.tagsCsv),
    notes: String(metadata.notes || defaults.notes),
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
  const hasSchema = Boolean(
    strategyDetail &&
    (strategyDetail.paramFields.length > 0 || Object.keys(strategyDetail.defaultParams).length > 0)
  )
  const applyIfSupported = (paramKey: string, value: unknown) => {
    if (!hasSchema || strategyHasParamKey(strategyDetail, paramKey)) {
      next[paramKey] = value
    } else {
      delete next[paramKey]
    }
  }
  applyIfSupported('min_signal_score', config.minSignalScore)
  applyIfSupported('min_edge_percent', config.minEdgePercent)
  applyIfSupported('min_confidence', confidencePercentToFraction(config.minConfidence))
  applyIfSupported('lookback_minutes', config.lookbackMinutes)
  applyIfSupported('scan_batch_size', config.scanBatchSize)
  applyIfSupported('max_signals_per_cycle', config.maxSignalsPerCycle)
  applyIfSupported('require_second_source', config.requireSecondSource)
  applyIfSupported('source_priority', csvToList(config.sourcePriorityCsv))
  applyIfSupported('blocked_market_keywords', csvToList(config.blockedKeywordsCsv))

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
    resume_policy: config.resumePolicy,
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
  const [workTab, setWorkTab] = useState<'terminal' | 'positions' | 'trades' | 'decisions' | 'risk'>('terminal')
  const [activityStreamOpen, setActivityStreamOpen] = useState(true)

  const [traderFlyoutOpen, setTraderFlyoutOpen] = useState(false)
  const [traderFlyoutMode, setTraderFlyoutMode] = useState<'create' | 'edit'>('create')
  const [draftName, setDraftName] = useState('')
  const [draftDescription, setDraftDescription] = useState('')
  const [, setDraftStrategyKey] = useState(DEFAULT_STRATEGY_KEY)
  const [draftSourceStrategies, setDraftSourceStrategies] = useState<Record<string, string>>({})
  const [draftInterval, setDraftInterval] = useState('60')
  const [draftSources, setDraftSources] = useState('')
  const [draftEnabled, setDraftEnabled] = useState(true)
  const [draftPaused, setDraftPaused] = useState(false)
  const [draftTradersScopeModes, setDraftTradersScopeModes] = useState<TradersScopeMode[]>(['tracked', 'pool'])
  const [draftTradersIndividualWallets, setDraftTradersIndividualWallets] = useState<string[]>([])
  const [draftTradersGroupIds, setDraftTradersGroupIds] = useState<string[]>([])
  const [draftParams, setDraftParams] = useState('{}')
  const [draftRisk, setDraftRisk] = useState('{}')
  const [draftMetadata, setDraftMetadata] = useState('{}')
  const [advancedConfig, setAdvancedConfig] = useState<TraderAdvancedConfig>(defaultAdvancedConfig())
  const [saveError, setSaveError] = useState<string | null>(null)
  const [deleteAction, setDeleteAction] = useState<'block' | 'disable' | 'force_delete'>('disable')
  const [deleteConfirmName, setDeleteConfirmName] = useState('')
  const [draftCopyTrading, setDraftCopyTrading] = useState<CopyTradingFormState>(DEFAULT_COPY_TRADING)
  const [draftSignalFilters, setDraftSignalFilters] = useState<TraderSignalFilters>(DEFAULT_SIGNAL_FILTERS)

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
    if (!selectedTraderId && traders.length > 0) {
      setSelectedTraderId(traders[0].id)
    }
  }, [selectedTraderId, traders])

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
    setDraftEnabled(true)
    setDraftPaused(false)
    setDraftTradersScopeModes(['tracked', 'pool'])
    setDraftTradersIndividualWallets([])
    setDraftTradersGroupIds([])
    setDraftParams('{}')
    setDraftRisk('{}')
    setDraftMetadata('{}')
    setAdvancedConfig(defaultAdvancedConfig())
    setDeleteAction('disable')
    setDeleteConfirmName('')
    setSaveError(null)
    setDraftCopyTrading(activeCopyMode && activeCopyMode.mode !== 'disabled'
      ? copyTradingFromActiveMode(activeCopyMode)
      : DEFAULT_COPY_TRADING)
    setDraftSignalFilters(signalFiltersFromDiscoverySettings(discoverySettingsQuery.data))
    setTraderFlyoutOpen(true)
  }

  const openEditTraderFlyout = (trader: Trader) => {
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
    const tradersScope = traderSourceConfigs.find((config) => normalizeSourceKey(String(config.source_key || '')) === 'traders')?.traders_scope

    setSelectedTraderId(trader.id)
    setTraderFlyoutMode('edit')
    setDraftName(trader.name)
    setDraftDescription(trader.description || '')
    setDraftStrategyKey(normalizeStrategyKey(sourceStrategyMap.crypto || DEFAULT_STRATEGY_KEY))
    setDraftSourceStrategies(sourceStrategyMap)
    setDraftInterval(String(trader.interval_seconds || 60))
    setDraftSources(normalizedSourceKeys.join(', ') || defaultSourceCsv)
    setDraftEnabled(Boolean(trader.is_enabled))
    setDraftPaused(Boolean(trader.is_paused))
    const risk = trader.risk_limits || {}
    const metadata = trader.metadata || {}
    setDraftParams(JSON.stringify(primaryParams, null, 2))
    setDraftRisk(JSON.stringify(risk, null, 2))
    setDraftMetadata(JSON.stringify(metadata, null, 2))
    setAdvancedConfig(computeAdvancedConfig(cryptoParams, risk, metadata))
    setDraftTradersScopeModes(
      (tradersScope?.modes || ['tracked', 'pool'])
        .map((mode) => String(mode || '').toLowerCase())
        .filter((mode): mode is TradersScopeMode => TRADERS_SCOPE_OPTIONS.some((option) => option.key === mode))
    )
    setDraftTradersIndividualWallets(
      (tradersScope?.individual_wallets || []).map((wallet) => String(wallet || '').trim().toLowerCase()).filter(Boolean)
    )
    setDraftTradersGroupIds(
      (tradersScope?.group_ids || []).map((groupId) => String(groupId || '').trim()).filter(Boolean)
    )
    setDeleteAction('disable')
    setDeleteConfirmName('')
    setSaveError(null)
    setDraftCopyTrading(activeCopyMode && activeCopyMode.mode !== 'disabled'
      ? copyTradingFromActiveMode(activeCopyMode)
      : DEFAULT_COPY_TRADING)
    setDraftSignalFilters(signalFiltersFromDiscoverySettings(discoverySettingsQuery.data))
    setTraderFlyoutOpen(true)
  }

  const setCt = (partial: Partial<CopyTradingFormState>) =>
    setDraftCopyTrading((prev) => ({ ...prev, ...partial }))

  const setSf = (partial: Partial<TraderSignalFilters>) =>
    setDraftSignalFilters((prev) => ({ ...prev, ...partial }))

  const setAdvancedValue = <K extends keyof TraderAdvancedConfig>(key: K, value: TraderAdvancedConfig[K]) => {
    setAdvancedConfig((current) => ({ ...current, [key]: value }))
  }

  const toggleTradersScopeMode = (mode: TradersScopeMode) => {
    setDraftTradersScopeModes((current) => {
      const hasMode = current.includes(mode)
      const next = hasMode ? current.filter((item) => item !== mode) : [...current, mode]
      return next.length > 0 ? next : ['tracked']
    })
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
        nextConfig.traders_scope = {
          modes: draftTradersScopeModes,
          individual_wallets: draftTradersIndividualWallets,
          group_ids: draftTradersGroupIds,
        }
      }
      configs.push(nextConfig)
    }
    return configs
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

      return createTrader({
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        source_configs: buildDraftSourceConfigs(parsedParams.value),
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

      return updateTrader(traderId, {
        name: draftName.trim(),
        description: draftDescription.trim() || null,
        interval_seconds: Math.max(1, Math.trunc(toNumber(draftInterval || 60))),
        source_configs: buildDraftSourceConfigs(parsedParams.value),
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
      setSaveError(errorMessage(error, 'Failed to delete or disable bot'))
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
  const startStopIsConfigured = orchestratorEnabled
  const startStopIsRunning = orchestratorRunning
  const startStopDisabled = startStopIsConfigured ? !canStopOrchestrator : !canStartOrchestrator
  const startStopPending = startStopIsConfigured ? stopByModeMutation.isPending : startBySelectedAccountMutation.isPending

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
      <div className="shrink-0 rounded-lg border border-cyan-500/30 bg-gradient-to-r from-cyan-500/[0.06] via-card to-emerald-500/[0.06] px-3 py-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
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
            {startStopIsConfigured
              ? (startStopIsRunning ? 'Stop' : 'Start')
              : selectedAccountMode.toUpperCase()}
          </Button>
          <div className="flex items-center gap-1.5 rounded border border-red-500/30 bg-red-500/5 px-1.5 py-0.5">
            <ShieldAlert className="w-3 h-3 text-red-400" />
            <Switch
              checked={killSwitchOn}
              onCheckedChange={(enabled) => killSwitchMutation.mutate(enabled)}
              disabled={controlBusy}
              className="scale-[0.8]"
            />
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
          <span className={cn('w-1.5 h-1.5 rounded-full', worker?.last_error ? 'bg-amber-400' : 'bg-emerald-500')} />
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
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-1.5 space-y-0.5">
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
                All Bots
              </button>
              {traders.map((trader) => {
                const traderStatus = !trader.is_enabled ? 'disabled' : trader.is_paused ? 'paused' : 'running'
                const isActive = trader.id === selectedTraderId
                const traderSources = traderSourceKeys(trader)
                return (
                  <button
                    key={trader.id}
                    type="button"
                    onClick={() => setSelectedTraderId(trader.id)}
                    className={cn(
                      'w-full text-left rounded-md px-2 py-1.5 transition-colors group',
                      isActive
                        ? 'bg-cyan-500/15 text-foreground'
                        : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
                    )}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className={cn(
                        'w-1.5 h-1.5 rounded-full shrink-0',
                        traderStatus === 'running' && orchestratorRunning ? 'bg-emerald-500' :
                        traderStatus === 'paused' ? 'bg-amber-400' : 'bg-zinc-500'
                      )} />
                      <span className="text-[11px] font-medium truncate leading-tight">{trader.name}</span>
                    </div>
                    <div className="flex flex-wrap gap-0.5 mt-0.5 pl-3">
                      {traderSources.map((source) => (
                        <span key={source} className="px-1 py-0 text-[8px] rounded bg-muted/60 text-muted-foreground leading-relaxed">{source}</span>
                      ))}
                    </div>
                  </button>
                )
              })}
            </div>
          </ScrollArea>
        </div>

        {/* Right — Work Area */}
        <div className="flex flex-col min-h-0 gap-1.5">
          {/* Bot header bar (when a specific bot is selected) */}
          {selectedTrader && (
            <div className="shrink-0 rounded-lg border border-border/70 bg-card px-3 py-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="text-sm font-semibold">{selectedTrader.name}</span>
              <Badge className="h-5 px-1.5 text-[10px]" variant={selectedTraderStatusLabel === 'Running' ? 'default' : selectedTraderStatusLabel === 'Paused' ? 'secondary' : 'outline'}>
                {selectedTraderStatusLabel}
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
                <Button size="sm" variant="outline" className="h-6 px-2 text-[10px]" disabled={!selectedTraderCanResume} onClick={() => traderStartMutation.mutate(selectedTrader.id)}>
                  <Play className="w-3 h-3 mr-0.5" /> Resume
                </Button>
                <Button size="sm" variant="outline" className="h-6 px-2 text-[10px]" disabled={!selectedTraderCanPause} onClick={() => traderPauseMutation.mutate(selectedTrader.id)}>
                  <Pause className="w-3 h-3 mr-0.5" /> Pause
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

          {/* Tab bar */}
          <div className="shrink-0 flex items-center gap-0.5 border-b border-border/50 px-1">
            {([
              { key: 'terminal' as const, label: 'Terminal' },
              { key: 'positions' as const, label: 'Positions' },
              { key: 'trades' as const, label: 'Trades' },
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

          {/* Tab content — fills remaining space */}
          <div className="flex-1 min-h-0 overflow-hidden">
            {/* ── Terminal ── */}
            {workTab === 'terminal' && (
              <div className="h-full flex flex-col min-h-0 gap-1.5">
                <div className="shrink-0 flex flex-wrap items-center gap-1 px-1">
                  {(['all', 'decision', 'order', 'event'] as FeedFilter[]).map((kind) => (
                    <Button key={kind} size="sm" variant={selectedTraderId ? (traderFeedFilter === kind ? 'default' : 'outline') : (feedFilter === kind ? 'default' : 'outline')} onClick={() => selectedTraderId ? setTraderFeedFilter(kind) : setFeedFilter(kind)} className="h-5 px-2 text-[10px]">
                      {kind}
                    </Button>
                  ))}
                  {!selectedTraderId && (
                    <div className="ml-auto flex items-center gap-1">
                      <Button size="sm" variant={scopeFilter === 'all' ? 'default' : 'outline'} className="h-5 px-2 text-[10px]" onClick={() => setScopeFilter('all')}>all</Button>
                      <Button size="sm" variant={scopeFilter === 'selected' ? 'default' : 'outline'} className="h-5 px-2 text-[10px]" onClick={() => setScopeFilter('selected')} disabled={!selectedTraderId}>selected</Button>
                    </div>
                  )}
                </div>
                {selectedTraderId && selectedTraderNoNewRows && (
                  <div className="shrink-0 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-700 dark:text-amber-100 mx-1">
                    No new rows since last cycle ({formatTimestamp(selectedTrader?.last_run_at || worker?.last_run_at)}).
                  </div>
                )}
                <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/50 bg-muted/10 mx-1">
                  <div className="space-y-0.5 p-1.5 font-mono text-[11px] leading-relaxed">
                    {(selectedTraderId ? filteredTraderActivityRows : filteredActivityRows).length === 0 ? (
                      <div className="py-8 text-center text-muted-foreground text-xs">No events matching filters.</div>
                    ) : (
                      (selectedTraderId ? filteredTraderActivityRows : filteredActivityRows).map((row) => (
                        <div
                          key={`${row.kind}:${row.id}`}
                          className={cn(
                            'rounded border px-2 py-0.5',
                            row.tone === 'positive' && 'border-emerald-500/25 text-emerald-700 dark:text-emerald-100',
                            row.tone === 'negative' && 'border-red-500/30 text-red-700 dark:text-red-100',
                            row.tone === 'warning' && 'border-amber-500/30 text-amber-700 dark:text-amber-100',
                            row.tone === 'neutral' && 'border-border/50 text-foreground'
                          )}
                        >
                          <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>{' '}
                          <span className="uppercase text-[10px]">{row.kind}</span>{' '}
                          {!selectedTraderId && <><span className="text-muted-foreground">{traderNameById[String(row.traderId || '')] || shortId(row.traderId || '')}</span>{' '}</>}
                          <span>{row.title}</span>
                          <span className="text-muted-foreground"> :: {row.detail}</span>
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </div>
            )}

            {/* ── Positions ── */}
            {workTab === 'positions' && (
              <div className="h-full min-h-0 overflow-hidden px-1">
                {(selectedTraderId ? selectedPositionBook : globalPositionBook).length === 0 ? (
                  <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No open positions.</div>
                ) : (
                  <ScrollArea className="h-full min-h-0">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          {!selectedTraderId && <TableHead className="text-[11px]">Bot</TableHead>}
                          <TableHead className="text-[11px]">Market</TableHead>
                          <TableHead className="text-[11px]">Dir</TableHead>
                          <TableHead className="text-[11px] text-right">Exposure</TableHead>
                          <TableHead className="text-[11px] text-right">AvgPx</TableHead>
                          <TableHead className="text-[11px] text-right">Edge</TableHead>
                          <TableHead className="text-[11px] text-right">Conf</TableHead>
                          <TableHead className="text-[11px] text-right">Orders</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(selectedTraderId ? selectedPositionBook : globalPositionBook).map((row) => (
                          <TableRow key={row.key} className="text-xs">
                            {!selectedTraderId && <TableCell className="font-mono py-1 text-[11px]">{row.traderName}</TableCell>}
                            <TableCell className="max-w-[240px] truncate py-1" title={row.marketQuestion}>{row.marketQuestion}</TableCell>
                            <TableCell className="py-1">
                              <Badge variant={row.direction === 'YES' ? 'default' : 'secondary'} className="text-[10px] h-5 px-1.5">{row.direction}</Badge>
                            </TableCell>
                            <TableCell className="text-right font-mono py-1">{formatCurrency(row.exposureUsd)}</TableCell>
                            <TableCell className="text-right font-mono py-1">{row.averagePrice !== null ? row.averagePrice.toFixed(3) : 'n/a'}</TableCell>
                            <TableCell className="text-right font-mono py-1">{row.weightedEdge !== null ? formatPercent(row.weightedEdge) : 'n/a'}</TableCell>
                            <TableCell className="text-right font-mono py-1">{row.weightedConfidence !== null ? formatPercent(normalizeConfidencePercent(row.weightedConfidence)) : 'n/a'}</TableCell>
                            <TableCell className="text-right font-mono py-1">{row.orderCount}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ScrollArea>
                )}
              </div>
            )}

            {/* ── Trades ── */}
            {workTab === 'trades' && (
              <div className="h-full flex flex-col min-h-0 gap-1.5">
                <div className="shrink-0 flex flex-wrap items-center gap-1 px-1">
                  <Input value={tradeSearch} onChange={(event) => setTradeSearch(event.target.value)} placeholder="Search..." className="h-6 w-36 text-[11px]" />
                  {(['all', 'open', 'resolved', 'failed'] as TradeStatusFilter[]).map((status) => (
                    <Button key={status} size="sm" variant={tradeStatusFilter === status ? 'default' : 'outline'} onClick={() => setTradeStatusFilter(status)} className="h-5 px-2 text-[10px]">{status}</Button>
                  ))}
                </div>
                <div className="flex-1 min-h-0 overflow-hidden px-1">
                  {filteredTradeHistory.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No trades matching filters.</div>
                  ) : (
                    <ScrollArea className="h-full min-h-0">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            {!selectedTraderId && <TableHead className="text-[11px]">Bot</TableHead>}
                            <TableHead className="text-[11px]">Market</TableHead>
                            <TableHead className="text-[11px]">Dir</TableHead>
                            <TableHead className="text-[11px]">Status</TableHead>
                            <TableHead className="text-[11px] text-right">Notional</TableHead>
                            <TableHead className="text-[11px] text-right">Edge</TableHead>
                            <TableHead className="text-[11px] text-right">P&amp;L</TableHead>
                            <TableHead className="text-[11px]">Mode</TableHead>
                            <TableHead className="text-[11px]">Created</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {filteredTradeHistory.map((order) => {
                            const status = normalizeStatus(order.status)
                            const pnl = toNumber(order.actual_profit)
                            return (
                              <TableRow key={order.id} className="text-xs">
                                {!selectedTraderId && <TableCell className="font-mono py-1 text-[11px]">{traderNameById[String(order.trader_id || '')] || shortId(order.trader_id || '')}</TableCell>}
                                <TableCell className="max-w-[220px] truncate py-1" title={order.market_question || order.market_id}>{order.market_question || shortId(order.market_id)}</TableCell>
                                <TableCell className="py-1">
                                  <Badge variant={String(order.direction || '').toUpperCase() === 'YES' ? 'default' : 'secondary'} className="text-[10px] h-5 px-1.5">{String(order.direction || '').toUpperCase()}</Badge>
                                </TableCell>
                                <TableCell className="py-1">
                                  <Badge variant={OPEN_ORDER_STATUSES.has(status) ? 'default' : RESOLVED_ORDER_STATUSES.has(status) ? (pnl >= 0 ? 'default' : 'destructive') : 'outline'} className="text-[10px] h-5 px-1.5">{status}</Badge>
                                </TableCell>
                                <TableCell className="text-right font-mono py-1">{formatCurrency(toNumber(order.notional_usd))}</TableCell>
                                <TableCell className="text-right font-mono py-1">{formatPercent(toNumber(order.edge_percent))}</TableCell>
                                <TableCell className={cn('text-right font-mono py-1', pnl > 0 ? 'text-emerald-500' : pnl < 0 ? 'text-red-500' : '')}>{RESOLVED_ORDER_STATUSES.has(status) ? formatCurrency(pnl) : '\u2014'}</TableCell>
                                <TableCell className="py-1 uppercase text-[10px]">{String(order.mode || '').toUpperCase()}</TableCell>
                                <TableCell className="py-1 text-[10px] text-muted-foreground">{formatShortDate(order.created_at)}</TableCell>
                              </TableRow>
                            )
                          })}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  )}
                </div>
              </div>
            )}

            {/* ── Decisions ── */}
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
                                <span className="truncate max-w-[180px]" title={decision.market_question || decision.market_id}>{decision.market_question || shortId(decision.market_id)}</span>
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
                          <span>Direction: {String(selectedDecision.direction || 'n/a').toUpperCase()}</span>
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
                                  <span className="font-medium">{check.check_name}</span>
                                </div>
                                {check.message ? <p className="text-[10px] text-muted-foreground mt-0.5 pl-4">{check.message}</p> : null}
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
                                {decisionOrders.map((order: any) => (
                                  <div key={order.id} className="rounded border border-border px-2 py-1 font-mono text-[10px]">
                                    {normalizeStatus(order.status).toUpperCase()} {'\u2022'} {formatCurrency(toNumber(order.notional_usd))} {'\u2022'} {String(order.direction || '').toUpperCase()}
                                  </div>
                                ))}
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

            {/* ── Risk ── */}
            {workTab === 'risk' && (
              <div className="h-full min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] px-1">
                {/* Governance */}
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

                {/* Risk + Failure Terminal */}
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
                          <div key={`${row.kind}:${row.id}`} className={cn('rounded border px-2 py-0.5', row.tone === 'negative' ? 'border-red-500/30 text-red-700 dark:text-red-100' : 'border-amber-500/30 text-amber-700 dark:text-amber-100')}>
                            <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>{' '}
                            <span className="uppercase text-[10px]">{row.kind}</span>{' '}
                            <span>{row.title}</span>
                            <span className="text-muted-foreground"> :: {row.detail}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </div>
              </div>
            )}
          </div>

          {/* ── Collapsible Activity Stream ── */}
          <div className="shrink-0">
            <button
              type="button"
              onClick={() => setActivityStreamOpen((prev) => !prev)}
              className="w-full flex items-center gap-1.5 px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors border-t border-border/40"
            >
              {activityStreamOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
              Recent Selections ({recentSelectedDecisions.length})
            </button>
            {activityStreamOpen && (
              <div className="max-h-[100px] overflow-auto border-t border-border/30 px-2 py-1 space-y-0.5">
                {recentSelectedDecisions.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground">No recent selections.</p>
                ) : (
                  recentSelectedDecisions.slice(0, 8).map((decision) => (
                    <div key={decision.id} className="flex items-center justify-between gap-2 text-[10px] font-mono">
                      <span className="truncate max-w-[300px] text-muted-foreground" title={decision.market_question || ''}>{decision.market_question || shortId(decision.market_id)}</span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <Badge variant="outline" className="text-[8px] h-3.5 px-1">{decision.source}</Badge>
                        <span className="text-muted-foreground">{formatShortDate(decision.created_at)}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
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
                    ? 'Configure a new bot profile with explicit strategy, source, risk, and lifecycle controls.'
                    : 'Update runtime configuration and lifecycle state for this bot.'}
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
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-orange-400">Wallet Scope</p>
                        <div className="flex flex-wrap gap-1.5">
                          {TRADERS_SCOPE_OPTIONS.map((option) => (
                            <Button
                              key={option.key}
                              type="button"
                              size="sm"
                              variant={draftTradersScopeModes.includes(option.key) ? 'default' : 'outline'}
                              className="h-6 px-2 text-[11px]"
                              onClick={() => toggleTradersScopeMode(option.key)}
                            >
                              {option.label}
                            </Button>
                          ))}
                        </div>
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
                      <div className="flex items-center gap-1.5">
                        <Filter className="w-3.5 h-3.5 text-orange-400" />
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-orange-400">Wallet Signal Filters (Global)</p>
                      </div>
                      <p className="text-[10px] text-muted-foreground/80">
                        These filters are shared with wallet-opportunity discovery settings.
                      </p>
                      <div className="grid grid-cols-2 gap-2.5">
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Source Filter</Label>
                          <select
                            value={draftSignalFilters.source_filter}
                            onChange={(e) => setSf({ source_filter: e.target.value as TraderSignalFilters['source_filter'] })}
                            className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                          >
                            <option value="all">All sources</option>
                            <option value="tracked">Tracked Wallets</option>
                            <option value="pool">Pool Wallets</option>
                          </select>
                        </div>
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Min Confluence Tier</Label>
                          <select
                            value={draftSignalFilters.min_tier}
                            onChange={(e) => setSf({ min_tier: e.target.value as TraderSignalFilters['min_tier'] })}
                            className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                          >
                            <option value="WATCH">Watch (5+)</option>
                            <option value="HIGH">High (10+)</option>
                            <option value="EXTREME">Extreme (15+)</option>
                          </select>
                        </div>
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Side Filter</Label>
                          <select
                            value={draftSignalFilters.side_filter}
                            onChange={(e) => setSf({ side_filter: e.target.value as TraderSignalFilters['side_filter'] })}
                            className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                          >
                            <option value="all">All sides</option>
                            <option value="BUY">Buy only</option>
                            <option value="SELL">Sell only</option>
                          </select>
                        </div>
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Confluence Limit</Label>
                          <Input
                            type="number"
                            value={draftSignalFilters.confluence_limit}
                            onChange={(e) => setSf({ confluence_limit: Math.round(clamp(Number(e.target.value) || 1, 1, 200)) })}
                            min={1} max={200} className="mt-0.5 text-xs h-7"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2.5">
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Individual Trade Limit</Label>
                          <Input
                            type="number"
                            value={draftSignalFilters.individual_trade_limit}
                            onChange={(e) => setSf({ individual_trade_limit: Math.round(clamp(Number(e.target.value) || 1, 1, 500)) })}
                            min={1} max={500} className="mt-0.5 text-xs h-7"
                          />
                        </div>
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Min Conf</Label>
                          <Input
                            type="number"
                            value={draftSignalFilters.individual_trade_min_confidence}
                            onChange={(e) => setSf({ individual_trade_min_confidence: clamp(Number(e.target.value) || 0, 0, 1) })}
                            min={0} max={1} step={0.01} className="mt-0.5 text-xs h-7"
                          />
                        </div>
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Max Age (min)</Label>
                          <Input
                            type="number"
                            value={draftSignalFilters.individual_trade_max_age_minutes}
                            onChange={(e) => setSf({ individual_trade_max_age_minutes: Math.round(clamp(Number(e.target.value) || 1, 1, 1440)) })}
                            min={1} max={1440} className="mt-0.5 text-xs h-7"
                          />
                        </div>
                      </div>

                      {/* Copy Trading config */}
                      <div className="border-t border-orange-500/20 pt-3 space-y-2.5">
                        <div className="flex items-center gap-1.5">
                          <Copy className="w-3.5 h-3.5 text-green-400" />
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-green-400">Copy Trading</p>
                          {draftCopyTrading.copy_mode_type !== 'disabled' && (
                            <span className="ml-auto text-[9px] px-1.5 py-0.5 rounded-full bg-green-500/20 text-green-400 font-medium">ACTIVE</span>
                          )}
                        </div>
                        <div>
                          <Label className="text-[11px] text-muted-foreground leading-tight">Mode</Label>
                          <select
                            value={draftCopyTrading.copy_mode_type}
                            onChange={(e) => setCt({ copy_mode_type: e.target.value as CopyTradingMode })}
                            className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                          >
                            <option value="disabled">Disabled (signals only)</option>
                            <option value="pool">Pool Wallets</option>
                            <option value="tracked_group">Tracked Wallets (all tracked)</option>
                            <option value="individual">Individual Wallet</option>
                          </select>
                        </div>

                        {draftCopyTrading.copy_mode_type === 'individual' && (
                          <div>
                            <Label className="text-[11px] text-muted-foreground leading-tight">Wallet Address</Label>
                            <Input
                              type="text"
                              value={draftCopyTrading.individual_wallet}
                              onChange={(e) => setCt({ individual_wallet: e.target.value })}
                              placeholder="0x..."
                              className="mt-0.5 text-xs h-7 font-mono"
                            />
                          </div>
                        )}

                        {draftCopyTrading.copy_mode_type !== 'disabled' && (
                          <>
                            <div className="grid grid-cols-2 gap-2.5">
                              <div>
                                <Label className="text-[11px] text-muted-foreground leading-tight">Copy Mode</Label>
                                <select
                                  value={draftCopyTrading.copy_trade_mode}
                                  onChange={(e) => setCt({ copy_trade_mode: e.target.value as 'all_trades' | 'arb_only' })}
                                  className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                                >
                                  <option value="all_trades">All Trades</option>
                                  <option value="arb_only">Arb-Matching Only</option>
                                </select>
                              </div>
                              <div>
                                <Label className="text-[11px] text-muted-foreground leading-tight">Max Position ($)</Label>
                                <Input
                                  type="number" value={draftCopyTrading.max_position_size}
                                  onChange={(e) => setCt({ max_position_size: clamp(Number(e.target.value) || 10, 10, 1000000) })}
                                  min={10} max={1000000} step={10} className="mt-0.5 text-xs h-7"
                                />
                              </div>
                              <div>
                                <Label className="text-[11px] text-muted-foreground leading-tight">Copy Delay (sec)</Label>
                                <Input
                                  type="number" value={draftCopyTrading.copy_delay_seconds}
                                  onChange={(e) => setCt({ copy_delay_seconds: Math.round(clamp(Number(e.target.value) || 0, 0, 300)) })}
                                  min={0} max={300} className="mt-0.5 text-xs h-7"
                                />
                              </div>
                              <div>
                                <Label className="text-[11px] text-muted-foreground leading-tight">Slippage (%)</Label>
                                <Input
                                  type="number" value={draftCopyTrading.slippage_tolerance}
                                  onChange={(e) => setCt({ slippage_tolerance: clamp(Number(e.target.value) || 0, 0, 10) })}
                                  min={0} max={10} step={0.1} className="mt-0.5 text-xs h-7"
                                />
                              </div>
                              {draftCopyTrading.copy_trade_mode === 'arb_only' && (
                                <div>
                                  <Label className="text-[11px] text-muted-foreground leading-tight">Min ROI (%)</Label>
                                  <Input
                                    type="number" value={draftCopyTrading.min_roi_threshold}
                                    onChange={(e) => setCt({ min_roi_threshold: clamp(Number(e.target.value) || 0, 0, 100) })}
                                    min={0} max={100} step={0.5} className="mt-0.5 text-xs h-7"
                                  />
                                </div>
                              )}
                            </div>
                            <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                              <label className="flex items-center gap-1.5 text-[11px]">
                                <input type="checkbox" checked={draftCopyTrading.copy_buys} onChange={(e) => setCt({ copy_buys: e.target.checked })} className="accent-green-500" />
                                Copy Buys
                              </label>
                              <label className="flex items-center gap-1.5 text-[11px]">
                                <input type="checkbox" checked={draftCopyTrading.copy_sells} onChange={(e) => setCt({ copy_sells: e.target.checked })} className="accent-green-500" />
                                Copy Sells
                              </label>
                              <label className="flex items-center gap-1.5 text-[11px]">
                                <input type="checkbox" checked={draftCopyTrading.proportional_sizing} onChange={(e) => setCt({ proportional_sizing: e.target.checked })} className="accent-green-500" />
                                Proportional Sizing
                              </label>
                            </div>
                            {draftCopyTrading.proportional_sizing && (
                              <div className="ml-4">
                                <Label className="text-[11px] text-muted-foreground leading-tight">Multiplier</Label>
                                <Input
                                  type="number" value={draftCopyTrading.proportional_multiplier}
                                  onChange={(e) => setCt({ proportional_multiplier: clamp(Number(e.target.value) || 0.01, 0.01, 100) })}
                                  min={0.01} max={100} step={0.1} className="mt-0.5 text-xs h-7 w-32"
                                />
                                <p className="text-[10px] text-muted-foreground/60 mt-0.5">0.1 = 10% of source size</p>
                              </div>
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
                          </>
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
                  subtitle="Scheduling only. Strategy/selection logic is configured in Signal Sources and Signal Gating."
                >
                  <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                    <div>
                      <Label>Bot Interval Seconds</Label>
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
                        Saves to this bot&apos;s <span className="font-mono">interval_seconds</span>.
                      </p>
                    </div>
                    <div className="rounded-md border border-border/60 bg-muted/15 px-3 py-2">
                      <p className="text-[11px] font-medium">Global Orchestrator Loop</p>
                      <p className="mt-1 text-sm font-mono">{toNumber(overviewQuery.data?.control?.run_interval_seconds)}s</p>
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        Separate worker-level cadence. Bots run only when due on both schedules.
                      </p>
                    </div>
                  </div>
                  {isCryptoStrategyDraft && cryptoStrategyKeyDraft === 'btc_eth_highfreq' && Number(draftInterval || 0) >= 60 ? (
                    <p className="text-xs text-amber-700 dark:text-amber-100">
                      60s is too slow for short-horizon crypto execution. Recommended cadence is 2s to 10s.
                    </p>
                  ) : null}
                </FlyoutSection>

                <FlyoutSection
                  title="Runtime State"
                  icon={Play}
                  iconClassName="text-emerald-500"
                  count={`${draftEnabled ? 'enabled' : 'disabled'} / ${draftPaused ? 'paused' : 'active'}`}
                  defaultOpen={false}
                  subtitle="Lifecycle controls applied when this bot is loaded by the orchestrator."
                >
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="rounded-md border border-border p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Enabled</span>
                        <Switch checked={draftEnabled} onCheckedChange={(checked) => setDraftEnabled(checked)} />
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">Disabled bots are excluded from orchestrator cycles.</p>
                    </div>
                    <div className="rounded-md border border-border p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Paused</span>
                        <Switch checked={draftPaused} onCheckedChange={(checked) => setDraftPaused(checked)} />
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">Paused bots stay loaded but do not execute decisions.</p>
                    </div>
                  </div>
                </FlyoutSection>

                <FlyoutSection
                  title="Signal Gating"
                  icon={ShieldAlert}
                  iconClassName="text-amber-500"
                  count="7 controls"
                  defaultOpen={false}
                >
                  <div className="grid gap-3 md:grid-cols-3">
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
                      Crypto target scope and strategy mode are configured in the Signal Sources section.
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
                  count="5 fields"
                  defaultOpen={false}
                >
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <Label>Resume Policy</Label>
                      <Select
                        value={advancedConfig.resumePolicy}
                        onValueChange={(value) => setAdvancedValue('resumePolicy', normalizeResumePolicy(value))}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="resume_full">Resume Full</SelectItem>
                          <SelectItem value="manage_only">Manage Existing Only</SelectItem>
                          <SelectItem value="flatten_then_start">Flatten Then Start</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
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
                        <SelectItem value="disable">Disable + Pause (Recommended)</SelectItem>
                        <SelectItem value="block">Delete (No Open Positions)</SelectItem>
                        <SelectItem value="force_delete">Force Delete (Danger)</SelectItem>
                      </SelectContent>
                    </Select>
                    {deleteAction === 'force_delete' ? (
                      <div>
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
                      {deleteAction === 'disable' ? 'Disable Bot' : 'Delete Bot'}
                    </Button>
                  </FlyoutSection>
                ) : null}

              </div>
            </ScrollArea>

            <div className="border-t border-border px-4 py-3 flex flex-wrap items-center justify-end gap-2">
              {saveError ? (
                <div className="mr-auto text-xs text-red-500 max-w-[65%] truncate" title={saveError}>
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
                  effectiveDraftSources.length === 0
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
