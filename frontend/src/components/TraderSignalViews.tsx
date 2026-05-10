import { useState, useEffect, useRef, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useAtomValue } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  CheckCircle2,
  Clock,
  ExternalLink,
  MessageCircle,
  ShieldAlert,
  Terminal,
  TrendingUp,
  TrendingDown,
  Users,
  Wallet,
  Maximize2,
  Minimize2,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import {
  buildOutcomeFallbacks,
  buildOutcomeSparklineSeries,
  toTimeValueSeries,
} from '../lib/priceHistory'
import { Liveline } from 'liveline'
import type { LivelineSeries } from 'liveline'
import { Badge } from './ui/badge'
import { CryptoMarketCard } from './CryptoMarketsPanel'
import { themeAtom } from '../store/atoms'
import type { TrackedTraderOpportunity } from '../services/discoveryApi'
import type { Opportunity, CryptoMarket } from '../services/api'
import { getCryptoMarkets } from '../services/api'
import BuyButton from './BuyButton'

// ─── Unified Type ──────────────────────────────────────────

export interface UnifiedTraderSignal {
  id: string
  source: 'confluence' | 'insider'
  strategy_sdk?: string
  market_id: string
  market_question: string
  market_slug?: string | null
  yes_price?: number | null
  no_price?: number | null
  current_yes_price?: number | null
  current_no_price?: number | null
  outcome_labels?: string[]
  outcome_prices?: number[]
  yes_label?: string | null
  no_label?: string | null
  price_history?: Array<Record<string, unknown> | unknown[]>
  direction: 'BUY' | 'SELL' | null
  confidence: number
  wallet_count: number
  tier: 'WATCH' | 'HIGH' | 'EXTREME' | 'INSIDER'

  // Confluence fields
  signal_type?: string
  conviction_score?: number
  window_minutes?: number
  cluster_adjusted_wallet_count?: number
  unique_core_wallets?: number
  net_notional?: number
  top_wallets?: Array<{
    address: string
    username: string | null
    rank_score: number
    composite_score: number
    quality_score: number
    activity_score: number
  }>
  outcome?: string | null

  // Insider fields
  insider_score?: number | null
  edge_percent?: number | null
  cluster_count?: number
  pre_news_lead_minutes?: number
  freshness_minutes?: number
  suggested_size_usd?: number | null
  market_liquidity?: number
  top_wallet?: {
    address: string
    username?: string | null
    insider_score?: number
    insider_confidence?: number
  } | null
  selected_outcome_label?: string | null
  wallets?: Array<{
    address: string
    username?: string | null
    insider_score?: number
    insider_confidence?: number
  }>

  // Timing
  detected_at: string
  last_seen_at?: string | null
  first_seen_at?: string | null

  // Computed
  market_url: string
  source_flags?: {
    from_pool?: boolean
    from_tracked_traders?: boolean
    from_trader_groups?: boolean
    qualified?: boolean
  } | null
  source_breakdown?: {
    wallets_considered?: number
    pool_wallets?: number
    tracked_wallets?: number
    group_wallets?: number
    group_count?: number
    group_ids?: string[]
  } | null
  validation?: {
    is_valid?: boolean
    is_actionable?: boolean
    is_tradeable?: boolean
    checks?: Record<string, boolean>
    reasons?: string[]
  } | null
  is_valid: boolean
  is_actionable: boolean
  is_tradeable: boolean
  validation_reasons: string[]
  source_coverage_score: number
}

// ─── Normalization ─────────────────────────────────────────

function extractOutcomeLabels(raw: unknown): string[] {
  if (!Array.isArray(raw)) return []
  const labels: string[] = []
  for (const item of raw) {
    if (typeof item === 'string') {
      const label = item.trim()
      if (label && !labels.includes(label)) labels.push(label)
      continue
    }
    if (!item || typeof item !== 'object') continue
    const row = item as Record<string, unknown>
    const label = String(
      row.outcome
      ?? row.label
      ?? row.name
      ?? row.title
      ?? row.value
      ?? '',
    ).trim()
    if (label && !labels.includes(label)) labels.push(label)
  }
  return labels
}

function labelsLookLikeYesNo(labels: string[]): boolean {
  const normalized = labels
    .slice(0, 2)
    .map((label) => label.trim().toLowerCase())
    .filter(Boolean)
    .sort()
  return normalized.length === 2 && normalized[0] === 'no' && normalized[1] === 'yes'
}

function inferBinaryOutcomeLabelsFromQuestion(question: string): string[] {
  const text = String(question || '').trim()
  if (!text) return []
  const lowered = text.toLowerCase()
  const splitter = lowered.includes(' vs. ')
    ? ' vs. '
    : lowered.includes(' vs ')
      ? ' vs '
      : ''
  if (!splitter) return []
  const chunks = text.split(splitter).map((chunk) => chunk.trim()).filter(Boolean)
  if (chunks.length < 2) return []
  const left = chunks[0].includes(':')
    ? chunks[0].split(':').slice(-1)[0].trim() || chunks[0]
    : chunks[0]
  return [left, chunks[1]]
}

function resolveBinaryOutcomeLabels(
  sources: unknown[],
  question: string,
): string[] {
  let genericBinaryLabels: string[] = []
  for (const source of sources) {
    const labels = extractOutcomeLabels(source)
    if (labels.length < 2) continue
    const binaryLabels = [labels[0], labels[1]]
    if (!labelsLookLikeYesNo(binaryLabels)) return binaryLabels
    if (genericBinaryLabels.length === 0) genericBinaryLabels = binaryLabels
  }
  const inferredFromQuestion = inferBinaryOutcomeLabelsFromQuestion(question)
  if (inferredFromQuestion.length >= 2) return inferredFromQuestion
  if (genericBinaryLabels.length >= 2) return genericBinaryLabels
  return ['Yes', 'No']
}

function normalizeOutcomeSide(value: unknown): 'YES' | 'NO' | null {
  const normalized = String(value || '').trim().toUpperCase()
  if (normalized === 'YES') return 'YES'
  if (normalized === 'NO') return 'NO'
  return null
}

function getConfluenceDirection(signal: TrackedTraderOpportunity): 'BUY' | 'SELL' | null {
  const outcome = normalizeOutcomeSide(signal.outcome)
  if (outcome === 'YES') return 'BUY'
  if (outcome === 'NO') return 'SELL'
  const signalType = (signal.signal_type || '').toLowerCase()
  if (signalType.includes('sell')) return 'SELL'
  if (signalType.includes('buy') || signalType.includes('accumulation')) return 'BUY'
  return null
}

function toTier(value: string | null | undefined): 'WATCH' | 'HIGH' | 'EXTREME' {
  const normalized = (value || 'WATCH').toString().trim().toUpperCase()
  if (normalized === 'EXTREME') return 'EXTREME'
  if (normalized === 'HIGH') return 'HIGH'
  if (normalized === 'LOW' || normalized === 'MEDIUM' || normalized === 'WATCH') return 'WATCH'
  return 'WATCH'
}

function normalizeSourceCoverageScore(
  sourceFlags: UnifiedTraderSignal['source_flags'],
): number {
  if (!sourceFlags) return 0
  return (
    (sourceFlags.from_pool ? 1 : 0)
    + (sourceFlags.from_tracked_traders ? 1 : 0)
    + (sourceFlags.from_trader_groups ? 1 : 0)
  )
}

function normalizeSignalQualityFlags(signal: {
  source_flags?: UnifiedTraderSignal['source_flags']
  validation?: UnifiedTraderSignal['validation']
  is_valid?: boolean
  is_actionable?: boolean
  is_tradeable?: boolean
  validation_reasons?: string[]
}): {
  sourceFlags: UnifiedTraderSignal['source_flags']
  validation: UnifiedTraderSignal['validation']
  isValid: boolean
  isActionable: boolean
  isTradeable: boolean
  reasons: string[]
  sourceCoverageScore: number
} {
  const sourceFlags = signal.source_flags ?? null
  const validation = signal.validation ?? null
  const isValid = Boolean(
    signal.is_valid
    ?? validation?.is_valid
    ?? (
      validation?.checks?.has_market_id
      && validation?.checks?.has_wallets
      && validation?.checks?.has_direction
      && validation?.checks?.has_price_reference
      && validation?.checks?.price_in_bounds
    ),
  )
  const isActionable = Boolean(
    signal.is_actionable
    ?? validation?.is_actionable
    ?? (isValid && sourceFlags?.qualified),
  )
  const isTradeable = Boolean(
    signal.is_tradeable
    ?? validation?.is_tradeable
    ?? isActionable,
  )
  const reasons = (
    signal.validation_reasons
    || validation?.reasons
    || []
  ).map((reason) => String(reason))
  const sourceCoverageScore = normalizeSourceCoverageScore(sourceFlags)

  return {
    sourceFlags,
    validation,
    isValid,
    isActionable,
    isTradeable,
    reasons,
    sourceCoverageScore,
  }
}

function resolveTraderStrategySdk(candidates: unknown[], fallback = 'traders_confluence'): string {
  for (const candidate of candidates) {
    const value = String(candidate || '').trim().toLowerCase()
    if (value) return value
  }
  return fallback
}

export function normalizeConfluenceSignal(signal: TrackedTraderOpportunity): UnifiedTraderSignal {
  const direction = getConfluenceDirection(signal)
  const quality = normalizeSignalQualityFlags(signal)
  const raw = signal as unknown as Record<string, unknown>
  const outcomeLabels = resolveBinaryOutcomeLabels(
    [
      signal.outcome_labels,
      [signal.yes_label, signal.no_label],
    ],
    signal.market_question || signal.market_id,
  )
  const strategySdk = resolveTraderStrategySdk([
    raw.strategy_sdk,
    raw.strategy_slug,
    raw.strategy_key,
    raw.strategy_type,
  ])
  return {
    id: signal.id,
    source: 'confluence',
    strategy_sdk: strategySdk,
    market_id: signal.market_id,
    market_question: signal.market_question || signal.market_id,
    market_slug: signal.market_slug,
    yes_price: signal.yes_price,
    no_price: signal.no_price,
    current_yes_price: signal.current_yes_price ?? signal.yes_price ?? null,
    current_no_price: signal.current_no_price ?? signal.no_price ?? null,
    outcome_labels: outcomeLabels,
    outcome_prices: signal.outcome_prices,
    yes_label: outcomeLabels[0] ?? null,
    no_label: outcomeLabels[1] ?? null,
    price_history: signal.price_history,
    direction,
    confidence: signal.conviction_score || Math.round(signal.strength * 100) || 0,
    wallet_count: signal.wallet_count,
    tier: toTier(signal.tier),
    signal_type: signal.signal_type,
    conviction_score: signal.conviction_score,
    window_minutes: signal.window_minutes,
    cluster_adjusted_wallet_count: signal.cluster_adjusted_wallet_count,
    unique_core_wallets: signal.unique_core_wallets,
    net_notional: signal.net_notional ?? undefined,
    top_wallets: signal.top_wallets,
    outcome: signal.outcome,
    selected_outcome_label: direction === 'BUY'
      ? outcomeLabels[0] ?? null
      : direction === 'SELL'
        ? outcomeLabels[1] ?? null
        : null,
    detected_at: signal.detected_at,
    last_seen_at: signal.last_seen_at ?? undefined,
    first_seen_at: signal.first_seen_at ?? undefined,
    market_url: buildPolymarketMarketUrl({ eventSlug: signal.market_slug }) || '',
    source_flags: quality.sourceFlags,
    source_breakdown: signal.source_breakdown ?? null,
    validation: quality.validation,
    is_valid: quality.isValid,
    is_actionable: quality.isActionable,
    is_tradeable: quality.isTradeable,
    validation_reasons: quality.reasons,
    source_coverage_score: quality.sourceCoverageScore,
  }
}

function asObject(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>
  return {}
}

function resolveYesPrice(mkt: Record<string, unknown>): number {
  const raw = Number(mkt.current_yes_price ?? mkt.yes_price)
  return Number.isFinite(raw) ? raw : 0
}

export function normalizeTraderOpportunity(opportunity: Opportunity): UnifiedTraderSignal {
  const markets = opportunity.markets || []
  const market = markets[0] || ({} as Record<string, unknown>)
  const isMultiMarket = markets.length > 1
  const strategyContext = asObject(opportunity.strategy_context)
  const firehose = asObject(strategyContext.firehose)
  const sourceFlags = asObject(strategyContext.source_flags)
  const validation = asObject(strategyContext.validation)

  const side = String(strategyContext.side || firehose.side || '').trim().toLowerCase()
  const selectedOutcomeSide = normalizeOutcomeSide(
    strategyContext.outcome
    ?? firehose.outcome
    ?? opportunity.positions_to_take?.[0]?.outcome,
  )
  const direction: 'BUY' | 'SELL' | null = selectedOutcomeSide === 'YES'
    ? 'BUY'
    : selectedOutcomeSide === 'NO'
      ? 'SELL'
      : side === 'buy'
        ? 'BUY'
        : side === 'sell'
          ? 'SELL'
          : null

  const confidenceRatio = Number(strategyContext.confidence ?? opportunity.confidence ?? 0)
  const confidence = Number.isFinite(confidenceRatio) ? Math.max(0, Math.min(100, Math.round(confidenceRatio * 100))) : 0
  const tier = toTier(String(strategyContext.tier || firehose.tier || 'HIGH'))
  const marketSlug = String(market.event_slug || opportunity.event_slug || market.slug || '')
  const validationReasons = Array.isArray(validation.reasons)
    ? validation.reasons.map((reason) => String(reason))
    : []
  const strategySdk = resolveTraderStrategySdk([
    strategyContext.strategy_sdk,
    strategyContext.strategy_slug,
    strategyContext.strategy_key,
    opportunity.strategy,
  ])

  const normalizedSourceFlags: UnifiedTraderSignal['source_flags'] = {
    from_pool: Boolean(sourceFlags.from_pool),
    from_tracked_traders: Boolean(sourceFlags.from_tracked_traders),
    from_trader_groups: Boolean(sourceFlags.from_trader_groups),
    qualified: Boolean(sourceFlags.qualified),
  }

  const normalizedValidation: UnifiedTraderSignal['validation'] = {
    is_valid: validation.is_valid === undefined ? true : Boolean(validation.is_valid),
    is_actionable: validation.is_actionable === undefined ? true : Boolean(validation.is_actionable),
    is_tradeable: validation.is_tradeable === undefined ? true : Boolean(validation.is_tradeable),
    checks: asObject(validation.checks) as Record<string, boolean>,
    reasons: validationReasons,
  }

  // For multi-market opportunities, build outcome labels/prices from all sub-markets
  let outcomeLabels: string[] | undefined
  let outcomePrices: number[] | undefined
  let mergedPriceHistory: Array<Record<string, unknown>> | undefined
  if (isMultiMarket) {
    outcomeLabels = markets.map((mkt, i) => {
      const raw = String((mkt as any).group_item_title || mkt.question || '')
      const label = raw.replace(/^(Will |What will |Which |Who will )/i, '').split('?')[0].trim()
      return label || `Market ${i + 1}`
    })
    outcomePrices = markets.map((mkt) => resolveYesPrice(mkt as unknown as Record<string, unknown>))
    // Merge price histories from all sub-markets
    const maxLen = Math.max(
      ...markets.map((m) => Array.isArray(m.price_history) ? m.price_history.length : 0),
      0,
    )
    if (maxLen >= 2) {
      const merged: Record<string, unknown>[] = []
      for (let j = 0; j < maxLen; j++) {
        const point: Record<string, unknown> = {}
        markets.forEach((m, i) => {
          const hist = Array.isArray(m.price_history) ? m.price_history : []
          const entry = hist[j] as Record<string, unknown> | undefined
          if (entry) {
            const yesVal = entry.yes ?? entry.y ?? entry.p ?? entry.price
            if (yesVal != null) point[`idx_${i}`] = yesVal
            if (i === 0 && entry.t != null) point.t = entry.t
          }
        })
        if (Object.keys(point).length > 1 || (Object.keys(point).length === 1 && !('t' in point))) {
          merged.push(point)
        }
      }
      if (merged.length >= 2) mergedPriceHistory = merged
    }
  } else {
    outcomeLabels = resolveBinaryOutcomeLabels(
      [
        market.outcome_labels,
        market.outcomes,
        market.tokens,
        firehose.outcome_labels,
        firehose.outcomes,
        firehose.tokens,
      ],
      String(market.question || opportunity.title || ''),
    )
    outcomePrices = Array.isArray(market.outcome_prices) ? market.outcome_prices as number[] : undefined
  }

  return {
    id: opportunity.id,
    source: 'confluence',
    strategy_sdk: strategySdk,
    market_id: String(market.id || ''),
    market_question: String(market.question || opportunity.title || ''),
    market_slug: String(market.slug || '') || null,
    yes_price: typeof market.yes_price === 'number' ? market.yes_price : null,
    no_price: typeof market.no_price === 'number' ? market.no_price : null,
    current_yes_price: typeof market.current_yes_price === 'number' ? market.current_yes_price : (typeof market.yes_price === 'number' ? market.yes_price : null),
    current_no_price: typeof market.current_no_price === 'number' ? market.current_no_price : (typeof market.no_price === 'number' ? market.no_price : null),
    outcome_labels: outcomeLabels,
    outcome_prices: outcomePrices,
    yes_label: outcomeLabels?.[0] ?? null,
    no_label: outcomeLabels?.[1] ?? null,
    price_history: mergedPriceHistory ?? (Array.isArray(market.price_history) ? market.price_history : undefined),
    direction,
    confidence,
    wallet_count: Number(strategyContext.wallet_count ?? firehose.wallet_count ?? 0) || 0,
    tier,
    signal_type: String(firehose.signal_type || ''),
    conviction_score: confidence,
    window_minutes: Number(firehose.window_minutes || 60),
    cluster_adjusted_wallet_count: Number(firehose.cluster_adjusted_wallet_count || firehose.wallet_count || 0),
    unique_core_wallets: Number(firehose.unique_core_wallets || 0),
    net_notional: typeof firehose.net_notional === 'number' ? firehose.net_notional : undefined,
    top_wallets: Array.isArray(firehose.top_wallets) ? (firehose.top_wallets as UnifiedTraderSignal['top_wallets']) : undefined,
    outcome: String(strategyContext.outcome || firehose.outcome || '') || null,
    selected_outcome_label: direction === 'BUY'
      ? outcomeLabels?.[0] ?? null
      : direction === 'SELL'
        ? outcomeLabels?.[1] ?? null
        : null,
    detected_at: opportunity.detected_at,
    last_seen_at: opportunity.last_seen_at ?? null,
    first_seen_at: firehose.first_seen_at ? String(firehose.first_seen_at) : null,
    market_url: buildPolymarketMarketUrl({ eventSlug: marketSlug }) || '',
    source_flags: normalizedSourceFlags,
    source_breakdown: asObject(strategyContext.source_breakdown) as UnifiedTraderSignal['source_breakdown'],
    validation: normalizedValidation,
    is_valid: normalizedValidation.is_valid ?? true,
    is_actionable: normalizedValidation.is_actionable ?? true,
    is_tradeable: normalizedValidation.is_tradeable ?? true,
    validation_reasons: validationReasons,
    source_coverage_score: normalizeSourceCoverageScore(normalizedSourceFlags),
    wallets: Array.isArray(firehose.wallets) ? (firehose.wallets as UnifiedTraderSignal['wallets']) : undefined,
  }
}

// ─── Shared Utilities ──────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  WATCH: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  HIGH: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  EXTREME: 'bg-red-500/10 text-red-400 border-red-500/20',
  INSIDER: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
}

const TIER_BORDER_COLORS: Record<string, string> = {
  WATCH: 'border-yellow-500/30',
  HIGH: 'border-orange-500/30',
  EXTREME: 'border-red-500/30',
  INSIDER: 'border-purple-500/30',
}

const ACCENT_BAR_COLORS: Record<string, string> = {
  WATCH: 'bg-yellow-500',
  HIGH: 'bg-orange-500',
  EXTREME: 'bg-red-500',
  INSIDER: 'bg-purple-500',
}

const SOURCE_COLORS: Record<string, string> = {
  confluence: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  insider: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
}

const SPARKLINE_COLORS = [
  '#22c55e',
  '#ef4444',
  '#38bdf8',
  '#f59e0b',
  '#a78bfa',
  '#14b8a6',
  '#f97316',
  '#ec4899',
]

const SPARKLINE_TEXT_CLASSES = [
  'text-green-300',
  'text-red-300',
  'text-sky-300',
  'text-amber-300',
  'text-violet-700 dark:text-violet-300',
  'text-teal-300',
  'text-orange-300',
  'text-pink-300',
]

const OUTCOME_BOX_CLASSES = [
  'border-green-500/20 bg-green-500/10 text-green-300',
  'border-red-500/20 bg-red-500/10 text-red-300',
  'border-sky-500/20 bg-sky-500/10 text-sky-300',
  'border-amber-500/20 bg-amber-500/10 text-amber-300',
  'border-violet-500/20 bg-violet-500/10 text-violet-700 dark:text-violet-300',
  'border-teal-500/20 bg-teal-500/10 text-teal-300',
  'border-orange-500/20 bg-orange-500/10 text-orange-300',
  'border-pink-500/20 bg-pink-500/10 text-pink-300',
]

function timeAgo(dateStr?: string | null): string {
  if (!dateStr) return '\u2014'
  const diffMs = Date.now() - new Date(dateStr).getTime()
  if (diffMs < 0 || Number.isNaN(diffMs)) return 'now'
  const sec = Math.floor(diffMs / 1000)
  if (sec < 60) return `${sec}s`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h`
  return `${Math.floor(hr / 24)}d`
}

function formatCompact(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '$\u2014'
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1000) return `$${(n / 1000).toFixed(1)}K`
  if (n >= 100) return `$${n.toFixed(0)}`
  if (n >= 1) return `$${n.toFixed(2)}`
  return `$${n.toFixed(4)}`
}

function convictionColor(value: number): string {
  if (value >= 80) return 'bg-green-500'
  if (value >= 60) return 'bg-yellow-500'
  return 'bg-red-500'
}

function shortAddress(address: string): string {
  if (!address) return 'unknown'
  if (address.length <= 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function humanizeSlug(slug: string | null | undefined): string {
  const raw = String(slug || '').trim().replace(/_/g, '-')
  if (!raw) return ''
  return raw
    .split('-')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function normalizeMarketLabel(signal: UnifiedTraderSignal): string {
  const question = String(signal.market_question || '').trim()
  const placeholderTail = question.replace(/^market\s+/i, '').trim()
  const looksLikePlaceholder = /^market\s+/i.test(question)
    && (
      /^0x[0-9a-f.]+$/i.test(placeholderTail)
      || /^\d{12,}$/.test(placeholderTail)
    )
  if (question && !looksLikePlaceholder) return question
  const fromSlug = humanizeSlug(signal.market_slug || null)
  if (fromSlug) return fromSlug
  const marketId = String(signal.market_id || '').trim()
  if (/^0x[0-9a-f]+$/i.test(marketId) && marketId.length > 14) {
    return `Market ${marketId.slice(0, 10)}...${marketId.slice(-4)}`
  }
  return question || marketId || 'Unknown market'
}

function sanitizeOutcomeLabel(value: string | null | undefined, fallback: string): string {
  const text = String(value || '').trim()
  return text || fallback
}

function yesOutcomeLabel(signal: UnifiedTraderSignal): string {
  return sanitizeOutcomeLabel(signal.yes_label || signal.outcome_labels?.[0], 'Yes')
}

function noOutcomeLabel(signal: UnifiedTraderSignal): string {
  return sanitizeOutcomeLabel(signal.no_label || signal.outcome_labels?.[1], 'No')
}

function directionOutcomeLabel(signal: UnifiedTraderSignal): string {
  if (signal.direction === 'BUY') return yesOutcomeLabel(signal)
  if (signal.direction === 'SELL') return noOutcomeLabel(signal)
  return sanitizeOutcomeLabel(signal.outcome, 'Signal')
}

function selectedOutcomeIndex(signal: UnifiedTraderSignal): 0 | 1 | null {
  if (signal.direction === 'BUY') return 0
  if (signal.direction === 'SELL') return 1
  return null
}

function isSelectedOutcomeRow(
  signal: UnifiedTraderSignal,
  row: { key: string; label: string },
  index: number,
): boolean {
  const selectedIndex = selectedOutcomeIndex(signal)
  if (selectedIndex === null) return false
  if (row.key === 'yes' || row.key === 'idx_0') return selectedIndex === 0
  if (row.key === 'no' || row.key === 'idx_1') return selectedIndex === 1
  const selectedLabel = selectedIndex === 0 ? yesOutcomeLabel(signal) : noOutcomeLabel(signal)
  return row.label.trim().toLowerCase() === selectedLabel.trim().toLowerCase() || index === selectedIndex
}

function compactOutcomeLabel(label: string, maxChars = 14): string {
  const text = String(label || '').trim()
  if (!text) return '\u2014'
  if (text.length <= maxChars) return text
  return `${text.slice(0, Math.max(1, maxChars - 1))}\u2026`
}

function buildSignalOutcomeFallbacks(
  signal: UnifiedTraderSignal,
  currentYes: number | null | undefined,
  currentNo: number | null | undefined,
) {
  return buildOutcomeFallbacks({
    labels: signal.outcome_labels,
    prices: signal.outcome_prices,
    yesPrice: currentYes,
    noPrice: currentNo,
    yesLabel: signal.yes_label,
    noLabel: signal.no_label,
    preferIndexedKeys: (signal.outcome_labels?.length || 0) > 2,
  })
}

function formatSignalReason(reason: string): string {
  return reason
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatPriceCents(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '\u2014'
  return `${(value * 100).toFixed(0)}\u00a2`
}

// ═══════════════════════════════════════════════════════════
// CARD VIEW
// ═══════════════════════════════════════════════════════════

interface CardProps {
  signals: UnifiedTraderSignal[]
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
}

export function TraderSignalCards({ signals, onNavigateToWallet, onOpenCopilot }: CardProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
      {signals.map((signal) => (
        <TraderSignalCard
          key={signal.id}
          signal={signal}
          onNavigateToWallet={onNavigateToWallet}
          onOpenCopilot={onOpenCopilot}
        />
      ))}
    </div>
  )
}

function getSourceLabel(t: (k: string) => string, source: string): string {
  return source === 'insider' ? t('traderSignalViews.sourceInsider') : t('traderSignalViews.sourceConfluence')
}

function getSourceTagShort(t: (k: string) => string, source: string): string {
  return source === 'insider' ? t('traderSignalViews.sourceInsiderShort') : t('traderSignalViews.sourceConfluenceShort')
}

function getTierLabel(t: (k: string) => string, tier: string): string {
  switch (tier) {
    case 'WATCH': return t('traderSignalViews.tierWatch')
    case 'HIGH': return t('traderSignalViews.tierHigh')
    case 'EXTREME': return t('traderSignalViews.tierExtreme')
    case 'INSIDER': return t('traderSignalViews.tierInsider')
    default: return tier
  }
}

function getTierTagShort(t: (k: string) => string, tier: string): string {
  switch (tier) {
    case 'WATCH': return t('traderSignalViews.tierWatchShort')
    case 'HIGH': return t('traderSignalViews.tierHighShort')
    case 'EXTREME': return t('traderSignalViews.tierExtremeShort')
    case 'INSIDER': return t('traderSignalViews.tierInsiderShort')
    default: return tier.slice(0, 3)
  }
}

function TraderSignalCard({
  signal,
  onNavigateToWallet,
  onOpenCopilot,
  isModalView = false,
  onCloseModal,
}: {
  signal: UnifiedTraderSignal
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
  isModalView?: boolean
  onCloseModal?: () => void
}) {
  const { t } = useTranslation()
  const expanded = isModalView
  const [modalOpen, setModalOpen] = useState(false)
  const themeMode = useAtomValue(themeAtom)
  const isBuy = signal.direction === 'BUY'
  const marketLabel = normalizeMarketLabel(signal)
  const yesLabel = yesOutcomeLabel(signal)
  const noLabel = noOutcomeLabel(signal)
  const directionLabel = directionOutcomeLabel(signal)
  const selectedOutcomeLabel = signal.selected_outcome_label || directionLabel
  const directionLabelCompact = compactOutcomeLabel(directionLabel, 18)
  const currentYes = signal.current_yes_price ?? signal.yes_price
  const currentNo = signal.current_no_price ?? signal.no_price
  const sparkSeries = useMemo(
    () => buildOutcomeSparklineSeries(
      signal.price_history,
      buildSignalOutcomeFallbacks(signal, currentYes, currentNo),
    ),
    [signal, currentYes, currentNo],
  )
  const hasSparkline = sparkSeries.length > 0
  const nowSec = useMemo(() => Math.floor(Date.now() / 1000), [sparkSeries])
  const livelineSeries = useMemo<LivelineSeries[]>(
    () => sparkSeries.map((row, index) => ({
      id: row.key,
      data: toTimeValueSeries(row.data, nowSec),
      value: row.latest ?? row.data[row.data.length - 1] ?? 0,
      color: SPARKLINE_COLORS[index % SPARKLINE_COLORS.length],
      label: row.label,
    })),
    [sparkSeries, nowSec],
  )
  const primaryLivelineData = livelineSeries[0]?.data ?? []
  const primaryLivelineValue = livelineSeries[0]?.value ?? 0
  const livelineWindow = primaryLivelineData.length >= 2
    ? primaryLivelineData[primaryLivelineData.length - 1].time - primaryLivelineData[0].time
    : 60
  const accentBar = ACCENT_BAR_COLORS[signal.tier] || 'bg-yellow-500'

  const poolWallets = signal.source_breakdown?.pool_wallets || 0
  const trackedWallets = signal.source_breakdown?.tracked_wallets || 0
  const groupWallets = signal.source_breakdown?.group_wallets || 0
  const walletsConsidered = signal.source_breakdown?.wallets_considered || signal.wallet_count || 0
  const actionPrice = isBuy
    ? currentYes
    : signal.direction === 'SELL'
      ? currentNo
      : currentYes ?? currentNo
  const sourceCoverageLabel = `${signal.source_coverage_score}/3`
  const strategySdk = String(signal.strategy_sdk || 'traders_confluence').trim().toLowerCase()

  const qualityPillClass = signal.is_tradeable
    ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
    : signal.is_valid
      ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
      : 'bg-red-500/10 text-red-300 border-red-500/30'
  const qualityLabel = signal.is_tradeable
    ? t('traderSignalViews.qualityTradeable')
    : signal.is_valid
      ? t('traderSignalViews.qualitySourceCheck')
      : t('traderSignalViews.qualityInvalid')

  const bgGradient = signal.source === 'insider'
    ? 'from-purple-500/[0.04] via-transparent to-transparent'
    : signal.tier === 'EXTREME'
      ? 'from-red-500/[0.04] via-transparent to-transparent'
      : signal.tier === 'HIGH'
        ? 'from-orange-500/[0.04] via-transparent to-transparent'
        : 'from-yellow-500/[0.03] via-transparent to-transparent'
  const closeModal = () => setModalOpen(false)

  useEffect(() => {
    if (isModalView || !modalOpen) return undefined
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [isModalView, modalOpen])

  useEffect(() => {
    if (isModalView || !modalOpen) return undefined
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeModal()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isModalView, modalOpen])

  return (
    <>
    <div
      className={cn(
        'relative rounded-lg border bg-card/80 overflow-hidden transition-all cursor-pointer',
        !isModalView && 'hover:shadow-md',
        isModalView && 'w-[min(1100px,calc(100vw-2rem))] max-h-[90vh] overflow-hidden rounded-2xl border-border/70 bg-background shadow-[0_40px_120px_rgba(0,0,0,0.55)]',
        TIER_BORDER_COLORS[signal.tier],
      )}
      onClick={!isModalView ? () => setModalOpen(true) : undefined}
    >
      {/* Accent bar */}
      <div className={cn('absolute left-0 top-0 bottom-0 w-1', accentBar)} />

      {/* ── Modal Header Bar ── */}
      {isModalView && (
        <div className="border-b border-border/60 px-4 py-3 flex-shrink-0">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-1.5">
                <h3 className="text-sm font-semibold truncate max-w-[620px]" title={marketLabel}>
                  {marketLabel}
                </h3>
                <Badge variant="outline" className={cn('h-5 px-1.5 text-[10px] font-medium border', SOURCE_COLORS[signal.source])}>
                  {getSourceLabel(t, signal.source)}
                </Badge>
                <Badge variant="outline" className={cn('h-5 px-1.5 text-[10px] font-bold', TIER_COLORS[signal.tier])}>
                  {getTierLabel(t, signal.tier)}
                </Badge>
              </div>
              <p className="mt-1 text-[11px] text-muted-foreground">
                {t('traderSignalViews.buySummary', { outcome: selectedOutcomeLabel, count: signal.wallet_count, confidence: signal.confidence })}
              </p>
            </div>
            <div className="flex items-center gap-1 flex-shrink-0">
              <BuyButton traderSignal={signal} variant="compact" />
              <button
                onClick={(e) => { e.stopPropagation(); onCloseModal?.() }}
                className="inline-flex items-center gap-1 h-7 px-2 text-[11px] rounded-md border border-border/60 bg-background text-foreground hover:bg-muted/60 transition-colors font-medium"
              >
                <Minimize2 className="w-3 h-3" />
                {t('traderSignalViews.close')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Card Body (scrollable in modal) ── */}
      <div className={cn(isModalView && 'max-h-[calc(90vh-72px)] overflow-y-auto')}>

      <div className={cn('bg-gradient-to-r p-4 pl-4', bgGradient)}>
        {/* Row 1: Header badges */}
        <div className="flex items-center gap-1.5 flex-wrap mb-2">
          <Badge
            variant="outline"
            className={cn('text-[10px] font-medium border', SOURCE_COLORS[signal.source])}
          >
            {getSourceLabel(t, signal.source)}
          </Badge>
          <Badge
            variant="outline"
            className={cn('text-[10px] font-medium border', TIER_COLORS[signal.tier])}
          >
            {getTierLabel(t, signal.tier)}
          </Badge>
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] font-medium',
              isBuy
                ? 'bg-green-500/10 text-green-400 border-green-500/20'
                : signal.direction === 'SELL'
                  ? 'bg-red-500/10 text-red-400 border-red-500/20'
                  : 'bg-blue-500/10 text-blue-400 border-blue-500/20',
            )}
          >
            {signal.direction === 'BUY' || signal.direction === 'SELL'
              ? t('traderSignalViews.buyAction', { outcome: directionLabelCompact })
              : signal.outcome || signal.signal_type?.replace(/_/g, ' ') || t('traderSignalViews.signal')}
          </Badge>
          <Badge
            variant="outline"
            className={cn('text-[10px] font-medium border inline-flex items-center gap-1', qualityPillClass)}
          >
            {signal.is_tradeable ? <CheckCircle2 className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
            {qualityLabel}
          </Badge>
          <Badge
            variant="outline"
            className="max-w-[170px] truncate text-[9px] px-1.5 py-0 font-mono border-border/50 bg-muted/25 text-muted-foreground"
            title={t('traderSignalViews.strategySdkTitle', { sdk: strategySdk })}
          >
            {t('traderSignalViews.sdkLabel', { sdk: strategySdk })}
          </Badge>

          <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground/70">
            <Clock className="w-3 h-3" />
            {timeAgo(signal.last_seen_at || signal.detected_at)}
          </div>
        </div>

        {/* Row 2: Market title */}
        <h3 className="font-medium text-sm text-foreground line-clamp-2 mb-3">
          {marketLabel}
        </h3>

        {/* Row 3: source coverage */}
        <div className="mb-3 flex flex-wrap items-center gap-1.5">
          {poolWallets > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
              {t('traderSignalViews.poolBadge', { count: poolWallets })}
            </Badge>
          )}
          {trackedWallets > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-blue-500/30 bg-blue-500/10 text-blue-300">
              {t('traderSignalViews.trackedBadge', { count: trackedWallets })}
            </Badge>
          )}
          {groupWallets > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-amber-500/30 bg-amber-500/10 text-amber-300">
              {t('traderSignalViews.groupsBadge', { count: groupWallets })}
            </Badge>
          )}
          {signal.source_coverage_score === 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-red-500/30 bg-red-500/10 text-red-300">
              {t('traderSignalViews.noQualifiedSource')}
            </Badge>
          )}
        </div>

        {/* Row 4: Liveline chart */}
        {hasSparkline && primaryLivelineData.length >= 2 && (
          <div className="mb-3 w-full">
            <Liveline
              data={primaryLivelineData}
              value={primaryLivelineValue}
              series={livelineSeries.length > 1 ? livelineSeries : undefined}
              color={SPARKLINE_COLORS[0]}
              theme={themeMode}
              window={livelineWindow}
              paused={livelineSeries.length <= 1}
              grid={expanded || isModalView}
              badge={false}
              fill={livelineSeries.length <= 1}
              pulse={false}
              momentum={expanded || isModalView}
              scrub={expanded || isModalView}
              seriesToggleCompact
              lerpSpeed={0.15}
              padding={expanded || isModalView
                ? { top: 6, right: 6, bottom: 6, left: 6 }
                : { top: 4, right: 4, bottom: 4, left: 4 }}
              formatValue={(v) => `${(v * 100).toFixed(0)}¢`}
              style={{ height: expanded || isModalView ? 120 : 56 }}
            />
            <div className="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5 px-0.5 text-[11px] font-data font-bold">
              {sparkSeries.map((row, index) => (
                <span
                  key={`${signal.id}-spark-${row.key}`}
                  className={cn(
                    'whitespace-nowrap',
                    SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length],
                    isSelectedOutcomeRow(signal, row, index) && 'underline decoration-current underline-offset-2',
                  )}
                >
                  {compactOutcomeLabel(row.label, 14)} {row.latest != null && Number.isFinite(row.latest) ? `${(row.latest * 100).toFixed(0)}¢` : '—'}
                </span>
              ))}
            </div>
          </div>
        )}

        {sparkSeries.length > 0 ? (
          <div className={cn('grid gap-2 mb-3', sparkSeries.length <= 2 ? 'grid-cols-2' : sparkSeries.length <= 4 ? 'grid-cols-2 sm:grid-cols-4' : 'grid-cols-3 sm:grid-cols-4')}>
            {sparkSeries.slice(0, 8).map((row, index) => (
              <div
                key={`${signal.id}-outcome-${row.key}`}
                className={cn(
                  'rounded-md border px-2 py-1.5',
                  OUTCOME_BOX_CLASSES[index % OUTCOME_BOX_CLASSES.length],
                  isSelectedOutcomeRow(signal, row, index) && 'ring-1 ring-white/40 shadow-[0_0_0_1px_rgba(255,255,255,0.08)]',
                )}
              >
                {isSelectedOutcomeRow(signal, row, index) && (
                  <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-foreground/75">{t('traderSignalViews.selected')}</p>
                )}
                <p className="text-sm font-bold truncate">{compactOutcomeLabel(row.label, 14)}</p>
                <p className="text-xs font-semibold font-data">
                  {row.latest != null && Number.isFinite(row.latest) ? formatPriceCents(row.latest) : '\u2014'}
                </p>
              </div>
            ))}
            {sparkSeries.length > 8 && (
              <div className="flex items-center justify-center text-xs text-muted-foreground/60">
                +{sparkSeries.length - 8}
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className={cn(
              'rounded-md border border-green-500/20 bg-green-500/10 px-2 py-1.5',
              isBuy && 'ring-1 ring-white/40 shadow-[0_0_0_1px_rgba(255,255,255,0.08)]',
            )}>
              {isBuy && (
                <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-green-200/80">{t('traderSignalViews.selected')}</p>
              )}
              <p className="text-sm font-bold text-green-300 truncate">{yesLabel}</p>
              <p className="text-xs font-semibold text-green-300 font-data">{formatPriceCents(currentYes)}</p>
            </div>
            <div className={cn(
              'rounded-md border border-red-500/20 bg-red-500/10 px-2 py-1.5',
              signal.direction === 'SELL' && 'ring-1 ring-white/40 shadow-[0_0_0_1px_rgba(255,255,255,0.08)]',
            )}>
              {signal.direction === 'SELL' && (
                <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-red-200/80">{t('traderSignalViews.selected')}</p>
              )}
              <p className="text-sm font-bold text-red-300 truncate">{noLabel}</p>
              <p className="text-xs font-semibold text-red-300 font-data">{formatPriceCents(currentNo)}</p>
            </div>
          </div>
        )}

        {/* Row 5: Metrics grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mb-3">
          <div>
            <p className="text-muted-foreground/70">{t('traderSignalViews.selectedSide')}</p>
            <p className="font-semibold text-foreground truncate" title={selectedOutcomeLabel}>
              {selectedOutcomeLabel}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground/70">{t('traderSignalViews.entry')}</p>
            <p className="font-semibold text-foreground font-data">
              {formatPriceCents(actionPrice)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground/70">{t('traderSignalViews.confidence')}</p>
            <p className="font-semibold text-foreground font-data">{signal.confidence}</p>
          </div>
          <div>
            <p className="text-muted-foreground/70">{t('traderSignalViews.wallets')}</p>
            <p className="font-semibold text-foreground font-data flex items-center gap-1">
              <Users className="w-3 h-3 text-muted-foreground/70" />
              {signal.source === 'confluence'
                ? signal.cluster_adjusted_wallet_count || signal.wallet_count
                : signal.wallet_count}
            </p>
          </div>
        </div>

        {/* Row 6: Secondary metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mb-3">
          {signal.source === 'confluence' ? (
            <>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.netNotional')}</p>
                <p className="font-semibold text-foreground font-data">
                  {formatCompact(signal.net_notional)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.window')}</p>
                <p className="font-semibold text-foreground font-data">
                  {signal.window_minutes || 60}m
                </p>
              </div>
            </>
          ) : (
            <>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.edge')}</p>
                <p className="font-semibold text-foreground font-data">
                  {signal.edge_percent != null ? `${signal.edge_percent.toFixed(1)}%` : '\u2014'}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.preNewsLead')}</p>
                <p className="font-semibold text-foreground font-data">
                  {signal.pre_news_lead_minutes != null ? `${signal.pre_news_lead_minutes.toFixed(0)}m` : '\u2014'}
                </p>
              </div>
            </>
          )}
          <div>
            <p className="text-muted-foreground/70">{t('traderSignalViews.sourceFit')}</p>
            <p className="font-semibold text-foreground font-data">{sourceCoverageLabel}</p>
          </div>
          <div>
            <p className="text-muted-foreground/70">{t('traderSignalViews.qualified')}</p>
            <p className={cn('font-semibold font-data', signal.is_tradeable ? 'text-emerald-300' : 'text-red-300')}>
              {signal.is_tradeable ? t('traderSignalViews.yes') : t('traderSignalViews.no')}
            </p>
          </div>
          {isModalView && (
            <div>
              <p className="text-muted-foreground/70">{t('traderSignalViews.walletBase')}</p>
              <p className="font-semibold text-foreground font-data">{walletsConsidered}</p>
            </div>
          )}
        </div>

        {/* Row 7: Confidence bar */}
        <div>
          <div className="flex items-center justify-between text-[11px] mb-1">
            <span className="text-muted-foreground/70">
              {signal.source === 'insider' ? t('traderSignalViews.confidence') : t('traderSignalViews.conviction')}
            </span>
            <span className="text-foreground/90 font-medium font-data">{signal.confidence}/100</span>
          </div>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={cn('h-full rounded-full', convictionColor(signal.confidence))}
              style={{ width: `${Math.max(0, Math.min(100, signal.confidence))}%` }}
            />
          </div>
        </div>

        {/* Row 8: Validation reasons */}
        {!signal.is_tradeable && signal.validation_reasons.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {signal.validation_reasons.slice(0, 3).map((reason) => (
              <span
                key={`${signal.id}-${reason}`}
                className="inline-flex items-center rounded-md border border-red-500/20 bg-red-500/10 px-1.5 py-0.5 text-[10px] text-red-200/90"
              >
                {formatSignalReason(reason)}
              </span>
            ))}
          </div>
        )}

        {/* Row 9: Insider-specific metrics */}
        {signal.source === 'insider' && (
          <div className="flex items-center gap-3 mt-2 text-[11px] text-muted-foreground/70">
            {signal.insider_score != null && (
              <span>{t('traderSignalViews.insiderScore')}: <span className="text-purple-400 font-data">{signal.insider_score.toFixed(2)}</span></span>
            )}
            {signal.freshness_minutes != null && (
              <span>{t('traderSignalViews.freshness')}: <span className="text-foreground/80 font-data">{signal.freshness_minutes.toFixed(0)}m</span></span>
            )}
            {signal.cluster_count != null && (
              <span>{t('traderSignalViews.clusters')}: <span className="text-foreground/80 font-data">{signal.cluster_count}</span></span>
            )}
          </div>
        )}

        {/* Row 10: Wallets */}
        {signal.source === 'confluence' && signal.top_wallets && signal.top_wallets.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {signal.top_wallets.slice(0, 4).map((wallet) => (
              <button
                key={`${signal.id}-${wallet.address}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onNavigateToWallet?.(wallet.address)
                }}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted/70 hover:bg-muted transition-colors text-[11px] text-foreground/85"
              >
                <Wallet className="w-3 h-3 text-muted-foreground/70" />
                <span>{wallet.username || shortAddress(wallet.address)}</span>
                <span className="text-muted-foreground/70">
                  {(wallet.composite_score * 100).toFixed(0)}
                </span>
              </button>
            ))}
          </div>
        )}

        {signal.source === 'insider' && signal.top_wallet?.address && (
          <div className="mt-3">
            <button
              onClick={(e) => {
                e.stopPropagation()
                onNavigateToWallet?.(signal.top_wallet!.address)
              }}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-purple-500/10 hover:bg-purple-500/20 transition-colors text-[11px] text-purple-300"
            >
              <Wallet className="w-3 h-3" />
              {signal.top_wallet.username || shortAddress(signal.top_wallet.address)}
              {signal.top_wallet.insider_score != null && (
                <span className="text-purple-400/70 ml-1">IS:{signal.top_wallet.insider_score.toFixed(2)}</span>
              )}
            </button>
          </div>
        )}

        {/* Row 11: Actions */}
        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border/50">
          {signal.market_url && (
            <a
              href={signal.market_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
            >
              <ExternalLink className="w-3 h-3" />
              {t('traderSignalViews.market')}
            </a>
          )}
          {onOpenCopilot && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onOpenCopilot(signal)
              }}
              className="inline-flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300"
            >
              <MessageCircle className="w-3 h-3" />
              {t('traderSignalViews.copilot')}
            </button>
          )}
          <BuyButton traderSignal={signal} variant="inline" />
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="mt-3 pt-3 border-t border-border/50 space-y-2 text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.poolWallets')}</p>
                <p className="font-semibold text-foreground">{poolWallets}</p>
              </div>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.trackedWallets')}</p>
                <p className="font-semibold text-foreground">{trackedWallets}</p>
              </div>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.groupWallets')}</p>
                <p className="font-semibold text-foreground">{groupWallets}</p>
              </div>
              <div>
                <p className="text-muted-foreground/70">{t('traderSignalViews.coverage')}</p>
                <p className="font-semibold text-foreground">{sourceCoverageLabel}</p>
              </div>
            </div>

            {signal.source === 'confluence' && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.coreWallets')}</p>
                    <p className="font-semibold text-foreground">{signal.unique_core_wallets || 0}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.signalType')}</p>
                    <p className="font-semibold text-foreground">{signal.signal_type?.replace(/_/g, ' ') || '\u2014'}</p>
                  </div>
                </div>
              </>
            )}
            {signal.source === 'insider' && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.suggestedSize')}</p>
                    <p className="font-semibold text-foreground">
                      {signal.suggested_size_usd != null ? formatCompact(signal.suggested_size_usd) : '\u2014'}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.marketLiquidity')}</p>
                    <p className="font-semibold text-foreground">
                      {signal.market_liquidity != null ? formatCompact(signal.market_liquidity) : '\u2014'}
                    </p>
                  </div>
                </div>
                {signal.wallets && signal.wallets.length > 1 && (
                  <div>
                    <p className="text-muted-foreground/70 mb-1">{t('traderSignalViews.involvedWallets')}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {signal.wallets.slice(0, 6).map((w) => (
                        <button
                          key={w.address}
                          onClick={(e) => {
                            e.stopPropagation()
                            onNavigateToWallet?.(w.address)
                          }}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-muted/70 hover:bg-muted text-[11px] text-foreground/80"
                        >
                          <Wallet className="w-3 h-3 text-muted-foreground/70" />
                          {w.username || shortAddress(w.address)}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {signal.validation_reasons.length > 0 && (
              <div>
                <p className="text-muted-foreground/70 mb-1">{t('traderSignalViews.validation')}</p>
                <div className="flex flex-wrap gap-1.5">
                  {signal.validation_reasons.map((reason) => (
                    <span
                      key={`${signal.id}-validation-${reason}`}
                      className="inline-flex items-center rounded-md border border-border/60 bg-muted/50 px-1.5 py-0.5 text-[10px] text-foreground/80"
                    >
                      {formatSignalReason(reason)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="flex flex-wrap gap-4 text-[11px] text-muted-foreground/70 pt-1">
              <span className="inline-flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {t('traderSignalViews.first')}: {timeAgo(signal.first_seen_at || signal.detected_at)}
              </span>
              <span className="inline-flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {t('traderSignalViews.last')}: {timeAgo(signal.last_seen_at || signal.detected_at)}
              </span>
              <span className="inline-flex items-center gap-1">
                <Users className="w-3 h-3" />
                {t('traderSignalViews.walletsCount', { count: signal.wallet_count })}
              </span>
            </div>

          </div>
        )}
      </div>

      </div>
    </div>
      {!isModalView && typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {modalOpen && (
            <motion.div
              key={`trader-signal-modal-${signal.id}`}
              className="fixed inset-0 z-[120] flex items-center justify-center p-4 sm:p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={closeModal}
                aria-hidden
              />
              <motion.div
                className="relative z-10"
                role="dialog"
                aria-modal="true"
                aria-label={t('traderSignalViews.expandedSignalAria', { market: marketLabel })}
                initial={{ scale: 0.94, opacity: 0, y: 22 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 14 }}
                transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}
              >
                <TraderSignalCard
                  signal={signal}
                  onNavigateToWallet={onNavigateToWallet}
                  onOpenCopilot={onOpenCopilot}
                  isModalView
                  onCloseModal={closeModal}
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </>
  )
}

// ═══════════════════════════════════════════════════════════
// TABLE VIEW
// ═══════════════════════════════════════════════════════════

interface TableProps {
  signals: UnifiedTraderSignal[]
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
}

export function TraderSignalTable({ signals, onNavigateToWallet, onOpenCopilot }: TableProps) {
  const { t } = useTranslation()
  const themeMode = useAtomValue(themeAtom)
  const [modalMarket, setModalMarket] = useState<CryptoMarket | null>(null)
  const closeModal = () => setModalMarket(null)

  const { data: cryptoMarkets } = useQuery({
    queryKey: ['crypto-markets-signals'],
    queryFn: () => getCryptoMarkets(),
    refetchInterval: 5000,
    staleTime: 3000,
  })

  const cryptoMarketsMap = useMemo(() => {
    const map = new Map<string, CryptoMarket>()
    if (cryptoMarkets) {
      for (const m of cryptoMarkets) map.set(m.id, m)
    }
    return map
  }, [cryptoMarkets])

  // Body scroll lock + escape key
  useEffect(() => {
    if (!modalMarket) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeModal() }
    window.addEventListener('keydown', onKey)
    return () => { document.body.style.overflow = prev; window.removeEventListener('keydown', onKey) }
  }, [modalMarket])

  return (
    <>
    <div className="border border-border/50 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="grid grid-cols-[44px_36px_minmax(0,1fr)_108px_60px_56px_64px_72px_52px_28px] gap-0 bg-muted/50 border-b border-border/50 text-[9px] text-muted-foreground uppercase tracking-wider font-medium">
        <div className="px-2 py-2">{t('traderSignalViews.colType')}</div>
        <div className="px-2 py-2">{t('traderSignalViews.colTier')}</div>
        <div className="px-2 py-2">{t('traderSignalViews.colMarket')}</div>
        <div className="px-2 py-2 text-center">{t('traderSignalViews.colSide')}</div>
        <div className="px-2 py-2 text-right">{t('traderSignalViews.colConf')}</div>
        <div className="px-2 py-2 text-right">{t('traderSignalViews.colWallets')}</div>
        <div className="px-2 py-2 text-right">{t('traderSignalViews.colEdgeNet')}</div>
        <div className="px-2 py-2 text-center">{t('traderSignalViews.colScore')}</div>
        <div className="px-2 py-2 text-right">{t('traderSignalViews.colAge')}</div>
        <div className="px-1 py-2" />
      </div>

      {/* Body */}
      <div className="divide-y divide-border/30">
        {signals.map((signal) => (
          <TraderSignalTableRow
            key={signal.id}
            signal={signal}
            onNavigateToWallet={onNavigateToWallet}
            onOpenCopilot={onOpenCopilot}
            cryptoMarket={cryptoMarketsMap.get(signal.market_id)}
            onOpenMarketModal={setModalMarket}
          />
        ))}
      </div>
    </div>

    {/* Market modal portal */}
    {typeof document !== 'undefined' && createPortal(
      <AnimatePresence>
        {modalMarket && (
          <motion.div
            key={`signal-market-modal-${modalMarket.id}`}
            className="fixed inset-0 z-[120] flex items-center justify-center p-4 sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={closeModal}
              aria-hidden
            />
            <motion.div
              className="relative z-10"
              role="dialog"
              aria-modal="true"
              initial={{ scale: 0.94, opacity: 0, y: 22 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.97, opacity: 0, y: 14 }}
              transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}
            >
              <CryptoMarketCard
                market={modalMarket}
                themeMode={themeMode}
                nowMs={Date.now()}
                isModalView
                onCloseModal={closeModal}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>,
      document.body
    )}
    </>
  )
}

function TraderSignalTableRow({
  signal,
  onNavigateToWallet,
  onOpenCopilot,
  cryptoMarket,
  onOpenMarketModal,
}: {
  signal: UnifiedTraderSignal
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
  cryptoMarket?: CryptoMarket
  onOpenMarketModal?: (market: CryptoMarket) => void
}) {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)
  const isBuy = signal.direction === 'BUY'
  const marketLabel = normalizeMarketLabel(signal)
  const directionLabel = compactOutcomeLabel(directionOutcomeLabel(signal), 14)

  const confidenceColor = signal.confidence >= 80
    ? 'text-green-400'
    : signal.confidence >= 60
      ? 'text-yellow-400'
      : 'text-red-400'

  const edgeOrNet = signal.source === 'insider'
    ? (signal.edge_percent != null ? `${signal.edge_percent.toFixed(1)}%` : '\u2014')
    : formatCompact(signal.net_notional)

  const scoreValue = signal.source === 'insider'
    ? signal.insider_score
    : null
  const scoreDisplay = scoreValue != null ? scoreValue.toFixed(2) : '\u2014'

  return (
    <>
      <div
        className={cn(
          'grid grid-cols-[44px_36px_minmax(0,1fr)_108px_60px_56px_64px_72px_52px_28px] gap-0 cursor-pointer transition-colors',
          expanded ? 'bg-muted/30' : 'hover:bg-muted/20',
        )}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Type */}
        <div className="px-2 py-2.5">
          <span
            className={cn(
              'inline-block px-1 py-0.5 rounded text-[9px] font-bold border leading-none',
              SOURCE_COLORS[signal.source],
            )}
          >
            {getSourceTagShort(t, signal.source)}
          </span>
        </div>

        {/* Tier */}
        <div className="px-2 py-2.5">
          <span
            className={cn(
              'inline-block px-1 py-0.5 rounded text-[9px] font-bold border leading-none',
              TIER_COLORS[signal.tier],
            )}
          >
            {getTierTagShort(t, signal.tier)}
          </span>
        </div>

        {/* Market */}
        <div className="px-2 py-2.5 min-w-0">
          <p className="text-xs text-foreground truncate">{marketLabel}</p>
        </div>

        {/* Direction */}
        <div className="px-2 py-2.5 flex justify-center">
          <span
            title={directionOutcomeLabel(signal)}
            className={cn(
              'inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium',
              isBuy
                ? 'bg-green-500/10 text-green-400'
                : signal.direction === 'SELL'
                  ? 'bg-red-500/10 text-red-400'
                  : 'bg-blue-500/10 text-blue-400',
            )}
          >
            {isBuy ? <TrendingUp className="w-2.5 h-2.5" /> : signal.direction === 'SELL' ? <TrendingDown className="w-2.5 h-2.5" /> : null}
            {signal.direction === 'BUY' || signal.direction === 'SELL'
              ? directionLabel
              : '\u2014'}
          </span>
        </div>

        {/* Confidence */}
        <div className="px-2 py-2.5 text-right">
          <span className={cn('text-xs font-data font-bold', confidenceColor)}>
            {signal.confidence}
          </span>
        </div>

        {/* Wallets */}
        <div className="px-2 py-2.5 text-right text-xs text-foreground font-data">
          {signal.source === 'confluence'
            ? signal.cluster_adjusted_wallet_count || signal.wallet_count
            : signal.wallet_count}
        </div>

        {/* Edge / Net Notional */}
        <div className="px-2 py-2.5 text-right text-xs text-foreground font-data">
          {edgeOrNet}
        </div>

        {/* Score */}
        <div className="px-2 py-2.5 text-center">
          {signal.source === 'insider' && scoreValue != null ? (
            <div className="flex items-center justify-center gap-1">
              <div className="w-8 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-400 rounded-full"
                  style={{ width: `${Math.min(100, scoreValue * 100)}%` }}
                />
              </div>
              <span className="text-[10px] text-purple-400 font-data">{scoreDisplay}</span>
            </div>
          ) : signal.source === 'confluence' ? (
            <div className="flex items-center justify-center gap-1">
              <div className="w-8 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className={cn('h-full rounded-full', convictionColor(signal.confidence))}
                  style={{ width: `${Math.min(100, signal.confidence)}%` }}
                />
              </div>
              <span className="text-[10px] text-foreground/70 font-data">{signal.confidence}</span>
            </div>
          ) : (
            <span className="text-[10px] text-muted-foreground/40">\u2014</span>
          )}
        </div>

        {/* Age */}
        <div className="px-2 py-2.5 text-right text-[10px] text-muted-foreground/70 font-data">
          {timeAgo(signal.last_seen_at || signal.detected_at)}
        </div>

        {/* Expand market */}
        <div className="px-1 py-2.5 flex items-center justify-center">
          {cryptoMarket && (
            <button
              onClick={(e) => { e.stopPropagation(); onOpenMarketModal?.(cryptoMarket) }}
              className="inline-flex h-5 w-5 items-center justify-center rounded border border-border/40 text-muted-foreground/60 hover:text-foreground hover:border-border transition-colors"
              title={t('traderSignalViews.viewLiveMarket')}
            >
              <Maximize2 className="w-2.5 h-2.5" />
            </button>
          )}
        </div>
      </div>

      {/* Expanded row */}
      {expanded && (
        <div className="grid grid-cols-1 bg-background/30 border-t border-border/30 px-4 py-3">
          <div className="flex gap-6">
            {/* Left column: Details */}
            <div className="flex-1 space-y-2">
              <p className="text-sm text-foreground">{marketLabel}</p>

              {signal.source === 'confluence' && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.coreWallets')}</p>
                    <p className="font-semibold">{signal.unique_core_wallets || 0}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.signalType')}</p>
                    <p className="font-semibold">{signal.signal_type?.replace(/_/g, ' ') || '\u2014'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.window')}</p>
                    <p className="font-semibold">{signal.window_minutes || 60}m</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.lastReinforced')}</p>
                    <p className="font-semibold">{timeAgo(signal.last_seen_at || signal.detected_at)}</p>
                  </div>
                </div>
              )}

              {signal.source === 'insider' && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.insiderScore')}</p>
                    <p className="font-semibold text-purple-400">{signal.insider_score?.toFixed(2) || '\u2014'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.preNewsLead')}</p>
                    <p className="font-semibold">{signal.pre_news_lead_minutes?.toFixed(0) || '\u2014'}m</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.clusters')}</p>
                    <p className="font-semibold">{signal.cluster_count || '\u2014'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">{t('traderSignalViews.freshness')}</p>
                    <p className="font-semibold">{signal.freshness_minutes?.toFixed(0) || '\u2014'}m</p>
                  </div>
                </div>
              )}

              {/* Wallets */}
              {signal.source === 'confluence' && signal.top_wallets && signal.top_wallets.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {signal.top_wallets.slice(0, 6).map((w) => (
                    <button
                      key={w.address}
                      onClick={(e) => {
                        e.stopPropagation()
                        onNavigateToWallet?.(w.address)
                      }}
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-muted/70 hover:bg-muted text-[11px] text-foreground/80"
                    >
                      <Wallet className="w-3 h-3 text-muted-foreground/70" />
                      {w.username || shortAddress(w.address)}
                      <span className="text-muted-foreground/60">{(w.composite_score * 100).toFixed(0)}</span>
                    </button>
                  ))}
                </div>
              )}

              {signal.source === 'insider' && signal.top_wallet?.address && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onNavigateToWallet?.(signal.top_wallet!.address)
                  }}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded bg-purple-500/10 hover:bg-purple-500/20 text-[11px] text-purple-300"
                >
                  <Wallet className="w-3 h-3" />
                  {signal.top_wallet.username || shortAddress(signal.top_wallet.address)}
                </button>
              )}
            </div>

            {/* Right column: Actions */}
            <div className="flex flex-col gap-2 items-end shrink-0">
              {signal.market_url && (
                <a
                  href={signal.market_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-blue-500/15 text-xs text-blue-300 hover:bg-blue-500/25"
                >
                  <ExternalLink className="w-3 h-3" />
                  {t('traderSignalViews.openMarket')}
                </a>
              )}
              {onOpenCopilot && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onOpenCopilot(signal)
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-purple-500/15 text-xs text-purple-300 hover:bg-purple-500/25"
                >
                  <MessageCircle className="w-3 h-3" />
                  {t('traderSignalViews.copilot')}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

// ═══════════════════════════════════════════════════════════
// TERMINAL VIEW
// ═══════════════════════════════════════════════════════════

interface TerminalProps {
  signals: UnifiedTraderSignal[]
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
  isConnected?: boolean
  totalCount?: number
}

export function TraderSignalTerminal({
  signals,
  onNavigateToWallet,
  onOpenCopilot,
  isConnected,
  totalCount,
}: TerminalProps) {
  const { t } = useTranslation()
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const [cursorVisible, setCursorVisible] = useState(true)
  useEffect(() => {
    const iv = setInterval(() => setCursorVisible((v) => !v), 530)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="terminal-view terminal-surface border rounded-lg overflow-hidden font-data text-[11px] leading-relaxed">
      {/* Terminal Header */}
      <div className="terminal-header flex items-center justify-between px-3 py-1.5">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-green-400" />
          <span className="text-green-400 font-bold text-xs">{t('traderSignalViews.terminalTitle')}</span>
          <span className="text-green-400/40">v2.0</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-green-400/60">
            {t('traderSignalViews.signalsCount', { count: totalCount ?? signals.length })}
          </span>
          <div className="flex items-center gap-1">
            <div
              className={cn(
                'w-1.5 h-1.5 rounded-full',
                isConnected ? 'bg-green-400 live-dot' : 'bg-red-400',
              )}
            />
            <span
              className={cn(
                'text-[10px]',
                isConnected ? 'text-green-400/70' : 'text-red-400/70',
              )}
            >
              {isConnected ? t('traderSignalViews.live') : t('traderSignalViews.disconnected')}
            </span>
          </div>
        </div>
      </div>

      {/* Terminal Body */}
      <div ref={scrollRef} className="max-h-[calc(100vh-280px)] overflow-y-auto p-3 space-y-0">
        {/* Boot sequence */}
        <div className="text-green-500/30 mb-3 space-y-0.5">
          <p>{'>'} {t('traderSignalViews.bootInitializing')}</p>
          <p>{'>'} {t('traderSignalViews.bootConnected')}</p>
          <p>{'>'} {t('traderSignalViews.bootLoaded', { count: signals.length })}</p>
          <p className="text-green-500/15">{'\u2500'.repeat(72)}</p>
        </div>

        {/* Signals */}
        {signals.map((signal, idx) => (
          <TerminalSignalEntry
            key={signal.id}
            signal={signal}
            isSelected={selectedIdx === idx}
            onSelect={() => setSelectedIdx(selectedIdx === idx ? null : idx)}
            onNavigateToWallet={onNavigateToWallet}
            onOpenCopilot={onOpenCopilot}
          />
        ))}

        {/* Cursor line */}
        <div className="text-green-400/60 mt-2 flex items-center">
          <span className="text-green-400/30">{'>'} </span>
          <span className="text-green-400/40">{t('traderSignalViews.awaitingNextSignal')}</span>
          <span
            className={cn(
              'inline-block w-2 h-3.5 bg-green-400/60 ml-1 -mb-0.5',
              cursorVisible ? 'opacity-100' : 'opacity-0',
            )}
          />
        </div>
      </div>
    </div>
  )
}

function TerminalSignalEntry({
  signal,
  isSelected,
  onSelect,
  onNavigateToWallet,
  onOpenCopilot,
}: {
  signal: UnifiedTraderSignal
  isSelected: boolean
  onSelect: () => void
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
}) {
  const { t } = useTranslation()
  const sourceTag = getSourceTagShort(t, signal.source)
  const tierTag = signal.tier === 'INSIDER' ? t('traderSignalViews.tierInsiderShort') : signal.tier
  const isBuy = signal.direction === 'BUY'
  const marketLabel = normalizeMarketLabel(signal)
  const directionLabel = compactOutcomeLabel(directionOutcomeLabel(signal), 18)
    .toUpperCase()
    .replace(/\s+/g, '_')

  const tierColor = signal.tier === 'EXTREME'
    ? 'text-red-400'
    : signal.tier === 'HIGH'
      ? 'text-orange-400'
      : signal.tier === 'INSIDER'
        ? 'text-purple-400'
        : 'text-yellow-400'

  const dirColor = isBuy
    ? 'text-green-400'
    : signal.direction === 'SELL'
      ? 'text-red-400'
      : 'text-blue-400'

  const confColor = signal.confidence >= 80
    ? 'text-green-400'
    : signal.confidence >= 60
      ? 'text-yellow-400'
      : 'text-red-400'

  return (
    <div
      className={cn(
        'cursor-pointer transition-colors rounded px-2 py-1 -mx-2',
        isSelected ? 'bg-green-500/[0.06]' : 'hover:bg-green-500/[0.03]',
      )}
      onClick={onSelect}
    >
      {/* Main line */}
      <div className="flex items-center gap-0">
        <span className="text-green-500/30 mr-1">{'>'}</span>
        <span className={cn('mr-2', signal.source === 'insider' ? 'text-purple-400' : 'text-cyan-400')}>
          [{sourceTag}]
        </span>
        <span className={cn('mr-2', tierColor)}>
          [{tierTag}]
        </span>
        <span className={cn('font-bold mr-2', confColor)}>
          CONF:{signal.confidence}
        </span>
        <span className="text-green-300/80 mr-2">
          WLTS:{signal.source === 'confluence' ? signal.cluster_adjusted_wallet_count || signal.wallet_count : signal.wallet_count}
        </span>
        <span className={cn('font-bold mr-2', dirColor)}>
          {signal.direction === 'BUY' || signal.direction === 'SELL'
            ? `BUY_${directionLabel}`
            : signal.outcome || t('traderSignalViews.signal')}
        </span>
        {signal.source === 'insider' && signal.edge_percent != null && (
          <span className="text-purple-300/80 mr-2">EDGE:{signal.edge_percent.toFixed(1)}%</span>
        )}
        {signal.source === 'confluence' && signal.net_notional != null && (
          <span className="text-green-300/60 mr-2">NET:{formatCompact(signal.net_notional)}</span>
        )}
        <span className="text-green-500/25 ml-auto">{timeAgo(signal.detected_at)} ago</span>
      </div>

      {/* Title */}
      <div className="text-green-100/70 pl-4 truncate">
        &quot;{marketLabel}&quot;
      </div>

      {/* Metrics line */}
      <div className="text-green-400/40 pl-4">
        {signal.source === 'confluence' ? (
          <>
            CORE:{signal.unique_core_wallets || 0}
            {' | '}WIN:{signal.window_minutes || 60}m
            {signal.net_notional != null && <>{' | '}NET:{formatCompact(signal.net_notional)}</>}
            {signal.signal_type && <>{' | '}SIG:{signal.signal_type.replace(/_/g, ' ').toUpperCase()}</>}
          </>
        ) : (
          <>
            IS:{signal.insider_score?.toFixed(2) || '\u2014'}
            {' | '}EDGE:{signal.edge_percent?.toFixed(1) || '\u2014'}%
            {' | '}CLUST:{signal.cluster_count || '\u2014'}
            {signal.pre_news_lead_minutes != null && <>{' | '}PRE_NEWS:{signal.pre_news_lead_minutes.toFixed(0)}m</>}
            {signal.freshness_minutes != null && <>{' | '}FRESH:{signal.freshness_minutes.toFixed(0)}m</>}
          </>
        )}
      </div>

      {/* Expanded */}
      {isSelected && (
        <div className="pl-4 mt-1 space-y-1 border-l-2 border-green-500/20 ml-1">
          {signal.source === 'confluence' && signal.top_wallets && signal.top_wallets.length > 0 && (
            <div className="text-green-400/50">
              WALLETS:{' '}
              {signal.top_wallets.slice(0, 6).map((w, i) => (
                <span key={w.address}>
                  {i > 0 && ' | '}
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onNavigateToWallet?.(w.address)
                    }}
                    className="text-orange-400/70 hover:text-orange-400 underline underline-offset-2"
                  >
                    {w.username || shortAddress(w.address)}
                  </button>
                  <span className="text-green-400/30">({(w.composite_score * 100).toFixed(0)})</span>
                </span>
              ))}
            </div>
          )}

          {signal.source === 'insider' && signal.top_wallet?.address && (
            <div className="text-purple-400/50">
              TOP_WALLET:{' '}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onNavigateToWallet?.(signal.top_wallet!.address)
                }}
                className="text-purple-400/70 hover:text-purple-400 underline underline-offset-2"
              >
                {signal.top_wallet.username || shortAddress(signal.top_wallet.address)}
              </button>
              {signal.top_wallet.insider_score != null && (
                <span className="text-purple-400/30"> IS:{signal.top_wallet.insider_score.toFixed(2)}</span>
              )}
            </div>
          )}

          {signal.source === 'insider' && (
            <div className="text-purple-300/40 text-[10px]">
              DETAIL: confidence={signal.confidence}% edge={signal.edge_percent?.toFixed(1) || '\u2014'}%
              {' '}wallets={signal.wallet_count} clusters={signal.cluster_count || '\u2014'}
              {signal.suggested_size_usd != null && <> size={formatCompact(signal.suggested_size_usd)}</>}
              {signal.market_liquidity != null && <> liq={formatCompact(signal.market_liquidity)}</>}
            </div>
          )}

          {signal.source === 'confluence' && (
            <div className="text-green-300/40 text-[10px]">
              DETAIL: conviction={signal.confidence} tier={signal.tier}
              {' '}adj_wallets={signal.cluster_adjusted_wallet_count || signal.wallet_count}
              {' '}core={signal.unique_core_wallets || 0}
              {signal.net_notional != null && <> net={formatCompact(signal.net_notional)}</>}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 pt-1">
            {signal.market_url && (
              <a
                href={signal.market_url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-[10px] text-blue-400/70 hover:text-blue-400 transition-colors underline underline-offset-2"
              >
                [polymarket]
              </a>
            )}
            {onOpenCopilot && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onOpenCopilot(signal)
                }}
                className="text-[10px] text-purple-400/70 hover:text-purple-400 transition-colors underline underline-offset-2"
              >
                [copilot]
              </button>
            )}
            {signal.source === 'insider' && signal.top_wallet?.address && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onNavigateToWallet?.(signal.top_wallet!.address)
                }}
                className="text-[10px] text-purple-400/70 hover:text-purple-400 transition-colors underline underline-offset-2"
              >
                [wallet]
              </button>
            )}
          </div>

          <div className="text-green-500/15">{'\u2500'.repeat(72)}</div>
        </div>
      )}

      {/* Separator */}
      {!isSelected && <div className="text-green-500/10 mt-0.5">{'\u2500'.repeat(72)}</div>}
    </div>
  )
}
