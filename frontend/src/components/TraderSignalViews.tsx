import { useState, useEffect, useRef, useMemo } from 'react'
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  ExternalLink,
  MessageCircle,
  ShieldAlert,
  Terminal,
  TrendingUp,
  TrendingDown,
  Users,
  Wallet,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import {
  buildOutcomeFallbacks,
  buildOutcomeSparklineSeries,
} from '../lib/priceHistory'
import { Badge } from './ui/badge'
import Sparkline from './Sparkline'
import type { InsiderOpportunity, TrackedTraderOpportunity } from '../services/discoveryApi'

// ─── Unified Type ──────────────────────────────────────────

export interface UnifiedTraderSignal {
  id: string
  source: 'confluence' | 'insider'
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

function getConfluenceDirection(signal: TrackedTraderOpportunity): 'BUY' | 'SELL' | null {
  const outcome = (signal.outcome || '').toUpperCase()
  if (outcome === 'YES') return 'BUY'
  if (outcome === 'NO') return 'SELL'
  const signalType = (signal.signal_type || '').toLowerCase()
  if (signalType.includes('sell')) return 'SELL'
  if (signalType.includes('buy') || signalType.includes('accumulation')) return 'BUY'
  return null
}

function toTier(value: string | null | undefined): 'WATCH' | 'HIGH' | 'EXTREME' {
  const normalized = (value || 'WATCH').toUpperCase()
  if (normalized === 'EXTREME') return 'EXTREME'
  if (normalized === 'HIGH') return 'HIGH'
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

export function normalizeConfluenceSignal(signal: TrackedTraderOpportunity): UnifiedTraderSignal {
  const direction = getConfluenceDirection(signal)
  const quality = normalizeSignalQualityFlags(signal)
  return {
    id: signal.id,
    source: 'confluence',
    market_id: signal.market_id,
    market_question: signal.market_question || signal.market_id,
    market_slug: signal.market_slug,
    yes_price: signal.yes_price,
    no_price: signal.no_price,
    current_yes_price: signal.current_yes_price ?? signal.yes_price ?? null,
    current_no_price: signal.current_no_price ?? signal.no_price ?? null,
    outcome_labels: signal.outcome_labels,
    outcome_prices: signal.outcome_prices,
    yes_label: signal.yes_label ?? signal.outcome_labels?.[0] ?? null,
    no_label: signal.no_label ?? signal.outcome_labels?.[1] ?? null,
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

export function normalizeInsiderSignal(opp: InsiderOpportunity): UnifiedTraderSignal {
  const isYes = (opp.direction || '').toLowerCase() === 'buy_yes'
  const quality = normalizeSignalQualityFlags(opp)
  return {
    id: opp.id,
    source: 'insider',
    market_id: opp.market_id,
    market_question: opp.market_question || opp.market_id,
    market_slug: opp.market_slug,
    yes_price: opp.yes_price,
    no_price: opp.no_price,
    current_yes_price: opp.current_yes_price ?? opp.yes_price ?? null,
    current_no_price: opp.current_no_price ?? opp.no_price ?? null,
    outcome_labels: opp.outcome_labels,
    outcome_prices: opp.outcome_prices,
    yes_label: opp.yes_label ?? opp.outcome_labels?.[0] ?? null,
    no_label: opp.no_label ?? opp.outcome_labels?.[1] ?? null,
    price_history: opp.price_history,
    direction: isYes ? 'BUY' : 'SELL',
    confidence: Math.round((opp.confidence || 0) * 100),
    wallet_count: opp.wallet_count,
    tier: 'INSIDER' as const,
    insider_score: opp.insider_score,
    edge_percent: opp.edge_percent,
    cluster_count: opp.cluster_count,
    pre_news_lead_minutes: opp.pre_news_lead_minutes,
    freshness_minutes: opp.freshness_minutes,
    suggested_size_usd: opp.suggested_size_usd,
    market_liquidity: opp.market_liquidity,
    top_wallet: opp.top_wallet,
    wallets: opp.wallets,
    detected_at: opp.created_at || new Date().toISOString(),
    market_url: buildPolymarketMarketUrl({ marketId: opp.market_id }) || '',
    source_flags: quality.sourceFlags,
    source_breakdown: opp.source_breakdown ?? null,
    validation: quality.validation,
    is_valid: quality.isValid,
    is_actionable: quality.isActionable,
    is_tradeable: quality.isTradeable,
    validation_reasons: quality.reasons,
    source_coverage_score: quality.sourceCoverageScore,
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
  'text-violet-300',
  'text-teal-300',
  'text-orange-300',
  'text-pink-300',
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

function TraderSignalCard({
  signal,
  onNavigateToWallet,
  onOpenCopilot,
}: {
  signal: UnifiedTraderSignal
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const chartRef = useRef<HTMLDivElement>(null)
  const [chartWidth, setChartWidth] = useState(280)
  const isBuy = signal.direction === 'BUY'
  const marketLabel = normalizeMarketLabel(signal)
  const yesLabel = yesOutcomeLabel(signal)
  const noLabel = noOutcomeLabel(signal)
  const directionLabel = directionOutcomeLabel(signal)
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
  const hasBackfilledHistory = (
    Array.isArray(signal.price_history)
    && signal.price_history.length >= 2
  )
  const hasSparkline = hasBackfilledHistory && sparkSeries.length > 0
  const sparklineSeries = useMemo(
    () => sparkSeries.map((row, index) => ({
      data: row.data,
      color: SPARKLINE_COLORS[index % SPARKLINE_COLORS.length],
      lineWidth: index === 0 ? 1.6 : 1.3,
      fill: false,
      showDot: true,
    })),
    [sparkSeries],
  )
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

  const qualityPillClass = signal.is_tradeable
    ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
    : signal.is_valid
      ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
      : 'bg-red-500/10 text-red-300 border-red-500/30'
  const qualityLabel = signal.is_tradeable
    ? 'TRADEABLE'
    : signal.is_valid
      ? 'SOURCE CHECK'
      : 'INVALID'

  const bgGradient = signal.source === 'insider'
    ? 'from-purple-500/[0.04] via-transparent to-transparent'
    : signal.tier === 'EXTREME'
      ? 'from-red-500/[0.04] via-transparent to-transparent'
      : signal.tier === 'HIGH'
        ? 'from-orange-500/[0.04] via-transparent to-transparent'
        : 'from-yellow-500/[0.03] via-transparent to-transparent'

  useEffect(() => {
    if (!chartRef.current) return
    const measure = () => {
      if (!chartRef.current) return
      setChartWidth(Math.max(140, chartRef.current.offsetWidth - 4))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(chartRef.current)
    return () => ro.disconnect()
  }, [])

  return (
    <div
      className={cn(
        'relative rounded-lg border bg-card/80 overflow-hidden transition-all hover:shadow-md cursor-pointer',
        TIER_BORDER_COLORS[signal.tier],
      )}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Accent bar */}
      <div className={cn('absolute left-0 top-0 bottom-0 w-1', accentBar)} />

      <div className={cn('bg-gradient-to-r p-4 pl-4', bgGradient)}>
        {/* Row 1: Header badges */}
        <div className="flex items-center gap-1.5 flex-wrap mb-2">
          <Badge
            variant="outline"
            className={cn('text-[10px] font-medium border', SOURCE_COLORS[signal.source])}
          >
            {signal.source === 'insider' ? 'INSIDER' : 'CONFLUENCE'}
          </Badge>
          <Badge
            variant="outline"
            className={cn('text-[10px] font-medium border', TIER_COLORS[signal.tier])}
          >
            {signal.tier}
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
              ? `BUY ${directionLabelCompact}`
              : signal.outcome || signal.signal_type?.replace(/_/g, ' ') || 'SIGNAL'}
          </Badge>
          <Badge
            variant="outline"
            className={cn('text-[10px] font-medium border inline-flex items-center gap-1', qualityPillClass)}
          >
            {signal.is_tradeable ? <CheckCircle2 className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
            {qualityLabel}
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
              Pool {poolWallets}
            </Badge>
          )}
          {trackedWallets > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-blue-500/30 bg-blue-500/10 text-blue-300">
              Tracked {trackedWallets}
            </Badge>
          )}
          {groupWallets > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-amber-500/30 bg-amber-500/10 text-amber-300">
              Groups {groupWallets}
            </Badge>
          )}
          {signal.source_coverage_score === 0 && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-red-500/30 bg-red-500/10 text-red-300">
              No qualified source
            </Badge>
          )}
        </div>

        {/* Row 4: Sparkline */}
        {hasSparkline && (
          <div ref={chartRef} className="mb-3 w-full">
            <Sparkline
              data={sparkSeries[0]?.data || []}
              series={sparklineSeries}
              width={chartWidth}
              height={44}
              lineWidth={1.5}
              showDots
            />
            <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 px-0.5 text-[11px] font-data font-bold">
              {sparkSeries.map((row, index) => (
                <span
                  key={`${signal.id}-spark-${row.key}`}
                  className={SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length]}
                >
                  {compactOutcomeLabel(row.label, 14)} {row.latest != null && Number.isFinite(row.latest) ? `${(row.latest * 100).toFixed(0)}¢` : '—'}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="rounded-md border border-green-500/20 bg-green-500/10 px-2 py-1.5">
            <p className="text-sm font-bold text-green-300 truncate">{yesLabel}</p>
            <p className="text-xs font-semibold text-green-300 font-data">
              {formatPriceCents(currentYes)}
            </p>
          </div>
          <div className="rounded-md border border-red-500/20 bg-red-500/10 px-2 py-1.5">
            <p className="text-sm font-bold text-red-300 truncate">{noLabel}</p>
            <p className="text-xs font-semibold text-red-300 font-data">
              {formatPriceCents(currentNo)}
            </p>
          </div>
        </div>

        {/* Row 5: Metrics grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mb-3">
          <div>
            <p className="text-muted-foreground/70">Confidence</p>
            <p className="font-semibold text-foreground font-data">{signal.confidence}</p>
          </div>
          <div>
            <p className="text-muted-foreground/70">Wallets</p>
            <p className="font-semibold text-foreground font-data flex items-center gap-1">
              <Users className="w-3 h-3 text-muted-foreground/70" />
              {signal.source === 'confluence'
                ? signal.cluster_adjusted_wallet_count || signal.wallet_count
                : signal.wallet_count}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground/70">Entry</p>
            <p className="font-semibold text-foreground font-data">
              {formatPriceCents(actionPrice)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground/70">Source Fit</p>
            <p className="font-semibold text-foreground font-data">{sourceCoverageLabel}</p>
          </div>
        </div>

        {/* Row 6: Secondary metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mb-3">
          {signal.source === 'confluence' ? (
            <>
              <div>
                <p className="text-muted-foreground/70">Net Notional</p>
                <p className="font-semibold text-foreground font-data">
                  {formatCompact(signal.net_notional)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground/70">Window</p>
                <p className="font-semibold text-foreground font-data">
                  {signal.window_minutes || 60}m
                </p>
              </div>
            </>
          ) : (
            <>
              <div>
                <p className="text-muted-foreground/70">Edge</p>
                <p className="font-semibold text-foreground font-data">
                  {signal.edge_percent != null ? `${signal.edge_percent.toFixed(1)}%` : '\u2014'}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground/70">Pre-news Lead</p>
                <p className="font-semibold text-foreground font-data">
                  {signal.pre_news_lead_minutes != null ? `${signal.pre_news_lead_minutes.toFixed(0)}m` : '\u2014'}
                </p>
              </div>
            </>
          )}
          <div>
            <p className="text-muted-foreground/70">Qualified</p>
            <p className={cn('font-semibold font-data', signal.is_tradeable ? 'text-emerald-300' : 'text-red-300')}>
              {signal.is_tradeable ? 'YES' : 'NO'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground/70">Wallet Base</p>
            <p className="font-semibold text-foreground font-data">{walletsConsidered}</p>
          </div>
        </div>

        {/* Row 7: Confidence bar */}
        <div>
          <div className="flex items-center justify-between text-[11px] mb-1">
            <span className="text-muted-foreground/70">
              {signal.source === 'insider' ? 'Confidence' : 'Conviction'}
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
              <span>Insider Score: <span className="text-purple-400 font-data">{signal.insider_score.toFixed(2)}</span></span>
            )}
            {signal.freshness_minutes != null && (
              <span>Freshness: <span className="text-foreground/80 font-data">{signal.freshness_minutes.toFixed(0)}m</span></span>
            )}
            {signal.cluster_count != null && (
              <span>Clusters: <span className="text-foreground/80 font-data">{signal.cluster_count}</span></span>
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
              Market
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
              Copilot
            </button>
          )}
          <div className="ml-auto">
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-muted-foreground/60" />
            ) : (
              <ChevronDown className="w-4 h-4 text-muted-foreground/60" />
            )}
          </div>
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="mt-3 pt-3 border-t border-border/50 space-y-2 text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div>
                <p className="text-muted-foreground/70">Pool Wallets</p>
                <p className="font-semibold text-foreground">{poolWallets}</p>
              </div>
              <div>
                <p className="text-muted-foreground/70">Tracked Wallets</p>
                <p className="font-semibold text-foreground">{trackedWallets}</p>
              </div>
              <div>
                <p className="text-muted-foreground/70">Group Wallets</p>
                <p className="font-semibold text-foreground">{groupWallets}</p>
              </div>
              <div>
                <p className="text-muted-foreground/70">Coverage</p>
                <p className="font-semibold text-foreground">{sourceCoverageLabel}</p>
              </div>
            </div>

            {signal.source === 'confluence' && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-muted-foreground/70">Core Wallets</p>
                    <p className="font-semibold text-foreground">{signal.unique_core_wallets || 0}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Signal Type</p>
                    <p className="font-semibold text-foreground">{signal.signal_type?.replace(/_/g, ' ') || '\u2014'}</p>
                  </div>
                </div>
              </>
            )}
            {signal.source === 'insider' && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-muted-foreground/70">Suggested Size</p>
                    <p className="font-semibold text-foreground">
                      {signal.suggested_size_usd != null ? formatCompact(signal.suggested_size_usd) : '\u2014'}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Market Liquidity</p>
                    <p className="font-semibold text-foreground">
                      {signal.market_liquidity != null ? formatCompact(signal.market_liquidity) : '\u2014'}
                    </p>
                  </div>
                </div>
                {signal.wallets && signal.wallets.length > 1 && (
                  <div>
                    <p className="text-muted-foreground/70 mb-1">Involved Wallets</p>
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
                <p className="text-muted-foreground/70 mb-1">Validation</p>
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
                First: {timeAgo(signal.first_seen_at || signal.detected_at)}
              </span>
              <span className="inline-flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Last: {timeAgo(signal.last_seen_at || signal.detected_at)}
              </span>
              <span className="inline-flex items-center gap-1">
                <Users className="w-3 h-3" />
                {signal.wallet_count} wallets
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
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
  return (
    <div className="border border-border/50 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="grid grid-cols-[44px_36px_minmax(0,1fr)_60px_60px_56px_64px_72px_52px] gap-0 bg-muted/50 border-b border-border/50 text-[9px] text-muted-foreground uppercase tracking-wider font-medium">
        <div className="px-2 py-2">Type</div>
        <div className="px-2 py-2">Tier</div>
        <div className="px-2 py-2">Market</div>
        <div className="px-2 py-2 text-center">Dir</div>
        <div className="px-2 py-2 text-right">Conf</div>
        <div className="px-2 py-2 text-right">Wlts</div>
        <div className="px-2 py-2 text-right">Edge/Net</div>
        <div className="px-2 py-2 text-center">Score</div>
        <div className="px-2 py-2 text-right">Age</div>
      </div>

      {/* Body */}
      <div className="divide-y divide-border/30">
        {signals.map((signal) => (
          <TraderSignalTableRow
            key={signal.id}
            signal={signal}
            onNavigateToWallet={onNavigateToWallet}
            onOpenCopilot={onOpenCopilot}
          />
        ))}
      </div>
    </div>
  )
}

function TraderSignalTableRow({
  signal,
  onNavigateToWallet,
  onOpenCopilot,
}: {
  signal: UnifiedTraderSignal
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (signal: UnifiedTraderSignal) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const isBuy = signal.direction === 'BUY'
  const marketLabel = normalizeMarketLabel(signal)
  const directionLabel = compactOutcomeLabel(directionOutcomeLabel(signal), 9)

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
          'grid grid-cols-[44px_36px_minmax(0,1fr)_60px_60px_56px_64px_72px_52px] gap-0 cursor-pointer transition-colors',
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
            {signal.source === 'insider' ? 'INS' : 'CNF'}
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
            {signal.tier === 'INSIDER' ? 'INS' : signal.tier.slice(0, 3)}
          </span>
        </div>

        {/* Market */}
        <div className="px-2 py-2.5 min-w-0">
          <p className="text-xs text-foreground truncate">{marketLabel}</p>
        </div>

        {/* Direction */}
        <div className="px-2 py-2.5 flex justify-center">
          <span
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
                    <p className="text-muted-foreground/70">Core Wallets</p>
                    <p className="font-semibold">{signal.unique_core_wallets || 0}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Signal Type</p>
                    <p className="font-semibold">{signal.signal_type?.replace(/_/g, ' ') || '\u2014'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Window</p>
                    <p className="font-semibold">{signal.window_minutes || 60}m</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Last Reinforced</p>
                    <p className="font-semibold">{timeAgo(signal.last_seen_at || signal.detected_at)}</p>
                  </div>
                </div>
              )}

              {signal.source === 'insider' && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div>
                    <p className="text-muted-foreground/70">Insider Score</p>
                    <p className="font-semibold text-purple-400">{signal.insider_score?.toFixed(2) || '\u2014'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Pre-news Lead</p>
                    <p className="font-semibold">{signal.pre_news_lead_minutes?.toFixed(0) || '\u2014'}m</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Clusters</p>
                    <p className="font-semibold">{signal.cluster_count || '\u2014'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground/70">Freshness</p>
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
                  Open Market
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
                  Copilot
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
          <span className="text-green-400 font-bold text-xs">TRADER SIGNAL FEED</span>
          <span className="text-green-400/40">v2.0</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-green-400/60">
            {totalCount ?? signals.length} signals
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
              {isConnected ? 'LIVE' : 'DISCONNECTED'}
            </span>
          </div>
        </div>
      </div>

      {/* Terminal Body */}
      <div ref={scrollRef} className="max-h-[calc(100vh-280px)] overflow-y-auto p-3 space-y-0">
        {/* Boot sequence */}
        <div className="text-green-500/30 mb-3 space-y-0.5">
          <p>{'>'} Initializing trader signal feed...</p>
          <p>{'>'} Connected to confluence + insider pipelines</p>
          <p>{'>'} {signals.length} signals loaded</p>
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
          <span className="text-green-400/40">awaiting next signal</span>
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
  const sourceTag = signal.source === 'insider' ? 'INS' : 'CNF'
  const tierTag = signal.tier === 'INSIDER' ? 'INS' : signal.tier
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
            : signal.outcome || 'SIGNAL'}
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
