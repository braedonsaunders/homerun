import { useState, useMemo } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  TrendingUp,
  ExternalLink,
  Play,
  Brain,
  Shield,
  RefreshCw,
  MessageCircle,
  Clock,
  CalendarDays,
  Layers,
  Newspaper,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { getOpportunityPlatformLinks } from '../lib/marketUrls'
import {
  buildOutcomeFallbacks,
  buildOutcomeSparklineSeries,
  extractOutcomeLabels,
  extractOutcomePrices,
} from '../lib/priceHistory'
import { Opportunity, WeatherForecastSource, judgeOpportunity } from '../services/api'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Separator } from './ui/separator'
import Sparkline from './Sparkline'

// ─── Constants ────────────────────────────────────────────

const STRATEGY_COLORS: Record<string, string> = {
  search: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  basic: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  negrisk: 'bg-green-500/10 text-green-400 border-green-500/20',
  mutually_exclusive: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  contradiction: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  must_happen: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  cross_platform: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
  bayesian_cascade: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
  liquidity_vacuum: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  entropy_arb: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  event_driven: 'bg-lime-500/10 text-lime-400 border-lime-500/20',
  temporal_decay: 'bg-teal-500/10 text-teal-400 border-teal-500/20',
  correlation_arb: 'bg-sky-500/10 text-sky-400 border-sky-500/20',
  market_making: 'bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20',
  stat_arb: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
}

const STRATEGY_ABBREV: Record<string, string> = {
  search: 'MKT',
  basic: 'ARB',
  negrisk: 'NEG',
  mutually_exclusive: 'MXL',
  contradiction: 'CTR',
  must_happen: 'MH',
  cross_platform: 'XPL',
  bayesian_cascade: 'BAY',
  liquidity_vacuum: 'LVA',
  entropy_arb: 'ENT',
  event_driven: 'EVT',
  temporal_decay: 'TMP',
  correlation_arb: 'COR',
  market_making: 'MMK',
  stat_arb: 'SAR',
}

const STRATEGY_NAMES: Record<string, string> = {
  search: 'Market',
  basic: 'Basic Arb',
  negrisk: 'NegRisk',
  mutually_exclusive: 'Mutually Exclusive',
  contradiction: 'Contradiction',
  must_happen: 'Must-Happen',
  cross_platform: 'Cross-Platform',
  bayesian_cascade: 'Bayesian Cascade',
  liquidity_vacuum: 'Liquidity Vacuum',
  entropy_arb: 'Entropy Arb',
  event_driven: 'Event-Driven',
  temporal_decay: 'Temporal Decay',
  correlation_arb: 'Correlation Arb',
  market_making: 'Market Making',
  stat_arb: 'Statistical Arb',
}

const RECOMMENDATION_COLORS: Record<string, string> = {
  strong_execute: 'bg-green-500/20 text-green-400 border-green-500/30',
  execute: 'bg-green-500/15 text-green-400 border-green-500/20',
  review: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20',
  skip: 'bg-red-500/15 text-red-400 border-red-500/20',
  strong_skip: 'bg-red-500/20 text-red-400 border-red-500/30',
  safe: 'bg-green-500/15 text-green-400 border-green-500/20',
  caution: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20',
  avoid: 'bg-red-500/15 text-red-400 border-red-500/20',
  pending: 'bg-muted-foreground/15 text-muted-foreground border-muted-foreground/20',
}

const ACCENT_BAR_COLORS: Record<string, string> = {
  strong_execute: 'bg-green-400',
  execute: 'bg-green-500',
  review: 'bg-yellow-500',
  skip: 'bg-red-400',
  strong_skip: 'bg-red-500',
  safe: 'bg-green-500',
  caution: 'bg-yellow-500',
  avoid: 'bg-red-500',
}

const CARD_BG_GRADIENT: Record<string, string> = {
  strong_execute: 'from-green-500/[0.04] via-transparent to-transparent',
  execute: 'from-green-500/[0.03] via-transparent to-transparent',
  review: 'from-yellow-500/[0.03] via-transparent to-transparent',
  skip: 'from-red-500/[0.03] via-transparent to-transparent',
  strong_skip: 'from-red-500/[0.04] via-transparent to-transparent',
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
  'text-green-400/70',
  'text-red-400/70',
  'text-sky-300/80',
  'text-amber-300/80',
  'text-violet-300/80',
  'text-teal-300/80',
  'text-orange-300/80',
  'text-pink-300/80',
]

// ─── Utilities ────────────────────────────────────────────

export function timeAgo(dateStr: string): string {
  if (!dateStr) return '—'
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

/** Format a resolution date as a human-readable "time remaining" string */
function timeUntil(dateStr?: string | null): string {
  if (!dateStr) return '—'
  const diffMs = new Date(dateStr).getTime() - Date.now()
  if (Number.isNaN(diffMs)) return '—'
  if (diffMs <= 0) return 'Ended'
  const days = Math.floor(diffMs / 86_400_000)
  if (days > 365) return `${Math.floor(days / 365)}y`
  if (days > 30) return `${Math.floor(days / 30)}mo`
  if (days > 0) return `${days}d`
  const hrs = Math.floor(diffMs / 3_600_000)
  if (hrs > 0) return `${hrs}h`
  return `${Math.floor(diffMs / 60_000)}m`
}

function formatWeatherTargetDisplay(dateStr?: string | null): { date: string; time: string | null } | null {
  if (!dateStr) return null
  const dt = new Date(dateStr)
  if (Number.isNaN(dt.getTime())) return null
  const hasExplicitTime = !(
    dt.getUTCHours() === 0
    && dt.getUTCMinutes() === 0
    && dt.getUTCSeconds() === 0
  )
  return {
    date: dt.toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      timeZone: 'UTC',
    }),
    time: hasExplicitTime
      ? `${dt.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', timeZone: 'UTC' })} UTC`
      : null,
  }
}

/** Safely format a number with toFixed, returning a fallback for null/undefined/NaN */
function safeFixed(n: number | null | undefined, digits: number, fallback = '—'): string {
  if (n == null || Number.isNaN(n)) return fallback
  return n.toFixed(digits)
}

export function formatCompact(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '$—'
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `$${(n / 1000).toFixed(1)}K`
  if (n >= 1000) return `$${(n / 1000).toFixed(1)}K`
  if (n >= 100) return `$${n.toFixed(0)}`
  if (n >= 1) return `$${n.toFixed(2)}`
  return `$${n.toFixed(4)}`
}

function formatTemp(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${value.toFixed(1)}F`
}

function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function compactOutcomeLabel(value: string, maxChars = 12): string {
  const text = String(value || '').trim()
  if (!text) return '—'
  if (text.length <= maxChars) return text
  return `${text.slice(0, Math.max(1, maxChars - 1))}…`
}

function resolveMarketOutcomes(market: Opportunity['markets'][number] | undefined): {
  labels: string[]
  prices: number[]
} {
  if (!market) return { labels: [], prices: [] }
  const marketRow = market as unknown as Record<string, unknown>
  const labels = extractOutcomeLabels(
    marketRow.outcome_labels
    ?? marketRow.outcomes
    ?? marketRow.tokens
  )
  const prices = extractOutcomePrices(
    marketRow.outcome_prices
    ?? marketRow.prices
  )
  return { labels, prices }
}

function formatOutcomePriceSummary(
  market: Opportunity['markets'][number],
  maxOutcomes = 4,
): string {
  const { labels, prices } = resolveMarketOutcomes(market)
  if (prices.length < 1) {
    return `Yes:${safeFixed(market.yes_price, 3)} No:${safeFixed(market.no_price, 3)}`
  }
  const visible = prices.slice(0, maxOutcomes).map((price, index) => {
    const label = compactOutcomeLabel(labels[index] || `Outcome ${index + 1}`, 10)
    return `${label}:${safeFixed(price, 3)}`
  })
  const suffix = prices.length > maxOutcomes
    ? ` +${prices.length - maxOutcomes}`
    : ''
  return `${visible.join(' ')}${suffix}`
}

// ─── Props ────────────────────────────────────────────────

interface Props {
  opportunity: Opportunity
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
  onSearchNews?: (opportunity: Opportunity) => void
}

// ─── Main Component ───────────────────────────────────────

export default function OpportunityCard({ opportunity, onExecute, onOpenCopilot, onSearchNews }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [aiExpanded, setAiExpanded] = useState(false)
  const queryClient = useQueryClient()

  // AI analysis
  const inlineAnalysis = opportunity.ai_analysis
  const forceWeatherLlm = (
    (opportunity.strategy === 'weather_edge' || Boolean(opportunity.markets?.[0]?.weather))
    && opportunity.max_position_size > 0
  )
  const judgeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await judgeOpportunity({
        opportunity_id: opportunity.id,
        force_llm: forceWeatherLlm,
      })
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] })
    },
  })
  const isPending = inlineAnalysis?.recommendation === 'pending'
  const judgment = judgeMutation.data || (inlineAnalysis && !isPending ? inlineAnalysis : null)
  const resolutions = inlineAnalysis?.resolution_analyses || []
  const recommendation = judgment?.recommendation || (isPending ? 'pending' : '')
  // Search result mode (market listing, not an arbitrage opportunity)
  const isSearch = opportunity.strategy === 'search'

  // Risk color
  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

  const riskBarColor = opportunity.risk_score < 0.3
    ? 'bg-green-500'
    : opportunity.risk_score < 0.6
      ? 'bg-yellow-500'
      : 'bg-red-500'

  // Sparkline data
  const market = opportunity.markets[0]
  const weather = market?.weather
  const isWeatherOpportunity = !isSearch && (opportunity.strategy === 'weather_edge' || Boolean(weather))

  const weatherSources = useMemo((): WeatherForecastSource[] => {
    if (!weather) return []
    const explicit = Array.isArray(weather.forecast_sources)
      ? weather.forecast_sources.filter((s): s is WeatherForecastSource => Boolean(s?.source_id))
      : []
    if (explicit.length > 0) {
      return [...explicit].sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0))
    }
    const fallback: WeatherForecastSource[] = []
    if (weather.gfs_value != null) {
      fallback.push({
        source_id: 'open_meteo:gfs_seamless',
        provider: 'open_meteo',
        model: 'gfs_seamless',
        value_c: weather.gfs_value ?? null,
        value_f: weather.gfs_value != null ? weather.gfs_value * 9 / 5 + 32 : null,
        probability: weather.gfs_probability ?? null,
        weight: null,
        target_time: weather.target_time ?? null,
      })
    }
    if (weather.ecmwf_value != null) {
      fallback.push({
        source_id: 'open_meteo:ecmwf_ifs04',
        provider: 'open_meteo',
        model: 'ecmwf_ifs04',
        value_c: weather.ecmwf_value ?? null,
        value_f: weather.ecmwf_value != null ? weather.ecmwf_value * 9 / 5 + 32 : null,
        probability: weather.ecmwf_probability ?? null,
        weight: null,
        target_time: weather.target_time ?? null,
      })
    }
    return fallback
  }, [weather])

  const primaryOutcome = String(opportunity.positions_to_take?.[0]?.outcome ?? '').toUpperCase()
  const isBuyNoWeather = primaryOutcome === 'NO'
  const weatherSourceCount = weather?.source_count ?? weatherSources.length ?? 0
  const hasWeatherModelSignal = weatherSourceCount > 0 || weather?.consensus_probability != null
  const consensusYesProbability = (
    hasWeatherModelSignal
      ? ((weather?.consensus_probability ?? opportunity.expected_payout) ?? null)
      : null
  )
  const modelProbability = (
    consensusYesProbability != null
      ? Math.max(0, Math.min(1, isBuyNoWeather ? 1 - consensusYesProbability : consensusYesProbability))
      : null
  )
  const marketProbability = (
    weather?.market_probability
    ?? (market ? (isBuyNoWeather ? market.no_price : market.yes_price) : null)
  ) ?? null
  const signalEdgePercent = (
    modelProbability != null && marketProbability != null
      ? (modelProbability - marketProbability) * 100
      : null
  )
  const consensusTempF = weather?.consensus_temp_f
    ?? (weather?.consensus_temp_c != null ? (weather.consensus_temp_c * 9 / 5) + 32 : null)
  const marketImpliedTempF = weather?.market_implied_temp_f
    ?? (weather?.market_implied_temp_c != null ? (weather.market_implied_temp_c * 9 / 5) + 32 : null)
  const tempDeltaF = (
    consensusTempF != null && marketImpliedTempF != null
      ? consensusTempF - marketImpliedTempF
      : null
  )
  const weatherContractLabel = useMemo(() => {
    if (!weather) return '—'
    const unit = weather.raw_unit || 'F'
    if (weather.raw_threshold != null) {
      const op = (weather.operator || 'gt').toLowerCase()
      const opText = op === 'lt' || op === 'lte' ? '<' : '>'
      return `${opText} ${safeFixed(weather.raw_threshold, 1)}${unit}`
    }
    if (weather.raw_threshold_low != null && weather.raw_threshold_high != null) {
      return `${safeFixed(weather.raw_threshold_low, 1)}-${safeFixed(weather.raw_threshold_high, 1)}${unit}`
    }
    return '—'
  }, [weather])
  const weatherTargetDisplay = useMemo(
    () => formatWeatherTargetDisplay(weather?.target_time ?? opportunity.resolution_date ?? null),
    [weather?.target_time, opportunity.resolution_date]
  )
  const weatherTargetLabel = weatherTargetDisplay
    ? `${weatherTargetDisplay.date}${weatherTargetDisplay.time ? `, ${weatherTargetDisplay.time}` : ''}`
    : '—'

  const marketOutcomes = useMemo(() => resolveMarketOutcomes(market), [market])
  const primaryOutcomeLabel = marketOutcomes.labels[0] || 'Yes'
  const secondaryOutcomeLabel = marketOutcomes.labels[1] || 'No'
  const sparkSeries = useMemo(
    () => buildOutcomeSparklineSeries(
      market?.price_history,
      buildOutcomeFallbacks({
        labels: marketOutcomes.labels,
        prices: marketOutcomes.prices,
        yesPrice: market?.yes_price,
        noPrice: market?.no_price,
        yesLabel: marketOutcomes.labels[0] || 'Yes',
        noLabel: marketOutcomes.labels[1] || 'No',
        preferIndexedKeys: marketOutcomes.labels.length > 2 || marketOutcomes.prices.length > 2,
      }),
    ),
    [market, marketOutcomes],
  )
  const hasSparkline = sparkSeries.length > 0
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

  // Accent bar color
  const accentColor = recommendation ? (ACCENT_BAR_COLORS[recommendation] || 'bg-border') : 'bg-border/50'
  const bgGradient = recommendation ? (CARD_BG_GRADIENT[recommendation] || '') : ''

  const opportunityLinks = useMemo(
    () => getOpportunityPlatformLinks(opportunity as any),
    [opportunity]
  )
  const polyUrl = opportunityLinks.polymarketUrl
  const kalshiUrl = opportunityLinks.kalshiUrl

  // ROI direction
  const roiPositive = opportunity.roi_percent >= 0

  return (
    <Card className={cn(
      "overflow-hidden relative group transition-all duration-200",
      "hover:shadow-lg hover:shadow-black/20 hover:border-border/80",
      bgGradient && `bg-gradient-to-r ${bgGradient}`
    )}>
      {/* Left accent bar */}
      <div className={cn("absolute left-0 top-0 bottom-0 w-1 rounded-l-lg transition-all", accentColor)} />

      <div className="pl-4 pr-3 py-2.5 space-y-2">
        {/* ── Row 1: Badges + ROI / Prices ── */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 flex-wrap min-w-0">
            {isWeatherOpportunity && weatherTargetDisplay && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-cyan-500/12 text-cyan-200 border-cyan-500/25">
                <span className="inline-flex items-center gap-1">
                  <CalendarDays className="w-2.5 h-2.5" />
                  {weatherTargetDisplay.date}
                </span>
              </Badge>
            )}
            {!isSearch && !isWeatherOpportunity && (
              <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0", STRATEGY_COLORS[opportunity.strategy])}>
                {STRATEGY_NAMES[opportunity.strategy] || opportunity.strategy}
              </Badge>
            )}
            {isSearch && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-blue-500/10 text-blue-400 border-blue-500/20">
                {(opportunity as any).platform === 'kalshi' ? 'Kalshi' : 'Polymarket'}
              </Badge>
            )}
            {opportunity.category && !isWeatherOpportunity && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-muted-foreground border-border/60">
                {opportunity.category}
              </Badge>
            )}
            {judgment && (
              <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 font-bold", RECOMMENDATION_COLORS[recommendation])}>
                {recommendation.replace('_', ' ').toUpperCase()}
              </Badge>
            )}
            {isPending && !judgeMutation.isPending && (
              <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                <RefreshCw className="w-2.5 h-2.5 animate-spin" /> queued
              </span>
            )}
          </div>
          <div className="text-right shrink-0">
            {isSearch && market ? (
              <>
                {/* Search results: show all market outcomes with real labels */}
                <div className="flex items-center justify-end gap-2 flex-wrap">
                  {sparkSeries.map((row, index) => (
                    <span
                      key={`${opportunity.id}-search-outcome-${row.key}`}
                      className={cn('text-sm font-bold font-data leading-none', SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length])}
                    >
                      {compactOutcomeLabel(row.label, 12)} {safeFixed((row.latest ?? 0) * 100, 0)}¢
                    </span>
                  ))}
                </div>
                <p className="text-[10px] text-muted-foreground font-data mt-0.5">
                  {formatCompact(opportunity.volume ?? opportunity.min_liquidity)} vol
                </p>
              </>
            ) : (
              <>
                <div className="flex items-center gap-1 justify-end">
                  <TrendingUp className={cn("w-3.5 h-3.5", roiPositive ? "text-green-400" : "text-red-400")} />
                  <span className={cn(
                    "text-base font-bold font-data leading-none",
                    roiPositive ? "text-green-400 data-glow-green" : "text-red-400 data-glow-red"
                  )}>
                    {roiPositive ? '+' : ''}{safeFixed(opportunity.roi_percent, 2, '0.00')}%
                  </span>
                </div>
                <p className="text-[10px] text-muted-foreground font-data mt-0.5">
                  {formatCompact(opportunity.net_profit)} net
                </p>
              </>
            )}
          </div>
        </div>

        {/* ── Row 2: Title ── */}
        <h3 className="text-sm font-medium text-foreground truncate leading-tight" title={opportunity.title}>
          {opportunity.title}
        </h3>

        {isWeatherOpportunity && weatherTargetDisplay && (
          <div className="flex items-center justify-between rounded-md border border-cyan-500/25 bg-cyan-500/[0.08] px-2 py-1">
            <span className="inline-flex items-center gap-1.5 min-w-0">
              <CalendarDays className="w-3 h-3 text-cyan-300 shrink-0" />
              <span className="text-xs font-semibold text-cyan-100 truncate">{weatherTargetDisplay.date}</span>
            </span>
            {weatherTargetDisplay.time && (
              <span className="text-[10px] font-data text-cyan-200/90 ml-2 shrink-0">{weatherTargetDisplay.time}</span>
            )}
          </div>
        )}

        {/* ── Row 3: Sparkline + Metrics ── */}
        <div className="flex items-stretch gap-3">
          {/* Sparkline */}
          {market && hasSparkline && (
            <div className="shrink-0">
              <Sparkline
                data={sparkSeries[0]?.data || []}
                series={sparklineSeries}
                width={96}
                height={40}
                lineWidth={1.5}
                showDots
              />
              <div className="mt-0.5 flex flex-wrap gap-x-1.5 gap-y-0.5 px-0.5 text-[9px] font-data">
                {sparkSeries.map((row, index) => (
                  <span
                    key={`${opportunity.id}-spark-${row.key}`}
                    className={SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length]}
                  >
                    {compactOutcomeLabel(row.label, 10)} {safeFixed(row.latest, 2)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Metrics Grid */}
          {isSearch ? (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 flex-1 min-w-0">
              <MiniMetric label="Volume" value={formatCompact(opportunity.volume ?? opportunity.min_liquidity)} />
              <MiniMetric label="Liquidity" value={formatCompact(opportunity.min_liquidity)} />
              <MiniMetric label="Ends" value={timeUntil(opportunity.resolution_date)} />
              <MiniMetric
                label="Competitive"
                value={market ? `${safeFixed(Math.abs(market.yes_price - 0.5) * 200, 0)}%` : '—'}
                valueClass={market && Math.abs(market.yes_price - 0.5) < 0.1 ? 'text-green-400' : undefined}
              />
            </div>
          ) : (
            isWeatherOpportunity ? (
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 flex-1 min-w-0">
                <MiniMetric
                  label="Market Px"
                  value={marketProbability != null ? `${safeFixed(marketProbability * 100, 1)}¢` : '—'}
                />
                <MiniMetric
                  label="Model Px"
                  value={modelProbability != null ? `${safeFixed(modelProbability * 100, 1)}¢` : '—'}
                />
                <MiniMetric
                  label="Temp Delta"
                  value={tempDeltaF != null ? `${tempDeltaF >= 0 ? '+' : ''}${safeFixed(tempDeltaF, 1)}F` : '—'}
                  valueClass={tempDeltaF != null ? (tempDeltaF >= 0 ? 'text-cyan-300' : 'text-orange-300') : undefined}
                />
                <MiniMetric
                  label="Sources"
                  value={`${weather?.source_count ?? weatherSources.length ?? 0} src`}
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 flex-1 min-w-0">
                <MiniMetric label="Cost" value={formatCompact(opportunity.total_cost)} />
                <MiniMetric label="Liq" value={formatCompact(opportunity.min_liquidity)} />
                <MiniMetric
                  label="Risk"
                  value={`${safeFixed((opportunity.risk_score ?? 0) * 100, 0)}%`}
                  valueClass={riskColor}
                  bar={opportunity.risk_score}
                  barClass={riskBarColor}
                />
                <MiniMetric label="Max Pos" value={formatCompact(opportunity.max_position_size)} />
              </div>
            )
          )}
        </div>

        {isWeatherOpportunity && (
          <div className="flex items-center justify-between rounded-md border border-cyan-500/20 bg-cyan-500/[0.06] px-2 py-1">
            <span className="text-[10px] text-cyan-300/90 font-medium truncate">
              Mkt {formatTemp(marketImpliedTempF)} vs Consensus {formatTemp(consensusTempF)}
            </span>
            <span className="text-[10px] font-data text-muted-foreground ml-2 shrink-0">
              Edge {signalEdgePercent != null ? `${signalEdgePercent >= 0 ? '+' : ''}${safeFixed(signalEdgePercent, 1)}%` : '—'}
            </span>
          </div>
        )}

        {/* ── Row 4: AI Score Bar ── */}
        {judgment ? (
          <div className="flex items-center gap-2 bg-purple-500/[0.06] rounded-md px-2 py-1.5 border border-purple-500/10">
            <Brain className="w-3 h-3 text-purple-400 shrink-0" />
            <div className="flex-1 h-1.5 bg-muted/80 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-400 transition-all"
                style={{ width: `${Math.min(100, judgment.overall_score * 100)}%` }}
              />
            </div>
            <span className="text-[10px] font-data font-bold text-purple-300 shrink-0">
              {safeFixed((judgment.overall_score ?? 0) * 100, 0)}
            </span>
            <Separator orientation="vertical" className="h-3 bg-purple-500/20" />
            <div className="flex gap-1.5 text-[9px] font-data text-muted-foreground shrink-0">
              <span title="Profit">P{safeFixed((judgment.profit_viability ?? 0) * 100, 0)}</span>
              <span title="Resolution">R{safeFixed((judgment.resolution_safety ?? 0) * 100, 0)}</span>
              <span title="Execution">E{safeFixed((judgment.execution_feasibility ?? 0) * 100, 0)}</span>
              <span title="Efficiency">M{safeFixed((judgment.market_efficiency ?? 0) * 100, 0)}</span>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); judgeMutation.mutate() }}
              disabled={judgeMutation.isPending}
              className="text-muted-foreground hover:text-purple-400 transition-colors shrink-0"
            >
              <RefreshCw className={cn("w-2.5 h-2.5", judgeMutation.isPending && "animate-spin")} />
            </button>
          </div>
        ) : !isPending ? (
          <div className="flex items-center justify-between bg-muted/30 rounded-md px-2 py-1.5 border border-border/50">
            <span className="text-[10px] text-muted-foreground flex items-center gap-1.5">
              <Brain className="w-3 h-3" /> No AI analysis
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); judgeMutation.mutate() }}
              disabled={judgeMutation.isPending}
              className="text-[10px] text-purple-400 hover:text-purple-300 font-medium transition-colors"
            >
              {judgeMutation.isPending ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        ) : null}

        {judgment?.reasoning && (
          <p
            className={cn(
              "text-[10px] text-muted-foreground leading-relaxed px-0.5 cursor-pointer hover:text-muted-foreground/80 transition-colors",
              !aiExpanded && "line-clamp-2"
            )}
            onClick={(e) => { e.stopPropagation(); setAiExpanded(!aiExpanded) }}
            title={aiExpanded ? "Click to collapse" : "Click to expand full analysis"}
          >
            {judgment.reasoning}
          </p>
        )}

        {/* ── Row 5: Positions + Time ── */}
        <div className="flex items-center text-[10px] text-muted-foreground gap-1.5 overflow-hidden">
          {isSearch ? (
            <div className="flex items-center gap-1 truncate min-w-0">
              {market && (
                <div className="flex items-center gap-1.5 flex-wrap font-data">
                  {sparkSeries.map((row, index) => (
                    <span
                      key={`${opportunity.id}-search-line-${row.key}`}
                      className={SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length]}
                    >
                      {compactOutcomeLabel(row.label, 12)} {safeFixed(row.latest, 3)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-1 truncate min-w-0">
              {opportunity.positions_to_take.slice(0, 3).map((pos, i) => (
                <span key={i} className="inline-flex items-center gap-0.5 shrink-0">
                  {i > 0 && <span className="text-border">·</span>}
                  <span className={cn(
                    "font-data font-medium",
                    pos.outcome === 'YES' ? 'text-green-400/80' : 'text-red-400/80'
                  )}>
                    {pos.action} {pos.outcome}
                  </span>
                  <span className="font-data">@{safeFixed(pos.price, 2)}</span>
                </span>
              ))}
              {opportunity.positions_to_take.length > 3 && (
                <span className="text-muted-foreground/60">+{opportunity.positions_to_take.length - 3}</span>
              )}
            </div>
          )}
          <div className="flex items-center gap-1.5 ml-auto shrink-0">
            {!isSearch && (
              <span className="flex items-center gap-0.5">
                <Layers className="w-2.5 h-2.5" />
                {opportunity.markets.length}
              </span>
            )}
            <span className="flex items-center gap-0.5">
              <Clock className="w-2.5 h-2.5" />
              {timeAgo(opportunity.detected_at)}
            </span>
          </div>
        </div>

        {/* ── Row 6: Action Buttons ── */}
        <div className="flex items-center gap-1.5 pt-0.5">
          {polyUrl && (
            <a
              href={polyUrl}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1 h-6 px-2 text-[10px] rounded border bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20 transition-colors font-medium"
            >
              <ExternalLink className="w-2.5 h-2.5" />
              PM
            </a>
          )}
          {kalshiUrl && (
            <a
              href={kalshiUrl}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1 h-6 px-2 text-[10px] rounded border bg-indigo-500/10 text-indigo-400 border-indigo-500/20 hover:bg-indigo-500/20 transition-colors font-medium"
            >
              <ExternalLink className="w-2.5 h-2.5" />
              KL
            </a>
          )}
          {onOpenCopilot && (
            <button
              onClick={(e) => { e.stopPropagation(); onOpenCopilot(opportunity) }}
              className="inline-flex items-center gap-1 h-6 px-2 text-[10px] rounded border bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20 transition-colors font-medium"
            >
              <MessageCircle className="w-2.5 h-2.5" />
              AI
            </button>
          )}
          {onSearchNews && (
            <button
              onClick={(e) => { e.stopPropagation(); onSearchNews(opportunity) }}
              className="inline-flex items-center gap-1 h-6 px-2 text-[10px] rounded border bg-orange-500/10 text-orange-400 border-orange-500/20 hover:bg-orange-500/20 transition-colors font-medium"
              title="Search related news articles"
            >
              <Newspaper className="w-2.5 h-2.5" />
              News
            </button>
          )}
          <div
            className="ml-auto flex items-center gap-1 text-[10px] text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Less' : 'More'}
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </div>
        </div>

        {judgeMutation.error && (
          <div className="text-[10px] text-red-400">
            Analysis failed{(judgeMutation.error as any)?.code === 'ECONNABORTED'
              ? ' (request timed out — try again)'
              : `: ${(judgeMutation.error as Error).message}`}
          </div>
        )}
      </div>

      {/* ── Expanded Details ── */}
      {expanded && (
        <>
          <Separator />
          <div className="p-3 pl-4 space-y-3">
            {/* Description */}
            {opportunity.description && (
              <p className="text-xs text-muted-foreground leading-relaxed">
                {opportunity.description}
              </p>
            )}

            {isSearch ? (
              <>
                {/* ── Search result expanded: Market Details ── */}
                <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">Market Details</h4>
                  <div className="grid grid-cols-3 gap-3 text-xs">
                    <div>
                      <p className="text-[10px] text-muted-foreground">{compactOutcomeLabel(primaryOutcomeLabel, 16)} Price</p>
                      <p className="font-data text-green-400">{safeFixed((market?.yes_price ?? 0) * 100, 1)}¢</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{compactOutcomeLabel(secondaryOutcomeLabel, 16)} Price</p>
                      <p className="font-data text-red-400">{safeFixed((market?.no_price ?? 0) * 100, 1)}¢</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Spread</p>
                      <p className="font-data text-foreground">{market ? safeFixed(Math.abs(1 - market.yes_price - market.no_price) * 100, 1) : '—'}¢</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Volume</p>
                      <p className="font-data text-foreground">{formatCompact(opportunity.volume ?? 0)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Liquidity</p>
                      <p className="font-data text-foreground">{formatCompact(opportunity.min_liquidity)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Ends</p>
                      <p className="font-data text-foreground">{timeUntil(opportunity.resolution_date)}</p>
                    </div>
                  </div>
                </div>

                {/* Market link */}
                {opportunity.markets.map((mkt, idx) => {
                  const marketLink = opportunityLinks.marketLinks[idx]
                  const url = marketLink?.url || null
                  return (
                    <div key={idx} className="flex items-center justify-between bg-muted/50 rounded-md px-2.5 py-1.5 gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-[11px] text-foreground/80 truncate">{mkt.question}</p>
                        <p className="text-[9px] text-muted-foreground font-data mt-0.5">
                          {formatOutcomePriceSummary(mkt)} Vol:{formatCompact((mkt as any).volume)} Liq:{formatCompact((mkt as any).liquidity || mkt.liquidity)}
                        </p>
                      </div>
                      {url && (
                        <a
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </div>
                  )
                })}
              </>
            ) : (
              <>
                {isWeatherOpportunity && weather && (
                  <div className="bg-cyan-500/[0.06] rounded-lg p-3 border border-cyan-500/20">
                    <h4 className="text-[10px] font-medium text-cyan-300 mb-2 uppercase tracking-wider">Weather Intelligence</h4>
                    <div className="grid grid-cols-3 gap-3 text-xs">
                      <div>
                        <p className="text-[10px] text-muted-foreground">Location</p>
                        <p className="font-data text-foreground truncate">{weather.location || '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Contract</p>
                        <p className="font-data text-foreground">{weatherContractLabel}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Target</p>
                        <p className="font-data text-foreground">{weatherTargetLabel}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Market</p>
                        <p className="font-data text-foreground">{marketProbability != null ? `${safeFixed(marketProbability * 100, 1)}%` : '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Model</p>
                        <p className="font-data text-cyan-300">{modelProbability != null ? `${safeFixed(modelProbability * 100, 1)}%` : '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Edge</p>
                        <p className={cn(
                          "font-data",
                          signalEdgePercent != null && signalEdgePercent >= 0 ? 'text-green-400' : 'text-red-400'
                        )}>
                          {signalEdgePercent != null ? `${signalEdgePercent >= 0 ? '+' : ''}${safeFixed(signalEdgePercent, 1)}%` : '—'}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Market-Implied Temp</p>
                        <p className="font-data text-foreground">{formatTemp(marketImpliedTempF)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Consensus Temp</p>
                        <p className="font-data text-cyan-300">{formatTemp(consensusTempF)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">Model Agreement</p>
                        <p className="font-data text-foreground">{formatPct(weather.agreement)}</p>
                      </div>
                    </div>
                    {weatherSources.length > 0 && (
                      <div className="mt-3 space-y-1.5">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Forecast Sources ({weatherSources.length})</p>
                        {weatherSources.map((src) => (
                          <div key={src.source_id} className="flex items-center justify-between gap-2 rounded-md border border-border/50 bg-muted/40 px-2.5 py-1.5">
                            <div className="min-w-0">
                              <p className="text-[11px] text-foreground truncate">
                                {src.provider}:{src.model}
                              </p>
                              <p className="text-[9px] text-muted-foreground font-data">
                                Temp {formatTemp(src.value_f)} · Prob {formatPct(src.probability)}
                              </p>
                            </div>
                            <span className="text-[10px] text-cyan-300 font-data shrink-0">
                              {src.weight != null ? `w ${safeFixed(src.weight * 100, 0)}%` : 'w —'}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* ── Arbitrage opportunity expanded: Positions + Profit ── */}
                {/* Positions to Take */}
                <div>
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">Positions</h4>
                    <div className="space-y-1.5">
                      {opportunity.positions_to_take.map((pos, idx) => {
                        const platform = (pos as any).platform
                        return (
                          <div key={idx} className="flex items-center justify-between bg-muted/50 rounded-md px-2.5 py-1.5">
                          <div className="flex items-center gap-1.5 min-w-0 flex-1">
                            <Badge variant="outline" className={cn(
                              "text-[10px] px-1.5 py-0",
                              pos.outcome === 'YES' ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'
                            )}>
                              {pos.action} {pos.outcome}
                            </Badge>
                            {platform && (
                              <Badge variant="outline" className={cn(
                                "text-[9px] px-1 py-0",
                                platform === 'kalshi' ? 'text-indigo-400 border-indigo-500/20' : 'text-blue-400 border-blue-500/20'
                              )}>
                                {platform === 'kalshi' ? 'KL' : 'PM'}
                              </Badge>
                            )}
                            <span className="text-[11px] text-foreground/70 truncate min-w-0 flex-1">{pos.market}</span>
                          </div>
                          <span className="font-data text-xs text-foreground shrink-0">${safeFixed(pos.price, 4)}</span>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Profit Breakdown */}
                <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">Profit Breakdown</h4>
                  <div className="grid grid-cols-3 gap-3 text-xs">
                    <div>
                      <p className="text-[10px] text-muted-foreground">Cost</p>
                      <p className="font-data text-foreground">${safeFixed(opportunity.total_cost, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Payout</p>
                      <p className="font-data text-foreground">${safeFixed(opportunity.expected_payout, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Gross</p>
                      <p className="font-data text-foreground">${safeFixed(opportunity.gross_profit, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Fee (2%)</p>
                      <p className="font-data text-red-400">-${safeFixed(opportunity.fee, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">Net</p>
                      <p className="font-data text-green-400">${safeFixed(opportunity.net_profit, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">ROI</p>
                      <p className="font-data text-green-400">{safeFixed(opportunity.roi_percent, 2)}%</p>
                    </div>
                  </div>
                </div>

                {/* Markets */}
                <div>
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">Markets ({opportunity.markets.length})</h4>
                  <div className="space-y-1.5">
                    {opportunity.markets.map((mkt, idx) => {
                      const marketLink = opportunityLinks.marketLinks[idx]
                      const url = marketLink?.url || null
                      return (
                        <div key={idx} className="flex items-center justify-between bg-muted/50 rounded-md px-2.5 py-1.5 gap-2">
                          <div className="min-w-0 flex-1">
                            <p className="text-[11px] text-foreground/80 truncate">{mkt.question}</p>
                            <p className="text-[9px] text-muted-foreground font-data mt-0.5">
                              {formatOutcomePriceSummary(mkt)} Liq:{formatCompact(mkt.liquidity)}
                            </p>
                          </div>
                          {url && (
                            <a
                              href={url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </>
            )}

            {/* Risk Factors */}
            {opportunity.risk_factors.length > 0 && (
              <div>
                <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">Risk Factors</h4>
                <div className="flex flex-wrap gap-1">
                  {opportunity.risk_factors.map((f, i) => (
                    <span key={i} className="inline-flex items-center gap-1 text-[10px] text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded border border-yellow-500/10">
                      <AlertTriangle className="w-2.5 h-2.5" />
                      {f.length > 50 ? f.slice(0, 50) + '...' : f}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Resolution Analysis */}
            {resolutions.length > 0 && resolutions[0].summary && (
              <div>
                <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">Resolution</h4>
                {resolutions.map((r: any, i: number) => (
                  <div key={i} className="bg-muted/30 rounded-md p-2 space-y-1 border border-border/50">
                    <div className="flex items-center gap-1.5">
                      <Shield className="w-2.5 h-2.5 text-muted-foreground" />
                      <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0', RECOMMENDATION_COLORS[r.recommendation])}>
                        {r.recommendation}
                      </Badge>
                      <span className="text-[9px] text-muted-foreground/60 font-data">
                        C:{safeFixed((r.clarity_score ?? 0) * 100, 0)} R:{safeFixed((r.risk_score ?? 0) * 100, 0)}
                      </span>
                    </div>
                    {r.summary && <p className="text-[10px] text-muted-foreground">{r.summary}</p>}
                  </div>
                ))}
              </div>
            )}

            {/* Execute Button */}
            {onExecute && !isSearch && opportunity.max_position_size > 0 && (
              <Button
                onClick={(e) => { e.stopPropagation(); onExecute(opportunity) }}
                size="sm"
                className="w-full bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600 shadow-lg shadow-blue-500/20 h-8 text-xs"
              >
                <Play className="w-3 h-3 mr-1.5" />
                Execute Trade
              </Button>
            )}
          </div>
        </>
      )}
    </Card>
  )
}

// ─── Sub-components ───────────────────────────────────────

function MiniMetric({
  label,
  value,
  valueClass,
  bar,
  barClass,
}: {
  label: string
  value: string
  valueClass?: string
  bar?: number
  barClass?: string
}) {
  return (
    <div className="min-w-0">
      <p className="text-[9px] text-muted-foreground/70 leading-none mb-0.5 uppercase tracking-wider">{label}</p>
      <div className="flex items-center gap-1.5">
        <span className={cn("text-xs font-data font-medium leading-none", valueClass || "text-foreground")}>{value}</span>
        {bar !== undefined && (
          <div className="flex-1 h-1 bg-muted/80 rounded-full overflow-hidden max-w-[40px]">
            <div className={cn("h-full rounded-full", barClass)} style={{ width: `${Math.min(100, bar * 100)}%` }} />
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Exports for shared use ───────────────────────────────

export { STRATEGY_COLORS, STRATEGY_NAMES, STRATEGY_ABBREV, RECOMMENDATION_COLORS, ACCENT_BAR_COLORS }
