import { memo, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  TrendingUp,
  ExternalLink,
  Brain,
  Shield,
  RefreshCw,
  MessageCircle,
  Clock,
  CalendarDays,
  Layers,
  Minimize2,
  Newspaper,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { getOpportunityPlatformLinks } from '../lib/marketUrls'
import {
  buildOutcomeFallbacks,
  buildOutcomeSparklineSeries,
  extractOutcomeLabels,
  extractOutcomePrices,
  toTimeValueSeries,
} from '../lib/priceHistory'
import { Opportunity, WeatherForecastSource, judgeOpportunity, getWeatherWorkflowSettings } from '../services/api'
import { Liveline } from 'liveline'
import type { LivelineSeries } from 'liveline'
import { useAtomValue } from 'jotai'
import { themeAtom } from '../store/atoms'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { Separator } from './ui/separator'
import BuyButton from './BuyButton'

// ─── Constants ────────────────────────────────────────────

const STRATEGY_COLORS: Record<string, string> = {
  // Scanner detectors
  search: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  basic: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  negrisk: 'bg-green-500/10 text-green-400 border-green-500/20',
  mutually_exclusive: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  contradiction: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  must_happen: 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border-cyan-500/20',
  cross_platform: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
  ctf_basic_arb: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
  entropy_arb: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  event_driven: 'bg-lime-500/10 text-lime-400 border-lime-500/20',
  market_making: 'bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20',
  stat_arb: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  miracle: 'bg-pink-500/10 text-pink-400 border-pink-500/20',
  combinatorial: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
  settlement_lag: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  flash_crash_reversion: 'bg-red-500/10 text-red-400 border-red-500/20',
  news_momentum_breakout: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  tail_end_carry: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20',
  spread_dislocation: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
  // Weather detectors
  weather_edge: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  weather_ensemble_edge: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  weather_distribution: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  // Pipeline detectors
  news_edge: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  btc_eth_maker_quote: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  btc_eth_directional_edge: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  btc_eth_convergence: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  traders_confluence: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  sports_overreaction_fader: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
}

const STRATEGY_ABBREV: Record<string, string> = {
  search: 'MKT',
  basic: 'ARB',
  negrisk: 'NEG',
  mutually_exclusive: 'MXL',
  contradiction: 'CTR',
  must_happen: 'MH',
  cross_platform: 'XPL',
  ctf_basic_arb: 'CTF',
  entropy_arb: 'ENT',
  event_driven: 'EVT',
  market_making: 'MMK',
  stat_arb: 'SAR',
  miracle: 'MIR',
  combinatorial: 'CMB',
  settlement_lag: 'SET',
  flash_crash_reversion: 'FCR',
  news_momentum_breakout: 'NMB',
  tail_end_carry: 'TEC',
  spread_dislocation: 'SPR',
  weather_edge: 'WTH',
  weather_ensemble_edge: 'WEN',
  weather_distribution: 'WDI',
  news_edge: 'NEW',
  btc_eth_maker_quote: 'BTC',
  btc_eth_directional_edge: 'BTD',
  btc_eth_convergence: 'BTC',
  traders_confluence: 'TRD',
  sports_overreaction_fader: 'SFD',
}

const STRATEGY_NAMES: Record<string, string> = {
  search: 'Market',
  basic: 'Basic Arb',
  negrisk: 'NegRisk',
  mutually_exclusive: 'Mutually Exclusive',
  contradiction: 'Contradiction',
  must_happen: 'Must-Happen',
  cross_platform: 'Cross-Platform',
  ctf_basic_arb: 'CTF Basic Arb',
  entropy_arb: 'Entropy Arb',
  event_driven: 'Event-Driven',
  market_making: 'Market Making',
  stat_arb: 'Statistical Arb',
  miracle: 'Miracle',
  combinatorial: 'Combinatorial',
  settlement_lag: 'Settlement Lag',
  flash_crash_reversion: 'Flash Crash',
  news_momentum_breakout: 'News Momentum',
  tail_end_carry: 'Tail-End Carry',
  spread_dislocation: 'Spread Dislocation',
  weather_edge: 'Weather Edge',
  weather_ensemble_edge: 'Weather Ensemble',
  weather_distribution: 'Weather Distribution',
  news_edge: 'News Edge',
  btc_eth_maker_quote: 'Crypto Maker Quote',
  btc_eth_directional_edge: 'Crypto Directional Edge',
  btc_eth_convergence: 'Crypto Convergence',
  traders_confluence: 'Traders Flow',
  sports_overreaction_fader: 'Sports Fader',
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
  'text-green-600/70 dark:text-green-400/70',
  'text-red-600/70 dark:text-red-400/70',
  'text-sky-600/80 dark:text-sky-300/80',
  'text-amber-600/80 dark:text-amber-300/80',
  'text-violet-600/80 dark:text-violet-300/80',
  'text-teal-600/80 dark:text-teal-300/80',
  'text-orange-600/80 dark:text-orange-300/80',
  'text-pink-600/80 dark:text-pink-300/80',
]

// ─── Utilities ────────────────────────────────────────────

export function timeAgo(dateStr: string, t?: TFunction): string {
  if (!dateStr) return '—'
  const diffMs = Date.now() - new Date(dateStr).getTime()
  if (diffMs < 0 || Number.isNaN(diffMs)) return t ? t('opportunityCard.now') : 'now'
  const sec = Math.floor(diffMs / 1000)
  if (sec < 60) return `${sec}s`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h`
  return `${Math.floor(hr / 24)}d`
}

function resolveOpportunityFreshnessTimestamp(opportunity: Opportunity): string {
  const candidates = [
    opportunity.last_priced_at,
    opportunity.last_detected_at,
    opportunity.last_seen_at,
    opportunity.detected_at,
  ]
  for (const candidate of candidates) {
    if (!candidate) continue
    const ts = new Date(candidate).getTime()
    if (!Number.isNaN(ts)) return candidate
  }
  return opportunity.detected_at
}

/** Format a resolution date as a human-readable "time remaining" string */
function timeUntil(dateStr: string | null | undefined, t: TFunction): string {
  if (!dateStr) return '—'
  const diffMs = new Date(dateStr).getTime() - Date.now()
  if (Number.isNaN(diffMs)) return '—'
  if (diffMs <= 0) return t('opportunityCard.ended')
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

function formatTemp(value: number | null | undefined, unit: 'F' | 'C' = 'F'): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${value.toFixed(1)}°${unit}`
}

function fToC(f: number): number {
  return (f - 32) * 5 / 9
}

function cToF(c: number): number {
  return (c * 9 / 5) + 32
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

function resolveMarketYesPrice(market: Opportunity['markets'][number] | undefined): number {
  if (!market) return 0
  const raw = Number((market as any).current_yes_price ?? market.yes_price)
  if (Number.isFinite(raw)) return raw
  const fallback = Number(market.yes_price)
  return Number.isFinite(fallback) ? fallback : 0
}

function resolveMarketNoPrice(market: Opportunity['markets'][number] | undefined): number {
  if (!market) return 0
  const raw = Number((market as any).current_no_price ?? market.no_price)
  if (Number.isFinite(raw)) return raw
  const fallback = Number(market.no_price)
  return Number.isFinite(fallback) ? fallback : 0
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
  const outcomePriceSource = marketRow.outcome_prices
    ?? marketRow.prices
  let prices = extractOutcomePrices(outcomePriceSource)
  if (prices.length === 0) {
    prices = extractOutcomePrices(marketRow.tokens)
  }
  return { labels, prices }
}

/** For multi-market opportunities (e.g. negrisk with 12 sub-markets),
 *  build a global outcome view: each sub-market is one "outcome",
 *  labeled by its short title, with YES price as the outcome price. */
function resolveMultiMarketOutcomes(markets: Opportunity['markets']): {
  labels: string[]
  prices: number[]
} {
  const labels: string[] = []
  const prices: number[] = []
  for (const mkt of markets) {
    // Use group_item_title if available, otherwise extract short label from question
    const raw = (mkt as any).group_item_title || mkt.question || ''
    // Take the first meaningful segment — remove common prefixes like "Will..."
    const label = raw.replace(/^(Will |What will |Which |Who will )/i, '').split('?')[0].trim()
    labels.push(label || `Market ${labels.length + 1}`)
    prices.push(resolveMarketYesPrice(mkt))
  }
  return { labels, prices }
}

function formatOutcomePriceSummary(
  market: Opportunity['markets'][number],
  maxOutcomes = 4,
): string {
  const { labels, prices } = resolveMarketOutcomes(market)
  if (prices.length < 1) {
    return `Yes:${safeFixed(resolveMarketYesPrice(market), 3)} No:${safeFixed(resolveMarketNoPrice(market), 3)}`
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
  onOpenCopilot?: (opportunity: Opportunity) => void
  onSearchNews?: (opportunity: Opportunity) => void
  isModalView?: boolean
  onCloseModal?: () => void
}

// ─── Main Component ───────────────────────────────────────

function OpportunityCard({
  opportunity,
  onOpenCopilot,
  onSearchNews,
  isModalView = false,
  onCloseModal,
}: Props) {
  const { t } = useTranslation()
  const [aiExpanded, setAiExpanded] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const themeMode = useAtomValue(themeAtom)
  const queryClient = useQueryClient()
  const strategyLabel = STRATEGY_NAMES[opportunity.strategy]
    ? t(`opportunityCard.strategyName.${opportunity.strategy}`, { defaultValue: STRATEGY_NAMES[opportunity.strategy] })
    : opportunity.strategy
  const translateRecommendation = (rec: string): string => rec
    ? t(`opportunityCard.recommendation.${rec}`, { defaultValue: rec.replace('_', ' ').toUpperCase() })
    : ''

  // Temperature unit preference
  const { data: weatherSettings } = useQuery({
    queryKey: ['weather-workflow-settings'],
    queryFn: getWeatherWorkflowSettings,
    staleTime: 60_000,
  })
  const tempUnit: 'F' | 'C' = weatherSettings?.temperature_unit ?? 'F'

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
  const strategySdk = String(opportunity.strategy || '').trim().toLowerCase()
  const freshnessTimestamp = resolveOpportunityFreshnessTimestamp(opportunity)
  const firstDetectedTimestamp = opportunity.first_detected_at || opportunity.detected_at

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
  const primaryMarketQuestion = String(market?.question || '').trim()
  const preferMarketQuestionTitle = strategySdk === 'tail_end_carry' && primaryMarketQuestion.length > 0
  const headerTitle = preferMarketQuestionTitle ? primaryMarketQuestion : opportunity.title
  const headerSubtitle = preferMarketQuestionTitle ? opportunity.title : null
  const marketYesPrice = resolveMarketYesPrice(market)
  const marketNoPrice = resolveMarketNoPrice(market)
  const weather = market?.weather
  const WEATHER_STRATEGIES = new Set(['weather_edge', 'weather_ensemble_edge', 'weather_distribution'])
  const isWeatherOpportunity = !isSearch && (WEATHER_STRATEGIES.has(opportunity.strategy) || Boolean(weather))

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
    ?? (market ? (isBuyNoWeather ? marketNoPrice : marketYesPrice) : null)
  ) ?? null
  const signalEdgePercent = (
    modelProbability != null && marketProbability != null
      ? (modelProbability - marketProbability) * 100
      : null
  )
  // Resolve temperatures in preferred unit
  const consensusTemp = tempUnit === 'C'
    ? (weather?.consensus_temp_c ?? (weather?.consensus_temp_f != null ? fToC(weather.consensus_temp_f) : null))
    : (weather?.consensus_temp_f ?? (weather?.consensus_temp_c != null ? cToF(weather.consensus_temp_c) : null))
  const marketImpliedTemp = tempUnit === 'C'
    ? (weather?.market_implied_temp_c ?? (weather?.market_implied_temp_f != null ? fToC(weather.market_implied_temp_f) : null))
    : (weather?.market_implied_temp_f ?? (weather?.market_implied_temp_c != null ? cToF(weather.market_implied_temp_c) : null))
  const tempDelta = (
    consensusTemp != null && marketImpliedTemp != null
      ? consensusTemp - marketImpliedTemp
      : null
  )
  const weatherContractLabel = useMemo(() => {
    if (!weather) return '—'
    const rawUnit = weather.raw_unit || 'F'
    // Convert thresholds to preferred display unit
    const convertThreshold = (val: number): number => {
      if (rawUnit === tempUnit) return val
      return tempUnit === 'C' ? fToC(val) : cToF(val)
    }
    if (weather.raw_threshold != null) {
      const op = (weather.operator || 'gt').toLowerCase()
      const opText = op === 'lt' || op === 'lte' ? '<' : '>'
      return `${opText} ${safeFixed(convertThreshold(weather.raw_threshold), 1)}°${tempUnit}`
    }
    if (weather.raw_threshold_low != null && weather.raw_threshold_high != null) {
      return `${safeFixed(convertThreshold(weather.raw_threshold_low), 1)}-${safeFixed(convertThreshold(weather.raw_threshold_high), 1)}°${tempUnit}`
    }
    return '—'
  }, [weather, tempUnit])
  const weatherTargetDisplay = useMemo(
    () => formatWeatherTargetDisplay(weather?.target_time ?? opportunity.resolution_date ?? null),
    [weather?.target_time, opportunity.resolution_date]
  )
  const weatherTargetLabel = weatherTargetDisplay
    ? `${weatherTargetDisplay.date}${weatherTargetDisplay.time ? `, ${weatherTargetDisplay.time}` : ''}`
    : '—'

  // For multi-market opportunities (negrisk, mutually_exclusive, etc.),
  // build a global view where each sub-market is an outcome.
  const isMultiMarket = opportunity.markets.length > 1
  const multiMarketOutcomes = useMemo(
    () => isMultiMarket ? resolveMultiMarketOutcomes(opportunity.markets) : null,
    [isMultiMarket, opportunity.markets],
  )
  const singleMarketOutcomes = useMemo(() => resolveMarketOutcomes(market), [market])
  const marketOutcomes = multiMarketOutcomes ?? singleMarketOutcomes
  const primaryOutcomeLabel = marketOutcomes.labels[0] || t('opportunityCard.yes')
  const secondaryOutcomeLabel = marketOutcomes.labels[1] || t('opportunityCard.no')
  const sparkSeries = useMemo(
    () => {
      if (isMultiMarket && multiMarketOutcomes) {
        // Build sparkline fallbacks from all sub-markets' YES prices
        const fallbacks = multiMarketOutcomes.labels.map((label, i) => ({
          key: `idx_${i}`,
          label,
          price: multiMarketOutcomes.prices[i] ?? null,
        }))
        // Merge price histories by timestamp (not by index) so uneven history
        // lengths still align correctly across legs.
        const mergedByTs = new Map<number, Record<string, unknown>>()
        let syntheticTs = 1
        const coerceTs = (value: unknown): number | null => {
          const raw = Number(value)
          if (!Number.isFinite(raw)) return null
          return raw > 10_000_000_000 ? raw : raw * 1000
        }
        const coercePrice = (value: unknown): number | null => {
          const raw = Number(value)
          if (!Number.isFinite(raw)) return null
          if (raw < 0 || raw > 1.01) return null
          return raw
        }
        const pointTs = (entry: Record<string, unknown> | unknown[]): number => {
          if (Array.isArray(entry)) {
            const first = coerceTs(entry[0])
            if (first != null) return first
          } else {
            const ts = (
              coerceTs(entry.t)
              ?? coerceTs(entry.ts)
              ?? coerceTs(entry.time)
              ?? coerceTs(entry.timestamp)
            )
            if (ts != null) return ts
          }
          syntheticTs += 1
          return syntheticTs
        }
        const pointYes = (entry: Record<string, unknown> | unknown[]): number | null => {
          if (Array.isArray(entry)) {
            const start = coerceTs(entry[0]) != null ? 1 : 0
            return coercePrice(entry[start])
          }
          const direct = (
            coercePrice(entry.yes)
            ?? coercePrice(entry.y)
            ?? coercePrice(entry.idx_0)
            ?? coercePrice(entry.p)
            ?? coercePrice(entry.price)
          )
          if (direct != null) return direct
          const rawOutcomePrices = entry.outcome_prices
          if (Array.isArray(rawOutcomePrices) && rawOutcomePrices.length > 0) {
            return coercePrice(rawOutcomePrices[0])
          }
          return null
        }
        opportunity.markets.forEach((m, i) => {
          const hist = Array.isArray(m.price_history) ? m.price_history : []
          hist.forEach((rawEntry) => {
            const entry = (
              Array.isArray(rawEntry)
                ? rawEntry as unknown[]
                : (rawEntry as Record<string, unknown> | null)
            )
            if (!entry) return
            const yesVal = pointYes(entry)
            if (yesVal == null) return
            const ts = pointTs(entry)
            const existing = mergedByTs.get(ts) ?? { t: ts }
            existing[`idx_${i}`] = yesVal
            mergedByTs.set(ts, existing)
          })
        })
        const mergedHistory = Array.from(mergedByTs.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([, point]) => point)
        return buildOutcomeSparklineSeries(
          mergedHistory.length > 0 ? mergedHistory : undefined,
          fallbacks,
        )
      }
      return buildOutcomeSparklineSeries(
        market?.price_history,
        buildOutcomeFallbacks({
          labels: singleMarketOutcomes.labels,
          prices: singleMarketOutcomes.prices,
          yesPrice: marketYesPrice,
          noPrice: marketNoPrice,
          yesLabel: singleMarketOutcomes.labels[0] || t('opportunityCard.yes'),
          noLabel: singleMarketOutcomes.labels[1] || t('opportunityCard.no'),
          preferIndexedKeys: singleMarketOutcomes.labels.length > 2 || singleMarketOutcomes.prices.length > 2,
        }),
      )
    },
    [market, marketYesPrice, marketNoPrice, isMultiMarket, multiMarketOutcomes, singleMarketOutcomes, opportunity.markets, t],
  )
  const hasSparkline = (
    sparkSeries.length > 0
    || (
      isWeatherOpportunity
      && (
        marketProbability != null
        || modelProbability != null
        || (market && marketYesPrice >= 0 && marketYesPrice <= 1)
      )
    )
  )
  const nowSec = Math.floor(Date.now() / 1000)
  const livelineSeries = useMemo<LivelineSeries[]>(() => {
    // If there's sparkline data from price_history, use it directly
    if (sparkSeries.length > 0 && sparkSeries.some((row) => row.data.length >= 2)) {
      return sparkSeries
        .filter((row) => row.data.length >= 2)
        .map((row, index) => ({
        id: row.key,
        data: toTimeValueSeries(row.data, nowSec),
        value: row.latest ?? row.data[row.data.length - 1] ?? 0,
        color: SPARKLINE_COLORS[index % SPARKLINE_COLORS.length],
        label: row.label,
      }))
    }
    // For weather opportunities without price history, synthesise series
    // from market probability vs model consensus probability.
    if (isWeatherOpportunity) {
      const series: LivelineSeries[] = []
      const weatherMarketProbability = marketProbability ?? (market ? marketYesPrice : null)
      const mktData = weatherMarketProbability != null
        ? toTimeValueSeries([weatherMarketProbability, weatherMarketProbability], nowSec)
        : []
      if (mktData.length >= 2 && weatherMarketProbability != null) {
        series.push({
          id: 'market',
          data: mktData,
          value: weatherMarketProbability,
          color: SPARKLINE_COLORS[0],
          label: t('opportunityCard.field.market'),
        })
      }
      if (modelProbability != null) {
        const modelData = toTimeValueSeries([modelProbability, modelProbability], nowSec)
        if (modelData.length >= 2) {
          series.push({
            id: 'model',
            data: modelData,
            value: modelProbability,
            color: '#06b6d4', // cyan-500 for weather model
            label: t('opportunityCard.field.model'),
          })
        }
      }
      return series
    }
    // Generic fallback: just convert sparkSeries as-is
    return sparkSeries
      .filter((row) => row.data.length >= 2)
      .map((row, index) => ({
      id: row.key,
      data: toTimeValueSeries(row.data, nowSec),
      value: row.latest ?? row.data[row.data.length - 1] ?? 0,
      color: SPARKLINE_COLORS[index % SPARKLINE_COLORS.length],
      label: row.label,
    }))
  }, [sparkSeries, nowSec, isWeatherOpportunity, market, marketProbability, marketYesPrice, modelProbability, t])
  const primaryLivelineData = livelineSeries[0]?.data ?? []
  const primaryLivelineValue = livelineSeries[0]?.value ?? 0
  const livelineWindow = primaryLivelineData.length >= 2
    ? Math.max(
      600,
      primaryLivelineData[primaryLivelineData.length - 1].time - primaryLivelineData[0].time,
    )
    : 600

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
  const closeModal = () => setModalOpen(false)

  useEffect(() => {
    if (!isModalView) return
    setAiExpanded(true)
  }, [isModalView])

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
    <Card
      className={cn(
        "overflow-hidden relative group transition-all duration-200",
        !isModalView && "hover:shadow-lg hover:shadow-black/20 hover:border-border/80 cursor-pointer",
        isModalView && "w-[min(1100px,calc(100vw-2rem))] max-h-[90vh] overflow-hidden rounded-2xl border-border/70 bg-background shadow-[0_40px_120px_rgba(0,0,0,0.55)]",
        bgGradient && `bg-gradient-to-r ${bgGradient}`
      )}
      onClick={!isModalView ? () => setModalOpen(true) : undefined}
    >
      {/* Left accent bar */}
      <div className={cn("absolute left-0 top-0 bottom-0 w-1 rounded-l-lg transition-all", accentColor)} />

      {/* ── Modal Header Bar ── */}
      {isModalView && (
        <div className="border-b border-border/60 px-4 py-3 flex-shrink-0">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-1.5">
                <h3 className="text-sm font-semibold truncate max-w-[620px]" title={headerTitle}>
                  {headerTitle}
                </h3>
                <Badge variant="outline" className={cn("h-5 px-1.5 text-[10px]", STRATEGY_COLORS[opportunity.strategy])}>
                  {strategyLabel}
                </Badge>
                {opportunity.category && (
                  <Badge variant="outline" className="h-5 px-1.5 text-[10px] border-border/80 bg-muted/60 text-muted-foreground">
                    {opportunity.category}
                  </Badge>
                )}
              </div>
              <p className="mt-1 text-[11px] text-muted-foreground truncate max-w-[620px]">
                {headerSubtitle || opportunity.description || opportunity.title}
              </p>
            </div>
            <div className="flex items-center gap-1 flex-shrink-0">
              {!isSearch && (
                <BuyButton opportunity={opportunity} variant="compact" />
              )}
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-7 px-2 text-[11px]"
                onClick={(e) => { e.stopPropagation(); onCloseModal?.() }}
              >
                <Minimize2 className="w-3 h-3 mr-1" />
                {t('opportunityCard.close')}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* ── Card Body (scrollable in modal) ── */}
      <div className={cn(isModalView && "max-h-[calc(90vh-72px)] overflow-y-auto")}>

      <div className="pl-4 pr-3 py-2.5 space-y-2">
        {/* ── Row 1: Badges + ROI / Prices ── */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 flex-wrap min-w-0">
            {isWeatherOpportunity && weatherTargetDisplay && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-cyan-500/12 text-cyan-700 dark:text-cyan-200 border-cyan-500/25">
                <span className="inline-flex items-center gap-1">
                  <CalendarDays className="w-2.5 h-2.5" />
                  {weatherTargetDisplay.date}
                </span>
              </Badge>
            )}
            {!isSearch && !isWeatherOpportunity && (
              <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0", STRATEGY_COLORS[opportunity.strategy])}>
                {strategyLabel}
              </Badge>
            )}
            {isSearch && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-blue-500/10 text-blue-400 border-blue-500/20">
                {(opportunity as any).platform === 'kalshi' ? 'Kalshi' : 'Polymarket'}
              </Badge>
            )}
            {!isSearch && strategySdk && (
              <Badge
                variant="outline"
                className="max-w-[170px] truncate text-[9px] px-1.5 py-0 font-mono border-border/50 bg-muted/25 text-muted-foreground"
                title={t('opportunityCard.sdkTooltip', { sdk: strategySdk })}
              >
                SDK {strategySdk}
              </Badge>
            )}
            {opportunity.category && !isWeatherOpportunity && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-muted-foreground border-border/60">
                {opportunity.category}
              </Badge>
            )}
            {judgment && (
              <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 font-bold", RECOMMENDATION_COLORS[recommendation])}>
                {translateRecommendation(recommendation)}
              </Badge>
            )}
            {isPending && !judgeMutation.isPending && (
              <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                <RefreshCw className="w-2.5 h-2.5 animate-spin" /> {t('opportunityCard.queued')}
              </span>
            )}
          </div>
          <div className="text-right shrink-0">
            {isSearch && market ? (
              <>
                {/* Search results: show all market outcomes with real labels */}
                <div className="flex items-center justify-end gap-2 flex-wrap">
                  {sparkSeries.slice(0, isMultiMarket ? 4 : undefined).map((row, index) => (
                    <span
                      key={`${opportunity.id}-search-outcome-${row.key}`}
                      className={cn('text-sm font-bold font-data leading-none', SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length])}
                    >
                      {compactOutcomeLabel(row.label, isMultiMarket ? 10 : 12)} {safeFixed((row.latest ?? 0) * 100, 0)}¢
                    </span>
                  ))}
                  {isMultiMarket && sparkSeries.length > 4 && (
                    <span className="text-xs text-muted-foreground/60">+{sparkSeries.length - 4}</span>
                  )}
                </div>
                <p className="text-[10px] text-muted-foreground font-data mt-0.5">
                  {t('opportunityCard.volSuffix', { value: formatCompact(opportunity.volume ?? opportunity.min_liquidity) })}
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
                  {t('opportunityCard.netSuffix', { value: formatCompact(opportunity.net_profit) })}
                </p>
              </>
            )}
          </div>
        </div>

        {/* ── Row 2: Title ── */}
        <div className="min-w-0 space-y-0.5">
          <h3 className="text-sm font-medium text-foreground truncate leading-tight" title={headerTitle}>
            {headerTitle}
          </h3>
          {headerSubtitle && (
            <p className="text-[11px] text-muted-foreground truncate" title={headerSubtitle}>
              {headerSubtitle}
            </p>
          )}
        </div>

        {/* ── Row 3: Sparkline + Metrics ── */}
        <div className="space-y-1.5">
          {/* Sparkline */}
          {hasSparkline && primaryLivelineData.length >= 2 && (
            <div className="w-full">
              <Liveline
                data={primaryLivelineData}
                value={primaryLivelineValue}
                series={livelineSeries.length > 1 ? livelineSeries : undefined}
                color={SPARKLINE_COLORS[0]}
                theme={themeMode}
                window={livelineWindow}
                paused={livelineSeries.length <= 1}
                grid={isModalView}
                badge={false}
                fill={livelineSeries.length <= 1}
                pulse={false}
                momentum={isModalView}
                scrub={isModalView}
                seriesToggleCompact
                lerpSpeed={0.15}
                padding={isModalView
                  ? { top: 6, right: 6, bottom: 6, left: 6 }
                  : { top: 4, right: 4, bottom: 4, left: 4 }}
                formatValue={(v) => v.toFixed(2)}
                style={{ height: isModalView ? 132 : 64 }}
              />
              <div className={cn(
                "mt-0 flex flex-wrap gap-x-1.5 gap-y-0.5 px-0.5 text-[9px] font-data"
              )}>
                {sparkSeries.slice(0, isMultiMarket ? 6 : undefined).map((row, index) => (
                  <span
                    key={`${opportunity.id}-spark-${row.key}`}
                    className={cn('whitespace-nowrap', SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length])}
                  >
                    {compactOutcomeLabel(row.label, isMultiMarket ? 8 : 10)} {safeFixed(row.latest, 2)}
                  </span>
                ))}
                {isMultiMarket && sparkSeries.length > 6 && (
                  <span className="text-muted-foreground/60">+{sparkSeries.length - 6}</span>
                )}
              </div>
            </div>
          )}

          {/* Metrics Grid */}
          {isSearch ? (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
              <MiniMetric label={t('opportunityCard.metrics.volume')} value={formatCompact(opportunity.volume ?? opportunity.min_liquidity)} />
              <MiniMetric label={t('opportunityCard.metrics.liquidity')} value={formatCompact(opportunity.min_liquidity)} />
              <MiniMetric label={t('opportunityCard.metrics.ends')} value={timeUntil(opportunity.resolution_date, t)} />
              <MiniMetric
                label={t('opportunityCard.metrics.competitive')}
                value={market ? `${safeFixed(Math.abs(marketYesPrice - 0.5) * 200, 0)}%` : '—'}
                valueClass={market && Math.abs(marketYesPrice - 0.5) < 0.1 ? 'text-green-400' : undefined}
              />
            </div>
          ) : (
            isWeatherOpportunity ? (
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                <MiniMetric
                  label={t('opportunityCard.metrics.marketPx')}
                  value={marketProbability != null ? `${safeFixed(marketProbability * 100, 1)}¢` : '—'}
                />
                <MiniMetric
                  label={t('opportunityCard.metrics.modelPx')}
                  value={modelProbability != null ? `${safeFixed(modelProbability * 100, 1)}¢` : '—'}
                />
                <MiniMetric
                  label={t('opportunityCard.metrics.tempDelta')}
                  value={tempDelta != null ? `${tempDelta >= 0 ? '+' : ''}${safeFixed(tempDelta, 1)}°${tempUnit}` : '—'}
                  valueClass={tempDelta != null ? (tempDelta >= 0 ? 'text-cyan-700 dark:text-cyan-300' : 'text-orange-600 dark:text-orange-300') : undefined}
                />
                <MiniMetric
                  label={t('opportunityCard.metrics.sources')}
                  value={t('opportunityCard.srcCount', { n: weather?.source_count ?? weatherSources.length ?? 0 })}
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                <MiniMetric label={t('opportunityCard.metrics.cost')} value={formatCompact(opportunity.total_cost)} />
                <MiniMetric label={t('opportunityCard.metrics.liq')} value={formatCompact(opportunity.min_liquidity)} />
                <MiniMetric
                  label={t('opportunityCard.metrics.risk')}
                  value={`${safeFixed((opportunity.risk_score ?? 0) * 100, 0)}%`}
                  valueClass={riskColor}
                  bar={opportunity.risk_score}
                  barClass={riskBarColor}
                />
                <MiniMetric label={t('opportunityCard.metrics.maxPos')} value={formatCompact(opportunity.max_position_size)} />
              </div>
            )
          )}
        </div>

        {isWeatherOpportunity && (
          <div className="flex items-center justify-between rounded-md border border-cyan-600/20 dark:border-cyan-500/20 bg-cyan-500/[0.06] px-2 py-1">
            <span className="text-[10px] text-cyan-700 dark:text-cyan-300/90 font-medium truncate">
              {t('opportunityCard.mktVsConsensus', { mkt: formatTemp(marketImpliedTemp, tempUnit), consensus: formatTemp(consensusTemp, tempUnit) })}
            </span>
            <span className="text-[10px] font-data text-muted-foreground ml-2 shrink-0">
              {t('opportunityCard.edgeLabel', { value: signalEdgePercent != null ? `${signalEdgePercent >= 0 ? '+' : ''}${safeFixed(signalEdgePercent, 1)}%` : '—' })}
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
              <span title={t('opportunityCard.profitTooltip')}>P{safeFixed((judgment.profit_viability ?? 0) * 100, 0)}</span>
              <span title={t('opportunityCard.resolutionTooltip')}>R{safeFixed((judgment.resolution_safety ?? 0) * 100, 0)}</span>
              <span title={t('opportunityCard.executionTooltip')}>E{safeFixed((judgment.execution_feasibility ?? 0) * 100, 0)}</span>
              <span title={t('opportunityCard.efficiencyTooltip')}>M{safeFixed((judgment.market_efficiency ?? 0) * 100, 0)}</span>
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
              <Brain className="w-3 h-3" /> {t('opportunityCard.noAiAnalysis')}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); judgeMutation.mutate() }}
              disabled={judgeMutation.isPending}
              className="text-[10px] text-purple-400 hover:text-purple-300 font-medium transition-colors"
            >
              {judgeMutation.isPending ? t('opportunityCard.analyzing') : t('opportunityCard.analyze')}
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
            title={aiExpanded ? t('opportunityCard.clickToCollapse') : t('opportunityCard.clickToExpand')}
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
                  {sparkSeries.slice(0, isMultiMarket ? 4 : undefined).map((row, index) => (
                    <span
                      key={`${opportunity.id}-search-line-${row.key}`}
                      className={SPARKLINE_TEXT_CLASSES[index % SPARKLINE_TEXT_CLASSES.length]}
                    >
                      {compactOutcomeLabel(row.label, isMultiMarket ? 10 : 12)} {safeFixed(row.latest, 3)}
                    </span>
                  ))}
                  {isMultiMarket && sparkSeries.length > 4 && (
                    <span className="text-muted-foreground/60">+{sparkSeries.length - 4}</span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-1 truncate min-w-0">
              {opportunity.positions_to_take.slice(0, isMultiMarket ? 2 : 3).map((pos, i) => (
                <span key={i} className="inline-flex items-center gap-0.5 shrink-0">
                  {i > 0 && <span className="text-border">·</span>}
                  <span className={cn(
                    "font-data font-medium",
                    pos.outcome === 'YES' ? 'text-green-400/80' : 'text-red-400/80'
                  )}>
                    {pos.action} {pos.outcome}
                  </span>
                  {isMultiMarket && pos.market && (
                    <span className="text-foreground/50 font-data">{compactOutcomeLabel(pos.market, 12)}</span>
                  )}
                  <span className="font-data">@{safeFixed(pos.price, 2)}</span>
                </span>
              ))}
              {opportunity.positions_to_take.length > (isMultiMarket ? 2 : 3) && (
                <span className="text-muted-foreground/60">+{opportunity.positions_to_take.length - (isMultiMarket ? 2 : 3)}</span>
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
              <span title={t('opportunityCard.firstDetectedAgo', { time: timeAgo(firstDetectedTimestamp, t) })}>
                {timeAgo(freshnessTimestamp, t)}
              </span>
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
              title={t('opportunityCard.searchNewsTooltip')}
            >
              <Newspaper className="w-2.5 h-2.5" />
              {t('opportunityCard.newsButton')}
            </button>
          )}
        </div>

        {judgeMutation.error && (
          <div className="text-[10px] text-red-400">
            {(judgeMutation.error as any)?.code === 'ECONNABORTED'
              ? t('opportunityCard.analyzeFailedTimeout')
              : t('opportunityCard.analyzeFailedWithMessage', { message: (judgeMutation.error as Error).message })}
          </div>
        )}
      </div>

      {/* ── Expanded Details (modal only) ── */}
      {isModalView && (
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
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">{t('opportunityCard.sections.marketDetails')}</h4>
                  <div className="grid grid-cols-3 gap-3 text-xs">
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.outcomePrice', { outcome: compactOutcomeLabel(primaryOutcomeLabel, 16) })}</p>
                      <p className="font-data text-green-400">{safeFixed(marketYesPrice * 100, 1)}¢</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.outcomePrice', { outcome: compactOutcomeLabel(secondaryOutcomeLabel, 16) })}</p>
                      <p className="font-data text-red-400">{safeFixed(marketNoPrice * 100, 1)}¢</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.spread')}</p>
                      <p className="font-data text-foreground">{market ? safeFixed(Math.abs(1 - marketYesPrice - marketNoPrice) * 100, 1) : '—'}¢</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.volume')}</p>
                      <p className="font-data text-foreground">{formatCompact(opportunity.volume ?? 0)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.liquidity')}</p>
                      <p className="font-data text-foreground">{formatCompact(opportunity.min_liquidity)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.ends')}</p>
                      <p className="font-data text-foreground">{timeUntil(opportunity.resolution_date, t)}</p>
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
                          {t('opportunityCard.outcomeVolLiq', { outcomes: formatOutcomePriceSummary(mkt), vol: formatCompact((mkt as any).volume), liq: formatCompact((mkt as any).liquidity || mkt.liquidity) })}
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
                  <div className="bg-cyan-500/[0.06] rounded-lg p-3 border border-cyan-600/20 dark:border-cyan-500/20">
                    <h4 className="text-[10px] font-medium text-cyan-700 dark:text-cyan-300 mb-2 uppercase tracking-wider">{t('opportunityCard.sections.weatherIntelligence')}</h4>
                    <div className="grid grid-cols-3 gap-3 text-xs">
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.location')}</p>
                        <p className="font-data text-foreground truncate">{weather.location || '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.contract')}</p>
                        <p className="font-data text-foreground">{weatherContractLabel}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.target')}</p>
                        <p className="font-data text-foreground">{weatherTargetLabel}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.market')}</p>
                        <p className="font-data text-foreground">{marketProbability != null ? `${safeFixed(marketProbability * 100, 1)}%` : '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.model')}</p>
                        <p className="font-data text-cyan-700 dark:text-cyan-300">{modelProbability != null ? `${safeFixed(modelProbability * 100, 1)}%` : '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.edge')}</p>
                        <p className={cn(
                          "font-data",
                          signalEdgePercent != null && signalEdgePercent >= 0 ? 'text-green-400' : 'text-red-400'
                        )}>
                          {signalEdgePercent != null ? `${signalEdgePercent >= 0 ? '+' : ''}${safeFixed(signalEdgePercent, 1)}%` : '—'}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.marketImpliedTemp')}</p>
                        <p className="font-data text-foreground">{formatTemp(marketImpliedTemp, tempUnit)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.consensusTemp')}</p>
                        <p className="font-data text-cyan-700 dark:text-cyan-300">{formatTemp(consensusTemp, tempUnit)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.modelAgreement')}</p>
                        <p className="font-data text-foreground">{formatPct(weather.agreement)}</p>
                      </div>
                    </div>
                    {weatherSources.length > 0 && (
                      <div className="mt-3 space-y-1.5">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{t('opportunityCard.forecastSourcesCount', { n: weatherSources.length })}</p>
                        {weatherSources.map((src) => (
                          <div key={src.source_id} className="flex items-center justify-between gap-2 rounded-md border border-border/50 bg-muted/40 px-2.5 py-1.5">
                            <div className="min-w-0">
                              <p className="text-[11px] text-foreground truncate">
                                {src.provider}:{src.model}
                              </p>
                              <p className="text-[9px] text-muted-foreground font-data">
                                {t('opportunityCard.tempProbInline', { temp: formatTemp(tempUnit === 'C' ? (src.value_c ?? (src.value_f != null ? fToC(src.value_f) : null)) : (src.value_f ?? (src.value_c != null ? cToF(src.value_c) : null)), tempUnit), prob: formatPct(src.probability) })}
                              </p>
                            </div>
                            <span className="text-[10px] text-cyan-700 dark:text-cyan-300 font-data shrink-0">
                              {src.weight != null ? t('opportunityCard.weightShort', { value: safeFixed(src.weight * 100, 0) }) : t('opportunityCard.weightShortEmpty')}
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
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">{t('opportunityCard.sections.positions')}</h4>
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
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">{t('opportunityCard.sections.profitBreakdown')}</h4>
                  <div className="grid grid-cols-3 gap-3 text-xs">
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.cost')}</p>
                      <p className="font-data text-foreground">${safeFixed(opportunity.total_cost, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.payout')}</p>
                      <p className="font-data text-foreground">${safeFixed(opportunity.expected_payout, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.gross')}</p>
                      <p className="font-data text-foreground">${safeFixed(opportunity.gross_profit, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.fee2pct')}</p>
                      <p className="font-data text-red-400">-${safeFixed(opportunity.fee, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.net')}</p>
                      <p className="font-data text-green-400">${safeFixed(opportunity.net_profit, 4)}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground">{t('opportunityCard.field.roi')}</p>
                      <p className="font-data text-green-400">{safeFixed(opportunity.roi_percent, 2)}%</p>
                    </div>
                  </div>
                </div>

                {/* Markets */}
                <div>
                  <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">{t('opportunityCard.sections.marketsCount', { n: opportunity.markets.length })}</h4>
                  <div className="space-y-1.5">
                    {opportunity.markets.map((mkt, idx) => {
                      const marketLink = opportunityLinks.marketLinks[idx]
                      const url = marketLink?.url || null
                      return (
                        <div key={idx} className="flex items-center justify-between bg-muted/50 rounded-md px-2.5 py-1.5 gap-2">
                          <div className="min-w-0 flex-1">
                            <p className="text-[11px] text-foreground/80 truncate">{mkt.question}</p>
                            <p className="text-[9px] text-muted-foreground font-data mt-0.5">
                              {t('opportunityCard.outcomeLiq', { outcomes: formatOutcomePriceSummary(mkt), liq: formatCompact(mkt.liquidity) })}
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
                <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">{t('opportunityCard.sections.riskFactors')}</h4>
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
                <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">{t('opportunityCard.sections.resolution')}</h4>
                {resolutions.map((r: any, i: number) => (
                  <div key={i} className="bg-muted/30 rounded-md p-2 space-y-1 border border-border/50">
                    <div className="flex items-center gap-1.5">
                      <Shield className="w-2.5 h-2.5 text-muted-foreground" />
                      <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0', RECOMMENDATION_COLORS[r.recommendation])}>
                        {translateRecommendation(r.recommendation)}
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

          </div>
        </>
      )}

      </div>
    </Card>
      {!isModalView && typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {modalOpen && (
            <motion.div
              key={`opportunity-modal-${opportunity.id}`}
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
                aria-label={t('opportunityCard.expandedOpportunityAria', { title: headerTitle })}
                initial={{ scale: 0.94, opacity: 0, y: 22 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 14 }}
                transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}
              >
                <OpportunityCard
                  opportunity={opportunity}
                  onOpenCopilot={onOpenCopilot}
                  onSearchNews={onSearchNews}
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

export default memo(OpportunityCard)

export { STRATEGY_COLORS, STRATEGY_NAMES, STRATEGY_ABBREV, RECOMMENDATION_COLORS, ACCENT_BAR_COLORS }
