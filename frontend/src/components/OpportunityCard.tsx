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
  Layers,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Opportunity, judgeOpportunity } from '../services/api'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Separator } from './ui/separator'
import Sparkline from './Sparkline'

// ─── Constants ────────────────────────────────────────────

const STRATEGY_COLORS: Record<string, string> = {
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

// ─── Utilities ────────────────────────────────────────────

function generatePriceHistory(id: string, currentPrice: number, points = 20): number[] {
  let seed = 0
  for (let i = 0; i < id.length; i++) {
    seed = ((seed << 5) - seed + id.charCodeAt(i)) | 0
  }
  const next = () => {
    seed = (seed * 16807) % 2147483647
    return (seed & 0x7fffffff) / 2147483647
  }
  const data: number[] = []
  let price = currentPrice + (next() - 0.5) * 0.12
  for (let i = 0; i < points - 1; i++) {
    price += (next() - 0.48) * 0.015
    price = Math.max(0.01, Math.min(0.99, price))
    data.push(price)
  }
  data.push(currentPrice)
  return data
}

export function timeAgo(dateStr: string): string {
  const diffMs = Date.now() - new Date(dateStr).getTime()
  const sec = Math.floor(diffMs / 1000)
  if (sec < 60) return `${sec}s`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h`
  return `${Math.floor(hr / 24)}d`
}

export function formatCompact(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `$${(n / 1000).toFixed(1)}K`
  if (n >= 1000) return `$${(n / 1000).toFixed(1)}K`
  if (n >= 100) return `$${n.toFixed(0)}`
  if (n >= 1) return `$${n.toFixed(2)}`
  return `$${n.toFixed(4)}`
}

// ─── Props ────────────────────────────────────────────────

interface Props {
  opportunity: Opportunity
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}

// ─── Main Component ───────────────────────────────────────

export default function OpportunityCard({ opportunity, onExecute, onOpenCopilot }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [aiExpanded, setAiExpanded] = useState(false)
  const queryClient = useQueryClient()

  // AI analysis
  const inlineAnalysis = opportunity.ai_analysis
  const judgeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await judgeOpportunity({ opportunity_id: opportunity.id })
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
    },
  })
  const isPending = inlineAnalysis?.recommendation === 'pending'
  const judgment = judgeMutation.data || (inlineAnalysis && !isPending ? inlineAnalysis : null)
  const resolutions = inlineAnalysis?.resolution_analyses || []
  const recommendation = judgment?.recommendation || (isPending ? 'pending' : '')

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
  const sparkData = useMemo(() => {
    if (!market) return { yes: [], no: [] }
    const id = opportunity.stable_id || opportunity.id
    return {
      yes: generatePriceHistory(id, market.yes_price),
      no: generatePriceHistory(id + '_no', market.no_price),
    }
  }, [opportunity.stable_id, opportunity.id, market?.yes_price, market?.no_price])

  // Accent bar color
  const accentColor = recommendation ? (ACCENT_BAR_COLORS[recommendation] || 'bg-border') : 'bg-border/50'
  const bgGradient = recommendation ? (CARD_BG_GRADIENT[recommendation] || '') : ''

  // Platform URLs
  const polyMarket = opportunity.markets.find((m: any) => !m.platform || m.platform === 'polymarket')
  const polyUrl = polyMarket
    ? (opportunity.event_slug
        ? `https://polymarket.com/event/${opportunity.event_slug}`
        : polyMarket.slug
          ? `https://polymarket.com/event/${polyMarket.slug}`
          : null)
    : null
  const kalshiMarket = opportunity.markets.find((m: any) => m.platform === 'kalshi')
  const kalshiUrl = kalshiMarket ? `https://kalshi.com/markets/${kalshiMarket.id}` : null

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
        {/* ── Row 1: Badges + ROI ── */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 flex-wrap min-w-0">
            <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0", STRATEGY_COLORS[opportunity.strategy])}>
              {STRATEGY_NAMES[opportunity.strategy] || opportunity.strategy}
            </Badge>
            {opportunity.category && (
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
            <div className="flex items-center gap-1 justify-end">
              <TrendingUp className={cn("w-3.5 h-3.5", roiPositive ? "text-green-400" : "text-red-400")} />
              <span className={cn(
                "text-base font-bold font-data leading-none",
                roiPositive ? "text-green-400 data-glow-green" : "text-red-400 data-glow-red"
              )}>
                {roiPositive ? '+' : ''}{opportunity.roi_percent.toFixed(2)}%
              </span>
            </div>
            <p className="text-[10px] text-muted-foreground font-data mt-0.5">
              {formatCompact(opportunity.net_profit)} net
            </p>
          </div>
        </div>

        {/* ── Row 2: Title ── */}
        <h3 className="text-sm font-medium text-foreground truncate leading-tight" title={opportunity.title}>
          {opportunity.title}
        </h3>

        {/* ── Row 3: Sparkline + Metrics ── */}
        <div className="flex items-stretch gap-3">
          {/* Sparkline */}
          {market && sparkData.yes.length >= 2 && (
            <div className="shrink-0">
              <Sparkline
                data={sparkData.yes}
                data2={sparkData.no}
                width={96}
                height={40}
                color="#22c55e"
                color2="#ef4444"
                lineWidth={1.5}
                showDots
              />
              <div className="flex justify-between text-[9px] text-muted-foreground font-data mt-0.5 px-0.5">
                <span className="text-green-400/70">Y {market.yes_price.toFixed(2)}</span>
                <span className="text-red-400/70">N {market.no_price.toFixed(2)}</span>
              </div>
            </div>
          )}

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 flex-1 min-w-0">
            <MiniMetric label="Cost" value={formatCompact(opportunity.total_cost)} />
            <MiniMetric label="Liq" value={formatCompact(opportunity.min_liquidity)} />
            <MiniMetric
              label="Risk"
              value={`${(opportunity.risk_score * 100).toFixed(0)}%`}
              valueClass={riskColor}
              bar={opportunity.risk_score}
              barClass={riskBarColor}
            />
            <MiniMetric label="Max Pos" value={formatCompact(opportunity.max_position_size)} />
          </div>
        </div>

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
              {(judgment.overall_score * 100).toFixed(0)}
            </span>
            <Separator orientation="vertical" className="h-3 bg-purple-500/20" />
            <div className="flex gap-1.5 text-[9px] font-data text-muted-foreground shrink-0">
              <span title="Profit">P{(judgment.profit_viability * 100).toFixed(0)}</span>
              <span title="Resolution">R{(judgment.resolution_safety * 100).toFixed(0)}</span>
              <span title="Execution">E{(judgment.execution_feasibility * 100).toFixed(0)}</span>
              <span title="Efficiency">M{(judgment.market_efficiency * 100).toFixed(0)}</span>
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
                <span className="font-data">@{pos.price.toFixed(2)}</span>
              </span>
            ))}
            {opportunity.positions_to_take.length > 3 && (
              <span className="text-muted-foreground/60">+{opportunity.positions_to_take.length - 3}</span>
            )}
          </div>
          <div className="flex items-center gap-1.5 ml-auto shrink-0">
            <span className="flex items-center gap-0.5">
              <Layers className="w-2.5 h-2.5" />
              {opportunity.markets.length}
            </span>
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
            Analysis failed: {(judgeMutation.error as Error).message}
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

            {/* Positions to Take */}
            <div>
              <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">Positions</h4>
              <div className="space-y-1.5">
                {opportunity.positions_to_take.map((pos, idx) => {
                  const platform = (pos as any).platform
                  return (
                    <div key={idx} className="flex items-center justify-between bg-muted/50 rounded-md px-2.5 py-1.5">
                      <div className="flex items-center gap-1.5">
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
                        <span className="text-[11px] text-foreground/70 truncate">{pos.market}</span>
                      </div>
                      <span className="font-data text-xs text-foreground">${pos.price.toFixed(4)}</span>
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
                  <p className="font-data text-foreground">${opportunity.total_cost.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground">Payout</p>
                  <p className="font-data text-foreground">${opportunity.expected_payout.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground">Gross</p>
                  <p className="font-data text-foreground">${opportunity.gross_profit.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground">Fee (2%)</p>
                  <p className="font-data text-red-400">-${opportunity.fee.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground">Net</p>
                  <p className="font-data text-green-400">${opportunity.net_profit.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground">ROI</p>
                  <p className="font-data text-green-400">{opportunity.roi_percent.toFixed(2)}%</p>
                </div>
              </div>
            </div>

            {/* Markets */}
            <div>
              <h4 className="text-[10px] font-medium text-muted-foreground mb-1.5 uppercase tracking-wider">Markets ({opportunity.markets.length})</h4>
              <div className="space-y-1.5">
                {opportunity.markets.map((mkt, idx) => {
                  const isKalshi = (mkt as any).platform === 'kalshi'
                  const url = isKalshi
                    ? `https://kalshi.com/markets/${mkt.id}`
                    : mkt.slug
                      ? `https://polymarket.com/event/${mkt.slug}`
                      : `https://polymarket.com/event/${mkt.id}`
                  return (
                    <div key={idx} className="flex items-center justify-between bg-muted/50 rounded-md px-2.5 py-1.5 gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-[11px] text-foreground/80 truncate">{mkt.question}</p>
                        <p className="text-[9px] text-muted-foreground font-data mt-0.5">
                          Y:{mkt.yes_price.toFixed(3)} N:{mkt.no_price.toFixed(3)} Liq:{formatCompact(mkt.liquidity)}
                        </p>
                      </div>
                      <a
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  )
                })}
              </div>
            </div>

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
                        C:{(r.clarity_score * 100).toFixed(0)} R:{(r.risk_score * 100).toFixed(0)}
                      </span>
                    </div>
                    {r.summary && <p className="text-[10px] text-muted-foreground">{r.summary}</p>}
                  </div>
                ))}
              </div>
            )}

            {/* Execute Button */}
            {onExecute && (
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

export { STRATEGY_COLORS, STRATEGY_NAMES, STRATEGY_ABBREV, RECOMMENDATION_COLORS, ACCENT_BAR_COLORS, generatePriceHistory }
