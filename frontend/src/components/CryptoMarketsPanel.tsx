import { useState, useEffect, useMemo, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  AlertCircle,
  Play,
  Brain,
  Clock,
  Zap,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Activity,
  BarChart3,
  Shield,
  Target,
  MessageCircle,
  DollarSign,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  getOpportunities,
  judgeOpportunity,
  judgeOpportunitiesBulk,
  Opportunity,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Separator } from './ui/separator'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import Sparkline from './Sparkline'
import AnimatedNumber from './AnimatedNumber'

// ─── Constants ────────────────────────────────────────────

const ASSETS = ['ALL', 'BTC', 'ETH', 'SOL', 'XRP'] as const
type Asset = typeof ASSETS[number]

const TIMEFRAMES = ['ALL', '5min', '15min', '1hr'] as const
type Timeframe = typeof TIMEFRAMES[number]

const SUB_STRATEGY_LABELS: Record<string, { label: string; color: string; desc: string }> = {
  pure_arb: { label: 'PURE ARB', color: 'text-green-400 bg-green-500/15 border-green-500/25', desc: 'YES + NO < $1.00 = guaranteed profit' },
  dump_hedge: { label: 'DUMP HEDGE', color: 'text-amber-400 bg-amber-500/15 border-amber-500/25', desc: 'Buy dumped side + hedge opposite' },
  pre_placed_limits: { label: 'LIMIT ORDER', color: 'text-blue-400 bg-blue-500/15 border-blue-500/25', desc: 'Pre-placed limits on thin markets' },
}

const ASSET_COLORS: Record<string, string> = {
  BTC: 'text-orange-400',
  ETH: 'text-blue-400',
  SOL: 'text-purple-400',
  XRP: 'text-cyan-400',
}

const ASSET_BG: Record<string, string> = {
  BTC: 'bg-orange-500/10 border-orange-500/20',
  ETH: 'bg-blue-500/10 border-blue-500/20',
  SOL: 'bg-purple-500/10 border-purple-500/20',
  XRP: 'bg-cyan-500/10 border-cyan-500/20',
}

const RECOMMENDATION_COLORS: Record<string, string> = {
  strong_execute: 'text-green-400 bg-green-500/20 border-green-500/30',
  execute: 'text-green-400 bg-green-500/15 border-green-500/20',
  review: 'text-yellow-400 bg-yellow-500/15 border-yellow-500/20',
  skip: 'text-red-400 bg-red-500/15 border-red-500/20',
  strong_skip: 'text-red-400 bg-red-500/20 border-red-500/30',
  pending: 'text-muted-foreground bg-muted/30 border-muted-foreground/20',
}

// ─── Helpers ─────────────────────────────────────────────

function detectAsset(title: string): string {
  const t = title.toLowerCase()
  if (t.includes('btc') || t.includes('bitcoin')) return 'BTC'
  if (t.includes('eth') || t.includes('ethereum')) return 'ETH'
  if (t.includes('sol') || t.includes('solana')) return 'SOL'
  if (t.includes('xrp') || t.includes('ripple')) return 'XRP'
  return 'BTC'
}

function detectTimeframe(title: string): string {
  const t = title.toLowerCase()
  // Check 15m before 5m since "15min" contains the substring "5m"
  if (t.includes('15m') || t.includes('15 min') || t.includes('15-min') || t.includes('quarter')) return '15min'
  if (t.includes('5m') || t.includes('5 min') || t.includes('5-min')) return '5min'
  if (t.includes('1h') || t.includes('1 hour') || t.includes('1-hour') || t.includes('60m') || t.includes('hourly')) return '1hr'
  return '15min'
}

function detectSubStrategy(opp: Opportunity): string {
  const desc = (opp.description || '').toLowerCase()
  const title = (opp.title || '').toLowerCase()
  const combined = desc + ' ' + title
  if (combined.includes('pure arb') || combined.includes('pure_arb') || combined.includes('guaranteed')) return 'pure_arb'
  if (combined.includes('dump') || combined.includes('hedge')) return 'dump_hedge'
  if (combined.includes('limit') || combined.includes('pre-place') || combined.includes('thin')) return 'pre_placed_limits'
  // Infer from prices: if combined cost is close to 1.0, likely pure arb
  const market = opp.markets[0]
  if (market) {
    const combined_cost = market.yes_price + market.no_price
    if (combined_cost < 0.99) return 'pure_arb'
  }
  return 'pure_arb'
}

function getResolutionCountdown(opp: Opportunity): { text: string; urgency: 'low' | 'medium' | 'high' | 'expired' } {
  if (!opp.resolution_date) return { text: 'No expiry', urgency: 'low' }
  const diff = new Date(opp.resolution_date).getTime() - Date.now()
  if (diff <= 0) return { text: 'EXPIRED', urgency: 'expired' }
  const totalSec = Math.floor(diff / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  if (min >= 60) {
    const hr = Math.floor(min / 60)
    const remainMin = min % 60
    return { text: `${hr}h ${remainMin}m`, urgency: hr > 1 ? 'low' : 'medium' }
  }
  if (min > 5) return { text: `${min}m ${sec}s`, urgency: 'medium' }
  return { text: `${min}m ${sec}s`, urgency: 'high' }
}

function formatUsd(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  if (n >= 1) return `$${n.toFixed(2)}`
  return `$${n.toFixed(4)}`
}

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

// ─── Props ───────────────────────────────────────────────

interface Props {
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}

// ─── Sub-components ──────────────────────────────────────

function PriceBar({ label, price, color, bgColor }: { label: string; price: number; color: string; bgColor: string }) {
  return (
    <div className="flex items-center gap-2 flex-1">
      <span className={cn("text-[10px] font-bold font-data w-6 shrink-0", color)}>{label}</span>
      <div className="flex-1 h-5 bg-muted/30 rounded-sm overflow-hidden relative">
        <div
          className={cn("h-full rounded-sm transition-all duration-500", bgColor)}
          style={{ width: `${Math.min(price * 100, 100)}%` }}
        />
        <span className={cn("absolute inset-0 flex items-center justify-center text-[10px] font-bold font-data", color)}>
          {price.toFixed(3)}
        </span>
      </div>
    </div>
  )
}

function SpreadGauge({ yesPrice, noPrice }: { yesPrice: number; noPrice: number }) {
  const combined = yesPrice + noPrice
  const spread = 1 - combined
  const isProfitable = spread > 0.02 // After 2% fee
  const spreadPct = (spread * 100).toFixed(2)

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-muted-foreground uppercase tracking-wider font-medium">Spread</span>
        <span className={cn("font-data font-bold", isProfitable ? 'text-green-400' : spread > 0 ? 'text-yellow-400' : 'text-red-400')}>
          {spread > 0 ? '+' : ''}{spreadPct}%
        </span>
      </div>
      <div className="h-1.5 bg-muted/30 rounded-full overflow-hidden relative">
        {/* Fee zone */}
        <div className="absolute right-0 top-0 h-full bg-red-500/20" style={{ width: '2%' }} />
        {/* Spread fill */}
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            isProfitable ? 'bg-green-500/70' : spread > 0 ? 'bg-yellow-500/70' : 'bg-red-500/70'
          )}
          style={{ width: `${Math.max(Math.min(Math.abs(spread) * 100 * 5, 100), 2)}%` }}
        />
      </div>
      <div className="flex justify-between text-[9px] text-muted-foreground/60 font-data">
        <span>Combined: {combined.toFixed(3)}</span>
        <span>Fee: 2%</span>
      </div>
    </div>
  )
}

function CountdownTimer({ opp }: { opp: Opportunity }) {
  const [countdown, setCountdown] = useState(() => getResolutionCountdown(opp))

  useEffect(() => {
    if (!opp.resolution_date) return
    const iv = setInterval(() => {
      setCountdown(getResolutionCountdown(opp))
    }, 1000)
    return () => clearInterval(iv)
  }, [opp.resolution_date])

  const urgencyColors: Record<string, string> = {
    low: 'text-muted-foreground',
    medium: 'text-yellow-400',
    high: 'text-red-400 animate-pulse',
    expired: 'text-red-500',
  }

  return (
    <div className={cn("flex items-center gap-1 text-[10px] font-data", urgencyColors[countdown.urgency])}>
      <Clock className="w-3 h-3" />
      <span className="font-bold">{countdown.text}</span>
    </div>
  )
}

// ─── Market Card ─────────────────────────────────────────

function CryptoMarketCard({
  opportunity,
  onExecute,
  onOpenCopilot,
}: {
  opportunity: Opportunity
  onExecute?: (opp: Opportunity) => void
  onOpenCopilot?: (opp: Opportunity) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const queryClient = useQueryClient()

  const asset = detectAsset(opportunity.title)
  const timeframe = detectTimeframe(opportunity.title)
  const subStrategy = detectSubStrategy(opportunity)
  const subStrategyInfo = SUB_STRATEGY_LABELS[subStrategy] || SUB_STRATEGY_LABELS.pure_arb
  const market = opportunity.markets[0]
  const roiPositive = opportunity.roi_percent >= 0

  const analysis = opportunity.ai_analysis
  const recommendation = analysis?.recommendation || ''
  const isPending = recommendation === 'pending'

  const judgeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await judgeOpportunity({ opportunity_id: opportunity.id })
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['crypto-opportunities'] })
    },
  })

  const sparkYes = useMemo(() => {
    if (!market) return []
    return generatePriceHistory(opportunity.stable_id || opportunity.id, market.yes_price)
  }, [opportunity.stable_id, opportunity.id, market?.yes_price])

  const sparkNo = useMemo(() => {
    if (!market) return []
    return generatePriceHistory((opportunity.stable_id || opportunity.id) + '_no', market.no_price)
  }, [opportunity.stable_id, opportunity.id, market?.no_price])

  const polyUrl = opportunity.event_slug
    ? `https://polymarket.com/event/${opportunity.event_slug}`
    : market?.slug
      ? `https://polymarket.com/event/${market.slug}`
      : null

  return (
    <Card className={cn(
      "overflow-hidden relative group transition-all duration-200",
      "hover:shadow-lg hover:shadow-black/20 hover:border-border/80",
      recommendation === 'strong_execute' && 'ring-1 ring-green-500/20',
      recommendation === 'execute' && 'ring-1 ring-green-500/10',
    )}>
      {/* Asset color accent bar */}
      <div className={cn(
        "absolute left-0 top-0 bottom-0 w-1 rounded-l-lg",
        asset === 'BTC' ? 'bg-orange-400' : asset === 'ETH' ? 'bg-blue-400' : asset === 'SOL' ? 'bg-purple-400' : 'bg-cyan-400'
      )} />

      <div className="pl-4 pr-3 py-3 space-y-2.5">
        {/* ── Row 1: Asset + Timeframe + Sub-Strategy + ROI ── */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 flex-wrap min-w-0">
            {/* Asset badge */}
            <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 font-bold", ASSET_BG[asset], ASSET_COLORS[asset])}>
              {asset}
            </Badge>
            {/* Timeframe */}
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-muted-foreground border-border/60 font-data">
              {timeframe === '5min' ? '5m' : timeframe === '15min' ? '15m' : '1h'}
            </Badge>
            {/* Sub-strategy */}
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <Badge variant="outline" className={cn("text-[9px] px-1.5 py-0 font-bold", subStrategyInfo.color)}>
                  {subStrategyInfo.label}
                </Badge>
              </TooltipTrigger>
              <TooltipContent side="top" className="text-xs max-w-[200px]">{subStrategyInfo.desc}</TooltipContent>
            </Tooltip>
            {/* AI recommendation */}
            {analysis && !isPending && (
              <Badge variant="outline" className={cn("text-[9px] px-1.5 py-0 font-bold", RECOMMENDATION_COLORS[recommendation])}>
                {recommendation.replace('_', ' ').toUpperCase()}
              </Badge>
            )}
            {isPending && (
              <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                <RefreshCw className="w-2.5 h-2.5 animate-spin" /> queued
              </span>
            )}
          </div>
          {/* ROI */}
          <div className="text-right shrink-0">
            <div className="flex items-center gap-1 justify-end">
              {roiPositive ? (
                <TrendingUp className="w-3.5 h-3.5 text-green-400" />
              ) : (
                <TrendingDown className="w-3.5 h-3.5 text-red-400" />
              )}
              <span className={cn(
                "text-base font-bold font-data leading-none",
                roiPositive ? "text-green-400" : "text-red-400"
              )}>
                {roiPositive ? '+' : ''}{opportunity.roi_percent.toFixed(2)}%
              </span>
            </div>
            <p className="text-[10px] text-muted-foreground font-data mt-0.5">
              {formatUsd(opportunity.net_profit)} net
            </p>
          </div>
        </div>

        {/* ── Row 2: Title + Countdown ── */}
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-medium text-foreground truncate leading-tight flex-1" title={opportunity.title}>
            {opportunity.title}
          </h3>
          <CountdownTimer opp={opportunity} />
        </div>

        {/* ── Row 3: Price visualization ── */}
        {market && (
          <div className="grid grid-cols-[1fr_auto] gap-3 items-center">
            {/* YES/NO price bars */}
            <div className="space-y-1.5">
              <PriceBar label="YES" price={market.yes_price} color="text-green-400" bgColor="bg-green-500/40" />
              <PriceBar label="NO" price={market.no_price} color="text-red-400" bgColor="bg-red-500/40" />
            </div>
            {/* Mini sparkline */}
            {sparkYes.length >= 2 && (
              <div className="shrink-0">
                <Sparkline
                  data={sparkYes}
                  data2={sparkNo}
                  width={72}
                  height={36}
                  color="#22c55e"
                  color2="#ef4444"
                  lineWidth={1.5}
                  showDots
                />
              </div>
            )}
          </div>
        )}

        {/* ── Row 4: Spread gauge ── */}
        {market && (
          <SpreadGauge yesPrice={market.yes_price} noPrice={market.no_price} />
        )}

        {/* ── Row 5: Metrics row ── */}
        <div className="grid grid-cols-4 gap-2">
          <div className="text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Cost</div>
            <div className="text-xs font-data font-bold text-foreground">{formatUsd(opportunity.total_cost)}</div>
          </div>
          <div className="text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Payout</div>
            <div className="text-xs font-data font-bold text-foreground">{formatUsd(opportunity.expected_payout)}</div>
          </div>
          <div className="text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Liq</div>
            <div className="text-xs font-data font-bold text-foreground">{formatUsd(opportunity.min_liquidity)}</div>
          </div>
          <div className="text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Risk</div>
            <div className={cn(
              "text-xs font-data font-bold",
              opportunity.risk_score < 0.3 ? 'text-green-400' : opportunity.risk_score < 0.6 ? 'text-yellow-400' : 'text-red-400'
            )}>
              {(opportunity.risk_score * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* ── Row 6: AI Score bar (if analyzed) ── */}
        {analysis && !isPending && analysis.overall_score > 0 && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-muted-foreground flex items-center gap-1">
                <Brain className="w-3 h-3" /> AI Score
              </span>
              <span className="font-data font-bold text-foreground">{(analysis.overall_score * 100).toFixed(0)}/100</span>
            </div>
            <div className="h-1 bg-muted/30 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-500",
                  analysis.overall_score >= 0.7 ? 'bg-green-500' : analysis.overall_score >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                )}
                style={{ width: `${analysis.overall_score * 100}%` }}
              />
            </div>
            {/* Sub-scores */}
            <div className="grid grid-cols-4 gap-1 mt-1">
              {[
                { label: 'Profit', value: analysis.profit_viability, icon: DollarSign },
                { label: 'Safety', value: analysis.resolution_safety, icon: Shield },
                { label: 'Exec', value: analysis.execution_feasibility, icon: Target },
                { label: 'Efficiency', value: analysis.market_efficiency, icon: BarChart3 },
              ].map(({ label, value, icon: Icon }) => (
                <Tooltip key={label} delayDuration={0}>
                  <TooltipTrigger asChild>
                    <div className="text-center cursor-default">
                      <Icon className="w-2.5 h-2.5 mx-auto text-muted-foreground/50 mb-0.5" />
                      <div className="h-0.5 bg-muted/30 rounded-full overflow-hidden">
                        <div
                          className={cn(
                            "h-full rounded-full",
                            value >= 0.7 ? 'bg-green-500/70' : value >= 0.4 ? 'bg-yellow-500/70' : 'bg-red-500/70'
                          )}
                          style={{ width: `${(value || 0) * 100}%` }}
                        />
                      </div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="text-xs">{label}: {((value || 0) * 100).toFixed(0)}%</TooltipContent>
                </Tooltip>
              ))}
            </div>
          </div>
        )}

        {/* ── Row 7: Actions ── */}
        <div className="flex items-center gap-2 pt-0.5">
          <Button
            size="sm"
            className="h-7 text-xs gap-1.5 flex-1 bg-green-600 hover:bg-green-500 text-white"
            onClick={() => onExecute?.(opportunity)}
          >
            <Play className="w-3 h-3" />
            Execute
          </Button>
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant="outline"
                className="h-7 w-7 p-0"
                onClick={() => onOpenCopilot?.(opportunity)}
              >
                <MessageCircle className="w-3.5 h-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="text-xs">AI Copilot</TooltipContent>
          </Tooltip>
          {!analysis && !isPending && (
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 w-7 p-0"
                  onClick={() => judgeMutation.mutate()}
                  disabled={judgeMutation.isPending}
                >
                  {judgeMutation.isPending ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Brain className="w-3.5 h-3.5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs">AI Analyze</TooltipContent>
            </Tooltip>
          )}
          {polyUrl && (
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <a href={polyUrl} target="_blank" rel="noopener noreferrer">
                  <Button size="sm" variant="outline" className="h-7 w-7 p-0">
                    <ExternalLink className="w-3.5 h-3.5" />
                  </Button>
                </a>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs">View on Polymarket</TooltipContent>
            </Tooltip>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0 ml-auto"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </Button>
        </div>

        {/* ── Expanded details ── */}
        {expanded && (
          <div className="space-y-2 pt-1">
            <Separator className="opacity-30" />
            {/* Description */}
            {opportunity.description && (
              <p className="text-xs text-muted-foreground leading-relaxed">{opportunity.description}</p>
            )}
            {/* Positions to take */}
            {opportunity.positions_to_take.length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Positions</span>
                {opportunity.positions_to_take.map((pos, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs font-data bg-muted/20 rounded px-2 py-1">
                    <Badge variant="outline" className={cn(
                      "text-[9px] px-1 py-0",
                      pos.outcome === 'yes' ? 'text-green-400 border-green-500/20' : 'text-red-400 border-red-500/20'
                    )}>
                      {pos.action.toUpperCase()} {pos.outcome.toUpperCase()}
                    </Badge>
                    <span className="text-muted-foreground truncate flex-1">{pos.market}</span>
                    <span className="font-bold">@ {pos.price.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
            {/* Risk factors */}
            {opportunity.risk_factors.length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Risk Factors</span>
                {opportunity.risk_factors.map((rf, i) => (
                  <div key={i} className="flex items-start gap-1.5 text-[11px] text-muted-foreground">
                    <AlertCircle className="w-3 h-3 text-yellow-400/60 mt-0.5 shrink-0" />
                    <span>{rf}</span>
                  </div>
                ))}
              </div>
            )}
            {/* AI reasoning */}
            {analysis?.reasoning && (
              <div className="space-y-1">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium flex items-center gap-1">
                  <Brain className="w-3 h-3" /> AI Analysis
                </span>
                <p className="text-xs text-muted-foreground leading-relaxed bg-muted/10 rounded p-2 border border-border/30">
                  {analysis.reasoning}
                </p>
              </div>
            )}
            {/* Timing info */}
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground font-data">
              <span>Detected: {new Date(opportunity.detected_at).toLocaleTimeString()}</span>
              {opportunity.resolution_date && (
                <span>Resolves: {new Date(opportunity.resolution_date).toLocaleTimeString()}</span>
              )}
              <span>Max size: {formatUsd(opportunity.max_position_size)}</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

// ─── Main Component ──────────────────────────────────────

export default function CryptoMarketsPanel({ onExecute, onOpenCopilot }: Props) {
  const queryClient = useQueryClient()
  const { isConnected } = useWebSocket('/ws')
  const [selectedAsset, setSelectedAsset] = useState<Asset>('ALL')
  const [selectedTimeframe, setSelectedTimeframe] = useState<Timeframe>('ALL')
  const [sortBy, setSortBy] = useState<'roi' | 'profit' | 'liquidity' | 'risk' | 'ai_score' | 'time'>('roi')
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc')

  // Fetch crypto-specific opportunities
  const { data: oppsData, isLoading } = useQuery({
    queryKey: ['crypto-opportunities'],
    queryFn: () => getOpportunities({
      strategy: 'btc_eth_highfreq',
      sort_by: 'roi',
      sort_dir: 'desc',
      limit: 100,
      offset: 0,
    }),
    refetchInterval: isConnected ? 15000 : 8000, // Faster refresh for high-freq
  })

  const allOpportunities = oppsData?.opportunities || []

  // Client-side filtering and sorting
  const filtered = useMemo(() => {
    let result = [...allOpportunities]

    // Filter by asset
    if (selectedAsset !== 'ALL') {
      result = result.filter(opp => detectAsset(opp.title) === selectedAsset)
    }

    // Filter by timeframe
    if (selectedTimeframe !== 'ALL') {
      result = result.filter(opp => detectTimeframe(opp.title) === selectedTimeframe)
    }

    // Sort
    const dir = sortDir === 'desc' ? -1 : 1
    result.sort((a, b) => {
      switch (sortBy) {
        case 'roi': return (a.roi_percent - b.roi_percent) * dir
        case 'profit': return (a.net_profit - b.net_profit) * dir
        case 'liquidity': return (a.min_liquidity - b.min_liquidity) * dir
        case 'risk': return (a.risk_score - b.risk_score) * dir
        case 'ai_score': {
          const aScore = a.ai_analysis?.overall_score || 0
          const bScore = b.ai_analysis?.overall_score || 0
          return (aScore - bScore) * dir
        }
        case 'time': {
          const aTime = a.resolution_date ? new Date(a.resolution_date).getTime() : Infinity
          const bTime = b.resolution_date ? new Date(b.resolution_date).getTime() : Infinity
          return (aTime - bTime) * (sortDir === 'asc' ? 1 : -1)
        }
        default: return 0
      }
    })

    return result
  }, [allOpportunities, selectedAsset, selectedTimeframe, sortBy, sortDir])

  // Aggregate stats
  const stats = useMemo(() => {
    const total = filtered.length
    const totalLiquidity = filtered.reduce((sum, o) => sum + o.min_liquidity, 0)
    const bestRoi = filtered.length > 0 ? Math.max(...filtered.map(o => o.roi_percent)) : 0
    const avgRoi = filtered.length > 0 ? filtered.reduce((sum, o) => sum + o.roi_percent, 0) / total : 0
    const totalProfit = filtered.reduce((sum, o) => sum + o.net_profit, 0)
    const pureArb = filtered.filter(o => detectSubStrategy(o) === 'pure_arb').length
    const dumpHedge = filtered.filter(o => detectSubStrategy(o) === 'dump_hedge').length
    const limits = filtered.filter(o => detectSubStrategy(o) === 'pre_placed_limits').length
    const analyzed = filtered.filter(o => o.ai_analysis && o.ai_analysis.recommendation !== 'pending').length
    const executing = filtered.filter(o => o.ai_analysis?.recommendation === 'execute' || o.ai_analysis?.recommendation === 'strong_execute').length

    // Per-asset counts
    const assetCounts: Record<string, number> = {}
    for (const opp of allOpportunities) {
      const a = detectAsset(opp.title)
      assetCounts[a] = (assetCounts[a] || 0) + 1
    }

    return { total, totalLiquidity, bestRoi, avgRoi, totalProfit, pureArb, dumpHedge, limits, analyzed, executing, assetCounts }
  }, [filtered, allOpportunities])

  const analyzeAllMutation = useMutation({
    mutationFn: () => {
      const ids = filtered.filter(o => !o.ai_analysis || o.ai_analysis.recommendation === 'pending').map(o => o.id)
      return judgeOpportunitiesBulk(ids.length > 0 ? { opportunity_ids: ids } : undefined)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['crypto-opportunities'] })
    },
  })

  const handleSort = useCallback((key: typeof sortBy) => {
    if (sortBy === key) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(key)
      setSortDir('desc')
    }
  }, [sortBy])

  return (
    <div className="space-y-4">
      {/* ═══════════════ Header Dashboard ═══════════════ */}
      <div className="rounded-lg border border-border/50 bg-card/60 overflow-hidden">
        {/* Title bar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/30">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Zap className="w-4 h-4 text-orange-400" />
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            </div>
            <span className="text-sm font-semibold text-foreground">Crypto High-Frequency Markets</span>
            <Badge variant="outline" className="text-[9px] text-orange-400 border-orange-500/20 bg-orange-500/10">
              LIVE
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs gap-1.5"
              onClick={() => analyzeAllMutation.mutate()}
              disabled={analyzeAllMutation.isPending || filtered.length === 0}
            >
              {analyzeAllMutation.isPending ? (
                <RefreshCw className="w-3 h-3 animate-spin" />
              ) : (
                <Brain className="w-3 h-3" />
              )}
              Analyze All
            </Button>
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-6 divide-x divide-border/20">
          {[
            { label: 'Opportunities', value: <AnimatedNumber value={stats.total} decimals={0} className="text-sm font-bold font-data text-foreground" />, sub: `${stats.analyzed} analyzed` },
            { label: 'Best ROI', value: <span className="text-sm font-bold font-data text-green-400">{stats.bestRoi > 0 ? `+${stats.bestRoi.toFixed(2)}%` : '--'}</span>, sub: `avg ${stats.avgRoi.toFixed(2)}%` },
            { label: 'Total Profit', value: <span className="text-sm font-bold font-data text-green-400">{formatUsd(stats.totalProfit)}</span>, sub: 'potential' },
            { label: 'Liquidity', value: <span className="text-sm font-bold font-data text-foreground">{formatUsd(stats.totalLiquidity)}</span>, sub: 'available' },
            { label: 'AI Execute', value: <span className="text-sm font-bold font-data text-green-400">{stats.executing}</span>, sub: `of ${stats.analyzed} scored` },
            { label: 'Sub-Strategies', value: (
              <div className="flex items-center gap-1.5">
                {stats.pureArb > 0 && <span className="text-[10px] font-data font-bold text-green-400 bg-green-500/10 px-1 rounded">{stats.pureArb} PA</span>}
                {stats.dumpHedge > 0 && <span className="text-[10px] font-data font-bold text-amber-400 bg-amber-500/10 px-1 rounded">{stats.dumpHedge} DH</span>}
                {stats.limits > 0 && <span className="text-[10px] font-data font-bold text-blue-400 bg-blue-500/10 px-1 rounded">{stats.limits} LO</span>}
                {stats.pureArb === 0 && stats.dumpHedge === 0 && stats.limits === 0 && <span className="text-[10px] text-muted-foreground">--</span>}
              </div>
            ), sub: 'distribution' },
          ].map((stat, i) => (
            <div key={i} className="px-3 py-2.5 text-center">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider mb-1">{stat.label}</div>
              <div>{stat.value}</div>
              <div className="text-[9px] text-muted-foreground/60 mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ═══════════════ Filters ═══════════════ */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Asset filter */}
        <div className="flex items-center gap-0.5 border border-border/50 rounded-lg p-0.5 bg-card/50">
          {ASSETS.map((asset) => {
            const count = asset === 'ALL' ? allOpportunities.length : (stats.assetCounts[asset] || 0)
            return (
              <button
                key={asset}
                onClick={() => setSelectedAsset(asset)}
                className={cn(
                  "px-2.5 py-1 rounded-md text-xs font-medium transition-all flex items-center gap-1",
                  selectedAsset === asset
                    ? asset === 'ALL'
                      ? 'bg-primary/20 text-primary shadow-sm'
                      : cn('shadow-sm', ASSET_BG[asset], ASSET_COLORS[asset])
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                )}
              >
                {asset}
                {count > 0 && (
                  <span className="text-[9px] opacity-70 font-data">{count}</span>
                )}
              </button>
            )
          })}
        </div>

        {/* Timeframe filter */}
        <div className="flex items-center gap-0.5 border border-border/50 rounded-lg p-0.5 bg-card/50">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => setSelectedTimeframe(tf)}
              className={cn(
                "px-2.5 py-1 rounded-md text-xs font-medium transition-all",
                selectedTimeframe === tf
                  ? 'bg-primary/20 text-primary shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              )}
            >
              {tf === 'ALL' ? 'All' : tf === '5min' ? '5 min' : tf === '15min' ? '15 min' : '1 hour'}
            </button>
          ))}
        </div>

        {/* Sort controls */}
        <div className="flex items-center gap-1 ml-auto">
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider mr-1">Sort:</span>
          {([
            ['roi', 'ROI'],
            ['ai_score', 'AI'],
            ['profit', 'Profit'],
            ['liquidity', 'Liq'],
            ['risk', 'Risk'],
            ['time', 'Time'],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => handleSort(key)}
              className={cn(
                'px-2 py-1 rounded text-xs font-medium transition-colors',
                sortBy === key
                  ? 'bg-primary/20 text-primary'
                  : 'bg-muted/50 text-muted-foreground hover:bg-muted'
              )}
            >
              {label}
              {sortBy === key && (
                sortDir === 'desc'
                  ? <ChevronDown className="w-3 h-3 inline ml-0.5" />
                  : <ChevronUp className="w-3 h-3 inline ml-0.5" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* ═══════════════ Content ═══════════════ */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <RefreshCw className="w-8 h-8 animate-spin text-orange-400" />
          <span className="ml-3 text-muted-foreground">Loading crypto markets...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16">
          <div className="relative mb-4">
            <Activity className="w-12 h-12 text-muted-foreground/30" />
          </div>
          <p className="text-muted-foreground font-medium">No crypto market opportunities found</p>
          <p className="text-sm text-muted-foreground/70 mt-1">
            {selectedAsset !== 'ALL' || selectedTimeframe !== 'ALL'
              ? 'Try adjusting your asset or timeframe filters'
              : 'BTC/ETH 15-minute markets will appear here when detected by the scanner'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {filtered.map((opp) => (
            <CryptoMarketCard
              key={opp.stable_id || opp.id}
              opportunity={opp}
              onExecute={onExecute}
              onOpenCopilot={onOpenCopilot}
            />
          ))}
        </div>
      )}
    </div>
  )
}
