import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ExternalLink,
  Brain,
  RefreshCw,
  MessageCircle,
  Play,
  Shield,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Opportunity, judgeOpportunity } from '../services/api'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import Sparkline from './Sparkline'
import {
  STRATEGY_COLORS,
  STRATEGY_ABBREV,
  RECOMMENDATION_COLORS,
  ACCENT_BAR_COLORS,
  generatePriceHistory,
  timeAgo,
  formatCompact,
} from './OpportunityCard'

interface Props {
  opportunities: Opportunity[]
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}

export default function OpportunityTable({ opportunities, onExecute, onOpenCopilot }: Props) {
  return (
    <div className="border border-border/50 rounded-lg overflow-hidden">
      {/* Table Header */}
      <div className="grid grid-cols-[36px_minmax(0,1fr)_72px_76px_64px_64px_64px_64px_72px_52px] gap-0 bg-muted/50 border-b border-border/50 text-[9px] text-muted-foreground uppercase tracking-wider font-medium">
        <div className="px-2 py-2">Stg</div>
        <div className="px-2 py-2">Market</div>
        <div className="px-2 py-2">Trend</div>
        <div className="px-2 py-2 text-right">ROI</div>
        <div className="px-2 py-2 text-right">Net</div>
        <div className="px-2 py-2 text-right">Cost</div>
        <div className="px-2 py-2 text-right">Risk</div>
        <div className="px-2 py-2 text-right">Liq</div>
        <div className="px-2 py-2 text-center">AI</div>
        <div className="px-2 py-2 text-right">Age</div>
      </div>

      {/* Table Body */}
      <div className="divide-y divide-border/30">
        {opportunities.map((opp) => (
          <TableRow
            key={opp.stable_id || opp.id}
            opportunity={opp}
            onExecute={onExecute}
            onOpenCopilot={onOpenCopilot}
          />
        ))}
      </div>
    </div>
  )
}

function TableRow({
  opportunity,
  onExecute,
  onOpenCopilot,
}: {
  opportunity: Opportunity
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [aiExpanded, setAiExpanded] = useState(false)
  const queryClient = useQueryClient()

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
  const recommendation = judgment?.recommendation || ''
  const resolutions = inlineAnalysis?.resolution_analyses || []

  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

  const market = opportunity.markets[0]
  const id = opportunity.stable_id || opportunity.id
  const sparkYes = market ? generatePriceHistory(id, market.yes_price, 16) : []

  const roiPositive = opportunity.roi_percent >= 0
  const accentColor = recommendation ? (ACCENT_BAR_COLORS[recommendation] || '') : ''

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
  const kalshiUrl = kalshiMarket
    ? (() => {
        const ticker = kalshiMarket.id.toLowerCase()
        const eventTicker = ticker.split('-')[0]
        return `https://kalshi.com/markets/${eventTicker}/${ticker}`
      })()
    : null

  return (
    <div className="group">
      {/* Main Row */}
      <div
        className={cn(
          "grid grid-cols-[36px_minmax(0,1fr)_72px_76px_64px_64px_64px_64px_72px_52px] gap-0 items-center cursor-pointer transition-colors relative",
          "hover:bg-muted/30",
          expanded && "bg-muted/20"
        )}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Accent bar */}
        {accentColor && <div className={cn("absolute left-0 top-0 bottom-0 w-0.5", accentColor)} />}

        {/* Strategy */}
        <div className="px-2 py-1.5">
          <Badge variant="outline" className={cn("text-[8px] px-1 py-0 leading-tight", STRATEGY_COLORS[opportunity.strategy])}>
            {STRATEGY_ABBREV[opportunity.strategy] || opportunity.strategy.slice(0, 3).toUpperCase()}
          </Badge>
        </div>

        {/* Title */}
        <div className="px-2 py-1.5 min-w-0">
          <p className="text-xs font-medium text-foreground truncate leading-tight">{opportunity.title}</p>
          {opportunity.category && (
            <span className="text-[9px] text-muted-foreground/60">{opportunity.category}</span>
          )}
        </div>

        {/* Sparkline */}
        <div className="px-1 py-1">
          {sparkYes.length >= 2 && (
            <Sparkline
              data={sparkYes}
              width={60}
              height={20}
              color={roiPositive ? '#22c55e' : '#ef4444'}
              lineWidth={1}
              showDots
            />
          )}
        </div>

        {/* ROI */}
        <div className="px-2 py-1.5 text-right">
          <span className={cn(
            "text-xs font-data font-bold",
            roiPositive ? "text-green-400" : "text-red-400"
          )}>
            {roiPositive ? '+' : ''}{opportunity.roi_percent.toFixed(2)}%
          </span>
        </div>

        {/* Net Profit */}
        <div className="px-2 py-1.5 text-right">
          <span className="text-xs font-data text-green-400">{formatCompact(opportunity.net_profit)}</span>
        </div>

        {/* Cost */}
        <div className="px-2 py-1.5 text-right">
          <span className="text-xs font-data text-foreground/80">{formatCompact(opportunity.total_cost)}</span>
        </div>

        {/* Risk */}
        <div className="px-2 py-1.5 text-right">
          <span className={cn("text-xs font-data font-medium", riskColor)}>
            {(opportunity.risk_score * 100).toFixed(0)}%
          </span>
        </div>

        {/* Liquidity */}
        <div className="px-2 py-1.5 text-right">
          <span className="text-xs font-data text-foreground/80">{formatCompact(opportunity.min_liquidity)}</span>
        </div>

        {/* AI Score */}
        <div className="px-2 py-1.5 text-center">
          {judgment ? (
            <div className="flex items-center justify-center gap-1">
              <span className="text-[10px] font-data font-bold text-purple-300">
                {(judgment.overall_score * 100).toFixed(0)}
              </span>
              <div className="w-8 h-1 bg-muted/80 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-400"
                  style={{ width: `${Math.min(100, judgment.overall_score * 100)}%` }}
                />
              </div>
            </div>
          ) : isPending ? (
            <RefreshCw className="w-3 h-3 animate-spin text-muted-foreground mx-auto" />
          ) : (
            <span className="text-[10px] text-muted-foreground/40">--</span>
          )}
        </div>

        {/* Time */}
        <div className="px-2 py-1.5 text-right">
          <span className="text-[10px] font-data text-muted-foreground">{timeAgo(opportunity.detected_at)}</span>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="bg-muted/10 border-t border-border/20 px-3 py-2.5 space-y-2">
          <div className="flex items-start gap-4">
            {/* Left: Details */}
            <div className="flex-1 space-y-2 min-w-0">
              {opportunity.description && (
                <p className="text-[11px] text-muted-foreground leading-relaxed">{opportunity.description}</p>
              )}

              {/* Positions */}
              <div className="flex items-center gap-2 flex-wrap">
                {opportunity.positions_to_take.map((pos, i) => (
                  <span key={i} className="inline-flex items-center gap-1 text-[10px]">
                    <Badge variant="outline" className={cn(
                      "text-[9px] px-1 py-0",
                      pos.outcome === 'YES' ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'
                    )}>
                      {pos.action} {pos.outcome}
                    </Badge>
                    <span className="font-data text-foreground/70">@${pos.price.toFixed(4)}</span>
                    <span className="text-muted-foreground/40 truncate max-w-[200px]">{pos.market}</span>
                  </span>
                ))}
              </div>

              {/* Markets */}
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                {opportunity.markets.map((mkt, i) => (
                  <span key={i} className="font-data">
                    Y:{mkt.yes_price.toFixed(3)} N:{mkt.no_price.toFixed(3)} Liq:{formatCompact(mkt.liquidity)}
                  </span>
                ))}
              </div>

              {/* Risk Factors */}
              {opportunity.risk_factors.length > 0 && (
                <div className="flex items-center gap-1 flex-wrap">
                  {opportunity.risk_factors.map((f, i) => (
                    <span key={i} className="text-[9px] text-yellow-400/80 bg-yellow-500/5 px-1 py-0 rounded">
                      {f.length > 40 ? f.slice(0, 40) + '...' : f}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Right: AI + Actions */}
            <div className="shrink-0 space-y-2 w-48">
              {judgment && (
                <div className="bg-purple-500/[0.06] rounded-md p-2 border border-purple-500/10 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <Brain className="w-3 h-3 text-purple-400" />
                    <Badge variant="outline" className={cn("text-[9px] px-1 py-0 font-bold", RECOMMENDATION_COLORS[recommendation])}>
                      {recommendation.replace('_', ' ').toUpperCase()}
                    </Badge>
                    <span className="text-[9px] font-data text-purple-300 ml-auto">
                      {(judgment.overall_score * 100).toFixed(0)}/100
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-1 text-[8px] font-data text-muted-foreground">
                    <span>P{(judgment.profit_viability * 100).toFixed(0)}</span>
                    <span>R{(judgment.resolution_safety * 100).toFixed(0)}</span>
                    <span>E{(judgment.execution_feasibility * 100).toFixed(0)}</span>
                    <span>M{(judgment.market_efficiency * 100).toFixed(0)}</span>
                  </div>
                  {judgment.reasoning && (
                    <p
                      className={`text-[9px] text-muted-foreground cursor-pointer hover:text-muted-foreground/80 transition-colors ${!aiExpanded ? 'line-clamp-2' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setAiExpanded(!aiExpanded) }}
                      title={aiExpanded ? "Click to collapse" : "Click to expand full analysis"}
                    >
                      {judgment.reasoning}
                    </p>
                  )}
                </div>
              )}

              {/* Resolution */}
              {resolutions.length > 0 && resolutions[0].summary && (
                <div className="bg-muted/30 rounded-md p-1.5 border border-border/50">
                  <div className="flex items-center gap-1">
                    <Shield className="w-2.5 h-2.5 text-muted-foreground" />
                    <Badge variant="outline" className={cn('text-[8px] px-1 py-0', RECOMMENDATION_COLORS[resolutions[0].recommendation])}>
                      {resolutions[0].recommendation}
                    </Badge>
                  </div>
                  <p className="text-[9px] text-muted-foreground mt-0.5 line-clamp-2">{resolutions[0].summary}</p>
                </div>
              )}

              <div className="flex items-center gap-1">
                {polyUrl && (
                  <a
                    href={polyUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-0.5 h-5 px-1.5 text-[9px] rounded border bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20 transition-colors"
                  >
                    <ExternalLink className="w-2 h-2" /> PM
                  </a>
                )}
                {kalshiUrl && (
                  <a
                    href={kalshiUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-0.5 h-5 px-1.5 text-[9px] rounded border bg-indigo-500/10 text-indigo-400 border-indigo-500/20 hover:bg-indigo-500/20 transition-colors"
                  >
                    <ExternalLink className="w-2 h-2" /> KL
                  </a>
                )}
                {onOpenCopilot && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onOpenCopilot(opportunity) }}
                    className="inline-flex items-center gap-0.5 h-5 px-1.5 text-[9px] rounded border bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20 transition-colors"
                  >
                    <MessageCircle className="w-2 h-2" /> AI
                  </button>
                )}
                {!judgment && !isPending && (
                  <button
                    onClick={(e) => { e.stopPropagation(); judgeMutation.mutate() }}
                    disabled={judgeMutation.isPending}
                    className="inline-flex items-center gap-0.5 h-5 px-1.5 text-[9px] rounded border bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20 transition-colors"
                  >
                    <Brain className="w-2 h-2" /> {judgeMutation.isPending ? '...' : 'Analyze'}
                  </button>
                )}
                {onExecute && (
                  <Button
                    onClick={(e) => { e.stopPropagation(); onExecute(opportunity) }}
                    size="sm"
                    className="h-5 px-2 text-[9px] bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600 ml-auto"
                  >
                    <Play className="w-2 h-2 mr-0.5" /> Execute
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
