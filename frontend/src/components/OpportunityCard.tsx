import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  TrendingUp,
  DollarSign,
  Target,
  ExternalLink,
  Play,
  Brain,
  Shield,
  RefreshCw,
  MessageCircle,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Opportunity, judgeOpportunity } from '../services/api'
import { Card, CardContent, CardHeader } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Separator } from './ui/separator'

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

const STRATEGY_NAMES: Record<string, string> = {
  basic: 'Basic Arb',
  negrisk: 'NegRisk',
  mutually_exclusive: 'Mutually Exclusive',
  contradiction: 'Contradiction',
  must_happen: 'Must-Happen',
  cross_platform: 'Cross-Platform Oracle',
  bayesian_cascade: 'Bayesian Cascade',
  liquidity_vacuum: 'Liquidity Vacuum',
  entropy_arb: 'Entropy Arbitrage',
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

const RECOMMENDATION_BG: Record<string, string> = {
  strong_execute: 'border-green-500/30',
  execute: 'border-green-500/20',
  review: 'border-yellow-500/20',
  skip: 'border-red-500/20',
  strong_skip: 'border-red-500/30',
}

interface Props {
  opportunity: Opportunity
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}

export default function OpportunityCard({ opportunity, onExecute, onOpenCopilot }: Props) {
  const [expanded, setExpanded] = useState(false)
  const queryClient = useQueryClient()

  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

  // Use inline ai_analysis from the opportunity (populated by scanner)
  const inlineAnalysis = opportunity.ai_analysis

  // Judge opportunity mutation (manual re-analysis)
  const judgeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await judgeOpportunity({ opportunity_id: opportunity.id })
      return data
    },
    onSuccess: () => {
      // Refetch opportunities so the backend's updated ai_analysis is included
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
    },
  })

  // Prefer fresh mutation result, fall back to inline analysis
  const judgment = judgeMutation.data || (inlineAnalysis && inlineAnalysis.recommendation !== 'pending' ? inlineAnalysis : null)
  const resolutions = inlineAnalysis?.resolution_analyses || []
  const isPending = inlineAnalysis?.recommendation === 'pending'

  const recommendationBorder = judgment
    ? RECOMMENDATION_BG[judgment.recommendation] || ''
    : ''

  return (
    <Card className={cn("overflow-hidden", recommendationBorder)}>
      {/* Card Header - Always visible with key info */}
      <CardHeader
        className="p-4 pb-0 space-y-0 cursor-pointer hover:bg-muted/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
              <Badge variant="outline" className={cn("text-xs", STRATEGY_COLORS[opportunity.strategy])}>
                {STRATEGY_NAMES[opportunity.strategy] || opportunity.strategy}
              </Badge>
              {opportunity.category && (
                <Badge variant="outline" className="text-xs text-muted-foreground border-border">
                  {opportunity.category}
                </Badge>
              )}
              {opportunity.event_title && (
                <span className="text-xs text-muted-foreground truncate">{opportunity.event_title}</span>
              )}
            </div>
            <h3 className="font-medium text-foreground leading-tight">{opportunity.title}</h3>
            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{opportunity.description}</p>
          </div>

          <div className="text-right shrink-0">
            <div className="flex items-center gap-1 justify-end">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-xl font-bold text-green-400 font-data data-glow-green">
                {opportunity.roi_percent.toFixed(2)}%
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-1 font-data">
              ${opportunity.net_profit.toFixed(4)} profit
            </p>
          </div>
        </div>
      </CardHeader>

      {/* Stats + AI Analysis - Always visible */}
      <CardContent className="p-4 pt-3 space-y-3">
        {/* Stats Row */}
        <div className="grid grid-cols-4 gap-3 text-sm">
          <div className="flex items-center gap-1.5">
            <DollarSign className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            <div>
              <p className="text-[10px] text-muted-foreground leading-none mb-0.5">Cost</p>
              <p className="text-foreground font-medium font-data">${opportunity.total_cost.toFixed(4)}</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <Target className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            <div>
              <p className="text-[10px] text-muted-foreground leading-none mb-0.5">Liquidity</p>
              <p className="text-foreground font-medium font-data">${opportunity.min_liquidity.toFixed(0)}</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <AlertTriangle className={cn("w-3.5 h-3.5 shrink-0", riskColor)} />
            <div>
              <p className="text-[10px] text-muted-foreground leading-none mb-0.5">Risk</p>
              <p className={cn("font-medium font-data", riskColor)}>{(opportunity.risk_score * 100).toFixed(0)}%</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <DollarSign className="w-3.5 h-3.5 text-green-400 shrink-0" />
            <div>
              <p className="text-[10px] text-muted-foreground leading-none mb-0.5">Max Position</p>
              <p className="text-foreground font-medium font-data">${opportunity.max_position_size.toFixed(0)}</p>
            </div>
          </div>
        </div>

        {/* AI Analysis Section - Always shown */}
        <div className={cn(
          "rounded-lg p-3 space-y-2",
          judgment
            ? "bg-gradient-to-r from-purple-500/5 to-blue-500/5 border border-purple-500/15"
            : "bg-muted/50 border border-border"
        )}>
          {isPending && !judgeMutation.isPending && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <RefreshCw className="w-3 h-3 animate-spin" />
              AI analysis queued...
            </div>
          )}

          {judgeMutation.isPending && !judgment && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Running AI analysis...
            </div>
          )}

          {judgment ? (
            <>
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <Brain className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                  <Badge variant="outline" className={cn(
                    'text-xs font-bold',
                    RECOMMENDATION_COLORS[judgment.recommendation]
                  )}>
                    {judgment.recommendation?.replace('_', ' ').toUpperCase()}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Score: {(judgment.overall_score * 100).toFixed(0)}/100
                  </span>
                  {/* Resolution badges */}
                  {resolutions.map((r: any, i: number) => (
                    <Badge key={i} variant="outline" className={cn(
                      'text-[10px] px-1.5 py-0.5 flex items-center gap-1',
                      RECOMMENDATION_COLORS[r.recommendation]
                    )}>
                      <Shield className="w-2.5 h-2.5" />
                      {r.recommendation}
                    </Badge>
                  ))}
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      judgeMutation.mutate()
                    }}
                    disabled={judgeMutation.isPending}
                    className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-purple-400 transition-colors"
                  >
                    <RefreshCw className={cn("w-3 h-3", judgeMutation.isPending && "animate-spin")} />
                  </button>
                </div>
              </div>

              {/* Score breakdown */}
              <div className="grid grid-cols-4 gap-2">
                <ScoreMini label="Profit" value={judgment.profit_viability} />
                <ScoreMini label="Resolution" value={judgment.resolution_safety} />
                <ScoreMini label="Execution" value={judgment.execution_feasibility} />
                <ScoreMini label="Efficiency" value={judgment.market_efficiency} />
              </div>

              {/* Reasoning */}
              {judgment.reasoning && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {judgment.reasoning}
                </p>
              )}

              {/* Resolution details if available */}
              {resolutions.length > 0 && resolutions[0].summary && (
                <div className="pt-1 border-t border-purple-500/10">
                  <p className="text-xs text-muted-foreground">
                    <span className="text-purple-400 font-medium">Resolution: </span>
                    {resolutions[0].summary}
                  </p>
                  {resolutions[0].ambiguities?.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {resolutions[0].ambiguities.slice(0, 2).map((a: string, j: number) => (
                        <span key={j} className="text-[10px] text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                          {a.length > 60 ? a.slice(0, 60) + '...' : a}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          ) : !isPending && !judgeMutation.isPending ? (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground flex items-center gap-1.5">
                <Brain className="w-3.5 h-3.5" />
                No AI analysis yet
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  judgeMutation.mutate()
                }}
                className="h-6 px-2 text-[10px] bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20"
              >
                Analyze
              </Button>
            </div>
          ) : null}

          {judgeMutation.error && (
            <div className="text-xs text-red-400">
              Analysis failed: {(judgeMutation.error as Error).message}
            </div>
          )}
        </div>

        {/* Action buttons row */}
        <div className="flex items-center gap-2">
          {/* Polymarket link - only show if opportunity has a Polymarket market */}
          {(() => {
            const polyMarket = opportunity.markets.find((m: any) => !m.platform || m.platform === 'polymarket')
            if (!polyMarket) return null
            const polyUrl = opportunity.event_slug
              ? `https://polymarket.com/event/${opportunity.event_slug}`
              : polyMarket.slug
                ? `https://polymarket.com/event/${polyMarket.slug}`
                : null
            return polyUrl ? (
              <a
                href={polyUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-1 h-7 px-2.5 text-xs rounded-md border bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20 transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
                Polymarket
              </a>
            ) : null
          })()}
          {/* Kalshi link - only show if opportunity has a Kalshi market */}
          {(() => {
            const kalshiMarket = opportunity.markets.find((m: any) => m.platform === 'kalshi')
            if (!kalshiMarket) return null
            const kalshiUrl = `https://kalshi.com/markets/${kalshiMarket.id}`
            return (
              <a
                href={kalshiUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-1 h-7 px-2.5 text-xs rounded-md border bg-indigo-500/10 text-indigo-400 border-indigo-500/20 hover:bg-indigo-500/20 transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
                Kalshi
              </a>
            )
          })()}
          {onOpenCopilot && (
            <Button
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                onOpenCopilot(opportunity)
              }}
              className="h-7 text-xs bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20"
            >
              <MessageCircle className="w-3 h-3 mr-1" />
              Ask AI
            </Button>
          )}
          <div
            className="ml-auto flex items-center gap-1 text-xs text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Less' : 'More details'}
            {expanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </div>
        </div>
      </CardContent>

      {/* Expanded Details */}
      {expanded && (
        <>
          <Separator />
          <CardContent className="p-4 pt-4 space-y-4">
            {/* Positions to Take */}
            <div>
              <h4 className="text-sm font-medium text-muted-foreground mb-2">Positions to Take</h4>
              <div className="space-y-2">
                {opportunity.positions_to_take.map((pos, idx) => {
                  const positionPlatform = (pos as any).platform
                  return (
                    <div
                      key={idx}
                      className="flex items-center justify-between bg-muted rounded-lg p-3"
                    >
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge variant="outline" className={cn(
                          "text-xs",
                          pos.outcome === 'YES' ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'
                        )}>
                          {pos.action} {pos.outcome}
                        </Badge>
                        {positionPlatform && (
                          <Badge variant="outline" className={cn(
                            "text-[10px]",
                            positionPlatform === 'kalshi'
                              ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20'
                              : 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                          )}>
                            {positionPlatform === 'kalshi' ? 'Kalshi' : 'Polymarket'}
                          </Badge>
                        )}
                        <span className="text-sm text-foreground/80">{pos.market}</span>
                      </div>
                      <span className="font-data text-foreground">${pos.price.toFixed(4)}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Risk Factors */}
            {opportunity.risk_factors.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">Risk Factors</h4>
                <ul className="space-y-1">
                  {opportunity.risk_factors.map((factor, idx) => (
                    <li key={idx} className="flex items-center gap-2 text-sm">
                      <AlertTriangle className="w-4 h-4 text-yellow-500" />
                      <span className="text-muted-foreground">{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Markets */}
            <div>
              <h4 className="text-sm font-medium text-muted-foreground mb-2">Markets Involved</h4>
              <div className="space-y-2">
                {opportunity.markets.map((market, idx) => {
                  const isKalshi = (market as any).platform === 'kalshi'
                  const marketUrl = isKalshi
                    ? `https://kalshi.com/markets/${market.id}`
                    : market.slug
                      ? `https://polymarket.com/event/${market.slug}`
                      : `https://polymarket.com/event/${market.id}`
                  return (
                    <div
                      key={idx}
                      className="flex items-center justify-between bg-muted rounded-lg p-3"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className="text-sm text-foreground/80">{market.question}</p>
                          {isKalshi && (
                            <Badge variant="outline" className="text-[10px] bg-indigo-500/10 text-indigo-400 border-indigo-500/20">
                              Kalshi
                            </Badge>
                          )}
                          {!isKalshi && opportunity.strategy === 'cross_platform' && (
                            <Badge variant="outline" className="text-[10px] bg-blue-500/10 text-blue-400 border-blue-500/20">
                              Polymarket
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          YES: ${market.yes_price.toFixed(3)} | NO: ${market.no_price.toFixed(3)} |
                          Liquidity: ${market.liquidity.toFixed(0)}
                        </p>
                      </div>
                      <a
                        href={marketUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={cn(
                          "p-2 rounded-lg transition-colors",
                          isKalshi ? "hover:bg-indigo-500/20" : "hover:bg-accent"
                        )}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <ExternalLink className={cn("w-4 h-4", isKalshi ? "text-indigo-400" : "text-muted-foreground")} />
                      </a>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Profit Breakdown */}
            <div className="bg-muted rounded-lg p-4">
              <h4 className="text-sm font-medium text-muted-foreground mb-3">Profit Breakdown</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Total Cost</p>
                  <p className="font-data text-foreground">${opportunity.total_cost.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Expected Payout</p>
                  <p className="font-data text-foreground">${opportunity.expected_payout.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Gross Profit</p>
                  <p className="font-data text-foreground">${opportunity.gross_profit.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Fee (2%)</p>
                  <p className="font-data text-red-400">-${opportunity.fee.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Net Profit</p>
                  <p className="font-data text-green-400 data-glow-green">${opportunity.net_profit.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">ROI</p>
                  <p className="font-data text-green-400 data-glow-green">{opportunity.roi_percent.toFixed(2)}%</p>
                </div>
              </div>
            </div>

            {/* Extended Resolution Analysis */}
            {resolutions.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">Resolution Analysis</h4>
                <div className="space-y-2">
                  {resolutions.map((r: any, i: number) => (
                    <div key={i} className="bg-muted rounded-lg p-3 space-y-1">
                      <div className="flex items-center gap-2">
                        <Shield className="w-3 h-3 text-muted-foreground" />
                        <Badge variant="outline" className={cn(
                          'text-[10px] px-1.5 py-0.5',
                          RECOMMENDATION_COLORS[r.recommendation]
                        )}>
                          {r.recommendation}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground/50">
                          Clarity: {(r.clarity_score * 100).toFixed(0)} | Risk: {(r.risk_score * 100).toFixed(0)}
                        </span>
                      </div>
                      {r.summary && (
                        <p className="text-xs text-muted-foreground">{r.summary}</p>
                      )}
                      {r.ambiguities?.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {r.ambiguities.map((a: string, j: number) => (
                            <span key={j} className="text-[10px] text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                              {a.length > 60 ? a.slice(0, 60) + '...' : a}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Execute Button */}
            {onExecute && (
              <Button
                onClick={(e) => {
                  e.stopPropagation()
                  onExecute(opportunity)
                }}
                className="w-full bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600 shadow-lg shadow-blue-500/20"
              >
                <Play className="w-4 h-4 mr-2" />
                Execute Trade
              </Button>
            )}
          </CardContent>
        </>
      )}
    </Card>
  )
}

function ScoreMini({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'text-green-400' : value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="text-center">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={cn('text-sm font-bold', color)}>
        {typeof value === 'number' ? (value * 100).toFixed(0) : 'N/A'}
      </p>
    </div>
  )
}
