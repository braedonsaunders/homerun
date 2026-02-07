import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
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
  Sparkles,
  MessageCircle,
} from 'lucide-react'
import clsx from 'clsx'
import { Opportunity, judgeOpportunity, getOpportunityAISummary } from '../services/api'

const STRATEGY_COLORS: Record<string, string> = {
  basic: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  negrisk: 'bg-green-500/10 text-green-400 border-green-500/20',
  mutually_exclusive: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  contradiction: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  must_happen: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
}

const STRATEGY_NAMES: Record<string, string> = {
  basic: 'Basic Arb',
  negrisk: 'NegRisk',
  mutually_exclusive: 'Mutually Exclusive',
  contradiction: 'Contradiction',
  must_happen: 'Must-Happen',
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
}

interface Props {
  opportunity: Opportunity
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}

export default function OpportunityCard({ opportunity, onExecute, onOpenCopilot }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [showAIInsights, setShowAIInsights] = useState(false)

  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

  // Fetch AI summary when card expands
  const { data: aiSummary, isLoading: aiSummaryLoading } = useQuery({
    queryKey: ['ai-opportunity-summary', opportunity.id],
    queryFn: () => getOpportunityAISummary(opportunity.id),
    enabled: expanded,
    staleTime: 60000,
  })

  // Judge opportunity mutation
  const judgeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await judgeOpportunity({ opportunity_id: opportunity.id })
      return data
    },
  })

  const judgment = judgeMutation.data || aiSummary?.judgment
  const resolutions = aiSummary?.resolution_analyses || []

  return (
    <div className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div
        className="p-4 cursor-pointer hover:bg-[#1a1a1a] transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className={clsx(
                "px-2 py-0.5 rounded text-xs font-medium border",
                STRATEGY_COLORS[opportunity.strategy] || 'bg-gray-500/10 text-gray-400'
              )}>
                {STRATEGY_NAMES[opportunity.strategy] || opportunity.strategy}
              </span>
              {opportunity.event_title && (
                <span className="text-xs text-gray-500">{opportunity.event_title}</span>
              )}
              {/* AI Score Badge - shows inline when available */}
              {judgment && (
                <span className={clsx(
                  'px-2 py-0.5 rounded text-xs font-medium border flex items-center gap-1',
                  RECOMMENDATION_COLORS[judgment.recommendation] || 'bg-gray-500/10 text-gray-400'
                )}>
                  <Brain className="w-3 h-3" />
                  {(judgment.overall_score * 100).toFixed(0)}
                </span>
              )}
              {/* Resolution Safety Badge */}
              {resolutions.length > 0 && (
                <span className={clsx(
                  'px-2 py-0.5 rounded text-xs font-medium border flex items-center gap-1',
                  RECOMMENDATION_COLORS[resolutions[0].recommendation] || 'bg-gray-500/10 text-gray-400'
                )}>
                  <Shield className="w-3 h-3" />
                  {resolutions[0].recommendation}
                </span>
              )}
            </div>
            <h3 className="font-medium text-white">{opportunity.title}</h3>
            <p className="text-sm text-gray-400 mt-1">{opportunity.description}</p>
          </div>

          <div className="text-right">
            <div className="flex items-center gap-1 justify-end">
              <TrendingUp className="w-4 h-4 text-green-500" />
              <span className="text-xl font-bold text-green-500">
                {opportunity.roi_percent.toFixed(2)}%
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              ${opportunity.net_profit.toFixed(4)} profit
            </p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="flex items-center gap-4 mt-4 text-sm">
          <div className="flex items-center gap-1">
            <DollarSign className="w-4 h-4 text-gray-500" />
            <span className="text-gray-400">
              Cost: <span className="text-white">${opportunity.total_cost.toFixed(4)}</span>
            </span>
          </div>
          <div className="flex items-center gap-1">
            <Target className="w-4 h-4 text-gray-500" />
            <span className="text-gray-400">
              Liquidity: <span className="text-white">${opportunity.min_liquidity.toFixed(0)}</span>
            </span>
          </div>
          <div className="flex items-center gap-1">
            <AlertTriangle className={clsx("w-4 h-4", riskColor)} />
            <span className="text-gray-400">
              Risk: <span className={riskColor}>{(opportunity.risk_score * 100).toFixed(0)}%</span>
            </span>
          </div>
          <div className="ml-auto">
            {expanded ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="border-t border-gray-800 p-4 space-y-4">
          {/* AI Actions Bar */}
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation()
                judgeMutation.mutate()
              }}
              disabled={judgeMutation.isPending}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border',
                judgeMutation.isPending
                  ? 'bg-gray-800 text-gray-600 border-gray-700 cursor-not-allowed'
                  : 'bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20'
              )}
            >
              {judgeMutation.isPending ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Brain className="w-3.5 h-3.5" />
              )}
              AI Judge
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                setShowAIInsights(!showAIInsights)
              }}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border',
                showAIInsights
                  ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                  : 'bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20'
              )}
            >
              <Sparkles className="w-3.5 h-3.5" />
              AI Insights
            </button>
            {onOpenCopilot && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onOpenCopilot(opportunity)
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20"
              >
                <MessageCircle className="w-3.5 h-3.5" />
                Ask AI
              </button>
            )}
            {aiSummaryLoading && (
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-gray-500 ml-auto" />
            )}
          </div>

          {/* AI Insights Panel */}
          {showAIInsights && (judgment || resolutions.length > 0) && (
            <div className="bg-gradient-to-r from-purple-500/5 to-blue-500/5 border border-purple-500/20 rounded-xl p-4 space-y-3">
              <h4 className="text-sm font-medium text-purple-400 flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                AI Intelligence
              </h4>

              {judgment && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      'px-2 py-1 rounded-lg text-xs font-bold border',
                      RECOMMENDATION_COLORS[judgment.recommendation]
                    )}>
                      {judgment.recommendation?.replace('_', ' ').toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-500">
                      Overall: {(judgment.overall_score * 100).toFixed(0)}/100
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <ScoreMini label="Profit" value={judgment.profit_viability} />
                    <ScoreMini label="Resolution" value={judgment.resolution_safety} />
                    <ScoreMini label="Execution" value={judgment.execution_feasibility} />
                    <ScoreMini label="Efficiency" value={judgment.market_efficiency} />
                  </div>
                  {judgment.reasoning && (
                    <p className="text-xs text-gray-400 bg-[#1a1a1a] p-2 rounded-lg">
                      {judgment.reasoning}
                    </p>
                  )}
                </div>
              )}

              {resolutions.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-purple-500/10">
                  <p className="text-xs text-gray-500 font-medium">Resolution Analysis</p>
                  {resolutions.map((r: any, i: number) => (
                    <div key={i} className="bg-[#1a1a1a] rounded-lg p-2 space-y-1">
                      <div className="flex items-center gap-2">
                        <Shield className="w-3 h-3 text-gray-500" />
                        <span className={clsx(
                          'px-1.5 py-0.5 rounded text-[10px] font-medium border',
                          RECOMMENDATION_COLORS[r.recommendation]
                        )}>
                          {r.recommendation}
                        </span>
                        <span className="text-[10px] text-gray-600">
                          Clarity: {(r.clarity_score * 100).toFixed(0)} | Risk: {(r.risk_score * 100).toFixed(0)}
                        </span>
                      </div>
                      {r.summary && (
                        <p className="text-xs text-gray-400">{r.summary}</p>
                      )}
                      {r.ambiguities?.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {r.ambiguities.slice(0, 3).map((a: string, j: number) => (
                            <span key={j} className="text-[10px] text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                              {a.length > 60 ? a.slice(0, 60) + '...' : a}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {!judgment && resolutions.length === 0 && (
                <p className="text-xs text-gray-500">
                  No AI analysis cached. Click "AI Judge" to run a fresh analysis.
                </p>
              )}
            </div>
          )}

          {/* Judgment result inline (when just triggered) */}
          {judgeMutation.data && !showAIInsights && (
            <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-3">
              <div className="flex items-center gap-2 text-sm">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className={clsx(
                  'px-2 py-0.5 rounded text-xs font-medium border',
                  RECOMMENDATION_COLORS[judgeMutation.data.recommendation]
                )}>
                  {judgeMutation.data.recommendation?.replace('_', ' ')}
                </span>
                <span className="text-gray-400">
                  Score: {(judgeMutation.data.overall_score * 100).toFixed(0)}/100
                </span>
              </div>
            </div>
          )}
          {judgeMutation.error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-2 text-xs text-red-400">
              AI Judge error: {(judgeMutation.error as Error).message}
            </div>
          )}

          {/* Positions to Take */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 mb-2">Positions to Take</h4>
            <div className="space-y-2">
              {opportunity.positions_to_take.map((pos, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3"
                >
                  <div>
                    <span className={clsx(
                      "px-2 py-0.5 rounded text-xs font-medium mr-2",
                      pos.outcome === 'YES' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    )}>
                      {pos.action} {pos.outcome}
                    </span>
                    <span className="text-sm text-gray-300">{pos.market}</span>
                  </div>
                  <span className="font-mono text-white">${pos.price.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Risk Factors */}
          {opportunity.risk_factors.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-400 mb-2">Risk Factors</h4>
              <ul className="space-y-1">
                {opportunity.risk_factors.map((factor, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-sm">
                    <AlertTriangle className="w-4 h-4 text-yellow-500" />
                    <span className="text-gray-400">{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Markets */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 mb-2">Markets Involved</h4>
            <div className="space-y-2">
              {opportunity.markets.map((market, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3"
                >
                  <div className="flex-1">
                    <p className="text-sm text-gray-300">{market.question}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      YES: ${market.yes_price.toFixed(3)} | NO: ${market.no_price.toFixed(3)} |
                      Liquidity: ${market.liquidity.toFixed(0)}
                    </p>
                  </div>
                  <a
                    href={`https://polymarket.com/event/${market.id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <ExternalLink className="w-4 h-4 text-gray-500" />
                  </a>
                </div>
              ))}
            </div>
          </div>

          {/* Profit Breakdown */}
          <div className="bg-[#1a1a1a] rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-400 mb-3">Profit Breakdown</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Total Cost</p>
                <p className="font-mono text-white">${opportunity.total_cost.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-gray-500">Expected Payout</p>
                <p className="font-mono text-white">${opportunity.expected_payout.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-gray-500">Gross Profit</p>
                <p className="font-mono text-white">${opportunity.gross_profit.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-gray-500">Fee (2%)</p>
                <p className="font-mono text-red-400">-${opportunity.fee.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-gray-500">Net Profit</p>
                <p className="font-mono text-green-400">${opportunity.net_profit.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-gray-500">ROI</p>
                <p className="font-mono text-green-400">{opportunity.roi_percent.toFixed(2)}%</p>
              </div>
            </div>
          </div>

          {/* Max Position */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Max Position Size (10% of liquidity)</span>
            <span className="font-mono text-white">${opportunity.max_position_size.toFixed(2)}</span>
          </div>

          {/* Execute Button */}
          {onExecute && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onExecute(opportunity)
              }}
              className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600 rounded-xl text-sm font-semibold text-white transition-all shadow-lg shadow-blue-500/20"
            >
              <Play className="w-4 h-4" />
              Execute Trade
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function ScoreMini({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'text-green-400' : value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="text-center">
      <p className="text-[10px] text-gray-500">{label}</p>
      <p className={clsx('text-sm font-bold', color)}>
        {typeof value === 'number' ? (value * 100).toFixed(0) : 'N/A'}
      </p>
    </div>
  )
}
