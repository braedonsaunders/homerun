import { useState, useEffect, useRef } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Terminal,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildKalshiMarketUrl, buildPolymarketMarketUrl } from '../lib/marketUrls'
import { Opportunity, judgeOpportunity } from '../services/api'
import {
  STRATEGY_ABBREV,
  timeAgo,
  formatCompact,
} from './OpportunityCard'

interface Props {
  opportunities: Opportunity[]
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
  isConnected?: boolean
  totalCount?: number
}

export default function OpportunityTerminal({ opportunities, onExecute, onOpenCopilot, isConnected, totalCount }: Props) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Terminal-style blinking cursor
  const [cursorVisible, setCursorVisible] = useState(true)
  useEffect(() => {
    const iv = setInterval(() => setCursorVisible(v => !v), 530)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="terminal-view bg-[#0a0c10] border border-green-500/20 rounded-lg overflow-hidden font-data text-[11px] leading-relaxed">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#0d1017] border-b border-green-500/15">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-green-400" />
          <span className="text-green-400 font-bold text-xs">HOMERUN SCANNER</span>
          <span className="text-green-400/40">v2.0</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-green-400/60">
            {totalCount ?? opportunities.length} opportunities
          </span>
          <div className="flex items-center gap-1">
            <div className={cn(
              "w-1.5 h-1.5 rounded-full",
              isConnected ? "bg-green-400 live-dot" : "bg-red-400"
            )} />
            <span className={cn("text-[10px]", isConnected ? "text-green-400/70" : "text-red-400/70")}>
              {isConnected ? 'LIVE' : 'DISCONNECTED'}
            </span>
          </div>
        </div>
      </div>

      {/* Terminal Body */}
      <div ref={scrollRef} className="max-h-[calc(100vh-280px)] overflow-y-auto p-3 space-y-0">
        {/* Boot sequence header */}
        <div className="text-green-500/30 mb-3 space-y-0.5">
          <p>{'>'} Initializing arbitrage scanner...</p>
          <p>{'>'} Connected to Polymarket + Kalshi feeds</p>
          <p>{'>'} {opportunities.length} opportunities loaded</p>
          <p className="text-green-500/15">{'─'.repeat(72)}</p>
        </div>

        {/* Opportunities */}
        {opportunities.map((opp, idx) => (
          <TerminalEntry
            key={opp.stable_id || opp.id}
            opportunity={opp}
            isSelected={selectedIdx === idx}
            onSelect={() => setSelectedIdx(selectedIdx === idx ? null : idx)}
            onExecute={onExecute}
            onOpenCopilot={onOpenCopilot}
          />
        ))}

        {/* Cursor line */}
        <div className="text-green-400/60 mt-2 flex items-center">
          <span className="text-green-400/30">{'>'} </span>
          <span className="text-green-400/40">awaiting next scan</span>
          <span className={cn(
            "inline-block w-2 h-3.5 bg-green-400/60 ml-1 -mb-0.5",
            cursorVisible ? "opacity-100" : "opacity-0"
          )} />
        </div>
      </div>
    </div>
  )
}

function TerminalEntry({
  opportunity,
  isSelected,
  onSelect,
  onExecute,
  onOpenCopilot,
}: {
  opportunity: Opportunity
  isSelected: boolean
  onSelect: () => void
  onExecute?: (opportunity: Opportunity) => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}) {
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

  const strat = STRATEGY_ABBREV[opportunity.strategy] || opportunity.strategy.slice(0, 3).toUpperCase()
  const riskPct = (opportunity.risk_score * 100).toFixed(0)
  const roiStr = (opportunity.roi_percent >= 0 ? '+' : '') + opportunity.roi_percent.toFixed(2)

  // Color coding for recommendation
  const recColor = recommendation.includes('execute') || recommendation === 'safe'
    ? 'text-green-400'
    : recommendation === 'review' || recommendation === 'caution'
      ? 'text-yellow-400'
      : recommendation.includes('skip') || recommendation === 'avoid'
        ? 'text-red-400'
        : 'text-muted-foreground/40'

  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

  // Platform URLs
  const polyMarket = opportunity.markets.find((m: any) => !m.platform || m.platform === 'polymarket')
  const polyUrl = polyMarket
    ? buildPolymarketMarketUrl({
        eventSlug: opportunity.event_slug || (polyMarket as any).event_slug,
        marketSlug: (polyMarket as any).slug,
        marketId: (polyMarket as any).id,
      })
    : null
  const kalshiMarket = opportunity.markets.find((m: any) => m.platform === 'kalshi')
  const kalshiUrl = kalshiMarket
    ? buildKalshiMarketUrl({
        marketTicker: kalshiMarket.id,
        eventTicker: (kalshiMarket as any).event_slug,
        eventSlug: (kalshiMarket as any).slug,
      })
    : null

  return (
    <div
      className={cn(
        "cursor-pointer transition-colors rounded px-2 py-1 -mx-2",
        isSelected ? "bg-green-500/[0.06]" : "hover:bg-green-500/[0.03]"
      )}
      onClick={onSelect}
    >
      {/* Main line */}
      <div className="flex items-center gap-0">
        <span className="text-green-500/30 mr-1">{'>'}</span>
        <span className="text-cyan-400 mr-2">[{strat}]</span>
        <span className="text-green-400 font-bold mr-2">ROI:{roiStr}%</span>
        <span className="text-green-300/80 mr-2">NET:{formatCompact(opportunity.net_profit)}</span>
        <span className={cn("mr-2", riskColor)}>RISK:{riskPct}%</span>
        {judgment && (
          <span className={cn("font-bold mr-2", recColor)}>
            AI:{(judgment.overall_score * 100).toFixed(0)} {recommendation.replace('_', ' ').toUpperCase()}
          </span>
        )}
        <span className="text-green-500/25 ml-auto">{timeAgo(opportunity.detected_at)} ago</span>
      </div>

      {/* Title */}
      <div className="text-green-100/70 pl-4 truncate">
        &quot;{opportunity.title}&quot;
      </div>

      {/* Positions line */}
      <div className="text-green-400/50 pl-4">
        POSITIONS:{' '}
        {opportunity.positions_to_take.map((pos, i) => (
          <span key={i}>
            {i > 0 && ' | '}
            <span className={pos.outcome === 'YES' ? 'text-green-400/70' : 'text-red-400/70'}>
              {pos.action} {pos.outcome}
            </span>
            <span className="text-green-400/40"> @${pos.price.toFixed(4)}</span>
          </span>
        ))}
      </div>

      {/* Metrics line */}
      <div className="text-green-400/40 pl-4">
        MKTS:{opportunity.markets.length}
        {' | '}LIQ:{formatCompact(opportunity.min_liquidity)}
        {' | '}COST:{formatCompact(opportunity.total_cost)}
        {' | '}MAX:{formatCompact(opportunity.max_position_size)}
        {opportunity.category && <>{' | '}CAT:{opportunity.category.toUpperCase()}</>}
      </div>

      {/* Expanded details */}
      {isSelected && (
        <div className="pl-4 mt-1 space-y-1 border-l-2 border-green-500/20 ml-1">
          {/* Market details */}
          {opportunity.markets.map((mkt, i) => (
            <div key={i} className="text-green-400/50">
              MKT[{i}]: Y:{mkt.yes_price.toFixed(4)} N:{mkt.no_price.toFixed(4)} LIQ:{formatCompact(mkt.liquidity)}
              <span className="text-green-400/25 truncate ml-2">{mkt.question}</span>
            </div>
          ))}

          {/* Profit breakdown */}
          <div className="text-green-300/50">
            PROFIT: cost=${opportunity.total_cost.toFixed(4)} payout=${opportunity.expected_payout.toFixed(4)} gross=${opportunity.gross_profit.toFixed(4)} fee=-${opportunity.fee.toFixed(4)}{' '}
            <span className="text-green-400">net=${opportunity.net_profit.toFixed(4)} roi={opportunity.roi_percent.toFixed(2)}%</span>
          </div>

          {/* AI details */}
          {judgment && (
            <>
              <div className="text-purple-400/70">
                AI_SCORE: {(judgment.overall_score * 100).toFixed(0)}/100
                {' '}P:{(judgment.profit_viability * 100).toFixed(0)}
                {' '}R:{(judgment.resolution_safety * 100).toFixed(0)}
                {' '}E:{(judgment.execution_feasibility * 100).toFixed(0)}
                {' '}M:{(judgment.market_efficiency * 100).toFixed(0)}
                {' '}<span className={cn("font-bold", recColor)}>{recommendation.replace('_', ' ').toUpperCase()}</span>
              </div>
              {judgment.reasoning && (
                <div className="text-purple-300/40 text-[10px]">
                  REASONING: {judgment.reasoning}
                </div>
              )}
            </>
          )}

          {/* Resolution */}
          {resolutions.length > 0 && resolutions[0].summary && (
            <div className="text-blue-400/50 text-[10px]">
              RESOLUTION: [{resolutions[0].recommendation}] C:{(resolutions[0].clarity_score * 100).toFixed(0)} R:{(resolutions[0].risk_score * 100).toFixed(0)}
              {' '}{resolutions[0].summary}
            </div>
          )}

          {/* Risk factors */}
          {opportunity.risk_factors.length > 0 && (
            <div className="text-yellow-400/50 text-[10px]">
              RISKS: {opportunity.risk_factors.join(' | ')}
            </div>
          )}

          {opportunity.description && (
            <div className="text-green-400/25 text-[10px]">
              DESC: {opportunity.description}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 pt-1">
            {polyUrl && (
              <a
                href={polyUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-[10px] text-blue-400/70 hover:text-blue-400 transition-colors underline underline-offset-2"
              >
                [polymarket]
              </a>
            )}
            {kalshiUrl && (
              <a
                href={kalshiUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-[10px] text-indigo-400/70 hover:text-indigo-400 transition-colors underline underline-offset-2"
              >
                [kalshi]
              </a>
            )}
            {onOpenCopilot && (
              <button
                onClick={(e) => { e.stopPropagation(); onOpenCopilot(opportunity) }}
                className="text-[10px] text-emerald-400/70 hover:text-emerald-400 transition-colors underline underline-offset-2"
              >
                [ask-ai]
              </button>
            )}
            {!judgment && !isPending && (
              <button
                onClick={(e) => { e.stopPropagation(); judgeMutation.mutate() }}
                disabled={judgeMutation.isPending}
                className="text-[10px] text-purple-400/70 hover:text-purple-400 transition-colors underline underline-offset-2"
              >
                [{judgeMutation.isPending ? 'analyzing...' : 'analyze'}]
              </button>
            )}
            {onExecute && (
              <button
                onClick={(e) => { e.stopPropagation(); onExecute(opportunity) }}
                className="text-[10px] text-green-400 hover:text-green-300 transition-colors font-bold underline underline-offset-2"
              >
                [EXECUTE]
              </button>
            )}
          </div>

          <div className="text-green-500/15">{'─'.repeat(72)}</div>
        </div>
      )}

      {/* Separator between entries */}
      {!isSelected && <div className="text-green-500/10 mt-0.5">{'─'.repeat(72)}</div>}
    </div>
  )
}
