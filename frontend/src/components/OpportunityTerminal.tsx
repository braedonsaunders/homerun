import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Terminal,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { getOpportunityPlatformLinks } from '../lib/marketUrls'
import { Opportunity, judgeOpportunity } from '../services/api'
import {
  STRATEGY_ABBREV,
  timeAgo,
  formatCompact,
} from './OpportunityCard'
import BuyButton from './BuyButton'

interface Props {
  opportunities: Opportunity[]
  onOpenCopilot?: (opportunity: Opportunity) => void
  isConnected?: boolean
  totalCount?: number
}

export default function OpportunityTerminal({ opportunities, onOpenCopilot, isConnected, totalCount }: Props) {
  const { t } = useTranslation()
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Terminal-style blinking cursor
  const [cursorVisible, setCursorVisible] = useState(true)
  useEffect(() => {
    const iv = setInterval(() => setCursorVisible(v => !v), 530)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="terminal-view terminal-surface border rounded-lg overflow-hidden font-data text-[11px] leading-relaxed">
      {/* Terminal Header */}
      <div className="terminal-header flex items-center justify-between px-3 py-1.5">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-green-400" />
          <span className="text-green-400 font-bold text-xs">{t('opportunityTerminal.scannerTitle')}</span>
          <span className="text-green-400/40">v2.0</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-green-400/60">
            {t('opportunityTerminal.opportunitiesCount', { n: totalCount ?? opportunities.length })}
          </span>
          <div className="flex items-center gap-1">
            <div className={cn(
              "w-1.5 h-1.5 rounded-full",
              isConnected ? "bg-green-400 live-dot" : "bg-red-400"
            )} />
            <span className={cn("text-[10px]", isConnected ? "text-green-400/70" : "text-red-400/70")}>
              {isConnected ? t('opportunityTerminal.live') : t('opportunityTerminal.disconnected')}
            </span>
          </div>
        </div>
      </div>

      {/* Terminal Body */}
      <div ref={scrollRef} className="max-h-[calc(100vh-280px)] overflow-y-auto p-3 space-y-0">
        {/* Boot sequence header */}
        <div className="text-green-500/30 mb-3 space-y-0.5">
          <p>{'>'} {t('opportunityTerminal.bootInitializing')}</p>
          <p>{'>'} {t('opportunityTerminal.bootConnected')}</p>
          <p>{'>'} {t('opportunityTerminal.bootLoaded', { n: opportunities.length })}</p>
          <p className="text-green-500/15">{'─'.repeat(72)}</p>
        </div>

        {/* Opportunities */}
        {opportunities.map((opp, idx) => (
          <TerminalEntry
            key={opp.stable_id || opp.id}
            opportunity={opp}
            isSelected={selectedIdx === idx}
            onSelect={() => setSelectedIdx(selectedIdx === idx ? null : idx)}
            onOpenCopilot={onOpenCopilot}
          />
        ))}

        {/* Cursor line */}
        <div className="text-green-400/60 mt-2 flex items-center">
          <span className="text-green-400/30">{'>'} </span>
          <span className="text-green-400/40">{t('opportunityTerminal.awaitingNextScan')}</span>
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
  onOpenCopilot,
}: {
  opportunity: Opportunity
  isSelected: boolean
  onSelect: () => void
  onOpenCopilot?: (opportunity: Opportunity) => void
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const translateRecommendation = (rec: string): string => rec
    ? t(`opportunityCard.recommendation.${rec}`, { defaultValue: rec.replace('_', ' ').toUpperCase() })
    : ''
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
        : 'text-green-500/35'

  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

  const { polymarketUrl: polyUrl, kalshiUrl } = getOpportunityPlatformLinks(opportunity as any)

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
        <span className="text-green-400 font-bold mr-2">{t('opportunityTerminal.roiInline', { value: roiStr })}</span>
        <span className="text-green-300/80 mr-2">{t('opportunityTerminal.netInline', { value: formatCompact(opportunity.net_profit) })}</span>
        <span className={cn("mr-2", riskColor)}>{t('opportunityTerminal.riskInline', { value: riskPct })}</span>
        {judgment && (
          <span className={cn("font-bold mr-2", recColor)}>
            {t('opportunityTerminal.aiInline', { score: (judgment.overall_score * 100).toFixed(0), recommendation: translateRecommendation(recommendation) })}
          </span>
        )}
        <span className="text-green-500/25 ml-auto">{t('opportunityTerminal.timeAgoSuffix', { time: timeAgo(opportunity.detected_at, t) })}</span>
      </div>

      {/* Title */}
      <div className="text-green-100/70 pl-4 truncate">
        &quot;{opportunity.title}&quot;
      </div>

      {/* Positions line */}
      <div className="text-green-400/50 pl-4">
        {t('opportunityTerminal.positionsLabel')}{' '}
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
        {t('opportunityTerminal.mktsInline', { value: opportunity.markets.length })}
        {' | '}{t('opportunityTerminal.liqInline', { value: formatCompact(opportunity.min_liquidity) })}
        {' | '}{t('opportunityTerminal.costInline', { value: formatCompact(opportunity.total_cost) })}
        {' | '}{t('opportunityTerminal.maxInline', { value: formatCompact(opportunity.max_position_size) })}
        {opportunity.category && <>{' | '}{t('opportunityTerminal.catInline', { value: opportunity.category.toUpperCase() })}</>}
      </div>

      {/* Expanded details */}
      {isSelected && (
        <div className="pl-4 mt-1 space-y-1 border-l-2 border-green-500/20 ml-1">
          {/* Market details */}
          {opportunity.markets.map((mkt, i) => (
            <div key={i} className="text-green-400/50">
              {t('opportunityTerminal.marketLine', { i, yes: mkt.yes_price.toFixed(4), no: mkt.no_price.toFixed(4), liq: formatCompact(mkt.liquidity) })}
              <span className="text-green-400/25 truncate ml-2">{mkt.question}</span>
            </div>
          ))}

          {/* Profit breakdown */}
          <div className="text-green-300/50">
            {t('opportunityTerminal.profitLine', { cost: opportunity.total_cost.toFixed(4), payout: opportunity.expected_payout.toFixed(4), gross: opportunity.gross_profit.toFixed(4), fee: opportunity.fee.toFixed(4) })}{' '}
            <span className="text-green-400">{t('opportunityTerminal.profitNetRoi', { net: opportunity.net_profit.toFixed(4), roi: opportunity.roi_percent.toFixed(2) })}</span>
          </div>

          {/* AI details */}
          {judgment && (
            <>
              <div className="text-purple-400/70">
                {t('opportunityTerminal.aiScoreLine', {
                  score: (judgment.overall_score * 100).toFixed(0),
                  p: (judgment.profit_viability * 100).toFixed(0),
                  r: (judgment.resolution_safety * 100).toFixed(0),
                  e: (judgment.execution_feasibility * 100).toFixed(0),
                  m: (judgment.market_efficiency * 100).toFixed(0),
                })}
                {' '}<span className={cn("font-bold", recColor)}>{translateRecommendation(recommendation)}</span>
              </div>
              {judgment.reasoning && (
                <div className="text-purple-300/40 text-[10px]">
                  {t('opportunityTerminal.reasoningLabel')} {judgment.reasoning}
                </div>
              )}
            </>
          )}

          {/* Resolution */}
          {resolutions.length > 0 && resolutions[0].summary && (
            <div className="text-blue-400/50 text-[10px]">
              {t('opportunityTerminal.resolutionLine', {
                recommendation: translateRecommendation(resolutions[0].recommendation),
                clarity: (resolutions[0].clarity_score * 100).toFixed(0),
                risk: (resolutions[0].risk_score * 100).toFixed(0),
              })}
              {' '}{resolutions[0].summary}
            </div>
          )}

          {/* Risk factors */}
          {opportunity.risk_factors.length > 0 && (
            <div className="text-yellow-400/50 text-[10px]">
              {t('opportunityTerminal.risksLabel')} {opportunity.risk_factors.join(' | ')}
            </div>
          )}

          {opportunity.description && (
            <div className="text-green-400/25 text-[10px]">
              {t('opportunityTerminal.descLabel')} {opportunity.description}
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
                {t('opportunityTerminal.actionPolymarket')}
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
                {t('opportunityTerminal.actionKalshi')}
              </a>
            )}
            {onOpenCopilot && (
              <button
                onClick={(e) => { e.stopPropagation(); onOpenCopilot(opportunity) }}
                className="text-[10px] text-emerald-400/70 hover:text-emerald-400 transition-colors underline underline-offset-2"
              >
                {t('opportunityTerminal.actionAskAi')}
              </button>
            )}
            {!judgment && !isPending && (
              <button
                onClick={(e) => { e.stopPropagation(); judgeMutation.mutate() }}
                disabled={judgeMutation.isPending}
                className="text-[10px] text-purple-400/70 hover:text-purple-400 transition-colors underline underline-offset-2"
              >
                {judgeMutation.isPending ? t('opportunityTerminal.actionAnalyzing') : t('opportunityTerminal.actionAnalyze')}
              </button>
            )}
            <BuyButton opportunity={opportunity} className="w-20" />
          </div>

          <div className="text-green-500/15">{'─'.repeat(72)}</div>
        </div>
      )}

      {/* Separator between entries */}
      {!isSelected && <div className="text-green-500/10 mt-0.5">{'─'.repeat(72)}</div>}
    </div>
  )
}
