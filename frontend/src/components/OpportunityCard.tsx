import { useState } from 'react'
import {
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  TrendingUp,
  DollarSign,
  Target,
  ExternalLink,
  Play
} from 'lucide-react'
import clsx from 'clsx'
import { Opportunity } from '../services/api'

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

interface Props {
  opportunity: Opportunity
  onExecute?: (opportunity: Opportunity) => void
}

export default function OpportunityCard({ opportunity, onExecute }: Props) {
  const [expanded, setExpanded] = useState(false)

  const riskColor = opportunity.risk_score < 0.3
    ? 'text-green-400'
    : opportunity.risk_score < 0.6
      ? 'text-yellow-400'
      : 'text-red-400'

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
