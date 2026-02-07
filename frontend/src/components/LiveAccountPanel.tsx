import { useQuery } from '@tanstack/react-query'
import {
  Wallet,
  Briefcase,
  TrendingUp,
  TrendingDown,
  ExternalLink,
  RefreshCw,
  DollarSign,
} from 'lucide-react'
import clsx from 'clsx'
import {
  getTradingStatus,
  getTradingPositions,
  getTradingBalance,
} from '../services/api'

interface TradingPosition {
  token_id: string
  market_id: string
  market_question: string
  outcome: string
  size: number
  average_cost: number
  current_price: number
  unrealized_pnl: number
}

export default function LiveAccountPanel() {
  const { data: tradingStatus } = useQuery({
    queryKey: ['trading-status'],
    queryFn: getTradingStatus,
    refetchInterval: 10000,
  })

  const { data: livePositions = [], isLoading } = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    refetchInterval: 15000,
  })

  const { data: balance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
  })

  const positionsTotalValue = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.current_price, 0)
  const positionsCostBasis = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.average_cost, 0)
  const positionsUnrealizedPnl = livePositions.reduce((s: number, p: TradingPosition) => s + p.unrealized_pnl, 0)

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">Live Account</h2>
        <p className="text-sm text-gray-500">Your connected trading wallet and real positions</p>
      </div>

      {/* Wallet Info */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <p className="text-sm font-medium">Trading Wallet</p>
              <p className="text-xs text-gray-500 font-mono">
                {tradingStatus?.wallet_address
                  ? `${tradingStatus.wallet_address.slice(0, 10)}...${tradingStatus.wallet_address.slice(-8)}`
                  : 'Not connected'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-xs text-gray-500">USDC Balance</p>
              <p className="font-mono font-bold text-lg">${balance?.balance?.toFixed(2) || '0.00'}</p>
            </div>
            <div className={clsx(
              "px-3 py-1.5 rounded-lg text-xs font-medium",
              tradingStatus?.initialized ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-gray-400"
            )}>
              {tradingStatus?.initialized ? 'Connected' : 'Not Initialized'}
            </div>
          </div>
        </div>
      </div>

      {/* Account Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1">
            <DollarSign className="w-4 h-4 text-green-400" />
            <p className="text-xs text-gray-500">USDC Balance</p>
          </div>
          <p className="text-2xl font-mono font-bold">${balance?.balance?.toFixed(2) || '0.00'}</p>
        </div>
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1">
            <Briefcase className="w-4 h-4 text-blue-400" />
            <p className="text-xs text-gray-500">Open Positions</p>
          </div>
          <p className="text-2xl font-mono font-bold">{livePositions.length}</p>
        </div>
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1">
            <Briefcase className="w-4 h-4 text-purple-400" />
            <p className="text-xs text-gray-500">Positions Value</p>
          </div>
          <p className="text-2xl font-mono font-bold">${positionsTotalValue.toFixed(2)}</p>
        </div>
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1">
            {positionsUnrealizedPnl >= 0
              ? <TrendingUp className="w-4 h-4 text-green-400" />
              : <TrendingDown className="w-4 h-4 text-red-400" />
            }
            <p className="text-xs text-gray-500">Unrealized P&L</p>
          </div>
          <p className={clsx("text-2xl font-mono font-bold", positionsUnrealizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
            {positionsUnrealizedPnl >= 0 ? '+' : ''}${positionsUnrealizedPnl.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Positions Table */}
      {livePositions.length === 0 ? (
        <div className="text-center py-12 bg-[#141414] border border-gray-800 rounded-lg">
          <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-3" />
          <p className="text-gray-400">No open live trading positions</p>
          <p className="text-sm text-gray-600">Start auto-trading in the Trading tab to open positions</p>
        </div>
      ) : (
        <div className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <h3 className="font-medium text-sm">Open Positions</h3>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 text-xs">
                <th className="text-left px-4 py-3">Market</th>
                <th className="text-center px-3 py-3">Side</th>
                <th className="text-right px-3 py-3">Size</th>
                <th className="text-right px-3 py-3">Avg Cost</th>
                <th className="text-right px-3 py-3">Curr Price</th>
                <th className="text-right px-3 py-3">Cost Basis</th>
                <th className="text-right px-3 py-3">Mkt Value</th>
                <th className="text-right px-4 py-3">Unrealized P&L</th>
              </tr>
            </thead>
            <tbody>
              {livePositions.map((pos: TradingPosition, idx: number) => {
                const costBasis = pos.size * pos.average_cost
                const mktValue = pos.size * pos.current_price
                const pnlPct = costBasis > 0 ? (pos.unrealized_pnl / costBasis) * 100 : 0
                return (
                  <tr key={idx} className="border-b border-gray-800/50 hover:bg-[#1a1a1a] transition-colors">
                    <td className="px-4 py-3">
                      <p className="font-medium text-sm line-clamp-1">{pos.market_question}</p>
                      {pos.market_id && (
                        <a
                          href={`https://polymarket.com/event/${pos.market_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                        >
                          View <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </td>
                    <td className="text-center px-3 py-3">
                      <span className={clsx(
                        "px-2 py-0.5 rounded text-xs font-medium",
                        pos.outcome.toLowerCase() === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                      )}>
                        {pos.outcome.toUpperCase()}
                      </span>
                    </td>
                    <td className="text-right px-3 py-3 font-mono">{pos.size.toFixed(2)}</td>
                    <td className="text-right px-3 py-3 font-mono">${pos.average_cost.toFixed(4)}</td>
                    <td className="text-right px-3 py-3 font-mono">${pos.current_price.toFixed(4)}</td>
                    <td className="text-right px-3 py-3 font-mono">${costBasis.toFixed(2)}</td>
                    <td className="text-right px-3 py-3 font-mono">${mktValue.toFixed(2)}</td>
                    <td className="text-right px-4 py-3">
                      <span className={clsx("font-mono font-medium", pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                        {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                      </span>
                      <span className={clsx("text-xs ml-1", pnlPct >= 0 ? "text-green-400/70" : "text-red-400/70")}>
                        ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%)
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
            <tfoot>
              <tr className="border-t border-gray-700 font-medium">
                <td className="px-4 py-3 text-gray-400" colSpan={5}>Totals</td>
                <td className="text-right px-3 py-3 font-mono">${positionsCostBasis.toFixed(2)}</td>
                <td className="text-right px-3 py-3 font-mono">${positionsTotalValue.toFixed(2)}</td>
                <td className="text-right px-4 py-3">
                  <span className={clsx("font-mono font-medium", positionsUnrealizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
                    {positionsUnrealizedPnl >= 0 ? '+' : ''}${positionsUnrealizedPnl.toFixed(2)}
                  </span>
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      )}

      {/* Trading Limits */}
      {tradingStatus && (
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <h4 className="font-medium mb-3">Trading Safety Limits</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-500 text-xs">Max Trade Size</p>
              <p className="font-mono">${tradingStatus.limits.max_trade_size_usd}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Max Daily Volume</p>
              <p className="font-mono">${tradingStatus.limits.max_daily_volume}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Max Open Positions</p>
              <p className="font-mono">{tradingStatus.limits.max_open_positions}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Min Order Size</p>
              <p className="font-mono">${tradingStatus.limits.min_order_size_usd}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Max Slippage</p>
              <p className="font-mono">{tradingStatus.limits.max_slippage_percent}%</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
