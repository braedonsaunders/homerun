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
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import {
  getTradingStatus,
  getTradingPositions,
  getTradingBalance,
} from '../services/api'
import type { TradingPosition } from '../services/api'

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
    enabled: !!tradingStatus?.initialized,
    retry: false,
  })

  const positionsTotalValue = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.current_price, 0)
  const positionsCostBasis = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.average_cost, 0)
  const positionsUnrealizedPnl = livePositions.reduce((s: number, p: TradingPosition) => s + p.unrealized_pnl, 0)

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">Live Account</h2>
        <p className="text-sm text-muted-foreground">Your connected trading wallet and real positions</p>
      </div>

      {/* Wallet Info */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <p className="text-sm font-medium">Trading Wallet</p>
              <p className="text-xs text-muted-foreground font-mono">
                {tradingStatus?.wallet_address
                  ? `${tradingStatus.wallet_address.slice(0, 10)}...${tradingStatus.wallet_address.slice(-8)}`
                  : 'Not connected'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-xs text-muted-foreground">USDC Balance</p>
              <p className="font-mono font-bold text-lg">${balance?.balance?.toFixed(2) || '0.00'}</p>
            </div>
            <Badge
              variant="outline"
              className={cn(
                "rounded-lg border-transparent px-3 py-1.5 text-xs font-medium",
                tradingStatus?.initialized ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-muted-foreground"
              )}
            >
              {tradingStatus?.initialized ? 'Connected' : 'Not Initialized'}
            </Badge>
          </div>
        </div>
      </Card>

      {/* Account Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <DollarSign className="w-4 h-4 text-green-400" />
            <p className="text-xs text-muted-foreground">USDC Balance</p>
          </div>
          <p className="text-2xl font-mono font-bold">${balance?.balance?.toFixed(2) || '0.00'}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Briefcase className="w-4 h-4 text-blue-400" />
            <p className="text-xs text-muted-foreground">Open Positions</p>
          </div>
          <p className="text-2xl font-mono font-bold">{livePositions.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Briefcase className="w-4 h-4 text-purple-400" />
            <p className="text-xs text-muted-foreground">Positions Value</p>
          </div>
          <p className="text-2xl font-mono font-bold">${positionsTotalValue.toFixed(2)}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-1">
            {positionsUnrealizedPnl >= 0
              ? <TrendingUp className="w-4 h-4 text-green-400" />
              : <TrendingDown className="w-4 h-4 text-red-400" />
            }
            <p className="text-xs text-muted-foreground">Unrealized P&L</p>
          </div>
          <p className={cn("text-2xl font-mono font-bold", positionsUnrealizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
            {positionsUnrealizedPnl >= 0 ? '+' : ''}${positionsUnrealizedPnl.toFixed(2)}
          </p>
        </Card>
      </div>

      {/* Positions Table */}
      {livePositions.length === 0 ? (
        <Card className="text-center py-12">
          <Briefcase className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">No open live trading positions</p>
          <p className="text-sm text-muted-foreground">Start auto-trading in the Trading tab to open positions</p>
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="font-medium text-sm">Open Positions</h3>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-muted-foreground text-xs">
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
                  <tr key={idx} className="border-b border-border/50 hover:bg-muted transition-colors">
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
                      <Badge
                        variant="outline"
                        className={cn(
                          "rounded border-transparent",
                          pos.outcome.toLowerCase() === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                        )}
                      >
                        {pos.outcome.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="text-right px-3 py-3 font-mono">{pos.size.toFixed(2)}</td>
                    <td className="text-right px-3 py-3 font-mono">${pos.average_cost.toFixed(4)}</td>
                    <td className="text-right px-3 py-3 font-mono">${pos.current_price.toFixed(4)}</td>
                    <td className="text-right px-3 py-3 font-mono">${costBasis.toFixed(2)}</td>
                    <td className="text-right px-3 py-3 font-mono">${mktValue.toFixed(2)}</td>
                    <td className="text-right px-4 py-3">
                      <span className={cn("font-mono font-medium", pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                        {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                      </span>
                      <span className={cn("text-xs ml-1", pnlPct >= 0 ? "text-green-400/70" : "text-red-400/70")}>
                        ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%)
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
            <tfoot>
              <tr className="border-t border-border font-medium">
                <td className="px-4 py-3 text-muted-foreground" colSpan={5}>Totals</td>
                <td className="text-right px-3 py-3 font-mono">${positionsCostBasis.toFixed(2)}</td>
                <td className="text-right px-3 py-3 font-mono">${positionsTotalValue.toFixed(2)}</td>
                <td className="text-right px-4 py-3">
                  <span className={cn("font-mono font-medium", positionsUnrealizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
                    {positionsUnrealizedPnl >= 0 ? '+' : ''}${positionsUnrealizedPnl.toFixed(2)}
                  </span>
                </td>
              </tr>
            </tfoot>
          </table>
        </Card>
      )}

      {/* Trading Limits */}
      {tradingStatus && (
        <Card className="p-4">
          <h4 className="font-medium mb-3">Trading Safety Limits</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground text-xs">Max Trade Size</p>
              <p className="font-mono">${tradingStatus.limits.max_trade_size_usd}</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Max Daily Volume</p>
              <p className="font-mono">${tradingStatus.limits.max_daily_volume}</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Max Open Positions</p>
              <p className="font-mono">{tradingStatus.limits.max_open_positions}</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Min Order Size</p>
              <p className="font-mono">${tradingStatus.limits.min_order_size_usd}</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Max Slippage</p>
              <p className="font-mono">{tradingStatus.limits.max_slippage_percent}%</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
