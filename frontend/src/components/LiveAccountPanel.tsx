import { useAtom } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import {
  Wallet,
  Briefcase,
  TrendingUp,
  TrendingDown,
  ExternalLink,
  RefreshCw,
  DollarSign,
  BarChart3,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { selectedAccountIdAtom } from '../store/atoms'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import {
  getTradingStatus,
  getTradingPositions,
  getTradingBalance,
  getKalshiStatus,
  getKalshiPositions,
  getKalshiBalance,
} from '../services/api'
import type { TradingPosition, KalshiPosition } from '../services/api'

export default function LiveAccountPanel() {
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const platform = selectedAccountId === 'live:kalshi' ? 'kalshi' : 'polymarket'

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">Live Account</h2>
        <p className="text-sm text-muted-foreground">
          Viewing {platform === 'polymarket' ? 'Polymarket' : 'Kalshi'} — change account in the header
        </p>
      </div>

      {platform === 'polymarket' ? (
        <PolymarketAccount />
      ) : (
        <KalshiAccount />
      )}
    </div>
  )
}

// ==================== POLYMARKET ACCOUNT ====================

function PolymarketAccount() {
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
      {/* Wallet Info */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-medium">Polymarket Wallet</p>
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
      <PositionsTable
        positions={livePositions}
        platform="polymarket"
        positionsCostBasis={positionsCostBasis}
        positionsTotalValue={positionsTotalValue}
        positionsUnrealizedPnl={positionsUnrealizedPnl}
      />

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

// ==================== KALSHI ACCOUNT ====================

function KalshiAccount() {
  const { data: kalshiStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['kalshi-status'],
    queryFn: getKalshiStatus,
    refetchInterval: 10000,
    retry: false,
  })

  const { data: kalshiPositions = [], isLoading: positionsLoading } = useQuery({
    queryKey: ['kalshi-positions'],
    queryFn: getKalshiPositions,
    refetchInterval: 15000,
    enabled: !!kalshiStatus?.authenticated,
    retry: false,
  })

  const { data: kalshiBalance } = useQuery({
    queryKey: ['kalshi-balance'],
    queryFn: getKalshiBalance,
    enabled: !!kalshiStatus?.authenticated,
    refetchInterval: 15000,
    retry: false,
  })

  const positionsTotalValue = kalshiPositions.reduce((s: number, p: KalshiPosition) => s + p.size * p.current_price, 0)
  const positionsCostBasis = kalshiPositions.reduce((s: number, p: KalshiPosition) => s + p.size * p.average_cost, 0)
  const positionsUnrealizedPnl = kalshiPositions.reduce((s: number, p: KalshiPosition) => s + p.unrealized_pnl, 0)

  if (statusLoading || positionsLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Account Info */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-500/20 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <p className="text-sm font-medium">Kalshi Account</p>
              <p className="text-xs text-muted-foreground font-mono">
                {kalshiStatus?.email || (kalshiStatus?.member_id ? `Member: ${kalshiStatus.member_id}` : 'Not connected')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-xs text-muted-foreground">USD Balance</p>
              <p className="font-mono font-bold text-lg">
                ${kalshiBalance?.balance?.toFixed(2) ?? kalshiStatus?.balance?.balance?.toFixed(2) ?? '0.00'}
              </p>
            </div>
            <Badge
              variant="outline"
              className={cn(
                "rounded-lg border-transparent px-3 py-1.5 text-xs font-medium",
                kalshiStatus?.authenticated ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-muted-foreground"
              )}
            >
              {kalshiStatus?.authenticated ? 'Connected' : 'Not Connected'}
            </Badge>
          </div>
        </div>
      </Card>

      {!kalshiStatus?.authenticated ? (
        <Card className="text-center py-12">
          <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">Kalshi account not connected</p>
          <p className="text-sm text-muted-foreground mt-1">
            Configure your Kalshi credentials in the Settings tab to connect your account
          </p>
        </Card>
      ) : (
        <>
          {/* Account Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-1">
                <DollarSign className="w-4 h-4 text-green-400" />
                <p className="text-xs text-muted-foreground">USD Balance</p>
              </div>
              <p className="text-2xl font-mono font-bold">
                ${kalshiBalance?.balance?.toFixed(2) ?? kalshiStatus?.balance?.balance?.toFixed(2) ?? '0.00'}
              </p>
            </Card>
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-1">
                <Briefcase className="w-4 h-4 text-indigo-400" />
                <p className="text-xs text-muted-foreground">Open Positions</p>
              </div>
              <p className="text-2xl font-mono font-bold">{kalshiPositions.length}</p>
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
          <PositionsTable
            positions={kalshiPositions}
            platform="kalshi"
            positionsCostBasis={positionsCostBasis}
            positionsTotalValue={positionsTotalValue}
            positionsUnrealizedPnl={positionsUnrealizedPnl}
          />
        </>
      )}
    </div>
  )
}

// ==================== SHARED POSITIONS TABLE ====================

interface PositionsTableProps {
  positions: (TradingPosition | KalshiPosition)[]
  platform: 'polymarket' | 'kalshi'
  positionsCostBasis: number
  positionsTotalValue: number
  positionsUnrealizedPnl: number
}

function PositionsTable({ positions, platform, positionsCostBasis, positionsTotalValue, positionsUnrealizedPnl }: PositionsTableProps) {
  if (positions.length === 0) {
    return (
      <Card className="text-center py-12">
        <Briefcase className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
        <p className="text-muted-foreground">No open positions on {platform === 'polymarket' ? 'Polymarket' : 'Kalshi'}</p>
        <p className="text-sm text-muted-foreground">
          {platform === 'polymarket'
            ? 'Start auto-trading in the Trading tab to open positions'
            : 'Cross-platform opportunities will open positions here'}
        </p>
      </Card>
    )
  }

  const getMarketLink = (pos: TradingPosition | KalshiPosition) => {
    if (platform === 'kalshi') {
      const ticker = pos.market_id.toLowerCase()
      const eventTicker = ticker.split('-')[0]
      return `https://kalshi.com/markets/${eventTicker}/${ticker}`
    }
    return `https://polymarket.com/event/${pos.market_id}`
  }

  return (
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
          {positions.map((pos, idx) => {
            const costBasis = pos.size * pos.average_cost
            const mktValue = pos.size * pos.current_price
            const pnlPct = costBasis > 0 ? (pos.unrealized_pnl / costBasis) * 100 : 0
            return (
              <tr key={idx} className="border-b border-border/50 hover:bg-muted transition-colors">
                <td className="px-4 py-3">
                  <p className="font-medium text-sm line-clamp-1">{pos.market_question}</p>
                  {pos.market_id && (
                    <a
                      href={getMarketLink(pos)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={cn(
                        "text-xs hover:opacity-80 flex items-center gap-1",
                        platform === 'kalshi' ? "text-indigo-400" : "text-blue-400"
                      )}
                    >
                      View on {platform === 'kalshi' ? 'Kalshi' : 'Polymarket'} <ExternalLink className="w-3 h-3" />
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
  )
}
