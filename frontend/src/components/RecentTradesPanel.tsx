import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Clock,
  TrendingUp,
  TrendingDown,
  Wallet,
  RefreshCw,
  AlertCircle,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Zap,
  Filter,
  DollarSign,
  Hash
} from 'lucide-react'
import { cn } from '../lib/utils'
import { getRecentTradesFromWallets, RecentTradeFromWallet } from '../services/api'

interface Props {
  onNavigateToWallet?: (address: string) => void
}

export default function RecentTradesPanel({ onNavigateToWallet }: Props) {
  const [hoursFilter, setHoursFilter] = useState(24)
  const [sideFilter, setSideFilter] = useState<'all' | 'BUY' | 'SELL'>('all')
  const [tradeLimit, setTradeLimit] = useState(100)
  const [expandedTrades, setExpandedTrades] = useState<Set<string>>(new Set())

  const { data, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ['recent-trades-from-wallets', hoursFilter, tradeLimit],
    queryFn: () => getRecentTradesFromWallets({ limit: tradeLimit, hours: hoursFilter }),
    refetchInterval: 30000,
  })

  const trades = data?.trades || []
  const filteredTrades = sideFilter === 'all'
    ? trades
    : trades.filter(t => t.side?.toUpperCase() === sideFilter)

  const toggleExpanded = (tradeId: string) => {
    const newExpanded = new Set(expandedTrades)
    if (newExpanded.has(tradeId)) {
      newExpanded.delete(tradeId)
    } else {
      newExpanded.add(tradeId)
    }
    setExpandedTrades(newExpanded)
  }

  const parseTimestamp = (trade: RecentTradeFromWallet): Date | null => {
    // Prefer the normalized ISO timestamp from the backend
    const candidates = [
      trade.timestamp_iso,
      trade.match_time,
      trade.timestamp,
      trade.time,
      trade.created_at,
    ]

    for (const ts of candidates) {
      if (!ts) continue
      try {
        // If it's a string containing date separators, parse as ISO
        if (typeof ts === 'string' && (ts.includes('T') || ts.includes('-'))) {
          const date = new Date(ts)
          if (!isNaN(date.getTime()) && date.getFullYear() > 2000) return date
        }
        // If it's a numeric string or number, treat as Unix seconds
        const num = Number(ts)
        if (!isNaN(num) && num > 0) {
          // If the number is too small to be milliseconds (before year 2001 in ms),
          // it's likely seconds
          const ms = num < 4102444800 ? num * 1000 : num
          const date = new Date(ms)
          if (!isNaN(date.getTime()) && date.getFullYear() > 2000) return date
        }
      } catch {
        continue
      }
    }
    return null
  }

  const formatTimestamp = (trade: RecentTradeFromWallet) => {
    const date = parseTimestamp(trade)
    if (!date) return 'Unknown'

    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)

    if (diffMs < 0) return 'Just now' // future timestamp (clock skew)
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffHours < 48) return 'Yesterday'
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  const formatFullTimestamp = (trade: RecentTradeFromWallet) => {
    const date = parseTimestamp(trade)
    if (!date) return 'Unknown'
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    })
  }

  const getTradeId = (trade: RecentTradeFromWallet, index: number) => {
    return trade.id || trade.transaction_hash || `trade-${index}`
  }

  const getMarketName = (trade: RecentTradeFromWallet) => {
    return trade.market_title || trade.market || 'Unknown Market'
  }

  const isConditionId = (value: string) => {
    return value.startsWith('0x') && value.length > 20
  }

  const formatCurrency = (value: number) => {
    if (value >= 1000) return `$${(value / 1000).toFixed(1)}k`
    if (value >= 100) return `$${value.toFixed(0)}`
    return `$${value.toFixed(2)}`
  }

  const totalVolume = filteredTrades.reduce((sum, t) => {
    const size = t.size ?? 0
    const price = t.price ?? 0
    return sum + size * price
  }, 0)

  const buyCount = filteredTrades.filter(t => t.side?.toUpperCase() === 'BUY').length
  const sellCount = filteredTrades.filter(t => t.side?.toUpperCase() === 'SELL').length

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/10 rounded-lg">
            <Zap className="w-5 h-5 text-orange-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Recent Wallet Trades</h2>
            <p className="text-sm text-muted-foreground/70">
              Live feed from {data?.tracked_wallets || 0} tracked wallets
            </p>
          </div>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isRefetching}
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm",
            "bg-muted text-foreground/80 hover:bg-accent transition-colors",
            isRefetching && "opacity-50"
          )}
        >
          <RefreshCw className={cn("w-4 h-4", isRefetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 p-3 bg-card rounded-lg border border-border">
        <Filter className="w-4 h-4 text-muted-foreground/70" />
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Time:</span>
          <select
            value={hoursFilter}
            onChange={(e) => setHoursFilter(Number(e.target.value))}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value={1}>Last hour</option>
            <option value={6}>Last 6 hours</option>
            <option value={24}>Last 24 hours</option>
            <option value={48}>Last 48 hours</option>
            <option value={168}>Last 7 days</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Side:</span>
          <select
            value={sideFilter}
            onChange={(e) => setSideFilter(e.target.value as 'all' | 'BUY' | 'SELL')}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value="all">All</option>
            <option value="BUY">Buys only</option>
            <option value="SELL">Sells only</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Show:</span>
          <select
            value={tradeLimit}
            onChange={(e) => setTradeLimit(Number(e.target.value))}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value={50}>50 trades</option>
            <option value={100}>100 trades</option>
            <option value={200}>200 trades</option>
            <option value={500}>500 trades</option>
          </select>
        </div>
      </div>

      {/* Summary Stats */}
      {filteredTrades.length > 0 && (
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-card border border-border rounded-lg p-3">
            <p className="text-xs text-muted-foreground/70">Total Trades</p>
            <p className="text-lg font-semibold text-foreground">{filteredTrades.length}</p>
          </div>
          <div className="bg-card border border-border rounded-lg p-3">
            <p className="text-xs text-muted-foreground/70">Volume</p>
            <p className="text-lg font-semibold text-foreground">{formatCurrency(totalVolume)}</p>
          </div>
          <div className="bg-card border border-border rounded-lg p-3">
            <p className="text-xs text-green-500">Buys</p>
            <p className="text-lg font-semibold text-green-400">{buyCount}</p>
          </div>
          <div className="bg-card border border-border rounded-lg p-3">
            <p className="text-xs text-red-500">Sells</p>
            <p className="text-lg font-semibold text-red-400">{sellCount}</p>
          </div>
        </div>
      )}

      {/* Trades List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground/70" />
        </div>
      ) : filteredTrades.length === 0 ? (
        <div className="text-center py-12 bg-card rounded-lg border border-border">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <p className="text-muted-foreground">No recent trades found</p>
          <p className="text-sm text-muted-foreground/50 mt-1">
            {data?.tracked_wallets === 0
              ? 'Start tracking wallets to see their trades here'
              : 'Try expanding the time window or changing filters'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredTrades.map((trade, index) => {
            const tradeId = getTradeId(trade, index)
            const isExpanded = expandedTrades.has(tradeId)
            const isBuy = trade.side?.toUpperCase() === 'BUY'
            const marketName = getMarketName(trade)
            const hasRealMarketName = trade.market_title && !isConditionId(trade.market_title)
            const cost = trade.cost ?? (trade.size ?? 0) * (trade.price ?? 0)

            return (
              <div
                key={tradeId}
                className="bg-card border border-border rounded-lg overflow-hidden hover:border-border transition-colors"
              >
                {/* Trade Row */}
                <div
                  className="p-3 cursor-pointer"
                  onClick={() => toggleExpanded(tradeId)}
                >
                  <div className="flex items-center gap-3">
                    {/* Side indicator */}
                    <div className={cn(
                      "flex-shrink-0 w-1 h-12 rounded-full",
                      isBuy ? "bg-green-500" : "bg-red-500"
                    )} />

                    {/* Main content */}
                    <div className="flex-1 min-w-0">
                      {/* Top row: side badge, market name, time */}
                      <div className="flex items-center gap-2 mb-1">
                        <span className={cn(
                          "flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-bold",
                          isBuy
                            ? "bg-green-500/10 text-green-400"
                            : "bg-red-500/10 text-red-400"
                        )}>
                          {isBuy ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                          {trade.side?.toUpperCase() || 'TRADE'}
                        </span>

                        {trade.outcome && (
                          <span className={cn(
                            "px-1.5 py-0.5 rounded text-xs font-medium",
                            trade.outcome === 'Yes'
                              ? "bg-blue-500/10 text-blue-400"
                              : "bg-purple-500/10 text-purple-400"
                          )}>
                            {trade.outcome}
                          </span>
                        )}

                        <span className="flex items-center gap-1 text-xs text-muted-foreground/70 ml-auto flex-shrink-0">
                          <Clock className="w-3 h-3" />
                          {formatTimestamp(trade)}
                        </span>
                      </div>

                      {/* Market name */}
                      <h3 className={cn(
                        "font-medium text-sm truncate",
                        hasRealMarketName ? "text-foreground" : "text-muted-foreground/70"
                      )}>
                        {hasRealMarketName ? marketName : (
                          isConditionId(marketName) ? `Market ${marketName.slice(0, 10)}...` : marketName
                        )}
                      </h3>

                      {/* Wallet */}
                      <div className="flex items-center gap-2 mt-1">
                        <Wallet className="w-3 h-3 text-muted-foreground/50" />
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            onNavigateToWallet?.(trade.wallet_address)
                          }}
                          className="text-xs text-orange-400 hover:text-orange-300 hover:underline truncate"
                        >
                          {trade.wallet_username || trade.wallet_label}
                        </button>
                        {trade.wallet_username && trade.wallet_username !== trade.wallet_label && (
                          <span className="text-xs text-muted-foreground/50 font-mono">
                            {trade.wallet_address.slice(0, 6)}...{trade.wallet_address.slice(-4)}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Price info */}
                    <div className="flex-shrink-0 text-right">
                      <div className="flex items-center gap-1 justify-end">
                        <span className="text-xs text-muted-foreground/70">@</span>
                        <span className={cn(
                          "text-sm font-semibold",
                          isBuy ? "text-green-400" : "text-red-400"
                        )}>
                          {((trade.price ?? 0) * 100).toFixed(1)}c
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {(trade.size ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })} shares
                      </p>
                      <p className="text-xs text-muted-foreground/70">
                        <DollarSign className="w-3 h-3 inline" />
                        {cost.toFixed(2)}
                      </p>
                    </div>

                    {/* Expand Icon */}
                    <div className="flex-shrink-0 self-center">
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-muted-foreground/50" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-muted-foreground/50" />
                      )}
                    </div>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-border p-4 bg-background">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground/70 text-xs">Price</p>
                        <p className="font-mono text-foreground">
                          ${(trade.price ?? 0).toFixed(4)}
                          <span className="text-muted-foreground/70 ml-1">
                            ({((trade.price ?? 0) * 100).toFixed(1)}c)
                          </span>
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground/70 text-xs">Shares</p>
                        <p className="font-mono text-foreground">
                          {(trade.size ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground/70 text-xs">Total Cost</p>
                        <p className="font-mono text-foreground">${cost.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground/70 text-xs">Outcome</p>
                        <p className={cn(
                          "font-medium",
                          trade.outcome === 'Yes' ? 'text-green-400' : 'text-red-400'
                        )}>
                          {trade.outcome || 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground/70 text-xs">Time</p>
                        <p className="font-mono text-foreground/80 text-xs">{formatFullTimestamp(trade)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground/70 text-xs">Wallet</p>
                        <p className="font-mono text-foreground/80 text-xs">
                          {trade.wallet_address.slice(0, 6)}...{trade.wallet_address.slice(-4)}
                        </p>
                      </div>
                      {trade.market && isConditionId(trade.market) && (
                        <div className="col-span-2">
                          <p className="text-muted-foreground/70 text-xs">Condition ID</p>
                          <p className="font-mono text-muted-foreground text-xs truncate">{trade.market}</p>
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-3 mt-4 pt-3 border-t border-border">
                      {trade.market_slug && (
                        <a
                          href={`https://polymarket.com/event/${trade.market_slug}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
                        >
                          <ExternalLink className="w-3 h-3" />
                          View on Polymarket
                        </a>
                      )}
                      {trade.transaction_hash && (
                        <a
                          href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground/80"
                        >
                          <ExternalLink className="w-3 h-3" />
                          Transaction
                        </a>
                      )}
                      {trade.wallet_address && (
                        <a
                          href={`https://polygonscan.com/address/${trade.wallet_address}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground/80"
                        >
                          <Hash className="w-3 h-3" />
                          Wallet
                        </a>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
