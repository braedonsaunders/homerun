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
  Filter
} from 'lucide-react'
import clsx from 'clsx'
import { getRecentTradesFromWallets, RecentTradeFromWallet } from '../services/api'

interface Props {
  onNavigateToWallet?: (address: string) => void
}

export default function RecentTradesPanel({ onNavigateToWallet }: Props) {
  const [hoursFilter, setHoursFilter] = useState(24)
  const [sideFilter, setSideFilter] = useState<'all' | 'BUY' | 'SELL'>('all')
  const [expandedTrades, setExpandedTrades] = useState<Set<string>>(new Set())

  const { data, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ['recent-trades-from-wallets', hoursFilter],
    queryFn: () => getRecentTradesFromWallets({ limit: 100, hours: hoursFilter }),
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

  const formatTimestamp = (trade: RecentTradeFromWallet) => {
    const ts = trade.timestamp || trade.time || trade.created_at
    if (!ts) return 'Unknown'

    try {
      const date = new Date(ts)
      const now = new Date()
      const diffMs = now.getTime() - date.getTime()
      const diffMins = Math.floor(diffMs / 60000)
      const diffHours = Math.floor(diffMins / 60)

      if (diffMins < 1) return 'Just now'
      if (diffMins < 60) return `${diffMins}m ago`
      if (diffHours < 24) return `${diffHours}h ago`
      return date.toLocaleDateString()
    } catch {
      return 'Unknown'
    }
  }

  const getTradeId = (trade: RecentTradeFromWallet, index: number) => {
    return trade.id || trade.transaction_hash || `trade-${index}`
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/10 rounded-lg">
            <Zap className="w-5 h-5 text-orange-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Recent Wallet Trades</h2>
            <p className="text-sm text-gray-500">
              Live feed from {data?.tracked_wallets || 0} tracked wallets
            </p>
          </div>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isRefetching}
          className={clsx(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm",
            "bg-[#1a1a1a] text-gray-300 hover:bg-gray-700 transition-colors",
            isRefetching && "opacity-50"
          )}
        >
          <RefreshCw className={clsx("w-4 h-4", isRefetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 p-3 bg-[#141414] rounded-lg border border-gray-800">
        <Filter className="w-4 h-4 text-gray-500" />
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Time:</span>
          <select
            value={hoursFilter}
            onChange={(e) => setHoursFilter(Number(e.target.value))}
            className="bg-[#1a1a1a] border border-gray-700 rounded px-2 py-1 text-sm"
          >
            <option value={1}>Last hour</option>
            <option value={6}>Last 6 hours</option>
            <option value={24}>Last 24 hours</option>
            <option value={48}>Last 48 hours</option>
            <option value={168}>Last 7 days</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Side:</span>
          <select
            value={sideFilter}
            onChange={(e) => setSideFilter(e.target.value as 'all' | 'BUY' | 'SELL')}
            className="bg-[#1a1a1a] border border-gray-700 rounded px-2 py-1 text-sm"
          >
            <option value="all">All</option>
            <option value="BUY">Buys only</option>
            <option value="SELL">Sells only</option>
          </select>
        </div>
        <div className="ml-auto text-sm text-gray-500">
          {filteredTrades.length} trades found
        </div>
      </div>

      {/* Trades List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
        </div>
      ) : filteredTrades.length === 0 ? (
        <div className="text-center py-12 bg-[#141414] rounded-lg border border-gray-800">
          <AlertCircle className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">No recent trades found</p>
          <p className="text-sm text-gray-600 mt-1">
            {data?.tracked_wallets === 0
              ? 'Start tracking wallets to see their trades here'
              : 'Try expanding the time window or changing filters'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredTrades.map((trade, index) => {
            const tradeId = getTradeId(trade, index)
            const isExpanded = expandedTrades.has(tradeId)
            const isBuy = trade.side?.toUpperCase() === 'BUY'

            return (
              <div
                key={tradeId}
                className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden hover:border-gray-700 transition-colors"
              >
                {/* Trade Header */}
                <div
                  className="p-4 cursor-pointer"
                  onClick={() => toggleExpanded(tradeId)}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        {/* Side Badge */}
                        <span className={clsx(
                          "flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                          isBuy
                            ? "bg-green-500/10 text-green-400 border border-green-500/20"
                            : "bg-red-500/10 text-red-400 border border-red-500/20"
                        )}>
                          {isBuy ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                          {trade.side?.toUpperCase() || 'TRADE'}
                        </span>

                        {/* Outcome Badge */}
                        {trade.outcome && (
                          <span className={clsx(
                            "px-2 py-0.5 rounded text-xs font-medium",
                            trade.outcome === 'Yes'
                              ? "bg-blue-500/10 text-blue-400"
                              : "bg-purple-500/10 text-purple-400"
                          )}>
                            {trade.outcome}
                          </span>
                        )}

                        {/* Time */}
                        <span className="flex items-center gap-1 text-xs text-gray-500">
                          <Clock className="w-3 h-3" />
                          {formatTimestamp(trade)}
                        </span>
                      </div>

                      {/* Market */}
                      <h3 className="font-medium text-white text-sm">
                        {trade.market || 'Unknown Market'}
                      </h3>

                      {/* Wallet Info */}
                      <div className="flex items-center gap-2 mt-2">
                        <Wallet className="w-3 h-3 text-gray-500" />
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            onNavigateToWallet?.(trade.wallet_address)
                          }}
                          className="text-xs text-orange-400 hover:text-orange-300 hover:underline"
                        >
                          {trade.wallet_label}
                        </button>
                        <span className="text-xs text-gray-600 font-mono">
                          {trade.wallet_address.slice(0, 6)}...{trade.wallet_address.slice(-4)}
                        </span>
                      </div>
                    </div>

                    {/* Price & Size */}
                    <div className="text-right">
                      <div className="text-lg font-bold text-white">
                        ${(trade.price ?? 0).toFixed(2)}
                      </div>
                      <p className="text-xs text-gray-500">
                        {(trade.size ?? 0).toFixed(2)} shares
                      </p>
                      {trade.cost !== undefined && (
                        <p className="text-xs text-gray-500">
                          ${trade.cost.toFixed(2)} total
                        </p>
                      )}
                    </div>

                    {/* Expand Icon */}
                    <div className="self-center">
                      {isExpanded ? (
                        <ChevronUp className="w-5 h-5 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-gray-500" />
                      )}
                    </div>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-gray-800 p-4 bg-[#0f0f0f]">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Price</p>
                        <p className="font-mono text-white">${(trade.price ?? 0).toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Size</p>
                        <p className="font-mono text-white">{(trade.size ?? 0).toFixed(4)} shares</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Total Cost</p>
                        <p className="font-mono text-white">${(trade.cost ?? 0).toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Outcome</p>
                        <p className={clsx(
                          "font-medium",
                          trade.outcome === 'Yes' ? 'text-green-400' : 'text-red-400'
                        )}>
                          {trade.outcome || 'N/A'}
                        </p>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-3 mt-4 pt-4 border-t border-gray-800">
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
                          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300"
                        >
                          <ExternalLink className="w-3 h-3" />
                          View Transaction
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
