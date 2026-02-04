import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Search,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  ArrowUpRight,
  ArrowDownRight,
  Wallet,
  BarChart3,
  History,
  Briefcase
} from 'lucide-react'
import clsx from 'clsx'
import {
  getWalletTradesAnalysis,
  getWalletPositionsAnalysis,
  getWalletSummary,
  WalletTrade,
  WalletPosition,
  WalletSummary
} from '../services/api'

export default function WalletAnalysisPanel() {
  const [searchAddress, setSearchAddress] = useState('')
  const [activeWallet, setActiveWallet] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'summary' | 'trades' | 'positions'>('summary')

  const summaryQuery = useQuery({
    queryKey: ['wallet-summary', activeWallet],
    queryFn: () => getWalletSummary(activeWallet!),
    enabled: !!activeWallet,
  })

  const tradesQuery = useQuery({
    queryKey: ['wallet-trades', activeWallet],
    queryFn: () => getWalletTradesAnalysis(activeWallet!, 200),
    enabled: !!activeWallet && activeTab === 'trades',
  })

  const positionsQuery = useQuery({
    queryKey: ['wallet-positions', activeWallet],
    queryFn: () => getWalletPositionsAnalysis(activeWallet!),
    enabled: !!activeWallet && activeTab === 'positions',
  })

  const handleAnalyze = () => {
    if (searchAddress.trim()) {
      setActiveWallet(searchAddress.trim().toLowerCase())
      setActiveTab('summary')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAnalyze()
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold">Wallet Analysis</h2>
        <p className="text-sm text-gray-500">
          Analyze any wallet's trading activity, positions, and performance
        </p>
      </div>

      {/* Search */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <h3 className="font-medium mb-3 flex items-center gap-2">
          <Wallet className="w-4 h-4" />
          Analyze a Wallet
        </h3>
        <div className="flex gap-3">
          <input
            type="text"
            value={searchAddress}
            onChange={(e) => setSearchAddress(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter wallet address (0x...)"
            className="flex-1 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-2 font-mono text-sm"
          />
          <button
            onClick={handleAnalyze}
            disabled={!searchAddress.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg font-medium disabled:opacity-50"
          >
            <Search className="w-4 h-4" />
            Analyze
          </button>
        </div>
      </div>

      {/* Results */}
      {activeWallet && (
        <div className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden">
          {/* Wallet Header */}
          <div className="p-4 border-b border-gray-800">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-mono text-sm text-gray-400">Wallet</p>
                <p className="font-mono font-medium">{activeWallet}</p>
              </div>
              <button
                onClick={() => {
                  summaryQuery.refetch()
                  if (activeTab === 'trades') tradesQuery.refetch()
                  if (activeTab === 'positions') positionsQuery.refetch()
                }}
                className="p-2 hover:bg-gray-700 rounded-lg"
              >
                <RefreshCw className={clsx(
                  "w-4 h-4",
                  (summaryQuery.isFetching || tradesQuery.isFetching || positionsQuery.isFetching) && "animate-spin"
                )} />
              </button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex border-b border-gray-800">
            <button
              onClick={() => setActiveTab('summary')}
              className={clsx(
                "flex-1 py-3 px-4 text-sm font-medium transition-colors flex items-center justify-center gap-2",
                activeTab === 'summary'
                  ? "bg-blue-500/10 text-blue-400 border-b-2 border-blue-500"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              )}
            >
              <BarChart3 className="w-4 h-4" />
              Summary
            </button>
            <button
              onClick={() => setActiveTab('trades')}
              className={clsx(
                "flex-1 py-3 px-4 text-sm font-medium transition-colors flex items-center justify-center gap-2",
                activeTab === 'trades'
                  ? "bg-blue-500/10 text-blue-400 border-b-2 border-blue-500"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              )}
            >
              <History className="w-4 h-4" />
              Trades
            </button>
            <button
              onClick={() => setActiveTab('positions')}
              className={clsx(
                "flex-1 py-3 px-4 text-sm font-medium transition-colors flex items-center justify-center gap-2",
                activeTab === 'positions'
                  ? "bg-blue-500/10 text-blue-400 border-b-2 border-blue-500"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              )}
            >
              <Briefcase className="w-4 h-4" />
              Positions
            </button>
          </div>

          {/* Tab Content */}
          <div className="p-4">
            {activeTab === 'summary' && (
              <SummaryTab data={summaryQuery.data} isLoading={summaryQuery.isLoading} />
            )}
            {activeTab === 'trades' && (
              <TradesTab data={tradesQuery.data} isLoading={tradesQuery.isLoading} />
            )}
            {activeTab === 'positions' && (
              <PositionsTab data={positionsQuery.data} isLoading={positionsQuery.isLoading} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function SummaryTab({ data, isLoading }: { data?: WalletSummary; isLoading: boolean }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-6 h-6 animate-spin text-gray-500" />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="text-center py-12 text-gray-500">
        No data available
      </div>
    )
  }

  const { summary } = data
  const isProfitable = summary.total_pnl > 0

  return (
    <div className="space-y-6">
      {/* Main Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total PnL"
          value={`$${summary.total_pnl.toFixed(2)}`}
          trend={isProfitable ? 'up' : 'down'}
          color={isProfitable ? 'green' : 'red'}
        />
        <StatCard
          label="ROI"
          value={`${summary.roi_percent.toFixed(2)}%`}
          trend={summary.roi_percent > 0 ? 'up' : 'down'}
          color={summary.roi_percent > 0 ? 'green' : 'red'}
        />
        <StatCard
          label="Total Trades"
          value={summary.total_trades.toString()}
        />
        <StatCard
          label="Open Positions"
          value={summary.open_positions.toString()}
        />
      </div>

      {/* Detailed Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Investment Flow */}
        <div className="bg-[#1a1a1a] rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-400 mb-3">Investment Flow</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-500">Total Invested</span>
              <span className="font-mono">${summary.total_invested.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Total Returned</span>
              <span className="font-mono">${summary.total_returned.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Position Value</span>
              <span className="font-mono">${summary.position_value.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* PnL Breakdown */}
        <div className="bg-[#1a1a1a] rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-400 mb-3">PnL Breakdown</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-500">Realized PnL</span>
              <span className={clsx(
                "font-mono",
                summary.realized_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                ${summary.realized_pnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Unrealized PnL</span>
              <span className={clsx(
                "font-mono",
                summary.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                ${summary.unrealized_pnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between border-t border-gray-700 pt-2 mt-2">
              <span className="text-gray-300 font-medium">Total PnL</span>
              <span className={clsx(
                "font-mono font-medium",
                summary.total_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                ${summary.total_pnl.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Trade Breakdown */}
      <div className="bg-[#1a1a1a] rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-400 mb-3">Trade Activity</h4>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-green-400">{summary.buys}</p>
            <p className="text-xs text-gray-500">Buys</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-red-400">{summary.sells}</p>
            <p className="text-xs text-gray-500">Sells</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-gray-300">{summary.total_trades}</p>
            <p className="text-xs text-gray-500">Total</p>
          </div>
        </div>
      </div>
    </div>
  )
}

function TradesTab({ data, isLoading }: { data?: { wallet: string; total: number; trades: WalletTrade[] }; isLoading: boolean }) {
  const [expandedTrades, setExpandedTrades] = useState<Set<string>>(new Set())

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-6 h-6 animate-spin text-gray-500" />
      </div>
    )
  }

  if (!data || data.trades.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No trades found
      </div>
    )
  }

  const toggleTrade = (id: string) => {
    const newExpanded = new Set(expandedTrades)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedTrades(newExpanded)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-gray-500">
          Showing {data.trades.length} of {data.total} trades
        </p>
      </div>

      {/* Trade List */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {data.trades.map((trade) => (
          <TradeRow
            key={trade.id}
            trade={trade}
            isExpanded={expandedTrades.has(trade.id)}
            onToggle={() => toggleTrade(trade.id)}
          />
        ))}
      </div>
    </div>
  )
}

function TradeRow({ trade, isExpanded, onToggle }: { trade: WalletTrade; isExpanded: boolean; onToggle: () => void }) {
  const isBuy = trade.side === 'BUY'
  const timestamp = trade.timestamp ? new Date(trade.timestamp).toLocaleString() : 'Unknown'

  return (
    <div className="bg-[#1a1a1a] rounded-lg overflow-hidden">
      <div
        className="flex items-center justify-between p-3 cursor-pointer hover:bg-[#222]"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          <div className={clsx(
            "p-2 rounded-lg",
            isBuy ? "bg-green-500/20" : "bg-red-500/20"
          )}>
            {isBuy ? (
              <ArrowUpRight className="w-4 h-4 text-green-400" />
            ) : (
              <ArrowDownRight className="w-4 h-4 text-red-400" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={clsx(
                "text-xs font-medium px-2 py-0.5 rounded",
                isBuy ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
              )}>
                {trade.side}
              </span>
              <span className="text-sm text-gray-400">{trade.outcome || 'Unknown'}</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">{timestamp}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="font-mono font-medium">${trade.cost.toFixed(2)}</p>
            <p className="text-xs text-gray-500">
              {trade.size.toFixed(2)} @ ${trade.price.toFixed(4)}
            </p>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="px-3 pb-3 pt-0 border-t border-gray-800">
          <div className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Market</span>
              <span className="font-mono text-xs">{trade.market.slice(0, 20)}...</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Size</span>
              <span className="font-mono">{trade.size.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Price</span>
              <span className="font-mono">${trade.price.toFixed(4)}</span>
            </div>
            {trade.transaction_hash && (
              <div className="flex justify-between items-center">
                <span className="text-gray-500">Transaction</span>
                <a
                  href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-blue-400 hover:text-blue-300"
                >
                  <span className="font-mono text-xs">{trade.transaction_hash.slice(0, 10)}...</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function PositionsTab({ data, isLoading }: { data?: { wallet: string; total_positions: number; total_value: number; total_unrealized_pnl: number; positions: WalletPosition[] }; isLoading: boolean }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-6 h-6 animate-spin text-gray-500" />
      </div>
    )
  }

  if (!data || data.positions.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No open positions
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-[#1a1a1a] rounded-lg p-4">
          <p className="text-xs text-gray-500">Total Value</p>
          <p className="text-xl font-mono font-bold">${data.total_value.toFixed(2)}</p>
        </div>
        <div className="bg-[#1a1a1a] rounded-lg p-4">
          <p className="text-xs text-gray-500">Unrealized PnL</p>
          <p className={clsx(
            "text-xl font-mono font-bold",
            data.total_unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
          )}>
            ${data.total_unrealized_pnl.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Positions List */}
      <div className="space-y-2 max-h-[500px] overflow-y-auto">
        {data.positions.map((position, idx) => (
          <PositionRow key={idx} position={position} />
        ))}
      </div>
    </div>
  )
}

function PositionRow({ position }: { position: WalletPosition }) {
  const isProfitable = position.unrealized_pnl >= 0

  return (
    <div className="bg-[#1a1a1a] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-sm font-medium">{position.outcome || 'Unknown'}</p>
          <p className="text-xs text-gray-500 font-mono">{position.market.slice(0, 30)}...</p>
        </div>
        <div className={clsx(
          "flex items-center gap-1 px-2 py-1 rounded",
          isProfitable ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
        )}>
          {isProfitable ? (
            <TrendingUp className="w-3 h-3" />
          ) : (
            <TrendingDown className="w-3 h-3" />
          )}
          <span className="text-sm font-medium">{position.roi_percent.toFixed(1)}%</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <p className="text-xs text-gray-500">Size</p>
          <p className="font-mono">{position.size.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Avg Price</p>
          <p className="font-mono">${position.avg_price.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Current Price</p>
          <p className="font-mono">${position.current_price.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Unrealized PnL</p>
          <p className={clsx(
            "font-mono",
            isProfitable ? "text-green-400" : "text-red-400"
          )}>
            ${position.unrealized_pnl.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, trend, color }: { label: string; value: string; trend?: 'up' | 'down'; color?: 'green' | 'red' }) {
  return (
    <div className="bg-[#1a1a1a] rounded-lg p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <div className="flex items-center gap-2">
        <p className={clsx(
          "text-xl font-mono font-bold",
          color === 'green' && "text-green-400",
          color === 'red' && "text-red-400"
        )}>
          {value}
        </p>
        {trend && (
          trend === 'up' ? (
            <TrendingUp className="w-4 h-4 text-green-400" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-400" />
          )
        )}
      </div>
    </div>
  )
}
