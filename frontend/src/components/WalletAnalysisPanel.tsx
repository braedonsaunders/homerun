import { useState, useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
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
  Briefcase,
  DollarSign,
  Activity,
  ArrowRight,
  CheckCircle2,
  Clock,
  Sparkles
} from 'lucide-react'
import clsx from 'clsx'
import {
  getWalletTradesAnalysis,
  getWalletPositionsAnalysis,
  getWalletSummary,
  getWalletWinRate,
  WalletTrade,
  WalletPosition,
  WalletSummary,
  WalletWinRate
} from '../services/api'

interface WalletAnalysisPanelProps {
  initialWallet?: string | null
  onWalletAnalyzed?: () => void
}

// Mini Sparkline Component
function Sparkline({ data, color = '#22c55e', height = 40 }: { data: number[]; color?: string; height?: number }) {
  if (!data || data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const width = 120
  const padding = 2
  const points = data.map((value, index) => {
    const x = padding + (index / (data.length - 1)) * (width - padding * 2)
    const y = height - padding - ((value - min) / range) * (height - padding * 2)
    return `${x},${y}`
  }).join(' ')

  const areaPath = `M ${padding},${height - padding} L ${points} L ${width - padding},${height - padding} Z`

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={`sparkline-gradient-${color.replace('#', '')}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d={areaPath}
        fill={`url(#sparkline-gradient-${color.replace('#', '')})`}
      />
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={width - padding}
        cy={height - padding - ((data[data.length - 1] - min) / range) * (height - padding * 2)}
        r="3"
        fill={color}
      />
    </svg>
  )
}

// Circular Progress Component
function CircularProgress({ percentage, size = 80, strokeWidth = 6, color = '#22c55e' }: {
  percentage: number
  size?: number
  strokeWidth?: number
  color?: string
}) {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (Math.min(100, Math.max(0, percentage)) / 100) * circumference

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold" style={{ color }}>
          {percentage.toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

export default function WalletAnalysisPanel({ initialWallet, onWalletAnalyzed }: WalletAnalysisPanelProps) {
  const [searchAddress, setSearchAddress] = useState('')
  const [activeWallet, setActiveWallet] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'summary' | 'trades' | 'positions'>('summary')

  // Auto-analyze when initialWallet changes
  useEffect(() => {
    if (initialWallet) {
      setSearchAddress(initialWallet)
      setActiveWallet(initialWallet.toLowerCase())
      setActiveTab('summary')
      if (onWalletAnalyzed) {
        onWalletAnalyzed()
      }
    }
  }, [initialWallet, onWalletAnalyzed])

  const summaryQuery = useQuery({
    queryKey: ['wallet-summary', activeWallet],
    queryFn: () => getWalletSummary(activeWallet!),
    enabled: !!activeWallet,
  })

  const winRateQuery = useQuery({
    queryKey: ['wallet-win-rate', activeWallet],
    queryFn: () => getWalletWinRate(activeWallet!),
    enabled: !!activeWallet,
  })

  const tradesQuery = useQuery({
    queryKey: ['wallet-trades', activeWallet],
    queryFn: () => getWalletTradesAnalysis(activeWallet!, 200),
    enabled: !!activeWallet,
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

  const isLoading = summaryQuery.isLoading || winRateQuery.isLoading

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-purple-400" />
            Wallet Analysis
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Deep dive into any trader's performance and strategy
          </p>
        </div>
      </div>

      {/* Search Card */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-500/10 via-blue-500/10 to-cyan-500/10 border border-white/10 p-6">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-purple-500/5 via-transparent to-transparent" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
              <Search className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Analyze a Wallet</h3>
              <p className="text-xs text-gray-400">Enter any Polymarket wallet address</p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <input
                type="text"
                value={searchAddress}
                onChange={(e) => setSearchAddress(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="0x..."
                className="w-full bg-black/30 backdrop-blur border border-white/10 rounded-xl px-4 py-3 font-mono text-sm focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all placeholder:text-gray-600"
              />
            </div>
            <button
              onClick={handleAnalyze}
              disabled={!searchAddress.trim()}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-purple-500/25"
            >
              <Search className="w-4 h-4" />
              Analyze
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {activeWallet && (
        <div className="space-y-6">
          {/* Hero Profile Card */}
          <WalletHeroCard
            address={activeWallet}
            summary={summaryQuery.data}
            winRate={winRateQuery.data}
            trades={tradesQuery.data?.trades || []}
            isLoading={isLoading}
            onRefresh={() => {
              summaryQuery.refetch()
              winRateQuery.refetch()
              tradesQuery.refetch()
            }}
          />

          {/* Tab Navigation */}
          <div className="flex gap-2 p-1 bg-[#141414] rounded-xl border border-gray-800">
            {[
              { id: 'summary' as const, label: 'Overview', icon: BarChart3 },
              { id: 'trades' as const, label: 'Trade History', icon: History },
              { id: 'positions' as const, label: 'Open Positions', icon: Briefcase },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={clsx(
                  "flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg text-sm font-medium transition-all",
                  activeTab === tab.id
                    ? "bg-gradient-to-r from-purple-500/20 to-blue-500/20 text-white border border-purple-500/30"
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="bg-[#141414] border border-gray-800 rounded-2xl overflow-hidden">
            <div className="p-6">
              {activeTab === 'summary' && (
                <SummaryTab
                  data={summaryQuery.data}
                  winRate={winRateQuery.data}
                  trades={tradesQuery.data?.trades || []}
                  isLoading={summaryQuery.isLoading}
                />
              )}
              {activeTab === 'trades' && (
                <TradesTab data={tradesQuery.data} isLoading={tradesQuery.isLoading} />
              )}
              {activeTab === 'positions' && (
                <PositionsTab data={positionsQuery.data} isLoading={positionsQuery.isLoading} />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!activeWallet && (
        <div className="text-center py-16">
          <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center">
            <Wallet className="w-10 h-10 text-purple-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No wallet selected</h3>
          <p className="text-gray-500 max-w-md mx-auto">
            Enter a wallet address above to analyze trading performance, win rates, and strategy patterns.
          </p>
        </div>
      )}
    </div>
  )
}

function WalletHeroCard({
  address,
  summary,
  winRate,
  trades,
  isLoading,
  onRefresh
}: {
  address: string
  summary?: WalletSummary
  winRate?: WalletWinRate
  trades: WalletTrade[]
  isLoading: boolean
  onRefresh: () => void
}) {
  // Generate sparkline data from trades
  const sparklineData = useMemo(() => {
    if (!trades || trades.length === 0) return []

    // Calculate cumulative PnL over time
    const sortedTrades = [...trades].sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )

    let cumulative = 0
    const pnlData: number[] = []

    sortedTrades.slice(-20).forEach(trade => {
      // Approximate PnL: sells add value, buys subtract
      if (trade.side === 'SELL') {
        cumulative += trade.cost
      } else {
        cumulative -= trade.cost
      }
      pnlData.push(cumulative)
    })

    return pnlData.length > 1 ? pnlData : []
  }, [trades])

  const isProfitable = summary ? summary.summary.total_pnl >= 0 : true
  const winRateValue = winRate?.win_rate ?? 0
  const winRateColor = winRateValue >= 70 ? '#22c55e' : winRateValue >= 50 ? '#eab308' : '#ef4444'

  if (isLoading) {
    return (
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#1a1a2e] to-[#16162a] border border-white/10 p-8">
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-purple-400" />
        </div>
      </div>
    )
  }

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#1a1a2e] via-[#1e1e3f] to-[#16162a] border border-white/10">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,_var(--tw-gradient-stops))] from-purple-500/10 via-transparent to-transparent" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-blue-500/10 via-transparent to-transparent" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple-500/50 to-transparent" />

      <div className="relative p-8">
        {/* Header Row */}
        <div className="flex items-start justify-between mb-8">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="relative">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center shadow-xl shadow-purple-500/20">
                <Wallet className="w-8 h-8 text-white" />
              </div>
              {isProfitable && (
                <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-green-500 flex items-center justify-center border-2 border-[#1a1a2e]">
                  <TrendingUp className="w-3 h-3 text-white" />
                </div>
              )}
            </div>

            {/* Identity */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                <h2 className="text-xl font-bold text-white">
                  {`${address.slice(0, 6)}...${address.slice(-4)}`}
                </h2>
                <a
                  href={`https://polymarket.com/profile/${address}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                  title="View on Polymarket"
                >
                  <ExternalLink className="w-4 h-4 text-gray-400" />
                </a>
              </div>
              <p className="text-sm text-gray-400 font-mono">{address}</p>
            </div>
          </div>

          {/* Refresh */}
          <button
            onClick={onRefresh}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
            title="Refresh data"
          >
            <RefreshCw className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Win Rate */}
          <div className="bg-black/20 backdrop-blur rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Win Rate</p>
                <p className="text-2xl font-bold" style={{ color: winRateColor }}>
                  {winRateValue.toFixed(1)}%
                </p>
                {winRate && (
                  <p className="text-xs text-gray-500 mt-1">
                    <span className="text-green-400">{winRate.wins}W</span>
                    {' / '}
                    <span className="text-red-400">{winRate.losses}L</span>
                  </p>
                )}
              </div>
              <CircularProgress
                percentage={winRateValue}
                size={56}
                strokeWidth={4}
                color={winRateColor}
              />
            </div>
          </div>

          {/* Total PnL */}
          <div className="bg-black/20 backdrop-blur rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Total P&L</p>
                <p className={clsx(
                  "text-2xl font-bold",
                  summary && summary.summary.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                )}>
                  {summary ? (summary.summary.total_pnl >= 0 ? '+' : '') + '$' + summary.summary.total_pnl.toFixed(2) : '$0.00'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  ROI: {summary ? (summary.summary.roi_percent >= 0 ? '+' : '') + summary.summary.roi_percent.toFixed(1) + '%' : '0%'}
                </p>
              </div>
              {sparklineData.length > 1 && (
                <Sparkline
                  data={sparklineData}
                  color={summary && summary.summary.total_pnl >= 0 ? '#22c55e' : '#ef4444'}
                  height={40}
                />
              )}
            </div>
          </div>

          {/* Volume */}
          <div className="bg-black/20 backdrop-blur rounded-xl p-4 border border-white/5">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Total Volume</p>
            <p className="text-2xl font-bold text-white">
              ${summary ? (summary.summary.total_invested + summary.summary.total_returned).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {summary?.summary.total_trades || 0} total trades
            </p>
          </div>

          {/* Open Positions */}
          <div className="bg-black/20 backdrop-blur rounded-xl p-4 border border-white/5">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Open Positions</p>
            <p className="text-2xl font-bold text-blue-400">
              {summary?.summary.open_positions || 0}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              ${summary?.summary.position_value.toFixed(2) || '0.00'} value
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function SummaryTab({
  data,
  winRate,
  trades,
  isLoading
}: {
  data?: WalletSummary
  winRate?: WalletWinRate
  trades: WalletTrade[]
  isLoading: boolean
}) {
  // Generate trade activity sparkline
  const activityData = useMemo(() => {
    if (!trades || trades.length === 0) return []

    // Group trades by day and count
    const sortedTrades = [...trades].sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )

    const dailyCounts: { [key: string]: number } = {}
    sortedTrades.forEach(trade => {
      const day = new Date(trade.timestamp).toISOString().split('T')[0]
      dailyCounts[day] = (dailyCounts[day] || 0) + 1
    })

    return Object.values(dailyCounts).slice(-14) // Last 14 days
  }, [trades])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-purple-400 mx-auto mb-4" />
          <p className="text-gray-500">Analyzing wallet...</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="text-center py-16">
        <Activity className="w-12 h-12 text-gray-600 mx-auto mb-4" />
        <p className="text-gray-500">No data available for this wallet</p>
      </div>
    )
  }

  const { summary } = data
  const isProfitable = summary.total_pnl >= 0

  return (
    <div className="space-y-6">
      {/* Performance Grid */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Performance Breakdown</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Realized PnL */}
          <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/5 border border-green-500/20 p-5">
            <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                <p className="text-sm text-gray-400">Realized P&L</p>
              </div>
              <p className={clsx(
                "text-2xl font-bold",
                summary.realized_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                {summary.realized_pnl >= 0 ? '+' : ''}${summary.realized_pnl.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">From closed positions</p>
            </div>
          </div>

          {/* Unrealized PnL */}
          <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border border-blue-500/20 p-5">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-blue-400" />
                <p className="text-sm text-gray-400">Unrealized P&L</p>
              </div>
              <p className={clsx(
                "text-2xl font-bold",
                summary.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                {summary.unrealized_pnl >= 0 ? '+' : ''}${summary.unrealized_pnl.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">From open positions</p>
            </div>
          </div>

          {/* Total PnL */}
          <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/5 border border-purple-500/20 p-5">
            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-2">
                <DollarSign className="w-4 h-4 text-purple-400" />
                <p className="text-sm text-gray-400">Total P&L</p>
              </div>
              <p className={clsx(
                "text-2xl font-bold",
                isProfitable ? "text-green-400" : "text-red-400"
              )}>
                {isProfitable ? '+' : ''}${summary.total_pnl.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {summary.roi_percent >= 0 ? '+' : ''}{summary.roi_percent.toFixed(1)}% ROI
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Investment Flow & Trading Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Investment Flow */}
        <div className="rounded-xl bg-[#1a1a1a] border border-gray-800 p-5">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
              <ArrowRight className="w-4 h-4 text-blue-400" />
            </div>
            <h4 className="font-semibold text-white">Investment Flow</h4>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-400" />
                <span className="text-sm text-gray-400">Total Invested</span>
              </div>
              <span className="font-mono font-medium text-white">${summary.total_invested.toFixed(2)}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-400" />
                <span className="text-sm text-gray-400">Total Returned</span>
              </div>
              <span className="font-mono font-medium text-white">${summary.total_returned.toFixed(2)}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-400" />
                <span className="text-sm text-gray-400">Position Value</span>
              </div>
              <span className="font-mono font-medium text-white">${summary.position_value.toFixed(2)}</span>
            </div>
            <div className="pt-3 mt-3 border-t border-gray-800">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-300">Net Flow</span>
                <span className={clsx(
                  "font-mono font-bold text-lg",
                  summary.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                )}>
                  {summary.total_pnl >= 0 ? '+' : ''}${summary.total_pnl.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Trading Activity */}
        <div className="rounded-xl bg-[#1a1a1a] border border-gray-800 p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
                <Activity className="w-4 h-4 text-purple-400" />
              </div>
              <h4 className="font-semibold text-white">Trading Activity</h4>
            </div>
            {activityData.length > 1 && (
              <Sparkline data={activityData} color="#a855f7" height={30} />
            )}
          </div>

          {/* Trade Counts */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center p-3 rounded-lg bg-green-500/10 border border-green-500/20">
              <p className="text-2xl font-bold text-green-400">{summary.buys}</p>
              <p className="text-xs text-gray-500">Buys</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <p className="text-2xl font-bold text-red-400">{summary.sells}</p>
              <p className="text-xs text-gray-500">Sells</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-gray-500/10 border border-gray-500/20">
              <p className="text-2xl font-bold text-gray-300">{summary.total_trades}</p>
              <p className="text-xs text-gray-500">Total</p>
            </div>
          </div>

          {/* Win Rate Bar */}
          {winRate && (
            <div className="pt-3 border-t border-gray-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Win Rate</span>
                <span className={clsx(
                  "font-medium",
                  winRate.win_rate >= 70 ? "text-green-400" :
                  winRate.win_rate >= 50 ? "text-yellow-400" : "text-red-400"
                )}>
                  {winRate.win_rate.toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    "h-full rounded-full transition-all duration-500",
                    winRate.win_rate >= 70 ? "bg-gradient-to-r from-green-500 to-emerald-400" :
                    winRate.win_rate >= 50 ? "bg-gradient-to-r from-yellow-500 to-amber-400" :
                    "bg-gradient-to-r from-red-500 to-rose-400"
                  )}
                  style={{ width: `${Math.min(100, winRate.win_rate)}%` }}
                />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-xs text-green-400">{winRate.wins} wins</span>
                <span className="text-xs text-red-400">{winRate.losses} losses</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function TradesTab({ data, isLoading }: { data?: { wallet: string; total: number; trades: WalletTrade[] }; isLoading: boolean }) {
  const [expandedTrades, setExpandedTrades] = useState<Set<string>>(new Set())

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-purple-400 mx-auto mb-4" />
          <p className="text-gray-500">Loading trade history...</p>
        </div>
      </div>
    )
  }

  if (!data || data.trades.length === 0) {
    return (
      <div className="text-center py-16">
        <History className="w-12 h-12 text-gray-600 mx-auto mb-4" />
        <p className="text-gray-500">No trades found for this wallet</p>
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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-500">
          Showing {data.trades.length} of {data.total} trades
        </p>
      </div>

      {/* Trade List */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
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
    <div className={clsx(
      "rounded-xl overflow-hidden transition-all",
      isExpanded ? "bg-[#1a1a1a]" : "bg-[#1a1a1a]/50 hover:bg-[#1a1a1a]"
    )}>
      <div
        className="flex items-center justify-between p-4 cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex items-center gap-4">
          <div className={clsx(
            "w-10 h-10 rounded-xl flex items-center justify-center",
            isBuy ? "bg-green-500/20" : "bg-red-500/20"
          )}>
            {isBuy ? (
              <ArrowUpRight className="w-5 h-5 text-green-400" />
            ) : (
              <ArrowDownRight className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={clsx(
                "text-xs font-semibold px-2 py-0.5 rounded-full",
                isBuy ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
              )}>
                {trade.side}
              </span>
              <span className="text-sm font-medium text-white">{trade.outcome || 'Unknown'}</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">{timestamp}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="font-mono font-semibold text-white">${trade.cost.toFixed(2)}</p>
            <p className="text-xs text-gray-500">
              {trade.size.toFixed(2)} @ ${trade.price.toFixed(4)}
            </p>
          </div>
          <div className={clsx(
            "p-2 rounded-lg transition-colors",
            isExpanded ? "bg-purple-500/20" : "bg-white/5"
          )}>
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-purple-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            )}
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-800/50">
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Market</p>
              <p className="font-mono text-xs text-gray-300 truncate" title={trade.market}>
                {trade.market.slice(0, 30)}...
              </p>
            </div>
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Size</p>
              <p className="font-mono text-white">{trade.size.toFixed(4)}</p>
            </div>
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Price</p>
              <p className="font-mono text-white">${trade.price.toFixed(4)}</p>
            </div>
            {trade.transaction_hash && (
              <div className="bg-black/20 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Transaction</p>
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
      <div className="flex items-center justify-center py-16">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-purple-400 mx-auto mb-4" />
          <p className="text-gray-500">Loading positions...</p>
        </div>
      </div>
    )
  }

  if (!data || data.positions.length === 0) {
    return (
      <div className="text-center py-16">
        <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-4" />
        <p className="text-gray-500">No open positions</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border border-blue-500/20 p-5">
          <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Total Position Value</p>
            <p className="text-2xl font-bold text-white">${data.total_value.toFixed(2)}</p>
            <p className="text-xs text-gray-500 mt-1">{data.total_positions} open positions</p>
          </div>
        </div>
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/5 border border-purple-500/20 p-5">
          <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Unrealized P&L</p>
            <p className={clsx(
              "text-2xl font-bold",
              data.total_unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {data.total_unrealized_pnl >= 0 ? '+' : ''}${data.total_unrealized_pnl.toFixed(2)}
            </p>
          </div>
        </div>
      </div>

      {/* Positions List */}
      <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
        {data.positions.map((position, idx) => (
          <PositionRow key={idx} position={position} />
        ))}
      </div>
    </div>
  )
}

function PositionRow({ position }: { position: WalletPosition }) {
  const isProfitable = position.unrealized_pnl >= 0
  const roiColor = position.roi_percent >= 20 ? 'text-green-400' :
                   position.roi_percent >= 0 ? 'text-emerald-400' :
                   position.roi_percent >= -20 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="rounded-xl bg-[#1a1a1a] border border-gray-800 p-5 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <p className="font-medium text-white truncate">{position.outcome || 'Unknown'}</p>
          <p className="text-xs text-gray-500 font-mono truncate mt-1" title={position.market}>
            {position.market.slice(0, 40)}...
          </p>
        </div>
        <div className={clsx(
          "flex items-center gap-1.5 px-3 py-1.5 rounded-full ml-4",
          isProfitable ? "bg-green-500/20" : "bg-red-500/20"
        )}>
          {isProfitable ? (
            <TrendingUp className="w-4 h-4 text-green-400" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-400" />
          )}
          <span className={clsx("font-semibold", roiColor)}>
            {position.roi_percent >= 0 ? '+' : ''}{position.roi_percent.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-black/20 rounded-lg p-3">
          <p className="text-xs text-gray-500 mb-1">Size</p>
          <p className="font-mono font-medium text-white">{position.size.toFixed(2)}</p>
        </div>
        <div className="bg-black/20 rounded-lg p-3">
          <p className="text-xs text-gray-500 mb-1">Avg Price</p>
          <p className="font-mono font-medium text-white">${position.avg_price.toFixed(4)}</p>
        </div>
        <div className="bg-black/20 rounded-lg p-3">
          <p className="text-xs text-gray-500 mb-1">Current Price</p>
          <p className="font-mono font-medium text-white">${position.current_price.toFixed(4)}</p>
        </div>
        <div className="bg-black/20 rounded-lg p-3">
          <p className="text-xs text-gray-500 mb-1">Unrealized P&L</p>
          <p className={clsx(
            "font-mono font-medium",
            isProfitable ? "text-green-400" : "text-red-400"
          )}>
            {isProfitable ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  )
}
