import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart3,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Target,
  Award,
  Calendar,
  Activity,
  PieChart,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react'
import clsx from 'clsx'
import {
  getSimulationAccounts,
  getAccountTrades,
  getAutoTraderStats,
  getAutoTraderTrades,
  SimulationAccount,
  SimulationTrade
} from '../services/api'

type ViewMode = 'simulation' | 'live' | 'all'
type TimeRange = '7d' | '30d' | '90d' | 'all'

export default function PerformancePanel() {
  const [viewMode, setViewMode] = useState<ViewMode>('all')
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('30d')

  // Fetch simulation accounts
  const { data: accounts = [], isLoading: accountsLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  // Fetch auto trader stats
  const { isLoading: autoStatsLoading } = useQuery({
    queryKey: ['auto-trader-stats'],
    queryFn: getAutoTraderStats,
    enabled: viewMode === 'live' || viewMode === 'all',
  })

  // Fetch simulation trades for all accounts or selected account
  const { data: simulationTrades = [], isLoading: simTradesLoading, refetch: refetchSimTrades } = useQuery({
    queryKey: ['all-simulation-trades', selectedAccount],
    queryFn: async () => {
      if (selectedAccount) {
        return getAccountTrades(selectedAccount, 200)
      }
      const allTrades: (SimulationTrade & { accountName: string })[] = []
      for (const account of accounts) {
        const trades = await getAccountTrades(account.id, 200)
        trades.forEach((trade: SimulationTrade) => {
          allTrades.push({ ...trade, accountName: account.name })
        })
      }
      return allTrades.sort((a, b) =>
        new Date(b.executed_at).getTime() - new Date(a.executed_at).getTime()
      )
    },
    enabled: accounts.length > 0 && (viewMode === 'simulation' || viewMode === 'all'),
  })

  // Fetch auto trader trades
  const { data: autoTrades = [], isLoading: autoTradesLoading, refetch: refetchAutoTrades } = useQuery({
    queryKey: ['auto-trader-trades'],
    queryFn: () => getAutoTraderTrades(200),
    enabled: viewMode === 'live' || viewMode === 'all',
  })

  const isLoading = accountsLoading || simTradesLoading || autoStatsLoading || autoTradesLoading

  // Filter trades by time range
  const filterByTimeRange = <T extends { executed_at: string }>(trades: T[]): T[] => {
    if (timeRange === 'all') return trades
    const now = new Date()
    const daysMap = { '7d': 7, '30d': 30, '90d': 90, 'all': Infinity }
    const cutoff = new Date(now.getTime() - daysMap[timeRange] * 24 * 60 * 60 * 1000)
    return trades.filter(t => new Date(t.executed_at) >= cutoff)
  }

  const filteredSimTrades = useMemo(() =>
    filterByTimeRange(simulationTrades), [simulationTrades, timeRange])
  const filteredAutoTrades = useMemo(() =>
    filterByTimeRange(autoTrades), [autoTrades, timeRange])

  // Calculate simulation performance metrics
  const simMetrics = useMemo(() => {
    const trades = filteredSimTrades
    const resolved = trades.filter(t => t.status === 'resolved_win' || t.status === 'resolved_loss')
    const wins = resolved.filter(t => t.status === 'resolved_win')
    const losses = resolved.filter(t => t.status === 'resolved_loss')

    const totalPnl = trades.reduce((sum, t) => sum + (t.actual_pnl || 0), 0)
    const totalCost = trades.reduce((sum, t) => sum + t.total_cost, 0)
    const winRate = resolved.length > 0 ? (wins.length / resolved.length) * 100 : 0
    const avgWin = wins.length > 0
      ? wins.reduce((sum, t) => sum + (t.actual_pnl || 0), 0) / wins.length
      : 0
    const avgLoss = losses.length > 0
      ? Math.abs(losses.reduce((sum, t) => sum + (t.actual_pnl || 0), 0) / losses.length)
      : 0
    const roi = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0

    // Performance by strategy
    const byStrategy: Record<string, { trades: number; pnl: number; wins: number; losses: number }> = {}
    trades.forEach(t => {
      if (!byStrategy[t.strategy_type]) {
        byStrategy[t.strategy_type] = { trades: 0, pnl: 0, wins: 0, losses: 0 }
      }
      byStrategy[t.strategy_type].trades++
      byStrategy[t.strategy_type].pnl += t.actual_pnl || 0
      if (t.status === 'resolved_win') byStrategy[t.strategy_type].wins++
      if (t.status === 'resolved_loss') byStrategy[t.strategy_type].losses++
    })

    // Daily P&L data for chart
    const dailyPnl: Record<string, number> = {}
    trades.forEach(t => {
      const date = new Date(t.executed_at).toISOString().split('T')[0]
      dailyPnl[date] = (dailyPnl[date] || 0) + (t.actual_pnl || 0)
    })

    return {
      totalTrades: trades.length,
      resolvedTrades: resolved.length,
      openTrades: trades.filter(t => t.status === 'open' || t.status === 'pending').length,
      wins: wins.length,
      losses: losses.length,
      totalPnl,
      totalCost,
      winRate,
      avgWin,
      avgLoss,
      roi,
      byStrategy,
      dailyPnl
    }
  }, [filteredSimTrades])

  // Calculate auto trader performance metrics
  const autoMetrics = useMemo(() => {
    const trades = filteredAutoTrades
    const resolved = trades.filter(t => t.status === 'resolved' || t.status === 'win' || t.status === 'loss')

    const totalPnl = trades.reduce((sum, t) => sum + (t.actual_profit || 0), 0)
    const totalCost = trades.reduce((sum, t) => sum + t.total_cost, 0)
    const wins = trades.filter(t => (t.actual_profit || 0) > 0)
    const losses = trades.filter(t => (t.actual_profit || 0) < 0)
    const winRate = resolved.length > 0 ? (wins.length / resolved.length) * 100 : 0
    const roi = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0

    // Performance by strategy
    const byStrategy: Record<string, { trades: number; pnl: number; wins: number; losses: number }> = {}
    trades.forEach(t => {
      if (!byStrategy[t.strategy]) {
        byStrategy[t.strategy] = { trades: 0, pnl: 0, wins: 0, losses: 0 }
      }
      byStrategy[t.strategy].trades++
      byStrategy[t.strategy].pnl += t.actual_profit || 0
      if ((t.actual_profit || 0) > 0) byStrategy[t.strategy].wins++
      if ((t.actual_profit || 0) < 0) byStrategy[t.strategy].losses++
    })

    return {
      totalTrades: trades.length,
      wins: wins.length,
      losses: losses.length,
      totalPnl,
      totalCost,
      winRate,
      roi,
      byStrategy
    }
  }, [filteredAutoTrades])

  const handleRefresh = () => {
    if (viewMode === 'simulation' || viewMode === 'all') refetchSimTrades()
    if (viewMode === 'live' || viewMode === 'all') refetchAutoTrades()
  }

  // Cumulative P&L calculation
  const cumulativePnlData = useMemo(() => {
    let data: { date: string; simPnl: number; autoPnl: number; cumSimPnl: number; cumAutoPnl: number }[] = []
    const dates = new Set<string>()

    filteredSimTrades.forEach(t => dates.add(new Date(t.executed_at).toISOString().split('T')[0]))
    filteredAutoTrades.forEach(t => dates.add(new Date(t.executed_at).toISOString().split('T')[0]))

    const sortedDates = Array.from(dates).sort()
    let cumSimPnl = 0
    let cumAutoPnl = 0

    sortedDates.forEach(date => {
      const simPnl = filteredSimTrades
        .filter(t => new Date(t.executed_at).toISOString().split('T')[0] === date)
        .reduce((sum, t) => sum + (t.actual_pnl || 0), 0)
      const autoPnl = filteredAutoTrades
        .filter(t => new Date(t.executed_at).toISOString().split('T')[0] === date)
        .reduce((sum, t) => sum + (t.actual_profit || 0), 0)

      cumSimPnl += simPnl
      cumAutoPnl += autoPnl

      data.push({ date, simPnl, autoPnl, cumSimPnl, cumAutoPnl })
    })

    return data
  }, [filteredSimTrades, filteredAutoTrades])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-purple-500" />
            Performance Analytics
          </h2>
          <p className="text-sm text-gray-500">Track your trading performance over time</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-[#1a1a1a] hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
        >
          <RefreshCw className={clsx("w-4 h-4", isLoading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex bg-[#141414] rounded-lg p-1 border border-gray-800">
          <button
            onClick={() => setViewMode('all')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              viewMode === 'all' ? "bg-purple-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            All Trading
          </button>
          <button
            onClick={() => setViewMode('simulation')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              viewMode === 'simulation' ? "bg-purple-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            Paper Trading
          </button>
          <button
            onClick={() => setViewMode('live')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              viewMode === 'live' ? "bg-purple-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            Live Trading
          </button>
        </div>

        <div className="flex bg-[#141414] rounded-lg p-1 border border-gray-800">
          {(['7d', '30d', '90d', 'all'] as TimeRange[]).map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={clsx(
                "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                timeRange === range ? "bg-gray-700 text-white" : "text-gray-400 hover:text-white"
              )}
            >
              {range === 'all' ? 'All Time' : range.toUpperCase()}
            </button>
          ))}
        </div>

        {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
          <select
            value={selectedAccount || ''}
            onChange={(e) => setSelectedAccount(e.target.value || null)}
            className="bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All Simulation Accounts</option>
            {accounts.map((account: SimulationAccount) => (
              <option key={account.id} value={account.id}>{account.name}</option>
            ))}
          </select>
        )}
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {(viewMode === 'simulation' || viewMode === 'all') && (
              <>
                <MetricCard
                  icon={<Activity className="w-5 h-5 text-blue-500" />}
                  label="Paper Trades"
                  value={simMetrics.totalTrades.toString()}
                  subtitle={`${simMetrics.openTrades} open`}
                />
                <MetricCard
                  icon={simMetrics.totalPnl >= 0
                    ? <TrendingUp className="w-5 h-5 text-green-500" />
                    : <TrendingDown className="w-5 h-5 text-red-500" />
                  }
                  label="Paper P&L"
                  value={`${simMetrics.totalPnl >= 0 ? '+' : ''}$${simMetrics.totalPnl.toFixed(2)}`}
                  valueColor={simMetrics.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
                />
                <MetricCard
                  icon={<Award className="w-5 h-5 text-yellow-500" />}
                  label="Paper Win Rate"
                  value={`${simMetrics.winRate.toFixed(1)}%`}
                  subtitle={`${simMetrics.wins}W / ${simMetrics.losses}L`}
                />
              </>
            )}
            {(viewMode === 'live' || viewMode === 'all') && (
              <>
                <MetricCard
                  icon={<Target className="w-5 h-5 text-purple-500" />}
                  label="Live Trades"
                  value={autoMetrics.totalTrades.toString()}
                />
                <MetricCard
                  icon={autoMetrics.totalPnl >= 0
                    ? <TrendingUp className="w-5 h-5 text-green-500" />
                    : <TrendingDown className="w-5 h-5 text-red-500" />
                  }
                  label="Live P&L"
                  value={`${autoMetrics.totalPnl >= 0 ? '+' : ''}$${autoMetrics.totalPnl.toFixed(2)}`}
                  valueColor={autoMetrics.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
                />
                <MetricCard
                  icon={<Award className="w-5 h-5 text-yellow-500" />}
                  label="Live Win Rate"
                  value={`${autoMetrics.winRate.toFixed(1)}%`}
                  subtitle={`${autoMetrics.wins}W / ${autoMetrics.losses}L`}
                />
              </>
            )}
          </div>

          {/* P&L Chart */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-purple-500" />
              Cumulative P&L Over Time
            </h3>
            {cumulativePnlData.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No trade data available for the selected time range
              </div>
            ) : (
              <div className="h-64">
                <SimplePnlChart data={cumulativePnlData} viewMode={viewMode} />
              </div>
            )}
          </div>

          {/* Strategy Performance */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {(viewMode === 'simulation' || viewMode === 'all') && Object.keys(simMetrics.byStrategy).length > 0 && (
              <div className="bg-[#141414] border border-gray-800 rounded-lg p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-blue-500" />
                  Paper Trading by Strategy
                </h3>
                <div className="space-y-3">
                  {Object.entries(simMetrics.byStrategy).map(([strategy, stats]) => (
                    <StrategyRow
                      key={strategy}
                      strategy={strategy}
                      trades={stats.trades}
                      pnl={stats.pnl}
                      wins={stats.wins}
                      losses={stats.losses}
                    />
                  ))}
                </div>
              </div>
            )}

            {(viewMode === 'live' || viewMode === 'all') && Object.keys(autoMetrics.byStrategy).length > 0 && (
              <div className="bg-[#141414] border border-gray-800 rounded-lg p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-purple-500" />
                  Live Trading by Strategy
                </h3>
                <div className="space-y-3">
                  {Object.entries(autoMetrics.byStrategy).map(([strategy, stats]) => (
                    <StrategyRow
                      key={strategy}
                      strategy={strategy}
                      trades={stats.trades}
                      pnl={stats.pnl}
                      wins={stats.wins}
                      losses={stats.losses}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Recent Trades */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Calendar className="w-5 h-5 text-gray-400" />
              Recent Trades
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {(viewMode === 'simulation' || viewMode === 'all') && filteredSimTrades.slice(0, 20).map((trade) => (
                <TradeRow
                  key={trade.id}
                  type="paper"
                  strategy={trade.strategy_type}
                  cost={trade.total_cost}
                  pnl={trade.actual_pnl}
                  status={trade.status}
                  date={trade.executed_at}
                  accountName={(trade as SimulationTrade & { accountName?: string }).accountName}
                />
              ))}
              {(viewMode === 'live' || viewMode === 'all') && filteredAutoTrades.slice(0, 20).map((trade) => (
                <TradeRow
                  key={trade.id}
                  type="live"
                  strategy={trade.strategy}
                  cost={trade.total_cost}
                  pnl={trade.actual_profit}
                  status={trade.status}
                  date={trade.executed_at}
                />
              ))}
              {filteredSimTrades.length === 0 && filteredAutoTrades.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No trades found for the selected time range
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function MetricCard({
  icon,
  label,
  value,
  subtitle,
  valueColor = 'text-white'
}: {
  icon: React.ReactNode
  label: string
  value: string
  subtitle?: string
  valueColor?: string
}) {
  return (
    <div className="bg-[#141414] rounded-lg p-4 border border-gray-800">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-[#1a1a1a] rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-gray-500">{label}</p>
          <p className={clsx("text-lg font-semibold", valueColor)}>{value}</p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
        </div>
      </div>
    </div>
  )
}

function StrategyRow({
  strategy,
  trades,
  pnl,
  wins,
  losses
}: {
  strategy: string
  trades: number
  pnl: number
  wins: number
  losses: number
}) {
  const winRate = wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0
  const isProfitable = pnl >= 0

  return (
    <div className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
      <div className="flex items-center gap-3">
        <div className={clsx(
          "w-2 h-2 rounded-full",
          isProfitable ? "bg-green-500" : "bg-red-500"
        )} />
        <div>
          <p className="font-medium text-sm">{strategy}</p>
          <p className="text-xs text-gray-500">{trades} trades | {winRate.toFixed(0)}% win rate</p>
        </div>
      </div>
      <div className="text-right">
        <p className={clsx(
          "font-mono font-medium",
          isProfitable ? "text-green-400" : "text-red-400"
        )}>
          {isProfitable ? '+' : ''}${pnl.toFixed(2)}
        </p>
        <p className="text-xs text-gray-500">{wins}W / {losses}L</p>
      </div>
    </div>
  )
}

function TradeRow({
  type,
  strategy,
  cost,
  pnl,
  status,
  date,
  accountName
}: {
  type: 'paper' | 'live'
  strategy: string
  cost: number
  pnl: number | null | undefined
  status: string
  date: string
  accountName?: string
}) {
  const isProfitable = (pnl || 0) >= 0
  const statusColors: Record<string, string> = {
    open: 'bg-blue-500/20 text-blue-400',
    pending: 'bg-yellow-500/20 text-yellow-400',
    resolved_win: 'bg-green-500/20 text-green-400',
    resolved_loss: 'bg-red-500/20 text-red-400',
    win: 'bg-green-500/20 text-green-400',
    loss: 'bg-red-500/20 text-red-400',
    resolved: 'bg-gray-500/20 text-gray-400',
    executed: 'bg-blue-500/20 text-blue-400'
  }

  return (
    <div className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
      <div className="flex items-center gap-3">
        {isProfitable ? (
          <ArrowUpRight className="w-4 h-4 text-green-400" />
        ) : (
          <ArrowDownRight className="w-4 h-4 text-red-400" />
        )}
        <div>
          <div className="flex items-center gap-2">
            <p className="font-medium text-sm">{strategy}</p>
            <span className={clsx(
              "px-1.5 py-0.5 rounded text-xs",
              type === 'paper' ? "bg-blue-500/20 text-blue-400" : "bg-purple-500/20 text-purple-400"
            )}>
              {type === 'paper' ? 'Paper' : 'Live'}
            </span>
          </div>
          <p className="text-xs text-gray-500">
            Cost: ${cost.toFixed(2)}
            {accountName && ` | ${accountName}`}
          </p>
        </div>
      </div>
      <div className="text-right">
        <span className={clsx("px-2 py-0.5 rounded text-xs", statusColors[status] || 'bg-gray-500/20 text-gray-400')}>
          {status.replace('_', ' ')}
        </span>
        {pnl != null && (
          <p className={clsx(
            "font-mono text-sm mt-1",
            isProfitable ? "text-green-400" : "text-red-400"
          )}>
            {isProfitable ? '+' : ''}${pnl.toFixed(2)}
          </p>
        )}
        <p className="text-xs text-gray-500 mt-1">
          {new Date(date).toLocaleDateString()}
        </p>
      </div>
    </div>
  )
}

// Simple ASCII-style chart component
function SimplePnlChart({
  data,
  viewMode
}: {
  data: { date: string; cumSimPnl: number; cumAutoPnl: number }[]
  viewMode: ViewMode
}) {
  if (data.length === 0) return null

  const showSim = viewMode === 'simulation' || viewMode === 'all'
  const showAuto = viewMode === 'live' || viewMode === 'all'

  const allValues = data.flatMap(d => [
    ...(showSim ? [d.cumSimPnl] : []),
    ...(showAuto ? [d.cumAutoPnl] : [])
  ])
  const maxVal = Math.max(...allValues, 0)
  const minVal = Math.min(...allValues, 0)
  const range = maxVal - minVal || 1

  const chartHeight = 200

  const getY = (val: number) => {
    return chartHeight - ((val - minVal) / range) * chartHeight
  }

  const zeroY = getY(0)

  // Generate path for simulation P&L
  const simPath = showSim ? data.map((d, i) => {
    const x = (i / (data.length - 1 || 1)) * 100
    const y = getY(d.cumSimPnl)
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ') : ''

  // Generate path for auto trader P&L
  const autoPath = showAuto ? data.map((d, i) => {
    const x = (i / (data.length - 1 || 1)) * 100
    const y = getY(d.cumAutoPnl)
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ') : ''

  return (
    <div className="relative h-full">
      <svg viewBox={`0 0 100 ${chartHeight}`} className="w-full h-full" preserveAspectRatio="none">
        {/* Zero line */}
        <line
          x1="0"
          y1={zeroY}
          x2="100"
          y2={zeroY}
          stroke="#374151"
          strokeWidth="0.5"
          strokeDasharray="2,2"
        />

        {/* Simulation P&L line */}
        {showSim && simPath && (
          <path
            d={simPath}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Auto trader P&L line */}
        {showAuto && autoPath && (
          <path
            d={autoPath}
            fill="none"
            stroke="#a855f7"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>

      {/* Legend */}
      <div className="absolute bottom-0 left-0 flex gap-4 text-xs">
        {showSim && (
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-blue-500" />
            <span className="text-gray-400">Paper: ${data[data.length - 1]?.cumSimPnl.toFixed(2)}</span>
          </div>
        )}
        {showAuto && (
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-purple-500" />
            <span className="text-gray-400">Live: ${data[data.length - 1]?.cumAutoPnl.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Y-axis labels */}
      <div className="absolute top-0 right-0 h-full flex flex-col justify-between text-xs text-gray-500">
        <span>${maxVal.toFixed(0)}</span>
        <span>${minVal.toFixed(0)}</span>
      </div>
    </div>
  )
}
