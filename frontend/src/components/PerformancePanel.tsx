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
import { cn } from '../lib/utils'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import {
  getSimulationAccounts,
  getAccountTrades,
  getAutoTraderStats,
  getAutoTraderTrades,
  SimulationAccount,
  SimulationTrade
} from '../services/api'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { ScrollArea } from './ui/scroll-area'

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
          <p className="text-sm text-muted-foreground">Track your trading performance over time</p>
        </div>
        <Button
          variant="secondary"
          onClick={handleRefresh}
          disabled={isLoading}
        >
          <RefreshCw className={cn("w-4 h-4 mr-2", isLoading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList>
            <TabsTrigger value="all">All Trading</TabsTrigger>
            <TabsTrigger value="simulation">Sandbox Trading</TabsTrigger>
            <TabsTrigger value="live">Live Trading</TabsTrigger>
          </TabsList>
        </Tabs>

        <Tabs value={timeRange} onValueChange={(v) => setTimeRange(v as TimeRange)}>
          <TabsList>
            <TabsTrigger value="7d">7D</TabsTrigger>
            <TabsTrigger value="30d">30D</TabsTrigger>
            <TabsTrigger value="90d">90D</TabsTrigger>
            <TabsTrigger value="all">All Time</TabsTrigger>
          </TabsList>
        </Tabs>

        {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
          <select
            value={selectedAccount || ''}
            onChange={(e) => setSelectedAccount(e.target.value || null)}
            className="bg-muted border border-border rounded-lg px-3 py-2 text-sm"
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
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 card-stagger">
            {(viewMode === 'simulation' || viewMode === 'all') && (
              <>
                <MetricCard
                  icon={<Activity className="w-5 h-5 text-blue-500" />}
                  label="Sandbox Trades"
                  value={simMetrics.totalTrades.toString()}
                  subtitle={`${simMetrics.openTrades} open`}
                />
                <MetricCard
                  icon={simMetrics.totalPnl >= 0
                    ? <TrendingUp className="w-5 h-5 text-green-500" />
                    : <TrendingDown className="w-5 h-5 text-red-500" />
                  }
                  label="Sandbox P&L"
                  value={`${simMetrics.totalPnl >= 0 ? '+' : ''}$${simMetrics.totalPnl.toFixed(2)}`}
                  valueColor={simMetrics.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
                />
                <MetricCard
                  icon={<Award className="w-5 h-5 text-yellow-500" />}
                  label="Sandbox Win Rate"
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
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-semibold flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-500" />
                Cumulative P&L Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              {cumulativePnlData.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No trade data available for the selected time range
                </div>
              ) : (
                <div className="h-64">
                  <SimplePnlChart data={cumulativePnlData} viewMode={viewMode} />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Strategy Performance */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {(viewMode === 'simulation' || viewMode === 'all') && Object.keys(simMetrics.byStrategy).length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base font-semibold flex items-center gap-2">
                    <PieChart className="w-5 h-5 text-blue-500" />
                    Sandbox Trading by Strategy
                  </CardTitle>
                </CardHeader>
                <CardContent>
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
                </CardContent>
              </Card>
            )}

            {(viewMode === 'live' || viewMode === 'all') && Object.keys(autoMetrics.byStrategy).length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base font-semibold flex items-center gap-2">
                    <PieChart className="w-5 h-5 text-purple-500" />
                    Live Trading by Strategy
                  </CardTitle>
                </CardHeader>
                <CardContent>
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
                </CardContent>
              </Card>
            )}
          </div>

          {/* Recent Trades */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-semibold flex items-center gap-2">
                <Calendar className="w-5 h-5 text-muted-foreground" />
                Recent Trades
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-2">
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
                    <div className="text-center py-8 text-muted-foreground">
                      No trades found for the selected time range
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
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
  valueColor = 'text-foreground'
}: {
  icon: React.ReactNode
  label: string
  value: string
  subtitle?: string
  valueColor?: string
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 p-4">
        <div className="p-2 bg-muted rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className={cn("text-lg font-semibold font-data", valueColor)}>{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
      </CardContent>
    </Card>
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
    <div className="flex items-center justify-between bg-muted rounded-lg p-3">
      <div className="flex items-center gap-3">
        <div className={cn(
          "w-2 h-2 rounded-full",
          isProfitable ? "bg-green-500" : "bg-red-500"
        )} />
        <div>
          <p className="font-medium text-sm">{strategy}</p>
          <p className="text-xs text-muted-foreground">{trades} trades | {winRate.toFixed(0)}% win rate</p>
        </div>
      </div>
      <div className="text-right">
        <p className={cn(
          "font-data font-medium",
          isProfitable ? "text-green-400 data-glow-green" : "text-red-400 data-glow-red"
        )}>
          {isProfitable ? '+' : ''}${pnl.toFixed(2)}
        </p>
        <p className="text-xs text-muted-foreground">{wins}W / {losses}L</p>
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
    open: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    pending: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    resolved_win: 'bg-green-500/20 text-green-400 border-green-500/30',
    resolved_loss: 'bg-red-500/20 text-red-400 border-red-500/30',
    win: 'bg-green-500/20 text-green-400 border-green-500/30',
    loss: 'bg-red-500/20 text-red-400 border-red-500/30',
    resolved: 'bg-gray-500/20 text-muted-foreground border-gray-500/30',
    executed: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
  }

  return (
    <div className="flex items-center justify-between bg-muted rounded-lg p-3">
      <div className="flex items-center gap-3">
        {isProfitable ? (
          <ArrowUpRight className="w-4 h-4 text-green-400" />
        ) : (
          <ArrowDownRight className="w-4 h-4 text-red-400" />
        )}
        <div>
          <div className="flex items-center gap-2">
            <p className="font-medium text-sm">{strategy}</p>
            <Badge
              variant="outline"
              className={cn(
                type === 'paper'
                  ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
                  : "bg-purple-500/20 text-purple-400 border-purple-500/30"
              )}
            >
              {type === 'paper' ? 'Sandbox' : 'Live'}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            Cost: ${cost.toFixed(2)}
            {accountName && ` | ${accountName}`}
          </p>
        </div>
      </div>
      <div className="text-right">
        <Badge
          variant="outline"
          className={cn(statusColors[status] || 'bg-gray-500/20 text-muted-foreground border-gray-500/30')}
        >
          {status.replace('_', ' ')}
        </Badge>
        {pnl != null && (
          <p className={cn(
            "font-data text-sm mt-1",
            isProfitable ? "text-green-400" : "text-red-400"
          )}>
            {isProfitable ? '+' : ''}${pnl.toFixed(2)}
          </p>
        )}
        <p className="text-xs text-muted-foreground mt-1">
          {new Date(date).toLocaleDateString()}
        </p>
      </div>
    </div>
  )
}

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

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return `${date.getMonth() + 1}/${date.getDate()}`
  }

  const formatDollar = (value: number) => {
    return `$${value.toFixed(2)}`
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
        <defs>
          <linearGradient id="simGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="autoGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tickFormatter={formatDate}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 11 }}
          axisLine={{ stroke: 'hsl(var(--border))' }}
        />
        <YAxis
          tickFormatter={formatDollar}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 11 }}
          axisLine={{ stroke: 'hsl(var(--border))' }}
          width={65}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--popover))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
            fontSize: '12px',
            color: 'hsl(var(--popover-foreground))',
          }}
          labelFormatter={(label) => `Date: ${label}`}
          formatter={(value: any, name: any) => [
            typeof value === 'number' ? `$${value.toFixed(2)}` : '$0.00',
            name === 'cumSimPnl' ? 'Sandbox P&L' : 'Live P&L'
          ]}
        />
        <Legend
          formatter={(value) => value === 'cumSimPnl' ? 'Sandbox P&L' : 'Live P&L'}
          wrapperStyle={{ fontSize: '12px' }}
        />
        {showSim && (
          <Area
            type="monotone"
            dataKey="cumSimPnl"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#simGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#3b82f6', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}
        {showAuto && (
          <Area
            type="monotone"
            dataKey="cumAutoPnl"
            stroke="#a855f7"
            strokeWidth={2}
            fill="url(#autoGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#a855f7', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}
