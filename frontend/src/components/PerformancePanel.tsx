import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Calendar,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  getAccountTrades,
  getAllTraderOrders,
  getSimulationAccounts,
  getTraderOrchestratorStats,
  SimulationAccount,
  SimulationTrade,
  TraderOrder,
} from '../services/api'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { ScrollArea } from './ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import ValidationEnginePanel from './ValidationEnginePanel'

type ViewMode = 'simulation' | 'live' | 'all'
type TimeRange = '7d' | '30d' | '90d' | 'all'
type PerformanceSubTab = 'overview' | 'validation'

type UnifiedTrade = {
  id: string
  source: 'sandbox' | 'live'
  strategy: string
  cost: number
  pnl: number | null
  status: string
  executedAt: string
  accountName?: string
  isResolved: boolean
  isWin: boolean
  isLoss: boolean
}

type StrategyRollup = {
  strategy: string
  trades: number
  wins: number
  losses: number
  pnl: number
  sandboxTrades: number
  liveTrades: number
}

const VIEW_MODE_OPTIONS: Array<{ id: ViewMode; label: string }> = [
  { id: 'all', label: 'Unified' },
  { id: 'simulation', label: 'Sandbox' },
  { id: 'live', label: 'Live' },
]

const RANGE_OPTIONS: Array<{ id: TimeRange; label: string }> = [
  { id: '7d', label: '7D' },
  { id: '30d', label: '30D' },
  { id: '90d', label: '90D' },
  { id: 'all', label: 'All Time' },
]

function formatCurrency(value: number, compact = false): string {
  if (!Number.isFinite(value)) return '$0.00'
  if (compact) {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency: 'USD',
      notation: 'compact',
      maximumFractionDigits: 1,
    }).format(value)
  }
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function formatSignedCurrency(value: number, compact = false): string {
  const prefix = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${prefix}${formatCurrency(Math.abs(value), compact)}`
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) return '0.0%'
  return `${value.toFixed(1)}%`
}

function formatDateLabel(dateStr: string): string {
  const date = new Date(dateStr)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: '2-digit',
  })
}

function getTradeStatusClass(status: string): string {
  switch (status.toLowerCase()) {
    case 'resolved_win':
    case 'win':
      return 'bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30'
    case 'resolved_loss':
    case 'loss':
      return 'bg-red-100 text-red-800 border-red-300 dark:bg-red-500/15 dark:text-red-300 dark:border-red-500/30'
    case 'resolved':
      return 'bg-slate-100 text-slate-800 border-slate-300 dark:bg-slate-500/15 dark:text-slate-300 dark:border-slate-500/30'
    case 'open':
    case 'executed':
      return 'bg-cyan-100 text-cyan-800 border-cyan-300 dark:bg-cyan-500/15 dark:text-cyan-300 dark:border-cyan-500/30'
    case 'pending':
    case 'queued':
      return 'bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-500/15 dark:text-amber-300 dark:border-amber-500/30'
    default:
      return 'bg-muted text-muted-foreground border-border'
  }
}

export default function PerformancePanel() {
  const [activeSubTab, setActiveSubTab] = useState<PerformanceSubTab>('overview')
  const [viewMode, setViewMode] = useState<ViewMode>('all')
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('30d')

  const { data: accounts = [], isLoading: accountsLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    enabled: activeSubTab === 'overview',
  })

  const { data: orchestratorStats } = useQuery({
    queryKey: ['trader-orchestrator-stats'],
    queryFn: getTraderOrchestratorStats,
    enabled: activeSubTab === 'overview' && (viewMode === 'live' || viewMode === 'all'),
  })

  const {
    data: simulationTrades = [],
    isLoading: simTradesLoading,
    refetch: refetchSimTrades,
  } = useQuery({
    queryKey: ['all-simulation-trades', selectedAccount],
    queryFn: async () => {
      if (selectedAccount) {
        const rows = await getAccountTrades(selectedAccount, 250)
        return rows.map((row) => ({ ...row, accountName: 'Selected account' }))
      }

      const responses = await Promise.all(
        accounts.map(async (account) => {
          const rows = await getAccountTrades(account.id, 250)
          return rows.map((row: SimulationTrade) => ({
            ...row,
            accountName: account.name,
          }))
        })
      )

      return responses
        .flat()
        .sort((left, right) => new Date(right.executed_at).getTime() - new Date(left.executed_at).getTime())
    },
    enabled: activeSubTab === 'overview' && accounts.length > 0 && (viewMode === 'simulation' || viewMode === 'all'),
  })

  const {
    data: autoTrades = [],
    isLoading: autoTradesLoading,
    refetch: refetchAutoTrades,
  } = useQuery({
    queryKey: ['trader-orders'],
    queryFn: async () => {
      const rows = await getAllTraderOrders(250)
      return rows.map((row: TraderOrder) => ({
        ...row,
        executed_at: row.executed_at || row.created_at || new Date().toISOString(),
        total_cost: Number(row.notional_usd || 0),
        strategy: String(row.source || 'unknown'),
      }))
    },
    enabled: activeSubTab === 'overview' && (viewMode === 'live' || viewMode === 'all'),
  })

  const isLoading = accountsLoading || simTradesLoading || autoTradesLoading

  const filterByTimeRange = <T extends { executed_at: string }>(rows: T[]): T[] => {
    if (timeRange === 'all') return rows
    const now = Date.now()
    const daysByRange: Record<Exclude<TimeRange, 'all'>, number> = {
      '7d': 7,
      '30d': 30,
      '90d': 90,
    }
    const cutoff = now - daysByRange[timeRange] * 24 * 60 * 60 * 1000
    return rows.filter((row) => {
      const ts = new Date(row.executed_at).getTime()
      return Number.isFinite(ts) && ts >= cutoff
    })
  }

  const filteredSimTrades = useMemo(
    () => filterByTimeRange(simulationTrades),
    [simulationTrades, timeRange]
  )

  const filteredAutoTrades = useMemo(
    () => filterByTimeRange(autoTrades),
    [autoTrades, timeRange]
  )

  const unifiedTrades = useMemo(() => {
    const rows: UnifiedTrade[] = []

    if (viewMode === 'simulation' || viewMode === 'all') {
      filteredSimTrades.forEach((trade) => {
        const status = String(trade.status || 'unknown')
        rows.push({
          id: `sim-${trade.id}`,
          source: 'sandbox',
          strategy: trade.strategy_type || 'unknown',
          cost: Number(trade.total_cost || 0),
          pnl: typeof trade.actual_pnl === 'number' ? trade.actual_pnl : null,
          status,
          executedAt: trade.executed_at,
          accountName: (trade as SimulationTrade & { accountName?: string }).accountName,
          isResolved: status === 'resolved_win' || status === 'resolved_loss',
          isWin: status === 'resolved_win',
          isLoss: status === 'resolved_loss',
        })
      })
    }

    if (viewMode === 'live' || viewMode === 'all') {
      filteredAutoTrades.forEach((trade) => {
        const status = String(trade.status || 'unknown').toLowerCase()
        const pnl = typeof trade.actual_profit === 'number' ? trade.actual_profit : null
        const isResolved = ['resolved', 'win', 'loss', 'resolved_win', 'resolved_loss'].includes(status)
        rows.push({
          id: `live-${trade.id}`,
          source: 'live',
          strategy: String(trade.strategy || 'unknown'),
          cost: Number(trade.total_cost || 0),
          pnl,
          status,
          executedAt: trade.executed_at,
          isResolved,
          isWin: (pnl ?? 0) > 0 || status === 'win' || status === 'resolved_win',
          isLoss: (pnl ?? 0) < 0 || status === 'loss' || status === 'resolved_loss',
        })
      })
    }

    rows.sort((left, right) => new Date(right.executedAt).getTime() - new Date(left.executedAt).getTime())
    return rows
  }, [filteredAutoTrades, filteredSimTrades, viewMode])

  const summary = useMemo(() => {
    const resolved = unifiedTrades.filter((trade) => trade.isResolved)
    const wins = resolved.filter((trade) => trade.isWin)
    const losses = resolved.filter((trade) => trade.isLoss)

    const totalPnl = unifiedTrades.reduce((sum, trade) => sum + (trade.pnl ?? 0), 0)
    const totalCost = unifiedTrades.reduce((sum, trade) => sum + trade.cost, 0)
    const openTrades = unifiedTrades.filter((trade) => !trade.isResolved).length
    const winRate = resolved.length > 0 ? (wins.length / resolved.length) * 100 : 0
    const roi = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0
    const avgPnl = unifiedTrades.length > 0 ? totalPnl / unifiedTrades.length : 0

    const grossWins = wins.reduce((sum, trade) => sum + Math.max(0, trade.pnl ?? 0), 0)
    const grossLosses = losses.reduce((sum, trade) => sum + Math.abs(Math.min(0, trade.pnl ?? 0)), 0)
    const profitFactor = grossLosses > 0
      ? grossWins / grossLosses
      : grossWins > 0
        ? Infinity
        : 0

    return {
      totalTrades: unifiedTrades.length,
      resolvedTrades: resolved.length,
      openTrades,
      wins: wins.length,
      losses: losses.length,
      totalPnl,
      totalCost,
      winRate,
      roi,
      avgPnl,
      profitFactor,
    }
  }, [unifiedTrades])

  const strategyLeaderboard = useMemo(() => {
    const byStrategy = new Map<string, StrategyRollup>()

    unifiedTrades.forEach((trade) => {
      const key = trade.strategy || 'unknown'
      const current = byStrategy.get(key) || {
        strategy: key,
        trades: 0,
        wins: 0,
        losses: 0,
        pnl: 0,
        sandboxTrades: 0,
        liveTrades: 0,
      }

      current.trades += 1
      current.pnl += trade.pnl ?? 0
      current.wins += trade.isWin ? 1 : 0
      current.losses += trade.isLoss ? 1 : 0
      current.sandboxTrades += trade.source === 'sandbox' ? 1 : 0
      current.liveTrades += trade.source === 'live' ? 1 : 0

      byStrategy.set(key, current)
    })

    return Array.from(byStrategy.values()).sort((left, right) => {
      if (left.pnl === right.pnl) return right.trades - left.trades
      return right.pnl - left.pnl
    })
  }, [unifiedTrades])

  const cumulativePnlData = useMemo(() => {
    const daily = new Map<string, { sim: number; live: number }>()

    filteredSimTrades.forEach((trade) => {
      const day = trade.executed_at?.split('T')[0]
      if (!day) return
      const current = daily.get(day) || { sim: 0, live: 0 }
      current.sim += trade.actual_pnl || 0
      daily.set(day, current)
    })

    filteredAutoTrades.forEach((trade) => {
      const day = trade.executed_at?.split('T')[0]
      if (!day) return
      const current = daily.get(day) || { sim: 0, live: 0 }
      current.live += trade.actual_profit || 0
      daily.set(day, current)
    })

    const sortedDays = Array.from(daily.keys()).sort()
    let cumSim = 0
    let cumLive = 0

    return sortedDays.map((day) => {
      const row = daily.get(day) || { sim: 0, live: 0 }
      cumSim += row.sim
      cumLive += row.live
      return {
        date: day,
        dailySimPnl: row.sim,
        dailyLivePnl: row.live,
        cumSimPnl: cumSim,
        cumLivePnl: cumLive,
        cumTotalPnl: cumSim + cumLive,
      }
    })
  }, [filteredAutoTrades, filteredSimTrades])

  const maxDrawdown = useMemo(() => {
    if (cumulativePnlData.length === 0) return 0

    let peak = Number.NEGATIVE_INFINITY
    let drawdown = 0

    cumulativePnlData.forEach((point) => {
      const value = viewMode === 'simulation'
        ? point.cumSimPnl
        : viewMode === 'live'
          ? point.cumLivePnl
          : point.cumTotalPnl
      if (value > peak) peak = value
      drawdown = Math.max(drawdown, peak - value)
    })

    return drawdown
  }, [cumulativePnlData, viewMode])

  const handleRefresh = () => {
    if (viewMode === 'simulation' || viewMode === 'all') {
      void refetchSimTrades()
    }
    if (viewMode === 'live' || viewMode === 'all') {
      void refetchAutoTrades()
    }
  }

  const viewModeLabel = viewMode === 'all'
    ? 'Unified stream'
    : viewMode === 'simulation'
      ? 'Sandbox stream'
      : 'Live stream'

  return (
    <div className="space-y-5">
      <Card className="overflow-hidden border-border/80 bg-card/80">
        <CardContent className="p-5">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-300 bg-cyan-100 px-2.5 py-1 text-[10px] uppercase tracking-wider text-cyan-800 dark:border-cyan-500/25 dark:bg-cyan-500/10 dark:text-cyan-200">
                <Sparkles className="h-3 w-3" />
                Performance Command Center
              </div>
              <h2 className="mt-2 flex items-center gap-2 text-xl font-semibold">
                <BarChart3 className="h-5 w-5 text-cyan-700 dark:text-cyan-300" />
                Performance Intelligence
              </h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Data-dense execution telemetry, trade quality diagnostics, and validation controls in one surface.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <Badge variant="outline" className="border-cyan-300 bg-cyan-100 text-cyan-800 dark:border-cyan-500/25 dark:bg-cyan-500/10 dark:text-cyan-200">
                {viewModeLabel}
              </Badge>
              <Badge variant="outline" className="border-emerald-300 bg-emerald-100 text-emerald-800 dark:border-emerald-500/25 dark:bg-emerald-500/10 dark:text-emerald-200">
                {summary.totalTrades} trades in scope
              </Badge>
              <Badge variant="outline" className="border-amber-300 bg-amber-100 text-amber-800 dark:border-amber-500/25 dark:bg-amber-500/10 dark:text-amber-200">
                Range: {RANGE_OPTIONS.find((option) => option.id === timeRange)?.label}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs value={activeSubTab} onValueChange={(value) => setActiveSubTab(value as PerformanceSubTab)} className="space-y-4">
        <TabsList className="h-auto w-full justify-start gap-2 rounded-xl border border-border/60 bg-card/70 p-1.5">
          <TabsTrigger
            value="overview"
            className="h-auto min-w-[220px] items-start justify-start rounded-lg px-3 py-2 data-[state=active]:border data-[state=active]:border-cyan-300 data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-900 dark:data-[state=active]:border-cyan-500/40 dark:data-[state=active]:bg-cyan-500/10 dark:data-[state=active]:text-cyan-100"
          >
            <div className="flex items-start gap-2 text-left">
              <BarChart3 className="mt-0.5 h-4 w-4 text-cyan-700 dark:text-cyan-300" />
              <div>
                <p className="text-sm font-medium">Overview Grid</p>
                <p className="text-[11px] text-muted-foreground">P&L, strategy mix, and trade tape</p>
              </div>
            </div>
          </TabsTrigger>
          <TabsTrigger
            value="validation"
            className="h-auto min-w-[220px] items-start justify-start rounded-lg px-3 py-2 data-[state=active]:border data-[state=active]:border-emerald-300 data-[state=active]:bg-emerald-100 data-[state=active]:text-emerald-900 dark:data-[state=active]:border-emerald-500/40 dark:data-[state=active]:bg-emerald-500/10 dark:data-[state=active]:text-emerald-100"
          >
            <div className="flex items-start gap-2 text-left">
              <ShieldCheck className="mt-0.5 h-4 w-4 text-emerald-700 dark:text-emerald-300" />
              <div>
                <p className="text-sm font-medium">Validation Ops</p>
                <p className="text-[11px] text-muted-foreground">Jobs, guardrails, and strategy health</p>
              </div>
            </div>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-0 space-y-4">
          <Card className="border-border/60 bg-card/75">
            <CardContent className="p-4">
              <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                <div className="space-y-2">
                  <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Data Scope</p>
                  <div className="flex flex-wrap gap-2">
                    {VIEW_MODE_OPTIONS.map((option) => (
                      <button
                        key={option.id}
                        type="button"
                        onClick={() => setViewMode(option.id)}
                        className={cn(
                          'h-8 rounded-md border px-3 text-xs transition-colors',
                          viewMode === option.id
                            ? 'border-cyan-300 bg-cyan-100 text-cyan-900 dark:border-cyan-500/40 dark:bg-cyan-500/15 dark:text-cyan-100'
                            : 'border-border bg-background/60 text-muted-foreground hover:text-foreground'
                        )}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Time Horizon</p>
                  <div className="flex flex-wrap gap-2">
                    {RANGE_OPTIONS.map((option) => (
                      <button
                        key={option.id}
                        type="button"
                        onClick={() => setTimeRange(option.id)}
                        className={cn(
                          'h-8 rounded-md border px-3 text-xs transition-colors',
                          timeRange === option.id
                            ? 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/40 dark:bg-emerald-500/15 dark:text-emerald-100'
                            : 'border-border bg-background/60 text-muted-foreground hover:text-foreground'
                        )}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
                    <select
                      value={selectedAccount || ''}
                      onChange={(event) => setSelectedAccount(event.target.value || null)}
                      className="h-8 rounded-md border border-border bg-background/80 px-2.5 text-xs"
                    >
                      <option value="">All simulation accounts</option>
                      {accounts.map((account: SimulationAccount) => (
                        <option key={account.id} value={account.id}>
                          {account.name}
                        </option>
                      ))}
                    </select>
                  )}

                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleRefresh}
                    disabled={isLoading}
                    className="h-8"
                  >
                    <RefreshCw className={cn('mr-1.5 h-3.5 w-3.5', isLoading && 'animate-spin')} />
                    Refresh
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6 card-stagger">
            <PerformanceMetricTile
              label="Net P&L"
              value={formatSignedCurrency(summary.totalPnl, true)}
              helper={formatCurrency(summary.totalPnl)}
              icon={summary.totalPnl >= 0 ? TrendingUp : TrendingDown}
              tone={summary.totalPnl >= 0 ? 'good' : 'bad'}
            />
            <PerformanceMetricTile
              label="ROI"
              value={formatPercent(summary.roi)}
              helper={summary.totalCost > 0 ? `on ${formatCurrency(summary.totalCost, true)} deployed` : 'no deployed capital'}
              icon={Target}
              tone={summary.roi >= 0 ? 'good' : 'bad'}
            />
            <PerformanceMetricTile
              label="Win Rate"
              value={formatPercent(summary.winRate)}
              helper={`${summary.wins}W / ${summary.losses}L`}
              icon={Activity}
              tone={summary.winRate >= 50 ? 'good' : 'warn'}
            />
            <PerformanceMetricTile
              label="Open Trades"
              value={String(summary.openTrades)}
              helper={`${summary.resolvedTrades} resolved`}
              icon={Calendar}
              tone={summary.openTrades > 0 ? 'info' : 'neutral'}
            />
            <PerformanceMetricTile
              label="Avg Trade P&L"
              value={formatSignedCurrency(summary.avgPnl)}
              helper={`${summary.totalTrades} total trades`}
              icon={summary.avgPnl >= 0 ? ArrowUpRight : ArrowDownRight}
              tone={summary.avgPnl >= 0 ? 'good' : 'bad'}
            />
            <PerformanceMetricTile
              label="Max Drawdown"
              value={formatCurrency(maxDrawdown, true)}
              helper={
                Number.isFinite(summary.profitFactor)
                  ? `profit factor ${summary.profitFactor === Infinity ? '∞' : summary.profitFactor.toFixed(2)}`
                  : 'profit factor n/a'
              }
              icon={TrendingDown}
              tone={maxDrawdown > 0 ? 'warn' : 'neutral'}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <Card className="xl:col-span-8 border-border/60 bg-card/80">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center justify-between gap-3 text-base font-semibold">
                  <span className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-cyan-700 dark:text-cyan-300" />
                    Cumulative P&L Stream
                  </span>
                  {orchestratorStats?.last_trade_at && (
                    <span className="text-[11px] font-normal text-muted-foreground">
                      last orchestrator trade: {new Date(orchestratorStats.last_trade_at).toLocaleString()}
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {cumulativePnlData.length === 0 ? (
                  <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-border/60 bg-background/20 text-sm text-muted-foreground">
                    No trade history in the selected range.
                  </div>
                ) : (
                  <div className="h-72">
                    <PerformancePnlChart data={cumulativePnlData} viewMode={viewMode} />
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="xl:col-span-4 border-border/60 bg-card/80">
              <CardHeader className="pb-3">
                <CardTitle className="text-base font-semibold">Strategy Leaderboard</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {strategyLeaderboard.length === 0 && (
                  <div className="rounded-md border border-dashed border-border/60 bg-background/20 px-3 py-5 text-center text-sm text-muted-foreground">
                    No strategy activity yet.
                  </div>
                )}

                {strategyLeaderboard.slice(0, 12).map((row) => {
                  const denominator = Math.max(1, summary.totalTrades)
                  const volumeShare = (row.trades / denominator) * 100
                  const winRate = row.wins + row.losses > 0
                    ? (row.wins / (row.wins + row.losses)) * 100
                    : 0
                  return (
                    <div key={row.strategy} className="rounded-lg border border-border/55 bg-background/30 p-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <p className="truncate text-sm font-medium">{row.strategy}</p>
                        <p className={cn(
                          'text-xs font-data',
                          row.pnl >= 0 ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-700 dark:text-red-300'
                        )}>
                          {formatSignedCurrency(row.pnl, true)}
                        </p>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                        <span>{row.trades} trades</span>
                        <span>•</span>
                        <span>{formatPercent(winRate)} hit rate</span>
                        <span>•</span>
                        <span>{row.liveTrades} live / {row.sandboxTrades} sandbox</span>
                      </div>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-border/40">
                        <div
                          className={cn(
                            'h-full rounded-full',
                            row.pnl >= 0 ? 'bg-emerald-500/75 dark:bg-emerald-400/80' : 'bg-red-500/75 dark:bg-red-400/80'
                          )}
                          style={{ width: `${Math.max(6, Math.min(100, volumeShare))}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </CardContent>
            </Card>
          </div>

          <Card className="border-border/60 bg-card/80">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-semibold">Trade Tape</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <ScrollArea className="h-[420px] pr-3">
                <Table>
                  <TableHeader>
                    <TableRow className="border-border/60 bg-background/30">
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Timestamp</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Source</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Strategy</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Status</TableHead>
                      <TableHead className="h-9 py-2 text-right text-[11px] uppercase tracking-wide">Cost</TableHead>
                      <TableHead className="h-9 py-2 text-right text-[11px] uppercase tracking-wide">P&L</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Account</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {unifiedTrades.slice(0, 150).map((trade) => (
                      <TableRow key={trade.id} className="border-border/45">
                        <TableCell className="py-2 font-data text-xs text-muted-foreground">
                          {new Date(trade.executedAt).toLocaleString()}
                        </TableCell>
                        <TableCell className="py-2">
                          <Badge
                            variant="outline"
                            className={cn(
                              'text-[10px] uppercase tracking-wide',
                              trade.source === 'sandbox'
                                ? 'border-amber-300 bg-amber-100 text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200'
                                : 'border-cyan-300 bg-cyan-100 text-cyan-800 dark:border-cyan-500/30 dark:bg-cyan-500/10 dark:text-cyan-200'
                            )}
                          >
                            {trade.source}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-2 text-sm">{trade.strategy}</TableCell>
                        <TableCell className="py-2">
                          <Badge variant="outline" className={cn('text-[10px] uppercase', getTradeStatusClass(trade.status))}>
                            {trade.status.replace(/_/g, ' ')}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-2 text-right font-data text-xs">
                          {formatCurrency(trade.cost)}
                        </TableCell>
                        <TableCell className={cn(
                          'py-2 text-right font-data text-xs',
                          (trade.pnl ?? 0) >= 0 ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-700 dark:text-red-300'
                        )}>
                          {trade.pnl == null ? '—' : formatSignedCurrency(trade.pnl)}
                        </TableCell>
                        <TableCell className="py-2 text-xs text-muted-foreground">
                          {trade.accountName || (trade.source === 'live' ? 'Orchestrator' : '—')}
                        </TableCell>
                      </TableRow>
                    ))}
                    {unifiedTrades.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={7} className="py-8 text-center text-sm text-muted-foreground">
                          No trades found for this view and range.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="validation" className="mt-0 space-y-4">
          <ValidationEnginePanel />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function PerformanceMetricTile({
  label,
  value,
  helper,
  icon: Icon,
  tone,
}: {
  label: string
  value: string
  helper: string
  icon: React.ElementType
  tone: 'good' | 'bad' | 'warn' | 'neutral' | 'info'
}) {
  const toneClasses: Record<typeof tone, string> = {
    good: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200',
    bad: 'border-red-300 bg-red-100 text-red-900 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200',
    warn: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200',
    neutral: 'border-border/70 bg-background/35 text-foreground',
    info: 'border-cyan-300 bg-cyan-100 text-cyan-900 dark:border-cyan-500/30 dark:bg-cyan-500/10 dark:text-cyan-200',
  }

  return (
    <Card className={cn('border', toneClasses[tone])}>
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[10px] uppercase tracking-wide opacity-85">{label}</p>
          <Icon className="h-3.5 w-3.5 opacity-80" />
        </div>
        <p className="mt-2 font-data text-lg font-semibold">{value}</p>
        <p className="mt-1 text-[11px] opacity-85">{helper}</p>
      </CardContent>
    </Card>
  )
}

function PerformancePnlChart({
  data,
  viewMode,
}: {
  data: Array<{
    date: string
    cumSimPnl: number
    cumLivePnl: number
    cumTotalPnl: number
  }>
  viewMode: ViewMode
}) {
  const showSandbox = viewMode === 'simulation' || viewMode === 'all'
  const showLive = viewMode === 'live' || viewMode === 'all'

  const tooltipFormatter = (value: number, key: string) => {
    const label = key === 'cumSimPnl'
      ? 'Sandbox cumulative'
      : key === 'cumLivePnl'
        ? 'Live cumulative'
        : 'Unified cumulative'
    return [formatCurrency(value), label]
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <defs>
          <linearGradient id="sandboxGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.32} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.04} />
          </linearGradient>
          <linearGradient id="liveGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.32} />
            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.04} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="hsl(var(--border) / 0.45)" strokeDasharray="3 3" />

        <XAxis
          dataKey="date"
          tickFormatter={formatDateLabel}
          tick={{ fontSize: 11 }}
          stroke="hsl(var(--muted-foreground))"
          axisLine={{ stroke: 'hsl(var(--border))' }}
          tickLine={{ stroke: 'hsl(var(--border))' }}
        />
        <YAxis
          tickFormatter={(value: number) => formatCurrency(value, true)}
          tick={{ fontSize: 11 }}
          width={72}
          stroke="hsl(var(--muted-foreground))"
          axisLine={{ stroke: 'hsl(var(--border))' }}
          tickLine={{ stroke: 'hsl(var(--border))' }}
        />

        <Tooltip
          formatter={(value, key) => tooltipFormatter(Number(value), String(key))}
          labelFormatter={(label) => `Date ${label}`}
          contentStyle={{
            borderRadius: 10,
            border: '1px solid hsl(var(--border))',
            background: 'hsl(var(--popover))',
            color: 'hsl(var(--popover-foreground))',
            fontSize: 12,
          }}
        />

        <Legend
          wrapperStyle={{ fontSize: '11px' }}
          formatter={(value) => {
            if (value === 'cumSimPnl') return 'Sandbox'
            if (value === 'cumLivePnl') return 'Live'
            return 'Unified'
          }}
        />

        {showSandbox && (
          <Area
            type="monotone"
            dataKey="cumSimPnl"
            stroke="#f59e0b"
            strokeWidth={2}
            fill="url(#sandboxGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#f59e0b', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}

        {showLive && (
          <Area
            type="monotone"
            dataKey="cumLivePnl"
            stroke="#22d3ee"
            strokeWidth={2}
            fill="url(#liveGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#22d3ee', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}

        {viewMode === 'all' && (
          <Line
            type="monotone"
            dataKey="cumTotalPnl"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            strokeDasharray="5 3"
            activeDot={{ r: 4, stroke: '#34d399', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}
