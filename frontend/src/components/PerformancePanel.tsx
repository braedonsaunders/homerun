import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Calendar,
  RefreshCw,
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
import { ScrollArea } from './ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table'

type ViewMode = 'simulation' | 'live' | 'all'
type TimeRange = '7d' | '30d' | '90d' | 'all'

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

type LiveTraderOrder = TraderOrder & {
  executed_at: string
  total_cost: number
  strategy: string
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
    case 'closed_win':
    case 'win':
      return 'bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30'
    case 'resolved_loss':
    case 'closed_loss':
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
  const [viewMode, setViewMode] = useState<ViewMode>('all')
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('30d')

  const { data: accounts = [], isLoading: accountsLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  const simulationAccountKey = useMemo(
    () => accounts.map((account) => account.id).sort().join('|'),
    [accounts]
  )

  const { data: orchestratorStats } = useQuery({
    queryKey: ['trader-orchestrator-stats'],
    queryFn: getTraderOrchestratorStats,
    enabled: viewMode === 'live' || viewMode === 'all',
  })

  const {
    data: simulationTrades = [],
    isLoading: simTradesLoading,
    refetch: refetchSimTrades,
  } = useQuery({
    queryKey: ['all-simulation-trades', selectedAccount, simulationAccountKey],
    queryFn: async () => {
      if (selectedAccount) {
        const rows = await getAccountTrades(selectedAccount, 250)
        const accountName = accounts.find((account) => account.id === selectedAccount)?.name || 'Selected account'
        return rows.map((row) => ({ ...row, accountName }))
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
    enabled: accounts.length > 0 && (viewMode === 'simulation' || viewMode === 'all'),
  })

  const {
    data: autoTrades = [],
    isLoading: autoTradesLoading,
    refetch: refetchAutoTrades,
  } = useQuery<LiveTraderOrder[]>({
    queryKey: ['trader-orders'],
    queryFn: async () => {
      const rows = await getAllTraderOrders(250)
      const normalizedRows: LiveTraderOrder[] = []
      rows.forEach((row) => {
        if (String(row.mode || '').toLowerCase() !== 'live') return

        const executedAt = row.executed_at || row.created_at
        if (!executedAt) return

        normalizedRows.push({
          ...row,
          executed_at: executedAt,
          total_cost: Number(row.notional_usd || 0),
          strategy: String(row.source || 'unknown'),
        })
      })
      return normalizedRows
    },
    enabled: viewMode === 'live' || viewMode === 'all',
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
          isResolved: ['resolved_win', 'resolved_loss', 'closed_win', 'closed_loss'].includes(status),
          isWin: status === 'resolved_win' || status === 'closed_win',
          isLoss: status === 'resolved_loss' || status === 'closed_loss',
        })
      })
    }

    if (viewMode === 'live' || viewMode === 'all') {
      filteredAutoTrades.forEach((trade) => {
        const status = String(trade.status || 'unknown').toLowerCase()
        const pnl = typeof trade.actual_profit === 'number' ? trade.actual_profit : null
        const isResolved = [
          'resolved',
          'win',
          'loss',
          'resolved_win',
          'resolved_loss',
          'closed_win',
          'closed_loss',
        ].includes(status)
        rows.push({
          id: `live-${trade.id}`,
          source: 'live',
          strategy: String(trade.strategy || 'unknown'),
          cost: Number(trade.total_cost || 0),
          pnl,
          status,
          executedAt: trade.executed_at,
          isResolved,
          isWin: (pnl ?? 0) > 0 || status === 'win' || status === 'resolved_win' || status === 'closed_win',
          isLoss: (pnl ?? 0) < 0 || status === 'loss' || status === 'resolved_loss' || status === 'closed_loss',
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

  return (
    <div className="h-full min-h-0 flex flex-col gap-1.5">
      {/* Control Strip */}
      <div className="shrink-0 space-y-2">
        {/* Row 1: Title + refresh */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-300" />
              Performance
            </h2>
            <Badge variant="outline" className="text-[10px] border-border/50">
              {summary.totalTrades} trades
            </Badge>
          </div>

          <Button variant="ghost" size="sm" onClick={handleRefresh} disabled={isLoading} className="h-7 px-2">
            <RefreshCw className={cn('w-3.5 h-3.5', isLoading && 'animate-spin')} />
          </Button>
        </div>

        {/* Row 2: Scope + time range + account */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1">
            {VIEW_MODE_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setViewMode(option.id)}
                className={cn(
                  'h-7 rounded-md border px-2.5 text-xs transition-colors',
                  viewMode === option.id
                    ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-100'
                    : 'border-border bg-background/60 text-muted-foreground hover:text-foreground'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="w-px h-5 bg-border/50" />

          <div className="flex items-center gap-1">
            {RANGE_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setTimeRange(option.id)}
                className={cn(
                  'h-7 rounded-md border px-2.5 text-xs transition-colors',
                  timeRange === option.id
                    ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-100'
                    : 'border-border bg-background/60 text-muted-foreground hover:text-foreground'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
            <>
              <div className="w-px h-5 bg-border/50" />
              <select
                value={selectedAccount || ''}
                onChange={(event) => setSelectedAccount(event.target.value || null)}
                className="h-7 rounded-md border border-border bg-background/80 px-2 text-xs"
              >
                <option value="">All accounts</option>
                {accounts.map((account: SimulationAccount) => (
                  <option key={account.id} value={account.id}>{account.name}</option>
                ))}
              </select>
            </>
          )}
        </div>
      </div>

      {/* Metric Strip */}
      <div className="shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 border-y border-border/50 py-1.5 px-0.5">
        <MetricChip
          label="P&L"
          value={formatSignedCurrency(summary.totalPnl, true)}
          detail={formatCurrency(summary.totalPnl)}
          icon={summary.totalPnl >= 0 ? TrendingUp : TrendingDown}
          valueClassName={summary.totalPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
        />
        <MetricChip
          label="ROI"
          value={formatPercent(summary.roi)}
          detail={`on ${formatCurrency(summary.totalCost, true)}`}
          icon={Target}
          valueClassName={summary.roi >= 0 ? 'text-emerald-300' : 'text-red-300'}
        />
        <MetricChip
          label="Win Rate"
          value={formatPercent(summary.winRate)}
          detail={`${summary.wins}W / ${summary.losses}L`}
          icon={Activity}
          valueClassName={summary.winRate >= 50 ? 'text-emerald-300' : 'text-amber-300'}
        />
        <MetricChip
          label="Open"
          value={String(summary.openTrades)}
          detail={`${summary.resolvedTrades} closed`}
          icon={Calendar}
        />
        <MetricChip
          label="Avg P&L"
          value={formatSignedCurrency(summary.avgPnl)}
          icon={summary.avgPnl >= 0 ? ArrowUpRight : ArrowDownRight}
          valueClassName={summary.avgPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
        />
        <MetricChip
          label="Drawdown"
          value={formatCurrency(maxDrawdown, true)}
          detail={Number.isFinite(summary.profitFactor) ? `PF ${summary.profitFactor === Infinity ? '∞' : summary.profitFactor.toFixed(2)}` : ''}
          icon={TrendingDown}
          valueClassName={maxDrawdown > 0 ? 'text-amber-300' : undefined}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-0 flex flex-col gap-2">
        {/* Chart + Leaderboard */}
        <div className="shrink-0 h-[260px] grid gap-2 xl:grid-cols-12">
          <div className="xl:col-span-8 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col">
            <div className="shrink-0 flex items-center justify-between gap-2 mb-2">
              <p className="text-xs font-semibold flex items-center gap-1.5">
                <BarChart3 className="h-3.5 w-3.5 text-cyan-300" />
                Cumulative P&L
              </p>
              {orchestratorStats?.last_trade_at && (
                <span className="text-[10px] text-muted-foreground">
                  last trade: {new Date(orchestratorStats.last_trade_at).toLocaleString()}
                </span>
              )}
            </div>
            <div className="flex-1 min-h-0">
              {cumulativePnlData.length === 0 ? (
                <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                  No trade history in range.
                </div>
              ) : (
                <PerformancePnlChart data={cumulativePnlData} viewMode={viewMode} />
              )}
            </div>
          </div>

          <div className="xl:col-span-4 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col">
            <p className="shrink-0 text-xs font-semibold mb-2">Strategy Leaderboard</p>
            <ScrollArea className="flex-1 min-h-0">
              <div className="space-y-1.5 pr-2">
                {strategyLeaderboard.length === 0 ? (
                  <div className="text-xs text-muted-foreground text-center py-4">No strategy activity.</div>
                ) : (
                  strategyLeaderboard.slice(0, 12).map((row) => {
                    const winRate = row.wins + row.losses > 0 ? (row.wins / (row.wins + row.losses)) * 100 : 0
                    return (
                      <div key={row.strategy} className="rounded border border-border/50 bg-background/30 px-2 py-1.5">
                        <div className="flex items-center justify-between gap-1">
                          <p className="truncate text-xs">{row.strategy}</p>
                          <p className={cn('text-[11px] font-mono', row.pnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                            {formatSignedCurrency(row.pnl, true)}
                          </p>
                        </div>
                        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground mt-0.5">
                          <span>{row.trades}t</span>
                          <span>{formatPercent(winRate)}</span>
                          <span>{row.liveTrades}L/{row.sandboxTrades}S</span>
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Trade Tape — fills remaining space */}
        <div className="flex-1 min-h-0 rounded-md border border-border/60 bg-card/80 flex flex-col">
          <div className="shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-b border-border/40">
            <p className="text-xs font-semibold">Trade Tape</p>
            <span className="text-[10px] text-muted-foreground">{unifiedTrades.length} trades</span>
          </div>
          <ScrollArea className="flex-1 min-h-0">
            <Table>
              <TableHeader>
                <TableRow className="sticky top-0 z-10 bg-card/95 backdrop-blur-sm">
                  <TableHead className="h-7 py-1 text-[10px] uppercase tracking-wide">Timestamp</TableHead>
                  <TableHead className="h-7 py-1 text-[10px] uppercase tracking-wide">Source</TableHead>
                  <TableHead className="h-7 py-1 text-[10px] uppercase tracking-wide">Strategy</TableHead>
                  <TableHead className="h-7 py-1 text-[10px] uppercase tracking-wide">Status</TableHead>
                  <TableHead className="h-7 py-1 text-right text-[10px] uppercase tracking-wide">Cost</TableHead>
                  <TableHead className="h-7 py-1 text-right text-[10px] uppercase tracking-wide">P&L</TableHead>
                  <TableHead className="h-7 py-1 text-[10px] uppercase tracking-wide">Account</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {unifiedTrades.slice(0, 150).map((trade) => (
                  <TableRow key={trade.id} className="border-border/40">
                    <TableCell className="py-1.5 font-mono text-[11px] text-muted-foreground">
                      {new Date(trade.executedAt).toLocaleString()}
                    </TableCell>
                    <TableCell className="py-1.5">
                      <Badge
                        variant="outline"
                        className={cn(
                          'text-[9px] uppercase',
                          trade.source === 'sandbox'
                            ? 'border-amber-500/30 bg-amber-500/10 text-amber-200'
                            : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200'
                        )}
                      >
                        {trade.source}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-1.5 text-xs">{trade.strategy}</TableCell>
                    <TableCell className="py-1.5">
                      <Badge variant="outline" className={cn('text-[9px] uppercase', getTradeStatusClass(trade.status))}>
                        {trade.status.replace(/_/g, ' ')}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-1.5 text-right font-mono text-[11px]">{formatCurrency(trade.cost)}</TableCell>
                    <TableCell className={cn(
                      'py-1.5 text-right font-mono text-[11px]',
                      (trade.pnl ?? 0) >= 0 ? 'text-emerald-300' : 'text-red-300'
                    )}>
                      {trade.pnl == null ? '—' : formatSignedCurrency(trade.pnl)}
                    </TableCell>
                    <TableCell className="py-1.5 text-[11px] text-muted-foreground">
                      {trade.accountName || (trade.source === 'live' ? 'Orchestrator' : '—')}
                    </TableCell>
                  </TableRow>
                ))}
                {unifiedTrades.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7} className="py-6 text-center text-xs text-muted-foreground">
                      No trades for this view and range.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </ScrollArea>
        </div>
      </div>
    </div>
  )
}

function MetricChip({
  label,
  value,
  detail,
  icon: Icon,
  valueClassName,
}: {
  label: string
  value: string
  detail?: string
  icon: React.ElementType
  valueClassName?: string
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs" title={detail || undefined}>
      <Icon className="w-3.5 h-3.5 opacity-70" />
      <span className="text-muted-foreground">{label}</span>
      <span className={cn('font-mono font-semibold', valueClassName)}>{value}</span>
    </div>
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
