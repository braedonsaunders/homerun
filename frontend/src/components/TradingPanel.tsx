import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Play,
  Square,
  AlertTriangle,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  ShieldAlert,
  BarChart3,
  Briefcase,
  Award,
  Target,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  ExternalLink,
  Wallet,
  Radio,
  ChevronDown,
  ChevronUp,
  Eye,
  Crosshair,
  Flame,
  Shield,
  Cpu,
  Settings,
  Save,
  RotateCcw,
  Brain,
  Percent,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { Separator } from './ui/separator'
import {
  getAutoTraderStatus,
  startAutoTrader,
  stopAutoTrader,
  getAutoTraderTrades,
  resetCircuitBreaker,
  emergencyStopAutoTrader,
  updateAutoTraderConfig,
  getTradingStatus,
  getTradingPositions,
  getTradingBalance,
  getOrders,
  getSimulationAccounts,
} from '../services/api'
import type { AutoTraderConfig, TradingPosition } from '../services/api'

type DashboardTab = 'overview' | 'holdings' | 'orders' | 'settings'

// ==================== Activity Feed Event Types ====================

interface FeedEvent {
  id: string
  type: 'trade_executed' | 'trade_resolved' | 'opportunity_scanned' | 'circuit_breaker' | 'system' | 'position_update'
  timestamp: Date
  title: string
  detail: string
  value?: string
  valueColor?: string
  icon: 'trade' | 'win' | 'loss' | 'scan' | 'alert' | 'system' | 'position'
}

const ALL_STRATEGIES = [
  { key: 'basic', label: 'Basic Arb' },
  { key: 'negrisk', label: 'NegRisk' },
  { key: 'mutually_exclusive', label: 'Mutually Exclusive' },
  { key: 'contradiction', label: 'Contradiction' },
  { key: 'must_happen', label: 'Must-Happen' },
  { key: 'cross_platform', label: 'Cross-Platform Oracle' },
  { key: 'bayesian_cascade', label: 'Bayesian Cascade' },
  { key: 'liquidity_vacuum', label: 'Liquidity Vacuum' },
  { key: 'entropy_arb', label: 'Entropy Arbitrage' },
  { key: 'event_driven', label: 'Event-Driven' },
  { key: 'temporal_decay', label: 'Temporal Decay' },
  { key: 'correlation_arb', label: 'Correlation Arb' },
  { key: 'market_making', label: 'Market Making' },
  { key: 'stat_arb', label: 'Statistical Arb' },
]

export default function TradingPanel() {
  const [dashboardTab, setDashboardTab] = useState<DashboardTab>('overview')
  const [tradeFilter, setTradeFilter] = useState<string>('all')
  const [tradeSort, setTradeSort] = useState<'date' | 'pnl' | 'cost'>('date')
  const [tradeSortDir, setTradeSortDir] = useState<'asc' | 'desc'>('desc')
  const [feedEvents, setFeedEvents] = useState<FeedEvent[]>([])
  const [expandedPositions, setExpandedPositions] = useState<Set<number>>(new Set())
  const [showAccountPicker, setShowAccountPicker] = useState(false)
  const feedRef = useRef<HTMLDivElement>(null)
  const prevTradesRef = useRef<string[]>([])
  const prevStatsRef = useRef<{ seen: number; executed: number; skipped: number } | null>(null)
  const queryClient = useQueryClient()

  const { data: status, isLoading } = useQuery({
    queryKey: ['auto-trader-status'],
    queryFn: getAutoTraderStatus,
    refetchInterval: 3000,
  })

  const { data: trades = [] } = useQuery({
    queryKey: ['auto-trader-trades'],
    queryFn: () => getAutoTraderTrades(500),
    refetchInterval: 5000,
  })

  const { data: tradingStatus } = useQuery({
    queryKey: ['trading-status'],
    queryFn: getTradingStatus,
    refetchInterval: 10000,
  })

  const { data: livePositions = [] } = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    refetchInterval: 10000,
  })

  const { data: balance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
    enabled: !!tradingStatus?.initialized,
    retry: false,
  })

  useQuery({
    queryKey: ['trading-orders'],
    queryFn: () => getOrders(100),
    refetchInterval: 15000,
  })

  const { data: simulationAccounts = [] } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    enabled: showAccountPicker,
  })

  const startMutation = useMutation({
    mutationFn: ({ mode, accountId }: { mode: string; accountId?: string }) => startAutoTrader(mode, accountId),
    onSuccess: () => {
      setShowAccountPicker(false)
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      addFeedEvent({ type: 'system', title: 'Auto Trader Started', detail: 'Trading engine is now active', icon: 'system' })
    }
  })

  const stopMutation = useMutation({
    mutationFn: stopAutoTrader,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      addFeedEvent({ type: 'system', title: 'Auto Trader Stopped', detail: 'Trading engine has been stopped', icon: 'system' })
    }
  })

  const resetCircuitMutation = useMutation({
    mutationFn: resetCircuitBreaker,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      addFeedEvent({ type: 'system', title: 'Circuit Breaker Reset', detail: 'Protection has been manually reset', icon: 'alert' })
    }
  })

  const emergencyStopMutation = useMutation({
    mutationFn: emergencyStopAutoTrader,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      addFeedEvent({ type: 'system', title: 'EMERGENCY STOP', detail: 'All trading halted, orders cancelled', icon: 'alert' })
    }
  })

  const configMutation = useMutation({
    mutationFn: (updates: Partial<AutoTraderConfig>) => updateAutoTraderConfig(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      setConfigDirty(false)
      addFeedEvent({ type: 'system', title: 'Config Updated', detail: 'Auto trader settings saved', icon: 'system' })
    }
  })

  const [configDraft, setConfigDraft] = useState<Partial<AutoTraderConfig>>({})
  const [configDirty, setConfigDirty] = useState(false)

  // Sync draft from server config
  useEffect(() => {
    if (status?.config && !configDirty) {
      setConfigDraft(status.config)
    }
  }, [status?.config, configDirty])

  const updateDraft = (key: keyof AutoTraderConfig, value: unknown) => {
    setConfigDraft(prev => ({ ...prev, [key]: value }))
    setConfigDirty(true)
  }

  const saveDraft = () => {
    if (!configDirty || !status?.config) return
    // Only send changed fields
    const changes: Partial<AutoTraderConfig> = {}
    for (const [k, v] of Object.entries(configDraft)) {
      const key = k as keyof AutoTraderConfig
      if (JSON.stringify(v) !== JSON.stringify(status.config[key])) {
        (changes as Record<string, unknown>)[key] = v
      }
    }
    if (Object.keys(changes).length > 0) {
      configMutation.mutate(changes)
    } else {
      setConfigDirty(false)
    }
  }

  const resetDraft = () => {
    if (status?.config) {
      setConfigDraft(status.config)
      setConfigDirty(false)
    }
  }

  // Close account picker when clicking outside
  const accountPickerRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!showAccountPicker) return
    const handler = (e: MouseEvent) => {
      if (accountPickerRef.current && !accountPickerRef.current.contains(e.target as Node)) {
        setShowAccountPicker(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showAccountPicker])

  const stats = status?.stats
  const config = status?.config

  // ==================== Activity Feed Logic ====================

  const addFeedEvent = useCallback((event: Omit<FeedEvent, 'id' | 'timestamp'>) => {
    setFeedEvents(prev => [{
      ...event,
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      timestamp: new Date(),
    }, ...prev].slice(0, 200))
  }, [])

  // Generate feed events from trade changes
  useEffect(() => {
    if (!trades.length) return
    const currentIds = trades.map(t => t.id)
    const prevIds = prevTradesRef.current

    if (prevIds.length > 0) {
      const newTrades = trades.filter(t => !prevIds.includes(t.id))
      newTrades.forEach(t => {
        addFeedEvent({
          type: 'trade_executed',
          title: `Trade Executed: ${t.strategy}`,
          detail: `${t.mode.toUpperCase()} | Cost: $${t.total_cost.toFixed(2)} | Expected: +$${t.expected_profit.toFixed(2)}`,
          value: `$${t.total_cost.toFixed(2)}`,
          valueColor: 'text-blue-400',
          icon: 'trade',
        })
      })

      // Check for resolved trades
      trades.forEach(t => {
        const prevTrade = prevIds.includes(t.id)
        if (prevTrade && t.actual_profit !== null) {
          const isWin = (t.actual_profit || 0) > 0
          if (isWin) {
            addFeedEvent({
              type: 'trade_resolved',
              title: `Trade Won: ${t.strategy}`,
              detail: `Resolved with profit`,
              value: `+$${(t.actual_profit || 0).toFixed(2)}`,
              valueColor: 'text-green-400',
              icon: 'win',
            })
          } else {
            addFeedEvent({
              type: 'trade_resolved',
              title: `Trade Lost: ${t.strategy}`,
              detail: `Resolved with loss`,
              value: `-$${Math.abs(t.actual_profit || 0).toFixed(2)}`,
              valueColor: 'text-red-400',
              icon: 'loss',
            })
          }
        }
      })
    }
    prevTradesRef.current = currentIds
  }, [trades, addFeedEvent])

  // Generate scan events from status changes
  useEffect(() => {
    if (!stats) return
    const prev = prevStatsRef.current
    if (prev) {
      const newSeen = stats.opportunities_seen - prev.seen
      const newExecuted = stats.opportunities_executed - prev.executed
      const newSkipped = stats.opportunities_skipped - prev.skipped
      if (newSeen > 0) {
        addFeedEvent({
          type: 'opportunity_scanned',
          title: `Scan Complete`,
          detail: `${newSeen} new opportunities | ${newExecuted} executed | ${newSkipped} skipped`,
          icon: 'scan',
        })
      }
      if (stats.circuit_breaker_active && !prev) {
        addFeedEvent({
          type: 'circuit_breaker',
          title: 'Circuit Breaker Triggered',
          detail: `${stats.consecutive_losses} consecutive losses detected`,
          icon: 'alert',
        })
      }
    }
    prevStatsRef.current = { seen: stats.opportunities_seen, executed: stats.opportunities_executed, skipped: stats.opportunities_skipped }
  }, [stats, addFeedEvent])

  // Seed feed with recent trades on first load
  useEffect(() => {
    if (trades.length > 0 && feedEvents.length === 0) {
      const recentTrades = trades.slice(0, 15).reverse()
      const seedEvents: FeedEvent[] = recentTrades.map(t => ({
        id: `seed-${t.id}`,
        timestamp: new Date(t.executed_at),
        type: t.actual_profit !== null
          ? 'trade_resolved' as const
          : 'trade_executed' as const,
        title: t.actual_profit !== null
          ? `${(t.actual_profit || 0) >= 0 ? 'Won' : 'Lost'}: ${t.strategy}`
          : `Executed: ${t.strategy}`,
        detail: `${t.mode.toUpperCase()} | Cost: $${t.total_cost.toFixed(2)}`,
        value: t.actual_profit !== null
          ? `${(t.actual_profit || 0) >= 0 ? '+' : ''}$${(t.actual_profit || 0).toFixed(2)}`
          : `$${t.total_cost.toFixed(2)}`,
        valueColor: t.actual_profit !== null
          ? (t.actual_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'
          : 'text-blue-400',
        icon: t.actual_profit !== null
          ? ((t.actual_profit || 0) >= 0 ? 'win' as const : 'loss' as const)
          : 'trade' as const,
      }))
      setFeedEvents(seedEvents.reverse())
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trades.length])

  // ==================== Performance Metrics ====================

  const performanceMetrics = useMemo(() => {
    if (!trades.length) return null
    const resolved = trades.filter(t => t.actual_profit !== null)
    const wins = resolved.filter(t => (t.actual_profit || 0) > 0)
    const losses = resolved.filter(t => (t.actual_profit || 0) < 0)
    const totalPnl = resolved.reduce((s, t) => s + (t.actual_profit || 0), 0)
    const totalCost = trades.reduce((s, t) => s + t.total_cost, 0)
    const totalGains = wins.reduce((s, t) => s + (t.actual_profit || 0), 0)
    const totalLosses = Math.abs(losses.reduce((s, t) => s + (t.actual_profit || 0), 0))
    const profitFactor = totalLosses > 0 ? totalGains / totalLosses : 0
    const avgWin = wins.length > 0 ? totalGains / wins.length : 0
    const avgLoss = losses.length > 0 ? totalLosses / losses.length : 0
    const bestTrade = resolved.length > 0 ? Math.max(...resolved.map(t => t.actual_profit || 0)) : 0
    const worstTrade = resolved.length > 0 ? Math.min(...resolved.map(t => t.actual_profit || 0)) : 0

    let peak = 0, maxDD = 0, cumPnl = 0
    const sortedTrades = [...resolved].sort((a, b) => new Date(a.executed_at).getTime() - new Date(b.executed_at).getTime())
    sortedTrades.forEach(t => {
      cumPnl += t.actual_profit || 0
      if (cumPnl > peak) peak = cumPnl
      const dd = peak - cumPnl
      if (dd > maxDD) maxDD = dd
    })

    const byStrategy: Record<string, { trades: number; pnl: number; wins: number; losses: number; cost: number }> = {}
    trades.forEach(t => {
      if (!byStrategy[t.strategy]) byStrategy[t.strategy] = { trades: 0, pnl: 0, wins: 0, losses: 0, cost: 0 }
      byStrategy[t.strategy].trades++
      byStrategy[t.strategy].pnl += t.actual_profit || 0
      byStrategy[t.strategy].cost += t.total_cost
      if ((t.actual_profit || 0) > 0) byStrategy[t.strategy].wins++
      if ((t.actual_profit || 0) < 0) byStrategy[t.strategy].losses++
    })

    const equityPoints: { date: string; equity: number }[] = []
    let equity = 0
    sortedTrades.forEach(t => {
      equity += t.actual_profit || 0
      equityPoints.push({ date: t.executed_at, equity })
    })

    return { totalPnl, totalCost, profitFactor, avgWin, avgLoss, bestTrade, worstTrade, maxDrawdown: maxDD, byStrategy, equityPoints, winCount: wins.length, lossCount: losses.length }
  }, [trades])

  // Filtered/sorted trades
  const processedTrades = useMemo(() => {
    let filtered = [...trades]
    if (tradeFilter !== 'all') {
      if (tradeFilter === 'wins') filtered = filtered.filter(t => (t.actual_profit || 0) > 0)
      else if (tradeFilter === 'losses') filtered = filtered.filter(t => (t.actual_profit || 0) < 0)
      else if (tradeFilter === 'open') filtered = filtered.filter(t => t.status === 'open' || t.status === 'pending')
      else filtered = filtered.filter(t => t.status === tradeFilter)
    }
    filtered.sort((a, b) => {
      let cmp = 0
      if (tradeSort === 'date') cmp = new Date(a.executed_at).getTime() - new Date(b.executed_at).getTime()
      else if (tradeSort === 'pnl') cmp = (a.actual_profit || 0) - (b.actual_profit || 0)
      else if (tradeSort === 'cost') cmp = a.total_cost - b.total_cost
      return tradeSortDir === 'desc' ? -cmp : cmp
    })
    return filtered
  }, [trades, tradeFilter, tradeSort, tradeSortDir])

  const positionsTotalValue = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.current_price, 0)
  const positionsCostBasis = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.average_cost, 0)
  const positionsUnrealizedPnl = livePositions.reduce((s: number, p: TradingPosition) => s + p.unrealized_pnl, 0)

  const togglePosition = (idx: number) => {
    setExpandedPositions(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <div className="w-12 h-12 rounded-full border-2 border-border border-t-green-500 animate-spin" />
            <Cpu className="w-5 h-5 text-green-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="text-sm text-muted-foreground">Initializing trading engine...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">

      {/* ==================== Top Status Bar ==================== */}
      <div className="shrink-0 flex items-center justify-between px-3 py-2 bg-card/40 border border-border/40 rounded-xl mb-2">
        {/* Left: Mode indicator + circuit breaker */}
        <div className="flex items-center gap-3 shrink-0">
          <div className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all",
            status?.running
              ? status?.config.mode === 'live'
                ? "bg-green-500/5 border-green-500/30"
                : "bg-blue-500/5 border-blue-500/30"
              : "bg-muted/50 border-border/40"
          )}>
            <div className="relative">
              <div className={cn(
                "w-2 h-2 rounded-full",
                status?.running
                  ? status?.config.mode === 'live' ? "bg-green-500" : "bg-blue-500"
                  : "bg-gray-500"
              )} />
              {status?.running && (
                <div className={cn(
                  "absolute inset-0 w-2 h-2 rounded-full animate-ping",
                  status?.config.mode === 'live' ? "bg-green-500/40" : "bg-blue-500/40"
                )} />
              )}
            </div>
            <span className="text-xs font-semibold leading-none whitespace-nowrap">
              {status?.running
                ? `${status?.config.mode?.toUpperCase()} MODE`
                : 'OFFLINE'
              }
            </span>
            <span className="text-[10px] text-muted-foreground font-mono whitespace-nowrap">
              {stats?.opportunities_seen || 0} scanned
            </span>
          </div>

          {/* Circuit Breaker Warning */}
          {stats?.circuit_breaker_active && (
            <div className="flex items-center gap-2 px-2 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded-lg animate-pulse">
              <AlertTriangle className="w-3.5 h-3.5 text-yellow-500" />
              <span className="text-yellow-500 text-[10px] font-medium whitespace-nowrap">CIRCUIT BREAKER</span>
              <Button
                variant="ghost"
                onClick={() => resetCircuitMutation.mutate()}
                className="px-2 py-0.5 bg-yellow-500/20 hover:bg-yellow-500/30 rounded text-[10px] text-yellow-400 font-medium h-auto"
              >
                Reset
              </Button>
            </div>
          )}
        </div>

        {/* Center: Key metrics inline */}
        <div className="flex items-center gap-4 mx-4">
          <div className="flex items-center gap-1.5">
            <DollarSign className="w-3 h-3 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wide">P&L</span>
            <span className={cn("text-xs font-mono font-bold", (stats?.total_profit || 0) >= 0 ? "text-green-400" : "text-red-400")}>
              {(stats?.total_profit || 0) >= 0 ? '+' : ''}${(stats?.total_profit || 0).toFixed(2)}
            </span>
          </div>
          <div className="w-px h-4 bg-border/40" />
          <div className="flex items-center gap-1.5">
            <Award className="w-3 h-3 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wide">Win</span>
            <span className={cn("text-xs font-mono font-bold", (stats?.win_rate || 0) >= 0.5 ? "text-green-400" : (stats?.win_rate || 0) > 0 ? "text-yellow-400" : "text-muted-foreground")}>
              {((stats?.win_rate || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-px h-4 bg-border/40" />
          <div className="flex items-center gap-1.5">
            <Target className="w-3 h-3 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wide">ROI</span>
            <span className={cn("text-xs font-mono font-bold", (stats?.roi_percent || 0) >= 0 ? "text-green-400" : "text-red-400")}>
              {(stats?.roi_percent || 0) >= 0 ? '+' : ''}{(stats?.roi_percent || 0).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-2 shrink-0">
          {status?.running ? (
            <Button
              variant="secondary"
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              className="gap-1.5 h-auto px-3 py-1.5 text-xs"
            >
              <Square className="w-3 h-3" />
              Stop
            </Button>
          ) : (
            <>
              <div className="relative" ref={accountPickerRef}>
                <Button
                  variant="ghost"
                  onClick={() => setShowAccountPicker(prev => !prev)}
                  disabled={startMutation.isPending}
                  className="gap-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg h-auto px-3 py-1.5 text-xs font-medium border border-blue-500/20"
                >
                  <Play className="w-3 h-3" />
                  Paper
                  <ChevronDown className="w-2.5 h-2.5" />
                </Button>
                {showAccountPicker && (
                  <div className="absolute top-full mt-1 right-0 z-50 w-64 bg-popover border border-border rounded-lg shadow-xl overflow-hidden">
                    <div className="px-3 py-2 border-b border-border">
                      <span className="text-xs font-medium text-muted-foreground">Select Paper Account</span>
                    </div>
                    <div className="max-h-48 overflow-y-auto">
                      {simulationAccounts.map(acc => (
                        <button
                          key={acc.id}
                          onClick={() => startMutation.mutate({ mode: 'paper', accountId: acc.id })}
                          className="w-full text-left px-3 py-2 hover:bg-accent transition-colors border-b border-border/50 last:border-b-0"
                        >
                          <div className="text-sm text-foreground">{acc.name}</div>
                          <div className="text-xs text-muted-foreground">${acc.current_capital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} capital</div>
                        </button>
                      ))}
                    </div>
                    <button
                      onClick={() => startMutation.mutate({ mode: 'paper' })}
                      className="w-full text-left px-3 py-2 hover:bg-blue-500/10 transition-colors border-t border-border text-blue-400 text-sm"
                    >
                      + New Account
                    </button>
                  </div>
                )}
              </div>
              <Button
                variant="ghost"
                onClick={() => {
                  if (confirm('Enable LIVE trading? This will use REAL MONEY.')) {
                    startMutation.mutate({ mode: 'live' })
                  }
                }}
                disabled={startMutation.isPending}
                className="gap-1.5 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded-lg h-auto px-3 py-1.5 text-xs font-medium border border-green-500/20"
              >
                <Zap className="w-3 h-3" />
                Live
              </Button>
            </>
          )}

          {/* Emergency Stop */}
          <Button
            variant="ghost"
            onClick={() => {
              if (confirm('EMERGENCY STOP - Cancel all orders and stop trading?')) {
                emergencyStopMutation.mutate()
              }
            }}
            className="gap-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg h-auto px-2.5 py-1.5 text-xs border border-red-500/20"
          >
            <ShieldAlert className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {/* ==================== Three-Column Grid ==================== */}
      <div className="grid grid-cols-12 gap-2 flex-1 min-h-0">

        {/* ==================== LEFT COLUMN: Live Activity Feed (4 cols) ==================== */}
        <div className="col-span-12 lg:col-span-4 flex flex-col min-h-0">
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden flex flex-col flex-1">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border/40 shrink-0">
              <div className="flex items-center gap-2">
                <Radio className={cn("w-3.5 h-3.5", status?.running ? "text-green-500 animate-pulse" : "text-muted-foreground")} />
                <h3 className="text-[10px] uppercase tracking-widest font-semibold">Live Activity</h3>
                <span className="text-[10px] text-muted-foreground font-mono">{feedEvents.length}</span>
              </div>
              <div className="flex items-center gap-2">
                {status?.running && (
                  <span className="flex items-center gap-1 text-[10px] text-green-500/70 font-mono">
                    <span className="w-1 h-1 rounded-full bg-green-500 animate-pulse" />
                    LIVE
                  </span>
                )}
              </div>
            </div>

            <CardContent className="p-0 flex-1 min-h-0">
              <div ref={feedRef} className="h-full max-h-[calc(100vh-180px)] overflow-y-auto scrollbar-thin">
                {feedEvents.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-muted-foreground py-16">
                    <Activity className="w-8 h-8 mb-2 opacity-30" />
                    <p className="text-sm">Waiting for activity...</p>
                    <p className="text-xs text-gray-700 mt-1">Start the auto trader to see live events</p>
                  </div>
                ) : (
                  <div className="divide-y divide-border/30">
                    {feedEvents.map((event, idx) => (
                      <FeedEventRow key={event.id} event={event} isNew={idx === 0} />
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ==================== CENTER COLUMN: Tab Content (5 cols) ==================== */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-2 min-h-0">
          {/* Sub-tabs */}
          <Tabs value={dashboardTab} onValueChange={(v) => setDashboardTab(v as DashboardTab)} className="w-full shrink-0">
            <TabsList className="bg-card/40 rounded-xl p-0.5 border border-border/40 h-auto w-full justify-start">
              {([
                { key: 'overview', label: 'Performance', icon: <BarChart3 className="w-3 h-3" /> },
                { key: 'holdings', label: 'Holdings', icon: <Briefcase className="w-3 h-3" /> },
                { key: 'orders', label: 'Trades', icon: <Activity className="w-3 h-3" /> },
                { key: 'settings', label: 'Settings', icon: <Settings className="w-3 h-3" /> },
              ] as { key: DashboardTab; label: string; icon: React.ReactNode }[]).map(tab => (
                <TabsTrigger
                  key={tab.key}
                  value={tab.key}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium data-[state=active]:bg-green-500/15 data-[state=active]:text-green-400 data-[state=active]:shadow-sm data-[state=active]:shadow-green-500/5 text-muted-foreground"
                >
                  {tab.icon}
                  {tab.label}
                  {tab.key === 'holdings' && livePositions.length > 0 && (
                    <Badge className="ml-0.5 px-1 py-0 bg-green-500/20 text-green-400 rounded-full text-[9px] font-mono border-0">
                      {livePositions.length}
                    </Badge>
                  )}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>

          {/* Tab Content Area */}
          <div className="flex-1 overflow-y-auto min-h-0 scrollbar-thin">
            {dashboardTab === 'overview' && (
              <div className="space-y-2">
                {/* Equity Curve */}
                <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5">
                      <BarChart3 className="w-3.5 h-3.5 text-green-500" />
                      Cumulative P&L
                    </h4>
                    {performanceMetrics && (
                      <span className={cn(
                        "text-xs font-mono font-bold",
                        performanceMetrics.totalPnl >= 0 ? "text-green-400" : "text-red-400"
                      )}>
                        {performanceMetrics.totalPnl >= 0 ? '+' : ''}${performanceMetrics.totalPnl.toFixed(2)}
                      </span>
                    )}
                  </div>
                  {performanceMetrics && performanceMetrics.equityPoints.length > 1 ? (
                    <div className="h-44">
                      <PnlChart points={performanceMetrics.equityPoints} />
                    </div>
                  ) : (
                    <div className="text-center py-10 text-muted-foreground">
                      <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-20" />
                      <p className="text-xs">Start trading to see your equity curve</p>
                    </div>
                  )}
                </Card>

                {/* Strategy Performance */}
                {performanceMetrics && Object.keys(performanceMetrics.byStrategy).length > 0 && (
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                    <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-2">
                      <PieChart className="w-3.5 h-3.5 text-indigo-500" />
                      Strategy Breakdown
                    </h4>
                    <div className="space-y-1.5">
                      {Object.entries(performanceMetrics.byStrategy)
                        .sort((a, b) => b[1].pnl - a[1].pnl)
                        .map(([strategy, data]) => {
                          const winRate = (data.wins + data.losses) > 0 ? (data.wins / (data.wins + data.losses)) * 100 : 0
                          const maxPnl = Math.max(...Object.values(performanceMetrics.byStrategy).map(d => Math.abs(d.pnl)), 1)
                          const barWidth = Math.abs(data.pnl) / maxPnl * 100
                          return (
                            <div key={strategy} className="group relative bg-card/40 rounded-lg p-2.5 overflow-hidden">
                              {/* Background bar */}
                              <div
                                className={cn(
                                  "absolute inset-y-0 left-0 opacity-[0.07] transition-all",
                                  data.pnl >= 0 ? "bg-green-500" : "bg-red-500"
                                )}
                                style={{ width: `${barWidth}%` }}
                              />
                              <div className="relative flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <div className={cn("w-1 h-6 rounded-full", data.pnl >= 0 ? "bg-green-500" : "bg-red-500")} />
                                  <div>
                                    <p className="font-medium text-xs">{strategy}</p>
                                    <p className="text-[10px] text-muted-foreground">
                                      {data.trades} trades | {winRate.toFixed(0)}% WR | ${data.cost.toFixed(2)} inv
                                    </p>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <p className={cn("font-mono font-semibold text-xs", data.pnl >= 0 ? "text-green-400" : "text-red-400")}>
                                    {data.pnl >= 0 ? '+' : ''}${data.pnl.toFixed(2)}
                                  </p>
                                  <p className="text-[10px] text-muted-foreground font-mono">{data.wins}W / {data.losses}L</p>
                                </div>
                              </div>
                            </div>
                          )
                        })}
                    </div>
                  </Card>
                )}

                {/* Best/Worst + Daily */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                  {performanceMetrics && (
                    <>
                      <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Best Trade</p>
                        <p className="text-sm font-mono font-bold text-green-400">+${performanceMetrics.bestTrade.toFixed(2)}</p>
                      </Card>
                      <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Worst Trade</p>
                        <p className="text-sm font-mono font-bold text-red-400">${performanceMetrics.worstTrade.toFixed(2)}</p>
                      </Card>
                    </>
                  )}
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Executed Today</p>
                    <p className="text-sm font-mono font-bold">{stats?.opportunities_executed || 0}</p>
                  </Card>
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Skipped Today</p>
                    <p className="text-sm font-mono font-bold text-muted-foreground">{stats?.opportunities_skipped || 0}</p>
                  </Card>
                </div>
              </div>
            )}

            {dashboardTab === 'holdings' && (
              <div className="space-y-2">
                {/* Wallet Info */}
                {tradingStatus && (
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Wallet className="w-4 h-4 text-purple-400" />
                        <div>
                          <p className="text-xs font-medium">Trading Wallet</p>
                          <p className="text-[10px] text-muted-foreground font-mono">
                            {tradingStatus.wallet_address
                              ? `${tradingStatus.wallet_address.slice(0, 10)}...${tradingStatus.wallet_address.slice(-8)}`
                              : 'Not connected'}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-[10px] text-muted-foreground uppercase">USDC</p>
                          <p className="font-mono font-bold text-sm">${balance?.balance?.toFixed(2) || '0.00'}</p>
                        </div>
                        <Badge className={cn(
                          "rounded-lg text-[10px] font-medium border-0",
                          tradingStatus.initialized ? "bg-green-500/15 text-green-400" : "bg-gray-500/15 text-muted-foreground"
                        )}>
                          {tradingStatus.initialized ? 'Connected' : 'Offline'}
                        </Badge>
                      </div>
                    </div>
                  </Card>
                )}

                {/* Holdings Summary */}
                <div className="grid grid-cols-3 gap-2">
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Positions</p>
                    <p className="text-lg font-mono font-bold">{livePositions.length}</p>
                  </Card>
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Cost Basis</p>
                    <p className="text-lg font-mono font-bold">${positionsCostBasis.toFixed(2)}</p>
                  </Card>
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-2.5">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-0.5">Market Value</p>
                    <p className="text-lg font-mono font-bold">${positionsTotalValue.toFixed(2)}</p>
                  </Card>
                </div>

                {/* Positions */}
                {livePositions.length === 0 ? (
                  <Card className="text-center py-10 bg-card/40 border-border/40 rounded-xl shadow-none">
                    <Briefcase className="w-8 h-8 text-gray-700 mx-auto mb-2" />
                    <p className="text-muted-foreground text-xs">No open positions</p>
                    <p className="text-[10px] text-gray-700">Start trading to see holdings</p>
                  </Card>
                ) : (
                  <div className="space-y-1.5">
                    {livePositions.map((pos: TradingPosition, idx: number) => {
                      const costBasis = pos.size * pos.average_cost
                      const mktValue = pos.size * pos.current_price
                      const pnlPct = costBasis > 0 ? (pos.unrealized_pnl / costBasis) * 100 : 0
                      const isExpanded = expandedPositions.has(idx)
                      return (
                        <Card key={idx} className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden transition-all">
                          <Button
                            variant="ghost"
                            onClick={() => togglePosition(idx)}
                            className="w-full flex items-center justify-between p-3 h-auto rounded-none hover:bg-card/60"
                          >
                            <div className="flex items-center gap-2.5 min-w-0">
                              <Badge className={cn(
                                "w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-bold shrink-0 border-0",
                                pos.outcome.toLowerCase() === 'yes'
                                  ? "bg-green-500/15 text-green-400"
                                  : "bg-red-500/15 text-red-400"
                              )}>
                                {pos.outcome.toUpperCase().slice(0, 1)}
                              </Badge>
                              <div className="text-left min-w-0">
                                <p className="text-xs font-medium truncate">{pos.market_question}</p>
                                <p className="text-[10px] text-muted-foreground font-mono">
                                  {pos.size.toFixed(2)} shares @ ${pos.average_cost.toFixed(4)}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-3 shrink-0">
                              <div className="text-right">
                                <p className={cn("font-mono font-semibold text-xs", pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                                  {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                                </p>
                                <p className={cn("text-[10px] font-mono", pnlPct >= 0 ? "text-green-400/60" : "text-red-400/60")}>
                                  {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
                                </p>
                              </div>
                              {isExpanded ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
                            </div>
                          </Button>
                          {isExpanded && (
                            <div className="px-3 pb-2.5 pt-0 border-t border-border/40">
                              <div className="grid grid-cols-4 gap-2 pt-2.5">
                                <div>
                                  <p className="text-[10px] text-muted-foreground">Current Price</p>
                                  <p className="font-mono text-xs">${pos.current_price.toFixed(4)}</p>
                                </div>
                                <div>
                                  <p className="text-[10px] text-muted-foreground">Cost Basis</p>
                                  <p className="font-mono text-xs">${costBasis.toFixed(2)}</p>
                                </div>
                                <div>
                                  <p className="text-[10px] text-muted-foreground">Market Value</p>
                                  <p className="font-mono text-xs">${mktValue.toFixed(2)}</p>
                                </div>
                                <div>
                                  {pos.market_id && (
                                    <a
                                      href={`https://polymarket.com/event/${pos.market_id}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 mt-2"
                                    >
                                      View Market <ExternalLink className="w-2.5 h-2.5" />
                                    </a>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </Card>
                      )
                    })}

                    {/* Totals */}
                    <Card className="flex items-center justify-between px-3 py-2.5 bg-card/40 border-border/40 rounded-xl shadow-none">
                      <span className="text-xs text-muted-foreground font-medium">Total Unrealized P&L</span>
                      <span className={cn("font-mono font-bold text-sm", positionsUnrealizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
                        {positionsUnrealizedPnl >= 0 ? '+' : ''}${positionsUnrealizedPnl.toFixed(2)}
                      </span>
                    </Card>
                  </div>
                )}
              </div>
            )}

            {dashboardTab === 'orders' && (
              <div className="space-y-2">
                {/* Filters */}
                <div className="flex items-center gap-2">
                  <select
                    value={tradeFilter}
                    onChange={(e) => setTradeFilter(e.target.value)}
                    className="bg-card/40 border border-border/40 rounded-lg px-2.5 py-1.5 text-xs"
                  >
                    <option value="all">All Trades</option>
                    <option value="open">Open</option>
                    <option value="wins">Wins</option>
                    <option value="losses">Losses</option>
                  </select>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    {(['date', 'pnl', 'cost'] as const).map(s => (
                      <Button
                        key={s}
                        variant="ghost"
                        onClick={() => {
                          if (tradeSort === s) setTradeSortDir(d => d === 'desc' ? 'asc' : 'desc')
                          else { setTradeSort(s); setTradeSortDir('desc') }
                        }}
                        className={cn(
                          "px-2 py-1 h-auto rounded-md text-[10px]",
                          tradeSort === s ? "bg-green-500/15 text-green-400" : "hover:bg-muted"
                        )}
                      >
                        {s.charAt(0).toUpperCase() + s.slice(1)}
                        {tradeSort === s && (tradeSortDir === 'desc' ? <ChevronDown className="w-2.5 h-2.5 inline ml-0.5" /> : <ChevronUp className="w-2.5 h-2.5 inline ml-0.5" />)}
                      </Button>
                    ))}
                  </div>
                  <span className="text-[10px] text-muted-foreground ml-auto font-mono">{processedTrades.length} trades</span>
                </div>

                {/* Trades */}
                {processedTrades.length === 0 ? (
                  <Card className="text-center py-10 bg-card/40 border-border/40 rounded-xl shadow-none">
                    <p className="text-muted-foreground text-xs">No trades found</p>
                  </Card>
                ) : (
                  <Card className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden">
                    <CardContent className="p-0">
                      <div className="max-h-[500px] overflow-y-auto">
                        <table className="w-full text-xs">
                          <thead className="sticky top-0 bg-card z-10">
                            <tr className="border-b border-border/40 text-muted-foreground text-[10px] uppercase tracking-widest">
                              <th className="text-left px-3 py-2">Date</th>
                              <th className="text-left px-2 py-2">Strategy</th>
                              <th className="text-center px-2 py-2">Mode</th>
                              <th className="text-right px-2 py-2">Cost</th>
                              <th className="text-center px-2 py-2">Status</th>
                              <th className="text-right px-3 py-2">P&L</th>
                            </tr>
                          </thead>
                          <tbody>
                            {processedTrades.map((trade) => (
                              <tr key={trade.id} className="border-b border-border/30 hover:bg-card/60 transition-colors">
                                <td className="px-3 py-2">
                                  <p className="font-mono text-[10px]">{new Date(trade.executed_at).toLocaleDateString()}</p>
                                  <p className="font-mono text-[9px] text-muted-foreground">{new Date(trade.executed_at).toLocaleTimeString()}</p>
                                </td>
                                <td className="px-2 py-2 font-medium text-xs">{trade.strategy}</td>
                                <td className="text-center px-2 py-2">
                                  <Badge className={cn(
                                    "rounded-md text-[9px] font-semibold border-0",
                                    trade.mode === 'live' ? "bg-green-500/15 text-green-400" :
                                    trade.mode === 'paper' ? "bg-blue-500/15 text-blue-400" : "bg-gray-500/15 text-muted-foreground"
                                  )}>
                                    {trade.mode.toUpperCase()}
                                  </Badge>
                                </td>
                                <td className="text-right px-2 py-2 font-mono text-xs">${trade.total_cost.toFixed(2)}</td>
                                <td className="text-center px-2 py-2">
                                  <StatusBadge status={trade.status} />
                                </td>
                                <td className="text-right px-3 py-2">
                                  {trade.actual_profit !== null ? (
                                    <span className={cn("font-mono font-semibold text-xs", (trade.actual_profit || 0) >= 0 ? "text-green-400" : "text-red-400")}>
                                      {(trade.actual_profit || 0) >= 0 ? '+' : ''}${(trade.actual_profit || 0).toFixed(2)}
                                    </span>
                                  ) : (
                                    <span className="text-muted-foreground font-mono text-[10px]">+${trade.expected_profit.toFixed(2)} exp</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}

            {/* Settings tab: show empty center when settings overlay is active */}
            {dashboardTab === 'settings' && (
              <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                <Settings className="w-8 h-8 mb-2 opacity-20" />
                <p className="text-xs">Settings panel is open</p>
              </div>
            )}
          </div>
        </div>

        {/* ==================== RIGHT COLUMN: Metrics Sidebar (3 cols) ==================== */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-2 min-h-0 overflow-y-auto scrollbar-thin">

          {/* Key Stats */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-0.5">
            <h4 className="text-[10px] text-muted-foreground uppercase tracking-widest mb-2 flex items-center gap-1.5">
              <Crosshair className="w-3 h-3" /> Key Metrics
            </h4>

            <MetricRow
              label="Total P&L"
              value={`${(stats?.total_profit || 0) >= 0 ? '+' : ''}$${(stats?.total_profit || 0).toFixed(2)}`}
              valueColor={(stats?.total_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
              icon={<DollarSign className="w-3 h-3" />}
              sparkData={performanceMetrics?.equityPoints.slice(-20).map(p => p.equity)}
            />
            <MetricRow
              label="Daily P&L"
              value={`${(stats?.daily_profit || 0) >= 0 ? '+' : ''}$${(stats?.daily_profit || 0).toFixed(2)}`}
              valueColor={(stats?.daily_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
              icon={<TrendingUp className="w-3 h-3" />}
              sub={`${stats?.daily_trades || 0} today`}
            />
            <MetricRow
              label="Win Rate"
              value={`${((stats?.win_rate || 0) * 100).toFixed(1)}%`}
              icon={<Award className="w-3 h-3" />}
              sub={`${stats?.winning_trades || 0}W / ${stats?.losing_trades || 0}L`}
              valueColor={(stats?.win_rate || 0) >= 0.5 ? 'text-green-400' : (stats?.win_rate || 0) > 0 ? 'text-yellow-400' : 'text-muted-foreground'}
            />
            <MetricRow
              label="Total Trades"
              value={stats?.total_trades?.toString() || '0'}
              icon={<Activity className="w-3 h-3" />}
            />
            <MetricRow
              label="ROI"
              value={`${(stats?.roi_percent || 0) >= 0 ? '+' : ''}${(stats?.roi_percent || 0).toFixed(2)}%`}
              valueColor={(stats?.roi_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
              icon={<Target className="w-3 h-3" />}
            />
            <MetricRow
              label="Total Invested"
              value={`$${(stats?.total_invested || 0).toFixed(2)}`}
              icon={<Briefcase className="w-3 h-3" />}
            />
          </Card>

          {/* Advanced Metrics */}
          {performanceMetrics && (
            <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-0.5">
              <h4 className="text-[10px] text-muted-foreground uppercase tracking-widest mb-2 flex items-center gap-1.5">
                <Flame className="w-3 h-3" /> Advanced
              </h4>
              <MetricRow
                label="Profit Factor"
                value={performanceMetrics.profitFactor > 0 ? performanceMetrics.profitFactor.toFixed(2) : 'N/A'}
                icon={<PieChart className="w-3 h-3" />}
                valueColor={performanceMetrics.profitFactor >= 1.5 ? 'text-green-400' : performanceMetrics.profitFactor >= 1 ? 'text-yellow-400' : 'text-red-400'}
              />
              <MetricRow
                label="Max Drawdown"
                value={`$${performanceMetrics.maxDrawdown.toFixed(2)}`}
                icon={<AlertTriangle className="w-3 h-3" />}
                valueColor="text-orange-400"
              />
              <MetricRow
                label="Avg Win"
                value={`$${performanceMetrics.avgWin.toFixed(2)}`}
                icon={<ArrowUpRight className="w-3 h-3" />}
                valueColor="text-green-400"
              />
              <MetricRow
                label="Avg Loss"
                value={`$${performanceMetrics.avgLoss.toFixed(2)}`}
                icon={<ArrowDownRight className="w-3 h-3" />}
                valueColor="text-red-400"
              />
              <MetricRow
                label="Positions Value"
                value={`$${positionsTotalValue.toFixed(2)}`}
                icon={<Eye className="w-3 h-3" />}
              />
              <MetricRow
                label="Unrealized P&L"
                value={`${positionsUnrealizedPnl >= 0 ? '+' : ''}$${positionsUnrealizedPnl.toFixed(2)}`}
                valueColor={positionsUnrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}
                icon={positionsUnrealizedPnl >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
              />
            </Card>
          )}

          {/* System Health */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] text-muted-foreground uppercase tracking-widest mb-2 flex items-center gap-1.5">
              <Shield className="w-3 h-3" /> System
            </h4>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Engine</span>
                <span className={cn("text-[10px] font-mono font-medium", status?.running ? "text-green-400" : "text-muted-foreground")}>
                  {status?.running ? 'ACTIVE' : 'STOPPED'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Mode</span>
                <span className={cn("text-[10px] font-mono font-medium",
                  config?.mode === 'live' ? "text-green-400" :
                  config?.mode === 'paper' ? "text-blue-400" : "text-muted-foreground"
                )}>
                  {config?.mode?.toUpperCase() || 'DISABLED'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Circuit Breaker</span>
                <span className={cn("text-[10px] font-mono font-medium",
                  stats?.circuit_breaker_active ? "text-yellow-400" : "text-muted-foreground"
                )}>
                  {stats?.circuit_breaker_active ? 'TRIGGERED' : 'OK'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Consecutive Losses</span>
                <span className={cn("text-[10px] font-mono font-medium",
                  (stats?.consecutive_losses || 0) >= 3 ? "text-orange-400" : "text-muted-foreground"
                )}>
                  {stats?.consecutive_losses || 0} / {config?.circuit_breaker_losses || '?'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Last Trade</span>
                <span className="text-[10px] font-mono text-muted-foreground">
                  {stats?.last_trade_at ? timeAgo(new Date(stats.last_trade_at)) : 'Never'}
                </span>
              </div>
              {tradingStatus && (
                <>
                  <Separator className="my-1.5" />
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground">Wallet</span>
                    <span className={cn("text-[10px] font-mono font-medium",
                      tradingStatus.initialized ? "text-green-400" : "text-muted-foreground"
                    )}>
                      {tradingStatus.initialized ? 'CONNECTED' : 'OFFLINE'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground">Balance</span>
                    <span className="text-[10px] font-mono text-muted-foreground">
                      ${balance?.balance?.toFixed(2) || '0.00'} USDC
                    </span>
                  </div>
                </>
              )}
            </div>
          </Card>

          {/* Live Positions mini-list (always visible) */}
          {livePositions.length > 0 && dashboardTab !== 'holdings' && (
            <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
              <h4 className="text-[10px] text-muted-foreground uppercase tracking-widest mb-2 flex items-center gap-1.5">
                <Briefcase className="w-3 h-3" /> Open Positions
                <span className="ml-auto text-green-400 font-mono">{livePositions.length}</span>
              </h4>
              <div className="space-y-1.5">
                {livePositions.slice(0, 5).map((pos: TradingPosition, idx: number) => (
                  <div key={idx} className="flex items-center justify-between py-1">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <Badge className={cn(
                        "w-4 h-4 rounded flex items-center justify-center text-[8px] font-bold shrink-0 border-0",
                        pos.outcome.toLowerCase() === 'yes' ? "bg-green-500/15 text-green-400" : "bg-red-500/15 text-red-400"
                      )}>
                        {pos.outcome[0].toUpperCase()}
                      </Badge>
                      <p className="text-[10px] truncate text-muted-foreground">{pos.market_question}</p>
                    </div>
                    <span className={cn(
                      "text-[10px] font-mono font-medium shrink-0 ml-1.5",
                      pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
                    )}>
                      {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                    </span>
                  </div>
                ))}
                {livePositions.length > 5 && (
                  <Button
                    variant="ghost"
                    onClick={() => setDashboardTab('holdings')}
                    className="text-[10px] text-muted-foreground hover:text-foreground w-full h-auto pt-1 justify-center"
                  >
                    +{livePositions.length - 5} more...
                  </Button>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* ==================== Settings Slide-Out Overlay ==================== */}
      {dashboardTab === 'settings' && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-background/80 z-40 transition-opacity"
            onClick={() => setDashboardTab('overview')}
          />
          {/* Drawer */}
          <div className="fixed top-0 right-0 bottom-0 w-full max-w-2xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
            {/* Drawer Header */}
            <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 bg-background border-b border-border/40">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-green-500" />
                <h3 className="text-sm font-semibold">Auto Trader Settings</h3>
              </div>
              <div className="flex items-center gap-2">
                {configDirty && (
                  <>
                    <Button variant="ghost" onClick={resetDraft} className="gap-1 text-[10px] h-auto px-2.5 py-1">
                      <RotateCcw className="w-3 h-3" /> Reset
                    </Button>
                    <Button
                      onClick={saveDraft}
                      disabled={configMutation.isPending}
                      className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white"
                    >
                      <Save className="w-3 h-3" /> {configMutation.isPending ? 'Saving...' : 'Save'}
                    </Button>
                  </>
                )}
                <Button
                  variant="ghost"
                  onClick={() => setDashboardTab('overview')}
                  className="text-xs h-auto px-2.5 py-1 hover:bg-card"
                >
                  Close
                </Button>
              </div>
            </div>

            {/* Unsaved banner */}
            {configDirty && (
              <div className="mx-4 mt-3 flex items-center justify-between px-3 py-2 bg-blue-500/10 border border-blue-500/30 rounded-xl">
                <span className="text-xs text-blue-400 font-medium">Unsaved changes</span>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" onClick={resetDraft} className="gap-1 text-[10px] h-auto px-2.5 py-1">
                    <RotateCcw className="w-3 h-3" /> Reset
                  </Button>
                  <Button
                    onClick={saveDraft}
                    disabled={configMutation.isPending}
                    className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white"
                  >
                    <Save className="w-3 h-3" /> {configMutation.isPending ? 'Saving...' : 'Save'}
                  </Button>
                </div>
              </div>
            )}

            {/* Settings content: 2-column grid */}
            <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3">

              {/* Spread Trading Exits */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Percent className="w-3.5 h-3.5 text-green-500" />
                  Spread Trading Exits
                </h4>
                <div className="space-y-3">
                  <SettingToggle
                    label="Enable Spread Exits"
                    description="Automatically set take-profit & stop-loss on new positions"
                    checked={configDraft.enable_spread_exits ?? true}
                    onChange={v => updateDraft('enable_spread_exits', v)}
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <SettingNumber
                      label="Take Profit %"
                      description="Sell when price rises above entry"
                      value={configDraft.take_profit_pct ?? 5}
                      onChange={v => updateDraft('take_profit_pct', v)}
                      min={0} max={100} step={0.5}
                      suffix="%"
                    />
                    <SettingNumber
                      label="Stop Loss %"
                      description="Sell when price drops below entry"
                      value={configDraft.stop_loss_pct ?? 10}
                      onChange={v => updateDraft('stop_loss_pct', v)}
                      min={0} max={100} step={0.5}
                      suffix="%"
                    />
                  </div>
                </div>
              </Card>

              {/* Position Sizing */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <DollarSign className="w-3.5 h-3.5 text-yellow-500" />
                  Position Sizing
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <SettingNumber
                    label="Base Position Size"
                    description="Default trade size"
                    value={configDraft.base_position_size_usd ?? 10}
                    onChange={v => updateDraft('base_position_size_usd', v)}
                    min={1} max={10000} step={1}
                    prefix="$"
                  />
                  <SettingNumber
                    label="Max Position Size"
                    description="Maximum per trade"
                    value={configDraft.max_position_size_usd ?? 100}
                    onChange={v => updateDraft('max_position_size_usd', v)}
                    min={1} max={50000} step={1}
                    prefix="$"
                  />
                  <SettingNumber
                    label="Paper Account Capital"
                    description="Starting capital for paper trading"
                    value={configDraft.paper_account_capital ?? 10000}
                    onChange={v => updateDraft('paper_account_capital', v)}
                    min={100} max={1000000} step={100}
                    prefix="$"
                  />
                </div>
              </Card>

              {/* Entry Criteria */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Target className="w-3.5 h-3.5 text-cyan-500" />
                  Entry Criteria
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <SettingNumber
                    label="Min ROI %"
                    description="Minimum return to trade"
                    value={configDraft.min_roi_percent ?? 2.5}
                    onChange={v => updateDraft('min_roi_percent', v)}
                    min={0} max={100} step={0.5}
                    suffix="%"
                  />
                  <SettingNumber
                    label="Max Risk Score"
                    description="Maximum acceptable risk (0-1)"
                    value={configDraft.max_risk_score ?? 0.5}
                    onChange={v => updateDraft('max_risk_score', v)}
                    min={0} max={1} step={0.05}
                  />
                  <SettingNumber
                    label="Min Liquidity"
                    description="Minimum market liquidity"
                    value={configDraft.min_liquidity_usd ?? 500}
                    onChange={v => updateDraft('min_liquidity_usd', v)}
                    min={0} max={100000} step={100}
                    prefix="$"
                  />
                  <SettingNumber
                    label="Min Volume"
                    description="Minimum trading volume"
                    value={configDraft.min_volume_usd ?? 0}
                    onChange={v => updateDraft('min_volume_usd', v)}
                    min={0} max={1000000} step={100}
                    prefix="$"
                  />
                </div>
              </Card>

              {/* Risk Management */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Shield className="w-3.5 h-3.5 text-orange-500" />
                  Risk Management
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <SettingNumber
                    label="Max Daily Trades"
                    description="Maximum trades per day"
                    value={configDraft.max_daily_trades ?? 50}
                    onChange={v => updateDraft('max_daily_trades', Math.round(v))}
                    min={1} max={1000} step={1}
                  />
                  <SettingNumber
                    label="Max Daily Loss"
                    description="Stop trading if daily loss exceeds"
                    value={configDraft.max_daily_loss_usd ?? 100}
                    onChange={v => updateDraft('max_daily_loss_usd', v)}
                    min={0} max={100000} step={10}
                    prefix="$"
                  />
                  <SettingNumber
                    label="Circuit Breaker"
                    description="Pause after N consecutive losses"
                    value={configDraft.circuit_breaker_losses ?? 3}
                    onChange={v => updateDraft('circuit_breaker_losses', Math.round(v))}
                    min={1} max={50} step={1}
                  />
                  <SettingNumber
                    label="Max Per Event"
                    description="Max trades per event"
                    value={configDraft.max_trades_per_event ?? 3}
                    onChange={v => updateDraft('max_trades_per_event', Math.round(v))}
                    min={1} max={50} step={1}
                  />
                  <SettingNumber
                    label="Max Event Exposure"
                    description="Max $ exposure per event"
                    value={configDraft.max_exposure_per_event_usd ?? 50}
                    onChange={v => updateDraft('max_exposure_per_event_usd', v)}
                    min={0} max={100000} step={10}
                    prefix="$"
                  />
                </div>
              </Card>

              {/* AI Resolution Gate */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Brain className="w-3.5 h-3.5 text-purple-500" />
                  AI Resolution Gate
                  <Badge className={cn(
                    "ml-auto rounded-md text-[9px] font-semibold border-0",
                    configDraft.ai_resolution_gate ? "bg-green-500/15 text-green-400" : "bg-gray-500/15 text-muted-foreground"
                  )}>
                    {configDraft.ai_resolution_gate ? 'ON' : 'OFF'}
                  </Badge>
                </h4>
                <p className="text-[10px] text-muted-foreground mb-3">
                  Uses AI to analyze market resolution criteria before trading. For spread trading (buy/sell on price movement),
                  this should be OFF since you&apos;re not holding to resolution.
                </p>
                <div className="space-y-3">
                  <SettingToggle
                    label="Enable Resolution Gate"
                    description="Block trades that fail AI resolution analysis"
                    checked={configDraft.ai_resolution_gate ?? false}
                    onChange={v => updateDraft('ai_resolution_gate', v)}
                  />
                  {configDraft.ai_resolution_gate && (
                    <>
                      <SettingToggle
                        label="Block 'Avoid' Recommendations"
                        description="Hard block when AI recommends avoiding a market"
                        checked={configDraft.ai_resolution_block_avoid ?? true}
                        onChange={v => updateDraft('ai_resolution_block_avoid', v)}
                      />
                      <SettingToggle
                        label="Skip on Analysis Failure"
                        description="If true, block trade when AI fails. If false, allow through (fail-open)."
                        checked={configDraft.ai_skip_on_analysis_failure ?? false}
                        onChange={v => updateDraft('ai_skip_on_analysis_failure', v)}
                      />
                      <div className="grid grid-cols-2 gap-3">
                        <SettingNumber
                          label="Max Risk Score"
                          description="Block if risk exceeds this"
                          value={configDraft.ai_max_resolution_risk ?? 0.5}
                          onChange={v => updateDraft('ai_max_resolution_risk', v)}
                          min={0} max={1} step={0.05}
                        />
                        <SettingNumber
                          label="Min Clarity Score"
                          description="Block if clarity below this"
                          value={configDraft.ai_min_resolution_clarity ?? 0.5}
                          onChange={v => updateDraft('ai_min_resolution_clarity', v)}
                          min={0} max={1} step={0.05}
                        />
                      </div>
                    </>
                  )}
                </div>
              </Card>

              {/* AI Position Sizing */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Zap className="w-3.5 h-3.5 text-amber-500" />
                  AI Position Sizing
                  <Badge className={cn(
                    "ml-auto rounded-md text-[9px] font-semibold border-0",
                    configDraft.ai_position_sizing ? "bg-green-500/15 text-green-400" : "bg-gray-500/15 text-muted-foreground"
                  )}>
                    {configDraft.ai_position_sizing ? 'ON' : 'OFF'}
                  </Badge>
                </h4>
                <div className="space-y-3">
                  <SettingToggle
                    label="Enable AI Position Sizing"
                    description="Use AI judge score to scale position sizes"
                    checked={configDraft.ai_position_sizing ?? true}
                    onChange={v => updateDraft('ai_position_sizing', v)}
                  />
                  {configDraft.ai_position_sizing && (
                    <>
                      <SettingToggle
                        label="Score-Based Multiplier"
                        description="Scale position size by AI score (0.8 score = 80% size)"
                        checked={configDraft.ai_score_size_multiplier ?? true}
                        onChange={v => updateDraft('ai_score_size_multiplier', v)}
                      />
                      <div className="grid grid-cols-2 gap-3">
                        <SettingNumber
                          label="Min Score to Trade"
                          description="Block if AI score below this (0 = disabled)"
                          value={configDraft.ai_min_score_to_trade ?? 0}
                          onChange={v => updateDraft('ai_min_score_to_trade', v)}
                          min={0} max={1} step={0.05}
                        />
                        <SettingNumber
                          label="Boost Threshold"
                          description="Boost size when score exceeds"
                          value={configDraft.ai_score_boost_threshold ?? 0.85}
                          onChange={v => updateDraft('ai_score_boost_threshold', v)}
                          min={0} max={1} step={0.05}
                        />
                        <SettingNumber
                          label="Boost Multiplier"
                          description="Size multiplier for high confidence"
                          value={configDraft.ai_score_boost_multiplier ?? 1.2}
                          onChange={v => updateDraft('ai_score_boost_multiplier', v)}
                          min={1} max={3} step={0.1}
                          suffix="x"
                        />
                      </div>
                    </>
                  )}
                </div>
              </Card>

              {/* Settlement Filters */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Crosshair className="w-3.5 h-3.5 text-teal-500" />
                  Settlement Filters
                </h4>
                <div className="space-y-3">
                  <SettingToggle
                    label="Prefer Near Settlement"
                    description="Boost score for markets settling sooner"
                    checked={configDraft.prefer_near_settlement ?? true}
                    onChange={v => updateDraft('prefer_near_settlement', v)}
                  />
                  <SettingToggle
                    label="Require Profit Guarantee"
                    description="Only trade if guaranteed profit covers gas + slippage (Proposition 4.1)"
                    checked={configDraft.use_profit_guarantee ?? true}
                    onChange={v => updateDraft('use_profit_guarantee', v)}
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <SettingNumber
                      label="Max Days to Settlement"
                      description="Skip markets further out (0 = no limit)"
                      value={configDraft.max_end_date_days ?? 0}
                      onChange={v => updateDraft('max_end_date_days', v === 0 ? null : Math.round(v))}
                      min={0} max={365} step={1}
                    />
                    <SettingNumber
                      label="Min Days to Settlement"
                      description="Skip markets settling too soon (0 = no limit)"
                      value={configDraft.min_end_date_days ?? 0}
                      onChange={v => updateDraft('min_end_date_days', v === 0 ? null : Math.round(v))}
                      min={0} max={365} step={1}
                    />
                  </div>
                </div>
              </Card>

              {/* AI Integration */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Brain className="w-3.5 h-3.5 text-purple-500" />
                  AI Integration
                </h4>
                <div className="space-y-3">
                  <SettingToggle
                    label="LLM Verify Before Trading"
                    description="Use AI to verify opportunities before executing trades"
                    checked={configDraft.llm_verify_trades ?? false}
                    onChange={v => updateDraft('llm_verify_trades', v)}
                  />
                  {configDraft.llm_verify_trades && (
                    <div>
                      <p className="text-xs font-medium mb-0.5">Strategies to LLM-Verify</p>
                      <p className="text-[10px] text-muted-foreground mb-1.5">Comma-separated list (empty = verify all)</p>
                      <input
                        type="text"
                        value={(configDraft.llm_verify_strategies || []).join(', ')}
                        onChange={e => updateDraft('llm_verify_strategies', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                        placeholder="e.g. cross_platform, bayesian_cascade, stat_arb"
                        className="w-full bg-card border border-border rounded-lg py-1.5 px-2.5 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-green-500/50 focus:border-green-500/50"
                      />
                    </div>
                  )}
                  <SettingToggle
                    label="Auto AI Scoring"
                    description="When enabled, scanner auto-scores all opportunities with AI. Disable for faster scans."
                    checked={configDraft.auto_ai_scoring ?? false}
                    onChange={v => updateDraft('auto_ai_scoring', v)}
                  />
                </div>
              </Card>

              {/* Enabled Strategies - full width */}
              <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 md:col-span-2">
                <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
                  <Target className="w-3.5 h-3.5 text-emerald-500" />
                  Enabled Strategies
                </h4>
                <p className="text-[10px] text-muted-foreground mb-2">Select which strategies the auto trader should use</p>
                <div className="flex flex-wrap gap-1.5">
                  {ALL_STRATEGIES.map(s => {
                    const enabled = (configDraft.enabled_strategies || []).includes(s.key)
                    return (
                      <button
                        key={s.key}
                        type="button"
                        onClick={() => {
                          const current = configDraft.enabled_strategies || []
                          updateDraft('enabled_strategies', enabled
                            ? current.filter((k: string) => k !== s.key)
                            : [...current, s.key]
                          )
                        }}
                        className={cn(
                          "px-2.5 py-1 rounded-lg text-[10px] font-medium border transition-colors",
                          enabled
                            ? "bg-green-500/15 text-green-400 border-green-500/30"
                            : "bg-card text-muted-foreground border-border hover:border-green-500/20"
                        )}
                      >
                        {s.label}
                      </button>
                    )
                  })}
                </div>
                <div className="flex items-center gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => updateDraft('enabled_strategies', ALL_STRATEGIES.map(s => s.key))}
                    className="text-[10px] text-muted-foreground hover:text-green-400 transition-colors"
                  >
                    Select All
                  </button>
                  <span className="text-muted-foreground text-[10px]">|</span>
                  <button
                    type="button"
                    onClick={() => updateDraft('enabled_strategies', [])}
                    className="text-[10px] text-muted-foreground hover:text-red-400 transition-colors"
                  >
                    Clear All
                  </button>
                </div>
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ==================== Sub-Components ====================

function FeedEventRow({ event, isNew }: { event: FeedEvent; isNew: boolean }) {
  const iconMap = {
    trade: <Zap className="w-3 h-3 text-blue-400" />,
    win: <ArrowUpRight className="w-3 h-3 text-green-400" />,
    loss: <ArrowDownRight className="w-3 h-3 text-red-400" />,
    scan: <Crosshair className="w-3 h-3 text-cyan-400" />,
    alert: <AlertTriangle className="w-3 h-3 text-yellow-400" />,
    system: <Cpu className="w-3 h-3 text-purple-400" />,
    position: <Briefcase className="w-3 h-3 text-indigo-400" />,
  }

  const borderColorMap = {
    trade: 'border-l-blue-500/50',
    win: 'border-l-green-500/50',
    loss: 'border-l-red-500/50',
    scan: 'border-l-cyan-500/30',
    alert: 'border-l-yellow-500/50',
    system: 'border-l-purple-500/50',
    position: 'border-l-indigo-500/50',
  }

  return (
    <div className={cn(
      "flex items-start gap-2 px-3 py-2 border-l-2 transition-all",
      borderColorMap[event.icon],
      isNew && "bg-white/[0.02]"
    )}>
      <div className="mt-0.5 shrink-0">{iconMap[event.icon]}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <p className="text-xs font-medium truncate">{event.title}</p>
          {event.value && (
            <span className={cn("text-[10px] font-mono font-semibold shrink-0", event.valueColor || 'text-muted-foreground')}>
              {event.value}
            </span>
          )}
        </div>
        <p className="text-[10px] text-muted-foreground truncate">{event.detail}</p>
      </div>
      <span className="text-[9px] text-muted-foreground font-mono shrink-0 mt-0.5">
        {formatFeedTime(event.timestamp)}
      </span>
    </div>
  )
}

function MetricRow({ label, value, valueColor = 'text-foreground', icon, sub, sparkData }: {
  label: string
  value: string
  valueColor?: string
  icon: React.ReactNode
  sub?: string
  sparkData?: number[]
}) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border/20 last:border-0">
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">{icon}</span>
        <div>
          <p className="text-[10px] text-muted-foreground">{label}</p>
          {sub && <p className="text-[9px] text-muted-foreground">{sub}</p>}
        </div>
      </div>
      <div className="flex items-center gap-1.5">
        {sparkData && sparkData.length > 2 && <MiniSparkline data={sparkData} />}
        <p className={cn("text-xs font-mono font-semibold", valueColor)}>{value}</p>
      </div>
    </div>
  )
}

function MiniSparkline({ data }: { data: number[] }) {
  const width = 48
  const height = 16
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 2) - 1
    return `${x},${y}`
  }).join(' ')

  const lastVal = data[data.length - 1]
  const color = lastVal >= (data[0] || 0) ? '#22c55e' : '#ef4444'

  return (
    <svg width={width} height={height} className="opacity-60">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    open: 'bg-blue-500/15 text-blue-400',
    resolved_win: 'bg-green-500/15 text-green-400',
    resolved_loss: 'bg-red-500/15 text-red-400',
    pending: 'bg-yellow-500/15 text-yellow-400',
    executed: 'bg-blue-500/15 text-blue-400',
    failed: 'bg-red-500/15 text-red-400',
    shadow: 'bg-gray-500/15 text-muted-foreground',
  }
  return (
    <Badge className={cn("px-1.5 py-0.5 rounded-md text-[9px] font-semibold border-0", colors[status] || 'bg-gray-500/15 text-muted-foreground')}>
      {status.replace('_', ' ').toUpperCase()}
    </Badge>
  )
}

function PnlChart({ points }: { points: { date: string; equity: number }[] }) {
  const chartHeight = 200
  const chartWidth = 100

  const equities = points.map(p => p.equity)
  const maxVal = Math.max(...equities, 0)
  const minVal = Math.min(...equities, 0)
  const range = maxVal - minVal || 1

  const getY = (val: number) => chartHeight - ((val - minVal) / range) * (chartHeight - 20) - 10
  const getX = (i: number) => (i / (points.length - 1 || 1)) * chartWidth

  const linePath = points.map((p, i) => {
    const x = getX(i)
    const y = getY(p.equity)
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ')

  const areaPath = `${linePath} L ${chartWidth} ${chartHeight} L 0 ${chartHeight} Z`
  const zeroY = getY(0)

  const lastEquity = points[points.length - 1]?.equity || 0
  const isProfitable = lastEquity >= 0
  const lineColor = isProfitable ? '#22c55e' : '#ef4444'

  return (
    <div className="relative h-full">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={isProfitable ? '#22c55e' : '#ef4444'} stopOpacity="0.15" />
            <stop offset="100%" stopColor={isProfitable ? '#22c55e' : '#ef4444'} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#chartGradient)" />
        <line
          x1="0" y1={zeroY}
          x2={chartWidth} y2={zeroY}
          stroke="currentColor" opacity={0.15}
          strokeWidth="0.3"
          strokeDasharray="1,1"
        />
        <path
          d={linePath}
          fill="none"
          stroke={lineColor}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
        {/* Glow effect */}
        <path
          d={linePath}
          fill="none"
          stroke={lineColor}
          strokeWidth="4"
          vectorEffect="non-scaling-stroke"
          opacity="0.15"
        />
      </svg>

      <div className="absolute top-1 left-2 text-[10px] font-mono text-muted-foreground">
        ${maxVal.toFixed(0)}
      </div>
      <div className="absolute bottom-1 left-2 text-[10px] font-mono text-muted-foreground">
        ${minVal.toFixed(0)}
      </div>
      <div className="absolute bottom-1 right-2 flex items-center gap-2">
        <span className={cn("text-xs font-mono font-semibold", isProfitable ? 'text-green-400' : 'text-red-400')}>
          {isProfitable ? '+' : ''}${lastEquity.toFixed(2)}
        </span>
      </div>
      <div className="absolute top-1 right-2 text-[10px] font-mono text-gray-700">
        {points.length > 0 && new Date(points[0].date).toLocaleDateString()} - {points.length > 0 && new Date(points[points.length - 1].date).toLocaleDateString()}
      </div>
    </div>
  )
}

// ==================== Utilities ====================

function formatFeedTime(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${diffSec}s`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h`
  return date.toLocaleDateString()
}

function timeAgo(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  const diffDays = Math.floor(diffHr / 24)
  return `${diffDays}d ago`
}

// ==================== Settings Components ====================

function SettingToggle({ label, description, checked, onChange }: {
  label: string
  description: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <div className="flex-1 min-w-0 mr-3">
        <p className="text-xs font-medium">{label}</p>
        <p className="text-[10px] text-muted-foreground">{description}</p>
      </div>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors",
          checked ? "bg-green-500" : "bg-muted-foreground/40"
        )}
      >
        <span className={cn(
          "pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow transform transition-transform",
          checked ? "translate-x-4" : "translate-x-0"
        )} />
      </button>
    </div>
  )
}

function SettingNumber({ label, description, value, onChange, min, max, step, prefix, suffix }: {
  label: string
  description: string
  value: number | null
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  prefix?: string
  suffix?: string
}) {
  return (
    <div>
      <p className="text-xs font-medium mb-0.5">{label}</p>
      <p className="text-[10px] text-muted-foreground mb-1.5">{description}</p>
      <div className="relative">
        {prefix && (
          <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground">{prefix}</span>
        )}
        <input
          type="number"
          value={value ?? 0}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          min={min}
          max={max}
          step={step}
          className={cn(
            "w-full bg-card border border-border rounded-lg py-1.5 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-green-500/50 focus:border-green-500/50",
            prefix ? "pl-6 pr-2.5" : "px-2.5",
            suffix ? "pr-7" : ""
          )}
        />
        {suffix && (
          <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground">{suffix}</span>
        )}
      </div>
    </div>
  )
}
