import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Play,
  Square,
  AlertTriangle,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Settings,
  Zap,
  RefreshCw,
  ShieldAlert,
  BarChart3,
  Briefcase,
  Award,
  Target,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  BookOpen,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Wallet,
} from 'lucide-react'
import clsx from 'clsx'
import {
  getAutoTraderStatus,
  startAutoTrader,
  stopAutoTrader,
  updateAutoTraderConfig,
  getAutoTraderTrades,
  resetCircuitBreaker,
  emergencyStopAutoTrader,
  getTradingStatus,
  getTradingPositions,
  getTradingBalance,
  getOrders,
  AutoTraderStatus,
  AutoTraderTrade,
} from '../services/api'

interface TradingPosition {
  token_id: string
  market_id: string
  market_question: string
  outcome: string
  size: number
  average_cost: number
  current_price: number
  unrealized_pnl: number
}

type DashboardTab = 'overview' | 'holdings' | 'orders' | 'config'

export default function TradingPanel() {
  const [showConfig, setShowConfig] = useState(false)
  const [dashboardTab, setDashboardTab] = useState<DashboardTab>('overview')
  const [tradeFilter, setTradeFilter] = useState<string>('all')
  const [tradeSort, setTradeSort] = useState<'date' | 'pnl' | 'cost'>('date')
  const [tradeSortDir, setTradeSortDir] = useState<'asc' | 'desc'>('desc')
  const [configForm, setConfigForm] = useState({
    min_roi_percent: 2.5,
    max_risk_score: 0.5,
    base_position_size_usd: 10,
    max_position_size_usd: 100,
    max_daily_trades: 50,
    max_daily_loss_usd: 100,
    paper_account_capital: 10000
  })
  const queryClient = useQueryClient()

  const { data: status, isLoading } = useQuery({
    queryKey: ['auto-trader-status'],
    queryFn: getAutoTraderStatus,
    refetchInterval: 5000,
  })

  const { data: trades = [] } = useQuery({
    queryKey: ['auto-trader-trades'],
    queryFn: () => getAutoTraderTrades(500),
    refetchInterval: 10000,
  })

  const { data: tradingStatus } = useQuery({
    queryKey: ['trading-status'],
    queryFn: getTradingStatus,
    refetchInterval: 10000,
  })

  const { data: livePositions = [] } = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    refetchInterval: 15000,
  })

  const { data: balance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
  })

  const { data: orders = [] } = useQuery({
    queryKey: ['trading-orders'],
    queryFn: () => getOrders(100),
    refetchInterval: 15000,
  })

  const startMutation = useMutation({
    mutationFn: (mode: string) => startAutoTrader(mode),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
  })

  const stopMutation = useMutation({
    mutationFn: stopAutoTrader,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
  })

  const configMutation = useMutation({
    mutationFn: updateAutoTraderConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      setShowConfig(false)
    }
  })

  const resetCircuitMutation = useMutation({
    mutationFn: resetCircuitBreaker,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
  })

  const emergencyStopMutation = useMutation({
    mutationFn: emergencyStopAutoTrader,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
  })

  const stats = status?.stats
  const config = status?.config

  // Compute performance metrics from trades
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

    // Max drawdown
    let peak = 0, maxDD = 0, cumPnl = 0
    const sortedTrades = [...resolved].sort((a, b) => new Date(a.executed_at).getTime() - new Date(b.executed_at).getTime())
    sortedTrades.forEach(t => {
      cumPnl += t.actual_profit || 0
      if (cumPnl > peak) peak = cumPnl
      const dd = peak - cumPnl
      if (dd > maxDD) maxDD = dd
    })

    // Strategy breakdown
    const byStrategy: Record<string, { trades: number; pnl: number; wins: number; losses: number; cost: number }> = {}
    trades.forEach(t => {
      if (!byStrategy[t.strategy]) byStrategy[t.strategy] = { trades: 0, pnl: 0, wins: 0, losses: 0, cost: 0 }
      byStrategy[t.strategy].trades++
      byStrategy[t.strategy].pnl += t.actual_profit || 0
      byStrategy[t.strategy].cost += t.total_cost
      if ((t.actual_profit || 0) > 0) byStrategy[t.strategy].wins++
      if ((t.actual_profit || 0) < 0) byStrategy[t.strategy].losses++
    })

    // Equity curve data
    const equityPoints: { date: string; equity: number }[] = []
    let equity = 0
    sortedTrades.forEach(t => {
      equity += t.actual_profit || 0
      equityPoints.push({ date: t.executed_at, equity })
    })

    return {
      totalPnl,
      totalCost,
      profitFactor,
      avgWin,
      avgLoss,
      bestTrade,
      worstTrade,
      maxDrawdown: maxDD,
      byStrategy,
      equityPoints,
      winCount: wins.length,
      lossCount: losses.length,
    }
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

  // Live position totals
  const positionsTotalValue = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.current_price, 0)
  const positionsCostBasis = livePositions.reduce((s: number, p: TradingPosition) => s + p.size * p.average_cost, 0)
  const positionsUnrealizedPnl = livePositions.reduce((s: number, p: TradingPosition) => s + p.unrealized_pnl, 0)

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Autonomous Trading</h2>
          <p className="text-sm text-gray-500">
            Automatically execute arbitrage opportunities
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
          >
            <Settings className="w-4 h-4" />
            Config
          </button>

          {status?.running ? (
            <button
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 rounded-lg text-sm font-medium"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => startMutation.mutate('paper')}
                disabled={startMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-sm font-medium"
              >
                <Play className="w-4 h-4" />
                Paper Mode
              </button>
              <button
                onClick={() => {
                  if (confirm('Enable LIVE trading? This will use REAL MONEY.')) {
                    startMutation.mutate('live')
                  }
                }}
                disabled={startMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
              >
                <Zap className="w-4 h-4" />
                Live Mode
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className={clsx(
        "flex items-center justify-between p-4 rounded-lg border",
        status?.running
          ? status?.config.mode === 'live'
            ? "bg-green-500/10 border-green-500/50"
            : "bg-blue-500/10 border-blue-500/50"
          : "bg-gray-800 border-gray-700"
      )}>
        <div className="flex items-center gap-3">
          <div className={clsx(
            "w-3 h-3 rounded-full",
            status?.running
              ? status?.config.mode === 'live' ? "bg-green-500 animate-pulse" : "bg-blue-500 animate-pulse"
              : "bg-gray-500"
          )} />
          <div>
            <p className="font-medium">
              {status?.running
                ? `Running in ${status?.config.mode?.toUpperCase()} mode`
                : 'Stopped'
              }
            </p>
            <p className="text-xs text-gray-400">
              {stats?.opportunities_seen || 0} opportunities scanned
            </p>
          </div>
        </div>

        {stats?.circuit_breaker_active && (
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <span className="text-yellow-500 text-sm">Circuit Breaker Active</span>
            <button
              onClick={() => resetCircuitMutation.mutate()}
              className="px-2 py-1 bg-yellow-500/20 hover:bg-yellow-500/30 rounded text-xs"
            >
              Reset
            </button>
          </div>
        )}

        <button
          onClick={() => {
            if (confirm('EMERGENCY STOP - Cancel all orders and stop trading?')) {
              emergencyStopMutation.mutate()
            }
          }}
          className="flex items-center gap-2 px-3 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm"
        >
          <ShieldAlert className="w-4 h-4" />
          Emergency Stop
        </button>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MiniStat
          label="Total Trades"
          value={stats?.total_trades?.toString() || '0'}
          icon={<Activity className="w-4 h-4 text-blue-400" />}
        />
        <MiniStat
          label="Win Rate"
          value={`${((stats?.win_rate || 0) * 100).toFixed(1)}%`}
          icon={<Award className="w-4 h-4 text-yellow-400" />}
          subtitle={`${stats?.winning_trades || 0}W / ${stats?.losing_trades || 0}L`}
        />
        <MiniStat
          label="Total P&L"
          value={`${(stats?.total_profit || 0) >= 0 ? '+' : ''}$${(stats?.total_profit || 0).toFixed(2)}`}
          icon={(stats?.total_profit || 0) >= 0 ? <TrendingUp className="w-4 h-4 text-green-400" /> : <TrendingDown className="w-4 h-4 text-red-400" />}
          valueColor={(stats?.total_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <MiniStat
          label="ROI"
          value={`${(stats?.roi_percent || 0) >= 0 ? '+' : ''}${(stats?.roi_percent || 0).toFixed(2)}%`}
          icon={<Target className="w-4 h-4 text-purple-400" />}
          valueColor={(stats?.roi_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <MiniStat
          label="Daily P&L"
          value={`${(stats?.daily_profit || 0) >= 0 ? '+' : ''}$${(stats?.daily_profit || 0).toFixed(2)}`}
          icon={<DollarSign className="w-4 h-4 text-yellow-400" />}
          valueColor={(stats?.daily_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
          subtitle={`${stats?.daily_trades || 0} today`}
        />
        <MiniStat
          label="Total Invested"
          value={`$${(stats?.total_invested || 0).toFixed(2)}`}
          icon={<BookOpen className="w-4 h-4 text-cyan-400" />}
        />
      </div>

      {/* Advanced Metrics */}
      {performanceMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <MiniStat
            label="Profit Factor"
            value={performanceMetrics.profitFactor > 0 ? performanceMetrics.profitFactor.toFixed(2) : 'N/A'}
            icon={<PieChart className="w-4 h-4 text-indigo-400" />}
          />
          <MiniStat
            label="Max Drawdown"
            value={`$${performanceMetrics.maxDrawdown.toFixed(2)}`}
            icon={<AlertTriangle className="w-4 h-4 text-orange-400" />}
            valueColor="text-orange-400"
          />
          <MiniStat
            label="Avg Win"
            value={`$${performanceMetrics.avgWin.toFixed(2)}`}
            icon={<ArrowUpRight className="w-4 h-4 text-green-400" />}
            valueColor="text-green-400"
          />
          <MiniStat
            label="Avg Loss"
            value={`$${performanceMetrics.avgLoss.toFixed(2)}`}
            icon={<ArrowDownRight className="w-4 h-4 text-red-400" />}
            valueColor="text-red-400"
          />
          <MiniStat
            label="Positions Value"
            value={`$${positionsTotalValue.toFixed(2)}`}
            icon={<Briefcase className="w-4 h-4 text-blue-400" />}
          />
          <MiniStat
            label="Unrealized P&L"
            value={`${positionsUnrealizedPnl >= 0 ? '+' : ''}$${positionsUnrealizedPnl.toFixed(2)}`}
            icon={positionsUnrealizedPnl >= 0 ? <TrendingUp className="w-4 h-4 text-green-400" /> : <TrendingDown className="w-4 h-4 text-red-400" />}
            valueColor={positionsUnrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}
          />
        </div>
      )}

      {/* Dashboard Tabs */}
      <div className="flex bg-[#141414] rounded-lg p-1 border border-gray-800 w-fit">
        {([
          { key: 'overview', label: 'Performance', icon: <BarChart3 className="w-3.5 h-3.5" /> },
          { key: 'holdings', label: 'Holdings', icon: <Briefcase className="w-3.5 h-3.5" /> },
          { key: 'orders', label: 'Trade History', icon: <Activity className="w-3.5 h-3.5" /> },
          { key: 'config', label: 'Settings', icon: <Settings className="w-3.5 h-3.5" /> },
        ] as { key: DashboardTab; label: string; icon: React.ReactNode }[]).map(tab => (
          <button
            key={tab.key}
            onClick={() => setDashboardTab(tab.key)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-colors",
              dashboardTab === tab.key ? "bg-green-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            {tab.icon}
            {tab.label}
            {tab.key === 'holdings' && livePositions.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded text-xs">
                {livePositions.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {dashboardTab === 'overview' && (
        <div className="space-y-4">
          {/* Equity Curve */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-6">
            <h4 className="font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-green-500" />
              Cumulative P&L Over Time
            </h4>
            {performanceMetrics && performanceMetrics.equityPoints.length > 1 ? (
              <div className="h-64">
                <PnlChart points={performanceMetrics.equityPoints} />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                Start trading to see your performance chart
              </div>
            )}
          </div>

          {/* Strategy Breakdown */}
          {performanceMetrics && Object.keys(performanceMetrics.byStrategy).length > 0 && (
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-6">
              <h4 className="font-semibold mb-4 flex items-center gap-2">
                <PieChart className="w-5 h-5 text-indigo-500" />
                Performance by Strategy
              </h4>
              <div className="space-y-2">
                {Object.entries(performanceMetrics.byStrategy)
                  .sort((a, b) => b[1].pnl - a[1].pnl)
                  .map(([strategy, data]) => {
                    const winRate = (data.wins + data.losses) > 0 ? (data.wins / (data.wins + data.losses)) * 100 : 0
                    return (
                      <div key={strategy} className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
                        <div className="flex items-center gap-3">
                          <div className={clsx("w-2 h-2 rounded-full", data.pnl >= 0 ? "bg-green-500" : "bg-red-500")} />
                          <div>
                            <p className="font-medium text-sm">{strategy}</p>
                            <p className="text-xs text-gray-500">
                              {data.trades} trades | {winRate.toFixed(0)}% win rate | Cost: ${data.cost.toFixed(2)}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={clsx("font-mono font-medium", data.pnl >= 0 ? "text-green-400" : "text-red-400")}>
                            {data.pnl >= 0 ? '+' : ''}${data.pnl.toFixed(2)}
                          </p>
                          <p className="text-xs text-gray-500">{data.wins}W / {data.losses}L</p>
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}

          {/* Best/Worst + Daily Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {performanceMetrics && (
              <>
                <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Best Trade</p>
                  <p className="text-xl font-mono font-bold text-green-400">
                    +${performanceMetrics.bestTrade.toFixed(2)}
                  </p>
                </div>
                <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Worst Trade</p>
                  <p className="text-xl font-mono font-bold text-red-400">
                    ${performanceMetrics.worstTrade.toFixed(2)}
                  </p>
                </div>
              </>
            )}
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-500 mb-1">Executed Today</p>
              <p className="text-xl font-mono font-bold">{stats?.opportunities_executed || 0}</p>
            </div>
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-500 mb-1">Skipped Today</p>
              <p className="text-xl font-mono font-bold text-gray-400">{stats?.opportunities_skipped || 0}</p>
            </div>
          </div>
        </div>
      )}

      {dashboardTab === 'holdings' && (
        <div className="space-y-4">
          {/* Wallet Info */}
          {tradingStatus && (
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Wallet className="w-5 h-5 text-purple-400" />
                  <div>
                    <p className="text-sm font-medium">Trading Wallet</p>
                    <p className="text-xs text-gray-500 font-mono">
                      {tradingStatus.wallet_address
                        ? `${tradingStatus.wallet_address.slice(0, 10)}...${tradingStatus.wallet_address.slice(-8)}`
                        : 'Not connected'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <p className="text-xs text-gray-500">USDC Balance</p>
                    <p className="font-mono font-bold">${balance?.balance?.toFixed(2) || '0.00'}</p>
                  </div>
                  <div className={clsx(
                    "px-2 py-1 rounded text-xs font-medium",
                    tradingStatus.initialized ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-gray-400"
                  )}>
                    {tradingStatus.initialized ? 'Connected' : 'Not Initialized'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Holdings Summary */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-500 mb-1">Open Positions</p>
              <p className="text-2xl font-mono font-bold">{livePositions.length}</p>
            </div>
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-500 mb-1">Book Value (Cost Basis)</p>
              <p className="text-2xl font-mono font-bold">${positionsCostBasis.toFixed(2)}</p>
            </div>
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-500 mb-1">Market Value</p>
              <p className="text-2xl font-mono font-bold">${positionsTotalValue.toFixed(2)}</p>
            </div>
          </div>

          {/* Positions Table */}
          {livePositions.length === 0 ? (
            <div className="text-center py-8 bg-[#141414] border border-gray-800 rounded-lg">
              <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400">No open live trading positions</p>
              <p className="text-sm text-gray-600">Start trading to see your holdings here</p>
            </div>
          ) : (
            <div className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 text-xs">
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
                  {livePositions.map((pos: TradingPosition, idx: number) => {
                    const costBasis = pos.size * pos.average_cost
                    const mktValue = pos.size * pos.current_price
                    const pnlPct = costBasis > 0 ? (pos.unrealized_pnl / costBasis) * 100 : 0
                    return (
                      <tr key={idx} className="border-b border-gray-800/50 hover:bg-[#1a1a1a] transition-colors">
                        <td className="px-4 py-3">
                          <p className="font-medium text-sm line-clamp-1">{pos.market_question}</p>
                          {pos.market_id && (
                            <a
                              href={`https://polymarket.com/event/${pos.market_id}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                            >
                              View <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </td>
                        <td className="text-center px-3 py-3">
                          <span className={clsx(
                            "px-2 py-0.5 rounded text-xs font-medium",
                            pos.outcome.toLowerCase() === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                          )}>
                            {pos.outcome.toUpperCase()}
                          </span>
                        </td>
                        <td className="text-right px-3 py-3 font-mono">{pos.size.toFixed(2)}</td>
                        <td className="text-right px-3 py-3 font-mono">${pos.average_cost.toFixed(4)}</td>
                        <td className="text-right px-3 py-3 font-mono">${pos.current_price.toFixed(4)}</td>
                        <td className="text-right px-3 py-3 font-mono">${costBasis.toFixed(2)}</td>
                        <td className="text-right px-3 py-3 font-mono">${mktValue.toFixed(2)}</td>
                        <td className="text-right px-4 py-3">
                          <span className={clsx("font-mono font-medium", pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                            {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                          </span>
                          <span className={clsx("text-xs ml-1", pnlPct >= 0 ? "text-green-400/70" : "text-red-400/70")}>
                            ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%)
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
                <tfoot>
                  <tr className="border-t border-gray-700 font-medium">
                    <td className="px-4 py-3 text-gray-400" colSpan={5}>Totals</td>
                    <td className="text-right px-3 py-3 font-mono">${positionsCostBasis.toFixed(2)}</td>
                    <td className="text-right px-3 py-3 font-mono">${positionsTotalValue.toFixed(2)}</td>
                    <td className="text-right px-4 py-3">
                      <span className={clsx("font-mono font-medium", positionsUnrealizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
                        {positionsUnrealizedPnl >= 0 ? '+' : ''}${positionsUnrealizedPnl.toFixed(2)}
                      </span>
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
          )}
        </div>
      )}

      {dashboardTab === 'orders' && (
        <div className="space-y-4">
          {/* Trade Filters */}
          <div className="flex items-center gap-3">
            <select
              value={tradeFilter}
              onChange={(e) => setTradeFilter(e.target.value)}
              className="bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
            >
              <option value="all">All Trades</option>
              <option value="open">Open</option>
              <option value="wins">Wins</option>
              <option value="losses">Losses</option>
            </select>
            <div className="flex items-center gap-1 text-xs text-gray-500">
              Sort:
              {(['date', 'pnl', 'cost'] as const).map(s => (
                <button
                  key={s}
                  onClick={() => {
                    if (tradeSort === s) setTradeSortDir(d => d === 'desc' ? 'asc' : 'desc')
                    else { setTradeSort(s); setTradeSortDir('desc') }
                  }}
                  className={clsx(
                    "px-2 py-1 rounded",
                    tradeSort === s ? "bg-green-500/20 text-green-400" : "hover:bg-gray-800"
                  )}
                >
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                  {tradeSort === s && (tradeSortDir === 'desc' ? <ChevronDown className="w-3 h-3 inline ml-0.5" /> : <ChevronUp className="w-3 h-3 inline ml-0.5" />)}
                </button>
              ))}
            </div>
            <span className="text-xs text-gray-500 ml-auto">{processedTrades.length} trades</span>
          </div>

          {/* Trades Table */}
          {processedTrades.length === 0 ? (
            <div className="text-center py-8 bg-[#141414] border border-gray-800 rounded-lg">
              <p className="text-gray-400">No trades found</p>
            </div>
          ) : (
            <div className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden">
              <div className="max-h-[600px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-[#141414]">
                    <tr className="border-b border-gray-800 text-gray-500 text-xs">
                      <th className="text-left px-4 py-3">Date</th>
                      <th className="text-left px-3 py-3">Strategy</th>
                      <th className="text-center px-3 py-3">Mode</th>
                      <th className="text-right px-3 py-3">Cost</th>
                      <th className="text-right px-3 py-3">Expected</th>
                      <th className="text-center px-3 py-3">Status</th>
                      <th className="text-right px-4 py-3">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {processedTrades.map((trade) => (
                      <tr key={trade.id} className="border-b border-gray-800/50 hover:bg-[#1a1a1a] transition-colors">
                        <td className="px-4 py-3">
                          <p className="font-mono text-xs">{new Date(trade.executed_at).toLocaleDateString()}</p>
                          <p className="font-mono text-xs text-gray-500">{new Date(trade.executed_at).toLocaleTimeString()}</p>
                        </td>
                        <td className="px-3 py-3">
                          <p className="font-medium">{trade.strategy}</p>
                        </td>
                        <td className="text-center px-3 py-3">
                          <span className={clsx(
                            "px-2 py-0.5 rounded text-xs font-medium",
                            trade.mode === 'live' ? "bg-green-500/20 text-green-400" :
                            trade.mode === 'paper' ? "bg-blue-500/20 text-blue-400" : "bg-gray-500/20 text-gray-400"
                          )}>
                            {trade.mode.toUpperCase()}
                          </span>
                        </td>
                        <td className="text-right px-3 py-3 font-mono">${trade.total_cost.toFixed(2)}</td>
                        <td className="text-right px-3 py-3 font-mono text-gray-400">${trade.expected_profit.toFixed(2)}</td>
                        <td className="text-center px-3 py-3">
                          <StatusBadge status={trade.status} />
                        </td>
                        <td className="text-right px-4 py-3">
                          {trade.actual_profit !== null ? (
                            <span className={clsx("font-mono font-medium", (trade.actual_profit || 0) >= 0 ? "text-green-400" : "text-red-400")}>
                              {(trade.actual_profit || 0) >= 0 ? '+' : ''}${(trade.actual_profit || 0).toFixed(2)}
                            </span>
                          ) : (
                            <span className="text-gray-400 font-mono">+${trade.expected_profit.toFixed(2)} exp</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {dashboardTab === 'config' && (
        <div className="space-y-4">
          {/* Current Config */}
          {config && (
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium">Current Configuration</h4>
                <button
                  onClick={() => setShowConfig(!showConfig)}
                  className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded-lg text-sm hover:bg-blue-500/30"
                >
                  Edit
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-500 text-xs">Min ROI</p>
                  <p className="font-mono">{config.min_roi_percent}%</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Max Risk</p>
                  <p className="font-mono">{config.max_risk_score}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Position Size</p>
                  <p className="font-mono">${config.base_position_size_usd} - ${config.max_position_size_usd}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Daily Limits</p>
                  <p className="font-mono">{config.max_daily_trades} trades / ${config.max_daily_loss_usd} loss</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Strategies</p>
                  <p className="font-mono text-xs">{config.enabled_strategies?.join(', ')}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Circuit Breaker</p>
                  <p className="font-mono">{config.circuit_breaker_losses} losses</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Paper Capital</p>
                  <p className="font-mono">${config.paper_account_capital?.toLocaleString() || '10,000'}</p>
                </div>
              </div>
            </div>
          )}

          {/* Edit Config */}
          {showConfig && (
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <h4 className="font-medium mb-4">Edit Configuration</h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Min ROI %</label>
                  <input
                    type="number"
                    value={configForm.min_roi_percent}
                    onChange={(e) => setConfigForm({ ...configForm, min_roi_percent: parseFloat(e.target.value) })}
                    step="0.5"
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max Risk Score</label>
                  <input
                    type="number"
                    value={configForm.max_risk_score}
                    onChange={(e) => setConfigForm({ ...configForm, max_risk_score: parseFloat(e.target.value) })}
                    step="0.1"
                    min="0"
                    max="1"
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Base Position Size ($)</label>
                  <input
                    type="number"
                    value={configForm.base_position_size_usd}
                    onChange={(e) => setConfigForm({ ...configForm, base_position_size_usd: parseFloat(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max Position Size ($)</label>
                  <input
                    type="number"
                    value={configForm.max_position_size_usd}
                    onChange={(e) => setConfigForm({ ...configForm, max_position_size_usd: parseFloat(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max Daily Trades</label>
                  <input
                    type="number"
                    value={configForm.max_daily_trades}
                    onChange={(e) => setConfigForm({ ...configForm, max_daily_trades: parseInt(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max Daily Loss ($)</label>
                  <input
                    type="number"
                    value={configForm.max_daily_loss_usd}
                    onChange={(e) => setConfigForm({ ...configForm, max_daily_loss_usd: parseFloat(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
                <div className="col-span-2">
                  <label className="block text-xs text-gray-500 mb-1">Paper Account Capital ($)</label>
                  <input
                    type="number"
                    value={configForm.paper_account_capital}
                    onChange={(e) => setConfigForm({ ...configForm, paper_account_capital: parseFloat(e.target.value) })}
                    min={100}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2"
                  />
                </div>
              </div>
              <div className="flex gap-3 mt-4">
                <button
                  onClick={() => configMutation.mutate(configForm)}
                  disabled={configMutation.isPending}
                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-sm font-medium"
                >
                  Save Config
                </button>
                <button
                  onClick={() => setShowConfig(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Trading Limits */}
          {tradingStatus && (
            <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
              <h4 className="font-medium mb-3">Live Trading Safety Limits</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-gray-500 text-xs">Max Trade Size</p>
                  <p className="font-mono">${tradingStatus.limits.max_trade_size_usd}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Max Daily Volume</p>
                  <p className="font-mono">${tradingStatus.limits.max_daily_volume}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Max Open Positions</p>
                  <p className="font-mono">{tradingStatus.limits.max_open_positions}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Min Order Size</p>
                  <p className="font-mono">${tradingStatus.limits.min_order_size_usd}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Max Slippage</p>
                  <p className="font-mono">{tradingStatus.limits.max_slippage_percent}%</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ==================== Sub-Components ====================

function MiniStat({
  label,
  value,
  icon,
  subtitle,
  valueColor = 'text-white'
}: {
  label: string
  value: string
  icon: React.ReactNode
  subtitle?: string
  valueColor?: string
}) {
  return (
    <div className="bg-[#141414] border border-gray-800 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <p className="text-xs text-gray-500">{label}</p>
      </div>
      <p className={clsx("text-lg font-semibold font-mono", valueColor)}>{value}</p>
      {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    open: 'bg-blue-500/20 text-blue-400',
    resolved_win: 'bg-green-500/20 text-green-400',
    resolved_loss: 'bg-red-500/20 text-red-400',
    pending: 'bg-yellow-500/20 text-yellow-400',
    executed: 'bg-blue-500/20 text-blue-400',
    failed: 'bg-red-500/20 text-red-400',
    shadow: 'bg-gray-500/20 text-gray-400',
  }
  return (
    <span className={clsx("px-2 py-0.5 rounded text-xs font-medium", colors[status] || 'bg-gray-500/20 text-gray-400')}>
      {status.replace('_', ' ').toUpperCase()}
    </span>
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
  const fillColor = isProfitable ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'

  return (
    <div className="relative h-full">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-full" preserveAspectRatio="none">
        <path d={areaPath} fill={fillColor} />
        <line
          x1="0" y1={zeroY}
          x2={chartWidth} y2={zeroY}
          stroke="#374151"
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
      </svg>

      <div className="absolute top-1 left-2 text-xs text-gray-500">
        ${maxVal.toFixed(0)}
      </div>
      <div className="absolute bottom-1 left-2 text-xs text-gray-500">
        ${minVal.toFixed(0)}
      </div>
      <div className="absolute bottom-1 right-2 flex items-center gap-2 text-xs">
        <span className={isProfitable ? 'text-green-400' : 'text-red-400'}>
          {isProfitable ? '+' : ''}${lastEquity.toFixed(2)}
        </span>
      </div>
      <div className="absolute top-1 right-2 text-xs text-gray-600">
        {points.length > 0 && new Date(points[0].date).toLocaleDateString()} - {points.length > 0 && new Date(points[points.length - 1].date).toLocaleDateString()}
      </div>
    </div>
  )
}
