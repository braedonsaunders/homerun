import { useState } from 'react'
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
  Power,
  ShieldAlert
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
  AutoTraderStatus,
  AutoTraderTrade
} from '../services/api'

export default function TradingPanel() {
  const [showConfig, setShowConfig] = useState(false)
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
    queryFn: () => getAutoTraderTrades(50),
    refetchInterval: 10000,
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

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
      </div>
    )
  }

  const stats = status?.stats
  const config = status?.config

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

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total Trades"
          value={stats?.total_trades?.toString() || '0'}
          icon={<Activity className="w-5 h-5 text-blue-500" />}
        />
        <StatCard
          label="Win Rate"
          value={`${((stats?.win_rate || 0) * 100).toFixed(1)}%`}
          icon={<TrendingUp className="w-5 h-5 text-green-500" />}
        />
        <StatCard
          label="Total P/L"
          value={`$${(stats?.total_profit || 0).toFixed(2)}`}
          valueColor={(stats?.total_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
          icon={<DollarSign className="w-5 h-5 text-yellow-500" />}
        />
        <StatCard
          label="ROI"
          value={`${(stats?.roi_percent || 0).toFixed(2)}%`}
          valueColor={(stats?.roi_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
          icon={<TrendingUp className="w-5 h-5 text-purple-500" />}
        />
      </div>

      {/* Daily Stats */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <h3 className="font-medium mb-3">Today's Activity</h3>
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-500">Trades</p>
            <p className="text-lg font-mono">{stats?.daily_trades || 0}</p>
          </div>
          <div>
            <p className="text-gray-500">P/L</p>
            <p className={clsx(
              "text-lg font-mono",
              (stats?.daily_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'
            )}>
              ${(stats?.daily_profit || 0).toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Executed</p>
            <p className="text-lg font-mono">{stats?.opportunities_executed || 0}</p>
          </div>
          <div>
            <p className="text-gray-500">Skipped</p>
            <p className="text-lg font-mono">{stats?.opportunities_skipped || 0}</p>
          </div>
        </div>
      </div>

      {/* Config Panel */}
      {showConfig && (
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <h3 className="font-medium mb-4">Trading Configuration</h3>
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
              <p className="text-xs text-gray-600 mt-1">Starting capital for paper trading simulation</p>
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

      {/* Current Config Display */}
      {!showConfig && config && (
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <h3 className="font-medium mb-3">Current Settings</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Min ROI</p>
              <p className="font-mono">{config.min_roi_percent}%</p>
            </div>
            <div>
              <p className="text-gray-500">Max Risk</p>
              <p className="font-mono">{config.max_risk_score}</p>
            </div>
            <div>
              <p className="text-gray-500">Position Size</p>
              <p className="font-mono">${config.base_position_size_usd} - ${config.max_position_size_usd}</p>
            </div>
            <div>
              <p className="text-gray-500">Daily Limits</p>
              <p className="font-mono">{config.max_daily_trades} trades / ${config.max_daily_loss_usd} loss</p>
            </div>
            <div>
              <p className="text-gray-500">Strategies</p>
              <p className="font-mono text-xs">{config.enabled_strategies?.join(', ')}</p>
            </div>
            <div>
              <p className="text-gray-500">Circuit Breaker</p>
              <p className="font-mono">{config.circuit_breaker_losses} losses</p>
            </div>
            <div>
              <p className="text-gray-500">Paper Capital</p>
              <p className="font-mono">${config.paper_account_capital?.toLocaleString() || '10,000'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Recent Trades */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <h3 className="font-medium mb-4">Recent Trades</h3>
        {trades.length === 0 ? (
          <p className="text-gray-500 text-center py-4">No trades yet</p>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {trades.map((trade) => (
              <TradeRow key={trade.id} trade={trade} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({
  label,
  value,
  icon,
  valueColor
}: {
  label: string
  value: string
  icon: React.ReactNode
  valueColor?: string
}) {
  return (
    <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-[#1a1a1a] rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-gray-500">{label}</p>
          <p className={clsx("text-lg font-semibold font-mono", valueColor)}>{value}</p>
        </div>
      </div>
    </div>
  )
}

function TradeRow({ trade }: { trade: AutoTraderTrade }) {
  const statusColors: Record<string, string> = {
    open: 'bg-blue-500/20 text-blue-400',
    pending: 'bg-yellow-500/20 text-yellow-400',
    resolved_win: 'bg-green-500/20 text-green-400',
    resolved_loss: 'bg-red-500/20 text-red-400',
    failed: 'bg-red-500/20 text-red-400',
    shadow: 'bg-gray-500/20 text-gray-400'
  }

  const modeColors: Record<string, string> = {
    paper: 'text-blue-400',
    live: 'text-green-400',
    shadow: 'text-gray-400'
  }

  return (
    <div className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
      <div className="flex items-center gap-3">
        {trade.actual_profit !== null && trade.actual_profit >= 0 ? (
          <TrendingUp className="w-4 h-4 text-green-400" />
        ) : trade.actual_profit !== null ? (
          <TrendingDown className="w-4 h-4 text-red-400" />
        ) : (
          <Activity className="w-4 h-4 text-gray-400" />
        )}
        <div>
          <p className="text-sm">{trade.strategy}</p>
          <p className="text-xs text-gray-500">
            Cost: ${trade.total_cost.toFixed(2)} |{' '}
            <span className={modeColors[trade.mode] || 'text-gray-400'}>
              {trade.mode.toUpperCase()}
            </span>
          </p>
        </div>
      </div>
      <div className="text-right">
        <span className={clsx("px-2 py-0.5 rounded text-xs", statusColors[trade.status] || 'bg-gray-500/20')}>
          {trade.status.replace('_', ' ')}
        </span>
        {trade.actual_profit !== null && (
          <p className={clsx(
            "text-sm font-mono mt-1",
            trade.actual_profit >= 0 ? 'text-green-400' : 'text-red-400'
          )}>
            {trade.actual_profit >= 0 ? '+' : ''}${trade.actual_profit.toFixed(2)}
          </p>
        )}
        {trade.actual_profit === null && (
          <p className="text-sm font-mono mt-1 text-gray-400">
            +${trade.expected_profit.toFixed(2)} exp
          </p>
        )}
      </div>
    </div>
  )
}
