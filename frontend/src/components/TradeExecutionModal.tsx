import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  X,
  Play,
  AlertTriangle,
  DollarSign,
  TrendingUp,
  Target,
  Zap,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Info,
  RefreshCw
} from 'lucide-react'
import clsx from 'clsx'
import {
  Opportunity,
  getSimulationAccounts,
  executeOpportunity,
  executeOpportunityLive,
} from '../services/api'

interface TradeExecutionModalProps {
  opportunity: Opportunity
  onClose: () => void
}

export default function TradeExecutionModal({ opportunity, onClose }: TradeExecutionModalProps) {
  const [mode, setMode] = useState<'paper' | 'live'>('paper')
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null)
  const [positionSize, setPositionSize] = useState<number>(opportunity.max_position_size)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [takeProfitEnabled, setTakeProfitEnabled] = useState(false)
  const [takeProfitPercent, setTakeProfitPercent] = useState(50)
  const [stopLossEnabled, setStopLossEnabled] = useState(false)
  const [stopLossPercent, setStopLossPercent] = useState(25)
  const [executionResult, setExecutionResult] = useState<{ success: boolean; message: string } | null>(null)
  const queryClient = useQueryClient()

  const { data: accounts = [] } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    enabled: mode === 'paper',
  })

  // Auto-select first account
  useEffect(() => {
    if (accounts.length > 0 && !selectedAccountId) {
      setSelectedAccountId(accounts[0].id)
    }
  }, [accounts, selectedAccountId])

  // Clamp position size to max
  useEffect(() => {
    if (positionSize > opportunity.max_position_size) {
      setPositionSize(opportunity.max_position_size)
    }
  }, [opportunity.max_position_size, positionSize])

  const estimatedCost = opportunity.total_cost * positionSize
  const estimatedProfit = opportunity.net_profit * positionSize

  const paperMutation = useMutation({
    mutationFn: () => {
      if (!selectedAccountId) throw new Error('No account selected')
      // Calculate take-profit/stop-loss as price thresholds from entry
      const avgEntryPrice = opportunity.positions_to_take.length > 0
        ? opportunity.positions_to_take.reduce((sum, p) => sum + p.price, 0) / opportunity.positions_to_take.length
        : 0.5
      const tp = takeProfitEnabled
        ? Math.min(0.99, avgEntryPrice * (1 + takeProfitPercent / 100))
        : undefined
      const sl = stopLossEnabled
        ? Math.max(0.01, avgEntryPrice * (1 - stopLossPercent / 100))
        : undefined
      return executeOpportunity(
        selectedAccountId,
        opportunity.id,
        positionSize > 0 ? positionSize : undefined,
        tp,
        sl
      )
    },
    onSuccess: (data) => {
      setExecutionResult({
        success: true,
        message: `Trade executed! Cost: $${data.total_cost?.toFixed(2) || estimatedCost.toFixed(2)}, Expected profit: $${data.expected_profit?.toFixed(4) || estimatedProfit.toFixed(4)}`
      })
      queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
      queryClient.invalidateQueries({ queryKey: ['account-trades'] })
    },
    onError: (error: any) => {
      setExecutionResult({
        success: false,
        message: error?.response?.data?.detail || error?.message || 'Failed to execute trade'
      })
    }
  })

  const liveMutation = useMutation({
    mutationFn: () => {
      return executeOpportunityLive({
        opportunity_id: opportunity.id,
        positions: opportunity.positions_to_take,
        size_usd: positionSize,
      })
    },
    onSuccess: (data) => {
      setExecutionResult({
        success: true,
        message: `Live trade executed! ${data.message || 'Orders placed successfully.'}`
      })
      queryClient.invalidateQueries({ queryKey: ['trading-positions'] })
    },
    onError: (error: any) => {
      setExecutionResult({
        success: false,
        message: error?.response?.data?.detail || error?.message || 'Failed to execute live trade'
      })
    }
  })

  const isPending = paperMutation.isPending || liveMutation.isPending
  const canExecute = mode === 'paper'
    ? !!selectedAccountId && positionSize > 0 && !isPending
    : positionSize > 0 && !isPending

  const handleExecute = () => {
    setExecutionResult(null)
    if (mode === 'paper') {
      paperMutation.mutate()
    } else {
      if (!confirm('You are about to execute a LIVE trade with REAL MONEY. Continue?')) return
      liveMutation.mutate()
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a1a] border border-gray-800 rounded-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-gray-800">
          <div>
            <h2 className="text-lg font-bold text-white">Execute Trade</h2>
            <p className="text-xs text-gray-500 mt-0.5">{opportunity.title}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        <div className="p-5 space-y-5">
          {/* Opportunity Summary */}
          <div className="bg-black/30 rounded-xl p-4">
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div>
                <p className="text-xs text-gray-500">ROI</p>
                <p className="font-mono font-bold text-green-400">{opportunity.roi_percent.toFixed(2)}%</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Cost / Unit</p>
                <p className="font-mono text-white">${opportunity.total_cost.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Risk</p>
                <p className={clsx("font-mono", opportunity.risk_score < 0.3 ? "text-green-400" : opportunity.risk_score < 0.6 ? "text-yellow-400" : "text-red-400")}>
                  {(opportunity.risk_score * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-gray-700">
              <p className="text-xs text-gray-500 mb-1.5">Positions</p>
              {opportunity.positions_to_take.map((pos, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm py-1">
                  <span className="text-gray-300">
                    <span className={clsx(
                      "px-1.5 py-0.5 rounded text-xs font-medium mr-1.5",
                      pos.outcome === 'YES' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    )}>
                      {pos.action} {pos.outcome}
                    </span>
                    {pos.market.length > 40 ? pos.market.slice(0, 40) + '...' : pos.market}
                  </span>
                  <span className="font-mono text-gray-400">${pos.price.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Mode Toggle */}
          <div>
            <label className="block text-xs text-gray-500 mb-2">Trading Mode</label>
            <div className="flex gap-2">
              <button
                onClick={() => setMode('paper')}
                className={clsx(
                  "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-all",
                  mode === 'paper'
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "bg-black/20 text-gray-500 border border-gray-800 hover:border-gray-700"
                )}
              >
                <Play className="w-4 h-4" />
                Paper Trading
              </button>
              <button
                onClick={() => setMode('live')}
                className={clsx(
                  "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-all",
                  mode === 'live'
                    ? "bg-green-500/20 text-green-400 border border-green-500/30"
                    : "bg-black/20 text-gray-500 border border-gray-800 hover:border-gray-700"
                )}
              >
                <Zap className="w-4 h-4" />
                Live Trading
              </button>
            </div>
          </div>

          {/* Paper Account Selector */}
          {mode === 'paper' && (
            <div>
              <label className="block text-xs text-gray-500 mb-2">Paper Account</label>
              {accounts.length === 0 ? (
                <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-3 text-sm text-yellow-400">
                  <div className="flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    No paper accounts. Create one in the Paper Trading tab first.
                  </div>
                </div>
              ) : (
                <select
                  value={selectedAccountId || ''}
                  onChange={(e) => setSelectedAccountId(e.target.value)}
                  className="w-full bg-black/30 border border-gray-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-blue-500/50"
                >
                  {accounts.map((acc) => (
                    <option key={acc.id} value={acc.id}>
                      {acc.name} — ${(acc.current_capital ?? 0).toFixed(2)} available
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          {/* Live Mode Warning */}
          {mode === 'live' && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-sm text-red-400">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>Live trading uses real money. Make sure the trading client is initialized and funded.</span>
              </div>
            </div>
          )}

          {/* Position Size */}
          <div>
            <label className="block text-xs text-gray-500 mb-2">
              Position Size (Units)
            </label>
            <div className="relative">
              <input
                type="number"
                value={positionSize}
                onChange={(e) => setPositionSize(Math.max(0, Number(e.target.value)))}
                min={0}
                max={opportunity.max_position_size}
                step={10}
                className="w-full bg-black/30 border border-gray-700 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-blue-500/50"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-600">
                max: {opportunity.max_position_size.toFixed(0)}
              </div>
            </div>
            {/* Quick size buttons */}
            <div className="flex gap-2 mt-2">
              {[25, 50, 75, 100].map(pct => (
                <button
                  key={pct}
                  onClick={() => setPositionSize(Math.floor(opportunity.max_position_size * pct / 100))}
                  className="flex-1 py-1.5 text-xs text-gray-400 bg-black/20 border border-gray-800 rounded-lg hover:bg-gray-800 hover:text-gray-300 transition-colors"
                >
                  {pct}%
                </button>
              ))}
            </div>
          </div>

          {/* Estimated Cost & Profit */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-black/30 rounded-xl p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <DollarSign className="w-3.5 h-3.5 text-gray-500" />
                <p className="text-xs text-gray-500">Estimated Cost</p>
              </div>
              <p className="font-mono font-semibold text-white">${estimatedCost.toFixed(2)}</p>
            </div>
            <div className="bg-black/30 rounded-xl p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <TrendingUp className="w-3.5 h-3.5 text-green-500" />
                <p className="text-xs text-gray-500">Expected Profit</p>
              </div>
              <p className="font-mono font-semibold text-green-400">+${estimatedProfit.toFixed(4)}</p>
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-300 transition-colors w-full"
          >
            {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            Advanced Options
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4 bg-black/20 rounded-xl p-4 border border-gray-800">
              {/* Take Profit */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="flex items-center gap-2 text-sm text-gray-300">
                    <Target className="w-4 h-4 text-green-400" />
                    Take Profit (Auto-Sell)
                  </label>
                  <button
                    onClick={() => setTakeProfitEnabled(!takeProfitEnabled)}
                    className={clsx(
                      "w-10 h-5 rounded-full transition-colors relative",
                      takeProfitEnabled ? "bg-green-500" : "bg-gray-700"
                    )}
                  >
                    <div className={clsx(
                      "w-4 h-4 rounded-full bg-white absolute top-0.5 transition-all",
                      takeProfitEnabled ? "left-5.5 right-0.5" : "left-0.5"
                    )} style={{ left: takeProfitEnabled ? '22px' : '2px' }} />
                  </button>
                </div>
                {takeProfitEnabled && (
                  <div>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={5}
                        max={200}
                        step={5}
                        value={takeProfitPercent}
                        onChange={(e) => setTakeProfitPercent(Number(e.target.value))}
                        className="flex-1 accent-green-500"
                      />
                      <span className="text-sm font-mono text-green-400 w-16 text-right">+{takeProfitPercent}%</span>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      Auto-sell when position gains {takeProfitPercent}% of entry value
                    </p>
                  </div>
                )}
              </div>

              {/* Stop Loss */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="flex items-center gap-2 text-sm text-gray-300">
                    <AlertTriangle className="w-4 h-4 text-red-400" />
                    Stop Loss
                  </label>
                  <button
                    onClick={() => setStopLossEnabled(!stopLossEnabled)}
                    className={clsx(
                      "w-10 h-5 rounded-full transition-colors relative",
                      stopLossEnabled ? "bg-red-500" : "bg-gray-700"
                    )}
                  >
                    <div className={clsx(
                      "w-4 h-4 rounded-full bg-white absolute top-0.5 transition-all",
                    )} style={{ left: stopLossEnabled ? '22px' : '2px' }} />
                  </button>
                </div>
                {stopLossEnabled && (
                  <div>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={5}
                        max={80}
                        step={5}
                        value={stopLossPercent}
                        onChange={(e) => setStopLossPercent(Number(e.target.value))}
                        className="flex-1 accent-red-500"
                      />
                      <span className="text-sm font-mono text-red-400 w-16 text-right">-{stopLossPercent}%</span>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      Auto-sell when position loses {stopLossPercent}% of entry value
                    </p>
                  </div>
                )}
              </div>

              <p className="text-xs text-gray-600 flex items-start gap-1.5">
                <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                Take-profit and stop-loss are monitored while the auto-trader is running. They will trigger sell orders automatically when price thresholds are hit.
              </p>
            </div>
          )}

          {/* Execution Result */}
          {executionResult && (
            <div className={clsx(
              "rounded-xl p-3 text-sm flex items-start gap-2",
              executionResult.success
                ? "bg-green-500/10 border border-green-500/20 text-green-400"
                : "bg-red-500/10 border border-red-500/20 text-red-400"
            )}>
              {executionResult.success ? (
                <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
              ) : (
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              )}
              <span>{executionResult.message}</span>
            </div>
          )}

          {/* Execute Button */}
          <div className="flex gap-3 pt-2">
            <button
              onClick={onClose}
              className="flex-1 py-3 text-sm font-medium text-gray-400 bg-gray-800 hover:bg-gray-700 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleExecute}
              disabled={!canExecute}
              className={clsx(
                "flex-1 flex items-center justify-center gap-2 py-3 text-sm font-semibold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed",
                mode === 'paper'
                  ? "bg-blue-500 hover:bg-blue-600 text-white"
                  : "bg-green-500 hover:bg-green-600 text-white"
              )}
            >
              {isPending ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Executing...
                </>
              ) : (
                <>
                  {mode === 'paper' ? <Play className="w-4 h-4" /> : <Zap className="w-4 h-4" />}
                  {mode === 'paper' ? 'Execute Paper Trade' : 'Execute Live Trade'}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

