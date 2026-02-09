import { useState, useEffect } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
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
  RefreshCw,
  Shield,
  Sparkles,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { accountModeAtom, selectedAccountIdAtom } from '../store/atoms'
import {
  Opportunity,
  executeOpportunity,
  executeOpportunityLive,
} from '../services/api'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from './ui/dialog'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Card, CardContent } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import { Separator } from './ui/separator'

interface TradeExecutionModalProps {
  opportunity: Opportunity
  onClose: () => void
}

export default function TradeExecutionModal({ opportunity, onClose }: TradeExecutionModalProps) {
  const [accountMode] = useAtom(accountModeAtom)
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const isLive = accountMode === 'live'
  const [positionSize, setPositionSize] = useState<number>(opportunity.max_position_size)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [takeProfitEnabled, setTakeProfitEnabled] = useState(false)
  const [takeProfitPercent, setTakeProfitPercent] = useState(50)
  const [stopLossEnabled, setStopLossEnabled] = useState(false)
  const [stopLossPercent, setStopLossPercent] = useState(25)
  const [executionResult, setExecutionResult] = useState<{ success: boolean; message: string } | null>(null)
  const queryClient = useQueryClient()

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
  const canExecute = isLive
    ? positionSize > 0 && !isPending
    : !!selectedAccountId && positionSize > 0 && !isPending

  const handleExecute = () => {
    setExecutionResult(null)
    if (isLive) {
      if (!confirm('You are about to execute a LIVE trade with REAL MONEY. Continue?')) return
      liveMutation.mutate()
    } else {
      paperMutation.mutate()
    }
  }

  return (
    <Dialog open={true} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto p-0 gap-0">
        {/* Header */}
        <DialogHeader className="p-5 pb-0">
          <DialogTitle>Execute Trade</DialogTitle>
          <DialogDescription>{opportunity.title}</DialogDescription>
        </DialogHeader>

        <div className="p-5 space-y-5">
          {/* Opportunity Summary */}
          <Card className="border-border bg-muted/50">
            <CardContent className="p-4">
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground">ROI</p>
                  <p className="font-mono font-bold text-green-400">{opportunity.roi_percent.toFixed(2)}%</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Cost / Unit</p>
                  <p className="font-mono text-foreground">${opportunity.total_cost.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Risk</p>
                  <p className={cn("font-mono", opportunity.risk_score < 0.3 ? "text-green-400" : opportunity.risk_score < 0.6 ? "text-yellow-400" : "text-red-400")}>
                    {(opportunity.risk_score * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
              <Separator className="my-3" />
              <p className="text-xs text-muted-foreground mb-1.5">Positions</p>
              {opportunity.positions_to_take.map((pos, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm py-1">
                  <span className="text-muted-foreground">
                    <Badge
                      variant="outline"
                      className={cn(
                        "mr-1.5 text-xs font-medium",
                        pos.outcome === 'YES' ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'
                      )}
                    >
                      {pos.action} {pos.outcome}
                    </Badge>
                    {pos.market.length > 40 ? pos.market.slice(0, 40) + '...' : pos.market}
                  </span>
                  <span className="font-mono text-muted-foreground">${pos.price.toFixed(4)}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* AI Trade Advisor */}
          <AITradeAdvisor opportunity={opportunity} />

          {/* Account Mode Indicator */}
          <div className={cn(
            "rounded-xl p-3 text-sm flex items-center gap-2",
            isLive
              ? "bg-green-500/10 border border-green-500/20 text-green-400"
              : "bg-amber-500/10 border border-amber-500/20 text-amber-400"
          )}>
            {isLive ? <Zap className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
            <span className="font-medium">
              {isLive ? 'Live Trading' : 'Sandbox Trading'}
            </span>
            <span className="text-muted-foreground text-xs ml-auto">
              Change account in the header dropdown
            </span>
          </div>

          {/* Live Mode Warning */}
          {isLive && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-sm text-red-400">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>Live trading uses real money. Make sure the trading client is initialized and funded.</span>
              </div>
            </div>
          )}

          {/* No account warning */}
          {!isLive && !selectedAccountId && (
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-3 text-sm text-yellow-400">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4" />
                No sandbox account selected. Select one from the header dropdown.
              </div>
            </div>
          )}

          {/* Position Size */}
          <div>
            <Label className="block text-xs text-muted-foreground mb-2">
              Position Size (Units)
            </Label>
            <div className="relative">
              <Input
                type="number"
                value={positionSize}
                onChange={(e) => setPositionSize(Math.max(0, Number(e.target.value)))}
                min={0}
                max={opportunity.max_position_size}
                step={10}
                className="w-full bg-muted/50 border-border rounded-xl px-4 py-2.5 text-sm font-mono pr-28"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground/60">
                max: {opportunity.max_position_size.toFixed(0)}
              </div>
            </div>
            {/* Quick size buttons */}
            <div className="flex gap-2 mt-2">
              {[25, 50, 75, 100].map(pct => (
                <Button
                  key={pct}
                  variant="outline"
                  size="sm"
                  onClick={() => setPositionSize(Math.floor(opportunity.max_position_size * pct / 100))}
                  className="flex-1 py-1.5 text-xs text-muted-foreground bg-muted/50 border-border rounded-lg hover:bg-muted hover:text-foreground transition-colors"
                >
                  {pct}%
                </Button>
              ))}
            </div>
          </div>

          {/* Estimated Cost & Profit */}
          <div className="grid grid-cols-2 gap-3">
            <Card className="border-border bg-muted/50">
              <CardContent className="p-3">
                <div className="flex items-center gap-1.5 mb-1">
                  <DollarSign className="w-3.5 h-3.5 text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">Estimated Cost</p>
                </div>
                <p className="font-mono font-semibold text-foreground">${estimatedCost.toFixed(2)}</p>
              </CardContent>
            </Card>
            <Card className="border-border bg-muted/50">
              <CardContent className="p-3">
                <div className="flex items-center gap-1.5 mb-1">
                  <TrendingUp className="w-3.5 h-3.5 text-green-500" />
                  <p className="text-xs text-muted-foreground">Expected Profit</p>
                </div>
                <p className="font-mono font-semibold text-green-400">+${estimatedProfit.toFixed(4)}</p>
              </CardContent>
            </Card>
          </div>

          {/* Advanced Options Toggle */}
          <Button
            variant="ghost"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors w-full justify-start px-0"
          >
            {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            Advanced Options
          </Button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4 bg-muted/50 rounded-xl p-4 border border-border">
              {/* Take Profit */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label className="flex items-center gap-2 text-sm">
                    <Target className="w-4 h-4 text-green-400" />
                    Take Profit (Auto-Sell)
                  </Label>
                  <Switch
                    checked={takeProfitEnabled}
                    onCheckedChange={setTakeProfitEnabled}
                  />
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
                    <p className="text-xs text-muted-foreground/60 mt-1">
                      Auto-sell when position gains {takeProfitPercent}% of entry value
                    </p>
                  </div>
                )}
              </div>

              {/* Stop Loss */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label className="flex items-center gap-2 text-sm">
                    <AlertTriangle className="w-4 h-4 text-red-400" />
                    Stop Loss
                  </Label>
                  <Switch
                    checked={stopLossEnabled}
                    onCheckedChange={setStopLossEnabled}
                  />
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
                    <p className="text-xs text-muted-foreground/60 mt-1">
                      Auto-sell when position loses {stopLossPercent}% of entry value
                    </p>
                  </div>
                )}
              </div>

              <p className="text-xs text-muted-foreground/60 flex items-start gap-1.5">
                <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                Take-profit and stop-loss are monitored while the auto-trader is running. They will trigger sell orders automatically when price thresholds are hit.
              </p>
            </div>
          )}

          {/* Execution Result */}
          {executionResult && (
            <div className={cn(
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
            <Button
              variant="secondary"
              onClick={onClose}
              className="flex-1 py-3 text-sm font-medium rounded-xl"
            >
              Cancel
            </Button>
            <Button
              onClick={handleExecute}
              disabled={!canExecute}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 py-3 text-sm font-semibold rounded-xl transition-all",
                isLive
                  ? "bg-green-500 hover:bg-green-600 text-white"
                  : "bg-amber-500 hover:bg-amber-600 text-white"
              )}
            >
              {isPending ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Executing...
                </>
              ) : (
                <>
                  {isLive ? <Zap className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isLive ? 'Execute Live Trade' : 'Execute Sandbox Trade'}
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// AI Trade Advisor - pre-trade intelligence briefing
function AITradeAdvisor({ opportunity }: { opportunity: Opportunity }) {
  const [showDetails, setShowDetails] = useState(false)

  // Use inline ai_analysis from the opportunity (populated by scanner)
  const inlineAnalysis = opportunity.ai_analysis
  const judgment = inlineAnalysis && inlineAnalysis.recommendation !== 'pending' ? inlineAnalysis : null
  const resolutions = inlineAnalysis?.resolution_analyses || []
  const hasData = judgment || resolutions.length > 0
  const isLoading = inlineAnalysis?.recommendation === 'pending'

  // Compute overall signal
  const getSignal = () => {
    if (!judgment) return null
    if (judgment.overall_score >= 0.65) return { label: 'GO', color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/20' }
    if (judgment.overall_score >= 0.45) return { label: 'REVIEW', color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20' }
    return { label: 'CAUTION', color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20' }
  }
  const signal = getSignal()

  return (
    <div className={cn(
      'rounded-xl border transition-colors',
      hasData
        ? signal?.bg || 'bg-purple-500/5 border-purple-500/20'
        : 'bg-muted border-border'
    )}>
      <Button
        variant="ghost"
        onClick={() => setShowDetails(!showDetails)}
        className="w-full flex items-center gap-3 px-4 py-3 h-auto hover:bg-transparent"
      >
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center flex-shrink-0">
          <Sparkles className="w-4 h-4 text-purple-400" />
        </div>
        <div className="flex-1 text-left">
          <p className="text-xs font-medium text-foreground/80">AI Trade Advisor</p>
          {isLoading ? (
            <p className="text-[10px] text-muted-foreground flex items-center gap-1">
              <RefreshCw className="w-3 h-3 animate-spin" /> Checking intelligence...
            </p>
          ) : hasData ? (
            <p className="text-[10px] text-muted-foreground">
              {signal && <span className={cn('font-bold mr-1', signal.color)}>{signal.label}</span>}
              {judgment && <span>Score: {(judgment.overall_score * 100).toFixed(0)}/100</span>}
              {resolutions.length > 0 && <span> | Resolution: {resolutions[0].recommendation}</span>}
            </p>
          ) : (
            <p className="text-[10px] text-muted-foreground">No cached AI analysis. Expand for details.</p>
          )}
        </div>
        {hasData && (
          signal?.label === 'GO' ? (
            <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
          ) : signal?.label === 'CAUTION' ? (
            <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
          ) : (
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
          )
        )}
        {showDetails ? (
          <ChevronUp className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        )}
      </Button>

      {showDetails && hasData && (
        <div className="px-4 pb-3 space-y-2">
          {judgment && (
            <div className="grid grid-cols-4 gap-2">
              {[
                { label: 'Profit', value: judgment.profit_viability },
                { label: 'Resolution', value: judgment.resolution_safety },
                { label: 'Execution', value: judgment.execution_feasibility },
                { label: 'Efficiency', value: judgment.market_efficiency },
              ].map((item) => {
                const color = item.value >= 0.7 ? 'text-green-400' : item.value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                return (
                  <div key={item.label} className="text-center bg-muted/50 rounded-lg py-1.5">
                    <p className="text-[10px] text-muted-foreground">{item.label}</p>
                    <p className={cn('text-sm font-bold', color)}>
                      {(item.value * 100).toFixed(0)}
                    </p>
                  </div>
                )
              })}
            </div>
          )}
          {judgment?.reasoning && (
            <p className="text-xs text-muted-foreground bg-muted/50 rounded-lg p-2">
              {judgment.reasoning}
            </p>
          )}
          {resolutions.map((r: any, i: number) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <Shield className="w-3 h-3 text-muted-foreground" />
              <Badge
                variant="outline"
                className={cn(
                  'text-xs font-medium',
                  r.recommendation === 'safe' ? 'bg-green-500/10 text-green-400 border-green-500/30' :
                  r.recommendation === 'caution' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30' :
                  'bg-red-500/10 text-red-400 border-red-500/30'
                )}
              >
                {r.recommendation}
              </Badge>
              <span className="text-muted-foreground">
                Clarity: {(r.clarity_score * 100).toFixed(0)}% | Risk: {(r.risk_score * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
