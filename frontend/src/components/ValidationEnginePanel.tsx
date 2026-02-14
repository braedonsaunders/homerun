import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle,
  Loader2,
  Play,
  RefreshCw,
  ShieldAlert,
  SlidersHorizontal,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Progress } from './ui/progress'
import { Switch } from './ui/switch'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import {
  activateValidationParameterSet,
  cancelValidationJob,
  clearValidationStrategyOverride,
  evaluateValidationGuardrails,
  getOptimizationResults,
  getValidationGuardrailConfig,
  getValidationJobs,
  getValidationOverview,
  getValidationParameterSets,
  overrideValidationStrategy,
  runValidationBacktest,
  runValidationOptimization,
  updateValidationGuardrailConfig,
} from '../services/api'

type ValidationSubTab = 'runs' | 'strategy' | 'guardrails' | 'sets'
type Notice = { type: 'success' | 'error'; text: string } | null

function intOr(value: string, fallback: number): number {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function floatOr(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function getErrorMessage(error: unknown, fallback: string): string {
  const detail = (error as any)?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (error instanceof Error && error.message.trim()) return error.message
  return fallback
}

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseFloat(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function formatCurrency(value: number | null, compact = false): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 1 : 2,
    minimumFractionDigits: compact ? 0 : 2,
  }).format(value)
}

function formatPercentFromRatio(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  return `${(value * 100).toFixed(1)}%`
}

function formatSignedPercent(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  const prefix = value >= 0 ? '+' : '-'
  return `${prefix}${Math.abs(value).toFixed(2)}%`
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString()
}

function getJobStatusClass(status: string): string {
  switch (status) {
    case 'completed':
      return 'bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/35'
    case 'running':
      return 'bg-cyan-100 text-cyan-800 border-cyan-300 dark:bg-cyan-500/15 dark:text-cyan-300 dark:border-cyan-500/35'
    case 'queued':
      return 'bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-500/15 dark:text-amber-300 dark:border-amber-500/35'
    case 'failed':
      return 'bg-red-100 text-red-800 border-red-300 dark:bg-red-500/15 dark:text-red-300 dark:border-red-500/35'
    case 'cancelled':
      return 'bg-zinc-100 text-zinc-800 border-zinc-300 dark:bg-zinc-500/15 dark:text-zinc-300 dark:border-zinc-500/35'
    default:
      return 'bg-muted text-muted-foreground border-border'
  }
}

function getStrategyStatusClass(status: string): string {
  if (status === 'demoted') {
    return 'bg-red-100 text-red-800 border-red-300 dark:bg-red-500/15 dark:text-red-300 dark:border-red-500/35'
  }
  if (status === 'active') {
    return 'bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/35'
  }
  return 'bg-muted text-muted-foreground border-border'
}

export default function ValidationEnginePanel() {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<ValidationSubTab>('runs')
  const [notice, setNotice] = useState<Notice>(null)

  const [backtestSetName, setBacktestSetName] = useState('')
  const [saveBacktestSet, setSaveBacktestSet] = useState(false)
  const [activateBacktestSet, setActivateBacktestSet] = useState(false)

  const [method, setMethod] = useState<'grid' | 'random'>('grid')
  const [walkForward, setWalkForward] = useState(true)
  const [randomSamples, setRandomSamples] = useState(200)
  const [topK, setTopK] = useState(20)
  const [saveBest, setSaveBest] = useState(true)
  const [bestSetName, setBestSetName] = useState('')

  const [guardrailsDirty, setGuardrailsDirty] = useState(false)
  const [guardrailsForm, setGuardrailsForm] = useState({
    enabled: true,
    min_samples: 25,
    min_directional_accuracy: 0.52,
    max_mae_roi: 12,
    lookback_days: 90,
    auto_promote: true,
  })

  const {
    data: overview,
    isLoading: loadingOverview,
    isFetching: fetchingOverview,
    refetch: refetchOverview,
  } = useQuery({
    queryKey: ['validation-overview'],
    queryFn: getValidationOverview,
    refetchInterval: 20000,
  })

  const {
    data: parameterSets,
    isFetching: fetchingParameterSets,
    refetch: refetchParameterSets,
  } = useQuery({
    queryKey: ['validation-parameter-sets'],
    queryFn: getValidationParameterSets,
    refetchInterval: 30000,
  })

  const {
    data: jobsData,
    isFetching: fetchingJobs,
    refetch: refetchJobs,
  } = useQuery({
    queryKey: ['validation-jobs'],
    queryFn: () => getValidationJobs(60),
    refetchInterval: 3000,
  })

  const {
    data: guardrailsConfig,
    isFetching: fetchingGuardrails,
    refetch: refetchGuardrails,
  } = useQuery({
    queryKey: ['validation-guardrails-config'],
    queryFn: getValidationGuardrailConfig,
    refetchInterval: 30000,
  })

  const {
    data: optimizationResults,
    isFetching: fetchingOptimizationResults,
    refetch: refetchOptimizationResults,
  } = useQuery({
    queryKey: ['validation-optimization-results'],
    queryFn: () => getOptimizationResults(15),
    refetchInterval: 45000,
  })

  useEffect(() => {
    if (guardrailsConfig && !guardrailsDirty) {
      setGuardrailsForm({
        enabled: guardrailsConfig.enabled,
        min_samples: guardrailsConfig.min_samples,
        min_directional_accuracy: guardrailsConfig.min_directional_accuracy,
        max_mae_roi: guardrailsConfig.max_mae_roi,
        lookback_days: guardrailsConfig.lookback_days,
        auto_promote: guardrailsConfig.auto_promote,
      })
    }
  }, [guardrailsConfig, guardrailsDirty])

  useEffect(() => {
    if (!notice) return
    const timeoutId = window.setTimeout(() => setNotice(null), 3500)
    return () => window.clearTimeout(timeoutId)
  }, [notice])

  const backtestMutation = useMutation({
    mutationFn: runValidationBacktest,
    onSuccess: (data) => {
      setNotice({ type: 'success', text: `Backtest queued (${data.job_id})` })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to queue backtest') })
    },
  })

  const optimizeMutation = useMutation({
    mutationFn: runValidationOptimization,
    onSuccess: (data) => {
      setNotice({ type: 'success', text: `Optimization queued (${data.job_id})` })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['validation-optimization-results'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to queue optimization') })
    },
  })

  const activateMutation = useMutation({
    mutationFn: activateValidationParameterSet,
    onSuccess: () => {
      setNotice({ type: 'success', text: 'Parameter set activated' })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-parameter-sets'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to activate parameter set') })
    },
  })

  const cancelJobMutation = useMutation({
    mutationFn: cancelValidationJob,
    onSuccess: () => {
      setNotice({ type: 'success', text: 'Job cancelled' })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to cancel job') })
    },
  })

  const saveGuardrailsMutation = useMutation({
    mutationFn: updateValidationGuardrailConfig,
    onSuccess: () => {
      setNotice({ type: 'success', text: 'Guardrails saved' })
      setGuardrailsDirty(false)
      queryClient.invalidateQueries({ queryKey: ['validation-guardrails-config'] })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to save guardrails') })
    },
  })

  const evaluateGuardrailsMutation = useMutation({
    mutationFn: evaluateValidationGuardrails,
    onSuccess: () => {
      setNotice({ type: 'success', text: 'Guardrails evaluated' })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to evaluate guardrails') })
    },
  })

  const overrideStrategyMutation = useMutation({
    mutationFn: ({
      strategyType,
      status,
    }: {
      strategyType: string
      status: 'active' | 'demoted'
    }) => overrideValidationStrategy(strategyType, status),
    onSuccess: (_, vars) => {
      setNotice({ type: 'success', text: `${vars.strategyType} set to ${vars.status}` })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to override strategy') })
    },
  })

  const clearOverrideMutation = useMutation({
    mutationFn: clearValidationStrategyOverride,
    onSuccess: (_, strategyType) => {
      setNotice({ type: 'success', text: `Override cleared for ${strategyType}` })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
    onError: (error) => {
      setNotice({ type: 'error', text: getErrorMessage(error, 'Failed to clear override') })
    },
  })

  const isRefreshing =
    fetchingOverview || fetchingJobs || fetchingGuardrails || fetchingParameterSets || fetchingOptimizationResults

  const calibration = overview?.calibration_90d?.overall
  const directionalAccuracy = toNumber(calibration?.directional_accuracy)
  const combinatorialAccuracy = toNumber((overview?.combinatorial_validation as Record<string, unknown> | undefined)?.accuracy)

  const trendTail = useMemo(
    () => (overview?.calibration_trend_90d || []).slice(-12),
    [overview?.calibration_trend_90d]
  )

  const strategyHealth = useMemo(() => {
    const rows = [...(overview?.strategy_health || [])]
    rows.sort((a, b) => {
      if (a.status === b.status) return b.sample_size - a.sample_size
      if (a.status === 'demoted') return -1
      if (b.status === 'demoted') return 1
      return a.strategy_type.localeCompare(b.strategy_type)
    })
    return rows
  }, [overview?.strategy_health])

  const demotedCount = strategyHealth.filter((row) => row.status === 'demoted').length

  const jobs = useMemo(() => jobsData?.jobs || [], [jobsData?.jobs])
  const runningJobs = jobs.filter((job) => job.status === 'running').length
  const queuedJobs = jobs.filter((job) => job.status === 'queued').length
  const failedJobs = jobs.filter((job) => job.status === 'failed').length

  const executionFailureRate = toNumber(overview?.trader_orchestrator_execution_30d?.failure_rate)
  const resolverTradableRate = toNumber(overview?.world_intel_resolver_7d?.tradable_rate)
  const executionSample = toNumber(overview?.trader_orchestrator_execution_30d?.sample_size)

  const executionBySource = useMemo(() => {
    const rows = overview?.trader_orchestrator_execution_30d?.by_source || []
    return rows
      .map((entry) => {
        const record = entry as Record<string, unknown>
        const total = toNumber(record.total) || 0
        const failed = toNumber(record.failed) || 0
        return {
          source: String(record.source || 'unknown'),
          total,
          failed,
          failureRate: total > 0 ? failed / total : 0,
          notional: toNumber(record.notional_total_usd) || 0,
          pnl: toNumber(record.realized_pnl_total) || 0,
        }
      })
      .sort((left, right) => right.total - left.total)
      .slice(0, 8)
  }, [overview?.trader_orchestrator_execution_30d?.by_source])

  const resolverByType = useMemo(() => {
    const rows = overview?.world_intel_resolver_7d?.by_signal_type || []
    return rows
      .map((entry) => {
        const record = entry as Record<string, unknown>
        return {
          signalType: String(record.signal_type || 'unknown'),
          candidates: toNumber(record.candidates) || 0,
          tradable: toNumber(record.tradable) || 0,
          tradableRate: toNumber(record.tradable_rate) || 0,
        }
      })
      .sort((left, right) => right.candidates - left.candidates)
      .slice(0, 8)
  }, [overview?.world_intel_resolver_7d?.by_signal_type])

  const activeSetLabel = useMemo(() => {
    const activeSet = overview?.active_parameter_set
    if (!activeSet) return 'None'
    const activeName = (activeSet as Record<string, unknown>).name
    const activeId = (activeSet as Record<string, unknown>).id
    if (typeof activeName === 'string' && activeName.trim()) return activeName
    if (typeof activeId === 'string' && activeId.trim()) return activeId
    return 'Active set'
  }, [overview?.active_parameter_set])

  const latestOptimizationScore = useMemo(() => {
    const latest = overview?.latest_optimization as Record<string, unknown> | null
    if (!latest) return null
    return toNumber(latest.score)
  }, [overview?.latest_optimization])

  const refreshAll = async () => {
    await Promise.all([
      refetchOverview(),
      refetchJobs(),
      refetchGuardrails(),
      refetchParameterSets(),
      refetchOptimizationResults(),
    ])
  }

  if (loadingOverview) {
    return (
      <div className="flex justify-center py-10">
        <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Card className="overflow-hidden border-border/80 bg-card/80">
        <CardContent className="p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-widest text-cyan-800 dark:text-cyan-200">Validation Engine</p>
              <h3 className="mt-1 text-base font-semibold">Model Quality Operations</h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Manage backtests, optimization jobs, strategy guardrails, and deployment parameter sets.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="border-border/70 bg-background/40 text-[10px] text-muted-foreground">
                Auto refresh: jobs 3s, overview 20s
              </Badge>
              <Button
                variant="secondary"
                size="sm"
                onClick={refreshAll}
                disabled={isRefreshing}
              >
                <RefreshCw className={cn('mr-1.5 h-3.5 w-3.5', isRefreshing && 'animate-spin')} />
                Refresh
              </Button>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8 card-stagger">
            <SnapshotStat
              label="Calibration Sample"
              value={String(overview?.calibration_90d?.sample_size ?? 0)}
            />
            <SnapshotStat
              label="Directional Accuracy"
              value={formatPercentFromRatio(directionalAccuracy)}
              tone={directionalAccuracy != null && directionalAccuracy >= guardrailsForm.min_directional_accuracy ? 'good' : 'warn'}
            />
            <SnapshotStat
              label="Combinatorial"
              value={formatPercentFromRatio(combinatorialAccuracy)}
              tone={combinatorialAccuracy != null && combinatorialAccuracy >= 0.5 ? 'good' : 'warn'}
            />
            <SnapshotStat
              label="Demoted Strategies"
              value={String(demotedCount)}
              tone={demotedCount > 0 ? 'warn' : 'good'}
            />
            <SnapshotStat
              label="Execution Failure"
              value={formatPercentFromRatio(executionFailureRate)}
              tone={executionFailureRate != null && executionFailureRate <= 0.1 ? 'good' : 'warn'}
            />
            <SnapshotStat
              label="Resolver Tradable"
              value={formatPercentFromRatio(resolverTradableRate)}
              tone={resolverTradableRate != null && resolverTradableRate >= 0.2 ? 'good' : 'neutral'}
            />
            <SnapshotStat
              label="Running Jobs"
              value={String(runningJobs)}
              tone={runningJobs > 0 ? 'info' : 'neutral'}
            />
            <SnapshotStat
              label="Active Set"
              value={activeSetLabel}
              compact
            />
          </div>
        </CardContent>
      </Card>

      {notice && (
        <div
          className={cn(
            'rounded-lg border px-3 py-2 text-xs',
            notice.type === 'success'
              ? 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200'
              : 'border-red-300 bg-red-100 text-red-900 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200'
          )}
        >
          {notice.text}
        </div>
      )}

      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as ValidationSubTab)} className="space-y-4">
        <TabsList className="h-auto w-full justify-start gap-2 rounded-xl border border-border/60 bg-card/70 p-1.5">
          <TabsTrigger value="runs" className="gap-1.5 data-[state=active]:bg-cyan-100 data-[state=active]:text-cyan-900 dark:data-[state=active]:bg-cyan-500/15 dark:data-[state=active]:text-cyan-100">
            Runs & Queue
            <Badge variant="outline" className="ml-1 border-cyan-300 bg-cyan-100 text-[10px] text-cyan-800 dark:border-cyan-500/30 dark:bg-cyan-500/10 dark:text-cyan-200">
              {queuedJobs + runningJobs}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="strategy" className="gap-1.5 data-[state=active]:bg-amber-100 data-[state=active]:text-amber-900 dark:data-[state=active]:bg-amber-500/15 dark:data-[state=active]:text-amber-100">
            Strategy Health
          </TabsTrigger>
          <TabsTrigger value="guardrails" className="gap-1.5 data-[state=active]:bg-emerald-100 data-[state=active]:text-emerald-900 dark:data-[state=active]:bg-emerald-500/15 dark:data-[state=active]:text-emerald-100">
            Guardrails
          </TabsTrigger>
          <TabsTrigger value="sets" className="gap-1.5 data-[state=active]:bg-violet-100 data-[state=active]:text-violet-900 dark:data-[state=active]:bg-violet-500/15 dark:data-[state=active]:text-violet-100">
            Parameter Sets
          </TabsTrigger>
        </TabsList>

        <TabsContent value="runs" className="mt-0 space-y-4">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <div className="space-y-4 xl:col-span-4">
              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                    <Activity className="h-4 w-4 text-cyan-700 dark:text-cyan-300" />
                    Queue Backtest
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">Save parameter set</Label>
                      <Switch checked={saveBacktestSet} onCheckedChange={setSaveBacktestSet} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">Activate after run</Label>
                      <Switch
                        checked={activateBacktestSet}
                        onCheckedChange={setActivateBacktestSet}
                        disabled={!saveBacktestSet}
                      />
                    </div>
                  </div>

                  {saveBacktestSet && (
                    <div>
                      <Label className="text-xs text-muted-foreground">Parameter set name</Label>
                      <Input
                        value={backtestSetName}
                        onChange={(event) => setBacktestSetName(event.target.value)}
                        placeholder="Backtest baseline"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                  )}

                  <Button
                    size="sm"
                    className="w-full"
                    disabled={backtestMutation.isPending}
                    onClick={() =>
                      backtestMutation.mutate({
                        save_parameter_set: saveBacktestSet,
                        parameter_set_name: backtestSetName || undefined,
                        activate_saved_set: activateBacktestSet,
                      })
                    }
                  >
                    {backtestMutation.isPending ? (
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Play className="mr-1.5 h-3.5 w-3.5" />
                    )}
                    Queue Backtest
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                    <SlidersHorizontal className="h-4 w-4 text-violet-700 dark:text-violet-300" />
                    Queue Optimization
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs text-muted-foreground">Method</Label>
                      <select
                        value={method}
                        onChange={(event) => setMethod(event.target.value as 'grid' | 'random')}
                        className="mt-1 h-8 w-full rounded-md border border-border bg-background px-2 text-sm"
                      >
                        <option value="grid">Grid</option>
                        <option value="random">Random</option>
                      </select>
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Top K</Label>
                      <Input
                        type="number"
                        min={1}
                        max={100}
                        value={topK}
                        onChange={(event) => setTopK(intOr(event.target.value, 20))}
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                  </div>

                  {method === 'random' && (
                    <div>
                      <Label className="text-xs text-muted-foreground">Random samples</Label>
                      <Input
                        type="number"
                        min={5}
                        max={2000}
                        value={randomSamples}
                        onChange={(event) => setRandomSamples(intOr(event.target.value, 200))}
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-muted-foreground">Walk-forward validation</Label>
                    <Switch checked={walkForward} onCheckedChange={setWalkForward} />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-muted-foreground">Save best as active</Label>
                    <Switch checked={saveBest} onCheckedChange={setSaveBest} />
                  </div>

                  {saveBest && (
                    <div>
                      <Label className="text-xs text-muted-foreground">Best set name</Label>
                      <Input
                        value={bestSetName}
                        onChange={(event) => setBestSetName(event.target.value)}
                        placeholder="Production candidate"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                  )}

                  <Button
                    size="sm"
                    className="w-full"
                    disabled={optimizeMutation.isPending}
                    onClick={() =>
                      optimizeMutation.mutate({
                        method,
                        n_random_samples: randomSamples,
                        walk_forward: walkForward,
                        top_k: topK,
                        save_best_as_active: saveBest,
                        best_set_name: bestSetName || undefined,
                      })
                    }
                  >
                    {optimizeMutation.isPending ? (
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <BarChart3 className="mr-1.5 h-3.5 w-3.5" />
                    )}
                    Queue Optimization
                  </Button>
                </CardContent>
              </Card>
            </div>

            <div className="space-y-4 xl:col-span-8">
              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center justify-between gap-2 text-sm font-medium">
                    <span className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-cyan-700 dark:text-cyan-300" />
                      Validation Queue
                    </span>
                    <div className="flex items-center gap-2 text-[11px]">
                      <Badge variant="outline" className="border-cyan-300 bg-cyan-100 text-cyan-800 dark:border-cyan-500/30 dark:bg-cyan-500/10 dark:text-cyan-200">{runningJobs} running</Badge>
                      <Badge variant="outline" className="border-amber-300 bg-amber-100 text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">{queuedJobs} queued</Badge>
                      <Badge variant="outline" className="border-red-300 bg-red-100 text-red-800 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">{failedJobs} failed</Badge>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border/60 bg-background/25">
                        <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Job</TableHead>
                        <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Status</TableHead>
                        <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Progress</TableHead>
                        <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Started</TableHead>
                        <TableHead className="h-9 py-2 text-right text-[11px] uppercase tracking-wide">Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {jobs.slice(0, 18).map((job) => {
                        const progress = Math.round(Math.max(0, Math.min(100, Number(job.progress || 0) * 100)))
                        return (
                          <TableRow key={job.id} className="border-border/45">
                            <TableCell className="py-2">
                              <p className="text-sm font-medium">{job.job_type}</p>
                              <p className="text-[11px] text-muted-foreground">{job.message || 'No message'}</p>
                            </TableCell>
                            <TableCell className="py-2">
                              <Badge className={cn('border text-[10px] uppercase', getJobStatusClass(job.status))}>
                                {job.status}
                              </Badge>
                            </TableCell>
                            <TableCell className="py-2">
                              <div className="space-y-1">
                                <Progress value={progress} className="h-2 bg-muted/70" />
                                <p className="text-[11px] text-muted-foreground">{progress}%</p>
                              </div>
                            </TableCell>
                            <TableCell className="py-2 text-xs text-muted-foreground">
                              {formatDateTime(job.created_at)}
                            </TableCell>
                            <TableCell className="py-2 text-right">
                              {(job.status === 'queued' || job.status === 'running') ? (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-7"
                                  disabled={cancelJobMutation.isPending}
                                  onClick={() => cancelJobMutation.mutate(job.id)}
                                >
                                  Cancel
                                </Button>
                              ) : (
                                <span className="text-xs text-muted-foreground">—</span>
                              )}
                            </TableCell>
                          </TableRow>
                        )
                      })}
                      {jobs.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={5} className="py-8 text-center text-sm text-muted-foreground">
                            No validation jobs yet.
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                    <BarChart3 className="h-4 w-4 text-violet-700 dark:text-violet-300" />
                    Optimization Candidates
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {(optimizationResults?.results || []).slice(0, 8).map((result, index) => {
                    const record = result as Record<string, unknown>
                    const score = toNumber(record.score)
                    const expectedRoi = toNumber(record.expected_roi)
                    const sampleSize = toNumber(record.sample_size)
                    return (
                      <div
                        key={`opt-${index}`}
                        className="flex items-center justify-between rounded-md border border-border/50 bg-background/25 px-2.5 py-2"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm">Candidate #{index + 1}</p>
                          <p className="text-[11px] text-muted-foreground">
                            score {score != null ? score.toFixed(4) : 'N/A'} • expected ROI {formatSignedPercent(expectedRoi)} • n={sampleSize ?? 'N/A'}
                          </p>
                        </div>
                        <Badge variant="outline" className="border-violet-300 bg-violet-100 text-violet-800 dark:border-violet-500/30 dark:bg-violet-500/10 dark:text-violet-200">
                          rank {index + 1}
                        </Badge>
                      </div>
                    )
                  })}
                  {(optimizationResults?.results || []).length === 0 && (
                    <p className="text-xs text-muted-foreground">
                      No optimization leaderboard yet.
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="strategy" className="mt-0 space-y-4">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <Card className="xl:col-span-8 border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                  <ShieldAlert className="h-4 w-4 text-amber-700 dark:text-amber-300" />
                  Strategy Health Matrix
                  </CardTitle>
                </CardHeader>
              <CardContent className="pt-0">
                <Table>
                  <TableHeader>
                    <TableRow className="border-border/60 bg-background/25">
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Strategy</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Status</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Sample</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Accuracy</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">MAE</TableHead>
                      <TableHead className="h-9 py-2 text-right text-[11px] uppercase tracking-wide">Override</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {strategyHealth.slice(0, 40).map((strategy) => (
                      <TableRow key={strategy.strategy_type} className="border-border/45">
                        <TableCell className="py-2">
                          <p className="text-sm font-medium">{strategy.strategy_type}</p>
                          {strategy.last_reason && (
                            <p className="text-[11px] text-muted-foreground">{strategy.last_reason}</p>
                          )}
                        </TableCell>
                        <TableCell className="py-2">
                          <Badge className={cn('border text-[10px] uppercase', getStrategyStatusClass(strategy.status))}>
                            {strategy.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-2 text-xs font-data">{strategy.sample_size}</TableCell>
                        <TableCell className="py-2 text-xs font-data">
                          {formatPercentFromRatio(toNumber(strategy.directional_accuracy))}
                        </TableCell>
                        <TableCell className="py-2 text-xs font-data">
                          {toNumber(strategy.mae_roi)?.toFixed(4) ?? 'N/A'}
                        </TableCell>
                        <TableCell className="py-2 text-right">
                          <div className="flex justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7"
                              disabled={overrideStrategyMutation.isPending}
                              onClick={() =>
                                overrideStrategyMutation.mutate({
                                  strategyType: strategy.strategy_type,
                                  status: 'active',
                                })
                              }
                            >
                              Activate
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7"
                              disabled={overrideStrategyMutation.isPending}
                              onClick={() =>
                                overrideStrategyMutation.mutate({
                                  strategyType: strategy.strategy_type,
                                  status: 'demoted',
                                })
                              }
                            >
                              Demote
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7"
                              disabled={clearOverrideMutation.isPending}
                              onClick={() => clearOverrideMutation.mutate(strategy.strategy_type)}
                            >
                              Clear
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                    {strategyHealth.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={6} className="py-8 text-center text-sm text-muted-foreground">
                          No strategy health data yet.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <div className="space-y-4 xl:col-span-4">
              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Calibration Trend</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {trendTail.map((bucket) => (
                    <div
                      key={bucket.bucket_start}
                      className="rounded-md border border-border/45 bg-background/25 px-2.5 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs text-muted-foreground">{new Date(bucket.bucket_start).toLocaleDateString()}</p>
                        <Badge variant="outline" className="border-border/70 bg-background/50 text-[10px] text-muted-foreground">
                          n={bucket.sample_size}
                        </Badge>
                      </div>
                      <p className="mt-1 text-xs">
                        MAE {bucket.mae_roi} • Accuracy {(bucket.directional_accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                  ))}
                  {trendTail.length === 0 && (
                    <p className="text-xs text-muted-foreground">No trend buckets yet.</p>
                  )}
                </CardContent>
              </Card>

              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Execution by Source (30d)</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {executionBySource.map((row) => (
                    <div key={row.source} className="rounded-md border border-border/45 bg-background/25 px-2.5 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm">{row.source}</p>
                        <p className={cn('text-xs font-data', row.pnl >= 0 ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-700 dark:text-red-300')}>
                          {formatCurrency(row.pnl, true)}
                        </p>
                      </div>
                      <p className="mt-1 text-[11px] text-muted-foreground">
                        {row.total} orders • {row.failed} failed • {(row.failureRate * 100).toFixed(1)}% failure
                      </p>
                    </div>
                  ))}
                  {executionBySource.length === 0 && (
                    <p className="text-xs text-muted-foreground">Execution telemetry unavailable.</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="guardrails" className="mt-0 space-y-4">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <Card className="xl:col-span-8 border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                  <ShieldAlert className="h-4 w-4 text-emerald-700 dark:text-emerald-300" />
                  Guardrail Control Plane
                  </CardTitle>
                </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                  <GuardrailField label="Enabled">
                    <Switch
                      checked={guardrailsForm.enabled}
                      onCheckedChange={(value) => {
                        setGuardrailsForm((prev) => ({ ...prev, enabled: value }))
                        setGuardrailsDirty(true)
                      }}
                    />
                  </GuardrailField>
                  <GuardrailField label="Min Samples">
                    <Input
                      type="number"
                      value={guardrailsForm.min_samples}
                      onChange={(event) => {
                        setGuardrailsForm((prev) => ({ ...prev, min_samples: intOr(event.target.value, 25) }))
                        setGuardrailsDirty(true)
                      }}
                      className="h-8 text-sm"
                    />
                  </GuardrailField>
                  <GuardrailField label="Min Directional Accuracy">
                    <Input
                      type="number"
                      step="0.01"
                      value={guardrailsForm.min_directional_accuracy}
                      onChange={(event) => {
                        setGuardrailsForm((prev) => ({
                          ...prev,
                          min_directional_accuracy: floatOr(event.target.value, 0.52),
                        }))
                        setGuardrailsDirty(true)
                      }}
                      className="h-8 text-sm"
                    />
                  </GuardrailField>
                  <GuardrailField label="Max MAE (ROI)">
                    <Input
                      type="number"
                      step="0.1"
                      value={guardrailsForm.max_mae_roi}
                      onChange={(event) => {
                        setGuardrailsForm((prev) => ({
                          ...prev,
                          max_mae_roi: floatOr(event.target.value, 12),
                        }))
                        setGuardrailsDirty(true)
                      }}
                      className="h-8 text-sm"
                    />
                  </GuardrailField>
                  <GuardrailField label="Lookback Days">
                    <Input
                      type="number"
                      value={guardrailsForm.lookback_days}
                      onChange={(event) => {
                        setGuardrailsForm((prev) => ({
                          ...prev,
                          lookback_days: intOr(event.target.value, 90),
                        }))
                        setGuardrailsDirty(true)
                      }}
                      className="h-8 text-sm"
                    />
                  </GuardrailField>
                  <GuardrailField label="Auto Promote">
                    <Switch
                      checked={guardrailsForm.auto_promote}
                      onCheckedChange={(value) => {
                        setGuardrailsForm((prev) => ({ ...prev, auto_promote: value }))
                        setGuardrailsDirty(true)
                      }}
                    />
                  </GuardrailField>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    disabled={!guardrailsDirty || saveGuardrailsMutation.isPending}
                    onClick={() => saveGuardrailsMutation.mutate(guardrailsForm)}
                  >
                    {saveGuardrailsMutation.isPending && <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
                    Save Guardrails
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={evaluateGuardrailsMutation.isPending}
                    onClick={() => evaluateGuardrailsMutation.mutate()}
                  >
                    {evaluateGuardrailsMutation.isPending && <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
                    Evaluate Now
                  </Button>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-4 xl:col-span-4">
              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Guardrail Snapshot</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-xs text-muted-foreground">
                  <p>
                    Accuracy target: <span className="text-foreground">{(guardrailsForm.min_directional_accuracy * 100).toFixed(1)}%</span>
                  </p>
                  <p>
                    Current accuracy: <span className={cn(directionalAccuracy != null && directionalAccuracy >= guardrailsForm.min_directional_accuracy ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300')}>
                      {formatPercentFromRatio(directionalAccuracy)}
                    </span>
                  </p>
                  <p>
                    MAE cap: <span className="text-foreground">{guardrailsForm.max_mae_roi.toFixed(2)}</span>
                  </p>
                  <p>
                    Execution sample: <span className="text-foreground">{executionSample ?? 'N/A'}</span>
                  </p>
                  {!guardrailsForm.enabled && (
                    <div className="rounded-md border border-amber-300 bg-amber-100 px-2.5 py-2 text-amber-900 dark:border-amber-500/25 dark:bg-amber-500/10 dark:text-amber-200">
                      Guardrails are disabled. Strategy status will not auto-adjust.
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Resolver Coverage (7d)</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {resolverByType.map((row) => (
                    <div key={row.signalType} className="rounded-md border border-border/45 bg-background/25 px-2.5 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm">{row.signalType}</p>
                        <Badge variant="outline" className="border-border/70 bg-background/40 text-[10px] text-muted-foreground">
                          {formatPercentFromRatio(row.tradableRate)}
                        </Badge>
                      </div>
                      <p className="text-[11px] text-muted-foreground">
                        {row.tradable}/{row.candidates} tradable
                      </p>
                    </div>
                  ))}
                  {resolverByType.length === 0 && (
                    <p className="text-xs text-muted-foreground">No resolver telemetry in current window.</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="sets" className="mt-0 space-y-4">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <div className="space-y-4 xl:col-span-4">
              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                    <CheckCircle className="h-4 w-4 text-emerald-700 dark:text-emerald-300" />
                    Active Parameter Set
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-1">
                  <p className="text-sm font-medium">{activeSetLabel}</p>
                  <p className="text-xs text-muted-foreground">
                    {overview?.parameter_set_count ?? 0} total saved sets
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Latest optimization score: {latestOptimizationScore != null ? latestOptimizationScore.toFixed(4) : 'N/A'}
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border/60 bg-card/80">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Current Guardrail Config</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-xs text-muted-foreground">
                  <p>Enabled: {guardrailsForm.enabled ? 'Yes' : 'No'}</p>
                  <p>Min sample: {guardrailsForm.min_samples}</p>
                  <p>Min directional accuracy: {(guardrailsForm.min_directional_accuracy * 100).toFixed(1)}%</p>
                  <p>Max MAE ROI: {guardrailsForm.max_mae_roi.toFixed(2)}</p>
                  <p>Lookback: {guardrailsForm.lookback_days} days</p>
                </CardContent>
              </Card>
            </div>

            <Card className="xl:col-span-8 border-border/60 bg-card/80">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Saved Parameter Sets</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <Table>
                  <TableHeader>
                    <TableRow className="border-border/60 bg-background/25">
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Name</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Created</TableHead>
                      <TableHead className="h-9 py-2 text-[11px] uppercase tracking-wide">Status</TableHead>
                      <TableHead className="h-9 py-2 text-right text-[11px] uppercase tracking-wide">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(parameterSets?.parameter_sets || []).slice(0, 40).map((parameterSet) => {
                      const isActive = Boolean(parameterSet.is_active)
                      const label = String(parameterSet.name || parameterSet.id)
                      return (
                        <TableRow key={String(parameterSet.id)} className="border-border/45">
                          <TableCell className="py-2 text-sm">{label}</TableCell>
                          <TableCell className="py-2 text-xs text-muted-foreground">
                            {parameterSet.created_at
                              ? new Date(String(parameterSet.created_at)).toLocaleString()
                              : 'unknown date'}
                          </TableCell>
                          <TableCell className="py-2">
                            {isActive ? (
                              <Badge className="border border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/35 dark:bg-emerald-500/15 dark:text-emerald-200">
                                Active
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="border-border/70 bg-background/40 text-muted-foreground">
                                Saved
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell className="py-2 text-right">
                            {isActive ? (
                              <span className="text-xs text-muted-foreground">—</span>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-7"
                                disabled={activateMutation.isPending}
                                onClick={() => activateMutation.mutate(String(parameterSet.id))}
                              >
                                Activate
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                    {(parameterSets?.parameter_sets || []).length === 0 && (
                      <TableRow>
                        <TableCell colSpan={4} className="py-8 text-center text-sm text-muted-foreground">
                          No saved parameter sets.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {failedJobs > 0 && (
        <div className="rounded-lg border border-red-300 bg-red-100 px-3 py-2 text-xs text-red-900 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-3.5 w-3.5" />
            {failedJobs} validation job{failedJobs === 1 ? '' : 's'} failed recently. Check queue details for error messages.
          </div>
        </div>
      )}
    </div>
  )
}

function SnapshotStat({
  label,
  value,
  tone = 'neutral',
  compact = false,
}: {
  label: string
  value: string
  tone?: 'good' | 'warn' | 'info' | 'neutral'
  compact?: boolean
}) {
  const toneClass = tone === 'good'
    ? 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-100'
    : tone === 'warn'
      ? 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100'
      : tone === 'info'
        ? 'border-cyan-300 bg-cyan-100 text-cyan-900 dark:border-cyan-500/30 dark:bg-cyan-500/10 dark:text-cyan-100'
        : 'border-border/70 bg-background/35 text-foreground'

  return (
    <div className={cn('rounded-md border px-2.5 py-2', toneClass)}>
      <p className="text-[10px] uppercase tracking-wide opacity-80">{label}</p>
      <p className={cn('mt-1 font-data font-semibold', compact ? 'text-sm truncate' : 'text-base')}>{value}</p>
    </div>
  )
}

function GuardrailField({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  )
}
