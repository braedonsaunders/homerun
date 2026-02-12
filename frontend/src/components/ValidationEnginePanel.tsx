import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
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
import { Switch } from './ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import {
  activateValidationParameterSet,
  cancelValidationJob,
  clearValidationStrategyOverride,
  evaluateValidationGuardrails,
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

function getJobStatusClass(status: string): string {
  switch (status) {
    case 'completed':
      return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
    case 'running':
      return 'bg-blue-500/15 text-blue-400 border-blue-500/30'
    case 'queued':
      return 'bg-amber-500/15 text-amber-400 border-amber-500/30'
    case 'failed':
      return 'bg-red-500/15 text-red-400 border-red-500/30'
    case 'cancelled':
      return 'bg-zinc-500/15 text-zinc-300 border-zinc-500/30'
    default:
      return 'bg-muted text-muted-foreground border-border'
  }
}

function getStrategyStatusClass(status: string): string {
  if (status === 'demoted') {
    return 'bg-red-500/15 text-red-400 border-red-500/30'
  }
  if (status === 'active') {
    return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
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
    queryFn: () => getValidationJobs(40),
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
    fetchingOverview || fetchingJobs || fetchingGuardrails || fetchingParameterSets

  const calibration = overview?.calibration_90d?.overall
  const directionalAccuracyPct = useMemo(() => {
    const value = calibration?.directional_accuracy
    if (typeof value !== 'number') return 'N/A'
    return `${(value * 100).toFixed(1)}%`
  }, [calibration])

  const combinatorialAccuracyPct = useMemo(() => {
    const value = (overview?.combinatorial_validation as Record<string, unknown> | undefined)?.accuracy
    if (typeof value !== 'number') return 'N/A'
    return `${(value * 100).toFixed(1)}%`
  }, [overview?.combinatorial_validation])

  const trendTail = useMemo(
    () => (overview?.calibration_trend_90d || []).slice(-10),
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
  const jobs = jobsData?.jobs || []
  const executionFailureRate = overview?.autotrader_execution_30d?.failure_rate
  const resolverTradableRate = overview?.world_intel_resolver_7d?.tradable_rate

  const activeSetLabel = useMemo(() => {
    const activeSet = overview?.active_parameter_set
    if (!activeSet) return 'None'
    const activeName = (activeSet as Record<string, unknown>).name
    const activeId = (activeSet as Record<string, unknown>).id
    if (typeof activeName === 'string' && activeName.trim()) return activeName
    if (typeof activeId === 'string' && activeId.trim()) return activeId
    return 'Active set'
  }, [overview?.active_parameter_set])

  const refreshAll = async () => {
    await Promise.all([
      refetchOverview(),
      refetchJobs(),
      refetchGuardrails(),
      refetchParameterSets(),
    ])
  }

  if (loadingOverview) {
    return (
      <div className="flex justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold">Validation Engine</h3>
          <p className="text-xs text-muted-foreground">
            Backtests, optimization, strategy health, and guardrail controls.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-[10px] bg-muted/40">
            Auto refresh: jobs 3s, overview 20s
          </Badge>
          <Button
            variant="secondary"
            size="sm"
            onClick={refreshAll}
            disabled={isRefreshing}
          >
            <RefreshCw className={cn('w-3.5 h-3.5 mr-1.5', isRefreshing && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {notice && (
        <div
          className={cn(
            'rounded-lg border px-3 py-2 text-xs',
            notice.type === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
              : 'bg-red-500/10 border-red-500/20 text-red-300'
          )}
        >
          {notice.text}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-7 gap-3">
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Resolved Sample</p>
            <p className="text-lg font-semibold">{overview?.calibration_90d?.sample_size ?? 0}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Directional Accuracy</p>
            <p className="text-lg font-semibold">{directionalAccuracyPct}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">MAE (ROI)</p>
            <p className="text-lg font-semibold">{calibration?.mae_roi ?? 'N/A'}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Combinatorial Accuracy</p>
            <p className="text-lg font-semibold">{combinatorialAccuracyPct}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Demoted Strategies</p>
            <p className="text-lg font-semibold">{demotedCount}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Execution Failure (30d)</p>
            <p className="text-lg font-semibold">
              {typeof executionFailureRate === 'number' ? `${(executionFailureRate * 100).toFixed(1)}%` : 'N/A'}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40 border-border/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">World Resolver Tradable (7d)</p>
            <p className="text-lg font-semibold">
              {typeof resolverTradableRate === 'number' ? `${(resolverTradableRate * 100).toFixed(1)}%` : 'N/A'}
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as ValidationSubTab)} className="space-y-4">
        <TabsList className="h-auto flex-wrap justify-start">
          <TabsTrigger value="runs">Runs & Jobs</TabsTrigger>
          <TabsTrigger value="strategy">Strategy Health</TabsTrigger>
          <TabsTrigger value="guardrails">Guardrails</TabsTrigger>
          <TabsTrigger value="sets">Parameter Sets</TabsTrigger>
        </TabsList>

        <TabsContent value="runs" className="space-y-4 mt-0">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Card className="bg-muted/30 border-border/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  Run Backtest
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">Save Parameter Set</Label>
                    <div className="mt-1">
                      <Switch checked={saveBacktestSet} onCheckedChange={setSaveBacktestSet} />
                    </div>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Activate Saved Set</Label>
                    <div className="mt-1">
                      <Switch
                        checked={activateBacktestSet}
                        onCheckedChange={setActivateBacktestSet}
                        disabled={!saveBacktestSet}
                      />
                    </div>
                  </div>
                </div>

                {saveBacktestSet && (
                  <div>
                    <Label className="text-xs text-muted-foreground">Parameter Set Name</Label>
                    <Input
                      value={backtestSetName}
                      onChange={(e) => setBacktestSetName(e.target.value)}
                      placeholder="Backtest baseline"
                      className="mt-1 text-sm"
                    />
                  </div>
                )}

                <Button
                  size="sm"
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
                    <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 mr-1.5" />
                  )}
                  Queue Backtest
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-muted/30 border-border/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4 text-violet-400" />
                  Run Optimization
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">Method</Label>
                    <select
                      value={method}
                      onChange={(e) => setMethod(e.target.value as 'grid' | 'random')}
                      className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm mt-1"
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
                      onChange={(e) => setTopK(intOr(e.target.value, 20))}
                      className="mt-1 text-sm"
                    />
                  </div>
                </div>

                {method === 'random' && (
                  <div>
                    <Label className="text-xs text-muted-foreground">Random Samples</Label>
                    <Input
                      type="number"
                      min={5}
                      max={2000}
                      value={randomSamples}
                      onChange={(e) => setRandomSamples(intOr(e.target.value, 200))}
                      className="mt-1 text-sm"
                    />
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Walk Forward Validation</Label>
                  <Switch checked={walkForward} onCheckedChange={setWalkForward} />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Save Best As Active</Label>
                  <Switch checked={saveBest} onCheckedChange={setSaveBest} />
                </div>

                {saveBest && (
                  <div>
                    <Label className="text-xs text-muted-foreground">Best Set Name</Label>
                    <Input
                      value={bestSetName}
                      onChange={(e) => setBestSetName(e.target.value)}
                      placeholder="World-class candidate"
                      className="mt-1 text-sm"
                    />
                  </div>
                )}

                <Button
                  size="sm"
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
                    <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <BarChart3 className="w-3.5 h-3.5 mr-1.5" />
                  )}
                  Queue Optimization
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-muted/30 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                Validation Jobs
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {jobs.length === 0 && (
                <p className="text-xs text-muted-foreground">No jobs yet.</p>
              )}
              {jobs.slice(0, 12).map((job) => (
                <div key={job.id} className="p-2 rounded-lg border border-border/40 bg-card/40">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm">{job.job_type}</p>
                    <Badge className={cn('text-[10px] border', getJobStatusClass(job.status))}>
                      {job.status}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-1">
                    {job.message || 'No message'} ·{' '}
                    {typeof job.progress === 'number' ? `${Math.round(job.progress * 100)}%` : '0%'}
                  </p>
                  <p className="text-[10px] text-muted-foreground/80 mt-0.5">
                    {job.created_at ? new Date(job.created_at).toLocaleString() : 'unknown start time'}
                  </p>
                  {(job.status === 'queued' || job.status === 'running') && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 mt-1"
                      disabled={cancelJobMutation.isPending}
                      onClick={() => cancelJobMutation.mutate(job.id)}
                    >
                      Cancel
                    </Button>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="strategy" className="space-y-4 mt-0">
          <Card className="bg-muted/30 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                Strategy Health
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {strategyHealth.length === 0 && (
                <p className="text-xs text-muted-foreground">No strategy health data yet.</p>
              )}
              {strategyHealth.slice(0, 30).map((strategy) => (
                <div key={strategy.strategy_type} className="p-2 rounded-lg border border-border/40 bg-card/40">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm">{strategy.strategy_type}</p>
                    <Badge className={cn('border text-[10px]', getStrategyStatusClass(strategy.status))}>
                      {strategy.status}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-1">
                    n={strategy.sample_size} · acc=
                    {typeof strategy.directional_accuracy === 'number'
                      ? `${(strategy.directional_accuracy * 100).toFixed(1)}%`
                      : 'N/A'}
                    {' '}· mae={strategy.mae_roi ?? 'N/A'}
                  </p>
                  {strategy.last_reason && (
                    <p className="text-[11px] text-yellow-400/80 mt-1">{strategy.last_reason}</p>
                  )}
                  <div className="flex items-center gap-1 mt-1">
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
                      Force Active
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
                      Force Demote
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7"
                      disabled={clearOverrideMutation.isPending}
                      onClick={() => clearOverrideMutation.mutate(strategy.strategy_type)}
                    >
                      Clear Override
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-muted/30 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-violet-400" />
                Calibration Trend (Recent Buckets)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              {trendTail.length === 0 && <p className="text-xs text-muted-foreground">No trend data yet.</p>}
              {trendTail.map((bucket) => (
                <div
                  key={bucket.bucket_start}
                  className="text-xs text-muted-foreground flex items-center justify-between rounded-md bg-card/40 border border-border/40 px-2 py-1.5"
                >
                  <span>{new Date(bucket.bucket_start).toLocaleDateString()}</span>
                  <span>
                    MAE {bucket.mae_roi} · Acc {(bucket.directional_accuracy * 100).toFixed(1)}% · n=
                    {bucket.sample_size}
                  </span>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="guardrails" className="space-y-4 mt-0">
          <Card className="bg-muted/30 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-yellow-400" />
                Guardrail Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <Label className="text-xs text-muted-foreground">Enabled</Label>
                  <div className="mt-1">
                    <Switch
                      checked={guardrailsForm.enabled}
                      onCheckedChange={(value) => {
                        setGuardrailsForm((prev) => ({ ...prev, enabled: value }))
                        setGuardrailsDirty(true)
                      }}
                    />
                  </div>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Min Samples</Label>
                  <Input
                    type="number"
                    value={guardrailsForm.min_samples}
                    onChange={(e) => {
                      setGuardrailsForm((prev) => ({ ...prev, min_samples: intOr(e.target.value, 25) }))
                      setGuardrailsDirty(true)
                    }}
                    className="mt-1 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Min Directional Accuracy</Label>
                  <Input
                    type="number"
                    step="0.01"
                    value={guardrailsForm.min_directional_accuracy}
                    onChange={(e) => {
                      setGuardrailsForm((prev) => ({
                        ...prev,
                        min_directional_accuracy: floatOr(e.target.value, 0.52),
                      }))
                      setGuardrailsDirty(true)
                    }}
                    className="mt-1 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Max MAE (ROI)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={guardrailsForm.max_mae_roi}
                    onChange={(e) => {
                      setGuardrailsForm((prev) => ({
                        ...prev,
                        max_mae_roi: floatOr(e.target.value, 12),
                      }))
                      setGuardrailsDirty(true)
                    }}
                    className="mt-1 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Lookback Days</Label>
                  <Input
                    type="number"
                    value={guardrailsForm.lookback_days}
                    onChange={(e) => {
                      setGuardrailsForm((prev) => ({
                        ...prev,
                        lookback_days: intOr(e.target.value, 90),
                      }))
                      setGuardrailsDirty(true)
                    }}
                    className="mt-1 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Auto Promote</Label>
                  <div className="mt-1">
                    <Switch
                      checked={guardrailsForm.auto_promote}
                      onCheckedChange={(value) => {
                        setGuardrailsForm((prev) => ({ ...prev, auto_promote: value }))
                        setGuardrailsDirty(true)
                      }}
                    />
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  disabled={!guardrailsDirty || saveGuardrailsMutation.isPending}
                  onClick={() => saveGuardrailsMutation.mutate(guardrailsForm)}
                >
                  Save Guardrails
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={evaluateGuardrailsMutation.isPending}
                  onClick={() => evaluateGuardrailsMutation.mutate()}
                >
                  Evaluate Now
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sets" className="space-y-4 mt-0">
          <Card className="bg-muted/30 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                Active Parameter Set
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              <p className="text-sm">{activeSetLabel}</p>
              <p className="text-xs text-muted-foreground">
                {overview?.parameter_set_count ?? 0} total saved sets
              </p>
            </CardContent>
          </Card>

          <Card className="bg-muted/30 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                Saved Parameter Sets
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {(parameterSets?.parameter_sets || []).length === 0 && (
                <p className="text-xs text-muted-foreground">No saved parameter sets.</p>
              )}
              {(parameterSets?.parameter_sets || []).slice(0, 20).map((parameterSet) => {
                const isActive = Boolean(parameterSet.is_active)
                return (
                  <div
                    key={String(parameterSet.id)}
                    className="flex items-center justify-between gap-3 p-2 rounded-lg border border-border/40 bg-card/40"
                  >
                    <div className="min-w-0">
                      <p className="text-sm truncate">{String(parameterSet.name || parameterSet.id)}</p>
                      <p className="text-[11px] text-muted-foreground truncate">
                        {parameterSet.created_at
                          ? new Date(String(parameterSet.created_at)).toLocaleString()
                          : 'unknown date'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {isActive ? (
                        <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30">
                          Active
                        </Badge>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={activateMutation.isPending}
                          onClick={() => activateMutation.mutate(String(parameterSet.id))}
                        >
                          Activate
                        </Button>
                      )}
                    </div>
                  </div>
                )
              })}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
