import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  BarChart3,
  CheckCircle,
  Loader2,
  Play,
  ShieldAlert,
  SlidersHorizontal,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
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

export default function ValidationEnginePanel() {
  const queryClient = useQueryClient()
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

  const { data: overview, isLoading: loadingOverview } = useQuery({
    queryKey: ['validation-overview'],
    queryFn: getValidationOverview,
    refetchInterval: 20000,
  })

  const { data: parameterSets } = useQuery({
    queryKey: ['validation-parameter-sets'],
    queryFn: getValidationParameterSets,
    refetchInterval: 30000,
  })

  const { data: jobsData } = useQuery({
    queryKey: ['validation-jobs'],
    queryFn: () => getValidationJobs(40),
    refetchInterval: 3000,
  })

  const { data: guardrailsConfig } = useQuery({
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

  const backtestMutation = useMutation({
    mutationFn: runValidationBacktest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
    },
  })

  const optimizeMutation = useMutation({
    mutationFn: runValidationOptimization,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
    },
  })

  const activateMutation = useMutation({
    mutationFn: activateValidationParameterSet,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-parameter-sets'] })
    },
  })

  const cancelJobMutation = useMutation({
    mutationFn: cancelValidationJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
  })

  const saveGuardrailsMutation = useMutation({
    mutationFn: updateValidationGuardrailConfig,
    onSuccess: () => {
      setGuardrailsDirty(false)
      queryClient.invalidateQueries({ queryKey: ['validation-guardrails-config'] })
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
  })

  const evaluateGuardrailsMutation = useMutation({
    mutationFn: evaluateValidationGuardrails,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
      queryClient.invalidateQueries({ queryKey: ['validation-jobs'] })
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
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['validation-overview'] }),
  })

  const clearOverrideMutation = useMutation({
    mutationFn: clearValidationStrategyOverride,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['validation-overview'] }),
  })

  const calibration = overview?.calibration_90d?.overall
  const directionalAccuracyPct = useMemo(() => {
    const v = calibration?.directional_accuracy
    if (typeof v !== 'number') return null
    return (v * 100).toFixed(1)
  }, [calibration])
  const trendTail = (overview?.calibration_trend_90d || []).slice(-8)

  if (loadingOverview) {
    return (
      <div className="flex justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <Card className="bg-muted/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Resolved Sample</p>
            <p className="text-lg font-semibold">{overview?.calibration_90d?.sample_size ?? 0}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Directional Accuracy</p>
            <p className="text-lg font-semibold">{directionalAccuracyPct ? `${directionalAccuracyPct}%` : 'N/A'}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">MAE (ROI)</p>
            <p className="text-lg font-semibold">{calibration?.mae_roi ?? 'N/A'}</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/40">
          <CardContent className="p-3">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Combinatorial Accuracy</p>
            <p className="text-lg font-semibold">
              {typeof overview?.combinatorial_validation?.accuracy === 'number'
                ? `${(overview.combinatorial_validation.accuracy * 100).toFixed(1)}%`
                : 'N/A'}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card className="bg-muted/30 border-border/40">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <p className="text-sm font-medium">Run Backtest</p>
            </div>

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
                  <Switch checked={activateBacktestSet} onCheckedChange={setActivateBacktestSet} disabled={!saveBacktestSet} />
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
              {backtestMutation.isPending ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Play className="w-3.5 h-3.5 mr-1.5" />}
              Queue Backtest
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-muted/30 border-border/40">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-violet-400" />
              <p className="text-sm font-medium">Run Optimization</p>
            </div>

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
                  onChange={(e) => setTopK(parseInt(e.target.value || '20', 10))}
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
                  onChange={(e) => setRandomSamples(parseInt(e.target.value || '200', 10))}
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
              {optimizeMutation.isPending ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <BarChart3 className="w-3.5 h-3.5 mr-1.5" />}
              Queue Optimization
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-muted/30 border-border/40">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <p className="text-sm font-medium">Validation Jobs</p>
          </div>
          <div className="space-y-2">
            {(jobsData?.jobs || []).slice(0, 8).map((job) => (
              <div key={job.id} className="p-2 rounded-lg border border-border/40 bg-card/40">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm">{job.job_type}</p>
                  <Badge className="text-[10px]">{job.status}</Badge>
                </div>
                <p className="text-[11px] text-muted-foreground mt-1">
                  {job.message || 'No message'} · {typeof job.progress === 'number' ? `${Math.round(job.progress * 100)}%` : '0%'}
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
          </div>
        </CardContent>
      </Card>

      <Card className="bg-muted/30 border-border/40">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-yellow-400" />
            <p className="text-sm font-medium">Guardrail Controls</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <div>
              <Label className="text-xs text-muted-foreground">Enabled</Label>
              <div className="mt-1">
                <Switch
                  checked={guardrailsForm.enabled}
                  onCheckedChange={(v) => {
                    setGuardrailsForm((p) => ({ ...p, enabled: v }))
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
                  setGuardrailsForm((p) => ({ ...p, min_samples: parseInt(e.target.value || '25', 10) }))
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
                  setGuardrailsForm((p) => ({ ...p, min_directional_accuracy: parseFloat(e.target.value || '0.52') }))
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
                  setGuardrailsForm((p) => ({ ...p, max_mae_roi: parseFloat(e.target.value || '12') }))
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
                  setGuardrailsForm((p) => ({ ...p, lookback_days: parseInt(e.target.value || '90', 10) }))
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
                  onCheckedChange={(v) => {
                    setGuardrailsForm((p) => ({ ...p, auto_promote: v }))
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

      <Card className="bg-muted/30 border-border/40">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-violet-400" />
            <p className="text-sm font-medium">Calibration Trend (Recent Buckets)</p>
          </div>
          <div className="space-y-1">
            {trendTail.length === 0 && <p className="text-xs text-muted-foreground">No trend data yet</p>}
            {trendTail.map((bucket) => (
              <div key={bucket.bucket_start} className="text-xs text-muted-foreground flex items-center justify-between">
                <span>{new Date(bucket.bucket_start).toLocaleDateString()}</span>
                <span>MAE {bucket.mae_roi} · Acc {(bucket.directional_accuracy * 100).toFixed(1)}% · n={bucket.sample_size}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-muted/30 border-border/40">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            <p className="text-sm font-medium">Strategy Health</p>
          </div>
          <div className="space-y-2">
            {(overview?.strategy_health || []).slice(0, 20).map((s) => (
              <div key={s.strategy_type} className="p-2 rounded-lg border border-border/40 bg-card/40">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm">{s.strategy_type}</p>
                  <Badge className={s.status === 'demoted' ? 'bg-red-500/15 text-red-400 border-red-500/30' : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'}>
                    {s.status}
                  </Badge>
                </div>
                <p className="text-[11px] text-muted-foreground mt-1">
                  n={s.sample_size} · acc={typeof s.directional_accuracy === 'number' ? (s.directional_accuracy * 100).toFixed(1) : 'N/A'}% · mae={s.mae_roi ?? 'N/A'}
                </p>
                {s.last_reason && <p className="text-[11px] text-yellow-400/80 mt-1">{s.last_reason}</p>}
                <div className="flex items-center gap-1 mt-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7"
                    disabled={overrideStrategyMutation.isPending}
                    onClick={() => overrideStrategyMutation.mutate({ strategyType: s.strategy_type, status: 'active' })}
                  >
                    Force Active
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7"
                    disabled={overrideStrategyMutation.isPending}
                    onClick={() => overrideStrategyMutation.mutate({ strategyType: s.strategy_type, status: 'demoted' })}
                  >
                    Force Demote
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7"
                    disabled={clearOverrideMutation.isPending}
                    onClick={() => clearOverrideMutation.mutate(s.strategy_type)}
                  >
                    Clear Override
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-muted/30 border-border/40">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <p className="text-sm font-medium">Saved Parameter Sets</p>
          </div>
          <div className="space-y-2">
            {(parameterSets?.parameter_sets || []).slice(0, 10).map((ps) => {
              const isActive = Boolean(ps.is_active)
              return (
                <div key={String(ps.id)} className="flex items-center justify-between gap-3 p-2 rounded-lg border border-border/40 bg-card/40">
                  <div className="min-w-0">
                    <p className="text-sm truncate">{String(ps.name || ps.id)}</p>
                    <p className="text-[11px] text-muted-foreground truncate">
                      {ps.created_at ? new Date(String(ps.created_at)).toLocaleString() : 'unknown date'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {isActive ? (
                      <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30">Active</Badge>
                    ) : (
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={activateMutation.isPending}
                        onClick={() => activateMutation.mutate(String(ps.id))}
                      >
                        Activate
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

