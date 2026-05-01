import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  Archive,
  Brain,
  Database,
  Loader2,
  Rocket,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react'
import FillModelPanel from './FillModelPanel'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import {
  archiveMLAdapter,
  archiveMLModel,
  deleteMLAdapter,
  deleteMLData,
  deleteMLModel,
  getMLAdapters,
  getMLCapabilities,
  getMLDataStats,
  getMLDeployments,
  getMLJobs,
  getMLModels,
  getMLRecorderConfig,
  importMLModel,
  pruneMLData,
  trainMLAdapter,
  triggerMLEvaluation,
  updateMLDeployment,
  updateMLRecorderConfig,
  type MLAdapter,
  type MLCapabilities,
  type MLDeployment,
  type MLJob,
  type MLModel,
  type MLRecorderStats,
  type MLDataStats,
} from '../services/apiMachineLearning'

type TabId = 'fill-model' | 'data' | 'import' | 'models' | 'adapters' | 'deployments' | 'jobs'

const TABS: { id: TabId; label: string; icon: typeof Database }[] = [
  { id: 'fill-model', label: 'Fill Model', icon: Sparkles },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'import', label: 'Import', icon: Upload },
  { id: 'models', label: 'Models', icon: Brain },
  { id: 'adapters', label: 'Adapters', icon: Activity },
  { id: 'deployments', label: 'Deploy', icon: Rocket },
  { id: 'jobs', label: 'Jobs', icon: Activity },
]

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-card/40 p-3">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-base font-semibold">{value}</div>
      {sub ? <div className="mt-0.5 text-[11px] text-muted-foreground">{sub}</div> : null}
    </div>
  )
}

function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null
  return (
    <div className="rounded-md border border-red-500/30 bg-red-500/5 px-3 py-2 text-xs text-red-300">
      {message}
    </div>
  )
}

function getErrorMessage(error: unknown): string | null {
  const detail = (error as any)?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  const message = (error as any)?.message
  if (typeof message === 'string' && message.trim()) return message.trim()
  return null
}

function parseList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function formatMetric(value: number | null | undefined, decimals = 3): string {
  if (value == null || Number.isNaN(value)) return '-'
  return Number(value).toFixed(decimals)
}

function DataTab() {
  const queryClient = useQueryClient()
  const { data: capabilities } = useQuery<MLCapabilities>({ queryKey: ['ml-capabilities'], queryFn: getMLCapabilities, refetchInterval: 15000 })
  const { data: recorder, isLoading } = useQuery<MLRecorderStats>({ queryKey: ['ml-recorder'], queryFn: getMLRecorderConfig, refetchInterval: 10000 })
  const { data: stats } = useQuery<MLDataStats>({ queryKey: ['ml-data-stats'], queryFn: getMLDataStats, refetchInterval: 30000 })
  const [intervalSeconds, setIntervalSeconds] = useState('60')
  const [retentionDays, setRetentionDays] = useState('90')
  const [assets, setAssets] = useState('')
  const [timeframes, setTimeframes] = useState('')

  useEffect(() => {
    if (recorder?.config) {
      setIntervalSeconds(String(recorder.config.interval_seconds))
      setRetentionDays(String(recorder.config.retention_days))
      setAssets((recorder.config.assets ?? []).join(', '))
      setTimeframes((recorder.config.timeframes ?? []).join(', '))
    }
  }, [recorder?.config])

  const updateMutation = useMutation({
    mutationFn: updateMLRecorderConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-recorder'] })
      queryClient.invalidateQueries({ queryKey: ['ml-data-stats'] })
    },
  })
  const pruneMutation = useMutation({
    mutationFn: () => pruneMLData(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ml-data-stats'] }),
  })
  const deleteMutation = useMutation({
    mutationFn: deleteMLData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-data-stats'] })
      queryClient.invalidateQueries({ queryKey: ['ml-recorder'] })
      queryClient.invalidateQueries({ queryKey: ['ml-jobs'] })
    },
  })

  const errorMessage = getErrorMessage(updateMutation.error ?? pruneMutation.error ?? deleteMutation.error)

  if (isLoading) {
    return <div className="flex items-center justify-center p-8"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
  }

  return (
    <div className="space-y-5 p-4">
      <ErrorBanner message={errorMessage} />
      <div className="rounded-lg border border-border/50 bg-card/30 p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium">Runtime State</div>
            <div className="text-xs text-muted-foreground">
              ML runtime stays cold until a deployment is active. Recording can run independently when enabled.
            </div>
          </div>
          <Badge variant="outline" className={cn(capabilities?.runtime_active ? 'border-emerald-500/30 text-emerald-400' : 'border-border/50 text-muted-foreground')}>
            {capabilities?.runtime_active ? 'runtime active' : 'runtime idle'}
          </Badge>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <StatCard label="Active Deployments" value={capabilities?.active_deployment_count ?? 0} />
        <StatCard label="Recorded Snapshots" value={stats?.total_snapshots?.toLocaleString() ?? '0'} />
        <StatCard label="Oldest Snapshot" value={recorder?.oldest_snapshot ? new Date(recorder.oldest_snapshot).toLocaleDateString() : '-'} />
        <StatCard label="Newest Snapshot" value={recorder?.newest_snapshot ? new Date(recorder.newest_snapshot).toLocaleString() : '-'} />
      </div>

      <div className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium">Recorder</div>
            <div className="text-xs text-muted-foreground">Only the live recorder runs when you explicitly enable it.</div>
          </div>
          <Switch
            checked={Boolean(recorder?.config?.is_recording)}
            onCheckedChange={(checked) => updateMutation.mutate({ is_recording: checked })}
            disabled={updateMutation.isPending}
          />
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-xs">Interval (seconds)</Label>
            <div className="flex gap-2">
              <Input className="h-8 text-xs" type="number" min={5} max={3600} value={intervalSeconds} onChange={(event) => setIntervalSeconds(event.target.value)} />
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => updateMutation.mutate({ interval_seconds: Number(intervalSeconds) })} disabled={updateMutation.isPending}>Set</Button>
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Retention (days)</Label>
            <div className="flex gap-2">
              <Input className="h-8 text-xs" type="number" min={1} max={365} value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} />
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => updateMutation.mutate({ retention_days: Number(retentionDays) })} disabled={updateMutation.isPending}>Set</Button>
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Assets</Label>
            <div className="flex gap-2">
              <Input className="h-8 text-xs" value={assets} onChange={(event) => setAssets(event.target.value)} />
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => updateMutation.mutate({ assets: parseList(assets) })} disabled={updateMutation.isPending}>Set</Button>
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Timeframes</Label>
            <div className="flex gap-2">
              <Input className="h-8 text-xs" value={timeframes} onChange={(event) => setTimeframes(event.target.value)} />
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => updateMutation.mutate({ timeframes: parseList(timeframes) })} disabled={updateMutation.isPending}>Set</Button>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-3">
        <div className="text-sm font-medium">Recorded Scope</div>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
          {stats?.groups?.map((group) => (
            <div key={`${group.task_key}-${group.asset}-${group.timeframe}`} className="rounded-md border border-border/40 px-3 py-2 text-xs">
              <div className="font-medium uppercase">{group.asset}/{group.timeframe}</div>
              <div className="text-muted-foreground">{group.count.toLocaleString()} snapshots</div>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="text-xs" onClick={() => pruneMutation.mutate()} disabled={pruneMutation.isPending}>
            {pruneMutation.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Trash2 className="mr-1 h-3 w-3" />}
            Prune
          </Button>
          <Button size="sm" variant="outline" className="text-xs text-red-300" onClick={() => { if (confirm('Delete all recorded ML data?')) deleteMutation.mutate() }} disabled={deleteMutation.isPending}>
            {deleteMutation.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Trash2 className="mr-1 h-3 w-3" />}
            Delete All
          </Button>
        </div>
      </div>
    </div>
  )
}

function ImportTab() {
  const queryClient = useQueryClient()
  const { data: capabilities } = useQuery<MLCapabilities>({ queryKey: ['ml-capabilities'], queryFn: getMLCapabilities, refetchInterval: 15000 })
  const [sourceUri, setSourceUri] = useState('')
  const [manifestUri, setManifestUri] = useState('')
  const [backend, setBackend] = useState('sklearn_joblib')
  const [name, setName] = useState('')
  const [version, setVersion] = useState('1')
  const [featureNames, setFeatureNames] = useState('price, spread, combined, liquidity_log, volume_24h_log, seconds_left_norm, oracle_distance, ptb_distance, return_1, return_2, return_3, return_4, return_5, spread_change_1')
  const [assets, setAssets] = useState('btc, eth, sol, xrp')
  const [timeframes, setTimeframes] = useState('5m, 15m, 1h, 4h')

  const importMutation = useMutation({
    mutationFn: importMLModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] })
      queryClient.invalidateQueries({ queryKey: ['ml-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['ml-capabilities'] })
    },
  })

  const importError = getErrorMessage(importMutation.error)
  const backendOptions = capabilities?.import_backends ?? []

  return (
    <div className="space-y-5 p-4">
      <ErrorBanner message={importError} />
      <div className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-3">
        <div>
          <div className="text-sm font-medium">Import External Base Model</div>
          <div className="text-xs text-muted-foreground">Point Homerun at a local artifact and declare the feature contract it should use.</div>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-xs">Artifact Path</Label>
            <Input className="h-8 text-xs" value={sourceUri} onChange={(event) => setSourceUri(event.target.value)} placeholder="C:\\models\\crypto-directional.joblib" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Manifest Path (optional)</Label>
            <Input className="h-8 text-xs" value={manifestUri} onChange={(event) => setManifestUri(event.target.value)} placeholder="C:\\models\\manifest.json" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Backend</Label>
            <select value={backend} onChange={(event) => setBackend(event.target.value)} className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs">
              {backendOptions.map((item) => (
                <option key={item.backend} value={item.backend} disabled={!item.available}>
                  {item.label ?? item.backend}{item.available ? '' : ' (unavailable)'}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Name</Label>
            <Input className="h-8 text-xs" value={name} onChange={(event) => setName(event.target.value)} placeholder="crypto_directional_external" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Version</Label>
            <Input className="h-8 text-xs" value={version} onChange={(event) => setVersion(event.target.value)} />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Assets</Label>
            <Input className="h-8 text-xs" value={assets} onChange={(event) => setAssets(event.target.value)} />
          </div>
          <div className="space-y-1 md:col-span-2">
            <Label className="text-xs">Timeframes</Label>
            <Input className="h-8 text-xs" value={timeframes} onChange={(event) => setTimeframes(event.target.value)} />
          </div>
          <div className="space-y-1 md:col-span-2">
            <Label className="text-xs">Feature Names</Label>
            <Input className="h-8 text-xs" value={featureNames} onChange={(event) => setFeatureNames(event.target.value)} />
          </div>
        </div>
        <Button
          size="sm"
          className="text-xs"
          disabled={importMutation.isPending || !sourceUri.trim()}
          onClick={() => importMutation.mutate({
            source_uri: sourceUri.trim(),
            manifest_uri: manifestUri.trim() || undefined,
            backend,
            task_key: 'crypto_directional',
            name: name.trim() || undefined,
            version: version.trim() || undefined,
            metadata: {
              feature_names: parseList(featureNames),
              assets: parseList(assets),
              timeframes: parseList(timeframes),
            },
          })}
        >
          {importMutation.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Upload className="mr-1 h-3 w-3" />}
          Import Model
        </Button>
      </div>
    </div>
  )
}

function ModelsTab() {
  const queryClient = useQueryClient()
  const { data: models = [] } = useQuery<MLModel[]>({ queryKey: ['ml-models'], queryFn: () => getMLModels(), refetchInterval: 10000 })
  const archiveMutation = useMutation({
    mutationFn: archiveMLModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] })
      queryClient.invalidateQueries({ queryKey: ['ml-deployments'] })
    },
  })
  const deleteMutation = useMutation({
    mutationFn: deleteMLModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] })
      queryClient.invalidateQueries({ queryKey: ['ml-deployments'] })
    },
  })
  const evalMutation = useMutation({
    mutationFn: (modelId: string) => triggerMLEvaluation({ target_type: 'model', target_id: modelId }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ml-jobs'] }),
  })

  return (
    <div className="space-y-3 p-4">
      <ErrorBanner message={getErrorMessage(archiveMutation.error ?? deleteMutation.error ?? evalMutation.error)} />
      {models.map((model) => (
        <div key={model.id} className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{model.name}</span>
                <Badge variant="outline">{model.status}</Badge>
                <Badge variant="outline" className={cn(model.runtime_ready ? 'border-emerald-500/30 text-emerald-400' : 'text-muted-foreground')}>
                  {model.runtime_ready ? 'runtime ready' : 'runtime blocked'}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground">{model.backend} - v{model.version}</div>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <div>AUC {formatMetric(model.evaluation?.auc)}</div>
              <div>Acc {formatMetric(model.evaluation?.accuracy)}</div>
            </div>
          </div>
          <div className="text-xs text-muted-foreground">{model.availability_reason || 'Imported model artifact ready for activation.'}</div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" className="text-xs" onClick={() => evalMutation.mutate(model.id)} disabled={evalMutation.isPending}>Evaluate</Button>
            {model.status !== 'archived' ? <Button size="sm" variant="outline" className="text-xs" onClick={() => archiveMutation.mutate(model.id)} disabled={archiveMutation.isPending}><Archive className="mr-1 h-3 w-3" />Archive</Button> : null}
            <Button size="sm" variant="outline" className="text-xs text-red-300" onClick={() => { if (confirm('Delete this model?')) deleteMutation.mutate(model.id) }} disabled={deleteMutation.isPending}><Trash2 className="mr-1 h-3 w-3" />Delete</Button>
          </div>
        </div>
      ))}
      {models.length === 0 ? <div className="py-8 text-center text-sm text-muted-foreground">No imported models yet.</div> : null}
    </div>
  )
}

function AdaptersTab() {
  const queryClient = useQueryClient()
  const { data: models = [] } = useQuery<MLModel[]>({ queryKey: ['ml-models'], queryFn: () => getMLModels(), refetchInterval: 10000 })
  const { data: adapters = [] } = useQuery<MLAdapter[]>({ queryKey: ['ml-adapters'], queryFn: () => getMLAdapters(), refetchInterval: 10000 })
  const [baseModelId, setBaseModelId] = useState('')
  const [adapterKind, setAdapterKind] = useState('platt_scaler')
  const [name, setName] = useState('')
  const [trainingWindowDays, setTrainingWindowDays] = useState('90')
  const [holdoutDays, setHoldoutDays] = useState('7')

  useEffect(() => {
    if (!baseModelId && models[0]?.id) setBaseModelId(models[0].id)
  }, [models, baseModelId])

  const trainMutation = useMutation({
    mutationFn: trainMLAdapter,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['ml-adapters'] })
    },
  })
  const archiveMutation = useMutation({
    mutationFn: archiveMLAdapter,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ml-adapters'] }),
  })
  const deleteMutation = useMutation({
    mutationFn: deleteMLAdapter,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ml-adapters'] }),
  })
  const evalMutation = useMutation({
    mutationFn: (adapterId: string) => triggerMLEvaluation({ target_type: 'adapter', target_id: adapterId }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ml-jobs'] }),
  })

  return (
    <div className="space-y-5 p-4">
      <ErrorBanner message={getErrorMessage(trainMutation.error ?? archiveMutation.error ?? deleteMutation.error ?? evalMutation.error)} />
      <div className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-3">
        <div className="text-sm font-medium">Train Local Adapter</div>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-xs">Base Model</Label>
            <select value={baseModelId} onChange={(event) => setBaseModelId(event.target.value)} className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs">
              {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Adapter Kind</Label>
            <select value={adapterKind} onChange={(event) => setAdapterKind(event.target.value)} className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs">
              <option value="platt_scaler">Platt Scaler</option>
              <option value="residual_logistic">Residual Logistic</option>
            </select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Name</Label>
            <Input className="h-8 text-xs" value={name} onChange={(event) => setName(event.target.value)} placeholder="adapter_v1" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Training Window (days)</Label>
            <Input className="h-8 text-xs" type="number" min={7} max={365} value={trainingWindowDays} onChange={(event) => setTrainingWindowDays(event.target.value)} />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Holdout (days)</Label>
            <Input className="h-8 text-xs" type="number" min={1} max={90} value={holdoutDays} onChange={(event) => setHoldoutDays(event.target.value)} />
          </div>
        </div>
        <Button
          size="sm"
          className="text-xs"
          disabled={trainMutation.isPending || !baseModelId}
          onClick={() => trainMutation.mutate({
            task_key: 'crypto_directional',
            base_model_id: baseModelId,
            adapter_kind: adapterKind,
            name: name.trim() || undefined,
            training_window_days: Number(trainingWindowDays),
            holdout_days: Number(holdoutDays),
          })}
        >
          {trainMutation.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Brain className="mr-1 h-3 w-3" />}
          Train Adapter
        </Button>
      </div>

      {adapters.map((adapter) => (
        <div key={adapter.id} className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-2">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{adapter.name}</span>
                <Badge variant="outline">{adapter.status}</Badge>
              </div>
              <div className="text-xs text-muted-foreground">{adapter.adapter_kind} - base {adapter.base_model_name ?? adapter.base_model_id}</div>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <div>AUC {formatMetric(adapter.evaluation?.auc)}</div>
              <div>Acc {formatMetric(adapter.evaluation?.accuracy)}</div>
            </div>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" className="text-xs" onClick={() => evalMutation.mutate(adapter.id)} disabled={evalMutation.isPending}>Evaluate</Button>
            {adapter.status !== 'archived' ? <Button size="sm" variant="outline" className="text-xs" onClick={() => archiveMutation.mutate(adapter.id)} disabled={archiveMutation.isPending}><Archive className="mr-1 h-3 w-3" />Archive</Button> : null}
            <Button size="sm" variant="outline" className="text-xs text-red-300" onClick={() => { if (confirm('Delete this adapter?')) deleteMutation.mutate(adapter.id) }} disabled={deleteMutation.isPending}><Trash2 className="mr-1 h-3 w-3" />Delete</Button>
          </div>
        </div>
      ))}
      {adapters.length === 0 ? <div className="py-8 text-center text-sm text-muted-foreground">No adapters trained yet.</div> : null}
    </div>
  )
}

function DeploymentCard({
  deployment,
  models,
  adapters,
  isPending,
  onSave,
}: {
  deployment: MLDeployment
  models: MLModel[]
  adapters: MLAdapter[]
  isPending: boolean
  onSave: (taskKey: string, payload: { base_model_id?: string | null; adapter_id?: string | null; is_active: boolean }) => void
}) {
  const baseOptions = useMemo(
    () => models.filter((model) => model.task_key === deployment.task_key && model.status === 'ready'),
    [deployment.task_key, models]
  )
  const [selectedModelId, setSelectedModelId] = useState(deployment.base_model_id ?? baseOptions[0]?.id ?? '')
  const adapterOptions = useMemo(
    () => adapters.filter((adapter) => adapter.task_key === deployment.task_key && (!selectedModelId || adapter.base_model_id === selectedModelId)),
    [adapters, deployment.task_key, selectedModelId]
  )
  const [selectedAdapterId, setSelectedAdapterId] = useState(deployment.adapter_id ?? '')

  useEffect(() => {
    const nextModelId = deployment.base_model_id ?? baseOptions[0]?.id ?? ''
    setSelectedModelId(nextModelId)
  }, [deployment.base_model_id, deployment.updated_at, baseOptions])

  useEffect(() => {
    if (deployment.base_model_id === selectedModelId && deployment.adapter_id && adapterOptions.some((adapter) => adapter.id === deployment.adapter_id)) {
      setSelectedAdapterId(deployment.adapter_id)
      return
    }
    setSelectedAdapterId((current) => (adapterOptions.some((adapter) => adapter.id === current) ? current : ''))
  }, [adapterOptions, deployment.adapter_id, deployment.base_model_id, deployment.updated_at, selectedModelId])

  return (
    <div className="rounded-lg border border-border/50 bg-card/30 p-4 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="font-medium">{deployment.name ?? deployment.task_key}</div>
          <div className="text-xs text-muted-foreground">{deployment.runtime_message ?? 'No deployment configured.'}</div>
        </div>
        <Badge variant="outline" className={cn(deployment.is_active ? 'border-emerald-500/30 text-emerald-400' : 'text-muted-foreground')}>
          {deployment.is_active ? 'active' : 'inactive'}
        </Badge>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <Label className="text-xs">Base Model</Label>
          <select
            value={selectedModelId}
            className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs"
            onChange={(event) => {
              const nextModelId = event.target.value
              setSelectedModelId(nextModelId)
              setSelectedAdapterId('')
            }}
          >
            <option value="">No model</option>
            {baseOptions.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Adapter</Label>
          <select
            value={selectedAdapterId}
            className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs"
            onChange={(event) => setSelectedAdapterId(event.target.value)}
          >
            <option value="">No adapter</option>
            {adapterOptions.map((adapter) => <option key={adapter.id} value={adapter.id}>{adapter.name}</option>)}
          </select>
        </div>
      </div>
      <div className="flex gap-2">
        <Button
          size="sm"
          className="text-xs"
          disabled={isPending || !selectedModelId}
          onClick={() => onSave(deployment.task_key, { base_model_id: selectedModelId, adapter_id: selectedAdapterId || null, is_active: true })}
        >
          Activate
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="text-xs"
          disabled={isPending}
          onClick={() => onSave(deployment.task_key, { is_active: false })}
        >
          Deactivate
        </Button>
      </div>
    </div>
  )
}

function DeploymentsTab() {
  const queryClient = useQueryClient()
  const { data: deployments = [] } = useQuery<MLDeployment[]>({ queryKey: ['ml-deployments'], queryFn: getMLDeployments, refetchInterval: 10000 })
  const { data: models = [] } = useQuery<MLModel[]>({ queryKey: ['ml-models'], queryFn: () => getMLModels(), refetchInterval: 10000 })
  const { data: adapters = [] } = useQuery<MLAdapter[]>({ queryKey: ['ml-adapters'], queryFn: () => getMLAdapters(), refetchInterval: 10000 })
  const deploymentMutation = useMutation({
    mutationFn: ({ taskKey, payload }: { taskKey: string; payload: { base_model_id?: string | null; adapter_id?: string | null; is_active: boolean } }) => updateMLDeployment(taskKey, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-deployments'] })
      queryClient.invalidateQueries({ queryKey: ['ml-capabilities'] })
      queryClient.invalidateQueries({ queryKey: ['ml-jobs'] })
    },
  })

  return (
    <div className="space-y-4 p-4">
      <ErrorBanner message={getErrorMessage(deploymentMutation.error)} />
      {deployments.map((deployment) => (
        <DeploymentCard
          key={deployment.task_key}
          deployment={deployment}
          models={models}
          adapters={adapters}
          isPending={deploymentMutation.isPending}
          onSave={(taskKey, payload) => deploymentMutation.mutate({ taskKey, payload })}
        />
      ))}
    </div>
  )
}

function JobsTab() {
  const { data: jobs = [] } = useQuery<MLJob[]>({ queryKey: ['ml-jobs'], queryFn: () => getMLJobs({ limit: 50 }), refetchInterval: 5000 })
  return (
    <div className="space-y-2 p-4">
      {jobs.map((job) => (
        <div key={job.id} className="rounded-md border border-border/40 bg-card/30 px-3 py-2 text-xs">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{job.status}</Badge>
              <span className="font-medium">{job.kind}</span>
            </div>
            <span className="text-muted-foreground">{job.created_at ? new Date(job.created_at).toLocaleString() : ''}</span>
          </div>
          {job.message ? <div className="mt-1 text-muted-foreground">{job.message}</div> : null}
          {job.error ? <div className="mt-1 text-red-300">{job.error}</div> : null}
        </div>
      ))}
      {jobs.length === 0 ? <div className="py-8 text-center text-sm text-muted-foreground">No ML jobs yet.</div> : null}
    </div>
  )
}

export default function MLModelsPanel() {
  const [activeTab, setActiveTab] = useState<TabId>('fill-model')
  const { data: capabilities } = useQuery<MLCapabilities>({ queryKey: ['ml-capabilities'], queryFn: getMLCapabilities, refetchInterval: 15000 })
  const headerNote = useMemo(() => capabilities?.notes?.[0] ?? 'External models load lazily and only affect trading when you activate a deployment.', [capabilities])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-border/50 px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium">Machine Learning</div>
            <div className="text-xs text-muted-foreground">{headerNote}</div>
          </div>
          <Badge variant="outline" className={cn(capabilities?.runtime_active ? 'border-emerald-500/30 text-emerald-400' : 'text-muted-foreground')}>
            {capabilities?.runtime_active ? 'active deployment' : 'idle'}
          </Badge>
        </div>
        <div className="mt-3 flex flex-wrap gap-1">
          {TABS.map((tab) => {
            const Icon = tab.icon
            const active = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                  active ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:bg-muted/30 hover:text-foreground'
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {tab.label}
              </button>
            )
          })}
        </div>
      </div>
      <ScrollArea className="flex-1 min-h-0">
        {activeTab === 'fill-model' ? <FillModelPanel /> : null}
        {activeTab === 'data' ? <DataTab /> : null}
        {activeTab === 'import' ? <ImportTab /> : null}
        {activeTab === 'models' ? <ModelsTab /> : null}
        {activeTab === 'adapters' ? <AdaptersTab /> : null}
        {activeTab === 'deployments' ? <DeploymentsTab /> : null}
        {activeTab === 'jobs' ? <JobsTab /> : null}
      </ScrollArea>
    </div>
  )
}
