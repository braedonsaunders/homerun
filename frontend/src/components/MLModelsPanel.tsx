import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
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
import { cn } from '../lib/utils'
import {
  archiveMLAdapter,
  archiveMLModel,
  deleteMLAdapter,
  deleteMLModel,
  getMLAdapters,
  getMLCapabilities,
  getMLDeployments,
  getMLJobs,
  getMLModels,
  importMLModel,
  trainMLAdapter,
  triggerMLEvaluation,
  updateMLDeployment,
  type MLAdapter,
  type MLCapabilities,
  type MLDeployment,
  type MLJob,
  type MLModel,
} from '../services/apiMachineLearning'
import {
  listRecordingSessions,
  type RecordingSession,
} from '../services/apiDataset'

// Data tab removed — recording controls + dataset browsing live in
// Research → Data Lab.  ML training can pull from the same Data Lab
// API (see /api/dataset/* and the Cox PH trainer's existing direct
// reads of MarketMicrostructureSnapshot + TraderOrder).
type TabId = 'fill-model' | 'import' | 'models' | 'adapters' | 'deployments' | 'jobs'

const TABS: { id: TabId; labelKey: string; icon: typeof Database }[] = [
  { id: 'fill-model', labelKey: 'mlModels.tabFillModel', icon: Sparkles },
  { id: 'import', labelKey: 'mlModels.tabImport', icon: Upload },
  { id: 'models', labelKey: 'mlModels.tabModels', icon: Brain },
  { id: 'adapters', labelKey: 'mlModels.tabAdapters', icon: Activity },
  { id: 'deployments', labelKey: 'mlModels.tabDeploy', icon: Rocket },
  { id: 'jobs', labelKey: 'mlModels.tabJobs', icon: Activity },
]


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

function ImportTab() {
  const { t } = useTranslation()
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
          <div className="text-sm font-medium">{t('mlModels.importTitle')}</div>
          <div className="text-xs text-muted-foreground">{t('mlModels.importDesc')}</div>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.artifactPath')}</Label>
            <Input className="h-8 text-xs" value={sourceUri} onChange={(event) => setSourceUri(event.target.value)} placeholder="C:\\models\\crypto-directional.joblib" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.manifestPath')}</Label>
            <Input className="h-8 text-xs" value={manifestUri} onChange={(event) => setManifestUri(event.target.value)} placeholder="C:\\models\\manifest.json" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.backend')}</Label>
            <select value={backend} onChange={(event) => setBackend(event.target.value)} className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs">
              {backendOptions.map((item) => (
                <option key={item.backend} value={item.backend} disabled={!item.available}>
                  {item.label ?? item.backend}{item.available ? '' : ` ${t('mlModels.unavailableSuffix')}`}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.name')}</Label>
            <Input className="h-8 text-xs" value={name} onChange={(event) => setName(event.target.value)} placeholder="crypto_directional_external" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.version')}</Label>
            <Input className="h-8 text-xs" value={version} onChange={(event) => setVersion(event.target.value)} />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.assets')}</Label>
            <Input className="h-8 text-xs" value={assets} onChange={(event) => setAssets(event.target.value)} />
          </div>
          <div className="space-y-1 md:col-span-2">
            <Label className="text-xs">{t('mlModels.timeframes')}</Label>
            <Input className="h-8 text-xs" value={timeframes} onChange={(event) => setTimeframes(event.target.value)} />
          </div>
          <div className="space-y-1 md:col-span-2">
            <Label className="text-xs">{t('mlModels.featureNames')}</Label>
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
          {t('mlModels.importModel')}
        </Button>
      </div>
    </div>
  )
}

function ModelsTab() {
  const { t } = useTranslation()
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
                  {model.runtime_ready ? t('mlModels.runtimeReady') : t('mlModels.runtimeBlocked')}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground">{model.backend} - v{model.version}</div>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <div>AUC {formatMetric(model.evaluation?.auc)}</div>
              <div>Acc {formatMetric(model.evaluation?.accuracy)}</div>
            </div>
          </div>
          <div className="text-xs text-muted-foreground">{model.availability_reason || t('mlModels.importedReady')}</div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" className="text-xs" onClick={() => evalMutation.mutate(model.id)} disabled={evalMutation.isPending}>{t('mlModels.evaluate')}</Button>
            {model.status !== 'archived' ? <Button size="sm" variant="outline" className="text-xs" onClick={() => archiveMutation.mutate(model.id)} disabled={archiveMutation.isPending}><Archive className="mr-1 h-3 w-3" />{t('mlModels.archive')}</Button> : null}
            <Button size="sm" variant="outline" className="text-xs text-red-300" onClick={() => { if (confirm(t('mlModels.confirmDeleteModel'))) deleteMutation.mutate(model.id) }} disabled={deleteMutation.isPending}><Trash2 className="mr-1 h-3 w-3" />{t('mlModels.delete')}</Button>
          </div>
        </div>
      ))}
      {models.length === 0 ? <div className="py-8 text-center text-sm text-muted-foreground">{t('mlModels.noModelsYet')}</div> : null}
    </div>
  )
}

function AdaptersTab() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const { data: models = [] } = useQuery<MLModel[]>({ queryKey: ['ml-models'], queryFn: () => getMLModels(), refetchInterval: 10000 })
  const { data: adapters = [] } = useQuery<MLAdapter[]>({ queryKey: ['ml-adapters'], queryFn: () => getMLAdapters(), refetchInterval: 10000 })
  // Recording sessions usable as training sources — anything that's
  // captured rows (running, completed, paused).  Pending/scheduled
  // have no rows yet so they're useless for training.
  const { data: trainableSessions = [] } = useQuery<RecordingSession[]>({
    queryKey: ['ml-trainable-sessions'],
    queryFn: () => listRecordingSessions(['running', 'completed', 'paused'], 100),
    refetchInterval: 30_000,
  })
  const [baseModelId, setBaseModelId] = useState('')
  const [adapterKind, setAdapterKind] = useState('platt_scaler')
  const [name, setName] = useState('')
  const [trainingWindowDays, setTrainingWindowDays] = useState('90')
  const [holdoutDays, setHoldoutDays] = useState('7')
  const [trainingSourceSessionId, setTrainingSourceSessionId] = useState<string>('')

  const selectedSession = useMemo(
    () => trainableSessions.find((s) => s.id === trainingSourceSessionId) ?? null,
    [trainableSessions, trainingSourceSessionId],
  )

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
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium">{t('mlModels.trainLocalAdapter')}</div>
        </div>

        {/* Training source — Global vs Recording session */}
        <div className="space-y-1">
          <Label className="text-xs">{t('mlModels.trainingSource')}</Label>
          <div className="flex gap-1.5">
            <button
              type="button"
              onClick={() => setTrainingSourceSessionId('')}
              className={cn(
                'rounded-md border px-3 py-1.5 text-xs transition-colors',
                trainingSourceSessionId === ''
                  ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                  : 'border-border/40 text-muted-foreground hover:text-foreground',
              )}
            >
              {t('mlModels.globalRollingWindow')}
            </button>
            <button
              type="button"
              onClick={() => {
                if (trainableSessions.length > 0) {
                  setTrainingSourceSessionId(trainableSessions[0].id)
                }
              }}
              disabled={trainableSessions.length === 0}
              className={cn(
                'rounded-md border px-3 py-1.5 text-xs transition-colors',
                trainingSourceSessionId !== ''
                  ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                  : 'border-border/40 text-muted-foreground hover:text-foreground disabled:opacity-50',
              )}
            >
              {t('mlModels.recordingSession', { n: trainableSessions.length })}
            </button>
          </div>
          {trainingSourceSessionId !== '' ? (
            <div className="mt-1 space-y-1">
              <select
                value={trainingSourceSessionId}
                onChange={(e) => setTrainingSourceSessionId(e.target.value)}
                className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs"
              >
                {trainableSessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} — {s.status} · {s.rows_captured.toLocaleString()} rows
                  </option>
                ))}
              </select>
              {selectedSession ? (
                <div className="rounded-md border border-violet-500/20 bg-violet-500/5 px-3 py-2 text-[11px] text-violet-100">
                  <div className="flex items-center gap-2">
                    <span className="rounded-sm bg-violet-500/20 px-1.5 py-0 text-[10px] uppercase tracking-wide">
                      {selectedSession.status}
                    </span>
                    <span className="font-medium">{selectedSession.name}</span>
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-2 text-[10px] text-muted-foreground md:grid-cols-4">
                    <div>
                      <span className="block text-muted-foreground/70">{t('mlModels.targets')}</span>
                      <span className="font-mono">
                        {selectedSession.target_kind} · {t('mlModels.tokenCount', { n: selectedSession.target_token_ids.length })}
                      </span>
                    </div>
                    <div>
                      <span className="block text-muted-foreground/70">{t('mlModels.capture')}</span>
                      <span className="font-mono">{selectedSession.capture_types.join(', ')}</span>
                    </div>
                    <div>
                      <span className="block text-muted-foreground/70">{t('mlModels.rowsCaptured')}</span>
                      <span className="font-mono">
                        {selectedSession.rows_captured.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="block text-muted-foreground/70">{t('mlModels.window')}</span>
                      <span className="font-mono">
                        {selectedSession.started_at
                          ? new Date(selectedSession.started_at).toLocaleString()
                          : '—'}
                        {selectedSession.ended_at
                          ? ` → ${new Date(selectedSession.ended_at).toLocaleString()}`
                          : selectedSession.status === 'running'
                          ? ` → ${t('mlModels.now')}`
                          : ''}
                      </span>
                    </div>
                  </div>
                  <div className="mt-1 text-[10px] text-violet-700 dark:text-violet-200/70">
                    {t('mlModels.sessionAttribution')}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.baseModel')}</Label>
            <select value={baseModelId} onChange={(event) => setBaseModelId(event.target.value)} className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs">
              {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.adapterKind')}</Label>
            <select value={adapterKind} onChange={(event) => setAdapterKind(event.target.value)} className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs">
              <option value="platt_scaler">{t('mlModels.plattScaler')}</option>
              <option value="residual_logistic">{t('mlModels.residualLogistic')}</option>
            </select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.name')}</Label>
            <Input className="h-8 text-xs" value={name} onChange={(event) => setName(event.target.value)} placeholder="adapter_v1" />
          </div>
          <div className="space-y-1">
            <Label
              className={cn(
                'text-xs',
                trainingSourceSessionId !== '' && 'text-muted-foreground/50',
              )}
            >
              {t('mlModels.trainingWindowDays')}
              {trainingSourceSessionId !== '' ? (
                <span className="ml-1 text-[10px] italic">{t('mlModels.overriddenBySession')}</span>
              ) : null}
            </Label>
            <Input
              className="h-8 text-xs"
              type="number"
              min={7}
              max={365}
              value={trainingWindowDays}
              onChange={(event) => setTrainingWindowDays(event.target.value)}
              disabled={trainingSourceSessionId !== ''}
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">{t('mlModels.holdoutDays')}</Label>
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
            recording_session_id: trainingSourceSessionId || undefined,
          })}
        >
          {trainMutation.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Brain className="mr-1 h-3 w-3" />}
          {trainingSourceSessionId !== '' && selectedSession
            ? t('mlModels.trainOnSession', { name: selectedSession.name })
            : t('mlModels.trainAdapter')}
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
              <div className="text-xs text-muted-foreground">{adapter.adapter_kind} - {t('mlModels.baseLabel')} {adapter.base_model_name ?? adapter.base_model_id}</div>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <div>AUC {formatMetric(adapter.evaluation?.auc)}</div>
              <div>{t('mlModels.accLabel')} {formatMetric(adapter.evaluation?.accuracy)}</div>
            </div>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" className="text-xs" onClick={() => evalMutation.mutate(adapter.id)} disabled={evalMutation.isPending}>{t('mlModels.evaluate')}</Button>
            {adapter.status !== 'archived' ? <Button size="sm" variant="outline" className="text-xs" onClick={() => archiveMutation.mutate(adapter.id)} disabled={archiveMutation.isPending}><Archive className="mr-1 h-3 w-3" />{t('mlModels.archive')}</Button> : null}
            <Button size="sm" variant="outline" className="text-xs text-red-300" onClick={() => { if (confirm(t('mlModels.confirmDeleteAdapter'))) deleteMutation.mutate(adapter.id) }} disabled={deleteMutation.isPending}><Trash2 className="mr-1 h-3 w-3" />{t('mlModels.delete')}</Button>
          </div>
        </div>
      ))}
      {adapters.length === 0 ? <div className="py-8 text-center text-sm text-muted-foreground">{t('mlModels.noAdaptersYet')}</div> : null}
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
  const { t } = useTranslation()
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
          <div className="text-xs text-muted-foreground">{deployment.runtime_message ?? t('mlModels.noDeploymentConfigured')}</div>
        </div>
        <Badge variant="outline" className={cn(deployment.is_active ? 'border-emerald-500/30 text-emerald-400' : 'text-muted-foreground')}>
          {deployment.is_active ? t('mlModels.active') : t('mlModels.inactive')}
        </Badge>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <Label className="text-xs">{t('mlModels.baseModel')}</Label>
          <select
            value={selectedModelId}
            className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs"
            onChange={(event) => {
              const nextModelId = event.target.value
              setSelectedModelId(nextModelId)
              setSelectedAdapterId('')
            }}
          >
            <option value="">{t('mlModels.noModel')}</option>
            {baseOptions.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">{t('mlModels.adapter')}</Label>
          <select
            value={selectedAdapterId}
            className="h-8 w-full rounded-md border border-input bg-background px-3 text-xs"
            onChange={(event) => setSelectedAdapterId(event.target.value)}
          >
            <option value="">{t('mlModels.noAdapter')}</option>
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
          {t('mlModels.activate')}
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="text-xs"
          disabled={isPending}
          onClick={() => onSave(deployment.task_key, { is_active: false })}
        >
          {t('mlModels.deactivate')}
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
  const { t } = useTranslation()
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
      {jobs.length === 0 ? <div className="py-8 text-center text-sm text-muted-foreground">{t('mlModels.noJobsYet')}</div> : null}
    </div>
  )
}

export default function MLModelsPanel() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState<TabId>('fill-model')
  const { data: capabilities } = useQuery<MLCapabilities>({ queryKey: ['ml-capabilities'], queryFn: getMLCapabilities, refetchInterval: 15000 })
  const headerNote = useMemo(() => capabilities?.notes?.[0] ?? t('mlModels.headerNoteDefault'), [capabilities, t])

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header — match ValidationPanel / AutoresearchPanel sizing
          (px-4 py-2 + leading-tight text + 10px subtitle) so ML tab
          doesn't visually "zoom in" relative to its siblings. */}
      <div className="border-b border-border/40 shrink-0">
        <div className="flex items-center gap-3 px-4 py-2">
          <Brain className="w-4 h-4 text-violet-400 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold leading-tight">{t('mlModels.title')}</p>
            <p className="text-[10px] text-muted-foreground leading-tight">
              {headerNote}
              <span className="ml-1 inline-flex items-center gap-0.5 text-violet-700 dark:text-violet-300/80">
                <Database className="h-2.5 w-2.5" />
                {t('mlModels.trainingDataLabel')} <span className="font-medium">{t('mlModels.trainingDataLink')}</span>
                {' '}(microstructure_snapshot, book_delta_event, trader_order)
              </span>
            </p>
          </div>
          <Badge variant="outline" className={cn('shrink-0', capabilities?.runtime_active ? 'border-emerald-500/30 text-emerald-400' : 'text-muted-foreground')}>
            {capabilities?.runtime_active ? t('mlModels.activeDeployment') : t('mlModels.idle')}
          </Badge>
        </div>
        <div className="flex flex-wrap gap-1 px-4 pb-2">
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
                {t(tab.labelKey)}
              </button>
            )
          })}
        </div>
      </div>
      <ScrollArea className="flex-1 min-h-0">
        {activeTab === 'fill-model' ? <FillModelPanel /> : null}
        {activeTab === 'import' ? <ImportTab /> : null}
        {activeTab === 'models' ? <ModelsTab /> : null}
        {activeTab === 'adapters' ? <AdaptersTab /> : null}
        {activeTab === 'deployments' ? <DeploymentsTab /> : null}
        {activeTab === 'jobs' ? <JobsTab /> : null}
      </ScrollArea>
    </div>
  )
}
