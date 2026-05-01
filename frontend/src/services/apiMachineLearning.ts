import { api, unwrapApiData } from './apiClient'

export interface MLBackendCapability {
  backend: string
  label?: string | null
  available: boolean
  reason?: string | null
  lazy?: boolean | null
  supported_formats?: string[] | null
}

export interface MLTaskCapability {
  task_key: string
  label?: string | null
  description?: string | null
  supports_recording?: boolean
  supports_import?: boolean
  supports_adapters?: boolean
  supports_evaluation?: boolean
}

export interface MLAdapterKindCapability {
  kind: string
  label?: string | null
  description?: string | null
}

export interface MLCapabilities {
  available: boolean
  runtime_active: boolean
  active_deployment_count: number
  import_backends: MLBackendCapability[]
  tasks: MLTaskCapability[]
  adapter_kinds: MLAdapterKindCapability[]
  notes?: string[] | null
}

export interface MLRecorderConfig {
  task_key?: string
  is_recording: boolean
  interval_seconds: number
  retention_days: number
  assets?: string[]
  timeframes?: string[]
  schedule_enabled?: boolean
  schedule_start_utc?: string | null
  schedule_end_utc?: string | null
}

export interface MLRecorderStats {
  config: MLRecorderConfig
  total_snapshots: number
  oldest_snapshot: string | null
  newest_snapshot: string | null
  last_recorded_at?: string | null
  last_error?: string | null
}

export interface MLDataStatsGroup {
  task_key?: string | null
  asset?: string | null
  timeframe?: string | null
  count: number
  labeled_count?: number | null
  newest_at?: string | null
}

export interface MLDataStats {
  total_snapshots: number
  total_labels?: number | null
  total_size_bytes?: number | null
  last_pruned_at?: string | null
  groups: MLDataStatsGroup[]
}

export interface MLEvaluationSummary {
  sample_count?: number | null
  accuracy?: number | null
  auc?: number | null
  brier_score?: number | null
  calibration_error?: number | null
  updated_at?: string | null
}

export interface MLModel {
  id: string
  name: string
  version?: string | null
  task_key: string
  backend: string
  model_kind?: string | null
  status: string
  source_uri?: string | null
  manifest_uri?: string | null
  runtime_ready?: boolean | null
  availability_reason?: string | null
  evaluation?: MLEvaluationSummary | null
  deployment_count?: number | null
  imported_at?: string | null
  created_at?: string | null
  archived_at?: string | null
  metadata?: Record<string, unknown> | null
}

export interface MLAdapter {
  id: string
  name: string
  version?: string | null
  task_key: string
  adapter_kind: string
  base_model_id: string
  base_model_name?: string | null
  status: string
  runtime_ready?: boolean | null
  availability_reason?: string | null
  evaluation?: MLEvaluationSummary | null
  training_window_days?: number | null
  holdout_days?: number | null
  created_at?: string | null
  archived_at?: string | null
  metadata?: Record<string, unknown> | null
}

export interface MLDeployment {
  task_key: string
  name?: string | null
  is_active: boolean
  base_model_id?: string | null
  base_model_name?: string | null
  adapter_id?: string | null
  adapter_name?: string | null
  runtime_state?: string | null
  runtime_message?: string | null
  activated_at?: string | null
  updated_at?: string | null
  evaluation?: MLEvaluationSummary | null
}

export interface MLJob {
  id: string
  kind: string
  status: string
  task_key?: string | null
  model_id?: string | null
  adapter_id?: string | null
  deployment_task_key?: string | null
  progress?: number | null
  message?: string | null
  error?: string | null
  created_at?: string | null
  updated_at?: string | null
  completed_at?: string | null
  result?: Record<string, unknown> | null
}

export interface MLAsyncResult {
  status: string
  job_id?: string | null
  model_id?: string | null
  adapter_id?: string | null
  message?: string | null
}

export interface MLImportModelRequest {
  source_uri: string
  backend: string
  task_key: string
  name?: string
  version?: string
  manifest_uri?: string
  metadata?: Record<string, unknown>
  options?: Record<string, unknown>
}

export interface MLTrainAdapterRequest {
  task_key: string
  base_model_id: string
  adapter_kind: string
  name?: string
  training_window_days?: number
  holdout_days?: number
  params?: Record<string, unknown>
  // When set, the trainer scopes its rows to the session's
  // (target_token_ids × started_at..ended_at) slice instead of
  // the rolling training_window_days lookback.  Preserved in the
  // resulting adapter's training_source manifest.
  recording_session_id?: string
}

export interface MLUpdateDeploymentRequest {
  base_model_id?: string | null
  adapter_id?: string | null
  is_active: boolean
}

export interface MLTriggerEvaluationRequest {
  target_type: 'model' | 'adapter' | 'deployment'
  target_id: string
}

export interface MLMarketPrediction {
  probability_yes: number
  probability_no: number
  base_probability_yes?: number | null
  confidence?: number | null
  predicted_at?: string | null
}

export interface MLMarketRuntimePayload {
  task_key: string
  deployment_id: string
  base_model_id: string
  base_model_name?: string | null
  backend: string
  adapter_id?: string | null
  adapter_name?: string | null
  adapter_type?: string | null
  prediction: MLMarketPrediction
}

function unwrapNamedList<T>(data: unknown, key: string): T[] {
  const unwrapped = unwrapApiData(data)
  if (Array.isArray(unwrapped)) {
    return unwrapped as T[]
  }
  if (unwrapped && typeof unwrapped === 'object') {
    const value = (unwrapped as Record<string, unknown>)[key]
    if (Array.isArray(value)) {
      return value as T[]
    }
  }
  return []
}

function unwrapNamedObject<T>(data: unknown, key: string): T {
  const unwrapped = unwrapApiData(data)
  if (unwrapped && typeof unwrapped === 'object' && key in (unwrapped as Record<string, unknown>)) {
    return (unwrapped as Record<string, unknown>)[key] as T
  }
  return unwrapped as T
}

export const getMLCapabilities = async (): Promise<MLCapabilities> => {
  const { data } = await api.get('/ml/capabilities')
  const payload = unwrapApiData(data) as Partial<MLCapabilities> | undefined
  return {
    available: Boolean(payload?.available),
    runtime_active: Boolean(payload?.runtime_active),
    active_deployment_count: Number(payload?.active_deployment_count ?? 0),
    import_backends: Array.isArray(payload?.import_backends) ? payload!.import_backends : [],
    tasks: Array.isArray(payload?.tasks) ? payload!.tasks : [],
    adapter_kinds: Array.isArray(payload?.adapter_kinds) ? payload!.adapter_kinds : [],
    notes: Array.isArray(payload?.notes) ? payload!.notes : [],
  }
}

export const getMLRecorderConfig = async (): Promise<MLRecorderStats> => {
  const { data } = await api.get('/ml/recorder/config')
  return unwrapApiData(data)
}

export const updateMLRecorderConfig = async (
  config: Partial<MLRecorderConfig>
): Promise<MLRecorderStats> => {
  const { data } = await api.put('/ml/recorder/config', config)
  return unwrapApiData(data)
}

export const getMLDataStats = async (): Promise<MLDataStats> => {
  const { data } = await api.get('/ml/data/stats')
  const payload = unwrapApiData(data) as Partial<MLDataStats> | undefined
  return {
    total_snapshots: Number(payload?.total_snapshots ?? 0),
    total_labels: payload?.total_labels ?? null,
    total_size_bytes: payload?.total_size_bytes ?? null,
    last_pruned_at: payload?.last_pruned_at ?? null,
    groups: Array.isArray(payload?.groups) ? payload!.groups : [],
  }
}

export const pruneMLData = async (olderThanDays?: number): Promise<{ status: string; deleted_rows?: number }> => {
  const { data } = await api.post('/ml/data/prune', { older_than_days: olderThanDays, confirm: true })
  return unwrapApiData(data)
}

export const deleteMLData = async (): Promise<{ status: string; deleted_rows?: number }> => {
  const { data } = await api.delete('/ml/data', { params: { confirm: true } })
  return unwrapApiData(data)
}

export const importMLModel = async (payload: MLImportModelRequest): Promise<MLAsyncResult> => {
  const { data } = await api.post('/ml/models/import', payload)
  return unwrapApiData(data)
}

export const getMLModels = async (status?: string): Promise<MLModel[]> => {
  const { data } = await api.get('/ml/models', { params: { status } })
  return unwrapNamedList<MLModel>(data, 'models')
}

export const getMLModel = async (modelId: string): Promise<MLModel> => {
  const { data } = await api.get(`/ml/models/${modelId}`)
  return unwrapNamedObject<MLModel>(data, 'model')
}

export const archiveMLModel = async (modelId: string): Promise<{ status: string; model_id: string }> => {
  const { data } = await api.post(`/ml/models/${modelId}/archive`)
  return unwrapApiData(data)
}

export const deleteMLModel = async (modelId: string): Promise<{ status: string; model_id: string }> => {
  const { data } = await api.delete(`/ml/models/${modelId}`, { params: { confirm: true } })
  return unwrapApiData(data)
}

export const trainMLAdapter = async (payload: MLTrainAdapterRequest): Promise<MLAsyncResult> => {
  const { data } = await api.post('/ml/adapters/train', payload)
  return unwrapApiData(data)
}

export const getMLAdapters = async (status?: string): Promise<MLAdapter[]> => {
  const { data } = await api.get('/ml/adapters', { params: { status } })
  return unwrapNamedList<MLAdapter>(data, 'adapters')
}

export const getMLAdapter = async (adapterId: string): Promise<MLAdapter> => {
  const { data } = await api.get(`/ml/adapters/${adapterId}`)
  return unwrapNamedObject<MLAdapter>(data, 'adapter')
}

export const archiveMLAdapter = async (adapterId: string): Promise<{ status: string; adapter_id: string }> => {
  const { data } = await api.post(`/ml/adapters/${adapterId}/archive`)
  return unwrapApiData(data)
}

export const deleteMLAdapter = async (adapterId: string): Promise<{ status: string; adapter_id: string }> => {
  const { data } = await api.delete(`/ml/adapters/${adapterId}`, { params: { confirm: true } })
  return unwrapApiData(data)
}

export const getMLDeployments = async (): Promise<MLDeployment[]> => {
  const { data } = await api.get('/ml/deployments')
  return unwrapNamedList<MLDeployment>(data, 'deployments')
}

export const getMLDeployment = async (taskKey: string): Promise<MLDeployment> => {
  const { data } = await api.get(`/ml/deployments/${taskKey}`)
  return unwrapNamedObject<MLDeployment>(data, 'deployment')
}

export const updateMLDeployment = async (
  taskKey: string,
  payload: MLUpdateDeploymentRequest
): Promise<MLDeployment> => {
  const { data } = await api.put(`/ml/deployments/${taskKey}`, payload)
  return unwrapNamedObject<MLDeployment>(data, 'deployment')
}

export const getMLJobs = async (params?: {
  kind?: string
  status?: string
  limit?: number
}): Promise<MLJob[]> => {
  const { data } = await api.get('/ml/jobs', { params })
  return unwrapNamedList<MLJob>(data, 'jobs')
}

export const getMLJob = async (jobId: string): Promise<MLJob> => {
  const { data } = await api.get(`/ml/jobs/${jobId}`)
  return unwrapNamedObject<MLJob>(data, 'job')
}

export const triggerMLEvaluation = async (
  payload: MLTriggerEvaluationRequest
): Promise<MLAsyncResult> => {
  const { data } = await api.post('/ml/evaluations', payload)
  return unwrapApiData(data)
}
