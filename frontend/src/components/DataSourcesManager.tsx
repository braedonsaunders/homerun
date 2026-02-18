import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code2,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings2,
  Trash2,
  X,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import CodeEditor from './CodeEditor'
import DataSourceApiDocsFlyout from './DataSourceApiDocsFlyout'
import StrategyConfigForm from './StrategyConfigForm'
import { getWorldSourceStatus } from '../services/worldIntelligenceApi'
import {
  createUnifiedDataSource,
  deleteUnifiedDataSource,
  getUnifiedDataSourceRecords,
  getUnifiedDataSourceRuns,
  getUnifiedDataSourceTemplate,
  getUnifiedDataSources,
  reloadUnifiedDataSource,
  runUnifiedDataSource,
  UnifiedDataSource,
  updateUnifiedDataSource,
  validateUnifiedDataSource,
} from '../services/api'

function parseJsonObject(value: string): { value?: Record<string, unknown>; error?: string } {
  try {
    const parsed = JSON.parse(value || '{}')
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { error: 'must be a JSON object' }
    }
    return { value: parsed as Record<string, unknown> }
  } catch (error: any) {
    return { error: error?.message || 'invalid JSON' }
  }
}

function normalizeSlug(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, '_')
}

function inferClassName(sourceCode: string): string | null {
  const classPattern = /class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:/gm
  let match: RegExpExecArray | null = classPattern.exec(sourceCode)
  while (match) {
    const className = String(match[1] || '').trim()
    const bases = String(match[2] || '')
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    const isSourceClass = bases.some(
      (base) => base === 'BaseDataSource' || base.endsWith('.BaseDataSource')
    )
    if (className && isSourceClass) {
      return className
    }
    match = classPattern.exec(sourceCode)
  }
  return null
}

function errorMessage(error: unknown, fallback: string): string {
  const err = error as any
  const detail = err?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (detail && typeof detail === 'object') {
    if (Array.isArray(detail.errors) && detail.errors.length > 0) {
      return detail.errors.join('; ')
    }
    if (typeof detail.message === 'string') return detail.message
  }
  if (typeof err?.message === 'string' && err.message.trim()) return err.message
  return fallback
}

function classifySourceTone(details: any): 'ok' | 'degraded' | 'error' {
  if (details?.degraded) return 'degraded'
  const hardError = details?.ok === false
  if (!hardError) return 'ok'
  const rawError = String(details?.error || details?.last_error || '').toLowerCase()
  const degradedMarkers = [
    'credentials_missing',
    'missing_api_key',
    'missing_api_token',
    'disabled',
    'rate-limited',
    'rate limited',
    'http 429',
    "client error '429",
    'status code 429',
    "client error '403",
    'status code 403',
    '403 forbidden',
    'soft rate-limit',
    'soft rate-limited',
    'nodename nor servname provided',
    'name or service not known',
    'temporary failure in name resolution',
  ]
  if (degradedMarkers.some((marker) => rawError.includes(marker))) {
    return 'degraded'
  }
  return 'error'
}

function sourceToneClasses(tone: 'ok' | 'degraded' | 'error'): string {
  if (tone === 'ok') return 'text-emerald-400'
  if (tone === 'degraded') return 'text-yellow-400'
  return 'text-red-400'
}

function sourceToneLabel(tone: 'ok' | 'degraded' | 'error', count: number): string {
  if (tone === 'ok') return `ok (${count})`
  if (tone === 'degraded') return `degraded (${count})`
  return 'error'
}

function resolveLiveHealthSourceName(source: UnifiedDataSource): string | null {
  const slug = String(source.slug || '').trim().toLowerCase()
  if (!slug) return null
  const mapped = EVENT_SOURCE_HEALTH_KEY_BY_SLUG[slug]
  if (mapped) return mapped
  if (String(source.source_key || '').trim().toLowerCase() !== 'events') return null
  if (!slug.startsWith('events_')) return null
  const suffix = slug.slice('events_'.length).trim().toLowerCase()
  return suffix || null
}

const STATUS_COLORS: Record<string, string> = {
  loaded: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  error: 'bg-red-500/15 text-red-400 border-red-500/30',
  unloaded: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  draft: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
}

const SOURCE_LABELS: Record<string, string> = {
  custom: 'Custom',
  stories: 'Stories',
  events: 'Events',
  weather: 'Weather',
  crypto: 'Crypto',
  traders: 'Traders',
}

const SOURCE_KIND_LABELS: Record<string, string> = {
  python: 'Python',
  rss: 'RSS',
  gdelt: 'GDELT',
  events: 'Events',
  stories: 'Stories',
  bridge: 'Bridge',
}

const EVENT_SOURCE_HEALTH_KEY_BY_SLUG: Record<string, string> = {
  events_acled: 'acled',
  events_gdelt_tensions: 'gdelt_tensions',
  events_military: 'military',
  events_infrastructure: 'infrastructure',
  events_gdelt_news: 'gdelt_news',
  events_usgs: 'usgs',
  events_rss_news: 'rss_news',
  events_chokepoints: 'chokepoints',
  events_convergence: 'convergence',
  events_instability: 'instability',
  events_anomaly: 'anomaly',
}

const FALLBACK_TEMPLATE = [
  'from services.data_source_sdk import BaseDataSource',
  '',
  '',
  'class CustomDataSource(BaseDataSource):',
  '    name = "Custom Data Source"',
  '    description = "Fetches and normalizes external records"',
  '',
  '    default_config = {',
  '        "limit": 100,',
  '    }',
  '',
  '    async def fetch_async(self):',
  '        return []',
].join('\n')

export default function DataSourcesManager() {
  const queryClient = useQueryClient()

  const [showApiDocs, setShowApiDocs] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [showRawJson, setShowRawJson] = useState(false)

  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<string>('all')
  const [newPresetId, setNewPresetId] = useState<string>('')

  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null)
  const [draftToken, setDraftToken] = useState<string | null>(null)

  const [editorSlug, setEditorSlug] = useState('')
  const [editorName, setEditorName] = useState('')
  const [editorDescription, setEditorDescription] = useState('')
  const [editorSourceKey, setEditorSourceKey] = useState('custom')
  const [editorSourceKind, setEditorSourceKind] = useState('python')
  const [editorEnabled, setEditorEnabled] = useState(true)
  const [editorCode, setEditorCode] = useState('')
  const [editorConfigJson, setEditorConfigJson] = useState('{}')
  const [editorSchemaJson, setEditorSchemaJson] = useState('{}')
  const [editorError, setEditorError] = useState<string | null>(null)
  const [runLimit, setRunLimit] = useState(500)
  const [validation, setValidation] = useState<{
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
  } | null>(null)

  const sourcesQuery = useQuery({
    queryKey: ['unified-data-sources'],
    queryFn: () => getUnifiedDataSources(),
    staleTime: 15000,
    refetchInterval: 15000,
  })

  const templateQuery = useQuery({
    queryKey: ['unified-data-source-template'],
    queryFn: getUnifiedDataSourceTemplate,
    staleTime: Infinity,
  })

  const worldSourceStatusQuery = useQuery({
    queryKey: ['world-intelligence-sources'],
    queryFn: getWorldSourceStatus,
    staleTime: 10000,
    refetchInterval: 15000,
  })

  const catalog = sourcesQuery.data || []
  const templatePresets = useMemo(() => templateQuery.data?.presets || [], [templateQuery.data?.presets])

  const selectedPreset = useMemo(
    () => templatePresets.find((preset) => preset.id === newPresetId) || templatePresets[0] || null,
    [templatePresets, newPresetId]
  )

  const sourceKeys = useMemo(() => {
    const known = ['custom', 'stories', 'events', 'weather', 'crypto', 'traders']
    const dynamic = [...new Set(catalog.map((s) => String(s.source_key || '').toLowerCase()).filter(Boolean))]
    return [...new Set([...known, ...dynamic])]
  }, [catalog])

  const grouped = useMemo(() => {
    let rows = [...catalog]
    if (sourceFilter !== 'all') {
      rows = rows.filter((s) => s.source_key === sourceFilter)
    }
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase()
      rows = rows.filter(
        (s) =>
          (s.name || '').toLowerCase().includes(q)
          || (s.slug || '').toLowerCase().includes(q)
          || (s.source_key || '').toLowerCase().includes(q)
          || (s.source_kind || '').toLowerCase().includes(q)
          || (s.description || '').toLowerCase().includes(q)
      )
    }

    const groups: Record<string, UnifiedDataSource[]> = {}
    for (const source of rows) {
      const key = source.source_key || 'custom'
      if (!groups[key]) groups[key] = []
      groups[key].push(source)
    }
    return groups
  }, [catalog, searchQuery, sourceFilter])

  const flatFiltered = useMemo(() => Object.values(grouped).flat(), [grouped])

  const selectedSource = useMemo(
    () => catalog.find((source) => source.id === selectedSourceId) || null,
    [catalog, selectedSourceId]
  )

  const runsQuery = useQuery({
    queryKey: ['unified-data-source-runs', selectedSourceId],
    queryFn: () => getUnifiedDataSourceRuns(String(selectedSourceId), { limit: 10 }),
    enabled: !!selectedSourceId,
    refetchInterval: 15000,
  })

  const recordsPreviewQuery = useQuery({
    queryKey: ['unified-data-source-records-preview', selectedSourceId],
    queryFn: () => getUnifiedDataSourceRecords(String(selectedSourceId), { limit: 1, offset: 0 }),
    enabled: !!selectedSourceId,
    refetchInterval: 15000,
  })

  const inferredClassName = useMemo(() => inferClassName(editorCode), [editorCode])

  useEffect(() => {
    if (selectedSourceId) return
    if (draftToken) return
    if (catalog.length > 0) {
      setSelectedSourceId(catalog[0].id)
    }
  }, [catalog, selectedSourceId, draftToken])

  useEffect(() => {
    const defaultPresetId = String(templateQuery.data?.default_preset || '').trim()
    const hasCurrent = templatePresets.some((preset) => preset.id === newPresetId)
    if (!hasCurrent) {
      const fallbackId = templatePresets.some((preset) => preset.id === defaultPresetId)
        ? defaultPresetId
        : templatePresets[0]?.id || ''
      if (newPresetId !== fallbackId) {
        setNewPresetId(fallbackId)
      }
    }
  }, [templateQuery.data?.default_preset, templatePresets, newPresetId])

  useEffect(() => {
    if (!selectedSourceId) return
    const source = catalog.find((item) => item.id === selectedSourceId)
    if (!source) return

    setDraftToken(null)
    setEditorSlug(source.slug || '')
    setEditorName(source.name || '')
    setEditorDescription(source.description || '')
    setEditorSourceKey(source.source_key || 'custom')
    setEditorSourceKind(source.source_kind || 'python')
    setEditorEnabled(Boolean(source.enabled))
    setEditorCode(source.source_code || '')
    setEditorConfigJson(JSON.stringify(source.config || {}, null, 2))
    setEditorSchemaJson(JSON.stringify(source.config_schema || {}, null, 2))
    setEditorError(null)
    setValidation(null)
  }, [catalog, selectedSourceId])

  const refreshCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-data-sources'] })
    queryClient.invalidateQueries({ queryKey: ['unified-data-source-runs'] })
    queryClient.invalidateQueries({ queryKey: ['unified-data-source-records-preview'] })
  }

  const saveMutation = useMutation({
    mutationFn: async () => {
      const parsedConfig = parseJsonObject(editorConfigJson || '{}')
      if (!parsedConfig.value) {
        throw new Error(`Config JSON error: ${parsedConfig.error || 'invalid object'}`)
      }
      const parsedSchema = parseJsonObject(editorSchemaJson || '{}')
      if (!parsedSchema.value) {
        throw new Error(`Schema JSON error: ${parsedSchema.error || 'invalid object'}`)
      }

      const payload = {
        slug: normalizeSlug(editorSlug),
        source_key: String(editorSourceKey || '').trim().toLowerCase(),
        source_kind: String(editorSourceKind || '').trim().toLowerCase(),
        name: String(editorName || '').trim(),
        description: editorDescription.trim() || undefined,
        source_code: editorCode,
        config: parsedConfig.value,
        config_schema: parsedSchema.value,
        enabled: editorEnabled,
      }

      if (!payload.slug) throw new Error('Source slug is required')
      if (!payload.name) throw new Error('Name is required')

      const selected = catalog.find((source) => source.id === selectedSourceId)
      if (selected) {
        return updateUnifiedDataSource(selected.id, {
          ...payload,
          unlock_system: Boolean(selected.is_system),
        })
      }
      return createUnifiedDataSource(payload)
    },
    onSuccess: (source) => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedSourceId(source.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Failed to save data source'))
    },
  })

  const validateMutation = useMutation({
    mutationFn: async () => validateUnifiedDataSource(editorCode),
    onSuccess: (result) => {
      setValidation({
        valid: Boolean(result.valid),
        class_name: result.class_name || null,
        errors: result.errors || [],
        warnings: result.warnings || [],
      })
      setEditorError(null)
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Validation failed'))
    },
  })

  const reloadMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((source) => source.id === selectedSourceId)
      if (!selected) throw new Error('Select a source to reload')
      return reloadUnifiedDataSource(selected.id)
    },
    onSuccess: () => {
      setEditorError(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Reload failed'))
    },
  })

  const runMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((source) => source.id === selectedSourceId)
      if (!selected) throw new Error('Select a source to run')
      return runUnifiedDataSource(selected.id, { max_records: runLimit })
    },
    onSuccess: () => {
      setEditorError(null)
      queryClient.invalidateQueries({ queryKey: ['unified-data-source-runs', selectedSourceId] })
      queryClient.invalidateQueries({ queryKey: ['unified-data-source-records-preview', selectedSourceId] })
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Run failed'))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((source) => source.id === selectedSourceId)
      if (!selected) throw new Error('Select a source to delete')
      return deleteUnifiedDataSource(selected.id)
    },
    onSuccess: () => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedSourceId(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Delete failed'))
    },
  })

  const busy =
    saveMutation.isPending
    || validateMutation.isPending
    || reloadMutation.isPending
    || runMutation.isPending
    || deleteMutation.isPending

  const startNewDraft = (presetId?: string) => {
    const preset = templatePresets.find((item) => item.id === (presetId || newPresetId)) || selectedPreset
    if (!preset) {
      setEditorError('Template presets are unavailable from backend. Reload sources and retry.')
      return
    }
    const suffix = Date.now().toString().slice(-6)
    const slugPrefix = normalizeSlug(String(preset.slug_prefix || preset.id || 'source'))
    const nextSlug = normalizeSlug(`${slugPrefix}_${suffix}`)
    const nextConfig = preset.config || {}
    const nextSchema = preset.config_schema || { param_fields: [] }
    const nextLimit = Number((nextConfig as Record<string, unknown>).limit)

    setSelectedSourceId(null)
    setDraftToken(`draft_${Date.now()}`)
    setEditorSlug(nextSlug)
    setEditorSourceKey(String(preset.source_key || 'custom').trim().toLowerCase())
    setEditorSourceKind(String(preset.source_kind || 'python').trim().toLowerCase())
    setEditorName(preset.name || 'Custom Data Source')
    setEditorDescription(preset.description || '')
    setEditorEnabled(true)
    setEditorCode(preset.source_code || templateQuery.data?.template || FALLBACK_TEMPLATE)
    setEditorConfigJson(JSON.stringify(nextConfig, null, 2))
    setEditorSchemaJson(JSON.stringify(nextSchema, null, 2))
    setRunLimit(Number.isFinite(nextLimit) ? Math.max(1, Math.min(5000, Math.round(nextLimit))) : 500)
    setEditorError(null)
    setValidation(null)
  }

  const hasSelection = Boolean(selectedSource || draftToken)
  const displayStatus = selectedSource?.status || 'draft'
  const statusColor = STATUS_COLORS[displayStatus] || STATUS_COLORS.draft

  const configSchemaFields = useMemo(() => {
    try {
      const schema = JSON.parse(editorSchemaJson || '{}')
      return schema?.param_fields || []
    } catch {
      return []
    }
  }, [editorSchemaJson])

  const latestRun = runsQuery.data?.runs?.[0] || null
  const sourceHealthStatus = worldSourceStatusQuery.data?.sources || {}
  const sourceHealthError = worldSourceStatusQuery.data?.errors?.[0] || null

  return (
    <div className="h-full min-h-0 flex gap-3">
      <div className="w-[300px] shrink-0 min-h-0 flex flex-col rounded-lg border border-border/70 bg-card/50">
        <div className="shrink-0 p-3 space-y-3 border-b border-border/50">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              {selectedPreset ? (
                <Select value={selectedPreset.id} onValueChange={setNewPresetId}>
                  <SelectTrigger className="h-7 text-[11px] flex-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {templatePresets.map((preset) => (
                      <SelectItem key={preset.id} value={preset.id}>
                        {preset.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <div className="h-7 flex-1 rounded-md border border-red-500/30 bg-red-500/10 px-2 text-[10px] text-red-300 flex items-center">
                  No backend presets available
                </div>
              )}
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-7 px-2 text-[11px]"
                onClick={() => refreshCatalog()}
                disabled={busy}
                title="Refresh sources"
              >
                <RefreshCw className={cn('w-3 h-3', sourcesQuery.isFetching && 'animate-spin')} />
              </Button>
            </div>
            <Button
              type="button"
              size="sm"
              className="h-7 gap-1.5 px-2.5 text-[11px] w-full"
              onClick={() => startNewDraft(selectedPreset?.id)}
              disabled={busy || !selectedPreset}
            >
              <Plus className="w-3 h-3" />
              {selectedPreset ? `New ${selectedPreset.name}` : 'New Source'}
            </Button>
          </div>

          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="h-7 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources ({catalog.length})</SelectItem>
              {sourceKeys.map((key) => (
                <SelectItem key={key} value={key}>
                  {SOURCE_LABELS[key] || key} ({catalog.filter((item) => item.source_key === key).length})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
            <Input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search data sources..."
              className="h-7 pl-8 pr-7 text-xs"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 hover:bg-muted/50 rounded"
              >
                <X className="w-3 h-3 text-muted-foreground" />
              </button>
            )}
          </div>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="p-1.5 space-y-1">
            {flatFiltered.length === 0 ? (
              <p className="px-3 py-6 text-xs text-muted-foreground text-center">No data sources found.</p>
            ) : (
              Object.entries(grouped).map(([groupKey, sources]) => (
                <div key={groupKey}>
                  {Object.keys(grouped).length > 1 && (
                    <div className="px-2.5 pt-2.5 pb-1 flex items-center gap-1.5">
                      <div className="flex-1 h-px bg-border/50" />
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-medium shrink-0">
                        {SOURCE_LABELS[groupKey] || groupKey} ({sources.length})
                      </p>
                      <div className="flex-1 h-px bg-border/50" />
                    </div>
                  )}
                  {sources.map((source) => {
                    const active = selectedSourceId === source.id
                    const badgeColor = STATUS_COLORS[source.status] || STATUS_COLORS.draft
                    const healthSourceName = resolveLiveHealthSourceName(source)
                    const healthDetails = healthSourceName ? sourceHealthStatus[healthSourceName] : null
                    const healthTone = healthDetails ? classifySourceTone(healthDetails) : null
                    const healthCount = Number(healthDetails?.count ?? 0)
                    return (
                      <button
                        key={source.id}
                        type="button"
                        onClick={() => {
                          setDraftToken(null)
                          setSelectedSourceId(source.id)
                        }}
                        className={cn(
                          'w-full rounded-md px-2.5 py-2 text-left transition-all duration-150',
                          active ? 'bg-cyan-500/10 ring-1 ring-cyan-500/30' : 'hover:bg-muted/50'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-medium truncate" title={source.name}>
                            {source.name}
                          </p>
                          <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0 h-4 border shrink-0', badgeColor)}>
                            {source.status}
                          </Badge>
                        </div>
                        <p className="text-[10px] font-mono text-muted-foreground mt-1 truncate">
                          {source.slug}
                        </p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">
                          {SOURCE_KIND_LABELS[source.source_kind] || source.source_kind}
                        </p>
                        {healthDetails && healthTone && (
                          <div className="mt-1 flex items-center justify-between rounded bg-background/50 px-1.5 py-1">
                            <span className="text-[10px] font-mono text-muted-foreground">
                              {healthSourceName}
                            </span>
                            <span className={cn('text-[10px] font-mono', sourceToneClasses(healthTone))}>
                              {sourceToneLabel(healthTone, healthCount)}
                            </span>
                          </div>
                        )}
                      </button>
                    )
                  })}
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        <div className="shrink-0 px-3 py-2 border-t border-border/50 text-[10px] text-muted-foreground space-y-1">
          {sourceHealthError && (
            <div className="text-red-400 truncate" title={sourceHealthError}>{sourceHealthError}</div>
          )}
          <div className="flex justify-between">
            <span>{flatFiltered.length} sources</span>
            <span>{catalog.filter((source) => source.enabled).length} enabled</span>
          </div>
        </div>
      </div>

      <div className="flex-1 min-w-0 min-h-0 flex flex-col rounded-lg border border-border/70">
        {!hasSelection ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center space-y-2">
              <Code2 className="w-8 h-8 mx-auto opacity-40" />
              <p className="text-sm">Select a source or create a new one</p>
            </div>
          </div>
        ) : (
          <>
            <div className="shrink-0 px-4 py-2.5 border-b border-border/50 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="flex items-center gap-2 min-w-0">
                  <Code2 className="w-4 h-4 text-cyan-400 shrink-0" />
                  <span className="text-sm font-medium truncate">{editorName || 'Untitled Source'}</span>
                </div>
                <Badge variant="outline" className={cn('text-[10px] shrink-0 border', statusColor)}>
                  {displayStatus}
                </Badge>
                {selectedSource?.is_system && (
                  <Badge variant="secondary" className="text-[10px] shrink-0">System</Badge>
                )}
                {selectedSource && (
                  <span className="text-[10px] font-mono text-muted-foreground shrink-0">v{selectedSource.version}</span>
                )}
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <div className="flex items-center gap-2 mr-2 pr-2 border-r border-border/50">
                  <span className="text-[10px] text-muted-foreground">Enabled</span>
                  <Switch checked={editorEnabled} onCheckedChange={setEditorEnabled} className="scale-75" />
                </div>

                <div className="flex items-center gap-1 rounded-md border border-border/50 px-1.5 h-7">
                  <Label className="text-[10px] text-muted-foreground">Run</Label>
                  <Input
                    type="number"
                    min={1}
                    max={5000}
                    value={runLimit}
                    onChange={(event) => setRunLimit(Math.max(1, Math.min(5000, Number(event.target.value) || 1)))}
                    className="h-5 w-[72px] text-[10px] px-1.5 border-0 bg-transparent"
                  />
                </div>

                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => setShowApiDocs(true)}
                >
                  <BookOpen className="w-3 h-3" />
                  API Docs
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => validateMutation.mutate()}
                  disabled={busy || !editorCode.trim()}
                >
                  {validateMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                  Validate
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => reloadMutation.mutate()}
                  disabled={busy || !selectedSource}
                >
                  {reloadMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                  Reload
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => runMutation.mutate()}
                  disabled={busy || !selectedSource}
                >
                  {runMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
                  Run
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px] bg-cyan-600 hover:bg-cyan-500 text-white"
                  onClick={() => saveMutation.mutate()}
                  disabled={busy || !editorCode.trim() || !editorName.trim() || !editorSlug.trim()}
                >
                  {saveMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                  Save
                </Button>
                {selectedSource && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 px-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    onClick={() => {
                      const message = selectedSource.is_system
                        ? `Delete system source "${selectedSource.name}"? It will be tombstoned and will not auto-reseed.`
                        : `Delete "${selectedSource.name}"? This cannot be undone.`
                      if (window.confirm(message)) deleteMutation.mutate()
                    }}
                    disabled={busy}
                    title="Delete source"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                )}
              </div>
            </div>

            {validation && (
              <div
                className={cn(
                  'shrink-0 px-4 py-2 text-xs flex items-start gap-2 border-b',
                  validation.valid
                    ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
                    : 'bg-red-500/5 border-red-500/20 text-red-400'
                )}
              >
                {validation.valid ? <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" /> : <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />}
                <div className="min-w-0 flex-1">
                  <p className="font-medium">
                    {validation.valid ? 'Validation passed' : 'Validation failed'}
                    {validation.class_name && <span className="font-mono ml-2 opacity-70">{validation.class_name}</span>}
                  </p>
                  {validation.errors.map((error, index) => (
                    <p key={`validation-error-${index}`} className="font-mono text-[11px] mt-0.5">{error}</p>
                  ))}
                  {validation.warnings.map((warning, index) => (
                    <p key={`validation-warning-${index}`} className="font-mono text-[11px] mt-0.5 text-amber-400">{warning}</p>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setValidation(null)}
                  className="shrink-0 p-0.5 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {editorError && (
              <div className="shrink-0 px-4 py-2 text-xs flex items-start gap-2 bg-red-500/5 border-b border-red-500/20 text-red-400">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <p className="min-w-0 flex-1">{editorError}</p>
                <button
                  type="button"
                  onClick={() => setEditorError(null)}
                  className="shrink-0 p-0.5 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {(selectedSource || latestRun) && (
              <div className="shrink-0 px-4 py-2 border-b border-border/40 bg-card/30 text-[11px] flex items-center gap-4 text-muted-foreground">
                <span>
                  Last run: <span className="font-data text-foreground">{latestRun?.status || 'never'}</span>
                </span>
                <span>
                  Records: <span className="font-data text-foreground">{recordsPreviewQuery.data?.total ?? 0}</span>
                </span>
                {latestRun && (
                  <>
                    <span>
                      Upserts: <span className="font-data text-foreground">{latestRun.upserted_count}</span>
                    </span>
                    <span>
                      Duration: <span className="font-data text-foreground">{latestRun.duration_ms ?? 0}ms</span>
                    </span>
                  </>
                )}
              </div>
            )}

            <div className="flex-1 min-h-0 flex flex-col">
              <div className="shrink-0 border-b border-border/50">
                <button
                  type="button"
                  onClick={() => setShowSettings((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showSettings ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  <Settings2 className="w-3 h-3" />
                  <span>Source Settings</span>
                  <span className="ml-auto font-mono text-[10px] opacity-60">{editorSlug || 'no-slug'}</span>
                </button>

                {showSettings && (
                  <div className="px-4 pb-3 space-y-3 animate-in fade-in duration-200">
                    <div className="grid gap-3 grid-cols-2 xl:grid-cols-5">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source Slug</Label>
                        <Input
                          value={editorSlug}
                          onChange={(event) => setEditorSlug(normalizeSlug(event.target.value))}
                          className="mt-1 h-8 text-xs font-mono"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Name</Label>
                        <Input
                          value={editorName}
                          onChange={(event) => setEditorName(event.target.value)}
                          className="mt-1 h-8 text-xs"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source Key</Label>
                        <Select value={editorSourceKey} onValueChange={setEditorSourceKey}>
                          <SelectTrigger className="mt-1 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {sourceKeys.map((key) => (
                              <SelectItem key={key} value={key}>
                                {SOURCE_LABELS[key] || key}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Kind</Label>
                        <Select value={editorSourceKind} onValueChange={setEditorSourceKind}>
                          <SelectTrigger className="mt-1 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(SOURCE_KIND_LABELS).map(([kind, label]) => (
                              <SelectItem key={kind} value={kind}>{label}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Class Name</Label>
                        <Input
                          value={validation?.class_name || inferredClassName || selectedSource?.class_name || 'auto-detected'}
                          className="mt-1 h-8 text-xs font-mono"
                          disabled
                        />
                      </div>
                    </div>
                    <div>
                      <Label className="text-[11px] text-muted-foreground">Description</Label>
                      <Input
                        value={editorDescription}
                        onChange={(event) => setEditorDescription(event.target.value)}
                        className="mt-1 h-8 text-xs"
                        placeholder="Describe what this source ingests and transforms..."
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="flex-1 min-h-0 flex flex-col">
                <div className="px-4 py-2 flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-xs font-medium">Source Code</span>
                    <span className="text-[10px] text-muted-foreground font-mono">Python</span>
                  </div>
                  {inferredClassName && (
                    <span className="text-[10px] font-mono text-muted-foreground">class {inferredClassName}</span>
                  )}
                </div>
                <div className="flex-1 min-h-0 px-3 pb-2">
                  <CodeEditor
                    value={editorCode}
                    onChange={setEditorCode}
                    language="python"
                    className="h-full"
                    minHeight="100%"
                    placeholder="Write your data source source code here..."
                  />
                </div>
              </div>

              <div className="shrink-0 border-t border-border/50">
                <button
                  type="button"
                  onClick={() => setShowConfig((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfig ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  <Settings2 className="w-3 h-3" />
                  <span>Runtime Config</span>
                  {configSchemaFields.length > 0 && (
                    <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 ml-1">
                      {configSchemaFields.length} fields
                    </Badge>
                  )}
                </button>
                {showConfig && (
                  <div className="px-3 pb-3 animate-in fade-in duration-200 space-y-3">
                    {configSchemaFields.length > 0 && !showRawJson && (
                      <>
                        <StrategyConfigForm
                          schema={{ param_fields: configSchemaFields }}
                          values={(() => {
                            try {
                              return JSON.parse(editorConfigJson || '{}')
                            } catch {
                              return {}
                            }
                          })()}
                          onChange={(values) => setEditorConfigJson(JSON.stringify(values, null, 2))}
                        />
                        <button
                          type="button"
                          onClick={() => setShowRawJson(true)}
                          className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                        >
                          Show Raw JSON
                        </button>
                      </>
                    )}

                    {(configSchemaFields.length === 0 || showRawJson) && (
                      <>
                        <div className="grid gap-3 grid-cols-1 lg:grid-cols-2">
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-[11px] font-medium text-muted-foreground">Config</span>
                              <span className="text-[10px] text-muted-foreground font-mono">JSON</span>
                            </div>
                            <CodeEditor
                              value={editorConfigJson}
                              onChange={setEditorConfigJson}
                              language="json"
                              minHeight="140px"
                              placeholder="{}"
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-[11px] font-medium text-muted-foreground">Config Schema</span>
                              <span className="text-[10px] text-muted-foreground font-mono">JSON</span>
                            </div>
                            <CodeEditor
                              value={editorSchemaJson}
                              onChange={setEditorSchemaJson}
                              language="json"
                              minHeight="140px"
                              placeholder='{"param_fields": []}'
                            />
                          </div>
                        </div>
                        {configSchemaFields.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setShowRawJson(false)}
                            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                          >
                            Show Config Form
                          </button>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <DataSourceApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} />
    </div>
  )
}
