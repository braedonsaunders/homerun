import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code2,
  Eye,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import CodeEditor from './CodeEditor'
import DataSourceApiDocsFlyout from './DataSourceApiDocsFlyout'
import DataSourcePreviewFlyout from './DataSourcePreviewFlyout'
import RestApiSourceForm from './RestApiSourceForm'
import RssSourceForm from './RssSourceForm'
import TwitterSourceForm from './TwitterSourceForm'
import StrategyConfigForm from './StrategyConfigForm'
import { getWorldSourceStatus } from '../services/eventsApi'
import {
  createUnifiedDataSource,
  deleteUnifiedDataSource,
  getUnifiedDataSource,
  getUnifiedDataSourceRecords,
  getUnifiedDataSourceRuns,
  getUnifiedDataSourceTemplate,
  getUnifiedDataSources,
  generateAIDataSourceDraft,
  reloadUnifiedDataSource,
  runUnifiedDataSource,
  AIDataSourceDraftGenerationResponse,
  UnifiedDataSource,
  UnifiedDataSourceTemplatePreset,
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

function pythonStringLiteral(value: string): string {
  return `"${String(value || '')
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\r/g, '\\r')
    .replace(/\n/g, '\\n')}"`
}

function applySourceMetadataToCode(
  sourceCode: string,
  {
    sourceName,
    sourceDescription,
  }: {
    sourceName: string
    sourceDescription: string
  }
): string {
  let out = String(sourceCode || '')
  out = out.replace(/^\s*name\s*=\s*".*"$/m, `    name = ${pythonStringLiteral(sourceName)}`)
  out = out.replace(/^\s*description\s*=\s*".*"$/m, `    description = ${pythonStringLiteral(sourceDescription)}`)
  return out
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

function normalizeRetentionPolicy(input: unknown): { max_records?: number; max_age_days?: number } {
  const value = input && typeof input === 'object' ? input as Record<string, unknown> : {}
  const next: { max_records?: number; max_age_days?: number } = {}

  const maxRecords = Number(value.max_records)
  if (Number.isFinite(maxRecords)) {
    const parsed = Math.max(1, Math.min(250000, Math.round(maxRecords)))
    next.max_records = parsed
  }

  const maxAgeDays = Number(value.max_age_days)
  if (Number.isFinite(maxAgeDays)) {
    const parsed = Math.max(1, Math.min(3650, Math.round(maxAgeDays)))
    next.max_age_days = parsed
  }

  return next
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
  rss: 'RSS Feed',
  rest_api: 'REST API',
  twitter: 'Twitter/X',
}

interface NewDataSourceTemplate {
  id: string
  slug_prefix: string
  name: string
  description: string
  source_key: string
  source_kind: string
  source_code: string
  retention?: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  is_system_seed?: boolean
}

const EVENT_SOURCE_HEALTH_KEY_BY_SLUG: Record<string, string> = {
  events_acled: 'acled',
  events_gdelt_tensions: 'gdelt_tensions',
  events_military: 'military',
  events_infrastructure: 'infrastructure',
  events_gdelt_news: 'gdelt_news',
  events_usgs: 'usgs',
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

const DEFAULT_NEW_SOURCE_PRESET_ID = 'python_custom'
const DEFAULT_REST_API_PRESET_ID = 'rest_api_custom'
const CREATE_SOURCE_KIND_ORDER = ['python', 'rss', 'rest_api', 'twitter']

export default function DataSourcesManager({
  onOpenCopilot,
}: {
  onOpenCopilot?: (options?: {
    contextType?: string
    contextId?: string
    label?: string
    prompt?: string
    autoSend?: boolean
  }) => void
}) {
  const queryClient = useQueryClient()
  const loadedEditorSourceKeyRef = useRef<string | null>(null)

  const [showApiDocs, setShowApiDocs] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [showRawJson, setShowRawJson] = useState(false)
  const [showAdvancedCode, setShowAdvancedCode] = useState(false)

  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  const [newSourceName, setNewSourceName] = useState('Custom Data Source')
  const [newSourceSlug, setNewSourceSlug] = useState(() => `custom_source_${Date.now().toString().slice(-6)}`)
  const [newSourceDescription, setNewSourceDescription] = useState('')
  const [newSourceSourceKey, setNewSourceSourceKey] = useState('custom')
  const [newSourceTemplateKind, setNewSourceTemplateKind] = useState('python')
  const [newSourceTemplateKey, setNewSourceTemplateKey] = useState(DEFAULT_NEW_SOURCE_PRESET_ID)
  const [newSourceSlugDirty, setNewSourceSlugDirty] = useState(false)
  const [newSourceError, setNewSourceError] = useState<string | null>(null)
  const [newSourceAiPrompt, setNewSourceAiPrompt] = useState('')
  const [newSourceAiDraft, setNewSourceAiDraft] = useState<AIDataSourceDraftGenerationResponse | null>(null)
  const [newSourceUseAiDraft, setNewSourceUseAiDraft] = useState(false)

  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null)
  const [draftToken, setDraftToken] = useState<string | null>(null)

  const [editorSlug, setEditorSlug] = useState('')
  const [editorName, setEditorName] = useState('')
  const [editorDescription, setEditorDescription] = useState('')
  const [editorSourceKey, setEditorSourceKey] = useState('custom')
  const [editorSourceKind, setEditorSourceKind] = useState('python')
  const [editorEnabled, setEditorEnabled] = useState(true)
  const [editorCode, setEditorCode] = useState('')
  const [editorRetention, setEditorRetention] = useState<{ max_records?: number; max_age_days?: number }>({})
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
    queryFn: () => getUnifiedDataSources({ include_code: false }),
    staleTime: 15000,
    refetchInterval: 15000,
  })

  const templateQuery = useQuery({
    queryKey: ['unified-data-source-template'],
    queryFn: getUnifiedDataSourceTemplate,
    staleTime: Infinity,
    enabled: showCreateModal,
  })

  const worldSourceStatusQuery = useQuery({
    queryKey: ['events-sources'],
    queryFn: getWorldSourceStatus,
    staleTime: 10000,
    refetchInterval: 15000,
  })

  const catalog = useMemo(() => {
    const payload = sourcesQuery.data as UnifiedDataSource[] | { items?: UnifiedDataSource[] } | undefined
    if (Array.isArray(payload)) return payload
    if (payload && Array.isArray(payload.items)) return payload.items
    return []
  }, [sourcesQuery.data])
  const templatePresets = useMemo<NewDataSourceTemplate[]>(() => {
    const presets = (templateQuery.data?.presets || []).map((preset) => {
      const mapped = preset as UnifiedDataSourceTemplatePreset
      return {
        id: String(mapped.id || '').trim(),
        slug_prefix: String(mapped.slug_prefix || mapped.id || '').trim(),
        name: String(mapped.name || '').trim() || 'Custom Data Source',
        description: String(mapped.description || '').trim() || 'Custom data source template',
        source_key: String(mapped.source_key || 'custom').trim().toLowerCase(),
        source_kind: String(mapped.source_kind || 'python').trim().toLowerCase(),
        source_code: String(mapped.source_code || templateQuery.data?.template || FALLBACK_TEMPLATE),
        retention: normalizeRetentionPolicy(mapped.retention || {}),
        config: mapped.config || {},
        config_schema: mapped.config_schema || { param_fields: [] },
        is_system_seed: Boolean(mapped.is_system_seed),
      }
    })

    if (presets.length > 0) {
      const hasPythonTemplate = presets.some((preset) => preset.source_kind === 'python')
      if (!hasPythonTemplate) {
        presets.push({
          id: DEFAULT_NEW_SOURCE_PRESET_ID,
          slug_prefix: 'custom_source',
          name: 'Custom Python Source',
          description: 'Full custom Python source with BaseDataSource',
          source_key: 'custom',
          source_kind: 'python',
          source_code: templateQuery.data?.template || FALLBACK_TEMPLATE,
          retention: {},
          config: {},
          config_schema: { param_fields: [] },
          is_system_seed: false,
        })
      }
    } else {
      presets.push({
        id: DEFAULT_NEW_SOURCE_PRESET_ID,
        slug_prefix: 'custom_source',
        name: 'Custom Python Source',
        description: 'Full custom Python source with BaseDataSource',
        source_key: 'custom',
        source_kind: 'python',
        source_code: templateQuery.data?.template || FALLBACK_TEMPLATE,
        retention: {},
        config: {},
        config_schema: { param_fields: [] },
        is_system_seed: false,
      })
    }

    const hasRestApiTemplate = presets.some((preset) => preset.source_kind === 'rest_api')
    if (!hasRestApiTemplate) {
      presets.push({
        id: DEFAULT_REST_API_PRESET_ID,
        slug_prefix: 'custom_rest_api',
        name: 'Custom REST API Source',
        description: 'Simple REST API connector source',
        source_key: 'custom',
        source_kind: 'rest_api',
        source_code: templateQuery.data?.template || FALLBACK_TEMPLATE,
        retention: {},
        config: {},
        config_schema: { param_fields: [] },
        is_system_seed: false,
      })
    }

    const hasTwitterTemplate = presets.some((preset) => preset.source_kind === 'twitter')
    if (!hasTwitterTemplate) {
      presets.push({
        id: 'twitter_custom',
        slug_prefix: 'custom_twitter',
        name: 'Custom Twitter/X Source',
        description: 'Ingest tweets from Twitter/X handles or keyword searches',
        source_key: 'custom',
        source_kind: 'twitter',
        source_code: templateQuery.data?.template || FALLBACK_TEMPLATE,
        retention: {},
        config: {},
        config_schema: { param_fields: [] },
        is_system_seed: false,
      })
    }

    return presets
  }, [templateQuery.data?.presets, templateQuery.data?.template])

  const templateKinds = useMemo(() => {
    const seen = new Set<string>()
    const present = new Set(
      templatePresets.map((preset) => String(preset.source_kind || '').trim().toLowerCase())
    )
    const ordered = [...CREATE_SOURCE_KIND_ORDER]
    for (const kind of present) {
      if (!ordered.includes(kind)) ordered.push(kind)
    }
    const unique: string[] = []
    for (const kind of ordered) {
      const normalized = String(kind || '').trim().toLowerCase()
      if (!normalized || seen.has(normalized)) continue
      seen.add(normalized)
      unique.push(normalized)
    }
    return unique
  }, [templatePresets])

  const templatesForCreateKind = useMemo(() => {
    const normalizedKind = String(newSourceTemplateKind || '').trim().toLowerCase()
    return templatePresets.filter((preset) => String(preset.source_kind || '').trim().toLowerCase() === normalizedKind)
  }, [templatePresets, newSourceTemplateKind])

  const selectedCreateTemplate = useMemo(() => {
    return (
      templatePresets.find((preset) => preset.id === newSourceTemplateKey)
      || templatesForCreateKind[0]
      || templatePresets[0]
      || null
    )
  }, [newSourceTemplateKey, templatesForCreateKind, templatePresets])

  const newSourceTemplatePreviewCode = useMemo(
    () => {
      if (newSourceUseAiDraft && newSourceAiDraft?.source_code) {
        const title = String(newSourceName || '').trim() || 'Custom Data Source'
        const description = String(newSourceDescription || '').trim() || `${title} source`
        return applySourceMetadataToCode(String(newSourceAiDraft.source_code || ''), {
          sourceName: title,
          sourceDescription: description,
        })
      }
      return selectedCreateTemplate?.source_code || templateQuery.data?.template || FALLBACK_TEMPLATE
    },
    [
      selectedCreateTemplate?.source_code,
      templateQuery.data?.template,
      newSourceUseAiDraft,
      newSourceAiDraft?.source_code,
      newSourceName,
      newSourceDescription,
    ]
  )

  const sourceKeys = useMemo(() => {
    const known = ['custom', 'stories', 'events']
    const dynamic = [
      ...new Set(
        [
          ...catalog.map((s) => String(s.source_key || '').toLowerCase()),
          String(newSourceSourceKey || '').toLowerCase(),
          String(editorSourceKey || '').toLowerCase(),
        ].filter(Boolean)
      ),
    ]
    return [...new Set([...known, ...dynamic])]
  }, [catalog, newSourceSourceKey, editorSourceKey])

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

  const selectedSourceQuery = useQuery({
    queryKey: ['unified-data-source', selectedSourceId],
    queryFn: () => getUnifiedDataSource(String(selectedSourceId)),
    enabled: !!selectedSourceId,
    staleTime: 10000,
  })

  const selectedSourceResolved = useMemo(() => {
    if (selectedSourceQuery.data && selectedSourceQuery.data.id === selectedSourceId) {
      return selectedSourceQuery.data
    }
    return selectedSource
  }, [selectedSourceId, selectedSourceQuery.data, selectedSource])

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

  const templateDefaultsAppliedRef = useRef(false)
  useEffect(() => {
    if (templatePresets.length === 0 || templateDefaultsAppliedRef.current) return
    templateDefaultsAppliedRef.current = true
    const defaultKind = String(templateQuery.data?.default_preset || '').trim()
    const defaultTemplate = templatePresets.find((preset) => preset.id === defaultKind)
      || templatePresets[0]
    const normalizedKind = String(defaultTemplate.source_kind || 'python').trim().toLowerCase() || 'python'
    const kindTemplates = templatePresets.filter((preset) => String(preset.source_kind || '').trim().toLowerCase() === normalizedKind)
    setNewSourceTemplateKind(normalizedKind)
    setNewSourceTemplateKey(kindTemplates[0]?.id || templatePresets[0]?.id || '')
  }, [templatePresets, templateQuery.data?.default_preset])

  useEffect(() => {
    if (!selectedSourceId) {
      loadedEditorSourceKeyRef.current = null
      return
    }
    const source = selectedSourceResolved
    if (!source) return
    const sourceVersion = Number(source.version || 0)
    const hasDetailPayload = Boolean(selectedSourceQuery.data && selectedSourceQuery.data.id === source.id)
    const nextLoadedKey = `${source.id}:${sourceVersion}:${hasDetailPayload ? 'detail' : 'summary'}`
    if (loadedEditorSourceKeyRef.current === nextLoadedKey) return
    loadedEditorSourceKeyRef.current = nextLoadedKey

    setDraftToken(null)
    setEditorSlug(source.slug || '')
    setEditorName(source.name || '')
    setEditorDescription(source.description || '')
    setEditorSourceKey(source.source_key || 'custom')
    setEditorSourceKind(source.source_kind || 'python')
    setEditorEnabled(Boolean(source.enabled))
    setEditorCode(source.source_code || '')
    setEditorRetention(normalizeRetentionPolicy(source.retention || {}))
    setEditorConfigJson(JSON.stringify(source.config || {}, null, 2))
    setEditorSchemaJson(JSON.stringify(source.config_schema || {}, null, 2))
    setEditorError(null)
    setValidation(null)
    setShowAdvancedCode(false)
  }, [selectedSourceId, selectedSourceResolved, selectedSourceQuery.data])

  const refreshCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-data-sources'] })
    queryClient.invalidateQueries({ queryKey: ['unified-data-source'] })
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
        retention: normalizeRetentionPolicy(editorRetention),
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
      queryClient.setQueryData(['unified-data-source', source.id], source)
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

  const generateAiDraftMutation = useMutation({
    mutationFn: async () => {
      const prompt = String(newSourceAiPrompt || '').trim()
      if (!prompt) throw new Error('Describe the data source you want to generate.')
      return generateAIDataSourceDraft({
        description: prompt,
        source_key: String(newSourceSourceKey || '').trim().toLowerCase() || 'custom',
        source_kind: String(newSourceTemplateKind || '').trim().toLowerCase() || 'python',
      })
    },
    onSuccess: (draft) => {
      setNewSourceAiDraft(draft)
      setNewSourceUseAiDraft(true)
      const requestedKind = String(draft.source_kind || 'python').trim().toLowerCase() || 'python'
      const normalizedKind = templateKinds.includes(requestedKind) ? requestedKind : 'python'
      const normalizedSourceKey = String(draft.source_key || 'custom').trim().toLowerCase() || 'custom'
      const normalizedSlug = normalizeSlug(String(draft.slug || '').trim() || 'custom_source')
      setNewSourceName(String(draft.name || '').trim() || 'Custom Data Source')
      setNewSourceSlug(normalizedSlug)
      setNewSourceSlugDirty(true)
      setNewSourceDescription(String(draft.description || '').trim())
      setNewSourceTemplateKind(normalizedKind)
      setNewSourceSourceKey(normalizedSourceKey)
      const preferredTemplate = templatePresets.find((preset) => String(preset.source_kind || '').trim().toLowerCase() === normalizedKind)
      if (preferredTemplate) {
        setNewSourceTemplateKey(preferredTemplate.id)
      }
      setNewSourceError(null)
    },
    onError: (error: unknown) => {
      setNewSourceError(errorMessage(error, 'AI generation failed'))
    },
  })

  const busy =
    generateAiDraftMutation.isPending
    || templateQuery.isFetching
    saveMutation.isPending
    || validateMutation.isPending
    || reloadMutation.isPending
    || runMutation.isPending
    || deleteMutation.isPending

  const openCreateModal = () => {
    const defaultPresetId = String(templateQuery.data?.default_preset || '').trim()
    const fallback = templatePresets.find((item) => item.id === defaultPresetId) || selectedCreateTemplate || templatePresets[0]
    const nonce = Date.now().toString().slice(-6)
    const slugPrefix = normalizeSlug(fallback?.slug_prefix || fallback?.id || 'source')
    const normalizedKind = String(fallback?.source_kind || 'python').trim().toLowerCase() || 'python'
    const normalizedSourceKey = String(fallback?.source_key || 'custom').trim().toLowerCase() || 'custom'

    templateDefaultsAppliedRef.current = true
    setShowCreateModal(true)
    setNewSourceName(fallback?.name || 'Custom Data Source')
    setNewSourceDescription(fallback?.description || '')
    setNewSourceSourceKey(normalizedSourceKey)
    setNewSourceTemplateKind(normalizedKind)
    setNewSourceTemplateKey(fallback?.id || DEFAULT_NEW_SOURCE_PRESET_ID)
    setNewSourceSlug(`${slugPrefix}_${nonce}`)
    setNewSourceSlugDirty(false)
    setNewSourceError(null)
    setNewSourceAiPrompt('')
    setNewSourceAiDraft(null)
    setNewSourceUseAiDraft(false)
  }

  const createDraftFromModal = () => {
    const trimmedName = String(newSourceName || '').trim()
    const normalizedSlug = normalizeSlug(newSourceSlug)
    const normalizedSourceKind = String(newSourceTemplateKind || '').trim().toLowerCase()
    const normalizedSourceKey = String(newSourceSourceKey || '').trim().toLowerCase() || 'custom'
    const selectedTemplate = selectedCreateTemplate
    const useAiDraft = Boolean(newSourceUseAiDraft && newSourceAiDraft)

    if (!trimmedName) {
      setNewSourceError('Source name is required.')
      return
    }
    if (!normalizedSlug) {
      setNewSourceError('Source slug is required.')
      return
    }
    if (!selectedTemplate) {
      setNewSourceError('Select a template before creating the draft.')
      return
    }

    const aiDraft = newSourceAiDraft
    const templateConfig = useAiDraft && aiDraft && aiDraft.config && typeof aiDraft.config === 'object' && !Array.isArray(aiDraft.config)
      ? (aiDraft.config as Record<string, unknown>)
      : selectedTemplate.config || {}
    const templateSchema = useAiDraft && aiDraft && aiDraft.config_schema && typeof aiDraft.config_schema === 'object' && !Array.isArray(aiDraft.config_schema)
      ? (aiDraft.config_schema as Record<string, unknown>)
      : selectedTemplate.config_schema || { param_fields: [] }
    const templateRetention = useAiDraft && aiDraft
      ? normalizeRetentionPolicy(aiDraft.retention || {})
      : normalizeRetentionPolicy(selectedTemplate.retention || {})
    const nextLimit = Number((templateConfig as Record<string, unknown>).limit)
    const baseSourceCode = useAiDraft
      ? String(aiDraft?.source_code || '').trim()
      : String(selectedTemplate.source_code || templateQuery.data?.template || FALLBACK_TEMPLATE)
    const renderedSourceCode = applySourceMetadataToCode(baseSourceCode, {
      sourceName: trimmedName,
      sourceDescription: String(newSourceDescription || '').trim() || `${trimmedName} source`,
    })

    setSelectedSourceId(null)
    loadedEditorSourceKeyRef.current = null
    setDraftToken(`draft_${Date.now()}`)
    setEditorSlug(normalizedSlug)
    setEditorSourceKind(normalizedSourceKind || 'python')
    setEditorSourceKey(normalizedSourceKey)
    setEditorName(trimmedName)
    setEditorDescription(String(newSourceDescription || '').trim())
    setEditorEnabled(true)
    setEditorCode(renderedSourceCode || FALLBACK_TEMPLATE)
    setEditorRetention(templateRetention)
    setEditorConfigJson(JSON.stringify(templateConfig, null, 2))
    setEditorSchemaJson(JSON.stringify(templateSchema, null, 2))
    setRunLimit(Number.isFinite(nextLimit) ? Math.max(1, Math.min(5000, Math.round(nextLimit))) : 500)
    setEditorError(null)
    setValidation(null)
    setShowAdvancedCode(false)
    setShowSettings(true)
    setShowCreateModal(false)
    setNewSourceError(null)
  }

  const hasSelection = Boolean(selectedSource || draftToken)
  const sourceCodeEditorKey = useMemo(() => {
    const scope = draftToken || selectedSourceId || 'new'
    const version = selectedSource?.version || 0
    return `${scope}:${version}:${editorSourceKind}`
  }, [draftToken, selectedSourceId, selectedSource?.version, editorSourceKind])
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

  const hasSimpleForm = editorSourceKind === 'rss' || editorSourceKind === 'rest_api' || editorSourceKind === 'twitter'
  const saveDisabled = busy || !editorName.trim() || !editorSlug.trim() || (!hasSimpleForm && !editorCode.trim())

  const latestRun = runsQuery.data?.runs?.[0] || null
  const sourceHealthStatus = worldSourceStatusQuery.data?.sources || {}
  const sourceHealthError = worldSourceStatusQuery.data?.errors?.[0] || null

  return (
    <div className="h-full min-h-0 flex gap-3">
      <div className="w-[300px] shrink-0 min-h-0 flex flex-col rounded-lg border border-border/70 bg-card/50">
        <div className="shrink-0 p-3 space-y-3 border-b border-border/50">
          <div className="flex items-center justify-between gap-2">
            <Button
              type="button"
              size="sm"
              className="h-7 gap-1.5 px-2.5 text-[11px] flex-1"
              onClick={openCreateModal}
              disabled={busy}
            >
              <Plus className="w-3 h-3" />
              New Source
            </Button>
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

          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="h-7 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="z-[180]">
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

        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
          <div className="p-1.5 pr-2 space-y-1">
            {sourcesQuery.isError ? (
              <p className="px-3 py-6 text-xs text-red-300 text-center">
                {errorMessage(sourcesQuery.error, 'Failed to load data sources.')}
              </p>
            ) : sourcesQuery.isPending ? (
              <div className="space-y-1">
                {Array.from({ length: 7 }).map((_, index) => (
                  <div
                    key={`source-skeleton-${index}`}
                    className="rounded-md border border-border/40 bg-background/35 px-2.5 py-2 animate-pulse"
                  >
                    <div className="flex items-center justify-between gap-2 min-w-0">
                      <div className="h-3 w-28 rounded bg-muted/60" />
                      <div className="h-4 w-12 rounded bg-muted/55" />
                    </div>
                    <div className="mt-2 h-2.5 w-36 rounded bg-muted/55" />
                    <div className="mt-1.5 h-2.5 w-16 rounded bg-muted/50" />
                    <div className="mt-1.5 h-6 w-full rounded bg-muted/45" />
                  </div>
                ))}
              </div>
            ) : flatFiltered.length === 0 ? (
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
                        <div className="flex items-center justify-between gap-2 min-w-0">
                          <p className="min-w-0 flex-1 text-xs font-medium truncate" title={source.name}>
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
                          <div className="mt-1 flex items-center justify-between rounded bg-background/50 px-1.5 py-1 gap-2 min-w-0">
                            <span className="min-w-0 flex-1 truncate text-[10px] font-mono text-muted-foreground">
                              {healthSourceName}
                            </span>
                            <span className={cn('text-[10px] font-mono shrink-0', sourceToneClasses(healthTone))}>
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
        </div>

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
                {selectedSource && onOpenCopilot && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1 px-2 text-[11px]"
                    onClick={() =>
                      onOpenCopilot({
                        contextType: 'data_source',
                        contextId: selectedSource.id,
                        label: selectedSource.name || selectedSource.slug || 'Data Source',
                        prompt: 'Review this data source and suggest robust parsing/normalization improvements. Apply direct changes when appropriate.',
                        autoSend: false,
                      })
                    }
                    disabled={busy}
                  >
                    <Sparkles className="w-3 h-3" />
                    Copilot
                  </Button>
                )}
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
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => setShowPreview(true)}
                  disabled={busy || !selectedSource}
                >
                  <Eye className="w-3 h-3" />
                  Preview
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px] bg-cyan-600 hover:bg-cyan-500 text-white"
                  onClick={() => saveMutation.mutate()}
                  disabled={saveDisabled}
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
                          <SelectContent className="z-[180]">
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
                          <SelectContent className="z-[180]">
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
                    <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Retention Max Records</Label>
                        <Input
                          type="number"
                          min={1}
                          max={250000}
                          value={editorRetention.max_records != null ? String(editorRetention.max_records) : ''}
                          onChange={(event) => {
                            const value = event.target.value
                            setEditorRetention((prev) => {
                              const next = { ...prev }
                              if (!value.trim()) {
                                delete next.max_records
                                return next
                              }
                              const parsed = Number(value)
                              if (!Number.isFinite(parsed)) {
                                delete next.max_records
                                return next
                              }
                              next.max_records = Math.max(1, Math.min(250000, Math.round(parsed)))
                              return next
                            })
                          }}
                          className="mt-1 h-8 text-xs font-mono"
                          placeholder="Disabled"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Retention Max Age (days)</Label>
                        <Input
                          type="number"
                          min={1}
                          max={3650}
                          value={editorRetention.max_age_days != null ? String(editorRetention.max_age_days) : ''}
                          onChange={(event) => {
                            const value = event.target.value
                            setEditorRetention((prev) => {
                              const next = { ...prev }
                              if (!value.trim()) {
                                delete next.max_age_days
                                return next
                              }
                              const parsed = Number(value)
                              if (!Number.isFinite(parsed)) {
                                delete next.max_age_days
                                return next
                              }
                              next.max_age_days = Math.max(1, Math.min(3650, Math.round(parsed)))
                              return next
                            })
                          }}
                          className="mt-1 h-8 text-xs font-mono"
                          placeholder="Disabled"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex-1 min-h-0 flex flex-col">
                {hasSimpleForm && !showAdvancedCode ? (
                  <>
                    <div className="px-4 py-2 flex items-center justify-between shrink-0">
                      <div className="flex items-center gap-2">
                        <Settings2 className="w-3.5 h-3.5 text-cyan-400" />
                        <span className="text-xs font-medium">
                          {editorSourceKind === 'rss' ? 'RSS Feed Configuration' : editorSourceKind === 'twitter' ? 'Twitter/X Configuration' : 'REST API Configuration'}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setShowAdvancedCode(true)}
                        className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono flex items-center gap-1"
                      >
                        <Code2 className="w-3 h-3" />
                        Advanced
                      </button>
                    </div>
                    <div className="flex-1 min-h-0 overflow-auto px-4 pb-3">
                      {editorSourceKind === 'rss' ? (
                        <RssSourceForm
                          config={(() => {
                            try { return JSON.parse(editorConfigJson || '{}') } catch { return {} }
                          })()}
                          onChange={(config) => setEditorConfigJson(JSON.stringify(config, null, 2))}
                        />
                      ) : editorSourceKind === 'twitter' ? (
                        <TwitterSourceForm
                          config={(() => {
                            try { return JSON.parse(editorConfigJson || '{}') } catch { return {} }
                          })()}
                          onChange={(config) => setEditorConfigJson(JSON.stringify(config, null, 2))}
                        />
                      ) : (
                        <RestApiSourceForm
                          config={(() => {
                            try { return JSON.parse(editorConfigJson || '{}') } catch { return {} }
                          })()}
                          onChange={(config) => setEditorConfigJson(JSON.stringify(config, null, 2))}
                        />
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="px-4 py-2 flex items-center justify-between shrink-0">
                      <div className="flex items-center gap-2">
                        <Code2 className="w-3.5 h-3.5 text-cyan-400" />
                        <span className="text-xs font-medium">Source Code</span>
                        <span className="text-[10px] text-muted-foreground font-mono">Python</span>
                      </div>
                      <div className="flex items-center gap-3">
                        {inferredClassName && (
                          <span className="text-[10px] font-mono text-muted-foreground">class {inferredClassName}</span>
                        )}
                        {hasSimpleForm && (
                          <button
                            type="button"
                            onClick={() => setShowAdvancedCode(false)}
                            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono flex items-center gap-1"
                          >
                            <Settings2 className="w-3 h-3" />
                            Simple
                          </button>
                        )}
                      </div>
                    </div>
                    <div className="flex-1 min-h-0 px-3 pb-2">
                      <CodeEditor
                        key={sourceCodeEditorKey}
                        value={editorCode}
                        onChange={setEditorCode}
                        language="python"
                        className="h-full"
                        minHeight="100%"
                        placeholder="Write your data source source code here..."
                      />
                    </div>
                  </>
                )}
              </div>

              <div className="min-h-0 border-t border-border/50 flex flex-col" style={{ maxHeight: showConfig ? '50%' : undefined }}>
                <button
                  type="button"
                  onClick={() => setShowConfig((prev) => !prev)}
                  className="shrink-0 w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
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
                  <div className="px-3 pb-3 animate-in fade-in duration-200 space-y-3 overflow-y-auto min-h-0">
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

      {typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {showCreateModal && (
            <motion.div
              key="create-data-source-modal"
              className="fixed inset-0 z-[140] flex items-start sm:items-center justify-center overflow-y-auto p-3 sm:p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="absolute inset-0 bg-black/35 dark:bg-black/75 backdrop-blur-[2px] dark:backdrop-blur-[3px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={() => setShowCreateModal(false)}
                aria-hidden
              />
              <motion.div
                role="dialog"
                aria-modal="true"
                aria-label="Create data source draft"
                className="relative z-10 flex max-h-[calc(100vh-1.5rem)] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border/70 bg-gradient-to-br from-background via-background to-cyan-50/70 dark:border-cyan-500/30 dark:from-card dark:via-card dark:to-cyan-950/20 shadow-[0_40px_120px_rgba(0,0,0,0.35)] dark:shadow-[0_40px_120px_rgba(0,0,0,0.55)] sm:max-h-[calc(100vh-3rem)]"
                initial={{ scale: 0.94, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.98, opacity: 0, y: 12 }}
                transition={{ type: 'spring', stiffness: 250, damping: 28, mass: 0.95 }}
              >
                <div className="border-b border-border/60 px-5 py-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold">Create Data Source</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Pick a source type and template, then continue to configure it in the editor.
                      </p>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-7 w-7 p-0"
                      onClick={() => setShowCreateModal(false)}
                    >
                      <X className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>

                <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-4 lg:grid-cols-[1.2fr_0.8fr]">
                  <div className="space-y-4">
                    <div className="rounded-lg border border-cyan-500/25 bg-cyan-500/5 px-3 py-3 space-y-2.5">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-xs font-semibold flex items-center gap-1.5">
                            <Sparkles className="w-3.5 h-3.5 text-cyan-300" />
                            Generate with AI
                          </p>
                          <p className="mt-0.5 text-[10px] text-muted-foreground">
                            Describe the external site/API/feed and required normalized records.
                          </p>
                        </div>
                        <Button
                          type="button"
                          size="sm"
                          className="h-7 gap-1.5 text-[11px] bg-cyan-600 hover:bg-cyan-500 text-white"
                          onClick={() => generateAiDraftMutation.mutate()}
                          disabled={generateAiDraftMutation.isPending || !newSourceAiPrompt.trim()}
                        >
                          {generateAiDraftMutation.isPending ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                          ) : (
                            <Sparkles className="w-3 h-3" />
                          )}
                          Generate
                        </Button>
                      </div>
                      <textarea
                        value={newSourceAiPrompt}
                        onChange={(event) => {
                          setNewSourceAiPrompt(event.target.value)
                          setNewSourceError(null)
                        }}
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-xs leading-5 text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        rows={3}
                        placeholder="Connect obscure event feed from xyzsite.com and normalize records with stable id, title, timestamp, category, and geopoint when available."
                      />
                      {newSourceAiDraft && (
                        <div className="rounded-md border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-2 space-y-1.5">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-[10px] text-cyan-100">
                              Generated by {newSourceAiDraft.model || 'AI'}{newSourceAiDraft.used_repair_pass ? ' with repair pass' : ''}
                            </p>
                            <div className="flex items-center gap-1.5">
                              <Badge
                                variant="outline"
                                className={cn(
                                  'h-4 px-1.5 text-[9px] border',
                                  newSourceAiDraft.validation?.valid
                                    ? 'border-emerald-500/35 text-emerald-300 bg-emerald-500/10'
                                    : 'border-amber-500/35 text-amber-300 bg-amber-500/10'
                                )}
                              >
                                {newSourceAiDraft.validation?.valid ? 'Validated' : 'Needs review'}
                              </Badge>
                              <div className="flex items-center gap-1">
                                <span className="text-[10px] text-muted-foreground">Use AI Draft</span>
                                <Switch checked={newSourceUseAiDraft} onCheckedChange={setNewSourceUseAiDraft} />
                              </div>
                            </div>
                          </div>
                          {!newSourceAiDraft.validation?.valid && (newSourceAiDraft.validation?.errors || []).slice(0, 2).map((error, index) => (
                            <p key={`source-ai-error-${index}`} className="text-[10px] text-amber-300 font-mono">
                              {error}
                            </p>
                          ))}
                          {(newSourceAiDraft.validation?.warnings || []).slice(0, 2).map((warning, index) => (
                            <p key={`source-ai-warning-${index}`} className="text-[10px] text-muted-foreground font-mono">
                              {warning}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Name</Label>
                        <Input
                          value={newSourceName}
                          onChange={(event) => {
                            const value = event.target.value
                            setNewSourceName(value)
                            setNewSourceError(null)
                            if (!newSourceSlugDirty) {
                              const autoSlug = normalizeSlug(value)
                              setNewSourceSlug(autoSlug || `custom_${Date.now().toString().slice(-6)}`)
                            }
                          }}
                          className="mt-1 h-9 text-xs"
                          placeholder="My Data Source"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source Slug</Label>
                        <Input
                          value={newSourceSlug}
                          onChange={(event) => {
                            setNewSourceSlugDirty(true)
                            setNewSourceSlug(normalizeSlug(event.target.value))
                            setNewSourceError(null)
                          }}
                          className="mt-1 h-9 text-xs font-mono"
                          placeholder="my_data_source"
                        />
                      </div>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source</Label>
                        <Select value={newSourceSourceKey} onValueChange={setNewSourceSourceKey}>
                          <SelectTrigger className="mt-1 h-9 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="z-[180]">
                            {sourceKeys.map((sourceKey) => (
                              <SelectItem key={sourceKey} value={sourceKey}>
                                {SOURCE_LABELS[sourceKey] || sourceKey}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source Kind</Label>
                        <Select
                          value={String(newSourceTemplateKind || 'python')}
                          onValueChange={(value) => {
                            setNewSourceTemplateKind(value)
                            setNewSourceError(null)
                          }}
                        >
                          <SelectTrigger className="mt-1 h-9 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="z-[180]">
                            {templateKinds.map((kind) => (
                              <SelectItem key={kind} value={kind}>
                                {SOURCE_KIND_LABELS[kind] || kind}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">Description</Label>
                      <textarea
                        value={newSourceDescription}
                        onChange={(event) => {
                          setNewSourceDescription(event.target.value)
                          setNewSourceError(null)
                        }}
                        className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-xs leading-5 text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        rows={3}
                        placeholder="Describe what this data source fetches."
                      />
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">
                        Template ({templatesForCreateKind.length} available for type {SOURCE_KIND_LABELS[newSourceTemplateKind] || newSourceTemplateKind})
                      </Label>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        {templatesForCreateKind.map((preset) => {
                          const active = preset.id === (selectedCreateTemplate?.id || '')
                          return (
                            <button
                              type="button"
                              key={preset.id}
                              onClick={() => {
                                setNewSourceTemplateKey(preset.id)
                                setNewSourceTemplateKind(String(preset.source_kind || 'python').trim().toLowerCase())
                                setNewSourceSourceKey(String(preset.source_key || newSourceSourceKey).trim().toLowerCase() || 'custom')
                                setNewSourceError(null)
                              }}
                              className={cn(
                                'rounded-lg border px-3 py-2 text-left transition-all',
                                active
                                  ? 'border-cyan-500/70 bg-cyan-500/10 shadow-[0_0_0_1px_rgba(6,182,212,0.22)]'
                                  : 'border-border/70 bg-card/50 hover:border-border'
                              )}
                            >
                              <p className="text-xs font-semibold">{preset.name}</p>
                              <p className="text-[10px] text-muted-foreground leading-relaxed mt-1">{preset.description}</p>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  </div>

                  <div className="min-h-0 rounded-xl border border-border/60 bg-muted/30 dark:bg-black/30">
                    <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
                      <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Template Preview</span>
                      <span className="text-[10px] font-mono text-muted-foreground">Read-only</span>
                    </div>
                    <div className="max-h-[460px] overflow-auto px-3 py-2">
                      <pre className="whitespace-pre-wrap text-[11px] leading-5 text-muted-foreground font-mono">{newSourceTemplatePreviewCode}</pre>
                    </div>
                  </div>
                </div>

                {newSourceError && (
                  <div className="mx-4 mb-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                    {newSourceError}
                  </div>
                )}

                <div className="flex items-center justify-end gap-2 border-t border-border/60 px-4 py-3">
                  <Button type="button" variant="outline" size="sm" onClick={() => setShowCreateModal(false)}>
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    className="bg-cyan-600 hover:bg-cyan-500 text-white"
                    onClick={createDraftFromModal}
                    disabled={busy || !newSourceName.trim() || !normalizeSlug(newSourceSlug)}
                  >
                    Continue to Editor
                  </Button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      <DataSourceApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} />
      <DataSourcePreviewFlyout open={showPreview} onOpenChange={setShowPreview} sourceId={selectedSourceId} />
    </div>
  )
}
