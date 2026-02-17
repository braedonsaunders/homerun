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
  Copy,
  FlaskConical,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings2,
  Trash2,
  X,
  Zap,
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
import StrategyConfigForm from './StrategyConfigForm'
import {
  getUnifiedStrategies,
  createUnifiedStrategy,
  updateUnifiedStrategy,
  deleteUnifiedStrategy,
  validateUnifiedStrategy,
  reloadUnifiedStrategy,
  getUnifiedStrategyTemplate,
  UnifiedStrategy,
} from '../services/api'
import StrategyApiDocsFlyout from './StrategyApiDocsFlyout'
import StrategyBacktestFlyout from './StrategyBacktestFlyout'

// ==================== Helpers ====================

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

function uniqueStrings(values: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of values) {
    const value = String(raw || '').trim().toLowerCase()
    if (!value || seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
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
    const isStrategyClass = bases.some(
      (base) =>
        base === 'BaseStrategy' ||
        base.endsWith('.BaseStrategy') ||
        base === 'BaseTraderStrategy' ||
        base.endsWith('.BaseTraderStrategy')
    )
    if (className && isStrategyClass) {
      return className
    }
    match = classPattern.exec(sourceCode)
  }
  return null
}

function parseAliases(csv: string): string[] {
  return uniqueStrings(
    String(csv || '')
      .split(',')
      .map((item) => item.trim())
  )
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

// ==================== Constants ====================

const STATUS_COLORS: Record<string, string> = {
  loaded: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  active: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  error: 'bg-red-500/15 text-red-400 border-red-500/30',
  unloaded: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  draft: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
}

const SOURCE_LABELS: Record<string, string> = {
  scanner: 'Scanner',
  news: 'News',
  crypto: 'Crypto',
  weather: 'Weather',
  traders: 'Traders',
}

// ==================== Capability Detection ====================

interface Capabilities {
  has_detect: boolean
  has_detect_async: boolean
  has_evaluate: boolean
  has_should_exit: boolean
}

function CapabilityBadges({ capabilities }: { capabilities?: Capabilities }) {
  const caps = capabilities || { has_detect: false, has_detect_async: false, has_evaluate: false, has_should_exit: false }
  return (
    <div className="flex items-center gap-1 mt-1">
      {(caps.has_detect || caps.has_detect_async) && (
        <span className="text-[8px] font-semibold uppercase tracking-wider px-1.5 py-0 rounded bg-amber-500/15 text-amber-400 border border-amber-500/25">
          Detect
        </span>
      )}
      {caps.has_evaluate && (
        <span className="text-[8px] font-semibold uppercase tracking-wider px-1.5 py-0 rounded bg-violet-500/15 text-violet-400 border border-violet-500/25">
          Evaluate
        </span>
      )}
      {caps.has_should_exit && (
        <span className="text-[8px] font-semibold uppercase tracking-wider px-1.5 py-0 rounded bg-rose-500/15 text-rose-400 border border-rose-500/25">
          Exit
        </span>
      )}
    </div>
  )
}

// ==================== Fallback Template ====================

const FALLBACK_TEMPLATE = [
  'from models import Market, Event, ArbitrageOpportunity',
  'from services.strategies.base import BaseStrategy',
  '',
  '',
  'class CustomStrategy(BaseStrategy):',
  '    name = "Custom Strategy"',
  '    description = "Describe what this strategy does"',
  '',
  '    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:',
  '        """Scan market data and return tradeable opportunities."""',
  '        opportunities = []',
  '        # TODO: add detection logic',
  '        return opportunities',
  '',
  '    def evaluate(self, signal, context):',
  '        """Decide whether to trade a detected opportunity."""',
  '        # TODO: add evaluation logic',
  '        return None',
  '',
  '    def should_exit(self, position, context):',
  '        """Decide whether to exit an open position."""',
  '        # TODO: add exit logic',
  '        return False',
  '',
].join('\n')

// ==================== Main Component ====================

export default function UnifiedStrategiesManager() {
  const queryClient = useQueryClient()

  // UI toggles
  const [showSettings, setShowSettings] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [showRawJson, setShowRawJson] = useState(false)
  const [showApiDocs, setShowApiDocs] = useState(false)
  const [showBacktest, setShowBacktest] = useState(false)

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  // Selection
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null)
  const [draftToken, setDraftToken] = useState<string | null>(null)

  // Editor state
  const [editorSlug, setEditorSlug] = useState('')
  const [editorName, setEditorName] = useState('')
  const [editorDescription, setEditorDescription] = useState('')
  const [editorSourceKey, setEditorSourceKey] = useState('scanner')
  const [editorEnabled, setEditorEnabled] = useState(true)
  const [editorCode, setEditorCode] = useState('')
  const [editorConfigJson, setEditorConfigJson] = useState('{}')
  const [editorSchemaJson, setEditorSchemaJson] = useState('{}')
  const [editorAliasesCsv, setEditorAliasesCsv] = useState('')
  const [editorError, setEditorError] = useState<string | null>(null)
  const [validation, setValidation] = useState<{
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
  } | null>(null)

  // ── Queries ──

  const strategiesQuery = useQuery({
    queryKey: ['unified-strategies'],
    queryFn: () => getUnifiedStrategies(),
    staleTime: 15000,
    refetchInterval: 15000,
  })

  const templateQuery = useQuery({
    queryKey: ['unified-strategy-template'],
    queryFn: getUnifiedStrategyTemplate,
    staleTime: Infinity,
  })

  const catalog = strategiesQuery.data || []

  // ── Derived state ──

  const sourceKeys = useMemo(() => {
    const fromCatalog = catalog.map((s) => String(s.source_key || '').toLowerCase())
    const merged = uniqueStrings(fromCatalog)
    return merged.length > 0 ? merged : ['scanner']
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
          (s.name || '').toLowerCase().includes(q) ||
          (s.slug || '').toLowerCase().includes(q) ||
          (s.source_key || '').toLowerCase().includes(q) ||
          (s.description || '').toLowerCase().includes(q) ||
          (s.class_name || '').toLowerCase().includes(q)
      )
    }
    // Group by source_key
    const groups: Record<string, UnifiedStrategy[]> = {}
    for (const s of rows) {
      const key = s.source_key || 'other'
      if (!groups[key]) groups[key] = []
      groups[key].push(s)
    }
    return groups
  }, [catalog, sourceFilter, searchQuery])

  const flatFiltered = useMemo(() => Object.values(grouped).flat(), [grouped])

  const selectedStrategy = useMemo(
    () => catalog.find((s) => s.id === selectedStrategyId) || null,
    [selectedStrategyId, catalog]
  )

  const inferredClassName = useMemo(() => inferClassName(editorCode), [editorCode])

  // ── Auto-select first ──

  useEffect(() => {
    if (selectedStrategyId) return
    if (draftToken) return
    if (catalog.length > 0) {
      setSelectedStrategyId(catalog[0].id)
    }
  }, [selectedStrategyId, catalog, draftToken])

  // ── Sync editor from selection ──

  useEffect(() => {
    if (!selectedStrategyId) return
    const strategy = catalog.find((s) => s.id === selectedStrategyId)
    if (!strategy) return
    setDraftToken(null)
    setEditorSlug(strategy.slug || '')
    setEditorName(strategy.name || '')
    setEditorDescription(strategy.description || '')
    setEditorSourceKey(strategy.source_key || 'scanner')
    setEditorEnabled(Boolean(strategy.enabled))
    setEditorCode(strategy.source_code || '')
    setEditorConfigJson(JSON.stringify(strategy.config || {}, null, 2))
    setEditorSchemaJson(JSON.stringify(strategy.config_schema || {}, null, 2))
    setEditorAliasesCsv((strategy.aliases || []).join(', '))
    setEditorError(null)
    setValidation(null)
  }, [selectedStrategyId, catalog])

  // ── Refresh helper ──

  const refreshCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-strategies'] })
    queryClient.invalidateQueries({ queryKey: ['strategies'] })
    queryClient.invalidateQueries({ queryKey: ['plugins'] })
    queryClient.invalidateQueries({ queryKey: ['trader-strategies-catalog'] })
  }

  // ── Mutations ──

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
        name: String(editorName || '').trim(),
        description: editorDescription.trim() || undefined,
        source_code: editorCode,
        config: parsedConfig.value,
        config_schema: parsedSchema.value,
        aliases: parseAliases(editorAliasesCsv),
        enabled: editorEnabled,
      }

      if (!payload.slug) throw new Error('Strategy key is required')
      if (!payload.name) throw new Error('Name is required')

      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (selected) {
        return updateUnifiedStrategy(selected.id, {
          ...payload,
          unlock_system: Boolean(selected.is_system),
        })
      }
      return createUnifiedStrategy(payload)
    },
    onSuccess: (strategy) => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedStrategyId(strategy.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Failed to save strategy'))
    },
  })

  const validateMutation = useMutation({
    mutationFn: async () => validateUnifiedStrategy(editorCode),
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
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to reload')
      return reloadUnifiedStrategy(selected.id)
    },
    onSuccess: () => {
      setEditorError(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Reload failed'))
    },
  })

  const cloneMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to clone')
      return createUnifiedStrategy({
        slug: `${selected.slug}_clone_${Date.now().toString().slice(-6)}`,
        source_key: selected.source_key || 'scanner',
        name: `${selected.name} (Clone)`,
        description: selected.description || undefined,
        source_code: selected.source_code || '',
        config: selected.config || {},
        config_schema: selected.config_schema || {},
        aliases: selected.aliases || [],
        enabled: true,
      })
    },
    onSuccess: (strategy) => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedStrategyId(strategy.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Clone failed'))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to delete')
      return deleteUnifiedStrategy(selected.id)
    },
    onSuccess: () => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedStrategyId(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Delete failed'))
    },
  })

  const busy =
    saveMutation.isPending ||
    validateMutation.isPending ||
    reloadMutation.isPending ||
    cloneMutation.isPending ||
    deleteMutation.isPending

  // ── New draft ──

  const startNewDraft = () => {
    setSelectedStrategyId(null)
    setDraftToken(`draft_${Date.now()}`)
    setEditorSlug(`custom_${Date.now().toString().slice(-6)}`)
    setEditorSourceKey('scanner')
    setEditorName('Custom Strategy')
    setEditorDescription('')
    setEditorEnabled(true)
    setEditorCode(templateQuery.data?.template || FALLBACK_TEMPLATE)
    setEditorConfigJson('{}')
    setEditorSchemaJson('{"param_fields": []}')
    setEditorAliasesCsv('')
    setEditorError(null)
    setValidation(null)
  }

  // ── Derived display state ──

  const hasSelection = Boolean(selectedStrategy || draftToken)
  const displayStatus = selectedStrategy?.status || 'draft'
  const statusColor = STATUS_COLORS[displayStatus] || STATUS_COLORS.draft

  // Determine flyout variant based on capabilities
  const flyoutVariant: 'opportunity' | 'trader' = useMemo(() => {
    const code = editorCode || ''
    const hasDetect = /def\s+(detect|detect_async)\s*\(/.test(code) || /BaseWeatherStrategy/.test(code)
    const hasEvaluate = /def\s+evaluate\s*\(/.test(code) || /Base(Weather)?Strategy/.test(code)
    if (hasEvaluate && !hasDetect) return 'trader'
    return 'opportunity'
  }, [editorCode])

  // Parse config_schema for StrategyConfigForm
  const configSchemaFields = useMemo(() => {
    try {
      const schema = JSON.parse(editorSchemaJson || '{}')
      return schema?.param_fields || []
    } catch {
      return []
    }
  }, [editorSchemaJson])

  // ==================== Render ====================

  return (
    <div className="h-full min-h-0 flex gap-3">
      {/* ══════════════════ Left sidebar: Strategy list ══════════════════ */}
      <div className="w-[280px] shrink-0 min-h-0 flex flex-col rounded-lg border border-border/70 bg-card/50">
        {/* Header actions */}
        <div className="shrink-0 p-3 space-y-3 border-b border-border/50">
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              className="h-7 gap-1.5 px-2.5 text-[11px] flex-1"
              onClick={startNewDraft}
              disabled={busy}
            >
              <Plus className="w-3 h-3" />
              New Strategy
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 gap-1 px-2 text-[11px]"
              onClick={() => cloneMutation.mutate()}
              disabled={busy || !selectedStrategy}
              title="Clone selected strategy"
            >
              <Copy className="w-3 h-3" />
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px]"
              onClick={() => refreshCatalog()}
              disabled={busy}
              title="Refresh catalog"
            >
              <RefreshCw className={cn('w-3 h-3', strategiesQuery.isFetching && 'animate-spin')} />
            </Button>
          </div>

          {/* Source filter */}
          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="h-7 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources ({catalog.length})</SelectItem>
              {sourceKeys.map((sk) => (
                <SelectItem key={sk} value={sk}>
                  {SOURCE_LABELS[sk] || sk} ({catalog.filter((s) => s.source_key === sk).length})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search strategies..."
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

        {/* Strategy list — grouped by source_key */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-1.5 space-y-1">
            {flatFiltered.length === 0 ? (
              <p className="px-3 py-6 text-xs text-muted-foreground text-center">No strategies found.</p>
            ) : (
              Object.entries(grouped).map(([sourceKey, strategies]) => (
                <div key={sourceKey}>
                  {/* Section divider */}
                  {Object.keys(grouped).length > 1 && (
                    <div className="px-2.5 pt-2.5 pb-1 flex items-center gap-1.5">
                      <div className="flex-1 h-px bg-border/50" />
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-medium shrink-0">
                        {SOURCE_LABELS[sourceKey] || sourceKey} ({strategies.length})
                      </p>
                      <div className="flex-1 h-px bg-border/50" />
                    </div>
                  )}
                  {strategies.map((strategy) => {
                    const active = selectedStrategyId === strategy.id
                    const sColor = STATUS_COLORS[strategy.status] || STATUS_COLORS.draft
                    return (
                      <button
                        key={strategy.id}
                        type="button"
                        onClick={() => {
                          setDraftToken(null)
                          setSelectedStrategyId(strategy.id)
                        }}
                        className={cn(
                          'w-full rounded-md px-2.5 py-2 text-left transition-all duration-150',
                          active ? 'bg-violet-500/10 ring-1 ring-violet-500/30' : 'hover:bg-muted/50'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-medium truncate" title={strategy.name}>
                            {strategy.name}
                          </p>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {!strategy.enabled && (
                              <span className="w-1.5 h-1.5 rounded-full bg-zinc-500" title="Disabled" />
                            )}
                            <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0 h-4 border', sColor)}>
                              {strategy.status}
                            </Badge>
                          </div>
                        </div>
                        <p className="text-[10px] font-mono text-muted-foreground mt-1 truncate">
                          {strategy.slug}
                        </p>
                        {strategy.capabilities && <CapabilityBadges capabilities={strategy.capabilities} />}
                      </button>
                    )
                  })}
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer stats */}
        <div className="shrink-0 px-3 py-2 border-t border-border/50 text-[10px] text-muted-foreground flex justify-between">
          <span>{flatFiltered.length} strategies</span>
          <span>{catalog.filter((s) => s.enabled).length} enabled</span>
        </div>
      </div>

      {/* ══════════════════ Right panel: Editor ══════════════════ */}
      <div className="flex-1 min-w-0 min-h-0 flex flex-col rounded-lg border border-border/70">
        {!hasSelection ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center space-y-2">
              <Code2 className="w-8 h-8 mx-auto opacity-40" />
              <p className="text-sm">Select a strategy or create a new one</p>
            </div>
          </div>
        ) : (
          <>
            {/* ── Editor toolbar ── */}
            <div className="shrink-0 px-4 py-2.5 border-b border-border/50 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="flex items-center gap-2 min-w-0">
                  <Code2 className="w-4 h-4 text-violet-400 shrink-0" />
                  <span className="text-sm font-medium truncate">
                    {editorName || 'Untitled Strategy'}
                  </span>
                </div>
                <Badge variant="outline" className={cn('text-[10px] shrink-0 border', statusColor)}>
                  {displayStatus}
                </Badge>
                {selectedStrategy?.is_system && (
                  <Badge variant="secondary" className="text-[10px] shrink-0">System</Badge>
                )}
                {selectedStrategy && (
                  <span className="text-[10px] font-mono text-muted-foreground shrink-0">
                    v{selectedStrategy.version}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <div className="flex items-center gap-2 mr-2 pr-2 border-r border-border/50">
                  <span className="text-[10px] text-muted-foreground">Enabled</span>
                  <Switch
                    checked={editorEnabled}
                    onCheckedChange={setEditorEnabled}
                    className="scale-75"
                  />
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => setShowBacktest(true)}
                  disabled={!editorCode.trim()}
                >
                  <FlaskConical className="w-3 h-3" />
                  Backtest
                </Button>
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
                  {validateMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Check className="w-3 h-3" />
                  )}
                  Validate
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => reloadMutation.mutate()}
                  disabled={busy || !selectedStrategy}
                >
                  {reloadMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Zap className="w-3 h-3" />
                  )}
                  Reload
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px] bg-violet-600 hover:bg-violet-500 text-white"
                  onClick={() => saveMutation.mutate()}
                  disabled={
                    busy ||
                    !editorCode.trim() ||
                    !editorName.trim() ||
                    !editorSlug.trim()
                  }
                >
                  {saveMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Save className="w-3 h-3" />
                  )}
                  Save
                </Button>
                {selectedStrategy && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 px-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    onClick={() => {
                      const message = selectedStrategy.is_system
                        ? `Delete system strategy "${selectedStrategy.name}"? It will be tombstoned and will NOT auto-reseed.`
                        : `Delete "${selectedStrategy.name}"? This cannot be undone.`
                      if (window.confirm(message)) deleteMutation.mutate()
                    }}
                    disabled={busy}
                    title="Delete strategy"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                )}
              </div>
            </div>

            {/* ── Validation / Error banners ── */}
            {validation && (
              <div
                className={cn(
                  'shrink-0 px-4 py-2 text-xs flex items-start gap-2 border-b',
                  validation.valid
                    ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
                    : 'bg-red-500/5 border-red-500/20 text-red-400'
                )}
              >
                {validation.valid ? (
                  <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                ) : (
                  <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                )}
                <div className="min-w-0">
                  <p className="font-medium">
                    {validation.valid ? 'Validation passed' : 'Validation failed'}
                    {validation.class_name && (
                      <span className="font-mono ml-2 opacity-70">{validation.class_name}</span>
                    )}
                  </p>
                  {validation.errors.map((err, i) => (
                    <p key={`e-${i}`} className="font-mono text-[11px] mt-0.5">{err}</p>
                  ))}
                  {validation.warnings.map((warn, i) => (
                    <p key={`w-${i}`} className="font-mono text-[11px] mt-0.5 text-amber-400">{warn}</p>
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

            {/* ── Main editor content ── */}
            <div className="flex-1 min-h-0 flex flex-col">
              {/* Collapsible Strategy Settings */}
              <div className="shrink-0 border-b border-border/50">
                <button
                  type="button"
                  onClick={() => setShowSettings((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showSettings ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Strategy Settings</span>
                  <span className="ml-auto font-mono text-[10px] opacity-60">
                    {editorSlug || 'no-key'}
                  </span>
                </button>

                {showSettings && (
                  <div className="px-4 pb-3 space-y-3 animate-in fade-in duration-200">
                    <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Strategy Key</Label>
                        <Input
                          value={editorSlug}
                          onChange={(e) => setEditorSlug(normalizeSlug(e.target.value))}
                          className="mt-1 h-8 text-xs font-mono"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Name</Label>
                        <Input
                          value={editorName}
                          onChange={(e) => setEditorName(e.target.value)}
                          className="mt-1 h-8 text-xs"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source</Label>
                        <Select value={editorSourceKey} onValueChange={setEditorSourceKey}>
                          <SelectTrigger className="mt-1 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {sourceKeys.map((sk) => (
                              <SelectItem key={sk} value={sk}>
                                {SOURCE_LABELS[sk] || sk}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Class Name</Label>
                        <Input
                          value={
                            validation?.class_name ||
                            inferredClassName ||
                            selectedStrategy?.class_name ||
                            'auto-detected'
                          }
                          className="mt-1 h-8 text-xs font-mono"
                          disabled
                        />
                      </div>
                    </div>
                    <div className="grid gap-3 grid-cols-1 xl:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Description</Label>
                        <Input
                          value={editorDescription}
                          onChange={(e) => setEditorDescription(e.target.value)}
                          className="mt-1 h-8 text-xs"
                          placeholder="Describe what this strategy does..."
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Aliases (comma separated)</Label>
                        <Input
                          value={editorAliasesCsv}
                          onChange={(e) => setEditorAliasesCsv(e.target.value)}
                          className="mt-1 h-8 text-xs font-mono"
                          placeholder="alias_one, alias_two"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Code editor — takes remaining space */}
              <div className="flex-1 min-h-0 flex flex-col">
                <div className="px-4 py-2 flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-3.5 h-3.5 text-violet-400" />
                    <span className="text-xs font-medium">Source Code</span>
                    <span className="text-[10px] text-muted-foreground font-mono">Python</span>
                  </div>
                  {inferredClassName && (
                    <span className="text-[10px] font-mono text-muted-foreground">
                      class {inferredClassName}
                    </span>
                  )}
                </div>
                <div className="flex-1 min-h-0 px-3 pb-2">
                  <CodeEditor
                    value={editorCode}
                    onChange={setEditorCode}
                    language="python"
                    className="h-full"
                    minHeight="100%"
                    placeholder="Write your strategy source code here..."
                  />
                </div>
              </div>

              {/* Collapsible Runtime Config section */}
              <div className="shrink-0 border-t border-border/50">
                <button
                  type="button"
                  onClick={() => setShowConfig((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfig ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
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
                    {/* Dynamic config form when schema has param_fields */}
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
                          onChange={(vals) => setEditorConfigJson(JSON.stringify(vals, null, 2))}
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
                    {/* Raw JSON editors — shown when no schema or toggled */}
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

      {/* ── Flyouts ── */}
      <StrategyApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} variant={flyoutVariant} />
      <StrategyBacktestFlyout
        open={showBacktest}
        onOpenChange={setShowBacktest}
        sourceCode={editorCode}
        slug={editorSlug || '_backtest_preview'}
        config={(() => { try { return JSON.parse(editorConfigJson || '{}') } catch { return {} } })()}
        variant={flyoutVariant}
      />
    </div>
  )
}
