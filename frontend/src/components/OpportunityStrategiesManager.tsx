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
  Link2,
  Loader2,
  Plus,
  RefreshCw,
  Save,
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
  createOpportunityStrategy,
  deleteOpportunityStrategy,
  getOpportunityStrategies,
  getPluginTemplate,
  reloadOpportunityStrategy,
  updateOpportunityStrategy,
  validateOpportunityStrategy,
} from '../services/api'
import StrategyApiDocsFlyout from './StrategyApiDocsFlyout'
import StrategyBacktestFlyout from './StrategyBacktestFlyout'

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

function inferOpportunityClassName(sourceCode: string): string | null {
  const classPattern = /class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:/gm
  let match: RegExpExecArray | null = classPattern.exec(sourceCode)
  while (match) {
    const className = String(match[1] || '').trim()
    const bases = String(match[2] || '')
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    const isStrategyClass = bases.some((base) => base === 'BaseStrategy' || base.endsWith('.BaseStrategy'))
    if (className && isStrategyClass) {
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

const STATUS_COLORS: Record<string, string> = {
  loaded: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  error: 'bg-red-500/15 text-red-400 border-red-500/30',
  unloaded: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  draft: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
}

const PIPELINE_SOURCES = new Set(['news', 'crypto', 'weather', 'traders'])

const SOURCE_LABELS: Record<string, string> = {
  news: 'News tab',
  crypto: 'Crypto tab',
  weather: 'Weather tab',
  traders: 'Traders tab',
}

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

interface OpportunityStrategiesManagerProps {
  initialSourceFilter?: string
  searchQuery?: string
}

export default function OpportunityStrategiesManager({ initialSourceFilter, searchQuery }: OpportunityStrategiesManagerProps) {
  const queryClient = useQueryClient()
  const [showDocs, setShowDocs] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [showRawJson, setShowRawJson] = useState(false)
  const [showApiDocs, setShowApiDocs] = useState(false)
  const [showBacktest, setShowBacktest] = useState(false)
  const [strategyFilterSource, setStrategyFilterSource] = useState<string>(initialSourceFilter || 'all')
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null)
  const [strategyDraftToken, setStrategyDraftToken] = useState<string | null>(null)
  const [strategyEditorSlug, setStrategyEditorSlug] = useState('')
  const [strategyEditorName, setStrategyEditorName] = useState('')
  const [strategyEditorDescription, setStrategyEditorDescription] = useState('')
  const [strategyEditorSourceKey, setStrategyEditorSourceKey] = useState('scanner')
  const [strategyEditorEnabled, setStrategyEditorEnabled] = useState(true)
  const [strategyEditorCode, setStrategyEditorCode] = useState('')
  const [strategyEditorConfigJson, setStrategyEditorConfigJson] = useState('{}')
  const [strategyEditorError, setStrategyEditorError] = useState<string | null>(null)
  const [strategyValidation, setStrategyValidation] = useState<{
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
  } | null>(null)

  const strategiesQuery = useQuery({
    queryKey: ['plugins'],
    queryFn: getOpportunityStrategies,
    staleTime: 15000,
    refetchInterval: 15000,
  })

  const templateQuery = useQuery({
    queryKey: ['plugin-template'],
    queryFn: getPluginTemplate,
    staleTime: Infinity,
  })

  // Sync filter when initialSourceFilter prop changes (e.g., cross-link navigation)
  useEffect(() => {
    if (initialSourceFilter) {
      setStrategyFilterSource(initialSourceFilter)
    }
  }, [initialSourceFilter])

  const strategyCatalog = strategiesQuery.data || []

  const sourceKeys = useMemo(() => {
    const fromCatalog = strategyCatalog.map((strategy) => String(strategy.source_key || '').toLowerCase())
    const merged = uniqueStrings(fromCatalog)
    return merged.length > 0 ? merged : ['scanner']
  }, [strategyCatalog])

  const filteredStrategies = useMemo(() => {
    let rows = [...strategyCatalog]
    if (strategyFilterSource !== 'all') {
      rows = rows.filter((item) => item.source_key === strategyFilterSource)
    }
    if (searchQuery && searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase()
      rows = rows.filter(
        (item) =>
          (item.name || '').toLowerCase().includes(q) ||
          (item.slug || '').toLowerCase().includes(q) ||
          (item.source_key || '').toLowerCase().includes(q) ||
          (item.description || '').toLowerCase().includes(q)
      )
    }
    return rows
  }, [strategyCatalog, strategyFilterSource, searchQuery])

  const selectedStrategy = useMemo(
    () => strategyCatalog.find((item) => item.id === selectedStrategyId) || null,
    [selectedStrategyId, strategyCatalog]
  )
  const inferredClassName = useMemo(
    () => inferOpportunityClassName(strategyEditorCode),
    [strategyEditorCode]
  )

  useEffect(() => {
    if (selectedStrategyId) return
    if (strategyDraftToken) return
    if (strategyCatalog.length > 0) {
      setSelectedStrategyId(strategyCatalog[0].id)
    }
  }, [selectedStrategyId, strategyCatalog, strategyDraftToken])

  useEffect(() => {
    if (!selectedStrategyId) return
    const strategy = strategyCatalog.find((item) => item.id === selectedStrategyId)
    if (!strategy) return
    setStrategyDraftToken(null)
    setStrategyEditorSlug(strategy.slug || '')
    setStrategyEditorName(strategy.name || '')
    setStrategyEditorDescription(strategy.description || '')
    setStrategyEditorSourceKey(strategy.source_key || 'scanner')
    setStrategyEditorEnabled(Boolean(strategy.enabled))
    setStrategyEditorCode(strategy.source_code || '')
    setStrategyEditorConfigJson(JSON.stringify(strategy.config || {}, null, 2))
    setStrategyEditorError(null)
    setStrategyValidation(null)
  }, [selectedStrategyId, strategyCatalog])

  const refreshCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['plugins'] })
    queryClient.invalidateQueries({ queryKey: ['strategies'] })
    queryClient.invalidateQueries({ queryKey: ['opportunity-strategy-counts'] })
  }

  const saveStrategyMutation = useMutation({
    mutationFn: async () => {
      const parsedConfig = parseJsonObject(strategyEditorConfigJson || '{}')
      if (!parsedConfig.value) {
        throw new Error(`Config JSON error: ${parsedConfig.error || 'invalid object'}`)
      }

      const payload = {
        slug: normalizeSlug(strategyEditorSlug),
        source_key: String(strategyEditorSourceKey || '').trim().toLowerCase(),
        name: String(strategyEditorName || '').trim(),
        description: strategyEditorDescription.trim() || undefined,
        source_code: strategyEditorCode,
        config: parsedConfig.value,
        enabled: strategyEditorEnabled,
      }

      if (!payload.slug) {
        throw new Error('Strategy key is required')
      }
      if (!payload.name) {
        throw new Error('Name is required')
      }
      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (selected) {
        return updateOpportunityStrategy(selected.id, payload)
      }
      return createOpportunityStrategy(payload)
    },
    onSuccess: (strategy) => {
      setStrategyEditorError(null)
      setStrategyDraftToken(null)
      setSelectedStrategyId(strategy.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Failed to save strategy'))
    },
  })

  const validateStrategyMutation = useMutation({
    mutationFn: async () => validateOpportunityStrategy(strategyEditorCode),
    onSuccess: (result) => {
      setStrategyValidation({
        valid: Boolean(result.valid),
        class_name: result.class_name || null,
        errors: result.errors || [],
        warnings: result.warnings || [],
      })
      setStrategyEditorError(null)
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Validation failed'))
    },
  })

  const reloadStrategyMutation = useMutation({
    mutationFn: async () => {
      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to reload')
      return reloadOpportunityStrategy(selected.id)
    },
    onSuccess: () => {
      setStrategyEditorError(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Reload failed'))
    },
  })

  const cloneStrategyMutation = useMutation({
    mutationFn: async () => {
      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to clone')
      return createOpportunityStrategy({
        slug: `${selected.slug}_clone_${Date.now().toString().slice(-6)}`,
        source_key: selected.source_key || 'scanner',
        name: `${selected.name} (Clone)`,
        description: selected.description || undefined,
        source_code: selected.source_code || '',
        config: selected.config || {},
        enabled: true,
      })
    },
    onSuccess: (strategy) => {
      setStrategyEditorError(null)
      setStrategyDraftToken(null)
      setSelectedStrategyId(strategy.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Clone failed'))
    },
  })

  const deleteStrategyMutation = useMutation({
    mutationFn: async () => {
      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to delete')
      return deleteOpportunityStrategy(selected.id)
    },
    onSuccess: () => {
      setStrategyEditorError(null)
      setStrategyDraftToken(null)
      setSelectedStrategyId(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Delete failed'))
    },
  })

  const managerBusy =
    saveStrategyMutation.isPending ||
    validateStrategyMutation.isPending ||
    reloadStrategyMutation.isPending ||
    cloneStrategyMutation.isPending ||
    deleteStrategyMutation.isPending

  const startNewStrategyDraft = () => {
    const fallbackTemplate = [
      'from models import Market, Event, ArbitrageOpportunity',
      'from services.strategies.base import BaseStrategy',
      '',
      'class CustomOpportunityStrategy(BaseStrategy):',
      '    name = "Custom Detector"',
      '    description = "Describe what this strategy detects"',
      '',
      '    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:',
      '        opportunities = []',
      '        # TODO: add strategy logic',
      '        return opportunities',
      '',
    ].join('\n')

    setSelectedStrategyId(null)
    setStrategyDraftToken(`draft_${Date.now()}`)
    setStrategyEditorSlug(`custom_${Date.now().toString().slice(-6)}`)
    setStrategyEditorSourceKey('scanner')
    setStrategyEditorName('Custom Detector')
    setStrategyEditorDescription('')
    setStrategyEditorEnabled(true)
    setStrategyEditorCode(templateQuery.data?.template || fallbackTemplate)
    setStrategyEditorConfigJson('{}')
    setStrategyEditorError(null)
    setStrategyValidation(null)
  }

  const hasSelection = Boolean(selectedStrategy || strategyDraftToken)
  const displayStatus = selectedStrategy?.status || 'draft'
  const statusColor = STATUS_COLORS[displayStatus] || STATUS_COLORS.draft

  return (
    <div className="h-full min-h-0 flex gap-3">
      {/* ── Left sidebar: Strategy list ── */}
      <div className="w-[280px] shrink-0 min-h-0 flex flex-col rounded-lg border border-border/70 bg-card/50">
        {/* Header actions */}
        <div className="shrink-0 p-3 space-y-3 border-b border-border/50">
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              className="h-7 gap-1.5 px-2.5 text-[11px] flex-1"
              onClick={startNewStrategyDraft}
              disabled={managerBusy}
            >
              <Plus className="w-3 h-3" />
              New Strategy
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 gap-1 px-2 text-[11px]"
              onClick={() => cloneStrategyMutation.mutate()}
              disabled={managerBusy || !selectedStrategy}
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
              disabled={managerBusy}
              title="Refresh catalog"
            >
              <RefreshCw className={cn('w-3 h-3', strategiesQuery.isFetching && 'animate-spin')} />
            </Button>
          </div>

          <Select value={strategyFilterSource} onValueChange={setStrategyFilterSource}>
            <SelectTrigger className="h-7 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources ({strategyCatalog.length})</SelectItem>
              {sourceKeys.map((sourceKey) => (
                <SelectItem key={sourceKey} value={sourceKey}>
                  {sourceKey}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Strategy list — grouped by scanner vs pipeline */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-1.5 space-y-1">
            {filteredStrategies.length === 0 ? (
              <p className="px-3 py-6 text-xs text-muted-foreground text-center">No strategies found.</p>
            ) : (
              <>
                {/* Scanner strategies */}
                {(() => {
                  const scannerStrategies = filteredStrategies.filter((s) => !PIPELINE_SOURCES.has(s.source_key))
                  const pipelineStrategies = filteredStrategies.filter((s) => PIPELINE_SOURCES.has(s.source_key))
                  return (
                    <>
                      {scannerStrategies.length > 0 && (
                        <>
                          {pipelineStrategies.length > 0 && (
                            <p className="px-2.5 pt-1.5 pb-1 text-[9px] uppercase tracking-wider text-muted-foreground/60 font-medium">
                              Scanner Detectors ({scannerStrategies.length})
                            </p>
                          )}
                          {scannerStrategies.map((strategy) => {
                            const active = selectedStrategyId === strategy.id
                            const sColor = STATUS_COLORS[strategy.status] || STATUS_COLORS.draft
                            return (
                              <button
                                key={strategy.id}
                                type="button"
                                onClick={() => {
                                  setStrategyDraftToken(null)
                                  setSelectedStrategyId(strategy.id)
                                }}
                                className={cn(
                                  'w-full rounded-md px-2.5 py-2 text-left transition-all duration-150',
                                  active ? 'bg-amber-500/10 ring-1 ring-amber-500/30' : 'hover:bg-muted/50'
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
                        </>
                      )}
                      {/* Pipeline strategies */}
                      {pipelineStrategies.length > 0 && (
                        <>
                          <div className="px-2.5 pt-3 pb-1 flex items-center gap-1.5">
                            <div className="flex-1 h-px bg-border/50" />
                            <p className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-medium shrink-0">
                              Pipeline Detectors ({pipelineStrategies.length})
                            </p>
                            <div className="flex-1 h-px bg-border/50" />
                          </div>
                          {pipelineStrategies.map((strategy) => {
                            const active = selectedStrategyId === strategy.id
                            const sColor = STATUS_COLORS[strategy.status] || STATUS_COLORS.draft
                            const tabLabel = SOURCE_LABELS[strategy.source_key] || strategy.source_key
                            return (
                              <button
                                key={strategy.id}
                                type="button"
                                onClick={() => {
                                  setStrategyDraftToken(null)
                                  setSelectedStrategyId(strategy.id)
                                }}
                                className={cn(
                                  'w-full rounded-md px-2.5 py-2 text-left transition-all duration-150',
                                  active ? 'bg-amber-500/10 ring-1 ring-amber-500/30' : 'hover:bg-muted/50'
                                )}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <div className="flex items-center gap-1.5 min-w-0">
                                    <Link2 className="w-3 h-3 text-muted-foreground shrink-0" />
                                    <p className="text-xs font-medium truncate" title={strategy.name}>
                                      {strategy.name}
                                    </p>
                                  </div>
                                  <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0 h-4 border', sColor)}>
                                    {strategy.status}
                                  </Badge>
                                </div>
                                <p className="text-[10px] text-muted-foreground mt-1 truncate pl-[18px]">
                                  → {tabLabel}
                                </p>
                                {strategy.capabilities && (
                                  <div className="pl-[18px]">
                                    <CapabilityBadges capabilities={strategy.capabilities} />
                                  </div>
                                )}
                              </button>
                            )
                          })}
                        </>
                      )}
                    </>
                  )
                })()}
              </>
            )}
          </div>
        </ScrollArea>

        {/* Footer stats */}
        <div className="shrink-0 px-3 py-2 border-t border-border/50 text-[10px] text-muted-foreground flex justify-between">
          <span>{filteredStrategies.length} strategies</span>
          <span>{strategyCatalog.filter((s) => s.enabled).length} enabled</span>
        </div>
      </div>

      {/* ── Right panel: Editor ── */}
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
                  <Code2 className="w-4 h-4 text-amber-400 shrink-0" />
                  <span className="text-sm font-medium truncate">
                    {strategyEditorName || 'Untitled Strategy'}
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
                    checked={strategyEditorEnabled}
                    onCheckedChange={setStrategyEditorEnabled}
                    className="scale-75"
                  />
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => setShowBacktest(true)}
                  disabled={!strategyEditorCode.trim()}
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
                  onClick={() => validateStrategyMutation.mutate()}
                  disabled={managerBusy || !strategyEditorCode.trim()}
                >
                  {validateStrategyMutation.isPending ? (
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
                  onClick={() => reloadStrategyMutation.mutate()}
                  disabled={managerBusy || !selectedStrategy}
                >
                  {reloadStrategyMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Zap className="w-3 h-3" />
                  )}
                  Reload
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px] bg-amber-600 hover:bg-amber-500 text-white"
                  onClick={() => saveStrategyMutation.mutate()}
                  disabled={
                    managerBusy ||
                    !strategyEditorCode.trim() ||
                    !strategyEditorName.trim() ||
                    !strategyEditorSlug.trim()
                  }
                >
                  {saveStrategyMutation.isPending ? (
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
                      const isPipeline = PIPELINE_SOURCES.has(selectedStrategy.source_key) && selectedStrategy.is_system
                      const pipelineLabel = SOURCE_LABELS[selectedStrategy.source_key] || selectedStrategy.source_key
                      const message = isPipeline
                        ? `⚠️ "${selectedStrategy.name}" is a pipeline strategy that powers the ${pipelineLabel}.\n\nDeleting it will tombstone it permanently and break that tab's functionality.\n\nAre you sure?`
                        : selectedStrategy.is_system
                          ? `Delete system strategy "${selectedStrategy.name}"? It will be tombstoned and will NOT auto-reseed.`
                          : `Delete "${selectedStrategy.name}"? This cannot be undone.`
                      const confirmed = window.confirm(message)
                      if (confirmed) deleteStrategyMutation.mutate()
                    }}
                    disabled={managerBusy}
                    title={
                      PIPELINE_SOURCES.has(selectedStrategy.source_key) && selectedStrategy.is_system
                        ? `Pipeline detector — powers the ${SOURCE_LABELS[selectedStrategy.source_key] || selectedStrategy.source_key}`
                        : 'Delete detector'
                    }
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                )}
              </div>
            </div>

            {/* ── Validation / Error banners ── */}
            {strategyValidation && (
              <div
                className={cn(
                  'shrink-0 px-4 py-2 text-xs flex items-start gap-2 border-b',
                  strategyValidation.valid
                    ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
                    : 'bg-red-500/5 border-red-500/20 text-red-400'
                )}
              >
                {strategyValidation.valid ? (
                  <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                ) : (
                  <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                )}
                <div className="min-w-0">
                  <p className="font-medium">
                    {strategyValidation.valid ? 'Validation passed' : 'Validation failed'}
                    {strategyValidation.class_name && (
                      <span className="font-mono ml-2 opacity-70">{strategyValidation.class_name}</span>
                    )}
                  </p>
                  {strategyValidation.errors.map((error, i) => (
                    <p key={`e-${i}`} className="font-mono text-[11px] mt-0.5">{error}</p>
                  ))}
                  {strategyValidation.warnings.map((warning, i) => (
                    <p key={`w-${i}`} className="font-mono text-[11px] mt-0.5 text-amber-400">{warning}</p>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setStrategyValidation(null)}
                  className="shrink-0 p-0.5 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {strategyEditorError && (
              <div className="shrink-0 px-4 py-2 text-xs flex items-start gap-2 bg-red-500/5 border-b border-red-500/20 text-red-400">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <p className="min-w-0 flex-1">{strategyEditorError}</p>
                <button
                  type="button"
                  onClick={() => setStrategyEditorError(null)}
                  className="shrink-0 p-0.5 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {/* ── Main editor content ── */}
            <div className="flex-1 min-h-0 flex flex-col">
              {/* Collapsible metadata section */}
              <div className="shrink-0 border-b border-border/50">
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
                  <span>Strategy Settings</span>
                  <span className="ml-auto font-mono text-[10px] opacity-60">
                    {strategyEditorSlug || 'no-key'}
                  </span>
                </button>

                {showConfig && (
                  <div className="px-4 pb-3 space-y-3 animate-in fade-in duration-200">
                    <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Strategy Key</Label>
                        <Input
                          value={strategyEditorSlug}
                          onChange={(event) => setStrategyEditorSlug(normalizeSlug(event.target.value))}
                          className="mt-1 h-8 text-xs font-mono"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Name</Label>
                        <Input
                          value={strategyEditorName}
                          onChange={(event) => setStrategyEditorName(event.target.value)}
                          className="mt-1 h-8 text-xs"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source</Label>
                        <Select value={strategyEditorSourceKey} onValueChange={setStrategyEditorSourceKey}>
                          <SelectTrigger className="mt-1 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {sourceKeys.map((sourceKey) => (
                              <SelectItem key={sourceKey} value={sourceKey}>
                                {sourceKey}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Class Name</Label>
                        <Input
                          value={strategyValidation?.class_name || inferredClassName || selectedStrategy?.class_name || 'auto-detected'}
                          className="mt-1 h-8 text-xs font-mono"
                          disabled
                        />
                      </div>
                    </div>
                    <div>
                      <Label className="text-[11px] text-muted-foreground">Description</Label>
                      <Input
                        value={strategyEditorDescription}
                        onChange={(event) => setStrategyEditorDescription(event.target.value)}
                        className="mt-1 h-8 text-xs"
                        placeholder="Describe what this strategy detects..."
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Code editor - takes remaining space */}
              <div className="flex-1 min-h-0 flex flex-col">
                <div className="px-4 py-2 flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-3.5 h-3.5 text-amber-400" />
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
                    value={strategyEditorCode}
                    onChange={setStrategyEditorCode}
                    language="python"
                    className="h-full"
                    minHeight="100%"
                    placeholder="Write your strategy source code here..."
                  />
                </div>
              </div>

              {/* Collapsible config JSON section */}
              <div className="shrink-0 border-t border-border/50">
                <button
                  type="button"
                  onClick={() => setShowDocs((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showDocs ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Runtime Config</span>
                  {(selectedStrategy?.config_schema?.param_fields?.length ?? 0) > 0 && (
                    <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 ml-1">
                      {selectedStrategy!.config_schema!.param_fields.length} fields
                    </Badge>
                  )}
                </button>
                {showDocs && (
                  <div className="px-3 pb-3 animate-in fade-in duration-200 space-y-3">
                    {/* Dynamic config form when schema is available */}
                    {(selectedStrategy?.config_schema?.param_fields?.length ?? 0) > 0 && !showRawJson && (
                      <>
                        <StrategyConfigForm
                          schema={selectedStrategy!.config_schema!}
                          values={(() => {
                            try {
                              return JSON.parse(strategyEditorConfigJson || '{}')
                            } catch {
                              return {}
                            }
                          })()}
                          onChange={(vals) => setStrategyEditorConfigJson(JSON.stringify(vals, null, 2))}
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
                    {/* Raw JSON editor — shown always when no schema, or when toggled */}
                    {(!(selectedStrategy?.config_schema?.param_fields?.length) || showRawJson) && (
                      <>
                        <CodeEditor
                          value={strategyEditorConfigJson}
                          onChange={setStrategyEditorConfigJson}
                          language="json"
                          minHeight="120px"
                          placeholder="{}"
                        />
                        {(selectedStrategy?.config_schema?.param_fields?.length ?? 0) > 0 && (
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

      <StrategyApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} variant="opportunity" />
      <StrategyBacktestFlyout
        open={showBacktest}
        onOpenChange={setShowBacktest}
        sourceCode={strategyEditorCode}
        slug={strategyEditorSlug || '_backtest_preview'}
        config={(() => { try { return JSON.parse(strategyEditorConfigJson || '{}') } catch { return {} } })()}
        variant="opportunity"
      />
    </div>
  )
}
