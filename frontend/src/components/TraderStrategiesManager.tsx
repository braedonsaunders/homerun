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
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Settings2,
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
  cloneTraderStrategy,
  createTraderStrategy,
  getTraderConfigSchema,
  getTraderStrategies,
  reloadTraderStrategy,
  updateTraderStrategy,
  validateTraderStrategy,
} from '../services/api'
import StrategyApiDocsFlyout from './StrategyApiDocsFlyout'

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
  active: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  error: 'bg-red-500/15 text-red-400 border-red-500/30',
  unloaded: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  draft: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
}

export default function TraderStrategiesManager() {
  const queryClient = useQueryClient()
  const [showDocs, setShowDocs] = useState(false)
  const [showParams, setShowParams] = useState(false)
  const [showApiDocs, setShowApiDocs] = useState(false)
  const [strategyFilterSource, setStrategyFilterSource] = useState<string>('all')
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null)
  const [strategyEditorCode, setStrategyEditorCode] = useState('')
  const [strategyEditorClassName, setStrategyEditorClassName] = useState('')
  const [strategyEditorLabel, setStrategyEditorLabel] = useState('')
  const [strategyEditorDescription, setStrategyEditorDescription] = useState('')
  const [strategyEditorSourceKey, setStrategyEditorSourceKey] = useState('crypto')
  const [strategyEditorEnabled, setStrategyEditorEnabled] = useState(true)
  const [strategyEditorParamsJson, setStrategyEditorParamsJson] = useState('{}')
  const [strategyEditorSchemaJson, setStrategyEditorSchemaJson] = useState('{}')
  const [strategyEditorAliasesCsv, setStrategyEditorAliasesCsv] = useState('')
  const [strategyDraftKey, setStrategyDraftKey] = useState('')
  const [strategyEditorError, setStrategyEditorError] = useState<string | null>(null)
  const [showRawParams, setShowRawParams] = useState(false)
  const [strategyValidation, setStrategyValidation] = useState<{
    valid: boolean
    errors: string[]
    warnings: string[]
  } | null>(null)

  const traderConfigSchemaQuery = useQuery({
    queryKey: ['trader-config-schema'],
    queryFn: getTraderConfigSchema,
    staleTime: 300000,
  })

  const traderStrategiesQuery = useQuery({
    queryKey: ['trader-strategies-catalog'],
    queryFn: () => getTraderStrategies(),
    staleTime: 15000,
    refetchInterval: 15000,
  })

  const strategyCatalog = traderStrategiesQuery.data || []

  const selectedStrategy = useMemo(
    () => strategyCatalog.find((item) => item.id === selectedStrategyId) || null,
    [selectedStrategyId, strategyCatalog]
  )

  const sourceKeys = useMemo(() => {
    const fromSchema = (traderConfigSchemaQuery.data?.sources || []).map((source: any) => String(source.key || '').toLowerCase())
    const fromCatalog = strategyCatalog.map((strategy) => String(strategy.source_key || '').toLowerCase())
    const merged = uniqueStrings([...fromSchema, ...fromCatalog])
    return merged.length > 0 ? merged : ['crypto']
  }, [strategyCatalog, traderConfigSchemaQuery.data?.sources])

  const filteredStrategies = useMemo(() => {
    const rows = [...strategyCatalog]
    if (strategyFilterSource === 'all') return rows
    return rows.filter((item) => item.source_key === strategyFilterSource)
  }, [strategyCatalog, strategyFilterSource])

  useEffect(() => {
    if (selectedStrategyId) return
    if (strategyDraftKey) return
    if (strategyCatalog.length > 0) {
      setSelectedStrategyId(strategyCatalog[0].id)
    }
  }, [selectedStrategyId, strategyCatalog, strategyDraftKey])

  useEffect(() => {
    if (!selectedStrategyId) return
    const strategy = strategyCatalog.find((item) => item.id === selectedStrategyId)
    if (!strategy) return
    setStrategyEditorCode(strategy.source_code || '')
    setStrategyEditorClassName(strategy.class_name || '')
    setStrategyEditorLabel(strategy.label || '')
    setStrategyEditorDescription(strategy.description || '')
    setStrategyEditorSourceKey(strategy.source_key || 'crypto')
    setStrategyEditorEnabled(Boolean(strategy.enabled))
    setStrategyEditorParamsJson(JSON.stringify(strategy.default_params_json || {}, null, 2))
    setStrategyEditorSchemaJson(JSON.stringify(strategy.param_schema_json || {}, null, 2))
    setStrategyEditorAliasesCsv((strategy.aliases_json || []).join(', '))
    setStrategyDraftKey(strategy.strategy_key || '')
    setStrategyEditorError(null)
    setStrategyValidation(null)
  }, [selectedStrategyId, strategyCatalog])

  const refreshStrategyCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['trader-strategies-catalog'] })
    queryClient.invalidateQueries({ queryKey: ['trader-config-schema'] })
    queryClient.invalidateQueries({ queryKey: ['trader-sources'] })
  }

  const saveStrategyMutation = useMutation({
    mutationFn: async () => {
      const parsedParams = parseJsonObject(strategyEditorParamsJson || '{}')
      if (!parsedParams.value) {
        throw new Error(`Default params JSON error: ${parsedParams.error || 'invalid object'}`)
      }
      const parsedSchema = parseJsonObject(strategyEditorSchemaJson || '{}')
      if (!parsedSchema.value) {
        throw new Error(`Param schema JSON error: ${parsedSchema.error || 'invalid object'}`)
      }

      const payload = {
        strategy_key: String(strategyDraftKey || '').trim().toLowerCase(),
        source_key: String(strategyEditorSourceKey || '').trim().toLowerCase(),
        label: String(strategyEditorLabel || '').trim(),
        description: strategyEditorDescription.trim() || null,
        class_name: String(strategyEditorClassName || '').trim(),
        source_code: strategyEditorCode,
        default_params_json: parsedParams.value,
        param_schema_json: parsedSchema.value,
        aliases_json: uniqueStrings(
          String(strategyEditorAliasesCsv || '')
            .split(',')
            .map((item) => item.trim())
        ),
        enabled: strategyEditorEnabled,
      }

      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (selected) {
        return updateTraderStrategy(selected.id, {
          ...payload,
          unlock_system: Boolean(selected.is_system),
        })
      }
      if (!payload.strategy_key) {
        throw new Error('strategy_key is required for new strategy')
      }
      return createTraderStrategy(payload)
    },
    onSuccess: (strategy) => {
      setStrategyEditorError(null)
      setSelectedStrategyId(strategy.id)
      refreshStrategyCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Failed to save strategy'))
    },
  })

  const validateStrategyMutation = useMutation({
    mutationFn: async () => {
      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to validate')
      return validateTraderStrategy(selected.id, {
        source_code: strategyEditorCode,
        class_name: strategyEditorClassName,
      })
    },
    onSuccess: (result) => {
      setStrategyValidation({
        valid: Boolean(result.valid),
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
      return reloadTraderStrategy(selected.id)
    },
    onSuccess: () => {
      setStrategyEditorError(null)
      refreshStrategyCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Reload failed'))
    },
  })

  const cloneStrategyMutation = useMutation({
    mutationFn: async () => {
      const selected = strategyCatalog.find((item) => item.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to clone')
      return cloneTraderStrategy(selected.id, {
        strategy_key: `${selected.strategy_key}_clone_${Date.now().toString().slice(-6)}`,
        label: `${selected.label} (Clone)`,
        enabled: true,
      })
    },
    onSuccess: (strategy) => {
      setStrategyEditorError(null)
      setSelectedStrategyId(strategy.id)
      refreshStrategyCatalog()
    },
    onError: (error: unknown) => {
      setStrategyEditorError(errorMessage(error, 'Clone failed'))
    },
  })

  const strategyManagerBusy =
    saveStrategyMutation.isPending ||
    validateStrategyMutation.isPending ||
    reloadStrategyMutation.isPending ||
    cloneStrategyMutation.isPending

  const startNewStrategyDraft = () => {
    setSelectedStrategyId(null)
    setStrategyDraftKey(`custom_${Date.now().toString().slice(-6)}`)
    setStrategyEditorSourceKey('crypto')
    setStrategyEditorLabel('Custom Strategy')
    setStrategyEditorDescription('')
    setStrategyEditorClassName('CustomTraderStrategy')
    setStrategyEditorCode(
      [
        'from services.trader_orchestrator.strategies.base import BaseTraderStrategy, StrategyDecision, DecisionCheck',
        '',
        'class CustomTraderStrategy(BaseTraderStrategy):',
        '    key = "custom_strategy"',
        '',
        '    def evaluate(self, signal, context):',
        '        checks = [',
        '            DecisionCheck("example", "Example check", True, detail="replace with your rules"),',
        '        ]',
        '        return StrategyDecision(decision="skipped", reason="Template strategy", score=0.0, checks=checks, payload={})',
        '',
      ].join('\n')
    )
    setStrategyEditorEnabled(true)
    setStrategyEditorParamsJson('{}')
    setStrategyEditorSchemaJson('{"param_fields": []}')
    setStrategyEditorAliasesCsv('')
    setStrategyEditorError(null)
    setStrategyValidation(null)
  }

  const hasSelection = Boolean(selectedStrategy || strategyDraftKey)
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
              disabled={strategyManagerBusy}
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
              disabled={strategyManagerBusy || !selectedStrategy}
              title="Clone selected strategy"
            >
              <Copy className="w-3 h-3" />
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px]"
              onClick={() => refreshStrategyCatalog()}
              disabled={strategyManagerBusy}
              title="Refresh catalog"
            >
              <RefreshCw className={cn('w-3 h-3', traderStrategiesQuery.isFetching && 'animate-spin')} />
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

        {/* Strategy list */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-1.5 space-y-1">
            {filteredStrategies.length === 0 ? (
              <p className="px-3 py-6 text-xs text-muted-foreground text-center">No strategies found.</p>
            ) : (
              filteredStrategies.map((strategy) => {
                const active = selectedStrategyId === strategy.id
                const sColor = STATUS_COLORS[strategy.status] || STATUS_COLORS.draft
                return (
                  <button
                    key={strategy.id}
                    type="button"
                    onClick={() => setSelectedStrategyId(strategy.id)}
                    className={cn(
                      'w-full rounded-md px-2.5 py-2 text-left transition-all duration-150',
                      active
                        ? 'bg-cyan-500/10 ring-1 ring-cyan-500/30'
                        : 'hover:bg-muted/50'
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-medium truncate" title={strategy.label || strategy.strategy_key}>
                        {strategy.label || strategy.strategy_key}
                      </p>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {!strategy.enabled && (
                          <span className="w-1.5 h-1.5 rounded-full bg-zinc-500" title="Disabled" />
                        )}
                        <Badge
                          variant="outline"
                          className={cn('text-[9px] px-1.5 py-0 h-4 border', sColor)}
                        >
                          {strategy.status}
                        </Badge>
                      </div>
                    </div>
                    <p className="text-[10px] font-mono text-muted-foreground mt-1 truncate">
                      {strategy.strategy_key}
                    </p>
                  </button>
                )
              })
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
                  <Code2 className="w-4 h-4 text-cyan-400 shrink-0" />
                  <span className="text-sm font-medium truncate">
                    {strategyEditorLabel || 'Untitled Strategy'}
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
                  disabled={strategyManagerBusy || !selectedStrategy}
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
                  disabled={strategyManagerBusy || !selectedStrategy}
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
                  className="h-7 gap-1 px-2 text-[11px] bg-cyan-600 hover:bg-cyan-500 text-white"
                  onClick={() => saveStrategyMutation.mutate()}
                  disabled={
                    strategyManagerBusy ||
                    !strategyEditorCode.trim() ||
                    !strategyEditorClassName.trim() ||
                    !strategyEditorLabel.trim() ||
                    !strategyDraftKey.trim()
                  }
                >
                  {saveStrategyMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Save className="w-3 h-3" />
                  )}
                  Save
                </Button>
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
                  onClick={() => setShowDocs((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showDocs ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Strategy Settings</span>
                  <span className="ml-auto font-mono text-[10px] opacity-60">
                    {strategyDraftKey || 'no-key'}
                  </span>
                </button>

                {showDocs && (
                  <div className="px-4 pb-3 space-y-3 animate-in fade-in duration-200">
                    <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Strategy Key</Label>
                        <Input
                          value={strategyDraftKey}
                          onChange={(event) => setStrategyDraftKey(event.target.value)}
                          className="mt-1 h-8 text-xs font-mono"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Label</Label>
                        <Input
                          value={strategyEditorLabel}
                          onChange={(event) => setStrategyEditorLabel(event.target.value)}
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
                          value={strategyEditorClassName}
                          onChange={(event) => setStrategyEditorClassName(event.target.value)}
                          className="mt-1 h-8 text-xs font-mono"
                        />
                      </div>
                    </div>
                    <div className="grid gap-3 grid-cols-1 xl:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Description</Label>
                        <Input
                          value={strategyEditorDescription}
                          onChange={(event) => setStrategyEditorDescription(event.target.value)}
                          className="mt-1 h-8 text-xs"
                          placeholder="Describe what this strategy evaluates..."
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Aliases (comma separated)</Label>
                        <Input
                          value={strategyEditorAliasesCsv}
                          onChange={(event) => setStrategyEditorAliasesCsv(event.target.value)}
                          className="mt-1 h-8 text-xs font-mono"
                          placeholder="alias_one, alias_two"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Code editor - takes remaining space */}
              <div className="flex-1 min-h-0 flex flex-col">
                <div className="px-4 py-2 flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-xs font-medium">Source Code</span>
                    <span className="text-[10px] text-muted-foreground font-mono">Python</span>
                  </div>
                  {strategyEditorClassName && (
                    <span className="text-[10px] font-mono text-muted-foreground">
                      class {strategyEditorClassName}
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
                    placeholder="Write your trader strategy source code here..."
                  />
                </div>
              </div>

              {/* Collapsible params/schema JSON section */}
              <div className="shrink-0 border-t border-border/50">
                <button
                  type="button"
                  onClick={() => setShowParams((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showParams ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Parameters &amp; Schema</span>
                  {(() => {
                    try {
                      const schema = JSON.parse(strategyEditorSchemaJson || '{}')
                      const count = schema?.param_fields?.length || 0
                      if (count > 0) return (
                        <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 ml-1">
                          {count} fields
                        </Badge>
                      )
                    } catch { /* ignore */ }
                    return null
                  })()}
                </button>
                {showParams && (
                  <div className="px-3 pb-3 space-y-3 animate-in fade-in duration-200">
                    {/* Dynamic config form when schema has param_fields */}
                    {(() => {
                      try {
                        const schema = JSON.parse(strategyEditorSchemaJson || '{}')
                        if (schema?.param_fields?.length > 0 && !showRawParams) {
                          const paramValues = (() => {
                            try { return JSON.parse(strategyEditorParamsJson || '{}') } catch { return {} }
                          })()
                          return (
                            <>
                              <StrategyConfigForm
                                schema={schema}
                                values={paramValues}
                                onChange={(vals) => setStrategyEditorParamsJson(JSON.stringify(vals, null, 2))}
                              />
                              <button
                                type="button"
                                onClick={() => setShowRawParams(true)}
                                className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                              >
                                Show Raw JSON
                              </button>
                            </>
                          )
                        }
                      } catch { /* ignore */ }
                      return null
                    })()}
                    {/* Raw JSON editors — shown when no schema or toggled */}
                    {(() => {
                      let hasSchema = false
                      try {
                        const schema = JSON.parse(strategyEditorSchemaJson || '{}')
                        hasSchema = (schema?.param_fields?.length || 0) > 0
                      } catch { /* ignore */ }
                      if (hasSchema && !showRawParams) return null
                      return (
                        <>
                          <div className="grid gap-3 grid-cols-1 lg:grid-cols-2">
                            <div>
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-[11px] font-medium text-muted-foreground">Default Params</span>
                                <span className="text-[10px] text-muted-foreground font-mono">JSON</span>
                              </div>
                              <CodeEditor
                                value={strategyEditorParamsJson}
                                onChange={setStrategyEditorParamsJson}
                                language="json"
                                minHeight="140px"
                                placeholder="{}"
                              />
                            </div>
                            <div>
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-[11px] font-medium text-muted-foreground">Param Schema</span>
                                <span className="text-[10px] text-muted-foreground font-mono">JSON</span>
                              </div>
                              <CodeEditor
                                value={strategyEditorSchemaJson}
                                onChange={setStrategyEditorSchemaJson}
                                language="json"
                                minHeight="140px"
                                placeholder='{"param_fields": []}'
                              />
                            </div>
                          </div>
                          {hasSchema && (
                            <button
                              type="button"
                              onClick={() => setShowRawParams(false)}
                              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                            >
                              Show Config Form
                            </button>
                          )}
                        </>
                      )
                    })()}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <StrategyApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} variant="trader" />
    </div>
  )
}
