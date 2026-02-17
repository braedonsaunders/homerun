import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import {
  cloneTraderStrategy,
  createTraderStrategy,
  getTraderConfigSchema,
  getTraderStrategyDocs,
  getTraderStrategies,
  reloadTraderStrategy,
  updateTraderStrategy,
  validateTraderStrategy,
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

export default function TraderStrategiesManager() {
  const queryClient = useQueryClient()
  const [showDocs, setShowDocs] = useState(false)
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

  const traderDocsQuery = useQuery({
    queryKey: ['trader-strategy-docs'],
    queryFn: getTraderStrategyDocs,
    staleTime: Infinity,
  })

  const strategyCatalog = traderStrategiesQuery.data || []

  const selectedStrategy = useMemo(
    () => strategyCatalog.find((item) => item.id === selectedStrategyId) || null,
    [selectedStrategyId, strategyCatalog]
  )

  const sourceKeys = useMemo(() => {
    const fromSchema = (traderConfigSchemaQuery.data?.sources || []).map((source) => String(source.key || '').toLowerCase())
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

  return (
    <div className="h-full min-h-0 flex flex-col gap-3">
      <div className="rounded-lg border border-border/70 p-3">
        <button
          type="button"
          onClick={() => setShowDocs((prev) => !prev)}
          className="flex w-full items-center justify-between text-left text-xs font-medium"
        >
          <span>AutoTrader Strategy API Reference</span>
          <span className="text-[11px] text-muted-foreground">{showDocs ? 'Hide' : 'Show'}</span>
        </button>
        {showDocs ? (
          <div className="mt-2 space-y-2 text-[11px] text-muted-foreground">
            <p>{traderDocsQuery.data?.overview?.description || 'Strategies evaluate trade signals and return StrategyDecision.'}</p>
            <p className="font-mono text-[10px] rounded border border-border/70 bg-muted/40 px-2 py-1">
              {traderDocsQuery.data?.evaluate_method?.signature || 'def evaluate(self, signal, context) -> StrategyDecision'}
            </p>
          </div>
        ) : null}
      </div>

      <div className="min-h-0 grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-3">
      <div className="rounded-lg border border-border/70 p-3 min-h-0 flex flex-col gap-3">
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground">Source Filter</Label>
          <Select value={strategyFilterSource} onValueChange={setStrategyFilterSource}>
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources</SelectItem>
              {sourceKeys.map((sourceKey) => (
                <SelectItem key={sourceKey} value={sourceKey}>
                  {sourceKey}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="sm"
            className="h-7 px-2 text-[11px]"
            onClick={startNewStrategyDraft}
            disabled={strategyManagerBusy}
          >
            New Strategy
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7 px-2 text-[11px]"
            onClick={() => cloneStrategyMutation.mutate()}
            disabled={strategyManagerBusy || !selectedStrategy}
          >
            Clone
          </Button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>{filteredStrategies.length} strategies</span>
          {selectedStrategy ? (
            <span className="font-mono">
              v{selectedStrategy.version} • {selectedStrategy.status}
            </span>
          ) : (
            <span className="font-mono">new draft</span>
          )}
        </div>

        <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/70">
          <div className="space-y-1 p-1.5">
            {filteredStrategies.length === 0 ? (
              <p className="px-2 py-1 text-xs text-muted-foreground">No strategies for this source.</p>
            ) : (
              filteredStrategies.map((strategy) => {
                const active = selectedStrategyId === strategy.id
                return (
                  <button
                    key={strategy.id}
                    type="button"
                    onClick={() => setSelectedStrategyId(strategy.id)}
                    className={cn(
                      'w-full rounded-md border px-2 py-1.5 text-left transition-colors',
                      active ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-border/70 hover:bg-muted/40'
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-medium truncate" title={strategy.label || strategy.strategy_key}>
                        {strategy.label || strategy.strategy_key}
                      </p>
                      <Badge variant={strategy.enabled ? 'default' : 'secondary'} className="text-[10px]">
                        {strategy.enabled ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                    <p className="text-[10px] font-mono text-muted-foreground mt-1 truncate" title={strategy.strategy_key}>
                      {strategy.strategy_key}
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-0.5 truncate" title={strategy.description || ''}>
                      {strategy.description || 'No description'}
                    </p>
                  </button>
                )
              })
            )}
          </div>
        </ScrollArea>
      </div>

      <div className="rounded-lg border border-border/70 min-h-0 flex flex-col">
        <ScrollArea className="flex-1 min-h-0 px-4 py-3">
          <div className="space-y-3 pb-2">
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <Label>Strategy Key</Label>
                <Input
                  value={strategyDraftKey}
                  onChange={(event) => setStrategyDraftKey(event.target.value)}
                  className="mt-1 font-mono"
                />
              </div>
              <div>
                <Label>Source Key</Label>
                <Select value={strategyEditorSourceKey} onValueChange={setStrategyEditorSourceKey}>
                  <SelectTrigger className="mt-1">
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
                <Label>Label</Label>
                <Input
                  value={strategyEditorLabel}
                  onChange={(event) => setStrategyEditorLabel(event.target.value)}
                  className="mt-1"
                />
              </div>
              <div>
                <Label>Class Name</Label>
                <Input
                  value={strategyEditorClassName}
                  onChange={(event) => setStrategyEditorClassName(event.target.value)}
                  className="mt-1 font-mono"
                />
              </div>
              <div className="sm:col-span-2">
                <Label>Description</Label>
                <Input
                  value={strategyEditorDescription}
                  onChange={(event) => setStrategyEditorDescription(event.target.value)}
                  className="mt-1"
                />
              </div>
              <div className="sm:col-span-2">
                <Label>Aliases (comma separated)</Label>
                <Input
                  value={strategyEditorAliasesCsv}
                  onChange={(event) => setStrategyEditorAliasesCsv(event.target.value)}
                  className="mt-1 font-mono"
                  placeholder="alias_one, alias_two"
                />
              </div>
              <div className="sm:col-span-2 rounded-md border border-border/70 p-2 flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium">Enabled</p>
                  <p className="text-[11px] text-muted-foreground">
                    Disabled strategies are excluded from runtime load and source schema options.
                  </p>
                </div>
                <Switch checked={strategyEditorEnabled} onCheckedChange={setStrategyEditorEnabled} />
              </div>
            </div>

            <div className="rounded-md border border-border/70 p-2">
              <Label>Source Code</Label>
              <textarea
                className="mt-1 w-full min-h-[280px] rounded-md border bg-background p-2 text-xs font-mono"
                value={strategyEditorCode}
                onChange={(event) => setStrategyEditorCode(event.target.value)}
              />
            </div>

            <div className="grid gap-3 lg:grid-cols-2">
              <div className="rounded-md border border-border/70 p-2">
                <Label>Default Params JSON</Label>
                <textarea
                  className="mt-1 w-full min-h-[160px] rounded-md border bg-background p-2 text-xs font-mono"
                  value={strategyEditorParamsJson}
                  onChange={(event) => setStrategyEditorParamsJson(event.target.value)}
                />
              </div>
              <div className="rounded-md border border-border/70 p-2">
                <Label>Param Schema JSON</Label>
                <textarea
                  className="mt-1 w-full min-h-[160px] rounded-md border bg-background p-2 text-xs font-mono"
                  value={strategyEditorSchemaJson}
                  onChange={(event) => setStrategyEditorSchemaJson(event.target.value)}
                />
              </div>
            </div>

              <div className="rounded-md border border-border/70 p-2 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={selectedStrategy?.is_system ? 'secondary' : 'default'}>
                    {selectedStrategy?.is_system ? 'System' : 'Custom'}
                  </Badge>
                  <Badge variant="outline">status:{selectedStrategy?.status || 'draft'}</Badge>
                  <Badge variant="outline">version:{selectedStrategy?.version || 1}</Badge>
                </div>
              </div>

            {strategyValidation ? (
              <div className={cn(
                'rounded-md border px-3 py-2 text-xs',
                strategyValidation.valid
                  ? 'border-emerald-500/40 bg-emerald-500/10'
                  : 'border-red-500/40 bg-red-500/10'
              )}>
                <p className="font-semibold">{strategyValidation.valid ? 'Validation passed' : 'Validation failed'}</p>
                {strategyValidation.errors.length > 0 ? (
                  <div className="mt-1 space-y-1">
                    {strategyValidation.errors.map((error, index) => (
                      <p key={`${error}-${index}`} className="font-mono text-[11px]">{error}</p>
                    ))}
                  </div>
                ) : null}
                {strategyValidation.warnings.length > 0 ? (
                  <div className="mt-1 space-y-1">
                    {strategyValidation.warnings.map((warning, index) => (
                      <p key={`${warning}-${index}`} className="font-mono text-[11px]">{warning}</p>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}

            {strategyEditorError ? (
              <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-500">
                {strategyEditorError}
              </div>
            ) : null}
          </div>
        </ScrollArea>

        <div className="border-t border-border px-4 py-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant="outline"
              className="h-8 text-xs"
              onClick={() => validateStrategyMutation.mutate()}
              disabled={strategyManagerBusy || !selectedStrategy}
            >
              Validate
            </Button>
            <Button
              type="button"
              variant="outline"
              className="h-8 text-xs"
              onClick={() => reloadStrategyMutation.mutate()}
              disabled={strategyManagerBusy || !selectedStrategy}
            >
              Reload Runtime
            </Button>
            <Button
              type="button"
              variant="outline"
              className="h-8 text-xs"
              onClick={() => refreshStrategyCatalog()}
              disabled={strategyManagerBusy}
            >
              Refresh Catalog
            </Button>
          </div>
          <Button
            type="button"
            className="h-8 text-xs"
            onClick={() => saveStrategyMutation.mutate()}
              disabled={
                strategyManagerBusy ||
                !strategyEditorCode.trim() ||
                !strategyEditorClassName.trim() ||
                !strategyEditorLabel.trim() ||
                !strategyDraftKey.trim()
              }
            >
              Save Strategy
          </Button>
        </div>
      </div>
      </div>
    </div>
  )
}
