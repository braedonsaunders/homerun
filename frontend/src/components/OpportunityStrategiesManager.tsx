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
  createOpportunityStrategy,
  deleteOpportunityStrategy,
  getOpportunityStrategies,
  getPluginDocs,
  getPluginTemplate,
  reloadOpportunityStrategy,
  updateOpportunityStrategy,
  validateOpportunityStrategy,
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

export default function OpportunityStrategiesManager() {
  const queryClient = useQueryClient()
  const [showDocs, setShowDocs] = useState(false)
  const [strategyFilterSource, setStrategyFilterSource] = useState<string>('all')
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

  const docsQuery = useQuery({
    queryKey: ['plugin-docs'],
    queryFn: getPluginDocs,
    staleTime: Infinity,
  })

  const templateQuery = useQuery({
    queryKey: ['plugin-template'],
    queryFn: getPluginTemplate,
    staleTime: Infinity,
  })

  const strategyCatalog = strategiesQuery.data || []

  const sourceKeys = useMemo(() => {
    const fromCatalog = strategyCatalog.map((strategy) => String(strategy.source_key || '').toLowerCase())
    const merged = uniqueStrings(fromCatalog)
    return merged.length > 0 ? merged : ['scanner']
  }, [strategyCatalog])

  const filteredStrategies = useMemo(() => {
    const rows = [...strategyCatalog]
    if (strategyFilterSource === 'all') return rows
    return rows.filter((item) => item.source_key === strategyFilterSource)
  }, [strategyCatalog, strategyFilterSource])

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
      '    name = "Custom Opportunity Strategy"',
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
    setStrategyEditorName('Custom Opportunity Strategy')
    setStrategyEditorDescription('')
    setStrategyEditorEnabled(true)
    setStrategyEditorCode(templateQuery.data?.template || fallbackTemplate)
    setStrategyEditorConfigJson('{}')
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
          <span>Opportunity Strategy API Reference</span>
          <span className="text-[11px] text-muted-foreground">{showDocs ? 'Hide' : 'Show'}</span>
        </button>
        {showDocs ? (
          <div className="mt-2 space-y-2 text-[11px] text-muted-foreground">
            <p>{docsQuery.data?.overview?.description || 'Strategies implement detect(events, markets, prices).'}</p>
            <p className="font-mono text-[10px] rounded border border-border/70 bg-muted/40 px-2 py-1">
              {docsQuery.data?.detect_method?.signature || 'def detect(self, events, markets, prices) -> list[ArbitrageOpportunity]'}
            </p>
          </div>
        ) : null}
      </div>

      <div className="h-full min-h-0 grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-3">
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
              disabled={managerBusy}
            >
              New Strategy
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px]"
              onClick={() => cloneStrategyMutation.mutate()}
              disabled={managerBusy || !selectedStrategy}
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
                      onClick={() => {
                        setStrategyDraftToken(null)
                        setSelectedStrategyId(strategy.id)
                      }}
                      className={cn(
                        'w-full rounded-md border px-2 py-1.5 text-left transition-colors',
                        active ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-border/70 hover:bg-muted/40'
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs font-medium truncate" title={strategy.name}>
                          {strategy.name}
                        </p>
                        <Badge variant={strategy.enabled ? 'default' : 'secondary'} className="text-[10px]">
                          {strategy.enabled ? 'Enabled' : 'Disabled'}
                        </Badge>
                      </div>
                      <p className="text-[10px] font-mono text-muted-foreground mt-1 truncate" title={strategy.slug}>
                        {strategy.slug}
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
                  <Label>Strategy Key (slug)</Label>
                  <Input
                    value={strategyEditorSlug}
                    onChange={(event) => setStrategyEditorSlug(normalizeSlug(event.target.value))}
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
                  <Label>Name</Label>
                  <Input
                    value={strategyEditorName}
                    onChange={(event) => setStrategyEditorName(event.target.value)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label>Class Name</Label>
                  <Input
                    value={strategyValidation?.class_name || inferredClassName || selectedStrategy?.class_name || 'derived from source'}
                    className="mt-1 font-mono"
                    disabled
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
                <div className="sm:col-span-2 rounded-md border border-border/70 p-2 flex items-center justify-between">
                  <div>
                    <p className="text-xs font-medium">Enabled</p>
                    <p className="text-[11px] text-muted-foreground">
                      Disabled strategies are excluded from runtime load and opportunities filtering.
                    </p>
                  </div>
                  <Switch checked={strategyEditorEnabled} onCheckedChange={setStrategyEditorEnabled} />
                </div>
              </div>

              <div className="rounded-md border border-border/70 p-2">
                <Label>Source Code</Label>
                <textarea
                  className="mt-1 w-full min-h-[320px] rounded-md border bg-background p-2 text-xs font-mono"
                  value={strategyEditorCode}
                  onChange={(event) => setStrategyEditorCode(event.target.value)}
                />
              </div>

              <div className="rounded-md border border-border/70 p-2">
                <Label>Runtime Config JSON</Label>
                <textarea
                  className="mt-1 w-full min-h-[180px] rounded-md border bg-background p-2 text-xs font-mono"
                  value={strategyEditorConfigJson}
                  onChange={(event) => setStrategyEditorConfigJson(event.target.value)}
                />
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
                disabled={managerBusy || !strategyEditorCode.trim()}
              >
                Validate
              </Button>
              <Button
                type="button"
                variant="outline"
                className="h-8 text-xs"
                onClick={() => reloadStrategyMutation.mutate()}
                disabled={managerBusy || !selectedStrategy}
              >
                Reload Runtime
              </Button>
              <Button
                type="button"
                variant="outline"
                className="h-8 text-xs border-red-500/40 text-red-500 hover:bg-red-500/10"
                onClick={() => {
                  if (!selectedStrategy) return
                  const confirmed = window.confirm(
                    selectedStrategy.is_system
                      ? `Delete system strategy "${selectedStrategy.name}" (${selectedStrategy.slug})? It will be tombstoned and will NOT auto-reseed. This cannot be undone from the UI.`
                      : `Delete strategy "${selectedStrategy.name}" (${selectedStrategy.slug})? This cannot be undone.`
                  )
                  if (!confirmed) return
                  deleteStrategyMutation.mutate()
                }}
                disabled={managerBusy || !selectedStrategy}
              >
                Delete
              </Button>
              <Button
                type="button"
                variant="outline"
                className="h-8 text-xs"
                onClick={() => refreshCatalog()}
                disabled={managerBusy}
              >
                Refresh Catalog
              </Button>
            </div>
            <Button
              type="button"
              className="h-8 text-xs"
              onClick={() => saveStrategyMutation.mutate()}
              disabled={
                managerBusy ||
                !strategyEditorCode.trim() ||
                !strategyEditorName.trim() ||
                !strategyEditorSlug.trim()
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
