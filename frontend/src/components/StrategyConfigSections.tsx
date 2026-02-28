import { useCallback, useEffect, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronRight,
  Code2,
  CheckCircle,
  AlertCircle,
  Save,
  Loader2,
  ExternalLink,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Switch } from './ui/switch'
import StrategyConfigForm from './StrategyConfigForm'
import {
  getPlugins,
  updatePlugin,
  type StrategyPlugin,
} from '../services/api'

/**
 * Renders collapsible config sections for each opportunity strategy
 * matching the given `sourceKey`. Each strategy with a `config_schema`
 * containing `param_fields` gets its own collapsible card with dynamic
 * form controls rendered via StrategyConfigForm.
 *
 * This is designed to be embedded in any settings flyout — Weather, News,
 * Crypto, Traders, Markets (scanner). Coders only need to define
 * `config_schema` in their Python strategy seed to get controls here.
 */
export default function StrategyConfigSections({
  sourceKey,
  enabled,
}: {
  sourceKey: string
  enabled?: boolean
}) {
  const normalizedSourceKey = String(sourceKey || '').trim().toLowerCase()
  const queryClient = useQueryClient()

  const { data: allPlugins } = useQuery({
    queryKey: ['plugins'],
    queryFn: getPlugins,
    enabled: enabled !== false,
  })

  // Filter to strategies matching this subtab's source_key that have config schemas
  const strategies = (allPlugins || []).filter(
    (p) =>
      normalizeSourceKey(p.source_key) === normalizedSourceKey &&
      p.config_schema &&
      p.config_schema.param_fields &&
      p.config_schema.param_fields.length > 0
  )

  if (strategies.length === 0) return null

  return (
    <>
      {strategies.map((strategy) => (
        <StrategyConfigCard
          key={strategy.id}
          strategy={strategy}
          queryClient={queryClient}
        />
      ))}
    </>
  )
}

function StrategyConfigCard({
  strategy,
  queryClient,
}: {
  strategy: StrategyPlugin
  queryClient: ReturnType<typeof useQueryClient>
}) {
  const [open, setOpen] = useState(false)
  const [localConfig, setLocalConfig] = useState<Record<string, unknown>>({})
  const [localEnabled, setLocalEnabled] = useState(Boolean(strategy.enabled))
  const [dirty, setDirty] = useState(false)
  const [saveMsg, setSaveMsg] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  // Sync local config from strategy data
  useEffect(() => {
    const cfg = { ...(strategy.config || {}) }
    // Strip internal _schema key
    delete cfg._schema
    setLocalConfig(cfg)
    setLocalEnabled(Boolean(strategy.enabled))
    setDirty(false)
  }, [strategy.config, strategy.enabled])

  const handleChange = useCallback((next: Record<string, unknown>) => {
    setLocalConfig(next)
    setDirty(true)
  }, [])

  const saveMutation = useMutation({
    mutationFn: ({
      config,
      enabled,
    }: {
      config: Record<string, unknown>
      enabled: boolean
    }) =>
      updatePlugin(strategy.id, {
        config,
        enabled,
        unlock_system: Boolean(strategy.is_system),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
      setSaveMsg({ type: 'success', text: 'Saved' })
      setDirty(false)
      setTimeout(() => setSaveMsg(null), 2000)
    },
    onError: (error: unknown) => {
      const message =
        error && typeof error === 'object' && 'message' in error
          ? String((error as { message?: string }).message || 'Save failed')
          : 'Save failed'
      setSaveMsg({ type: 'error', text: message })
      setTimeout(() => setSaveMsg(null), 4000)
    },
  })

  const handleSave = () => {
    saveMutation.mutate({
      config: localConfig,
      enabled: localEnabled,
    })
  }

  const fieldCount = strategy.config_schema?.param_fields?.length ?? 0

  return (
    <Card className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <Code2 className="w-3.5 h-3.5 text-violet-400" />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">
            {strategy.name}
          </h4>
          <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/60 text-muted-foreground">
            {fieldCount} params
          </span>
          {dirty && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
              unsaved
            </span>
          )}
          {saveMsg && (
            <span
              className={cn(
                'text-[9px] px-1.5 py-0.5 rounded-full flex items-center gap-1',
                saveMsg.type === 'success'
                  ? 'bg-emerald-500/10 text-emerald-400'
                  : 'bg-red-500/10 text-red-400'
              )}
            >
              {saveMsg.type === 'success' ? (
                <CheckCircle className="w-2.5 h-2.5" />
              ) : (
                <AlertCircle className="w-2.5 h-2.5" />
              )}
              {saveMsg.text}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {!localEnabled && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-red-500/10 text-red-400">
              disabled
            </span>
          )}
          <Switch
            checked={localEnabled}
            onClick={(e) => e.stopPropagation()}
            onCheckedChange={(next) => {
              setLocalEnabled(next)
              saveMutation.mutate({
                config: localConfig,
                enabled: next,
              })
            }}
            disabled={saveMutation.isPending}
            className="scale-75"
          />
          {open ? (
            <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
          )}
        </div>
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2.5">
          {strategy.description && (
            <p className="text-[10px] text-muted-foreground/60 leading-tight">
              {strategy.description}
            </p>
          )}

          <StrategyConfigForm
            schema={strategy.config_schema!}
            values={localConfig}
            onChange={handleChange}
          />

          <div className="flex items-center justify-between pt-1">
            <Button
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                window.dispatchEvent(
                  new CustomEvent('navigate-to-tab', { detail: 'strategies' })
                )
                window.dispatchEvent(
                  new CustomEvent('navigate-strategies-subtab', {
                    detail: {
                      subtab: 'opportunity',
                      sourceFilter: strategy.source_key,
                      source: strategy.source_key,
                    },
                  })
                )
              }}
              className="gap-1 text-[10px] h-6"
            >
              <ExternalLink className="w-2.5 h-2.5" />
              Edit Code
            </Button>

            <Button
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                handleSave()
              }}
              disabled={!dirty || saveMutation.isPending}
              className="gap-1 text-[10px] h-6 px-3 bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-40"
            >
              {saveMutation.isPending ? (
                <Loader2 className="w-2.5 h-2.5 animate-spin" />
              ) : (
                <Save className="w-2.5 h-2.5" />
              )}
              {saveMutation.isPending ? 'Saving...' : 'Save Config'}
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}

function normalizeSourceKey(sourceKey: string): string {
  const key = String(sourceKey || '').trim().toLowerCase()
  return key
}
