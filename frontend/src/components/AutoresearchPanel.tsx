/**
 * AutoresearchPanel — strategy-scoped Research subview.
 *
 * Lives in the Strategies tab next to "Strategies" and "Machine
 * Learning". Has two inner subtabs:
 *
 *   * **Code Experiments** — LLM-driven strategy code evolution
 *     against the backtest data plane. No bot involved. Hits the
 *     ``/autoresearch/strategy/{strategy_id}/*`` endpoints which key
 *     off ``strategy_id`` only and persist code versions on the
 *     Strategy record itself.
 *   * **Backtest Studio** — institutional-grade L2-replay workbench
 *     wired to the Cox PH fill model, ensemble PnL bands,
 *     counterfactual queue replay, and triangulation. Mounted via
 *     ``BacktestStudio``.
 *
 * Per-bot live parameter tuning lives in TradingPanel/Tune (separate
 * surface, separate concept).
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Check,
  Code2,
  Database,
  FlaskConical,
  Loader2,
  Play,
  Sparkles,
  Square,
  X,
} from 'lucide-react'

import { getTraderStrategies, getTraderStrategy } from '../services/apiTraders'
import { getLLMModels, type LLMModelOption } from '../services/apiSettings'
import {
  getAutoresearchSettings,
  getStrategyAutoresearchHistory,
  getStrategyAutoresearchStatus,
  stopStrategyAutoresearchExperiment,
  streamStrategyAutoresearchExperiment,
  updateAutoresearchSettings,
  type AutoresearchExperimentStatus,
  type AutoresearchIteration,
  type AutoresearchSettings,
} from '../services/apiIntelligence'

import BacktestStudio from './BacktestStudio'
import DataLab from './DataLab'
import StrategyReverseEngineer from './StrategyReverseEngineer'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { cn } from '../lib/utils'


type InnerTab = 'code' | 'studio' | 'data' | 'reverse'


export default function AutoresearchPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  // ── Strategy picker (the only context the Research subview needs) ──
  const strategiesQuery = useQuery({
    queryKey: ['autoresearch-panel-strategies'],
    queryFn: () => getTraderStrategies(),
    staleTime: 30_000,
  })
  const strategyCatalog = useMemo(
    () => strategiesQuery.data || [],
    [strategiesQuery.data],
  )

  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null)
  const [innerTab, setInnerTab] = useState<InnerTab>(() => {
    // Honor a deep-link hand-off (e.g. WalletAnalysisPanel →
    // "Reverse-engineer strategy" stashes 'reverse' in sessionStorage).
    try {
      const requested = sessionStorage.getItem('homerun:research:inner')
      if (requested) {
        sessionStorage.removeItem('homerun:research:inner')
        if (
          requested === 'code'
          || requested === 'studio'
          || requested === 'data'
          || requested === 'reverse'
        ) {
          return requested
        }
      }
    } catch {
      /* ignore */
    }
    return 'studio'
  })

  // Resolve strategy: deep-link signal → first item in catalog.
  useEffect(() => {
    if (selectedStrategyId && strategyCatalog.find((s) => s.id === selectedStrategyId)) return
    if (strategyCatalog.length === 0) return
    let target: { id: string } | undefined
    try {
      const requestedSlug = sessionStorage.getItem('homerun:research:strategy') || ''
      if (requestedSlug) {
        sessionStorage.removeItem('homerun:research:strategy')
        target = strategyCatalog.find(
          (s: any) =>
            String(s.slug || '').trim().toLowerCase() === requestedSlug.toLowerCase()
            || String(s.strategy_key || '').trim().toLowerCase() === requestedSlug.toLowerCase(),
        )
      }
    } catch {
      target = undefined
    }
    if (!target) target = strategyCatalog[0] as any
    if (target) setSelectedStrategyId((target as any).id)
  }, [strategyCatalog, selectedStrategyId])

  const selectedStrategyQuery = useQuery({
    queryKey: ['autoresearch-panel-strategy', selectedStrategyId],
    queryFn: () =>
      selectedStrategyId ? getTraderStrategy(selectedStrategyId) : Promise.resolve(null),
    enabled: Boolean(selectedStrategyId),
    staleTime: 15_000,
  })
  const selectedStrategy = selectedStrategyQuery.data || null

  return (
    <div className="h-full min-h-0 flex flex-col">
      {/* Header: title + strategy picker + inner subtab strip */}
      <div className="border-b border-border/40 shrink-0">
        <div className="flex items-center gap-3 px-4 py-2">
          <Sparkles className="w-4 h-4 text-violet-400 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold leading-tight">{t('autoresearch.title')}</p>
            <p className="text-[10px] text-muted-foreground leading-tight">
              {t('autoresearch.subtitle')}
            </p>
          </div>
          <div className="shrink-0 flex items-center gap-2">
            <FlaskConical className="w-3.5 h-3.5 text-muted-foreground" />
            <Select
              value={selectedStrategyId ?? undefined}
              onValueChange={(value) => setSelectedStrategyId(value)}
            >
              <SelectTrigger className="h-8 text-xs min-w-[220px]">
                <SelectValue placeholder={t('autoresearch.pickStrategy')} />
              </SelectTrigger>
              <SelectContent>
                {strategyCatalog.map((s: any) => (
                  <SelectItem key={s.id} value={s.id} className="text-xs">
                    {s.label || s.name || s.slug}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex items-center gap-0.5 px-3">
          <button
            type="button"
            onClick={() => setInnerTab('code')}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors',
              innerTab === 'code'
                ? 'border-purple-500 text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
          >
            <Code2 className="w-3 h-3" /> {t('autoresearch.tabCode')}
          </button>
          <button
            type="button"
            onClick={() => setInnerTab('studio')}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors',
              innerTab === 'studio'
                ? 'border-amber-500 text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
          >
            <FlaskConical className="w-3 h-3" /> {t('autoresearch.tabStudio')}
          </button>
          <button
            type="button"
            onClick={() => setInnerTab('data')}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors',
              innerTab === 'data'
                ? 'border-violet-500 text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
          >
            <Database className="w-3 h-3" /> {t('autoresearch.tabData')}
          </button>
          <button
            type="button"
            onClick={() => setInnerTab('reverse')}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors',
              innerTab === 'reverse'
                ? 'border-cyan-500 text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
            title={t('autoresearch.tabReverseTitle')}
          >
            <Sparkles className="w-3 h-3" /> {t('autoresearch.tabReverse')}
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {innerTab === 'code' ? (
          selectedStrategyId ? (
            <StrategyCodeExperiments
              strategyId={selectedStrategyId}
              strategyLabel={selectedStrategy
                ? String((selectedStrategy as any).label || (selectedStrategy as any).name || (selectedStrategy as any).slug || '')
                : ''
              }
              queryClient={queryClient}
            />
          ) : (
            <PanelEmpty message={t('autoresearch.pickStrategyForCode')} />
          )
        ) : innerTab === 'data' ? (
          <DataLab />
        ) : innerTab === 'reverse' ? (
          <StrategyReverseEngineer
            initialWalletAddress={(() => {
              try {
                const w = sessionStorage.getItem('homerun:reverse-engineer:wallet')
                if (w) sessionStorage.removeItem('homerun:reverse-engineer:wallet')
                return w
              } catch {
                return null
              }
            })()}
          />
        ) : selectedStrategy ? (
          <BacktestStudio
            initialSourceCode={(selectedStrategy as any).source_code || ''}
            // Strategy UUID (NOT slug) — required for the param-iteration
            // endpoint /api/autoresearch/strategy/{id}/params/stream which
            // looks up Strategy by primary key, not slug.
            initialStrategyId={String((selectedStrategy as any).id || '')}
            initialSlug={String((selectedStrategy as any).slug || (selectedStrategy as any).strategy_key || '_research')}
            initialConfig={(selectedStrategy as any).default_params_json || (selectedStrategy as any).config || {}}
            // Pass the strategy's declared param schema so the studio
            // can render the dynamic Strategy parameters panel — the
            // SAME field schema the bot orchestrator's tune subtab
            // uses.  Either of the two field names is acceptable
            // (apiTraders.getTraderStrategy normalizes both):
            // ``param_schema_json`` is the canonical wire shape from
            // /strategy-manager/{id}; ``config_schema`` is the legacy
            // shape some callers still pass through.
            initialParamSchema={
              (selectedStrategy as any).param_schema_json
              || (selectedStrategy as any).config_schema
              || null
            }
            strategyLabel={String((selectedStrategy as any).label || (selectedStrategy as any).name || (selectedStrategy as any).slug || '')}
          />
        ) : (
          <PanelEmpty
            title={selectedStrategyQuery.isLoading ? t('autoresearch.loadingStrategy') : t('autoresearch.pickStrategyTitle')}
            message={t('autoresearch.studioNeedsStrategy')}
          />
        )}
      </div>
    </div>
  )
}


// ────────────────────────────────────────────────────────────────────────
// Strategy-scoped code-evolution UI
// ────────────────────────────────────────────────────────────────────────

interface StreamIteration {
  iteration: number
  decision: 'kept' | 'reverted' | 'pending'
  new_score: number
  score_delta: number
  reasoning: string
  validation_passed?: boolean | null
}


function StrategyCodeExperiments({
  strategyId,
  strategyLabel,
  queryClient,
}: {
  strategyId: string
  strategyLabel: string
  queryClient: ReturnType<typeof useQueryClient>
}) {
  const { t } = useTranslation()
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamIterations, setStreamIterations] = useState<StreamIteration[]>([])
  const [streamPhase, setStreamPhase] = useState<string>('')
  const [streamError, setStreamError] = useState<string>('')
  const [showSettings, setShowSettings] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const statusQuery = useQuery({
    queryKey: ['strategy-autoresearch-status', strategyId],
    queryFn: () => getStrategyAutoresearchStatus(strategyId),
    refetchInterval: isStreaming ? false : 10_000,
  })
  const historyQuery = useQuery({
    queryKey: ['strategy-autoresearch-history', strategyId],
    queryFn: () => getStrategyAutoresearchHistory(strategyId, 30),
    refetchInterval: isStreaming ? false : 15_000,
  })
  const settingsQuery = useQuery({
    queryKey: ['autoresearch-settings'],
    queryFn: getAutoresearchSettings,
  })

  const settingsMutation = useMutation({
    mutationFn: updateAutoresearchSettings,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['autoresearch-settings'] }),
  })
  const stopMutation = useMutation({
    mutationFn: () => stopStrategyAutoresearchExperiment(strategyId),
    onSuccess: () => {
      abortRef.current?.abort()
      setIsStreaming(false)
      queryClient.invalidateQueries({ queryKey: ['strategy-autoresearch-status', strategyId] })
      queryClient.invalidateQueries({ queryKey: ['strategy-autoresearch-history', strategyId] })
    },
  })

  const status: AutoresearchExperimentStatus | undefined = statusQuery.data
  const dbIterations: AutoresearchIteration[] = historyQuery.data?.iterations || []
  const settings: AutoresearchSettings | undefined = settingsQuery.data
  const isExperimentRunning = isStreaming || status?.status === 'running'

  const allIterations: StreamIteration[] = isStreaming
    ? streamIterations
    : dbIterations.map((it) => ({
        iteration: it.iteration_number ?? 0,
        decision: ((it.decision as 'kept' | 'reverted' | 'pending') ?? 'pending'),
        new_score: it.new_score ?? 0,
        score_delta: it.score_delta ?? 0,
        reasoning: it.reasoning ?? '',
        validation_passed: (it.validation_result as any)?.valid ?? null,
      }))

  const handleStart = () => {
    if (isExperimentRunning || !strategyId) return
    setIsStreaming(true)
    setStreamIterations([])
    setStreamPhase('starting')
    setStreamError('')

    const controller = new AbortController()
    abortRef.current = controller

    streamStrategyAutoresearchExperiment(
      strategyId,
      (event) => {
        const { event: type, data } = event as { event: string; data: any }
        if (type === 'experiment_start') {
          setStreamPhase('running')
        } else if (type === 'iteration_start') {
          setStreamIterations((prev) => [
            ...prev,
            {
              iteration: data.iteration ?? prev.length + 1,
              decision: 'pending',
              new_score: 0,
              score_delta: 0,
              reasoning: '',
            },
          ])
        } else if (type === 'proposal') {
          setStreamIterations((prev) => prev.map((it) =>
            it.iteration === data.iteration ? { ...it, reasoning: data.reasoning ?? it.reasoning } : it,
          ))
        } else if (type === 'decision') {
          setStreamIterations((prev) => prev.map((it) =>
            it.iteration === data.iteration
              ? {
                  ...it,
                  decision: (data.decision as 'kept' | 'reverted') ?? 'reverted',
                  new_score: data.new_score ?? 0,
                  score_delta: data.score_delta ?? 0,
                  validation_passed: data.validation_passed ?? null,
                }
              : it,
          ))
        } else if (type === 'error') {
          setStreamError(String(data?.error || t('autoresearch.unknownError')))
        }
      },
      () => {
        setIsStreaming(false)
        setStreamPhase('done')
        queryClient.invalidateQueries({ queryKey: ['strategy-autoresearch-status', strategyId] })
        queryClient.invalidateQueries({ queryKey: ['strategy-autoresearch-history', strategyId] })
      },
      (err) => {
        setStreamError(err)
        setIsStreaming(false)
      },
      controller.signal,
      settings ? {
        model: (settings as any).model,
        max_iterations: (settings as any).max_iterations,
        mandate: (settings as any).mandate,
      } : undefined,
    )
  }

  const handleStop = () => {
    stopMutation.mutate()
  }

  const baseline = status?.baseline_score ?? 0
  const best = status?.best_score ?? 0
  const delta = best - baseline

  return (
    <div className="h-full min-h-0 flex flex-col p-3 gap-3 overflow-hidden">
      {/* Pitch + run controls */}
      <div className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-3 space-y-2 shrink-0">
        <div className="flex items-center gap-2">
          <Code2 className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-semibold">{t('autoresearch.codeEvolutionLoop')}</h3>
          <span className="text-[10px] text-muted-foreground">
            {t('autoresearch.forLabel')} <span className="font-mono text-foreground">{strategyLabel}</span>
          </span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-[10px]"
              onClick={() => setShowSettings(!showSettings)}
              disabled={isExperimentRunning}
            >
              {t('autoresearch.settings')}
            </Button>
            {isExperimentRunning ? (
              <Button
                type="button"
                size="sm"
                className="h-7 px-3 text-[11px] bg-red-600 hover:bg-red-500 text-white"
                onClick={handleStop}
                disabled={stopMutation.isPending}
              >
                <Square className="w-3 h-3 mr-1" />
                {t('autoresearch.stop')}
              </Button>
            ) : (
              <Button
                type="button"
                size="sm"
                className="h-7 px-3 text-[11px] bg-purple-600 hover:bg-purple-500 text-white"
                onClick={handleStart}
                disabled={!strategyId || isExperimentRunning}
              >
                <Play className="w-3 h-3 mr-1" />
                {t('autoresearch.start')}
              </Button>
            )}
          </div>
        </div>
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          {t('autoresearch.loopDescStart')}{' '}
          <span className="font-mono text-foreground">{(settings as any)?.max_iterations ?? 50}</span>{' '}
          {t('autoresearch.loopDescEnd')}
        </p>

        {/* Score header */}
        <div className="grid grid-cols-4 gap-2 text-[11px]">
          <ScoreCard label={t('autoresearch.scoreStatus')} value={isStreaming ? streamPhase || t('autoresearch.running') : (status?.status ?? t('autoresearch.idle'))} tone={isExperimentRunning ? 'good' : 'neutral'} />
          <ScoreCard label={t('autoresearch.scoreBaseline')} value={baseline.toFixed(3)} tone="neutral" />
          <ScoreCard label={t('autoresearch.scoreBest')} value={best.toFixed(3)} tone={delta > 0 ? 'good' : 'neutral'} />
          <ScoreCard label="Δ" value={(delta >= 0 ? '+' : '') + delta.toFixed(3)} tone={delta > 0 ? 'good' : delta < 0 ? 'bad' : 'neutral'} />
        </div>

        {showSettings && settings && (
          <SettingsEditor
            settings={settings}
            onSave={(s) => settingsMutation.mutate(s)}
            isSaving={settingsMutation.isPending}
          />
        )}
      </div>

      {streamError && (
        <div className="rounded border border-red-500/30 bg-red-500/5 p-2 text-[11px] text-red-300 shrink-0">
          <AlertTriangle className="inline w-3.5 h-3.5 mr-1 align-text-bottom" />
          {streamError}
        </div>
      )}

      {/* Iteration log */}
      <div className="flex-1 min-h-0 rounded-lg border border-border/40 bg-card/30 overflow-hidden flex flex-col">
        <div className="shrink-0 px-3 py-1.5 border-b border-border/30 flex items-center gap-2 text-[11px]">
          <span className="font-medium">{t('autoresearch.iterations')}</span>
          <span className="text-muted-foreground">
            {allIterations.length} {isStreaming ? `· ${t('autoresearch.streaming')}` : ''}
          </span>
          {status?.kept_count !== undefined && (
            <span className="ml-auto text-[10px] text-muted-foreground font-mono">
              {t('autoresearch.keptLabel')} {status.kept_count} · {t('autoresearch.revertedLabel')} {status.reverted_count}
            </span>
          )}
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto">
          {allIterations.length === 0 ? (
            <div className="h-full flex items-center justify-center text-[11px] text-muted-foreground">
              {t('autoresearch.noIterationsYet')}
            </div>
          ) : (
            <table className="w-full text-[10px] font-mono">
              <thead className="sticky top-0 bg-card/95">
                <tr className="border-b border-border/30 text-muted-foreground">
                  <th className="text-left py-1 px-2 w-12">#</th>
                  <th className="text-left py-1 px-2 w-20">{t('autoresearch.colDecision')}</th>
                  <th className="text-right py-1 px-2 w-20">{t('autoresearch.colScore')}</th>
                  <th className="text-right py-1 px-2 w-20">Δ</th>
                  <th className="text-left py-1 px-2">{t('autoresearch.colReasoning')}</th>
                </tr>
              </thead>
              <tbody>
                {allIterations.map((it, i) => (
                  <tr key={i} className="border-b border-border/10 hover:bg-card/60">
                    <td className="py-1 px-2 text-muted-foreground">{it.iteration}</td>
                    <td className={cn(
                      'py-1 px-2 font-medium',
                      it.decision === 'kept' ? 'text-emerald-400' :
                      it.decision === 'reverted' ? 'text-red-400' : 'text-muted-foreground',
                    )}>
                      {it.decision === 'kept' && <Check className="inline w-3 h-3 mr-0.5 align-text-bottom" />}
                      {it.decision === 'reverted' && <X className="inline w-3 h-3 mr-0.5 align-text-bottom" />}
                      {it.decision === 'pending' && <Loader2 className="inline w-3 h-3 mr-0.5 align-text-bottom animate-spin" />}
                      {it.decision}
                    </td>
                    <td className="py-1 px-2 text-right">{Number(it.new_score || 0).toFixed(3)}</td>
                    <td className={cn(
                      'py-1 px-2 text-right',
                      it.score_delta > 0 ? 'text-emerald-400' : it.score_delta < 0 ? 'text-red-400' : 'text-muted-foreground',
                    )}>
                      {(it.score_delta >= 0 ? '+' : '') + Number(it.score_delta || 0).toFixed(3)}
                    </td>
                    <td className="py-1 px-2 text-muted-foreground truncate max-w-[400px]">
                      {it.reasoning?.slice(0, 200) || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}


function ScoreCard({ label, value, tone }: { label: string; value: string; tone: 'good' | 'bad' | 'neutral' }) {
  return (
    <div className={cn(
      'rounded-md border p-2',
      tone === 'good' ? 'border-emerald-500/30 bg-emerald-500/5' :
      tone === 'bad' ? 'border-red-500/30 bg-red-500/5' :
      'border-border/40 bg-card/30',
    )}>
      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-sm font-mono font-bold">{value}</p>
    </div>
  )
}


function SettingsEditor({
  settings,
  onSave,
  isSaving,
}: {
  settings: AutoresearchSettings
  onSave: (s: Partial<AutoresearchSettings>) => void
  isSaving: boolean
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState({
    model: String((settings as any).model || ''),
    max_iterations: String((settings as any).max_iterations ?? 50),
    mandate: String((settings as any).mandate || ''),
  })

  // Fetch available models from configured providers — same source the
  // global Settings → LLM page uses, so the dropdown stays in sync with
  // whatever providers/keys the user has set up.
  const llmModelsQuery = useQuery({
    queryKey: ['llm-models', 'all'],
    queryFn: () => getLLMModels(),
    staleTime: 60_000,
  })
  const groupedModels = llmModelsQuery.data?.models || {}
  const flatModels: Array<{ provider: string; option: LLMModelOption }> = []
  for (const [provider, opts] of Object.entries(groupedModels)) {
    for (const option of opts || []) {
      flatModels.push({ provider, option })
    }
  }
  const hasModelOptions = flatModels.length > 0
  // If the saved model isn't present in the catalog, surface it anyway
  // so we don't silently drop it.
  const draftModelMissing =
    Boolean(draft.model)
    && hasModelOptions
    && !flatModels.some(({ option }) => option.id === draft.model)

  return (
    <div className="rounded border border-border/40 bg-background/40 p-2 space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-[10px] text-muted-foreground">{t('autoresearch.model')}</Label>
          <Select
            value={draft.model || undefined}
            onValueChange={(value) => setDraft({ ...draft, model: value })}
            disabled={llmModelsQuery.isLoading || !hasModelOptions}
          >
            <SelectTrigger className="mt-1 h-7 text-[11px]">
              <SelectValue placeholder={
                llmModelsQuery.isLoading
                  ? t('autoresearch.loadingModels')
                  : hasModelOptions
                    ? t('autoresearch.pickModel')
                    : t('autoresearch.noProviders')
              } />
            </SelectTrigger>
            <SelectContent>
              {draftModelMissing && (
                <SelectItem value={draft.model} className="text-xs italic text-muted-foreground">
                  {draft.model} <span className="opacity-60">{t('autoresearch.notInCatalog')}</span>
                </SelectItem>
              )}
              {Object.entries(groupedModels).map(([provider, opts]) => (
                <div key={provider}>
                  <div className="px-2 py-1 text-[9px] uppercase tracking-wider text-muted-foreground/70 font-mono">
                    {provider}
                  </div>
                  {(opts || []).map((option) => (
                    <SelectItem key={option.id} value={option.id} className="text-xs">
                      {option.name || option.id}
                    </SelectItem>
                  ))}
                </div>
              ))}
            </SelectContent>
          </Select>
          {!hasModelOptions && !llmModelsQuery.isLoading && (
            <p className="mt-1 text-[9px] text-muted-foreground">
              {t('autoresearch.configureProviderHint')}
            </p>
          )}
        </div>
        <div>
          <Label className="text-[10px] text-muted-foreground">{t('autoresearch.maxIterations')}</Label>
          <Input
            value={draft.max_iterations}
            onChange={(e) => setDraft({ ...draft, max_iterations: e.target.value })}
            className="mt-1 h-7 text-[11px]"
            placeholder="50"
          />
        </div>
      </div>
      <div>
        <Label className="text-[10px] text-muted-foreground">{t('autoresearch.mandate')}</Label>
        <Input
          value={draft.mandate}
          onChange={(e) => setDraft({ ...draft, mandate: e.target.value })}
          className="mt-1 h-7 text-[11px]"
          placeholder={t('autoresearch.mandatePlaceholder')}
        />
      </div>
      <div className="flex justify-end">
        <Button
          type="button"
          size="sm"
          className="h-7 px-3 text-[10px]"
          onClick={() => onSave({
            model: draft.model || undefined,
            max_iterations: Number(draft.max_iterations) || undefined,
            mandate: draft.mandate || undefined,
          } as any)}
          disabled={isSaving}
        >
          {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : t('autoresearch.saveSettings')}
        </Button>
      </div>
    </div>
  )
}


function PanelEmpty({ title, message }: { title?: string; message: string }) {
  return (
    <div className="h-full min-h-0 flex items-center justify-center">
      <div className="text-center max-w-sm space-y-2 px-4">
        <FlaskConical className="w-8 h-8 mx-auto text-muted-foreground/40" />
        {title && <p className="text-sm font-medium">{title}</p>}
        <p className="text-[11px] text-muted-foreground leading-snug">{message}</p>
      </div>
    </div>
  )
}
