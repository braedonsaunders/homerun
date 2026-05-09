import { useState, useRef, useCallback, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient, type UseMutationResult } from '@tanstack/react-query'
import {
  Play,
  Square,
  Check,
  X,
  FlaskConical,
  Settings2,
  Loader2,
  TrendingUp,
  Code2,
  SlidersHorizontal,
  FlaskRound,
  ChevronDown,
  ChevronRight,
  ArrowRight,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Switch } from './ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Label } from './ui/label'
import StrategyConfigForm from './StrategyConfigForm'
import {
  getAutoresearchStatus,
  getAutoresearchHistory,
  getAutoresearchSettings,
  updateAutoresearchSettings,
  stopAutoresearchExperiment,
  streamAutoresearchExperiment,
  createAbExperimentFromAutoresearch,
  getPlugins,
  type ChatStreamEvent,
  type Trader,
} from '../services/api'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StrategyParamSection {
  sectionKey: string
  sourceKey: string
  sourceLabel: string
  strategyLabel: string
  fieldKeys: string[]
  values: Record<string, unknown>
  groups: { key: string; label: string; fields: unknown[] }[]
}

interface RevertSnapshot {
  traderId: string
  capturedAt: string
}

interface AutoresearchViewProps {
  trader: Trader
  dynamicStrategyParamSections: StrategyParamSection[]
  tuneParamSectionTab: string
  setTuneParamSectionTab: (tab: string) => void
  tuneDraftDirty: boolean
  setTuneDraftDirty: (dirty: boolean) => void
  applyTraderDraftSettings: (trader: Trader, options?: Record<string, boolean>) => void
  applyDynamicStrategyFormValues: (sourceKey: string, keys: string[], values: Record<string, unknown>) => void
  saveTuneParametersMutation: UseMutationResult<unknown, unknown, void, unknown>
  revertTuneParametersMutation: UseMutationResult<unknown, unknown, void, unknown>
  tuneRevertSnapshot: RevertSnapshot | null
  tuneSaveError: string | null
  tuneRevertError: string | null
  formatTimestamp: (iso: string) => string
  /**
   * Pin the view to a specific subtab. When provided, the in-component
   * Parameters/Autoresearch toggle is hidden — TradingPanel keeps both
   * (omits this prop) so users can switch between the manual Parameter
   * Workspace and the params-mode autoresearcher; Strategies → Research
   * pins to "autoresearch" (no manual workspace there — that's per-bot).
   */
  forceTopTab?: 'parameters' | 'autoresearch'
  /**
   * Pin the autoresearcher's inner Parameters/Code toggle to one mode.
   * The Tune tab pins to "params" because the autoresearcher there
   * adjusts the same live bot parameters a human can edit. The Research
   * subview pins to "code" because that's where strategy source
   * evolution belongs (alongside the backtest suite).
   */
  forceArMode?: 'params' | 'code'
}

interface StreamIteration {
  iteration: number
  decision: 'kept' | 'reverted' | 'pending'
  new_score: number
  score_delta: number
  reasoning: string
  changed_params: Record<string, unknown> | null
  duration_seconds: number
  source_diff?: string | null
  source_diff_lines?: number
  validation_passed?: boolean | null
  validation_result?: Record<string, unknown> | null
  backtest_result?: Record<string, unknown> | null
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AutoresearchView({
  trader,
  dynamicStrategyParamSections,
  tuneParamSectionTab,
  setTuneParamSectionTab,
  tuneDraftDirty,
  setTuneDraftDirty,
  applyTraderDraftSettings,
  applyDynamicStrategyFormValues,
  saveTuneParametersMutation,
  revertTuneParametersMutation,
  tuneRevertSnapshot,
  tuneSaveError,
  tuneRevertError,
  formatTimestamp,
  forceTopTab,
  forceArMode,
}: AutoresearchViewProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const abortRef = useRef<AbortController | null>(null)

  // Top-level subtab: "parameters" or "autoresearch".
  // ``forceTopTab`` (when provided by the caller) hides the toggle and
  // pins the view to one half — used to physically separate the live
  // parameter editor (TradingPanel/Tune) from the autoresearcher
  // (Strategies/Research).
  const [topTab, setTopTab] = useState<'parameters' | 'autoresearch'>(forceTopTab || 'parameters')
  useEffect(() => {
    if (forceTopTab && forceTopTab !== topTab) {
      setTopTab(forceTopTab)
    }
  }, [forceTopTab, topTab])

  // Autoresearch inner mode: "params" or "code".
  // ``forceArMode`` (when set) hides the inner toggle and pins the mode.
  const [arMode, setArMode] = useState<'params' | 'code'>(forceArMode || 'params')
  useEffect(() => {
    if (forceArMode && forceArMode !== arMode) {
      setArMode(forceArMode)
    }
  }, [forceArMode, arMode])

  // Streaming state — streamingMode tracks which mode started the experiment
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingMode, setStreamingMode] = useState<'params' | 'code' | null>(null)
  const [streamIterations, setStreamIterations] = useState<StreamIteration[]>([])
  const [streamPhase, setStreamPhase] = useState<string>('')
  const [streamError, setStreamError] = useState<string>('')

  // Settings panel
  const [showSettings, setShowSettings] = useState(false)

  // Code mode state
  const [selectedStrategyKey, setSelectedStrategyKey] = useState<string>('')
  const [expandedIteration, setExpandedIteration] = useState<number | null>(null)
  const [abCreating, setAbCreating] = useState(false)
  const [doneData, setDoneData] = useState<Record<string, unknown> | null>(null)

  // Params mode — expanded row for reasoning detail
  const [expandedParamIteration, setExpandedParamIteration] = useState<number | null>(null)

  // Strategy list to resolve strategy_key → strategy ID
  const { data: strategies } = useQuery({
    queryKey: ['strategy-plugins'],
    queryFn: getPlugins,
    staleTime: 60000,
  })

  // Derive trader's strategies from source_configs
  const traderStrategies = (trader.source_configs ?? [])
    .map(sc => sc.strategy_key)
    .filter((v, i, arr) => v && arr.indexOf(v) === i)  // unique, non-empty

  // Auto-select if trader has exactly one strategy
  useEffect(() => {
    if (traderStrategies.length === 1 && !selectedStrategyKey) {
      setSelectedStrategyKey(traderStrategies[0])
    }
  }, [traderStrategies, selectedStrategyKey])

  // Resolve strategy_key → strategy ID via the plugins list
  const selectedStrategyId = (strategies ?? []).find(
    s => s.slug === selectedStrategyKey || s.name === selectedStrategyKey
  )?.id ?? ''

  // Queries
  const { data: status } = useQuery({
    queryKey: ['autoresearch-status', trader.id],
    queryFn: () => getAutoresearchStatus(trader.id),
    refetchInterval: isStreaming ? false : 10000,
  })

  const { data: historyData } = useQuery({
    queryKey: ['autoresearch-history', trader.id],
    queryFn: () => getAutoresearchHistory(trader.id, 30),
    refetchInterval: isStreaming ? false : 15000,
  })

  const { data: settings } = useQuery({
    queryKey: ['autoresearch-settings'],
    queryFn: getAutoresearchSettings,
  })

  const settingsMutation = useMutation({
    mutationFn: updateAutoresearchSettings,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['autoresearch-settings'] }),
  })

  // Whether the current mode owns the running experiment
  const isModeStreaming = isStreaming && streamingMode === arMode

  // Display iterations — filter by current arMode so params/code views
  // show only their own iterations instead of rendering the same table.
  const dbIterations = historyData?.iterations ?? []
  const allIterations = isModeStreaming
    ? streamIterations
    : dbIterations.map(i => ({
        iteration: i.iteration_number,
        decision: i.decision,
        new_score: i.new_score,
        score_delta: i.score_delta,
        reasoning: i.reasoning,
        changed_params: i.changed_params,
        duration_seconds: i.duration_seconds,
        source_diff: i.source_diff,
        source_diff_lines: i.source_diff_lines ?? (i.source_diff ? i.source_diff.split('\n').length : 0),
        validation_passed: typeof i.validation_result?.valid === 'boolean' ? i.validation_result.valid : null,
        validation_result: i.validation_result ?? null,
        backtest_result: i.backtest_result,
      } as StreamIteration))
  const latestExperimentMode = isModeStreaming ? streamingMode : (status?.mode ?? 'params')
  const displayIterations = allIterations.filter(() => {
    if (arMode === 'code') return latestExperimentMode === 'code'
    return latestExperimentMode !== 'code'
  })

  // Derived — use isModeStreaming so stats reflect the current tab's experiment
  const experimentStatus = isModeStreaming ? 'running' : (status?.status ?? 'idle')
  const bestScore = isModeStreaming
    ? (streamIterations.length > 0 ? Math.max(...streamIterations.filter(i => i.decision === 'kept').map(i => i.new_score), status?.baseline_score ?? 0) : status?.baseline_score ?? 0)
    : (status?.best_score ?? 0)
  const baselineScore = status?.baseline_score ?? 0
  const iterationCount = isModeStreaming ? streamIterations.length : (status?.iteration_count ?? 0)
  const keptCount = isModeStreaming ? streamIterations.filter(i => i.decision === 'kept').length : (status?.kept_count ?? 0)
  const revertedCount = isModeStreaming ? streamIterations.filter(i => i.decision === 'reverted').length : (status?.reverted_count ?? 0)

  // Start experiment
  const handleStart = useCallback(() => {
    setIsStreaming(true)
    setStreamingMode(arMode)
    setStreamIterations([])
    setStreamPhase('starting')
    setStreamError('')
    setDoneData(null)
    const controller = new AbortController()
    abortRef.current = controller

    streamAutoresearchExperiment(
      trader.id,
      (event: ChatStreamEvent) => {
        const { event: type, data } = event
        switch (type) {
          case 'experiment_start':
            setStreamPhase('running')
            break
          case 'iteration_start':
            setStreamPhase(`iteration ${data.iteration}/${data.total}`)
            setStreamIterations(prev => {
              const iteration = Number(data.iteration)
              const score = Number(data.baseline_score ?? 0)
              return [
                {
                  iteration,
                  decision: 'pending',
                  new_score: Number.isFinite(score) ? score : 0,
                  score_delta: 0,
                  reasoning: t('autoresearchView.iterationStartedWaiting'),
                  changed_params: null,
                  duration_seconds: 0,
                  source_diff: null,
                  source_diff_lines: 0,
                  validation_passed: null,
                  validation_result: null,
                  backtest_result: null,
                },
                ...prev.filter(iter => iter.iteration !== iteration),
              ]
            })
            break
          case 'agent_progress':
            setStreamPhase(String(data.message || 'researching...'))
            setStreamIterations(prev => {
              const iteration = Number(data.iteration)
              const message = String(data.message || '')
              return prev.map(iter => iter.iteration === iteration ? {
                ...iter,
                reasoning: message || iter.reasoning,
              } : iter)
            })
            break
          case 'proposal':
            setStreamPhase('evaluating proposal...')
            setStreamIterations(prev => {
              const iteration = Number(data.iteration)
              const proposedChanges = data.proposed_changes as Record<string, unknown> | undefined
              const reasoning = String(data.reasoning || '')
              return prev.map(iter => iter.iteration === iteration ? {
                ...iter,
                reasoning: reasoning || iter.reasoning,
                changed_params: proposedChanges ?? iter.changed_params,
              } : iter)
            })
            break
          case 'decision': {
            const iteration = Number(data.iteration)
            setStreamIterations(prev => [
              {
                iteration,
                decision: data.decision as 'kept' | 'reverted',
                new_score: data.new_score as number,
                score_delta: data.score_delta as number,
                reasoning: (data.reasoning as string) || '',
                changed_params: (data.changed_params as Record<string, unknown>) || null,
                duration_seconds: data.duration_seconds as number,
                source_diff: (data.source_diff as string | null) || null,
                source_diff_lines: (data.source_diff_lines as number) || 0,
                validation_passed: data.validation_passed as boolean | null,
                validation_result: (data.validation_result as Record<string, unknown> | null) || null,
                backtest_result: (data.backtest_result as Record<string, unknown> | null) || null,
              },
              ...prev.filter(iter => iter.iteration !== iteration),
            ])
            setStreamPhase(`iteration ${data.iteration} — ${data.decision}`)
            break
          }
          case 'error':
            setStreamError(String(data.error || 'Unknown error'))
            break
          case 'done':
            setStreamPhase('completed')
            setDoneData(data)
            break
        }
      },
      () => {
        setIsStreaming(false)
        setStreamingMode(null)
        abortRef.current = null
        queryClient.invalidateQueries({ queryKey: ['autoresearch-status', trader.id] })
        queryClient.invalidateQueries({ queryKey: ['autoresearch-history', trader.id] })
      },
      (error) => {
        setIsStreaming(false)
        setStreamingMode(null)
        abortRef.current = null
        setStreamError(error)
      },
      controller.signal,
      arMode === 'code' ? { mode: 'code', strategy_id: selectedStrategyId || undefined } : { mode: 'params' },
    )
  }, [trader.id, queryClient, arMode, selectedStrategyId, selectedStrategyKey, t])

  const handleStop = useCallback(async () => {
    abortRef.current?.abort()
    setIsStreaming(false)
    setStreamingMode(null)
    try { await stopAutoresearchExperiment(trader.id) } catch { /* ignore */ }
    queryClient.invalidateQueries({ queryKey: ['autoresearch-status', trader.id] })
    queryClient.invalidateQueries({ queryKey: ['autoresearch-history', trader.id] })
  }, [trader.id, queryClient])

  useEffect(() => { return () => { abortRef.current?.abort() } }, [])

  const statusColor = {
    running: 'bg-emerald-500',
    paused: 'bg-amber-500',
    completed: 'bg-blue-500',
    failed: 'bg-red-500',
    idle: 'bg-muted-foreground',
  }[experimentStatus] ?? 'bg-muted-foreground'
  const expandedCodeIteration = expandedIteration === null
    ? null
    : displayIterations.find(iter => iter.iteration === expandedIteration) ?? null
  const expandedValidationErrors = Array.isArray(expandedCodeIteration?.validation_result?.errors)
    ? expandedCodeIteration.validation_result.errors
    : []

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="h-full min-h-0 overflow-hidden px-1">
      <div className="h-full min-h-0 flex flex-col rounded-md border border-border/50 bg-muted/10">
        {/* ===== Top-level subtabs: Parameters | Autoresearch =====
            Hidden when ``forceTopTab`` is set so callers can pin the view
            to a single subtab (TradingPanel/Tune → parameters,
            Strategies/Research → autoresearch). */}
        {!forceTopTab && (
          <div className="shrink-0 flex items-center gap-0.5 border-b border-border/50 px-2 pt-1">
            <button
              onClick={() => setTopTab('parameters')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors',
                topTab === 'parameters'
                  ? 'border-cyan-500 text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              <SlidersHorizontal className="w-3 h-3" /> {t('autoresearchView.parameters')}
            </button>
            <button
              onClick={() => setTopTab('autoresearch')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors',
                topTab === 'autoresearch'
                  ? 'border-purple-500 text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              <FlaskConical className="w-3 h-3" /> {t('autoresearchView.autoresearch')}
            </button>
          </div>
        )}

        {/* ===== Parameters subtab ===== */}
        {topTab === 'parameters' && (
          <div className="flex-1 min-h-0 overflow-hidden p-2.5">
            <div className="flex h-full min-h-0 flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <p className="text-[11px] font-medium">{t('autoresearchView.parameterWorkspace')}</p>
                  <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                    {t('autoresearchView.fieldsCount', { n: dynamicStrategyParamSections.reduce((sum, s) => sum + s.fieldKeys.length, 0) })}
                  </Badge>
                  {tuneDraftDirty && (
                    <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-amber-500">{t('autoresearchView.unsaved')}</span>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[10px]"
                    onClick={() => { applyTraderDraftSettings(trader); setTuneDraftDirty(false) }}>
                    {t('autoresearchView.discardEdits')}
                  </Button>
                  <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[10px]"
                    onClick={() => revertTuneParametersMutation.mutate()}
                    disabled={revertTuneParametersMutation.isPending || !tuneRevertSnapshot || tuneRevertSnapshot.traderId !== trader.id}>
                    {revertTuneParametersMutation.isPending && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
                    {t('autoresearchView.revertLastApplied')}
                  </Button>
                  <Button type="button" size="sm" className="h-6 px-2 text-[10px]"
                    onClick={() => saveTuneParametersMutation.mutate()}
                    disabled={saveTuneParametersMutation.isPending || !tuneDraftDirty}>
                    {saveTuneParametersMutation.isPending && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
                    {t('autoresearchView.saveParameters')}
                  </Button>
                </div>
              </div>
              {tuneRevertSnapshot && tuneRevertSnapshot.traderId === trader.id && (
                <p className="text-[10px] text-muted-foreground/80">{t('autoresearchView.revertSnapshotCaptured', { time: formatTimestamp(tuneRevertSnapshot.capturedAt) })}</p>
              )}
              {tuneSaveError && <p className="text-[10px] text-red-500">{tuneSaveError}</p>}
              {tuneRevertError && <p className="text-[10px] text-red-500">{tuneRevertError}</p>}
              {dynamicStrategyParamSections.length === 0 ? (
                <p className="text-[10px] text-muted-foreground/80">{t('autoresearchView.noDynamicFields')}</p>
              ) : (
                <Tabs value={tuneParamSectionTab} onValueChange={setTuneParamSectionTab}
                  className="flex min-h-0 flex-1 flex-col overflow-hidden">
                  <div className="shrink-0 overflow-x-auto pb-1">
                    <TabsList className="h-auto w-max min-w-full justify-start gap-1 rounded-md border border-border/50 bg-background/60 p-1">
                      {dynamicStrategyParamSections.map((section) => (
                        <TabsTrigger key={section.sectionKey} value={section.sectionKey} className="h-6 gap-1 px-2 text-[10px]">
                          <span className="max-w-[220px] truncate">{section.sourceLabel} &middot; {section.strategyLabel}</span>
                          <span className="text-[9px] text-muted-foreground">{section.fieldKeys.length}</span>
                        </TabsTrigger>
                      ))}
                    </TabsList>
                  </div>
                  {dynamicStrategyParamSections.map((section) => (
                    <TabsContent key={section.sectionKey} value={section.sectionKey} className="mt-0 flex min-h-0 flex-1 flex-col overflow-hidden">
                      {section.groups.length === 0 ? (
                        <p className="text-[10px] text-muted-foreground/80">{t('autoresearchView.noGroupedFields')}</p>
                      ) : (
                        <Tabs defaultValue={section.groups[0].key} className="flex min-h-0 flex-1 flex-col overflow-hidden">
                          <div className="shrink-0 overflow-x-auto pb-1">
                            <TabsList className="h-auto w-max min-w-full justify-start gap-1 rounded-md border border-border/50 bg-background/50 p-1">
                              {section.groups.map((group) => (
                                <TabsTrigger key={`${section.sectionKey}:${group.key}`} value={group.key} className="h-6 gap-1 px-2 text-[10px]">
                                  <span>{group.label}</span>
                                  <span className="text-[9px] text-muted-foreground">{group.fields.length}</span>
                                </TabsTrigger>
                              ))}
                            </TabsList>
                          </div>
                          {section.groups.map((group) => (
                            <TabsContent key={`${section.sectionKey}:panel:${group.key}`} value={group.key}
                              className="mt-0 min-h-0 flex-1 overflow-auto rounded-md border border-border/50 bg-background/65 p-2">
                              <StrategyConfigForm
                                schema={{ param_fields: group.fields as any[] }}
                                values={section.values}
                                  onChange={(nextValues: Record<string, unknown>) =>
                                    applyDynamicStrategyFormValues(section.sourceKey, section.fieldKeys, nextValues)
                                  }
                              />
                            </TabsContent>
                          ))}
                        </Tabs>
                      )}
                    </TabsContent>
                  ))}
                </Tabs>
              )}
            </div>
          </div>
        )}

        {/* ===== Autoresearch subtab ===== */}
        {topTab === 'autoresearch' && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            {/* Inner mode tabs: Params | Code.
                Hidden when ``forceArMode`` pins the mode externally. */}
            <div className="shrink-0 flex items-center justify-between gap-2 border-b border-border/40 px-3 py-1.5">
              {forceArMode ? (
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                  {forceArMode === 'params' ? (
                    <>
                      <SlidersHorizontal className="w-3 h-3 text-cyan-400" />
                      <span>{t('autoresearchView.paramExperimenter')}</span>
                    </>
                  ) : (
                    <>
                      <Code2 className="w-3 h-3 text-purple-400" />
                      <span>{t('autoresearchView.codeExperimenter')}</span>
                    </>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-1 rounded-md border border-border/60 bg-background p-0.5">
                  <button
                    onClick={() => setArMode('params')}
                    className={cn(
                      'flex items-center gap-1 rounded px-2.5 py-1 text-[10px] font-medium transition-colors',
                      arMode === 'params' ? 'bg-cyan-500/20 text-cyan-400' : 'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <SlidersHorizontal className="w-3 h-3" /> {t('autoresearchView.parameters')}
                  </button>
                  <button
                    onClick={() => setArMode('code')}
                    className={cn(
                      'flex items-center gap-1 rounded px-2.5 py-1 text-[10px] font-medium transition-colors',
                      arMode === 'code' ? 'bg-purple-500/20 text-purple-400' : 'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <Code2 className="w-3 h-3" /> {t('autoresearchView.code')}
                  </button>
                </div>
              )}

              {/* Strategy from trader's source_configs (code mode only) */}
              {arMode === 'code' && traderStrategies.length > 0 && (
                traderStrategies.length === 1 ? (
                  <Badge variant="outline" className="h-5 px-2 text-[10px] font-mono">
                    {traderStrategies[0]}
                  </Badge>
                ) : (
                  <select
                    value={selectedStrategyKey}
                    onChange={(e) => setSelectedStrategyKey(e.target.value)}
                    className="h-7 max-w-[220px] rounded-md border border-border/60 bg-background px-2 text-[10px]"
                  >
                    <option value="">{t('autoresearchView.selectStrategy')}</option>
                    {traderStrategies.map(key => (
                      <option key={key} value={key}>{key}</option>
                    ))}
                  </select>
                )
              )}
              {arMode === 'code' && traderStrategies.length === 0 && (
                <span className="text-[10px] text-amber-400">{t('autoresearchView.noStrategiesConfigured')}</span>
              )}

              {/* Status + controls */}
              <div className="flex items-center gap-2">
                <div className={cn('w-2 h-2 rounded-full', statusColor)} />
                <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono uppercase">
                  {experimentStatus}
                </Badge>
                <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
                  <span>{iterationCount}i</span>
                  <span className="text-emerald-500">{keptCount}K</span>
                  <span className="text-red-400">{revertedCount}R</span>
                </div>
                {bestScore > baselineScore && baselineScore > 0 && (
                  <span className="flex items-center gap-0.5 text-[9px] text-emerald-500">
                    <TrendingUp className="w-3 h-3" />+{(bestScore - baselineScore).toFixed(2)}
                  </span>
                )}
                <button onClick={() => setShowSettings(s => !s)}
                  className="rounded p-1 text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-colors">
                  <Settings2 className="w-3.5 h-3.5" />
                </button>
                {isModeStreaming ? (
                  <Button size="sm" variant="destructive" className="h-6 px-2 text-[10px]" onClick={handleStop}>
                    <Square className="w-3 h-3 mr-1" /> {t('autoresearchView.stop')}
                  </Button>
                ) : (
                  <Button size="sm" className="h-6 px-2 text-[10px]" onClick={handleStart}
                    disabled={(arMode === 'code' && !selectedStrategyKey) || (isStreaming && streamingMode !== arMode)}>
                    <Play className="w-3 h-3 mr-1" /> {t('autoresearchView.start')}
                  </Button>
                )}
                {isStreaming && streamingMode !== arMode && (
                  <span className="text-[9px] text-amber-400">{t('autoresearchView.modeRunning', { mode: streamingMode })}</span>
                )}
              </div>
            </div>

            {/* Settings collapsible */}
            {showSettings && (
              <div className="shrink-0 border-b border-border/40 px-3 py-2 space-y-2">
                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                  <div>
                    <Label className="text-[10px] text-muted-foreground">{t('autoresearchView.modelOverride')}</Label>
                    <Input value={settings?.model ?? ''} onChange={(e) => settingsMutation.mutate({ model: e.target.value || null })}
                      placeholder={t('autoresearchView.appDefault')} className="mt-0.5 h-7 text-xs font-mono" />
                  </div>
                  <div>
                    <Label className="text-[10px] text-muted-foreground">{t('autoresearchView.maxIterations')}</Label>
                    <Input type="number" min={1} max={500} value={settings?.max_iterations ?? 50}
                      onChange={(e) => settingsMutation.mutate({ max_iterations: parseInt(e.target.value) || 50 })}
                      className="mt-0.5 h-7 text-xs font-mono" />
                  </div>
                  <div>
                    <Label className="text-[10px] text-muted-foreground">{t('autoresearchView.temperature')}</Label>
                    <Input type="number" min={0} max={2} step={0.1} value={settings?.temperature ?? 0.2}
                      onChange={(e) => settingsMutation.mutate({ temperature: parseFloat(e.target.value) || 0.2 })}
                      className="mt-0.5 h-7 text-xs font-mono" />
                  </div>
                  <div className="flex items-end gap-2 pb-0.5">
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <Switch checked={settings?.auto_apply ?? true}
                        onCheckedChange={(v) => settingsMutation.mutate({ auto_apply: v })} />
                      <span className="text-[10px]">{t('autoresearchView.autoApplyKept')}</span>
                    </label>
                  </div>
                </div>
                {arMode === 'params' && (
                  <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                    <div>
                      <Label className="text-[10px] text-muted-foreground">{t('autoresearchView.walkForwardWindows')}</Label>
                      <Input type="number" min={1} max={20} value={settings?.walk_forward_windows ?? 5}
                        onChange={(e) => settingsMutation.mutate({ walk_forward_windows: parseInt(e.target.value) || 5 })}
                        className="mt-0.5 h-7 text-xs font-mono" />
                    </div>
                    <div>
                      <Label className="text-[10px] text-muted-foreground">{t('autoresearchView.trainRatio')}</Label>
                      <Input type="number" min={0.5} max={0.9} step={0.05} value={settings?.train_ratio ?? 0.7}
                        onChange={(e) => settingsMutation.mutate({ train_ratio: parseFloat(e.target.value) || 0.7 })}
                        className="mt-0.5 h-7 text-xs font-mono" />
                    </div>
                  </div>
                )}
                <div>
                  <Label className="text-[10px] text-muted-foreground">{t('autoresearchView.mandateConstraints')}</Label>
                  <textarea value={settings?.mandate ?? ''}
                    onChange={(e) => settingsMutation.mutate({ mandate: e.target.value || null })}
                    placeholder={t('autoresearchView.mandatePlaceholder')}
                    className="mt-0.5 min-h-[40px] max-h-[64px] w-full rounded-md border border-border/60 bg-background px-2 py-1 text-xs leading-relaxed resize-y" />
                </div>
              </div>
            )}

            {streamError && <p className="shrink-0 px-3 py-1 text-[10px] text-red-500">{streamError}</p>}
            {isModeStreaming && <p className="shrink-0 px-3 py-0.5 text-[9px] text-muted-foreground">{streamPhase}</p>}

            {/* Iteration log */}
            <div className="flex-1 min-h-0 overflow-auto px-2 py-1">
              {displayIterations.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <div className="text-center space-y-1">
                    <FlaskConical className="w-8 h-8 mx-auto opacity-20" />
                    <p className="text-[10px]">
                      {isModeStreaming ? t('autoresearchView.waitingFirstIteration') : (
                        arMode === 'code'
                          ? t('autoresearchView.selectStrategyToStart')
                          : t('autoresearchView.clickStartToOptimize')
                      )}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-1">
                  {arMode === 'params' ? (
                    /* ===== Params mode: card-based iteration list with visible reasoning ===== */
                    <div className="space-y-1">
                      {displayIterations.map((iter, idx) => {
                        const isExpanded = expandedParamIteration === iter.iteration
                        const paramEntries = iter.changed_params ? Object.entries(iter.changed_params) : []
                        return (
                          <div key={idx}
                            className={cn(
                              'rounded border transition-colors',
                              iter.decision === 'kept'
                                ? 'border-emerald-500/30 bg-emerald-500/5'
                                : 'border-border/30 bg-background/50'
                            )}
                          >
                            {/* Iteration header row */}
                            <button
                              onClick={() => setExpandedParamIteration(isExpanded ? null : iter.iteration)}
                              className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-muted/20 transition-colors"
                            >
                              {isExpanded
                                ? <ChevronDown className="w-3 h-3 shrink-0 text-muted-foreground" />
                                : <ChevronRight className="w-3 h-3 shrink-0 text-muted-foreground" />}
                              <span className="text-[10px] font-mono text-muted-foreground w-5">#{iter.iteration}</span>
                              {iter.decision === 'kept'
                                ? <Check className="w-3.5 h-3.5 shrink-0 text-emerald-500" />
                                : iter.decision === 'reverted'
                                ? <X className="w-3.5 h-3.5 shrink-0 text-red-400" />
                                : <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin text-muted-foreground" />}
                              <span className="text-[10px] font-mono">{iter.new_score.toFixed(2)}</span>
                              <span className={cn('text-[10px] font-mono',
                                iter.score_delta > 0 ? 'text-emerald-500' : iter.score_delta < 0 ? 'text-red-400' : 'text-muted-foreground')}>
                                {iter.score_delta > 0 ? '+' : ''}{iter.score_delta.toFixed(3)}
                              </span>
                              <span className="flex-1 text-[10px] text-muted-foreground truncate">
                                {paramEntries.length > 0 ? paramEntries.map(([k]) => k).join(', ') : <span className="italic">{t('autoresearchView.noChanges')}</span>}
                              </span>
                              <span className="text-[9px] font-mono text-muted-foreground shrink-0">{iter.duration_seconds.toFixed(0)}s</span>
                            </button>

                            {/* Expanded detail */}
                            {isExpanded && (
                              <div className="border-t border-border/30 px-3 py-2 space-y-2">
                                {/* Reasoning */}
                                {iter.reasoning && (
                                  <div>
                                    <p className="text-[9px] font-medium text-muted-foreground mb-0.5">{t('autoresearchView.llmReasoning')}</p>
                                    <p className="text-[10px] leading-relaxed text-foreground/90 whitespace-pre-wrap">{iter.reasoning}</p>
                                  </div>
                                )}

                                {/* Parameter changes */}
                                {paramEntries.length > 0 && (
                                  <div>
                                    <p className="text-[9px] font-medium text-muted-foreground mb-0.5">{t('autoresearchView.parameterChanges')}</p>
                                    <div className="grid gap-1">
                                      {paramEntries.map(([key, val]) => (
                                        <div key={key} className="flex items-center gap-2 rounded bg-muted/30 px-2 py-1">
                                          <span className="text-[10px] font-mono text-cyan-400">{key}</span>
                                          <ArrowRight className="w-3 h-3 text-muted-foreground shrink-0" />
                                          <span className="text-[10px] font-mono text-foreground">{JSON.stringify(val)}</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    /* ===== Code mode: table with diff viewer ===== */
                    <>
                      <table className="w-full text-[10px]">
                        <thead className="sticky top-0 bg-background/90 backdrop-blur-sm">
                          <tr className="text-left text-muted-foreground">
                            <th className="px-1.5 py-1 font-medium w-8">#</th>
                            <th className="px-1.5 py-1 font-medium w-16">{t('autoresearchView.colScore')}</th>
                            <th className="px-1.5 py-1 font-medium w-14">&Delta;</th>
                            <th className="px-1.5 py-1 font-medium w-10"></th>
                            <th className="px-1.5 py-1 font-medium w-10">{t('autoresearchView.colValid')}</th>
                            <th className="px-1.5 py-1 font-medium">{t('autoresearchView.colChanges')}</th>
                            <th className="px-1.5 py-1 font-medium w-12">{t('autoresearchView.colTime')}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {displayIterations.map((iter, idx) => {
                            const hasDetails = Boolean(iter.reasoning || iter.source_diff || iter.validation_result)
                            return (
                              <tr key={idx}
                                className={cn('border-t border-border/30 hover:bg-muted/30', iter.decision === 'kept' && 'bg-emerald-500/5')}
                                title={iter.reasoning}>
                                <td className="px-1.5 py-1 font-mono text-muted-foreground">{iter.iteration}</td>
                                <td className="px-1.5 py-1 font-mono">{iter.new_score.toFixed(2)}</td>
                                <td className={cn('px-1.5 py-1 font-mono',
                                  iter.score_delta > 0 ? 'text-emerald-500' : iter.score_delta < 0 ? 'text-red-400' : 'text-muted-foreground')}>
                                  {iter.score_delta > 0 ? '+' : ''}{iter.score_delta.toFixed(3)}
                                </td>
                                <td className="px-1.5 py-1">
                                  {iter.decision === 'kept' ? <Check className="w-3.5 h-3.5 text-emerald-500" />
                                    : iter.decision === 'reverted' ? <X className="w-3.5 h-3.5 text-red-400" />
                                    : <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />}
                                </td>
                                <td className="px-1.5 py-1">
                                  {iter.validation_passed === true ? <Check className="w-3 h-3 text-emerald-500" />
                                    : iter.validation_passed === false ? <X className="w-3 h-3 text-red-400" />
                                    : <span className="text-muted-foreground">-</span>}
                                </td>
                                <td className="px-1.5 py-1 text-muted-foreground truncate max-w-[200px]">
                                  {iter.source_diff_lines ? (
                                    <button onClick={() => setExpandedIteration(expandedIteration === iter.iteration ? null : iter.iteration)}
                                      className="text-[10px] text-purple-400 hover:underline">
                                      {t('autoresearchView.linesCount', { n: iter.source_diff_lines })}
                                    </button>
                                  ) : hasDetails ? (
                                    <button onClick={() => setExpandedIteration(expandedIteration === iter.iteration ? null : iter.iteration)}
                                      className="text-[10px] text-purple-400 hover:underline">
                                      {t('autoresearchView.details')}
                                    </button>
                                  ) : <span className="italic">{t('autoresearchView.none')}</span>}
                                </td>
                                <td className="px-1.5 py-1 font-mono text-muted-foreground">{iter.duration_seconds.toFixed(0)}s</td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>

                      {/* Expanded code iteration detail */}
                      {expandedCodeIteration && (
                        <div className="rounded border border-purple-500/30 bg-background/90 p-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[9px] text-muted-foreground">{t('autoresearchView.iterationDetails', { n: expandedCodeIteration.iteration })}</span>
                            <button onClick={() => setExpandedIteration(null)} className="text-[9px] text-muted-foreground hover:text-foreground">{t('autoresearchView.close')}</button>
                          </div>
                          {expandedCodeIteration.reasoning && (
                            <div className="mb-2">
                              <p className="mb-0.5 text-[9px] font-medium text-muted-foreground">{t('autoresearchView.reasoning')}</p>
                              <p className="whitespace-pre-wrap text-[10px] leading-relaxed text-foreground/90">{expandedCodeIteration.reasoning}</p>
                            </div>
                          )}
                          {expandedCodeIteration.validation_result && (
                            <div className="mb-2">
                              <p className="mb-0.5 text-[9px] font-medium text-muted-foreground">{t('autoresearchView.validation')}</p>
                              <p className="text-[10px] text-foreground/90">
                                {expandedCodeIteration.validation_passed === true
                                  ? t('autoresearchView.passed')
                                  : expandedCodeIteration.validation_passed === false
                                  ? t('autoresearchView.failed')
                                  : t('autoresearchView.notRun')}
                              </p>
                              {expandedValidationErrors.length > 0 && (
                                <pre className="mt-1 max-h-[80px] overflow-auto rounded bg-muted/30 p-1 text-[9px] leading-tight text-red-400">
                                  {expandedValidationErrors.map(String).join('\n')}
                                </pre>
                              )}
                            </div>
                          )}
                          {expandedCodeIteration.source_diff && (
                            <pre className="text-[9px] font-mono overflow-auto max-h-[150px] leading-tight">
                              {expandedCodeIteration.source_diff.split('\n').map((line: string, i: number) => (
                                <div key={i} className={cn(
                                  line.startsWith('+') && !line.startsWith('+++') ? 'text-emerald-400' :
                                  line.startsWith('-') && !line.startsWith('---') ? 'text-red-400' :
                                  'text-muted-foreground'
                                )}>{line}</div>
                              ))}
                            </pre>
                          )}
                        </div>
                      )}

                      {/* A/B experiment button */}
                      {!isModeStreaming && Boolean(doneData?.can_create_ab_experiment) && (
                        <div className="flex items-center gap-2 rounded border border-purple-500/30 bg-purple-500/5 px-2 py-1.5">
                          <FlaskRound className="w-3.5 h-3.5 text-purple-400" />
                          <span className="text-[10px] text-muted-foreground flex-1">
                            {t('autoresearchView.bestVersionImproved', { v: String(doneData!.best_version), improvement: String(doneData!.improvement) })}
                          </span>
                          <Button size="sm" variant="outline" className="h-6 px-2 text-[10px]" disabled={abCreating}
                            onClick={async () => {
                              setAbCreating(true)
                              try { await createAbExperimentFromAutoresearch(String(doneData!.experiment_id)); setDoneData(null) } catch { /* */ }
                              setAbCreating(false)
                            }}>
                            {abCreating && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                            {t('autoresearchView.createAbExperiment')}
                          </Button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
