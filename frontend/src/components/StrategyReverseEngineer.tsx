/**
 * Strategy Research → Reverse Engineer tab.
 *
 * Full operator UI for the wallet-strategy reverse-engineer pipeline.
 * Lives in Strategies → Research alongside Code Experiments, Backtest
 * Studio, and Data Lab — strategy domain owns the deliverable, even
 * though the input is a wallet.
 *
 * Layout (split view):
 *   ┌──────────── job list ─────────────┐ ┌──── job detail ────────┐
 *   │  + New job                         │  • profile + score        │
 *   │  RE-1234… wallet 0x… 0.78          │  • iteration timeline     │
 *   │  RE-5678… wallet 0x… running 23%   │  • strategy code viewer   │
 *   │                                    │  • promote / download PDF │
 *   └────────────────────────────────────┘ └───────────────────────┘
 *
 * Polling is automatic (1.5s while a job is running, 30s otherwise).
 * Iterations stream in as the agent submits them.
 */
import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Award,
  Brain,
  Code,
  Copy,
  Database,
  FileDown,
  Layers,
  Loader2,
  Plus,
  Rocket,
  StopCircle,
  Trash2,
  Wallet,
  X,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { ScrollArea } from './ui/scroll-area'
import { cn } from '../lib/utils'
import {
  cancelReverseEngineerJob,
  deleteReverseEngineerJob,
  createReverseEngineerJob,
  getReverseEngineerJob,
  listReverseEngineerIterations,
  listReverseEngineerJobs,
  promoteReverseEngineerJob,
  reverseEngineerPdfUrl,
  type ReverseEngineerIteration,
  type ReverseEngineerJob,
  type ReverseEngineerJobStatus,
} from '../services/apiReverseEngineer'
import { listProviderDatasets, type ProviderDataset } from '../services/apiProviders'
import { listRecordingSessions, type RecordingSession } from '../services/apiDataset'
import { getLLMModels, type LLMModelOption } from '../services/apiSettings'


export interface StrategyReverseEngineerProps {
  /** When set the picker is preloaded with this wallet — used by the
   *  WalletAnalysisPanel deep-link. */
  initialWalletAddress?: string | null
}


// localStorage key for the currently-viewed reverse-engineer job.  The
// jobs themselves live server-side so navigation only needs to remember
// "which one was the user looking at?" — the lifecycle (queued ->
// running -> completed) is the backend's responsibility.
const RE_SELECTED_JOB_KEY = 'hr_reverse_engineer_selected_job_id'

const readSelectedJobId = (): string | null => {
  try {
    return localStorage.getItem(RE_SELECTED_JOB_KEY)
  } catch {
    return null
  }
}
const writeSelectedJobId = (id: string | null) => {
  try {
    if (id) localStorage.setItem(RE_SELECTED_JOB_KEY, id)
    else localStorage.removeItem(RE_SELECTED_JOB_KEY)
  } catch {
    /* silent — quota / disabled storage */
  }
}

export default function StrategyReverseEngineer({
  initialWalletAddress = null,
}: StrategyReverseEngineerProps) {
  const queryClient = useQueryClient()
  // Initialise from localStorage so navigating back to this tab restores
  // the previously-viewed job without flicker.  The job's own data is
  // refetched by jobsQuery on mount; we just need the pointer.
  const [selectedJobId, setSelectedJobIdState] = useState<string | null>(
    () => readSelectedJobId(),
  )
  // Wrap setter so every selection write persists.  Survives tab
  // navigation, page reload, and browser restart (the server-side
  // job_id stays valid for the run's full retention window).
  const setSelectedJobId = (id: string | null) => {
    writeSelectedJobId(id)
    setSelectedJobIdState(id)
  }
  const [showCreate, setShowCreate] = useState<boolean>(!!initialWalletAddress)

  const jobsQuery = useQuery({
    queryKey: ['reverse-engineer', 'jobs'],
    queryFn: () => listReverseEngineerJobs({ limit: 50 }),
    refetchInterval: (q) => {
      const data = q.state.data as ReverseEngineerJob[] | undefined
      const anyActive = (data ?? []).some((j) =>
        ['queued', 'profiling', 'importing_data', 'running'].includes(j.status),
      )
      return anyActive ? 2_000 : 30_000
    },
  })
  const jobs = jobsQuery.data ?? []

  // Auto-select fallback: if storage pointed at a job that no longer
  // exists (deleted, retention window passed), fall back to the most
  // recent.  Otherwise leave the user's selection alone.
  useEffect(() => {
    if (jobs.length === 0) return
    const stillExists = selectedJobId != null && jobs.some((j) => j.id === selectedJobId)
    if (!stillExists) {
      setSelectedJobId(jobs[0].id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs])

  return (
    <div className="flex h-full min-h-0">
      {/* ── job list ── */}
      <div className="flex w-72 shrink-0 flex-col border-r border-border/40">
        <div className="flex items-center justify-between gap-2 border-b border-border/40 px-3 py-2">
          <div className="flex items-center gap-1.5">
            <Brain className="h-4 w-4 text-violet-400" />
            <span className="text-sm font-semibold">Reverse Engineer</span>
          </div>
          <Button
            size="sm"
            className="h-7 gap-1 text-[11px]"
            onClick={() => setShowCreate(true)}
          >
            <Plus className="h-3 w-3" /> New job
          </Button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          {jobsQuery.isLoading ? (
            <div className="flex h-32 items-center justify-center text-[11px] text-muted-foreground">
              <Loader2 className="mr-2 h-3 w-3 animate-spin" /> Loading…
            </div>
          ) : jobs.length === 0 ? (
            <div className="px-3 py-4 text-[11px] text-muted-foreground">
              No reverse-engineer jobs yet. Click <strong>New job</strong> above
              to start one.
            </div>
          ) : (
            <div className="divide-y divide-border/20">
              {jobs.map((job) => (
                <JobListRow
                  key={job.id}
                  job={job}
                  active={selectedJobId === job.id}
                  onSelect={() => setSelectedJobId(job.id)}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </div>

      {/* ── detail or create ── */}
      <div className="min-w-0 flex-1">
        {showCreate ? (
          <CreateJobView
            initialWalletAddress={initialWalletAddress}
            onClose={() => setShowCreate(false)}
            onCreated={(job) => {
              setSelectedJobId(job.id)
              setShowCreate(false)
              queryClient.invalidateQueries({ queryKey: ['reverse-engineer', 'jobs'] })
            }}
          />
        ) : selectedJobId ? (
          <JobDetailView jobId={selectedJobId} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            Pick a job on the left or click <strong className="mx-1">+ New job</strong> to start.
          </div>
        )}
      </div>
    </div>
  )
}


function statusBadgeClass(status: ReverseEngineerJobStatus): string {
  switch (status) {
    case 'completed':
      return 'border-emerald-500/40 text-emerald-700 dark:text-emerald-300'
    case 'failed':
      return 'border-rose-500/40 text-rose-700 dark:text-rose-300'
    case 'cancelled':
      return 'border-zinc-500/40 text-zinc-300'
    case 'running':
    case 'profiling':
    case 'importing_data':
      return 'border-blue-500/40 text-blue-700 dark:text-blue-300'
    default:
      return 'border-amber-500/40 text-amber-700 dark:text-amber-300'
  }
}


function JobListRow({
  job,
  active,
  onSelect,
}: {
  job: ReverseEngineerJob
  active: boolean
  onSelect: () => void
}) {
  const pct = Math.max(0, Math.min(1, job.progress)) * 100
  const isActive = ['queued', 'profiling', 'importing_data', 'running'].includes(job.status)
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'block w-full px-3 py-2 text-left transition-colors',
        active ? 'bg-violet-500/10' : 'hover:bg-card/40',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-mono text-[10.5px] text-muted-foreground">{job.id}</span>
        <Badge variant="outline" className={cn('text-[9px]', statusBadgeClass(job.status))}>
          {job.status}
        </Badge>
      </div>
      <div className="mt-0.5 truncate text-[11px] font-medium">
        {job.label || `wallet ${job.wallet_address.slice(0, 6)}…${job.wallet_address.slice(-4)}`}
      </div>
      <div className="mt-0.5 flex items-center justify-between text-[10px] text-muted-foreground">
        {job.report_mode === 'report' ? (
          <>
            <span>report mode</span>
            <span>${(job.total_cost_usd ?? 0).toFixed(4)}</span>
          </>
        ) : (
          <>
            <span>iter {job.current_iteration}/{job.max_iterations}</span>
            <span>{job.best_score != null ? `score ${(job.best_score * 100).toFixed(1)}%` : '—'}</span>
          </>
        )}
      </div>
      {isActive ? (
        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-card/40">
          <div
            className="h-full rounded-full bg-violet-500/60 transition-all"
            style={{ width: `${pct.toFixed(1)}%` }}
          />
        </div>
      ) : null}
    </button>
  )
}


// ─── Create job ──────────────────────────────────────────────────────


function CreateJobView({
  initialWalletAddress,
  onClose,
  onCreated,
}: {
  initialWalletAddress?: string | null
  onClose: () => void
  onCreated: (job: ReverseEngineerJob) => void
}) {
  const [walletAddress, setWalletAddress] = useState<string>(initialWalletAddress ?? '')
  const [label, setLabel] = useState<string>('')
  // Default to the iterative agent ("Deep reverse-engineer") — that's
  // the actual reverse-engineering process, not the 10-second
  // analytical-report deterministic path.  Operators consistently
  // click into the dialog expecting the iterative depth and were
  // surprised to get a 10s "100% complete" report; making the deep
  // mode the default closes that mismatch.
  const [reportMode, setReportMode] = useState<'report' | 'strategy_seed'>('strategy_seed')
  const [dataSourceKind, setDataSourceKind] = useState<
    'auto' | 'recording_session' | 'provider_dataset'
  >('auto')
  const [providerDatasetIds, setProviderDatasetIds] = useState<string[]>([])
  const [recordingSessionIds, setRecordingSessionIds] = useState<string[]>([])
  const [llmModel, setLlmModel] = useState<string>('')
  const [maxIterations, setMaxIterations] = useState<string>('10')
  const [targetScore, setTargetScore] = useState<string>('0.7')
  const [maxCostUsd, setMaxCostUsd] = useState<string>('')
  const [maxWalletTrades, setMaxWalletTrades] = useState<string>('2000')

  const modelsQuery = useQuery({
    queryKey: ['llm-models'],
    queryFn: () => getLLMModels(),
    staleTime: 5 * 60_000,
  })
  // The LLM-models endpoint returns { models: { provider: [...] } } —
  // flatten across providers for the dropdown so the operator can pick
  // any model regardless of which provider supplies it.
  const models: LLMModelOption[] = useMemo(() => {
    const out: LLMModelOption[] = []
    const grouped = modelsQuery.data?.models ?? {}
    for (const list of Object.values(grouped)) {
      if (Array.isArray(list)) out.push(...list)
    }
    return out
  }, [modelsQuery.data])

  const datasetsQuery = useQuery({
    queryKey: ['providers', 'datasets', 'reverse-engineer-picker'],
    queryFn: () => listProviderDatasets({ limit: 200 }),
    staleTime: 60_000,
  })
  const datasets: ProviderDataset[] = datasetsQuery.data ?? []

  const sessionsQuery = useQuery({
    queryKey: ['recording-sessions', 'reverse-engineer-picker'],
    queryFn: () => listRecordingSessions(['running', 'completed']),
    staleTime: 60_000,
  })
  const sessions: RecordingSession[] = sessionsQuery.data ?? []

  const createMutation = useMutation({
    mutationFn: createReverseEngineerJob,
    onSuccess: onCreated,
  })

  const submit = () => {
    createMutation.mutate({
      wallet_address: walletAddress.trim(),
      label: label.trim() || undefined,
      report_mode: reportMode,
      data_source_kind: dataSourceKind,
      provider_dataset_ids: dataSourceKind === 'provider_dataset' ? providerDatasetIds : undefined,
      recording_session_ids: dataSourceKind === 'recording_session' ? recordingSessionIds : undefined,
      llm_model: llmModel.trim() || undefined,
      max_iterations: maxIterations ? parseInt(maxIterations, 10) : undefined,
      target_score: targetScore ? parseFloat(targetScore) : undefined,
      max_cost_usd: maxCostUsd ? parseFloat(maxCostUsd) : undefined,
      max_wallet_trades: maxWalletTrades ? parseInt(maxWalletTrades, 10) : undefined,
    })
  }

  const error = createMutation.isError
    ? (createMutation.error as { response?: { data?: { detail?: string } }; message?: string })
    : null

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-3xl space-y-4 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold">New reverse-engineer job</h2>
            <p className="text-[11px] text-muted-foreground">
              The agent will pull the wallet's full trade history, pick a dataset,
              iteratively write a strategy that mimics it, and backtest until the
              composite score plateaus.
            </p>
          </div>
          <Button size="sm" variant="ghost" onClick={onClose}>
            <X className="h-3 w-3" />
          </Button>
        </div>

        <div className="rounded-md border border-border/40 bg-card/40 p-3 space-y-2">
          <div>
            <Label className="text-[10px] uppercase text-muted-foreground">Wallet address</Label>
            <Input
              value={walletAddress}
              onChange={(e) => setWalletAddress(e.target.value)}
              placeholder="0x…"
              className="h-8 font-mono text-xs"
            />
          </div>
          <div>
            <Label className="text-[10px] uppercase text-muted-foreground">Label (optional)</Label>
            <Input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Whale BTC mimic"
              className="h-8 text-xs"
            />
          </div>
        </div>

        {/* ── Mode picker — what kind of deliverable to produce ── */}
        <div className="rounded-md border border-border/40 bg-card/40 p-3 space-y-2">
          <div>
            <div className="text-xs font-semibold flex items-center gap-1.5">
              <FileDown className="h-3.5 w-3.5 text-violet-400" />
              Deliverable
            </div>
            <p className="text-[10.5px] text-muted-foreground">
              Pick what the job produces.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-1">
            {(
              [
                {
                  key: 'strategy_seed',
                  label: 'Deep reverse-engineer',
                  hint: 'Iterative LLM agent: hypothesizes → writes a BaseStrategy class → backtests → critiques → refines until target_score or max_iterations.  This is the actual reverse-engineering process.',
                  eta: '5–30 min · $0.10–$2.00',
                  recommended: true,
                },
                {
                  key: 'report',
                  label: 'Quick analytical report',
                  hint: 'Deterministic stat tables (two-leg P/L decomposition, dominance buckets, filter ledger) + one LLM call per section.  Useful for a fast PDF brief; NOT a backtest of any strategy.',
                  eta: '5–15 sec · $0.01–$0.05',
                  recommended: false,
                },
              ] as const
            ).map((opt) => (
              <button
                key={opt.key}
                type="button"
                onClick={() => setReportMode(opt.key)}
                className={cn(
                  'rounded-sm border px-2 py-1.5 text-left text-[11px] transition-colors',
                  reportMode === opt.key
                    ? 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                    : 'border-border/40 hover:bg-card/40',
                )}
              >
                <div className="flex items-center gap-1.5">
                  <span className="font-semibold">{opt.label}</span>
                  {opt.recommended ? (
                    <span className="rounded-sm bg-emerald-500/15 px-1.5 py-0.5 text-[8.5px] font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                      recommended
                    </span>
                  ) : null}
                </div>
                <div className="mt-0.5 text-[9.5px] text-muted-foreground leading-tight">{opt.hint}</div>
                <div className="mt-1 font-mono text-[9px] text-muted-foreground/80">{opt.eta}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-border/40 bg-card/40 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold flex items-center gap-1.5">
                <Database className="h-3.5 w-3.5 text-violet-400" />
                Data source
              </div>
              <p className="text-[10.5px] text-muted-foreground">
                {reportMode === 'report'
                  ? 'Optional — analytical reports use the wallet trade history directly. Provider/session data is used for deep dives when available.'
                  : 'What market data the agent backtests against.'}
              </p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {(
              [
                { key: 'auto', label: 'Auto', hint: 'Wallet trade window, any data we have' },
                { key: 'provider_dataset', label: 'Provider dataset', hint: 'Polybacktest etc.' },
                { key: 'recording_session', label: 'Recording session', hint: 'A captured local session' },
              ] as const
            ).map((opt) => (
              <button
                key={opt.key}
                type="button"
                onClick={() => setDataSourceKind(opt.key)}
                className={cn(
                  'rounded-sm border px-2 py-1.5 text-left text-[11px] transition-colors',
                  dataSourceKind === opt.key
                    ? 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                    : 'border-border/40 hover:bg-card/40',
                )}
              >
                <div className="font-semibold">{opt.label}</div>
                <div className="text-[9.5px] text-muted-foreground">{opt.hint}</div>
              </button>
            ))}
          </div>
          {dataSourceKind === 'provider_dataset' ? (
            <div className="rounded-sm border border-border/30 bg-background/40 p-2 max-h-44 overflow-auto">
              {datasets.length === 0 ? (
                <div className="text-[11px] text-muted-foreground">
                  No imported datasets — go to Data Lab → Providers first.
                </div>
              ) : (
                datasets.map((d) => {
                  const checked = providerDatasetIds.includes(d.id)
                  return (
                    <label key={d.id} className="flex items-center gap-2 py-0.5 text-[11px]">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setProviderDatasetIds([...providerDatasetIds, d.id])
                          } else {
                            setProviderDatasetIds(providerDatasetIds.filter((x) => x !== d.id))
                          }
                        }}
                        className="h-3 w-3 accent-violet-500"
                      />
                      <span className="flex-1 truncate">
                        {d.title || d.external_slug || d.external_id}
                      </span>
                      <span className="text-[9px] text-muted-foreground">
                        {(d.coin || '?').toUpperCase()} · {d.snapshot_count.toLocaleString()}
                      </span>
                    </label>
                  )
                })
              )}
            </div>
          ) : null}
          {dataSourceKind === 'recording_session' ? (
            <div className="rounded-sm border border-border/30 bg-background/40 p-2 max-h-44 overflow-auto">
              {sessions.length === 0 ? (
                <div className="text-[11px] text-muted-foreground">
                  No recording sessions — start one in Data Lab → Record.
                </div>
              ) : (
                sessions.map((s) => {
                  const checked = recordingSessionIds.includes(s.id)
                  return (
                    <label key={s.id} className="flex items-center gap-2 py-0.5 text-[11px]">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setRecordingSessionIds([...recordingSessionIds, s.id])
                          } else {
                            setRecordingSessionIds(
                              recordingSessionIds.filter((x) => x !== s.id),
                            )
                          }
                        }}
                        className="h-3 w-3 accent-violet-500"
                      />
                      <span className="flex-1 truncate">{s.name}</span>
                      <span className="text-[9px] text-muted-foreground">{s.status}</span>
                    </label>
                  )
                })
              )}
            </div>
          ) : null}
          {dataSourceKind === 'auto' ? (
            <div className="rounded-sm border border-amber-500/30 bg-amber-500/5 p-2 text-[10.5px] text-amber-700 dark:text-amber-200">
              Auto mode uses whatever microstructure data you already have for the
              wallet's trade window. Best-effort with possible coverage gaps. For
              crypto wallets prefer a Polybacktest provider dataset; for niche
              event markets capture a recording session first.
            </div>
          ) : null}
        </div>

        <div className="rounded-md border border-border/40 bg-card/40 p-3 grid grid-cols-2 gap-3">
          <div>
            <Label className="text-[10px] uppercase text-muted-foreground">Max iterations</Label>
            <Input
              value={maxIterations}
              onChange={(e) => setMaxIterations(e.target.value)}
              className="h-8 text-xs"
              placeholder="10"
            />
          </div>
          <div>
            <Label className="text-[10px] uppercase text-muted-foreground">Target score</Label>
            <Input
              value={targetScore}
              onChange={(e) => setTargetScore(e.target.value)}
              className="h-8 text-xs"
              placeholder="0.7"
            />
          </div>
          <div>
            <Label className="text-[10px] uppercase text-muted-foreground">Max cost (USD)</Label>
            <Input
              value={maxCostUsd}
              onChange={(e) => setMaxCostUsd(e.target.value)}
              className="h-8 text-xs"
              placeholder="(no cap)"
            />
          </div>
          <div>
            <Label className="text-[10px] uppercase text-muted-foreground">Max wallet trades</Label>
            <Input
              value={maxWalletTrades}
              onChange={(e) => setMaxWalletTrades(e.target.value)}
              className="h-8 text-xs"
              placeholder="50000"
            />
          </div>
          <div className="col-span-2">
            <Label className="text-[10px] uppercase text-muted-foreground">
              Model override (optional — defaults to AI → Models → "Strategy Reverse-Engineer")
            </Label>
            <Select
              value={llmModel || '__default__'}
              onValueChange={(v) => setLlmModel(v === '__default__' ? '' : v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="(use AI → Models default)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__default__" className="text-xs">
                  (use AI → Models default)
                </SelectItem>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id} className="text-xs">
                    {m.name || m.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {error ? (
          <div className="rounded-sm border border-rose-500/30 bg-rose-500/5 p-2 text-[11px] text-rose-700 dark:text-rose-300">
            {error.response?.data?.detail || error.message || 'Failed'}
          </div>
        ) : null}

        <div className="flex items-center justify-end gap-2">
          <Button size="sm" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={submit}
            disabled={!walletAddress.trim() || createMutation.isPending}
            className="gap-1.5"
          >
            {createMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Rocket className="h-3 w-3" />
            )}
            Start reverse-engineer
          </Button>
        </div>
      </div>
    </ScrollArea>
  )
}


// ─── Job detail ─────────────────────────────────────────────────────


function JobDetailView({ jobId }: { jobId: string }) {
  const queryClient = useQueryClient()
  const jobQuery = useQuery({
    queryKey: ['reverse-engineer', 'job', jobId],
    queryFn: () => getReverseEngineerJob(jobId),
    refetchInterval: (q) => {
      const data = q.state.data as ReverseEngineerJob | undefined
      const active = data && ['queued', 'profiling', 'importing_data', 'running'].includes(data.status)
      return active ? 1_500 : 30_000
    },
  })
  const iterationsQuery = useQuery({
    queryKey: ['reverse-engineer', 'iterations', jobId],
    queryFn: () => listReverseEngineerIterations(jobId),
    refetchInterval: 2_000,
  })
  const cancelMutation = useMutation({
    mutationFn: () => cancelReverseEngineerJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reverse-engineer'] })
    },
  })
  const deleteMutation = useMutation({
    mutationFn: () => deleteReverseEngineerJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reverse-engineer'] })
    },
  })

  const handleDelete = () => {
    const yes = window.confirm(
      `Delete reverse-engineer job ${jobId}?  This wipes the job and all its iteration rows.  This cannot be undone.`,
    )
    if (!yes) return
    deleteMutation.mutate()
  }

  const job = jobQuery.data ?? null
  const iterations: ReverseEngineerIteration[] = iterationsQuery.data ?? []
  const [selectedIterationId, setSelectedIterationId] = useState<string | null>(null)

  // Auto-select the best iteration when one becomes available.
  useEffect(() => {
    if (selectedIterationId == null && job?.best_iteration_id) {
      setSelectedIterationId(job.best_iteration_id)
    }
  }, [job?.best_iteration_id, selectedIterationId])

  const selectedIteration = useMemo(
    () => iterations.find((i) => i.id === selectedIterationId) ?? null,
    [iterations, selectedIterationId],
  )

  if (jobQuery.isLoading || !job) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        <Loader2 className="mr-2 h-3 w-3 animate-spin" /> Loading…
      </div>
    )
  }

  const isActive = ['queued', 'profiling', 'importing_data', 'running'].includes(job.status)

  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 p-4">
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-base font-semibold">
                {job.label || `Wallet ${job.wallet_address.slice(0, 8)}…`}
              </h2>
              <Badge variant="outline" className={cn('text-[10px]', statusBadgeClass(job.status))}>
                {job.status}
              </Badge>
            </div>
            <div className="mt-0.5 flex items-center gap-2 text-[10.5px] text-muted-foreground">
              <span className="font-mono">{job.id}</span>
              <span>·</span>
              <Wallet className="h-3 w-3" />
              <span className="font-mono">{job.wallet_address}</span>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {isActive ? (
              <Button
                size="sm"
                variant="outline"
                className="h-7 gap-1 border-rose-500/30 text-rose-700 dark:text-rose-300 hover:bg-rose-500/10"
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
              >
                <StopCircle className="h-3 w-3" /> Cancel
              </Button>
            ) : null}
            {job.status === 'completed' && job.best_strategy_code ? (
              <>
                <a
                  href={reverseEngineerPdfUrl(job.id)}
                  target="_blank"
                  rel="noreferrer"
                >
                  <Button size="sm" variant="outline" className="h-7 gap-1 text-[11px]">
                    <FileDown className="h-3 w-3" /> PDF report
                  </Button>
                </a>
                <PromoteButton jobId={job.id} suggestedSlug={(job.best_strategy_class || '').toLowerCase()} />
              </>
            ) : null}
            {/* Delete — allowed in any status.  Confirms first, then
                wipes job + iteration rows.  Useful when a run got
                stuck or the operator wants the queue clean. */}
            <Button
              size="sm"
              variant="outline"
              className="h-7 gap-1 text-[11px] border-rose-500/30 text-rose-700 dark:text-rose-300 hover:bg-rose-500/10"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
              title="Delete this run and all its iterations"
            >
              <Trash2 className="h-3 w-3" /> Delete
            </Button>
          </div>
        </div>

        {/* Activity / progress */}
        {isActive ? (
          <div className="rounded-md border border-blue-500/30 bg-blue-500/5 p-2.5">
            <div className="flex items-center gap-2 text-[11px]">
              <Loader2 className="h-3 w-3 animate-spin text-blue-700 dark:text-blue-300" />
              <span className="font-medium">{job.activity || 'Working…'}</span>
              <span className="ml-auto text-muted-foreground">
                {(job.progress * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-card/40">
              <div
                className="h-full rounded-full bg-blue-500/60 transition-all"
                style={{ width: `${(job.progress * 100).toFixed(1)}%` }}
              />
            </div>
          </div>
        ) : null}

        {job.error ? (
          <div className="rounded-md border border-rose-500/30 bg-rose-500/5 p-2.5 text-[11px] text-rose-700 dark:text-rose-300">
            <div className="font-semibold">Error</div>
            <div className="mt-0.5">{job.error}</div>
          </div>
        ) : null}

        {/* Score card.  Report-mode jobs run a deterministic analytics
            pipeline + LLM section drafters — there's no iterative
            optimization, so the composite-score gauge and target are
            irrelevant.  Show "Sections / Tokens / Cost" instead. */}
        {job.report_mode === 'report' ? (
          <div className="rounded-md border border-border/40 bg-card/40 p-3">
            <div className="flex items-end justify-between">
              <div>
                <div className="text-[10px] uppercase text-muted-foreground">Analytical report</div>
                <div className="text-lg font-semibold">
                  {job.status === 'completed' ? 'Sections drafted' : 'Pipeline running'}
                </div>
              </div>
              <div className="text-right text-[10.5px] text-muted-foreground">
                <div>tokens in/out: {(job.total_input_tokens ?? 0).toLocaleString()} / {(job.total_output_tokens ?? 0).toLocaleString()}</div>
                <div>${(job.total_cost_usd ?? 0).toFixed(4)} spent</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-md border border-border/40 bg-card/40 p-3">
            <div className="flex items-end justify-between">
              <div>
                <div className="text-[10px] uppercase text-muted-foreground">Composite score</div>
                <div className="text-2xl font-bold">
                  {job.best_score != null ? `${(job.best_score * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
              <div className="text-right text-[10.5px] text-muted-foreground">
                <div>iter {job.current_iteration}/{job.max_iterations}</div>
                <div>target {(job.target_score * 100).toFixed(0)}%</div>
                <div>${job.total_cost_usd.toFixed(4)} spent</div>
              </div>
            </div>
          </div>
        )}

        {/* Wallet profile summary */}
        {job.wallet_profile?.summary ? (
          <div className="rounded-md border border-border/40 bg-card/40 p-3">
            <div className="text-xs font-semibold mb-1">Wallet profile</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Trades</span>
                <span>{job.wallet_profile.summary.trade_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Markets</span>
                <span>{job.wallet_profile.summary.unique_markets}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg notional</span>
                <span>
                  {job.wallet_profile.summary.notional?.mean != null
                    ? `$${job.wallet_profile.summary.notional.mean.toFixed(2)}`
                    : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Median gap (s)</span>
                <span>
                  {job.wallet_profile.summary.inter_trade_seconds?.median != null
                    ? job.wallet_profile.summary.inter_trade_seconds.median.toFixed(0)
                    : '—'}
                </span>
              </div>
            </div>
          </div>
        ) : null}

        {/* Iterations */}
        <div className="rounded-md border border-border/40 bg-card/40 p-3">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold">
            <Layers className="h-3.5 w-3.5 text-violet-400" />
            Iterations ({iterations.length})
          </div>
          {iterations.length === 0 ? (
            <div className="py-3 text-center text-[11px] text-muted-foreground">
              No iterations yet. The agent submits one per candidate strategy.
            </div>
          ) : (
            <div className="space-y-1">
              {iterations.map((it) => (
                <IterationRow
                  key={it.id}
                  iteration={it}
                  active={selectedIterationId === it.id}
                  isBest={job.best_iteration_id === it.id}
                  onSelect={() => setSelectedIterationId(it.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Selected iteration detail */}
        {selectedIteration ? <IterationDetail iteration={selectedIteration} /> : null}

        {/* Run metadata */}
        <RunMetadata job={job} />

        {/* Best strategy / analytical report */}
        {job.best_strategy_code ? (
          job.best_strategy_class === 'AnalyticalReport' ? (
            <AnalyticalReportView job={job} />
          ) : (
            <StrategySeedView job={job} />
          )
        ) : null}
      </div>
    </ScrollArea>
  )
}


// ─── Run metadata ──────────────────────────────────────────────────


function fmtDate(s: string | null): string {
  if (!s) return '—'
  try {
    return new Date(s).toLocaleString()
  } catch {
    return s
  }
}

function fmtDuration(start: string | null, end: string | null): string {
  if (!start || !end) return '—'
  try {
    const ms = new Date(end).getTime() - new Date(start).getTime()
    if (!isFinite(ms) || ms < 0) return '—'
    const sec = Math.round(ms / 1000)
    if (sec < 60) return `${sec}s`
    const min = Math.floor(sec / 60)
    const remSec = sec % 60
    if (min < 60) return `${min}m ${remSec}s`
    const hr = Math.floor(min / 60)
    const remMin = min % 60
    return `${hr}h ${remMin}m`
  } catch {
    return '—'
  }
}

function RunMetadata({ job }: { job: ReverseEngineerJob }) {
  const totalTokens = (job.total_input_tokens || 0) + (job.total_output_tokens || 0)
  const isReport = job.report_mode === 'report'
  return (
    <div className="rounded-md border border-border/40 bg-card/40 p-3">
      <div className="mb-2 text-xs font-semibold">Run metadata</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] md:grid-cols-3">
        <MetaRow label="Mode" value={job.report_mode} mono />
        <MetaRow label="Model" value={job.llm_model || '—'} mono />
        <MetaRow label="Best class" value={job.best_strategy_class || '—'} mono />
        <MetaRow label="Cost" value={`$${(job.total_cost_usd || 0).toFixed(4)}`} />
        <MetaRow label="Tokens" value={totalTokens.toLocaleString()} />
        <MetaRow
          label="In/Out tokens"
          value={`${(job.total_input_tokens || 0).toLocaleString()} / ${(job.total_output_tokens || 0).toLocaleString()}`}
        />
        {/* Iterations + score widgets only meaningful in strategy_seed
            mode.  Report mode runs a single deterministic pass + N
            section drafters in parallel, so showing "1/10" or "70%
            target" is misleading. */}
        {!isReport && (
          <MetaRow label="Iterations" value={`${job.current_iteration} / ${job.max_iterations}`} />
        )}
        {!isReport && (
          <MetaRow label="Target score" value={`${(job.target_score * 100).toFixed(0)}%`} />
        )}
        {!isReport && (
          <MetaRow
            label="Best score"
            value={job.best_score != null ? `${(job.best_score * 100).toFixed(1)}%` : '—'}
          />
        )}
        <MetaRow label="Created" value={fmtDate(job.created_at)} />
        <MetaRow label="Started" value={fmtDate(job.started_at)} />
        <MetaRow label="Finished" value={fmtDate(job.finished_at)} />
        <MetaRow
          label="Duration"
          value={fmtDuration(job.started_at, job.finished_at)}
        />
        <MetaRow
          label="Wallet trades"
          value={(job.wallet_trade_count || 0).toLocaleString()}
        />
        <MetaRow
          label="Trade window"
          value={
            job.wallet_window_start && job.wallet_window_end
              ? `${fmtDate(job.wallet_window_start)} → ${fmtDate(job.wallet_window_end)}`
              : '—'
          }
        />
      </div>
    </div>
  )
}

function MetaRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between gap-2 truncate">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn('truncate text-right', mono && 'font-mono text-[10.5px]')}>{value}</span>
    </div>
  )
}


// ─── Analytical report renderer ────────────────────────────────────


interface ReportSections {
  headline_oneliner?: string
  at_a_glance?: string
  analysis_narrative?: string
  two_leg_explainer?: string
  dominance_explainer?: string
  filter_recommendation?: string
  playbook_brief?: string
  what_to_copy?: string[]
  what_not_to_copy?: string[]
  pseudocode?: string
  bankroll_paragraph?: string
}

interface AnalyticalPayload {
  mode?: string
  sections?: ReportSections
  spotlight?: Record<string, unknown> | null
}

function downloadFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function AnalyticalReportView({ job }: { job: ReverseEngineerJob }) {
  const payload = useMemo<AnalyticalPayload | null>(() => {
    if (!job.best_strategy_code) return null
    try {
      return JSON.parse(job.best_strategy_code) as AnalyticalPayload
    } catch {
      return null
    }
  }, [job.best_strategy_code])

  if (!payload) {
    return (
      <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-[11px] text-amber-700 dark:text-amber-300">
        Failed to parse analytical report payload — try the PDF download instead.
      </div>
    )
  }

  const s = payload.sections || {}
  const pseudocode = s.pseudocode || ''

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
        <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold">
          <Award className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-300" />
          Analytical report
          {job.promoted_strategy_id ? (
            <Badge
              variant="outline"
              className="ml-auto border-emerald-500/40 text-emerald-700 dark:text-emerald-300 text-[9px]"
            >
              Promoted: {job.promoted_strategy_id}
            </Badge>
          ) : null}
        </div>

        {s.headline_oneliner ? (
          <div className="mb-3 rounded-sm border border-emerald-500/20 bg-card/40 p-2.5 text-[12.5px] italic leading-snug">
            “{s.headline_oneliner}”
          </div>
        ) : null}

        <ReportSection title="At a glance" body={s.at_a_glance} />
        <ReportSection title="Analysis" body={s.analysis_narrative} />
        <ReportSection title="Two-leg decomposition" body={s.two_leg_explainer} />
        <ReportSection title="Dominance buckets" body={s.dominance_explainer} />
        <ReportSection title="Filter recommendation" body={s.filter_recommendation} />
        <ReportSection title="Playbook brief" body={s.playbook_brief} />
        <ReportSection title="Bankroll" body={s.bankroll_paragraph} />

        <BulletSection
          title="What to copy"
          items={s.what_to_copy}
          tone="positive"
        />
        <BulletSection
          title="What not to copy"
          items={s.what_not_to_copy}
          tone="negative"
        />
      </div>

      {pseudocode ? (
        <div className="rounded-md border border-violet-500/30 bg-violet-500/5 p-3">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold">
            <Code className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
            Pseudocode
            <div className="ml-auto flex items-center gap-1">
              <Button
                size="sm"
                variant="outline"
                className="h-6 gap-1 text-[10px]"
                onClick={() => navigator.clipboard?.writeText(pseudocode)}
              >
                <Copy className="h-3 w-3" /> Copy
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="h-6 gap-1 text-[10px]"
                onClick={() =>
                  downloadFile(
                    `reverse_engineer_${job.id}_pseudocode.py`,
                    pseudocode,
                    'text/x-python',
                  )
                }
              >
                <FileDown className="h-3 w-3" /> .py
              </Button>
            </div>
          </div>
          <pre className="max-h-96 overflow-auto rounded-sm bg-zinc-950 p-3 font-mono text-[10.5px] leading-relaxed text-zinc-200">
            {pseudocode}
          </pre>
        </div>
      ) : null}
    </div>
  )
}

function ReportSection({ title, body }: { title: string; body?: string }) {
  if (!body) return null
  return (
    <div className="mb-3">
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </div>
      <div className="whitespace-pre-wrap text-[11.5px] leading-relaxed text-foreground/90">
        {body}
      </div>
    </div>
  )
}

function BulletSection({
  title,
  items,
  tone,
}: {
  title: string
  items?: string[]
  tone: 'positive' | 'negative'
}) {
  if (!items || items.length === 0) return null
  const dotClass =
    tone === 'positive'
      ? 'bg-emerald-500/70'
      : 'bg-rose-500/70'
  return (
    <div className="mb-3">
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </div>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2 text-[11.5px] leading-relaxed text-foreground/90">
            <span className={cn('mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full', dotClass)} />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}


// ─── Strategy-seed (Python) view ───────────────────────────────────


function StrategySeedView({ job }: { job: ReverseEngineerJob }) {
  const code = job.best_strategy_code || ''
  return (
    <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold">
        <Award className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-300" />
        Best strategy
        {job.best_strategy_class ? (
          <code className="rounded bg-card/40 px-1 py-0.5 font-mono text-[10px]">
            {job.best_strategy_class}
          </code>
        ) : null}
        <div className="ml-auto flex items-center gap-1">
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 text-[10px]"
            onClick={() => navigator.clipboard?.writeText(code)}
          >
            <Copy className="h-3 w-3" /> Copy
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 text-[10px]"
            onClick={() =>
              downloadFile(
                `reverse_engineer_${job.id}_${(job.best_strategy_class || 'strategy').toLowerCase()}.py`,
                code,
                'text/x-python',
              )
            }
          >
            <FileDown className="h-3 w-3" /> .py
          </Button>
          {job.promoted_strategy_id ? (
            <Badge
              variant="outline"
              className="border-emerald-500/40 text-emerald-700 dark:text-emerald-300 text-[9px]"
            >
              Promoted: {job.promoted_strategy_id}
            </Badge>
          ) : null}
        </div>
      </div>
      <pre className="max-h-96 overflow-auto rounded-sm bg-zinc-950 p-3 font-mono text-[10.5px] leading-relaxed text-zinc-200">
        {code}
      </pre>
    </div>
  )
}


function IterationRow({
  iteration,
  active,
  isBest,
  onSelect,
}: {
  iteration: ReverseEngineerIteration
  active: boolean
  isBest: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'block w-full rounded-sm border px-2 py-1.5 text-left text-[11px] transition-colors',
        active
          ? 'border-violet-500/40 bg-violet-500/10'
          : 'border-border/30 hover:bg-card/40',
      )}
    >
      <div className="flex items-center gap-2">
        <span className="font-semibold">#{iteration.iteration}</span>
        <code className="font-mono text-[10px] text-muted-foreground truncate flex-1">
          {iteration.strategy_class || '—'}
        </code>
        <Badge variant="outline" className={cn('text-[9px]', statusBadgeClass(iteration.status as ReverseEngineerJobStatus))}>
          {iteration.status}
        </Badge>
        {isBest ? (
          <Badge variant="outline" className="text-[9px] border-emerald-500/40 text-emerald-700 dark:text-emerald-300">
            best
          </Badge>
        ) : null}
        <span className="ml-1 font-mono">
          {iteration.score != null ? `${(iteration.score * 100).toFixed(1)}%` : '—'}
        </span>
      </div>
      {iteration.divergence_summary ? (
        <div className="mt-0.5 truncate text-[9.5px] font-mono text-muted-foreground">
          {iteration.divergence_summary}
        </div>
      ) : null}
    </button>
  )
}


function IterationDetail({ iteration }: { iteration: ReverseEngineerIteration }) {
  const breakdown = iteration.score_breakdown
  return (
    <div className="rounded-md border border-border/40 bg-card/40 p-3 space-y-2">
      <div className="text-xs font-semibold">
        Iteration #{iteration.iteration} —{' '}
        <code className="font-mono text-[11px]">{iteration.strategy_class || '?'}</code>
      </div>
      {iteration.error ? (
        <div className="rounded-sm border border-rose-500/30 bg-rose-500/5 p-2 text-[10.5px] text-rose-700 dark:text-rose-300">
          {iteration.error}
        </div>
      ) : null}
      {breakdown ? (
        <div className="grid grid-cols-4 gap-2">
          <ScoreTile label="Trade overlap" value={breakdown.trade_overlap_pct * 100} suffix="%" />
          <ScoreTile label="Side agreement" value={breakdown.side_agreement_pct * 100} suffix="%" />
          <ScoreTile label="PnL corr" value={breakdown.pnl_correlation} suffix="" digits={3} />
          <ScoreTile label="Frequency" value={breakdown.frequency_match * 100} suffix="%" />
        </div>
      ) : null}
      {iteration.notes ? (
        <div className="text-[10.5px] text-muted-foreground">
          <span className="font-semibold">Notes:</span> {iteration.notes}
        </div>
      ) : null}
      {iteration.strategy_code ? (
        <details className="rounded-sm border border-border/30">
          <summary className="cursor-pointer px-2 py-1 text-[10.5px] hover:bg-card/40">
            View strategy code
          </summary>
          <pre className="max-h-72 overflow-auto bg-zinc-950 p-3 font-mono text-[10.5px] leading-relaxed text-zinc-200">
            {iteration.strategy_code}
          </pre>
        </details>
      ) : null}
    </div>
  )
}


function ScoreTile({
  label,
  value,
  suffix,
  digits = 1,
}: {
  label: string
  value: number | null
  suffix: string
  digits?: number
}) {
  return (
    <div className="rounded-sm border border-border/30 p-2 text-center">
      <div className="text-[9.5px] uppercase text-muted-foreground">{label}</div>
      <div className="text-sm font-bold">
        {value != null ? `${value.toFixed(digits)}${suffix}` : '—'}
      </div>
    </div>
  )
}


// ─── Promote ────────────────────────────────────────────────────────


function PromoteButton({ jobId, suggestedSlug }: { jobId: string; suggestedSlug: string }) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [name, setName] = useState<string>('')
  const [slug, setSlug] = useState<string>(suggestedSlug || '')
  const [enabled, setEnabled] = useState(false)

  const mutation = useMutation({
    mutationFn: () =>
      promoteReverseEngineerJob(jobId, {
        name: name.trim(),
        slug: slug.trim().toLowerCase().replace(/\s+/g, '_'),
        enabled,
      }),
    onSuccess: () => {
      setOpen(false)
      queryClient.invalidateQueries({ queryKey: ['reverse-engineer', 'job', jobId] })
    },
  })

  if (!open) {
    return (
      <Button
        size="sm"
        className="h-7 gap-1 text-[11px]"
        onClick={() => setOpen(true)}
      >
        <Rocket className="h-3 w-3" /> Promote to library
      </Button>
    )
  }

  const error = mutation.isError
    ? (mutation.error as { response?: { data?: { detail?: string } }; message?: string })
    : null

  return (
    <div className="rounded-md border border-violet-500/40 bg-card p-2 absolute right-4 z-10 mt-8 w-72 shadow-lg">
      <div className="mb-2 text-xs font-semibold">Promote to strategy library</div>
      <div className="space-y-1.5">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Strategy name"
          className="h-7 text-xs"
        />
        <Input
          value={slug}
          onChange={(e) => setSlug(e.target.value)}
          placeholder="strategy_slug"
          className="h-7 font-mono text-xs"
        />
        <label className="flex items-center gap-1.5 text-[10.5px] text-muted-foreground">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
            className="h-3 w-3 accent-violet-500"
          />
          Enable immediately (not recommended — review first)
        </label>
      </div>
      {error ? (
        <div className="mt-1 text-[10px] text-rose-700 dark:text-rose-300">
          {error.response?.data?.detail || error.message || 'Failed'}
        </div>
      ) : null}
      <div className="mt-2 flex justify-end gap-1">
        <Button size="sm" variant="ghost" onClick={() => setOpen(false)}>
          Cancel
        </Button>
        <Button
          size="sm"
          onClick={() => mutation.mutate()}
          disabled={!name.trim() || !slug.trim() || mutation.isPending}
        >
          {mutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Promote'}
        </Button>
      </div>
    </div>
  )
}
