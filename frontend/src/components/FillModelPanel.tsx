import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Activity,
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Clock,
  Gauge,
  Layers3,
  Loader2,
  RefreshCcw,
  Rocket,
  Sparkles,
  TrendingUp,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { cn } from '../lib/utils'
import {
  type EmpiricalConstantsResponse,
  type FillModelRow,
  getActiveFillModel,
  getDecompositionSummary,
  getEmpiricalConstants,
  getFillModelHistory,
  getLatencyDistribution,
  listMLCapabilities,
  promoteModel,
  setEmpiricalOverrides,
  triggerRetrain,
  updateLatencyFallbacks,
} from '../services/apiFillModel'

const COVARIATE_LABEL_KEYS: Record<string, string> = {
  queue_ahead_shares: 'fillModelPanel.covariates.queue_ahead_shares',
  depth_behind_shares: 'fillModelPanel.covariates.depth_behind_shares',
  spread_bps: 'fillModelPanel.covariates.spread_bps',
  mid_distance_bps: 'fillModelPanel.covariates.mid_distance_bps',
  recent_trade_intensity_per_sec: 'fillModelPanel.covariates.recent_trade_intensity_per_sec',
  time_to_resolution_seconds: 'fillModelPanel.covariates.time_to_resolution_seconds',
  side_imbalance: 'fillModelPanel.covariates.side_imbalance',
  underlying_volatility_bps_per_min: 'fillModelPanel.covariates.underlying_volatility_bps_per_min',
  latency_p95_ms: 'fillModelPanel.covariates.latency_p95_ms',
  book_age_ms: 'fillModelPanel.covariates.book_age_ms',
  notional_usd: 'fillModelPanel.covariates.notional_usd',
}

const CONSTANT_LABEL_KEYS: Record<keyof EmpiricalConstantsResponse['values'], string> = {
  displayed_depth_factor: 'fillModelPanel.constants.displayed_depth_factor',
  maker_queue_ahead_fraction: 'fillModelPanel.constants.maker_queue_ahead_fraction',
  maker_trade_flow_multiplier: 'fillModelPanel.constants.maker_trade_flow_multiplier',
  adverse_selection_multiplier: 'fillModelPanel.constants.adverse_selection_multiplier',
  stale_depth_decay: 'fillModelPanel.constants.stale_depth_decay',
  min_depth_factor: 'fillModelPanel.constants.min_depth_factor',
}

function StatPill({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string
  value: string
  hint?: string
  tone?: 'good' | 'warn' | 'bad' | 'neutral'
}) {
  return (
    <div
      className={cn(
        'rounded-md border px-3 py-2 text-xs leading-tight',
        tone === 'good' && 'border-emerald-500/30 bg-emerald-500/5 text-emerald-300',
        tone === 'warn' && 'border-amber-500/30 bg-amber-500/5 text-amber-300',
        tone === 'bad' && 'border-red-500/30 bg-red-500/5 text-red-300',
        tone === 'neutral' && 'border-border/50 bg-card/40 text-foreground',
      )}
    >
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-base font-semibold">{value}</div>
      {hint ? <div className="mt-0.5 text-[10px] text-muted-foreground">{hint}</div> : null}
    </div>
  )
}

function HazardRatioBar({ hr, label }: { hr: number; label: string }) {
  // HR > 1 means the covariate INCREASES fill hazard (faster fill).
  // HR < 1 means it DECREASES fill hazard.  Bar centered on 1.0.
  const clamped = Math.max(0.2, Math.min(2.5, hr))
  const isPos = clamped >= 1.0
  const widthPct = Math.min(100, Math.abs(Math.log2(clamped)) * 100)
  return (
    <div className="grid grid-cols-[180px,1fr,80px] items-center gap-2 py-1">
      <div className="text-xs text-muted-foreground truncate">{label}</div>
      <div className="relative h-3 rounded-sm bg-muted/30">
        <div className="absolute inset-y-0 left-1/2 w-[1px] bg-border/70" />
        <div
          className={cn(
            'absolute inset-y-0 rounded-sm',
            isPos ? 'left-1/2 bg-emerald-500/60' : 'right-1/2 bg-red-500/60',
          )}
          style={{ width: `${widthPct / 2}%` }}
        />
      </div>
      <div
        className={cn(
          'text-right font-mono text-xs',
          isPos ? 'text-emerald-300' : 'text-red-300',
        )}
      >
        {hr.toFixed(2)}×
      </div>
    </div>
  )
}

// Latency card with editable fallbacks.  When sample_count == 0 the
// values shown are operator-tunable defaults; clicking "Edit" turns
// the three quantiles into number inputs and exposes a Save button.
// Saved overrides go to AppSettings via /fill-model/latency/fallbacks
// and the values immediately reflect in /fill-model/latency.
function LatencyCard({
  latency,
}: {
  latency:
    | {
        p50_ms: number
        p95_ms: number
        p99_ms: number
        sample_count: number
        fallback_p50_ms?: number
        fallback_p95_ms?: number
        fallback_p99_ms?: number
      }
    | undefined
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<{ p50: string; p95: string; p99: string }>({
    p50: '',
    p95: '',
    p99: '',
  })

  const saveMutation = useMutation({
    mutationFn: updateLatencyFallbacks,
    onSuccess: () => {
      setEditing(false)
      queryClient.invalidateQueries({ queryKey: ['fill-model', 'latency'] })
    },
  })

  const isFallback = latency != null && latency.sample_count === 0
  const fallbackHint =
    latency != null && latency.fallback_p50_ms != null
      ? t('fillModelPanel.latency.fallbackDefaults', {
          p50: Math.round(latency.fallback_p50_ms),
          p95: Math.round(latency.fallback_p95_ms ?? 0),
          p99: Math.round(latency.fallback_p99_ms ?? 0),
        })
      : ''

  return (
    <div className="rounded-md border border-border/50 bg-card/40 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium">
        <Clock className="h-3.5 w-3.5 text-sky-300" />
        {latency && latency.sample_count > 0
          ? t('fillModelPanel.latency.measuredTitle')
          : t('fillModelPanel.latency.defaultsTitle')}
        {isFallback ? (
          <span className="ml-auto rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-amber-300">
            {t('fillModelPanel.latency.fallbackBadge')}
          </span>
        ) : null}
        {isFallback && !editing ? (
          <button
            type="button"
            onClick={() => {
              setDraft({
                p50: String(Math.round(latency?.fallback_p50_ms ?? latency?.p50_ms ?? 200)),
                p95: String(Math.round(latency?.fallback_p95_ms ?? latency?.p95_ms ?? 600)),
                p99: String(Math.round(latency?.fallback_p99_ms ?? latency?.p99_ms ?? 1500)),
              })
              setEditing(true)
            }}
            className="rounded-sm border border-border/60 px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-muted/40"
          >
            {t('fillModelPanel.latency.edit')}
          </button>
        ) : null}
      </div>
      {latency ? (
        editing ? (
          <div className="grid grid-cols-3 gap-2">
            <div>
              <Label className="text-[9px] text-muted-foreground">{t('fillModelPanel.latency.p50Label')}</Label>
              <Input
                type="number"
                min={1}
                max={60000}
                value={draft.p50}
                onChange={(e) => setDraft({ ...draft, p50: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
            <div>
              <Label className="text-[9px] text-muted-foreground">{t('fillModelPanel.latency.p95Label')}</Label>
              <Input
                type="number"
                min={1}
                max={60000}
                value={draft.p95}
                onChange={(e) => setDraft({ ...draft, p95: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
            <div>
              <Label className="text-[9px] text-muted-foreground">{t('fillModelPanel.latency.p99Label')}</Label>
              <Input
                type="number"
                min={1}
                max={60000}
                value={draft.p99}
                onChange={(e) => setDraft({ ...draft, p99: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
          </div>
        ) : (
          <div className={`grid grid-cols-3 gap-2 ${isFallback ? 'opacity-70' : ''}`}>
            <StatPill label={t('fillModelPanel.latency.p50')} value={`${Math.round(latency.p50_ms)} ms`} tone="neutral" />
            <StatPill
              label={t('fillModelPanel.latency.p95')}
              value={`${Math.round(latency.p95_ms)} ms`}
              tone={latency.sample_count > 0 && latency.p95_ms > 800 ? 'warn' : 'neutral'}
            />
            <StatPill
              label={t('fillModelPanel.latency.p99')}
              value={`${Math.round(latency.p99_ms)} ms`}
              tone={latency.sample_count > 0 && latency.p99_ms > 1500 ? 'bad' : 'neutral'}
            />
          </div>
        )
      ) : (
        <div className="text-xs text-muted-foreground">{t('fillModelPanel.common.loading')}</div>
      )}
      {editing ? (
        <div className="mt-2 flex items-center gap-2">
          <Button
            size="sm"
            className="h-6 text-[10px]"
            onClick={() =>
              saveMutation.mutate({
                p50_ms: parseFloat(draft.p50) || undefined,
                p95_ms: parseFloat(draft.p95) || undefined,
                p99_ms: parseFloat(draft.p99) || undefined,
              })
            }
            disabled={saveMutation.isPending}
          >
            {saveMutation.isPending ? t('fillModelPanel.latency.saving') : t('fillModelPanel.latency.save')}
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 text-[10px]"
            onClick={() => setEditing(false)}
          >
            {t('fillModelPanel.latency.cancel')}
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 text-[10px] text-muted-foreground"
            onClick={() => saveMutation.mutate({ p50_ms: 0, p95_ms: 0, p99_ms: 0 })}
            disabled={saveMutation.isPending}
            title={t('fillModelPanel.latency.resetTooltip')}
          >
            {t('fillModelPanel.latency.resetToDefaults')}
          </Button>
          {saveMutation.error ? (
            <span className="text-[10px] text-red-300">{t('fillModelPanel.latency.saveFailed')}</span>
          ) : null}
        </div>
      ) : latency ? (
        <div className="mt-2 text-[10px] text-muted-foreground">
          {latency.sample_count > 0
            ? t('fillModelPanel.latency.usageMeasured', { samples: latency.sample_count.toLocaleString() })
            : t('fillModelPanel.latency.usageFallback', { hint: fallbackHint })}
        </div>
      ) : null}
    </div>
  )
}


function BaselineSurvivalChart({ baseline }: { baseline: Record<string, number> }) {
  const { t } = useTranslation()
  const points = Object.entries(baseline)
    .map(([t, s]) => ({ t: parseFloat(t), s: Number(s) }))
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.s))
    .sort((a, b) => a.t - b.t)
  if (points.length < 2) {
    return (
      <div className="text-xs text-muted-foreground italic">
        {t('fillModelPanel.baseline.empty')}
      </div>
    )
  }
  const maxT = points[points.length - 1].t
  const w = 320
  const h = 120
  const path = points
    .map((p, i) => {
      const x = (p.t / maxT) * (w - 12) + 6
      const y = h - 8 - p.s * (h - 24)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
  return (
    <div className="rounded-md border border-border/40 bg-card/40 p-2">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{t('fillModelPanel.baseline.legendY')}</span>
        <span>{t('fillModelPanel.baseline.legendRange', { max: Math.round(maxT) })}</span>
      </div>
      <svg width={w} height={h} className="mt-1">
        <line x1={6} y1={h - 8} x2={w - 6} y2={h - 8} stroke="rgb(var(--border))" strokeWidth={0.5} />
        <line x1={6} y1={8} x2={6} y2={h - 8} stroke="rgb(var(--border))" strokeWidth={0.5} />
        <path d={path} fill="none" stroke="hsl(160, 80%, 55%)" strokeWidth={1.5} />
      </svg>
      <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
        <span>0</span>
        <span>{t('fillModelPanel.baseline.legendCenter')}</span>
        <span>1</span>
      </div>
    </div>
  )
}

function fmtTs(iso: string | null): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

export default function FillModelPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const activeQuery = useQuery({
    queryKey: ['fill-model', 'active', 'pooled'],
    queryFn: () => getActiveFillModel('pooled'),
    refetchInterval: 30000,
  })
  const historyQuery = useQuery({
    queryKey: ['fill-model', 'history'],
    queryFn: () => getFillModelHistory(undefined, 20),
    refetchInterval: 60000,
  })
  const constantsQuery = useQuery({
    queryKey: ['fill-model', 'empirical-constants'],
    queryFn: getEmpiricalConstants,
    refetchInterval: 30000,
  })
  const latencyQuery = useQuery({
    queryKey: ['fill-model', 'latency'],
    queryFn: getLatencyDistribution,
    refetchInterval: 15000,
  })
  const capabilitiesQuery = useQuery({
    queryKey: ['ml', 'capabilities'],
    queryFn: listMLCapabilities,
    refetchInterval: 60_000,
  })
  const decompQuery = useQuery({
    queryKey: ['fill-model', 'decomposition'],
    queryFn: () => getDecompositionSummary(24),
    refetchInterval: 30000,
  })

  const retrainMutation = useMutation({
    mutationFn: () => triggerRetrain(30),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fill-model'] })
    },
  })
  const promoteMutation = useMutation({
    mutationFn: (modelId: string) => promoteModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fill-model'] })
    },
  })

  const overridesMutation = useMutation({
    mutationFn: setEmpiricalOverrides,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['fill-model', 'empirical-constants'] }),
  })

  const [overrides, setOverrides] = useState<Partial<EmpiricalConstantsResponse['values']>>({})

  const applyOverrides = () => {
    const filtered: Partial<EmpiricalConstantsResponse['values']> = {}
    for (const [k, v] of Object.entries(overrides) as Array<[
      keyof EmpiricalConstantsResponse['values'],
      number,
    ]>) {
      if (Number.isFinite(v)) filtered[k] = v
    }
    overridesMutation.mutate(filtered)
  }

  const active = activeQuery.data
  const history = historyQuery.data ?? []
  const constants = constantsQuery.data
  const latency = latencyQuery.data
  const decomp = decompQuery.data

  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 p-3">
        {/* HEADER */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-amber-300" />
              <div className="text-sm font-semibold">{t('fillModelPanel.header.title')}</div>
              <Badge variant="outline" className="text-[10px]">
                {active?.family ?? t('fillModelPanel.header.noModelLoaded')}
              </Badge>
              {active?.active ? (
                <Badge className="bg-emerald-500/10 text-emerald-300 text-[10px]">{t('fillModelPanel.header.active')}</Badge>
              ) : null}
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              {t('fillModelPanel.header.subtitle')}
            </div>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => retrainMutation.mutate()}
            disabled={retrainMutation.isPending}
          >
            {retrainMutation.isPending ? (
              <Loader2 className="mr-1 h-3 w-3 animate-spin" />
            ) : (
              <RefreshCcw className="mr-1 h-3 w-3" />
            )}
            {t('fillModelPanel.header.retrain')}
          </Button>
        </div>

        {/* ACTIVE MODEL SUMMARY */}
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <StatPill
            label={t('fillModelPanel.summary.cIndex')}
            value={active?.concordance_index != null ? fmtNum(active.concordance_index, 3) : '—'}
            hint={active?.family === 'cox_ph' ? t('fillModelPanel.summary.cIndexHintCox') : t('fillModelPanel.summary.cIndexHintKm')}
            tone={
              active?.concordance_index != null
                ? active.concordance_index > 0.62
                  ? 'good'
                  : active.concordance_index > 0.55
                    ? 'warn'
                    : 'bad'
                : 'neutral'
            }
          />
          <StatPill
            label={t('fillModelPanel.summary.events')}
            value={active ? active.n_events.toLocaleString() : '—'}
            hint={active ? t('fillModelPanel.summary.eventsHint', { obs: active.n_observations.toLocaleString() }) : undefined}
          />
          <StatPill
            label={t('fillModelPanel.summary.strata')}
            value={active?.strata_key ?? '—'}
            hint={active?.notes ? active.notes.slice(0, 80) : undefined}
          />
          <StatPill
            label={t('fillModelPanel.summary.trained')}
            value={active?.trained_at ? new Date(active.trained_at).toLocaleString() : '—'}
            hint={active?.promoted_at ? t('fillModelPanel.summary.promotedHint', { ts: fmtTs(active.promoted_at) }) : undefined}
          />
        </div>

        {/* COX HAZARD RATIOS */}
        {active && active.family === 'cox_ph' && Object.keys(active.coefficients).length > 0 ? (
          <div className="rounded-md border border-border/50 bg-card/40 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium">
              <TrendingUp className="h-3.5 w-3.5 text-amber-300" />
              {t('fillModelPanel.hazard.title')}
              <span className="ml-auto text-[10px] text-muted-foreground">{t('fillModelPanel.hazard.legend')}</span>
            </div>
            <div className="space-y-0.5">
              {Object.entries(active.coefficients)
                .sort((a, b) => Math.abs(Math.log(b[1])) - Math.abs(Math.log(a[1])))
                .map(([cov, hr]) => (
                  <HazardRatioBar key={cov} label={COVARIATE_LABEL_KEYS[cov] ? t(COVARIATE_LABEL_KEYS[cov]) : cov} hr={hr} />
                ))}
            </div>
          </div>
        ) : null}

        {/* BASELINE SURVIVAL */}
        {active ? (
          <div className="rounded-md border border-border/50 bg-card/40 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium">
              <Activity className="h-3.5 w-3.5 text-emerald-300" />
              {t('fillModelPanel.baseline.title')}
            </div>
            <BaselineSurvivalChart baseline={active.baseline_survival} />
          </div>
        ) : null}

        {/* LATENCY + DECOMPOSITION */}
        <div className="grid gap-2 md:grid-cols-2">
          <LatencyCard latency={latency} />

          <div className="rounded-md border border-border/50 bg-card/40 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium">
              <Layers3 className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
              {t('fillModelPanel.decomposition.title')}
            </div>
            {decomp ? (
              <div className="grid grid-cols-2 gap-2">
                <StatPill
                  label={t('fillModelPanel.decomposition.tradeEvents')}
                  value={decomp.trade_count.toLocaleString()}
                  hint={
                    decomp.trade_count_pct != null
                      ? t('fillModelPanel.decomposition.byCount', { pct: fmtNum(decomp.trade_count_pct, 1) })
                      : undefined
                  }
                  tone="good"
                />
                <StatPill
                  label={t('fillModelPanel.decomposition.cancelEvents')}
                  value={decomp.cancel_count.toLocaleString()}
                  hint={
                    decomp.trade_count_pct != null
                      ? t('fillModelPanel.decomposition.byCount', { pct: fmtNum(100 - decomp.trade_count_pct, 1) })
                      : undefined
                  }
                  tone={
                    decomp.trade_count_pct != null && decomp.trade_count_pct < 30
                      ? 'warn'
                      : 'neutral'
                  }
                />
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">{t('fillModelPanel.common.loading')}</div>
            )}
            <div className="mt-2 text-[10px] text-muted-foreground">
              {t('fillModelPanel.decomposition.hint')}
            </div>
          </div>
        </div>

        {/* EMPIRICAL CONSTANTS — operator overrides */}
        <div className="rounded-md border border-border/50 bg-card/40 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 font-medium">
              <Gauge className="h-3.5 w-3.5 text-amber-300" />
              {t('fillModelPanel.constantsSection.title')}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              {constants?.measured ? (
                <Badge className="bg-emerald-500/10 text-emerald-300">{t('fillModelPanel.constantsSection.measuredBadge')}</Badge>
              ) : (
                <Badge className="bg-amber-500/10 text-amber-300">{t('fillModelPanel.constantsSection.defaultsBadge')}</Badge>
              )}
              <span>{constants?.notes ?? ''}</span>
            </div>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {constants
              ? (Object.keys(CONSTANT_LABEL_KEYS) as Array<keyof EmpiricalConstantsResponse['values']>).map(
                  (key) => {
                    const measured = constants.values[key]
                    const override = constants.overrides[key]
                    const displayed = override ?? measured
                    return (
                      <div
                        key={key}
                        className="flex flex-col gap-1 rounded-md border border-border/40 bg-background/40 px-2 py-1.5"
                      >
                        <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                          {t(CONSTANT_LABEL_KEYS[key])}
                        </Label>
                        <div className="flex items-center gap-2">
                          <Input
                            value={
                              overrides[key] != null
                                ? String(overrides[key])
                                : fmtNum(displayed, 3)
                            }
                            onChange={(e) =>
                              setOverrides((prev) => ({
                                ...prev,
                                [key]: parseFloat(e.target.value),
                              }))
                            }
                            className="h-6 text-xs"
                            placeholder={String(measured.toFixed(3))}
                          />
                          <span className="font-mono text-[10px] text-muted-foreground">
                            {override != null
                              ? t('fillModelPanel.constantsSection.overrideLabel')
                              : t('fillModelPanel.constantsSection.measuredLabel', { value: fmtNum(measured, 3) })}
                          </span>
                        </div>
                      </div>
                    )
                  },
                )
              : null}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <Button
              size="sm"
              variant="default"
              onClick={applyOverrides}
              disabled={overridesMutation.isPending || Object.keys(overrides).length === 0}
            >
              {overridesMutation.isPending ? (
                <Loader2 className="mr-1 h-3 w-3 animate-spin" />
              ) : (
                <CheckCircle2 className="mr-1 h-3 w-3" />
              )}
              {t('fillModelPanel.constantsSection.applyOverrides')}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                setOverrides({})
                overridesMutation.mutate({} as Partial<EmpiricalConstantsResponse['values']>)
              }}
              disabled={overridesMutation.isPending}
            >
              {t('fillModelPanel.constantsSection.revertToMeasured')}
            </Button>
          </div>
          {overridesMutation.isError ? (
            <div className="mt-2 flex items-center gap-1 text-xs text-red-300">
              <AlertTriangle className="h-3 w-3" />
              {(overridesMutation.error as { message?: string })?.message ?? t('fillModelPanel.constantsSection.overrideFailed')}
            </div>
          ) : null}
        </div>

        {/* ML CAPABILITY REGISTRY — strategies that own ML tasks */}
        <div className="rounded-md border border-border/50 bg-card/40 p-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium">
            <Boxes className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
            {t('fillModelPanel.capabilities.title')}
            <span className="ml-auto text-[10px] text-muted-foreground">
              {(capabilitiesQuery.data?.length ?? 0) === 1
                ? t('fillModelPanel.capabilities.taskCountOne', { n: capabilitiesQuery.data?.length ?? 0 })
                : t('fillModelPanel.capabilities.taskCountOther', { n: capabilitiesQuery.data?.length ?? 0 })}
            </span>
          </div>
          <div className="space-y-1">
            {(capabilitiesQuery.data ?? []).map((cap) => (
              <div
                key={cap.task_key}
                className="rounded-sm border border-border/40 bg-background/40 px-2 py-1.5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs">{cap.task_key}</span>
                  {cap.owner_strategy_slug ? (
                    <Badge className="bg-violet-500/10 text-violet-700 dark:text-violet-300 text-[9px]">
                      {t('fillModelPanel.capabilities.strategyBadge', { slug: cap.owner_strategy_slug })}
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-[9px]">
                      {t('fillModelPanel.capabilities.builtInFallback')}
                    </Badge>
                  )}
                </div>
                <div className="mt-0.5 text-[10px] text-muted-foreground">
                  {cap.label}
                </div>
                <div className="mt-1 flex flex-wrap gap-1 text-[9px]">
                  {cap.allowed_assets.slice(0, 8).map((a) => (
                    <span key={a} className="rounded-sm bg-muted/40 px-1 py-px font-mono">
                      {a}
                    </span>
                  ))}
                  {cap.allowed_timeframes.slice(0, 6).map((t) => (
                    <span key={t} className="rounded-sm bg-amber-500/10 px-1 py-px font-mono text-amber-300">
                      {t}
                    </span>
                  ))}
                  <span className="ml-auto rounded-sm bg-emerald-500/10 px-1 py-px font-mono text-emerald-300">
                    {t('fillModelPanel.capabilities.featureCount', { n: cap.feature_names.length })}
                  </span>
                </div>
              </div>
            ))}
            {(capabilitiesQuery.data ?? []).length === 0 ? (
              <div className="text-xs text-muted-foreground">
                {t('fillModelPanel.capabilities.emptyPrefix')}{' '}
                <code className="rounded-sm bg-muted/40 px-1 font-mono text-[10px]">ml_capability = MLCapability(...)</code>.
              </div>
            ) : null}
          </div>
          <div className="mt-2 text-[10px] text-muted-foreground">
            {t('fillModelPanel.capabilities.hint')}
          </div>
        </div>

        {/* MODEL HISTORY (promotion list) */}
        <div className="rounded-md border border-border/50 bg-card/40 p-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium">
            <Layers3 className="h-3.5 w-3.5 text-emerald-300" />
            {t('fillModelPanel.history.title')}
          </div>
          <div className="space-y-1">
            {history.length === 0 ? (
              <div className="text-xs text-muted-foreground">
                {t('fillModelPanel.history.emptyPrefix')} <strong>{t('fillModelPanel.header.retrain')}</strong> {t('fillModelPanel.history.emptySuffix')}
              </div>
            ) : null}
            {history.map((row: FillModelRow) => (
              <div
                key={row.id}
                className={cn(
                  'grid grid-cols-[120px,80px,80px,1fr,90px] items-center gap-2 rounded-sm px-2 py-1 text-xs',
                  row.active && 'bg-emerald-500/5',
                )}
              >
                <div className="font-mono text-[11px] text-muted-foreground">
                  {row.trained_at ? new Date(row.trained_at).toLocaleString() : '—'}
                </div>
                <div>{row.family}</div>
                <div className="font-mono">{row.strata_key}</div>
                <div className="text-muted-foreground">
                  {t('fillModelPanel.history.eventsCidx', {
                    events: row.n_events.toLocaleString(),
                    cidx: row.concordance_index != null ? fmtNum(row.concordance_index, 3) : '—',
                  })}
                </div>
                <div className="flex justify-end">
                  {row.active ? (
                    <Badge className="bg-emerald-500/10 text-emerald-300 text-[10px]">{t('fillModelPanel.header.active')}</Badge>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 text-[10px]"
                      onClick={() => promoteMutation.mutate(row.id)}
                      disabled={promoteMutation.isPending}
                    >
                      <Rocket className="mr-1 h-3 w-3" />
                      {t('fillModelPanel.history.promote')}
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}
