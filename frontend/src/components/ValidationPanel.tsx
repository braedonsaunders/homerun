/**
 * ValidationPanel — global validation guardrail config + strategy health
 * roster. Lives under the Strategies tab as the "Validation" subview
 * (next to Strategies / Research / ML Models).
 *
 * Two responsibilities:
 *
 *   1. Configure the auto-demotion guardrail thresholds globally:
 *      enable/disable, min sample size, min directional accuracy,
 *      max MAE on ROI, lookback window, auto-promote on/off. The same
 *      values back the per-cycle gate that short-circuits demoted
 *      strategies in the orchestrator.
 *   2. Show every strategy's health row with manual override controls,
 *      so an operator can flip status from one place — same buttons
 *      that exist on the strategy editor's Health subtab and on the
 *      TradingPanel banner, just consolidated here.
 */

import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Gauge,
  Loader2,
  Play,
  RotateCcw,
  Save,
  Shield,
  ShieldCheck,
  ShieldOff,
  X,
} from 'lucide-react'

import {
  clearValidationStrategyOverride,
  getValidationGuardrailConfig,
  getValidationStrategyHealth,
  overrideValidationStrategy,
  runValidationGuardrails,
  updateValidationGuardrailConfig,
  type GuardrailConfig,
  type StrategyHealthRow,
} from '../services/apiSettings'

import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'


export default function ValidationPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const configQuery = useQuery({
    queryKey: ['validation-guardrail-config'],
    queryFn: getValidationGuardrailConfig,
    staleTime: 30_000,
  })
  const healthQuery = useQuery({
    queryKey: ['validation-strategy-health'],
    queryFn: getValidationStrategyHealth,
    staleTime: 15_000,
    refetchInterval: 30_000,
  })

  const [draft, setDraft] = useState<Partial<GuardrailConfig>>({})
  const [saveError, setSaveError] = useState<string | null>(null)

  // Seed draft from server config when it loads / changes.
  useEffect(() => {
    if (configQuery.data) {
      setDraft(configQuery.data)
      setSaveError(null)
    }
  }, [configQuery.data])

  const dirty = useMemo(() => {
    if (!configQuery.data) return false
    const a = draft
    const b = configQuery.data
    return (
      a.enabled !== b.enabled
      || a.min_samples !== b.min_samples
      || a.min_directional_accuracy !== b.min_directional_accuracy
      || a.max_mae_roi !== b.max_mae_roi
      || a.lookback_days !== b.lookback_days
      || a.auto_promote !== b.auto_promote
    )
  }, [draft, configQuery.data])

  const updateConfigMutation = useMutation({
    mutationFn: updateValidationGuardrailConfig,
    onSuccess: (data) => {
      queryClient.setQueryData(['validation-guardrail-config'], data)
      queryClient.invalidateQueries({ queryKey: ['validation-strategy-health'] })
      setSaveError(null)
    },
    onError: (err) => setSaveError(err instanceof Error ? err.message : String(err)),
  })

  const evaluateMutation = useMutation({
    mutationFn: runValidationGuardrails,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation-strategy-health'] })
      queryClient.invalidateQueries({ queryKey: ['validation-guardrail-config'] })
    },
  })

  const overrideMutation = useMutation({
    mutationFn: async ({ strategyType, status }: { strategyType: string; status: 'active' | 'demoted' }) =>
      overrideValidationStrategy(strategyType, status),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['validation-strategy-health'] }),
  })
  const clearOverrideMutation = useMutation({
    mutationFn: clearValidationStrategyOverride,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['validation-strategy-health'] }),
  })

  const healthRows: StrategyHealthRow[] = healthQuery.data || []
  const summary = useMemo(() => {
    let active = 0
    let demoted = 0
    let untracked = 0
    let manual = 0
    for (const r of healthRows) {
      if (r.status === 'demoted') demoted++
      else if (r.status === 'active') active++
      else untracked++
      if (r.manual_override) manual++
    }
    return { active, demoted, untracked, manual, total: healthRows.length }
  }, [healthRows])

  const enabled = draft.enabled ?? configQuery.data?.enabled ?? false

  return (
    <div className="h-full min-h-0 flex flex-col">
      {/* Header */}
      <div className="shrink-0 px-4 py-2.5 border-b border-border/40 flex items-center gap-3">
        <Shield className={cn('w-4 h-4 shrink-0', enabled ? 'text-emerald-400' : 'text-muted-foreground')} />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold leading-tight">{t('validation.title')}</p>
          <p className="text-[10px] text-muted-foreground leading-tight">
            {t('validation.subtitle')}
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="h-8 gap-1.5 px-3 text-xs"
          onClick={() => evaluateMutation.mutate()}
          disabled={evaluateMutation.isPending}
          title={t('validation.evaluateNowTitle')}
        >
          {evaluateMutation.isPending ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Play className="w-3.5 h-3.5" />
          )}
          {t('validation.evaluateNow')}
        </Button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
        {/* ── Guardrail config card ── */}
        <div className="rounded-lg border border-border/40 bg-card/30 p-3 space-y-3">
          <div className="flex items-center gap-2">
            <Gauge className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-xs font-semibold">{t('validation.thresholds')}</span>
            {configQuery.isLoading && <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />}
          </div>

          {configQuery.data ? (
            <>
              <div className="rounded border border-border/30 bg-background/40 px-3 py-2 flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-medium">{t('validation.autoDemotionEngine')}</p>
                  <p className="text-[10px] text-muted-foreground leading-snug">
                    {t('validation.autoDemotionEngineDesc')}{' '}
                    {draft.auto_promote
                      ? t('validation.autoPromoteOnDesc')
                      : t('validation.autoPromoteOffDesc')}
                  </p>
                </div>
                <Switch
                  checked={Boolean(draft.enabled)}
                  onCheckedChange={(v) => setDraft((d) => ({ ...d, enabled: v }))}
                />
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <NumberField
                  label={t('validation.minSampleSize')}
                  hint={t('validation.minSampleSizeHint')}
                  value={draft.min_samples ?? 25}
                  step={1}
                  min={1}
                  onChange={(v) => setDraft((d) => ({ ...d, min_samples: Math.max(1, Math.round(v)) }))}
                  disabled={!enabled}
                />
                <NumberField
                  label={t('validation.minDirectionalAccuracy')}
                  hint={t('validation.minDirectionalAccuracyHint')}
                  value={(draft.min_directional_accuracy ?? 0.52)}
                  step={0.01}
                  min={0}
                  max={1}
                  format={(n) => `${(n * 100).toFixed(1)}%`}
                  onChange={(v) => setDraft((d) => ({ ...d, min_directional_accuracy: Math.max(0, Math.min(1, v)) }))}
                  disabled={!enabled}
                />
                <NumberField
                  label={t('validation.maxMaeRoi')}
                  hint={t('validation.maxMaeRoiHint')}
                  value={draft.max_mae_roi ?? 12}
                  step={0.5}
                  min={0}
                  onChange={(v) => setDraft((d) => ({ ...d, max_mae_roi: Math.max(0, v) }))}
                  disabled={!enabled}
                />
                <NumberField
                  label={t('validation.lookbackDays')}
                  hint={t('validation.lookbackDaysHint')}
                  value={draft.lookback_days ?? 90}
                  step={1}
                  min={7}
                  max={365}
                  onChange={(v) => setDraft((d) => ({ ...d, lookback_days: Math.max(7, Math.round(v)) }))}
                  disabled={!enabled}
                />
              </div>

              <div className="rounded border border-border/30 bg-background/40 px-3 py-2 flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-medium">{t('validation.autoPromoteOnRecovery')}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {t('validation.autoPromoteOnRecoveryDesc')}
                  </p>
                </div>
                <Switch
                  checked={Boolean(draft.auto_promote)}
                  onCheckedChange={(v) => setDraft((d) => ({ ...d, auto_promote: v }))}
                  disabled={!enabled}
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-1">
                {saveError && (
                  <span className="mr-auto text-[10px] text-red-300">
                    <AlertTriangle className="inline w-3 h-3 mr-1 align-text-bottom" />
                    {saveError}
                  </span>
                )}
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[11px]"
                  onClick={() => configQuery.data && setDraft(configQuery.data)}
                  disabled={!dirty || updateConfigMutation.isPending}
                >
                  <RotateCcw className="w-3 h-3 mr-1" />
                  {t('validation.discard')}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 px-3 text-[11px] bg-cyan-600 hover:bg-cyan-500 text-white"
                  onClick={() => updateConfigMutation.mutate(draft)}
                  disabled={!dirty || updateConfigMutation.isPending}
                >
                  {updateConfigMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin mr-1" />
                  ) : (
                    <Save className="w-3 h-3 mr-1" />
                  )}
                  {t('validation.saveThresholds')}
                </Button>
              </div>
            </>
          ) : (
            <div className="text-[11px] text-muted-foreground py-4 text-center">
              {configQuery.isError ? t('validation.loadConfigFailed') : t('validation.loadingEllipsis')}
            </div>
          )}
        </div>

        {/* ── Strategy health roster ── */}
        <div className="rounded-lg border border-border/40 bg-card/30 overflow-hidden">
          <div className="px-3 py-2 border-b border-border/30 flex items-center gap-2">
            <ShieldCheck className="w-3.5 h-3.5 text-violet-400" />
            <span className="text-xs font-semibold">{t('validation.strategyRoster')}</span>
            <span className="ml-auto text-[10px] text-muted-foreground font-mono flex items-center gap-3">
              <span className="text-emerald-300">{t('validation.activeCount', { n: summary.active })}</span>
              <span className="text-red-300">{t('validation.demotedCount', { n: summary.demoted })}</span>
              <span>{t('validation.untrackedCount', { n: summary.untracked })}</span>
              <span className="text-cyan-300">{t('validation.manualCount', { n: summary.manual })}</span>
            </span>
          </div>

          {healthRows.length === 0 ? (
            <div className="py-8 text-center text-[11px] text-muted-foreground">
              {healthQuery.isLoading ? t('validation.loadingHealth') : t('validation.noHealthRows')}
            </div>
          ) : (
            <table className="w-full text-[11px]">
              <thead className="bg-card/95 sticky top-0">
                <tr className="border-b border-border/30 text-muted-foreground text-[10px] uppercase tracking-wider">
                  <th className="text-left py-1.5 px-3">{t('validation.colStrategy')}</th>
                  <th className="text-left py-1.5 px-2">{t('validation.colStatus')}</th>
                  <th className="text-right py-1.5 px-2">{t('validation.colSamples')}</th>
                  <th className="text-right py-1.5 px-2">{t('validation.colAccuracy')}</th>
                  <th className="text-right py-1.5 px-2">{t('validation.colMae')}</th>
                  <th className="text-left py-1.5 px-2">{t('validation.colLastUpdate')}</th>
                  <th className="text-right py-1.5 px-3">{t('validation.colActions')}</th>
                </tr>
              </thead>
              <tbody>
                {healthRows.map((row) => {
                  const overrideBusy = overrideMutation.isPending || clearOverrideMutation.isPending
                  const isActive = row.status === 'active'
                  const isDemoted = row.status === 'demoted'
                  return (
                    <tr key={row.strategy_type} className="border-b border-border/10 hover:bg-card/60">
                      <td className="py-1.5 px-3 font-mono text-foreground">{row.strategy_type}</td>
                      <td className="py-1.5 px-2">
                        <span className={cn(
                          'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] border font-medium',
                          isDemoted
                            ? 'border-red-500/30 bg-red-500/10 text-red-300'
                            : isActive
                              ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                              : 'border-amber-500/30 bg-amber-500/10 text-amber-300',
                        )}>
                          <span className={cn(
                            'inline-block w-1.5 h-1.5 rounded-full',
                            isDemoted ? 'bg-red-400' : isActive ? 'bg-emerald-400' : 'bg-amber-400',
                          )} />
                          {row.status}
                          {row.manual_override && <span className="opacity-60">· {t('validation.manual')}</span>}
                        </span>
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono">
                        {Number(row.sample_size || 0).toLocaleString()}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono">
                        {Number.isFinite(Number(row.directional_accuracy))
                          ? `${(Number(row.directional_accuracy) * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono">
                        {Number.isFinite(Number(row.mae_roi)) ? Number(row.mae_roi).toFixed(2) : '—'}
                      </td>
                      <td className="py-1.5 px-2 text-[10px] text-muted-foreground">
                        {row.updated_at ? formatTimestamp(row.updated_at) : '—'}
                      </td>
                      <td className="py-1.5 px-3">
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className={cn(
                              'h-6 gap-1 px-2 text-[10px]',
                              isActive ? '' : 'border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/10',
                            )}
                            disabled={overrideBusy || isActive}
                            onClick={() => overrideMutation.mutate({ strategyType: row.strategy_type, status: 'active' })}
                          >
                            <CheckCircle2 className="w-3 h-3" />
                            {t('validation.activate')}
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className={cn(
                              'h-6 gap-1 px-2 text-[10px]',
                              isDemoted ? '' : 'border-red-500/30 text-red-300 hover:bg-red-500/10',
                            )}
                            disabled={overrideBusy || isDemoted}
                            onClick={() => overrideMutation.mutate({ strategyType: row.strategy_type, status: 'demoted' })}
                          >
                            <ShieldOff className="w-3 h-3" />
                            {t('validation.demote')}
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-6 gap-1 px-2 text-[10px]"
                            disabled={overrideBusy || !row.manual_override}
                            onClick={() => clearOverrideMutation.mutate(row.strategy_type)}
                            title={row.manual_override
                              ? t('validation.clearOverrideTitle')
                              : t('validation.noOverrideActive')}
                          >
                            <X className="w-3 h-3" />
                            {t('validation.clear')}
                          </Button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Last evaluation summary */}
        {evaluateMutation.data && (
          <div className="rounded border border-cyan-500/30 bg-cyan-500/5 px-3 py-2 text-[11px] text-muted-foreground">
            <Clock className="inline w-3 h-3 mr-1 align-text-bottom text-cyan-400" />
            {t('validation.lastEvaluation')}
            <span className="ml-2 font-mono">
              {t('validation.updatesCount', { n: evaluateMutation.data.updated ?? 0 })}
              {evaluateMutation.data.demoted?.length
                ? ` · ${t('validation.demotedSuffix', { n: evaluateMutation.data.demoted.length })}`
                : ''}
              {evaluateMutation.data.restored?.length
                ? ` · ${t('validation.restoredSuffix', { n: evaluateMutation.data.restored.length })}`
                : ''}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}


function NumberField({
  label,
  hint,
  value,
  onChange,
  step,
  min,
  max,
  format,
  disabled,
}: {
  label: string
  hint?: string
  value: number
  onChange: (v: number) => void
  step?: number
  min?: number
  max?: number
  format?: (v: number) => string
  disabled?: boolean
}) {
  const [text, setText] = useState(String(value))
  useEffect(() => {
    setText(String(value))
  }, [value])
  const formatted = format ? format(value) : null
  return (
    <div className="space-y-1">
      <Label className="text-[10px] text-muted-foreground">{label}</Label>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          step={step}
          min={min}
          max={max}
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            const parsed = Number(e.target.value)
            if (Number.isFinite(parsed)) onChange(parsed)
          }}
          disabled={disabled}
          className="h-7 text-[11px] font-mono"
        />
        {formatted && (
          <span className="text-[10px] text-muted-foreground font-mono shrink-0">{formatted}</span>
        )}
      </div>
      {hint && <p className="text-[9px] text-muted-foreground/70 leading-snug">{hint}</p>}
    </div>
  )
}


function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return iso
    return d.toLocaleString()
  } catch {
    return iso
  }
}
