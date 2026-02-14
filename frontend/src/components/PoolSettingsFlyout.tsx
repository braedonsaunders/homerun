import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle, Save, Settings, X } from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'

export type PoolSettingsForm = {
  pool_recompute_mode: 'quality_only' | 'balanced'
  pool_target_size: number
  pool_min_size: number
  pool_max_size: number
  pool_active_window_hours: number
  pool_selection_score_floor: number
  pool_max_hourly_replacement_rate: number
  pool_replacement_score_cutoff: number
  pool_max_cluster_share: number
  pool_high_conviction_threshold: number
  pool_insider_priority_threshold: number
  pool_min_eligible_trades: number
  pool_max_eligible_anomaly: number
  pool_core_min_win_rate: number
  pool_core_min_sharpe: number
  pool_core_min_profit_factor: number
  pool_rising_min_win_rate: number
  pool_slo_min_analyzed_pct: number
  pool_slo_min_profitable_pct: number
  pool_leaderboard_wallet_trade_sample: number
  pool_incremental_wallet_trade_sample: number
  pool_full_sweep_interval_seconds: number
  pool_incremental_refresh_interval_seconds: number
  pool_activity_reconciliation_interval_seconds: number
  pool_recompute_interval_seconds: number
}

function NumericField({
  label,
  help,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string
  help?: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <div>
      <Label className="text-[11px] text-muted-foreground leading-tight">{label}</Label>
      <Input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        className="mt-0.5 text-xs h-7"
      />
      {help ? <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">{help}</p> : null}
    </div>
  )
}

export default function PoolSettingsFlyout({
  isOpen,
  onClose,
  initial,
  onSave,
  savePending,
  saveMessage,
}: {
  isOpen: boolean
  onClose: () => void
  initial: PoolSettingsForm
  onSave: (next: PoolSettingsForm) => void
  savePending?: boolean
  saveMessage?: { type: 'success' | 'error'; text: string } | null
}) {
  const [form, setForm] = useState<PoolSettingsForm>(initial)

  useEffect(() => {
    if (!isOpen) return
    setForm(initial)
  }, [initial, isOpen])

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40 transition-opacity" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-2xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-amber-300" />
            <h3 className="text-sm font-semibold">Pool Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={() => onSave(form)}
              disabled={savePending}
              className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white"
            >
              <Save className="w-3 h-3" />
              {savePending ? 'Saving...' : 'Save'}
            </Button>
            <Button
              variant="ghost"
              onClick={onClose}
              className="text-xs h-auto px-2.5 py-1 hover:bg-card"
            >
              <X className="w-3.5 h-3.5 mr-1" />
              Close
            </Button>
          </div>
        </div>

        {saveMessage && (
          <div
            className={cn(
              'fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300',
              saveMessage.type === 'success'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-red-500/10 text-red-400 border-red-500/20',
            )}
          >
            {saveMessage.type === 'success' ? (
              <CheckCircle className="w-4 h-4 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 shrink-0" />
            )}
            {saveMessage.text}
          </div>
        )}

        <div className="p-3 space-y-3 pb-6">
          <p className="text-[11px] text-muted-foreground/70">
            Full runtime configuration for smart-wallet pool selection. Values are persisted and applied by the
            tracked-traders worker.
          </p>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Mode and Sizing</h4>
            <div>
              <Label className="text-[11px] text-muted-foreground leading-tight">Recompute Mode</Label>
              <select
                value={form.pool_recompute_mode}
                onChange={(e) =>
                  setForm((prev) => ({
                    ...prev,
                    pool_recompute_mode: e.target.value as PoolSettingsForm['pool_recompute_mode'],
                  }))
                }
                className="mt-0.5 h-8 w-full rounded-md border border-border bg-muted px-2 text-xs"
              >
                <option value="quality_only">Quality only (strict)</option>
                <option value="balanced">Balanced (fill-focused)</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Target Pool Size"
                value={form.pool_target_size}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_target_size: v }))}
                min={10}
                max={5000}
                step={1}
              />
              <NumericField
                label="Min Pool Size"
                value={form.pool_min_size}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_min_size: v }))}
                min={0}
                max={5000}
                step={1}
              />
              <NumericField
                label="Max Pool Size"
                value={form.pool_max_size}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_max_size: v }))}
                min={1}
                max={10000}
                step={1}
              />
              <NumericField
                label="Active Window (hours)"
                value={form.pool_active_window_hours}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_active_window_hours: v }))}
                min={1}
                max={720}
                step={1}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Selection and Churn</h4>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Selection Floor"
                value={form.pool_selection_score_floor}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_selection_score_floor: v }))}
                min={0}
                max={1}
                step={0.01}
              />
              <NumericField
                label="Max Hourly Replacement Rate"
                value={form.pool_max_hourly_replacement_rate}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_max_hourly_replacement_rate: v }))}
                min={0}
                max={1}
                step={0.01}
              />
              <NumericField
                label="Replacement Score Cutoff"
                value={form.pool_replacement_score_cutoff}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_replacement_score_cutoff: v }))}
                min={0}
                max={1}
                step={0.01}
              />
              <NumericField
                label="Max Cluster Share"
                value={form.pool_max_cluster_share}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_max_cluster_share: v }))}
                min={0.01}
                max={1}
                step={0.01}
              />
              <NumericField
                label="High Conviction Threshold"
                value={form.pool_high_conviction_threshold}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_high_conviction_threshold: v }))}
                min={0}
                max={1}
                step={0.01}
              />
              <NumericField
                label="Insider Priority Threshold"
                value={form.pool_insider_priority_threshold}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_insider_priority_threshold: v }))}
                min={0}
                max={1}
                step={0.01}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Eligibility Gates</h4>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Min Eligible Trades"
                value={form.pool_min_eligible_trades}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_min_eligible_trades: v }))}
                min={1}
                max={100000}
                step={1}
              />
              <NumericField
                label="Max Eligible Anomaly"
                value={form.pool_max_eligible_anomaly}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_max_eligible_anomaly: v }))}
                min={0}
                max={5}
                step={0.01}
              />
              <NumericField
                label="Core Min Win Rate"
                value={form.pool_core_min_win_rate}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_core_min_win_rate: v }))}
                min={0}
                max={1}
                step={0.01}
              />
              <NumericField
                label="Core Min Sharpe"
                value={form.pool_core_min_sharpe}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_core_min_sharpe: v }))}
                min={-10}
                max={20}
                step={0.1}
              />
              <NumericField
                label="Core Min Profit Factor"
                value={form.pool_core_min_profit_factor}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_core_min_profit_factor: v }))}
                min={0}
                max={20}
                step={0.1}
              />
              <NumericField
                label="Rising Min Win Rate"
                value={form.pool_rising_min_win_rate}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_rising_min_win_rate: v }))}
                min={0}
                max={1}
                step={0.01}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Pool Health SLO</h4>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Min Analyzed %"
                value={form.pool_slo_min_analyzed_pct}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_slo_min_analyzed_pct: v }))}
                min={0}
                max={100}
                step={0.1}
              />
              <NumericField
                label="Min Profitable %"
                value={form.pool_slo_min_profitable_pct}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_slo_min_profitable_pct: v }))}
                min={0}
                max={100}
                step={0.1}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Sampling and Cadence</h4>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Leaderboard Wallet Trade Sample"
                value={form.pool_leaderboard_wallet_trade_sample}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_leaderboard_wallet_trade_sample: v }))}
                min={1}
                max={5000}
                step={1}
              />
              <NumericField
                label="Incremental Wallet Trade Sample"
                value={form.pool_incremental_wallet_trade_sample}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_incremental_wallet_trade_sample: v }))}
                min={1}
                max={5000}
                step={1}
              />
              <NumericField
                label="Full Sweep Interval (sec)"
                value={form.pool_full_sweep_interval_seconds}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_full_sweep_interval_seconds: v }))}
                min={10}
                max={86400}
                step={1}
              />
              <NumericField
                label="Incremental Refresh Interval (sec)"
                value={form.pool_incremental_refresh_interval_seconds}
                onChange={(v) =>
                  setForm((prev) => ({ ...prev, pool_incremental_refresh_interval_seconds: v }))
                }
                min={10}
                max={86400}
                step={1}
              />
              <NumericField
                label="Activity Reconciliation Interval (sec)"
                value={form.pool_activity_reconciliation_interval_seconds}
                onChange={(v) =>
                  setForm((prev) => ({ ...prev, pool_activity_reconciliation_interval_seconds: v }))
                }
                min={10}
                max={86400}
                step={1}
              />
              <NumericField
                label="Pool Recompute Interval (sec)"
                value={form.pool_recompute_interval_seconds}
                onChange={(v) => setForm((prev) => ({ ...prev, pool_recompute_interval_seconds: v }))}
                min={10}
                max={86400}
                step={1}
              />
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}

