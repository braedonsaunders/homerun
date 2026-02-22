import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle, Save, Settings, X } from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'

export type DiscoverySettingsForm = {
  maintenance_enabled: boolean
  max_discovered_wallets: number
  maintenance_batch: number
  keep_recent_trade_days: number
  keep_new_discoveries_days: number
  stale_analysis_hours: number
  analysis_priority_batch_limit: number
  delay_between_markets: number
  delay_between_wallets: number
  max_markets_per_run: number
  max_wallets_per_market: number
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

export default function DiscoverySettingsFlyout({
  isOpen,
  onClose,
  initial,
  onSave,
  savePending,
  saveMessage,
}: {
  isOpen: boolean
  onClose: () => void
  initial: DiscoverySettingsForm
  onSave: (next: DiscoverySettingsForm) => void
  savePending?: boolean
  saveMessage?: { type: 'success' | 'error'; text: string } | null
}) {
  const [form, setForm] = useState<DiscoverySettingsForm>(initial)

  useEffect(() => {
    if (!isOpen) return
    setForm(initial)
  }, [initial, isOpen])

  if (!isOpen) return null

  const handleSave = () => {
    onSave({
      maintenance_enabled: form.maintenance_enabled,
      max_discovered_wallets: Math.round(clamp(form.max_discovered_wallets, 10, 1_000_000)),
      maintenance_batch: Math.round(clamp(form.maintenance_batch, 10, 5_000)),
      keep_recent_trade_days: Math.round(clamp(form.keep_recent_trade_days, 1, 365)),
      keep_new_discoveries_days: Math.round(clamp(form.keep_new_discoveries_days, 1, 365)),
      stale_analysis_hours: Math.round(clamp(form.stale_analysis_hours, 1, 720)),
      analysis_priority_batch_limit: Math.round(clamp(form.analysis_priority_batch_limit, 100, 10_000)),
      delay_between_markets: clamp(form.delay_between_markets, 0, 10),
      delay_between_wallets: clamp(form.delay_between_wallets, 0, 10),
      max_markets_per_run: Math.round(clamp(form.max_markets_per_run, 1, 1_000)),
      max_wallets_per_market: Math.round(clamp(form.max_wallets_per_market, 1, 500)),
    })
  }

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40 transition-opacity" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold">Discovery Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={handleSave}
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
            Configure discovery catalog maintenance, retention windows, and scan pacing for the discovery worker.
          </p>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Catalog Retention</h4>
            <div className="rounded-md border border-border/50 bg-muted/40 p-2.5 flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-medium">Enable maintenance</p>
                <p className="text-[10px] text-muted-foreground">Control catalog growth and pruning each discovery cycle.</p>
              </div>
              <Switch
                checked={form.maintenance_enabled}
                onCheckedChange={(checked) => setForm((prev) => ({ ...prev, maintenance_enabled: checked }))}
              />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Max Discovered Wallets"
                help="Rows kept in wallet catalog."
                value={form.max_discovered_wallets}
                onChange={(v) => setForm((prev) => ({ ...prev, max_discovered_wallets: v }))}
                min={10}
                max={1_000_000}
                step={1}
              />
              <NumericField
                label="Discovery Maintenance Batch"
                help="Chunk size for remove/insert operations."
                value={form.maintenance_batch}
                onChange={(v) => setForm((prev) => ({ ...prev, maintenance_batch: v }))}
                min={10}
                max={5_000}
                step={1}
              />
              <NumericField
                label="Keep Recent Trades (days)"
                help="Protect wallets with recent trades."
                value={form.keep_recent_trade_days}
                onChange={(v) => setForm((prev) => ({ ...prev, keep_recent_trade_days: v }))}
                min={1}
                max={365}
                step={1}
              />
              <NumericField
                label="Keep New Discoveries (days)"
                help="Protect newly discovered wallets."
                value={form.keep_new_discoveries_days}
                onChange={(v) => setForm((prev) => ({ ...prev, keep_new_discoveries_days: v }))}
                min={1}
                max={365}
                step={1}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Analysis Queue</h4>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Stale Analysis Threshold (hours)"
                help="Re-analyze wallets older than this age."
                value={form.stale_analysis_hours}
                onChange={(v) => setForm((prev) => ({ ...prev, stale_analysis_hours: v }))}
                min={1}
                max={720}
                step={1}
              />
              <NumericField
                label="Priority Queue Limit"
                help="Cap for new/stale wallet queue."
                value={form.analysis_priority_batch_limit}
                onChange={(v) => setForm((prev) => ({ ...prev, analysis_priority_batch_limit: v }))}
                min={100}
                max={10_000}
                step={1}
              />
              <NumericField
                label="Delay Between Markets (sec)"
                help="Throttle between market samples."
                value={form.delay_between_markets}
                onChange={(v) => setForm((prev) => ({ ...prev, delay_between_markets: v }))}
                min={0}
                max={10}
                step={0.05}
              />
              <NumericField
                label="Delay Between Wallets (sec)"
                help="Throttle wallet analysis loop."
                value={form.delay_between_wallets}
                onChange={(v) => setForm((prev) => ({ ...prev, delay_between_wallets: v }))}
                min={0}
                max={10}
                step={0.05}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Discovery Sampling</h4>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Max Markets Per Run"
                help="Active markets sampled each run."
                value={form.max_markets_per_run}
                onChange={(v) => setForm((prev) => ({ ...prev, max_markets_per_run: v }))}
                min={1}
                max={1_000}
                step={1}
              />
              <NumericField
                label="Max Wallets Per Market"
                help="Wallets extracted per sampled market."
                value={form.max_wallets_per_market}
                onChange={(v) => setForm((prev) => ({ ...prev, max_wallets_per_market: v }))}
                min={1}
                max={500}
                step={1}
              />
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}
