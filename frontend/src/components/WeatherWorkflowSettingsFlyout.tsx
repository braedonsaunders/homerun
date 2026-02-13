import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
  CloudRain,
  Bot,
  Brain,
  Thermometer,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import {
  getWeatherWorkflowSettings,
  updateWeatherWorkflowSettings,
  type WeatherWorkflowSettings,
} from '../services/api'

const DEFAULTS: WeatherWorkflowSettings = {
  enabled: true,
  auto_run: true,
  scan_interval_seconds: 14400,
  entry_max_price: 0.25,
  take_profit_price: 0.85,
  stop_loss_pct: 50,
  min_edge_percent: 8,
  min_confidence: 0.6,
  min_model_agreement: 0.75,
  min_liquidity: 500,
  max_markets_per_scan: 200,
  orchestrator_enabled: true,
  orchestrator_min_edge: 10,
  orchestrator_max_age_minutes: 240,
  default_size_usd: 10,
  max_size_usd: 50,
  model: null,
  temperature_unit: 'F',
}

function NumericField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
}) {
  return (
    <div className={cn(disabled && 'opacity-40 pointer-events-none')}>
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <Input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="mt-1 h-8 text-xs"
      />
    </div>
  )
}

export default function WeatherWorkflowSettingsFlyout({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [form, setForm] = useState<WeatherWorkflowSettings>(DEFAULTS)
  const [saveMessage, setSaveMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  const { data: settings } = useQuery({
    queryKey: ['weather-workflow-settings'],
    queryFn: getWeatherWorkflowSettings,
    enabled: isOpen,
  })

  useEffect(() => {
    if (settings) {
      setForm({ ...DEFAULTS, ...settings })
    }
  }, [settings])

  const set = <K extends keyof WeatherWorkflowSettings>(
    key: K,
    value: WeatherWorkflowSettings[K]
  ) => setForm((prev) => ({ ...prev, [key]: value }))

  const saveMutation = useMutation({
    mutationFn: updateWeatherWorkflowSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-settings'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] })
      setSaveMessage({ type: 'success', text: 'Weather settings saved' })
      setTimeout(() => setSaveMessage(null), 2500)
    },
    onError: (error: unknown) => {
      const message =
        error && typeof error === 'object' && 'message' in error
          ? String((error as { message?: string }).message || 'Save failed')
          : 'Save failed'
      setSaveMessage({ type: 'error', text: message })
      setTimeout(() => setSaveMessage(null), 4000)
    },
  })

  const handleSave = () => {
    saveMutation.mutate(form)
  }

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
            <h3 className="text-sm font-semibold">Weather Workflow Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={handleSave}
              disabled={saveMutation.isPending}
              className="gap-1 text-[10px] h-auto px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-white"
            >
              <Save className="w-3 h-3" />
              {saveMutation.isPending ? 'Saving...' : 'Save'}
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
                : 'bg-red-500/10 text-red-400 border-red-500/20'
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
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <CloudRain className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Pipeline</h4>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable Workflow</p>
                <p className="text-[10px] text-muted-foreground">Turn weather discovery and intent generation on/off</p>
              </div>
              <Switch checked={form.enabled} onCheckedChange={(v) => set('enabled', v)} className="scale-75" />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Auto-Run</p>
                <p className="text-[10px] text-muted-foreground">Run scans on interval without manual trigger</p>
              </div>
              <Switch
                checked={form.auto_run}
                onCheckedChange={(v) => set('auto_run', v)}
                className="scale-75"
                disabled={!form.enabled}
              />
            </div>
            <NumericField
              label="Scan Interval Seconds"
              value={form.scan_interval_seconds}
              onChange={(v) => set('scan_interval_seconds', Math.max(300, Math.min(86400, Math.round(v))))}
              min={300}
              max={86400}
              step={60}
              disabled={!form.enabled}
            />
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <CloudRain className="w-3.5 h-3.5 text-blue-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Signal Thresholds</h4>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField label="Entry Max Price" value={form.entry_max_price} onChange={(v) => set('entry_max_price', v)} min={0.01} max={0.99} step={0.01} disabled={!form.enabled} />
              <NumericField label="Take Profit Price" value={form.take_profit_price} onChange={(v) => set('take_profit_price', v)} min={0.01} max={0.99} step={0.01} disabled={!form.enabled} />
              <NumericField label="Stop Loss %" value={form.stop_loss_pct} onChange={(v) => set('stop_loss_pct', v)} min={0} max={100} step={1} disabled={!form.enabled} />
              <NumericField label="Min Edge %" value={form.min_edge_percent} onChange={(v) => set('min_edge_percent', v)} min={0} max={100} step={0.5} disabled={!form.enabled} />
              <NumericField label="Min Confidence" value={form.min_confidence} onChange={(v) => set('min_confidence', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
              <NumericField label="Min Model Agreement" value={form.min_model_agreement} onChange={(v) => set('min_model_agreement', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
              <NumericField label="Min Liquidity ($)" value={form.min_liquidity} onChange={(v) => set('min_liquidity', v)} min={0} max={1000000} step={10} disabled={!form.enabled} />
              <NumericField label="Max Markets/Scan" value={form.max_markets_per_scan} onChange={(v) => set('max_markets_per_scan', Math.max(10, Math.round(v)))} min={10} max={5000} step={10} disabled={!form.enabled} />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <Bot className="w-3.5 h-3.5 text-violet-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Orchestrator Handoff</h4>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Orchestrator Enabled</p>
                <p className="text-[10px] text-muted-foreground">Allow trader orchestrator to consume weather intents</p>
              </div>
              <Switch
                checked={form.orchestrator_enabled}
                onCheckedChange={(v) => set('orchestrator_enabled', v)}
                className="scale-75"
                disabled={!form.enabled}
              />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Auto Min Edge %"
                value={form.orchestrator_min_edge}
                onChange={(v) => set('orchestrator_min_edge', v)}
                min={0}
                max={100}
                step={0.5}
                disabled={!form.enabled || !form.orchestrator_enabled}
              />
              <NumericField
                label="Auto Max Intent Age (min)"
                value={form.orchestrator_max_age_minutes}
                onChange={(v) => set('orchestrator_max_age_minutes', Math.max(1, Math.round(v)))}
                min={1}
                max={1440}
                step={1}
                disabled={!form.enabled || !form.orchestrator_enabled}
              />
              <NumericField
                label="Default Size ($)"
                value={form.default_size_usd}
                onChange={(v) => set('default_size_usd', v)}
                min={1}
                max={1000}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Max Size ($)"
                value={form.max_size_usd}
                onChange={(v) => set('max_size_usd', v)}
                min={1}
                max={5000}
                step={1}
                disabled={!form.enabled}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <Brain className="w-3.5 h-3.5 text-indigo-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Model Override</h4>
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">Model</Label>
              <Input
                type="text"
                value={form.model || ''}
                onChange={(e) => set('model', e.target.value.trim() || null)}
                placeholder="Default provider adapter model"
                className="mt-1 h-8 text-xs font-mono"
                disabled={!form.enabled}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <Thermometer className="w-3.5 h-3.5 text-orange-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Display</h4>
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">Temperature Unit</Label>
              <div className="flex gap-1 mt-1">
                <Button
                  variant={form.temperature_unit === 'F' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('temperature_unit', 'F')}
                  className={cn(
                    'h-8 px-4 text-xs',
                    form.temperature_unit === 'F'
                      ? 'bg-cyan-600 hover:bg-cyan-500 text-white'
                      : ''
                  )}
                >
                  Fahrenheit
                </Button>
                <Button
                  variant={form.temperature_unit === 'C' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('temperature_unit', 'C')}
                  className={cn(
                    'h-8 px-4 text-xs',
                    form.temperature_unit === 'C'
                      ? 'bg-cyan-600 hover:bg-cyan-500 text-white'
                      : ''
                  )}
                >
                  Celsius
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}

