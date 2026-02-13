import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
  Zap,
  Clock,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import {
  getSettings,
  updateSettings,
} from '../services/api'

const DEFAULTS = {
  btc_eth_hf_enabled: true,
  btc_eth_hf_series_btc_15m: '10192',
  btc_eth_hf_series_eth_15m: '10191',
  btc_eth_hf_series_sol_15m: '10423',
  btc_eth_hf_series_xrp_15m: '10422',
  btc_eth_hf_series_btc_5m: '10684',
  btc_eth_hf_series_eth_5m: '',
  btc_eth_hf_series_sol_5m: '',
  btc_eth_hf_series_xrp_5m: '',
  btc_eth_hf_series_btc_1h: '10114',
  btc_eth_hf_series_eth_1h: '10117',
  btc_eth_hf_series_sol_1h: '10122',
  btc_eth_hf_series_xrp_1h: '10123',
  btc_eth_hf_series_btc_4h: '10331',
  btc_eth_hf_series_eth_4h: '10332',
  btc_eth_hf_series_sol_4h: '10326',
  btc_eth_hf_series_xrp_4h: '10327',
  btc_eth_pure_arb_max_combined: 0.98,
  btc_eth_dump_hedge_drop_pct: 0.05,
  btc_eth_thin_liquidity_usd: 500,
}

function NumericField({
  label,
  help,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string
  help: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
}) {
  return (
    <div className={cn(disabled && 'opacity-40 pointer-events-none')}>
      <Label className="text-[11px] text-muted-foreground leading-tight">{label}</Label>
      <Input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="mt-0.5 text-xs h-7"
      />
      <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">{help}</p>
    </div>
  )
}

function TextField({
  label,
  help,
  value,
  onChange,
  disabled,
  placeholder,
}: {
  label: string
  help: string
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  placeholder?: string
}) {
  return (
    <div className={cn(disabled && 'opacity-40 pointer-events-none')}>
      <Label className="text-[11px] text-muted-foreground leading-tight">{label}</Label>
      <Input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={placeholder}
        className="mt-0.5 text-xs h-7 font-mono"
      />
      <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">{help}</p>
    </div>
  )
}

function StrategyToggle({
  label,
  enabled,
  onToggle,
  icon: Icon,
  color,
  badge,
}: {
  label: string
  enabled: boolean
  onToggle: (v: boolean) => void
  icon: any
  color: string
  badge?: string
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-1.5">
        <Icon className={cn('w-3.5 h-3.5', color)} />
        <span className="text-xs font-medium">{label}</span>
        {badge && (
          <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
            {badge}
          </span>
        )}
      </div>
      <Switch
        checked={enabled}
        onCheckedChange={onToggle}
        className="scale-75"
      />
    </div>
  )
}

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="bg-card/40 border-border/40 rounded-xl p-3">
      <h4 className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground mb-3 flex items-center gap-1.5">
        <Clock className="w-3.5 h-3.5" />
        {title}
      </h4>
      {children}
    </Card>
  )
}

export default function CryptoSettingsFlyout({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [form, setForm] = useState(DEFAULTS)
  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  useEffect(() => {
    if (settings?.search_filters) {
      setForm((prev) => {
        const merged = { ...prev, ...settings.search_filters } as Record<keyof typeof DEFAULTS, string | number | boolean>
        (Object.keys(DEFAULTS) as Array<keyof typeof DEFAULTS>).forEach((key) => {
          const val = merged[key]
          if (typeof val === 'string' && val === '') {
            merged[key] = DEFAULTS[key]
          }
        })
        return merged as typeof DEFAULTS
      })
    }
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setSaveMessage({ type: 'success', text: 'Crypto settings saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save crypto settings' })
      setTimeout(() => setSaveMessage(null), 5000)
    },
  })

  const handleSave = () => {
    saveMutation.mutate({ search_filters: form })
  }

  const set = <K extends keyof typeof DEFAULTS>(key: K, val: (typeof DEFAULTS)[K]) => {
    setForm((p) => ({ ...p, [key]: val }))
  }

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40 transition-opacity" onClick={onClose} />

      <div className="fixed top-0 right-0 bottom-0 w-full max-w-3xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-purple-500" />
            <h3 className="text-sm font-semibold">Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending} className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white">
              <Save className="w-3 h-3" /> {saveMutation.isPending ? 'Saving...' : 'Save'}
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
              "fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300",
              saveMessage.type === 'success'
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                : "bg-red-500/10 text-red-400 border-red-500/20"
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
          <SectionCard title="BTC/ETH High-Frequency">
            <StrategyToggle
              label="Enable BTC/ETH High-Frequency"
              enabled={form.btc_eth_hf_enabled}
              onToggle={(v) => set('btc_eth_hf_enabled', v)}
              icon={Zap}
              color="text-yellow-400"
              badge="hf"
            />
            <p className="text-[10px] text-muted-foreground/60 mt-2">
              Configure 5m/15m/1h/4h BTC, ETH, SOL, and XRP series IDs used by the crypto strategy.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 mt-2">
              <NumericField
                label="Pure Arb Max Combined"
                help="Pure arb when YES+NO < this"
                value={form.btc_eth_pure_arb_max_combined}
                onChange={(v) => set('btc_eth_pure_arb_max_combined', v)}
                min={0.5}
                max={1}
                step={0.01}
                disabled={!form.btc_eth_hf_enabled}
              />
              <NumericField
                label="Dump-Hedge Drop %"
                help="Min drop to trigger hedge"
                value={form.btc_eth_dump_hedge_drop_pct}
                onChange={(v) => set('btc_eth_dump_hedge_drop_pct', v)}
                min={0.01}
                max={0.5}
                step={0.01}
                disabled={!form.btc_eth_hf_enabled}
              />
              <NumericField
                label="Thin Liquidity ($)"
                help="Below = thin order book"
                value={form.btc_eth_thin_liquidity_usd}
                onChange={(v) => set('btc_eth_thin_liquidity_usd', v)}
                min={0}
                disabled={!form.btc_eth_hf_enabled}
              />
            </div>
          </SectionCard>

          <SectionCard title="15m Series IDs">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <TextField
                label="BTC 15m"
                help="Series ID"
                value={form.btc_eth_hf_series_btc_15m}
                onChange={(v) => set('btc_eth_hf_series_btc_15m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10192"
              />
              <TextField
                label="ETH 15m"
                help="Series ID"
                value={form.btc_eth_hf_series_eth_15m}
                onChange={(v) => set('btc_eth_hf_series_eth_15m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10191"
              />
              <TextField
                label="SOL 15m"
                help="Series ID"
                value={form.btc_eth_hf_series_sol_15m}
                onChange={(v) => set('btc_eth_hf_series_sol_15m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10423"
              />
              <TextField
                label="XRP 15m"
                help="Series ID"
                value={form.btc_eth_hf_series_xrp_15m}
                onChange={(v) => set('btc_eth_hf_series_xrp_15m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10422"
              />
            </div>
          </SectionCard>

          <SectionCard title="5m Series IDs">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <TextField
                label="BTC 5m"
                help="Series ID"
                value={form.btc_eth_hf_series_btc_5m}
                onChange={(v) => set('btc_eth_hf_series_btc_5m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10684"
              />
              <TextField
                label="ETH 5m"
                help="Series ID"
                value={form.btc_eth_hf_series_eth_5m}
                onChange={(v) => set('btc_eth_hf_series_eth_5m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder=""
              />
              <TextField
                label="SOL 5m"
                help="Series ID"
                value={form.btc_eth_hf_series_sol_5m}
                onChange={(v) => set('btc_eth_hf_series_sol_5m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder=""
              />
              <TextField
                label="XRP 5m"
                help="Series ID"
                value={form.btc_eth_hf_series_xrp_5m}
                onChange={(v) => set('btc_eth_hf_series_xrp_5m', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder=""
              />
            </div>
          </SectionCard>

          <SectionCard title="1h Series IDs">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <TextField
                label="BTC 1h"
                help="Series ID"
                value={form.btc_eth_hf_series_btc_1h}
                onChange={(v) => set('btc_eth_hf_series_btc_1h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10114"
              />
              <TextField
                label="ETH 1h"
                help="Series ID"
                value={form.btc_eth_hf_series_eth_1h}
                onChange={(v) => set('btc_eth_hf_series_eth_1h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10117"
              />
              <TextField
                label="SOL 1h"
                help="Series ID"
                value={form.btc_eth_hf_series_sol_1h}
                onChange={(v) => set('btc_eth_hf_series_sol_1h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10122"
              />
              <TextField
                label="XRP 1h"
                help="Series ID"
                value={form.btc_eth_hf_series_xrp_1h}
                onChange={(v) => set('btc_eth_hf_series_xrp_1h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10123"
              />
            </div>
          </SectionCard>

          <SectionCard title="4h Series IDs">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <TextField
                label="BTC 4h"
                help="Series ID"
                value={form.btc_eth_hf_series_btc_4h}
                onChange={(v) => set('btc_eth_hf_series_btc_4h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10331"
              />
              <TextField
                label="ETH 4h"
                help="Series ID"
                value={form.btc_eth_hf_series_eth_4h}
                onChange={(v) => set('btc_eth_hf_series_eth_4h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10332"
              />
              <TextField
                label="SOL 4h"
                help="Series ID"
                value={form.btc_eth_hf_series_sol_4h}
                onChange={(v) => set('btc_eth_hf_series_sol_4h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10326"
              />
              <TextField
                label="XRP 4h"
                help="Series ID"
                value={form.btc_eth_hf_series_xrp_4h}
                onChange={(v) => set('btc_eth_hf_series_xrp_4h', v)}
                disabled={!form.btc_eth_hf_enabled}
                placeholder="10327"
              />
            </div>
          </SectionCard>
        </div>
      </div>
    </>
  )
}
