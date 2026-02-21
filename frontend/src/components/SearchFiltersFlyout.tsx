import { useEffect, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
  Shield,
  BarChart3,
  Puzzle,
  ChevronDown,
  ChevronRight,
  ExternalLink,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import {
  getSettings,
  updateSettings,
} from '../services/api'
import StrategyConfigSections from './StrategyConfigSections'

const DEFAULTS = {
  min_liquidity_hard: 1000,
  min_position_size: 50,
  min_absolute_profit: 10,
  min_annualized_roi: 10,
  max_resolution_months: 18,
  max_plausible_roi: 30,
  max_trade_legs: 6,
  min_liquidity_per_leg: 500,
  risk_very_short_days: 2,
  risk_short_days: 7,
  risk_long_lockup_days: 180,
  risk_extended_lockup_days: 90,
  risk_low_liquidity: 1000,
  risk_moderate_liquidity: 5000,
  risk_complex_legs: 5,
  risk_multiple_legs: 3,
}

const SEARCH_FILTER_KEYS = Object.keys(DEFAULTS) as Array<keyof typeof DEFAULTS>

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
  help: string
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
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        className="mt-0.5 text-xs h-7"
      />
      <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">{help}</p>
    </div>
  )
}

function CollapsibleSection({
  title,
  icon: Icon,
  color,
  children,
  defaultOpen = false,
  count,
}: {
  title: string
  icon: any
  color: string
  children: ReactNode
  defaultOpen?: boolean
  count?: number
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <Card className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <Icon className={cn('w-3.5 h-3.5', color)} />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">{title}</h4>
          {count !== undefined && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/60 text-muted-foreground">
              {count}
            </span>
          )}
        </div>
        {open
          ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
          : <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />}
      </button>
      {open && <div className="px-3 pb-3 space-y-3">{children}</div>}
    </Card>
  )
}

export default function SearchFiltersFlyout({
  isOpen,
  onClose,
  onManageStrategies,
}: {
  isOpen: boolean
  onClose: () => void
  onManageStrategies?: () => void
}) {
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [form, setForm] = useState(DEFAULTS)
  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
    enabled: isOpen,
  })

  useEffect(() => {
    if (!settings?.search_filters) return

    setForm(() => {
      const next = { ...DEFAULTS }
      SEARCH_FILTER_KEYS.forEach((key) => {
        const value = settings.search_filters[key]
        if (value !== undefined && value !== null) {
          ;(next as any)[key] = value
        }
      })
      return next
    })
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setSaveMessage({ type: 'success', text: 'Settings saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save settings' })
      setTimeout(() => setSaveMessage(null), 5000)
    },
  })

  const handleSave = () => {
    const payload = SEARCH_FILTER_KEYS.reduce((acc, key) => {
      ;(acc as any)[key] = form[key]
      return acc
    }, {} as Partial<typeof DEFAULTS>)
    saveMutation.mutate({ search_filters: payload })
  }

  const set = <K extends keyof typeof DEFAULTS>(key: K, val: (typeof DEFAULTS)[K]) => {
    setForm((prev) => ({ ...prev, [key]: val }))
  }

  if (!isOpen) return null

  return (
    <>
      <div
        className="fixed inset-0 bg-background/80 z-40 transition-opacity"
        onClick={onClose}
      />

      <div className="fixed top-0 right-0 bottom-0 w-full max-w-3xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-orange-500" />
            <h3 className="text-sm font-semibold">Market Settings</h3>
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
              {SEARCH_FILTER_KEYS.length} global filters
            </span>
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
          <div className={cn(
            'fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300',
            saveMessage.type === 'success'
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-red-500/10 text-red-400 border-red-500/20',
          )}>
            {saveMessage.type === 'success'
              ? <CheckCircle className="w-4 h-4 shrink-0" />
              : <AlertCircle className="w-4 h-4 shrink-0" />}
            {saveMessage.text}
          </div>
        )}

        <div className="p-3 space-y-2 pb-6">
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <p className="text-[11px] text-muted-foreground/80 leading-relaxed">
              Global scanner thresholds are configured here. Strategy-specific detection and execution filters are configured below per strategy via StrategySDK config schemas.
            </p>
          </Card>

          <CollapsibleSection title="Global Rejection Filters" icon={Shield} color="text-red-500" defaultOpen={true} count={8}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Hard rejection thresholds applied to every opportunity before execution.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <NumericField label="Min Liquidity ($)" help="Hard reject below this" value={form.min_liquidity_hard} onChange={(v) => set('min_liquidity_hard', v)} min={0} />
              <NumericField label="Min Position Size ($)" help="Reject if max position below" value={form.min_position_size} onChange={(v) => set('min_position_size', v)} min={0} />
              <NumericField label="Min Absolute Profit ($)" help="Reject if net profit below" value={form.min_absolute_profit} onChange={(v) => set('min_absolute_profit', v)} min={0} step={0.5} />
              <NumericField label="Min Annualized ROI (%)" help="Reject if ROI below" value={form.min_annualized_roi} onChange={(v) => set('min_annualized_roi', v)} min={0} step={1} />
              <NumericField label="Max Resolution (months)" help="Reject if too far out" value={form.max_resolution_months} onChange={(v) => set('max_resolution_months', v)} min={1} max={120} />
              <NumericField label="Max Plausible ROI (%)" help="Above = false positive" value={form.max_plausible_roi} onChange={(v) => set('max_plausible_roi', v)} min={1} />
              <NumericField label="Max Trade Legs" help="Max legs in multi-leg trade" value={form.max_trade_legs} onChange={(v) => set('max_trade_legs', v)} min={2} max={20} />
              <NumericField label="Min Liquidity / Leg ($)" help="Required liquidity per leg" value={form.min_liquidity_per_leg} onChange={(v) => set('min_liquidity_per_leg', v)} min={0} />
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Risk Scoring Thresholds" icon={BarChart3} color="text-orange-500" count={8}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Global risk model thresholds used when scoring opportunities.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <NumericField label="Very Short Days" help="High risk if < this" value={form.risk_very_short_days} onChange={(v) => set('risk_very_short_days', v)} min={0} max={30} />
              <NumericField label="Short Days" help="Moderate risk if < this" value={form.risk_short_days} onChange={(v) => set('risk_short_days', v)} min={1} max={60} />
              <NumericField label="Long Lockup Days" help="High risk if > this" value={form.risk_long_lockup_days} onChange={(v) => set('risk_long_lockup_days', v)} min={30} max={3650} />
              <NumericField label="Extended Lockup Days" help="Moderate risk if > this" value={form.risk_extended_lockup_days} onChange={(v) => set('risk_extended_lockup_days', v)} min={14} max={1825} />
              <NumericField label="Low Liquidity ($)" help="High risk below this" value={form.risk_low_liquidity} onChange={(v) => set('risk_low_liquidity', v)} min={0} />
              <NumericField label="Moderate Liquidity ($)" help="Moderate risk below this" value={form.risk_moderate_liquidity} onChange={(v) => set('risk_moderate_liquidity', v)} min={0} />
              <NumericField label="Complex Legs" help="Above = complex trade risk" value={form.risk_complex_legs} onChange={(v) => set('risk_complex_legs', v)} min={2} max={20} />
              <NumericField label="Multiple Legs" help="Above = multi-position risk" value={form.risk_multiple_legs} onChange={(v) => set('risk_multiple_legs', v)} min={2} max={20} />
            </div>
          </CollapsibleSection>

          <StrategyConfigSections sourceKey="scanner" enabled={isOpen} />

          <CollapsibleSection
            title="Strategy Management"
            icon={Puzzle}
            color="text-violet-500"
            defaultOpen={false}
          >
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Full strategy source and lifecycle management is in the Strategies tab.
            </p>
            <div className="rounded-lg bg-muted/20 p-3 text-center">
              <p className="text-xs text-muted-foreground">Open Strategies to manage all opportunity strategy code and status.</p>
              {onManageStrategies ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onManageStrategies}
                  className="mt-3 gap-1.5 text-[10px]"
                >
                  <ExternalLink className="w-3 h-3" />
                  Open Strategies
                </Button>
              ) : null}
            </div>
          </CollapsibleSection>
        </div>
      </div>
    </>
  )
}
