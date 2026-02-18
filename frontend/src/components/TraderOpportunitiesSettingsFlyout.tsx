import { useEffect, useState } from 'react'
import {
  AlertCircle,
  CheckCircle,
  ExternalLink,
  Filter,
  Save,
  Settings,
  Users,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import StrategyConfigSections from './StrategyConfigSections'

export type TraderOpportunitiesSettingsForm = {
  confluence_limit: number
  individual_trade_limit: number
  individual_trade_min_confidence: number
  individual_trade_max_age_minutes: number
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
        value={Number.isFinite(value) ? value : 0}
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

export default function TraderOpportunitiesSettingsFlyout({
  isOpen,
  onClose,
  initial,
  onSave,
  savePending,
  saveMessage,
}: {
  isOpen: boolean
  onClose: () => void
  initial: TraderOpportunitiesSettingsForm
  onSave: (next: TraderOpportunitiesSettingsForm) => void
  savePending?: boolean
  saveMessage?: { type: 'success' | 'error'; text: string } | null
}) {
  const [form, setForm] = useState<TraderOpportunitiesSettingsForm>(initial)

  useEffect(() => {
    if (!isOpen) return
    setForm(initial)
  }, [initial, isOpen])

  if (!isOpen) return null

  const handleSave = () => {
    onSave({
      confluence_limit: Math.round(clamp(form.confluence_limit, 1, 200)),
      individual_trade_limit: Math.round(clamp(form.individual_trade_limit, 1, 500)),
      individual_trade_min_confidence: clamp(form.individual_trade_min_confidence, 0, 1),
      individual_trade_max_age_minutes: Math.round(clamp(form.individual_trade_max_age_minutes, 1, 1440)),
    })
  }

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40 transition-opacity" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold">Trader Signal Display</h3>
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
            Traders opportunities are fully strategy-owned. Firehose eligibility filters are defined
            in the Traders Confluence strategy config below.
            Auto-trader and copy trading settings are configured in the Auto Trader flyout.
          </p>

          {/* ============ SIGNAL FILTERS ============ */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-orange-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Display Limits</h4>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Confluence Fetch Limit"
                help="1-200"
                value={form.confluence_limit}
                onChange={(v) => setForm((prev) => ({ ...prev, confluence_limit: v }))}
                min={1}
                max={200}
                step={1}
              />
            </div>
          </Card>

          {/* ============ STRATEGY-OWNED FIREHOSE FILTERING ============ */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-amber-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Firehose Filtering Ownership</h4>
            </div>
            <p className="text-[11px] text-muted-foreground/80 leading-relaxed">
              Traders firehose inclusion/exclusion gates (tradability, crypto exclusion, source qualification,
              age windows) are controlled in the <span className="text-foreground">Traders Confluence strategy config</span> below.
            </p>
          </Card>

          {/* Dynamic strategy config sections from config_schema */}
          <StrategyConfigSections sourceKey="traders" enabled={isOpen} />

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Strategy Code</p>
                <p className="text-[10px] text-muted-foreground">Edit the Traders Confluence detector source code</p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onClose()
                  setTimeout(() => {
                    window.dispatchEvent(new CustomEvent('navigate-to-tab', { detail: 'strategies' }))
                    window.dispatchEvent(new CustomEvent('navigate-strategies-subtab', { detail: { subtab: 'opportunity', sourceFilter: 'traders' } }))
                  }, 150)
                }}
                className="gap-1.5 text-[10px] h-7"
              >
                <ExternalLink className="w-3 h-3" />
                Edit Strategy Code
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}
