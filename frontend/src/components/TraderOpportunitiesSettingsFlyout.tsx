import { useEffect, useState } from 'react'
import {
  AlertCircle,
  CheckCircle,
  Filter,
  Save,
  Settings,
  Target,
  Users,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'

export type TraderOpportunitiesSettingsForm = {
  source_filter: 'all' | 'confluence' | 'insider'
  min_tier: 'WATCH' | 'HIGH' | 'EXTREME'
  side_filter: 'all' | 'BUY' | 'SELL'
  confluence_limit: number
  insider_limit: number
  insider_min_confidence: number
  insider_max_age_minutes: number
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
      source_filter: form.source_filter,
      min_tier: form.min_tier,
      side_filter: form.side_filter,
      confluence_limit: Math.round(clamp(form.confluence_limit, 1, 200)),
      insider_limit: Math.round(clamp(form.insider_limit, 1, 500)),
      insider_min_confidence: clamp(form.insider_min_confidence, 0, 1),
      insider_max_age_minutes: Math.round(clamp(form.insider_max_age_minutes, 1, 1440)),
    })
  }

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40 transition-opacity" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold">Trader Opportunities Settings</h3>
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
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-orange-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Confluence + View Filters</h4>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <div>
                <Label className="text-[11px] text-muted-foreground leading-tight">Source Filter</Label>
                <select
                  value={form.source_filter}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      source_filter: e.target.value as TraderOpportunitiesSettingsForm['source_filter'],
                    }))
                  }
                  className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                >
                  <option value="all">All sources</option>
                  <option value="confluence">Confluence only</option>
                  <option value="insider">Insider only</option>
                </select>
                <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">Merged feed source selector</p>
              </div>

              <div>
                <Label className="text-[11px] text-muted-foreground leading-tight">Min Confluence Tier</Label>
                <select
                  value={form.min_tier}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      min_tier: e.target.value as TraderOpportunitiesSettingsForm['min_tier'],
                    }))
                  }
                  className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                >
                  <option value="WATCH">Watch (5+ wallets)</option>
                  <option value="HIGH">High (10+ wallets)</option>
                  <option value="EXTREME">Extreme (15+ wallets)</option>
                </select>
                <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">Applied to tracked-trader endpoint</p>
              </div>

              <div>
                <Label className="text-[11px] text-muted-foreground leading-tight">Side Filter</Label>
                <select
                  value={form.side_filter}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      side_filter: e.target.value as TraderOpportunitiesSettingsForm['side_filter'],
                    }))
                  }
                  className="mt-0.5 h-7 w-full rounded-md border border-border bg-muted px-2 text-xs"
                >
                  <option value="all">All sides</option>
                  <option value="BUY">Buy clusters</option>
                  <option value="SELL">Sell clusters</option>
                </select>
                <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">Maps to insider direction filter</p>
              </div>

              <NumericField
                label="Confluence Fetch Limit"
                help="Backend range: 1-200"
                value={form.confluence_limit}
                onChange={(v) => setForm((prev) => ({ ...prev, confluence_limit: v }))}
                min={1}
                max={200}
                step={1}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-amber-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Insider Feed</h4>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Insider Fetch Limit"
                help="Backend range: 1-500"
                value={form.insider_limit}
                onChange={(v) => setForm((prev) => ({ ...prev, insider_limit: v }))}
                min={1}
                max={500}
                step={1}
              />
              <NumericField
                label="Insider Min Confidence"
                help="Backend range: 0.00-1.00"
                value={form.insider_min_confidence}
                onChange={(v) => setForm((prev) => ({ ...prev, insider_min_confidence: v }))}
                min={0}
                max={1}
                step={0.01}
              />
              <NumericField
                label="Insider Max Age (min)"
                help="Backend range: 1-1440"
                value={form.insider_max_age_minutes}
                onChange={(v) => setForm((prev) => ({ ...prev, insider_max_age_minutes: v }))}
                min={1}
                max={1440}
                step={1}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <div className="flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-cyan-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Backend Constants Applied</h4>
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground/80">
              Tier thresholds are WATCH=5+, HIGH=10+, EXTREME=15+ aligned to confluence detector constants.
              API validation bounds are enforced on save for confluence limit, insider limit, confidence, and max age.
            </p>
          </Card>
        </div>
      </div>
    </>
  )
}
