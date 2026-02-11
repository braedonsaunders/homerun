import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
  Newspaper,
  Search,
  Brain,
  Bot,
  Zap,
  Target,
  Layers,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import {
  getNewsWorkflowSettings,
  updateNewsWorkflowSettings,
  type NewsWorkflowSettings,
} from '../services/api'

// ─── Helpers ────────────────────────────────────────────────

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

function Section({
  title,
  icon: Icon,
  color,
  children,
}: {
  title: string
  icon: React.ElementType
  color: string
  children: React.ReactNode
}) {
  return (
    <Card className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden">
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border/20">
        <Icon className={cn('w-3.5 h-3.5', color)} />
        <h4 className="text-[10px] uppercase tracking-widest font-semibold">{title}</h4>
      </div>
      <div className="px-3 pb-3 pt-2 space-y-3">{children}</div>
    </Card>
  )
}

// ─── Defaults ───────────────────────────────────────────────

const DEFAULTS: NewsWorkflowSettings = {
  enabled: true,
  auto_run: true,
  top_k: 8,
  rerank_top_n: 5,
  similarity_threshold: 0.35,
  keyword_weight: 0.25,
  semantic_weight: 0.45,
  event_weight: 0.30,
  min_edge_percent: 8.0,
  min_confidence: 0.6,
  require_second_source: false,
  auto_trader_enabled: true,
  auto_trader_min_edge: 10.0,
  auto_trader_max_age_minutes: 120,
  model: null,
}

// ─── Main Component ─────────────────────────────────────────

export default function NewsWorkflowSettingsFlyout({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [form, setForm] = useState<NewsWorkflowSettings>(DEFAULTS)

  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['news-workflow-settings'],
    queryFn: getNewsWorkflowSettings,
    enabled: isOpen,
  })

  useEffect(() => {
    if (settings) {
      setForm({ ...DEFAULTS, ...settings })
    }
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateNewsWorkflowSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-workflow-settings'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
      setSaveMessage({ type: 'success', text: 'Workflow settings saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save' })
      setTimeout(() => setSaveMessage(null), 5000)
    },
  })

  const handleSave = () => {
    saveMutation.mutate(form)
  }

  const set = <K extends keyof NewsWorkflowSettings>(key: K, val: NewsWorkflowSettings[K]) =>
    setForm((p) => ({ ...p, [key]: val }))

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-background/80 z-40 transition-opacity"
        onClick={onClose}
      />
      {/* Drawer */}
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-orange-500" />
            <h3 className="text-sm font-semibold">News Workflow Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending} className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white">
              <Save className="w-3 h-3" /> {saveMutation.isPending ? 'Saving...' : 'Save'}
            </Button>
            <Button variant="ghost" onClick={onClose} className="text-xs h-auto px-2.5 py-1 hover:bg-card">
              <X className="w-3.5 h-3.5 mr-1" /> Close
            </Button>
          </div>
        </div>

        {/* Toast */}
        {saveMessage && (
          <div className={cn(
            "fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300",
            saveMessage.type === 'success'
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
              : "bg-red-500/10 text-red-400 border-red-500/20"
          )}>
            {saveMessage.type === 'success' ? <CheckCircle className="w-4 h-4 shrink-0" /> : <AlertCircle className="w-4 h-4 shrink-0" />}
            {saveMessage.text}
          </div>
        )}

        {/* Content */}
        <div className="p-3 space-y-2">

          {/* Pipeline */}
          <Section title="Pipeline" icon={Newspaper} color="text-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable Workflow</p>
                <p className="text-[10px] text-muted-foreground">Run the full article-to-intent pipeline</p>
              </div>
              <Switch checked={form.enabled} onCheckedChange={(v) => set('enabled', v)} className="scale-75" />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Auto-Run</p>
                <p className="text-[10px] text-muted-foreground">Automatically run on each news scan cycle</p>
              </div>
              <Switch checked={form.auto_run} onCheckedChange={(v) => set('auto_run', v)} className="scale-75" disabled={!form.enabled} />
            </div>
          </Section>

          {/* Retrieval */}
          <Section title="Hybrid Retrieval" icon={Search} color="text-blue-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Control how articles are matched to prediction markets using keyword, semantic, and event-type scoring.
            </p>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField label="Top-K Candidates" help="Max markets per article from retriever" value={form.top_k} onChange={(v) => set('top_k', v)} min={1} max={50} disabled={!form.enabled} />
              <NumericField label="Rerank Top-N" help="Markets sent to LLM reranker" value={form.rerank_top_n} onChange={(v) => set('rerank_top_n', v)} min={1} max={20} disabled={!form.enabled} />
              <NumericField label="Similarity Threshold" help="Min combined score to include" value={form.similarity_threshold} onChange={(v) => set('similarity_threshold', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
            </div>

            <p className="text-[10px] text-muted-foreground/80 font-medium mt-2">Scoring Weights</p>
            <div className="grid grid-cols-3 gap-2.5">
              <NumericField label="Keyword" help="BM25 weight" value={form.keyword_weight} onChange={(v) => set('keyword_weight', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
              <NumericField label="Semantic" help="Embedding weight" value={form.semantic_weight} onChange={(v) => set('semantic_weight', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
              <NumericField label="Event Type" help="Category affinity" value={form.event_weight} onChange={(v) => set('event_weight', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
            </div>
          </Section>

          {/* Edge Detection */}
          <Section title="Edge Detection" icon={Zap} color="text-green-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              LLM probability estimation thresholds. Only findings above these thresholds are marked actionable.
            </p>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField label="Min Edge %" help="Min price divergence to flag" value={form.min_edge_percent} onChange={(v) => set('min_edge_percent', v)} min={0} max={100} step={0.5} disabled={!form.enabled} />
              <NumericField label="Min Confidence" help="Min LLM confidence (0-1)" value={form.min_confidence} onChange={(v) => set('min_confidence', v)} min={0} max={1} step={0.05} disabled={!form.enabled} />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Require Second Source</p>
                <p className="text-[10px] text-muted-foreground">Only flag if 2+ articles match the same market</p>
              </div>
              <Switch checked={form.require_second_source} onCheckedChange={(v) => set('require_second_source', v)} className="scale-75" disabled={!form.enabled} />
            </div>
          </Section>

          {/* Auto-Trader Handoff */}
          <Section title="Auto-Trader Handoff" icon={Bot} color="text-purple-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Control how high-conviction findings are fed to the auto-trader for execution.
            </p>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable Auto-Trading</p>
                <p className="text-[10px] text-muted-foreground">Allow auto-trader to consume news trade intents</p>
              </div>
              <Switch checked={form.auto_trader_enabled} onCheckedChange={(v) => set('auto_trader_enabled', v)} className="scale-75" disabled={!form.enabled} />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField label="Min Edge for Auto-Trade %" help="Higher bar than detection" value={form.auto_trader_min_edge} onChange={(v) => set('auto_trader_min_edge', v)} min={0} max={100} step={0.5} disabled={!form.enabled || !form.auto_trader_enabled} />
              <NumericField label="Max Intent Age (min)" help="Expire stale intents" value={form.auto_trader_max_age_minutes} onChange={(v) => set('auto_trader_max_age_minutes', v)} min={1} max={1440} disabled={!form.enabled || !form.auto_trader_enabled} />
            </div>
            {/* Info */}
            <div className="flex items-start gap-2 p-2.5 bg-purple-500/5 border border-purple-500/20 rounded-lg">
              <Target className="w-3.5 h-3.5 text-purple-400 mt-0.5 shrink-0" />
              <p className="text-[10px] text-muted-foreground">
                Trade intents flow through the auto-trader's full safety pipeline: circuit breakers, risk scoring, AI judge, depth analysis, and position limits.
              </p>
            </div>
          </Section>

          {/* Model Override */}
          <Section title="LLM Model" icon={Brain} color="text-violet-500">
            <div>
              <Label className="text-[11px] text-muted-foreground">Model Override</Label>
              <Input
                type="text"
                value={form.model || ''}
                onChange={(e) => set('model', e.target.value || null)}
                placeholder="Default (from global settings)"
                className="mt-0.5 text-xs h-7 font-mono"
                disabled={!form.enabled}
              />
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                Override the LLM model for event extraction, reranking, and edge estimation. Leave blank to use the globally configured model.
              </p>
            </div>
          </Section>

          {/* Bottom Save */}
          <div className="flex items-center gap-2 pt-1 pb-4">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending} className="gap-1.5">
              <Save className="w-3.5 h-3.5" />
              {saveMutation.isPending ? 'Saving...' : 'Save All Settings'}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}
