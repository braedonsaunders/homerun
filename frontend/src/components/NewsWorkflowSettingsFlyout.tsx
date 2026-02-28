import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
  Newspaper,
  Brain,
  ExternalLink,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import {
  getNewsWorkflowSettings,
  updateNewsWorkflowSettings,
  type NewsWorkflowSettings,
} from '../services/api'
import StrategyConfigSections from './StrategyConfigSections'

const DEFAULTS: NewsWorkflowSettings = {
  enabled: true,
  auto_run: true,
  scan_interval_seconds: 120,
  top_k: 20,
  rerank_top_n: 8,
  similarity_threshold: 0.2,
  keyword_weight: 0.25,
  semantic_weight: 0.45,
  event_weight: 0.3,
  require_verifier: true,
  market_min_liquidity: 500,
  market_max_days_to_resolution: 365,
  min_keyword_signal: 0.04,
  min_semantic_signal: 0.05,
  min_edge_percent: 5,
  min_confidence: 0.45,
  require_second_source: false,
  orchestrator_enabled: true,
  orchestrator_min_edge: 10,
  orchestrator_max_age_minutes: 120,
  model: null,
  article_max_age_hours: 6,
  cycle_spend_cap_usd: 0.25,
  hourly_spend_cap_usd: 2.0,
  cycle_llm_call_cap: 30,
  cache_ttl_minutes: 30,
  max_edge_evals_per_article: 6,
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

export default function NewsWorkflowSettingsFlyout({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [form, setForm] = useState<NewsWorkflowSettings>(DEFAULTS)
  const [lastNonZeroLlmCap, setLastNonZeroLlmCap] = useState(DEFAULTS.cycle_llm_call_cap)
  const [saveMessage, setSaveMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  const { data: settings } = useQuery({
    queryKey: ['news-workflow-settings'],
    queryFn: getNewsWorkflowSettings,
    enabled: isOpen,
  })

  useEffect(() => {
    if (settings) {
      setForm({ ...DEFAULTS, ...settings })
      if (settings.cycle_llm_call_cap > 0) {
        setLastNonZeroLlmCap(settings.cycle_llm_call_cap)
      }
    }
  }, [settings])

  const set = <K extends keyof NewsWorkflowSettings>(
    key: K,
    value: NewsWorkflowSettings[K]
  ) => setForm((prev) => ({ ...prev, [key]: value }))

  const saveMutation = useMutation({
    mutationFn: updateNewsWorkflowSettings,
    onSuccess: (result) => {
      queryClient.setQueryData(['news-workflow-settings'], result.settings)
      queryClient.invalidateQueries({ queryKey: ['news-workflow-settings'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-findings'] })
      setSaveMessage({ type: 'success', text: 'News settings saved' })
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
    const { article_max_age_hours: ignoredArticleMaxAgeHours, ...payload } = form
    void ignoredArticleMaxAgeHours
    saveMutation.mutate(payload)
  }

  const llmCallsEnabled = form.cycle_llm_call_cap > 0

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold">News Workflow Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={handleSave}
              disabled={saveMutation.isPending}
              className="gap-1 text-[10px] h-auto px-3 py-1 bg-orange-500 hover:bg-orange-400 text-white"
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
              <Newspaper className="w-3.5 h-3.5 text-orange-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Pipeline</h4>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable Workflow</p>
                <p className="text-[10px] text-muted-foreground">Turn news scanning and intent generation on/off</p>
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
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Scan Interval Seconds"
                value={form.scan_interval_seconds}
                onChange={(v) => set('scan_interval_seconds', Math.max(30, Math.min(3600, Math.round(v))))}
                min={30}
                max={3600}
                step={10}
                disabled={!form.enabled}
              />
              <NumericField
                label="Article Max Age (Hours)"
                value={form.article_max_age_hours}
                onChange={() => undefined}
                min={1}
                max={48}
                step={1}
                disabled
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <Newspaper className="w-3.5 h-3.5 text-violet-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Retrieval</h4>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Top K Candidates"
                value={form.top_k}
                onChange={(v) => set('top_k', Math.max(1, Math.min(50, Math.round(v))))}
                min={1}
                max={50}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Rerank Top N"
                value={form.rerank_top_n}
                onChange={(v) => set('rerank_top_n', Math.max(1, Math.min(20, Math.round(v))))}
                min={1}
                max={20}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Similarity Threshold"
                value={form.similarity_threshold}
                onChange={(v) => set('similarity_threshold', Math.max(0, Math.min(1, v)))}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Min Keyword Signal"
                value={form.min_keyword_signal}
                onChange={(v) => set('min_keyword_signal', Math.max(0, Math.min(1, v)))}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Min Semantic Signal"
                value={form.min_semantic_signal}
                onChange={(v) => set('min_semantic_signal', Math.max(0, Math.min(1, v)))}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Max Edge Evals / Article"
                value={form.max_edge_evals_per_article}
                onChange={(v) => set('max_edge_evals_per_article', Math.max(1, Math.min(20, Math.round(v))))}
                min={1}
                max={20}
                step={1}
                disabled={!form.enabled}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <Newspaper className="w-3.5 h-3.5 text-sky-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">Scoring Weights</h4>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Keyword Weight"
                value={form.keyword_weight}
                onChange={(v) => set('keyword_weight', Math.max(0, Math.min(1, v)))}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Semantic Weight"
                value={form.semantic_weight}
                onChange={(v) => set('semantic_weight', Math.max(0, Math.min(1, v)))}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Event Weight"
                value={form.event_weight}
                onChange={(v) => set('event_weight', Math.max(0, Math.min(1, v)))}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Cache TTL (Minutes)"
                value={form.cache_ttl_minutes}
                onChange={(v) => set('cache_ttl_minutes', Math.max(1, Math.min(1440, Math.round(v))))}
                min={1}
                max={1440}
                step={1}
                disabled={!form.enabled}
              />
            </div>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 space-y-3">
            <div className="flex items-center gap-2">
              <Brain className="w-3.5 h-3.5 text-amber-400" />
              <h4 className="text-[10px] uppercase tracking-widest font-semibold">LLM Calls</h4>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable News LLM Calls</p>
                <p className="text-[10px] text-muted-foreground">Disables event extraction, rerank, and edge-estimation LLM calls when off</p>
              </div>
              <Switch
                checked={llmCallsEnabled}
                onCheckedChange={(next) => {
                  if (next) {
                    const restoredCap = lastNonZeroLlmCap > 0
                      ? lastNonZeroLlmCap
                      : DEFAULTS.cycle_llm_call_cap
                    set('cycle_llm_call_cap', restoredCap)
                    return
                  }
                  if (form.cycle_llm_call_cap > 0) {
                    setLastNonZeroLlmCap(form.cycle_llm_call_cap)
                  }
                  set('cycle_llm_call_cap', 0)
                }}
                className="scale-75"
                disabled={!form.enabled}
              />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Cycle LLM Call Cap"
                value={form.cycle_llm_call_cap}
                onChange={(v) => {
                  const nextValue = Math.max(0, Math.min(500, Math.round(v)))
                  set('cycle_llm_call_cap', nextValue)
                  if (nextValue > 0) {
                    setLastNonZeroLlmCap(nextValue)
                  }
                }}
                min={0}
                max={500}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Cycle Spend Cap (USD)"
                value={form.cycle_spend_cap_usd}
                onChange={(v) => set('cycle_spend_cap_usd', Math.max(0, Math.min(100, v)))}
                min={0}
                max={100}
                step={0.01}
                disabled={!form.enabled || !llmCallsEnabled}
              />
              <NumericField
                label="Hourly Spend Cap (USD)"
                value={form.hourly_spend_cap_usd}
                onChange={(v) => set('hourly_spend_cap_usd', Math.max(0, Math.min(1000, v)))}
                min={0}
                max={1000}
                step={0.01}
                disabled={!form.enabled || !llmCallsEnabled}
              />
              <NumericField
                label="Min Market Liquidity"
                value={form.market_min_liquidity}
                onChange={(v) => set('market_min_liquidity', Math.max(0, Math.min(1_000_000, v)))}
                min={0}
                max={1_000_000}
                step={10}
                disabled={!form.enabled}
              />
              <NumericField
                label="Max Days To Resolution"
                value={form.market_max_days_to_resolution}
                onChange={(v) => set('market_max_days_to_resolution', Math.max(1, Math.min(3650, Math.round(v))))}
                min={1}
                max={3650}
                step={1}
                disabled={!form.enabled}
              />
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">Model Override</Label>
              <Input
                type="text"
                value={form.model || ''}
                onChange={(e) => set('model', e.target.value.trim() || null)}
                placeholder="Default provider adapter model"
                className="mt-1 h-8 text-xs font-mono"
                disabled={!form.enabled || !llmCallsEnabled}
              />
            </div>
          </Card>

          <StrategyConfigSections sourceKey="news" enabled={isOpen} />

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Strategy Code</p>
                <p className="text-[10px] text-muted-foreground">Edit News opportunity strategies in the Strategy Manager</p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onClose()
                  setTimeout(() => {
                    window.dispatchEvent(new CustomEvent('navigate-to-tab', { detail: 'strategies' }))
                    window.dispatchEvent(new CustomEvent('navigate-strategies-subtab', { detail: { subtab: 'opportunity', sourceFilter: 'news' } }))
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
