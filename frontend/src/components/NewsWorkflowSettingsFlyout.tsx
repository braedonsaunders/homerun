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
  Timer,
  Shield,
  Plus,
  Trash2,
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
  scan_interval_seconds: 120,
  top_k: 20,
  rerank_top_n: 8,
  similarity_threshold: 0.20,
  keyword_weight: 0.25,
  semantic_weight: 0.45,
  event_weight: 0.30,
  require_verifier: true,
  market_min_liquidity: 500,
  market_max_days_to_resolution: 365,
  min_keyword_signal: 0.04,
  min_semantic_signal: 0.05,
  min_edge_percent: 8.0,
  min_confidence: 0.6,
  require_second_source: false,
  orchestrator_enabled: true,
  orchestrator_min_edge: 10.0,
  orchestrator_max_age_minutes: 120,
  cycle_spend_cap_usd: 0.25,
  hourly_spend_cap_usd: 2.0,
  cycle_llm_call_cap: 30,
  cache_ttl_minutes: 30,
  max_edge_evals_per_article: 6,
  rss_feeds: [],
  rss_enabled: true,
  rss_sources: [],
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
      const normalized: NewsWorkflowSettings = {
        ...DEFAULTS,
        ...settings,
        rss_enabled: settings.rss_enabled ?? settings.gov_rss_enabled ?? DEFAULTS.rss_enabled,
        rss_sources: settings.rss_sources ?? settings.gov_rss_feeds ?? DEFAULTS.rss_sources,
      }
      setForm(normalized)
    }
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateNewsWorkflowSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-workflow-settings'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
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

  const addCustomFeed = () => {
    set('rss_feeds', [
      ...(form.rss_feeds || []),
      {
        id: `custom_${Date.now()}`,
        name: '',
        url: '',
        enabled: true,
        category: '',
      },
    ])
  }

  const updateCustomFeed = (index: number, updates: Partial<NewsWorkflowSettings['rss_feeds'][number]>) => {
    const next = [...(form.rss_feeds || [])]
    const row = next[index]
    if (!row) return
    next[index] = { ...row, ...updates }
    set('rss_feeds', next)
  }

  const removeCustomFeed = (index: number) => {
    set('rss_feeds', (form.rss_feeds || []).filter((_, i) => i !== index))
  }

  const addGovFeed = () => {
    set('rss_sources', [
      ...(form.rss_sources || []),
      {
        id: `rss_${Date.now()}`,
        agency: 'government',
        name: '',
        url: '',
        priority: 'medium',
        country_iso3: 'USA',
        enabled: true,
      },
    ])
  }

  const updateGovFeed = (index: number, updates: Partial<NewsWorkflowSettings['rss_sources'][number]>) => {
    const next = [...(form.rss_sources || [])]
    const row = next[index]
    if (!row) return
    next[index] = { ...row, ...updates }
    set('rss_sources', next)
  }

  const removeGovFeed = (index: number) => {
    set('rss_sources', (form.rss_sources || []).filter((_, i) => i !== index))
  }

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
        <div className="p-3 space-y-2 pb-6">

          {/* Pipeline */}
          <Section title="Pipeline" icon={Newspaper} color="text-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable Workflow</p>
                <p className="text-[10px] text-muted-foreground">Run the full cluster-to-intent pipeline</p>
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
            <NumericField
              label="Scan Interval (sec)"
              help="Worker scan cadence"
              value={form.scan_interval_seconds}
              onChange={(v) => set('scan_interval_seconds', v)}
              min={30}
              max={3600}
              step={10}
              disabled={!form.enabled}
            />
          </Section>

          {/* RSS Feed Configuration */}
          <Section title="RSS Feeds" icon={Newspaper} color="text-emerald-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Configure custom and official RSS sources used by the news ingest pipeline.
            </p>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium">Custom RSS Feeds</p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-6 text-[10px] gap-1"
                  onClick={addCustomFeed}
                >
                  <Plus className="w-3 h-3" />
                  Add
                </Button>
              </div>
              {(form.rss_feeds || []).length === 0 ? (
                <p className="text-[10px] text-muted-foreground/70">No custom feeds configured.</p>
              ) : (
                <div className="space-y-2">
                  {(form.rss_feeds || []).map((feed, idx) => (
                    <div key={feed.id || `rss-${idx}`} className="rounded-lg border border-border/40 p-2 space-y-2">
                      <div className="grid grid-cols-12 gap-2">
                        <div className="col-span-4">
                          <Label className="text-[10px] text-muted-foreground">Name</Label>
                          <Input
                            value={feed.name || ''}
                            onChange={(e) => updateCustomFeed(idx, { name: e.target.value })}
                            placeholder="Feed label"
                            className="h-7 text-xs mt-0.5"
                          />
                        </div>
                        <div className="col-span-6">
                          <Label className="text-[10px] text-muted-foreground">URL</Label>
                          <Input
                            value={feed.url || ''}
                            onChange={(e) => updateCustomFeed(idx, { url: e.target.value })}
                            placeholder="https://example.com/rss.xml"
                            className="h-7 text-xs mt-0.5 font-mono"
                          />
                        </div>
                        <div className="col-span-2 flex items-end justify-end">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2 text-red-400 hover:text-red-300"
                            onClick={() => removeCustomFeed(idx)}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      </div>
                      <div className="grid grid-cols-12 gap-2 items-center">
                        <div className="col-span-6">
                          <Label className="text-[10px] text-muted-foreground">Category (optional)</Label>
                          <Input
                            value={feed.category || ''}
                            onChange={(e) => updateCustomFeed(idx, { category: e.target.value })}
                            placeholder="politics / crypto / world"
                            className="h-7 text-xs mt-0.5"
                          />
                        </div>
                        <div className="col-span-6 flex items-center justify-end gap-2 pt-4">
                          <span className="text-[10px] text-muted-foreground">Enabled</span>
                          <Switch
                            checked={Boolean(feed.enabled)}
                            onCheckedChange={(v) => updateCustomFeed(idx, { enabled: v })}
                            className="scale-75"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2 pt-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium">Official RSS Sources</p>
                  <p className="text-[10px] text-muted-foreground/70">Official policy and agency feeds</p>
                </div>
                <Switch
                  checked={Boolean(form.rss_enabled)}
                  onCheckedChange={(v) => set('rss_enabled', v)}
                  className="scale-75"
                />
              </div>

              <div className="flex items-center justify-between">
                <p className="text-[10px] text-muted-foreground">Configured feeds: {(form.rss_sources || []).length}</p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-6 text-[10px] gap-1"
                  onClick={addGovFeed}
                >
                  <Plus className="w-3 h-3" />
                  Add
                </Button>
              </div>

              {(form.rss_sources || []).length === 0 ? (
                <p className="text-[10px] text-muted-foreground/70">No official RSS sources configured.</p>
              ) : (
                <div className="space-y-2">
                  {(form.rss_sources || []).map((feed, idx) => (
                    <div key={feed.id || `rss-${idx}`} className="rounded-lg border border-border/40 p-2 space-y-2">
                      <div className="grid grid-cols-12 gap-2">
                        <div className="col-span-3">
                          <Label className="text-[10px] text-muted-foreground">Agency</Label>
                          <Input
                            value={feed.agency || ''}
                            onChange={(e) => updateGovFeed(idx, { agency: e.target.value })}
                            placeholder="state_department"
                            className="h-7 text-xs mt-0.5"
                          />
                        </div>
                        <div className="col-span-4">
                          <Label className="text-[10px] text-muted-foreground">Name</Label>
                          <Input
                            value={feed.name || ''}
                            onChange={(e) => updateGovFeed(idx, { name: e.target.value })}
                            placeholder="Feed name"
                            className="h-7 text-xs mt-0.5"
                          />
                        </div>
                        <div className="col-span-4">
                          <Label className="text-[10px] text-muted-foreground">Country ISO3</Label>
                          <Input
                            value={feed.country_iso3 || ''}
                            onChange={(e) => updateGovFeed(idx, { country_iso3: e.target.value.toUpperCase() })}
                            placeholder="USA"
                            className="h-7 text-xs mt-0.5 font-mono"
                            maxLength={3}
                          />
                        </div>
                        <div className="col-span-1 flex items-end justify-end">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2 text-red-400 hover:text-red-300"
                            onClick={() => removeGovFeed(idx)}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      </div>
                      <div className="grid grid-cols-12 gap-2 items-center">
                        <div className="col-span-8">
                          <Label className="text-[10px] text-muted-foreground">URL</Label>
                          <Input
                            value={feed.url || ''}
                            onChange={(e) => updateGovFeed(idx, { url: e.target.value })}
                            placeholder="https://agency.gov/feed.xml"
                            className="h-7 text-xs mt-0.5 font-mono"
                          />
                        </div>
                        <div className="col-span-2">
                          <Label className="text-[10px] text-muted-foreground">Priority</Label>
                          <Input
                            value={feed.priority || 'medium'}
                            onChange={(e) => updateGovFeed(idx, { priority: e.target.value as 'critical' | 'high' | 'medium' | 'low' })}
                            placeholder="medium"
                            className="h-7 text-xs mt-0.5"
                          />
                        </div>
                        <div className="col-span-2 flex items-center justify-end gap-2 pt-4">
                          <span className="text-[10px] text-muted-foreground">On</span>
                          <Switch
                            checked={Boolean(feed.enabled)}
                            onCheckedChange={(v) => updateGovFeed(idx, { enabled: v })}
                            className="scale-75"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Section>

          {/* Retrieval */}
          <Section title="Hybrid Retrieval" icon={Search} color="text-blue-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Control how clustered news topics are matched to prediction markets using keyword, semantic, and event alignment scoring.
            </p>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField label="Top-K Candidates" help="Max markets per cluster from retriever" value={form.top_k} onChange={(v) => set('top_k', v)} min={1} max={50} disabled={!form.enabled} />
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

          {/* Precision Guards */}
          <Section title="Precision Guards" icon={Shield} color="text-cyan-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Strict market/event compatibility guards to reduce false positives.
            </p>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Market Min Liquidity"
                help="Exclude thin markets from matching universe"
                value={form.market_min_liquidity}
                onChange={(v) => set('market_min_liquidity', v)}
                min={0}
                max={1000000}
                step={50}
                disabled={!form.enabled}
              />
              <NumericField
                label="Max Days To Resolution"
                help="Ignore far-expiry markets for current news"
                value={form.market_max_days_to_resolution}
                onChange={(v) => set('market_max_days_to_resolution', v)}
                min={1}
                max={3650}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Min Keyword Signal"
                help="Minimum lexical strength to keep candidate"
                value={form.min_keyword_signal}
                onChange={(v) => set('min_keyword_signal', v)}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
              <NumericField
                label="Min Semantic Signal"
                help="Minimum embedding similarity to keep candidate"
                value={form.min_semantic_signal}
                onChange={(v) => set('min_semantic_signal', v)}
                min={0}
                max={1}
                step={0.01}
                disabled={!form.enabled}
              />
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

          {/* Orchestrator Handoff */}
          <Section title="Orchestrator Handoff" icon={Bot} color="text-purple-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Control how high-conviction findings are fed to the trader orchestrator for execution.
            </p>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium">Enable Auto-Trading</p>
                <p className="text-[10px] text-muted-foreground">Allow trader orchestrator to consume news trade intents</p>
              </div>
              <Switch checked={form.orchestrator_enabled} onCheckedChange={(v) => set('orchestrator_enabled', v)} className="scale-75" disabled={!form.enabled} />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField label="Min Edge for Orchestrator Trade %" help="Higher bar than detection" value={form.orchestrator_min_edge} onChange={(v) => set('orchestrator_min_edge', v)} min={0} max={100} step={0.5} disabled={!form.enabled || !form.orchestrator_enabled} />
              <NumericField label="Max Intent Age (min)" help="Expire stale intents" value={form.orchestrator_max_age_minutes} onChange={(v) => set('orchestrator_max_age_minutes', v)} min={1} max={1440} disabled={!form.enabled || !form.orchestrator_enabled} />
            </div>
            {/* Info */}
            <div className="flex items-start gap-2 p-2.5 bg-purple-500/5 border border-purple-500/20 rounded-lg">
              <Target className="w-3.5 h-3.5 text-purple-400 mt-0.5 shrink-0" />
              <p className="text-[10px] text-muted-foreground">
                Trade intents flow through the trader orchestrator's full safety pipeline: circuit breakers, risk scoring, AI judge, depth analysis, and position limits.
              </p>
            </div>
          </Section>

          {/* Model Override */}
          <Section title="LLM Budget Guards" icon={Timer} color="text-amber-500">
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Hard caps applied on top of global AI monthly limits to control workflow spend and call volume.
            </p>
            <div className="grid grid-cols-2 gap-2.5">
              <NumericField
                label="Cycle Spend Cap ($)"
                help="Max LLM spend per cycle"
                value={form.cycle_spend_cap_usd}
                onChange={(v) => set('cycle_spend_cap_usd', v)}
                min={0}
                max={100}
                step={0.05}
                disabled={!form.enabled}
              />
              <NumericField
                label="Hourly Spend Cap ($)"
                help="Max workflow spend per rolling hour"
                value={form.hourly_spend_cap_usd}
                onChange={(v) => set('hourly_spend_cap_usd', v)}
                min={0}
                max={1000}
                step={0.1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Cycle LLM Call Cap"
                help="Max provider calls per cycle"
                value={form.cycle_llm_call_cap}
                onChange={(v) => set('cycle_llm_call_cap', v)}
                min={0}
                max={500}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Cache TTL (min)"
                help="Reuse edge estimations in this window"
                value={form.cache_ttl_minutes}
                onChange={(v) => set('cache_ttl_minutes', v)}
                min={1}
                max={1440}
                step={1}
                disabled={!form.enabled}
              />
              <NumericField
                label="Max Edge Evals / Cluster"
                help="Cap expensive edge calls per cluster"
                value={form.max_edge_evals_per_article}
                onChange={(v) => set('max_edge_evals_per_article', v)}
                min={1}
                max={20}
                step={1}
                disabled={!form.enabled}
              />
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

        </div>
      </div>
    </>
  )
}
