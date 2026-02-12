import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertCircle,
  CheckCircle,
  Save,
  SlidersHorizontal,
  X,
} from 'lucide-react'

import { cn } from '../../lib/utils'
import {
  getAutoTraderStatus,
  getStrategies,
  updateAutoTraderConfig,
  type AutoTraderConfig,
  type Strategy,
} from '../../services/api'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Card, CardContent } from '../ui/card'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Separator } from '../ui/separator'
import { Switch } from '../ui/switch'

interface ConfigForm {
  mode: string
  run_interval_seconds: number
  trading_domains: string[]
  enabled_strategies: string[]
  llm_verify_trades: boolean
  llm_verify_strategies: string
  auto_ai_scoring: boolean
  paper_account_id: string
  paper_enable_spread_exits: boolean
  paper_take_profit_pct: number
  paper_stop_loss_pct: number
  max_daily_loss_usd: number
  max_concurrent_positions: number
  max_per_market_exposure: number
  max_per_event_exposure: number
  news_workflow_enabled: boolean
  weather_workflow_enabled: boolean
}

const DEFAULT_FORM: ConfigForm = {
  mode: 'paper',
  run_interval_seconds: 2,
  trading_domains: ['event_markets', 'crypto'],
  enabled_strategies: [],
  llm_verify_trades: false,
  llm_verify_strategies: '',
  auto_ai_scoring: false,
  paper_account_id: '',
  paper_enable_spread_exits: true,
  paper_take_profit_pct: 5,
  paper_stop_loss_pct: 10,
  max_daily_loss_usd: 0,
  max_concurrent_positions: 0,
  max_per_market_exposure: 0,
  max_per_event_exposure: 0,
  news_workflow_enabled: true,
  weather_workflow_enabled: true,
}

function toNumber(value: string, fallback = 0): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const DOMAIN_LABELS: Record<string, string> = {
  event_markets: 'Event Markets',
  crypto: 'Crypto 15m',
}

export default function AutoTraderSettingsFlyout({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [form, setForm] = useState<ConfigForm>(DEFAULT_FORM)
  const [dirty, setDirty] = useState(false)
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const { data: autoTraderStatus } = useQuery({
    queryKey: ['auto-trader-status'],
    queryFn: getAutoTraderStatus,
    enabled: isOpen,
  })

  const { data: strategyList = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: getStrategies,
    enabled: isOpen,
    staleTime: 60000,
  })

  const strategyOptions = useMemo(() => {
    const dedup = new Map<string, Strategy>()
    for (const strategy of strategyList as Strategy[]) {
      const key =
        strategy.is_plugin && strategy.plugin_slug
          ? strategy.plugin_slug
          : strategy.type
      if (!dedup.has(key)) {
        dedup.set(key, strategy)
      }
    }
    return Array.from(dedup.entries()).map(([key, strategy]) => ({
      key,
      label: strategy.name,
      description: strategy.description,
      domain: strategy.domain || 'event_markets',
      timeframe: strategy.timeframe || 'event',
      sources: strategy.sources || [],
      validationStatus: strategy.validation_status || 'unknown',
    }))
  }, [strategyList])

  const groupedStrategies = useMemo(() => {
    const groups = new Map<string, typeof strategyOptions>()
    for (const strategy of strategyOptions) {
      const groupKey = strategy.domain || 'event_markets'
      if (!groups.has(groupKey)) {
        groups.set(groupKey, [])
      }
      groups.get(groupKey)?.push(strategy)
    }
    for (const strategies of groups.values()) {
      strategies.sort((a, b) => a.label.localeCompare(b.label))
    }
    return Array.from(groups.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([domain, strategies]) => ({ domain, strategies }))
  }, [strategyOptions])

  useEffect(() => {
    if (!autoTraderStatus?.config || dirty) return
    const cfg = autoTraderStatus.config
    setForm({
      mode: cfg.mode || 'paper',
      run_interval_seconds: Number(cfg.run_interval_seconds || 2),
      trading_domains: Array.isArray(cfg.trading_domains) && cfg.trading_domains.length > 0
        ? cfg.trading_domains
        : ['event_markets', 'crypto'],
      enabled_strategies: cfg.enabled_strategies || [],
      llm_verify_trades: Boolean(cfg.llm_verify_trades),
      llm_verify_strategies: (cfg.llm_verify_strategies || []).join(', '),
      auto_ai_scoring: Boolean(cfg.auto_ai_scoring),
      paper_account_id: String(cfg.paper_account_id || ''),
      paper_enable_spread_exits: Boolean(cfg.paper_enable_spread_exits ?? true),
      paper_take_profit_pct: Number(cfg.paper_take_profit_pct ?? 5),
      paper_stop_loss_pct: Number(cfg.paper_stop_loss_pct ?? 10),
      max_daily_loss_usd: Number(cfg.max_daily_loss_usd || 0),
      max_concurrent_positions: Number(cfg.max_concurrent_positions || 0),
      max_per_market_exposure: Number(cfg.max_per_market_exposure || 0),
      max_per_event_exposure: Number(cfg.max_per_event_exposure || 0),
      news_workflow_enabled: Boolean(cfg.news_workflow_enabled),
      weather_workflow_enabled: Boolean(cfg.weather_workflow_enabled),
    })
  }, [autoTraderStatus?.config, dirty])

  const saveMutation = useMutation({
    mutationFn: (updates: Partial<AutoTraderConfig> & { requested_by?: string; reason?: string }) =>
      updateAutoTraderConfig(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-overview'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-events'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-metrics'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-exposure'] })
      setDirty(false)
      setSaveMessage({ type: 'success', text: 'AutoTrader settings saved' })
      setTimeout(() => setSaveMessage(null), 2500)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error?.message || 'Failed to save AutoTrader settings' })
      setTimeout(() => setSaveMessage(null), 4000)
    },
  })

  const handleSave = () => {
    const verifyStrategies = form.llm_verify_strategies
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0)

    saveMutation.mutate({
      mode: form.mode,
      run_interval_seconds: form.run_interval_seconds,
      trading_domains: form.trading_domains,
      enabled_strategies: form.enabled_strategies,
      llm_verify_trades: form.llm_verify_trades,
      llm_verify_strategies: verifyStrategies,
      auto_ai_scoring: form.auto_ai_scoring,
      paper_account_id: form.paper_account_id || null,
      paper_enable_spread_exits: form.paper_enable_spread_exits,
      paper_take_profit_pct: form.paper_take_profit_pct,
      paper_stop_loss_pct: form.paper_stop_loss_pct,
      max_daily_loss_usd: form.max_daily_loss_usd,
      max_concurrent_positions: form.max_concurrent_positions,
      max_per_market_exposure: form.max_per_market_exposure,
      max_per_event_exposure: form.max_per_event_exposure,
      news_workflow_enabled: form.news_workflow_enabled,
      weather_workflow_enabled: form.weather_workflow_enabled,
      reason: 'settings_flyout_save',
    })
  }

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40" onClick={onClose} />

      <div className="fixed top-0 right-0 bottom-0 w-full max-w-xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold">AutoTrader Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={handleSave}
              disabled={saveMutation.isPending || !dirty}
              className="gap-1 text-[10px] h-auto px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white"
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
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none">
            <CardContent className="p-3 space-y-3">
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Execution Control</h4>
                <p className="text-xs text-muted-foreground">All fields below map directly to persisted backend config.</p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs text-muted-foreground">Mode</Label>
                  <select
                    value={form.mode}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, mode: e.target.value }))
                      setDirty(true)
                    }}
                    className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs"
                  >
                    <option value="paper">paper</option>
                    <option value="shadow">shadow</option>
                    <option value="live">live</option>
                    <option value="mock">mock</option>
                  </select>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Run Interval (sec)</Label>
                  <Input
                    type="number"
                    min={1}
                    max={60}
                    value={form.run_interval_seconds}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, run_interval_seconds: toNumber(e.target.value, 2) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
              </div>

              <div>
                <Label className="text-xs text-muted-foreground">Trading Domains</Label>
                <div className="mt-1 grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {(['event_markets', 'crypto'] as const).map((domain) => {
                    const enabled = form.trading_domains.includes(domain)
                    return (
                      <button
                        key={domain}
                        type="button"
                        onClick={() => {
                          setForm((prev) => {
                            const hasDomain = prev.trading_domains.includes(domain)
                            const nextDomains = hasDomain
                              ? prev.trading_domains.filter((item) => item !== domain)
                              : [...prev.trading_domains, domain]
                            return {
                              ...prev,
                              trading_domains: nextDomains.length > 0 ? nextDomains : prev.trading_domains,
                            }
                          })
                          setDirty(true)
                        }}
                        className={cn(
                          'rounded-md border px-2 py-1.5 text-xs text-left transition-colors',
                          enabled
                            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                            : 'border-border bg-background text-muted-foreground hover:border-emerald-500/20'
                        )}
                      >
                        {DOMAIN_LABELS[domain]}
                      </button>
                    )
                  })}
                </div>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Event markets and crypto can run together when both are enabled.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2">
                  <Label className="text-xs text-muted-foreground">Paper Account ID</Label>
                  <Input
                    type="text"
                    value={form.paper_account_id}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, paper_account_id: e.target.value }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Paper Take Profit (%)</Label>
                  <Input
                    type="number"
                    min={0}
                    step="0.1"
                    value={form.paper_take_profit_pct}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, paper_take_profit_pct: toNumber(e.target.value, 5) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Paper Stop Loss (%)</Label>
                  <Input
                    type="number"
                    min={0}
                    step="0.1"
                    value={form.paper_stop_loss_pct}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, paper_stop_loss_pct: toNumber(e.target.value, 10) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 px-2 py-1.5">
                <Label className="text-xs text-muted-foreground">Enable Paper TP/SL Exits</Label>
                <Switch
                  checked={form.paper_enable_spread_exits}
                  onCheckedChange={(checked) => {
                    setForm((prev) => ({ ...prev, paper_enable_spread_exits: checked }))
                    setDirty(true)
                  }}
                />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none">
            <CardContent className="p-3 space-y-3">
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Risk Limits</h4>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs text-muted-foreground">Max Daily Loss (USD)</Label>
                  <Input
                    type="number"
                    min={0}
                    step="1"
                    value={form.max_daily_loss_usd}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, max_daily_loss_usd: toNumber(e.target.value) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Max Open Positions</Label>
                  <Input
                    type="number"
                    min={0}
                    step="1"
                    value={form.max_concurrent_positions}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, max_concurrent_positions: toNumber(e.target.value) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Max Per Market Exposure (USD)</Label>
                  <Input
                    type="number"
                    min={0}
                    step="1"
                    value={form.max_per_market_exposure}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, max_per_market_exposure: toNumber(e.target.value) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Max Per Event Exposure (USD)</Label>
                  <Input
                    type="number"
                    min={0}
                    step="1"
                    value={form.max_per_event_exposure}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, max_per_event_exposure: toNumber(e.target.value) }))
                      setDirty(true)
                    }}
                    className="mt-1 h-8 text-xs"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none">
            <CardContent className="p-3 space-y-3">
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">AI + Source Controls</h4>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium">LLM Verify Trades (Legacy)</p>
                  <p className="text-[10px] text-muted-foreground">
                    Applies to legacy scanner opportunity flow only; source workflows can still use their own LLM steps.
                  </p>
                </div>
                <Switch
                  checked={form.llm_verify_trades}
                  onCheckedChange={(checked) => {
                    setForm((prev) => ({ ...prev, llm_verify_trades: checked }))
                    setDirty(true)
                  }}
                  className="scale-75"
                />
              </div>

              <div>
                <Label className="text-xs text-muted-foreground">LLM Verify Strategies (Legacy)</Label>
                <Input
                  type="text"
                  value={form.llm_verify_strategies}
                  onChange={(e) => {
                    setForm((prev) => ({ ...prev, llm_verify_strategies: e.target.value }))
                    setDirty(true)
                  }}
                  placeholder="basic, negrisk"
                  className="mt-1 h-8 text-xs"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium">Auto AI Scoring</p>
                  <p className="text-[10px] text-muted-foreground">Enable AI scoring on incoming opportunities.</p>
                </div>
                <Switch
                  checked={form.auto_ai_scoring}
                  onCheckedChange={(checked) => {
                    setForm((prev) => ({ ...prev, auto_ai_scoring: checked }))
                    setDirty(true)
                  }}
                  className="scale-75"
                />
              </div>

              <Separator className="opacity-30" />

              <div className="grid grid-cols-1 gap-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs">News Workflow Enabled</p>
                  <Switch
                    checked={form.news_workflow_enabled}
                    onCheckedChange={(checked) => {
                      setForm((prev) => ({ ...prev, news_workflow_enabled: checked }))
                      setDirty(true)
                    }}
                    className="scale-75"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <p className="text-xs">Weather Workflow Enabled</p>
                  <Switch
                    checked={form.weather_workflow_enabled}
                    onCheckedChange={(checked) => {
                      setForm((prev) => ({ ...prev, weather_workflow_enabled: checked }))
                      setDirty(true)
                    }}
                    className="scale-75"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none">
            <CardContent className="p-3 space-y-3">
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Enabled Strategies</h4>
                <p className="text-xs text-muted-foreground">Only selected strategies are eligible for execution.</p>
              </div>

              <div className="space-y-2">
                {groupedStrategies.map((group) => (
                  <div key={group.domain} className="rounded-lg border border-border/60 bg-background/40 p-2">
                    <div className="mb-1.5 flex items-center justify-between">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                        {DOMAIN_LABELS[group.domain] || group.domain}
                      </p>
                      <Badge variant="outline" className="text-[9px] bg-background/70">
                        {group.strategies.length}
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {group.strategies.map((strategy) => {
                        const enabled = form.enabled_strategies.includes(strategy.key)
                        return (
                          <button
                            key={strategy.key}
                            type="button"
                            onClick={() => {
                              setForm((prev) => ({
                                ...prev,
                                enabled_strategies: enabled
                                  ? prev.enabled_strategies.filter((k) => k !== strategy.key)
                                  : [...prev.enabled_strategies, strategy.key],
                              }))
                              setDirty(true)
                            }}
                            className={cn(
                              'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                              enabled
                                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                : 'bg-muted text-muted-foreground border-border hover:border-emerald-500/20'
                            )}
                            title={`${strategy.timeframe} • ${strategy.sources.join(', ')}`}
                          >
                            {strategy.label}
                            {strategy.validationStatus === 'demoted' ? ' (demoted)' : ''}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs h-7"
                  onClick={() => {
                    setForm((prev) => ({ ...prev, enabled_strategies: strategyOptions.map((s) => s.key) }))
                    setDirty(true)
                  }}
                >
                  Select All
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs h-7"
                  onClick={() => {
                    setForm((prev) => ({ ...prev, enabled_strategies: [] }))
                    setDirty(true)
                  }}
                >
                  Clear All
                </Button>
                {dirty && (
                  <Badge variant="outline" className="text-[10px] text-yellow-400 border-yellow-500/30 bg-yellow-500/10">
                    Unsaved changes
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  )
}
