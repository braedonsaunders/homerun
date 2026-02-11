import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  AlertCircle,
  Brain,
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

export default function AutoTraderSettingsFlyout({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [form, setForm] = useState({
    llm_verify_trades: false,
    llm_verify_strategies: '',
    auto_ai_scoring: false,
    enabled_strategies: [] as string[],
  })
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
    const dedup = new Map<string, string>()
    for (const strategy of strategyList as Strategy[]) {
      const key =
        strategy.is_plugin && strategy.plugin_slug
          ? strategy.plugin_slug
          : strategy.type
      if (!dedup.has(key)) {
        dedup.set(key, strategy.name)
      }
    }
    return Array.from(dedup.entries()).map(([key, label]) => ({ key, label }))
  }, [strategyList])

  useEffect(() => {
    if (!autoTraderStatus?.config || dirty) return
    const cfg = autoTraderStatus.config
    setForm({
      llm_verify_trades: cfg.llm_verify_trades ?? false,
      llm_verify_strategies: (cfg.llm_verify_strategies ?? []).join(', '),
      auto_ai_scoring: cfg.auto_ai_scoring ?? false,
      enabled_strategies: cfg.enabled_strategies ?? [],
    })
  }, [autoTraderStatus?.config, dirty])

  const saveMutation = useMutation({
    mutationFn: (updates: Partial<AutoTraderConfig>) => updateAutoTraderConfig(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
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
    const strategies = form.llm_verify_strategies
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0)

    saveMutation.mutate({
      llm_verify_trades: form.llm_verify_trades,
      llm_verify_strategies: strategies,
      auto_ai_scoring: form.auto_ai_scoring,
      enabled_strategies: form.enabled_strategies,
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
              <div className="flex items-center gap-2">
                <Brain className="w-3.5 h-3.5 text-emerald-400" />
                <h4 className="text-[10px] uppercase tracking-widest font-semibold">AI Execution</h4>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium">LLM Verify Before Trading</p>
                  <p className="text-[10px] text-muted-foreground">
                    Consult AI before each auto-trade. Trades scored as skip/strong_skip are blocked.
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

              {form.llm_verify_trades && (
                <div>
                  <Label className="text-xs text-muted-foreground">Strategies to LLM-Verify</Label>
                  <Input
                    type="text"
                    value={form.llm_verify_strategies}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, llm_verify_strategies: e.target.value }))
                      setDirty(true)
                    }}
                    placeholder="basic, negrisk, miracle (empty = verify all)"
                    className="mt-1 text-sm"
                  />
                  <p className="text-[11px] text-muted-foreground/70 mt-1">
                    Comma-separated strategy types. Leave empty to verify all strategies.
                  </p>
                </div>
              )}

              <Separator className="opacity-30" />

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium">Auto AI Scoring</p>
                  <p className="text-[10px] text-muted-foreground">
                    Automatically AI-score opportunities after each scan cycle.
                  </p>
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

              <div className="flex items-start gap-2 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                <Activity className="w-4 h-4 text-blue-400 mt-0.5 shrink-0" />
                <p className="text-xs text-muted-foreground">
                  LLM verification improves decision quality but increases latency. Disable it for faster reaction time.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none">
            <CardContent className="p-3 space-y-3">
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Enabled Strategies</h4>
                <p className="text-xs text-muted-foreground">Select which strategies AutoTrader can execute</p>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {strategyOptions.map((strategy) => {
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
                    >
                      {strategy.label}
                    </button>
                  )
                })}
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
