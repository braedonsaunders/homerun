import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Pause, Play, ShieldAlert, Square, Zap } from 'lucide-react'
import {
  armTraderOrchestratorLiveStart,
  createTraderFromTemplate,
  getTraderDecisions,
  getTraderEvents,
  getTraderOrchestratorOverview,
  getTraderOrders,
  getTraderTemplates,
  getTraders,
  pauseTrader,
  runTraderOnce,
  runTraderOrchestratorLivePreflight,
  setTraderOrchestratorLiveKillSwitch,
  startTrader,
  startTraderOrchestrator,
  startTraderOrchestratorLive,
  stopTraderOrchestrator,
  stopTraderOrchestratorLive,
  type Trader,
  updateTrader,
} from '../services/api'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'

type FeedFilter = 'all' | 'decision' | 'order' | 'event'

export default function TradingPanel() {
  const queryClient = useQueryClient()
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [feedFilter, setFeedFilter] = useState<FeedFilter>('all')
  const [draftName, setDraftName] = useState('')
  const [draftInterval, setDraftInterval] = useState('60')
  const [draftSources, setDraftSources] = useState('')
  const [draftParams, setDraftParams] = useState('{}')
  const [draftRisk, setDraftRisk] = useState('{}')
  const [saveError, setSaveError] = useState<string | null>(null)

  const overviewQuery = useQuery({
    queryKey: ['trader-orchestrator-overview'],
    queryFn: getTraderOrchestratorOverview,
    refetchInterval: 4000,
  })

  const tradersQuery = useQuery({
    queryKey: ['traders-list'],
    queryFn: getTraders,
    refetchInterval: 5000,
  })

  const templatesQuery = useQuery({
    queryKey: ['traders-templates'],
    queryFn: getTraderTemplates,
    staleTime: 60000,
  })

  const traders = tradersQuery.data || []
  const selectedTrader = useMemo(
    () => traders.find((t) => t.id === selectedTraderId) || null,
    [traders, selectedTraderId]
  )

  useEffect(() => {
    if (!selectedTraderId && traders.length > 0) {
      setSelectedTraderId(traders[0].id)
    }
  }, [selectedTraderId, traders])

  useEffect(() => {
    if (!selectedTrader) return
    setDraftName(selectedTrader.name)
    setDraftInterval(String(selectedTrader.interval_seconds || 60))
    setDraftSources((selectedTrader.sources || []).join(', '))
    setDraftParams(JSON.stringify(selectedTrader.params || {}, null, 2))
    setDraftRisk(JSON.stringify(selectedTrader.risk_limits || {}, null, 2))
    setSaveError(null)
  }, [selectedTrader])

  const decisionsQuery = useQuery({
    queryKey: ['traders-decisions', selectedTraderId],
    queryFn: () => getTraderDecisions(String(selectedTraderId), { limit: 120 }),
    enabled: Boolean(selectedTraderId),
    refetchInterval: 3500,
  })

  const ordersQuery = useQuery({
    queryKey: ['traders-orders', selectedTraderId],
    queryFn: () => getTraderOrders(String(selectedTraderId), { limit: 120 }),
    enabled: Boolean(selectedTraderId),
    refetchInterval: 3500,
  })

  const eventsQuery = useQuery({
    queryKey: ['traders-events', selectedTraderId],
    queryFn: () => getTraderEvents(String(selectedTraderId), { limit: 120 }),
    enabled: Boolean(selectedTraderId),
    refetchInterval: 3500,
  })

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
    queryClient.invalidateQueries({ queryKey: ['traders-list'] })
    queryClient.invalidateQueries({ queryKey: ['traders-decisions'] })
    queryClient.invalidateQueries({ queryKey: ['traders-orders'] })
    queryClient.invalidateQueries({ queryKey: ['traders-events'] })
  }

  const startMutation = useMutation({
    mutationFn: () => startTraderOrchestrator({ mode: 'paper' }),
    onSuccess: refreshAll,
  })
  const stopMutation = useMutation({ mutationFn: stopTraderOrchestrator, onSuccess: refreshAll })
  const killOnMutation = useMutation({
    mutationFn: () => setTraderOrchestratorLiveKillSwitch(true),
    onSuccess: refreshAll,
  })
  const killOffMutation = useMutation({
    mutationFn: () => setTraderOrchestratorLiveKillSwitch(false),
    onSuccess: refreshAll,
  })
  const liveStartMutation = useMutation({
    mutationFn: async () => {
      const preflight = await runTraderOrchestratorLivePreflight({ mode: 'live' })
      if (preflight.status !== 'passed') throw new Error('Live preflight failed')
      const armed = await armTraderOrchestratorLiveStart({ preflight_id: preflight.preflight_id })
      return startTraderOrchestratorLive({ arm_token: armed.arm_token, mode: 'live' })
    },
    onSuccess: refreshAll,
  })
  const liveStopMutation = useMutation({ mutationFn: () => stopTraderOrchestratorLive(), onSuccess: refreshAll })
  const traderStartMutation = useMutation({ mutationFn: (id: string) => startTrader(id), onSuccess: refreshAll })
  const traderPauseMutation = useMutation({ mutationFn: (id: string) => pauseTrader(id), onSuccess: refreshAll })
  const traderRunOnceMutation = useMutation({ mutationFn: (id: string) => runTraderOnce(id), onSuccess: refreshAll })
  const saveTraderMutation = useMutation({
    mutationFn: async (trader: Trader) => {
      const params = JSON.parse(draftParams || '{}')
      const risk = JSON.parse(draftRisk || '{}')
      const sources = draftSources.split(',').map((x) => x.trim()).filter(Boolean)
      return updateTrader(trader.id, {
        name: draftName.trim(),
        interval_seconds: Math.max(1, Number(draftInterval || 60)),
        sources,
        params,
        risk_limits: risk,
      })
    },
    onSuccess: refreshAll,
    onError: (error: any) => setSaveError(error?.message || 'Failed to save trader'),
  })

  const deployTemplatesMutation = useMutation({
    mutationFn: async () => {
      const templates = templatesQuery.data || []
      for (const template of templates) {
        await createTraderFromTemplate({ template_id: template.id })
      }
    },
    onSuccess: refreshAll,
  })

  const feedRows = useMemo(() => {
    const decisions = (decisionsQuery.data || []).map((item) => ({
      kind: 'decision',
      ts: item.created_at || '',
      id: item.id,
      title: `${item.decision.toUpperCase()} • ${item.source}`,
      detail: item.reason || '',
    }))
    const orders = (ordersQuery.data || []).map((item) => ({
      kind: 'order',
      ts: item.created_at || '',
      id: item.id,
      title: `${item.status.toUpperCase()} • ${item.market_question || item.market_id}`,
      detail: `$${Number(item.notional_usd || 0).toFixed(2)} • ${item.mode}`,
    }))
    const events = (eventsQuery.data?.events || []).map((item) => ({
      kind: 'event',
      ts: item.created_at || '',
      id: item.id,
      title: `${item.event_type} • ${item.severity}`,
      detail: item.message || '',
    }))
    const merged = [...decisions, ...orders, ...events].sort((a, b) => {
      return new Date(b.ts || 0).getTime() - new Date(a.ts || 0).getTime()
    })
    if (feedFilter === 'all') return merged
    return merged.filter((row) => row.kind === feedFilter)
  }, [decisionsQuery.data, ordersQuery.data, eventsQuery.data, feedFilter])

  const killSwitchOn = Boolean(overviewQuery.data?.control?.kill_switch)
  const globalMode = String(overviewQuery.data?.control?.mode || 'paper')
  const orchestratorRunning = Boolean(overviewQuery.data?.control?.is_enabled) && !Boolean(overviewQuery.data?.control?.is_paused)

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex flex-wrap items-center gap-2">
            <span>Trader Orchestrator Cockpit</span>
            <Badge variant={orchestratorRunning ? 'default' : 'secondary'}>{orchestratorRunning ? 'Running' : 'Stopped'}</Badge>
            <Badge variant={globalMode === 'live' ? 'destructive' : 'outline'}>{globalMode.toUpperCase()}</Badge>
            <Badge variant={killSwitchOn ? 'destructive' : 'outline'}>{killSwitchOn ? 'Kill Switch ON' : 'Kill Switch OFF'}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Button onClick={() => startMutation.mutate()}><Play className="w-4 h-4 mr-2" />Start</Button>
          <Button variant="secondary" onClick={() => stopMutation.mutate()}><Square className="w-4 h-4 mr-2" />Stop</Button>
          <Button variant="outline" onClick={() => liveStartMutation.mutate()}><Zap className="w-4 h-4 mr-2" />Live Start</Button>
          <Button variant="outline" onClick={() => liveStopMutation.mutate()}><Pause className="w-4 h-4 mr-2" />Live Stop</Button>
          <Button variant="destructive" onClick={() => killOnMutation.mutate()}><ShieldAlert className="w-4 h-4 mr-2" />Kill On</Button>
          <Button variant="outline" onClick={() => killOffMutation.mutate()}>Kill Off</Button>
          <Button variant="outline" onClick={() => deployTemplatesMutation.mutate()} disabled={deployTemplatesMutation.isPending}>
            {deployTemplatesMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            Seed Templates
          </Button>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)_360px]">
        <Card>
          <CardHeader><CardTitle>Trader Roster</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {traders.map((trader) => (
              <div key={trader.id} className={`rounded-md border p-2 ${selectedTraderId === trader.id ? 'border-primary' : 'border-border'}`}>
                <button className="w-full text-left" onClick={() => setSelectedTraderId(trader.id)}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-medium text-sm">{trader.name}</div>
                    <Badge variant={trader.is_paused || !trader.is_enabled ? 'secondary' : 'default'}>
                      {trader.is_paused || !trader.is_enabled ? 'Paused' : 'Active'}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">{trader.strategy_key} • {trader.interval_seconds}s</div>
                  <div className="text-xs text-muted-foreground mt-1">{(trader.sources || []).join(', ')}</div>
                </button>
                <div className="mt-2 flex gap-1">
                  <Button size="sm" variant="outline" onClick={() => traderStartMutation.mutate(trader.id)}>Start</Button>
                  <Button size="sm" variant="outline" onClick={() => traderPauseMutation.mutate(trader.id)}>Pause</Button>
                  <Button size="sm" variant="outline" onClick={() => traderRunOnceMutation.mutate(trader.id)}>Run Once</Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between">
              <span>Live Feed</span>
              <div className="flex gap-1">
                {(['all', 'decision', 'order', 'event'] as FeedFilter[]).map((filter) => (
                  <Button key={filter} size="sm" variant={feedFilter === filter ? 'default' : 'outline'} onClick={() => setFeedFilter(filter)}>
                    {filter}
                  </Button>
                ))}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 max-h-[65vh] overflow-auto">
            {feedRows.map((row) => (
              <div key={`${row.kind}:${row.id}`} className="rounded border p-2">
                <div className="flex items-center justify-between gap-2">
                  <Badge variant="outline">{row.kind}</Badge>
                  <span className="text-xs text-muted-foreground">{row.ts ? new Date(row.ts).toLocaleString() : 'n/a'}</span>
                </div>
                <div className="text-sm font-medium mt-1">{row.title}</div>
                <div className="text-xs text-muted-foreground mt-1">{row.detail}</div>
              </div>
            ))}
            {feedRows.length === 0 ? <div className="text-sm text-muted-foreground">No feed rows yet.</div> : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Trader Drawer</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {!selectedTrader ? <div className="text-sm text-muted-foreground">Select a trader.</div> : (
              <>
                <div>
                  <Label>Name</Label>
                  <Input value={draftName} onChange={(e) => setDraftName(e.target.value)} />
                </div>
                <div>
                  <Label>Interval Seconds</Label>
                  <Input type="number" value={draftInterval} onChange={(e) => setDraftInterval(e.target.value)} />
                </div>
                <div>
                  <Label>Sources (comma separated)</Label>
                  <Input value={draftSources} onChange={(e) => setDraftSources(e.target.value)} />
                </div>
                <div>
                  <Label>Strategy Params (JSON)</Label>
                  <textarea className="w-full min-h-[110px] rounded-md border bg-background p-2 text-xs" value={draftParams} onChange={(e) => setDraftParams(e.target.value)} />
                </div>
                <div>
                  <Label>Risk Limits (JSON)</Label>
                  <textarea className="w-full min-h-[110px] rounded-md border bg-background p-2 text-xs" value={draftRisk} onChange={(e) => setDraftRisk(e.target.value)} />
                </div>
                {saveError ? <div className="text-xs text-red-500">{saveError}</div> : null}
                <Button onClick={() => selectedTrader && saveTraderMutation.mutate(selectedTrader)} disabled={saveTraderMutation.isPending}>
                  {saveTraderMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                  Save Trader
                </Button>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
