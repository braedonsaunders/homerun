import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
  CheckCircle2,
  FileSearch,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Square,
  XCircle,
} from 'lucide-react'

import {
  armAutoTraderLiveStart,
  emergencyStopAutoTrader,
  getAccountPositions,
  getStrategies,
  getSimulationAccount,
  getAutoTraderDecisionDetail,
  getAutoTraderDecisions,
  getAutoTraderEvents,
  getAutoTraderOverview,
  getAutoTraderTrades,
  getCopyTradingStatus,
  getSignalStats,
  getTradingBalance,
  getTradingPositions,
  runAutoTraderLivePreflight,
  startAutoTrader,
  startAutoTraderLive,
  stopAutoTrader,
  stopAutoTraderLive,
  updateAutoTraderConfig,
  updateAutoTraderPolicies,
  type AutoTraderDecision,
  type AutoTraderEvent,
  type AutoTraderOverview,
  type AutoTraderStatus,
  type SimulationPosition,
  type Strategy,
  type TradingPosition,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { selectedAccountIdAtom } from '../store/atoms'
import { cn } from '../lib/utils'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import CommandOutputPanel, { type CommandLogLine } from './autotrader/CommandOutputPanel'
import CommandCenterHeader from './autotrader/CommandCenterHeader'
import CurrentHoldingsPanel from './autotrader/CurrentHoldingsPanel'
import RiskBudgetPanel from './autotrader/RiskBudgetPanel'
import SourceMatrix from './autotrader/SourceMatrix'
import TradeHistoryPanel from './autotrader/TradeHistoryPanel'
import AutoTraderSettingsFlyout from './autotrader/AutoTraderSettingsFlyout'

type Section =
  | 'live-console'
  | 'control'
  | 'timeline'
  | 'decision-trace'
  | 'risk'
  | 'portfolio'
  | 'performance'
  | 'sources'

type TimelineFilter = 'all' | 'decision' | 'trade' | 'status' | 'operator' | 'config'

const FILTER_EVENT_TYPES: Record<TimelineFilter, string[] | undefined> = {
  all: undefined,
  decision: ['decision'],
  trade: ['trade'],
  status: ['started', 'stopped', 'paused', 'run_once_requested', 'live_started', 'live_stopped', 'control_updated'],
  operator: ['control_updated', 'policies_updated', 'config_updated', 'kill_switch'],
  config: ['config_updated', 'policies_updated'],
}

const DOMAIN_LABELS: Record<string, string> = {
  event_markets: 'Event Markets',
  crypto: 'Crypto 15m',
}

const DOMAIN_OPTIONS = [
  {
    key: 'event_markets',
    title: 'Event Markets',
    description: 'News, weather, world intelligence, and event signal sources.',
  },
  {
    key: 'crypto',
    title: 'Crypto 15m',
    description: 'Short-interval crypto strategies and market microstructure entries.',
  },
] as const

interface StrategyLane {
  id: string
  label: string
  domain: string
  timeframe: string
  validationStatus: string
}

function toStatusView(overview?: AutoTraderOverview): AutoTraderStatus | undefined {
  if (!overview) return undefined
  const control = overview.control
  const worker = overview.worker
  const performance = overview.performance
  const running = Boolean(control.is_enabled) && !Boolean(control.is_paused) && !Boolean(control.kill_switch)
  const totalTrades = Object.values(performance.trade_counts || {}).reduce((sum, count) => sum + Number(count || 0), 0)

  return {
    mode: control.mode || 'paper',
    running,
    trading_active: running,
    worker_running: Boolean(worker.running),
    control: {
      is_enabled: Boolean(control.is_enabled),
      is_paused: Boolean(control.is_paused),
      kill_switch: Boolean(control.kill_switch),
      requested_run_at: control.requested_run_at || null,
      run_interval_seconds: Number(control.run_interval_seconds || 2),
      updated_at: control.updated_at || null,
    },
    snapshot: {
      running: Boolean(worker.running),
      enabled: Boolean(worker.enabled),
      current_activity: worker.current_activity || null,
      interval_seconds: Number(worker.interval_seconds || 2),
      last_run_at: worker.last_run_at || null,
      last_error: worker.last_error || null,
      updated_at: worker.updated_at || null,
      signals_seen: Number(worker.signals_seen || 0),
      signals_selected: Number(worker.signals_selected || 0),
      trades_count: Number(worker.trades_count || 0),
      daily_pnl: Number(worker.daily_pnl || 0),
    },
    config: overview.config,
    stats: {
      total_trades: totalTrades,
      winning_trades: Number(performance.winning_trades || 0),
      losing_trades: Number(performance.losing_trades || 0),
      win_rate: Number(performance.win_rate || 0),
      total_profit: Number((performance.total_pnl ?? performance.actual_profit_total) || 0),
      total_invested: Number(performance.notional_total || 0),
      roi_percent:
        Number(performance.notional_total || 0) > 0
          ? Number((Number((performance.total_pnl ?? performance.actual_profit_total) || 0) / Number(performance.notional_total || 0)) * 100)
          : 0,
      daily_trades: Number(worker.trades_count || 0),
      daily_profit: Number(worker.daily_pnl || 0),
      consecutive_losses: 0,
      circuit_breaker_active: Boolean(control.kill_switch),
      last_trade_at: worker.last_run_at || null,
      opportunities_seen: Number(worker.signals_seen || 0),
      opportunities_executed: Number(worker.signals_selected || 0),
      opportunities_skipped: Math.max(0, Number(worker.signals_seen || 0) - Number(worker.signals_selected || 0)),
    },
  }
}

function resolveDecisionId(event?: AutoTraderEvent | null): string | null {
  if (!event) return null
  const payload = event.payload || {}
  if (typeof payload.decision_id === 'string' && payload.decision_id.length > 0) return payload.decision_id
  return null
}

function eventTone(event: AutoTraderEvent): string {
  if (event.severity === 'error') return 'bg-rose-500/15 border-rose-500/25'
  if (event.severity === 'warn') return 'bg-amber-500/10 border-amber-500/25'
  if (event.event_type === 'decision') return 'bg-blue-500/10 border-blue-500/25'
  if (event.event_type === 'trade') return 'bg-emerald-500/10 border-emerald-500/25'
  return 'bg-background/40 border-border/50'
}

function mapSimulationPositionToTradingPosition(position: SimulationPosition): TradingPosition {
  return {
    token_id: position.token_id || position.id,
    market_id: position.market_id,
    market_slug: position.market_slug,
    event_slug: position.event_slug,
    market_question: position.market_question || position.market_id,
    outcome: String(position.side || '').toLowerCase() === 'no' ? 'NO' : 'YES',
    size: Number(position.quantity || 0),
    average_cost: Number(position.entry_price || 0),
    current_price: Number(position.current_price ?? position.entry_price ?? 0),
    unrealized_pnl: Number(position.unrealized_pnl || 0),
  }
}

function logFromSocketMessage(message: any): CommandLogLine | null {
  const type = String(message?.type || '')
  const data = message?.data || {}
  const rawTimestamp = data.updated_at || data.created_at || data.last_run_at || new Date().toISOString()
  const ts = new Date(rawTimestamp).toLocaleTimeString()

  if (type === 'autotrader_decision') {
    const decision = String(data.decision || 'decision')
    const level: CommandLogLine['level'] =
      decision === 'failed' ? 'error' : decision === 'skipped' ? 'warn' : 'event'
    return {
      id: `decision:${String(data.id || rawTimestamp)}:${Math.random().toString(16).slice(2, 8)}`,
      ts,
      type: 'decision',
      level,
      message: `${decision.toUpperCase()} • ${data.reason || 'No reason recorded'}`,
      source: data.source,
      status: decision,
      payload: data.payload || {},
      raw: data,
    }
  }

  if (type === 'autotrader_trade') {
    const status = String(data.status || 'unknown')
    const level: CommandLogLine['level'] =
      status === 'failed' ? 'error' : status === 'resolved_loss' ? 'warn' : 'info'
    return {
      id: `trade:${String(data.id || rawTimestamp)}:${Math.random().toString(16).slice(2, 8)}`,
      ts,
      type: 'trade',
      level,
      message: `${status.toUpperCase()} • ${data.market_question || data.market_id || 'market'} • $${Number(data.notional_usd || 0).toFixed(2)}`,
      source: data.source,
      status,
      payload: data.payload || {},
      raw: data,
    }
  }

  if (type === 'autotrader_status') {
    const statusText = data.running ? 'running' : 'idle'
    const cycleSeen = Number(data?.stats?.cycle_signals_seen ?? data?.stats?.signals_seen_cycle ?? 0)
    const cycleSelected = Number(data?.stats?.cycle_signals_selected ?? data?.stats?.signals_selected_cycle ?? 0)
    const totalSeen = Number(data.signals_seen || 0)
    const totalSelected = Number(data.signals_selected || 0)
    return {
      id: `status:${rawTimestamp}:${Math.random().toString(16).slice(2, 8)}`,
      ts,
      type: 'status',
      level: 'info',
      message: `Worker ${statusText} • cycle ${cycleSeen}/${cycleSelected} • total ${totalSeen}/${totalSelected}`,
      status: statusText,
      payload: data,
      raw: data,
    }
  }

  return null
}

function shouldSuppressSocketStatusLog(message: any): boolean {
  const type = String(message?.type || '')
  if (type !== 'autotrader_status') return false
  const data = message?.data || {}
  const enabled = Boolean(data.enabled)
  const hasError = Boolean(data.last_error)
  return !enabled && !hasError
}

function logFromTimelineEvent(event: AutoTraderEvent): CommandLogLine {
  return {
    id: `event:${event.id}`,
    ts: event.created_at ? new Date(event.created_at).toLocaleTimeString() : 'n/a',
    type: event.event_type,
    level: event.severity === 'error' ? 'error' : event.severity === 'warn' ? 'warn' : 'event',
    message: event.message || '(no message)',
    source: event.source || undefined,
    status: event.event_type,
    payload: event.payload || {},
    raw: {
      id: event.id,
      event_type: event.event_type,
      severity: event.severity,
      source: event.source,
      operator: event.operator,
      trace_id: event.trace_id,
      payload: event.payload || {},
      created_at: event.created_at,
    },
  }
}

interface TradingPanelProps {
  initialSection?: Section
  legacyTabLabel?: string
}

export default function TradingPanel({
  initialSection = 'live-console',
  legacyTabLabel,
}: TradingPanelProps) {
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const queryClient = useQueryClient()
  const { isConnected, lastMessage } = useWebSocket('/ws')

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [section, setSection] = useState<Section>(initialSection)
  const [timelineFilter, setTimelineFilter] = useState<TimelineFilter>('all')
  const [selectedEvent, setSelectedEvent] = useState<AutoTraderEvent | null>(null)
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const [updatingSources, setUpdatingSources] = useState<Set<string>>(new Set())
  const [commandLogs, setCommandLogs] = useState<CommandLogLine[]>([])

  useEffect(() => {
    setSection(initialSection)
  }, [initialSection])

  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['auto-trader-overview'],
    queryFn: getAutoTraderOverview,
    refetchInterval: isConnected ? 4000 : 6000,
  })

  const { data: eventsData, isLoading: eventsLoading } = useQuery({
    queryKey: ['auto-trader-events', timelineFilter],
    queryFn: () =>
      getAutoTraderEvents({
        limit: 250,
        types: FILTER_EVENT_TYPES[timelineFilter],
      }),
    refetchInterval: isConnected ? 4000 : 6000,
  })

  const { data: decisionsData } = useQuery({
    queryKey: ['auto-trader-decisions'],
    queryFn: () => getAutoTraderDecisions({ limit: 250 }),
    refetchInterval: isConnected ? 5000 : 7000,
  })

  const { data: trades = [] } = useQuery({
    queryKey: ['auto-trader-trades'],
    queryFn: () => getAutoTraderTrades(300),
    refetchInterval: isConnected ? 5000 : 7000,
  })

  const { data: signalStats } = useQuery({
    queryKey: ['signals-stats'],
    queryFn: getSignalStats,
    refetchInterval: 10000,
  })

  const { data: copyStatus } = useQuery({
    queryKey: ['copy-trading-status'],
    queryFn: getCopyTradingStatus,
    refetchInterval: 10000,
  })

  const { data: strategyCatalog = [] } = useQuery({
    queryKey: ['strategies', 'control'],
    queryFn: getStrategies,
    staleTime: 60000,
  })

  const mode = String(overview?.control.mode || 'paper')
  const isLiveMode = mode === 'live'
  const isPaperMode = mode === 'paper'
  const configuredPaperAccountId = String(overview?.config.paper_account_id || '').trim()
  const paperAccountId = configuredPaperAccountId || String(selectedAccountId || '').trim()
  const missingPaperAccount = isPaperMode && paperAccountId.length === 0

  const { data: liveHoldings = [] } = useQuery({
    queryKey: ['trading-positions'],
    queryFn: getTradingPositions,
    enabled: !isPaperMode,
    refetchInterval: isConnected ? 5000 : 8000,
  })

  const { data: simulationPositions = [] } = useQuery({
    queryKey: ['simulation-account-positions', paperAccountId],
    queryFn: () => getAccountPositions(paperAccountId),
    enabled: isPaperMode && paperAccountId.length > 0,
    refetchInterval: isConnected ? 5000 : 8000,
  })

  const { data: simulationAccount } = useQuery({
    queryKey: ['simulation-account', paperAccountId],
    queryFn: () => getSimulationAccount(paperAccountId),
    enabled: isPaperMode && paperAccountId.length > 0,
    refetchInterval: 10000,
  })

  const { data: liveTradingBalance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
    enabled: !isPaperMode,
    refetchInterval: 10000,
  })

  const currentHoldings = useMemo(
    () => (isPaperMode ? simulationPositions.map(mapSimulationPositionToTradingPosition) : liveHoldings),
    [isPaperMode, simulationPositions, liveHoldings]
  )
  const tradingBalance = useMemo(() => {
    if (!isPaperMode) return liveTradingBalance
    if (!simulationAccount) return undefined
    const reserved = simulationPositions.reduce((sum, pos) => sum + Number(pos.entry_cost || 0), 0)
    const available = Number(simulationAccount.current_capital || 0)
    return {
      balance: available + reserved,
      available,
      reserved,
      currency: 'USD',
      timestamp: new Date().toISOString(),
    }
  }, [isPaperMode, liveTradingBalance, simulationAccount, simulationPositions])

  const derivedStatus = useMemo(() => toStatusView(overview), [overview])
  const tradingActive = Boolean(derivedStatus?.trading_active)
  const canStart = isLiveMode || !isPaperMode || Boolean(selectedAccountId || configuredPaperAccountId)
  const activeTradingDomains = useMemo(() => {
    const configured = Array.isArray(overview?.config?.trading_domains)
      ? overview.config.trading_domains.map((domain) => String(domain).toLowerCase())
      : []
    const filtered = configured.filter((domain) => domain === 'event_markets' || domain === 'crypto')
    return filtered.length > 0 ? filtered : ['event_markets', 'crypto']
  }, [overview?.config?.trading_domains])
  const strategyLanes = useMemo(() => {
    const dedup = new Map<string, StrategyLane>()
    for (const strategy of strategyCatalog as Strategy[]) {
      const key =
        strategy.is_plugin && strategy.plugin_slug
          ? strategy.plugin_slug
          : strategy.type
      if (!key || dedup.has(key)) continue
      dedup.set(key, {
        id: key,
        label: strategy.name,
        domain: strategy.domain || 'event_markets',
        timeframe: strategy.timeframe || 'event',
        validationStatus: strategy.validation_status || 'unknown',
      })
    }

    const grouped = new Map<string, StrategyLane[]>()
    for (const lane of dedup.values()) {
      const groupKey = lane.domain || 'event_markets'
      if (!grouped.has(groupKey)) {
        grouped.set(groupKey, [])
      }
      grouped.get(groupKey)?.push(lane)
    }

    return ['event_markets', 'crypto'].map((domain) => ({
      domain,
      strategies: (grouped.get(domain) || []).sort((a, b) => a.label.localeCompare(b.label)),
    }))
  }, [strategyCatalog])
  const timelineRows = eventsData?.events || []
  const decisionRows = decisionsData?.decisions || []
  const selectedTimelineDecisionId = useMemo(() => resolveDecisionId(selectedEvent), [selectedEvent])
  const effectiveDecisionId = selectedTimelineDecisionId || selectedDecisionId

  useEffect(() => {
    if (!selectedEvent && timelineRows.length > 0) {
      setSelectedEvent(timelineRows[0])
      return
    }
    if (selectedEvent && !timelineRows.find((row) => row.id === selectedEvent.id) && timelineRows.length > 0) {
      setSelectedEvent(timelineRows[0])
    }
  }, [selectedEvent, timelineRows])

  useEffect(() => {
    if (!selectedDecisionId && decisionRows.length > 0) {
      setSelectedDecisionId(decisionRows[0].id)
    }
  }, [selectedDecisionId, decisionRows])

  const { data: decisionDetail, isLoading: decisionDetailLoading } = useQuery({
    queryKey: ['auto-trader-decision-detail', effectiveDecisionId],
    queryFn: () => getAutoTraderDecisionDetail(String(effectiveDecisionId)),
    enabled: Boolean(effectiveDecisionId),
  })

  const refreshAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['auto-trader-overview'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-events'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-decisions'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-trades'] })
    queryClient.invalidateQueries({ queryKey: ['signals-stats'] })
    queryClient.invalidateQueries({ queryKey: ['copy-trading-status'] })
    queryClient.invalidateQueries({ queryKey: ['trading-positions'] })
    queryClient.invalidateQueries({ queryKey: ['trading-balance'] })
    queryClient.invalidateQueries({ queryKey: ['simulation-account-positions'] })
    queryClient.invalidateQueries({ queryKey: ['simulation-account'] })
  }, [queryClient])

  useEffect(() => {
    if (!lastMessage?.type) return
    const suppressStatusLog = shouldSuppressSocketStatusLog(lastMessage)
    const streamRow = logFromSocketMessage(lastMessage)
    if (streamRow && !suppressStatusLog) {
      setCommandLogs((prev) => [...prev, streamRow].slice(-500))
    }
    if (
      !suppressStatusLog &&
      (
        lastMessage.type === 'autotrader_status' ||
        lastMessage.type === 'autotrader_decision' ||
        lastMessage.type === 'autotrader_trade'
      )
    ) {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-overview'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-events'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-decisions'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-trades'] })
      queryClient.invalidateQueries({ queryKey: ['trading-positions'] })
      queryClient.invalidateQueries({ queryKey: ['trading-balance'] })
      queryClient.invalidateQueries({ queryKey: ['simulation-account-positions'] })
      queryClient.invalidateQueries({ queryKey: ['simulation-account'] })
    }
  }, [lastMessage, queryClient])

  useEffect(() => {
    if (commandLogs.length > 0 || timelineRows.length === 0) return
    const seed = [...timelineRows].reverse().slice(-120).map(logFromTimelineEvent)
    setCommandLogs(seed)
  }, [commandLogs.length, timelineRows])

  const startMutation = useMutation({
    mutationFn: ({ mode, accountId }: { mode: string; accountId?: string }) => startAutoTrader(mode, accountId),
    onSuccess: refreshAll,
  })

  const stopMutation = useMutation({
    mutationFn: stopAutoTrader,
    onSuccess: refreshAll,
  })

  const emergencyStopMutation = useMutation({
    mutationFn: emergencyStopAutoTrader,
    onSuccess: refreshAll,
  })

  const policiesMutation = useMutation({
    mutationFn: updateAutoTraderPolicies,
    onSuccess: refreshAll,
  })

  const domainMutation = useMutation({
    mutationFn: (domains: string[]) =>
      updateAutoTraderConfig({
        trading_domains: domains,
        reason: 'control_tab_domain_toggle',
      }),
    onSuccess: refreshAll,
  })

  const preflightMutation = useMutation({
    mutationFn: () => runAutoTraderLivePreflight({ mode: 'live' }),
    onSuccess: refreshAll,
  })

  const armMutation = useMutation({
    mutationFn: (preflightId: string) => armAutoTraderLiveStart({ preflight_id: preflightId, ttl_seconds: 300 }),
    onSuccess: refreshAll,
  })

  const liveStartMutation = useMutation({
    mutationFn: (armToken: string) => startAutoTraderLive({ arm_token: armToken, mode: 'live' }),
    onSuccess: refreshAll,
  })

  const liveStopMutation = useMutation({
    mutationFn: () => stopAutoTraderLive(),
    onSuccess: refreshAll,
  })

  const latestPreflight = preflightMutation.data
  const latestArm = armMutation.data

  const handleStart = () => {
    if (isLiveMode) {
      setSection('control')
      return
    }
    startMutation.mutate({
      mode,
      accountId: isPaperMode ? selectedAccountId || configuredPaperAccountId || undefined : undefined,
    })
  }

  const handleStop = () => {
    stopMutation.mutate()
  }

  const handleEmergencyStop = () => {
    if (!confirm('Enable kill switch and halt auto-trader immediately?')) return
    emergencyStopMutation.mutate()
  }

  const handleToggleSource = (source: string, enabled: boolean) => {
    setUpdatingSources((prev) => {
      const next = new Set(prev)
      next.add(source)
      return next
    })

    policiesMutation.mutate(
      {
        sources: {
          [source]: { enabled },
        },
      },
      {
        onSettled: () => {
          setUpdatingSources((prev) => {
            const next = new Set(prev)
            next.delete(source)
            return next
          })
        },
      }
    )
  }

  const handleToggleDomain = (domain: 'event_markets' | 'crypto') => {
    if (domainMutation.isPending) return
    const hasDomain = activeTradingDomains.includes(domain)
    const next = hasDomain
      ? activeTradingDomains.filter((item) => item !== domain)
      : [...activeTradingDomains, domain]
    if (next.length === 0) return
    domainMutation.mutate(next)
  }

  if (overviewLoading) {
    return (
      <div className="flex items-center justify-center py-16 text-muted-foreground text-sm">
        Loading command center...
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-3 overflow-hidden">
      {legacyTabLabel && (
        <Card className="border-blue-500/25 bg-blue-500/5 shrink-0">
          <CardContent className="p-3 text-xs text-blue-200">
            {legacyTabLabel} is now served by the unified AutoTrader Command Center for one-release compatibility.
          </CardContent>
        </Card>
      )}

      {!canStart && (
        <Card className="border-amber-500/30 bg-amber-500/5 shrink-0">
          <CardContent className="p-3 text-xs text-amber-200">
            Select a sandbox account before starting AutoTrader in paper mode.
          </CardContent>
        </Card>
      )}

      {missingPaperAccount && (
        <Card className="border-rose-500/35 bg-rose-500/5 shrink-0">
          <CardContent className="p-3 text-xs text-rose-200">
            Paper mode is active but no paper account is configured. Set an account before relying on holdings/P&L.
          </CardContent>
        </Card>
      )}

      <div className="shrink-0">
        <CommandCenterHeader
          status={derivedStatus}
          canStart={canStart}
          startPending={startMutation.isPending}
          stopPending={stopMutation.isPending}
          emergencyPending={emergencyStopMutation.isPending}
          onOpenSettings={() => setSettingsOpen(true)}
          onStart={handleStart}
          onStop={handleStop}
          onEmergencyStop={handleEmergencyStop}
        />
      </div>

      <Tabs
        value={section}
        onValueChange={(value) => setSection(value as Section)}
        className="h-full min-h-0 w-full flex flex-col"
      >
        <TabsList className="mb-2 h-auto w-full shrink-0 grid grid-cols-8 gap-1">
          <TabsTrigger value="live-console" className="text-xs">Live Console</TabsTrigger>
          <TabsTrigger value="control" className="text-xs">Control</TabsTrigger>
          <TabsTrigger value="timeline" className="text-xs">Timeline</TabsTrigger>
          <TabsTrigger value="decision-trace" className="text-xs">Decision Trace</TabsTrigger>
          <TabsTrigger value="risk" className="text-xs">Risk</TabsTrigger>
          <TabsTrigger value="portfolio" className="text-xs">Portfolio</TabsTrigger>
          <TabsTrigger value="performance" className="text-xs">Performance</TabsTrigger>
          <TabsTrigger value="sources" className="text-xs">Sources</TabsTrigger>
        </TabsList>

        <TabsContent value="live-console" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <div className="flex-1 min-h-0">
            <CommandOutputPanel logs={commandLogs} connected={isConnected} />
          </div>
        </TabsContent>

        <TabsContent value="control" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 flex-1 min-h-0">
            <Card className="xl:col-span-12 border-border/50 bg-card/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Domain Modes and Strategy Lanes</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {DOMAIN_OPTIONS.map((option) => {
                    const enabled = activeTradingDomains.includes(option.key)
                    return (
                      <button
                        key={option.key}
                        type="button"
                        disabled={domainMutation.isPending}
                        onClick={() => handleToggleDomain(option.key)}
                        className={cn(
                          'rounded-md border p-3 text-left transition-colors',
                          enabled
                            ? 'border-emerald-500/40 bg-emerald-500/10'
                            : 'border-border/50 bg-background/40 hover:border-border'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-semibold">{option.title}</p>
                          <Badge className={enabled ? 'bg-emerald-500/20 text-emerald-300' : 'bg-background/70 text-muted-foreground'}>
                            {enabled ? 'active' : 'off'}
                          </Badge>
                        </div>
                        <p className="mt-1 text-[11px] text-muted-foreground">{option.description}</p>
                      </button>
                    )
                  })}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {strategyLanes.map((lane) => {
                    const domainEnabled = activeTradingDomains.includes(lane.domain)
                    return (
                      <div
                        key={lane.domain}
                        className={cn(
                          'rounded-md border border-border/50 bg-background/40 p-3',
                          !domainEnabled && 'opacity-70'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-semibold">{DOMAIN_LABELS[lane.domain] || lane.domain}</p>
                          <Badge className="bg-background/70 text-muted-foreground text-[10px]">
                            {lane.strategies.length} strategies
                          </Badge>
                        </div>
                        {lane.strategies.length === 0 ? (
                          <p className="mt-1 text-[11px] text-muted-foreground">No strategies discovered for this domain.</p>
                        ) : (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {lane.strategies.slice(0, 7).map((strategy) => (
                              <Badge
                                key={strategy.id}
                                className={cn(
                                  'text-[10px] bg-background/70',
                                  strategy.validationStatus === 'validated'
                                    ? 'text-emerald-300'
                                    : strategy.validationStatus === 'backtest_only'
                                    ? 'text-amber-300'
                                    : 'text-muted-foreground'
                                )}
                              >
                                {strategy.label} | {strategy.timeframe}
                              </Badge>
                            ))}
                            {lane.strategies.length > 7 && (
                              <Badge className="text-[10px] bg-background/70 text-muted-foreground">
                                +{lane.strategies.length - 7} more
                              </Badge>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>

            <Card className="xl:col-span-7 border-border/50 bg-card/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-300" />
                  {isLiveMode ? 'Strict Live Start Wizard' : `Execution Controls (${mode})`}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {isLiveMode ? (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                      <Button
                        onClick={() => preflightMutation.mutate()}
                        disabled={preflightMutation.isPending}
                        className="h-8 text-xs"
                      >
                        {preflightMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 mr-1 animate-spin" /> : <FileSearch className="w-3.5 h-3.5 mr-1" />}
                        1. Run Preflight
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => latestPreflight?.preflight_id && armMutation.mutate(latestPreflight.preflight_id)}
                        disabled={armMutation.isPending || !latestPreflight?.preflight_id || latestPreflight?.status !== 'passed'}
                        className="h-8 text-xs"
                      >
                        {armMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 mr-1 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5 mr-1" />}
                        2. Arm
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => latestArm?.arm_token && liveStartMutation.mutate(latestArm.arm_token)}
                        disabled={liveStartMutation.isPending || !latestArm?.arm_token}
                        className="h-8 text-xs border-emerald-500/40 text-emerald-300"
                      >
                        {liveStartMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 mr-1 animate-spin" /> : <Play className="w-3.5 h-3.5 mr-1" />}
                        3. Start Live
                      </Button>
                    </div>

                    {latestPreflight && (
                      <div className="rounded-lg border border-border/50 bg-background/40 p-3 space-y-2">
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-semibold">Latest Preflight: {latestPreflight.preflight_id}</p>
                          <Badge className={cn(
                            'text-[10px] uppercase',
                            latestPreflight.status === 'passed'
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : 'bg-rose-500/20 text-rose-300'
                          )}>
                            {latestPreflight.status}
                          </Badge>
                        </div>
                        <div className="space-y-1.5 text-xs">
                          {latestPreflight.checks.map((check, index) => (
                            <div key={`${check.id || 'check'}-${index}`} className="flex items-start gap-2">
                              {check.ok ? (
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-300 mt-0.5" />
                              ) : (
                                <XCircle className="w-3.5 h-3.5 text-rose-300 mt-0.5" />
                              )}
                              <div>
                                <p className={cn('font-medium', check.ok ? 'text-emerald-200' : 'text-rose-200')}>
                                  {check.id}
                                </p>
                                <p className="text-muted-foreground">{check.message}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {latestArm && (
                      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs">
                        <p className="font-semibold text-emerald-300">Arm token issued</p>
                        <p className="text-muted-foreground">
                          Token expires at {new Date(latestArm.expires_at).toLocaleString()}.
                        </p>
                      </div>
                    )}

                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        onClick={() => liveStopMutation.mutate()}
                        disabled={liveStopMutation.isPending}
                        className="h-8 text-xs"
                      >
                        <Square className="w-3.5 h-3.5 mr-1" />
                        Stop Live
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleEmergencyStop}
                        disabled={emergencyStopMutation.isPending}
                        className="h-8 text-xs border-rose-500/40 text-rose-300"
                      >
                        <ShieldAlert className="w-3.5 h-3.5 mr-1" />
                        Kill Switch
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="space-y-3">
                    <div className="rounded-md border border-border/50 bg-background/40 p-3 text-xs text-muted-foreground">
                      Live preflight/arm is only required in live mode. Use settings to switch modes.
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {tradingActive ? (
                        <Button
                          variant="outline"
                          onClick={handleStop}
                          disabled={stopMutation.isPending}
                          className="h-8 text-xs"
                        >
                          <Square className="w-3.5 h-3.5 mr-1" />
                          Stop {mode}
                        </Button>
                      ) : (
                        <Button
                          onClick={handleStart}
                          disabled={!canStart || startMutation.isPending}
                          className="h-8 text-xs bg-emerald-600 hover:bg-emerald-500"
                        >
                          {startMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 mr-1 animate-spin" /> : <Play className="w-3.5 h-3.5 mr-1" />}
                          Start {mode}
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        onClick={handleEmergencyStop}
                        disabled={emergencyStopMutation.isPending}
                        className="h-8 text-xs border-rose-500/40 text-rose-300"
                      >
                        <ShieldAlert className="w-3.5 h-3.5 mr-1" />
                        Kill Switch
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="xl:col-span-5 border-border/50 bg-card/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">System Health</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {(overview?.health.checks || []).map((check) => (
                  <div key={check.id} className="rounded-md border border-border/50 bg-background/40 p-2">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-semibold">{check.id}</p>
                      <Badge className={check.ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}>
                        {check.ok ? 'pass' : 'fail'}
                      </Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground mt-1">{check.message}</p>
                  </div>
                ))}

                <div className="rounded-md border border-border/50 bg-background/40 p-2 text-xs space-y-1">
                  <p><span className="text-muted-foreground">Worker:</span> {overview?.worker.current_activity || 'idle'}</p>
                  <p><span className="text-muted-foreground">Last run:</span> {overview?.worker.last_run_at ? new Date(overview.worker.last_run_at).toLocaleString() : 'n/a'}</p>
                  <p><span className="text-muted-foreground">Last error:</span> {overview?.worker.last_error || 'none'}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="timeline" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 flex-1 min-h-0">
            <Card className="xl:col-span-8 border-border/50 bg-card/40 min-h-0 flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-sm">Execution Timeline</CardTitle>
                  <Button variant="outline" size="sm" onClick={refreshAll} className="h-7 text-xs">
                    <RefreshCw className="w-3.5 h-3.5 mr-1" />
                    Refresh
                  </Button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(['all', 'decision', 'trade', 'status', 'operator', 'config'] as TimelineFilter[]).map((filterKey) => (
                    <Button
                      key={filterKey}
                      variant={timelineFilter === filterKey ? 'default' : 'outline'}
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => setTimelineFilter(filterKey)}
                    >
                      {filterKey}
                    </Button>
                  ))}
                </div>
              </CardHeader>
              <CardContent className="flex-1 min-h-0">
                {eventsLoading ? (
                  <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                    Loading events...
                  </div>
                ) : timelineRows.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                    No timeline events yet.
                  </div>
                ) : (
                  <div className="h-full space-y-1.5 overflow-y-auto pr-1">
                    {timelineRows.map((event) => (
                      <button
                        key={event.id}
                        type="button"
                        onClick={() => setSelectedEvent(event)}
                        className={cn(
                          'w-full text-left rounded-lg border p-2 transition-colors',
                          eventTone(event),
                          selectedEvent?.id === event.id && 'ring-1 ring-blue-500/40'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="min-w-0 flex items-center gap-2">
                            <Badge className="text-[10px] uppercase bg-background/70 text-muted-foreground">
                              {event.event_type}
                            </Badge>
                            <p className="text-xs font-medium truncate">{event.message || '(no message)'}</p>
                          </div>
                          <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                            {event.created_at ? new Date(event.created_at).toLocaleTimeString() : 'n/a'}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
                          <span>{event.source || 'unknown source'}</span>
                          {event.operator && <span>• {event.operator}</span>}
                          {event.trace_id && <span className="font-mono">• {event.trace_id.slice(0, 8)}</span>}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="xl:col-span-4 border-border/50 bg-card/40 min-h-0 flex flex-col">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Why This Happened</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 overflow-y-auto pr-1">
                {!selectedEvent ? (
                  <p className="text-sm text-muted-foreground">Select an event to inspect trace details.</p>
                ) : (
                  <div className="space-y-3 text-xs">
                    <div className="rounded-md border border-border/50 bg-background/40 p-2">
                      <p className="text-[10px] uppercase text-muted-foreground">Event</p>
                      <p className="font-semibold mt-1">{selectedEvent.event_type}</p>
                      <p className="text-muted-foreground mt-1">{selectedEvent.message || 'No message provided'}</p>
                    </div>

                    {selectedTimelineDecisionId ? (
                      decisionDetailLoading ? (
                        <p className="text-muted-foreground">Loading decision trace...</p>
                      ) : decisionDetail ? (
                        <>
                          <div className="rounded-md border border-border/50 bg-background/40 p-2">
                            <p className="text-[10px] uppercase text-muted-foreground">Decision</p>
                            <p className="font-semibold mt-1">{decisionDetail.decision.decision}</p>
                            <p className="text-muted-foreground mt-1">{decisionDetail.decision.reason || 'No reason'}</p>
                            <p className="mt-1 font-mono">Score: {decisionDetail.decision.score ?? 'n/a'}</p>
                          </div>
                          <div className="space-y-1.5">
                            {decisionDetail.checks.map((check) => (
                              <div key={check.id} className="rounded-md border border-border/50 bg-background/40 p-2">
                                <div className="flex items-center justify-between gap-2">
                                  <p className="font-semibold">{check.check_label}</p>
                                  {check.passed ? (
                                    <Badge className="bg-emerald-500/20 text-emerald-300">pass</Badge>
                                  ) : (
                                    <Badge className="bg-rose-500/20 text-rose-300">fail</Badge>
                                  )}
                                </div>
                                <p className="text-muted-foreground mt-1">{check.detail || 'No detail recorded.'}</p>
                              </div>
                            ))}
                          </div>
                        </>
                      ) : (
                        <p className="text-muted-foreground">Decision detail not found.</p>
                      )
                    ) : (
                      <div className="rounded-md border border-border/50 bg-background/40 p-2">
                        <p className="text-[10px] uppercase text-muted-foreground">Payload</p>
                        <pre className="mt-1 text-[10px] whitespace-pre-wrap break-words text-muted-foreground">
                          {JSON.stringify(selectedEvent.payload || {}, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="decision-trace" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 flex-1 min-h-0">
            <Card className="xl:col-span-5 border-border/50 bg-card/40 min-h-0 flex flex-col">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Decision Index</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-1.5">
                {decisionRows.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No decisions yet.</p>
                ) : (
                  decisionRows.map((decision: AutoTraderDecision) => (
                    <button
                      key={decision.id}
                      type="button"
                      onClick={() => setSelectedDecisionId(decision.id)}
                      className={cn(
                        'w-full text-left rounded-md border border-border/50 bg-background/40 p-2',
                        selectedDecisionId === decision.id && 'ring-1 ring-blue-500/40'
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs font-semibold">{decision.source}</p>
                        <Badge className="text-[10px] uppercase bg-background/70 text-muted-foreground">{decision.decision}</Badge>
                      </div>
                      <p className="text-[11px] text-muted-foreground mt-1 truncate">{decision.reason || 'No reason'}</p>
                    </button>
                  ))
                )}
              </CardContent>
            </Card>

            <Card className="xl:col-span-7 border-border/50 bg-card/40 min-h-0 flex flex-col">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Decision Detail</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 overflow-y-auto pr-1">
                {!effectiveDecisionId ? (
                  <p className="text-sm text-muted-foreground">Select a decision to inspect checks and risk snapshots.</p>
                ) : decisionDetailLoading ? (
                  <p className="text-sm text-muted-foreground">Loading decision detail...</p>
                ) : !decisionDetail ? (
                  <p className="text-sm text-muted-foreground">Decision detail unavailable.</p>
                ) : (
                  <div className="space-y-3">
                    <div className="rounded-md border border-border/50 bg-background/40 p-3 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <p className="font-semibold">{decisionDetail.decision.decision}</p>
                        <span className="font-mono">score {decisionDetail.decision.score ?? 'n/a'}</span>
                      </div>
                      <p className="text-muted-foreground mt-1">{decisionDetail.decision.reason || 'No reason.'}</p>
                    </div>
                    {decisionDetail.checks.map((check) => (
                      <div key={check.id} className="rounded-md border border-border/50 bg-background/40 p-3 text-xs">
                        <div className="flex items-center justify-between gap-2">
                          <p className="font-semibold">{check.check_label}</p>
                          {check.passed ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-300" />
                          ) : (
                            <XCircle className="w-3.5 h-3.5 text-rose-300" />
                          )}
                        </div>
                        <p className="text-muted-foreground mt-1">{check.detail || 'No detail recorded.'}</p>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="risk" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <RiskBudgetPanel status={derivedStatus} exposure={overview?.risk} />
        </TabsContent>

        <TabsContent value="portfolio" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 flex-1 min-h-0">
            <div className="xl:col-span-5 min-h-0">
              <CurrentHoldingsPanel positions={currentHoldings} balance={tradingBalance} />
            </div>
            <div className="xl:col-span-7 min-h-0">
              <TradeHistoryPanel trades={trades} maxItems={140} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 flex-1 min-h-0">
            <Card className="xl:col-span-7 border-border/50 bg-card/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">P&L and Win Rate</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  <PerfStat label="Realized Trades" value={overview?.performance.realized_trade_count || 0} />
                  <PerfStat label="Win Rate" value={`${((overview?.performance.win_rate || 0) * 100).toFixed(1)}%`} />
                  <PerfStat
                    label="Realized P&L"
                    value={`$${Number((overview?.performance.realized_pnl_total ?? overview?.performance.actual_profit_total) || 0).toFixed(2)}`}
                    good={Number((overview?.performance.realized_pnl_total ?? overview?.performance.actual_profit_total) || 0) >= 0}
                  />
                  <PerfStat
                    label="Unrealized P&L"
                    value={`$${Number(overview?.performance.unrealized_pnl_total || 0).toFixed(2)}`}
                    good={Number(overview?.performance.unrealized_pnl_total || 0) >= 0}
                  />
                  <PerfStat
                    label="Total P&L"
                    value={`$${Number((overview?.performance.total_pnl ?? overview?.performance.actual_profit_total) || 0).toFixed(2)}`}
                    good={Number((overview?.performance.total_pnl ?? overview?.performance.actual_profit_total) || 0) >= 0}
                  />
                  <PerfStat label="Notional" value={`$${Number(overview?.performance.notional_total || 0).toFixed(2)}`} />
                </div>
                <div className="space-y-1.5">
                  {Object.entries(overview?.performance.trade_counts || {}).map(([status, count]) => (
                    <div key={status} className="flex items-center justify-between text-xs rounded-md border border-border/50 bg-background/40 p-2">
                      <span className="text-muted-foreground uppercase">{status}</span>
                      <span className="font-mono font-semibold">{count}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="xl:col-span-5 border-border/50 bg-card/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Decision Funnel</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1.5 text-xs">
                {Object.entries(overview?.metrics.decision_funnel || {}).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 p-2">
                    <span className="text-muted-foreground uppercase">{key}</span>
                    <span className="font-mono font-semibold">{value}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="sources" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1 overflow-hidden">
          <div className="w-full flex-1 min-h-0">
            <SourceMatrix
              policies={overview?.policies}
              signalStats={signalStats}
              metrics={overview?.metrics}
              exposure={overview?.risk}
              copyStatus={copyStatus}
              updatingSources={updatingSources}
              onToggleSource={handleToggleSource}
            />
          </div>
        </TabsContent>
      </Tabs>

      <AutoTraderSettingsFlyout isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  )
}

function PerfStat({
  label,
  value,
  good,
}: {
  label: string
  value: number | string
  good?: boolean
}) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/40 p-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className={cn('text-sm font-mono font-semibold', good === undefined ? '' : good ? 'text-emerald-300' : 'text-rose-300')}>
        {value}
      </p>
    </div>
  )
}
