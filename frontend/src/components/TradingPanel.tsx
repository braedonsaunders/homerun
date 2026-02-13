import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Loader2,
  Play,
  ShieldAlert,
  Sparkles,
  Square,
  Zap,
} from 'lucide-react'
import {
  armTraderOrchestratorLiveStart,
  createTraderFromTemplate,
  getAllTraderOrders,
  getTraderDecisionDetail,
  getTraderDecisions,
  getTraderEvents,
  getTraderOrchestratorOverview,
  getSimulationAccounts,
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
  type TraderOrder,
  updateTrader,
} from '../services/api'
import { cn } from '../lib/utils'
import { selectedAccountIdAtom } from '../store/atoms'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Switch } from './ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

type FeedFilter = 'all' | 'decision' | 'order' | 'event'
type ScopeFilter = 'all' | 'selected'
type TradeStatusFilter = 'all' | 'open' | 'resolved' | 'failed'

type ActivityRow = {
  kind: 'decision' | 'order' | 'event'
  id: string
  ts: string | null
  traderId: string | null
  title: string
  detail: string
  tone: 'neutral' | 'positive' | 'negative' | 'warning'
}

type PositionBookRow = {
  key: string
  traderId: string
  traderName: string
  marketId: string
  marketQuestion: string
  direction: string
  exposureUsd: number
  averagePrice: number | null
  weightedEdge: number | null
  weightedConfidence: number | null
  orderCount: number
  lastUpdated: string | null
  statusSummary: string
}

const OPEN_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])
const RESOLVED_ORDER_STATUSES = new Set(['resolved', 'resolved_win', 'resolved_loss', 'win', 'loss'])
const FAILED_ORDER_STATUSES = new Set(['failed', 'rejected', 'error', 'cancelled'])

function toNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function normalizeStatus(value: string | null | undefined): string {
  return String(value || 'unknown').trim().toLowerCase()
}

function toTs(value: string | null | undefined): number {
  if (!value) return 0
  const ts = new Date(value).getTime()
  return Number.isFinite(ts) ? ts : 0
}

function formatCurrency(value: number, compact = false): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 1 : 2,
  }).format(value)
}

function formatPercent(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`
}

function normalizeConfidencePercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (Math.abs(value) <= 1) return value * 100
  return value
}

function normalizeEdgePercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (Math.abs(value) <= 1) return value * 100
  if (Math.abs(value) > 200) return value / 100
  return value
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return 'n/a'
  const ts = new Date(value)
  if (Number.isNaN(ts.getTime())) return 'n/a'
  return ts.toLocaleString()
}

function formatShortDate(value: string | null | undefined): string {
  if (!value) return 'n/a'
  const ts = new Date(value)
  if (Number.isNaN(ts.getTime())) return 'n/a'
  return ts.toLocaleDateString()
}

function shortId(value: string | null | undefined): string {
  if (!value) return 'n/a'
  return value.length <= 12 ? value : `${value.slice(0, 6)}...${value.slice(-4)}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseJsonObject(text: string): { value: Record<string, unknown> | null; error: string | null } {
  try {
    const parsed: unknown = JSON.parse(text || '{}')
    if (!isRecord(parsed)) {
      return {
        value: null,
        error: 'Must be a JSON object.',
      }
    }
    return { value: parsed, error: null }
  } catch (error) {
    return { value: null, error: error instanceof Error ? error.message : 'Invalid JSON' }
  }
}

function humanizeKey(raw: string): string {
  return raw
    .replace(/_/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim()
}

function buildPositionBookRows(orders: TraderOrder[], traderNameById: Record<string, string>): PositionBookRow[] {
  const buckets = new Map<string, {
    traderId: string
    traderName: string
    marketId: string
    marketQuestion: string
    direction: string
    exposureUsd: number
    weightedPrice: number
    weightedEdge: number
    edgeWeight: number
    weightedConfidence: number
    confidenceWeight: number
    orderCount: number
    lastUpdated: string | null
    statuses: Set<string>
  }>()

  for (const order of orders) {
    const status = normalizeStatus(order.status)
    if (!OPEN_ORDER_STATUSES.has(status)) continue

    const traderId = String(order.trader_id || 'unknown')
    const marketId = String(order.market_id || 'unknown')
    const direction = String(order.direction || 'flat').toUpperCase()
    const key = `${traderId}:${marketId}:${direction}`
    const notional = Math.abs(toNumber(order.notional_usd))
    const px = toNumber(order.effective_price ?? order.entry_price)
    const edge = toNumber(order.edge_percent)
    const confidence = toNumber(order.confidence)
    const traderName = traderNameById[traderId] || shortId(traderId)

    if (!buckets.has(key)) {
      buckets.set(key, {
        traderId,
        traderName,
        marketId,
        marketQuestion: String(order.market_question || order.market_id || 'Unknown market'),
        direction,
        exposureUsd: 0,
        weightedPrice: 0,
        weightedEdge: 0,
        edgeWeight: 0,
        weightedConfidence: 0,
        confidenceWeight: 0,
        orderCount: 0,
        lastUpdated: null,
        statuses: new Set<string>(),
      })
    }

    const bucket = buckets.get(key)
    if (!bucket) continue

    bucket.exposureUsd += notional
    bucket.weightedPrice += px > 0 && notional > 0 ? px * notional : 0
    bucket.weightedEdge += edge !== 0 && notional > 0 ? edge * notional : 0
    bucket.edgeWeight += edge !== 0 && notional > 0 ? notional : 0
    bucket.weightedConfidence += confidence !== 0 && notional > 0 ? confidence * notional : 0
    bucket.confidenceWeight += confidence !== 0 && notional > 0 ? notional : 0
    bucket.orderCount += 1
    bucket.lastUpdated = toTs(order.updated_at) > toTs(bucket.lastUpdated)
      ? (order.updated_at || order.executed_at || order.created_at || null)
      : bucket.lastUpdated
    bucket.statuses.add(status)
  }

  return Array.from(buckets.entries())
    .map(([key, bucket]) => ({
      key,
      traderId: bucket.traderId,
      traderName: bucket.traderName,
      marketId: bucket.marketId,
      marketQuestion: bucket.marketQuestion,
      direction: bucket.direction,
      exposureUsd: bucket.exposureUsd,
      averagePrice: bucket.exposureUsd > 0 ? bucket.weightedPrice / bucket.exposureUsd : null,
      weightedEdge: bucket.edgeWeight > 0 ? bucket.weightedEdge / bucket.edgeWeight : null,
      weightedConfidence: bucket.confidenceWeight > 0 ? bucket.weightedConfidence / bucket.confidenceWeight : null,
      orderCount: bucket.orderCount,
      lastUpdated: bucket.lastUpdated,
      statusSummary: Array.from(bucket.statuses).join(', '),
    }))
    .sort((a, b) => b.exposureUsd - a.exposureUsd)
}

export default function TradingPanel() {
  const queryClient = useQueryClient()
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const [feedFilter, setFeedFilter] = useState<FeedFilter>('all')
  const [traderFeedFilter, setTraderFeedFilter] = useState<FeedFilter>('all')
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('all')
  const [tradeStatusFilter, setTradeStatusFilter] = useState<TradeStatusFilter>('all')
  const [tradeSearch, setTradeSearch] = useState('')
  const [decisionSearch, setDecisionSearch] = useState('')

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

  const simulationAccountsQuery = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    staleTime: 30000,
  })

  const traders = tradersQuery.data || []

  const traderIds = useMemo(() => traders.map((trader) => trader.id), [traders])
  const traderIdsKey = useMemo(() => traderIds.join('|'), [traderIds])

  const allOrdersQuery = useQuery({
    queryKey: ['trader-orders-all'],
    queryFn: () => getAllTraderOrders(220),
    enabled: traderIds.length > 0,
    refetchInterval: 5000,
  })

  const allDecisionsQuery = useQuery({
    queryKey: ['trader-decisions-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: 5000,
    queryFn: async () => {
      const grouped = await Promise.all(
        traderIds.map((traderId) => getTraderDecisions(traderId, { limit: 160 }))
      )
      return grouped
        .flat()
        .sort((a, b) => toTs(b.created_at) - toTs(a.created_at))
    },
  })

  const allEventsQuery = useQuery({
    queryKey: ['trader-events-all', traderIdsKey],
    enabled: traderIds.length > 0,
    refetchInterval: 5000,
    queryFn: async () => {
      const grouped = await Promise.all(
        traderIds.map((traderId) => getTraderEvents(traderId, { limit: 80 }))
      )
      return grouped
        .flatMap((group) => group.events)
        .sort((a, b) => toTs(b.created_at) - toTs(a.created_at))
    },
  })

  const allOrders = allOrdersQuery.data || []
  const allDecisions = allDecisionsQuery.data || []
  const allEvents = allEventsQuery.data || []

  const selectedTrader = useMemo(
    () => traders.find((trader) => trader.id === selectedTraderId) || null,
    [traders, selectedTraderId]
  )

  const selectedOrders = useMemo(
    () => allOrders.filter((order) => order.trader_id === selectedTraderId),
    [allOrders, selectedTraderId]
  )

  const selectedDecisions = useMemo(
    () => allDecisions.filter((decision) => decision.trader_id === selectedTraderId),
    [allDecisions, selectedTraderId]
  )

  const selectedEvents = useMemo(
    () => allEvents.filter((event) => event.trader_id === selectedTraderId),
    [allEvents, selectedTraderId]
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

  useEffect(() => {
    if (selectedDecisions.length === 0) {
      setSelectedDecisionId(null)
      return
    }

    setSelectedDecisionId((current) => {
      if (current && selectedDecisions.some((decision) => decision.id === current)) {
        return current
      }
      return selectedDecisions[0].id
    })
  }, [selectedDecisions])

  const decisionDetailQuery = useQuery({
    queryKey: ['trader-decision-detail', selectedDecisionId],
    queryFn: () => getTraderDecisionDetail(String(selectedDecisionId)),
    enabled: Boolean(selectedDecisionId),
    refetchInterval: 7000,
  })

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
    queryClient.invalidateQueries({ queryKey: ['traders-list'] })
    queryClient.invalidateQueries({ queryKey: ['trader-orders-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decisions-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    queryClient.invalidateQueries({ queryKey: ['trader-decision-detail'] })
  }

  const startBySelectedAccountMutation = useMutation({
    mutationFn: async () => {
      if (!selectedAccountId || !selectedAccountValid) {
        throw new Error('Select a valid global account in the top control bar.')
      }
      if (selectedAccountIsLive) {
        const preflight = await runTraderOrchestratorLivePreflight({ mode: 'live' })
        if (preflight.status !== 'passed') {
          throw new Error('Live preflight did not pass. Review checks before live launch.')
        }
        const armed = await armTraderOrchestratorLiveStart({ preflight_id: preflight.preflight_id })
        return startTraderOrchestratorLive({ arm_token: armed.arm_token, mode: 'live' })
      }
      if (!selectedSandboxAccount?.id) {
        throw new Error('No sandbox account is selected for paper mode.')
      }
      return startTraderOrchestrator({ mode: 'paper', paper_account_id: selectedSandboxAccount.id })
    },
    onSuccess: refreshAll,
  })

  const stopByModeMutation = useMutation({
    mutationFn: async () => {
      const mode = String(overviewQuery.data?.control?.mode || 'paper').toLowerCase()
      if (mode === 'live') {
        return stopTraderOrchestratorLive()
      }
      return stopTraderOrchestrator()
    },
    onSuccess: refreshAll,
  })

  const killSwitchMutation = useMutation({
    mutationFn: (enabled: boolean) => setTraderOrchestratorLiveKillSwitch(enabled),
    onSuccess: refreshAll,
  })

  const traderStartMutation = useMutation({
    mutationFn: (traderId: string) => startTrader(traderId),
    onSuccess: refreshAll,
  })

  const traderPauseMutation = useMutation({
    mutationFn: (traderId: string) => pauseTrader(traderId),
    onSuccess: refreshAll,
  })

  const traderRunOnceMutation = useMutation({
    mutationFn: (traderId: string) => runTraderOnce(traderId),
    onSuccess: refreshAll,
  })

  const saveTraderMutation = useMutation({
    mutationFn: async (trader: Trader) => {
      const parsedParams = parseJsonObject(draftParams || '{}')
      if (!parsedParams.value) {
        throw new Error(`Strategy params JSON error: ${parsedParams.error || 'invalid object'}`)
      }

      const parsedRisk = parseJsonObject(draftRisk || '{}')
      if (!parsedRisk.value) {
        throw new Error(`Risk limits JSON error: ${parsedRisk.error || 'invalid object'}`)
      }

      return updateTrader(trader.id, {
        name: draftName.trim(),
        interval_seconds: Math.max(1, Number(draftInterval || 60)),
        sources: draftSources.split(',').map((item) => item.trim()).filter(Boolean),
        params: parsedParams.value,
        risk_limits: parsedRisk.value,
      })
    },
    onSuccess: () => {
      setSaveError(null)
      refreshAll()
    },
    onError: (error: unknown) => {
      const message = error instanceof Error ? error.message : 'Failed to save trader'
      setSaveError(message)
    },
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

  const killSwitchOn = Boolean(overviewQuery.data?.control?.kill_switch)
  const globalMode = String(overviewQuery.data?.control?.mode || 'paper').toLowerCase()
  const orchestratorRunning = Boolean(overviewQuery.data?.control?.is_enabled) && !Boolean(overviewQuery.data?.control?.is_paused)
  const worker = overviewQuery.data?.worker
  const metrics = overviewQuery.data?.metrics

  const simulationAccounts = simulationAccountsQuery.data || []
  const selectedSandboxAccount = simulationAccounts.find((account) => account.id === selectedAccountId)
  const selectedAccountIsLive = Boolean(selectedAccountId?.startsWith('live:'))
  const selectedAccountValid = selectedAccountIsLive || Boolean(selectedSandboxAccount)
  const selectedAccountMode = selectedAccountIsLive ? 'live' : 'paper'
  const modeMismatch = selectedAccountValid && orchestratorRunning && globalMode !== selectedAccountMode

  const controlBusy =
    startBySelectedAccountMutation.isPending ||
    stopByModeMutation.isPending ||
    killSwitchMutation.isPending

  const traderNameById = useMemo(
    () => Object.fromEntries(traders.map((trader) => [trader.id, trader.name])) as Record<string, string>,
    [traders]
  )

  const globalSummary = useMemo(() => {
    let resolved = 0
    let wins = 0
    let losses = 0
    let failed = 0
    let open = 0
    let totalNotional = 0
    let resolvedPnl = 0
    let edgeSum = 0
    let edgeCount = 0
    let confidenceSum = 0
    let confidenceCount = 0

    const byTrader = new Map<string, {
      traderId: string
      traderName: string
      orders: number
      open: number
      resolved: number
      pnl: number
      notional: number
      wins: number
      losses: number
    }>()

    const bySource = new Map<string, {
      source: string
      orders: number
      resolved: number
      pnl: number
      notional: number
      wins: number
      losses: number
    }>()

    for (const order of allOrders) {
      const status = normalizeStatus(order.status)
      const notional = Math.abs(toNumber(order.notional_usd))
      const pnl = toNumber(order.actual_profit)
      const edge = toNumber(order.edge_percent)
      const confidence = toNumber(order.confidence)
      const traderId = String(order.trader_id || 'unknown')
      const traderName = traderNameById[traderId] || shortId(traderId)
      const source = String(order.source || 'unknown')

      totalNotional += notional
      if (edge !== 0) {
        edgeSum += edge
        edgeCount += 1
      }
      if (confidence !== 0) {
        confidenceSum += confidence
        confidenceCount += 1
      }

      if (!byTrader.has(traderId)) {
        byTrader.set(traderId, {
          traderId,
          traderName,
          orders: 0,
          open: 0,
          resolved: 0,
          pnl: 0,
          notional: 0,
          wins: 0,
          losses: 0,
        })
      }

      if (!bySource.has(source)) {
        bySource.set(source, {
          source,
          orders: 0,
          resolved: 0,
          pnl: 0,
          notional: 0,
          wins: 0,
          losses: 0,
        })
      }

      const traderRow = byTrader.get(traderId)
      const sourceRow = bySource.get(source)
      if (!traderRow || !sourceRow) continue

      traderRow.orders += 1
      traderRow.notional += notional
      sourceRow.orders += 1
      sourceRow.notional += notional

      if (OPEN_ORDER_STATUSES.has(status)) {
        open += 1
        traderRow.open += 1
      }

      if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolved += 1
        resolvedPnl += pnl
        traderRow.resolved += 1
        sourceRow.resolved += 1
        traderRow.pnl += pnl
        sourceRow.pnl += pnl

        if (pnl > 0) {
          wins += 1
          traderRow.wins += 1
          sourceRow.wins += 1
        }
        if (pnl < 0) {
          losses += 1
          traderRow.losses += 1
          sourceRow.losses += 1
        }
      }

      if (FAILED_ORDER_STATUSES.has(status)) {
        failed += 1
      }
    }

    const winRate = resolved > 0 ? (wins / resolved) * 100 : 0
    const topTraderRows = Array.from(byTrader.values())
      .sort((a, b) => b.pnl - a.pnl)
      .slice(0, 8)

    const sourceRows = Array.from(bySource.values())
      .sort((a, b) => b.orders - a.orders)
      .slice(0, 8)

    return {
      open,
      resolved,
      wins,
      losses,
      failed,
      totalNotional,
      resolvedPnl,
      winRate,
      avgEdge: edgeCount > 0 ? edgeSum / edgeCount : 0,
      avgConfidence: confidenceCount > 0 ? confidenceSum / confidenceCount : 0,
      topTraderRows,
      sourceRows,
    }
  }, [allOrders, traderNameById])

  const globalPositionBook = useMemo(
    () => buildPositionBookRows(allOrders, traderNameById),
    [allOrders, traderNameById]
  )

  const selectedPositionBook = useMemo(
    () => globalPositionBook.filter((row) => row.traderId === selectedTraderId),
    [globalPositionBook, selectedTraderId]
  )

  const selectedTraderSummary = useMemo(() => {
    let resolved = 0
    let wins = 0
    let losses = 0
    let failed = 0
    let open = 0
    let pnl = 0
    let notional = 0
    let edgeSum = 0
    let edgeCount = 0
    let confidenceSum = 0
    let confidenceCount = 0

    for (const order of selectedOrders) {
      const status = normalizeStatus(order.status)
      const orderPnl = toNumber(order.actual_profit)
      const orderNotional = Math.abs(toNumber(order.notional_usd))
      const edge = toNumber(order.edge_percent)
      const confidence = toNumber(order.confidence)
      notional += orderNotional

      if (OPEN_ORDER_STATUSES.has(status)) {
        open += 1
      }
      if (RESOLVED_ORDER_STATUSES.has(status)) {
        resolved += 1
        pnl += orderPnl
        if (orderPnl > 0) wins += 1
        if (orderPnl < 0) losses += 1
      }
      if (FAILED_ORDER_STATUSES.has(status)) {
        failed += 1
      }
      if (edge !== 0) {
        edgeSum += edge
        edgeCount += 1
      }
      if (confidence !== 0) {
        confidenceSum += confidence
        confidenceCount += 1
      }
    }

    const decisions = selectedDecisions.length
    const selectedDecisionsCount = selectedDecisions.filter(
      (decision) => String(decision.decision).toLowerCase() === 'selected'
    ).length

    return {
      resolved,
      wins,
      losses,
      failed,
      open,
      pnl,
      notional,
      winRate: resolved > 0 ? (wins / resolved) * 100 : 0,
      decisions,
      selectedDecisions: selectedDecisionsCount,
      events: selectedEvents.length,
      conversion: decisions > 0 ? (selectedOrders.length / decisions) * 100 : 0,
      selectionRate: decisions > 0 ? (selectedDecisionsCount / decisions) * 100 : 0,
      avgEdge: edgeCount > 0 ? edgeSum / edgeCount : 0,
      avgConfidence: confidenceCount > 0 ? confidenceSum / confidenceCount : 0,
    }
  }, [selectedOrders, selectedDecisions, selectedEvents.length])

  const filteredDecisions = useMemo(() => {
    const q = decisionSearch.trim().toLowerCase()
    return selectedDecisions
      .filter((decision) => {
        if (!q) return true
        const haystack = `${decision.source} ${decision.strategy_key} ${decision.reason || ''} ${decision.decision}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 200)
  }, [selectedDecisions, decisionSearch])

  const filteredTradeHistory = useMemo(() => {
    const q = tradeSearch.trim().toLowerCase()
    return selectedOrders
      .filter((order) => {
        const status = normalizeStatus(order.status)
        const matchesStatus =
          tradeStatusFilter === 'all' ||
          (tradeStatusFilter === 'open' && OPEN_ORDER_STATUSES.has(status)) ||
          (tradeStatusFilter === 'resolved' && RESOLVED_ORDER_STATUSES.has(status)) ||
          (tradeStatusFilter === 'failed' && FAILED_ORDER_STATUSES.has(status))

        if (!matchesStatus) return false
        if (!q) return true

        const haystack = `${order.market_question || ''} ${order.market_id || ''} ${order.source || ''} ${order.direction || ''}`.toLowerCase()
        return haystack.includes(q)
      })
      .slice(0, 250)
  }, [selectedOrders, tradeSearch, tradeStatusFilter])

  const activityRows = useMemo(() => {
    const decisionRows: ActivityRow[] = allDecisions.map((decision) => ({
      kind: 'decision',
      id: decision.id,
      ts: decision.created_at,
      traderId: decision.trader_id,
      title: `${String(decision.decision).toUpperCase()} • ${decision.source}`,
      detail: decision.reason || decision.strategy_key,
      tone: String(decision.decision).toLowerCase() === 'selected' ? 'positive' : 'neutral',
    }))

    const orderRows: ActivityRow[] = allOrders.map((order) => {
      const status = normalizeStatus(order.status)
      const pnl = toNumber(order.actual_profit)
      const tone: ActivityRow['tone'] =
        FAILED_ORDER_STATUSES.has(status) ? 'negative' :
        RESOLVED_ORDER_STATUSES.has(status) && pnl > 0 ? 'positive' :
        RESOLVED_ORDER_STATUSES.has(status) && pnl < 0 ? 'negative' : 'neutral'

      return {
        kind: 'order',
        id: order.id,
        ts: order.created_at,
        traderId: order.trader_id,
        title: `${status.toUpperCase()} • ${order.market_question || order.market_id}`,
        detail: `${formatCurrency(toNumber(order.notional_usd))} • ${String(order.mode || '').toUpperCase()}`,
        tone,
      }
    })

    const eventRows: ActivityRow[] = allEvents.map((event) => ({
      kind: 'event',
      id: event.id,
      ts: event.created_at,
      traderId: event.trader_id,
      title: `${event.event_type} • ${event.severity}`,
      detail: event.message || 'No message provided',
      tone: String(event.severity || '').toLowerCase() === 'warn' ? 'warning' : 'neutral',
    }))

    return [...decisionRows, ...orderRows, ...eventRows]
      .sort((a, b) => toTs(b.ts) - toTs(a.ts))
      .slice(0, 350)
  }, [allDecisions, allOrders, allEvents])

  const filteredActivityRows = useMemo(() => {
    return activityRows.filter((row) => {
      const kindMatches = feedFilter === 'all' || row.kind === feedFilter
      const scopeMatches = scopeFilter === 'all' || row.traderId === selectedTraderId
      return kindMatches && scopeMatches
    })
  }, [activityRows, feedFilter, scopeFilter, selectedTraderId])

  const selectedTraderActivityRows = useMemo(
    () => activityRows.filter((row) => row.traderId === selectedTraderId),
    [activityRows, selectedTraderId]
  )

  const filteredTraderActivityRows = useMemo(() => {
    return selectedTraderActivityRows
      .filter((row) => traderFeedFilter === 'all' || row.kind === traderFeedFilter)
      .slice(0, 240)
  }, [selectedTraderActivityRows, traderFeedFilter])

  const riskActivityRows = useMemo(
    () => activityRows.filter((row) => row.tone === 'negative' || row.tone === 'warning').slice(0, 240),
    [activityRows]
  )

  const recentSelectedDecisions = useMemo(
    () => allDecisions.filter((decision) => String(decision.decision).toLowerCase() === 'selected').slice(0, 24),
    [allDecisions]
  )

  const selectedDecisionTotal = useMemo(
    () => allDecisions.filter((decision) => String(decision.decision).toLowerCase() === 'selected').length,
    [allDecisions]
  )

  const selectedTraderExposure = useMemo(
    () => selectedPositionBook.reduce((sum, row) => sum + row.exposureUsd, 0),
    [selectedPositionBook]
  )

  const selectedDecision = useMemo(
    () => selectedDecisions.find((decision) => decision.id === selectedDecisionId) || null,
    [selectedDecisions, selectedDecisionId]
  )

  const paramsPreview = useMemo(() => {
    const parsed = parseJsonObject(draftParams || '{}')
    if (!parsed.value) return []
    return Object.entries(parsed.value).slice(0, 10)
  }, [draftParams])

  const riskPreview = useMemo(() => {
    const parsed = parseJsonObject(draftRisk || '{}')
    if (!parsed.value) return []
    return Object.entries(parsed.value).slice(0, 10)
  }, [draftRisk])

  const runRate = toNumber(metrics?.decisions_count) > 0
    ? (toNumber(metrics?.orders_count) / toNumber(metrics?.decisions_count)) * 100
    : 0
  const displayAvgEdge = normalizeEdgePercent(globalSummary.avgEdge)
  const displayAvgConfidence = normalizeConfidencePercent(globalSummary.avgConfidence)

  const shellLoading = overviewQuery.isLoading || tradersQuery.isLoading

  if (shellLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 flex items-center justify-center gap-3 text-sm text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading orchestrator control plane...
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-3">
      <Card className="shrink-0 border-cyan-500/30 bg-gradient-to-br from-cyan-500/[0.08] via-card to-emerald-500/[0.08]">
        <CardContent className="px-3 py-2 space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-300" />
            <span className="text-sm font-semibold leading-none">Autotrading Orchestration Hub</span>
            <Badge className="h-5 px-1.5 text-[10px]" variant={orchestratorRunning ? 'default' : 'secondary'}>
              {orchestratorRunning ? 'RUNNING' : 'STOPPED'}
            </Badge>
            <Badge className="h-5 px-1.5 text-[10px]" variant={globalMode === 'live' ? 'destructive' : 'outline'}>
              {globalMode.toUpperCase()}
            </Badge>
            <Badge className="h-5 px-1.5 text-[10px]" variant={killSwitchOn ? 'destructive' : 'outline'}>
              {killSwitchOn ? 'KILL ON' : 'KILL OFF'}
            </Badge>
            <div className="ml-auto text-[11px] text-muted-foreground flex items-center gap-1">
              <Clock3 className="w-3 h-3" />
              {formatTimestamp(worker?.last_run_at)}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            <Button
              onClick={() => startBySelectedAccountMutation.mutate()}
              disabled={controlBusy || !selectedAccountId || !selectedAccountValid || (selectedAccountIsLive && killSwitchOn)}
              className="h-7 min-w-[152px] text-[11px]"
            >
              {startBySelectedAccountMutation.isPending ? (
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              ) : selectedAccountIsLive ? (
                <Zap className="w-3.5 h-3.5 mr-1.5" />
              ) : (
                <Play className="w-3.5 h-3.5 mr-1.5" />
              )}
              Start ({selectedAccountMode.toUpperCase()})
            </Button>
            <Button
              variant="secondary"
              onClick={() => stopByModeMutation.mutate()}
              disabled={controlBusy}
              className="h-7 min-w-[112px] text-[11px]"
            >
              {stopByModeMutation.isPending ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Square className="w-3.5 h-3.5 mr-1.5" />}
              Stop Engine
            </Button>
            <Button
              variant="outline"
              onClick={() => deployTemplatesMutation.mutate()}
              disabled={deployTemplatesMutation.isPending}
              className="h-7 min-w-[126px] text-[11px]"
            >
              {deployTemplatesMutation.isPending ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : null}
              Seed Templates
            </Button>
            <div className="ml-auto flex items-center gap-1.5 rounded-md border border-red-500/30 bg-red-500/10 px-2 py-1">
              <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
              <span className="text-[11px] text-red-300">Kill Switch</span>
              <Switch
                checked={killSwitchOn}
                onCheckedChange={(enabled) => killSwitchMutation.mutate(enabled)}
                disabled={killSwitchMutation.isPending}
              />
            </div>
            <div className={cn(
              'rounded-md border px-2 py-1 text-[11px]',
              !selectedAccountValid || modeMismatch
                ? 'border-amber-500/40 bg-amber-500/10 text-amber-100'
                : 'border-border/70 bg-background/70 text-muted-foreground'
            )}>
              {!selectedAccountValid
                ? 'Select global account'
                : modeMismatch
                  ? `Mode mismatch (${globalMode.toUpperCase()} vs ${selectedAccountMode.toUpperCase()})`
                  : `Mode synced (${selectedAccountMode.toUpperCase()})`}
            </div>
          </div>

          <div className="grid gap-1.5 grid-cols-2 sm:grid-cols-4 xl:grid-cols-8">
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Worker</p>
              <p className="text-[11px] truncate" title={worker?.current_activity || 'Idle'}>
                {worker?.current_activity || 'Idle'}
              </p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Health</p>
              <p className="text-[11px] flex items-center gap-1">
                {worker?.last_error ? <AlertTriangle className="w-3 h-3 text-amber-400" /> : <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
                {worker?.last_error ? 'Degraded' : 'Healthy'}
              </p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Traders</p>
              <p className="text-[11px]">{toNumber(metrics?.traders_running)} / {toNumber(metrics?.traders_total)} active</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Capture</p>
              <p className="text-[11px]">{formatPercent(runRate)}</p>
            </div>
            <div className={cn(
              'rounded-md border px-2 py-1',
              toNumber(metrics?.daily_pnl) >= 0 ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'
            )}>
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Daily PnL</p>
              <p className="text-[11px] font-mono">{formatCurrency(toNumber(metrics?.daily_pnl))}</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Exposure</p>
              <p className="text-[11px] font-mono">{formatCurrency(toNumber(metrics?.gross_exposure_usd), true)}</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Open / Orders</p>
              <p className="text-[11px] font-mono">{globalSummary.open} / {allOrders.length}</p>
            </div>
            <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground">Win / Edge / Conf</p>
              <p className="text-[11px] font-mono">
                {formatPercent(globalSummary.winRate)} / {formatPercent(displayAvgEdge)} / {formatPercent(displayAvgConfidence)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="command" className="flex-1 min-h-0 flex flex-col gap-2">
        <TabsList className="w-full justify-start overflow-auto shrink-0 h-9">
          <TabsTrigger value="command">Command Center</TabsTrigger>
          <TabsTrigger value="traders">Trader Intelligence</TabsTrigger>
          <TabsTrigger value="governance">Governance</TabsTrigger>
        </TabsList>

        <TabsContent value="command" className="mt-0 flex-1 min-h-0">
          <div className="h-full min-h-0 grid gap-3 grid-rows-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="grid gap-3 xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)] min-h-0">
              <Card className="h-full flex flex-col min-h-0">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                    <span>Live Global Terminal</span>
                    <Badge variant="outline">{filteredActivityRows.length} rows</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0">
                  <div className="flex flex-wrap items-center gap-1 mb-2">
                    {(['all', 'decision', 'order', 'event'] as FeedFilter[]).map((kind) => (
                      <Button
                        key={kind}
                        size="sm"
                        variant={feedFilter === kind ? 'default' : 'outline'}
                        onClick={() => setFeedFilter(kind)}
                        className="h-6 px-2 text-[11px]"
                      >
                        {kind}
                      </Button>
                    ))}
                    <div className="ml-auto flex items-center gap-1">
                      <Button
                        size="sm"
                        variant={scopeFilter === 'all' ? 'default' : 'outline'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setScopeFilter('all')}
                      >
                        all
                      </Button>
                      <Button
                        size="sm"
                        variant={scopeFilter === 'selected' ? 'default' : 'outline'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setScopeFilter('selected')}
                        disabled={!selectedTraderId}
                      >
                        selected
                      </Button>
                    </div>
                  </div>

                  {filteredActivityRows.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No events matching filters.</div>
                  ) : (
                    <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 bg-black/20">
                      <div className="space-y-1 p-2 font-mono text-[11px] leading-relaxed">
                        {filteredActivityRows.map((row) => (
                          <div
                            key={`${row.kind}:${row.id}`}
                            className={cn(
                              'rounded border px-2 py-1',
                              row.tone === 'positive' && 'border-emerald-500/30 text-emerald-100',
                              row.tone === 'negative' && 'border-red-500/35 text-red-100',
                              row.tone === 'warning' && 'border-amber-500/35 text-amber-100',
                              row.tone === 'neutral' && 'border-border text-foreground'
                            )}
                          >
                            <span className="text-muted-foreground">[{formatTimestamp(row.ts)}]</span>{' '}
                            <span className="uppercase">{row.kind}</span>{' '}
                            <span className="text-muted-foreground">{traderNameById[String(row.traderId || '')] || shortId(row.traderId || '')}</span>{' '}
                            <span>{row.title}</span>
                            <span className="text-muted-foreground"> :: {row.detail}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>

              <Card className="h-full flex flex-col min-h-0">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm">Execution Radar</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 space-y-2">
                  <div className="grid gap-2 grid-cols-2">
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Decisions</p>
                      <p className="text-sm font-mono">{allDecisions.length}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Orders</p>
                      <p className="text-sm font-mono">{allOrders.length}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Conversion</p>
                      <p className="text-sm font-mono">{formatPercent(runRate)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Worker State</p>
                      <p className="text-sm">{worker?.last_error ? 'Degraded' : 'Healthy'}</p>
                    </div>
                  </div>

                  <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 p-2">
                    <div className="space-y-3 pr-2">
                      <div>
                        <p className="text-[11px] uppercase tracking-widest text-muted-foreground mb-1">Source Leaders</p>
                        {globalSummary.sourceRows.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No source-level data.</p>
                        ) : (
                          <div className="space-y-1.5">
                            {globalSummary.sourceRows.slice(0, 6).map((row) => (
                              <div key={row.source} className="rounded-md border border-border/70 px-2 py-1.5">
                                <div className="flex items-center justify-between text-xs">
                                  <span>{row.source}</span>
                                  <span className="font-mono">{row.orders}</span>
                                </div>
                                <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                                  <span>{formatCurrency(row.notional, true)}</span>
                                  <span className={cn(row.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>{formatCurrency(row.pnl)}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      <div>
                        <p className="text-[11px] uppercase tracking-widest text-muted-foreground mb-1">Trader Leaders</p>
                        {globalSummary.topTraderRows.slice(0, 6).map((row) => (
                          <div key={row.traderId} className="rounded-md border border-border/70 px-2 py-1.5 mb-1.5">
                            <div className="flex items-center justify-between text-xs">
                              <span className="truncate">{row.traderName}</span>
                              <span className={cn('font-mono', row.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                {formatCurrency(row.pnl)}
                              </span>
                            </div>
                            <div className="mt-1 text-[11px] text-muted-foreground">
                              {row.orders} orders • {row.resolved} resolved
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-3 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] min-h-0">
              <Card className="h-full flex flex-col min-h-0">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm">Open Position Book</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0">
                  {globalPositionBook.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-sm text-muted-foreground">No live positions are currently held by orchestrator traders.</div>
                  ) : (
                    <ScrollArea className="h-full min-h-0">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Trader</TableHead>
                            <TableHead>Market</TableHead>
                            <TableHead>Dir</TableHead>
                            <TableHead className="text-right">Exposure</TableHead>
                            <TableHead className="text-right">Avg Px</TableHead>
                            <TableHead className="text-right">Edge</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {globalPositionBook.slice(0, 150).map((row) => (
                            <TableRow key={row.key}>
                              <TableCell className="font-medium">{row.traderName}</TableCell>
                              <TableCell>
                                <div className="max-w-[320px] truncate" title={row.marketQuestion}>{row.marketQuestion}</div>
                                <div className="text-[11px] text-muted-foreground">{shortId(row.marketId)}</div>
                              </TableCell>
                              <TableCell>
                                <Badge variant={row.direction === 'BUY' ? 'default' : row.direction === 'SELL' ? 'secondary' : 'outline'}>{row.direction}</Badge>
                              </TableCell>
                              <TableCell className="text-right font-mono">{formatCurrency(row.exposureUsd)}</TableCell>
                              <TableCell className="text-right font-mono">{row.averagePrice !== null ? row.averagePrice.toFixed(3) : 'n/a'}</TableCell>
                              <TableCell className="text-right font-mono">{row.weightedEdge !== null ? formatPercent(normalizeEdgePercent(row.weightedEdge)) : 'n/a'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>

              <Card className="h-full flex flex-col min-h-0">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm">Decision Funnel + Selection Queue</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 space-y-2">
                  <div className="grid gap-2 grid-cols-2">
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Observed</p>
                      <p className="text-sm font-mono">{allDecisions.length}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Selected</p>
                      <p className="text-sm font-mono">{selectedDecisionTotal}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Orders Emitted</p>
                      <p className="text-sm font-mono">{toNumber(metrics?.orders_count)}</p>
                    </div>
                    <div className="rounded-md border border-border/70 bg-background/70 px-2 py-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Failed Orders</p>
                      <p className="text-sm font-mono">{globalSummary.failed}</p>
                    </div>
                  </div>

                  <div className={cn(
                    'rounded-md border px-2 py-1.5 text-xs',
                    worker?.last_error ? 'border-amber-500/40 bg-amber-500/10 text-amber-100' : 'border-border/70 bg-background/70 text-muted-foreground'
                  )}>
                    {worker?.last_error ? `Worker alert: ${worker.last_error}` : 'No worker alerts. Decision and order pipeline healthy.'}
                  </div>

                  <ScrollArea className="h-full min-h-0 rounded-md border border-border/70 bg-black/20">
                    <div className="space-y-1 p-2">
                      {recentSelectedDecisions.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No selected decisions yet.</p>
                      ) : (
                        recentSelectedDecisions.map((decision) => (
                          <div key={decision.id} className="rounded-md border border-border/70 px-2 py-1.5">
                            <div className="flex items-center justify-between text-[11px]">
                              <span className="font-mono text-muted-foreground">{formatTimestamp(decision.created_at)}</span>
                              <span className="font-mono">{decision.score !== null ? decision.score.toFixed(2) : 'n/a'}</span>
                            </div>
                            <p className="text-xs mt-1 truncate" title={`${decision.source} • ${decision.strategy_key}`}>
                              {decision.source} • {decision.strategy_key}
                            </p>
                            <p className="text-[11px] text-muted-foreground truncate" title={decision.reason || ''}>
                              {decision.reason || 'No decision rationale recorded.'}
                            </p>
                          </div>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="traders" className="mt-0 flex-1 min-h-0">
          <div className="h-full min-h-0 grid gap-3 xl:grid-cols-[250px_minmax(0,1fr)]">
            <Card className="h-full flex flex-col min-h-0">
              <CardHeader className="py-2">
                <CardTitle className="text-sm">Trader Roster</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0">
                <ScrollArea className="h-full min-h-0">
                  <div className="space-y-2 pr-2">
                    {traders.map((trader) => {
                      const isSelected = selectedTraderId === trader.id
                      const traderPnl = globalSummary.topTraderRows.find((row) => row.traderId === trader.id)?.pnl || 0

                      return (
                        <div
                          key={trader.id}
                          className={cn(
                            'rounded-md border p-2',
                            isSelected ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-border'
                          )}
                        >
                          <button className="w-full text-left" onClick={() => setSelectedTraderId(trader.id)}>
                            <div className="flex items-center justify-between gap-2">
                              <p className="font-medium text-sm truncate">{trader.name}</p>
                              <Badge variant={trader.is_paused || !trader.is_enabled ? 'secondary' : 'default'}>
                                {trader.is_paused || !trader.is_enabled ? 'Paused' : 'Active'}
                              </Badge>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">{trader.strategy_key} • {trader.interval_seconds}s</p>
                            <p className={cn('text-xs mt-1 font-mono', traderPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                              {formatCurrency(traderPnl)} realized
                            </p>
                          </button>
                          <div className="mt-2 flex gap-1">
                            <Button size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={() => traderStartMutation.mutate(trader.id)} disabled={traderStartMutation.isPending}>Start</Button>
                            <Button size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={() => traderPauseMutation.mutate(trader.id)} disabled={traderPauseMutation.isPending}>Pause</Button>
                            <Button size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={() => traderRunOnceMutation.mutate(trader.id)} disabled={traderRunOnceMutation.isPending}>Run</Button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <div className="h-full min-h-0 grid gap-3 grid-rows-[minmax(0,0.7fr)_minmax(0,1fr)] min-w-0">
              <Card className="h-full flex flex-col min-h-0">
                <CardHeader className="py-2">
                  <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                    <span>{selectedTrader?.name || 'Trader'} Performance + Positioning</span>
                    <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                      <Badge variant={selectedTrader?.is_paused || !selectedTrader?.is_enabled ? 'secondary' : 'default'}>
                        {selectedTrader?.is_paused || !selectedTrader?.is_enabled ? 'Paused' : 'Running'}
                      </Badge>
                      <span>{formatTimestamp(selectedTrader?.last_run_at)}</span>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 grid gap-2 grid-cols-1 xl:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)]">
                  <div className="rounded-md border border-border p-2 h-full min-h-0">
                    {selectedPnlSeries.length > 1 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={selectedPnlSeries}>
                          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.25} />
                          <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                          <YAxis tick={{ fontSize: 11 }} width={64} />
                          <Tooltip />
                          <Area type="monotone" dataKey="cumulativePnl" name="Cumulative PnL" stroke="#22d3ee" fill="#22d3ee22" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-sm text-muted-foreground">Need more order history for chart.</div>
                    )}
                  </div>
                  <div className="space-y-2 min-h-0">
                    <div className="grid gap-2 grid-cols-2">
                      <div className="rounded-md border border-border px-2 py-1.5">
                        <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Open</p>
                        <p className="text-base font-mono font-semibold">{selectedTraderSummary.open}</p>
                      </div>
                      <div className="rounded-md border border-border px-2 py-1.5">
                        <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Win Rate</p>
                        <p className="text-base font-mono font-semibold">{formatPercent(selectedTraderSummary.winRate)}</p>
                      </div>
                      <div className="rounded-md border border-border px-2 py-1.5">
                        <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Decisions</p>
                        <p className="text-base font-mono font-semibold">{selectedTraderSummary.decisions}</p>
                      </div>
                      <div className="rounded-md border border-border px-2 py-1.5">
                        <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Realized PnL</p>
                        <p className={cn('text-base font-mono font-semibold', selectedTraderSummary.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                          {formatCurrency(selectedTraderSummary.pnl)}
                        </p>
                      </div>
                    </div>
                    <div className="rounded-md border border-border p-2 min-h-0">
                      <p className="text-[11px] font-medium mb-2">Decision Mix</p>
                      {decisionMix.length === 0 ? (
                        <p className="text-xs text-muted-foreground">No decision history yet.</p>
                      ) : (
                        <div className="space-y-1.5">
                          {decisionMix.map((row) => {
                            const percent = selectedDecisions.length > 0 ? (row.count / selectedDecisions.length) * 100 : 0
                            return (
                              <div key={row.decision}>
                                <div className="flex items-center justify-between text-[11px] mb-1">
                                  <span>{row.decision}</span>
                                  <span className="font-mono">{row.count}</span>
                                </div>
                                <Progress value={percent} className="h-1.5" />
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <div className="grid gap-3 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] min-h-0">
                <Card className="h-full flex flex-col min-h-0">
                  <CardHeader className="py-2">
                    <CardTitle className="text-sm">Decision Logic Inspector</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 min-h-0 grid gap-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                    <div className="space-y-2 min-h-0">
                      <Input
                        value={decisionSearch}
                        onChange={(event) => setDecisionSearch(event.target.value)}
                        placeholder="Filter decisions..."
                      />
                      <ScrollArea className="h-full min-h-0 rounded-md border border-border/80 p-2">
                        <div className="space-y-2 pr-2">
                          {filteredDecisions.map((decision) => (
                            <button
                              key={decision.id}
                              onClick={() => setSelectedDecisionId(decision.id)}
                              className={cn(
                                'w-full text-left rounded-md border p-2 transition-colors',
                                selectedDecisionId === decision.id ? 'border-cyan-500/40 bg-cyan-500/10' : 'border-border hover:bg-muted/30'
                              )}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <Badge variant={String(decision.decision).toLowerCase() === 'selected' ? 'default' : 'outline'}>
                                  {String(decision.decision).toUpperCase()}
                                </Badge>
                                <span className="text-[11px] text-muted-foreground">{formatShortDate(decision.created_at)}</span>
                              </div>
                              <p className="text-xs mt-1 font-medium">{decision.source} • {decision.strategy_key}</p>
                              <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">{decision.reason || 'No reason captured'}</p>
                            </button>
                          ))}
                          {filteredDecisions.length === 0 ? <p className="text-sm text-muted-foreground">No decisions matching filter.</p> : null}
                        </div>
                      </ScrollArea>
                    </div>

                    <div className="space-y-2 min-h-0">
                      {!selectedDecision ? (
                        <p className="text-sm text-muted-foreground">Select a decision to inspect checks and risk logic.</p>
                      ) : (
                        <>
                          <div className="rounded-md border border-border p-2">
                            <div className="flex items-center justify-between gap-2">
                              <p className="font-medium text-sm">{selectedDecision.strategy_key}</p>
                              <Badge variant={String(selectedDecision.decision).toLowerCase() === 'selected' ? 'default' : 'outline'}>
                                {String(selectedDecision.decision).toUpperCase()}
                              </Badge>
                            </div>
                            <p className="text-[11px] text-muted-foreground mt-1">{selectedDecision.reason || 'No reason text'}</p>
                            <div className="grid grid-cols-2 gap-2 mt-2 text-[11px] text-muted-foreground">
                              <span>Score: {selectedDecision.score !== null ? selectedDecision.score.toFixed(2) : 'n/a'}</span>
                              <span>{formatTimestamp(selectedDecision.created_at)}</span>
                            </div>
                          </div>

                          <ScrollArea className="h-full min-h-0 rounded-md border border-border/80 p-2">
                            {(decisionDetailQuery.data?.checks || []).length === 0 ? (
                              <p className="text-sm text-muted-foreground">No check records for this decision.</p>
                            ) : (
                              <div className="space-y-2 pr-2">
                                {(decisionDetailQuery.data?.checks || []).map((check) => (
                                  <div key={check.id} className={cn(
                                    'rounded-md border p-2',
                                    check.passed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'
                                  )}>
                                    <div className="flex items-center justify-between gap-2">
                                      <p className="text-xs font-medium">{check.check_label}</p>
                                      <Badge variant={check.passed ? 'default' : 'destructive'}>{check.passed ? 'PASS' : 'FAIL'}</Badge>
                                    </div>
                                    <p className="text-[11px] text-muted-foreground mt-1">{check.detail || 'No detail provided'}</p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </ScrollArea>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card className="h-full flex flex-col min-h-0">
                  <CardHeader className="py-2">
                    <CardTitle className="text-sm flex flex-wrap items-center justify-between gap-2">
                      <span>Trade History + Financial Outcomes</span>
                      <div className="flex items-center gap-1 flex-wrap">
                        {(['all', 'open', 'resolved', 'failed'] as TradeStatusFilter[]).map((filter) => (
                          <Button
                            key={filter}
                            size="sm"
                            variant={tradeStatusFilter === filter ? 'default' : 'outline'}
                            className="h-6 px-2 text-[11px]"
                            onClick={() => setTradeStatusFilter(filter)}
                          >
                            {filter}
                          </Button>
                        ))}
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 min-h-0 space-y-2">
                    <Input
                      value={tradeSearch}
                      onChange={(event) => setTradeSearch(event.target.value)}
                      placeholder="Search market, source, direction..."
                    />

                    {selectedPositionBook.length > 0 ? (
                      <div className="rounded-md border border-border px-2 py-1.5">
                        <p className="text-[11px] text-muted-foreground mb-1">Positions held by selected trader</p>
                        <div className="flex flex-wrap gap-x-3 gap-y-1">
                          {selectedPositionBook.slice(0, 6).map((row) => (
                            <div key={row.key} className="text-[11px]">
                              <span className="text-muted-foreground">{row.direction}</span>{' '}
                              <span className="font-mono">{formatCurrency(row.exposureUsd)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    <ScrollArea className="h-full min-h-0 rounded-md border border-border/80">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Market</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>Dir</TableHead>
                            <TableHead className="text-right">Notional</TableHead>
                            <TableHead className="text-right">Edge</TableHead>
                            <TableHead className="text-right">P/L</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {filteredTradeHistory.map((order) => {
                            const status = normalizeStatus(order.status)
                            const pnl = toNumber(order.actual_profit)
                            return (
                              <TableRow key={order.id}>
                                <TableCell>
                                  <div className="max-w-[320px] truncate" title={order.market_question || order.market_id}>{order.market_question || order.market_id}</div>
                                  <div className="text-[11px] text-muted-foreground">{formatShortDate(order.executed_at || order.created_at)}</div>
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    variant={
                                      FAILED_ORDER_STATUSES.has(status) ? 'destructive' :
                                        RESOLVED_ORDER_STATUSES.has(status) ? 'default' :
                                          'outline'
                                    }
                                  >
                                    {status}
                                  </Badge>
                                </TableCell>
                                <TableCell>{String(order.direction || 'n/a').toUpperCase()}</TableCell>
                                <TableCell className="text-right font-mono">{formatCurrency(toNumber(order.notional_usd))}</TableCell>
                                <TableCell className="text-right font-mono">{order.edge_percent !== null ? formatPercent(normalizeEdgePercent(toNumber(order.edge_percent))) : 'n/a'}</TableCell>
                                <TableCell className={cn('text-right font-mono', pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                  {order.actual_profit !== null ? formatCurrency(pnl) : 'n/a'}
                                </TableCell>
                              </TableRow>
                            )
                          })}
                          {filteredTradeHistory.length === 0 ? (
                            <TableRow>
                              <TableCell colSpan={6} className="text-center text-muted-foreground">No trades match the selected filters.</TableCell>
                            </TableRow>
                          ) : null}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="governance" className="mt-0 flex-1 min-h-0">
          <div className="h-full min-h-0 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Card className="h-full flex flex-col min-h-0">
              <CardHeader className="py-2">
                <CardTitle className="text-sm">Trader Configuration Workbench</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0">
                {!selectedTrader ? (
                  <p className="text-sm text-muted-foreground">Select a trader from the intelligence tab to edit its runtime profile.</p>
                ) : (
                  <ScrollArea className="h-full min-h-0 pr-2">
                    <div className="space-y-3 pb-2">
                    <div>
                      <Label>Name</Label>
                      <Input value={draftName} onChange={(event) => setDraftName(event.target.value)} />
                    </div>
                    <div>
                      <Label>Interval Seconds</Label>
                      <Input type="number" min={1} value={draftInterval} onChange={(event) => setDraftInterval(event.target.value)} />
                    </div>
                    <div>
                      <Label>Sources (comma separated)</Label>
                      <Input value={draftSources} onChange={(event) => setDraftSources(event.target.value)} />
                    </div>

                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-md border border-border p-2">
                        <p className="text-xs font-medium mb-2">Strategy parameter highlights</p>
                        {paramsPreview.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No top-level parameters.</p>
                        ) : (
                          <div className="space-y-1">
                            {paramsPreview.map(([key, value]) => (
                              <div key={key} className="text-xs flex items-center justify-between gap-2">
                                <span className="text-muted-foreground">{humanizeKey(key)}</span>
                                <span className="font-mono truncate" title={String(value)}>{String(value)}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="rounded-md border border-border p-2">
                        <p className="text-xs font-medium mb-2">Risk limit highlights</p>
                        {riskPreview.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No top-level limits.</p>
                        ) : (
                          <div className="space-y-1">
                            {riskPreview.map(([key, value]) => (
                              <div key={key} className="text-xs flex items-center justify-between gap-2">
                                <span className="text-muted-foreground">{humanizeKey(key)}</span>
                                <span className="font-mono truncate" title={String(value)}>{String(value)}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <details className="rounded-md border border-border p-2">
                      <summary className="cursor-pointer text-xs font-medium">Advanced JSON editor: strategy params</summary>
                      <textarea
                        className="mt-2 w-full min-h-[190px] rounded-md border bg-background p-2 text-xs font-mono"
                        value={draftParams}
                        onChange={(event) => setDraftParams(event.target.value)}
                      />
                    </details>

                    <details className="rounded-md border border-border p-2">
                      <summary className="cursor-pointer text-xs font-medium">Advanced JSON editor: risk limits</summary>
                      <textarea
                        className="mt-2 w-full min-h-[190px] rounded-md border bg-background p-2 text-xs font-mono"
                        value={draftRisk}
                        onChange={(event) => setDraftRisk(event.target.value)}
                      />
                    </details>

                    {saveError ? <div className="text-xs text-red-500">{saveError}</div> : null}

                    <Button
                      onClick={() => selectedTrader && saveTraderMutation.mutate(selectedTrader)}
                      disabled={saveTraderMutation.isPending}
                    >
                      {saveTraderMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                      Save Trader Profile
                    </Button>
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>

            <Card className="h-full flex flex-col min-h-0">
              <CardHeader className="py-2">
                <CardTitle className="text-sm">Governance Snapshot</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0">
                <ScrollArea className="h-full min-h-0 pr-2">
                  <div className="space-y-3 pb-2">
                <div className="rounded-md border border-border p-3">
                  <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Control state</p>
                  <div className="mt-2 space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Mode</span>
                      <Badge variant={globalMode === 'live' ? 'destructive' : 'outline'}>{globalMode.toUpperCase()}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Engine</span>
                      <span className="font-medium">{orchestratorRunning ? 'Running' : 'Stopped'}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Kill switch</span>
                      <span className={cn('font-medium', killSwitchOn ? 'text-red-400' : 'text-emerald-400')}>{killSwitchOn ? 'Enabled' : 'Disabled'}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Run interval</span>
                      <span className="font-medium font-mono">{toNumber(overviewQuery.data?.control?.run_interval_seconds)}s</span>
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-border p-3">
                  <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Live guardrails</p>
                  <div className="mt-2 grid grid-cols-1 gap-2 text-xs text-muted-foreground">
                    <p>Live start requires preflight pass and arm token handoff.</p>
                    <p>Kill switch blocks any future decision state of <span className="font-mono">selected</span>.</p>
                    <p>Global pause state in workers still overrides orchestrator start calls.</p>
                  </div>
                </div>

                <div className="rounded-md border border-border p-3">
                  <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Last decision checks sync</p>
                  <div className="mt-2 flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Selected decision</span>
                    <span className="font-mono">{shortId(selectedDecisionId)}</span>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Checks loaded</span>
                    <span className="font-mono">{decisionDetailQuery.data?.checks?.length || 0}</span>
                  </div>
                  <div className="mt-2 text-xs text-muted-foreground">Updated {formatTimestamp(decisionDetailQuery.data?.decision?.created_at || null)}</div>
                </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
