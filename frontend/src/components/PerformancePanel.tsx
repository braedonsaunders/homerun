import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Calendar,
  RefreshCw,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  getAccountTrades,
  getAllTraderOrders,
  getSimulationAccounts,
  getTraderOrchestratorStats,
  SimulationAccount,
  SimulationTrade,
  TraderOrder,
} from '../services/api'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { ScrollArea } from './ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table'

type ViewMode = 'simulation' | 'live' | 'all'
type TimeRange = '7d' | '30d' | '90d' | 'all'

type VenuePresentation = {
  label: string
  detail: string
  className: string
}

type UnifiedTrade = {
  id: string
  source: 'sandbox' | 'live'
  strategy: string
  marketQuestion: string
  marketId?: string
  directionLabel: string
  lifecycleLabel: string
  outcomeHeadline: string
  outcomeDetail: string
  outcomeDetailCompact: string
  modeLabel: string
  venuePresentation: VenuePresentation
  fillPx: number
  fillProgressPercent: number | null
  markPx: number
  unrealized: number
  dynamicEdgePercent: number
  exitProgressPercent: number | null
  providerSnapshotStatus?: string
  createdAt?: string
  updatedAt?: string
  cost: number
  pnl: number | null
  status: string
  executedAt: string
  accountName?: string
  isResolved: boolean
  isWin: boolean
  isLoss: boolean
}

type StrategyRollup = {
  strategy: string
  trades: number
  wins: number
  losses: number
  pnl: number
  sandboxTrades: number
  liveTrades: number
}

type LiveTraderOrder = TraderOrder & {
  executed_at: string
  total_cost: number
  strategy: string
}

const VIEW_MODE_OPTIONS: Array<{ id: ViewMode; label: string }> = [
  { id: 'all', label: 'Unified' },
  { id: 'simulation', label: 'Sandbox' },
  { id: 'live', label: 'Live' },
]

const RANGE_OPTIONS: Array<{ id: TimeRange; label: string }> = [
  { id: '7d', label: '7D' },
  { id: '30d', label: '30D' },
  { id: '90d', label: '90D' },
  { id: 'all', label: 'All Time' },
]
const TRADE_TAPE_PAGE_SIZE = 100

const OPEN_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])
const RESOLVED_ORDER_STATUSES = new Set([
  'resolved',
  'resolved_win',
  'resolved_loss',
  'closed_win',
  'closed_loss',
  'win',
  'loss',
])
const FAILED_ORDER_STATUSES = new Set(['failed', 'rejected', 'error', 'cancelled'])
const WIN_ORDER_STATUSES = new Set(['resolved_win', 'closed_win', 'win'])
const LOSS_ORDER_STATUSES = new Set(['resolved_loss', 'closed_loss', 'loss'])

function formatCurrency(value: number, compact = false): string {
  if (!Number.isFinite(value)) return '$0.00'
  if (compact) {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency: 'USD',
      notation: 'compact',
      maximumFractionDigits: 1,
    }).format(value)
  }
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function formatSignedCurrency(value: number, compact = false): string {
  const prefix = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${prefix}${formatCurrency(Math.abs(value), compact)}`
}

function formatPercent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '0.0%'
  return `${value.toFixed(digits)}%`
}

function formatDateLabel(dateStr: string): string {
  const date = new Date(dateStr)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: '2-digit',
  })
}

function normalizeStatus(value: unknown): string {
  return String(value || 'unknown').trim().toLowerCase()
}

function toTs(value: string | null | undefined): number {
  if (!value) return 0
  const ts = new Date(value).getTime()
  return Number.isFinite(ts) ? ts : 0
}

function formatRelativeAge(value: string | null | undefined): string {
  const ts = toTs(value)
  if (ts <= 0) return '—'
  const ageMs = Math.max(0, Date.now() - ts)
  if (ageMs < 60_000) return `${Math.round(ageMs / 1000)}s`
  if (ageMs < 3_600_000) return `${Math.round(ageMs / 60_000)}m`
  if (ageMs < 86_400_000) return `${Math.round(ageMs / 3_600_000)}h`
  return `${Math.round(ageMs / 86_400_000)}d`
}

function toFiniteNumber(value: unknown): number | null {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function clamp(value: number, minValue: number, maxValue: number): number {
  if (value < minValue) return minValue
  if (value > maxValue) return maxValue
  return value
}

function normalizeEdgePercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (Math.abs(value) <= 1) return value * 100
  if (Math.abs(value) > 200) return value / 100
  return value
}

function titleCaseStatusLabel(value: string): string {
  const normalized = normalizeStatus(value)
  if (!normalized) return 'Unknown'
  return normalized
    .split('_')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}

function resolveOrderLifecycleLabel(status: string): string {
  if (status === 'submitted' || status === 'pending' || status === 'queued') return 'Submitted'
  if (status === 'open') return 'Working'
  if (status === 'executed') return 'Filled'
  if (status === 'cancelled') return 'Canceled'
  if (status === 'rejected') return 'Rejected'
  if (status === 'failed' || status === 'error') return 'Failed'
  if (status === 'resolved_win') return 'Settled (Profit)'
  if (status === 'resolved_loss') return 'Settled (Loss)'
  if (status === 'closed_win') return 'Closed (Profit)'
  if (status === 'closed_loss') return 'Closed (Loss)'
  if (status === 'win') return 'Settled (Profit)'
  if (status === 'loss') return 'Settled (Loss)'
  if (status === 'resolved') return 'Settled'
  return titleCaseStatusLabel(status)
}

function resolveOrderStatusBadgePresentation(
  status: string,
  pnl: number
): { variant: 'default' | 'secondary' | 'destructive' | 'outline'; className: string } {
  if (status === 'cancelled') {
    return {
      variant: 'outline',
      className: 'border-zinc-300 bg-zinc-100 text-zinc-900 dark:border-zinc-400/45 dark:bg-zinc-500/20 dark:text-zinc-100',
    }
  }
  if (status === 'open') {
    return {
      variant: 'outline',
      className: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/20 dark:text-sky-100',
    }
  }
  if (status === 'executed') {
    return {
      variant: 'outline',
      className: 'border-indigo-300 bg-indigo-100 text-indigo-900 dark:border-indigo-400/45 dark:bg-indigo-500/20 dark:text-indigo-100',
    }
  }
  if (status === 'submitted' || status === 'pending' || status === 'queued') {
    return {
      variant: 'outline',
      className: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/20 dark:text-amber-100',
    }
  }
  if (WIN_ORDER_STATUSES.has(status)) {
    return {
      variant: 'outline',
      className: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/20 dark:text-emerald-100',
    }
  }
  if (LOSS_ORDER_STATUSES.has(status) || FAILED_ORDER_STATUSES.has(status)) {
    return {
      variant: 'outline',
      className: 'border-red-300 bg-red-100 text-red-900 dark:border-red-400/45 dark:bg-red-500/20 dark:text-red-100',
    }
  }
  if (status === 'resolved') {
    return {
      variant: 'outline',
      className: 'border-slate-300 bg-slate-100 text-slate-900 dark:border-slate-400/45 dark:bg-slate-500/20 dark:text-slate-100',
    }
  }
  if (RESOLVED_ORDER_STATUSES.has(status)) {
    if (pnl > 0) {
      return {
        variant: 'outline',
        className: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/20 dark:text-emerald-100',
      }
    }
    if (pnl < 0) {
      return {
        variant: 'outline',
        className: 'border-red-300 bg-red-100 text-red-900 dark:border-red-400/45 dark:bg-red-500/20 dark:text-red-100',
      }
    }
    return {
      variant: 'outline',
      className: 'border-slate-300 bg-slate-100 text-slate-900 dark:border-slate-400/45 dark:bg-slate-500/20 dark:text-slate-100',
    }
  }
  return {
    variant: 'outline',
    className: 'border-border bg-muted/50 text-foreground',
  }
}

function resolveOrderOutcomeBadgeClassName(status: string): string {
  if (status === 'cancelled') {
    return 'border-zinc-300/90 bg-zinc-100/80 text-zinc-900 dark:border-zinc-400/45 dark:bg-zinc-500/12 dark:text-zinc-200'
  }
  if (FAILED_ORDER_STATUSES.has(status)) {
    return 'border-red-300/90 bg-red-100/80 text-red-900 dark:border-red-400/45 dark:bg-red-500/12 dark:text-red-200'
  }
  if (RESOLVED_ORDER_STATUSES.has(status)) {
    return 'border-emerald-300/90 bg-emerald-100/80 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/12 dark:text-emerald-200'
  }
  return 'border-zinc-300/90 bg-zinc-100/80 text-zinc-800 dark:border-zinc-500/45 dark:bg-zinc-500/12 dark:text-zinc-200'
}

function resolveVenueStatusPresentation(providerSnapshotStatus: string): VenuePresentation {
  const key = normalizeStatus(providerSnapshotStatus)
  if (key === 'filled') {
    return {
      label: 'Filled',
      detail: 'Venue reports the order as filled.',
      className: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-400/45 dark:bg-emerald-500/12 dark:text-emerald-200',
    }
  }
  if (key === 'partially_filled') {
    return {
      label: 'Partial',
      detail: 'Venue reports a partial fill.',
      className: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/12 dark:text-sky-200',
    }
  }
  if (key === 'open') {
    return {
      label: 'Working',
      detail: 'Venue order remains working on book.',
      className: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/45 dark:bg-sky-500/12 dark:text-sky-200',
    }
  }
  if (key === 'pending') {
    return {
      label: 'Pending',
      detail: 'Venue has accepted but not yet worked the order.',
      className: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/12 dark:text-amber-200',
    }
  }
  if (key === 'cancelled' || key === 'expired') {
    return {
      label: 'Canceled',
      detail: 'Venue confirms cancellation/expiry.',
      className: 'border-zinc-300 bg-zinc-100 text-zinc-900 dark:border-zinc-400/45 dark:bg-zinc-500/12 dark:text-zinc-200',
    }
  }
  if (key === 'failed' || key === 'rejected') {
    return {
      label: 'Rejected',
      detail: 'Venue reports failed/rejected execution.',
      className: 'border-red-300 bg-red-100 text-red-900 dark:border-red-400/45 dark:bg-red-500/12 dark:text-red-200',
    }
  }
  return {
    label: '—',
    detail: 'No venue status snapshot available.',
    className: 'border-border bg-muted/50 text-muted-foreground',
  }
}

function compactText(value: string, maxChars = 96): string {
  const text = String(value || '').trim()
  if (!text) return 'No reason provided'
  if (text.length <= maxChars) return text
  return `${text.slice(0, Math.max(1, maxChars - 1)).trimEnd()}…`
}

function resolveOutcomePresentation(params: {
  status: string
  pnl: number | null
  reason: string
}): {
  headline: string
  detail: string
  detailCompact: string
} {
  const { status, pnl, reason } = params
  if (FAILED_ORDER_STATUSES.has(status)) {
    const detail = reason || 'Execution did not complete.'
    return {
      headline: 'Execution Failed',
      detail,
      detailCompact: compactText(detail),
    }
  }
  if (RESOLVED_ORDER_STATUSES.has(status)) {
    if ((pnl ?? 0) > 0) {
      const detail = `Realized gain ${formatSignedCurrency(pnl || 0)}.`
      return { headline: 'Profit Locked', detail, detailCompact: compactText(detail) }
    }
    if ((pnl ?? 0) < 0) {
      const detail = `Realized loss ${formatSignedCurrency(pnl || 0)}.`
      return { headline: 'Loss Realized', detail, detailCompact: compactText(detail) }
    }
    const detail = reason || 'Resolved without realized gain or loss.'
    return { headline: 'Resolved Flat', detail, detailCompact: compactText(detail) }
  }
  if (OPEN_ORDER_STATUSES.has(status)) {
    const detail = reason || 'Order is open and actively managed.'
    return { headline: 'Position Open', detail, detailCompact: compactText(detail) }
  }
  const detail = reason || titleCaseStatusLabel(status)
  return {
    headline: titleCaseStatusLabel(status),
    detail,
    detailCompact: compactText(detail),
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function resolveDirectionLabel(order: TraderOrder): string {
  const explicitLabel = String(order.direction_label || '').trim()
  if (explicitLabel) return explicitLabel
  const side = String(order.direction_side || order.direction || '').trim().toUpperCase()
  return side || '—'
}

function resolvePendingExitProgressPercent(payload: Record<string, unknown> | null): number | null {
  if (!payload || !isRecord(payload.pending_exit)) return null
  const pendingExit = payload.pending_exit
  const fillRatio = toFiniteNumber(pendingExit.fill_ratio)
  if (fillRatio !== null && fillRatio >= 0) return clamp(fillRatio * 100, 0, 100)
  const filledSize = toFiniteNumber(pendingExit.filled_size)
  const exitSize = toFiniteNumber(pendingExit.exit_size)
  if (filledSize !== null && exitSize !== null && exitSize > 0) {
    return clamp((Math.max(0, filledSize) / exitSize) * 100, 0, 100)
  }
  return null
}

function computeDynamicEdgePercent(params: {
  status: string
  fillPx: number
  markPx: number
  pnl: number | null
  cost: number
  edgePercent: number
}): number {
  const {
    status,
    fillPx,
    markPx,
    pnl,
    cost,
    edgePercent,
  } = params
  if (OPEN_ORDER_STATUSES.has(status) && fillPx > 0 && markPx > 0) {
    return ((markPx - fillPx) / fillPx) * 100
  }
  if (RESOLVED_ORDER_STATUSES.has(status) && cost > 0 && pnl !== null) {
    return (pnl / cost) * 100
  }
  return edgePercent
}

export default function PerformancePanel() {
  const [viewMode, setViewMode] = useState<ViewMode>('all')
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('30d')
  const [tradeTapePage, setTradeTapePage] = useState(1)

  const { data: accounts = [], isLoading: accountsLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  const simulationAccountKey = useMemo(
    () => accounts.map((account) => account.id).sort().join('|'),
    [accounts]
  )

  const { data: orchestratorStats } = useQuery({
    queryKey: ['trader-orchestrator-stats'],
    queryFn: getTraderOrchestratorStats,
    enabled: viewMode === 'live' || viewMode === 'all',
  })

  const {
    data: simulationTrades = [],
    isLoading: simTradesLoading,
    refetch: refetchSimTrades,
  } = useQuery({
    queryKey: ['all-simulation-trades', selectedAccount, simulationAccountKey],
    queryFn: async () => {
      if (selectedAccount) {
        const rows = await getAccountTrades(selectedAccount, 250)
        const accountName = accounts.find((account) => account.id === selectedAccount)?.name || 'Selected account'
        return rows.map((row) => ({ ...row, accountName }))
      }

      const responses = await Promise.all(
        accounts.map(async (account) => {
          const rows = await getAccountTrades(account.id, 250)
          return rows.map((row: SimulationTrade) => ({
            ...row,
            accountName: account.name,
          }))
        })
      )

      return responses
        .flat()
        .sort((left, right) => new Date(right.executed_at).getTime() - new Date(left.executed_at).getTime())
    },
    enabled: accounts.length > 0 && (viewMode === 'simulation' || viewMode === 'all'),
  })

  const {
    data: autoTrades = [],
    isLoading: autoTradesLoading,
    refetch: refetchAutoTrades,
  } = useQuery<LiveTraderOrder[]>({
    queryKey: ['trader-orders'],
    queryFn: async () => {
      const rows = await getAllTraderOrders(250)
      const normalizedRows: LiveTraderOrder[] = []
      rows.forEach((row) => {
        if (String(row.mode || '').toLowerCase() !== 'live') return

        const executedAt = row.executed_at || row.created_at
        if (!executedAt) return

        normalizedRows.push({
          ...row,
          executed_at: executedAt,
          total_cost: Number(row.notional_usd || 0),
          strategy: String(row.source || 'unknown'),
        })
      })
      return normalizedRows
    },
    enabled: viewMode === 'live' || viewMode === 'all',
  })

  const isLoading = accountsLoading || simTradesLoading || autoTradesLoading

  const filterByTimeRange = <T extends { executed_at: string }>(rows: T[]): T[] => {
    if (timeRange === 'all') return rows
    const now = Date.now()
    const daysByRange: Record<Exclude<TimeRange, 'all'>, number> = {
      '7d': 7,
      '30d': 30,
      '90d': 90,
    }
    const cutoff = now - daysByRange[timeRange] * 24 * 60 * 60 * 1000
    return rows.filter((row) => {
      const ts = new Date(row.executed_at).getTime()
      return Number.isFinite(ts) && ts >= cutoff
    })
  }

  const filteredSimTrades = useMemo(
    () => filterByTimeRange(simulationTrades),
    [simulationTrades, timeRange]
  )

  const filteredAutoTrades = useMemo(
    () => filterByTimeRange(autoTrades),
    [autoTrades, timeRange]
  )

  const unifiedTrades = useMemo(() => {
    const rows: UnifiedTrade[] = []

    if (viewMode === 'simulation' || viewMode === 'all') {
      filteredSimTrades.forEach((trade) => {
        const status = normalizeStatus(trade.status)
        const pnl = typeof trade.actual_pnl === 'number' ? trade.actual_pnl : null
        const reason = `Sandbox execution · ${trade.strategy_type || 'unknown'}`
        const outcome = resolveOutcomePresentation({ status, pnl, reason })
        const cost = Number(trade.total_cost || 0)
        const isResolved = RESOLVED_ORDER_STATUSES.has(status)
        rows.push({
          id: `sim-${trade.id}`,
          source: 'sandbox',
          strategy: trade.strategy_type || 'unknown',
          marketQuestion: trade.opportunity_id || trade.strategy_type || 'sandbox position',
          marketId: trade.opportunity_id,
          directionLabel: '—',
          lifecycleLabel: resolveOrderLifecycleLabel(status),
          outcomeHeadline: outcome.headline,
          outcomeDetail: outcome.detail,
          outcomeDetailCompact: outcome.detailCompact,
          modeLabel: 'PAPER',
          venuePresentation: {
            label: 'Sandbox',
            detail: 'Paper simulation fill.',
            className: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/12 dark:text-amber-200',
          },
          fillPx: 0,
          fillProgressPercent: null,
          markPx: 0,
          unrealized: 0,
          dynamicEdgePercent: pnl !== null && cost > 0 ? (pnl / cost) * 100 : 0,
          exitProgressPercent: null,
          createdAt: trade.executed_at,
          updatedAt: trade.resolved_at || trade.executed_at,
          cost,
          pnl,
          status,
          executedAt: trade.executed_at,
          accountName: (trade as SimulationTrade & { accountName?: string }).accountName,
          isResolved,
          isWin: WIN_ORDER_STATUSES.has(status) || (isResolved && (pnl ?? 0) > 0),
          isLoss: LOSS_ORDER_STATUSES.has(status) || (isResolved && (pnl ?? 0) < 0),
        })
      })
    }

    if (viewMode === 'live' || viewMode === 'all') {
      filteredAutoTrades.forEach((trade) => {
        const status = normalizeStatus(trade.status)
        const pnl = typeof trade.actual_profit === 'number' ? trade.actual_profit : null
        const cost = Number(trade.total_cost || 0)
        const fillPx = toFiniteNumber(trade.average_fill_price ?? trade.effective_price ?? trade.entry_price) ?? 0
        const markPx = toFiniteNumber(trade.current_price) ?? 0
        const filledNotional = toFiniteNumber(trade.filled_notional_usd) ?? 0
        const fillProgressPercent = cost > 0 && filledNotional > 0
          ? clamp((filledNotional / cost) * 100, 0, 100)
          : null
        const unrealized = toFiniteNumber(trade.unrealized_pnl) ?? 0
        const edgePercent = normalizeEdgePercent(toFiniteNumber(trade.edge_percent) ?? 0)
        const payload = isRecord(trade.payload) ? trade.payload : null
        const exitProgressPercent = resolvePendingExitProgressPercent(payload)
        const reason = String(trade.close_reason || trade.reason || trade.error_message || '').trim()
        const outcome = resolveOutcomePresentation({ status, pnl, reason })
        const isResolved = RESOLVED_ORDER_STATUSES.has(status)
        rows.push({
          id: `live-${trade.id}`,
          source: 'live',
          strategy: String(trade.strategy || 'unknown'),
          marketQuestion: String(trade.market_question || trade.market_id || 'Unknown market'),
          marketId: String(trade.market_id || ''),
          directionLabel: resolveDirectionLabel(trade),
          lifecycleLabel: resolveOrderLifecycleLabel(status),
          outcomeHeadline: outcome.headline,
          outcomeDetail: outcome.detail,
          outcomeDetailCompact: outcome.detailCompact,
          modeLabel: String(trade.mode || '').trim().toUpperCase() || 'LIVE',
          venuePresentation: resolveVenueStatusPresentation(String(trade.provider_snapshot_status || '')),
          fillPx,
          fillProgressPercent,
          markPx,
          unrealized,
          dynamicEdgePercent: computeDynamicEdgePercent({
            status,
            fillPx,
            markPx,
            pnl,
            cost,
            edgePercent,
          }),
          exitProgressPercent,
          providerSnapshotStatus: normalizeStatus(trade.provider_snapshot_status),
          createdAt: trade.created_at || trade.executed_at,
          updatedAt: trade.updated_at || trade.mark_updated_at || trade.executed_at,
          cost,
          pnl,
          status,
          executedAt: trade.executed_at,
          isResolved,
          isWin: WIN_ORDER_STATUSES.has(status) || (isResolved && (pnl ?? 0) > 0),
          isLoss: LOSS_ORDER_STATUSES.has(status) || (isResolved && (pnl ?? 0) < 0),
        })
      })
    }

    rows.sort((left, right) => new Date(right.executedAt).getTime() - new Date(left.executedAt).getTime())
    return rows
  }, [filteredAutoTrades, filteredSimTrades, viewMode])

  const tradeTapePageCount = useMemo(
    () => Math.max(1, Math.ceil(unifiedTrades.length / TRADE_TAPE_PAGE_SIZE)),
    [unifiedTrades.length]
  )

  useEffect(() => {
    setTradeTapePage((current) => Math.min(current, tradeTapePageCount))
  }, [tradeTapePageCount])

  const pagedUnifiedTrades = useMemo(() => {
    const start = (tradeTapePage - 1) * TRADE_TAPE_PAGE_SIZE
    return unifiedTrades.slice(start, start + TRADE_TAPE_PAGE_SIZE)
  }, [tradeTapePage, unifiedTrades])

  const summary = useMemo(() => {
    const resolved = unifiedTrades.filter((trade) => trade.isResolved)
    const wins = resolved.filter((trade) => trade.isWin)
    const losses = resolved.filter((trade) => trade.isLoss)

    const totalPnl = unifiedTrades.reduce((sum, trade) => sum + (trade.pnl ?? 0), 0)
    const totalCost = unifiedTrades.reduce((sum, trade) => sum + trade.cost, 0)
    const openTrades = unifiedTrades.filter((trade) => !trade.isResolved).length
    const winRate = resolved.length > 0 ? (wins.length / resolved.length) * 100 : 0
    const roi = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0
    const avgPnl = unifiedTrades.length > 0 ? totalPnl / unifiedTrades.length : 0

    const grossWins = wins.reduce((sum, trade) => sum + Math.max(0, trade.pnl ?? 0), 0)
    const grossLosses = losses.reduce((sum, trade) => sum + Math.abs(Math.min(0, trade.pnl ?? 0)), 0)
    const profitFactor = grossLosses > 0
      ? grossWins / grossLosses
      : grossWins > 0
        ? Infinity
        : 0

    return {
      totalTrades: unifiedTrades.length,
      resolvedTrades: resolved.length,
      openTrades,
      wins: wins.length,
      losses: losses.length,
      totalPnl,
      totalCost,
      winRate,
      roi,
      avgPnl,
      profitFactor,
    }
  }, [unifiedTrades])

  const strategyLeaderboard = useMemo(() => {
    const byStrategy = new Map<string, StrategyRollup>()

    unifiedTrades.forEach((trade) => {
      const key = trade.strategy || 'unknown'
      const current = byStrategy.get(key) || {
        strategy: key,
        trades: 0,
        wins: 0,
        losses: 0,
        pnl: 0,
        sandboxTrades: 0,
        liveTrades: 0,
      }

      current.trades += 1
      current.pnl += trade.pnl ?? 0
      current.wins += trade.isWin ? 1 : 0
      current.losses += trade.isLoss ? 1 : 0
      current.sandboxTrades += trade.source === 'sandbox' ? 1 : 0
      current.liveTrades += trade.source === 'live' ? 1 : 0

      byStrategy.set(key, current)
    })

    return Array.from(byStrategy.values()).sort((left, right) => {
      if (left.pnl === right.pnl) return right.trades - left.trades
      return right.pnl - left.pnl
    })
  }, [unifiedTrades])

  const cumulativePnlData = useMemo(() => {
    const daily = new Map<string, { sim: number; live: number }>()

    filteredSimTrades.forEach((trade) => {
      const day = trade.executed_at?.split('T')[0]
      if (!day) return
      const current = daily.get(day) || { sim: 0, live: 0 }
      current.sim += trade.actual_pnl || 0
      daily.set(day, current)
    })

    filteredAutoTrades.forEach((trade) => {
      const day = trade.executed_at?.split('T')[0]
      if (!day) return
      const current = daily.get(day) || { sim: 0, live: 0 }
      current.live += trade.actual_profit || 0
      daily.set(day, current)
    })

    const sortedDays = Array.from(daily.keys()).sort()
    let cumSim = 0
    let cumLive = 0

    return sortedDays.map((day) => {
      const row = daily.get(day) || { sim: 0, live: 0 }
      cumSim += row.sim
      cumLive += row.live
      return {
        date: day,
        dailySimPnl: row.sim,
        dailyLivePnl: row.live,
        cumSimPnl: cumSim,
        cumLivePnl: cumLive,
        cumTotalPnl: cumSim + cumLive,
      }
    })
  }, [filteredAutoTrades, filteredSimTrades])

  const maxDrawdown = useMemo(() => {
    if (cumulativePnlData.length === 0) return 0

    let peak = Number.NEGATIVE_INFINITY
    let drawdown = 0

    cumulativePnlData.forEach((point) => {
      const value = viewMode === 'simulation'
        ? point.cumSimPnl
        : viewMode === 'live'
          ? point.cumLivePnl
          : point.cumTotalPnl
      if (value > peak) peak = value
      drawdown = Math.max(drawdown, peak - value)
    })

    return drawdown
  }, [cumulativePnlData, viewMode])

  const handleRefresh = () => {
    if (viewMode === 'simulation' || viewMode === 'all') {
      void refetchSimTrades()
    }
    if (viewMode === 'live' || viewMode === 'all') {
      void refetchAutoTrades()
    }
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-1.5">
      {/* Control Strip */}
      <div className="shrink-0 space-y-2">
        {/* Row 1: Title + refresh */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-300" />
              Performance
            </h2>
            <Badge variant="outline" className="text-[10px] border-border/50">
              {summary.totalTrades} trades
            </Badge>
          </div>

          <Button variant="ghost" size="sm" onClick={handleRefresh} disabled={isLoading} className="h-7 px-2">
            <RefreshCw className={cn('w-3.5 h-3.5', isLoading && 'animate-spin')} />
          </Button>
        </div>

        {/* Row 2: Scope + time range + account */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1">
            {VIEW_MODE_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setViewMode(option.id)}
                className={cn(
                  'h-7 rounded-md border px-2.5 text-xs transition-colors',
                  viewMode === option.id
                    ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-100'
                    : 'border-border bg-background/60 text-muted-foreground hover:text-foreground'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="w-px h-5 bg-border/50" />

          <div className="flex items-center gap-1">
            {RANGE_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setTimeRange(option.id)}
                className={cn(
                  'h-7 rounded-md border px-2.5 text-xs transition-colors',
                  timeRange === option.id
                    ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-100'
                    : 'border-border bg-background/60 text-muted-foreground hover:text-foreground'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
            <>
              <div className="w-px h-5 bg-border/50" />
              <select
                value={selectedAccount || ''}
                onChange={(event) => setSelectedAccount(event.target.value || null)}
                className="h-7 rounded-md border border-border bg-background/80 px-2 text-xs"
              >
                <option value="">All accounts</option>
                {accounts.map((account: SimulationAccount) => (
                  <option key={account.id} value={account.id}>{account.name}</option>
                ))}
              </select>
            </>
          )}
        </div>
      </div>

      {/* Metric Strip */}
      <div className="shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 border-y border-border/50 py-1.5 px-0.5">
        <MetricChip
          label="P&L"
          value={formatSignedCurrency(summary.totalPnl, true)}
          detail={formatCurrency(summary.totalPnl)}
          icon={summary.totalPnl >= 0 ? TrendingUp : TrendingDown}
          valueClassName={summary.totalPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
        />
        <MetricChip
          label="ROI"
          value={formatPercent(summary.roi)}
          detail={`on ${formatCurrency(summary.totalCost, true)}`}
          icon={Target}
          valueClassName={summary.roi >= 0 ? 'text-emerald-300' : 'text-red-300'}
        />
        <MetricChip
          label="Win Rate"
          value={formatPercent(summary.winRate)}
          detail={`${summary.wins}W / ${summary.losses}L`}
          icon={Activity}
          valueClassName={summary.winRate >= 50 ? 'text-emerald-300' : 'text-amber-300'}
        />
        <MetricChip
          label="Open"
          value={String(summary.openTrades)}
          detail={`${summary.resolvedTrades} closed`}
          icon={Calendar}
        />
        <MetricChip
          label="Avg P&L"
          value={formatSignedCurrency(summary.avgPnl)}
          icon={summary.avgPnl >= 0 ? ArrowUpRight : ArrowDownRight}
          valueClassName={summary.avgPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
        />
        <MetricChip
          label="Drawdown"
          value={formatCurrency(maxDrawdown, true)}
          detail={Number.isFinite(summary.profitFactor) ? `PF ${summary.profitFactor === Infinity ? '∞' : summary.profitFactor.toFixed(2)}` : ''}
          icon={TrendingDown}
          valueClassName={maxDrawdown > 0 ? 'text-amber-300' : undefined}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-0 flex flex-col gap-2">
        {/* Chart + Leaderboard */}
        <div className="shrink-0 h-[260px] grid gap-2 xl:grid-cols-12">
          <div className="xl:col-span-8 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col">
            <div className="shrink-0 flex items-center justify-between gap-2 mb-2">
              <p className="text-xs font-semibold flex items-center gap-1.5">
                <BarChart3 className="h-3.5 w-3.5 text-cyan-300" />
                Cumulative P&L
              </p>
              {orchestratorStats?.last_trade_at && (
                <span className="text-[10px] text-muted-foreground">
                  last trade: {new Date(orchestratorStats.last_trade_at).toLocaleString()}
                </span>
              )}
            </div>
            <div className="flex-1 min-h-0">
              {cumulativePnlData.length === 0 ? (
                <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                  No trade history in range.
                </div>
              ) : (
                <PerformancePnlChart data={cumulativePnlData} viewMode={viewMode} />
              )}
            </div>
          </div>

          <div className="xl:col-span-4 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col">
            <p className="shrink-0 text-xs font-semibold mb-2">Strategy Leaderboard</p>
            <ScrollArea className="flex-1 min-h-0">
              <div className="space-y-1.5 pr-2">
                {strategyLeaderboard.length === 0 ? (
                  <div className="text-xs text-muted-foreground text-center py-4">No strategy activity.</div>
                ) : (
                  strategyLeaderboard.slice(0, 12).map((row) => {
                    const winRate = row.wins + row.losses > 0 ? (row.wins / (row.wins + row.losses)) * 100 : 0
                    return (
                      <div key={row.strategy} className="rounded border border-border/50 bg-background/30 px-2 py-1.5">
                        <div className="flex items-center justify-between gap-1">
                          <p className="truncate text-xs">{row.strategy}</p>
                          <p className={cn('text-[11px] font-mono', row.pnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                            {formatSignedCurrency(row.pnl, true)}
                          </p>
                        </div>
                        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground mt-0.5">
                          <span>{row.trades}t</span>
                          <span>{formatPercent(winRate)}</span>
                          <span>{row.liveTrades}L/{row.sandboxTrades}S</span>
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Trade Tape — fills remaining space */}
        <div className="flex-1 min-h-0 rounded-md border border-border/60 bg-card/80 flex flex-col">
          <div className="shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-b border-border/40">
            <p className="text-xs font-semibold">Trade Tape</p>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-muted-foreground">
                {unifiedTrades.length} trades • {TRADE_TAPE_PAGE_SIZE}/page
              </span>
              <Button
                size="sm"
                variant="outline"
                className="h-5 px-1.5 text-[10px]"
                onClick={() => setTradeTapePage((page) => Math.max(1, page - 1))}
                disabled={tradeTapePage <= 1}
              >
                Prev
              </Button>
              <span className="min-w-[62px] text-center text-[10px] font-mono text-muted-foreground">
                {tradeTapePage}/{tradeTapePageCount}
              </span>
              <Button
                size="sm"
                variant="outline"
                className="h-5 px-1.5 text-[10px]"
                onClick={() => setTradeTapePage((page) => Math.min(tradeTapePageCount, page + 1))}
                disabled={tradeTapePage >= tradeTapePageCount}
              >
                Next
              </Button>
            </div>
          </div>
          <ScrollArea className="flex-1 min-h-0">
            <div className="w-full overflow-x-auto">
              <Table className="w-full table-fixed">
                <TableHeader>
                  <TableRow className="sticky top-0 z-10 bg-card/95 backdrop-blur-sm">
                    <TableHead className="w-[26%] text-[10px]">Market</TableHead>
                    <TableHead className="w-[6%] text-[10px]">Dir</TableHead>
                    <TableHead className="w-[24%] text-[10px]">Lifecycle / Outcome</TableHead>
                    <TableHead className="w-[8%] text-[10px] text-right">Notional</TableHead>
                    <TableHead className="w-[6%] text-[10px] text-right">Fill</TableHead>
                    <TableHead className="w-[6%] text-[10px] text-right">Fill Progress</TableHead>
                    <TableHead className="w-[6%] text-[10px] text-right">Mark</TableHead>
                    <TableHead className="w-[8%] text-[10px] text-right">U-P&amp;L</TableHead>
                    <TableHead className="w-[7%] text-[10px] text-right">Edge Δ</TableHead>
                    <TableHead className="w-[8%] text-[10px] text-right">R-P&amp;L</TableHead>
                    <TableHead className="w-[8%] text-[10px]">Venue</TableHead>
                    <TableHead className="w-[6%] text-[10px] text-right">Exit %</TableHead>
                    <TableHead className="w-[5%] text-[10px]">Age</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pagedUnifiedTrades.map((trade) => {
                    const statusBadge = resolveOrderStatusBadgePresentation(trade.status, trade.pnl ?? 0)
                    const outcomeBadgeClassName = resolveOrderOutcomeBadgeClassName(trade.status)
                    const updatedAt = trade.updatedAt || trade.executedAt
                    const venueTitle = `${trade.venuePresentation.detail}`
                      + (trade.providerSnapshotStatus ? ` • provider:${trade.providerSnapshotStatus}` : '')
                    return (
                      <TableRow key={trade.id} className="text-[11px] leading-tight hover:bg-muted/30">
                        <TableCell className="max-w-[260px] py-0.5">
                          <div className="flex min-w-0 items-center gap-1">
                            <Badge
                              variant="outline"
                              className={cn(
                                'h-4 px-1 text-[9px] font-semibold',
                                trade.source === 'sandbox'
                                  ? 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/45 dark:bg-amber-500/12 dark:text-amber-200'
                                  : 'border-cyan-300 bg-cyan-100 text-cyan-900 dark:border-cyan-400/45 dark:bg-cyan-500/12 dark:text-cyan-200'
                              )}
                            >
                              {trade.source === 'sandbox' ? 'S' : 'L'}
                            </Badge>
                            <span className="truncate" title={trade.marketQuestion}>{trade.marketQuestion}</span>
                          </div>
                          <p className="truncate text-[9px] leading-none text-muted-foreground" title={`${trade.strategy}${trade.accountName ? ` • ${trade.accountName}` : ''}`}>
                            {trade.strategy}{trade.accountName ? ` • ${trade.accountName}` : ''}
                          </p>
                        </TableCell>
                        <TableCell className="py-0.5">
                          <Badge
                            variant="outline"
                            className="h-4 max-w-[120px] truncate border-border/80 bg-muted/60 px-1 text-[9px] text-muted-foreground"
                            title={trade.directionLabel}
                          >
                            {trade.directionLabel}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-0.5 min-w-[260px]">
                          <div className="min-w-0 space-y-0">
                            <div className="flex min-w-0 items-center gap-0.5 overflow-hidden">
                              <Badge
                                variant={statusBadge.variant}
                                title={`Raw status: ${trade.status}`}
                                className={cn('h-4 shrink-0 whitespace-nowrap px-1 text-[9px] font-semibold', statusBadge.className)}
                              >
                                {trade.lifecycleLabel}
                              </Badge>
                              <Badge
                                variant="outline"
                                title={trade.outcomeDetail}
                                className={cn('h-4 max-w-[180px] truncate px-1 text-[9px] font-medium', outcomeBadgeClassName)}
                              >
                                {trade.outcomeHeadline}
                              </Badge>
                              <Badge variant="outline" className="h-4 max-w-[100px] truncate border-border/80 bg-background/80 px-1 text-[9px] font-medium text-foreground/85">
                                {trade.modeLabel}
                              </Badge>
                            </div>
                            <p className="truncate text-[9px] leading-none text-foreground/85" title={trade.outcomeDetail}>
                              <span className="font-medium text-muted-foreground">Reason:</span> {trade.outcomeDetailCompact}
                            </p>
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatCurrency(trade.cost, true)}</TableCell>
                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{trade.fillPx > 0 ? trade.fillPx.toFixed(3) : '—'}</TableCell>
                        <TableCell className="text-right font-mono py-0.5 text-[10px]">
                          {trade.fillProgressPercent !== null ? formatPercent(trade.fillProgressPercent, 0) : '—'}
                        </TableCell>
                        <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', trade.markPx > 0 && 'text-sky-300')}>
                          {trade.markPx > 0 ? trade.markPx.toFixed(3) : '—'}
                        </TableCell>
                        <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', trade.unrealized > 0 ? 'text-emerald-500' : trade.unrealized < 0 ? 'text-red-500' : '')}>
                          {OPEN_ORDER_STATUSES.has(trade.status) ? formatCurrency(trade.unrealized, true) : '—'}
                        </TableCell>
                        <TableCell className="text-right font-mono py-0.5 text-[10px]">{formatPercent(trade.dynamicEdgePercent)}</TableCell>
                        <TableCell className={cn('text-right font-mono py-0.5 text-[10px]', (trade.pnl ?? 0) > 0 ? 'text-emerald-500' : (trade.pnl ?? 0) < 0 ? 'text-red-500' : '')}>
                          {RESOLVED_ORDER_STATUSES.has(trade.status) && trade.pnl !== null ? formatSignedCurrency(trade.pnl) : '—'}
                        </TableCell>
                        <TableCell className="py-0.5">
                          <Badge
                            variant="outline"
                            title={venueTitle}
                            className={cn('h-4 max-w-[120px] truncate px-1 text-[9px] font-semibold', trade.venuePresentation.className)}
                          >
                            {trade.venuePresentation.label}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right font-mono py-0.5 text-[10px]">
                          {trade.exitProgressPercent !== null ? formatPercent(trade.exitProgressPercent, 0) : '—'}
                        </TableCell>
                        <TableCell className="py-0.5 text-[9px] text-muted-foreground">
                          <span title={`${trade.modeLabel} • created:${trade.createdAt ? new Date(trade.createdAt).toLocaleString() : 'n/a'} • updated:${trade.updatedAt ? new Date(trade.updatedAt).toLocaleString() : 'n/a'}`}>
                            {formatRelativeAge(updatedAt)}
                          </span>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                  {unifiedTrades.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={13} className="py-6 text-center text-xs text-muted-foreground">
                        No trades for this view and range.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  )
}

function MetricChip({
  label,
  value,
  detail,
  icon: Icon,
  valueClassName,
}: {
  label: string
  value: string
  detail?: string
  icon: React.ElementType
  valueClassName?: string
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs" title={detail || undefined}>
      <Icon className="w-3.5 h-3.5 opacity-70" />
      <span className="text-muted-foreground">{label}</span>
      <span className={cn('font-mono font-semibold', valueClassName)}>{value}</span>
    </div>
  )
}

function PerformancePnlChart({
  data,
  viewMode,
}: {
  data: Array<{
    date: string
    cumSimPnl: number
    cumLivePnl: number
    cumTotalPnl: number
  }>
  viewMode: ViewMode
}) {
  const showSandbox = viewMode === 'simulation' || viewMode === 'all'
  const showLive = viewMode === 'live' || viewMode === 'all'

  const tooltipFormatter = (value: number, key: string) => {
    const label = key === 'cumSimPnl'
      ? 'Sandbox cumulative'
      : key === 'cumLivePnl'
        ? 'Live cumulative'
        : 'Unified cumulative'
    return [formatCurrency(value), label]
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <defs>
          <linearGradient id="sandboxGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.32} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.04} />
          </linearGradient>
          <linearGradient id="liveGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.32} />
            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.04} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="hsl(var(--border) / 0.45)" strokeDasharray="3 3" />

        <XAxis
          dataKey="date"
          tickFormatter={formatDateLabel}
          tick={{ fontSize: 11 }}
          stroke="hsl(var(--muted-foreground))"
          axisLine={{ stroke: 'hsl(var(--border))' }}
          tickLine={{ stroke: 'hsl(var(--border))' }}
        />
        <YAxis
          tickFormatter={(value: number) => formatCurrency(value, true)}
          tick={{ fontSize: 11 }}
          width={72}
          stroke="hsl(var(--muted-foreground))"
          axisLine={{ stroke: 'hsl(var(--border))' }}
          tickLine={{ stroke: 'hsl(var(--border))' }}
        />

        <Tooltip
          formatter={(value, key) => tooltipFormatter(Number(value), String(key))}
          labelFormatter={(label) => `Date ${label}`}
          contentStyle={{
            borderRadius: 10,
            border: '1px solid hsl(var(--border))',
            background: 'hsl(var(--popover))',
            color: 'hsl(var(--popover-foreground))',
            fontSize: 12,
          }}
        />

        <Legend
          wrapperStyle={{ fontSize: '11px' }}
          formatter={(value) => {
            if (value === 'cumSimPnl') return 'Sandbox'
            if (value === 'cumLivePnl') return 'Live'
            return 'Unified'
          }}
        />

        {showSandbox && (
          <Area
            type="monotone"
            dataKey="cumSimPnl"
            stroke="#f59e0b"
            strokeWidth={2}
            fill="url(#sandboxGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#f59e0b', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}

        {showLive && (
          <Area
            type="monotone"
            dataKey="cumLivePnl"
            stroke="#22d3ee"
            strokeWidth={2}
            fill="url(#liveGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#22d3ee', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}

        {viewMode === 'all' && (
          <Line
            type="monotone"
            dataKey="cumTotalPnl"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            strokeDasharray="5 3"
            activeDot={{ r: 4, stroke: '#34d399', strokeWidth: 2, fill: 'hsl(var(--background))' }}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}
