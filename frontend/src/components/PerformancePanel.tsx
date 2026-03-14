import { type ElementType, useDeferredValue, useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  BarChart3,
  Calendar,
  Clock3,
  RefreshCw,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { cn } from '../lib/utils'
import {
  getAccountTrades,
  getLiveWalletPerformance,
  getSimulationAccounts,
  getTraderOrchestratorStats,
  type LiveWalletOpenLot,
  type LiveWalletRoundTrip,
  type SimulationAccount,
  type SimulationTrade,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

type ViewMode = 'simulation' | 'live'
type DetailTab = 'overview' | 'insights' | 'history'
type TimeRange = '7d' | '30d' | '90d' | 'all'
type HistorySortKey = 'closedAt' | 'marketQuestion' | 'category' | 'pnl' | 'roiPercent' | 'holdMinutes' | 'buyNotional' | 'sellNotional'
type HistorySortDirection = 'asc' | 'desc'
type HistoryOutcomeFilter = 'all' | 'wins' | 'losses' | 'breakeven'

type SimulationTradeWithAccount = SimulationTrade & {
  accountName?: string
}

type UnifiedTradeRow = {
  id: string
  source: 'sandbox' | 'live'
  strategy: string
  marketQuestion: string
  marketId: string
  outcome: string
  category: string
  quantity: number | null
  buyPrice: number | null
  sellPrice: number | null
  buyNotional: number
  sellNotional: number
  pnl: number | null
  roiPercent: number
  holdMinutes: number | null
  status: string
  openedAt: string
  closedAt: string
  accountName?: string
  isResolved: boolean
  isWin: boolean
  isLoss: boolean
}

type RollupRow = {
  key: string
  label: string
  trades: number
  wins: number
  losses: number
  pnl: number
  buyNotional: number
  roiSum: number
}

type PnlPoint = {
  date: string
  cumulativePnl: number
}

const VIEW_MODE_OPTIONS: Array<{ id: ViewMode; label: string }> = [
  { id: 'live', label: 'Live' },
  { id: 'simulation', label: 'Sandbox' },
]

const RANGE_OPTIONS: Array<{ id: TimeRange; label: string }> = [
  { id: '7d', label: '7D' },
  { id: '30d', label: '30D' },
  { id: '90d', label: '90D' },
  { id: 'all', label: 'All Time' },
]

const TRADE_TAPE_PAGE_SIZE = 100

const HISTORY_SORT_OPTIONS: Array<{ id: HistorySortKey; label: string }> = [
  { id: 'closedAt', label: 'Closed Time' },
  { id: 'pnl', label: 'P&L' },
  { id: 'roiPercent', label: 'ROI' },
  { id: 'holdMinutes', label: 'Hold Duration' },
  { id: 'buyNotional', label: 'Buy Notional' },
  { id: 'sellNotional', label: 'Sell Notional' },
  { id: 'marketQuestion', label: 'Market' },
  { id: 'category', label: 'Category' },
]

const SIM_RESOLVED_STATUSES = new Set(['resolved_win', 'resolved_loss', 'closed_win', 'closed_loss', 'win', 'loss'])
const SIM_WIN_STATUSES = new Set(['resolved_win', 'closed_win', 'win'])
const SIM_LOSS_STATUSES = new Set(['resolved_loss', 'closed_loss', 'loss'])

const DAY_ORDER = [1, 2, 3, 4, 5, 6, 0]
const DAY_LABEL_BY_INDEX: Record<number, string> = {
  0: 'Sun',
  1: 'Mon',
  2: 'Tue',
  3: 'Wed',
  4: 'Thu',
  5: 'Fri',
  6: 'Sat',
}

const TIME_BUCKETS: Array<{ key: string; label: string; startHour: number; endHour: number }> = [
  { key: 'h00_03', label: '00-03', startHour: 0, endHour: 4 },
  { key: 'h04_07', label: '04-07', startHour: 4, endHour: 8 },
  { key: 'h08_11', label: '08-11', startHour: 8, endHour: 12 },
  { key: 'h12_15', label: '12-15', startHour: 12, endHour: 16 },
  { key: 'h16_19', label: '16-19', startHour: 16, endHour: 20 },
  { key: 'h20_23', label: '20-23', startHour: 20, endHour: 24 },
]

function getSimulationTradeFetchLimit(range: TimeRange): number {
  if (range === '7d') return 150
  if (range === '30d') return 250
  if (range === '90d') return 400
  return 500
}

function getLivePerformanceFetchLimit(range: TimeRange): number {
  if (range === '7d') return 200
  if (range === '30d') return 500
  if (range === '90d') return 900
  return 1500
}

function normalizeStatus(value: unknown): string {
  return String(value || 'unknown').trim().toLowerCase()
}

function toTs(value: string | null | undefined): number {
  if (!value) return 0
  const parsed = new Date(value).getTime()
  return Number.isFinite(parsed) ? parsed : 0
}

function inTimeRange(value: string | null | undefined, range: TimeRange): boolean {
  if (!value) return false
  if (range === 'all') return true
  const ts = toTs(value)
  if (ts <= 0) return false
  const now = Date.now()
  const dayCount = range === '7d' ? 7 : range === '30d' ? 30 : 90
  const cutoff = now - dayCount * 24 * 60 * 60 * 1000
  return ts >= cutoff
}

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

function formatSignedPercent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '0.0%'
  const prefix = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${prefix}${Math.abs(value).toFixed(digits)}%`
}

function formatDateLabel(value: string): string {
  const ts = toTs(value)
  if (ts <= 0) return '--'
  return new Date(ts).toLocaleDateString(undefined, {
    month: 'short',
    day: '2-digit',
  })
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

function formatDurationMinutes(minutes: number | null): string {
  if (minutes === null || !Number.isFinite(minutes) || minutes < 0) return '—'
  if (minutes < 60) return `${minutes.toFixed(0)}m`
  const hours = minutes / 60
  if (hours < 24) return `${hours.toFixed(1)}h`
  return `${(hours / 24).toFixed(1)}d`
}

function formatPrice(price: number | null): string {
  if (price === null || !Number.isFinite(price)) return '—'
  return price.toFixed(3)
}

function formatQuantity(size: number | null): string {
  if (size === null || !Number.isFinite(size)) return '—'
  return size.toFixed(2)
}

function normalizeCategory(value: string | null | undefined): string {
  const text = String(value || '').trim()
  return text || 'Uncategorized'
}

function shortAddress(value: string | null | undefined): string {
  const text = String(value || '').trim()
  if (!text) return '--'
  if (text.length <= 14) return text
  return `${text.slice(0, 8)}...${text.slice(-6)}`
}

function toRoundTripPnl(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function toRoundTripNotional(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
}

function aggregateRollup(rows: UnifiedTradeRow[], keySelector: (row: UnifiedTradeRow) => string): RollupRow[] {
  const byKey = new Map<string, RollupRow>()

  rows.forEach((row) => {
    if (!row.isResolved || row.pnl === null) return
    const key = keySelector(row)
    const current = byKey.get(key) || {
      key,
      label: key,
      trades: 0,
      wins: 0,
      losses: 0,
      pnl: 0,
      buyNotional: 0,
      roiSum: 0,
    }

    current.trades += 1
    current.pnl += row.pnl
    current.buyNotional += row.buyNotional
    current.roiSum += row.roiPercent
    if (row.isWin) current.wins += 1
    if (row.isLoss) current.losses += 1

    byKey.set(key, current)
  })

  return Array.from(byKey.values()).sort((left, right) => {
    if (left.pnl === right.pnl) return right.trades - left.trades
    return right.pnl - left.pnl
  })
}

export default function PerformancePanel() {
  const [viewMode, setViewMode] = useState<ViewMode>('live')
  const [detailTab, setDetailTab] = useState<DetailTab>('overview')
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('30d')
  const [tradeTapePage, setTradeTapePage] = useState(1)
  const [historyCategoryFilter, setHistoryCategoryFilter] = useState<string>('all')
  const [historyOutcomeFilter, setHistoryOutcomeFilter] = useState<HistoryOutcomeFilter>('all')
  const [historySearch, setHistorySearch] = useState('')
  const [historySortKey, setHistorySortKey] = useState<HistorySortKey>('closedAt')
  const [historySortDirection, setHistorySortDirection] = useState<HistorySortDirection>('desc')
  const deferredHistorySearch = useDeferredValue(historySearch)
  const simulationTradeFetchLimit = useMemo(() => getSimulationTradeFetchLimit(timeRange), [timeRange])
  const livePerformanceFetchLimit = useMemo(() => getLivePerformanceFetchLimit(timeRange), [timeRange])

  const { data: accounts = [], isLoading: accountsLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    enabled: viewMode === 'simulation',
  })

  const simulationAccountKey = useMemo(
    () => accounts.map((account) => account.id).sort().join('|'),
    [accounts]
  )

  const { data: orchestratorStats } = useQuery({
    queryKey: ['trader-orchestrator-stats'],
    queryFn: getTraderOrchestratorStats,
    enabled: viewMode === 'live',
  })

  const {
    data: simulationTrades = [],
    isLoading: simulationTradesLoading,
    refetch: refetchSimulationTrades,
  } = useQuery<SimulationTradeWithAccount[]>({
    queryKey: ['all-simulation-trades', selectedAccount, simulationAccountKey, simulationTradeFetchLimit],
    queryFn: async () => {
      if (selectedAccount) {
        const rows = await getAccountTrades(selectedAccount, simulationTradeFetchLimit)
        const accountName = accounts.find((account) => account.id === selectedAccount)?.name || 'Selected account'
        return rows.map((row) => ({ ...row, accountName }))
      }

      const responses = await Promise.all(
        accounts.map(async (account) => {
          const rows = await getAccountTrades(account.id, simulationTradeFetchLimit)
          return rows.map((row) => ({
            ...row,
            accountName: account.name,
          }))
        })
      )

      return responses
        .flat()
        .sort((left, right) => toTs(right.resolved_at || right.executed_at) - toTs(left.resolved_at || left.executed_at))
    },
    enabled: accounts.length > 0 && viewMode === 'simulation',
  })

  const {
    data: livePerformance,
    isLoading: livePerformanceLoading,
    refetch: refetchLivePerformance,
  } = useQuery({
    queryKey: ['live-wallet-performance', livePerformanceFetchLimit],
    queryFn: () => getLiveWalletPerformance(livePerformanceFetchLimit),
    enabled: viewMode === 'live',
    retry: false,
    refetchInterval: 30000,
    placeholderData: (previousData) => previousData,
  })

  const filteredSimulationTrades = useMemo(
    () => simulationTrades.filter((trade) => inTimeRange(trade.resolved_at || trade.executed_at, timeRange)),
    [simulationTrades, timeRange]
  )

  const filteredLiveRoundTrips = useMemo(
    () => (livePerformance?.round_trips || []).filter((row) => inTimeRange(row.closed_at, timeRange)),
    [livePerformance?.round_trips, timeRange]
  )

  const filteredLiveOpenLots = useMemo(
    () => (livePerformance?.open_lots || []).filter((row) => inTimeRange(row.opened_at, timeRange)),
    [livePerformance?.open_lots, timeRange]
  )

  const unifiedTrades = useMemo(() => {
    const rows: UnifiedTradeRow[] = []

    if (viewMode === 'simulation') {
      filteredSimulationTrades.forEach((trade) => {
        const status = normalizeStatus(trade.status)
        const pnl = typeof trade.actual_pnl === 'number' ? trade.actual_pnl : null
        const buyNotional = Number(trade.total_cost || 0)
        const isResolved = SIM_RESOLVED_STATUSES.has(status)
        const openedAt = trade.executed_at
        const closedAt = trade.resolved_at || trade.executed_at
        const holdMinutes = trade.resolved_at
          ? Math.max(0, (toTs(trade.resolved_at) - toTs(trade.executed_at)) / 60_000)
          : null

        rows.push({
          id: `sim-${trade.id}`,
          source: 'sandbox',
          strategy: trade.strategy_type || 'unknown',
          marketQuestion: trade.opportunity_id || trade.strategy_type || 'sandbox position',
          marketId: trade.opportunity_id || '',
          outcome: 'N/A',
          category: trade.strategy_type || 'Sandbox',
          quantity: null,
          buyPrice: null,
          sellPrice: null,
          buyNotional,
          sellNotional: pnl !== null ? buyNotional + pnl : 0,
          pnl,
          roiPercent: pnl !== null && buyNotional > 0 ? (pnl / buyNotional) * 100 : 0,
          holdMinutes,
          status,
          openedAt,
          closedAt,
          accountName: trade.accountName,
          isResolved,
          isWin: SIM_WIN_STATUSES.has(status) || (isResolved && (pnl ?? 0) > 0),
          isLoss: SIM_LOSS_STATUSES.has(status) || (isResolved && (pnl ?? 0) < 0),
        })
      })
    } else {
      filteredLiveRoundTrips.forEach((trade: LiveWalletRoundTrip) => {
        const pnl = toRoundTripPnl(trade.realized_pnl)
        const buyNotional = toRoundTripNotional(trade.buy_notional)
        rows.push({
          id: `live-${trade.id}`,
          source: 'live',
          strategy: 'wallet_roundtrip',
          marketQuestion: String(trade.market_title || trade.condition_id || 'Unknown market'),
          marketId: String(trade.condition_id || trade.token_id || ''),
          outcome: String(trade.outcome || 'N/A').toUpperCase(),
          category: normalizeCategory(trade.category),
          quantity: Number.isFinite(Number(trade.quantity)) ? Number(trade.quantity) : null,
          buyPrice: Number.isFinite(Number(trade.avg_buy_price)) ? Number(trade.avg_buy_price) : null,
          sellPrice: Number.isFinite(Number(trade.avg_sell_price)) ? Number(trade.avg_sell_price) : null,
          buyNotional,
          sellNotional: toRoundTripNotional(trade.sell_notional),
          pnl,
          roiPercent: Number.isFinite(Number(trade.roi_percent)) ? Number(trade.roi_percent) : (buyNotional > 0 ? (pnl / buyNotional) * 100 : 0),
          holdMinutes: Number.isFinite(Number(trade.hold_minutes)) ? Number(trade.hold_minutes) : null,
          status: 'closed',
          openedAt: String(trade.opened_at || trade.closed_at),
          closedAt: String(trade.closed_at),
          isResolved: true,
          isWin: pnl > 0,
          isLoss: pnl < 0,
        })
      })
    }

    rows.sort((left, right) => toTs(right.closedAt) - toTs(left.closedAt))
    return rows
  }, [filteredLiveRoundTrips, filteredSimulationTrades, viewMode])

  const historyCategoryOptions = useMemo(() => {
    const categories = new Set<string>()
    unifiedTrades.forEach((trade) => {
      const category = normalizeCategory(trade.category)
      if (category) categories.add(category)
    })
    return Array.from(categories).sort((left, right) => left.localeCompare(right))
  }, [unifiedTrades])

  const filteredSortedHistoryTrades = useMemo(() => {
    const query = deferredHistorySearch.trim().toLowerCase()

    const filtered = unifiedTrades.filter((trade) => {
      const category = normalizeCategory(trade.category)
      if (historyCategoryFilter !== 'all' && category !== historyCategoryFilter) return false
      if (historyOutcomeFilter === 'wins' && !trade.isWin) return false
      if (historyOutcomeFilter === 'losses' && !trade.isLoss) return false
      if (historyOutcomeFilter === 'breakeven' && (trade.isWin || trade.isLoss)) return false

      if (!query) return true
      const haystack = `${trade.marketQuestion} ${trade.marketId} ${trade.outcome} ${trade.category} ${trade.strategy} ${trade.accountName || ''}`.toLowerCase()
      return haystack.includes(query)
    })

    const sorted = [...filtered]
    sorted.sort((left, right) => {
      if (historySortKey === 'marketQuestion' || historySortKey === 'category') {
        const leftText = historySortKey === 'marketQuestion' ? left.marketQuestion : normalizeCategory(left.category)
        const rightText = historySortKey === 'marketQuestion' ? right.marketQuestion : normalizeCategory(right.category)
        const cmp = leftText.localeCompare(rightText)
        return historySortDirection === 'asc' ? cmp : -cmp
      }

      let leftValue = 0
      let rightValue = 0
      switch (historySortKey) {
        case 'closedAt':
          leftValue = toTs(left.closedAt)
          rightValue = toTs(right.closedAt)
          break
        case 'pnl':
          leftValue = left.pnl ?? 0
          rightValue = right.pnl ?? 0
          break
        case 'roiPercent':
          leftValue = left.roiPercent
          rightValue = right.roiPercent
          break
        case 'holdMinutes':
          leftValue = left.holdMinutes ?? -1
          rightValue = right.holdMinutes ?? -1
          break
        case 'buyNotional':
          leftValue = left.buyNotional
          rightValue = right.buyNotional
          break
        case 'sellNotional':
          leftValue = left.sellNotional
          rightValue = right.sellNotional
          break
      }

      const cmp = leftValue - rightValue
      if (cmp === 0) return toTs(right.closedAt) - toTs(left.closedAt)
      return historySortDirection === 'asc' ? cmp : -cmp
    })

    return sorted
  }, [
    historyCategoryFilter,
    historyOutcomeFilter,
    deferredHistorySearch,
    historySortDirection,
    historySortKey,
    unifiedTrades,
  ])

  const tradeTapePageCount = useMemo(
    () => Math.max(1, Math.ceil(filteredSortedHistoryTrades.length / TRADE_TAPE_PAGE_SIZE)),
    [filteredSortedHistoryTrades.length]
  )

  useEffect(() => {
    setTradeTapePage((current) => Math.min(current, tradeTapePageCount))
  }, [tradeTapePageCount])

  useEffect(() => {
    setHistoryCategoryFilter('all')
    setHistoryOutcomeFilter('all')
    setHistorySearch('')
    setHistorySortKey('closedAt')
    setHistorySortDirection('desc')
  }, [viewMode])

  useEffect(() => {
    setTradeTapePage(1)
  }, [
    detailTab,
    historyCategoryFilter,
    historyOutcomeFilter,
    historySearch,
    historySortDirection,
    historySortKey,
    timeRange,
    viewMode,
  ])

  const pagedUnifiedTrades = useMemo(() => {
    const start = (tradeTapePage - 1) * TRADE_TAPE_PAGE_SIZE
    return filteredSortedHistoryTrades.slice(start, start + TRADE_TAPE_PAGE_SIZE)
  }, [filteredSortedHistoryTrades, tradeTapePage])

  const resolvedRows = useMemo(
    () => unifiedTrades.filter((trade) => trade.isResolved && trade.pnl !== null),
    [unifiedTrades]
  )

  const summary = useMemo(() => {
    const wins = resolvedRows.filter((trade) => trade.isWin)
    const losses = resolvedRows.filter((trade) => trade.isLoss)

    const totalPnl = resolvedRows.reduce((sum, trade) => sum + (trade.pnl || 0), 0)
    const totalCost = resolvedRows.reduce((sum, trade) => sum + trade.buyNotional, 0)
    const averagePnl = resolvedRows.length > 0 ? totalPnl / resolvedRows.length : 0
    const averageHoldMinutes = resolvedRows.length > 0
      ? resolvedRows.reduce((sum, trade) => sum + (trade.holdMinutes || 0), 0) / resolvedRows.length
      : 0

    const grossWins = wins.reduce((sum, trade) => sum + Math.max(0, trade.pnl || 0), 0)
    const grossLosses = losses.reduce((sum, trade) => sum + Math.abs(Math.min(0, trade.pnl || 0)), 0)
    const profitFactor = grossLosses > 0
      ? grossWins / grossLosses
      : grossWins > 0
        ? 999
        : 0

    const openSimulationTrades = filteredSimulationTrades.filter(
      (trade) => !SIM_RESOLVED_STATUSES.has(normalizeStatus(trade.status))
    ).length
    const openLiveLots = filteredLiveOpenLots.length
    const openLiveNotional = filteredLiveOpenLots.reduce((sum, lot: LiveWalletOpenLot) => sum + Number(lot.cost_basis || 0), 0)

    return {
      totalTrades: unifiedTrades.length,
      resolvedTrades: resolvedRows.length,
      openTrades: viewMode === 'live' ? openLiveLots : openSimulationTrades,
      openLiveLots,
      openLiveNotional,
      wins: wins.length,
      losses: losses.length,
      totalPnl,
      totalCost,
      winRate: resolvedRows.length > 0 ? (wins.length / resolvedRows.length) * 100 : 0,
      roi: totalCost > 0 ? (totalPnl / totalCost) * 100 : 0,
      avgPnl: averagePnl,
      avgHoldMinutes: averageHoldMinutes,
      profitFactor,
      grossWins,
      grossLosses,
    }
  }, [filteredLiveOpenLots, filteredSimulationTrades, resolvedRows, unifiedTrades, viewMode])

  const advancedMetrics = useMemo(() => {
    const wins = resolvedRows.filter((row) => row.isWin)
    const losses = resolvedRows.filter((row) => row.isLoss)

    const avgWin = wins.length > 0
      ? wins.reduce((sum, row) => sum + (row.pnl || 0), 0) / wins.length
      : 0

    const avgLoss = losses.length > 0
      ? losses.reduce((sum, row) => sum + (row.pnl || 0), 0) / losses.length
      : 0

    const winRate = resolvedRows.length > 0 ? wins.length / resolvedRows.length : 0
    const expectancy = (winRate * avgWin) + ((1 - winRate) * avgLoss)

    const sortedRoi = resolvedRows
      .map((row) => row.roiPercent)
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b)

    const sortedHold = resolvedRows
      .map((row) => row.holdMinutes)
      .filter((value): value is number => value !== null && Number.isFinite(value))
      .sort((a, b) => a - b)

    const median = (arr: number[]): number => {
      if (arr.length === 0) return 0
      const mid = Math.floor(arr.length / 2)
      return arr.length % 2 === 0 ? (arr[mid - 1] + arr[mid]) / 2 : arr[mid]
    }

    const bestTrade = resolvedRows.reduce((best, row) => {
      const pnl = row.pnl || 0
      if (best === null || pnl > (best.pnl || 0)) return row
      return best
    }, null as UnifiedTradeRow | null)

    const worstTrade = resolvedRows.reduce((worst, row) => {
      const pnl = row.pnl || 0
      if (worst === null || pnl < (worst.pnl || 0)) return row
      return worst
    }, null as UnifiedTradeRow | null)

    return {
      avgWin,
      avgLoss,
      expectancy,
      medianRoi: median(sortedRoi),
      medianHold: median(sortedHold),
      bestTrade,
      worstTrade,
    }
  }, [resolvedRows])

  const cumulativePnlData = useMemo(() => {
    const daily = new Map<string, number>()

    resolvedRows.forEach((trade) => {
      const day = (trade.closedAt || '').split('T')[0]
      if (!day) return
      daily.set(day, (daily.get(day) || 0) + Number(trade.pnl || 0))
    })

    const sortedDays = Array.from(daily.keys()).sort()
    let cumulative = 0

    return sortedDays.map((day) => {
      cumulative += daily.get(day) || 0
      return {
        date: day,
        cumulativePnl: cumulative,
      }
    })
  }, [resolvedRows])

  const maxDrawdown = useMemo(() => {
    if (cumulativePnlData.length === 0) return 0

    let peak = Number.NEGATIVE_INFINITY
    let drawdown = 0

    cumulativePnlData.forEach((point) => {
      if (point.cumulativePnl > peak) peak = point.cumulativePnl
      drawdown = Math.max(drawdown, peak - point.cumulativePnl)
    })

    return drawdown
  }, [cumulativePnlData])

  const categoryRollup = useMemo(
    () => aggregateRollup(
      resolvedRows,
      (row) => viewMode === 'live' ? normalizeCategory(row.category) : row.strategy
    ),
    [resolvedRows, viewMode]
  )

  const weekdayRollup = useMemo(() => {
    const rows = aggregateRollup(resolvedRows, (row) => {
      const dayIndex = new Date(row.closedAt).getDay()
      return DAY_LABEL_BY_INDEX[dayIndex] || 'Unknown'
    })

    const byLabel = new Map(rows.map((row) => [row.label, row]))
    return DAY_ORDER.map((dayIndex) => {
      const label = DAY_LABEL_BY_INDEX[dayIndex]
      return byLabel.get(label) || {
        key: label,
        label,
        trades: 0,
        wins: 0,
        losses: 0,
        pnl: 0,
        buyNotional: 0,
        roiSum: 0,
      }
    })
  }, [resolvedRows])

  const timeOfDayRollup = useMemo(() => {
    const byBucket = new Map<string, RollupRow>()

    TIME_BUCKETS.forEach((bucket) => {
      byBucket.set(bucket.key, {
        key: bucket.key,
        label: bucket.label,
        trades: 0,
        wins: 0,
        losses: 0,
        pnl: 0,
        buyNotional: 0,
        roiSum: 0,
      })
    })

    resolvedRows.forEach((row) => {
      const hour = new Date(row.closedAt).getHours()
      const bucket = TIME_BUCKETS.find((entry) => hour >= entry.startHour && hour < entry.endHour)
      if (!bucket) return
      const current = byBucket.get(bucket.key)
      if (!current) return

      current.trades += 1
      current.pnl += row.pnl || 0
      current.buyNotional += row.buyNotional
      current.roiSum += row.roiPercent
      if (row.isWin) current.wins += 1
      if (row.isLoss) current.losses += 1
    })

    return TIME_BUCKETS.map((bucket) => byBucket.get(bucket.key) as RollupRow)
  }, [resolvedRows])

  const patternInsights = useMemo(() => {
    const ordered = [...resolvedRows].sort((left, right) => toTs(left.closedAt) - toTs(right.closedAt))
    let currentWinStreak = 0
    let currentLossStreak = 0
    let maxWinStreak = 0
    let maxLossStreak = 0

    ordered.forEach((trade) => {
      const pnl = trade.pnl ?? 0
      if (pnl > 0) {
        currentWinStreak += 1
        currentLossStreak = 0
      } else if (pnl < 0) {
        currentLossStreak += 1
        currentWinStreak = 0
      } else {
        currentWinStreak = 0
        currentLossStreak = 0
      }
      if (currentWinStreak > maxWinStreak) maxWinStreak = currentWinStreak
      if (currentLossStreak > maxLossStreak) maxLossStreak = currentLossStreak
    })

    const activeDays = new Set(
      ordered
        .map((trade) => (trade.closedAt || '').split('T')[0])
        .filter(Boolean)
    ).size
    const tradesPerActiveDay = activeDays > 0 ? ordered.length / activeDays : 0

    const dayRows = weekdayRollup.filter((row) => row.trades > 0)
    const bestDay = dayRows.reduce((best, row) => {
      if (best === null || row.pnl > best.pnl) return row
      return best
    }, null as RollupRow | null)
    const worstDay = dayRows.reduce((worst, row) => {
      if (worst === null || row.pnl < worst.pnl) return row
      return worst
    }, null as RollupRow | null)

    const timeRows = timeOfDayRollup.filter((row) => row.trades > 0)
    const bestSession = timeRows.reduce((best, row) => {
      if (best === null || row.pnl > best.pnl) return row
      return best
    }, null as RollupRow | null)
    const worstSession = timeRows.reduce((worst, row) => {
      if (worst === null || row.pnl < worst.pnl) return row
      return worst
    }, null as RollupRow | null)

    const totalCategoryNotional = categoryRollup.reduce((sum, row) => sum + row.buyNotional, 0)
    const topCategory = categoryRollup.length > 0
      ? [...categoryRollup].sort((left, right) => right.buyNotional - left.buyNotional)[0]
      : null
    const topCategoryNotionalShare = (
      totalCategoryNotional > 0 && topCategory
        ? (topCategory.buyNotional / totalCategoryNotional) * 100
        : 0
    )
    const topCategoryTradeShare = (
      resolvedRows.length > 0 && topCategory
        ? (topCategory.trades / resolvedRows.length) * 100
        : 0
    )

    return {
      maxWinStreak,
      maxLossStreak,
      activeDays,
      tradesPerActiveDay,
      bestDay,
      worstDay,
      bestSession,
      worstSession,
      topCategory,
      topCategoryNotionalShare,
      topCategoryTradeShare,
    }
  }, [categoryRollup, resolvedRows, timeOfDayRollup, weekdayRollup])

  const isLoading = accountsLoading || simulationTradesLoading || livePerformanceLoading
  const showLoadingSkeleton = (
    viewMode === 'live'
      ? livePerformanceLoading && !livePerformance
      : simulationTradesLoading && simulationTrades.length === 0 && (accountsLoading || accounts.length > 0)
  )

  const handleRefresh = () => {
    if (viewMode === 'simulation') {
      void refetchSimulationTrades()
      return
    }
    void refetchLivePerformance()
  }

  const modeLabel = viewMode === 'live' ? 'Live' : 'Sandbox'

  return (
    <div className="h-full min-h-0 flex flex-col gap-2 overflow-hidden">
      <div className="shrink-0 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-300" />
              Performance
            </h2>
            <Badge variant="outline" className="text-[10px] border-border/50">
              {modeLabel} • {summary.totalTrades} trades
            </Badge>
            {viewMode === 'live' && (
              <Badge variant="outline" className="text-[10px] border-border/50">
                Wallet {shortAddress(livePerformance?.wallet_address)}
              </Badge>
            )}
          </div>

          <Button variant="ghost" size="sm" onClick={handleRefresh} disabled={isLoading} className="h-7 px-2">
            <RefreshCw className={cn('w-3.5 h-3.5', isLoading && 'animate-spin')} />
          </Button>
        </div>

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

          {viewMode === 'simulation' && accounts.length > 0 && (
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

      {showLoadingSkeleton ? (
        <PerformanceMetricSkeleton />
      ) : (
        <div className="shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 border-y border-border/50 py-1.5 px-0.5">
          <MetricChip
            label="Realized P&L"
            value={formatSignedCurrency(summary.totalPnl, true)}
            detail={formatCurrency(summary.totalPnl)}
            icon={summary.totalPnl >= 0 ? TrendingUp : TrendingDown}
            valueClassName={summary.totalPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
          />
          <MetricChip
            label="ROI"
            value={formatSignedPercent(summary.roi)}
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
            detail={viewMode === 'live' ? `${formatCurrency(summary.openLiveNotional, true)} inventory` : `${summary.resolvedTrades} closed`}
            icon={Calendar}
          />
          <MetricChip
            label="Expectancy"
            value={formatSignedCurrency(advancedMetrics.expectancy, true)}
            detail={`${formatSignedCurrency(advancedMetrics.avgWin, true)} avg win / ${formatSignedCurrency(advancedMetrics.avgLoss, true)} avg loss`}
            icon={TrendingUp}
            valueClassName={advancedMetrics.expectancy >= 0 ? 'text-emerald-300' : 'text-red-300'}
          />
          <MetricChip
            label="Median Hold"
            value={formatDurationMinutes(advancedMetrics.medianHold)}
            icon={Clock3}
          />
          <MetricChip
            label="Drawdown"
            value={formatCurrency(maxDrawdown, true)}
            detail={`PF ${summary.profitFactor >= 999 ? '∞' : summary.profitFactor.toFixed(2)}`}
            icon={TrendingDown}
            valueClassName={maxDrawdown > 0 ? 'text-amber-300' : undefined}
          />
        </div>
      )}

      <Tabs
        value={detailTab}
        onValueChange={(value) => setDetailTab(value as DetailTab)}
        className="flex-1 min-h-0 flex flex-col"
      >
        <TabsList className="h-auto w-fit rounded-lg border border-border/80 bg-background/70 p-1">
          <TabsTrigger value="overview" className="h-7 px-3 text-xs">Overview</TabsTrigger>
          <TabsTrigger value="insights" className="h-7 px-3 text-xs">Insights</TabsTrigger>
          <TabsTrigger value="history" className="h-7 px-3 text-xs">History</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-2 flex-1 min-h-0">
          {showLoadingSkeleton ? (
            <PerformanceOverviewSkeleton />
          ) : (
            <div className="h-full min-h-0 grid gap-2 lg:grid-cols-12">
              <div className="lg:col-span-8 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col min-h-[260px]">
                <div className="shrink-0 flex items-center justify-between gap-2 mb-2">
                  <p className="text-xs font-semibold flex items-center gap-1.5">
                    <BarChart3 className="h-3.5 w-3.5 text-cyan-300" />
                    {modeLabel} Cumulative Realized P&L
                  </p>
                  {viewMode === 'live' && orchestratorStats?.last_trade_at && (
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
                    <PerformancePnlChart data={cumulativePnlData} mode={viewMode} />
                  )}
                </div>
              </div>

              <div className="lg:col-span-4 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col min-h-[260px]">
                <p className="shrink-0 text-xs font-semibold mb-2">Quick Insights</p>
                <div className="space-y-2 text-xs">
                  <InsightStat
                    label="Best Trade"
                    value={advancedMetrics.bestTrade ? formatSignedCurrency(advancedMetrics.bestTrade.pnl || 0, true) : '—'}
                    hint={advancedMetrics.bestTrade ? advancedMetrics.bestTrade.marketQuestion : 'No closed trades'}
                    positive={(advancedMetrics.bestTrade?.pnl || 0) >= 0}
                  />
                  <InsightStat
                    label="Worst Trade"
                    value={advancedMetrics.worstTrade ? formatSignedCurrency(advancedMetrics.worstTrade.pnl || 0, true) : '—'}
                    hint={advancedMetrics.worstTrade ? advancedMetrics.worstTrade.marketQuestion : 'No closed trades'}
                    positive={false}
                  />
                  <InsightStat
                    label="Median ROI"
                    value={formatSignedPercent(advancedMetrics.medianRoi)}
                    hint="Robust against outliers"
                    positive={advancedMetrics.medianRoi >= 0}
                  />
                  <InsightStat
                    label="Gross Win/Loss"
                    value={`${formatCurrency(summary.grossWins, true)} / ${formatCurrency(summary.grossLosses, true)}`}
                    hint="Absolute dollars"
                  />
                </div>
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="insights" className="mt-2 flex-1 min-h-0">
          {showLoadingSkeleton ? (
            <PerformanceInsightsSkeleton />
          ) : (
            <div className="h-full min-h-0 grid gap-2 lg:grid-cols-12">
            <div className="lg:col-span-5 rounded-md border border-border/60 bg-card/80 p-3 min-h-0 flex flex-col">
              <p className="shrink-0 text-xs font-semibold mb-2">
                {viewMode === 'live' ? 'Market Category Performance' : 'Strategy Performance'}
              </p>
              <ScrollArea className="flex-1 min-h-0">
                <div className="space-y-1.5 pr-2">
                  {categoryRollup.length === 0 ? (
                    <p className="text-[11px] text-muted-foreground">No closed trades in range.</p>
                  ) : (
                    categoryRollup.map((row) => {
                      const winRate = row.trades > 0 ? (row.wins / row.trades) * 100 : 0
                      const avgRoi = row.trades > 0 ? row.roiSum / row.trades : 0
                      return (
                        <div key={row.key} className="rounded border border-border/50 bg-background/30 px-2 py-1.5">
                          <div className="flex items-center justify-between gap-1">
                            <p className="truncate text-xs">{row.label}</p>
                            <p className={cn('text-[11px] font-mono', row.pnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                              {formatSignedCurrency(row.pnl, true)}
                            </p>
                          </div>
                          <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                            <span>{row.trades}t</span>
                            <span>{formatPercent(winRate)}</span>
                            <span>{formatSignedPercent(avgRoi)}</span>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              </ScrollArea>
            </div>

            <div className="lg:col-span-4 rounded-md border border-border/60 bg-card/80 p-3 min-h-0 flex flex-col">
              <p className="shrink-0 text-xs font-semibold mb-2">Time Performance</p>
              <ScrollArea className="flex-1 min-h-0">
                <div className="space-y-3 pr-2">
                  <section>
                    <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">Day of Week</p>
                    <div className="space-y-1">
                      {weekdayRollup.map((row) => {
                        const winRate = row.trades > 0 ? (row.wins / row.trades) * 100 : 0
                        return (
                          <div key={row.key} className="flex items-center justify-between rounded border border-border/40 bg-background/20 px-2 py-1">
                            <span className="text-[10px] text-muted-foreground">{row.label}</span>
                            <div className="flex items-center gap-2 text-[10px]">
                              <span className={cn('font-mono', row.pnl >= 0 ? 'text-emerald-300' : row.pnl < 0 ? 'text-red-300' : 'text-muted-foreground')}>
                                {formatSignedCurrency(row.pnl, true)}
                              </span>
                              <span className="text-muted-foreground">{row.trades}t</span>
                              <span className="text-muted-foreground">{formatPercent(winRate)}</span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </section>

                  <section>
                    <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">Time of Day</p>
                    <div className="space-y-1">
                      {timeOfDayRollup.map((row) => {
                        const winRate = row.trades > 0 ? (row.wins / row.trades) * 100 : 0
                        return (
                          <div key={row.key} className="flex items-center justify-between rounded border border-border/40 bg-background/20 px-2 py-1">
                            <span className="text-[10px] text-muted-foreground">{row.label}</span>
                            <div className="flex items-center gap-2 text-[10px]">
                              <span className={cn('font-mono', row.pnl >= 0 ? 'text-emerald-300' : row.pnl < 0 ? 'text-red-300' : 'text-muted-foreground')}>
                                {formatSignedCurrency(row.pnl, true)}
                              </span>
                              <span className="text-muted-foreground">{row.trades}t</span>
                              <span className="text-muted-foreground">{formatPercent(winRate)}</span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </section>
                </div>
              </ScrollArea>
            </div>

            <div className="lg:col-span-3 rounded-md border border-border/60 bg-card/80 p-3 min-h-0 flex flex-col">
              <p className="shrink-0 text-xs font-semibold mb-2">Pattern Insights</p>
              <ScrollArea className="flex-1 min-h-0">
                <div className="space-y-2 pr-2 text-xs">
                  <InsightStat
                    label="Best Day"
                    value={patternInsights.bestDay ? `${patternInsights.bestDay.label} ${formatSignedCurrency(patternInsights.bestDay.pnl, true)}` : '—'}
                    hint={patternInsights.bestDay ? `${patternInsights.bestDay.trades} trades, ${formatPercent((patternInsights.bestDay.wins / Math.max(1, patternInsights.bestDay.trades)) * 100)} win rate` : 'No closed trades in range'}
                    positive={(patternInsights.bestDay?.pnl || 0) >= 0}
                  />
                  <InsightStat
                    label="Weakest Day"
                    value={patternInsights.worstDay ? `${patternInsights.worstDay.label} ${formatSignedCurrency(patternInsights.worstDay.pnl, true)}` : '—'}
                    hint={patternInsights.worstDay ? `${patternInsights.worstDay.trades} trades, ${formatPercent((patternInsights.worstDay.wins / Math.max(1, patternInsights.worstDay.trades)) * 100)} win rate` : 'No closed trades in range'}
                    positive={false}
                  />
                  <InsightStat
                    label="Best Session"
                    value={patternInsights.bestSession ? `${patternInsights.bestSession.label} ${formatSignedCurrency(patternInsights.bestSession.pnl, true)}` : '—'}
                    hint={patternInsights.bestSession ? `${patternInsights.bestSession.trades} trades in this time bucket` : 'No closed trades in range'}
                    positive={(patternInsights.bestSession?.pnl || 0) >= 0}
                  />
                  <InsightStat
                    label="Streak Profile"
                    value={`${patternInsights.maxWinStreak}W / ${patternInsights.maxLossStreak}L`}
                    hint={`${patternInsights.activeDays} active days, ${patternInsights.tradesPerActiveDay.toFixed(1)} trades/day`}
                    positive={patternInsights.maxWinStreak >= patternInsights.maxLossStreak}
                  />
                  <InsightStat
                    label="Top Category Concentration"
                    value={patternInsights.topCategory ? `${patternInsights.topCategory.label} ${formatPercent(patternInsights.topCategoryNotionalShare)}` : '—'}
                    hint={patternInsights.topCategory ? `${formatPercent(patternInsights.topCategoryTradeShare)} of closed trades` : 'No category concentration in range'}
                  />
                </div>
              </ScrollArea>
            </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="history" className="mt-2 flex-1 min-h-0">
          {showLoadingSkeleton ? (
            <PerformanceHistorySkeleton />
          ) : (
            <div className="h-full min-h-0 rounded-md border border-border/60 bg-card/80 flex flex-col">
            <div className="shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-b border-border/40">
              <p className="text-xs font-semibold">Trade History</p>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-muted-foreground">
                  {filteredSortedHistoryTrades.length}/{unifiedTrades.length} trades • {TRADE_TAPE_PAGE_SIZE}/page
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

            <div className="shrink-0 border-b border-border/40 px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <input
                  value={historySearch}
                  onChange={(event) => setHistorySearch(event.target.value)}
                  placeholder="Search market, outcome, category, strategy"
                  className="h-7 min-w-[220px] flex-1 rounded-md border border-border bg-background/80 px-2 text-xs"
                />
                <select
                  value={historyCategoryFilter}
                  onChange={(event) => setHistoryCategoryFilter(event.target.value)}
                  className="h-7 rounded-md border border-border bg-background/80 px-2 text-xs"
                >
                  <option value="all">{viewMode === 'live' ? 'All categories' : 'All strategy categories'}</option>
                  {historyCategoryOptions.map((category) => (
                    <option key={category} value={category}>{category}</option>
                  ))}
                </select>
                <select
                  value={historyOutcomeFilter}
                  onChange={(event) => setHistoryOutcomeFilter(event.target.value as HistoryOutcomeFilter)}
                  className="h-7 rounded-md border border-border bg-background/80 px-2 text-xs"
                >
                  <option value="all">All outcomes</option>
                  <option value="wins">Wins only</option>
                  <option value="losses">Losses only</option>
                  <option value="breakeven">Breakeven only</option>
                </select>
                <select
                  value={historySortKey}
                  onChange={(event) => setHistorySortKey(event.target.value as HistorySortKey)}
                  className="h-7 rounded-md border border-border bg-background/80 px-2 text-xs"
                >
                  {HISTORY_SORT_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>{option.label}</option>
                  ))}
                </select>
                <select
                  value={historySortDirection}
                  onChange={(event) => setHistorySortDirection(event.target.value as HistorySortDirection)}
                  className="h-7 rounded-md border border-border bg-background/80 px-2 text-xs"
                >
                  <option value="desc">Desc</option>
                  <option value="asc">Asc</option>
                </select>
              </div>
            </div>

            <ScrollArea className="flex-1 min-h-0">
              <div className="w-full overflow-x-auto">
                <Table className="min-w-[1024px]">
                  <TableHeader>
                    <TableRow className="sticky top-0 z-10 bg-card/95 backdrop-blur-sm">
                      <TableHead className="text-[10px]">Market</TableHead>
                      <TableHead className="text-[10px]">Outcome</TableHead>
                      <TableHead className="text-[10px] text-right">Qty</TableHead>
                      <TableHead className="text-[10px] text-right">Buy Px</TableHead>
                      <TableHead className="text-[10px] text-right">Sell Px</TableHead>
                      <TableHead className="text-[10px] text-right">Buy $</TableHead>
                      <TableHead className="text-[10px] text-right">Sell $</TableHead>
                      <TableHead className="text-[10px] text-right">P&amp;L</TableHead>
                      <TableHead className="text-[10px] text-right">ROI</TableHead>
                      <TableHead className="text-[10px] text-right">Hold</TableHead>
                      <TableHead className="text-[10px]">Category</TableHead>
                      <TableHead className="text-[10px]">Closed</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pagedUnifiedTrades.map((trade) => (
                      <TableRow key={trade.id} className="text-[11px] leading-tight hover:bg-muted/30">
                        <TableCell className="max-w-[260px] py-1">
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
                          <p
                            className="truncate text-[9px] leading-none text-muted-foreground"
                            title={`${trade.strategy}${trade.accountName ? ` • ${trade.accountName}` : ''}`}
                          >
                            {trade.strategy}{trade.accountName ? ` • ${trade.accountName}` : ''}
                          </p>
                        </TableCell>
                        <TableCell className="py-1">
                          <Badge variant="outline" className="h-4 max-w-[120px] truncate border-border/80 bg-muted/60 px-1 text-[9px] text-muted-foreground">
                            {trade.outcome}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right font-mono py-1 text-[10px]">{formatQuantity(trade.quantity)}</TableCell>
                        <TableCell className="text-right font-mono py-1 text-[10px]">{formatPrice(trade.buyPrice)}</TableCell>
                        <TableCell className="text-right font-mono py-1 text-[10px]">{formatPrice(trade.sellPrice)}</TableCell>
                        <TableCell className="text-right font-mono py-1 text-[10px]">{trade.buyNotional > 0 ? formatCurrency(trade.buyNotional, true) : '—'}</TableCell>
                        <TableCell className="text-right font-mono py-1 text-[10px]">{trade.sellNotional > 0 ? formatCurrency(trade.sellNotional, true) : '—'}</TableCell>
                        <TableCell className={cn('text-right font-mono py-1 text-[10px]', (trade.pnl ?? 0) > 0 ? 'text-emerald-500' : (trade.pnl ?? 0) < 0 ? 'text-red-500' : '')}>
                          {trade.pnl !== null ? formatSignedCurrency(trade.pnl, true) : '—'}
                        </TableCell>
                        <TableCell className={cn('text-right font-mono py-1 text-[10px]', trade.roiPercent > 0 ? 'text-emerald-500' : trade.roiPercent < 0 ? 'text-red-500' : '')}>
                          {trade.pnl !== null ? formatSignedPercent(trade.roiPercent) : '—'}
                        </TableCell>
                        <TableCell className="text-right font-mono py-1 text-[10px]">{formatDurationMinutes(trade.holdMinutes)}</TableCell>
                        <TableCell className="py-1 text-[10px] text-muted-foreground truncate" title={trade.category}>{trade.category}</TableCell>
                        <TableCell className="py-1 text-[9px] text-muted-foreground">
                          <span title={new Date(trade.closedAt).toLocaleString()}>{formatRelativeAge(trade.closedAt)}</span>
                        </TableCell>
                      </TableRow>
                    ))}
                    {filteredSortedHistoryTrades.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={12} className="py-6 text-center text-xs text-muted-foreground">
                          No trades match the current filters.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </ScrollArea>
            </div>
          )}
        </TabsContent>
      </Tabs>
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
  icon: ElementType
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

function PerformanceMetricSkeleton() {
  return (
    <div className="shrink-0 flex flex-wrap items-center gap-3 border-y border-border/50 py-1.5 px-0.5 animate-pulse">
      {Array.from({ length: 7 }).map((_, index) => (
        <div key={`performance-metric-skeleton-${index}`} className="flex items-center gap-2 text-xs">
          <div className="h-3.5 w-3.5 rounded-full bg-muted/55" />
          <div className="h-2.5 w-14 rounded bg-muted/50" />
          <div className="h-3 w-16 rounded bg-muted/60" />
        </div>
      ))}
    </div>
  )
}

function PerformanceOverviewSkeleton() {
  return (
    <div className="h-full min-h-0 grid gap-2 lg:grid-cols-12 animate-pulse">
      <div className="lg:col-span-8 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col min-h-[260px]">
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="h-3 w-40 rounded bg-muted/55" />
          <div className="h-2.5 w-28 rounded bg-muted/45" />
        </div>
        <div className="flex-1 rounded-md bg-muted/20" />
      </div>
      <div className="lg:col-span-4 rounded-md border border-border/60 bg-card/80 p-3 flex flex-col min-h-[260px]">
        <div className="h-3 w-24 rounded bg-muted/55" />
        <div className="mt-3 space-y-2">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={`performance-overview-skeleton-${index}`} className="rounded border border-border/40 bg-background/35 px-2.5 py-2">
              <div className="h-2.5 w-20 rounded bg-muted/50" />
              <div className="mt-2 h-3.5 w-24 rounded bg-muted/60" />
              <div className="mt-2 h-2.5 w-full rounded bg-muted/40" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function PerformanceInsightsSkeleton() {
  return (
    <div className="h-full min-h-0 grid gap-2 lg:grid-cols-12 animate-pulse">
      <div className="lg:col-span-5 rounded-md border border-border/60 bg-card/80 p-3 min-h-0">
        <div className="h-3 w-36 rounded bg-muted/55" />
        <div className="mt-3 space-y-1.5">
          {Array.from({ length: 7 }).map((_, index) => (
            <div key={`performance-insight-left-${index}`} className="rounded border border-border/40 bg-background/30 px-2 py-1.5">
              <div className="flex items-center justify-between gap-2">
                <div className="h-2.5 w-24 rounded bg-muted/45" />
                <div className="h-2.5 w-16 rounded bg-muted/55" />
              </div>
              <div className="mt-2 h-2 w-20 rounded bg-muted/40" />
            </div>
          ))}
        </div>
      </div>
      <div className="lg:col-span-4 rounded-md border border-border/60 bg-card/80 p-3 min-h-0">
        <div className="h-3 w-24 rounded bg-muted/55" />
        <div className="mt-3 space-y-3">
          {Array.from({ length: 2 }).map((_, sectionIndex) => (
            <div key={`performance-insight-mid-${sectionIndex}`}>
              <div className="h-2 w-16 rounded bg-muted/40 mb-2" />
              <div className="space-y-1">
                {Array.from({ length: 4 }).map((_, rowIndex) => (
                  <div key={`performance-insight-mid-row-${sectionIndex}-${rowIndex}`} className="flex items-center justify-between rounded border border-border/40 bg-background/20 px-2 py-1">
                    <div className="h-2 w-10 rounded bg-muted/40" />
                    <div className="h-2 w-24 rounded bg-muted/50" />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="lg:col-span-3 rounded-md border border-border/60 bg-card/80 p-3 min-h-0">
        <div className="h-3 w-24 rounded bg-muted/55" />
        <div className="mt-3 space-y-2">
          {Array.from({ length: 5 }).map((_, index) => (
            <div key={`performance-insight-right-${index}`} className="rounded border border-border/40 bg-background/30 px-2.5 py-2">
              <div className="h-2.5 w-20 rounded bg-muted/45" />
              <div className="mt-2 h-3 w-28 rounded bg-muted/55" />
              <div className="mt-2 h-2 w-full rounded bg-muted/40" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function PerformanceHistorySkeleton() {
  return (
    <div className="h-full min-h-0 rounded-md border border-border/60 bg-card/80 flex flex-col animate-pulse">
      <div className="shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-b border-border/40">
        <div className="h-3 w-20 rounded bg-muted/55" />
        <div className="h-2.5 w-32 rounded bg-muted/45" />
      </div>
      <div className="shrink-0 border-b border-border/40 px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <div className="h-7 min-w-[220px] flex-1 rounded-md bg-muted/35" />
          <div className="h-7 w-32 rounded-md bg-muted/35" />
          <div className="h-7 w-28 rounded-md bg-muted/35" />
          <div className="h-7 w-28 rounded-md bg-muted/35" />
          <div className="h-7 w-20 rounded-md bg-muted/35" />
        </div>
      </div>
      <div className="flex-1 p-3">
        <div className="rounded-md border border-border/40 bg-background/25">
          {Array.from({ length: 10 }).map((_, index) => (
            <div key={`performance-history-skeleton-${index}`} className="grid grid-cols-[3fr_repeat(5,minmax(0,1fr))_2fr] gap-3 border-b border-border/30 px-3 py-2 last:border-b-0">
              <div className="h-2.5 rounded bg-muted/45" />
              <div className="h-2.5 rounded bg-muted/35" />
              <div className="h-2.5 rounded bg-muted/35" />
              <div className="h-2.5 rounded bg-muted/35" />
              <div className="h-2.5 rounded bg-muted/35" />
              <div className="h-2.5 rounded bg-muted/35" />
              <div className="h-2.5 rounded bg-muted/40" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function InsightStat({
  label,
  value,
  hint,
  positive,
}: {
  label: string
  value: string
  hint: string
  positive?: boolean
}) {
  return (
    <div className="rounded border border-border/50 bg-background/30 px-2 py-1.5">
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className={cn('text-sm font-mono', positive === undefined ? 'text-foreground' : positive ? 'text-emerald-300' : 'text-red-300')}>
        {value}
      </p>
      <p className="truncate text-[10px] text-muted-foreground" title={hint}>{hint}</p>
    </div>
  )
}

function PerformancePnlChart({
  data,
  mode,
}: {
  data: PnlPoint[]
  mode: ViewMode
}) {
  const stroke = mode === 'live' ? '#22d3ee' : '#f59e0b'
  const gradientId = mode === 'live' ? 'liveModeGradient' : 'sandboxModeGradient'
  const label = mode === 'live' ? 'Live cumulative' : 'Sandbox cumulative'

  const tooltipFormatter = (value: number | string | undefined) => {
    const numericValue = Number(value)
    return [formatCurrency(Number.isFinite(numericValue) ? numericValue : 0), label] as [string, string]
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={stroke} stopOpacity={0.32} />
            <stop offset="95%" stopColor={stroke} stopOpacity={0.04} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="hsl(var(--border) / 0.45)" strokeDasharray="3 3" />

        <XAxis
          dataKey="date"
          tickFormatter={formatDateLabel}
          tick={{ fontSize: 11 }}
          stroke="hsl(var(--muted-foreground))"
        />
        <YAxis
          tick={{ fontSize: 11 }}
          stroke="hsl(var(--muted-foreground))"
          tickFormatter={(value) => formatCurrency(value, true)}
        />

        <Tooltip
          formatter={tooltipFormatter}
          labelFormatter={(value) => new Date(value).toLocaleDateString()}
          contentStyle={{
            borderRadius: 10,
            borderColor: 'hsl(var(--border))',
            backgroundColor: 'hsl(var(--background) / 0.95)',
            fontSize: 12,
          }}
        />

        <Area
          type="monotone"
          dataKey="cumulativePnl"
          stroke={stroke}
          fill={`url(#${gradientId})`}
          strokeWidth={2}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
