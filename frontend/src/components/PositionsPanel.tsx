import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useAtomValue } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  ArrowDownAZ,
  ArrowDownUp,
  ArrowUpAZ,
  Briefcase,
  CheckCircle2,
  CircleDollarSign,
  ExternalLink,
  Gauge,
  Layers,
  Maximize2,
  RefreshCw,
  Search,
  Shield,
  Sigma,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildKalshiMarketUrl, buildPolymarketMarketUrl } from '../lib/marketUrls'
import { selectedAccountIdAtom, themeAtom } from '../store/atoms'
import {
  getAccountPositions,
  getAllTraderOrders,
  getCryptoMarkets,
  getKalshiPositions,
  getKalshiStatus,
  getSimulationAccounts,
  getTradingPositions,
  type CryptoMarket,
  type KalshiAccountStatus,
  type KalshiPosition,
  type SimulationAccount,
  type SimulationPosition,
  type TraderOrder,
  type TradingPosition,
} from '../services/api'
import { CryptoMarketCard } from './CryptoMarketsPanel'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'

type ViewMode = 'all' | 'sandbox' | 'live'
type LiveVenueFilter = 'all' | 'polymarket' | 'kalshi'
type PositionVenue = 'sandbox' | 'autotrader-paper' | 'polymarket-live' | 'kalshi-live'
type PriceMarkMode = 'live' | 'entry_estimate'
type SideFilter = 'all' | 'yes' | 'no' | 'other'
type MarkFilter = 'all' | PriceMarkMode
type SortField = 'exposure' | 'unrealized' | 'pnl_percent' | 'cost_basis' | 'updated' | 'market'
type SortDirection = 'asc' | 'desc'
type ExposureFloor = 'all' | '100' | '500' | '1000' | '5000'

const OPEN_PAPER_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])
const POSITIONS_TABLE_PAGE_SIZE = 100

const VENUE_META: Record<PositionVenue, {
  label: string
}> = {
  sandbox: {
    label: 'Sandbox',
  },
  'autotrader-paper': {
    label: 'Autotrader Paper',
  },
  'polymarket-live': {
    label: 'Polymarket Live',
  },
  'kalshi-live': {
    label: 'Kalshi Live',
  },
}

interface SimulationPositionWithAccount extends SimulationPosition {
  accountName: string
  accountId: string
}

interface SimulationPositionsPayload {
  positions: SimulationPositionWithAccount[]
  failedAccounts: string[]
}

interface PositionRow {
  key: string
  venue: PositionVenue
  venueLabel: string
  accountLabel: string
  marketId: string
  marketQuestion: string
  side: string
  sideLabel: string
  status: string | null
  size: number | null
  entryPrice: number | null
  currentPrice: number | null
  costBasis: number
  marketValue: number
  unrealizedPnl: number | null
  pnlPercent: number | null
  openedAt: string | null
  marketUrl: string | null
  markMode: PriceMarkMode
}

const EMPTY_SIMULATION_PAYLOAD: SimulationPositionsPayload = {
  positions: [],
  failedAccounts: [],
}

function toNumber(value: unknown): number {
  if (typeof value === 'number') return Number.isFinite(value) ? value : 0
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function toTimestamp(value: string | null | undefined): number {
  if (!value) return 0
  const ts = new Date(value).getTime()
  return Number.isFinite(ts) ? ts : 0
}

function readString(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function formatUsd(value: number, decimals = 2): string {
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`
}

function formatCompactUsd(value: number): string {
  if (!Number.isFinite(value)) return '$0'
  return `$${Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value)}`
}

function formatSignedUsd(value: number): string {
  return `${value >= 0 ? '+' : '-'}${formatUsd(Math.abs(value))}`
}

function formatSignedPct(value: number): string {
  return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(2)}%`
}

function formatOptionalPrice(value: number | null): string {
  if (value === null) return 'n/a'
  return `$${value.toFixed(4)}`
}

function formatRelativeTime(value: string | null): string {
  if (!value) return 'n/a'
  const ts = toTimestamp(value)
  if (ts <= 0) return 'n/a'

  const deltaMs = Date.now() - ts
  if (deltaMs < 60_000) return 'just now'

  const minutes = Math.floor(deltaMs / 60_000)
  if (minutes < 60) return `${minutes}m ago`

  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`

  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function normalizeDirection(raw: string | null | undefined): string {
  const direction = String(raw || '').trim().toUpperCase()
  if (!direction) return 'N/A'
  if (direction === 'BUY_YES' || direction === 'SELL_YES') return 'YES'
  if (direction === 'BUY_NO' || direction === 'SELL_NO') return 'NO'
  if (direction === 'BUY' || direction === 'LONG' || direction === 'UP') return 'YES'
  if (direction === 'SELL' || direction === 'SHORT' || direction === 'DOWN') return 'NO'
  return direction
}

function isYesSide(side: string): boolean {
  const normalized = side.trim().toUpperCase()
  return normalized === 'YES' || normalized === 'BUY' || normalized === 'LONG' || normalized === 'UP'
}

function isNoSide(side: string): boolean {
  const normalized = side.trim().toUpperCase()
  return normalized === 'NO' || normalized === 'SELL' || normalized === 'SHORT' || normalized === 'DOWN'
}

function sideBadgeClass(): string {
  return 'border-border/80 bg-muted/60 text-muted-foreground'
}

function exposureFloorValue(value: ExposureFloor): number {
  if (value === 'all') return 0
  return Number(value)
}

function compareNullable(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0
  if (a === null) return 1
  if (b === null) return -1
  return direction === 'asc' ? a - b : b - a
}

export default function PositionsPanel() {
  const globalSelectedAccountId = useAtomValue(selectedAccountIdAtom)

  const [viewMode, setViewMode] = useState<ViewMode>(() => (
    globalSelectedAccountId?.startsWith('live:') ? 'live' : 'all'
  ))
  const [selectedSandboxAccount, setSelectedSandboxAccount] = useState<string | null>(() => {
    if (!globalSelectedAccountId || globalSelectedAccountId.startsWith('live:')) return null
    return globalSelectedAccountId
  })
  const [liveVenueFilter, setLiveVenueFilter] = useState<LiveVenueFilter>(() => {
    if (globalSelectedAccountId === 'live:kalshi') return 'kalshi'
    if (globalSelectedAccountId === 'live:polymarket') return 'polymarket'
    return 'all'
  })

  const [searchQuery, setSearchQuery] = useState('')
  const [sideFilter, setSideFilter] = useState<SideFilter>('all')
  const [markFilter, setMarkFilter] = useState<MarkFilter>('all')
  const [accountFilter, setAccountFilter] = useState<string>('all')
  const [exposureFloor, setExposureFloor] = useState<ExposureFloor>('all')
  const [sortField, setSortField] = useState<SortField>('exposure')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [positionsPage, setPositionsPage] = useState(1)
  const [modalMarket, setModalMarket] = useState<CryptoMarket | null>(null)
  const closeMarketModal = () => setModalMarket(null)
  const themeMode = useAtomValue(themeAtom)

  const { data: cryptoMarkets } = useQuery({
    queryKey: ['crypto-markets-positions'],
    queryFn: () => getCryptoMarkets(),
    refetchInterval: 5000,
    staleTime: 3000,
  })

  const cryptoMarketsMap = useMemo(() => {
    const map = new Map<string, CryptoMarket>()
    if (cryptoMarkets) {
      for (const m of cryptoMarkets) map.set(m.id, m)
    }
    return map
  }, [cryptoMarkets])

  // Market modal: body scroll lock + escape key
  useEffect(() => {
    if (!modalMarket) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeMarketModal() }
    window.addEventListener('keydown', onKey)
    return () => { document.body.style.overflow = prev; window.removeEventListener('keydown', onKey) }
  }, [modalMarket])

  const shouldShowSandbox = viewMode === 'sandbox' || viewMode === 'all'
  const shouldShowLive = viewMode === 'live' || viewMode === 'all'
  const shouldFetchPolymarketLive = shouldShowLive && (liveVenueFilter === 'all' || liveVenueFilter === 'polymarket')
  const shouldFetchKalshiLive = shouldShowLive && (liveVenueFilter === 'all' || liveVenueFilter === 'kalshi')

  const {
    data: accounts = [],
    isLoading: accountsLoading,
    refetch: refetchAccounts,
  } = useQuery<SimulationAccount[]>({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  const simulationAccountKey = useMemo(
    () => accounts.map((account) => account.id).sort().join('|'),
    [accounts]
  )

  const {
    data: simulationPayload = EMPTY_SIMULATION_PAYLOAD,
    isLoading: simulationPositionsLoading,
    refetch: refetchSimulationPositions,
  } = useQuery<SimulationPositionsPayload>({
    queryKey: ['positions-panel', 'simulation-open-positions', selectedSandboxAccount, simulationAccountKey],
    queryFn: async () => {
      if (accounts.length === 0) return EMPTY_SIMULATION_PAYLOAD

      const targetAccounts = selectedSandboxAccount
        ? accounts.filter((account) => account.id === selectedSandboxAccount)
        : accounts

      if (targetAccounts.length === 0) return EMPTY_SIMULATION_PAYLOAD

      const results = await Promise.allSettled(
        targetAccounts.map(async (account) => ({
          account,
          positions: await getAccountPositions(account.id),
        }))
      )

      const positions: SimulationPositionWithAccount[] = []
      const failedAccounts: string[] = []

      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          result.value.positions.forEach((position) => {
            positions.push({
              ...position,
              accountName: result.value.account.name,
              accountId: result.value.account.id,
            })
          })
          return
        }
        failedAccounts.push(targetAccounts[index]?.name || targetAccounts[index]?.id || 'Unknown account')
      })

      return { positions, failedAccounts }
    },
    enabled: shouldShowSandbox && accounts.length > 0,
  })

  const {
    data: traderOrders = [],
    isLoading: traderOrdersLoading,
    refetch: refetchTraderOrders,
  } = useQuery<TraderOrder[]>({
    queryKey: ['positions-panel', 'trader-orders-open-paper'],
    queryFn: async () => {
      try {
        return await getAllTraderOrders(220)
      } catch {
        return []
      }
    },
    enabled: shouldShowSandbox,
    retry: false,
  })

  const {
    data: polymarketLivePositions = [],
    isLoading: polymarketLiveLoading,
    refetch: refetchPolymarketLivePositions,
  } = useQuery<TradingPosition[]>({
    queryKey: ['positions-panel', 'polymarket-live-open-positions'],
    queryFn: async () => {
      try {
        return await getTradingPositions()
      } catch {
        return []
      }
    },
    enabled: shouldFetchPolymarketLive,
    retry: false,
  })

  const {
    data: kalshiStatus,
    isLoading: kalshiStatusLoading,
    refetch: refetchKalshiStatus,
  } = useQuery<KalshiAccountStatus>({
    queryKey: ['kalshi-status'],
    queryFn: getKalshiStatus,
    enabled: shouldFetchKalshiLive,
    retry: false,
  })

  const {
    data: kalshiLivePositions = [],
    isLoading: kalshiLiveLoading,
    refetch: refetchKalshiLivePositions,
  } = useQuery<KalshiPosition[]>({
    queryKey: ['positions-panel', 'kalshi-live-open-positions'],
    queryFn: async () => {
      try {
        return await getKalshiPositions()
      } catch {
        return []
      }
    },
    enabled: shouldFetchKalshiLive && Boolean(kalshiStatus?.authenticated),
    retry: false,
  })

  const simulationRows = useMemo<PositionRow[]>(() => {
    return simulationPayload.positions.map((position) => {
      const currentPrice = position.current_price ?? position.entry_price
      const marketValue = position.quantity * currentPrice
      const costBasis = position.entry_cost
      const unrealizedPnl = position.unrealized_pnl
      const pnlPercent = costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0
      const side = normalizeDirection(position.side)
      return {
        key: `sim:${position.accountId}:${position.id}`,
        venue: 'sandbox',
        venueLabel: 'Sandbox',
        accountLabel: position.accountName,
        marketId: position.market_id,
        marketQuestion: position.market_question,
        side,
        sideLabel: side,
        status: position.status,
        size: position.quantity,
        entryPrice: position.entry_price,
        currentPrice,
        costBasis,
        marketValue,
        unrealizedPnl,
        pnlPercent,
        openedAt: position.opened_at,
        marketUrl: buildPolymarketMarketUrl({
          eventSlug: position.event_slug,
          marketSlug: position.market_slug,
          marketId: position.market_id,
        }),
        markMode: position.current_price === null ? 'entry_estimate' : 'live',
      }
    })
  }, [simulationPayload.positions])

  const accountNameById = useMemo(() => {
    return new Map(accounts.map((account) => [account.id, account.name]))
  }, [accounts])

  const simulationCoverageKeys = useMemo(() => {
    const keys = new Set<string>()
    simulationPayload.positions.forEach((position) => {
      const key = `${position.accountId}:${position.market_id}:${normalizeDirection(position.side)}`
      keys.add(key)
    })
    return keys
  }, [simulationPayload.positions])

  const autotraderPaperRows = useMemo<PositionRow[]>(() => {
    const buckets = new Map<string, {
      marketId: string
      marketQuestion: string
      side: string
      sideLabel: string
      linkedAccountId: string | null
      costBasis: number
      weightedEntry: number
      weightedSize: number
      lastUpdated: string | null
      status: string | null
      marketUrl: string | null
    }>()

    traderOrders.forEach((order) => {
      const mode = String(order.mode || '').toLowerCase()
      const status = String(order.status || '').toLowerCase()
      if (mode !== 'paper' || !OPEN_PAPER_ORDER_STATUSES.has(status)) return

      const marketId = readString(order.market_id) || ''
      if (!marketId) return

      const side = normalizeDirection(order.direction_side ?? order.direction)
      const sideLabel = readString(order.direction_label) || side
      const notional = Math.abs(toNumber(order.notional_usd))
      const entryPrice = toNumber(order.effective_price ?? order.entry_price)
      const payload = (order.payload && typeof order.payload === 'object')
        ? order.payload as Record<string, unknown>
        : {}
      const simulationLedger = (payload.simulation_ledger && typeof payload.simulation_ledger === 'object')
        ? payload.simulation_ledger as Record<string, unknown>
        : null
      const linkedAccountId = readString(simulationLedger?.account_id)
      const bucketScope = linkedAccountId || 'unassigned'
      const key = `${bucketScope}:${marketId}:${side}`

      if (selectedSandboxAccount && linkedAccountId !== selectedSandboxAccount) return

      if (!buckets.has(key)) {
        buckets.set(key, {
          marketId,
          marketQuestion: readString(order.market_question) || marketId,
          side,
          sideLabel,
          linkedAccountId,
          costBasis: 0,
          weightedEntry: 0,
          weightedSize: 0,
          lastUpdated: order.updated_at || order.executed_at || order.created_at || null,
          status: status || null,
          marketUrl: buildPolymarketMarketUrl({
            eventSlug: readString(payload.event_slug),
            marketSlug: readString(payload.market_slug) || readString(payload.market_slug_hint),
            marketId,
          }),
        })
      }

      const bucket = buckets.get(key)
      if (!bucket) return

      if (!bucket.linkedAccountId && linkedAccountId) {
        bucket.linkedAccountId = linkedAccountId
      }
      if (bucket.sideLabel === bucket.side && sideLabel !== side) {
        bucket.sideLabel = sideLabel
      }

      if (notional <= 0) return
      bucket.costBasis += notional
      if (entryPrice > 0 && notional > 0) {
        bucket.weightedEntry += entryPrice * notional
        bucket.weightedSize += notional / entryPrice
      }

      const currentTs = toTimestamp(bucket.lastUpdated)
      const nextTs = toTimestamp(order.updated_at || order.executed_at || order.created_at)
      if (nextTs > currentTs) {
        bucket.lastUpdated = order.updated_at || order.executed_at || order.created_at || bucket.lastUpdated
      }
    })

    return Array.from(buckets.entries())
      .map<PositionRow | null>(([key, bucket]) => {
        if (bucket.costBasis <= 0) return null

        if (bucket.linkedAccountId) {
          const coverageKey = `${bucket.linkedAccountId}:${bucket.marketId}:${bucket.side}`
          if (simulationCoverageKeys.has(coverageKey)) {
            return null
          }
        }

        const entryPrice = bucket.costBasis > 0 ? bucket.weightedEntry / bucket.costBasis : null
        const size = bucket.weightedSize > 0 ? bucket.weightedSize : null
        const accountLabel = bucket.linkedAccountId
          ? (accountNameById.get(bucket.linkedAccountId) || 'Autotrader (Paper)')
          : 'Autotrader (Paper)'
        return {
          key: `paper:${key}`,
          venue: 'autotrader-paper',
          venueLabel: 'Autotrader Paper',
          accountLabel,
          marketId: bucket.marketId,
          marketQuestion: bucket.marketQuestion,
          side: bucket.side,
          sideLabel: bucket.sideLabel,
          status: bucket.status,
          size,
          entryPrice,
          currentPrice: null,
          costBasis: bucket.costBasis,
          marketValue: bucket.costBasis,
          unrealizedPnl: null,
          pnlPercent: null,
          openedAt: bucket.lastUpdated,
          marketUrl: bucket.marketUrl,
          markMode: 'entry_estimate',
        }
      })
      .filter((row): row is PositionRow => row !== null)
      .sort((left, right) => right.marketValue - left.marketValue)
  }, [accountNameById, selectedSandboxAccount, simulationCoverageKeys, traderOrders])

  const polymarketLiveRows = useMemo<PositionRow[]>(() => {
    return polymarketLivePositions.map((position) => {
      const costBasis = position.size * position.average_cost
      const marketValue = position.size * position.current_price
      const unrealizedPnl = position.unrealized_pnl
      const pnlPercent = costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0
      const side = normalizeDirection(position.outcome)
      return {
        key: `pm-live:${position.market_id}:${position.token_id}:${position.outcome}`,
        venue: 'polymarket-live',
        venueLabel: 'Polymarket Live',
        accountLabel: 'Polymarket',
        marketId: position.market_id,
        marketQuestion: position.market_question,
        side,
        sideLabel: side,
        status: 'open',
        size: position.size,
        entryPrice: position.average_cost,
        currentPrice: position.current_price,
        costBasis,
        marketValue,
        unrealizedPnl,
        pnlPercent,
        openedAt: null,
        marketUrl: buildPolymarketMarketUrl({
          eventSlug: position.event_slug,
          marketSlug: position.market_slug,
          marketId: position.market_id,
        }),
        markMode: 'live',
      }
    })
  }, [polymarketLivePositions])

  const kalshiLiveRows = useMemo<PositionRow[]>(() => {
    return kalshiLivePositions.map((position) => {
      const costBasis = position.size * position.average_cost
      const marketValue = position.size * position.current_price
      const unrealizedPnl = position.unrealized_pnl
      const pnlPercent = costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0
      const side = normalizeDirection(position.outcome)
      return {
        key: `kalshi-live:${position.market_id}:${position.token_id}:${position.outcome}`,
        venue: 'kalshi-live',
        venueLabel: 'Kalshi Live',
        accountLabel: 'Kalshi',
        marketId: position.market_id,
        marketQuestion: position.market_question,
        side,
        sideLabel: side,
        status: 'open',
        size: position.size,
        entryPrice: position.average_cost,
        currentPrice: position.current_price,
        costBasis,
        marketValue,
        unrealizedPnl,
        pnlPercent,
        openedAt: null,
        marketUrl: buildKalshiMarketUrl({
          marketTicker: position.market_id,
          eventTicker: position.event_slug,
        }),
        markMode: 'live',
      }
    })
  }, [kalshiLivePositions])

  const baseRows = useMemo(() => {
    if (viewMode === 'sandbox') return [...simulationRows, ...autotraderPaperRows]
    if (viewMode === 'live') return [...polymarketLiveRows, ...kalshiLiveRows]
    return [...simulationRows, ...autotraderPaperRows, ...polymarketLiveRows, ...kalshiLiveRows]
  }, [viewMode, simulationRows, autotraderPaperRows, polymarketLiveRows, kalshiLiveRows])

  const accountOptions = useMemo(() => {
    return Array.from(new Set(baseRows.map((row) => row.accountLabel))).sort((left, right) => left.localeCompare(right))
  }, [baseRows])

  const effectiveAccountFilter = accountOptions.includes(accountFilter) ? accountFilter : 'all'

  const filteredRows = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    const minExposure = exposureFloorValue(exposureFloor)

    return baseRows.filter((row) => {
      if (query) {
        const haystack = `${row.marketQuestion} ${row.marketId} ${row.accountLabel} ${row.venueLabel} ${row.side} ${row.sideLabel}`.toLowerCase()
        if (!haystack.includes(query)) return false
      }

      if (sideFilter === 'yes' && !isYesSide(row.side)) return false
      if (sideFilter === 'no' && !isNoSide(row.side)) return false
      if (sideFilter === 'other' && (isYesSide(row.side) || isNoSide(row.side))) return false
      if (markFilter !== 'all' && row.markMode !== markFilter) return false
      if (effectiveAccountFilter !== 'all' && row.accountLabel !== effectiveAccountFilter) return false
      if (row.marketValue < minExposure) return false

      return true
    })
  }, [baseRows, searchQuery, sideFilter, markFilter, effectiveAccountFilter, exposureFloor])

  const sortedRows = useMemo(() => {
    const rows = [...filteredRows]

    rows.sort((left, right) => {
      if (sortField === 'exposure') {
        return sortDirection === 'asc'
          ? left.marketValue - right.marketValue
          : right.marketValue - left.marketValue
      }

      if (sortField === 'cost_basis') {
        return sortDirection === 'asc'
          ? left.costBasis - right.costBasis
          : right.costBasis - left.costBasis
      }

      if (sortField === 'unrealized') {
        const delta = compareNullable(left.unrealizedPnl, right.unrealizedPnl, sortDirection)
        if (delta !== 0) return delta
        return right.marketValue - left.marketValue
      }

      if (sortField === 'pnl_percent') {
        const delta = compareNullable(left.pnlPercent, right.pnlPercent, sortDirection)
        if (delta !== 0) return delta
        return right.marketValue - left.marketValue
      }

      if (sortField === 'updated') {
        const leftTs = toTimestamp(left.openedAt)
        const rightTs = toTimestamp(right.openedAt)
        return sortDirection === 'asc' ? leftTs - rightTs : rightTs - leftTs
      }

      const lexical = left.marketQuestion.localeCompare(right.marketQuestion)
      if (lexical !== 0) return sortDirection === 'asc' ? lexical : -lexical
      return right.marketValue - left.marketValue
    })

    return rows
  }, [filteredRows, sortField, sortDirection])

  const positionsPageCount = useMemo(
    () => Math.max(1, Math.ceil(sortedRows.length / POSITIONS_TABLE_PAGE_SIZE)),
    [sortedRows.length]
  )

  useEffect(() => {
    setPositionsPage((current) => Math.min(current, positionsPageCount))
  }, [positionsPageCount])

  const pagedRows = useMemo(() => {
    const start = (positionsPage - 1) * POSITIONS_TABLE_PAGE_SIZE
    return sortedRows.slice(start, start + POSITIONS_TABLE_PAGE_SIZE)
  }, [positionsPage, sortedRows])

  const metrics = useMemo(() => {
    const totalCostBasis = sortedRows.reduce((sum, row) => sum + row.costBasis, 0)
    const totalMarketValue = sortedRows.reduce((sum, row) => sum + row.marketValue, 0)
    const markableRows = sortedRows.filter((row) => row.unrealizedPnl !== null)
    const markableCostBasis = markableRows.reduce((sum, row) => sum + row.costBasis, 0)
    const totalUnrealizedPnl = markableRows.reduce((sum, row) => sum + (row.unrealizedPnl ?? 0), 0)
    const pnlPercent = markableCostBasis > 0 ? (totalUnrealizedPnl / markableCostBasis) * 100 : 0
    const markCoverage = sortedRows.length > 0 ? (markableRows.length / sortedRows.length) * 100 : 0

    const yesExposure = sortedRows
      .filter((row) => isYesSide(row.side))
      .reduce((sum, row) => sum + row.marketValue, 0)
    const noExposure = sortedRows
      .filter((row) => isNoSide(row.side))
      .reduce((sum, row) => sum + row.marketValue, 0)
    const otherExposure = Math.max(0, totalMarketValue - yesExposure - noExposure)

    const directionalNet = yesExposure - noExposure

    const largestPosition = sortedRows.reduce<PositionRow | null>(
      (max, row) => (max && max.marketValue > row.marketValue ? max : row),
      null
    )

    const concentrationHhi = totalMarketValue > 0
      ? sortedRows.reduce((sum, row) => {
          const weight = row.marketValue / totalMarketValue
          return sum + (weight * weight)
        }, 0)
      : 0

    return {
      totalCostBasis,
      totalMarketValue,
      totalUnrealizedPnl,
      pnlPercent,
      markCoverage,
      yesExposure,
      noExposure,
      otherExposure,
      directionalNet,
      largestPosition,
      concentrationHhi,
    }
  }, [sortedRows])

  const sourceBreakdown = useMemo(() => {
    const totalExposure = sortedRows.reduce((sum, row) => sum + row.marketValue, 0)
    const order: PositionVenue[] = ['sandbox', 'autotrader-paper', 'polymarket-live', 'kalshi-live']

    return order.map((venue) => {
      const rows = sortedRows.filter((row) => row.venue === venue)
      const exposure = rows.reduce((sum, row) => sum + row.marketValue, 0)
      const knownPnl = rows
        .filter((row) => row.unrealizedPnl !== null)
        .reduce((sum, row) => sum + (row.unrealizedPnl ?? 0), 0)
      const share = totalExposure > 0 ? (exposure / totalExposure) * 100 : 0
      const liveMarks = rows.filter((row) => row.markMode === 'live').length
      return {
        venue,
        label: VENUE_META[venue].label,
        rows: rows.length,
        exposure,
        knownPnl,
        share,
        liveMarks,
      }
    })
  }, [sortedRows])

  const totalExposure = metrics.totalMarketValue

  useEffect(() => {
    setPositionsPage(1)
  }, [viewMode, liveVenueFilter, selectedSandboxAccount, searchQuery, sideFilter, markFilter, effectiveAccountFilter, exposureFloor, sortField, sortDirection])

  const isLoading = (
    (shouldShowSandbox && (accountsLoading || simulationPositionsLoading || traderOrdersLoading))
    || (shouldFetchPolymarketLive && polymarketLiveLoading)
    || (shouldFetchKalshiLive && (kalshiStatusLoading || (Boolean(kalshiStatus?.authenticated) && kalshiLiveLoading)))
  )

  const handleRefresh = () => {
    if (shouldShowSandbox) {
      void refetchAccounts()
      void refetchSimulationPositions()
      void refetchTraderOrders()
    }
    if (shouldFetchPolymarketLive) {
      void refetchPolymarketLivePositions()
    }
    if (shouldFetchKalshiLive) {
      void refetchKalshiStatus()
      if (kalshiStatus?.authenticated) {
        void refetchKalshiLivePositions()
      }
    }
  }

  const clearFilters = () => {
    setSearchQuery('')
    setSideFilter('all')
    setMarkFilter('all')
    setAccountFilter('all')
    setExposureFloor('all')
    setSortField('exposure')
    setSortDirection('desc')
    setPositionsPage(1)
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-1.5">
      {/* Control Strip */}
      <div className="shrink-0 space-y-2">
        {/* Row 1: Title + badges + refresh */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <Briefcase className="w-4 h-4 text-blue-400" />
              Positions
            </h2>
            <Badge className="rounded-md border-transparent bg-muted text-muted-foreground text-xs">
              {sortedRows.length} / {baseRows.length}
            </Badge>
            {sortedRows.length > 0 && (
              <Badge className={cn('rounded-md border-transparent text-xs', metrics.totalUnrealizedPnl >= 0 ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300')}>
                {formatSignedUsd(metrics.totalUnrealizedPnl)}
              </Badge>
            )}
          </div>
          <Button variant="ghost" size="sm" onClick={handleRefresh} disabled={isLoading} className="h-7 px-2">
            <RefreshCw className={cn('w-3.5 h-3.5', isLoading && 'animate-spin')} />
          </Button>
        </div>

        {/* Row 2: View tabs + venue selectors */}
        <div className="flex flex-wrap items-center gap-2">
          <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as ViewMode)}>
            <TabsList className="h-8">
              <TabsTrigger value="all" className="text-xs h-7 px-3">All</TabsTrigger>
              <TabsTrigger value="sandbox" className="text-xs h-7 px-3">Sandbox</TabsTrigger>
              <TabsTrigger value="live" className="text-xs h-7 px-3">Live</TabsTrigger>
            </TabsList>
          </Tabs>

          {shouldShowSandbox && (
            <Select value={selectedSandboxAccount ?? 'all'} onValueChange={(value) => setSelectedSandboxAccount(value === 'all' ? null : value)}>
              <SelectTrigger className="h-8 w-[180px] bg-background/60 text-xs">
                <SelectValue placeholder="Sandbox Account" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sandbox</SelectItem>
                {accounts.map((account) => (
                  <SelectItem key={account.id} value={account.id}>{account.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          {shouldShowLive && (
            <Select value={liveVenueFilter} onValueChange={(value) => setLiveVenueFilter(value as LiveVenueFilter)}>
              <SelectTrigger className="h-8 w-[160px] bg-background/60 text-xs">
                <SelectValue placeholder="Live Venue" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Live</SelectItem>
                <SelectItem value="polymarket">Polymarket</SelectItem>
                <SelectItem value="kalshi">Kalshi</SelectItem>
              </SelectContent>
            </Select>
          )}
        </div>

        {/* Row 3: Filters */}
        <div className="flex flex-wrap items-end gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search..."
              className="h-7 w-[180px] bg-background/60 pl-7 text-xs"
            />
          </div>

          <Select value={sideFilter} onValueChange={(value) => setSideFilter(value as SideFilter)}>
            <SelectTrigger className="h-7 w-[100px] bg-background/60 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sides</SelectItem>
              <SelectItem value="yes">YES</SelectItem>
              <SelectItem value="no">NO</SelectItem>
              <SelectItem value="other">Other</SelectItem>
            </SelectContent>
          </Select>

          <Select value={markFilter} onValueChange={(value) => setMarkFilter(value as MarkFilter)}>
            <SelectTrigger className="h-7 w-[110px] bg-background/60 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Marks</SelectItem>
              <SelectItem value="live">Live</SelectItem>
              <SelectItem value="entry_estimate">Estimated</SelectItem>
            </SelectContent>
          </Select>

          <Select value={effectiveAccountFilter} onValueChange={setAccountFilter}>
            <SelectTrigger className="h-7 w-[120px] bg-background/60 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Desks</SelectItem>
              {accountOptions.map((account) => (
                <SelectItem key={account} value={account}>{account}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={exposureFloor} onValueChange={(value) => setExposureFloor(value as ExposureFloor)}>
            <SelectTrigger className="h-7 w-[90px] bg-background/60 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Any Size</SelectItem>
              <SelectItem value="100">$100+</SelectItem>
              <SelectItem value="500">$500+</SelectItem>
              <SelectItem value="1000">$1K+</SelectItem>
              <SelectItem value="5000">$5K+</SelectItem>
            </SelectContent>
          </Select>

          <div className="flex items-center gap-1">
            <Select value={sortField} onValueChange={(value) => setSortField(value as SortField)}>
              <SelectTrigger className="h-7 w-[100px] bg-background/60 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="exposure">Exposure</SelectItem>
                <SelectItem value="unrealized">Unrealized</SelectItem>
                <SelectItem value="pnl_percent">P&L %</SelectItem>
                <SelectItem value="cost_basis">Cost Basis</SelectItem>
                <SelectItem value="updated">Updated</SelectItem>
                <SelectItem value="market">Market</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              className="h-7 w-7 px-0"
              onClick={() => setSortDirection((current) => current === 'asc' ? 'desc' : 'asc')}
            >
              {sortDirection === 'asc' ? <ArrowUpAZ className="w-3.5 h-3.5" /> : <ArrowDownAZ className="w-3.5 h-3.5" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Metric Strip */}
      {!isLoading && sortedRows.length > 0 && (
        <div className="shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 border-y border-border/50 py-1.5 px-0.5">
          <MetricChip icon={<Layers className="w-3.5 h-3.5 text-blue-300" />} label="Risk" value={sortedRows.length.toString()} />
          <MetricChip icon={<CircleDollarSign className="w-3.5 h-3.5 text-blue-300" />} label="Exposure" value={formatCompactUsd(metrics.totalMarketValue)} detail={formatUsd(metrics.totalMarketValue)} />
          <MetricChip icon={<Shield className="w-3.5 h-3.5 text-amber-300" />} label="Cost" value={formatCompactUsd(metrics.totalCostBasis)} detail={formatUsd(metrics.totalCostBasis)} />
          <MetricChip
            icon={metrics.totalUnrealizedPnl >= 0 ? <TrendingUp className="w-3.5 h-3.5 text-emerald-300" /> : <TrendingDown className="w-3.5 h-3.5 text-red-300" />}
            label="Unrealized"
            value={formatSignedUsd(metrics.totalUnrealizedPnl)}
            detail={formatSignedPct(metrics.pnlPercent)}
            valueClassName={metrics.totalUnrealizedPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
          />
          <MetricChip
            icon={<Sigma className="w-3.5 h-3.5 text-cyan-300" />}
            label="Net"
            value={formatSignedUsd(metrics.directionalNet)}
            valueClassName={metrics.directionalNet >= 0 ? 'text-emerald-300' : 'text-red-300'}
          />
          <MetricChip
            icon={<Gauge className="w-3.5 h-3.5 text-purple-300" />}
            label="HHI"
            value={`${(metrics.concentrationHhi * 100).toFixed(1)}%`}
            valueClassName={metrics.concentrationHhi >= 0.24 ? 'text-red-300' : metrics.concentrationHhi >= 0.14 ? 'text-amber-300' : 'text-emerald-300'}
          />
          <MetricChip
            icon={<CheckCircle2 className="w-3.5 h-3.5 text-emerald-300" />}
            label="Mark"
            value={`${metrics.markCoverage.toFixed(0)}%`}
            valueClassName={metrics.markCoverage >= 80 ? 'text-emerald-300' : metrics.markCoverage >= 45 ? 'text-amber-300' : 'text-red-300'}
          />
          <MetricChip
            icon={<Target className="w-3.5 h-3.5 text-orange-300" />}
            label="Largest"
            value={metrics.largestPosition ? formatCompactUsd(metrics.largestPosition.marketValue) : '$0'}
            detail={metrics.largestPosition?.marketQuestion}
          />
        </div>
      )}

      {/* Main content — fills remaining space */}
      <div className="flex-1 min-h-0">
        {isLoading ? (
          <div className="flex h-full items-center justify-center">
            <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : sortedRows.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center">
            <Briefcase className="w-10 h-10 text-muted-foreground/50 mb-3" />
            <p className="text-sm text-muted-foreground">No positions match the current filters.</p>
            <Button variant="outline" size="sm" className="mt-3" onClick={clearFilters}>Clear filters</Button>
          </div>
        ) : (
          <div className="h-full grid gap-2 xl:grid-cols-[minmax(0,1fr)_280px]">
            {/* Left: Position Table */}
            <div className="flex min-h-0 flex-col gap-1.5">
              <div className="shrink-0 flex items-center justify-between px-1">
                <span className="text-[10px] font-mono text-muted-foreground">
                  {sortedRows.length} rows • {POSITIONS_TABLE_PAGE_SIZE}/page
                </span>
                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-5 px-2 text-[10px]"
                    onClick={() => setPositionsPage((page) => Math.max(1, page - 1))}
                    disabled={positionsPage <= 1}
                  >
                    Prev
                  </Button>
                  <span className="min-w-[64px] text-center text-[10px] font-mono text-muted-foreground">
                    {positionsPage}/{positionsPageCount}
                  </span>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-5 px-2 text-[10px]"
                    onClick={() => setPositionsPage((page) => Math.min(positionsPageCount, page + 1))}
                    disabled={positionsPage >= positionsPageCount}
                  >
                    Next
                  </Button>
                </div>
              </div>
              <ScrollArea className="flex-1 min-h-0 rounded-md border border-border/60 bg-card/60">
                <div className="w-full overflow-x-auto">
                  <Table className="min-w-[1240px]">
                    <TableHeader>
                      <TableRow>
                        <TableHead className="text-[11px]">Market</TableHead>
                        <TableHead className="text-[11px]">L</TableHead>
                        <TableHead className="text-[11px]">Dir</TableHead>
                        <TableHead className="text-[11px] text-right">Exposure</TableHead>
                        <TableHead className="text-[11px] text-right">Avg Px</TableHead>
                        <TableHead className="text-[11px] text-right">Mark</TableHead>
                        <TableHead className="text-[11px] text-right">U-P&amp;L</TableHead>
                        <TableHead className="text-[11px] text-right">Edge</TableHead>
                        <TableHead className="text-[11px] text-right">Conf</TableHead>
                        <TableHead className="text-[11px] text-right">Orders</TableHead>
                        <TableHead className="text-[11px] text-right">Mode</TableHead>
                        <TableHead className="text-[11px]">Updated</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pagedRows.map((row) => {
                        const hasLiveMark = row.markMode === 'live' && row.currentPrice !== null
                        const modeLabel = row.venue === 'polymarket-live' || row.venue === 'kalshi-live' ? 'LIVE' : 'PAPER'
                        const exposureShare = totalExposure > 0 ? (row.marketValue / totalExposure) * 100 : 0
                        return (
                          <TableRow key={row.key} className="text-xs hover:bg-muted/30">
                            <TableCell className="max-w-[260px] truncate py-1" title={row.marketQuestion}>
                              <p className="truncate">{row.marketQuestion}</p>
                              <p className="text-[10px] text-muted-foreground truncate" title={`${row.marketId} • ${row.accountLabel} • ${row.venueLabel}${row.status ? ` • ${row.status}` : ''}`}>
                                {row.marketId} • {row.accountLabel} • {row.venueLabel}{row.status ? ` • ${row.status}` : ''}
                              </p>
                            </TableCell>
                            <TableCell className="py-1">
                              <div className="flex items-center gap-1">
                                {row.marketUrl ? (
                                  <a
                                    href={row.marketUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                    title="Open market"
                                  >
                                    <ExternalLink className="h-3 w-3" />
                                  </a>
                                ) : null}
                                {cryptoMarketsMap.has(row.marketId) ? (
                                  <button
                                    onClick={() => setModalMarket(cryptoMarketsMap.get(row.marketId)!)}
                                    className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                    title="View live market"
                                  >
                                    <Maximize2 className="h-3 w-3" />
                                  </button>
                                ) : null}
                                {!row.marketUrl && !cryptoMarketsMap.has(row.marketId) ? (
                                  <span className="text-[9px] text-muted-foreground">—</span>
                                ) : null}
                              </div>
                            </TableCell>
                            <TableCell className="py-1">
                              <Badge variant="outline" className={cn('h-5 max-w-[140px] truncate border-border/80 bg-muted/60 px-1.5 text-[10px] text-muted-foreground', sideBadgeClass())} title={row.sideLabel}>
                                {row.sideLabel}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right font-mono py-1">
                              {formatUsd(row.marketValue)}
                              <div className="text-[9px] text-muted-foreground">{exposureShare.toFixed(1)}%</div>
                            </TableCell>
                            <TableCell className="text-right font-mono py-1">{formatOptionalPrice(row.entryPrice)}</TableCell>
                            <TableCell className={cn('text-right font-mono py-1', hasLiveMark && 'text-sky-300')}>
                              {formatOptionalPrice(row.currentPrice)}
                            </TableCell>
                            <TableCell className={cn('text-right font-mono py-1', (row.unrealizedPnl || 0) > 0 ? 'text-emerald-500' : (row.unrealizedPnl || 0) < 0 ? 'text-red-500' : '')}>
                              {row.unrealizedPnl !== null ? formatSignedUsd(row.unrealizedPnl) : '—'}
                            </TableCell>
                            <TableCell className="text-right font-mono py-1">
                              {row.pnlPercent !== null ? formatSignedPct(row.pnlPercent) : '—'}
                            </TableCell>
                            <TableCell className="text-right font-mono py-1">—</TableCell>
                            <TableCell className="text-right font-mono py-1">—</TableCell>
                            <TableCell className="text-right font-mono py-1">{modeLabel}</TableCell>
                            <TableCell className="py-1 text-[10px] text-muted-foreground">
                              {formatRelativeTime(row.openedAt)}
                            </TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </div>
              </ScrollArea>
            </div>

            {/* Right: Analytics Sidebar */}
            <div className="hidden xl:flex flex-col gap-2 min-h-0">
              <ScrollArea className="h-full">
                <div className="space-y-3 pr-2">
                  {/* Book Contribution */}
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <p className="text-xs font-semibold">Book Contribution</p>
                      <Badge className="rounded-md border-transparent bg-muted text-muted-foreground text-[10px]">
                        {sourceBreakdown.filter((row) => row.rows > 0).length} active
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      {sourceBreakdown.map((book) => (
                        <BreakdownLane
                          key={book.venue}
                          label={book.label}
                          rowCount={book.rows}
                          share={book.share}
                          exposure={book.exposure}
                          pnl={book.knownPnl}
                          liveMarks={book.liveMarks}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Directional Pressure */}
                  <div className="border-t border-border/50 pt-3">
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <p className="text-xs font-semibold">Directional Pressure</p>
                      <ArrowDownUp className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                    <div className="space-y-2">
                      <PressureLane label="YES" value={metrics.yesExposure} total={totalExposure} tone="green" />
                      <PressureLane label="NO" value={metrics.noExposure} total={totalExposure} tone="red" />
                      <PressureLane label="OTHER" value={metrics.otherExposure} total={totalExposure} tone="neutral" />
                    </div>
                    <div className="mt-3 rounded-lg border border-border/60 bg-muted/25 p-2">
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Net Bias</p>
                      <p className={cn('mt-0.5 text-sm font-semibold font-mono', metrics.directionalNet >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                        {formatSignedUsd(metrics.directionalNet)}
                      </p>
                    </div>
                  </div>
                </div>
              </ScrollArea>
            </div>
          </div>
        )}
      </div>

      {/* Error alerts */}
      {simulationPayload.failedAccounts.length > 0 && (
        <div className="shrink-0 flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-300 shrink-0" />
          Failed: {simulationPayload.failedAccounts.join(', ')}
        </div>
      )}

      {/* Market modal portal */}
      {typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {modalMarket && (
            <motion.div
              key={`position-market-modal-${modalMarket.id}`}
              className="fixed inset-0 z-[120] flex items-center justify-center p-4 sm:p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={closeMarketModal}
                aria-hidden
              />
              <motion.div
                className="relative z-10"
                role="dialog"
                aria-modal="true"
                initial={{ scale: 0.94, opacity: 0, y: 22 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 14 }}
                transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}
              >
                <CryptoMarketCard
                  market={modalMarket}
                  themeMode={themeMode}
                  isModalView
                  onCloseModal={closeMarketModal}
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </div>
  )
}

function MetricChip({
  icon,
  label,
  value,
  detail,
  valueClassName,
}: {
  icon: ReactNode
  label: string
  value: string
  detail?: string
  valueClassName?: string
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs" title={detail || undefined}>
      {icon}
      <span className="text-muted-foreground">{label}</span>
      <span className={cn('font-mono font-semibold', valueClassName)}>{value}</span>
    </div>
  )
}

function BreakdownLane({
  label,
  rowCount,
  share,
  exposure,
  pnl,
  liveMarks,
}: {
  label: string
  rowCount: number
  share: number
  exposure: number
  pnl: number
  liveMarks: number
}) {
  return (
    <div className="rounded-lg border border-border/75 bg-muted/25 px-2.5 py-2">
      <div className="flex items-center justify-between gap-2 text-xs">
        <div className="min-w-0">
          <p className="truncate">{label}</p>
          <p className="text-[10px] text-muted-foreground">{rowCount} rows · {liveMarks} live marks</p>
        </div>
        <div className="text-right">
          <p className="font-mono">{formatCompactUsd(exposure)}</p>
          <p className={cn('text-[10px] font-mono', pnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
            {formatSignedUsd(pnl)}
          </p>
        </div>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-blue-400" style={{ width: `${Math.max(2, Math.min(100, share))}%` }} />
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground">{share.toFixed(1)}% share</div>
    </div>
  )
}

function PressureLane({
  label,
  value,
  total,
  tone,
}: {
  label: string
  value: number
  total: number
  tone: 'green' | 'red' | 'neutral'
}) {
  const share = total > 0 ? (value / total) * 100 : 0
  const toneClass = tone === 'green' ? 'bg-emerald-400' : tone === 'red' ? 'bg-red-400' : 'bg-slate-400'

  return (
    <div>
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{formatCompactUsd(value)}</span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
        <div className={cn('h-full rounded-full', toneClass)} style={{ width: `${Math.max(2, Math.min(100, share))}%` }} />
      </div>
      <div className="mt-0.5 text-[10px] text-muted-foreground">{share.toFixed(1)}%</div>
    </div>
  )
}
