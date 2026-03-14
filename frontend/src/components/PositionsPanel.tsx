import { type ReactNode, useDeferredValue, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useAtomValue } from 'jotai'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Liveline, type WindowOption } from 'liveline'
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
  Loader2,
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
  adoptTraderLiveWalletPosition,
  getAccountPositions,
  getAllTraderOrders,
  getCryptoMarkets,
  getKalshiPositions,
  getKalshiStatus,
  getSimulationAccounts,
  getTraderMarketHistory,
  getTraders,
  getTradingPositions,
  type CryptoMarket,
  type KalshiAccountStatus,
  type KalshiPosition,
  type SimulationAccount,
  type SimulationPosition,
  type Trader,
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
import { FlashNumber } from './AnimatedNumber'

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
const OPEN_LIVE_MANAGED_ORDER_STATUSES = new Set([
  'pending',
  'submitted',
  'open',
  'partially_filled',
  'executed',
])
const POSITIONS_TABLE_PAGE_SIZE = 100
const LIVE_MARK_FRESH_MS = 15_000
const MODAL_MARKET_HISTORY_LIMIT = 960
const MODAL_MARKET_HISTORY_REFRESH_MS = 10_000
const LIVELINE_WINDOW_PRESETS: WindowOption[] = [
  { label: 'All', secs: 60 * 60 * 24 * 365 * 10 },
  { label: '7d', secs: 60 * 60 * 24 * 7 },
  { label: '3d', secs: 60 * 60 * 24 * 3 },
  { label: '24h', secs: 60 * 60 * 24 },
  { label: '6h', secs: 60 * 60 * 6 },
  { label: '1h', secs: 60 * 60 },
  { label: '15m', secs: 60 * 15 },
]

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
  markUpdatedAt: string | null
  markFresh: boolean
  tokenId: string | null
  managedByBot: string | null
  managedBotId: string | null
  managedOrderId: string | null
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

function readTimestamp(record: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = readString(record[key])
    if (value) return value
  }
  return null
}

function normalizeTokenKey(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

function extractManagedTokenFromOrder(order: TraderOrder): string | null {
  const payload = (order.payload && typeof order.payload === 'object')
    ? order.payload as Record<string, unknown>
    : {}
  const liveMarket = (payload.live_market && typeof payload.live_market === 'object')
    ? payload.live_market as Record<string, unknown>
    : {}
  const providerReconciliation = (payload.provider_reconciliation && typeof payload.provider_reconciliation === 'object')
    ? payload.provider_reconciliation as Record<string, unknown>
    : {}
  const providerSnapshot = (providerReconciliation.snapshot && typeof providerReconciliation.snapshot === 'object')
    ? providerReconciliation.snapshot as Record<string, unknown>
    : {}

  const candidates: unknown[] = [
    payload.selected_token_id,
    payload.token_id,
    payload.asset_id,
    liveMarket.selected_token_id,
    liveMarket.token_id,
    liveMarket.asset_id,
    providerSnapshot.token_id,
    providerSnapshot.asset_id,
  ]
  for (const candidate of candidates) {
    const token = normalizeTokenKey(candidate)
    if (token) return token
  }
  return null
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

function readErrorMessage(error: unknown, fallback: string): string {
  if (typeof error === 'object' && error) {
    const candidate = error as {
      message?: unknown
      response?: {
        data?: {
          detail?: unknown
          message?: unknown
          error?: unknown
        }
      }
    }
    const responseData = candidate.response?.data
    const detail = String(
      responseData?.detail
      ?? responseData?.message
      ?? responseData?.error
      ?? candidate.message
      ?? ''
    ).trim()
    if (detail) return detail
  }
  return fallback
}

function readFinite(...values: unknown[]): number | null {
  for (const value of values) {
    if (value === null || value === undefined || value === '') continue
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function toUnixSeconds(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (value > 1e12) return Math.trunc(value / 1000)
  if (value > 1e10) return Math.trunc(value / 1000)
  return Math.trunc(value)
}

function readHistoryPoint(point: unknown): { time: number; yes: number | null; no: number | null } | null {
  if (Array.isArray(point)) {
    const rawTime = readFinite(point[0])
    if (rawTime === null || rawTime <= 0) return null
    let yes = readFinite(point[1])
    let no = readFinite(point[2])
    if (yes === null && no !== null) yes = 1 - no
    if (no === null && yes !== null) no = 1 - yes
    return {
      time: Math.max(1, toUnixSeconds(rawTime)),
      yes,
      no,
    }
  }

  if (!point || typeof point !== 'object') return null
  const row = point as Record<string, unknown>
  const rawTime = readFinite(row.t, row.ts, row.time, row.timestamp)
  if (rawTime === null || rawTime <= 0) return null

  const outcomePrices = Array.isArray(row.outcome_prices)
    ? row.outcome_prices
    : Array.isArray(row.outcomePrices)
      ? row.outcomePrices
      : null

  let yes = readFinite(row.yes, row.idx_0, outcomePrices?.[0])
  let no = readFinite(row.no, row.idx_1, outcomePrices?.[1])
  if (yes === null && no !== null) yes = 1 - no
  if (no === null && yes !== null) no = 1 - yes

  return {
    time: Math.max(1, toUnixSeconds(rawTime)),
    yes,
    no,
  }
}

export default function PositionsPanel() {
  const queryClient = useQueryClient()
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
  const deferredSearchQuery = useDeferredValue(searchQuery)
  const [sideFilter, setSideFilter] = useState<SideFilter>('all')
  const [markFilter, setMarkFilter] = useState<MarkFilter>('all')
  const [accountFilter, setAccountFilter] = useState<string>('all')
  const [exposureFloor, setExposureFloor] = useState<ExposureFloor>('all')
  const [sortField, setSortField] = useState<SortField>('exposure')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [positionsPage, setPositionsPage] = useState(1)
  const [modalMarket, setModalMarket] = useState<CryptoMarket | null>(null)
  const [modalRow, setModalRow] = useState<PositionRow | null>(null)
  const [assignLiveTraderId, setAssignLiveTraderId] = useState('')
  const [adoptError, setAdoptError] = useState<string | null>(null)
  const [adoptSuccess, setAdoptSuccess] = useState<string | null>(null)
  const closeMarketModal = () => setModalMarket(null)
  const closeRowModal = () => {
    setModalRow(null)
    setAdoptError(null)
    setAdoptSuccess(null)
  }
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

  // Modal stack: body scroll lock + escape key
  useEffect(() => {
    if (!modalMarket && !modalRow) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (modalMarket) {
        closeMarketModal()
        return
      }
      closeRowModal()
    }
    window.addEventListener('keydown', onKey)
    return () => { document.body.style.overflow = prev; window.removeEventListener('keydown', onKey) }
  }, [modalMarket, modalRow])

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
    queryKey: ['positions-panel', 'trader-orders-open'],
    queryFn: async () => {
      try {
        return await getAllTraderOrders(220)
      } catch {
        return []
      }
    },
    enabled: shouldShowSandbox || shouldShowLive,
    refetchInterval: shouldShowLive ? 2000 : (shouldShowSandbox ? 3000 : false),
    staleTime: 0,
    refetchOnMount: 'always',
    retry: false,
  })

  const {
    data: liveTraders = [],
  } = useQuery<Trader[]>({
    queryKey: ['positions-panel', 'live-traders'],
    queryFn: () => getTraders({ mode: 'live' }),
    enabled: shouldShowLive,
    refetchInterval: shouldShowLive ? 5000 : false,
    staleTime: 0,
    refetchOnMount: 'always',
    retry: false,
  })

  const {
    data: polymarketLivePositions = [],
    dataUpdatedAt: polymarketLiveUpdatedAt,
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
    refetchInterval: shouldFetchPolymarketLive ? 2000 : false,
    staleTime: 0,
    refetchOnMount: 'always',
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
    dataUpdatedAt: kalshiLiveUpdatedAt,
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
    refetchInterval: shouldFetchKalshiLive && Boolean(kalshiStatus?.authenticated) ? 5000 : false,
    staleTime: 0,
    refetchOnMount: 'always',
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
        markUpdatedAt: position.current_price === null ? null : position.opened_at,
        markFresh: false,
        tokenId: position.token_id ?? null,
        managedByBot: null,
        managedBotId: null,
        managedOrderId: null,
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

  const liveTraderNameById = useMemo(() => {
    return new Map(liveTraders.map((trader) => [trader.id, trader.name]))
  }, [liveTraders])

  const sortedLiveTraders = useMemo(
    () => [...liveTraders].sort((left, right) => left.name.localeCompare(right.name)),
    [liveTraders]
  )

  const openRowModal = (row: PositionRow) => {
    setModalRow(row)
    setAdoptError(null)
    setAdoptSuccess(null)
  }

  useEffect(() => {
    if (!modalRow || modalRow.venue !== 'polymarket-live' || modalRow.managedBotId) return
    setAssignLiveTraderId((current) => {
      if (current && liveTraderNameById.has(current)) return current
      return sortedLiveTraders[0]?.id || ''
    })
  }, [liveTraderNameById, modalRow, sortedLiveTraders])

  const assignLivePositionMutation = useMutation({
    mutationFn: async (params: { traderId: string; tokenId: string }) => {
      return adoptTraderLiveWalletPosition(params.traderId, {
        token_id: params.tokenId,
        reason: 'manual_wallet_position_adopt_positions_tab',
        requested_by: 'positions_panel',
      })
    },
    onMutate: () => {
      setAdoptError(null)
      setAdoptSuccess(null)
    },
    onSuccess: (_, variables) => {
      const assignedBotName = liveTraderNameById.get(variables.traderId) || 'selected bot'
      setAdoptSuccess(`Position is now managed by ${assignedBotName}.`)
      setModalRow((current) => {
        if (!current || current.tokenId !== variables.tokenId) return current
        return {
          ...current,
          managedByBot: assignedBotName,
          managedBotId: variables.traderId,
        }
      })
      void queryClient.invalidateQueries({ queryKey: ['positions-panel', 'trader-orders-open'] })
      void queryClient.invalidateQueries({ queryKey: ['positions-panel', 'polymarket-live-open-positions'] })
    },
    onError: (error: unknown) => {
      setAdoptError(readErrorMessage(error, 'Failed to assign position to bot'))
    },
  })

  const modalRowMarketIds = useMemo(() => {
    if (!modalRow) return []
    const ids = new Set<string>()
    const marketId = readString(modalRow.marketId)
    if (marketId) ids.add(marketId.toLowerCase())
    const tokenId = readString(modalRow.tokenId)
    if (tokenId) ids.add(tokenId.toLowerCase())
    return Array.from(ids)
  }, [modalRow])

  const modalRowMarketIdsKey = modalRowMarketIds.join('|')

  const modalRowHistoryQuery = useQuery({
    queryKey: ['positions-panel', 'modal-market-history', modalRowMarketIdsKey],
    enabled: modalRowMarketIds.length > 0,
    refetchInterval: modalRow ? MODAL_MARKET_HISTORY_REFRESH_MS : false,
    staleTime: 5000,
    refetchOnMount: 'always',
    queryFn: async () => {
      if (modalRowMarketIds.length === 0) return {}
      return getTraderMarketHistory(modalRowMarketIds, MODAL_MARKET_HISTORY_LIMIT)
    },
  })

  const modalRowHistorySeries = useMemo(() => {
    if (!modalRow) return [] as Array<{ time: number; value: number }>
    const allHistories = modalRowHistoryQuery.data || {}
    const candidateKeys = [
      normalizeTokenKey(modalRow.marketId),
      normalizeTokenKey(modalRow.tokenId),
      readString(modalRow.marketId),
      readString(modalRow.tokenId),
    ]
    let rawHistory: unknown = null
    for (const key of candidateKeys) {
      if (!key) continue
      const candidate = allHistories[key]
      if (!Array.isArray(candidate)) continue
      rawHistory = candidate
      break
    }
    if (!Array.isArray(rawHistory)) return [] as Array<{ time: number; value: number }>

    const useNoSeries = isNoSide(modalRow.side)
    const normalized: Array<{ time: number; value: number }> = []

    rawHistory.forEach((point) => {
      const parsedPoint = readHistoryPoint(point)
      if (!parsedPoint) return
      const value = useNoSeries ? parsedPoint.no : parsedPoint.yes
      if (value === null || value < 0) return
      normalized.push({
        time: parsedPoint.time,
        value,
      })
    })

    normalized.sort((left, right) => left.time - right.time)

    const deduped: Array<{ time: number; value: number }> = []
    normalized.forEach((point) => {
      const previous = deduped[deduped.length - 1]
      if (previous && previous.time === point.time) {
        deduped[deduped.length - 1] = point
        return
      }
      deduped.push(point)
    })

    if (deduped.length >= 2) return deduped
    return [] as Array<{ time: number; value: number }>
  }, [modalRow, modalRowHistoryQuery.data])

  const modalRowLivelineValue = useMemo(() => {
    const latest = modalRowHistorySeries[modalRowHistorySeries.length - 1]
    return latest ? latest.value : 0
  }, [modalRowHistorySeries])

  const modalRowLivelineColor = useMemo(() => {
    if (!modalRow) return '#22c55e'
    return (modalRow.unrealizedPnl || 0) >= 0 ? '#22c55e' : '#ef4444'
  }, [modalRow])

  const managedBotByTokenId = useMemo(() => {
    const map = new Map<string, {
      traderId: string
      traderName: string
      orderId: string | null
      updatedAtTs: number
    }>()
    traderOrders.forEach((order) => {
      const mode = String(order.mode || '').trim().toLowerCase()
      if (mode !== 'live') return
      const status = String(order.status || '').trim().toLowerCase()
      if (!OPEN_LIVE_MANAGED_ORDER_STATUSES.has(status)) return

      const tokenKey = extractManagedTokenFromOrder(order)
      if (!tokenKey) return

      const traderId = String(order.trader_id || '').trim()
      const traderName = liveTraderNameById.get(traderId) || traderId || 'Unknown bot'
      const updatedAtTs = toTimestamp(order.updated_at || order.executed_at || order.created_at)
      const current = map.get(tokenKey)
      if (current && current.updatedAtTs > updatedAtTs) return
      map.set(tokenKey, {
        traderId,
        traderName,
        orderId: readString(order.id),
        updatedAtTs,
      })
    })
    return map
  }, [liveTraderNameById, traderOrders])

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
          markUpdatedAt: null,
          markFresh: false,
          tokenId: null,
          managedByBot: null,
          managedBotId: null,
          managedOrderId: null,
          marketUrl: bucket.marketUrl,
          markMode: 'entry_estimate',
        }
      })
      .filter((row): row is PositionRow => row !== null)
      .sort((left, right) => right.marketValue - left.marketValue)
  }, [accountNameById, selectedSandboxAccount, simulationCoverageKeys, traderOrders])

  const polymarketLiveRows = useMemo<PositionRow[]>(() => {
    const fallbackMarkTs = polymarketLiveUpdatedAt > 0
      ? new Date(polymarketLiveUpdatedAt).toISOString()
      : null
    const now = Date.now()
    const freshCutoff = now - LIVE_MARK_FRESH_MS
    return polymarketLivePositions.map((position) => {
      const record = position as unknown as Record<string, unknown>
      const costBasis = position.size * position.average_cost
      const marketValue = position.size * position.current_price
      const unrealizedPnl = position.unrealized_pnl
      const pnlPercent = costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0
      const side = normalizeDirection(position.outcome)
      const markUpdatedAt = readTimestamp(record, ['mark_updated_at', 'last_marked_at', 'updated_at']) || fallbackMarkTs
      const markFresh = position.current_price > 0 && toTimestamp(markUpdatedAt) >= freshCutoff
      const managed = managedBotByTokenId.get(normalizeTokenKey(position.token_id))
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
        openedAt: markUpdatedAt,
        markUpdatedAt,
        markFresh,
        tokenId: position.token_id,
        managedByBot: managed?.traderName || null,
        managedBotId: managed?.traderId || null,
        managedOrderId: managed?.orderId || null,
        marketUrl: buildPolymarketMarketUrl({
          eventSlug: position.event_slug,
          marketSlug: position.market_slug,
          marketId: position.market_id,
        }),
        markMode: 'live',
      }
    })
  }, [managedBotByTokenId, polymarketLivePositions, polymarketLiveUpdatedAt])

  const kalshiLiveRows = useMemo<PositionRow[]>(() => {
    const fallbackMarkTs = kalshiLiveUpdatedAt > 0
      ? new Date(kalshiLiveUpdatedAt).toISOString()
      : null
    const now = Date.now()
    const freshCutoff = now - LIVE_MARK_FRESH_MS
    return kalshiLivePositions.map((position) => {
      const record = position as unknown as Record<string, unknown>
      const costBasis = position.size * position.average_cost
      const marketValue = position.size * position.current_price
      const unrealizedPnl = position.unrealized_pnl
      const pnlPercent = costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0
      const side = normalizeDirection(position.outcome)
      const markUpdatedAt = readTimestamp(record, ['mark_updated_at', 'last_marked_at', 'updated_at']) || fallbackMarkTs
      const markFresh = position.current_price > 0 && toTimestamp(markUpdatedAt) >= freshCutoff
      const managed = managedBotByTokenId.get(normalizeTokenKey(position.token_id))
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
        openedAt: markUpdatedAt,
        markUpdatedAt,
        markFresh,
        tokenId: position.token_id,
        managedByBot: managed?.traderName || null,
        managedBotId: managed?.traderId || null,
        managedOrderId: managed?.orderId || null,
        marketUrl: buildKalshiMarketUrl({
          marketTicker: position.market_id,
          eventTicker: position.event_slug,
        }),
        markMode: 'live',
      }
    })
  }, [kalshiLivePositions, kalshiLiveUpdatedAt, managedBotByTokenId])

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
    const query = deferredSearchQuery.trim().toLowerCase()
    const minExposure = exposureFloorValue(exposureFloor)

    return baseRows.filter((row) => {
      if (query) {
        const haystack = `${row.marketQuestion} ${row.marketId} ${row.accountLabel} ${row.venueLabel} ${row.side} ${row.sideLabel} ${row.managedByBot || ''}`.toLowerCase()
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
  }, [baseRows, deferredSearchQuery, sideFilter, markFilter, effectiveAccountFilter, exposureFloor])

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
        const leftTs = toTimestamp(left.markUpdatedAt || left.openedAt)
        const rightTs = toTimestamp(right.markUpdatedAt || right.openedAt)
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
    || (shouldShowLive && traderOrdersLoading)
    || (shouldFetchPolymarketLive && polymarketLiveLoading)
    || (shouldFetchKalshiLive && (kalshiStatusLoading || (Boolean(kalshiStatus?.authenticated) && kalshiLiveLoading)))
  )
  const showLoadingSkeleton = isLoading && baseRows.length === 0
  const showModalHistorySkeleton = (
    modalRowHistoryQuery.isPending
    || (modalRowHistoryQuery.isFetching && modalRowHistorySeries.length === 0)
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
      {showLoadingSkeleton ? (
        <PositionsMetricSkeleton />
      ) : !isLoading && sortedRows.length > 0 ? (
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
      ) : null}

      {/* Main content — fills remaining space */}
      <div className="flex-1 min-h-0">
        {showLoadingSkeleton ? (
          <PositionsLoadingSkeleton />
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
                  <Table className="min-w-[1320px]">
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
                        <TableHead className="text-[11px]">Bot</TableHead>
                        <TableHead className="text-[11px]">Updated</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pagedRows.map((row) => {
                        const hasLiveMark = row.markMode === 'live' && row.currentPrice !== null
                        const modeLabel = row.venue === 'polymarket-live' || row.venue === 'kalshi-live' ? 'LIVE' : 'PAPER'
                        const exposureShare = totalExposure > 0 ? (row.marketValue / totalExposure) * 100 : 0
                        return (
                          <TableRow
                            key={row.key}
                            className="cursor-pointer text-xs hover:bg-muted/30"
                            onClick={() => openRowModal(row)}
                          >
                            <TableCell className="max-w-[260px] truncate py-1" title={row.marketQuestion}>
                              <p className="truncate">{row.marketQuestion}</p>
                              <p className="text-[10px] text-muted-foreground truncate" title={`${row.marketId} • ${row.accountLabel} • ${row.venueLabel}${row.status ? ` • ${row.status}` : ''}${row.managedByBot ? ` • Bot ${row.managedByBot}` : ''}`}>
                                {row.marketId} • {row.accountLabel} • {row.venueLabel}{row.status ? ` • ${row.status}` : ''}{row.managedByBot ? ` • Bot ${row.managedByBot}` : ''}
                              </p>
                            </TableCell>
                            <TableCell className="py-1">
                              <div className="flex items-center gap-1">
                                {row.marketUrl ? (
                                  <a
                                    href={row.marketUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(event) => event.stopPropagation()}
                                    className="inline-flex h-4 w-4 items-center justify-center rounded border border-border/70 text-muted-foreground transition-colors hover:text-foreground"
                                    title="Open market"
                                  >
                                    <ExternalLink className="h-3 w-3" />
                                  </a>
                                ) : null}
                                {cryptoMarketsMap.has(row.marketId) ? (
                                  <button
                                    onClick={(event) => {
                                      event.stopPropagation()
                                      setModalMarket(cryptoMarketsMap.get(row.marketId)!)
                                    }}
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
                            <TableCell className={cn('text-right font-mono py-1', row.markFresh && 'text-sky-300')}>
                              {row.currentPrice !== null ? (
                                hasLiveMark ? (
                                  <FlashNumber
                                    value={row.currentPrice}
                                    decimals={4}
                                    className={cn('font-mono text-xs', row.markFresh && 'data-glow-blue')}
                                    positiveClass="data-glow-green"
                                    negativeClass="data-glow-red"
                                  />
                                ) : formatOptionalPrice(row.currentPrice)
                              ) : formatOptionalPrice(row.currentPrice)}
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
                            <TableCell className="py-1 text-[10px]">
                              {row.managedByBot ? (
                                <Badge
                                  variant="outline"
                                  className="h-5 max-w-[180px] truncate border-border/80 bg-muted/60 px-1.5 text-[10px] text-cyan-300"
                                  title={row.managedOrderId ? `${row.managedByBot} • order ${row.managedOrderId}` : row.managedByBot}
                                >
                                  Managed: {row.managedByBot}
                                </Badge>
                              ) : (
                                row.venue === 'polymarket-live' ? (
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className="h-5 px-1.5 text-[10px]"
                                    onClick={(event) => {
                                      event.stopPropagation()
                                      openRowModal(row)
                                    }}
                                  >
                                    Assign
                                  </Button>
                                ) : (
                                  <span className="text-muted-foreground">—</span>
                                )
                              )}
                            </TableCell>
                            <TableCell className="py-1 text-[10px] text-muted-foreground">
                              {formatRelativeTime(row.markUpdatedAt || row.openedAt)}
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
          {modalRow && (
            <motion.div
              key={`position-row-modal-${modalRow.key}`}
              className="fixed inset-0 z-[121] flex items-center justify-center p-4 sm:p-6"
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
                onClick={closeRowModal}
                aria-hidden
              />
              <motion.div
                className="relative z-10 w-full max-w-[960px] max-h-[90vh] overflow-y-auto rounded-2xl border border-border/70 bg-background p-4 shadow-2xl"
                role="dialog"
                aria-modal="true"
                initial={{ scale: 0.94, opacity: 0, y: 22 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 14 }}
                transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="truncate text-sm font-semibold" title={modalRow.marketQuestion}>{modalRow.marketQuestion}</h3>
                    <p className="truncate text-[11px] text-muted-foreground" title={modalRow.marketId}>
                      {modalRow.marketId}
                    </p>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <Badge variant="outline" className="h-5 px-1.5 text-[10px]">{modalRow.venueLabel}</Badge>
                      <Badge variant="outline" className="h-5 px-1.5 text-[10px]">{modalRow.sideLabel}</Badge>
                      <Badge variant="outline" className="h-5 px-1.5 text-[10px] border-border/80 bg-muted/60 text-muted-foreground">{modalRow.accountLabel}</Badge>
                      {modalRow.managedByBot ? (
                        <Badge
                          variant="outline"
                          className="h-5 max-w-[220px] truncate border-border/80 bg-muted/60 px-1.5 text-[10px] text-cyan-300"
                          title={modalRow.managedOrderId ? `${modalRow.managedByBot} • order ${modalRow.managedOrderId}` : modalRow.managedByBot}
                        >
                          Managed by {modalRow.managedByBot}
                        </Badge>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {modalRow.marketUrl ? (
                      <a
                        href={modalRow.marketUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border/60 text-muted-foreground transition-colors hover:text-foreground hover:bg-muted/60"
                        title="Open market"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    ) : null}
                    {cryptoMarketsMap.has(modalRow.marketId) ? (
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-7 px-2 text-[11px]"
                        onClick={() => setModalMarket(cryptoMarketsMap.get(modalRow.marketId)!)}
                      >
                        <Maximize2 className="mr-1 h-3 w-3" />
                        Market
                      </Button>
                    ) : null}
                    <Button type="button" size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={closeRowModal}>
                      Close
                    </Button>
                  </div>
                </div>

                <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Exposure</p>
                    <p className="text-sm font-mono">{formatUsd(modalRow.marketValue)}</p>
                    <p className="text-[10px] text-muted-foreground">Cost: {formatUsd(modalRow.costBasis)}</p>
                  </div>
                  <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Entry / Mark</p>
                    <p className="text-sm font-mono">
                      {formatOptionalPrice(modalRow.entryPrice)}
                      <span className="mx-1 text-muted-foreground">→</span>
                      {formatOptionalPrice(modalRow.currentPrice)}
                    </p>
                    <p className="text-[10px] text-muted-foreground">{modalRow.markMode === 'live' ? 'Live mark' : 'Entry estimate'}</p>
                  </div>
                  <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Unrealized</p>
                    <p className={cn('text-sm font-mono', (modalRow.unrealizedPnl || 0) > 0 ? 'text-emerald-500' : (modalRow.unrealizedPnl || 0) < 0 ? 'text-red-500' : '')}>
                      {modalRow.unrealizedPnl !== null ? formatSignedUsd(modalRow.unrealizedPnl) : '—'}
                    </p>
                    <p className="text-[10px] text-muted-foreground">{modalRow.pnlPercent !== null ? formatSignedPct(modalRow.pnlPercent) : '—'}</p>
                  </div>
                  <div className="rounded-md border border-border/60 bg-card/80 px-2.5 py-2">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Timing</p>
                    <p className="text-sm font-mono">{formatRelativeTime(modalRow.markUpdatedAt || modalRow.openedAt)}</p>
                    <p className="truncate text-[10px] text-muted-foreground" title={modalRow.tokenId || undefined}>
                      Token: {modalRow.tokenId || 'n/a'}
                    </p>
                  </div>
                </div>

                <div className={cn(
                  'mt-3 overflow-hidden rounded-lg border',
                  themeMode === 'dark'
                    ? 'border-slate-700/40 bg-gradient-to-b from-slate-900/75 via-slate-950/80 to-black/90'
                    : 'border-slate-200/90 bg-gradient-to-b from-white via-slate-50 to-slate-100/70'
                )}>
                  {modalRowHistorySeries.length >= 2 ? (
                    <Liveline
                      data={modalRowHistorySeries}
                      value={modalRowLivelineValue}
                      color={modalRowLivelineColor}
                      theme={themeMode}
                      showValue
                      valueMomentumColor
                      grid
                      badge
                      pulse
                      fill
                      windows={LIVELINE_WINDOW_PRESETS}
                      windowStyle="rounded"
                      lerpSpeed={0.1}
                      padding={{ top: 8, right: 80, bottom: 24, left: 14 }}
                      formatValue={(value) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 })}`}
                      referenceLine={modalRow.entryPrice !== null ? { value: modalRow.entryPrice, label: 'Entry' } : undefined}
                      style={{ height: 260 }}
                    />
                  ) : showModalHistorySkeleton ? (
                    <ModalHistorySkeleton />
                  ) : (
                    <div className="h-[260px] flex items-center justify-center text-xs text-muted-foreground">
                      No market history available yet.
                    </div>
                  )}
                </div>

                {modalRow.venue === 'polymarket-live' && !modalRow.managedByBot && (
                  <div className="mt-3 rounded-md border border-border/60 bg-card/80 p-3">
                    <p className="text-[11px] font-semibold">Assign To Live Bot</p>
                    <p className="mt-0.5 text-[10px] text-muted-foreground">
                      Attach this live wallet position to a bot so lifecycle logic can continue managing it.
                    </p>
                    {sortedLiveTraders.length === 0 ? (
                      <div className="mt-2 rounded-md border border-amber-500/35 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-300">
                        No live bots available. Create or enable a LIVE bot first.
                      </div>
                    ) : !modalRow.tokenId ? (
                      <div className="mt-2 rounded-md border border-amber-500/35 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-300">
                        This row does not include a token id, so it cannot be assigned.
                      </div>
                    ) : (
                      <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center">
                        <Select value={assignLiveTraderId} onValueChange={setAssignLiveTraderId}>
                          <SelectTrigger className="h-8 w-full sm:w-[240px] bg-background/70 text-xs">
                            <SelectValue placeholder="Select live bot" />
                          </SelectTrigger>
                          <SelectContent>
                            {sortedLiveTraders.map((trader) => (
                              <SelectItem key={trader.id} value={trader.id}>{trader.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button
                          size="sm"
                          className="h-8 px-3 text-xs"
                          disabled={!assignLiveTraderId || assignLivePositionMutation.isPending}
                          onClick={() => assignLivePositionMutation.mutate({
                            traderId: assignLiveTraderId,
                            tokenId: modalRow.tokenId!,
                          })}
                        >
                          {assignLivePositionMutation.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                          Assign Position
                        </Button>
                      </div>
                    )}
                    {adoptError ? <p className="mt-2 text-xs text-red-500">{adoptError}</p> : null}
                    {adoptSuccess ? <p className="mt-2 text-xs text-emerald-400">{adoptSuccess}</p> : null}
                  </div>
                )}
              </motion.div>
            </motion.div>
          )}
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
                  nowMs={Date.now()}
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

function PositionsMetricSkeleton() {
  return (
    <div className="shrink-0 flex flex-wrap items-center gap-3 border-y border-border/50 py-1.5 px-0.5 animate-pulse">
      {Array.from({ length: 8 }).map((_, index) => (
        <div key={`positions-metric-skeleton-${index}`} className="flex items-center gap-2 text-xs">
          <div className="h-3.5 w-3.5 rounded-full bg-muted/55" />
          <div className="h-2.5 w-12 rounded bg-muted/45" />
          <div className="h-3 w-14 rounded bg-muted/60" />
        </div>
      ))}
    </div>
  )
}

function PositionsLoadingSkeleton() {
  return (
    <div className="h-full grid gap-2 xl:grid-cols-[minmax(0,1fr)_280px] animate-pulse">
      <div className="flex min-h-0 flex-col gap-1.5">
        <div className="shrink-0 flex items-center justify-between px-1">
          <div className="h-2.5 w-28 rounded bg-muted/45" />
          <div className="h-5 w-24 rounded bg-muted/35" />
        </div>
        <div className="flex-1 min-h-0 rounded-md border border-border/60 bg-card/60 p-3">
          <div className="space-y-2">
            {Array.from({ length: 11 }).map((_, index) => (
              <div key={`positions-table-skeleton-${index}`} className="grid grid-cols-[2.8fr_0.4fr_0.8fr_1fr_0.9fr_0.9fr_1fr_0.9fr_0.7fr_0.7fr_0.7fr_1.2fr_0.9fr] gap-3 rounded border border-border/35 bg-background/25 px-3 py-2">
                <div className="h-2.5 rounded bg-muted/45" />
                <div className="h-2.5 rounded bg-muted/35" />
                <div className="h-2.5 rounded bg-muted/40" />
                <div className="h-2.5 rounded bg-muted/35" />
                <div className="h-2.5 rounded bg-muted/35" />
                <div className="h-2.5 rounded bg-muted/35" />
                <div className="h-2.5 rounded bg-muted/40" />
                <div className="h-2.5 rounded bg-muted/35" />
                <div className="h-2.5 rounded bg-muted/30" />
                <div className="h-2.5 rounded bg-muted/30" />
                <div className="h-2.5 rounded bg-muted/30" />
                <div className="h-2.5 rounded bg-muted/35" />
                <div className="h-2.5 rounded bg-muted/35" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="hidden xl:flex flex-col gap-2 min-h-0">
        <div className="h-full rounded-md border border-border/60 bg-card/60 p-3">
          <div className="space-y-3">
            <div>
              <div className="h-3 w-28 rounded bg-muted/55" />
              <div className="mt-3 space-y-2">
                {Array.from({ length: 4 }).map((_, index) => (
                  <div key={`positions-sidebar-book-${index}`} className="rounded-lg border border-border/75 bg-muted/20 px-2.5 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="h-2.5 w-20 rounded bg-muted/45" />
                      <div className="h-2.5 w-14 rounded bg-muted/55" />
                    </div>
                    <div className="mt-2 h-1.5 rounded-full bg-muted/35" />
                    <div className="mt-2 h-2 w-12 rounded bg-muted/40" />
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t border-border/50 pt-3">
              <div className="h-3 w-24 rounded bg-muted/55" />
              <div className="mt-3 space-y-2">
                {Array.from({ length: 3 }).map((_, index) => (
                  <div key={`positions-sidebar-pressure-${index}`}>
                    <div className="flex items-center justify-between gap-2">
                      <div className="h-2.5 w-10 rounded bg-muted/40" />
                      <div className="h-2.5 w-14 rounded bg-muted/50" />
                    </div>
                    <div className="mt-1 h-2 rounded-full bg-muted/35" />
                    <div className="mt-1.5 h-2 w-12 rounded bg-muted/40" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ModalHistorySkeleton() {
  return (
    <div className="h-[260px] px-4 py-3 animate-pulse">
      <div className="flex items-center justify-between gap-2">
        <div className="h-3 w-28 rounded bg-muted/45" />
        <div className="h-6 w-16 rounded-full bg-muted/35" />
      </div>
      <div className="mt-4 h-[170px] rounded-md bg-muted/20" />
      <div className="mt-4 flex items-center gap-2">
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={`modal-history-window-skeleton-${index}`} className="h-6 w-10 rounded-full bg-muted/35" />
        ))}
      </div>
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
