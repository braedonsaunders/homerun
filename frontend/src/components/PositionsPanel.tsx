import { type ReactNode, useMemo, useState } from 'react'
import { useAtomValue } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  Briefcase,
  CircleDollarSign,
  ExternalLink,
  Layers,
  RefreshCw,
  Shield,
  Target,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildKalshiMarketUrl, buildPolymarketMarketUrl } from '../lib/marketUrls'
import { selectedAccountIdAtom } from '../store/atoms'
import {
  getAccountPositions,
  getAllTraderOrders,
  getKalshiPositions,
  getKalshiStatus,
  getSimulationAccounts,
  getTradingPositions,
  type KalshiPosition,
  type SimulationAccount,
  type SimulationPosition,
  type TraderOrder,
  type TradingPosition,
} from '../services/api'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { Table, TableBody, TableCell, TableFooter, TableHead, TableHeader, TableRow } from './ui/table'

type ViewMode = 'all' | 'sandbox' | 'live'
type LiveVenueFilter = 'all' | 'polymarket' | 'kalshi'
type PositionVenue = 'sandbox' | 'autotrader-paper' | 'polymarket-live' | 'kalshi-live'
type PriceMarkMode = 'live' | 'entry_estimate'

const OPEN_PAPER_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])

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

function formatSignedUsd(value: number): string {
  return `${value >= 0 ? '+' : ''}${formatUsd(value)}`
}

function formatSignedPct(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatOptionalPrice(value: number | null): string {
  if (value === null) return 'n/a'
  return `$${value.toFixed(4)}`
}

function formatSize(value: number | null): string {
  if (value === null) return 'n/a'
  return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
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
  if (direction === 'BUY' || direction === 'LONG' || direction === 'UP') return 'YES'
  if (direction === 'SELL' || direction === 'SHORT' || direction === 'DOWN') return 'NO'
  return direction
}

function sideBadgeClass(side: string): string {
  const normalized = side.trim().toUpperCase()
  if (normalized === 'YES' || normalized === 'BUY' || normalized === 'LONG' || normalized === 'UP') {
    return 'bg-green-500/20 text-green-300 border-transparent'
  }
  if (normalized === 'NO' || normalized === 'SELL' || normalized === 'SHORT' || normalized === 'DOWN') {
    return 'bg-red-500/20 text-red-300 border-transparent'
  }
  return 'bg-muted text-muted-foreground border-transparent'
}

function venueBadgeClass(venue: PositionVenue): string {
  if (venue === 'sandbox') return 'bg-amber-500/20 text-amber-300 border-transparent'
  if (venue === 'autotrader-paper') return 'bg-cyan-500/20 text-cyan-300 border-transparent'
  if (venue === 'polymarket-live') return 'bg-blue-500/20 text-blue-300 border-transparent'
  return 'bg-indigo-500/20 text-indigo-300 border-transparent'
}

function sortRows(rows: PositionRow[]): PositionRow[] {
  return [...rows].sort((left, right) => {
    const exposureDelta = right.marketValue - left.marketValue
    if (Math.abs(exposureDelta) > 0.001) return exposureDelta
    return toTimestamp(right.openedAt) - toTimestamp(left.openedAt)
  })
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
  } = useQuery({
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
      return {
        key: `sim:${position.accountId}:${position.id}`,
        venue: 'sandbox',
        venueLabel: 'Sandbox',
        accountLabel: position.accountName,
        marketId: position.market_id,
        marketQuestion: position.market_question,
        side: normalizeDirection(position.side),
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

  const autotraderPaperRows = useMemo<PositionRow[]>(() => {
    const buckets = new Map<string, {
      marketId: string
      marketQuestion: string
      side: string
      costBasis: number
      weightedEntry: number
      weightedSize: number
      lastUpdated: string | null
      status: string | null
      marketUrl: string | null
      orderCount: number
    }>()

    traderOrders.forEach((order) => {
      const mode = String(order.mode || '').toLowerCase()
      const status = String(order.status || '').toLowerCase()
      if (mode !== 'paper' || !OPEN_PAPER_ORDER_STATUSES.has(status)) return

      const marketId = readString(order.market_id) || ''
      if (!marketId) return

      const side = normalizeDirection(order.direction)
      const notional = Math.abs(toNumber(order.notional_usd))
      const entryPrice = toNumber(order.effective_price ?? order.entry_price)
      const key = `${marketId}:${side}`
      const payload = (order.payload && typeof order.payload === 'object')
        ? order.payload as Record<string, unknown>
        : {}

      if (!buckets.has(key)) {
        buckets.set(key, {
          marketId,
          marketQuestion: readString(order.market_question) || marketId,
          side,
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
          orderCount: 0,
        })
      }

      const bucket = buckets.get(key)
      if (!bucket) return

      bucket.costBasis += notional
      if (entryPrice > 0 && notional > 0) {
        bucket.weightedEntry += entryPrice * notional
        bucket.weightedSize += notional / entryPrice
      }
      bucket.orderCount += 1

      const currentTs = toTimestamp(bucket.lastUpdated)
      const nextTs = toTimestamp(order.updated_at || order.executed_at || order.created_at)
      if (nextTs > currentTs) {
        bucket.lastUpdated = order.updated_at || order.executed_at || order.created_at || bucket.lastUpdated
      }
    })

    return Array.from(buckets.entries())
      .map<PositionRow>(([key, bucket]) => {
        const entryPrice = bucket.costBasis > 0 ? bucket.weightedEntry / bucket.costBasis : null
        const size = bucket.weightedSize > 0 ? bucket.weightedSize : null
        return {
          key: `paper:${key}`,
          venue: 'autotrader-paper',
          venueLabel: 'Autotrader Paper',
          accountLabel: 'Autotrader (Paper)',
          marketId: bucket.marketId,
          marketQuestion: bucket.marketQuestion,
          side: bucket.side,
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
      .sort((left, right) => right.marketValue - left.marketValue)
  }, [traderOrders])

  const polymarketLiveRows = useMemo<PositionRow[]>(() => {
    return polymarketLivePositions.map((position) => {
      const costBasis = position.size * position.average_cost
      const marketValue = position.size * position.current_price
      const unrealizedPnl = position.unrealized_pnl
      const pnlPercent = costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0
      return {
        key: `pm-live:${position.market_id}:${position.token_id}:${position.outcome}`,
        venue: 'polymarket-live',
        venueLabel: 'Polymarket Live',
        accountLabel: 'Polymarket',
        marketId: position.market_id,
        marketQuestion: position.market_question,
        side: normalizeDirection(position.outcome),
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
      return {
        key: `kalshi-live:${position.market_id}:${position.token_id}:${position.outcome}`,
        venue: 'kalshi-live',
        venueLabel: 'Kalshi Live',
        accountLabel: 'Kalshi',
        marketId: position.market_id,
        marketQuestion: position.market_question,
        side: normalizeDirection(position.outcome),
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

  const sandboxRows = useMemo(
    () => sortRows([...simulationRows, ...autotraderPaperRows]),
    [simulationRows, autotraderPaperRows]
  )

  const liveRows = useMemo(
    () => sortRows([...polymarketLiveRows, ...kalshiLiveRows]),
    [polymarketLiveRows, kalshiLiveRows]
  )

  const visibleRows = useMemo(() => {
    if (viewMode === 'sandbox') return sandboxRows
    if (viewMode === 'live') return liveRows
    return sortRows([...sandboxRows, ...liveRows])
  }, [viewMode, sandboxRows, liveRows])

  const summary = useMemo(() => {
    const totalCostBasis = visibleRows.reduce((sum, row) => sum + row.costBasis, 0)
    const totalMarketValue = visibleRows.reduce((sum, row) => sum + row.marketValue, 0)
    const markableRows = visibleRows.filter((row) => row.unrealizedPnl !== null)
    const totalUnrealizedPnl = markableRows.reduce((sum, row) => sum + (row.unrealizedPnl ?? 0), 0)
    const pnlPercent = totalCostBasis > 0 ? (totalUnrealizedPnl / totalCostBasis) * 100 : 0
    const largestPosition = visibleRows.reduce((max, row) => (row.marketValue > max.marketValue ? row : max), visibleRows[0] || null)
    const estimatedRows = visibleRows.filter((row) => row.markMode === 'entry_estimate').length
    const markCoverage = visibleRows.length > 0 ? (markableRows.length / visibleRows.length) * 100 : 0
    return {
      totalCostBasis,
      totalMarketValue,
      totalUnrealizedPnl,
      pnlPercent,
      largestPosition,
      estimatedRows,
      markCoverage,
    }
  }, [visibleRows])

  const sourceCards = useMemo(() => {
    return [
      {
        id: 'sandbox-manual',
        label: 'Sandbox Manual',
        rows: simulationRows,
        icon: Shield,
        badgeClassName: 'bg-amber-500/20 text-amber-300 border-transparent',
      },
      {
        id: 'autotrader-paper',
        label: 'Autotrader Paper',
        rows: autotraderPaperRows,
        icon: Target,
        badgeClassName: 'bg-cyan-500/20 text-cyan-300 border-transparent',
      },
      {
        id: 'polymarket-live',
        label: 'Polymarket Live',
        rows: polymarketLiveRows,
        icon: CircleDollarSign,
        badgeClassName: 'bg-blue-500/20 text-blue-300 border-transparent',
      },
      {
        id: 'kalshi-live',
        label: 'Kalshi Live',
        rows: kalshiLiveRows,
        icon: Zap,
        badgeClassName: 'bg-indigo-500/20 text-indigo-300 border-transparent',
      },
    ].map((card) => {
      const exposure = card.rows.reduce((sum, row) => sum + row.marketValue, 0)
      const knownPnl = card.rows
        .filter((row) => row.unrealizedPnl !== null)
        .reduce((sum, row) => sum + (row.unrealizedPnl ?? 0), 0)
      return { ...card, exposure, knownPnl }
    })
  }, [simulationRows, autotraderPaperRows, polymarketLiveRows, kalshiLiveRows])

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

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-blue-400" />
            Positions Command Center
          </h2>
          <p className="text-sm text-muted-foreground">
            Unified open-position view across sandbox, autotrader paper, and live venue books.
          </p>
        </div>
        <Button variant="secondary" onClick={handleRefresh} disabled={isLoading}>
          <RefreshCw className={cn('w-4 h-4 mr-2', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as ViewMode)}>
          <TabsList>
            <TabsTrigger value="all">All Books</TabsTrigger>
            <TabsTrigger value="sandbox">Sandbox + Paper</TabsTrigger>
            <TabsTrigger value="live">Live Venues</TabsTrigger>
          </TabsList>
        </Tabs>

        {shouldShowSandbox && accounts.length > 0 && (
          <select
            value={selectedSandboxAccount || ''}
            onChange={(event) => setSelectedSandboxAccount(event.target.value || null)}
            className="h-9 rounded-md border border-border bg-muted px-3 text-sm"
          >
            <option value="">All Sandbox Accounts</option>
            {accounts.map((account) => (
              <option key={account.id} value={account.id}>{account.name}</option>
            ))}
          </select>
        )}

        {shouldShowLive && (
          <select
            value={liveVenueFilter}
            onChange={(event) => setLiveVenueFilter(event.target.value as LiveVenueFilter)}
            className="h-9 rounded-md border border-border bg-muted px-3 text-sm"
          >
            <option value="all">All Live Venues</option>
            <option value="polymarket">Polymarket Live</option>
            <option value="kalshi">Kalshi Live</option>
          </select>
        )}

        <Badge className="rounded-lg border-transparent bg-muted text-muted-foreground">
          {visibleRows.length} open positions
        </Badge>
      </div>

      {simulationPayload.failedAccounts.length > 0 && (
        <Card className="border-amber-500/30 bg-amber-500/10">
          <CardContent className="p-3 text-sm text-amber-100 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-300 shrink-0" />
            Positions could not load for: {simulationPayload.failedAccounts.join(', ')}
          </CardContent>
        </Card>
      )}

      {shouldFetchKalshiLive && !kalshiStatusLoading && !kalshiStatus?.authenticated && (
        <Card className="border-indigo-500/30 bg-indigo-500/10">
          <CardContent className="p-3 text-sm text-indigo-100">
            Kalshi is not authenticated, so live Kalshi positions are unavailable in this panel.
          </CardContent>
        </Card>
      )}

      {isLoading ? (
        <div className="flex justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
            <MetricCard
              icon={<Layers className="w-4 h-4 text-blue-300" />}
              label="Open Positions"
              value={visibleRows.length.toString()}
            />
            <MetricCard
              icon={<CircleDollarSign className="w-4 h-4 text-blue-300" />}
              label="Exposure"
              value={formatUsd(summary.totalMarketValue)}
            />
            <MetricCard
              icon={<Shield className="w-4 h-4 text-amber-300" />}
              label="Cost Basis"
              value={formatUsd(summary.totalCostBasis)}
            />
            <MetricCard
              icon={summary.totalUnrealizedPnl >= 0
                ? <TrendingUp className="w-4 h-4 text-emerald-300" />
                : <TrendingDown className="w-4 h-4 text-red-300" />
              }
              label="Unrealized P&L"
              value={formatSignedUsd(summary.totalUnrealizedPnl)}
              valueClassName={summary.totalUnrealizedPnl >= 0 ? 'text-emerald-300' : 'text-red-300'}
              detail={formatSignedPct(summary.pnlPercent)}
            />
            <MetricCard
              icon={<Target className="w-4 h-4 text-purple-300" />}
              label="Largest Position"
              value={summary.largestPosition ? formatUsd(summary.largestPosition.marketValue) : '$0.00'}
              detail={summary.largestPosition ? summary.largestPosition.marketQuestion : 'n/a'}
            />
            <MetricCard
              icon={<Shield className="w-4 h-4 text-cyan-300" />}
              label="Mark Coverage"
              value={`${summary.markCoverage.toFixed(0)}%`}
              detail={`${summary.estimatedRows} entry-estimated`}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
            {sourceCards.map((card) => (
              <Card key={card.id} className="overflow-hidden">
                <CardContent className="p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-xs text-muted-foreground">{card.label}</p>
                      <p className="text-lg font-semibold">{card.rows.length}</p>
                    </div>
                    <Badge className={cn('rounded border px-2 py-0.5 text-[10px] uppercase tracking-wide', card.badgeClassName)}>
                      <card.icon className="w-3 h-3 mr-1" />
                      book
                    </Badge>
                  </div>
                  <div className="mt-2 text-xs text-muted-foreground">
                    Exposure {formatUsd(card.exposure)} · P&L {formatSignedUsd(card.knownPnl)}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {shouldShowSandbox && (
            <PositionSection
              title="Sandbox + Autotrader Paper Holdings"
              subtitle="Manual sandbox positions and open autotrader paper Polymarket exposure."
              rows={sandboxRows}
              emptyTitle="No sandbox or paper positions"
              emptySubtitle="Run sandbox trades or autotrader paper mode to populate this section."
            />
          )}

          {shouldShowLive && (
            <PositionSection
              title="Live Venue Holdings"
              subtitle="Open positions synced from connected live venue accounts."
              rows={liveRows}
              emptyTitle="No live venue positions"
              emptySubtitle="Connect a venue and open live positions to see them here."
            />
          )}
        </>
      )}
    </div>
  )
}

function MetricCard({
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
    <Card>
      <CardContent className="p-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {icon}
          <span>{label}</span>
        </div>
        <p className={cn('mt-1 text-base font-semibold font-mono', valueClassName)}>{value}</p>
        {detail && (
          <p className="mt-1 text-[11px] text-muted-foreground line-clamp-1">{detail}</p>
        )}
      </CardContent>
    </Card>
  )
}

function PositionSection({
  title,
  subtitle,
  rows,
  emptyTitle,
  emptySubtitle,
}: {
  title: string
  subtitle: string
  rows: PositionRow[]
  emptyTitle: string
  emptySubtitle: string
}) {
  const totalCostBasis = rows.reduce((sum, row) => sum + row.costBasis, 0)
  const totalMarketValue = rows.reduce((sum, row) => sum + row.marketValue, 0)
  const pnlRows = rows.filter((row) => row.unrealizedPnl !== null)
  const totalUnrealized = pnlRows.reduce((sum, row) => sum + (row.unrealizedPnl ?? 0), 0)
  const unknownPnlRows = rows.length - pnlRows.length

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 border-b border-border bg-muted/30">
        <CardTitle className="text-sm">{title}</CardTitle>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </CardHeader>

      {rows.length === 0 ? (
        <CardContent className="py-10 text-center">
          <Briefcase className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">{emptyTitle}</p>
          <p className="text-sm text-muted-foreground">{emptySubtitle}</p>
        </CardContent>
      ) : (
        <div className="overflow-x-auto">
          <Table className="min-w-[1120px]">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[360px]">Market</TableHead>
                <TableHead>Desk</TableHead>
                <TableHead className="text-center">Side</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Current</TableHead>
                <TableHead className="text-right">Cost Basis</TableHead>
                <TableHead className="text-right">Mark Value</TableHead>
                <TableHead className="text-right">Unrealized</TableHead>
                <TableHead className="text-right">Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.key}>
                  <TableCell className="py-2.5">
                    <div className="space-y-0.5">
                      <p className="font-medium leading-tight line-clamp-2">{row.marketQuestion}</p>
                      <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                        <span className="font-mono">{row.marketId}</span>
                        {row.marketUrl && (
                          <a
                            href={row.marketUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-blue-300 hover:text-blue-200"
                          >
                            Open
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="py-2.5">
                    <div className="flex flex-col gap-1">
                      <Badge className={cn('w-fit rounded border px-1.5 py-0 text-[10px] uppercase tracking-wide', venueBadgeClass(row.venue))}>
                        {row.venueLabel}
                      </Badge>
                      <span className="text-[11px] text-muted-foreground">{row.accountLabel}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-center py-2.5">
                    <Badge className={cn('rounded border px-2 py-0.5 text-[10px] uppercase', sideBadgeClass(row.side))}>
                      {row.side}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right py-2.5 font-mono">{formatSize(row.size)}</TableCell>
                  <TableCell className="text-right py-2.5 font-mono">{formatOptionalPrice(row.entryPrice)}</TableCell>
                  <TableCell className="text-right py-2.5 font-mono">{formatOptionalPrice(row.currentPrice)}</TableCell>
                  <TableCell className="text-right py-2.5 font-mono">{formatUsd(row.costBasis)}</TableCell>
                  <TableCell className="text-right py-2.5 font-mono">{formatUsd(row.marketValue)}</TableCell>
                  <TableCell className="text-right py-2.5">
                    {row.unrealizedPnl === null ? (
                      <span className="text-xs text-muted-foreground">n/a</span>
                    ) : (
                      <span className={cn(
                        'font-mono',
                        (row.unrealizedPnl ?? 0) >= 0 ? 'text-emerald-300' : 'text-red-300'
                      )}>
                        {formatSignedUsd(row.unrealizedPnl)}
                        {row.pnlPercent !== null && (
                          <span className="text-[11px] text-muted-foreground ml-1">
                            ({formatSignedPct(row.pnlPercent)})
                          </span>
                        )}
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-right py-2.5 text-xs text-muted-foreground">
                    {formatRelativeTime(row.openedAt)}
                    {row.status && (
                      <div className="text-[10px] uppercase tracking-wide">{row.status}</div>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
            <TableFooter>
              <TableRow>
                <TableCell colSpan={6} className="text-xs text-muted-foreground">
                  Totals
                </TableCell>
                <TableCell className="text-right font-mono">{formatUsd(totalCostBasis)}</TableCell>
                <TableCell className="text-right font-mono">{formatUsd(totalMarketValue)}</TableCell>
                <TableCell className="text-right font-mono">
                  {unknownPnlRows > 0 && (
                    <div className="text-[10px] text-muted-foreground">
                      n/a for {unknownPnlRows}
                    </div>
                  )}
                  <span className={cn(totalUnrealized >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                    {formatSignedUsd(totalUnrealized)}
                  </span>
                </TableCell>
                <TableCell />
              </TableRow>
            </TableFooter>
          </Table>
        </div>
      )}
    </Card>
  )
}
