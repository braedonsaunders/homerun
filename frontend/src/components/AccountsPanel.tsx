import { useMemo, useState } from 'react'
import { useAtom } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  ChevronRight,
  BarChart3,
  Briefcase,
  DollarSign,
  LayoutDashboard,
  ListChecks,
  Receipt,
  RefreshCw,
  Shield,
  TrendingDown,
  TrendingUp,
  Wallet,
  Zap,
  Settings,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { accountModeAtom, selectedAccountIdAtom } from '../store/atoms'
import { Card, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { ScrollArea } from './ui/scroll-area'
import {
  getAccountPositions,
  getAccountTrades,
  getAllTraderOrders,
  getOrders,
  getKalshiBalance,
  getKalshiPositions,
  getKalshiStatus,
  getSimulationAccounts,
  getTraderOrchestratorOverview,
  getTradingBalance,
  getTradingPositions,
  getTradingStatus,
} from '../services/api'

type AccountsWorkspaceTab = 'overview' | 'sandbox' | 'live'
type DeskView = 'overview' | 'positions' | 'activity'
type LiveVenue = 'polymarket' | 'kalshi'
const OPEN_PAPER_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])

const WORKSPACE_TAB_CONFIG: { id: AccountsWorkspaceTab; label: string; description: string; icon: React.ElementType }[] = [
  {
    id: 'overview',
    label: 'Overview',
    description: 'Cross-account balance sheet, exposure, and quick routing',
    icon: LayoutDashboard,
  },
  {
    id: 'sandbox',
    label: 'Sandbox Desk',
    description: 'Simulation account execution and performance workflows',
    icon: Shield,
  },
  {
    id: 'live',
    label: 'Live Desk',
    description: 'Live venue balances, holdings, and safety posture',
    icon: Zap,
  },
]

interface AccountsPanelProps {
  onOpenSettings: () => void
}

interface LiveVenueSnapshot {
  id: LiveVenue
  label: string
  connected: boolean
  accountLabel: string
  balance: number
  available: number
  exposure: number
  openPositions: number
  unrealizedPnl: number
}

function toFiniteNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function formatUsd(value: number): string {
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatSignedUsd(value: number): string {
  return `${value >= 0 ? '+' : ''}${formatUsd(value)}`
}

function formatSignedPct(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
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

function tradeStatusClass(statusRaw: string): string {
  const status = String(statusRaw || '').trim().toLowerCase()
  if (status.includes('win') || status.includes('resolved_win') || status === 'closed_win') {
    return 'border-emerald-500/40 text-emerald-300'
  }
  if (status.includes('loss') || status.includes('failed') || status === 'closed_loss') {
    return 'border-red-500/40 text-red-300'
  }
  if (status.includes('open') || status.includes('pending')) {
    return 'border-cyan-500/40 text-cyan-300'
  }
  return 'border-border/60 text-muted-foreground'
}

function liveOrderStatusClass(statusRaw: string): string {
  const status = String(statusRaw || '').trim().toLowerCase()
  if (status === 'filled' || status === 'executed' || status === 'complete' || status === 'completed') {
    return 'border-emerald-500/40 text-emerald-300'
  }
  if (status === 'open' || status === 'pending' || status === 'partially_filled' || status === 'submitted') {
    return 'border-cyan-500/40 text-cyan-300'
  }
  if (status === 'cancelled' || status === 'canceled') {
    return 'border-amber-500/40 text-amber-300'
  }
  if (status === 'failed' || status === 'rejected') {
    return 'border-red-500/40 text-red-300'
  }
  return 'border-border/60 text-muted-foreground'
}

export default function AccountsPanel({ onOpenSettings }: AccountsPanelProps) {
  const [accountMode, setAccountMode] = useAtom(accountModeAtom)
  const [selectedAccountId, setSelectedAccountId] = useAtom(selectedAccountIdAtom)
  const [workspaceTab, setWorkspaceTab] = useState<AccountsWorkspaceTab>(accountMode === 'live' ? 'live' : 'overview')
  const [sandboxView, setSandboxView] = useState<DeskView>('overview')
  const [liveView, setLiveView] = useState<DeskView>('overview')

  const { data: sandboxAccounts = [] } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
    refetchInterval: 10000,
  })

  const { data: tradingStatus } = useQuery({
    queryKey: ['trading-status'],
    queryFn: getTradingStatus,
    refetchInterval: 10000,
    retry: false,
  })
  const polymarketReady = Boolean(tradingStatus?.authenticated || tradingStatus?.initialized)

  const { data: tradingPositions = [] } = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    enabled: polymarketReady,
    refetchInterval: 15000,
    retry: false,
  })

  const { data: tradingBalance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
    enabled: polymarketReady,
    refetchInterval: 15000,
    retry: false,
  })

  const { data: kalshiStatus } = useQuery({
    queryKey: ['kalshi-status'],
    queryFn: getKalshiStatus,
    refetchInterval: 10000,
    retry: false,
  })

  const { data: kalshiPositions = [] } = useQuery({
    queryKey: ['kalshi-positions'],
    queryFn: getKalshiPositions,
    enabled: !!kalshiStatus?.authenticated,
    refetchInterval: 15000,
    retry: false,
  })

  const { data: kalshiBalance } = useQuery({
    queryKey: ['kalshi-balance'],
    queryFn: getKalshiBalance,
    enabled: !!kalshiStatus?.authenticated,
    refetchInterval: 15000,
    retry: false,
  })

  const { data: orchestratorOverview } = useQuery({
    queryKey: ['trader-orchestrator-overview'],
    queryFn: getTraderOrchestratorOverview,
    refetchInterval: 10000,
    retry: false,
  })

  const paperModeActive = Boolean(orchestratorOverview?.control?.is_enabled)
    && String(orchestratorOverview?.control?.mode || '').toLowerCase() === 'paper'

  const { data: traderOrders = [] } = useQuery({
    queryKey: ['accounts-panel', 'trader-orders'],
    queryFn: async () => {
      try {
        return await getAllTraderOrders(300)
      } catch {
        return []
      }
    },
    enabled: paperModeActive,
    refetchInterval: 10000,
    retry: false,
  })

  const autotraderPaperMetrics = useMemo(() => {
    const paperAccountId = String(
      orchestratorOverview?.config?.paper_account_id
      || orchestratorOverview?.control?.settings?.paper_account_id
      || ''
    ).trim() || null

    const openPaperOrders = traderOrders.filter((order) => {
      const mode = String(order.mode || '').toLowerCase()
      const status = String(order.status || '').toLowerCase()
      return mode === 'paper' && OPEN_PAPER_ORDER_STATUSES.has(status)
    })

    const positionKeys = new Set<string>()
    let exposureUsd = 0

    for (const order of openPaperOrders) {
      const marketId = String(order.market_id || '').trim()
      if (!marketId) continue
      const side = normalizeDirection(order.direction_side ?? order.direction)
      positionKeys.add(`${marketId}:${side}`)
      exposureUsd += Math.abs(Number(order.notional_usd || 0))
    }

    const fallbackExposure = Number(orchestratorOverview?.metrics?.gross_exposure_usd || 0)

    return {
      active: paperModeActive,
      paperAccountId,
      openOrders: openPaperOrders.length,
      openPositions: positionKeys.size,
      exposureUsd: exposureUsd > 0 ? exposureUsd : fallbackExposure,
    }
  }, [paperModeActive, orchestratorOverview, traderOrders])

  const autotraderOverlay = useMemo(() => {
    const linkedAccount = autotraderPaperMetrics.paperAccountId
      ? sandboxAccounts.find((account) => account.id === autotraderPaperMetrics.paperAccountId)
      : undefined

    const shouldOverlay = Boolean(
      linkedAccount
      && autotraderPaperMetrics.openPositions > 0
      && (linkedAccount.open_positions || 0) === 0
    )

    return {
      accountId: shouldOverlay ? autotraderPaperMetrics.paperAccountId : null,
      openPositions: shouldOverlay ? autotraderPaperMetrics.openPositions : 0,
      openOrders: shouldOverlay ? autotraderPaperMetrics.openOrders : 0,
      exposureUsd: shouldOverlay ? autotraderPaperMetrics.exposureUsd : 0,
    }
  }, [autotraderPaperMetrics, sandboxAccounts])

  const sandboxMetrics = useMemo(() => {
    const totalInitial = sandboxAccounts.reduce((sum, account) => sum + (account.initial_capital || 0), 0)
    const totalCapital = sandboxAccounts.reduce((sum, account) => sum + (account.current_capital || 0), 0)
    const realizedPnl = sandboxAccounts.reduce((sum, account) => sum + (account.total_pnl || 0), 0)
    const unrealizedPnl = sandboxAccounts.reduce((sum, account) => sum + (account.unrealized_pnl || 0), 0)
    const totalPnl = realizedPnl + unrealizedPnl
    const totalTrades = sandboxAccounts.reduce((sum, account) => sum + (account.total_trades || 0), 0)
    const totalOpenPositions = sandboxAccounts.reduce((sum, account) => sum + (account.open_positions || 0), 0)
      + autotraderOverlay.openPositions
    const roi = totalInitial > 0 ? (totalPnl / totalInitial) * 100 : 0
    const deployableCapital = Math.max(0, totalCapital - autotraderOverlay.exposureUsd)

    return {
      count: sandboxAccounts.length,
      totalInitial,
      totalCapital,
      deployableCapital,
      totalPnl,
      roi,
      totalTrades,
      totalOpenPositions,
      autotraderOverlay,
    }
  }, [sandboxAccounts, autotraderOverlay])

  const polymarketSnapshot = useMemo<LiveVenueSnapshot>(() => {
    const exposure = tradingPositions.reduce(
      (sum, pos) => sum + toFiniteNumber(pos.size) * toFiniteNumber(pos.current_price),
      0
    )
    const unrealizedPnl = tradingPositions.reduce((sum, pos) => sum + toFiniteNumber(pos.unrealized_pnl), 0)
    return {
      id: 'polymarket',
      label: 'Polymarket',
      connected: polymarketReady,
      accountLabel: tradingStatus?.wallet_address
        ? `${tradingStatus.wallet_address.slice(0, 8)}...${tradingStatus.wallet_address.slice(-6)}`
        : 'No wallet',
      balance: toFiniteNumber(tradingBalance?.balance),
      available: toFiniteNumber(tradingBalance?.available),
      exposure,
      openPositions: tradingPositions.filter((position) => toFiniteNumber(position.size) > 0).length,
      unrealizedPnl,
    }
  }, [
    polymarketReady,
    tradingStatus?.wallet_address,
    tradingBalance?.balance,
    tradingBalance?.available,
    tradingPositions,
  ])

  const kalshiSnapshot = useMemo<LiveVenueSnapshot>(() => {
    const exposure = kalshiPositions.reduce(
      (sum, pos) => sum + toFiniteNumber(pos.size) * toFiniteNumber(pos.current_price),
      0
    )
    const unrealizedPnl = kalshiPositions.reduce((sum, pos) => sum + toFiniteNumber(pos.unrealized_pnl), 0)
    return {
      id: 'kalshi',
      label: 'Kalshi',
      connected: Boolean(kalshiStatus?.authenticated),
      accountLabel: kalshiStatus?.email || (kalshiStatus?.member_id ? `Member ${kalshiStatus.member_id}` : 'No account'),
      balance: toFiniteNumber(kalshiBalance?.balance ?? kalshiStatus?.balance?.balance),
      available: toFiniteNumber(kalshiBalance?.available ?? kalshiStatus?.balance?.available),
      exposure,
      openPositions: kalshiPositions.filter((position) => toFiniteNumber(position.size) > 0).length,
      unrealizedPnl,
    }
  }, [kalshiStatus, kalshiBalance?.balance, kalshiBalance?.available, kalshiPositions])

  const venueSnapshots = useMemo(() => [polymarketSnapshot, kalshiSnapshot], [polymarketSnapshot, kalshiSnapshot])

  const liveMetrics = useMemo(() => {
    const totalBalance = venueSnapshots.reduce((sum, venue) => sum + venue.balance, 0)
    const totalAvailable = venueSnapshots.reduce((sum, venue) => sum + venue.available, 0)
    const totalExposure = venueSnapshots.reduce((sum, venue) => sum + venue.exposure, 0)
    const totalOpenPositions = venueSnapshots.reduce((sum, venue) => sum + venue.openPositions, 0)
    const totalUnrealizedPnl = venueSnapshots.reduce((sum, venue) => sum + venue.unrealizedPnl, 0)
    const connectedVenues = venueSnapshots.filter((venue) => venue.connected).length

    return {
      totalBalance,
      totalAvailable,
      totalExposure,
      totalOpenPositions,
      totalUnrealizedPnl,
      connectedVenues,
    }
  }, [venueSnapshots])

  const activeSandboxAccountId = useMemo(() => {
    if (selectedAccountId && !selectedAccountId.startsWith('live:')) return selectedAccountId
    return sandboxAccounts[0]?.id || null
  }, [selectedAccountId, sandboxAccounts])

  const activeSandboxAccount = useMemo(
    () => sandboxAccounts.find((account) => account.id === activeSandboxAccountId) || null,
    [sandboxAccounts, activeSandboxAccountId]
  )

  const { data: sandboxPositions = [] } = useQuery({
    queryKey: ['accounts-panel', 'sandbox-positions', activeSandboxAccountId],
    queryFn: () => (activeSandboxAccountId ? getAccountPositions(activeSandboxAccountId) : Promise.resolve([])),
    enabled: workspaceTab === 'sandbox' && Boolean(activeSandboxAccountId),
    refetchInterval: 10000,
  })

  const { data: sandboxTrades = [] } = useQuery({
    queryKey: ['accounts-panel', 'sandbox-trades', activeSandboxAccountId],
    queryFn: () => (activeSandboxAccountId ? getAccountTrades(activeSandboxAccountId, 250) : Promise.resolve([])),
    enabled: workspaceTab === 'sandbox' && Boolean(activeSandboxAccountId),
    refetchInterval: 10000,
  })

  const { data: liveOrders = [] } = useQuery({
    queryKey: ['accounts-panel', 'live-orders'],
    queryFn: () => getOrders(250),
    enabled: workspaceTab === 'live' && polymarketReady,
    refetchInterval: 10000,
    retry: false,
  })

  const activeContext = useMemo(() => {
    if (selectedAccountId?.startsWith('live:')) {
      const venue = selectedAccountId === 'live:kalshi' ? kalshiSnapshot : polymarketSnapshot
      return {
        modeLabel: 'Live',
        accountLabel: venue.label,
        status: venue.connected ? 'Connected' : 'Disconnected',
        tone: venue.connected ? 'green' : 'amber',
      }
    }

    if (activeSandboxAccount) {
      return {
        modeLabel: 'Sandbox',
        accountLabel: activeSandboxAccount.name,
        status: `${activeSandboxAccount.total_trades} trades`,
        tone: 'blue',
      }
    }

    return {
      modeLabel: accountMode === 'live' ? 'Live' : 'Sandbox',
      accountLabel: 'No account selected',
      status: 'Select account from header',
      tone: 'neutral',
    }
  }, [selectedAccountId, accountMode, activeSandboxAccount, kalshiSnapshot, polymarketSnapshot])

  const allocationRows = useMemo(() => {
    const sandboxRows = sandboxAccounts.map((account) => ({
      id: account.id,
      label: `Sandbox · ${account.name}`,
      value: account.current_capital || 0,
      tone: 'amber' as const,
    }))

    const liveRows = venueSnapshots.map((venue) => ({
      id: `live:${venue.id}`,
      label: `Live · ${venue.label}`,
      value: venue.balance + venue.exposure,
      tone: 'green' as const,
    }))

    const rows = [...sandboxRows, ...liveRows]
    const total = rows.reduce((sum, row) => sum + row.value, 0)

    return rows
      .sort((a, b) => b.value - a.value)
      .map((row) => ({
        ...row,
        share: total > 0 ? (row.value / total) * 100 : 0,
      }))
  }, [sandboxAccounts, venueSnapshots])

  const riskSignals = useMemo(() => {
    return [
      {
        label: sandboxMetrics.count === 0 ? 'No sandbox account provisioned' : `${sandboxMetrics.count} sandbox accounts online`,
        tone: sandboxMetrics.count === 0 ? 'amber' : 'green',
      },
      {
        label: liveMetrics.connectedVenues === 2 ? 'Both live venues connected' : `${liveMetrics.connectedVenues}/2 live venues connected`,
        tone: liveMetrics.connectedVenues === 2 ? 'green' : 'amber',
      },
      {
        label: liveMetrics.totalUnrealizedPnl >= 0 ? 'Live book currently green' : 'Live book currently red',
        tone: liveMetrics.totalUnrealizedPnl >= 0 ? 'green' : 'red',
      },
      {
        label: sandboxMetrics.totalOpenPositions + liveMetrics.totalOpenPositions > 40
          ? 'High aggregate position count'
          : 'Position count in normal range',
        tone: sandboxMetrics.totalOpenPositions + liveMetrics.totalOpenPositions > 40 ? 'amber' : 'green',
      },
      {
        label: sandboxMetrics.autotraderOverlay.openPositions > 0
          ? `Autotrader paper active (${sandboxMetrics.autotraderOverlay.openPositions} positions)`
          : 'No autotrader paper overlay',
        tone: sandboxMetrics.autotraderOverlay.openPositions > 0 ? 'amber' : 'green',
      },
    ] as const
  }, [
    sandboxMetrics.count,
    sandboxMetrics.totalOpenPositions,
    sandboxMetrics.autotraderOverlay.openPositions,
    liveMetrics.connectedVenues,
    liveMetrics.totalOpenPositions,
    liveMetrics.totalUnrealizedPnl,
  ])

  const sandboxPositionRows = useMemo(() => {
    return sandboxPositions
      .map((position) => {
        const quantity = toFiniteNumber(position.quantity)
        const entryPrice = toFiniteNumber(position.entry_price)
        const markPrice = position.current_price != null ? toFiniteNumber(position.current_price) : entryPrice
        const entryCost = toFiniteNumber(position.entry_cost) || (quantity * entryPrice)
        const marketValue = quantity * markPrice
        const unrealizedPnl = toFiniteNumber(position.unrealized_pnl) || (marketValue - entryCost)
        return {
          ...position,
          quantity,
          entryPrice,
          markPrice,
          entryCost,
          marketValue,
          unrealizedPnl,
        }
      })
      .sort((left, right) => Math.abs(right.marketValue) - Math.abs(left.marketValue))
  }, [sandboxPositions])

  const sandboxTradeRows = useMemo(() => {
    return [...sandboxTrades].sort(
      (left, right) => new Date(right.executed_at).getTime() - new Date(left.executed_at).getTime()
    )
  }, [sandboxTrades])

  const livePositionRows = useMemo(() => {
    const polymarketRows = tradingPositions.map((position) => {
      const size = toFiniteNumber(position.size)
      const markPrice = toFiniteNumber(position.current_price)
      const entryPrice = toFiniteNumber(position.average_cost)
      const marketValue = size * markPrice
      const costBasis = size * entryPrice
      const unrealizedPnl = toFiniteNumber(position.unrealized_pnl) || (marketValue - costBasis)
      return {
        id: `polymarket:${position.token_id}:${position.market_id}`,
        venue: 'Polymarket' as const,
        marketQuestion: String(position.market_question || '').trim() || position.market_id,
        marketId: String(position.market_id || '').trim(),
        outcome: normalizeDirection(position.outcome),
        size,
        entryPrice,
        markPrice,
        costBasis,
        marketValue,
        unrealizedPnl,
      }
    })

    const kalshiRows = kalshiPositions.map((position) => {
      const size = toFiniteNumber(position.size)
      const markPrice = toFiniteNumber(position.current_price)
      const entryPrice = toFiniteNumber(position.average_cost)
      const marketValue = size * markPrice
      const costBasis = size * entryPrice
      const unrealizedPnl = toFiniteNumber(position.unrealized_pnl) || (marketValue - costBasis)
      return {
        id: `kalshi:${position.token_id}:${position.market_id}`,
        venue: 'Kalshi' as const,
        marketQuestion: String(position.market_question || '').trim() || position.market_id,
        marketId: String(position.market_id || '').trim(),
        outcome: normalizeDirection(position.outcome),
        size,
        entryPrice,
        markPrice,
        costBasis,
        marketValue,
        unrealizedPnl,
      }
    })

    return [...polymarketRows, ...kalshiRows].sort((left, right) => Math.abs(right.marketValue) - Math.abs(left.marketValue))
  }, [tradingPositions, kalshiPositions])

  const liveOrderRows = useMemo(() => {
    return [...liveOrders].sort(
      (left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
    )
  }, [liveOrders])

  const liveOpenOrderCount = useMemo(() => {
    return liveOrderRows.filter((order) => {
      const status = String(order.status || '').trim().toLowerCase()
      return status === 'open' || status === 'pending' || status === 'partially_filled' || status === 'submitted'
    }).length
  }, [liveOrderRows])

  const selectedSandboxOverlayOpen = activeSandboxAccount && sandboxMetrics.autotraderOverlay.accountId === activeSandboxAccount.id
    ? sandboxMetrics.autotraderOverlay.openPositions
    : 0
  const selectedSandboxOverlayExposure = activeSandboxAccount && sandboxMetrics.autotraderOverlay.accountId === activeSandboxAccount.id
    ? sandboxMetrics.autotraderOverlay.exposureUsd
    : 0
  const selectedSandboxOpenPositions = (activeSandboxAccount?.open_positions || 0) + selectedSandboxOverlayOpen
  const selectedSandboxTotalPnl = (activeSandboxAccount?.total_pnl || 0) + (activeSandboxAccount?.unrealized_pnl || 0)

  const activeLiveVenue: LiveVenue = selectedAccountId === 'live:kalshi' ? 'kalshi' : 'polymarket'
  const activeLiveSnapshot = activeLiveVenue === 'kalshi' ? kalshiSnapshot : polymarketSnapshot
  const activeLivePositions = useMemo(
    () => livePositionRows.filter((row) => row.venue === (activeLiveVenue === 'kalshi' ? 'Kalshi' : 'Polymarket')),
    [livePositionRows, activeLiveVenue]
  )

  const sandboxStrategyRows = useMemo(() => {
    const rollup = new Map<string, { strategy: string; trades: number; pnl: number; notional: number }>()
    for (const trade of sandboxTradeRows) {
      const strategy = String(trade.strategy_type || 'unknown').trim() || 'unknown'
      const current = rollup.get(strategy) || { strategy, trades: 0, pnl: 0, notional: 0 }
      current.trades += 1
      current.pnl += toFiniteNumber(trade.actual_pnl)
      current.notional += toFiniteNumber(trade.total_cost)
      rollup.set(strategy, current)
    }
    return Array.from(rollup.values()).sort((left, right) => Math.abs(right.pnl) - Math.abs(left.pnl))
  }, [sandboxTradeRows])

  const sandboxTradeStatusRows = useMemo(() => {
    const rollup = new Map<string, number>()
    for (const trade of sandboxTradeRows) {
      const status = String(trade.status || 'unknown').trim().toLowerCase() || 'unknown'
      rollup.set(status, (rollup.get(status) || 0) + 1)
    }
    return Array.from(rollup.entries())
      .map(([status, count]) => ({ status, count }))
      .sort((left, right) => right.count - left.count)
  }, [sandboxTradeRows])

  const openSandboxDesk = (accountId?: string) => {
    if (accountId) {
      setSelectedAccountId(accountId)
    } else if (!selectedAccountId || selectedAccountId.startsWith('live:')) {
      if (sandboxAccounts.length > 0) {
        setSelectedAccountId(sandboxAccounts[0].id)
      }
    }

    setAccountMode('sandbox')
    setWorkspaceTab('sandbox')
    setSandboxView('overview')
  }

  const openLiveDesk = (venue: LiveVenue = 'polymarket') => {
    setSelectedAccountId(`live:${venue}`)
    setAccountMode('live')
    setWorkspaceTab('live')
    setLiveView('overview')
  }

  return (
    <div className="h-full flex min-h-0 flex-col gap-3">
      <Card className="shrink-0 rounded-xl border border-border bg-card/60 shadow-none">
        <CardContent className="p-0">
          <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/80 px-3 py-2.5">
            <div className="min-w-0">
              <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground/70">Account Command Center</p>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <p className="text-sm font-medium text-foreground">{activeContext.accountLabel}</p>
                <Badge
                  variant="outline"
                  className={cn(
                    'h-5 rounded border-transparent px-2 text-[10px] font-semibold uppercase tracking-[0.08em]',
                    activeContext.tone === 'green' && 'bg-green-500/20 text-green-300',
                    activeContext.tone === 'blue' && 'bg-blue-500/20 text-blue-300',
                    activeContext.tone === 'amber' && 'bg-amber-500/20 text-amber-300',
                    activeContext.tone === 'neutral' && 'bg-muted text-muted-foreground'
                  )}
                >
                  {activeContext.modeLabel}
                </Badge>
                <p className="text-xs text-muted-foreground">{activeContext.status}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={onOpenSettings}
                className="h-8 gap-1.5 bg-background/40 text-xs"
              >
                <Settings className="h-3.5 w-3.5" />
                Account Settings
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 p-2.5 md:grid-cols-4 xl:grid-cols-8">
            <DenseMetric
              label="Simulation Equity"
              value={formatUsd(sandboxMetrics.deployableCapital)}
              hint={
                sandboxMetrics.autotraderOverlay.exposureUsd > 0
                  ? `Ledger ${formatUsd(sandboxMetrics.totalCapital)} • ${formatUsd(sandboxMetrics.autotraderOverlay.exposureUsd)} deployed`
                  : `${sandboxMetrics.count} accounts`
              }
              icon={Wallet}
            />
            <DenseMetric
              label="Simulation P&L"
              value={formatSignedUsd(sandboxMetrics.totalPnl)}
              hint={`${sandboxMetrics.totalTrades} trades`}
              icon={sandboxMetrics.totalPnl >= 0 ? TrendingUp : TrendingDown}
              tone={sandboxMetrics.totalPnl >= 0 ? 'green' : 'red'}
            />
            <DenseMetric
              label="Simulation ROI"
              value={formatSignedPct(sandboxMetrics.roi)}
              hint="Aggregate"
              icon={BarChart3}
              tone={sandboxMetrics.roi >= 0 ? 'green' : 'red'}
            />
            <DenseMetric
              label="Simulation Positions"
              value={sandboxMetrics.totalOpenPositions.toString()}
              hint={
                sandboxMetrics.autotraderOverlay.openPositions > 0
                  ? `Open (${sandboxMetrics.autotraderOverlay.openPositions} autotrader)`
                  : 'Open'
              }
              icon={Briefcase}
            />
            <DenseMetric
              label="Live Free Cash"
              value={formatUsd(liveMetrics.totalAvailable)}
              hint={`${liveMetrics.connectedVenues}/2 venues connected`}
              icon={DollarSign}
            />
            <DenseMetric
              label="Live Exposure"
              value={formatUsd(liveMetrics.totalExposure)}
              hint={`${liveMetrics.totalOpenPositions} open positions`}
              icon={Activity}
            />
            <DenseMetric
              label="Live Unrealized"
              value={formatSignedUsd(liveMetrics.totalUnrealizedPnl)}
              hint="Cross-venue"
              icon={liveMetrics.totalUnrealizedPnl >= 0 ? TrendingUp : TrendingDown}
              tone={liveMetrics.totalUnrealizedPnl >= 0 ? 'green' : 'red'}
            />
            <DenseMetric
              label="Fleet Balance"
              value={formatUsd(sandboxMetrics.totalCapital + liveMetrics.totalBalance)}
              hint="Simulation + Live"
              icon={Wallet}
            />
          </div>
        </CardContent>
      </Card>

      <div className="shrink-0 px-1 overflow-x-auto">
        <div className="flex min-w-max items-center gap-1">
          {WORKSPACE_TAB_CONFIG.map((tab) => (
            <Button
              key={tab.id}
              variant="outline"
              size="sm"
              onClick={() => setWorkspaceTab(tab.id)}
              className={cn(
                'h-8 gap-1.5 text-xs',
                workspaceTab === tab.id
                  ? (
                    tab.id === 'overview'
                      ? 'bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400'
                      : tab.id === 'sandbox'
                        ? 'bg-amber-500/20 text-amber-300 border-amber-500/30 hover:bg-amber-500/30 hover:text-amber-300'
                        : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30 hover:text-emerald-400'
                  )
                  : 'bg-card text-muted-foreground hover:text-foreground border-border'
              )}
            >
              <tab.icon className="h-3.5 w-3.5" />
              {tab.label}
            </Button>
          ))}
        </div>
      </div>

      {workspaceTab === 'overview' && (
        <div className="flex-1 min-h-0 grid grid-rows-[minmax(0,1fr)_minmax(0,0.8fr)] gap-3">
          <div className="grid min-h-0 grid-cols-1 gap-3 xl:grid-cols-12">
            <Card className="xl:col-span-8 min-h-0 border-border bg-card/40 shadow-none">
              <CardContent className="flex h-full min-h-0 flex-col p-0">
                <div className="flex shrink-0 items-center justify-between border-b border-border/70 px-3 py-2.5">
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">Sandbox Fleet</p>
                    <p className="text-xs text-muted-foreground">All paper accounts with real-time P&L context</p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openSandboxDesk()}
                    className="h-6 text-[11px]"
                  >
                    Open Sandbox Desk
                  </Button>
                </div>

                {sandboxAccounts.length === 0 ? (
                  <div className="px-4 py-8 text-center">
                    <Shield className="mx-auto mb-2 h-6 w-6 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">No sandbox accounts yet. Create one in Sandbox Desk.</p>
                  </div>
                ) : (
                  <div className="min-h-0 flex-1 overflow-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-border/70 text-muted-foreground">
                          <th className="px-4 py-2 text-left">Account</th>
                          <th className="px-3 py-2 text-right">Capital</th>
                          <th className="px-3 py-2 text-right">P&L</th>
                          <th className="px-3 py-2 text-right">ROI</th>
                          <th className="px-3 py-2 text-right">Trades</th>
                          <th className="px-3 py-2 text-right">Open</th>
                          <th className="px-4 py-2 text-right">Desk</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sandboxAccounts.map((account) => {
                          const autotraderOpenPositions = sandboxMetrics.autotraderOverlay.accountId === account.id
                            ? sandboxMetrics.autotraderOverlay.openPositions
                            : 0
                          const autotraderExposure = sandboxMetrics.autotraderOverlay.accountId === account.id
                            ? sandboxMetrics.autotraderOverlay.exposureUsd
                            : 0
                          const totalOpenPositions = (account.open_positions || 0) + autotraderOpenPositions
                          const totalPnl = (account.total_pnl || 0) + (account.unrealized_pnl || 0)
                          const isSelected = selectedAccountId === account.id
                          return (
                            <tr
                              key={account.id}
                              className={cn(
                                'border-b border-border/40 transition-colors hover:bg-muted/40',
                                isSelected && 'bg-blue-500/10'
                              )}
                            >
                              <td className="px-3 py-2">
                                <button
                                  type="button"
                                  onClick={() => {
                                    setSelectedAccountId(account.id)
                                    setAccountMode('sandbox')
                                  }}
                                  className="text-left"
                                >
                                  <p className="font-medium text-foreground">{account.name}</p>
                                  <p className="text-[11px] text-muted-foreground">
                                    {account.win_rate.toFixed(1)}% win rate • {account.winning_trades}W/{account.losing_trades}L
                                  </p>
                                </button>
                              </td>
                              <td className="px-3 py-2 text-right font-mono">{formatUsd(account.current_capital || 0)}</td>
                              <td className="px-3 py-2 text-right font-mono">
                                <span className={cn(totalPnl >= 0 ? 'text-green-400' : 'text-red-400')}>
                                  {formatSignedUsd(totalPnl)}
                                </span>
                              </td>
                              <td className="px-3 py-2 text-right font-mono">
                                <span className={cn((account.roi_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400')}>
                                  {formatSignedPct(account.roi_percent || 0)}
                                </span>
                              </td>
                              <td className="px-3 py-2 text-right font-mono">{account.total_trades || 0}</td>
                              <td className="px-3 py-2 text-right font-mono">
                                <p>{totalOpenPositions}</p>
                                {autotraderOpenPositions > 0 && (
                                  <p className="text-[10px] font-medium text-cyan-300">
                                    +{autotraderOpenPositions} auto · {formatUsd(autotraderExposure)}
                                  </p>
                                )}
                              </td>
                              <td className="px-3 py-2 text-right">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => openSandboxDesk(account.id)}
                                  className="h-6 px-2 text-[11px]"
                                >
                                  Open
                                </Button>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="xl:col-span-4 min-h-0 border-border bg-card/40 shadow-none">
              <CardContent className="h-full min-h-0 space-y-2.5 overflow-y-auto p-3">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">Live Venues</p>
                  <p className="text-xs text-muted-foreground">Connection status, balances, and deployment footprint</p>
                </div>
                {venueSnapshots.map((venue) => (
                  <div key={venue.id} className="rounded-lg border border-border/70 bg-background/40 p-2.5">
                    <div className="mb-1.5 flex items-start justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className={cn('h-2 w-2 rounded-full', venue.connected ? 'bg-green-400' : 'bg-amber-400')} />
                          <p className="text-sm font-medium">{venue.label}</p>
                        </div>
                        <p className="text-[11px] text-muted-foreground">{venue.accountLabel}</p>
                      </div>
                      <Badge
                        variant="outline"
                        className={cn(
                          'border-transparent px-1.5 py-0.5 text-[10px]',
                          venue.connected ? 'bg-green-500/20 text-green-300' : 'bg-amber-500/20 text-amber-300'
                        )}
                      >
                        {venue.connected ? 'Connected' : 'Disconnected'}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <MetricPair label="Balance" value={formatUsd(venue.balance)} />
                      <MetricPair label="Available" value={formatUsd(venue.available)} />
                      <MetricPair label="Exposure" value={formatUsd(venue.exposure)} />
                      <MetricPair
                        label="Unrealized"
                        value={formatSignedUsd(venue.unrealizedPnl)}
                        valueClass={venue.unrealizedPnl >= 0 ? 'text-green-300' : 'text-red-300'}
                      />
                    </div>
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-muted-foreground">
                      <span>{venue.openPositions} open positions</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => openLiveDesk(venue.id)}
                        className="h-6 px-2 text-[11px]"
                      >
                        Focus Desk
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card className="min-h-0 border-border bg-card/40 shadow-none">
            <CardContent className="h-full min-h-0 space-y-3 overflow-y-auto p-3">
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">Allocation & Risk Radar</p>
                  <p className="text-xs text-muted-foreground">Capital concentration by account and high-level operating posture</p>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  {riskSignals.map((signal) => (
                    <Badge
                      key={signal.label}
                      variant="outline"
                      className={cn(
                        'h-5 border-transparent px-1.5 text-[10px]',
                        signal.tone === 'green' && 'bg-green-500/20 text-green-300',
                        signal.tone === 'amber' && 'bg-amber-500/20 text-amber-300',
                        signal.tone === 'red' && 'bg-red-500/20 text-red-300'
                      )}
                    >
                      {signal.label}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 gap-2 lg:grid-cols-4">
                {allocationRows.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No account balances available yet.</p>
                ) : (
                  allocationRows.map((row) => (
                    <div key={row.id} className="rounded-lg border border-border/60 bg-background/40 p-2">
                      <div className="mb-1 flex items-center justify-between gap-2 text-xs">
                        <span className="truncate text-muted-foreground">{row.label}</span>
                        <span className="font-mono text-foreground">{formatUsd(row.value)}</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-muted/80">
                        <div
                          className={cn('h-full rounded-full', row.tone === 'green' ? 'bg-green-400/80' : 'bg-amber-400/80')}
                          style={{ width: `${Math.max(row.share, 3)}%` }}
                        />
                      </div>
                      <p className="mt-0.5 text-right text-[11px] text-muted-foreground">{row.share.toFixed(1)}%</p>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {workspaceTab === 'sandbox' && (
        <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[250px_minmax(0,1fr)]">
          <div className="hidden xl:flex min-h-0 flex-col rounded-lg border border-border/70 bg-card overflow-hidden">
            <div className="shrink-0 border-b border-border/50 px-2.5 py-2">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Sandbox Accounts</p>
              <p className="text-[10px] text-muted-foreground">{sandboxAccounts.length} desks configured</p>
            </div>
            <ScrollArea className="flex-1 min-h-0">
              <div className="space-y-1.5 p-1.5">
                {sandboxAccounts.length === 0 ? (
                  <p className="px-2 py-6 text-center text-[11px] text-muted-foreground">No sandbox accounts configured.</p>
                ) : (
                  sandboxAccounts.map((account) => {
                    const isActive = activeSandboxAccountId === account.id
                    const totalPnl = (account.total_pnl || 0) + (account.unrealized_pnl || 0)
                    const autotraderPositions = sandboxMetrics.autotraderOverlay.accountId === account.id
                      ? sandboxMetrics.autotraderOverlay.openPositions
                      : 0
                    return (
                      <button
                        key={account.id}
                        type="button"
                        onClick={() => {
                          setSelectedAccountId(account.id)
                          setAccountMode('sandbox')
                        }}
                        className={cn(
                          'w-full rounded-md px-2 py-1.5 text-left transition-colors',
                          isActive ? 'bg-amber-500/15 text-foreground' : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate text-[11px] font-medium">{account.name}</p>
                          <span className={cn('text-[10px] font-mono', totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                            {formatSignedUsd(totalPnl)}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[9px] text-muted-foreground">
                          {account.total_trades} trades · {(account.win_rate || 0).toFixed(1)}% WR
                        </p>
                        <p className="text-[9px] text-muted-foreground">
                          {account.open_positions + autotraderPositions} open · {formatUsd(account.current_capital || 0)}
                        </p>
                      </button>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </div>

          <div className="min-h-0 flex flex-col gap-2">
            <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-md border border-amber-500/25 bg-amber-500/10 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Active Desk</p>
                <p className="truncate text-[12px] font-semibold">{activeSandboxAccount?.name || 'None'}</p>
                <p className="text-[10px] text-muted-foreground">{activeSandboxAccountId ? activeSandboxAccountId : 'Select account'}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Open Positions</p>
                <p className="text-[12px] font-mono">{selectedSandboxOpenPositions}</p>
                <p className="text-[10px] text-muted-foreground">
                  {selectedSandboxOverlayOpen > 0 ? `Includes ${selectedSandboxOverlayOpen} autotrader` : 'Manual + strategy fills'}
                </p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Desk P&L</p>
                <p className={cn('text-[12px] font-mono', selectedSandboxTotalPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                  {formatSignedUsd(selectedSandboxTotalPnl)}
                </p>
                <p className="text-[10px] text-muted-foreground">ROI {formatSignedPct(activeSandboxAccount?.roi_percent || 0)}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Deployable Cash</p>
                <p className="text-[12px] font-mono">{formatUsd(Math.max(0, (activeSandboxAccount?.current_capital || 0) - selectedSandboxOverlayExposure))}</p>
                <p className="text-[10px] text-muted-foreground">
                  {selectedSandboxOverlayExposure > 0 ? `${formatUsd(selectedSandboxOverlayExposure)} auto reserved` : 'No auto reserve'}
                </p>
              </div>
            </div>

            <div className="shrink-0 flex flex-wrap items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSandboxView('overview')}
                className={cn(
                  'h-7 gap-1.5 text-[11px]',
                  sandboxView === 'overview'
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/30 hover:bg-amber-500/30 hover:text-amber-300'
                    : 'bg-card text-muted-foreground hover:text-foreground border-border'
                )}
              >
                <LayoutDashboard className="h-3.5 w-3.5" />
                Overview
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSandboxView('positions')}
                className={cn(
                  'h-7 gap-1.5 text-[11px]',
                  sandboxView === 'positions'
                    ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400'
                    : 'bg-card text-muted-foreground hover:text-foreground border-border'
                )}
              >
                <Briefcase className="h-3.5 w-3.5" />
                Positions
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSandboxView('activity')}
                className={cn(
                  'h-7 gap-1.5 text-[11px]',
                  sandboxView === 'activity'
                    ? 'bg-violet-500/20 text-violet-400 border-violet-500/30 hover:bg-violet-500/30 hover:text-violet-400'
                    : 'bg-card text-muted-foreground hover:text-foreground border-border'
                )}
              >
                <Receipt className="h-3.5 w-3.5" />
                Trade Log
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openSandboxDesk(activeSandboxAccountId || undefined)}
                className="ml-auto h-7 gap-1.5 text-[11px]"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh Desk
              </Button>
            </div>

            {sandboxView === 'overview' && (
              <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
                <div className="min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                  <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Desk Snapshot</span>
                    <span className="text-[10px] font-mono text-muted-foreground">{sandboxPositionRows.length} positions</span>
                  </div>
                  <ScrollArea className="h-[260px] xl:h-full">
                    <table className="w-full text-[11px]">
                      <thead className="sticky top-0 z-10 bg-background/95">
                        <tr className="border-b border-border/70 text-muted-foreground">
                          <th className="px-2 py-1.5 text-left">Market</th>
                          <th className="px-2 py-1.5 text-right">Side</th>
                          <th className="px-2 py-1.5 text-right">Qty</th>
                          <th className="px-2 py-1.5 text-right">Entry</th>
                          <th className="px-2 py-1.5 text-right">Mark</th>
                          <th className="px-2 py-1.5 text-right">U-P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sandboxPositionRows.length === 0 ? (
                          <tr>
                            <td colSpan={6} className="px-2 py-6 text-center text-muted-foreground">No open positions.</td>
                          </tr>
                        ) : (
                          sandboxPositionRows.map((position) => (
                            <tr key={position.id} className="border-b border-border/40">
                              <td className="px-2 py-1.5">
                                <p className="max-w-[360px] truncate">{position.market_question}</p>
                                <p className="text-[9px] text-muted-foreground">{position.market_id}</p>
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono">{position.side}</td>
                              <td className="px-2 py-1.5 text-right font-mono">{position.quantity.toFixed(2)}</td>
                              <td className="px-2 py-1.5 text-right font-mono">{position.entryPrice.toFixed(3)}</td>
                              <td className="px-2 py-1.5 text-right font-mono">{position.markPrice.toFixed(3)}</td>
                              <td className={cn('px-2 py-1.5 text-right font-mono', position.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                {formatSignedUsd(position.unrealizedPnl)}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </ScrollArea>
                </div>

                <div className="min-h-0 flex flex-col gap-2">
                  <div className="rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                    <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Strategy Mix</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{sandboxStrategyRows.length} rows</span>
                    </div>
                    <div className="space-y-1 p-2">
                      {sandboxStrategyRows.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground">No trade history available yet.</p>
                      ) : (
                        sandboxStrategyRows.slice(0, 8).map((row) => (
                          <div key={row.strategy} className="rounded border border-border/50 px-2 py-1">
                            <div className="flex items-center justify-between gap-2">
                              <span className="truncate text-[11px]">{row.strategy}</span>
                              <span className={cn('text-[10px] font-mono', row.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                {formatSignedUsd(row.pnl)}
                              </span>
                            </div>
                            <div className="mt-0.5 flex items-center justify-between text-[10px] text-muted-foreground">
                              <span>{row.trades} trades</span>
                              <span>Notional {formatUsd(row.notional)}</span>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                    <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Lifecycle Mix</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{sandboxTradeRows.length} trades</span>
                    </div>
                    <div className="space-y-1 p-2">
                      {sandboxTradeStatusRows.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground">No lifecycle data captured.</p>
                      ) : (
                        sandboxTradeStatusRows.map((row) => (
                          <div key={row.status} className="rounded border border-border/50 px-2 py-1">
                            <div className="flex items-center justify-between">
                              <span className="text-[11px] uppercase">{row.status.replace(/_/g, ' ')}</span>
                              <span className="text-[10px] font-mono text-muted-foreground">{row.count}</span>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {sandboxView === 'positions' && (
              <div className="flex-1 min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                <ScrollArea className="h-full min-h-0">
                  <table className="w-full text-[11px]">
                    <thead className="sticky top-0 z-10 bg-background/95">
                      <tr className="border-b border-border/70 text-muted-foreground">
                        <th className="px-2 py-1.5 text-left">Market</th>
                        <th className="px-2 py-1.5 text-right">Side</th>
                        <th className="px-2 py-1.5 text-right">Qty</th>
                        <th className="px-2 py-1.5 text-right">Entry Px</th>
                        <th className="px-2 py-1.5 text-right">Mark Px</th>
                        <th className="px-2 py-1.5 text-right">Cost</th>
                        <th className="px-2 py-1.5 text-right">Mkt Value</th>
                        <th className="px-2 py-1.5 text-right">U-P&L</th>
                        <th className="px-2 py-1.5 text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sandboxPositionRows.length === 0 ? (
                        <tr>
                          <td colSpan={9} className="px-2 py-8 text-center text-muted-foreground">No positions for this sandbox desk.</td>
                        </tr>
                      ) : (
                        sandboxPositionRows.map((position) => (
                          <tr key={position.id} className="border-b border-border/40">
                            <td className="px-2 py-1.5">
                              <p className="max-w-[420px] truncate">{position.market_question}</p>
                              <p className="text-[9px] text-muted-foreground">{position.market_id}</p>
                            </td>
                            <td className="px-2 py-1.5 text-right font-mono">{position.side}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{position.quantity.toFixed(2)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{position.entryPrice.toFixed(3)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{position.markPrice.toFixed(3)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(position.entryCost)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(position.marketValue)}</td>
                            <td className={cn('px-2 py-1.5 text-right font-mono', position.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                              {formatSignedUsd(position.unrealizedPnl)}
                            </td>
                            <td className="px-2 py-1.5 text-right">
                              <Badge variant="outline" className="h-4 px-1 text-[9px] uppercase">
                                {position.status}
                              </Badge>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </ScrollArea>
              </div>
            )}

            {sandboxView === 'activity' && (
              <div className="flex-1 min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                <ScrollArea className="h-full min-h-0">
                  <table className="w-full text-[11px]">
                    <thead className="sticky top-0 z-10 bg-background/95">
                      <tr className="border-b border-border/70 text-muted-foreground">
                        <th className="px-2 py-1.5 text-left">Executed</th>
                        <th className="px-2 py-1.5 text-left">Strategy</th>
                        <th className="px-2 py-1.5 text-right">Notional</th>
                        <th className="px-2 py-1.5 text-right">Expected</th>
                        <th className="px-2 py-1.5 text-right">Actual P&L</th>
                        <th className="px-2 py-1.5 text-right">Fees</th>
                        <th className="px-2 py-1.5 text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sandboxTradeRows.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="px-2 py-8 text-center text-muted-foreground">No trades for this sandbox desk.</td>
                        </tr>
                      ) : (
                        sandboxTradeRows.map((trade) => (
                          <tr key={trade.id} className="border-b border-border/40">
                            <td className="px-2 py-1.5 text-[10px] text-muted-foreground">
                              {new Date(trade.executed_at).toLocaleString()}
                            </td>
                            <td className="px-2 py-1.5">
                              <p className="font-medium">{trade.strategy_type}</p>
                              <p className="text-[9px] text-muted-foreground">{trade.opportunity_id}</p>
                            </td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(trade.total_cost)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(trade.expected_profit || 0)}</td>
                            <td className={cn('px-2 py-1.5 text-right font-mono', toFiniteNumber(trade.actual_pnl) >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                              {trade.actual_pnl == null ? '—' : formatSignedUsd(toFiniteNumber(trade.actual_pnl))}
                            </td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(trade.fees_paid || 0)}</td>
                            <td className="px-2 py-1.5 text-right">
                              <Badge variant="outline" className={cn('h-4 px-1 text-[9px] uppercase', tradeStatusClass(String(trade.status || 'unknown')))}>
                                {String(trade.status || 'unknown').replace(/_/g, ' ')}
                              </Badge>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </ScrollArea>
              </div>
            )}
          </div>
        </div>
      )}
      {workspaceTab === 'live' && (
        <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[250px_minmax(0,1fr)]">
          <div className="hidden xl:flex min-h-0 flex-col rounded-lg border border-border/70 bg-card overflow-hidden">
            <div className="shrink-0 border-b border-border/50 px-2.5 py-2">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Live Venues</p>
              <p className="text-[10px] text-muted-foreground">{liveMetrics.connectedVenues}/2 connected</p>
            </div>
            <ScrollArea className="flex-1 min-h-0">
              <div className="space-y-1.5 p-1.5">
                {venueSnapshots.map((venue) => {
                  const isActive = selectedAccountId === `live:${venue.id}` || (!selectedAccountId?.startsWith('live:') && venue.id === 'polymarket')
                  return (
                    <button
                      key={venue.id}
                      type="button"
                      onClick={() => {
                        setSelectedAccountId(`live:${venue.id}`)
                        setAccountMode('live')
                      }}
                      className={cn(
                        'w-full rounded-md px-2 py-1.5 text-left transition-colors',
                        isActive
                          ? 'bg-emerald-500/15 text-foreground'
                          : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-[11px] font-medium">{venue.label}</p>
                        <span className={cn('h-1.5 w-1.5 rounded-full', venue.connected ? 'bg-emerald-400' : 'bg-amber-400')} />
                      </div>
                      <p className="mt-0.5 text-[9px] text-muted-foreground">{venue.accountLabel}</p>
                      <p className="text-[9px] text-muted-foreground">
                        {formatUsd(venue.balance)} cash · {venue.openPositions} positions
                      </p>
                      <p className={cn('text-[9px] font-mono', venue.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                        {formatSignedUsd(venue.unrealizedPnl)}
                      </p>
                    </button>
                  )
                })}
              </div>
            </ScrollArea>
          </div>

          <div className="min-h-0 flex flex-col gap-2">
            <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Active Venue</p>
                <p className="text-[12px] font-semibold">{activeLiveSnapshot.label}</p>
                <p className={cn('text-[10px]', activeLiveSnapshot.connected ? 'text-emerald-400' : 'text-amber-400')}>
                  {activeLiveSnapshot.connected ? 'Connected' : 'Disconnected'}
                </p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Free Cash</p>
                <p className="text-[12px] font-mono">{formatUsd(activeLiveSnapshot.available)}</p>
                <p className="text-[10px] text-muted-foreground">Balance {formatUsd(activeLiveSnapshot.balance)}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Open Risk</p>
                <p className="text-[12px] font-mono">{activeLiveSnapshot.openPositions} positions</p>
                <p className="text-[10px] text-muted-foreground">Exposure {formatUsd(activeLiveSnapshot.exposure)}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Recent Orders</p>
                <p className="text-[12px] font-mono">{liveOpenOrderCount} open</p>
                <p className="text-[10px] text-muted-foreground">{liveOrderRows.length} total cached</p>
              </div>
            </div>

            <div className="shrink-0 flex flex-wrap items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setLiveView('overview')}
                className={cn(
                  'h-7 gap-1.5 text-[11px]',
                  liveView === 'overview'
                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30 hover:text-emerald-400'
                    : 'bg-card text-muted-foreground hover:text-foreground border-border'
                )}
              >
                <LayoutDashboard className="h-3.5 w-3.5" />
                Overview
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setLiveView('positions')}
                className={cn(
                  'h-7 gap-1.5 text-[11px]',
                  liveView === 'positions'
                    ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400'
                    : 'bg-card text-muted-foreground hover:text-foreground border-border'
                )}
              >
                <Briefcase className="h-3.5 w-3.5" />
                Positions
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setLiveView('activity')}
                className={cn(
                  'h-7 gap-1.5 text-[11px]',
                  liveView === 'activity'
                    ? 'bg-violet-500/20 text-violet-400 border-violet-500/30 hover:bg-violet-500/30 hover:text-violet-400'
                    : 'bg-card text-muted-foreground hover:text-foreground border-border'
                )}
              >
                <ListChecks className="h-3.5 w-3.5" />
                Orders
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openLiveDesk(activeLiveVenue)}
                className="ml-auto h-7 gap-1.5 text-[11px]"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh Venue
              </Button>
            </div>

            {liveView === 'overview' && (
              <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
                <div className="min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                  <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Venue Balance Sheet</span>
                    <span className="text-[10px] font-mono text-muted-foreground">Fleet {formatUsd(liveMetrics.totalBalance)}</span>
                  </div>
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="border-b border-border/60 text-muted-foreground">
                        <th className="px-2 py-1.5 text-left">Venue</th>
                        <th className="px-2 py-1.5 text-right">Balance</th>
                        <th className="px-2 py-1.5 text-right">Available</th>
                        <th className="px-2 py-1.5 text-right">Exposure</th>
                        <th className="px-2 py-1.5 text-right">U-P&L</th>
                        <th className="px-2 py-1.5 text-right">State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {venueSnapshots.map((venue) => (
                        <tr key={venue.id} className="border-b border-border/40">
                          <td className="px-2 py-1.5">
                            <p className="font-medium">{venue.label}</p>
                            <p className="text-[9px] text-muted-foreground">{venue.accountLabel}</p>
                          </td>
                          <td className="px-2 py-1.5 text-right font-mono">{formatUsd(venue.balance)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{formatUsd(venue.available)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{formatUsd(venue.exposure)}</td>
                          <td className={cn('px-2 py-1.5 text-right font-mono', venue.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                            {formatSignedUsd(venue.unrealizedPnl)}
                          </td>
                          <td className="px-2 py-1.5 text-right">
                            <Badge variant="outline" className={cn('h-4 px-1 text-[9px]', venue.connected ? 'border-emerald-500/40 text-emerald-300' : 'border-amber-500/40 text-amber-300')}>
                              {venue.connected ? 'Connected' : 'Offline'}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="min-h-0 flex flex-col gap-2">
                  <div className="rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                    <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Polymarket Limits</span>
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                    </div>
                    <div className="grid grid-cols-2 gap-1.5 p-2">
                      <MetricPair label="Max Trade" value={formatUsd(toFiniteNumber(tradingStatus?.limits.max_trade_size_usd))} />
                      <MetricPair label="Max Daily" value={formatUsd(toFiniteNumber(tradingStatus?.limits.max_daily_volume))} />
                      <MetricPair label="Max Open" value={`${toFiniteNumber(tradingStatus?.limits.max_open_positions)}`} />
                      <MetricPair label="Min Order" value={formatUsd(toFiniteNumber(tradingStatus?.limits.min_order_size_usd))} />
                    </div>
                  </div>
                  <div className="rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                    <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Active Venue Book</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{activeLivePositions.length} positions</span>
                    </div>
                    <div className="space-y-1 p-2">
                      {activeLivePositions.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground">No open positions on {activeLiveSnapshot.label}.</p>
                      ) : (
                        activeLivePositions.slice(0, 8).map((row) => (
                          <div key={row.id} className="rounded border border-border/50 px-2 py-1">
                            <p className="truncate text-[11px]" title={row.marketQuestion}>{row.marketQuestion}</p>
                            <div className="mt-0.5 flex items-center justify-between text-[10px]">
                              <span className="text-muted-foreground">{row.outcome} · {row.size.toFixed(2)}</span>
                              <span className={cn('font-mono', row.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                {formatSignedUsd(row.unrealizedPnl)}
                              </span>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {liveView === 'positions' && (
              <div className="flex-1 min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                <ScrollArea className="h-full min-h-0">
                  <table className="w-full text-[11px]">
                    <thead className="sticky top-0 z-10 bg-background/95">
                      <tr className="border-b border-border/70 text-muted-foreground">
                        <th className="px-2 py-1.5 text-left">Market</th>
                        <th className="px-2 py-1.5 text-right">Venue</th>
                        <th className="px-2 py-1.5 text-right">Side</th>
                        <th className="px-2 py-1.5 text-right">Size</th>
                        <th className="px-2 py-1.5 text-right">Avg</th>
                        <th className="px-2 py-1.5 text-right">Mark</th>
                        <th className="px-2 py-1.5 text-right">Cost</th>
                        <th className="px-2 py-1.5 text-right">Mkt Value</th>
                        <th className="px-2 py-1.5 text-right">U-P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {livePositionRows.length === 0 ? (
                        <tr>
                          <td colSpan={9} className="px-2 py-8 text-center text-muted-foreground">No live positions across connected venues.</td>
                        </tr>
                      ) : (
                        livePositionRows.map((row) => (
                          <tr key={row.id} className="border-b border-border/40">
                            <td className="px-2 py-1.5">
                              <p className="max-w-[420px] truncate">{row.marketQuestion}</p>
                              <p className="text-[9px] text-muted-foreground">{row.marketId}</p>
                            </td>
                            <td className="px-2 py-1.5 text-right">
                              <Badge variant="outline" className={cn('h-4 px-1 text-[9px]', row.venue === 'Polymarket' ? 'border-cyan-500/40 text-cyan-300' : 'border-indigo-500/40 text-indigo-300')}>
                                {row.venue}
                              </Badge>
                            </td>
                            <td className="px-2 py-1.5 text-right font-mono">{row.outcome}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{row.size.toFixed(2)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{row.entryPrice.toFixed(3)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{row.markPrice.toFixed(3)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(row.costBasis)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{formatUsd(row.marketValue)}</td>
                            <td className={cn('px-2 py-1.5 text-right font-mono', row.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                              {formatSignedUsd(row.unrealizedPnl)}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </ScrollArea>
              </div>
            )}

            {liveView === 'activity' && (
              <div className="flex-1 min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                <ScrollArea className="h-full min-h-0">
                  <table className="w-full text-[11px]">
                    <thead className="sticky top-0 z-10 bg-background/95">
                      <tr className="border-b border-border/70 text-muted-foreground">
                        <th className="px-2 py-1.5 text-left">Created</th>
                        <th className="px-2 py-1.5 text-left">Market</th>
                        <th className="px-2 py-1.5 text-right">Side</th>
                        <th className="px-2 py-1.5 text-right">Type</th>
                        <th className="px-2 py-1.5 text-right">Size</th>
                        <th className="px-2 py-1.5 text-right">Price</th>
                        <th className="px-2 py-1.5 text-right">Filled</th>
                        <th className="px-2 py-1.5 text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {liveOrderRows.length === 0 ? (
                        <tr>
                          <td colSpan={8} className="px-2 py-8 text-center text-muted-foreground">No live order history available.</td>
                        </tr>
                      ) : (
                        liveOrderRows.map((order) => (
                          <tr key={order.id} className="border-b border-border/40">
                            <td className="px-2 py-1.5 text-[10px] text-muted-foreground">
                              {new Date(order.created_at).toLocaleString()}
                            </td>
                            <td className="px-2 py-1.5">
                              <p className="max-w-[360px] truncate">{order.market_question || order.token_id}</p>
                              <p className="text-[9px] text-muted-foreground">{order.token_id}</p>
                            </td>
                            <td className="px-2 py-1.5 text-right font-mono">{order.side}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{order.order_type}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{toFiniteNumber(order.size).toFixed(2)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{toFiniteNumber(order.price).toFixed(3)}</td>
                            <td className="px-2 py-1.5 text-right font-mono">{toFiniteNumber(order.filled_size).toFixed(2)}</td>
                            <td className="px-2 py-1.5 text-right">
                              <Badge variant="outline" className={cn('h-4 px-1 text-[9px] uppercase', liveOrderStatusClass(order.status))}>
                                {order.status}
                              </Badge>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </ScrollArea>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function DenseMetric({
  label,
  value,
  hint,
  icon: Icon,
  tone = 'neutral',
}: {
  label: string
  value: string
  hint: string
  icon: React.ElementType
  tone?: 'neutral' | 'green' | 'red'
}) {
  return (
    <div className="rounded-lg border border-border/70 bg-background/40 px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.08em] text-muted-foreground/70">
        <Icon className="h-3.5 w-3.5" />
        <span>{label}</span>
      </div>
      <p
        className={cn(
          'mt-1 font-mono text-sm font-semibold',
          tone === 'neutral' && 'text-foreground',
          tone === 'green' && 'text-green-400',
          tone === 'red' && 'text-red-400'
        )}
      >
        {value}
      </p>
      <p className="text-[11px] text-muted-foreground">{hint}</p>
    </div>
  )
}

function MetricPair({
  label,
  value,
  valueClass,
}: {
  label: string
  value: string
  valueClass?: string
}) {
  return (
    <div className="rounded-md border border-border/50 bg-background/30 px-2 py-1.5">
      <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground/70">{label}</p>
      <p className={cn('font-mono text-xs text-foreground', valueClass)}>{value}</p>
    </div>
  )
}
