import { useMemo, useState } from 'react'
import { useAtom } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
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
const OPEN_SHADOW_ORDER_STATUSES = new Set(['submitted', 'executed', 'open'])

const WORKSPACE_TAB_CONFIG: { id: AccountsWorkspaceTab; icon: React.ElementType }[] = [
  { id: 'overview', icon: LayoutDashboard },
  { id: 'sandbox', icon: Shield },
  { id: 'live', icon: Zap },
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

function formatNativeGas(value: number): string {
  return `${value.toFixed(4)} POL`
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
  const { t } = useTranslation()
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

  const shadowModeActive = Boolean(orchestratorOverview?.control?.is_enabled)
    && String(orchestratorOverview?.control?.mode || '').toLowerCase() === 'shadow'

  const { data: traderOrders = [] } = useQuery({
    queryKey: ['accounts-panel', 'trader-orders'],
    queryFn: async () => {
      try {
        return await getAllTraderOrders(300)
      } catch {
        return []
      }
    },
    enabled: shadowModeActive,
    refetchInterval: 10000,
    retry: false,
  })

  const autotraderShadowMetrics = useMemo(() => {
    const linkedSandboxAccountId: string | null = null

    const openShadowOrders = traderOrders.filter((order) => {
      const mode = String(order.mode || '').toLowerCase()
      const status = String(order.status || '').toLowerCase()
      return mode === 'shadow' && OPEN_SHADOW_ORDER_STATUSES.has(status)
    })

    const positionKeys = new Set<string>()
    let exposureUsd = 0

    for (const order of openShadowOrders) {
      const marketId = String(order.market_id || '').trim()
      if (!marketId) continue
      const side = normalizeDirection(order.direction_side ?? order.direction)
      positionKeys.add(`${marketId}:${side}`)
      exposureUsd += Math.abs(Number(order.notional_usd || 0))
    }

    const fallbackExposure = Number(orchestratorOverview?.metrics?.gross_exposure_usd || 0)

    return {
      active: shadowModeActive,
      linkedSandboxAccountId,
      openOrders: openShadowOrders.length,
      openPositions: positionKeys.size,
      exposureUsd: exposureUsd > 0 ? exposureUsd : fallbackExposure,
    }
  }, [shadowModeActive, orchestratorOverview, traderOrders])

  const autotraderOverlay = useMemo(() => {
    const linkedAccount = autotraderShadowMetrics.linkedSandboxAccountId
      ? sandboxAccounts.find((account) => account.id === autotraderShadowMetrics.linkedSandboxAccountId)
      : undefined

    const shouldOverlay = Boolean(
      linkedAccount
      && autotraderShadowMetrics.openPositions > 0
      && (linkedAccount.open_positions || 0) === 0
    )

    return {
      accountId: shouldOverlay ? autotraderShadowMetrics.linkedSandboxAccountId : null,
      openPositions: shouldOverlay ? autotraderShadowMetrics.openPositions : 0,
      openOrders: shouldOverlay ? autotraderShadowMetrics.openOrders : 0,
      exposureUsd: shouldOverlay ? autotraderShadowMetrics.exposureUsd : 0,
    }
  }, [autotraderShadowMetrics, sandboxAccounts])

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
    const visibleTradingPositions = polymarketReady
      ? tradingPositions.filter((position) => toFiniteNumber(position.size) > 0)
      : []
    const openRiskTradingPositions = visibleTradingPositions.filter((position) => position.counts_as_open !== false)
    const exposure = visibleTradingPositions.reduce(
      (sum, pos) => sum + toFiniteNumber(pos.size) * toFiniteNumber(pos.current_price),
      0
    )
    const unrealizedPnl = visibleTradingPositions.reduce((sum, pos) => sum + toFiniteNumber(pos.unrealized_pnl), 0)
    return {
      id: 'polymarket',
      label: 'Polymarket',
      connected: polymarketReady,
      accountLabel: tradingStatus?.wallet_address
        ? `${tradingStatus.wallet_address.slice(0, 8)}...${tradingStatus.wallet_address.slice(-6)}`
        : t('accounts.noWallet'),
      balance: polymarketReady ? toFiniteNumber(tradingBalance?.balance) : 0,
      available: polymarketReady ? toFiniteNumber(tradingBalance?.available) : 0,
      exposure,
      openPositions: openRiskTradingPositions.length,
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
    const kalshiConnected = Boolean(kalshiStatus?.authenticated)
    const visibleKalshiPositions = kalshiConnected
      ? kalshiPositions.filter((position) => toFiniteNumber(position.size) > 0)
      : []
    const exposure = visibleKalshiPositions.reduce(
      (sum, pos) => sum + toFiniteNumber(pos.size) * toFiniteNumber(pos.current_price),
      0
    )
    const unrealizedPnl = visibleKalshiPositions.reduce((sum, pos) => sum + toFiniteNumber(pos.unrealized_pnl), 0)
    return {
      id: 'kalshi',
      label: 'Kalshi',
      connected: kalshiConnected,
      accountLabel: kalshiStatus?.email || (kalshiStatus?.member_id ? t('accounts.memberPrefix', { id: kalshiStatus.member_id }) : t('accounts.noAccount')),
      balance: kalshiConnected ? toFiniteNumber(kalshiBalance?.balance ?? kalshiStatus?.balance?.balance) : 0,
      available: kalshiConnected ? toFiniteNumber(kalshiBalance?.available ?? kalshiStatus?.balance?.available) : 0,
      exposure,
      openPositions: visibleKalshiPositions.length,
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
        modeLabel: t('accounts.modeLive'),
        accountLabel: venue.label,
        status: venue.connected ? t('accounts.connected') : t('accounts.disconnected'),
        tone: venue.connected ? 'green' : 'amber',
      }
    }

    if (activeSandboxAccount) {
      return {
        modeLabel: t('accounts.modeSandbox'),
        accountLabel: activeSandboxAccount.name,
        status: t('accounts.tradesCount', { n: activeSandboxAccount.total_trades }),
        tone: 'blue',
      }
    }

    return {
      modeLabel: accountMode === 'live' ? t('accounts.modeLive') : t('accounts.modeSandbox'),
      accountLabel: t('accounts.noAccountSelected'),
      status: t('accounts.selectAccountFromHeader'),
      tone: 'neutral',
    }
  }, [selectedAccountId, accountMode, activeSandboxAccount, kalshiSnapshot, polymarketSnapshot, t])

  const allocationRows = useMemo(() => {
    const sandboxRows = sandboxAccounts.map((account) => ({
      id: account.id,
      label: `${t('accounts.modeSandbox')} · ${account.name}`,
      value: account.current_capital || 0,
      tone: 'amber' as const,
    }))

    const liveRows = venueSnapshots.map((venue) => ({
      id: `live:${venue.id}`,
      label: `${t('accounts.modeLive')} · ${venue.label}`,
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
  }, [sandboxAccounts, venueSnapshots, t])

  const riskSignals = useMemo(() => {
    return [
      {
        label: sandboxMetrics.count === 0 ? t('accounts.riskNoSandbox') : t('accounts.riskSandboxOnline', { n: sandboxMetrics.count }),
        tone: sandboxMetrics.count === 0 ? 'amber' : 'green',
      },
      {
        label: liveMetrics.connectedVenues === 2 ? t('accounts.riskBothVenues') : t('accounts.riskVenuesConnected', { n: liveMetrics.connectedVenues }),
        tone: liveMetrics.connectedVenues === 2 ? 'green' : 'amber',
      },
      {
        label: liveMetrics.totalUnrealizedPnl >= 0 ? t('accounts.riskBookGreen') : t('accounts.riskBookRed'),
        tone: liveMetrics.totalUnrealizedPnl >= 0 ? 'green' : 'red',
      },
      {
        label: sandboxMetrics.totalOpenPositions + liveMetrics.totalOpenPositions > 40
          ? t('accounts.riskHighPositions')
          : t('accounts.riskNormalPositions'),
        tone: sandboxMetrics.totalOpenPositions + liveMetrics.totalOpenPositions > 40 ? 'amber' : 'green',
      },
      {
        label: sandboxMetrics.autotraderOverlay.openPositions > 0
          ? t('accounts.riskAutotraderActive', { n: sandboxMetrics.autotraderOverlay.openPositions })
          : t('accounts.riskNoAutotrader'),
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
    t,
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
    const polymarketRows = (polymarketReady ? tradingPositions : [])
      .filter((position) => toFiniteNumber(position.size) > 0)
      .map((position) => {
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

    const kalshiRows = (kalshiStatus?.authenticated ? kalshiPositions : [])
      .filter((position) => toFiniteNumber(position.size) > 0)
      .map((position) => {
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
  }, [polymarketReady, tradingPositions, kalshiStatus?.authenticated, kalshiPositions])

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
              <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground/70">{t('accounts.commandCenter')}</p>
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
                {t('accounts.accountSettings')}
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 p-2.5 md:grid-cols-4 xl:grid-cols-8">
            <DenseMetric
              label={t('accounts.simEquity')}
              value={formatUsd(sandboxMetrics.deployableCapital)}
              hint={
                sandboxMetrics.autotraderOverlay.exposureUsd > 0
                  ? t('accounts.ledgerDeployedHint', { ledger: formatUsd(sandboxMetrics.totalCapital), deployed: formatUsd(sandboxMetrics.autotraderOverlay.exposureUsd) })
                  : t('accounts.accountsCount', { n: sandboxMetrics.count })
              }
              icon={Wallet}
            />
            <DenseMetric
              label={t('accounts.simPnl')}
              value={formatSignedUsd(sandboxMetrics.totalPnl)}
              hint={t('accounts.tradesCount', { n: sandboxMetrics.totalTrades })}
              icon={sandboxMetrics.totalPnl >= 0 ? TrendingUp : TrendingDown}
              tone={sandboxMetrics.totalPnl >= 0 ? 'green' : 'red'}
            />
            <DenseMetric
              label={t('accounts.simRoi')}
              value={formatSignedPct(sandboxMetrics.roi)}
              hint={t('accounts.aggregate')}
              icon={BarChart3}
              tone={sandboxMetrics.roi >= 0 ? 'green' : 'red'}
            />
            <DenseMetric
              label={t('accounts.simPositions')}
              value={sandboxMetrics.totalOpenPositions.toString()}
              hint={
                sandboxMetrics.autotraderOverlay.openPositions > 0
                  ? t('accounts.openWithAutotrader', { n: sandboxMetrics.autotraderOverlay.openPositions })
                  : t('accounts.openLabel')
              }
              icon={Briefcase}
            />
            <DenseMetric
              label={t('accounts.liveFreeCash')}
              value={formatUsd(liveMetrics.totalAvailable)}
              hint={t('accounts.venuesConnected', { n: liveMetrics.connectedVenues })}
              icon={DollarSign}
            />
            <DenseMetric
              label={t('accounts.liveExposure')}
              value={formatUsd(liveMetrics.totalExposure)}
              hint={t('accounts.openPositionsCount', { n: liveMetrics.totalOpenPositions })}
              icon={Activity}
            />
            <DenseMetric
              label={t('accounts.liveUnrealized')}
              value={formatSignedUsd(liveMetrics.totalUnrealizedPnl)}
              hint={t('accounts.crossVenue')}
              icon={liveMetrics.totalUnrealizedPnl >= 0 ? TrendingUp : TrendingDown}
              tone={liveMetrics.totalUnrealizedPnl >= 0 ? 'green' : 'red'}
            />
            <DenseMetric
              label={t('accounts.fleetBalance')}
              value={formatUsd(sandboxMetrics.totalCapital + liveMetrics.totalBalance)}
              hint={t('accounts.simulationPlusLive')}
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
              {tab.id === 'overview' ? t('accounts.tabOverview') : tab.id === 'sandbox' ? t('accounts.tabSandboxDesk') : t('accounts.tabLiveDesk')}
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
                    <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">{t('accounts.sandboxFleet')}</p>
                    <p className="text-xs text-muted-foreground">{t('accounts.sandboxFleetDesc')}</p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openSandboxDesk()}
                    className="h-6 text-[11px]"
                  >
                    {t('accounts.openSandboxDesk')}
                  </Button>
                </div>

                {sandboxAccounts.length === 0 ? (
                  <div className="px-4 py-8 text-center">
                    <Shield className="mx-auto mb-2 h-6 w-6 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">{t('accounts.noSandboxAccountsYet')}</p>
                  </div>
                ) : (
                  <div className="min-h-0 flex-1 overflow-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-border/70 text-muted-foreground">
                          <th className="px-4 py-2 text-left">{t('accounts.colAccount')}</th>
                          <th className="px-3 py-2 text-right">{t('accounts.colCapital')}</th>
                          <th className="px-3 py-2 text-right">{t('accounts.colPnl')}</th>
                          <th className="px-3 py-2 text-right">{t('accounts.colRoi')}</th>
                          <th className="px-3 py-2 text-right">{t('accounts.colTrades')}</th>
                          <th className="px-3 py-2 text-right">{t('accounts.colOpen')}</th>
                          <th className="px-4 py-2 text-right">{t('accounts.colDesk')}</th>
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
                                    {t('accounts.winRateLine', { rate: account.win_rate.toFixed(1), w: account.winning_trades, l: account.losing_trades })}
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
                                    {t('accounts.autoExposureLine', { n: autotraderOpenPositions, value: formatUsd(autotraderExposure) })}
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
                                  {t('accounts.open')}
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
                  <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">{t('accounts.liveVenues')}</p>
                  <p className="text-xs text-muted-foreground">{t('accounts.liveVenuesDesc')}</p>
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
                        {venue.connected ? t('accounts.connected') : t('accounts.disconnected')}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <MetricPair label={t('accounts.balance')} value={formatUsd(venue.balance)} />
                      <MetricPair label={t('accounts.available')} value={formatUsd(venue.available)} />
                      <MetricPair label={t('accounts.exposure')} value={formatUsd(venue.exposure)} />
                      <MetricPair
                        label={t('accounts.unrealized')}
                        value={formatSignedUsd(venue.unrealizedPnl)}
                        valueClass={venue.unrealizedPnl >= 0 ? 'text-green-300' : 'text-red-300'}
                      />
                    </div>
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-muted-foreground">
                      <span>{t('accounts.openPositionsCount', { n: venue.openPositions })}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => openLiveDesk(venue.id)}
                        className="h-6 px-2 text-[11px]"
                      >
                        {t('accounts.focusDesk')}
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
                  <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">{t('accounts.allocationRiskRadar')}</p>
                  <p className="text-xs text-muted-foreground">{t('accounts.allocationRiskRadarDesc')}</p>
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
                  <p className="text-xs text-muted-foreground">{t('accounts.noAccountBalances')}</p>
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
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.sandboxAccounts')}</p>
              <p className="text-[10px] text-muted-foreground">{t('accounts.desksConfigured', { n: sandboxAccounts.length })}</p>
            </div>
            <ScrollArea className="flex-1 min-h-0">
              <div className="space-y-1.5 p-1.5">
                {sandboxAccounts.length === 0 ? (
                  <p className="px-2 py-6 text-center text-[11px] text-muted-foreground">{t('accounts.noSandboxConfigured')}</p>
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
                          {t('accounts.tradesWrLine', { n: account.total_trades, rate: (account.win_rate || 0).toFixed(1) })}
                        </p>
                        <p className="text-[9px] text-muted-foreground">
                          {t('accounts.openCapitalLine', { n: account.open_positions + autotraderPositions, value: formatUsd(account.current_capital || 0) })}
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
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.activeDesk')}</p>
                <p className="truncate text-[12px] font-semibold">{activeSandboxAccount?.name || t('accounts.none')}</p>
                <p className="text-[10px] text-muted-foreground">{activeSandboxAccountId ? activeSandboxAccountId : t('accounts.selectAccount')}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.openPositions')}</p>
                <p className="text-[12px] font-mono">{selectedSandboxOpenPositions}</p>
                <p className="text-[10px] text-muted-foreground">
                  {selectedSandboxOverlayOpen > 0 ? t('accounts.includesAutotrader', { n: selectedSandboxOverlayOpen }) : t('accounts.manualStrategyFills')}
                </p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.deskPnl')}</p>
                <p className={cn('text-[12px] font-mono', selectedSandboxTotalPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                  {formatSignedUsd(selectedSandboxTotalPnl)}
                </p>
                <p className="text-[10px] text-muted-foreground">{t('accounts.roiPrefix', { value: formatSignedPct(activeSandboxAccount?.roi_percent || 0) })}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.deployableCash')}</p>
                <p className="text-[12px] font-mono">{formatUsd(Math.max(0, (activeSandboxAccount?.current_capital || 0) - selectedSandboxOverlayExposure))}</p>
                <p className="text-[10px] text-muted-foreground">
                  {selectedSandboxOverlayExposure > 0 ? t('accounts.autoReserved', { value: formatUsd(selectedSandboxOverlayExposure) }) : t('accounts.noAutoReserve')}
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
                {t('accounts.viewOverview')}
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
                {t('accounts.viewPositions')}
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
                {t('accounts.viewTradeLog')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openSandboxDesk(activeSandboxAccountId || undefined)}
                className="ml-auto h-7 gap-1.5 text-[11px]"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                {t('accounts.refreshDesk')}
              </Button>
            </div>

            {sandboxView === 'overview' && (
              <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
                <div className="min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                  <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.deskSnapshot')}</span>
                    <span className="text-[10px] font-mono text-muted-foreground">{t('accounts.positionsCount', { n: sandboxPositionRows.length })}</span>
                  </div>
                  <ScrollArea className="h-[260px] xl:h-full">
                    <table className="w-full text-[11px]">
                      <thead className="sticky top-0 z-10 bg-background/95">
                        <tr className="border-b border-border/70 text-muted-foreground">
                          <th className="px-2 py-1.5 text-left">{t('accounts.colMarket')}</th>
                          <th className="px-2 py-1.5 text-right">{t('accounts.colSide')}</th>
                          <th className="px-2 py-1.5 text-right">{t('accounts.colQty')}</th>
                          <th className="px-2 py-1.5 text-right">{t('accounts.colEntry')}</th>
                          <th className="px-2 py-1.5 text-right">{t('accounts.colMark')}</th>
                          <th className="px-2 py-1.5 text-right">{t('accounts.colUPnl')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sandboxPositionRows.length === 0 ? (
                          <tr>
                            <td colSpan={6} className="px-2 py-6 text-center text-muted-foreground">{t('accounts.noOpenPositions')}</td>
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
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.strategyMix')}</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{t('accounts.rowsCount', { n: sandboxStrategyRows.length })}</span>
                    </div>
                    <div className="space-y-1 p-2">
                      {sandboxStrategyRows.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground">{t('accounts.noTradeHistory')}</p>
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
                              <span>{t('accounts.tradesCount', { n: row.trades })}</span>
                              <span>{t('accounts.notionalLabel', { value: formatUsd(row.notional) })}</span>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                    <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.lifecycleMix')}</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{t('accounts.tradesCount', { n: sandboxTradeRows.length })}</span>
                    </div>
                    <div className="space-y-1 p-2">
                      {sandboxTradeStatusRows.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground">{t('accounts.noLifecycleData')}</p>
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
                        <th className="px-2 py-1.5 text-left">{t('accounts.colMarket')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colSide')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colQty')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colEntryPx')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colMarkPx')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colCost')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colMktValue')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colUPnl')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colStatus')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sandboxPositionRows.length === 0 ? (
                        <tr>
                          <td colSpan={9} className="px-2 py-8 text-center text-muted-foreground">{t('accounts.noPositionsForDesk')}</td>
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
                        <th className="px-2 py-1.5 text-left">{t('accounts.colExecuted')}</th>
                        <th className="px-2 py-1.5 text-left">{t('accounts.colStrategy')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colNotional')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colExpected')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colActualPnl')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colFees')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colStatus')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sandboxTradeRows.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="px-2 py-8 text-center text-muted-foreground">{t('accounts.noTradesForDesk')}</td>
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
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.liveVenues')}</p>
              <p className="text-[10px] text-muted-foreground">{t('accounts.venuesConnectedShort', { n: liveMetrics.connectedVenues })}</p>
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
                        {t('accounts.cashPositionsLine', { value: formatUsd(venue.balance), n: venue.openPositions })}
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
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.activeVenue')}</p>
                <p className="text-[12px] font-semibold">{activeLiveSnapshot.label}</p>
                <p className={cn('text-[10px]', activeLiveSnapshot.connected ? 'text-emerald-400' : 'text-amber-400')}>
                  {activeLiveSnapshot.connected ? t('accounts.connected') : t('accounts.disconnected')}
                </p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.freeCash')}</p>
                <p className="text-[12px] font-mono">{formatUsd(activeLiveSnapshot.available)}</p>
                <p className="text-[10px] text-muted-foreground">{t('accounts.balanceLabel', { value: formatUsd(activeLiveSnapshot.balance) })}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.openRisk')}</p>
                <p className="text-[12px] font-mono">{t('accounts.positionsCount', { n: activeLiveSnapshot.openPositions })}</p>
                <p className="text-[10px] text-muted-foreground">{t('accounts.exposureLabel', { value: formatUsd(activeLiveSnapshot.exposure) })}</p>
              </div>
              <div className="rounded-md border border-border/60 bg-background/70 px-2.5 py-1.5">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('accounts.recentOrders')}</p>
                <p className="text-[12px] font-mono">{t('accounts.openCount', { n: liveOpenOrderCount })}</p>
                <p className="text-[10px] text-muted-foreground">{t('accounts.totalCached', { n: liveOrderRows.length })}</p>
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
                {t('accounts.viewOverview')}
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
                {t('accounts.viewPositions')}
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
                {t('accounts.viewOrders')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openLiveDesk(activeLiveVenue)}
                className="ml-auto h-7 gap-1.5 text-[11px]"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                {t('accounts.refreshVenue')}
              </Button>
            </div>

            {liveView === 'overview' && (
              <div className="flex-1 min-h-0 grid gap-2 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
                <div className="min-h-0 rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                  <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.venueBalanceSheet')}</span>
                    <span className="text-[10px] font-mono text-muted-foreground">{t('accounts.fleetLabel', { value: formatUsd(liveMetrics.totalBalance) })}</span>
                  </div>
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="border-b border-border/60 text-muted-foreground">
                        <th className="px-2 py-1.5 text-left">{t('accounts.colVenue')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.balance')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.available')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.exposure')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colUPnl')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colState')}</th>
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
                              {venue.connected ? t('accounts.connected') : t('accounts.offline')}
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
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.polymarketLimits')}</span>
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                    </div>
                    <div className="grid grid-cols-2 gap-1.5 p-2">
                      <MetricPair label={t('accounts.maxTrade')} value={formatUsd(toFiniteNumber(tradingStatus?.limits.max_trade_size_usd))} />
                      <MetricPair label={t('accounts.maxDaily')} value={formatUsd(toFiniteNumber(tradingStatus?.limits.max_daily_volume))} />
                      <MetricPair label={t('accounts.minOrder')} value={formatUsd(toFiniteNumber(tradingStatus?.limits.min_order_size_usd))} />
                      <MetricPair
                        label={t('accounts.nativeGas')}
                        value={tradingStatus?.native_gas ? formatNativeGas(toFiniteNumber(tradingStatus.native_gas.balance_native)) : '--'}
                        valueClass={
                          tradingStatus?.native_gas
                            ? (toFiniteNumber(tradingStatus.native_gas.balance_native) > 0 ? 'text-emerald-300' : 'text-red-300')
                            : undefined
                        }
                      />
                      <MetricPair
                        label={t('accounts.gasNeeded')}
                        value={tradingStatus?.native_gas ? formatNativeGas(toFiniteNumber(tradingStatus.native_gas.required_native_for_approval)) : '--'}
                      />
                      <MetricPair
                        label={t('accounts.gasReady')}
                        value={tradingStatus?.native_gas ? (tradingStatus.native_gas.affordable_for_approval ? t('accounts.yes') : t('accounts.no')) : '--'}
                        valueClass={
                          tradingStatus?.native_gas
                            ? (tradingStatus.native_gas.affordable_for_approval ? 'text-emerald-300' : 'text-red-300')
                            : undefined
                        }
                      />
                      <MetricPair
                        label={t('accounts.orderPath')}
                        value={
                          tradingStatus?.execution_paths?.normal_trading === 'clob_only'
                            ? t('accounts.clobOnly')
                            : '--'
                        }
                      />
                    </div>
                  </div>
                  <div className="rounded-lg border border-border/70 bg-card/80 overflow-hidden">
                    <div className="px-2.5 py-2 border-b border-border/50 flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('accounts.activeVenueBook')}</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{t('accounts.positionsCount', { n: activeLivePositions.length })}</span>
                    </div>
                    <div className="space-y-1 p-2">
                      {activeLivePositions.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground">{t('accounts.noOpenPositionsOn', { venue: activeLiveSnapshot.label })}</p>
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
                        <th className="px-2 py-1.5 text-left">{t('accounts.colMarket')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colVenue')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colSide')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colSize')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colAvg')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colMark')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colCost')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colMktValue')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colUPnl')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {livePositionRows.length === 0 ? (
                        <tr>
                          <td colSpan={9} className="px-2 py-8 text-center text-muted-foreground">{t('accounts.noLivePositions')}</td>
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
                        <th className="px-2 py-1.5 text-left">{t('accounts.colCreated')}</th>
                        <th className="px-2 py-1.5 text-left">{t('accounts.colMarket')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colSide')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colType')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colSize')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colPrice')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colFilled')}</th>
                        <th className="px-2 py-1.5 text-right">{t('accounts.colStatus')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {liveOrderRows.length === 0 ? (
                        <tr>
                          <td colSpan={8} className="px-2 py-8 text-center text-muted-foreground">{t('accounts.noLiveOrders')}</td>
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
