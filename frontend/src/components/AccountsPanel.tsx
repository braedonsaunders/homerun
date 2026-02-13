import { useMemo, useState } from 'react'
import { useAtom } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  BarChart3,
  Briefcase,
  DollarSign,
  LayoutDashboard,
  Shield,
  TrendingDown,
  TrendingUp,
  Wallet,
  Zap,
  ArrowUpRight,
  Settings,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { accountModeAtom, selectedAccountIdAtom } from '../store/atoms'
import { Card, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import SimulationPanel from './SimulationPanel'
import LiveAccountPanel from './LiveAccountPanel'
import {
  getKalshiBalance,
  getKalshiPositions,
  getKalshiStatus,
  getSimulationAccounts,
  getTradingBalance,
  getTradingPositions,
  getTradingStatus,
} from '../services/api'

type AccountsWorkspaceTab = 'overview' | 'sandbox' | 'live'
type LiveVenue = 'polymarket' | 'kalshi'

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

function formatUsd(value: number): string {
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatSignedUsd(value: number): string {
  return `${value >= 0 ? '+' : ''}${formatUsd(value)}`
}

function formatSignedPct(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

export default function AccountsPanel({ onOpenSettings }: AccountsPanelProps) {
  const [accountMode, setAccountMode] = useAtom(accountModeAtom)
  const [selectedAccountId, setSelectedAccountId] = useAtom(selectedAccountIdAtom)
  const [workspaceTab, setWorkspaceTab] = useState<AccountsWorkspaceTab>(accountMode === 'live' ? 'live' : 'overview')

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

  const { data: tradingPositions = [] } = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    enabled: !!tradingStatus?.initialized,
    refetchInterval: 15000,
    retry: false,
  })

  const { data: tradingBalance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
    enabled: !!tradingStatus?.initialized,
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

  const sandboxMetrics = useMemo(() => {
    const totalInitial = sandboxAccounts.reduce((sum, account) => sum + (account.initial_capital || 0), 0)
    const totalCapital = sandboxAccounts.reduce((sum, account) => sum + (account.current_capital || 0), 0)
    const realizedPnl = sandboxAccounts.reduce((sum, account) => sum + (account.total_pnl || 0), 0)
    const unrealizedPnl = sandboxAccounts.reduce((sum, account) => sum + (account.unrealized_pnl || 0), 0)
    const totalPnl = realizedPnl + unrealizedPnl
    const totalTrades = sandboxAccounts.reduce((sum, account) => sum + (account.total_trades || 0), 0)
    const totalOpenPositions = sandboxAccounts.reduce((sum, account) => sum + (account.open_positions || 0), 0)
    const roi = totalInitial > 0 ? (totalPnl / totalInitial) * 100 : 0

    return {
      count: sandboxAccounts.length,
      totalInitial,
      totalCapital,
      totalPnl,
      roi,
      totalTrades,
      totalOpenPositions,
    }
  }, [sandboxAccounts])

  const polymarketSnapshot = useMemo<LiveVenueSnapshot>(() => {
    const exposure = tradingPositions.reduce((sum, pos) => sum + pos.size * pos.current_price, 0)
    const unrealizedPnl = tradingPositions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0)
    return {
      id: 'polymarket',
      label: 'Polymarket',
      connected: Boolean(tradingStatus?.initialized),
      accountLabel: tradingStatus?.wallet_address
        ? `${tradingStatus.wallet_address.slice(0, 8)}...${tradingStatus.wallet_address.slice(-6)}`
        : 'No wallet',
      balance: tradingBalance?.balance ?? 0,
      available: tradingBalance?.available ?? 0,
      exposure,
      openPositions: tradingPositions.length,
      unrealizedPnl,
    }
  }, [tradingStatus?.initialized, tradingStatus?.wallet_address, tradingBalance?.balance, tradingBalance?.available, tradingPositions])

  const kalshiSnapshot = useMemo<LiveVenueSnapshot>(() => {
    const exposure = kalshiPositions.reduce((sum, pos) => sum + pos.size * pos.current_price, 0)
    const unrealizedPnl = kalshiPositions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0)
    return {
      id: 'kalshi',
      label: 'Kalshi',
      connected: Boolean(kalshiStatus?.authenticated),
      accountLabel: kalshiStatus?.email || (kalshiStatus?.member_id ? `Member ${kalshiStatus.member_id}` : 'No account'),
      balance: kalshiBalance?.balance ?? kalshiStatus?.balance?.balance ?? 0,
      available: kalshiBalance?.available ?? kalshiStatus?.balance?.available ?? 0,
      exposure,
      openPositions: kalshiPositions.length,
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

  const activeSandboxAccount = sandboxAccounts.find((account) => account.id === selectedAccountId)

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
    ] as const
  }, [sandboxMetrics.count, sandboxMetrics.totalOpenPositions, liveMetrics.connectedVenues, liveMetrics.totalOpenPositions, liveMetrics.totalUnrealizedPnl])

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
  }

  const openLiveDesk = (venue: LiveVenue = 'polymarket') => {
    setSelectedAccountId(`live:${venue}`)
    setAccountMode('live')
    setWorkspaceTab('live')
  }

  const isOverview = workspaceTab === 'overview'

  return (
    <div
      className={cn(
        'h-full flex min-h-0 flex-col',
        isOverview ? 'gap-3' : 'gap-4 overflow-y-auto pr-1'
      )}
    >
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
                onClick={() => setWorkspaceTab(accountMode === 'live' ? 'live' : 'sandbox')}
                className="h-8 gap-1.5 bg-background/40 text-xs"
              >
                <ArrowUpRight className="h-3.5 w-3.5" />
                Open Active Desk
              </Button>
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
              value={formatUsd(sandboxMetrics.totalCapital)}
              hint={`${sandboxMetrics.count} accounts`}
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
              hint="Open"
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

      <div className="shrink-0 rounded-xl border border-border bg-card/40 px-3 py-2.5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground/70">Accounts Workspace</p>
            <p className="text-xs text-muted-foreground/90">
              {WORKSPACE_TAB_CONFIG.find((tab) => tab.id === workspaceTab)?.description}
            </p>
          </div>
          <div className="overflow-x-auto">
            <div className="flex min-w-max items-center gap-1 rounded-lg border border-border/80 bg-background/70 p-1">
              {WORKSPACE_TAB_CONFIG.map((tab) => (
                <Button
                  key={tab.id}
                  variant="ghost"
                  size="sm"
                  onClick={() => setWorkspaceTab(tab.id)}
                  className={cn(
                    'h-7 gap-1.5 px-2.5 text-xs',
                    workspaceTab === tab.id
                      ? 'bg-primary/15 text-primary hover:bg-primary/20'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  <tab.icon className="h-3.5 w-3.5" />
                  {tab.label}
                </Button>
              ))}
            </div>
          </div>
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
                              <td className="px-3 py-2 text-right font-mono">{account.open_positions || 0}</td>
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
              <CardContent className="space-y-2.5 p-3">
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
            <CardContent className="space-y-3 p-3">
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
        <div className="min-h-0 flex-1">
          <SimulationPanel />
        </div>
      )}
      {workspaceTab === 'live' && (
        <div className="min-h-0 flex-1">
          <LiveAccountPanel />
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
