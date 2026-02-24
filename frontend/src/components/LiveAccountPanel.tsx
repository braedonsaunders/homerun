import { useMemo } from 'react'
import { useAtom } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  BarChart3,
  Briefcase,
  DollarSign,
  ExternalLink,
  RefreshCw,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Wallet,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildKalshiMarketUrl, buildPolymarketMarketUrl } from '../lib/marketUrls'
import { selectedAccountIdAtom } from '../store/atoms'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import {
  getTradingStatus,
  getTradingPositions,
  getTradingBalance,
  getKalshiStatus,
  getKalshiPositions,
  getKalshiBalance,
} from '../services/api'
import type { KalshiPosition, TradingPosition } from '../services/api'

type Platform = 'polymarket' | 'kalshi'

type NormalizedPosition = {
  key: string
  tokenId: string
  marketId: string
  marketQuestion: string
  outcome: string
  size: number
  averageCost: number
  currentPrice: number
  unrealizedPnl: number
  marketSlug: string | null
  eventSlug: string | null
}

function toNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function formatUsd(value: number): string {
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatSignedUsd(value: number): string {
  return `${value >= 0 ? '+' : ''}${formatUsd(value)}`
}

function formatPrice(value: number): string {
  return `$${value.toFixed(4)}`
}

function formatSize(value: number): string {
  return value.toFixed(2)
}

function formatPct(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
}

function shortText(value: string | null | undefined, head = 10, tail = 8): string {
  const text = String(value || '').trim()
  if (!text) return ''
  if (text.length <= head + tail + 3) return text
  return `${text.slice(0, head)}...${text.slice(-tail)}`
}

function normalizeOutcome(raw: string | null | undefined): string {
  const value = String(raw || '').trim().toUpperCase()
  if (!value) return 'N/A'
  if (value === 'BUY_YES' || value === 'SELL_YES') return 'YES'
  if (value === 'BUY_NO' || value === 'SELL_NO') return 'NO'
  if (value === 'LONG' || value === 'BUY' || value === 'UP') return 'YES'
  if (value === 'SHORT' || value === 'SELL' || value === 'DOWN') return 'NO'
  return value
}

function readRecordValue(record: Record<string, unknown>, key: string): string | null {
  const value = String(record[key] || '').trim()
  return value || null
}

function normalizePositions(positions: Array<TradingPosition | KalshiPosition>): NormalizedPosition[] {
  return positions
    .map((position) => {
      const record = position as unknown as Record<string, unknown>
      const marketId = String(position.market_id || '').trim()
      const tokenId = String(position.token_id || '').trim() || `${marketId}:${position.outcome}`
      const size = toNumber(position.size)
      const averageCost = toNumber(position.average_cost)
      const currentPrice = toNumber(position.current_price)
      const unrealizedFromPayload = toNumber(position.unrealized_pnl)
      const unrealizedPnl = unrealizedFromPayload !== 0
        ? unrealizedFromPayload
        : (currentPrice - averageCost) * size

      if (!marketId || size <= 0) return null

      return {
        key: `${marketId}:${tokenId}:${normalizeOutcome(position.outcome)}`,
        tokenId,
        marketId,
        marketQuestion: String(position.market_question || '').trim() || marketId,
        outcome: normalizeOutcome(position.outcome),
        size,
        averageCost,
        currentPrice,
        unrealizedPnl,
        marketSlug: readRecordValue(record, 'market_slug'),
        eventSlug: readRecordValue(record, 'event_slug'),
      }
    })
    .filter((position): position is NormalizedPosition => position !== null)
}

function marketLink(platform: Platform, position: NormalizedPosition): string | null {
  if (platform === 'kalshi') {
    return buildKalshiMarketUrl({
      marketTicker: position.marketId,
      eventTicker: position.eventSlug,
      eventSlug: position.marketSlug,
    })
  }
  return buildPolymarketMarketUrl({
    eventSlug: position.eventSlug,
    marketSlug: position.marketSlug,
    marketId: position.marketId,
  })
}

export default function LiveAccountPanel() {
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const platform: Platform = selectedAccountId === 'live:kalshi' ? 'kalshi' : 'polymarket'

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      {platform === 'polymarket' ? <PolymarketAccount /> : <KalshiAccount />}
    </div>
  )
}

function PolymarketAccount() {
  const tradingStatusQuery = useQuery({
    queryKey: ['trading-status'],
    queryFn: getTradingStatus,
    refetchInterval: 10000,
    retry: false,
  })
  const tradingStatus = tradingStatusQuery.data
  const connected = Boolean(tradingStatus?.authenticated || tradingStatus?.initialized)

  const positionsQuery = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    enabled: connected,
    refetchInterval: 15000,
    retry: false,
  })

  const balanceQuery = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
    enabled: connected,
    refetchInterval: 15000,
    retry: false,
  })

  const positions = useMemo(
    () => normalizePositions(positionsQuery.data ?? []),
    [positionsQuery.data]
  )
  const balance = toNumber(balanceQuery.data?.balance)
  const available = toNumber(balanceQuery.data?.available)
  const reserved = toNumber(balanceQuery.data?.reserved)
  const marketValue = positions.reduce((sum, position) => sum + position.size * position.currentPrice, 0)
  const costBasis = positions.reduce((sum, position) => sum + position.size * position.averageCost, 0)
  const unrealizedPnl = positions.reduce((sum, position) => sum + position.unrealizedPnl, 0)
  const markedPositions = positions.filter((position) => position.currentPrice > 0).length
  const markCoverage = positions.length > 0 ? (markedPositions / positions.length) * 100 : 0
  const isLoading = tradingStatusQuery.isLoading || (connected && (positionsQuery.isLoading || balanceQuery.isLoading))
  const isFetching = tradingStatusQuery.isFetching || positionsQuery.isFetching || balanceQuery.isFetching

  const refresh = () => {
    void tradingStatusQuery.refetch()
    if (connected) {
      void positionsQuery.refetch()
      void balanceQuery.refetch()
    }
  }

  return (
    <DeskShell
      platform="polymarket"
      title="Polymarket Live Desk"
      subtitle="Marked live positions, wallet balances, and execution footprint from your active wallet."
      connected={connected}
      identityLabel={tradingStatus?.wallet_address ? shortText(tradingStatus.wallet_address) : 'No wallet linked'}
      statusHint={connected ? 'Authenticated and ready for live reads' : 'Connect Polymarket credentials in Settings'}
      isLoading={isLoading}
      isFetching={isFetching}
      onRefresh={refresh}
      metrics={[
        {
          label: 'Balance',
          value: formatUsd(balance),
          hint: 'USDC wallet',
          icon: DollarSign,
        },
        {
          label: 'Available',
          value: formatUsd(available),
          hint: `Reserved ${formatUsd(reserved)}`,
          icon: Wallet,
        },
        {
          label: 'Open Positions',
          value: positions.length.toString(),
          hint: 'Active contracts',
          icon: Briefcase,
        },
        {
          label: 'Marked Value',
          value: formatUsd(marketValue),
          hint: `Cost basis ${formatUsd(costBasis)}`,
          icon: Activity,
        },
        {
          label: 'Unrealized P&L',
          value: formatSignedUsd(unrealizedPnl),
          hint: positions.length > 0 ? formatPct(costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0) : '--',
          icon: unrealizedPnl >= 0 ? TrendingUp : TrendingDown,
          tone: unrealizedPnl >= 0 ? 'positive' : 'negative',
        },
        {
          label: 'Mark Coverage',
          value: `${markCoverage.toFixed(0)}%`,
          hint: `${markedPositions}/${positions.length || 0} with live marks`,
          icon: BarChart3,
        },
      ]}
      positions={positions}
      limits={
        tradingStatus
          ? [
              { label: 'Max Trade Size', value: formatUsd(toNumber(tradingStatus.limits.max_trade_size_usd)) },
              { label: 'Max Daily Volume', value: formatUsd(toNumber(tradingStatus.limits.max_daily_volume)) },
              { label: 'Max Open Positions', value: `${toNumber(tradingStatus.limits.max_open_positions)}` },
              { label: 'Min Order Size', value: formatUsd(toNumber(tradingStatus.limits.min_order_size_usd)) },
              { label: 'Max Slippage', value: `${toNumber(tradingStatus.limits.max_slippage_percent)}%` },
            ]
          : null
      }
    />
  )
}

function KalshiAccount() {
  const statusQuery = useQuery({
    queryKey: ['kalshi-status'],
    queryFn: getKalshiStatus,
    refetchInterval: 10000,
    retry: false,
  })
  const status = statusQuery.data
  const connected = Boolean(status?.authenticated)

  const positionsQuery = useQuery({
    queryKey: ['kalshi-positions'],
    queryFn: getKalshiPositions,
    enabled: connected,
    refetchInterval: 15000,
    retry: false,
  })

  const balanceQuery = useQuery({
    queryKey: ['kalshi-balance'],
    queryFn: getKalshiBalance,
    enabled: connected,
    refetchInterval: 15000,
    retry: false,
  })

  const positions = useMemo(
    () => normalizePositions(positionsQuery.data ?? []),
    [positionsQuery.data]
  )
  const balance = toNumber(balanceQuery.data?.balance ?? status?.balance?.balance)
  const available = toNumber(balanceQuery.data?.available ?? status?.balance?.available)
  const reserved = toNumber(balanceQuery.data?.reserved ?? status?.balance?.reserved)
  const marketValue = positions.reduce((sum, position) => sum + position.size * position.currentPrice, 0)
  const costBasis = positions.reduce((sum, position) => sum + position.size * position.averageCost, 0)
  const unrealizedPnl = positions.reduce((sum, position) => sum + position.unrealizedPnl, 0)
  const markedPositions = positions.filter((position) => position.currentPrice > 0).length
  const markCoverage = positions.length > 0 ? (markedPositions / positions.length) * 100 : 0
  const isLoading = statusQuery.isLoading || (connected && (positionsQuery.isLoading || balanceQuery.isLoading))
  const isFetching = statusQuery.isFetching || positionsQuery.isFetching || balanceQuery.isFetching

  const refresh = () => {
    void statusQuery.refetch()
    if (connected) {
      void positionsQuery.refetch()
      void balanceQuery.refetch()
    }
  }

  return (
    <DeskShell
      platform="kalshi"
      title="Kalshi Live Desk"
      subtitle="Venue-level live holdings with marked exposure and unrealized P&L."
      connected={connected}
      identityLabel={status?.email || (status?.member_id ? `Member ${status.member_id}` : 'No account linked')}
      statusHint={connected ? 'Authenticated with Kalshi' : 'Configure Kalshi credentials in Settings'}
      isLoading={isLoading}
      isFetching={isFetching}
      onRefresh={refresh}
      metrics={[
        {
          label: 'Balance',
          value: formatUsd(balance),
          hint: 'USD wallet',
          icon: DollarSign,
        },
        {
          label: 'Available',
          value: formatUsd(available),
          hint: `Reserved ${formatUsd(reserved)}`,
          icon: Wallet,
        },
        {
          label: 'Open Positions',
          value: positions.length.toString(),
          hint: 'Active contracts',
          icon: Briefcase,
        },
        {
          label: 'Marked Value',
          value: formatUsd(marketValue),
          hint: `Cost basis ${formatUsd(costBasis)}`,
          icon: Activity,
        },
        {
          label: 'Unrealized P&L',
          value: formatSignedUsd(unrealizedPnl),
          hint: positions.length > 0 ? formatPct(costBasis > 0 ? (unrealizedPnl / costBasis) * 100 : 0) : '--',
          icon: unrealizedPnl >= 0 ? TrendingUp : TrendingDown,
          tone: unrealizedPnl >= 0 ? 'positive' : 'negative',
        },
        {
          label: 'Mark Coverage',
          value: `${markCoverage.toFixed(0)}%`,
          hint: `${markedPositions}/${positions.length || 0} with live marks`,
          icon: BarChart3,
        },
      ]}
      positions={positions}
      limits={null}
    />
  )
}

function DeskShell({
  platform,
  title,
  subtitle,
  connected,
  identityLabel,
  statusHint,
  isLoading,
  isFetching,
  onRefresh,
  metrics,
  positions,
  limits,
}: {
  platform: Platform
  title: string
  subtitle: string
  connected: boolean
  identityLabel: string
  statusHint: string
  isLoading: boolean
  isFetching: boolean
  onRefresh: () => void
  metrics: Array<{
    label: string
    value: string
    hint: string
    icon: React.ElementType
    tone?: 'positive' | 'negative'
  }>
  positions: NormalizedPosition[]
  limits: Array<{ label: string; value: string }> | null
}) {
  const accent = platform === 'polymarket'
    ? {
        badge: 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200',
        button: 'border-cyan-500/35 text-cyan-200 hover:bg-cyan-500/10',
      }
    : {
        badge: 'border-indigo-500/30 bg-indigo-500/10 text-indigo-200',
        button: 'border-indigo-500/35 text-indigo-200 hover:bg-indigo-500/10',
      }

  return (
    <>
      <Card className="shrink-0 border-border/80 bg-card/80">
        <CardContent className="p-4">
          <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Live Trading Desk</p>
              <h2 className="mt-1 text-lg font-semibold text-foreground">{title}</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className={cn(accent.badge)}>
                {connected ? 'Connected' : 'Disconnected'}
              </Badge>
              <Badge variant="outline" className="border-border bg-background/60 text-muted-foreground">
                {identityLabel}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={onRefresh}
                className={cn('h-8 gap-1.5', accent.button)}
              >
                <RefreshCw className={cn('h-3.5 w-3.5', isFetching && 'animate-spin')} />
                Refresh
              </Button>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">{statusHint}</p>
        </CardContent>
      </Card>

      {isLoading ? (
        <Card className="border-border/80 bg-card/60">
          <CardContent className="flex justify-center py-12">
            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
          </CardContent>
        </Card>
      ) : !connected ? (
        <Card className="border-border/80 bg-card/60">
          <CardContent className="py-12 text-center">
            <ShieldAlert className="mx-auto mb-3 h-12 w-12 text-muted-foreground/40" />
            <p className="text-sm text-foreground">Live account not connected</p>
            <p className="mt-1 text-xs text-muted-foreground">Open Settings to configure credentials and enable live portfolio reads.</p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
            {metrics.map((metric) => (
              <MetricCard
                key={metric.label}
                label={metric.label}
                value={metric.value}
                hint={metric.hint}
                icon={metric.icon}
                tone={metric.tone}
              />
            ))}
          </div>

          <PositionsTable platform={platform} positions={positions} />

          {limits && limits.length > 0 && (
            <Card className="border-border/80 bg-card/70">
              <CardContent className="p-4">
                <div className="mb-3 flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 text-muted-foreground" />
                  <p className="text-sm font-medium">Trading Safety Limits</p>
                </div>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
                  {limits.map((limit) => (
                    <div key={limit.label} className="rounded-lg border border-border/70 bg-background/40 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{limit.label}</p>
                      <p className="mt-1 font-mono text-sm text-foreground">{limit.value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </>
  )
}

function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
  tone,
}: {
  label: string
  value: string
  hint: string
  icon: React.ElementType
  tone?: 'positive' | 'negative'
}) {
  return (
    <Card className="border-border/80 bg-card/70 shadow-none">
      <CardContent className="p-3">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
          <Icon className="h-3.5 w-3.5" />
          <span>{label}</span>
        </div>
        <p
          className={cn(
            'mt-1 font-mono text-base font-semibold',
            tone === 'positive' && 'text-emerald-300',
            tone === 'negative' && 'text-red-300'
          )}
        >
          {value}
        </p>
        <p className="text-[11px] text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  )
}

function PositionsTable({
  platform,
  positions,
}: {
  platform: Platform
  positions: NormalizedPosition[]
}) {
  if (positions.length === 0) {
    return (
      <Card className="border-border/80 bg-card/70">
        <CardContent className="py-12 text-center">
          <Briefcase className="mx-auto mb-3 h-10 w-10 text-muted-foreground/35" />
          <p className="text-sm text-foreground">No open positions</p>
          <p className="mt-1 text-xs text-muted-foreground">
            New live positions will appear here as soon as orders are filled.
          </p>
        </CardContent>
      </Card>
    )
  }

  const totalCostBasis = positions.reduce((sum, position) => sum + position.size * position.averageCost, 0)
  const totalMarketValue = positions.reduce((sum, position) => sum + position.size * position.currentPrice, 0)
  const totalUnrealizedPnl = positions.reduce((sum, position) => sum + position.unrealizedPnl, 0)

  return (
    <Card className="border-border/80 bg-card/80">
      <div className="grid shrink-0 grid-cols-1 gap-3 border-b border-border/70 px-4 py-3 md:grid-cols-3">
        <div className="rounded-lg border border-border/70 bg-background/40 p-3">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Cost Basis</p>
          <p className="mt-1 font-mono text-sm text-foreground">{formatUsd(totalCostBasis)}</p>
        </div>
        <div className="rounded-lg border border-border/70 bg-background/40 p-3">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Market Value</p>
          <p className="mt-1 font-mono text-sm text-foreground">{formatUsd(totalMarketValue)}</p>
        </div>
        <div className="rounded-lg border border-border/70 bg-background/40 p-3">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Unrealized P&L</p>
          <p className={cn('mt-1 font-mono text-sm', totalUnrealizedPnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
            {formatSignedUsd(totalUnrealizedPnl)}
          </p>
        </div>
      </div>

      <div className="min-h-0 overflow-auto">
        <Table className="text-xs">
          <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
            <TableRow className="border-b border-border/80 bg-muted/40">
              <TableHead className="h-9 min-w-[260px] px-3">Market</TableHead>
              <TableHead className="h-9 px-3">Side</TableHead>
              <TableHead className="h-9 px-3 text-right">Size</TableHead>
              <TableHead className="h-9 px-3 text-right">Avg</TableHead>
              <TableHead className="h-9 px-3 text-right">Current</TableHead>
              <TableHead className="h-9 px-3 text-right">Cost Basis</TableHead>
              <TableHead className="h-9 px-3 text-right">Market Value</TableHead>
              <TableHead className="h-9 px-3 text-right">Unrealized</TableHead>
              <TableHead className="h-9 px-3 text-right">Link</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.map((position) => {
              const costBasis = position.size * position.averageCost
              const marketValue = position.size * position.currentPrice
              const pnlPercent = costBasis > 0 ? (position.unrealizedPnl / costBasis) * 100 : 0
              const link = marketLink(platform, position)
              return (
                <TableRow key={position.key} className="border-border/70">
                  <TableCell className="px-3 py-2.5">
                    <div className="space-y-0.5">
                      <p className="max-w-[420px] truncate text-foreground" title={position.marketQuestion}>
                        {position.marketQuestion}
                      </p>
                      <p className="font-mono text-[10px] text-muted-foreground">{position.marketId}</p>
                    </div>
                  </TableCell>
                  <TableCell className="px-3 py-2.5">
                    <Badge
                      variant="outline"
                      className="h-5 border-border/80 bg-muted/60 px-1.5 text-[10px] text-muted-foreground"
                    >
                      {position.outcome}
                    </Badge>
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatSize(position.size)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatPrice(position.averageCost)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatPrice(position.currentPrice)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatUsd(costBasis)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatUsd(marketValue)}</TableCell>
                  <TableCell className={cn('px-3 py-2.5 text-right font-mono', position.unrealizedPnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                    {formatSignedUsd(position.unrealizedPnl)}
                    <span className="ml-1 text-[10px] text-muted-foreground">({formatPct(pnlPercent)})</span>
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-right">
                    {link ? (
                      <a
                        href={link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={cn(
                          'inline-flex h-7 w-7 items-center justify-center rounded border border-border bg-background/70 text-muted-foreground transition-colors hover:text-foreground',
                          platform === 'kalshi' ? 'hover:border-indigo-500/40 hover:text-indigo-200' : 'hover:border-cyan-500/40 hover:text-cyan-200'
                        )}
                        title={`Open on ${platform === 'kalshi' ? 'Kalshi' : 'Polymarket'}`}
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    ) : (
                      <span className="text-muted-foreground">--</span>
                    )}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </Card>
  )
}
