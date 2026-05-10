import { useEffect, useMemo, useState, type ComponentType } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Briefcase,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  History,
  Percent,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  User,
  Wallet,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'
import { Input } from './ui/input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import {
  analyzeWallet,
  analyzeWalletPnL,
  getWalletPositionsAnalysis,
  getWalletProfile,
  getWalletSummary,
  getWalletTradesAnalysis,
  getWalletWinRate,
  type WalletAnalysis,
  type WalletPosition,
  type WalletTrade,
} from '../services/api'

interface WalletAnalysisPanelProps {
  initialWallet?: string | null
  initialUsername?: string | null
  onWalletAnalyzed?: () => void
}

type AnalysisTab = 'overview' | 'trades' | 'positions' | 'risk'
type TimePeriod = 'DAY' | 'WEEK' | 'MONTH' | 'ALL'

const TIME_PERIOD_OPTIONS: Array<{ value: TimePeriod; labelKey: string }> = [
  { value: 'DAY', labelKey: 'walletAnalysisPanel.period24h' },
  { value: 'WEEK', labelKey: 'walletAnalysisPanel.period7d' },
  { value: 'MONTH', labelKey: 'walletAnalysisPanel.period30d' },
  { value: 'ALL', labelKey: 'walletAnalysisPanel.periodAll' },
]

const TAB_OPTIONS: Array<{ id: AnalysisTab; labelKey: string; icon: ComponentType<{ className?: string }> }> = [
  { id: 'overview', labelKey: 'walletAnalysisPanel.tabOverview', icon: BarChart3 },
  { id: 'trades', labelKey: 'walletAnalysisPanel.tabTrades', icon: History },
  { id: 'positions', labelKey: 'walletAnalysisPanel.tabPositions', icon: Briefcase },
  { id: 'risk', labelKey: 'walletAnalysisPanel.tabRisk', icon: ShieldAlert },
]

const PAGE_SIZE_OPTIONS = [10, 20, 50]

function formatCurrency(value: number, decimals = 2): string {
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`
}

function formatSignedCurrency(value: number, decimals = 2): string {
  const prefix = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${prefix}${formatCurrency(Math.abs(value), decimals)}`
}

function formatSignedPercent(value: number, decimals = 1): string {
  const prefix = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${prefix}${Math.abs(value).toFixed(decimals)}%`
}

function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return '0'
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

function shortAddress(address: string): string {
  if (address.length <= 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function formatTimestamp(timestamp: string): string {
  if (!timestamp) return '--'
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function readErrorMessage(error: unknown, fallback = 'Request failed'): string {
  if (typeof error === 'object' && error && 'message' in error) {
    return String((error as { message?: unknown }).message ?? fallback)
  }
  return fallback
}

function riskModel(score: number): {
  labelKey: string
  badgeClass: string
  textClass: string
  borderClass: string
} {
  if (score >= 0.7) {
    return {
      labelKey: 'walletAnalysisPanel.riskHigh',
      badgeClass: 'bg-red-500/15 text-red-300 border-red-500/30',
      textClass: 'text-red-300',
      borderClass: 'border-red-500/25',
    }
  }
  if (score >= 0.3) {
    return {
      labelKey: 'walletAnalysisPanel.riskModerate',
      badgeClass: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
      textClass: 'text-amber-300',
      borderClass: 'border-amber-500/25',
    }
  }
  return {
    labelKey: 'walletAnalysisPanel.riskLow',
    badgeClass: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    textClass: 'text-emerald-300',
    borderClass: 'border-emerald-500/25',
  }
}

function Sparkline({ values, positive }: { values: number[]; positive: boolean }) {
  const { t } = useTranslation()
  if (values.length < 2) {
    return (
      <div className="flex h-[84px] items-center justify-center text-xs text-muted-foreground/70">
        {t('walletAnalysisPanel.notEnoughTradeHistory')}
      </div>
    )
  }

  const width = 420
  const height = 84
  const padding = 4
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const points = values
    .map((value, index) => {
      const x = padding + (index / (values.length - 1)) * (width - padding * 2)
      const y = height - padding - ((value - min) / range) * (height - padding * 2)
      return `${x},${y}`
    })
    .join(' ')

  const strokeColor = positive ? '#34d399' : '#f87171'

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke={strokeColor}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}

function StatTile({
  label,
  value,
  delta,
  positive,
  icon: Icon,
}: {
  label: string
  value: string
  delta?: string
  positive?: boolean
  icon: ComponentType<{ className?: string }>
}) {
  return (
    <div className="rounded-xl border border-border/70 bg-background/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <p className="mt-1 text-lg font-semibold text-foreground">{value}</p>
      {delta && (
        <p className={cn('text-xs', positive ? 'text-emerald-300' : 'text-red-300')}>
          {delta}
        </p>
      )}
    </div>
  )
}

function SectionLoading() {
  return (
    <div className="flex h-full min-h-[180px] items-center justify-center">
      <RefreshCw className="h-7 w-7 animate-spin text-muted-foreground" />
    </div>
  )
}

interface PaginationControlsProps {
  page: number
  pageSize: number
  total: number
  itemLabel: string
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}

function PaginationControls({
  page,
  pageSize,
  total,
  itemLabel,
  onPageChange,
  onPageSizeChange,
}: PaginationControlsProps) {
  const { t } = useTranslation()
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const start = total === 0 ? 0 : (page - 1) * pageSize + 1
  const end = total === 0 ? 0 : Math.min(total, page * pageSize)

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-muted-foreground">
        {t('walletAnalysisPanel.paginationRange', { start, end, total, itemLabel })}
      </span>
      <select
        value={pageSize}
        onChange={(event) => onPageSizeChange(Number(event.target.value))}
        className="h-8 rounded-md border border-border bg-background px-2 text-xs"
      >
        {PAGE_SIZE_OPTIONS.map((size) => (
          <option key={size} value={size}>
            {t('walletAnalysisPanel.pageSizeOption', { size })}
          </option>
        ))}
      </select>
      <Button
        variant="outline"
        size="sm"
        className="h-8 px-2"
        onClick={() => onPageChange(Math.max(1, page - 1))}
        disabled={page <= 1}
      >
        <ChevronLeft className="h-3.5 w-3.5" />
      </Button>
      <span className="w-[74px] text-center text-muted-foreground">
        {t('walletAnalysisPanel.pageOf', { page, totalPages })}
      </span>
      <Button
        variant="outline"
        size="sm"
        className="h-8 px-2"
        onClick={() => onPageChange(Math.min(totalPages, page + 1))}
        disabled={page >= totalPages}
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}

function EmptyData({ icon: Icon, title, subtitle }: { icon: ComponentType<{ className?: string }>; title: string; subtitle: string }) {
  return (
    <div className="flex h-full min-h-[220px] flex-col items-center justify-center px-6 text-center">
      <Icon className="mb-3 h-10 w-10 text-muted-foreground/35" />
      <p className="text-sm text-foreground">{title}</p>
      <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
    </div>
  )
}

function OverviewHeroPanel({
  isLoading,
  activeWallet,
  username,
  timePeriod,
  anomalyScore,
  riskLabel,
  riskBadgeClass,
  totalPnl,
  roiPercent,
  isProfitable,
  winRate,
  wins,
  losses,
  volume,
  totalTrades,
  sparklineValues,
  realizedPnl,
  unrealizedPnl,
  isHeaderLoading,
  positionsCount,
  anomaliesCount,
}: {
  isLoading: boolean
  activeWallet: string
  username: string | null
  timePeriod: TimePeriod
  anomalyScore: number
  riskLabel: string
  riskBadgeClass: string
  totalPnl: number
  roiPercent: number
  isProfitable: boolean
  winRate: number
  wins: number
  losses: number
  volume: number
  totalTrades: number
  sparklineValues: number[]
  realizedPnl: number
  unrealizedPnl: number
  isHeaderLoading: boolean
  positionsCount: number
  anomaliesCount: number
}) {
  const { t } = useTranslation()
  if (isLoading) {
    return <SectionLoading />
  }

  return (
    <div className="h-full p-4">
      <section className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 border-border/80 bg-card/75 lg:col-span-8">
          <CardContent className="p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex items-start gap-3">
                <div className="rounded-xl border border-cyan-500/25 bg-cyan-500/10 p-3">
                  {username ? <User className="h-5 w-5 text-cyan-700 dark:text-cyan-200" /> : <Wallet className="h-5 w-5 text-cyan-700 dark:text-cyan-200" />}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-semibold text-foreground">
                      {username || shortAddress(activeWallet)}
                    </h3>
                    <a
                      href={`https://polymarket.com/profile/${activeWallet}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex h-6 w-6 items-center justify-center rounded border border-border bg-background/70 text-muted-foreground transition-colors hover:text-foreground"
                      title={t('walletAnalysisPanel.openProfile')}
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  </div>
                  <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">{activeWallet}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Badge variant="outline" className={riskBadgeClass}>
                  {t('walletAnalysisPanel.anomalyPercent', { percent: (anomalyScore * 100).toFixed(0) })}
                </Badge>
                <Badge variant="outline" className="border-border bg-background/60 text-muted-foreground">
                  {(() => {
                    const opt = TIME_PERIOD_OPTIONS.find((option) => option.value === timePeriod)
                    return opt ? t(opt.labelKey) : ''
                  })()}
                </Badge>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
              <StatTile
                label={t('walletAnalysisPanel.totalPnl')}
                value={formatSignedCurrency(totalPnl)}
                delta={formatSignedPercent(roiPercent)}
                positive={isProfitable}
                icon={isProfitable ? TrendingUp : TrendingDown}
              />
              <StatTile
                label={t('walletAnalysisPanel.winRate')}
                value={`${winRate.toFixed(1)}%`}
                delta={t('walletAnalysisPanel.winsLosses', { wins, losses })}
                positive={winRate >= 50}
                icon={Percent}
              />
              <StatTile
                label={t('walletAnalysisPanel.volume')}
                value={formatCurrency(volume, 0)}
                delta={t('walletAnalysisPanel.tradesDelta', { count: formatCompact(totalTrades) })}
                positive
                icon={Activity}
              />
              <StatTile
                label={t('walletAnalysisPanel.riskScore')}
                value={`${(anomalyScore * 100).toFixed(0)}%`}
                delta={riskLabel}
                positive={anomalyScore < 0.3}
                icon={ShieldAlert}
              />
            </div>

            <div className="mt-4 rounded-xl border border-border/70 bg-background/40 p-3">
              <div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground">
                <span>{t('walletAnalysisPanel.pnlTrend')}</span>
                <span>{t('walletAnalysisPanel.sampledTrades', { count: totalTrades })}</span>
              </div>
              <Sparkline values={sparklineValues} positive={isProfitable} />
            </div>
          </CardContent>
        </Card>

        <Card className="col-span-12 border-border/80 bg-card/75 lg:col-span-4">
          <CardContent className="space-y-3 p-5">
            <div className="rounded-xl border border-border/70 bg-background/40 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.realizedPnl')}</p>
              <p className={cn('mt-1 text-lg font-semibold', realizedPnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                {formatSignedCurrency(realizedPnl)}
              </p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background/40 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.unrealizedPnl')}</p>
              <p className={cn('mt-1 text-lg font-semibold', unrealizedPnl >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                {formatSignedCurrency(unrealizedPnl)}
              </p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background/40 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.dataHealth')}</p>
              <p className="mt-1 text-sm text-foreground">
                {isHeaderLoading ? t('walletAnalysisPanel.loadingFreshMetrics') : t('walletAnalysisPanel.metricsSynchronized')}
              </p>
              <p className="mt-1 text-[11px] text-muted-foreground">
                {t('walletAnalysisPanel.dataHealthDetail', { trades: totalTrades, positions: positionsCount, anomalies: anomaliesCount })}
              </p>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

function TradesPanel({
  isLoading,
  trades,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: {
  isLoading: boolean
  trades: WalletTrade[]
  page: number
  pageSize: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}) {
  const { t } = useTranslation()
  if (isLoading) return <SectionLoading />

  if (trades.length === 0) {
    return <EmptyData icon={History} title={t('walletAnalysisPanel.noTradesFound')} subtitle={t('walletAnalysisPanel.noTradesSubtitle')} />
  }

  const totalPages = Math.max(1, Math.ceil(trades.length / pageSize))
  const safePage = Math.min(page, totalPages)
  const startIndex = (safePage - 1) * pageSize
  const pageRows = trades.slice(startIndex, startIndex + pageSize)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between border-b border-border/70 px-4 py-3">
        <p className="text-xs text-muted-foreground">{t('walletAnalysisPanel.tradesSubtitle')}</p>
        <PaginationControls
          page={safePage}
          pageSize={pageSize}
          total={trades.length}
          itemLabel={t('walletAnalysisPanel.itemTrades')}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <Table className="text-xs">
          <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
            <TableRow className="border-b border-border/80 bg-muted/40">
              <TableHead className="h-9 px-3">{t('walletAnalysisPanel.colTime')}</TableHead>
              <TableHead className="h-9 px-3 min-w-[240px]">{t('walletAnalysisPanel.colMarket')}</TableHead>
              <TableHead className="h-9 px-3">{t('walletAnalysisPanel.colSide')}</TableHead>
              <TableHead className="h-9 px-3">{t('walletAnalysisPanel.colOutcome')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colSize')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colPrice')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colNotional')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colLinks')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pageRows.map((trade) => {
              const marketUrl = buildPolymarketMarketUrl({
                eventSlug: trade.event_slug,
                marketSlug: trade.market_slug,
                marketId: trade.market,
              })
              const isBuy = trade.side === 'BUY'

              return (
                <TableRow key={trade.id} className="border-border/70">
                  <TableCell className="px-3 py-2.5 font-mono text-[11px] text-muted-foreground">
                    {formatTimestamp(trade.timestamp)}
                  </TableCell>
                  <TableCell className="px-3 py-2.5">
                    <div className="space-y-0.5">
                      <p className="max-w-[360px] truncate text-foreground" title={trade.market_title || trade.market}>
                        {trade.market_title || trade.market}
                      </p>
                      <p className="font-mono text-[10px] text-muted-foreground">{shortAddress(trade.market)}</p>
                    </div>
                  </TableCell>
                  <TableCell className="px-3 py-2.5">
                    <Badge
                      variant="outline"
                      className={cn(
                        'text-[10px]',
                        isBuy ? 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300' : 'border-red-500/30 bg-red-500/15 text-red-300',
                      )}
                    >
                      {trade.side}
                    </Badge>
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-foreground/90">{trade.outcome || '--'}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{trade.size.toFixed(2)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">${trade.price.toFixed(4)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatCurrency(trade.cost)}</TableCell>
                  <TableCell className="px-3 py-2.5">
                    <div className="flex items-center justify-end gap-2">
                      {marketUrl && (
                        <a
                          href={marketUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex h-7 w-7 items-center justify-center rounded border border-border bg-background/70 text-muted-foreground transition-colors hover:text-foreground"
                          title={t('walletAnalysisPanel.openMarket')}
                        >
                          <ExternalLink className="h-3.5 w-3.5" />
                        </a>
                      )}
                      {trade.transaction_hash && (
                        <a
                          href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex h-7 w-7 items-center justify-center rounded border border-border bg-background/70 text-muted-foreground transition-colors hover:text-foreground"
                          title={t('walletAnalysisPanel.openTransaction')}
                        >
                          <ArrowUpRight className="h-3.5 w-3.5" />
                        </a>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

function PositionsPanel({
  isLoading,
  data,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: {
  isLoading: boolean
  data?: {
    wallet: string
    total_positions: number
    total_value: number
    total_unrealized_pnl: number
    positions: WalletPosition[]
  }
  page: number
  pageSize: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}) {
  const { t } = useTranslation()
  if (isLoading) return <SectionLoading />

  const positions = data?.positions ?? []

  if (positions.length === 0) {
    return <EmptyData icon={Briefcase} title={t('walletAnalysisPanel.noOpenPositions')} subtitle={t('walletAnalysisPanel.noOpenPositionsSubtitle')} />
  }

  const totalPages = Math.max(1, Math.ceil(positions.length / pageSize))
  const safePage = Math.min(page, totalPages)
  const startIndex = (safePage - 1) * pageSize
  const pageRows = positions.slice(startIndex, startIndex + pageSize)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid shrink-0 grid-cols-1 gap-3 border-b border-border/70 px-4 py-3 md:grid-cols-2">
        <div className="rounded-lg border border-border/70 bg-background/40 p-3">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.positionValue')}</p>
          <p className="mt-1 text-lg font-semibold text-foreground">{formatCurrency(data?.total_value ?? 0)}</p>
        </div>
        <div className="rounded-lg border border-border/70 bg-background/40 p-3">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.unrealizedPnl')}</p>
          <p className={cn('mt-1 text-lg font-semibold', (data?.total_unrealized_pnl ?? 0) >= 0 ? 'text-emerald-300' : 'text-red-300')}>
            {formatSignedCurrency(data?.total_unrealized_pnl ?? 0)}
          </p>
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-end border-b border-border/70 px-4 py-3">
        <PaginationControls
          page={safePage}
          pageSize={pageSize}
          total={positions.length}
          itemLabel={t('walletAnalysisPanel.itemPositions')}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <Table className="text-xs">
          <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
            <TableRow className="border-b border-border/80 bg-muted/40">
              <TableHead className="h-9 px-3 min-w-[220px]">{t('walletAnalysisPanel.colMarket')}</TableHead>
              <TableHead className="h-9 px-3">{t('walletAnalysisPanel.colOutcome')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colSize')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colAvg')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colCurrent')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colValue')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colUnrealized')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colRoi')}</TableHead>
              <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colLink')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pageRows.map((position) => {
              const marketUrl = buildPolymarketMarketUrl({
                eventSlug: position.event_slug,
                marketSlug: position.market_slug,
                marketId: position.market,
              })
              const positive = position.unrealized_pnl >= 0

              return (
                <TableRow key={`${position.market}-${position.outcome}`} className="border-border/70">
                  <TableCell className="px-3 py-2.5">
                    <div className="space-y-0.5">
                      <p className="max-w-[360px] truncate text-foreground" title={position.title || position.market}>
                        {position.title || position.market}
                      </p>
                      <p className="font-mono text-[10px] text-muted-foreground">{shortAddress(position.market)}</p>
                    </div>
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-foreground/90">{position.outcome || '--'}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{position.size.toFixed(2)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">${position.avg_price.toFixed(4)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">${position.current_price.toFixed(4)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{formatCurrency(position.current_value)}</TableCell>
                  <TableCell className={cn('px-3 py-2.5 text-right font-mono', positive ? 'text-emerald-300' : 'text-red-300')}>
                    {formatSignedCurrency(position.unrealized_pnl)}
                  </TableCell>
                  <TableCell className={cn('px-3 py-2.5 text-right font-mono', position.roi_percent >= 0 ? 'text-emerald-300' : 'text-red-300')}>
                    {formatSignedPercent(position.roi_percent)}
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-right">
                    {marketUrl ? (
                      <a
                        href={marketUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex h-7 w-7 items-center justify-center rounded border border-border bg-background/70 text-muted-foreground transition-colors hover:text-foreground"
                        title={t('walletAnalysisPanel.openMarket')}
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
    </div>
  )
}

function RiskPanel({
  isLoading,
  data,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: {
  isLoading: boolean
  data?: WalletAnalysis
  page: number
  pageSize: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}) {
  const { t } = useTranslation()
  if (isLoading) return <SectionLoading />

  if (!data) {
    return <EmptyData icon={ShieldAlert} title={t('walletAnalysisPanel.noRiskAnalysis')} subtitle={t('walletAnalysisPanel.noRiskAnalysisSubtitle')} />
  }

  const risk = riskModel(data.anomaly_score)
  const anomalies = data.anomalies ?? []
  const totalPages = Math.max(1, Math.ceil(anomalies.length / pageSize))
  const safePage = Math.min(page, totalPages)
  const startIndex = (safePage - 1) * pageSize
  const pageRows = anomalies.slice(startIndex, startIndex + pageSize)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid shrink-0 grid-cols-1 gap-3 border-b border-border/70 px-4 py-3 lg:grid-cols-3">
        <div className={cn('rounded-lg border bg-background/40 p-3', risk.borderClass)}>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.anomalyScore')}</p>
          <p className={cn('mt-1 text-2xl font-semibold', risk.textClass)}>{(data.anomaly_score * 100).toFixed(0)}%</p>
          <p className="mt-1 text-xs text-muted-foreground">{t(risk.labelKey)}</p>
        </div>

        <div className="rounded-lg border border-border/70 bg-background/40 p-3 lg:col-span-2">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('walletAnalysisPanel.recommendation')}</p>
          <p className="mt-1 text-sm text-foreground">{data.recommendation}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge variant="outline" className={risk.badgeClass}>
              {t(risk.labelKey)}
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                data.is_profitable_pattern
                  ? 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300'
                  : 'border-border bg-background/60 text-muted-foreground',
              )}
            >
              {data.is_profitable_pattern ? t('walletAnalysisPanel.profitablePattern') : t('walletAnalysisPanel.patternUnclear')}
            </Badge>
          </div>
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-between border-b border-border/70 px-4 py-3">
        <div className="flex flex-wrap gap-1.5">
          {data.strategies_detected.slice(0, 5).map((strategy) => (
            <Badge key={strategy} variant="outline" className="border-cyan-500/30 bg-cyan-500/10 text-cyan-700 dark:text-cyan-200">
              {strategy}
            </Badge>
          ))}
          {data.strategies_detected.length === 0 && (
            <span className="text-xs text-muted-foreground">{t('walletAnalysisPanel.noStrategyFingerprint')}</span>
          )}
        </div>
        <PaginationControls
          page={safePage}
          pageSize={pageSize}
          total={anomalies.length}
          itemLabel={t('walletAnalysisPanel.itemAnomalies')}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {anomalies.length === 0 ? (
          <EmptyData
            icon={ShieldCheck}
            title={t('walletAnalysisPanel.noAnomaliesDetected')}
            subtitle={t('walletAnalysisPanel.noAnomaliesSubtitle')}
          />
        ) : (
          <Table className="text-xs">
            <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
              <TableRow className="border-b border-border/80 bg-muted/40">
                <TableHead className="h-9 px-3">{t('walletAnalysisPanel.colSeverity')}</TableHead>
                <TableHead className="h-9 px-3">{t('walletAnalysisPanel.colTypeHeader')}</TableHead>
                <TableHead className="h-9 px-3 text-right">{t('walletAnalysisPanel.colScore')}</TableHead>
                <TableHead className="h-9 px-3 min-w-[280px]">{t('walletAnalysisPanel.colDescription')}</TableHead>
                <TableHead className="h-9 px-3 min-w-[200px]">{t('walletAnalysisPanel.colEvidence')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {pageRows.map((anomaly, index) => (
                <TableRow key={`${anomaly.type}-${index}`} className="border-border/70 align-top">
                  <TableCell className="px-3 py-2.5">
                    <SeverityBadge severity={anomaly.severity} />
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-foreground/90">
                    {anomaly.type.replace(/_/g, ' ')}
                  </TableCell>
                  <TableCell className="px-3 py-2.5 text-right font-mono text-foreground">{anomaly.score.toFixed(2)}</TableCell>
                  <TableCell className="px-3 py-2.5 text-foreground/90">{anomaly.description}</TableCell>
                  <TableCell className="px-3 py-2.5 text-muted-foreground">
                    <EvidencePreview evidence={anomaly.evidence} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  )
}

function SeverityBadge({ severity }: { severity: string }) {
  const normalized = severity.toLowerCase()
  const tone =
    normalized === 'critical'
      ? 'border-red-500/35 bg-red-500/15 text-red-300'
      : normalized === 'high'
      ? 'border-orange-500/35 bg-orange-500/15 text-orange-300'
      : normalized === 'medium'
      ? 'border-amber-500/35 bg-amber-500/15 text-amber-300'
      : 'border-cyan-500/35 bg-cyan-500/15 text-cyan-700 dark:text-cyan-300'

  return (
    <Badge variant="outline" className={cn('text-[10px] uppercase tracking-wide', tone)}>
      {severity}
    </Badge>
  )
}

function EvidencePreview({ evidence }: { evidence: Record<string, unknown> }) {
  const entries = Object.entries(evidence || {})

  if (entries.length === 0) {
    return <span className="text-xs text-muted-foreground">--</span>
  }

  return (
    <div className="space-y-1">
      {entries.slice(0, 2).map(([key, value]) => (
        <div key={key} className="truncate text-[11px]">
          <span className="text-muted-foreground">{key.replace(/_/g, ' ')}:</span>{' '}
          <span className="text-foreground/90">
            {typeof value === 'number' ? value.toFixed(2) : String(value)}
          </span>
        </div>
      ))}
      {entries.length > 2 && <p className="text-[10px] text-muted-foreground">+{entries.length - 2} more</p>}
    </div>
  )
}

export default function WalletAnalysisPanel({ initialWallet, initialUsername, onWalletAnalyzed }: WalletAnalysisPanelProps) {
  const { t } = useTranslation()
  const [searchAddress, setSearchAddress] = useState('')
  const [activeWallet, setActiveWallet] = useState<string | null>(null)
  const [passedUsername, setPassedUsername] = useState<string | null>(null)
  const [inputError, setInputError] = useState<string | null>(null)
  const [isResolvingInput, setIsResolvingInput] = useState(false)
  const [activeTab, setActiveTab] = useState<AnalysisTab>('overview')
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('ALL')

  const [tradesPage, setTradesPage] = useState(1)
  const [tradesPageSize, setTradesPageSize] = useState(20)
  const [positionsPage, setPositionsPage] = useState(1)
  const [positionsPageSize, setPositionsPageSize] = useState(20)
  const [anomaliesPage, setAnomaliesPage] = useState(1)
  const [anomaliesPageSize, setAnomaliesPageSize] = useState(20)

  useEffect(() => {
    if (initialWallet && initialWallet !== activeWallet) {
      setSearchAddress(initialWallet)
      setActiveWallet(initialWallet.trim())
      setPassedUsername(initialUsername || null)
      setActiveTab('overview')
      if (onWalletAnalyzed) {
        onWalletAnalyzed()
      }
    }
  }, [activeWallet, initialUsername, initialWallet, onWalletAnalyzed])

  const pnlQuery = useQuery({
    queryKey: ['wallet-pnl-discover', activeWallet, timePeriod],
    queryFn: () => analyzeWalletPnL(activeWallet!, timePeriod),
    enabled: !!activeWallet,
  })

  const summaryQuery = useQuery({
    queryKey: ['wallet-summary', activeWallet],
    queryFn: () => getWalletSummary(activeWallet!),
    enabled: !!activeWallet,
  })

  const winRateQuery = useQuery({
    queryKey: ['wallet-win-rate', activeWallet, timePeriod],
    queryFn: () => getWalletWinRate(activeWallet!, timePeriod),
    enabled: !!activeWallet,
  })

  const tradesQuery = useQuery({
    queryKey: ['wallet-trades', activeWallet],
    queryFn: () => getWalletTradesAnalysis(activeWallet!, 500),
    enabled: !!activeWallet,
  })

  const positionsQuery = useQuery({
    queryKey: ['wallet-positions', activeWallet],
    queryFn: () => getWalletPositionsAnalysis(activeWallet!),
    enabled: !!activeWallet,
  })

  const anomalyQuery = useQuery({
    queryKey: ['wallet-anomaly', activeWallet],
    queryFn: () => analyzeWallet(activeWallet!),
    enabled: !!activeWallet,
    staleTime: 300000,
    retry: 1,
  })

  const profileQuery = useQuery({
    queryKey: ['wallet-profile', activeWallet],
    queryFn: () => getWalletProfile(activeWallet!),
    enabled: !!activeWallet,
    staleTime: 300000,
  })

  const username = passedUsername || profileQuery.data?.username || null

  const trades = tradesQuery.data?.trades ?? []
  const positions = positionsQuery.data?.positions ?? []
  const anomalies = anomalyQuery.data?.anomalies ?? []

  useEffect(() => {
    setTradesPage(1)
    setPositionsPage(1)
    setAnomaliesPage(1)
  }, [activeWallet, timePeriod])

  useEffect(() => {
    setTradesPage((current) => Math.min(current, Math.max(1, Math.ceil(trades.length / tradesPageSize))))
  }, [trades.length, tradesPageSize])

  useEffect(() => {
    setPositionsPage((current) => Math.min(current, Math.max(1, Math.ceil(positions.length / positionsPageSize))))
  }, [positions.length, positionsPageSize])

  useEffect(() => {
    setAnomaliesPage((current) => Math.min(current, Math.max(1, Math.ceil(anomalies.length / anomaliesPageSize))))
  }, [anomalies.length, anomaliesPageSize])

  const handleAnalyze = async () => {
    const value = searchAddress.trim()
    if (!value) return
    setInputError(null)
    setIsResolvingInput(true)
    try {
      const profile = await getWalletProfile(value)
      const resolvedAddress = String(profile.address || '').trim().toLowerCase()
      if (!resolvedAddress) {
        throw new Error(t('walletAnalysisPanel.unableToResolveWallet'))
      }
      setActiveWallet(resolvedAddress)
      setSearchAddress(resolvedAddress)
      setPassedUsername(profile.username || null)
      setActiveTab('overview')
    } catch (error) {
      setInputError(readErrorMessage(error))
    } finally {
      setIsResolvingInput(false)
    }
  }

  const handleRefresh = () => {
    void pnlQuery.refetch()
    void summaryQuery.refetch()
    void winRateQuery.refetch()
    void tradesQuery.refetch()
    void positionsQuery.refetch()
    void anomalyQuery.refetch()
    void profileQuery.refetch()
  }

  const summaryData = summaryQuery.data?.summary
  const totalPnl = pnlQuery.data?.total_pnl ?? summaryData?.total_pnl ?? 0
  const roiPercent = pnlQuery.data?.roi_percent ?? summaryData?.roi_percent ?? 0
  const totalInvested = pnlQuery.data?.total_invested ?? summaryData?.total_invested ?? 0
  const totalReturned = pnlQuery.data?.total_returned ?? summaryData?.total_returned ?? 0
  const totalTrades = pnlQuery.data?.total_trades ?? summaryData?.total_trades ?? trades.length
  const volume = totalInvested + totalReturned
  const winRate = winRateQuery.data?.win_rate ?? 0
  const isProfitable = totalPnl >= 0

  const sparklineValues = useMemo(() => {
    if (trades.length < 2) return []

    const ordered = [...trades].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    let cumulative = 0

    return ordered.map((trade) => {
      cumulative += trade.side === 'SELL' ? trade.cost : -trade.cost
      return cumulative
    })
  }, [trades])

  const firstError =
    pnlQuery.error ||
    summaryQuery.error ||
    winRateQuery.error ||
    tradesQuery.error ||
    positionsQuery.error ||
    anomalyQuery.error

  const risk = riskModel(anomalyQuery.data?.anomaly_score ?? 0)
  const isHeaderLoading = pnlQuery.isLoading || summaryQuery.isLoading || winRateQuery.isLoading

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <Card className="shrink-0 border-border/80 bg-card/80 dark:bg-gradient-to-r dark:from-slate-900/50 dark:via-cyan-950/25 dark:to-emerald-950/20">
        <CardContent className="p-4">
          <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-wide text-cyan-700 dark:text-cyan-200/90">{t('walletAnalysisPanel.headerEyebrow')}</p>
              <h2 className="mt-1 text-lg font-semibold text-foreground">{t('walletAnalysisPanel.headerTitle')}</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {t('walletAnalysisPanel.headerSubtitle')}
              </p>
            </div>

            <div className="flex flex-wrap items-end gap-2">
              <div className="relative w-[360px] max-w-full">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  type="text"
                  value={searchAddress}
                  onChange={(event) => setSearchAddress(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      void handleAnalyze()
                    }
                  }}
                  placeholder={t('walletAnalysisPanel.searchPlaceholder')}
                  className="h-9 border-border bg-background/80 pl-10 font-mono text-xs"
                />
              </div>

              <Button
                onClick={() => {
                  void handleAnalyze()
                }}
                disabled={!searchAddress.trim() || isResolvingInput}
                className="h-9 bg-cyan-500 text-slate-950 hover:bg-cyan-400"
              >
                <Search className={cn('mr-1.5 h-3.5 w-3.5', isResolvingInput && 'animate-spin')} />
                {t('walletAnalysisPanel.analyze')}
              </Button>

              <Button
                variant="outline"
                className="h-9"
                onClick={handleRefresh}
                disabled={!activeWallet}
              >
                <RefreshCw className={cn('mr-1.5 h-3.5 w-3.5', (pnlQuery.isFetching || summaryQuery.isFetching) && 'animate-spin')} />
                {t('walletAnalysisPanel.refresh')}
              </Button>

              <div className="flex h-9 items-center rounded-lg border border-border bg-background/70 p-0.5">
                {TIME_PERIOD_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setTimePeriod(option.value)}
                    className={cn(
                      'h-8 rounded-md px-2.5 text-xs transition-colors',
                      timePeriod === option.value
                        ? 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-200'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    {t(option.labelKey)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {activeWallet && (
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
              <Badge variant="outline" className="border-cyan-500/30 bg-cyan-500/10 text-cyan-700 dark:text-cyan-200">
                {t('walletAnalysisPanel.activeBadge', { address: shortAddress(activeWallet) })}
              </Badge>
              <Badge variant="outline" className={risk.badgeClass}>
                {t('walletAnalysisPanel.riskBadge', { label: t(risk.labelKey) })}
              </Badge>
              <Button
                size="sm"
                variant="outline"
                className="ml-auto h-7 gap-1.5 border-violet-500/40 text-violet-700 dark:text-violet-200 hover:bg-violet-500/10 text-[11px]"
                onClick={() => {
                  // Hand the wallet over to Strategy Research → Reverse
                  // Engineer.  Stored in sessionStorage so the receiving
                  // page can pre-fill the form without us coupling to
                  // App.tsx routing internals.
                  try {
                    sessionStorage.setItem('homerun:reverse-engineer:wallet', activeWallet)
                  } catch {
                    /* ignore */
                  }
                  // Best-effort programmatic navigation: dispatch a
                  // CustomEvent that App.tsx listens for, then surface
                  // a manual click target as a fallback.
                  try {
                    window.dispatchEvent(
                      new CustomEvent('homerun:navigate', {
                        detail: { tab: 'strategies', subtab: 'research', researchInner: 'reverse' },
                      }),
                    )
                  } catch {
                    /* ignore */
                  }
                }}
                title={t('walletAnalysisPanel.reverseEngineerTitle')}
              >
                <Sparkles className="h-3 w-3" />
                {t('walletAnalysisPanel.reverseEngineerStrategy')}
              </Button>
            </div>
          )}

          {firstError && activeWallet && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
              {readErrorMessage(firstError)}
            </div>
          )}

          {inputError && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
              {inputError}
            </div>
          )}
        </CardContent>
      </Card>

      {!activeWallet ? (
        <Card className="flex-1 border-border/80">
          <CardContent className="flex h-full flex-col items-center justify-center px-6 text-center">
            <Wallet className="mb-4 h-12 w-12 text-muted-foreground/35" />
            <p className="text-sm text-foreground">{t('walletAnalysisPanel.noWalletSelected')}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {t('walletAnalysisPanel.noWalletSubtitle')}
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card className="flex min-h-0 flex-1 flex-col overflow-hidden border-border/80 bg-card/80">
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border/70 bg-background/40 px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                {TAB_OPTIONS.map((tab) => (
                  <Button
                    key={tab.id}
                    variant="outline"
                    size="sm"
                    onClick={() => setActiveTab(tab.id)}
                    className={cn(
                      'h-8 gap-1.5 text-xs',
                      activeTab === tab.id
                        ? 'border-cyan-500/30 bg-cyan-500/15 text-cyan-700 dark:text-cyan-200 hover:bg-cyan-500/20'
                        : 'border-border bg-background/70 text-muted-foreground hover:text-foreground',
                    )}
                  >
                    <tab.icon className="h-3.5 w-3.5" />
                    {t(tab.labelKey)}
                  </Button>
                ))}
              </div>
            </div>

            <div className="min-h-0 flex-1">
              {activeTab === 'overview' && (
                <OverviewHeroPanel
                  isLoading={summaryQuery.isLoading || pnlQuery.isLoading || winRateQuery.isLoading}
                  activeWallet={activeWallet}
                  username={username}
                  timePeriod={timePeriod}
                  anomalyScore={anomalyQuery.data?.anomaly_score ?? 0}
                  riskLabel={t(risk.labelKey)}
                  riskBadgeClass={risk.badgeClass}
                  totalPnl={totalPnl}
                  roiPercent={roiPercent}
                  isProfitable={isProfitable}
                  winRate={winRate}
                  wins={winRateQuery.data?.wins ?? 0}
                  losses={winRateQuery.data?.losses ?? 0}
                  volume={volume}
                  totalTrades={totalTrades}
                  sparklineValues={sparklineValues}
                  realizedPnl={pnlQuery.data?.realized_pnl ?? summaryData?.realized_pnl ?? 0}
                  unrealizedPnl={pnlQuery.data?.unrealized_pnl ?? summaryData?.unrealized_pnl ?? 0}
                  isHeaderLoading={isHeaderLoading}
                  positionsCount={positions.length}
                  anomaliesCount={anomalies.length}
                />
              )}

              {activeTab === 'trades' && (
                <TradesPanel
                  isLoading={tradesQuery.isLoading}
                  trades={trades}
                  page={tradesPage}
                  pageSize={tradesPageSize}
                  onPageChange={setTradesPage}
                  onPageSizeChange={(size) => {
                    setTradesPageSize(size)
                    setTradesPage(1)
                  }}
                />
              )}

              {activeTab === 'positions' && (
                <PositionsPanel
                  isLoading={positionsQuery.isLoading}
                  data={positionsQuery.data}
                  page={positionsPage}
                  pageSize={positionsPageSize}
                  onPageChange={setPositionsPage}
                  onPageSizeChange={(size) => {
                    setPositionsPageSize(size)
                    setPositionsPage(1)
                  }}
                />
              )}

              {activeTab === 'risk' && (
                <RiskPanel
                  isLoading={anomalyQuery.isLoading}
                  data={anomalyQuery.data}
                  page={anomaliesPage}
                  pageSize={anomaliesPageSize}
                  onPageChange={setAnomaliesPage}
                  onPageSizeChange={(size) => {
                    setAnomaliesPageSize(size)
                    setAnomaliesPage(1)
                  }}
                />
              )}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
