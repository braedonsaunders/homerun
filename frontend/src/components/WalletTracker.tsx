import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  Trash2,
  Wallet,
  ExternalLink,
  RefreshCw,
  Star,
  UserPlus,
  Activity,
  Filter,
  Search,
  Trophy,
  Target,
  Bot,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import RecentTradesPanel from './RecentTradesPanel'
import { discoveryApi, type PoolMember } from '../services/discoveryApi'
import {
  getWallets,
  addWallet,
  removeWallet,
  analyzeWalletPnL,
  getWalletWinRate,
  Wallet as WalletType,
  discoverTopTraders,
  discoverByWinRate,
  analyzeAndTrackWallet,
  TimePeriod,
  OrderBy,
  Category,
} from '../services/api'
import AddWalletToBotDialog from './AddWalletToBotDialog'
import type { AddWalletToTraderBotResult } from '../lib/traderBotActions'

interface WalletTrackerProps {
  onAnalyzeWallet?: (address: string, username?: string) => void
  section?: 'tracked' | 'discover'
  discoverMode?: 'leaderboard' | 'winrate'
  onNavigateToWallet?: (address: string) => void
  showManagementPanel?: boolean
}

type MetricTone = 'good' | 'warn' | 'bad' | 'neutral' | 'info'
type WalletFallbackMetrics = {
  totalPnl: number | null
  winRate: number | null
  totalTrades: number | null
  openPositions: number | null
}

const SCORE_DELTA_HIDE_THRESHOLD = 0.0025

const METRIC_TONE_CLASSES: Record<MetricTone, string> = {
  good: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/35 dark:bg-sky-500/15 dark:text-sky-100',
  warn: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/40 dark:bg-amber-500/18 dark:text-amber-100',
  bad: 'border-rose-300 bg-rose-100 text-rose-900 dark:border-rose-400/45 dark:bg-rose-500/18 dark:text-rose-100',
  neutral: 'border-slate-300 bg-slate-100 text-slate-800 dark:border-border/85 dark:bg-muted/55 dark:text-foreground/90',
  info: 'border-indigo-300 bg-indigo-100 text-indigo-900 dark:border-indigo-400/40 dark:bg-indigo-500/16 dark:text-indigo-100',
}

const METRIC_BAR_CLASSES: Record<MetricTone, string> = {
  good: 'bg-sky-600 dark:bg-sky-300/95',
  warn: 'bg-amber-600 dark:bg-amber-300/95',
  bad: 'bg-rose-600 dark:bg-rose-300/95',
  neutral: 'bg-slate-500 dark:bg-muted-foreground/80',
  info: 'bg-indigo-600 dark:bg-indigo-300/95',
}

function shortAddress(address: string, unknownLabel = 'unknown'): string {
  if (!address) return unknownLabel
  if (address.length <= 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function walletDisplayName(wallet: WalletType, unknownLabel = 'unknown'): string {
  return wallet.username || wallet.label || shortAddress(wallet.address, unknownLabel)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return clamp(value, 0, 1)
}

function normalizePercentRatio(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.abs(value) <= 1 ? value * 100 : value
}

function formatScorePct(value: number): string {
  return `${(clamp01(value) * 100).toFixed(1)}%`
}

function formatNumber(value: number): string {
  if (value == null) return '0'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

function formatWinRate(value: number): string {
  return `${normalizePercentRatio(value).toFixed(1)}%`
}

function formatPnl(value: number): string {
  if (value == null) return '0.00'
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return value.toFixed(2)
}

function timeAgo(
  dateStr: string | null,
  t: (key: string, opts?: Record<string, unknown>) => string,
): string {
  if (!dateStr) return t('walletTracker.timeNever')
  const diff = Date.now() - new Date(dateStr).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return t('walletTracker.timeJustNow')
  if (minutes < 60) return t('walletTracker.timeMinutesAgo', { n: minutes })
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return t('walletTracker.timeHoursAgo', { n: hours })
  const days = Math.floor(hours / 24)
  return t('walletTracker.timeDaysAgo', { n: days })
}

function scoreTone(value: number, goodThreshold: number, warnThreshold: number): MetricTone {
  if (!Number.isFinite(value)) return 'neutral'
  if (value >= goodThreshold) return 'good'
  if (value >= warnThreshold) return 'warn'
  return 'neutral'
}

function inverseScoreTone(value: number, badThreshold: number, warnThreshold: number): MetricTone {
  if (!Number.isFinite(value)) return 'neutral'
  if (value >= badThreshold) return 'bad'
  if (value >= warnThreshold) return 'warn'
  return 'good'
}

function selectionReasonTone(code: string): MetricTone {
  const normalized = code.toLowerCase()
  if (
    normalized.includes('manual_include')
    || normalized.includes('core')
    || normalized.includes('quality')
    || normalized.includes('active')
    || normalized.includes('tracked')
    || normalized.includes('elite')
  ) {
    return 'good'
  }
  if (
    normalized.includes('exclude')
    || normalized.includes('blacklist')
    || normalized.includes('below')
    || normalized.includes('blocked')
    || normalized.includes('anomaly')
  ) {
    return 'bad'
  }
  if (
    normalized.includes('rising')
    || normalized.includes('churn')
    || normalized.includes('tier')
    || normalized.includes('pending')
  ) {
    return 'warn'
  }
  return 'neutral'
}

function reasonLabelFromCode(
  code: string,
  t: (key: string, opts?: Record<string, unknown>) => string,
): string {
  const clean = String(code || '').trim()
  if (!clean) return t('walletTracker.reasonSelectionDetail')
  return clean
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function selectionReasons(
  member: PoolMember,
  t: (key: string, opts?: Record<string, unknown>) => string,
): Array<{ code: string; label: string; detail?: string }> {
  const raw = Array.isArray(member.selection_reasons) ? member.selection_reasons : []
  const normalized: Array<{ code: string; label: string; detail?: string }> = []
  for (const item of raw) {
    const code = String(item?.code || '').trim()
    const label = String(item?.label || '').trim()
    const detail = typeof item?.detail === 'string' ? item.detail.trim() : undefined
    if (!code && !label) continue
    normalized.push({
      code: code || `reason_${label.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
      label: label || reasonLabelFromCode(code, t),
      detail,
    })
  }

  if (normalized.length > 0) return normalized
  if (member.pool_membership_reason) {
    return [{
      code: member.pool_membership_reason,
      label: reasonLabelFromCode(member.pool_membership_reason, t),
    }]
  }
  if (member.tracked_wallet) {
    return [{
      code: 'tracked_wallet',
      label: t('walletTracker.reasonTrackedWallet'),
      detail: t('walletTracker.reasonTrackedWalletDetail'),
    }]
  }
  return [{
    code: 'selection_reason_unknown',
    label: t('walletTracker.reasonUnavailable'),
  }]
}

function MetricPill({
  label,
  value,
  tone = 'neutral',
  className,
  mono = true,
}: {
  label: string
  value: string
  tone?: MetricTone
  className?: string
  mono?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-none',
        METRIC_TONE_CLASSES[tone],
        className,
      )}
    >
      <span className="uppercase tracking-wide opacity-90">{label}</span>
      <span className={cn('font-semibold', mono && 'font-mono')}>{value}</span>
    </span>
  )
}

function ScoreSparkline({
  points,
}: {
  points: Array<{ key: string; label: string; value: number; tone?: MetricTone }>
}) {
  if (!points.length) return null
  return (
    <div className="inline-flex items-end gap-0.5 rounded-md border border-slate-300 bg-slate-100 px-1.5 py-0.5 dark:border-border/85 dark:bg-muted/45">
      {points.map((point) => {
        const normalized = clamp01(point.value)
        const height = Math.max(3, Math.round(normalized * 13))
        const tone = point.tone || 'neutral'
        return (
          <span
            key={point.key}
            title={`${point.label} ${(normalized * 100).toFixed(1)}%`}
            className={cn('w-1.5 rounded-sm', METRIC_BAR_CLASSES[tone])}
            style={{ height }}
          />
        )
      })}
    </div>
  )
}

function PnlDisplay({ value, className }: { value: number; className?: string }) {
  const isPositive = value >= 0
  return (
    <span
      className={cn(
        'font-medium font-mono text-sm',
        isPositive ? 'text-sky-700 dark:text-sky-300' : 'text-rose-700 dark:text-rose-300',
        className,
      )}
    >
      {isPositive ? '+' : ''}${formatPnl(value)}
    </span>
  )
}

export default function WalletTracker({
  onAnalyzeWallet,
  section: propSection,
  discoverMode: propDiscoverMode,
  onNavigateToWallet,
  showManagementPanel = true,
}: WalletTrackerProps) {
  const { t } = useTranslation()
  const [newAddress, setNewAddress] = useState('')
  const [newLabel, setNewLabel] = useState('')
  const [activeSection, setActiveSection] = useState<'tracked' | 'discover'>('discover')
  const [discoverModeState, setDiscoverMode] = useState<'leaderboard' | 'winrate'>('leaderboard')
  const [showFilters, setShowFilters] = useState(false)
  const [addToBotWallet, setAddToBotWallet] = useState<{ address: string; label?: string | null } | null>(null)
  const [addToBotMessage, setAddToBotMessage] = useState<string | null>(null)

  // Use props if provided, otherwise use internal state
  const currentSection = propSection ?? activeSection
  const currentDiscoverMode = propDiscoverMode ?? discoverModeState

  // Filter states
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('ALL')
  const [orderBy, setOrderBy] = useState<OrderBy>('PNL')
  const [category, setCategory] = useState<Category>('OVERALL')
  const [minWinRate, setMinWinRate] = useState(70)
  const [minTrades, setMinTrades] = useState(10)
  const [minVolume, setMinVolume] = useState(0)
  const [maxVolume, setMaxVolume] = useState(0)
  const [scanCount, setScanCount] = useState(200)
  const [resultLimit, setResultLimit] = useState(100)

  const queryClient = useQueryClient()

  const invalidateTrackedQueries = () => {
    queryClient.invalidateQueries({ queryKey: ['wallets'] })
    queryClient.invalidateQueries({ queryKey: ['recent-trades-from-wallets'] })
    queryClient.invalidateQueries({ queryKey: ['trader-groups'] })
    queryClient.invalidateQueries({ queryKey: ['trader-group-suggestions'] })
    queryClient.invalidateQueries({ queryKey: ['opportunities', 'traders'] })
    queryClient.invalidateQueries({ queryKey: ['traders-overview'] })
  }

  const { data: wallets = [], isLoading } = useQuery({
    queryKey: ['wallets'],
    queryFn: getWallets,
    refetchInterval: 30000,
  })

  const trackedWalletAddressKey = useMemo(
    () => wallets.map((wallet) => wallet.address.toLowerCase()).sort().join(','),
    [wallets],
  )

  const {
    data: trackedPoolMembers = [],
    isLoading: trackedPoolMembersLoading,
    isError: trackedPoolMembersError,
  } = useQuery({
    queryKey: ['tracked-wallet-pool-members', trackedWalletAddressKey],
    enabled: currentSection === 'tracked' && wallets.length > 0,
    staleTime: 60_000,
    queryFn: async () => {
      const pageSize = 500
      let offset = 0
      let total = 0
      const members: PoolMember[] = []

      do {
        const page = await discoveryApi.getPoolMembers({
          limit: pageSize,
          offset,
          pool_only: false,
          tracked_only: true,
          include_blacklisted: true,
          sort_by: 'selection_score',
          sort_dir: 'desc',
        })

        const pageMembers = Array.isArray(page.members) ? page.members : []
        members.push(...pageMembers)
        total = Number(page.total || members.length)
        if (pageMembers.length === 0) break
        offset += pageMembers.length
        if (pageMembers.length < pageSize) break
      } while (offset < total)

      return members
    },
  })

  // Leaderboard query
  const { data: discoveredTraders = [], isLoading: discoveringTraders, refetch: refreshTraders } = useQuery({
    queryKey: ['discovered-traders', timePeriod, orderBy, category],
    queryFn: () => discoverTopTraders(50, 5, { time_period: timePeriod, order_by: orderBy, category }),
    refetchInterval: 60000,
    enabled: currentDiscoverMode === 'leaderboard',
  })

  // Win rate discovery query
  const { data: winRateTraders = [], isLoading: loadingWinRate, refetch: refreshWinRate } = useQuery({
    queryKey: ['win-rate-traders', minWinRate, minTrades, timePeriod, category, minVolume, maxVolume, scanCount, resultLimit],
    queryFn: () => discoverByWinRate({
      min_win_rate: minWinRate,
      min_trades: minTrades,
      limit: resultLimit,
      time_period: timePeriod,
      category,
      min_volume: minVolume > 0 ? minVolume : undefined,
      max_volume: maxVolume > 0 ? maxVolume : undefined,
      scan_count: scanCount
    }),
    refetchInterval: 120000,
    enabled: currentDiscoverMode === 'winrate',
  })

  const addMutation = useMutation({
    mutationFn: ({ address, label }: { address: string; label?: string }) =>
      addWallet(address, label),
    onSuccess: () => {
      invalidateTrackedQueries()
      setNewAddress('')
      setNewLabel('')
    },
  })

  const removeMutation = useMutation({
    mutationFn: removeWallet,
    onSuccess: () => {
      invalidateTrackedQueries()
    },
  })

  const trackWalletMutation = useMutation({
    mutationFn: (params: { address: string; label?: string }) =>
      analyzeAndTrackWallet(params),
    onSuccess: () => {
      invalidateTrackedQueries()
    },
  })

  const handleAdd = () => {
    if (!newAddress.trim()) return
    addMutation.mutate({ address: newAddress.trim(), label: newLabel.trim() || undefined })
  }

  const handleAnalyze = (address: string, username?: string) => {
    if (onAnalyzeWallet) {
      onAnalyzeWallet(address, username)
      return
    }
    onNavigateToWallet?.(address)
  }

  const handleTrackOnly = (address: string) => {
    const allTraders = currentDiscoverMode === 'winrate' ? winRateTraders : discoveredTraders
    const trader = allTraders.find(tr => tr.address === address)
    const winRateStr = trader?.win_rate
      ? ` | ${t('walletTracker.discoveredLabelWinRate', { wr: trader.win_rate.toFixed(1) })}`
      : ''
    const label = t('walletTracker.discoveredLabel', {
      vol: trader?.volume?.toFixed(0) || '?',
      wr: winRateStr,
    })
    trackWalletMutation.mutate({
      address,
      label,
    })
  }

  const openAddToBotDialog = (address: string, label?: string | null) => {
    setAddToBotWallet({ address, label })
  }

  const handleAddToBotSuccess = (result: AddWalletToTraderBotResult) => {
    const action = result.created
      ? t('walletTracker.addToBotCreated')
      : t('walletTracker.addToBotAdded')
    setAddToBotMessage(`${action}: ${result.trader.name}`)
    setTimeout(() => setAddToBotMessage(null), 4500)
  }

  const currentTraders = currentDiscoverMode === 'winrate' ? winRateTraders : discoveredTraders
  const isLoadingTraders = currentDiscoverMode === 'winrate' ? loadingWinRate : discoveringTraders
  const refreshCurrentTraders = currentDiscoverMode === 'winrate' ? refreshWinRate : refreshTraders
  const trackedPoolMemberMap = useMemo(() => {
    const map = new Map<string, PoolMember>()
    for (const member of trackedPoolMembers) {
      map.set(member.address.toLowerCase(), member)
    }
    return map
  }, [trackedPoolMembers])

  const missingPoolMetricAddresses = useMemo(
    () =>
      wallets
        .map((wallet) => wallet.address.toLowerCase())
        .filter((address) => !trackedPoolMemberMap.has(address)),
    [wallets, trackedPoolMemberMap],
  )
  const missingPoolMetricKey = missingPoolMetricAddresses.join(',')

  const {
    data: fallbackWalletMetrics = {},
    isLoading: fallbackWalletMetricsLoading,
  } = useQuery({
    queryKey: ['tracked-wallet-fallback-metrics', missingPoolMetricKey],
    enabled: currentSection === 'tracked' && missingPoolMetricAddresses.length > 0,
    staleTime: 60_000,
    queryFn: async () => {
      const entries = await Promise.all(
        missingPoolMetricAddresses.map(async (address) => {
          const [pnlResult, winRateResult] = await Promise.allSettled([
            analyzeWalletPnL(address),
            getWalletWinRate(address),
          ])

          const pnl = pnlResult.status === 'fulfilled' ? pnlResult.value : null
          const winRate = winRateResult.status === 'fulfilled' ? winRateResult.value : null

          const metrics: WalletFallbackMetrics = {
            totalPnl: pnl?.total_pnl ?? null,
            winRate: winRate?.win_rate ?? null,
            totalTrades: pnl?.total_trades ?? winRate?.trade_count ?? null,
            openPositions: pnl?.open_positions ?? null,
          }
          return [address, metrics] as const
        }),
      )
      return Object.fromEntries(entries) as Record<string, WalletFallbackMetrics>
    },
  })

  const fallbackMetricCount = useMemo(
    () =>
      Object.values(fallbackWalletMetrics).filter(
        (metrics) =>
          metrics.totalPnl != null
          || metrics.winRate != null
          || metrics.totalTrades != null
          || metrics.openPositions != null,
      ).length,
    [fallbackWalletMetrics],
  )

  const unknownLabel = t('walletTracker.unknownAddress')

  const trackedWalletRows = useMemo(() => {
    const rows = wallets.map((wallet) => ({
      wallet,
      member: trackedPoolMemberMap.get(wallet.address.toLowerCase()),
    }))

    rows.sort((a, b) => {
      const aHasMember = Boolean(a.member)
      const bHasMember = Boolean(b.member)
      if (aHasMember && bHasMember) {
        const aSelection = Number(a.member?.selection_score ?? a.member?.composite_score ?? 0)
        const bSelection = Number(b.member?.selection_score ?? b.member?.composite_score ?? 0)
        if (aSelection !== bSelection) return bSelection - aSelection
      } else if (aHasMember) {
        return -1
      } else if (bHasMember) {
        return 1
      }

      return walletDisplayName(a.wallet, unknownLabel).localeCompare(walletDisplayName(b.wallet, unknownLabel), undefined, {
        sensitivity: 'base',
      })
    })

    return rows
  }, [wallets, trackedPoolMemberMap, unknownLabel])

  // Check if navigation is controlled by parent
  const isControlledByParent = propSection !== undefined

  return (
    <div className="space-y-6">
      {/* Section Tabs - only show if not controlled by parent */}
      {!isControlledByParent && (
        <Tabs value={currentSection} onValueChange={(v) => setActiveSection(v as 'tracked' | 'discover')}>
          <TabsList className="flex h-auto gap-2 bg-transparent p-0">
            <TabsTrigger
              value="discover"
              className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-green-500/20 data-[state=active]:text-green-400 data-[state=active]:border data-[state=active]:border-green-500/50 data-[state=active]:shadow-none"
            >
              <Search className="w-4 h-4" />
              {t('walletTracker.tabDiscoverTopTraders')}
            </TabsTrigger>
            <TabsTrigger
              value="tracked"
              className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400 data-[state=active]:border data-[state=active]:border-blue-500/50 data-[state=active]:shadow-none"
            >
              <Wallet className="w-4 h-4" />
              {t('walletTracker.tabTrackedWallets', { count: wallets.length })}
            </TabsTrigger>
          </TabsList>
        </Tabs>
      )}

      {addToBotMessage && (
        <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
          {addToBotMessage}
        </div>
      )}

      {currentSection === 'discover' && (
        <>
          {/* Discovery Mode Toggle - only show if not controlled by parent */}
          {!isControlledByParent && (
            <Tabs value={currentDiscoverMode} onValueChange={(v) => setDiscoverMode(v as 'leaderboard' | 'winrate')}>
              <TabsList className="flex h-auto gap-2 mb-4 bg-transparent p-0">
                <TabsTrigger
                  value="leaderboard"
                  className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-yellow-500/20 data-[state=active]:text-yellow-400 data-[state=active]:border data-[state=active]:border-yellow-500/50 data-[state=active]:shadow-none"
                >
                  <Trophy className="w-4 h-4" />
                  {t('walletTracker.tabLeaderboard')}
                </TabsTrigger>
                <TabsTrigger
                  value="winrate"
                  className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-400 data-[state=active]:border data-[state=active]:border-emerald-500/50 data-[state=active]:shadow-none"
                >
                  <Target className="w-4 h-4" />
                  {t('walletTracker.tabHighWinRate')}
                </TabsTrigger>
              </TabsList>
            </Tabs>
          )}

          {/* Discovery Header */}
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-lg font-medium flex items-center gap-2">
                  {currentDiscoverMode === 'winrate' ? (
                    <>
                      <Target className="w-5 h-5 text-emerald-500" />
                      {t('walletTracker.headingDiscoverTraders')}
                    </>
                  ) : (
                    <>
                      <Star className="w-5 h-5 text-yellow-500" />
                      {t('walletTracker.headingTopActiveTraders')}
                    </>
                  )}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {currentDiscoverMode === 'winrate'
                    ? minVolume > 0
                      ? t('walletTracker.scanningWithVolume', {
                          n: scanCount * 2,
                          wr: minWinRate,
                          vol: minVolume.toLocaleString(),
                        })
                      : t('walletTracker.scanning', { n: scanCount * 2, wr: minWinRate })
                    : t('walletTracker.discoveredFromLeaderboard')}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  onClick={() => setShowFilters(!showFilters)}
                  className={cn(
                    "h-auto gap-2 px-3 py-1.5 rounded-lg text-sm",
                    showFilters ? "bg-blue-500/20 text-blue-400" : "bg-muted hover:bg-accent"
                  )}
                >
                  <Filter className="w-4 h-4" />
                  {t('walletTracker.filters')}
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => refreshCurrentTraders()}
                  disabled={isLoadingTraders}
                  className="h-auto gap-2 px-3 py-1.5 bg-muted rounded-lg text-sm hover:bg-accent"
                >
                  <RefreshCw className={cn("w-4 h-4", isLoadingTraders && "animate-spin")} />
                  {t('walletTracker.refresh')}
                </Button>
              </div>
            </div>

            {/* Filters Panel */}
            {showFilters && (
              <div className="mb-4 p-3 bg-muted rounded-lg space-y-3">
                {/* Row 1: Basic filters */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterTimePeriod')}</label>
                    <select
                      value={timePeriod}
                      onChange={(e) => setTimePeriod(e.target.value as TimePeriod)}
                      className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                    >
                      <option value="ALL">{t('walletTracker.timeAllTime')}</option>
                      <option value="MONTH">{t('walletTracker.timeLast30Days')}</option>
                      <option value="WEEK">{t('walletTracker.timeLast7Days')}</option>
                      <option value="DAY">{t('walletTracker.timeLast24Hours')}</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterCategory')}</label>
                    <select
                      value={category}
                      onChange={(e) => setCategory(e.target.value as Category)}
                      className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                    >
                      <option value="OVERALL">{t('walletTracker.categoryAll')}</option>
                      <option value="POLITICS">{t('walletTracker.categoryPolitics')}</option>
                      <option value="SPORTS">{t('walletTracker.categorySports')}</option>
                      <option value="CRYPTO">{t('walletTracker.categoryCrypto')}</option>
                      <option value="CULTURE">{t('walletTracker.categoryCulture')}</option>
                      <option value="ECONOMICS">{t('walletTracker.categoryEconomics')}</option>
                      <option value="TECH">{t('walletTracker.categoryTech')}</option>
                      <option value="FINANCE">{t('walletTracker.categoryFinance')}</option>
                    </select>
                  </div>
                  {currentDiscoverMode === 'leaderboard' && (
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterSortBy')}</label>
                      <select
                        value={orderBy}
                        onChange={(e) => setOrderBy(e.target.value as OrderBy)}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value="PNL">{t('walletTracker.sortProfitLoss')}</option>
                        <option value="VOL">{t('walletTracker.sortVolume')}</option>
                      </select>
                    </div>
                  )}
                  {currentDiscoverMode === 'winrate' && (
                    <>
                      <div>
                        <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterMinWinRate')}</label>
                        <select
                          value={minWinRate}
                          onChange={(e) => setMinWinRate(Number(e.target.value))}
                          className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                        >
                          <option value={50}>50%+</option>
                          <option value={60}>60%+</option>
                          <option value={70}>70%+</option>
                          <option value={80}>80%+</option>
                          <option value={90}>90%+</option>
                          <option value={95}>95%+</option>
                          <option value={97}>97%+</option>
                          <option value={98}>98%+</option>
                          <option value={99}>99%+</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterMinTrades')}</label>
                        <select
                          value={minTrades}
                          onChange={(e) => setMinTrades(Number(e.target.value))}
                          className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                        >
                          <option value={3}>{t('walletTracker.tradesPlus', { n: 3 })}</option>
                          <option value={5}>{t('walletTracker.tradesPlus', { n: 5 })}</option>
                          <option value={10}>{t('walletTracker.tradesPlus', { n: 10 })}</option>
                          <option value={20}>{t('walletTracker.tradesPlus', { n: 20 })}</option>
                          <option value={50}>{t('walletTracker.tradesPlus', { n: 50 })}</option>
                          <option value={100}>{t('walletTracker.tradesPlus', { n: 100 })}</option>
                          <option value={200}>{t('walletTracker.tradesPlus', { n: 200 })}</option>
                          <option value={500}>{t('walletTracker.tradesPlus', { n: 500 })}</option>
                          <option value={1000}>{t('walletTracker.tradesPlus', { n: 1000 })}</option>
                        </select>
                      </div>
                    </>
                  )}
                </div>

                {/* Row 2: Advanced win rate filters */}
                {currentDiscoverMode === 'winrate' && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-border">
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterMinVolume')}</label>
                      <select
                        value={minVolume}
                        onChange={(e) => setMinVolume(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={0}>{t('walletTracker.volumeNoMin')}</option>
                        <option value={1000}>$1,000+</option>
                        <option value={5000}>$5,000+</option>
                        <option value={10000}>$10,000+</option>
                        <option value={25000}>$25,000+</option>
                        <option value={50000}>$50,000+</option>
                        <option value={100000}>$100,000+</option>
                        <option value={500000}>$500,000+</option>
                        <option value={1000000}>$1,000,000+</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterMaxVolume')}</label>
                      <select
                        value={maxVolume}
                        onChange={(e) => setMaxVolume(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={0}>{t('walletTracker.volumeNoMax')}</option>
                        <option value={10000}>$10,000</option>
                        <option value={50000}>$50,000</option>
                        <option value={100000}>$100,000</option>
                        <option value={500000}>$500,000</option>
                        <option value={1000000}>$1,000,000</option>
                        <option value={5000000}>$5,000,000</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterScanDepth')}</label>
                      <select
                        value={scanCount}
                        onChange={(e) => setScanCount(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={200}>{t('walletTracker.scanDepthFast', { n: 200 })}</option>
                        <option value={500}>{t('walletTracker.scanDepthDefault', { n: 500 })}</option>
                        <option value={750}>{t('walletTracker.scanDepthDeep', { n: 750 })}</option>
                        <option value={1000}>{t('walletTracker.scanDepthMax', { n: 1000 })}</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">{t('walletTracker.filterShowResults')}</label>
                      <select
                        value={resultLimit}
                        onChange={(e) => setResultLimit(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={25}>{t('walletTracker.resultsCount', { n: 25 })}</option>
                        <option value={50}>{t('walletTracker.resultsCount', { n: 50 })}</option>
                        <option value={100}>{t('walletTracker.resultsCount', { n: 100 })}</option>
                        <option value={200}>{t('walletTracker.resultsCount', { n: 200 })}</option>
                        <option value={500}>{t('walletTracker.resultsCount', { n: 500 })}</option>
                      </select>
                    </div>
                  </div>
                )}

                {/* Tip for high win rate searches */}
                {currentDiscoverMode === 'winrate' && minWinRate >= 95 && (
                  <div className="text-xs text-yellow-500/80 flex items-center gap-1 pt-1">
                    <span>{t('walletTracker.tipHighWinRate')}</span>
                  </div>
                )}
              </div>
            )}

            {isLoadingTraders ? (
              <div className="flex flex-col items-center justify-center py-8">
                <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
                <span className="mt-2 text-muted-foreground">
                  {currentDiscoverMode === 'winrate'
                    ? t('walletTracker.scanningLoading', { n: scanCount * 2, wr: minWinRate })
                    : t('walletTracker.verifyingActivity')}
                </span>
                {currentDiscoverMode === 'winrate' && (
                  <span className="text-xs text-muted-foreground mt-1">
                    {t('walletTracker.analyzingClosed')}
                  </span>
                )}
              </div>
            ) : currentTraders.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-muted-foreground">
                  {currentDiscoverMode === 'winrate'
                    ? t('walletTracker.noTradersWithWr', { wr: minWinRate })
                    : t('walletTracker.noTradersDiscovered')}
                </p>
                {currentDiscoverMode === 'winrate' && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {t('walletTracker.noTradersTip')}
                  </p>
                )}
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-sm text-muted-foreground">
                    {currentDiscoverMode === 'winrate'
                      ? t('walletTracker.foundTradersWithWr', { count: currentTraders.length, wr: minWinRate })
                      : t('walletTracker.foundTraders', { count: currentTraders.length })}
                  </span>
                  {currentDiscoverMode === 'winrate' && currentTraders.length > 0 && (
                    <span className="text-xs text-muted-foreground">
                      {t('walletTracker.avgWinRate', {
                        wr: (currentTraders.reduce((sum, tr) => sum + (tr.win_rate || 0), 0) / currentTraders.length).toFixed(1),
                      })}
                    </span>
                  )}
                </div>
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {currentTraders.map((trader, idx) => (
                  <div
                    key={trader.address}
                    className="flex items-center justify-between p-3 rounded-lg transition-colors bg-muted hover:bg-accent"
                  >
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold",
                        currentDiscoverMode === 'winrate' ? "bg-emerald-500/20" : "bg-muted"
                      )}>
                        #{trader.rank || idx + 1}
                      </div>
                      <div>
                        <p className="font-medium text-sm">
                          {trader.username || `${trader.address.slice(0, 6)}...${trader.address.slice(-4)}`}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {trader.win_rate !== undefined && (
                            <span className={cn(
                              "mr-2 font-medium",
                              trader.win_rate >= 80 ? "text-emerald-400" :
                              trader.win_rate >= 60 ? "text-green-400" :
                              trader.win_rate >= 50 ? "text-yellow-400" : "text-red-400"
                            )}>
                              {t('walletTracker.wrShort', { wr: trader.win_rate.toFixed(1) })}
                            </span>
                          )}
                          {trader.wins !== undefined && trader.losses !== undefined && (
                            <span className="text-muted-foreground mr-2">
                              ({trader.wins}{t('walletTracker.winsShort')}/{trader.losses}{t('walletTracker.lossesShort')})
                            </span>
                          )}
                          {t('walletTracker.volShort', { vol: `$${trader.volume.toLocaleString(undefined, { maximumFractionDigits: 0 })}` })}
                          {trader.pnl !== undefined && (
                            <span className={trader.pnl >= 0 ? 'text-green-400 ml-2' : 'text-red-400 ml-2'}>
                              {trader.pnl >= 0 ? '+' : ''}${trader.pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })} {t('walletTracker.pnlLabel')}
                            </span>
                          )}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        onClick={() => handleAnalyze(trader.address, trader.username)}
                        className="h-auto gap-1 px-2 py-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded text-xs"
                      >
                        <Activity className="w-3 h-3" />
                        {t('walletTracker.analyze')}
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => handleTrackOnly(trader.address)}
                        disabled={trackWalletMutation.isPending}
                        className="h-auto gap-1 px-2 py-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded text-xs"
                      >
                        <UserPlus className="w-3 h-3" />
                        {t('walletTracker.track')}
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => openAddToBotDialog(trader.address, trader.username || null)}
                        className="h-auto gap-1 px-2 py-1 bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 rounded text-xs"
                      >
                        <Bot className="w-3 h-3" />
                        {t('walletTracker.addToBot')}
                      </Button>
                      <a
                        href={`https://polymarket.com/profile/${trader.address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1 hover:bg-accent rounded"
                        title={t('walletTracker.viewOnPolymarket')}
                      >
                        <ExternalLink className="w-3 h-3 text-muted-foreground" />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
              </>
            )}
          </Card>
        </>
      )}

      {currentSection === 'tracked' && (
        <>
          {/* Add Wallet Form */}
          <Card className="p-4">
            <h3 className="text-lg font-medium mb-4">{t('walletTracker.trackWalletHeading')}</h3>
            <div className="flex gap-3">
              <Input
                type="text"
                value={newAddress}
                onChange={(e) => setNewAddress(e.target.value)}
                placeholder={t('walletTracker.addressPlaceholder')}
                className="flex-1 bg-muted rounded-lg"
              />
              <Input
                type="text"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder={t('walletTracker.labelPlaceholder')}
                className="w-48 bg-muted rounded-lg"
              />
              <Button
                onClick={handleAdd}
                disabled={addMutation.isPending || !newAddress.trim()}
                className={cn(
                  "h-auto gap-2 px-4 py-2 rounded-lg font-medium text-sm",
                  "bg-blue-500 hover:bg-blue-600 transition-colors",
                  (addMutation.isPending || !newAddress.trim()) && "opacity-50 cursor-not-allowed"
                )}
              >
                <Plus className="w-4 h-4" />
                {t('walletTracker.add')}
              </Button>
            </div>
          </Card>

          {/* Tracked Wallets */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">{t('walletTracker.trackedWalletsHeading', { count: wallets.length })}</h3>
              <p className="text-xs text-muted-foreground">
                {fallbackWalletMetricsLoading
                  ? t('walletTracker.loadingWalletStats')
                  : trackedPoolMembersLoading
                    ? t('walletTracker.poolMetricsLoading', { live: fallbackMetricCount })
                  : trackedPoolMembersError
                    ? t('walletTracker.poolMetricsUnavailable')
                    : t('walletTracker.poolMetricsSummary', {
                        pool: trackedPoolMembers.length,
                        total: wallets.length,
                        live: fallbackMetricCount,
                      })}
              </p>
            </div>

            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
              </div>
            ) : wallets.length === 0 ? (
              <Card className="text-center py-12">
                <Wallet className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">{t('walletTracker.noWalletsTracked')}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {t('walletTracker.noWalletsTrackedHint')}
                </p>
              </Card>
            ) : (
              <Card className="border-border overflow-hidden">
                <div className="max-h-[620px] overflow-auto bg-background/20">
                  <Table className="text-[11px] leading-tight">
                    <TableHeader className="sticky top-0 z-10 bg-background/85 backdrop-blur-sm">
                      <TableRow className="bg-muted/55 border-b border-border/80">
                        <TableHead className="h-9 px-2 min-w-[210px]">{t('walletTracker.colTrader')}</TableHead>
                        <TableHead className="h-9 px-2 min-w-[220px]">{t('walletTracker.colPerformance')}</TableHead>
                        <TableHead className="h-9 px-2 min-w-[190px]">{t('walletTracker.colSelection')}</TableHead>
                        <TableHead className="h-9 px-2 min-w-[220px]">{t('walletTracker.colWhySelected')}</TableHead>
                        <TableHead className="h-9 px-2 min-w-[130px]">{t('walletTracker.colFlags')}</TableHead>
                        <TableHead className="h-9 px-2 min-w-[140px]">{t('walletTracker.colActions')}</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {trackedWalletRows.map(({ wallet, member }, rowIndex) => {
                        const displayName = member?.display_name || walletDisplayName(wallet, unknownLabel)
                        const username = member?.username || wallet.username || null
                        const fallbackMetrics = fallbackWalletMetrics[wallet.address.toLowerCase()]
                        const hasFallbackMetrics = Boolean(
                          fallbackMetrics
                          && (
                            fallbackMetrics.totalPnl != null
                            || fallbackMetrics.winRate != null
                            || fallbackMetrics.totalTrades != null
                            || fallbackMetrics.openPositions != null
                          ),
                        )
                        const reasons = member ? selectionReasons(member, t) : []
                        const flags = member?.pool_flags || { manual_include: false, manual_exclude: false, blacklisted: false }
                        const selectionValue = Number(member?.selection_score ?? member?.composite_score ?? 0)
                        const compositeValue = Number(member?.composite_score ?? 0)
                        const selectionDelta = Math.abs(selectionValue - compositeValue)
                        const showCompositeScore = selectionDelta >= SCORE_DELTA_HIDE_THRESHOLD
                        const percentile = member?.selection_percentile != null
                          ? `${(Number(member.selection_percentile) * 100).toFixed(1)}%`
                          : null
                        const breakdown = (member?.selection_breakdown || {}) as Record<string, number>
                        const qualityScore = Number(breakdown.quality_score ?? member?.quality_score ?? 0)
                        const activityScore = Number(breakdown.activity_score ?? member?.activity_score ?? 0)
                        const stabilityScore = Number(breakdown.stability_score ?? member?.stability_score ?? 0)
                        const insiderScore = Number(breakdown.insider_score ?? 0)
                        const selectionSparkline = [
                          { key: 'quality', label: t('walletTracker.scoreQuality'), value: qualityScore, tone: scoreTone(qualityScore, 0.7, 0.45) },
                          { key: 'activity', label: t('walletTracker.scoreActivity'), value: activityScore, tone: scoreTone(activityScore, 0.55, 0.3) },
                          { key: 'stability', label: t('walletTracker.scoreStability'), value: stabilityScore, tone: scoreTone(stabilityScore, 0.65, 0.45) },
                          { key: 'insider', label: t('walletTracker.scoreInsider'), value: insiderScore, tone: inverseScoreTone(insiderScore, 0.72, 0.6) },
                        ]
                        const hasSparkline = member ? selectionSparkline.some((point) => point.value > 0) : false
                        const rowHighlight = rowIndex % 2 === 0 ? 'bg-background/30' : ''

                        return (
                          <TableRow
                            key={wallet.address}
                            className={cn('border-border/70 transition-colors hover:bg-muted/40', rowHighlight)}
                          >
                            <TableCell className="px-2 py-1.5 align-middle">
                              <div className="min-w-0 space-y-0.5">
                                <button
                                  type="button"
                                  onClick={() => handleAnalyze(wallet.address, username || undefined)}
                                  className="max-w-full text-left hover:opacity-90"
                                >
                                  <p className="truncate text-[12px] font-semibold text-foreground">{displayName}</p>
                                </button>
                                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                                  {username && (
                                    <span className="max-w-[120px] truncate" title={`@${username}`}>@{username}</span>
                                  )}
                                  <span className="font-mono">{shortAddress(wallet.address, unknownLabel)}</span>
                                </div>
                                {wallet.label && wallet.label !== displayName && (
                                  <div className="truncate text-[10px] text-muted-foreground" title={wallet.label}>
                                    {t('walletTracker.labelPrefix', { label: wallet.label })}
                                  </div>
                                )}
                              </div>
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              {member ? (
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center gap-1">
                                    <PnlDisplay value={member.total_pnl || 0} className="text-xs" />
                                    <MetricPill
                                      label={t('walletTracker.pillWr')}
                                      value={formatWinRate(member.win_rate || 0)}
                                      tone={scoreTone(normalizePercentRatio(member.win_rate || 0) / 100, 0.6, 0.45)}
                                    />
                                    <MetricPill label={t('walletTracker.pillTrades')} value={formatNumber(member.total_trades || 0)} />
                                  </div>
                                  <div className="flex flex-wrap items-center gap-1">
                                    <MetricPill
                                      label={t('walletTracker.pill24h')}
                                      value={formatNumber(member.trades_24h || 0)}
                                      tone={scoreTone(clamp01((member.trades_24h || 0) / 12), 0.6, 0.25)}
                                    />
                                    <MetricPill
                                      label={t('walletTracker.pill1h')}
                                      value={formatNumber(member.trades_1h || 0)}
                                      tone={scoreTone(clamp01((member.trades_1h || 0) / 4), 0.55, 0.25)}
                                    />
                                    <span className="text-[10px] text-muted-foreground">{timeAgo(member.last_trade_at || null, t)}</span>
                                  </div>
                                </div>
                              ) : (
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center gap-1">
                                    {fallbackMetrics?.totalPnl != null && (
                                      <PnlDisplay value={fallbackMetrics.totalPnl} className="text-xs" />
                                    )}
                                    <MetricPill
                                      label={t('walletTracker.pillWr')}
                                      value={fallbackMetrics?.winRate != null ? `${fallbackMetrics.winRate.toFixed(1)}%` : '--'}
                                      tone={
                                        fallbackMetrics?.winRate != null
                                          ? scoreTone(clamp01(fallbackMetrics.winRate / 100), 0.6, 0.45)
                                          : 'neutral'
                                      }
                                    />
                                    <MetricPill
                                      label={t('walletTracker.pillTrades')}
                                      value={fallbackMetrics?.totalTrades != null ? formatNumber(fallbackMetrics.totalTrades) : '--'}
                                    />
                                    <MetricPill
                                      label={t('walletTracker.pillPositions')}
                                      value={fallbackMetrics?.openPositions != null ? formatNumber(fallbackMetrics.openPositions) : '--'}
                                    />
                                  </div>
                                  <span className="text-[10px] text-muted-foreground">
                                    {hasFallbackMetrics
                                      ? t('walletTracker.liveWalletStats')
                                      : t('walletTracker.walletStatsUnavailable')}
                                  </span>
                                </div>
                              )}
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              {member ? (
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center gap-1">
                                    <MetricPill
                                      label={showCompositeScore ? t('walletTracker.pillSel') : t('walletTracker.pillSelCmp')}
                                      value={formatScorePct(selectionValue)}
                                      tone={scoreTone(selectionValue, 0.7, 0.5)}
                                    />
                                    {showCompositeScore && (
                                      <MetricPill
                                        label={t('walletTracker.pillCmp')}
                                        value={formatScorePct(compositeValue)}
                                        tone={scoreTone(compositeValue, 0.7, 0.5)}
                                      />
                                    )}
                                    {percentile && <MetricPill label={t('walletTracker.pillTop')} value={percentile} tone="info" />}
                                  </div>
                                  {hasSparkline && <ScoreSparkline points={selectionSparkline} />}
                                </div>
                              ) : (
                                <span className="text-[10px] text-muted-foreground">{t('walletTracker.noSelectionScore')}</span>
                              )}
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              {member ? (
                                <div className="flex flex-wrap gap-1">
                                  {reasons.slice(0, 2).map((reason) => (
                                    <span
                                      key={`${wallet.address}-${reason.code}`}
                                      title={reason.detail}
                                      className={cn(
                                        'inline-flex max-w-[180px] items-center rounded-full border px-2 py-0.5 text-[10px] font-medium leading-none truncate',
                                        METRIC_TONE_CLASSES[selectionReasonTone(reason.code)],
                                      )}
                                    >
                                      {reason.label}
                                    </span>
                                  ))}
                                  {reasons.length > 2 && (
                                    <span className="text-[10px] text-muted-foreground">+{reasons.length - 2}</span>
                                  )}
                                </div>
                              ) : (
                                <span className="text-[10px] text-muted-foreground">{t('walletTracker.awaitingAnalysis')}</span>
                              )}
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              <div className="flex flex-wrap gap-1">
                                <MetricPill label={t('walletTracker.flagTracked')} value={t('walletTracker.flagYes')} tone="info" mono={false} />
                                {member?.in_top_pool && (
                                  <MetricPill
                                    label={t('walletTracker.flagPool')}
                                    value={member.pool_tier ? member.pool_tier.toUpperCase() : t('walletTracker.flagPoolTop')}
                                    tone="good"
                                    mono={false}
                                  />
                                )}
                                {flags.manual_include && <MetricPill label={t('walletTracker.flagManualPlus')} value={t('walletTracker.flagOn')} tone="good" mono={false} />}
                                {flags.manual_exclude && <MetricPill label={t('walletTracker.flagManualMinus')} value={t('walletTracker.flagOn')} tone="warn" mono={false} />}
                                {flags.blacklisted && <MetricPill label={t('walletTracker.flagBlacklist')} value={t('walletTracker.flagOn')} tone="bad" mono={false} />}
                              </div>
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              <div className="flex flex-wrap gap-1">
                                <button
                                  onClick={() => handleAnalyze(wallet.address, username || undefined)}
                                  className="p-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                                  title={t('walletTracker.tooltipAnalyzeWallet')}
                                >
                                  <Activity className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  onClick={() => openAddToBotDialog(wallet.address, displayName)}
                                  className="p-1 rounded bg-sky-500/10 text-sky-300 hover:bg-sky-500/20 transition-colors"
                                  title={t('walletTracker.tooltipAddToBot')}
                                >
                                  <Bot className="w-3.5 h-3.5" />
                                </button>
                                <a
                                  href={`https://polymarket.com/profile/${wallet.address}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
                                  title={t('walletTracker.viewOnPolymarket')}
                                >
                                  <ExternalLink className="w-3.5 h-3.5" />
                                </a>
                                <button
                                  onClick={() => removeMutation.mutate(wallet.address)}
                                  disabled={removeMutation.isPending}
                                  className="p-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors disabled:opacity-50"
                                  title={t('walletTracker.tooltipRemoveWallet')}
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </div>
                            </TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </div>
              </Card>
            )}
          </div>

          {/* Trader Group + List Management */}
          {showManagementPanel && (
            <div className="mt-6">
              <RecentTradesPanel
                mode="management"
                onNavigateToWallet={(address) => {
                  if (onNavigateToWallet) {
                    onNavigateToWallet(address)
                  } else if (onAnalyzeWallet) {
                    onAnalyzeWallet(address)
                  }
                }}
              />
            </div>
          )}
        </>
      )}
      <AddWalletToBotDialog
        open={Boolean(addToBotWallet)}
        walletAddress={addToBotWallet?.address || null}
        walletLabel={addToBotWallet?.label || null}
        onOpenChange={(open) => {
          if (!open) setAddToBotWallet(null)
        }}
        onAdded={handleAddToBotSuccess}
      />

    </div>
  )
}
