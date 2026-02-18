import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  Trash2,
  Wallet,
  ExternalLink,
  RefreshCw,
  Star,
  Copy,
  UserPlus,
  Activity,
  Filter,
  Search,
  Trophy,
  Target,
  X,
  DollarSign,
  FileText
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
  getSimulationAccounts,
  SimulationAccount,
  TimePeriod,
  OrderBy,
  Category,
  DiscoveredTrader
} from '../services/api'

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

function shortAddress(address: string): string {
  if (!address) return 'unknown'
  if (address.length <= 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function walletDisplayName(wallet: WalletType): string {
  return wallet.username || wallet.label || shortAddress(wallet.address)
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

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return 'Never'
  const diff = Date.now() - new Date(dateStr).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
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

function reasonLabelFromCode(code: string): string {
  const clean = String(code || '').trim()
  if (!clean) return 'Selection detail'
  return clean
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function selectionReasons(member: PoolMember): Array<{ code: string; label: string; detail?: string }> {
  const raw = Array.isArray(member.selection_reasons) ? member.selection_reasons : []
  const normalized: Array<{ code: string; label: string; detail?: string }> = []
  for (const item of raw) {
    const code = String(item?.code || '').trim()
    const label = String(item?.label || '').trim()
    const detail = typeof item?.detail === 'string' ? item.detail.trim() : undefined
    if (!code && !label) continue
    normalized.push({
      code: code || `reason_${label.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
      label: label || reasonLabelFromCode(code),
      detail,
    })
  }

  if (normalized.length > 0) return normalized
  if (member.pool_membership_reason) {
    return [{
      code: member.pool_membership_reason,
      label: reasonLabelFromCode(member.pool_membership_reason),
    }]
  }
  if (member.tracked_wallet) {
    return [{
      code: 'tracked_wallet',
      label: 'Tracked wallet',
      detail: 'Included from tracked-wallet workflows.',
    }]
  }
  return [{
    code: 'selection_reason_unknown',
    label: 'Selection reason unavailable',
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
  const [newAddress, setNewAddress] = useState('')
  const [newLabel, setNewLabel] = useState('')
  const [activeSection, setActiveSection] = useState<'tracked' | 'discover'>('discover')
  const [discoverModeState, setDiscoverMode] = useState<'leaderboard' | 'winrate'>('leaderboard')
  const [showFilters, setShowFilters] = useState(false)

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

  // Copy trade modal state
  const [showCopyModal, setShowCopyModal] = useState(false)
  const [selectedTrader, setSelectedTrader] = useState<DiscoveredTrader | null>(null)
  const [selectedAccountId, setSelectedAccountId] = useState<string>('')

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

  const { data: simAccounts = [] } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
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

  const trackAndCopyMutation = useMutation({
    mutationFn: (params: { address: string; label?: string; auto_copy?: boolean; simulation_account_id?: string }) =>
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
    const trader = allTraders.find(t => t.address === address)
    const winRateStr = trader?.win_rate ? ` | ${trader.win_rate.toFixed(1)}% WR` : ''
    const label = `Discovered Trader (${trader?.volume?.toFixed(0) || '?'} vol${winRateStr})`
    trackAndCopyMutation.mutate({
      address,
      label,
      auto_copy: false,
    })
  }

  const handleOpenCopyModal = (address: string) => {
    const allTraders = currentDiscoverMode === 'winrate' ? winRateTraders : discoveredTraders
    const trader = allTraders.find(t => t.address === address)
    if (trader) {
      setSelectedTrader(trader)
      setSelectedAccountId(simAccounts.length > 0 ? simAccounts[0].id : '')
      setShowCopyModal(true)
    }
  }

  const handleCopyTradeConfirm = async (usePaper: boolean) => {
    if (!selectedTrader) return

    const winRateStr = selectedTrader?.win_rate ? ` | ${selectedTrader.win_rate.toFixed(1)}% WR` : ''
    const label = `Discovered Trader (${selectedTrader?.volume?.toFixed(0) || '?'} vol${winRateStr})`

    if (usePaper && selectedAccountId) {
      // Sandbox mode: track and copy to simulation account
      trackAndCopyMutation.mutate({
        address: selectedTrader.address,
        label,
        auto_copy: true,
        simulation_account_id: selectedAccountId
      })
    } else if (!usePaper) {
      // Live mode: just track for now (user can set up live copy trading separately)
      // For now we track the wallet - live copy trading would require additional implementation
      trackAndCopyMutation.mutate({
        address: selectedTrader.address,
        label,
        auto_copy: false,
      })
      // TODO: Could integrate with live trading system in the future
    }

    setShowCopyModal(false)
    setSelectedTrader(null)
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

      return walletDisplayName(a.wallet).localeCompare(walletDisplayName(b.wallet), undefined, {
        sensitivity: 'base',
      })
    })

    return rows
  }, [wallets, trackedPoolMemberMap])

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
              Discover Top Traders
            </TabsTrigger>
            <TabsTrigger
              value="tracked"
              className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400 data-[state=active]:border data-[state=active]:border-blue-500/50 data-[state=active]:shadow-none"
            >
              <Wallet className="w-4 h-4" />
              Tracked Wallets ({wallets.length})
            </TabsTrigger>
          </TabsList>
        </Tabs>
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
                  Leaderboard
                </TabsTrigger>
                <TabsTrigger
                  value="winrate"
                  className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-400 data-[state=active]:border data-[state=active]:border-emerald-500/50 data-[state=active]:shadow-none"
                >
                  <Target className="w-4 h-4" />
                  High Win Rate
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
                      Discover Traders
                    </>
                  ) : (
                    <>
                      <Star className="w-5 h-5 text-yellow-500" />
                      Top Active Traders
                    </>
                  )}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {currentDiscoverMode === 'winrate'
                    ? `Scanning ~${scanCount * 2} traders for ${minWinRate}%+ win rate${minVolume > 0 ? `, $${minVolume.toLocaleString()}+ volume` : ''}`
                    : 'Discovered from Polymarket leaderboard'}
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
                  Filters
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => refreshCurrentTraders()}
                  disabled={isLoadingTraders}
                  className="h-auto gap-2 px-3 py-1.5 bg-muted rounded-lg text-sm hover:bg-accent"
                >
                  <RefreshCw className={cn("w-4 h-4", isLoadingTraders && "animate-spin")} />
                  Refresh
                </Button>
              </div>
            </div>

            {/* Filters Panel */}
            {showFilters && (
              <div className="mb-4 p-3 bg-muted rounded-lg space-y-3">
                {/* Row 1: Basic filters */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">Time Period</label>
                    <select
                      value={timePeriod}
                      onChange={(e) => setTimePeriod(e.target.value as TimePeriod)}
                      className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                    >
                      <option value="ALL">All Time</option>
                      <option value="MONTH">Last 30 Days</option>
                      <option value="WEEK">Last 7 Days</option>
                      <option value="DAY">Last 24 Hours</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">Category</label>
                    <select
                      value={category}
                      onChange={(e) => setCategory(e.target.value as Category)}
                      className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                    >
                      <option value="OVERALL">All Categories</option>
                      <option value="POLITICS">Politics</option>
                      <option value="SPORTS">Sports</option>
                      <option value="CRYPTO">Crypto</option>
                      <option value="CULTURE">Culture</option>
                      <option value="ECONOMICS">Economics</option>
                      <option value="TECH">Tech</option>
                      <option value="FINANCE">Finance</option>
                    </select>
                  </div>
                  {currentDiscoverMode === 'leaderboard' && (
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">Sort By</label>
                      <select
                        value={orderBy}
                        onChange={(e) => setOrderBy(e.target.value as OrderBy)}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value="PNL">Profit/Loss</option>
                        <option value="VOL">Volume</option>
                      </select>
                    </div>
                  )}
                  {currentDiscoverMode === 'winrate' && (
                    <>
                      <div>
                        <label className="block text-xs text-muted-foreground mb-1">Min Win Rate</label>
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
                        <label className="block text-xs text-muted-foreground mb-1">Min Trades</label>
                        <select
                          value={minTrades}
                          onChange={(e) => setMinTrades(Number(e.target.value))}
                          className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                        >
                          <option value={3}>3+ trades</option>
                          <option value={5}>5+ trades</option>
                          <option value={10}>10+ trades</option>
                          <option value={20}>20+ trades</option>
                          <option value={50}>50+ trades</option>
                          <option value={100}>100+ trades</option>
                          <option value={200}>200+ trades</option>
                          <option value={500}>500+ trades</option>
                          <option value={1000}>1000+ trades</option>
                        </select>
                      </div>
                    </>
                  )}
                </div>

                {/* Row 2: Advanced win rate filters */}
                {currentDiscoverMode === 'winrate' && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-border">
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">Min Volume ($)</label>
                      <select
                        value={minVolume}
                        onChange={(e) => setMinVolume(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={0}>No minimum</option>
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
                      <label className="block text-xs text-muted-foreground mb-1">Max Volume ($)</label>
                      <select
                        value={maxVolume}
                        onChange={(e) => setMaxVolume(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={0}>No maximum</option>
                        <option value={10000}>$10,000</option>
                        <option value={50000}>$50,000</option>
                        <option value={100000}>$100,000</option>
                        <option value={500000}>$500,000</option>
                        <option value={1000000}>$1,000,000</option>
                        <option value={5000000}>$5,000,000</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">Scan Depth</label>
                      <select
                        value={scanCount}
                        onChange={(e) => setScanCount(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={200}>200 (fast)</option>
                        <option value={500}>500 (default)</option>
                        <option value={750}>750 (deep)</option>
                        <option value={1000}>1000 (max)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-muted-foreground mb-1">Show Results</label>
                      <select
                        value={resultLimit}
                        onChange={(e) => setResultLimit(Number(e.target.value))}
                        className="w-full bg-[#222] border border-border rounded px-2 py-1.5 text-sm"
                      >
                        <option value={25}>25 results</option>
                        <option value={50}>50 results</option>
                        <option value={100}>100 results</option>
                        <option value={200}>200 results</option>
                        <option value={500}>500 results</option>
                      </select>
                    </div>
                  </div>
                )}

                {/* Tip for high win rate searches */}
                {currentDiscoverMode === 'winrate' && minWinRate >= 95 && (
                  <div className="text-xs text-yellow-500/80 flex items-center gap-1 pt-1">
                    <span>Tip: For 95%+ win rates, set Scan Depth to 1000 (max). Scans both PNL and VOL leaderboards (~2x depth).</span>
                  </div>
                )}
              </div>
            )}

            {isLoadingTraders ? (
              <div className="flex flex-col items-center justify-center py-8">
                <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
                <span className="mt-2 text-muted-foreground">
                  {currentDiscoverMode === 'winrate'
                    ? `Scanning ~${scanCount * 2} traders from PNL + VOL leaderboards for ${minWinRate}%+ win rate...`
                    : 'Verifying trader activity across leaderboard...'}
                </span>
                {currentDiscoverMode === 'winrate' && (
                  <span className="text-xs text-muted-foreground mt-1">
                    Analyzing closed positions for each trader (this is fast)
                  </span>
                )}
              </div>
            ) : currentTraders.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-muted-foreground">
                  {currentDiscoverMode === 'winrate'
                    ? `No traders found with ${minWinRate}%+ win rate.`
                    : 'No traders discovered yet'}
                </p>
                {currentDiscoverMode === 'winrate' && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Try: Lower the win rate threshold, increase scan count, or reduce min trades/volume filters
                  </p>
                )}
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-sm text-muted-foreground">
                    Found {currentTraders.length} trader{currentTraders.length !== 1 ? 's' : ''}
                    {currentDiscoverMode === 'winrate' && ` with ${minWinRate}%+ win rate`}
                  </span>
                  {currentDiscoverMode === 'winrate' && currentTraders.length > 0 && (
                    <span className="text-xs text-muted-foreground">
                      Avg: {(currentTraders.reduce((sum, t) => sum + (t.win_rate || 0), 0) / currentTraders.length).toFixed(1)}% WR
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
                              {trader.win_rate.toFixed(1)}% WR
                            </span>
                          )}
                          {trader.wins !== undefined && trader.losses !== undefined && (
                            <span className="text-muted-foreground mr-2">
                              ({trader.wins}W/{trader.losses}L)
                            </span>
                          )}
                          ${trader.volume.toLocaleString(undefined, { maximumFractionDigits: 0 })} vol
                          {trader.pnl !== undefined && (
                            <span className={trader.pnl >= 0 ? 'text-green-400 ml-2' : 'text-red-400 ml-2'}>
                              {trader.pnl >= 0 ? '+' : ''}${trader.pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })} P/L
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
                        Analyze
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => handleTrackOnly(trader.address)}
                        disabled={trackAndCopyMutation.isPending}
                        className="h-auto gap-1 px-2 py-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded text-xs"
                      >
                        <UserPlus className="w-3 h-3" />
                        Track
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => handleOpenCopyModal(trader.address)}
                        disabled={trackAndCopyMutation.isPending}
                        title="Track and copy trades"
                        className="h-auto gap-1 px-2 py-1 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded text-xs disabled:opacity-50"
                      >
                        <Copy className="w-3 h-3" />
                        Copy Trade
                      </Button>
                      <a
                        href={`https://polymarket.com/profile/${trader.address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1 hover:bg-accent rounded"
                        title="View on Polymarket"
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
            <h3 className="text-lg font-medium mb-4">Track a Wallet</h3>
            <div className="flex gap-3">
              <Input
                type="text"
                value={newAddress}
                onChange={(e) => setNewAddress(e.target.value)}
                placeholder="Wallet address (0x...)"
                className="flex-1 bg-muted rounded-lg"
              />
              <Input
                type="text"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder="Label (optional)"
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
                Add
              </Button>
            </div>
          </Card>

          {/* Tracked Wallets */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Tracked Wallets ({wallets.length})</h3>
              <p className="text-xs text-muted-foreground">
                {fallbackWalletMetricsLoading
                  ? 'Loading wallet stats...'
                  : trackedPoolMembersLoading
                    ? `Pool metrics still loading · Live stats ${fallbackMetricCount}`
                  : trackedPoolMembersError
                    ? 'Pool metrics unavailable'
                    : `Pool metrics ${trackedPoolMembers.length}/${wallets.length} · Live stats ${fallbackMetricCount}`}
              </p>
            </div>

            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
              </div>
            ) : wallets.length === 0 ? (
              <Card className="text-center py-12">
                <Wallet className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No wallets being tracked</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Use the Discover tab to find top traders, or add a wallet manually
                </p>
              </Card>
            ) : (
              <Card className="border-border overflow-hidden">
                <div className="max-h-[620px] overflow-auto bg-background/20">
                  <Table className="text-[11px] leading-tight">
                    <TableHeader className="sticky top-0 z-10 bg-background/85 backdrop-blur-sm">
                      <TableRow className="bg-muted/55 border-b border-border/80">
                        <TableHead className="h-9 px-2 min-w-[210px]">Trader</TableHead>
                        <TableHead className="h-9 px-2 min-w-[220px]">Performance</TableHead>
                        <TableHead className="h-9 px-2 min-w-[190px]">Selection</TableHead>
                        <TableHead className="h-9 px-2 min-w-[220px]">Why Selected</TableHead>
                        <TableHead className="h-9 px-2 min-w-[130px]">Flags</TableHead>
                        <TableHead className="h-9 px-2 min-w-[140px]">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {trackedWalletRows.map(({ wallet, member }, rowIndex) => {
                        const displayName = member?.display_name || walletDisplayName(wallet)
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
                        const reasons = member ? selectionReasons(member) : []
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
                          { key: 'quality', label: 'Quality', value: qualityScore, tone: scoreTone(qualityScore, 0.7, 0.45) },
                          { key: 'activity', label: 'Activity', value: activityScore, tone: scoreTone(activityScore, 0.55, 0.3) },
                          { key: 'stability', label: 'Stability', value: stabilityScore, tone: scoreTone(stabilityScore, 0.65, 0.45) },
                          { key: 'insider', label: 'Insider', value: insiderScore, tone: inverseScoreTone(insiderScore, 0.72, 0.6) },
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
                                  <span className="font-mono">{shortAddress(wallet.address)}</span>
                                </div>
                                {wallet.label && wallet.label !== displayName && (
                                  <div className="truncate text-[10px] text-muted-foreground" title={wallet.label}>
                                    Label: {wallet.label}
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
                                      label="WR"
                                      value={formatWinRate(member.win_rate || 0)}
                                      tone={scoreTone(normalizePercentRatio(member.win_rate || 0) / 100, 0.6, 0.45)}
                                    />
                                    <MetricPill label="T" value={formatNumber(member.total_trades || 0)} />
                                  </div>
                                  <div className="flex flex-wrap items-center gap-1">
                                    <MetricPill
                                      label="24h"
                                      value={formatNumber(member.trades_24h || 0)}
                                      tone={scoreTone(clamp01((member.trades_24h || 0) / 12), 0.6, 0.25)}
                                    />
                                    <MetricPill
                                      label="1h"
                                      value={formatNumber(member.trades_1h || 0)}
                                      tone={scoreTone(clamp01((member.trades_1h || 0) / 4), 0.55, 0.25)}
                                    />
                                    <span className="text-[10px] text-muted-foreground">{timeAgo(member.last_trade_at || null)}</span>
                                  </div>
                                </div>
                              ) : (
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center gap-1">
                                    {fallbackMetrics?.totalPnl != null && (
                                      <PnlDisplay value={fallbackMetrics.totalPnl} className="text-xs" />
                                    )}
                                    <MetricPill
                                      label="WR"
                                      value={fallbackMetrics?.winRate != null ? `${fallbackMetrics.winRate.toFixed(1)}%` : '--'}
                                      tone={
                                        fallbackMetrics?.winRate != null
                                          ? scoreTone(clamp01(fallbackMetrics.winRate / 100), 0.6, 0.45)
                                          : 'neutral'
                                      }
                                    />
                                    <MetricPill
                                      label="T"
                                      value={fallbackMetrics?.totalTrades != null ? formatNumber(fallbackMetrics.totalTrades) : '--'}
                                    />
                                    <MetricPill
                                      label="Pos"
                                      value={fallbackMetrics?.openPositions != null ? formatNumber(fallbackMetrics.openPositions) : '--'}
                                    />
                                  </div>
                                  <span className="text-[10px] text-muted-foreground">
                                    {hasFallbackMetrics
                                      ? 'Live wallet stats'
                                      : 'Wallet stats unavailable right now'}
                                  </span>
                                </div>
                              )}
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              {member ? (
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center gap-1">
                                    <MetricPill
                                      label={showCompositeScore ? 'Sel' : 'Sel/Cmp'}
                                      value={formatScorePct(selectionValue)}
                                      tone={scoreTone(selectionValue, 0.7, 0.5)}
                                    />
                                    {showCompositeScore && (
                                      <MetricPill
                                        label="Cmp"
                                        value={formatScorePct(compositeValue)}
                                        tone={scoreTone(compositeValue, 0.7, 0.5)}
                                      />
                                    )}
                                    {percentile && <MetricPill label="Top" value={percentile} tone="info" />}
                                  </div>
                                  {hasSparkline && <ScoreSparkline points={selectionSparkline} />}
                                </div>
                              ) : (
                                <span className="text-[10px] text-muted-foreground">No selection score yet</span>
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
                                <span className="text-[10px] text-muted-foreground">Awaiting analysis refresh</span>
                              )}
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              <div className="flex flex-wrap gap-1">
                                <MetricPill label="Tracked" value="Yes" tone="info" mono={false} />
                                {member?.in_top_pool && (
                                  <MetricPill
                                    label="Pool"
                                    value={member.pool_tier ? member.pool_tier.toUpperCase() : 'TOP'}
                                    tone="good"
                                    mono={false}
                                  />
                                )}
                                {flags.manual_include && <MetricPill label="Manual+" value="On" tone="good" mono={false} />}
                                {flags.manual_exclude && <MetricPill label="Manual-" value="On" tone="warn" mono={false} />}
                                {flags.blacklisted && <MetricPill label="BL" value="On" tone="bad" mono={false} />}
                              </div>
                            </TableCell>

                            <TableCell className="px-2 py-1.5 align-middle">
                              <div className="flex flex-wrap gap-1">
                                <button
                                  onClick={() => handleAnalyze(wallet.address, username || undefined)}
                                  className="p-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                                  title="Analyze wallet"
                                >
                                  <Activity className="w-3.5 h-3.5" />
                                </button>
                                <a
                                  href={`https://polymarket.com/profile/${wallet.address}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
                                  title="View on Polymarket"
                                >
                                  <ExternalLink className="w-3.5 h-3.5" />
                                </a>
                                <button
                                  onClick={() => removeMutation.mutate(wallet.address)}
                                  disabled={removeMutation.isPending}
                                  className="p-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors disabled:opacity-50"
                                  title="Remove wallet"
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

      {/* Copy Trade Account Selection Modal */}
      {showCopyModal && selectedTrader && (
        <div className="fixed inset-0 bg-background/80 flex items-center justify-center z-50">
          <div className="bg-muted border border-border rounded-xl w-full max-w-md mx-4 overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div>
                <h3 className="text-lg font-semibold">Copy Trade</h3>
                <p className="text-sm text-muted-foreground">
                  {selectedTrader.username || `${selectedTrader.address.slice(0, 6)}...${selectedTrader.address.slice(-4)}`}
                </p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setShowCopyModal(false)
                  setSelectedTrader(null)
                }}
                className="h-auto p-2 hover:bg-accent rounded-lg"
              >
                <X className="w-5 h-5 text-muted-foreground" />
              </Button>
            </div>

            {/* Modal Content */}
            <div className="p-4 space-y-4">
              <p className="text-sm text-foreground/80">
                Choose how you want to copy trades from this trader:
              </p>

              {/* Sandbox Trading Option */}
              <div className="space-y-3">
                <div
                  className={cn(
                    "p-4 rounded-lg border-2 cursor-pointer transition-all",
                    simAccounts.length > 0
                      ? "border-amber-500/50 bg-amber-500/10 hover:bg-amber-500/20"
                      : "border-border bg-muted/50 opacity-60 cursor-not-allowed"
                  )}
                  onClick={() => simAccounts.length > 0 && handleCopyTradeConfirm(true)}
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-amber-400" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-amber-400">Sandbox Trading</h4>
                      <p className="text-sm text-muted-foreground mt-1">
                        Copy trades to a simulation account with virtual money. Safe for testing.
                      </p>
                      {simAccounts.length > 0 ? (
                        <div className="mt-3">
                          <label className="block text-xs text-muted-foreground mb-1">Select Account</label>
                          <select
                            value={selectedAccountId}
                            onChange={(e) => {
                              e.stopPropagation()
                              setSelectedAccountId(e.target.value)
                            }}
                            onClick={(e) => e.stopPropagation()}
                            className="w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-sm"
                          >
                            {simAccounts.map((account: SimulationAccount) => (
                              <option key={account.id} value={account.id}>
                                {account.name} (${account.current_capital.toLocaleString()})
                              </option>
                            ))}
                          </select>
                        </div>
                      ) : (
                        <p className="text-xs text-yellow-500 mt-2">
                          No sandbox accounts available. Create one in the Accounts tab first.
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Live Trading Option */}
                <div
                  className="p-4 rounded-lg border-2 border-green-500/50 bg-green-500/10 hover:bg-green-500/20 cursor-pointer transition-all"
                  onClick={() => handleCopyTradeConfirm(false)}
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center flex-shrink-0">
                      <DollarSign className="w-5 h-5 text-green-400" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-green-400">Live Trading</h4>
                      <p className="text-sm text-muted-foreground mt-1">
                        Track this trader and receive alerts for live copy trading. Uses real money.
                      </p>
                      <p className="text-xs text-yellow-500 mt-2">
                        Configure live copy trading in the Trading tab after tracking.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-border bg-card">
              <Button
                variant="ghost"
                onClick={() => {
                  setShowCopyModal(false)
                  setSelectedTrader(null)
                }}
                className="w-full h-auto py-2 bg-muted hover:bg-accent rounded-lg text-sm font-medium"
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
