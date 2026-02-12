import { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Trophy,
  RefreshCw,
  Users,
  Target,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Copy,
  Search,
  TrendingUp,
  ExternalLink,
  Activity,
  UserPlus,
  PauseCircle,
  AlertTriangle,
  Ban,
  UserCheck,
  UserX,
  Trash2,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from './ui/table'
import {
  discoveryApi,
  type DiscoveredWallet,
  type TagInfo,
  type DiscoveryStats,
  type PoolStats,
  type PoolMember,
  type PoolMembersResponse,
} from '../services/discoveryApi'
import { analyzeAndTrackWallet } from '../services/api'

type SortField =
  | 'rank_score'
  | 'composite_score'
  | 'quality_score'
  | 'activity_score'
  | 'insider_score'
  | 'last_trade_at'
  | 'total_pnl'
  | 'win_rate'
  | 'sharpe_ratio'
  | 'total_trades'
  | 'avg_roi'

type SortDir = 'asc' | 'desc'
type RecommendationFilter = '' | 'copy_candidate' | 'monitor' | 'avoid'
type TimePeriod = '24h' | '7d' | '30d' | '90d' | 'all'

const TIME_PERIODS: { value: TimePeriod; label: string; description: string }[] = [
  { value: '24h', label: '24H', description: 'last 24 hours' },
  { value: '7d', label: '7D', description: 'last 7 days' },
  { value: '30d', label: '1M', description: 'last 30 days' },
  { value: '90d', label: '3M', description: 'last 90 days' },
  { value: 'all', label: 'All', description: 'all time' },
]

const RECOMMENDATION_COLORS: Record<string, string> = {
  copy_candidate: 'bg-green-500/15 text-green-400 border-green-500/20',
  monitor: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20',
  avoid: 'bg-red-500/15 text-red-400 border-red-500/20',
}

const RECOMMENDATION_LABELS: Record<string, string> = {
  copy_candidate: 'Copy Candidate',
  monitor: 'Monitor',
  avoid: 'Avoid',
}

const ITEMS_PER_PAGE = 25

function formatPnl(value: number): string {
  if (value == null) return '0.00'
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return value.toFixed(2)
}

function formatPercent(value: number): string {
  if (value == null) return '0.0%'
  return `${value.toFixed(1)}%`
}

function formatNumber(value: number): string {
  if (value == null) return '0'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

function truncateAddress(address: string): string {
  if (!address || address.length < 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
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

interface DiscoveryPanelProps {
  onAnalyzeWallet?: (address: string, username?: string) => void
  view?: 'discovery' | 'pool'
}

export default function DiscoveryPanel({ onAnalyzeWallet, view = 'discovery' }: DiscoveryPanelProps) {
  const isPoolView = view === 'pool'
  const [sortBy, setSortBy] = useState<SortField>('rank_score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [currentPage, setCurrentPage] = useState(0)
  const [minTrades, setMinTrades] = useState(0)
  const [minPnl, setMinPnl] = useState(0)
  const [recommendationFilter, setRecommendationFilter] = useState<RecommendationFilter>('')
  const [marketCategoryFilter, setMarketCategoryFilter] = useState<'all' | 'politics' | 'sports' | 'crypto' | 'culture' | 'economics' | 'tech' | 'finance' | 'weather'>('all')
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('24h')
  const [insiderOnly, setInsiderOnly] = useState(false)
  const [minInsiderScore, setMinInsiderScore] = useState(0)
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [tagSearch, setTagSearch] = useState('')
  const [tagPicker, setTagPicker] = useState('')
  const [poolSearch, setPoolSearch] = useState('')
  const [poolTierFilter, setPoolTierFilter] = useState<'all' | 'core' | 'rising'>('all')
  const [poolSortBy, setPoolSortBy] = useState<'selection_score' | 'composite_score' | 'activity_score' | 'quality_score' | 'trades_24h' | 'last_trade_at'>('composite_score')
  const [poolSortDir, setPoolSortDir] = useState<'asc' | 'desc'>('desc')
  const [poolOnly, setPoolOnly] = useState(true)
  const [includeBlacklisted, setIncludeBlacklisted] = useState(true)
  const [manualPoolAddress, setManualPoolAddress] = useState('')

  const queryClient = useQueryClient()

  useEffect(() => {
    setCurrentPage(0)
  }, [
    sortBy,
    sortDir,
    minTrades,
    minPnl,
    recommendationFilter,
    marketCategoryFilter,
    timePeriod,
    selectedTags,
    insiderOnly,
    minInsiderScore,
  ])

  const { data: stats } = useQuery<DiscoveryStats>({
    queryKey: ['discovery-stats'],
    queryFn: discoveryApi.getDiscoveryStats,
    refetchInterval: 15000,
  })

  const { data: poolStats } = useQuery<PoolStats>({
    queryKey: ['discovery-pool-stats'],
    queryFn: discoveryApi.getPoolStats,
    refetchInterval: 30000,
    enabled: isPoolView,
  })

  const { data: poolMembersData, isLoading: poolMembersLoading, error: poolMembersError } = useQuery<PoolMembersResponse>({
    queryKey: [
      'discovery-pool-members',
      poolSearch,
      poolTierFilter,
      poolSortBy,
      poolSortDir,
      poolOnly,
      includeBlacklisted,
    ],
    queryFn: async () => {
      const params = {
        limit: 300,
        offset: 0,
        pool_only: poolOnly,
        include_blacklisted: includeBlacklisted,
        tier: poolTierFilter === 'all' ? undefined : poolTierFilter,
        search: poolSearch.trim() || undefined,
        sort_by: poolSortBy,
        sort_dir: poolSortDir,
      } as const
      try {
        return await discoveryApi.getPoolMembers(params)
      } catch (error: any) {
        const status = error?.response?.status
        const detail = String(error?.response?.data?.detail || '')
        const unsupportedSelectionSort =
          poolSortBy === 'selection_score'
          && status === 400
          && detail.toLowerCase().includes('invalid sort_by')
        if (unsupportedSelectionSort) {
          return discoveryApi.getPoolMembers({
            ...params,
            sort_by: 'composite_score',
          })
        }
        throw error
      }
    },
    refetchInterval: 30000,
    enabled: isPoolView,
  })

  const { data: tags = [], isLoading: tagsLoading } = useQuery<TagInfo[]>({
    queryKey: ['discovery-tags'],
    queryFn: discoveryApi.getTags,
    refetchInterval: 120000,
    enabled: !isPoolView,
  })

  const selectedTagString = selectedTags.join(',')

  const { data: leaderboardData, isLoading: leaderboardLoading } = useQuery({
    queryKey: [
      'discovery-leaderboard',
      sortBy,
      sortDir,
      currentPage,
      minTrades,
      minPnl,
      recommendationFilter,
      marketCategoryFilter,
      selectedTagString,
      timePeriod,
      insiderOnly,
      minInsiderScore,
    ],
    queryFn: () =>
      discoveryApi.getLeaderboard({
        sort_by: sortBy,
        sort_dir: sortDir,
        limit: ITEMS_PER_PAGE,
        offset: currentPage * ITEMS_PER_PAGE,
        min_trades: minTrades,
        min_pnl: minPnl || undefined,
        insider_only: insiderOnly || undefined,
        min_insider_score: minInsiderScore > 0 ? minInsiderScore : undefined,
        recommendation: recommendationFilter || undefined,
        market_category: marketCategoryFilter !== 'all' ? marketCategoryFilter : undefined,
        tags: selectedTagString || undefined,
        time_period: timePeriod !== 'all' ? timePeriod : undefined,
      }),
    refetchInterval: 30000,
    enabled: !isPoolView,
  })

  const wallets: DiscoveredWallet[] = leaderboardData?.wallets || leaderboardData || []
  const totalWallets: number = leaderboardData?.total || wallets.length
  const isWindowActive = timePeriod !== 'all' && !!leaderboardData?.window_key

  const trackWalletMutation = useMutation({
    mutationFn: (params: { address: string; username?: string | null }) =>
      analyzeAndTrackWallet({
        address: params.address,
        label:
          params.username ||
          `Discovered ${params.address.slice(0, 6)}...${params.address.slice(-4)}`,
        auto_copy: false,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
    },
  })

  const invalidatePoolQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['discovery-pool-members'] })
    queryClient.invalidateQueries({ queryKey: ['discovery-pool-stats'] })
    queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
  }, [queryClient])

  const manualIncludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.poolManualInclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const clearManualIncludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.clearPoolManualInclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const manualExcludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.poolManualExclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const clearManualExcludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.clearPoolManualExclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const blacklistMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.blacklistPoolWallet(address),
    onSuccess: invalidatePoolQueries,
  })
  const unblacklistMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.unblacklistPoolWallet(address),
    onSuccess: invalidatePoolQueries,
  })
  const deletePoolWalletMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.deletePoolWallet(address),
    onSuccess: invalidatePoolQueries,
  })
  const promoteTrackedMutation = useMutation({
    mutationFn: () => discoveryApi.promoteTrackedWalletsToPool(500),
    onSuccess: invalidatePoolQueries,
  })

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortBy === field) {
        setSortDir((d: SortDir) => (d === 'desc' ? 'asc' : 'desc'))
      } else {
        setSortBy(field)
        setSortDir('desc')
      }
    },
    [sortBy]
  )

  const handleCopyAddress = useCallback((address: string) => {
    navigator.clipboard.writeText(address).then(() => {
      setCopiedAddress(address)
      setTimeout(() => setCopiedAddress(null), 2000)
    })
  }, [])

  const toggleTagFilter = useCallback((tagName: string) => {
    setSelectedTags(prev =>
      prev.includes(tagName)
        ? prev.filter(t => t !== tagName)
        : [...prev, tagName]
    )
  }, [])

  const filteredTags = useMemo(() => {
    const q = tagSearch.trim().toLowerCase()
    if (!q) return tags
    return tags.filter(tag => {
      const display = (tag.display_name || tag.name).toLowerCase()
      const desc = (tag.description || '').toLowerCase()
      return display.includes(q) || desc.includes(q) || tag.name.toLowerCase().includes(q)
    })
  }, [tags, tagSearch])

  const poolMembers: PoolMember[] = poolMembersData?.members || []
  const poolMemberStats = poolMembersData?.stats
  const poolMembersErrorMessage = useMemo(() => {
    if (!poolMembersError) return null
    const detail = (poolMembersError as any)?.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) return detail.trim()
    return 'Failed to load pool members.'
  }, [poolMembersError])

  const totalPages = Math.ceil(totalWallets / ITEMS_PER_PAGE)
  const poolActionBusy =
    manualIncludeMutation.isPending ||
    clearManualIncludeMutation.isPending ||
    manualExcludeMutation.isPending ||
    clearManualExcludeMutation.isPending ||
    blacklistMutation.isPending ||
    unblacklistMutation.isPending ||
    deletePoolWalletMutation.isPending ||
    promoteTrackedMutation.isPending

  const statusBadge = stats?.is_running
    ? (
      <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
        <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
        Worker scanning
      </Badge>
    )
    : stats?.paused
      ? (
        <Badge variant="outline" className="text-xs bg-yellow-500/10 text-yellow-400 border-yellow-500/20">
          <PauseCircle className="w-3 h-3 mr-1" />
          Paused
        </Badge>
      )
      : (
        <Badge variant="outline" className="text-xs bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
          Auto every {stats?.interval_minutes || 60}m
        </Badge>
      )

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {statusBadge}
        {stats?.current_activity && (
          <span className="max-w-[320px] truncate">{stats.current_activity}</span>
        )}
        {stats?.last_run_at && <span>Last run: {timeAgo(stats.last_run_at)}</span>}
      </div>

      <div className={cn('grid gap-2', isPoolView ? 'grid-cols-2 lg:grid-cols-5' : 'grid-cols-2 lg:grid-cols-4')}>
        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Users className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Discovered</p>
              <p className="text-lg font-semibold">{formatNumber(stats?.total_discovered || 0)}</p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <TrendingUp className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Profitable</p>
              <p className="text-lg font-semibold">{formatNumber(stats?.total_profitable || 0)}</p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-yellow-500/10 rounded-lg">
              <Target className="w-5 h-5 text-yellow-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Copy Candidates</p>
              <p className="text-lg font-semibold">{formatNumber(stats?.total_copy_candidates || 0)}</p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <Trophy className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Analyzed (last run)</p>
              <p className="text-lg font-semibold">{formatNumber(stats?.wallets_analyzed_last_run || 0)}</p>
            </div>
          </CardContent>
        </Card>

        {isPoolView && (
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-4">
              <div className="p-2 bg-cyan-500/10 rounded-lg">
                <Users className="w-5 h-5 text-cyan-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Top Pool</p>
                <p className="text-lg font-semibold">
                  {formatNumber(poolStats?.pool_size || 0)}
                  <span className="text-[11px] text-muted-foreground ml-1">/ {poolStats?.target_pool_size || 500}</span>
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {isPoolView && poolStats && (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-2">
          <Card className="border-border">
            <CardContent className="p-2.5">
              <p className="text-xs text-muted-foreground">Pool Active (1h)</p>
              <p className="text-sm font-semibold">{poolStats.active_1h ?? 0} ({(poolStats.active_1h_pct ?? 0).toFixed(1)}%)</p>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="p-2.5">
              <p className="text-xs text-muted-foreground">Pool Active (24h)</p>
              <p className="text-sm font-semibold">{poolStats.active_24h ?? 0} ({(poolStats.active_24h_pct ?? 0).toFixed(1)}%)</p>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="p-2.5">
              <p className="text-xs text-muted-foreground">Hourly Churn</p>
              <p className="text-sm font-semibold">{((poolStats.churn_rate ?? 0) * 100).toFixed(2)}%</p>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="p-2.5">
              <p className="text-xs text-muted-foreground">Pool Recompute</p>
              <p className="text-sm font-semibold">{timeAgo(poolStats.last_pool_recompute_at)}</p>
            </CardContent>
          </Card>
        </div>
      )}

      {isPoolView && (
      <Card className="border-border overflow-hidden">
        <CardContent className="p-3 space-y-2.5">
          <div className="overflow-x-auto pb-1">
            <div className="flex min-w-max items-end gap-2">
              <div className="flex w-[190px] flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Manual add</span>
                <Input
                  value={manualPoolAddress}
                  onChange={e => setManualPoolAddress(e.target.value)}
                  placeholder="0x... add to pool"
                  className="h-8 text-xs bg-card border-border"
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={poolActionBusy || !manualPoolAddress.trim()}
                className="h-8 text-xs gap-1.5 mb-0.5"
                onClick={() => {
                  manualIncludeMutation.mutate(manualPoolAddress.trim().toLowerCase())
                  setManualPoolAddress('')
                }}
              >
                <UserCheck className="w-3.5 h-3.5" />
                Add
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => promoteTrackedMutation.mutate()}
                disabled={poolActionBusy}
                className="h-8 text-xs gap-1.5 mb-0.5"
              >
                {promoteTrackedMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <UserPlus className="w-3.5 h-3.5" />}
                Add tracked
              </Button>
              <div className="h-6 w-px bg-border/70 mb-1" />
              <div className="flex w-[210px] flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Search</span>
                <Input
                  value={poolSearch}
                  onChange={e => setPoolSearch(e.target.value)}
                  placeholder="Wallet / username"
                  className="h-8 text-xs bg-card border-border"
                />
              </div>
              <div className="flex w-[116px] flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Tier</span>
                <select
                  value={poolTierFilter}
                  onChange={e => setPoolTierFilter(e.target.value as 'all' | 'core' | 'rising')}
                  className="w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8"
                >
                  <option value="all">All tiers</option>
                  <option value="core">Core</option>
                  <option value="rising">Rising</option>
                </select>
              </div>
              <div className="flex w-[152px] flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Sort by</span>
                <select
                  value={poolSortBy}
                  onChange={e => setPoolSortBy(e.target.value as 'selection_score' | 'composite_score' | 'activity_score' | 'quality_score' | 'trades_24h' | 'last_trade_at')}
                  className="w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8"
                >
                  <option value="selection_score">Selection</option>
                  <option value="composite_score">Composite</option>
                  <option value="activity_score">Activity</option>
                  <option value="quality_score">Quality</option>
                  <option value="trades_24h">Trades 24h</option>
                  <option value="last_trade_at">Last trade</option>
                </select>
              </div>
              <div className="flex w-[94px] flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Direction</span>
                <select
                  value={poolSortDir}
                  onChange={e => setPoolSortDir(e.target.value as 'asc' | 'desc')}
                  className="w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8"
                >
                  <option value="desc">Desc</option>
                  <option value="asc">Asc</option>
                </select>
              </div>
              <label className="mb-0.5 flex h-8 items-center gap-1.5 rounded-md border border-border bg-background/40 px-2 text-[11px] text-muted-foreground">
                <input type="checkbox" checked={poolOnly} onChange={e => setPoolOnly(e.target.checked)} className="h-3.5 w-3.5" />
                Pool only
              </label>
              <label className="mb-0.5 flex h-8 items-center gap-1.5 rounded-md border border-border bg-background/40 px-2 text-[11px] text-muted-foreground">
                <input type="checkbox" checked={includeBlacklisted} onChange={e => setIncludeBlacklisted(e.target.checked)} className="h-3.5 w-3.5" />
                Show blacklisted
              </label>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2 text-[11px]">
            <div className="rounded border border-border bg-background/30 px-2 py-1.5">
              Pool: <span className="font-semibold">{poolMemberStats?.pool_members ?? 0}</span>
            </div>
            <div className="rounded border border-border bg-background/30 px-2 py-1.5">
              Tracked in Pool: <span className="font-semibold">{poolMemberStats?.tracked_in_pool ?? 0}</span>
            </div>
            <div className="rounded border border-border bg-background/30 px-2 py-1.5">
              Tracked Total: <span className="font-semibold">{poolMemberStats?.tracked_total ?? 0}</span>
            </div>
            <div className="rounded border border-border bg-background/30 px-2 py-1.5">
              Manual+ : <span className="font-semibold">{poolMemberStats?.manual_included ?? 0}</span>
            </div>
            <div className="rounded border border-border bg-background/30 px-2 py-1.5">
              Manual- : <span className="font-semibold">{poolMemberStats?.manual_excluded ?? 0}</span>
            </div>
            <div className="rounded border border-border bg-background/30 px-2 py-1.5">
              Blacklisted: <span className="font-semibold">{poolMemberStats?.blacklisted ?? 0}</span>
            </div>
          </div>

          {poolMembersLoading ? (
            <div className="py-8 flex items-center justify-center">
              <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : poolMembersErrorMessage ? (
            <div className="py-6 text-sm text-red-300 text-center">
              {poolMembersErrorMessage}
            </div>
          ) : poolMembers.length === 0 ? (
            <div className="py-6 text-sm text-muted-foreground text-center">
              No pool members match current filters.
            </div>
          ) : (
            <div className="max-h-[360px] overflow-auto rounded border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Trader</TableHead>
                    <TableHead>Tier</TableHead>
                    <TableHead>Activity</TableHead>
                    <TableHead>Selection</TableHead>
                    <TableHead>Why Selected</TableHead>
                    <TableHead>Flags</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {poolMembers.map(member => {
                    const flags = member.pool_flags || { manual_include: false, manual_exclude: false, blacklisted: false }
                    const canAnalyze = !!onAnalyzeWallet
                    const reasons = member.selection_reasons || []
                    const displayName = member.display_name || member.username || 'Unknown Trader'
                    return (
                      <TableRow key={member.address}>
                        <TableCell>
                          <div className="text-sm font-medium">{displayName}</div>
                          {member.username && member.display_name !== member.username && (
                            <div className="text-[11px] text-muted-foreground">@{member.username}</div>
                          )}
                          {member.name_source === 'tracked_label' && member.tracked_label && (
                            <div className="text-[11px] text-muted-foreground">Tracked label: {member.tracked_label}</div>
                          )}
                          <div className="text-[11px] text-muted-foreground font-mono">{truncateAddress(member.address)}</div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-[10px]">
                            {(member.pool_tier || 'out').toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs">
                          <div>1h: {member.trades_1h || 0}</div>
                          <div>24h: {member.trades_24h || 0}</div>
                          <div className="text-muted-foreground">{timeAgo(member.last_trade_at || null)}</div>
                        </TableCell>
                        <TableCell className="text-xs">
                          <div className="font-mono">
                            Sel: {(Number(member.selection_score ?? member.composite_score ?? 0) * 100).toFixed(1)}
                          </div>
                          <div className="font-mono text-muted-foreground">
                            Cmp: {(Number(member.composite_score || 0) * 100).toFixed(1)}
                          </div>
                          {member.selection_percentile != null && (
                            <div className="text-muted-foreground">
                              Top {(Number(member.selection_percentile) * 100).toFixed(1)}%
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="text-xs max-w-[280px]">
                          {reasons.length === 0 ? (
                            <span className="text-muted-foreground">No reason metadata</span>
                          ) : (
                            <div className="space-y-1">
                              {reasons.slice(0, 3).map((reason, i) => (
                                <div key={`${member.address}-${reason.code}-${i}`}>
                                  <div className="font-medium">{reason.label}</div>
                                  {reason.detail && (
                                    <div className="text-[11px] text-muted-foreground">{reason.detail}</div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="text-xs">
                          <div className="flex gap-1 flex-wrap">
                            {member.tracked_wallet && <Badge variant="outline" className="text-[9px]">Tracked</Badge>}
                            {flags.manual_include && <Badge variant="outline" className="text-[9px] bg-emerald-500/10 text-emerald-300 border-emerald-500/20">Manual+</Badge>}
                            {flags.manual_exclude && <Badge variant="outline" className="text-[9px] bg-amber-500/10 text-amber-300 border-amber-500/20">Manual-</Badge>}
                            {flags.blacklisted && <Badge variant="outline" className="text-[9px] bg-red-500/10 text-red-300 border-red-500/20">Blacklisted</Badge>}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-1 flex-wrap">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-6 px-2 text-[10px]"
                              disabled={!canAnalyze || poolActionBusy}
                              onClick={() => onAnalyzeWallet?.(member.address, member.username || member.display_name || undefined)}
                            >
                              <Activity className="w-3 h-3 mr-1" />
                              Analyze
                            </Button>
                            {!flags.manual_include ? (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-[10px]"
                                disabled={poolActionBusy}
                                onClick={() => manualIncludeMutation.mutate(member.address)}
                              >
                                <UserCheck className="w-3 h-3 mr-1" />
                                Include
                              </Button>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-[10px]"
                                disabled={poolActionBusy}
                                onClick={() => clearManualIncludeMutation.mutate(member.address)}
                              >
                                Clear+
                              </Button>
                            )}
                            {!flags.manual_exclude ? (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-[10px]"
                                disabled={poolActionBusy}
                                onClick={() => manualExcludeMutation.mutate(member.address)}
                              >
                                <UserX className="w-3 h-3 mr-1" />
                                Exclude
                              </Button>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-[10px]"
                                disabled={poolActionBusy}
                                onClick={() => clearManualExcludeMutation.mutate(member.address)}
                              >
                                Clear-
                              </Button>
                            )}
                            {!flags.blacklisted ? (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-[10px] text-red-300 border-red-500/30"
                                disabled={poolActionBusy}
                                onClick={() => blacklistMutation.mutate(member.address)}
                              >
                                <Ban className="w-3 h-3 mr-1" />
                                Blacklist
                              </Button>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 text-[10px]"
                                disabled={poolActionBusy}
                                onClick={() => unblacklistMutation.mutate(member.address)}
                              >
                                Unblacklist
                              </Button>
                            )}
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-6 px-2 text-[10px] text-red-300 border-red-500/30"
                              disabled={poolActionBusy}
                              onClick={() => {
                                if (window.confirm(`Delete ${member.address} from discovery/pool/tracking datasets?`)) {
                                  deletePoolWalletMutation.mutate(member.address)
                                }
                              }}
                            >
                              <Trash2 className="w-3 h-3 mr-1" />
                              Delete
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
      )}

      {!isPoolView && (
      <div className="space-y-3">
        <Card className="border-border">
          <CardContent className="p-3">
            <div className="overflow-x-auto pb-1">
              <div className="flex min-w-max items-end gap-2">
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Period</span>
                  <div className="flex h-8 items-center bg-muted/50 rounded-lg p-0.5 border border-border">
                    {TIME_PERIODS.map(tp => (
                      <button
                        key={tp.value}
                        onClick={() => setTimePeriod(tp.value)}
                        className={cn(
                          'h-7 px-2.5 rounded-md text-xs font-medium transition-all',
                          timePeriod === tp.value
                            ? 'bg-primary text-primary-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                        )}
                      >
                        {tp.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex w-[170px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Search tag</span>
                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                    <Input
                      type="text"
                      value={tagSearch}
                      onChange={e => setTagSearch(e.target.value)}
                      placeholder="Tag keyword"
                      className="bg-card border-border h-8 text-xs pl-8"
                    />
                  </div>
                </div>

                <div className="flex w-[190px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Add tag</span>
                  <select
                    value={tagPicker}
                    onChange={(e) => {
                      const value = e.target.value
                      if (value) {
                        toggleTagFilter(value)
                      }
                      setTagPicker('')
                    }}
                    disabled={tagsLoading || filteredTags.length === 0}
                    className="w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8"
                  >
                    <option value="">
                      {tagsLoading ? 'Loading...' : filteredTags.length === 0 ? 'No matching tags' : 'Select tag'}
                    </option>
                    {filteredTags.map((tag) => (
                      <option key={tag.name} value={tag.name}>
                        {tag.display_name || tag.name} ({tag.wallet_count})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex w-[250px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Selected tags</span>
                  <div className="flex h-8 items-center gap-1 rounded-md border border-border bg-background/40 px-1.5 overflow-x-auto">
                    {selectedTags.length === 0 ? (
                      <span className="text-xs text-muted-foreground/80">None</span>
                    ) : (
                      selectedTags.map((name) => {
                        const tagMeta = tags.find((tag) => tag.name === name)
                        return (
                          <button
                            key={name}
                            onClick={() => toggleTagFilter(name)}
                            className="inline-flex h-6 items-center rounded-full border border-border bg-background/70 px-2 text-[10px] text-muted-foreground hover:text-foreground whitespace-nowrap"
                            title="Remove tag filter"
                          >
                            {tagMeta?.display_name || name}
                            <span className="ml-1 text-muted-foreground/70">x</span>
                          </button>
                        )
                      })
                    )}
                  </div>
                </div>

                <div className="flex w-[112px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Min trades</span>
                  <Input
                    type="number"
                    value={minTrades}
                    onChange={e => setMinTrades(parseInt(e.target.value) || 0)}
                    min={0}
                    className="bg-card border-border h-8 text-xs"
                  />
                </div>

                <div className="flex w-[120px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Min pnl ($)</span>
                  <Input
                    type="number"
                    value={minPnl}
                    onChange={e => setMinPnl(parseFloat(e.target.value) || 0)}
                    step={100}
                    className="bg-card border-border h-8 text-xs"
                  />
                </div>

                <div className="flex w-[148px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Recommendation</span>
                  <select
                    value={recommendationFilter}
                    onChange={e => setRecommendationFilter(e.target.value as RecommendationFilter)}
                    className="w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8"
                  >
                    <option value="">All</option>
                    <option value="copy_candidate">Copy candidate</option>
                    <option value="monitor">Monitor</option>
                    <option value="avoid">Avoid</option>
                  </select>
                </div>

                <div className="flex w-[140px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Category</span>
                  <select
                    value={marketCategoryFilter}
                    onChange={e => setMarketCategoryFilter(e.target.value as 'all' | 'politics' | 'sports' | 'crypto' | 'culture' | 'economics' | 'tech' | 'finance' | 'weather')}
                    className="w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8"
                  >
                    <option value="all">All</option>
                    <option value="politics">Politics</option>
                    <option value="sports">Sports</option>
                    <option value="crypto">Crypto</option>
                    <option value="culture">Culture</option>
                    <option value="economics">Economics</option>
                    <option value="tech">Tech</option>
                    <option value="finance">Finance</option>
                    <option value="weather">Weather</option>
                  </select>
                </div>

                <div className="flex w-[138px] flex-col gap-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">Min insider</span>
                  <Input
                    type="number"
                    value={minInsiderScore}
                    onChange={e => setMinInsiderScore(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
                    step={0.05}
                    min={0}
                    max={1}
                    className="bg-card border-border h-8 text-xs"
                  />
                </div>

                <label className="mb-0.5 flex h-8 items-center gap-1.5 rounded-md border border-border bg-background/40 px-2 text-[11px] text-muted-foreground whitespace-nowrap">
                  <input
                    type="checkbox"
                    checked={insiderOnly}
                    onChange={e => setInsiderOnly(e.target.checked)}
                    className="h-3.5 w-3.5 rounded border-border"
                  />
                  Insider only
                </label>

                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 px-2 text-xs mb-0.5 text-muted-foreground"
                  onClick={() => {
                    setTimePeriod('24h')
                    setTagSearch('')
                    setTagPicker('')
                    setSelectedTags([])
                    setMinTrades(0)
                    setMinPnl(0)
                    setRecommendationFilter('')
                    setMarketCategoryFilter('all')
                    setMinInsiderScore(0)
                    setInsiderOnly(false)
                  }}
                >
                  Reset
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {leaderboardLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : wallets.length === 0 ? (
          <Card className="border-border">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Trophy className="w-12 h-12 text-muted-foreground/30 mb-4" />
              <p className="text-muted-foreground">No wallets found</p>
              <p className="text-sm text-muted-foreground/70 mt-1">
                Try clearing filters or wait for the discovery worker to complete the next run
              </p>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card className="border-border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead className="min-w-[220px]">Trader</TableHead>
                    <TableHead>
                      <SortButton
                        field="composite_score"
                        label="Composite"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="activity_score"
                        label="Activity"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="quality_score"
                        label="Quality"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="insider_score"
                        label="Insider"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="last_trade_at"
                        label="Last Trade"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="total_pnl"
                        label={isWindowActive ? 'Period PnL' : 'PnL'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="win_rate"
                        label={isWindowActive ? 'Period WR' : 'Win Rate'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="sharpe_ratio"
                        label={isWindowActive ? 'Period Sharpe' : 'Sharpe'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="total_trades"
                        label={isWindowActive ? 'Period Trades' : 'Trades'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>
                      <SortButton
                        field="avg_roi"
                        label={isWindowActive ? 'Period ROI' : 'Avg ROI'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead>Rec.</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {wallets.map((wallet, idx) => (
                    <LeaderboardRow
                      key={wallet.address}
                      wallet={wallet}
                      rank={currentPage * ITEMS_PER_PAGE + idx + 1}
                      copiedAddress={copiedAddress}
                      onCopyAddress={handleCopyAddress}
                      onAnalyze={onAnalyzeWallet}
                      onTrack={(address, username) =>
                        trackWalletMutation.mutate({ address, username })
                      }
                      isTracking={trackWalletMutation.isPending}
                      useWindowMetrics={isWindowActive}
                    />
                  ))}
                </TableBody>
              </Table>
            </Card>

            {totalPages > 1 && (
              <div className="flex items-center justify-between pt-2">
                <div className="text-sm text-muted-foreground">
                  Showing {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalWallets)} of {totalWallets}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                    disabled={currentPage === 0}
                  >
                    Previous
                  </Button>
                  <span className="px-3 py-1.5 bg-card rounded-lg text-sm border border-border">
                    Page {currentPage + 1} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(p => p + 1)}
                    disabled={currentPage >= totalPages - 1}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
      )}
    </div>
  )
}

function SortButton({
  field,
  label,
  currentSort,
  currentDir,
  onSort,
}: {
  field: SortField
  label: string
  currentSort: SortField
  currentDir: SortDir
  onSort: (field: SortField) => void
}) {
  const isActive = currentSort === field
  return (
    <button
      onClick={() => onSort(field)}
      className={cn(
        'flex items-center gap-1 text-xs font-medium transition-colors whitespace-nowrap',
        isActive ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
      )}
    >
      {label}
      {isActive &&
        (currentDir === 'desc' ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronUp className="w-3 h-3" />
        ))}
    </button>
  )
}

function WalletAddress({
  address,
  username,
  copiedAddress,
  onCopy,
}: {
  address: string
  username: string | null
  copiedAddress: string | null
  onCopy: (address: string) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <div>
        {username && <p className="text-sm font-medium text-foreground">{username}</p>}
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-xs text-muted-foreground">
            {truncateAddress(address)}
          </span>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => onCopy(address)}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                {copiedAddress === address ? (
                  <CheckCircle className="w-3 h-3 text-green-400" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent>
              {copiedAddress === address ? 'Copied!' : 'Copy address'}
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  )
}

function PnlDisplay({ value }: { value: number }) {
  const isPositive = value >= 0
  return (
    <span
      className={cn(
        'font-medium font-mono text-sm',
        isPositive ? 'text-green-400' : 'text-red-400'
      )}
    >
      {isPositive ? '+' : ''}${formatPnl(value)}
    </span>
  )
}

function RecommendationBadge({ recommendation }: { recommendation: string }) {
  const colorClass =
    RECOMMENDATION_COLORS[recommendation] ||
    'bg-muted-foreground/15 text-muted-foreground border-muted-foreground/20'
  const label = RECOMMENDATION_LABELS[recommendation] || recommendation
  return (
    <Badge variant="outline" className={cn('text-[10px] font-semibold', colorClass)}>
      {label}
    </Badge>
  )
}

function LeaderboardRow({
  wallet,
  rank,
  copiedAddress,
  onCopyAddress,
  onAnalyze,
  onTrack,
  isTracking,
  useWindowMetrics,
}: {
  wallet: DiscoveredWallet
  rank: number
  copiedAddress: string | null
  onCopyAddress: (address: string) => void
  onAnalyze?: (address: string, username?: string) => void
  onTrack?: (address: string, username?: string | null) => void
  isTracking?: boolean
  useWindowMetrics?: boolean
}) {
  const rankDisplay = useWindowMetrics ? rank : wallet.rank_position || rank

  const pnl =
    useWindowMetrics && wallet.period_pnl != null
      ? wallet.period_pnl
      : (wallet.total_pnl ?? 0)
  const winRate =
    useWindowMetrics && wallet.period_win_rate != null
      ? wallet.period_win_rate
      : (wallet.win_rate ?? 0)
  const sharpe = useWindowMetrics
    ? wallet.period_sharpe ?? wallet.sharpe_ratio
    : wallet.sharpe_ratio
  const trades =
    useWindowMetrics && wallet.period_trades != null
      ? wallet.period_trades
      : (wallet.total_trades ?? 0)
  const roi =
    useWindowMetrics && wallet.period_roi != null ? wallet.period_roi : (wallet.avg_roi ?? 0)
  const composite = wallet.composite_score ?? wallet.rank_score ?? 0
  const activity = wallet.activity_score ?? 0
  const quality = wallet.quality_score ?? wallet.rank_score ?? 0
  const marketCategories = wallet.market_categories || []
  const tagPills = [...(wallet.tags || []).slice(0, 3)]
  const categoryPills = marketCategories.slice(0, 2).map(cat => `mkt:${cat}`)
  const pills = [...tagPills, ...categoryPills]

  return (
    <TableRow>
      <TableCell className="font-medium text-muted-foreground">
        <span
          className={cn(
            'flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold',
            rankDisplay === 1
              ? 'bg-yellow-500/20 text-yellow-400'
              : rankDisplay === 2
                ? 'bg-muted-foreground/20 text-muted-foreground'
                : rankDisplay === 3
                  ? 'bg-amber-600/20 text-amber-500'
                  : 'bg-muted text-muted-foreground'
          )}
        >
          {rankDisplay}
        </span>
      </TableCell>

      <TableCell>
        <WalletAddress
          address={wallet.address}
          username={wallet.username}
          copiedAddress={copiedAddress}
          onCopy={onCopyAddress}
        />
        {pills.length > 0 && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {pills.map(tag => (
              <span
                key={tag}
                className="px-1.5 py-0.5 text-[10px] rounded bg-muted text-muted-foreground border border-border"
              >
                {tag}
              </span>
            ))}
            {(wallet.tags.length + marketCategories.length) > pills.length && (
              <span className="text-[10px] text-muted-foreground">
                +{wallet.tags.length + marketCategories.length - pills.length}
              </span>
            )}
          </div>
        )}
      </TableCell>

      <TableCell>
        <span
          className={cn(
            'font-mono text-sm',
            composite >= 0.7
              ? 'text-green-400'
              : composite >= 0.5
                ? 'text-yellow-400'
                : 'text-muted-foreground'
          )}
        >
          {(composite * 100).toFixed(1)}
        </span>
      </TableCell>

      <TableCell>
        <span
          className={cn(
            'font-mono text-sm',
            activity >= 0.6
              ? 'text-green-400'
              : activity >= 0.3
                ? 'text-yellow-400'
                : 'text-muted-foreground'
          )}
        >
          {(activity * 100).toFixed(1)}
        </span>
      </TableCell>

      <TableCell>
        <span
          className={cn(
            'font-mono text-sm',
            quality >= 0.6
              ? 'text-green-400'
              : quality >= 0.4
                ? 'text-yellow-400'
                : 'text-muted-foreground'
          )}
        >
          {(quality * 100).toFixed(1)}
        </span>
      </TableCell>

      <TableCell>
        {wallet.insider_score != null ? (
          <div className="space-y-0.5">
            <span
              className={cn(
                'font-mono text-sm',
                (wallet.insider_score || 0) >= 0.72
                  ? 'text-red-400'
                  : (wallet.insider_score || 0) >= 0.60
                    ? 'text-yellow-400'
                    : 'text-muted-foreground'
              )}
            >
              {(wallet.insider_score || 0).toFixed(2)}
            </span>
            <div className="text-[10px] text-muted-foreground">
              conf {(wallet.insider_confidence || 0).toFixed(2)} · n{wallet.insider_sample_size || 0}
            </div>
            {(wallet.insider_score || 0) >= 0.72 &&
              (wallet.insider_confidence || 0) >= 0.60 &&
              (wallet.insider_sample_size || 0) >= 25 && (
                <Badge variant="outline" className="text-[9px] bg-red-500/10 text-red-300 border-red-500/20">
                  <AlertTriangle className="w-2.5 h-2.5 mr-1" />
                  Insider suspect
                </Badge>
              )}
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">--</span>
        )}
      </TableCell>

      <TableCell>
        <span className="text-xs text-muted-foreground">
          {timeAgo(wallet.last_trade_at || null)}
        </span>
      </TableCell>

      <TableCell>
        <div>
          <PnlDisplay value={pnl} />
          {useWindowMetrics && wallet.period_pnl != null && (
            <div className="text-[10px] text-muted-foreground/60 mt-0.5">
              All: ${formatPnl(wallet.total_pnl)}
            </div>
          )}
        </div>
      </TableCell>

      <TableCell>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'font-medium text-sm',
              winRate >= 60
                ? 'text-green-400'
                : winRate >= 45
                  ? 'text-yellow-400'
                  : 'text-red-400'
            )}
          >
            {formatPercent(winRate)}
          </span>
          {!useWindowMetrics && (
            <span className="text-[10px] text-muted-foreground">
              {wallet.wins}W/{wallet.losses}L
            </span>
          )}
        </div>
      </TableCell>

      <TableCell>
        {sharpe != null ? (
          <span
            className={cn(
              'font-mono text-sm',
              sharpe >= 2
                ? 'text-green-400'
                : sharpe >= 1
                  ? 'text-yellow-400'
                  : 'text-muted-foreground'
            )}
          >
            {sharpe.toFixed(2)}
          </span>
        ) : (
          <span className="text-muted-foreground text-xs">--</span>
        )}
      </TableCell>

      <TableCell className="text-muted-foreground text-sm">
        {trades}
        {!useWindowMetrics && (
          <span className="text-[10px] text-muted-foreground/70 ml-1">
            ({(wallet.trades_per_day ?? 0).toFixed(1)}/d)
          </span>
        )}
      </TableCell>

      <TableCell>
        <span
          className={cn(
            'font-mono text-sm',
            roi >= 0 ? 'text-green-400' : 'text-red-400'
          )}
        >
          {roi >= 0 ? '+' : ''}
          {formatPercent(roi)}
        </span>
      </TableCell>

      <TableCell>
        <RecommendationBadge recommendation={wallet.recommendation} />
      </TableCell>

      <TableCell>
        <div className="flex items-center gap-1">
          {onAnalyze && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onAnalyze(wallet.address, wallet.username || undefined)}
                  className="p-1.5 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                >
                  <Activity className="w-3.5 h-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Analyze wallet</TooltipContent>
            </Tooltip>
          )}
          {onTrack && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onTrack(wallet.address, wallet.username)}
                  disabled={isTracking}
                  className="p-1.5 rounded bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors disabled:opacity-50"
                >
                  <UserPlus className="w-3.5 h-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Track wallet</TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <a
                href={`https://polymarket.com/profile/${wallet.address}`}
                target="_blank"
                rel="noopener noreferrer"
                className="p-1.5 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </TooltipTrigger>
            <TooltipContent>View on Polymarket</TooltipContent>
          </Tooltip>
        </div>
      </TableCell>
    </TableRow>
  )
}
