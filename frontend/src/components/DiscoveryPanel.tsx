import { useState, useEffect, useCallback, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Trophy,
  RefreshCw,
  Users,
  Target,
  Tag,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  Copy,
  Search,
  Play,
  TrendingUp,
  Zap,
  ExternalLink,
  Activity,
  UserPlus,
  DollarSign,
  ArrowUpRight,
  Clock,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
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
  type ConfluenceSignal,
  type TagInfo,
  type DiscoveryStats,
  type PoolStats,
} from '../services/discoveryApi'
import { analyzeAndTrackWallet, type Opportunity, getOpportunities } from '../services/api'

// ==================== TYPES ====================

type DiscoveryTab = 'leaderboard' | 'confluence' | 'tags'

type SortField =
  | 'rank_score'
  | 'composite_score'
  | 'quality_score'
  | 'activity_score'
  | 'last_trade_at'
  | 'total_pnl'
  | 'win_rate'
  | 'sharpe_ratio'
  | 'total_trades'
  | 'avg_roi'

type SortDir = 'asc' | 'desc'

type RecommendationFilter = '' | 'copy_candidate' | 'monitor' | 'avoid'

type TimePeriod = '24h' | '7d' | '30d' | '90d' | 'all'

// ==================== CONSTANTS ====================

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

const SIGNAL_TYPE_COLORS: Record<string, string> = {
  multi_wallet_buy: 'bg-green-500/15 text-green-400 border-green-500/20',
  multi_wallet_sell: 'bg-red-500/15 text-red-400 border-red-500/20',
  accumulation: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
  coordinated_buy: 'bg-green-500/15 text-green-400 border-green-500/20',
  coordinated_sell: 'bg-red-500/15 text-red-400 border-red-500/20',
  whale_movement: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  smart_money: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
  consensus: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/20',
}

const ITEMS_PER_PAGE = 25

// ==================== HELPERS ====================

function formatPnl(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return value.toFixed(2)
}

function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`
}

function formatNumber(value: number): string {
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

// ==================== MAIN COMPONENT ====================

interface DiscoveryPanelProps {
  onAnalyzeWallet?: (address: string, username?: string) => void
  onExecuteTrade?: (opportunity: Opportunity) => void
}

export default function DiscoveryPanel({ onAnalyzeWallet, onExecuteTrade }: DiscoveryPanelProps) {
  const [activeTab, setActiveTab] = useState<DiscoveryTab>('leaderboard')
  const [sortBy, setSortBy] = useState<SortField>('rank_score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [currentPage, setCurrentPage] = useState(0)
  const [minTrades, setMinTrades] = useState(0)
  const [minPnl, setMinPnl] = useState(0)
  const [recommendationFilter, setRecommendationFilter] = useState<RecommendationFilter>('')
  const [tagFilter, setTagFilter] = useState('')
  const [tagFilters, setTagFilters] = useState<string[]>([])
  const [tagSortBy, setTagSortBy] = useState<SortField>('rank_score')
  const [tagSortDir, setTagSortDir] = useState<SortDir>('desc')
  const [tagPage, setTagPage] = useState(0)
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('all')
  const [tagTimePeriod, setTagTimePeriod] = useState<TimePeriod>('all')
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null)
  const [confluenceMinStrength, setConfluenceMinStrength] = useState(0)
  const queryClient = useQueryClient()
  const refreshTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Auto-refresh every 60 seconds
  useEffect(() => {
    refreshTimerRef.current = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-stats'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-confluence'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-tags'] })
    }, 60000)

    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current)
    }
  }, [queryClient])

  // Reset page on filter changes
  useEffect(() => {
    setCurrentPage(0)
  }, [sortBy, sortDir, minTrades, minPnl, recommendationFilter, tagFilter, timePeriod])

  // ==================== QUERIES ====================

  const { data: stats } = useQuery<DiscoveryStats>({
    queryKey: ['discovery-stats'],
    queryFn: discoveryApi.getDiscoveryStats,
    refetchInterval: 30000,
  })

  const { data: poolStats } = useQuery<PoolStats>({
    queryKey: ['discovery-pool-stats'],
    queryFn: discoveryApi.getPoolStats,
    refetchInterval: 30000,
  })

  const { data: leaderboardData, isLoading: leaderboardLoading } = useQuery({
    queryKey: ['discovery-leaderboard', sortBy, sortDir, currentPage, minTrades, minPnl, recommendationFilter, tagFilter, timePeriod],
    queryFn: () => discoveryApi.getLeaderboard({
      sort_by: sortBy,
      sort_dir: sortDir,
      limit: ITEMS_PER_PAGE,
      offset: currentPage * ITEMS_PER_PAGE,
      min_trades: minTrades,
      min_pnl: minPnl || undefined,
      recommendation: recommendationFilter || undefined,
      tags: tagFilter || undefined,
      time_period: timePeriod !== 'all' ? timePeriod : undefined,
    }),
    enabled: activeTab === 'leaderboard',
  })

  const wallets: DiscoveredWallet[] = leaderboardData?.wallets || leaderboardData || []
  const totalWallets: number = leaderboardData?.total || wallets.length
  const isWindowActive = timePeriod !== 'all' && leaderboardData?.window_key

  const { data: confluenceSignals = [], isLoading: confluenceLoading } = useQuery<ConfluenceSignal[]>({
    queryKey: ['discovery-confluence', confluenceMinStrength],
    queryFn: () => discoveryApi.getConfluenceSignals(confluenceMinStrength, 50),
    enabled: activeTab === 'confluence',
  })

  const { data: tags = [], isLoading: tagsLoading } = useQuery<TagInfo[]>({
    queryKey: ['discovery-tags'],
    queryFn: discoveryApi.getTags,
    enabled: activeTab === 'tags',
  })

  const tagFilterString = tagFilters.join(',')

  const { data: tagLeaderboardData, isLoading: tagWalletsLoading } = useQuery({
    queryKey: ['discovery-tags-leaderboard', tagSortBy, tagSortDir, tagPage, tagFilterString, tagTimePeriod],
    queryFn: () => discoveryApi.getLeaderboard({
      sort_by: tagSortBy,
      sort_dir: tagSortDir,
      limit: ITEMS_PER_PAGE,
      offset: tagPage * ITEMS_PER_PAGE,
      min_trades: 0,
      tags: tagFilterString || undefined,
      time_period: tagTimePeriod !== 'all' ? tagTimePeriod : undefined,
    }),
    enabled: activeTab === 'tags',
  })

  const tagWallets: DiscoveredWallet[] = tagLeaderboardData?.wallets || tagLeaderboardData || []
  const tagTotalWallets: number = tagLeaderboardData?.total || tagWallets.length
  const isTagWindowActive = tagTimePeriod !== 'all' && tagLeaderboardData?.window_key

  // ==================== MUTATIONS ====================

  const discoveryMutation = useMutation({
    mutationFn: () => discoveryApi.triggerDiscovery(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-stats'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
    },
  })

  const refreshLeaderboardMutation = useMutation({
    mutationFn: discoveryApi.refreshLeaderboard,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-stats'] })
    },
  })

  const confluenceScanMutation = useMutation({
    mutationFn: discoveryApi.triggerConfluenceScan,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-confluence'] })
    },
  })

  const trackWalletMutation = useMutation({
    mutationFn: (params: { address: string; username?: string | null }) =>
      analyzeAndTrackWallet({
        address: params.address,
        label: params.username || `Discovered ${params.address.slice(0, 6)}...${params.address.slice(-4)}`,
        auto_copy: false,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
    },
  })

  const { data: activeSignalCount = 0 } = useQuery({
    queryKey: ['discovery-active-signal-count'],
    queryFn: async () => {
      const signals = await discoveryApi.getConfluenceSignals(0, 100)
      return signals.filter((s: ConfluenceSignal) => s.is_active).length
    },
    refetchInterval: 60000,
  })

  // ==================== HANDLERS ====================

  const handleSort = useCallback((field: SortField) => {
    if (sortBy === field) {
      setSortDir((d: SortDir) => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(field)
      setSortDir('desc')
    }
  }, [sortBy])

  const handleTagSort = useCallback((field: SortField) => {
    if (tagSortBy === field) {
      setTagSortDir((d: SortDir) => d === 'desc' ? 'asc' : 'desc')
    } else {
      setTagSortBy(field)
      setTagSortDir('desc')
    }
    setTagPage(0)
  }, [tagSortBy])

  const toggleTagFilter = useCallback((tagName: string) => {
    setTagFilters(prev =>
      prev.includes(tagName)
        ? prev.filter(t => t !== tagName)
        : [...prev, tagName]
    )
    setTagPage(0)
  }, [])

  const handleCopyAddress = useCallback((address: string) => {
    navigator.clipboard.writeText(address).then(() => {
      setCopiedAddress(address)
      setTimeout(() => setCopiedAddress(null), 2000)
    })
  }, [])

  const totalPages = Math.ceil(totalWallets / ITEMS_PER_PAGE)

  // ==================== TAB DEFINITIONS ====================

  const tabDefs: { id: DiscoveryTab; icon: typeof Trophy; label: string; color: string }[] = [
    { id: 'leaderboard', icon: Trophy, label: 'Leaderboard', color: 'yellow' },
    { id: 'confluence', icon: Zap, label: 'Confluence', color: 'cyan' },
    { id: 'tags', icon: Tag, label: 'Tags', color: 'purple' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Trader Discovery</h2>
          <p className="text-sm text-muted-foreground">
            Leaderboard rankings, confluence signals, and behavioral tags
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Discovery Status */}
          {stats && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {stats.is_running && (
                <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                  <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                  Scanning...
                </Badge>
              )}
              {stats.last_run_at && (
                <span>Last run: {timeAgo(stats.last_run_at)}</span>
              )}
            </div>
          )}

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refreshLeaderboardMutation.mutate()}
                disabled={refreshLeaderboardMutation.isPending}
                className="flex items-center gap-1.5"
              >
                <RefreshCw className={cn("w-3.5 h-3.5", refreshLeaderboardMutation.isPending && "animate-spin")} />
                Refresh
              </Button>
            </TooltipTrigger>
            <TooltipContent>Refresh leaderboard rankings</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={() => discoveryMutation.mutate()}
                disabled={discoveryMutation.isPending || stats?.is_running}
                className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white"
              >
                <Play className={cn("w-4 h-4", discoveryMutation.isPending && "animate-spin")} />
                Run Discovery
              </Button>
            </TooltipTrigger>
            <TooltipContent>Scan markets to discover new traders</TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-5 gap-4">
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
              <Zap className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Active Signals</p>
              <p className="text-lg font-semibold">{formatNumber(activeSignalCount)}</p>
            </div>
          </CardContent>
        </Card>
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
      </div>

      {poolStats && (
        <div className="grid grid-cols-4 gap-4">
          <Card className="border-border">
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Pool Active (1h)</p>
              <p className="text-sm font-semibold">{poolStats.active_1h} ({poolStats.active_1h_pct.toFixed(1)}%)</p>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Pool Active (24h)</p>
              <p className="text-sm font-semibold">{poolStats.active_24h} ({poolStats.active_24h_pct.toFixed(1)}%)</p>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Hourly Churn</p>
              <p className="text-sm font-semibold">{(poolStats.churn_rate * 100).toFixed(2)}%</p>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Pool Recompute</p>
              <p className="text-sm font-semibold">{timeAgo(poolStats.last_pool_recompute_at)}</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Mutation Feedback */}
      {discoveryMutation.isSuccess && (
        <div className="flex items-center gap-2 p-3 rounded-lg text-sm bg-green-500/10 text-green-400 border border-green-500/20">
          <CheckCircle className="w-4 h-4" />
          Discovery scan triggered successfully
        </div>
      )}
      {discoveryMutation.isError && (
        <div className="flex items-center gap-2 p-3 rounded-lg text-sm bg-red-500/10 text-red-400 border border-red-500/20">
          <AlertCircle className="w-4 h-4" />
          Failed to trigger discovery: {(discoveryMutation.error as Error).message}
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex items-center gap-2">
        {tabDefs.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          const colorMap: Record<string, string> = {
            yellow: isActive
              ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30 hover:text-yellow-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border',
            cyan: isActive
              ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border',
            purple: isActive
              ? 'bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30 hover:text-purple-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border',
          }
          return (
            <Button
              key={tab.id}
              variant="outline"
              size="sm"
              onClick={() => setActiveTab(tab.id)}
              className={cn("flex items-center gap-2", colorMap[tab.color])}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </Button>
          )
        })}
      </div>

      {/* ==================== LEADERBOARD TAB ==================== */}
      {activeTab === 'leaderboard' && (
        <div className="space-y-4">
          {/* Time Period Filter */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Clock className="w-3.5 h-3.5" />
              <span>Period</span>
            </div>
            <div className="flex items-center bg-muted/50 rounded-lg p-0.5 border border-border">
              {TIME_PERIODS.map(tp => (
                <button
                  key={tp.value}
                  onClick={() => setTimePeriod(tp.value)}
                  className={cn(
                    'px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                    timePeriod === tp.value
                      ? 'bg-primary text-primary-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  )}
                >
                  {tp.label}
                </button>
              ))}
            </div>
            {timePeriod !== 'all' && (
              <span className="text-[10px] text-muted-foreground/70">
                Ranked by trading performance in the {TIME_PERIODS.find(p => p.value === timePeriod)?.description}
              </span>
            )}
          </div>

          {/* Filters */}
          <div className="flex items-center gap-4 flex-wrap">
            <div className="w-36">
              <label className="block text-xs text-muted-foreground mb-1">Min Trades</label>
              <Input
                type="number"
                value={minTrades}
                onChange={e => setMinTrades(parseInt(e.target.value) || 0)}
                min={0}
                className="bg-card border-border h-8 text-sm"
              />
            </div>
            <div className="w-36">
              <label className="block text-xs text-muted-foreground mb-1">Min PnL ($)</label>
              <Input
                type="number"
                value={minPnl}
                onChange={e => setMinPnl(parseFloat(e.target.value) || 0)}
                step={100}
                className="bg-card border-border h-8 text-sm"
              />
            </div>
            <div className="w-44">
              <label className="block text-xs text-muted-foreground mb-1">Recommendation</label>
              <select
                value={recommendationFilter}
                onChange={e => setRecommendationFilter(e.target.value as RecommendationFilter)}
                className="w-full bg-card border border-border rounded-lg px-3 py-1.5 text-sm h-8"
              >
                <option value="">All</option>
                <option value="copy_candidate">Copy Candidate</option>
                <option value="monitor">Monitor</option>
                <option value="avoid">Avoid</option>
              </select>
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs text-muted-foreground mb-1">Filter by Tag</label>
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                <Input
                  type="text"
                  value={tagFilter}
                  onChange={e => setTagFilter(e.target.value)}
                  placeholder="e.g. whale, high_frequency"
                  className="bg-card border-border h-8 text-sm pl-8"
                />
              </div>
            </div>
          </div>

          {/* Leaderboard Table */}
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
                  Run discovery to find profitable traders, or adjust filters
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
                      <TableHead className="min-w-[180px]">Trader</TableHead>
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
                      <TableHead>Tags</TableHead>
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
                        onTrack={(address, username) => trackWalletMutation.mutate({ address, username })}
                        isTracking={trackWalletMutation.isPending}
                        useWindowMetrics={!!isWindowActive}
                      />
                    ))}
                  </TableBody>
                </Table>
              </Card>

              {/* Pagination */}
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

      {/* ==================== CONFLUENCE TAB ==================== */}
      {activeTab === 'confluence' && (
        <div className="space-y-4">
          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-48">
                <label className="block text-xs text-muted-foreground mb-1">Min Strength</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    value={confluenceMinStrength}
                    onChange={e => setConfluenceMinStrength(parseFloat(e.target.value))}
                    step="0.1"
                    min="0"
                    max="1"
                    className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer"
                  />
                  <span className="text-xs text-muted-foreground w-10 text-right">
                    {(confluenceMinStrength * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => confluenceScanMutation.mutate()}
              disabled={confluenceScanMutation.isPending}
              className="flex items-center gap-1.5"
            >
              <RefreshCw className={cn("w-3.5 h-3.5", confluenceScanMutation.isPending && "animate-spin")} />
              Scan Confluence
            </Button>
          </div>

          {/* Signals */}
          {confluenceLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : confluenceSignals.length === 0 ? (
            <Card className="border-border">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Zap className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">No confluence signals detected</p>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  Run a confluence scan to detect coordinated trading patterns
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {confluenceSignals.map(signal => (
                <ConfluenceCard
                  key={signal.id}
                  signal={signal}
                  onExecuteTrade={onExecuteTrade}
                  onAnalyzeWallet={onAnalyzeWallet}
                  onTrackWallet={(address, username) => trackWalletMutation.mutate({ address, username })}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ==================== TAGS TAB ==================== */}
      {activeTab === 'tags' && (
        <div className="space-y-4">
          {/* Time Period Filter for Tags */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Clock className="w-3.5 h-3.5" />
              <span>Period</span>
            </div>
            <div className="flex items-center bg-muted/50 rounded-lg p-0.5 border border-border">
              {TIME_PERIODS.map(tp => (
                <button
                  key={tp.value}
                  onClick={() => { setTagTimePeriod(tp.value); setTagPage(0) }}
                  className={cn(
                    'px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                    tagTimePeriod === tp.value
                      ? 'bg-primary text-primary-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  )}
                >
                  {tp.label}
                </button>
              ))}
            </div>
            {tagTimePeriod !== 'all' && (
              <span className="text-[10px] text-muted-foreground/70">
                Ranked by trading performance in the {TIME_PERIODS.find(p => p.value === tagTimePeriod)?.description}
              </span>
            )}
          </div>

          {/* Tag Filters */}
          {tagsLoading ? (
            <div className="flex items-center justify-center py-4">
              <RefreshCw className="w-5 h-5 animate-spin text-muted-foreground" />
            </div>
          ) : tags.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-muted-foreground font-medium">Filter by Tags</label>
                {tagFilters.length > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => { setTagFilters([]); setTagPage(0) }}
                    className="text-xs text-muted-foreground h-6 px-2"
                  >
                    Clear filters
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {tags.map(tag => {
                  const isActive = tagFilters.includes(tag.name)
                  const tagStyle = tag.color
                    ? isActive
                      ? { borderColor: tag.color, backgroundColor: `${tag.color}25`, color: tag.color }
                      : { borderColor: `${tag.color}40`, color: `${tag.color}99` }
                    : {}
                  return (
                    <button
                      key={tag.name}
                      onClick={() => toggleTagFilter(tag.name)}
                      className={cn(
                        "px-2.5 py-1 rounded-full text-xs font-medium border transition-all",
                        isActive
                          ? "ring-1 ring-primary/30"
                          : "hover:ring-1 hover:ring-primary/20 opacity-70 hover:opacity-100"
                      )}
                      style={tagStyle}
                    >
                      {tag.display_name || tag.name}
                      <span className="ml-1.5 opacity-60">{tag.wallet_count}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {/* Wallets Table */}
          {tagWalletsLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : tagWallets.length === 0 ? (
            <Card className="border-border">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Tag className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">No wallets found</p>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  {tagFilters.length > 0
                    ? 'No wallets match the selected tag filters. Try removing some filters.'
                    : 'Run discovery to find and tag traders.'}
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
                      <TableHead className="min-w-[180px]">Trader</TableHead>
                      <TableHead>
                        <SortButton field="composite_score" label="Composite" currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="activity_score" label="Activity" currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="quality_score" label="Quality" currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="last_trade_at" label="Last Trade" currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="total_pnl" label={isTagWindowActive ? 'Period PnL' : 'PnL'} currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="win_rate" label={isTagWindowActive ? 'Period WR' : 'Win Rate'} currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="sharpe_ratio" label={isTagWindowActive ? 'Period Sharpe' : 'Sharpe'} currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="total_trades" label={isTagWindowActive ? 'Period Trades' : 'Trades'} currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>
                        <SortButton field="avg_roi" label={isTagWindowActive ? 'Period ROI' : 'Avg ROI'} currentSort={tagSortBy} currentDir={tagSortDir} onSort={handleTagSort} />
                      </TableHead>
                      <TableHead>Tags</TableHead>
                      <TableHead>Rec.</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {tagWallets.map((wallet, idx) => (
                      <LeaderboardRow
                        key={wallet.address}
                        wallet={wallet}
                        rank={tagPage * ITEMS_PER_PAGE + idx + 1}
                        copiedAddress={copiedAddress}
                        onCopyAddress={handleCopyAddress}
                        onAnalyze={onAnalyzeWallet}
                        onTrack={(address, username) => trackWalletMutation.mutate({ address, username })}
                        isTracking={trackWalletMutation.isPending}
                        useWindowMetrics={!!isTagWindowActive}
                      />
                    ))}
                  </TableBody>
                </Table>
              </Card>

              {/* Pagination */}
              {(() => {
                const tagTotalPages = Math.ceil(tagTotalWallets / ITEMS_PER_PAGE)
                return tagTotalPages > 1 ? (
                  <div className="flex items-center justify-between pt-2">
                    <div className="text-sm text-muted-foreground">
                      Showing {tagPage * ITEMS_PER_PAGE + 1} - {Math.min((tagPage + 1) * ITEMS_PER_PAGE, tagTotalWallets)} of {tagTotalWallets}
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setTagPage(p => Math.max(0, p - 1))}
                        disabled={tagPage === 0}
                      >
                        Previous
                      </Button>
                      <span className="px-3 py-1.5 bg-card rounded-lg text-sm border border-border">
                        Page {tagPage + 1} of {tagTotalPages}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setTagPage(p => p + 1)}
                        disabled={tagPage >= tagTotalPages - 1}
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                ) : null
              })()}
            </>
          )}
        </div>
      )}

    </div>
  )
}

// ==================== SUB-COMPONENTS ====================

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
      {isActive && (
        currentDir === 'desc'
          ? <ChevronDown className="w-3 h-3" />
          : <ChevronUp className="w-3 h-3" />
      )}
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
        {username && (
          <p className="text-sm font-medium text-foreground">{username}</p>
        )}
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
    <span className={cn(
      'font-medium font-mono text-sm',
      isPositive ? 'text-green-400' : 'text-red-400'
    )}>
      {isPositive ? '+' : ''}${formatPnl(value)}
    </span>
  )
}

function RecommendationBadge({ recommendation }: { recommendation: string }) {
  const colorClass = RECOMMENDATION_COLORS[recommendation] || 'bg-muted-foreground/15 text-muted-foreground border-muted-foreground/20'
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
  const rankDisplay = useWindowMetrics ? rank : (wallet.rank_position || rank)

  // Use period-specific metrics when a rolling window is active
  const pnl = useWindowMetrics && wallet.period_pnl != null ? wallet.period_pnl : wallet.total_pnl
  const winRate = useWindowMetrics && wallet.period_win_rate != null ? wallet.period_win_rate : wallet.win_rate
  const sharpe = useWindowMetrics ? (wallet.period_sharpe ?? wallet.sharpe_ratio) : wallet.sharpe_ratio
  const trades = useWindowMetrics && wallet.period_trades != null ? wallet.period_trades : wallet.total_trades
  const roi = useWindowMetrics && wallet.period_roi != null ? wallet.period_roi : wallet.avg_roi
  const composite = wallet.composite_score ?? wallet.rank_score ?? 0
  const activity = wallet.activity_score ?? 0
  const quality = wallet.quality_score ?? wallet.rank_score ?? 0

  return (
    <TableRow>
      {/* Rank */}
      <TableCell className="font-medium text-muted-foreground">
        <span className={cn(
          'flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold',
          rankDisplay === 1 ? 'bg-yellow-500/20 text-yellow-400' :
          rankDisplay === 2 ? 'bg-muted-foreground/20 text-muted-foreground' :
          rankDisplay === 3 ? 'bg-amber-600/20 text-amber-500' :
          'bg-muted text-muted-foreground'
        )}>
          {rankDisplay}
        </span>
      </TableCell>

      {/* Trader */}
      <TableCell>
        <WalletAddress
          address={wallet.address}
          username={wallet.username}
          copiedAddress={copiedAddress}
          onCopy={onCopyAddress}
        />
        {/* Tags inline */}
        {wallet.tags.length > 0 && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {wallet.tags.slice(0, 3).map(tag => (
              <span
                key={tag}
                className="px-1.5 py-0.5 text-[10px] rounded bg-muted text-muted-foreground border border-border"
              >
                {tag}
              </span>
            ))}
            {wallet.tags.length > 3 && (
              <span className="text-[10px] text-muted-foreground">
                +{wallet.tags.length - 3}
              </span>
            )}
          </div>
        )}
      </TableCell>

      {/* Composite */}
      <TableCell>
        <span className={cn(
          'font-mono text-sm',
          composite >= 0.7 ? 'text-green-400' : composite >= 0.5 ? 'text-yellow-400' : 'text-muted-foreground'
        )}>
          {(composite * 100).toFixed(1)}
        </span>
      </TableCell>

      {/* Activity */}
      <TableCell>
        <span className={cn(
          'font-mono text-sm',
          activity >= 0.6 ? 'text-green-400' : activity >= 0.3 ? 'text-yellow-400' : 'text-muted-foreground'
        )}>
          {(activity * 100).toFixed(1)}
        </span>
      </TableCell>

      {/* Quality */}
      <TableCell>
        <span className={cn(
          'font-mono text-sm',
          quality >= 0.6 ? 'text-green-400' : quality >= 0.4 ? 'text-yellow-400' : 'text-muted-foreground'
        )}>
          {(quality * 100).toFixed(1)}
        </span>
      </TableCell>

      {/* Last Trade */}
      <TableCell>
        <span className="text-xs text-muted-foreground">
          {timeAgo(wallet.last_trade_at || null)}
        </span>
      </TableCell>

      {/* PnL */}
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

      {/* Win Rate */}
      <TableCell>
        <div className="flex items-center gap-2">
          <span className={cn(
            'font-medium text-sm',
            winRate >= 60 ? 'text-green-400' : winRate >= 45 ? 'text-yellow-400' : 'text-red-400'
          )}>
            {formatPercent(winRate)}
          </span>
          {!useWindowMetrics && (
            <span className="text-[10px] text-muted-foreground">
              {wallet.wins}W/{wallet.losses}L
            </span>
          )}
        </div>
      </TableCell>

      {/* Sharpe */}
      <TableCell>
        {sharpe != null ? (
          <span className={cn(
            'font-mono text-sm',
            sharpe >= 2 ? 'text-green-400' : sharpe >= 1 ? 'text-yellow-400' : 'text-muted-foreground'
          )}>
            {sharpe.toFixed(2)}
          </span>
        ) : (
          <span className="text-muted-foreground text-xs">--</span>
        )}
      </TableCell>

      {/* Trades */}
      <TableCell className="text-muted-foreground text-sm">
        {trades}
        {!useWindowMetrics && (
          <span className="text-[10px] text-muted-foreground/70 ml-1">
            ({wallet.trades_per_day.toFixed(1)}/d)
          </span>
        )}
      </TableCell>

      {/* Avg ROI */}
      <TableCell>
        <span className={cn(
          'font-mono text-sm',
          roi >= 0 ? 'text-green-400' : 'text-red-400'
        )}>
          {roi >= 0 ? '+' : ''}{formatPercent(roi)}
        </span>
      </TableCell>

      {/* Tags (column) */}
      <TableCell>
        {wallet.tags.length > 0 ? (
          <div className="flex flex-wrap gap-1 max-w-[120px]">
            {wallet.tags.slice(0, 2).map(tag => (
              <span
                key={tag}
                className="px-1.5 py-0.5 text-[10px] rounded bg-muted text-muted-foreground border border-border truncate max-w-[80px]"
              >
                {tag}
              </span>
            ))}
            {wallet.tags.length > 2 && (
              <span className="text-[10px] text-muted-foreground">+{wallet.tags.length - 2}</span>
            )}
          </div>
        ) : (
          <span className="text-[10px] text-muted-foreground">--</span>
        )}
      </TableCell>

      {/* Recommendation */}
      <TableCell>
        <RecommendationBadge recommendation={wallet.recommendation} />
      </TableCell>

      {/* Actions */}
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

function ConfluenceCard({
  signal,
  onExecuteTrade,
  onAnalyzeWallet,
  onTrackWallet,
}: {
  signal: ConfluenceSignal
  onExecuteTrade?: (opportunity: Opportunity) => void
  onAnalyzeWallet?: (address: string, username?: string) => void
  onTrackWallet?: (address: string, username?: string | null) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [searchingOpportunity, setSearchingOpportunity] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const strengthPercent = Math.round(signal.strength * 100)
  const convictionScore = Math.round(signal.conviction_score ?? strengthPercent)
  const tier = (signal.tier || 'WATCH').toUpperCase()
  const signalColor = SIGNAL_TYPE_COLORS[signal.signal_type] || 'bg-muted-foreground/15 text-muted-foreground border-muted-foreground/20'
  const tierColor =
    tier === 'EXTREME'
      ? 'bg-red-500/15 text-red-400 border-red-500/20'
      : tier === 'HIGH'
        ? 'bg-orange-500/15 text-orange-400 border-orange-500/20'
        : 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20'

  const strengthBarColor =
    convictionScore >= 80 ? 'bg-green-500' :
    convictionScore >= 50 ? 'bg-yellow-500' :
    'bg-red-500'

  const polymarketMarketUrl = buildPolymarketMarketUrl({
    eventSlug: signal.market_slug,
    marketId: signal.market_id,
  })

  const handleFindOpportunity = async () => {
    if (!onExecuteTrade) return
    setSearchingOpportunity(true)
    setSearchError(null)
    try {
      const searchTerm = signal.market_question || signal.market_id
      const resp = await getOpportunities({ search: searchTerm, limit: 5 })
      if (resp.opportunities.length > 0) {
        onExecuteTrade(resp.opportunities[0])
      } else {
        setSearchError('No matching opportunity found for this signal')
      }
    } catch {
      setSearchError('Failed to search for opportunities')
    } finally {
      setSearchingOpportunity(false)
    }
  }

  return (
    <Card className={cn(
      "border-border transition-colors",
      signal.is_active ? "" : "opacity-60"
    )}>
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <Badge variant="outline" className={cn("text-xs", signalColor)}>
                {signal.signal_type.replace(/_/g, ' ')}
              </Badge>
              <Badge variant="outline" className={cn("text-xs font-semibold", tierColor)}>
                {tier}
              </Badge>
              {!signal.is_active && (
                <Badge variant="outline" className="text-[10px] bg-muted-foreground/10 text-muted-foreground border-muted-foreground/20">
                  Inactive
                </Badge>
              )}
            </div>
            {polymarketMarketUrl ? (
              <a
                href={polymarketMarketUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-foreground leading-tight line-clamp-2 hover:text-primary transition-colors inline-flex items-start gap-1 group"
              >
                <span className="group-hover:underline">
                  {signal.market_question || signal.market_id}
                </span>
                <ArrowUpRight className="w-3 h-3 mt-0.5 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
              </a>
            ) : (
              <p className="text-sm text-foreground leading-tight line-clamp-2">
                {signal.market_question || signal.market_id}
              </p>
            )}
          </div>
          <div className="text-right shrink-0">
            {signal.outcome && (
              <Badge variant="outline" className={cn(
                "text-xs font-bold",
                signal.outcome === 'YES'
                  ? 'bg-green-500/15 text-green-400 border-green-500/20'
                  : 'bg-red-500/15 text-red-400 border-red-500/20'
              )}>
                {signal.outcome}
              </Badge>
            )}
          </div>
        </div>

        {/* Strength Bar */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-muted-foreground">Conviction</span>
            <span className="text-xs font-medium">{convictionScore}/100</span>
          </div>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", strengthBarColor)}
              style={{ width: `${Math.max(0, Math.min(convictionScore, 100))}%` }}
            />
          </div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            Strength {strengthPercent}%{signal.window_minutes ? ` • ${signal.window_minutes}m window` : ''}
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-3 text-xs">
          <div>
            <p className="text-muted-foreground">Wallets</p>
            <p className="font-medium text-foreground flex items-center gap-1">
              <Users className="w-3 h-3 text-muted-foreground" />
              {signal.wallet_count}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Adj. Wallets</p>
            <p className="font-medium text-foreground">
              {signal.cluster_adjusted_wallet_count ?? signal.wallet_count}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Core Wallets</p>
            <p className="font-medium text-foreground">
              {signal.unique_core_wallets ?? 0}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Last Seen</p>
            <p className="font-medium text-foreground">
              {timeAgo(signal.last_seen_at || signal.detected_at)}
            </p>
          </div>
          {signal.avg_entry_price != null && (
            <div>
              <p className="text-muted-foreground">Avg Entry</p>
              <p className="font-mono font-medium text-foreground">
                ${signal.avg_entry_price.toFixed(3)}
              </p>
            </div>
          )}
          {signal.total_size != null && (
            <div>
              <p className="text-muted-foreground">Total Size</p>
              <p className="font-mono font-medium text-foreground">
                ${formatPnl(signal.total_size)}
              </p>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 pt-1">
          {onExecuteTrade && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleFindOpportunity}
                  disabled={searchingOpportunity}
                  className="flex items-center gap-1.5 text-xs bg-green-500/10 text-green-400 border-green-500/20 hover:bg-green-500/20 hover:text-green-400"
                >
                  {searchingOpportunity ? (
                    <RefreshCw className="w-3 h-3 animate-spin" />
                  ) : (
                    <DollarSign className="w-3 h-3" />
                  )}
                  Execute Trade
                </Button>
              </TooltipTrigger>
              <TooltipContent>Find matching opportunity and execute trade</TooltipContent>
            </Tooltip>
          )}
          {polymarketMarketUrl && (
            <Tooltip>
              <TooltipTrigger asChild>
                <a
                  href={polymarketMarketUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs border bg-muted/50 text-muted-foreground hover:text-foreground border-border hover:border-border transition-colors"
                >
                  <ExternalLink className="w-3 h-3" />
                  Polymarket
                </a>
              </TooltipTrigger>
              <TooltipContent>View market on Polymarket</TooltipContent>
            </Tooltip>
          )}
          <div className="flex-1" />
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => setExpanded(!expanded)}
                className={cn(
                  "p-1.5 rounded text-muted-foreground hover:text-foreground transition-colors",
                  expanded && "bg-muted"
                )}
              >
                {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            </TooltipTrigger>
            <TooltipContent>{expanded ? 'Hide wallets' : 'Show participating wallets'}</TooltipContent>
          </Tooltip>
        </div>

        {/* Search Error Feedback */}
        {searchError && (
          <div className="flex items-center gap-2 text-xs bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg px-3 py-2">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            <span>{searchError}</span>
          </div>
        )}

        {/* Expanded Wallet List */}
        {expanded && signal.wallets.length > 0 && (
          <div className="pt-2 border-t border-border space-y-1.5">
            <p className="text-[10px] text-muted-foreground font-medium">
              Participating Wallets ({signal.wallets.length})
            </p>
            {signal.wallets.map(address => (
              <div
                key={address}
                className="flex items-center justify-between bg-muted/30 px-2.5 py-1.5 rounded-lg"
              >
                <span className="font-mono text-xs text-muted-foreground">
                  {truncateAddress(address)}
                </span>
                <div className="flex items-center gap-1">
                  {onAnalyzeWallet && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          onClick={() => onAnalyzeWallet(address)}
                          className="p-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                        >
                          <Activity className="w-3 h-3" />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent>Analyze wallet</TooltipContent>
                    </Tooltip>
                  )}
                  {onTrackWallet && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          onClick={() => onTrackWallet(address)}
                          className="p-1 rounded bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors"
                        >
                          <UserPlus className="w-3 h-3" />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent>Track wallet</TooltipContent>
                    </Tooltip>
                  )}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <a
                        href={`https://polymarket.com/profile/${address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
                      >
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </TooltipTrigger>
                    <TooltipContent>View on Polymarket</TooltipContent>
                  </Tooltip>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Detected Time */}
        <div className="text-[10px] text-muted-foreground pt-1 border-t border-border flex items-center justify-between">
          <span>Detected {timeAgo(signal.detected_at)}</span>
          {signal.avg_wallet_rank != null && (
            <span>Avg Rank Score: {(signal.avg_wallet_rank * 100).toFixed(0)}</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
