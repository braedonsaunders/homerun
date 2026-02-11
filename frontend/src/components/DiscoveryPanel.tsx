import { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Trophy,
  RefreshCw,
  Users,
  Target,
  Tag,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Copy,
  Search,
  TrendingUp,
  ExternalLink,
  Activity,
  UserPlus,
  Clock,
  PauseCircle,
  AlertTriangle,
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
}

export default function DiscoveryPanel({ onAnalyzeWallet }: DiscoveryPanelProps) {
  const [sortBy, setSortBy] = useState<SortField>('rank_score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [currentPage, setCurrentPage] = useState(0)
  const [minTrades, setMinTrades] = useState(0)
  const [minPnl, setMinPnl] = useState(0)
  const [recommendationFilter, setRecommendationFilter] = useState<RecommendationFilter>('')
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('24h')
  const [insiderOnly, setInsiderOnly] = useState(false)
  const [minInsiderScore, setMinInsiderScore] = useState(0)
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [tagSearch, setTagSearch] = useState('')

  const queryClient = useQueryClient()

  useEffect(() => {
    setCurrentPage(0)
  }, [
    sortBy,
    sortDir,
    minTrades,
    minPnl,
    recommendationFilter,
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
  })

  const { data: tags = [], isLoading: tagsLoading } = useQuery<TagInfo[]>({
    queryKey: ['discovery-tags'],
    queryFn: discoveryApi.getTags,
    refetchInterval: 120000,
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
        tags: selectedTagString || undefined,
        time_period: timePeriod !== 'all' ? timePeriod : undefined,
      }),
    refetchInterval: 30000,
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

  const totalPages = Math.ceil(totalWallets / ITEMS_PER_PAGE)

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
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Trader Discovery</h2>
          <p className="text-sm text-muted-foreground">
            Worker-driven leaderboard with behavioral tags for copy-trading candidates
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {statusBadge}
          {stats?.current_activity && (
            <span className="max-w-[320px] truncate">{stats.current_activity}</span>
          )}
          {stats?.last_run_at && <span>Last run: {timeAgo(stats.last_run_at)}</span>}
        </div>
      </div>

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
              <Trophy className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Analyzed (last run)</p>
              <p className="text-lg font-semibold">{formatNumber(stats?.wallets_analyzed_last_run || 0)}</p>
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

      <div className="space-y-4">
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

        <Card className="border-border">
          <CardContent className="p-3 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Tag className="w-3.5 h-3.5" />
                <span>Behavioral tags</span>
              </div>
              {selectedTags.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs text-muted-foreground"
                  onClick={() => setSelectedTags([])}
                >
                  Clear tags
                </Button>
              )}
            </div>

            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
              <Input
                type="text"
                value={tagSearch}
                onChange={e => setTagSearch(e.target.value)}
                placeholder="Search tags"
                className="bg-card border-border h-8 text-sm pl-8"
              />
            </div>

            {tagsLoading ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Loading tags...
              </div>
            ) : (
              <div className="flex items-center gap-2 flex-wrap">
                {filteredTags.map(tag => {
                  const isActive = selectedTags.includes(tag.name)
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
                        'px-2.5 py-1 rounded-full text-xs font-medium border transition-all',
                        isActive
                          ? 'ring-1 ring-primary/30'
                          : 'hover:ring-1 hover:ring-primary/20 opacity-75 hover:opacity-100'
                      )}
                      style={tagStyle}
                    >
                      {tag.display_name || tag.name}
                      <span className="ml-1.5 opacity-60">{tag.wallet_count}</span>
                    </button>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

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
          <div className="w-44">
            <label className="block text-xs text-muted-foreground mb-1">Min Insider Score</label>
            <Input
              type="number"
              value={minInsiderScore}
              onChange={e => setMinInsiderScore(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
              step={0.05}
              min={0}
              max={1}
              className="bg-card border-border h-8 text-sm"
            />
          </div>
          <label className="flex items-end gap-2 pb-1 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={insiderOnly}
              onChange={e => setInsiderOnly(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border"
            />
            Insider only
          </label>
        </div>

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
