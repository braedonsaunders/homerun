import { useState, useEffect, useCallback, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Trophy,
  RefreshCw,
  Users,
  Target,
  Tag,
  Layers,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  Copy,
  Search,
  Play,
  TrendingUp,
  Eye,
  Zap,
  ExternalLink,
  Activity,
  UserPlus,
  DollarSign,
  ArrowUpRight,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Separator } from './ui/separator'
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
  type WalletCluster,
  type TagInfo,
  type DiscoveryStats,
} from '../services/discoveryApi'
import { analyzeAndTrackWallet, type Opportunity, getOpportunities } from '../services/api'

// ==================== TYPES ====================

type DiscoveryTab = 'leaderboard' | 'confluence' | 'tags' | 'clusters'

type SortField = 'rank_score' | 'total_pnl' | 'win_rate' | 'sharpe_ratio' | 'total_trades' | 'avg_roi'

type SortDir = 'asc' | 'desc'

type RecommendationFilter = '' | 'copy_candidate' | 'monitor' | 'avoid'

// ==================== CONSTANTS ====================

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
  parentTab?: 'leaderboard' | 'discover'
  onAnalyzeWallet?: (address: string, username?: string) => void
  onExecuteTrade?: (opportunity: Opportunity) => void
}

export default function DiscoveryPanel({ parentTab = 'leaderboard', onAnalyzeWallet, onExecuteTrade }: DiscoveryPanelProps) {
  const [discoverSubTab, setDiscoverSubTab] = useState<'confluence' | 'tags' | 'clusters'>('confluence')
  const activeTab: DiscoveryTab = parentTab === 'leaderboard' ? 'leaderboard' : discoverSubTab
  const [sortBy, setSortBy] = useState<SortField>('rank_score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [currentPage, setCurrentPage] = useState(0)
  const [minTrades, setMinTrades] = useState(10)
  const [minPnl, setMinPnl] = useState(0)
  const [recommendationFilter, setRecommendationFilter] = useState<RecommendationFilter>('')
  const [tagFilter, setTagFilter] = useState('')
  const [selectedTag, setSelectedTag] = useState<string | null>(null)
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null)
  const [confluenceMinStrength, setConfluenceMinStrength] = useState(0)
  const [clusterMinWallets, setClusterMinWallets] = useState(2)
  const queryClient = useQueryClient()
  const refreshTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Auto-refresh every 60 seconds
  useEffect(() => {
    refreshTimerRef.current = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-stats'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-confluence'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-clusters'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-tags'] })
    }, 60000)

    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current)
    }
  }, [queryClient])

  // Reset page on filter changes
  useEffect(() => {
    setCurrentPage(0)
  }, [sortBy, sortDir, minTrades, minPnl, recommendationFilter, tagFilter])

  // ==================== QUERIES ====================

  const { data: stats } = useQuery<DiscoveryStats>({
    queryKey: ['discovery-stats'],
    queryFn: discoveryApi.getDiscoveryStats,
    refetchInterval: 30000,
  })

  const { data: leaderboardData, isLoading: leaderboardLoading } = useQuery({
    queryKey: ['discovery-leaderboard', sortBy, sortDir, currentPage, minTrades, minPnl, recommendationFilter, tagFilter],
    queryFn: () => discoveryApi.getLeaderboard({
      sort_by: sortBy,
      sort_dir: sortDir,
      limit: ITEMS_PER_PAGE,
      offset: currentPage * ITEMS_PER_PAGE,
      min_trades: minTrades,
      min_pnl: minPnl || undefined,
      recommendation: recommendationFilter || undefined,
      tags: tagFilter || undefined,
    }),
    enabled: activeTab === 'leaderboard',
  })

  const wallets: DiscoveredWallet[] = leaderboardData?.wallets || leaderboardData || []
  const totalWallets: number = leaderboardData?.total || wallets.length

  const { data: confluenceSignals = [], isLoading: confluenceLoading } = useQuery<ConfluenceSignal[]>({
    queryKey: ['discovery-confluence', confluenceMinStrength],
    queryFn: () => discoveryApi.getConfluenceSignals(confluenceMinStrength, 50),
    enabled: activeTab === 'confluence',
  })

  const { data: clusters = [], isLoading: clustersLoading } = useQuery<WalletCluster[]>({
    queryKey: ['discovery-clusters', clusterMinWallets],
    queryFn: () => discoveryApi.getClusters(clusterMinWallets),
    enabled: activeTab === 'clusters',
  })

  const { data: tags = [], isLoading: tagsLoading } = useQuery<TagInfo[]>({
    queryKey: ['discovery-tags'],
    queryFn: discoveryApi.getTags,
    enabled: activeTab === 'tags',
  })

  const { data: tagWalletsData, isLoading: tagWalletsLoading } = useQuery({
    queryKey: ['discovery-tag-wallets', selectedTag],
    queryFn: () => discoveryApi.getWalletsByTag(selectedTag!, 100),
    enabled: !!selectedTag && activeTab === 'tags',
  })

  const tagWallets: DiscoveredWallet[] = tagWalletsData?.wallets || []

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
    { id: 'clusters', icon: Layers, label: 'Clusters', color: 'blue' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">
            {parentTab === 'leaderboard' ? 'Leaderboard' : 'Trader Discovery'}
          </h2>
          <p className="text-sm text-muted-foreground">
            {parentTab === 'leaderboard'
              ? 'Top performing traders ranked by metrics'
              : 'Discover confluence signals, wallet clusters, and behavioral tags'}
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
      <div className="grid grid-cols-4 gap-4">
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
      </div>

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

      {/* Tab Navigation - only show sub-tabs in discover mode */}
      {parentTab === 'discover' && (
        <div className="flex items-center gap-2">
          {tabDefs.filter(tab => tab.id !== 'leaderboard').map(tab => {
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
              blue: isActive
                ? 'bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400'
                : 'bg-card text-muted-foreground hover:text-foreground border-border',
            }
            return (
              <Button
                key={tab.id}
                variant="outline"
                size="sm"
                onClick={() => setDiscoverSubTab(tab.id as 'confluence' | 'tags' | 'clusters')}
                className={cn("flex items-center gap-2", colorMap[tab.color])}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </Button>
            )
          })}
        </div>
      )}

      {/* ==================== LEADERBOARD TAB ==================== */}
      {activeTab === 'leaderboard' && (
        <div className="space-y-4">
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
                          field="total_pnl"
                          label="PnL"
                          currentSort={sortBy}
                          currentDir={sortDir}
                          onSort={handleSort}
                        />
                      </TableHead>
                      <TableHead>
                        <SortButton
                          field="win_rate"
                          label="Win Rate"
                          currentSort={sortBy}
                          currentDir={sortDir}
                          onSort={handleSort}
                        />
                      </TableHead>
                      <TableHead>
                        <SortButton
                          field="sharpe_ratio"
                          label="Sharpe"
                          currentSort={sortBy}
                          currentDir={sortDir}
                          onSort={handleSort}
                        />
                      </TableHead>
                      <TableHead>
                        <SortButton
                          field="total_trades"
                          label="Trades"
                          currentSort={sortBy}
                          currentDir={sortDir}
                          onSort={handleSort}
                        />
                      </TableHead>
                      <TableHead>
                        <SortButton
                          field="avg_roi"
                          label="Avg ROI"
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
          {tagsLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : tags.length === 0 ? (
            <Card className="border-border">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Tag className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">No tags available</p>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  Tags are generated automatically during discovery analysis
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Tag Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {tags.map(tag => (
                  <TagCard
                    key={tag.name}
                    tag={tag}
                    isSelected={selectedTag === tag.name}
                    onSelect={() => setSelectedTag(selectedTag === tag.name ? null : tag.name)}
                  />
                ))}
              </div>

              {/* Selected Tag Wallets */}
              {selectedTag && (
                <>
                  <Separator />
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-medium flex items-center gap-2">
                        <Tag className="w-4 h-4 text-muted-foreground" />
                        Wallets tagged &quot;{selectedTag}&quot;
                      </h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedTag(null)}
                        className="text-xs text-muted-foreground"
                      >
                        Clear
                      </Button>
                    </div>

                    {tagWalletsLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : tagWallets.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No wallets found with this tag
                      </p>
                    ) : (
                      <Card className="border-border overflow-hidden">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Trader</TableHead>
                              <TableHead>PnL</TableHead>
                              <TableHead>Win Rate</TableHead>
                              <TableHead>Trades</TableHead>
                              <TableHead>Recommendation</TableHead>
                              <TableHead>Actions</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {tagWallets.map(wallet => (
                              <TableRow key={wallet.address}>
                                <TableCell>
                                  <WalletAddress
                                    address={wallet.address}
                                    username={wallet.username}
                                    copiedAddress={copiedAddress}
                                    onCopy={handleCopyAddress}
                                  />
                                </TableCell>
                                <TableCell>
                                  <PnlDisplay value={wallet.total_pnl} />
                                </TableCell>
                                <TableCell>
                                  <span className={cn(
                                    "font-medium",
                                    wallet.win_rate >= 60 ? 'text-green-400' : wallet.win_rate >= 45 ? 'text-yellow-400' : 'text-red-400'
                                  )}>
                                    {formatPercent(wallet.win_rate)}
                                  </span>
                                </TableCell>
                                <TableCell className="text-muted-foreground">
                                  {wallet.total_trades}
                                </TableCell>
                                <TableCell>
                                  <RecommendationBadge recommendation={wallet.recommendation} />
                                </TableCell>
                                <TableCell>
                                  <div className="flex items-center gap-1">
                                    {onAnalyzeWallet && (
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button
                                            onClick={() => onAnalyzeWallet(wallet.address, wallet.username || undefined)}
                                            className="p-1.5 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                                          >
                                            <Activity className="w-3.5 h-3.5" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>Analyze wallet</TooltipContent>
                                      </Tooltip>
                                    )}
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <button
                                          onClick={() => trackWalletMutation.mutate({ address: wallet.address, username: wallet.username })}
                                          disabled={trackWalletMutation.isPending}
                                          className="p-1.5 rounded bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors disabled:opacity-50"
                                        >
                                          <UserPlus className="w-3.5 h-3.5" />
                                        </button>
                                      </TooltipTrigger>
                                      <TooltipContent>Track wallet</TooltipContent>
                                    </Tooltip>
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
                            ))}
                          </TableBody>
                        </Table>
                      </Card>
                    )}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* ==================== CLUSTERS TAB ==================== */}
      {activeTab === 'clusters' && (
        <div className="space-y-4">
          {/* Controls */}
          <div className="flex items-center gap-4">
            <div className="w-48">
              <label className="block text-xs text-muted-foreground mb-1">Min Wallets per Cluster</label>
              <Input
                type="number"
                value={clusterMinWallets}
                onChange={e => setClusterMinWallets(parseInt(e.target.value) || 2)}
                min={2}
                max={50}
                className="bg-card border-border h-8 text-sm"
              />
            </div>
          </div>

          {/* Clusters */}
          {clustersLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : clusters.length === 0 ? (
            <Card className="border-border">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Layers className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">No wallet clusters found</p>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  Clusters are formed when wallets exhibit similar trading patterns
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {clusters.map(cluster => (
                <ClusterCard
                  key={cluster.id}
                  cluster={cluster}
                  copiedAddress={copiedAddress}
                  onCopyAddress={handleCopyAddress}
                />
              ))}
            </div>
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
}: {
  wallet: DiscoveredWallet
  rank: number
  copiedAddress: string | null
  onCopyAddress: (address: string) => void
  onAnalyze?: (address: string, username?: string) => void
  onTrack?: (address: string, username?: string | null) => void
  isTracking?: boolean
}) {
  const rankDisplay = wallet.rank_position || rank

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

      {/* PnL */}
      <TableCell>
        <PnlDisplay value={wallet.total_pnl} />
      </TableCell>

      {/* Win Rate */}
      <TableCell>
        <div className="flex items-center gap-2">
          <span className={cn(
            'font-medium text-sm',
            wallet.win_rate >= 60 ? 'text-green-400' : wallet.win_rate >= 45 ? 'text-yellow-400' : 'text-red-400'
          )}>
            {formatPercent(wallet.win_rate)}
          </span>
          <span className="text-[10px] text-muted-foreground">
            {wallet.wins}W/{wallet.losses}L
          </span>
        </div>
      </TableCell>

      {/* Sharpe */}
      <TableCell>
        {wallet.sharpe_ratio != null ? (
          <span className={cn(
            'font-mono text-sm',
            wallet.sharpe_ratio >= 2 ? 'text-green-400' : wallet.sharpe_ratio >= 1 ? 'text-yellow-400' : 'text-muted-foreground'
          )}>
            {wallet.sharpe_ratio.toFixed(2)}
          </span>
        ) : (
          <span className="text-muted-foreground text-xs">--</span>
        )}
      </TableCell>

      {/* Trades */}
      <TableCell className="text-muted-foreground text-sm">
        {wallet.total_trades}
        <span className="text-[10px] text-muted-foreground/70 ml-1">
          ({wallet.trades_per_day.toFixed(1)}/d)
        </span>
      </TableCell>

      {/* Avg ROI */}
      <TableCell>
        <span className={cn(
          'font-mono text-sm',
          wallet.avg_roi >= 0 ? 'text-green-400' : 'text-red-400'
        )}>
          {wallet.avg_roi >= 0 ? '+' : ''}{formatPercent(wallet.avg_roi)}
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
  const signalColor = SIGNAL_TYPE_COLORS[signal.signal_type] || 'bg-muted-foreground/15 text-muted-foreground border-muted-foreground/20'

  const strengthBarColor =
    strengthPercent >= 80 ? 'bg-green-500' :
    strengthPercent >= 50 ? 'bg-yellow-500' :
    'bg-red-500'

  const polymarketMarketUrl = signal.market_slug
    ? `https://polymarket.com/event/${signal.market_slug}`
    : null

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
            <span className="text-[10px] text-muted-foreground">Strength</span>
            <span className="text-xs font-medium">{strengthPercent}%</span>
          </div>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", strengthBarColor)}
              style={{ width: `${strengthPercent}%` }}
            />
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div>
            <p className="text-muted-foreground">Wallets</p>
            <p className="font-medium text-foreground flex items-center gap-1">
              <Users className="w-3 h-3 text-muted-foreground" />
              {signal.wallet_count}
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

function TagCard({
  tag,
  isSelected,
  onSelect,
}: {
  tag: TagInfo
  isSelected: boolean
  onSelect: () => void
}) {
  // Parse the tag color and generate a tailwind-compatible style
  const tagColorStyle = tag.color ? { borderColor: tag.color, backgroundColor: `${tag.color}15` } : {}
  const tagTextStyle = tag.color ? { color: tag.color } : {}

  return (
    <Card
      className={cn(
        "border-border cursor-pointer transition-all hover:border-primary/30",
        isSelected && "ring-1 ring-primary/50 border-primary/30"
      )}
      onClick={onSelect}
    >
      <CardContent className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span
            className="px-2 py-0.5 rounded-full text-xs font-semibold border"
            style={{ ...tagColorStyle, ...tagTextStyle }}
          >
            {tag.display_name || tag.name}
          </span>
          <span className="text-xs text-muted-foreground font-medium">
            {tag.wallet_count}
          </span>
        </div>
        {tag.description && (
          <p className="text-[10px] text-muted-foreground line-clamp-2">
            {tag.description}
          </p>
        )}
        <div className="flex items-center justify-between">
          <Badge variant="outline" className="text-[10px] bg-muted text-muted-foreground border-border">
            {tag.category}
          </Badge>
          {isSelected && (
            <Eye className="w-3 h-3 text-primary" />
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function ClusterCard({
  cluster,
  copiedAddress,
  onCopyAddress,
}: {
  cluster: WalletCluster
  copiedAddress: string | null
  onCopyAddress: (address: string) => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card className="border-border">
      <CardContent className="p-4">
        {/* Cluster Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Layers className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-medium text-foreground">
                  {cluster.label || `Cluster ${cluster.id.slice(0, 8)}`}
                </h3>
                <Badge variant="outline" className="text-[10px]">
                  {cluster.total_wallets} wallets
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Confidence: {(cluster.confidence * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Stats inline */}
            <div className="flex items-center gap-3 text-xs">
              <div className="text-right">
                <p className="text-muted-foreground">Combined PnL</p>
                <PnlDisplay value={cluster.combined_pnl} />
              </div>
              <div className="text-right">
                <p className="text-muted-foreground">Avg Win Rate</p>
                <p className={cn(
                  'font-medium',
                  cluster.avg_win_rate >= 60 ? 'text-green-400' : cluster.avg_win_rate >= 45 ? 'text-yellow-400' : 'text-red-400'
                )}>
                  {formatPercent(cluster.avg_win_rate)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-muted-foreground">Trades</p>
                <p className="font-medium text-foreground">{formatNumber(cluster.combined_trades)}</p>
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(!expanded)}
              className="px-2"
            >
              {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </div>
        </div>

        {/* Expanded Members */}
        {expanded && cluster.wallets && cluster.wallets.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-xs text-muted-foreground mb-3">Cluster Members</p>
            <div className="space-y-2">
              {cluster.wallets.map(wallet => (
                <div
                  key={wallet.address}
                  className="flex items-center justify-between bg-muted/30 px-3 py-2 rounded-lg"
                >
                  <WalletAddress
                    address={wallet.address}
                    username={wallet.username}
                    copiedAddress={copiedAddress}
                    onCopy={onCopyAddress}
                  />
                  <div className="flex items-center gap-4 text-xs">
                    <PnlDisplay value={wallet.total_pnl} />
                    <span className={cn(
                      wallet.win_rate >= 60 ? 'text-green-400' : wallet.win_rate >= 45 ? 'text-yellow-400' : 'text-red-400'
                    )}>
                      {formatPercent(wallet.win_rate)}
                    </span>
                    <span className="text-muted-foreground">{wallet.total_trades} trades</span>
                    <RecommendationBadge recommendation={wallet.recommendation} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
