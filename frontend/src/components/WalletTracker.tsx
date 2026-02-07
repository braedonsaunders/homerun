import { useState } from 'react'
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
import {
  getWallets,
  addWallet,
  removeWallet,
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
}

export default function WalletTracker({ onAnalyzeWallet, section: propSection, discoverMode: propDiscoverMode }: WalletTrackerProps) {
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

  const { data: wallets = [], isLoading } = useQuery({
    queryKey: ['wallets'],
    queryFn: getWallets,
    refetchInterval: 30000,
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
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
      setNewAddress('')
      setNewLabel('')
    },
  })

  const removeMutation = useMutation({
    mutationFn: removeWallet,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
    },
  })

  const trackAndCopyMutation = useMutation({
    mutationFn: (params: { address: string; label?: string; auto_copy?: boolean; simulation_account_id?: string }) =>
      analyzeAndTrackWallet(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
    },
  })

  const handleAdd = () => {
    if (!newAddress.trim()) return
    addMutation.mutate({ address: newAddress.trim(), label: newLabel.trim() || undefined })
  }

  const handleAnalyze = (address: string, username?: string) => {
    if (onAnalyzeWallet) {
      onAnalyzeWallet(address, username)
    }
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
      // Paper mode: track and copy to simulation account
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
                    showFilters ? "bg-blue-500/20 text-blue-400" : "bg-muted hover:bg-gray-700"
                  )}
                >
                  <Filter className="w-4 h-4" />
                  Filters
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => refreshCurrentTraders()}
                  disabled={isLoadingTraders}
                  className="h-auto gap-2 px-3 py-1.5 bg-muted rounded-lg text-sm hover:bg-gray-700"
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
                    className="flex items-center justify-between p-3 rounded-lg transition-colors bg-muted hover:bg-[#222]"
                  >
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold",
                        currentDiscoverMode === 'winrate' ? "bg-emerald-900" : "bg-gray-700"
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
                        className="p-1 hover:bg-gray-700 rounded"
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
          <div>
            <h3 className="text-lg font-medium mb-4">Tracked Wallets</h3>

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
              <div className="space-y-3">
                {wallets.map((wallet) => (
                  <WalletCard
                    key={wallet.address}
                    wallet={wallet}
                    onRemove={() => removeMutation.mutate(wallet.address)}
                    onAnalyze={() => handleAnalyze(wallet.address, wallet.username)}
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* Copy Trade Account Selection Modal */}
      {showCopyModal && selectedTrader && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
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
                className="h-auto p-2 hover:bg-gray-700 rounded-lg"
              >
                <X className="w-5 h-5 text-muted-foreground" />
              </Button>
            </div>

            {/* Modal Content */}
            <div className="p-4 space-y-4">
              <p className="text-sm text-gray-300">
                Choose how you want to copy trades from this trader:
              </p>

              {/* Paper Trading Option */}
              <div className="space-y-3">
                <div
                  className={cn(
                    "p-4 rounded-lg border-2 cursor-pointer transition-all",
                    simAccounts.length > 0
                      ? "border-blue-500/50 bg-blue-500/10 hover:bg-blue-500/20"
                      : "border-border bg-gray-800/50 opacity-60 cursor-not-allowed"
                  )}
                  onClick={() => simAccounts.length > 0 && handleCopyTradeConfirm(true)}
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-blue-400" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-blue-400">Paper Trading</h4>
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
                          No paper accounts available. Create one in the Accounts tab first.
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
                className="w-full h-auto py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium"
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

function WalletCard({ wallet, onRemove, onAnalyze }: { wallet: WalletType; onRemove: () => void; onAnalyze: () => void }) {
  const displayName = wallet.username || wallet.label
  const hasUsername = !!wallet.username
  const posCount = wallet.positions?.length || 0
  const tradeCount = wallet.recent_trades?.length || 0

  return (
    <Card className="p-4 hover:border-border transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
            <Wallet className="w-5 h-5 text-purple-500" />
          </div>
          <div>
            <p className="font-medium text-foreground">{displayName}</p>
            {hasUsername && wallet.label !== wallet.username && (
              <p className="text-xs text-muted-foreground">{wallet.label}</p>
            )}
            <p className="text-xs text-muted-foreground font-mono">
              {wallet.address.slice(0, 6)}...{wallet.address.slice(-4)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-4 mr-2">
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Positions</p>
              <p className="text-sm font-medium text-foreground">{posCount}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Trades</p>
              <p className="text-sm font-medium text-foreground">{tradeCount}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            onClick={onAnalyze}
            className="h-auto gap-1.5 px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg text-xs font-medium"
          >
            <Activity className="w-3.5 h-3.5" />
            Analyze
          </Button>
          <a
            href={`https://polymarket.com/profile/${wallet.address}`}
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            title="View on Polymarket"
          >
            <ExternalLink className="w-4 h-4 text-muted-foreground" />
          </a>
          <Button
            variant="ghost"
            onClick={onRemove}
            className="h-auto p-2 hover:bg-red-500/10 rounded-lg text-red-400"
            title="Remove wallet"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </Card>
  )
}
