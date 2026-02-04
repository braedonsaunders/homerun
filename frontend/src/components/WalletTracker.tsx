import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  Trash2,
  Wallet,
  ExternalLink,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Search,
  Star,
  Copy,
  UserPlus,
  Activity,
  Filter,
  Trophy,
  Target
} from 'lucide-react'
import clsx from 'clsx'
import {
  getWallets,
  addWallet,
  removeWallet,
  Wallet as WalletType,
  discoverTopTraders,
  discoverByWinRate,
  analyzeAndTrackWallet,
  getSimulationAccounts,
  TimePeriod,
  OrderBy,
  Category
} from '../services/api'

interface WalletTrackerProps {
  onAnalyzeWallet?: (address: string) => void
}

export default function WalletTracker({ onAnalyzeWallet }: WalletTrackerProps) {
  const [newAddress, setNewAddress] = useState('')
  const [newLabel, setNewLabel] = useState('')
  const [activeSection, setActiveSection] = useState<'tracked' | 'discover'>('discover')
  const [discoverMode, setDiscoverMode] = useState<'leaderboard' | 'winrate'>('leaderboard')
  const [showFilters, setShowFilters] = useState(false)

  // Filter states
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('ALL')
  const [orderBy, setOrderBy] = useState<OrderBy>('PNL')
  const [category, setCategory] = useState<Category>('OVERALL')
  const [minWinRate, setMinWinRate] = useState(70)
  const [minTrades, setMinTrades] = useState(10)

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
    enabled: discoverMode === 'leaderboard',
  })

  // Win rate discovery query
  const { data: winRateTraders = [], isLoading: loadingWinRate, refetch: refreshWinRate } = useQuery({
    queryKey: ['win-rate-traders', minWinRate, minTrades, timePeriod, category],
    queryFn: () => discoverByWinRate({
      min_win_rate: minWinRate,
      min_trades: minTrades,
      limit: 30,
      time_period: timePeriod,
      category
    }),
    refetchInterval: 120000,
    enabled: discoverMode === 'winrate',
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

  const handleAnalyze = (address: string) => {
    if (onAnalyzeWallet) {
      onAnalyzeWallet(address)
    }
  }

  const handleTrackAndCopy = (address: string, autoCopy: boolean = false) => {
    const allTraders = discoverMode === 'winrate' ? winRateTraders : discoveredTraders
    const trader = allTraders.find(t => t.address === address)
    const winRateStr = trader?.win_rate ? ` | ${trader.win_rate.toFixed(1)}% WR` : ''
    const label = `Discovered Trader (${trader?.volume?.toFixed(0) || '?'} vol${winRateStr})`
    trackAndCopyMutation.mutate({
      address,
      label,
      auto_copy: autoCopy,
      simulation_account_id: autoCopy && simAccounts.length > 0 ? simAccounts[0].id : undefined
    })
  }

  const currentTraders = discoverMode === 'winrate' ? winRateTraders : discoveredTraders
  const isLoadingTraders = discoverMode === 'winrate' ? loadingWinRate : discoveringTraders
  const refreshCurrentTraders = discoverMode === 'winrate' ? refreshWinRate : refreshTraders

  return (
    <div className="space-y-6">
      {/* Section Tabs */}
      <div className="flex gap-2">
        <button
          onClick={() => setActiveSection('discover')}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
            activeSection === 'discover'
              ? "bg-green-500/20 text-green-400 border border-green-500/50"
              : "bg-[#1a1a1a] text-gray-400 hover:text-white"
          )}
        >
          <Search className="w-4 h-4" />
          Discover Top Traders
        </button>
        <button
          onClick={() => setActiveSection('tracked')}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
            activeSection === 'tracked'
              ? "bg-blue-500/20 text-blue-400 border border-blue-500/50"
              : "bg-[#1a1a1a] text-gray-400 hover:text-white"
          )}
        >
          <Wallet className="w-4 h-4" />
          Tracked Wallets ({wallets.length})
        </button>
      </div>

      {activeSection === 'discover' && (
        <>
          {/* Discovery Mode Toggle */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setDiscoverMode('leaderboard')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                discoverMode === 'leaderboard'
                  ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/50"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white"
              )}
            >
              <Trophy className="w-4 h-4" />
              Leaderboard
            </button>
            <button
              onClick={() => setDiscoverMode('winrate')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                discoverMode === 'winrate'
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white"
              )}
            >
              <Target className="w-4 h-4" />
              High Win Rate
            </button>
          </div>

          {/* Discovery Header */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-lg font-medium flex items-center gap-2">
                  {discoverMode === 'winrate' ? (
                    <>
                      <Target className="w-5 h-5 text-emerald-500" />
                      High Win Rate Traders
                    </>
                  ) : (
                    <>
                      <Star className="w-5 h-5 text-yellow-500" />
                      Top Active Traders
                    </>
                  )}
                </h3>
                <p className="text-sm text-gray-500">
                  {discoverMode === 'winrate'
                    ? `Traders with ${minWinRate}%+ win rate (analyzing trade history...)`
                    : 'Discovered from Polymarket leaderboard'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={clsx(
                    "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm",
                    showFilters ? "bg-blue-500/20 text-blue-400" : "bg-[#1a1a1a] hover:bg-gray-700"
                  )}
                >
                  <Filter className="w-4 h-4" />
                  Filters
                </button>
                <button
                  onClick={() => refreshCurrentTraders()}
                  disabled={isLoadingTraders}
                  className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1a1a] rounded-lg text-sm hover:bg-gray-700"
                >
                  <RefreshCw className={clsx("w-4 h-4", isLoadingTraders && "animate-spin")} />
                  Refresh
                </button>
              </div>
            </div>

            {/* Filters Panel */}
            {showFilters && (
              <div className="mb-4 p-3 bg-[#1a1a1a] rounded-lg space-y-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Time Period</label>
                    <select
                      value={timePeriod}
                      onChange={(e) => setTimePeriod(e.target.value as TimePeriod)}
                      className="w-full bg-[#222] border border-gray-700 rounded px-2 py-1.5 text-sm"
                    >
                      <option value="ALL">All Time</option>
                      <option value="MONTH">Last 30 Days</option>
                      <option value="WEEK">Last 7 Days</option>
                      <option value="DAY">Last 24 Hours</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Category</label>
                    <select
                      value={category}
                      onChange={(e) => setCategory(e.target.value as Category)}
                      className="w-full bg-[#222] border border-gray-700 rounded px-2 py-1.5 text-sm"
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
                  {discoverMode === 'leaderboard' && (
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Sort By</label>
                      <select
                        value={orderBy}
                        onChange={(e) => setOrderBy(e.target.value as OrderBy)}
                        className="w-full bg-[#222] border border-gray-700 rounded px-2 py-1.5 text-sm"
                      >
                        <option value="PNL">Profit/Loss</option>
                        <option value="VOL">Volume</option>
                      </select>
                    </div>
                  )}
                  {discoverMode === 'winrate' && (
                    <>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Min Win Rate</label>
                        <select
                          value={minWinRate}
                          onChange={(e) => setMinWinRate(Number(e.target.value))}
                          className="w-full bg-[#222] border border-gray-700 rounded px-2 py-1.5 text-sm"
                        >
                          <option value={50}>50%+</option>
                          <option value={60}>60%+</option>
                          <option value={70}>70%+</option>
                          <option value={80}>80%+</option>
                          <option value={90}>90%+</option>
                          <option value={95}>95%+</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Min Trades</label>
                        <select
                          value={minTrades}
                          onChange={(e) => setMinTrades(Number(e.target.value))}
                          className="w-full bg-[#222] border border-gray-700 rounded px-2 py-1.5 text-sm"
                        >
                          <option value={5}>5+ trades</option>
                          <option value={10}>10+ trades</option>
                          <option value={20}>20+ trades</option>
                          <option value={50}>50+ trades</option>
                          <option value={100}>100+ trades</option>
                        </select>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {isLoadingTraders ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="w-6 h-6 animate-spin text-gray-500" />
                <span className="ml-2 text-gray-500">
                  {discoverMode === 'winrate' ? 'Analyzing win rates...' : 'Scanning Polymarket trades...'}
                </span>
              </div>
            ) : currentTraders.length === 0 ? (
              <p className="text-center text-gray-500 py-8">
                {discoverMode === 'winrate'
                  ? `No traders found with ${minWinRate}%+ win rate. Try lowering the threshold.`
                  : 'No traders discovered yet'}
              </p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {currentTraders.map((trader, idx) => (
                  <div
                    key={trader.address}
                    className="flex items-center justify-between p-3 rounded-lg transition-colors bg-[#1a1a1a] hover:bg-[#222]"
                  >
                    <div className="flex items-center gap-3">
                      <div className={clsx(
                        "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold",
                        discoverMode === 'winrate' ? "bg-emerald-900" : "bg-gray-700"
                      )}>
                        #{trader.rank || idx + 1}
                      </div>
                      <div>
                        <p className="font-medium text-sm">
                          {trader.username || `${trader.address.slice(0, 6)}...${trader.address.slice(-4)}`}
                        </p>
                        <p className="text-xs text-gray-500">
                          {trader.win_rate !== undefined && (
                            <span className={clsx(
                              "mr-2 font-medium",
                              trader.win_rate >= 80 ? "text-emerald-400" :
                              trader.win_rate >= 60 ? "text-green-400" :
                              trader.win_rate >= 50 ? "text-yellow-400" : "text-red-400"
                            )}>
                              {trader.win_rate.toFixed(1)}% WR
                            </span>
                          )}
                          {trader.wins !== undefined && trader.losses !== undefined && (
                            <span className="text-gray-400 mr-2">
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
                      <button
                        onClick={() => handleAnalyze(trader.address)}
                        className="flex items-center gap-1 px-2 py-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded text-xs"
                      >
                        <Activity className="w-3 h-3" />
                        Analyze
                      </button>
                      <button
                        onClick={() => handleTrackAndCopy(trader.address, false)}
                        disabled={trackAndCopyMutation.isPending}
                        className="flex items-center gap-1 px-2 py-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded text-xs"
                      >
                        <UserPlus className="w-3 h-3" />
                        Track
                      </button>
                      <button
                        onClick={() => handleTrackAndCopy(trader.address, true)}
                        disabled={trackAndCopyMutation.isPending || simAccounts.length === 0}
                        title={simAccounts.length === 0 ? "Create a simulation account first" : "Track and copy trades in paper mode"}
                        className="flex items-center gap-1 px-2 py-1 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded text-xs disabled:opacity-50"
                      >
                        <Copy className="w-3 h-3" />
                        Copy Trade
                      </button>
                      <a
                        href={`https://polymarket.com/profile/${trader.address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1 hover:bg-gray-700 rounded"
                        title="View on Polymarket"
                      >
                        <ExternalLink className="w-3 h-3 text-gray-500" />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {activeSection === 'tracked' && (
        <>
          {/* Add Wallet Form */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-4">Track a Wallet</h3>
            <div className="flex gap-3">
              <input
                type="text"
                value={newAddress}
                onChange={(e) => setNewAddress(e.target.value)}
                placeholder="Wallet address (0x...)"
                className="flex-1 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
              />
              <input
                type="text"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder="Label (optional)"
                className="w-48 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={handleAdd}
                disabled={addMutation.isPending || !newAddress.trim()}
                className={clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm",
                  "bg-blue-500 hover:bg-blue-600 transition-colors",
                  (addMutation.isPending || !newAddress.trim()) && "opacity-50 cursor-not-allowed"
                )}
              >
                <Plus className="w-4 h-4" />
                Add
              </button>
            </div>
          </div>

          {/* Tracked Wallets */}
          <div>
            <h3 className="text-lg font-medium mb-4">Tracked Wallets</h3>

            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
              </div>
            ) : wallets.length === 0 ? (
              <div className="text-center py-12 bg-[#141414] border border-gray-800 rounded-lg">
                <Wallet className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">No wallets being tracked</p>
                <p className="text-sm text-gray-600 mt-1">
                  Use the Discover tab to find top traders, or add a wallet manually
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {wallets.map((wallet) => (
                  <WalletCard
                    key={wallet.address}
                    wallet={wallet}
                    onRemove={() => removeMutation.mutate(wallet.address)}
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function WalletCard({ wallet, onRemove }: { wallet: WalletType; onRemove: () => void }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-[#141414] border border-gray-800 rounded-lg overflow-hidden">
      <div
        className="p-4 cursor-pointer hover:bg-[#1a1a1a] transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="font-medium">{wallet.label}</p>
              <p className="text-xs text-gray-500 font-mono">{wallet.address}</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm text-gray-400">Positions</p>
              <p className="font-medium">{wallet.positions?.length || 0}</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Recent Trades</p>
              <p className="font-medium">{wallet.recent_trades?.length || 0}</p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onRemove()
              }}
              className="p-2 hover:bg-red-500/10 rounded-lg text-red-400 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-gray-800 p-4">
          {/* Positions */}
          {wallet.positions && wallet.positions.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-400 mb-2">Open Positions</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {wallet.positions.slice(0, 10).map((pos: any, idx: number) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3 text-sm"
                  >
                    <span className="text-gray-300">{pos.market || pos.condition_id}</span>
                    <span className={clsx(
                      "font-mono",
                      pos.pnl >= 0 ? "text-green-400" : "text-red-400"
                    )}>
                      {pos.pnl >= 0 ? '+' : ''}{pos.pnl?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Trades */}
          {wallet.recent_trades && wallet.recent_trades.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-400 mb-2">Recent Trades</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {wallet.recent_trades.slice(0, 10).map((trade: any, idx: number) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3 text-sm"
                  >
                    <div className="flex items-center gap-2">
                      {trade.side === 'BUY' ? (
                        <TrendingUp className="w-4 h-4 text-green-400" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-400" />
                      )}
                      <span className="text-gray-300">{trade.market || 'Unknown'}</span>
                    </div>
                    <span className="font-mono text-gray-400">
                      ${trade.amount?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {(!wallet.positions || wallet.positions.length === 0) &&
           (!wallet.recent_trades || wallet.recent_trades.length === 0) && (
            <p className="text-gray-500 text-sm text-center py-4">
              No position or trade data available yet
            </p>
          )}
        </div>
      )}
    </div>
  )
}
