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
  DollarSign,
  Activity
} from 'lucide-react'
import clsx from 'clsx'
import {
  getWallets,
  addWallet,
  removeWallet,
  Wallet as WalletType,
  discoverTopTraders,
  analyzeWalletPnL,
  analyzeAndTrackWallet,
  DiscoveredTrader,
  WalletPnL,
  getSimulationAccounts
} from '../services/api'

export default function WalletTracker() {
  const [newAddress, setNewAddress] = useState('')
  const [newLabel, setNewLabel] = useState('')
  const [activeSection, setActiveSection] = useState<'tracked' | 'discover'>('discover')
  const [selectedWallet, setSelectedWallet] = useState<string | null>(null)
  const [walletAnalysis, setWalletAnalysis] = useState<WalletPnL | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const queryClient = useQueryClient()

  const { data: wallets = [], isLoading } = useQuery({
    queryKey: ['wallets'],
    queryFn: getWallets,
    refetchInterval: 30000,
  })

  const { data: discoveredTraders = [], isLoading: discoveringTraders, refetch: refreshTraders } = useQuery({
    queryKey: ['discovered-traders'],
    queryFn: () => discoverTopTraders(50, 5),
    refetchInterval: 60000,
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

  const handleAnalyze = async (address: string) => {
    setSelectedWallet(address)
    setAnalyzing(true)
    try {
      const analysis = await analyzeWalletPnL(address)
      setWalletAnalysis(analysis)
    } catch (e) {
      console.error('Analysis failed:', e)
    }
    setAnalyzing(false)
  }

  const handleTrackAndCopy = (address: string, autoCopy: boolean = false) => {
    const label = `Discovered Trader (${discoveredTraders.find(t => t.address === address)?.volume?.toFixed(0) || '?'} vol)`
    trackAndCopyMutation.mutate({
      address,
      label,
      auto_copy: autoCopy,
      simulation_account_id: autoCopy && simAccounts.length > 0 ? simAccounts[0].id : undefined
    })
  }

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
          {/* Discovery Header */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-lg font-medium flex items-center gap-2">
                  <Star className="w-5 h-5 text-yellow-500" />
                  Top Active Traders
                </h3>
                <p className="text-sm text-gray-500">
                  Discovered from recent Polymarket trading activity
                </p>
              </div>
              <button
                onClick={() => refreshTraders()}
                disabled={discoveringTraders}
                className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1a1a] rounded-lg text-sm hover:bg-gray-700"
              >
                <RefreshCw className={clsx("w-4 h-4", discoveringTraders && "animate-spin")} />
                Refresh
              </button>
            </div>

            {discoveringTraders ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="w-6 h-6 animate-spin text-gray-500" />
                <span className="ml-2 text-gray-500">Scanning Polymarket trades...</span>
              </div>
            ) : discoveredTraders.length === 0 ? (
              <p className="text-center text-gray-500 py-8">No traders discovered yet</p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {discoveredTraders.map((trader, idx) => (
                  <div
                    key={trader.address}
                    className={clsx(
                      "flex items-center justify-between p-3 rounded-lg transition-colors",
                      selectedWallet === trader.address ? "bg-green-500/10 border border-green-500/30" : "bg-[#1a1a1a] hover:bg-[#222]"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center text-xs font-bold">
                        #{trader.rank || idx + 1}
                      </div>
                      <div>
                        <p className="font-medium text-sm">
                          {trader.username || `${trader.address.slice(0, 6)}...${trader.address.slice(-4)}`}
                        </p>
                        <p className="text-xs text-gray-500">
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
                        disabled={analyzing && selectedWallet === trader.address}
                        className={clsx(
                          "flex items-center gap-1 px-2 py-1 rounded text-xs",
                          selectedWallet === trader.address && walletAnalysis
                            ? "bg-green-500/20 text-green-400 border border-green-500/30"
                            : "bg-gray-700 hover:bg-gray-600"
                        )}
                      >
                        {analyzing && selectedWallet === trader.address ? (
                          <RefreshCw className="w-3 h-3 animate-spin" />
                        ) : (
                          <Activity className="w-3 h-3" />
                        )}
                        {analyzing && selectedWallet === trader.address ? 'Analyzing...' : 'Analyze'}
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

          {/* Wallet Analysis Panel */}
          {selectedWallet && walletAnalysis && (
            <div className="bg-[#141414] border border-green-500/30 rounded-lg p-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-green-500" />
                Wallet Analysis: {selectedWallet.slice(0, 8)}...
              </h3>
              {walletAnalysis.error ? (
                <p className="text-red-400">{walletAnalysis.error}</p>
              ) : (
                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500">Total Trades</p>
                    <p className="text-lg font-mono">{walletAnalysis.total_trades}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Open Positions</p>
                    <p className="text-lg font-mono">{walletAnalysis.open_positions}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Total Invested</p>
                    <p className="text-lg font-mono">${walletAnalysis.total_invested.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Total P/L</p>
                    <p className={clsx(
                      "text-lg font-mono",
                      walletAnalysis.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                    )}>
                      {walletAnalysis.total_pnl >= 0 ? '+' : ''}${walletAnalysis.total_pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500">ROI</p>
                    <p className={clsx(
                      "text-lg font-mono",
                      walletAnalysis.roi_percent >= 0 ? "text-green-400" : "text-red-400"
                    )}>
                      {walletAnalysis.roi_percent >= 0 ? '+' : ''}{walletAnalysis.roi_percent.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500">Realized P/L</p>
                    <p className="font-mono">${walletAnalysis.realized_pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Unrealized P/L</p>
                    <p className="font-mono">${walletAnalysis.unrealized_pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Position Value</p>
                    <p className="font-mono">${walletAnalysis.position_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                  </div>
                </div>
              )}
            </div>
          )}
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
