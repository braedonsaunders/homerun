import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  Trash2,
  Wallet,
  ExternalLink,
  RefreshCw,
  TrendingUp,
  TrendingDown
} from 'lucide-react'
import clsx from 'clsx'
import { getWallets, addWallet, removeWallet, Wallet as WalletType } from '../services/api'

export default function WalletTracker() {
  const [newAddress, setNewAddress] = useState('')
  const [newLabel, setNewLabel] = useState('')
  const queryClient = useQueryClient()

  const { data: wallets = [], isLoading } = useQuery({
    queryKey: ['wallets'],
    queryFn: getWallets,
    refetchInterval: 30000,
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

  const handleAdd = () => {
    if (!newAddress.trim()) return
    addMutation.mutate({ address: newAddress.trim(), label: newLabel.trim() || undefined })
  }

  // Known profitable wallets to suggest
  const suggestedWallets = [
    { address: '0x...', label: 'anoin123 (Iran Arb Trader)', note: '$1M in 7 days' },
  ]

  return (
    <div className="space-y-6">
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
        <p className="text-xs text-gray-500 mt-2">
          Track wallets to monitor their positions and get alerts when they enter new trades.
        </p>
      </div>

      {/* Suggested Wallets */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Suggested Profitable Wallets</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
            <div>
              <p className="text-sm font-medium">anoin123</p>
              <p className="text-xs text-gray-500">$1M profit in 7 days using Iran date sweep strategy</p>
            </div>
            <a
              href="https://polymarket.com/@anoin123"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-blue-400 text-sm hover:underline"
            >
              View Profile <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          <p className="text-xs text-gray-600">
            Note: Find wallet addresses from Polymarket profiles or blockchain explorers
          </p>
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
              Add a wallet address above to start tracking
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
