import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Activity,
  RefreshCw,
  Play,
  Trash2
} from 'lucide-react'
import clsx from 'clsx'
import {
  getSimulationAccounts,
  createSimulationAccount,
  deleteSimulationAccount,
  getAccountTrades,
  getOpportunities,
  SimulationAccount,
  SimulationTrade,
  Opportunity
} from '../services/api'
import TradeExecutionModal from './TradeExecutionModal'

export default function SimulationPanel() {
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [newAccountName, setNewAccountName] = useState('')
  const [newAccountCapital, setNewAccountCapital] = useState(10000)
  const [accountToDelete, setAccountToDelete] = useState<SimulationAccount | null>(null)
  const [executingOpportunity, setExecutingOpportunity] = useState<Opportunity | null>(null)
  const queryClient = useQueryClient()

  const { data: accounts = [], isLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  const { data: opportunitiesData } = useQuery({
    queryKey: ['opportunities'],
    queryFn: () => getOpportunities({ limit: 20 }),
  })
  const opportunities = opportunitiesData?.opportunities ?? []

  const { data: trades = [] } = useQuery({
    queryKey: ['account-trades', selectedAccount],
    queryFn: () => selectedAccount ? getAccountTrades(selectedAccount) : Promise.resolve([]),
    enabled: !!selectedAccount,
  })

  const createMutation = useMutation({
    mutationFn: () => createSimulationAccount({
      name: newAccountName,
      initial_capital: newAccountCapital
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
      setShowCreateForm(false)
      setNewAccountName('')
    }
  })

  const deleteMutation = useMutation({
    mutationFn: (accountId: string) => deleteSimulationAccount(accountId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
      if (selectedAccount === accountToDelete?.id) {
        setSelectedAccount(null)
      }
      setAccountToDelete(null)
    }
  })

  const selectedAccountData = accounts.find(a => a.id === selectedAccount)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Paper Trading Simulation</h2>
          <p className="text-sm text-gray-500">Practice trading without risking real money</p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-sm font-medium"
        >
          <Plus className="w-4 h-4" />
          New Account
        </button>
      </div>

      {/* Create Account Form */}
      {showCreateForm && (
        <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
          <h3 className="font-medium mb-4">Create Simulation Account</h3>
          <div className="flex gap-4">
            <input
              type="text"
              value={newAccountName}
              onChange={(e) => setNewAccountName(e.target.value)}
              placeholder="Account name"
              className="flex-1 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-2"
            />
            <input
              type="number"
              value={newAccountCapital}
              onChange={(e) => setNewAccountCapital(Number(e.target.value))}
              placeholder="Initial capital"
              className="w-40 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-2"
            />
            <button
              onClick={() => createMutation.mutate()}
              disabled={!newAccountName || createMutation.isPending}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg font-medium disabled:opacity-50"
            >
              Create
            </button>
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Accounts List */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
        </div>
      ) : accounts.length === 0 ? (
        <div className="text-center py-12 bg-[#141414] border border-gray-800 rounded-lg">
          <DollarSign className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">No simulation accounts yet</p>
          <p className="text-sm text-gray-600">Create one to start paper trading</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {accounts.map((account) => (
            <AccountCard
              key={account.id}
              account={account}
              isSelected={selectedAccount === account.id}
              onSelect={() => setSelectedAccount(account.id)}
              onDelete={() => setAccountToDelete(account)}
            />
          ))}
        </div>
      )}

      {/* Selected Account Details */}
      {selectedAccountData && (
        <div className="space-y-4">
          {/* Quick Execute */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
            <h3 className="font-medium mb-4">Execute Opportunity</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {opportunities.slice(0, 5).map((opp) => (
                <div
                  key={opp.id}
                  className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3"
                >
                  <div className="flex-1">
                    <p className="text-sm font-medium">{opp.title}</p>
                    <p className="text-xs text-gray-500">
                      ROI: {opp.roi_percent.toFixed(2)}% | Cost: ${opp.total_cost.toFixed(4)}
                    </p>
                  </div>
                  <button
                    onClick={() => setExecutingOpportunity(opp)}
                    className="flex items-center gap-1 px-3 py-1 bg-green-500/20 text-green-400 rounded-lg text-sm hover:bg-green-500/30"
                  >
                    <Play className="w-3 h-3" />
                    Execute
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Trade History */}
          <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
            <h3 className="font-medium mb-4">Trade History</h3>
            {trades.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No trades yet</p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {trades.map((trade) => (
                  <TradeRow key={trade.id} trade={trade} />
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Trade Execution Modal */}
      {executingOpportunity && (
        <TradeExecutionModal
          opportunity={executingOpportunity}
          onClose={() => {
            setExecutingOpportunity(null)
            queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
            queryClient.invalidateQueries({ queryKey: ['account-trades'] })
          }}
        />
      )}

      {/* Delete Confirmation Modal */}
      {accountToDelete && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-gray-800 rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-lg font-medium mb-2">Delete Account</h3>
            <p className="text-gray-400 mb-4">
              Are you sure you want to delete "{accountToDelete.name}"? This will also delete all trades and positions associated with this account. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setAccountToDelete(null)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteMutation.mutate(accountToDelete.id)}
                disabled={deleteMutation.isPending}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 rounded-lg disabled:opacity-50"
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function AccountCard({
  account,
  isSelected,
  onSelect,
  onDelete
}: {
  account: SimulationAccount
  isSelected: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  const pnlColor = account.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
  const roiColor = account.roi_percent >= 0 ? 'text-green-400' : 'text-red-400'

  return (
    <div
      onClick={onSelect}
      className={clsx(
        "bg-[#141414] border rounded-lg p-4 cursor-pointer transition-colors",
        isSelected ? "border-blue-500" : "border-gray-800 hover:border-gray-700"
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium">{account.name}</h3>
        <div className="flex items-center gap-2">
          {isSelected && <Activity className="w-4 h-4 text-blue-500" />}
          <button
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            className="p-1 text-gray-500 hover:text-red-400 rounded transition-colors"
            title="Delete account"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-gray-500">Capital</p>
          <p className="font-mono">${(account.current_capital ?? 0).toFixed(2)}</p>
        </div>
        <div>
          <p className="text-gray-500">PnL</p>
          <p className={clsx("font-mono", pnlColor)}>
            {(account.total_pnl ?? 0) >= 0 ? '+' : ''}${(account.total_pnl ?? 0).toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-gray-500">ROI</p>
          <p className={clsx("font-mono", roiColor)}>
            {(account.roi_percent ?? 0) >= 0 ? '+' : ''}{(account.roi_percent ?? 0).toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-gray-500">Win Rate</p>
          <p className="font-mono">{(account.win_rate ?? 0).toFixed(1)}%</p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between text-xs text-gray-500">
        <span>{account.total_trades ?? 0} trades</span>
        <span>{account.open_positions ?? 0} open</span>
      </div>
    </div>
  )
}

function TradeRow({ trade }: { trade: SimulationTrade }) {
  const statusColors: Record<string, string> = {
    open: 'bg-blue-500/20 text-blue-400',
    resolved_win: 'bg-green-500/20 text-green-400',
    resolved_loss: 'bg-red-500/20 text-red-400',
    pending: 'bg-yellow-500/20 text-yellow-400'
  }

  return (
    <div className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
      <div className="flex items-center gap-3">
        {trade.actual_pnl !== null && trade.actual_pnl >= 0 ? (
          <TrendingUp className="w-4 h-4 text-green-400" />
        ) : (
          <TrendingDown className="w-4 h-4 text-red-400" />
        )}
        <div>
          <p className="text-sm">{trade.strategy_type}</p>
          <p className="text-xs text-gray-500">
            Cost: ${trade.total_cost.toFixed(2)}
            {trade.copied_from && ` | Copied`}
          </p>
        </div>
      </div>
      <div className="text-right">
        <span className={clsx("px-2 py-0.5 rounded text-xs", statusColors[trade.status] || 'bg-gray-500/20')}>
          {trade.status.replace('_', ' ')}
        </span>
        {trade.actual_pnl !== null && (
          <p className={clsx("text-sm font-mono mt-1", trade.actual_pnl >= 0 ? 'text-green-400' : 'text-red-400')}>
            {trade.actual_pnl >= 0 ? '+' : ''}${trade.actual_pnl.toFixed(2)}
          </p>
        )}
      </div>
    </div>
  )
}
