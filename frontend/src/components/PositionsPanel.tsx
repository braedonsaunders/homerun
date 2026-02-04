import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Briefcase,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Calendar,
  ExternalLink
} from 'lucide-react'
import clsx from 'clsx'
import {
  getSimulationAccounts,
  getAccountPositions,
  getTradingPositions,
  SimulationAccount
} from '../services/api'

interface SimulationPosition {
  id: string
  market_id: string
  market_question: string
  token_id: string
  side: string
  quantity: number
  entry_price: number
  entry_cost: number
  current_price: number | null
  unrealized_pnl: number
  opened_at: string
  resolution_date: string | null
  status: string
}

interface TradingPosition {
  market: string
  market_slug: string
  outcome: string
  size: number
  avgPrice: number
  currentPrice: number
  costBasis: number
  currentValue: number
  unrealizedPnl: number
  roiPercent: number
}

type ViewMode = 'simulation' | 'live' | 'all'

export default function PositionsPanel() {
  const [viewMode, setViewMode] = useState<ViewMode>('all')
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)

  // Fetch simulation accounts
  const { data: accounts = [], isLoading: accountsLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  // Fetch positions for selected simulation account or all accounts
  const { data: simulationPositions = [], isLoading: simPosLoading, refetch: refetchSimPositions } = useQuery({
    queryKey: ['simulation-positions', selectedAccount],
    queryFn: async () => {
      if (selectedAccount) {
        return getAccountPositions(selectedAccount)
      }
      // Fetch positions from all accounts
      const allPositions: (SimulationPosition & { accountName: string; accountId: string })[] = []
      for (const account of accounts) {
        const positions = await getAccountPositions(account.id)
        positions.forEach((pos: SimulationPosition) => {
          allPositions.push({ ...pos, accountName: account.name, accountId: account.id })
        })
      }
      return allPositions
    },
    enabled: accounts.length > 0 && (viewMode === 'simulation' || viewMode === 'all'),
  })

  // Fetch live trading positions
  const { data: livePositions = [], isLoading: livePosLoading, refetch: refetchLivePositions } = useQuery({
    queryKey: ['live-positions'],
    queryFn: getTradingPositions,
    enabled: viewMode === 'live' || viewMode === 'all',
  })

  const isLoading = accountsLoading || simPosLoading || livePosLoading

  // Calculate totals
  const simTotalValue = simulationPositions.reduce((sum: number, pos: SimulationPosition) =>
    sum + (pos.quantity * (pos.current_price || pos.entry_price)), 0)
  const simTotalUnrealizedPnl = simulationPositions.reduce((sum: number, pos: SimulationPosition) =>
    sum + pos.unrealized_pnl, 0)

  const liveTotalValue = livePositions.reduce((sum: number, pos: TradingPosition) =>
    sum + pos.currentValue, 0)
  const liveTotalUnrealizedPnl = livePositions.reduce((sum: number, pos: TradingPosition) =>
    sum + pos.unrealizedPnl, 0)

  const handleRefresh = () => {
    if (viewMode === 'simulation' || viewMode === 'all') refetchSimPositions()
    if (viewMode === 'live' || viewMode === 'all') refetchLivePositions()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Briefcase className="w-6 h-6 text-blue-500" />
            Open Positions
          </h2>
          <p className="text-sm text-gray-500">Track your active positions across all accounts</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-[#1a1a1a] hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
        >
          <RefreshCw className={clsx("w-4 h-4", isLoading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* View Mode Selector */}
      <div className="flex items-center gap-4">
        <div className="flex bg-[#141414] rounded-lg p-1 border border-gray-800">
          <button
            onClick={() => setViewMode('all')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              viewMode === 'all' ? "bg-blue-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            All Positions
          </button>
          <button
            onClick={() => setViewMode('simulation')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              viewMode === 'simulation' ? "bg-blue-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            Paper Trading
          </button>
          <button
            onClick={() => setViewMode('live')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              viewMode === 'live' ? "bg-blue-500 text-white" : "text-gray-400 hover:text-white"
            )}
          >
            Live Trading
          </button>
        </div>

        {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
          <select
            value={selectedAccount || ''}
            onChange={(e) => setSelectedAccount(e.target.value || null)}
            className="bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All Simulation Accounts</option>
            {accounts.map((account: SimulationAccount) => (
              <option key={account.id} value={account.id}>{account.name}</option>
            ))}
          </select>
        )}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {(viewMode === 'simulation' || viewMode === 'all') && (
          <>
            <StatCard
              icon={<Briefcase className="w-5 h-5 text-blue-500" />}
              label="Paper Positions"
              value={simulationPositions.length.toString()}
            />
            <StatCard
              icon={<DollarSign className="w-5 h-5 text-green-500" />}
              label="Paper Value"
              value={`$${simTotalValue.toFixed(2)}`}
            />
          </>
        )}
        {(viewMode === 'live' || viewMode === 'all') && (
          <>
            <StatCard
              icon={<Target className="w-5 h-5 text-purple-500" />}
              label="Live Positions"
              value={livePositions.length.toString()}
            />
            <StatCard
              icon={<DollarSign className="w-5 h-5 text-yellow-500" />}
              label="Live Value"
              value={`$${liveTotalValue.toFixed(2)}`}
            />
          </>
        )}
        <StatCard
          icon={simTotalUnrealizedPnl + liveTotalUnrealizedPnl >= 0
            ? <TrendingUp className="w-5 h-5 text-green-500" />
            : <TrendingDown className="w-5 h-5 text-red-500" />
          }
          label="Total Unrealized P&L"
          value={`${(simTotalUnrealizedPnl + liveTotalUnrealizedPnl) >= 0 ? '+' : ''}$${(
            (viewMode === 'all' ? simTotalUnrealizedPnl + liveTotalUnrealizedPnl :
             viewMode === 'simulation' ? simTotalUnrealizedPnl : liveTotalUnrealizedPnl)
          ).toFixed(2)}`}
          valueColor={(viewMode === 'all' ? simTotalUnrealizedPnl + liveTotalUnrealizedPnl :
                      viewMode === 'simulation' ? simTotalUnrealizedPnl : liveTotalUnrealizedPnl) >= 0
            ? 'text-green-400' : 'text-red-400'}
        />
      </div>

      {/* Positions List */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* Simulation Positions */}
          {(viewMode === 'simulation' || viewMode === 'all') && (
            <div className="space-y-4">
              {viewMode === 'all' && (
                <h3 className="text-lg font-semibold text-gray-300 flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Paper Trading Positions
                </h3>
              )}
              {simulationPositions.length === 0 ? (
                <div className="text-center py-8 bg-[#141414] border border-gray-800 rounded-lg">
                  <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-400">No open paper trading positions</p>
                  <p className="text-sm text-gray-600">Execute opportunities from the Paper Trading tab</p>
                </div>
              ) : (
                <div className="grid gap-3">
                  {simulationPositions.map((position: SimulationPosition & { accountName?: string; accountId?: string }) => (
                    <SimulationPositionCard
                      key={position.id}
                      position={position}
                      showAccount={!selectedAccount}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Live Positions */}
          {(viewMode === 'live' || viewMode === 'all') && (
            <div className="space-y-4">
              {viewMode === 'all' && (
                <h3 className="text-lg font-semibold text-gray-300 flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  Live Trading Positions
                </h3>
              )}
              {livePositions.length === 0 ? (
                <div className="text-center py-8 bg-[#141414] border border-gray-800 rounded-lg">
                  <Target className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-400">No open live trading positions</p>
                  <p className="text-sm text-gray-600">Start live trading from the Auto Trading tab</p>
                </div>
              ) : (
                <div className="grid gap-3">
                  {livePositions.map((position: TradingPosition, idx: number) => (
                    <LivePositionCard key={idx} position={position} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  valueColor = 'text-white'
}: {
  icon: React.ReactNode
  label: string
  value: string
  valueColor?: string
}) {
  return (
    <div className="bg-[#141414] rounded-lg p-4 border border-gray-800">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-[#1a1a1a] rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-gray-500">{label}</p>
          <p className={clsx("text-lg font-semibold", valueColor)}>{value}</p>
        </div>
      </div>
    </div>
  )
}

function SimulationPositionCard({
  position,
  showAccount = false
}: {
  position: SimulationPosition & { accountName?: string; accountId?: string }
  showAccount?: boolean
}) {
  const currentPrice = position.current_price || position.entry_price
  const currentValue = position.quantity * currentPrice
  const pnlPercent = position.entry_cost > 0
    ? (position.unrealized_pnl / position.entry_cost) * 100
    : 0
  const isProfitable = position.unrealized_pnl >= 0

  return (
    <div className="bg-[#141414] border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className={clsx(
              "px-2 py-0.5 rounded text-xs font-medium",
              position.side === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
            )}>
              {position.side.toUpperCase()}
            </span>
            {showAccount && position.accountName && (
              <span className="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400">
                {position.accountName}
              </span>
            )}
            <span className={clsx(
              "px-2 py-0.5 rounded text-xs",
              position.status === 'open' ? "bg-blue-500/20 text-blue-400" : "bg-gray-500/20 text-gray-400"
            )}>
              {position.status}
            </span>
          </div>
          <h4 className="font-medium text-sm mb-2 line-clamp-2">{position.market_question}</h4>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-500 text-xs">Quantity</p>
              <p className="font-mono">{position.quantity.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Entry Price</p>
              <p className="font-mono">${position.entry_price.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Current Price</p>
              <p className="font-mono">${currentPrice.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Entry Cost</p>
              <p className="font-mono">${position.entry_cost.toFixed(2)}</p>
            </div>
          </div>
        </div>

        <div className="text-right ml-4">
          <div className="mb-2">
            <p className="text-gray-500 text-xs">Current Value</p>
            <p className="font-mono font-medium">${currentValue.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Unrealized P&L</p>
            <p className={clsx(
              "font-mono font-medium",
              isProfitable ? "text-green-400" : "text-red-400"
            )}>
              {isProfitable ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
              <span className="text-xs ml-1">({isProfitable ? '+' : ''}{pnlPercent.toFixed(1)}%)</span>
            </p>
          </div>
        </div>
      </div>

      {position.resolution_date && (
        <div className="mt-3 pt-3 border-t border-gray-800 flex items-center gap-2 text-xs text-gray-500">
          <Calendar className="w-3 h-3" />
          Resolution: {new Date(position.resolution_date).toLocaleDateString()}
        </div>
      )}
    </div>
  )
}

function LivePositionCard({ position }: { position: TradingPosition }) {
  const isProfitable = position.unrealizedPnl >= 0

  return (
    <div className="bg-[#141414] border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className={clsx(
              "px-2 py-0.5 rounded text-xs font-medium",
              position.outcome.toLowerCase() === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
            )}>
              {position.outcome.toUpperCase()}
            </span>
            <span className="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400">
              LIVE
            </span>
          </div>
          <h4 className="font-medium text-sm mb-2">{position.market}</h4>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-500 text-xs">Size</p>
              <p className="font-mono">{position.size.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Avg Price</p>
              <p className="font-mono">${position.avgPrice.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Current Price</p>
              <p className="font-mono">${position.currentPrice.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Cost Basis</p>
              <p className="font-mono">${position.costBasis.toFixed(2)}</p>
            </div>
          </div>
        </div>

        <div className="text-right ml-4">
          <div className="mb-2">
            <p className="text-gray-500 text-xs">Current Value</p>
            <p className="font-mono font-medium">${position.currentValue.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Unrealized P&L</p>
            <p className={clsx(
              "font-mono font-medium",
              isProfitable ? "text-green-400" : "text-red-400"
            )}>
              {isProfitable ? '+' : ''}${position.unrealizedPnl.toFixed(2)}
              <span className="text-xs ml-1">({isProfitable ? '+' : ''}{position.roiPercent.toFixed(1)}%)</span>
            </p>
          </div>
        </div>
      </div>

      {position.market_slug && (
        <div className="mt-3 pt-3 border-t border-gray-800">
          <a
            href={`https://polymarket.com/event/${position.market_slug}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
          >
            View on Polymarket
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      )}
    </div>
  )
}
