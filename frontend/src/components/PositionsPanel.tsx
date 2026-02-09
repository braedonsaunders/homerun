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
import { cn } from '../lib/utils'
import {
  getSimulationAccounts,
  getAccountPositions,
  getTradingPositions,
} from '../services/api'
import type { SimulationAccount, SimulationPosition, TradingPosition } from '../services/api'
import { Card, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { Separator } from './ui/separator'

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
    sum + (pos.size * pos.current_price), 0)
  const liveTotalUnrealizedPnl = livePositions.reduce((sum: number, pos: TradingPosition) =>
    sum + pos.unrealized_pnl, 0)

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
          <p className="text-sm text-muted-foreground">Track your active positions across all accounts</p>
        </div>
        <Button
          variant="secondary"
          onClick={handleRefresh}
          disabled={isLoading}
        >
          <RefreshCw className={cn("w-4 h-4 mr-2", isLoading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* View Mode Selector */}
      <div className="flex items-center gap-4">
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList>
            <TabsTrigger value="all">All Positions</TabsTrigger>
            <TabsTrigger value="simulation">Sandbox Trading</TabsTrigger>
            <TabsTrigger value="live">Live Trading</TabsTrigger>
          </TabsList>
        </Tabs>

        {(viewMode === 'simulation' || viewMode === 'all') && accounts.length > 0 && (
          <select
            value={selectedAccount || ''}
            onChange={(e) => setSelectedAccount(e.target.value || null)}
            className="bg-muted border border-border rounded-lg px-3 py-2 text-sm"
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
              label="Sandbox Positions"
              value={simulationPositions.length.toString()}
            />
            <StatCard
              icon={<DollarSign className="w-5 h-5 text-green-500" />}
              label="Sandbox Value"
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
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* Simulation Positions */}
          {(viewMode === 'simulation' || viewMode === 'all') && (
            <div className="space-y-4">
              {viewMode === 'all' && (
                <h3 className="text-lg font-semibold text-muted-foreground flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Sandbox Trading Positions
                </h3>
              )}
              {simulationPositions.length === 0 ? (
                <Card>
                  <CardContent className="text-center py-8">
                    <Briefcase className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
                    <p className="text-muted-foreground">No open sandbox trading positions</p>
                    <p className="text-sm text-muted-foreground">Execute opportunities from the Sandbox tab</p>
                  </CardContent>
                </Card>
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
                <h3 className="text-lg font-semibold text-muted-foreground flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  Live Trading Positions
                </h3>
              )}
              {livePositions.length === 0 ? (
                <Card>
                  <CardContent className="text-center py-8">
                    <Target className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
                    <p className="text-muted-foreground">No open live trading positions</p>
                    <p className="text-sm text-muted-foreground">Start live trading from the Auto Trading tab</p>
                  </CardContent>
                </Card>
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
  valueColor = 'text-foreground'
}: {
  icon: React.ReactNode
  label: string
  value: string
  valueColor?: string
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 p-4">
        <div className="p-2 bg-muted rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className={cn("text-lg font-semibold", valueColor)}>{value}</p>
        </div>
      </CardContent>
    </Card>
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
    <Card className="transition-colors">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Badge className={cn(
                "rounded border-transparent",
                position.side === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
              )}>
                {position.side.toUpperCase()}
              </Badge>
              {showAccount && position.accountName && (
                <Badge className="rounded border-transparent bg-blue-500/20 text-blue-400">
                  {position.accountName}
                </Badge>
              )}
              <Badge className={cn(
                "rounded border-transparent",
                position.status === 'open' ? "bg-blue-500/20 text-blue-400" : "bg-muted text-muted-foreground"
              )}>
                {position.status}
              </Badge>
            </div>
            <h4 className="font-medium text-sm mb-2 line-clamp-2">{position.market_question}</h4>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground text-xs">Quantity</p>
                <p className="font-mono">{position.quantity.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Entry Price</p>
                <p className="font-mono">${position.entry_price.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Current Price</p>
                <p className="font-mono">${currentPrice.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Entry Cost</p>
                <p className="font-mono">${position.entry_cost.toFixed(2)}</p>
              </div>
            </div>
          </div>

          <div className="text-right ml-4">
            <div className="mb-2">
              <p className="text-muted-foreground text-xs">Current Value</p>
              <p className="font-mono font-medium">${currentValue.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Unrealized P&L</p>
              <p className={cn(
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
          <div className="mt-3">
            <Separator />
            <div className="pt-3 flex items-center gap-2 text-xs text-muted-foreground">
              <Calendar className="w-3 h-3" />
              Resolution: {new Date(position.resolution_date).toLocaleDateString()}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function LivePositionCard({ position }: { position: TradingPosition }) {
  // Calculate derived values from API response
  const costBasis = position.size * position.average_cost
  const currentValue = position.size * position.current_price
  const roiPercent = costBasis > 0 ? (position.unrealized_pnl / costBasis) * 100 : 0
  const isProfitable = position.unrealized_pnl >= 0

  return (
    <Card className="transition-colors">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Badge className={cn(
                "rounded border-transparent",
                position.outcome.toLowerCase() === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
              )}>
                {position.outcome.toUpperCase()}
              </Badge>
              <Badge className="rounded border-transparent bg-purple-500/20 text-purple-400">
                LIVE
              </Badge>
            </div>
            <h4 className="font-medium text-sm mb-2">{position.market_question}</h4>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground text-xs">Size</p>
                <p className="font-mono">{position.size.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Avg Price</p>
                <p className="font-mono">${position.average_cost.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Current Price</p>
                <p className="font-mono">${position.current_price.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Cost Basis</p>
                <p className="font-mono">${costBasis.toFixed(2)}</p>
              </div>
            </div>
          </div>

          <div className="text-right ml-4">
            <div className="mb-2">
              <p className="text-muted-foreground text-xs">Current Value</p>
              <p className="font-mono font-medium">${currentValue.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Unrealized P&L</p>
              <p className={cn(
                "font-mono font-medium",
                isProfitable ? "text-green-400" : "text-red-400"
              )}>
                {isProfitable ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
                <span className="text-xs ml-1">({isProfitable ? '+' : ''}{roiPercent.toFixed(1)}%)</span>
              </p>
            </div>
          </div>
        </div>

        {position.market_id && (
          <div className="mt-3">
            <Separator />
            <div className="pt-3">
              <a
                href={`https://polymarket.com/event/${position.market_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
              >
                View on Polymarket
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
