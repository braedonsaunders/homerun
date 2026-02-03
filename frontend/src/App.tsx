import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  TrendingUp,
  RefreshCw,
  Wallet,
  AlertCircle,
  Clock,
  DollarSign,
  Target,
  Zap,
  Activity,
  PlayCircle,
  Copy,
  Shield
} from 'lucide-react'
import clsx from 'clsx'
import {
  getOpportunities,
  getScannerStatus,
  triggerScan,
  getStrategies,
  Opportunity,
  Strategy
} from './services/api'
import { useWebSocket } from './hooks/useWebSocket'
import OpportunityCard from './components/OpportunityCard'
import WalletTracker from './components/WalletTracker'
import SimulationPanel from './components/SimulationPanel'
import AnomalyPanel from './components/AnomalyPanel'

type Tab = 'opportunities' | 'wallets' | 'simulation' | 'anomaly'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('opportunities')
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [minProfit, setMinProfit] = useState(2.5)
  const queryClient = useQueryClient()

  // WebSocket for real-time updates
  const { isConnected, lastMessage } = useWebSocket('/ws')

  // Update opportunities when WebSocket message received
  useEffect(() => {
    if (lastMessage?.type === 'opportunities_update') {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
    }
  }, [lastMessage, queryClient])

  // Queries
  const { data: opportunities = [], isLoading: oppsLoading } = useQuery({
    queryKey: ['opportunities', selectedStrategy, minProfit],
    queryFn: () => getOpportunities({
      strategy: selectedStrategy || undefined,
      min_profit: minProfit,
      limit: 50
    }),
    refetchInterval: 30000,
  })

  const { data: status } = useQuery({
    queryKey: ['scanner-status'],
    queryFn: getScannerStatus,
    refetchInterval: 10000,
  })

  const { data: strategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: getStrategies,
  })

  // Mutations
  const scanMutation = useMutation({
    mutationFn: triggerScan,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    },
  })

  // Stats
  const totalProfit = opportunities.reduce((sum, o) => sum + o.net_profit, 0)
  const avgROI = opportunities.length > 0
    ? opportunities.reduce((sum, o) => sum + o.roi_percent, 0) / opportunities.length
    : 0

  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <DollarSign className="w-6 h-6 text-green-500" />
              </div>
              <div>
                <h1 className="text-xl font-bold">Polymarket Arbitrage</h1>
                <p className="text-xs text-gray-500">Buy dollars for 94 cents</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs",
                isConnected ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"
              )}>
                <span className={clsx(
                  "w-2 h-2 rounded-full",
                  isConnected ? "bg-green-500" : "bg-red-500"
                )} />
                {isConnected ? 'Live' : 'Disconnected'}
              </div>

              {/* Scanner Status */}
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <Activity className="w-4 h-4" />
                {status?.running ? 'Scanning' : 'Idle'}
              </div>

              {/* Scan Button */}
              <button
                onClick={() => scanMutation.mutate()}
                disabled={scanMutation.isPending}
                className={clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm",
                  "bg-blue-500 hover:bg-blue-600 transition-colors",
                  scanMutation.isPending && "opacity-50 cursor-not-allowed"
                )}
              >
                <RefreshCw className={clsx("w-4 h-4", scanMutation.isPending && "animate-spin")} />
                Scan Now
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="border-b border-gray-800 bg-[#0a0a0a]">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="grid grid-cols-4 gap-4">
            <StatCard
              icon={<Target className="w-5 h-5 text-blue-500" />}
              label="Opportunities"
              value={opportunities.length.toString()}
            />
            <StatCard
              icon={<TrendingUp className="w-5 h-5 text-green-500" />}
              label="Avg ROI"
              value={`${avgROI.toFixed(2)}%`}
            />
            <StatCard
              icon={<DollarSign className="w-5 h-5 text-yellow-500" />}
              label="Total Profit"
              value={`$${totalProfit.toFixed(4)}`}
            />
            <StatCard
              icon={<Clock className="w-5 h-5 text-purple-500" />}
              label="Last Scan"
              value={status?.last_scan
                ? new Date(status.last_scan).toLocaleTimeString()
                : 'Never'
              }
            />
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            <TabButton
              active={activeTab === 'opportunities'}
              onClick={() => setActiveTab('opportunities')}
              icon={<Zap className="w-4 h-4" />}
              label="Opportunities"
            />
            <TabButton
              active={activeTab === 'simulation'}
              onClick={() => setActiveTab('simulation')}
              icon={<PlayCircle className="w-4 h-4" />}
              label="Paper Trading"
            />
            <TabButton
              active={activeTab === 'wallets'}
              onClick={() => setActiveTab('wallets')}
              icon={<Wallet className="w-4 h-4" />}
              label="Wallet Tracker"
            />
            <TabButton
              active={activeTab === 'anomaly'}
              onClick={() => setActiveTab('anomaly')}
              icon={<Shield className="w-4 h-4" />}
              label="Anomaly Detection"
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'opportunities' && (
          <div>
            {/* Filters */}
            <div className="flex gap-4 mb-6">
              <div className="flex-1">
                <label className="block text-xs text-gray-500 mb-1">Strategy</label>
                <select
                  value={selectedStrategy}
                  onChange={(e) => setSelectedStrategy(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="">All Strategies</option>
                  {strategies.map((s) => (
                    <option key={s.type} value={s.type}>{s.name}</option>
                  ))}
                </select>
              </div>
              <div className="w-48">
                <label className="block text-xs text-gray-500 mb-1">Min Profit %</label>
                <input
                  type="number"
                  value={minProfit}
                  onChange={(e) => setMinProfit(parseFloat(e.target.value) || 0)}
                  step="0.5"
                  min="0"
                  className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm"
                />
              </div>
            </div>

            {/* Opportunities List */}
            {oppsLoading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
              </div>
            ) : opportunities.length === 0 ? (
              <div className="text-center py-12">
                <AlertCircle className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">No arbitrage opportunities found</p>
                <p className="text-sm text-gray-600 mt-1">
                  Try lowering the minimum profit threshold
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {opportunities.map((opp) => (
                  <OpportunityCard key={opp.id} opportunity={opp} />
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'simulation' && <SimulationPanel />}
        {activeTab === 'wallets' && <WalletTracker />}
        {activeTab === 'anomaly' && <AnomalyPanel />}
      </main>
    </div>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-[#141414] rounded-lg p-4 border border-gray-800">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-[#1a1a1a] rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-gray-500">{label}</p>
          <p className="text-lg font-semibold">{value}</p>
        </div>
      </div>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  icon,
  label
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors",
        active
          ? "border-blue-500 text-white"
          : "border-transparent text-gray-500 hover:text-gray-300"
      )}
    >
      {icon}
      {label}
    </button>
  )
}

export default App
