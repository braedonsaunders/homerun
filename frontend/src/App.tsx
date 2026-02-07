import { useState, useEffect, useCallback } from 'react'
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
  Bot,
  Search,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  Settings,
  Terminal,
  Briefcase,
  BarChart3,
  Trophy,
  Users,
  Brain,
  Sparkles,
  Command,
} from 'lucide-react'
import clsx from 'clsx'
import {
  getOpportunities,
  getScannerStatus,
  triggerScan,
  getStrategies,
  startScanner,
  pauseScanner,
  Opportunity
} from './services/api'
import { useWebSocket } from './hooks/useWebSocket'
import OpportunityCard from './components/OpportunityCard'
import TradeExecutionModal from './components/TradeExecutionModal'
import WalletTracker from './components/WalletTracker'
import SimulationPanel from './components/SimulationPanel'
import LiveAccountPanel from './components/LiveAccountPanel'
import WalletAnalysisPanel from './components/WalletAnalysisPanel'
import TradingPanel from './components/TradingPanel'
import PositionsPanel from './components/PositionsPanel'
import PerformancePanel from './components/PerformancePanel'
import RecentTradesPanel from './components/RecentTradesPanel'
import SettingsPanel from './components/SettingsPanel'
import AIPanel from './components/AIPanel'
import AICopilotPanel from './components/AICopilotPanel'
import AICommandBar from './components/AICommandBar'

type Tab = 'opportunities' | 'trading' | 'accounts' | 'traders' | 'positions' | 'performance' | 'ai' | 'settings'
type AccountsSubTab = 'paper' | 'live'
type TradersSubTab = 'tracked' | 'leaderboard' | 'discover' | 'analysis'

const ITEMS_PER_PAGE = 20

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('opportunities')
  const [accountsSubTab, setAccountsSubTab] = useState<AccountsSubTab>('paper')
  const [tradersSubTab, setTradersSubTab] = useState<TradersSubTab>('leaderboard')
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [minProfit, setMinProfit] = useState(2.5)
  const [maxRisk, setMaxRisk] = useState(1.0)
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(0)
  const [walletToAnalyze, setWalletToAnalyze] = useState<string | null>(null)
  const [walletUsername, setWalletUsername] = useState<string | null>(null)
  const [opportunitiesView, setOpportunitiesView] = useState<'arbitrage' | 'recent_trades'>('arbitrage')
  const [executingOpportunity, setExecutingOpportunity] = useState<Opportunity | null>(null)
  const [copilotOpen, setCopilotOpen] = useState(false)
  const [copilotContext, setCopilotContext] = useState<{ type?: string; id?: string; label?: string }>({})
  const [commandBarOpen, setCommandBarOpen] = useState(false)
  const queryClient = useQueryClient()

  // Cmd+K to open command bar
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setCommandBarOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Open copilot with context
  const handleOpenCopilot = useCallback((contextType?: string, contextId?: string, label?: string) => {
    setCopilotContext({ type: contextType, id: contextId, label })
    setCopilotOpen(true)
  }, [])

  // Open copilot from opportunity card
  const handleOpenCopilotForOpportunity = useCallback((opp: Opportunity) => {
    handleOpenCopilot('opportunity', opp.id, opp.title)
  }, [handleOpenCopilot])

  // Navigate to AI tab with specific section
  const handleNavigateToAI = useCallback((section: string) => {
    setActiveTab('ai')
    // Dispatch event for the AI panel to pick up the section
    window.dispatchEvent(new CustomEvent('navigate-ai-section', { detail: section }))
  }, [])

  // Callback for navigating to wallet analysis from WalletTracker
  const handleAnalyzeWallet = (address: string, username?: string) => {
    setWalletToAnalyze(address)
    setWalletUsername(username || null)
    setActiveTab('traders')
    setTradersSubTab('analysis')
  }

  // WebSocket for real-time updates
  const { isConnected, lastMessage } = useWebSocket('/ws')

  // Update data when WebSocket message received
  useEffect(() => {
    if (lastMessage?.type === 'opportunities_update') {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    }
    if (lastMessage?.type === 'scanner_status') {
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    }
  }, [lastMessage, queryClient])

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(0)
  }, [selectedStrategy, selectedCategory, minProfit, maxRisk, searchQuery])

  // Queries
  const { data: opportunitiesData, isLoading: oppsLoading } = useQuery({
    queryKey: ['opportunities', selectedStrategy, selectedCategory, minProfit, maxRisk, searchQuery, currentPage],
    queryFn: () => getOpportunities({
      strategy: selectedStrategy || undefined,
      category: selectedCategory || undefined,
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
      limit: ITEMS_PER_PAGE,
      offset: currentPage * ITEMS_PER_PAGE
    }),
    refetchInterval: 30000,
  })

  const opportunities = opportunitiesData?.opportunities || []
  const totalOpportunities = opportunitiesData?.total || 0

  const { data: status } = useQuery({
    queryKey: ['scanner-status'],
    queryFn: getScannerStatus,
    refetchInterval: 5000,
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

  const startMutation = useMutation({
    mutationFn: startScanner,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    },
  })

  const pauseMutation = useMutation({
    mutationFn: pauseScanner,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    },
  })

  // Stats
  const totalProfit = opportunities.reduce((sum, o) => sum + o.net_profit, 0)
  const avgROI = opportunities.length > 0
    ? opportunities.reduce((sum, o) => sum + o.roi_percent, 0) / opportunities.length
    : 0

  const totalPages = Math.ceil(totalOpportunities / ITEMS_PER_PAGE)

  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Terminal className="w-6 h-6 text-green-500" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-green-400">HOMERUN</h1>
                <p className="text-xs text-gray-500">Polymarket Arbitrage Scanner</p>
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

              {/* Scanner Status & Controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => status?.enabled ? pauseMutation.mutate() : startMutation.mutate()}
                  disabled={pauseMutation.isPending || startMutation.isPending}
                  className={clsx(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
                    status?.enabled
                      ? "bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20"
                      : "bg-green-500/10 text-green-500 hover:bg-green-500/20"
                  )}
                >
                  {status?.enabled ? (
                    <>
                      <Pause className="w-3.5 h-3.5" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      Start
                    </>
                  )}
                </button>
              </div>

              {/* AI Command Bar Toggle */}
              <button
                onClick={() => setCommandBarOpen(true)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 border border-purple-500/20 transition-colors"
                title="AI Command Bar (Cmd+K)"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">AI</span>
                <kbd className="hidden sm:inline px-1.5 py-0.5 bg-purple-500/10 rounded text-[10px] text-purple-400 border border-purple-500/20">
                  <Command className="w-2.5 h-2.5 inline" />K
                </kbd>
              </button>

              {/* AI Copilot Toggle */}
              <button
                onClick={() => setCopilotOpen(!copilotOpen)}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border",
                  copilotOpen
                    ? "bg-purple-500/20 text-purple-400 border-purple-500/30"
                    : "bg-[#1a1a1a] text-gray-400 hover:text-purple-400 border-gray-800 hover:border-purple-500/30"
                )}
                title="AI Copilot"
              >
                <Bot className="w-3.5 h-3.5" />
              </button>

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
          <div className="grid grid-cols-5 gap-4">
            <StatCard
              icon={<Target className="w-5 h-5 text-blue-500" />}
              label="Total Opportunities"
              value={totalOpportunities.toString()}
            />
            <StatCard
              icon={<Activity className="w-5 h-5 text-cyan-500" />}
              label="Showing"
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
              active={activeTab === 'trading'}
              onClick={() => setActiveTab('trading')}
              icon={<Bot className="w-4 h-4" />}
              label="Trading"
            />
            <TabButton
              active={activeTab === 'accounts'}
              onClick={() => setActiveTab('accounts')}
              icon={<Wallet className="w-4 h-4" />}
              label="Accounts"
            />
            <TabButton
              active={activeTab === 'traders'}
              onClick={() => setActiveTab('traders')}
              icon={<Users className="w-4 h-4" />}
              label="Traders"
            />
            <TabButton
              active={activeTab === 'positions'}
              onClick={() => setActiveTab('positions')}
              icon={<Briefcase className="w-4 h-4" />}
              label="Positions"
            />
            <TabButton
              active={activeTab === 'performance'}
              onClick={() => setActiveTab('performance')}
              icon={<BarChart3 className="w-4 h-4" />}
              label="Performance"
            />
            <TabButton
              active={activeTab === 'ai'}
              onClick={() => setActiveTab('ai')}
              icon={<Brain className="w-4 h-4" />}
              label="AI"
            />
            <TabButton
              active={activeTab === 'settings'}
              onClick={() => setActiveTab('settings')}
              icon={<Settings className="w-4 h-4" />}
              label="Settings"
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'opportunities' && (
          <div>
            {/* View Toggle */}
            <div className="flex items-center gap-2 mb-6">
              <button
                onClick={() => setOpportunitiesView('arbitrage')}
                className={clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                  opportunitiesView === 'arbitrage'
                    ? "bg-green-500/20 text-green-400 border border-green-500/30"
                    : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
                )}
              >
                <Zap className="w-4 h-4" />
                Arbitrage Opportunities
              </button>
              <button
                onClick={() => setOpportunitiesView('recent_trades')}
                className={clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                  opportunitiesView === 'recent_trades'
                    ? "bg-orange-500/20 text-orange-400 border border-orange-500/30"
                    : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
                )}
              >
                <Activity className="w-4 h-4" />
                Recent Wallet Trades
              </button>
            </div>

            {opportunitiesView === 'recent_trades' ? (
              <RecentTradesPanel
                onNavigateToWallet={(address) => {
                  setWalletToAnalyze(address)
                  setActiveTab('traders')
                  setTradersSubTab('analysis')
                }}
              />
            ) : (
            <>
            {/* Search */}
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search opportunities by market, event, or keyword..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:border-green-500"
                />
              </div>
            </div>

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
              <div className="flex-1">
                <label className="block text-xs text-gray-500 mb-1">Category</label>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="">All Categories</option>
                  <option value="politics">Politics</option>
                  <option value="sports">Sports</option>
                  <option value="crypto">Crypto</option>
                  <option value="culture">Culture</option>
                  <option value="economics">Economics</option>
                  <option value="tech">Tech</option>
                  <option value="finance">Finance</option>
                  <option value="weather">Weather</option>
                </select>
              </div>
              <div className="w-40">
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
              <div className="w-48">
                <label className="block text-xs text-gray-500 mb-1">Max Risk Score: {maxRisk.toFixed(1)}</label>
                <input
                  type="range"
                  value={maxRisk}
                  onChange={(e) => setMaxRisk(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                  max="1"
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer mt-2"
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
              <>
                <div className="space-y-4">
                  {opportunities.map((opp) => (
                    <OpportunityCard
                      key={opp.id}
                      opportunity={opp}
                      onExecute={setExecutingOpportunity}
                      onOpenCopilot={handleOpenCopilotForOpportunity}
                    />
                  ))}
                </div>

                {/* Pagination */}
                <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-800">
                  <div className="text-sm text-gray-500">
                    Showing {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalOpportunities)} of {totalOpportunities}
                    {searchQuery && ` (filtered by "${searchQuery}")`}
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                      disabled={currentPage === 0}
                      className={clsx(
                        "flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm",
                        currentPage === 0
                          ? "bg-gray-800 text-gray-600 cursor-not-allowed"
                          : "bg-[#1a1a1a] text-gray-300 hover:bg-gray-700"
                      )}
                    >
                      <ChevronLeft className="w-4 h-4" />
                      Previous
                    </button>
                    <span className="px-3 py-1.5 bg-[#1a1a1a] rounded-lg text-sm">
                      Page {currentPage + 1} of {totalPages || 1}
                    </span>
                    <button
                      onClick={() => setCurrentPage(p => p + 1)}
                      disabled={currentPage >= totalPages - 1}
                      className={clsx(
                        "flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm",
                        currentPage >= totalPages - 1
                          ? "bg-gray-800 text-gray-600 cursor-not-allowed"
                          : "bg-[#1a1a1a] text-gray-300 hover:bg-gray-700"
                      )}
                    >
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </>
            )}
            </>
            )}
          </div>
        )}

        {/* Trading Tab - Auto Trading */}
        <div className={activeTab === 'trading' ? '' : 'hidden'}>
          <TradingPanel />
        </div>

        {/* Accounts Tab with Paper/Live Subtabs */}
        <div className={activeTab === 'accounts' ? '' : 'hidden'}>
          {/* Accounts Subtabs Navigation */}
          <div className="flex items-center gap-2 mb-6">
            <button
              onClick={() => setAccountsSubTab('paper')}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                accountsSubTab === 'paper'
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
              )}
            >
              <PlayCircle className="w-4 h-4" />
              Paper Accounts
            </button>
            <button
              onClick={() => setAccountsSubTab('live')}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                accountsSubTab === 'live'
                  ? "bg-green-500/20 text-green-400 border border-green-500/30"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
              )}
            >
              <DollarSign className="w-4 h-4" />
              Live Account
            </button>
          </div>
          {/* Accounts Subtab Content */}
          <div className={accountsSubTab === 'paper' ? '' : 'hidden'}>
            <SimulationPanel />
          </div>
          <div className={accountsSubTab === 'live' ? '' : 'hidden'}>
            <LiveAccountPanel />
          </div>
        </div>

        {/* Traders Tab with Subtabs */}
        <div className={activeTab === 'traders' ? '' : 'hidden'}>
          {/* Traders Subtabs Navigation */}
          <div className="flex items-center gap-2 mb-6">
            <button
              onClick={() => setTradersSubTab('tracked')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                tradersSubTab === 'tracked'
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
              )}
            >
              <Users className="w-4 h-4" />
              Tracked
            </button>
            <button
              onClick={() => setTradersSubTab('leaderboard')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                tradersSubTab === 'leaderboard'
                  ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
              )}
            >
              <Trophy className="w-4 h-4" />
              Leaderboard
            </button>
            <button
              onClick={() => setTradersSubTab('discover')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                tradersSubTab === 'discover'
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
              )}
            >
              <Target className="w-4 h-4" />
              Discover
            </button>
            <button
              onClick={() => setTradersSubTab('analysis')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                tradersSubTab === 'analysis'
                  ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                  : "bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800"
              )}
            >
              <Search className="w-4 h-4" />
              Analysis
            </button>
          </div>
          {/* Traders Subtab Content */}
          <div className={tradersSubTab === 'tracked' ? '' : 'hidden'}>
            <WalletTracker section="tracked" onAnalyzeWallet={handleAnalyzeWallet} />
          </div>
          <div className={tradersSubTab === 'leaderboard' ? '' : 'hidden'}>
            <WalletTracker section="discover" discoverMode="leaderboard" onAnalyzeWallet={handleAnalyzeWallet} />
          </div>
          <div className={tradersSubTab === 'discover' ? '' : 'hidden'}>
            <WalletTracker section="discover" discoverMode="winrate" onAnalyzeWallet={handleAnalyzeWallet} />
          </div>
          <div className={tradersSubTab === 'analysis' ? '' : 'hidden'}>
            <WalletAnalysisPanel
              initialWallet={walletToAnalyze}
              initialUsername={walletUsername}
              onWalletAnalyzed={() => { setWalletToAnalyze(null); setWalletUsername(null) }}
            />
          </div>
        </div>

        <div className={activeTab === 'positions' ? '' : 'hidden'}>
          <PositionsPanel />
        </div>
        <div className={activeTab === 'performance' ? '' : 'hidden'}>
          <PerformancePanel />
        </div>
        <div className={activeTab === 'ai' ? '' : 'hidden'}>
          <AIPanel />
        </div>
        <div className={activeTab === 'settings' ? '' : 'hidden'}>
          <SettingsPanel />
        </div>
      </main>

      {/* Trade Execution Modal */}
      {executingOpportunity && (
        <TradeExecutionModal
          opportunity={executingOpportunity}
          onClose={() => setExecutingOpportunity(null)}
        />
      )}

      {/* AI Copilot Panel (floating) */}
      <AICopilotPanel
        isOpen={copilotOpen}
        onClose={() => setCopilotOpen(false)}
        contextType={copilotContext.type}
        contextId={copilotContext.id}
        contextLabel={copilotContext.label}
      />

      {/* AI Command Bar (Cmd+K) */}
      <AICommandBar
        isOpen={commandBarOpen}
        onClose={() => setCommandBarOpen(false)}
        onNavigateToAI={handleNavigateToAI}
        onOpenCopilot={handleOpenCopilot}
      />
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
          ? "border-green-500 text-white"
          : "border-transparent text-gray-500 hover:text-gray-300"
      )}
    >
      {icon}
      {label}
    </button>
  )
}

export default App
