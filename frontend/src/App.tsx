import { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
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
  ChevronDown,
  ChevronUp,
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
  Keyboard,
  Copy,
  Globe,
} from 'lucide-react'
import { cn } from './lib/utils'
import {
  getOpportunities,
  searchPolymarketOpportunities,
  getScannerStatus,
  triggerScan,
  getStrategies,
  startScanner,
  pauseScanner,
  Opportunity
} from './services/api'
import { useWebSocket } from './hooks/useWebSocket'
import { useKeyboardShortcuts, Shortcut } from './hooks/useKeyboardShortcuts'
import { useDataSimulation } from './hooks/useDataSimulation'
import { shortcutsHelpOpenAtom, simulationEnabledAtom } from './store/atoms'

// shadcn/ui components
import { Button } from './components/ui/button'
import { Card, CardContent } from './components/ui/card'
import { Badge } from './components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs'
import { Input } from './components/ui/input'
import { Separator } from './components/ui/separator'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip'

// App components
import OpportunityCard from './components/OpportunityCard'
import TradeExecutionModal from './components/TradeExecutionModal'
import WalletTracker from './components/WalletTracker'
import SimulationPanel from './components/SimulationPanel'
import LiveAccountPanel from './components/LiveAccountPanel'
import WalletAnalysisPanel from './components/WalletAnalysisPanel'
import TradingPanel from './components/TradingPanel'
import CopyTradingPanel from './components/CopyTradingPanel'
import PositionsPanel from './components/PositionsPanel'
import PerformancePanel from './components/PerformancePanel'
import RecentTradesPanel from './components/RecentTradesPanel'
import SettingsPanel from './components/SettingsPanel'
import AIPanel from './components/AIPanel'
import AICopilotPanel from './components/AICopilotPanel'
import AICommandBar from './components/AICommandBar'
import DataFreshnessIndicator from './components/DataFreshnessIndicator'
import ThemeToggle from './components/ThemeToggle'
import KeyboardShortcutsHelp from './components/KeyboardShortcutsHelp'
import DiscoveryPanel from './components/DiscoveryPanel'

type Tab = 'opportunities' | 'trading' | 'accounts' | 'traders' | 'positions' | 'performance' | 'ai' | 'settings'
type AccountsSubTab = 'paper' | 'live'
type TradersSubTab = 'tracked' | 'leaderboard' | 'discover' | 'analysis'
type TradingSubTab = 'auto' | 'copy'

const ITEMS_PER_PAGE = 20

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('opportunities')
  const [accountsSubTab, setAccountsSubTab] = useState<AccountsSubTab>('paper')
  const [tradersSubTab, setTradersSubTab] = useState<TradersSubTab>('leaderboard')
  const [tradingSubTab, setTradingSubTab] = useState<TradingSubTab>('auto')
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [minProfit, setMinProfit] = useState(2.5)
  const [maxRisk, setMaxRisk] = useState(1.0)
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(0)
  const [sortBy, setSortBy] = useState<string>('ai_score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [walletToAnalyze, setWalletToAnalyze] = useState<string | null>(null)
  const [walletUsername, setWalletUsername] = useState<string | null>(null)
  const [opportunitiesView, setOpportunitiesView] = useState<'arbitrage' | 'recent_trades'>('arbitrage')
  const [polymarketSearchQuery, setPolymarketSearchQuery] = useState('')
  const [polymarketSearchSubmitted, setPolymarketSearchSubmitted] = useState('')
  const [searchMode, setSearchMode] = useState<'current' | 'polymarket'>('current')
  const [executingOpportunity, setExecutingOpportunity] = useState<Opportunity | null>(null)
  const [copilotOpen, setCopilotOpen] = useState(false)
  const [copilotContext, setCopilotContext] = useState<{ type?: string; id?: string; label?: string }>({})
  const [commandBarOpen, setCommandBarOpen] = useState(false)
  const [shortcutsHelpOpen, setShortcutsHelpOpen] = useAtom(shortcutsHelpOpenAtom)
  const [simulationEnabled] = useAtom(simulationEnabledAtom)
  const queryClient = useQueryClient()

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
    queryKey: ['opportunities', selectedStrategy, selectedCategory, minProfit, maxRisk, searchQuery, sortBy, sortDir, currentPage],
    queryFn: () => getOpportunities({
      strategy: selectedStrategy || undefined,
      category: selectedCategory || undefined,
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
      sort_by: sortBy,
      sort_dir: sortDir,
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

  // Polymarket search query (only runs when user submits a search)
  const { data: polymarketSearchData, isLoading: polySearchLoading } = useQuery({
    queryKey: ['polymarket-search', polymarketSearchSubmitted],
    queryFn: () => searchPolymarketOpportunities({ q: polymarketSearchSubmitted, limit: 50 }),
    enabled: !!polymarketSearchSubmitted && searchMode === 'polymarket',
    staleTime: 60000,
  })

  const polymarketResults = polymarketSearchData?.opportunities || []
  const polymarketTotal = polymarketSearchData?.total || 0

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

  // Data simulation between scan cycles
  const { simulatedData: displayOpportunities } = useDataSimulation(
    opportunities,
    { enabled: simulationEnabled && activeTab === 'opportunities' && opportunitiesView === 'arbitrage' }
  )

  // Keyboard shortcuts
  const shortcuts: Shortcut[] = useMemo(() => [
    { key: '1', description: 'Go to Opportunities', category: 'Navigation', action: () => setActiveTab('opportunities') },
    { key: '2', description: 'Go to Trading', category: 'Navigation', action: () => setActiveTab('trading') },
    { key: '3', description: 'Go to Accounts', category: 'Navigation', action: () => setActiveTab('accounts') },
    { key: '4', description: 'Go to Traders', category: 'Navigation', action: () => setActiveTab('traders') },
    { key: '5', description: 'Go to Positions', category: 'Navigation', action: () => setActiveTab('positions') },
    { key: '6', description: 'Go to Performance', category: 'Navigation', action: () => setActiveTab('performance') },
    { key: '7', description: 'Go to AI', category: 'Navigation', action: () => setActiveTab('ai') },
    { key: '8', description: 'Go to Settings', category: 'Navigation', action: () => setActiveTab('settings') },
    { key: 'k', ctrl: true, description: 'Open AI Command Bar', category: 'Actions', action: () => setCommandBarOpen(v => !v) },
    { key: 'r', ctrl: true, description: 'Trigger Manual Scan', category: 'Actions', action: () => scanMutation.mutate() },
    { key: '/', description: 'Focus Search', category: 'Actions', action: () => {
      setActiveTab('opportunities')
      setTimeout(() => document.querySelector<HTMLInputElement>('input[placeholder*="Search"]')?.focus(), 100)
    }},
    { key: '.', ctrl: true, description: 'Toggle AI Copilot', category: 'Actions', action: () => setCopilotOpen(v => !v) },
    { key: '?', shift: true, description: 'Show Keyboard Shortcuts', category: 'Help', action: () => setShortcutsHelpOpen(v => !v) },
    { key: 'Escape', description: 'Close Modals / Panels', category: 'Help', action: () => {
      setShortcutsHelpOpen(false)
      setCommandBarOpen(false)
      setCopilotOpen(false)
      setExecutingOpportunity(null)
    }},
  ], [scanMutation, setShortcutsHelpOpen])

  useKeyboardShortcuts(shortcuts)

  // Stats
  const totalProfit = displayOpportunities.reduce((sum, o) => sum + o.net_profit, 0)
  const avgROI = displayOpportunities.length > 0
    ? displayOpportunities.reduce((sum, o) => sum + o.roi_percent, 0) / displayOpportunities.length
    : 0

  const totalPages = Math.ceil(totalOpportunities / ITEMS_PER_PAGE)

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background">
        {/* Header */}
        <header className="border-b border-border bg-background/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                  <Terminal className="w-6 h-6 text-green-500" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-green-400">HOMERUN</h1>
                  <p className="text-xs text-muted-foreground">Polymarket Arbitrage Scanner</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {/* Connection Status */}
                <Badge
                  variant="outline"
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full font-normal",
                    isConnected
                      ? "border-green-500/30 bg-green-500/10 text-green-500"
                      : "border-red-500/30 bg-red-500/10 text-red-500"
                  )}
                >
                  <span className={cn(
                    "w-2 h-2 rounded-full",
                    isConnected ? "bg-green-500" : "bg-red-500"
                  )} />
                  {isConnected ? 'Live' : 'Disconnected'}
                </Badge>

                {/* Data Freshness Indicator */}
                <DataFreshnessIndicator lastUpdated={status?.last_scan} />

                {/* Theme Toggle */}
                <ThemeToggle />

                {/* Keyboard Shortcuts Help */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShortcutsHelpOpen(true)}
                      className="px-2 text-muted-foreground hover:text-foreground"
                    >
                      <Keyboard className="w-3.5 h-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Keyboard shortcuts (?)</TooltipContent>
                </Tooltip>

                {/* Scanner Status & Controls */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => status?.enabled ? pauseMutation.mutate() : startMutation.mutate()}
                      disabled={pauseMutation.isPending || startMutation.isPending}
                      className={cn(
                        "flex items-center gap-1.5",
                        status?.enabled
                          ? "bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20 hover:text-yellow-500"
                          : "bg-green-500/10 text-green-500 hover:bg-green-500/20 hover:text-green-500"
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
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {status?.enabled ? 'Pause scanner' : 'Start scanner'}
                  </TooltipContent>
                </Tooltip>

                {/* AI Command Bar Toggle */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCommandBarOpen(true)}
                      className="flex items-center gap-2 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 hover:text-purple-400 border-purple-500/20"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      <span className="hidden sm:inline">AI</span>
                      <kbd className="hidden sm:inline px-1.5 py-0.5 bg-purple-500/10 rounded text-[10px] text-purple-400 border border-purple-500/20">
                        <Command className="w-2.5 h-2.5 inline" />K
                      </kbd>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>AI Command Bar (Cmd+K)</TooltipContent>
                </Tooltip>

                {/* AI Copilot Toggle */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCopilotOpen(!copilotOpen)}
                      className={cn(
                        "flex items-center gap-1.5",
                        copilotOpen
                          ? "bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30 hover:text-purple-400"
                          : "bg-card text-muted-foreground hover:text-purple-400 border-border hover:border-purple-500/30"
                      )}
                    >
                      <Bot className="w-3.5 h-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>AI Copilot (Ctrl+.)</TooltipContent>
                </Tooltip>

                {/* Scan Button */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={() => scanMutation.mutate()}
                      disabled={scanMutation.isPending}
                      className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white"
                    >
                      <RefreshCw className={cn("w-4 h-4", scanMutation.isPending && "animate-spin")} />
                      Scan Now
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Trigger Manual Scan (Ctrl+R)</TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </header>

        {/* Stats Bar */}
        <div className="border-b border-border bg-background">
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

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as Tab)}>
          {/* Navigation Tabs */}
          <div className="border-b border-border">
            <div className="max-w-7xl mx-auto px-4">
              <TabsList className="h-auto w-full justify-start bg-transparent p-0 gap-1 rounded-none">
                <TabsTrigger
                  value="opportunities"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Zap className="w-4 h-4" />
                  Opportunities
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">1</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="trading"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Bot className="w-4 h-4" />
                  Trading
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">2</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="accounts"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Wallet className="w-4 h-4" />
                  Accounts
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">3</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="traders"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Users className="w-4 h-4" />
                  Traders
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">4</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="positions"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Briefcase className="w-4 h-4" />
                  Positions
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">5</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="performance"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <BarChart3 className="w-4 h-4" />
                  Performance
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">6</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="ai"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Brain className="w-4 h-4" />
                  AI
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">7</kbd>
                </TabsTrigger>
                <TabsTrigger
                  value="settings"
                  className="flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-none border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent text-muted-foreground hover:text-foreground/80"
                >
                  <Settings className="w-4 h-4" />
                  Settings
                  <kbd className="hidden lg:inline px-1 py-0.5 text-[10px] font-mono bg-muted rounded text-muted-foreground border border-border">8</kbd>
                </TabsTrigger>
              </TabsList>
            </div>
          </div>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto px-4 py-6">
            {/* Opportunities Tab */}
            <TabsContent value="opportunities" className="mt-0">
              <div>
                {/* View Toggle */}
                <div className="flex items-center gap-2 mb-6">
                  <Button
                    variant="outline"
                    onClick={() => setOpportunitiesView('arbitrage')}
                    className={cn(
                      "flex items-center gap-2",
                      opportunitiesView === 'arbitrage'
                        ? "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30 hover:text-green-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Zap className="w-4 h-4" />
                    Markets
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setOpportunitiesView('recent_trades')}
                    className={cn(
                      "flex items-center gap-2",
                      opportunitiesView === 'recent_trades'
                        ? "bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30 hover:text-orange-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Activity className="w-4 h-4" />
                    Tracked Traders
                  </Button>
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
                {/* Search Mode Toggle + Search Input */}
                <div className="mb-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setSearchMode('current')}
                      className={cn(
                        'px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
                        searchMode === 'current'
                          ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                          : 'bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent'
                      )}
                    >
                      Current Opportunities
                    </button>
                    <button
                      onClick={() => setSearchMode('polymarket')}
                      className={cn(
                        'px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5',
                        searchMode === 'polymarket'
                          ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                          : 'bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent'
                      )}
                    >
                      <Globe className="w-3.5 h-3.5" />
                      Search All Polymarket
                    </button>
                  </div>

                  {searchMode === 'current' ? (
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <Input
                        type="text"
                        placeholder="Search current opportunities by market, event, or keyword..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 bg-card border-border"
                      />
                    </div>
                  ) : (
                    <form
                      onSubmit={(e) => {
                        e.preventDefault()
                        if (polymarketSearchQuery.trim()) {
                          setPolymarketSearchSubmitted(polymarketSearchQuery.trim())
                        }
                      }}
                      className="flex gap-2"
                    >
                      <div className="relative flex-1">
                        <Globe className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-blue-400" />
                        <Input
                          type="text"
                          placeholder="Search all Polymarket markets (e.g. 'bitcoin', 'election', 'FIFA')..."
                          value={polymarketSearchQuery}
                          onChange={(e) => setPolymarketSearchQuery(e.target.value)}
                          className="pl-10 bg-card border-blue-500/20 focus:border-blue-500/40"
                        />
                      </div>
                      <Button
                        type="submit"
                        disabled={!polymarketSearchQuery.trim() || polySearchLoading}
                        className="bg-blue-500 hover:bg-blue-600 text-white"
                      >
                        {polySearchLoading ? (
                          <RefreshCw className="w-4 h-4 animate-spin" />
                        ) : (
                          <Search className="w-4 h-4" />
                        )}
                        Search
                      </Button>
                    </form>
                  )}
                </div>

                {searchMode === 'polymarket' ? (
                  /* ========== Polymarket Search Results ========== */
                  <>
                    {polySearchLoading ? (
                      <div className="flex items-center justify-center py-12">
                        <RefreshCw className="w-8 h-8 animate-spin text-blue-400" />
                        <span className="ml-3 text-muted-foreground">Searching Polymarket and analyzing opportunities...</span>
                      </div>
                    ) : !polymarketSearchSubmitted ? (
                      <div className="text-center py-12">
                        <Globe className="w-12 h-12 text-blue-400/30 mx-auto mb-4" />
                        <p className="text-muted-foreground">Search all of Polymarket for arbitrage opportunities</p>
                        <p className="text-sm text-muted-foreground/70 mt-1">
                          Enter a keyword above and press Search to find markets and analyze them
                        </p>
                      </div>
                    ) : polymarketResults.length === 0 ? (
                      <div className="text-center py-12">
                        <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
                        <p className="text-muted-foreground">No arbitrage opportunities found for &quot;{polymarketSearchSubmitted}&quot;</p>
                        <p className="text-sm text-muted-foreground/70 mt-1">
                          Try different keywords or broader search terms
                        </p>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-center gap-2 mb-4">
                          <Badge variant="outline" className="text-xs text-blue-400 border-blue-500/20 bg-blue-500/10">
                            {polymarketTotal} opportunities found for &quot;{polymarketSearchSubmitted}&quot;
                          </Badge>
                        </div>
                        <div className="space-y-4">
                          {polymarketResults.map((opp) => (
                            <OpportunityCard
                              key={opp.id}
                              opportunity={opp}
                              onExecute={setExecutingOpportunity}
                              onOpenCopilot={handleOpenCopilotForOpportunity}
                            />
                          ))}
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  /* ========== Current Opportunities (filters + list + pagination) ========== */
                  <>
                {/* Filters */}
                <div className="flex gap-4 mb-6">
                  <div className="flex-1">
                    <label className="block text-xs text-muted-foreground mb-1">Strategy</label>
                    <select
                      value={selectedStrategy}
                      onChange={(e) => setSelectedStrategy(e.target.value)}
                      className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm"
                    >
                      <option value="">All Strategies</option>
                      {strategies.map((s) => (
                        <option key={s.type} value={s.type}>{s.name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex-1">
                    <label className="block text-xs text-muted-foreground mb-1">Category</label>
                    <select
                      value={selectedCategory}
                      onChange={(e) => setSelectedCategory(e.target.value)}
                      className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm"
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
                    <label className="block text-xs text-muted-foreground mb-1">Min Profit %</label>
                    <Input
                      type="number"
                      value={minProfit}
                      onChange={(e) => setMinProfit(parseFloat(e.target.value) || 0)}
                      step={0.5}
                      min={0}
                      className="bg-card border-border"
                    />
                  </div>
                  <div className="w-48">
                    <label className="block text-xs text-muted-foreground mb-1">Max Risk Score: {maxRisk.toFixed(1)}</label>
                    <input
                      type="range"
                      value={maxRisk}
                      onChange={(e) => setMaxRisk(parseFloat(e.target.value))}
                      step="0.1"
                      min="0"
                      max="1"
                      className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer mt-2"
                    />
                  </div>
                </div>

                {/* Sort Controls */}
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xs text-muted-foreground">Sort:</span>
                  {([
                    ['roi', 'ROI'],
                    ['ai_score', 'AI Score'],
                    ['profit', 'Profit'],
                    ['liquidity', 'Liquidity'],
                    ['risk', 'Risk'],
                  ] as const).map(([key, label]) => (
                    <button
                      key={key}
                      onClick={() => {
                        if (sortBy === key) {
                          setSortDir(d => d === 'desc' ? 'asc' : 'desc')
                        } else {
                          setSortBy(key)
                          setSortDir('desc')
                        }
                      }}
                      className={cn(
                        'px-2.5 py-1 rounded text-xs font-medium transition-colors',
                        sortBy === key
                          ? 'bg-primary/20 text-primary'
                          : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                      )}
                    >
                      {label}
                      {sortBy === key && (
                        sortDir === 'desc'
                          ? <ChevronDown className="w-3 h-3 inline ml-0.5" />
                          : <ChevronUp className="w-3 h-3 inline ml-0.5" />
                      )}
                    </button>
                  ))}
                </div>

                {/* Opportunities List */}
                {oppsLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
                  </div>
                ) : displayOpportunities.length === 0 ? (
                  <div className="text-center py-12">
                    <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
                    <p className="text-muted-foreground">No arbitrage opportunities found</p>
                    <p className="text-sm text-muted-foreground/70 mt-1">
                      Try lowering the minimum profit threshold
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="space-y-4">
                      {displayOpportunities.map((opp) => (
                        <OpportunityCard
                          key={opp.id}
                          opportunity={opp}
                          onExecute={setExecutingOpportunity}
                          onOpenCopilot={handleOpenCopilotForOpportunity}
                        />
                      ))}
                    </div>

                    {/* Pagination */}
                    <div className="mt-6">
                      <Separator />
                      <div className="flex items-center justify-between pt-4">
                        <div className="text-sm text-muted-foreground">
                          Showing {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalOpportunities)} of {totalOpportunities}
                          {searchQuery && ` (filtered by "${searchQuery}")`}
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                            disabled={currentPage === 0}
                          >
                            <ChevronLeft className="w-4 h-4" />
                            Previous
                          </Button>
                          <span className="px-3 py-1.5 bg-card rounded-lg text-sm border border-border">
                            Page {currentPage + 1} of {totalPages || 1}
                          </span>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentPage(p => p + 1)}
                            disabled={currentPage >= totalPages - 1}
                          >
                            Next
                            <ChevronRight className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </>
                )}
                  </>
                )}
                </>
                )}
              </div>
            </TabsContent>

            {/* Trading Tab - Auto Trading + Copy Trading */}
            <TabsContent value="trading" forceMount className="mt-0 data-[state=inactive]:hidden">
              {/* Trading Subtabs Navigation */}
              <div className="flex items-center gap-2 mb-6">
                <Button
                  variant="outline"
                  onClick={() => setTradingSubTab('auto')}
                  className={cn(
                    "flex items-center gap-2",
                    tradingSubTab === 'auto'
                      ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <Bot className="w-4 h-4" />
                  Auto Trader
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setTradingSubTab('copy')}
                  className={cn(
                    "flex items-center gap-2",
                    tradingSubTab === 'copy'
                      ? "bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30 hover:text-purple-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <Copy className="w-4 h-4" />
                  Copy Trading
                </Button>
              </div>
              {/* Trading Subtab Content */}
              <div className={tradingSubTab === 'auto' ? '' : 'hidden'}>
                <TradingPanel />
              </div>
              <div className={tradingSubTab === 'copy' ? '' : 'hidden'}>
                <CopyTradingPanel />
              </div>
            </TabsContent>

            {/* Accounts Tab with Paper/Live Subtabs */}
            <TabsContent value="accounts" forceMount className="mt-0 data-[state=inactive]:hidden">
              {/* Accounts Subtabs Navigation */}
              <div className="flex items-center gap-2 mb-6">
                <Button
                  variant="outline"
                  onClick={() => setAccountsSubTab('paper')}
                  className={cn(
                    "flex items-center gap-2",
                    accountsSubTab === 'paper'
                      ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <PlayCircle className="w-4 h-4" />
                  Paper Accounts
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setAccountsSubTab('live')}
                  className={cn(
                    "flex items-center gap-2",
                    accountsSubTab === 'live'
                      ? "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30 hover:text-green-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <DollarSign className="w-4 h-4" />
                  Live Account
                </Button>
              </div>
              {/* Accounts Subtab Content */}
              <div className={accountsSubTab === 'paper' ? '' : 'hidden'}>
                <SimulationPanel />
              </div>
              <div className={accountsSubTab === 'live' ? '' : 'hidden'}>
                <LiveAccountPanel />
              </div>
            </TabsContent>

            {/* Traders Tab with Subtabs */}
            <TabsContent value="traders" forceMount className="mt-0 data-[state=inactive]:hidden">
              {/* Traders Subtabs Navigation */}
              <div className="flex items-center gap-2 mb-6">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setTradersSubTab('tracked')}
                  className={cn(
                    "flex items-center gap-2",
                    tradersSubTab === 'tracked'
                      ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <Users className="w-4 h-4" />
                  Tracked
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setTradersSubTab('leaderboard')}
                  className={cn(
                    "flex items-center gap-2",
                    tradersSubTab === 'leaderboard'
                      ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30 hover:text-yellow-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <Trophy className="w-4 h-4" />
                  Leaderboard
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setTradersSubTab('discover')}
                  className={cn(
                    "flex items-center gap-2",
                    tradersSubTab === 'discover'
                      ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30 hover:text-emerald-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <Target className="w-4 h-4" />
                  Discover
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setTradersSubTab('analysis')}
                  className={cn(
                    "flex items-center gap-2",
                    tradersSubTab === 'analysis'
                      ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400"
                      : "bg-card text-muted-foreground hover:text-foreground border-border"
                  )}
                >
                  <Search className="w-4 h-4" />
                  Analysis
                </Button>
              </div>
              {/* Traders Subtab Content */}
              <div className={tradersSubTab === 'tracked' ? '' : 'hidden'}>
                <WalletTracker section="tracked" onAnalyzeWallet={handleAnalyzeWallet} />
              </div>
              <div className={tradersSubTab === 'leaderboard' || tradersSubTab === 'discover' ? '' : 'hidden'}>
                <DiscoveryPanel />
              </div>
              <div className={tradersSubTab === 'analysis' ? '' : 'hidden'}>
                <WalletAnalysisPanel
                  initialWallet={walletToAnalyze}
                  initialUsername={walletUsername}
                  onWalletAnalyzed={() => { setWalletToAnalyze(null); setWalletUsername(null) }}
                />
              </div>
            </TabsContent>

            <TabsContent value="positions" forceMount className="mt-0 data-[state=inactive]:hidden">
              <PositionsPanel />
            </TabsContent>
            <TabsContent value="performance" forceMount className="mt-0 data-[state=inactive]:hidden">
              <PerformancePanel />
            </TabsContent>
            <TabsContent value="ai" forceMount className="mt-0 data-[state=inactive]:hidden">
              <AIPanel />
            </TabsContent>
            <TabsContent value="settings" forceMount className="mt-0 data-[state=inactive]:hidden">
              <SettingsPanel />
            </TabsContent>
          </main>
        </Tabs>

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

        {/* Keyboard Shortcuts Help Modal */}
        <KeyboardShortcutsHelp
          isOpen={shortcutsHelpOpen}
          onClose={() => setShortcutsHelpOpen(false)}
          shortcuts={shortcuts}
        />
      </div>
    </TooltipProvider>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <Card className="border-border">
      <CardContent className="flex items-center gap-3 p-4">
        <div className="p-2 bg-muted rounded-lg">{icon}</div>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="text-lg font-semibold">{value}</p>
        </div>
      </CardContent>
    </Card>
  )
}

export default App
