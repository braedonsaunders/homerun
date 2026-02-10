import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
// framer-motion used in AnimatedNumber component
import {
  TrendingUp,
  RefreshCw,
  Wallet,
  AlertCircle,
  DollarSign,
  Target,
  Zap,
  Activity,
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
  Users,
  Brain,
  Sparkles,
  Command,
  Copy,
  Globe,
  SlidersHorizontal,
  LayoutGrid,
  List,
  Newspaper,
  ArrowUpDown,
} from 'lucide-react'
import { cn } from './lib/utils'
import {
  getOpportunities,
  getOpportunityCounts,
  searchPolymarketOpportunities,
  getScannerStatus,
  triggerScan,
  getStrategies,
  startScanner,
  pauseScanner,
  judgeOpportunitiesBulk,
  getSimulationAccounts,
  Opportunity
} from './services/api'
import { useWebSocket } from './hooks/useWebSocket'
import { useKeyboardShortcuts, Shortcut } from './hooks/useKeyboardShortcuts'
import { useDataSimulation } from './hooks/useDataSimulation'
import { shortcutsHelpOpenAtom, simulationEnabledAtom, accountModeAtom, selectedAccountIdAtom } from './store/atoms'

// shadcn/ui components
import { Button } from './components/ui/button'

import { Badge } from './components/ui/badge'
import { Input } from './components/ui/input'
import { Separator } from './components/ui/separator'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select'

// App components
import OpportunityCard from './components/OpportunityCard'
import OpportunityTable from './components/OpportunityTable'
import OpportunityTerminal from './components/OpportunityTerminal'
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
import ThemeToggle from './components/ThemeToggle'
import KeyboardShortcutsHelp from './components/KeyboardShortcutsHelp'
import DiscoveryPanel from './components/DiscoveryPanel'
import LiveTickerTape from './components/LiveTickerTape'
import AnimatedNumber, { FlashNumber } from './components/AnimatedNumber'
import AccountSettingsFlyout from './components/AccountSettingsFlyout'
import SearchFiltersFlyout from './components/SearchFiltersFlyout'
import AccountModeSelector from './components/AccountModeSelector'
import NewsIntelligencePanel from './components/NewsIntelligencePanel'
import CryptoMarketsPanel from './components/CryptoMarketsPanel'

type Tab = 'opportunities' | 'trading' | 'accounts' | 'traders' | 'positions' | 'performance' | 'ai' | 'settings'
type TradersSubTab = 'discovery' | 'tracked' | 'analysis'
type TradingSubTab = 'auto' | 'copy'

const ITEMS_PER_PAGE = 20

const NAV_ITEMS: { id: Tab; icon: React.ElementType; label: string; shortcut: string }[] = [
  { id: 'opportunities', icon: Zap, label: 'Opportunities', shortcut: '1' },
  { id: 'trading', icon: Bot, label: 'Trading', shortcut: '2' },
  { id: 'accounts', icon: Wallet, label: 'Accounts', shortcut: '3' },
  { id: 'traders', icon: Users, label: 'Traders', shortcut: '4' },
  { id: 'positions', icon: Briefcase, label: 'Positions', shortcut: '5' },
  { id: 'performance', icon: BarChart3, label: 'Performance', shortcut: '6' },
  { id: 'ai', icon: Brain, label: 'AI', shortcut: '7' },
  { id: 'settings', icon: Settings, label: 'Settings', shortcut: '8' },
]

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('opportunities')
  const [accountMode] = useAtom(accountModeAtom)
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const [tradersSubTab, setTradersSubTab] = useState<TradersSubTab>('discovery')
  const [tradingSubTab, setTradingSubTab] = useState<TradingSubTab>('auto')
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [minProfit, setMinProfit] = useState(0)
  const [maxRisk, setMaxRisk] = useState(1.0)
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(0)
  const [sortBy, setSortBy] = useState<string>('ai_score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [walletToAnalyze, setWalletToAnalyze] = useState<string | null>(null)
  const [walletUsername, setWalletUsername] = useState<string | null>(null)
  const [opportunitiesView, setOpportunitiesView] = useState<'arbitrage' | 'recent_trades' | 'news' | 'crypto_markets'>('arbitrage')
  const [oppsViewMode, setOppsViewMode] = useState<'card' | 'list' | 'terminal'>('card')
  const [, setPolymarketSearchQuery] = useState('')
  const [polymarketSearchSubmitted, setPolymarketSearchSubmitted] = useState('')
  const [searchMode, setSearchMode] = useState<'current' | 'polymarket'>('current')
  const [executingOpportunity, setExecutingOpportunity] = useState<Opportunity | null>(null)
  const [copilotOpen, setCopilotOpen] = useState(false)
  const [copilotContext, setCopilotContext] = useState<{ type?: string; id?: string; label?: string }>({})
  const [commandBarOpen, setCommandBarOpen] = useState(false)
  const [accountSettingsOpen, setAccountSettingsOpen] = useState(false)
  const [searchFiltersOpen, setSearchFiltersOpen] = useState(false)
  const [scannerActivity, setScannerActivity] = useState<string>('Idle')
  const [headerSearchQuery, setHeaderSearchQuery] = useState('')
  const [headerSearchOpen, setHeaderSearchOpen] = useState(false)
  const headerSearchRef = useRef<HTMLInputElement>(null)
  const headerSearchContainerRef = useRef<HTMLDivElement>(null)
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
    window.dispatchEvent(new CustomEvent('navigate-ai-section', { detail: section }))
  }, [])

  // Callback for navigating to wallet analysis from WalletTracker
  const handleAnalyzeWallet = (address: string, username?: string) => {
    setWalletToAnalyze(address)
    setWalletUsername(username || null)
    setActiveTab('traders')
    setTradersSubTab('analysis')
  }

  // Header search handler
  const handleHeaderSearch = useCallback((query: string) => {
    const trimmed = query.trim()
    if (!trimmed) return

    // Detect wallet address (0x prefix with hex chars)
    if (/^0x[a-fA-F0-9]{20,}$/.test(trimmed)) {
      setWalletToAnalyze(trimmed)
      setWalletUsername(null)
      setActiveTab('traders')
      setTradersSubTab('analysis')
    }
    // Detect potential username (starts with @ or short alphanumeric)
    else if (trimmed.startsWith('@')) {
      setWalletToAnalyze(trimmed.slice(1))
      setWalletUsername(trimmed.slice(1))
      setActiveTab('traders')
      setTradersSubTab('analysis')
    }
    // Default: search markets
    else {
      setActiveTab('opportunities')
      setOpportunitiesView('arbitrage')
      setSearchMode('polymarket')
      setPolymarketSearchQuery(trimmed)
      setPolymarketSearchSubmitted(trimmed)
    }

    setHeaderSearchQuery('')
    setHeaderSearchOpen(false)
  }, [])

  // Close header search dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (headerSearchContainerRef.current && !headerSearchContainerRef.current.contains(e.target as Node)) {
        setHeaderSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // WebSocket for real-time updates
  const { isConnected, lastMessage } = useWebSocket('/ws')

  // Update data when WebSocket message received — trust WS pushes as primary
  useEffect(() => {
    if (lastMessage?.type === 'opportunities_update' || lastMessage?.type === 'init') {
      // Invalidate so React Query refetches with current filters applied
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['opportunity-counts'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    }
    if (lastMessage?.type === 'scanner_status') {
      // Set status directly from WS push — no HTTP round-trip needed
      if (lastMessage.data) {
        queryClient.setQueryData(['scanner-status'], lastMessage.data)
      }
    }
    if (lastMessage?.type === 'scanner_activity') {
      setScannerActivity(lastMessage.data?.activity || 'Idle')
    }
    if (lastMessage?.type === 'wallet_trade') {
      // A tracked wallet traded — refresh copy trading and recent trades data
      queryClient.invalidateQueries({ queryKey: ['copy-trades'] })
      queryClient.invalidateQueries({ queryKey: ['copy-trading-status'] })
    }
    if (lastMessage?.type === 'news_update') {
      // New news articles arrived — refresh news panel data
      queryClient.invalidateQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-edges'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
    }
  }, [lastMessage, queryClient])

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(0)
  }, [selectedStrategy, selectedCategory, minProfit, maxRisk, searchQuery, polymarketSearchSubmitted])

  // Queries — WS pushes are primary; polling is a degraded fallback.
  // When WS is connected, polls are infrequent. When disconnected, revert to faster polling.
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
    refetchInterval: isConnected ? 30000 : 10000,
  })

  const opportunities = opportunitiesData?.opportunities || []
  const totalOpportunities = opportunitiesData?.total || 0

  const { data: status } = useQuery({
    queryKey: ['scanner-status'],
    queryFn: getScannerStatus,
    refetchInterval: isConnected ? 30000 : 5000,
  })

  // Sync scanner activity from polled status as fallback
  useEffect(() => {
    if (status?.current_activity) {
      setScannerActivity(status.current_activity)
    }
  }, [status?.current_activity])

  const { data: strategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: getStrategies,
  })

  // Fetch simulation accounts for header stats
  const { data: sandboxAccounts = [] } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  const selectedAccount = sandboxAccounts.find(a => a.id === selectedAccountId)

  const { data: opportunityCounts } = useQuery({
    queryKey: ['opportunity-counts', minProfit, maxRisk, searchQuery],
    queryFn: () => getOpportunityCounts({
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
    }),
    refetchInterval: isConnected ? 30000 : 15000,
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

  // Client-side sorting and filtering for polymarket search results
  const processedPolymarketResults = useMemo(() => {
    let results = [...polymarketResults]

    // Apply strategy filter
    if (selectedStrategy) {
      results = results.filter(r => r.strategy === selectedStrategy)
    }

    // Apply category filter
    if (selectedCategory) {
      results = results.filter(r => r.category?.toLowerCase() === selectedCategory.toLowerCase())
    }

    // Apply sort
    const reverse = sortDir !== 'asc'
    const dir = reverse ? -1 : 1
    const effectiveSort = sortBy || 'ai_score'

    if (effectiveSort === 'ai_score') {
      results.sort((a, b) => {
        const aScored = a.ai_analysis && a.ai_analysis.recommendation !== 'pending' ? 1 : 0
        const bScored = b.ai_analysis && b.ai_analysis.recommendation !== 'pending' ? 1 : 0
        if (aScored !== bScored) return (bScored - aScored) * dir
        const aScore = a.ai_analysis?.overall_score || 0
        const bScore = b.ai_analysis?.overall_score || 0
        if (aScore !== bScore) return (bScore - aScore) * dir
        return (b.roi_percent - a.roi_percent) * dir
      })
    } else if (effectiveSort === 'roi') {
      results.sort((a, b) => (b.roi_percent - a.roi_percent) * dir)
    } else if (effectiveSort === 'profit') {
      results.sort((a, b) => (b.net_profit - a.net_profit) * dir)
    } else if (effectiveSort === 'liquidity') {
      results.sort((a, b) => (b.min_liquidity - a.min_liquidity) * dir)
    } else if (effectiveSort === 'risk') {
      results.sort((a, b) => (b.risk_score - a.risk_score) * dir)
    }

    return results
  }, [polymarketResults, selectedStrategy, selectedCategory, sortBy, sortDir])

  const polymarketTotalFiltered = processedPolymarketResults.length
  const polymarketTotalPages = Math.ceil(polymarketTotalFiltered / ITEMS_PER_PAGE)
  const paginatedPolymarketResults = useMemo(() => {
    const start = currentPage * ITEMS_PER_PAGE
    return processedPolymarketResults.slice(start, start + ITEMS_PER_PAGE)
  }, [processedPolymarketResults, currentPage])

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

  const analyzeAllMutation = useMutation({
    mutationFn: () => judgeOpportunitiesBulk(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
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
      headerSearchRef.current?.focus()
      setHeaderSearchOpen(true)
    }},
    { key: '.', ctrl: true, description: 'Toggle AI Copilot', category: 'Actions', action: () => setCopilotOpen(v => !v) },
    { key: '?', shift: true, description: 'Show Keyboard Shortcuts', category: 'Help', action: () => setShortcutsHelpOpen(v => !v) },
    { key: 'Escape', description: 'Close Modals / Panels', category: 'Help', action: () => {
      setShortcutsHelpOpen(false)
      setCommandBarOpen(false)
      setCopilotOpen(false)
      setExecutingOpportunity(null)
      setAccountSettingsOpen(false)
      setSearchFiltersOpen(false)
    }},
  ], [scanMutation, setShortcutsHelpOpen])

  useKeyboardShortcuts(shortcuts)

  const totalPages = Math.ceil(totalOpportunities / ITEMS_PER_PAGE)

  return (
    <TooltipProvider>
      <div className="h-screen flex flex-col overflow-hidden bg-background">
        {/* ==================== Top Bar ==================== */}
        <header className="h-12 border-b border-border/40 bg-background/70 backdrop-blur-xl flex items-center px-4 shrink-0 z-50">
          <div className="flex items-center gap-3 mr-4">
            <div className="w-7 h-7 bg-green-500/15 rounded-lg flex items-center justify-center border border-green-500/20">
              <Terminal className="w-4 h-4 text-green-400" />
            </div>
            <span className="text-sm font-bold text-green-400 tracking-wider font-data">HOMERUN</span>
          </div>

          <AccountModeSelector />

          {/* Inline Account Stats */}
          <div className="hidden md:flex items-center gap-3 text-xs ml-3">
            <div className="stat-pill flex items-center gap-1.5 px-2.5 py-1 rounded-md">
              <Wallet className="w-3 h-3 text-blue-400" />
              <span className="text-muted-foreground">Balance</span>
              <FlashNumber value={selectedAccount?.current_capital ?? 0} prefix="$" decimals={2} className="font-data font-semibold text-foreground data-glow-blue" />
            </div>
            <div className="stat-pill flex items-center gap-1.5 px-2.5 py-1 rounded-md">
              <TrendingUp className="w-3 h-3 text-green-400" />
              <span className="text-muted-foreground">PnL</span>
              <FlashNumber value={selectedAccount?.total_pnl ?? 0} prefix="$" decimals={2} className={cn("font-data font-semibold", (selectedAccount?.total_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400")} />
            </div>
            <div className="stat-pill flex items-center gap-1.5 px-2.5 py-1 rounded-md">
              <DollarSign className="w-3 h-3 text-yellow-400" />
              <span className="text-muted-foreground">ROI</span>
              <FlashNumber value={selectedAccount?.roi_percent ?? 0} suffix="%" decimals={1} className={cn("font-data font-semibold", (selectedAccount?.roi_percent ?? 0) >= 0 ? "text-green-400" : "text-red-400")} />
            </div>
            <div className="stat-pill flex items-center gap-1.5 px-2.5 py-1 rounded-md">
              <Activity className="w-3 h-3 text-purple-400" />
              <span className="text-muted-foreground">Positions</span>
              <AnimatedNumber value={selectedAccount?.open_positions ?? 0} decimals={0} className="font-data font-semibold text-foreground" />
            </div>
          </div>

          {/* Universal Search Bar */}
          <div ref={headerSearchContainerRef} className="relative flex-1 max-w-md mx-4">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleHeaderSearch(headerSearchQuery)
              }}
            >
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                <input
                  ref={headerSearchRef}
                  type="text"
                  value={headerSearchQuery}
                  onChange={(e) => {
                    setHeaderSearchQuery(e.target.value)
                    setHeaderSearchOpen(e.target.value.trim().length > 0)
                  }}
                  onFocus={() => {
                    if (headerSearchQuery.trim()) setHeaderSearchOpen(true)
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      setHeaderSearchOpen(false)
                      headerSearchRef.current?.blur()
                    }
                  }}
                  placeholder="Search markets, wallets, traders..."
                  className="w-full h-7 pl-8 pr-12 text-xs bg-card/60 border border-border/50 rounded-md text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:border-green-500/40 focus:bg-card transition-colors"
                />
                <kbd className="absolute right-2 top-1/2 -translate-y-1/2 px-1.5 py-0.5 text-[9px] font-data bg-muted/50 rounded border border-border/50 text-muted-foreground">
                  /
                </kbd>
              </div>
            </form>

            {/* Search Dropdown */}
            {headerSearchOpen && headerSearchQuery.trim() && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border/60 rounded-lg shadow-xl shadow-black/20 overflow-hidden z-[100]">
                <div className="p-1.5">
                  <button
                    type="button"
                    onClick={() => handleHeaderSearch(headerSearchQuery)}
                    className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs hover:bg-muted/60 transition-colors text-left"
                  >
                    <Globe className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <span className="text-foreground">Search markets for </span>
                      <span className="text-blue-400 font-medium truncate">&quot;{headerSearchQuery.trim()}&quot;</span>
                    </div>
                    <kbd className="px-1 py-0.5 text-[9px] font-data bg-muted/50 rounded border border-border/50 text-muted-foreground shrink-0">Enter</kbd>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSearchQuery(headerSearchQuery.trim())
                      setActiveTab('opportunities')
                      setOpportunitiesView('arbitrage')
                      setSearchMode('current')
                      setHeaderSearchQuery('')
                      setHeaderSearchOpen(false)
                    }}
                    className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs hover:bg-muted/60 transition-colors text-left"
                  >
                    <Target className="w-3.5 h-3.5 text-green-400 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <span className="text-foreground">Filter opportunities for </span>
                      <span className="text-green-400 font-medium truncate">&quot;{headerSearchQuery.trim()}&quot;</span>
                    </div>
                  </button>
                  {/^0x[a-fA-F0-9]{6,}$/i.test(headerSearchQuery.trim()) && (
                    <button
                      type="button"
                      onClick={() => {
                        setWalletToAnalyze(headerSearchQuery.trim())
                        setWalletUsername(null)
                        setActiveTab('traders')
                        setTradersSubTab('analysis')
                        setHeaderSearchQuery('')
                        setHeaderSearchOpen(false)
                      }}
                      className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs hover:bg-muted/60 transition-colors text-left"
                    >
                      <Wallet className="w-3.5 h-3.5 text-yellow-400 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <span className="text-foreground">Analyze wallet </span>
                        <span className="text-yellow-400 font-medium font-data truncate">{headerSearchQuery.trim().slice(0, 10)}...{headerSearchQuery.trim().slice(-4)}</span>
                      </div>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      setActiveTab('traders')
                      setTradersSubTab('discovery')
                      setHeaderSearchQuery('')
                      setHeaderSearchOpen(false)
                    }}
                    className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs hover:bg-muted/60 transition-colors text-left"
                  >
                    <Users className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                    <span className="text-foreground">Browse traders</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-1.5 ml-auto">
            {/* Unified Status Indicator */}
            <Tooltip>
              <TooltipTrigger asChild>
                <div className={cn(
                  "flex items-center gap-2 px-2.5 py-1 rounded-full border text-[10px] font-medium transition-all",
                  isConnected
                    ? status?.last_scan && (Date.now() - new Date(status.last_scan).getTime()) < 60000
                      ? "border-green-500/30 bg-green-500/8 text-green-400"
                      : status?.last_scan && (Date.now() - new Date(status.last_scan).getTime()) < 120000
                        ? "border-yellow-500/30 bg-yellow-500/8 text-yellow-400"
                        : "border-orange-500/30 bg-orange-500/8 text-orange-400"
                    : "border-red-500/30 bg-red-500/8 text-red-400"
                )}>
                  <span className="relative flex h-1.5 w-1.5">
                    {isConnected && status?.last_scan && (Date.now() - new Date(status.last_scan).getTime()) < 60000 && (
                      <span className="absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75 animate-ping" />
                    )}
                    <span className={cn(
                      "relative inline-flex rounded-full h-1.5 w-1.5",
                      isConnected
                        ? status?.last_scan && (Date.now() - new Date(status.last_scan).getTime()) < 60000
                          ? "bg-green-400"
                          : status?.last_scan && (Date.now() - new Date(status.last_scan).getTime()) < 120000
                            ? "bg-yellow-400"
                            : "bg-orange-400"
                        : "bg-red-400"
                    )} />
                  </span>
                  <span>
                    {!isConnected
                      ? 'Offline'
                      : !status?.last_scan || isNaN(new Date(status.last_scan).getTime())
                        ? 'Waiting...'
                        : (() => {
                            const secs = Math.floor((Date.now() - new Date(status.last_scan).getTime()) / 1000)
                            if (secs < 5) return 'Live'
                            if (secs < 60) return `${secs}s`
                            if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
                            return `${Math.floor(secs / 3600)}h ago`
                          })()
                    }
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                {!isConnected
                  ? 'WebSocket disconnected'
                  : status?.last_scan && !isNaN(new Date(status.last_scan).getTime())
                    ? `Connected — Last scan: ${new Date(status.last_scan).toLocaleTimeString()}`
                    : 'Connected — No scan data yet'
                }
              </TooltipContent>
            </Tooltip>
            <ThemeToggle />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => status?.enabled ? pauseMutation.mutate() : startMutation.mutate()}
                  disabled={pauseMutation.isPending || startMutation.isPending}
                  className={cn(
                    "h-7 px-2 text-xs gap-1",
                    status?.enabled
                      ? "bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20 hover:text-yellow-500"
                      : "bg-green-500/10 text-green-500 hover:bg-green-500/20 hover:text-green-500"
                  )}
                >
                  {status?.enabled ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                  {status?.enabled ? 'Pause' : 'Start'}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{status?.enabled ? 'Pause scanner' : 'Start scanner'}</TooltipContent>
            </Tooltip>


          </div>
        </header>

        {/* ==================== Live Ticker Tape ==================== */}
        <LiveTickerTape
          opportunities={displayOpportunities}
          isConnected={isConnected}
          totalOpportunities={totalOpportunities}
          lastScan={status?.last_scan}
          activeStrategies={strategies.length}
        />

        {/* ==================== Main Layout ==================== */}
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar Navigation */}
          <nav className="w-[88px] border-r border-border/30 bg-card/20 backdrop-blur-sm flex flex-col items-center py-3 gap-0.5 shrink-0">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.id
              return (
                <Tooltip key={item.id} delayDuration={0}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setActiveTab(item.id)}
                      className={cn(
                        "w-[72px] h-12 rounded-xl flex flex-col items-center justify-center gap-0.5 transition-all relative group",
                        isActive
                          ? "sidebar-item-active text-green-400"
                          : "text-muted-foreground hover:text-foreground hover:bg-card/60"
                      )}
                    >
                      {isActive && (
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-green-400 rounded-r shadow-[0_0_8px_rgba(0,255,136,0.3)]" />
                      )}
                      <Icon className={cn("w-4 h-4", isActive && "drop-shadow-[0_0_4px_rgba(0,255,136,0.3)]")} />
                      <span className="text-[9px] font-medium leading-none truncate max-w-full">{item.label}</span>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="right" className="flex items-center gap-2">
                    {item.label}
                    <kbd className="px-1 py-0.5 text-[9px] font-data bg-muted rounded border border-border">{item.shortcut}</kbd>
                  </TooltipContent>
                </Tooltip>
              )
            })}
          </nav>

          {/* Content Area */}
          <main className="flex-1 overflow-hidden flex flex-col dot-grid-bg">
            {/* ==================== Opportunities ==================== */}
            {activeTab === 'opportunities' && (
              <div className="flex-1 overflow-y-auto section-enter">
                <div className={cn("mx-auto px-6 py-5", oppsViewMode === 'terminal' ? 'max-w-[1600px]' : 'max-w-[1600px]')}>
                  {/* View Toggle + View Mode */}
                  <div className="flex items-center gap-2 mb-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setOpportunitiesView('arbitrage')}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'arbitrage'
                          ? "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30 hover:text-green-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <Zap className="w-3.5 h-3.5" />
                      Markets
                      {totalOpportunities > 0 && (
                        <span className="ml-0.5 inline-flex items-center justify-center rounded-full bg-green-500/20 text-green-400 text-[10px] font-data font-semibold min-w-[20px] h-4 px-1.5">
                          <AnimatedNumber value={totalOpportunities} decimals={0} className="" />
                        </span>
                      )}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setOpportunitiesView('recent_trades')}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'recent_trades'
                          ? "bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30 hover:text-orange-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <Activity className="w-3.5 h-3.5" />
                      Tracked Traders
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setOpportunitiesView('news')}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'news'
                          ? "bg-amber-500/20 text-amber-400 border-amber-500/30 hover:bg-amber-500/30 hover:text-amber-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <Newspaper className="w-3.5 h-3.5" />
                      News
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setOpportunitiesView('crypto_markets')}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'crypto_markets'
                          ? "bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30 hover:text-orange-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <ArrowUpDown className="w-3.5 h-3.5" />
                      Crypto Markets
                    </Button>

                    {/* View Mode Switcher */}
                    {opportunitiesView === 'arbitrage' && (
                      <div className="flex items-center gap-0.5 ml-3 border border-border/50 rounded-lg p-0.5 bg-card/50">
                        {([
                          { mode: 'card' as const, icon: LayoutGrid, label: 'Cards' },
                          { mode: 'list' as const, icon: List, label: 'List' },
                          { mode: 'terminal' as const, icon: Terminal, label: 'Terminal' },
                        ]).map(({ mode, icon: Icon, label }) => (
                          <Tooltip key={mode} delayDuration={0}>
                            <TooltipTrigger asChild>
                              <button
                                onClick={() => setOppsViewMode(mode)}
                                className={cn(
                                  "p-1.5 rounded-md transition-all",
                                  oppsViewMode === mode
                                    ? "bg-primary/20 text-primary shadow-sm"
                                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                )}
                              >
                                <Icon className="w-3.5 h-3.5" />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">{label}</TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    )}

                    <div className="ml-auto">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSearchFiltersOpen(true)}
                        className="gap-1.5 text-xs h-8 bg-card text-muted-foreground hover:text-orange-400 border-border hover:border-orange-500/30"
                      >
                        <SlidersHorizontal className="w-3.5 h-3.5" />
                        Search Filters
                      </Button>
                    </div>
                  </div>

                  {/* Live Scanning Status Line */}
                  {status?.enabled && opportunitiesView === 'arbitrage' && (
                    <div className="flex items-center gap-2 mb-4 px-3 py-2 rounded-lg bg-card/60 border border-border/30">
                      {scannerActivity.startsWith('Idle') || scannerActivity.startsWith('Scan complete') || scannerActivity.startsWith('Fast scan complete') || scannerActivity.includes('unchanged, skipping') ? (
                        <>
                          <div className="relative flex h-2 w-2 shrink-0">
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
                          </div>
                          <span className="text-xs text-muted-foreground font-data truncate">{scannerActivity}</span>
                        </>
                      ) : scannerActivity.startsWith('Scan error') || scannerActivity.startsWith('Fast scan error') ? (
                        <>
                          <AlertCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                          <span className="text-xs text-red-400 font-data truncate">{scannerActivity}</span>
                        </>
                      ) : (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-400 shrink-0" />
                          <span className="text-xs text-blue-400 font-data truncate">{scannerActivity}</span>
                        </>
                      )}
                    </div>
                  )}

                  {opportunitiesView === 'crypto_markets' ? (
                    <CryptoMarketsPanel
                      onExecute={setExecutingOpportunity}
                      onOpenCopilot={handleOpenCopilotForOpportunity}
                    />
                  ) : opportunitiesView === 'news' ? (
                    <NewsIntelligencePanel />
                  ) : opportunitiesView === 'recent_trades' ? (
                    <RecentTradesPanel
                      onNavigateToWallet={(address) => {
                        setWalletToAnalyze(address)
                        setActiveTab('traders')
                        setTradersSubTab('analysis')
                      }}
                    />
                  ) : (
                    <>
                      {/* Search Input */}
                      <div className="mb-4">
                        {searchMode === 'polymarket' && polymarketSearchSubmitted && (
                          <div className="flex items-center gap-2 mb-3">
                            <Badge variant="outline" className="text-xs text-blue-400 border-blue-500/20 bg-blue-500/10 gap-1.5">
                              <Globe className="w-3 h-3" />
                              Market search: &quot;{polymarketSearchSubmitted}&quot;
                            </Badge>
                            <button
                              onClick={() => {
                                setSearchMode('current')
                                setPolymarketSearchSubmitted('')
                                setPolymarketSearchQuery('')
                              }}
                              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                            >
                              Clear
                            </button>
                          </div>
                        )}
                        {searchMode === 'current' && (
                          <div className="relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input
                              type="text"
                              placeholder="Filter opportunities by market, event, or keyword..."
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              className="pl-10 bg-card border-border h-9"
                            />
                          </div>
                        )}
                      </div>

                      {searchMode === 'polymarket' ? (
                        <>
                          {polySearchLoading ? (
                            <div className="flex items-center justify-center py-12">
                              <RefreshCw className="w-8 h-8 animate-spin text-blue-400" />
                              <span className="ml-3 text-muted-foreground">Searching Polymarket & Kalshi and analyzing opportunities...</span>
                            </div>
                          ) : polymarketResults.length === 0 ? (
                            <div className="text-center py-12">
                              <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
                              <p className="text-muted-foreground">No opportunities found for &quot;{polymarketSearchSubmitted}&quot;</p>
                              <p className="text-sm text-muted-foreground/70 mt-1">
                                Try different keywords or broader search terms
                              </p>
                            </div>
                          ) : (
                            <>
                              {/* Filters */}
                              <div className="flex gap-3 mb-4">
                                <div className="flex-1">
                                  <label className="block text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">Strategy</label>
                                  <Select value={selectedStrategy || '_all'} onValueChange={(v) => setSelectedStrategy(v === '_all' ? '' : v)}>
                                    <SelectTrigger className="w-full bg-card border-border h-8 text-sm">
                                      <SelectValue placeholder="All Strategies" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="_all">All Strategies</SelectItem>
                                      {strategies.map((s) => (
                                        <SelectItem key={s.type} value={s.type}>
                                          {s.name}
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                </div>
                                <div className="flex-1">
                                  <label className="block text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">Category</label>
                                  <Select value={selectedCategory || '_all'} onValueChange={(v) => setSelectedCategory(v === '_all' ? '' : v)}>
                                    <SelectTrigger className="w-full bg-card border-border h-8 text-sm">
                                      <SelectValue placeholder="All Categories" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="_all">All Categories</SelectItem>
                                      {[
                                        { value: 'politics', label: 'Politics' },
                                        { value: 'sports', label: 'Sports' },
                                        { value: 'crypto', label: 'Crypto' },
                                        { value: 'culture', label: 'Culture' },
                                        { value: 'economics', label: 'Economics' },
                                        { value: 'tech', label: 'Tech' },
                                        { value: 'finance', label: 'Finance' },
                                        { value: 'weather', label: 'Weather' },
                                      ].map((cat) => (
                                        <SelectItem key={cat.value} value={cat.value}>
                                          {cat.label}
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                </div>
                              </div>

                              {/* Sort Controls */}
                              <div className="flex items-center gap-2 mb-4">
                                <Badge variant="outline" className="text-xs text-blue-400 border-blue-500/20 bg-blue-500/10">
                                  {polymarketTotalFiltered}{polymarketTotalFiltered !== polymarketTotal ? ` / ${polymarketTotal}` : ''} opportunities for &quot;{polymarketSearchSubmitted}&quot;
                                </Badge>
                                <span className="text-[10px] text-muted-foreground uppercase tracking-wider ml-2">Sort:</span>
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
                                      'px-2 py-1 rounded text-xs font-medium transition-colors',
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

                                <div className="ml-auto">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => analyzeAllMutation.mutate()}
                                    disabled={analyzeAllMutation.isPending || processedPolymarketResults.length === 0}
                                    className="text-xs gap-1.5"
                                  >
                                    {analyzeAllMutation.isPending ? (
                                      <RefreshCw className="w-3 h-3 animate-spin" />
                                    ) : (
                                      <Brain className="w-3 h-3" />
                                    )}
                                    {analyzeAllMutation.isPending ? 'Analyzing...' : 'Analyze All'}
                                  </Button>
                                </div>
                              </div>

                              {/* Search Results Views */}
                              {oppsViewMode === 'terminal' ? (
                                <OpportunityTerminal
                                  opportunities={paginatedPolymarketResults}
                                  onExecute={setExecutingOpportunity}
                                  onOpenCopilot={handleOpenCopilotForOpportunity}
                                  isConnected={isConnected}
                                  totalCount={polymarketTotalFiltered}
                                />
                              ) : oppsViewMode === 'list' ? (
                                <OpportunityTable
                                  opportunities={paginatedPolymarketResults}
                                  onExecute={setExecutingOpportunity}
                                  onOpenCopilot={handleOpenCopilotForOpportunity}
                                />
                              ) : (
                                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
                                  {paginatedPolymarketResults.map((opp) => (
                                    <OpportunityCard
                                      key={opp.id}
                                      opportunity={opp}
                                      onExecute={setExecutingOpportunity}
                                      onOpenCopilot={handleOpenCopilotForOpportunity}
                                    />
                                  ))}
                                </div>
                              )}

                              {/* Pagination */}
                              {polymarketTotalPages > 1 && (
                                <div className="mt-5">
                                  <Separator />
                                  <div className="flex items-center justify-between pt-4">
                                    <div className="text-xs text-muted-foreground">
                                      {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, polymarketTotalFiltered)} of {polymarketTotalFiltered}
                                      {(selectedStrategy || selectedCategory) && ` (filtered from ${polymarketTotal})`}
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Button
                                        variant="outline"
                                        size="sm"
                                        className="h-7 text-xs"
                                        onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                                        disabled={currentPage === 0}
                                      >
                                        <ChevronLeft className="w-3.5 h-3.5" />
                                        Prev
                                      </Button>
                                      <span className="px-2.5 py-1 bg-card rounded-lg text-xs border border-border font-mono">
                                        {currentPage + 1}/{polymarketTotalPages}
                                      </span>
                                      <Button
                                        variant="outline"
                                        size="sm"
                                        className="h-7 text-xs"
                                        onClick={() => setCurrentPage(p => p + 1)}
                                        disabled={currentPage >= polymarketTotalPages - 1}
                                      >
                                        Next
                                        <ChevronRight className="w-3.5 h-3.5" />
                                      </Button>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </>
                          )}
                        </>
                      ) : (
                        <>
                          {/* Filters */}
                          <div className="flex gap-3 mb-4">
                            <div className="flex-1">
                              <label className="block text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">Strategy</label>
                              <Select value={selectedStrategy || '_all'} onValueChange={(v) => setSelectedStrategy(v === '_all' ? '' : v)}>
                                <SelectTrigger className="w-full bg-card border-border h-8 text-sm">
                                  <SelectValue placeholder="All Strategies" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="_all">All Strategies</SelectItem>
                                  {strategies.map((s) => (
                                    <SelectItem
                                      key={s.type}
                                      value={s.type}
                                      suffix={opportunityCounts?.strategies[s.type] != null ? (
                                        <span className="ml-auto pl-2 inline-flex items-center justify-center rounded-full bg-primary/15 text-primary text-[10px] font-medium min-w-[20px] h-4 px-1.5">
                                          {opportunityCounts.strategies[s.type]}
                                        </span>
                                      ) : undefined}
                                    >
                                      {s.name}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="flex-1">
                              <label className="block text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">Category</label>
                              <Select value={selectedCategory || '_all'} onValueChange={(v) => setSelectedCategory(v === '_all' ? '' : v)}>
                                <SelectTrigger className="w-full bg-card border-border h-8 text-sm">
                                  <SelectValue placeholder="All Categories" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="_all">All Categories</SelectItem>
                                  {[
                                    { value: 'politics', label: 'Politics' },
                                    { value: 'sports', label: 'Sports' },
                                    { value: 'crypto', label: 'Crypto' },
                                    { value: 'culture', label: 'Culture' },
                                    { value: 'economics', label: 'Economics' },
                                    { value: 'tech', label: 'Tech' },
                                    { value: 'finance', label: 'Finance' },
                                    { value: 'weather', label: 'Weather' },
                                  ].map((cat) => (
                                    <SelectItem
                                      key={cat.value}
                                      value={cat.value}
                                      suffix={opportunityCounts?.categories[cat.value] != null ? (
                                        <span className="ml-auto pl-2 inline-flex items-center justify-center rounded-full bg-primary/15 text-primary text-[10px] font-medium min-w-[20px] h-4 px-1.5">
                                          {opportunityCounts.categories[cat.value]}
                                        </span>
                                      ) : undefined}
                                    >
                                      {cat.label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="w-32">
                              <label className="block text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">Min Profit %</label>
                              <Input
                                type="number"
                                value={minProfit}
                                onChange={(e) => setMinProfit(parseFloat(e.target.value) || 0)}
                                step={0.5}
                                min={0}
                                className="bg-card border-border h-8"
                              />
                            </div>
                            <div className="w-40">
                              <label className="block text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">Risk: {maxRisk.toFixed(1)}</label>
                              <input
                                type="range"
                                value={maxRisk}
                                onChange={(e) => setMaxRisk(parseFloat(e.target.value))}
                                step="0.1"
                                min="0"
                                max="1"
                                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer mt-1.5"
                              />
                            </div>
                          </div>

                          {/* Sort Controls */}
                          <div className="flex items-center gap-2 mb-4">
                            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Sort:</span>
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
                                  'px-2 py-1 rounded text-xs font-medium transition-colors',
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

                            <div className="ml-auto">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => analyzeAllMutation.mutate()}
                                disabled={analyzeAllMutation.isPending || displayOpportunities.length === 0}
                                className="text-xs gap-1.5"
                              >
                                {analyzeAllMutation.isPending ? (
                                  <RefreshCw className="w-3 h-3 animate-spin" />
                                ) : (
                                  <Brain className="w-3 h-3" />
                                )}
                                {analyzeAllMutation.isPending ? 'Analyzing...' : 'Analyze All'}
                              </Button>
                            </div>
                          </div>

                          {/* Opportunities List */}
                          {oppsLoading ? (
                            <div className="flex items-center justify-center py-12">
                              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
                            </div>
                          ) : displayOpportunities.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-12">
                              {status?.enabled ? (
                                <>
                                  <RefreshCw className="w-10 h-10 animate-spin text-muted-foreground mb-4" />
                                  <p className="text-muted-foreground">
                                    {status?.opportunities_count > 0
                                      ? `${status.opportunities_count} opportunities found but none match current filters`
                                      : 'Scanning for opportunities...'}
                                  </p>
                                  {status?.opportunities_count > 0 && (
                                    <p className="text-sm text-muted-foreground/70 mt-1">
                                      Try lowering the minimum profit % or adjusting filters
                                    </p>
                                  )}
                                </>
                              ) : (
                                <>
                                  <AlertCircle className="w-12 h-12 text-muted-foreground/50 mb-4" />
                                  <p className="text-muted-foreground">No opportunities found</p>
                                  <p className="text-sm text-muted-foreground/70 mt-1">
                                    Try lowering the minimum profit threshold or start the scanner
                                  </p>
                                </>
                              )}
                            </div>
                          ) : (
                            <>
                              {oppsViewMode === 'terminal' ? (
                                <OpportunityTerminal
                                  opportunities={displayOpportunities}
                                  onExecute={setExecutingOpportunity}
                                  onOpenCopilot={handleOpenCopilotForOpportunity}
                                  isConnected={isConnected}
                                  totalCount={totalOpportunities}
                                />
                              ) : oppsViewMode === 'list' ? (
                                <OpportunityTable
                                  opportunities={displayOpportunities}
                                  onExecute={setExecutingOpportunity}
                                  onOpenCopilot={handleOpenCopilotForOpportunity}
                                />
                              ) : (
                                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
                                  {displayOpportunities.map((opp) => (
                                    <OpportunityCard
                                      key={opp.stable_id || opp.id}
                                      opportunity={opp}
                                      onExecute={setExecutingOpportunity}
                                      onOpenCopilot={handleOpenCopilotForOpportunity}
                                    />
                                  ))}
                                </div>
                              )}

                              {/* Pagination */}
                              <div className="mt-5">
                                <Separator />
                                <div className="flex items-center justify-between pt-4">
                                  <div className="text-xs text-muted-foreground">
                                    {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalOpportunities)} of {totalOpportunities}
                                    {searchQuery && ` (filtered)`}
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      className="h-7 text-xs"
                                      onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                                      disabled={currentPage === 0}
                                    >
                                      <ChevronLeft className="w-3.5 h-3.5" />
                                      Prev
                                    </Button>
                                    <span className="px-2.5 py-1 bg-card rounded-lg text-xs border border-border font-mono">
                                      {currentPage + 1}/{totalPages || 1}
                                    </span>
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      className="h-7 text-xs"
                                      onClick={() => setCurrentPage(p => p + 1)}
                                      disabled={currentPage >= totalPages - 1}
                                    >
                                      Next
                                      <ChevronRight className="w-3.5 h-3.5" />
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
              </div>
            )}

            {/* ==================== Trading ==================== */}
            {activeTab === 'trading' && (
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                {/* Sub-tab bar */}
                <div className="shrink-0 px-6 pt-4 pb-0 flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTradingSubTab('auto')}
                    className={cn(
                      "gap-1.5 text-xs h-8",
                      tradingSubTab === 'auto'
                        ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Bot className="w-3.5 h-3.5" />
                    Auto Trader
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTradingSubTab('copy')}
                    className={cn(
                      "gap-1.5 text-xs h-8",
                      tradingSubTab === 'copy'
                        ? "bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30 hover:text-purple-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Copy className="w-3.5 h-3.5" />
                    Copy Trading
                  </Button>
                </div>
                <div className="flex-1 overflow-y-auto px-6 py-4">
                  <div className={tradingSubTab === 'auto' ? '' : 'hidden'}>
                    <TradingPanel />
                  </div>
                  <div className={tradingSubTab === 'copy' ? '' : 'hidden'}>
                    <CopyTradingPanel />
                  </div>
                </div>
              </div>
            )}

            {/* ==================== Accounts ==================== */}
            {activeTab === 'accounts' && (
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                <div className="shrink-0 px-6 pt-4 pb-0 flex items-center gap-2">
                  <div className="ml-auto">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setAccountSettingsOpen(true)}
                      className="gap-1.5 text-xs h-8 bg-card text-muted-foreground hover:text-green-400 border-border hover:border-green-500/30"
                    >
                      <Settings className="w-3.5 h-3.5" />
                      Account Settings
                    </Button>
                  </div>
                </div>
                <div className="flex-1 overflow-y-auto px-6 py-4">
                  <div className={accountMode === 'sandbox' ? '' : 'hidden'}>
                    <SimulationPanel />
                  </div>
                  <div className={accountMode === 'live' ? '' : 'hidden'}>
                    <LiveAccountPanel />
                  </div>
                </div>
              </div>
            )}

            {/* ==================== Traders ==================== */}
            {activeTab === 'traders' && (
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                <div className="shrink-0 px-6 pt-4 pb-0 flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTradersSubTab('discovery')}
                    className={cn(
                      "gap-1.5 text-xs h-8",
                      tradersSubTab === 'discovery'
                        ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30 hover:text-emerald-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Target className="w-3.5 h-3.5" />
                    Discovery
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTradersSubTab('tracked')}
                    className={cn(
                      "gap-1.5 text-xs h-8",
                      tradersSubTab === 'tracked'
                        ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Users className="w-3.5 h-3.5" />
                    Tracked
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTradersSubTab('analysis')}
                    className={cn(
                      "gap-1.5 text-xs h-8",
                      tradersSubTab === 'analysis'
                        ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Search className="w-3.5 h-3.5" />
                    Analysis
                  </Button>
                </div>
                <div className="flex-1 overflow-y-auto px-6 py-4">
                  <div className={tradersSubTab === 'discovery' ? '' : 'hidden'}>
                    <DiscoveryPanel
                      onAnalyzeWallet={handleAnalyzeWallet}
                      onExecuteTrade={setExecutingOpportunity}
                    />
                  </div>
                  <div className={tradersSubTab === 'tracked' ? '' : 'hidden'}>
                    <WalletTracker
                      section="tracked"
                      onAnalyzeWallet={handleAnalyzeWallet}
                      onNavigateToWallet={(address) => {
                        setWalletToAnalyze(address)
                        setTradersSubTab('analysis')
                      }}
                    />
                  </div>
                  <div className={tradersSubTab === 'analysis' ? '' : 'hidden'}>
                    <WalletAnalysisPanel
                      initialWallet={walletToAnalyze}
                      initialUsername={walletUsername}
                      onWalletAnalyzed={() => { setWalletToAnalyze(null); setWalletUsername(null) }}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* ==================== Positions ==================== */}
            {activeTab === 'positions' && (
              <div className="flex-1 overflow-y-auto px-6 py-5 section-enter">
                <PositionsPanel />
              </div>
            )}

            {/* ==================== Performance ==================== */}
            {activeTab === 'performance' && (
              <div className="flex-1 overflow-y-auto px-6 py-5 section-enter">
                <PerformancePanel />
              </div>
            )}

            {/* ==================== AI ==================== */}
            {activeTab === 'ai' && (
              <div className="flex-1 overflow-y-auto px-6 py-5 section-enter">
                <AIPanel />
              </div>
            )}

            {/* ==================== Settings ==================== */}
            {activeTab === 'settings' && (
              <div className="flex-1 overflow-y-auto px-6 py-5 section-enter">
                <SettingsPanel />
              </div>
            )}
          </main>
        </div>

        {/* Trade Execution Modal */}
        {executingOpportunity && (
          <TradeExecutionModal
            opportunity={executingOpportunity}
            onClose={() => setExecutingOpportunity(null)}
          />
        )}

        {/* Floating AI FAB — bottom-right */}
        {!copilotOpen && (
          <div className="fixed bottom-5 right-5 z-40 flex flex-col items-end gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => setCopilotOpen(true)}
                  className="group relative w-11 h-11 rounded-full bg-gradient-to-br from-purple-600 to-blue-600 text-white shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 hover:scale-105 transition-all flex items-center justify-center"
                >
                  <Sparkles className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <kbd className="absolute -top-1 -right-1 px-1 py-0.5 text-[8px] font-data bg-background/90 rounded border border-border/60 text-muted-foreground leading-none">
                    <Command className="w-2 h-2 inline" />K
                  </kbd>
                </button>
              </TooltipTrigger>
              <TooltipContent side="left">AI Copilot (Ctrl+.)</TooltipContent>
            </Tooltip>
          </div>
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

        {/* Account Settings Flyout */}
        <AccountSettingsFlyout
          isOpen={accountSettingsOpen}
          onClose={() => setAccountSettingsOpen(false)}
        />

        {/* Search Filters Flyout */}
        <SearchFiltersFlyout
          isOpen={searchFiltersOpen}
          onClose={() => setSearchFiltersOpen(false)}
        />
      </div>
    </TooltipProvider>
  )
}

export default App
