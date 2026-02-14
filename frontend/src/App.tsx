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
  Globe,
  LayoutGrid,
  List,
  Newspaper,
  ArrowUpDown,
  CloudRain,
} from 'lucide-react'
import { cn } from './lib/utils'
import {
  getOpportunities,
  getOpportunityCounts,
  searchPolymarketOpportunities,
  evaluateSearchResults,
  getScannerStatus,
  triggerScan,
  clearOpportunities,
  getStrategies,
  getWorkersStatus,
  pauseAllWorkers,
  resumeAllWorkers,
  judgeOpportunitiesBulk,
  getSimulationAccounts,
  getNewsWorkflowFindings,
  getWeatherWorkflowOpportunities,
  getCryptoMarkets,
  Opportunity,
  WorkerStatus,
} from './services/api'
import { useWebSocket } from './hooks/useWebSocket'
import { useKeyboardShortcuts, Shortcut } from './hooks/useKeyboardShortcuts'
import { useRealtimeInvalidation } from './hooks/useRealtimeInvalidation'
import { useDisplayedOpportunityRefresh } from './hooks/useDisplayedOpportunityRefresh'
import { shortcutsHelpOpenAtom, selectedAccountIdAtom } from './store/atoms'
import { buildNewsSearchKeywords, processPolymarketSearchResults } from './lib/opportunitySearch'

// shadcn/ui components
import { Button } from './components/ui/button'

import { Input } from './components/ui/input'
import { Separator } from './components/ui/separator'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select'

// App components
import OpportunityCard from './components/OpportunityCard'
import OpportunityTable from './components/OpportunityTable'
import OpportunityTerminal from './components/OpportunityTerminal'
import TradeExecutionModal from './components/TradeExecutionModal'
import AccountsPanel from './components/AccountsPanel'
import WalletAnalysisPanel from './components/WalletAnalysisPanel'
import TradingPanel from './components/TradingPanel'
import RecentTradesPanel from './components/RecentTradesPanel'
import TrackedTradersPanel from './components/TrackedTradersPanel'
import SettingsPanel from './components/SettingsPanel'
import AIPanel from './components/AIPanel'
import AICopilotPanel from './components/AICopilotPanel'
import AICommandBar from './components/AICommandBar'
import PositionsPanel from './components/PositionsPanel'
import PerformancePanel from './components/PerformancePanel'
import ThemeToggle from './components/ThemeToggle'
import KeyboardShortcutsHelp from './components/KeyboardShortcutsHelp'
import DiscoveryPanel from './components/DiscoveryPanel'
import LiveTickerTape from './components/LiveTickerTape'
import AnimatedNumber, { FlashNumber } from './components/AnimatedNumber'
import AccountSettingsFlyout from './components/AccountSettingsFlyout'
import SearchFiltersFlyout from './components/SearchFiltersFlyout'
import CryptoSettingsFlyout from './components/CryptoSettingsFlyout'
import AccountModeSelector from './components/AccountModeSelector'
import NewsIntelligencePanel from './components/NewsIntelligencePanel'
import CryptoMarketsPanel from './components/CryptoMarketsPanel'
import WeatherOpportunitiesPanel from './components/WeatherOpportunitiesPanel'
import WorldIntelligencePanel from './components/WorldIntelligencePanel'
import OpportunityEmptyState from './components/OpportunityEmptyState'

type Tab = 'opportunities' | 'trading' | 'accounts' | 'traders' | 'positions' | 'performance' | 'ai' | 'settings'
type TradersSubTab = 'discovery' | 'pool' | 'tracked' | 'analysis'

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

type WorkerHealthTone = 'green' | 'amber' | 'red'

const WORKER_HEALTH_ORDER = [
  'scanner',
  'discovery',
  'weather',
  'news',
  'crypto',
  'tracked_traders',
  'trader_orchestrator',
  'world_intelligence',
] as const

const WORKER_HEALTH_LABELS: Record<string, string> = {
  scanner: 'Scanner',
  discovery: 'Discovery',
  weather: 'Weather',
  news: 'News',
  crypto: 'Crypto',
  tracked_traders: 'Tracked Traders',
  trader_orchestrator: 'Orchestrator',
  world_intelligence: 'World Intel',
}

const STRATEGY_SUBTYPE_LABELS: Record<string, Record<string, string>> = {
  temporal_decay: {
    certainty_shock: 'Certainty Shock',
    decay_curve: 'Decay Curve',
  },
  btc_eth_highfreq: {
    pure_arb: 'Pure Arb',
    dump_hedge: 'Dump Hedge',
    pre_placed_limits: 'Pre-Placed Limits',
    directional_edge: 'Directional Edge',
  },
  news_edge: {
    buy_yes: 'Buy YES',
    buy_no: 'Buy NO',
  },
  cross_platform: {
    poly_yes_kalshi_no: 'Poly YES + Kalshi NO',
    poly_no_kalshi_yes: 'Poly NO + Kalshi YES',
  },
  settlement_lag: {
    binary_market: 'Binary Market',
    negrisk_bundle: 'NegRisk Bundle',
  },
  negrisk: {
    binary_long: 'Binary Long',
    binary_short: 'Binary Short',
    multi_outcome_long: 'Multi-Outcome Long',
    multi_outcome_short: 'Multi-Outcome Short',
  },
  miracle: {
    impossibility_scan: 'Impossibility Scan',
    stale_market: 'Stale Market',
  },
}

function formatStrategySubtypeLabel(strategyType: string, subtypeKey: string): string {
  const map = STRATEGY_SUBTYPE_LABELS[strategyType] || {}
  if (map[subtypeKey]) return map[subtypeKey]
  return subtypeKey
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

const WORKER_TONE_CLASS: Record<WorkerHealthTone, string> = {
  green: 'bg-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.7)]',
  amber: 'bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.65)]',
  red: 'bg-red-400 shadow-[0_0_10px_rgba(248,113,113,0.7)]',
}

function resolveWorkerHealth(worker?: WorkerStatus): {
  state: string
  tone: WorkerHealthTone
  detail: string
} {
  if (!worker) {
    return { state: 'OFFLINE', tone: 'red', detail: 'No telemetry received yet.' }
  }

  const control = (worker.control || {}) as Record<string, any>
  const paused = Boolean(control.is_paused)
  const enabled =
    typeof control.is_enabled === 'boolean'
      ? Boolean(control.is_enabled)
      : Boolean(worker.enabled)
  const running = Boolean(worker.running)
  const lastError = String(worker.last_error || '').trim()

  if (lastError) {
    return { state: 'ERROR', tone: 'red', detail: lastError }
  }
  if (paused || !enabled) {
    return {
      state: 'PAUSED',
      tone: 'amber',
      detail: String(worker.current_activity || 'Paused by operator'),
    }
  }
  if (running) {
    return {
      state: 'RUNNING',
      tone: 'green',
      detail: String(worker.current_activity || 'Healthy'),
    }
  }
  return {
    state: 'IDLE',
    tone: 'amber',
    detail: String(worker.current_activity || 'Idle'),
  }
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('opportunities')
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const [tradersSubTab, setTradersSubTab] = useState<TradersSubTab>('discovery')
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [selectedStrategySubtype, setSelectedStrategySubtype] = useState<string>('')
  const [showZeroCountStrategies, setShowZeroCountStrategies] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [minProfit, setMinProfit] = useState(0)
  const [maxRisk, setMaxRisk] = useState(1.0)
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(0)
  const [sortBy, setSortBy] = useState<string>('ai_score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [walletToAnalyze, setWalletToAnalyze] = useState<string | null>(null)
  const [walletUsername, setWalletUsername] = useState<string | null>(null)
  const [opportunitiesView, setOpportunitiesView] = useState<'arbitrage' | 'recent_trades' | 'news' | 'weather' | 'crypto_markets' | 'world' | 'search'>('arbitrage')
  const [tradersExecutableCountOverride, setTradersExecutableCountOverride] = useState<number | null>(null)
  const [newsSearchQuery, setNewsSearchQuery] = useState('')
  const [oppsViewMode, setOppsViewMode] = useState<'card' | 'list' | 'terminal'>('card')
  const [polymarketSearchSubmitted, setPolymarketSearchSubmitted] = useState('')
  const [searchSort, setSearchSort] = useState<string>('competitive')
  const [executingOpportunity, setExecutingOpportunity] = useState<Opportunity | null>(null)
  const [copilotOpen, setCopilotOpen] = useState(false)
  const [copilotContext, setCopilotContext] = useState<{ type?: string; id?: string; label?: string }>({})
  const [commandBarOpen, setCommandBarOpen] = useState(false)
  const [accountSettingsOpen, setAccountSettingsOpen] = useState(false)
  const [searchFiltersOpen, setSearchFiltersOpen] = useState(false)
  const [cryptoSettingsOpen, setCryptoSettingsOpen] = useState(false)
  const [scannerActivity, setScannerActivity] = useState<string>('Idle')
  const [headerSearchQuery, setHeaderSearchQuery] = useState('')
  const [headerSearchOpen, setHeaderSearchOpen] = useState(false)
  const headerSearchRef = useRef<HTMLInputElement>(null)
  const headerSearchContainerRef = useRef<HTMLDivElement>(null)
  const [shortcutsHelpOpen, setShortcutsHelpOpen] = useAtom(shortcutsHelpOpenAtom)
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

  // Navigate to news tab with a keyword search from an opportunity
  const handleSearchNewsForOpportunity = useCallback((opp: Opportunity) => {
    setNewsSearchQuery(buildNewsSearchKeywords(opp))
    setOpportunitiesView('news')
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
      setOpportunitiesView('search')
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
  useRealtimeInvalidation(lastMessage, queryClient, setScannerActivity)
  useDisplayedOpportunityRefresh({
    activeTab,
    opportunitiesView,
    queryClient,
  })

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(0)
  }, [selectedStrategy, selectedStrategySubtype, selectedCategory, minProfit, maxRisk, searchQuery, polymarketSearchSubmitted, searchSort])

  useEffect(() => {
    setSelectedStrategySubtype('')
  }, [selectedStrategy])

  useEffect(() => {
    if (opportunitiesView !== 'arbitrage') {
      setSearchFiltersOpen(false)
    }
  }, [opportunitiesView])

  useEffect(() => {
    if (opportunitiesView !== 'recent_trades') {
      setTradersExecutableCountOverride(null)
    }
  }, [opportunitiesView])

  useEffect(() => {
    if (opportunitiesView !== 'crypto_markets') {
      setCryptoSettingsOpen(false)
    }
  }, [opportunitiesView])

  // Queries — WS pushes are primary; polling is a degraded fallback.
  // When WS is connected, polls are infrequent. When disconnected, revert to faster polling.
  const {
    data: opportunitiesData,
    isLoading: oppsLoading,
  } = useQuery({
    queryKey: ['opportunities', selectedStrategy, selectedStrategySubtype, selectedCategory, minProfit, maxRisk, searchQuery, sortBy, sortDir, currentPage],
    queryFn: () => getOpportunities({
      strategy: selectedStrategy || undefined,
      sub_strategy: selectedStrategySubtype || undefined,
      category: selectedCategory || undefined,
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
      sort_by: sortBy,
      sort_dir: sortDir,
      limit: ITEMS_PER_PAGE,
      offset: currentPage * ITEMS_PER_PAGE
    }),
    refetchInterval: isConnected ? false : 10000,
  })

  const opportunities = opportunitiesData?.opportunities || []
  const totalOpportunities = opportunitiesData?.total || 0

  const {
    data: status,
  } = useQuery({
    queryKey: ['scanner-status'],
    queryFn: getScannerStatus,
    refetchInterval: isConnected ? false : 5000,
  })

  const { data: workersData } = useQuery({
    queryKey: ['workers-status'],
    queryFn: getWorkersStatus,
    refetchInterval: 5000,
  })

  const workers = workersData?.workers || []
  const trackedTradersWorker = useMemo(
    () => workers.find((worker) => worker.worker_name === 'tracked_traders'),
    [workers],
  )
  const trackedTradersExecutableCount = useMemo<number | null>(() => {
    const stats = trackedTradersWorker?.stats
    if (!stats || typeof stats !== 'object') return null

    const confluenceRaw = Number((stats as Record<string, unknown>).confluence_executable)
    const insiderRaw = Number((stats as Record<string, unknown>).insider_intents_executable)
    const confluence = Number.isFinite(confluenceRaw) ? Math.max(0, Math.round(confluenceRaw)) : null
    const insider = Number.isFinite(insiderRaw) ? Math.max(0, Math.round(insiderRaw)) : null
    if (confluence == null && insider == null) return null
    return (confluence ?? 0) + (insider ?? 0)
  }, [trackedTradersWorker])

  const workerHealth = useMemo(() => {
    const byName = new Map(workers.map((worker) => [worker.worker_name, worker]))
    const rows = WORKER_HEALTH_ORDER.map((workerName) => {
      const worker = byName.get(workerName)
      const resolved = resolveWorkerHealth(worker)
      return {
        workerName,
        label: WORKER_HEALTH_LABELS[workerName] || workerName,
        ...resolved,
      }
    })
    const red = rows.filter((row) => row.tone === 'red').length
    const amber = rows.filter((row) => row.tone === 'amber').length
    const green = rows.filter((row) => row.tone === 'green').length

    let overallTone: WorkerHealthTone = 'green'
    if (red > 0) {
      overallTone = 'red'
    } else if (amber > 0) {
      overallTone = 'amber'
    }

    return {
      rows,
      overallTone,
      counts: { green, amber, red, total: rows.length },
    }
  }, [workers])

  const globallyPaused = useMemo(() => {
    if (workers.length === 0) {
      return Boolean(status && !status.enabled)
    }
    return workers.every((worker) => Boolean((worker.control || {}).is_paused))
  }, [workers, status?.enabled])

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

  const {
    data: strategyFacetCounts,
  } = useQuery({
    queryKey: ['opportunity-strategy-counts', selectedCategory, selectedStrategySubtype, minProfit, maxRisk, searchQuery],
    queryFn: () => getOpportunityCounts({
      category: selectedCategory || undefined,
      sub_strategy: selectedStrategySubtype || undefined,
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
    }),
    refetchInterval: isConnected ? false : 15000,
  })

  const {
    data: categoryFacetCounts,
  } = useQuery({
    queryKey: ['opportunity-category-counts', selectedStrategy, selectedStrategySubtype, minProfit, maxRisk, searchQuery],
    queryFn: () => getOpportunityCounts({
      strategy: selectedStrategy || undefined,
      sub_strategy: selectedStrategySubtype || undefined,
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
    }),
    refetchInterval: isConnected ? false : 15000,
  })

  const {
    data: subfilterCounts,
  } = useQuery({
    queryKey: ['opportunity-subfilters', selectedStrategy, selectedCategory, minProfit, maxRisk, searchQuery],
    queryFn: () => getOpportunityCounts({
      strategy: selectedStrategy || undefined,
      category: selectedCategory || undefined,
      min_profit: minProfit,
      max_risk: maxRisk,
      search: searchQuery || undefined,
    }),
    enabled: opportunitiesView === 'arbitrage' && !!selectedStrategy,
    refetchInterval: isConnected ? false : 15000,
  })

  const { data: newsWorkflowFindingsCount } = useQuery({
    queryKey: ['news-workflow-findings-count'],
    queryFn: () => getNewsWorkflowFindings({
      actionable_only: true,
      include_debug_rejections: false,
      max_age_hours: 24,
      limit: 1,
    }),
    enabled: activeTab === 'opportunities',
    refetchInterval: 30000,
  })

  const { data: weatherWorkflowExecutableCount } = useQuery({
    queryKey: ['weather-workflow-opportunities', 'count'],
    queryFn: () => getWeatherWorkflowOpportunities({
      include_report_only: false,
      limit: 1,
      offset: 0,
    }),
    enabled: activeTab === 'opportunities',
    refetchInterval: 30000,
  })

  const { data: cryptoMarketCounts } = useQuery({
    queryKey: ['crypto-markets-count'],
    queryFn: () => getCryptoMarkets({ viewer_active: false }),
    enabled: activeTab === 'opportunities',
    refetchInterval: 30000,
  })

  const tradersCount = trackedTradersExecutableCount ?? 0
  const tradersCountDisplay = tradersExecutableCountOverride ?? tradersCount
  const newsCount = newsWorkflowFindingsCount?.total || 0
  const weatherCount = weatherWorkflowExecutableCount?.total || 0
  const cryptoCount = cryptoMarketCounts?.length || 0

  // Polymarket search query (only runs when user submits a search)
  const { data: polymarketSearchData, isLoading: polySearchLoading } = useQuery({
    queryKey: ['polymarket-search', polymarketSearchSubmitted],
    queryFn: () => searchPolymarketOpportunities({ q: polymarketSearchSubmitted, limit: 50 }),
    enabled: !!polymarketSearchSubmitted && opportunitiesView === 'search',
    staleTime: 60000,
  })

  const polymarketResults = polymarketSearchData?.opportunities || []
  const polymarketTotal = polymarketSearchData?.total || 0

  // Resolve strategy filter: plugins are their own strategy type
  const strategyFilterSet = useMemo(() => {
    if (!selectedStrategy) return null
    // For plugins, the type is "plugin_<slug>" and the strategy on opportunities is the slug
    const plugin = strategies.find((s) => s.type === selectedStrategy && s.is_plugin)
    if (plugin?.plugin_slug) return new Set([plugin.plugin_slug])
    return new Set([selectedStrategy])
  }, [selectedStrategy, strategies])

  // Client-side sorting and filtering for polymarket search results
  const processedPolymarketResults = useMemo(
    () => processPolymarketSearchResults(
      polymarketResults,
      strategyFilterSet,
      selectedCategory,
      searchSort
    ),
    [polymarketResults, strategyFilterSet, selectedCategory, searchSort]
  )

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

  const resumeAllMutation = useMutation({
    mutationFn: resumeAllWorkers,
    onSuccess: () => {
      queryClient.invalidateQueries()
    },
  })

  const pauseAllMutation = useMutation({
    mutationFn: pauseAllWorkers,
    onSuccess: () => {
      queryClient.invalidateQueries()
    },
  })

  const analyzeAllMutation = useMutation({
    mutationFn: () => judgeOpportunitiesBulk({ force: true }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
    },
  })

  // Run Strategies on search results
  const [evalStatus, setEvalStatus] = useState<'idle' | 'running' | 'done'>('idle')
  const evaluateMutation = useMutation({
    mutationFn: (conditionIds: string[]) => evaluateSearchResults(conditionIds),
    onMutate: () => setEvalStatus('running'),
    onSuccess: () => {
      setEvalStatus('done')
      // Invalidate opportunities so any newly detected ones appear in the Markets tab
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
      // Reset status after a few seconds
      setTimeout(() => setEvalStatus('idle'), 4000)
    },
    onError: () => {
      setEvalStatus('idle')
    },
  })

  // Opportunities are always rendered from real scanner data.
  const displayOpportunities = opportunities

  const strategySubtypeOptions = useMemo(() => {
    const counts = subfilterCounts?.sub_strategies || {}
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([value, count]) => ({
        value,
        count,
        label: formatStrategySubtypeLabel(selectedStrategy, value),
      }))
  }, [subfilterCounts?.sub_strategies, selectedStrategy])

  const strategyCounts = strategyFacetCounts?.strategies || {}
  const categoryCounts = categoryFacetCounts?.categories || {}
  const visibleStrategies = useMemo(
    () => strategies.filter((s) => {
      if (s.type === selectedStrategy) return true
      if (showZeroCountStrategies) return true
      return (strategyCounts[s.type] || 0) > 0
    }),
    [strategies, selectedStrategy, showZeroCountStrategies, strategyCounts]
  )

  useEffect(() => {
    if (!selectedStrategySubtype) return
    if (!strategySubtypeOptions.some((option) => option.value === selectedStrategySubtype)) {
      setSelectedStrategySubtype('')
    }
  }, [selectedStrategySubtype, strategySubtypeOptions])

  const hasActiveOpportunityFilters =
    !!selectedStrategy
    || !!selectedStrategySubtype
    || !!selectedCategory
    || minProfit > 0
    || maxRisk < 1
    || searchQuery.trim().length > 0

  const refreshMarketsMutation = useMutation({
    mutationFn: async () => {
      await clearOpportunities()
      await triggerScan()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
      queryClient.invalidateQueries({ queryKey: ['opportunity-strategy-counts'] })
      queryClient.invalidateQueries({ queryKey: ['opportunity-category-counts'] })
      queryClient.invalidateQueries({ queryKey: ['opportunity-subfilters'] })
    },
  })

  const isRefreshingMarkets = refreshMarketsMutation.isPending

  const handleRefreshMarkets = useCallback(() => {
    refreshMarketsMutation.mutate()
  }, [refreshMarketsMutation])

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
    { key: 'r', ctrl: true, description: 'Trigger Manual Scan', category: 'Actions', action: () => {
      if (!globallyPaused) {
        scanMutation.mutate()
      }
    } },
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
  ], [globallyPaused, scanMutation, setShortcutsHelpOpen])

  useKeyboardShortcuts(shortcuts)

  const totalPages = Math.ceil(totalOpportunities / ITEMS_PER_PAGE)

  // If data shrinks while user is on a later page, clamp back to the last valid page.
  useEffect(() => {
    if (opportunitiesView !== 'arbitrage') return
    // During page transitions React Query may briefly clear `data`; avoid
    // clamping to page 0 until we have a real opportunities snapshot.
    if (!opportunitiesData) return
    const maxPage = Math.max(totalPages - 1, 0)
    if (currentPage > maxPage) {
      setCurrentPage(maxPage)
    }
  }, [opportunitiesView, currentPage, totalPages, opportunitiesData])

  const scannerIsSettled =
    scannerActivity.startsWith('Idle')
    || scannerActivity.startsWith('Scan complete')
    || scannerActivity.startsWith('Fast scan complete')
    || scannerActivity.includes('unchanged, skipping')
  const scannerHasError =
    scannerActivity.startsWith('Scan error')
    || scannerActivity.startsWith('Fast scan error')
  const marketEmptyState: { title: string; description: string } = (() => {
    if (!status?.enabled) {
      return {
        title: 'No executable market opportunities found',
        description: 'Try lowering the minimum profit threshold or start the scanner',
      }
    }

    if ((status.opportunities_count || 0) > 0) {
      return {
        title: hasActiveOpportunityFilters
          ? `${status.opportunities_count} opportunities found but none match current filters`
          : `${status.opportunities_count} opportunities found, but none currently pass display criteria`,
        description: 'Try lowering the minimum profit % or adjusting filters',
      }
    }

    if (scannerHasError) {
      return {
        title: 'Last scan failed. Check scanner logs and retry.',
        description: 'Retry a scan after resolving the scanner error.',
      }
    }

    if (scannerIsSettled) {
      return {
        title: 'Scan complete, but no opportunities are currently in the pool.',
        description: 'Try adjusting filters or run a fresh scan.',
      }
    }

    return {
      title: 'Scanning for opportunities...',
      description: 'Wait for the next scan cycle to complete.',
    }
  })()

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
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="Worker health status"
                  className="h-8 w-8 rounded-md border border-border/60 bg-card/60 hover:bg-card/90 transition-colors flex items-center justify-center cursor-help"
                >
                  <span className={cn("h-2.5 w-2.5 rounded-full", WORKER_TONE_CLASS[workerHealth.overallTone])} />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" align="end" className="w-[280px] p-2.5">
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>Worker health</span>
                    <span className="font-data">
                      {workerHealth.counts.green}/{workerHealth.counts.total} green
                    </span>
                  </div>
                  <div className="space-y-1">
                    {workerHealth.rows.map((worker) => (
                      <div key={worker.workerName} className="rounded-md border border-border/40 bg-background/70 px-2 py-1.5">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="flex items-center gap-1.5">
                            <span className={cn("h-1.5 w-1.5 rounded-full", WORKER_TONE_CLASS[worker.tone])} />
                            <span>{worker.label}</span>
                          </span>
                          <span className="font-data text-muted-foreground">{worker.state}</span>
                        </div>
                        <p className="mt-1 text-[10px] text-muted-foreground truncate">{worker.detail}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </TooltipContent>
            </Tooltip>

            <ThemeToggle />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => globallyPaused ? resumeAllMutation.mutate() : pauseAllMutation.mutate()}
                  disabled={pauseAllMutation.isPending || resumeAllMutation.isPending}
                  className={cn(
                    "h-7 px-2 text-xs gap-1",
                    globallyPaused
                      ? "bg-green-500/10 text-green-500 hover:bg-green-500/20 hover:text-green-500"
                      : "bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20 hover:text-yellow-500"
                  )}
                >
                  {globallyPaused ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
                  {globallyPaused ? 'Resume' : 'Pause'}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {globallyPaused
                  ? 'Resume all workers and services'
                  : 'Pause trading, scans, LLM calls, and ingestion'}
              </TooltipContent>
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
              <div className={cn("flex-1 section-enter", opportunitiesView === 'world' ? 'overflow-hidden' : 'overflow-y-auto')}>
                <div
                  className={cn(
                    "mx-auto",
                    opportunitiesView === 'world'
                      ? 'h-full min-h-0 px-6 py-4 flex flex-col overflow-hidden max-w-[1600px]'
                      : "px-6 py-5 max-w-[1600px]",
                    oppsViewMode === 'terminal' ? 'max-w-[1600px]' : 'max-w-[1600px]'
                  )}
                >
                  {/* View Toggle + View Mode */}
                  <div className={cn("flex items-center gap-2", opportunitiesView === 'world' ? 'mb-3 shrink-0' : 'mb-4')}>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => { setOpportunitiesView('arbitrage'); setNewsSearchQuery('') }}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'arbitrage'
                          ? "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30 hover:text-green-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <Zap className="w-3.5 h-3.5" />
                      Markets
                      <span className="ml-0.5 inline-flex items-center justify-center rounded-full bg-green-500/20 text-green-400 text-[10px] font-data font-semibold min-w-[20px] h-4 px-1.5">
                        <AnimatedNumber value={totalOpportunities} decimals={0} className="" />
                      </span>
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
                      Traders
                      <span className="ml-0.5 inline-flex items-center justify-center rounded-full bg-orange-500/20 text-orange-400 text-[10px] font-data font-semibold min-w-[20px] h-4 px-1.5">
                        <AnimatedNumber value={tradersCountDisplay} decimals={0} className="" />
                      </span>
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
                      <span className="ml-0.5 inline-flex items-center justify-center rounded-full bg-amber-500/20 text-amber-400 text-[10px] font-data font-semibold min-w-[20px] h-4 px-1.5">
                        <AnimatedNumber value={newsCount} decimals={0} className="" />
                      </span>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setOpportunitiesView('weather')}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'weather'
                          ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <CloudRain className="w-3.5 h-3.5" />
                      Weather
                      <span className="ml-0.5 inline-flex items-center justify-center rounded-full bg-cyan-500/20 text-cyan-400 text-[10px] font-data font-semibold min-w-[20px] h-4 px-1.5">
                        <AnimatedNumber value={weatherCount} decimals={0} className="" />
                      </span>
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
                      Crypto
                      <span className="ml-0.5 inline-flex items-center justify-center rounded-full bg-orange-500/20 text-orange-400 text-[10px] font-data font-semibold min-w-[20px] h-4 px-1.5">
                        <AnimatedNumber value={cryptoCount} decimals={0} className="" />
                      </span>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setOpportunitiesView('world')}
                      className={cn(
                        "gap-1.5 text-xs h-8",
                        opportunitiesView === 'world'
                          ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                          : "bg-card text-muted-foreground hover:text-foreground border-border"
                      )}
                    >
                      <Globe className="w-3.5 h-3.5" />
                      World Intel
                    </Button>

                    {/* Search results subtab — only visible when a search is active */}
                    {polymarketSearchSubmitted && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setOpportunitiesView('search')}
                        className={cn(
                          "gap-1.5 text-xs h-8",
                          opportunitiesView === 'search'
                            ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
                            : "bg-card text-muted-foreground hover:text-foreground border-border"
                        )}
                      >
                        <Globe className="w-3.5 h-3.5" />
                        Search
                        <span className="ml-0.5 max-w-[120px] truncate text-[10px] opacity-70">&quot;{polymarketSearchSubmitted}&quot;</span>
                        <span
                          role="button"
                          tabIndex={0}
                          onClick={(e) => {
                            e.stopPropagation()
                            setPolymarketSearchSubmitted('')
                            setOpportunitiesView('arbitrage')
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              e.stopPropagation()
                              setPolymarketSearchSubmitted('')
                              setOpportunitiesView('arbitrage')
                            }
                          }}
                          className="ml-1 hover:text-red-400 transition-colors cursor-pointer"
                        >
                          &times;
                        </span>
                      </Button>
                    )}

                    {/* View Mode Switcher */}
                    {(opportunitiesView === 'arbitrage' || opportunitiesView === 'search' || opportunitiesView === 'recent_trades' || opportunitiesView === 'weather') && (
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

                    {opportunitiesView === 'arbitrage' && (
                      <div className="ml-auto">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSearchFiltersOpen(true)}
                          className="gap-1.5 text-xs h-8 bg-card text-muted-foreground hover:text-orange-400 border-border hover:border-orange-500/30"
                        >
                          <Settings className="w-3.5 h-3.5" />
                          Settings
                        </Button>
                      </div>
                    )}
                  </div>

                  {/* Live Scanning Status Line */}
                  {status?.enabled && opportunitiesView === 'arbitrage' && (
                    <div className="flex items-center gap-2 mb-4 px-3 py-2 rounded-lg bg-card/60 border border-border/30">
                      {scannerIsSettled ? (
                        <>
                          <div className="relative flex h-2 w-2 shrink-0">
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
                          </div>
                          <span className="text-xs text-muted-foreground font-data truncate">{scannerActivity}</span>
                        </>
                      ) : scannerHasError ? (
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

                  {opportunitiesView === 'search' ? (
                    <>
                      {/* ====== SEARCH RESULTS VIEW (Polymarket-style) ====== */}
                      {polySearchLoading ? (
                        <div className="flex items-center justify-center py-12">
                          <RefreshCw className="w-8 h-8 animate-spin text-blue-400" />
                          <span className="ml-3 text-muted-foreground">Searching Polymarket &amp; Kalshi...</span>
                        </div>
                      ) : polymarketResults.length === 0 ? (
                        <div className="text-center py-12">
                          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
                          <p className="text-muted-foreground">No markets found for &quot;{polymarketSearchSubmitted}&quot;</p>
                          <p className="text-sm text-muted-foreground/70 mt-1">
                            Try different keywords or broader search terms
                          </p>
                        </div>
                      ) : (
                        <>
                          {/* Result count + Sort pills (Polymarket-style) */}
                          <div className="flex items-center gap-2 mb-4 flex-wrap">
                            <span className="text-xs text-muted-foreground font-data">
                              {polymarketTotalFiltered} result{polymarketTotalFiltered !== 1 ? 's' : ''} for <span className="text-blue-400 font-medium">&quot;{polymarketSearchSubmitted}&quot;</span>
                            </span>

                            <Separator orientation="vertical" className="h-4 mx-1" />

                            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Sort by</span>
                            {([
                              ['trending', 'Trending'],
                              ['liquidity', 'Liquidity'],
                              ['volume', 'Volume'],
                              ['newest', 'Newest'],
                              ['ending_soon', 'Ending Soon'],
                              ['competitive', 'Competitive'],
                            ] as const).map(([key, label]) => (
                              <button
                                key={key}
                                onClick={() => setSearchSort(key)}
                                className={cn(
                                  'px-2.5 py-1 rounded-md text-xs font-medium transition-all',
                                  searchSort === key
                                    ? 'bg-blue-500/20 text-blue-400 ring-1 ring-blue-500/30'
                                    : 'bg-muted/40 text-muted-foreground hover:bg-muted/70 hover:text-foreground'
                                )}
                              >
                                {label}
                              </button>
                            ))}

                            <div className="ml-auto flex items-center gap-2">
                              {evalStatus === 'done' && (
                                <span className="text-[10px] text-green-400 font-data animate-in fade-in">
                                  Scan triggered — check Markets tab
                                </span>
                              )}
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => {
                                  const conditionIds = processedPolymarketResults
                                    .map(r => r.markets?.[0]?.id)
                                    .filter(Boolean)
                                  evaluateMutation.mutate(conditionIds)
                                }}
                                disabled={processedPolymarketResults.length === 0 || evaluateMutation.isPending}
                                className={cn(
                                  "text-xs gap-1.5",
                                  evalStatus === 'done'
                                    ? "border-green-500/40 bg-green-500/10 text-green-400"
                                    : "border-green-500/30 text-green-400 hover:bg-green-500/10 hover:text-green-400"
                                )}
                              >
                                {evaluateMutation.isPending ? (
                                  <RefreshCw className="w-3 h-3 animate-spin" />
                                ) : evalStatus === 'done' ? (
                                  <Zap className="w-3 h-3" />
                                ) : (
                                  <Zap className="w-3 h-3" />
                                )}
                                {evaluateMutation.isPending ? 'Scanning...' : evalStatus === 'done' ? 'Scan Running' : 'Run Strategies'}
                              </Button>
                            </div>
                          </div>

                          {/* Category filter row */}
                          <div className="mb-4 rounded-xl border border-border/40 bg-card/40 p-3">
                            <div className="flex gap-3">
                              <div className="flex-1 max-w-xs">
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
                          </div>

                          {/* Search Results */}
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
                                  {selectedCategory && ` (filtered from ${polymarketTotal})`}
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
                  ) : opportunitiesView === 'crypto_markets' ? (
                    <CryptoMarketsPanel
                      onExecute={setExecutingOpportunity}
                      onOpenCopilot={handleOpenCopilotForOpportunity}
                      onOpenCryptoSettings={() => setCryptoSettingsOpen(true)}
                    />
                  ) : opportunitiesView === 'news' ? (
                    <NewsIntelligencePanel initialSearchQuery={newsSearchQuery} />
                  ) : opportunitiesView === 'weather' ? (
                    <WeatherOpportunitiesPanel
                      onExecute={setExecutingOpportunity}
                      viewMode={oppsViewMode}
                    />
                  ) : opportunitiesView === 'world' ? (
                    <div className="flex-1 min-h-0 overflow-hidden">
                      <WorldIntelligencePanel isConnected={isConnected} />
                    </div>
                  ) : opportunitiesView === 'recent_trades' ? (
                    <RecentTradesPanel
                      mode="opportunities"
                      viewMode={oppsViewMode}
                      onOpenCopilot={handleOpenCopilot}
                      onSignalStatsChange={({ executableCount }) => {
                        setTradersExecutableCountOverride(executableCount)
                      }}
                      onNavigateToWallet={(address) => {
                        setWalletToAnalyze(address)
                        setActiveTab('traders')
                        setTradersSubTab('analysis')
                      }}
                    />
                  ) : (
                    <>
                      {/* Markets Controls */}
                      <div className="mb-4 rounded-xl border border-border/40 bg-card/40 p-3">
                        <div className="flex items-center gap-2 overflow-x-auto pb-1">
                          <div className="relative min-w-[220px] flex-1">
                            <Search className="absolute left-2.5 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                            <Input
                              type="text"
                              placeholder="Search markets..."
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              className="pl-8 bg-card border-border h-8 text-xs"
                            />
                          </div>

                          <Select value={selectedStrategy || '_all'} onValueChange={(v) => setSelectedStrategy(v === '_all' ? '' : v)}>
                            <SelectTrigger className="w-[170px] shrink-0 bg-card border-border h-8 text-xs">
                              <SelectValue placeholder="Strategy" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="_all">All Strategies</SelectItem>
                              {visibleStrategies.map((s) => (
                                <SelectItem
                                  key={s.type}
                                  value={s.type}
                                  suffix={strategyCounts[s.type] != null ? (
                                    <span className="ml-auto pl-2 inline-flex items-center justify-center rounded-full bg-primary/15 text-primary text-[10px] font-medium min-w-[20px] h-4 px-1.5">
                                      {strategyCounts[s.type]}
                                    </span>
                                  ) : undefined}
                                >
                                  {s.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>

                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setShowZeroCountStrategies((value) => !value)}
                            className="shrink-0 h-8 text-[11px] px-2.5"
                          >
                            {showZeroCountStrategies ? 'Hide Empty' : 'Show Empty'}
                          </Button>

                          {selectedStrategy && strategySubtypeOptions.length > 0 && (
                            <Select
                              value={selectedStrategySubtype || '_all'}
                              onValueChange={(v) => setSelectedStrategySubtype(v === '_all' ? '' : v)}
                            >
                              <SelectTrigger className="w-[200px] shrink-0 bg-card border-border h-8 text-xs">
                                <SelectValue placeholder="Subfilter" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="_all">All Subfilters</SelectItem>
                                {strategySubtypeOptions.map((option) => (
                                  <SelectItem
                                    key={option.value}
                                    value={option.value}
                                    suffix={(
                                      <span className="ml-auto pl-2 inline-flex items-center justify-center rounded-full bg-primary/15 text-primary text-[10px] font-medium min-w-[20px] h-4 px-1.5">
                                        {option.count}
                                      </span>
                                    )}
                                  >
                                    {option.label}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          )}

                          <Select value={selectedCategory || '_all'} onValueChange={(v) => setSelectedCategory(v === '_all' ? '' : v)}>
                            <SelectTrigger className="w-[155px] shrink-0 bg-card border-border h-8 text-xs">
                              <SelectValue placeholder="Category" />
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
                                  suffix={categoryCounts[cat.value] != null ? (
                                    <span className="ml-auto pl-2 inline-flex items-center justify-center rounded-full bg-primary/15 text-primary text-[10px] font-medium min-w-[20px] h-4 px-1.5">
                                      {categoryCounts[cat.value]}
                                    </span>
                                  ) : undefined}
                                >
                                  {cat.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>

                          <Input
                            type="number"
                            value={minProfit}
                            onChange={(e) => setMinProfit(parseFloat(e.target.value) || 0)}
                            step={0.5}
                            min={0}
                            placeholder="Min %"
                            className="w-[84px] shrink-0 bg-card border-border h-8 text-xs"
                          />

                          <div className="w-[120px] shrink-0">
                            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                              <span>Risk</span>
                              <span>{maxRisk.toFixed(1)}</span>
                            </div>
                            <input
                              type="range"
                              value={maxRisk}
                              onChange={(e) => setMaxRisk(parseFloat(e.target.value))}
                              step="0.1"
                              min="0"
                              max="1"
                              className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer"
                            />
                          </div>

                          <Select
                            value={sortBy}
                            onValueChange={(value) => {
                              setSortBy(value)
                              setSortDir('desc')
                            }}
                          >
                            <SelectTrigger className="w-[130px] shrink-0 bg-card border-border h-8 text-xs">
                              <SelectValue placeholder="Sort" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="roi">ROI</SelectItem>
                              <SelectItem value="ai_score">AI Score</SelectItem>
                              <SelectItem value="profit">Profit</SelectItem>
                              <SelectItem value="liquidity">Liquidity</SelectItem>
                              <SelectItem value="risk">Risk</SelectItem>
                            </SelectContent>
                          </Select>

                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setSortDir((d) => d === 'desc' ? 'asc' : 'desc')}
                            className="w-8 h-8 p-0 shrink-0"
                            title={`Sort direction: ${sortDir === 'desc' ? 'descending' : 'ascending'}`}
                          >
                            {sortDir === 'desc' ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
                          </Button>

                          <Button
                            size="sm"
                            variant="outline"
                            onClick={handleRefreshMarkets}
                            disabled={isRefreshingMarkets}
                            className="text-xs gap-1.5 h-8 px-2.5 shrink-0 whitespace-nowrap"
                          >
                            <RefreshCw className={cn("w-3 h-3", isRefreshingMarkets && "animate-spin")} />
                            Refresh
                          </Button>

                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => analyzeAllMutation.mutate()}
                            disabled={analyzeAllMutation.isPending || displayOpportunities.length === 0}
                            className="text-xs gap-1.5 h-8 px-2.5 shrink-0 whitespace-nowrap"
                          >
                            {analyzeAllMutation.isPending ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : (
                              <Brain className="w-3 h-3" />
                            )}
                            {analyzeAllMutation.isPending ? 'Analyzing...' : 'Analyze'}
                          </Button>
                        </div>
                      </div>

                          {/* Opportunities List */}
                          {oppsLoading ? (
                            <div className="flex items-center justify-center py-12">
                              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
                            </div>
                          ) : displayOpportunities.length === 0 ? (
                            <OpportunityEmptyState
                              title={marketEmptyState.title}
                              description={marketEmptyState.description}
                            />
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
                                      onSearchNews={handleSearchNewsForOpportunity}
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
                </div>
              </div>
            )}

            {/* ==================== Trading ==================== */}
            {activeTab === 'trading' && (
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                <div className="flex-1 overflow-hidden px-6 py-4 min-h-0">
                  <TradingPanel />
                </div>
              </div>
            )}

            {/* ==================== Accounts ==================== */}
            {activeTab === 'accounts' && (
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                <div className="flex-1 overflow-hidden px-6 py-4 min-h-0">
                  <AccountsPanel onOpenSettings={() => setAccountSettingsOpen(true)} />
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
                    onClick={() => setTradersSubTab('pool')}
                    className={cn(
                      "gap-1.5 text-xs h-8",
                      tradersSubTab === 'pool'
                        ? "bg-amber-500/20 text-amber-300 border-amber-500/30 hover:bg-amber-500/30 hover:text-amber-300"
                        : "bg-card text-muted-foreground hover:text-foreground border-border"
                    )}
                  >
                    <Activity className="w-3.5 h-3.5" />
                    Pool
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
                <div className={cn(
                  "flex-1 min-h-0 px-6 py-4",
                  tradersSubTab === 'analysis' ? "overflow-hidden" : "overflow-y-auto"
                )}>
                  <div className={tradersSubTab === 'discovery' ? '' : 'hidden'}>
                    <DiscoveryPanel
                      view="discovery"
                      onAnalyzeWallet={handleAnalyzeWallet}
                    />
                  </div>
                  <div className={tradersSubTab === 'pool' ? '' : 'hidden'}>
                    <DiscoveryPanel
                      view="pool"
                      onAnalyzeWallet={handleAnalyzeWallet}
                    />
                  </div>
                  <div className={tradersSubTab === 'tracked' ? '' : 'hidden'}>
                    <TrackedTradersPanel
                      onAnalyzeWallet={handleAnalyzeWallet}
                      onNavigateToWallet={(address) => {
                        setWalletToAnalyze(address)
                        setTradersSubTab('analysis')
                      }}
                    />
                  </div>
                  <div className={cn("h-full min-h-0", tradersSubTab === 'analysis' ? '' : 'hidden')}>
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
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                <div className="flex-1 overflow-y-auto px-6 py-4 min-h-0">
                  <PositionsPanel />
                </div>
              </div>
            )}

            {/* ==================== Performance ==================== */}
            {activeTab === 'performance' && (
              <div className="flex-1 overflow-hidden flex flex-col section-enter">
                <div className="flex-1 overflow-y-auto px-6 py-4 min-h-0">
                  <PerformancePanel />
                </div>
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
        {!copilotOpen && !(activeTab === 'opportunities' && opportunitiesView === 'world') && (
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
          onManagePlugins={() => {
            setSearchFiltersOpen(false)
            setActiveTab('settings')
            window.dispatchEvent(new CustomEvent('navigate-settings-section', { detail: 'plugins' }))
          }}
        />

        {/* Crypto Settings Flyout */}
        <CryptoSettingsFlyout
          isOpen={cryptoSettingsOpen}
          onClose={() => setCryptoSettingsOpen(false)}
        />
      </div>
    </TooltipProvider>
  )
}

export default App
