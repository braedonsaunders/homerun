import { useState, useEffect, useRef, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Zap,
  Search,
  FileText,
  Activity,
  BookOpen,
  BarChart3,
  ChevronDown,
  ChevronRight,
  Newspaper,
  Layers,
  Clock,
  Shield,
  Target,
  TrendingUp,
  DollarSign,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import {
  getAIStatus,
  analyzeResolution,
  getJudgmentHistory,
  getAgreementStats,
  analyzeMarket,
  analyzeNewsSentiment,
  listSkills,
  executeSkill,
  getResearchSessions,
  getResearchSession,
  getAIUsage,
  searchMarkets,
  MarketSearchResult,
} from '../services/api'

type AITab = 'analyze' | 'judgments' | 'system'
type AnalysisTool = 'resolution' | 'market' | 'news'

export default function AIPanel() {
  const [activeTab, setActiveTab] = useState<AITab>('analyze')
  const [analysisTool, setAnalysisTool] = useState<AnalysisTool>('resolution')

  // Listen for navigation events from command bar (maps old section names to new tabs)
  useEffect(() => {
    const handler = (e: Event) => {
      const section = (e as CustomEvent).detail as string
      if (section === 'resolution' || section === 'market' || section === 'news') {
        setActiveTab('analyze')
        setAnalysisTool(section)
      } else if (section === 'judgments') {
        setActiveTab('judgments')
      } else if (['skills', 'sessions', 'usage', 'status'].includes(section)) {
        setActiveTab('system')
      }
    }
    window.addEventListener('navigate-ai-section', handler)
    return () => window.removeEventListener('navigate-ai-section', handler)
  }, [])

  const tabs = [
    { id: 'analyze' as const, label: 'Analyze', icon: Search },
    { id: 'judgments' as const, label: 'Judgments', icon: Target },
    { id: 'system' as const, label: 'System', icon: Activity },
  ]

  return (
    <div className="space-y-5">
      <StatusHeader />

      {/* Tab Navigation */}
      <div className="flex items-center gap-1 p-1 bg-muted/50 rounded-xl border border-border w-fit">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              activeTab === id
                ? 'bg-background text-foreground shadow-sm border border-border'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'analyze' && (
        <AnalyzeSection tool={analysisTool} onToolChange={setAnalysisTool} />
      )}
      {activeTab === 'judgments' && <JudgmentsSection />}
      {activeTab === 'system' && <SystemSection />}
    </div>
  )
}

// ============================================================
// Status Header (always visible, compact)
// ============================================================

function StatusHeader() {
  const { data: status, isLoading, error } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const { data } = await getAIStatus()
      return data
    },
    refetchInterval: 30000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center gap-3 px-4 py-3 bg-muted/50 rounded-xl border border-border">
        <RefreshCw className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Connecting to AI...</span>
      </div>
    )
  }

  if (error || !status?.enabled) {
    return (
      <div className="flex items-center gap-3 px-4 py-3 bg-yellow-500/5 rounded-xl border border-yellow-500/20">
        <div className="w-2 h-2 rounded-full bg-yellow-500 flex-shrink-0" />
        <span className="text-sm text-yellow-400">
          {error
            ? 'Unable to connect to AI module. Check that the backend is running.'
            : 'Configure an LLM provider (OpenAI or Anthropic) in Settings to enable AI features.'}
        </span>
      </div>
    )
  }

  const cost = status.usage?.estimated_cost ?? status.usage?.total_cost_usd ?? 0
  const spendLimit = status.usage?.spend_limit_usd
  const spendPct = spendLimit ? (cost / spendLimit) * 100 : null

  return (
    <div className="flex items-center justify-between gap-4 px-4 py-3 bg-muted/50 rounded-xl border border-border">
      <div className="flex items-center gap-3 min-w-0">
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium">AI Active</span>
        </div>
        <div className="h-4 w-px bg-border flex-shrink-0" />
        <span className="text-xs text-muted-foreground truncate">
          {status.providers_configured?.join(', ') || 'No providers'}
        </span>
        <div className="h-4 w-px bg-border flex-shrink-0" />
        <span className="text-xs text-muted-foreground flex-shrink-0">
          {status.skills_available ?? 0} skills
        </span>
      </div>
      <div className="flex items-center gap-4 flex-shrink-0">
        <div className="hidden sm:flex items-center gap-3 text-xs text-muted-foreground">
          <span>{status.usage?.total_requests ?? 0} req</span>
          <span>{formatNumber(status.usage?.total_tokens ?? 0)} tok</span>
          <span className="font-medium text-foreground">${cost.toFixed(2)}</span>
        </div>
        {spendPct != null && (
          <div className="flex items-center gap-2">
            <div className="w-16 bg-muted rounded-full h-1.5 border border-border">
              <div
                className={cn(
                  'h-full rounded-full transition-all',
                  spendPct >= 90 ? 'bg-red-500' : spendPct >= 70 ? 'bg-yellow-500' : 'bg-green-500'
                )}
                style={{ width: `${Math.min(100, spendPct)}%` }}
              />
            </div>
            <span className="text-[10px] text-muted-foreground whitespace-nowrap">
              / ${spendLimit!.toFixed(0)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================
// Analyze Section (Resolution + Market + News unified)
// ============================================================

function AnalyzeSection({
  tool,
  onToolChange,
}: {
  tool: AnalysisTool
  onToolChange: (t: AnalysisTool) => void
}) {
  // --- Resolution state ---
  const [marketId, setMarketId] = useState('')
  const [question, setQuestion] = useState('')
  const [description, setDescription] = useState('')
  const [resolutionSource, setResolutionSource] = useState('')
  const [endDate, setEndDate] = useState('')
  const [outcomes, setOutcomes] = useState('')
  const [marketSearch, setMarketSearch] = useState('')
  const [searchResults, setSearchResults] = useState<MarketSearchResult[]>([])
  const [showSearchResults, setShowSearchResults] = useState(false)
  const [showResolutionAdvanced, setShowResolutionAdvanced] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>()

  // --- Market analysis state ---
  const [marketQuery, setMarketQuery] = useState('')
  const [marketAnalysisId, setMarketAnalysisId] = useState('')
  const [marketAnalysisQuestion, setMarketAnalysisQuestion] = useState('')
  const [showMarketAdvanced, setShowMarketAdvanced] = useState(false)

  // --- News state ---
  const [newsQuery, setNewsQuery] = useState('')
  const [newsContext, setNewsContext] = useState('')
  const [maxArticles, setMaxArticles] = useState(5)
  const [showNewsAdvanced, setShowNewsAdvanced] = useState(false)

  // Market-selected event from command bar
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as MarketSearchResult
      if (detail) {
        onToolChange('resolution')
        setMarketId(detail.market_id)
        setQuestion(detail.question)
        setMarketSearch(detail.question)
        setShowSearchResults(false)
      }
    }
    window.addEventListener('market-selected', handler)
    return () => window.removeEventListener('market-selected', handler)
  }, [onToolChange])

  // Click outside to close search dropdown
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSearchResults(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  // Debounced market search
  const doSearch = useCallback(async (query: string) => {
    if (query.length < 2) {
      setSearchResults([])
      return
    }
    try {
      const data = await searchMarkets(query, 8)
      setSearchResults(data.results)
    } catch {
      setSearchResults([])
    }
  }, [])

  useEffect(() => {
    if (tool === 'resolution' && marketSearch.length >= 2) {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => doSearch(marketSearch), 300)
    } else {
      setSearchResults([])
    }
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [marketSearch, doSearch, tool])

  const selectMarket = (m: MarketSearchResult) => {
    setMarketId(m.market_id)
    setQuestion(m.question)
    setMarketSearch(m.question)
    setShowSearchResults(false)
  }

  const clearMarket = () => {
    setMarketId('')
    setQuestion('')
    setMarketSearch('')
  }

  // --- Mutations ---
  const resolutionMutation = useMutation({
    mutationFn: async () => {
      const { data } = await analyzeResolution({
        market_id: marketId,
        question,
        description,
        resolution_source: resolutionSource,
        end_date: endDate,
        outcomes: outcomes ? outcomes.split(',').map((o: string) => o.trim()) : [],
      })
      return data
    },
  })

  const marketMutation = useMutation({
    mutationFn: async () => {
      const { data } = await analyzeMarket({
        query: marketQuery,
        market_id: marketAnalysisId || undefined,
        market_question: marketAnalysisQuestion || undefined,
      })
      return data
    },
  })

  const newsMutation = useMutation({
    mutationFn: async () => {
      const { data } = await analyzeNewsSentiment({
        query: newsQuery,
        market_context: newsContext,
        max_articles: maxArticles,
      })
      return data
    },
  })

  // Tool definitions
  const tools = [
    { id: 'resolution' as const, label: 'Resolution', icon: Shield, desc: 'Analyze resolution criteria' },
    { id: 'market' as const, label: 'Market', icon: TrendingUp, desc: 'Deep-dive market analysis' },
    { id: 'news' as const, label: 'News', icon: Newspaper, desc: 'News sentiment analysis' },
  ]

  const toolColors: Record<AnalysisTool, { bg: string; text: string; border: string; ring: string }> = {
    resolution: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30', ring: 'focus-visible:ring-purple-500' },
    market: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30', ring: 'focus-visible:ring-blue-500' },
    news: { bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/30', ring: 'focus-visible:ring-orange-500' },
  }

  return (
    <div className="space-y-4">
      {/* Tool Picker */}
      <div className="grid grid-cols-3 gap-3">
        {tools.map((t) => {
          const active = tool === t.id
          const c = toolColors[t.id]
          return (
            <button
              key={t.id}
              onClick={() => onToolChange(t.id)}
              className={cn(
                'flex items-center gap-3 p-3 rounded-xl border transition-all text-left',
                active
                  ? `${c.bg} ${c.border}`
                  : 'bg-muted/50 border-border hover:bg-muted hover:border-border'
              )}
            >
              <t.icon className={cn('w-5 h-5 flex-shrink-0', active ? c.text : 'text-muted-foreground')} />
              <div className="min-w-0">
                <p className={cn('text-sm font-medium', active ? 'text-foreground' : 'text-muted-foreground')}>{t.label}</p>
                <p className="text-[11px] text-muted-foreground truncate">{t.desc}</p>
              </div>
            </button>
          )
        })}
      </div>

      {/* ---- Resolution Tool ---- */}
      {tool === 'resolution' && (
        <Card className="p-5">
          <div className="space-y-3">
            {/* Market Search */}
            <div ref={searchRef} className="relative">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                <Input
                  type="text"
                  value={marketSearch}
                  onChange={(e) => {
                    setMarketSearch(e.target.value)
                    setShowSearchResults(true)
                  }}
                  onFocus={() => searchResults.length > 0 && setShowSearchResults(true)}
                  placeholder="Search markets... (e.g., Bitcoin, Fed rate, Trump)"
                  className="pl-10 bg-muted rounded-xl focus-visible:ring-purple-500"
                />
              </div>
              {showSearchResults && searchResults.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-background border border-border rounded-xl shadow-2xl max-h-64 overflow-y-auto">
                  {searchResults.map((m) => (
                    <button
                      key={m.market_id}
                      onClick={() => selectMarket(m)}
                      className="w-full text-left px-3 py-2.5 hover:bg-muted transition-colors border-b border-border last:border-0"
                    >
                      <p className="text-sm text-foreground truncate">{m.question}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {m.event_title && <span>{m.event_title} | </span>}
                        {m.category && <span className="capitalize">{m.category} | </span>}
                        YES: ${m.yes_price?.toFixed(2)} | Liq: ${m.liquidity?.toFixed(0)}
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Selected market indicator */}
            {marketId && (
              <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg px-3 py-2 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-purple-400 flex-shrink-0" />
                <span className="text-xs text-purple-400 truncate flex-1">
                  {question || marketId.slice(0, 30)}
                </span>
                <button
                  onClick={clearMarket}
                  className="text-purple-400/40 hover:text-purple-400 transition-colors flex-shrink-0"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            <div>
              <label className="block text-xs text-muted-foreground mb-1">Question *</label>
              <Input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Auto-filled from search, or type manually"
                className="bg-muted rounded-lg focus-visible:ring-purple-500"
              />
            </div>

            {/* Advanced fields */}
            <button
              onClick={() => setShowResolutionAdvanced(!showResolutionAdvanced)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showResolutionAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Additional fields
            </button>

            {showResolutionAdvanced && (
              <div className="space-y-3 pl-3 border-l-2 border-border">
                <div>
                  <label className="block text-xs text-muted-foreground mb-1">Market ID</label>
                  <Input
                    type="text"
                    value={marketId}
                    onChange={(e) => setMarketId(e.target.value)}
                    placeholder="Auto-filled from search"
                    className="bg-muted rounded-lg focus-visible:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-muted-foreground mb-1">Description</label>
                  <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Market description and resolution criteria..."
                    rows={2}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500 resize-none"
                  />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">End Date</label>
                    <Input
                      type="text"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      placeholder="2025-12-31"
                      className="bg-muted rounded-lg focus-visible:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">Resolution Source</label>
                    <Input
                      type="text"
                      value={resolutionSource}
                      onChange={(e) => setResolutionSource(e.target.value)}
                      placeholder="e.g., CoinGecko"
                      className="bg-muted rounded-lg focus-visible:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">Outcomes</label>
                    <Input
                      type="text"
                      value={outcomes}
                      onChange={(e) => setOutcomes(e.target.value)}
                      placeholder="Yes, No"
                      className="bg-muted rounded-lg focus-visible:ring-purple-500"
                    />
                  </div>
                </div>
              </div>
            )}

            <Button
              onClick={() => resolutionMutation.mutate()}
              disabled={!marketId || !question || resolutionMutation.isPending}
              className={cn(
                'w-full h-auto gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-colors',
                !marketId || !question || resolutionMutation.isPending
                  ? 'bg-muted text-muted-foreground cursor-not-allowed'
                  : 'bg-purple-500 hover:bg-purple-600 text-white'
              )}
            >
              {resolutionMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Shield className="w-4 h-4" />
              )}
              Analyze Resolution
            </Button>
          </div>
        </Card>
      )}

      {/* ---- Market Analysis Tool ---- */}
      {tool === 'market' && (
        <Card className="p-5">
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Query *</label>
              <textarea
                value={marketQuery}
                onChange={(e) => setMarketQuery(e.target.value)}
                placeholder="e.g., What are the chances of a Fed rate cut in March? Analyze recent economic indicators..."
                rows={3}
                className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 resize-none"
              />
            </div>

            <button
              onClick={() => setShowMarketAdvanced(!showMarketAdvanced)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showMarketAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Link to specific market
            </button>

            {showMarketAdvanced && (
              <div className="grid grid-cols-2 gap-3 pl-3 border-l-2 border-border">
                <div>
                  <label className="block text-xs text-muted-foreground mb-1">Market ID</label>
                  <Input
                    type="text"
                    value={marketAnalysisId}
                    onChange={(e) => setMarketAnalysisId(e.target.value)}
                    placeholder="Link to specific market"
                    className="bg-muted rounded-lg focus-visible:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-muted-foreground mb-1">Market Question</label>
                  <Input
                    type="text"
                    value={marketAnalysisQuestion}
                    onChange={(e) => setMarketAnalysisQuestion(e.target.value)}
                    placeholder="Market question for context"
                    className="bg-muted rounded-lg focus-visible:ring-blue-500"
                  />
                </div>
              </div>
            )}

            <Button
              onClick={() => marketMutation.mutate()}
              disabled={!marketQuery || marketMutation.isPending}
              className={cn(
                'w-full h-auto gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-colors',
                !marketQuery || marketMutation.isPending
                  ? 'bg-muted text-muted-foreground cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              )}
            >
              {marketMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <TrendingUp className="w-4 h-4" />
              )}
              Analyze
            </Button>
          </div>
        </Card>
      )}

      {/* ---- News Sentiment Tool ---- */}
      {tool === 'news' && (
        <Card className="p-5">
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Search Query *</label>
              <Input
                type="text"
                value={newsQuery}
                onChange={(e) => setNewsQuery(e.target.value)}
                placeholder="e.g., Federal Reserve interest rate decision"
                className="bg-muted rounded-lg focus-visible:ring-orange-500"
              />
            </div>

            <div>
              <label className="block text-xs text-muted-foreground mb-1">Market Context</label>
              <Input
                type="text"
                value={newsContext}
                onChange={(e) => setNewsContext(e.target.value)}
                placeholder="e.g., Will the Fed cut rates in March 2025?"
                className="bg-muted rounded-lg focus-visible:ring-orange-500"
              />
            </div>

            <button
              onClick={() => setShowNewsAdvanced(!showNewsAdvanced)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showNewsAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Options
            </button>

            {showNewsAdvanced && (
              <div className="pl-3 border-l-2 border-border">
                <label className="block text-xs text-muted-foreground mb-1">Max Articles</label>
                <Input
                  type="number"
                  value={maxArticles}
                  onChange={(e) => setMaxArticles(parseInt(e.target.value) || 5)}
                  min={1}
                  max={20}
                  className="bg-muted rounded-lg focus-visible:ring-orange-500 w-24"
                />
              </div>
            )}

            <Button
              onClick={() => newsMutation.mutate()}
              disabled={!newsQuery || newsMutation.isPending}
              className={cn(
                'w-full h-auto gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-colors',
                !newsQuery || newsMutation.isPending
                  ? 'bg-muted text-muted-foreground cursor-not-allowed'
                  : 'bg-orange-500 hover:bg-orange-600 text-white'
              )}
            >
              {newsMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Search & Analyze
            </Button>
          </div>
        </Card>
      )}

      {/* ---- Results ---- */}

      {/* Resolution results */}
      {tool === 'resolution' && resolutionMutation.data && (
        <Card className="p-5">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Analysis Result</h4>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <ScoreCard label="Clarity" value={resolutionMutation.data.clarity_score} />
            <ScoreCard label="Risk" value={resolutionMutation.data.risk_score} />
            <ScoreCard label="Confidence" value={resolutionMutation.data.confidence} />
            <ScoreCard label="Resolution" value={resolutionMutation.data.resolution_likelihood} />
          </div>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
              <p className="text-sm bg-muted p-3 rounded-lg border border-border">
                {resolutionMutation.data.recommendation}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Summary</p>
              <p className="text-sm bg-muted p-3 rounded-lg border border-border">
                {resolutionMutation.data.summary}
              </p>
            </div>
            {resolutionMutation.data.ambiguities?.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Ambiguities</p>
                <ul className="list-disc list-inside text-sm bg-muted p-3 rounded-lg border border-border space-y-1">
                  {resolutionMutation.data.ambiguities.map((a: string, i: number) => (
                    <li key={i} className="text-yellow-400">{a}</li>
                  ))}
                </ul>
              </div>
            )}
            {resolutionMutation.data.edge_cases?.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Edge Cases</p>
                <ul className="list-disc list-inside text-sm bg-muted p-3 rounded-lg border border-border space-y-1">
                  {resolutionMutation.data.edge_cases.map((e: string, i: number) => (
                    <li key={i} className="text-orange-400">{e}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Card>
      )}
      {tool === 'resolution' && resolutionMutation.error && (
        <ErrorBanner message={(resolutionMutation.error as Error).message} />
      )}

      {/* Market analysis results */}
      {tool === 'market' && marketMutation.data && (
        <Card className="p-5">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Analysis Result</h4>
          <div className="bg-muted p-4 rounded-lg border border-border whitespace-pre-wrap text-sm">
            {typeof marketMutation.data === 'string'
              ? marketMutation.data
              : marketMutation.data.analysis || marketMutation.data.result || JSON.stringify(marketMutation.data, null, 2)}
          </div>
        </Card>
      )}
      {tool === 'market' && marketMutation.error && (
        <ErrorBanner message={(marketMutation.error as Error).message} />
      )}

      {/* News sentiment results */}
      {tool === 'news' && newsMutation.data && (
        <Card className="p-5">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Sentiment Result</h4>
          <div className="bg-muted p-4 rounded-lg border border-border whitespace-pre-wrap text-sm">
            {typeof newsMutation.data === 'string'
              ? newsMutation.data
              : newsMutation.data.summary || newsMutation.data.analysis || JSON.stringify(newsMutation.data, null, 2)}
          </div>
        </Card>
      )}
      {tool === 'news' && newsMutation.error && (
        <ErrorBanner message={(newsMutation.error as Error).message} />
      )}
    </div>
  )
}

// ============================================================
// Judgments Section
// ============================================================

function JudgmentsSection() {
  const { data: history, isLoading } = useQuery({
    queryKey: ['ai-judgment-history'],
    queryFn: async () => {
      const { data } = await getJudgmentHistory({ limit: 50 })
      return data
    },
  })

  const { data: agreementStats } = useQuery({
    queryKey: ['ai-agreement-stats'],
    queryFn: async () => {
      const { data } = await getAgreementStats()
      return data
    },
  })

  if (isLoading) return <LoadingSpinner />

  return (
    <div className="space-y-4">
      {/* Agreement Stats */}
      {agreementStats && (
        <Card className="p-5">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            ML vs LLM Agreement
          </h3>
          <div className="grid grid-cols-4 gap-3">
            <MiniStat label="Total Judged" value={agreementStats.total_judged ?? 0} />
            <MiniStat label="Agreement Rate" value={`${((agreementStats.agreement_rate ?? 0) * 100).toFixed(1)}%`} />
            <MiniStat label="ML Overrides" value={agreementStats.ml_overrides ?? 0} />
            <MiniStat label="Avg Score" value={(agreementStats.avg_score ?? 0).toFixed(2)} />
          </div>
        </Card>
      )}

      {/* Judgment History */}
      <Card className="p-5">
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Target className="w-4 h-4 text-green-400" />
          Recent Judgments
        </h3>

        {!history || history.length === 0 ? (
          <EmptyState message="No opportunity judgments yet. AI will judge opportunities during scans when enabled." />
        ) : (
          <div className="space-y-2 max-h-[500px] overflow-y-auto">
            {(Array.isArray(history) ? history : []).map((j: any, i: number) => (
              <div
                key={j.opportunity_id || i}
                className="flex items-center justify-between bg-muted p-3 rounded-lg border border-border"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{j.opportunity_id}</p>
                  <p className="text-xs text-muted-foreground">
                    {j.recommendation} | {j.strategy_type ?? 'unknown'}
                  </p>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <ScoreBadge label="Score" value={j.overall_score} />
                  <ScoreBadge label="Profit" value={j.profit_viability} />
                  <ScoreBadge label="Safety" value={j.resolution_safety} />
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

// ============================================================
// System Section (Usage + Sessions + Skills)
// ============================================================

function SystemSection() {
  return (
    <div className="space-y-4">
      <UsageBlock />
      <SessionsBlock />
      <SkillsBlock />
    </div>
  )
}

// --- Usage Block ---

function UsageBlock() {
  const { data: usage, isLoading, error } = useQuery({
    queryKey: ['ai-usage'],
    queryFn: async () => {
      const { data } = await getAIUsage()
      return data
    },
    refetchInterval: 30000,
  })

  if (isLoading) return <LoadingSpinner />

  if (error) {
    return (
      <Card className="p-5">
        <div className="flex items-center gap-3 text-muted-foreground">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">Unable to fetch usage stats.</span>
        </div>
      </Card>
    )
  }

  if (!usage) {
    return (
      <Card className="p-5">
        <EmptyState message="No usage data available yet." />
      </Card>
    )
  }

  return (
    <Card className="p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-yellow-400" />
          Usage
        </h3>
        {usage.active_model && (
          <span className="text-xs font-mono text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
            {usage.active_model}
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <UsageStat
          icon={<Zap className="w-3.5 h-3.5 text-blue-400" />}
          label="Requests"
          value={usage.total_requests ?? 0}
        />
        <UsageStat
          icon={<FileText className="w-3.5 h-3.5 text-green-400" />}
          label="Input Tokens"
          value={formatNumber(usage.total_input_tokens ?? 0)}
        />
        <UsageStat
          icon={<FileText className="w-3.5 h-3.5 text-purple-400" />}
          label="Output Tokens"
          value={formatNumber(usage.total_output_tokens ?? 0)}
        />
        <UsageStat
          icon={<DollarSign className="w-3.5 h-3.5 text-yellow-400" />}
          label="Est. Cost"
          value={`$${(usage.estimated_cost ?? usage.total_cost_usd ?? 0).toFixed(4)}`}
        />
        <UsageStat
          icon={<Clock className="w-3.5 h-3.5 text-cyan-400" />}
          label="Avg Latency"
          value={`${(usage.avg_latency_ms ?? 0).toFixed(0)}ms`}
        />
        <UsageStat
          icon={<Activity className="w-3.5 h-3.5 text-orange-400" />}
          label="Total Tokens"
          value={formatNumber(usage.total_tokens ?? 0)}
        />
        <UsageStat
          icon={<CheckCircle className="w-3.5 h-3.5 text-green-400" />}
          label="Successful"
          value={usage.successful_requests ?? usage.total_requests ?? 0}
        />
        <UsageStat
          icon={<AlertCircle className="w-3.5 h-3.5 text-red-400" />}
          label="Failed"
          value={usage.failed_requests ?? usage.error_count ?? 0}
        />
      </div>

      {/* Spend Limit */}
      {usage.spend_limit_usd != null && (
        <div className="p-3 bg-muted rounded-lg border border-border mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Shield className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-xs font-medium">Monthly Spend Limit</span>
            </div>
            <span className="text-xs font-semibold">
              ${(usage.estimated_cost ?? usage.total_cost_usd ?? 0).toFixed(2)} / ${usage.spend_limit_usd.toFixed(2)}
            </span>
          </div>
          <div className="w-full bg-background rounded-full h-2 border border-border">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                ((usage.estimated_cost ?? usage.total_cost_usd ?? 0) / usage.spend_limit_usd) >= 0.9
                  ? 'bg-red-500'
                  : ((usage.estimated_cost ?? usage.total_cost_usd ?? 0) / usage.spend_limit_usd) >= 0.7
                    ? 'bg-yellow-500'
                    : 'bg-green-500'
              )}
              style={{ width: `${Math.min(100, ((usage.estimated_cost ?? usage.total_cost_usd ?? 0) / usage.spend_limit_usd) * 100)}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-1">
            <span className="text-[10px] text-muted-foreground">
              ${(usage.spend_remaining_usd ?? 0).toFixed(2)} remaining
            </span>
            <span className="text-[10px] text-muted-foreground">
              {usage.month_start ? `Since ${new Date(usage.month_start).toLocaleDateString()}` : ''}
            </span>
          </div>
        </div>
      )}

      {/* By Model */}
      {usage.by_model && typeof usage.by_model === 'object' && Object.keys(usage.by_model).length > 0 && (
        <div>
          <p className="text-xs text-muted-foreground mb-2">By Model</p>
          <div className="space-y-1.5">
            {Object.entries(usage.by_model).map(([model, stats]: [string, any]) => {
              const isActive = usage.active_model && model === usage.active_model
              return (
                <div
                  key={model}
                  className={cn(
                    'flex items-center justify-between p-2.5 rounded-lg border',
                    isActive ? 'bg-purple-500/5 border-purple-500/20' : 'bg-muted border-border'
                  )}
                >
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-medium font-mono">{model}</p>
                    {isActive && (
                      <span className="text-[9px] text-purple-400 bg-purple-500/10 px-1 py-0.5 rounded border border-purple-500/20">
                        active
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                    <span>{stats.requests ?? 0} req</span>
                    <span>{formatNumber(stats.tokens ?? 0)} tok</span>
                    <span>${(stats.cost ?? 0).toFixed(4)}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </Card>
  )
}

// --- Sessions Block ---

function SessionsBlock() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [sessionTypeFilter, setSessionTypeFilter] = useState('')
  const [expanded, setExpanded] = useState(true)

  const { data: sessions, isLoading } = useQuery({
    queryKey: ['ai-sessions', sessionTypeFilter],
    queryFn: async () => {
      const { data } = await getResearchSessions({
        session_type: sessionTypeFilter || undefined,
        limit: 50,
      })
      return data
    },
  })

  const { data: sessionDetail, isLoading: detailLoading } = useQuery({
    queryKey: ['ai-session-detail', selectedSessionId],
    queryFn: async () => {
      if (!selectedSessionId) return null
      const { data } = await getResearchSession(selectedSessionId)
      return data
    },
    enabled: !!selectedSessionId,
  })

  return (
    <Card className="p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full mb-1"
      >
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-indigo-400" />
          Research Sessions
        </h3>
        {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
      </button>

      {expanded && (
        <div className="mt-3">
          {isLoading ? (
            <LoadingSpinner />
          ) : (
            <>
              <div className="mb-3">
                <select
                  value={sessionTypeFilter}
                  onChange={(e) => setSessionTypeFilter(e.target.value)}
                  className="bg-muted border border-border rounded-lg px-3 py-1.5 text-xs"
                >
                  <option value="">All Types</option>
                  <option value="resolution_analysis">Resolution Analysis</option>
                  <option value="opportunity_judgment">Opportunity Judgment</option>
                  <option value="market_analysis">Market Analysis</option>
                  <option value="news_sentiment">News Sentiment</option>
                </select>
              </div>

              {!sessions || (Array.isArray(sessions) && sessions.length === 0) ? (
                <EmptyState message="No research sessions found." />
              ) : (
                <div className="space-y-1.5 max-h-72 overflow-y-auto">
                  {(Array.isArray(sessions) ? sessions : []).map((s: any) => {
                    const id = s.session_id || s.id
                    const isSelected = selectedSessionId === id
                    return (
                      <div key={id}>
                        <button
                          onClick={() => setSelectedSessionId(isSelected ? null : id)}
                          className={cn(
                            'w-full text-left p-2.5 rounded-lg border transition-colors',
                            isSelected
                              ? 'bg-indigo-500/10 border-indigo-500/30'
                              : 'bg-muted border-border hover:border-border'
                          )}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-medium truncate">{s.session_type || 'Unknown'}</p>
                              <p className="text-[10px] text-muted-foreground truncate">{id}</p>
                            </div>
                            <div className="flex items-center gap-2 ml-3">
                              <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                                {s.created_at ? new Date(s.created_at).toLocaleString() : ''}
                              </span>
                              {isSelected ? <ChevronDown className="w-3 h-3 text-muted-foreground" /> : <ChevronRight className="w-3 h-3 text-muted-foreground" />}
                            </div>
                          </div>
                        </button>
                        {isSelected && (
                          <div className="mt-1 ml-3 border-l-2 border-border pl-3">
                            {detailLoading ? (
                              <div className="py-3"><RefreshCw className="w-4 h-4 animate-spin text-muted-foreground" /></div>
                            ) : sessionDetail ? (
                              <pre className="text-[11px] text-muted-foreground whitespace-pre-wrap overflow-auto max-h-48 py-2">
                                {JSON.stringify(sessionDetail, null, 2)}
                              </pre>
                            ) : (
                              <p className="text-xs text-muted-foreground py-2">Session not found.</p>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </Card>
  )
}

// --- Skills Block ---

function SkillsBlock() {
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
  const [skillContext, setSkillContext] = useState('')
  const [expanded, setExpanded] = useState(true)

  const { data: skills, isLoading } = useQuery({
    queryKey: ['ai-skills'],
    queryFn: async () => {
      const { data } = await listSkills()
      return data
    },
  })

  const executeMutation = useMutation({
    mutationFn: async () => {
      if (!selectedSkill) throw new Error('No skill selected')
      let ctx = {}
      try {
        ctx = skillContext ? JSON.parse(skillContext) : {}
      } catch {
        throw new Error('Invalid JSON context')
      }
      const { data } = await executeSkill({
        skill_name: selectedSkill,
        context: ctx,
      })
      return data
    },
  })

  return (
    <Card className="p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full mb-1"
      >
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-400" />
          AI Skills
        </h3>
        {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
      </button>

      {expanded && (
        <div className="mt-3">
          {isLoading ? (
            <LoadingSpinner />
          ) : !skills || (Array.isArray(skills) && skills.length === 0) ? (
            <EmptyState message="No AI skills available." />
          ) : (
            <div className="space-y-1.5">
              {(Array.isArray(skills) ? skills : []).map((skill: any) => (
                <div key={skill.name}>
                  <button
                    onClick={() => setSelectedSkill(skill.name === selectedSkill ? null : skill.name)}
                    className={cn(
                      'w-full text-left p-2.5 rounded-lg border cursor-pointer transition-colors',
                      selectedSkill === skill.name
                        ? 'bg-emerald-500/10 border-emerald-500/30'
                        : 'bg-muted border-border hover:border-border'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs font-medium">{skill.name}</p>
                        <p className="text-[10px] text-muted-foreground">{skill.description || 'No description'}</p>
                      </div>
                      {selectedSkill === skill.name ? (
                        <ChevronDown className="w-3 h-3 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-muted-foreground" />
                      )}
                    </div>
                  </button>

                  {/* Execution form inline */}
                  {selectedSkill === skill.name && (
                    <div className="mt-2 ml-3 pl-3 border-l-2 border-border space-y-2">
                      <div>
                        <label className="block text-[10px] text-muted-foreground mb-1">Context (JSON)</label>
                        <textarea
                          value={skillContext}
                          onChange={(e) => setSkillContext(e.target.value)}
                          placeholder='{"market_id": "...", "question": "..."}'
                          rows={3}
                          className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-emerald-500 resize-none"
                        />
                      </div>
                      <Button
                        onClick={() => executeMutation.mutate()}
                        disabled={executeMutation.isPending}
                        size="sm"
                        className={cn(
                          'h-auto gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                          executeMutation.isPending
                            ? 'bg-muted text-muted-foreground cursor-not-allowed'
                            : 'bg-emerald-500 hover:bg-emerald-600 text-white'
                        )}
                      >
                        {executeMutation.isPending ? (
                          <RefreshCw className="w-3 h-3 animate-spin" />
                        ) : (
                          <Zap className="w-3 h-3" />
                        )}
                        Execute
                      </Button>

                      {executeMutation.data && (
                        <pre className="text-[11px] text-muted-foreground bg-muted p-3 rounded-lg border border-border whitespace-pre-wrap overflow-auto max-h-48">
                          {typeof executeMutation.data === 'string'
                            ? executeMutation.data
                            : JSON.stringify(executeMutation.data, null, 2)}
                        </pre>
                      )}

                      {executeMutation.error && (
                        <ErrorBanner message={(executeMutation.error as Error).message} />
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

// ============================================================
// Shared Components
// ============================================================

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-8">
      <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-6">
      <AlertCircle className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
      <p className="text-xs text-muted-foreground">{message}</p>
    </div>
  )
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
      <div className="flex items-center gap-2">
        <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
        <p className="text-sm text-red-400">{message}</p>
      </div>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-muted rounded-lg p-3 border border-border">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold mt-0.5">{value}</p>
    </div>
  )
}

function ScoreCard({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'text-green-400' : value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="bg-muted rounded-lg p-3 border border-border text-center">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={cn('text-lg font-bold mt-0.5', color)}>
        {typeof value === 'number' ? value.toFixed(2) : value ?? 'N/A'}
      </p>
    </div>
  )
}

function ScoreBadge({ label, value }: { label: string; value: number }) {
  const color =
    value >= 0.7
      ? 'bg-green-500/10 text-green-400 border-green-500/20'
      : value >= 0.4
        ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
        : 'bg-red-500/10 text-red-400 border-red-500/20'
  return (
    <Badge variant="outline" className={cn('rounded border-transparent text-[10px]', color)}>
      {label}: {typeof value === 'number' ? value.toFixed(2) : 'N/A'}
    </Badge>
  )
}

function UsageStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="bg-muted rounded-lg p-3 border border-border">
      <div className="flex items-center gap-1.5 mb-0.5">
        {icon}
        <p className="text-[10px] text-muted-foreground">{label}</p>
      </div>
      <p className="text-sm font-semibold">{value}</p>
    </div>
  )
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`
  return num.toString()
}
