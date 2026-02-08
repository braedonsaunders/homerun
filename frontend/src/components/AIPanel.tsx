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
  Cpu,
  BookOpen,
  BarChart3,
  ChevronDown,
  ChevronRight,
  Send,
  Newspaper,
  Layers,
  Clock,
  Shield,
  Target,
  TrendingUp,
  DollarSign,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
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

type AISection = 'status' | 'resolution' | 'judgments' | 'market' | 'news' | 'skills' | 'sessions' | 'usage'

export default function AIPanel() {
  const [activeSection, setActiveSection] = useState<AISection>('status')

  // Listen for navigation events from command bar
  useEffect(() => {
    const handler = (e: Event) => {
      const section = (e as CustomEvent).detail as AISection
      if (section) {
        setActiveSection(section)
      }
    }
    window.addEventListener('navigate-ai-section', handler)
    return () => window.removeEventListener('navigate-ai-section', handler)
  }, [])

  return (
    <div>
      {/* Section Navigation */}
      <Tabs value={activeSection} onValueChange={(v) => setActiveSection(v as AISection)}>
        <TabsList className="flex flex-wrap h-auto justify-start gap-2 mb-6 bg-transparent p-0">
          <TabsTrigger
            value="status"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <Cpu className="w-4 h-4" />
            Status
          </TabsTrigger>
          <TabsTrigger
            value="resolution"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <Shield className="w-4 h-4" />
            Resolution Analysis
          </TabsTrigger>
          <TabsTrigger
            value="judgments"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <Target className="w-4 h-4" />
            Judgments
          </TabsTrigger>
          <TabsTrigger
            value="market"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <TrendingUp className="w-4 h-4" />
            Market Analysis
          </TabsTrigger>
          <TabsTrigger
            value="news"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <Newspaper className="w-4 h-4" />
            News Sentiment
          </TabsTrigger>
          <TabsTrigger
            value="skills"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <Layers className="w-4 h-4" />
            Skills
          </TabsTrigger>
          <TabsTrigger
            value="sessions"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <BookOpen className="w-4 h-4" />
            Sessions
          </TabsTrigger>
          <TabsTrigger
            value="usage"
            className="gap-2 rounded-lg bg-muted text-muted-foreground border border-border hover:text-foreground data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 data-[state=active]:border-purple-500/30 data-[state=active]:shadow-none"
          >
            <BarChart3 className="w-4 h-4" />
            Usage
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Section Content */}
      {activeSection === 'status' && <AIStatusSection />}
      {activeSection === 'resolution' && <ResolutionSection />}
      {activeSection === 'judgments' && <JudgmentsSection />}
      {activeSection === 'market' && <MarketAnalysisSection />}
      {activeSection === 'news' && <NewsSentimentSection />}
      {activeSection === 'skills' && <SkillsSection />}
      {activeSection === 'sessions' && <SessionsSection />}
      {activeSection === 'usage' && <UsageSection />}
    </div>
  )
}

// === AI Status ===

function AIStatusSection() {
  const { data: status, isLoading, error } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const { data } = await getAIStatus()
      return data
    },
    refetchInterval: 30000,
  })

  if (isLoading) return <LoadingSpinner />

  if (error) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-3 text-muted-foreground">
          <AlertCircle className="w-5 h-5" />
          <span>Unable to fetch AI status. The AI module may not be configured.</span>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className={cn(
            'w-3 h-3 rounded-full',
            status?.enabled ? 'bg-green-500' : 'bg-gray-600'
          )} />
          <h3 className="text-lg font-semibold">
            AI Intelligence {status?.enabled ? 'Active' : 'Inactive'}
          </h3>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Status</p>
            <p className={cn('text-sm font-medium', status?.enabled ? 'text-green-400' : 'text-muted-foreground')}>
              {status?.enabled ? 'Enabled' : 'No Providers Configured'}
            </p>
          </div>
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Providers</p>
            <p className="text-sm font-medium">
              {status?.providers_configured?.length > 0
                ? status.providers_configured.join(', ')
                : 'None'}
            </p>
          </div>
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Skills Available</p>
            <p className="text-sm font-medium">{status?.skills_available ?? 0}</p>
          </div>
        </div>

        {!status?.enabled && (
          <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <p className="text-sm text-yellow-400">
              Configure an LLM provider (OpenAI or Anthropic) in Settings to enable AI features.
            </p>
          </div>
        )}
      </Card>

      {status?.usage && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Quick Usage Summary</h4>
          <div className="grid grid-cols-4 gap-4">
            <MiniStat label="Total Requests" value={status.usage.total_requests ?? 0} />
            <MiniStat label="Total Tokens" value={formatNumber(status.usage.total_tokens ?? 0)} />
            <MiniStat label="Est. Cost" value={`$${(status.usage.estimated_cost ?? status.usage.total_cost_usd ?? 0).toFixed(4)}`} />
            <MiniStat label="Avg Latency" value={`${(status.usage.avg_latency_ms ?? 0).toFixed(0)}ms`} />
          </div>
          {status.usage.spend_limit_usd != null && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span>Monthly Spend</span>
                <span>${(status.usage.estimated_cost ?? status.usage.total_cost_usd ?? 0).toFixed(2)} / ${status.usage.spend_limit_usd.toFixed(2)}</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2 border border-border">
                <div
                  className={cn(
                    'h-full rounded-full transition-all',
                    ((status.usage.estimated_cost ?? status.usage.total_cost_usd ?? 0) / status.usage.spend_limit_usd) >= 0.9
                      ? 'bg-red-500'
                      : ((status.usage.estimated_cost ?? status.usage.total_cost_usd ?? 0) / status.usage.spend_limit_usd) >= 0.7
                        ? 'bg-yellow-500'
                        : 'bg-green-500'
                  )}
                  style={{ width: `${Math.min(100, ((status.usage.estimated_cost ?? status.usage.total_cost_usd ?? 0) / status.usage.spend_limit_usd) * 100)}%` }}
                />
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}

// === Resolution Analysis ===

function ResolutionSection() {
  const [marketId, setMarketId] = useState('')
  const [question, setQuestion] = useState('')
  const [description, setDescription] = useState('')
  const [resolutionSource, setResolutionSource] = useState('')
  const [endDate, setEndDate] = useState('')
  const [outcomes, setOutcomes] = useState('')
  const [marketSearch, setMarketSearch] = useState('')
  const [searchResults, setSearchResults] = useState<MarketSearchResult[]>([])
  const [showSearchResults, setShowSearchResults] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>()

  // Listen for market-selected events from command bar
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as MarketSearchResult
      if (detail) {
        setMarketId(detail.market_id)
        setQuestion(detail.question)
        setMarketSearch(detail.question)
        setShowSearchResults(false)
      }
    }
    window.addEventListener('market-selected', handler)
    return () => window.removeEventListener('market-selected', handler)
  }, [])

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
    if (marketSearch.length >= 2) {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => doSearch(marketSearch), 300)
    } else {
      setSearchResults([])
    }
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [marketSearch, doSearch])

  const selectMarket = (m: MarketSearchResult) => {
    setMarketId(m.market_id)
    setQuestion(m.question)
    setMarketSearch(m.question)
    setShowSearchResults(false)
  }

  const analyzeMutation = useMutation({
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

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-purple-400" />
          Resolution Criteria Analysis
        </h3>
        <p className="text-sm text-muted-foreground mb-4">
          Search for a market below or type a question to analyze resolution criteria.
        </p>

        <div className="space-y-3">
          {/* Smart Market Search */}
          <div ref={searchRef} className="relative">
            <label className="block text-xs text-muted-foreground mb-1">
              <Search className="w-3 h-3 inline mr-1" />
              Search Markets (type to find - no manual IDs needed)
            </label>
            <Input
              type="text"
              value={marketSearch}
              onChange={(e) => {
                setMarketSearch(e.target.value)
                setShowSearchResults(true)
              }}
              onFocus={() => searchResults.length > 0 && setShowSearchResults(true)}
              placeholder="e.g., Bitcoin, Fed rate, Trump, Super Bowl..."
              className="bg-muted rounded-lg focus-visible:ring-purple-500"
            />
            {showSearchResults && searchResults.length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-[#0f0f0f] border border-border rounded-xl shadow-2xl max-h-64 overflow-y-auto">
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

          {marketId && (
            <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-2 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-purple-400 flex-shrink-0" />
              <span className="text-xs text-purple-400 truncate">
                Market selected: {marketId.slice(0, 20)}...
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Market ID</label>
              <Input
                type="text"
                value={marketId}
                onChange={(e) => setMarketId(e.target.value)}
                placeholder="Auto-filled from search above"
                className="bg-muted rounded-lg focus-visible:ring-purple-500"
              />
            </div>
            <div>
              <label className="block text-xs text-muted-foreground mb-1">End Date</label>
              <Input
                type="text"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                placeholder="e.g., 2025-12-31"
                className="bg-muted rounded-lg focus-visible:ring-purple-500"
              />
            </div>
          </div>
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
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Market description and resolution criteria..."
              rows={3}
              className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Resolution Source</label>
              <Input
                type="text"
                value={resolutionSource}
                onChange={(e) => setResolutionSource(e.target.value)}
                placeholder="e.g., CoinGecko price feed"
                className="bg-muted rounded-lg focus-visible:ring-purple-500"
              />
            </div>
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Outcomes (comma-separated)</label>
              <Input
                type="text"
                value={outcomes}
                onChange={(e) => setOutcomes(e.target.value)}
                placeholder="e.g., Yes, No"
                className="bg-muted rounded-lg focus-visible:ring-purple-500"
              />
            </div>
          </div>

          <Button
            onClick={() => analyzeMutation.mutate()}
            disabled={!marketId || !question || analyzeMutation.isPending}
            className={cn(
              'h-auto gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              !marketId || !question || analyzeMutation.isPending
                ? 'bg-gray-800 text-muted-foreground cursor-not-allowed'
                : 'bg-purple-500 hover:bg-purple-600 text-foreground'
            )}
          >
            {analyzeMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            Analyze Resolution
          </Button>
        </div>
      </Card>

      {/* Analysis Result */}
      {analyzeMutation.data && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Analysis Result</h4>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <ScoreCard label="Clarity" value={analyzeMutation.data.clarity_score} />
            <ScoreCard label="Risk" value={analyzeMutation.data.risk_score} />
            <ScoreCard label="Confidence" value={analyzeMutation.data.confidence} />
            <ScoreCard label="Resolution Likelihood" value={analyzeMutation.data.resolution_likelihood} />
          </div>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
              <p className="text-sm bg-muted p-3 rounded-lg border border-border">
                {analyzeMutation.data.recommendation}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Summary</p>
              <p className="text-sm bg-muted p-3 rounded-lg border border-border">
                {analyzeMutation.data.summary}
              </p>
            </div>
            {analyzeMutation.data.ambiguities?.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Ambiguities</p>
                <ul className="list-disc list-inside text-sm bg-muted p-3 rounded-lg border border-border space-y-1">
                  {analyzeMutation.data.ambiguities.map((a: string, i: number) => (
                    <li key={i} className="text-yellow-400">{a}</li>
                  ))}
                </ul>
              </div>
            )}
            {analyzeMutation.data.edge_cases?.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Edge Cases</p>
                <ul className="list-disc list-inside text-sm bg-muted p-3 rounded-lg border border-border space-y-1">
                  {analyzeMutation.data.edge_cases.map((e: string, i: number) => (
                    <li key={i} className="text-orange-400">{e}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Card>
      )}

      {analyzeMutation.error && (
        <ErrorBanner message={(analyzeMutation.error as Error).message} />
      )}
    </div>
  )
}

// === Opportunity Judgments ===

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
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            ML vs LLM Agreement
          </h3>
          <div className="grid grid-cols-4 gap-4">
            <MiniStat label="Total Judged" value={agreementStats.total_judged ?? 0} />
            <MiniStat label="Agreement Rate" value={`${((agreementStats.agreement_rate ?? 0) * 100).toFixed(1)}%`} />
            <MiniStat label="ML Overrides" value={agreementStats.ml_overrides ?? 0} />
            <MiniStat label="Avg Score" value={(agreementStats.avg_score ?? 0).toFixed(2)} />
          </div>
        </Card>
      )}

      {/* Judgment History */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-green-400" />
          Recent Judgments
        </h3>

        {!history || history.length === 0 ? (
          <EmptyState message="No opportunity judgments yet. AI will judge opportunities during scans when enabled." />
        ) : (
          <div className="space-y-2">
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
                <div className="flex items-center gap-3 ml-4">
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

// === Market Analysis ===

function MarketAnalysisSection() {
  const [query, setQuery] = useState('')
  const [marketId, setMarketId] = useState('')
  const [marketQuestion, setMarketQuestion] = useState('')

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await analyzeMarket({
        query,
        market_id: marketId || undefined,
        market_question: marketQuestion || undefined,
      })
      return data
    },
  })

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-400" />
          AI Market Analysis
        </h3>
        <p className="text-sm text-muted-foreground mb-4">
          Ask the AI to analyze any market or topic with free-form queries.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Query *</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., What are the chances of a Fed rate cut in March? Analyze recent economic indicators..."
              rows={3}
              className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 resize-none"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Market ID (optional)</label>
              <Input
                type="text"
                value={marketId}
                onChange={(e) => setMarketId(e.target.value)}
                placeholder="Link to specific market"
                className="bg-muted rounded-lg focus-visible:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Market Question (optional)</label>
              <Input
                type="text"
                value={marketQuestion}
                onChange={(e) => setMarketQuestion(e.target.value)}
                placeholder="Market question for context"
                className="bg-muted rounded-lg focus-visible:ring-blue-500"
              />
            </div>
          </div>

          <Button
            onClick={() => analyzeMutation.mutate()}
            disabled={!query || analyzeMutation.isPending}
            className={cn(
              'h-auto gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              !query || analyzeMutation.isPending
                ? 'bg-gray-800 text-muted-foreground cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600 text-foreground'
            )}
          >
            {analyzeMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            Analyze
          </Button>
        </div>
      </Card>

      {analyzeMutation.data && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Analysis Result</h4>
          <div className="bg-muted p-4 rounded-lg border border-border whitespace-pre-wrap text-sm">
            {typeof analyzeMutation.data === 'string'
              ? analyzeMutation.data
              : analyzeMutation.data.analysis || analyzeMutation.data.result || JSON.stringify(analyzeMutation.data, null, 2)}
          </div>
        </Card>
      )}

      {analyzeMutation.error && (
        <ErrorBanner message={(analyzeMutation.error as Error).message} />
      )}
    </div>
  )
}

// === News Sentiment ===

function NewsSentimentSection() {
  const [query, setQuery] = useState('')
  const [marketContext, setMarketContext] = useState('')
  const [maxArticles, setMaxArticles] = useState(5)

  const sentimentMutation = useMutation({
    mutationFn: async () => {
      const { data } = await analyzeNewsSentiment({
        query,
        market_context: marketContext,
        max_articles: maxArticles,
      })
      return data
    },
  })

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-orange-400" />
          News Sentiment Analysis
        </h3>
        <p className="text-sm text-muted-foreground mb-4">
          Search recent news and analyze sentiment for market-relevant topics.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Search Query *</label>
            <Input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Federal Reserve interest rate decision"
              className="bg-muted rounded-lg focus-visible:ring-orange-500"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Market Context</label>
              <Input
                type="text"
                value={marketContext}
                onChange={(e) => setMarketContext(e.target.value)}
                placeholder="e.g., Will the Fed cut rates in March 2025?"
                className="bg-muted rounded-lg focus-visible:ring-orange-500"
              />
            </div>
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Max Articles</label>
              <Input
                type="number"
                value={maxArticles}
                onChange={(e) => setMaxArticles(parseInt(e.target.value) || 5)}
                min={1}
                max={20}
                className="bg-muted rounded-lg focus-visible:ring-orange-500"
              />
            </div>
          </div>

          <Button
            onClick={() => sentimentMutation.mutate()}
            disabled={!query || sentimentMutation.isPending}
            className={cn(
              'h-auto gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              !query || sentimentMutation.isPending
                ? 'bg-gray-800 text-muted-foreground cursor-not-allowed'
                : 'bg-orange-500 hover:bg-orange-600 text-foreground'
            )}
          >
            {sentimentMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            Search & Analyze
          </Button>
        </div>
      </Card>

      {sentimentMutation.data && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Sentiment Result</h4>
          <div className="bg-muted p-4 rounded-lg border border-border whitespace-pre-wrap text-sm">
            {typeof sentimentMutation.data === 'string'
              ? sentimentMutation.data
              : sentimentMutation.data.summary || sentimentMutation.data.analysis || JSON.stringify(sentimentMutation.data, null, 2)}
          </div>
        </Card>
      )}

      {sentimentMutation.error && (
        <ErrorBanner message={(sentimentMutation.error as Error).message} />
      )}
    </div>
  )
}

// === Skills ===

function SkillsSection() {
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
  const [skillContext, setSkillContext] = useState('')

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

  if (isLoading) return <LoadingSpinner />

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-emerald-400" />
          AI Skills
        </h3>

        {!skills || (Array.isArray(skills) && skills.length === 0) ? (
          <EmptyState message="No AI skills available. Skills are loaded when the AI layer initializes." />
        ) : (
          <div className="space-y-2">
            {(Array.isArray(skills) ? skills : []).map((skill: any) => (
              <div
                key={skill.name}
                onClick={() => setSelectedSkill(skill.name === selectedSkill ? null : skill.name)}
                className={cn(
                  'p-3 rounded-lg border cursor-pointer transition-colors',
                  selectedSkill === skill.name
                    ? 'bg-emerald-500/10 border-emerald-500/30'
                    : 'bg-muted border-border hover:border-border'
                )}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{skill.name}</p>
                    <p className="text-xs text-muted-foreground">{skill.description || 'No description'}</p>
                  </div>
                  {selectedSkill === skill.name ? (
                    <ChevronDown className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {selectedSkill && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">
            Execute: {selectedSkill}
          </h4>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-muted-foreground mb-1">Context (JSON)</label>
              <textarea
                value={skillContext}
                onChange={(e) => setSkillContext(e.target.value)}
                placeholder='{"market_id": "...", "question": "..."}'
                rows={4}
                className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-emerald-500 resize-none"
              />
            </div>
            <Button
              onClick={() => executeMutation.mutate()}
              disabled={executeMutation.isPending}
              className={cn(
                'h-auto gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                executeMutation.isPending
                  ? 'bg-gray-800 text-muted-foreground cursor-not-allowed'
                  : 'bg-emerald-500 hover:bg-emerald-600 text-foreground'
              )}
            >
              {executeMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              Execute Skill
            </Button>
          </div>

          {executeMutation.data && (
            <div className="mt-4 bg-muted p-4 rounded-lg border border-border">
              <pre className="text-sm whitespace-pre-wrap overflow-auto max-h-96">
                {typeof executeMutation.data === 'string'
                  ? executeMutation.data
                  : JSON.stringify(executeMutation.data, null, 2)}
              </pre>
            </div>
          )}

          {executeMutation.error && (
            <div className="mt-4">
              <ErrorBanner message={(executeMutation.error as Error).message} />
            </div>
          )}
        </Card>
      )}
    </div>
  )
}

// === Research Sessions ===

function SessionsSection() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [sessionTypeFilter, setSessionTypeFilter] = useState('')

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

  if (isLoading) return <LoadingSpinner />

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-indigo-400" />
          Research Sessions
        </h3>

        <div className="mb-4">
          <label className="block text-xs text-muted-foreground mb-1">Filter by Type</label>
          <select
            value={sessionTypeFilter}
            onChange={(e) => setSessionTypeFilter(e.target.value)}
            className="bg-muted border border-border rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All Types</option>
            <option value="resolution_analysis">Resolution Analysis</option>
            <option value="opportunity_judgment">Opportunity Judgment</option>
            <option value="market_analysis">Market Analysis</option>
            <option value="news_sentiment">News Sentiment</option>
          </select>
        </div>

        {!sessions || (Array.isArray(sessions) && sessions.length === 0) ? (
          <EmptyState message="No research sessions found. Sessions are created when AI features are used." />
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {(Array.isArray(sessions) ? sessions : []).map((s: any) => (
              <div
                key={s.session_id || s.id}
                onClick={() => setSelectedSessionId(s.session_id || s.id)}
                className={cn(
                  'p-3 rounded-lg border cursor-pointer transition-colors',
                  selectedSessionId === (s.session_id || s.id)
                    ? 'bg-indigo-500/10 border-indigo-500/30'
                    : 'bg-muted border-border hover:border-border'
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{s.session_type || 'Unknown'}</p>
                    <p className="text-xs text-muted-foreground">
                      {s.session_id || s.id}
                    </p>
                  </div>
                  <div className="text-xs text-muted-foreground ml-4">
                    <Clock className="w-3 h-3 inline mr-1" />
                    {s.created_at ? new Date(s.created_at).toLocaleString() : 'Unknown'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {selectedSessionId && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">
            Session Details
          </h4>
          {detailLoading ? (
            <LoadingSpinner />
          ) : sessionDetail ? (
            <div className="bg-muted p-4 rounded-lg border border-border">
              <pre className="text-sm whitespace-pre-wrap overflow-auto max-h-96">
                {JSON.stringify(sessionDetail, null, 2)}
              </pre>
            </div>
          ) : (
            <EmptyState message="Session not found or has been deleted." />
          )}
        </Card>
      )}
    </div>
  )
}

// === Usage Stats ===

function UsageSection() {
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
      <Card className="p-6">
        <div className="flex items-center gap-3 text-muted-foreground">
          <AlertCircle className="w-5 h-5" />
          <span>Unable to fetch usage stats. AI may not be configured.</span>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-yellow-400" />
          LLM Usage Statistics
        </h3>

        {!usage ? (
          <EmptyState message="No usage data available yet." />
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <UsageStat
                icon={<Zap className="w-4 h-4 text-blue-400" />}
                label="Total Requests"
                value={usage.total_requests ?? 0}
              />
              <UsageStat
                icon={<FileText className="w-4 h-4 text-green-400" />}
                label="Input Tokens"
                value={formatNumber(usage.total_input_tokens ?? 0)}
              />
              <UsageStat
                icon={<FileText className="w-4 h-4 text-purple-400" />}
                label="Output Tokens"
                value={formatNumber(usage.total_output_tokens ?? 0)}
              />
              <UsageStat
                icon={<DollarSign className="w-4 h-4 text-yellow-400" />}
                label="Estimated Cost"
                value={`$${(usage.estimated_cost ?? usage.total_cost_usd ?? 0).toFixed(4)}`}
              />
              <UsageStat
                icon={<Clock className="w-4 h-4 text-cyan-400" />}
                label="Avg Latency"
                value={`${(usage.avg_latency_ms ?? 0).toFixed(0)}ms`}
              />
              <UsageStat
                icon={<Activity className="w-4 h-4 text-orange-400" />}
                label="Total Tokens"
                value={formatNumber(usage.total_tokens ?? 0)}
              />
              <UsageStat
                icon={<CheckCircle className="w-4 h-4 text-green-400" />}
                label="Successful"
                value={usage.successful_requests ?? usage.total_requests ?? 0}
              />
              <UsageStat
                icon={<AlertCircle className="w-4 h-4 text-red-400" />}
                label="Failed"
                value={usage.failed_requests ?? usage.error_count ?? 0}
              />
            </div>

            {usage.spend_limit_usd != null && (
              <div className="mt-4 p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-medium">Monthly Spend Limit</span>
                  </div>
                  <span className="text-sm font-semibold">
                    ${(usage.estimated_cost ?? usage.total_cost_usd ?? 0).toFixed(2)} / ${usage.spend_limit_usd.toFixed(2)}
                  </span>
                </div>
                <div className="w-full bg-background rounded-full h-3 border border-border">
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
                  <span className="text-xs text-muted-foreground">
                    ${(usage.spend_remaining_usd ?? 0).toFixed(2)} remaining
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {usage.month_start ? `Since ${new Date(usage.month_start).toLocaleDateString()}` : ''}
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </Card>

      {usage?.by_model && typeof usage.by_model === 'object' && (
        <Card className="p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3">Usage by Model</h4>
          <div className="space-y-2">
            {Object.entries(usage.by_model).map(([model, stats]: [string, any]) => (
              <div key={model} className="flex items-center justify-between bg-muted p-3 rounded-lg border border-border">
                <p className="text-sm font-medium font-mono">{model}</p>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>{stats.requests ?? 0} requests</span>
                  <span>{formatNumber(stats.tokens ?? 0)} tokens</span>
                  <span>${(stats.cost ?? 0).toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

// === Shared Components ===

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-8">
      <AlertCircle className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  )
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
      <div className="flex items-center gap-2">
        <AlertCircle className="w-4 h-4 text-red-400" />
        <p className="text-sm text-red-400">{message}</p>
      </div>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-muted rounded-lg p-3 border border-border">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold mt-0.5">{value}</p>
    </div>
  )
}

function ScoreCard({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'text-green-400' : value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="bg-muted rounded-lg p-3 border border-border text-center">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={cn('text-lg font-bold mt-1', color)}>
        {typeof value === 'number' ? value.toFixed(2) : value ?? 'N/A'}
      </p>
    </div>
  )
}

function ScoreBadge({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'bg-green-500/10 text-green-400 border-green-500/20' : value >= 0.4 ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'
  return (
    <Badge variant="outline" className={cn('rounded border-transparent', color)}>
      {label}: {typeof value === 'number' ? value.toFixed(2) : 'N/A'}
    </Badge>
  )
}

function UsageStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="bg-muted rounded-lg p-4 border border-border">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  )
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`
  return num.toString()
}
