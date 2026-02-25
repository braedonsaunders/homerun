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
  Sparkles,
  Brain,
  Cpu,
  Bot,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { ScrollArea } from './ui/scroll-area'
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
type MetricTone = 'good' | 'bad' | 'warn' | 'neutral' | 'info'

export interface AICopilotLaunchOptions {
  contextType?: string
  contextId?: string
  label?: string
  prompt?: string
  autoSend?: boolean
}

interface AIPanelProps {
  onOpenCopilot?: (options?: AICopilotLaunchOptions) => void
}

const TONE_CLASSES: Record<MetricTone, string> = {
  good: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200',
  bad: 'border-red-300 bg-red-100 text-red-900 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200',
  warn: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200',
  neutral: 'border-border/70 bg-background/35 text-foreground',
  info: 'border-purple-300 bg-purple-100 text-purple-900 dark:border-purple-500/30 dark:bg-purple-500/10 dark:text-purple-200',
}

export default function AIPanel({ onOpenCopilot }: AIPanelProps) {
  const [activeTab, setActiveTab] = useState<AITab>('analyze')
  const [analysisTool, setAnalysisTool] = useState<AnalysisTool>('resolution')
  const { data: aiStatus } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const { data } = await getAIStatus()
      return data
    },
    refetchInterval: 30000,
  })
  const providersLabel = aiStatus?.providers_configured?.length
    ? aiStatus.providers_configured.join(', ')
    : 'No providers'
  const usageCost = aiStatus?.usage?.estimated_cost ?? aiStatus?.usage?.total_cost_usd ?? 0

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

  return (
    <div className="h-full min-h-0 flex flex-col gap-3">
      <div className="shrink-0 px-1 overflow-x-auto">
        <div className="flex min-w-max items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveTab('analyze')}
            className={cn(
              'h-8 gap-1.5 text-xs',
              activeTab === 'analyze'
                ? 'bg-violet-500/20 text-violet-300 border-violet-500/30 hover:bg-violet-500/30 hover:text-violet-300'
                : 'bg-card text-muted-foreground hover:text-foreground border-border'
            )}
          >
            <Brain className="h-3.5 w-3.5" />
            Analyze
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveTab('judgments')}
            className={cn(
              'h-8 gap-1.5 text-xs',
              activeTab === 'judgments'
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-300'
                : 'bg-card text-muted-foreground hover:text-foreground border-border'
            )}
          >
            <Target className="h-3.5 w-3.5" />
            Judgments
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveTab('system')}
            className={cn(
              'h-8 gap-1.5 text-xs',
              activeTab === 'system'
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/30 hover:text-emerald-300'
                : 'bg-card text-muted-foreground hover:text-foreground border-border'
            )}
          >
            <Cpu className="h-3.5 w-3.5" />
            System
          </Button>
        </div>
      </div>

      <div className="shrink-0 px-1">
        <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-border/60 bg-card/35 px-2.5 py-1.5 text-[11px]">
          <Badge
            variant="outline"
            className={cn(
              'h-5 border px-1.5 text-[10px]',
              aiStatus?.enabled
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300'
                : 'border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-300'
            )}
          >
            {aiStatus?.enabled ? 'AI Connected' : 'AI Disabled'}
          </Badge>
          <Badge variant="outline" className="h-5 max-w-[240px] truncate border-border/60 bg-background/60 px-1.5 text-[10px] text-muted-foreground">
            {providersLabel}
          </Badge>
          <Badge variant="outline" className="h-5 border-border/60 bg-background/60 px-1.5 text-[10px] text-muted-foreground">
            {aiStatus?.skills_available ?? 0} skills
          </Badge>
          <Badge variant="outline" className="h-5 border-border/60 bg-background/60 px-1.5 text-[10px] text-muted-foreground font-data">
            ${usageCost.toFixed(2)}
          </Badge>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto pr-1">
        {activeTab === 'analyze' && <AnalyzeSection tool={analysisTool} onToolChange={setAnalysisTool} onOpenCopilot={onOpenCopilot} />}
        {activeTab === 'judgments' && <JudgmentsSection />}
        {activeTab === 'system' && <SystemSection />}
      </div>
    </div>
  )
}

// ============================================================
// Analyze Section
// ============================================================

function AnalyzeSection({
  tool,
  onToolChange,
  onOpenCopilot,
}: {
  tool: AnalysisTool
  onToolChange: (t: AnalysisTool) => void
  onOpenCopilot?: (options?: AICopilotLaunchOptions) => void
}) {
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

  const [marketQuery, setMarketQuery] = useState('')
  const [marketAnalysisId, setMarketAnalysisId] = useState('')
  const [marketAnalysisQuestion, setMarketAnalysisQuestion] = useState('')
  const [showMarketAdvanced, setShowMarketAdvanced] = useState(false)

  const [newsQuery, setNewsQuery] = useState('')
  const [newsContext, setNewsContext] = useState('')
  const [maxArticles, setMaxArticles] = useState(5)
  const [showNewsAdvanced, setShowNewsAdvanced] = useState(false)
  const [customPrompt, setCustomPrompt] = useState('')

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

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSearchResults(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

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

  const activeToolLabel = tool === 'resolution' ? 'Resolution' : tool === 'market' ? 'Market' : 'News'
  const activeToolDescription =
    tool === 'resolution'
      ? 'Evaluate resolution criteria, ambiguities, and edge-case handling.'
      : tool === 'market'
        ? 'Run AI-assisted market context analysis with optional market linking.'
        : 'Analyze current news sentiment and infer directional market pressure.'
  const promptTemplates: Array<{ title: string; body: string }> = tool === 'resolution'
    ? [
        {
          title: 'Resolution Ambiguity Sweep',
          body: `Audit "${question || 'this market'}" for ambiguous resolution edge-cases and list monitoring triggers before settlement.`,
        },
        {
          title: 'Execution Guardrails',
          body: 'Create a strict pre-trade and post-trade checklist for this resolution setup including risk-off conditions.',
        },
        {
          title: 'Contrarian Failure Modes',
          body: 'Stress-test the obvious interpretation of this market and map plausible failure modes that can invalidate the trade thesis.',
        },
      ]
    : tool === 'market'
      ? [
          {
            title: 'Probability Tree',
            body: `Build bull/base/bear probability branches for "${marketQuery || 'the target market thesis'}" with explicit catalyst assumptions.`,
          },
          {
            title: 'Market-Microstructure Read',
            body: 'Explain what current pricing likely implies about informed flow, crowd positioning, and delayed repricing opportunities.',
          },
          {
            title: 'Actionable Trade Plan',
            body: 'Convert this market analysis into a concrete execution plan: entries, invalidation levels, and position sizing guidance.',
          },
        ]
      : [
          {
            title: 'Narrative Delta',
            body: `Summarize the dominant news narrative for "${newsQuery || 'this topic'}" and estimate probability impact versus market pricing.`,
          },
          {
            title: 'Signal Reliability Filter',
            body: 'Separate high-signal news from noise and rank sources by reliability, timeliness, and trade relevance.',
          },
          {
            title: 'Sentiment-To-Execution',
            body: 'Turn sentiment results into a trade decision rubric with clear thresholds for buy, wait, hedge, or skip.',
          },
        ]

  const launchPrompt = (prompt: string, autoSend = true) => {
    if (!onOpenCopilot) return
    const context = tool === 'resolution' && marketId
      ? {
          contextType: 'market',
          contextId: marketId,
          label: question || marketId,
        }
      : {
          contextType: 'general',
          contextId: 'ai-workspace',
          label: 'AI Workspace',
        }
    onOpenCopilot({
      ...context,
      prompt,
      autoSend,
    })
  }

  const handleSendCustomPrompt = () => {
    const trimmed = customPrompt.trim()
    if (!trimmed) return
    launchPrompt(trimmed, true)
    setCustomPrompt('')
  }

  return (
    <div className="space-y-4">
      <div className="shrink-0 px-1 overflow-x-auto">
        <div className="flex min-w-max items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToolChange('resolution')}
            className={cn(
              'h-8 gap-1.5 text-xs',
              tool === 'resolution'
                ? 'bg-violet-500/20 text-violet-300 border-violet-500/30 hover:bg-violet-500/30 hover:text-violet-300'
                : 'bg-card text-muted-foreground hover:text-foreground border-border'
            )}
          >
            <Shield className="h-3.5 w-3.5" />
            Resolution
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToolChange('market')}
            className={cn(
              'h-8 gap-1.5 text-xs',
              tool === 'market'
                ? 'bg-blue-500/20 text-blue-300 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-300'
                : 'bg-card text-muted-foreground hover:text-foreground border-border'
            )}
          >
            <TrendingUp className="h-3.5 w-3.5" />
            Market
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToolChange('news')}
            className={cn(
              'h-8 gap-1.5 text-xs',
              tool === 'news'
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/30 hover:bg-amber-500/30 hover:text-amber-300'
                : 'bg-card text-muted-foreground hover:text-foreground border-border'
            )}
          >
            <Newspaper className="h-3.5 w-3.5" />
            News
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-border/60 bg-card/35 px-3 py-2.5">
        <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">Analyze Workspace</p>
        <p className="mt-1 text-xs text-muted-foreground">
          {activeToolLabel} tool active. {activeToolDescription}
        </p>
      </div>

      <Card className="overflow-hidden border-border/60 bg-card/45 shadow-none">
        <div className="h-0.5 bg-gradient-to-r from-violet-500/60 via-cyan-500/50 to-amber-500/50" />
        <CardContent className="space-y-3 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">Prompt Studio</p>
              <p className="text-xs text-muted-foreground">Launch context-aware prompts directly into the floating AI copilot.</p>
            </div>
            <Badge variant="outline" className="h-5 border-border/60 bg-background/60 px-1.5 text-[10px]">
              Copilot Linked
            </Badge>
          </div>

          <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-3">
            {promptTemplates.map((template) => (
              <button
                key={template.title}
                type="button"
                onClick={() => launchPrompt(template.body, true)}
                className="rounded-lg border border-border/60 bg-background/55 p-3 text-left transition-colors hover:border-violet-500/30 hover:bg-violet-500/5"
                disabled={!onOpenCopilot}
              >
                <p className="text-xs font-semibold text-foreground">{template.title}</p>
                <p className="mt-1 text-[11px] text-muted-foreground line-clamp-3">{template.body}</p>
              </button>
            ))}
          </div>

          <div className="rounded-lg border border-border/60 bg-background/55 p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/70">Custom Prompt Box</p>
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              rows={3}
              placeholder="Draft your own request for the copilot..."
              className="mt-2 w-full resize-none rounded-lg border border-border bg-card/60 px-3 py-2 text-sm focus:outline-none focus:border-violet-500/50"
            />
            <div className="mt-2 flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => launchPrompt(customPrompt.trim(), false)}
                disabled={!customPrompt.trim() || !onOpenCopilot}
                className="h-7 gap-1.5 text-[11px]"
              >
                <Brain className="h-3 w-3" />
                Draft In Copilot
              </Button>
              <Button
                size="sm"
                onClick={handleSendCustomPrompt}
                disabled={!customPrompt.trim() || !onOpenCopilot}
                className="h-7 gap-1.5 bg-violet-500 text-white hover:bg-violet-600 text-[11px]"
              >
                <Sparkles className="h-3 w-3" />
                Send To Copilot
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Resolution Tool */}
      {tool === 'resolution' && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-purple-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Shield className="h-4 w-4 text-purple-400" />
              Resolution Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
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
                  className="pl-10 bg-muted/60 rounded-lg border-border focus-visible:ring-purple-500"
                />
              </div>
              {showSearchResults && searchResults.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-background border border-border rounded-xl shadow-2xl shadow-black/20 max-h-64 overflow-y-auto">
                  {searchResults.map((m) => (
                    <button
                      key={m.market_id}
                      onClick={() => selectMarket(m)}
                      className="w-full text-left px-3 py-2.5 hover:bg-muted/60 transition-colors border-b border-border/50 last:border-0"
                    >
                      <p className="text-sm text-foreground truncate">{m.question}</p>
                      <div className="flex items-center gap-2 mt-0.5 text-[10px] text-muted-foreground font-data">
                        {m.event_title && <span className="truncate max-w-[200px]">{m.event_title}</span>}
                        {m.category && <Badge variant="outline" className="text-[9px] h-4 px-1.5 capitalize">{m.category}</Badge>}
                        <span>YES: ${m.yes_price?.toFixed(2)}</span>
                        <span>Liq: ${m.liquidity?.toFixed(0)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {marketId && (
              <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg px-3 py-2.5 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-purple-400 flex-shrink-0" />
                <span className="text-xs text-purple-300 truncate flex-1 font-medium">
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
              <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Question *</label>
              <Input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Auto-filled from search, or type manually"
                className="bg-muted/60 rounded-lg focus-visible:ring-purple-500"
              />
            </div>

            <button
              onClick={() => setShowResolutionAdvanced(!showResolutionAdvanced)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showResolutionAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Additional fields
            </button>

            {showResolutionAdvanced && (
              <div className="space-y-3 pl-3 border-l-2 border-purple-500/20">
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Market ID</label>
                  <Input
                    type="text"
                    value={marketId}
                    onChange={(e) => setMarketId(e.target.value)}
                    placeholder="Auto-filled from search"
                    className="bg-muted/60 rounded-lg focus-visible:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Description</label>
                  <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Market description and resolution criteria..."
                    rows={2}
                    className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500 resize-none"
                  />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">End Date</label>
                    <Input type="text" value={endDate} onChange={(e) => setEndDate(e.target.value)} placeholder="2025-12-31" className="bg-muted/60 rounded-lg focus-visible:ring-purple-500" />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Resolution Source</label>
                    <Input type="text" value={resolutionSource} onChange={(e) => setResolutionSource(e.target.value)} placeholder="e.g., CoinGecko" className="bg-muted/60 rounded-lg focus-visible:ring-purple-500" />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Outcomes</label>
                    <Input type="text" value={outcomes} onChange={(e) => setOutcomes(e.target.value)} placeholder="Yes, No" className="bg-muted/60 rounded-lg focus-visible:ring-purple-500" />
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
              {resolutionMutation.isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
              Analyze Resolution
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Market Analysis Tool */}
      {tool === 'market' && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-blue-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              Market Intelligence
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Query *</label>
              <textarea
                value={marketQuery}
                onChange={(e) => setMarketQuery(e.target.value)}
                placeholder="e.g., What are the chances of a Fed rate cut in March? Analyze recent economic indicators..."
                rows={3}
                className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 resize-none"
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
              <div className="grid grid-cols-2 gap-3 pl-3 border-l-2 border-blue-500/20">
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Market ID</label>
                  <Input type="text" value={marketAnalysisId} onChange={(e) => setMarketAnalysisId(e.target.value)} placeholder="Link to specific market" className="bg-muted/60 rounded-lg focus-visible:ring-blue-500" />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Market Question</label>
                  <Input type="text" value={marketAnalysisQuestion} onChange={(e) => setMarketAnalysisQuestion(e.target.value)} placeholder="Market question for context" className="bg-muted/60 rounded-lg focus-visible:ring-blue-500" />
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
              {marketMutation.isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
              Analyze Market
            </Button>
          </CardContent>
        </Card>
      )}

      {/* News Sentiment Tool */}
      {tool === 'news' && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-orange-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Newspaper className="h-4 w-4 text-orange-400" />
              News Sentiment Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Search Query *</label>
              <Input
                type="text"
                value={newsQuery}
                onChange={(e) => setNewsQuery(e.target.value)}
                placeholder="e.g., Federal Reserve interest rate decision"
                className="bg-muted/60 rounded-lg focus-visible:ring-orange-500"
              />
            </div>

            <div>
              <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Market Context</label>
              <Input
                type="text"
                value={newsContext}
                onChange={(e) => setNewsContext(e.target.value)}
                placeholder="e.g., Will the Fed cut rates in March 2025?"
                className="bg-muted/60 rounded-lg focus-visible:ring-orange-500"
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
              <div className="pl-3 border-l-2 border-orange-500/20">
                <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Max Articles</label>
                <Input type="number" value={maxArticles} onChange={(e) => setMaxArticles(parseInt(e.target.value) || 5)} min={1} max={20} className="bg-muted/60 rounded-lg focus-visible:ring-orange-500 w-24" />
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
              {newsMutation.isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              Search & Analyze
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Resolution Results */}
      {tool === 'resolution' && resolutionMutation.data && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-purple-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <CheckCircle className="h-4 w-4 text-purple-400" />
              Resolution Analysis Result
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 card-stagger">
              <AIMetricTile
                label="Clarity"
                value={resolutionMutation.data.clarity_score?.toFixed(2) ?? 'N/A'}
                icon={<Shield className="h-3.5 w-3.5" />}
                tone={scoreTone(resolutionMutation.data.clarity_score)}
              />
              <AIMetricTile
                label="Risk"
                value={resolutionMutation.data.risk_score?.toFixed(2) ?? 'N/A'}
                icon={<AlertCircle className="h-3.5 w-3.5" />}
                tone={inverseScoreTone(resolutionMutation.data.risk_score)}
              />
              <AIMetricTile
                label="Confidence"
                value={resolutionMutation.data.confidence?.toFixed(2) ?? 'N/A'}
                icon={<Target className="h-3.5 w-3.5" />}
                tone={scoreTone(resolutionMutation.data.confidence)}
              />
              <AIMetricTile
                label="Resolution"
                value={resolutionMutation.data.resolution_likelihood?.toFixed(2) ?? 'N/A'}
                icon={<CheckCircle className="h-3.5 w-3.5" />}
                tone={scoreTone(resolutionMutation.data.resolution_likelihood)}
              />
            </div>

            <div className="space-y-3">
              <div className="rounded-lg border border-purple-500/20 bg-purple-500/5 p-4">
                <p className="text-[10px] uppercase tracking-wide text-purple-400 mb-1.5">Recommendation</p>
                <p className="text-sm text-foreground">{resolutionMutation.data.recommendation}</p>
              </div>
              <div className="rounded-lg border border-border/60 bg-muted/30 p-4">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Summary</p>
                <p className="text-sm text-foreground/90">{resolutionMutation.data.summary}</p>
              </div>
              {resolutionMutation.data.ambiguities?.length > 0 && (
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
                  <p className="text-[10px] uppercase tracking-wide text-amber-400 mb-2">Ambiguities</p>
                  <div className="space-y-1.5">
                    {resolutionMutation.data.ambiguities.map((a: string, i: number) => (
                      <div key={i} className="flex items-start gap-2">
                        <AlertCircle className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
                        <p className="text-sm text-amber-300/90">{a}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {resolutionMutation.data.edge_cases?.length > 0 && (
                <div className="rounded-lg border border-orange-500/20 bg-orange-500/5 p-4">
                  <p className="text-[10px] uppercase tracking-wide text-orange-400 mb-2">Edge Cases</p>
                  <div className="space-y-1.5">
                    {resolutionMutation.data.edge_cases.map((e: string, i: number) => (
                      <div key={i} className="flex items-start gap-2">
                        <Zap className="w-3 h-3 text-orange-400 mt-0.5 shrink-0" />
                        <p className="text-sm text-orange-300/90">{e}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
      {tool === 'resolution' && resolutionMutation.error && (
        <ErrorBanner message={(resolutionMutation.error as Error).message} />
      )}

      {/* Market Analysis Results */}
      {tool === 'market' && marketMutation.data && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-blue-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              Market Analysis Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-muted/30 p-4 rounded-lg border border-border/60 whitespace-pre-wrap text-sm leading-relaxed">
              {typeof marketMutation.data === 'string'
                ? marketMutation.data
                : marketMutation.data.analysis || marketMutation.data.result || JSON.stringify(marketMutation.data, null, 2)}
            </div>
          </CardContent>
        </Card>
      )}
      {tool === 'market' && marketMutation.error && (
        <ErrorBanner message={(marketMutation.error as Error).message} />
      )}

      {/* News Sentiment Results */}
      {tool === 'news' && newsMutation.data && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-orange-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Newspaper className="h-4 w-4 text-orange-400" />
              Sentiment Analysis Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-muted/30 p-4 rounded-lg border border-border/60 whitespace-pre-wrap text-sm leading-relaxed">
              {typeof newsMutation.data === 'string'
                ? newsMutation.data
                : newsMutation.data.summary || newsMutation.data.analysis || JSON.stringify(newsMutation.data, null, 2)}
            </div>
          </CardContent>
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

  const agRate = agreementStats?.agreement_rate ?? 0
  const avgScore = agreementStats?.avg_score ?? 0

  return (
    <div className="space-y-4">
      {/* Agreement Stats */}
      {agreementStats && (
        <Card className="overflow-hidden border-border/60 bg-card/80">
          <div className="h-0.5 bg-cyan-400" />
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Activity className="h-4 w-4 text-cyan-400" />
              ML vs LLM Agreement
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 card-stagger">
              <AIMetricTile
                label="Total Judged"
                value={String(agreementStats.total_judged ?? 0)}
                icon={<BarChart3 className="h-3.5 w-3.5" />}
                tone="info"
              />
              <AIMetricTile
                label="Agreement Rate"
                value={`${(agRate * 100).toFixed(1)}%`}
                icon={<Target className="h-3.5 w-3.5" />}
                tone={agRate >= 0.7 ? 'good' : agRate >= 0.4 ? 'warn' : 'bad'}
              />
              <AIMetricTile
                label="ML Overrides"
                value={String(agreementStats.ml_overrides ?? 0)}
                icon={<Zap className="h-3.5 w-3.5" />}
                tone="neutral"
              />
              <AIMetricTile
                label="Avg Score"
                value={avgScore.toFixed(2)}
                icon={<Activity className="h-3.5 w-3.5" />}
                tone={avgScore >= 0.7 ? 'good' : avgScore >= 0.4 ? 'warn' : 'neutral'}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Judgment History */}
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <div className="h-0.5 bg-emerald-400" />
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <Target className="h-4 w-4 text-emerald-400" />
            Recent Judgments
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!history || history.length === 0 ? (
            <EmptyState message="No opportunity judgments yet. AI will judge opportunities during scans when enabled." />
          ) : (
            <ScrollArea className="h-[480px] pr-3">
              <div className="space-y-2">
                {(Array.isArray(history) ? history : []).map((j: any, i: number) => (
                  <div
                    key={j.opportunity_id || i}
                    className="rounded-lg border border-border/55 bg-background/30 p-3 hover:border-border/80 transition-colors"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{j.opportunity_id}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className="text-[9px] h-4 px-1.5 bg-muted/30 border-border/40">
                            {j.strategy_type ?? 'unknown'}
                          </Badge>
                          <Badge
                            variant="outline"
                            className={cn(
                              'text-[9px] h-4 px-1.5',
                              j.recommendation === 'take' || j.recommendation === 'buy'
                                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                : j.recommendation === 'skip' || j.recommendation === 'avoid'
                                  ? 'bg-red-500/10 text-red-400 border-red-500/20'
                                  : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                            )}
                          >
                            {j.recommendation}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 ml-4 shrink-0">
                        <ScoreBadge label="Score" value={j.overall_score} />
                        <ScoreBadge label="Profit" value={j.profit_viability} />
                        <ScoreBadge label="Safety" value={j.resolution_safety} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ============================================================
// System Section
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
  const { data: status, isLoading: statusLoading, error: statusError } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const { data } = await getAIStatus()
      return data
    },
    refetchInterval: 30000,
  })
  const { data: usageFallback, isLoading: usageLoading, error: usageError } = useQuery({
    queryKey: ['ai-usage'],
    queryFn: async () => {
      const { data } = await getAIUsage()
      return data
    },
    refetchInterval: 30000,
    enabled: !!status?.enabled && status?.usage == null,
  })
  const usage = status?.usage ?? usageFallback
  const isLoading = statusLoading || (!!status?.enabled && usage == null && usageLoading)

  if (isLoading) return <LoadingSpinner />

  if (statusError || usageError) {
    return (
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <CardContent className="p-5">
          <div className="flex items-center gap-3 text-muted-foreground">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Unable to fetch usage stats.</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!usage) {
    return (
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <CardContent className="p-5">
          <EmptyState message="No usage data available yet." />
        </CardContent>
      </Card>
    )
  }

  const cost = usage.estimated_cost ?? usage.total_cost_usd ?? 0
  const failedReq = usage.failed_requests ?? usage.error_count ?? 0

  return (
    <Card className="overflow-hidden border-border/60 bg-card/80">
      <div className="h-0.5 bg-amber-400" />
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between gap-3 text-base font-semibold">
          <span className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-amber-400" />
            Usage Telemetry
          </span>
          {usage.active_model && (
            <Badge variant="outline" className="text-xs font-mono border-purple-500/25 bg-purple-500/10 text-purple-300">
              {usage.active_model}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 card-stagger">
          <AIMetricTile label="Requests" value={String(usage.total_requests ?? 0)} icon={<Zap className="h-3.5 w-3.5" />} tone="info" />
          <AIMetricTile label="Input Tokens" value={formatNumber(usage.total_input_tokens ?? 0)} icon={<FileText className="h-3.5 w-3.5" />} tone="info" />
          <AIMetricTile label="Output Tokens" value={formatNumber(usage.total_output_tokens ?? 0)} icon={<FileText className="h-3.5 w-3.5" />} tone="info" />
          <AIMetricTile label="Est. Cost" value={`$${cost.toFixed(4)}`} icon={<DollarSign className="h-3.5 w-3.5" />} tone={cost > 5 ? 'warn' : 'neutral'} />
          <AIMetricTile label="Avg Latency" value={`${(usage.avg_latency_ms ?? 0).toFixed(0)}ms`} icon={<Clock className="h-3.5 w-3.5" />} tone={(usage.avg_latency_ms ?? 0) > 5000 ? 'warn' : 'good'} />
          <AIMetricTile label="Total Tokens" value={formatNumber(usage.total_tokens ?? 0)} icon={<Activity className="h-3.5 w-3.5" />} tone="neutral" />
          <AIMetricTile label="Successful" value={String(usage.successful_requests ?? usage.total_requests ?? 0)} icon={<CheckCircle className="h-3.5 w-3.5" />} tone="good" />
          <AIMetricTile label="Failed" value={String(failedReq)} icon={<AlertCircle className="h-3.5 w-3.5" />} tone={failedReq > 0 ? 'bad' : 'good'} />
        </div>

        {/* Spend Limit */}
        {usage.spend_limit_usd != null && (
          <div className="rounded-lg border border-border/60 bg-muted/30 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Shield className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Monthly Spend Limit</span>
              </div>
              <span className="text-xs font-semibold font-data">
                ${cost.toFixed(2)} / ${usage.spend_limit_usd.toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-background rounded-full h-2 border border-border">
              <div
                className={cn(
                  'h-full rounded-full transition-all',
                  (cost / usage.spend_limit_usd) >= 0.9
                    ? 'bg-red-500'
                    : (cost / usage.spend_limit_usd) >= 0.7
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                )}
                style={{ width: `${Math.min(100, (cost / usage.spend_limit_usd) * 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between mt-1.5">
              <span className="text-[10px] text-muted-foreground font-data">
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
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">By Model</p>
            <div className="space-y-1.5">
              {Object.entries(usage.by_model).map(([model, stats]: [string, any]) => {
                const isActive = usage.active_model && model === usage.active_model
                return (
                  <div
                    key={model}
                    className={cn(
                      'flex items-center justify-between p-3 rounded-lg border transition-colors',
                      isActive
                        ? 'bg-purple-500/5 border-purple-500/20 shadow-sm shadow-purple-500/5'
                        : 'bg-background/30 border-border/55 hover:border-border/80'
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium font-mono">{model}</p>
                      {isActive && (
                        <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-purple-500/20 bg-purple-500/10 text-purple-400">
                          active
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-data">
                      <span>{stats.requests ?? 0} req</span>
                      <span>{formatNumber(stats.tokens ?? 0)} tok</span>
                      <span className="font-medium text-foreground/80">${(stats.cost ?? 0).toFixed(4)}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </CardContent>
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
    <Card className="overflow-hidden border-border/60 bg-card/80">
      <div className="h-0.5 bg-indigo-400" />
      <CardHeader className="pb-1">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center justify-between w-full"
        >
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <BookOpen className="h-4 w-4 text-indigo-400" />
            Research Sessions
          </CardTitle>
          {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        </button>
      </CardHeader>

      {expanded && (
        <CardContent className="pt-3">
          {isLoading ? (
            <LoadingSpinner />
          ) : (
            <>
              <div className="mb-3">
                <select
                  value={sessionTypeFilter}
                  onChange={(e) => setSessionTypeFilter(e.target.value)}
                  className="bg-muted/60 border border-border rounded-lg px-3 py-1.5 text-xs h-8"
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
                <ScrollArea className="h-72">
                  <div className="space-y-1.5 pr-3">
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
                                : 'bg-background/30 border-border/55 hover:border-border/80'
                            )}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <p className="text-xs font-medium truncate">{s.session_type || 'Unknown'}</p>
                                  <Badge variant="outline" className="text-[9px] h-4 px-1.5 bg-indigo-500/10 text-indigo-400 border-indigo-500/20">
                                    {s.session_type?.split('_')[0] || 'session'}
                                  </Badge>
                                </div>
                                <p className="text-[10px] text-muted-foreground truncate mt-0.5 font-mono">{id}</p>
                              </div>
                              <div className="flex items-center gap-2 ml-3">
                                <span className="text-[10px] text-muted-foreground whitespace-nowrap font-data">
                                  {s.created_at ? new Date(s.created_at).toLocaleString() : ''}
                                </span>
                                {isSelected ? <ChevronDown className="w-3 h-3 text-muted-foreground" /> : <ChevronRight className="w-3 h-3 text-muted-foreground" />}
                              </div>
                            </div>
                          </button>
                          {isSelected && (
                            <div className="mt-1 ml-3 border-l-2 border-indigo-500/20 pl-3">
                              {detailLoading ? (
                                <div className="py-3"><RefreshCw className="w-4 h-4 animate-spin text-indigo-400" /></div>
                              ) : sessionDetail ? (
                                <ScrollArea className="h-48">
                                  <pre className="text-[11px] text-muted-foreground whitespace-pre-wrap py-2 font-mono">
                                    {JSON.stringify(sessionDetail, null, 2)}
                                  </pre>
                                </ScrollArea>
                              ) : (
                                <p className="text-xs text-muted-foreground py-2">Session not found.</p>
                              )}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </ScrollArea>
              )}
            </>
          )}
        </CardContent>
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
    <Card className="overflow-hidden border-border/60 bg-card/80">
      <div className="h-0.5 bg-emerald-400" />
      <CardHeader className="pb-1">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center justify-between w-full"
        >
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <Layers className="h-4 w-4 text-emerald-400" />
            AI Skills
          </CardTitle>
          {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        </button>
      </CardHeader>

      {expanded && (
        <CardContent className="pt-3">
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
                      'w-full text-left p-3 rounded-lg border cursor-pointer transition-colors',
                      selectedSkill === skill.name
                        ? 'bg-emerald-500/10 border-emerald-500/30'
                        : 'bg-background/30 border-border/55 hover:border-border/80'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={cn(
                          'flex items-center justify-center w-7 h-7 rounded-md shrink-0',
                          selectedSkill === skill.name ? 'bg-emerald-500/15' : 'bg-muted/40'
                        )}>
                          <Zap className={cn('w-3.5 h-3.5', selectedSkill === skill.name ? 'text-emerald-400' : 'text-muted-foreground')} />
                        </div>
                        <div>
                          <p className="text-xs font-medium">{skill.name}</p>
                          <p className="text-[10px] text-muted-foreground">{skill.description || 'No description'}</p>
                        </div>
                      </div>
                      {selectedSkill === skill.name ? (
                        <ChevronDown className="w-3 h-3 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-muted-foreground" />
                      )}
                    </div>
                  </button>

                  {selectedSkill === skill.name && (
                    <div className="mt-2 ml-3 pl-3 border-l-2 border-emerald-500/20 space-y-2">
                      <div>
                        <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">Context (JSON)</label>
                        <textarea
                          value={skillContext}
                          onChange={(e) => setSkillContext(e.target.value)}
                          placeholder='{"market_id": "...", "question": "..."}'
                          rows={3}
                          className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-emerald-500 resize-none"
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
                        {executeMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                        Execute
                      </Button>

                      {executeMutation.data && (
                        <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                          <pre className="text-[11px] text-muted-foreground whitespace-pre-wrap overflow-auto max-h-48 font-mono">
                            {typeof executeMutation.data === 'string'
                              ? executeMutation.data
                              : JSON.stringify(executeMutation.data, null, 2)}
                          </pre>
                        </div>
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
        </CardContent>
      )}
    </Card>
  )
}

// ============================================================
// Shared Components
// ============================================================

function AIMetricTile({
  label,
  value,
  icon,
  tone,
  helper,
}: {
  label: string
  value: string
  icon: React.ReactNode
  tone: MetricTone
  helper?: string
}) {
  return (
    <Card className={cn('border', TONE_CLASSES[tone])}>
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[10px] uppercase tracking-wide opacity-85">{label}</p>
          <span className="opacity-80">{icon}</span>
        </div>
        <p className="mt-2 font-data text-lg font-semibold">{value}</p>
        {helper && <p className="mt-1 text-[11px] opacity-85">{helper}</p>}
      </CardContent>
    </Card>
  )
}

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-8">
      <RefreshCw className="w-6 h-6 animate-spin text-purple-400" />
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-8">
      <Bot className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
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

function ScoreBadge({ label, value }: { label: string; value: number }) {
  const color =
    value >= 0.7
      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
      : value >= 0.4
        ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
        : 'bg-red-500/10 text-red-400 border-red-500/20'
  return (
    <Badge variant="outline" className={cn('rounded text-[10px]', color)}>
      {label}: {typeof value === 'number' ? value.toFixed(2) : 'N/A'}
    </Badge>
  )
}

function scoreTone(value: number): MetricTone {
  if (value >= 0.7) return 'good'
  if (value >= 0.4) return 'warn'
  return 'bad'
}

function inverseScoreTone(value: number): MetricTone {
  if (value >= 0.7) return 'bad'
  if (value >= 0.4) return 'warn'
  return 'good'
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`
  return num.toString()
}
