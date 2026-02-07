import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Brain,
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
import clsx from 'clsx'
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
} from '../services/api'

type AISection = 'status' | 'resolution' | 'judgments' | 'market' | 'news' | 'skills' | 'sessions' | 'usage'

export default function AIPanel() {
  const [activeSection, setActiveSection] = useState<AISection>('status')
  const queryClient = useQueryClient()

  return (
    <div>
      {/* Section Navigation */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        <SectionButton
          active={activeSection === 'status'}
          onClick={() => setActiveSection('status')}
          icon={<Cpu className="w-4 h-4" />}
          label="Status"
        />
        <SectionButton
          active={activeSection === 'resolution'}
          onClick={() => setActiveSection('resolution')}
          icon={<Shield className="w-4 h-4" />}
          label="Resolution Analysis"
        />
        <SectionButton
          active={activeSection === 'judgments'}
          onClick={() => setActiveSection('judgments')}
          icon={<Target className="w-4 h-4" />}
          label="Judgments"
        />
        <SectionButton
          active={activeSection === 'market'}
          onClick={() => setActiveSection('market')}
          icon={<TrendingUp className="w-4 h-4" />}
          label="Market Analysis"
        />
        <SectionButton
          active={activeSection === 'news'}
          onClick={() => setActiveSection('news')}
          icon={<Newspaper className="w-4 h-4" />}
          label="News Sentiment"
        />
        <SectionButton
          active={activeSection === 'skills'}
          onClick={() => setActiveSection('skills')}
          icon={<Layers className="w-4 h-4" />}
          label="Skills"
        />
        <SectionButton
          active={activeSection === 'sessions'}
          onClick={() => setActiveSection('sessions')}
          icon={<BookOpen className="w-4 h-4" />}
          label="Sessions"
        />
        <SectionButton
          active={activeSection === 'usage'}
          onClick={() => setActiveSection('usage')}
          icon={<BarChart3 className="w-4 h-4" />}
          label="Usage"
        />
      </div>

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

// === Section Button ===

function SectionButton({
  active,
  onClick,
  icon,
  label,
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
        'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
        active
          ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
          : 'bg-[#1a1a1a] text-gray-400 hover:text-white border border-gray-800'
      )}
    >
      {icon}
      {label}
    </button>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <div className="flex items-center gap-3 text-gray-400">
          <AlertCircle className="w-5 h-5" />
          <span>Unable to fetch AI status. The AI module may not be configured.</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <div className="flex items-center gap-3 mb-4">
          <div className={clsx(
            'w-3 h-3 rounded-full',
            status?.enabled ? 'bg-green-500' : 'bg-gray-600'
          )} />
          <h3 className="text-lg font-semibold">
            AI Intelligence {status?.enabled ? 'Active' : 'Inactive'}
          </h3>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-800">
            <p className="text-xs text-gray-500 mb-1">Status</p>
            <p className={clsx('text-sm font-medium', status?.enabled ? 'text-green-400' : 'text-gray-500')}>
              {status?.enabled ? 'Enabled' : 'No Providers Configured'}
            </p>
          </div>
          <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-800">
            <p className="text-xs text-gray-500 mb-1">Providers</p>
            <p className="text-sm font-medium">
              {status?.providers_configured?.length > 0
                ? status.providers_configured.join(', ')
                : 'None'}
            </p>
          </div>
          <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-800">
            <p className="text-xs text-gray-500 mb-1">Skills Available</p>
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
      </div>

      {status?.usage && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Quick Usage Summary</h4>
          <div className="grid grid-cols-4 gap-4">
            <MiniStat label="Total Requests" value={status.usage.total_requests ?? 0} />
            <MiniStat label="Total Tokens" value={status.usage.total_tokens ?? 0} />
            <MiniStat label="Est. Cost" value={`$${(status.usage.estimated_cost ?? 0).toFixed(4)}`} />
            <MiniStat label="Avg Latency" value={`${(status.usage.avg_latency_ms ?? 0).toFixed(0)}ms`} />
          </div>
        </div>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-purple-400" />
          Resolution Criteria Analysis
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          Analyze how a market will resolve, identify ambiguities and edge cases.
        </p>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Market ID *</label>
              <input
                type="text"
                value={marketId}
                onChange={(e) => setMarketId(e.target.value)}
                placeholder="e.g., 0x1234..."
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">End Date</label>
              <input
                type="text"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                placeholder="e.g., 2025-12-31"
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Question *</label>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., Will Bitcoin reach $100k by end of 2025?"
              className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Market description and resolution criteria..."
              rows={3}
              className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Resolution Source</label>
              <input
                type="text"
                value={resolutionSource}
                onChange={(e) => setResolutionSource(e.target.value)}
                placeholder="e.g., CoinGecko price feed"
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Outcomes (comma-separated)</label>
              <input
                type="text"
                value={outcomes}
                onChange={(e) => setOutcomes(e.target.value)}
                placeholder="e.g., Yes, No"
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          <button
            onClick={() => analyzeMutation.mutate()}
            disabled={!marketId || !question || analyzeMutation.isPending}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              !marketId || !question || analyzeMutation.isPending
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                : 'bg-purple-500 hover:bg-purple-600 text-white'
            )}
          >
            {analyzeMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            Analyze Resolution
          </button>
        </div>
      </div>

      {/* Analysis Result */}
      {analyzeMutation.data && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Analysis Result</h4>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <ScoreCard label="Clarity" value={analyzeMutation.data.clarity_score} />
            <ScoreCard label="Risk" value={analyzeMutation.data.risk_score} />
            <ScoreCard label="Confidence" value={analyzeMutation.data.confidence} />
            <ScoreCard label="Resolution Likelihood" value={analyzeMutation.data.resolution_likelihood} />
          </div>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-gray-500 mb-1">Recommendation</p>
              <p className="text-sm bg-[#1a1a1a] p-3 rounded-lg border border-gray-800">
                {analyzeMutation.data.recommendation}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-1">Summary</p>
              <p className="text-sm bg-[#1a1a1a] p-3 rounded-lg border border-gray-800">
                {analyzeMutation.data.summary}
              </p>
            </div>
            {analyzeMutation.data.ambiguities?.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">Ambiguities</p>
                <ul className="list-disc list-inside text-sm bg-[#1a1a1a] p-3 rounded-lg border border-gray-800 space-y-1">
                  {analyzeMutation.data.ambiguities.map((a: string, i: number) => (
                    <li key={i} className="text-yellow-400">{a}</li>
                  ))}
                </ul>
              </div>
            )}
            {analyzeMutation.data.edge_cases?.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">Edge Cases</p>
                <ul className="list-disc list-inside text-sm bg-[#1a1a1a] p-3 rounded-lg border border-gray-800 space-y-1">
                  {analyzeMutation.data.edge_cases.map((e: string, i: number) => (
                    <li key={i} className="text-orange-400">{e}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
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
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
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
        </div>
      )}

      {/* Judgment History */}
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
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
                className="flex items-center justify-between bg-[#1a1a1a] p-3 rounded-lg border border-gray-800"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{j.opportunity_id}</p>
                  <p className="text-xs text-gray-500">
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
      </div>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-400" />
          AI Market Analysis
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          Ask the AI to analyze any market or topic with free-form queries.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Query *</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., What are the chances of a Fed rate cut in March? Analyze recent economic indicators..."
              rows={3}
              className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 resize-none"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Market ID (optional)</label>
              <input
                type="text"
                value={marketId}
                onChange={(e) => setMarketId(e.target.value)}
                placeholder="Link to specific market"
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Market Question (optional)</label>
              <input
                type="text"
                value={marketQuestion}
                onChange={(e) => setMarketQuestion(e.target.value)}
                placeholder="Market question for context"
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <button
            onClick={() => analyzeMutation.mutate()}
            disabled={!query || analyzeMutation.isPending}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              !query || analyzeMutation.isPending
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            )}
          >
            {analyzeMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            Analyze
          </button>
        </div>
      </div>

      {analyzeMutation.data && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Analysis Result</h4>
          <div className="bg-[#1a1a1a] p-4 rounded-lg border border-gray-800 whitespace-pre-wrap text-sm">
            {typeof analyzeMutation.data === 'string'
              ? analyzeMutation.data
              : analyzeMutation.data.analysis || analyzeMutation.data.result || JSON.stringify(analyzeMutation.data, null, 2)}
          </div>
        </div>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-orange-400" />
          News Sentiment Analysis
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          Search recent news and analyze sentiment for market-relevant topics.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Search Query *</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Federal Reserve interest rate decision"
              className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Market Context</label>
              <input
                type="text"
                value={marketContext}
                onChange={(e) => setMarketContext(e.target.value)}
                placeholder="e.g., Will the Fed cut rates in March 2025?"
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Max Articles</label>
              <input
                type="number"
                value={maxArticles}
                onChange={(e) => setMaxArticles(parseInt(e.target.value) || 5)}
                min={1}
                max={20}
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500"
              />
            </div>
          </div>

          <button
            onClick={() => sentimentMutation.mutate()}
            disabled={!query || sentimentMutation.isPending}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              !query || sentimentMutation.isPending
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                : 'bg-orange-500 hover:bg-orange-600 text-white'
            )}
          >
            {sentimentMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            Search & Analyze
          </button>
        </div>
      </div>

      {sentimentMutation.data && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Sentiment Result</h4>
          <div className="bg-[#1a1a1a] p-4 rounded-lg border border-gray-800 whitespace-pre-wrap text-sm">
            {typeof sentimentMutation.data === 'string'
              ? sentimentMutation.data
              : sentimentMutation.data.summary || sentimentMutation.data.analysis || JSON.stringify(sentimentMutation.data, null, 2)}
          </div>
        </div>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
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
                className={clsx(
                  'p-3 rounded-lg border cursor-pointer transition-colors',
                  selectedSkill === skill.name
                    ? 'bg-emerald-500/10 border-emerald-500/30'
                    : 'bg-[#1a1a1a] border-gray-800 hover:border-gray-700'
                )}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{skill.name}</p>
                    <p className="text-xs text-gray-500">{skill.description || 'No description'}</p>
                  </div>
                  {selectedSkill === skill.name ? (
                    <ChevronDown className="w-4 h-4 text-gray-500" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-gray-500" />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedSkill && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">
            Execute: {selectedSkill}
          </h4>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Context (JSON)</label>
              <textarea
                value={skillContext}
                onChange={(e) => setSkillContext(e.target.value)}
                placeholder='{"market_id": "...", "question": "..."}'
                rows={4}
                className="w-full bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-emerald-500 resize-none"
              />
            </div>
            <button
              onClick={() => executeMutation.mutate()}
              disabled={executeMutation.isPending}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                executeMutation.isPending
                  ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  : 'bg-emerald-500 hover:bg-emerald-600 text-white'
              )}
            >
              {executeMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              Execute Skill
            </button>
          </div>

          {executeMutation.data && (
            <div className="mt-4 bg-[#1a1a1a] p-4 rounded-lg border border-gray-800">
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
        </div>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-indigo-400" />
          Research Sessions
        </h3>

        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">Filter by Type</label>
          <select
            value={sessionTypeFilter}
            onChange={(e) => setSessionTypeFilter(e.target.value)}
            className="bg-[#1a1a1a] border border-gray-800 rounded-lg px-3 py-2 text-sm"
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
                className={clsx(
                  'p-3 rounded-lg border cursor-pointer transition-colors',
                  selectedSessionId === (s.session_id || s.id)
                    ? 'bg-indigo-500/10 border-indigo-500/30'
                    : 'bg-[#1a1a1a] border-gray-800 hover:border-gray-700'
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{s.session_type || 'Unknown'}</p>
                    <p className="text-xs text-gray-500">
                      {s.session_id || s.id}
                    </p>
                  </div>
                  <div className="text-xs text-gray-500 ml-4">
                    <Clock className="w-3 h-3 inline mr-1" />
                    {s.created_at ? new Date(s.created_at).toLocaleString() : 'Unknown'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedSessionId && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">
            Session Details
          </h4>
          {detailLoading ? (
            <LoadingSpinner />
          ) : sessionDetail ? (
            <div className="bg-[#1a1a1a] p-4 rounded-lg border border-gray-800">
              <pre className="text-sm whitespace-pre-wrap overflow-auto max-h-96">
                {JSON.stringify(sessionDetail, null, 2)}
              </pre>
            </div>
          ) : (
            <EmptyState message="Session not found or has been deleted." />
          )}
        </div>
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
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <div className="flex items-center gap-3 text-gray-400">
          <AlertCircle className="w-5 h-5" />
          <span>Unable to fetch usage stats. AI may not be configured.</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-yellow-400" />
          LLM Usage Statistics
        </h3>

        {!usage ? (
          <EmptyState message="No usage data available yet." />
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <UsageStat
              icon={<Zap className="w-4 h-4 text-blue-400" />}
              label="Total Requests"
              value={usage.total_requests ?? 0}
            />
            <UsageStat
              icon={<FileText className="w-4 h-4 text-green-400" />}
              label="Input Tokens"
              value={formatNumber(usage.input_tokens ?? usage.total_input_tokens ?? 0)}
            />
            <UsageStat
              icon={<FileText className="w-4 h-4 text-purple-400" />}
              label="Output Tokens"
              value={formatNumber(usage.output_tokens ?? usage.total_output_tokens ?? 0)}
            />
            <UsageStat
              icon={<DollarSign className="w-4 h-4 text-yellow-400" />}
              label="Estimated Cost"
              value={`$${(usage.estimated_cost ?? usage.total_cost ?? 0).toFixed(4)}`}
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
              value={usage.failed_requests ?? 0}
            />
          </div>
        )}
      </div>

      {usage?.by_model && typeof usage.by_model === 'object' && (
        <div className="bg-[#141414] rounded-lg p-6 border border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Usage by Model</h4>
          <div className="space-y-2">
            {Object.entries(usage.by_model).map(([model, stats]: [string, any]) => (
              <div key={model} className="flex items-center justify-between bg-[#1a1a1a] p-3 rounded-lg border border-gray-800">
                <p className="text-sm font-medium font-mono">{model}</p>
                <div className="flex items-center gap-4 text-xs text-gray-500">
                  <span>{stats.requests ?? 0} requests</span>
                  <span>{formatNumber(stats.tokens ?? 0)} tokens</span>
                  <span>${(stats.cost ?? 0).toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// === Shared Components ===

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-8">
      <AlertCircle className="w-10 h-10 text-gray-600 mx-auto mb-3" />
      <p className="text-sm text-gray-500">{message}</p>
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
    <div className="bg-[#1a1a1a] rounded-lg p-3 border border-gray-800">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-sm font-semibold mt-0.5">{value}</p>
    </div>
  )
}

function ScoreCard({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'text-green-400' : value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="bg-[#1a1a1a] rounded-lg p-3 border border-gray-800 text-center">
      <p className="text-xs text-gray-500">{label}</p>
      <p className={clsx('text-lg font-bold mt-1', color)}>
        {typeof value === 'number' ? value.toFixed(2) : value ?? 'N/A'}
      </p>
    </div>
  )
}

function ScoreBadge({ label, value }: { label: string; value: number }) {
  const color = value >= 0.7 ? 'bg-green-500/10 text-green-400' : value >= 0.4 ? 'bg-yellow-500/10 text-yellow-400' : 'bg-red-500/10 text-red-400'
  return (
    <div className={clsx('px-2 py-1 rounded text-xs font-medium', color)}>
      {label}: {typeof value === 'number' ? value.toFixed(2) : 'N/A'}
    </div>
  )
}

function UsageStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-800">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <p className="text-xs text-gray-500">{label}</p>
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
