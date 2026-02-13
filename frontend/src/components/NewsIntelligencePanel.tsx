import { useState, useMemo, useEffect, useCallback } from 'react'
import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Newspaper,
  RefreshCw,
  ExternalLink,
  Play,
  Pause,
  Target,
  Settings,
  Search,
  Trash2,
  Loader2,
  ChevronRight,
  X,
  Radio,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  getNewsFeedStatus,
  getNewsArticles,
  triggerNewsFetch,
  clearNewsArticles,
  getNewsWorkflowStatus,
  getNewsWorkflowFindings,
  getNewsWorkflowIntents,
  runNewsWorkflow,
  startNewsWorkflow,
  pauseNewsWorkflow,
  skipNewsWorkflowIntent,
  NewsArticle,
  NewsSupportingArticle,
  NewsWorkflowFinding,
  NewsTradeIntent,
} from '../services/api'
import { buildYesNoSparklineSeries } from '../lib/priceHistory'
import NewsWorkflowSettingsFlyout from './NewsWorkflowSettingsFlyout'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Card } from './ui/card'
import { Input } from './ui/input'
import Sparkline from './Sparkline'

type SubView = 'workflow' | 'feed'

const ISO_HAS_TIMEZONE_RE = /(Z|[+-]\d{2}:\d{2})$/i

function parseUtcDate(dateStr: string | null | undefined): Date | null {
  if (!dateStr) return null
  const trimmed = dateStr.trim()
  if (!trimmed) return null
  const normalized = ISO_HAS_TIMEZONE_RE.test(trimmed) ? trimmed : `${trimmed}Z`
  const date = new Date(normalized)
  if (isNaN(date.getTime())) return null
  return date
}

function timeAgo(dateStr: string | null | undefined): string {
  const date = parseUtcDate(dateStr)
  if (!date) return ''
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function articleRecencyMs(article: NewsArticle): number {
  return parseUtcDate(article.published)?.getTime()
    ?? parseUtcDate(article.fetched_at)?.getTime()
    ?? 0
}

function edgeColor(edge: number): string {
  if (edge >= 20) return 'text-green-400'
  if (edge >= 12) return 'text-emerald-400'
  if (edge >= 8) return 'text-yellow-400'
  return 'text-orange-400'
}

function confidenceColor(c: number): string {
  if (c >= 0.8) return 'text-green-400'
  if (c >= 0.6) return 'text-yellow-400'
  return 'text-orange-400'
}

const SOURCE_COLORS: Record<string, string> = {
  google_news: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  gdelt: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  custom_rss: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  rss: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  // Backward-compatible key for older cached rows.
  gov_rss: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
}

const CATEGORY_ICONS: Record<string, string> = {
  politics: '🏛',
  business: '💼',
  technology: '💻',
  science: '🔬',
  sports: '⚽',
  world: '🌍',
  cryptocurrency: '₿',
  entertainment: '🎬',
}

function toFiniteProbability(value: unknown): number | null {
  if (value === null || value === undefined) return null
  const n = Number(value)
  if (!Number.isFinite(n)) return null
  return Math.max(0, Math.min(1, n))
}

function formatCents(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}c`
}

function dedupeSupportingArticles(
  articles: NewsSupportingArticle[],
  maxItems = 24,
): NewsSupportingArticle[] {
  const deduped: NewsSupportingArticle[] = []
  const seen = new Set<string>()

  for (const article of articles) {
    const articleId = String(article.article_id || '').trim()
    const url = String(article.url || '').trim()
    const title = String(article.title || '').trim()
    const key = articleId || url || title.toLowerCase()
    if (!key || seen.has(key)) continue
    seen.add(key)
    deduped.push({
      article_id: articleId,
      title,
      url,
      source: String(article.source || '').trim(),
      published: article.published ?? null,
      fetched_at: article.fetched_at ?? null,
    })
    if (deduped.length >= maxItems) break
  }

  return deduped
}

function collectSupportingArticlesFromFinding(
  finding: NewsWorkflowFinding,
  maxItems = 24,
): NewsSupportingArticle[] {
  const evidence = finding.evidence as Record<string, unknown> | null
  const cluster = (evidence?.cluster && typeof evidence.cluster === 'object')
    ? (evidence.cluster as Record<string, unknown>)
    : null
  const supportingArticles: NewsSupportingArticle[] = []

  const payloadArticles = Array.isArray(finding.supporting_articles)
    ? finding.supporting_articles
    : []
  for (const article of payloadArticles) {
    const title = String(article?.title || '').trim()
    if (!title) continue
    supportingArticles.push({
      article_id: String(article?.article_id || '').trim(),
      title,
      url: String(article?.url || '').trim(),
      source: String(article?.source || '').trim(),
      published: article?.published ?? null,
      fetched_at: article?.fetched_at ?? null,
    })
  }

  const clusterRefs = Array.isArray(cluster?.article_refs)
    ? cluster.article_refs
    : []
  for (const raw of clusterRefs) {
    if (!raw || typeof raw !== 'object') continue
    const ref = raw as Record<string, unknown>
    const title = String(ref.title || '').trim()
    if (!title) continue
    supportingArticles.push({
      article_id: String(ref.article_id || '').trim(),
      title,
      url: String(ref.url || '').trim(),
      source: String(ref.source || '').trim(),
      published: typeof ref.published === 'string' ? ref.published : null,
      fetched_at: typeof ref.fetched_at === 'string' ? ref.fetched_at : null,
    })
  }

  if (!supportingArticles.length) {
    const fallbackTitle = String(finding.article_title || '').trim()
    if (fallbackTitle) {
      supportingArticles.push({
        article_id: String(finding.article_id || '').trim(),
        title: fallbackTitle,
        url: String(finding.article_url || '').trim(),
        source: String(finding.article_source || '').trim(),
        published: null,
        fetched_at: finding.created_at,
      })
    }
  }

  return dedupeSupportingArticles(supportingArticles, maxItems)
}

function resolveCurrentOddsForFinding(
  finding: NewsWorkflowFinding,
): { yes: number | null; no: number | null; signal: number | null } {
  const isBuyYes = finding.direction === 'buy_yes'

  let yes = (
    toFiniteProbability(finding.current_yes_price)
    ?? toFiniteProbability(finding.yes_price)
    ?? toFiniteProbability(finding.market_price)
  )
  let no = (
    toFiniteProbability(finding.current_no_price)
    ?? toFiniteProbability(finding.no_price)
  )

  if (no == null && yes != null) no = Math.max(0, Math.min(1, 1 - yes))
  if (yes == null && no != null) yes = Math.max(0, Math.min(1, 1 - no))

  return {
    yes,
    no,
    signal: isBuyYes ? yes : no,
  }
}

function findingCreatedAtMs(finding: NewsWorkflowFinding): number {
  return parseUtcDate(finding.created_at)?.getTime() ?? 0
}

function mergeFindingsByMarket(findings: NewsWorkflowFinding[]): NewsWorkflowFinding[] {
  const grouped = new Map<string, NewsWorkflowFinding>()

  for (const finding of findings) {
    const key = `${finding.market_id}::${finding.direction || 'buy_yes'}`
    const normalizedSupporting = collectSupportingArticlesFromFinding(finding)
    const candidate: NewsWorkflowFinding = {
      ...finding,
      supporting_articles: normalizedSupporting,
      supporting_article_count: Math.max(
        Number(finding.supporting_article_count ?? 0),
        normalizedSupporting.length,
      ),
    }

    const existing = grouped.get(key)
    if (!existing) {
      grouped.set(key, candidate)
      continue
    }

    const primary = candidate.edge_percent > existing.edge_percent ? candidate : existing
    const secondary = primary === candidate ? existing : candidate
    const mergedArticles = dedupeSupportingArticles(
      [...(existing.supporting_articles ?? []), ...(candidate.supporting_articles ?? [])],
      24,
    )

    const primaryHistoryLen = Array.isArray(primary.price_history) ? primary.price_history.length : 0
    const secondaryHistoryLen = Array.isArray(secondary.price_history) ? secondary.price_history.length : 0
    const snapshotSource = primaryHistoryLen >= secondaryHistoryLen ? primary : secondary
    const fallbackSnapshot = snapshotSource === primary ? secondary : primary

    const merged: NewsWorkflowFinding = {
      ...primary,
      actionable: existing.actionable || candidate.actionable,
      confidence: Math.max(existing.confidence, candidate.confidence),
      retrieval_score: Math.max(existing.retrieval_score, candidate.retrieval_score),
      semantic_score: Math.max(existing.semantic_score, candidate.semantic_score),
      keyword_score: Math.max(existing.keyword_score, candidate.keyword_score),
      event_score: Math.max(existing.event_score, candidate.event_score),
      rerank_score: Math.max(existing.rerank_score, candidate.rerank_score),
      created_at: findingCreatedAtMs(candidate) >= findingCreatedAtMs(existing)
        ? candidate.created_at
        : existing.created_at,
      supporting_articles: mergedArticles,
      supporting_article_count: Math.max(
        Number(existing.supporting_article_count ?? 0),
        Number(candidate.supporting_article_count ?? 0),
        mergedArticles.length,
      ),
      price_history: (
        Array.isArray(snapshotSource.price_history) && snapshotSource.price_history.length >= 2
      ) ? snapshotSource.price_history : fallbackSnapshot.price_history,
      yes_price: (
        snapshotSource.yes_price
        ?? fallbackSnapshot.yes_price
        ?? primary.yes_price
        ?? secondary.yes_price
      ),
      no_price: (
        snapshotSource.no_price
        ?? fallbackSnapshot.no_price
        ?? primary.no_price
        ?? secondary.no_price
      ),
      current_yes_price: (
        snapshotSource.current_yes_price
        ?? fallbackSnapshot.current_yes_price
        ?? primary.current_yes_price
        ?? secondary.current_yes_price
      ),
      current_no_price: (
        snapshotSource.current_no_price
        ?? fallbackSnapshot.current_no_price
        ?? primary.current_no_price
        ?? secondary.current_no_price
      ),
    }

    grouped.set(key, merged)
  }

  return Array.from(grouped.values())
}

function ArticleRow({ article }: { article: NewsArticle }) {
  const publishedAgo = timeAgo(article.published)
  const fetchedAgo = timeAgo(article.fetched_at)
  const timeStr = publishedAgo || fetchedAgo
  const timeLabel = publishedAgo ? `Published ${timeStr}` : `Added ${timeStr}`

  return (
    <div className="flex items-center gap-3 px-3 py-2.5 hover:bg-muted/30 rounded-lg transition-colors group">
      <div className="shrink-0 text-base">{CATEGORY_ICONS[article.category] || '📰'}</div>
      <div className="flex-1 min-w-0">
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-foreground hover:text-orange-400 transition-colors line-clamp-1 font-medium"
        >
          {article.title}
          <ExternalLink className="w-3 h-3 inline ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
        </a>
        <div className="flex items-center gap-2 mt-0.5">
          <Badge variant="outline" className={cn("text-[9px] h-4 px-1.5", SOURCE_COLORS[article.feed_source] || 'bg-muted/50 text-muted-foreground border-border')}>
            {article.feed_source.replace('_', ' ')}
          </Badge>
          <span className="text-[10px] text-muted-foreground truncate max-w-[180px]">{article.source}</span>
          {timeStr && <span className="text-[10px] text-muted-foreground font-data shrink-0">{timeLabel}</span>}
        </div>
      </div>
    </div>
  )
}

function FindingCard({ finding }: { finding: NewsWorkflowFinding }) {
  const [expandedArticles, setExpandedArticles] = useState(false)
  const evidence = finding.evidence as Record<string, unknown> | null
  const cluster = (evidence?.cluster && typeof evidence.cluster === 'object')
    ? (evidence.cluster as Record<string, unknown>)
    : null
  const clusterSources = Array.isArray(cluster?.sources)
    ? cluster.sources.filter((s): s is string => typeof s === 'string')
    : []
  const clusterSize = typeof cluster?.article_count === 'number'
    ? cluster.article_count
    : 0
  const dedupedSupportingArticles = useMemo(
    () => collectSupportingArticlesFromFinding(finding, 24),
    [finding],
  )
  const odds = useMemo(() => resolveCurrentOddsForFinding(finding), [finding])
  const sparkData = useMemo(
    () => buildYesNoSparklineSeries(finding.price_history, odds.yes, odds.no),
    [finding.price_history, odds.yes, odds.no],
  )
  const hasSparkline = sparkData.yes.length >= 2

  const articleCount = Math.max(
    Number(finding.supporting_article_count ?? 0),
    dedupedSupportingArticles.length,
    clusterSize,
  )
  const visibleSupportingArticles = expandedArticles
    ? dedupedSupportingArticles
    : dedupedSupportingArticles.slice(0, 2)
  const isBuyYes = finding.direction === 'buy_yes'

  return (
    <Card className="overflow-hidden border-border/40 hover:border-border/80 hover:shadow-lg hover:shadow-black/20 transition-all group">
      <div className={cn('h-0.5', finding.actionable ? (finding.edge_percent >= 15 ? 'bg-green-400' : 'bg-yellow-400') : 'bg-muted')} />
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline" className={cn(
              "text-[10px] font-semibold",
              isBuyYes ? "bg-green-500/10 text-green-400 border-green-500/20" : "bg-red-500/10 text-red-400 border-red-500/20"
            )}>
              {isBuyYes ? 'BUY YES' : 'BUY NO'}
            </Badge>
            {finding.actionable && (
              <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-400 border-green-500/20">
                Actionable
              </Badge>
            )}
            {articleCount > 1 && (
              <Badge variant="outline" className="text-[10px] bg-blue-500/10 text-blue-400 border-blue-500/20">
                {articleCount} articles
              </Badge>
            )}
          </div>
          <div className="text-right shrink-0">
            <span className={cn("text-lg font-bold font-data", edgeColor(finding.edge_percent))}>
              {finding.edge_percent.toFixed(1)}%
            </span>
            <span className="text-[10px] text-muted-foreground block">edge</span>
            <div className="text-[10px] font-data mt-0.5">
              <span className="text-green-400">Y {formatCents(odds.yes)}</span>
              <span className="text-muted-foreground mx-1">/</span>
              <span className="text-red-400">N {formatCents(odds.no)}</span>
            </div>
          </div>
        </div>

        <p className="text-sm font-medium text-foreground line-clamp-2 mb-2">{finding.market_question}</p>

        <div className="mb-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Newspaper className="w-3.5 h-3.5 text-orange-400 shrink-0" />
              <span className="text-[10px] text-muted-foreground">
                {articleCount > 1 ? `${articleCount} linked articles` : 'Linked article'}
              </span>
            </div>
            {dedupedSupportingArticles.length > 2 && (
              <button
                type="button"
                onClick={() => setExpandedArticles((v) => !v)}
                className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
              >
                {expandedArticles ? 'Hide' : `Show all (${articleCount})`}
              </button>
            )}
          </div>
          <div className="mt-1.5 space-y-1">
            {visibleSupportingArticles.map((article) => (
              article.url ? (
                <a
                  key={`${article.article_id}-${article.url}-${article.title}`}
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-xs text-muted-foreground hover:text-orange-400 transition-colors line-clamp-1"
                >
                  {article.title}
                  <ExternalLink className="w-3 h-3 inline ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                </a>
              ) : (
                <p
                  key={`${article.article_id}-${article.title}`}
                  className="text-xs text-muted-foreground line-clamp-1"
                >
                  {article.title}
                </p>
              )
            ))}
          </div>
        </div>

        <div className="flex items-stretch gap-2.5 mb-3">
          {hasSparkline && (
            <div className="shrink-0">
              <Sparkline
                data={sparkData.yes}
                data2={sparkData.no}
                width={96}
                height={40}
                color="#22c55e"
                color2="#ef4444"
                lineWidth={1.5}
                showDots
              />
              <div className="flex justify-between text-[9px] text-muted-foreground font-data mt-0.5 px-0.5">
                <span className="text-green-400/80">Y {odds.yes?.toFixed(2) ?? '—'}</span>
                <span className="text-red-400/80">N {odds.no?.toFixed(2) ?? '—'}</span>
              </div>
            </div>
          )}
          <div className="grid grid-cols-2 gap-1.5 flex-1">
            <div className="bg-muted/30 rounded-lg p-1.5 text-center">
              <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Current</div>
              <div className={cn("text-xs font-data font-semibold", edgeColor(finding.edge_percent))}>
                {formatCents(odds.signal)}
              </div>
            </div>
            <div className="bg-muted/30 rounded-lg p-1.5 text-center">
              <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Model</div>
              <div className={cn("text-xs font-data font-semibold", edgeColor(finding.edge_percent))}>{formatCents(finding.model_probability)}</div>
            </div>
            <div className="bg-muted/30 rounded-lg p-1.5 text-center">
              <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Conf</div>
              <div className={cn("text-xs font-data font-semibold", confidenceColor(finding.confidence))}>{(finding.confidence * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-muted/30 rounded-lg p-1.5 text-center">
              <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Rerank</div>
              <div className="text-xs font-data font-semibold text-blue-400">{(finding.rerank_score * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>

        <div className="text-[10px] text-muted-foreground mb-2 line-clamp-2">{finding.reasoning}</div>

        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <div className="flex items-center gap-2">
            {clusterSources.slice(0, 2).map((source) => (
              <Badge key={source} variant="outline" className="text-[9px] h-4 px-1.5 bg-muted/30 border-border/40">
                {source}
              </Badge>
            ))}
          </div>
          <span className="font-data">{timeAgo(finding.created_at)}</span>
        </div>
      </div>
    </Card>
  )
}

function IntentRow({ intent, onSkip, isSkipping }: { intent: NewsTradeIntent; onSkip: (id: string) => void; isSkipping: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const statusColors: Record<string, string> = {
    pending: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    submitted: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    executed: 'bg-green-500/10 text-green-400 border-green-500/20',
    skipped: 'bg-muted/50 text-muted-foreground border-border',
    expired: 'bg-red-500/10 text-red-400 border-red-500/20',
  }
  const supportingArticles: NewsSupportingArticle[] = (
    Array.isArray(intent.metadata?.supporting_articles)
      ? intent.metadata?.supporting_articles
      : []
  ).filter((article): article is NewsSupportingArticle => Boolean(article && article.title && article.url))
  const articleCount = Number(intent.metadata?.supporting_article_count ?? supportingArticles.length ?? 0)

  return (
    <div className="px-3 py-3 hover:bg-muted/30 rounded-lg transition-colors">
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground line-clamp-1">{intent.market_question}</p>
          <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
            <Badge variant="outline" className={cn("text-[9px] h-4 px-1.5", intent.direction === 'buy_yes' ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20')}>
              {intent.direction === 'buy_yes' ? 'BUY YES' : 'BUY NO'}
            </Badge>
            <span className="font-data">Edge: {intent.edge_percent.toFixed(1)}%</span>
            <span className="font-data">Size: ${intent.suggested_size_usd.toFixed(0)}</span>
            <Badge variant="outline" className={cn("text-[9px] h-4 px-1.5", statusColors[intent.status] || '')}>
              {intent.status}
            </Badge>
            {articleCount > 0 && (
              <Badge variant="outline" className="text-[9px] h-4 px-1.5 bg-blue-500/10 text-blue-400 border-blue-500/20">
                {articleCount} articles
              </Badge>
            )}
          </div>
          {supportingArticles.length > 0 && (
            <div className="mt-2">
              <button
                onClick={() => setExpanded((v) => !v)}
                className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
              >
                {expanded ? 'Hide linked articles' : 'Show linked articles'}
              </button>
              {expanded && (
                <div className="mt-1.5 space-y-1">
                  {supportingArticles.slice(0, 5).map((article) => (
                    <a
                      key={`${article.article_id}-${article.url}`}
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-[10px] text-muted-foreground hover:text-orange-400 transition-colors line-clamp-1"
                    >
                      {article.title}
                      <ExternalLink className="w-2.5 h-2.5 inline ml-1" />
                    </a>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
        {intent.status === 'pending' && (
          <Button
            variant="outline"
            size="sm"
            className="text-xs h-7 gap-1 text-muted-foreground hover:text-foreground"
            onClick={() => onSkip(intent.id)}
            disabled={isSkipping}
          >
            <X className="w-3 h-3" />
            Skip
          </Button>
        )}
      </div>
    </div>
  )
}

interface NewsIntelligencePanelProps {
  initialSearchQuery?: string
}

export default function NewsIntelligencePanel({ initialSearchQuery }: NewsIntelligencePanelProps = {}) {
  const queryClient = useQueryClient()
  const [subView, setSubView] = useState<SubView>(initialSearchQuery ? 'feed' : 'workflow')
  const [searchFilter, setSearchFilter] = useState(initialSearchQuery || '')
  const [feedSourceFilter, setFeedSourceFilter] = useState<string | null>(null)
  const [workflowSettingsOpen, setWorkflowSettingsOpen] = useState(false)

  useEffect(() => {
    if (initialSearchQuery !== undefined) {
      setSearchFilter(initialSearchQuery)
      if (initialSearchQuery) setSubView('feed')
    }
  }, [initialSearchQuery])

  const { data: workflowStatus } = useQuery({
    queryKey: ['news-workflow-status'],
    queryFn: getNewsWorkflowStatus,
    refetchInterval: 30000,
  })

  const { data: workflowFindingsData, isLoading: findingsLoading } = useQuery({
    queryKey: ['news-workflow-findings'],
    queryFn: () => getNewsWorkflowFindings({
      actionable_only: false,
      include_debug_rejections: false,
      max_age_hours: 24,
      limit: 150,
    }),
    refetchInterval: 30000,
    enabled: subView === 'workflow',
  })

  const { data: workflowIntentsData } = useQuery({
    queryKey: ['news-workflow-intents'],
    queryFn: () => getNewsWorkflowIntents({ limit: 50 }),
    refetchInterval: 15000,
    enabled: subView === 'workflow',
  })

  const runWorkflowMutation = useMutation({
    mutationFn: runNewsWorkflow,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-workflow-findings'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
    },
  })

  const startWorkflowMutation = useMutation({
    mutationFn: startNewsWorkflow,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
    },
  })

  const pauseWorkflowMutation = useMutation({
    mutationFn: pauseNewsWorkflow,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
    },
  })

  const skipIntentMutation = useMutation({
    mutationFn: skipNewsWorkflowIntent,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
    },
  })

  const { data: feedStatus } = useQuery({
    queryKey: ['news-feed-status'],
    queryFn: getNewsFeedStatus,
    refetchInterval: 60000,
  })

  const PAGE_SIZE = 100
  const {
    data: articlesPages,
    isLoading: articlesLoading,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ['news-articles', feedSourceFilter],
    queryFn: ({ pageParam = 0 }) =>
      getNewsArticles({
        max_age_hours: 168,
        limit: PAGE_SIZE,
        offset: pageParam,
        source: feedSourceFilter || undefined,
      }),
    getNextPageParam: (lastPage, _allPages, lastPageParam) => {
      if (lastPage.has_more) return (lastPageParam as number) + PAGE_SIZE
      return undefined
    },
    initialPageParam: 0,
    refetchInterval: 120000,
  })

  const fetchMutation = useMutation({
    mutationFn: triggerNewsFetch,
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
    },
  })

  const clearMutation = useMutation({
    mutationFn: clearNewsArticles,
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
      setSearchFilter('')
      setFeedSourceFilter(null)
    },
  })

  const unsortedArticles = useMemo(
    () => articlesPages?.pages.flatMap(page => page.articles) || [],
    [articlesPages],
  )
  const articles = useMemo(
    () => [...unsortedArticles].sort((a, b) => articleRecencyMs(b) - articleRecencyMs(a)),
    [unsortedArticles],
  )
  const articlesTotalCount = articlesPages?.pages[0]?.total ?? 0
  const articlesLoadedCount = articles.length

  const workflowFindings = useMemo(() => {
    const findings = workflowFindingsData?.findings || []
    const groupedFindings = mergeFindingsByMarket(findings)
    return groupedFindings.sort((a, b) => {
      if (a.actionable !== b.actionable) return a.actionable ? -1 : 1
      if (b.edge_percent !== a.edge_percent) return b.edge_percent - a.edge_percent
      return (parseUtcDate(b.created_at)?.getTime() || 0) - (parseUtcDate(a.created_at)?.getTime() || 0)
    })
  }, [workflowFindingsData])

  const actionableFindingsCount = useMemo(
    () => workflowFindings.filter(f => f.actionable).length,
    [workflowFindings],
  )

  const filteredArticles = useMemo(() => {
    if (!searchFilter) return articles
    const q = searchFilter.toLowerCase()
    return articles.filter(a =>
      a.title.toLowerCase().includes(q)
      || a.source.toLowerCase().includes(q)
      || a.category.toLowerCase().includes(q)
    )
  }, [articles, searchFilter])

  const filteredFindings = useMemo(() => {
    if (!searchFilter) return workflowFindings
    const q = searchFilter.toLowerCase()
    return workflowFindings.filter(f =>
      f.market_question.toLowerCase().includes(q)
      || f.article_title.toLowerCase().includes(q)
      || f.article_source.toLowerCase().includes(q)
      || collectSupportingArticlesFromFinding(f).some((article) =>
        article.title.toLowerCase().includes(q)
        || article.source.toLowerCase().includes(q)
      )
    )
  }, [workflowFindings, searchFilter])

  const sourceBreakdown = feedStatus?.sources || {}
  const sourceKeys = useMemo(() => Object.keys(sourceBreakdown), [sourceBreakdown])

  const handleLoadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) fetchNextPage()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])

  return (
    <div className="max-w-[1600px] mx-auto">
      <div className="mb-2 flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSubView('workflow')}
          className={cn(
            "gap-1.5 text-xs h-8",
            subView === 'workflow'
              ? "bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30 hover:text-purple-400"
              : "bg-card text-muted-foreground hover:text-foreground border-border"
          )}
        >
          <Target className="w-3.5 h-3.5" />
          Workflow
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSubView('feed')}
          className={cn(
            "gap-1.5 text-xs h-8",
            subView === 'feed'
              ? "bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30 hover:text-orange-400"
              : "bg-card text-muted-foreground hover:text-foreground border-border"
          )}
        >
          <Newspaper className="w-3.5 h-3.5" />
          Feed
          {articlesTotalCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-orange-500/15 text-orange-400 text-[10px] font-data">
              {articlesLoadedCount < articlesTotalCount ? `${articlesLoadedCount}/${articlesTotalCount}` : articlesTotalCount}
            </span>
          )}
        </Button>
      </div>

      <div className="mb-4 rounded-xl border border-border/40 bg-card/40 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <Newspaper className="w-4 h-4 text-orange-400 shrink-0" />
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] h-6',
              workflowStatus?.paused
                ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
                : workflowStatus?.enabled
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                  : 'bg-muted/40 text-muted-foreground border-border'
            )}
          >
            Autopilot {workflowStatus?.paused ? 'Paused' : workflowStatus?.enabled ? 'Active' : 'Disabled'}
          </Badge>
          {feedStatus?.running && (
            <Badge variant="outline" className="text-[10px] h-6 bg-green-500/10 text-green-400 border-green-500/20 gap-1">
              <Radio className="w-3 h-3" />
              Live
            </Badge>
          )}

          <Badge variant="outline" className="text-[10px] h-6 bg-card border-border/60 text-muted-foreground">
            Last {timeAgo(workflowStatus?.last_scan)}
          </Badge>
          <span className="text-xs text-muted-foreground truncate max-w-[280px]">
            {workflowStatus?.current_activity || 'Waiting for news worker'}
          </span>

          {subView === 'feed' && sourceKeys.length > 0 && (
            <select
              value={feedSourceFilter ?? '_all'}
              onChange={(e) => setFeedSourceFilter(e.target.value === '_all' ? null : e.target.value)}
              className="h-8 min-w-[150px] rounded-md border border-border bg-card px-2 text-xs text-foreground"
            >
              <option value="_all">All sources ({feedStatus?.article_count ?? 0})</option>
              {sourceKeys.map((src) => (
                <option key={src} value={src}>
                  {src.replace('_', ' ')} ({sourceBreakdown[src] as number})
                </option>
              ))}
            </select>
          )}

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Filter..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className={cn("pl-8 h-8 text-xs bg-card border-border", searchFilter ? "w-56 pr-8" : "w-48")}
              />
              {searchFilter && (
                <button
                  onClick={() => setSearchFilter('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {subView === 'workflow' ? (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 gap-1.5"
                  onClick={() => (workflowStatus?.paused ? startWorkflowMutation.mutate() : pauseWorkflowMutation.mutate())}
                  disabled={startWorkflowMutation.isPending || pauseWorkflowMutation.isPending}
                >
                  {workflowStatus?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                  {workflowStatus?.paused ? 'Resume' : 'Pause'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 gap-1 text-purple-400 border-purple-500/20 hover:bg-purple-500/10 hover:text-purple-400"
                  onClick={() => runWorkflowMutation.mutate()}
                  disabled={runWorkflowMutation.isPending}
                >
                  {runWorkflowMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                  Run
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 gap-1"
                  onClick={() => fetchMutation.mutate()}
                  disabled={fetchMutation.isPending}
                >
                  {fetchMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                  Fetch
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 gap-1.5"
                  onClick={() => setWorkflowSettingsOpen(true)}
                >
                  <Settings className="w-3.5 h-3.5" />
                  Settings
                </Button>
              </>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 gap-1"
                  onClick={() => fetchMutation.mutate()}
                  disabled={fetchMutation.isPending}
                >
                  {fetchMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                  Fetch
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs h-8 gap-1 text-red-400 hover:text-red-400 hover:bg-red-500/10"
                  onClick={() => clearMutation.mutate()}
                  disabled={clearMutation.isPending}
                >
                  <Trash2 className="w-3 h-3" />
                  Clear
                </Button>
              </>
            )}
          </div>
        </div>
      </div>

      {subView === 'workflow' && (
        <>
          {(workflowIntentsData?.intents?.length ?? 0) > 0 && (
            <div className="mb-4">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Target className="w-3 h-3 text-purple-400" />
                Trade Intents ({workflowIntentsData?.intents.length})
              </div>
              <div className="bg-card/40 rounded-xl border border-border/30 divide-y divide-border/20">
                {workflowIntentsData?.intents.map((intent) => (
                  <IntentRow
                    key={intent.id}
                    intent={intent}
                    onSkip={(id) => skipIntentMutation.mutate(id)}
                    isSkipping={skipIntentMutation.isPending}
                  />
                ))}
              </div>
            </div>
          )}

          {findingsLoading || runWorkflowMutation.isPending ? (
            <div className="flex flex-col items-center justify-center py-16">
              <RefreshCw className="w-8 h-8 animate-spin text-purple-400 mb-3" />
              <p className="text-sm text-muted-foreground">
                {runWorkflowMutation.isPending ? 'Queueing workflow cycle...' : 'Loading findings...'}
              </p>
            </div>
          ) : filteredFindings.length === 0 ? (
            <div className="text-center py-16">
              <Target className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {searchFilter ? 'No findings match your filter' : 'No findings yet'}
              </p>
            </div>
          ) : (
            <>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Zap className="w-3 h-3 text-green-400" />
                Findings ({workflowFindings.length})
                {actionableFindingsCount ? ` / ${actionableFindingsCount} actionable` : ''}
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
                {filteredFindings.map((finding) => (
                  <FindingCard key={finding.id} finding={finding} />
                ))}
              </div>
            </>
          )}
        </>
      )}

      {subView === 'feed' && (
        <>
          {articlesLoading ? (
            <div className="flex items-center justify-center py-16">
              <RefreshCw className="w-8 h-8 animate-spin text-orange-400" />
            </div>
          ) : filteredArticles.length === 0 ? (
            <div className="text-center py-16">
              <Newspaper className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {searchFilter ? 'No articles match your filter' : 'No articles in feed'}
              </p>
              {!searchFilter && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 gap-1.5 mt-3"
                  onClick={() => fetchMutation.mutate()}
                  disabled={fetchMutation.isPending}
                >
                  {fetchMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Newspaper className="w-3.5 h-3.5" />}
                  Fetch News
                </Button>
              )}
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-2 text-xs text-muted-foreground">
                <span>
                  Showing {filteredArticles.length}
                  {searchFilter ? ` of ${articlesLoadedCount} loaded` : ''}
                  {articlesTotalCount > articlesLoadedCount ? ` (${articlesTotalCount} total)` : ''}
                </span>
              </div>

              <div className="bg-card/40 rounded-xl border border-border/30 divide-y divide-border/20">
                {filteredArticles.map((article) => (
                  <ArticleRow key={article.article_id} article={article} />
                ))}
              </div>

              {hasNextPage && !searchFilter && (
                <div className="flex justify-center mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    className="text-xs h-8 gap-1.5 px-6"
                    onClick={handleLoadMore}
                    disabled={isFetchingNextPage}
                  >
                    {isFetchingNextPage ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      <>
                        <ChevronRight className="w-3.5 h-3.5" />
                        Load More ({articlesTotalCount - articlesLoadedCount} remaining)
                      </>
                    )}
                  </Button>
                </div>
              )}
            </>
          )}
        </>
      )}

      <NewsWorkflowSettingsFlyout
        isOpen={workflowSettingsOpen}
        onClose={() => setWorkflowSettingsOpen(false)}
      />
    </div>
  )
}
