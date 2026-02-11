import { useState, useMemo, useEffect, useCallback } from 'react'
import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Newspaper,
  RefreshCw,
  ExternalLink,
  Zap,
  Globe,
  ChevronDown,
  ChevronUp,
  Play,
  Pause,
  Brain,
  Shield,
  Target,
  Settings,
  Search,
  ArrowUpRight,
  ArrowDownRight,
  Trash2,
  Radio,
  Layers,
  Eye,
  Filter,
  Loader2,
  ChevronRight,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  getNewsFeedStatus,
  getNewsArticles,
  triggerNewsFetch,
  clearNewsArticles,
  getNewsEdgesCached,
  detectNewsEdges,
  analyzeNewsEdgeSingle,
  runNewsMatching,
  runForecastCommittee,
  getNewsWorkflowStatus,
  getNewsWorkflowFindings,
  getNewsWorkflowIntents,
  runNewsWorkflow,
  startNewsWorkflow,
  pauseNewsWorkflow,
  skipNewsWorkflowIntent,
  NewsArticle,
  NewsEdge,
  NewsMatch,
  ForecastResult,
  NewsWorkflowFinding,
  NewsTradeIntent,
} from '../services/api'
import NewsWorkflowSettingsFlyout from './NewsWorkflowSettingsFlyout'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'

// ─── Sub-view types ─────────────────────────────────────────
type SubView = 'edges' | 'feed' | 'matches' | 'workflow'

// ─── Helpers ────────────────────────────────────────────────

function timeAgo(dateStr: string | null | undefined): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const ms = date.getTime()
  if (isNaN(ms)) return ''
  const seconds = Math.floor((Date.now() - ms) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
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

// ─── Edge Card ──────────────────────────────────────────────

function EdgeCard({ edge, onForecast }: { edge: NewsEdge; onForecast: (edge: NewsEdge) => void }) {
  const [expanded, setExpanded] = useState(false)
  const isBuyYes = edge.direction === 'buy_yes'

  return (
    <Card className="overflow-hidden border-border/40 hover:border-border/80 hover:shadow-lg hover:shadow-black/20 transition-all group">
      {/* Accent bar */}
      <div className={cn('h-0.5', edge.edge_percent >= 15 ? 'bg-green-400' : edge.edge_percent >= 10 ? 'bg-yellow-400' : 'bg-orange-400')} />

      <div className="p-4">
        {/* Row 1: Direction badge + Edge % */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline" className={cn(
              "text-[10px] font-semibold gap-1",
              isBuyYes
                ? "bg-green-500/10 text-green-400 border-green-500/20"
                : "bg-red-500/10 text-red-400 border-red-500/20"
            )}>
              {isBuyYes ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
              {isBuyYes ? 'BUY YES' : 'BUY NO'}
            </Badge>
            <Badge variant="outline" className="text-[10px] bg-blue-500/10 text-blue-400 border-blue-500/20">
              <Globe className="w-3 h-3 mr-0.5" />
              {edge.article_source}
            </Badge>
          </div>
          <div className="text-right shrink-0">
            <span className={cn("text-lg font-bold font-data", edgeColor(edge.edge_percent))}>
              {edge.edge_percent.toFixed(1)}%
            </span>
            <span className="text-[10px] text-muted-foreground block">edge</span>
          </div>
        </div>

        {/* Row 2: Market question */}
        <p className="text-sm font-medium text-foreground line-clamp-2 mb-2">{edge.market_question}</p>

        {/* Row 3: Article title */}
        <div className="flex items-start gap-2 mb-3">
          <Newspaper className="w-3.5 h-3.5 text-orange-400 shrink-0 mt-0.5" />
          <a
            href={edge.article_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-muted-foreground hover:text-orange-400 transition-colors line-clamp-1"
          >
            {edge.article_title}
            <ExternalLink className="w-3 h-3 inline ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
        </div>

        {/* Row 4: Metrics */}
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="bg-muted/30 rounded-lg p-2 text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Market</div>
            <div className="text-sm font-data font-semibold text-foreground">{(edge.market_price * 100).toFixed(0)}¢</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-2 text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Model</div>
            <div className={cn("text-sm font-data font-semibold", edgeColor(edge.edge_percent))}>{(edge.model_probability * 100).toFixed(0)}¢</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-2 text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Conf</div>
            <div className={cn("text-sm font-data font-semibold", confidenceColor(edge.confidence))}>{(edge.confidence * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-2 text-center">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Sim</div>
            <div className="text-sm font-data font-semibold text-blue-400">{(edge.similarity * 100).toFixed(0)}%</div>
          </div>
        </div>

        {/* Row 5: Actions */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 text-xs h-7 gap-1 bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20 hover:text-purple-400"
            onClick={() => onForecast(edge)}
          >
            <Brain className="w-3 h-3" />
            Deep Analysis
          </Button>
          <a
            href={edge.article_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7 gap-1 bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20 hover:text-blue-400"
            >
              <ExternalLink className="w-3 h-3" />
              Source
            </Button>
          </a>
          <Button
            variant="ghost"
            size="sm"
            className="text-xs h-7 px-2 text-muted-foreground hover:text-foreground"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </Button>
        </div>

        {/* Expanded reasoning */}
        {expanded && (
          <div className="mt-3 pt-3 border-t border-border/30">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">AI Reasoning</div>
            <p className="text-xs text-muted-foreground leading-relaxed">{edge.reasoning}</p>
            <div className="mt-2 flex items-center gap-3 text-[10px] text-muted-foreground">
              <span className="font-data">ID: {edge.market_id.slice(0, 8)}...</span>
              <span className="font-data">{timeAgo(edge.estimated_at)}</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

// ─── Article Row ────────────────────────────────────────────

function ArticleRow({ article }: { article: NewsArticle }) {
  const timeStr = timeAgo(article.published) || timeAgo(article.fetched_at)
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
          {timeStr && <span className="text-[10px] text-muted-foreground font-data shrink-0">{timeStr}</span>}
        </div>
      </div>
      <div className="shrink-0 flex items-center gap-2">
        {article.has_embedding && (
          <Tooltip>
            <TooltipTrigger>
              <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
            </TooltipTrigger>
            <TooltipContent className="text-xs">Embedded</TooltipContent>
          </Tooltip>
        )}
      </div>
    </div>
  )
}

// ─── Match Row ──────────────────────────────────────────────

function MatchRow({ match, onAnalyze, isAnalyzing }: { match: NewsMatch; onAnalyze: (match: NewsMatch) => void; isAnalyzing: boolean }) {
  return (
    <div className="px-3 py-3 hover:bg-muted/30 rounded-lg transition-colors">
      <div className="flex items-start gap-3">
        <div className="shrink-0 mt-1">
          <div className={cn(
            'w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold font-data',
            match.similarity >= 0.7 ? 'bg-green-500/15 text-green-400' :
            match.similarity >= 0.5 ? 'bg-yellow-500/15 text-yellow-400' :
            'bg-orange-500/15 text-orange-400'
          )}>
            {(match.similarity * 100).toFixed(0)}
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground line-clamp-1">{match.market_question}</p>
          <div className="flex items-center gap-1.5 mt-1">
            <Newspaper className="w-3 h-3 text-orange-400" />
            <a
              href={match.article_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted-foreground hover:text-orange-400 transition-colors line-clamp-1"
            >
              {match.article_title}
            </a>
          </div>
          <div className="flex items-center gap-3 mt-1 text-[10px] text-muted-foreground">
            <span>{match.article_source}</span>
            <span className="font-data">Price: {(match.market_price * 100).toFixed(0)}¢</span>
            <Badge variant="outline" className="text-[9px] h-4 px-1.5 bg-blue-500/10 text-blue-400 border-blue-500/20">
              {match.match_method}
            </Badge>
          </div>
        </div>
        <div className="shrink-0 mt-1">
          <Button
            variant="outline"
            size="sm"
            className="text-xs h-7 gap-1 bg-green-500/10 text-green-400 border-green-500/20 hover:bg-green-500/20 hover:text-green-400"
            onClick={() => onAnalyze(match)}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            {isAnalyzing ? 'Analyzing...' : 'Analyze'}
          </Button>
        </div>
      </div>
    </div>
  )
}

// ─── Forecast Panel ─────────────────────────────────────────

function ForecastPanel({ result, onClose }: { result: ForecastResult; onClose: () => void }) {
  const isBuyYes = result.direction === 'buy_yes'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-2xl mx-4 bg-card border border-border rounded-xl shadow-2xl max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="p-5">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Brain className="w-5 h-5 text-purple-400" />
                <h3 className="text-lg font-semibold text-foreground">Forecaster Committee</h3>
              </div>
              <p className="text-sm text-muted-foreground line-clamp-2">{result.market_question}</p>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose} className="text-muted-foreground">
              &times;
            </Button>
          </div>

          {/* Verdict */}
          <div className={cn(
            "rounded-xl p-4 mb-4 border",
            isBuyYes ? "bg-green-500/5 border-green-500/20" : "bg-red-500/5 border-red-500/20"
          )}>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Committee Verdict</div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={cn(
                    "text-sm font-semibold gap-1 px-2.5 py-1",
                    isBuyYes ? "bg-green-500/10 text-green-400 border-green-500/20" : "bg-red-500/10 text-red-400 border-red-500/20"
                  )}>
                    {isBuyYes ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                    {isBuyYes ? 'BUY YES' : 'BUY NO'}
                  </Badge>
                  <span className={cn("text-2xl font-bold font-data", edgeColor(result.edge_percent))}>
                    {result.edge_percent.toFixed(1)}%
                  </span>
                  <span className="text-sm text-muted-foreground">edge</span>
                </div>
              </div>
              <div className="text-right">
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  <span className="text-muted-foreground">Market:</span>
                  <span className="font-data font-semibold">{(result.market_price * 100).toFixed(0)}¢</span>
                  <span className="text-muted-foreground">Model:</span>
                  <span className={cn("font-data font-semibold", edgeColor(result.edge_percent))}>{(result.final_probability * 100).toFixed(0)}¢</span>
                  <span className="text-muted-foreground">Confidence:</span>
                  <span className={cn("font-data font-semibold", confidenceColor(result.confidence))}>{(result.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Agent Estimates */}
          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Agent Breakdown</div>
          <div className="space-y-3 mb-4">
            {result.agents.map((agent, i) => (
              <div key={i} className="bg-muted/20 rounded-lg p-3 border border-border/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {agent.name.includes('Outside') ? <Globe className="w-3.5 h-3.5 text-blue-400" /> :
                     agent.name.includes('Inside') ? <Eye className="w-3.5 h-3.5 text-green-400" /> :
                     <Shield className="w-3.5 h-3.5 text-red-400" />}
                    <span className="text-sm font-medium text-foreground">{agent.name}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs font-data">
                    <span className={cn("font-semibold", edgeColor(Math.abs(agent.probability - result.market_price) * 100))}>
                      {(agent.probability * 100).toFixed(0)}%
                    </span>
                    <span className={cn(confidenceColor(agent.confidence))}>
                      {(agent.confidence * 100).toFixed(0)}% conf
                    </span>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{agent.reasoning}</p>
                <div className="mt-1.5 text-[10px] text-muted-foreground/60 font-data">{agent.model}</div>
              </div>
            ))}
          </div>

          {/* Meta */}
          <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-3 border-t border-border/30">
            <span>Method: {result.aggregation_method}</span>
            <span className="font-data">{new Date(result.analyzed_at).toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Finding Card ───────────────────────────────────────────

function FindingCard({ finding }: { finding: NewsWorkflowFinding }) {
  const [expanded, setExpanded] = useState(false)
  const isBuyYes = finding.direction === 'buy_yes'
  const eventGraph = finding.event_graph as Record<string, unknown> | null
  const eventType = typeof eventGraph?.event_type === 'string' ? eventGraph.event_type : null
  const eventAction = typeof eventGraph?.action === 'string' ? eventGraph.action : null
  const eventRegion = typeof eventGraph?.region === 'string' ? eventGraph.region : null
  const eventActors = Array.isArray(eventGraph?.actors)
    ? eventGraph.actors.filter((actor): actor is string => typeof actor === 'string')
    : []
  const eventEntities = Array.isArray(eventGraph?.key_entities)
    ? eventGraph.key_entities.filter((entity): entity is string => typeof entity === 'string')
    : []

  return (
    <Card className="overflow-hidden border-border/40 hover:border-border/80 hover:shadow-lg hover:shadow-black/20 transition-all group">
      <div className={cn('h-0.5', finding.actionable ? (finding.edge_percent >= 15 ? 'bg-green-400' : 'bg-yellow-400') : 'bg-muted')} />
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline" className={cn(
              "text-[10px] font-semibold gap-1",
              isBuyYes ? "bg-green-500/10 text-green-400 border-green-500/20" : "bg-red-500/10 text-red-400 border-red-500/20"
            )}>
              {isBuyYes ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
              {isBuyYes ? 'BUY YES' : 'BUY NO'}
            </Badge>
            {finding.actionable && (
              <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-400 border-green-500/20">Actionable</Badge>
            )}
            {eventType && (
              <Badge variant="outline" className="text-[10px] bg-violet-500/10 text-violet-400 border-violet-500/20">
                {eventType.replace('_', ' ')}
              </Badge>
            )}
          </div>
          <div className="text-right shrink-0">
            <span className={cn("text-lg font-bold font-data", edgeColor(finding.edge_percent))}>
              {finding.edge_percent.toFixed(1)}%
            </span>
            <span className="text-[10px] text-muted-foreground block">edge</span>
          </div>
        </div>

        <p className="text-sm font-medium text-foreground line-clamp-2 mb-2">{finding.market_question}</p>

        <div className="flex items-start gap-2 mb-3">
          <Newspaper className="w-3.5 h-3.5 text-orange-400 shrink-0 mt-0.5" />
          <a href={finding.article_url} target="_blank" rel="noopener noreferrer"
            className="text-xs text-muted-foreground hover:text-orange-400 transition-colors line-clamp-1">
            {finding.article_title}
            <ExternalLink className="w-3 h-3 inline ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
        </div>

        <div className="grid grid-cols-5 gap-1.5 mb-3">
          <div className="bg-muted/30 rounded-lg p-1.5 text-center">
            <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Mkt</div>
            <div className="text-xs font-data font-semibold">{(finding.market_price * 100).toFixed(0)}c</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-1.5 text-center">
            <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Model</div>
            <div className={cn("text-xs font-data font-semibold", edgeColor(finding.edge_percent))}>{(finding.model_probability * 100).toFixed(0)}c</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-1.5 text-center">
            <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Conf</div>
            <div className={cn("text-xs font-data font-semibold", confidenceColor(finding.confidence))}>{(finding.confidence * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-1.5 text-center">
            <div className="text-[8px] text-muted-foreground uppercase tracking-wider">KW</div>
            <div className="text-xs font-data font-semibold text-blue-400">{(finding.keyword_score * 100).toFixed(0)}</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-1.5 text-center">
            <div className="text-[8px] text-muted-foreground uppercase tracking-wider">Sem</div>
            <div className="text-xs font-data font-semibold text-blue-400">{(finding.semantic_score * 100).toFixed(0)}</div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <a href={finding.article_url} target="_blank" rel="noopener noreferrer">
            <Button variant="outline" size="sm" className="text-xs h-7 gap-1 bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20 hover:text-blue-400">
              <ExternalLink className="w-3 h-3" /> Source
            </Button>
          </a>
          <Button variant="ghost" size="sm" className="text-xs h-7 px-2 text-muted-foreground hover:text-foreground"
            onClick={() => setExpanded(!expanded)}>
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </Button>
        </div>

        {expanded && (
          <div className="mt-3 pt-3 border-t border-border/30 space-y-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">AI Reasoning</div>
            <p className="text-xs text-muted-foreground leading-relaxed">{finding.reasoning}</p>
            {eventGraph && (
              <div>
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 mt-2">Event Graph</div>
                <div className="text-[10px] text-muted-foreground/80 font-mono bg-muted/20 p-2 rounded-lg">
                  {eventActors.length > 0 && (
                    <div>Actors: {eventActors.join(', ')}</div>
                  )}
                  {eventAction && <div>Action: {eventAction}</div>}
                  {eventEntities.length > 0 && (
                    <div>Entities: {eventEntities.join(', ')}</div>
                  )}
                  {eventRegion && <div>Region: {eventRegion}</div>}
                </div>
              </div>
            )}
            <div className="mt-2 flex items-center gap-3 text-[10px] text-muted-foreground">
              <span className="font-data">Rerank: {(finding.rerank_score * 100).toFixed(0)}%</span>
              <span className="font-data">{timeAgo(finding.created_at)}</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

// ─── Intent Row ─────────────────────────────────────────────

function IntentRow({ intent, onSkip, isSkipping }: { intent: NewsTradeIntent; onSkip: (id: string) => void; isSkipping: boolean }) {
  const isBuyYes = intent.direction === 'buy_yes'
  const statusColors: Record<string, string> = {
    pending: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    submitted: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    executed: 'bg-green-500/10 text-green-400 border-green-500/20',
    skipped: 'bg-muted/50 text-muted-foreground border-border',
    expired: 'bg-red-500/10 text-red-400 border-red-500/20',
  }

  return (
    <div className="px-3 py-3 hover:bg-muted/30 rounded-lg transition-colors">
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground line-clamp-1">{intent.market_question}</p>
          <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
            <Badge variant="outline" className={cn("text-[9px] h-4 px-1.5", isBuyYes ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20')}>
              {isBuyYes ? 'BUY YES' : 'BUY NO'}
            </Badge>
            <span className="font-data">Edge: {intent.edge_percent.toFixed(1)}%</span>
            <span className="font-data">Size: ${intent.suggested_size_usd.toFixed(0)}</span>
            <Badge variant="outline" className={cn("text-[9px] h-4 px-1.5", statusColors[intent.status] || '')}>
              {intent.status}
            </Badge>
            <span className="font-data">{timeAgo(intent.created_at)}</span>
          </div>
        </div>
        {intent.status === 'pending' && (
          <Button variant="outline" size="sm" className="text-xs h-7 gap-1 text-muted-foreground hover:text-foreground"
            onClick={() => onSkip(intent.id)} disabled={isSkipping}>
            <X className="w-3 h-3" /> Skip
          </Button>
        )}
      </div>
    </div>
  )
}

// ─── Main Component ─────────────────────────────────────────

interface NewsIntelligencePanelProps {
  initialSearchQuery?: string
}

export default function NewsIntelligencePanel({ initialSearchQuery }: NewsIntelligencePanelProps = {}) {
  const queryClient = useQueryClient()
  const [subView, setSubView] = useState<SubView>(initialSearchQuery ? 'feed' : 'workflow')
  const [searchFilter, setSearchFilter] = useState(initialSearchQuery || '')
  const [edgeSortBy, setEdgeSortBy] = useState<'edge' | 'confidence' | 'similarity'>('edge')
  const [feedSourceFilter, setFeedSourceFilter] = useState<string | null>(null)

  // Sync initialSearchQuery prop changes (useState only uses it on mount)
  useEffect(() => {
    if (initialSearchQuery !== undefined) {
      setSearchFilter(initialSearchQuery)
      if (initialSearchQuery) setSubView('feed')
    }
  }, [initialSearchQuery])
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null)
  const [, setForecastingEdge] = useState<string | null>(null)
  const [analyzingMatchId, setAnalyzingMatchId] = useState<string | null>(null)
  const [workflowSettingsOpen, setWorkflowSettingsOpen] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  // ─── Workflow Queries ─────────────────────────────────────

  const { data: workflowStatus } = useQuery({
    queryKey: ['news-workflow-status'],
    queryFn: getNewsWorkflowStatus,
    refetchInterval: 30000,
  })

  const { data: workflowFindingsData, isLoading: findingsLoading } = useQuery({
    queryKey: ['news-workflow-findings'],
    queryFn: () => getNewsWorkflowFindings({ actionable_only: false, max_age_hours: 24, limit: 100 }),
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

  // ─── Queries ──────────────────────────────────────────────

  // WS pushes `news_update` when new articles arrive — polling is fallback only
  const { data: feedStatus } = useQuery({
    queryKey: ['news-feed-status'],
    queryFn: getNewsFeedStatus,
    refetchInterval: 60000, // fallback (WS push refreshes instantly)
  })

  // Paginated article fetching — load 100 at a time with "Load More"
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

  const { data: edgesData, isLoading: edgesLoading } = useQuery({
    queryKey: ['news-edges'],
    queryFn: getNewsEdgesCached,
    refetchInterval: 30000,
    staleTime: 10000,
  })

  // Matches are NOT auto-polled — they require embedding + ML work.
  // Users trigger via "Run Matching" button; results are cached in state.
  const [matchesData, setMatchesData] = useState<{
    total_articles?: number
    total_markets?: number
    total_matches?: number
    matcher_mode?: string
    matches: NewsMatch[]
  } | null>(null)
  const [matchesLoading, setMatchesLoading] = useState(false)

  // ─── Mutations ────────────────────────────────────────────

  const fetchMutation = useMutation({
    mutationFn: triggerNewsFetch,
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
    },
  })

  // Manual matching trigger (avoids infinite spinner)
  // Use 168h to match ALL articles in the feed, and higher top_k for more matches
  const runMatchingMutation = useMutation({
    mutationFn: () => runNewsMatching({ max_age_hours: 168, top_k: 10 }),
    onMutate: () => setMatchesLoading(true),
    onSuccess: (data) => {
      setMatchesData(data)
      setMatchesLoading(false)
    },
    onError: () => setMatchesLoading(false),
  })

  const clearMutation = useMutation({
    mutationFn: clearNewsArticles,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-articles'] })
      queryClient.removeQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
      setMatchesData(null)
      setSearchFilter('')
      setFeedSourceFilter(null)
    },
  })

  const edgeRefreshMutation = useMutation({
    mutationFn: () => detectNewsEdges({ max_age_hours: 168, top_k: 10 }),
    onSuccess: (data) => {
      queryClient.setQueryData(['news-edges'], data)
    },
  })

  const singleAnalyzeMutation = useMutation({
    mutationFn: (match: NewsMatch) => analyzeNewsEdgeSingle({
      article_id: match.article_id,
      market_id: match.market_id,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['news-edges'] })
      setAnalyzingMatchId(null)
    },
    onError: () => {
      setAnalyzingMatchId(null)
    },
  })

  const handleAnalyzeMatch = (match: NewsMatch) => {
    setAnalyzingMatchId(`${match.article_id}:${match.market_id}`)
    singleAnalyzeMutation.mutate(match)
  }

  const forecastMutation = useMutation({
    mutationFn: (edge: NewsEdge) => runForecastCommittee({
      market_question: edge.market_question,
      market_price: edge.market_price,
      news_context: `Article: ${edge.article_title}\nSource: ${edge.article_source}\nReasoning: ${edge.reasoning}`,
    }),
    onSuccess: (data) => {
      setForecastResult(data)
      setForecastingEdge(null)
    },
    onError: () => {
      setForecastingEdge(null)
    },
  })

  const handleForecast = (edge: NewsEdge) => {
    setForecastingEdge(edge.market_id)
    forecastMutation.mutate(edge)
  }

  // ─── Derived data ─────────────────────────────────────────

  // Flatten paginated articles
  const articles = useMemo(
    () => articlesPages?.pages.flatMap(p => p.articles) || [],
    [articlesPages],
  )
  const articlesTotalCount = articlesPages?.pages[0]?.total ?? 0
  const articlesLoadedCount = articles.length

  const edges = edgesData?.edges || []
  const matches = matchesData?.matches || []

  const filteredEdges = useMemo(() => {
    let result = [...edges]
    if (searchFilter) {
      const q = searchFilter.toLowerCase()
      result = result.filter(e =>
        e.market_question.toLowerCase().includes(q) ||
        e.article_title.toLowerCase().includes(q) ||
        e.article_source.toLowerCase().includes(q)
      )
    }
    result.sort((a, b) => {
      if (edgeSortBy === 'edge') return b.edge_percent - a.edge_percent
      if (edgeSortBy === 'confidence') return b.confidence - a.confidence
      return b.similarity - a.similarity
    })
    return result
  }, [edges, searchFilter, edgeSortBy])

  const filteredArticles = useMemo(() => {
    if (!searchFilter) return articles
    const q = searchFilter.toLowerCase()
    return articles.filter(a =>
      a.title.toLowerCase().includes(q) ||
      a.source.toLowerCase().includes(q) ||
      a.category.toLowerCase().includes(q)
    )
  }, [articles, searchFilter])

  const filteredMatches = useMemo(() => {
    if (!searchFilter) return matches
    const q = searchFilter.toLowerCase()
    return matches.filter(m =>
      m.market_question.toLowerCase().includes(q) ||
      m.article_title.toLowerCase().includes(q)
    )
  }, [matches, searchFilter])

  // ─── Source breakdown ─────────────────────────────────────

  const sourceBreakdown = feedStatus?.sources || {}
  const sourceKeys = useMemo(() => Object.keys(sourceBreakdown), [sourceBreakdown])

  // Handle "Load More" for infinite scroll
  const handleLoadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) fetchNextPage()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])

  return (
    <div className="max-w-[1600px] mx-auto">
      {/* ─── Header ──────────────────────────────────────────── */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-8 h-8 rounded-lg bg-orange-500/15 flex items-center justify-center border border-orange-500/20">
              <Newspaper className="w-4.5 h-4.5 text-orange-400" />
            </div>
            <h2 className="text-lg font-bold text-foreground">News Intelligence</h2>
            {feedStatus?.running && (
              <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-400 border-green-500/20 gap-1">
                <Radio className="w-3 h-3" />
                Live
              </Badge>
            )}
          </div>
          <p className="text-xs text-muted-foreground ml-[42px]">
            Detect informational edges where breaking news diverges from market prices
          </p>
        </div>

        <Badge
          variant="outline"
          className={cn(
            'text-[10px] h-6 px-2.5',
            workflowStatus?.paused
              ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
              : workflowStatus?.enabled
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-muted/40 text-muted-foreground border-border'
          )}
        >
          Autopilot {workflowStatus?.paused ? 'Paused' : workflowStatus?.enabled ? 'Active' : 'Disabled'}
        </Badge>
      </div>

      {/* ─── Status Bar ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <div className="bg-card/60 rounded-xl border border-border/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <Newspaper className="w-3.5 h-3.5 text-orange-400" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Articles</span>
          </div>
          <div className="text-xl font-bold font-data text-foreground">{feedStatus?.article_count ?? 0}</div>
          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
            {Object.entries(sourceBreakdown).map(([src, count]) => (
              <Badge key={src} variant="outline" className={cn("text-[9px] h-4 px-1.5", SOURCE_COLORS[src] || 'bg-muted/50 text-muted-foreground border-border')}>
                {src.replace('_', ' ')}: {count as number}
              </Badge>
            ))}
          </div>
        </div>

        <div className="bg-card/60 rounded-xl border border-border/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Matches</span>
          </div>
          <div className="text-xl font-bold font-data text-blue-400">{matchesData?.total_matches ?? 0}</div>
          <div className="text-[10px] text-muted-foreground mt-1">
            {matchesData?.matcher_mode === 'semantic' ? 'Semantic (ML)' : 'TF-IDF'} matching
          </div>
        </div>

        <div className="bg-card/60 rounded-xl border border-border/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-3.5 h-3.5 text-green-400" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Edges Found</span>
          </div>
          <div className="text-xl font-bold font-data text-green-400">{edgesData?.total_edges ?? 0}</div>
          <div className="text-[10px] text-muted-foreground mt-1">
            {edges.filter(e => e.edge_percent >= 15).length} high-conviction
          </div>
        </div>

        <div className="bg-card/60 rounded-xl border border-border/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Edge</span>
          </div>
          <div className={cn("text-xl font-bold font-data", edges.length > 0 ? 'text-green-400' : 'text-muted-foreground')}>
            {edges.length > 0 ? `${(edges.reduce((s, e) => s + e.edge_percent, 0) / edges.length).toFixed(1)}%` : '—'}
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">
            Avg confidence: {edges.length > 0 ? `${(edges.reduce((s, e) => s + e.confidence, 0) / edges.length * 100).toFixed(0)}%` : '—'}
          </div>
        </div>
      </div>

      {/* ─── Sub-view Tabs + Search ──────────────────────────── */}
      <div className="flex items-center gap-2 mb-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSubView('edges')}
          className={cn(
            "gap-1.5 text-xs h-8",
            subView === 'edges'
              ? "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30 hover:text-green-400"
              : "bg-card text-muted-foreground hover:text-foreground border-border"
          )}
        >
          <Zap className="w-3.5 h-3.5" />
          Edges
          {edges.length > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-green-500/15 text-green-400 text-[10px] font-data">{edges.length}</span>
          )}
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
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSubView('matches')}
          className={cn(
            "gap-1.5 text-xs h-8",
            subView === 'matches'
              ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400"
              : "bg-card text-muted-foreground hover:text-foreground border-border"
          )}
        >
          <Layers className="w-3.5 h-3.5" />
          Matches
          {matches.length > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-blue-500/15 text-blue-400 text-[10px] font-data">{matches.length}</span>
          )}
        </Button>
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
          {(Number(workflowStatus?.stats?.findings ?? 0) ?? 0) > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-purple-500/15 text-purple-400 text-[10px] font-data">
              {Number(workflowStatus?.stats?.findings ?? 0)}
            </span>
          )}
        </Button>

        <div className="ml-auto flex items-center gap-2">
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

          {subView === 'edges' && (
            <div className="flex items-center gap-0.5 border border-border/50 rounded-lg p-0.5 bg-card/50">
              {(['edge', 'confidence', 'similarity'] as const).map(key => (
                <button
                  key={key}
                  onClick={() => setEdgeSortBy(key)}
                  className={cn(
                    'px-2 py-1 rounded-md text-[10px] font-medium transition-colors',
                    edgeSortBy === key ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  )}
                >
                  {key === 'edge' ? 'Edge %' : key === 'confidence' ? 'Conf' : 'Sim'}
                </button>
              ))}
            </div>
          )}

          {subView === 'feed' && (
            <Button
              variant="ghost"
              size="sm"
              className="text-xs h-7 gap-1 text-red-400 hover:text-red-400 hover:bg-red-500/10"
              onClick={() => clearMutation.mutate()}
              disabled={clearMutation.isPending}
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* ─── Edges View ──────────────────────────────────────── */}
      {subView === 'edges' && (
        <>
          {edgesLoading || edgeRefreshMutation.isPending ? (
            <div className="flex flex-col items-center justify-center py-16">
              <RefreshCw className="w-8 h-8 animate-spin text-green-400 mb-3" />
              <p className="text-sm text-muted-foreground">
                {edgeRefreshMutation.isPending ? 'Analyzing matches with LLM...' : 'Loading cached edges...'}
              </p>
              {edgeRefreshMutation.isPending && (
                <p className="text-xs text-muted-foreground/60 mt-1">Estimating probabilities via configured AI provider...</p>
              )}
            </div>
          ) : filteredEdges.length === 0 ? (
            <div className="text-center py-16">
              <Zap className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {searchFilter ? 'No edges match your filter' : 'No news edges analyzed yet'}
              </p>
              {!searchFilter ? (
                <>
                  <p className="text-sm text-muted-foreground/70 mt-1 mb-4">
                    Edge detection uses AI to estimate where news creates market mispricings.
                    <br />
                    Run matching first, then analyze matches — or click "Analyze All" for the full pipeline.
                  </p>
                  <div className="flex items-center justify-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-8 gap-1.5 bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20 hover:text-purple-400"
                      onClick={() => edgeRefreshMutation.mutate()}
                      disabled={edgeRefreshMutation.isPending}
                    >
                      <Zap className="w-3.5 h-3.5" />
                      Analyze All (Full Pipeline)
                    </Button>
                  </div>
                </>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1 mt-3"
                  onClick={() => setSearchFilter('')}
                >
                  <X className="w-3 h-3" />
                  Clear Filter
                </Button>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
              {filteredEdges.map((edge, i) => (
                <EdgeCard
                  key={`${edge.market_id}-${i}`}
                  edge={edge}
                  onForecast={handleForecast}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* ─── Feed View ───────────────────────────────────────── */}
      {subView === 'feed' && (
        <>
          {/* Source filter tabs */}
          {sourceKeys.length > 0 && (
            <div className="flex items-center gap-1.5 mb-3">
              <Filter className="w-3 h-3 text-muted-foreground" />
              <button
                onClick={() => setFeedSourceFilter(null)}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors',
                  !feedSourceFilter ? 'bg-orange-500/20 text-orange-400' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                )}
              >
                All ({feedStatus?.article_count ?? 0})
              </button>
              {sourceKeys.map(src => (
                <button
                  key={src}
                  onClick={() => setFeedSourceFilter(feedSourceFilter === src ? null : src)}
                  className={cn(
                    'px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors',
                    feedSourceFilter === src ? 'bg-orange-500/20 text-orange-400' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  )}
                >
                  {src.replace('_', ' ')} ({sourceBreakdown[src] as number})
                </button>
              ))}
              {feedSourceFilter && (
                <button
                  onClick={() => setFeedSourceFilter(null)}
                  className="text-muted-foreground hover:text-foreground transition-colors ml-1"
                  title="Clear filter"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          )}

          {/* Articles list */}
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
              <p className="text-sm text-muted-foreground/70 mt-1">
                {searchFilter
                  ? 'Try a different search term or clear the filter'
                  : 'Click "Fetch News" to ingest articles from Google News, GDELT, and custom RSS feeds'}
              </p>
              {searchFilter && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1 mt-3"
                  onClick={() => setSearchFilter('')}
                >
                  <X className="w-3 h-3" />
                  Clear Filter
                </Button>
              )}
            </div>
          ) : (
            <>
              {/* Count summary */}
              <div className="flex items-center justify-between mb-2 text-xs text-muted-foreground">
                <span>
                  Showing {filteredArticles.length}
                  {searchFilter ? ` of ${articlesLoadedCount} loaded` : ''}
                  {articlesTotalCount > articlesLoadedCount ? ` (${articlesTotalCount} total)` : ''}
                </span>
                {articles.length > 0 && articles[0].published && (
                  <span className="font-data">Latest: {timeAgo(articles[0].published) || timeAgo(articles[0].fetched_at)}</span>
                )}
              </div>

              <div className="bg-card/40 rounded-xl border border-border/30 divide-y divide-border/20">
                {filteredArticles.map((article) => (
                  <ArticleRow key={article.article_id} article={article} />
                ))}
              </div>

              {/* Load More button */}
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

              {/* All loaded indicator */}
              {!hasNextPage && articlesLoadedCount > PAGE_SIZE && !searchFilter && (
                <div className="text-center text-xs text-muted-foreground/60 mt-3">
                  All {articlesTotalCount} articles loaded
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ─── Matches View ────────────────────────────────────── */}
      {subView === 'matches' && (
        <>
          {matchesLoading || runMatchingMutation.isPending ? (
            <div className="flex flex-col items-center justify-center py-16">
              <RefreshCw className="w-8 h-8 animate-spin text-blue-400 mb-3" />
              <p className="text-sm text-muted-foreground">Running semantic matching...</p>
              <p className="text-xs text-muted-foreground/60 mt-1">Embedding articles and matching to markets (this may take a moment)</p>
            </div>
          ) : filteredMatches.length === 0 ? (
            <div className="text-center py-16">
              <Layers className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {searchFilter ? 'No matches for your filter' : 'No article-market matches yet'}
              </p>
              {!searchFilter && (
                <>
                  <p className="text-sm text-muted-foreground/70 mt-1 mb-1">
                    Matching embeds all articles and finds semantic matches to active markets.
                  </p>
                  <p className="text-xs text-muted-foreground/50 mb-4">
                    {feedStatus?.article_count
                      ? `${feedStatus.article_count} articles available • `
                      : 'No articles yet — fetch news first • '}
                    {feedStatus?.matcher?.markets_indexed
                      ? `${feedStatus.matcher.markets_indexed} markets indexed`
                      : 'Markets indexed on first run'}
                  </p>
                  <div className="flex items-center justify-center gap-2">
                    {!feedStatus?.article_count && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="text-xs h-8 gap-1.5 bg-orange-500/10 text-orange-400 border-orange-500/20 hover:bg-orange-500/20 hover:text-orange-400"
                        onClick={() => fetchMutation.mutate()}
                        disabled={fetchMutation.isPending}
                      >
                        {fetchMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Newspaper className="w-3.5 h-3.5" />}
                        Fetch News First
                      </Button>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-8 gap-1.5 bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20 hover:text-blue-400"
                      onClick={() => runMatchingMutation.mutate()}
                      disabled={runMatchingMutation.isPending || !feedStatus?.article_count}
                    >
                      <Layers className="w-3.5 h-3.5" />
                      Run Matching
                    </Button>
                  </div>
                </>
              )}
              {searchFilter && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1 mt-3"
                  onClick={() => setSearchFilter('')}
                >
                  <X className="w-3 h-3" />
                  Clear Filter
                </Button>
              )}
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>{matchesData?.total_articles ?? 0} articles</span>
                  <span className="text-border">|</span>
                  <span>{matchesData?.total_markets ?? 0} markets</span>
                  <span className="text-border">|</span>
                  <span>{filteredMatches.length} matches</span>
                  <span className="text-border">|</span>
                  <Badge variant="outline" className="text-[9px] h-4 px-1.5 bg-blue-500/10 text-blue-400 border-blue-500/20">
                    {matchesData?.matcher_mode === 'semantic' ? 'Semantic (ML)' : 'TF-IDF'}
                  </Badge>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1 text-blue-400 border-blue-500/20 hover:bg-blue-500/10 hover:text-blue-400"
                  onClick={() => runMatchingMutation.mutate()}
                  disabled={runMatchingMutation.isPending}
                >
                  <RefreshCw className={cn("w-3 h-3", runMatchingMutation.isPending && "animate-spin")} />
                  Re-run
                </Button>
              </div>
              <div className="bg-card/40 rounded-xl border border-border/30 divide-y divide-border/20">
                {filteredMatches.map((match, i) => (
                  <MatchRow
                    key={`${match.market_id}-${i}`}
                    match={match}
                    onAnalyze={handleAnalyzeMatch}
                    isAnalyzing={analyzingMatchId === `${match.article_id}:${match.market_id}`}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ─── Workflow View ─────────────────────────────────────── */}
      {subView === 'workflow' && (
        <>
          <div className="space-y-3 mb-4">
            <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-card/40 rounded-xl border border-border/30">
              <div className="flex items-center gap-2 text-xs text-muted-foreground min-w-0">
                <div className={cn('w-2 h-2 rounded-full', workflowStatus?.running ? 'bg-green-400 animate-pulse' : 'bg-muted')} />
                <span className="truncate">
                  {workflowStatus?.current_activity || 'Waiting for news worker'}
                </span>
                {workflowStatus?.last_error && (
                  <Badge variant="outline" className="text-[9px] bg-red-500/10 text-red-400 border-red-500/20">
                    Error
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1.5"
                  onClick={() => (workflowStatus?.paused ? startWorkflowMutation.mutate() : pauseWorkflowMutation.mutate())}
                  disabled={startWorkflowMutation.isPending || pauseWorkflowMutation.isPending}
                >
                  {workflowStatus?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                  {workflowStatus?.paused ? 'Resume' : 'Pause'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1.5"
                  onClick={() => setWorkflowSettingsOpen(true)}
                >
                  <Settings className="w-3.5 h-3.5" />
                  Settings
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-5 gap-2.5">
              <Card className="border-border/40 bg-card/40 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Last Cycle</p>
                <p className="text-sm font-data font-semibold text-foreground mt-0.5">
                  {timeAgo(workflowStatus?.last_scan)}
                </p>
              </Card>
              <Card className="border-border/40 bg-card/40 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Next Cycle</p>
                <p className="text-sm font-data font-semibold text-blue-400 mt-0.5">
                  {workflowStatus?.next_scan ? timeAgo(workflowStatus.next_scan) : 'N/A'}
                </p>
              </Card>
              <Card className="border-border/40 bg-card/40 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Pending Intents</p>
                <p className="text-sm font-data font-semibold text-yellow-400 mt-0.5">
                  {workflowStatus?.pending_intents ?? 0}
                </p>
              </Card>
              <Card className="border-border/40 bg-card/40 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Budget Skips</p>
                <p className="text-sm font-data font-semibold text-orange-400 mt-0.5">
                  {Number(workflowStatus?.stats?.budget_skip_count ?? workflowStatus?.stats?.llm_calls_skipped ?? 0)}
                </p>
              </Card>
              <Card className="border-border/40 bg-card/40 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Mode</p>
                <p className={cn(
                  'text-sm font-data font-semibold mt-0.5',
                  workflowStatus?.degraded_mode ? 'text-amber-400' : 'text-emerald-400'
                )}>
                  {workflowStatus?.degraded_mode ? 'Degraded' : 'Normal'}
                </p>
              </Card>
            </div>

            <Card className="border-border/40 bg-card/30 overflow-hidden">
              <button
                className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => setAdvancedOpen((v) => !v)}
              >
                <span className="uppercase tracking-wider">Advanced / Debug</span>
                {advancedOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
              {advancedOpen && (
                <div className="px-3 pb-3 pt-1 border-t border-border/30 space-y-2">
                  <p className="text-[10px] text-muted-foreground">
                    Manual controls for diagnostics only. Autopilot worker handles normal operation.
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-7 gap-1 text-purple-400 border-purple-500/20 hover:bg-purple-500/10 hover:text-purple-400"
                      onClick={() => runWorkflowMutation.mutate()}
                      disabled={runWorkflowMutation.isPending}
                    >
                      {runWorkflowMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                      Queue Scan
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-7 gap-1"
                      onClick={() => fetchMutation.mutate()}
                      disabled={fetchMutation.isPending}
                    >
                      {fetchMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                      Fetch News
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-7 gap-1 text-blue-400 border-blue-500/20 hover:bg-blue-500/10 hover:text-blue-400"
                      onClick={() => runMatchingMutation.mutate()}
                      disabled={runMatchingMutation.isPending}
                    >
                      {runMatchingMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Layers className="w-3 h-3" />}
                      Run Matching
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-7 gap-1 text-purple-400 border-purple-500/20 hover:bg-purple-500/10 hover:text-purple-400"
                      onClick={() => edgeRefreshMutation.mutate()}
                      disabled={edgeRefreshMutation.isPending}
                    >
                      {edgeRefreshMutation.isPending ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                      Analyze All
                    </Button>
                  </div>
                </div>
              )}
            </Card>
          </div>

          {/* Trade Intents */}
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

          {/* Findings */}
          {findingsLoading || runWorkflowMutation.isPending ? (
            <div className="flex flex-col items-center justify-center py-16">
              <RefreshCw className="w-8 h-8 animate-spin text-purple-400 mb-3" />
              <p className="text-sm text-muted-foreground">
                {runWorkflowMutation.isPending ? 'Queueing workflow cycle...' : 'Loading findings...'}
              </p>
              {runWorkflowMutation.isPending && (
                <p className="text-xs text-muted-foreground/60 mt-1">
                  Worker will pick up the request on the next cycle.
                </p>
              )}
            </div>
          ) : (workflowFindingsData?.findings?.length ?? 0) === 0 ? (
            <div className="text-center py-16">
              <Target className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">No workflow findings yet</p>
              <p className="text-sm text-muted-foreground/70 mt-1 mb-4">
                Autopilot runs continuously in the background. Use Advanced / Debug only for manual diagnostics.
              </p>
            </div>
          ) : (
            <>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Zap className="w-3 h-3 text-green-400" />
                Findings ({workflowFindingsData?.total ?? 0})
                {workflowFindingsData?.findings.filter(f => f.actionable).length
                  ? ` / ${workflowFindingsData.findings.filter(f => f.actionable).length} actionable`
                  : ''}
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
                {workflowFindingsData?.findings.map((finding) => (
                  <FindingCard key={finding.id} finding={finding} />
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ─── Workflow Settings Flyout ─────────────────────────── */}
      <NewsWorkflowSettingsFlyout
        isOpen={workflowSettingsOpen}
        onClose={() => setWorkflowSettingsOpen(false)}
      />

      {/* ─── Forecast Modal ──────────────────────────────────── */}
      {forecastResult && (
        <ForecastPanel result={forecastResult} onClose={() => setForecastResult(null)} />
      )}

      {/* ─── Forecasting Overlay ─────────────────────────────── */}
      {forecastMutation.isPending && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-card border border-border rounded-xl p-8 text-center shadow-2xl">
            <Brain className="w-10 h-10 text-purple-400 mx-auto mb-3 animate-pulse" />
            <h3 className="text-lg font-semibold text-foreground mb-1">Running Forecaster Committee</h3>
            <p className="text-sm text-muted-foreground">
              Deploying Outside View, Inside View, and Adversarial Critic agents...
            </p>
            <div className="mt-4 flex items-center justify-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 rounded-full bg-green-400 animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 rounded-full bg-red-400 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
