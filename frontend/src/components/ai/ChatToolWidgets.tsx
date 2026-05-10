/**
 * Compact inline widgets for rendering agent tool results in chat.
 *
 * Each tool type gets a custom visual renderer — market cards, portfolio
 * tables, status badges, etc. Falls back to a compact JSON summary for
 * unknown tools.
 */

import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { cn } from '../../lib/utils'
import {
  TrendingUp,
  TrendingDown,
  Search,
  BarChart3,
  Wallet,
  Globe,
  Newspaper,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Wrench,
  Database,
  Brain,
  Activity,
  DollarSign,
  Target,
  Zap,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ToolWidgetProps {
  tool: string
  output: Record<string, unknown>
}

interface ToolStartProps {
  tool: string
  input: Record<string, unknown>
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

export function ThinkingIndicator({ text }: { text?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="flex items-center gap-2 py-1.5"
    >
      <div className="flex items-center gap-1">
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-purple-400"
            animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.1, 0.8] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
      {text && (
        <span className="text-xs text-muted-foreground/70 italic">{text}</span>
      )}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Tool start indicator
// ---------------------------------------------------------------------------

const TOOL_ICONS: Record<string, React.ElementType> = {
  search_markets: Search,
  get_market_details: BarChart3,
  get_live_prices: TrendingUp,
  get_price_history: Activity,
  check_orderbook: BarChart3,
  find_related_markets: Target,
  get_market_regime: Brain,
  get_active_markets_summary: Globe,
  get_open_positions: Wallet,
  get_open_orders: Wallet,
  get_trade_history: DollarSign,
  get_account_balance: Wallet,
  get_portfolio_performance: TrendingUp,
  place_order: Zap,
  cancel_order: XCircle,
  cancel_all_orders: XCircle,
  search_news: Newspaper,
  get_news_edges: Newspaper,
  analyze_market_sentiment: Brain,
  analyze_resolution: CheckCircle,
  web_search: Globe,
  fetch_webpage: Globe,
  get_system_status: Activity,
  get_settings: Wrench,
  update_setting: Wrench,
  query_database: Database,
  list_strategies: Target,
  get_strategy_details: Target,
  get_strategy_performance: TrendingUp,
  update_strategy_config: Wrench,
  validate_strategy_code: CheckCircle,
  run_strategy_backtest: Activity,
  create_strategy: Zap,
  get_wallet_profile: Wallet,
  get_wallet_leaderboard: TrendingUp,
  get_smart_pool_stats: BarChart3,
  get_confluence_signals: Zap,
  get_tracked_wallets: Wallet,
  get_trader_groups: Target,
  get_whale_clusters: TrendingUp,
  get_active_signals: Zap,
  get_recent_decisions: Brain,
  get_trader_overview: BarChart3,
  detect_anomalies: AlertTriangle,
  get_cross_platform_arb: DollarSign,
  get_llm_usage_stats: Activity,
  get_data_sources: Database,
  query_data_source: Database,
  cortex_recall: Brain,
  cortex_remember: Brain,
  cortex_expire_memory: Brain,
  cortex_get_fleet_status: Activity,
  cortex_pause_trader: AlertTriangle,
  cortex_enable_strategy: Target,
  cortex_update_risk_clamps: Wrench,
}

// Tool label keys live under `aiChatToolWidgets.toolLabels.<tool_name>`.
// Falls back to the prettified tool name when no translation is found.
function useToolLabel(tool: string): string {
  const { t } = useTranslation()
  return t(`aiChatToolWidgets.toolLabels.${tool}`, { defaultValue: tool.replace(/_/g, ' ') })
}

export function ToolStartWidget({ tool, input }: ToolStartProps) {
  const Icon = TOOL_ICONS[tool] || Wrench
  const label = useToolLabel(tool)
  const params = Object.entries(input || {})
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(', ')

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.2 }}
      className="flex items-center gap-2 py-1 px-2.5 rounded-lg bg-muted/30 border border-border/20 my-1"
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      >
        <Loader2 className="w-3 h-3 text-purple-400" />
      </motion.div>
      <Icon className="w-3 h-3 text-muted-foreground/70" />
      <span className="text-xs text-muted-foreground/80">{label}</span>
      {params && (
        <span className="text-[10px] text-muted-foreground/50 font-mono truncate max-w-[200px]">
          {params}
        </span>
      )}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Tool error widget
// ---------------------------------------------------------------------------

export function ToolErrorWidget({ tool, error }: { tool: string; error: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex items-center gap-2 py-1 px-2.5 rounded-lg bg-red-500/5 border border-red-500/20 my-1"
    >
      <AlertTriangle className="w-3 h-3 text-red-400 shrink-0" />
      <span className="text-xs text-red-400/80">
        {tool.replace(/_/g, ' ')}: {error}
      </span>
    </motion.div>
  )
}

function MarketSearchHeader({ count, query }: { count: number; query: string }) {
  const { t } = useTranslation()
  return (
    <span className="text-[10px] font-medium text-muted-foreground">
      {count === 1
        ? t('aiChatToolWidgets.marketSearch.foundOne', { count, query })
        : t('aiChatToolWidgets.marketSearch.foundOther', { count, query })}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Tool result widget — dispatches to type-specific renderers
// ---------------------------------------------------------------------------

export function ToolResultWidget({ tool, output }: ToolWidgetProps) {
  if (output.error) {
    return <ToolErrorWidget tool={tool} error={String(output.error)} />
  }

  // Dispatch to specialized renderers
  if (tool === 'search_markets') return <MarketSearchWidget data={output} />
  if (tool === 'get_market_details') return <MarketDetailWidget data={output} />
  if (tool === 'get_live_prices') return <LivePricesWidget data={output} />
  if (tool === 'get_open_positions') return <PositionsWidget data={output} />
  if (tool === 'get_account_balance') return <BalanceWidget data={output} />
  if (tool === 'search_news' || tool === 'get_news_edges') return <NewsWidget data={output} />
  if (tool === 'analyze_market_sentiment') return <SentimentWidget data={output} />
  if (tool === 'web_search') return <WebSearchWidget data={output} />
  if (tool === 'get_system_status') return <SystemStatusWidget data={output} />
  if (tool === 'list_strategies') return <StrategiesWidget data={output} />
  if (tool === 'get_portfolio_performance') return <PerformanceWidget data={output} />
  if (tool === 'check_orderbook') return <OrderbookWidget data={output} />
  if (tool === 'get_trader_overview') return <TraderOverviewWidget data={output} />
  if (tool === 'cortex_get_fleet_status') return <FleetStatusWidget data={output} />
  if (tool === 'get_active_signals') return <SignalsWidget data={output} />

  // Generic fallback
  return <GenericResultWidget tool={tool} data={output} />
}

// ---------------------------------------------------------------------------
// Market search results
// ---------------------------------------------------------------------------

function MarketSearchWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const markets = (data.markets as any[]) || []
  const count = (data.count as number) || 0
  const query = data.query as string

  if (count === 0) {
    return (
      <CompactResultCard
        icon={Search}
        label={t('aiChatToolWidgets.marketSearch.noResults', { query })}
        color="amber"
      />
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 overflow-hidden"
    >
      <div className="px-2.5 py-1.5 border-b border-border/20 flex items-center gap-2">
        <Search className="w-3 h-3 text-purple-400" />
        <MarketSearchHeader count={count} query={query} />
      </div>
      <div className="divide-y divide-border/10">
        {markets.slice(0, 6).map((m: any, i: number) => {
          const prices = parsePrices(m.outcome_prices)
          return (
            <div key={i} className="px-2.5 py-1.5 flex items-center gap-2 hover:bg-muted/20 transition-colors">
              <div className="flex-1 min-w-0">
                <p className="text-xs text-foreground/90 truncate">{m.question}</p>
                <div className="flex items-center gap-3 mt-0.5">
                  <PricePill label={t('aiChatToolWidgets.priceLabels.yes')} price={prices[0]} color="emerald" />
                  <PricePill label={t('aiChatToolWidgets.priceLabels.no')} price={prices[1]} color="red" />
                  <span className="text-[10px] text-muted-foreground/50">
                    {t('aiChatToolWidgets.volumeShort', { value: formatCompact(parseFloat(m.volume || '0')) })}
                  </span>
                  <span className="text-[10px] text-muted-foreground/50">
                    {t('aiChatToolWidgets.liquidityShort', { value: formatCompact(parseFloat(m.liquidity || '0')) })}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
      {count > 6 && (
        <div className="px-2.5 py-1 border-t border-border/20 text-center">
          <span className="text-[10px] text-muted-foreground/50">{t('aiChatToolWidgets.plusMore', { count: count - 6 })}</span>
        </div>
      )}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Market detail
// ---------------------------------------------------------------------------

function MarketDetailWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const market = (data.market as any) || data
  const question = market.question || market.title || t('aiChatToolWidgets.marketDetail.defaultTitle')
  const prices = parsePrices(market.outcome_prices || market.outcomes)

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-purple-500/20 bg-purple-500/5 p-2.5"
    >
      <p className="text-xs font-medium text-foreground/90 mb-1.5">{question}</p>
      <div className="flex items-center gap-3 flex-wrap">
        <PricePill label={t('aiChatToolWidgets.priceLabels.yes')} price={prices[0]} color="emerald" />
        <PricePill label={t('aiChatToolWidgets.priceLabels.no')} price={prices[1]} color="red" />
        {market.volume && (
          <span className="text-[10px] text-muted-foreground/60">
            {t('aiChatToolWidgets.volumeShort', { value: formatCompact(parseFloat(market.volume)) })}
          </span>
        )}
        {market.liquidity && (
          <span className="text-[10px] text-muted-foreground/60">
            {t('aiChatToolWidgets.liquidityShort', { value: formatCompact(parseFloat(market.liquidity)) })}
          </span>
        )}
        {market.end_date && (
          <span className="text-[10px] text-muted-foreground/60">
            {t('aiChatToolWidgets.endsLabel', { date: new Date(market.end_date).toLocaleDateString() })}
          </span>
        )}
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Live prices
// ---------------------------------------------------------------------------

function LivePricesWidget({ data }: { data: Record<string, unknown> }) {
  const prices = (data.prices as any) || data
  const entries = Object.entries(prices).filter(([k]) => k !== 'count' && k !== 'error')

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="my-1.5 flex flex-wrap gap-1.5"
    >
      {entries.slice(0, 8).map(([token, price]) => (
        <span key={token} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-muted/30 border border-border/20 text-[10px] font-mono">
          <span className="text-muted-foreground/60 truncate max-w-[80px]">{token.slice(0, 8)}...</span>
          <span className="text-foreground/90 font-bold">{typeof price === 'number' ? `${(price * 100).toFixed(1)}c` : String(price)}</span>
        </span>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

function PositionsWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const positions = (data.positions as any[]) || []

  if (positions.length === 0) {
    return <CompactResultCard icon={Wallet} label={t('aiChatToolWidgets.positions.empty')} color="muted" />
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 overflow-hidden"
    >
      <div className="px-2.5 py-1.5 border-b border-border/20 flex items-center gap-2">
        <Wallet className="w-3 h-3 text-cyan-400" />
        <span className="text-[10px] font-medium text-muted-foreground">
          {positions.length === 1
            ? t('aiChatToolWidgets.positions.headerOne', { count: positions.length })
            : t('aiChatToolWidgets.positions.headerOther', { count: positions.length })}
        </span>
      </div>
      {positions.slice(0, 5).map((pos: any, i: number) => (
        <div key={i} className="px-2.5 py-1.5 border-b border-border/10 last:border-0 flex items-center justify-between">
          <span className="text-xs text-foreground/80 truncate flex-1 mr-2">{pos.question || pos.market || pos.id}</span>
          <div className="flex items-center gap-2 shrink-0">
            <span className={cn('text-[10px] font-bold', pos.side === 'YES' ? 'text-emerald-400' : 'text-red-400')}>
              {pos.side}
            </span>
            {pos.pnl !== undefined && (
              <span className={cn('text-[10px] font-mono', pos.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {pos.pnl >= 0 ? '+' : ''}{typeof pos.pnl === 'number' ? `$${pos.pnl.toFixed(2)}` : pos.pnl}
              </span>
            )}
          </div>
        </div>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Balance
// ---------------------------------------------------------------------------

function BalanceWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const balance = data.balance ?? data.total ?? data.available
  const value = typeof balance === 'number' ? balance.toFixed(2) : String(balance ?? t('aiChatToolWidgets.notAvailable'))
  return (
    <CompactResultCard
      icon={DollarSign}
      label={t('aiChatToolWidgets.balance.label', { value })}
      color="emerald"
    />
  )
}

// ---------------------------------------------------------------------------
// News results
// ---------------------------------------------------------------------------

function NewsWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const articles = (data.articles as any[]) || (data.edges as any[]) || (data.news as any[]) || []
  const count = articles.length || (data.count as number) || 0

  if (count === 0) {
    return <CompactResultCard icon={Newspaper} label={t('aiChatToolWidgets.news.empty')} color="amber" />
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 overflow-hidden"
    >
      <div className="px-2.5 py-1.5 border-b border-border/20 flex items-center gap-2">
        <Newspaper className="w-3 h-3 text-amber-400" />
        <span className="text-[10px] font-medium text-muted-foreground">
          {count === 1
            ? t('aiChatToolWidgets.news.headerOne', { count })
            : t('aiChatToolWidgets.news.headerOther', { count })}
        </span>
      </div>
      {articles.slice(0, 4).map((a: any, i: number) => (
        <div key={i} className="px-2.5 py-1.5 border-b border-border/10 last:border-0">
          <p className="text-xs text-foreground/80 truncate">{a.title || a.headline || a.summary || JSON.stringify(a).slice(0, 80)}</p>
          {a.source && <span className="text-[10px] text-muted-foreground/50">{a.source}</span>}
        </div>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Sentiment
// ---------------------------------------------------------------------------

function SentimentWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const score = (data.sentiment_score ?? data.score ?? 0) as number
  const rawLabel = (data.overall_sentiment ?? data.sentiment ?? 'neutral') as string
  const translatedLabel = t(`aiChatToolWidgets.sentiment.labels.${rawLabel}`, { defaultValue: rawLabel })
  const pct = ((score + 1) / 2) * 100

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 p-2.5"
    >
      <div className="flex items-center gap-2 mb-1.5">
        <Brain className="w-3 h-3 text-cyan-400" />
        <span className="text-[10px] font-medium text-muted-foreground">{t('aiChatToolWidgets.sentiment.title')}</span>
        <span className={cn(
          'text-[10px] font-bold ml-auto',
          score > 0.3 ? 'text-emerald-400' : score < -0.3 ? 'text-red-400' : 'text-amber-400'
        )}>
          {translatedLabel.toUpperCase()} ({score > 0 ? '+' : ''}{score.toFixed(2)})
        </span>
      </div>
      <div className="relative h-1.5 rounded-full bg-muted/40 overflow-hidden">
        <div className="absolute inset-0 flex">
          <div className="w-1/2 bg-red-500/15" />
          <div className="w-1/2 bg-emerald-500/15" />
        </div>
        <motion.div
          className="absolute top-0 h-full w-1 bg-foreground rounded-full"
          initial={{ left: '50%' }}
          animate={{ left: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{ transform: 'translateX(-50%)' }}
        />
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Web search
// ---------------------------------------------------------------------------

function WebSearchWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const results = (data.results as any[]) || []
  const count = results.length || (data.count as number) || 0

  if (count === 0) {
    return <CompactResultCard icon={Globe} label={t('aiChatToolWidgets.webSearch.empty')} color="amber" />
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 overflow-hidden"
    >
      <div className="px-2.5 py-1.5 border-b border-border/20 flex items-center gap-2">
        <Globe className="w-3 h-3 text-blue-400" />
        <span className="text-[10px] font-medium text-muted-foreground">
          {count === 1
            ? t('aiChatToolWidgets.webSearch.headerOne', { count })
            : t('aiChatToolWidgets.webSearch.headerOther', { count })}
        </span>
      </div>
      {results.slice(0, 4).map((r: any, i: number) => (
        <div key={i} className="px-2.5 py-1.5 border-b border-border/10 last:border-0">
          <p className="text-xs text-foreground/80 truncate">{r.title}</p>
          {r.snippet && <p className="text-[10px] text-muted-foreground/60 truncate mt-0.5">{r.snippet}</p>}
        </div>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// System status
// ---------------------------------------------------------------------------

function SystemStatusWidget({ data }: { data: Record<string, unknown> }) {
  const status = (data.status as Record<string, any>) || data
  const entries = Object.entries(status).filter(([k]) => k !== 'error')

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="my-1.5 flex flex-wrap gap-1.5"
    >
      {entries.map(([key, val]) => {
        const active = val?.active !== false
        return (
          <span key={key} className={cn(
            'inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px]',
            active
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-muted/30 text-muted-foreground/60 border-border/20'
          )}>
            <span className={cn('w-1.5 h-1.5 rounded-full', active ? 'bg-emerald-400' : 'bg-muted-foreground/40')} />
            {key.replace(/_/g, ' ')}
          </span>
        )
      })}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

function StrategiesWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const strategies = (data.strategies as any[]) || []
  const count = strategies.length || (data.count as number) || 0

  return (
    <CompactResultCard
      icon={Target}
      label={count === 1
        ? t('aiChatToolWidgets.strategies.loadedOne', { count })
        : t('aiChatToolWidgets.strategies.loadedOther', { count })}
      color="purple"
    />
  )
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

function PerformanceWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const totalPnl = data.total_pnl ?? data.pnl ?? data.total_profit
  const winRate = data.win_rate ?? data.winRate

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="my-1.5 flex flex-wrap gap-2"
    >
      {totalPnl !== undefined && (
        <span className={cn(
          'inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-mono',
          Number(totalPnl) >= 0
            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
            : 'bg-red-500/10 text-red-400 border-red-500/20'
        )}>
          {Number(totalPnl) >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          {t('aiChatToolWidgets.performance.pnl', { sign: Number(totalPnl) >= 0 ? '+' : '', value: Number(totalPnl).toFixed(2) })}
        </span>
      )}
      {winRate !== undefined && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-muted/30 border border-border/20 text-[10px]">
          {t('aiChatToolWidgets.performance.winRate', { value: (Number(winRate) * 100).toFixed(1) })}
        </span>
      )}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Orderbook
// ---------------------------------------------------------------------------

function OrderbookWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const bids = (data.bids as number) ?? (data.bid_depth as number)
  const asks = (data.asks as number) ?? (data.ask_depth as number)
  const spread = data.spread as number | undefined

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="my-1.5 flex flex-wrap gap-2"
    >
      {bids !== undefined && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-[10px] text-emerald-400">
          {t('aiChatToolWidgets.orderbook.bids', { value: formatCompact(bids) })}
        </span>
      )}
      {asks !== undefined && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/20 text-[10px] text-red-400">
          {t('aiChatToolWidgets.orderbook.asks', { value: formatCompact(asks) })}
        </span>
      )}
      {spread !== undefined && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-muted/30 border border-border/20 text-[10px]">
          {t('aiChatToolWidgets.orderbook.spread', { value: (spread * 100).toFixed(2) })}
        </span>
      )}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Trader overview
// ---------------------------------------------------------------------------

function TraderOverviewWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const traders = (data.traders as any[]) || []
  const count = (data.count as number) || traders.length

  if (count === 0) {
    return <CompactResultCard icon={BarChart3} label={t('aiChatToolWidgets.traderOverview.empty')} color="amber" />
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 overflow-hidden"
    >
      <div className="px-2.5 py-1.5 border-b border-border/20 flex items-center gap-2">
        <BarChart3 className="w-3 h-3 text-cyan-400" />
        <span className="text-[10px] font-medium text-muted-foreground">
          {count === 1
            ? t('aiChatToolWidgets.traderOverview.headerOne', { count })
            : t('aiChatToolWidgets.traderOverview.headerOther', { count })}
        </span>
      </div>
      {traders.slice(0, 6).map((tr: any, i: number) => (
        <div key={i} className="px-2.5 py-1.5 border-b border-border/10 last:border-0 flex items-center justify-between">
          <span className="text-xs text-foreground/80 truncate flex-1 mr-2">
            {tr.name || tr.id || t('aiChatToolWidgets.traderOverview.fallbackName', { n: i + 1 })}
          </span>
          <div className="flex items-center gap-2 shrink-0">
            {tr.status && (
              <span className={cn('text-[10px]', tr.status === 'active' ? 'text-emerald-400' : 'text-muted-foreground/50')}>
                {t(`aiChatToolWidgets.traderOverview.status.${tr.status}`, { defaultValue: tr.status })}
              </span>
            )}
            {tr.pnl !== undefined && (
              <span className={cn('text-[10px] font-mono', Number(tr.pnl) >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {Number(tr.pnl) >= 0 ? '+' : ''}${Number(tr.pnl).toFixed(2)}
              </span>
            )}
            {tr.open_positions !== undefined && (
              <span className="text-[10px] text-muted-foreground/50">{t('aiChatToolWidgets.traderOverview.positionsShort', { count: tr.open_positions })}</span>
            )}
          </div>
        </div>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Fleet status (cortex)
// ---------------------------------------------------------------------------

function FleetStatusWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const strategies = (data.strategies as any[]) || (data.active_strategies as any[]) || []

  const items: { label: string; value: string; color: string }[] = []
  if (data.total_traders !== undefined) items.push({ label: t('aiChatToolWidgets.fleetStatus.traders'), value: String(data.total_traders), color: 'cyan' })
  if (data.active_traders !== undefined) items.push({ label: t('aiChatToolWidgets.fleetStatus.active'), value: String(data.active_traders), color: 'emerald' })
  if (strategies.length) items.push({ label: t('aiChatToolWidgets.fleetStatus.strategies'), value: String(strategies.length), color: 'purple' })
  if (data.total_pnl !== undefined) items.push({ label: t('aiChatToolWidgets.fleetStatus.pnl'), value: `$${Number(data.total_pnl).toFixed(2)}`, color: Number(data.total_pnl) >= 0 ? 'emerald' : 'red' })

  if (items.length === 0) {
    // Fallback: summarize whatever keys we got
    const keys = Object.keys(data).filter(k => k !== 'error')
    return <CompactResultCard icon={Activity} label={t('aiChatToolWidgets.fleetStatus.fallback', { count: keys.length })} color="cyan" />
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="my-1.5 flex flex-wrap gap-1.5"
    >
      {items.map(({ label, value, color }) => (
        <span key={label} className={cn(
          'inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px]',
          color === 'emerald' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
          color === 'red' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
          color === 'cyan' ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20' :
          color === 'purple' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
          'bg-muted/30 text-muted-foreground/60 border-border/20'
        )}>
          {label}: <span className="font-mono font-bold">{value}</span>
        </span>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Active signals
// ---------------------------------------------------------------------------

function SignalsWidget({ data }: { data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const signals = (data.signals as any[]) || []
  const count = signals.length || (data.count as number) || 0

  if (count === 0) {
    return <CompactResultCard icon={Zap} label={t('aiChatToolWidgets.signals.empty')} color="muted" />
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-1.5 rounded-lg border border-border/30 bg-muted/10 overflow-hidden"
    >
      <div className="px-2.5 py-1.5 border-b border-border/20 flex items-center gap-2">
        <Zap className="w-3 h-3 text-amber-400" />
        <span className="text-[10px] font-medium text-muted-foreground">
          {count === 1
            ? t('aiChatToolWidgets.signals.headerOne', { count })
            : t('aiChatToolWidgets.signals.headerOther', { count })}
        </span>
      </div>
      {signals.slice(0, 5).map((s: any, i: number) => (
        <div key={i} className="px-2.5 py-1.5 border-b border-border/10 last:border-0 flex items-center justify-between">
          <span className="text-xs text-foreground/80 truncate flex-1 mr-2">
            {s.type || s.signal_type || s.name || t('aiChatToolWidgets.signals.fallbackName', { n: i + 1 })}
          </span>
          <div className="flex items-center gap-2 shrink-0">
            {s.strength !== undefined && (
              <span className={cn('text-[10px] font-mono font-bold',
                Number(s.strength) > 0.7 ? 'text-emerald-400' :
                Number(s.strength) > 0.4 ? 'text-amber-400' : 'text-muted-foreground/50'
              )}>
                {(Number(s.strength) * 100).toFixed(0)}%
              </span>
            )}
            {s.market && <span className="text-[10px] text-muted-foreground/50 truncate max-w-[120px]">{s.market}</span>}
          </div>
        </div>
      ))}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Generic fallback — compact summary
// ---------------------------------------------------------------------------

function GenericResultWidget({ tool, data }: { tool: string; data: Record<string, unknown> }) {
  const { t } = useTranslation()
  const Icon = TOOL_ICONS[tool] || CheckCircle
  const label = useToolLabel(tool)
  const entries = Object.entries(data).filter(([, v]) => v !== null && v !== undefined)

  // Build a smart summary: count arrays, show scalars
  const parts: string[] = []
  for (const [k, v] of entries.slice(0, 4)) {
    if (Array.isArray(v)) {
      parts.push(`${v.length} ${k}`)
    } else if (typeof v === 'number') {
      parts.push(`${k}: ${v}`)
    } else if (typeof v === 'string' && v.length < 40) {
      parts.push(`${k}: ${v}`)
    } else if (typeof v === 'boolean') {
      parts.push(`${k}: ${v ? t('aiChatToolWidgets.boolYes') : t('aiChatToolWidgets.boolNo')}`)
    }
  }

  return (
    <CompactResultCard
      icon={Icon}
      label={parts.length > 0
        ? t('aiChatToolWidgets.summaryWithParts', { label, parts: parts.join(', ') })
        : t('aiChatToolWidgets.summaryComplete', { label })}
      color="muted"
    />
  )
}

// ---------------------------------------------------------------------------
// Shared components
// ---------------------------------------------------------------------------

function CompactResultCard({ icon: Icon, label, color }: { icon: React.ElementType; label: string; color: string }) {
  const colors: Record<string, string> = {
    emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    red: 'bg-red-500/10 border-red-500/20 text-red-400',
    amber: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
    purple: 'bg-purple-500/10 border-purple-500/20 text-purple-400',
    cyan: 'bg-cyan-500/10 border-cyan-500/20 text-cyan-400',
    muted: 'bg-muted/20 border-border/20 text-muted-foreground/70',
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={cn('my-1 inline-flex items-center gap-2 py-1 px-2.5 rounded-lg border text-[10px]', colors[color] || colors.muted)}
    >
      <Icon className="w-3 h-3" />
      {label}
    </motion.div>
  )
}

function PricePill({ label, price, color }: { label: string; price: number; color: string }) {
  const cents = (price * 100).toFixed(1)
  return (
    <span className={cn(
      'inline-flex items-center gap-1 text-[10px]',
      color === 'emerald' ? 'text-emerald-400' : 'text-red-400'
    )}>
      <span className={cn('w-1.5 h-1.5 rounded-full', color === 'emerald' ? 'bg-emerald-500' : 'bg-red-500')} />
      <span className="text-muted-foreground/60">{label}</span>
      <span className="font-mono font-bold">{cents}c</span>
    </span>
  )
}

// ---------------------------------------------------------------------------
// Animated text for final answer
// ---------------------------------------------------------------------------

export function AnimatedText({ text }: { text: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
    >
      {text}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function parsePrices(raw: unknown): [number, number] {
  if (typeof raw === 'string') {
    try {
      const arr = JSON.parse(raw)
      return [parseFloat(arr[0]) || 0, parseFloat(arr[1]) || 0]
    } catch {
      return [0, 0]
    }
  }
  if (Array.isArray(raw)) {
    return [parseFloat(raw[0]) || 0, parseFloat(raw[1]) || 0]
  }
  return [0, 0]
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}
