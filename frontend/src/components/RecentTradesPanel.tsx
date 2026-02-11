import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  AlertCircle,
  Bell,
  ChevronDown,
  ChevronUp,
  Clock,
  ExternalLink,
  Filter,
  Hash,
  RefreshCw,
  Target,
  TrendingDown,
  TrendingUp,
  Users,
  Wallet,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import {
  discoveryApi,
  type TrackedTraderOpportunity,
} from '../services/discoveryApi'
import {
  getRecentTradesFromWallets,
  type RecentTradeFromWallet,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'

interface Props {
  onNavigateToWallet?: (address: string) => void
}

type TierFilter = 'WATCH' | 'HIGH' | 'EXTREME'
type SignalSideFilter = 'all' | 'BUY' | 'SELL'

type LiveSignalAlert = {
  id: string
  market: string
  tier: TierFilter
  conviction: number
  outcome: string | null
}

const TIER_COLORS: Record<TierFilter, string> = {
  WATCH: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  HIGH: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  EXTREME: 'bg-red-500/10 text-red-400 border-red-500/20',
}

const TIER_BORDER_COLORS: Record<TierFilter, string> = {
  WATCH: 'border-yellow-500/30',
  HIGH: 'border-orange-500/30',
  EXTREME: 'border-red-500/30',
}

function safeParseTime(value?: string | number | null): Date | null {
  if (value == null) return null
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    const millis = value < 4102444800 ? value * 1000 : value
    const date = new Date(millis)
    return Number.isNaN(date.getTime()) ? null : date
  }

  const str = String(value)
  if (!str) return null
  const direct = new Date(str)
  if (!Number.isNaN(direct.getTime())) return direct

  const asNum = Number(str)
  if (!Number.isNaN(asNum) && asNum > 0) {
    const millis = asNum < 4102444800 ? asNum * 1000 : asNum
    const numericDate = new Date(millis)
    return Number.isNaN(numericDate.getTime()) ? null : numericDate
  }

  return null
}

function formatTimeAgo(value?: string | number | null): string {
  const date = safeParseTime(value)
  if (!date) return 'Unknown'

  const now = Date.now()
  const diffMs = now - date.getTime()
  if (diffMs < 0) return 'Just now'

  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`

  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffHours < 48) return 'Yesterday'
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatDateTime(value?: string | number | null): string {
  const date = safeParseTime(value)
  if (!date) return 'Unknown'
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  })
}

function formatCurrency(value: number): string {
  if (!Number.isFinite(value)) return '$0.00'
  if (Math.abs(value) >= 1000) return `$${(value / 1000).toFixed(1)}k`
  if (Math.abs(value) >= 100) return `$${value.toFixed(0)}`
  return `$${value.toFixed(2)}`
}

function normalizeTradeSide(raw?: string | null): 'BUY' | 'SELL' | '' {
  const side = (raw || '').toUpperCase()
  if (side === 'BUY' || side === 'YES') return 'BUY'
  if (side === 'SELL' || side === 'NO') return 'SELL'
  return ''
}

function isConditionId(value: string): boolean {
  return value.startsWith('0x') && value.length > 20
}

function getMarketName(trade: RecentTradeFromWallet): string {
  if (trade.market_title && trade.market_title.trim()) return trade.market_title
  if (trade.market && !isConditionId(trade.market)) return trade.market
  return ''
}

function getSignalSide(signal: TrackedTraderOpportunity): 'BUY' | 'SELL' | null {
  const outcome = (signal.outcome || '').toUpperCase()
  if (outcome === 'YES') return 'BUY'
  if (outcome === 'NO') return 'SELL'

  const signalType = (signal.signal_type || '').toLowerCase()
  if (signalType.includes('sell')) return 'SELL'
  if (signalType.includes('buy') || signalType.includes('accumulation')) return 'BUY'
  return null
}

function buildSignalMarketUrl(signal: TrackedTraderOpportunity): string {
  return buildPolymarketMarketUrl({ eventSlug: signal.market_slug }) || ''
}

function getPolymarketTradeUrl(trade: RecentTradeFromWallet): string {
  return buildPolymarketMarketUrl({
    eventSlug: trade.event_slug,
    marketSlug: trade.market_slug,
    marketId: trade.market,
  }) || ''
}

function getTradeId(trade: RecentTradeFromWallet, index: number): string {
  return trade.id || trade.transaction_hash || `${trade.wallet_address}-${index}`
}

function convictionColor(value: number): string {
  if (value >= 80) return 'bg-green-500'
  if (value >= 60) return 'bg-yellow-500'
  return 'bg-red-500'
}

function tierRank(value: string | null | undefined): number {
  const normalized = (value || 'WATCH').toUpperCase()
  if (normalized === 'EXTREME') return 3
  if (normalized === 'HIGH') return 2
  return 1
}

function toTier(value: string | null | undefined): TierFilter {
  const normalized = (value || 'WATCH').toUpperCase()
  if (normalized === 'EXTREME') return 'EXTREME'
  if (normalized === 'HIGH') return 'HIGH'
  return 'WATCH'
}

function tradeMatchesSignal(
  trade: RecentTradeFromWallet,
  signal: TrackedTraderOpportunity,
): boolean {
  const signalMarketId = (signal.market_id || '').toLowerCase()
  const tradeMarketId = (trade.market || '').toLowerCase()
  if (signalMarketId && tradeMarketId && signalMarketId === tradeMarketId) {
    return true
  }

  const signalSlug = (signal.market_slug || '').toLowerCase()
  const tradeEventSlug = (trade.event_slug || '').toLowerCase()
  const tradeMarketSlug = (trade.market_slug || '').toLowerCase()
  if (signalSlug && (signalSlug === tradeEventSlug || signalSlug === tradeMarketSlug)) {
    return true
  }

  const signalQuestion = (signal.market_question || '').trim().toLowerCase()
  const tradeTitle = (trade.market_title || '').trim().toLowerCase()
  if (signalQuestion && tradeTitle) {
    if (signalQuestion === tradeTitle) return true
    if (signalQuestion.includes(tradeTitle) || tradeTitle.includes(signalQuestion)) {
      return true
    }
  }

  return false
}

export default function RecentTradesPanel({ onNavigateToWallet }: Props) {
  const [hoursFilter, setHoursFilter] = useState(24)
  const [minTier, setMinTier] = useState<TierFilter>('WATCH')
  const [sideFilter, setSideFilter] = useState<SignalSideFilter>('all')
  const [signalLimit, setSignalLimit] = useState(50)
  const [expandedSignals, setExpandedSignals] = useState<Set<string>>(new Set())
  const [liveAlerts, setLiveAlerts] = useState<LiveSignalAlert[]>([])

  const { lastMessage } = useWebSocket('/ws')

  const {
    data: opportunities = [],
    isLoading: opportunitiesLoading,
    refetch: refetchSignals,
    isRefetching: isRefetchingSignals,
  } = useQuery({
    queryKey: ['tracked-trader-opportunities', minTier, signalLimit],
    queryFn: () => discoveryApi.getTrackedTraderOpportunities(signalLimit, minTier),
    refetchInterval: 30000,
  })

  const {
    data: rawTradesData,
    isLoading: rawTradesLoading,
    refetch: refetchRawTrades,
    isRefetching: isRefetchingRawTrades,
  } = useQuery({
    queryKey: ['recent-trades-from-wallets', hoursFilter, 500],
    queryFn: () => getRecentTradesFromWallets({ limit: 500, hours: hoursFilter }),
    refetchInterval: 30000,
  })

  useEffect(() => {
    if (lastMessage?.type !== 'tracked_trader_signal' || !lastMessage.data) return

    const tier = toTier(lastMessage.data.tier)
    if (tierRank(tier) < tierRank('HIGH')) return

    const conviction = Math.max(
      0,
      Math.min(100, Math.round(Number(lastMessage.data.conviction_score || 0))),
    )

    const alert: LiveSignalAlert = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      market:
        String(lastMessage.data.market_question || '').trim() ||
        String(lastMessage.data.market_id || 'Unknown market'),
      tier,
      conviction,
      outcome: lastMessage.data.outcome || null,
    }

    setLiveAlerts((prev) => [alert, ...prev].slice(0, 4))
    const timer = setTimeout(() => {
      setLiveAlerts((prev) => prev.filter((item) => item.id !== alert.id))
    }, 8000)

    return () => clearTimeout(timer)
  }, [lastMessage])

  const rawTrades = rawTradesData?.trades || []
  const trackedWallets = rawTradesData?.tracked_wallets || 0
  const isLoading = opportunitiesLoading || rawTradesLoading
  const isRefetching = isRefetchingSignals || isRefetchingRawTrades

  const sortedSignals = useMemo(() => {
    return [...opportunities].sort((a, b) => {
      const convictionDiff = (b.conviction_score || 0) - (a.conviction_score || 0)
      if (convictionDiff !== 0) return convictionDiff
      const bTime = safeParseTime(b.last_seen_at || b.detected_at)?.getTime() || 0
      const aTime = safeParseTime(a.last_seen_at || a.detected_at)?.getTime() || 0
      return bTime - aTime
    })
  }, [opportunities])

  const filteredSignals = useMemo(() => {
    if (sideFilter === 'all') return sortedSignals
    return sortedSignals.filter((signal) => getSignalSide(signal) === sideFilter)
  }, [sortedSignals, sideFilter])

  const signalTradesMap = useMemo(() => {
    const map = new Map<string, RecentTradeFromWallet[]>()

    for (const signal of filteredSignals) {
      const expectedSide = getSignalSide(signal)
      const matched = rawTrades
        .filter((trade) => tradeMatchesSignal(trade, signal))
        .filter((trade) =>
          expectedSide ? normalizeTradeSide(trade.side) === expectedSide : true,
        )
        .sort((a, b) => {
          const bTime =
            safeParseTime(
              b.timestamp_iso || b.match_time || b.timestamp || b.time || b.created_at,
            )?.getTime() || 0
          const aTime =
            safeParseTime(
              a.timestamp_iso || a.match_time || a.timestamp || a.time || a.created_at,
            )?.getTime() || 0
          return bTime - aTime
        })

      map.set(signal.id, matched)
    }

    return map
  }, [filteredSignals, rawTrades])

  const uniqueSignalMarkets = useMemo(() => {
    const keys = new Set<string>()
    for (const signal of filteredSignals) {
      keys.add(`${signal.market_id}:${signal.outcome || ''}`)
    }
    return keys.size
  }, [filteredSignals])

  const highSignals = filteredSignals.filter(
    (signal) => tierRank(signal.tier) >= tierRank('HIGH'),
  ).length
  const extremeSignals = filteredSignals.filter(
    (signal) => toTier(signal.tier) === 'EXTREME',
  ).length

  const avgConviction = filteredSignals.length
    ? filteredSignals.reduce((sum, signal) => sum + (signal.conviction_score || 0), 0) /
      filteredSignals.length
    : 0

  const auditedTradeCount = Array.from(signalTradesMap.values()).reduce(
    (sum, trades) => sum + trades.length,
    0,
  )

  const toggleExpanded = (signalId: string) => {
    setExpandedSignals((prev) => {
      const next = new Set(prev)
      if (next.has(signalId)) {
        next.delete(signalId)
      } else {
        next.add(signalId)
      }
      return next
    })
  }

  const handleRefresh = () => {
    refetchSignals()
    refetchRawTrades()
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/10 rounded-lg">
            <Zap className="w-5 h-5 text-orange-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              Tracked Trader Opportunities
            </h2>
            <p className="text-sm text-muted-foreground/70">
              Signal-first feed of clustered smart-wallet entries with expandable raw
              trade audits
            </p>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={isRefetching}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm',
            'bg-muted text-foreground/80 hover:bg-accent transition-colors',
            isRefetching && 'opacity-50',
          )}
        >
          <RefreshCw className={cn('w-4 h-4', isRefetching && 'animate-spin')} />
          Refresh
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-4 p-3 bg-card rounded-lg border border-border">
        <Filter className="w-4 h-4 text-muted-foreground/70" />

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Min tier:</span>
          <select
            value={minTier}
            onChange={(e) => setMinTier(e.target.value as TierFilter)}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value="WATCH">Watch (5+)</option>
            <option value="HIGH">High (10+)</option>
            <option value="EXTREME">Extreme (15+)</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Side:</span>
          <select
            value={sideFilter}
            onChange={(e) => setSideFilter(e.target.value as SignalSideFilter)}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value="all">All</option>
            <option value="BUY">Buy clusters</option>
            <option value="SELL">Sell clusters</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Raw window:</span>
          <select
            value={hoursFilter}
            onChange={(e) => setHoursFilter(Number(e.target.value))}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value={1}>Last hour</option>
            <option value={6}>Last 6 hours</option>
            <option value={24}>Last 24 hours</option>
            <option value={48}>Last 48 hours</option>
            <option value={168}>Last 7 days</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground/70">Max signals:</span>
          <select
            value={signalLimit}
            onChange={(e) => setSignalLimit(Number(e.target.value))}
            className="bg-muted border border-border rounded px-2 py-1 text-sm"
          >
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70">Signals</p>
          <p className="text-lg font-semibold text-foreground">{filteredSignals.length}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70">High / Extreme</p>
          <p className="text-lg font-semibold text-orange-400">
            {highSignals}
            <span className="text-muted-foreground/60 text-sm ml-1">/ {extremeSignals}</span>
          </p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70">Avg Conviction</p>
          <p className="text-lg font-semibold text-foreground">{avgConviction.toFixed(1)}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70">Markets / Wallets</p>
          <p className="text-lg font-semibold text-foreground">
            <span className="text-blue-400">{uniqueSignalMarkets}</span>
            <span className="text-muted-foreground/50 mx-1">/</span>
            <span className="text-orange-400">{trackedWallets}</span>
          </p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70">Audited Trades</p>
          <p className="text-lg font-semibold text-foreground">{auditedTradeCount}</p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground/70" />
        </div>
      ) : filteredSignals.length === 0 ? (
        <div className="text-center py-12 bg-card rounded-lg border border-border">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <p className="text-muted-foreground">No tracked-trader signals found</p>
          <p className="text-sm text-muted-foreground/50 mt-1">
            Try lowering tier/side filters or wait for new clustered entries
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredSignals.map((signal) => {
            const tier = toTier(signal.tier)
            const conviction = Math.round(signal.conviction_score || signal.strength * 100 || 0)
            const signalSide = getSignalSide(signal)
            const isBuy = signalSide === 'BUY'
            const isExpanded = expandedSignals.has(signal.id)
            const marketUrl = buildSignalMarketUrl(signal)
            const relatedTrades = signalTradesMap.get(signal.id) || []
            const topWallets = signal.top_wallets || []

            return (
              <div
                key={signal.id}
                className={cn(
                  'bg-card border rounded-lg overflow-hidden transition-colors',
                  TIER_BORDER_COLORS[tier],
                )}
              >
                <div
                  className="p-4 cursor-pointer"
                  onClick={() => toggleExpanded(signal.id)}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={cn(
                        'flex-shrink-0 w-1 h-16 rounded-full mt-0.5',
                        isBuy ? 'bg-green-500' : signalSide === 'SELL' ? 'bg-red-500' : 'bg-blue-500',
                      )}
                    />

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                        <span
                          className={cn(
                            'px-1.5 py-0.5 rounded text-xs font-medium border',
                            TIER_COLORS[tier],
                          )}
                        >
                          {tier}
                        </span>
                        <span
                          className={cn(
                            'px-1.5 py-0.5 rounded text-xs font-medium',
                            isBuy
                              ? 'bg-green-500/10 text-green-400'
                              : signalSide === 'SELL'
                                ? 'bg-red-500/10 text-red-400'
                                : 'bg-blue-500/10 text-blue-400',
                          )}
                        >
                          {signal.outcome || signal.signal_type.replace(/_/g, ' ')}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/80">
                          Conviction {conviction}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/80">
                          {signal.window_minutes || 60}m window
                        </span>
                      </div>

                      <h3 className="font-medium text-sm text-foreground line-clamp-2">
                        {signal.market_question || signal.market_id}
                      </h3>

                      <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                        <div>
                          <p className="text-muted-foreground/70">Adjusted Wallets</p>
                          <p className="font-semibold text-foreground flex items-center gap-1">
                            <Users className="w-3 h-3 text-muted-foreground/70" />
                            {signal.cluster_adjusted_wallet_count || signal.wallet_count}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground/70">Core Wallets</p>
                          <p className="font-semibold text-foreground">
                            {signal.unique_core_wallets || 0}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground/70">Net Notional</p>
                          <p className="font-semibold text-foreground">
                            {formatCurrency(signal.net_notional || 0)}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground/70">Last Reinforced</p>
                          <p className="font-semibold text-foreground">
                            {formatTimeAgo(signal.last_seen_at || signal.detected_at)}
                          </p>
                        </div>
                      </div>

                      <div className="mt-3">
                        <div className="flex items-center justify-between text-[11px] mb-1">
                          <span className="text-muted-foreground/70">Conviction meter</span>
                          <span className="text-foreground/90 font-medium">{conviction}/100</span>
                        </div>
                        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className={cn('h-full rounded-full', convictionColor(conviction))}
                            style={{ width: `${Math.max(0, Math.min(100, conviction))}%` }}
                          />
                        </div>
                      </div>

                      {topWallets.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {topWallets.slice(0, 4).map((wallet) => (
                            <button
                              key={`${signal.id}-${wallet.address}`}
                              onClick={(e) => {
                                e.stopPropagation()
                                onNavigateToWallet?.(wallet.address)
                              }}
                              className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted/70 hover:bg-muted transition-colors text-[11px] text-foreground/85"
                            >
                              <Wallet className="w-3 h-3 text-muted-foreground/70" />
                              <span>
                                {wallet.username || `${wallet.address.slice(0, 6)}...${wallet.address.slice(-4)}`}
                              </span>
                              <span className="text-muted-foreground/70">
                                {(wallet.composite_score * 100).toFixed(0)}
                              </span>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="flex-shrink-0 text-right">
                      <div className="flex items-center gap-1 text-xs text-muted-foreground/70 justify-end">
                        <Clock className="w-3 h-3" />
                        {formatTimeAgo(signal.detected_at)}
                      </div>
                      <div className="mt-2 flex items-center gap-2 justify-end">
                        {marketUrl && (
                          <a
                            href={marketUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
                          >
                            <ExternalLink className="w-3 h-3" />
                            Market
                          </a>
                        )}
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4 text-muted-foreground/60" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-muted-foreground/60" />
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-border p-4 bg-background/30">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-muted-foreground/70" />
                        <p className="text-sm font-medium text-foreground">Raw Trade Audit Trail</p>
                      </div>
                      <p className="text-xs text-muted-foreground/70">
                        {relatedTrades.length} matching trades in last {hoursFilter}h
                      </p>
                    </div>

                    {relatedTrades.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border p-3 text-sm text-muted-foreground/70">
                        No matching raw trades were found in the selected audit window.
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {relatedTrades.slice(0, 20).map((trade, index) => {
                          const tradeId = getTradeId(trade, index)
                          const tradeSide = normalizeTradeSide(trade.side)
                          const tradeIsBuy = tradeSide === 'BUY'
                          const cost = trade.cost ?? (trade.size ?? 0) * (trade.price ?? 0)
                          const marketName = getMarketName(trade)
                          const polymarketTradeUrl = getPolymarketTradeUrl(trade)

                          return (
                            <div
                              key={tradeId}
                              className="rounded-lg border border-border bg-card/70 p-3"
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span
                                      className={cn(
                                        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium',
                                        tradeIsBuy
                                          ? 'bg-green-500/10 text-green-400'
                                          : 'bg-red-500/10 text-red-400',
                                      )}
                                    >
                                      {tradeIsBuy ? (
                                        <TrendingUp className="w-3 h-3" />
                                      ) : (
                                        <TrendingDown className="w-3 h-3" />
                                      )}
                                      {tradeSide || trade.side || 'TRADE'}
                                    </span>
                                    {trade.outcome && (
                                      <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/80">
                                        {trade.outcome}
                                      </span>
                                    )}
                                    <span className="text-xs text-muted-foreground/70">
                                      {formatTimeAgo(
                                        trade.timestamp_iso ||
                                          trade.match_time ||
                                          trade.timestamp ||
                                          trade.time ||
                                          trade.created_at,
                                      )}
                                    </span>
                                  </div>

                                  <p className="mt-1 text-sm text-foreground truncate">
                                    {marketName || signal.market_question || 'Unknown market'}
                                  </p>

                                  <div className="mt-1 flex items-center gap-2 text-xs">
                                    <button
                                      onClick={() => onNavigateToWallet?.(trade.wallet_address)}
                                      className="text-orange-400 hover:text-orange-300 hover:underline font-mono"
                                    >
                                      {trade.wallet_username || trade.wallet_label}
                                    </button>
                                    <span className="text-muted-foreground/50">
                                      {trade.wallet_address.slice(0, 6)}...{trade.wallet_address.slice(-4)}
                                    </span>
                                  </div>
                                </div>

                                <div className="text-right">
                                  <p
                                    className={cn(
                                      'text-sm font-semibold',
                                      tradeIsBuy ? 'text-green-400' : 'text-red-400',
                                    )}
                                  >
                                    {((trade.price || 0) * 100).toFixed(1)}c
                                  </p>
                                  <p className="text-xs text-muted-foreground/70">
                                    {(trade.size || 0).toLocaleString(undefined, {
                                      maximumFractionDigits: 0,
                                    })}{' '}
                                    shares
                                  </p>
                                  <p className="text-xs text-muted-foreground/70">
                                    {formatCurrency(cost)}
                                  </p>
                                </div>
                              </div>

                              <div className="mt-2 pt-2 border-t border-border/70 flex items-center gap-3 text-xs">
                                <span className="text-muted-foreground/70">
                                  {formatDateTime(
                                    trade.timestamp_iso ||
                                      trade.match_time ||
                                      trade.timestamp ||
                                      trade.time ||
                                      trade.created_at,
                                  )}
                                </span>
                                {polymarketTradeUrl && (
                                  <a
                                    href={polymarketTradeUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300"
                                  >
                                    <ExternalLink className="w-3 h-3" />
                                    Market
                                  </a>
                                )}
                                {trade.transaction_hash && (
                                  <a
                                    href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground/80"
                                  >
                                    <Hash className="w-3 h-3" />
                                    Tx
                                  </a>
                                )}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    )}

                    <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground/70 flex flex-wrap gap-4">
                      <span className="inline-flex items-center gap-1">
                        <Target className="w-3 h-3" />
                        First seen: {formatTimeAgo(signal.first_seen_at || signal.detected_at)}
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Last seen: {formatTimeAgo(signal.last_seen_at || signal.detected_at)}
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Users className="w-3 h-3" />
                        Wallets: {signal.wallet_count}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {liveAlerts.length > 0 && (
        <div className="fixed bottom-4 right-4 z-50 space-y-2 pointer-events-none">
          {liveAlerts.map((alert) => (
            <div
              key={alert.id}
              className={cn(
                'pointer-events-auto min-w-[280px] max-w-[360px] rounded-lg border px-3 py-2 shadow-lg backdrop-blur',
                alert.tier === 'EXTREME'
                  ? 'bg-red-500/15 border-red-500/30 text-red-100'
                  : 'bg-orange-500/15 border-orange-500/30 text-orange-100',
              )}
            >
              <div className="flex items-start gap-2">
                <Bell className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-xs font-semibold">
                    {alert.tier} tracked-trader signal {alert.outcome ? `(${alert.outcome})` : ''}
                  </p>
                  <p className="text-xs mt-0.5 line-clamp-2">{alert.market}</p>
                  <p className="text-[11px] mt-1 opacity-90">
                    Conviction {alert.conviction}/100
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
