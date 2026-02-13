import { useState, useEffect, useMemo, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ExternalLink,
  Activity,
  ChevronRight,
  ArrowUpDown,
  Settings,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import { getCryptoMarkets, CryptoMarket } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import Sparkline from './Sparkline'

// ─── Constants ────────────────────────────────────────────

const WS_MARKETS_STALE_MS = 30000

const ASSET_BAR: Record<string, string> = {
  BTC: 'bg-orange-400',
  ETH: 'bg-blue-400',
  SOL: 'bg-purple-400',
  XRP: 'bg-cyan-400',
}

const ASSET_ICONS: Record<string, string> = {
  BTC: 'https://polymarket-upload.s3.us-east-2.amazonaws.com/BTC+fullsize.png',
  ETH: 'https://polymarket-upload.s3.us-east-2.amazonaws.com/ETH+fullsize.jpg',
  SOL: 'https://polymarket-upload.s3.us-east-2.amazonaws.com/SOL+fullsize.png',
  XRP: 'https://polymarket-upload.s3.us-east-2.amazonaws.com/XRP-logo.png',
}

// ─── Helpers ─────────────────────────────────────────────

function formatUsd(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  if (n >= 1) return `$${n.toFixed(2)}`
  return `$${n.toFixed(4)}`
}

function formatPrice(n: number | null | undefined, decimals = 2): string {
  if (n === null || n === undefined) return '--'
  return `$${n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`
}

function toFiniteNumber(value: unknown): number | null {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function normalizeTimeframe(value: string | null | undefined): string {
  const raw = String(value || '').toLowerCase()
  if (!raw) return ''
  if (raw.includes('4h') || raw.includes('240') || raw.includes('4hr')) return '4h'
  if (raw.includes('1h') || raw.includes('60')) return '1h'
  if (raw.includes('15m') || raw.includes('15min') || raw.includes('15-min') || raw.includes('15 minute')) return '15m'
  if (raw.includes('5m') || raw.includes('5min') || raw.includes('5-min') || raw.includes('5 minute') || (raw.startsWith('5') && !raw.includes('15'))) return '5m'
  return raw
}

type TimeframeFilter = 'all' | '15m' | '5m' | '1h' | '4h'

function extractCryptoMarketsFromInit(payload: any): CryptoMarket[] | null {
  const workers = payload?.workers_status
  if (!Array.isArray(workers)) return null
  const cryptoWorker = workers.find((worker) => worker?.worker_name === 'crypto')
  const markets = cryptoWorker?.stats?.markets
  return Array.isArray(markets) ? markets as CryptoMarket[] : null
}

// ─── Countdown Timer ─────────────────────────────────────

function LiveCountdown({ endTime }: { endTime: string | null }) {
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    const iv = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(iv)
  }, [])

  if (!endTime) return <span className="text-muted-foreground">--:--</span>

  const endMs = new Date(endTime).getTime()
  const diff = Math.max(0, endMs - now)
  const totalSec = Math.floor(diff / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60

  const urgency = totalSec <= 0 ? 'text-red-500' : min < 2 ? 'text-red-400 animate-pulse' : min < 5 ? 'text-yellow-400' : 'text-green-400'

  if (totalSec <= 0) return <span className="text-red-500 font-bold font-data">RESOLVING</span>

  return (
    <div className={cn("flex items-center gap-2 font-data", urgency)}>
      <div className="flex items-baseline gap-0.5">
        <span className="text-2xl font-bold tabular-nums">{String(min).padStart(2, '0')}</span>
        <span className="text-xs text-muted-foreground">MINS</span>
      </div>
      <span className="text-lg font-bold text-muted-foreground/40">:</span>
      <div className="flex items-baseline gap-0.5">
        <span className="text-2xl font-bold tabular-nums">{String(sec).padStart(2, '0')}</span>
        <span className="text-xs text-muted-foreground">SECS</span>
      </div>
    </div>
  )
}

// ─── Oracle Price Display ─────────────────────────────────

function OraclePriceDisplay({
  price,
  priceToBeat,
  source,
  sourceMap,
}: {
  price: number | null
  priceToBeat: number | null
  source: string | null
  sourceMap?: Record<
    string,
    {
      source: string
      price: number | null
      updated_at_ms: number | null
      age_seconds: number | null
    }
  >
}) {
  if (price === null) return null

  const delta = (priceToBeat !== null && priceToBeat !== undefined) ? price - priceToBeat : null
  const isUp = delta !== null && delta >= 0
  const sourceLabel = source
    ? source.toLowerCase().includes('chainlink')
      ? 'chainlink'
      : source.toLowerCase().includes('binance')
        ? 'binance'
        : source
    : 'chainlink'

  const sourceRows = Object.values(sourceMap || {})
    .filter((row) => row && typeof row.price === 'number')
    .map((row) => ({
      label: row.source,
      price: row.price as number,
      age: row.age_seconds,
    }))

  const chainlink = sourceRows.find((row) => row.label.includes('chainlink'))
  const binance = sourceRows.find((row) => row.label.includes('binance'))
  const sourceDelta = chainlink && binance ? (binance.price - chainlink.price) : null

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">oracle source</span>
        <span className="text-[10px] font-medium text-muted-foreground">{sourceLabel}</span>
      </div>
      {/* Price to beat */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">price to beat</span>
        {priceToBeat !== null && priceToBeat !== undefined ? (
          <span className="text-sm font-bold font-data text-muted-foreground">{formatPrice(priceToBeat, 2)}</span>
        ) : (
          <span className="text-[10px] text-muted-foreground/50 italic">waiting for window start...</span>
        )}
      </div>
      {/* Current oracle price */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">current price</span>
        <div className="flex items-center gap-2">
          <span className={cn("text-lg font-bold font-data tabular-nums", delta !== null ? (isUp ? 'text-green-400' : 'text-red-400') : 'text-foreground')}>
            {formatPrice(price, 2)}
          </span>
          {delta !== null && (
            <span className={cn("text-xs font-data font-bold", isUp ? 'text-green-400' : 'text-red-400')}>
              {isUp ? <TrendingUp className="w-3.5 h-3.5 inline" /> : <TrendingDown className="w-3.5 h-3.5 inline" />}
              {' '}{isUp ? '+' : ''}{formatPrice(delta, 2)}
            </span>
          )}
        </div>
      </div>

      {sourceRows.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-border/30">
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider pt-1.5">source view</div>
          {sourceRows.map((row) => (
            <div key={row.label} className="flex items-center justify-between gap-2">
              <span className="text-[10px] text-muted-foreground/70 uppercase tracking-wide">{row.label}</span>
              <span className="text-xs font-data text-muted-foreground">
                {formatPrice(row.price, 2)}
                {row.age !== null ? <span className="text-[10px] text-muted-foreground/60"> · {row.age}s</span> : null}
              </span>
            </div>
          ))}
          {sourceDelta !== null && (
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wide">binance - chainlink</span>
              <span className={cn("text-xs font-data", sourceDelta >= 0 ? 'text-green-400' : 'text-red-400')}>
                {sourceDelta >= 0 ? '+' : ''}{formatPrice(sourceDelta, 2)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Market Card ─────────────────────────────────────────

function CryptoMarketCard({ market }: { market: CryptoMarket }) {
  const chartRef = useRef<HTMLDivElement>(null)
  const [chartWidth, setChartWidth] = useState(300)

  useEffect(() => {
    if (!chartRef.current) return
    const measure = () => {
      if (chartRef.current) setChartWidth(chartRef.current.offsetWidth)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(chartRef.current)
    return () => ro.disconnect()
  }, [])

  const asset = market.asset
  const upPrice = toFiniteNumber(market.up_price)
  const downPrice = toFiniteNumber(market.down_price)
  const combined = toFiniteNumber(market.combined) ?? (
    upPrice !== null && downPrice !== null ? upPrice + downPrice : null
  )
  const spread = combined !== null ? 1 - combined : null

  const polyUrl = buildPolymarketMarketUrl({
    eventSlug: market.event_slug,
  })

  const oracleSeries = useMemo(() => {
    const raw = Array.isArray(market.oracle_history) ? market.oracle_history : []
    const points = raw
      .map((pt) => toFiniteNumber((pt as { p?: unknown; price?: unknown })?.p ?? (pt as { price?: unknown })?.price))
      .filter((v): v is number => Number.isFinite(v))

    if (points.length >= 2) {
      return points
    }

    const now = toFiniteNumber(market.oracle_price)
    return now !== null ? [now, now] : []
  }, [market.oracle_history, market.oracle_price])

  // Parse time window from title (e.g. "Bitcoin Up or Down - February 10, 10:45AM-11:00AM ET")
  const timeWindow = market.event_title?.match(/(\d{1,2}:\d{2}[AP]M)-(\d{1,2}:\d{2}[AP]M)\s*ET/)?.[0] || ''

  return (
    <Card className={cn(
      "overflow-hidden relative group transition-all duration-200",
      "hover:shadow-lg hover:shadow-black/20 hover:border-border/80",
      market.is_live && 'ring-1 ring-green-500/10',
    )}>
      {/* Asset color accent bar */}
      <div className={cn("absolute left-0 top-0 bottom-0 w-1.5 rounded-l-lg", ASSET_BAR[asset] || 'bg-gray-400')} />

      <div className="pl-5 pr-4 py-4 space-y-3">
        {/* Header: Asset icon + name + status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src={ASSET_ICONS[asset]} alt={asset} className="w-8 h-8 rounded-full" onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }} />
            <div>
              <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-foreground">{asset} Up or Down</h3>
              <Badge variant="outline" className="text-[9px] px-1.5 py-0 text-muted-foreground border-muted-foreground/20">
                {market.timeframe.toUpperCase()}
              </Badge>
                {market.is_live ? (
                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 font-bold text-green-400 bg-green-500/15 border-green-500/25">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse mr-1" />
                    LIVE
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 text-yellow-400 bg-yellow-500/10 border-yellow-500/20">NEXT</Badge>
                )}
              </div>
              <p className="text-[11px] text-muted-foreground font-data">{timeWindow}</p>
            </div>
          </div>
          {polyUrl && (
            <a href={polyUrl} target="_blank" rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors p-1">
              <ExternalLink className="w-4 h-4" />
            </a>
          )}
        </div>

        {/* Oracle price + Price to beat */}
        <OraclePriceDisplay
          price={market.oracle_price}
          priceToBeat={market.price_to_beat}
          source={market.oracle_source}
          sourceMap={market.oracle_prices_by_source}
        />

        {/* Oracle price sparkline chart */}
        <div ref={chartRef} className="relative h-14 w-full bg-muted/10 rounded-lg overflow-hidden">
          {oracleSeries.length >= 2 ? (
            <>
              {market.price_to_beat !== null && (
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-dashed border-muted-foreground/20" />
                </div>
              )}
              <Sparkline
                data={oracleSeries}
                width={chartWidth}
                height={56}
                color={market.oracle_price !== null && market.price_to_beat !== null
                  ? (market.oracle_price >= market.price_to_beat ? '#4ade80' : '#f87171')
                  : '#a1a1aa'}
                animated={false}
              />
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-[10px] text-muted-foreground/40">
              Waiting for price data...
            </div>
          )}
        </div>

        {/* Countdown timer */}
        <div className="flex items-center justify-center py-2 bg-muted/20 rounded-lg">
          <LiveCountdown endTime={market.end_time} />
        </div>

        {/* Up / Down prices */}
        <div className="grid grid-cols-2 gap-2">
          <div className={cn(
            "rounded-lg p-2.5 text-center border",
            upPrice !== null && downPrice !== null && upPrice > downPrice
              ? 'bg-green-500/10 border-green-500/20'
              : 'bg-muted/20 border-border/30',
          )}>
            <div className="text-[10px] text-green-400 uppercase tracking-wider font-medium mb-1">
              <TrendingUp className="w-3 h-3 inline mr-1" />Up
            </div>
            <div className="text-lg font-bold font-data tabular-nums text-green-400">
              {upPrice !== null ? `${(upPrice * 100).toFixed(0)}%` : '--'}
            </div>
            <div className="text-[10px] text-muted-foreground font-data">
              {upPrice !== null ? `$${upPrice.toFixed(2)}` : '--'}
            </div>
          </div>
          <div className={cn(
            "rounded-lg p-2.5 text-center border",
            upPrice !== null && downPrice !== null && downPrice > upPrice
              ? 'bg-red-500/10 border-red-500/20'
              : 'bg-muted/20 border-border/30',
          )}>
            <div className="text-[10px] text-red-400 uppercase tracking-wider font-medium mb-1">
              <TrendingDown className="w-3 h-3 inline mr-1" />Down
            </div>
            <div className="text-lg font-bold font-data tabular-nums text-red-400">
              {downPrice !== null ? `${(downPrice * 100).toFixed(0)}%` : '--'}
            </div>
            <div className="text-[10px] text-muted-foreground font-data">
              {downPrice !== null ? `$${downPrice.toFixed(2)}` : '--'}
            </div>
          </div>
        </div>

        {/* Spread / Combined info */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Combined Cost</span>
            <div className="flex items-center gap-2">
              <span className="font-data font-bold text-foreground">
                {combined !== null ? `$${combined.toFixed(3)}` : '--'}
              </span>
              {spread !== null && spread > 0.001 && (
                <span className={cn("font-data font-bold text-green-400")}>
                  ({(spread * 100).toFixed(1)}% spread)
                </span>
              )}
            </div>
          </div>
          {market.best_bid !== null && market.best_ask !== null && (
            <div className="flex justify-between text-[9px] text-muted-foreground/60 font-data">
              <span>Up Bid: ${market.best_bid.toFixed(2)} / Up Ask: ${market.best_ask.toFixed(2)}</span>
              <span>Book spread: ${(market.best_ask - market.best_bid).toFixed(2)}</span>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Liquidity</div>
            <div className="text-xs font-bold font-data text-foreground">{formatUsd(market.liquidity)}</div>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Volume</div>
            <div className="text-xs font-bold font-data text-foreground">{formatUsd(market.volume)}</div>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Last Trade</div>
            <div className="text-xs font-bold font-data text-foreground">{market.last_trade_price !== null ? `$${market.last_trade_price.toFixed(2)}` : '--'}</div>
          </div>
        </div>

        {/* Upcoming markets timeline */}
        {market.upcoming_markets && market.upcoming_markets.length > 0 && (
          <div className="space-y-1 pt-1 border-t border-border/20">
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider font-medium">Upcoming</div>
            {market.upcoming_markets.map((um, i) => {
              const umTime = um.event_title?.match(/(\d{1,2}:\d{2}[AP]M)-(\d{1,2}:\d{2}[AP]M)/)?.[0] || ''
              return (
                <div key={um.id || i} className="flex items-center justify-between text-[10px] font-data text-muted-foreground py-0.5">
                  <div className="flex items-center gap-1.5">
                    <ChevronRight className="w-3 h-3 text-muted-foreground/40" />
                    <span>{umTime}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {um.up_price !== null && um.down_price !== null && (
                      <>
                        <span className="text-green-400">{(um.up_price * 100).toFixed(0)}%</span>
                        <span className="text-muted-foreground/40">/</span>
                        <span className="text-red-400">{(um.down_price * 100).toFixed(0)}%</span>
                      </>
                    )}
                    <span className="text-muted-foreground/50">{formatUsd(um.liquidity)}</span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </Card>
  )
}

// ─── Main Panel ──────────────────────────────────────────

interface Props {
  onExecute?: (opportunity: any) => void
  onOpenCopilot?: (opportunity: any) => void
  onOpenCryptoSettings?: () => void
}

export default function CryptoMarketsPanel({ onExecute, onOpenCopilot, onOpenCryptoSettings }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const [timeframeFilter, setTimeframeFilter] = useState<TimeframeFilter>('all')
  const [isDocumentVisible, setIsDocumentVisible] = useState(
    () => (typeof document === 'undefined' ? true : document.visibilityState === 'visible')
  )
  const [isPanelInViewport, setIsPanelInViewport] = useState(true)
  // Intentionally kept for interface parity with other panels and App wiring.
  void onExecute
  void onOpenCopilot
  const { isConnected, lastMessage } = useWebSocket('/ws')
  const [wsMarkets, setWsMarkets] = useState<CryptoMarket[] | null>(null)
  const [wsMarketsUpdatedAtMs, setWsMarketsUpdatedAtMs] = useState<number | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())

  // Listen for real-time WebSocket pushes
  useEffect(() => {
    if (lastMessage?.type === 'crypto_markets_update' && Array.isArray(lastMessage.data?.markets)) {
      setWsMarkets(lastMessage.data.markets)
      setWsMarketsUpdatedAtMs(Date.now())
      return
    }

    if (lastMessage?.type === 'init') {
      const seededMarkets = extractCryptoMarketsFromInit(lastMessage.data)
      if (seededMarkets) {
        setWsMarkets(seededMarkets)
        setWsMarketsUpdatedAtMs(Date.now())
      }
    }
  }, [lastMessage])

  // If websocket disconnects, immediately stop trusting any stale ws cache.
  useEffect(() => {
    if (!isConnected) {
      setWsMarkets(null)
      setWsMarketsUpdatedAtMs(null)
    }
  }, [isConnected])

  useEffect(() => {
    const iv = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(iv)
  }, [])

  useEffect(() => {
    const onVisibilityChange = () => {
      setIsDocumentVisible(document.visibilityState === 'visible')
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [])

  useEffect(() => {
    const el = panelRef.current
    if (!el) return
    if (typeof window === 'undefined' || !('IntersectionObserver' in window)) {
      setIsPanelInViewport(true)
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0]
        setIsPanelInViewport(Boolean(entry?.isIntersecting))
      },
      { threshold: 0.05 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const isViewerActive = isDocumentVisible && isPanelInViewport

  const hasFreshWsMarkets =
    !!wsMarkets &&
    wsMarketsUpdatedAtMs !== null &&
    nowMs - wsMarketsUpdatedAtMs <= WS_MARKETS_STALE_MS

  // HTTP polling as fallback only
  const { data: httpMarkets, isLoading } = useQuery({
    queryKey: ['crypto-live-markets'],
    queryFn: () => getCryptoMarkets({ viewer_active: true }),
    enabled: isViewerActive,
    refetchInterval: isViewerActive ? (hasFreshWsMarkets ? 5000 : 2000) : false,
    refetchIntervalInBackground: false,
    staleTime: 1000,
  })

  const allMarkets = hasFreshWsMarkets
    ? (wsMarkets || httpMarkets || [])
    : (httpMarkets || wsMarkets || [])

  const timeframeCounts = useMemo(() => {
    const counts: Record<'5m' | '15m' | '1h' | '4h', number> = {
      '5m': 0,
      '15m': 0,
      '1h': 0,
      '4h': 0,
    }

    for (const market of allMarkets) {
      const normalized = normalizeTimeframe(market.timeframe)
      if ((['5m', '15m', '1h', '4h'] as const).includes(normalized as any)) {
        counts[normalized as '5m' | '15m' | '1h' | '4h'] += 1
      }
    }

    return counts
  }, [allMarkets])

  const filteredMarkets = useMemo(() => {
    if (timeframeFilter === 'all') return allMarkets
    return allMarkets.filter((market) => normalizeTimeframe(market.timeframe) === timeframeFilter)
  }, [timeframeFilter, allMarkets])

  // Stats from series data
  const stats = useMemo(() => {
    const live = filteredMarkets.filter(m => m.is_live).length
    const totalLiquidity = filteredMarkets.reduce((acc, m) => acc + (m.series_liquidity || m.liquidity || 0), 0)
    const totalVolume24h = filteredMarkets.reduce((acc, m) => acc + (m.series_volume_24h || 0), 0)
    const avgSpread = filteredMarkets.length > 0
      ? filteredMarkets.reduce((acc, m) => acc + (1 - (m.combined ?? 1)), 0) / filteredMarkets.length
      : 0
    return { total: filteredMarkets.length, live, totalLiquidity, totalVolume24h, avgSpread }
  }, [filteredMarkets])

  return (
    <div ref={panelRef} className="space-y-4">
      {/* Header */}
      <div className="rounded-xl border border-border/40 bg-card/60 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <div className="relative">
              <ArrowUpDown className="w-4 h-4 text-orange-400" />
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            </div>
              <span className="text-sm font-semibold text-foreground">Crypto 5m/15m/1h/4h Markets</span>
            <Badge variant="outline" className="text-[9px] text-orange-400 border-orange-500/20 bg-orange-500/10">
              LIVE
            </Badge>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            <div className="flex items-center rounded-lg border border-border/50 overflow-hidden p-0.5 bg-card/70">
              {([
                { label: 'All', value: 'all', count: allMarkets.length },
                { label: '5m', value: '5m', count: timeframeCounts['5m'] },
                { label: '15m', value: '15m', count: timeframeCounts['15m'] },
                { label: '1h', value: '1h', count: timeframeCounts['1h'] },
                { label: '4h', value: '4h', count: timeframeCounts['4h'] },
              ] as Array<{ label: string; value: TimeframeFilter; count: number }>).map((option) => (
                <button
                  key={option.value}
                  onClick={() => setTimeframeFilter(option.value)}
                  type="button"
                  className={cn(
                    "px-2.5 py-1 text-xs font-medium transition-colors",
                    timeframeFilter === option.value
                      ? "bg-primary/20 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
                  )}
                >
                  {option.label} ({option.count})
                </button>
              ))}
            </div>
                {onOpenCryptoSettings && (
                  <Button size="sm" variant="outline" onClick={onOpenCryptoSettings} className="h-7 px-2.5 text-xs gap-1.5">
                <Settings className="w-3.5 h-3.5" />
                Settings
              </Button>
            )}
            <div className="text-[11px] text-muted-foreground font-data flex items-center gap-1.5">
              {!isViewerActive ? (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50" />
                  Updates paused (view hidden)
                </>
              ) : isConnected ? (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  {hasFreshWsMarkets ? 'Real-time via WebSocket' : 'WebSocket connected'}
                </>
              ) : (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
                  Polling fallback (2s)
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {[
          { label: 'Markets', value: <span className="text-lg font-bold font-data text-foreground">{stats.total}</span>, sub: `${stats.live} live` },
          { label: 'Series Liquidity', value: <span className="text-lg font-bold font-data text-foreground">{formatUsd(stats.totalLiquidity)}</span>, sub: 'all assets' },
          { label: 'Series 24h Vol', value: <span className="text-lg font-bold font-data text-foreground">{formatUsd(stats.totalVolume24h)}</span>, sub: 'combined' },
          { label: 'Avg Spread', value: <span className={cn("text-lg font-bold font-data", stats.avgSpread > 0.005 ? 'text-green-400' : 'text-muted-foreground')}>{(stats.avgSpread * 100).toFixed(2)}%</span>, sub: 'from $1.00' },
          { label: 'Taker Fee', value: <span className="text-lg font-bold font-data text-muted-foreground">1.56%</span>, sub: 'max at 50%' },
        ].map((stat, i) => (
          <Card key={i} className="rounded-lg border border-border/40 bg-card/40 p-3">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{stat.label}</div>
            <div>{stat.value}</div>
            <div className="text-[10px] text-muted-foreground/60 mt-0.5">{stat.sub}</div>
          </Card>
        ))}
      </div>

      {/* Content */}
      {isLoading && !hasFreshWsMarkets ? (
        <div className="flex items-center justify-center py-16">
          <RefreshCw className="w-8 h-8 animate-spin text-orange-400" />
          <span className="ml-3 text-muted-foreground">Loading crypto markets...</span>
        </div>
      ) : filteredMarkets.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16">
          <Activity className="w-12 h-12 text-muted-foreground/30 mb-4" />
          <p className="text-muted-foreground font-medium">
            No live crypto markets found for this timeframe
          </p>
          <p className="text-sm text-muted-foreground/70 mt-1">
            Check that series IDs are configured correctly in Settings
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {filteredMarkets.map((market) => (
            <CryptoMarketCard key={market.id} market={market} />
          ))}
        </div>
      )}
    </div>
  )
}
