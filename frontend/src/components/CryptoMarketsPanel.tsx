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
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import { getCryptoMarkets, CryptoMarket } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import Sparkline from './Sparkline'

// ─── Constants ────────────────────────────────────────────

const ASSETS = ['ALL', 'BTC', 'ETH', 'SOL', 'XRP'] as const
type Asset = typeof ASSETS[number]

const ASSET_COLORS: Record<string, string> = {
  BTC: 'text-orange-400',
  ETH: 'text-blue-400',
  SOL: 'text-purple-400',
  XRP: 'text-cyan-400',
}

const ASSET_BG: Record<string, string> = {
  BTC: 'bg-orange-500/10 border-orange-500/20',
  ETH: 'bg-blue-500/10 border-blue-500/20',
  SOL: 'bg-purple-500/10 border-purple-500/20',
  XRP: 'bg-cyan-500/10 border-cyan-500/20',
}

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

function OraclePriceDisplay({ price, priceToBeat }: { price: number | null; priceToBeat: number | null }) {
  if (price === null) return null

  const delta = (priceToBeat !== null && priceToBeat !== undefined) ? price - priceToBeat : null
  const isUp = delta !== null && delta >= 0

  return (
    <div className="space-y-1.5">
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
  const upPrice = market.up_price ?? 0.5
  const downPrice = market.down_price ?? 0.5
  const combined = market.combined ?? (upPrice + downPrice)
  const spread = 1 - combined

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
        <OraclePriceDisplay price={market.oracle_price} priceToBeat={market.price_to_beat} />

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
          <div className={cn("rounded-lg p-2.5 text-center border", upPrice > downPrice ? 'bg-green-500/10 border-green-500/20' : 'bg-muted/20 border-border/30')}>
            <div className="text-[10px] text-green-400 uppercase tracking-wider font-medium mb-1">
              <TrendingUp className="w-3 h-3 inline mr-1" />Up
            </div>
            <div className="text-lg font-bold font-data tabular-nums text-green-400">
              {(upPrice * 100).toFixed(0)}%
            </div>
            <div className="text-[10px] text-muted-foreground font-data">${upPrice.toFixed(2)}</div>
          </div>
          <div className={cn("rounded-lg p-2.5 text-center border", downPrice > upPrice ? 'bg-red-500/10 border-red-500/20' : 'bg-muted/20 border-border/30')}>
            <div className="text-[10px] text-red-400 uppercase tracking-wider font-medium mb-1">
              <TrendingDown className="w-3 h-3 inline mr-1" />Down
            </div>
            <div className="text-lg font-bold font-data tabular-nums text-red-400">
              {(downPrice * 100).toFixed(0)}%
            </div>
            <div className="text-[10px] text-muted-foreground font-data">${downPrice.toFixed(2)}</div>
          </div>
        </div>

        {/* Spread / Combined info */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Combined Cost</span>
            <div className="flex items-center gap-2">
              <span className="font-data font-bold text-foreground">${combined.toFixed(3)}</span>
              {spread > 0.001 && (
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
}

export default function CryptoMarketsPanel({ onExecute, onOpenCopilot }: Props) {
  // Intentionally kept for interface parity with other panels and App wiring.
  void onExecute
  void onOpenCopilot
  const { isConnected, lastMessage } = useWebSocket('/ws')
  const [selectedAsset, setSelectedAsset] = useState<Asset>('ALL')
  const [wsMarkets, setWsMarkets] = useState<CryptoMarket[] | null>(null)

  // Listen for real-time WebSocket pushes
  useEffect(() => {
    if (lastMessage?.type === 'crypto_markets_update' && lastMessage.data?.markets) {
      setWsMarkets(lastMessage.data.markets)
    }
  }, [lastMessage])

  // HTTP polling as fallback only
  const { data: httpMarkets, isLoading } = useQuery({
    queryKey: ['crypto-live-markets'],
    queryFn: getCryptoMarkets,
    refetchInterval: isConnected ? 30000 : 5000,
    staleTime: 2000,
  })

  const allMarkets = wsMarkets || httpMarkets || []

  // Filter by asset
  const filtered = useMemo(() => {
    if (selectedAsset === 'ALL') return allMarkets
    return allMarkets.filter(m => m.asset === selectedAsset)
  }, [allMarkets, selectedAsset])

  // Stats from series data
  const stats = useMemo(() => {
    const live = allMarkets.filter(m => m.is_live).length
    const totalLiquidity = allMarkets.reduce((acc, m) => acc + (m.series_liquidity || m.liquidity || 0), 0)
    const totalVolume24h = allMarkets.reduce((acc, m) => acc + (m.series_volume_24h || 0), 0)
    const avgSpread = allMarkets.length > 0
      ? allMarkets.reduce((acc, m) => acc + (1 - (m.combined ?? 1)), 0) / allMarkets.length
      : 0
    return { total: allMarkets.length, live, totalLiquidity, totalVolume24h, avgSpread }
  }, [allMarkets])

  return (
    <div className="space-y-4">
      {/* Header Dashboard */}
      <div className="rounded-lg border border-border/50 bg-card/60 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/30">
          <div className="flex items-center gap-2">
            <div className="relative">
              <ArrowUpDown className="w-4 h-4 text-orange-400" />
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            </div>
            <span className="text-sm font-semibold text-foreground">Crypto 15-Min Markets</span>
            <Badge variant="outline" className="text-[9px] text-orange-400 border-orange-500/20 bg-orange-500/10">
              LIVE
            </Badge>
          </div>
          <div className="text-[10px] text-muted-foreground font-data flex items-center gap-1.5">
            {isConnected && wsMarkets ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Real-time via WebSocket
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
                Polling
              </>
            )}
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-5 divide-x divide-border/20">
          {[
            { label: 'Markets', value: <span className="text-sm font-bold font-data text-foreground">{stats.total}</span>, sub: `${stats.live} live` },
            { label: 'Series Liquidity', value: <span className="text-sm font-bold font-data text-foreground">{formatUsd(stats.totalLiquidity)}</span>, sub: 'all assets' },
            { label: 'Series 24h Vol', value: <span className="text-sm font-bold font-data text-foreground">{formatUsd(stats.totalVolume24h)}</span>, sub: 'combined' },
            { label: 'Avg Spread', value: <span className={cn("text-sm font-bold font-data", stats.avgSpread > 0.005 ? 'text-green-400' : 'text-muted-foreground')}>{(stats.avgSpread * 100).toFixed(2)}%</span>, sub: 'from $1.00' },
            { label: 'Taker Fee', value: <span className="text-sm font-bold font-data text-muted-foreground">1.56%</span>, sub: 'max at 50%' },
          ].map((stat, i) => (
            <div key={i} className="px-3 py-2.5 text-center">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider mb-1">{stat.label}</div>
              <div>{stat.value}</div>
              <div className="text-[9px] text-muted-foreground/60 mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Asset filter */}
      <div className="flex items-center gap-0.5 border border-border/50 rounded-lg p-0.5 bg-card/50 w-fit">
        {ASSETS.map((asset) => {
          const count = asset === 'ALL' ? allMarkets.length : allMarkets.filter(m => m.asset === asset).length
          return (
            <button
              key={asset}
              onClick={() => setSelectedAsset(asset)}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
                selectedAsset === asset
                  ? asset === 'ALL'
                    ? 'bg-primary/20 text-primary'
                    : cn('bg-opacity-20', ASSET_BG[asset], ASSET_COLORS[asset])
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              )}
            >
              {asset}
              {count > 0 && <span className="text-[9px] font-data bg-muted/50 px-1 rounded">{count}</span>}
            </button>
          )
        })}
      </div>

      {/* Content */}
      {isLoading && !wsMarkets ? (
        <div className="flex items-center justify-center py-16">
          <RefreshCw className="w-8 h-8 animate-spin text-orange-400" />
          <span className="ml-3 text-muted-foreground">Loading crypto markets...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16">
          <Activity className="w-12 h-12 text-muted-foreground/30 mb-4" />
          <p className="text-muted-foreground font-medium">No live crypto markets found</p>
          <p className="text-sm text-muted-foreground/70 mt-1">
            Check that series IDs are configured correctly in Settings
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {filtered.map((market) => (
            <CryptoMarketCard key={market.id} market={market} />
          ))}
        </div>
      )}
    </div>
  )
}
