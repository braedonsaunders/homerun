import { useState, useEffect, useMemo, useRef } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { useAtomValue } from 'jotai'
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ExternalLink,
  ChevronRight,
  Settings,
  Maximize2,
  Minimize2,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import { getCryptoMarkets, CryptoMarket } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { Liveline } from 'liveline'
import type { LivelinePoint } from 'liveline'
import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import Sparkline from './Sparkline'
import OpportunityEmptyState from './OpportunityEmptyState'
import { themeAtom } from '../store/atoms'

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

function toUnixSeconds(value: number): number {
  if (value > 1_000_000_000_000) return Math.floor(value / 1000)
  if (value > 10_000_000_000) return Math.floor(value / 1000)
  return Math.floor(value)
}

function clampProbability(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function polymarketTakerFeePerShare(price: number): number {
  const p = clampProbability(price)
  return p * 0.25 * (p * (1 - p)) ** 2
}

function polymarketTakerFeePct(price: number): number {
  const p = clampProbability(price)
  if (p <= 0) return 0
  return polymarketTakerFeePerShare(p) / p
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

function marketIdentity(market: Partial<CryptoMarket>): string {
  const id = String(market.id || '').trim()
  if (id) return `id:${id}`
  const conditionId = String(market.condition_id || '').trim()
  if (conditionId) return `condition:${conditionId}`
  const slug = String(market.slug || '').trim()
  if (slug) return `slug:${slug}`
  const asset = String(market.asset || '').trim().toUpperCase()
  const timeframe = normalizeTimeframe(market.timeframe)
  const endTime = String(market.end_time || '').trim()
  return `fallback:${asset}:${timeframe}:${endTime}`
}

function dedupeMarkets(markets: CryptoMarket[]): CryptoMarket[] {
  if (markets.length <= 1) return markets
  const byKey = new Map<string, CryptoMarket>()
  for (const market of markets) {
    byKey.set(marketIdentity(market), market)
  }
  return Array.from(byKey.values())
}

function mergeMarkets(base: CryptoMarket[], updates: CryptoMarket[], appendUnknown = true): CryptoMarket[] {
  if (updates.length === 0) return base
  if (base.length === 0) return dedupeMarkets(updates)

  const merged = [...base]
  const indexByKey = new Map<string, number>()
  for (let i = 0; i < merged.length; i += 1) {
    indexByKey.set(marketIdentity(merged[i]), i)
  }

  for (const market of updates) {
    const key = marketIdentity(market)
    const index = indexByKey.get(key)
    if (index === undefined) {
      if (!appendUnknown) continue
      indexByKey.set(key, merged.length)
      merged.push(market)
      continue
    }
    merged[index] = {
      ...merged[index],
      ...market,
    }
  }

  return merged
}

function marketTakerFeePct(market: CryptoMarket): number | null {
  if (!market.fees_enabled) return 0

  const priceSamples = [toFiniteNumber(market.up_price), toFiniteNumber(market.down_price)]
    .filter((v): v is number => v !== null)
    .map((price) => polymarketTakerFeePct(price))

  if (priceSamples.length > 0) {
    return priceSamples.reduce((acc, value) => acc + value, 0) / priceSamples.length
  }

  const combined = toFiniteNumber(market.combined)
  if (combined !== null) {
    return polymarketTakerFeePct(clampProbability(combined / 2))
  }

  return null
}

function resolveCryptoStrategySdk(market: CryptoMarket): string {
  const explicit = String(
    market.strategy_sdk
    || market.strategy_key
    || market.strategy
    || '',
  ).trim().toLowerCase()
  if (explicit) return explicit
  return 'btc_eth_highfreq'
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

function CryptoMarketCard({
  market,
  themeMode,
  isModalView = false,
  onCloseModal,
}: {
  market: CryptoMarket
  themeMode: 'dark' | 'light'
  isModalView?: boolean
  onCloseModal?: () => void
}) {
  const chartRef = useRef<HTMLDivElement>(null)
  const [chartWidth, setChartWidth] = useState(300)
  const [modalOpen, setModalOpen] = useState(false)
  const closeModal = () => setModalOpen(false)

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

  useEffect(() => {
    if (isModalView || !modalOpen) return undefined
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [isModalView, modalOpen])

  useEffect(() => {
    if (isModalView || !modalOpen) return undefined
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeModal()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isModalView, modalOpen])

  const asset = market.asset
  const upPrice = toFiniteNumber(market.up_price)
  const downPrice = toFiniteNumber(market.down_price)
  const combined = toFiniteNumber(market.combined) ?? (
    upPrice !== null && downPrice !== null ? upPrice + downPrice : null
  )
  const spread = combined !== null ? 1 - combined : null
  const takerFeePct = marketTakerFeePct(market)
  const isDarkTheme = themeMode === 'dark'
  const strategySdk = resolveCryptoStrategySdk(market)

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

  const livelineData = useMemo<LivelinePoint[]>(() => {
    const raw = Array.isArray(market.oracle_history) ? market.oracle_history : []
    const rawPoints = raw
      .map((point) => {
        const value = toFiniteNumber((point as { p?: unknown; price?: unknown })?.p ?? (point as { price?: unknown })?.price)
        const rawTime = toFiniteNumber((point as { t?: unknown; time?: unknown })?.t ?? (point as { time?: unknown })?.time)
        if (value === null || rawTime === null) return null
        return {
          time: toUnixSeconds(rawTime),
          value,
        }
      })
      .filter((point): point is LivelinePoint => point !== null)
      .sort((a, b) => a.time - b.time)

    const current = toFiniteNumber(market.oracle_price)
    const now = Math.floor(Date.now() / 1000)
    if (rawPoints.length === 0) {
      if (current === null) return []
      return [
        { time: now - 1, value: current },
        { time: now, value: current },
      ]
    }

    const normalized: LivelinePoint[] = []
    for (const point of rawPoints) {
      const previous = normalized[normalized.length - 1]
      const safeTime = previous ? Math.max(point.time, previous.time + 1) : point.time
      normalized.push({ time: safeTime, value: point.value })
    }

    if (current !== null) {
      const last = normalized[normalized.length - 1]
      const shouldAppendCurrent = Math.abs(last.value - current) > 1e-9 || now > last.time
      if (shouldAppendCurrent) {
        normalized.push({
          time: Math.max(now, last.time + 1),
          value: current,
        })
      }
    }

    const MAX_POINTS = 600
    return normalized.length > MAX_POINTS
      ? normalized.slice(normalized.length - MAX_POINTS)
      : normalized
  }, [market.oracle_history, market.oracle_price])

  const livelineValue = (
    toFiniteNumber(market.oracle_price)
    ?? livelineData[livelineData.length - 1]?.value
    ?? 0
  )
  const livelineWindow = (() => {
    const tf = normalizeTimeframe(market.timeframe)
    if (tf === '5m') return 300
    if (tf === '15m') return 900
    if (tf === '1h') return 3600
    if (tf === '4h') return 14_400
    return 900
  })()
  const livelineColor = (
    market.oracle_price !== null && market.price_to_beat !== null
      ? (
        market.oracle_price >= market.price_to_beat
          ? (isDarkTheme ? '#22c55e' : '#16a34a')
          : (isDarkTheme ? '#f87171' : '#dc2626')
      )
      : (isDarkTheme ? '#60a5fa' : '#2563eb')
  )

  // Parse time window from title (e.g. "Bitcoin Up or Down - February 10, 10:45AM-11:00AM ET")
  const timeWindow = market.event_title?.match(/(\d{1,2}:\d{2}[AP]M)-(\d{1,2}:\d{2}[AP]M)\s*ET/)?.[0] || ''

  return (
    <>
    <Card className={cn(
      "overflow-hidden relative group transition-all duration-200",
      !isModalView && "hover:shadow-lg hover:shadow-black/20 hover:border-border/80",
      isModalView && "w-[min(1100px,calc(100vw-2rem))] max-h-[90vh] overflow-y-auto rounded-2xl border-border/70 bg-background shadow-[0_40px_120px_rgba(0,0,0,0.55)] [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden",
      market.is_live && 'ring-1 ring-green-500/10',
    )}>
      {/* Asset color accent bar */}
      <div className={cn("absolute left-0 top-0 bottom-0 w-1.5 rounded-l-lg", ASSET_BAR[asset] || 'bg-gray-400')} />

      <div className="pl-5 pr-4 py-4 space-y-3">
        {/* Header: Asset icon + name + status */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-start gap-3">
            <img src={ASSET_ICONS[asset]} alt={asset} className="w-8 h-8 rounded-full shrink-0" onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }} />
            <div className="min-w-0">
              <h3 className="text-base font-semibold text-foreground truncate">{asset} Up or Down</h3>
              <div className="mt-0.5 flex items-center gap-1.5 flex-wrap">
                <Badge variant="outline" className="text-[9px] px-1.5 py-0 text-muted-foreground border-muted-foreground/20">
                  {market.timeframe.toUpperCase()}
                </Badge>
                <Badge
                  variant="outline"
                  className="max-w-[170px] truncate text-[9px] px-1.5 py-0 font-mono border-border/50 bg-muted/25 text-muted-foreground"
                  title={`StrategySDK: ${strategySdk}`}
                >
                  SDK {strategySdk}
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
              <p className="text-[11px] text-muted-foreground font-data truncate">{timeWindow}</p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            {!isModalView ? (
              <button
                type="button"
                onClick={() => setModalOpen(true)}
                className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border/50 bg-background/40 text-muted-foreground transition-colors hover:border-border hover:bg-background/70 hover:text-foreground"
                title="Expand this card"
              >
                <Maximize2 className="w-2.5 h-2.5" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => onCloseModal?.()}
                className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border/50 bg-background/40 text-muted-foreground transition-colors hover:border-border hover:bg-background/70 hover:text-foreground"
                title="Return to grid"
              >
                <Minimize2 className="w-2.5 h-2.5" />
              </button>
            )}
            {polyUrl && (
              <a href={polyUrl} target="_blank" rel="noopener noreferrer"
                className="text-muted-foreground hover:text-foreground transition-colors p-1">
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
          </div>
        </div>

        {/* Oracle price + Price to beat */}
        <OraclePriceDisplay
          price={market.oracle_price}
          priceToBeat={market.price_to_beat}
          source={market.oracle_source}
          sourceMap={market.oracle_prices_by_source}
        />

        {/* Oracle price sparkline chart */}
        {isModalView ? (
          <div className={cn(
            "relative h-56 w-full rounded-lg overflow-hidden border",
            isDarkTheme
              ? "border-slate-700/40 bg-gradient-to-b from-slate-900/75 via-slate-950/80 to-black/90"
              : "border-slate-200/90 bg-gradient-to-b from-white via-slate-50 to-slate-100/70",
          )}>
            {livelineData.length >= 2 ? (
              <Liveline
                data={livelineData}
                value={livelineValue}
                color={livelineColor}
                theme={isDarkTheme ? 'dark' : 'light'}
                showValue
                valueMomentumColor
                grid
                badge
                badgeVariant={isDarkTheme ? 'default' : 'minimal'}
                badgeTail={isDarkTheme}
                pulse
                fill
                window={livelineWindow}
                lerpSpeed={0.1}
                padding={{ top: 16, right: 84, bottom: 30, left: 14 }}
                tooltipOutline={isDarkTheme}
                formatValue={(value) => formatPrice(value, 2)}
                referenceLine={market.price_to_beat !== null ? { value: market.price_to_beat, label: 'Price to beat' } : undefined}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-[10px] text-muted-foreground/40">
                Waiting for price data...
              </div>
            )}
          </div>
        ) : (
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
        )}

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
        <div className="grid grid-cols-4 gap-2 text-center">
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Liquidity</div>
            <div className="text-xs font-bold font-data text-foreground">{formatUsd(market.liquidity)}</div>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Volume</div>
            <div className="text-xs font-bold font-data text-foreground">{formatUsd(market.volume)}</div>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Taker Fee</div>
            <div className={cn(
              "text-xs font-bold font-data",
              takerFeePct !== null && takerFeePct > 0 ? 'text-orange-400' : 'text-muted-foreground',
            )}>
              {takerFeePct !== null ? `${(takerFeePct * 100).toFixed(2)}%` : '--'}
            </div>
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
      {!isModalView && typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {modalOpen && (
            <motion.div
              key={`crypto-market-modal-${market.id}`}
              className="fixed inset-0 z-[120] flex items-center justify-center p-4 sm:p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={closeModal}
                aria-hidden
              />
              <motion.div
                className="relative z-10"
                role="dialog"
                aria-modal="true"
                aria-label={`Expanded crypto market: ${asset} ${market.timeframe}`}
                initial={{ scale: 0.94, opacity: 0, y: 22 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 14 }}
                transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}
              >
                <CryptoMarketCard
                  market={market}
                  themeMode={themeMode}
                  isModalView
                  onCloseModal={closeModal}
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </>
  )
}

// ─── Main Panel ──────────────────────────────────────────

interface Props {
  onExecute?: (opportunity: any) => void
  onOpenCopilot?: (opportunity: any) => void
  onOpenCryptoSettings?: () => void
  showSettingsButton?: boolean
}

export default function CryptoMarketsPanel({
  onExecute,
  onOpenCopilot,
  onOpenCryptoSettings,
  showSettingsButton = true,
}: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const themeMode = useAtomValue(themeAtom)
  const [timeframeFilter, setTimeframeFilter] = useState<TimeframeFilter>('all')
  const [isDocumentVisible, setIsDocumentVisible] = useState(
    () => (typeof document === 'undefined' ? true : document.visibilityState === 'visible')
  )
  const [isPanelInViewport, setIsPanelInViewport] = useState(true)
  // Intentionally kept for interface parity with other panels and App wiring.
  void onExecute
  void onOpenCopilot
  const { isConnected, lastMessage } = useWebSocket('/ws')
  const [wsMarkets, setWsMarkets] = useState<CryptoMarket[]>([])
  const [wsMarketsUpdatedAtMs, setWsMarketsUpdatedAtMs] = useState<number | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())

  // Listen for real-time WebSocket pushes
  useEffect(() => {
    if (lastMessage?.type === 'crypto_markets_update' && Array.isArray(lastMessage.data?.markets)) {
      setWsMarkets((current) => mergeMarkets(current, dedupeMarkets(lastMessage.data.markets as CryptoMarket[])))
      setWsMarketsUpdatedAtMs(Date.now())
      return
    }

    if (lastMessage?.type === 'init') {
      const seededMarkets = extractCryptoMarketsFromInit(lastMessage.data)
      if (seededMarkets) {
        setWsMarkets((current) => mergeMarkets(current, dedupeMarkets(seededMarkets)))
        setWsMarketsUpdatedAtMs(Date.now())
      }
    }
  }, [lastMessage])

  // If websocket disconnects, immediately stop trusting any stale ws cache.
  useEffect(() => {
    if (!isConnected) {
      setWsMarkets([])
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
    wsMarkets.length > 0 &&
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

  useEffect(() => {
    if (!httpMarkets || httpMarkets.length === 0) return
    const snapshotKeys = new Set(dedupeMarkets(httpMarkets).map((market) => marketIdentity(market)))
    setWsMarkets((current) => current.filter((market) => snapshotKeys.has(marketIdentity(market))))
  }, [httpMarkets])

  const allMarkets = useMemo(() => {
    const snapshotMarkets = dedupeMarkets(httpMarkets || [])
    if (snapshotMarkets.length === 0) return wsMarkets
    return mergeMarkets(snapshotMarkets, wsMarkets, false)
  }, [httpMarkets, wsMarkets])

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
    const seriesByKey = new Map<string, { liquidity: number; volume24h: number }>()
    filteredMarkets.forEach((market) => {
      const tf = normalizeTimeframe(market.timeframe) || String(market.timeframe || '').toLowerCase()
      const key = `${market.asset}:${tf}`
      if (seriesByKey.has(key)) return
      seriesByKey.set(key, {
        liquidity: market.series_liquidity || market.liquidity || 0,
        volume24h: market.series_volume_24h || market.volume_24h || market.volume || 0,
      })
    })

    const totalLiquidity = Array.from(seriesByKey.values()).reduce((acc, row) => acc + row.liquidity, 0)
    const totalVolume24h = Array.from(seriesByKey.values()).reduce((acc, row) => acc + row.volume24h, 0)

    const spreadSamples = filteredMarkets
      .map((market) => toFiniteNumber(market.combined))
      .filter((combined): combined is number => combined !== null)
      .map((combined) => Math.abs(1 - combined))
    const avgSpread = spreadSamples.length > 0
      ? spreadSamples.reduce((acc, value) => acc + value, 0) / spreadSamples.length
      : 0

    const takerFeeSamples = filteredMarkets
      .map((market) => marketTakerFeePct(market))
      .filter((fee): fee is number => fee !== null)
    const avgTakerFeePct = takerFeeSamples.length > 0
      ? takerFeeSamples.reduce((acc, fee) => acc + fee, 0) / takerFeeSamples.length
      : null
    const maxTakerFeePct = takerFeeSamples.length > 0
      ? Math.max(...takerFeeSamples)
      : null

    return {
      total: filteredMarkets.length,
      live,
      totalLiquidity,
      totalVolume24h,
      avgSpread,
      spreadSampleCount: spreadSamples.length,
      seriesCount: seriesByKey.size,
      avgTakerFeePct,
      maxTakerFeePct,
      takerFeeSampleCount: takerFeeSamples.length,
    }
  }, [filteredMarkets])

  const wsStatus = !isViewerActive
    ? {
      label: 'Updates Paused',
      toneClass: 'text-muted-foreground border-border/50 bg-card/70',
    }
    : isConnected
      ? hasFreshWsMarkets
        ? {
          label: 'WebSocket Live',
          toneClass: 'text-green-400 border-green-500/30 bg-green-500/10',
        }
        : {
          label: 'WebSocket Connected',
          toneClass: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
        }
      : {
        label: 'Polling 2s',
        toneClass: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10',
      }

  return (
    <div ref={panelRef} className="space-y-4">
      {/* Header */}
      <div className="rounded-xl border border-border/40 bg-card/60 px-3 py-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
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
          <div className="flex items-center gap-2 flex-wrap justify-end">
            {showSettingsButton && onOpenCryptoSettings && (
              <Button size="sm" variant="outline" onClick={onOpenCryptoSettings} className="h-7 px-2.5 text-xs gap-1.5">
                <Settings className="w-3.5 h-3.5" />
                Settings
              </Button>
            )}
            <Button type="button" size="sm" variant="outline" className={cn("h-7 px-2.5 text-xs", wsStatus.toneClass)}>
              {wsStatus.label}
            </Button>
          </div>
        </div>

        <div className="mt-2 border-t border-border/30 pt-2">
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 md:grid-cols-5">
            <div className="min-w-0">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Markets</div>
              <div className="text-sm font-bold font-data leading-tight text-foreground">
                {stats.total}
                <span className="ml-1 text-[10px] font-medium text-muted-foreground">{stats.live} live</span>
              </div>
            </div>
            <div className="min-w-0">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Series Liquidity</div>
              <div className="text-sm font-bold font-data leading-tight text-foreground">
                {formatUsd(stats.totalLiquidity)}
                <span className="ml-1 text-[10px] font-medium text-muted-foreground">{stats.seriesCount} series</span>
              </div>
            </div>
            <div className="min-w-0">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Series 24h Vol</div>
              <div className="text-sm font-bold font-data leading-tight text-foreground">
                {formatUsd(stats.totalVolume24h)}
                <span className="ml-1 text-[10px] font-medium text-muted-foreground">{stats.seriesCount} series</span>
              </div>
            </div>
            <div className="min-w-0">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Avg Spread</div>
              <div className={cn("text-sm font-bold font-data leading-tight", stats.avgSpread > 0.005 ? 'text-green-400' : 'text-muted-foreground')}>
                {(stats.avgSpread * 100).toFixed(2)}%
                <span className="ml-1 text-[10px] font-medium text-muted-foreground">{stats.spreadSampleCount} mkts</span>
              </div>
            </div>
            <div className="min-w-0">
              <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Taker Fee</div>
              <div className={cn("text-sm font-bold font-data leading-tight", stats.avgTakerFeePct !== null && stats.avgTakerFeePct > 0 ? 'text-orange-400' : 'text-muted-foreground')}>
                {stats.avgTakerFeePct !== null ? `${(stats.avgTakerFeePct * 100).toFixed(2)}%` : '--'}
                {stats.maxTakerFeePct !== null && (
                  <span className="ml-1 text-[10px] font-medium text-muted-foreground">
                    max {(stats.maxTakerFeePct * 100).toFixed(2)}%
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      {isLoading && !hasFreshWsMarkets ? (
        <div className="flex items-center justify-center py-16">
          <RefreshCw className="w-8 h-8 animate-spin text-orange-400" />
          <span className="ml-3 text-muted-foreground">Loading crypto markets...</span>
        </div>
      ) : filteredMarkets.length === 0 ? (
        <OpportunityEmptyState
          title={
            timeframeFilter === 'all'
              ? 'No executable crypto opportunities found'
              : 'No crypto opportunities found for this timeframe'
          }
          description={
            timeframeFilter === 'all'
              ? 'Try waiting for new windows or verify series IDs in Settings'
              : 'Try switching timeframe filters or wait for new windows'
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {filteredMarkets.map((market) => (
            <CryptoMarketCard key={market.id} market={market} themeMode={themeMode} />
          ))}
        </div>
      )}
    </div>
  )
}
