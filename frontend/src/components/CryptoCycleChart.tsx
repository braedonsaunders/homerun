/**
 * Per-cycle synchronized chart for crypto up/down markets.
 *
 * Single canvas that overlays, in cycle-relative time:
 *   - Oracle price line (sourced from market.oracle_history + live updates)
 *   - UP-share mid price line (cents, right axis, frontend rolling buffer)
 *   - DOWN-share mid price line (cents, right axis, frontend rolling buffer)
 *   - Tracked-wallet trade markers (BUY=▲, SELL=▼; UP=green, DOWN=red)
 *
 * Why a custom canvas instead of stacking Liveline instances: trade markers
 * and dual-axis rendering aren't first-class in Liveline. Pattern follows
 * UpDownWalletMonitor's IndicatorChart but bound to homerun's CryptoMarket
 * shape and React Query data layer.
 *
 * NOTE: this component renders correctly under code review, but the visual
 * layout and styling will likely need tweaking once exercised in the
 * browser (line weights, axis labels, marker sizes). Treat the first
 * deploy as a v0 and iterate.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { api } from '../services/apiClient'
import type { CryptoMarket } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'

type Theme = 'dark' | 'light'

interface MarketWalletTradesResponse {
  condition_id: string
  up_token_id: string | null
  down_token_id: string | null
  trades: WalletTradeMarker[]
}

export interface WalletTradeMarker {
  id: string
  wallet: string
  token_id: string
  outcome: 'UP' | 'DOWN'
  side: 'BUY' | 'SELL'
  price: number | null
  size: number | null
  tx_hash: string | null
  block_number: number | null
  detected_at_ms: number | null
  // Optional. Populated by the WS push only — REST baseline events are
  // implicitly confirmed/onchain. ``confirmed === false`` markers are
  // drawn hollow to indicate "pending block confirmation."
  confirmed?: boolean
  source?: 'onchain' | 'rtds'
}

interface Point {
  t: number // unix seconds
  v: number
}

const PALETTE = {
  dark: {
    background: '#0a0a0e',
    grid: 'rgba(255,255,255,0.05)',
    text: 'rgba(236,236,243,0.55)',
    oracle: '#5eaaff',
    up: '#34d08c',
    down: '#ff5d73',
    tradeBuy: '#ffb347',
    tradeSell: '#ff7ab0',
    pendingMarker: 'rgba(255,255,255,0.4)',
  },
  light: {
    background: '#ffffff',
    grid: 'rgba(0,0,0,0.05)',
    text: 'rgba(20,20,30,0.55)',
    oracle: '#2563eb',
    up: '#16a34a',
    down: '#dc2626',
    tradeBuy: '#d97706',
    tradeSell: '#db2777',
    pendingMarker: 'rgba(0,0,0,0.35)',
  },
} as const

// Reject UP/DOWN mid jumps > 30¢ — same heuristic as UpDownWalletMonitor.
// A clean cycle-reset book can briefly emit (0.5, 0.5) which would crush
// the chart scale.
const SHARE_BARCODE_THRESHOLD = 0.30

function toUnixSec(value: number | null | undefined): number | null {
  if (value === null || value === undefined) return null
  if (!Number.isFinite(value)) return null
  if (value > 1_000_000_000_000) return Math.floor(value / 1000)
  return Math.floor(value)
}

function parseIso(value: string | null): number | null {
  if (!value) return null
  const ms = Date.parse(value)
  if (Number.isNaN(ms)) return null
  return Math.floor(ms / 1000)
}

export function CryptoCycleChart({
  market,
  theme = 'dark',
  height = 360,
}: {
  market: CryptoMarket
  theme?: Theme
  height?: number
}) {
  const { t } = useTranslation()
  const palette = PALETTE[theme]
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // Frontend-side rolling buffers for UP/DOWN mid history. These start
  // empty when the component mounts and accumulate as market data
  // refreshes. Persistence across mounts is intentionally out of scope —
  // a per-cycle view should reset on cycle change.
  const upHistoryRef = useRef<Point[]>([])
  const downHistoryRef = useRef<Point[]>([])

  // Cycle window in unix seconds. Stable across the cycle.
  const cycleStart = parseIso(market.start_time)
  const cycleEnd = parseIso(market.end_time)

  // Reset rolling buffers on cycle change so each cycle gets a fresh chart.
  const cycleKey = `${market.condition_id}|${cycleStart ?? '?'}`
  const lastCycleKeyRef = useRef(cycleKey)
  if (lastCycleKeyRef.current !== cycleKey) {
    upHistoryRef.current = []
    downHistoryRef.current = []
    lastCycleKeyRef.current = cycleKey
  }

  // Capture UP/DOWN mid into the rolling buffer whenever the market
  // refreshes. Reject barcode spikes — see SHARE_BARCODE_THRESHOLD.
  useEffect(() => {
    const now = Math.floor(Date.now() / 1000)
    if (
      market.up_price !== null &&
      market.up_price > 0.005 &&
      market.up_price < 0.995
    ) {
      const prev = upHistoryRef.current[upHistoryRef.current.length - 1]
      if (
        !prev ||
        Math.abs(market.up_price - prev.v) <= SHARE_BARCODE_THRESHOLD
      ) {
        upHistoryRef.current.push({ t: now, v: market.up_price })
      }
    }
    if (
      market.down_price !== null &&
      market.down_price > 0.005 &&
      market.down_price < 0.995
    ) {
      const prev = downHistoryRef.current[downHistoryRef.current.length - 1]
      if (
        !prev ||
        Math.abs(market.down_price - prev.v) <= SHARE_BARCODE_THRESHOLD
      ) {
        downHistoryRef.current.push({ t: now, v: market.down_price })
      }
    }
  }, [market.up_price, market.down_price])

  // Wallet trade markers from the backend endpoint — used as the
  // baseline / catch-up source. The polling interval is intentionally
  // long because live updates flow through the WS subscription below;
  // the poll just covers reconnects and initial mount.
  const { data: walletTradesData } = useQuery({
    queryKey: ['crypto-cycle-wallet-trades', market.condition_id, cycleStart],
    queryFn: async (): Promise<MarketWalletTradesResponse> => {
      const sinceSeconds = cycleEnd && cycleStart
        ? Math.min(14_400, Math.max(60, cycleEnd - cycleStart + 60))
        : 900
      const { data } = await api.get(
        `/crypto/markets/${encodeURIComponent(market.condition_id)}/wallet-trades`,
        { params: { since_seconds: sinceSeconds } },
      )
      return data
    },
    // Live updates come from the WS push; the poll just covers the gap
    // when the socket is offline. 30s is plenty.
    refetchInterval: market.is_live ? 30_000 : false,
    enabled: Boolean(market.condition_id) && market.clob_token_ids.length >= 2,
    staleTime: 25_000,
  })

  // Live overlay: WS-pushed wallet trade events. The backend emits a
  // ``wallet_trade_event`` message for both on-chain confirmations and
  // RTDS preliminary detections. We merge them with the polled baseline
  // by tx_hash so a trade is never plotted twice. RTDS-source events
  // arrive 1-3s ahead of block confirmation, so markers appear sub-
  // second instead of waiting for the next 30s poll.
  const { lastMessage } = useWebSocket('/ws', ['wallet_trade_event'])
  const [liveTrades, setLiveTrades] = useState<WalletTradeMarker[]>([])
  const upTokenId = walletTradesData?.up_token_id ?? null
  const downTokenId = walletTradesData?.down_token_id ?? null

  useEffect(() => {
    if (!lastMessage || lastMessage.type !== 'wallet_trade_event') return
    const data = lastMessage.data as Record<string, unknown> | null
    if (!data) return
    const tokenId = String(data.token_id || '')
    // Only keep events on this market's tokens.
    if (!tokenId || (tokenId !== upTokenId && tokenId !== downTokenId)) return
    const detectedAtMs =
      typeof data.detected_at_ms === 'number' ? data.detected_at_ms : null
    if (detectedAtMs === null) return
    const tradeAtMs =
      typeof data.trade_at_ms === 'number' ? data.trade_at_ms : detectedAtMs
    // Drop events outside the chart's cycle window with a 5s grace.
    if (cycleStart !== null && tradeAtMs < cycleStart * 1000 - 5_000) return
    if (cycleEnd !== null && tradeAtMs > cycleEnd * 1000 + 5_000) return

    const txHash = String(data.tx_hash || '')
    const sideRaw = String(data.side || '').toUpperCase()
    const side: 'BUY' | 'SELL' = sideRaw === 'SELL' ? 'SELL' : 'BUY'
    const sourceRaw = String(data.source || 'onchain')
    const source: 'onchain' | 'rtds' = sourceRaw === 'rtds' ? 'rtds' : 'onchain'
    const marker: WalletTradeMarker = {
      // Use tx_hash + token_id as stable id; falls back to a synthesized
      // key only if tx_hash is missing (RTDS pre-confirmation can
      // sometimes lack it).
      id: txHash
        ? `${tokenId}|${txHash}`
        : `${tokenId}|${data.wallet_address || ''}|${tradeAtMs}|${data.price ?? ''}`,
      wallet: String(data.wallet_address || ''),
      token_id: tokenId,
      outcome: tokenId === upTokenId ? 'UP' : 'DOWN',
      side,
      price: typeof data.price === 'number' ? data.price : null,
      size: typeof data.size === 'number' ? data.size : null,
      tx_hash: txHash || null,
      block_number:
        typeof data.block_number === 'number' && data.block_number > 0
          ? data.block_number
          : null,
      detected_at_ms: tradeAtMs,
      confirmed: data.confirmed === undefined ? true : Boolean(data.confirmed),
      source,
    }

    setLiveTrades((prev) => {
      // Dedupe by id (tx_hash-derived). On-chain confirmation arriving
      // after an RTDS preliminary will overwrite — both keys collapse
      // to the same id.
      const exists = prev.some((t) => t.id === marker.id)
      if (exists) {
        return prev.map((t) => (t.id === marker.id ? marker : t))
      }
      // Cap the live buffer to avoid unbounded growth across long sessions.
      const next = [...prev, marker]
      return next.length > 500 ? next.slice(-500) : next
    })
  }, [lastMessage, upTokenId, downTokenId, cycleStart, cycleEnd])

  // Reset live trades on cycle change so the chart starts fresh per cycle.
  useEffect(() => {
    setLiveTrades([])
  }, [cycleKey])

  // Filter trade markers to just trades that fall within the chart's
  // cycle window, then merge live + polled by id (live wins).
  const tradeMarkers = useMemo<WalletTradeMarker[]>(() => {
    const polled = walletTradesData?.trades ?? []
    const inWindow = (t: WalletTradeMarker) => {
      if (t.detected_at_ms === null) return false
      if (cycleStart === null || cycleEnd === null) return true
      const startMs = cycleStart * 1000
      const endMs = cycleEnd * 1000
      return t.detected_at_ms >= startMs - 5_000 && t.detected_at_ms <= endMs + 5_000
    }
    const byId = new Map<string, WalletTradeMarker>()
    for (const t of polled) if (inWindow(t)) byId.set(t.id, t)
    // Live overrides polled when the same tx_hash arrives — typically
    // the live RTDS event lands first, then the on-chain version
    // upgrades it. Either way we end up with one marker.
    for (const t of liveTrades) if (inWindow(t)) byId.set(t.id, t)
    return Array.from(byId.values()).sort(
      (a, b) => (a.detected_at_ms ?? 0) - (b.detected_at_ms ?? 0),
    )
  }, [walletTradesData, liveTrades, cycleStart, cycleEnd])

  // Oracle history → array of points sorted by time. Includes the live
  // oracle_price as a trailing point if it's newer than the last history
  // entry (mirrors what CryptoMarketsPanel's existing Liveline integration
  // does).
  const oracleHistory = useMemo<Point[]>(() => {
    const raw = Array.isArray(market.oracle_history) ? market.oracle_history : []
    const points: Point[] = []
    for (const row of raw) {
      const t = toUnixSec(row?.t)
      const v = typeof row?.p === 'number' && Number.isFinite(row.p) ? row.p : null
      if (t === null || v === null) continue
      points.push({ t, v })
    }
    points.sort((a, b) => a.t - b.t)

    if (
      market.oracle_price !== null &&
      market.oracle_price !== undefined &&
      Number.isFinite(market.oracle_price)
    ) {
      const last = points[points.length - 1]
      const liveT = toUnixSec(market.oracle_updated_at_ms) ?? Math.floor(Date.now() / 1000)
      if (!last || liveT > last.t) {
        points.push({ t: liveT, v: market.oracle_price })
      }
    }
    return points
  }, [market.oracle_history, market.oracle_price, market.oracle_updated_at_ms])

  // Drawing happens on every animation frame. drawChartRef pattern (refs
  // not closures) lets us read latest data without recreating the rAF loop.
  const drawRef = useRef<() => void>(() => {})
  drawRef.current = () => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (cycleStart === null || cycleEnd === null) return

    const dpr = window.devicePixelRatio || 1
    const rect = container.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return
    const w = rect.width
    const h = rect.height
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr
      canvas.height = h * dpr
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const pad = { top: 12, right: 64, bottom: 28, left: 72 }
    const plotW = w - pad.left - pad.right
    const plotH = h - pad.top - pad.bottom
    if (plotW <= 0 || plotH <= 0) return

    const vStart = cycleStart
    const vEnd = cycleEnd
    const vDur = vEnd - vStart
    if (vDur <= 0) return

    const oracleVals = oracleHistory.map((p) => p.v).filter((v) => Number.isFinite(v))
    let oracleMin = Math.min(...oracleVals)
    let oracleMax = Math.max(...oracleVals)
    if (
      !Number.isFinite(oracleMin) ||
      !Number.isFinite(oracleMax) ||
      oracleMin === oracleMax
    ) {
      const center = market.oracle_price ?? market.price_to_beat ?? 0
      oracleMin = center - 1
      oracleMax = center + 1
    }
    // Pad the y-range by 5% so lines don't kiss the frame.
    const oraclePad = (oracleMax - oracleMin) * 0.05
    oracleMin -= oraclePad
    oracleMax += oraclePad

    const timeToX = (t: number) =>
      pad.left + ((t - vStart) / vDur) * plotW
    const oracleToY = (v: number) =>
      pad.top + plotH - ((v - oracleMin) / (oracleMax - oracleMin)) * plotH
    // Share axis: 0¢ at bottom, 100¢ at top. Mirrors UpDown's chart.
    const shareToY = (cents: number) => pad.top + plotH - (cents / 100) * plotH

    // Background
    ctx.fillStyle = palette.background
    ctx.fillRect(0, 0, w, h)

    // Horizontal grid (every 10¢ on the share axis)
    ctx.strokeStyle = palette.grid
    ctx.lineWidth = 1
    for (let c = 0; c <= 100; c += 10) {
      const y = shareToY(c)
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
    }

    // Vertical grid (cycle quarters)
    for (let q = 0; q <= 4; q += 1) {
      const t = vStart + (vDur * q) / 4
      const x = timeToX(t)
      ctx.beginPath()
      ctx.moveTo(x, pad.top)
      ctx.lineTo(x, pad.top + plotH)
      ctx.stroke()
    }

    // Y-axis labels (left = oracle USD, right = share cents)
    ctx.fillStyle = palette.text
    ctx.font = "11px 'SF Mono', 'Fira Code', monospace"
    ctx.textAlign = 'right'
    for (let q = 0; q <= 4; q += 1) {
      const v = oracleMin + ((oracleMax - oracleMin) * q) / 4
      const y = oracleToY(v)
      ctx.fillText(v.toFixed(v >= 100 ? 0 : 2), pad.left - 6, y + 3)
    }
    ctx.textAlign = 'left'
    for (let c = 0; c <= 100; c += 25) {
      const y = shareToY(c)
      ctx.fillText(`${c}¢`, w - pad.right + 6, y + 3)
    }

    // X-axis labels: cycle 0%, 50%, 100%
    ctx.textAlign = 'center'
    ctx.fillText('0%', timeToX(vStart), pad.top + plotH + 16)
    ctx.fillText('50%', timeToX(vStart + vDur / 2), pad.top + plotH + 16)
    ctx.fillText('100%', timeToX(vEnd), pad.top + plotH + 16)

    // Oracle line
    if (oracleHistory.length >= 2) {
      ctx.strokeStyle = palette.oracle
      ctx.lineWidth = 2
      ctx.beginPath()
      let firstDrawn = false
      for (const p of oracleHistory) {
        if (p.t < vStart - 30 || p.t > vEnd + 30) continue
        const x = timeToX(p.t)
        const y = oracleToY(p.v)
        if (!firstDrawn) {
          ctx.moveTo(x, y)
          firstDrawn = true
        } else {
          ctx.lineTo(x, y)
        }
      }
      if (firstDrawn) ctx.stroke()
    }

    // UP / DOWN share lines
    const drawShareLine = (points: Point[], color: string) => {
      if (points.length < 2) return
      ctx.strokeStyle = color
      ctx.lineWidth = 1.75
      ctx.beginPath()
      let started = false
      for (const p of points) {
        if (p.t < vStart - 5 || p.t > vEnd + 5) continue
        const x = timeToX(p.t)
        const y = shareToY(p.v * 100)
        if (!started) {
          ctx.moveTo(x, y)
          started = true
        } else {
          ctx.lineTo(x, y)
        }
      }
      if (started) ctx.stroke()
    }
    drawShareLine(upHistoryRef.current, palette.up)
    drawShareLine(downHistoryRef.current, palette.down)

    // Wallet trade markers — outcome colors the marker; side picks ▲/▼.
    for (const trade of tradeMarkers) {
      if (trade.detected_at_ms === null) continue
      const t = Math.floor(trade.detected_at_ms / 1000)
      if (t < vStart - 5 || t > vEnd + 5) continue
      const x = timeToX(t)
      // Plot at the trade's reported share price, falling back to the
      // outcome's mid history if missing.
      const cents =
        trade.price !== null && trade.price >= 0 && trade.price <= 1
          ? trade.price * 100
          : trade.outcome === 'UP'
          ? (upHistoryRef.current[upHistoryRef.current.length - 1]?.v ?? 0.5) * 100
          : (downHistoryRef.current[downHistoryRef.current.length - 1]?.v ?? 0.5) * 100
      const y = shareToY(cents)
      const color = trade.outcome === 'UP' ? palette.up : palette.down
      const size = 6
      ctx.beginPath()
      if (trade.side === 'BUY') {
        // up-pointing triangle
        ctx.moveTo(x, y - size)
        ctx.lineTo(x + size, y + size)
        ctx.lineTo(x - size, y + size)
      } else {
        ctx.moveTo(x, y + size)
        ctx.lineTo(x + size, y - size)
        ctx.lineTo(x - size, y - size)
      }
      ctx.closePath()
      // Preliminary RTDS detections render as a hollow outline so the
      // viewer can tell pre-confirmation events from confirmed ones.
      // When the on-chain version arrives we replace the marker (same
      // id) and it fills in.
      if (trade.confirmed === false) {
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        ctx.stroke()
      } else {
        ctx.fillStyle = color
        ctx.fill()
      }
    }
  }

  // Animation loop — repaint on rAF so live data + cycle progress feel
  // smooth. Cleans up on unmount.
  useEffect(() => {
    let rafId = 0
    const tick = () => {
      drawRef.current()
      rafId = window.requestAnimationFrame(tick)
    }
    rafId = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(rafId)
  }, [])

  if (cycleStart === null || cycleEnd === null) {
    return (
      <div className="flex items-center justify-center text-xs text-muted-foreground" style={{ height }}>
        {t('cryptoCycleChart.waitingForCycleStart')}
      </div>
    )
  }

  return (
    <div ref={containerRef} className="relative w-full" style={{ height }}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      {/* Legend */}
      <div className="pointer-events-none absolute left-3 top-2 flex items-center gap-3 text-[10px] font-mono uppercase tracking-wider">
        <span className="flex items-center gap-1" style={{ color: palette.oracle }}>
          <span className="h-0.5 w-3" style={{ backgroundColor: palette.oracle }} />
          {t('cryptoCycleChart.legend.oracle')}
        </span>
        <span className="flex items-center gap-1" style={{ color: palette.up }}>
          <span className="h-0.5 w-3" style={{ backgroundColor: palette.up }} />
          {t('cryptoCycleChart.legend.up')}
        </span>
        <span className="flex items-center gap-1" style={{ color: palette.down }}>
          <span className="h-0.5 w-3" style={{ backgroundColor: palette.down }} />
          {t('cryptoCycleChart.legend.down')}
        </span>
        {tradeMarkers.length > 0 && (
          <span className="text-muted-foreground">
            {tradeMarkers.length === 1
              ? t('cryptoCycleChart.walletTradeCount_one', { n: tradeMarkers.length })
              : t('cryptoCycleChart.walletTradeCount_other', { n: tradeMarkers.length })}
            {tradeMarkers.some((t) => t.confirmed === false) && (
              <span className="ml-1 text-muted-foreground/70">
                {t('cryptoCycleChart.hollowPending')}
              </span>
            )}
          </span>
        )}
      </div>
    </div>
  )
}
