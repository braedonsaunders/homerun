import { useEffect, useRef } from 'react'

// Browser-direct Binance bookTicker feed for the crypto opportunity cards.
//
// The display row is purely cosmetic — there is no reason to round-trip
// the price through our backend (debounce 25ms + broadcast rate-limit
// 100ms + WS send + JSON parse) when the browser can subscribe to
// Binance's public stream and stamp updated_at_ms against its own
// wall clock.  This is the same approach the user's polymarket-btc-pulse-overlay
// chrome extension takes.
//
// Backend continues to maintain its own connection (services/binance_feed.py)
// for orchestrator/strategy execution gates that read from snapshot data.

// NOTE: use the US endpoint (stream.binance.us), NOT the global
// stream.binance.com.  From US-region networks binance.com does NOT hard-block
// the market-data socket — it *degrades*: it keeps flooding bookTicker messages
// (so the displayed age stays ~0s) but the top-of-book PRICE is frozen (only
// quantities churn).  Verified 2026-05: binance.com → 125 msgs/15s but 1
// distinct price; binance.us → 47 msgs/15s, 35 distinct prices (live).  Both
// are public/keyless; binance.us serves the same btc/eth/sol/xrp usdt pairs.
const BINANCE_STREAM_URL =
  'wss://stream.binance.us:9443/stream?streams=' +
  'btcusdt@bookTicker/ethusdt@bookTicker/solusdt@bookTicker/xrpusdt@bookTicker'

const SYMBOL_TO_ASSET: Record<string, string> = {
  btcusdt: 'BTC',
  ethusdt: 'ETH',
  solusdt: 'SOL',
  xrpusdt: 'XRP',
}

const RECONNECT_DELAY_MS = 500

export interface BinanceDirectPrice {
  asset: string
  price: number
  bid: number
  ask: number
  updatedAtMs: number
}

export type BinanceDirectPrices = Record<string, BinanceDirectPrice>

/**
 * Subscribes to Binance bookTicker streams for BTC/ETH/SOL/XRP and writes
 * the latest tick into a ref.  Returns the ref so callers can read the
 * latest snapshot inside their existing render path (eg. driven by the
 * 250ms `nowMs` ticker) without re-rendering on every tick.
 *
 * Pass `enabled=false` to tear down the connection — used to pause when
 * the panel is offscreen / the tab is hidden.
 */
export function useBinanceDirectFeed(enabled: boolean): {
  current: BinanceDirectPrices
} {
  const pricesRef = useRef<BinanceDirectPrices>({})

  useEffect(() => {
    if (!enabled) return

    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let stopped = false

    const scheduleReconnect = () => {
      if (stopped || reconnectTimer != null) return
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        connect()
      }, RECONNECT_DELAY_MS)
    }

    const connect = () => {
      if (stopped) return
      try {
        ws = new WebSocket(BINANCE_STREAM_URL)
      } catch {
        scheduleReconnect()
        return
      }

      ws.onmessage = (event) => {
        let data: any
        try {
          data = JSON.parse(typeof event.data === 'string' ? event.data : '')
        } catch {
          return
        }
        const ticker = (data && typeof data === 'object' && 'data' in data) ? data.data : data
        if (!ticker || typeof ticker !== 'object') return
        const symbol = String(ticker.s || '').toLowerCase()
        const asset = SYMBOL_TO_ASSET[symbol]
        if (!asset) return
        const bid = parseFloat(ticker.b)
        const ask = parseFloat(ticker.a)
        if (!Number.isFinite(bid) || !Number.isFinite(ask) || bid <= 0 || ask <= 0) return
        const mid = (bid + ask) / 2
        pricesRef.current = {
          ...pricesRef.current,
          [asset]: { asset, price: mid, bid, ask, updatedAtMs: Date.now() },
        }
      }

      ws.onerror = () => {
        try { ws?.close() } catch { /* noop */ }
      }

      ws.onclose = () => {
        scheduleReconnect()
      }
    }

    connect()

    return () => {
      stopped = true
      if (reconnectTimer != null) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
      if (ws) {
        try { ws.close() } catch { /* noop */ }
        ws = null
      }
      pricesRef.current = {}
    }
  }, [enabled])

  return pricesRef
}
