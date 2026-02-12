import { useEffect, useRef, useState, useCallback } from 'react'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'

interface WebSocketMessage {
  type: string
  data: any
}

// ─── Shared singleton so multiple hook consumers reuse one connection ───

let sharedWs: WebSocket | null = null
let sharedConnected = false
let sharedReconnectTimeout: ReturnType<typeof setTimeout> | null = null
let sharedReconnectAttempts = 0
let sharedPingInterval: ReturnType<typeof setInterval> | null = null
const MAX_RECONNECT_DELAY = 10_000
const BASE_RECONNECT_DELAY = 1_000
const CLIENT_PING_INTERVAL_MS = 15_000
const listeners = new Set<(msg: WebSocketMessage) => void>()
const statusListeners = new Set<(connected: boolean) => void>()
let mountedCount = 0
let disposed = false

function clearPingLoop() {
  if (sharedPingInterval) {
    clearInterval(sharedPingInterval)
    sharedPingInterval = null
  }
}

function startPingLoop() {
  clearPingLoop()
  sharedPingInterval = setInterval(() => {
    if (sharedWs?.readyState !== WebSocket.OPEN) return
    try {
      sharedWs.send(JSON.stringify({ type: 'ping' }))
    } catch {
      // Let the regular close/reconnect path handle this.
    }
  }, CLIENT_PING_INTERVAL_MS)
}

function notifyStatus(connected: boolean) {
  sharedConnected = connected
  statusListeners.forEach((fn) => fn(connected))
}

function sharedConnect(url: string) {
  if (disposed) return
  if (sharedWs) {
    const state = sharedWs.readyState
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = url.startsWith('ws') ? url : `${protocol}//${window.location.host}${url}`

  try {
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      if (disposed) {
        ws.close()
        return
      }
      sharedReconnectAttempts = 0
      notifyStatus(true)
      startPingLoop()
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        normalizeUtcTimestampsInPlace(message)
        listeners.forEach((fn) => fn(message))
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      if (sharedWs !== ws) return
      sharedWs = null
      clearPingLoop()
      notifyStatus(false)
      // Don't reconnect if all consumers have unmounted
      if (disposed || mountedCount <= 0) return
      scheduleReconnect(url)
    }

    ws.onerror = () => {
      // onerror is always followed by onclose — no action needed here
    }

    sharedWs = ws
  } catch {
    // WebSocket constructor can throw on invalid URL etc.
    scheduleReconnect(url)
  }
}

function scheduleReconnect(url: string) {
  if (sharedReconnectTimeout) clearTimeout(sharedReconnectTimeout)
  if (disposed || mountedCount <= 0) return

  const delay = Math.min(
    BASE_RECONNECT_DELAY * Math.pow(2, sharedReconnectAttempts),
    MAX_RECONNECT_DELAY,
  )
  sharedReconnectAttempts++
  sharedReconnectTimeout = setTimeout(() => {
    sharedReconnectTimeout = null
    sharedConnect(url)
  }, delay)
}

function sharedDisconnect() {
  disposed = true
  if (sharedReconnectTimeout) {
    clearTimeout(sharedReconnectTimeout)
    sharedReconnectTimeout = null
  }
  clearPingLoop()
  if (sharedWs) {
    try { sharedWs.close() } catch { /* ignore */ }
    sharedWs = null
  }
  notifyStatus(false)
}

// ─── Hook ────────────────────────────────────────────────

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(sharedConnected)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const urlRef = useRef(url)
  urlRef.current = url

  useEffect(() => {
    mountedCount++
    disposed = false

    // Register listeners
    const onMsg = (msg: WebSocketMessage) => setLastMessage(msg)
    const onStatus = (connected: boolean) => setIsConnected(connected)
    listeners.add(onMsg)
    statusListeners.add(onStatus)

    // Start connection if not already connected
    sharedConnect(urlRef.current)

    const reconnectNow = () => {
      if (disposed || mountedCount <= 0) return
      if (
        sharedWs &&
        (sharedWs.readyState === WebSocket.OPEN || sharedWs.readyState === WebSocket.CONNECTING)
      ) {
        return
      }
      if (sharedReconnectTimeout) {
        clearTimeout(sharedReconnectTimeout)
        sharedReconnectTimeout = null
      }
      sharedReconnectAttempts = 0
      sharedConnect(urlRef.current)
    }

    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        reconnectNow()
      }
    }

    document.addEventListener('visibilitychange', onVisibilityChange)
    window.addEventListener('focus', reconnectNow)
    window.addEventListener('online', reconnectNow)

    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('focus', reconnectNow)
      window.removeEventListener('online', reconnectNow)
      listeners.delete(onMsg)
      statusListeners.delete(onStatus)
      mountedCount--

      // Tear down the shared connection when the last consumer unmounts
      if (mountedCount <= 0) {
        sharedDisconnect()
        sharedReconnectAttempts = 0
      }
    }
  }, []) // empty deps — url is read from ref

  const sendMessage = useCallback((message: object) => {
    if (sharedWs?.readyState === WebSocket.OPEN) {
      sharedWs.send(JSON.stringify(message))
    }
  }, [])

  const reconnect = useCallback(() => {
    // Force a fresh connection
    if (sharedWs) {
      try { sharedWs.close() } catch { /* ignore */ }
      sharedWs = null
    }
    sharedReconnectAttempts = 0
    disposed = false
    sharedConnect(urlRef.current)
  }, [])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    reconnect,
  }
}
