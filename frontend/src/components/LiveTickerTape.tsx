import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { cn } from '../lib/utils'
import {
  AlertTriangle,
  BarChart3,
  Bot,
  TrendingUp,
  TrendingDown,
  Zap,
  Radio,
  DollarSign,
  Shield,
  Clock,
  Activity,
  Target,
} from 'lucide-react'

type WorkerHealthTone = 'green' | 'amber' | 'red'

interface WorkerHealthSummary {
  overallTone: WorkerHealthTone
  counts: {
    green: number
    amber: number
    red: number
    total: number
  }
}

interface SourceCounts {
  scanner: number
  traders: number
  news: number
  weather: number
  crypto: number
}

interface SignalTotals {
  pending: number
  selected: number
  submitted: number
  executed: number
  skipped: number
  expired: number
  failed: number
}

interface RealtimeMessage {
  type?: string
  data?: Record<string, unknown> | null
}

interface OrchestratorControl {
  mode?: string
  is_enabled?: boolean
  is_paused?: boolean
  kill_switch?: boolean
}

interface OrchestratorSnapshot {
  running: boolean
  enabled: boolean
  decisions_count: number
  orders_count: number
  open_orders: number
  gross_exposure_usd: number
  daily_pnl: number
  last_error: string | null
}

interface TickerTapeProps {
  isConnected: boolean
  globallyPaused: boolean
  lastScan?: string | null
  workerHealth: WorkerHealthSummary
  sourceCounts: SourceCounts
  signalTotals?: SignalTotals | null
  orchestratorControl?: OrchestratorControl | null
  lastMessage?: RealtimeMessage | null
  activeStrategies?: number
  className?: string
}

interface TickerItem {
  id: string
  label: string
  value: string
  secondaryValue?: string
  change?: number
  icon?: 'up' | 'down' | 'zap' | 'dollar' | 'shield' | 'clock' | 'activity' | 'target' | 'alert' | 'bot' | 'chart'
  badge?: string
  badgeColor?: string
}

interface EventRollup {
  total: number
  scans: number
  decisions: number
  orders: number
  signals: number
  alerts: number
}

const ROLLUP_TTL_SECONDS = 180
const COMPACT_INT = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })

const ICON_MAP = {
  up: TrendingUp,
  down: TrendingDown,
  zap: Zap,
  dollar: DollarSign,
  shield: Shield,
  clock: Clock,
  activity: Activity,
  target: Target,
  alert: AlertTriangle,
  bot: Bot,
  chart: BarChart3,
}

const ICON_COLOR_MAP: Record<string, string> = {
  up: 'text-green-400',
  down: 'text-red-400',
  zap: 'text-blue-400',
  dollar: 'text-yellow-400',
  shield: 'text-cyan-400',
  clock: 'text-orange-400',
  activity: 'text-emerald-400',
  target: 'text-blue-400',
  alert: 'text-red-400',
  bot: 'text-violet-400',
  chart: 'text-sky-400',
}

function zeroRollup(): EventRollup {
  return { total: 0, scans: 0, decisions: 0, orders: 0, signals: 0, alerts: 0 }
}

function normalizeCount(value: unknown): number {
  const num = Number(value)
  if (!Number.isFinite(num)) return 0
  return Math.max(0, Math.round(num))
}

function normalizeNumber(value: unknown): number {
  const num = Number(value)
  if (!Number.isFinite(num)) return 0
  return num
}

function compactCount(value: number): string {
  if (Math.abs(value) >= 1000) return COMPACT_INT.format(value)
  return `${value}`
}

function formatUsd(value: number): string {
  const sign = value > 0 ? '+' : value < 0 ? '-' : ''
  const abs = Math.abs(value)
  if (abs >= 1000) return `${sign}$${COMPACT_INT.format(abs)}`
  return `${sign}$${abs.toFixed(2)}`
}

function truncate(str: string, len: number): string {
  if (str.length <= len) return str
  return str.slice(0, len - 1) + '\u2026'
}

function formatTimeSince(iso: string): string {
  const secs = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (secs < 5) return 'just now'
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  return `${Math.floor(secs / 3600)}h ago`
}

function formatSinceTimestamp(epochMs: number): string {
  const secs = Math.max(0, Math.floor((Date.now() - epochMs) / 1000))
  if (secs < 5) return 'just now'
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  return `${Math.floor(secs / 3600)}h ago`
}

function coerceRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object') return {}
  return value as Record<string, unknown>
}

function normalizeOrchestratorSnapshot(value: unknown): OrchestratorSnapshot | null {
  const raw = coerceRecord(value)
  if (Object.keys(raw).length < 1) return null
  return {
    running: Boolean(raw.running),
    enabled: Boolean(raw.enabled),
    decisions_count: normalizeCount(raw.decisions_count),
    orders_count: normalizeCount(raw.orders_count),
    open_orders: normalizeCount(raw.open_orders),
    gross_exposure_usd: normalizeNumber(raw.gross_exposure_usd),
    daily_pnl: normalizeNumber(raw.daily_pnl),
    last_error: String(raw.last_error || '').trim() || null,
  }
}

function appendPatch(target: EventRollup, patch: Partial<EventRollup>): void {
  target.total += patch.total || 0
  target.scans += patch.scans || 0
  target.decisions += patch.decisions || 0
  target.orders += patch.orders || 0
  target.signals += patch.signals || 0
  target.alerts += patch.alerts || 0
}

function mergeRollup(
  buckets: Map<number, EventRollup>,
  timestampMs: number,
  patch: Partial<EventRollup>,
): void {
  const second = Math.floor(timestampMs / 1000)
  const bucket = buckets.get(second) || zeroRollup()
  appendPatch(bucket, patch)
  buckets.set(second, bucket)
}

function pruneBuckets(
  buckets: Map<number, EventRollup>,
  nowMs: number,
): void {
  const minSecond = Math.floor(nowMs / 1000) - ROLLUP_TTL_SECONDS
  for (const second of buckets.keys()) {
    if (second < minSecond) buckets.delete(second)
  }
}

function summarizeBuckets(
  buckets: Map<number, EventRollup>,
  nowMs: number,
  windowSeconds: number,
): EventRollup {
  const out = zeroRollup()
  const minSecond = Math.floor(nowMs / 1000) - windowSeconds + 1
  for (const [second, bucket] of buckets.entries()) {
    if (second < minSecond) continue
    appendPatch(out, bucket)
  }
  return out
}

function eventLabel(type: string): string | null {
  if (type === 'scanner_status') return 'Scanner status'
  if (type === 'opportunities_update') return 'Opportunity snapshot'
  if (type === 'opportunity_events') return 'Opportunity delta'
  if (type === 'worker_status_update') return 'Worker heartbeat'
  if (type === 'signals_update') return 'Signal snapshot'
  if (type === 'trader_orchestrator_status') return 'Orchestrator snapshot'
  if (type === 'trader_decision') return 'Trader decision'
  if (type === 'trader_order') return 'Trader order'
  if (type === 'trader_event') return 'Trader event'
  if (type === 'weather_update' || type === 'weather_status') return 'Weather update'
  if (type === 'news_workflow_update' || type === 'news_workflow_status' || type === 'news_update') return 'News update'
  if (type === 'crypto_markets_update') return 'Crypto update'
  return null
}

function rollupPatchFromMessage(
  type: string,
  data: Record<string, unknown>,
  scannerLastSeenRef: { current: string | null },
): Partial<EventRollup> | null {
  if (
    type === 'ping'
    || type === 'pong'
    || type === 'subscribed'
    || type === 'init'
    || type === 'scan_requested'
  ) {
    return null
  }

  if (type === 'scanner_status') {
    const patch: Partial<EventRollup> = { total: 1 }
    const nextScan = typeof data.last_scan === 'string' ? data.last_scan : null
    if (nextScan && nextScan !== scannerLastSeenRef.current) {
      scannerLastSeenRef.current = nextScan
      patch.scans = 1
    }
    const activity = String(data.current_activity || '').toLowerCase()
    if (activity.includes('error')) {
      patch.alerts = 1
    }
    return patch
  }

  if (type === 'scanner_activity') {
    const activity = String(data.activity || '').toLowerCase()
    return {
      total: 1,
      alerts: activity.includes('error') ? 1 : 0,
    }
  }

  if (type === 'worker_status_update') {
    const workers = Array.isArray(data.workers) ? data.workers : []
    let alerts = 0
    for (const worker of workers) {
      const row = coerceRecord(worker)
      if (String(row.last_error || '').trim()) alerts += 1
    }
    return {
      total: 1,
      alerts,
    }
  }

  if (type === 'trader_decision') return { total: 1, decisions: 1 }
  if (type === 'trader_order') return { total: 1, orders: 1 }
  if (type === 'signals_update') return { total: 1, signals: 1 }

  if (type === 'trader_event') {
    const severity = String(data.severity || '').toLowerCase()
    return {
      total: 1,
      alerts: severity === 'warn' || severity === 'warning' || severity === 'error' ? 1 : 0,
    }
  }

  if (type === 'trader_orchestrator_status') {
    return {
      total: 1,
      alerts: String(data.last_error || '').trim() ? 1 : 0,
    }
  }

  if (
    type === 'opportunities_update'
    || type === 'opportunity_events'
    || type === 'weather_update'
    || type === 'weather_status'
    || type === 'news_workflow_update'
    || type === 'news_workflow_status'
    || type === 'news_update'
    || type === 'crypto_markets_update'
  ) {
    return { total: 1 }
  }

  return null
}

export default function LiveTickerTape({
  isConnected,
  globallyPaused,
  lastScan,
  workerHealth,
  sourceCounts,
  signalTotals,
  orchestratorControl,
  lastMessage,
  activeStrategies,
  className,
}: TickerTapeProps) {
  const [tick, setTick] = useState(0)
  const [orchestratorSnapshot, setOrchestratorSnapshot] = useState<OrchestratorSnapshot | null>(null)
  const rollupBucketsRef = useRef<Map<number, EventRollup>>(new Map())
  const lastEventRef = useRef<{ label: string; at: number } | null>(null)
  const scannerLastSeenRef = useRef<string | null>(lastScan || null)
  const orchestratorSigRef = useRef<string>('')

  useEffect(() => {
    if (!lastScan || isNaN(new Date(lastScan).getTime())) return
    scannerLastSeenRef.current = lastScan
  }, [lastScan])

  useEffect(() => {
    const id = window.setInterval(() => {
      pruneBuckets(rollupBucketsRef.current, Date.now())
      setTick((prev) => prev + 1)
    }, 1000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    if (!lastMessage?.type) return
    const type = String(lastMessage.type)
    const data = coerceRecord(lastMessage.data)
    const now = Date.now()

    if (type === 'init') {
      const scannerStatus = coerceRecord(data.scanner_status)
      const initLastScan = typeof scannerStatus.last_scan === 'string' ? scannerStatus.last_scan : null
      if (initLastScan) {
        scannerLastSeenRef.current = initLastScan
      }
      const initSnapshot = normalizeOrchestratorSnapshot(data.trader_orchestrator_status)
      if (initSnapshot) {
        const sig = JSON.stringify(initSnapshot)
        if (sig !== orchestratorSigRef.current) {
          orchestratorSigRef.current = sig
          setOrchestratorSnapshot(initSnapshot)
        }
      }
      lastEventRef.current = { label: 'Session init', at: now }
      return
    }

    if (type === 'trader_orchestrator_status') {
      const snapshot = normalizeOrchestratorSnapshot(data)
      if (snapshot) {
        const sig = JSON.stringify(snapshot)
        if (sig !== orchestratorSigRef.current) {
          orchestratorSigRef.current = sig
          setOrchestratorSnapshot(snapshot)
        }
      }
    }

    const patch = rollupPatchFromMessage(type, data, scannerLastSeenRef)
    if (!patch) return
    mergeRollup(rollupBucketsRef.current, now, patch)
    pruneBuckets(rollupBucketsRef.current, now)

    const label = eventLabel(type)
    if (label) {
      lastEventRef.current = { label, at: now }
    }
  }, [lastMessage])

  const rollup1m = useMemo(
    () => summarizeBuckets(rollupBucketsRef.current, Date.now(), 60),
    [tick],
  )

  const lastEvent = lastEventRef.current
  const workers = workerHealth.counts
  const signalPending = normalizeCount(signalTotals?.pending)
  const signalSelected = normalizeCount(signalTotals?.selected)
  const signalExecuted = normalizeCount(signalTotals?.executed)
  const signalFailed = normalizeCount(signalTotals?.failed)

  const mode = String(orchestratorControl?.mode || 'paper').toUpperCase()
  const killSwitch = Boolean(orchestratorControl?.kill_switch)
  const orchestratorEnabled = Boolean(orchestratorControl?.is_enabled)
  const orchestratorPaused = Boolean(orchestratorControl?.is_paused)
  const orchestratorActive = orchestratorEnabled && !orchestratorPaused && !killSwitch
  const orchestratorDecisions = normalizeCount(orchestratorSnapshot?.decisions_count)
  const orchestratorOrders = normalizeCount(orchestratorSnapshot?.orders_count)
  const orchestratorOpenOrders = normalizeCount(orchestratorSnapshot?.open_orders)
  const orchestratorExposure = normalizeNumber(orchestratorSnapshot?.gross_exposure_usd)
  const orchestratorDailyPnl = normalizeNumber(orchestratorSnapshot?.daily_pnl)
  const orchestratorConversion = orchestratorDecisions > 0
    ? (orchestratorOrders / orchestratorDecisions) * 100
    : 0

  const tickerItems = useMemo<TickerItem[]>(() => {
    const items: TickerItem[] = []

    items.push({
      id: 'system',
      label: 'SYSTEM',
      value: isConnected ? 'LIVE' : 'OFFLINE',
      icon: 'activity',
      change: isConnected && !globallyPaused ? 1 : -1,
      badge: globallyPaused ? 'PAUSED' : undefined,
      badgeColor: globallyPaused ? 'text-yellow-400 bg-yellow-400/10' : undefined,
    })

    if (lastScan && !isNaN(new Date(lastScan).getTime())) {
      items.push({
        id: 'scan',
        label: 'SCAN',
        value: formatTimeSince(lastScan),
        icon: 'clock',
      })
    }

    const workerBadge = workers.red > 0
      ? `${workers.red} ERR`
      : workers.amber > 0
        ? `${workers.amber} WARN`
        : 'HEALTHY'
    const workerBadgeColor = workers.red > 0
      ? 'text-red-400 bg-red-400/10'
      : workers.amber > 0
        ? 'text-amber-400 bg-amber-400/10'
        : 'text-emerald-400 bg-emerald-400/10'

    items.push({
      id: 'workers',
      label: 'WORKERS',
      value: `${workers.green}/${workers.total}`,
      icon: 'shield',
      badge: workerBadge,
      badgeColor: workerBadgeColor,
    })

    items.push({
      id: 'source-scanner',
      label: 'MARKETS',
      value: compactCount(sourceCounts.scanner),
      icon: 'zap',
    })
    items.push({
      id: 'source-traders',
      label: 'TRADERS',
      value: compactCount(sourceCounts.traders),
      icon: 'target',
    })
    items.push({
      id: 'source-news',
      label: 'NEWS',
      value: compactCount(sourceCounts.news),
      icon: 'activity',
    })
    items.push({
      id: 'source-weather',
      label: 'WEATHER',
      value: compactCount(sourceCounts.weather),
      icon: 'activity',
    })
    items.push({
      id: 'source-crypto',
      label: 'CRYPTO',
      value: compactCount(sourceCounts.crypto),
      icon: 'chart',
    })

    items.push({
      id: 'signals',
      label: 'SIGNALS',
      value: `P:${compactCount(signalPending)} S:${compactCount(signalSelected)} X:${compactCount(signalExecuted)}`,
      icon: 'activity',
      badge: `F:${compactCount(signalFailed)}`,
      badgeColor: signalFailed > 0 ? 'text-red-400 bg-red-400/10' : 'text-muted-foreground bg-muted/50',
    })

    if (activeStrategies !== undefined) {
      items.push({
        id: 'strategies',
        label: 'STRATEGIES',
        value: compactCount(activeStrategies),
        icon: 'activity',
      })
    }

    items.push({
      id: 'orchestrator-mode',
      label: 'ORCH',
      value: `${mode} ${orchestratorActive ? 'ACTIVE' : 'PAUSED'}`,
      icon: 'bot',
      change: orchestratorActive ? 1 : -1,
      badge: killSwitch ? 'KILL SWITCH' : undefined,
      badgeColor: killSwitch ? 'text-red-400 bg-red-400/10' : undefined,
    })

    items.push({
      id: 'orchestrator-funnel',
      label: 'FUNNEL',
      value: `${compactCount(orchestratorDecisions)}\u2192${compactCount(orchestratorOrders)}`,
      secondaryValue: `${orchestratorConversion.toFixed(1)}%`,
      icon: 'target',
    })

    items.push({
      id: 'orchestrator-pnl',
      label: 'DAY PNL',
      value: formatUsd(orchestratorDailyPnl),
      icon: orchestratorDailyPnl >= 0 ? 'up' : 'down',
      change: orchestratorDailyPnl,
    })

    items.push({
      id: 'orchestrator-exposure',
      label: 'EXPOSURE',
      value: formatUsd(orchestratorExposure),
      icon: 'dollar',
    })

    items.push({
      id: 'orchestrator-open-orders',
      label: 'OPEN ORD',
      value: compactCount(orchestratorOpenOrders),
      icon: 'chart',
    })

    items.push({
      id: 'events-total',
      label: 'EVENTS 1M',
      value: compactCount(rollup1m.total),
      icon: 'activity',
    })

    items.push({
      id: 'events-flow',
      label: 'FLOW 1M',
      value: `D:${compactCount(rollup1m.decisions)} O:${compactCount(rollup1m.orders)} S:${compactCount(rollup1m.signals)}`,
      icon: 'chart',
    })

    items.push({
      id: 'events-scans',
      label: 'SCANS 1M',
      value: compactCount(rollup1m.scans),
      icon: 'clock',
    })

    items.push({
      id: 'events-alerts',
      label: 'ALERTS 1M',
      value: compactCount(rollup1m.alerts),
      icon: rollup1m.alerts > 0 ? 'alert' : 'shield',
      change: rollup1m.alerts > 0 ? -rollup1m.alerts : 0,
      badge: rollup1m.alerts > 0 ? 'ATTN' : undefined,
      badgeColor: rollup1m.alerts > 0 ? 'text-red-400 bg-red-400/10' : undefined,
    })

    items.push({
      id: 'last-event',
      label: 'LAST EVT',
      value: lastEvent ? truncate(lastEvent.label, 24) : 'No recent events',
      secondaryValue: lastEvent ? formatSinceTimestamp(lastEvent.at) : undefined,
      icon: 'activity',
    })

    return items
  }, [
    activeStrategies,
    globallyPaused,
    isConnected,
    killSwitch,
    lastEvent,
    lastScan,
    mode,
    orchestratorActive,
    orchestratorConversion,
    orchestratorDailyPnl,
    orchestratorDecisions,
    orchestratorExposure,
    orchestratorOpenOrders,
    orchestratorOrders,
    rollup1m.alerts,
    rollup1m.decisions,
    rollup1m.orders,
    rollup1m.scans,
    rollup1m.signals,
    rollup1m.total,
    signalExecuted,
    signalFailed,
    signalPending,
    signalSelected,
    sourceCounts.crypto,
    sourceCounts.news,
    sourceCounts.scanner,
    sourceCounts.traders,
    sourceCounts.weather,
    workers.amber,
    workers.green,
    workers.red,
    workers.total,
  ])

  if (tickerItems.length === 0) return null

  // Duplicate items to create seamless loop
  const allItems = [...tickerItems, ...tickerItems]

  return (
    <div className={cn(
      "h-8 border-b border-border/30 bg-card/40 backdrop-blur-sm overflow-hidden relative shrink-0",
      className
    )}>
      {/* Left edge fade */}
      <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-background to-transparent z-10" />
      {/* Right edge fade */}
      <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-background to-transparent z-10" />

      {/* Live indicator */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 z-20 flex items-center gap-1">
        <Radio className={cn(
          "w-2.5 h-2.5",
          isConnected ? "text-green-400" : "text-red-400"
        )} />
      </div>

      {/* Scrolling content */}
      <div
        className="ticker-animate flex items-center h-full whitespace-nowrap pl-8"
        style={{ '--ticker-duration': `${Math.max(30, tickerItems.length * 3.6)}s` } as CSSProperties}
      >
        {allItems.map((item, i) => {
          const IconComponent = item.icon ? ICON_MAP[item.icon] : null
          const iconColor = item.icon ? ICON_COLOR_MAP[item.icon] : ''

          return (
            <div key={`${item.id}-${i}`} className="inline-flex items-center gap-1.5 mx-3 text-[11px]">
              {IconComponent && <IconComponent className={cn("w-3 h-3", iconColor)} />}
              <span className="text-muted-foreground font-medium">{item.label}</span>
              <span className={cn(
                "font-data font-semibold",
                item.change !== undefined
                  ? item.change >= 0 ? "text-green-400" : "text-red-400"
                  : "text-foreground"
              )}>
                {item.value}
              </span>
              {item.secondaryValue && (
                <span className="text-muted-foreground/70 font-data text-[10px]">
                  {item.secondaryValue}
                </span>
              )}
              {item.badge && (
                <span className={cn(
                  "px-1 py-0.5 rounded text-[9px] font-semibold leading-none",
                  item.badgeColor || "text-muted-foreground bg-muted/50"
                )}>
                  {item.badge}
                </span>
              )}
              <span className="text-border/60 mx-1.5">|</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
