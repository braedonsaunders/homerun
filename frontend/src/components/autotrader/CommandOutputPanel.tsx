import { useCallback, useEffect, useMemo, useRef, useState, type UIEvent } from 'react'
import { Terminal } from 'lucide-react'

import { cn } from '../../lib/utils'
import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

const STICKY_BOTTOM_THRESHOLD_PX = 24
const DETAIL_ROW_LIMIT = 14
const SOURCE_LABELS: Record<string, string> = {
  scanner: 'Markets',
  crypto: 'Crypto',
  news: 'News',
  weather: 'Weather',
  world_intelligence: 'World Intelligence',
  tracked_traders: 'Tracked Traders',
  insider: 'Insider',
  copy: 'Copy Trading',
}

export interface CommandLogLine {
  id: string
  ts: string
  type: string
  level: 'info' | 'warn' | 'error' | 'event'
  message: string
  source?: string
  status?: string
  payload?: Record<string, any>
  raw?: Record<string, any>
}

interface DetailRow {
  label: string
  value: string
  mono?: boolean
}

export default function CommandOutputPanel({
  logs,
  connected,
}: {
  logs: CommandLogLine[]
  connected: boolean
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [stickToBottom, setStickToBottom] = useState(true)
  const logViewportRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (logs.length === 0) {
      setSelectedId(null)
      return
    }
    const latestId = logs[logs.length - 1].id
    setSelectedId((current) => {
      if (!current) return latestId
      if (stickToBottom) return latestId
      return logs.some((row) => row.id === current) ? current : latestId
    })
  }, [logs, stickToBottom])

  useEffect(() => {
    const viewport = logViewportRef.current
    if (!viewport || !stickToBottom) return
    viewport.scrollTop = viewport.scrollHeight
  }, [logs, stickToBottom])

  const handleLogScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    const viewport = event.currentTarget
    const distanceFromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
    setStickToBottom(distanceFromBottom <= STICKY_BOTTOM_THRESHOLD_PX)
  }, [])

  const jumpToLatest = useCallback(() => {
    const viewport = logViewportRef.current
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight
    }
    setStickToBottom(true)
    if (logs.length > 0) {
      setSelectedId(logs[logs.length - 1].id)
    }
  }, [logs])

  const selected = useMemo(() => {
    if (logs.length === 0) return null
    if (!selectedId) return logs[logs.length - 1]
    return logs.find((row) => row.id === selectedId) || logs[logs.length - 1]
  }, [logs, selectedId])

  const summaryRows = useMemo(() => (selected ? buildSummaryRows(selected) : []), [selected])
  const payloadRows = useMemo(
    () =>
      buildDetailRows(selected?.payload, {
        max: DETAIL_ROW_LIMIT,
      }),
    [selected]
  )
  const contextRows = useMemo(
    () =>
      buildDetailRows(selected?.raw, {
        exclude: ['payload'],
        max: DETAIL_ROW_LIMIT,
      }),
    [selected]
  )
  const selectedSource = useMemo(
    () => formatSourceLabel(firstValue(selected?.source, selected?.raw?.source, selected?.payload?.source)),
    [selected]
  )

  return (
    <Card className="border-border/50 bg-card/40 h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-600 dark:text-emerald-300" />
            Live Action Console
          </span>
          <span className="flex items-center gap-2">
            <Badge className="text-[10px] uppercase bg-background/70 text-muted-foreground">
              {logs.length} events
            </Badge>
            <Badge
              className={cn(
                'text-[9px] uppercase bg-background/60',
                stickToBottom ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300'
              )}
            >
              {stickToBottom ? 'following' : 'scrolled up'}
            </Badge>
            <span
              className={`text-[10px] ${
                connected ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300'
              }`}
            >
              {connected ? 'WS ONLINE' : 'WS DEGRADED'}
            </span>
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        {logs.length === 0 ? (
          <div className="h-full min-h-[260px] rounded-md border border-border/50 bg-background/70 dark:bg-black/70 p-3 font-mono text-[11px] text-foreground">
            <p className="text-muted-foreground">[idle] awaiting autotrader actions...</p>
          </div>
        ) : (
          <div className="h-full min-h-[280px] grid grid-cols-1 xl:grid-cols-12 gap-3">
            <div className="xl:col-span-5 min-h-0 rounded-md border border-border/50 bg-background/70 dark:bg-black/70 p-2 flex flex-col">
              <div ref={logViewportRef} onScroll={handleLogScroll} className="min-h-0 flex-1 overflow-y-auto space-y-1 pr-1">
                {logs.map((line) => (
                  <button
                    key={line.id}
                    type="button"
                    onClick={() => setSelectedId(line.id)}
                    className={cn(
                      'w-full text-left rounded-md border border-transparent bg-background/10 p-2 transition-colors',
                      'hover:border-border/60 hover:bg-background/20',
                      selected?.id === line.id && 'border-emerald-500/40 bg-emerald-500/10'
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-[10px] text-emerald-700 dark:text-emerald-300">[{line.ts}]</span>
                      <div className="flex items-center gap-1">
                        <Badge className="text-[9px] uppercase bg-background/60 text-muted-foreground">{line.type}</Badge>
                        <span className={cn('text-[10px] uppercase', levelColor(line.level))}>{line.level}</span>
                      </div>
                    </div>
                    <p className="mt-1 text-[11px] text-foreground truncate">{line.message}</p>
                    <p className="mt-0.5 text-[10px] text-muted-foreground truncate">
                      {formatSourceLabel(firstValue(line.source, line.raw?.source, line.payload?.source)) || 'Unknown Source'}
                    </p>
                  </button>
                ))}
              </div>
              {!stickToBottom && (
                <button
                  type="button"
                  onClick={jumpToLatest}
                  className="mt-2 rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-1 text-[10px] font-medium text-emerald-700 dark:text-emerald-200 hover:bg-emerald-500/20"
                >
                  Jump to latest
                </button>
              )}
            </div>

            <div className="xl:col-span-7 min-h-0 rounded-md border border-border/50 bg-background/70 dark:bg-black/70 p-2 overflow-y-auto">
              {!selected ? (
                <p className="text-[11px] text-muted-foreground">Select an action to inspect payload.</p>
              ) : (
                <div className="space-y-2">
                  <div className="rounded-md border border-border/50 bg-background/20 p-2 text-[11px] text-foreground">
                    <div className="flex items-center justify-between gap-2">
                      <p className="font-semibold">{selected.message}</p>
                      <Badge className="text-[9px] uppercase bg-background/60 text-muted-foreground">{selected.type}</Badge>
                    </div>
                    <p className="mt-1 text-muted-foreground">
                      {selectedSource || 'Unknown Source'}
                      {selected.status ? ` | ${selected.status}` : ''}
                      {selected.ts ? ` | ${selected.ts}` : ''}
                    </p>
                  </div>
                  <DetailGrid
                    title="Key Details"
                    rows={summaryRows}
                    emptyLabel="No structured details available for this action."
                  />
                  <div className="grid grid-cols-1 2xl:grid-cols-2 gap-2">
                    <DetailGrid title="Payload" rows={payloadRows} emptyLabel="No payload fields." />
                    <DetailGrid title="Context" rows={contextRows} emptyLabel="No context fields." />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function DetailGrid({
  title,
  rows,
  emptyLabel,
}: {
  title: string
  rows: DetailRow[]
  emptyLabel: string
}) {
  return (
    <div className="rounded-md border border-border/50 bg-background/20 p-2">
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{title}</p>
      {rows.length === 0 ? (
        <p className="mt-2 text-[11px] text-muted-foreground">{emptyLabel}</p>
      ) : (
        <div className="mt-2 grid grid-cols-1 gap-1.5">
          {rows.map((row, index) => (
            <div
              key={`${title}:${row.label}:${index}`}
              className="rounded border border-border/40 bg-background/40 px-2 py-1.5 flex items-start justify-between gap-2"
            >
              <span className="text-[9px] uppercase text-muted-foreground">{row.label}</span>
              <span className={cn('text-[11px] text-right text-foreground break-all', row.mono && 'font-mono')}>
                {row.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function buildSummaryRows(log: CommandLogLine): DetailRow[] {
  const raw = log.raw || {}
  const payload = log.payload || {}
  const merged: Record<string, unknown> = {
    ...payload,
    ...raw,
  }
  const rows: DetailRow[] = []

  pushRow(rows, 'Level', log.level.toUpperCase())
  pushRow(rows, 'Event Type', log.type)
  pushRow(rows, 'Source', formatSourceLabel(firstValue(log.source, merged.source)))
  pushRow(rows, 'Status', firstValue(log.status, merged.status))
  pushRow(rows, 'Event ID', merged.id, { mono: true })
  pushRow(rows, 'Trace ID', merged.trace_id, { mono: true })
  pushRow(rows, 'Decision ID', merged.decision_id, { mono: true })
  pushRow(rows, 'Operator', merged.operator)
  pushRow(rows, 'Market', firstValue(merged.market_question, merged.market_id))
  pushRow(rows, 'Direction', merged.direction)
  pushRow(rows, 'Mode', merged.mode)

  const score = asNumber(firstValue(merged.score, merged.computed_score))
  if (score != null) {
    rows.push({
      label: 'Score',
      value: score.toFixed(4),
      mono: true,
    })
  }

  const notional = asNumber(firstValue(merged.notional_usd, merged.total_cost))
  if (notional != null) {
    rows.push({
      label: 'Notional',
      value: `$${notional.toFixed(2)}`,
      mono: true,
    })
  }

  const price = asNumber(firstValue(merged.effective_price, merged.live_price, merged.paper_price, merged.entry_price))
  if (price != null) {
    rows.push({
      label: 'Price',
      value: `$${price.toFixed(4)}`,
      mono: true,
    })
  }

  const pnl = asNumber(firstValue(merged.actual_profit, merged.unrealized_pnl))
  if (pnl != null) {
    rows.push({
      label: 'PnL',
      value: `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`,
      mono: true,
    })
  }

  pushRow(rows, 'Error', firstValue(merged.error_message, payload.error))
  return rows.slice(0, DETAIL_ROW_LIMIT)
}

function buildDetailRows(
  data?: Record<string, any>,
  options: {
    exclude?: string[]
    max?: number
  } = {}
): DetailRow[] {
  if (!data || typeof data !== 'object') return []
  const max = options.max ?? DETAIL_ROW_LIMIT
  const exclude = new Set(options.exclude || [])
  const rows: DetailRow[] = []

  for (const [key, value] of Object.entries(data)) {
    if (exclude.has(key)) continue
    const formatted = formatDetailValue(value)
    if (!formatted) continue
    rows.push({
      label: humanizeKey(key),
      value: formatted,
      mono: shouldUseMonoForKey(key),
    })
    if (rows.length >= max) break
  }

  return rows
}

function pushRow(
  rows: DetailRow[],
  label: string,
  value: unknown,
  options: {
    mono?: boolean
  } = {}
) {
  const text = normalizeText(value)
  if (!text) return
  rows.push({
    label,
    value: text,
    mono: options.mono,
  })
}

function firstValue(...values: unknown[]): unknown {
  for (const value of values) {
    if (value === undefined || value === null) continue
    if (typeof value === 'string' && value.trim().length === 0) continue
    return value
  }
  return null
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function normalizeText(value: unknown): string | null {
  if (value === undefined || value === null) return null
  if (typeof value === 'string') {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimForDisplay(trimmed) : null
  }
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return null
}

function formatDetailValue(value: unknown): string | null {
  if (value === undefined || value === null) return null
  if (typeof value === 'string') {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimForDisplay(trimmed) : null
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null
    if (Number.isInteger(value)) return String(value)
    return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(4)
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) {
    if (value.length === 0) return 'empty list'
    const sample = value
      .slice(0, 3)
      .map((item) => normalizeText(item))
      .filter((item): item is string => Boolean(item))
    if (sample.length > 0) {
      const suffix = value.length > 3 ? ` (+${value.length - 3} more)` : ''
      return trimForDisplay(`${sample.join(', ')}${suffix}`)
    }
    return `${value.length} items`
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value as Record<string, unknown>)
    if (keys.length === 0) return 'empty object'
    const preview = keys.slice(0, 3).map(humanizeKey).join(', ')
    const suffix = keys.length > 3 ? ` +${keys.length - 3} more` : ''
    return `${keys.length} fields (${preview}${suffix})`
  }
  return trimForDisplay(String(value))
}

function humanizeKey(key: string): string {
  const words = key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => {
      const lower = word.toLowerCase()
      if (lower === 'id') return 'ID'
      if (lower === 'pnl') return 'PnL'
      if (lower === 'usd') return 'USD'
      return lower.charAt(0).toUpperCase() + lower.slice(1)
    })
  return words.join(' ')
}

function shouldUseMonoForKey(key: string): boolean {
  return /(id|_at|hash|trace|token|cursor|key)$/i.test(key)
}

function formatSourceLabel(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const key = value.trim()
  if (!key) return null
  return SOURCE_LABELS[key] || humanizeKey(key)
}

function trimForDisplay(value: string, limit = 120): string {
  if (value.length <= limit) return value
  return `${value.slice(0, limit - 3)}...`
}

function levelColor(level: CommandLogLine['level']): string {
  switch (level) {
    case 'warn':
      return 'text-amber-300'
    case 'error':
      return 'text-rose-300'
    case 'event':
      return 'text-sky-300'
    default:
      return 'text-emerald-300'
  }
}
