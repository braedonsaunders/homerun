import { useState, useMemo, useEffect, lazy, Suspense } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  TrendingUp,
  Activity,
  Shield,
  Zap,
  MapPin,
  Radio,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Swords,
  Wifi,
  Map as MapIcon,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { formatCountry, normalizeCountryCode } from '../lib/worldCountries'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import ErrorBoundary from './ErrorBoundary'

const WorldMap = lazy(() => import('./WorldMap'))
import {
  getWorldSignals,
  getInstabilityScores,
  getTensionPairs,
  getConvergenceZones,
  getTemporalAnomalies,
  WorldSignal,
} from '../services/eventsApi'

type WorldSubView = 'map' | 'signals' | 'countries' | 'tensions' | 'convergences' | 'anomalies'

const SIGNAL_TYPE_CONFIG: Record<string, { icon: React.ElementType; color: string; labelKey: string }> = {
  conflict: { icon: Swords, color: 'text-red-400', labelKey: 'conflict' },
  tension: { icon: Activity, color: 'text-orange-400', labelKey: 'tension' },
  instability: { icon: AlertTriangle, color: 'text-yellow-400', labelKey: 'instability' },
  convergence: { icon: Radio, color: 'text-purple-400', labelKey: 'convergence' },
  anomaly: { icon: Zap, color: 'text-cyan-400', labelKey: 'anomaly' },
  military: { icon: Shield, color: 'text-blue-400', labelKey: 'military' },
  infrastructure: { icon: Wifi, color: 'text-emerald-400', labelKey: 'infrastructure' },
  earthquake: { icon: Zap, color: 'text-amber-400', labelKey: 'earthquake' },
  news: { icon: Radio, color: 'text-violet-400', labelKey: 'news' },
}

const METADATA_CHIPS_CONFIG: Record<string, Array<{ key: string; labelKey: string; format?: (v: unknown, t: (k: string) => string) => string }>> = {
  earthquake: [
    { key: 'magnitude', labelKey: 'mag', format: (v) => `M${Number(v).toFixed(1)}` },
    { key: 'depth_km', labelKey: 'depth', format: (v) => `${Number(v).toFixed(0)}km` },
    { key: 'tsunami', labelKey: 'tsunami', format: (v, t) => v ? t('eventsPanel.yes') : t('eventsPanel.no') },
    { key: 'alert', labelKey: 'alert' },
  ],
  military: [
    { key: 'activity_type', labelKey: 'type' },
    { key: 'callsign', labelKey: 'callsign' },
    { key: 'aircraft_type', labelKey: 'aircraft' },
    { key: 'region', labelKey: 'region' },
    { key: 'is_unusual', labelKey: 'unusual', format: (v, t) => v ? t('eventsPanel.yes') : t('eventsPanel.no') },
  ],
  anomaly: [
    { key: 'z_score', labelKey: 'z', format: (v) => Number(v).toFixed(1) },
    { key: 'current_value', labelKey: 'current', format: (v) => String(v) },
    { key: 'baseline_mean', labelKey: 'baseline', format: (v) => Number(v).toFixed(1) },
  ],
  infrastructure: [
    { key: 'event_type', labelKey: 'type' },
    { key: 'affected_services', labelKey: 'services', format: (v) => Array.isArray(v) ? v.join(', ') : String(v) },
    { key: 'cascade_risk_score', labelKey: 'cascade', format: (v) => `${(Number(v) * 100).toFixed(0)}%` },
  ],
  conflict: [
    { key: 'event_type', labelKey: 'type' },
    { key: 'sub_event_type', labelKey: 'subType' },
    { key: 'fatalities', labelKey: 'fatalities', format: (v) => String(v) },
  ],
  tension: [
    { key: 'trend', labelKey: 'trend' },
    { key: 'event_count', labelKey: 'events', format: (v) => String(v) },
  ],
  convergence: [
    { key: 'signal_count', labelKey: 'signals', format: (v) => String(v) },
  ],
}

type SignalsGroupBy = 'none' | 'type' | 'country' | 'severity' | 'source'
type SignalsLayout = 'list' | 'cards'

const SIGNAL_GROUP_OPTIONS: Array<{ value: SignalsGroupBy; labelKey: string }> = [
  { value: 'none', labelKey: 'noGrouping' },
  { value: 'type', labelKey: 'signalType' },
  { value: 'country', labelKey: 'country' },
  { value: 'severity', labelKey: 'severity' },
  { value: 'source', labelKey: 'source' },
]

function severityLevel(severity: number): 'critical' | 'high' | 'medium' | 'low' {
  if (severity >= 0.8) return 'critical'
  if (severity >= 0.6) return 'high'
  if (severity >= 0.3) return 'medium'
  return 'low'
}

function detectedAtValue(detectedAt: string | null | undefined): number {
  if (!detectedAt) return Number.NEGATIVE_INFINITY
  const parsed = Date.parse(detectedAt)
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY
}

function buildSignalGroups(
  signals: WorldSignal[],
  groupBy: SignalsGroupBy,
  t: (key: string) => string,
): Array<{
  key: string
  label: string
  order: number
  signals: WorldSignal[]
}> {
  const groups = new Map<string, { key: string; label: string; order: number; signals: WorldSignal[] }>()

  for (const signal of signals) {
    let key = 'all'
    let label = t('eventsPanel.allSignals')
    let order = 0

    if (groupBy === 'type') {
      const typeConfig = SIGNAL_TYPE_CONFIG[signal.signal_type] || SIGNAL_TYPE_CONFIG.conflict
      key = `type:${signal.signal_type}`
      label = t(`eventsPanel.signalTypes.${typeConfig.labelKey}`)
    } else if (groupBy === 'country') {
      const normalizedCountry = signal.country ? normalizeCountryCode(signal.country) || signal.country.toUpperCase() : 'UNKNOWN'
      key = `country:${normalizedCountry}`
      label = signal.country ? formatCountry(signal.country) : t('eventsPanel.unknownLocation')
    } else if (groupBy === 'severity') {
      const level = severityLevel(signal.severity)
      key = `severity:${level}`
      if (level === 'critical') {
        label = t('eventsPanel.severityRanges.critical')
        order = 0
      } else if (level === 'high') {
        label = t('eventsPanel.severityRanges.high')
        order = 1
      } else if (level === 'medium') {
        label = t('eventsPanel.severityRanges.medium')
        order = 2
      } else {
        label = t('eventsPanel.severityRanges.low')
        order = 3
      }
    } else if (groupBy === 'source') {
      const source = signal.source || t('eventsPanel.unknownSource')
      key = `source:${source.toLowerCase()}`
      label = source
    }

    const existing = groups.get(key)
    if (existing) {
      existing.signals.push(signal)
    } else {
      groups.set(key, { key, label, order, signals: [signal] })
    }
  }

  return [...groups.values()]
    .map((group) => ({
      ...group,
      signals: [...group.signals].sort((a, b) => {
        if (b.severity !== a.severity) return b.severity - a.severity
        return detectedAtValue(b.detected_at) - detectedAtValue(a.detected_at)
      }),
    }))
    .sort((a, b) => {
      if (groupBy === 'severity' && a.order !== b.order) return a.order - b.order
      if (b.signals.length !== a.signals.length) return b.signals.length - a.signals.length
      return a.label.localeCompare(b.label)
    })
}

function SeverityBadge({ severity }: { severity: number }) {
  const level = severityLevel(severity)
  const colors = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    low: 'bg-green-500/20 text-green-400 border-green-500/30',
  }
  return (
    <Badge variant="outline" className={cn('text-[10px] font-mono uppercase', colors[level])}>
      {level} ({(severity * 100).toFixed(0)}%)
    </Badge>
  )
}

function TrendIndicator({ trend }: { trend: string }) {
  if (trend === 'rising') return <TrendingUp className="w-3 h-3 text-red-400" />
  if (trend === 'falling') return <TrendingUp className="w-3 h-3 text-green-400 rotate-180" />
  return <Activity className="w-3 h-3 text-muted-foreground" />
}

function MarketRelevanceBadge({ score }: { score: number | null }) {
  if (score == null) return null
  const color = score >= 0.7
    ? 'bg-green-500/20 text-green-400 border-green-500/30'
    : score >= 0.3
      ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      : 'bg-muted/40 text-muted-foreground border-border'
  return (
    <Badge variant="outline" className={cn('text-[10px] font-mono', color)}>
      MR {(score * 100).toFixed(0)}%
    </Badge>
  )
}

function SignalCard({ signal, layout }: { signal: WorldSignal; layout: SignalsLayout }) {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)
  const config = SIGNAL_TYPE_CONFIG[signal.signal_type] || SIGNAL_TYPE_CONFIG.conflict
  const Icon = config.icon

  const metadataChips = useMemo(() => {
    const chipDefs = METADATA_CHIPS_CONFIG[signal.signal_type] || []
    const meta = signal.metadata
    if (!meta) return []
    return chipDefs
      .filter((def) => meta[def.key] != null && meta[def.key] !== '')
      .map((def) => ({
        key: def.key,
        label: t(`eventsPanel.chipLabels.${def.labelKey}`),
        value: def.format ? def.format(meta[def.key], t) : String(meta[def.key]),
      }))
  }, [signal.signal_type, signal.metadata, t])

  const contextParts = useMemo(() => {
    const parts: string[] = []
    if (signal.country) parts.push(formatCountry(signal.country))
    if (signal.source) parts.push(signal.source)
    if (signal.detected_at) parts.push(new Date(signal.detected_at).toLocaleString())
    return parts
  }, [signal.country, signal.source, signal.detected_at])

  return (
    <div
      className={cn(
        'rounded-lg transition-colors cursor-pointer',
        layout === 'cards'
          ? 'border border-border bg-card/70 hover:bg-card px-3 py-2.5 h-full'
          : 'py-1.5 px-2 bg-background/50 hover:bg-background/80',
      )}
      onClick={() => setExpanded((v) => !v)}
    >
      <div className="flex items-start gap-2">
        <Icon className={cn('w-4 h-4 mt-0.5 shrink-0', config.color)} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-sm font-medium leading-5">{signal.title}</span>
            <Badge variant="outline" className={cn('text-[9px] h-4 px-1.5 font-data', config.color, 'border-current/20')}>
              {t(`eventsPanel.signalTypes.${config.labelKey}`)}
            </Badge>
            <SeverityBadge severity={signal.severity} />
            <MarketRelevanceBadge score={signal.market_relevance_score} />
          </div>
          {contextParts.length > 0 && (
            <div className="text-[10px] text-muted-foreground mt-0.5">
              {contextParts.join(' · ')}
            </div>
          )}
          {signal.related_market_ids && signal.related_market_ids.length > 0 && (
            <div className="flex items-center gap-1 mt-1">
              <MapPin className="w-3 h-3 text-primary" />
              <span className="text-[10px] text-primary">{t('eventsPanel.relatedMarkets', { count: signal.related_market_ids.length })}</span>
            </div>
          )}
        </div>
        <div className="shrink-0 mt-0.5 text-muted-foreground">
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </div>
      </div>
      {expanded && (
        <div className={cn('mt-2 space-y-1.5', layout === 'cards' ? 'pl-0' : 'ml-6')}>
          {signal.description && (
            <p className="text-xs text-muted-foreground">{signal.description}</p>
          )}
          {metadataChips.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              {metadataChips.map((chip) => (
                <Badge key={chip.key} variant="outline" className="text-[9px] h-4 px-1.5 bg-muted/30 border-border/40 font-mono">
                  {chip.label}: {chip.value}
                </Badge>
              ))}
            </div>
          )}
          {signal.latitude != null && signal.longitude != null && (
            <div className="text-[10px] text-muted-foreground font-mono">
              {signal.latitude.toFixed(2)}, {signal.longitude.toFixed(2)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SignalTypeSummaryBar({ signals }: { signals: WorldSignal[] }) {
  const { t } = useTranslation()
  const counts = useMemo(() => {
    const map: Record<string, number> = {}
    for (const s of signals) {
      map[s.signal_type] = (map[s.signal_type] || 0) + 1
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1])
  }, [signals])

  if (counts.length === 0) return null

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {counts.map(([type, count]) => {
        const config = SIGNAL_TYPE_CONFIG[type] || SIGNAL_TYPE_CONFIG.conflict
        return (
          <Badge key={type} variant="outline" className={cn('text-[9px] h-5 px-1.5 gap-1 font-data', config.color, 'bg-transparent border-current/20')}>
            {t(`eventsPanel.signalTypes.${config.labelKey}`)} {count}
          </Badge>
        )
      })}
    </div>
  )
}

// ==================== SIGNALS SUB-VIEW ====================

function SignalsView({ isConnected }: { isConnected: boolean }) {
  const { t } = useTranslation()
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [pageSize, setPageSize] = useState<number>(100)
  const [page, setPage] = useState<number>(1)
  const [groupBy, setGroupBy] = useState<SignalsGroupBy>('type')
  const [layout, setLayout] = useState<SignalsLayout>('cards')
  const offset = (page - 1) * pageSize
  const { data, isLoading, isError } = useQuery({
    queryKey: ['world-signals', { signal_type: typeFilter || undefined, limit: pageSize, offset }],
    queryFn: () => getWorldSignals({ signal_type: typeFilter || undefined, limit: pageSize, offset }),
    refetchInterval: isConnected ? false : 120000,
  })
  const signals = data?.signals || []
  const groupedSignals = useMemo(() => buildSignalGroups(signals, groupBy, t), [signals, groupBy, t])
  const totalSignals = Math.max(Number(data?.total || 0), signals.length)
  const totalPages = Math.max(1, Math.ceil(totalSignals / pageSize))
  const currentPage = Math.min(page, totalPages)
  const currentOffset = (currentPage - 1) * pageSize
  const pageStart = signals.length === 0 ? 0 : currentOffset + 1
  const pageEnd = signals.length === 0 ? 0 : currentOffset + signals.length

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages)
    }
  }, [page, totalPages])

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="shrink-0 space-y-3 border-b border-border/40 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={typeFilter || 'all'}
            onValueChange={(value) => {
              setTypeFilter(value === 'all' ? '' : value)
              setPage(1)
            }}
          >
            <SelectTrigger className="h-8 w-[180px] text-xs">
              <SelectValue placeholder={t('eventsPanel.signalTypePlaceholder')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t('eventsPanel.allSignalTypes')}</SelectItem>
              {Object.entries(SIGNAL_TYPE_CONFIG).map(([type, config]) => (
                <SelectItem key={type} value={type}>
                  {t(`eventsPanel.signalTypes.${config.labelKey}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            value={String(pageSize)}
            onValueChange={(value) => {
              setPageSize(Math.max(1, Number(value) || 100))
              setPage(1)
            }}
          >
          <SelectTrigger className="h-8 w-[180px] text-xs">
            <SelectValue placeholder={t('eventsPanel.pageSizePlaceholder')} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="25">{t('eventsPanel.rows', { count: 25 })}</SelectItem>
            <SelectItem value="50">{t('eventsPanel.rows', { count: 50 })}</SelectItem>
            <SelectItem value="100">{t('eventsPanel.rows', { count: 100 })}</SelectItem>
            <SelectItem value="250">{t('eventsPanel.rows', { count: 250 })}</SelectItem>
            <SelectItem value="500">{t('eventsPanel.rows', { count: 500 })}</SelectItem>
            <SelectItem value="1000">{t('eventsPanel.rows', { count: 1000 })}</SelectItem>
          </SelectContent>
        </Select>
          <Select value={groupBy} onValueChange={(value) => setGroupBy(value as SignalsGroupBy)}>
            <SelectTrigger className="h-8 w-[170px] text-xs">
              <SelectValue placeholder={t('eventsPanel.groupByPlaceholder')} />
            </SelectTrigger>
            <SelectContent>
              {SIGNAL_GROUP_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {t(`eventsPanel.groupOptions.${option.labelKey}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="inline-flex items-center rounded-md border border-border bg-background p-0.5 ml-auto">
            <Button
              variant={layout === 'list' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 px-2 text-[11px]"
              onClick={() => setLayout('list')}
            >
              {t('eventsPanel.layoutList')}
            </Button>
            <Button
              variant={layout === 'cards' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 px-2 text-[11px]"
              onClick={() => setLayout('cards')}
            >
              {t('eventsPanel.layoutCards')}
            </Button>
          </div>
        </div>

        {!isLoading && !isError && signals.length > 0 && (
          <SignalTypeSummaryBar signals={signals} />
        )}

        <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="font-data">
            {t('eventsPanel.showingRange', { start: pageStart, end: pageEnd, total: totalSignals })}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={() => setPage(1)}
              disabled={currentPage <= 1 || isLoading}
            >
              {t('eventsPanel.first')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              disabled={currentPage <= 1 || isLoading}
            >
              {t('eventsPanel.prev')}
            </Button>
            <span className="px-2 text-[11px] font-mono">
              {currentPage} / {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={currentPage >= totalPages || isLoading}
            >
              {t('eventsPanel.next')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={() => setPage(totalPages)}
              disabled={currentPage >= totalPages || isLoading}
            >
              {t('eventsPanel.last')}
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto pt-3 pr-1">
        {isLoading ? (
          <div className="text-center text-muted-foreground py-8">{t('eventsPanel.loadingSignals')}</div>
        ) : isError ? (
          <div className="text-center text-red-400 py-8">{t('eventsPanel.failedSignals')}</div>
        ) : (
          <div className="space-y-3">
            {groupedSignals.map((group) => (
              <section key={group.key} className="space-y-2">
                {groupBy !== 'none' && (
                  <div className="flex items-center justify-between rounded-md border border-border/60 bg-muted/20 px-2.5 py-1.5">
                    <span className="text-[11px] font-data tracking-wide text-muted-foreground">
                      {group.label}
                    </span>
                    <Badge variant="outline" className="text-[9px] h-4 px-1.5 font-mono">
                      {group.signals.length}
                    </Badge>
                  </div>
                )}
                <div
                  className={cn(
                    layout === 'cards'
                      ? 'grid grid-cols-1 gap-2 xl:grid-cols-2 2xl:grid-cols-3'
                      : 'space-y-2',
                  )}
                >
                  {group.signals.map((signal) => (
                    <SignalCard key={signal.signal_id} signal={signal} layout={layout} />
                  ))}
                </div>
              </section>
            ))}
            {groupedSignals.length === 0 && (
              <div className="text-center text-muted-foreground py-8">{t('eventsPanel.noSignalsMatchingFilter')}</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ==================== COUNTRIES SUB-VIEW ====================

function CountriesView({ isConnected }: { isConnected: boolean }) {
  const { t } = useTranslation()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['world-instability'],
    queryFn: () => getInstabilityScores({ min_score: 0, limit: 100 }),
    refetchInterval: isConnected ? false : 180000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">{t('eventsPanel.loadingInstability')}</div>
  if (isError) return <div className="text-center text-red-400 py-8">{t('eventsPanel.failedInstability')}</div>

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-12 gap-2 px-2 text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">
        <div className="col-span-1">{t('eventsPanel.countriesTable.iso3')}</div>
        <div className="col-span-3">{t('eventsPanel.countriesTable.country')}</div>
        <div className="col-span-2 text-right">{t('eventsPanel.countriesTable.ciiScore')}</div>
        <div className="col-span-1 text-center">{t('eventsPanel.countriesTable.trend')}</div>
        <div className="col-span-2 text-right">{t('eventsPanel.countriesTable.change24h')}</div>
        <div className="col-span-3">{t('eventsPanel.countriesTable.topFactor')}</div>
      </div>
      {(data?.scores || []).map((s) => (
        <div key={s.iso3} className="grid grid-cols-12 gap-2 px-2 py-1.5 rounded bg-card border border-border items-center">
          <div className="col-span-1 font-mono text-xs font-bold">{normalizeCountryCode(s.iso3 || s.country) || s.iso3}</div>
          <div className="col-span-3 text-sm truncate">{formatCountry(s.country || s.iso3)}</div>
          <div className={cn('col-span-2 text-right font-mono font-bold text-sm', s.score >= 80 ? 'text-red-400' : s.score >= 60 ? 'text-orange-400' : s.score >= 40 ? 'text-yellow-400' : 'text-green-400')}>
            {s.score.toFixed(1)}
          </div>
          <div className="col-span-1 flex justify-center">
            <TrendIndicator trend={s.trend} />
          </div>
          <div className={cn('col-span-2 text-right font-mono text-xs', (s.change_24h || 0) > 0 ? 'text-red-400' : (s.change_24h || 0) < 0 ? 'text-green-400' : 'text-muted-foreground')}>
            {s.change_24h != null ? `${s.change_24h > 0 ? '+' : ''}${s.change_24h.toFixed(1)}` : '—'}
          </div>
          <div className="col-span-3 text-[10px] text-muted-foreground truncate">
            {s.contributing_signals?.[0] ? JSON.stringify(s.contributing_signals[0]).slice(0, 40) : '—'}
          </div>
        </div>
      ))}
      {(!data?.scores || data.scores.length === 0) && (
        <div className="text-center text-muted-foreground py-8">
          {t('eventsPanel.noInstability')}
        </div>
      )}
    </div>
  )
}

// ==================== TENSIONS SUB-VIEW ====================

function TensionsView({ isConnected }: { isConnected: boolean }) {
  const { t } = useTranslation()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['world-tensions'],
    queryFn: () => getTensionPairs({ min_tension: 0, limit: 20 }),
    refetchInterval: isConnected ? false : 180000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">{t('eventsPanel.loadingTensions')}</div>
  if (isError) return <div className="text-center text-red-400 py-8">{t('eventsPanel.failedTensions')}</div>

  return (
    <div className="space-y-2">
      {(data?.tensions || []).map((tensionRow) => (
        <div key={`${tensionRow.country_a}-${tensionRow.country_b}`} className="p-3 rounded-lg bg-card border border-border">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Swords className="w-4 h-4 text-orange-400" />
              <span className="text-sm font-bold">{formatCountry(tensionRow.country_a_name || tensionRow.country_a_iso3 || tensionRow.country_a)}</span>
              <ChevronRight className="w-3 h-3 text-muted-foreground" />
              <span className="text-sm font-bold">{formatCountry(tensionRow.country_b_name || tensionRow.country_b_iso3 || tensionRow.country_b)}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendIndicator trend={tensionRow.trend} />
              <span className={cn('font-mono text-lg font-bold', tensionRow.tension_score >= 70 ? 'text-red-400' : tensionRow.tension_score >= 40 ? 'text-orange-400' : 'text-yellow-400')}>
                {tensionRow.tension_score.toFixed(0)}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
            <span>{t('eventsPanel.eventsCount', { count: tensionRow.event_count })}</span>
            {tensionRow.avg_goldstein_scale != null && <span>{t('eventsPanel.goldsteinLabel')}: {tensionRow.avg_goldstein_scale.toFixed(1)}</span>}
            {tensionRow.top_event_types?.length > 0 && <span>{tensionRow.top_event_types.slice(0, 3).join(', ')}</span>}
          </div>
        </div>
      ))}
      {(!data?.tensions || data.tensions.length === 0) && (
        <div className="text-center text-muted-foreground py-8">
          {t('eventsPanel.noTensions')}
        </div>
      )}
    </div>
  )
}

// ==================== CONVERGENCES SUB-VIEW ====================

function ConvergencesView({ isConnected }: { isConnected: boolean }) {
  const { t } = useTranslation()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['world-convergences'],
    queryFn: getConvergenceZones,
    refetchInterval: isConnected ? false : 180000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">{t('eventsPanel.loadingConvergences')}</div>
  if (isError) return <div className="text-center text-red-400 py-8">{t('eventsPanel.failedConvergences')}</div>

  return (
    <div className="space-y-2">
      {(data?.zones || []).length === 0 && (
        <div className="text-center text-muted-foreground py-8">{t('eventsPanel.noConvergences')}</div>
      )}
      {(data?.zones || []).map((z) => (
        <div key={z.grid_key} className="p-3 rounded-lg bg-card border border-border">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-medium">{z.country ? formatCountry(z.country) : `${z.latitude.toFixed(1)}, ${z.longitude.toFixed(1)}`}</span>
            </div>
            <Badge variant="outline" className={cn('text-[10px] font-mono', z.urgency_score >= 70 ? 'bg-red-500/20 text-red-400' : z.urgency_score >= 40 ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400')}>
              {t('eventsPanel.urgencyLabel')}: {z.urgency_score.toFixed(0)}
            </Badge>
          </div>
          <div className="flex items-center gap-2 flex-wrap mb-1">
            {z.signal_types.map((type) => {
              const config = SIGNAL_TYPE_CONFIG[type] || SIGNAL_TYPE_CONFIG.conflict
              return (
                <Badge key={type} variant="outline" className={cn('text-[10px]', config.color)}>
                  {t(`eventsPanel.signalTypes.${config.labelKey}`)}
                </Badge>
              )
            })}
          </div>
          <div className="text-[10px] text-muted-foreground">
            {t('eventsPanel.convergenceFooter', { signals: z.signal_count, markets: z.nearby_markets?.length || 0 })}
          </div>
        </div>
      ))}
    </div>
  )
}

// ==================== ANOMALIES SUB-VIEW ====================

function AnomaliesView({ isConnected }: { isConnected: boolean }) {
  const { t } = useTranslation()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['world-anomalies'],
    queryFn: () => getTemporalAnomalies({ min_severity: 'medium' }),
    refetchInterval: isConnected ? false : 180000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">{t('eventsPanel.loadingAnomalies')}</div>
  if (isError) return <div className="text-center text-red-400 py-8">{t('eventsPanel.failedAnomalies')}</div>

  return (
    <div className="space-y-2">
      {(data?.anomalies || []).length === 0 && (
        <div className="text-center text-muted-foreground py-8">{t('eventsPanel.noAnomalies')}</div>
      )}
      {(data?.anomalies || []).map((a, i) => (
        <div key={i} className="p-3 rounded-lg bg-card border border-border">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <Zap className={cn('w-4 h-4', a.severity === 'critical' ? 'text-red-400' : a.severity === 'high' ? 'text-orange-400' : 'text-yellow-400')} />
              <span className="text-sm font-medium">{formatCountry(a.country)} — {a.signal_type.replace(/_/g, ' ')}</span>
            </div>
            <Badge variant="outline" className={cn('text-[10px] font-mono uppercase', a.severity === 'critical' ? 'bg-red-500/20 text-red-400' : a.severity === 'high' ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400')}>
              {t(`eventsPanel.severity.${a.severity}`)}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mb-1">{a.description}</p>
          <div className="flex items-center gap-3 text-[10px] text-muted-foreground font-mono">
            <span>z={a.z_score.toFixed(1)}</span>
            <span>{t('eventsPanel.anomalyCurrent')}={a.current_value}</span>
            <span>{t('eventsPanel.anomalyBaseline')}={a.baseline_mean.toFixed(1)} ± {a.baseline_std.toFixed(1)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

// ==================== MAIN COMPONENT ====================

const SUB_NAV: { id: WorldSubView; labelKey: string; icon: React.ElementType }[] = [
  { id: 'map', labelKey: 'map', icon: MapIcon },
  { id: 'signals', labelKey: 'signals', icon: Radio },
]

export default function EventsPanel({
  isConnected = true,
  eventsOnly = false,
}: {
  isConnected?: boolean
  eventsOnly?: boolean
}) {
  const { t } = useTranslation()
  const [subView, setSubView] = useState<WorldSubView>(eventsOnly ? 'signals' : 'map')

  if (eventsOnly) {
    return (
      <div className="h-full min-h-0 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0 p-4">
          <ErrorBoundary fallback={<div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-500">{t('eventsPanel.eventsViewFailed')}</div>}>
            <SignalsView isConnected={isConnected} />
          </ErrorBoundary>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 flex flex-col overflow-hidden">
      {/* Sub-navigation */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-border bg-card/50 overflow-x-auto shrink-0">
        {SUB_NAV.map((item) => (
          <Button
            key={item.id}
            variant={subView === item.id ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setSubView(item.id)}
            className="h-7 text-xs gap-1 shrink-0"
          >
            <item.icon className="w-3 h-3" />
            {t(`eventsPanel.subNav.${item.labelKey}`)}
          </Button>
        ))}
      </div>

      {/* Content */}
      {subView === 'map' ? (
        <div className="flex-1 min-h-0 relative overflow-hidden">
          <ErrorBoundary fallback={<div className="m-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-500">{t('eventsPanel.mapViewCrashed')}</div>}>
            <Suspense fallback={<div className="h-full w-full" />}>
              <WorldMap isConnected={isConnected} />
            </Suspense>
          </ErrorBoundary>
        </div>
      ) : (
        <div className={cn('flex-1 p-4', subView === 'signals' ? 'min-h-0' : 'overflow-y-auto')}>
          <ErrorBoundary fallback={<div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-500">{t('eventsPanel.eventsViewFailed')}</div>}>
            {subView === 'signals' && <SignalsView isConnected={isConnected} />}
            {subView === 'countries' && <CountriesView isConnected={isConnected} />}
            {subView === 'tensions' && <TensionsView isConnected={isConnected} />}
            {subView === 'convergences' && <ConvergencesView isConnected={isConnected} />}
            {subView === 'anomalies' && <AnomaliesView isConnected={isConnected} />}
          </ErrorBoundary>
        </div>
      )}
    </div>
  )
}
