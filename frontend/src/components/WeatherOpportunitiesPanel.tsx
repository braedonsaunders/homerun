import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CloudRain,
  RefreshCw,
  Play,
  Pause,
  Settings,
  ChevronLeft,
  ChevronRight,
  CalendarDays,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Separator } from './ui/separator'
import OpportunityCard from './OpportunityCard'
import OpportunityTable from './OpportunityTable'
import OpportunityTerminal from './OpportunityTerminal'
import {
  getWeatherWorkflowStatus,
  runWeatherWorkflow,
  startWeatherWorkflow,
  pauseWeatherWorkflow,
  getWeatherWorkflowOpportunityDates,
  getWeatherWorkflowOpportunities,
  type WeatherOpportunityDateBucket,
  type Opportunity,
} from '../services/api'
import WeatherWorkflowSettingsFlyout from './WeatherWorkflowSettingsFlyout'

type DirectionFilter = 'all' | 'buy_yes' | 'buy_no'
type TargetDateFilter = 'all' | string
const ITEMS_PER_PAGE = 20
const DATE_PAGE_SIZE = 8

function timeAgo(value: string | null | undefined): string {
  if (!value) return 'Never'
  const ts = new Date(value).getTime()
  if (Number.isNaN(ts)) return 'Unknown'
  const diff = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function parseDateKey(value: string): Date {
  return new Date(`${value}T12:00:00`)
}

function toUtcDateKey(value: string | null | undefined): string | null {
  if (!value) return null
  const raw = String(value).trim()
  if (!raw) return null
  const isoPrefix = raw.match(/^\d{4}-\d{2}-\d{2}/)
  if (isoPrefix) return isoPrefix[0]
  const dt = new Date(raw)
  if (Number.isNaN(dt.getTime())) return null
  return dt.toISOString().slice(0, 10)
}

function opportunityDateKey(opportunity: Opportunity): string | null {
  for (const market of opportunity.markets ?? []) {
    const key = toUtcDateKey(market?.weather?.target_time ?? null)
    if (key) return key
  }
  return toUtcDateKey(opportunity.resolution_date ?? null)
}

function formatDateButtonTop(value: string): string {
  const dt = parseDateKey(value)
  return dt.toLocaleDateString(undefined, { weekday: 'short' }).toUpperCase()
}

function formatDateButtonBottom(value: string): string {
  const dt = parseDateKey(value)
  return dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function formatDateButtonLabel(value: string): string {
  return `${formatDateButtonTop(value)} ${formatDateButtonBottom(value)}`
}

export default function WeatherOpportunitiesPanel({
  onExecute,
  viewMode = 'card',
}: {
  onExecute: (opportunity: Opportunity) => void
  viewMode?: 'card' | 'list' | 'terminal'
}) {
  const queryClient = useQueryClient()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [direction, setDirection] = useState<DirectionFilter>('all')
  const [city, setCity] = useState('')
  const [minEdge, setMinEdge] = useState(0)
  const [maxEntry, setMaxEntry] = useState('')
  const [targetDate, setTargetDate] = useState<TargetDateFilter>('all')
  const [datePage, setDatePage] = useState(0)
  const [currentPage, setCurrentPage] = useState(0)
  const parsedMaxEntry = Number.parseFloat(maxEntry)
  const maxEntryFilter = Number.isFinite(parsedMaxEntry) && parsedMaxEntry > 0
    ? parsedMaxEntry
    : undefined

  useEffect(() => {
    setCurrentPage(0)
  }, [direction, city, minEdge, maxEntry, targetDate])

  useEffect(() => {
    setDatePage(0)
  }, [direction, city, minEdge, maxEntry])

  const { data: status } = useQuery({
    queryKey: ['weather-workflow-status'],
    queryFn: getWeatherWorkflowStatus,
    refetchInterval: 30000,
  })

  const { data: oppData, isLoading: oppsLoading } = useQuery({
    queryKey: ['weather-workflow-opportunities', direction, city, minEdge, maxEntry, targetDate, currentPage],
    queryFn: () => {
      return getWeatherWorkflowOpportunities({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        target_date: targetDate === 'all' ? undefined : targetDate,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntryFilter,
        limit: ITEMS_PER_PAGE,
        offset: currentPage * ITEMS_PER_PAGE,
      })
    },
    refetchInterval: 30000,
  })

  const { data: dateSourceOppData } = useQuery({
    queryKey: ['weather-workflow-opportunity-date-source', direction, city, minEdge, maxEntry],
    queryFn: () =>
      getWeatherWorkflowOpportunities({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntryFilter,
        limit: 500,
        offset: 0,
      }),
    refetchInterval: 30000,
  })

  const { data: dateData, isLoading: dateBucketsLoading } = useQuery({
    queryKey: ['weather-workflow-opportunity-dates', direction, city, minEdge, maxEntry],
    queryFn: () =>
      getWeatherWorkflowOpportunityDates({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntryFilter,
      }),
    refetchInterval: 30000,
  })

  const refreshMutation = useMutation({
    mutationFn: runWeatherWorkflow,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] }),
        queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] }),
      ])
      await Promise.all([
        queryClient.refetchQueries({ queryKey: ['weather-workflow-status'] }),
        queryClient.refetchQueries({ queryKey: ['weather-workflow-opportunities'] }),
      ])
    },
  })

  const startMutation = useMutation({
    mutationFn: startWeatherWorkflow,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
    },
  })

  const pauseMutation = useMutation({
    mutationFn: pauseWeatherWorkflow,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
    },
  })

  const opportunities = oppData?.opportunities ?? []
  const totalOpportunities = oppData?.total ?? opportunities.length
  const totalPages = Math.ceil(totalOpportunities / ITEMS_PER_PAGE)
  const pageDateBuckets = useMemo((): WeatherOpportunityDateBucket[] => {
    const counts = new Map<string, number>()
    for (const opportunity of opportunities) {
      const key = opportunityDateKey(opportunity)
      if (!key) continue
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([date, count]) => ({ date, count }))
  }, [opportunities])
  const derivedDateBuckets = useMemo((): WeatherOpportunityDateBucket[] => {
    const counts = new Map<string, number>()
    const sourceOpportunities = dateSourceOppData?.opportunities ?? []
    for (const opportunity of sourceOpportunities) {
      const key = opportunityDateKey(opportunity)
      if (!key) continue
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([date, count]) => ({ date, count }))
  }, [dateSourceOppData?.opportunities])
  const serverDateBuckets: WeatherOpportunityDateBucket[] = dateData?.dates ?? []
  const availableDateBuckets: WeatherOpportunityDateBucket[] = (
    serverDateBuckets.length > 0
      ? serverDateBuckets
      : (derivedDateBuckets.length > 0 ? derivedDateBuckets : pageDateBuckets)
  )
  const totalDatePages = Math.max(1, Math.ceil(availableDateBuckets.length / DATE_PAGE_SIZE))
  const visibleDateBuckets = useMemo(() => {
    const start = datePage * DATE_PAGE_SIZE
    return availableDateBuckets.slice(start, start + DATE_PAGE_SIZE)
  }, [availableDateBuckets, datePage])
  const activeDateBucket = useMemo(
    () => (
      targetDate === 'all'
        ? null
        : (availableDateBuckets.find((bucket) => bucket.date === targetDate) ?? { date: targetDate, count: 0 })
    ),
    [availableDateBuckets, targetDate]
  )
  const renderDateBuckets = useMemo(() => {
    if (!activeDateBucket) return visibleDateBuckets
    if (visibleDateBuckets.some((bucket) => bucket.date === activeDateBucket.date)) {
      return visibleDateBuckets
    }
    return [activeDateBucket, ...visibleDateBuckets]
  }, [activeDateBucket, visibleDateBuckets])

  useEffect(() => {
    const maxPage = Math.max(totalDatePages - 1, 0)
    if (datePage > maxPage) {
      setDatePage(maxPage)
    }
  }, [datePage, totalDatePages])

  useEffect(() => {
    if (targetDate === 'all') return
    if (availableDateBuckets.length === 0) return
    if (!availableDateBuckets.some((b) => b.date === targetDate)) {
      setTargetDate('all')
    }
  }, [targetDate, availableDateBuckets])

  useEffect(() => {
    if (!oppData) return
    const maxPage = Math.max(totalPages - 1, 0)
    if (currentPage > maxPage) {
      setCurrentPage(maxPage)
    }
  }, [currentPage, totalPages, oppData])

  const workflowStateLabel = status?.paused
    ? 'Paused'
    : status?.enabled
      ? 'Running'
      : 'Disabled'

  const workflowConnected = Boolean(status?.enabled) && !status?.paused

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-border/40 bg-card/40 p-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <CloudRain className="w-4 h-4 text-cyan-400 shrink-0" />
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] h-6',
              status?.paused
                ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
                : status?.enabled
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                  : 'bg-muted/50 text-muted-foreground border-border'
            )}
          >
            {workflowStateLabel}
          </Badge>
          <Badge variant="outline" className="text-[10px] h-6 bg-card border-border/60 text-muted-foreground">
            Last {timeAgo(status?.last_scan)}
          </Badge>
          <Badge variant="outline" className="text-[10px] h-6 bg-card border-border/60 text-muted-foreground">
            Opps {totalOpportunities}
          </Badge>

          <select
            value={direction}
            onChange={(e) => setDirection(e.target.value as DirectionFilter)}
            className="h-8 rounded-md border border-border bg-card px-2 text-xs text-foreground"
          >
            <option value="all">All</option>
            <option value="buy_yes">Buy YES</option>
            <option value="buy_no">Buy NO</option>
          </select>

          <Input
            value={city}
            onChange={(e) => setCity(e.target.value)}
            placeholder="City/location"
            className="h-8 w-[132px] text-xs bg-card border-border"
          />
          <Input
            type="number"
            min={0}
            max={100}
            step={0.5}
            value={minEdge}
            onChange={(e) => setMinEdge(parseFloat(e.target.value) || 0)}
            className="h-8 w-[82px] text-xs bg-card border-border"
            placeholder="Edge%"
          />
          <Input
            type="number"
            min={0}
            max={0.99}
            step={0.01}
            value={maxEntry}
            onChange={(e) => setMaxEntry(e.target.value)}
            className="h-8 w-[82px] text-xs bg-card border-border"
            placeholder="Entry≤"
          />

          <span className="hidden xl:block text-xs text-muted-foreground truncate max-w-[220px]">
            {status?.current_activity || 'Waiting'}
          </span>

          <div className="ml-auto flex w-full sm:w-auto items-center justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10 hover:text-cyan-400"
              onClick={() => refreshMutation.mutate()}
              disabled={refreshMutation.isPending}
            >
              <RefreshCw className={cn("w-3.5 h-3.5", refreshMutation.isPending && "animate-spin")} />
              Refresh
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5"
              onClick={() => (status?.paused ? startMutation.mutate() : pauseMutation.mutate())}
              disabled={startMutation.isPending || pauseMutation.isPending}
            >
              {status?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
              {status?.paused ? 'Resume' : 'Pause'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings className="w-3.5 h-3.5" />
              Settings
            </Button>
          </div>
        </div>
        <div className="mt-2 flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground uppercase tracking-wide shrink-0">
            <CalendarDays className="w-3 h-3 text-cyan-400/80" />
            Date
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 shrink-0"
            onClick={() => setDatePage((p) => p - 1)}
            title="Show earlier dates"
            disabled={datePage === 0}
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </Button>
          <div className="flex-1 overflow-x-auto">
            <div className="flex min-w-max items-center gap-1">
              <button
                type="button"
                onClick={() => setTargetDate('all')}
                className={cn(
                  'h-7 px-2.5 rounded-md border text-[10px] font-medium whitespace-nowrap transition-colors',
                  targetDate === 'all'
                    ? 'border-cyan-300 bg-cyan-100 text-cyan-900 dark:border-cyan-500/40 dark:bg-cyan-500/20 dark:text-cyan-100'
                    : 'border-border bg-card text-muted-foreground hover:text-foreground hover:bg-cyan-50 dark:hover:bg-cyan-900/20'
                )}
              >
                All Dates
              </button>
              {renderDateBuckets.map((bucket) => {
                const key = bucket.date
                const isActive = targetDate === key
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setTargetDate(key)}
                    className={cn(
                      'h-7 min-w-[104px] px-2 rounded-md border text-[10px] leading-none transition-colors',
                      isActive
                        ? 'border-cyan-300 bg-cyan-100 text-cyan-900 dark:border-cyan-500/40 dark:bg-cyan-500/20 dark:text-cyan-100'
                        : 'border-border bg-card text-muted-foreground hover:text-foreground hover:bg-cyan-50 dark:hover:bg-cyan-900/20'
                    )}
                    title={`Filter to ${key}`}
                  >
                    <span className="font-data">{formatDateButtonLabel(key)}</span>
                    <span className={cn(
                      'ml-1 text-[9px]',
                      isActive ? 'text-cyan-800/90 dark:text-cyan-100/90' : 'text-muted-foreground/80'
                    )}>
                      · {bucket.count}
                    </span>
                  </button>
                )
              })}
              {availableDateBuckets.length === 0 && !dateBucketsLoading && !oppsLoading && (
                <span className="h-7 px-2.5 rounded-md border border-border text-[10px] text-muted-foreground inline-flex items-center">
                  No dated opportunities
                </span>
              )}
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 shrink-0"
            onClick={() => setDatePage((p) => p + 1)}
            title="Show later dates"
            disabled={datePage >= totalDatePages - 1 || availableDateBuckets.length === 0}
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      <div className="space-y-2.5">
        {oppsLoading ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground">
            <RefreshCw className="w-4 h-4 animate-spin mr-2" />
            Loading weather opportunities...
          </div>
        ) : totalOpportunities === 0 ? (
          <div className="text-center py-10 border border-border/40 rounded-xl bg-card/20">
            <CloudRain className="w-8 h-8 text-muted-foreground/50 mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No weather opportunities match current filters.</p>
          </div>
        ) : viewMode === 'terminal' ? (
          <OpportunityTerminal
            opportunities={opportunities}
            onExecute={onExecute}
            isConnected={workflowConnected}
            totalCount={totalOpportunities}
          />
        ) : viewMode === 'list' ? (
          <OpportunityTable
            opportunities={opportunities}
            onExecute={onExecute}
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
            {opportunities.map((opp) => (
              <OpportunityCard
                key={opp.stable_id || opp.id}
                opportunity={opp}
                onExecute={onExecute}
              />
            ))}
          </div>
        )}
      </div>

      {totalOpportunities > 0 && (
        <div className="mt-5">
          <Separator />
          <div className="flex items-center justify-between pt-4">
            <div className="text-xs text-muted-foreground">
              {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalOpportunities)} of {totalOpportunities}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs"
                onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                disabled={currentPage === 0}
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                Prev
              </Button>
              <span className="px-2.5 py-1 bg-card rounded-lg text-xs border border-border font-mono">
                {currentPage + 1}/{totalPages || 1}
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs"
                onClick={() => setCurrentPage((p) => p + 1)}
                disabled={currentPage >= totalPages - 1}
              >
                Next
                <ChevronRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        </div>
      )}

      <WeatherWorkflowSettingsFlyout
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}
