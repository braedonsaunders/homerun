import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
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
import OpportunityEmptyState from './OpportunityEmptyState'
import {
  getWeatherWorkflowStatus,
  runWeatherWorkflow,
  startWeatherWorkflow,
  pauseWeatherWorkflow,
  getWeatherWorkflowOpportunityDates,
  getWeatherWorkflowOpportunityIds,
  getWeatherWorkflowOpportunities,
  type WeatherOpportunityDateBucket,
  type Opportunity,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import WeatherWorkflowSettingsFlyout from './WeatherWorkflowSettingsFlyout'

const EMPTY_OPPORTUNITIES: Opportunity[] = []

type DirectionFilter = 'all' | 'buy_yes' | 'buy_no'
type TargetDateFilter = 'all' | string
const ITEMS_PER_PAGE = 20
const DATE_PAGE_SIZE = 8
const ANALYZE_ALL_LIMIT = 5000

function useTimeAgo() {
  const { t } = useTranslation()
  return (value: string | null | undefined): string => {
    if (!value) return t('weatherOpportunitiesPanel.time.never')
    const ts = new Date(value).getTime()
    if (Number.isNaN(ts)) return t('weatherOpportunitiesPanel.time.unknown')
    const diff = Math.max(0, Math.floor((Date.now() - ts) / 1000))
    if (diff < 60) return t('weatherOpportunitiesPanel.time.secondsAgo', { n: diff })
    if (diff < 3600) return t('weatherOpportunitiesPanel.time.minutesAgo', { n: Math.floor(diff / 60) })
    if (diff < 86400) return t('weatherOpportunitiesPanel.time.hoursAgo', { n: Math.floor(diff / 3600) })
    return t('weatherOpportunitiesPanel.time.daysAgo', { n: Math.floor(diff / 86400) })
  }
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
  viewMode = 'card',
  showSettingsButton = true,
  onAnalyzeTargetsChange,
}: {
  viewMode?: 'card' | 'list' | 'terminal'
  showSettingsButton?: boolean
  onAnalyzeTargetsChange?: (targets: { visibleIds: string[]; allIds: string[] }) => void
}) {
  const { t } = useTranslation()
  const timeAgo = useTimeAgo()
  const queryClient = useQueryClient()
  const { isConnected } = useWebSocket('/ws')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [direction, setDirection] = useState<DirectionFilter>('all')
  const [city, setCity] = useState('')
  const [minEdge, setMinEdge] = useState(0)
  const [maxEntry, setMaxEntry] = useState('')
  const [showFiltered, setShowFiltered] = useState(false)
  const [targetDate, setTargetDate] = useState<TargetDateFilter>('all')
  const [datePage, setDatePage] = useState(0)
  const [currentPage, setCurrentPage] = useState(0)
  const parsedMaxEntry = Number.parseFloat(maxEntry)
  const maxEntryFilter = Number.isFinite(parsedMaxEntry) && parsedMaxEntry > 0
    ? parsedMaxEntry
    : undefined

  useEffect(() => {
    setCurrentPage(0)
  }, [direction, city, minEdge, maxEntry, targetDate, showFiltered])

  useEffect(() => {
    setDatePage(0)
  }, [direction, city, minEdge, maxEntry, showFiltered])

  useEffect(() => {
    const handleOpenSettings = () => setSettingsOpen(true)
    window.addEventListener('open-weather-workflow-settings', handleOpenSettings as EventListener)
    return () => window.removeEventListener('open-weather-workflow-settings', handleOpenSettings as EventListener)
  }, [])

  const { data: status } = useQuery({
    queryKey: ['weather-workflow-status'],
    queryFn: getWeatherWorkflowStatus,
    refetchInterval: isConnected ? false : 30000,
  })

  const { data: oppData, isLoading: oppsLoading } = useQuery({
    queryKey: ['weather-workflow-opportunities', direction, city, minEdge, maxEntry, targetDate, currentPage, showFiltered],
    queryFn: () => {
      return getWeatherWorkflowOpportunities({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        target_date: targetDate === 'all' ? undefined : targetDate,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntryFilter,
        include_report_only: showFiltered,
        limit: ITEMS_PER_PAGE,
        offset: currentPage * ITEMS_PER_PAGE,
      })
    },
    refetchInterval: isConnected ? false : 30000,
  })

  const { data: dateData, isLoading: dateBucketsLoading } = useQuery({
    queryKey: ['weather-workflow-opportunity-dates', direction, city, minEdge, maxEntry, showFiltered],
    queryFn: () =>
      getWeatherWorkflowOpportunityDates({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntryFilter,
        include_report_only: showFiltered,
      }),
    refetchInterval: isConnected ? false : 30000,
  })

  const refreshMutation = useMutation({
    mutationFn: runWeatherWorkflow,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] }),
        queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] }),
        queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunity-dates'] }),
        queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunity-ids-analyze-all'] }),
      ])
      await Promise.all([
        queryClient.refetchQueries({ queryKey: ['weather-workflow-status'] }),
        queryClient.refetchQueries({ queryKey: ['weather-workflow-opportunities'] }),
        queryClient.refetchQueries({ queryKey: ['weather-workflow-opportunity-dates'] }),
        queryClient.refetchQueries({ queryKey: ['weather-workflow-opportunity-ids-analyze-all'] }),
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

  const opportunities = oppData?.opportunities ?? EMPTY_OPPORTUNITIES
  const totalOpportunities = oppData?.total ?? opportunities.length
  const analyzeAllLimit = Math.max(
    ITEMS_PER_PAGE,
    Math.min(Math.max(totalOpportunities, ITEMS_PER_PAGE), ANALYZE_ALL_LIMIT),
  )
  const { data: allAnalyzeOppData } = useQuery({
    queryKey: [
      'weather-workflow-opportunity-ids-analyze-all',
      direction,
      city,
      minEdge,
      maxEntry,
      targetDate,
      showFiltered,
      analyzeAllLimit,
    ],
    queryFn: () =>
      getWeatherWorkflowOpportunityIds({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        target_date: targetDate === 'all' ? undefined : targetDate,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntryFilter,
        include_report_only: showFiltered,
        limit: analyzeAllLimit,
        offset: 0,
      }),
    enabled: totalOpportunities > 0,
    refetchInterval: isConnected ? false : 30000,
  })
  const visibleAnalyzeIds = useMemo(
    () => Array.from(new Set(opportunities.map((opportunity) => opportunity.id))),
    [opportunities],
  )
  const allAnalyzeIds = useMemo(
    () => Array.from(new Set(allAnalyzeOppData?.ids ?? visibleAnalyzeIds)),
    [allAnalyzeOppData?.ids, visibleAnalyzeIds],
  )
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
  const serverDateBuckets: WeatherOpportunityDateBucket[] = dateData?.dates ?? []
  const availableDateBuckets: WeatherOpportunityDateBucket[] = (
    serverDateBuckets.length > 0
      ? serverDateBuckets
      : pageDateBuckets
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

  const prevAnalyzeRef = useRef<{ visibleIds: string[]; allIds: string[] }>({ visibleIds: [], allIds: [] })
  useEffect(() => {
    const prev = prevAnalyzeRef.current
    if (prev.visibleIds === visibleAnalyzeIds && prev.allIds === allAnalyzeIds) return
    prevAnalyzeRef.current = { visibleIds: visibleAnalyzeIds, allIds: allAnalyzeIds }
    onAnalyzeTargetsChange?.({ visibleIds: visibleAnalyzeIds, allIds: allAnalyzeIds })
  }, [onAnalyzeTargetsChange, visibleAnalyzeIds, allAnalyzeIds])

  useEffect(() => {
    return () => {
      onAnalyzeTargetsChange?.({ visibleIds: [], allIds: [] })
    }
  }, [onAnalyzeTargetsChange])

  const workflowStateLabel = status?.paused
    ? t('weatherOpportunitiesPanel.state.paused')
    : status?.enabled
      ? t('weatherOpportunitiesPanel.state.running')
      : t('weatherOpportunitiesPanel.state.disabled')

  const workflowConnected = Boolean(status?.enabled) && !status?.paused

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-border/40 bg-card/40 p-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <CloudRain className="w-4 h-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
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
            {t('weatherOpportunitiesPanel.lastScan', { time: timeAgo(status?.last_scan) })}
          </Badge>
          <Badge variant="outline" className="text-[10px] h-6 bg-card border-border/60 text-muted-foreground">
            {t('weatherOpportunitiesPanel.oppsCount', { n: totalOpportunities })}
          </Badge>
          {showFiltered && (
            <Badge variant="outline" className="text-[10px] h-6 bg-amber-500/10 text-amber-300 border-amber-500/30">
              {t('weatherOpportunitiesPanel.showingFiltered')}
            </Badge>
          )}

          <select
            value={direction}
            onChange={(e) => setDirection(e.target.value as DirectionFilter)}
            className="h-8 rounded-md border border-border bg-card px-2 text-xs text-foreground"
          >
            <option value="all">{t('weatherOpportunitiesPanel.direction.all')}</option>
            <option value="buy_yes">{t('weatherOpportunitiesPanel.direction.buyYes')}</option>
            <option value="buy_no">{t('weatherOpportunitiesPanel.direction.buyNo')}</option>
          </select>

          <Input
            value={city}
            onChange={(e) => setCity(e.target.value)}
            placeholder={t('weatherOpportunitiesPanel.placeholder.city')}
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
            placeholder={t('weatherOpportunitiesPanel.placeholder.edge')}
          />
          <Input
            type="number"
            min={0}
            max={0.99}
            step={0.01}
            value={maxEntry}
            onChange={(e) => setMaxEntry(e.target.value)}
            className="h-8 w-[82px] text-xs bg-card border-border"
            placeholder={t('weatherOpportunitiesPanel.placeholder.entry')}
          />

          <span className="hidden xl:block text-xs text-muted-foreground truncate max-w-[220px]">
            {status?.current_activity || t('weatherOpportunitiesPanel.waiting')}
          </span>

          <div className="ml-auto flex w-full sm:w-auto items-center justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5 border-cyan-600/30 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 hover:bg-cyan-500/10 hover:text-cyan-700 dark:hover:text-cyan-400"
              onClick={() => refreshMutation.mutate()}
              disabled={refreshMutation.isPending}
            >
              <RefreshCw className={cn("w-3.5 h-3.5", refreshMutation.isPending && "animate-spin")} />
              {t('weatherOpportunitiesPanel.refresh')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5"
              onClick={() => (status?.paused ? startMutation.mutate() : pauseMutation.mutate())}
              disabled={startMutation.isPending || pauseMutation.isPending}
            >
              {status?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
              {status?.paused ? t('weatherOpportunitiesPanel.resume') : t('weatherOpportunitiesPanel.pause')}
            </Button>
            {showSettingsButton && (
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs gap-1.5"
                onClick={() => setSettingsOpen(true)}
              >
                <Settings className="w-3.5 h-3.5" />
                {t('weatherOpportunitiesPanel.settings')}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              className={cn(
                "h-8 text-xs gap-1.5",
                showFiltered
                  ? "border-amber-500/30 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20"
                  : "",
              )}
              onClick={() => setShowFiltered((prev) => !prev)}
            >
              {showFiltered ? t('weatherOpportunitiesPanel.hideFiltered') : t('weatherOpportunitiesPanel.showFiltered')}
            </Button>
          </div>
        </div>
        <div className="mt-2 flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground uppercase tracking-wide shrink-0">
            <CalendarDays className="w-3 h-3 text-cyan-600/80 dark:text-cyan-400/80" />
            {t('weatherOpportunitiesPanel.date')}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 shrink-0"
            onClick={() => setDatePage((p) => p - 1)}
            title={t('weatherOpportunitiesPanel.showEarlierDates')}
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
                {t('weatherOpportunitiesPanel.allDates')}
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
                    title={t('weatherOpportunitiesPanel.filterToDate', { date: key })}
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
                  {t('weatherOpportunitiesPanel.noDatedOpportunities')}
                </span>
              )}
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 shrink-0"
            onClick={() => setDatePage((p) => p + 1)}
            title={t('weatherOpportunitiesPanel.showLaterDates')}
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
            {t('weatherOpportunitiesPanel.loading')}
          </div>
        ) : totalOpportunities === 0 ? (
          <OpportunityEmptyState
            title={showFiltered ? t('weatherOpportunitiesPanel.empty.scannedTitle') : t('weatherOpportunitiesPanel.empty.executableTitle')}
            description={
              showFiltered
                ? t('weatherOpportunitiesPanel.empty.scannedDescription')
                : t('weatherOpportunitiesPanel.empty.executableDescription')
            }
          />
        ) : viewMode === 'terminal' ? (
          <OpportunityTerminal
            opportunities={opportunities}
            isConnected={workflowConnected}
            totalCount={totalOpportunities}
          />
        ) : viewMode === 'list' ? (
          <OpportunityTable
            opportunities={opportunities}
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 card-stagger">
            {opportunities.map((opp) => (
              <OpportunityCard
                key={opp.stable_id || opp.id}
                opportunity={opp}
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
              {t('weatherOpportunitiesPanel.pagination.range', {
                from: currentPage * ITEMS_PER_PAGE + 1,
                to: Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalOpportunities),
                total: totalOpportunities,
              })}
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
                {t('weatherOpportunitiesPanel.pagination.prev')}
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
                {t('weatherOpportunitiesPanel.pagination.next')}
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
