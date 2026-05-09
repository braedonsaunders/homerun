import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Zap,
  Activity,
  BarChart3,
  Clock,
  DollarSign,
  FileText,
  Filter,
  Shield,
  Bot,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { ScrollArea } from '../ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import {
  listLLMUsageLog,
  getAIStatus,
  getAIUsage,
  type LLMUsageLogEntry,
} from '../../services/api'

// ── Shared metric helpers ──

type MetricTone = 'good' | 'bad' | 'warn' | 'neutral' | 'info'

const TONE_CLASSES: Record<MetricTone, string> = {
  good: 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200',
  bad: 'border-red-300 bg-red-100 text-red-900 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200',
  warn: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200',
  neutral: 'border-border/70 bg-background/35 text-foreground',
  info: 'border-purple-300 bg-purple-100 text-purple-900 dark:border-purple-500/30 dark:bg-purple-500/10 dark:text-purple-200',
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`
  return num.toString()
}

function MetricTile({ label, value, icon, tone }: { label: string; value: string; icon: React.ReactNode; tone: MetricTone }) {
  return (
    <Card className={cn('border', TONE_CLASSES[tone])}>
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[10px] uppercase tracking-wide opacity-85">{label}</p>
          <span className="opacity-80">{icon}</span>
        </div>
        <p className="mt-2 font-data text-lg font-semibold">{value}</p>
      </CardContent>
    </Card>
  )
}

// ── Usage Overview (merged from AISystemView) ──

function UsageOverview() {
  const { t } = useTranslation()
  const { data: status, isLoading: statusLoading, error: statusError } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const { data } = await getAIStatus()
      return data
    },
    refetchInterval: 30000,
  })
  const { data: usageFallback, isLoading: usageLoading, error: usageError } = useQuery({
    queryKey: ['ai-usage'],
    queryFn: async () => {
      const { data } = await getAIUsage()
      return data
    },
    refetchInterval: 30000,
    enabled: !!status?.enabled && status?.usage == null,
  })
  const usage = status?.usage ?? usageFallback
  const isLoading = statusLoading || (!!status?.enabled && usage == null && usageLoading)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <RefreshCw className="w-5 h-5 animate-spin text-emerald-400" />
      </div>
    )
  }

  if (statusError || usageError) {
    return (
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <CardContent className="p-4">
          <div className="flex items-center gap-3 text-muted-foreground">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">{t('ai.activityView.unableToFetch')}</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!usage) return null

  const cost = usage.estimated_cost ?? usage.total_cost_usd ?? 0
  const failedReq = usage.failed_requests ?? usage.error_count ?? 0

  return (
    <Card className="overflow-hidden border-border/60 bg-card/80">
      <div className="h-0.5 bg-emerald-400" />
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between gap-3 text-base font-semibold">
          <span className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-emerald-400" />
            {t('ai.activityView.usageOverview')}
          </span>
          {usage.active_model && (
            <Badge variant="outline" className="text-xs font-mono border-purple-500/25 bg-purple-500/10 text-purple-300">
              {usage.active_model}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 card-stagger">
          <MetricTile label={t('ai.activityView.metricRequests')} value={String(usage.total_requests ?? 0)} icon={<Zap className="h-3.5 w-3.5" />} tone="info" />
          <MetricTile label={t('ai.activityView.metricInputTokens')} value={formatNumber(usage.total_input_tokens ?? 0)} icon={<FileText className="h-3.5 w-3.5" />} tone="info" />
          <MetricTile label={t('ai.activityView.metricOutputTokens')} value={formatNumber(usage.total_output_tokens ?? 0)} icon={<FileText className="h-3.5 w-3.5" />} tone="info" />
          <MetricTile label={t('ai.activityView.metricCost')} value={`$${cost.toFixed(4)}`} icon={<DollarSign className="h-3.5 w-3.5" />} tone={cost > 5 ? 'warn' : 'neutral'} />
          <MetricTile label={t('ai.activityView.metricLatency')} value={`${(usage.avg_latency_ms ?? 0).toFixed(0)}ms`} icon={<Clock className="h-3.5 w-3.5" />} tone={(usage.avg_latency_ms ?? 0) > 5000 ? 'warn' : 'good'} />
          <MetricTile label={t('ai.activityView.metricTotalTokens')} value={formatNumber(usage.total_tokens ?? 0)} icon={<Activity className="h-3.5 w-3.5" />} tone="neutral" />
          <MetricTile label={t('ai.activityView.metricSuccessful')} value={String(usage.successful_requests ?? usage.total_requests ?? 0)} icon={<CheckCircle className="h-3.5 w-3.5" />} tone="good" />
          <MetricTile label={t('ai.activityView.metricFailed')} value={String(failedReq)} icon={<AlertCircle className="h-3.5 w-3.5" />} tone={failedReq > 0 ? 'bad' : 'good'} />
        </div>

        {usage.spend_limit_usd != null && (
          <div className="rounded-lg border border-border/60 bg-muted/30 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Shield className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('ai.activityView.spendLimit')}</span>
              </div>
              <span className="text-xs font-semibold font-data">
                ${cost.toFixed(2)} / ${usage.spend_limit_usd.toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-background rounded-full h-2 border border-border">
              <div
                className={cn(
                  'h-full rounded-full transition-all',
                  (cost / usage.spend_limit_usd) >= 0.9
                    ? 'bg-red-500'
                    : (cost / usage.spend_limit_usd) >= 0.7
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                )}
                style={{ width: `${Math.min(100, (cost / usage.spend_limit_usd) * 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between mt-1.5">
              <span className="text-[10px] text-muted-foreground font-data">
                {t('ai.activityView.remaining', { amount: `$${(usage.spend_remaining_usd ?? 0).toFixed(2)}` })}
              </span>
              <span className="text-[10px] text-muted-foreground">
                {usage.month_start ? t('ai.activityView.since', { date: new Date(usage.month_start).toLocaleDateString() }) : ''}
              </span>
            </div>
          </div>
        )}

        {usage.by_model && typeof usage.by_model === 'object' && Object.keys(usage.by_model).length > 0 && (
          <div>
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">{t('ai.activityView.byModel')}</p>
            <div className="space-y-1.5">
              {Object.entries(usage.by_model).map(([model, stats]: [string, any]) => {
                const isActive = usage.active_model && model === usage.active_model
                return (
                  <div
                    key={model}
                    className={cn(
                      'flex items-center justify-between p-3 rounded-lg border transition-colors',
                      isActive
                        ? 'bg-purple-500/5 border-purple-500/20 shadow-sm shadow-purple-500/5'
                        : 'bg-background/30 border-border/55 hover:border-border/80'
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium font-mono">{model}</p>
                      {isActive && (
                        <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-purple-500/20 bg-purple-500/10 text-purple-400">
                          {t('ai.activityView.active')}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-data">
                      <span>{stats.requests ?? 0} {t('ai.activityView.reqSuffix')}</span>
                      <span>{formatNumber(stats.tokens ?? 0)} {t('ai.activityView.tokSuffix')}</span>
                      <span className="font-medium text-foreground/80">${(stats.cost ?? 0).toFixed(4)}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ── Call Log Section ──

const PAGE_SIZE = 50

function formatTime(iso: string) {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
      ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

function formatCost(usd: number) {
  if (usd === 0) return '$0'
  if (usd < 0.001) return '<$0.001'
  return `$${usd.toFixed(4)}`
}

function formatLatency(ms: number | null) {
  if (ms === null || ms === undefined) return '-'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function CallLogSection() {
  const { t } = useTranslation()
  const [purposeFilter, setPurposeFilter] = useState<string>('__all__')
  const [providerFilter, setProviderFilter] = useState<string>('__all__')
  const [successFilter, setSuccessFilter] = useState<string>('__all__')
  const [offset, setOffset] = useState(0)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const queryParams = {
    limit: PAGE_SIZE,
    offset,
    purpose: purposeFilter !== '__all__' ? purposeFilter : undefined,
    provider: providerFilter !== '__all__' ? providerFilter : undefined,
    success: successFilter === '__all__' ? undefined : successFilter === 'true',
  }

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['llm-usage-log', queryParams],
    queryFn: () => listLLMUsageLog(queryParams),
    refetchInterval: 30_000,
  })

  const entries = data?.entries ?? []
  const total = data?.total ?? 0

  const purposes = [...new Set(entries.map(e => e.purpose).filter(Boolean))] as string[]
  const providers = [...new Set(entries.map(e => e.provider).filter(Boolean))] as string[]

  const resetOffset = () => setOffset(0)
  const hasPrev = offset > 0
  const hasNext = offset + PAGE_SIZE < total

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <div className="h-0.5 bg-blue-400" />
        <CardContent className="p-3">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-1.5">
              <Filter className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('ai.activityView.filters')}</span>
            </div>

            <Select value={purposeFilter} onValueChange={(v) => { setPurposeFilter(v); resetOffset() }}>
              <SelectTrigger className="h-7 w-[140px] text-xs bg-muted/60 border-border">
                <SelectValue placeholder={t('ai.activityView.purpose')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">{t('ai.activityView.allPurposes')}</SelectItem>
                {purposes.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}
              </SelectContent>
            </Select>

            <Select value={providerFilter} onValueChange={(v) => { setProviderFilter(v); resetOffset() }}>
              <SelectTrigger className="h-7 w-[140px] text-xs bg-muted/60 border-border">
                <SelectValue placeholder={t('ai.activityView.provider')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">{t('ai.activityView.allProviders')}</SelectItem>
                {providers.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}
              </SelectContent>
            </Select>

            <Select value={successFilter} onValueChange={(v) => { setSuccessFilter(v); resetOffset() }}>
              <SelectTrigger className="h-7 w-[120px] text-xs bg-muted/60 border-border">
                <SelectValue placeholder={t('ai.activityView.statusLbl')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">{t('ai.activityView.allStatus')}</SelectItem>
                <SelectItem value="true">{t('ai.activityView.success')}</SelectItem>
                <SelectItem value="false">{t('ai.activityView.error')}</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" size="sm" onClick={() => refetch()} className="h-7 gap-1.5 text-xs ml-auto">
              <RefreshCw className="h-3 w-3" />
              {t('ai.activityView.refresh')}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Call log table */}
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <div className="h-0.5 bg-blue-400" />
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <Clock className="h-4 w-4 text-blue-400" />
            {t('ai.activityView.callLog')}
            <Badge variant="outline" className="text-[10px] ml-2">{t('ai.activityView.totalCount', { count: total })}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 animate-spin text-blue-400" />
            </div>
          ) : entries.length === 0 ? (
            <div className="text-center py-8">
              <Bot className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-xs text-muted-foreground">{t('ai.activityView.noCalls')}</p>
            </div>
          ) : (
            <ScrollArea className="h-[420px]">
              {/* Header */}
              <div className="sticky top-0 z-10 grid grid-cols-[100px_90px_80px_1fr_90px_70px_70px_60px] gap-2 px-4 py-2 text-[10px] uppercase tracking-wide text-muted-foreground bg-card border-b border-border/40">
                <span>{t('ai.activityView.colTime')}</span>
                <span>{t('ai.activityView.purpose')}</span>
                <span>{t('ai.activityView.colProvider')}</span>
                <span>{t('ai.activityView.colModel')}</span>
                <span>{t('ai.activityView.colTokens')}</span>
                <span>{t('ai.activityView.colCost')}</span>
                <span>{t('ai.activityView.colLatency')}</span>
                <span className="text-center">{t('ai.activityView.colStatus')}</span>
              </div>

              {/* Rows */}
              <div className="divide-y divide-border/30">
                {entries.map((entry: LLMUsageLogEntry) => {
                  const isExpanded = expandedRow === entry.id
                  const isError = !entry.success

                  return (
                    <div key={entry.id}>
                      <button
                        onClick={() => setExpandedRow(isExpanded ? null : entry.id)}
                        className={cn(
                          'w-full grid grid-cols-[100px_90px_80px_1fr_90px_70px_70px_60px] gap-2 px-4 py-2 text-left text-xs transition-colors',
                          isError ? 'bg-red-500/5 hover:bg-red-500/10' : 'hover:bg-muted/40'
                        )}
                      >
                        <span className="text-muted-foreground font-data truncate">{formatTime(entry.requested_at)}</span>
                        <span className="truncate">
                          {entry.purpose ? (
                            <Badge variant="outline" className="text-[9px] h-4 px-1.5 bg-muted/30 border-border/40">
                              {entry.purpose}
                            </Badge>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </span>
                        <span className="truncate text-muted-foreground">{entry.provider}</span>
                        <span className="truncate font-data">{entry.model}</span>
                        <span className="font-data text-muted-foreground">{entry.input_tokens}/{entry.output_tokens}</span>
                        <span className="font-data">{formatCost(entry.cost_usd)}</span>
                        <span className="font-data text-muted-foreground">{formatLatency(entry.latency_ms)}</span>
                        <span className="flex justify-center">
                          {entry.success ? (
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                          ) : (
                            <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                          )}
                        </span>
                      </button>

                      {isExpanded && (
                        <div className={cn(
                          'px-4 py-3 text-xs border-t',
                          isError ? 'bg-red-500/5 border-red-500/20' : 'bg-muted/20 border-border/40'
                        )}>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div>
                              <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">{t('ai.activityView.detailId')}</p>
                              <p className="font-data text-foreground/80 break-all">{entry.id}</p>
                            </div>
                            <div>
                              <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">{t('ai.activityView.detailSession')}</p>
                              <p className="font-data text-foreground/80 break-all">{entry.session_id ?? '-'}</p>
                            </div>
                            <div>
                              <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">{t('ai.activityView.detailInputTokens')}</p>
                              <p className="font-data">{entry.input_tokens.toLocaleString()}</p>
                            </div>
                            <div>
                              <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">{t('ai.activityView.detailOutputTokens')}</p>
                              <p className="font-data">{entry.output_tokens.toLocaleString()}</p>
                            </div>
                          </div>
                          {isError && entry.error && (
                            <div className="mt-3 p-2.5 rounded-lg bg-red-500/10 border border-red-500/20">
                              <p className="text-[10px] uppercase tracking-wide text-red-400 mb-1">{t('ai.activityView.detailError')}</p>
                              <p className="text-red-300 font-mono text-[11px] whitespace-pre-wrap">{entry.error}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </ScrollArea>
          )}

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-border/40">
              <p className="text-xs text-muted-foreground">
                {t('ai.activityView.pagination', { from: offset + 1, to: Math.min(offset + PAGE_SIZE, total), total })}
              </p>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" disabled={!hasPrev} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} className="h-7 text-xs">
                  {t('ai.activityView.previous')}
                </Button>
                <Button variant="outline" size="sm" disabled={!hasNext} onClick={() => setOffset(offset + PAGE_SIZE)} className="h-7 text-xs">
                  {t('ai.activityView.next')}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ── Main Activity View ──

export default function AIActivityView() {
  return (
    <div className="space-y-4">
      <UsageOverview />
      <CallLogSection />
    </div>
  )
}
