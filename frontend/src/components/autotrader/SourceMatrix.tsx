import type { ComponentType } from 'react'
import { Activity, Bot, CloudRain, Copy, Globe2, Newspaper, Radar, TrendingUp, Users } from 'lucide-react'

import type {
  AutoTraderExposure,
  AutoTraderMetrics,
  AutoTraderSourcePolicy,
  CopyTradingStatus,
} from '../../services/api'
import { cn } from '../../lib/utils'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

const SOURCE_ORDER = ['scanner', 'crypto', 'news', 'weather', 'world_intelligence', 'tracked_traders', 'insider', 'copy']

const SOURCE_META: Record<string, { label: string; icon: ComponentType<{ className?: string }> }> = {
  scanner: { label: 'Markets', icon: Radar },
  crypto: { label: 'Crypto', icon: TrendingUp },
  news: { label: 'News', icon: Newspaper },
  weather: { label: 'Weather', icon: CloudRain },
  world_intelligence: { label: 'World Intelligence', icon: Globe2 },
  tracked_traders: { label: 'Tracked Traders', icon: Users },
  insider: { label: 'Insider', icon: Activity },
  copy: { label: 'Copy', icon: Copy },
}

interface SourceMatrixProps {
  policies?: { global: AutoTraderSourcePolicy; sources: Record<string, AutoTraderSourcePolicy> }
  signalStats?: { totals: Record<string, number>; sources: Array<Record<string, any>> }
  metrics?: AutoTraderMetrics
  exposure?: AutoTraderExposure
  copyStatus?: CopyTradingStatus
  updatingSources?: Set<string>
  onToggleSource: (source: string, enabled: boolean) => void
}

export default function SourceMatrix({
  policies,
  signalStats,
  metrics,
  exposure,
  copyStatus,
  updatingSources,
  onToggleSource,
}: SourceMatrixProps) {
  const signalBySource = new Map(
    (signalStats?.sources || []).map((row) => [String(row.source), row])
  )
  const metricsBySource = new Map(
    (metrics?.sources || []).map((row) => [String(row.source), row])
  )
  const exposureBySource = new Map(
    (exposure?.sources || []).map((row) => [String(row.source), row])
  )

  const sources = new Set<string>(SOURCE_ORDER)
  Object.keys(policies?.sources || {}).forEach((source) => sources.add(source))
  for (const row of signalBySource.keys()) sources.add(row)
  for (const row of metricsBySource.keys()) sources.add(row)

  const orderedSources = Array.from(sources).sort((a, b) => {
    const ia = SOURCE_ORDER.indexOf(a)
    const ib = SOURCE_ORDER.indexOf(b)
    if (ia === -1 && ib === -1) return a.localeCompare(b)
    if (ia === -1) return 1
    if (ib === -1) return -1
    return ia - ib
  })

  return (
    <Card className="border-border/50 bg-card/40 w-full h-full min-h-0 flex flex-col">
      <CardHeader className="pb-3 shrink-0">
        <CardTitle className="text-sm">Source Matrix</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 flex-1 min-h-0 overflow-y-auto pr-1">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
          {orderedSources.map((source) => {
            const signal = signalBySource.get(source) || {}
            const sourceMetrics = metricsBySource.get(source)
            const sourceExposure = exposureBySource.get(source)
            const policy = policies?.sources?.[source]
            const sourceEnabled = Boolean(policy?.enabled)
            const pending = Number(signal.pending_count || sourceMetrics?.pending_signals || 0)
            const executed = Number(signal.executed_count || sourceMetrics?.executed || 0)
            const skipped = Number(signal.skipped_count || sourceMetrics?.skipped || 0)
            const budgetUsed = Number(sourceExposure?.budget_used_usd || 0)
            const budgetTotal = Number(sourceExposure?.daily_budget_usd || 0)
            const budgetPct = budgetTotal > 0 ? budgetUsed / budgetTotal : 0

            const meta = SOURCE_META[source] || { label: source, icon: Bot }
            const Icon = meta.icon
            const isCopy = source === 'copy'
            const copyStats = isCopy
              ? {
                  configs: copyStatus?.total_configs || 0,
                  enabled: copyStatus?.enabled_configs || 0,
                }
              : null

            return (
              <div
                key={source}
                className={cn(
                  'rounded-lg border p-3 bg-background/40 transition-colors',
                  sourceEnabled ? 'border-emerald-500/20' : 'border-border/60'
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4 text-muted-foreground" />
                      <p className="text-xs font-semibold truncate">{meta.label}</p>
                      <Badge
                        className={cn(
                          'text-[9px] uppercase tracking-wide',
                          sourceEnabled
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-muted text-muted-foreground'
                        )}
                      >
                        {sourceEnabled ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                    {copyStats && (
                      <p className="text-[11px] text-muted-foreground mt-1">
                        {copyStats.enabled}/{copyStats.configs} copy configs enabled
                      </p>
                    )}
                  </div>

                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onToggleSource(source, !sourceEnabled)}
                    disabled={Boolean(updatingSources?.has(source))}
                    className="h-7 px-2 text-[11px]"
                  >
                    {sourceEnabled ? 'Disable' : 'Enable'}
                  </Button>
                </div>

                <div className="grid grid-cols-3 gap-2 mt-3 text-[11px]">
                  <Metric label="Pending" value={pending} />
                  <Metric label="Executed" value={executed} />
                  <Metric label="Skipped" value={skipped} />
                </div>

                <div className="mt-3">
                  <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                    <span>Budget</span>
                    <span>${budgetUsed.toFixed(1)} / ${budgetTotal.toFixed(1)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted mt-1 overflow-hidden">
                    <div
                      className={cn('h-full rounded-full', budgetPct < 0.85 ? 'bg-emerald-500' : 'bg-amber-500')}
                      style={{ width: `${Math.min(100, Math.max(0, budgetPct * 100))}%` }}
                    />
                  </div>
                </div>

                {sourceMetrics && (
                  <div className="mt-2 text-[10px] text-muted-foreground flex flex-wrap gap-2">
                    <span>Skip {(sourceMetrics.skip_rate * 100).toFixed(1)}%</span>
                    <span>1h {sourceMetrics.decisions_last_hour} decisions</span>
                    <span>{sourceMetrics.avg_decision_to_trade_latency_seconds.toFixed(2)}s latency</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-border/50 bg-background/60 p-2">
      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-xs font-mono font-semibold">{value}</p>
    </div>
  )
}
