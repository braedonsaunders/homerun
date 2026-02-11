import { useState, lazy, Suspense } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Globe,
  AlertTriangle,
  TrendingUp,
  Activity,
  Shield,
  Zap,
  MapPin,
  Radio,
  ChevronRight,
  RefreshCw,
  Flame,
  Swords,
  Waves,
  Wifi,
  Map as MapIcon,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Badge } from './ui/badge'

const WorldMap = lazy(() => import('./WorldMap'))
import {
  getWorldSignals,
  getInstabilityScores,
  getTensionPairs,
  getConvergenceZones,
  getTemporalAnomalies,
  getWorldIntelligenceSummary,
  getWorldIntelligenceStatus,
  WorldSignal,
  InstabilityScore,
  TensionPair,
  ConvergenceZone,
  TemporalAnomaly,
} from '../services/worldIntelligenceApi'

type WorldSubView = 'map' | 'overview' | 'signals' | 'countries' | 'tensions' | 'convergences' | 'anomalies'

const SIGNAL_TYPE_CONFIG: Record<string, { icon: React.ElementType; color: string; label: string }> = {
  conflict: { icon: Swords, color: 'text-red-400', label: 'Conflict' },
  tension: { icon: Activity, color: 'text-orange-400', label: 'Tension' },
  instability: { icon: AlertTriangle, color: 'text-yellow-400', label: 'Instability' },
  convergence: { icon: Radio, color: 'text-purple-400', label: 'Convergence' },
  anomaly: { icon: Zap, color: 'text-cyan-400', label: 'Anomaly' },
  military: { icon: Shield, color: 'text-blue-400', label: 'Military' },
  infrastructure: { icon: Wifi, color: 'text-emerald-400', label: 'Infrastructure' },
}

function SeverityBadge({ severity }: { severity: number }) {
  const level = severity >= 0.7 ? 'critical' : severity >= 0.4 ? 'high' : severity >= 0.2 ? 'medium' : 'low'
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

// ==================== OVERVIEW SUB-VIEW ====================

function OverviewView() {
  const { data: summary, isLoading } = useQuery({
    queryKey: ['world-intelligence-summary'],
    queryFn: getWorldIntelligenceSummary,
    refetchInterval: 60000,
  })

  const { data: signalsData } = useQuery({
    queryKey: ['world-signals', { min_severity: 0.5, limit: 10 }],
    queryFn: () => getWorldSignals({ min_severity: 0.5, limit: 10 }),
    refetchInterval: 60000,
  })

  const { data: statusData } = useQuery({
    queryKey: ['world-intelligence-status'],
    queryFn: getWorldIntelligenceStatus,
    refetchInterval: 30000,
  })

  if (isLoading) {
    return <div className="flex items-center justify-center h-64 text-muted-foreground">Loading world intelligence...</div>
  }

  const stats = statusData?.stats || {}
  const isRunning = statusData?.status?.running

  return (
    <div className="space-y-4">
      {/* Status Bar */}
      <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-card border border-border">
        <div className={cn('w-2 h-2 rounded-full', isRunning ? 'bg-green-400 animate-pulse' : 'bg-muted-foreground')} />
        <span className="text-xs text-muted-foreground">
          {isRunning ? 'Collecting' : 'Offline'} · {stats.total_signals || 0} signals · Last: {summary?.last_collection ? new Date(summary.last_collection).toLocaleTimeString() : 'Never'}
        </span>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Critical Signals" value={summary?.signal_summary?.critical || 0} icon={AlertTriangle} color="text-red-400" />
        <StatCard label="Countries at Risk" value={summary?.critical_countries?.length || 0} icon={Globe} color="text-orange-400" />
        <StatCard label="High Tensions" value={summary?.high_tensions?.length || 0} icon={Swords} color="text-yellow-400" />
        <StatCard label="Anomalies" value={summary?.critical_anomalies || 0} icon={Zap} color="text-cyan-400" />
      </div>

      {/* Critical Countries */}
      {summary?.critical_countries && summary.critical_countries.length > 0 && (
        <div className="p-3 rounded-lg bg-card border border-border">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Critical Countries (CII {'>'}60)</h3>
          <div className="space-y-1.5">
            {summary.critical_countries.map((c) => (
              <div key={c.iso3} className="flex items-center justify-between py-1 px-2 rounded bg-background/50">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold">{c.iso3}</span>
                  <span className="text-sm">{c.country}</span>
                </div>
                <div className="flex items-center gap-2">
                  <TrendIndicator trend={c.trend} />
                  <span className={cn('font-mono text-sm font-bold', c.score >= 80 ? 'text-red-400' : c.score >= 60 ? 'text-orange-400' : 'text-yellow-400')}>
                    {c.score.toFixed(0)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* High Tensions */}
      {summary?.high_tensions && summary.high_tensions.length > 0 && (
        <div className="p-3 rounded-lg bg-card border border-border">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Elevated Tensions</h3>
          <div className="space-y-1.5">
            {summary.high_tensions.map((t) => (
              <div key={t.pair} className="flex items-center justify-between py-1 px-2 rounded bg-background/50">
                <div className="flex items-center gap-2">
                  <Swords className="w-3 h-3 text-orange-400" />
                  <span className="text-sm font-mono">{t.pair}</span>
                </div>
                <div className="flex items-center gap-2">
                  <TrendIndicator trend={t.trend} />
                  <span className="font-mono text-sm font-bold text-orange-400">{t.score.toFixed(0)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top Signals */}
      {signalsData?.signals && signalsData.signals.length > 0 && (
        <div className="p-3 rounded-lg bg-card border border-border">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Top Signals</h3>
          <div className="space-y-2">
            {signalsData.signals.map((s) => (
              <SignalRow key={s.signal_id} signal={s} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, icon: Icon, color }: { label: string; value: number; icon: React.ElementType; color: string }) {
  return (
    <div className="p-3 rounded-lg bg-card border border-border">
      <div className="flex items-center gap-2 mb-1">
        <Icon className={cn('w-3.5 h-3.5', color)} />
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-2xl font-mono font-bold">{value}</span>
    </div>
  )
}

function SignalRow({ signal }: { signal: WorldSignal }) {
  const config = SIGNAL_TYPE_CONFIG[signal.signal_type] || SIGNAL_TYPE_CONFIG.conflict
  const Icon = config.icon

  return (
    <div className="flex items-start gap-2 py-1.5 px-2 rounded bg-background/50 hover:bg-background/80 transition-colors">
      <Icon className={cn('w-4 h-4 mt-0.5 shrink-0', config.color)} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">{signal.title}</span>
          <SeverityBadge severity={signal.severity} />
        </div>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-0.5">
          {signal.country && <span>{signal.country}</span>}
          <span>·</span>
          <span>{signal.source}</span>
          {signal.detected_at && (
            <>
              <span>·</span>
              <span>{new Date(signal.detected_at).toLocaleTimeString()}</span>
            </>
          )}
        </div>
        {signal.related_market_ids && signal.related_market_ids.length > 0 && (
          <div className="flex items-center gap-1 mt-1">
            <MapPin className="w-3 h-3 text-primary" />
            <span className="text-[10px] text-primary">{signal.related_market_ids.length} related market{signal.related_market_ids.length > 1 ? 's' : ''}</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ==================== SIGNALS SUB-VIEW ====================

function SignalsView() {
  const [typeFilter, setTypeFilter] = useState<string>('')
  const { data, isLoading } = useQuery({
    queryKey: ['world-signals', { signal_type: typeFilter || undefined, limit: 100 }],
    queryFn: () => getWorldSignals({ signal_type: typeFilter || undefined, limit: 100 }),
    refetchInterval: 30000,
  })

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Button variant={!typeFilter ? 'default' : 'outline'} size="sm" onClick={() => setTypeFilter('')} className="h-7 text-xs">All</Button>
        {Object.entries(SIGNAL_TYPE_CONFIG).map(([type, config]) => (
          <Button key={type} variant={typeFilter === type ? 'default' : 'outline'} size="sm" onClick={() => setTypeFilter(type)} className="h-7 text-xs gap-1">
            <config.icon className="w-3 h-3" />
            {config.label}
          </Button>
        ))}
      </div>

      {isLoading ? (
        <div className="text-center text-muted-foreground py-8">Loading signals...</div>
      ) : (
        <div className="space-y-2">
          {(data?.signals || []).map((s) => (
            <SignalRow key={s.signal_id} signal={s} />
          ))}
          {(!data?.signals || data.signals.length === 0) && (
            <div className="text-center text-muted-foreground py-8">No signals matching filter</div>
          )}
        </div>
      )}
    </div>
  )
}

// ==================== COUNTRIES SUB-VIEW ====================

function CountriesView() {
  const { data, isLoading } = useQuery({
    queryKey: ['world-instability'],
    queryFn: () => getInstabilityScores({ min_score: 10, limit: 50 }),
    refetchInterval: 60000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">Loading instability scores...</div>

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-12 gap-2 px-2 text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">
        <div className="col-span-1">ISO3</div>
        <div className="col-span-3">Country</div>
        <div className="col-span-2 text-right">CII Score</div>
        <div className="col-span-1 text-center">Trend</div>
        <div className="col-span-2 text-right">24h Change</div>
        <div className="col-span-3">Top Factor</div>
      </div>
      {(data?.scores || []).map((s) => (
        <div key={s.iso3} className="grid grid-cols-12 gap-2 px-2 py-1.5 rounded bg-card border border-border items-center">
          <div className="col-span-1 font-mono text-xs font-bold">{s.iso3}</div>
          <div className="col-span-3 text-sm truncate">{s.country}</div>
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
    </div>
  )
}

// ==================== TENSIONS SUB-VIEW ====================

function TensionsView() {
  const { data, isLoading } = useQuery({
    queryKey: ['world-tensions'],
    queryFn: () => getTensionPairs({ min_tension: 10, limit: 20 }),
    refetchInterval: 60000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">Loading tension data...</div>

  return (
    <div className="space-y-2">
      {(data?.tensions || []).map((t) => (
        <div key={`${t.country_a}-${t.country_b}`} className="p-3 rounded-lg bg-card border border-border">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Swords className="w-4 h-4 text-orange-400" />
              <span className="font-mono text-sm font-bold">{t.country_a}</span>
              <ChevronRight className="w-3 h-3 text-muted-foreground" />
              <span className="font-mono text-sm font-bold">{t.country_b}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendIndicator trend={t.trend} />
              <span className={cn('font-mono text-lg font-bold', t.tension_score >= 70 ? 'text-red-400' : t.tension_score >= 40 ? 'text-orange-400' : 'text-yellow-400')}>
                {t.tension_score.toFixed(0)}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
            <span>{t.event_count} events</span>
            {t.avg_goldstein_scale != null && <span>Goldstein: {t.avg_goldstein_scale.toFixed(1)}</span>}
            {t.top_event_types?.length > 0 && <span>{t.top_event_types.slice(0, 3).join(', ')}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

// ==================== CONVERGENCES SUB-VIEW ====================

function ConvergencesView() {
  const { data, isLoading } = useQuery({
    queryKey: ['world-convergences'],
    queryFn: getConvergenceZones,
    refetchInterval: 60000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">Loading convergence data...</div>

  return (
    <div className="space-y-2">
      {(data?.zones || []).length === 0 && (
        <div className="text-center text-muted-foreground py-8">No active convergence zones detected</div>
      )}
      {(data?.zones || []).map((z) => (
        <div key={z.grid_key} className="p-3 rounded-lg bg-card border border-border">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-medium">{z.country || `${z.latitude.toFixed(1)}, ${z.longitude.toFixed(1)}`}</span>
            </div>
            <Badge variant="outline" className={cn('text-[10px] font-mono', z.urgency_score >= 70 ? 'bg-red-500/20 text-red-400' : z.urgency_score >= 40 ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400')}>
              Urgency: {z.urgency_score.toFixed(0)}
            </Badge>
          </div>
          <div className="flex items-center gap-2 flex-wrap mb-1">
            {z.signal_types.map((type) => {
              const config = SIGNAL_TYPE_CONFIG[type] || SIGNAL_TYPE_CONFIG.conflict
              return (
                <Badge key={type} variant="outline" className={cn('text-[10px]', config.color)}>
                  {config.label}
                </Badge>
              )
            })}
          </div>
          <div className="text-[10px] text-muted-foreground">
            {z.signal_count} signals · {z.nearby_markets?.length || 0} related markets
          </div>
        </div>
      ))}
    </div>
  )
}

// ==================== ANOMALIES SUB-VIEW ====================

function AnomaliesView() {
  const { data, isLoading } = useQuery({
    queryKey: ['world-anomalies'],
    queryFn: () => getTemporalAnomalies({ min_severity: 'medium' }),
    refetchInterval: 60000,
  })

  if (isLoading) return <div className="text-center text-muted-foreground py-8">Loading anomaly data...</div>

  return (
    <div className="space-y-2">
      {(data?.anomalies || []).length === 0 && (
        <div className="text-center text-muted-foreground py-8">No significant anomalies detected</div>
      )}
      {(data?.anomalies || []).map((a, i) => (
        <div key={i} className="p-3 rounded-lg bg-card border border-border">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <Zap className={cn('w-4 h-4', a.severity === 'critical' ? 'text-red-400' : a.severity === 'high' ? 'text-orange-400' : 'text-yellow-400')} />
              <span className="text-sm font-medium">{a.country} — {a.signal_type.replace(/_/g, ' ')}</span>
            </div>
            <Badge variant="outline" className={cn('text-[10px] font-mono uppercase', a.severity === 'critical' ? 'bg-red-500/20 text-red-400' : a.severity === 'high' ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400')}>
              {a.severity}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mb-1">{a.description}</p>
          <div className="flex items-center gap-3 text-[10px] text-muted-foreground font-mono">
            <span>z={a.z_score.toFixed(1)}</span>
            <span>current={a.current_value}</span>
            <span>baseline={a.baseline_mean.toFixed(1)} ± {a.baseline_std.toFixed(1)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

// ==================== MAIN COMPONENT ====================

const SUB_NAV: { id: WorldSubView; label: string; icon: React.ElementType }[] = [
  { id: 'map', label: 'Map', icon: MapIcon },
  { id: 'overview', label: 'Overview', icon: Globe },
  { id: 'signals', label: 'Signals', icon: Radio },
  { id: 'countries', label: 'Countries', icon: MapPin },
  { id: 'tensions', label: 'Tensions', icon: Swords },
  { id: 'convergences', label: 'Convergences', icon: Flame },
  { id: 'anomalies', label: 'Anomalies', icon: Zap },
]

export default function WorldIntelligencePanel() {
  const [subView, setSubView] = useState<WorldSubView>('map')

  return (
    <div className="h-full flex flex-col">
      {/* Sub-navigation */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-border bg-card/50 overflow-x-auto">
        {SUB_NAV.map((item) => (
          <Button
            key={item.id}
            variant={subView === item.id ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setSubView(item.id)}
            className="h-7 text-xs gap-1 shrink-0"
          >
            <item.icon className="w-3 h-3" />
            {item.label}
          </Button>
        ))}
      </div>

      {/* Content */}
      {subView === 'map' ? (
        <div className="flex-1 relative">
          <Suspense fallback={<div className="flex items-center justify-center h-full text-muted-foreground">Loading map...</div>}>
            <WorldMap className="h-full" />
          </Suspense>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-4">
          {subView === 'overview' && <OverviewView />}
          {subView === 'signals' && <SignalsView />}
          {subView === 'countries' && <CountriesView />}
          {subView === 'tensions' && <TensionsView />}
          {subView === 'convergences' && <ConvergencesView />}
          {subView === 'anomalies' && <AnomaliesView />}
        </div>
      )}
    </div>
  )
}
