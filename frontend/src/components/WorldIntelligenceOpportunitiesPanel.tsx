import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, CheckCircle2, Globe2, RefreshCw, XCircle } from 'lucide-react'

import { cn } from '../lib/utils'
import { formatCountry } from '../lib/worldCountries'
import {
  getWorldIntelligenceOpportunities,
  type WorldIntelligenceOpportunity,
} from '../services/worldIntelligenceApi'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'

const SIGNAL_TYPE_OPTIONS = [
  { value: '', label: 'All types' },
  { value: 'conflict', label: 'Conflict' },
  { value: 'tension', label: 'Tension' },
  { value: 'instability', label: 'Instability' },
  { value: 'convergence', label: 'Convergence' },
  { value: 'anomaly', label: 'Anomaly' },
  { value: 'military', label: 'Military' },
  { value: 'infrastructure', label: 'Infrastructure' },
]

function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return `${(value * 100).toFixed(1)}%`
}

function signalTone(signal: WorldIntelligenceOpportunity): string {
  if (signal.tradable) return 'border-emerald-500/35 bg-emerald-500/5'
  if (signal.severity >= 0.8) return 'border-red-500/35 bg-red-500/5'
  return 'border-border bg-card/40'
}

export default function WorldIntelligenceOpportunitiesPanel() {
  const [tradableOnly, setTradableOnly] = useState(false)
  const [signalType, setSignalType] = useState('')
  const [minSeverity, setMinSeverity] = useState(0.5)

  const queryParams = useMemo(
    () => ({
      tradable_only: tradableOnly,
      signal_type: signalType || undefined,
      min_severity: minSeverity,
      min_relevance: 0.3,
      hours: 72,
      limit: 250,
    }),
    [tradableOnly, signalType, minSeverity]
  )

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['world-intel-opportunities', queryParams],
    queryFn: () => getWorldIntelligenceOpportunities(queryParams),
    refetchInterval: 30000,
  })

  const rows = data?.opportunities || []
  const summary = data?.summary || {}
  const shownCount = Number(summary.returned ?? rows.length)
  const tradableCount = Number(summary.tradable ?? rows.filter((row) => row.tradable).length)
  const avgSeverity = rows.length > 0
    ? rows.reduce((sum, row) => sum + Number(row.severity || 0), 0) / rows.length
    : 0
  const highSeverityCount = rows.filter((row) => Number(row.severity || 0) >= 0.8).length

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border/40 bg-card/60 p-3">
        <div className="flex flex-wrap items-center gap-2 justify-between">
          <div className="flex items-center gap-2">
            <Globe2 className="w-4 h-4 text-cyan-400" />
            <p className="text-sm font-semibold">World Intelligence Opportunities</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
            className="h-7 text-xs"
          >
            <RefreshCw className={cn('w-3.5 h-3.5 mr-1', isFetching && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="rounded-lg border border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Tradable</p>
          <p className="text-lg font-semibold text-emerald-300">{tradableCount}</p>
        </div>
        <div className="rounded-lg border border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Shown</p>
          <p className="text-lg font-semibold text-foreground">{shownCount}</p>
        </div>
        <div className="rounded-lg border border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Severity</p>
          <p className="text-lg font-semibold text-foreground">{(avgSeverity * 100).toFixed(1)}%</p>
        </div>
        <div className="rounded-lg border border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">High Severity</p>
          <p className="text-lg font-semibold text-amber-300">{highSeverityCount}</p>
        </div>
      </div>

      <div className="rounded-xl border border-border/40 bg-card/40 p-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <label className="text-xs text-muted-foreground">
            Signal Type
            <select
              value={signalType}
              onChange={(e) => setSignalType(e.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs"
            >
              {SIGNAL_TYPE_OPTIONS.map((option) => (
                <option key={option.value || 'all'} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted-foreground">
            Min Severity
            <Input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minSeverity}
              onChange={(e) => setMinSeverity(Number(e.target.value) || 0)}
              className="mt-1 h-8 text-xs"
            />
          </label>
          <label className="text-xs text-muted-foreground">
            Filters
            <button
              type="button"
              onClick={() => setTradableOnly((prev) => !prev)}
              className={cn(
                'mt-1 w-full rounded-md border px-2 py-1.5 text-xs text-left transition-colors',
                tradableOnly
                  ? 'border-emerald-500/35 bg-emerald-500/10 text-emerald-300'
                  : 'border-border bg-background text-muted-foreground'
              )}
            >
              {tradableOnly ? 'Tradable only' : 'Tradable + unresolved'}
            </button>
          </label>
        </div>
      </div>

      {isLoading ? (
        <div className="py-12 text-center text-muted-foreground">Loading opportunities...</div>
      ) : isError ? (
        <div className="rounded-lg border border-red-500/35 bg-red-500/5 p-4 text-sm text-red-300">
          Failed to load world intelligence opportunities.
        </div>
      ) : rows.length === 0 ? (
        <div className="rounded-lg border border-border/40 bg-card/40 p-6 text-center text-sm text-muted-foreground">
          No opportunities match the current filters.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {rows.map((row) => (
            <div key={row.id} className={cn('rounded-lg border p-3 space-y-2', signalTone(row))}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-sm font-semibold truncate">{row.title}</p>
                  <p className="text-[11px] text-muted-foreground truncate">
                    {row.market_question || row.market_id}
                  </p>
                </div>
                {row.tradable ? (
                  <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/35 gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    Tradable
                  </Badge>
                ) : (
                  <Badge className="bg-amber-500/20 text-amber-300 border-amber-500/35 gap-1">
                    <XCircle className="w-3 h-3" />
                    Incomplete
                  </Badge>
                )}
              </div>

              <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <span>{row.signal_type}</span>
                <span>·</span>
                <span>{row.country ? formatCountry(row.country) : 'Global'}</span>
                <span>·</span>
                <span>{new Date(row.detected_at || '').toLocaleString()}</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <Metric label="Severity" value={formatPct(row.severity)} />
                <Metric label="Relevance" value={formatPct(row.market_relevance_score)} />
                <Metric label="Confidence" value={formatPct(row.confidence)} />
                <Metric label="Edge" value={`${row.edge_percent.toFixed(1)}%`} />
              </div>

              <div className="rounded-md border border-border/60 bg-background/50 p-2 text-[11px]">
                <p>
                  <span className="text-muted-foreground">Direction:</span>{' '}
                  {row.direction || 'n/a'}
                </p>
                <p>
                  <span className="text-muted-foreground">Entry:</span>{' '}
                  {row.entry_price != null ? row.entry_price.toFixed(3) : 'n/a'}
                </p>
                <p className="truncate">
                  <span className="text-muted-foreground">Token:</span>{' '}
                  {row.token_id || 'n/a'}
                </p>
                {!row.tradable && row.missing_fields.length > 0 && (
                  <p className="mt-1 text-amber-300 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" />
                    Missing: {row.missing_fields.join(', ')}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/60 bg-background/50 px-2 py-1.5">
      <p className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="font-mono text-xs font-semibold">{value}</p>
    </div>
  )
}
