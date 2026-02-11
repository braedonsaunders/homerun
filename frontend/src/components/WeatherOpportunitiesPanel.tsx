import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CloudRain,
  RefreshCw,
  Play,
  Pause,
  Settings,
  MapPin,
  Timer,
  TrendingUp,
  Target,
  Bot,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Card } from './ui/card'
import { Input } from './ui/input'
import {
  getWeatherWorkflowStatus,
  runWeatherWorkflow,
  startWeatherWorkflow,
  pauseWeatherWorkflow,
  getWeatherWorkflowOpportunities,
  getWeatherWorkflowIntents,
  skipWeatherWorkflowIntent,
  getWeatherWorkflowPerformance,
  type Opportunity,
  type WeatherTradeIntent,
} from '../services/api'
import WeatherWorkflowSettingsFlyout from './WeatherWorkflowSettingsFlyout'

type DirectionFilter = 'all' | 'buy_yes' | 'buy_no'

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

function fmtDateTime(value: string | null | undefined): string {
  if (!value) return 'N/A'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return 'N/A'
  return d.toLocaleString()
}

function WeatherIntentRow({
  intent,
  onSkip,
  isSkipping,
}: {
  intent: WeatherTradeIntent
  onSkip: (id: string) => void
  isSkipping: boolean
}) {
  const isBuyYes = intent.direction === 'buy_yes'
  return (
    <div className="flex items-center justify-between gap-3 p-2.5 rounded-lg border border-border/40 bg-card/30">
      <div className="min-w-0">
        <p className="text-xs font-medium text-foreground line-clamp-1">{intent.market_question}</p>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-1">
          <Badge
            variant="outline"
            className={cn(
              'text-[9px] h-4 px-1.5',
              isBuyYes
                ? 'bg-green-500/10 text-green-400 border-green-500/20'
                : 'bg-red-500/10 text-red-400 border-red-500/20'
            )}
          >
            {isBuyYes ? 'BUY YES' : 'BUY NO'}
          </Badge>
          <span className="font-data">Edge {(intent.edge_percent ?? 0).toFixed(1)}%</span>
          <span className="font-data">${(intent.suggested_size_usd ?? 0).toFixed(0)}</span>
          <span className="font-data">{timeAgo(intent.created_at)}</span>
        </div>
      </div>
      {intent.status === 'pending' && (
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1"
          onClick={() => onSkip(intent.id)}
          disabled={isSkipping}
        >
          <X className="w-3 h-3" />
          Skip
        </Button>
      )}
    </div>
  )
}

export default function WeatherOpportunitiesPanel({
  onExecute,
}: {
  onExecute: (opportunity: Opportunity) => void
}) {
  const queryClient = useQueryClient()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [direction, setDirection] = useState<DirectionFilter>('all')
  const [city, setCity] = useState('')
  const [minEdge, setMinEdge] = useState(0)
  const [maxEntry, setMaxEntry] = useState(0.25)

  const { data: status } = useQuery({
    queryKey: ['weather-workflow-status'],
    queryFn: getWeatherWorkflowStatus,
    refetchInterval: 30000,
  })

  const { data: oppData, isLoading: oppsLoading } = useQuery({
    queryKey: ['weather-workflow-opportunities', direction, city, minEdge, maxEntry],
    queryFn: () =>
      getWeatherWorkflowOpportunities({
        direction: direction === 'all' ? undefined : direction,
        location: city.trim() || undefined,
        min_edge: minEdge > 0 ? minEdge : undefined,
        max_entry: maxEntry > 0 ? maxEntry : undefined,
        limit: 200,
      }),
    refetchInterval: 30000,
  })

  const { data: intentsData } = useQuery({
    queryKey: ['weather-workflow-intents'],
    queryFn: () => getWeatherWorkflowIntents({ limit: 100 }),
    refetchInterval: 15000,
  })

  const { data: perf } = useQuery({
    queryKey: ['weather-workflow-performance'],
    queryFn: () => getWeatherWorkflowPerformance(90),
    refetchInterval: 30000,
  })

  const runMutation = useMutation({
    mutationFn: runWeatherWorkflow,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-performance'] })
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

  const skipMutation = useMutation({
    mutationFn: skipWeatherWorkflowIntent,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
    },
  })

  const opportunities = oppData?.opportunities ?? []
  const intents = intentsData?.intents ?? []
  const pendingIntents = useMemo(
    () => intents.filter((i) => i.status === 'pending'),
    [intents]
  )

  const nextScanAt = useMemo(() => {
    if (!status?.last_scan || status.paused) return null
    const lastMs = new Date(status.last_scan).getTime()
    if (Number.isNaN(lastMs)) return null
    return new Date(lastMs + status.interval_seconds * 1000).toISOString()
  }, [status?.interval_seconds, status?.last_scan, status?.paused])

  const workflowStateLabel = status?.paused
    ? 'Paused'
    : status?.enabled
      ? 'Running'
      : 'Disabled'

  return (
    <div className="space-y-4">
      <Card className="border-border/40 bg-card/50">
        <div className="p-3 flex flex-wrap items-center gap-2 justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <CloudRain className="w-4 h-4 text-cyan-400 shrink-0" />
            <Badge
              variant="outline"
              className={cn(
                'text-[10px] h-5',
                status?.paused
                  ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
                  : status?.enabled
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    : 'bg-muted/50 text-muted-foreground border-border'
              )}
            >
              {workflowStateLabel}
            </Badge>
            <span className="text-xs text-muted-foreground truncate">
              {status?.current_activity || 'Waiting for weather worker'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10 hover:text-cyan-400"
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
            >
              {runMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              Run Now
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
        <div className="px-3 pb-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px] text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <Timer className="w-3.5 h-3.5 text-blue-400" />
            Last scan: <span className="font-data">{timeAgo(status?.last_scan)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Timer className="w-3.5 h-3.5 text-violet-400" />
            Next scan: <span className="font-data">{fmtDateTime(nextScanAt)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Target className="w-3.5 h-3.5 text-emerald-400" />
            Interval: <span className="font-data">{status?.interval_seconds ?? 0}s</span>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
        <Card className="border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Opportunities</p>
          <p className="text-lg font-data font-semibold text-foreground mt-0.5">{opportunities.length}</p>
        </Card>
        <Card className="border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Pending Intents</p>
          <p className="text-lg font-data font-semibold text-yellow-400 mt-0.5">
            {status?.pending_intents ?? pendingIntents.length}
          </p>
        </Card>
        <Card className="border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Executed Intents</p>
          <p className="text-lg font-data font-semibold text-cyan-400 mt-0.5">{perf?.executed_intents ?? 0}</p>
        </Card>
        <Card className="border-border/40 bg-card/40 p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Win Rate (90d)</p>
          <p className="text-lg font-data font-semibold text-emerald-400 mt-0.5">
            {((perf?.win_rate ?? 0) * 100).toFixed(1)}%
          </p>
        </Card>
      </div>

      <Card className="border-border/40 bg-card/40 p-3 space-y-3">
        <div className="flex items-center gap-1.5">
          <Target className="w-3.5 h-3.5 text-blue-400" />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">Filters</h4>
        </div>
        <div className="flex flex-wrap items-end gap-2.5">
          <div>
            <p className="text-[10px] text-muted-foreground mb-1">Direction</p>
            <div className="flex items-center gap-1">
              {([
                ['all', 'All'],
                ['buy_yes', 'BUY YES'],
                ['buy_no', 'BUY NO'],
              ] as const).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setDirection(key)}
                  className={cn(
                    'px-2.5 py-1 rounded-md text-[10px] border transition-colors',
                    direction === key
                      ? 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
                      : 'bg-card text-muted-foreground border-border hover:text-foreground'
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="w-48">
            <p className="text-[10px] text-muted-foreground mb-1">City/Location</p>
            <Input
              value={city}
              onChange={(e) => setCity(e.target.value)}
              placeholder="e.g. New York"
              className="h-8 text-xs"
            />
          </div>
          <div className="w-32">
            <p className="text-[10px] text-muted-foreground mb-1">Min Edge %</p>
            <Input
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={minEdge}
              onChange={(e) => setMinEdge(parseFloat(e.target.value) || 0)}
              className="h-8 text-xs"
            />
          </div>
          <div className="w-32">
            <p className="text-[10px] text-muted-foreground mb-1">Max Entry</p>
            <Input
              type="number"
              min={0.01}
              max={0.99}
              step={0.01}
              value={maxEntry}
              onChange={(e) => setMaxEntry(parseFloat(e.target.value) || 0.25)}
              className="h-8 text-xs"
            />
          </div>
        </div>
      </Card>

      <div className="space-y-2.5">
        {oppsLoading ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground">
            <RefreshCw className="w-4 h-4 animate-spin mr-2" />
            Loading weather opportunities...
          </div>
        ) : opportunities.length === 0 ? (
          <div className="text-center py-10 border border-border/40 rounded-xl bg-card/20">
            <CloudRain className="w-8 h-8 text-muted-foreground/50 mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No weather opportunities match current filters.</p>
          </div>
        ) : (
          opportunities.map((opp) => {
            const position = opp.positions_to_take?.[0]
            const market = opp.markets?.[0] as
              | {
                  weather?: {
                    location?: string
                    target_time?: string
                    agreement?: number
                    gfs_probability?: number
                    ecmwf_probability?: number
                  }
                }
              | undefined
            const weather = market?.weather
            const entry = position?.price ?? opp.total_cost
            const edgePct = ((opp.expected_payout - entry) * 100)
            const confidence = 1 - Math.max(0, Math.min(1, opp.risk_score))
            const directionLabel = position?.outcome === 'YES' ? 'BUY YES' : 'BUY NO'

            return (
              <Card key={opp.id} className="border-border/40 bg-card/35 p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground line-clamp-2">{opp.title}</p>
                    <div className="flex items-center gap-2 text-[11px] text-muted-foreground mt-1">
                      <MapPin className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                      <span className="truncate">{weather?.location || 'Unknown location'}</span>
                      <span className="font-data shrink-0">{fmtDateTime(weather?.target_time || null)}</span>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-lg font-data font-semibold text-emerald-400">{edgePct.toFixed(1)}%</p>
                    <p className="text-[10px] text-muted-foreground">edge</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mt-3 text-[10px]">
                  <div className="rounded-lg bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground uppercase tracking-wider">Direction</p>
                    <p className="font-data text-foreground">{directionLabel}</p>
                  </div>
                  <div className="rounded-lg bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground uppercase tracking-wider">Entry</p>
                    <p className="font-data text-foreground">${entry.toFixed(2)}</p>
                  </div>
                  <div className="rounded-lg bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground uppercase tracking-wider">Confidence</p>
                    <p className="font-data text-foreground">{(confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div className="rounded-lg bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground uppercase tracking-wider">Agreement</p>
                    <p className="font-data text-foreground">{(((weather?.agreement as number) || 0) * 100).toFixed(0)}%</p>
                  </div>
                  <div className="rounded-lg bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground uppercase tracking-wider">GFS</p>
                    <p className="font-data text-foreground">{(((weather?.gfs_probability as number) || 0) * 100).toFixed(0)}%</p>
                  </div>
                  <div className="rounded-lg bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground uppercase tracking-wider">ECMWF</p>
                    <p className="font-data text-foreground">{(((weather?.ecmwf_probability as number) || 0) * 100).toFixed(0)}%</p>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                    <Bot className="w-3.5 h-3.5 text-violet-400" />
                    Suggested size: <span className="font-data text-foreground">${opp.max_position_size.toFixed(0)}</span>
                  </div>
                  <Button
                    size="sm"
                    className="h-8 text-xs gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white"
                    onClick={() => onExecute(opp)}
                  >
                    <TrendingUp className="w-3.5 h-3.5" />
                    Trade
                  </Button>
                </div>
              </Card>
            )
          })
        )}
      </div>

      {pendingIntents.length > 0 && (
        <Card className="border-border/40 bg-card/35 p-3 space-y-2">
          <div className="flex items-center gap-1.5">
            <Bot className="w-3.5 h-3.5 text-yellow-400" />
            <h4 className="text-[10px] uppercase tracking-widest font-semibold">Pending Intents</h4>
          </div>
          <div className="space-y-2">
            {pendingIntents.slice(0, 8).map((intent) => (
              <WeatherIntentRow
                key={intent.id}
                intent={intent}
                onSkip={(id) => skipMutation.mutate(id)}
                isSkipping={skipMutation.isPending}
              />
            ))}
          </div>
        </Card>
      )}

      <WeatherWorkflowSettingsFlyout
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}
