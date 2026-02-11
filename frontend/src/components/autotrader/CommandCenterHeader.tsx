import { Activity, Layers3, Play, Settings2, ShieldAlert, Square } from 'lucide-react'

import type { AutoTraderStatus } from '../../services/api'
import { cn } from '../../lib/utils'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Card, CardContent } from '../ui/card'

interface CommandCenterHeaderProps {
  status?: AutoTraderStatus
  canStart: boolean
  startPending: boolean
  stopPending: boolean
  emergencyPending: boolean
  onOpenSettings: () => void
  onOpenSources: () => void
  onStart: () => void
  onStop: () => void
  onEmergencyStop: () => void
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export default function CommandCenterHeader({
  status,
  canStart,
  startPending,
  stopPending,
  emergencyPending,
  onOpenSettings,
  onOpenSources,
  onStart,
  onStop,
  onEmergencyStop,
}: CommandCenterHeaderProps) {
  const stats = status?.stats
  const tradingActive = Boolean(status?.trading_active)

  return (
    <Card className="border-border/50 bg-card/40">
      <CardContent className="p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                'w-2.5 h-2.5 rounded-full',
                tradingActive ? 'bg-emerald-500 animate-pulse' : 'bg-muted-foreground'
              )}
            />
            <div>
              <p className="text-sm font-semibold">AutoTrader Command Center</p>
              <p className="text-xs text-muted-foreground">
                Manual start required after launch.
              </p>
            </div>
            <Badge
              className={cn(
                'text-[10px] uppercase tracking-wider',
                tradingActive ? 'bg-emerald-500/20 text-emerald-300' : 'bg-muted text-muted-foreground'
              )}
            >
              {tradingActive ? 'Active' : 'Stopped'}
            </Badge>
            {status?.worker_running && !tradingActive && (
              <Badge className="text-[10px] bg-amber-500/20 text-amber-300">Worker Ready</Badge>
            )}
          </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={onOpenSettings}
            className="h-8 px-3 text-xs"
          >
            <Settings2 className="w-3.5 h-3.5 mr-1" />
            Settings
          </Button>
          <Button
            variant="outline"
            onClick={onOpenSources}
            className="h-8 px-3 text-xs"
          >
            <Layers3 className="w-3.5 h-3.5 mr-1" />
            Sources
          </Button>
          {tradingActive ? (
            <Button
                variant="secondary"
                onClick={onStop}
                disabled={stopPending}
                className="h-8 px-3 text-xs"
              >
                <Square className="w-3.5 h-3.5 mr-1" />
                Stop
              </Button>
            ) : (
              <Button
                onClick={onStart}
                disabled={!canStart || startPending}
                className="h-8 px-3 text-xs bg-emerald-600 hover:bg-emerald-500"
              >
                <Play className="w-3.5 h-3.5 mr-1" />
                Start
              </Button>
            )}
            <Button
              variant="outline"
              onClick={onEmergencyStop}
              disabled={emergencyPending}
              className="h-8 px-3 text-xs border-red-500/40 text-red-300 hover:bg-red-500/10"
            >
              <ShieldAlert className="w-3.5 h-3.5 mr-1" />
              Emergency Stop
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
          <MetricCard label="Signals Seen" value={stats?.opportunities_seen ?? 0} />
          <MetricCard label="Selected" value={stats?.opportunities_executed ?? 0} />
          <MetricCard label="Skipped" value={stats?.opportunities_skipped ?? 0} />
          <MetricCard
            label="Win Rate"
            value={formatPct(stats?.win_rate ?? 0)}
            tone={stats && stats.win_rate >= 0.5 ? 'good' : 'neutral'}
          />
          <MetricCard
            label="Daily P&L"
            value={`$${(stats?.daily_profit ?? 0).toFixed(2)}`}
            tone={(stats?.daily_profit ?? 0) >= 0 ? 'good' : 'bad'}
          />
        </div>

        {status?.snapshot?.current_activity && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Activity className="w-3.5 h-3.5" />
            <span>{status.snapshot.current_activity}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function MetricCard({
  label,
  value,
  tone = 'neutral',
}: {
  label: string
  value: number | string
  tone?: 'neutral' | 'good' | 'bad'
}) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/50 p-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p
        className={cn(
          'text-sm font-mono font-semibold',
          tone === 'good' && 'text-emerald-300',
          tone === 'bad' && 'text-rose-300'
        )}
      >
        {value}
      </p>
    </div>
  )
}
