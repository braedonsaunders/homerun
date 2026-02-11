import { AlertTriangle, Gauge, Layers3 } from 'lucide-react'

import type { AutoTraderExposure, AutoTraderStatus } from '../../services/api'
import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

interface RiskBudgetPanelProps {
  status?: AutoTraderStatus
  exposure?: AutoTraderExposure
}

export default function RiskBudgetPanel({ status, exposure }: RiskBudgetPanelProps) {
  const global = exposure?.global
  const markets = (exposure?.markets || []).slice(0, 5)
  const events = (exposure?.events || []).slice(0, 5)

  return (
    <Card className="border-border/50 bg-card/40 h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Risk & Budget</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 flex-1 min-h-0 overflow-y-auto pr-1">
        <div className="grid grid-cols-2 gap-2 text-xs">
          <Box label="Global Budget" value={`$${(global?.budget_used_usd || 0).toFixed(2)} / $${(global?.daily_budget_usd || 0).toFixed(2)}`} />
          <Box label="Open Positions" value={`${global?.open_positions || 0} / ${global?.max_total_open_positions || 0}`} />
          <Box label="Per-Market Cap" value={`$${(global?.max_per_market_exposure || 0).toFixed(2)}`} />
          <Box label="Per-Event Cap" value={`$${(global?.max_per_event_exposure || 0).toFixed(2)}`} />
        </div>

        <div className="flex items-center gap-2 text-xs">
          <Gauge className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-muted-foreground">Budget Utilization</span>
          <span className="font-mono">{((global?.budget_utilization_pct || 0) * 100).toFixed(1)}%</span>
          {status?.control.kill_switch && (
            <Badge className="bg-rose-500/20 text-rose-300 text-[10px]">
              <AlertTriangle className="w-3 h-3 mr-1" />
              Kill Switch
            </Badge>
          )}
        </div>

        <div className="space-y-3">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1.5">Top Market Exposure</p>
            {markets.length === 0 ? (
              <p className="text-xs text-muted-foreground">No active market exposure.</p>
            ) : (
              <div className="space-y-1.5">
                {markets.map((market) => (
                  <div key={market.market_id} className="flex items-center justify-between text-xs rounded-md border border-border/50 p-2 bg-background/40">
                    <div className="min-w-0">
                      <p className="font-mono truncate">{market.market_id}</p>
                      <p className="text-[10px] text-muted-foreground">{market.directions.join(', ') || 'n/a'}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono">${market.notional_usd.toFixed(2)}</p>
                      <p className="text-[10px] text-muted-foreground">{market.open_positions} pos</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1.5">Top Event Exposure</p>
            {events.length === 0 ? (
              <p className="text-xs text-muted-foreground">No active event exposure.</p>
            ) : (
              <div className="space-y-1.5">
                {events.map((eventRow) => (
                  <div key={eventRow.event_key} className="flex items-center justify-between text-xs rounded-md border border-border/50 p-2 bg-background/40">
                    <div className="min-w-0 flex items-center gap-1.5">
                      <Layers3 className="w-3.5 h-3.5 text-muted-foreground" />
                      <span className="font-mono truncate">{eventRow.event_key}</span>
                    </div>
                    <div className="text-right">
                      <p className="font-mono">${eventRow.notional_usd.toFixed(2)}</p>
                      <p className="text-[10px] text-muted-foreground">{eventRow.open_positions} pos</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function Box({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/40 p-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-xs font-mono font-semibold">{value}</p>
    </div>
  )
}
