import { ArrowDownRight, ArrowUpRight, Clock3 } from 'lucide-react'

import type { AutoTraderTrade } from '../../services/api'
import { cn } from '../../lib/utils'
import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

interface ExecutionTapeProps {
  trades: AutoTraderTrade[]
  maxItems?: number
}

const statusColor: Record<string, string> = {
  executed: 'bg-emerald-500/20 text-emerald-300',
  submitted: 'bg-sky-500/20 text-sky-300',
  failed: 'bg-rose-500/20 text-rose-300',
  skipped: 'bg-amber-500/20 text-amber-300',
}

export default function ExecutionTape({ trades, maxItems = 10 }: ExecutionTapeProps) {
  const rows = trades.slice(0, maxItems)

  return (
    <Card className="border-border/50 bg-card/40 h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Execution Tape</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        {rows.length === 0 ? (
          <div className="h-full min-h-[220px] flex items-center justify-center text-sm text-muted-foreground">
            No trades yet.
          </div>
        ) : (
          <div className="h-full min-h-[220px] space-y-1.5 overflow-y-auto pr-1">
            {rows.map((trade) => (
              <div key={trade.id} className="rounded-lg border border-border/50 bg-background/40 p-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    {trade.direction === 'buy_no' ? (
                      <ArrowDownRight className="w-3.5 h-3.5 text-rose-300" />
                    ) : (
                      <ArrowUpRight className="w-3.5 h-3.5 text-emerald-300" />
                    )}
                    <p className="text-xs font-semibold truncate">{trade.source || trade.strategy}</p>
                    <Badge className={cn('text-[10px] uppercase', statusColor[trade.status] || 'bg-muted text-muted-foreground')}>
                      {trade.status}
                    </Badge>
                  </div>

                  <div className="text-right">
                    <p className="text-xs font-mono font-semibold">${trade.total_cost.toFixed(2)}</p>
                    <p className="text-[10px] text-muted-foreground">{trade.mode}</p>
                  </div>
                </div>

                <div className="mt-1.5 text-[11px] text-muted-foreground flex items-center justify-between gap-2">
                  <span className="truncate">{trade.market_question || trade.market_id || trade.opportunity_id}</span>
                  <span className="flex items-center gap-1 whitespace-nowrap">
                    <Clock3 className="w-3 h-3" />
                    {trade.executed_at ? new Date(trade.executed_at).toLocaleTimeString() : 'n/a'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
