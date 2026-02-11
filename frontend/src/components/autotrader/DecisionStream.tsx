import { Brain, Clock3 } from 'lucide-react'

import type { AutoTraderDecision } from '../../services/api'
import { cn } from '../../lib/utils'
import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

interface DecisionStreamProps {
  decisions: AutoTraderDecision[]
  maxItems?: number
}

const decisionColor: Record<string, string> = {
  executed: 'bg-emerald-500/20 text-emerald-300',
  submitted: 'bg-sky-500/20 text-sky-300',
  failed: 'bg-rose-500/20 text-rose-300',
  skipped: 'bg-amber-500/20 text-amber-300',
}

export default function DecisionStream({ decisions, maxItems = 10 }: DecisionStreamProps) {
  const rows = decisions.slice(0, maxItems)

  return (
    <Card className="border-border/50 bg-card/40 h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Decision Stream</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        {rows.length === 0 ? (
          <div className="h-full min-h-[220px] flex items-center justify-center text-sm text-muted-foreground">
            No decisions yet.
          </div>
        ) : (
          <div className="h-full min-h-[220px] space-y-1.5 overflow-y-auto pr-1">
            {rows.map((decision) => (
              <div key={decision.id} className="rounded-lg border border-border/50 bg-background/40 p-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0 flex items-center gap-2">
                    <Brain className="w-3.5 h-3.5 text-muted-foreground" />
                    <p className="text-xs font-semibold truncate">{decision.source}</p>
                    <Badge className={cn('text-[10px] uppercase', decisionColor[decision.decision] || 'bg-muted text-muted-foreground')}>
                      {decision.decision}
                    </Badge>
                  </div>
                  <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                    <Clock3 className="w-3 h-3" />
                    {decision.created_at ? new Date(decision.created_at).toLocaleTimeString() : 'n/a'}
                  </div>
                </div>

                <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px]">
                  <p className="text-muted-foreground truncate">{decision.reason || 'No reason provided'}</p>
                  <span className="font-mono">{decision.score != null ? decision.score.toFixed(4) : '-'}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
