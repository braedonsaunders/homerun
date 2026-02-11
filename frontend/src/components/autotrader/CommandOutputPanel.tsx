import { Terminal } from 'lucide-react'

import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

export interface CommandLogLine {
  id: string
  ts: string
  level: 'info' | 'warn' | 'error' | 'event'
  message: string
}

export default function CommandOutputPanel({
  logs,
  connected,
}: {
  logs: CommandLogLine[]
  connected: boolean
}) {
  const lines = logs.slice(-14)

  return (
    <Card className="border-border/50 bg-card/40 h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-300" />
            Live Action Console
          </span>
          <span className={`text-[10px] ${connected ? 'text-emerald-300' : 'text-amber-300'}`}>
            {connected ? 'WS ONLINE' : 'WS DEGRADED'}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <div className="h-full min-h-[220px] rounded-md border border-border/50 bg-black/80 p-2 font-mono text-[11px] leading-5 text-emerald-100 overflow-hidden">
          {lines.length === 0 ? (
            <p className="text-muted-foreground">[idle] awaiting actions...</p>
          ) : (
            lines.map((line) => (
              <div key={line.id} className="truncate">
                <span className="text-emerald-300">[{line.ts}]</span>{' '}
                <span className={levelColor(line.level)}>{line.level.toUpperCase()}</span>{' '}
                <span>{line.message}</span>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function levelColor(level: CommandLogLine['level']): string {
  switch (level) {
    case 'warn':
      return 'text-amber-300'
    case 'error':
      return 'text-rose-300'
    case 'event':
      return 'text-sky-300'
    default:
      return 'text-emerald-300'
  }
}
