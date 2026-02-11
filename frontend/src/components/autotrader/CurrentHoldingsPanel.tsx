import type { ComponentType } from 'react'
import { Briefcase, DollarSign, Wallet } from 'lucide-react'

import type { TradingPosition } from '../../services/api'
import { cn } from '../../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

interface CurrentHoldingsPanelProps {
  positions: TradingPosition[]
  balance?: {
    balance: number
    available: number
    reserved: number
    currency: string
    timestamp: string
  }
}

export default function CurrentHoldingsPanel({ positions, balance }: CurrentHoldingsPanelProps) {
  const totalUnrealized = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0)
  const totalNotional = positions.reduce((sum, p) => sum + (p.size || 0) * (p.current_price || 0), 0)
  const visible = positions.slice(0, 8)

  return (
    <Card className="border-border/50 bg-card/40 h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Current Holdings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 flex-1 min-h-0">
        <div className="grid grid-cols-3 gap-2 text-xs">
          <Stat label="Open" value={String(positions.length)} icon={Briefcase} />
          <Stat label="Notional" value={`$${totalNotional.toFixed(2)}`} icon={DollarSign} />
          <Stat
            label="Unrealized"
            value={`${totalUnrealized >= 0 ? '+' : ''}$${totalUnrealized.toFixed(2)}`}
            icon={Wallet}
            tone={totalUnrealized >= 0 ? 'good' : 'bad'}
          />
        </div>

        {balance && (
          <div className="rounded-md border border-border/50 bg-background/40 p-2 text-[11px]">
            <p className="text-muted-foreground">Available / Reserved</p>
            <p className="font-mono font-semibold">
              ${balance.available.toFixed(2)} / ${balance.reserved.toFixed(2)}
            </p>
          </div>
        )}

        <div className="space-y-1.5">
          {visible.length === 0 ? (
            <div className="h-24 flex items-center justify-center text-xs text-muted-foreground rounded-md border border-border/50 bg-background/30">
              No open positions
            </div>
          ) : (
            visible.map((position) => {
              const pnl = Number(position.unrealized_pnl || 0)
              return (
                <div key={`${position.market_id}:${position.token_id}:${position.outcome}`} className="rounded-md border border-border/50 bg-background/40 p-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-[11px] font-medium truncate">{position.market_question}</p>
                    <p className={cn('text-[11px] font-mono', pnl >= 0 ? 'text-emerald-300' : 'text-rose-300')}>
                      {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                    </p>
                  </div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground flex items-center justify-between">
                    <span>{position.outcome} • {position.size.toFixed(2)} @ ${position.average_cost.toFixed(3)}</span>
                    <span>${position.current_price.toFixed(3)}</span>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function Stat({
  label,
  value,
  icon: Icon,
  tone = 'neutral',
}: {
  label: string
  value: string
  icon: ComponentType<{ className?: string }>
  tone?: 'neutral' | 'good' | 'bad'
}) {
  return (
    <div className="rounded-md border border-border/50 bg-background/40 p-2">
      <div className="flex items-center gap-1 text-muted-foreground text-[10px] uppercase tracking-wider">
        <Icon className="w-3 h-3" />
        {label}
      </div>
      <p className={cn('text-xs font-mono font-semibold mt-1', tone === 'good' && 'text-emerald-300', tone === 'bad' && 'text-rose-300')}>
        {value}
      </p>
    </div>
  )
}
