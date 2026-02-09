import { useMemo } from 'react'
import { cn } from '../lib/utils'
import { Opportunity } from '../services/api'
import { TrendingUp, TrendingDown, Zap, Radio } from 'lucide-react'

interface TickerTapeProps {
  opportunities: Opportunity[]
  isConnected: boolean
  totalOpportunities: number
  className?: string
}

interface TickerItem {
  id: string
  label: string
  value: string
  change?: number
  icon?: 'up' | 'down' | 'zap'
}

export default function LiveTickerTape({
  opportunities,
  isConnected,
  totalOpportunities,
  className,
}: TickerTapeProps) {
  const tickerItems = useMemo<TickerItem[]>(() => {
    const items: TickerItem[] = []

    // Summary stats
    if (opportunities.length > 0) {
      const avgRoi = opportunities.reduce((s, o) => s + o.roi_percent, 0) / opportunities.length
      items.push({
        id: 'avg-roi',
        label: 'AVG ROI',
        value: `${avgRoi.toFixed(2)}%`,
        change: avgRoi,
        icon: avgRoi >= 0 ? 'up' : 'down',
      })

      const totalProfit = opportunities.reduce((s, o) => s + o.net_profit, 0)
      items.push({
        id: 'total-profit',
        label: 'TOTAL PROFIT',
        value: `$${totalProfit.toFixed(2)}`,
        change: totalProfit,
        icon: totalProfit >= 0 ? 'up' : 'down',
      })
    }

    items.push({
      id: 'markets',
      label: 'MARKETS',
      value: totalOpportunities.toString(),
      icon: 'zap',
    })

    // Top opportunities by ROI
    const topOpps = [...opportunities]
      .sort((a, b) => b.roi_percent - a.roi_percent)
      .slice(0, 8)

    topOpps.forEach((opp) => {
      const stratName = opp.strategy.replace(/_/g, ' ').toUpperCase()
      items.push({
        id: opp.id,
        label: `${stratName}`,
        value: `+${opp.roi_percent.toFixed(2)}%`,
        change: opp.roi_percent,
        icon: 'up',
      })
    })

    return items
  }, [opportunities, totalOpportunities])

  if (tickerItems.length === 0) return null

  // Duplicate items to create seamless loop
  const allItems = [...tickerItems, ...tickerItems]

  return (
    <div className={cn(
      "h-7 border-b border-border/30 bg-card/40 backdrop-blur-sm overflow-hidden relative shrink-0",
      className
    )}>
      {/* Left edge fade */}
      <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-background to-transparent z-10" />
      {/* Right edge fade */}
      <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-background to-transparent z-10" />

      {/* Live indicator */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 z-20 flex items-center gap-1">
        <Radio className={cn(
          "w-2.5 h-2.5",
          isConnected ? "text-green-400" : "text-red-400"
        )} />
      </div>

      {/* Scrolling content */}
      <div
        className="ticker-animate flex items-center h-full whitespace-nowrap pl-8"
        style={{ '--ticker-duration': `${Math.max(25, tickerItems.length * 4)}s` } as React.CSSProperties}
      >
        {allItems.map((item, i) => (
          <div key={`${item.id}-${i}`} className="inline-flex items-center gap-1.5 mx-4 text-[11px]">
            {item.icon === 'up' && <TrendingUp className="w-3 h-3 text-green-400" />}
            {item.icon === 'down' && <TrendingDown className="w-3 h-3 text-red-400" />}
            {item.icon === 'zap' && <Zap className="w-3 h-3 text-blue-400" />}
            <span className="text-muted-foreground font-medium">{item.label}</span>
            <span className={cn(
              "font-data font-semibold",
              item.change !== undefined
                ? item.change >= 0 ? "text-green-400" : "text-red-400"
                : "text-foreground"
            )}>
              {item.value}
            </span>
            <span className="text-border mx-2">|</span>
          </div>
        ))}
      </div>
    </div>
  )
}
