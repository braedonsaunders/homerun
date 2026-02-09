import { useMemo } from 'react'
import { cn } from '../lib/utils'
import { Opportunity } from '../services/api'
import {
  TrendingUp,
  TrendingDown,
  Zap,
  Radio,
  DollarSign,
  Shield,
  Brain,
  Clock,
  Activity,
  Target,
} from 'lucide-react'

interface TickerTapeProps {
  opportunities: Opportunity[]
  isConnected: boolean
  totalOpportunities: number
  lastScan?: string | null
  activeStrategies?: number
  className?: string
}

interface TickerItem {
  id: string
  label: string
  value: string
  secondaryValue?: string
  change?: number
  icon?: 'up' | 'down' | 'zap' | 'dollar' | 'shield' | 'brain' | 'clock' | 'activity' | 'target'
  badge?: string
  badgeColor?: string
}

const ICON_MAP = {
  up: TrendingUp,
  down: TrendingDown,
  zap: Zap,
  dollar: DollarSign,
  shield: Shield,
  brain: Brain,
  clock: Clock,
  activity: Activity,
  target: Target,
}

const ICON_COLOR_MAP: Record<string, string> = {
  up: 'text-green-400',
  down: 'text-red-400',
  zap: 'text-blue-400',
  dollar: 'text-yellow-400',
  shield: 'text-cyan-400',
  brain: 'text-purple-400',
  clock: 'text-orange-400',
  activity: 'text-emerald-400',
  target: 'text-blue-400',
}

function riskLabel(score: number): { text: string; color: string } {
  if (score <= 0.3) return { text: 'LOW', color: 'text-green-400 bg-green-400/10' }
  if (score <= 0.6) return { text: 'MED', color: 'text-yellow-400 bg-yellow-400/10' }
  return { text: 'HIGH', color: 'text-red-400 bg-red-400/10' }
}

function truncate(str: string, len: number): string {
  if (str.length <= len) return str
  return str.slice(0, len - 1) + '\u2026'
}

function formatTimeSince(iso: string): string {
  const secs = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (secs < 5) return 'just now'
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  return `${Math.floor(secs / 3600)}h ago`
}

export default function LiveTickerTape({
  opportunities,
  isConnected,
  totalOpportunities,
  lastScan,
  activeStrategies,
  className,
}: TickerTapeProps) {
  const tickerItems = useMemo<TickerItem[]>(() => {
    const items: TickerItem[] = []

    // --- Aggregate stats ---
    if (opportunities.length > 0) {
      const avgRoi = opportunities.reduce((s, o) => s + o.roi_percent, 0) / opportunities.length
      items.push({
        id: 'avg-roi',
        label: 'AVG ROI',
        value: `${avgRoi.toFixed(2)}%`,
        change: avgRoi,
        icon: avgRoi >= 0 ? 'up' : 'down',
      })

      const bestOpp = opportunities.reduce((best, o) => o.roi_percent > best.roi_percent ? o : best, opportunities[0])
      items.push({
        id: 'best-roi',
        label: 'BEST',
        value: `${bestOpp.roi_percent >= 0 ? '+' : ''}${bestOpp.roi_percent.toFixed(2)}%`,
        secondaryValue: `$${bestOpp.net_profit.toFixed(2)}`,
        change: bestOpp.roi_percent,
        icon: 'target',
      })

      const totalProfit = opportunities.reduce((s, o) => s + o.net_profit, 0)
      items.push({
        id: 'total-profit',
        label: 'TOTAL PROFIT',
        value: `$${totalProfit.toFixed(2)}`,
        change: totalProfit,
        icon: totalProfit >= 0 ? 'up' : 'down',
      })

      const totalCost = opportunities.reduce((s, o) => s + o.total_cost, 0)
      items.push({
        id: 'capital-needed',
        label: 'CAPITAL',
        value: `$${totalCost >= 1000 ? (totalCost / 1000).toFixed(1) + 'k' : totalCost.toFixed(0)}`,
        icon: 'dollar',
      })

      // High-confidence count (AI score >= 70)
      const highConf = opportunities.filter(o => o.ai_analysis && o.ai_analysis.overall_score >= 70).length
      if (highConf > 0) {
        items.push({
          id: 'high-conf',
          label: 'AI PICKS',
          value: highConf.toString(),
          icon: 'brain',
          badge: '\u226570',
          badgeColor: 'text-purple-400 bg-purple-400/10',
        })
      }
    }

    items.push({
      id: 'markets',
      label: 'MARKETS',
      value: totalOpportunities.toString(),
      icon: 'zap',
    })

    if (activeStrategies !== undefined) {
      items.push({
        id: 'strategies',
        label: 'STRATEGIES',
        value: activeStrategies.toString(),
        icon: 'activity',
      })
    }

    if (lastScan && !isNaN(new Date(lastScan).getTime())) {
      items.push({
        id: 'last-scan',
        label: 'SCAN',
        value: formatTimeSince(lastScan),
        icon: 'clock',
      })
    }

    // --- Top opportunities with context ---
    const topOpps = [...opportunities]
      .sort((a, b) => b.roi_percent - a.roi_percent)
      .slice(0, 10)

    topOpps.forEach((opp) => {
      const roi = opp.roi_percent
      const risk = riskLabel(opp.risk_score)
      // Prefer event_title, fall back to title, then strategy name
      const displayName = opp.event_title
        ? truncate(opp.event_title, 32)
        : opp.title
          ? truncate(opp.title, 32)
          : opp.strategy.replace(/_/g, ' ').toUpperCase()

      const aiScore = opp.ai_analysis?.overall_score
      const aiStr = aiScore !== undefined && aiScore !== null ? `AI:${Math.round(aiScore)}` : undefined

      items.push({
        id: opp.id,
        label: displayName,
        value: `${roi >= 0 ? '+' : ''}${roi.toFixed(2)}%`,
        secondaryValue: `$${opp.net_profit.toFixed(2)}`,
        change: roi,
        icon: roi >= 0 ? 'up' : 'down',
        badge: aiStr || risk.text,
        badgeColor: aiStr
          ? (aiScore! >= 70 ? 'text-purple-400 bg-purple-400/10' : aiScore! >= 40 ? 'text-blue-400 bg-blue-400/10' : 'text-muted-foreground bg-muted/50')
          : risk.color,
      })
    })

    return items
  }, [opportunities, totalOpportunities, lastScan, activeStrategies])

  if (tickerItems.length === 0) return null

  // Duplicate items to create seamless loop
  const allItems = [...tickerItems, ...tickerItems]

  return (
    <div className={cn(
      "h-8 border-b border-border/30 bg-card/40 backdrop-blur-sm overflow-hidden relative shrink-0",
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
        style={{ '--ticker-duration': `${Math.max(30, tickerItems.length * 4)}s` } as React.CSSProperties}
      >
        {allItems.map((item, i) => {
          const IconComponent = item.icon ? ICON_MAP[item.icon] : null
          const iconColor = item.icon ? ICON_COLOR_MAP[item.icon] : ''

          return (
            <div key={`${item.id}-${i}`} className="inline-flex items-center gap-1.5 mx-3 text-[11px]">
              {IconComponent && <IconComponent className={cn("w-3 h-3", iconColor)} />}
              <span className="text-muted-foreground font-medium">{item.label}</span>
              <span className={cn(
                "font-data font-semibold",
                item.change !== undefined
                  ? item.change >= 0 ? "text-green-400" : "text-red-400"
                  : "text-foreground"
              )}>
                {item.value}
              </span>
              {item.secondaryValue && (
                <span className="text-muted-foreground/70 font-data text-[10px]">
                  {item.secondaryValue}
                </span>
              )}
              {item.badge && (
                <span className={cn(
                  "px-1 py-0.5 rounded text-[9px] font-semibold leading-none",
                  item.badgeColor || "text-muted-foreground bg-muted/50"
                )}>
                  {item.badge}
                </span>
              )}
              <span className="text-border/60 mx-1.5">|</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
