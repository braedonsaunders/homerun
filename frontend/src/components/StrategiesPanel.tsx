import { useEffect, useState } from 'react'
import { Bot, Puzzle } from 'lucide-react'
import { cn } from '../lib/utils'
import OpportunityStrategiesManager from './OpportunityStrategiesManager'
import TraderStrategiesManager from './TraderStrategiesManager'

type StrategiesSubTab = 'opportunity' | 'autotrader'

interface NavigateStrategiesDetail {
  subtab?: StrategiesSubTab
  sourceFilter?: string
}

export default function StrategiesPanel() {
  const [subTab, setSubTab] = useState<StrategiesSubTab>('opportunity')
  const [initialSourceFilter, setInitialSourceFilter] = useState<string | undefined>(undefined)

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<StrategiesSubTab | NavigateStrategiesDetail>).detail
      // Support both legacy string format and new object format
      if (typeof detail === 'string') {
        if (detail === 'opportunity' || detail === 'autotrader') {
          setSubTab(detail)
        }
      } else if (detail && typeof detail === 'object') {
        const next = detail.subtab || 'opportunity'
        if (next === 'opportunity' || next === 'autotrader') {
          setSubTab(next)
        }
        if (detail.sourceFilter) {
          setInitialSourceFilter(detail.sourceFilter)
        }
      }
    }
    window.addEventListener('navigate-strategies-subtab', handler as EventListener)
    return () => window.removeEventListener('navigate-strategies-subtab', handler as EventListener)
  }, [])

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="shrink-0 pb-3 flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => setSubTab('opportunity')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150',
            subTab === 'opportunity'
              ? 'bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
          )}
        >
          <Puzzle className="w-3.5 h-3.5" />
          Opportunity Strategies
        </button>
        <button
          type="button"
          onClick={() => setSubTab('autotrader')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150',
            subTab === 'autotrader'
              ? 'bg-cyan-500/15 text-cyan-400 ring-1 ring-cyan-500/30'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
          )}
        >
          <Bot className="w-3.5 h-3.5" />
          AutoTrader Strategies
        </button>
      </div>

      <div className={cn('flex-1 min-h-0', subTab === 'opportunity' ? '' : 'hidden')}>
        <OpportunityStrategiesManager initialSourceFilter={initialSourceFilter} />
      </div>

      <div className={cn('flex-1 min-h-0', subTab === 'autotrader' ? '' : 'hidden')}>
        <TraderStrategiesManager />
      </div>
    </div>
  )
}
