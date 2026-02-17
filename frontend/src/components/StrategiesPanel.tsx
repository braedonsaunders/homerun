import { useEffect, useState } from 'react'
import { Bot, Puzzle } from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import OpportunityStrategiesManager from './OpportunityStrategiesManager'
import TraderStrategiesManager from './TraderStrategiesManager'

type StrategiesSubTab = 'opportunity' | 'autotrader'

export default function StrategiesPanel() {
  const [subTab, setSubTab] = useState<StrategiesSubTab>('opportunity')

  useEffect(() => {
    const handler = (event: Event) => {
      const next = (event as CustomEvent<StrategiesSubTab>).detail
      if (next === 'opportunity' || next === 'autotrader') {
        setSubTab(next)
      }
    }
    window.addEventListener('navigate-strategies-subtab', handler as EventListener)
    return () => window.removeEventListener('navigate-strategies-subtab', handler as EventListener)
  }, [])

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="shrink-0 pb-3 flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSubTab('opportunity')}
          className={cn(
            'gap-1.5 text-xs h-8',
            subTab === 'opportunity'
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/30 hover:bg-amber-500/30 hover:text-amber-300'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Puzzle className="w-3.5 h-3.5" />
          Opportunity Strategies
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSubTab('autotrader')}
          className={cn(
            'gap-1.5 text-xs h-8',
            subTab === 'autotrader'
              ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Bot className="w-3.5 h-3.5" />
          AutoTrader Strategies
        </Button>
      </div>

      <div className={cn('flex-1 min-h-0', subTab === 'opportunity' ? '' : 'hidden')}>
        <OpportunityStrategiesManager />
      </div>

      <div className={cn('flex-1 min-h-0', subTab === 'autotrader' ? '' : 'hidden')}>
        <TraderStrategiesManager />
      </div>
    </div>
  )
}
