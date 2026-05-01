import { useEffect, useState } from 'react'
import { Brain, Layers3, Shield, Sparkles } from 'lucide-react'
import { cn } from '../lib/utils'
import UnifiedStrategiesManager from './UnifiedStrategiesManager'
import MLModelsPanel from './MLModelsPanel'
import AutoresearchPanel from './AutoresearchPanel'
import ValidationPanel from './ValidationPanel'

type StrategiesPanelProps = {
  initialSourceFilter?: string | null
  onSourceFilterApplied?: () => void
  onOpenCopilot?: (options?: {
    contextType?: string
    contextId?: string
    label?: string
    prompt?: string
    autoSend?: boolean
  }) => void
}

type ViewMode = 'strategies' | 'ml-models' | 'research' | 'validation'

export default function StrategiesPanel({
  initialSourceFilter,
  onSourceFilterApplied,
  onOpenCopilot,
}: StrategiesPanelProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('strategies')

  // Honor a "Research" deep-link from the strategy editor's Research button.
  // The button stashes ``homerun:research:open`` plus the strategy slug;
  // we land on the Research subview here, the AutoresearchPanel reads the
  // strategy slug to preselect a matching bot.
  useEffect(() => {
    try {
      if (sessionStorage.getItem('homerun:research:open') === '1') {
        sessionStorage.removeItem('homerun:research:open')
        setViewMode('research')
      }
    } catch {
      // ignore — sessionStorage unavailable in some contexts
    }
    const handleOpenResearch = () => {
      try {
        sessionStorage.removeItem('homerun:research:open')
      } catch {
        // ignore
      }
      setViewMode('research')
    }
    window.addEventListener('homerun:research:open', handleOpenResearch)
    return () => window.removeEventListener('homerun:research:open', handleOpenResearch)
  }, [])

  return (
    <div className="h-full min-h-0 flex flex-col">
      {/* Top toggle */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border/50 shrink-0">
        <button
          onClick={() => setViewMode('strategies')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
            viewMode === 'strategies'
              ? 'bg-accent text-accent-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          )}
        >
          <Layers3 className="w-3.5 h-3.5" />
          Strategies
        </button>
        <button
          onClick={() => setViewMode('research')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
            viewMode === 'research'
              ? 'bg-accent text-accent-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          )}
        >
          <Sparkles className="w-3.5 h-3.5" />
          Research
        </button>
        <button
          onClick={() => setViewMode('ml-models')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
            viewMode === 'ml-models'
              ? 'bg-accent text-accent-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          )}
        >
          <Brain className="w-3.5 h-3.5" />
          Machine Learning
        </button>
        <button
          onClick={() => setViewMode('validation')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
            viewMode === 'validation'
              ? 'bg-accent text-accent-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          )}
        >
          <Shield className="w-3.5 h-3.5" />
          Validation
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0">
        {viewMode === 'strategies' && (
          <UnifiedStrategiesManager
            initialSourceFilter={initialSourceFilter}
            onSourceFilterApplied={onSourceFilterApplied}
            onOpenCopilot={onOpenCopilot}
          />
        )}
        {viewMode === 'research' && <AutoresearchPanel />}
        {viewMode === 'ml-models' && <MLModelsPanel />}
        {viewMode === 'validation' && <ValidationPanel />}
      </div>
    </div>
  )
}
