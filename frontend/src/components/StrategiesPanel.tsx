import { useState } from 'react'
import { Brain, Layers3 } from 'lucide-react'
import { cn } from '../lib/utils'
import UnifiedStrategiesManager from './UnifiedStrategiesManager'
import MLModelsPanel from './MLModelsPanel'

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

type ViewMode = 'strategies' | 'ml-models'

export default function StrategiesPanel({
  initialSourceFilter,
  onSourceFilterApplied,
  onOpenCopilot,
}: StrategiesPanelProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('strategies')

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
          onClick={() => setViewMode('ml-models')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
            viewMode === 'ml-models'
              ? 'bg-accent text-accent-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          )}
        >
          <Brain className="w-3.5 h-3.5" />
          ML Models
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
        {viewMode === 'ml-models' && <MLModelsPanel />}
      </div>
    </div>
  )
}
