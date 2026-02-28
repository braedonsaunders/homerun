import { lazy, Suspense } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Database, Globe2, Newspaper, Radio } from 'lucide-react'

import { cn } from '../lib/utils'
import { getNewsFeedStatus, getUnifiedDataSources } from '../services/api'
import { getEventsSummary } from '../services/eventsApi'
import NewsIntelligencePanel from './NewsIntelligencePanel'
import EventsPanel from './EventsPanel'
import DataSourcesManager from './DataSourcesManager'
import ErrorBoundary from './ErrorBoundary'
import { Button } from './ui/button'

const WorldMap = lazy(() => import('./WorldMap'))

export type DataView = 'map' | 'events' | 'stories' | 'sources'

interface DataPanelProps {
  isConnected: boolean
  view: DataView
  onViewChange: (view: DataView) => void
  onOpenCopilot?: (options?: {
    contextType?: string
    contextId?: string
    label?: string
    prompt?: string
    autoSend?: boolean
  }) => void
}

export default function DataPanel({ isConnected, view, onViewChange, onOpenCopilot }: DataPanelProps) {
  const { data: worldSummary } = useQuery({
    queryKey: ['events-summary'],
    queryFn: getEventsSummary,
    refetchInterval: isConnected ? false : 120000,
  })

  const { data: feedStatus } = useQuery({
    queryKey: ['news-feed-status'],
    queryFn: getNewsFeedStatus,
    refetchInterval: isConnected ? false : 120000,
  })

  const { data: dataSources } = useQuery({
    queryKey: ['unified-data-sources'],
    queryFn: () => getUnifiedDataSources(),
    refetchInterval: isConnected ? false : 120000,
  })

  const eventCount = Number(worldSummary?.signal_summary?.total || 0)
  const storyCount = Number(feedStatus?.article_count || 0)
  const sourceCount = Array.isArray(dataSources) ? dataSources.length : 0

  return (
    <div className="flex-1 overflow-hidden flex flex-col section-enter">
      <div className="shrink-0 px-6 pt-4 pb-0 flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewChange('map')}
          className={cn(
            'gap-1.5 text-xs h-8',
            view === 'map'
              ? 'bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30 hover:text-blue-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Globe2 className="w-3.5 h-3.5" />
          Map
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewChange('events')}
          className={cn(
            'gap-1.5 text-xs h-8',
            view === 'events'
              ? 'bg-violet-500/20 text-violet-400 border-violet-500/30 hover:bg-violet-500/30 hover:text-violet-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Radio className="w-3.5 h-3.5" />
          Events
          {eventCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-violet-500/15 text-violet-400 text-[10px] font-data">
              {eventCount}
            </span>
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewChange('stories')}
          className={cn(
            'gap-1.5 text-xs h-8',
            view === 'stories'
              ? 'bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30 hover:text-orange-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Newspaper className="w-3.5 h-3.5" />
          Stories
          {storyCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-orange-500/15 text-orange-400 text-[10px] font-data">
              {storyCount}
            </span>
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewChange('sources')}
          className={cn(
            'gap-1.5 text-xs h-8',
            view === 'sources'
              ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Database className="w-3.5 h-3.5" />
          Sources
          {sourceCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 text-[10px] font-data">
              {sourceCount}
            </span>
          )}
        </Button>
      </div>

      {view === 'map' && (
        <div className="flex-1 min-h-0 overflow-hidden px-6 pt-4 pb-5">
          <div className="h-full min-h-0 rounded-lg border border-border/60 bg-card/20 relative overflow-hidden">
            <ErrorBoundary fallback={<div className="m-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-500">Map view crashed.</div>}>
              <Suspense fallback={<div className="h-full w-full" />}>
                <WorldMap isConnected={isConnected} />
              </Suspense>
            </ErrorBoundary>
          </div>
        </div>
      )}

      {view === 'events' && (
        <div className="flex-1 min-h-0 overflow-hidden px-6 py-4">
          <EventsPanel isConnected={isConnected} eventsOnly />
        </div>
      )}

      {view === 'stories' && (
        <div className="flex-1 min-h-0 overflow-hidden px-6 py-4">
          <NewsIntelligencePanel mode="feed" />
        </div>
      )}

      {view === 'sources' && (
        <div className="flex-1 min-h-0 overflow-hidden px-6 py-4">
          <ErrorBoundary fallback={<div className="m-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-500">Sources view crashed.</div>}>
            <DataSourcesManager onOpenCopilot={onOpenCopilot} />
          </ErrorBoundary>
        </div>
      )}
    </div>
  )
}
