import { useQuery } from '@tanstack/react-query'
import { Database, Globe2, Newspaper } from 'lucide-react'

import { cn } from '../lib/utils'
import { getNewsFeedStatus, getUnifiedDataSources } from '../services/api'
import { getWorldIntelligenceSummary } from '../services/worldIntelligenceApi'
import NewsIntelligencePanel from './NewsIntelligencePanel'
import WorldIntelligencePanel from './WorldIntelligencePanel'
import DataSourcesManager from './DataSourcesManager'
import { Button } from './ui/button'

export type DataView = 'map' | 'feed' | 'sources'

interface DataPanelProps {
  isConnected: boolean
  view: DataView
  onViewChange: (view: DataView) => void
}

export default function DataPanel({ isConnected, view, onViewChange }: DataPanelProps) {
  const { data: worldSummary } = useQuery({
    queryKey: ['world-intelligence-summary'],
    queryFn: getWorldIntelligenceSummary,
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

  const mapCount = Number(worldSummary?.signal_summary?.total || 0)
  const feedCount = Number(feedStatus?.article_count || 0)
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
          {mapCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-blue-500/15 text-blue-400 text-[10px] font-data">
              {mapCount}
            </span>
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewChange('feed')}
          className={cn(
            'gap-1.5 text-xs h-8',
            view === 'feed'
              ? 'bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30 hover:text-orange-400'
              : 'bg-card text-muted-foreground hover:text-foreground border-border'
          )}
        >
          <Newspaper className="w-3.5 h-3.5" />
          Feed
          {feedCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-orange-500/15 text-orange-400 text-[10px] font-data">
              {feedCount}
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
        <div className="flex-1 min-h-0 overflow-hidden px-6 py-4">
          <WorldIntelligencePanel isConnected={isConnected} />
        </div>
      )}

      {view === 'feed' && (
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <NewsIntelligencePanel mode="feed" />
        </div>
      )}

      {view === 'sources' && (
        <div className="flex-1 min-h-0 overflow-hidden px-6 py-4">
          <DataSourcesManager />
        </div>
      )}
    </div>
  )
}
