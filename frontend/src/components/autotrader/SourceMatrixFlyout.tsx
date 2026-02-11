import { SlidersHorizontal, X } from 'lucide-react'

import type {
  AutoTraderExposure,
  AutoTraderMetrics,
  AutoTraderSourcePolicy,
  CopyTradingStatus,
} from '../../services/api'
import { Button } from '../ui/button'
import SourceMatrix from './SourceMatrix'

export default function SourceMatrixFlyout({
  isOpen,
  onClose,
  policies,
  signalStats,
  metrics,
  exposure,
  copyStatus,
  updatingSources,
  onToggleSource,
}: {
  isOpen: boolean
  onClose: () => void
  policies?: { global: AutoTraderSourcePolicy; sources: Record<string, AutoTraderSourcePolicy> }
  signalStats?: { totals: Record<string, number>; sources: Array<Record<string, any>> }
  metrics?: AutoTraderMetrics
  exposure?: AutoTraderExposure
  copyStatus?: CopyTradingStatus
  updatingSources?: Set<string>
  onToggleSource: (source: string, enabled: boolean) => void
}) {
  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-3xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-sky-300" />
            <h3 className="text-sm font-semibold">Source Matrix</h3>
          </div>
          <Button
            variant="ghost"
            onClick={onClose}
            className="text-xs h-auto px-2.5 py-1 hover:bg-card"
          >
            <X className="w-3.5 h-3.5 mr-1" />
            Close
          </Button>
        </div>

        <div className="p-3">
          <SourceMatrix
            policies={policies}
            signalStats={signalStats}
            metrics={metrics}
            exposure={exposure}
            copyStatus={copyStatus}
            updatingSources={updatingSources}
            onToggleSource={onToggleSource}
          />
        </div>
      </div>
    </>
  )
}
