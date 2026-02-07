import { useState, useEffect } from 'react'
import clsx from 'clsx'

interface DataFreshnessIndicatorProps {
  lastUpdated: string | null | undefined
  className?: string
  showLabel?: boolean
  staleThresholdMs?: number
  warningThresholdMs?: number
}

type FreshnessLevel = 'fresh' | 'warning' | 'stale' | 'unknown'

const FRESHNESS_CONFIG: Record<FreshnessLevel, { color: string; bg: string; pulse: string; label: string }> = {
  fresh: {
    color: 'bg-green-500',
    bg: 'bg-green-500/10',
    pulse: 'animate-pulse',
    label: 'Fresh',
  },
  warning: {
    color: 'bg-yellow-500',
    bg: 'bg-yellow-500/10',
    pulse: '',
    label: 'Aging',
  },
  stale: {
    color: 'bg-red-500',
    bg: 'bg-red-500/10',
    pulse: '',
    label: 'Stale',
  },
  unknown: {
    color: 'bg-gray-500',
    bg: 'bg-gray-500/10',
    pulse: '',
    label: 'No Data',
  },
}

function getTimeSince(dateStr: string): string {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000)
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

export default function DataFreshnessIndicator({
  lastUpdated,
  className = '',
  showLabel = true,
  staleThresholdMs = 120000,  // 2 minutes
  warningThresholdMs = 60000, // 1 minute
}: DataFreshnessIndicatorProps) {
  const [, setTick] = useState(0)

  // Re-render every 5 seconds to update time display
  useEffect(() => {
    const interval = setInterval(() => setTick(t => t + 1), 5000)
    return () => clearInterval(interval)
  }, [])

  const getFreshnessLevel = (): FreshnessLevel => {
    if (!lastUpdated) return 'unknown'
    const age = Date.now() - new Date(lastUpdated).getTime()
    if (age < warningThresholdMs) return 'fresh'
    if (age < staleThresholdMs) return 'warning'
    return 'stale'
  }

  const level = getFreshnessLevel()
  const config = FRESHNESS_CONFIG[level]
  const timeSince = lastUpdated ? getTimeSince(lastUpdated) : null

  return (
    <div
      className={clsx(
        'flex items-center gap-2 px-2.5 py-1 rounded-full text-xs',
        config.bg,
        className
      )}
      title={lastUpdated ? `Last updated: ${new Date(lastUpdated).toLocaleString()}` : 'No data received yet'}
    >
      <span className="relative flex h-2.5 w-2.5">
        {level === 'fresh' && (
          <span className={clsx(
            'absolute inline-flex h-full w-full rounded-full opacity-75',
            config.color,
            'animate-ping'
          )} />
        )}
        <span className={clsx(
          'relative inline-flex rounded-full h-2.5 w-2.5',
          config.color
        )} />
      </span>
      {showLabel && (
        <span className={clsx(
          'font-medium',
          level === 'fresh' && 'text-green-400',
          level === 'warning' && 'text-yellow-400',
          level === 'stale' && 'text-red-400',
          level === 'unknown' && 'text-gray-400',
        )}>
          {timeSince || config.label}
        </span>
      )}
    </div>
  )
}
