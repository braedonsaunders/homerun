import { useEffect, useState } from 'react'
import type { QueryClient, QueryKey } from '@tanstack/react-query'

const DEFAULT_REFRESH_MS = 5000

function invalidateQueryKeys(queryClient: QueryClient, keys: QueryKey[]) {
  for (const queryKey of keys) {
    queryClient.invalidateQueries({ queryKey })
  }
}

export function useDisplayedOpportunityRefresh({
  activeTab,
  opportunitiesView,
  queryClient,
  refreshMs = DEFAULT_REFRESH_MS,
}: {
  activeTab: string
  opportunitiesView: string
  queryClient: QueryClient
  refreshMs?: number
}) {
  const [isVisible, setIsVisible] = useState(
    () => (typeof document === 'undefined' ? true : document.visibilityState === 'visible'),
  )

  useEffect(() => {
    const onVisibilityChange = () => {
      setIsVisible(document.visibilityState === 'visible')
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [])

  useEffect(() => {
    if (activeTab !== 'opportunities' || !isVisible) return

    const refreshDisplayedView = () => {
      if (opportunitiesView === 'arbitrage') {
        invalidateQueryKeys(queryClient, [
          ['opportunities'],
          ['opportunity-counts'],
          ['opportunity-subfilters'],
          ['scanner-status'],
        ])
        return
      }

      if (opportunitiesView === 'recent_trades') {
        invalidateQueryKeys(queryClient, [
          ['tracked-trader-opportunities'],
          ['insider-opportunities'],
          ['workers-status'],
        ])
        return
      }

      if (opportunitiesView === 'weather') {
        invalidateQueryKeys(queryClient, [
          ['weather-workflow-opportunities'],
          ['weather-workflow-opportunity-date-source'],
          ['weather-workflow-opportunity-dates'],
          ['weather-workflow-status'],
        ])
        return
      }

      if (opportunitiesView === 'news') {
        invalidateQueryKeys(queryClient, [
          ['news-workflow-findings'],
          ['news-workflow-intents'],
          ['news-workflow-status'],
        ])
      }
    }

    refreshDisplayedView()
    const interval = window.setInterval(refreshDisplayedView, refreshMs)
    return () => window.clearInterval(interval)
  }, [activeTab, opportunitiesView, queryClient, refreshMs, isVisible])
}
