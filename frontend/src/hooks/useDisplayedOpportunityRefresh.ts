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
  isConnected = false,
}: {
  activeTab: string
  opportunitiesView: string
  queryClient: QueryClient
  refreshMs?: number
  isConnected?: boolean
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
    // Skip polling when WS is connected and pushing data
    if (isConnected) return
    if (activeTab !== 'opportunities' || !isVisible) return

    const refreshDisplayedView = () => {
      if (opportunitiesView === 'scanner') {
        invalidateQueryKeys(queryClient, [
          ['opportunities'],
          ['opportunity-strategy-counts'],
          ['opportunity-category-counts'],
          ['opportunity-subfilters'],
          ['scanner-status'],
        ])
        return
      }

      if (opportunitiesView === 'traders') {
        invalidateQueryKeys(queryClient, [
          ['opportunities', 'traders'],
          ['workers-status'],
        ])
        return
      }

      if (opportunitiesView === 'weather') {
        invalidateQueryKeys(queryClient, [
          ['weather-workflow-opportunities'],
          ['weather-workflow-opportunity-ids-analyze-all'],
          ['weather-workflow-opportunity-dates'],
          ['weather-workflow-status'],
        ])
        return
      }

      if (opportunitiesView === 'news') {
        invalidateQueryKeys(queryClient, [
          ['news-workflow-findings'],
          ['news-workflow-status'],
        ])
      }
    }

    refreshDisplayedView()
    const interval = window.setInterval(refreshDisplayedView, refreshMs)
    return () => window.clearInterval(interval)
  }, [activeTab, opportunitiesView, queryClient, refreshMs, isVisible, isConnected])
}
