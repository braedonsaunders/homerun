import { useCallback, useEffect, useRef } from 'react'
import { QueryClient, QueryKey } from '@tanstack/react-query'

type WSMessage = {
  type?: string
  data?: Record<string, any>
} | null | undefined

type RealtimeContext = {
  activeTab?: string
  opportunitiesView?: string
  dataView?: string
}

const INVALIDATION_DEBOUNCE_MS = 120

export function useRealtimeInvalidation(
  lastMessage: WSMessage,
  queryClient: QueryClient,
  setScannerActivity: (activity: string) => void,
  context: RealtimeContext = {},
) {
  const pendingInvalidationsRef = useRef<Map<string, QueryKey>>(new Map())
  const flushTimerRef = useRef<number | null>(null)

  const flushInvalidations = useCallback(() => {
    if (flushTimerRef.current != null) {
      window.clearTimeout(flushTimerRef.current)
      flushTimerRef.current = null
    }
    if (pendingInvalidationsRef.current.size < 1) return

    const queryKeys = Array.from(pendingInvalidationsRef.current.values())
    pendingInvalidationsRef.current.clear()
    for (const queryKey of queryKeys) {
      queryClient.invalidateQueries({ queryKey })
    }
  }, [queryClient])

  const queueInvalidations = useCallback(
    (queryKeys: QueryKey[], options?: { immediate?: boolean }) => {
      const immediate = Boolean(options?.immediate)
      for (const queryKey of queryKeys) {
        pendingInvalidationsRef.current.set(JSON.stringify(queryKey), queryKey)
      }

      if (immediate) {
        flushInvalidations()
        return
      }

      if (flushTimerRef.current != null) {
        return
      }
      flushTimerRef.current = window.setTimeout(() => {
        flushInvalidations()
      }, INVALIDATION_DEBOUNCE_MS)
    },
    [flushInvalidations],
  )

  useEffect(() => {
    return () => {
      if (flushTimerRef.current != null) {
        window.clearTimeout(flushTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    const messageType = lastMessage?.type
    if (!messageType) return

    const mergeWorkerStatuses = (incomingWorkers: any[]) => {
      queryClient.setQueryData(['workers-status'], (old: any) => {
        const previousWorkers = Array.isArray(old?.workers) ? old.workers : []
        const workerMap = new Map<string, any>()

        for (const worker of previousWorkers) {
          if (!worker || typeof worker !== 'object') continue
          const workerName = String(worker.worker_name || '').trim()
          if (!workerName) continue
          workerMap.set(workerName, worker)
        }

        for (const worker of incomingWorkers) {
          if (!worker || typeof worker !== 'object') continue
          const workerName = String(worker.worker_name || '').trim()
          if (!workerName) continue
          const previous = workerMap.get(workerName) || {}
          workerMap.set(workerName, {
            ...previous,
            ...worker,
            stats: worker.stats ?? previous.stats ?? {},
            control: worker.control ?? previous.control,
          })
        }

        return {
          ...(old && typeof old === 'object' ? old : {}),
          workers: Array.from(workerMap.values()),
        }
      })
    }

    const activeTab = String(context.activeTab || '')
    const opportunitiesView = String(context.opportunitiesView || '')
    const dataView = String(context.dataView || '')
    const viewingOpportunities = activeTab === 'opportunities'
    const viewingData = activeTab === 'data'
    const viewingArbitrage = viewingOpportunities && opportunitiesView === 'scanner'
    const viewingNews =
      (viewingOpportunities && opportunitiesView === 'news')
      || (viewingData && dataView === 'feed')
    const viewingWeather = viewingOpportunities && opportunitiesView === 'weather'
    const viewingScannerFamily = viewingOpportunities
      && opportunitiesView !== 'search'
      && opportunitiesView !== 'crypto'
      && opportunitiesView !== 'news'
      && opportunitiesView !== 'weather'
      && opportunitiesView !== 'traders'
    const viewingWorld = viewingData && dataView === 'map'
    const viewingTrading = activeTab === 'trading'

    if (messageType === 'scanner_status' && lastMessage.data) {
      queryClient.setQueryData(['scanner-status'], lastMessage.data)
    }
    if (messageType === 'scanner_activity') {
      setScannerActivity(lastMessage.data?.activity || 'Idle')
    }

    if (messageType === 'opportunities_update' || messageType === 'init') {
      queueInvalidations([
        ['opportunities'],
        ['opportunity-strategy-counts'],
        ['opportunity-category-counts'],
        ['opportunity-subfilters'],
        ['scanner-status'],
      ], { immediate: viewingScannerFamily })
    }
    if (messageType === 'init' && Array.isArray(lastMessage.data?.workers_status)) {
      mergeWorkerStatuses(lastMessage.data.workers_status)
    }
    if (messageType === 'opportunity_events') {
      queueInvalidations(
        [
          ...(viewingArbitrage ? ([['opportunities']] as QueryKey[]) : []),
          ['opportunity-strategy-counts'],
          ['opportunity-category-counts'],
          ['opportunity-subfilters'],
        ],
        { immediate: viewingScannerFamily },
      )
    }
    if (messageType === 'wallet_trade') {
      queueInvalidations([
        ['copy-trades'],
        ['copy-trading-status'],
      ])
    }
    if (
      messageType === 'copy_trade_detected'
      || messageType === 'copy_trade_executed'
    ) {
      queueInvalidations([
        ['copy-trades'],
        ['copy-configs'],
        ['copy-trading-status'],
      ])
    }
    if (messageType === 'tracked_trader_signal') {
      queueInvalidations([
        ['opportunities', 'traders'],
        ['discovery-confluence'],
        ['discovery-active-signal-count'],
      ])
    }
    if (messageType === 'tracked_trader_pool_update') {
      queueInvalidations([
        ['discovery-pool-stats'],
        ['discovery-leaderboard'],
      ])
    }
    if (messageType === 'news_workflow_status' && lastMessage.data) {
      // Direct cache update for news workflow status
      queryClient.setQueryData(['news-workflow-status'], lastMessage.data)
      queueInvalidations([
        ['news-workflow-findings-count'],
        ...(viewingNews
          ? ([
              ['news-articles'],
              ['news-matches'],
              ['news-edges'],
              ['news-feed-status'],
              ['news-workflow-findings'],
              ['news-workflow-intents'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'news_workflow_update' && lastMessage.data) {
      // Direct cache update for news workflow status from the status sub-field
      if (lastMessage.data.status) {
        queryClient.setQueryData(['news-workflow-status'], lastMessage.data.status)
      }
      queueInvalidations([
        ['news-workflow-findings-count'],
        ...(viewingNews
          ? ([
              ['news-articles'],
              ['news-matches'],
              ['news-edges'],
              ['news-feed-status'],
              ['news-workflow-findings'],
              ['news-workflow-intents'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'news_update') {
      queueInvalidations([
        ['news-workflow-status'],
        ['news-workflow-findings-count'],
        ...(viewingNews
          ? ([
              ['news-articles'],
              ['news-matches'],
              ['news-edges'],
              ['news-feed-status'],
              ['news-workflow-findings'],
              ['news-workflow-intents'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'weather_update' && lastMessage.data) {
      if (lastMessage.data.status) {
        queryClient.setQueryData(['weather-workflow-status'], lastMessage.data.status)
      }
      queueInvalidations([
        ['weather-workflow-opportunities', 'count'],
        ...(viewingWeather
          ? ([
              ['weather-workflow-opportunities'],
              ['weather-workflow-opportunity-ids-analyze-all'],
              ['weather-workflow-opportunity-dates'],
              ['weather-workflow-intents'],
              ['weather-workflow-performance'],
            ] as QueryKey[])
          : []),
      ], { immediate: viewingWeather })
    }
    if (messageType === 'weather_status' && lastMessage.data) {
      queryClient.setQueryData(['weather-workflow-status'], lastMessage.data)
      queueInvalidations([
        ['weather-workflow-opportunities', 'count'],
        ...(viewingWeather
          ? ([
              ['weather-workflow-opportunities'],
              ['weather-workflow-opportunity-ids-analyze-all'],
              ['weather-workflow-opportunity-dates'],
              ['weather-workflow-intents'],
              ['weather-workflow-performance'],
            ] as QueryKey[])
          : []),
      ], { immediate: viewingWeather })
    }
    if (messageType === 'events_status' && lastMessage.data) {
      // Direct cache update for events status
      queryClient.setQueryData(['events-status'], lastMessage.data)
      queueInvalidations([
        ['events-summary'],
        ...(viewingWorld
          ? ([
              ['world-signals'],
              ['world-instability'],
              ['world-tensions'],
              ['world-convergences'],
              ['world-anomalies'],
              ['world-regions'],
              ['events-sources'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'events_update' && lastMessage.data) {
      // Direct cache update for events status and signals
      if (lastMessage.data.status) {
        queryClient.setQueryData(['events-status'], lastMessage.data.status)
      }
      if (lastMessage.data.signals) {
        queryClient.setQueryData(['world-signals'], (old: any) => ({
          ...old,
          ...lastMessage.data,
        }))
      }
      // Still invalidate derived queries
      queueInvalidations([
        ['events-summary'],
        ...(viewingWorld
          ? ([
              ['world-instability'],
              ['world-tensions'],
              ['world-convergences'],
              ['world-anomalies'],
              ['world-regions'],
              ['events-sources'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'worker_status_update' && lastMessage.data) {
      const workers = Array.isArray(lastMessage.data.workers) ? lastMessage.data.workers : []
      mergeWorkerStatuses(workers)
      // Still invalidate related status queries
      queueInvalidations([
        ['scanner-status'],
        ['news-workflow-status'],
        ['weather-workflow-status'],
      ])
    }
    if (messageType === 'signals_update' && lastMessage.data) {
      // Direct cache update for signals
      if (lastMessage.data.sources) {
        queryClient.setQueryData(['signals'], (old: any) => ({
          ...old,
          ...lastMessage.data,
        }))
      }
      // Still invalidate derived queries
      queueInvalidations([
        ['signals-stats'],
      ])
    }
    if (messageType === 'crypto_markets_update' && lastMessage.data) {
      // Direct cache update for crypto markets
      if (lastMessage.data.markets) {
        queryClient.setQueryData(['crypto-markets'], (old: any) => ({
          ...old,
          ...lastMessage.data,
        }))
      }
    }
    if (messageType === 'trader_orchestrator_status' && lastMessage.data) {
      // Direct cache update for trader orchestrator status
      queryClient.setQueryData(['trader-orchestrator-status'], lastMessage.data)
      queueInvalidations([
        ['trader-orchestrator-overview'],
        ...(viewingTrading ? ([['traders-list']] as QueryKey[]) : []),
      ])
    }
    if (messageType === 'trader_decision') {
      queueInvalidations([
        ['trader-orchestrator-overview'],
        ...(viewingTrading
          ? ([
              ['traders-decisions'],
              ['trader-decisions-all'],
              ['trader-events-all'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'trader_order') {
      queueInvalidations([
        ['trader-orchestrator-overview'],
        ...(viewingTrading
          ? ([
              ['traders-orders'],
              ['traders-decisions'],
              ['trader-orders-all'],
              ['trader-decisions-all'],
              ['trader-events-all'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'trader_event') {
      queueInvalidations(
        viewingTrading
          ? [
              ['traders-events'],
              ['trader-events-all'],
            ]
          : [],
      )
    }
  }, [context.activeTab, context.opportunitiesView, lastMessage, queryClient, queueInvalidations, setScannerActivity])
}
