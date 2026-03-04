import { useCallback, useEffect, useRef } from 'react'
import { QueryClient, QueryKey } from '@tanstack/react-query'

type WSMessage = {
  type?: string
  topic?: string
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
    const upsertTraderOrderCache = (incomingOrder: any) => {
      if (!incomingOrder || typeof incomingOrder !== 'object') return
      const incomingId = String(incomingOrder.id || '').trim()
      if (!incomingId) return
      const upsertRows = (old: any) => {
        if (!Array.isArray(old)) return old
        const previousRows = old
        let found = false
        const nextRows = previousRows.map((row: any) => {
          const rowId = String(row?.id || '').trim()
          if (rowId !== incomingId) return row
          found = true
          return {
            ...row,
            ...incomingOrder,
            payload: incomingOrder.payload ?? row?.payload ?? {},
          }
        })
        if (!found) {
          nextRows.unshift(incomingOrder)
        }
        return nextRows
      }

      queryClient.setQueryData(['trader-orders-all'], upsertRows)
      const traderId = String(incomingOrder.trader_id || '').trim()
      if (traderId) {
        queryClient.setQueryData(['trader-orders', traderId], upsertRows)
      }
    }

    const toTs = (value: unknown): number => {
      const parsed = Date.parse(String(value || ''))
      return Number.isFinite(parsed) ? parsed : 0
    }

    const upsertTraderDecisionCache = (incomingDecision: any) => {
      if (!incomingDecision || typeof incomingDecision !== 'object') return
      const incomingId = String(incomingDecision.id || '').trim()
      if (!incomingId) return
      const normalizeRows = (old: any) => {
        if (!Array.isArray(old)) return old
        let found = false
        const nextRows = old.map((row: any) => {
          const rowId = String(row?.id || '').trim()
          if (rowId !== incomingId) return row
          found = true
          return {
            ...row,
            ...incomingDecision,
            failed_checks: Array.isArray(incomingDecision.failed_checks)
              ? incomingDecision.failed_checks
              : (Array.isArray(row?.failed_checks) ? row.failed_checks : []),
            payload: incomingDecision.payload ?? row?.payload ?? {},
          }
        })
        if (!found) {
          nextRows.unshift({
            ...incomingDecision,
            failed_checks: Array.isArray(incomingDecision.failed_checks) ? incomingDecision.failed_checks : [],
            payload: incomingDecision.payload ?? {},
          })
        }
        return nextRows
          .sort((a: any, b: any) => toTs(b?.created_at) - toTs(a?.created_at))
          .slice(0, 1200)
      }
      queryClient.setQueriesData({ queryKey: ['trader-decisions-all'] }, normalizeRows)
      const traderId = String(incomingDecision.trader_id || '').trim()
      if (traderId) {
        queryClient.setQueryData(['trader-decisions', traderId], normalizeRows)
      }
    }

    const upsertTraderEventCache = (incomingEvent: any) => {
      if (!incomingEvent || typeof incomingEvent !== 'object') return
      const incomingId = String(incomingEvent.id || '').trim()
      if (!incomingId) return
      const normalizeRows = (old: any) => {
        if (!Array.isArray(old)) return old
        let found = false
        const nextRows = old.map((row: any) => {
          const rowId = String(row?.id || '').trim()
          if (rowId !== incomingId) return row
          found = true
          return {
            ...row,
            ...incomingEvent,
            payload: incomingEvent.payload ?? row?.payload ?? {},
          }
        })
        if (!found) {
          nextRows.unshift({
            ...incomingEvent,
            payload: incomingEvent.payload ?? {},
          })
        }
        return nextRows
          .sort((a: any, b: any) => toTs(b?.created_at) - toTs(a?.created_at))
          .slice(0, 800)
      }
      queryClient.setQueriesData({ queryKey: ['trader-events-all'] }, normalizeRows)
      const traderId = String(incomingEvent.trader_id || '').trim()
      if (traderId) {
        queryClient.setQueryData(['trader-events', traderId], normalizeRows)
      }
    }

    const mergeOrchestratorSnapshotIntoOverview = (snapshot: Record<string, any>) => {
      if (!snapshot || typeof snapshot !== 'object') return
      queryClient.setQueryData(['trader-orchestrator-overview'], (old: any) => {
        if (!old || typeof old !== 'object') return old

        const previousWorker =
          old.worker && typeof old.worker === 'object'
            ? old.worker
            : {}
        const previousControl =
          old.control && typeof old.control === 'object'
            ? old.control
            : {}
        const previousMetrics =
          old.metrics && typeof old.metrics === 'object'
            ? old.metrics
            : {}

        const snapshotControl =
          snapshot.control && typeof snapshot.control === 'object'
            ? snapshot.control
            : {}
        const nextEnabled = typeof snapshot.is_enabled === 'boolean'
          ? snapshot.is_enabled
          : (typeof snapshotControl.is_enabled === 'boolean' ? snapshotControl.is_enabled : previousControl.is_enabled)
        const nextPaused = typeof snapshot.is_paused === 'boolean'
          ? snapshot.is_paused
          : (typeof snapshotControl.is_paused === 'boolean' ? snapshotControl.is_paused : previousControl.is_paused)
        const snapshotMode = String(snapshot.mode || snapshotControl.mode || '').trim().toLowerCase()
        const nextMode = snapshotMode === 'shadow' || snapshotMode === 'live'
          ? snapshotMode
          : previousControl.mode

        return {
          ...old,
          control: {
            ...previousControl,
            is_enabled: Boolean(nextEnabled),
            is_paused: Boolean(nextPaused),
            mode: nextMode,
          },
          worker: {
            ...previousWorker,
            ...snapshot,
          },
          metrics: {
            ...previousMetrics,
            decisions_count: Number(snapshot.decisions_count ?? previousMetrics.decisions_count ?? 0),
            orders_count: Number(snapshot.orders_count ?? previousMetrics.orders_count ?? 0),
            open_orders: Number(snapshot.open_orders ?? previousMetrics.open_orders ?? 0),
            gross_exposure_usd: Number(snapshot.gross_exposure_usd ?? previousMetrics.gross_exposure_usd ?? 0),
            daily_pnl: Number(snapshot.daily_pnl ?? previousMetrics.daily_pnl ?? 0),
          },
        }
      })
    }

    const applyPricesUpdateToOpportunityCaches = (payload: any) => {
      if (!payload || typeof payload !== 'object') return
      const marketId = String(payload.market_id || '').trim().toLowerCase()
      if (!marketId) return
      const nextYes = Number(payload.yes_price)
      const nextNo = Number(payload.no_price)
      const yesPrice = Number.isFinite(nextYes) ? nextYes : null
      const noPrice = Number.isFinite(nextNo) ? nextNo : null
      const yesTsRaw = Number(payload.yes_ingest_ts)
      const noTsRaw = Number(payload.no_ingest_ts)
      const ingestTs = Math.max(
        Number.isFinite(yesTsRaw) ? yesTsRaw : 0,
        Number.isFinite(noTsRaw) ? noTsRaw : 0,
      )
      const pricedAtIso = ingestTs > 0 ? new Date(ingestTs * 1000).toISOString() : null
      const isFresh = Boolean(payload.is_fresh)

      const patchRows = (rows: any[]): any[] => rows.map((row: any) => {
        if (!row || typeof row !== 'object' || !Array.isArray(row.markets)) return row
        let touched = false
        const nextMarkets = row.markets.map((market: any) => {
          if (!market || typeof market !== 'object') return market
          const currentMarketId = String(
            market.condition_id || market.conditionId || market.id || '',
          ).trim().toLowerCase()
          if (currentMarketId !== marketId) return market
          touched = true
          const updated = {
            ...market,
            is_price_fresh: isFresh,
            price_age_seconds: isFresh ? 0 : market.price_age_seconds,
          }
          if (yesPrice !== null) {
            updated.current_yes_price = yesPrice
            updated.yes_price = yesPrice
          }
          if (noPrice !== null) {
            updated.current_no_price = noPrice
            updated.no_price = noPrice
          }
          if (pricedAtIso) {
            updated.price_updated_at = pricedAtIso
          }
          return updated
        })
        if (!touched) return row
        return {
          ...row,
          markets: nextMarkets,
          last_priced_at: pricedAtIso || row.last_priced_at,
        }
      })

      queryClient.setQueriesData({ queryKey: ['opportunities'] }, (old: any) => {
        if (Array.isArray(old)) {
          return patchRows(old)
        }
        if (!old || typeof old !== 'object') {
          return old
        }
        if (Array.isArray(old.opportunities)) {
          return {
            ...old,
            opportunities: patchRows(old.opportunities),
          }
        }
        return old
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
    if (messageType === 'opportunity_update') {
      queueInvalidations(
        [
          ['opportunities'],
          ['opportunity-strategy-counts'],
          ['opportunity-category-counts'],
          ['opportunity-subfilters'],
        ],
        { immediate: viewingScannerFamily },
      )
    }
    if (messageType === 'prices_update' && lastMessage.data) {
      applyPricesUpdateToOpportunityCaches(lastMessage.data)
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
      if (Array.isArray(lastMessage.data.markets)) {
        queryClient.setQueryData(['crypto-markets'], lastMessage.data.markets)
      }
    }
    if (messageType === 'trader_orchestrator_status' && lastMessage.data) {
      // Direct cache update for trader orchestrator status
      queryClient.setQueryData(['trader-orchestrator-status'], lastMessage.data)
      mergeOrchestratorSnapshotIntoOverview(lastMessage.data)
      queueInvalidations([
        ['trader-orchestrator-overview'],
        ...(viewingTrading ? ([['traders-list']] as QueryKey[]) : []),
      ], { immediate: true })
    }
    if (messageType === 'trader_decision') {
      upsertTraderDecisionCache(lastMessage.data)
      queueInvalidations([
        ['trader-orchestrator-overview'],
      ])
    }
    if (messageType === 'trader_order') {
      upsertTraderOrderCache(lastMessage.data)
      queueInvalidations([
        ['trader-orchestrator-overview'],
        ...(viewingTrading
          ? ([
              ['trader-orders-all'],
              ['trader-orders'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'execution_order') {
      queueInvalidations([
        ['trader-orchestrator-overview'],
        ...(viewingTrading
          ? ([
              ['trader-orders-all'],
              ['trader-orders'],
            ] as QueryKey[])
          : []),
      ])
    }
    if (messageType === 'trader_event') {
      upsertTraderEventCache(lastMessage.data)
      if (viewingTrading) {
        queueInvalidations([
          ['trader-orchestrator-overview'],
        ])
      }
    }
  }, [context.activeTab, context.opportunitiesView, lastMessage, queryClient, queueInvalidations, setScannerActivity])
}
