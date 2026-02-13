import { useEffect } from 'react'
import { QueryClient } from '@tanstack/react-query'

type WSMessage = {
  type?: string
  data?: Record<string, any>
} | null | undefined

export function useRealtimeInvalidation(
  lastMessage: WSMessage,
  queryClient: QueryClient,
  setScannerActivity: (activity: string) => void
) {
  useEffect(() => {
    if (lastMessage?.type === 'opportunities_update' || lastMessage?.type === 'init') {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['opportunity-counts'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
    }
    if (lastMessage?.type === 'opportunity_events') {
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['opportunity-counts'] })
    }
    if (lastMessage?.type === 'scanner_status' && lastMessage.data) {
      queryClient.setQueryData(['scanner-status'], lastMessage.data)
    }
    if (lastMessage?.type === 'scanner_activity') {
      setScannerActivity(lastMessage.data?.activity || 'Idle')
    }
    if (lastMessage?.type === 'wallet_trade') {
      queryClient.invalidateQueries({ queryKey: ['copy-trades'] })
      queryClient.invalidateQueries({ queryKey: ['copy-trading-status'] })
    }
    if (
      lastMessage?.type === 'copy_trade_detected'
      || lastMessage?.type === 'copy_trade_executed'
    ) {
      queryClient.invalidateQueries({ queryKey: ['copy-trades'] })
      queryClient.invalidateQueries({ queryKey: ['copy-configs'] })
      queryClient.invalidateQueries({ queryKey: ['copy-trading-status'] })
    }
    if (lastMessage?.type === 'tracked_trader_signal') {
      queryClient.invalidateQueries({ queryKey: ['tracked-trader-opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-confluence'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-active-signal-count'] })
    }
    if (lastMessage?.type === 'tracked_trader_pool_update') {
      queryClient.invalidateQueries({ queryKey: ['discovery-pool-stats'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
    }
    if (
      lastMessage?.type === 'news_update'
      || lastMessage?.type === 'news_workflow_update'
      || lastMessage?.type === 'news_workflow_status'
    ) {
      queryClient.invalidateQueries({ queryKey: ['news-articles'] })
      queryClient.invalidateQueries({ queryKey: ['news-matches'] })
      queryClient.invalidateQueries({ queryKey: ['news-edges'] })
      queryClient.invalidateQueries({ queryKey: ['news-feed-status'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-findings'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-findings-count'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
    }
    if (lastMessage?.type === 'weather_update' || lastMessage?.type === 'weather_status') {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-performance'] })
    }
    if (lastMessage?.type === 'world_intelligence_update' || lastMessage?.type === 'world_intelligence_status') {
      queryClient.invalidateQueries({ queryKey: ['world-signals'] })
      queryClient.invalidateQueries({ queryKey: ['world-instability'] })
      queryClient.invalidateQueries({ queryKey: ['world-tensions'] })
      queryClient.invalidateQueries({ queryKey: ['world-convergences'] })
      queryClient.invalidateQueries({ queryKey: ['world-anomalies'] })
      queryClient.invalidateQueries({ queryKey: ['world-regions'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-summary'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-status'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-sources'] })
    }
    if (lastMessage?.type === 'worker_status_update') {
      queryClient.invalidateQueries({ queryKey: ['workers-status'] })
      queryClient.invalidateQueries({ queryKey: ['scanner-status'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
    }
    if (lastMessage?.type === 'signals_update') {
      queryClient.invalidateQueries({ queryKey: ['signals'] })
      queryClient.invalidateQueries({ queryKey: ['signals-stats'] })
    }
    if (lastMessage?.type === 'trader_orchestrator_status') {
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-status'] })
      queryClient.invalidateQueries({ queryKey: ['traders-list'] })
    }
    if (lastMessage?.type === 'trader_decision') {
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
      queryClient.invalidateQueries({ queryKey: ['traders-decisions'] })
      queryClient.invalidateQueries({ queryKey: ['trader-decisions-all'] })
      queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    }
    if (lastMessage?.type === 'trader_order') {
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
      queryClient.invalidateQueries({ queryKey: ['traders-orders'] })
      queryClient.invalidateQueries({ queryKey: ['traders-decisions'] })
      queryClient.invalidateQueries({ queryKey: ['trader-orders-all'] })
      queryClient.invalidateQueries({ queryKey: ['trader-decisions-all'] })
      queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    }
    if (lastMessage?.type === 'trader_event') {
      queryClient.invalidateQueries({ queryKey: ['traders-events'] })
      queryClient.invalidateQueries({ queryKey: ['trader-events-all'] })
    }
  }, [lastMessage, queryClient, setScannerActivity])
}
