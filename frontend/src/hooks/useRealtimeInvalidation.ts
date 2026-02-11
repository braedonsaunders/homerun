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
      queryClient.invalidateQueries({ queryKey: ['news-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['news-workflow-status'] })
    }
    if (lastMessage?.type === 'weather_update' || lastMessage?.type === 'weather_status') {
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-intents'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-status'] })
      queryClient.invalidateQueries({ queryKey: ['weather-workflow-performance'] })
    }
  }, [lastMessage, queryClient, setScannerActivity])
}
