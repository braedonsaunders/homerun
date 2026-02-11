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
    if (lastMessage?.type === 'world_intelligence_update' || lastMessage?.type === 'world_intelligence_status') {
      queryClient.invalidateQueries({ queryKey: ['world-signals'] })
      queryClient.invalidateQueries({ queryKey: ['world-instability'] })
      queryClient.invalidateQueries({ queryKey: ['world-tensions'] })
      queryClient.invalidateQueries({ queryKey: ['world-convergences'] })
      queryClient.invalidateQueries({ queryKey: ['world-anomalies'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-summary'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-status'] })
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
    if (lastMessage?.type === 'autotrader_status' && lastMessage.data) {
      const snapshot = lastMessage.data
      queryClient.setQueryData(['auto-trader-status'], (prev: any) => {
        if (!prev) return prev
        const control = prev.control || {}
        const tradingActive = Boolean(control.is_enabled) && !Boolean(control.is_paused) && !Boolean(control.kill_switch)
        return {
          ...prev,
          running: tradingActive,
          trading_active: tradingActive,
          worker_running: Boolean(snapshot.running),
          snapshot: {
            ...prev.snapshot,
            ...snapshot,
          },
          stats: {
            ...prev.stats,
            total_trades: snapshot.trades_count ?? prev.stats?.total_trades ?? 0,
            daily_trades: snapshot.trades_count ?? prev.stats?.daily_trades ?? 0,
            total_profit: snapshot.daily_pnl ?? prev.stats?.total_profit ?? 0,
            daily_profit: snapshot.daily_pnl ?? prev.stats?.daily_profit ?? 0,
            opportunities_seen: snapshot.signals_seen ?? prev.stats?.opportunities_seen ?? 0,
            opportunities_executed: snapshot.signals_selected ?? prev.stats?.opportunities_executed ?? 0,
            opportunities_skipped: Math.max(
              0,
              (snapshot.signals_seen ?? prev.stats?.opportunities_seen ?? 0)
                - (snapshot.signals_selected ?? prev.stats?.opportunities_executed ?? 0)
            ),
            last_trade_at: snapshot.last_run_at ?? prev.stats?.last_trade_at ?? null,
          },
        }
      })
    }
    if (lastMessage?.type === 'autotrader_decision' && lastMessage.data) {
      const decision = lastMessage.data
      queryClient.setQueryData(['auto-trader-decisions'], (prev: any) => {
        const current = Array.isArray(prev?.decisions) ? prev.decisions : []
        if (current.some((row: any) => row.id === decision.id)) return prev
        const decisions = [decision, ...current].slice(0, 250)
        return {
          total: Math.max(Number(prev?.total || 0), decisions.length),
          decisions,
        }
      })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-metrics'] })
    }
    if (lastMessage?.type === 'autotrader_trade' && lastMessage.data) {
      const trade = lastMessage.data
      queryClient.setQueryData(['auto-trader-trades'], (prev: any) => {
        const current = Array.isArray(prev) ? prev : []
        if (current.some((row: any) => row.id === trade.id)) return prev
        const mapped = {
          id: trade.id,
          opportunity_id: trade.signal_id,
          strategy: trade.source,
          executed_at: trade.executed_at || trade.created_at,
          total_cost: trade.notional_usd || 0,
          expected_profit: 0,
          actual_profit: null,
          status: trade.status,
          mode: trade.mode,
          source: trade.source,
          market_id: trade.market_id,
          direction: trade.direction,
          created_at: trade.created_at,
        }
        return [mapped, ...current].slice(0, 250)
      })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-exposure'] })
      queryClient.invalidateQueries({ queryKey: ['auto-trader-metrics'] })
    }
  }, [lastMessage, queryClient, setScannerActivity])
}
