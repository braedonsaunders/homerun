import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'

import {
  emergencyStopAutoTrader,
  getAutoTraderDecisions,
  getAutoTraderExposure,
  getAutoTraderMetrics,
  getAutoTraderPolicies,
  getAutoTraderStatus,
  getAutoTraderTrades,
  getCopyTradingStatus,
  getSignalStats,
  getTradingBalance,
  getTradingPositions,
  startAutoTrader,
  stopAutoTrader,
  updateAutoTraderPolicies,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { accountModeAtom, selectedAccountIdAtom } from '../store/atoms'
import { Card, CardContent } from './ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

import CommandCenterHeader from './autotrader/CommandCenterHeader'
import DecisionStream from './autotrader/DecisionStream'
import ExecutionTape from './autotrader/ExecutionTape'
import RiskBudgetPanel from './autotrader/RiskBudgetPanel'
import AutoTraderSettingsFlyout from './autotrader/AutoTraderSettingsFlyout'
import SourceMatrixFlyout from './autotrader/SourceMatrixFlyout'
import CommandOutputPanel, { type CommandLogLine } from './autotrader/CommandOutputPanel'
import CurrentHoldingsPanel from './autotrader/CurrentHoldingsPanel'
import TradeHistoryPanel from './autotrader/TradeHistoryPanel'

function nowTime() {
  return new Date().toLocaleTimeString()
}

function shortReason(reason?: string | null): string {
  if (!reason) return 'n/a'
  return reason.length > 80 ? `${reason.slice(0, 77)}...` : reason
}

export default function TradingPanel() {
  const [accountMode] = useAtom(accountModeAtom)
  const [selectedAccountId] = useAtom(selectedAccountIdAtom)
  const queryClient = useQueryClient()
  const { isConnected, lastMessage } = useWebSocket('/ws')

  const [updatingSources, setUpdatingSources] = useState<Set<string>>(new Set())
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [commandLogs, setCommandLogs] = useState<CommandLogLine[]>([])

  const appendLog = useCallback((level: CommandLogLine['level'], message: string) => {
    setCommandLogs((prev) => {
      const line: CommandLogLine = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        ts: nowTime(),
        level,
        message,
      }
      return [...prev, line].slice(-120)
    })
  }, [])

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['auto-trader-status'],
    queryFn: getAutoTraderStatus,
    refetchInterval: 3000,
  })

  const { data: trades = [] } = useQuery({
    queryKey: ['auto-trader-trades'],
    queryFn: () => getAutoTraderTrades(250),
    refetchInterval: 5000,
  })

  const { data: decisionsData } = useQuery({
    queryKey: ['auto-trader-decisions'],
    queryFn: () => getAutoTraderDecisions({ limit: 250 }),
    refetchInterval: 5000,
  })

  const { data: policies } = useQuery({
    queryKey: ['auto-trader-policies'],
    queryFn: getAutoTraderPolicies,
    refetchInterval: 10000,
  })

  const { data: signalStats } = useQuery({
    queryKey: ['signals-stats'],
    queryFn: getSignalStats,
    refetchInterval: 10000,
  })

  const { data: exposure } = useQuery({
    queryKey: ['auto-trader-exposure'],
    queryFn: getAutoTraderExposure,
    refetchInterval: 5000,
  })

  const { data: metrics } = useQuery({
    queryKey: ['auto-trader-metrics'],
    queryFn: getAutoTraderMetrics,
    refetchInterval: 5000,
  })

  const { data: currentHoldings = [] } = useQuery({
    queryKey: ['trading-positions'],
    queryFn: getTradingPositions,
    refetchInterval: 5000,
  })

  const { data: tradingBalance } = useQuery({
    queryKey: ['trading-balance'],
    queryFn: getTradingBalance,
    refetchInterval: 10000,
  })

  const { data: copyStatus } = useQuery({
    queryKey: ['copy-trading-status'],
    queryFn: getCopyTradingStatus,
    refetchInterval: 10000,
  })

  useEffect(() => {
    if (!lastMessage?.type) return

    if (lastMessage.type === 'autotrader_status') {
      const activity = lastMessage.data?.current_activity || 'status update'
      appendLog('event', `status: ${activity}`)
      return
    }

    if (lastMessage.type === 'autotrader_decision') {
      const source = lastMessage.data?.source || 'unknown'
      const decision = lastMessage.data?.decision || 'n/a'
      const reason = shortReason(lastMessage.data?.reason)
      appendLog('event', `decision ${source} -> ${decision} (${reason})`)
      return
    }

    if (lastMessage.type === 'autotrader_trade') {
      const source = lastMessage.data?.source || 'unknown'
      const statusText = lastMessage.data?.status || 'n/a'
      const notional = Number(lastMessage.data?.notional_usd || 0)
      appendLog('event', `trade ${source} ${statusText} $${notional.toFixed(2)}`)
      return
    }

    if (lastMessage.type === 'copy_trade_executed') {
      const wallet = String(lastMessage.data?.source_wallet || '').slice(0, 10)
      appendLog('event', `copy executed from ${wallet || 'wallet'}`)
    }
  }, [appendLog, lastMessage])

  const refreshAutoTraderQueries = () => {
    queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-trades'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-decisions'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-policies'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-exposure'] })
    queryClient.invalidateQueries({ queryKey: ['auto-trader-metrics'] })
    queryClient.invalidateQueries({ queryKey: ['signals-stats'] })
    queryClient.invalidateQueries({ queryKey: ['trading-positions'] })
    queryClient.invalidateQueries({ queryKey: ['trading-balance'] })
  }

  const startMutation = useMutation({
    mutationFn: ({ mode, accountId }: { mode: string; accountId?: string }) =>
      startAutoTrader(mode, accountId),
    onMutate: ({ mode }) => {
      appendLog('info', `command: start autotrader (${mode})`)
    },
    onSuccess: () => {
      appendLog('info', 'autotrader started')
      refreshAutoTraderQueries()
    },
    onError: (error: any) => {
      appendLog('error', `start failed: ${error?.message || 'unknown error'}`)
    },
  })

  const stopMutation = useMutation({
    mutationFn: stopAutoTrader,
    onMutate: () => {
      appendLog('warn', 'command: stop autotrader')
    },
    onSuccess: () => {
      appendLog('warn', 'autotrader stopped')
      refreshAutoTraderQueries()
    },
    onError: (error: any) => {
      appendLog('error', `stop failed: ${error?.message || 'unknown error'}`)
    },
  })

  const emergencyStopMutation = useMutation({
    mutationFn: emergencyStopAutoTrader,
    onMutate: () => {
      appendLog('warn', 'command: emergency stop')
    },
    onSuccess: () => {
      appendLog('warn', 'kill-switch engaged')
      refreshAutoTraderQueries()
    },
    onError: (error: any) => {
      appendLog('error', `emergency stop failed: ${error?.message || 'unknown error'}`)
    },
  })

  const policiesMutation = useMutation({
    mutationFn: updateAutoTraderPolicies,
    onSuccess: () => {
      refreshAutoTraderQueries()
    },
    onError: (error: any) => {
      appendLog('error', `policy update failed: ${error?.message || 'unknown error'}`)
    },
  })

  const decisions = useMemo(() => (decisionsData?.decisions || []).slice(0, 30), [decisionsData])
  const recentTrades = useMemo(() => trades.slice(0, 30), [trades])
  const tradeHistory = useMemo(() => trades.slice(0, 120), [trades])

  const canStart = accountMode === 'live' ? true : Boolean(selectedAccountId)

  const handleStart = () => {
    if (accountMode === 'live') {
      if (!confirm('Start LIVE auto-trading? This uses real money.')) return
      startMutation.mutate({ mode: 'live' })
      return
    }
    startMutation.mutate({ mode: 'paper', accountId: selectedAccountId || undefined })
  }

  const handleStop = () => {
    stopMutation.mutate()
  }

  const handleEmergencyStop = () => {
    if (!confirm('Emergency stop will halt auto-trader decisions immediately. Continue?')) return
    emergencyStopMutation.mutate()
  }

  const handleToggleSource = (source: string, enabled: boolean) => {
    appendLog('info', `command: ${enabled ? 'enable' : 'disable'} source ${source}`)
    setUpdatingSources((prev) => {
      const next = new Set(prev)
      next.add(source)
      return next
    })

    policiesMutation.mutate(
      {
        sources: {
          [source]: { enabled },
        },
      },
      {
        onSuccess: () => {
          appendLog('info', `source ${source} ${enabled ? 'enabled' : 'disabled'}`)
        },
        onSettled: () => {
          setUpdatingSources((prev) => {
            const next = new Set(prev)
            next.delete(source)
            return next
          })
        },
      }
    )
  }

  if (statusLoading) {
    return (
      <div className="flex items-center justify-center py-16 text-muted-foreground text-sm">
        Loading command center...
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-3 overflow-hidden">
      {!canStart && (
        <Card className="border-amber-500/30 bg-amber-500/5 shrink-0">
          <CardContent className="p-3 text-xs text-amber-200">
            Select a sandbox account in the account selector before starting autotrader.
          </CardContent>
        </Card>
      )}

      <div className="shrink-0">
        <CommandCenterHeader
          status={status}
          canStart={canStart}
          startPending={startMutation.isPending}
          stopPending={stopMutation.isPending}
          emergencyPending={emergencyStopMutation.isPending}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenSources={() => setSourcesOpen(true)}
          onStart={handleStart}
          onStop={handleStop}
          onEmergencyStop={handleEmergencyStop}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 flex-1 min-h-0 items-stretch">
        <div className="xl:col-span-7 min-h-0 flex">
          <Tabs defaultValue="live-action-console" className="h-full min-h-0 w-full flex flex-col">
            <TabsList className="mb-3 h-auto w-full shrink-0 grid grid-cols-5 gap-1">
              <TabsTrigger value="live-action-console" className="w-full px-2 text-center whitespace-normal leading-tight">Live Action Console</TabsTrigger>
              <TabsTrigger value="decision-stream" className="w-full px-2 text-center whitespace-normal leading-tight">Decision Stream</TabsTrigger>
              <TabsTrigger value="execution-tape" className="w-full px-2 text-center whitespace-normal leading-tight">Execution Tape</TabsTrigger>
              <TabsTrigger value="current-holdings" className="w-full px-2 text-center whitespace-normal leading-tight">Current Holdings</TabsTrigger>
              <TabsTrigger value="history" className="w-full px-2 text-center whitespace-normal leading-tight">History</TabsTrigger>
            </TabsList>

            <TabsContent value="live-action-console" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1 data-[state=active]:flex-col">
              <CommandOutputPanel logs={commandLogs} connected={isConnected} />
            </TabsContent>

            <TabsContent value="decision-stream" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1 data-[state=active]:flex-col">
              <DecisionStream decisions={decisions} maxItems={8} />
            </TabsContent>

            <TabsContent value="execution-tape" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1 data-[state=active]:flex-col">
              <ExecutionTape trades={recentTrades} maxItems={8} />
            </TabsContent>

            <TabsContent value="current-holdings" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1 data-[state=active]:flex-col">
              <CurrentHoldingsPanel positions={currentHoldings} balance={tradingBalance} />
            </TabsContent>

            <TabsContent value="history" className="mt-0 min-h-0 data-[state=active]:flex data-[state=active]:flex-1 data-[state=active]:flex-col">
              <TradeHistoryPanel trades={tradeHistory} />
            </TabsContent>
          </Tabs>
        </div>
        <div className="xl:col-span-5 min-h-0 flex">
          <RiskBudgetPanel status={status} exposure={exposure} />
        </div>
      </div>

      <AutoTraderSettingsFlyout isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <SourceMatrixFlyout
        isOpen={sourcesOpen}
        onClose={() => setSourcesOpen(false)}
        policies={policies}
        signalStats={signalStats}
        metrics={metrics}
        exposure={exposure}
        copyStatus={copyStatus}
        updatingSources={updatingSources}
        onToggleSource={handleToggleSource}
      />
    </div>
  )
}
