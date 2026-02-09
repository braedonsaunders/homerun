import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getCopyConfigs,
  getCopyTrades,
  getCopyTradingStatus,
  deleteCopyConfig,
  enableCopyConfig,
  disableCopyConfig,
  forceSyncCopyConfig,
  updateCopyConfig,
  getSimulationAccounts,
  CopyConfig,
  CopiedTrade,
} from '../services/api'
import { cn } from '../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Separator } from './ui/separator'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import {
  Copy,
  Trash2,
  Power,
  PowerOff,
  RefreshCw,
  Settings,
  ChevronDown,
  ChevronUp,
  Activity,
  DollarSign,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react'

type SubView = 'configs' | 'trades' | 'status'

export default function CopyTradingPanel() {
  const [subView, setSubView] = useState<SubView>('configs')
  const [expandedConfig, setExpandedConfig] = useState<string | null>(null)
  const [tradeFilter, setTradeFilter] = useState<string>('')
  const queryClient = useQueryClient()

  // Queries
  const { data: configs = [], isLoading: configsLoading } = useQuery({
    queryKey: ['copy-configs'],
    queryFn: () => getCopyConfigs(),
    refetchInterval: 10000,
  })

  const { data: trades = [], isLoading: tradesLoading } = useQuery({
    queryKey: ['copy-trades', tradeFilter],
    queryFn: () => getCopyTrades({ status: tradeFilter || undefined, limit: 100 }),
    refetchInterval: 15000,
  })

  const { data: status } = useQuery({
    queryKey: ['copy-trading-status'],
    queryFn: getCopyTradingStatus,
    refetchInterval: 10000,
  })

  const { data: accounts = [] } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  // Mutations
  const deleteMutation = useMutation({
    mutationFn: deleteCopyConfig,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['copy-configs'] }),
  })

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      enabled ? disableCopyConfig(id) : enableCopyConfig(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['copy-configs'] }),
  })

  const syncMutation = useMutation({
    mutationFn: forceSyncCopyConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['copy-configs'] })
      queryClient.invalidateQueries({ queryKey: ['copy-trades'] })
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, params }: { id: string; params: Record<string, unknown> }) =>
      updateCopyConfig(id, params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['copy-configs'] }),
  })

  // Helpers
  const getAccountName = (accountId: string) => {
    const acc = accounts.find(a => a.id === accountId)
    return acc?.name || accountId.slice(0, 8) + '...'
  }

  const totalPnl = configs.reduce((sum, c) => sum + (c.stats?.total_pnl || 0), 0)
  const totalCopied = configs.reduce((sum, c) => sum + (c.stats?.total_copied || 0), 0)
  const totalSuccess = configs.reduce((sum, c) => sum + (c.stats?.successful_copies || 0), 0)
  const successRate = totalCopied > 0 ? (totalSuccess / totalCopied * 100) : 0

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Copy className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Active Configs</p>
              <p className="text-lg font-semibold">
                {configs.filter(c => c.enabled).length} / {configs.length}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Activity className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Total Copied</p>
              <p className="text-lg font-semibold">{totalCopied}</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className="p-2 bg-yellow-500/10 rounded-lg">
              <CheckCircle className="w-5 h-5 text-yellow-500" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Success Rate</p>
              <p className="text-lg font-semibold">{successRate.toFixed(1)}%</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardContent className="flex items-center gap-3 p-4">
            <div className={cn("p-2 rounded-lg", totalPnl >= 0 ? "bg-green-500/10" : "bg-red-500/10")}>
              <DollarSign className={cn("w-5 h-5", totalPnl >= 0 ? "text-green-500" : "text-red-500")} />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Total PnL</p>
              <p className={cn("text-lg font-semibold", totalPnl >= 0 ? "text-green-400" : "text-red-400")}>
                ${totalPnl.toFixed(2)}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Service Status Banner */}
      {status && (
        <div className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-lg text-sm",
          status.service_running
            ? "bg-green-500/10 text-green-400 border border-green-500/20"
            : "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"
        )}>
          <span className={cn(
            "w-2 h-2 rounded-full",
            status.service_running ? "bg-green-500 animate-pulse" : "bg-yellow-500"
          )} />
          {status.service_running
            ? `Copy trading service running - polling every ${status.poll_interval_seconds}s - tracking ${status.tracked_wallets?.length || 0} wallets`
            : 'Copy trading service not running - add a config to start'}
        </div>
      )}

      {/* Sub-navigation */}
      <div className="flex items-center gap-2">
        {(['configs', 'trades', 'status'] as const).map(view => (
          <Button
            key={view}
            variant="outline"
            size="sm"
            onClick={() => setSubView(view)}
            className={cn(
              "capitalize",
              subView === view
                ? "bg-blue-500/20 text-blue-400 border-blue-500/30"
                : "bg-card text-muted-foreground border-border"
            )}
          >
            {view === 'configs' ? 'Configurations' : view === 'trades' ? 'Trade History' : 'Service Status'}
          </Button>
        ))}
      </div>

      {/* Configurations View */}
      {subView === 'configs' && (
        <div className="space-y-4">
          {configsLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : configs.length === 0 ? (
            <Card className="border-border">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Copy className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">No copy trading configurations</p>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  Go to Traders tab and click "Copy Trade" on a trader to set up copy trading
                </p>
              </CardContent>
            </Card>
          ) : (
            configs.map(config => (
              <ConfigCard
                key={config.id}
                config={config}
                accountName={getAccountName(config.account_id)}
                expanded={expandedConfig === config.id}
                onToggleExpand={() => setExpandedConfig(
                  expandedConfig === config.id ? null : config.id
                )}
                onToggleEnabled={() => toggleMutation.mutate({ id: config.id, enabled: config.enabled })}
                onDelete={() => deleteMutation.mutate(config.id)}
                onSync={() => syncMutation.mutate(config.id)}
                onUpdate={(params) => updateMutation.mutate({ id: config.id, params })}
                isToggling={toggleMutation.isPending}
                isSyncing={syncMutation.isPending}
              />
            ))
          )}
        </div>
      )}

      {/* Trade History View */}
      {subView === 'trades' && (
        <div className="space-y-4">
          {/* Filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Filter:</span>
            {['', 'executed', 'pending', 'failed', 'skipped'].map(f => (
              <button
                key={f}
                onClick={() => setTradeFilter(f)}
                className={cn(
                  'px-2.5 py-1 rounded text-xs font-medium transition-colors capitalize',
                  tradeFilter === f
                    ? 'bg-primary/20 text-primary'
                    : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                )}
              >
                {f || 'All'}
              </button>
            ))}
          </div>

          {tradesLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : trades.length === 0 ? (
            <Card className="border-border">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Activity className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">No copied trades yet</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-2">
              {trades.map((trade: CopiedTrade) => (
                <TradeRow key={trade.id} trade={trade} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Service Status View */}
      {subView === 'status' && status && (
        <div className="space-y-4">
          <Card className="border-border">
            <CardHeader>
              <CardTitle className="text-sm">Service Overview</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Status:</span>{' '}
                  <Badge variant={status.service_running ? 'default' : 'destructive'}>
                    {status.service_running ? 'Running' : 'Stopped'}
                  </Badge>
                </div>
                <div>
                  <span className="text-muted-foreground">Poll Interval:</span>{' '}
                  {status.poll_interval_seconds}s
                </div>
                <div>
                  <span className="text-muted-foreground">Total Configs:</span>{' '}
                  {status.total_configs}
                </div>
                <div>
                  <span className="text-muted-foreground">Enabled Configs:</span>{' '}
                  {status.enabled_configs}
                </div>
              </div>

              {status.tracked_wallets && status.tracked_wallets.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <p className="text-sm font-medium mb-2">Tracked Wallets</p>
                    <div className="space-y-1">
                      {status.tracked_wallets.map(wallet => (
                        <div key={wallet} className="font-mono text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded">
                          {wallet}
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {status.configs_summary && status.configs_summary.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <p className="text-sm font-medium mb-2">Configs Summary</p>
                    <div className="space-y-2">
                      {status.configs_summary.map(cfg => (
                        <div key={cfg.id} className="flex items-center justify-between text-sm bg-muted/30 px-3 py-2 rounded-lg">
                          <div className="flex items-center gap-2">
                            <span className={cn(
                              "w-2 h-2 rounded-full",
                              cfg.enabled ? "bg-green-500" : "bg-gray-500"
                            )} />
                            <span className="font-mono text-xs">{cfg.source_wallet.slice(0, 8)}...{cfg.source_wallet.slice(-4)}</span>
                            <Badge variant="outline" className="text-[10px]">{cfg.copy_mode}</Badge>
                          </div>
                          <div className="flex items-center gap-3 text-xs text-muted-foreground">
                            <span>{cfg.total_copied} copied</span>
                            <span className="text-green-400">{cfg.successful_copies} success</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

// ==================== Sub-Components ====================

function ConfigCard({
  config,
  accountName,
  expanded,
  onToggleExpand,
  onToggleEnabled,
  onDelete,
  onSync,
  onUpdate,
  isToggling,
  isSyncing,
}: {
  config: CopyConfig
  accountName: string
  expanded: boolean
  onToggleExpand: () => void
  onToggleEnabled: () => void
  onDelete: () => void
  onSync: () => void
  onUpdate: (params: Record<string, unknown>) => void
  isToggling: boolean
  isSyncing: boolean
}) {
  const [editMode, setEditMode] = useState(false)
  const [editValues, setEditValues] = useState({
    max_position_size: config.settings?.max_position_size ?? 1000,
    copy_delay_seconds: config.settings?.copy_delay_seconds ?? 5,
    slippage_tolerance: config.settings?.slippage_tolerance ?? 1.0,
    min_roi_threshold: config.settings?.min_roi_threshold ?? 2.5,
    copy_buys: config.settings?.copy_buys ?? true,
    copy_sells: config.settings?.copy_sells ?? true,
  })

  const stats = config.stats || { total_copied: 0, successful_copies: 0, failed_copies: 0, total_pnl: 0 }

  return (
    <Card className={cn("border-border transition-colors", config.enabled ? "" : "opacity-60")}>
      <CardContent className="p-4">
        {/* Main row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className={cn(
              "w-3 h-3 rounded-full",
              config.enabled ? "bg-green-500" : "bg-gray-500"
            )} />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm">
                  {config.source_wallet.slice(0, 10)}...{config.source_wallet.slice(-6)}
                </span>
                <Badge variant="outline" className="text-[10px]">
                  {config.copy_mode || 'all_trades'}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Account: {accountName}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Stats inline */}
            <div className="flex items-center gap-3 text-xs">
              <span className="text-muted-foreground">{stats.total_copied} copied</span>
              <span className="text-green-400">{stats.successful_copies} ok</span>
              {stats.failed_copies > 0 && (
                <span className="text-red-400">{stats.failed_copies} fail</span>
              )}
              <span className={cn(
                "font-medium",
                stats.total_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                ${stats.total_pnl?.toFixed(2) || '0.00'}
              </span>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onSync}
                    disabled={isSyncing || !config.enabled}
                    className="px-2"
                  >
                    <RefreshCw className={cn("w-3.5 h-3.5", isSyncing && "animate-spin")} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Force sync now</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onToggleEnabled}
                    disabled={isToggling}
                    className={cn(
                      "px-2",
                      config.enabled ? "text-green-400 hover:text-red-400" : "text-muted-foreground hover:text-green-400"
                    )}
                  >
                    {config.enabled ? <Power className="w-3.5 h-3.5" /> : <PowerOff className="w-3.5 h-3.5" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{config.enabled ? 'Disable' : 'Enable'}</TooltipContent>
              </Tooltip>

              <Button
                variant="ghost"
                size="sm"
                onClick={onToggleExpand}
                className="px-2"
              >
                {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </Button>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onDelete}
                    className="px-2 text-muted-foreground hover:text-red-400"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Delete config</TooltipContent>
              </Tooltip>
            </div>
          </div>
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="mt-4 pt-4 border-t border-border">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Max Position Size</p>
                {editMode ? (
                  <Input
                    type="number"
                    value={editValues.max_position_size}
                    onChange={e => setEditValues({ ...editValues, max_position_size: Number(e.target.value) })}
                    className="h-7 text-sm bg-card"
                  />
                ) : (
                  <p className="font-medium">${config.settings?.max_position_size?.toFixed(0) || '1000'}</p>
                )}
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Copy Delay</p>
                {editMode ? (
                  <Input
                    type="number"
                    value={editValues.copy_delay_seconds}
                    onChange={e => setEditValues({ ...editValues, copy_delay_seconds: Number(e.target.value) })}
                    className="h-7 text-sm bg-card"
                  />
                ) : (
                  <p className="font-medium">{config.settings?.copy_delay_seconds || 5}s</p>
                )}
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Slippage Tolerance</p>
                {editMode ? (
                  <Input
                    type="number"
                    step="0.1"
                    value={editValues.slippage_tolerance}
                    onChange={e => setEditValues({ ...editValues, slippage_tolerance: Number(e.target.value) })}
                    className="h-7 text-sm bg-card"
                  />
                ) : (
                  <p className="font-medium">{config.settings?.slippage_tolerance || 1.0}%</p>
                )}
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Min ROI Threshold</p>
                {editMode ? (
                  <Input
                    type="number"
                    step="0.5"
                    value={editValues.min_roi_threshold}
                    onChange={e => setEditValues({ ...editValues, min_roi_threshold: Number(e.target.value) })}
                    className="h-7 text-sm bg-card"
                  />
                ) : (
                  <p className="font-medium">{config.settings?.min_roi_threshold || 2.5}%</p>
                )}
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Copy Buys</p>
                {editMode ? (
                  <button
                    onClick={() => setEditValues({ ...editValues, copy_buys: !editValues.copy_buys })}
                    className={cn(
                      "px-2 py-0.5 rounded text-xs font-medium",
                      editValues.copy_buys ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                    )}
                  >
                    {editValues.copy_buys ? 'Yes' : 'No'}
                  </button>
                ) : (
                  <Badge variant={config.settings?.copy_buys !== false ? 'default' : 'destructive'}>
                    {config.settings?.copy_buys !== false ? 'Yes' : 'No'}
                  </Badge>
                )}
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Copy Sells</p>
                {editMode ? (
                  <button
                    onClick={() => setEditValues({ ...editValues, copy_sells: !editValues.copy_sells })}
                    className={cn(
                      "px-2 py-0.5 rounded text-xs font-medium",
                      editValues.copy_sells ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                    )}
                  >
                    {editValues.copy_sells ? 'Yes' : 'No'}
                  </button>
                ) : (
                  <Badge variant={config.settings?.copy_sells !== false ? 'default' : 'destructive'}>
                    {config.settings?.copy_sells !== false ? 'Yes' : 'No'}
                  </Badge>
                )}
              </div>
            </div>

            {/* Edit/Save buttons */}
            <div className="flex items-center gap-2 mt-4">
              {editMode ? (
                <>
                  <Button
                    size="sm"
                    onClick={() => {
                      onUpdate(editValues)
                      setEditMode(false)
                    }}
                    className="bg-green-500 hover:bg-green-600 text-white"
                  >
                    Save Changes
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setEditMode(false)}
                  >
                    Cancel
                  </Button>
                </>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setEditMode(true)}
                  className="flex items-center gap-1"
                >
                  <Settings className="w-3 h-3" />
                  Edit Settings
                </Button>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function TradeRow({ trade }: { trade: CopiedTrade }) {
  const statusConfig = {
    executed: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-500/10' },
    pending: { icon: Clock, color: 'text-yellow-400', bg: 'bg-yellow-500/10' },
    failed: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-500/10' },
    skipped: { icon: AlertCircle, color: 'text-muted-foreground', bg: 'bg-muted-foreground/10' },
  }[trade.status] || { icon: Clock, color: 'text-muted-foreground', bg: 'bg-muted-foreground/10' }

  const StatusIcon = statusConfig.icon
  const isBuy = trade.side === 'BUY'

  return (
    <Card className="border-border">
      <CardContent className="flex items-center justify-between p-3">
        <div className="flex items-center gap-3">
          <div className={cn("p-1.5 rounded-lg", statusConfig.bg)}>
            <StatusIcon className={cn("w-3.5 h-3.5", statusConfig.color)} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1 text-sm">
                {isBuy ? (
                  <ArrowUpRight className="w-3 h-3 text-green-400" />
                ) : (
                  <ArrowDownRight className="w-3 h-3 text-red-400" />
                )}
                <span className={isBuy ? 'text-green-400' : 'text-red-400'}>{trade.side}</span>
              </span>
              <span className="text-sm">{trade.outcome}</span>
              <Badge variant="outline" className="text-[10px]">{trade.status}</Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5 max-w-md truncate">
              {trade.market_question || trade.market_id}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs">
          <div className="text-right">
            <p className="text-muted-foreground">Source</p>
            <p className="font-mono">{trade.source_size?.toFixed(1)} @ ${trade.source_price?.toFixed(3)}</p>
          </div>
          {trade.executed_price != null && (
            <div className="text-right">
              <p className="text-muted-foreground">Executed</p>
              <p className="font-mono">{trade.executed_size?.toFixed(1)} @ ${trade.executed_price?.toFixed(3)}</p>
            </div>
          )}
          {trade.realized_pnl != null && (
            <div className="text-right">
              <p className="text-muted-foreground">PnL</p>
              <p className={cn("font-mono font-medium", trade.realized_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                ${trade.realized_pnl.toFixed(2)}
              </p>
            </div>
          )}
          <div className="text-right text-muted-foreground">
            {trade.copied_at && new Date(trade.copied_at).toLocaleTimeString()}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
