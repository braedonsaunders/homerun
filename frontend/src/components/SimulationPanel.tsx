import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Activity,
  RefreshCw,
  Play,
  Trash2,
  BarChart3,
  Briefcase,
  Award,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  ChevronDown,
  ChevronUp,
  BookOpen,
  PieChart,
  ArrowLeft,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import {
  getSimulationAccounts,
  createSimulationAccount,
  deleteSimulationAccount,
  getAccountTrades,
  getAccountPositions,
  getAccountEquityHistory,
  getOpportunities,
} from '../services/api'
import type { SimulationAccount, SimulationPosition, Opportunity } from '../services/api'
import TradeExecutionModal from './TradeExecutionModal'

type DetailTab = 'overview' | 'holdings' | 'trades' | 'execute'

export default function SimulationPanel() {
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)
  const [newAccountName, setNewAccountName] = useState('')
  const [newAccountCapital, setNewAccountCapital] = useState(10000)
  const [accountToDelete, setAccountToDelete] = useState<SimulationAccount | null>(null)
  const [executingOpportunity, setExecutingOpportunity] = useState<Opportunity | null>(null)
  const [detailTab, setDetailTab] = useState<DetailTab>('overview')
  const [tradeSort, setTradeSort] = useState<'date' | 'pnl' | 'cost'>('date')
  const [tradeSortDir, setTradeSortDir] = useState<'asc' | 'desc'>('desc')
  const [tradeFilter, setTradeFilter] = useState<string>('all')
  const queryClient = useQueryClient()

  const { data: accounts = [], isLoading } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  const { data: opportunitiesData } = useQuery({
    queryKey: ['opportunities'],
    queryFn: () => getOpportunities({ limit: 20 }),
  })
  const opportunities = opportunitiesData?.opportunities ?? []

  const { data: trades = [] } = useQuery({
    queryKey: ['account-trades', selectedAccount],
    queryFn: () => selectedAccount ? getAccountTrades(selectedAccount, 500) : Promise.resolve([]),
    enabled: !!selectedAccount,
  })

  const { data: positions = [] } = useQuery({
    queryKey: ['account-positions', selectedAccount],
    queryFn: () => selectedAccount ? getAccountPositions(selectedAccount) : Promise.resolve([]),
    enabled: !!selectedAccount,
  })

  const { data: equityHistory } = useQuery({
    queryKey: ['account-equity', selectedAccount],
    queryFn: () => selectedAccount ? getAccountEquityHistory(selectedAccount) : Promise.resolve(null),
    enabled: !!selectedAccount,
  })

  const createMutation = useMutation({
    mutationFn: () => createSimulationAccount({
      name: newAccountName,
      initial_capital: newAccountCapital
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
      setShowCreateForm(false)
      setNewAccountName('')
    }
  })

  const deleteMutation = useMutation({
    mutationFn: (accountId: string) => deleteSimulationAccount(accountId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
      if (selectedAccount === accountToDelete?.id) {
        setSelectedAccount(null)
      }
      setAccountToDelete(null)
    }
  })

  const selectedAccountData = accounts.find(a => a.id === selectedAccount)
  const summary = equityHistory?.summary

  // Sorted/filtered trades
  const processedTrades = useMemo(() => {
    let filtered = [...trades]
    if (tradeFilter !== 'all') {
      filtered = filtered.filter(t => t.status === tradeFilter)
    }
    filtered.sort((a, b) => {
      let cmp = 0
      if (tradeSort === 'date') cmp = new Date(a.executed_at).getTime() - new Date(b.executed_at).getTime()
      else if (tradeSort === 'pnl') cmp = (a.actual_pnl || 0) - (b.actual_pnl || 0)
      else if (tradeSort === 'cost') cmp = a.total_cost - b.total_cost
      return tradeSortDir === 'desc' ? -cmp : cmp
    })
    return filtered
  }, [trades, tradeFilter, tradeSort, tradeSortDir])

  // Strategy breakdown from trades
  const strategyBreakdown = useMemo(() => {
    const map: Record<string, { trades: number; pnl: number; wins: number; losses: number; cost: number }> = {}
    trades.forEach(t => {
      if (!map[t.strategy_type]) map[t.strategy_type] = { trades: 0, pnl: 0, wins: 0, losses: 0, cost: 0 }
      map[t.strategy_type].trades++
      map[t.strategy_type].pnl += t.actual_pnl || 0
      map[t.strategy_type].cost += t.total_cost
      if (t.status === 'resolved_win') map[t.strategy_type].wins++
      if (t.status === 'resolved_loss') map[t.strategy_type].losses++
    })
    return map
  }, [trades])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Sandbox Trading</h2>
          <p className="text-sm text-muted-foreground">Practice trading without risking real money</p>
        </div>
        <Button
          onClick={() => setShowCreateForm(true)}
          className="gap-2"
        >
          <Plus className="w-4 h-4" />
          New Account
        </Button>
      </div>

      {/* Create Account Form */}
      {showCreateForm && (
        <Card className="bg-card border-border rounded-lg shadow-none p-4">
          <h3 className="font-medium mb-4">Create Simulation Account</h3>
          <div className="flex gap-4">
            <Input
              type="text"
              value={newAccountName}
              onChange={(e) => setNewAccountName(e.target.value)}
              placeholder="Account name"
              className="flex-1"
            />
            <Input
              type="number"
              value={newAccountCapital}
              onChange={(e) => setNewAccountCapital(Number(e.target.value))}
              placeholder="Initial capital"
              className="w-40"
            />
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!newAccountName || createMutation.isPending}
              className="bg-green-500 hover:bg-green-600"
            >
              Create
            </Button>
            <Button
              variant="secondary"
              onClick={() => setShowCreateForm(false)}
            >
              Cancel
            </Button>
          </div>
        </Card>
      )}

      {/* Accounts List */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : accounts.length === 0 ? (
        <Card className="text-center py-12 bg-card border-border shadow-none">
          <DollarSign className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No simulation accounts yet</p>
          <p className="text-sm text-muted-foreground">Create one to start sandbox trading</p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {accounts.map((account) => (
            <AccountCard
              key={account.id}
              account={account}
              isSelected={selectedAccount === account.id}
              onSelect={() => {
                setSelectedAccount(account.id)
                setDetailTab('overview')
              }}
              onDelete={() => setAccountToDelete(account)}
            />
          ))}
        </div>
      )}

      {/* Selected Account Dashboard */}
      {selectedAccountData && (
        <div className="space-y-4">
          {/* Dashboard Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                onClick={() => setSelectedAccount(null)}
                className="p-1.5 h-auto"
              >
                <ArrowLeft className="w-5 h-5 text-muted-foreground" />
              </Button>
              <div>
                <h3 className="text-lg font-bold">{selectedAccountData.name}</h3>
                <p className="text-xs text-muted-foreground">
                  Created {selectedAccountData.created_at ? new Date(selectedAccountData.created_at).toLocaleDateString() : 'N/A'}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              onClick={() => {
                queryClient.invalidateQueries({ queryKey: ['account-equity', selectedAccount] })
                queryClient.invalidateQueries({ queryKey: ['account-trades', selectedAccount] })
                queryClient.invalidateQueries({ queryKey: ['account-positions', selectedAccount] })
              }}
              className="gap-2 bg-muted hover:bg-muted/80 rounded-lg h-auto px-3 py-1.5 text-xs"
            >
              <RefreshCw className="w-3 h-3" />
              Refresh
            </Button>
          </div>

          {/* Key Metrics Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            <MiniStat
              label="Initial Capital"
              value={`$${selectedAccountData.initial_capital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
              icon={<DollarSign className="w-4 h-4 text-muted-foreground" />}
            />
            <MiniStat
              label="Current Capital"
              value={`$${selectedAccountData.current_capital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
              icon={<Briefcase className="w-4 h-4 text-blue-400" />}
            />
            <MiniStat
              label="Realized P&L"
              value={`${selectedAccountData.total_pnl >= 0 ? '+' : ''}$${selectedAccountData.total_pnl.toFixed(2)}`}
              icon={selectedAccountData.total_pnl >= 0 ? <TrendingUp className="w-4 h-4 text-green-400" /> : <TrendingDown className="w-4 h-4 text-red-400" />}
              valueColor={selectedAccountData.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <MiniStat
              label="Unrealized P&L"
              value={`${(selectedAccountData.unrealized_pnl || 0) >= 0 ? '+' : ''}$${(selectedAccountData.unrealized_pnl || 0).toFixed(2)}`}
              icon={<Activity className="w-4 h-4 text-yellow-400" />}
              valueColor={(selectedAccountData.unrealized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <MiniStat
              label="ROI"
              value={`${selectedAccountData.roi_percent >= 0 ? '+' : ''}${selectedAccountData.roi_percent.toFixed(2)}%`}
              icon={<Target className="w-4 h-4 text-purple-400" />}
              valueColor={selectedAccountData.roi_percent >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <MiniStat
              label="Win Rate"
              value={`${selectedAccountData.win_rate.toFixed(1)}%`}
              icon={<Award className="w-4 h-4 text-yellow-400" />}
              subtitle={`${selectedAccountData.winning_trades}W / ${selectedAccountData.losing_trades}L`}
            />
          </div>

          {/* Advanced Metrics */}
          {summary && (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              <MiniStat
                label="Book Value"
                value={`$${summary.book_value.toFixed(2)}`}
                icon={<BookOpen className="w-4 h-4 text-cyan-400" />}
              />
              <MiniStat
                label="Market Value"
                value={`$${summary.market_value.toFixed(2)}`}
                icon={<BarChart3 className="w-4 h-4 text-blue-400" />}
              />
              <MiniStat
                label="Total P&L"
                value={`${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl.toFixed(2)}`}
                icon={summary.total_pnl >= 0 ? <ArrowUpRight className="w-4 h-4 text-green-400" /> : <ArrowDownRight className="w-4 h-4 text-red-400" />}
                valueColor={summary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
              />
              <MiniStat
                label="Profit Factor"
                value={summary.profit_factor > 0 ? summary.profit_factor.toFixed(2) : 'N/A'}
                icon={<PieChart className="w-4 h-4 text-indigo-400" />}
              />
              <MiniStat
                label="Max Drawdown"
                value={`$${summary.max_drawdown.toFixed(2)}`}
                icon={<AlertTriangle className="w-4 h-4 text-orange-400" />}
                subtitle={`${summary.max_drawdown_pct.toFixed(1)}%`}
                valueColor="text-orange-400"
              />
              <MiniStat
                label="Avg Win / Loss"
                value={`$${summary.avg_win.toFixed(2)}`}
                subtitle={`/ -$${summary.avg_loss.toFixed(2)}`}
                icon={<Activity className="w-4 h-4 text-muted-foreground" />}
              />
            </div>
          )}

          {/* Tab Navigation */}
          <Tabs value={detailTab} onValueChange={(v) => setDetailTab(v as DetailTab)} className="w-fit">
            <TabsList className="bg-card rounded-lg p-1 border border-border h-auto">
              {([
                { key: 'overview', label: 'Performance', icon: <BarChart3 className="w-3.5 h-3.5" /> },
                { key: 'holdings', label: 'Holdings', icon: <Briefcase className="w-3.5 h-3.5" /> },
                { key: 'trades', label: 'Trade History', icon: <Activity className="w-3.5 h-3.5" /> },
                { key: 'execute', label: 'Execute', icon: <Play className="w-3.5 h-3.5" /> },
              ] as { key: DetailTab; label: string; icon: React.ReactNode }[]).map(tab => (
                <TabsTrigger
                  key={tab.key}
                  value={tab.key}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium data-[state=active]:bg-blue-500 data-[state=active]:text-foreground text-muted-foreground"
                >
                  {tab.icon}
                  {tab.label}
                  {tab.key === 'holdings' && positions.length > 0 && (
                    <Badge className="ml-1 px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs border-0">
                      {positions.length}
                    </Badge>
                  )}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>

          {/* Tab Content */}
          {detailTab === 'overview' && (
            <div className="space-y-4">
              {/* Equity Curve */}
              <Card className="bg-card border-border shadow-none p-6">
                <h4 className="font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  Account Equity Over Time
                </h4>
                {equityHistory && equityHistory.equity_points.length > 1 ? (
                  <div className="h-64">
                    <EquityChart
                      points={equityHistory.equity_points}
                      initialCapital={equityHistory.initial_capital}
                    />
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    Execute some trades to see your equity curve
                  </div>
                )}
              </Card>

              {/* Strategy Breakdown */}
              {Object.keys(strategyBreakdown).length > 0 && (
                <Card className="bg-card border-border shadow-none p-6">
                  <h4 className="font-semibold mb-4 flex items-center gap-2">
                    <PieChart className="w-5 h-5 text-indigo-500" />
                    Performance by Strategy
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(strategyBreakdown)
                      .sort((a, b) => b[1].pnl - a[1].pnl)
                      .map(([strategy, stats]) => {
                        const winRate = (stats.wins + stats.losses) > 0 ? (stats.wins / (stats.wins + stats.losses)) * 100 : 0
                        return (
                          <div key={strategy} className="flex items-center justify-between bg-muted rounded-lg p-3">
                            <div className="flex items-center gap-3">
                              <div className={cn("w-2 h-2 rounded-full", stats.pnl >= 0 ? "bg-green-500" : "bg-red-500")} />
                              <div>
                                <p className="font-medium text-sm">{strategy}</p>
                                <p className="text-xs text-muted-foreground">
                                  {stats.trades} trades | {winRate.toFixed(0)}% win rate | Cost: ${stats.cost.toFixed(2)}
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className={cn("font-mono font-medium", stats.pnl >= 0 ? "text-green-400" : "text-red-400")}>
                                {stats.pnl >= 0 ? '+' : ''}${stats.pnl.toFixed(2)}
                              </p>
                              <p className="text-xs text-muted-foreground">{stats.wins}W / {stats.losses}L</p>
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </Card>
              )}

              {/* Best/Worst Trades */}
              {summary && (summary.best_trade !== 0 || summary.worst_trade !== 0) && (
                <div className="grid grid-cols-2 gap-4">
                  <Card className="bg-card border-border shadow-none p-4">
                    <p className="text-xs text-muted-foreground mb-1">Best Trade</p>
                    <p className="text-xl font-mono font-bold text-green-400">
                      +${summary.best_trade.toFixed(2)}
                    </p>
                  </Card>
                  <Card className="bg-card border-border shadow-none p-4">
                    <p className="text-xs text-muted-foreground mb-1">Worst Trade</p>
                    <p className="text-xl font-mono font-bold text-red-400">
                      ${summary.worst_trade.toFixed(2)}
                    </p>
                  </Card>
                </div>
              )}
            </div>
          )}

          {detailTab === 'holdings' && (
            <div className="space-y-4">
              {/* Holdings Summary */}
              <div className="grid grid-cols-3 gap-4">
                <Card className="bg-card border-border shadow-none p-4">
                  <p className="text-xs text-muted-foreground mb-1">Open Positions</p>
                  <p className="text-2xl font-mono font-bold">{positions.length}</p>
                </Card>
                <Card className="bg-card border-border shadow-none p-4">
                  <p className="text-xs text-muted-foreground mb-1">Book Value (Cost Basis)</p>
                  <p className="text-2xl font-mono font-bold">
                    ${positions.reduce((s: number, p: SimulationPosition) => s + p.entry_cost, 0).toFixed(2)}
                  </p>
                </Card>
                <Card className="bg-card border-border shadow-none p-4">
                  <p className="text-xs text-muted-foreground mb-1">Market Value</p>
                  <p className="text-2xl font-mono font-bold">
                    ${positions.reduce((s: number, p: SimulationPosition) => s + p.quantity * (p.current_price || p.entry_price), 0).toFixed(2)}
                  </p>
                </Card>
              </div>

              {/* Positions Table */}
              {positions.length === 0 ? (
                <Card className="text-center py-8 bg-card border-border shadow-none">
                  <Briefcase className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
                  <p className="text-muted-foreground">No open positions</p>
                  <p className="text-sm text-muted-foreground">Execute opportunities to open positions</p>
                </Card>
              ) : (
                <Card className="bg-card border-border shadow-none overflow-hidden">
                  <CardContent className="p-0">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border text-muted-foreground text-xs">
                          <th className="text-left px-4 py-3">Market</th>
                          <th className="text-center px-3 py-3">Side</th>
                          <th className="text-right px-3 py-3">Qty</th>
                          <th className="text-right px-3 py-3">Entry Price</th>
                          <th className="text-right px-3 py-3">Curr Price</th>
                          <th className="text-right px-3 py-3">Cost Basis</th>
                          <th className="text-right px-3 py-3">Mkt Value</th>
                          <th className="text-right px-4 py-3">Unrealized P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {positions.map((pos: SimulationPosition) => {
                          const currPrice = pos.current_price || pos.entry_price
                          const mktValue = pos.quantity * currPrice
                          const pnlPct = pos.entry_cost > 0 ? (pos.unrealized_pnl / pos.entry_cost) * 100 : 0
                          return (
                            <tr key={pos.id} className="border-b border-border/50 hover:bg-muted transition-colors">
                              <td className="px-4 py-3">
                                <p className="font-medium text-sm line-clamp-1">{pos.market_question}</p>
                                <p className="text-xs text-muted-foreground">{pos.opened_at ? new Date(pos.opened_at).toLocaleDateString() : ''}</p>
                              </td>
                              <td className="text-center px-3 py-3">
                                <Badge className={cn(
                                  "rounded text-xs font-medium border-0",
                                  pos.side === 'yes' ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                                )}>
                                  {pos.side.toUpperCase()}
                                </Badge>
                              </td>
                              <td className="text-right px-3 py-3 font-mono">{pos.quantity.toFixed(2)}</td>
                              <td className="text-right px-3 py-3 font-mono">${pos.entry_price.toFixed(4)}</td>
                              <td className="text-right px-3 py-3 font-mono">${currPrice.toFixed(4)}</td>
                              <td className="text-right px-3 py-3 font-mono">${pos.entry_cost.toFixed(2)}</td>
                              <td className="text-right px-3 py-3 font-mono">${mktValue.toFixed(2)}</td>
                              <td className="text-right px-4 py-3">
                                <span className={cn("font-mono font-medium", pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                                  {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                                </span>
                                <span className={cn("text-xs ml-1", pnlPct >= 0 ? "text-green-400/70" : "text-red-400/70")}>
                                  ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%)
                                </span>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                      <tfoot>
                        <tr className="border-t border-border font-medium">
                          <td className="px-4 py-3 text-muted-foreground" colSpan={5}>Totals</td>
                          <td className="text-right px-3 py-3 font-mono">
                            ${positions.reduce((s: number, p: SimulationPosition) => s + p.entry_cost, 0).toFixed(2)}
                          </td>
                          <td className="text-right px-3 py-3 font-mono">
                            ${positions.reduce((s: number, p: SimulationPosition) => s + p.quantity * (p.current_price || p.entry_price), 0).toFixed(2)}
                          </td>
                          <td className="text-right px-4 py-3">
                            <span className={cn(
                              "font-mono font-medium",
                              positions.reduce((s: number, p: SimulationPosition) => s + p.unrealized_pnl, 0) >= 0 ? "text-green-400" : "text-red-400"
                            )}>
                              {positions.reduce((s: number, p: SimulationPosition) => s + p.unrealized_pnl, 0) >= 0 ? '+' : ''}
                              ${positions.reduce((s: number, p: SimulationPosition) => s + p.unrealized_pnl, 0).toFixed(2)}
                            </span>
                          </td>
                        </tr>
                      </tfoot>
                    </table>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {detailTab === 'trades' && (
            <div className="space-y-4">
              {/* Trade Filters */}
              <div className="flex items-center gap-3">
                <select
                  value={tradeFilter}
                  onChange={(e) => setTradeFilter(e.target.value)}
                  className="bg-muted border border-border rounded-lg px-3 py-2 text-sm"
                >
                  <option value="all">All Trades</option>
                  <option value="open">Open</option>
                  <option value="resolved_win">Wins</option>
                  <option value="resolved_loss">Losses</option>
                  <option value="pending">Pending</option>
                </select>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  Sort:
                  {(['date', 'pnl', 'cost'] as const).map(s => (
                    <Button
                      key={s}
                      variant="ghost"
                      onClick={() => {
                        if (tradeSort === s) setTradeSortDir(d => d === 'desc' ? 'asc' : 'desc')
                        else { setTradeSort(s); setTradeSortDir('desc') }
                      }}
                      className={cn(
                        "px-2 py-1 h-auto rounded text-xs",
                        tradeSort === s ? "bg-blue-500/20 text-blue-400" : "hover:bg-muted"
                      )}
                    >
                      {s.charAt(0).toUpperCase() + s.slice(1)}
                      {tradeSort === s && (tradeSortDir === 'desc' ? <ChevronDown className="w-3 h-3 inline ml-0.5" /> : <ChevronUp className="w-3 h-3 inline ml-0.5" />)}
                    </Button>
                  ))}
                </div>
                <span className="text-xs text-muted-foreground ml-auto">{processedTrades.length} trades</span>
              </div>

              {/* Trades Table */}
              {processedTrades.length === 0 ? (
                <Card className="text-center py-8 bg-card border-border shadow-none">
                  <p className="text-muted-foreground">No trades found</p>
                </Card>
              ) : (
                <Card className="bg-card border-border shadow-none overflow-hidden">
                  <CardContent className="p-0">
                    <div className="max-h-[600px] overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-card">
                          <tr className="border-b border-border text-muted-foreground text-xs">
                            <th className="text-left px-4 py-3">Date</th>
                            <th className="text-left px-3 py-3">Strategy</th>
                            <th className="text-right px-3 py-3">Cost</th>
                            <th className="text-right px-3 py-3">Expected</th>
                            <th className="text-right px-3 py-3">Slippage</th>
                            <th className="text-center px-3 py-3">Status</th>
                            <th className="text-right px-3 py-3">Payout</th>
                            <th className="text-right px-4 py-3">P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {processedTrades.map((trade) => (
                            <tr key={trade.id} className="border-b border-border/50 hover:bg-muted transition-colors">
                              <td className="px-4 py-3">
                                <p className="font-mono text-xs">{new Date(trade.executed_at).toLocaleDateString()}</p>
                                <p className="font-mono text-xs text-muted-foreground">{new Date(trade.executed_at).toLocaleTimeString()}</p>
                              </td>
                              <td className="px-3 py-3">
                                <p className="font-medium">{trade.strategy_type}</p>
                                {trade.copied_from && <p className="text-xs text-muted-foreground">Copied</p>}
                              </td>
                              <td className="text-right px-3 py-3 font-mono">${trade.total_cost.toFixed(2)}</td>
                              <td className="text-right px-3 py-3 font-mono text-muted-foreground">${trade.expected_profit.toFixed(2)}</td>
                              <td className="text-right px-3 py-3 font-mono text-muted-foreground">${trade.slippage.toFixed(4)}</td>
                              <td className="text-center px-3 py-3">
                                <StatusBadge status={trade.status} />
                              </td>
                              <td className="text-right px-3 py-3 font-mono">
                                {trade.actual_payout != null ? `$${trade.actual_payout.toFixed(2)}` : '-'}
                              </td>
                              <td className="text-right px-4 py-3">
                                {trade.actual_pnl != null ? (
                                  <span className={cn("font-mono font-medium", trade.actual_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                                    {trade.actual_pnl >= 0 ? '+' : ''}${trade.actual_pnl.toFixed(2)}
                                  </span>
                                ) : (
                                  <span className="text-muted-foreground font-mono">-</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {detailTab === 'execute' && (
            <Card className="bg-card border-border shadow-none p-4">
              <h4 className="font-medium mb-4">Execute Opportunity</h4>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {opportunities.length === 0 ? (
                  <p className="text-muted-foreground text-center py-4">No opportunities available. Run a scan first.</p>
                ) : (
                  opportunities.slice(0, 10).map((opp) => (
                    <div
                      key={opp.id}
                      className="flex items-center justify-between bg-muted rounded-lg p-3 hover:bg-accent transition-colors"
                    >
                      <div className="flex-1">
                        <p className="text-sm font-medium line-clamp-1">{opp.title}</p>
                        <p className="text-xs text-muted-foreground">
                          ROI: <span className="text-green-400">{opp.roi_percent.toFixed(2)}%</span> | Cost: ${opp.total_cost.toFixed(4)} | Risk: {opp.risk_score.toFixed(2)}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        onClick={() => setExecutingOpportunity(opp)}
                        className="gap-1 bg-green-500/20 text-green-400 hover:bg-green-500/30 ml-3 h-auto px-3 py-1.5 text-sm rounded-lg"
                      >
                        <Play className="w-3 h-3" />
                        Execute
                      </Button>
                    </div>
                  ))
                )}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Trade Execution Modal */}
      {executingOpportunity && (
        <TradeExecutionModal
          opportunity={executingOpportunity}
          onClose={() => {
            setExecutingOpportunity(null)
            queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
            queryClient.invalidateQueries({ queryKey: ['account-trades'] })
            queryClient.invalidateQueries({ queryKey: ['account-positions'] })
            queryClient.invalidateQueries({ queryKey: ['account-equity'] })
          }}
        />
      )}

      {/* Delete Confirmation Modal */}
      {accountToDelete && (
        <div className="fixed inset-0 bg-background/80 flex items-center justify-center z-50">
          <Card className="bg-muted border-border shadow-none p-6 max-w-md mx-4">
            <h3 className="text-lg font-medium mb-2">Delete Account</h3>
            <p className="text-muted-foreground mb-4">
              Are you sure you want to delete "{accountToDelete.name}"? This will also delete all trades and positions associated with this account. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <Button
                variant="secondary"
                onClick={() => setAccountToDelete(null)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={() => deleteMutation.mutate(accountToDelete.id)}
                disabled={deleteMutation.isPending}
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

// ==================== Sub-Components ====================

function AccountCard({
  account,
  isSelected,
  onSelect,
  onDelete
}: {
  account: SimulationAccount
  isSelected: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  const roiColor = account.roi_percent >= 0 ? 'text-green-400' : 'text-red-400'
  const totalPnl = account.total_pnl + (account.unrealized_pnl || 0)

  return (
    <Card
      onClick={onSelect}
      className={cn(
        "bg-card cursor-pointer transition-colors shadow-none p-4",
        isSelected ? "border-blue-500" : "border-border hover:border-border"
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium">{account.name}</h3>
        <div className="flex items-center gap-2">
          {isSelected && <Activity className="w-4 h-4 text-blue-500" />}
          <button
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            className="p-1 text-muted-foreground hover:text-red-400 rounded transition-colors"
            title="Delete account"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-muted-foreground">Capital</p>
          <p className="font-mono">${(account.current_capital ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
        </div>
        <div>
          <p className="text-muted-foreground">Total P&L</p>
          <p className={cn("font-mono", totalPnl >= 0 ? 'text-green-400' : 'text-red-400')}>
            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-muted-foreground">ROI</p>
          <p className={cn("font-mono", roiColor)}>
            {(account.roi_percent ?? 0) >= 0 ? '+' : ''}{(account.roi_percent ?? 0).toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-muted-foreground">Win Rate</p>
          <p className="font-mono">{(account.win_rate ?? 0).toFixed(1)}%</p>
        </div>
      </div>

      {/* Progress bar for capital */}
      <div className="mt-3 pt-3 border-t border-border">
        <div className="flex justify-between text-xs text-muted-foreground mb-1">
          <span>{account.total_trades ?? 0} trades</span>
          <span>{account.open_positions ?? 0} open</span>
        </div>
        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all", account.roi_percent >= 0 ? "bg-green-500" : "bg-red-500")}
            style={{ width: `${Math.min(Math.max(50 + account.roi_percent / 2, 5), 100)}%` }}
          />
        </div>
      </div>
    </Card>
  )
}

function MiniStat({
  label,
  value,
  icon,
  subtitle,
  valueColor = 'text-foreground'
}: {
  label: string
  value: string
  icon: React.ReactNode
  subtitle?: string
  valueColor?: string
}) {
  return (
    <Card className="bg-card border-border shadow-none p-3">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
      <p className={cn("text-lg font-semibold font-mono", valueColor)}>{value}</p>
      {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
    </Card>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    open: 'bg-blue-500/20 text-blue-400',
    resolved_win: 'bg-green-500/20 text-green-400',
    resolved_loss: 'bg-red-500/20 text-red-400',
    pending: 'bg-yellow-500/20 text-yellow-400',
    cancelled: 'bg-gray-500/20 text-muted-foreground',
    failed: 'bg-red-500/20 text-red-400'
  }
  return (
    <Badge className={cn("px-2 py-0.5 rounded text-xs font-medium border-0", colors[status] || 'bg-gray-500/20 text-muted-foreground')}>
      {status.replace('_', ' ').toUpperCase()}
    </Badge>
  )
}

function EquityChart({
  points,
  initialCapital
}: {
  points: { date: string; equity: number; cumulative_pnl: number }[]
  initialCapital: number
}) {
  const chartHeight = 200
  const chartWidth = 100

  const equities = points.map(p => p.equity)
  const maxVal = Math.max(...equities)
  const minVal = Math.min(...equities)
  const range = maxVal - minVal || 1

  const getY = (val: number) => chartHeight - ((val - minVal) / range) * (chartHeight - 20) - 10
  const getX = (i: number) => (i / (points.length - 1 || 1)) * chartWidth

  // Create area fill path
  const linePath = points.map((p, i) => {
    const x = getX(i)
    const y = getY(p.equity)
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ')

  const areaPath = `${linePath} L ${chartWidth} ${chartHeight} L 0 ${chartHeight} Z`

  // Zero line (initial capital)
  const zeroY = getY(initialCapital)

  // Determine if overall profitable
  const lastEquity = points[points.length - 1]?.equity || initialCapital
  const isProfitable = lastEquity >= initialCapital

  // Color based on P&L
  const lineColor = isProfitable ? '#22c55e' : '#ef4444'
  const fillColor = isProfitable ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'

  return (
    <div className="relative h-full">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-full" preserveAspectRatio="none">
        {/* Area fill */}
        <path d={areaPath} fill={fillColor} />

        {/* Initial capital reference line */}
        <line
          x1="0" y1={zeroY}
          x2={chartWidth} y2={zeroY}
          stroke="#374151"
          strokeWidth="0.3"
          strokeDasharray="1,1"
        />

        {/* Equity line */}
        <path
          d={linePath}
          fill="none"
          stroke={lineColor}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      {/* Labels */}
      <div className="absolute top-1 left-2 text-xs text-muted-foreground">
        ${maxVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </div>
      <div className="absolute bottom-1 left-2 text-xs text-muted-foreground">
        ${minVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </div>
      <div className="absolute bottom-1 right-2 flex items-center gap-2 text-xs">
        <span className={isProfitable ? 'text-green-400' : 'text-red-400'}>
          ${lastEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
        <span className="text-muted-foreground">
          ({isProfitable ? '+' : ''}{((lastEquity - initialCapital) / initialCapital * 100).toFixed(2)}%)
        </span>
      </div>
      <div className="absolute top-1 right-2 text-xs text-muted-foreground">
        {points.length > 0 && new Date(points[0].date).toLocaleDateString()} - {points.length > 0 && new Date(points[points.length - 1].date).toLocaleDateString()}
      </div>
    </div>
  )
}
