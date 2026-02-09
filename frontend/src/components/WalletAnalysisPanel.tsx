import { useState, useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Search,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  ArrowUpRight,
  ArrowDownRight,
  Wallet,
  BarChart3,
  History,
  Briefcase,
  DollarSign,
  Activity,
  ArrowRight,
  CheckCircle2,
  Clock,
  User,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Zap,
  Eye
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import {
  getWalletTradesAnalysis,
  getWalletPositionsAnalysis,
  getWalletSummary,
  getWalletWinRate,
  analyzeWalletPnL,
  getWalletProfile,
  analyzeWallet,
  WalletTrade,
  WalletPosition,
  WalletSummary,
  WalletWinRate,
  WalletPnL,
  WalletAnalysis,
} from '../services/api'

interface WalletAnalysisPanelProps {
  initialWallet?: string | null
  initialUsername?: string | null
  onWalletAnalyzed?: () => void
}

// Large Sparkline Component for hero section
function LargeSparkline({ data, color = '#22c55e', height = 80 }: { data: number[]; color?: string; height?: number }) {
  if (!data || data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const padding = 4
  const viewBoxWidth = 400
  const viewBoxHeight = height

  const points = data.map((value, index) => {
    const x = padding + (index / (data.length - 1)) * (viewBoxWidth - padding * 2)
    const y = viewBoxHeight - padding - ((value - min) / range) * (viewBoxHeight - padding * 2)
    return `${x},${y}`
  }).join(' ')

  const areaPath = `M ${padding},${viewBoxHeight - padding} L ${points} L ${viewBoxWidth - padding},${viewBoxHeight - padding} Z`

  const lastY = viewBoxHeight - padding - ((data[data.length - 1] - min) / range) * (viewBoxHeight - padding * 2)

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
      preserveAspectRatio="none"
      className="overflow-visible"
    >
      <defs>
        <linearGradient id={`large-sparkline-gradient-${color.replace('#', '')}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.4" />
          <stop offset="100%" stopColor={color} stopOpacity="0.05" />
        </linearGradient>
      </defs>
      <path
        d={areaPath}
        fill={`url(#large-sparkline-gradient-${color.replace('#', '')})`}
      />
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle
        cx={viewBoxWidth - padding}
        cy={lastY}
        r="6"
        fill={color}
        stroke="hsl(var(--card))"
        strokeWidth="2"
      />
    </svg>
  )
}

// Circular Progress Component
function CircularProgress({ percentage, size = 80, strokeWidth = 6, color = '#22c55e' }: {
  percentage: number
  size?: number
  strokeWidth?: number
  color?: string
}) {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (Math.min(100, Math.max(0, percentage)) / 100) * circumference

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor" opacity={0.1}
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold" style={{ color }}>
          {percentage.toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

type TimePeriod = 'DAY' | 'WEEK' | 'MONTH' | 'ALL'

const TIME_PERIOD_OPTIONS: { value: TimePeriod; label: string }[] = [
  { value: 'DAY', label: '24H' },
  { value: 'WEEK', label: '7D' },
  { value: 'MONTH', label: '30D' },
  { value: 'ALL', label: 'All Time' },
]

export default function WalletAnalysisPanel({ initialWallet, initialUsername, onWalletAnalyzed }: WalletAnalysisPanelProps) {
  const [searchAddress, setSearchAddress] = useState('')
  const [activeWallet, setActiveWallet] = useState<string | null>(null)
  const [passedUsername, setPassedUsername] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'summary' | 'trades' | 'positions' | 'anomaly'>('summary')
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('ALL')

  // Auto-analyze when initialWallet changes
  useEffect(() => {
    if (initialWallet && initialWallet !== activeWallet) {
      setSearchAddress(initialWallet)
      setActiveWallet(initialWallet.trim())
      setPassedUsername(initialUsername || null)
      setActiveTab('summary')
      if (onWalletAnalyzed) {
        onWalletAnalyzed()
      }
    }
  }, [initialWallet, initialUsername, onWalletAnalyzed, activeWallet])

  // Use the discover API for PnL data (same as wallet tracker)
  const pnlQuery = useQuery({
    queryKey: ['wallet-pnl-discover', activeWallet, timePeriod],
    queryFn: () => analyzeWalletPnL(activeWallet!, timePeriod),
    enabled: !!activeWallet,
  })

  // Also get summary for additional details
  const summaryQuery = useQuery({
    queryKey: ['wallet-summary', activeWallet],
    queryFn: () => getWalletSummary(activeWallet!),
    enabled: !!activeWallet,
  })

  const winRateQuery = useQuery({
    queryKey: ['wallet-win-rate', activeWallet, timePeriod],
    queryFn: () => getWalletWinRate(activeWallet!, timePeriod),
    enabled: !!activeWallet,
  })

  const tradesQuery = useQuery({
    queryKey: ['wallet-trades', activeWallet],
    queryFn: () => getWalletTradesAnalysis(activeWallet!, 200),
    enabled: !!activeWallet,
  })

  const positionsQuery = useQuery({
    queryKey: ['wallet-positions', activeWallet],
    queryFn: () => getWalletPositionsAnalysis(activeWallet!),
    enabled: !!activeWallet,
  })

  // Auto-run anomaly detection when wallet is opened
  const anomalyQuery = useQuery({
    queryKey: ['wallet-anomaly', activeWallet],
    queryFn: () => analyzeWallet(activeWallet!),
    enabled: !!activeWallet,
    staleTime: 300000, // Cache for 5 minutes
    retry: 1,
  })

  // Fetch user profile (username) directly from Polymarket
  const profileQuery = useQuery({
    queryKey: ['wallet-profile', activeWallet],
    queryFn: () => getWalletProfile(activeWallet!),
    enabled: !!activeWallet,
    staleTime: 300000, // Cache for 5 minutes
  })

  const username = passedUsername || profileQuery.data?.username || null

  const handleAnalyze = () => {
    if (searchAddress.trim()) {
      setActiveWallet(searchAddress.trim())
      setPassedUsername(null)
      setActiveTab('summary')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAnalyze()
    }
  }

  const isLoading = pnlQuery.isLoading || summaryQuery.isLoading || winRateQuery.isLoading

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Wallet Analysis</h2>
          <p className="text-sm text-muted-foreground">
            Deep dive into any trader's performance and strategy
          </p>
        </div>
      </div>

      {/* Search Card */}
      <Card className="border-border">
        <CardContent className="p-6">
          <div className="flex flex-col gap-3">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  type="text"
                  value={searchAddress}
                  onChange={(e) => setSearchAddress(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter wallet address (0x...) or username"
                  className="pl-10 bg-card border-border font-mono text-sm"
                />
              </div>
              <Button
                onClick={handleAnalyze}
                disabled={!searchAddress.trim()}
                className="flex items-center gap-2 bg-cyan-500 hover:bg-cyan-600 text-foreground"
              >
                <Search className="w-4 h-4" />
                Analyze
              </Button>
            </div>
            {/* Time Period Filter */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Time Period:</span>
              <div className="flex gap-1">
                {TIME_PERIOD_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setTimePeriod(option.value)}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
                      timePeriod === option.value
                        ? "bg-primary/20 text-primary"
                        : "bg-muted/50 text-muted-foreground hover:bg-muted"
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {activeWallet && (
        <div className="space-y-6">
          {/* Hero Profile Card */}
          <WalletHeroCard
            address={activeWallet}
            username={username}
            pnlData={pnlQuery.data}
            summary={summaryQuery.data}
            winRate={winRateQuery.data}
            trades={tradesQuery.data?.trades || []}
            isLoading={isLoading}
            timePeriod={timePeriod}
            anomalyData={anomalyQuery.data}
            onRefresh={() => {
              pnlQuery.refetch()
              summaryQuery.refetch()
              winRateQuery.refetch()
              tradesQuery.refetch()
              positionsQuery.refetch()
              anomalyQuery.refetch()
            }}
          />

          {/* Tab Navigation */}
          <div className="flex items-center gap-2">
            {[
              { id: 'summary' as const, label: 'Overview', icon: BarChart3 },
              { id: 'trades' as const, label: 'Trade History', icon: History },
              { id: 'positions' as const, label: 'Open Positions', icon: Briefcase },
              { id: 'anomaly' as const, label: 'Risk Analysis', icon: ShieldAlert },
            ].map((tab) => (
              <Button
                key={tab.id}
                variant="outline"
                size="sm"
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex items-center gap-2",
                  activeTab === tab.id
                    ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30 hover:text-cyan-400"
                    : "bg-card text-muted-foreground hover:text-foreground border-border"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </Button>
            ))}
          </div>

          {/* Tab Content */}
          <Card className="border-border">
            <CardContent className="p-6">
              {activeTab === 'summary' && (
                <SummaryTab
                  pnlData={pnlQuery.data}
                  summary={summaryQuery.data}
                  winRate={winRateQuery.data}
                  isLoading={summaryQuery.isLoading}
                />
              )}
              {activeTab === 'trades' && (
                <TradesTab data={tradesQuery.data} isLoading={tradesQuery.isLoading} />
              )}
              {activeTab === 'positions' && (
                <PositionsTab data={positionsQuery.data} isLoading={positionsQuery.isLoading} />
              )}
              {activeTab === 'anomaly' && (
                <AnomalyTab data={anomalyQuery.data} isLoading={anomalyQuery.isLoading} />
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Empty State */}
      {!activeWallet && (
        <Card className="border-border">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Wallet className="w-12 h-12 text-muted-foreground/30 mb-4" />
            <p className="text-muted-foreground">No wallet selected</p>
            <p className="text-sm text-muted-foreground/70 mt-1">
              Enter a wallet address above to analyze trading performance, win rates, and strategy patterns
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function WalletHeroCard({
  address,
  username,
  pnlData,
  summary,
  winRate,
  trades,
  isLoading,
  timePeriod,
  anomalyData,
  onRefresh
}: {
  address: string
  username: string | null
  pnlData?: WalletPnL
  summary?: WalletSummary
  winRate?: WalletWinRate
  trades: WalletTrade[]
  isLoading: boolean
  timePeriod: TimePeriod
  anomalyData?: WalletAnalysis
  onRefresh: () => void
}) {
  const timePeriodLabel = TIME_PERIOD_OPTIONS.find(o => o.value === timePeriod)?.label || 'All Time'
  // Generate sparkline data from trades (cumulative value over time)
  const sparklineData = useMemo(() => {
    if (!trades || trades.length === 0) return []

    const sortedTrades = [...trades].sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )

    let cumulative = 0
    const pnlData: number[] = []

    sortedTrades.forEach(trade => {
      if (trade.side === 'SELL') {
        cumulative += trade.cost
      } else {
        cumulative -= trade.cost
      }
      pnlData.push(cumulative)
    })

    // Normalize to show trend
    return pnlData.length > 1 ? pnlData : []
  }, [trades])

  // Use pnlData (from discover API) as primary source, fallback to summary
  const totalPnl = pnlData?.total_pnl ?? summary?.summary.total_pnl ?? 0
  const roiPercent = pnlData?.roi_percent ?? summary?.summary.roi_percent ?? 0
  const totalInvested = pnlData?.total_invested ?? summary?.summary.total_invested ?? 0
  const totalReturned = pnlData?.total_returned ?? summary?.summary.total_returned ?? 0
  const totalTrades = pnlData?.total_trades ?? summary?.summary.total_trades ?? 0

  const isProfitable = totalPnl >= 0
  const winRateValue = winRate?.win_rate ?? 0
  const winRateColor = winRateValue >= 70 ? '#22c55e' : winRateValue >= 50 ? '#eab308' : '#ef4444'
  const pnlColor = isProfitable ? '#22c55e' : '#ef4444'

  // Calculate volume
  const volume = totalInvested + totalReturned

  if (isLoading) {
    return (
      <Card className="border-border">
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border">
      <CardContent className="p-6">
        {/* Header Row */}
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="relative">
              <div className="p-3 bg-cyan-500/10 rounded-lg">
                {username ? (
                  <User className="w-6 h-6 text-cyan-500" />
                ) : (
                  <Wallet className="w-6 h-6 text-cyan-500" />
                )}
              </div>
              {isProfitable && (
                <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-green-500 flex items-center justify-center border-2 border-card">
                  <TrendingUp className="w-2.5 h-2.5 text-foreground" />
                </div>
              )}
            </div>

            {/* Identity */}
            <div>
              {username ? (
                <>
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="text-lg font-bold text-foreground">{username}</h2>
                    <a
                      href={`https://polymarket.com/profile/${address}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
                      title="View on Polymarket"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                  <p className="text-xs text-muted-foreground font-mono">{`${address.slice(0, 6)}...${address.slice(-4)}`}</p>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="text-lg font-bold text-foreground font-mono">
                      {`${address.slice(0, 6)}...${address.slice(-4)}`}
                    </h2>
                    <a
                      href={`https://polymarket.com/profile/${address}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
                      title="View on Polymarket"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                  <p className="text-[10px] text-muted-foreground font-mono truncate max-w-[300px]">{address}</p>
                </>
              )}
            </div>
          </div>

          {/* Badges & Refresh */}
          <div className="flex items-center gap-2">
            {anomalyData && (
              <Badge variant="outline" className={cn(
                "text-xs",
                anomalyData.anomaly_score > 0.7
                  ? "bg-red-500/15 text-red-400 border-red-500/20"
                  : anomalyData.anomaly_score > 0.3
                  ? "bg-yellow-500/15 text-yellow-400 border-yellow-500/20"
                  : "bg-green-500/15 text-green-400 border-green-500/20"
              )}>
                {anomalyData.anomaly_score > 0.7 ? (
                  <ShieldAlert className="w-3 h-3 mr-1" />
                ) : anomalyData.anomaly_score > 0.3 ? (
                  <AlertTriangle className="w-3 h-3 mr-1" />
                ) : (
                  <ShieldCheck className="w-3 h-3 mr-1" />
                )}
                Risk: {(anomalyData.anomaly_score * 100).toFixed(0)}%
              </Badge>
            )}
            <Badge variant="outline" className="text-xs bg-muted text-muted-foreground border-border">
              {timePeriodLabel}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              className="flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Key Metrics Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {/* Total P&L - Primary metric */}
          <div className="col-span-2 bg-muted rounded-lg p-4 border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Total P&L</p>
                <p className={cn(
                  "text-2xl font-bold",
                  isProfitable ? "text-green-400" : "text-red-400"
                )}>
                  {isProfitable ? '+' : '-'}${Math.abs(totalPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  ROI: <span className={isProfitable ? "text-green-400" : "text-red-400"}>
                    {roiPercent >= 0 ? '+' : ''}{roiPercent.toFixed(1)}%
                  </span>
                </p>
              </div>
              <div className={cn(
                "p-3 rounded-lg",
                isProfitable ? "bg-green-500/10" : "bg-red-500/10"
              )}>
                {isProfitable ? (
                  <TrendingUp className="w-6 h-6 text-green-400" />
                ) : (
                  <TrendingDown className="w-6 h-6 text-red-400" />
                )}
              </div>
            </div>
          </div>

          {/* Volume */}
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Volume</p>
            <p className="text-xl font-bold text-foreground">
              ${volume.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {totalTrades} trades
            </p>
          </div>

          {/* Win Rate */}
          <div className="bg-muted rounded-lg p-4 border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Win Rate</p>
                <p className="text-xl font-bold" style={{ color: winRateColor }}>
                  {winRateValue.toFixed(1)}%
                </p>
                {winRate && (
                  <p className="text-xs text-muted-foreground mt-1">
                    <span className="text-green-400">{winRate.wins}W</span>
                    {' / '}
                    <span className="text-red-400">{winRate.losses}L</span>
                  </p>
                )}
              </div>
              <CircularProgress
                percentage={winRateValue}
                size={48}
                strokeWidth={4}
                color={winRateColor}
              />
            </div>
          </div>
        </div>

        {/* Sparkline Section - Full Width */}
        {sparklineData.length > 1 && (
          <div className="bg-muted rounded-lg p-4 border border-border">
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm text-muted-foreground">Performance Trend</p>
              <p className="text-xs text-muted-foreground">Last {sparklineData.length} trades</p>
            </div>
            <LargeSparkline
              data={sparklineData}
              color={pnlColor}
              height={100}
            />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function SummaryTab({
  pnlData,
  summary,
  winRate,
  isLoading
}: {
  pnlData?: WalletPnL
  summary?: WalletSummary
  winRate?: WalletWinRate
  isLoading: boolean
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  // Use pnlData as primary, fallback to summary
  const data = summary?.summary
  const realizedPnl = pnlData?.realized_pnl ?? data?.realized_pnl ?? 0
  const unrealizedPnl = pnlData?.unrealized_pnl ?? data?.unrealized_pnl ?? 0
  const totalPnl = pnlData?.total_pnl ?? data?.total_pnl ?? 0
  const roiPercent = pnlData?.roi_percent ?? data?.roi_percent ?? 0
  const totalInvested = pnlData?.total_invested ?? data?.total_invested ?? 0
  const totalReturned = pnlData?.total_returned ?? data?.total_returned ?? 0
  const positionValue = pnlData?.position_value ?? data?.position_value ?? 0
  const buys = data?.buys ?? 0
  const sells = data?.sells ?? 0
  const totalTrades = pnlData?.total_trades ?? data?.total_trades ?? 0

  if (!data && !pnlData) {
    return (
      <div className="text-center py-12">
        <Activity className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
        <p className="text-muted-foreground">No data available for this wallet</p>
      </div>
    )
  }

  const isProfitable = totalPnl >= 0

  return (
    <div className="space-y-6">
      {/* Performance Grid */}
      <div>
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">Performance Breakdown</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Realized PnL */}
          <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/5 border border-green-500/20 p-5">
            <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                <p className="text-sm text-muted-foreground">Realized P&L</p>
              </div>
              <p className={cn(
                "text-2xl font-bold",
                realizedPnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                {realizedPnl >= 0 ? '+' : ''}${realizedPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
              <p className="text-xs text-muted-foreground/70 mt-1">From closed positions</p>
            </div>
          </div>

          {/* Unrealized PnL */}
          <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border border-blue-500/20 p-5">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-blue-400" />
                <p className="text-sm text-muted-foreground">Unrealized P&L</p>
              </div>
              <p className={cn(
                "text-2xl font-bold",
                unrealizedPnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                {unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
              <p className="text-xs text-muted-foreground/70 mt-1">From open positions</p>
            </div>
          </div>

          {/* Total PnL */}
          <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/5 border border-purple-500/20 p-5">
            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-2">
                <DollarSign className="w-4 h-4 text-purple-400" />
                <p className="text-sm text-muted-foreground">Total P&L</p>
              </div>
              <p className={cn(
                "text-2xl font-bold",
                isProfitable ? "text-green-400" : "text-red-400"
              )}>
                {isProfitable ? '+' : '-'}${Math.abs(totalPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                {roiPercent >= 0 ? '+' : ''}{roiPercent.toFixed(1)}% ROI
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Investment Flow & Trading Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Investment Flow */}
        <div className="rounded-xl bg-muted border border-border p-5">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
              <ArrowRight className="w-4 h-4 text-blue-400" />
            </div>
            <h4 className="font-semibold text-foreground">Investment Flow</h4>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-400" />
                <span className="text-sm text-muted-foreground">Total Invested</span>
              </div>
              <span className="font-mono font-medium text-foreground">
                ${totalInvested.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-400" />
                <span className="text-sm text-muted-foreground">Total Returned</span>
              </div>
              <span className="font-mono font-medium text-foreground">
                ${totalReturned.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-400" />
                <span className="text-sm text-muted-foreground">Position Value</span>
              </div>
              <span className="font-mono font-medium text-foreground">
                ${positionValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <div className="pt-3 mt-3 border-t border-border">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground/80">Net Flow</span>
                <span className={cn(
                  "font-mono font-bold text-lg",
                  totalPnl >= 0 ? "text-green-400" : "text-red-400"
                )}>
                  {totalPnl >= 0 ? '+' : '-'}${Math.abs(totalPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Trading Activity */}
        <div className="rounded-xl bg-muted border border-border p-5">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
              <Activity className="w-4 h-4 text-purple-400" />
            </div>
            <h4 className="font-semibold text-foreground">Trading Activity</h4>
          </div>

          {/* Trade Counts */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center p-3 rounded-lg bg-green-500/10 border border-green-500/20">
              <p className="text-2xl font-bold text-green-400">{buys}</p>
              <p className="text-xs text-muted-foreground/70">Buys</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <p className="text-2xl font-bold text-red-400">{sells}</p>
              <p className="text-xs text-muted-foreground/70">Sells</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-gray-500/10 border border-gray-500/20">
              <p className="text-2xl font-bold text-foreground/80">{totalTrades}</p>
              <p className="text-xs text-muted-foreground/70">Total</p>
            </div>
          </div>

          {/* Win Rate Bar */}
          {winRate && (
            <div className="pt-3 border-t border-border">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Win Rate</span>
                <span className={cn(
                  "font-medium",
                  winRate.win_rate >= 70 ? "text-green-400" :
                  winRate.win_rate >= 50 ? "text-yellow-400" : "text-red-400"
                )}>
                  {winRate.win_rate.toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    winRate.win_rate >= 70 ? "bg-gradient-to-r from-green-500 to-emerald-400" :
                    winRate.win_rate >= 50 ? "bg-gradient-to-r from-yellow-500 to-amber-400" :
                    "bg-gradient-to-r from-red-500 to-rose-400"
                  )}
                  style={{ width: `${Math.min(100, winRate.win_rate)}%` }}
                />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-xs text-green-400">{winRate.wins} wins</span>
                <span className="text-xs text-red-400">{winRate.losses} losses</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function TradesTab({ data, isLoading }: { data?: { wallet: string; total: number; trades: WalletTrade[] }; isLoading: boolean }) {
  const [expandedTrades, setExpandedTrades] = useState<Set<string>>(new Set())

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!data || data.trades.length === 0) {
    return (
      <div className="text-center py-12">
        <History className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
        <p className="text-muted-foreground">No trades found for this wallet</p>
      </div>
    )
  }

  const toggleTrade = (id: string) => {
    const newExpanded = new Set(expandedTrades)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedTrades(newExpanded)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {data.trades.length} of {data.total} trades
        </p>
      </div>

      {/* Trade List */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
        {data.trades.map((trade) => (
          <TradeRow
            key={trade.id}
            trade={trade}
            isExpanded={expandedTrades.has(trade.id)}
            onToggle={() => toggleTrade(trade.id)}
          />
        ))}
      </div>
    </div>
  )
}

function TradeRow({ trade, isExpanded, onToggle }: { trade: WalletTrade; isExpanded: boolean; onToggle: () => void }) {
  const isBuy = trade.side === 'BUY'
  const timestamp = trade.timestamp ? new Date(trade.timestamp).toLocaleString() : 'Unknown'

  return (
    <div className={cn(
      "rounded-xl overflow-hidden transition-all",
      isExpanded ? "bg-muted" : "bg-muted/50 hover:bg-muted"
    )}>
      <div
        className="flex items-center justify-between p-4 cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex items-center gap-4">
          <div className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center",
            isBuy ? "bg-green-500/20" : "bg-red-500/20"
          )}>
            {isBuy ? (
              <ArrowUpRight className="w-5 h-5 text-green-400" />
            ) : (
              <ArrowDownRight className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={cn(
                "text-xs font-semibold px-2 py-0.5 rounded-full",
                isBuy ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
              )}>
                {trade.side}
              </span>
              <span className="text-sm font-medium text-foreground">{trade.outcome || 'Unknown'}</span>
            </div>
            <p className="text-xs text-muted-foreground/70 mt-1">{timestamp}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="font-mono font-semibold text-foreground">${trade.cost.toFixed(2)}</p>
            <p className="text-xs text-muted-foreground/70">
              {trade.size.toFixed(2)} @ ${trade.price.toFixed(4)}
            </p>
          </div>
          <div className={cn(
            "p-2 rounded-lg transition-colors",
            isExpanded ? "bg-purple-500/20" : "bg-muted/50"
          )}>
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-purple-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-muted-foreground/70" />
            )}
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-border/50">
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-muted/60 rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70 mb-1">Market</p>
              <p className="font-mono text-xs text-foreground/80 truncate" title={trade.market}>
                {trade.market.length > 30 ? trade.market.slice(0, 30) + '...' : trade.market}
              </p>
            </div>
            <div className="bg-muted/60 rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70 mb-1">Size</p>
              <p className="font-mono text-foreground">{trade.size.toFixed(4)}</p>
            </div>
            <div className="bg-muted/60 rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70 mb-1">Price</p>
              <p className="font-mono text-foreground">${trade.price.toFixed(4)}</p>
            </div>
            {trade.transaction_hash && (
              <div className="bg-muted/60 rounded-lg p-3">
                <p className="text-xs text-muted-foreground/70 mb-1">Transaction</p>
                <a
                  href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-blue-400 hover:text-blue-300"
                >
                  <span className="font-mono text-xs">{trade.transaction_hash.slice(0, 10)}...</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            )}
          </div>
          {/* Actions */}
          <div className="flex items-center gap-3 mt-4 pt-3 border-t border-border">
            {trade.market_slug && (
              <a
                href={`https://polymarket.com/event/${trade.market_slug}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
              >
                <ExternalLink className="w-3 h-3" />
                View on Polymarket
              </a>
            )}
            {trade.transaction_hash && (
              <a
                href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground/80"
              >
                <ExternalLink className="w-3 h-3" />
                Transaction
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function PositionsTab({ data, isLoading }: { data?: { wallet: string; total_positions: number; total_value: number; total_unrealized_pnl: number; positions: WalletPosition[] }; isLoading: boolean }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!data || data.positions.length === 0) {
    return (
      <div className="text-center py-12">
        <Briefcase className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
        <p className="text-muted-foreground">No open positions</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border border-blue-500/20 p-5">
          <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <p className="text-xs text-muted-foreground/70 uppercase tracking-wider mb-1">Total Position Value</p>
            <p className="text-2xl font-bold text-foreground">${data.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            <p className="text-xs text-muted-foreground/70 mt-1">{data.total_positions} open positions</p>
          </div>
        </div>
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/5 border border-purple-500/20 p-5">
          <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <p className="text-xs text-muted-foreground/70 uppercase tracking-wider mb-1">Unrealized P&L</p>
            <p className={cn(
              "text-2xl font-bold",
              data.total_unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {data.total_unrealized_pnl >= 0 ? '+' : ''}${data.total_unrealized_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
        </div>
      </div>

      {/* Positions List */}
      <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
        {data.positions.map((position, idx) => (
          <PositionRow key={idx} position={position} />
        ))}
      </div>
    </div>
  )
}

function AnomalyTab({ data, isLoading }: { data?: WalletAnalysis; isLoading: boolean }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <ShieldAlert className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
        <p className="text-muted-foreground">No analysis data available</p>
      </div>
    )
  }

  const scoreColor = data.anomaly_score > 0.7 ? 'text-red-400' :
                     data.anomaly_score > 0.3 ? 'text-yellow-400' : 'text-green-400'
  const scoreBg = data.anomaly_score > 0.7 ? 'from-red-500/10 to-red-500/5 border-red-500/20' :
                  data.anomaly_score > 0.3 ? 'from-yellow-500/10 to-yellow-500/5 border-yellow-500/20' :
                  'from-green-500/10 to-green-500/5 border-green-500/20'
  const scoreLabel = data.anomaly_score > 0.7 ? 'High Risk' :
                     data.anomaly_score > 0.3 ? 'Moderate Risk' : 'Low Risk'

  const severityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      default: return 'bg-gray-500/20 text-muted-foreground border-gray-500/30'
    }
  }

  return (
    <div className="space-y-6">
      {/* Score & Recommendation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Anomaly Score */}
        <div className={cn("relative overflow-hidden rounded-xl bg-gradient-to-br border p-5", scoreBg)}>
          <div className="relative">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  {data.anomaly_score > 0.7 ? (
                    <ShieldAlert className="w-5 h-5 text-red-400" />
                  ) : data.anomaly_score > 0.3 ? (
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                  ) : (
                    <ShieldCheck className="w-5 h-5 text-green-400" />
                  )}
                  <p className="text-sm text-muted-foreground">Anomaly Score</p>
                </div>
                <p className={cn("text-3xl font-bold", scoreColor)}>
                  {(data.anomaly_score * 100).toFixed(0)}%
                </p>
                <p className={cn("text-sm font-medium mt-1", scoreColor)}>{scoreLabel}</p>
              </div>
              <CircularProgress
                percentage={data.anomaly_score * 100}
                size={72}
                strokeWidth={5}
                color={data.anomaly_score > 0.7 ? '#ef4444' : data.anomaly_score > 0.3 ? '#eab308' : '#22c55e'}
              />
            </div>
          </div>
        </div>

        {/* Recommendation */}
        <div className="rounded-xl bg-muted border border-border p-5">
          <div className="flex items-center gap-2 mb-3">
            <Eye className="w-5 h-5 text-purple-400" />
            <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Recommendation</p>
          </div>
          <p className="text-foreground leading-relaxed">{data.recommendation}</p>
          <div className="mt-3 flex items-center gap-2">
            <span className={cn(
              "px-2.5 py-1 rounded-full text-xs font-medium border",
              data.is_profitable_pattern
                ? "bg-green-500/20 text-green-400 border-green-500/30"
                : "bg-gray-500/20 text-muted-foreground border-gray-500/30"
            )}>
              {data.is_profitable_pattern ? 'Profitable Pattern' : 'Not Profitable'}
            </span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div>
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">Trading Profile</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground/70 mb-1">Total Trades</p>
            <p className="text-xl font-bold text-foreground">{data.stats.total_trades}</p>
          </div>
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground/70 mb-1">Win Rate</p>
            <p className="text-xl font-bold text-foreground">{(data.stats.win_rate * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground/70 mb-1">Avg ROI</p>
            <p className={cn("text-xl font-bold", data.stats.avg_roi >= 0 ? "text-green-400" : "text-red-400")}>
              {data.stats.avg_roi >= 0 ? '+' : ''}{data.stats.avg_roi.toFixed(1)}%
            </p>
          </div>
          <div className="bg-muted rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground/70 mb-1">Markets Traded</p>
            <p className="text-xl font-bold text-foreground">{data.stats.markets_traded ?? '-'}</p>
          </div>
        </div>
      </div>

      {/* Strategies Detected */}
      {data.strategies_detected.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">Detected Strategies</h3>
          <div className="flex flex-wrap gap-2">
            {data.strategies_detected.map((strategy, idx) => (
              <span
                key={idx}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20 text-sm text-purple-300"
              >
                <Zap className="w-3.5 h-3.5" />
                {strategy}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Anomalies Found */}
      <div>
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
          Anomalies Detected ({data.anomalies.length})
        </h3>
        {data.anomalies.length === 0 ? (
          <div className="text-center py-8 rounded-xl bg-green-500/5 border border-green-500/20">
            <ShieldCheck className="w-10 h-10 text-green-400 mx-auto mb-3" />
            <p className="text-green-400 font-medium">No anomalies detected</p>
            <p className="text-xs text-muted-foreground/70 mt-1">This wallet shows normal trading patterns</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
            {data.anomalies.map((anomaly, idx) => (
              <div
                key={idx}
                className="rounded-xl bg-muted border border-border p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className={cn(
                      "w-4 h-4",
                      anomaly.severity === 'critical' ? 'text-red-400' :
                      anomaly.severity === 'high' ? 'text-orange-400' :
                      anomaly.severity === 'medium' ? 'text-yellow-400' : 'text-blue-400'
                    )} />
                    <span className="font-medium text-foreground text-sm">
                      {anomaly.type.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                    </span>
                  </div>
                  <span className={cn(
                    "px-2 py-0.5 rounded-full text-xs font-medium border",
                    severityColor(anomaly.severity)
                  )}>
                    {anomaly.severity}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-2">{anomaly.description}</p>
                {anomaly.evidence && Object.keys(anomaly.evidence).length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {Object.entries(anomaly.evidence).map(([key, value]) => (
                      <span key={key} className="text-xs bg-muted/70 rounded px-2 py-1 text-muted-foreground/70">
                        {key.replace(/_/g, ' ')}: <span className="text-foreground/80">
                          {typeof value === 'number' ? value.toFixed(2) : String(value)}
                        </span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function PositionRow({ position }: { position: WalletPosition }) {
  const isProfitable = position.unrealized_pnl >= 0
  const roiColor = position.roi_percent >= 20 ? 'text-green-400' :
                   position.roi_percent >= 0 ? 'text-emerald-400' :
                   position.roi_percent >= -20 ? 'text-yellow-400' : 'text-red-400'

  const displayTitle = position.title || position.market
  const isConditionId = !position.title && position.market.length > 40

  return (
    <div className="rounded-xl bg-muted border border-border p-5 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <p className="font-medium text-foreground truncate" title={displayTitle}>
            {position.title || (isConditionId ? `${position.market.slice(0, 20)}...` : position.market)}
          </p>
          <p className="text-xs text-muted-foreground/70 mt-1">
            {position.outcome || 'Unknown'}
          </p>
        </div>
        <div className={cn(
          "flex items-center gap-1.5 px-3 py-1.5 rounded-full ml-4",
          isProfitable ? "bg-green-500/20" : "bg-red-500/20"
        )}>
          {isProfitable ? (
            <TrendingUp className="w-4 h-4 text-green-400" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-400" />
          )}
          <span className={cn("font-semibold", roiColor)}>
            {position.roi_percent >= 0 ? '+' : ''}{position.roi_percent.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-muted/60 rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70 mb-1">Size</p>
          <p className="font-mono font-medium text-foreground">{position.size.toFixed(2)}</p>
        </div>
        <div className="bg-muted/60 rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70 mb-1">Avg Price</p>
          <p className="font-mono font-medium text-foreground">${position.avg_price.toFixed(4)}</p>
        </div>
        <div className="bg-muted/60 rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70 mb-1">Current Price</p>
          <p className="font-mono font-medium text-foreground">${position.current_price.toFixed(4)}</p>
        </div>
        <div className="bg-muted/60 rounded-lg p-3">
          <p className="text-xs text-muted-foreground/70 mb-1">Unrealized P&L</p>
          <p className={cn(
            "font-mono font-medium",
            isProfitable ? "text-green-400" : "text-red-400"
          )}>
            {isProfitable ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Actions */}
      {position.market_slug && (
        <div className="flex items-center gap-3 mt-4 pt-3 border-t border-border">
          <a
            href={`https://polymarket.com/event/${position.market_slug}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
          >
            <ExternalLink className="w-3 h-3" />
            View on Polymarket
          </a>
        </div>
      )}
    </div>
  )
}
