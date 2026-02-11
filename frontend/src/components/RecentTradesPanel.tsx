import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Bell,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  ExternalLink,
  Filter,
  FolderPlus,
  Hash,
  Layers,
  RefreshCw,
  Target,
  Trash2,
  TrendingDown,
  TrendingUp,
  Users,
  Wallet,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { buildPolymarketMarketUrl } from '../lib/marketUrls'
import { Badge } from './ui/badge'
import {
  discoveryApi,
  type InsiderOpportunity,
  type TrackedTraderOpportunity,
  type TraderGroup,
  type TraderGroupSuggestion,
} from '../services/discoveryApi'
import {
  getRecentTradesFromWallets,
  type RecentTradeFromWallet,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'

interface Props {
  onNavigateToWallet?: (address: string) => void
  mode?: 'full' | 'management' | 'opportunities'
}

type TierFilter = 'WATCH' | 'HIGH' | 'EXTREME'
type SignalSideFilter = 'all' | 'BUY' | 'SELL'

type LiveSignalAlert = {
  id: string
  market: string
  tier: TierFilter
  conviction: number
  outcome: string | null
}

const TIER_COLORS: Record<TierFilter, string> = {
  WATCH: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  HIGH: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  EXTREME: 'bg-red-500/10 text-red-400 border-red-500/20',
}

const TIER_BORDER_COLORS: Record<TierFilter, string> = {
  WATCH: 'border-yellow-500/30',
  HIGH: 'border-orange-500/30',
  EXTREME: 'border-red-500/30',
}

function safeParseTime(value?: string | number | null): Date | null {
  if (value == null) return null
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    const millis = value < 4102444800 ? value * 1000 : value
    const date = new Date(millis)
    return Number.isNaN(date.getTime()) ? null : date
  }

  const str = String(value)
  if (!str) return null
  const direct = new Date(str)
  if (!Number.isNaN(direct.getTime())) return direct

  const asNum = Number(str)
  if (!Number.isNaN(asNum) && asNum > 0) {
    const millis = asNum < 4102444800 ? asNum * 1000 : asNum
    const numericDate = new Date(millis)
    return Number.isNaN(numericDate.getTime()) ? null : numericDate
  }

  return null
}

function formatTimeAgo(value?: string | number | null): string {
  const date = safeParseTime(value)
  if (!date) return 'Unknown'

  const now = Date.now()
  const diffMs = now - date.getTime()
  if (diffMs < 0) return 'Just now'

  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`

  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffHours < 48) return 'Yesterday'
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatDateTime(value?: string | number | null): string {
  const date = safeParseTime(value)
  if (!date) return 'Unknown'
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  })
}

function formatCurrency(value: number): string {
  if (!Number.isFinite(value)) return '$0.00'
  if (Math.abs(value) >= 1000) return `$${(value / 1000).toFixed(1)}k`
  if (Math.abs(value) >= 100) return `$${value.toFixed(0)}`
  return `$${value.toFixed(2)}`
}

function _safeNumber(value: unknown, fallback = 0): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function normalizeTradeSide(raw?: string | null): 'BUY' | 'SELL' | '' {
  const side = (raw || '').toUpperCase()
  if (side === 'BUY' || side === 'YES') return 'BUY'
  if (side === 'SELL' || side === 'NO') return 'SELL'
  return ''
}

function isConditionId(value: string): boolean {
  return value.startsWith('0x') && value.length > 20
}

function getMarketName(trade: RecentTradeFromWallet): string {
  if (trade.market_title && trade.market_title.trim()) return trade.market_title
  if (trade.market && !isConditionId(trade.market)) return trade.market
  return ''
}

function getSignalSide(signal: TrackedTraderOpportunity): 'BUY' | 'SELL' | null {
  const outcome = (signal.outcome || '').toUpperCase()
  if (outcome === 'YES') return 'BUY'
  if (outcome === 'NO') return 'SELL'

  const signalType = (signal.signal_type || '').toLowerCase()
  if (signalType.includes('sell')) return 'SELL'
  if (signalType.includes('buy') || signalType.includes('accumulation')) return 'BUY'
  return null
}

function buildSignalMarketUrl(signal: TrackedTraderOpportunity): string {
  return buildPolymarketMarketUrl({ eventSlug: signal.market_slug }) || ''
}

function getPolymarketTradeUrl(trade: RecentTradeFromWallet): string {
  return buildPolymarketMarketUrl({
    eventSlug: trade.event_slug,
    marketSlug: trade.market_slug,
    marketId: trade.market,
  }) || ''
}

function getTradeId(trade: RecentTradeFromWallet, index: number): string {
  return trade.id || trade.transaction_hash || `${trade.wallet_address}-${index}`
}

function shortAddress(address: string): string {
  if (!address) return 'unknown'
  if (address.length <= 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function parseWalletInput(raw: string): string[] {
  return Array.from(
    new Set(
      raw
        .split(/[\s,]+/)
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  )
}

function convictionColor(value: number): string {
  if (value >= 80) return 'bg-green-500'
  if (value >= 60) return 'bg-yellow-500'
  return 'bg-red-500'
}

function tierRank(value: string | null | undefined): number {
  const normalized = (value || 'WATCH').toUpperCase()
  if (normalized === 'EXTREME') return 3
  if (normalized === 'HIGH') return 2
  return 1
}

function toTier(value: string | null | undefined): TierFilter {
  const normalized = (value || 'WATCH').toUpperCase()
  if (normalized === 'EXTREME') return 'EXTREME'
  if (normalized === 'HIGH') return 'HIGH'
  return 'WATCH'
}

function tradeMatchesSignal(
  trade: RecentTradeFromWallet,
  signal: TrackedTraderOpportunity,
): boolean {
  const signalMarketId = (signal.market_id || '').toLowerCase()
  const tradeMarketId = (trade.market || '').toLowerCase()
  if (signalMarketId && tradeMarketId && signalMarketId === tradeMarketId) {
    return true
  }

  const signalSlug = (signal.market_slug || '').toLowerCase()
  const tradeEventSlug = (trade.event_slug || '').toLowerCase()
  const tradeMarketSlug = (trade.market_slug || '').toLowerCase()
  if (signalSlug && (signalSlug === tradeEventSlug || signalSlug === tradeMarketSlug)) {
    return true
  }

  const signalQuestion = (signal.market_question || '').trim().toLowerCase()
  const tradeTitle = (trade.market_title || '').trim().toLowerCase()
  if (signalQuestion && tradeTitle) {
    if (signalQuestion === tradeTitle) return true
    if (signalQuestion.includes(tradeTitle) || tradeTitle.includes(signalQuestion)) {
      return true
    }
  }

  return false
}

export default function RecentTradesPanel({
  onNavigateToWallet,
  mode = 'full',
}: Props) {
  const showManagement = mode !== 'opportunities'
  const showOpportunities = mode !== 'management'

  const [hoursFilter, setHoursFilter] = useState(24)
  const [minTier, setMinTier] = useState<TierFilter>('WATCH')
  const [sideFilter, setSideFilter] = useState<SignalSideFilter>('all')
  const [signalLimit, setSignalLimit] = useState(50)
  const [expandedSignals, setExpandedSignals] = useState<Set<string>>(new Set())
  const [liveAlerts, setLiveAlerts] = useState<LiveSignalAlert[]>([])
  const [groupName, setGroupName] = useState('')
  const [groupDescription, setGroupDescription] = useState('')
  const [groupWalletInput, setGroupWalletInput] = useState('')
  const [groupStatusMessage, setGroupStatusMessage] = useState<string | null>(null)
  const [showGroupForm, setShowGroupForm] = useState(false)

  const queryClient = useQueryClient()
  const { lastMessage } = useWebSocket('/ws')

  const {
    data: opportunities = [],
    isLoading: opportunitiesLoading,
    refetch: refetchSignals,
    isRefetching: isRefetchingSignals,
  } = useQuery({
    queryKey: ['tracked-trader-opportunities', minTier, signalLimit],
    queryFn: () => discoveryApi.getTrackedTraderOpportunities(signalLimit, minTier),
    refetchInterval: 30000,
    enabled: showOpportunities,
  })

  const {
    data: insiderOppData,
    isLoading: insiderOppLoading,
    refetch: refetchInsiderOpps,
    isRefetching: isRefetchingInsiderOpps,
  } = useQuery({
    queryKey: ['insider-opportunities', sideFilter],
    queryFn: () =>
      discoveryApi.getInsiderOpportunities({
        limit: 40,
        min_confidence: 0.62,
        max_age_minutes: 180,
        direction:
          sideFilter === 'BUY'
            ? 'buy_yes'
            : sideFilter === 'SELL'
              ? 'buy_no'
              : undefined,
      }),
    refetchInterval: 30000,
    enabled: showOpportunities,
  })

  const {
    data: rawTradesData,
    isLoading: rawTradesLoading,
    refetch: refetchRawTrades,
    isRefetching: isRefetchingRawTrades,
  } = useQuery({
    queryKey: ['recent-trades-from-wallets', hoursFilter, 500],
    queryFn: () => getRecentTradesFromWallets({ limit: 500, hours: hoursFilter }),
    refetchInterval: 30000,
  })

  const {
    data: traderGroups = [],
    isLoading: groupsLoading,
    refetch: refetchGroups,
  } = useQuery<TraderGroup[]>({
    queryKey: ['trader-groups'],
    queryFn: () => discoveryApi.getTraderGroups(true, 12),
    refetchInterval: 60000,
    enabled: showManagement,
  })

  const {
    data: groupSuggestions = [],
    isLoading: suggestionsLoading,
    refetch: refetchSuggestions,
  } = useQuery<TraderGroupSuggestion[]>({
    queryKey: ['trader-group-suggestions'],
    queryFn: () =>
      discoveryApi.getTraderGroupSuggestions({
        min_group_size: 3,
        max_suggestions: 8,
        min_composite_score: 0.6,
      }),
    refetchInterval: 90000,
    enabled: showManagement,
  })

  const createGroupMutation = useMutation({
    mutationFn: (payload: {
      name: string
      description?: string
      wallet_addresses: string[]
      source_type?: 'manual' | 'suggested_cluster' | 'suggested_tag' | 'suggested_pool'
      suggestion_key?: string
      criteria?: Record<string, unknown>
      source_label?: string
    }) =>
      discoveryApi.createTraderGroup({
        ...payload,
        auto_track_members: true,
      }),
    onSuccess: (result) => {
      setGroupStatusMessage(
        `Group created (${result.group?.member_count ?? 0} members, ${result.tracked_members} tracked).`,
      )
      setGroupName('')
      setGroupDescription('')
      setGroupWalletInput('')
      queryClient.invalidateQueries({ queryKey: ['trader-groups'] })
      queryClient.invalidateQueries({ queryKey: ['trader-group-suggestions'] })
      queryClient.invalidateQueries({ queryKey: ['recent-trades-from-wallets'] })
    },
    onError: (error: unknown) => {
      const message =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        || 'Failed to create group'
      setGroupStatusMessage(message)
    },
  })

  const trackGroupMembersMutation = useMutation({
    mutationFn: (groupId: string) => discoveryApi.trackTraderGroupMembers(groupId),
    onSuccess: (result) => {
      setGroupStatusMessage(`Tracking refreshed for ${result.tracked_members} group members.`)
      queryClient.invalidateQueries({ queryKey: ['recent-trades-from-wallets'] })
    },
    onError: () => setGroupStatusMessage('Failed to track group members'),
  })

  const deleteGroupMutation = useMutation({
    mutationFn: (groupId: string) => discoveryApi.deleteTraderGroup(groupId),
    onSuccess: () => {
      setGroupStatusMessage('Group deleted')
      queryClient.invalidateQueries({ queryKey: ['trader-groups'] })
      queryClient.invalidateQueries({ queryKey: ['trader-group-suggestions'] })
    },
    onError: () => setGroupStatusMessage('Failed to delete group'),
  })

  useEffect(() => {
    if (!showOpportunities) return
    if (lastMessage?.type !== 'tracked_trader_signal' || !lastMessage.data) return

    const tier = toTier(lastMessage.data.tier)
    if (tierRank(tier) < tierRank('HIGH')) return

    const conviction = Math.max(
      0,
      Math.min(100, Math.round(Number(lastMessage.data.conviction_score || 0))),
    )

    const alert: LiveSignalAlert = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      market:
        String(lastMessage.data.market_question || '').trim() ||
        String(lastMessage.data.market_id || 'Unknown market'),
      tier,
      conviction,
      outcome: lastMessage.data.outcome || null,
    }

    setLiveAlerts((prev) => [alert, ...prev].slice(0, 4))
    const timer = setTimeout(() => {
      setLiveAlerts((prev) => prev.filter((item) => item.id !== alert.id))
    }, 8000)

    return () => clearTimeout(timer)
  }, [lastMessage, showOpportunities])

  const rawTrades = rawTradesData?.trades || []
  const trackedWallets = rawTradesData?.tracked_wallets || 0
  const insiderOpportunities: InsiderOpportunity[] = insiderOppData?.opportunities || []
  const isLoading = opportunitiesLoading || rawTradesLoading || insiderOppLoading
  const isRefetching = isRefetchingSignals || isRefetchingRawTrades || isRefetchingInsiderOpps

  const sortedSignals = useMemo(() => {
    return [...opportunities].sort((a, b) => {
      const convictionDiff = (b.conviction_score || 0) - (a.conviction_score || 0)
      if (convictionDiff !== 0) return convictionDiff
      const bTime = safeParseTime(b.last_seen_at || b.detected_at)?.getTime() || 0
      const aTime = safeParseTime(a.last_seen_at || a.detected_at)?.getTime() || 0
      return bTime - aTime
    })
  }, [opportunities])

  const filteredSignals = useMemo(() => {
    if (sideFilter === 'all') return sortedSignals
    return sortedSignals.filter((signal) => getSignalSide(signal) === sideFilter)
  }, [sortedSignals, sideFilter])

  const signalTradesMap = useMemo(() => {
    const map = new Map<string, RecentTradeFromWallet[]>()

    for (const signal of filteredSignals) {
      const expectedSide = getSignalSide(signal)
      const matched = rawTrades
        .filter((trade) => tradeMatchesSignal(trade, signal))
        .filter((trade) =>
          expectedSide ? normalizeTradeSide(trade.side) === expectedSide : true,
        )
        .sort((a, b) => {
          const bTime =
            safeParseTime(
              b.timestamp_iso || b.match_time || b.timestamp || b.time || b.created_at,
            )?.getTime() || 0
          const aTime =
            safeParseTime(
              a.timestamp_iso || a.match_time || a.timestamp || a.time || a.created_at,
            )?.getTime() || 0
          return bTime - aTime
        })

      map.set(signal.id, matched)
    }

    return map
  }, [filteredSignals, rawTrades])

  const uniqueSignalMarkets = useMemo(() => {
    const keys = new Set<string>()
    for (const signal of filteredSignals) {
      keys.add(`${signal.market_id}:${signal.outcome || ''}`)
    }
    return keys.size
  }, [filteredSignals])

  const highSignals = filteredSignals.filter(
    (signal) => tierRank(signal.tier) >= tierRank('HIGH'),
  ).length
  const extremeSignals = filteredSignals.filter(
    (signal) => toTier(signal.tier) === 'EXTREME',
  ).length

  const avgConviction = filteredSignals.length
    ? filteredSignals.reduce((sum, signal) => sum + (signal.conviction_score || 0), 0) /
      filteredSignals.length
    : 0

  const auditedTradeCount = Array.from(signalTradesMap.values()).reduce(
    (sum, trades) => sum + trades.length,
    0,
  )

  const trackedWalletActivity = useMemo(() => {
    const map = new Map<
      string,
      {
        wallet_address: string
        wallet_username?: string
        wallet_label?: string
        trade_count: number
        latest_trade_at: Date | null
      }
    >()

    for (const trade of rawTrades) {
      const key = trade.wallet_address.toLowerCase()
      const existing = map.get(key)
      const tradeTime =
        safeParseTime(
          trade.timestamp_iso || trade.match_time || trade.timestamp || trade.time || trade.created_at,
        ) || null
      if (!existing) {
        map.set(key, {
          wallet_address: trade.wallet_address,
          wallet_username: trade.wallet_username,
          wallet_label: trade.wallet_label,
          trade_count: 1,
          latest_trade_at: tradeTime,
        })
        continue
      }
      existing.trade_count += 1
      if (
        tradeTime
        && (!existing.latest_trade_at || tradeTime.getTime() > existing.latest_trade_at.getTime())
      ) {
        existing.latest_trade_at = tradeTime
      }
    }

    return Array.from(map.values()).sort((a, b) => {
      if (b.trade_count !== a.trade_count) return b.trade_count - a.trade_count
      return (b.latest_trade_at?.getTime() || 0) - (a.latest_trade_at?.getTime() || 0)
    })
  }, [rawTrades])

  const totalGroupMembers = useMemo(
    () => traderGroups.reduce((sum, group) => sum + (group.member_count || 0), 0),
    [traderGroups],
  )

  const trackedWalletSet = useMemo(() => {
    const set = new Set<string>()
    trackedWalletActivity.forEach((item) => set.add(item.wallet_address.toLowerCase()))
    return set
  }, [trackedWalletActivity])

  const toggleExpanded = (signalId: string) => {
    setExpandedSignals((prev) => {
      const next = new Set(prev)
      if (next.has(signalId)) {
        next.delete(signalId)
      } else {
        next.add(signalId)
      }
      return next
    })
  }

  const handleRefresh = () => {
    refetchRawTrades()
    if (showOpportunities) {
      refetchSignals()
      refetchInsiderOpps()
    }
    if (showManagement) {
      refetchGroups()
      refetchSuggestions()
    }
  }

  const handleCreateManualGroup = () => {
    const name = groupName.trim()
    const walletAddresses = parseWalletInput(groupWalletInput)
    if (!name) {
      setGroupStatusMessage('Group name is required')
      return
    }
    if (walletAddresses.length === 0) {
      setGroupStatusMessage('Add at least one wallet address')
      return
    }

    createGroupMutation.mutate({
      name,
      description: groupDescription.trim() || undefined,
      wallet_addresses: walletAddresses,
      source_type: 'manual',
      source_label: 'manual',
    })
  }

  const handleCreateSuggestionGroup = (suggestion: TraderGroupSuggestion) => {
    createGroupMutation.mutate({
      name: suggestion.name,
      description: suggestion.description,
      wallet_addresses: suggestion.wallet_addresses,
      source_type:
        suggestion.kind === 'cluster'
          ? 'suggested_cluster'
          : suggestion.kind === 'tag'
            ? 'suggested_tag'
            : 'suggested_pool',
      suggestion_key: suggestion.id,
      criteria: suggestion.criteria || {},
      source_label: 'suggested',
    })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/10 rounded-lg">
            <Zap className="w-5 h-5 text-orange-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              {showOpportunities && showManagement
                ? 'Traders'
                : showOpportunities
                  ? 'Trader Opportunities'
                  : 'Trader Management'}
            </h2>
            <p className="text-sm text-muted-foreground/70">
              {showOpportunities && showManagement
                ? 'Tracked traders, trader groups, and discovery confluence from high-quality discovered wallets'
                : showOpportunities
                  ? 'Confluence and insider opportunities generated from tracked trader activity'
                  : 'Tracked trader lists, group management, and monitoring controls'}
            </p>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={isRefetching}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm',
            'bg-muted text-foreground/80 hover:bg-accent transition-colors',
            isRefetching && 'opacity-50',
          )}
        >
          <RefreshCw className={cn('w-4 h-4', isRefetching && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {showManagement && (
        <div className="rounded-lg border border-border bg-card/60 p-4 space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-foreground">Tracked Traders</h3>
          </div>
          <button
            onClick={() => setShowGroupForm((v) => !v)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs',
              showGroupForm
                ? 'border-blue-500/40 bg-blue-500/10 text-blue-300'
                : 'border-border bg-muted text-foreground/80 hover:bg-accent',
            )}
          >
            <FolderPlus className="w-3.5 h-3.5" />
            Manual Group
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">Tracked Wallets</p>
            <p className="text-sm font-semibold text-foreground">{trackedWallets}</p>
          </div>
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">Trader Groups</p>
            <p className="text-sm font-semibold text-foreground">{traderGroups.length}</p>
          </div>
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">Group Members</p>
            <p className="text-sm font-semibold text-foreground">{totalGroupMembers}</p>
          </div>
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">Recent Trades ({hoursFilter}h)</p>
            <p className="text-sm font-semibold text-foreground">{rawTrades.length}</p>
          </div>
        </div>

        {groupStatusMessage && (
          <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
            {groupStatusMessage}
          </div>
        )}

        {showGroupForm && (
          <div className="rounded-md border border-border bg-background/40 p-3 space-y-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <input
                type="text"
                value={groupName}
                onChange={(e) => setGroupName(e.target.value)}
                placeholder="Group name"
                className="bg-muted border border-border rounded px-2 py-1.5 text-sm"
              />
              <input
                type="text"
                value={groupDescription}
                onChange={(e) => setGroupDescription(e.target.value)}
                placeholder="Description (optional)"
                className="bg-muted border border-border rounded px-2 py-1.5 text-sm"
              />
            </div>
            <textarea
              value={groupWalletInput}
              onChange={(e) => setGroupWalletInput(e.target.value)}
              placeholder="Wallet addresses (comma/newline separated)"
              className="w-full min-h-[72px] bg-muted border border-border rounded px-2 py-1.5 text-sm"
            />
            <div className="flex items-center justify-end">
              <button
                onClick={handleCreateManualGroup}
                disabled={createGroupMutation.isPending}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium',
                  'bg-blue-500/20 text-blue-300 hover:bg-blue-500/30',
                  createGroupMutation.isPending && 'opacity-50',
                )}
              >
                {createGroupMutation.isPending ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <FolderPlus className="w-3.5 h-3.5" />
                )}
                Create + Track
              </button>
            </div>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-muted-foreground/70" />
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Existing Groups
            </p>
          </div>
          {groupsLoading ? (
            <div className="rounded-md border border-border bg-background/40 p-3 text-sm text-muted-foreground/70">
              Loading groups...
            </div>
          ) : traderGroups.length === 0 ? (
            <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
              No groups created yet.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {traderGroups.map((group) => (
                <div
                  key={group.id}
                  className="rounded-md border border-border bg-background/40 p-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium text-foreground">{group.name}</p>
                      <p className="text-[11px] text-muted-foreground/70">
                        {group.member_count} members
                        <span className="mx-1">•</span>
                        {group.source_type.replace(/_/g, ' ')}
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => trackGroupMembersMutation.mutate(group.id)}
                        disabled={trackGroupMembersMutation.isPending}
                        className="inline-flex items-center gap-1 rounded-md bg-blue-500/15 px-2 py-1 text-[11px] text-blue-300 hover:bg-blue-500/25"
                      >
                        <CheckCircle2 className="w-3 h-3" />
                        Track
                      </button>
                      <button
                        onClick={() => deleteGroupMutation.mutate(group.id)}
                        disabled={deleteGroupMutation.isPending}
                        className="inline-flex items-center gap-1 rounded-md bg-red-500/15 px-2 py-1 text-[11px] text-red-300 hover:bg-red-500/25"
                      >
                        <Trash2 className="w-3 h-3" />
                        Delete
                      </button>
                    </div>
                  </div>
                  {group.members && group.members.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {group.members.slice(0, 4).map((member) => (
                        <button
                          key={member.id}
                          onClick={() => onNavigateToWallet?.(member.wallet_address)}
                          className="inline-flex items-center gap-1 rounded bg-muted/80 px-2 py-0.5 text-[11px] text-foreground/80 hover:bg-muted"
                        >
                          <Wallet className="w-3 h-3 text-muted-foreground/70" />
                          {member.username || shortAddress(member.wallet_address)}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-muted-foreground/70" />
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Suggested Groups (Discovery)
            </p>
          </div>
          {suggestionsLoading ? (
            <div className="rounded-md border border-border bg-background/40 p-3 text-sm text-muted-foreground/70">
              Building suggestions...
            </div>
          ) : groupSuggestions.length === 0 ? (
            <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
              No suggestions available yet.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {groupSuggestions.map((suggestion) => {
                const trackedOverlap = suggestion.wallet_addresses.filter((address) =>
                  trackedWalletSet.has(address.toLowerCase()),
                ).length

                return (
                  <div
                    key={suggestion.id}
                    className="rounded-md border border-border bg-background/40 p-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium text-foreground">{suggestion.name}</p>
                        <p className="text-[11px] text-muted-foreground/70">
                          {suggestion.wallet_count} traders
                          <span className="mx-1">•</span>
                          {suggestion.kind.replace(/_/g, ' ')}
                        </p>
                      </div>
                      <button
                        onClick={() => handleCreateSuggestionGroup(suggestion)}
                        disabled={
                          createGroupMutation.isPending
                          || !!suggestion.already_exists
                        }
                        className={cn(
                          'inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px]',
                          suggestion.already_exists
                            ? 'bg-muted text-muted-foreground'
                            : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30',
                        )}
                      >
                        <FolderPlus className="w-3 h-3" />
                        {suggestion.already_exists ? 'Created' : 'Create + Track'}
                      </button>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground/70 line-clamp-2">
                      {suggestion.description}
                    </p>
                    <div className="mt-2 flex items-center gap-3 text-[11px] text-muted-foreground/70">
                      <span>Avg score {(suggestion.avg_composite_score ?? 0).toFixed(2)}</span>
                      <span>{trackedOverlap} already tracked</span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-muted-foreground/70" />
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Active Tracked Traders
            </p>
          </div>
          {trackedWalletActivity.length === 0 ? (
            <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
              No tracked-wallet activity found in the selected trade window.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {trackedWalletActivity.slice(0, 9).map((wallet) => (
                <button
                  key={wallet.wallet_address}
                  onClick={() => onNavigateToWallet?.(wallet.wallet_address)}
                  className="rounded-md border border-border bg-background/40 p-3 text-left hover:border-border/80 transition-colors"
                >
                  <p className="text-sm font-medium text-foreground">
                    {wallet.wallet_username || wallet.wallet_label || shortAddress(wallet.wallet_address)}
                  </p>
                  <p className="text-[11px] text-muted-foreground/70 font-mono">
                    {shortAddress(wallet.wallet_address)}
                  </p>
                  <div className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-blue-300">{wallet.trade_count} trades</span>
                    <span className="text-muted-foreground/70">
                      {wallet.latest_trade_at ? formatTimeAgo(wallet.latest_trade_at.toISOString()) : 'Unknown'}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      )}

      {showOpportunities && (
        <>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold text-foreground">
              Discovery Confluence (High-Quality Traders)
            </h3>
          </div>

          <div className="rounded-lg border border-border bg-card/60 p-4 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <h3 className="text-sm font-semibold text-foreground">Insider Opportunities</h3>
              </div>
              <p className="text-xs text-muted-foreground/70">
                {insiderOpportunities.length} active in last 180m
              </p>
            </div>

            {insiderOpportunities.length === 0 ? (
              <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
                No insider opportunities match current filters.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {insiderOpportunities.slice(0, 12).map((opp) => {
                  const marketUrl = buildPolymarketMarketUrl({ marketId: opp.market_id }) || ''
                  const isYes = (opp.direction || '').toLowerCase() === 'buy_yes'
                  return (
                    <div
                      key={opp.id}
                      className="rounded-md border border-red-500/20 bg-red-500/5 p-3 space-y-2"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-sm font-medium text-foreground line-clamp-2">
                          {opp.market_question || opp.market_id}
                        </p>
                        <Badge
                          variant="outline"
                          className={cn(
                            'text-[10px]',
                            isYes
                              ? 'bg-green-500/10 text-green-300 border-green-500/20'
                              : 'bg-red-500/10 text-red-300 border-red-500/20',
                          )}
                        >
                          {isYes ? 'BUY YES' : 'BUY NO'}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <p className="text-muted-foreground/70">Confidence</p>
                          <p className="font-semibold text-foreground">{((_safeNumber(opp.confidence)) * 100).toFixed(0)}%</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground/70">Edge</p>
                          <p className="font-semibold text-foreground">{_safeNumber(opp.edge_percent).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground/70">Wallets / Clusters</p>
                          <p className="font-semibold text-foreground">
                            {opp.wallet_count} / {opp.cluster_count}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground/70">Pre-news lead</p>
                          <p className="font-semibold text-foreground">
                            {_safeNumber(opp.pre_news_lead_minutes).toFixed(0)}m
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-muted-foreground/70">
                        <span>Freshness {(_safeNumber(opp.freshness_minutes)).toFixed(0)}m</span>
                        <span>Insider {_safeNumber(opp.insider_score).toFixed(2)}</span>
                      </div>

                      <div className="flex items-center gap-2 pt-1">
                        {marketUrl && (
                          <a
                            href={marketUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 rounded bg-blue-500/15 px-2 py-1 text-[11px] text-blue-300 hover:bg-blue-500/25"
                          >
                            <ExternalLink className="w-3 h-3" />
                            Open market
                          </a>
                        )}
                        {opp.top_wallet?.address && (
                          <button
                            onClick={() => onNavigateToWallet?.(opp.top_wallet!.address)}
                            className="inline-flex items-center gap-1 rounded bg-orange-500/15 px-2 py-1 text-[11px] text-orange-300 hover:bg-orange-500/25"
                          >
                            <Wallet className="w-3 h-3" />
                            {opp.top_wallet.username || shortAddress(opp.top_wallet.address)}
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-4 p-3 bg-card rounded-lg border border-border">
            <Filter className="w-4 h-4 text-muted-foreground/70" />

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground/70">Min tier:</span>
              <select
                value={minTier}
                onChange={(e) => setMinTier(e.target.value as TierFilter)}
                className="bg-muted border border-border rounded px-2 py-1 text-sm"
              >
                <option value="WATCH">Watch (5+)</option>
                <option value="HIGH">High (10+)</option>
                <option value="EXTREME">Extreme (15+)</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground/70">Side:</span>
              <select
                value={sideFilter}
                onChange={(e) => setSideFilter(e.target.value as SignalSideFilter)}
                className="bg-muted border border-border rounded px-2 py-1 text-sm"
              >
                <option value="all">All</option>
                <option value="BUY">Buy clusters</option>
                <option value="SELL">Sell clusters</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground/70">Raw window:</span>
              <select
                value={hoursFilter}
                onChange={(e) => setHoursFilter(Number(e.target.value))}
                className="bg-muted border border-border rounded px-2 py-1 text-sm"
              >
                <option value={1}>Last hour</option>
                <option value={6}>Last 6 hours</option>
                <option value={24}>Last 24 hours</option>
                <option value={48}>Last 48 hours</option>
                <option value={168}>Last 7 days</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground/70">Max signals:</span>
              <select
                value={signalLimit}
                onChange={(e) => setSignalLimit(Number(e.target.value))}
                className="bg-muted border border-border rounded px-2 py-1 text-sm"
              >
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70">Signals</p>
              <p className="text-lg font-semibold text-foreground">{filteredSignals.length}</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70">High / Extreme</p>
              <p className="text-lg font-semibold text-orange-400">
                {highSignals}
                <span className="text-muted-foreground/60 text-sm ml-1">/ {extremeSignals}</span>
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70">Avg Conviction</p>
              <p className="text-lg font-semibold text-foreground">{avgConviction.toFixed(1)}</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70">Markets / Wallets</p>
              <p className="text-lg font-semibold text-foreground">
                <span className="text-blue-400">{uniqueSignalMarkets}</span>
                <span className="text-muted-foreground/50 mx-1">/</span>
                <span className="text-orange-400">{trackedWallets}</span>
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70">Audited Trades</p>
              <p className="text-lg font-semibold text-foreground">{auditedTradeCount}</p>
            </div>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground/70" />
            </div>
          ) : filteredSignals.length === 0 ? (
            <div className="text-center py-12 bg-card rounded-lg border border-border">
              <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-muted-foreground">No confluence signals found</p>
              <p className="text-sm text-muted-foreground/50 mt-1">
                Try lowering tier/side filters or wait for new clustered entries
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredSignals.map((signal) => {
                const tier = toTier(signal.tier)
                const conviction = Math.round(signal.conviction_score || signal.strength * 100 || 0)
                const signalSide = getSignalSide(signal)
                const isBuy = signalSide === 'BUY'
                const isExpanded = expandedSignals.has(signal.id)
                const marketUrl = buildSignalMarketUrl(signal)
                const relatedTrades = signalTradesMap.get(signal.id) || []
                const topWallets = signal.top_wallets || []

                return (
                  <div
                    key={signal.id}
                    className={cn(
                      'bg-card border rounded-lg overflow-hidden transition-colors',
                      TIER_BORDER_COLORS[tier],
                    )}
                  >
                    <div
                      className="p-4 cursor-pointer"
                      onClick={() => toggleExpanded(signal.id)}
                    >
                      <div className="flex items-start gap-3">
                        <div
                          className={cn(
                            'flex-shrink-0 w-1 h-16 rounded-full mt-0.5',
                            isBuy ? 'bg-green-500' : signalSide === 'SELL' ? 'bg-red-500' : 'bg-blue-500',
                          )}
                        />

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                            <span
                              className={cn(
                                'px-1.5 py-0.5 rounded text-xs font-medium border',
                                TIER_COLORS[tier],
                              )}
                            >
                              {tier}
                            </span>
                            <span
                              className={cn(
                                'px-1.5 py-0.5 rounded text-xs font-medium',
                                isBuy
                                  ? 'bg-green-500/10 text-green-400'
                                  : signalSide === 'SELL'
                                    ? 'bg-red-500/10 text-red-400'
                                    : 'bg-blue-500/10 text-blue-400',
                              )}
                            >
                              {signal.outcome || signal.signal_type.replace(/_/g, ' ')}
                            </span>
                            <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/80">
                              Conviction {conviction}
                            </span>
                            <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/80">
                              {signal.window_minutes || 60}m window
                            </span>
                          </div>

                          <h3 className="font-medium text-sm text-foreground line-clamp-2">
                            {signal.market_question || signal.market_id}
                          </h3>

                          <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                            <div>
                              <p className="text-muted-foreground/70">Adjusted Wallets</p>
                              <p className="font-semibold text-foreground flex items-center gap-1">
                                <Users className="w-3 h-3 text-muted-foreground/70" />
                                {signal.cluster_adjusted_wallet_count || signal.wallet_count}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground/70">Core Wallets</p>
                              <p className="font-semibold text-foreground">
                                {signal.unique_core_wallets || 0}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground/70">Net Notional</p>
                              <p className="font-semibold text-foreground">
                                {formatCurrency(signal.net_notional || 0)}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground/70">Last Reinforced</p>
                              <p className="font-semibold text-foreground">
                                {formatTimeAgo(signal.last_seen_at || signal.detected_at)}
                              </p>
                            </div>
                          </div>

                          <div className="mt-3">
                            <div className="flex items-center justify-between text-[11px] mb-1">
                              <span className="text-muted-foreground/70">Conviction meter</span>
                              <span className="text-foreground/90 font-medium">{conviction}/100</span>
                            </div>
                            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                              <div
                                className={cn('h-full rounded-full', convictionColor(conviction))}
                                style={{ width: `${Math.max(0, Math.min(100, conviction))}%` }}
                              />
                            </div>
                          </div>

                          {topWallets.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-1.5">
                              {topWallets.slice(0, 4).map((wallet) => (
                                <button
                                  key={`${signal.id}-${wallet.address}`}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    onNavigateToWallet?.(wallet.address)
                                  }}
                                  className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted/70 hover:bg-muted transition-colors text-[11px] text-foreground/85"
                                >
                                  <Wallet className="w-3 h-3 text-muted-foreground/70" />
                                  <span>
                                    {wallet.username || `${wallet.address.slice(0, 6)}...${wallet.address.slice(-4)}`}
                                  </span>
                                  <span className="text-muted-foreground/70">
                                    {(wallet.composite_score * 100).toFixed(0)}
                                  </span>
                                </button>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="flex-shrink-0 text-right">
                          <div className="flex items-center gap-1 text-xs text-muted-foreground/70 justify-end">
                            <Clock className="w-3 h-3" />
                            {formatTimeAgo(signal.detected_at)}
                          </div>
                          <div className="mt-2 flex items-center gap-2 justify-end">
                            {marketUrl && (
                              <a
                                href={marketUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
                              >
                                <ExternalLink className="w-3 h-3" />
                                Market
                              </a>
                            )}
                            {isExpanded ? (
                              <ChevronUp className="w-4 h-4 text-muted-foreground/60" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-muted-foreground/60" />
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="border-t border-border p-4 bg-background/30">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-muted-foreground/70" />
                            <p className="text-sm font-medium text-foreground">Raw Trade Audit Trail</p>
                          </div>
                          <p className="text-xs text-muted-foreground/70">
                            {relatedTrades.length} matching trades in last {hoursFilter}h
                          </p>
                        </div>

                        {relatedTrades.length === 0 ? (
                          <div className="rounded-lg border border-dashed border-border p-3 text-sm text-muted-foreground/70">
                            No matching raw trades were found in the selected audit window.
                          </div>
                        ) : (
                          <div className="space-y-2">
                            {relatedTrades.slice(0, 20).map((trade, index) => {
                              const tradeId = getTradeId(trade, index)
                              const tradeSide = normalizeTradeSide(trade.side)
                              const tradeIsBuy = tradeSide === 'BUY'
                              const cost = trade.cost ?? (trade.size ?? 0) * (trade.price ?? 0)
                              const marketName = getMarketName(trade)
                              const polymarketTradeUrl = getPolymarketTradeUrl(trade)

                              return (
                                <div
                                  key={tradeId}
                                  className="rounded-lg border border-border bg-card/70 p-3"
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="min-w-0 flex-1">
                                      <div className="flex items-center gap-2 flex-wrap">
                                        <span
                                          className={cn(
                                            'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium',
                                            tradeIsBuy
                                              ? 'bg-green-500/10 text-green-400'
                                              : 'bg-red-500/10 text-red-400',
                                          )}
                                        >
                                          {tradeIsBuy ? (
                                            <TrendingUp className="w-3 h-3" />
                                          ) : (
                                            <TrendingDown className="w-3 h-3" />
                                          )}
                                          {tradeSide || trade.side || 'TRADE'}
                                        </span>
                                        {trade.outcome && (
                                          <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/80">
                                            {trade.outcome}
                                          </span>
                                        )}
                                        <span className="text-xs text-muted-foreground/70">
                                          {formatTimeAgo(
                                            trade.timestamp_iso ||
                                              trade.match_time ||
                                              trade.timestamp ||
                                              trade.time ||
                                              trade.created_at,
                                          )}
                                        </span>
                                      </div>

                                      <p className="mt-1 text-sm text-foreground truncate">
                                        {marketName || signal.market_question || 'Unknown market'}
                                      </p>

                                      <div className="mt-1 flex items-center gap-2 text-xs">
                                        <button
                                          onClick={() => onNavigateToWallet?.(trade.wallet_address)}
                                          className="text-orange-400 hover:text-orange-300 hover:underline font-mono"
                                        >
                                          {trade.wallet_username || trade.wallet_label}
                                        </button>
                                        <span className="text-muted-foreground/50">
                                          {trade.wallet_address.slice(0, 6)}...{trade.wallet_address.slice(-4)}
                                        </span>
                                      </div>
                                    </div>

                                    <div className="text-right">
                                      <p
                                        className={cn(
                                          'text-sm font-semibold',
                                          tradeIsBuy ? 'text-green-400' : 'text-red-400',
                                        )}
                                      >
                                        {((trade.price || 0) * 100).toFixed(1)}c
                                      </p>
                                      <p className="text-xs text-muted-foreground/70">
                                        {(trade.size || 0).toLocaleString(undefined, {
                                          maximumFractionDigits: 0,
                                        })}{' '}
                                        shares
                                      </p>
                                      <p className="text-xs text-muted-foreground/70">
                                        {formatCurrency(cost)}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="mt-2 pt-2 border-t border-border/70 flex items-center gap-3 text-xs">
                                    <span className="text-muted-foreground/70">
                                      {formatDateTime(
                                        trade.timestamp_iso ||
                                          trade.match_time ||
                                          trade.timestamp ||
                                          trade.time ||
                                          trade.created_at,
                                      )}
                                    </span>
                                    {polymarketTradeUrl && (
                                      <a
                                        href={polymarketTradeUrl}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300"
                                      >
                                        <ExternalLink className="w-3 h-3" />
                                        Market
                                      </a>
                                    )}
                                    {trade.transaction_hash && (
                                      <a
                                        href={`https://polygonscan.com/tx/${trade.transaction_hash}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground/80"
                                      >
                                        <Hash className="w-3 h-3" />
                                        Tx
                                      </a>
                                    )}
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        )}

                        <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground/70 flex flex-wrap gap-4">
                          <span className="inline-flex items-center gap-1">
                            <Target className="w-3 h-3" />
                            First seen: {formatTimeAgo(signal.first_seen_at || signal.detected_at)}
                          </span>
                          <span className="inline-flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            Last seen: {formatTimeAgo(signal.last_seen_at || signal.detected_at)}
                          </span>
                          <span className="inline-flex items-center gap-1">
                            <Users className="w-3 h-3" />
                            Wallets: {signal.wallet_count}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </>
      )}

      {showOpportunities && liveAlerts.length > 0 && (
        <div className="fixed bottom-4 right-4 z-50 space-y-2 pointer-events-none">
          {liveAlerts.map((alert) => (
            <div
              key={alert.id}
              className={cn(
                'pointer-events-auto min-w-[280px] max-w-[360px] rounded-lg border px-3 py-2 shadow-lg backdrop-blur',
                alert.tier === 'EXTREME'
                  ? 'bg-red-500/15 border-red-500/30 text-red-100'
                  : 'bg-orange-500/15 border-orange-500/30 text-orange-100',
              )}
            >
              <div className="flex items-start gap-2">
                <Bell className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-xs font-semibold">
                    {alert.tier} confluence signal {alert.outcome ? `(${alert.outcome})` : ''}
                  </p>
                  <p className="text-xs mt-0.5 line-clamp-2">{alert.market}</p>
                  <p className="text-[11px] mt-1 opacity-90">
                    Conviction {alert.conviction}/100
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
