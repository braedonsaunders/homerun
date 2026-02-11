import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  AlertCircle,
  Bell,
  CheckCircle2,
  Filter,
  FolderPlus,
  Layers,
  RefreshCw,
  Target,
  Trash2,
  Users,
  Wallet,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  discoveryApi,
  type InsiderOpportunity,
  type TrackedTraderOpportunity,
  type TraderGroup,
  type TraderGroupSuggestion,
} from '../services/discoveryApi'
import {
  getRecentTradesFromWallets,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import {
  type UnifiedTraderSignal,
  normalizeConfluenceSignal,
  normalizeInsiderSignal,
  TraderSignalCards,
  TraderSignalTable,
  TraderSignalTerminal,
} from './TraderSignalViews'

interface Props {
  onNavigateToWallet?: (address: string) => void
  mode?: 'full' | 'management' | 'opportunities'
  viewMode?: 'card' | 'list' | 'terminal'
}

type TierFilter = 'WATCH' | 'HIGH' | 'EXTREME'
type SignalSideFilter = 'all' | 'BUY' | 'SELL'
type SourceFilter = 'all' | 'confluence' | 'insider'

type LiveSignalAlert = {
  id: string
  market: string
  tier: TierFilter
  conviction: number
  outcome: string | null
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

function getSignalSide(signal: TrackedTraderOpportunity): 'BUY' | 'SELL' | null {
  const outcome = (signal.outcome || '').toUpperCase()
  if (outcome === 'YES') return 'BUY'
  if (outcome === 'NO') return 'SELL'

  const signalType = (signal.signal_type || '').toLowerCase()
  if (signalType.includes('sell')) return 'SELL'
  if (signalType.includes('buy') || signalType.includes('accumulation')) return 'BUY'
  return null
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

export default function RecentTradesPanel({
  onNavigateToWallet,
  mode = 'full',
  viewMode = 'card',
}: Props) {
  const showManagement = mode !== 'opportunities'
  const showOpportunities = mode !== 'management'

  const [hoursFilter] = useState(24)
  const [minTier, setMinTier] = useState<TierFilter>('WATCH')
  const [sideFilter, setSideFilter] = useState<SignalSideFilter>('all')
  const [signalLimit, setSignalLimit] = useState(50)
  const [liveAlerts, setLiveAlerts] = useState<LiveSignalAlert[]>([])
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')
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

  // Unified merged signal list (confluence + insider)
  const unifiedSignals = useMemo<UnifiedTraderSignal[]>(() => {
    const confluenceNormalized = filteredSignals.map(normalizeConfluenceSignal)
    const insiderNormalized = insiderOpportunities.map(normalizeInsiderSignal)

    let merged = [...confluenceNormalized, ...insiderNormalized]

    // Apply source filter
    if (sourceFilter === 'confluence') {
      merged = merged.filter((s) => s.source === 'confluence')
    } else if (sourceFilter === 'insider') {
      merged = merged.filter((s) => s.source === 'insider')
    }

    // Sort by confidence descending, then by recency
    merged.sort((a, b) => {
      const confDiff = b.confidence - a.confidence
      if (confDiff !== 0) return confDiff
      const bTime = new Date(b.detected_at).getTime() || 0
      const aTime = new Date(a.detected_at).getTime() || 0
      return bTime - aTime
    })

    return merged
  }, [filteredSignals, insiderOpportunities, sourceFilter])

  const insiderCount = insiderOpportunities.length
  const confluenceCount = filteredSignals.length
  const totalSignalCount = insiderCount + confluenceCount

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
          {/* Filters */}
          <div className="flex flex-wrap items-center gap-4 p-3 bg-card rounded-lg border border-border">
            <Filter className="w-4 h-4 text-muted-foreground/70" />

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground/70">Source:</span>
              <select
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value as SourceFilter)}
                className="bg-muted border border-border rounded px-2 py-1 text-sm"
              >
                <option value="all">All ({totalSignalCount})</option>
                <option value="confluence">Confluence ({confluenceCount})</option>
                <option value="insider">Insider ({insiderCount})</option>
              </select>
            </div>

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

          {/* Stats Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground/70">Total Signals</p>
              <p className="text-lg font-semibold text-foreground">
                {unifiedSignals.length}
                <span className="text-muted-foreground/50 text-sm ml-1">
                  ({confluenceCount}c + {insiderCount}i)
                </span>
              </p>
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
              <p className="text-xs text-muted-foreground/70">Insider Signals</p>
              <p className="text-lg font-semibold text-purple-400">{insiderCount}</p>
            </div>
          </div>

          {/* Unified Signal List */}
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground/70" />
            </div>
          ) : unifiedSignals.length === 0 ? (
            <div className="text-center py-12 bg-card rounded-lg border border-border">
              <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-muted-foreground">No trader signals found</p>
              <p className="text-sm text-muted-foreground/50 mt-1">
                Try lowering tier/side/source filters or wait for new signals
              </p>
            </div>
          ) : viewMode === 'terminal' ? (
            <TraderSignalTerminal
              signals={unifiedSignals}
              onNavigateToWallet={onNavigateToWallet}
              totalCount={unifiedSignals.length}
            />
          ) : viewMode === 'list' ? (
            <TraderSignalTable
              signals={unifiedSignals}
              onNavigateToWallet={onNavigateToWallet}
            />
          ) : (
            <TraderSignalCards
              signals={unifiedSignals}
              onNavigateToWallet={onNavigateToWallet}
            />
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
