import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Activity,
  AlertCircle,
  Bell,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Filter,
  FolderPlus,
  Layers,
  RefreshCw,
  Settings,
  Target,
  Trash2,
  Users,
  Wallet,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  discoveryApi,
  type TraderGroup,
  type TraderGroupSuggestion,
} from '../services/discoveryApi'
import {
  getRecentTradesFromWallets,
  getOpportunities,
  getDiscoverySettings,
  updateDiscoverySettings,
  type DiscoverySettings,
  type Opportunity,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import {
  type UnifiedTraderSignal,
  normalizeTraderOpportunity,
  TraderSignalCards,
  TraderSignalTable,
  TraderSignalTerminal,
} from './TraderSignalViews'
import TraderOpportunitiesSettingsFlyout, {
  type TraderOpportunitiesSettingsForm,
} from './TraderOpportunitiesSettingsFlyout'
import { Button } from './ui/button'
import { Separator } from './ui/separator'

interface Props {
  onNavigateToWallet?: (address: string) => void
  onOpenCopilot?: (contextType?: string, contextId?: string, label?: string) => void
  mode?: 'full' | 'management' | 'opportunities'
  managementVariant?: 'default' | 'groups'
  viewMode?: 'card' | 'list' | 'terminal'
  showSettingsButton?: boolean
  onAnalyzeTargetsChange?: (targets: { visibleIds: string[]; allIds: string[] }) => void
  onSignalStatsChange?: (stats: {
    scannedCount: number
    executableCount: number
    filteredCount: number
  }) => void
}

type TierFilter = 'WATCH' | 'HIGH' | 'EXTREME'
type TraderSourceScopeFilter = 'tracked' | 'group' | 'pool' | 'other'
type TraderSourceScopeFilterState = Record<TraderSourceScopeFilter, boolean>
const CONFLUENCE_FETCH_LIMIT_MAX = 200
const ITEMS_PER_PAGE = 20
const TRADER_SOURCE_SCOPE_OPTIONS: Array<{
  key: TraderSourceScopeFilter
  labelKey: string
}> = [
  { key: 'tracked', labelKey: 'recentTradesPanel.sourceScope.tracked' },
  { key: 'group', labelKey: 'recentTradesPanel.sourceScope.group' },
  { key: 'pool', labelKey: 'recentTradesPanel.sourceScope.pool' },
  { key: 'other', labelKey: 'recentTradesPanel.sourceScope.other' },
]
const DEFAULT_TRADER_SOURCE_SCOPE_FILTER: TraderSourceScopeFilterState = {
  tracked: true,
  group: true,
  pool: true,
  other: true,
}

const DEFAULT_TRADER_OPPORTUNITY_SETTINGS: TraderOpportunitiesSettingsForm = {
  confluence_limit: 50,
  individual_trade_limit: 40,
  individual_trade_min_confidence: 0.62,
  individual_trade_max_age_minutes: 180,
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

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

type TFunction = (key: string, options?: Record<string, unknown>) => string

function formatTimeAgo(value: string | number | null | undefined, t: TFunction): string {
  const date = safeParseTime(value)
  if (!date) return t('recentTradesPanel.time.unknown')

  const now = Date.now()
  const diffMs = now - date.getTime()
  if (diffMs < 0) return t('recentTradesPanel.time.justNow')

  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return t('recentTradesPanel.time.justNow')
  if (diffMins < 60) return t('recentTradesPanel.time.minutesAgo', { n: diffMins })

  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return t('recentTradesPanel.time.hoursAgo', { n: diffHours })
  if (diffHours < 48) return t('recentTradesPanel.time.yesterday')
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function shortAddress(address: string, t: TFunction): string {
  if (!address) return t('recentTradesPanel.unknownAddress')
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

function isQualifiedTraderSignal(signal: UnifiedTraderSignal): boolean {
  return signal.is_tradeable && signal.is_valid
}

function signalDetectedTimestamp(signal: UnifiedTraderSignal): number {
  return safeParseTime(signal.last_seen_at || signal.detected_at)?.getTime() || 0
}

function hasTrackedSource(signal: UnifiedTraderSignal): boolean {
  return Boolean(signal.source_flags?.from_tracked_traders)
}

function hasGroupSource(signal: UnifiedTraderSignal): boolean {
  return Boolean(signal.source_flags?.from_trader_groups)
}

function hasPoolSource(signal: UnifiedTraderSignal): boolean {
  return Boolean(signal.source_flags?.from_pool)
}

function hasOtherSource(signal: UnifiedTraderSignal): boolean {
  return !hasTrackedSource(signal) && !hasGroupSource(signal) && !hasPoolSource(signal)
}

function signalMatchesSourceScopeFilters(
  signal: UnifiedTraderSignal,
  filters: TraderSourceScopeFilterState,
): boolean {
  if (filters.tracked && hasTrackedSource(signal)) return true
  if (filters.group && hasGroupSource(signal)) return true
  if (filters.pool && hasPoolSource(signal)) return true
  if (filters.other && hasOtherSource(signal)) return true
  return false
}

function signalPriority(signal: UnifiedTraderSignal): number {
  const sourceCoverage = signal.source_coverage_score * 1000
  const confidence = signal.confidence * 10
  const tier = signal.source === 'confluence' ? tierRank(signal.tier) * 25 : 0
  const recency = signalDetectedTimestamp(signal) / 1_000_000_000
  return sourceCoverage + confidence + tier + recency
}

export default function RecentTradesPanel({
  onNavigateToWallet,
  onOpenCopilot,
  mode = 'full',
  managementVariant = 'default',
  viewMode = 'card',
  showSettingsButton = true,
  onAnalyzeTargetsChange,
  onSignalStatsChange,
}: Props) {
  const { t } = useTranslation()
  const showManagement = mode !== 'opportunities'
  const showOpportunities = mode !== 'management'
  const groupsOnlyManagement = showManagement && !showOpportunities && managementVariant === 'groups'

  const [hoursFilter] = useState(24)
  const [signalLimit, setSignalLimit] = useState(DEFAULT_TRADER_OPPORTUNITY_SETTINGS.confluence_limit)
  const [individualTradeLimit, setIndividualTradeLimit] = useState(
    DEFAULT_TRADER_OPPORTUNITY_SETTINGS.individual_trade_limit,
  )
  const [individualTradeMinConfidence, setIndividualTradeMinConfidence] = useState(
    DEFAULT_TRADER_OPPORTUNITY_SETTINGS.individual_trade_min_confidence,
  )
  const [individualTradeMaxAgeMinutes, setIndividualTradeMaxAgeMinutes] = useState(
    DEFAULT_TRADER_OPPORTUNITY_SETTINGS.individual_trade_max_age_minutes,
  )
  const [liveAlerts, setLiveAlerts] = useState<LiveSignalAlert[]>([])
  const [groupName, setGroupName] = useState('')
  const [groupDescription, setGroupDescription] = useState('')
  const [groupWalletInput, setGroupWalletInput] = useState('')
  const [groupStatusMessage, setGroupStatusMessage] = useState<string | null>(null)
  const [showGroupForm, setShowGroupForm] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showFilteredSignals, setShowFilteredSignals] = useState(false)
  const [sourceScopeFilter, setSourceScopeFilter] = useState<TraderSourceScopeFilterState>(
    () => ({ ...DEFAULT_TRADER_SOURCE_SCOPE_FILTER }),
  )
  const [currentPage, setCurrentPage] = useState(0)
  const [settingsSaveMessage, setSettingsSaveMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  const queryClient = useQueryClient()
  const { isConnected, lastMessage } = useWebSocket('/ws', ['tracked_trader_signal'])

  const { data: discoverySettings } = useQuery({
    queryKey: ['settings-discovery'],
    queryFn: getDiscoverySettings,
    enabled: showOpportunities,
    staleTime: 60_000,
  })

  const invalidateTrackedManagementQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['wallets'] })
    queryClient.invalidateQueries({ queryKey: ['recent-trades-from-wallets'] })
    queryClient.invalidateQueries({ queryKey: ['trader-groups'] })
    queryClient.invalidateQueries({ queryKey: ['trader-group-suggestions'] })
    queryClient.invalidateQueries({ queryKey: ['opportunities', 'traders'] })
    queryClient.invalidateQueries({ queryKey: ['traders-overview'] })
  }, [queryClient])

  const saveTraderSettingsMutation = useMutation({
    mutationFn: (payload: Partial<DiscoverySettings>) => updateDiscoverySettings(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      queryClient.invalidateQueries({ queryKey: ['settings-discovery'] })
      setSettingsSaveMessage({ type: 'success', text: t('recentTradesPanel.toast.settingsSaved') })
      setTimeout(() => setSettingsSaveMessage(null), 3000)
      setSettingsOpen(false)
    },
    onError: (error: unknown) => {
      const message =
        (error as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (error as { message?: string })?.message
        || t('recentTradesPanel.toast.settingsSaveFailed')
      setSettingsSaveMessage({ type: 'error', text: message })
      setTimeout(() => setSettingsSaveMessage(null), 5000)
    },
  })

  const sourceScopeFilterActive = TRADER_SOURCE_SCOPE_OPTIONS.some(
    (option) => !sourceScopeFilter[option.key],
  )
  const confluenceFetchLimit = showFilteredSignals || sourceScopeFilterActive
    ? CONFLUENCE_FETCH_LIMIT_MAX
    : Math.min(CONFLUENCE_FETCH_LIMIT_MAX, signalLimit)

  const {
    data: confluenceOpportunities = [],
    isLoading: opportunitiesLoading,
    refetch: refetchSignals,
    isRefetching: isRefetchingSignals,
  } = useQuery({
    queryKey: [
      'opportunities',
      'traders',
      confluenceFetchLimit,
      showFilteredSignals,
    ],
    queryFn: async (): Promise<Opportunity[]> => {
      const result = await getOpportunities({
        source: 'traders',
        limit: confluenceFetchLimit,
        offset: 0,
        include_price_history: true,
      })
      return result.opportunities
    },
    refetchInterval: isConnected ? false : 30000,
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
    enabled: showManagement,
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
        t('recentTradesPanel.toast.groupCreated', {
          members: result.group?.member_count ?? 0,
          tracked: result.tracked_members,
        }),
      )
      setGroupName('')
      setGroupDescription('')
      setGroupWalletInput('')
      invalidateTrackedManagementQueries()
    },
    onError: (error: unknown) => {
      const message =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        || t('recentTradesPanel.toast.createGroupFailed')
      setGroupStatusMessage(message)
    },
  })

  const trackGroupMembersMutation = useMutation({
    mutationFn: (groupId: string) => discoveryApi.trackTraderGroupMembers(groupId),
    onSuccess: (result) => {
      setGroupStatusMessage(
        t('recentTradesPanel.toast.trackingRefreshed', { n: result.tracked_members }),
      )
      invalidateTrackedManagementQueries()
    },
    onError: () => setGroupStatusMessage(t('recentTradesPanel.toast.trackMembersFailed')),
  })

  const deleteGroupMutation = useMutation({
    mutationFn: (groupId: string) => discoveryApi.deleteTraderGroup(groupId),
    onSuccess: () => {
      setGroupStatusMessage(t('recentTradesPanel.toast.groupDeleted'))
      invalidateTrackedManagementQueries()
    },
    onError: () => setGroupStatusMessage(t('recentTradesPanel.toast.deleteGroupFailed')),
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
        String(lastMessage.data.market_id || t('recentTradesPanel.unknownMarket')),
      tier,
      conviction,
      outcome: lastMessage.data.outcome || null,
    }

    setLiveAlerts((prev) => [alert, ...prev].slice(0, 4))
    const timer = setTimeout(() => {
      setLiveAlerts((prev) => prev.filter((item) => item.id !== alert.id))
    }, 8000)

    return () => clearTimeout(timer)
  }, [lastMessage, showOpportunities, t])

  useEffect(() => {
    if (!showOpportunities || !discoverySettings) return

    setSignalLimit(
      Math.round(clampNumber(discoverySettings.trader_opps_confluence_limit || 50, 1, 200)),
    )
    setIndividualTradeLimit(
      Math.round(clampNumber(discoverySettings.trader_opps_insider_limit || 40, 1, 500)),
    )
    setIndividualTradeMinConfidence(
      clampNumber(discoverySettings.trader_opps_insider_min_confidence || 0.62, 0, 1),
    )
    setIndividualTradeMaxAgeMinutes(
      Math.round(clampNumber(discoverySettings.trader_opps_insider_max_age_minutes || 180, 1, 1440)),
    )
  }, [showOpportunities, discoverySettings])

  useEffect(() => {
    if (!showOpportunities) return
    setCurrentPage(0)
  }, [showOpportunities, signalLimit, showFilteredSignals, sourceScopeFilter])

  useEffect(() => {
    const handleOpenSettings = () => setSettingsOpen(true)
    window.addEventListener('open-trader-opportunities-settings', handleOpenSettings as EventListener)
    return () => window.removeEventListener('open-trader-opportunities-settings', handleOpenSettings as EventListener)
  }, [])

  const rawTrades = rawTradesData?.trades || []
  const opportunities = confluenceOpportunities
  const isLoading = opportunitiesLoading || (showManagement && rawTradesLoading)
  const isRefetching =
    isRefetchingSignals
    || (showManagement && isRefetchingRawTrades)

  const sortedConfluenceSignals = useMemo(() => {
    return opportunities.map(normalizeTraderOpportunity).sort((a, b) => {
      const convictionDiff = (b.confidence || 0) - (a.confidence || 0)
      if (convictionDiff !== 0) return convictionDiff
      const bTime = safeParseTime(b.last_seen_at || b.detected_at)?.getTime() || 0
      const aTime = safeParseTime(a.last_seen_at || a.detected_at)?.getTime() || 0
      return bTime - aTime
    })
  }, [opportunities])
  const trackedWalletsFromSignals = useMemo(() => {
    const addresses = new Set<string>()
    for (const signal of sortedConfluenceSignals) {
      const wallets = Array.isArray(signal.wallets) ? signal.wallets : []
      for (const rawWallet of wallets) {
        const rawAddress =
          rawWallet && typeof rawWallet === 'object'
            ? String((rawWallet as { address?: string }).address || '')
            : String(rawWallet || '')
        const address = rawAddress.trim().toLowerCase()
        if (address.startsWith('0x')) addresses.add(address)
      }
    }
    return addresses.size
  }, [sortedConfluenceSignals])
  const trackedWallets = rawTradesData?.tracked_wallets || trackedWalletsFromSignals

  const toggleSourceScopeFilter = (scope: TraderSourceScopeFilter) => {
    setSourceScopeFilter((previous) => {
      const enabledCount = TRADER_SOURCE_SCOPE_OPTIONS.reduce(
        (sum, option) => sum + (previous[option.key] ? 1 : 0),
        0,
      )
      if (previous[scope] && enabledCount <= 1) {
        return previous
      }
      return {
        ...previous,
        [scope]: !previous[scope],
      }
    })
  }

  const clearSourceScopeFilter = () => {
    setSourceScopeFilter({ ...DEFAULT_TRADER_SOURCE_SCOPE_FILTER })
  }

  const signalView = useMemo(() => {
    const allScannedSignals = [...sortedConfluenceSignals]
      .sort((a, b) => signalPriority(b) - signalPriority(a))
    const scopedScannedSignals = allScannedSignals
      .filter((signal) => signalMatchesSourceScopeFilters(signal, sourceScopeFilter))

    const executableCandidates = scopedScannedSignals
      .filter(isQualifiedTraderSignal)
      .sort((a, b) => signalPriority(b) - signalPriority(a))
      .slice(0, signalLimit)
    const executableSignals = executableCandidates
    const visibleSignals = showFilteredSignals ? scopedScannedSignals : executableSignals

    return {
      allScannedSignals,
      scannedSignals: scopedScannedSignals,
      executableSignals,
      visibleSignals,
      visibleTrackedCount: visibleSignals.filter(hasTrackedSource).length,
      visibleGroupCount: visibleSignals.filter(hasGroupSource).length,
      visiblePoolCount: visibleSignals.filter(hasPoolSource).length,
      visibleOtherCount: visibleSignals.filter(hasOtherSource).length,
      sourceScopedOut: Math.max(0, allScannedSignals.length - scopedScannedSignals.length),
      filteredOut: Math.max(0, scopedScannedSignals.length - executableSignals.length),
    }
  }, [
    sortedConfluenceSignals,
    sourceScopeFilter,
    signalLimit,
    showFilteredSignals,
  ])

  const unifiedSignals = signalView.visibleSignals
  const executableSignals = signalView.executableSignals
  const filteredOutSignals = signalView.filteredOut
  const sourceScopedOutSignals = signalView.sourceScopedOut

  const handleOpenSignalCopilot = (signal: UnifiedTraderSignal) => {
    const contextId = `${signal.source}:${signal.id}`
    const label = signal.market_question || signal.market_id
    onOpenCopilot?.('trader_signal', contextId, label)
  }

  const rawSignalCount = signalView.scannedSignals.length
  const displayedTrackedCount = signalView.visibleTrackedCount
  const displayedGroupCount = signalView.visibleGroupCount
  const displayedPoolCount = signalView.visiblePoolCount
  const displayedOtherCount = signalView.visibleOtherCount
  const executableSignalCount = executableSignals.length
  const totalPages = Math.ceil(unifiedSignals.length / ITEMS_PER_PAGE)
  const paginatedSignals = useMemo(() => {
    const start = currentPage * ITEMS_PER_PAGE
    return unifiedSignals.slice(start, start + ITEMS_PER_PAGE)
  }, [unifiedSignals, currentPage])
  const uniqueSignalMarkets = useMemo(() => {
    const keys = new Set<string>()
    for (const signal of unifiedSignals) {
      keys.add(String(signal.market_id || '').trim().toLowerCase())
    }
    return keys.size
  }, [unifiedSignals])
  const visibleSignalWalletCount = useMemo(() => {
    const addresses = new Set<string>()
    for (const signal of unifiedSignals) {
      const wallets = Array.isArray(signal.wallets) ? signal.wallets : []
      for (const wallet of wallets) {
        const address = String(wallet.address || '').trim().toLowerCase()
        if (address.startsWith('0x')) {
          addresses.add(address)
        }
      }
    }
    return addresses.size
  }, [unifiedSignals])
  const selectedSourceScopeLabels = useMemo(
    () =>
      TRADER_SOURCE_SCOPE_OPTIONS
        .filter((option) => sourceScopeFilter[option.key])
        .map((option) => t(option.labelKey))
        .join(', '),
    [sourceScopeFilter, t],
  )
  const highSignals = unifiedSignals.filter(
    (signal) => signal.source === 'confluence' && tierRank(signal.tier) >= tierRank('HIGH'),
  ).length
  const extremeSignals = unifiedSignals.filter(
    (signal) => signal.source === 'confluence' && toTier(signal.tier) === 'EXTREME',
  ).length
  const avgConviction = unifiedSignals.length
    ? unifiedSignals.reduce((sum, signal) => sum + signal.confidence, 0) /
      unifiedSignals.length
    : 0

  useEffect(() => {
    if (!showOpportunities) return
    const maxPage = Math.max(totalPages - 1, 0)
    if (currentPage > maxPage) {
      setCurrentPage(maxPage)
    }
  }, [showOpportunities, currentPage, totalPages])

  useEffect(() => {
    if (!showOpportunities) return
    onSignalStatsChange?.({
      scannedCount: rawSignalCount,
      executableCount: executableSignalCount,
      filteredCount: filteredOutSignals,
    })
  }, [
    showOpportunities,
    onSignalStatsChange,
    rawSignalCount,
    executableSignalCount,
    filteredOutSignals,
  ])

  // Stabilize analyze targets by comparing serialized IDs to avoid
  // infinite re-render loops (setState → parent re-render → new array
  // refs → effect fires again).
  const analyzeVisibleKey = useMemo(
    () => (showOpportunities ? paginatedSignals.map((s) => s.id).join(',') : ''),
    [showOpportunities, paginatedSignals]
  )
  const analyzeAllKey = useMemo(
    () => (showOpportunities ? unifiedSignals.map((s) => s.id).join(',') : ''),
    [showOpportunities, unifiedSignals]
  )
  useEffect(() => {
    if (!showOpportunities) {
      onAnalyzeTargetsChange?.({ visibleIds: [], allIds: [] })
      return
    }
    onAnalyzeTargetsChange?.({
      visibleIds: analyzeVisibleKey ? analyzeVisibleKey.split(',') : [],
      allIds: analyzeAllKey ? analyzeAllKey.split(',') : [],
    })
  }, [showOpportunities, onAnalyzeTargetsChange, analyzeVisibleKey, analyzeAllKey])

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
    if (showManagement) {
      refetchRawTrades()
    }
    if (showOpportunities) {
      refetchSignals()
    }
    if (showManagement) {
      refetchGroups()
      refetchSuggestions()
    }
  }

  const handleSaveTraderOpportunitySettings = useCallback((
    next: TraderOpportunitiesSettingsForm,
  ) => {
    if (!discoverySettings) {
      setSettingsSaveMessage({
        type: 'error',
        text: t('recentTradesPanel.toast.discoveryLoading'),
      })
      setTimeout(() => setSettingsSaveMessage(null), 4000)
      return
    }

    setSignalLimit(next.confluence_limit)
    setIndividualTradeLimit(next.individual_trade_limit)
    setIndividualTradeMinConfidence(next.individual_trade_min_confidence)
    setIndividualTradeMaxAgeMinutes(next.individual_trade_max_age_minutes)

    saveTraderSettingsMutation.mutate({
      ...discoverySettings,
      trader_opps_confluence_limit: next.confluence_limit,
      trader_opps_insider_limit: next.individual_trade_limit,
      trader_opps_insider_min_confidence: next.individual_trade_min_confidence,
      trader_opps_insider_max_age_minutes: next.individual_trade_max_age_minutes,
    })
  }, [discoverySettings, saveTraderSettingsMutation, t])

  const handleCloseSettings = useCallback(() => {
    setSettingsOpen(false)
  }, [])

  const traderOpportunitySettingsInitial = useMemo(
    () => ({
      confluence_limit: signalLimit,
      individual_trade_limit: individualTradeLimit,
      individual_trade_min_confidence: individualTradeMinConfidence,
      individual_trade_max_age_minutes: individualTradeMaxAgeMinutes,
    }),
    [
      individualTradeLimit,
      individualTradeMaxAgeMinutes,
      individualTradeMinConfidence,
      signalLimit,
    ],
  )

  const handleCreateManualGroup = () => {
    const name = groupName.trim()
    const walletAddresses = parseWalletInput(groupWalletInput)
    if (!name) {
      setGroupStatusMessage(t('recentTradesPanel.toast.groupNameRequired'))
      return
    }
    if (walletAddresses.length === 0) {
      setGroupStatusMessage(t('recentTradesPanel.toast.walletRequired'))
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
      {mode !== 'opportunities' && (
        <div className="rounded-xl border border-border/40 bg-card/60 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              <div className="p-2 bg-orange-500/10 rounded-lg shrink-0">
                <Zap className="w-5 h-5 text-orange-500" />
              </div>
              <div className="min-w-0">
                <h2 className="text-lg font-semibold text-foreground truncate">
                  {showOpportunities && showManagement
                    ? t('recentTradesPanel.header.title.traders')
                    : showOpportunities
                      ? t('recentTradesPanel.header.title.opportunities')
                      : groupsOnlyManagement
                        ? t('recentTradesPanel.header.title.groups')
                        : t('recentTradesPanel.header.title.management')}
                </h2>
                <p className="text-sm text-muted-foreground/70 truncate">
                  {showOpportunities && showManagement
                    ? t('recentTradesPanel.header.subtitle.traders')
                    : showOpportunities
                      ? t('recentTradesPanel.header.subtitle.opportunities')
                      : groupsOnlyManagement
                        ? t('recentTradesPanel.header.subtitle.groups')
                        : t('recentTradesPanel.header.subtitle.management')}
                </p>
              </div>
            </div>

            <button
              onClick={handleRefresh}
              disabled={isRefetching}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-card px-2.5 py-1.5 text-xs text-muted-foreground',
                'hover:text-foreground hover:bg-muted/60 transition-colors',
                isRefetching && 'opacity-50',
              )}
            >
              <RefreshCw className={cn('w-3.5 h-3.5', isRefetching && 'animate-spin')} />
              {t('recentTradesPanel.refresh')}
            </button>
          </div>
        </div>
      )}

      {showManagement && (
        <div className="rounded-lg border border-border bg-card/60 p-4 space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-foreground">
              {groupsOnlyManagement
                ? t('recentTradesPanel.section.traderGroups')
                : t('recentTradesPanel.section.trackedTraders')}
            </h3>
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
            {t('recentTradesPanel.manualGroup')}
          </button>
        </div>

        <div className={cn('grid gap-3', groupsOnlyManagement ? 'grid-cols-1 sm:grid-cols-3' : 'grid-cols-2 sm:grid-cols-4')}>
          {!groupsOnlyManagement && (
            <div className="rounded-md border border-border bg-background/40 px-3 py-2">
              <p className="text-[11px] text-muted-foreground/70">{t('recentTradesPanel.stat.trackedWallets')}</p>
              <p className="text-sm font-semibold text-foreground">{trackedWallets}</p>
            </div>
          )}
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">{t('recentTradesPanel.stat.traderGroups')}</p>
            <p className="text-sm font-semibold text-foreground">{traderGroups.length}</p>
          </div>
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">{t('recentTradesPanel.stat.groupMembers')}</p>
            <p className="text-sm font-semibold text-foreground">{totalGroupMembers}</p>
          </div>
          <div className="rounded-md border border-border bg-background/40 px-3 py-2">
            <p className="text-[11px] text-muted-foreground/70">{t('recentTradesPanel.stat.recentTradesHours', { n: hoursFilter })}</p>
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
                placeholder={t('recentTradesPanel.form.groupNamePlaceholder')}
                className="bg-muted border border-border rounded px-2 py-1.5 text-sm"
              />
              <input
                type="text"
                value={groupDescription}
                onChange={(e) => setGroupDescription(e.target.value)}
                placeholder={t('recentTradesPanel.form.descriptionPlaceholder')}
                className="bg-muted border border-border rounded px-2 py-1.5 text-sm"
              />
            </div>
            <textarea
              value={groupWalletInput}
              onChange={(e) => setGroupWalletInput(e.target.value)}
              placeholder={t('recentTradesPanel.form.walletsPlaceholder')}
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
                {t('recentTradesPanel.createTrack')}
              </button>
            </div>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-muted-foreground/70" />
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              {t('recentTradesPanel.section.existingGroups')}
            </p>
          </div>
          {groupsLoading ? (
            <div className="rounded-md border border-border bg-background/40 p-3 text-sm text-muted-foreground/70">
              {t('recentTradesPanel.loading.groups')}
            </div>
          ) : traderGroups.length === 0 ? (
            <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
              {t('recentTradesPanel.empty.noGroups')}
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
                        {t('recentTradesPanel.group.memberCount', { n: group.member_count })}
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
                        {t('recentTradesPanel.action.track')}
                      </button>
                      <button
                        onClick={() => deleteGroupMutation.mutate(group.id)}
                        disabled={deleteGroupMutation.isPending}
                        className="inline-flex items-center gap-1 rounded-md bg-red-500/15 px-2 py-1 text-[11px] text-red-300 hover:bg-red-500/25"
                      >
                        <Trash2 className="w-3 h-3" />
                        {t('recentTradesPanel.action.delete')}
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
                          {member.username || shortAddress(member.wallet_address, t)}
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
              {t('recentTradesPanel.section.suggestedGroups')}
            </p>
          </div>
          {suggestionsLoading ? (
            <div className="rounded-md border border-border bg-background/40 p-3 text-sm text-muted-foreground/70">
              {t('recentTradesPanel.loading.suggestions')}
            </div>
          ) : groupSuggestions.length === 0 ? (
            <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
              {t('recentTradesPanel.empty.noSuggestions')}
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
                          {t('recentTradesPanel.suggestion.tradersCount', { n: suggestion.wallet_count })}
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
                        {suggestion.already_exists
                          ? t('recentTradesPanel.suggestion.created')
                          : t('recentTradesPanel.createTrack')}
                      </button>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground/70 line-clamp-2">
                      {suggestion.description}
                    </p>
                    <div className="mt-2 flex items-center gap-3 text-[11px] text-muted-foreground/70">
                      <span>{t('recentTradesPanel.suggestion.avgScore', { score: (suggestion.avg_composite_score ?? 0).toFixed(2) })}</span>
                      <span>{t('recentTradesPanel.suggestion.alreadyTracked', { n: trackedOverlap })}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {!groupsOnlyManagement && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-muted-foreground/70" />
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                {t('recentTradesPanel.section.activeTrackedTraders')}
              </p>
            </div>
            {trackedWalletActivity.length === 0 ? (
              <div className="rounded-md border border-dashed border-border bg-background/20 p-3 text-sm text-muted-foreground/70">
                {t('recentTradesPanel.empty.noActivity')}
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
                      {wallet.wallet_username || wallet.wallet_label || shortAddress(wallet.wallet_address, t)}
                    </p>
                    <p className="text-[11px] text-muted-foreground/70 font-mono">
                      {shortAddress(wallet.wallet_address, t)}
                    </p>
                    <div className="mt-2 flex items-center justify-between text-xs">
                      <span className="text-blue-300">{t('recentTradesPanel.tradesCount', { n: wallet.trade_count })}</span>
                      <span className="text-muted-foreground/70">
                        {wallet.latest_trade_at ? formatTimeAgo(wallet.latest_trade_at.toISOString(), t) : t('recentTradesPanel.time.unknown')}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      )}

      {showOpportunities && (
        <>
          {mode === 'opportunities' ? (
            <div className="flex flex-wrap items-center gap-2 p-3 rounded-xl border border-border/40 bg-card/40">
              <Filter className="w-4 h-4 text-muted-foreground/70" />

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{t('recentTradesPanel.filter.maxSignals')}</span>
                <select
                  value={signalLimit}
                  onChange={(e) => setSignalLimit(Number(e.target.value))}
                  className="h-8 rounded-md border border-border bg-card px-2 text-xs text-foreground"
                >
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value={200}>200</option>
                </select>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                {TRADER_SOURCE_SCOPE_OPTIONS.map((option) => (
                  <button
                    key={option.key}
                    onClick={() => toggleSourceScopeFilter(option.key)}
                    className={cn(
                      'inline-flex h-8 items-center rounded-md border px-2.5 text-xs transition-colors',
                      sourceScopeFilter[option.key]
                        ? 'border-primary/30 bg-primary/10 text-primary hover:bg-primary/15'
                        : 'border-border/60 bg-card text-muted-foreground hover:text-foreground hover:bg-muted/60',
                    )}
                  >
                    {t(option.labelKey)}
                  </button>
                ))}
                {sourceScopeFilterActive && (
                  <button
                    onClick={clearSourceScopeFilter}
                    className="inline-flex h-8 items-center rounded-md border border-border/60 bg-card px-2.5 text-xs text-muted-foreground transition-colors hover:text-foreground hover:bg-muted/60"
                  >
                    {t('recentTradesPanel.filter.allSources')}
                  </button>
                )}
              </div>

              <div className="ml-auto flex items-center gap-2">
                <button
                  onClick={() => setShowFilteredSignals((prev) => !prev)}
                  className={cn(
                    'inline-flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs transition-colors',
                    showFilteredSignals
                      ? 'border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20'
                      : 'border-border/60 bg-card text-muted-foreground hover:text-foreground hover:bg-muted/60',
                  )}
                >
                  {showFilteredSignals
                    ? t('recentTradesPanel.filter.hideFiltered')
                    : t('recentTradesPanel.filter.showFiltered')}
                </button>
                <button
                  onClick={handleRefresh}
                  disabled={isRefetching}
                  className={cn(
                    'inline-flex h-8 items-center gap-1.5 rounded-md border border-border/60 bg-card px-2.5 text-xs text-muted-foreground',
                    'hover:text-foreground hover:bg-muted/60 transition-colors',
                    isRefetching && 'opacity-50',
                  )}
                >
                  <RefreshCw className={cn('w-3.5 h-3.5', isRefetching && 'animate-spin')} />
                  {t('recentTradesPanel.refresh')}
                </button>
                {showSettingsButton && (
                  <button
                    onClick={() => setSettingsOpen(true)}
                    disabled={!discoverySettings}
                    className={cn(
                      'inline-flex h-8 items-center gap-1.5 rounded-md border border-border/60 bg-card px-2.5 text-xs text-muted-foreground',
                      'hover:text-orange-300 hover:bg-orange-500/10 hover:border-orange-500/40 transition-colors',
                      !discoverySettings && 'opacity-50 cursor-not-allowed',
                    )}
                  >
                    <Settings className="w-3.5 h-3.5" />
                    {t('recentTradesPanel.settings')}
                  </button>
                )}
              </div>
            </div>
          ) : (
            <>
              {/* Filters */}
              <div className="flex flex-wrap items-center gap-4 p-3 rounded-xl border border-border/40 bg-card/40">
                <Filter className="w-4 h-4 text-muted-foreground/70" />

                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground/70">{t('recentTradesPanel.filter.maxSignals')}</span>
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

                <div className="flex flex-wrap items-center gap-1.5">
                  {TRADER_SOURCE_SCOPE_OPTIONS.map((option) => (
                    <button
                      key={option.key}
                      onClick={() => toggleSourceScopeFilter(option.key)}
                      className={cn(
                        'inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-colors',
                        sourceScopeFilter[option.key]
                          ? 'border-primary/30 bg-primary/10 text-primary hover:bg-primary/15'
                          : 'border-border bg-muted text-muted-foreground hover:text-foreground hover:bg-accent',
                      )}
                    >
                      {t(option.labelKey)}
                    </button>
                  ))}
                  {sourceScopeFilterActive && (
                    <button
                      onClick={clearSourceScopeFilter}
                      className="inline-flex items-center rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground hover:bg-accent"
                    >
                      {t('recentTradesPanel.filter.allSources')}
                    </button>
                  )}
                </div>

                <button
                  onClick={() => setShowFilteredSignals((prev) => !prev)}
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors',
                    showFilteredSignals
                      ? 'border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20'
                      : 'border-border bg-muted text-muted-foreground hover:text-foreground hover:bg-accent',
                  )}
                >
                  {showFilteredSignals
                    ? t('recentTradesPanel.filter.hideFiltered')
                    : t('recentTradesPanel.filter.showFiltered')}
                </button>
              </div>

              {/* Stats Summary */}
              <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
                <div className="rounded-lg border border-border/40 bg-card/40 p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                    {showFilteredSignals
                      ? t('recentTradesPanel.summary.scannedSignals')
                      : t('recentTradesPanel.summary.executableSignals')}
                  </p>
                  <p className="text-lg font-semibold text-foreground">
                    {unifiedSignals.length}
                    <span className="text-muted-foreground/50 text-sm ml-1">
                      ({displayedTrackedCount}t / {displayedGroupCount}g / {displayedPoolCount}p)
                    </span>
                  </p>
                </div>
                <div className="rounded-lg border border-border/40 bg-card/40 p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{t('recentTradesPanel.summary.highExtreme')}</p>
                  <p className="text-lg font-semibold text-orange-400">
                    {highSignals}
                    <span className="text-muted-foreground/60 text-sm ml-1">/ {extremeSignals}</span>
                  </p>
                </div>
                <div className="rounded-lg border border-border/40 bg-card/40 p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{t('recentTradesPanel.summary.avgConviction')}</p>
                  <p className="text-lg font-semibold text-foreground">{avgConviction.toFixed(1)}</p>
                </div>
                <div className="rounded-lg border border-border/40 bg-card/40 p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{t('recentTradesPanel.summary.marketsWallets')}</p>
                  <p className="text-lg font-semibold text-foreground">
                    <span className="text-blue-400">{uniqueSignalMarkets}</span>
                    <span className="text-muted-foreground/50 mx-1">/</span>
                    <span className="text-orange-400">{visibleSignalWalletCount}</span>
                  </p>
                </div>
                <div className="rounded-lg border border-border/40 bg-card/40 p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{t('recentTradesPanel.summary.sourceScope')}</p>
                  <p className="text-sm font-semibold text-cyan-300">{selectedSourceScopeLabels}</p>
                  <p className="text-[10px] text-muted-foreground/70 mt-1">
                    {t('recentTradesPanel.summary.otherCount', { n: displayedOtherCount })}
                  </p>
                </div>
                <div className="rounded-lg border border-border/40 bg-card/40 p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{t('recentTradesPanel.summary.filteredOut')}</p>
                  <p className="text-lg font-semibold text-red-300">{filteredOutSignals}</p>
                  <p className="text-[10px] text-orange-300/80 mt-1">{t('recentTradesPanel.summary.scopeCount', { n: sourceScopedOutSignals })}</p>
                </div>
              </div>
            </>
          )}

          {/* Unified Signal List */}
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground/70" />
            </div>
          ) : unifiedSignals.length === 0 ? (
            <div className="text-center py-12 bg-card rounded-lg border border-border">
              <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {showFilteredSignals
                  ? t('recentTradesPanel.empty.noScannedSignals')
                  : t('recentTradesPanel.empty.noExecutableSignals')}
              </p>
              <p className="text-sm text-muted-foreground/50 mt-1">
                {sourceScopeFilterActive
                  ? t('recentTradesPanel.empty.noScopeMatch', { labels: selectedSourceScopeLabels })
                  : showFilteredSignals
                  ? t('recentTradesPanel.empty.noConfluence')
                  : filteredOutSignals > 0
                    ? t('recentTradesPanel.empty.filteredFromExec', { n: filteredOutSignals })
                    : t('recentTradesPanel.empty.noQualified')}
              </p>
            </div>
          ) : viewMode === 'terminal' ? (
            <TraderSignalTerminal
              signals={paginatedSignals}
              onNavigateToWallet={onNavigateToWallet}
              onOpenCopilot={handleOpenSignalCopilot}
              totalCount={unifiedSignals.length}
            />
          ) : viewMode === 'list' ? (
            <TraderSignalTable
              signals={paginatedSignals}
              onNavigateToWallet={onNavigateToWallet}
              onOpenCopilot={handleOpenSignalCopilot}
            />
          ) : (
            <TraderSignalCards
              signals={paginatedSignals}
              onNavigateToWallet={onNavigateToWallet}
              onOpenCopilot={handleOpenSignalCopilot}
            />
          )}

          {unifiedSignals.length > 0 && (
            <div className="mt-5">
              <Separator />
              <div className="flex items-center justify-between pt-4">
                <div className="text-xs text-muted-foreground">
                  {t('recentTradesPanel.pagination.range', {
                    from: currentPage * ITEMS_PER_PAGE + 1,
                    to: Math.min((currentPage + 1) * ITEMS_PER_PAGE, unifiedSignals.length),
                    total: unifiedSignals.length,
                  })}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                    disabled={currentPage === 0}
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    {t('recentTradesPanel.pagination.prev')}
                  </Button>
                  <span className="px-2.5 py-1 bg-card rounded-lg text-xs border border-border font-mono">
                    {currentPage + 1}/{totalPages || 1}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setCurrentPage((p) => p + 1)}
                    disabled={currentPage >= totalPages - 1}
                  >
                    {t('recentTradesPanel.pagination.next')}
                    <ChevronRight className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {showOpportunities && mode === 'opportunities' && (
        <TraderOpportunitiesSettingsFlyout
          isOpen={settingsOpen}
          onClose={handleCloseSettings}
          initial={traderOpportunitySettingsInitial}
          onSave={handleSaveTraderOpportunitySettings}
          savePending={saveTraderSettingsMutation.isPending}
          saveMessage={settingsSaveMessage}
        />
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
                    {alert.outcome
                      ? t('recentTradesPanel.alert.titleWithOutcome', { tier: alert.tier, outcome: alert.outcome })
                      : t('recentTradesPanel.alert.title', { tier: alert.tier })}
                  </p>
                  <p className="text-xs mt-0.5 line-clamp-2">{alert.market}</p>
                  <p className="text-[11px] mt-1 opacity-90">
                    {t('recentTradesPanel.alert.conviction', { n: alert.conviction })}
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
