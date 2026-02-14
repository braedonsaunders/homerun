import { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Trophy,
  RefreshCw,
  Users,
  Target,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Copy,
  Search,
  TrendingUp,
  ExternalLink,
  Activity,
  UserPlus,
  PauseCircle,
  AlertTriangle,
  Ban,
  UserCheck,
  UserX,
  Trash2,
  X,
  Settings,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from './ui/table'
import {
  discoveryApi,
  type DiscoveredWallet,
  type TagInfo,
  type DiscoveryStats,
  type PoolStats,
  type PoolMember,
  type PoolMembersResponse,
} from '../services/discoveryApi'
import {
  analyzeAndTrackWallet,
  getDiscoverySettings,
  runWorkerOnce,
  updateDiscoverySettings,
  type DiscoverySettings,
} from '../services/api'
import PoolSettingsFlyout, { type PoolSettingsForm } from './PoolSettingsFlyout'

type SortField =
  | 'rank_score'
  | 'composite_score'
  | 'quality_score'
  | 'activity_score'
  | 'insider_score'
  | 'last_trade_at'
  | 'total_pnl'
  | 'win_rate'
  | 'sharpe_ratio'
  | 'total_trades'
  | 'avg_roi'

type SortDir = 'asc' | 'desc'
type RecommendationFilter = '' | 'copy_candidate' | 'monitor' | 'avoid'
type TimePeriod = '24h' | '7d' | '30d' | '90d' | 'all'

const TIME_PERIODS: { value: TimePeriod; label: string; description: string }[] = [
  { value: '24h', label: '24H', description: 'last 24 hours' },
  { value: '7d', label: '7D', description: 'last 7 days' },
  { value: '30d', label: '1M', description: 'last 30 days' },
  { value: '90d', label: '3M', description: 'last 90 days' },
  { value: 'all', label: 'All', description: 'all time' },
]

const RECOMMENDATION_COLORS: Record<string, string> = {
  copy_candidate: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/30 dark:bg-sky-500/15 dark:text-sky-200',
  monitor: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/30 dark:bg-amber-500/15 dark:text-amber-200',
  avoid: 'border-rose-300 bg-rose-100 text-rose-900 dark:border-rose-400/30 dark:bg-rose-500/15 dark:text-rose-200',
}

const RECOMMENDATION_LABELS: Record<string, string> = {
  copy_candidate: 'Copy Candidate',
  monitor: 'Monitor',
  avoid: 'Avoid',
}

const ITEMS_PER_PAGE = 25
const FILTER_LABEL_CLASS = 'text-[10px] uppercase tracking-wide text-muted-foreground/80'
const FILTER_INPUT_CLASS = 'h-8 text-xs bg-card border-border'
const FILTER_SELECT_CLASS = 'w-full bg-card border border-border rounded-lg px-2 py-1.5 text-xs h-8'
const SCORE_DELTA_HIDE_THRESHOLD = 0.0025

type MetricTone = 'good' | 'warn' | 'bad' | 'neutral' | 'info'

const METRIC_TONE_CLASSES: Record<MetricTone, string> = {
  good: 'border-sky-300 bg-sky-100 text-sky-900 dark:border-sky-400/35 dark:bg-sky-500/15 dark:text-sky-100',
  warn: 'border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-400/40 dark:bg-amber-500/18 dark:text-amber-100',
  bad: 'border-rose-300 bg-rose-100 text-rose-900 dark:border-rose-400/45 dark:bg-rose-500/18 dark:text-rose-100',
  neutral: 'border-slate-300 bg-slate-100 text-slate-800 dark:border-border/85 dark:bg-muted/55 dark:text-foreground/90',
  info: 'border-indigo-300 bg-indigo-100 text-indigo-900 dark:border-indigo-400/40 dark:bg-indigo-500/16 dark:text-indigo-100',
}

const METRIC_BAR_CLASSES: Record<MetricTone, string> = {
  good: 'bg-sky-600 dark:bg-sky-300/95',
  warn: 'bg-amber-600 dark:bg-amber-300/95',
  bad: 'bg-rose-600 dark:bg-rose-300/95',
  neutral: 'bg-slate-500 dark:bg-muted-foreground/80',
  info: 'bg-indigo-600 dark:bg-indigo-300/95',
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return clamp(value, 0, 1)
}

function formatScorePct(value: number): string {
  return `${(clamp01(value) * 100).toFixed(1)}%`
}

function scoreTone(value: number, goodThreshold: number, warnThreshold: number): MetricTone {
  if (!Number.isFinite(value)) return 'neutral'
  if (value >= goodThreshold) return 'good'
  if (value >= warnThreshold) return 'warn'
  return 'neutral'
}

function inverseScoreTone(value: number, badThreshold: number, warnThreshold: number): MetricTone {
  if (!Number.isFinite(value)) return 'neutral'
  if (value >= badThreshold) return 'bad'
  if (value >= warnThreshold) return 'warn'
  return 'good'
}

function poolTierTone(tier: string | null | undefined): MetricTone {
  const normalized = (tier || '').toLowerCase()
  if (normalized === 'core') return 'good'
  if (normalized === 'rising') return 'info'
  return 'neutral'
}

function selectionReasonTone(code: string): MetricTone {
  const normalized = code.toLowerCase()
  if (
    normalized.includes('manual_include') ||
    normalized.includes('core') ||
    normalized.includes('elite') ||
    normalized.includes('quality') ||
    normalized.includes('active') ||
    normalized.includes('tracked')
  ) {
    return 'good'
  }
  if (
    normalized.includes('exclude') ||
    normalized.includes('blacklist') ||
    normalized.includes('below') ||
    normalized.includes('blocked') ||
    normalized.includes('anomaly')
  ) {
    return 'bad'
  }
  if (
    normalized.includes('rising') ||
    normalized.includes('churn') ||
    normalized.includes('tier') ||
    normalized.includes('pending')
  ) {
    return 'warn'
  }
  return 'neutral'
}

function formatPnl(value: number): string {
  if (value == null) return '0.00'
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return value.toFixed(2)
}

function formatPercent(value: number): string {
  if (value == null) return '0.0%'
  return `${value.toFixed(1)}%`
}

function normalizePercentRatio(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.abs(value) <= 1 ? value * 100 : value
}

function formatWinRate(value: number): string {
  return `${normalizePercentRatio(value).toFixed(1)}%`
}

function formatNumber(value: number): string {
  if (value == null) return '0'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

function truncateAddress(address: string): string {
  if (!address || address.length < 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return 'Never'
  const diff = Date.now() - new Date(dateStr).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function MetricPill({
  label,
  value,
  tone = 'neutral',
  className,
  mono = true,
  title,
}: {
  label: string
  value: string
  tone?: MetricTone
  className?: string
  mono?: boolean
  title?: string
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-none',
        METRIC_TONE_CLASSES[tone],
        className
      )}
    >
      <span className="uppercase tracking-wide opacity-90">{label}</span>
      <span className={cn('font-semibold', mono && 'font-mono')}>{value}</span>
    </span>
  )
}

function ScoreSparkline({
  points,
  className,
}: {
  points: Array<{ key: string; label: string; value: number; tone?: MetricTone }>
  className?: string
}) {
  if (!points.length) return null
  return (
    <div className={cn('inline-flex items-end gap-0.5 rounded-md border border-slate-300 bg-slate-100 px-1.5 py-0.5 dark:border-border/85 dark:bg-muted/45', className)}>
      {points.map((point) => {
        const normalized = clamp01(point.value)
        const height = Math.max(3, Math.round(normalized * 13))
        const tone = point.tone || 'neutral'
        return (
          <span
            key={point.key}
            title={`${point.label} ${(normalized * 100).toFixed(1)}%`}
            className={cn('w-1.5 rounded-sm', METRIC_BAR_CLASSES[tone])}
            style={{ height }}
          />
        )
      })}
    </div>
  )
}

type SelectionReason = {
  code: string
  label: string
  detail?: string
}

const POOL_SELECTION_REASON_LIBRARY: Record<string, SelectionReason> = {
  manual_include: {
    code: 'manual_include',
    label: 'Manual include override',
    detail: 'Manually added to the pool.',
  },
  manual_exclude: {
    code: 'manual_exclude',
    label: 'Manual exclude',
    detail: 'Manually excluded from the pool.',
  },
  blacklisted: {
    code: 'blacklisted',
    label: 'Blacklisted',
    detail: 'Blacklisted from pool actions.',
  },
  tracked: {
    code: 'tracked',
    label: 'Tracked wallet',
    detail: 'Included from tracked-wallet workflows.',
  },
  core_quality_gate: {
    code: 'core_quality_gate',
    label: 'Core quality tier',
    detail: 'Passed core quality gates.',
  },
  rising_quality_gate: {
    code: 'rising_quality_gate',
    label: 'Rising quality tier',
    detail: 'Passed rising-tier quality and activity gates.',
  },
  churn_guard_retained: {
    code: 'churn_guard_retained',
    label: 'Churn guard retention',
    detail: 'Kept to limit hourly churn.',
  },
  elite_composite: {
    code: 'elite_composite',
    label: 'Elite composite profile',
    detail: 'Strong quality, activity, and stability profile.',
  },
  active_momentum: {
    code: 'active_momentum',
    label: 'Active momentum',
    detail: 'Recent trade velocity met the threshold.',
  },
  active_recent: {
    code: 'active_recent',
    label: 'Recent activity',
    detail: 'Traded within the active window.',
  },
  insider_alignment: {
    code: 'insider_alignment',
    label: 'Insider-aligned signal',
    detail: 'High insider score with recent activity.',
  },
  cluster_capped: {
    code: 'cluster_capped',
    label: 'Cluster-capped slot',
    detail: 'Included within cluster concentration limits.',
  },
  quality_gate_pass: {
    code: 'quality_gate_pass',
    label: 'Quality gate pass',
    detail: 'Passed quality eligibility checks.',
  },
  below_selection_cutoff: {
    code: 'below_selection_cutoff',
    label: 'Below pool cutoff',
    detail: 'Did not clear current selection cutoff.',
  },
  tier_thresholds_not_met: {
    code: 'tier_thresholds_not_met',
    label: 'Tier thresholds not met',
    detail: 'Core and rising thresholds were not met.',
  },
  non_positive_pnl: {
    code: 'non_positive_pnl',
    label: 'Non-positive PnL',
    detail: 'Total PnL must be positive.',
  },
  insufficient_trades: {
    code: 'insufficient_trades',
    label: 'Insufficient trade sample',
    detail: 'Needs more trade history.',
  },
  anomaly_too_high: {
    code: 'anomaly_too_high',
    label: 'Anomaly score too high',
    detail: 'Anomaly score exceeded the limit.',
  },
  recommendation_blocked: {
    code: 'recommendation_blocked',
    label: 'Recommendation blocked',
    detail: 'Recommendation is not pool-eligible.',
  },
  not_analyzed: {
    code: 'not_analyzed',
    label: 'Analysis missing',
    detail: 'No completed discovery analysis yet.',
  },
}

const FALLBACK_REASON: SelectionReason = {
  code: 'selection_reason_unknown',
  label: 'Selection reason unavailable',
  detail: 'Reason metadata missing for this recompute cycle.',
}

const BLOCKER_REASON_CODES = new Set<string>([
  'not_analyzed',
  'recommendation_blocked',
  'insufficient_trades',
  'anomaly_too_high',
  'non_positive_pnl',
  'tier_thresholds_not_met',
  'below_selection_cutoff',
])

function normalizeReasonCode(code: string): string {
  return code.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_')
}

function getReasonDefinition(code: string): SelectionReason | null {
  const normalized = code.trim().toLowerCase()
  if (!normalized) return null
  const exact = POOL_SELECTION_REASON_LIBRARY[normalized]
  if (exact) return exact
  const dashed = normalized.replace(/[^a-z0-9]+/g, '_')
  return POOL_SELECTION_REASON_LIBRARY[dashed] || null
}

function formatReasonLabel(rawCode: string): string {
  const clean = rawCode.trim()
  if (!clean) return 'Pool membership detail'
  return clean
    .replace(/_/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase())
}

function normalizeSelectionReason(raw: unknown): SelectionReason | null {
  if (typeof raw === 'string') {
    const code = raw.trim()
    if (!code) return null
    const def = getReasonDefinition(code)
    return {
      code,
      label: def?.label ?? formatReasonLabel(code),
      detail: def?.detail,
    }
  }

  if (!raw || typeof raw !== 'object') return null
  const reason = raw as { code?: unknown; label?: unknown; detail?: unknown }
  const providedCode = typeof reason.code === 'string' && reason.code.trim()
    ? reason.code.trim()
    : ''
  const labelFromObj =
    typeof reason.label === 'string' && reason.label.trim() ? reason.label.trim() : ''
  const code = providedCode || (labelFromObj ? `reason_${labelFromObj.toLowerCase().replace(/[^a-z0-9]+/g, '_')}` : '')
  if (!code) return null
  const def = getReasonDefinition(code)
  const label =
    labelFromObj
      ? labelFromObj
    : def?.label ?? formatReasonLabel(code)
  const detail =
    typeof reason.detail === 'string' && reason.detail.trim()
      ? reason.detail.trim()
      : def?.detail
  return { code, label, detail }
}

function enrichMissingReasonSignals(member: PoolMember): SelectionReason[] {
  const inferred: SelectionReason[] = []
  const seen = new Set<string>()
  const seenPush = (reason: SelectionReason | null) => {
    if (!reason) return
    const key = reason.code.toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    inferred.push(reason)
  }

  const breakdown: Record<string, number> | undefined = member.selection_breakdown
  const qualityScore = Number(breakdown?.quality_score || 0)
  const activityScore = Number(breakdown?.activity_score || 0)
  const stabilityScore = Number(breakdown?.stability_score || 0)
  const insiderScore = Number(breakdown?.insider_score || 0)

  if (insiderScore >= 0.62) {
    seenPush(POOL_SELECTION_REASON_LIBRARY.insider_alignment)
  }
  if (qualityScore >= 0.70 || member.total_pnl > 50_000) {
    seenPush({
      ...POOL_SELECTION_REASON_LIBRARY.quality_gate_pass,
      detail: `Quality score ${(qualityScore * 100).toFixed(1)}% met threshold.`,
    })
  }
  if (activityScore >= 0.55 || member.trades_24h >= 6 || member.trades_1h > 0) {
    seenPush(POOL_SELECTION_REASON_LIBRARY.active_momentum)
  }
  if (stabilityScore >= 0.65) {
    seenPush({
      ...POOL_SELECTION_REASON_LIBRARY.quality_gate_pass,
      label: 'Stable risk-adjusted profile',
      detail: 'Drawdown and returns were relatively stable.',
    })
  }

  return inferred
}

function getPoolSelectionReasons(member: PoolMember): SelectionReason[] {
  const out: SelectionReason[] = []
  const seen = new Set<string>()
  const normalizedRaw = Array.isArray(member.selection_reasons)
    ? member.selection_reasons.map(normalizeSelectionReason).filter((reason): reason is SelectionReason => !!reason)
    : []
  const normalized = member.in_top_pool
    ? normalizedRaw.filter(reason => !BLOCKER_REASON_CODES.has(normalizeReasonCode(reason.code)))
    : normalizedRaw

  for (const reason of normalized) {
    const key = reason.code.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(reason)
  }

  if (out.length > 0) {
    return out
  }

  const manualReason: SelectionReason | null =
    member.pool_flags?.manual_include
      ? { code: 'manual_include', label: 'Manual include', detail: 'Manually included in the pool.' }
      : null
  const blacklistedReason: SelectionReason | null =
    member.pool_flags?.blacklisted
      ? { code: 'blacklisted', label: 'Blacklisted', detail: 'Wallet is blacklisted from pool membership.' }
      : null
  const trackedReason: SelectionReason | null =
    member.tracked_wallet
      ? { code: 'tracked', label: 'Tracked wallet', detail: 'Included from tracked-wallet updates.' }
      : null

  const fallbackReason: SelectionReason | null = (() => {
    const reason = (member.pool_membership_reason || '').trim()
    if (!reason) return null
    if (member.in_top_pool && BLOCKER_REASON_CODES.has(normalizeReasonCode(reason))) {
      return null
    }
    const def = getReasonDefinition(reason)
    return {
      code: reason,
      label: def?.label ?? formatReasonLabel(reason),
      detail: def?.detail ?? FALLBACK_REASON.detail,
    }
  })()

  const candidates = [manualReason, blacklistedReason, trackedReason, fallbackReason].filter(
    (reason): reason is SelectionReason => !!reason
  )

  for (const reason of candidates) {
    const key = reason.code.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(reason)
  }

  for (const inferred of enrichMissingReasonSignals(member)) {
    const key = inferred.code.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(inferred)
  }

  if (!out.length) {
    const statusLabel = member.in_top_pool
      ? 'Top-pool candidate'
      : 'Pool membership pending'
    out.push({
      code: member.in_top_pool ? 'top_pool_member' : 'pool_pending',
      label: statusLabel,
      detail: member.in_top_pool
        ? FALLBACK_REASON.detail
        : 'Wallet is not currently in the active pool.',
    })
  }

  return out
}

interface DiscoveryPanelProps {
  onAnalyzeWallet?: (address: string, username?: string) => void
  view?: 'discovery' | 'pool'
}

export default function DiscoveryPanel({ onAnalyzeWallet, view = 'discovery' }: DiscoveryPanelProps) {
  const isPoolView = view === 'pool'
  const [sortBy, setSortBy] = useState<SortField>('rank_score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [currentPage, setCurrentPage] = useState(0)
  const [minTrades, setMinTrades] = useState(0)
  const [minPnl, setMinPnl] = useState(0)
  const [recommendationFilter, setRecommendationFilter] = useState<RecommendationFilter>('')
  const [marketCategoryFilter, setMarketCategoryFilter] = useState<'all' | 'politics' | 'sports' | 'crypto' | 'culture' | 'economics' | 'tech' | 'finance' | 'weather'>('all')
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('24h')
  const [minInsiderScore, setMinInsiderScore] = useState(0)
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [tagSearch, setTagSearch] = useState('')
  const [tagPicker, setTagPicker] = useState('')
  const [poolSearch, setPoolSearch] = useState('')
  const [poolTierFilter, setPoolTierFilter] = useState<'all' | 'core' | 'rising'>('all')
  const [minPoolWinRate, setMinPoolWinRate] = useState(0)
  const [poolSortBy, setPoolSortBy] = useState<'selection_score' | 'composite_score' | 'activity_score' | 'quality_score' | 'trades_24h' | 'trades_1h' | 'total_trades' | 'total_pnl' | 'win_rate' | 'last_trade_at'>('composite_score')
  const [poolSortDir, setPoolSortDir] = useState<'asc' | 'desc'>('desc')
  const [includeBlacklisted, setIncludeBlacklisted] = useState(true)
  const [manualPoolAddress, setManualPoolAddress] = useState('')
  const [poolSettingsOpen, setPoolSettingsOpen] = useState(false)
  const [poolSettingsSaveMessage, setPoolSettingsSaveMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  const queryClient = useQueryClient()

  const { data: discoverySettings } = useQuery({
    queryKey: ['settings-discovery'],
    queryFn: getDiscoverySettings,
    enabled: isPoolView,
  })

  useEffect(() => {
    setCurrentPage(0)
  }, [
    sortBy,
    sortDir,
    minTrades,
    minPnl,
    recommendationFilter,
    marketCategoryFilter,
    timePeriod,
    selectedTags,
    minInsiderScore,
  ])

  const { data: stats } = useQuery<DiscoveryStats>({
    queryKey: ['discovery-stats'],
    queryFn: discoveryApi.getDiscoveryStats,
    refetchInterval: 15000,
  })

  const { data: poolStats } = useQuery<PoolStats>({
    queryKey: ['discovery-pool-stats'],
    queryFn: discoveryApi.getPoolStats,
    refetchInterval: 30000,
    enabled: isPoolView,
  })

  const { data: poolMembersData, isLoading: poolMembersLoading, error: poolMembersError } = useQuery<PoolMembersResponse>({
    queryKey: [
      'discovery-pool-members',
      poolSearch,
      poolTierFilter,
      minPoolWinRate,
      poolSortBy,
      poolSortDir,
      includeBlacklisted,
    ],
    queryFn: async () => {
      const params = {
        limit: 300,
        offset: 0,
        pool_only: true,
        include_blacklisted: includeBlacklisted,
        tier: poolTierFilter === 'all' ? undefined : poolTierFilter,
        min_win_rate: minPoolWinRate > 0 ? minPoolWinRate : undefined,
        search: poolSearch.trim() || undefined,
        sort_by: poolSortBy,
        sort_dir: poolSortDir,
      } as const
      try {
        return await discoveryApi.getPoolMembers(params)
      } catch (error: any) {
        const status = error?.response?.status
        const detail = String(error?.response?.data?.detail || '')
        const unsupportedSortBy =
          status === 400
          && detail.toLowerCase().includes('invalid sort_by')
        if (unsupportedSortBy && poolSortBy !== 'composite_score') {
          return discoveryApi.getPoolMembers({
            ...params,
            sort_by: 'composite_score',
          })
        }
        throw error
      }
    },
    refetchInterval: 30000,
    enabled: isPoolView,
  })

  const { data: tags = [], isLoading: tagsLoading } = useQuery<TagInfo[]>({
    queryKey: ['discovery-tags'],
    queryFn: discoveryApi.getTags,
    refetchInterval: 120000,
    enabled: !isPoolView,
  })

  const selectedTagString = selectedTags.join(',')

  const { data: leaderboardData, isLoading: leaderboardLoading } = useQuery({
    queryKey: [
      'discovery-leaderboard',
      sortBy,
      sortDir,
      currentPage,
      minTrades,
      minPnl,
      recommendationFilter,
      marketCategoryFilter,
      selectedTagString,
      timePeriod,
      minInsiderScore,
    ],
    queryFn: () =>
      discoveryApi.getLeaderboard({
        sort_by: sortBy,
        sort_dir: sortDir,
        limit: ITEMS_PER_PAGE,
        offset: currentPage * ITEMS_PER_PAGE,
        min_trades: minTrades,
        min_pnl: minPnl || undefined,
        min_insider_score: minInsiderScore > 0 ? minInsiderScore : undefined,
        recommendation: recommendationFilter || undefined,
        market_category: marketCategoryFilter !== 'all' ? marketCategoryFilter : undefined,
        tags: selectedTagString || undefined,
        time_period: timePeriod !== 'all' ? timePeriod : undefined,
      }),
    refetchInterval: 30000,
    enabled: !isPoolView,
  })

  const wallets: DiscoveredWallet[] = leaderboardData?.wallets || leaderboardData || []
  const totalWallets: number = leaderboardData?.total || wallets.length
  const isWindowActive = timePeriod !== 'all' && !!leaderboardData?.window_key

  const trackWalletMutation = useMutation({
    mutationFn: (params: { address: string; username?: string | null }) =>
      analyzeAndTrackWallet({
        address: params.address,
        label:
          params.username ||
          `Discovered ${params.address.slice(0, 6)}...${params.address.slice(-4)}`,
        auto_copy: false,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
      queryClient.invalidateQueries({ queryKey: ['traders-overview'] })
      queryClient.invalidateQueries({ queryKey: ['tracked-trader-opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-pool-members'] })
    },
  })

  const invalidatePoolQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['discovery-pool-members'] })
    queryClient.invalidateQueries({ queryKey: ['discovery-pool-stats'] })
    queryClient.invalidateQueries({ queryKey: ['discovery-leaderboard'] })
  }, [queryClient])

  const manualIncludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.poolManualInclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const clearManualIncludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.clearPoolManualInclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const manualExcludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.poolManualExclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const clearManualExcludeMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.clearPoolManualExclude(address),
    onSuccess: invalidatePoolQueries,
  })
  const blacklistMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.blacklistPoolWallet(address),
    onSuccess: invalidatePoolQueries,
  })
  const unblacklistMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.unblacklistPoolWallet(address),
    onSuccess: invalidatePoolQueries,
  })
  const deletePoolWalletMutation = useMutation({
    mutationFn: (address: string) => discoveryApi.deletePoolWallet(address),
    onSuccess: invalidatePoolQueries,
  })
  const promoteTrackedMutation = useMutation({
    mutationFn: () => discoveryApi.promoteTrackedWalletsToPool(500),
    onSuccess: invalidatePoolQueries,
  })

  const savePoolSettingsMutation = useMutation({
    mutationFn: (payload: Partial<DiscoverySettings>) => updateDiscoverySettings(payload),
    onSuccess: async () => {
      setPoolSettingsSaveMessage({ type: 'success', text: 'Pool settings saved' })
      setTimeout(() => setPoolSettingsSaveMessage(null), 3000)
      queryClient.invalidateQueries({ queryKey: ['settings-discovery'] })
      try {
        await runWorkerOnce('tracked_traders')
      } catch {
        // Non-fatal: setting is persisted and will apply on next worker cycle.
      }
      invalidatePoolQueries()
      setPoolSettingsOpen(false)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      const message = typeof detail === 'string' && detail.trim()
        ? detail.trim()
        : 'Failed to save pool settings'
      setPoolSettingsSaveMessage({ type: 'error', text: message })
      setTimeout(() => setPoolSettingsSaveMessage(null), 5000)
    },
  })

  const handleSavePoolSettings = useCallback((next: PoolSettingsForm) => {
    if (!discoverySettings) {
      setPoolSettingsSaveMessage({
        type: 'error',
        text: 'Pool settings are not loaded yet',
      })
      setTimeout(() => setPoolSettingsSaveMessage(null), 4000)
      return
    }

    savePoolSettingsMutation.mutate({
      ...discoverySettings,
      pool_recompute_mode: next.pool_recompute_mode,
      pool_target_size: next.pool_target_size,
      pool_min_size: next.pool_min_size,
      pool_max_size: next.pool_max_size,
      pool_active_window_hours: next.pool_active_window_hours,
      pool_selection_score_floor: next.pool_selection_score_floor,
      pool_max_hourly_replacement_rate: next.pool_max_hourly_replacement_rate,
      pool_replacement_score_cutoff: next.pool_replacement_score_cutoff,
      pool_max_cluster_share: next.pool_max_cluster_share,
      pool_high_conviction_threshold: next.pool_high_conviction_threshold,
      pool_insider_priority_threshold: next.pool_insider_priority_threshold,
      pool_min_eligible_trades: next.pool_min_eligible_trades,
      pool_max_eligible_anomaly: next.pool_max_eligible_anomaly,
      pool_core_min_win_rate: next.pool_core_min_win_rate,
      pool_core_min_sharpe: next.pool_core_min_sharpe,
      pool_core_min_profit_factor: next.pool_core_min_profit_factor,
      pool_rising_min_win_rate: next.pool_rising_min_win_rate,
      pool_slo_min_analyzed_pct: next.pool_slo_min_analyzed_pct,
      pool_slo_min_profitable_pct: next.pool_slo_min_profitable_pct,
      pool_leaderboard_wallet_trade_sample: next.pool_leaderboard_wallet_trade_sample,
      pool_incremental_wallet_trade_sample: next.pool_incremental_wallet_trade_sample,
      pool_full_sweep_interval_seconds: next.pool_full_sweep_interval_seconds,
      pool_incremental_refresh_interval_seconds: next.pool_incremental_refresh_interval_seconds,
      pool_activity_reconciliation_interval_seconds: next.pool_activity_reconciliation_interval_seconds,
      pool_recompute_interval_seconds: next.pool_recompute_interval_seconds,
    })
  }, [discoverySettings, savePoolSettingsMutation])

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortBy === field) {
        setSortDir((d: SortDir) => (d === 'desc' ? 'asc' : 'desc'))
      } else {
        setSortBy(field)
        setSortDir('desc')
      }
    },
    [sortBy]
  )

  const handleCopyAddress = useCallback((address: string) => {
    navigator.clipboard.writeText(address).then(() => {
      setCopiedAddress(address)
      setTimeout(() => setCopiedAddress(null), 2000)
    })
  }, [])

  const toggleTagFilter = useCallback((tagName: string) => {
    setSelectedTags(prev =>
      prev.includes(tagName)
        ? prev.filter(t => t !== tagName)
        : [...prev, tagName]
    )
  }, [])

  const filteredTags = useMemo(() => {
    const q = tagSearch.trim().toLowerCase()
    if (!q) return tags
    return tags.filter(tag => {
      const display = (tag.display_name || tag.name).toLowerCase()
      const desc = (tag.description || '').toLowerCase()
      return display.includes(q) || desc.includes(q) || tag.name.toLowerCase().includes(q)
    })
  }, [tags, tagSearch])

  const poolMembersRaw: PoolMember[] = poolMembersData?.members || []
  const poolMembers: PoolMember[] = useMemo(() => {
    const query = poolSearch.trim().toLowerCase()
    const minWr = Math.max(0, minPoolWinRate)
    const filtered = poolMembersRaw.filter(member => {
      const flags = member.pool_flags || { manual_include: false, manual_exclude: false, blacklisted: false }
      if (!includeBlacklisted && flags.blacklisted) return false
      if (poolTierFilter !== 'all' && (member.pool_tier || '').toLowerCase() !== poolTierFilter) return false
      if (minWr > 0 && normalizePercentRatio(member.win_rate || 0) < minWr) return false
      if (!query) return true

      const haystack = [
        member.address,
        member.username || '',
        member.display_name || '',
        member.tracked_label || '',
        member.cluster_label || '',
        member.pool_tier || '',
        ...(member.tags || []),
        ...(member.strategies_detected || []),
        ...(member.market_categories || []),
      ]
        .join(' ')
        .toLowerCase()

      return haystack.includes(query)
    })

    const sorted = [...filtered].sort((a, b) => {
      const valueOf = (member: PoolMember): number => {
        switch (poolSortBy) {
          case 'selection_score':
            return Number(member.selection_score ?? member.composite_score ?? 0)
          case 'composite_score':
            return Number(member.composite_score ?? 0)
          case 'activity_score':
            return Number(member.activity_score ?? 0)
          case 'quality_score':
            return Number(member.quality_score ?? 0)
          case 'win_rate':
            return normalizePercentRatio(member.win_rate || 0)
          case 'total_pnl':
            return Number(member.total_pnl ?? 0)
          case 'total_trades':
            return Number(member.total_trades ?? 0)
          case 'trades_24h':
            return Number(member.trades_24h ?? 0)
          case 'trades_1h':
            return Number(member.trades_1h ?? 0)
          case 'last_trade_at':
            return member.last_trade_at ? new Date(member.last_trade_at).getTime() : 0
          default:
            return Number(member.composite_score ?? 0)
        }
      }

      const av = valueOf(a)
      const bv = valueOf(b)
      if (av !== bv) {
        return poolSortDir === 'asc' ? av - bv : bv - av
      }

      const aTie = Number(a.composite_score ?? 0)
      const bTie = Number(b.composite_score ?? 0)
      if (aTie !== bTie) return bTie - aTie
      return String(a.address).localeCompare(String(b.address))
    })

    return sorted
  }, [
    poolMembersRaw,
    poolSearch,
    poolTierFilter,
    minPoolWinRate,
    poolSortBy,
    poolSortDir,
    includeBlacklisted,
  ])
  const poolMemberStats = poolMembersData?.stats
  const poolMembersErrorMessage = useMemo(() => {
    if (!poolMembersError) return null
    const detail = (poolMembersError as any)?.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) return detail.trim()
    return 'Failed to load pool members.'
  }, [poolMembersError])

  const totalPages = Math.ceil(totalWallets / ITEMS_PER_PAGE)
  const poolActionBusy =
    trackWalletMutation.isPending ||
    manualIncludeMutation.isPending ||
    clearManualIncludeMutation.isPending ||
    manualExcludeMutation.isPending ||
    clearManualExcludeMutation.isPending ||
    blacklistMutation.isPending ||
    unblacklistMutation.isPending ||
    deletePoolWalletMutation.isPending ||
    promoteTrackedMutation.isPending

  const statusBadge = stats?.is_running
    ? (
      <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
        <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
        Worker scanning
      </Badge>
    )
    : stats?.paused
      ? (
        <Badge variant="outline" className="text-xs bg-yellow-500/10 text-yellow-400 border-yellow-500/20">
          <PauseCircle className="w-3 h-3 mr-1" />
          Paused
        </Badge>
      )
      : (
        <Badge variant="outline" className="text-xs bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
          Auto every {stats?.interval_minutes || 60}m
        </Badge>
      )

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {statusBadge}
        {stats?.current_activity && (
          <span className="max-w-[320px] truncate">{stats.current_activity}</span>
        )}
        {stats?.last_run_at && <span>Last run: {timeAgo(stats.last_run_at)}</span>}
        {isPoolView && (
          <Button
            variant="outline"
            size="sm"
            className="ml-auto h-8 text-xs gap-1.5"
            onClick={() => setPoolSettingsOpen(true)}
            disabled={!discoverySettings || savePoolSettingsMutation.isPending}
          >
            <Settings className="w-3.5 h-3.5" />
            Settings
          </Button>
        )}
      </div>

      {isPoolView ? (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-2">
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-3">
              <div className="p-2 bg-cyan-500/10 rounded-lg">
                <Users className="w-5 h-5 text-cyan-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Top Pool</p>
                <p className="text-lg font-semibold">
                  {formatNumber(poolStats?.pool_size || poolMemberStats?.pool_members || 0)}
                  <span className="text-[11px] text-muted-foreground ml-1">
                    / {poolStats?.effective_target_pool_size || poolStats?.target_pool_size || 500}
                  </span>
                </p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-3">
              <div className="p-2 bg-emerald-500/10 rounded-lg">
                <Activity className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Pool Active (1h)</p>
                <p className="text-sm font-semibold">
                  {poolStats?.active_1h ?? 0} ({(poolStats?.active_1h_pct ?? 0).toFixed(1)}%)
                </p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <TrendingUp className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Pool Active (24h)</p>
                <p className="text-sm font-semibold">
                  {poolStats?.active_24h ?? 0} ({(poolStats?.active_24h_pct ?? 0).toFixed(1)}%)
                </p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-3">
              <div className="p-2 bg-amber-500/10 rounded-lg">
                <RefreshCw className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Hourly Churn</p>
                <p className="text-sm font-semibold">{((poolStats?.churn_rate || 0) * 100).toFixed(2)}%</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-3">
              <div className="p-2 bg-violet-500/10 rounded-lg">
                <Target className="w-5 h-5 text-violet-400" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Members</p>
                <p className="text-sm font-semibold">
                  <span>{poolMemberStats?.pool_members ?? 0}</span>
                  <span className="ml-1 text-[11px] text-muted-foreground">
                    Tracked {poolMemberStats?.tracked_in_pool ?? 0}/{poolMemberStats?.tracked_total ?? 0}
                  </span>
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className={cn('grid gap-2', isPoolView ? 'grid-cols-2 lg:grid-cols-5' : 'grid-cols-2 lg:grid-cols-4')}>
          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-4">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Users className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Discovered</p>
                <p className="text-lg font-semibold">{formatNumber(stats?.total_discovered || 0)}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-4">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <TrendingUp className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Profitable</p>
                <p className="text-lg font-semibold">{formatNumber(stats?.total_profitable || 0)}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-4">
              <div className="p-2 bg-yellow-500/10 rounded-lg">
                <Target className="w-5 h-5 text-yellow-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Copy Candidates</p>
                <p className="text-lg font-semibold">{formatNumber(stats?.total_copy_candidates || 0)}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="flex items-center gap-3 p-4">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <Trophy className="w-5 h-5 text-purple-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Analyzed (last run)</p>
                <p className="text-lg font-semibold">{formatNumber(stats?.wallets_analyzed_last_run || 0)}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {isPoolView && (
      <Card className="border-border">
        <CardContent className="p-3 space-y-2.5">
          <div className="overflow-x-auto pb-1">
            <div className="flex min-w-max items-end gap-2">
              <div className="flex w-[190px] flex-col gap-1">
                <span className={FILTER_LABEL_CLASS}>Manual add</span>
                <Input
                  value={manualPoolAddress}
                  onChange={e => setManualPoolAddress(e.target.value)}
                  placeholder="0x... add to pool"
                  className={FILTER_INPUT_CLASS}
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={poolActionBusy || !manualPoolAddress.trim()}
                className="h-8 text-xs gap-1.5 mb-0.5"
                onClick={() => {
                  manualIncludeMutation.mutate(manualPoolAddress.trim().toLowerCase())
                  setManualPoolAddress('')
                }}
              >
                <UserCheck className="w-3.5 h-3.5" />
                Add
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => promoteTrackedMutation.mutate()}
                disabled={poolActionBusy}
                className="h-8 text-xs gap-1.5 mb-0.5"
              >
                {promoteTrackedMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <UserPlus className="w-3.5 h-3.5" />}
                Add tracked
              </Button>
              <div className="h-6 w-px bg-border/70 mb-1" />
              <div className="flex w-[210px] flex-col gap-1">
                <span className={FILTER_LABEL_CLASS}>Search</span>
                <Input
                  value={poolSearch}
                  onChange={e => setPoolSearch(e.target.value)}
                  placeholder="Wallet / username"
                  className={FILTER_INPUT_CLASS}
                />
              </div>
              <div className="flex w-[116px] flex-col gap-1">
                <span className={FILTER_LABEL_CLASS}>Tier</span>
                <select
                  value={poolTierFilter}
                  onChange={e => setPoolTierFilter(e.target.value as 'all' | 'core' | 'rising')}
                  className={FILTER_SELECT_CLASS}
                >
                  <option value="all">All tiers</option>
                  <option value="core">Core</option>
                  <option value="rising">Rising</option>
                </select>
              </div>
              <div className="flex w-[98px] flex-col gap-1">
                <span className={FILTER_LABEL_CLASS}>Min WR %</span>
                <Input
                  type="number"
                  value={minPoolWinRate}
                  onChange={e => setMinPoolWinRate(Math.max(0, Math.min(100, parseFloat(e.target.value) || 0)))}
                  min={0}
                  max={100}
                  step={1}
                  className={FILTER_INPUT_CLASS}
                />
              </div>
              <div className="flex w-[152px] flex-col gap-1">
                <span className={FILTER_LABEL_CLASS}>Sort by</span>
                <select
                  value={poolSortBy}
                  onChange={e => setPoolSortBy(e.target.value as 'selection_score' | 'composite_score' | 'activity_score' | 'quality_score' | 'trades_24h' | 'trades_1h' | 'total_trades' | 'total_pnl' | 'win_rate' | 'last_trade_at')}
                  className={FILTER_SELECT_CLASS}
                >
                  <option value="selection_score">Selection</option>
                  <option value="composite_score">Composite</option>
                  <option value="activity_score">Activity</option>
                  <option value="quality_score">Quality</option>
                  <option value="win_rate">Win rate</option>
                  <option value="total_pnl">PnL</option>
                  <option value="total_trades">Trades</option>
                  <option value="trades_24h">Trades 24h</option>
                  <option value="trades_1h">Trades 1h</option>
                  <option value="last_trade_at">Last trade</option>
                </select>
              </div>
              <div className="flex w-[94px] flex-col gap-1">
                <span className={FILTER_LABEL_CLASS}>Direction</span>
                <select
                  value={poolSortDir}
                  onChange={e => setPoolSortDir(e.target.value as 'asc' | 'desc')}
                  className={FILTER_SELECT_CLASS}
                >
                  <option value="desc">Desc</option>
                  <option value="asc">Asc</option>
                </select>
              </div>
              <label className="mb-0.5 flex h-8 items-center gap-1.5 rounded-md border border-border bg-background/40 px-2 text-[11px] text-muted-foreground">
                <input type="checkbox" checked={includeBlacklisted} onChange={e => setIncludeBlacklisted(e.target.checked)} className="h-3.5 w-3.5" />
                Show blacklisted
              </label>
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs gap-1.5 mb-0.5"
                onClick={() => {
                  setPoolSearch('')
                  setPoolTierFilter('all')
                  setMinPoolWinRate(0)
                  setPoolSortBy('composite_score')
                  setPoolSortDir('desc')
                  setIncludeBlacklisted(true)
                }}
              >
                Reset
              </Button>
            </div>
          </div>

          {poolMembersLoading ? (
            <div className="py-8 flex items-center justify-center">
              <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : poolMembersErrorMessage ? (
            <div className="py-6 text-sm text-red-300 text-center">
              {poolMembersErrorMessage}
            </div>
          ) : poolMembers.length === 0 ? (
            <div className="py-6 text-sm text-muted-foreground text-center">
              No pool members match current filters.
            </div>
          ) : (
            <div className="overflow-auto rounded border border-border bg-background/20 grow min-h-[72vh]">
              <Table className="text-[11px] leading-tight">
                <TableHeader className="sticky top-0 z-10 bg-background/85 backdrop-blur-sm">
                  <TableRow className="bg-muted/55 border-b border-border/80">
                    <TableHead className="h-9 px-2 min-w-[210px]">Trader</TableHead>
                    <TableHead className="h-9 px-2">Tier</TableHead>
                    <TableHead className="h-9 px-2 min-w-[220px]">Performance</TableHead>
                    <TableHead className="h-9 px-2 min-w-[190px]">Selection</TableHead>
                    <TableHead className="h-9 px-2 min-w-[220px]">Why Selected</TableHead>
                    <TableHead className="h-9 px-2 min-w-[120px]">Flags</TableHead>
                    <TableHead className="h-9 px-2 min-w-[140px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {poolMembers.map((member, rowIndex) => {
                    const flags = member.pool_flags || { manual_include: false, manual_exclude: false, blacklisted: false }
                    const canAnalyze = !!onAnalyzeWallet
                    const reasons = getPoolSelectionReasons(member)
                    const displayName = member.display_name || member.username || 'Unknown Trader'
                    const rowHighlight = rowIndex % 2 === 0 ? 'bg-background/30' : ''
                    const selectionValue = Number(member.selection_score ?? member.composite_score ?? 0)
                    const compositeValue = Number(member.composite_score || 0)
                    const selectionDelta = Math.abs(selectionValue - compositeValue)
                    const showCompositeScore = selectionDelta >= SCORE_DELTA_HIDE_THRESHOLD
                    const percentile =
                      member.selection_percentile != null
                        ? `${(Number(member.selection_percentile) * 100).toFixed(1)}%`
                        : null
                    const breakdown = member.selection_breakdown || {}
                    const qualityScore = Number(breakdown.quality_score ?? member.quality_score ?? 0)
                    const activityScore = Number(breakdown.activity_score ?? member.activity_score ?? 0)
                    const stabilityScore = Number(breakdown.stability_score ?? member.stability_score ?? 0)
                    const insiderScore = Number(breakdown.insider_score ?? 0)
                    const selectionSparkline = [
                      { key: 'quality', label: 'Quality', value: qualityScore, tone: scoreTone(qualityScore, 0.7, 0.45) },
                      { key: 'activity', label: 'Activity', value: activityScore, tone: scoreTone(activityScore, 0.55, 0.3) },
                      { key: 'stability', label: 'Stability', value: stabilityScore, tone: scoreTone(stabilityScore, 0.65, 0.45) },
                      { key: 'insider', label: 'Insider', value: insiderScore, tone: inverseScoreTone(insiderScore, 0.72, 0.6) },
                    ]
                    const hasSparkline = selectionSparkline.some((point) => point.value > 0)

                    return (
                      <TableRow key={member.address} className={cn('border-border/70 transition-colors hover:bg-muted/40', rowHighlight)}>
                        <TableCell className="px-2 py-1.5 align-middle">
                          <div className="min-w-0 space-y-0.5">
                            <div className="truncate text-[12px] font-semibold text-foreground" title={displayName}>
                              {displayName}
                            </div>
                            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                              {member.username && member.display_name !== member.username && (
                                <span className="max-w-[120px] truncate" title={`@${member.username}`}>@{member.username}</span>
                              )}
                              <span className="font-mono">{truncateAddress(member.address)}</span>
                            </div>
                            {member.name_source === 'tracked_label' && member.tracked_label && (
                              <div className="truncate text-[10px] text-muted-foreground" title={member.tracked_label}>
                                Label: {member.tracked_label}
                              </div>
                            )}
                          </div>
                        </TableCell>

                        <TableCell className="px-2 py-1.5 align-middle">
                          <MetricPill
                            label="Tier"
                            value={(member.pool_tier || 'out').toUpperCase()}
                            tone={poolTierTone(member.pool_tier)}
                            mono={false}
                          />
                        </TableCell>

                        <TableCell className="px-2 py-1.5 align-middle">
                          <div className="space-y-1">
                            <div className="flex flex-wrap items-center gap-1">
                              <PnlDisplay value={member.total_pnl || 0} className="text-xs" />
                              <MetricPill
                                label="WR"
                                value={formatWinRate(member.win_rate || 0)}
                                tone={scoreTone(normalizePercentRatio(member.win_rate || 0) / 100, 0.6, 0.45)}
                              />
                              <MetricPill label="T" value={formatNumber(member.total_trades || 0)} />
                            </div>
                            <div className="flex flex-wrap items-center gap-1">
                              <MetricPill
                                label="24h"
                                value={formatNumber(member.trades_24h || 0)}
                                tone={scoreTone(clamp01((member.trades_24h || 0) / 12), 0.6, 0.25)}
                              />
                              <MetricPill
                                label="1h"
                                value={formatNumber(member.trades_1h || 0)}
                                tone={scoreTone(clamp01((member.trades_1h || 0) / 4), 0.55, 0.25)}
                              />
                              <span className="text-[10px] text-muted-foreground">{timeAgo(member.last_trade_at || null)}</span>
                            </div>
                          </div>
                        </TableCell>

                        <TableCell className="px-2 py-1.5 align-middle">
                          <div className="space-y-1">
                            <div className="flex flex-wrap items-center gap-1">
                              <MetricPill
                                label={showCompositeScore ? 'Sel' : 'Sel/Cmp'}
                                value={formatScorePct(selectionValue)}
                                tone={scoreTone(selectionValue, 0.7, 0.5)}
                              />
                              {showCompositeScore && (
                                <MetricPill
                                  label="Cmp"
                                  value={formatScorePct(compositeValue)}
                                  tone={scoreTone(compositeValue, 0.7, 0.5)}
                                />
                              )}
                              {percentile && (
                                <MetricPill label="Top" value={percentile} tone="info" />
                              )}
                            </div>
                            {hasSparkline && (
                              <ScoreSparkline points={selectionSparkline} />
                            )}
                          </div>
                        </TableCell>

                        <TableCell className="px-2 py-1.5 align-middle">
                          <div className="flex flex-wrap gap-1">
                            {reasons.slice(0, 2).map((reason) => (
                              <Tooltip key={`${member.address}-${reason.code}`}>
                                <TooltipTrigger asChild>
                                  <span
                                    className={cn(
                                      'inline-flex max-w-[180px] items-center rounded-full border px-2 py-0.5 text-[10px] font-medium leading-none truncate',
                                      METRIC_TONE_CLASSES[selectionReasonTone(reason.code)]
                                    )}
                                  >
                                    {reason.label}
                                  </span>
                                </TooltipTrigger>
                                {reason.detail && <TooltipContent>{reason.detail}</TooltipContent>}
                              </Tooltip>
                            ))}
                            {reasons.length > 2 && (
                              <span className="text-[10px] text-muted-foreground">+{reasons.length - 2}</span>
                            )}
                          </div>
                        </TableCell>

                        <TableCell className="px-2 py-1.5 align-middle">
                          <div className="flex flex-wrap gap-1">
                            {member.tracked_wallet && <MetricPill label="Tracked" value="Yes" tone="info" mono={false} />}
                            {flags.manual_include && <MetricPill label="Manual+" value="On" tone="good" mono={false} />}
                            {flags.manual_exclude && <MetricPill label="Manual-" value="On" tone="warn" mono={false} />}
                            {flags.blacklisted && <MetricPill label="BL" value="On" tone="bad" mono={false} />}
                          </div>
                        </TableCell>

                        <TableCell className="px-2 py-1.5 align-middle">
                          <div className="flex flex-wrap gap-1">
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button
                                  onClick={() => onAnalyzeWallet?.(member.address, member.username || member.display_name || undefined)}
                                  disabled={!canAnalyze || poolActionBusy}
                                  className="p-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors disabled:opacity-50"
                                >
                                  <Activity className="w-3.5 h-3.5" />
                                </button>
                              </TooltipTrigger>
                              <TooltipContent>Analyze wallet</TooltipContent>
                            </Tooltip>
                            {!member.tracked_wallet ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() =>
                                      trackWalletMutation.mutate({
                                        address: member.address,
                                        username: member.username || member.display_name || undefined,
                                      })
                                    }
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <UserPlus className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Add to tracked traders</TooltipContent>
                              </Tooltip>
                            ) : (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    disabled
                                    className="p-1 rounded bg-emerald-500/10 text-emerald-300 transition-colors disabled:opacity-70"
                                  >
                                    <CheckCircle className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Already tracked</TooltipContent>
                              </Tooltip>
                            )}
                            {!flags.manual_include ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() => manualIncludeMutation.mutate(member.address)}
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <UserCheck className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Manual include</TooltipContent>
                              </Tooltip>
                            ) : (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() => clearManualIncludeMutation.mutate(member.address)}
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <UserCheck className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Clear manual include</TooltipContent>
                              </Tooltip>
                            )}
                            {!flags.manual_exclude ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() => manualExcludeMutation.mutate(member.address)}
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <UserX className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Manual exclude</TooltipContent>
                              </Tooltip>
                            ) : (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() => clearManualExcludeMutation.mutate(member.address)}
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <UserX className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Clear manual exclude</TooltipContent>
                              </Tooltip>
                            )}
                            {!flags.blacklisted ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() => blacklistMutation.mutate(member.address)}
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <Ban className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Blacklist</TooltipContent>
                              </Tooltip>
                            ) : (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={() => unblacklistMutation.mutate(member.address)}
                                    disabled={poolActionBusy}
                                    className="p-1 rounded bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
                                  >
                                    <Ban className="w-3.5 h-3.5" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>Unblacklist</TooltipContent>
                              </Tooltip>
                            )}
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button
                                  onClick={() => {
                                    if (window.confirm(`Delete ${member.address} from discovery/pool/tracking datasets?`)) {
                                      deletePoolWalletMutation.mutate(member.address)
                                    }
                                  }}
                                  disabled={poolActionBusy}
                                  className="p-1 rounded bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 transition-colors disabled:opacity-50"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </TooltipTrigger>
                              <TooltipContent>Delete wallet</TooltipContent>
                            </Tooltip>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <a
                                  href={`https://polymarket.com/profile/${member.address}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
                                >
                                  <ExternalLink className="w-3.5 h-3.5" />
                                </a>
                              </TooltipTrigger>
                              <TooltipContent>View on Polymarket</TooltipContent>
                            </Tooltip>
                          </div>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
      )}

      {isPoolView && discoverySettings && (
        <PoolSettingsFlyout
          isOpen={poolSettingsOpen}
          onClose={() => setPoolSettingsOpen(false)}
          initial={{
            pool_recompute_mode: discoverySettings.pool_recompute_mode || 'quality_only',
            pool_target_size: discoverySettings.pool_target_size ?? 500,
            pool_min_size: discoverySettings.pool_min_size ?? 400,
            pool_max_size: discoverySettings.pool_max_size ?? 600,
            pool_active_window_hours: discoverySettings.pool_active_window_hours ?? 72,
            pool_selection_score_floor: discoverySettings.pool_selection_score_floor ?? 0.55,
            pool_max_hourly_replacement_rate: discoverySettings.pool_max_hourly_replacement_rate ?? 0.15,
            pool_replacement_score_cutoff: discoverySettings.pool_replacement_score_cutoff ?? 0.05,
            pool_max_cluster_share: discoverySettings.pool_max_cluster_share ?? 0.08,
            pool_high_conviction_threshold: discoverySettings.pool_high_conviction_threshold ?? 0.72,
            pool_insider_priority_threshold: discoverySettings.pool_insider_priority_threshold ?? 0.62,
            pool_min_eligible_trades: discoverySettings.pool_min_eligible_trades ?? 50,
            pool_max_eligible_anomaly: discoverySettings.pool_max_eligible_anomaly ?? 0.5,
            pool_core_min_win_rate: discoverySettings.pool_core_min_win_rate ?? 0.6,
            pool_core_min_sharpe: discoverySettings.pool_core_min_sharpe ?? 1.0,
            pool_core_min_profit_factor: discoverySettings.pool_core_min_profit_factor ?? 1.5,
            pool_rising_min_win_rate: discoverySettings.pool_rising_min_win_rate ?? 0.55,
            pool_slo_min_analyzed_pct: discoverySettings.pool_slo_min_analyzed_pct ?? 95.0,
            pool_slo_min_profitable_pct: discoverySettings.pool_slo_min_profitable_pct ?? 80.0,
            pool_leaderboard_wallet_trade_sample: discoverySettings.pool_leaderboard_wallet_trade_sample ?? 160,
            pool_incremental_wallet_trade_sample: discoverySettings.pool_incremental_wallet_trade_sample ?? 80,
            pool_full_sweep_interval_seconds: discoverySettings.pool_full_sweep_interval_seconds ?? 1800,
            pool_incremental_refresh_interval_seconds: discoverySettings.pool_incremental_refresh_interval_seconds ?? 120,
            pool_activity_reconciliation_interval_seconds: discoverySettings.pool_activity_reconciliation_interval_seconds ?? 120,
            pool_recompute_interval_seconds: discoverySettings.pool_recompute_interval_seconds ?? 60,
          }}
          onSave={handleSavePoolSettings}
          savePending={savePoolSettingsMutation.isPending}
          saveMessage={poolSettingsSaveMessage}
        />
      )}

      {!isPoolView && (
      <div className="space-y-3">
        <Card className="border-border">
          <CardContent className="p-3 space-y-2.5">
            <div className="overflow-x-auto pb-1">
              <div className="flex min-w-max items-end gap-2">
                <div className="flex flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Period</span>
                  <div className="flex h-8 items-center bg-muted/50 rounded-lg p-0.5 border border-border">
                    {TIME_PERIODS.map(tp => (
                      <button
                        key={tp.value}
                        onClick={() => setTimePeriod(tp.value)}
                        className={cn(
                          'h-7 px-2.5 rounded-md text-xs font-medium transition-all',
                          timePeriod === tp.value
                            ? 'bg-primary text-primary-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                        )}
                      >
                        {tp.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="h-6 w-px bg-border/70 mb-1" />

                <div className="flex w-[350px] flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Tags</span>
                  <div className="flex gap-1">
                    <div className="relative flex-1 min-w-[170px]">
                      <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                      <Input
                        type="text"
                        value={tagSearch}
                        onChange={e => setTagSearch(e.target.value)}
                        placeholder="Filter tags"
                        className={cn(FILTER_INPUT_CLASS, 'pl-8')}
                      />
                    </div>
                    <select
                      value={tagPicker}
                      onChange={(e) => {
                        const value = e.target.value
                        if (value) {
                          toggleTagFilter(value)
                        }
                        setTagPicker('')
                      }}
                      disabled={tagsLoading || filteredTags.length === 0}
                      className={cn(FILTER_SELECT_CLASS, 'w-[170px]')}
                    >
                      <option value="">
                        {tagsLoading ? 'Loading...' : filteredTags.length === 0 ? 'No matching tags' : 'Select tag'}
                      </option>
                      {filteredTags.map((tag) => (
                        <option key={tag.name} value={tag.name}>
                          {tag.display_name || tag.name} ({tag.wallet_count})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="flex w-[112px] flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Min trades</span>
                  <Input
                    type="number"
                    value={minTrades}
                    onChange={e => setMinTrades(parseInt(e.target.value) || 0)}
                    min={0}
                    className={FILTER_INPUT_CLASS}
                  />
                </div>

                <div className="flex w-[120px] flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Min pnl ($)</span>
                  <Input
                    type="number"
                    value={minPnl}
                    onChange={e => setMinPnl(parseFloat(e.target.value) || 0)}
                    step={100}
                    className={FILTER_INPUT_CLASS}
                  />
                </div>

                <div className="flex w-[148px] flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Recommendation</span>
                  <select
                    value={recommendationFilter}
                    onChange={e => setRecommendationFilter(e.target.value as RecommendationFilter)}
                    className={FILTER_SELECT_CLASS}
                  >
                    <option value="">All</option>
                    <option value="copy_candidate">Copy candidate</option>
                    <option value="monitor">Monitor</option>
                    <option value="avoid">Avoid</option>
                  </select>
                </div>

                <div className="flex w-[140px] flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Category</span>
                  <select
                    value={marketCategoryFilter}
                    onChange={e => setMarketCategoryFilter(e.target.value as 'all' | 'politics' | 'sports' | 'crypto' | 'culture' | 'economics' | 'tech' | 'finance' | 'weather')}
                    className={FILTER_SELECT_CLASS}
                  >
                    <option value="all">All</option>
                    <option value="politics">Politics</option>
                    <option value="sports">Sports</option>
                    <option value="crypto">Crypto</option>
                    <option value="culture">Culture</option>
                    <option value="economics">Economics</option>
                    <option value="tech">Tech</option>
                    <option value="finance">Finance</option>
                    <option value="weather">Weather</option>
                  </select>
                </div>

                <div className="flex w-[138px] flex-col gap-1">
                  <span className={FILTER_LABEL_CLASS}>Min insider</span>
                  <Input
                    type="number"
                    value={minInsiderScore}
                    onChange={e => setMinInsiderScore(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
                    step={0.05}
                    min={0}
                    max={1}
                    className={FILTER_INPUT_CLASS}
                  />
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs gap-1.5 mb-0.5"
                  onClick={() => {
                    setTimePeriod('24h')
                    setTagSearch('')
                    setTagPicker('')
                    setSelectedTags([])
                    setMinTrades(0)
                    setMinPnl(0)
                    setRecommendationFilter('')
                    setMarketCategoryFilter('all')
                    setMinInsiderScore(0)
                  }}
                >
                  Reset
                </Button>
              </div>
            </div>

            {selectedTags.length > 0 && (
              <div className="flex items-center gap-1 flex-wrap">
                {selectedTags.map((name) => {
                  const tagMeta = tags.find((tag) => tag.name === name)
                  return (
                    <button
                      key={name}
                      onClick={() => toggleTagFilter(name)}
                      className="inline-flex h-6 items-center gap-1 rounded-full border border-border bg-background/70 px-2 text-[10px] text-muted-foreground hover:text-foreground"
                      title="Remove tag filter"
                    >
                      <span className="max-w-[120px] truncate">{tagMeta?.display_name || name}</span>
                      <X className="w-3 h-3" />
                    </button>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {leaderboardLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : wallets.length === 0 ? (
          <Card className="border-border">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Trophy className="w-12 h-12 text-muted-foreground/30 mb-4" />
              <p className="text-muted-foreground">No wallets found</p>
              <p className="text-sm text-muted-foreground/70 mt-1">
                Try clearing filters or wait for the discovery worker to complete the next run
              </p>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card className="border-border overflow-hidden">
              <div className="max-h-[620px] overflow-auto bg-background/20">
                <Table className="text-[11px] leading-tight">
                <TableHeader className="sticky top-0 z-10 bg-background/85 backdrop-blur-sm">
                  <TableRow className="bg-muted/55 border-b border-border/80">
                    <TableHead className="h-9 px-2 w-[46px]">#</TableHead>
                    <TableHead className="h-9 px-2 min-w-[230px]">Trader</TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="composite_score"
                        label="Composite"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="activity_score"
                        label="Activity"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="quality_score"
                        label="Quality"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="insider_score"
                        label="Insider"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="last_trade_at"
                        label="Last Trade"
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="total_pnl"
                        label={isWindowActive ? 'Period PnL' : 'PnL'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="win_rate"
                        label={isWindowActive ? 'Period WR' : 'Win Rate'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="sharpe_ratio"
                        label={isWindowActive ? 'Period Sharpe' : 'Sharpe'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="total_trades"
                        label={isWindowActive ? 'Period Trades' : 'Trades'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">
                      <SortButton
                        field="avg_roi"
                        label={isWindowActive ? 'Period ROI' : 'Avg ROI'}
                        currentSort={sortBy}
                        currentDir={sortDir}
                        onSort={handleSort}
                      />
                    </TableHead>
                    <TableHead className="h-9 px-2">Rec.</TableHead>
                    <TableHead className="h-9 px-2">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {wallets.map((wallet, idx) => (
                  <LeaderboardRow
                      key={wallet.address}
                      wallet={wallet}
                      rank={currentPage * ITEMS_PER_PAGE + idx + 1}
                      rowIndex={idx}
                      copiedAddress={copiedAddress}
                      onCopyAddress={handleCopyAddress}
                      onAnalyze={onAnalyzeWallet}
                      onTrack={(address, username) =>
                        trackWalletMutation.mutate({ address, username })
                      }
                      isTracking={trackWalletMutation.isPending}
                      useWindowMetrics={isWindowActive}
                    />
                  ))}
                </TableBody>
              </Table>
              </div>
            </Card>

            {totalPages > 1 && (
              <div className="flex items-center justify-between pt-2">
                <div className="text-sm text-muted-foreground">
                  Showing {currentPage * ITEMS_PER_PAGE + 1} - {Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalWallets)} of {totalWallets}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                    disabled={currentPage === 0}
                  >
                    Previous
                  </Button>
                  <span className="px-3 py-1.5 bg-card rounded-lg text-sm border border-border">
                    Page {currentPage + 1} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(p => p + 1)}
                    disabled={currentPage >= totalPages - 1}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
      )}
    </div>
  )
}

function SortButton({
  field,
  label,
  currentSort,
  currentDir,
  onSort,
}: {
  field: SortField
  label: string
  currentSort: SortField
  currentDir: SortDir
  onSort: (field: SortField) => void
}) {
  const isActive = currentSort === field
  return (
    <button
      onClick={() => onSort(field)}
      className={cn(
        'flex items-center gap-1 text-xs font-medium transition-colors whitespace-nowrap',
        isActive ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
      )}
    >
      {label}
      {isActive &&
        (currentDir === 'desc' ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronUp className="w-3 h-3" />
        ))}
    </button>
  )
}

function WalletAddress({
  address,
  username,
  copiedAddress,
  onCopy,
}: {
  address: string
  username: string | null
  copiedAddress: string | null
  onCopy: (address: string) => void
}) {
  return (
    <div className="min-w-0">
      {username && (
        <p className="max-w-[180px] truncate text-[12px] font-semibold leading-tight text-foreground">
          {username}
        </p>
      )}
      <div className="mt-0.5 flex items-center gap-1.5">
        <span className="font-mono text-[10px] text-muted-foreground">
          {truncateAddress(address)}
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => onCopy(address)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              {copiedAddress === address ? (
                <CheckCircle className="w-3 h-3 text-green-400" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>
            {copiedAddress === address ? 'Copied!' : 'Copy address'}
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  )
}

function PnlDisplay({ value, className }: { value: number; className?: string }) {
  const isPositive = value >= 0
  return (
    <span
      className={cn(
        'font-medium font-mono text-sm',
        isPositive ? 'text-sky-700 dark:text-sky-300' : 'text-rose-700 dark:text-rose-300',
        className
      )}
    >
      {isPositive ? '+' : ''}${formatPnl(value)}
    </span>
  )
}

function RecommendationBadge({ recommendation }: { recommendation: string }) {
  const colorClass =
    RECOMMENDATION_COLORS[recommendation] ||
    'border-slate-300 bg-slate-100 text-slate-800 dark:bg-muted-foreground/15 dark:text-muted-foreground dark:border-muted-foreground/20'
  const label = RECOMMENDATION_LABELS[recommendation] || recommendation
  return (
    <Badge variant="outline" className={cn('text-[10px] font-semibold', colorClass)}>
      {label}
    </Badge>
  )
}

function LeaderboardRow({
  wallet,
  rank,
  rowIndex,
  copiedAddress,
  onCopyAddress,
  onAnalyze,
  onTrack,
  isTracking,
  useWindowMetrics,
}: {
  wallet: DiscoveredWallet
  rank: number
  rowIndex: number
  copiedAddress: string | null
  onCopyAddress: (address: string) => void
  onAnalyze?: (address: string, username?: string) => void
  onTrack?: (address: string, username?: string | null) => void
  isTracking?: boolean
  useWindowMetrics?: boolean
}) {
  const rankDisplay = useWindowMetrics ? rank : wallet.rank_position || rank

  const pnl =
    useWindowMetrics && wallet.period_pnl != null
      ? wallet.period_pnl
      : (wallet.total_pnl ?? 0)
  const winRate =
    useWindowMetrics && wallet.period_win_rate != null
      ? wallet.period_win_rate
      : (wallet.win_rate ?? 0)
  const winRatePct = normalizePercentRatio(winRate)
  const sharpe = useWindowMetrics
    ? wallet.period_sharpe ?? wallet.sharpe_ratio
    : wallet.sharpe_ratio
  const trades =
    useWindowMetrics && wallet.period_trades != null
      ? wallet.period_trades
      : (wallet.total_trades ?? 0)
  const roi =
    useWindowMetrics && wallet.period_roi != null ? wallet.period_roi : (wallet.avg_roi ?? 0)
  const composite = wallet.composite_score ?? wallet.rank_score ?? 0
  const activity = wallet.activity_score ?? 0
  const quality = wallet.quality_score ?? wallet.rank_score ?? 0
  const marketCategories = wallet.market_categories || []
  const allTags = wallet.tags || []
  const tagPills = [...allTags.slice(0, 3)]
  const categoryPills = marketCategories.slice(0, 2).map(cat => `mkt:${cat}`)
  const pills = [...tagPills, ...categoryPills]
  const totalPillsCount = allTags.length + marketCategories.length
  const insiderScore = wallet.insider_score
  const insiderSuspicious = (insiderScore || 0) >= 0.72 &&
    (wallet.insider_confidence || 0) >= 0.60 &&
    (wallet.insider_sample_size || 0) >= 25
  const metricSparkline = [
    { key: 'composite', label: 'Composite', value: composite, tone: scoreTone(composite, 0.7, 0.5) },
    { key: 'quality', label: 'Quality', value: quality, tone: scoreTone(quality, 0.6, 0.4) },
    { key: 'activity', label: 'Activity', value: activity, tone: scoreTone(activity, 0.6, 0.3) },
  ]

  return (
    <TableRow
      className={cn(rowIndex % 2 === 0 ? 'bg-background/40' : '', 'transition-colors hover:bg-muted/40')}
    >
      <TableCell className="px-2 py-1.5 font-medium text-muted-foreground align-middle">
        <span
          className={cn(
            'flex items-center justify-center w-6 h-6 rounded-full text-[10px] font-bold',
            rankDisplay === 1
              ? 'bg-yellow-500/20 text-yellow-400'
              : rankDisplay === 2
                ? 'bg-muted-foreground/20 text-muted-foreground'
                : rankDisplay === 3
                  ? 'bg-amber-600/20 text-amber-500'
                  : 'bg-muted text-muted-foreground'
          )}
        >
          {rankDisplay}
        </span>
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <WalletAddress
          address={wallet.address}
          username={wallet.username}
          copiedAddress={copiedAddress}
          onCopy={onCopyAddress}
        />
        <div className="mt-1 flex items-center justify-between gap-2">
          {pills.length > 0 ? (
            <div className="flex items-center gap-1 flex-wrap min-w-0">
              {pills.map(tag => (
                <span
                  key={tag}
                  className={cn(
                    'inline-flex max-w-[92px] truncate rounded-full border px-1.5 py-0.5 text-[9px] font-medium leading-none',
                    tag.startsWith('mkt:')
                      ? METRIC_TONE_CLASSES.info
                      : 'border-border/80 bg-muted/45 text-muted-foreground'
                  )}
                >
                  {tag}
                </span>
              ))}
              {totalPillsCount > pills.length && (
                <span className="text-[9px] text-muted-foreground">
                  +{totalPillsCount - pills.length}
                </span>
              )}
            </div>
          ) : (
            <span className="text-[9px] text-muted-foreground/60">no tags</span>
          )}
          <ScoreSparkline points={metricSparkline} className="shrink-0" />
        </div>
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <MetricPill
          label="C"
          value={formatScorePct(composite)}
          tone={scoreTone(composite, 0.7, 0.5)}
          className="min-w-[72px] justify-between"
        />
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <MetricPill
          label="A"
          value={formatScorePct(activity)}
          tone={scoreTone(activity, 0.6, 0.3)}
          className="min-w-[72px] justify-between"
        />
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <MetricPill
          label="Q"
          value={formatScorePct(quality)}
          tone={scoreTone(quality, 0.6, 0.4)}
          className="min-w-[72px] justify-between"
        />
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        {insiderScore != null ? (
          <div className="space-y-0.5">
            <div className="flex items-center gap-1">
              <MetricPill
                label="I"
                value={insiderScore.toFixed(2)}
                tone={inverseScoreTone(insiderScore, 0.72, 0.6)}
                className="min-w-[64px] justify-between"
              />
              {insiderSuspicious && (
                <Badge variant="outline" className="text-[9px] bg-rose-500/10 text-rose-300 border-rose-500/20">
                  <AlertTriangle className="w-2.5 h-2.5 mr-0.5" />
                  Suspect
                </Badge>
              )}
            </div>
            <div className="text-[9px] text-muted-foreground">
              c{(wallet.insider_confidence || 0).toFixed(2)} · n{wallet.insider_sample_size || 0}
            </div>
          </div>
        ) : (
          <span className="text-[10px] text-muted-foreground">--</span>
        )}
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <span className="text-[10px] text-muted-foreground">
          {timeAgo(wallet.last_trade_at || null)}
        </span>
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <div>
          <PnlDisplay value={pnl} className="text-xs" />
          {useWindowMetrics && wallet.period_pnl != null && (
            <div className="text-[9px] text-muted-foreground/65 mt-0.5">
              All: ${formatPnl(wallet.total_pnl)}
            </div>
          )}
        </div>
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <div className="space-y-0.5">
          <MetricPill
            label="WR"
            value={formatWinRate(winRate)}
            tone={winRatePct >= 60 ? 'good' : winRatePct >= 45 ? 'warn' : 'bad'}
            className="min-w-[78px] justify-between"
          />
          {!useWindowMetrics && (
            <span className="text-[9px] text-muted-foreground">
              {wallet.wins}W/{wallet.losses}L
            </span>
          )}
        </div>
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        {sharpe != null ? (
          <MetricPill
            label="S"
            value={sharpe.toFixed(2)}
            tone={sharpe >= 2 ? 'good' : sharpe >= 1 ? 'warn' : 'neutral'}
            className="min-w-[64px] justify-between"
          />
        ) : (
          <span className="text-muted-foreground text-[10px]">--</span>
        )}
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <div className="space-y-0.5">
          <MetricPill label="T" value={formatNumber(trades)} className="min-w-[70px] justify-between" />
          {!useWindowMetrics && (
            <span className="text-[9px] text-muted-foreground/75">
              {(wallet.trades_per_day ?? 0).toFixed(1)}/d
            </span>
          )}
        </div>
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <MetricPill
          label="ROI"
          value={`${roi >= 0 ? '+' : ''}${formatPercent(roi)}`}
          tone={roi >= 0 ? 'good' : 'bad'}
          className="min-w-[84px] justify-between"
        />
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <RecommendationBadge recommendation={wallet.recommendation} />
      </TableCell>

      <TableCell className="px-2 py-1.5 align-middle">
        <div className="flex items-center gap-1">
          {onAnalyze && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onAnalyze(wallet.address, wallet.username || undefined)}
                  className="p-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                >
                  <Activity className="w-3.5 h-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Analyze wallet</TooltipContent>
            </Tooltip>
          )}
          {onTrack && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onTrack(wallet.address, wallet.username)}
                  disabled={isTracking}
                  className="p-1 rounded bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors disabled:opacity-50"
                >
                  <UserPlus className="w-3.5 h-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Track wallet</TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <a
                href={`https://polymarket.com/profile/${wallet.address}`}
                target="_blank"
                rel="noopener noreferrer"
                className="p-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors inline-flex"
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </TooltipTrigger>
            <TooltipContent>View on Polymarket</TooltipContent>
          </Tooltip>
        </div>
      </TableCell>
    </TableRow>
  )
}
