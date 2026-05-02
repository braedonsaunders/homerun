import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Search,
  X,
  ArrowLeft,
  Shield,
  Calendar,
  Tag,
  TrendingUp,
  Database,
  Bot,
  Wallet,
  Newspaper,
  AlertTriangle,
  Brain,
  Zap,
  FileText,
  Clock,
  Layers,
  Sparkles,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Activity,
  Power,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  searchGlobal,
  type SearchResultItem,
  type SearchGlobalResponse,
} from '../services/api'
import { Button } from './ui/button'
import { Badge } from './ui/badge'

// =======================================================================
// Per-entity-type rendering metadata
// =======================================================================

interface TypeMeta {
  label: string
  plural: string
  icon: React.ReactNode
  fg: string
  bg: string
  border: string
  /** Cards in this row are this wide.  Tweaked per-type for visual rhythm. */
  cardWidth: number
  /** Optional gradient accent for the row header. */
  accent: string
  navigate: (item: SearchResultItem) => void
}

const dispatch = (eventName: string, item: SearchResultItem) => {
  window.dispatchEvent(new CustomEvent(eventName, { detail: item }))
}

const TYPE_META: Record<string, TypeMeta> = {
  market: {
    label: 'Market',
    plural: 'Markets',
    icon: <Shield className="w-4 h-4" />,
    fg: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    accent: 'from-emerald-500/15 to-transparent',
    cardWidth: 320,
    navigate: (item) =>
      dispatch('market-selected', { ...item, ...(item.metadata ?? {}) } as any),
  },
  event: {
    label: 'Event',
    plural: 'Events',
    icon: <Calendar className="w-4 h-4" />,
    fg: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/20',
    accent: 'from-cyan-500/15 to-transparent',
    cardWidth: 260,
    navigate: (i) => dispatch('event-selected', i),
  },
  category: {
    label: 'Category',
    plural: 'Categories',
    icon: <Tag className="w-4 h-4" />,
    fg: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    accent: 'from-amber-500/15 to-transparent',
    cardWidth: 200,
    navigate: (i) => dispatch('category-selected', i),
  },
  opportunity: {
    label: 'Opportunity',
    plural: 'Opportunities',
    icon: <Zap className="w-4 h-4" />,
    fg: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/20',
    accent: 'from-yellow-500/15 to-transparent',
    cardWidth: 320,
    navigate: (i) => dispatch('opportunity-selected', i),
  },
  strategy: {
    label: 'Strategy',
    plural: 'Strategies',
    icon: <TrendingUp className="w-4 h-4" />,
    fg: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/20',
    accent: 'from-cyan-500/15 to-transparent',
    cardWidth: 280,
    navigate: (i) => dispatch('strategy-selected', i),
  },
  data_source: {
    label: 'Data source',
    plural: 'Data sources',
    icon: <Database className="w-4 h-4" />,
    fg: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    accent: 'from-blue-500/15 to-transparent',
    cardWidth: 260,
    navigate: (i) => dispatch('data-source-selected', i),
  },
  trader: {
    label: 'Trader',
    plural: 'Traders',
    icon: <Bot className="w-4 h-4" />,
    fg: 'text-purple-400',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/20',
    accent: 'from-purple-500/15 to-transparent',
    cardWidth: 260,
    navigate: (i) => dispatch('trader-selected', i),
  },
  wallet: {
    label: 'Wallet',
    plural: 'Wallets',
    icon: <Wallet className="w-4 h-4" />,
    fg: 'text-indigo-400',
    bg: 'bg-indigo-500/10',
    border: 'border-indigo-500/20',
    accent: 'from-indigo-500/15 to-transparent',
    cardWidth: 280,
    navigate: (i) => dispatch('wallet-selected', i),
  },
  news: {
    label: 'News',
    plural: 'News',
    icon: <Newspaper className="w-4 h-4" />,
    fg: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/20',
    accent: 'from-orange-500/15 to-transparent',
    cardWidth: 300,
    navigate: (i) => dispatch('news-selected', i),
  },
  alert: {
    label: 'Alert',
    plural: 'Alerts',
    icon: <AlertTriangle className="w-4 h-4" />,
    fg: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/20',
    accent: 'from-red-500/15 to-transparent',
    cardWidth: 280,
    navigate: (i) => dispatch('alert-selected', i),
  },
  research: {
    label: 'Research',
    plural: 'Research',
    icon: <Brain className="w-4 h-4" />,
    fg: 'text-violet-400',
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/20',
    accent: 'from-violet-500/15 to-transparent',
    cardWidth: 300,
    navigate: (i) => dispatch('research-selected', i),
  },
}

const FALLBACK_META: TypeMeta = {
  label: 'Item',
  plural: 'Items',
  icon: <FileText className="w-4 h-4" />,
  fg: 'text-muted-foreground',
  bg: 'bg-muted/40',
  border: 'border-border',
  accent: 'from-muted/40 to-transparent',
  cardWidth: 260,
  navigate: () => undefined,
}

const getMeta = (entityType: string): TypeMeta => TYPE_META[entityType] ?? FALLBACK_META

// Display order — the rows render top-to-bottom in this order.
const TYPE_ORDER = [
  'market',
  'opportunity',
  'event',
  'trader',
  'wallet',
  'news',
  'alert',
  'strategy',
  'research',
  'data_source',
  'category',
]

// =======================================================================
// Helpers
// =======================================================================

function formatCompact(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  const abs = Math.abs(n)
  if (abs >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

function formatRelative(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const sec = (Date.now() - d.getTime()) / 1000
  if (sec < 60) return 'just now'
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  if (sec < 86400 * 7) return `${Math.floor(sec / 86400)}d ago`
  if (sec < 86400 * 30) return `${Math.floor(sec / 86400 / 7)}w ago`
  return d.toLocaleDateString()
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function initials(name: string): string {
  const parts = (name || '').trim().split(/\s+/).filter(Boolean)
  if (parts.length === 0) return '?'
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
}

function shortAddress(addr: string): string {
  if (!addr) return ''
  if (addr.length <= 12) return addr
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`
}

// =======================================================================
// Sort options
// =======================================================================

type SortMode = 'relevance' | 'recency' | 'liquidity'
const SORT_OPTIONS: Array<{ key: SortMode; label: string; icon: React.ReactNode }> = [
  { key: 'relevance', label: 'Relevance', icon: <Sparkles className="w-3.5 h-3.5" /> },
  { key: 'recency', label: 'Most recent', icon: <Clock className="w-3.5 h-3.5" /> },
  { key: 'liquidity', label: 'Most liquid', icon: <Layers className="w-3.5 h-3.5" /> },
]

function sortItems(items: SearchResultItem[], mode: SortMode): SearchResultItem[] {
  const copy = items.slice()
  if (mode === 'relevance') copy.sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
  else if (mode === 'recency') {
    copy.sort((a, b) => {
      const ta = a.recency ? new Date(a.recency).getTime() : 0
      const tb = b.recency ? new Date(b.recency).getTime() : 0
      return tb - ta
    })
  } else if (mode === 'liquidity') {
    copy.sort((a, b) => (b.liquidity ?? 0) - (a.liquidity ?? 0))
  }
  return copy
}

// =======================================================================
// Main component
// =======================================================================

export interface SearchResultsViewProps {
  query: string
  onClose: () => void
  onQueryChange?: (next: string) => void
}

export default function SearchResultsView({
  query,
  onClose,
  onQueryChange,
}: SearchResultsViewProps) {
  const [draftQuery, setDraftQuery] = useState(query)
  const [enabledTypes, setEnabledTypes] = useState<Set<string> | null>(null)
  const [sortMode, setSortMode] = useState<SortMode>('relevance')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => setDraftQuery(query), [query])
  useEffect(() => inputRef.current?.focus(), [])

  // Esc closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Run the search.
  const { data, isLoading, isFetching, refetch } = useQuery<SearchGlobalResponse>({
    queryKey: [
      'search-results',
      query,
      enabledTypes ? Array.from(enabledTypes).sort().join(',') : null,
    ],
    queryFn: () =>
      searchGlobal(query, {
        limit: 100,
        types: enabledTypes ? Array.from(enabledTypes) : undefined,
      }),
    enabled: !!query.trim(),
    staleTime: 15_000,
  })

  const submitDraft = useCallback(() => {
    const trimmed = draftQuery.trim()
    if (!trimmed || trimmed === query) return
    onQueryChange?.(trimmed)
  }, [draftQuery, query, onQueryChange])

  const orderedGroups = useMemo<Array<[string, SearchResultItem[]]>>(() => {
    if (!data) return []
    const out: Array<[string, SearchResultItem[]]> = []
    for (const t of TYPE_ORDER) {
      const items = data.groups[t]
      if (items?.length) out.push([t, sortItems(items, sortMode)])
    }
    for (const t of Object.keys(data.groups)) {
      if (!TYPE_ORDER.includes(t)) out.push([t, sortItems(data.groups[t], sortMode)])
    }
    return out
  }, [data, sortMode])

  const allResultsCount = data?.total ?? 0

  // Per-type counts (computed from full data so they're stable across filters).
  const perTypeCounts = useMemo<Record<string, number>>(() => {
    const counts: Record<string, number> = {}
    if (!data) return counts
    for (const [t, items] of Object.entries(data.groups)) counts[t] = items.length
    return counts
  }, [data])

  const toggleType = (t: string) => {
    setEnabledTypes((current) => {
      if (current === null) return new Set([t])
      const next = new Set(current)
      if (next.has(t)) next.delete(t)
      else next.add(t)
      return next.size === 0 ? null : next
    })
  }

  const handleNavigate = useCallback(
    (item: SearchResultItem) => {
      const meta = getMeta(item.entity_type)
      meta.navigate(item)
      onClose()
    },
    [onClose]
  )

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-background via-background to-muted/20 section-enter">
      {/* ---- Hero header --------------------------------------------- */}
      <div className="relative overflow-hidden border-b border-border">
        <div className="pointer-events-none absolute -top-32 -left-20 h-72 w-72 rounded-full bg-purple-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -top-32 right-0 h-80 w-80 rounded-full bg-blue-500/10 blur-3xl" />

        <div className="relative mx-auto max-w-[1600px] px-6 pt-6 pb-5">
          <div className="flex items-center gap-3 mb-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="w-4 h-4 mr-1.5" />
              Back
            </Button>
            <div className="text-xs text-muted-foreground">Global search</div>
          </div>

          {/* Search input */}
          <div className="flex items-center gap-3 px-4 py-3 rounded-2xl border border-border bg-card/50 backdrop-blur-sm shadow-sm focus-within:border-purple-500/60 focus-within:shadow-purple-500/10 transition-all">
            <Search className="w-5 h-5 text-muted-foreground flex-shrink-0" />
            <input
              ref={inputRef}
              value={draftQuery}
              onChange={(e) => setDraftQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  submitDraft()
                }
              }}
              placeholder="Search markets, traders, strategies, news, wallets — anything…"
              className="flex-1 bg-transparent text-base text-foreground placeholder:text-muted-foreground/70 focus:outline-none"
            />
            {(isLoading || isFetching) && (
              <RefreshCw className="w-4 h-4 animate-spin text-muted-foreground" />
            )}
            {draftQuery && (
              <button
                onClick={() => {
                  setDraftQuery('')
                  inputRef.current?.focus()
                }}
                className="p-1 rounded-md hover:bg-muted text-muted-foreground"
                title="Clear"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Result summary + sort */}
          <div className="flex items-center justify-between gap-3 mt-4 flex-wrap">
            <div className="flex items-baseline gap-2 text-sm">
              {data ? (
                <>
                  <span className="text-2xl font-semibold text-foreground tabular-nums">
                    {allResultsCount}
                  </span>
                  <span className="text-muted-foreground">
                    result{allResultsCount === 1 ? '' : 's'} for
                  </span>
                  <span className="text-foreground font-medium">"{query}"</span>
                  <Badge variant="secondary" className="text-[10px] tabular-nums">
                    {data.latency_ms.toFixed(0)}ms
                  </Badge>
                </>
              ) : isLoading ? (
                <span className="text-muted-foreground">Searching…</span>
              ) : (
                <span className="text-muted-foreground">Start typing to search.</span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Sort
              </span>
              {SORT_OPTIONS.map((opt) => (
                <button
                  key={opt.key}
                  onClick={() => setSortMode(opt.key)}
                  className={cn(
                    'flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors',
                    sortMode === opt.key
                      ? 'bg-foreground text-background'
                      : 'text-muted-foreground hover:bg-muted'
                  )}
                >
                  {opt.icon}
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Type filter chips */}
          {data && allResultsCount > 0 && (
            <div className="flex flex-wrap items-center gap-2 mt-4">
              <button
                onClick={() => setEnabledTypes(null)}
                className={cn(
                  'text-xs px-3 py-1 rounded-full border transition-all',
                  enabledTypes === null
                    ? 'bg-foreground text-background border-foreground'
                    : 'border-border text-muted-foreground hover:bg-muted'
                )}
              >
                All <span className="opacity-60">· {allResultsCount}</span>
              </button>
              {TYPE_ORDER.map((t) => {
                const count = perTypeCounts[t] ?? 0
                if (count === 0) return null
                const meta = getMeta(t)
                const active = enabledTypes !== null && enabledTypes.has(t)
                return (
                  <button
                    key={t}
                    onClick={() => toggleType(t)}
                    className={cn(
                      'flex items-center gap-1.5 text-xs px-3 py-1 rounded-full border transition-all',
                      active
                        ? `${meta.bg} ${meta.border} ${meta.fg} border`
                        : 'border-border text-muted-foreground hover:bg-muted'
                    )}
                  >
                    <span className={meta.fg}>{meta.icon}</span>
                    {meta.plural}
                    <span className="opacity-60 tabular-nums">· {count}</span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* ---- Body: Netflix-style rows --------------------------------- */}
      <div className="mx-auto max-w-[1600px] py-4">
        {!query.trim() && <EmptyState onPick={(q) => onQueryChange?.(q)} />}

        {query.trim() && isLoading && !data && <SkeletonRows />}

        {data && allResultsCount === 0 && (
          <NoResults query={query} onRetry={() => refetch()} />
        )}

        {data && orderedGroups.length > 0 && (
          <div className="flex flex-col gap-1">
            {orderedGroups.map(([entityType, items]) => (
              <CardRow
                key={entityType}
                entityType={entityType}
                items={items}
                onNavigate={handleNavigate}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// =======================================================================
// CardRow — horizontally scrollable Netflix-style row
// =======================================================================

interface CardRowProps {
  entityType: string
  items: SearchResultItem[]
  onNavigate: (item: SearchResultItem) => void
}

function CardRow({ entityType, items, onNavigate }: CardRowProps) {
  const meta = getMeta(entityType)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const [canLeft, setCanLeft] = useState(false)
  const [canRight, setCanRight] = useState(false)

  const updateScrollState = useCallback(() => {
    const el = scrollerRef.current
    if (!el) return
    setCanLeft(el.scrollLeft > 4)
    setCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4)
  }, [])

  useEffect(() => {
    updateScrollState()
    const el = scrollerRef.current
    if (!el) return
    el.addEventListener('scroll', updateScrollState, { passive: true })
    const ro = new ResizeObserver(updateScrollState)
    ro.observe(el)
    return () => {
      el.removeEventListener('scroll', updateScrollState)
      ro.disconnect()
    }
  }, [updateScrollState, items.length])

  const scrollByPage = (dir: 1 | -1) => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollBy({ left: dir * el.clientWidth * 0.85, behavior: 'smooth' })
  }

  return (
    <section className="group/row px-6 py-3">
      {/* Row header */}
      <header className="relative flex items-center gap-2.5 mb-3 px-1">
        <div
          className={cn(
            'absolute -left-2 top-1/2 -translate-y-1/2 h-7 w-1 rounded-full',
            meta.bg
          )}
        />
        <span className={cn('flex items-center justify-center w-7 h-7 rounded-lg', meta.bg, meta.fg)}>
          {meta.icon}
        </span>
        <h3 className="text-sm font-medium text-foreground tracking-wide">{meta.plural}</h3>
        <span className="text-xs text-muted-foreground tabular-nums">{items.length}</span>
        <div className="ml-auto flex items-center gap-1 opacity-0 group-hover/row:opacity-100 transition-opacity">
          <button
            disabled={!canLeft}
            onClick={() => scrollByPage(-1)}
            className={cn(
              'h-7 w-7 rounded-md border border-border bg-card/60 backdrop-blur-sm flex items-center justify-center transition-all',
              canLeft ? 'hover:bg-muted text-foreground' : 'opacity-30 cursor-not-allowed'
            )}
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <button
            disabled={!canRight}
            onClick={() => scrollByPage(1)}
            className={cn(
              'h-7 w-7 rounded-md border border-border bg-card/60 backdrop-blur-sm flex items-center justify-center transition-all',
              canRight ? 'hover:bg-muted text-foreground' : 'opacity-30 cursor-not-allowed'
            )}
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      {/* Card strip */}
      <div className="relative">
        {/* Edge fades — non-interactive cosmetic gradients */}
        {canLeft && (
          <div className="pointer-events-none absolute left-0 top-0 bottom-2 w-12 bg-gradient-to-r from-background to-transparent z-10" />
        )}
        {canRight && (
          <div className="pointer-events-none absolute right-0 top-0 bottom-2 w-12 bg-gradient-to-l from-background to-transparent z-10" />
        )}

        <div
          ref={scrollerRef}
          className="flex gap-3 overflow-x-auto pb-2 snap-x snap-mandatory scroll-smooth scrollbar-thin"
          style={{ scrollbarWidth: 'thin' }}
        >
          {items.map((item) => (
            <div
              key={`${item.entity_type}-${item.entity_id}`}
              style={{ width: meta.cardWidth }}
              className="flex-shrink-0 snap-start"
            >
              <CardForType item={item} onNavigate={onNavigate} />
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// =======================================================================
// CardForType — dispatches to the right widget
// =======================================================================

function CardForType({
  item,
  onNavigate,
}: {
  item: SearchResultItem
  onNavigate: (item: SearchResultItem) => void
}) {
  switch (item.entity_type) {
    case 'market':
      return <MarketCard item={item} onNavigate={onNavigate} />
    case 'opportunity':
      return <OpportunityResultCard item={item} onNavigate={onNavigate} />
    case 'event':
      return <EventCard item={item} onNavigate={onNavigate} />
    case 'category':
      return <CategoryCard item={item} onNavigate={onNavigate} />
    case 'trader':
      return <TraderCard item={item} onNavigate={onNavigate} />
    case 'wallet':
      return <WalletCard item={item} onNavigate={onNavigate} />
    case 'strategy':
      return <StrategyCard item={item} onNavigate={onNavigate} />
    case 'data_source':
      return <DataSourceCard item={item} onNavigate={onNavigate} />
    case 'news':
      return <NewsCard item={item} onNavigate={onNavigate} />
    case 'alert':
      return <AlertCard item={item} onNavigate={onNavigate} />
    case 'research':
      return <ResearchCard item={item} onNavigate={onNavigate} />
    default:
      return <GenericCard item={item} onNavigate={onNavigate} />
  }
}

// =======================================================================
// Shared card chrome
// =======================================================================

interface BaseCardProps {
  item: SearchResultItem
  onNavigate: (item: SearchResultItem) => void
}

function CardShell({
  meta,
  children,
  onClick,
  className,
}: {
  meta: TypeMeta
  children: React.ReactNode
  onClick: () => void
  className?: string
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'group relative h-full rounded-xl border bg-card/40 backdrop-blur-sm overflow-hidden cursor-pointer',
        'transition-all duration-150',
        'hover:bg-card/80 hover:border-foreground/30 hover:-translate-y-0.5 hover:shadow-lg',
        meta.border,
        className
      )}
    >
      {/* Subtle gradient accent in the top-left */}
      <div
        className={cn(
          'pointer-events-none absolute inset-x-0 top-0 h-12 bg-gradient-to-b opacity-60',
          meta.accent
        )}
      />
      <div className="relative h-full">{children}</div>
    </div>
  )
}

function HighlightedTitle({
  snippet,
  fallback,
  className,
}: {
  snippet?: string
  fallback: string
  className?: string
}) {
  return (
    <div
      className={cn(
        'text-sm font-medium leading-snug line-clamp-2 [&_mark]:bg-yellow-500/30 [&_mark]:text-foreground [&_mark]:rounded-sm [&_mark]:px-0.5',
        className
      )}
      dangerouslySetInnerHTML={{ __html: snippet || escapeHtml(fallback) }}
    />
  )
}

// =======================================================================
// MarketCard — compact opportunity-style card with YES/NO bars
// =======================================================================

function MarketCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('market')
  const md = item.metadata ?? {}
  const yes = typeof md.yes_price === 'number' ? md.yes_price : null
  const no = typeof md.no_price === 'number' ? md.no_price : null
  const liq = item.liquidity ?? md.liquidity ?? null
  const vol = item.volume ?? md.volume ?? null

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[156px]">
        <div className="flex items-center justify-between gap-2">
          {item.category ? (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {item.category}
            </Badge>
          ) : (
            <span />
          )}
          {item.recency && (
            <span className="text-[10px] text-muted-foreground/70 flex items-center gap-1">
              <Clock className="w-2.5 h-2.5" />
              {formatRelative(item.recency)}
            </span>
          )}
        </div>

        <HighlightedTitle snippet={item.snippet} fallback={item.title} />

        <div className="mt-auto space-y-1.5">
          {/* YES bar */}
          {yes !== null && (
            <div className="flex items-center gap-2 text-[10px] tabular-nums">
              <span className="w-7 text-emerald-400 font-medium">YES</span>
              <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full bg-emerald-500"
                  style={{ width: `${Math.min(100, Math.max(0, yes * 100))}%` }}
                />
              </div>
              <span className="w-12 text-right text-foreground">${yes.toFixed(2)}</span>
            </div>
          )}
          {no !== null && (
            <div className="flex items-center gap-2 text-[10px] tabular-nums">
              <span className="w-7 text-red-400 font-medium">NO</span>
              <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full bg-red-500"
                  style={{ width: `${Math.min(100, Math.max(0, no * 100))}%` }}
                />
              </div>
              <span className="w-12 text-right text-foreground">${no.toFixed(2)}</span>
            </div>
          )}
          <div className="flex items-center gap-3 text-[10px] text-muted-foreground tabular-nums pt-0.5">
            {liq !== null && (
              <span>
                LIQ <span className="text-foreground">${formatCompact(liq)}</span>
              </span>
            )}
            {vol !== null && (
              <span>
                VOL <span className="text-foreground">${formatCompact(vol)}</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// OpportunityResultCard — title + ROI/risk
// =======================================================================

function OpportunityResultCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('opportunity')
  const md = item.metadata ?? {}
  const roi = typeof md.expected_roi === 'number' ? md.expected_roi : null
  const risk = typeof md.risk_score === 'number' ? md.risk_score : null
  const profitable = md.was_profitable as boolean | null | undefined
  const stratLabel = item.category || md.strategy_type || ''

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[148px]">
        <div className="flex items-center justify-between gap-2">
          {stratLabel && (
            <Badge
              variant="secondary"
              className="text-[10px] px-1.5 py-0 bg-yellow-500/15 text-yellow-300"
            >
              {stratLabel}
            </Badge>
          )}
          {item.recency && (
            <span className="text-[10px] text-muted-foreground/70 flex items-center gap-1">
              <Clock className="w-2.5 h-2.5" />
              {formatRelative(item.recency)}
            </span>
          )}
        </div>

        <HighlightedTitle snippet={item.snippet} fallback={item.title} />

        <div className="mt-auto flex items-center gap-3 text-[11px] tabular-nums">
          {roi !== null && (
            <Stat
              label="ROI"
              value={`${roi.toFixed(2)}%`}
              accent={roi >= 0 ? 'text-emerald-400' : 'text-red-400'}
            />
          )}
          {risk !== null && (
            <Stat
              label="Risk"
              value={risk.toFixed(2)}
              accent={
                risk > 0.7
                  ? 'text-red-400'
                  : risk > 0.4
                    ? 'text-amber-400'
                    : 'text-emerald-400'
              }
            />
          )}
          {profitable !== null && profitable !== undefined && (
            <Badge
              variant="secondary"
              className={cn(
                'text-[10px] px-1.5 py-0 ml-auto',
                profitable ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'
              )}
            >
              {profitable ? 'profitable' : 'loss'}
            </Badge>
          )}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// EventCard
// =======================================================================

function EventCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('event')
  const md = item.metadata ?? {}
  const marketCount = typeof md.market_count === 'number' ? md.market_count : null

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[120px]">
        <Calendar className={cn('w-5 h-5', meta.fg)} />
        <HighlightedTitle snippet={item.snippet} fallback={item.title} />
        <div className="mt-auto flex items-center gap-2 text-[11px] text-muted-foreground tabular-nums">
          {item.category && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {item.category}
            </Badge>
          )}
          {marketCount !== null && (
            <span className="ml-auto">
              {marketCount} {marketCount === 1 ? 'market' : 'markets'}
            </span>
          )}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// CategoryCard
// =======================================================================

function CategoryCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('category')
  const md = item.metadata ?? {}
  const oppCount = typeof md.opportunity_count === 'number' ? md.opportunity_count : null

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-4 flex flex-col items-center justify-center gap-2 h-[110px] text-center">
        <Tag className={cn('w-6 h-6', meta.fg)} />
        <div className="text-sm font-medium text-foreground truncate w-full">{item.title}</div>
        {oppCount !== null && (
          <div className="text-[10px] text-muted-foreground tabular-nums">
            {oppCount} active
          </div>
        )}
      </div>
    </CardShell>
  )
}

// =======================================================================
// TraderCard — avatar + mode/latency badges
// =======================================================================

function TraderCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('trader')
  const md = item.metadata ?? {}
  const mode = (md.mode as string) || ''
  const enabled = md.is_enabled as boolean | undefined
  const paused = md.is_paused as boolean | undefined

  const status: { color: string; label: string } = paused
    ? { color: 'bg-amber-500', label: 'paused' }
    : enabled
      ? { color: 'bg-emerald-500', label: 'live' }
      : { color: 'bg-muted', label: 'off' }

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[140px]">
        <div className="flex items-center gap-2.5">
          <div
            className={cn(
              'w-9 h-9 rounded-full flex items-center justify-center font-semibold text-xs',
              meta.bg,
              meta.fg
            )}
          >
            {initials(item.title)}
          </div>
          <div className="flex-1 min-w-0">
            <HighlightedTitle
              snippet={item.snippet}
              fallback={item.title}
              className="text-[13px] font-medium line-clamp-1"
            />
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className={cn('w-1.5 h-1.5 rounded-full', status.color)} />
              <span className="text-[10px] text-muted-foreground">{status.label}</span>
            </div>
          </div>
        </div>

        {item.subtitle && (
          <p className="text-[11px] text-muted-foreground line-clamp-2">{item.subtitle}</p>
        )}

        <div className="mt-auto flex items-center gap-1.5 flex-wrap">
          {mode && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {mode}
            </Badge>
          )}
          {item.tags?.slice(0, 2).map((tag) => (
            <span
              key={tag}
              className="text-[9px] px-1.5 py-0 rounded bg-muted text-muted-foreground uppercase tracking-wide"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// WalletCard
// =======================================================================

function WalletCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('wallet')
  const md = item.metadata ?? {}
  const trades = typeof md.total_trades === 'number' ? md.total_trades : null
  const winRate = typeof md.win_rate === 'number' ? md.win_rate : null
  const pnl = typeof md.total_pnl === 'number' ? md.total_pnl : null
  const flagged = !!md.is_flagged
  const addr = (md.address as string) || item.entity_id

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[140px]">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <Wallet className={cn('w-3.5 h-3.5 flex-shrink-0', meta.fg)} />
            <span className="font-mono text-[10px] text-muted-foreground truncate">
              {shortAddress(addr)}
            </span>
          </div>
          {flagged && (
            <Badge
              variant="secondary"
              className="text-[10px] px-1.5 py-0 bg-amber-500/15 text-amber-300"
            >
              flagged
            </Badge>
          )}
        </div>

        <HighlightedTitle
          snippet={item.snippet}
          fallback={item.title}
          className="text-[13px]"
        />

        <div className="mt-auto grid grid-cols-3 gap-2 text-center text-[10px] tabular-nums">
          <div>
            <div className="text-muted-foreground/70 uppercase tracking-wide text-[9px]">
              Trades
            </div>
            <div className="text-foreground font-medium">
              {trades !== null ? formatCompact(trades) : '—'}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground/70 uppercase tracking-wide text-[9px]">
              Win
            </div>
            <div className="text-foreground font-medium">
              {winRate !== null ? `${(winRate * 100).toFixed(0)}%` : '—'}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground/70 uppercase tracking-wide text-[9px]">
              PnL
            </div>
            <div
              className={cn(
                'font-medium',
                pnl === null
                  ? 'text-foreground'
                  : pnl >= 0
                    ? 'text-emerald-400'
                    : 'text-red-400'
              )}
            >
              {pnl !== null ? `$${formatCompact(pnl)}` : '—'}
            </div>
          </div>
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// StrategyCard
// =======================================================================

function StrategyCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('strategy')
  const md = item.metadata ?? {}
  const enabled = md.enabled as boolean | undefined
  const status = (md.status as string) || ''
  const sourceKey = (md.source_key as string) || item.category || ''
  const isSystem = !!md.is_system

  const statusDot = status === 'loaded'
    ? 'bg-emerald-500'
    : status === 'error'
      ? 'bg-red-500'
      : 'bg-amber-500'

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[130px]">
        <div className="flex items-center gap-2">
          <span className={cn('w-1.5 h-1.5 rounded-full', statusDot)} />
          <HighlightedTitle
            snippet={item.snippet}
            fallback={item.title}
            className="text-[13px] line-clamp-1 flex-1"
          />
          {enabled === false && <Power className="w-3 h-3 text-muted-foreground/50" />}
        </div>

        {item.subtitle && (
          <p className="text-[11px] text-muted-foreground line-clamp-2">{item.subtitle}</p>
        )}

        <div className="mt-auto flex items-center gap-1.5 flex-wrap">
          {sourceKey && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {sourceKey}
            </Badge>
          )}
          {status && (
            <span className="text-[10px] text-muted-foreground">{status}</span>
          )}
          {isSystem && (
            <span className="text-[9px] uppercase tracking-wide text-muted-foreground/60 ml-auto">
              system
            </span>
          )}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// DataSourceCard
// =======================================================================

function DataSourceCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('data_source')
  const md = item.metadata ?? {}
  const kind = (md.source_kind as string) || ''
  const status = (md.status as string) || ''
  const enabled = md.enabled as boolean | undefined

  const statusDot = status === 'loaded'
    ? 'bg-emerald-500'
    : status === 'error'
      ? 'bg-red-500'
      : 'bg-amber-500'

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[120px]">
        <div className="flex items-center gap-2">
          <Database className={cn('w-4 h-4 flex-shrink-0', meta.fg)} />
          <HighlightedTitle
            snippet={item.snippet}
            fallback={item.title}
            className="text-[13px] line-clamp-1 flex-1"
          />
          <span className={cn('w-1.5 h-1.5 rounded-full', statusDot)} />
        </div>

        {item.subtitle && (
          <p className="text-[10px] text-muted-foreground line-clamp-2">{item.subtitle}</p>
        )}

        <div className="mt-auto flex items-center gap-1.5">
          {kind && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {kind}
            </Badge>
          )}
          {enabled === false && (
            <span className="text-[9px] uppercase text-muted-foreground/60">disabled</span>
          )}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// NewsCard
// =======================================================================

function NewsCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('news')
  const md = item.metadata ?? {}
  const source = (md.source as string) || (md.feed_source as string) || ''
  const url = (md.url as string) || ''

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[140px]">
        <div className="flex items-center justify-between gap-2">
          {source && (
            <Badge
              variant="secondary"
              className="text-[10px] px-1.5 py-0 bg-orange-500/15 text-orange-300"
            >
              {source}
            </Badge>
          )}
          {item.recency && (
            <span className="text-[10px] text-muted-foreground/70 flex items-center gap-1">
              <Clock className="w-2.5 h-2.5" />
              {formatRelative(item.recency)}
            </span>
          )}
        </div>

        <HighlightedTitle snippet={item.snippet} fallback={item.title} />

        {url && (
          <div className="mt-auto flex items-center gap-1.5 text-[10px] text-muted-foreground/70 truncate">
            <ExternalLink className="w-3 h-3 flex-shrink-0" />
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="hover:underline truncate"
              onClick={(e) => e.stopPropagation()}
            >
              {url.replace(/^https?:\/\//, '').slice(0, 40)}
            </a>
          </div>
        )}
      </div>
    </CardShell>
  )
}

// =======================================================================
// AlertCard — left-side severity stripe
// =======================================================================

function AlertCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('alert')
  const md = item.metadata ?? {}
  const severity = (md.severity as string) || 'info'
  const resolved = !!md.is_resolved

  const stripeColor =
    severity === 'critical'
      ? 'bg-red-600'
      : severity === 'high'
        ? 'bg-red-500'
        : severity === 'medium'
          ? 'bg-amber-500'
          : 'bg-blue-500'

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="flex h-[130px]">
        <div className={cn('w-1', stripeColor)} />
        <div className="p-3 flex flex-col gap-2 flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <Badge
              variant="secondary"
              className={cn(
                'text-[10px] px-1.5 py-0',
                severity === 'critical' && 'bg-red-500/20 text-red-200',
                severity === 'high' && 'bg-red-500/15 text-red-300',
                severity === 'medium' && 'bg-amber-500/15 text-amber-300',
                severity === 'low' && 'bg-blue-500/15 text-blue-300'
              )}
            >
              {severity}
            </Badge>
            {resolved && (
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                resolved
              </Badge>
            )}
          </div>

          <HighlightedTitle snippet={item.snippet} fallback={item.title} />

          {item.recency && (
            <span className="mt-auto text-[10px] text-muted-foreground/70 flex items-center gap-1">
              <Clock className="w-2.5 h-2.5" />
              {formatRelative(item.recency)}
            </span>
          )}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// ResearchCard
// =======================================================================

function ResearchCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta('research')
  const md = item.metadata ?? {}
  const sessionType = (md.session_type as string) || item.subtitle || ''
  const status = (md.status as string) || ''

  const statusDot =
    status === 'completed'
      ? 'bg-emerald-500'
      : status === 'failed'
        ? 'bg-red-500'
        : status === 'running'
          ? 'bg-blue-500 animate-pulse'
          : 'bg-amber-500'

  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[130px]">
        <div className="flex items-center gap-2">
          <Brain className={cn('w-4 h-4 flex-shrink-0', meta.fg)} />
          <span className={cn('w-1.5 h-1.5 rounded-full ml-auto', statusDot)} />
          {item.recency && (
            <span className="text-[10px] text-muted-foreground/70">
              {formatRelative(item.recency)}
            </span>
          )}
        </div>

        <HighlightedTitle snippet={item.snippet} fallback={item.title} />

        <div className="mt-auto flex items-center gap-1.5">
          {sessionType && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {sessionType.replace(/_/g, ' ')}
            </Badge>
          )}
          {status && (
            <span className="text-[10px] text-muted-foreground">{status}</span>
          )}
        </div>
      </div>
    </CardShell>
  )
}

// =======================================================================
// GenericCard — fallback for unknown entity types
// =======================================================================

function GenericCard({ item, onNavigate }: BaseCardProps) {
  const meta = getMeta(item.entity_type)
  return (
    <CardShell meta={meta} onClick={() => onNavigate(item)}>
      <div className="p-3 flex flex-col gap-2 h-[120px]">
        <div className="flex items-center gap-2">
          <Activity className={cn('w-4 h-4', meta.fg)} />
          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
            {item.entity_type}
          </span>
        </div>
        <HighlightedTitle snippet={item.snippet} fallback={item.title} />
        {item.subtitle && (
          <p className="mt-auto text-[10px] text-muted-foreground line-clamp-1">
            {item.subtitle}
          </p>
        )}
      </div>
    </CardShell>
  )
}

// =======================================================================
// Stat helper
// =======================================================================

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <span className="flex items-baseline gap-1">
      <span className="text-muted-foreground/70 uppercase tracking-wide text-[9px]">
        {label}
      </span>
      <span className={cn('text-foreground', accent)}>{value}</span>
    </span>
  )
}

// =======================================================================
// Empty / loading / no-results states
// =======================================================================

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const suggestions = ['trump', 'election', 'btc', 'arb', 'fed', 'kalshi']
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mb-4 border border-border/40">
        <Search className="w-7 h-7 text-muted-foreground" />
      </div>
      <h2 className="text-lg font-medium text-foreground mb-1">Search the entire system</h2>
      <p className="text-sm text-muted-foreground max-w-md mb-6">
        Markets, traders, strategies, news, wallets, alerts, research — every entity is indexed
        and ranked by relevance, recency, and liquidity.
      </p>
      <div className="flex flex-wrap items-center gap-2 justify-center">
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="text-xs px-3 py-1.5 rounded-full border border-border bg-card hover:bg-muted text-foreground"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

function NoResults({ query, onRetry }: { query: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="w-16 h-16 rounded-2xl bg-muted/40 flex items-center justify-center mb-4 border border-border/40">
        <Search className="w-7 h-7 text-muted-foreground" />
      </div>
      <h2 className="text-lg font-medium text-foreground mb-1">No matches for "{query}"</h2>
      <p className="text-sm text-muted-foreground max-w-md mb-4">
        Try different keywords, fewer terms, or remove type filters. Search supports OR (
        <code className="text-xs bg-muted px-1 rounded">btc | eth</code>) and exclusion (
        <code className="text-xs bg-muted px-1 rounded">trump -biden</code>).
      </p>
      <Button variant="secondary" size="sm" onClick={onRetry}>
        <RefreshCw className="w-3.5 h-3.5 mr-1.5" /> Retry
      </Button>
    </div>
  )
}

function SkeletonRows() {
  return (
    <div className="px-6">
      {[0, 1, 2].map((row) => (
        <div key={row} className="py-3">
          <div className="flex items-center gap-2 mb-3 px-1">
            <div className="w-7 h-7 rounded-lg bg-muted/40" />
            <div className="h-4 w-24 bg-muted rounded" />
          </div>
          <div className="flex gap-3 overflow-hidden">
            {[0, 1, 2, 3, 4].map((c) => (
              <div
                key={c}
                className="flex-shrink-0 w-[300px] h-[140px] rounded-xl bg-muted/30 border border-border animate-pulse"
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
