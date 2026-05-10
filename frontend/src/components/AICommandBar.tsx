import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Search,
  Brain,
  Shield,
  TrendingUp,
  Newspaper,
  MessageCircle,
  X,
  RefreshCw,
  Sparkles,
  Command as CommandIcon,
  Wallet,
  Bot,
  AlertTriangle,
  FileText,
  Database,
  Calendar,
  Tag,
  Pin,
  PinOff,
  History,
  Zap,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  sendAIChat,
  searchGlobal,
  getRecentSearches,
  type SearchResultItem,
  type SearchGlobalResponse,
  type RecentSearchEntry,
} from '../services/api'
import { Dialog, DialogContent, DialogTitle } from './ui/dialog'
import { Button } from './ui/button'
import { Badge } from './ui/badge'

interface AICommandBarProps {
  isOpen: boolean
  onClose: () => void
  onNavigateToAI?: (section: string) => void
  onOpenCopilot?: (contextType?: string, contextId?: string, label?: string) => void
  /** Open the dedicated full-page unified search results view. */
  onOpenSearchPage?: (query: string) => void
}

type CommandMode = 'global' | 'ask' | 'commands'

// ---------------------------------------------------------------------------
// Per-entity-type rendering metadata.  One stop-shop for icon, color, label,
// and the navigation handler.  Adding a new searchable entity in the
// backend → add an entry here and it appears in the UI.
// ---------------------------------------------------------------------------

interface TypeMeta {
  label: string
  plural: string
  icon: React.ReactNode
  color: string
  navigate: (
    item: SearchResultItem,
    helpers: {
      onNavigateToAI?: (section: string) => void
      onOpenCopilot?: (contextType?: string, contextId?: string, label?: string) => void
      close: () => void
    }
  ) => void
}

type TFn = (key: string, opts?: Record<string, unknown>) => string

const buildTypeMeta = (t: TFn): Record<string, TypeMeta> => ({
  market: {
    label: t('aiCommandBar.types.market.label'),
    plural: t('aiCommandBar.types.market.plural'),
    icon: <Shield className="w-4 h-4" />,
    color: 'text-green-400',
    navigate: (item, h) => {
      h.onNavigateToAI?.('resolution')
      h.close()
      // Preserve the legacy event for the resolution panel.
      window.dispatchEvent(
        new CustomEvent('market-selected', {
          detail: {
            market_id: item.metadata.market_id ?? item.entity_id,
            question: item.metadata.question ?? item.title,
            yes_price: item.metadata.yes_price ?? null,
            no_price: item.metadata.no_price ?? null,
            liquidity: item.metadata.liquidity ?? item.liquidity,
            event_title: item.metadata.event_title ?? item.subtitle,
            category: item.metadata.category ?? item.category,
          },
        })
      )
    },
  },
  event: {
    label: t('aiCommandBar.types.event.label'),
    plural: t('aiCommandBar.types.event.plural'),
    icon: <Calendar className="w-4 h-4" />,
    color: 'text-cyan-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('event-selected', { detail: item }))
    },
  },
  category: {
    label: t('aiCommandBar.types.category.label'),
    plural: t('aiCommandBar.types.category.plural'),
    icon: <Tag className="w-4 h-4" />,
    color: 'text-amber-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('category-selected', { detail: item }))
    },
  },
  strategy: {
    label: t('aiCommandBar.types.strategy.label'),
    plural: t('aiCommandBar.types.strategy.plural'),
    icon: <TrendingUp className="w-4 h-4" />,
    color: 'text-cyan-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('strategy-selected', { detail: item }))
    },
  },
  data_source: {
    label: t('aiCommandBar.types.dataSource.label'),
    plural: t('aiCommandBar.types.dataSource.plural'),
    icon: <Database className="w-4 h-4" />,
    color: 'text-blue-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('data-source-selected', { detail: item }))
    },
  },
  trader: {
    label: t('aiCommandBar.types.trader.label'),
    plural: t('aiCommandBar.types.trader.plural'),
    icon: <Bot className="w-4 h-4" />,
    color: 'text-purple-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('trader-selected', { detail: item }))
    },
  },
  wallet: {
    label: t('aiCommandBar.types.wallet.label'),
    plural: t('aiCommandBar.types.wallet.plural'),
    icon: <Wallet className="w-4 h-4" />,
    color: 'text-indigo-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('wallet-selected', { detail: item }))
    },
  },
  news: {
    label: t('aiCommandBar.types.news.label'),
    plural: t('aiCommandBar.types.news.plural'),
    icon: <Newspaper className="w-4 h-4" />,
    color: 'text-orange-400',
    navigate: (item, h) => {
      h.onNavigateToAI?.('news')
      h.close()
      window.dispatchEvent(new CustomEvent('news-selected', { detail: item }))
    },
  },
  alert: {
    label: t('aiCommandBar.types.alert.label'),
    plural: t('aiCommandBar.types.alert.plural'),
    icon: <AlertTriangle className="w-4 h-4" />,
    color: 'text-red-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('alert-selected', { detail: item }))
    },
  },
  research: {
    label: t('aiCommandBar.types.research.label'),
    plural: t('aiCommandBar.types.research.plural'),
    icon: <Brain className="w-4 h-4" />,
    color: 'text-violet-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('research-selected', { detail: item }))
    },
  },
  opportunity: {
    label: t('aiCommandBar.types.opportunity.label'),
    plural: t('aiCommandBar.types.opportunity.plural'),
    icon: <Zap className="w-4 h-4" />,
    color: 'text-yellow-400',
    navigate: (item, h) => {
      h.close()
      window.dispatchEvent(new CustomEvent('opportunity-selected', { detail: item }))
    },
  },
})

const buildFallbackMeta = (t: TFn): TypeMeta => ({
  label: t('aiCommandBar.types.fallback.label'),
  plural: t('aiCommandBar.types.fallback.plural'),
  icon: <FileText className="w-4 h-4" />,
  color: 'text-muted-foreground',
  navigate: (_item, h) => h.close(),
})

const useTypeMeta = () => {
  const { t } = useTranslation()
  const typeMeta = useMemo(() => buildTypeMeta(t), [t])
  const fallbackMeta = useMemo(() => buildFallbackMeta(t), [t])
  return useCallback(
    (entityType: string): TypeMeta => typeMeta[entityType] ?? fallbackMeta,
    [typeMeta, fallbackMeta]
  )
}

// Display order — most user-relevant types first.
const TYPE_DISPLAY_ORDER = [
  'market',
  'event',
  'opportunity',
  'strategy',
  'trader',
  'wallet',
  'news',
  'alert',
  'research',
  'data_source',
  'category',
]

// ---------------------------------------------------------------------------
// localStorage helpers for recents + pinned items.
// ---------------------------------------------------------------------------

const RECENTS_KEY = 'hr_search_recent_queries_v1'
const PINNED_KEY = 'hr_search_pinned_items_v1'
const RECENTS_LIMIT = 8
const PINNED_LIMIT = 12

const loadJSON = <T,>(key: string, fallback: T): T => {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallback
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

const saveJSON = (key: string, value: unknown) => {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* quota or disabled storage — silent ignore */
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function AICommandBar({
  isOpen,
  onClose,
  onNavigateToAI,
  onOpenCopilot,
  onOpenSearchPage,
}: AICommandBarProps) {
  const { t } = useTranslation()
  const getTypeMeta = useTypeMeta()
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<CommandMode>('global')
  const [searchData, setSearchData] = useState<SearchGlobalResponse | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [enabledTypes, setEnabledTypes] = useState<Set<string> | null>(null)
  const [recents, setRecents] = useState<string[]>([])
  const [pinned, setPinned] = useState<SearchResultItem[]>([])
  const [serverRecents, setServerRecents] = useState<RecentSearchEntry[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const requestSeq = useRef(0)

  // ---------- keyboard: open / close ----------
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        if (isOpen) onClose()
      }
      if (e.key === 'Escape' && isOpen) onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  // ---------- on open: reset, focus, hydrate recents/pinned ----------
  useEffect(() => {
    if (!isOpen) return
    setInput('')
    setMode('global')
    setSearchData(null)
    setSearchLoading(false)
    setSelectedIndex(0)
    setEnabledTypes(null)
    setRecents(loadJSON<string[]>(RECENTS_KEY, []))
    setPinned(loadJSON<SearchResultItem[]>(PINNED_KEY, []))
    setTimeout(() => inputRef.current?.focus(), 50)
    // Load server-side recents (deduped, popular).  Best-effort.
    getRecentSearches(8)
      .then((res) => setServerRecents(res.queries || []))
      .catch(() => setServerRecents([]))
  }, [isOpen])

  // ---------- debounced global search ----------
  const runSearch = useCallback(
    async (query: string, types: Set<string> | null) => {
      const seq = ++requestSeq.current
      setSearchLoading(true)
      try {
        const data = await searchGlobal(query, {
          limit: 30,
          types: types ? Array.from(types) : undefined,
        })
        if (seq !== requestSeq.current) return
        setSearchData(data)
        setSelectedIndex(0)
      } catch {
        if (seq !== requestSeq.current) return
        setSearchData(null)
      } finally {
        if (seq === requestSeq.current) setSearchLoading(false)
      }
    },
    []
  )

  useEffect(() => {
    if (mode !== 'global') return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (input.trim().length < 1) {
      setSearchData(null)
      setSearchLoading(false)
      return
    }
    debounceRef.current = setTimeout(() => {
      void runSearch(input.trim(), enabledTypes)
    }, 180)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [input, mode, enabledTypes, runSearch])

  // ---------- ask AI ----------
  const askMutation = useMutation({
    mutationFn: async (question: string) => sendAIChat({ message: question }),
  })

  // ---------- legacy commands (still reachable via Tab → Commands) ----------
  const commands = useMemo(
    () => [
      {
        id: 'ask-ai',
        label: t('aiCommandBar.commands.askAi.label'),
        description: t('aiCommandBar.commands.askAi.description'),
        icon: <MessageCircle className="w-4 h-4" />,
        color: 'text-purple-400',
        action: () => {
          setMode('ask')
          setInput('')
        },
      },
      {
        id: 'resolution-analysis',
        label: t('aiCommandBar.commands.resolutionAnalysis.label'),
        description: t('aiCommandBar.commands.resolutionAnalysis.description'),
        icon: <Shield className="w-4 h-4" />,
        color: 'text-green-400',
        action: () => {
          onNavigateToAI?.('resolution')
          onClose()
        },
      },
      {
        id: 'market-analysis',
        label: t('aiCommandBar.commands.marketAnalysis.label'),
        description: t('aiCommandBar.commands.marketAnalysis.description'),
        icon: <TrendingUp className="w-4 h-4" />,
        color: 'text-cyan-400',
        action: () => {
          onNavigateToAI?.('market')
          onClose()
        },
      },
      {
        id: 'news-sentiment',
        label: t('aiCommandBar.commands.newsSentiment.label'),
        description: t('aiCommandBar.commands.newsSentiment.description'),
        icon: <Newspaper className="w-4 h-4" />,
        color: 'text-orange-400',
        action: () => {
          onNavigateToAI?.('news')
          onClose()
        },
      },
      {
        id: 'open-copilot',
        label: t('aiCommandBar.commands.openCopilot.label'),
        description: t('aiCommandBar.commands.openCopilot.description'),
        icon: <Sparkles className="w-4 h-4" />,
        color: 'text-purple-400',
        action: () => {
          onOpenCopilot?.()
          onClose()
        },
      },
    ],
    [t, onNavigateToAI, onClose, onOpenCopilot]
  )

  // ---------- ordered groups for rendering ----------
  const orderedGroups = useMemo<Array<[string, SearchResultItem[]]>>(() => {
    if (!searchData) return []
    const out: Array<[string, SearchResultItem[]]> = []
    for (const t of TYPE_DISPLAY_ORDER) {
      const items = searchData.groups[t]
      if (items && items.length > 0) out.push([t, items])
    }
    // Any types we don't know about — append at the end.
    for (const t of Object.keys(searchData.groups)) {
      if (!TYPE_DISPLAY_ORDER.includes(t)) out.push([t, searchData.groups[t]])
    }
    return out
  }, [searchData])

  // Flat list in render order, used for keyboard navigation.
  const flatResults = useMemo<SearchResultItem[]>(() => {
    if (!searchData) return []
    return orderedGroups.flatMap(([, items]) => items)
  }, [searchData, orderedGroups])

  // ---------- helpers ----------
  const persistRecentQuery = (q: string) => {
    const trimmed = q.trim()
    if (!trimmed) return
    const next = [trimmed, ...recents.filter((r) => r !== trimmed)].slice(0, RECENTS_LIMIT)
    setRecents(next)
    saveJSON(RECENTS_KEY, next)
  }

  const handleResultSelect = (item: SearchResultItem) => {
    persistRecentQuery(input)
    const meta = getTypeMeta(item.entity_type)
    meta.navigate(item, { onNavigateToAI, onOpenCopilot, close: onClose })
  }

  const togglePin = (item: SearchResultItem) => {
    const key = (i: SearchResultItem) => `${i.entity_type}:${i.entity_id}`
    const isPinned = pinned.some((p) => key(p) === key(item))
    const next = isPinned
      ? pinned.filter((p) => key(p) !== key(item))
      : [item, ...pinned].slice(0, PINNED_LIMIT)
    setPinned(next)
    saveJSON(PINNED_KEY, next)
  }

  const isPinned = (item: SearchResultItem) =>
    pinned.some((p) => p.entity_type === item.entity_type && p.entity_id === item.entity_id)

  const toggleType = (t: string) => {
    setEnabledTypes((current) => {
      if (current === null) {
        // Was "all" — switch to "only this one"
        return new Set([t])
      }
      const next = new Set(current)
      if (next.has(t)) next.delete(t)
      else next.add(t)
      // Empty → back to "all"
      return next.size === 0 ? null : next
    })
    setSelectedIndex(0)
  }

  const openSearchPageForCurrentQuery = useCallback(() => {
    const q = input.trim()
    if (!q) return
    persistRecentQuery(q)
    onOpenSearchPage?.(q)
    onClose()
  }, [input, onOpenSearchPage, onClose])

  // ---------- keyboard ----------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (mode === 'global') {
      const items = flatResults
      // Cmd/Ctrl+Enter always jumps straight to the full search page
      // for the current query, regardless of which result is selected.
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        openSearchPageForCurrentQuery()
        return
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex((i) => Math.min(i + 1, items.length - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex((i) => Math.max(i - 1, 0))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (items.length === 0 && input.trim()) {
          // No specific result to open — jump to the full page so the
          // user always has a productive Enter.
          openSearchPageForCurrentQuery()
        } else if (items[selectedIndex]) {
          handleResultSelect(items[selectedIndex])
        }
      } else if (e.key === 'Tab') {
        e.preventDefault()
        setMode('commands')
        setInput('')
      }
      return
    }
    if (mode === 'commands') {
      const items = commands.filter(
        (c) =>
          input === '' ||
          c.label.toLowerCase().includes(input.toLowerCase()) ||
          c.description.toLowerCase().includes(input.toLowerCase())
      )
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex((i) => Math.min(i + 1, items.length - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex((i) => Math.max(i - 1, 0))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        items[selectedIndex]?.action()
      } else if (e.key === 'Backspace' && input === '') {
        setMode('global')
      }
      return
    }
    // mode === 'ask'
    if (e.key === 'Enter' && input.trim()) {
      e.preventDefault()
      askMutation.mutate(input.trim())
    } else if (e.key === 'Backspace' && input === '') {
      setMode('global')
    }
  }

  // ---------- render ----------
  const placeholder =
    mode === 'global'
      ? t('aiCommandBar.placeholders.global')
      : mode === 'ask'
        ? t('aiCommandBar.placeholders.ask')
        : t('aiCommandBar.placeholders.commands')

  const allRecents = useMemo(() => {
    // Merge local + server recents, dedup, cap.
    const seen = new Set<string>()
    const merged: string[] = []
    for (const q of recents) {
      if (!seen.has(q)) {
        seen.add(q)
        merged.push(q)
      }
    }
    for (const r of serverRecents) {
      if (!seen.has(r.query)) {
        seen.add(r.query)
        merged.push(r.query)
      }
    }
    return merged.slice(0, RECENTS_LIMIT)
  }, [recents, serverRecents])

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="overflow-hidden p-0 shadow-2xl shadow-purple-500/10 max-w-2xl gap-0 border-border bg-background rounded-2xl top-[12%] translate-y-0">
        <DialogTitle className="sr-only">{t('aiCommandBar.title')}</DialogTitle>

        {/* Input row */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          {mode === 'global' && <Search className="w-4 h-4 text-muted-foreground flex-shrink-0" />}
          {mode === 'ask' && <Brain className="w-4 h-4 text-purple-400 flex-shrink-0" />}
          {mode === 'commands' && (
            <div className="flex items-center gap-1 text-muted-foreground">
              <CommandIcon className="w-4 h-4" />
              <span className="text-xs">K</span>
            </div>
          )}
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              setSelectedIndex(0)
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          {searchLoading && mode === 'global' && (
            <RefreshCw className="w-4 h-4 animate-spin text-muted-foreground" />
          )}
          {searchData && mode === 'global' && !searchLoading && (
            <span className="text-[10px] text-muted-foreground tabular-nums">
              {searchData.total} · {t('aiCommandBar.latencyMs', { ms: searchData.latency_ms.toFixed(0) })}
            </span>
          )}
          {mode !== 'global' && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setMode('global')
                setInput('')
              }}
              className="text-xs h-7 px-2"
            >
              {t('aiCommandBar.escButton')}
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
            <X className="w-4 h-4 text-muted-foreground" />
          </Button>
        </div>

        {/* Type filter chips (global mode only) */}
        {mode === 'global' && input.trim().length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-4 py-2 border-b border-border bg-muted/30">
            <button
              onClick={() => setEnabledTypes(null)}
              className={cn(
                'text-[11px] px-2 py-0.5 rounded-full border transition-colors',
                enabledTypes === null
                  ? 'bg-foreground text-background border-foreground'
                  : 'border-border text-muted-foreground hover:bg-muted'
              )}
            >
              {t('aiCommandBar.allFilter')}
            </button>
            {TYPE_DISPLAY_ORDER.map((entityType) => {
              const meta = getTypeMeta(entityType)
              const active = enabledTypes !== null && enabledTypes.has(entityType)
              return (
                <button
                  key={entityType}
                  onClick={() => toggleType(entityType)}
                  className={cn(
                    'text-[11px] px-2 py-0.5 rounded-full border transition-colors flex items-center gap-1',
                    active
                      ? 'bg-foreground text-background border-foreground'
                      : 'border-border text-muted-foreground hover:bg-muted'
                  )}
                >
                  <span className={cn('inline-flex', !active && meta.color)}>{meta.icon}</span>
                  {meta.plural}
                </button>
              )
            })}
          </div>
        )}

        {/* Results body */}
        <div className="max-h-[60vh] overflow-y-auto">
          {/* Global: empty state — recents + pinned + commands shortcut */}
          {mode === 'global' && input.trim().length === 0 && (
            <div className="p-3 space-y-4">
              {pinned.length > 0 && (
                <div>
                  <div className="px-2 pb-1 text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                    <Pin className="w-3 h-3" /> {t('aiCommandBar.pinned')}
                  </div>
                  {pinned.map((item, i) => (
                    <ResultRow
                      key={`pin-${item.entity_type}-${item.entity_id}-${i}`}
                      item={item}
                      selected={false}
                      pinned={true}
                      onSelect={() => handleResultSelect(item)}
                      onTogglePin={() => togglePin(item)}
                    />
                  ))}
                </div>
              )}
              {allRecents.length > 0 && (
                <div>
                  <div className="px-2 pb-1 text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                    <History className="w-3 h-3" /> {t('aiCommandBar.recentSearches')}
                  </div>
                  <div className="flex flex-wrap gap-1.5 px-2">
                    {allRecents.map((q) => (
                      <button
                        key={q}
                        onClick={() => setInput(q)}
                        className="text-xs px-2 py-1 rounded-full bg-muted hover:bg-card text-foreground border border-border"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div>
                <div className="px-2 pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                  {t('aiCommandBar.quickActions')}
                </div>
                {commands.map((cmd) => (
                  <button
                    key={cmd.id}
                    onClick={cmd.action}
                    className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-left hover:bg-muted transition-colors"
                  >
                    <span className={cn('flex-shrink-0', cmd.color)}>{cmd.icon}</span>
                    <span className="flex-1 min-w-0">
                      <span className="block text-sm text-foreground">{cmd.label}</span>
                      <span className="block text-xs text-muted-foreground">{cmd.description}</span>
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Global: result groups */}
          {mode === 'global' && input.trim().length > 0 && (
            <div className="p-2">
              {searchData && searchData.results.length === 0 && !searchLoading && (
                <div className="text-center py-8 text-sm text-muted-foreground">
                  {t('aiCommandBar.noMatches')}
                </div>
              )}
              {searchData &&
                orderedGroups.map(([entityType, items]) => {
                  const meta = getTypeMeta(entityType)
                  // Compute the running flat-index offset so keyboard
                  // selection lights up the right row.
                  let flatBase = 0
                  for (const [innerType, list] of orderedGroups) {
                    if (innerType === entityType) break
                    flatBase += list.length
                  }
                  return (
                    <div key={entityType} className="mb-2">
                      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                        <span className={meta.color}>{meta.icon}</span>
                        {meta.plural}
                        <span className="text-muted-foreground/60">· {items.length}</span>
                      </div>
                      {items.map((item, i) => (
                        <ResultRow
                          key={`${entityType}-${item.entity_id}`}
                          item={item}
                          selected={flatBase + i === selectedIndex}
                          pinned={isPinned(item)}
                          onHover={() => setSelectedIndex(flatBase + i)}
                          onSelect={() => handleResultSelect(item)}
                          onTogglePin={() => togglePin(item)}
                        />
                      ))}
                    </div>
                  )
                })}

              {/* View all → opens the full unified search results page */}
              {searchData && onOpenSearchPage && (
                <button
                  onClick={openSearchPageForCurrentQuery}
                  className="mt-2 w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-left bg-gradient-to-r from-purple-500/10 to-blue-500/10 hover:from-purple-500/20 hover:to-blue-500/20 border border-purple-500/20 transition-colors"
                >
                  <span className="flex items-center gap-2 text-sm text-foreground">
                    <Search className="w-4 h-4 text-purple-400" />
                    {searchData.total === 1
                      ? t('aiCommandBar.viewAllResultOne', { count: searchData.total })
                      : t('aiCommandBar.viewAllResultsOther', { count: searchData.total })}
                    <span className="font-medium">"{input.trim()}"</span>
                  </span>
                  <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                    <kbd className="px-1.5 py-0.5 bg-background/60 rounded border border-border/60">⌘↵</kbd>
                  </span>
                </button>
              )}
            </div>
          )}

          {/* Commands mode */}
          {mode === 'commands' && (
            <div className="p-2">
              {commands
                .filter(
                  (c) =>
                    input === '' ||
                    c.label.toLowerCase().includes(input.toLowerCase()) ||
                    c.description.toLowerCase().includes(input.toLowerCase())
                )
                .map((cmd, i) => (
                  <button
                    key={cmd.id}
                    onClick={cmd.action}
                    onMouseEnter={() => setSelectedIndex(i)}
                    className={cn(
                      'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-colors',
                      i === selectedIndex ? 'bg-muted' : 'hover:bg-card'
                    )}
                  >
                    <span className={cn('flex-shrink-0', cmd.color)}>{cmd.icon}</span>
                    <span className="flex-1 min-w-0">
                      <span className="block text-sm text-foreground">{cmd.label}</span>
                      <span className="block text-xs text-muted-foreground">{cmd.description}</span>
                    </span>
                  </button>
                ))}
            </div>
          )}

          {/* Ask AI mode */}
          {mode === 'ask' && (
            <div className="p-4">
              {askMutation.isPending && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <RefreshCw className="w-4 h-4 animate-spin text-purple-400" />
                  {t('aiCommandBar.thinking')}
                </div>
              )}
              {askMutation.data && (
                <div className="bg-muted rounded-xl p-4 border border-border">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <Badge variant="secondary" className="text-xs">{t('aiCommandBar.aiResponse')}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                    {askMutation.data.response}
                  </p>
                </div>
              )}
              {askMutation.error && (
                <p className="text-sm text-red-400">
                  {(askMutation.error as Error).message}
                </p>
              )}
              {!askMutation.isPending && !askMutation.data && !askMutation.error && (
                <p className="text-sm text-muted-foreground text-center">
                  {t('aiCommandBar.askPrompt')}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 flex items-center gap-4 text-[10px] text-muted-foreground border-t border-border">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded">↵</kbd> {t('aiCommandBar.footer.open')}
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded">↑↓</kbd> {t('aiCommandBar.footer.navigate')}
          </span>
          {mode === 'global' && (
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-muted rounded">Tab</kbd> {t('aiCommandBar.footer.commands')}
            </span>
          )}
          <span className="flex items-center gap-1 ml-auto">
            <kbd className="px-1 py-0.5 bg-muted rounded">Esc</kbd> {t('aiCommandBar.footer.close')}
          </span>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Result row component — shared between groups, pinned, etc.
// ---------------------------------------------------------------------------

interface ResultRowProps {
  item: SearchResultItem
  selected: boolean
  pinned: boolean
  onHover?: () => void
  onSelect: () => void
  onTogglePin: () => void
}

function ResultRow({ item, selected, pinned, onHover, onSelect, onTogglePin }: ResultRowProps) {
  const { t } = useTranslation()
  const getTypeMeta = useTypeMeta()
  const meta = getTypeMeta(item.entity_type)
  const liquidity = item.liquidity ?? item.metadata?.liquidity
  const yesPrice = item.metadata?.yes_price
  const showFinance =
    item.entity_type === 'market' && (typeof yesPrice === 'number' || typeof liquidity === 'number')
  return (
    <div
      onClick={onSelect}
      onMouseEnter={onHover}
      className={cn(
        'group relative w-full flex items-start gap-3 px-3 py-2 rounded-xl text-left cursor-pointer transition-colors',
        selected ? 'bg-muted' : 'hover:bg-card'
      )}
    >
      <span className={cn('flex-shrink-0 mt-0.5', meta.color)}>{meta.icon}</span>
      <div className="flex-1 min-w-0">
        <div
          className="text-sm text-foreground truncate [&_mark]:bg-yellow-500/30 [&_mark]:text-foreground [&_mark]:rounded-sm [&_mark]:px-0.5"
          dangerouslySetInnerHTML={{ __html: item.snippet || escapeHtml(item.title) }}
        />
        <div className="text-xs text-muted-foreground truncate flex items-center gap-1.5 mt-0.5">
          {item.subtitle && <span className="truncate">{item.subtitle}</span>}
          {item.category && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {item.category}
            </Badge>
          )}
          {showFinance && (
            <span className="tabular-nums whitespace-nowrap">
              {typeof yesPrice === 'number' && <>{t('aiCommandBar.yesPrice', { price: yesPrice.toFixed(2) })}</>}
              {typeof yesPrice === 'number' && typeof liquidity === 'number' && ' · '}
              {typeof liquidity === 'number' && <>{t('aiCommandBar.liquidityShort', { value: formatCompact(liquidity) })}</>}
            </span>
          )}
        </div>
      </div>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onTogglePin()
        }}
        className={cn(
          'opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-md hover:bg-background',
          pinned && 'opacity-100'
        )}
        title={pinned ? t('aiCommandBar.unpin') : t('aiCommandBar.pin')}
      >
        {pinned ? (
          <PinOff className="w-3.5 h-3.5 text-muted-foreground" />
        ) : (
          <Pin className="w-3.5 h-3.5 text-muted-foreground" />
        )}
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}
