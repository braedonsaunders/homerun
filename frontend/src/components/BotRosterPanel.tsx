import { memo, useMemo } from 'react'
import { useAtom } from 'jotai'
import { useTranslation } from 'react-i18next'
import { Loader2, Plus } from 'lucide-react'
import type { Trader } from '../services/apiTraders'
import {
  type BotRosterGroupBy,
  type BotRosterSort,
  botRosterGroupByAtom,
  botRosterHideInactiveAtom,
  botRosterSearchAtom,
  botRosterSortAtom,
} from '../store/atoms'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'

type TraderToggleAction = 'start' | 'stop' | 'activate' | 'deactivate'

export type BotRosterRow = {
  trader: Trader
  status: { key: string; label: string; dotClassName: string }
  sourceKeys: string[]
  sourceLabels: string[]
  primarySourceKey: string
  open: number
  resolved: number
  partialOpenBundles: number
  pnl: number
  latestActivityTs: number
  isInactive: boolean
}

type BotRosterPanelProps = {
  rows: BotRosterRow[]
  totalTraderCount: number
  globalResolvedPnl: number
  selectedTraderId: string | null
  setSelectedTraderId: (id: string | null) => void
  traderTogglePendingById: Record<string, TraderToggleAction>
  showCreatingTraderSkeleton: boolean
  creatingTraderPreview: { name: string; mode: 'shadow' | 'live' } | null
  selectedAccountMode: 'shadow' | 'live'
  openCreateTraderFlyout: () => void
  sourceLabelByKey: Record<string, string>
  sourceGroupOrderByKey: Map<string, number>
}

function fmtCurrency(value: number, compact = true): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 1 : 2,
  }).format(value)
}

function BotRosterPanelImpl({
  rows,
  totalTraderCount,
  globalResolvedPnl,
  selectedTraderId,
  setSelectedTraderId,
  traderTogglePendingById,
  showCreatingTraderSkeleton,
  creatingTraderPreview,
  selectedAccountMode,
  openCreateTraderFlyout,
  sourceLabelByKey,
  sourceGroupOrderByKey,
}: BotRosterPanelProps) {
  const { t } = useTranslation()
  const [search, setSearch] = useAtom(botRosterSearchAtom)
  const [hideInactive, setHideInactive] = useAtom(botRosterHideInactiveAtom)
  const [sort, setSort] = useAtom(botRosterSortAtom)
  const [groupBy, setGroupBy] = useAtom(botRosterGroupByAtom)

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()
    const filteredRows = rows.filter((row) => {
      if (hideInactive && row.isInactive) return false
      if (!query) return true
      const haystack = `${row.trader.name} ${row.trader.id} ${row.status.label} ${row.sourceLabels.join(' ')}`.toLowerCase()
      return haystack.includes(query)
    })
    filteredRows.sort((left, right) => {
      if (sort === 'name_asc') return left.trader.name.localeCompare(right.trader.name)
      if (sort === 'name_desc') return right.trader.name.localeCompare(left.trader.name)
      if (sort === 'pnl_desc' && left.pnl !== right.pnl) return right.pnl - left.pnl
      if (sort === 'pnl_asc' && left.pnl !== right.pnl) return left.pnl - right.pnl
      if (sort === 'open_desc' && left.open !== right.open) return right.open - left.open
      if (sort === 'activity_desc' && left.latestActivityTs !== right.latestActivityTs) {
        return right.latestActivityTs - left.latestActivityTs
      }
      if (left.pnl !== right.pnl) return right.pnl - left.pnl
      return left.trader.name.localeCompare(right.trader.name)
    })
    return filteredRows
  }, [hideInactive, rows, search, sort])

  const grouped = useMemo(() => {
    if (groupBy === 'none') {
      return [{ key: 'all', label: t('botRosterPanel.allBots'), order: 0, rows: filtered }]
    }
    const groups = new Map<string, { key: string; label: string; order: number; rows: BotRosterRow[] }>()
    for (const row of filtered) {
      let key = ''
      let label = ''
      let order = 99
      if (groupBy === 'status') {
        key = row.status.key
        if (row.status.key === 'running') { label = t('botRosterPanel.statusRunning'); order = 0 }
        else if (row.status.key === 'engine_stopped') { label = t('botRosterPanel.statusEngineStopped'); order = 1 }
        else if (row.status.key === 'bot_stopped') { label = t('botRosterPanel.statusBotStopped'); order = 2 }
        else { label = t('botRosterPanel.statusInactive'); order = 3 }
      } else {
        key = row.primarySourceKey
        if (key === 'multi') { label = t('botRosterPanel.groupMultiSource'); order = sourceGroupOrderByKey.size + 1 }
        else if (key === 'unknown') { label = t('botRosterPanel.groupUnassigned'); order = sourceGroupOrderByKey.size + 2 }
        else {
          label = sourceLabelByKey[key] || key.toUpperCase()
          order = sourceGroupOrderByKey.get(key) ?? sourceGroupOrderByKey.size
        }
      }
      if (!groups.has(key)) groups.set(key, { key, label, order, rows: [] })
      groups.get(key)!.rows.push(row)
    }
    return Array.from(groups.values()).sort((a, b) => {
      if (a.order !== b.order) return a.order - b.order
      return a.label.localeCompare(b.label)
    })
  }, [filtered, groupBy, sourceGroupOrderByKey, sourceLabelByKey, t])

  return (
    <div className="hidden xl:flex flex-col min-h-0 rounded-lg border border-border/70 bg-card overflow-hidden">
      <div className="shrink-0 px-2.5 py-2 border-b border-border/50 flex items-center justify-between gap-1">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t('botRosterPanel.headerBots')}</span>
        <Button size="sm" className="h-6 w-6 p-0" variant="outline" onClick={openCreateTraderFlyout} title={t('botRosterPanel.newBot')}>
          <Plus className="w-3.5 h-3.5" />
        </Button>
      </div>
      <div className="shrink-0 px-1.5 py-1.5 border-b border-border/50 space-y-1">
        <Input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={t('botRosterPanel.searchPlaceholder')}
          className="h-6 text-[10px]"
        />
        <div className="grid grid-cols-2 gap-1">
          <Select value={sort} onValueChange={(value) => setSort(value as BotRosterSort)}>
            <SelectTrigger className="h-6 text-[10px] px-2">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name_asc">{t('botRosterPanel.sortNameAsc')}</SelectItem>
              <SelectItem value="name_desc">{t('botRosterPanel.sortNameDesc')}</SelectItem>
              <SelectItem value="pnl_desc">{t('botRosterPanel.sortPnlDesc')}</SelectItem>
              <SelectItem value="pnl_asc">{t('botRosterPanel.sortPnlAsc')}</SelectItem>
              <SelectItem value="open_desc">{t('botRosterPanel.sortOpenOrders')}</SelectItem>
              <SelectItem value="activity_desc">{t('botRosterPanel.sortLatestActivity')}</SelectItem>
            </SelectContent>
          </Select>
          <Select value={groupBy} onValueChange={(value) => setGroupBy(value as BotRosterGroupBy)}>
            <SelectTrigger className="h-6 text-[10px] px-2">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="status">{t('botRosterPanel.groupByStatus')}</SelectItem>
              <SelectItem value="source">{t('botRosterPanel.groupBySource')}</SelectItem>
              <SelectItem value="none">{t('botRosterPanel.groupByNone')}</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <label className="h-6 rounded border border-border/50 bg-background/50 px-2 flex items-center gap-1.5">
          <Switch
            checked={hideInactive}
            onCheckedChange={setHideInactive}
            className="scale-[0.75]"
          />
          <span className="text-[10px] text-muted-foreground">{t('botRosterPanel.hideInactive')}</span>
          <span className="ml-auto text-[10px] font-mono text-muted-foreground">
            {filtered.length}/{totalTraderCount}
          </span>
        </label>
      </div>
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-1.5 space-y-1.5">
          <button
            type="button"
            onClick={() => setSelectedTraderId(null)}
            className={cn(
              'w-full text-left rounded-md px-2 py-1.5 text-[11px] transition-colors',
              !selectedTraderId
                ? 'bg-cyan-500/15 text-foreground font-medium'
                : 'text-muted-foreground hover:bg-muted/40'
            )}
          >
            <div className="flex min-w-0 items-center justify-between gap-2">
              <span className="truncate">{t('botRosterPanel.allBots')}</span>
              <span className={cn('shrink-0 font-mono text-[10px]', globalResolvedPnl > 0 ? 'text-emerald-500' : globalResolvedPnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                {fmtCurrency(globalResolvedPnl)}
              </span>
            </div>
            <div className="mt-0.5 truncate text-[9px] text-muted-foreground">
              {t('botRosterPanel.showingCount', { shown: filtered.length, total: totalTraderCount })}
            </div>
          </button>
          {showCreatingTraderSkeleton ? (
            <div className="w-full rounded-md border border-cyan-500/35 bg-cyan-500/10 px-2 py-1.5 animate-pulse">
              <div className="flex items-center justify-between gap-1.5">
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
                    <span className="text-[11px] font-medium truncate leading-tight text-foreground">
                      {creatingTraderPreview?.name || t('botRosterPanel.creatingBot')}
                    </span>
                  </div>
                  <div className="pl-3 mt-0.5 text-[9px] text-muted-foreground">
                    {t('botRosterPanel.provisioningBot', { mode: String(creatingTraderPreview?.mode || selectedAccountMode).toUpperCase() })}
                  </div>
                </div>
                <Loader2 className="w-3.5 h-3.5 text-cyan-300 animate-spin shrink-0" />
              </div>
            </div>
          ) : null}
          {grouped.length === 0 ? (
            <p className="py-6 text-center text-[11px] text-muted-foreground">{t('botRosterPanel.noBotsMatch')}</p>
          ) : (
            grouped.map((group) => {
              const groupPnl = group.rows.reduce((sum, row) => sum + row.pnl, 0)
              return (
                <div key={group.key} className="space-y-0.5">
                  {groupBy !== 'none' ? (
                    <div className="px-1 py-0.5 flex items-center justify-between">
                      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{group.label}</span>
                      <span className={cn('text-[9px] font-mono', groupPnl > 0 ? 'text-emerald-500' : groupPnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                        {group.rows.length} · {fmtCurrency(groupPnl)}
                      </span>
                    </div>
                  ) : null}
                  {group.rows.map((row) => {
                    const isActive = row.trader.id === selectedTraderId
                    const pendingToggleAction = traderTogglePendingById[row.trader.id] || null
                    const rowTogglePending = pendingToggleAction !== null
                    return (
                      <button
                        key={row.trader.id}
                        type="button"
                        onClick={() => setSelectedTraderId(row.trader.id)}
                        disabled={rowTogglePending}
                        className={cn(
                          'w-full text-left rounded-md px-2 py-1.5 transition-colors group',
                          isActive
                            ? 'bg-cyan-500/15 text-foreground'
                            : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground',
                          rowTogglePending && 'cursor-wait border border-cyan-500/35 bg-cyan-500/10 text-foreground/85'
                        )}
                      >
                        {rowTogglePending ? (
                          <div className="flex items-center justify-between gap-2">
                            <div className="min-w-0 flex-1 animate-pulse space-y-1">
                              <div className="flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-300/80 shrink-0" />
                                <div className="h-2.5 w-24 max-w-full rounded bg-muted/70" />
                              </div>
                              <div className="pl-3 h-2 w-28 max-w-full rounded bg-muted/60" />
                              <div className="pl-3 flex gap-1">
                                <span className="h-2 w-10 rounded bg-muted/60" />
                                <span className="h-2 w-12 rounded bg-muted/50" />
                              </div>
                            </div>
                            <div className="shrink-0 flex items-center gap-1 text-[9px] text-cyan-300">
                              <Loader2 className="w-3 h-3 animate-spin" />
                              {pendingToggleAction === 'start'
                                ? t('botRosterPanel.actionStarting')
                                : pendingToggleAction === 'stop'
                                  ? t('botRosterPanel.actionStopping')
                                  : pendingToggleAction === 'activate'
                                    ? t('botRosterPanel.actionActivating')
                                    : t('botRosterPanel.actionDeactivating')}
                            </div>
                          </div>
                        ) : (
                          <>
                            <div className="flex items-center justify-between gap-1.5">
                              <div className="min-w-0">
                                <div className="flex items-center gap-1.5">
                                  <span className={cn(
                                    'w-1.5 h-1.5 rounded-full shrink-0',
                                    row.status.dotClassName
                                  )} />
                                  <span className="text-[11px] font-medium truncate leading-tight">{row.trader.name}</span>
                                </div>
                                <div className="pl-3 mt-0.5 text-[9px] text-muted-foreground">
                                  <span>{t('botRosterPanel.openCount', { n: row.open })}</span>
                                  {row.partialOpenBundles > 0 && (
                                    <span className="text-amber-500" title={t('botRosterPanel.partialBundlesTooltip')}>
                                      {' · '}{t('botRosterPanel.partialCount', { n: row.partialOpenBundles })}
                                    </span>
                                  )}
                                  <span>{' · '}{t('botRosterPanel.resolvedCount', { n: row.resolved })}</span>
                                </div>
                              </div>
                              <span className={cn('shrink-0 text-[10px] font-mono', row.pnl > 0 ? 'text-emerald-500' : row.pnl < 0 ? 'text-red-500' : 'text-muted-foreground')}>
                                {fmtCurrency(row.pnl)}
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-0.5 mt-0.5 pl-3">
                              {row.sourceLabels.length > 0 ? row.sourceLabels.map((sourceLabel) => (
                                <span key={`${row.trader.id}:${sourceLabel}`} className="px-1 py-0 text-[8px] rounded bg-muted/60 text-muted-foreground leading-relaxed">{sourceLabel}</span>
                              )) : (
                                <span className="px-1 py-0 text-[8px] rounded bg-muted/60 text-muted-foreground leading-relaxed">{t('botRosterPanel.groupUnassigned')}</span>
                              )}
                              {row.trader.block_new_orders ? (
                                <span
                                  className="px-1 py-0 text-[8px] rounded bg-red-500/15 text-red-300 border border-red-500/30 leading-relaxed"
                                  title={t('botRosterPanel.noNewOrdersTooltip')}
                                >
                                  {t('botRosterPanel.noNewOrders')}
                                </span>
                              ) : null}
                            </div>
                          </>
                        )}
                      </button>
                    )
                  })}
                </div>
              )
            })
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

export const BotRosterPanel = memo(BotRosterPanelImpl)
