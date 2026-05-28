/**
 * Data Lab — first-party browser for the five recorded datasets.
 *
 * Renders inside Research as a peer to Code Experiments and Backtest
 * Studio.  One uniform table component handles all five datasets;
 * column / filter shapes come from the backend's /api/dataset
 * router so adding a sixth dataset is zero frontend work.
 *
 * UX:
 *   - dataset picker pills across the top (label + row count)
 *   - filter bar below it (time range presets + dataset-specific
 *     filters) with a Reset button
 *   - toolbar (row range / per-page / refresh / column visibility /
 *     CSV download)
 *   - sticky-header table; click a column header to sort, click a
 *     row to open the JSON drawer
 *   - paginator footer with first/prev/next/last
 *   - right drawer with the full row payload
 */
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowDown,
  ArrowUp,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Database,
  Download,
  Filter,
  HardDrive,
  Layers3,
  Loader2,
  PlayCircle,
  RefreshCw,
  Search,
  Table as TableIcon,
  X,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { cn } from '../lib/utils'
import DataLabProviders from './DataLabProviders'
import DataLabTopics from './DataLabTopics'
import {
  type CreateRecordingSessionPayload,
  type DatasetColumn,
  type DatasetFilter,
  type DatasetFilterValues,
  type DatasetQueryResult,
  type DatasetSummary,
  type MicrostructureRecorderStatus,
  type ProactiveSubscriptionStatus,
  type RecordingCaptureType,
  type RecordingSession,
  type RecordingTargetKind,
  cancelRecordingSession,
  createRecordingSession,
  datasetCsvUrl,
  deleteRecordingSession,
  getDatasetStorageSummary,
  getMicrostructureRecorderStatus,
  getProactiveSubscriptionStatus,
  getRecordingState,
  setRecordingState,
  listDatasets,
  listRecordingSessions,
  queryDataset,
  startRecordingSession,
  stopRecordingSession,
} from '../services/apiDataset'

const PER_PAGE_OPTIONS = [50, 100, 250, 500] as const

const TIME_PRESETS: Array<{ label: string; hours: number | null }> = [
  { label: '1h', hours: 1 },
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
  { label: 'All', hours: null },
]

// ─── Helpers ───────────────────────────────────────────────────────────

function fmtNumber(n: number | string | null | undefined, digits = 2): string {
  if (n == null || n === '') return '—'
  const num = Number(n)
  if (!Number.isFinite(num)) return '—'
  return num.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  })
}

function fmtInt(n: number | string | null | undefined): string {
  if (n == null || n === '') return '—'
  const num = Number(n)
  if (!Number.isFinite(num)) return '—'
  return Math.round(num).toLocaleString()
}

function fmtDateTime(s: string | null | undefined): string {
  if (!s) return '—'
  try {
    const d = new Date(s)
    if (Number.isNaN(d.getTime())) return s
    return d.toLocaleString(undefined, {
      year: '2-digit',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
  } catch {
    return s
  }
}

function jsonPreview(value: unknown, max = 60): string {
  if (value == null) return ''
  try {
    const s = typeof value === 'string' ? value : JSON.stringify(value)
    if (s.length <= max) return s
    return s.slice(0, max) + '…'
  } catch {
    return String(value)
  }
}

function statusTone(status: string): 'good' | 'bad' | 'warn' | 'neutral' {
  const s = String(status || '').toLowerCase()
  if (s.includes('win') || s === 'ok' || s === 'closed_win' || s === 'resolved_win') return 'good'
  if (s.includes('loss') || s === 'failed' || s === 'closed_loss' || s === 'resolved_loss') return 'bad'
  if (s === 'partial' || s === 'pending' || s === 'submitted') return 'warn'
  return 'neutral'
}

// ─── Dataset picker ─────────────────────────────────────────────────────

function DatasetPicker({
  datasets,
  active,
  onSelect,
}: {
  datasets: DatasetSummary[]
  active: string | null
  onSelect: (name: string) => void
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {datasets.map((d) => {
        const isActive = d.name === active
        return (
          <button
            key={d.name}
            onClick={() => onSelect(d.name)}
            className={cn(
              'group flex items-center gap-2 rounded-md border px-2.5 py-1 text-[11px] transition-colors',
              isActive
                ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                : 'border-border/40 bg-card/40 text-muted-foreground hover:border-border/70 hover:text-foreground',
            )}
            title={d.description}
          >
            <span className="font-medium">{d.label}</span>
            <span
              className={cn(
                'rounded-sm px-1 py-0 font-mono text-[9px] tabular-nums',
                isActive
                  ? 'bg-violet-500/20 text-violet-100'
                  : 'bg-muted/40 text-muted-foreground group-hover:bg-muted/60',
              )}
            >
              {d.row_count.toLocaleString()}
            </span>
          </button>
        )
      })}
    </div>
  )
}

// ─── Filter bar ─────────────────────────────────────────────────────────

function FilterBar({
  filters,
  values,
  onChange,
  onReset,
  timePreset,
  onTimePreset,
}: {
  filters: DatasetFilter[]
  values: DatasetFilterValues
  onChange: (key: string, value: string | string[] | undefined) => void
  onReset: () => void
  timePreset: string | null
  onTimePreset: (label: string | null) => void
}) {
  const { t } = useTranslation()
  const hasTimeStart = filters.some((f) => f.kind === 'time_range_start')
  const hasTimeEnd = filters.some((f) => f.kind === 'time_range_end')

  return (
    <div className="flex flex-wrap items-end gap-2 rounded-md border border-border/40 bg-card/30 p-2">
      {/* Time presets */}
      {hasTimeStart || hasTimeEnd ? (
        <div className="flex items-center gap-1">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">{t('dataLab.window')}</Label>
          {TIME_PRESETS.map((p) => {
            const isActive = timePreset === p.label
            const displayLabel = p.label === 'All' ? t('dataLab.presetAll') : p.label
            return (
              <button
                key={p.label}
                onClick={() => onTimePreset(isActive ? null : p.label)}
                className={cn(
                  'rounded-sm border px-2 py-0.5 text-[10px] transition-colors',
                  isActive
                    ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                    : 'border-border/40 bg-background/40 text-muted-foreground hover:border-border/60 hover:text-foreground',
                )}
              >
                {displayLabel}
              </button>
            )
          })}
        </div>
      ) : null}

      {/* Per-filter inputs (excluding time range — handled by presets above) */}
      {filters
        .filter((f) => f.kind !== 'time_range_start' && f.kind !== 'time_range_end')
        .map((f) => {
          const v = values[f.key]
          if (f.kind === 'eq' || f.kind === 'contains') {
            return (
              <div key={f.key} className="flex flex-col gap-0.5">
                <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
                  {f.label}
                </Label>
                <div className="relative">
                  <Search className="absolute left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={typeof v === 'string' ? v : ''}
                    onChange={(e) => onChange(f.key, e.target.value || undefined)}
                    placeholder={f.kind === 'contains' ? t('dataLab.containsPlaceholder') : t('dataLab.eqPlaceholder')}
                    className="h-7 w-40 pl-6 text-[11px]"
                  />
                </div>
              </div>
            )
          }
          if (f.kind === 'enum_in') {
            // Multi-select enum chips.
            const arr = Array.isArray(v) ? v : v ? [v as string] : []
            return (
              <div key={f.key} className="flex flex-col gap-0.5">
                <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
                  {f.label}
                </Label>
                <div className="flex flex-wrap items-center gap-1">
                  <EnumFilter
                    selected={arr}
                    onChange={(next) => onChange(f.key, next.length ? next : undefined)}
                    placeholder={f.label}
                  />
                </div>
              </div>
            )
          }
          return null
        })}

      <div className="ml-auto flex items-center gap-1.5">
        <Button
          size="sm"
          variant="outline"
          className="h-7 gap-1 text-[10px]"
          onClick={onReset}
        >
          <X className="h-3 w-3" />
          {t('dataLab.reset')}
        </Button>
      </div>
    </div>
  )
}

function EnumFilter({
  selected,
  onChange,
  placeholder,
}: {
  selected: string[]
  onChange: (next: string[]) => void
  placeholder: string
}) {
  // For now, allow free text + chip removal.  When enum_values are
  // declared on the column we use them; otherwise the user can type.
  const [draft, setDraft] = useState('')
  return (
    <div className="flex h-7 items-center gap-1 rounded-sm border border-border/40 bg-background/40 px-1.5">
      {selected.map((s) => (
        <span
          key={s}
          className="flex items-center gap-1 rounded-sm bg-violet-500/15 px-1 py-0 text-[10px] text-violet-700 dark:text-violet-200"
        >
          {s}
          <button
            onClick={() => onChange(selected.filter((x) => x !== s))}
            className="hover:text-rose-300"
          >
            <X className="h-2.5 w-2.5" />
          </button>
        </span>
      ))}
      <input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && draft.trim()) {
            onChange(Array.from(new Set([...selected, draft.trim()])))
            setDraft('')
          }
        }}
        className="w-20 bg-transparent text-[11px] outline-none"
      />
    </div>
  )
}

// ─── Cell renderer ──────────────────────────────────────────────────────

function Cell({
  column,
  value,
}: {
  column: DatasetColumn
  value: unknown
}) {
  if (value == null) {
    return <span className="text-muted-foreground/50">—</span>
  }
  switch (column.type) {
    case 'datetime':
      return (
        <span className="font-mono text-[10px] tabular-nums" title={String(value)}>
          {fmtDateTime(String(value))}
        </span>
      )
    case 'int':
      return <span className="font-mono tabular-nums">{fmtInt(value as number)}</span>
    case 'float':
      return <span className="font-mono tabular-nums">{fmtNumber(value as number, 4)}</span>
    case 'json':
      return (
        <span
          className="font-mono text-[10px] text-muted-foreground"
          title={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
        >
          {jsonPreview(value, 40)}
        </span>
      )
    case 'enum': {
      const s = String(value)
      const tone = statusTone(s)
      return (
        <span
          className={cn(
            'rounded-sm px-1.5 py-0.5 text-[10px] font-medium',
            tone === 'good' && 'bg-emerald-500/15 text-emerald-200',
            tone === 'bad' && 'bg-rose-500/15 text-rose-200',
            tone === 'warn' && 'bg-amber-500/15 text-amber-200',
            tone === 'neutral' && 'bg-muted/40 text-muted-foreground',
          )}
        >
          {s}
        </span>
      )
    }
    case 'string':
    default: {
      const s = String(value)
      return (
        <span className="truncate" title={s}>
          {s.length > 60 ? s.slice(0, 60) + '…' : s}
        </span>
      )
    }
  }
}

// ─── Row drawer ─────────────────────────────────────────────────────────

function RowDrawer({
  row,
  columns,
  datasetLabel,
  onClose,
}: {
  row: Record<string, unknown> | null
  columns: DatasetColumn[]
  datasetLabel: string
  onClose: () => void
}) {
  const { t } = useTranslation()
  if (!row) return null
  return (
    <div className="absolute right-0 top-0 z-10 flex h-full w-[440px] flex-col border-l border-border/50 bg-background/95 shadow-2xl backdrop-blur">
      <div className="flex items-center justify-between border-b border-border/40 px-3 py-2">
        <div className="flex items-center gap-2">
          <Database className="h-3.5 w-3.5 text-violet-400" />
          <span className="text-xs font-semibold">{datasetLabel}</span>
          <span className="text-[10px] text-muted-foreground">{t('dataLab.rowDetail')}</span>
        </div>
        <button onClick={onClose} className="rounded-sm p-1 text-muted-foreground hover:bg-muted/40 hover:text-foreground">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-1 p-3">
          {columns.map((c) => {
            const v = row[c.key]
            if (v == null) return null
            const display =
              c.type === 'json'
                ? typeof v === 'string'
                  ? v
                  : JSON.stringify(v, null, 2)
                : c.type === 'datetime'
                ? fmtDateTime(String(v))
                : String(v)
            return (
              <div key={c.key} className="space-y-0.5">
                <div className="flex items-center justify-between">
                  <span className="text-[9px] uppercase tracking-wide text-muted-foreground">
                    {c.label}
                  </span>
                  <button
                    onClick={() => navigator.clipboard?.writeText(String(display))}
                    className="text-[9px] text-muted-foreground/60 hover:text-foreground"
                  >
                    {t('dataLab.copy')}
                  </button>
                </div>
                <pre
                  className={cn(
                    'overflow-x-auto whitespace-pre-wrap break-words rounded-sm bg-muted/20 px-2 py-1 text-[10px] leading-relaxed',
                    c.type === 'json' || c.type === 'string'
                      ? 'font-mono'
                      : 'font-mono tabular-nums',
                  )}
                >
                  {display || '—'}
                </pre>
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}

// ─── Column visibility menu ─────────────────────────────────────────────

function ColumnMenu({
  columns,
  visible,
  onToggle,
}: {
  columns: DatasetColumn[]
  visible: Set<string>
  onToggle: (key: string) => void
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <Button
        size="sm"
        variant="outline"
        className="h-7 gap-1 text-[10px]"
        onClick={() => setOpen((v) => !v)}
      >
        <TableIcon className="h-3 w-3" />
        {t('dataLab.columns', { visible: visible.size, total: columns.length })}
      </Button>
      {open ? (
        <div className="absolute right-0 top-full z-20 mt-1 w-56 rounded-md border border-border/60 bg-background/95 p-1.5 shadow-xl backdrop-blur">
          <div className="mb-1 px-1 text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.visibleColumns')}
          </div>
          <div className="max-h-72 space-y-0.5 overflow-y-auto">
            {columns.map((c) => {
              const checked = visible.has(c.key)
              return (
                <label
                  key={c.key}
                  className="flex cursor-pointer items-center gap-1.5 rounded-sm px-1.5 py-0.5 text-[11px] hover:bg-muted/40"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => onToggle(c.key)}
                    className="h-3 w-3"
                  />
                  <span className={cn(checked ? 'text-foreground' : 'text-muted-foreground')}>
                    {c.label}
                  </span>
                  <span className="ml-auto text-[9px] text-muted-foreground/60">{c.type}</span>
                </label>
              )
            })}
          </div>
        </div>
      ) : null}
    </div>
  )
}

// ─── Main component ─────────────────────────────────────────────────────

// ─── Record view ────────────────────────────────────────────────────────
//
// "Record mode" is the operator's control surface for getting NEW data
// into the system, complementary to Browse mode which reads what the
// orchestrator + scanner have already passively recorded.  Three
// sections:
//   * Storage Overview — per-table row count + on-disk size + age window
//   * Background Recorder — the always-on ML recorder (interval, retention,
//     assets, timeframes; powered by /ml/recorder/config)
//   * On-Demand Sessions — targeted captures triggered by the operator
//     for a specific market / window (UI scaffold; see notes)
// ────────────────────────────────────────────────────────────────────────

function fmtBytes(b: number | null | undefined): string {
  if (b == null) return '—'
  if (b === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = Math.abs(Number(b))
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v < 10 && i > 0 ? 2 : v < 100 && i > 0 ? 1 : 0)} ${units[i]}`
}

function fmtAge(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    const t = new Date(iso).getTime()
    if (!Number.isFinite(t)) return '—'
    const diffSec = (Date.now() - t) / 1000
    if (diffSec < 60) return `${Math.round(diffSec)}s ago`
    if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ago`
    if (diffSec < 86400) return `${Math.round(diffSec / 3600)}h ago`
    return `${Math.round(diffSec / 86400)}d ago`
  } catch {
    return '—'
  }
}

function StorageOverviewSection() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const storageQuery = useQuery({
    queryKey: ['data-lab', 'storage'],
    queryFn: getDatasetStorageSummary,
    refetchInterval: 60_000,
  })
  const data = storageQuery.data
  // Show skeleton on the very first load only — once we have data,
  // background refetches keep the previous values visible (the
  // sweeping refresh icon in the header signals the refetch).  This
  // is the institutional pattern: never replace populated data with
  // a shimmer, but always show one while we have nothing.
  const showSkeleton = storageQuery.isLoading || !data
  return (
    <div className="rounded-md border border-border/40 bg-card/30">
      <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
        <div className="flex items-center gap-2">
          <HardDrive className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
          <span className="text-xs font-semibold">{t('dataLab.storageOverview')}</span>
          <span className="text-[10px] text-muted-foreground">{t('dataLab.storageOverviewSub')}</span>
          {storageQuery.isFetching && data ? (
            // Subtle "still working" indicator on background refetches
            // that DOES NOT wipe the existing data.
            <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
          ) : null}
        </div>
        <Button
          size="sm"
          variant="outline"
          className="h-6 gap-1 text-[10px]"
          onClick={() => queryClient.invalidateQueries({ queryKey: ['data-lab', 'storage'] })}
          disabled={storageQuery.isFetching}
        >
          <RefreshCw className={cn('h-3 w-3', storageQuery.isFetching && 'animate-spin')} />
          {t('dataLab.refresh')}
        </Button>
      </div>

      {/* KPI tiles — three across.  Skeleton mirrors the same grid +
          tile dimensions so the layout doesn't jump when data lands. */}
      <div className="grid grid-cols-2 gap-3 px-3 py-3 md:grid-cols-3">
        {showSkeleton ? (
          <>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="rounded-md border border-border/30 bg-background/40 px-3 py-2"
              >
                <div className="h-2 w-16 animate-pulse rounded bg-muted/60" />
                <div
                  className="mt-2 h-5 w-24 animate-pulse rounded bg-muted/80"
                  style={{ animationDuration: '1.6s' }}
                />
              </div>
            ))}
          </>
        ) : (
          <>
            <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
              <div className="text-[9px] uppercase tracking-wide text-muted-foreground">{t('dataLab.totalRows')}</div>
              <div className="font-mono text-sm tabular-nums">
                {data?.total_rows != null ? data.total_rows.toLocaleString() : '—'}
              </div>
            </div>
            <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
              <div className="text-[9px] uppercase tracking-wide text-muted-foreground">{t('dataLab.onDisk')}</div>
              <div className="font-mono text-sm tabular-nums">{fmtBytes(data?.total_bytes)}</div>
            </div>
            <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
              <div className="text-[9px] uppercase tracking-wide text-muted-foreground">{t('dataLab.tables')}</div>
              <div className="font-mono text-sm tabular-nums">{data?.tables.length ?? 0}</div>
            </div>
          </>
        )}
      </div>

      {/* Per-table breakdown.  Skeleton renders 6 ghost rows (the
          typical real count) so the eventual layout is predicted
          accurately — operator immediately sees how much content is
          coming. */}
      <div className="overflow-x-auto px-3 pb-3">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wide text-muted-foreground">
              <th className="py-1 pr-3 text-left">{t('dataLab.colDataset')}</th>
              <th className="py-1 pr-3 text-right">{t('dataLab.colRows')}</th>
              <th className="py-1 pr-3 text-right">{t('dataLab.colSize')}</th>
              <th className="py-1 pr-3 text-left">{t('dataLab.colOldest')}</th>
              <th className="py-1 pr-3 text-left">{t('dataLab.colNewest')}</th>
            </tr>
          </thead>
          <tbody>
            {showSkeleton ? (
              [0, 1, 2, 3, 4, 5].map((i) => (
                <tr key={i} className="border-t border-border/20">
                  <td className="py-1.5 pr-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-2.5 w-32 animate-pulse rounded bg-muted/70"
                        style={{ animationDelay: `${i * 80}ms` }}
                      />
                      <div
                        className="h-2 w-20 animate-pulse rounded bg-muted/40"
                        style={{ animationDelay: `${i * 80}ms` }}
                      />
                    </div>
                  </td>
                  <td className="py-1.5 pr-3">
                    <div
                      className="ml-auto h-2.5 w-16 animate-pulse rounded bg-muted/70"
                      style={{ animationDelay: `${i * 80}ms` }}
                    />
                  </td>
                  <td className="py-1.5 pr-3">
                    <div
                      className="ml-auto h-2.5 w-14 animate-pulse rounded bg-muted/70"
                      style={{ animationDelay: `${i * 80}ms` }}
                    />
                  </td>
                  <td className="py-1.5 pr-3">
                    <div
                      className="h-2 w-12 animate-pulse rounded bg-muted/50"
                      style={{ animationDelay: `${i * 80}ms` }}
                    />
                  </td>
                  <td className="py-1.5 pr-3">
                    <div
                      className="h-2 w-12 animate-pulse rounded bg-muted/50"
                      style={{ animationDelay: `${i * 80}ms` }}
                    />
                  </td>
                </tr>
              ))
            ) : (
              (data?.tables ?? []).map((row) => (
                <tr key={row.name} className="border-t border-border/20">
                  <td className="py-1 pr-3">
                    <span className="font-medium">{row.label}</span>
                    <span className="ml-1.5 font-mono text-[10px] text-muted-foreground/60">
                      {row.table_name}
                    </span>
                  </td>
                  <td className="py-1 pr-3 text-right font-mono tabular-nums">
                    {row.row_count.toLocaleString()}
                  </td>
                  <td className="py-1 pr-3 text-right font-mono tabular-nums">
                    {fmtBytes(row.size_bytes)}
                  </td>
                  <td className="py-1 pr-3 text-[10px] text-muted-foreground">
                    {fmtAge(row.oldest_at)}
                  </td>
                  <td className="py-1 pr-3 text-[10px] text-muted-foreground">
                    {fmtAge(row.newest_at)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer caption — only shown while loading, calls out that
          the slow query is in progress so the operator doesn't think
          the UI hung.  Disappears once data lands. */}
      {showSkeleton ? (
        <div className="border-t border-border/30 px-3 py-1.5 text-[10px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <Loader2 className="h-3 w-3 animate-spin" />
            {t('dataLab.storageLoading', { defaultValue: 'Computing per-table row counts and on-disk sizes — this query inspects every dataset table and can take a few seconds.' })}
          </span>
        </div>
      ) : null}
    </div>
  )
}

function MicrostructureRecorderSection() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const statusQuery = useQuery<MicrostructureRecorderStatus>({
    queryKey: ['data-lab', 'recorder-microstructure'],
    queryFn: getMicrostructureRecorderStatus,
    refetchInterval: 5_000,
  })
  const recordingQuery = useQuery({
    queryKey: ['data-lab', 'recording-state'],
    queryFn: getRecordingState,
    refetchInterval: 5_000,
  })
  const recordingEnabled = recordingQuery.data?.enabled ?? true
  const actual = recordingQuery.data?.actual_recording
  const toggleRecording = useMutation({
    mutationFn: (enabled: boolean) => setRecordingState(enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'recording-state'] })
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'recorder-microstructure'] })
    },
  })
  const s = statusQuery.data
  const running = Boolean(s?.running)
  const acceptPct =
    s?.accept_rate != null ? `${(s.accept_rate * 100).toFixed(1)}%` : '—'
  const rejectsTotal = Object.values(s?.rejects_by_reason || {}).reduce(
    (a, b) => a + (b ?? 0),
    0,
  )
  const visibleRejects = Object.entries(s?.rejects_by_reason || {})
    .filter(([, n]) => (n ?? 0) > 0)
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))

  return (
    <div className="rounded-md border border-border/40 bg-card/30">
      <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
        <div className="flex items-center gap-2 flex-wrap">
          <Layers3 className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
          <span className="text-xs font-semibold">{t('dataLab.ingestorTitle')}</span>
          <span className="text-[10px] text-muted-foreground">
            {t('dataLab.ingestorSub')}
          </span>
          <Badge
            variant="outline"
            className={cn(
              'text-[9px]',
              running
                ? 'border-emerald-500/40 text-emerald-300'
                : 'border-border/40 text-muted-foreground',
            )}
          >
            {running ? t('dataLab.capturing') : t('dataLab.idle')}
          </Badge>
          {actual ? (
            <span className="text-[9px] text-muted-foreground">
              {actual.actively_recording
                ? `${actual.distinct_tokens} tokens · ${actual.book_rows.toLocaleString()} books / ${actual.trade_rows.toLocaleString()} deltas (last ${actual.window_minutes}m, ${actual.source})`
                : recordingEnabled
                ? 'no recent parquet activity'
                : 'recording OFF'}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={recordingEnabled ? 'outline' : 'destructive'}
            className="h-6 gap-1 text-[10px]"
            onClick={() => toggleRecording.mutate(!recordingEnabled)}
            disabled={toggleRecording.isPending || recordingQuery.isLoading}
            title="Global recording master switch — turns ALL market-data recording on/off"
          >
            <span
              className={cn(
                'inline-block h-2 w-2 rounded-full',
                recordingEnabled ? 'bg-emerald-400' : 'bg-rose-400',
              )}
            />
            {recordingEnabled ? 'Recording ON' : 'Recording OFF'}
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 text-[10px]"
            onClick={() =>
              queryClient.invalidateQueries({
                queryKey: ['data-lab', 'recorder-microstructure'],
              })
            }
            disabled={statusQuery.isFetching}
          >
            <RefreshCw className={cn('h-3 w-3', statusQuery.isFetching && 'animate-spin')} />
            {t('dataLab.refresh')}
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 px-3 py-3 md:grid-cols-3 lg:grid-cols-6">
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.tokensTracked')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {s?.tokens_tracked != null ? s.tokens_tracked.toLocaleString() : '—'}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.acceptRate')}
          </div>
          <div
            className={cn(
              'font-mono text-sm tabular-nums',
              (s?.accept_rate ?? 1) >= 0.99
                ? 'text-emerald-300'
                : (s?.accept_rate ?? 1) >= 0.95
                ? 'text-amber-300'
                : 'text-rose-300',
            )}
          >
            {acceptPct}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {(s?.accepted_books ?? 0).toLocaleString()} / {(s?.total_attempts ?? 0).toLocaleString()}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.sequenceGaps')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {s?.sequence_gaps_observed != null
              ? s.sequence_gaps_observed.toLocaleString()
              : '—'}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.queueDropsLabel')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {(s?.snapshot_queue_dropped ?? 0).toLocaleString()}
            <span className="text-muted-foreground"> / </span>
            {(s?.delta_queue_dropped ?? 0).toLocaleString()}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.queueDropsSub')}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.flushP50')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {s?.flush_latency_ms_p50 != null
              ? `${s.flush_latency_ms_p50.toFixed(1)} ms`
              : '—'}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.flushP50Sub')}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.flushP95')}
          </div>
          <div
            className={cn(
              'font-mono text-sm tabular-nums',
              (s?.flush_latency_ms_p95 ?? 0) > 500
                ? 'text-rose-300'
                : (s?.flush_latency_ms_p95 ?? 0) > 100
                ? 'text-amber-300'
                : '',
            )}
          >
            {s?.flush_latency_ms_p95 != null
              ? `${s.flush_latency_ms_p95.toFixed(1)} ms`
              : '—'}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.flushP95Sub')}
          </div>
        </div>
      </div>
      {rejectsTotal > 0 ? (
        <div className="border-t border-border/30 px-3 py-2">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.validationRejects', { n: rejectsTotal.toLocaleString() })}
          </div>
          <div className="flex flex-wrap gap-1">
            {visibleRejects.map(([reason, n]) => (
              <span
                key={reason}
                className="rounded-sm bg-rose-500/10 px-2 py-0.5 text-[10px] text-rose-200"
              >
                {reason} <span className="font-mono tabular-nums">{n.toLocaleString()}</span>
              </span>
            ))}
          </div>
        </div>
      ) : null}
      <div className="border-t border-border/30 px-3 py-2 text-[10px] text-muted-foreground">
        {t('dataLab.ingestorFootnote')}
      </div>
    </div>
  )
}


function ProactiveCoverageSection() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const statusQuery = useQuery<ProactiveSubscriptionStatus>({
    queryKey: ['data-lab', 'proactive-subscription'],
    queryFn: getProactiveSubscriptionStatus,
    refetchInterval: 15_000,
  })
  const s = statusQuery.data
  const hasRun = s != null && s.total_runs > 0
  const subscribed = s?.last_run_subscribed_count ?? 0
  const target = s?.last_run_target_count ?? 0
  const coveragePct =
    target > 0 ? Math.min(100, (subscribed / target) * 100) : 0
  const ageSec = s?.last_run_age_seconds ?? null

  return (
    <div className="rounded-md border border-border/40 bg-card/30">
      <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
        <div className="flex items-center gap-2">
          <Layers3 className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
          <span className="text-xs font-semibold">{t('dataLab.proactiveCoverage')}</span>
          <span className="text-[10px] text-muted-foreground">
            {t('dataLab.proactiveCoverageSub')}
          </span>
          <Badge
            variant="outline"
            className={cn(
              'text-[9px]',
              hasRun && (s?.last_error == null)
                ? 'border-emerald-500/40 text-emerald-300'
                : hasRun && s?.last_error != null
                ? 'border-rose-500/40 text-rose-300'
                : 'border-border/40 text-muted-foreground',
            )}
          >
            {hasRun ? (s?.last_error ? t('dataLab.errorBadge') : t('dataLab.active')) : t('dataLab.idle')}
          </Badge>
        </div>
        <Button
          size="sm"
          variant="outline"
          className="h-6 gap-1 text-[10px]"
          onClick={() =>
            queryClient.invalidateQueries({
              queryKey: ['data-lab', 'proactive-subscription'],
            })
          }
          disabled={statusQuery.isFetching}
        >
          <RefreshCw className={cn('h-3 w-3', statusQuery.isFetching && 'animate-spin')} />
          {t('dataLab.refresh')}
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-3 px-3 py-3 md:grid-cols-4">
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.subscribed')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {subscribed.toLocaleString()}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.subscribedSub', { target: target.toLocaleString(), pct: coveragePct.toFixed(0) })}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.catalogMarkets')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {(s?.last_run_catalog_market_count ?? 0).toLocaleString()}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.catalogMarketsSub', { n: (s?.last_run_catalog_token_count ?? 0).toLocaleString() })}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.cap')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {(s?.max_tokens ?? 0).toLocaleString()}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.minLiquidity', { value: s?.min_liquidity_usd ?? 0 })}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">
            {t('dataLab.lastRun')}
          </div>
          <div className="font-mono text-sm tabular-nums">
            {ageSec != null ? t('dataLab.lastRunAgo', { n: Math.round(ageSec) }) : '—'}
          </div>
          <div className="text-[9px] text-muted-foreground">
            {t('dataLab.lastRunSub', { ms: (s?.last_run_duration_ms ?? 0).toFixed(0), n: s?.total_runs ?? 0 })}
          </div>
        </div>
      </div>

      {hasRun ? (
        <div className="border-t border-border/30 px-3 py-2 text-[10px]">
          <div className="mb-1 uppercase tracking-wide text-muted-foreground">{t('dataLab.funnel')}</div>
          <div className="flex flex-wrap gap-1">
            <span className="rounded-sm bg-muted/40 px-2 py-0.5 font-mono">
              {t('dataLab.funnelCatalog', { n: (s?.last_run_catalog_token_count ?? 0).toLocaleString() })}
            </span>
            <span className="rounded-sm bg-muted/40 px-2 py-0.5 font-mono">
              {t('dataLab.funnelDroppedLowLiq', { n: (s?.last_run_dropped_low_liquidity ?? 0).toLocaleString() })}
            </span>
            <span className="rounded-sm bg-muted/40 px-2 py-0.5 font-mono">
              {t('dataLab.funnelDroppedOverCap', { n: (s?.last_run_dropped_over_cap ?? 0).toLocaleString() })}
            </span>
            <span className="rounded-sm bg-violet-500/15 px-2 py-0.5 font-mono text-violet-700 dark:text-violet-200">
              {t('dataLab.funnelTarget', { n: target.toLocaleString() })}
            </span>
            <span className="rounded-sm bg-emerald-500/15 px-2 py-0.5 font-mono text-emerald-200">
              {t('dataLab.funnelSubscribed', { n: subscribed.toLocaleString() })}
            </span>
          </div>
        </div>
      ) : null}

      {s?.last_error ? (
        <div className="border-t border-border/30 px-3 py-2 text-[10px] text-rose-300">
          {t('dataLab.lastError', { msg: s.last_error })}
        </div>
      ) : null}

      <div className="border-t border-border/30 px-3 py-2 text-[10px] text-muted-foreground" dangerouslySetInnerHTML={{ __html: t('dataLab.proactiveFootnote', { interval: s?.loop_interval_seconds ?? 60, cap: s?.max_tokens ?? 8000, floor: s?.min_liquidity_usd ?? 10 }) }} />
    </div>
  )
}

// ─── On-demand recording sessions ──────────────────────────────────────

const CAPTURE_TYPE_OPTIONS: RecordingCaptureType[] = ['book', 'trade', 'delta']
const TARGET_KIND_OPTIONS: { value: RecordingTargetKind; label: string; hint: string }[] = [
  { value: 'token', label: 'Token IDs', hint: 'paste raw clob_token_id values' },
  { value: 'condition', label: 'Condition IDs', hint: 'expand to all outcome tokens' },
  { value: 'event', label: 'Event slugs', hint: 'expand to all markets in the event' },
]

function statusToTone(s: RecordingSession['status']): 'good' | 'bad' | 'warn' | 'neutral' {
  if (s === 'running') return 'good'
  if (s === 'completed') return 'good'
  if (s === 'failed') return 'bad'
  if (s === 'cancelled') return 'bad'
  if (s === 'paused') return 'warn'
  if (s === 'scheduled') return 'warn'
  return 'neutral'
}

function fmtDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '—'
  const sec = Math.round(ms / 1000)
  if (sec < 60) return `${sec}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`
  return `${Math.floor(sec / 86400)}d ${Math.floor((sec % 86400) / 3600)}h`
}

function NewSessionFlyout({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [targetKind, setTargetKind] = useState<RecordingTargetKind>('token')
  const [targetText, setTargetText] = useState('')
  const [captureTypes, setCaptureTypes] = useState<Set<RecordingCaptureType>>(
    new Set(['book', 'trade']),
  )
  const [tickIntervalMs, setTickIntervalMs] = useState('500')
  const [maxDurationMin, setMaxDurationMin] = useState('60')
  const [scheduledStart, setScheduledStart] = useState('')
  const [scheduledEnd, setScheduledEnd] = useState('')
  const [error, setError] = useState<string | null>(null)

  const queryClient = useQueryClient()
  const createMutation = useMutation({
    mutationFn: createRecordingSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'recording-sessions'] })
      setName('')
      setDescription('')
      setTargetText('')
      setError(null)
      onClose()
    },
    onError: (err) => setError((err as Error).message || t('dataLab.errCreateFailed')),
  })

  // Close on Escape, lock background scroll when open.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const submit = () => {
    setError(null)
    const targets = targetText
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean)
    if (!name.trim()) {
      setError(t('dataLab.errNameRequired'))
      return
    }
    if (targets.length === 0) {
      setError(t('dataLab.errAtLeastOneTarget'))
      return
    }
    if (captureTypes.size === 0) {
      setError(t('dataLab.errAtLeastOneCapture'))
      return
    }
    const tick = Math.max(50, Math.min(60_000, parseInt(tickIntervalMs, 10) || 500))
    const maxDurationSeconds =
      maxDurationMin && parseInt(maxDurationMin, 10) > 0
        ? parseInt(maxDurationMin, 10) * 60
        : null
    const payload: CreateRecordingSessionPayload = {
      name: name.trim(),
      description: description.trim() || undefined,
      target_kind: targetKind,
      target_values: targets,
      capture_types: Array.from(captureTypes),
      tick_interval_ms: tick,
      max_duration_seconds: maxDurationSeconds ?? undefined,
      scheduled_start_at: scheduledStart ? new Date(scheduledStart).toISOString() : null,
      scheduled_end_at: scheduledEnd ? new Date(scheduledEnd).toISOString() : null,
    }
    createMutation.mutate(payload)
  }

  return (
    <>
      {/* Scrim — click to close */}
      <div
        className={cn(
          'fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-opacity',
          open ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        onClick={onClose}
      />
      {/* Flyout panel — slides in from the right */}
      <div
        className={cn(
          'fixed inset-y-0 right-0 z-50 flex w-[480px] max-w-[95vw] flex-col border-l border-border/60 bg-background/95 shadow-2xl backdrop-blur transition-transform duration-200',
          open ? 'translate-x-0' : 'translate-x-full',
        )}
      >
        <div className="flex items-center justify-between border-b border-border/40 px-4 py-3">
          <div className="flex items-center gap-2">
            <PlayCircle className="h-4 w-4 text-violet-700 dark:text-violet-300" />
            <div>
              <div className="text-sm font-semibold leading-tight">{t('dataLab.newSessionTitle')}</div>
              <div className="text-[10px] text-muted-foreground leading-tight">
                {t('dataLab.newSessionSub')}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-sm p-1 text-muted-foreground hover:bg-muted/40 hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-4 p-4">
            {/* Identity */}
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {t('dataLab.identity')}
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">{t('dataLab.name')}</Label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={t('dataLab.namePlaceholder')}
                  className="h-8 text-[12px]"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">{t('dataLab.descriptionOptional')}</Label>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder={t('dataLab.descriptionPlaceholder')}
                  className="h-8 text-[12px]"
                />
              </div>
            </div>

            {/* Targeting */}
            <div className="space-y-2 rounded-md border border-border/40 bg-card/30 p-3">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {t('dataLab.whatToCapture')}
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">{t('dataLab.targetKind')}</Label>
                <div className="grid grid-cols-3 gap-1">
                  {TARGET_KIND_OPTIONS.map((o) => {
                    const active = targetKind === o.value
                    const labelKey = o.value === 'token' ? 'targetKindToken' : o.value === 'condition' ? 'targetKindCondition' : 'targetKindEvent'
                    const hintKey = o.value === 'token' ? 'targetKindTokenHint' : o.value === 'condition' ? 'targetKindConditionHint' : 'targetKindEventHint'
                    return (
                      <button
                        key={o.value}
                        onClick={() => setTargetKind(o.value)}
                        className={cn(
                          'rounded-sm border px-2 py-1.5 text-left transition-colors',
                          active
                            ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                            : 'border-border/40 bg-background/40 text-muted-foreground hover:text-foreground',
                        )}
                      >
                        <div className="text-[11px] font-medium">{t(`dataLab.${labelKey}`)}</div>
                        <div className="text-[9px] text-muted-foreground/80">{t(`dataLab.${hintKey}`)}</div>
                      </button>
                    )
                  })}
                </div>
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">
                  {t('dataLab.targets')} <span className="text-muted-foreground">{t('dataLab.targetsHint')}</span>
                </Label>
                <textarea
                  value={targetText}
                  onChange={(e) => setTargetText(e.target.value)}
                  placeholder={
                    targetKind === 'token'
                      ? t('dataLab.targetsTokenPlaceholder')
                      : targetKind === 'condition'
                      ? t('dataLab.targetsConditionPlaceholder')
                      : t('dataLab.targetsEventPlaceholder')
                  }
                  className="min-h-[80px] w-full rounded-sm border border-border/40 bg-background/60 px-2 py-1.5 font-mono text-[11px]"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">{t('dataLab.captureTypes')}</Label>
                <div className="flex flex-wrap gap-1.5">
                  {CAPTURE_TYPE_OPTIONS.map((capType) => {
                    const active = captureTypes.has(capType)
                    return (
                      <button
                        key={capType}
                        onClick={() => {
                          setCaptureTypes((prev) => {
                            const next = new Set(prev)
                            if (next.has(capType)) next.delete(capType)
                            else next.add(capType)
                            return next
                          })
                        }}
                        className={cn(
                          'rounded-sm border px-2.5 py-1 text-[11px] transition-colors',
                          active
                            ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                            : 'border-border/40 text-muted-foreground hover:text-foreground',
                        )}
                      >
                        {capType}
                      </button>
                    )
                  })}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {t('dataLab.captureLegend')}
                </div>
              </div>
            </div>

            {/* Cadence + window */}
            <div className="space-y-2 rounded-md border border-border/40 bg-card/30 p-3">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {t('dataLab.cadenceWindow')}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLab.tickIntervalMs')}</Label>
                  <Input
                    type="number"
                    min={50}
                    max={60_000}
                    step={50}
                    value={tickIntervalMs}
                    onChange={(e) => setTickIntervalMs(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLab.maxDurationMin')}</Label>
                  <Input
                    type="number"
                    min={1}
                    max={1440}
                    value={maxDurationMin}
                    onChange={(e) => setMaxDurationMin(e.target.value)}
                    className="h-8 text-[12px]"
                    placeholder={t('dataLab.maxDurationMinPlaceholder')}
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLab.scheduledStart')}</Label>
                  <Input
                    type="datetime-local"
                    value={scheduledStart}
                    onChange={(e) => setScheduledStart(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLab.scheduledEnd')}</Label>
                  <Input
                    type="datetime-local"
                    value={scheduledEnd}
                    onChange={(e) => setScheduledEnd(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
              </div>
            </div>

            {error ? (
              <div className="rounded-sm bg-rose-500/10 px-3 py-2 text-[12px] text-rose-200">
                {error}
              </div>
            ) : null}
          </div>
        </ScrollArea>

        <div className="flex items-center justify-between gap-2 border-t border-border/40 px-4 py-3">
          <span className="text-[10px] text-muted-foreground">
            {t('dataLab.willBeCreatedAs')}{' '}
            <strong className="text-foreground">
              {scheduledStart ? t('dataLab.scheduled') : t('dataLab.pending')}
            </strong>
            {scheduledStart ? '' : ` ${t('dataLab.willBeCreatedHint')}`}.
          </span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-8 text-[11px]" onClick={onClose}>
              {t('dataLab.cancel')}
            </Button>
            <Button
              size="sm"
              className="h-8 gap-1 text-[11px]"
              onClick={submit}
              disabled={createMutation.isPending}
            >
              {createMutation.isPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <PlayCircle className="h-3 w-3" />
              )}
              {t('dataLab.createSession')}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}

function SessionRow({ s }: { s: RecordingSession }) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const startMutation = useMutation({
    mutationFn: () => startRecordingSession(s.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['data-lab', 'recording-sessions'] }),
  })
  const stopMutation = useMutation({
    mutationFn: () => stopRecordingSession(s.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['data-lab', 'recording-sessions'] }),
  })
  const cancelMutation = useMutation({
    mutationFn: () => cancelRecordingSession(s.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['data-lab', 'recording-sessions'] }),
  })
  const deleteMutation = useMutation({
    mutationFn: () => deleteRecordingSession(s.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['data-lab', 'recording-sessions'] }),
  })

  const tone = statusToTone(s.status)
  const duration =
    s.started_at && (s.ended_at || s.status === 'running')
      ? new Date(s.ended_at ?? new Date().toISOString()).getTime() -
        new Date(s.started_at).getTime()
      : 0

  const isTerminal = ['completed', 'failed', 'cancelled'].includes(s.status)
  const isStartable = ['pending', 'scheduled'].includes(s.status)
  const isStoppable = s.status === 'running'

  return (
    <tr className="border-t border-border/20">
      <td className="px-2 py-1.5">
        <div className="font-medium">{s.name}</div>
        {s.description ? (
          <div className="text-[10px] text-muted-foreground">{s.description}</div>
        ) : null}
        <div className="font-mono text-[9px] text-muted-foreground/60">{s.id.slice(0, 12)}</div>
      </td>
      <td className="px-2 py-1.5 text-[10px]">
        <span
          className={cn(
            'rounded-sm px-1.5 py-0.5',
            tone === 'good' && 'bg-emerald-500/15 text-emerald-200',
            tone === 'bad' && 'bg-rose-500/15 text-rose-200',
            tone === 'warn' && 'bg-amber-500/15 text-amber-200',
            tone === 'neutral' && 'bg-muted/40 text-muted-foreground',
          )}
        >
          {s.status}
        </span>
        {s.error ? (
          <div className="mt-1 max-w-[180px] truncate text-[9px] text-rose-300" title={s.error}>
            {s.error}
          </div>
        ) : null}
      </td>
      <td className="px-2 py-1.5 text-[10px]">
        <div className="text-muted-foreground">
          {s.target_kind} · {s.target_values.length === 1 ? t('dataLab.targetCount', { n: s.target_values.length }) : t('dataLab.targetCountPlural', { n: s.target_values.length })}
        </div>
        {s.target_token_ids.length > 0 ? (
          <div className="font-mono text-[9px] text-muted-foreground/70">
            {s.target_token_ids.length === 1 ? t('dataLab.tokenResolved', { n: s.target_token_ids.length }) : t('dataLab.tokensResolved', { n: s.target_token_ids.length })}
          </div>
        ) : null}
      </td>
      <td className="px-2 py-1.5 text-[10px]">
        <div className="flex flex-wrap gap-0.5">
          {s.capture_types.map((c) => (
            <span key={c} className="rounded-sm bg-violet-500/10 px-1 py-0 text-[9px] text-violet-700 dark:text-violet-200">
              {c}
            </span>
          ))}
        </div>
        <div className="mt-0.5 text-[9px] text-muted-foreground">{s.tick_interval_ms} ms</div>
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-[10px] tabular-nums">
        {s.rows_captured.toLocaleString()}
      </td>
      <td className="px-2 py-1.5 text-[10px] text-muted-foreground">
        {duration > 0 ? fmtDuration(duration) : '—'}
      </td>
      <td className="px-2 py-1.5">
        <div className="flex gap-1">
          {isStartable ? (
            <button
              onClick={() => startMutation.mutate()}
              className="rounded-sm border border-emerald-500/40 px-1.5 py-0.5 text-[9px] text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
              disabled={startMutation.isPending}
            >
              {t('dataLab.start')}
            </button>
          ) : null}
          {isStoppable ? (
            <button
              onClick={() => stopMutation.mutate()}
              className="rounded-sm border border-amber-500/40 px-1.5 py-0.5 text-[9px] text-amber-300 hover:bg-amber-500/10 disabled:opacity-50"
              disabled={stopMutation.isPending}
            >
              {t('dataLab.stop')}
            </button>
          ) : null}
          {!isTerminal ? (
            <button
              onClick={() => cancelMutation.mutate()}
              className="rounded-sm border border-rose-500/40 px-1.5 py-0.5 text-[9px] text-rose-300 hover:bg-rose-500/10 disabled:opacity-50"
              disabled={cancelMutation.isPending}
            >
              {t('dataLab.cancel')}
            </button>
          ) : null}
          {isTerminal ? (
            <button
              onClick={() => {
                if (confirm(t('dataLab.confirmDeleteSession', { name: s.name }))) {
                  deleteMutation.mutate()
                }
              }}
              className="rounded-sm border border-border/40 px-1.5 py-0.5 text-[9px] text-muted-foreground hover:bg-muted/40 disabled:opacity-50"
              disabled={deleteMutation.isPending}
            >
              {t('dataLab.delete')}
            </button>
          ) : null}
        </div>
      </td>
    </tr>
  )
}

function OnDemandSessionsSection() {
  const { t } = useTranslation()
  const sessionsQuery = useQuery({
    queryKey: ['data-lab', 'recording-sessions'],
    queryFn: () => listRecordingSessions(undefined, 100),
    refetchInterval: 5_000,
  })
  const sessions = sessionsQuery.data ?? []
  const [flyoutOpen, setFlyoutOpen] = useState(false)

  return (
    <>
      <div className="rounded-md border border-border/40 bg-card/30">
        <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
          <div className="flex items-center gap-2">
            <PlayCircle className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
            <span className="text-xs font-semibold">{t('dataLab.onDemandSessions')}</span>
            <span className="text-[10px] text-muted-foreground">
              {t('dataLab.onDemandSessionsSub')}
            </span>
            <Badge variant="outline" className="text-[9px]">
              {sessions.length === 1 ? t('dataLab.sessionCount', { n: sessions.length }) : t('dataLab.sessionCountPlural', { n: sessions.length })}
            </Badge>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 text-[10px]"
            onClick={() => setFlyoutOpen(true)}
          >
            <PlayCircle className="h-3 w-3" />
            {t('dataLab.newSession')}
          </Button>
        </div>

        <div className="max-h-[480px] overflow-y-auto">
          <table className="w-full text-[11px]">
            <thead className="sticky top-0 bg-card/95 text-[9px] uppercase tracking-wide text-muted-foreground backdrop-blur">
              <tr>
                <th className="px-2 py-1.5 text-left">{t('dataLab.colSession')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLab.colStatus')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLab.colTargets')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLab.colCapture')}</th>
                <th className="px-2 py-1.5 text-right">{t('dataLab.colRowsRight')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLab.colDuration')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLab.colActions')}</th>
              </tr>
            </thead>
            <tbody>
              {sessions.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-2 py-6 text-center text-[11px] text-muted-foreground">
                    <PlayCircle className="mx-auto mb-1 h-4 w-4 opacity-40" />
                    <span dangerouslySetInnerHTML={{ __html: t('dataLab.noSessionsYet') }} />
                  </td>
                </tr>
              ) : (
                sessions.map((s) => <SessionRow key={s.id} s={s} />)
              )}
            </tbody>
          </table>
        </div>

        <div className="border-t border-border/30 px-3 py-2 text-[10px] text-muted-foreground" dangerouslySetInnerHTML={{ __html: t('dataLab.sessionsFootnote') }} />
      </div>

      <NewSessionFlyout open={flyoutOpen} onClose={() => setFlyoutOpen(false)} />
    </>
  )
}

type RecordTab = 'microstructure' | 'coverage' | 'sessions'

// NOTE: the former 'crypto' subtab drove the retired SQL ml_training_snapshots
// recorder. That recorder is gone — crypto market state is now archived by the
// recorded-event-bus (Topics → crypto.update.dispatch) and books by the
// live_ingestor parquet (Providers). So the subtab was removed.
const RECORD_TABS: { key: RecordTab; labelKey: string; icon: typeof Layers3 }[] = [
  { key: 'microstructure', labelKey: 'subTabMicrostructure', icon: Layers3 },
  { key: 'coverage', labelKey: 'subTabCoverage', icon: Layers3 },
  { key: 'sessions', labelKey: 'subTabSessions', icon: PlayCircle },
]

function RecordView() {
  const { t } = useTranslation()
  const [tab, setTab] = useState<RecordTab>('microstructure')
  return (
    <div className="flex flex-col gap-3 p-3">
      {/* Architecture banner — explains the unified ingestor model so
          the operator understands why "Microstructure" and "Proactive
          coverage" coexist (one is the worker, the other is the
          subscription manager that feeds it). */}
      <div className="rounded-md border border-border/40 bg-card/30 px-3 py-2 text-[11px] text-muted-foreground">
        <div className="font-medium text-foreground">
          {t('dataLab.architectureBanner')}
        </div>
        <div className="mt-1 leading-relaxed" dangerouslySetInnerHTML={{ __html: t('dataLab.architectureBody') }} />
      </div>
      {/* Sub-tabs — one per data source so each section gets the
          full vertical real estate instead of stacking. */}
      <div className="flex items-center gap-1 border-b border-border/30">
        {RECORD_TABS.map(({ key, labelKey, icon: Icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={cn(
              '-mb-px flex items-center gap-1.5 border-b-2 px-3 py-1.5 text-[11px] font-medium transition-colors',
              tab === key
                ? 'border-violet-500 text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
          >
            <Icon className="h-3 w-3" />
            {t(`dataLab.${labelKey}`)}
          </button>
        ))}
      </div>
      {tab === 'microstructure' ? <MicrostructureRecorderSection /> : null}
      {tab === 'coverage' ? <ProactiveCoverageSection /> : null}
      {tab === 'sessions' ? <OnDemandSessionsSection /> : null}
    </div>
  )
}

function StorageView() {
  return (
    <div className="flex flex-col gap-3 p-3">
      <StorageOverviewSection />
    </div>
  )
}


type DataLabMode = 'browse' | 'record' | 'storage' | 'providers' | 'topics'

export default function DataLab() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  // Top-level mode: Browse passively-recorded data vs. Record new data
  // (background recorder + on-demand sessions).  Discoverable as a
  // segmented control at the top — most users land in Browse.
  const [mode, setMode] = useState<DataLabMode>('browse')

  // Datasets summary (loaded once, refreshed every 5 min)
  const datasetsQuery = useQuery({
    queryKey: ['data-lab', 'datasets'],
    queryFn: listDatasets,
    refetchInterval: 5 * 60_000,
    staleTime: 60_000,
  })
  const datasets = datasetsQuery.data ?? []

  // Active dataset
  const [active, setActive] = useState<string | null>(null)
  useEffect(() => {
    if (active == null && datasets.length > 0) {
      setActive(datasets[0].name)
    }
  }, [active, datasets])
  const activeSpec = datasets.find((d) => d.name === active) ?? null

  // Filter + sort + paging state
  const [filters, setFilters] = useState<DatasetFilterValues>({})
  const [timePreset, setTimePreset] = useState<string | null>('24h')
  const [orderBy, setOrderBy] = useState<string | null>(null)
  const [orderDir, setOrderDir] = useState<'asc' | 'desc'>('desc')
  const [perPage, setPerPage] = useState<number>(100)
  const [offset, setOffset] = useState<number>(0)

  // Reset filters / sort when switching dataset
  useEffect(() => {
    if (!activeSpec) return
    setFilters({})
    setTimePreset('24h')
    setOrderBy(activeSpec.default_sort)
    setOrderDir(activeSpec.default_sort_dir)
    setOffset(0)
  }, [activeSpec?.name])

  // Apply time preset to filters before sending to API
  const filtersWithTime = useMemo(() => {
    const out: DatasetFilterValues = { ...filters }
    if (timePreset && activeSpec) {
      const preset = TIME_PRESETS.find((p) => p.label === timePreset)
      const hasStart = activeSpec.filters.some((f) => f.kind === 'time_range_start')
      if (preset && hasStart) {
        if (preset.hours == null) {
          delete out.start
          delete out.end
        } else {
          const end = new Date()
          const start = new Date(end.getTime() - preset.hours * 3600 * 1000)
          out.start = start.toISOString()
        }
      }
    }
    return out
  }, [filters, timePreset, activeSpec?.name])

  // Visible columns
  const [visibleCols, setVisibleCols] = useState<Set<string>>(new Set())
  useEffect(() => {
    if (!activeSpec) return
    setVisibleCols(new Set(activeSpec.columns.filter((c) => c.default_visible).map((c) => c.key)))
  }, [activeSpec?.name])

  // Query
  const query = useQuery({
    queryKey: [
      'data-lab',
      'query',
      active,
      filtersWithTime,
      orderBy,
      orderDir,
      perPage,
      offset,
    ],
    queryFn: () => {
      if (!active) return Promise.resolve(null)
      return queryDataset(active, {
        limit: perPage,
        offset,
        order_by: orderBy ?? undefined,
        order_dir: orderDir,
        filters: filtersWithTime,
      })
    },
    enabled: !!active,
    refetchInterval: 30_000, // stay fresh for live datasets
  })

  const result: DatasetQueryResult | null = query.data ?? null
  const total = result?.total ?? 0
  const rows = result?.rows ?? []
  const allCols = result?.columns ?? activeSpec?.columns ?? []
  const renderedCols = allCols.filter((c) => visibleCols.has(c.key))

  // True until we have a definitive response for the active dataset.
  // Covers: initial datasets metadata fetch, the gap before `active`
  // is set, and any state where we have not yet received a payload
  // (`query.data` undefined). Background refetches that already have
  // cached data do NOT trigger this — those keep showing stale rows
  // and just spin the Refresh icon, so the table doesn't flicker.
  const isLoadingRows =
    datasetsQuery.isLoading || !active || (query.isFetching && query.data == null)

  // Pagination math
  const lastOffset = Math.max(0, Math.floor((total - 1) / perPage) * perPage)
  const pageStart = total === 0 ? 0 : offset + 1
  const pageEnd = Math.min(offset + perPage, total)

  // Selected row drawer
  const [selectedRowIdx, setSelectedRowIdx] = useState<number | null>(null)
  const selectedRow = selectedRowIdx != null ? rows[selectedRowIdx] ?? null : null

  // CSV download
  const csvHref = useMemo(() => {
    if (!active) return ''
    return datasetCsvUrl(active, {
      filters: filtersWithTime,
      order_by: orderBy ?? undefined,
      order_dir: orderDir,
      columns: renderedCols.map((c) => c.key),
      max_rows: 50_000,
    })
  }, [active, filtersWithTime, orderBy, orderDir, renderedCols.map((c) => c.key).join(',')])

  return (
    <div className="relative flex h-full min-h-0 flex-col gap-2 p-3">
      {/* HEADER */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 shrink-0 text-violet-400" />
            <span className="text-sm font-semibold leading-tight">{t('dataLab.title')}</span>
            {mode === 'browse' && activeSpec ? (
              <Badge variant="outline" className="text-[10px]">
                {t('dataLab.rowsCount', { n: activeSpec.row_count.toLocaleString() })}
              </Badge>
            ) : null}
          </div>
          <p className="mt-0.5 truncate text-[10px] text-muted-foreground">
            {mode === 'browse'
              ? (activeSpec?.description
                ?? t('dataLab.subtitleBrowse'))
              : mode === 'record'
              ? t('dataLab.subtitleRecord')
              : mode === 'storage'
              ? t('dataLab.subtitleStorage')
              : t('dataLab.subtitleProviders')}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          {/* Mode toggle */}
          <div className="flex rounded-md border border-border/40 bg-card/40 p-0.5">
            <button
              onClick={() => setMode('browse')}
              className={cn(
                'flex items-center gap-1 rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                mode === 'browse'
                  ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <Search className="h-3 w-3" />
              {t('dataLab.modeBrowse')}
            </button>
            <button
              onClick={() => setMode('record')}
              className={cn(
                'flex items-center gap-1 rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                mode === 'record'
                  ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <PlayCircle className="h-3 w-3" />
              {t('dataLab.modeRecord')}
            </button>
            <button
              onClick={() => setMode('storage')}
              className={cn(
                'flex items-center gap-1 rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                mode === 'storage'
                  ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <HardDrive className="h-3 w-3" />
              {t('dataLab.modeStorage')}
            </button>
            <button
              onClick={() => setMode('providers')}
              className={cn(
                'flex items-center gap-1 rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                mode === 'providers'
                  ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <Download className="h-3 w-3" />
              {t('dataLab.modeProviders')}
            </button>
            <button
              onClick={() => setMode('topics')}
              className={cn(
                'flex items-center gap-1 rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                mode === 'topics'
                  ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              title="Recorded-event-bus topic catalog"
            >
              <Layers3 className="h-3 w-3" />
              Topics
            </button>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1 text-[10px]"
            onClick={() => queryClient.invalidateQueries({ queryKey: ['data-lab'] })}
            disabled={query.isFetching}
          >
            <RefreshCw className={cn('h-3 w-3', query.isFetching && 'animate-spin')} />
            {t('dataLab.refresh')}
          </Button>
        </div>
      </div>

      {/* RECORD MODE — bail early so the browse-only state below doesn't run */}
      {mode === 'record' ? (
        <ScrollArea className="flex-1 min-h-0">
          <RecordView />
        </ScrollArea>
      ) : null}

      {/* STORAGE MODE — per-dataset on-disk size + age window */}
      {mode === 'storage' ? (
        <ScrollArea className="flex-1 min-h-0">
          <StorageView />
        </ScrollArea>
      ) : null}

      {/* PROVIDERS MODE — external data import (polybacktest etc.) */}
      {mode === 'providers' ? (
        <ScrollArea className="flex-1 min-h-0">
          <DataLabProviders />
        </ScrollArea>
      ) : null}

      {/* TOPICS MODE — recorded-event-bus topic catalog */}
      {mode === 'topics' ? (
        <ScrollArea className="flex-1 min-h-0">
          <DataLabTopics />
        </ScrollArea>
      ) : null}

      {mode === 'browse' ? (
        <DatasetPicker datasets={datasets} active={active} onSelect={setActive} />
      ) : null}

      {/* FILTER BAR */}
      {mode === 'browse' && activeSpec ? (
        <FilterBar
          filters={activeSpec.filters}
          values={filters}
          onChange={(key, value) => {
            setFilters((prev) => {
              const next = { ...prev }
              if (value == null || (Array.isArray(value) && value.length === 0)) {
                delete next[key]
              } else {
                next[key] = value
              }
              return next
            })
            setOffset(0)
          }}
          onReset={() => {
            setFilters({})
            setTimePreset('24h')
            setOffset(0)
          }}
          timePreset={timePreset}
          onTimePreset={(label) => {
            setTimePreset(label)
            setOffset(0)
          }}
        />
      ) : null}

      {mode === 'browse' ? (
      <>
      {/* TOOLBAR */}
      <div className="flex flex-wrap items-center gap-2 text-[10px]">
        <span className="flex items-center gap-1 text-muted-foreground">
          {total > 0 ? (
            t('dataLab.showing', { start: pageStart.toLocaleString(), end: pageEnd.toLocaleString(), total: total.toLocaleString() })
          ) : isLoadingRows ? (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              {t('dataLab.loading')}
            </>
          ) : (
            t('dataLab.noRowsMatchShort')
          )}
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="font-mono text-muted-foreground">
          {t('dataLab.sortLabel', { by: orderBy, dir: orderDir })}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">{t('dataLab.perPage')}</Label>
          <select
            value={perPage}
            onChange={(e) => {
              setPerPage(parseInt(e.target.value, 10))
              setOffset(0)
            }}
            className="h-7 rounded-sm border border-border/40 bg-background/60 px-1.5 text-[10px]"
          >
            {PER_PAGE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <ColumnMenu
            columns={allCols}
            visible={visibleCols}
            onToggle={(key) => {
              setVisibleCols((prev) => {
                const next = new Set(prev)
                if (next.has(key)) next.delete(key)
                else next.add(key)
                return next
              })
            }}
          />
          <a
            href={csvHref}
            download
            target="_blank"
            rel="noreferrer"
            className={cn(
              'flex h-7 items-center gap-1 rounded-md border border-border/40 bg-background/40 px-2 text-[10px] hover:bg-muted/40',
              !active && 'pointer-events-none opacity-50',
            )}
          >
            <Download className="h-3 w-3" />
            {t('dataLab.csv')}
          </a>
        </div>
      </div>

      {/* TABLE */}
      <div className="relative flex-1 min-h-0 overflow-hidden rounded-md border border-border/40 bg-card/30">
        <ScrollArea className="h-full">
          <table className="w-full text-[11px]">
            <thead className="sticky top-0 z-10 border-b border-border/40 bg-background/95 backdrop-blur">
              <tr>
                {renderedCols.map((c) => {
                  const isSorted = orderBy === c.key
                  return (
                    <th
                      key={c.key}
                      className={cn(
                        'select-none border-b border-border/30 px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wide',
                        c.type === 'int' || c.type === 'float'
                          ? 'text-right'
                          : '',
                        c.sortable
                          ? 'cursor-pointer text-muted-foreground hover:text-foreground'
                          : 'text-muted-foreground/60',
                      )}
                      onClick={() => {
                        if (!c.sortable) return
                        if (orderBy === c.key) {
                          setOrderDir(orderDir === 'desc' ? 'asc' : 'desc')
                        } else {
                          setOrderBy(c.key)
                          setOrderDir('desc')
                        }
                        setOffset(0)
                      }}
                    >
                      <span
                        className={cn(
                          'inline-flex items-center gap-1',
                          isSorted && 'text-violet-700 dark:text-violet-300',
                        )}
                      >
                        {c.label}
                        {isSorted ? (
                          orderDir === 'desc' ? (
                            <ArrowDown className="h-2.5 w-2.5" />
                          ) : (
                            <ArrowUp className="h-2.5 w-2.5" />
                          )
                        ) : null}
                      </span>
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {isLoadingRows ? (
                <tr>
                  <td
                    colSpan={renderedCols.length || 1}
                    className="px-2 py-10 text-center text-[11px] text-muted-foreground"
                  >
                    <Loader2 className="mx-auto mb-1.5 h-4 w-4 animate-spin opacity-60" />
                    {t('dataLab.loading')}
                  </td>
                </tr>
              ) : query.isError ? (
                <tr>
                  <td
                    colSpan={renderedCols.length || 1}
                    className="px-2 py-6 text-center text-[11px] text-rose-300"
                  >
                    {(query.error as Error).message || t('dataLab.queryFailed')}
                  </td>
                </tr>
              ) : rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={renderedCols.length || 1}
                    className="px-2 py-6 text-center text-[11px] text-muted-foreground"
                  >
                    <Filter className="mx-auto mb-1 h-4 w-4 opacity-40" />
                    {t('dataLab.noRowsMatch')}
                  </td>
                </tr>
              ) : (
                rows.map((row, i) => {
                  const isActiveRow = selectedRowIdx === i
                  return (
                    <tr
                      key={i}
                      onClick={() => setSelectedRowIdx(isActiveRow ? null : i)}
                      className={cn(
                        'cursor-pointer border-b border-border/20 transition-colors',
                        isActiveRow
                          ? 'bg-violet-500/10'
                          : 'hover:bg-muted/30',
                      )}
                    >
                      {renderedCols.map((c) => (
                        <td
                          key={c.key}
                          className={cn(
                            'whitespace-nowrap px-2 py-1',
                            c.type === 'int' || c.type === 'float' ? 'text-right' : '',
                            c.type === 'string' && 'max-w-[260px] truncate',
                            c.type === 'json' && 'max-w-[200px] truncate',
                          )}
                        >
                          <Cell column={c} value={row[c.key]} />
                        </td>
                      ))}
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </ScrollArea>

        {/* Drawer for selected row */}
        {selectedRow ? (
          <RowDrawer
            row={selectedRow}
            columns={allCols}
            datasetLabel={activeSpec?.label ?? ''}
            onClose={() => setSelectedRowIdx(null)}
          />
        ) : null}
      </div>

      {/* PAGINATOR */}
      <div className="flex items-center justify-between gap-2 text-[10px]">
        <div className="text-muted-foreground">
          {total === 0
            ? '—'
            : t('dataLab.page', { cur: Math.floor(offset / perPage) + 1, total: Math.floor(lastOffset / perPage) + 1 })}
        </div>
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-1.5 text-[10px]"
            onClick={() => setOffset(0)}
            disabled={offset === 0 || query.isFetching}
          >
            <ChevronsLeft className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-1.5 text-[10px]"
            onClick={() => setOffset(Math.max(0, offset - perPage))}
            disabled={offset === 0 || query.isFetching}
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-1.5 text-[10px]"
            onClick={() => setOffset(Math.min(lastOffset, offset + perPage))}
            disabled={offset >= lastOffset || query.isFetching}
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-1.5 text-[10px]"
            onClick={() => setOffset(lastOffset)}
            disabled={offset >= lastOffset || query.isFetching}
          >
            <ChevronsRight className="h-3 w-3" />
          </Button>
        </div>
      </div>
      </>
      ) : null}
    </div>
  )
}

