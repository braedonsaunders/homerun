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
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowDown,
  ArrowUp,
  Calendar,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  CircleStop,
  Clock,
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
  Trash2,
  X,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import {
  type DatasetColumn,
  type DatasetFilter,
  type DatasetFilterValues,
  type DatasetQueryResult,
  type DatasetSummary,
  datasetCsvUrl,
  getDatasetStorageSummary,
  listDatasets,
  queryDataset,
} from '../services/apiDataset'
import {
  deleteMLData,
  getMLDataStats,
  getMLRecorderConfig,
  pruneMLData,
  updateMLRecorderConfig,
  type MLRecorderConfig,
} from '../services/apiMachineLearning'

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
                ? 'border-violet-500/50 bg-violet-500/10 text-violet-200'
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
  const hasTimeStart = filters.some((f) => f.kind === 'time_range_start')
  const hasTimeEnd = filters.some((f) => f.kind === 'time_range_end')

  return (
    <div className="flex flex-wrap items-end gap-2 rounded-md border border-border/40 bg-card/30 p-2">
      {/* Time presets */}
      {hasTimeStart || hasTimeEnd ? (
        <div className="flex items-center gap-1">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">Window</Label>
          {TIME_PRESETS.map((p) => {
            const isActive = timePreset === p.label
            return (
              <button
                key={p.label}
                onClick={() => onTimePreset(isActive ? null : p.label)}
                className={cn(
                  'rounded-sm border px-2 py-0.5 text-[10px] transition-colors',
                  isActive
                    ? 'border-violet-500/50 bg-violet-500/10 text-violet-200'
                    : 'border-border/40 bg-background/40 text-muted-foreground hover:border-border/60 hover:text-foreground',
                )}
              >
                {p.label}
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
                    placeholder={f.kind === 'contains' ? 'contains…' : '='}
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
          Reset
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
          className="flex items-center gap-1 rounded-sm bg-violet-500/15 px-1 py-0 text-[10px] text-violet-200"
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
  if (!row) return null
  return (
    <div className="absolute right-0 top-0 z-10 flex h-full w-[440px] flex-col border-l border-border/50 bg-background/95 shadow-2xl backdrop-blur">
      <div className="flex items-center justify-between border-b border-border/40 px-3 py-2">
        <div className="flex items-center gap-2">
          <Database className="h-3.5 w-3.5 text-violet-400" />
          <span className="text-xs font-semibold">{datasetLabel}</span>
          <span className="text-[10px] text-muted-foreground">row detail</span>
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
                    copy
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
        Columns ({visible.size}/{columns.length})
      </Button>
      {open ? (
        <div className="absolute right-0 top-full z-20 mt-1 w-56 rounded-md border border-border/60 bg-background/95 p-1.5 shadow-xl backdrop-blur">
          <div className="mb-1 px-1 text-[9px] uppercase tracking-wide text-muted-foreground">
            Visible columns
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
  const queryClient = useQueryClient()
  const storageQuery = useQuery({
    queryKey: ['data-lab', 'storage'],
    queryFn: getDatasetStorageSummary,
    refetchInterval: 60_000,
  })
  const data = storageQuery.data
  return (
    <div className="rounded-md border border-border/40 bg-card/30">
      <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
        <div className="flex items-center gap-2">
          <HardDrive className="h-3.5 w-3.5 text-violet-300" />
          <span className="text-xs font-semibold">Storage overview</span>
          <span className="text-[10px] text-muted-foreground">all datasets, on-disk</span>
        </div>
        <Button
          size="sm"
          variant="outline"
          className="h-6 gap-1 text-[10px]"
          onClick={() => queryClient.invalidateQueries({ queryKey: ['data-lab', 'storage'] })}
          disabled={storageQuery.isFetching}
        >
          <RefreshCw className={cn('h-3 w-3', storageQuery.isFetching && 'animate-spin')} />
          Refresh
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-3 px-3 py-3 md:grid-cols-3">
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">Total rows</div>
          <div className="font-mono text-sm tabular-nums">
            {data?.total_rows != null ? data.total_rows.toLocaleString() : '—'}
          </div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">On disk</div>
          <div className="font-mono text-sm tabular-nums">{fmtBytes(data?.total_bytes)}</div>
        </div>
        <div className="rounded-md border border-border/30 bg-background/40 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wide text-muted-foreground">Tables</div>
          <div className="font-mono text-sm tabular-nums">{data?.tables.length ?? 0}</div>
        </div>
      </div>
      <div className="overflow-x-auto px-3 pb-3">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wide text-muted-foreground">
              <th className="py-1 pr-3 text-left">Dataset</th>
              <th className="py-1 pr-3 text-right">Rows</th>
              <th className="py-1 pr-3 text-right">Size</th>
              <th className="py-1 pr-3 text-left">Oldest</th>
              <th className="py-1 pr-3 text-left">Newest</th>
            </tr>
          </thead>
          <tbody>
            {(data?.tables ?? []).map((t) => (
              <tr key={t.name} className="border-t border-border/20">
                <td className="py-1 pr-3">
                  <span className="font-medium">{t.label}</span>
                  <span className="ml-1.5 font-mono text-[10px] text-muted-foreground/60">
                    {t.table_name}
                  </span>
                </td>
                <td className="py-1 pr-3 text-right font-mono tabular-nums">
                  {t.row_count.toLocaleString()}
                </td>
                <td className="py-1 pr-3 text-right font-mono tabular-nums">
                  {fmtBytes(t.size_bytes)}
                </td>
                <td className="py-1 pr-3 text-[10px] text-muted-foreground">
                  {fmtAge(t.oldest_at)}
                </td>
                <td className="py-1 pr-3 text-[10px] text-muted-foreground">
                  {fmtAge(t.newest_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function BackgroundRecorderSection() {
  const queryClient = useQueryClient()
  const cfgQuery = useQuery({
    queryKey: ['data-lab', 'recorder-config'],
    queryFn: getMLRecorderConfig,
    refetchInterval: 30_000,
  })
  const statsQuery = useQuery({
    queryKey: ['data-lab', 'recorder-stats'],
    queryFn: getMLDataStats,
    refetchInterval: 30_000,
  })
  const updateMutation = useMutation({
    mutationFn: updateMLRecorderConfig,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['data-lab', 'recorder-config'] }),
  })
  const pruneMutation = useMutation({
    mutationFn: () => pruneMLData(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'recorder-stats'] })
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'storage'] })
    },
  })
  const deleteMutation = useMutation({
    mutationFn: deleteMLData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'recorder-stats'] })
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'storage'] })
    },
  })

  const cfg = cfgQuery.data?.config
  const [intervalSeconds, setIntervalSeconds] = useState('60')
  const [retentionDays, setRetentionDays] = useState('90')
  const [assets, setAssets] = useState('')
  const [timeframes, setTimeframes] = useState('')

  useEffect(() => {
    if (cfg) {
      setIntervalSeconds(String(cfg.interval_seconds))
      setRetentionDays(String(cfg.retention_days))
      setAssets((cfg.assets ?? []).join(', '))
      setTimeframes((cfg.timeframes ?? []).join(', '))
    }
  }, [cfg?.interval_seconds, cfg?.retention_days, (cfg?.assets ?? []).join('|'), (cfg?.timeframes ?? []).join('|')])

  const isRecording = Boolean(cfg?.is_recording)

  const parseList = (s: string): string[] =>
    s.split(',').map((x) => x.trim()).filter(Boolean)

  const updateField = (patch: Partial<MLRecorderConfig>) => updateMutation.mutate(patch)

  const stats = statsQuery.data
  return (
    <div className="rounded-md border border-border/40 bg-card/30">
      <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
        <div className="flex items-center gap-2">
          <Clock className="h-3.5 w-3.5 text-violet-300" />
          <span className="text-xs font-semibold">Background recorder</span>
          <span className="text-[10px] text-muted-foreground">always-on, market-wide</span>
          <Badge
            variant="outline"
            className={cn(
              'text-[9px]',
              isRecording
                ? 'border-emerald-500/40 text-emerald-300'
                : 'border-border/40 text-muted-foreground',
            )}
          >
            {isRecording ? 'recording' : 'idle'}
          </Badge>
        </div>
        <Switch
          checked={isRecording}
          onCheckedChange={(checked) => updateField({ is_recording: checked })}
          disabled={updateMutation.isPending}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 px-3 py-3 md:grid-cols-4">
        <div className="space-y-1">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
            Tick interval (sec)
          </Label>
          <div className="flex gap-1">
            <Input
              type="number"
              min={5}
              max={3600}
              value={intervalSeconds}
              onChange={(e) => setIntervalSeconds(e.target.value)}
              className="h-7 text-[11px]"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[10px]"
              onClick={() => updateField({ interval_seconds: Number(intervalSeconds) })}
              disabled={updateMutation.isPending}
            >
              Set
            </Button>
          </div>
        </div>
        <div className="space-y-1">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
            Retention (days)
          </Label>
          <div className="flex gap-1">
            <Input
              type="number"
              min={1}
              max={365}
              value={retentionDays}
              onChange={(e) => setRetentionDays(e.target.value)}
              className="h-7 text-[11px]"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[10px]"
              onClick={() => updateField({ retention_days: Number(retentionDays) })}
              disabled={updateMutation.isPending}
            >
              Set
            </Button>
          </div>
        </div>
        <div className="space-y-1">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
            Assets (csv)
          </Label>
          <div className="flex gap-1">
            <Input
              value={assets}
              onChange={(e) => setAssets(e.target.value)}
              className="h-7 text-[11px]"
              placeholder="BTC, ETH"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[10px]"
              onClick={() => updateField({ assets: parseList(assets) })}
              disabled={updateMutation.isPending}
            >
              Set
            </Button>
          </div>
        </div>
        <div className="space-y-1">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
            Timeframes (csv)
          </Label>
          <div className="flex gap-1">
            <Input
              value={timeframes}
              onChange={(e) => setTimeframes(e.target.value)}
              className="h-7 text-[11px]"
              placeholder="1m, 5m, 15m"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[10px]"
              onClick={() => updateField({ timeframes: parseList(timeframes) })}
              disabled={updateMutation.isPending}
            >
              Set
            </Button>
          </div>
        </div>
      </div>

      {stats && stats.groups && stats.groups.length > 0 ? (
        <div className="border-t border-border/30 px-3 py-2">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            Recorded scope
          </div>
          <div className="grid gap-1 md:grid-cols-2 xl:grid-cols-4">
            {stats.groups.map((g) => (
              <div
                key={`${g.task_key}-${g.asset}-${g.timeframe}`}
                className="rounded-sm border border-border/30 bg-background/40 px-2 py-1 text-[10px]"
              >
                <div className="font-medium uppercase">
                  {g.asset}/{g.timeframe}
                </div>
                <div className="font-mono tabular-nums text-muted-foreground">
                  {g.count.toLocaleString()} snapshots
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="flex items-center gap-2 border-t border-border/30 px-3 py-2">
        <Button
          size="sm"
          variant="outline"
          className="h-7 gap-1 text-[10px]"
          onClick={() => pruneMutation.mutate()}
          disabled={pruneMutation.isPending}
        >
          {pruneMutation.isPending ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Trash2 className="h-3 w-3" />
          )}
          Prune older than retention
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="h-7 gap-1 text-[10px] text-rose-300 hover:bg-rose-500/10"
          onClick={() => {
            if (confirm('Delete ALL recorded ML data? This cannot be undone.')) {
              deleteMutation.mutate()
            }
          }}
          disabled={deleteMutation.isPending}
        >
          {deleteMutation.isPending ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Trash2 className="h-3 w-3" />
          )}
          Delete all
        </Button>
      </div>
    </div>
  )
}

function OnDemandSessionsSection() {
  return (
    <div className="rounded-md border border-dashed border-border/40 bg-card/20 p-3">
      <div className="flex items-center gap-2">
        <PlayCircle className="h-3.5 w-3.5 text-violet-300" />
        <span className="text-xs font-semibold">On-demand sessions</span>
        <Badge variant="outline" className="border-amber-500/30 text-[9px] text-amber-300">
          coming soon
        </Badge>
      </div>
      <p className="mt-1.5 text-[11px] text-muted-foreground">
        Capture targeted slices of market data on demand or on a schedule:
        pick markets, set the tick frequency, choose what to record (book L2,
        trade prints, deltas), and either run now or schedule for later.
        Each session writes to a labeled dataset that shows up in Browse.
      </p>
      <div className="mt-2 grid grid-cols-2 gap-2 opacity-50 md:grid-cols-4">
        <div className="rounded-sm border border-border/30 bg-background/40 px-2 py-1.5 text-[10px]">
          <div className="flex items-center gap-1 font-medium">
            <Layers3 className="h-3 w-3" />
            Markets
          </div>
          <div className="text-muted-foreground">multi-select picker</div>
        </div>
        <div className="rounded-sm border border-border/30 bg-background/40 px-2 py-1.5 text-[10px]">
          <div className="flex items-center gap-1 font-medium">
            <Clock className="h-3 w-3" />
            Tick interval
          </div>
          <div className="text-muted-foreground">100ms — 1h</div>
        </div>
        <div className="rounded-sm border border-border/30 bg-background/40 px-2 py-1.5 text-[10px]">
          <div className="flex items-center gap-1 font-medium">
            <Calendar className="h-3 w-3" />
            Schedule
          </div>
          <div className="text-muted-foreground">now / cron / window</div>
        </div>
        <div className="rounded-sm border border-border/30 bg-background/40 px-2 py-1.5 text-[10px]">
          <div className="flex items-center gap-1 font-medium">
            <CircleStop className="h-3 w-3" />
            Capture
          </div>
          <div className="text-muted-foreground">book / trades / deltas</div>
        </div>
      </div>
    </div>
  )
}

function RecordView() {
  return (
    <div className="flex flex-col gap-3 p-3">
      <StorageOverviewSection />
      <BackgroundRecorderSection />
      <OnDemandSessionsSection />
    </div>
  )
}


type DataLabMode = 'browse' | 'record'

export default function DataLab() {
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
            <span className="text-sm font-semibold leading-tight">Data Lab</span>
            {mode === 'browse' && activeSpec ? (
              <Badge variant="outline" className="text-[10px]">
                {activeSpec.row_count.toLocaleString()} rows
              </Badge>
            ) : null}
          </div>
          <p className="mt-0.5 truncate text-[10px] text-muted-foreground">
            {mode === 'browse'
              ? (activeSpec?.description
                ?? 'Browse data the orchestrator + scanner have already captured. Filter, preview, export, or hand it to the agent.')
              : 'Manage the background recorder, view storage usage, and (soon) run on-demand capture sessions for targeted markets / windows.'}
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
                  ? 'bg-violet-500/15 text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <Search className="h-3 w-3" />
              Browse
            </button>
            <button
              onClick={() => setMode('record')}
              className={cn(
                'flex items-center gap-1 rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                mode === 'record'
                  ? 'bg-violet-500/15 text-violet-200'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <PlayCircle className="h-3 w-3" />
              Record
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
            Refresh
          </Button>
        </div>
      </div>

      {/* RECORD MODE — bail early so the browse-only state below doesn't run */}
      {mode === 'record' ? (
        <ScrollArea className="flex-1 min-h-0">
          <RecordView />
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
        <span className="text-muted-foreground">
          {total > 0
            ? `Showing ${pageStart.toLocaleString()}–${pageEnd.toLocaleString()} of ${total.toLocaleString()}`
            : query.isLoading
            ? 'Loading…'
            : 'No rows match'}
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="font-mono text-muted-foreground">
          sort: {orderBy} {orderDir}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">Per page</Label>
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
            CSV
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
                          isSorted && 'text-violet-300',
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
              {query.isLoading ? (
                <tr>
                  <td
                    colSpan={renderedCols.length || 1}
                    className="px-2 py-6 text-center text-[11px] text-muted-foreground"
                  >
                    Loading…
                  </td>
                </tr>
              ) : query.isError ? (
                <tr>
                  <td
                    colSpan={renderedCols.length || 1}
                    className="px-2 py-6 text-center text-[11px] text-rose-300"
                  >
                    {(query.error as Error).message || 'Query failed'}
                  </td>
                </tr>
              ) : rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={renderedCols.length || 1}
                    className="px-2 py-6 text-center text-[11px] text-muted-foreground"
                  >
                    <Filter className="mx-auto mb-1 h-4 w-4 opacity-40" />
                    No rows match the current filters.
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
            : `Page ${Math.floor(offset / perPage) + 1} of ${
                Math.floor(lastOffset / perPage) + 1
              }`}
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

