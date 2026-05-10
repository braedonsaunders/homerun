import { type ReactNode, useState } from 'react'
import type { TFunction } from 'i18next'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'

export const DEFAULT_STRATEGY_KEY = 'btc_eth_maker_quote'

export type TradingScheduleDay = 'mon' | 'tue' | 'wed' | 'thu' | 'fri' | 'sat' | 'sun'

export type TradingScheduleDraft = {
  enabled: boolean
  days: TradingScheduleDay[]
  startTimeUtc: string
  endTimeUtc: string
  startDateUtc: string
  endDateUtc: string
  endAtUtc: string
}

export const TRADING_SCHEDULE_DAYS: TradingScheduleDay[] = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
export const TRADING_SCHEDULE_WEEKDAYS: TradingScheduleDay[] = ['mon', 'tue', 'wed', 'thu', 'fri']
export const TRADING_SCHEDULE_WEEKENDS: TradingScheduleDay[] = ['sat', 'sun']
export const TRADING_SCHEDULE_DAY_LABEL: Record<TradingScheduleDay, string> = {
  mon: 'Mon',
  tue: 'Tue',
  wed: 'Wed',
  thu: 'Thu',
  fri: 'Fri',
  sat: 'Sat',
  sun: 'Sun',
}

export const TRADING_SCHEDULE_DAY_LABEL_KEYS: Record<TradingScheduleDay, string> = {
  mon: 'tradingPanelShared.mon',
  tue: 'tradingPanelShared.tue',
  wed: 'tradingPanelShared.wed',
  thu: 'tradingPanelShared.thu',
  fri: 'tradingPanelShared.fri',
  sat: 'tradingPanelShared.sat',
  sun: 'tradingPanelShared.sun',
}

export function getTradingScheduleDayLabels(t: TFunction): Record<TradingScheduleDay, string> {
  return {
    mon: t('tradingPanelShared.mon'),
    tue: t('tradingPanelShared.tue'),
    wed: t('tradingPanelShared.wed'),
    thu: t('tradingPanelShared.thu'),
    fri: t('tradingPanelShared.fri'),
    sat: t('tradingPanelShared.sat'),
    sun: t('tradingPanelShared.sun'),
  }
}

export const DEFAULT_TRADING_SCHEDULE_DRAFT: TradingScheduleDraft = {
  enabled: false,
  days: [...TRADING_SCHEDULE_DAYS],
  startTimeUtc: '00:00',
  endTimeUtc: '23:59',
  startDateUtc: '',
  endDateUtc: '',
  endAtUtc: '',
}

function _isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function _toBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase()
    if (lowered === 'true' || lowered === '1' || lowered === 'yes') return true
    if (lowered === 'false' || lowered === '0' || lowered === 'no') return false
  }
  return fallback
}

export function normalizeTradingScheduleDay(value: unknown): TradingScheduleDay | null {
  const token = String(value || '').trim().toLowerCase()
  if (token.startsWith('mon')) return 'mon'
  if (token.startsWith('tue')) return 'tue'
  if (token.startsWith('wed')) return 'wed'
  if (token.startsWith('thu')) return 'thu'
  if (token.startsWith('fri')) return 'fri'
  if (token.startsWith('sat')) return 'sat'
  if (token.startsWith('sun')) return 'sun'
  return null
}

export function normalizeTradingScheduleDays(value: unknown): TradingScheduleDay[] {
  const raw = Array.isArray(value) ? value : []
  const next: TradingScheduleDay[] = []
  const seen = new Set<TradingScheduleDay>()
  for (const item of raw) {
    const normalized = normalizeTradingScheduleDay(item)
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    next.push(normalized)
  }
  return next.length > 0 ? next : [...TRADING_SCHEDULE_DAYS]
}

export function normalizeTradingScheduleDraft(value: unknown): TradingScheduleDraft {
  const raw = _isRecord(value) ? value : {}
  const normalizeDate = (input: unknown): string => {
    const text = String(input || '').trim()
    if (!text) return ''
    const datePart = text.includes('T') ? text.slice(0, 10) : text
    const parsed = new Date(`${datePart}T00:00:00Z`)
    if (Number.isNaN(parsed.getTime())) return ''
    return parsed.toISOString().slice(0, 10)
  }
  return {
    enabled: _toBoolean(raw.enabled, false),
    days: normalizeTradingScheduleDays(raw.days),
    startTimeUtc: String(raw.start_time || '00:00'),
    endTimeUtc: String(raw.end_time || '23:59'),
    startDateUtc: normalizeDate(raw.start_date),
    endDateUtc: normalizeDate(raw.end_date),
    endAtUtc: String(raw.end_at || '').trim(),
  }
}

export function buildTradingScheduleMetadata(schedule: TradingScheduleDraft): Record<string, unknown> {
  return {
    enabled: Boolean(schedule.enabled),
    days: normalizeTradingScheduleDays(schedule.days),
    start_time: String(schedule.startTimeUtc || '00:00'),
    end_time: String(schedule.endTimeUtc || '23:59'),
    start_date: schedule.startDateUtc || null,
    end_date: schedule.endDateUtc || null,
    end_at: schedule.endAtUtc || null,
  }
}

export type StrategyOptionDetail = {
  key: string
  label: string
  defaultParams: Record<string, unknown>
  paramFields: Array<Record<string, unknown>>
  version: number | null
  latestVersion: number | null
  versions: number[]
}

export type StrategyCatalogOption = {
  key: string
  label: string
  sourceKey: string
  sourceLabel: string
  detail: StrategyOptionDetail
}

export function normalizeStrategyKey(value: unknown): string {
  const key = String(value || '').trim().toLowerCase()
  return key || DEFAULT_STRATEGY_KEY
}

export function normalizeStrategyVersion(value: unknown): number | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return Math.trunc(value)
  }
  const raw = String(value || '').trim().toLowerCase()
  if (!raw || raw === 'latest') return null
  const parsed = Number(raw.startsWith('v') ? raw.slice(1) : raw)
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return Math.trunc(parsed)
}

export function normalizeVersionList(value: unknown): number[] {
  const raw = Array.isArray(value) ? value : []
  const seen = new Set<number>()
  const out: number[] = []
  for (const item of raw) {
    const normalized = normalizeStrategyVersion(item)
    if (normalized == null || seen.has(normalized)) continue
    seen.add(normalized)
    out.push(normalized)
  }
  out.sort((left, right) => right - left)
  return out
}

export function FlyoutSection({
  title,
  subtitle,
  icon: Icon,
  count,
  defaultOpen = true,
  iconClassName = 'text-orange-500',
  tone = 'default',
  children,
}: {
  title: string
  subtitle?: string
  icon: any
  count?: string
  defaultOpen?: boolean
  iconClassName?: string
  tone?: 'default' | 'danger'
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <Card
      className={cn(
        'rounded-xl shadow-none overflow-hidden',
        tone === 'danger' ? 'bg-red-500/5 border-red-500/25' : 'bg-card/40 border-border/40'
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className={cn(
          'w-full flex items-center justify-between gap-2 px-3 py-2 transition-colors border-b',
          tone === 'danger'
            ? 'border-red-500/20 hover:bg-red-500/10'
            : 'border-border/40 hover:bg-muted/25'
        )}
      >
        <div className="flex items-center gap-1.5">
          <Icon className={cn('w-3.5 h-3.5', iconClassName)} />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">{title}</h4>
        </div>
        <div className="flex items-center gap-1.5">
          {count ? (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/60 text-muted-foreground">
              {count}
            </span>
          ) : null}
          {open ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />}
        </div>
      </button>
      {open ? (
        <div className="px-3 py-3 space-y-3">
          {subtitle ? <p className="text-[10px] text-muted-foreground/70 -mt-0.5">{subtitle}</p> : null}
          {children}
        </div>
      ) : null}
    </Card>
  )
}
