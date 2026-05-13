/**
 * Shared helpers for working with trader strategy parameters.
 *
 * Previously these lived inside ``TradingPanel.tsx`` — the autoresearch
 * panel needs the same machinery (param-section building, source/scope
 * normalization, draft → wire shape conversion) so they're extracted
 * here to avoid duplicating ~150 lines.
 */

import type { StrategyOptionDetail } from '../components/tradingPanelFlyoutShared'

// ── Type definitions ─────────────────────────────────────────────────────

export type StrategyParamGroupKey =
  | 'signal'
  | 'scope'
  | 'timing'
  | 'entry'
  | 'sizing'
  | 'exit'
  | 'risk'
  | 'advanced'

export type StrategyParamGroup = {
  key: StrategyParamGroupKey
  label: string
  fields: Array<Record<string, unknown>>
}

export type DynamicStrategyParamSection = {
  sectionKey: string
  sourceKey: string
  sourceLabel: string
  strategyLabel: string
  groups: StrategyParamGroup[]
  fieldKeys: string[]
  values: Record<string, unknown>
}

export type TradersScopeMode = 'tracked' | 'pool' | 'individual' | 'group'

export type NormalizedTradersScope = {
  modes: TradersScopeMode[]
  individual_wallets: string[]
  group_ids: string[]
}

// ── Constants ────────────────────────────────────────────────────────────

export const STRATEGY_PARAM_GROUP_ORDER: readonly StrategyParamGroupKey[] = [
  'signal',
  'scope',
  'timing',
  'entry',
  'sizing',
  'exit',
  'risk',
  'advanced',
] as const

export const STRATEGY_PARAM_GROUP_LABELS: Record<StrategyParamGroupKey, string> = {
  signal: 'Signal Detection',
  scope: 'Scope & Modes',
  timing: 'Timing & Freshness',
  entry: 'Entry Filters',
  sizing: 'Sizing',
  exit: 'Exit Controls',
  risk: 'Risk Guards',
  advanced: 'Advanced',
}

// ── Tiny utilities ───────────────────────────────────────────────────────

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export function csvToList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

export function toStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean)
  }
  if (typeof value === 'string') {
    return csvToList(value)
  }
  return []
}

export function normalizeSourceKey(value: string): string {
  return String(value || '').trim().toLowerCase()
}

// ── Scope normalization (traders source) ─────────────────────────────────

export function normalizeTradersScopeConfig(value: unknown): NormalizedTradersScope {
  const raw = isRecord(value) ? value : {}
  const modes: TradersScopeMode[] = []
  const seenModes = new Set<TradersScopeMode>()
  for (const rawMode of toStringList(raw.modes)) {
    const mode = String(rawMode || '').trim().toLowerCase()
    if (mode !== 'tracked' && mode !== 'pool' && mode !== 'individual' && mode !== 'group') continue
    if (seenModes.has(mode)) continue
    seenModes.add(mode)
    modes.push(mode)
  }
  const individual_wallets: string[] = []
  const seenWallets = new Set<string>()
  for (const rawWallet of toStringList(raw.individual_wallets)) {
    const wallet = String(rawWallet || '').trim().toLowerCase()
    if (!wallet || seenWallets.has(wallet)) continue
    seenWallets.add(wallet)
    individual_wallets.push(wallet)
  }
  const group_ids: string[] = []
  const seenGroups = new Set<string>()
  for (const rawGroupId of toStringList(raw.group_ids)) {
    const groupId = String(rawGroupId || '').trim()
    if (!groupId || seenGroups.has(groupId)) continue
    seenGroups.add(groupId)
    group_ids.push(groupId)
  }
  return {
    modes: modes.length > 0 ? modes : ['tracked', 'pool'],
    individual_wallets,
    group_ids,
  }
}

// ── Strategy params building / merging ──────────────────────────────────

/**
 * Merge user-edited params over the strategy's declared defaults.
 * For the ``traders`` source, additionally normalizes the embedded
 * ``traders_scope`` config so it always has the canonical shape; for
 * other sources, drops any stray ``traders_scope`` entry that may have
 * leaked in from a previous edit.
 */
export function buildSourceStrategyParams(
  raw: Record<string, unknown>,
  sourceKey: string,
  strategyDetail: StrategyOptionDetail | null,
): Record<string, unknown> {
  const strategyDefaults = isRecord(strategyDetail?.defaultParams)
    ? (strategyDetail!.defaultParams as Record<string, unknown>)
    : {}
  const next: Record<string, unknown> = { ...strategyDefaults, ...raw }
  if (normalizeSourceKey(sourceKey) === 'traders') {
    next.traders_scope = normalizeTradersScopeConfig(next.traders_scope)
    return next
  }
  delete next.traders_scope
  return next
}

export function cloneStrategyParamsRecord(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) return {}
  try {
    return JSON.parse(JSON.stringify(value)) as Record<string, unknown>
  } catch {
    return { ...value }
  }
}

// ── Field grouping ──────────────────────────────────────────────────────

export function classifyStrategyParamGroup(
  fieldKey: string,
  field?: Record<string, unknown>,
): StrategyParamGroupKey {
  const phase = field ? String(field.phase || '').trim().toLowerCase() : ''
  if (phase === 'signal') return 'signal'
  const key = String(fieldKey || '').trim().toLowerCase()
  if (!key) return 'advanced'
  if (
    key.startsWith('strategy_mode')
    || key === 'mode'
    || key === 'traders_scope'
    || key.startsWith('include_')
    || key.startsWith('exclude_')
    || key === 'enabled_sub_strategies'
    || key.includes('sub_strategy')
  ) {
    return 'scope'
  }
  if (
    key.includes('signal_age')
    || key.includes('market_data_age')
    || key.includes('live_context_age')
    || key.includes('oracle_age')
    || key.includes('seconds_left')
    || key.includes('reentry_cooldown')
    || key.includes('freshness')
    || key.includes('timeout')
  ) {
    return 'timing'
  }
  if (
    key.includes('edge')
    || key.includes('confidence')
    || key.includes('liquidity')
    || key.includes('spread')
    || key.includes('imbalance')
    || key.includes('entry_price')
    || key.includes('entry_executable')
    || key.includes('opening_')
    || key.includes('guardrail')
    || key.includes('require_oracle')
  ) {
    return 'entry'
  }
  if (
    key.includes('take_profit')
    || key.includes('stop_loss')
    || key.includes('trailing')
    || key.includes('min_exit')
    || key.includes('exit_price')
    || key.includes('min_hold')
    || key.includes('max_hold')
    || key.startsWith('rapid_')
    || key.startsWith('reverse_')
    || key.startsWith('underwater_')
    || key.startsWith('force_flatten')
    || key.includes('close_on_inactive')
    || key.includes('resolve_only')
    || key.includes('preplace_take_profit')
  ) {
    return 'exit'
  }
  if (
    key.includes('size')
    || key.includes('sizing')
    || key.includes('notional')
    || key.includes('position')
    || key.includes('multiplier')
    || key.includes('kelly')
    || key.includes('capital')
  ) {
    return 'sizing'
  }
  if (key.startsWith('risk') || key.startsWith('max_risk') || key.startsWith('resolution_risk')) {
    return 'risk'
  }
  return 'advanced'
}

export function groupStrategyParamFields(
  fields: Array<Record<string, unknown>>,
): StrategyParamGroup[] {
  const grouped = new Map<StrategyParamGroupKey, Array<Record<string, unknown>>>()
  for (const field of fields) {
    const fieldKey = String(field.key || '').trim()
    if (!fieldKey) continue
    const groupKey = classifyStrategyParamGroup(fieldKey, field)
    const current = grouped.get(groupKey) || []
    current.push(field)
    grouped.set(groupKey, current)
  }
  const orderedGroups: StrategyParamGroup[] = []
  for (const groupKey of STRATEGY_PARAM_GROUP_ORDER) {
    const fieldsForGroup = grouped.get(groupKey)
    if (!fieldsForGroup || fieldsForGroup.length === 0) continue
    orderedGroups.push({
      key: groupKey,
      label: STRATEGY_PARAM_GROUP_LABELS[groupKey],
      fields: fieldsForGroup,
    })
  }
  return orderedGroups
}

// ── Section building (the high-level shape AutoresearchView consumes) ────

export type SourceLabelLookup = (sourceKey: string) => string

/**
 * Build the per-source-strategy sections that drive the autoresearch /
 * tune param editor. Each section has its current values, the param
 * field schema, and field groups split into Signal / Entry / Exit / etc.
 */
export function buildDynamicStrategyParamSections(
  trader: { source_configs?: Array<{ source_key?: string; strategy_key?: string; strategy_params?: unknown }> },
  detailLookup: (sourceKey: string, strategyKey: string) => StrategyOptionDetail | null,
  sourceLabelLookup: SourceLabelLookup,
  strategyLabelLookup: (sourceKey: string, strategyKey: string) => string,
): DynamicStrategyParamSection[] {
  const out: DynamicStrategyParamSection[] = []
  const seenKeys = new Set<string>()
  for (const sourceConfig of trader.source_configs || []) {
    const sourceKey = normalizeSourceKey(String(sourceConfig.source_key || ''))
    const strategyKey = String(sourceConfig.strategy_key || '').trim().toLowerCase()
    if (!sourceKey || !strategyKey) continue
    const sectionKey = `${sourceKey}:${strategyKey}`
    if (seenKeys.has(sectionKey)) continue
    seenKeys.add(sectionKey)
    const detail = detailLookup(sourceKey, strategyKey)
    const merged = buildSourceStrategyParams(
      isRecord(sourceConfig.strategy_params)
        ? (sourceConfig.strategy_params as Record<string, unknown>)
        : {},
      sourceKey,
      detail,
    )
    const fieldDefs = Array.isArray(detail?.paramFields) ? detail!.paramFields : []
    const fieldKeys = fieldDefs
      .map((f) => String((f as Record<string, unknown>).key || '').trim())
      .filter(Boolean)
    const groups = groupStrategyParamFields(fieldDefs as Array<Record<string, unknown>>)
    out.push({
      sectionKey,
      sourceKey,
      sourceLabel: sourceLabelLookup(sourceKey),
      strategyLabel: strategyLabelLookup(sourceKey, strategyKey),
      groups,
      fieldKeys,
      values: merged,
    })
  }
  return out
}
