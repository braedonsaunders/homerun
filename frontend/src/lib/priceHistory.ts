type HistoryObject = Record<string, unknown>

export interface OutcomeFallback {
  key?: string | null
  label?: string | null
  price?: unknown
}

export interface OutcomeSparklineSeries {
  key: string
  label: string
  data: number[]
  latest: number | null
}

const RESERVED_HISTORY_KEYS = new Set([
  't',
  'ts',
  'time',
  'timestamp',
  'date',
  'created_at',
  'updated_at',
  'prices',
  'values',
  'outcome_prices',
  'outcomePrices',
  'outcome_prices_by_label',
  'outcomePricesByLabel',
])

const HISTORY_KEY_ALIASES: Record<string, string> = {
  y: 'yes',
  yes: 'yes',
  up: 'yes',
  up_price: 'yes',
  p: 'yes',
  price: 'yes',
  n: 'no',
  no: 'no',
  down: 'no',
  down_price: 'no',
}

function toFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'string' && value.trim() === '') return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function normalizeProbability(value: number | null): number | null {
  if (value === null) return null
  // Support legacy 0..100 cent-style payloads.
  if (value > 1 && value <= 100) return value / 100
  return value
}

function isUnitRange(values: number[]): boolean {
  return values.every((v) => v >= 0 && v <= 1)
}

function sanitizeLabel(value: unknown): string {
  return String(value || '').trim()
}

function titleCaseKey(raw: string): string {
  const normalized = raw.replace(/[_-]+/g, ' ').trim()
  if (!normalized) return 'Outcome'
  return normalized
    .split(/\s+/g)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(' ')
}

function parseJsonArray(raw: unknown): unknown[] {
  if (Array.isArray(raw)) return raw
  if (typeof raw !== 'string') return []
  const text = raw.trim()
  if (!text || !text.startsWith('[') || !text.endsWith(']')) return []
  try {
    const parsed = JSON.parse(text)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function extractOutcomeLabels(raw: unknown): string[] {
  const arr = parseJsonArray(raw)
  const source = arr.length > 0
    ? arr
    : Array.isArray(raw)
      ? raw
      : []

  const labels: string[] = []
  for (const item of source) {
    if (typeof item === 'string') {
      const label = sanitizeLabel(item)
      if (label) labels.push(label)
      continue
    }
    if (!item || typeof item !== 'object') continue
    const row = item as HistoryObject
    const label = sanitizeLabel(
      row.outcome
      ?? row.label
      ?? row.name
      ?? row.title
      ?? row.value
    )
    if (label) labels.push(label)
  }
  return labels
}

export function extractOutcomePrices(raw: unknown): number[] {
  const arr = parseJsonArray(raw)
  const source = arr.length > 0
    ? arr
    : Array.isArray(raw)
      ? raw
      : []

  const prices: number[] = []
  for (const item of source) {
    if (item && typeof item === 'object') {
      const row = item as HistoryObject
      const parsed = normalizeProbability(
        toFiniteNumber(row.price ?? row.p ?? row.value)
      )
      if (parsed !== null) prices.push(parsed)
      continue
    }
    const parsed = normalizeProbability(toFiniteNumber(item))
    if (parsed !== null) prices.push(parsed)
  }
  return prices
}

function normalizeHistoryKey(rawKey: string): string {
  const key = rawKey.trim().toLowerCase().replace(/[ -]+/g, '_')
  if (!key) return ''

  const indexed = key.match(/^(outcome|option|idx|index|token)[_-]?(\d+)$/)
  if (indexed) return `idx_${indexed[2]}`
  if (/^\d+$/.test(key)) return `idx_${key}`

  return HISTORY_KEY_ALIASES[key] || key
}

function remapBinaryKeyToIndexed(
  key: string,
  fallbackKeys: Set<string>,
): string {
  if (key === 'yes' && fallbackKeys.has('idx_0') && !fallbackKeys.has('yes')) {
    return 'idx_0'
  }
  if (key === 'no' && fallbackKeys.has('idx_1') && !fallbackKeys.has('no')) {
    return 'idx_1'
  }
  return key
}

function inferLabelForKey(key: string, indexHint = 0): string {
  if (key === 'yes') return 'Yes'
  if (key === 'no') return 'No'
  const indexed = key.match(/^idx_(\d+)$/)
  if (indexed) return `Outcome ${Number(indexed[1]) + 1}`
  return titleCaseKey(key || `outcome_${indexHint + 1}`)
}

function parsePointValues(
  raw: unknown,
  fallbackKeys: Set<string>,
): Map<string, number> {
  const values = new Map<string, number>()

  const pushValue = (rawKey: string, rawValue: unknown) => {
    const normalizedKey = normalizeHistoryKey(rawKey)
    if (!normalizedKey) return
    const parsed = normalizeProbability(toFiniteNumber(rawValue))
    if (parsed === null) return
    values.set(remapBinaryKeyToIndexed(normalizedKey, fallbackKeys), parsed)
  }

  if (Array.isArray(raw)) {
    // [timestamp, p0, p1, p2, ...]
    for (let i = 1; i < raw.length; i += 1) {
      pushValue(`idx_${i - 1}`, raw[i])
    }
    return values
  }

  if (!raw || typeof raw !== 'object') {
    return values
  }

  const point = raw as HistoryObject

  const vectorPrices = (
    extractOutcomePrices(
      point.outcome_prices
      ?? point.outcomePrices
      ?? point.prices
      ?? point.values
    )
  )
  vectorPrices.forEach((value, i) => {
    values.set(`idx_${i}`, value)
  })

  const byLabel = point.outcome_prices_by_label ?? point.outcomePricesByLabel
  if (byLabel && typeof byLabel === 'object') {
    for (const [key, value] of Object.entries(byLabel as HistoryObject)) {
      pushValue(key, value)
    }
  }

  for (const [rawKey, rawValue] of Object.entries(point)) {
    if (RESERVED_HISTORY_KEYS.has(rawKey)) continue
    pushValue(rawKey, rawValue)
  }

  return values
}

export function buildOutcomeFallbacks({
  labels,
  prices,
  yesPrice,
  noPrice,
  yesLabel,
  noLabel,
  preferIndexedKeys = false,
}: {
  labels?: unknown
  prices?: unknown
  yesPrice?: unknown
  noPrice?: unknown
  yesLabel?: unknown
  noLabel?: unknown
  preferIndexedKeys?: boolean
}): OutcomeFallback[] {
  const labelList = extractOutcomeLabels(labels)
  const priceList = extractOutcomePrices(prices)
  const yes = normalizeProbability(toFiniteNumber(yesPrice))
  const no = normalizeProbability(toFiniteNumber(noPrice))
  const yesText = sanitizeLabel(yesLabel) || 'Yes'
  const noText = sanitizeLabel(noLabel) || 'No'

  const useIndexed = (
    preferIndexedKeys
    || labelList.length > 2
    || priceList.length > 2
  )

  if (!useIndexed) {
    return [
      {
        key: 'yes',
        label: labelList[0] || yesText,
        price: yes ?? priceList[0] ?? null,
      },
      {
        key: 'no',
        label: labelList[1] || noText,
        price: no ?? priceList[1] ?? null,
      },
    ]
  }

  const count = Math.max(labelList.length, priceList.length, 2)
  return Array.from({ length: count }, (_, i) => {
    const fallbackLabel = labelList[i]
      || (i === 0 ? yesText : i === 1 ? noText : `Outcome ${i + 1}`)
    const fallbackPrice = (
      priceList[i]
      ?? (i === 0 ? yes : i === 1 ? no : null)
    )
    return {
      key: `idx_${i}`,
      label: fallbackLabel,
      price: fallbackPrice,
    }
  })
}

export function buildOutcomeSparklineSeries(
  rawHistory: unknown,
  fallbacks: OutcomeFallback[] = [],
): OutcomeSparklineSeries[] {
  const fallbackOrder: string[] = []
  const fallbackKeys = new Set<string>()
  const fallbackLabels = new Map<string, string>()
  const fallbackPrices = new Map<string, number | null>()

  fallbacks.forEach((fallback, index) => {
    const key = normalizeHistoryKey(
      sanitizeLabel(fallback.key) || `idx_${index}`
    )
    if (!key) return
    if (!fallbackKeys.has(key)) {
      fallbackKeys.add(key)
      fallbackOrder.push(key)
    }
    const label = sanitizeLabel(fallback.label)
    if (label && !fallbackLabels.has(key)) {
      fallbackLabels.set(key, label)
    }
    const price = normalizeProbability(toFiniteNumber(fallback.price))
    if (!fallbackPrices.has(key)) {
      fallbackPrices.set(key, price)
    }
  })

  const history = Array.isArray(rawHistory) ? rawHistory : []
  const points = history.map((point) => parsePointValues(point, fallbackKeys))

  const discoveredOrder: string[] = []
  for (const point of points) {
    for (const key of point.keys()) {
      if (!discoveredOrder.includes(key)) {
        discoveredOrder.push(key)
      }
    }
  }

  const orderedKeys = [
    ...fallbackOrder,
    ...discoveredOrder.filter((key) => !fallbackKeys.has(key)),
  ]

  const rawSeriesByKey = new Map<string, number[]>()
  for (const key of orderedKeys) {
    const values = points
      .map((point) => point.get(key))
      .filter((value): value is number => Number.isFinite(value))
    rawSeriesByKey.set(key, values)
  }

  const yesRaw = rawSeriesByKey.get('yes') || []
  const noRaw = rawSeriesByKey.get('no') || []
  if (yesRaw.length >= 2 && noRaw.length < 2 && isUnitRange(yesRaw)) {
    rawSeriesByKey.set('no', yesRaw.map((value) => 1 - value))
  } else if (noRaw.length >= 2 && yesRaw.length < 2 && isUnitRange(noRaw)) {
    rawSeriesByKey.set('yes', noRaw.map((value) => 1 - value))
  }

  const keysWithHistory = new Set<string>()
  for (const [key, values] of rawSeriesByKey.entries()) {
    if (values.length >= 2) {
      keysWithHistory.add(key)
    }
  }
  const hasAnyHistory = keysWithHistory.size > 0

  const series: OutcomeSparklineSeries[] = []
  for (let i = 0; i < orderedKeys.length; i += 1) {
    const key = orderedKeys[i]
    let data = rawSeriesByKey.get(key) || []

    if (data.length < 2) {
      const fallbackPrice = fallbackPrices.get(key)
      const isPrimaryBinaryKey = (
        key === 'yes'
        || key === 'no'
        || key === 'idx_0'
        || key === 'idx_1'
      )
      // Avoid hallucinated extra outcomes: once real history exists for this market,
      // only allow synthetic flat fallback lines for the primary binary pair.
      const allowFallback = (
        fallbackPrice !== null
        && fallbackPrice !== undefined
        && (!hasAnyHistory || isPrimaryBinaryKey)
      )
      if (allowFallback) {
        data = [fallbackPrice, fallbackPrice]
      }
    }

    if (data.length < 2) continue

    const label = (
      fallbackLabels.get(key)
      || inferLabelForKey(key, i)
    )
    series.push({
      key,
      label,
      data,
      latest: data[data.length - 1] ?? null,
    })
  }

  return series
}

export function buildYesNoSparklineSeries(
  rawHistory: unknown,
  fallbackYes: unknown,
  fallbackNo: unknown
): { yes: number[]; no: number[] } {
  const series = buildOutcomeSparklineSeries(
    rawHistory,
    buildOutcomeFallbacks({
      yesPrice: fallbackYes,
      noPrice: fallbackNo,
      yesLabel: 'Yes',
      noLabel: 'No',
    }),
  )

  const yes = series.find((row) => row.key === 'yes' || row.key === 'idx_0')?.data || []
  let no = series.find((row) => row.key === 'no' || row.key === 'idx_1')?.data || []

  if (yes.length >= 2 && no.length < 2 && isUnitRange(yes)) {
    no = yes.map((value) => 1 - value)
  }

  return { yes, no }
}
