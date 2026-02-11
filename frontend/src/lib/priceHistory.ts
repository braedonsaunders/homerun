type HistoryObject = Record<string, unknown>

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

function readPoint(raw: unknown): { yes: number | null; no: number | null } {
  if (Array.isArray(raw)) {
    if (raw.length >= 3) {
      return {
        yes: normalizeProbability(toFiniteNumber(raw[1])),
        no: normalizeProbability(toFiniteNumber(raw[2])),
      }
    }
    if (raw.length >= 2) {
      return {
        yes: normalizeProbability(toFiniteNumber(raw[1])),
        no: null,
      }
    }
    return { yes: null, no: null }
  }

  if (!raw || typeof raw !== 'object') {
    return { yes: null, no: null }
  }

  const point = raw as HistoryObject
  return {
    yes: normalizeProbability(
      toFiniteNumber(
        point.yes ?? point.up ?? point.up_price ?? point.y ?? point.p ?? point.price
      )
    ),
    no: normalizeProbability(
      toFiniteNumber(point.no ?? point.down ?? point.down_price ?? point.n)
    ),
  }
}

export function buildYesNoSparklineSeries(
  rawHistory: unknown,
  fallbackYes: unknown,
  fallbackNo: unknown
): { yes: number[]; no: number[] } {
  const history = Array.isArray(rawHistory) ? rawHistory : []

  let yes = history
    .map((p) => readPoint(p).yes)
    .filter((v): v is number => Number.isFinite(v))

  let no = history
    .map((p) => readPoint(p).no)
    .filter((v): v is number => Number.isFinite(v))

  // If one side is missing, reconstruct from complement for binary markets.
  if (yes.length >= 2 && no.length < 2 && isUnitRange(yes)) {
    no = yes.map((v) => 1 - v)
  } else if (no.length >= 2 && yes.length < 2 && isUnitRange(no)) {
    yes = no.map((v) => 1 - v)
  }

  if (yes.length < 2) {
    const yesNow = normalizeProbability(toFiniteNumber(fallbackYes))
    if (yesNow !== null) {
      yes = [yesNow, yesNow]
    }
  }

  if (no.length < 2) {
    const noNow = normalizeProbability(toFiniteNumber(fallbackNo))
    if (noNow !== null) {
      no = [noNow, noNow]
    }
  }

  return { yes, no }
}
