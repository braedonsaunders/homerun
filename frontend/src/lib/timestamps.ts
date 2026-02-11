const ISO_AWARE_DATE_TIME_RE = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)(Z|[+-]\d{2}:\d{2})$/i
const NAIVE_ISO_DATE_TIME_RE = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)$/

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const proto = Object.getPrototypeOf(value)
  return proto === Object.prototype || proto === null
}

export function normalizeUtcTimestamp(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return value
  if (ISO_AWARE_DATE_TIME_RE.test(trimmed)) return trimmed

  const match = NAIVE_ISO_DATE_TIME_RE.exec(trimmed)
  if (!match) return value
  return `${match[1]}T${match[2]}Z`
}

export function normalizeUtcTimestampsInPlace(value: unknown): void {
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      const current = value[i]
      if (typeof current === 'string') {
        value[i] = normalizeUtcTimestamp(current)
      } else if (current && (Array.isArray(current) || isPlainObject(current))) {
        normalizeUtcTimestampsInPlace(current)
      }
    }
    return
  }

  if (!isPlainObject(value)) return

  for (const [key, current] of Object.entries(value)) {
    if (typeof current === 'string') {
      value[key] = normalizeUtcTimestamp(current)
    } else if (current && (Array.isArray(current) || isPlainObject(current))) {
      normalizeUtcTimestampsInPlace(current)
    }
  }
}
