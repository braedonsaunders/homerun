type NullableString = string | null | undefined

const POLYMARKET_BASE_URL = 'https://polymarket.com'
const KALSHI_BASE_URL = 'https://kalshi.com'

function cleanSegment(value: NullableString): string {
  return (value || '').trim().replace(/^\/+|\/+$/g, '')
}

function encodeSegment(value: string): string {
  return encodeURIComponent(value)
}

function isConditionId(value: string): boolean {
  return /^0x[0-9a-fA-F]+$/.test(value)
}

function normalizeKalshiTicker(value: NullableString): string {
  return cleanSegment(value).replace(/_(yes|no)$/i, '')
}

function normalizeKalshiSlug(value: NullableString): string {
  return cleanSegment(value)
    .toLowerCase()
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-z0-9-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export function deriveKalshiEventTicker(marketTicker: NullableString): string {
  const ticker = normalizeKalshiTicker(marketTicker)
  if (!ticker) return ''

  const parts = ticker.split('-').filter(Boolean)
  if (parts.length <= 1) return ticker
  return parts.slice(0, -1).join('-')
}

export function buildPolymarketMarketUrl(params: {
  eventSlug?: NullableString
  marketSlug?: NullableString
  marketId?: NullableString
}): string | null {
  const eventSlug = cleanSegment(params.eventSlug)
  const marketSlug = cleanSegment(params.marketSlug)
  const marketId = cleanSegment(params.marketId)

  if (eventSlug && marketSlug && eventSlug !== marketSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(eventSlug)}/${encodeSegment(marketSlug)}`
  }
  if (eventSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(eventSlug)}`
  }
  if (marketSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(marketSlug)}`
  }
  if (marketId && !isConditionId(marketId)) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(marketId)}`
  }
  return null
}

export function buildKalshiMarketUrl(params: {
  marketTicker?: NullableString
  eventTicker?: NullableString
  eventSlug?: NullableString
}): string | null {
  const marketTicker = normalizeKalshiTicker(params.marketTicker)
  if (!marketTicker) return null

  const eventTicker = cleanSegment(params.eventTicker) || deriveKalshiEventTicker(marketTicker)
  if (!eventTicker) return null

  const eventTickerSegment = encodeSegment(eventTicker.toLowerCase())
  const marketTickerSegment = encodeSegment(marketTicker.toLowerCase())
  const eventSlug = normalizeKalshiSlug(params.eventSlug)
  const eventSlugSegment = encodeSegment(eventSlug || eventTicker.toLowerCase())

  if (eventTicker.toLowerCase() === marketTicker.toLowerCase()) {
    if (eventSlug) {
      return `${KALSHI_BASE_URL}/markets/${eventTickerSegment}/${eventSlugSegment}`
    }
    return `${KALSHI_BASE_URL}/markets/${eventTickerSegment}`
  }

  return `${KALSHI_BASE_URL}/markets/${eventTickerSegment}/${eventSlugSegment}/${marketTickerSegment}`
}
