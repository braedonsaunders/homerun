type NullableString = string | null | undefined
export type MarketPlatform = 'polymarket' | 'kalshi'

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

function isLikelyKalshiTicker(value: NullableString): boolean {
  const ticker = cleanSegment(value).toUpperCase()
  return /^KX[A-Z0-9-]+$/.test(ticker)
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

export function inferMarketPlatform(params: {
  platform?: NullableString
  marketId?: NullableString
  marketSlug?: NullableString
  conditionId?: NullableString
}): MarketPlatform {
  const explicit = cleanSegment(params.platform).toLowerCase()
  if (explicit === 'kalshi') return 'kalshi'
  if (explicit === 'polymarket') return 'polymarket'

  const conditionId = cleanSegment(params.conditionId)
  if (isConditionId(conditionId)) return 'polymarket'

  if (isLikelyKalshiTicker(params.marketId) || isLikelyKalshiTicker(params.marketSlug)) {
    return 'kalshi'
  }

  return 'polymarket'
}

export function buildPolymarketMarketUrl(params: {
  eventSlug?: NullableString
  marketSlug?: NullableString
  marketId?: NullableString
  conditionId?: NullableString
}): string | null {
  const eventSlug = cleanSegment(params.eventSlug)
  const marketSlug = cleanSegment(params.marketSlug)
  const marketId = cleanSegment(params.marketId)
  const conditionId = cleanSegment(params.conditionId)

  if (eventSlug && marketSlug && eventSlug !== marketSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(eventSlug)}/${encodeSegment(marketSlug)}`
  }
  if (eventSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(eventSlug)}`
  }
  if (marketSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(marketSlug)}`
  }
  if (conditionId) {
    return `${POLYMARKET_BASE_URL}/market/${encodeSegment(conditionId)}`
  }
  if (marketId) {
    return `${POLYMARKET_BASE_URL}/market/${encodeSegment(marketId)}`
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

  const marketTickerSegment = encodeSegment(marketTicker.toLowerCase())
  const eventTicker = cleanSegment(params.eventTicker) || deriveKalshiEventTicker(marketTicker)
  if (!eventTicker) return `${KALSHI_BASE_URL}/markets/${marketTickerSegment}`

  const eventTickerSegment = encodeSegment(eventTicker.toLowerCase())
  const eventSlug = normalizeKalshiSlug(params.eventSlug)
  const marketTickerLower = marketTicker.toLowerCase()
  const eventTickerLower = eventTicker.toLowerCase()
  if (!eventSlug || eventSlug === marketTickerLower || eventSlug === eventTickerLower) {
    // Fallback to ticker-only path when we don't have a reliable SEO slug.
    return `${KALSHI_BASE_URL}/markets/${marketTickerSegment}`
  }

  const eventSlugSegment = encodeSegment(eventSlug)

  if (eventTickerLower === marketTickerLower) {
    return `${KALSHI_BASE_URL}/markets/${eventTickerSegment}/${eventSlugSegment}`
  }

  return `${KALSHI_BASE_URL}/markets/${eventTickerSegment}/${eventSlugSegment}/${marketTickerSegment}`
}
