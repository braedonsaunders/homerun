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

function isLikelyPolymarketSlug(value: NullableString): boolean {
  const slug = cleanSegment(value).toLowerCase()
  return /^(?=.*[a-z])[a-z0-9-]+$/.test(slug)
}

function normalizeKalshiTicker(value: NullableString): string {
  return cleanSegment(value).replace(/_(yes|no)$/i, '')
}

export function deriveKalshiEventTicker(marketTicker: NullableString): string {
  const ticker = normalizeKalshiTicker(marketTicker)
  if (!ticker) return ''

  // Kalshi event tickers are the first hyphen-separated segment of the
  // market ticker (e.g. "KXBTCD" from "KXBTCD-26FEB1314",
  // "KXATPCHALLENGERMATCH" from "KXATPCHALLENGERMATCH-26FEB14KASGOM-KAS").
  const parts = ticker.split('-').filter(Boolean)
  if (parts.length <= 1) return ticker
  return parts[0]
}

export function inferMarketPlatform(params: {
  platform?: NullableString
  marketId?: NullableString
  marketSlug?: NullableString
  conditionId?: NullableString
  eventTicker?: NullableString
}): MarketPlatform {
  const explicit = cleanSegment(params.platform).toLowerCase()
  if (explicit === 'kalshi') return 'kalshi'
  if (explicit === 'polymarket') return 'polymarket'

  const conditionId = cleanSegment(params.conditionId)
  if (isConditionId(conditionId)) return 'polymarket'

  if (
    isLikelyKalshiTicker(params.marketId)
    || isLikelyKalshiTicker(params.marketSlug)
    || isLikelyKalshiTicker(params.eventTicker)
  ) {
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
  const eventSlug = cleanSegment(params.eventSlug).toLowerCase()
  const marketSlug = cleanSegment(params.marketSlug).toLowerCase()
  const marketId = cleanSegment(params.marketId).toLowerCase()
  const conditionId = cleanSegment(params.conditionId)

  if (eventSlug && marketSlug && eventSlug !== marketSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(eventSlug)}/${encodeSegment(marketSlug)}`
  }
  if (marketSlug) {
    // /market/{market_slug} redirects to canonical event routes.
    return `${POLYMARKET_BASE_URL}/market/${encodeSegment(marketSlug)}`
  }
  if (eventSlug) {
    return `${POLYMARKET_BASE_URL}/event/${encodeSegment(eventSlug)}`
  }
  // condition/token/numeric IDs are not stable website routes on polymarket.com.
  if (!isConditionId(conditionId) && isLikelyPolymarketSlug(marketId) && !isLikelyKalshiTicker(marketId)) {
    return `${POLYMARKET_BASE_URL}/market/${encodeSegment(marketId)}`
  }
  return null
}

export function buildKalshiMarketUrl(params: {
  marketTicker?: NullableString
  eventTicker?: NullableString
  eventSlug?: NullableString
}): string | null {
  // Kalshi website URLs resolve via the event ticker (lowercase), e.g.
  // https://kalshi.com/markets/kxbtcmaxy — full market tickers 404.
  // Prefer explicit event_ticker / event_slug, then derive from market ticker.
  const eventTicker =
    cleanSegment(params.eventTicker) ||
    cleanSegment(params.eventSlug) ||
    deriveKalshiEventTicker(params.marketTicker)
  if (isLikelyKalshiTicker(eventTicker)) {
    return `${KALSHI_BASE_URL}/markets/${encodeSegment(eventTicker.toLowerCase())}`
  }

  return null
}

function cleanAbsoluteUrl(value: NullableString): string | null {
  const text = (value || '').trim()
  if (!text) return null
  if (text.startsWith('http://') || text.startsWith('https://')) return text
  return null
}

type OpportunityMarketForLinks = {
  id?: NullableString
  market_id?: NullableString
  ticker?: NullableString
  slug?: NullableString
  market_slug?: NullableString
  event_slug?: NullableString
  event_ticker?: NullableString
  condition_id?: NullableString
  conditionId?: NullableString
  platform?: NullableString
  url?: NullableString
  market_url?: NullableString
}

type OpportunityForLinks = {
  event_slug?: NullableString
  markets?: OpportunityMarketForLinks[]
  polymarket_url?: NullableString
  kalshi_url?: NullableString
}

export type OpportunityLinkEntry = {
  platform: MarketPlatform
  url: string | null
}

export type OpportunityPlatformLinks = {
  polymarketUrl: string | null
  kalshiUrl: string | null
  marketLinks: OpportunityLinkEntry[]
}

// Single global resolver used by opportunities views. It prefers API-provided links.
export function getOpportunityPlatformLinks(opportunity: OpportunityForLinks | null | undefined): OpportunityPlatformLinks {
  const markets = Array.isArray(opportunity?.markets) ? opportunity!.markets : []
  const eventSlug = opportunity?.event_slug

  let polymarketUrl = cleanAbsoluteUrl(opportunity?.polymarket_url)
  let kalshiUrl = cleanAbsoluteUrl(opportunity?.kalshi_url)

  const marketLinks = markets.map((market): OpportunityLinkEntry => {
    const platform = inferMarketPlatform({
      platform: market.platform,
      marketId: market.id || market.market_id || market.ticker,
      marketSlug: market.slug || market.market_slug,
      conditionId: market.condition_id || market.conditionId,
      eventTicker: market.event_ticker,
    })

    const apiUrl = cleanAbsoluteUrl(market.url || market.market_url)
    const fallbackUrl = platform === 'kalshi'
      ? buildKalshiMarketUrl({
          marketTicker: market.id || market.market_id || market.ticker,
          eventTicker: market.event_ticker,
          eventSlug: market.event_slug,
        })
      : buildPolymarketMarketUrl({
          eventSlug: market.event_slug || eventSlug,
          marketSlug: market.slug || market.market_slug,
          marketId: market.id || market.market_id,
          conditionId: market.condition_id || market.conditionId,
        })

    const url = apiUrl || fallbackUrl || null
    if (platform === 'polymarket' && !polymarketUrl && url) polymarketUrl = url
    if (platform === 'kalshi' && !kalshiUrl && url) kalshiUrl = url

    return { platform, url }
  })

  return { polymarketUrl, kalshiUrl, marketLinks }
}
