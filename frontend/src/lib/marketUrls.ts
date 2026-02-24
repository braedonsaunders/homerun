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

function isLikelyKalshiOutcomeSegment(value: string): boolean {
  return /^[A-Z0-9]{1,5}$/.test(value)
}

function isLikelyPolymarketSlug(value: NullableString): boolean {
  const slug = cleanSegment(value).toLowerCase()
  return /^(?=.*[a-z])[a-z0-9-]+$/.test(slug)
}

function normalizeKalshiTicker(value: NullableString): string {
  return cleanSegment(value).replace(/_(yes|no)$/i, '')
}

export function deriveKalshiEventTicker(marketTicker: NullableString): string {
  const ticker = normalizeKalshiTicker(marketTicker).toUpperCase()
  if (!ticker) return ''

  // Multi-outcome market tickers often append a short outcome code segment
  // to the event ticker (e.g. "...-MOR"). Strip that segment when present.
  const parts = ticker.split('-').filter(Boolean)
  if (parts.length >= 3) {
    const maybeOutcome = parts[parts.length - 1]
    if (isLikelyKalshiOutcomeSegment(maybeOutcome)) {
      const candidate = parts.slice(0, -1).join('-')
      if (isLikelyKalshiTicker(candidate)) return candidate
    }
  }
  return ticker
}

export function deriveKalshiSeriesTicker(value: NullableString): string {
  const ticker = normalizeKalshiTicker(value).toUpperCase()
  if (!ticker) return ''
  const parts = ticker.split('-').filter(Boolean)
  if (!parts.length) return ''
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
  seriesTicker?: NullableString
}): string | null {
  // Kalshi market pages resolve via:
  // /markets/{series_ticker}/{event_ticker}
  // which redirects to the full canonical path with title slug.
  const explicitEvent = cleanSegment(params.eventTicker) || cleanSegment(params.eventSlug)
  const eventTicker = isLikelyKalshiTicker(explicitEvent)
    ? explicitEvent.toUpperCase()
    : deriveKalshiEventTicker(params.marketTicker)

  const explicitSeries = cleanSegment(params.seriesTicker)
  const seriesTicker = isLikelyKalshiTicker(explicitSeries)
    ? explicitSeries.toUpperCase()
    : deriveKalshiSeriesTicker(eventTicker || params.marketTicker)

  if (isLikelyKalshiTicker(eventTicker)) {
    if (isLikelyKalshiTicker(seriesTicker) && seriesTicker !== eventTicker) {
      return `${KALSHI_BASE_URL}/markets/${encodeSegment(seriesTicker.toLowerCase())}/${encodeSegment(eventTicker.toLowerCase())}`
    }
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
  series_ticker?: NullableString
  seriesTicker?: NullableString
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
          seriesTicker: market.series_ticker || market.seriesTicker,
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

type TraderOrderLinkInput = {
  source?: NullableString
  marketId?: NullableString
  marketQuestion?: NullableString
  payload?: Record<string, unknown> | null | undefined
}

function toRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return {}
}

function normalizeSourcePlatform(source: NullableString): MarketPlatform | null {
  const key = cleanSegment(source).toLowerCase()
  if (!key) return null
  if (key.includes('kalshi')) return 'kalshi'
  if (key.includes('poly') || key.includes('crypto')) return 'polymarket'
  return null
}

function buildPolymarketSearchUrl(query: NullableString): string | null {
  const text = cleanSegment(query).replace(/[-_]+/g, ' ').trim()
  if (!text) return null
  return `${POLYMARKET_BASE_URL}/search?query=${encodeURIComponent(text)}`
}

function buildKalshiSearchUrl(query: NullableString): string | null {
  const text = cleanSegment(query).replace(/[-_]+/g, ' ').trim()
  if (!text) return null
  return `${KALSHI_BASE_URL}/markets?search=${encodeURIComponent(text)}`
}

function cleanOptionalText(value: unknown): string | null {
  if (value === null || value === undefined) return null
  const text = String(value).trim()
  return text || null
}

function cleanAbsoluteUrlFromUnknown(value: unknown): string | null {
  const text = cleanOptionalText(value)
  if (!text) return null
  return cleanAbsoluteUrl(text)
}

// Shared resolver for trader/order rows where payloads are often sparse and
// direct API URLs may be stale. Canonical builders are preferred first.
export function getTraderOrderPlatformLinks(input: TraderOrderLinkInput): OpportunityPlatformLinks {
  const payload = toRecord(input.payload)
  const legPayload = toRecord(payload.leg)
  const liveMarket = toRecord(payload.live_market || payload.liveMarket)
  const marketId = cleanSegment(
    String(
      input.marketId
      || payload.market_id
      || payload.marketId
      || legPayload.market_id
      || legPayload.marketId
      || liveMarket.market_id
      || liveMarket.marketId
      || ''
    )
  )
  const marketQuestion = cleanSegment(
    String(
      input.marketQuestion
      || payload.market_question
      || payload.marketQuestion
      || legPayload.market_question
      || legPayload.marketQuestion
      || liveMarket.market_question
      || liveMarket.marketQuestion
      || ''
    )
  )
  const rawMarkets = Array.isArray(payload.markets) ? payload.markets : []
  let matchedMarketPayload: Record<string, unknown> = {}
  if (rawMarkets.length > 0) {
    const normalizedMarketId = marketId.toLowerCase()
    for (const rawMarket of rawMarkets) {
      const market = toRecord(rawMarket)
      if (Object.keys(market).length === 0) continue
      if (!marketId) {
        matchedMarketPayload = market
        break
      }
      const candidates = [
        cleanOptionalText(market.id),
        cleanOptionalText(market.market_id),
        cleanOptionalText(market.marketId),
        cleanOptionalText(market.slug),
        cleanOptionalText(market.market_slug),
        cleanOptionalText(market.marketSlug),
        cleanOptionalText(market.ticker),
        cleanOptionalText(market.condition_id),
        cleanOptionalText(market.conditionId),
      ]
        .filter((value): value is string => Boolean(value))
        .map((value) => value.toLowerCase())
      if (candidates.includes(normalizedMarketId)) {
        matchedMarketPayload = market
        break
      }
    }
  }
  const payloadPlatform = cleanSegment(
    String(
      payload.platform
      || payload.exchange
      || payload.execution_platform
      || payload.market_platform
      || payload.source_platform
      || legPayload.platform
      || liveMarket.platform
      || matchedMarketPayload.platform
      || ''
    )
  ).toLowerCase()
  const sourcePlatform = normalizeSourcePlatform(input.source)
  const inferredPlatform = payloadPlatform === 'kalshi' || payloadPlatform === 'polymarket'
    ? (payloadPlatform as MarketPlatform)
    : inferMarketPlatform({
        platform: sourcePlatform || payloadPlatform || undefined,
        marketId,
        marketSlug: String(
          payload.market_slug
          || payload.marketSlug
          || payload.slug
          || legPayload.market_slug
          || legPayload.marketSlug
          || legPayload.slug
          || liveMarket.market_slug
          || liveMarket.marketSlug
          || liveMarket.slug
          || matchedMarketPayload.market_slug
          || matchedMarketPayload.marketSlug
          || matchedMarketPayload.slug
          || ''
        ),
        conditionId: String(
          payload.condition_id
          || payload.conditionId
          || legPayload.condition_id
          || legPayload.conditionId
          || liveMarket.condition_id
          || liveMarket.conditionId
          || matchedMarketPayload.condition_id
          || matchedMarketPayload.conditionId
          || (isConditionId(marketId) ? marketId : '')
          || ''
        ),
        eventTicker: String(
          payload.event_ticker
          || payload.eventTicker
          || legPayload.event_ticker
          || legPayload.eventTicker
          || liveMarket.event_ticker
          || liveMarket.eventTicker
          || matchedMarketPayload.event_ticker
          || matchedMarketPayload.eventTicker
          || ''
        ),
      })

  const eventSlug = cleanOptionalText(
    payload.event_slug
    || payload.eventSlug
    || legPayload.event_slug
    || legPayload.eventSlug
    || liveMarket.event_slug
    || liveMarket.eventSlug
    || matchedMarketPayload.event_slug
    || matchedMarketPayload.eventSlug
  )
  const marketSlug = cleanOptionalText(
    payload.market_slug
    || payload.marketSlug
    || payload.slug
    || legPayload.market_slug
    || legPayload.marketSlug
    || legPayload.slug
    || liveMarket.market_slug
    || liveMarket.marketSlug
    || liveMarket.slug
    || matchedMarketPayload.market_slug
    || matchedMarketPayload.marketSlug
    || matchedMarketPayload.slug
  )
  const conditionId = cleanOptionalText(
    payload.condition_id
    || payload.conditionId
    || legPayload.condition_id
    || legPayload.conditionId
    || liveMarket.condition_id
    || liveMarket.conditionId
    || matchedMarketPayload.condition_id
    || matchedMarketPayload.conditionId
    || (isConditionId(marketId) ? marketId : '')
  )
  const marketTicker = cleanOptionalText(
    payload.market_ticker
    || payload.marketTicker
    || payload.kalshi_ticker
    || payload.ticker
    || legPayload.market_ticker
    || legPayload.marketTicker
    || legPayload.kalshi_ticker
    || legPayload.ticker
    || liveMarket.market_ticker
    || liveMarket.marketTicker
    || liveMarket.kalshi_ticker
    || liveMarket.ticker
    || matchedMarketPayload.market_ticker
    || matchedMarketPayload.marketTicker
    || matchedMarketPayload.kalshi_ticker
    || matchedMarketPayload.ticker
    || (isLikelyKalshiTicker(marketId) ? marketId : '')
  )
  const eventTicker = cleanOptionalText(
    payload.event_ticker
    || payload.eventTicker
    || legPayload.event_ticker
    || legPayload.eventTicker
    || liveMarket.event_ticker
    || liveMarket.eventTicker
    || matchedMarketPayload.event_ticker
    || matchedMarketPayload.eventTicker
  )
  const seriesTicker = cleanOptionalText(
    payload.series_ticker
    || payload.seriesTicker
    || legPayload.series_ticker
    || legPayload.seriesTicker
    || liveMarket.series_ticker
    || liveMarket.seriesTicker
    || matchedMarketPayload.series_ticker
    || matchedMarketPayload.seriesTicker
  )

  let directPolymarketUrl = (
    cleanAbsoluteUrlFromUnknown(payload.polymarket_url)
    || cleanAbsoluteUrlFromUnknown(payload.polymarketUrl)
    || cleanAbsoluteUrlFromUnknown(legPayload.polymarket_url)
    || cleanAbsoluteUrlFromUnknown(legPayload.polymarketUrl)
    || cleanAbsoluteUrlFromUnknown(liveMarket.polymarket_url)
    || cleanAbsoluteUrlFromUnknown(liveMarket.polymarketUrl)
    || cleanAbsoluteUrlFromUnknown(matchedMarketPayload.polymarket_url)
    || cleanAbsoluteUrlFromUnknown(matchedMarketPayload.polymarketUrl)
  )
  let directKalshiUrl = (
    cleanAbsoluteUrlFromUnknown(payload.kalshi_url)
    || cleanAbsoluteUrlFromUnknown(payload.kalshiUrl)
    || cleanAbsoluteUrlFromUnknown(legPayload.kalshi_url)
    || cleanAbsoluteUrlFromUnknown(legPayload.kalshiUrl)
    || cleanAbsoluteUrlFromUnknown(liveMarket.kalshi_url)
    || cleanAbsoluteUrlFromUnknown(liveMarket.kalshiUrl)
    || cleanAbsoluteUrlFromUnknown(matchedMarketPayload.kalshi_url)
    || cleanAbsoluteUrlFromUnknown(matchedMarketPayload.kalshiUrl)
  )
  const genericUrlCandidates = [
    payload.market_url,
    payload.marketUrl,
    payload.url,
    legPayload.market_url,
    legPayload.marketUrl,
    legPayload.url,
    liveMarket.market_url,
    liveMarket.marketUrl,
    liveMarket.url,
    matchedMarketPayload.market_url,
    matchedMarketPayload.marketUrl,
    matchedMarketPayload.url,
  ]
  for (const candidate of genericUrlCandidates) {
    const url = cleanAbsoluteUrlFromUnknown(candidate)
    if (!url) continue
    const lowered = url.toLowerCase()
    if (!directPolymarketUrl && lowered.includes('polymarket.com')) {
      directPolymarketUrl = url
    }
    if (!directKalshiUrl && lowered.includes('kalshi.com')) {
      directKalshiUrl = url
    }
  }

  let polymarketUrl = buildPolymarketMarketUrl({
    eventSlug,
    marketSlug,
    marketId,
    conditionId,
  })
  let kalshiUrl = buildKalshiMarketUrl({
    marketTicker,
    eventTicker,
    eventSlug,
    seriesTicker,
  })
  if (!polymarketUrl) polymarketUrl = directPolymarketUrl
  if (!kalshiUrl) kalshiUrl = directKalshiUrl

  if (!polymarketUrl && inferredPlatform !== 'kalshi') {
    polymarketUrl = buildPolymarketSearchUrl(marketQuestion || marketSlug || eventSlug || marketId)
  }
  if (!kalshiUrl && inferredPlatform === 'kalshi') {
    kalshiUrl = buildKalshiSearchUrl(marketQuestion || marketTicker || eventTicker || marketId)
  }

  const marketLinks: OpportunityLinkEntry[] = []
  if (polymarketUrl) marketLinks.push({ platform: 'polymarket', url: polymarketUrl })
  if (kalshiUrl) marketLinks.push({ platform: 'kalshi', url: kalshiUrl })
  return { polymarketUrl, kalshiUrl, marketLinks }
}
