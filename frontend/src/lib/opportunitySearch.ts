import { Opportunity } from '../services/api'

export function buildNewsSearchKeywords(opportunity: Opportunity): string {
  let raw = ''
  if (opportunity.markets?.length > 0 && opportunity.markets[0].question) {
    raw = opportunity.markets[0].question
  } else if (opportunity.event_title) {
    raw = opportunity.event_title
  } else {
    raw = opportunity.title || ''
  }

  raw = raw.replace(/^[A-Za-z_ ]+:\s*/i, '')

  if (raw.includes(',')) {
    raw = raw
      .split(',')
      .map((part) => part.trim().replace(/^(yes|no)\s+/i, ''))
      .filter((part) => part.length > 0)
      .slice(0, 3)
      .join(' ')
  }

  raw = raw.replace(/\b(yes|no)\b\s*/gi, '')
  return raw
    .split(/\s+/)
    .filter((word) => word.length > 2)
    .slice(0, 6)
    .join(' ')
}

export function processPolymarketSearchResults(
  results: Opportunity[],
  strategyFilterSet: Set<string> | null,
  selectedCategory: string,
  searchSort: string
): Opportunity[] {
  const filtered = [...results]
    .filter((result) => !strategyFilterSet || strategyFilterSet.has(result.strategy))
    .filter((result) => {
      if (!selectedCategory) return true
      return result.category?.toLowerCase() === selectedCategory.toLowerCase()
    })

  if (searchSort === 'competitive') {
    filtered.sort((a, b) => {
      const aComp = Math.abs((a.markets?.[0]?.yes_price ?? 0.5) - 0.5)
      const bComp = Math.abs((b.markets?.[0]?.yes_price ?? 0.5) - 0.5)
      return aComp - bComp
    })
  } else if (searchSort === 'volume' || searchSort === 'trending') {
    filtered.sort(
      (a, b) => (b.volume ?? b.min_liquidity ?? 0) - (a.volume ?? a.min_liquidity ?? 0)
    )
  } else if (searchSort === 'liquidity') {
    filtered.sort((a, b) => (b.min_liquidity ?? 0) - (a.min_liquidity ?? 0))
  } else if (searchSort === 'newest') {
    filtered.sort((a, b) => (b.resolution_date ?? '').localeCompare(a.resolution_date ?? ''))
  } else if (searchSort === 'ending_soon') {
    filtered.sort((a, b) => (a.resolution_date ?? '9999').localeCompare(b.resolution_date ?? '9999'))
  }

  return filtered
}
