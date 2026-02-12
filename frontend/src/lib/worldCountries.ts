import countryReferenceJson from '../data/countryReference.json'

type CountryReferenceRow = {
  name: string
  alpha2: string
  alpha3: string
}

type GeoJSONGeometry =
  | { type: 'Polygon'; coordinates: number[][][] }
  | { type: 'MultiPolygon'; coordinates: number[][][][] }

type GeoJSONFeature = {
  id?: string | number
  properties?: Record<string, unknown>
  geometry?: GeoJSONGeometry | null
}

type GeoJSONFeatureCollection = {
  type: 'FeatureCollection'
  features: GeoJSONFeature[]
}

export type CountryCentroid = {
  iso3: string
  name: string
  latitude: number
  longitude: number
}

export type CountryBounds = {
  iso3: string
  name: string
  latMin: number
  latMax: number
  lonMin: number
  lonMax: number
}

const COUNTRY_ROWS: CountryReferenceRow[] = (countryReferenceJson as CountryReferenceRow[])
  .filter((row) => row && typeof row === 'object')
  .map((row) => ({
    name: String(row.name || '').trim(),
    alpha2: String(row.alpha2 || '').trim().toUpperCase(),
    alpha3: String(row.alpha3 || '').trim().toUpperCase(),
  }))
  .filter((row) => row.name && row.alpha2.length === 2 && row.alpha3.length === 3)

const BY_ALPHA3 = new Map<string, CountryReferenceRow>()
const BY_ALPHA2 = new Map<string, CountryReferenceRow>()
const BY_NAME_KEY = new Map<string, CountryReferenceRow>()

function normalizeNameKey(value: string): string {
  return String(value || '')
    .toUpperCase()
    .replace(/[^A-Z\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

for (const row of COUNTRY_ROWS) {
  BY_ALPHA3.set(row.alpha3, row)
  BY_ALPHA2.set(row.alpha2, row)
  BY_NAME_KEY.set(normalizeNameKey(row.name), row)
}

const EXTRA_ALIASES: Record<string, string> = {
  UK: 'GBR',
  UAE: 'ARE',
  KOSOVO: 'XKX',
  RUSSIA: 'RUS',
  BOLIVIA: 'BOL',
  VENEZUELA: 'VEN',
  VIETNAM: 'VNM',
  'SOUTH KOREA': 'KOR',
  'NORTH KOREA': 'PRK',
  SYRIA: 'SYR',
  TAIWAN: 'TWN',
  IRAN: 'IRN',
  LAOS: 'LAO',
  MOLDOVA: 'MDA',
  BRUNEI: 'BRN',
}

for (const [alias, iso3] of Object.entries(EXTRA_ALIASES)) {
  const normalized = normalizeNameKey(alias)
  const row = BY_ALPHA3.get(iso3)
  if (row && normalized) {
    BY_NAME_KEY.set(normalized, row)
  }
}

export function normalizeCountryCode(value: string | null | undefined): string | null {
  const raw = String(value || '').trim()
  if (!raw) return null

  const upper = raw.toUpperCase()
  if (upper.length === 3 && BY_ALPHA3.has(upper)) {
    return upper
  }
  if (upper.length === 2) {
    const match = BY_ALPHA2.get(upper)
    if (match) return match.alpha3
  }

  const nameMatch = BY_NAME_KEY.get(normalizeNameKey(raw))
  if (nameMatch) return nameMatch.alpha3

  const aliasIso3 = EXTRA_ALIASES[normalizeNameKey(raw)]
  if (aliasIso3) return aliasIso3

  return null
}

export function getCountryName(value: string | null | undefined): string | null {
  const iso3 = normalizeCountryCode(value)
  if (!iso3) return null
  return BY_ALPHA3.get(iso3)?.name || null
}

export function formatCountry(value: string | null | undefined): string {
  const text = String(value || '').trim()
  if (!text) return 'Unknown'
  const pair = parseCountryPair(text)
  if (pair) {
    return `${formatCountry(pair[0])} - ${formatCountry(pair[1])}`
  }
  return getCountryName(text) || text
}

export function parseCountryPair(value: string | null | undefined): [string, string] | null {
  const raw = String(value || '').trim()
  if (!raw) return null

  const separator = ['-', '/', '|'].find((token) => raw.includes(token))
  if (!separator) return null

  const parts = raw
    .split(separator)
    .map((part) => part.trim())
    .filter(Boolean)
  if (parts.length !== 2) return null

  const a = normalizeCountryCode(parts[0])
  const b = normalizeCountryCode(parts[1])
  if (!a || !b || a === b) return null
  return [a, b]
}

function extractCoordinates(geometry: GeoJSONGeometry | null | undefined): Array<[number, number]> {
  if (!geometry) return []
  if (geometry.type === 'Polygon') {
    const out: Array<[number, number]> = []
    for (const ring of geometry.coordinates) {
      for (const position of ring) {
        if (position.length < 2) continue
        const lon = Number(position[0])
        const lat = Number(position[1])
        if (Number.isFinite(lon) && Number.isFinite(lat)) {
          out.push([lon, lat])
        }
      }
    }
    return out
  }
  if (geometry.type === 'MultiPolygon') {
    const out: Array<[number, number]> = []
    for (const polygon of geometry.coordinates) {
      for (const ring of polygon) {
        for (const position of ring) {
          if (position.length < 2) continue
          const lon = Number(position[0])
          const lat = Number(position[1])
          if (Number.isFinite(lon) && Number.isFinite(lat)) {
            out.push([lon, lat])
          }
        }
      }
    }
    return out
  }
  return []
}

function normalizeLongitude(value: number): number {
  if (!Number.isFinite(value)) return 0
  let lon = value
  while (lon <= -180) lon += 360
  while (lon > 180) lon -= 360
  return lon
}

function unwrapLongitudes(points: Array<[number, number]>): number[] {
  if (points.length === 0) return []
  const out: number[] = []
  let previous = normalizeLongitude(points[0][0])
  out.push(previous)
  for (let idx = 1; idx < points.length; idx += 1) {
    let lon = normalizeLongitude(points[idx][0])
    while (lon - previous > 180) lon -= 360
    while (lon - previous < -180) lon += 360
    out.push(lon)
    previous = lon
  }
  return out
}

export function buildCountryCentroids(
  geojson: GeoJSONFeatureCollection | null | undefined
): Record<string, CountryCentroid> {
  if (!geojson || !Array.isArray(geojson.features)) return {}

  const out: Record<string, CountryCentroid> = {}

  for (const feature of geojson.features) {
    const rawId = String(feature.id || feature.properties?.id || '').trim()
    const iso3 = normalizeCountryCode(rawId)
      || normalizeCountryCode(String(feature.properties?.iso3 || ''))
      || normalizeCountryCode(String(feature.properties?.ISO_A3 || ''))
      || normalizeCountryCode(String(feature.properties?.['ISO3166-1-Alpha-3'] || ''))
    if (!iso3) continue

    const points = extractCoordinates(feature.geometry || null)
    if (points.length === 0) continue
    const longitudes = unwrapLongitudes(points)
    if (longitudes.length === 0) continue

    let minLon = longitudes[0]
    let maxLon = longitudes[0]
    let minLat = points[0][1]
    let maxLat = points[0][1]
    for (let idx = 0; idx < points.length; idx += 1) {
      const lon = longitudes[idx]
      const lat = points[idx][1]
      if (lon < minLon) minLon = lon
      if (lon > maxLon) maxLon = lon
      if (lat < minLat) minLat = lat
      if (lat > maxLat) maxLat = lat
    }

    const row = BY_ALPHA3.get(iso3)
    const name = row?.name || String(feature.properties?.name || iso3)
    out[iso3] = {
      iso3,
      name,
      latitude: Number(((minLat + maxLat) / 2).toFixed(6)),
      longitude: Number(normalizeLongitude((minLon + maxLon) / 2).toFixed(6)),
    }
  }

  return out
}

export function buildCountryBounds(
  geojson: GeoJSONFeatureCollection | null | undefined
): Record<string, CountryBounds> {
  if (!geojson || !Array.isArray(geojson.features)) return {}

  const out: Record<string, CountryBounds> = {}

  for (const feature of geojson.features) {
    const rawId = String(feature.id || feature.properties?.id || '').trim()
    const iso3 = normalizeCountryCode(rawId)
      || normalizeCountryCode(String(feature.properties?.iso3 || ''))
      || normalizeCountryCode(String(feature.properties?.ISO_A3 || ''))
      || normalizeCountryCode(String(feature.properties?.['ISO3166-1-Alpha-3'] || ''))
    if (!iso3) continue

    const points = extractCoordinates(feature.geometry || null)
    if (points.length === 0) continue
    const longitudes = unwrapLongitudes(points)
    if (longitudes.length === 0) continue

    let minLon = longitudes[0]
    let maxLon = longitudes[0]
    let minLat = points[0][1]
    let maxLat = points[0][1]
    for (let idx = 0; idx < points.length; idx += 1) {
      const lon = longitudes[idx]
      const lat = points[idx][1]
      if (lon < minLon) minLon = lon
      if (lon > maxLon) maxLon = lon
      if (lat < minLat) minLat = lat
      if (lat > maxLat) maxLat = lat
    }

    const row = BY_ALPHA3.get(iso3)
    const name = row?.name || String(feature.properties?.name || iso3)
    out[iso3] = {
      iso3,
      name,
      latMin: Number(minLat.toFixed(6)),
      latMax: Number(maxLat.toFixed(6)),
      lonMin: Number(minLon.toFixed(6)),
      lonMax: Number(maxLon.toFixed(6)),
    }
  }

  return out
}

export function formatCountryPair(a: string, b: string): string {
  const left = formatCountry(a)
  const right = formatCountry(b)
  return `${left} - ${right}`
}
