import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAtomValue } from 'jotai'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import {
  getConvergenceZones,
  getInstabilityScores,
  getTensionPairs,
  getWorldRegions,
  getWorldSignals,
  getWorldSourceStatus,
  type ConvergenceZone,
  type TensionPair,
  type WorldRegionChokepoint,
  type WorldRegionHotspot,
  type WorldSignal,
} from '../services/eventsApi'
import { getUnifiedDataSources, type UnifiedDataSource } from '../services/api'
import {
  buildCountryCentroids,
  formatCountry,
  formatCountryPair,
  getCountryName,
  normalizeCountryCode,
  parseCountryPair,
  type CountryCentroid,
} from '../lib/worldCountries'
import { themeAtom } from '../store/atoms'

const DARK_TILE_STYLE = {
  tiles: [
    'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
    'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
    'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
  ],
  background: '#0a0e17',
}

const LIGHT_TILE_STYLE = {
  tiles: [
    'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
    'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
    'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
  ],
  background: '#f8fafc',
}

type SignalPalette = Record<string, string>

type MapPresentationConfig = {
  map_color: string
  map_layer_label: string
  map_layer_short: string
  map_radius_min: number
  map_radius_max: number
  map_opacity: number
  map_blur: number
  map_stroke_width: number
  map_glow: boolean
  map_dedicated_toggle: boolean
  slug: string
}

const MAP_PRESENTATION_DEFAULTS: MapPresentationConfig = {
  map_color: '#64748b',
  map_layer_label: '',
  map_layer_short: '',
  map_radius_min: 4,
  map_radius_max: 9,
  map_opacity: 0.92,
  map_blur: 0.1,
  map_stroke_width: 1.0,
  map_glow: true,
  map_dedicated_toggle: false,
  slug: '',
}

type LngLatTuple = [number, number]
type MapGeoJSONFeature = {
  properties?: Record<string, unknown>
  geometry: {
    type?: string
    coordinates?: unknown
  }
}

type PointFeature = {
  type: 'Feature'
  geometry: {
    type: 'Point'
    coordinates: LngLatTuple
  }
  properties: Record<string, unknown>
}

type PolygonFeature = {
  type: 'Feature'
  geometry: {
    type: 'Polygon'
    coordinates: LngLatTuple[][]
  }
  properties: Record<string, unknown>
}

type LineFeature = {
  type: 'Feature'
  geometry: {
    type: 'LineString'
    coordinates: LngLatTuple[]
  }
  properties: Record<string, unknown>
}

type GeoFeatureCollection = {
  type: 'FeatureCollection'
  features: Array<PointFeature | PolygonFeature | LineFeature>
}

type CountryBoundaryFeature = {
  type: 'Feature'
  id: string
  properties: Record<string, unknown>
  geometry: {
    type: 'Polygon' | 'MultiPolygon'
    coordinates: any
  }
}

type CountryBoundaryFeatureCollection = {
  type: 'FeatureCollection'
  features: CountryBoundaryFeature[]
}

// Dynamic record — structural keys are statically known, but data sources
// with map_dedicated_toggle can add dsrc_<slug> keys at runtime.
type LayerToggles = Record<string, boolean>

type CountryMetric = {
  country_name: string
  instability_score: number
  instability_intensity: number
  tension_score: number
  tension_intensity: number
  combined_intensity: number
  display_intensity: number
  signal_count: number
}

type CountryPopupSummary = {
  totalSignals: number
  criticalSignals: number
  riskSignals: number
  newsSignals: number
  tensionArcCount: number
  convergenceCount: number
  typeCounts: Record<string, number>
  sourceCounts: Record<string, number>
  headlinePreviews: string[]
  arcPreviews: string[]
}

type FlyoutSelection = {
  category: string
  title: string
  subtitle?: string
  body?: string
  iso3?: string
  countryName?: string
  lat?: number
  lon?: number
  source?: string
  signalType?: string
}

type FlyoutTab = 'context' | 'layers' | 'sources'

type FlyoutRelatedEvent = {
  id: string
  kind: 'signal' | 'tension' | 'convergence'
  title: string
  subtitle: string
  score: number
}

const SIGNAL_COLORS_DARK: SignalPalette = {
  conflict: '#f87171',
  tension: '#fb923c',
  instability: '#facc15',
  convergence: '#c084fc',
  anomaly: '#22d3ee',
  military: '#60a5fa',
  infrastructure: '#34d399',
  earthquake: '#f59e0b',
  fire: '#ef4444',
  news: '#a78bfa',
}

const SIGNAL_COLORS_LIGHT: SignalPalette = {
  conflict: '#dc2626',
  tension: '#c2410c',
  instability: '#a16207',
  convergence: '#7c3aed',
  anomaly: '#0e7490',
  military: '#2563eb',
  infrastructure: '#15803d',
  earthquake: '#d97706',
  fire: '#dc2626',
  news: '#7c3aed',
}

const CLICKABLE_LAYERS = [
  'countries-fill-intensity',
  'countries-border-tension',
  'countries-focus-fill',
  'countries-focus-outline',
  'tension-arcs-line',
  'conflicts-dot',
  'signals-dot',
  'signals-glow',
  'signals-military-flight-icon',
  'signals-military-vessel-icon',
  'convergences-ring',
  'convergences-fill',
  'hotspots-fill',
  'hotspots-outline',
  'chokepoints-icon',
] as const

const DEFAULT_LAYER_TOGGLES: LayerToggles = {
  countryIntensity: true,
  tensionBorders: true,
  tensionArcs: true,
  countryBoundaries: true,
  conflictZones: true,
  signals: true,
  convergences: true,
  hotspots: true,
  chokepoints: true,
  // Dynamic dsrc_* entries are added at runtime from data source configs
}

// Structural layer groups map toggle keys to actual MapLibre layer IDs.
// Dynamic toggles (dsrc_*) work through source visibility filtering instead.
const STRUCTURAL_LAYER_GROUPS: Record<string, readonly string[]> = {
  countryIntensity: ['countries-fill-intensity'],
  tensionBorders: ['countries-border-tension'],
  tensionArcs: ['tension-arcs-glow', 'tension-arcs-line'],
  countryBoundaries: ['countries-focus-fill', 'countries-focus-outline'],
  conflictZones: ['conflicts-heat', 'conflicts-dot'],
  signals: [
    'signals-dot',
    'signals-glow',
    'signals-military-flight-icon',
    'signals-military-vessel-icon',
  ],
  convergences: ['convergences-fill', 'convergences-ring'],
  hotspots: ['hotspots-fill', 'hotspots-outline'],
  chokepoints: ['chokepoints-icon'],
}

const COUNTRY_BOUNDARY_URL = `${import.meta.env.BASE_URL}data/world_countries.geojson`
const mapSignalPageSizeEnv = Number(import.meta.env.VITE_WORLD_MAP_SIGNAL_PAGE_SIZE)
const mapSignalMaxEnv = Number(import.meta.env.VITE_WORLD_MAP_SIGNAL_MAX)
const MAP_SIGNAL_PAGE_SIZE = Number.isFinite(mapSignalPageSizeEnv)
  ? Math.max(250, Math.floor(mapSignalPageSizeEnv))
  : 2000
const MAP_SIGNAL_MAX = Number.isFinite(mapSignalMaxEnv)
  ? Math.max(2000, Math.floor(mapSignalMaxEnv))
  : 30000
const EMPTY_COUNTRY_BOUNDARY_COLLECTION: CountryBoundaryFeatureCollection = {
  type: 'FeatureCollection',
  features: [],
}

function emptyFeatureCollection(): GeoFeatureCollection {
  return { type: 'FeatureCollection', features: [] }
}

function useStickyValue<T>(value: T | null | undefined, initialValue: T): [T, boolean] {
  const ref = useRef<T>(initialValue)
  const hasLiveValueRef = useRef(false)
  if (value !== undefined && value !== null) {
    ref.current = value
    hasLiveValueRef.current = true
  }
  return [value ?? ref.current, hasLiveValueRef.current]
}

function buildStyle(theme: 'dark' | 'light'): maplibregl.StyleSpecification {
  const tileStyle = theme === 'light' ? LIGHT_TILE_STYLE : DARK_TILE_STYLE
  return {
    version: 8,
    name: `world-${theme}`,
    sources: {
      'osm-tiles': {
        type: 'raster',
        tiles: tileStyle.tiles,
        tileSize: 256,
        attribution: '&copy; CARTO &copy; OpenStreetMap contributors',
        maxzoom: 18,
      },
    },
    layers: [
      {
        id: 'background',
        type: 'background',
        paint: { 'background-color': tileStyle.background },
      },
      {
        id: 'osm-tiles',
        type: 'raster',
        source: 'osm-tiles',
        paint: { 'raster-opacity': 0.9 },
      },
    ],
  } as maplibregl.StyleSpecification
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

function isFiniteCoordinatePair(lat: unknown, lon: unknown): lat is number {
  return Number.isFinite(Number(lat)) && Number.isFinite(Number(lon))
}

function haversineDistanceKm(latA: number, lonA: number, latB: number, lonB: number): number {
  const earthRadiusKm = 6371
  const dLat = toRadians(latB - latA)
  const dLon = toRadians(lonB - lonA)
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2)
    + Math.cos(toRadians(latA)) * Math.cos(toRadians(latB)) * Math.sin(dLon / 2) * Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return earthRadiusKm * c
}

function toRadians(value: number): number {
  return (value * Math.PI) / 180
}

function toDegrees(value: number): number {
  return (value * 180) / Math.PI
}

function normalizeLongitude(value: number): number {
  if (!Number.isFinite(value)) return 0
  let lon = value
  while (lon <= -180) lon += 360
  while (lon > 180) lon -= 360
  return lon
}

function shortestLongitudeDelta(fromLon: number, toLon: number): number {
  const delta = normalizeLongitude(toLon - fromLon)
  if (delta === -180) return 180
  return delta
}

function geodesicMidpoint(a: CountryCentroid, b: CountryCentroid): LngLatTuple {
  const lat1 = toRadians(a.latitude)
  const lon1 = toRadians(a.longitude)
  const lat2 = toRadians(b.latitude)
  const lon2 = lon1 + toRadians(shortestLongitudeDelta(a.longitude, b.longitude))
  const dLon = lon2 - lon1

  const bx = Math.cos(lat2) * Math.cos(dLon)
  const by = Math.cos(lat2) * Math.sin(dLon)
  const lat3 = Math.atan2(
    Math.sin(lat1) + Math.sin(lat2),
    Math.sqrt((Math.cos(lat1) + bx) ** 2 + by ** 2)
  )
  const lon3 = lon1 + Math.atan2(by, Math.cos(lat1) + bx)

  return [
    Number(normalizeLongitude(toDegrees(lon3)).toFixed(6)),
    Number(toDegrees(lat3).toFixed(6)),
  ]
}

function interpolateGreatCircleArc(
  a: CountryCentroid,
  b: CountryCentroid,
  steps = 72
): LngLatTuple[] {
  const lat1 = toRadians(a.latitude)
  const lon1 = toRadians(a.longitude)
  const lat2 = toRadians(b.latitude)
  const lon2 = lon1 + toRadians(shortestLongitudeDelta(a.longitude, b.longitude))

  const ax = Math.cos(lat1) * Math.cos(lon1)
  const ay = Math.cos(lat1) * Math.sin(lon1)
  const az = Math.sin(lat1)

  const bx = Math.cos(lat2) * Math.cos(lon2)
  const by = Math.cos(lat2) * Math.sin(lon2)
  const bz = Math.sin(lat2)

  const omega = Math.acos(Math.max(-1, Math.min(1, ax * bx + ay * by + az * bz)))
  if (!Number.isFinite(omega) || omega < 1e-9) {
    return [
      [Number(a.longitude.toFixed(6)), Number(a.latitude.toFixed(6))],
      [Number(b.longitude.toFixed(6)), Number(b.latitude.toFixed(6))],
    ]
  }

  const sinOmega = Math.sin(omega)
  let previousLon = a.longitude
  const out: LngLatTuple[] = []
  const sampleCount = Math.max(8, steps)

  for (let idx = 0; idx <= sampleCount; idx += 1) {
    const t = idx / sampleCount
    const weightA = Math.sin((1 - t) * omega) / sinOmega
    const weightB = Math.sin(t * omega) / sinOmega

    const x = weightA * ax + weightB * bx
    const y = weightA * ay + weightB * by
    const z = weightA * az + weightB * bz
    const norm = Math.sqrt(x * x + y * y + z * z) || 1

    const lat = toDegrees(Math.asin(z / norm))
    let lon = toDegrees(Math.atan2(y / norm, x / norm))
    if (idx > 0) {
      while (lon - previousLon > 180) lon -= 360
      while (lon - previousLon < -180) lon += 360
    }
    previousLon = lon
    out.push([
      Number(lon.toFixed(6)),
      Number(lat.toFixed(6)),
    ])
  }

  return out
}

function pairCentroids(
  isoA: string | null | undefined,
  isoB: string | null | undefined,
  centroids: Record<string, CountryCentroid>
): [CountryCentroid, CountryCentroid] | null {
  const left = normalizeCountryCode(isoA)
  const right = normalizeCountryCode(isoB)
  if (!left || !right || left === right) return null
  const a = centroids[left]
  const b = centroids[right]
  if (!a || !b) return null
  return [a, b]
}

function midpoint(a: CountryCentroid, b: CountryCentroid): LngLatTuple {
  return geodesicMidpoint(a, b)
}

function pairFromTension(pair: TensionPair): [string, string] | null {
  const normalizedA = normalizeCountryCode(pair.country_a_iso3 || pair.country_a_name || pair.country_a)
  const normalizedB = normalizeCountryCode(pair.country_b_iso3 || pair.country_b_name || pair.country_b)
  if (!normalizedA || !normalizedB || normalizedA === normalizedB) return null
  return [normalizedA, normalizedB]
}

function toCountryBoundaryGeoJSON(value: unknown): CountryBoundaryFeatureCollection {
  const raw = (value || {}) as Record<string, unknown>
  const featuresRaw = Array.isArray(raw.features) ? raw.features : []
  const features: CountryBoundaryFeature[] = []
  for (const feature of featuresRaw) {
    const row = feature as Record<string, unknown>
    const properties = (row.properties as Record<string, unknown>) || {}
    const id = row.id as string | number | undefined
    const iso3 = normalizeCountryCode(
      String(
        id
        || properties.id
        || properties.iso3
        || properties.ISO_A3
        || properties.iso_a3
        || properties.ADM0_A3
        || properties['ISO3166-1-Alpha-3']
        || ''
      )
    )
    const geometry = row.geometry as { type?: string; coordinates?: unknown } | undefined
    if (!iso3 || !geometry || !geometry.type || !geometry.coordinates) continue
    if (geometry.type !== 'Polygon' && geometry.type !== 'MultiPolygon') continue
    features.push({
      type: 'Feature',
      id: iso3,
      properties: { ...properties, id: iso3 },
      geometry: {
        type: geometry.type,
        coordinates: geometry.coordinates as any,
      },
    })
  }

  return {
    type: 'FeatureCollection',
    features,
  }
}

function withCountryMetrics(
  geojson: CountryBoundaryFeatureCollection | null | undefined,
  metricsByIso3: Record<string, CountryMetric>
): CountryBoundaryFeatureCollection {
  if (!geojson || !Array.isArray(geojson.features)) {
    return { type: 'FeatureCollection', features: [] }
  }

  return {
    type: 'FeatureCollection',
    features: geojson.features.map((feature) => {
      const iso3 = normalizeCountryCode(String(feature.id || feature.properties?.id || '')) || null
      const metrics = iso3 ? metricsByIso3[iso3] : undefined
      return {
        ...feature,
        properties: {
          ...feature.properties,
          id: iso3 || String(feature.properties?.id || feature.id || ''),
          country_name: metrics?.country_name || (iso3 ? formatCountry(iso3) : ''),
          instability_score: Number(metrics?.instability_score || 0),
          instability_intensity: Number(metrics?.instability_intensity || 0),
          tension_score: Number(metrics?.tension_score || 0),
          tension_intensity: Number(metrics?.tension_intensity || 0),
          combined_intensity: Number(metrics?.combined_intensity || 0),
          display_intensity: Number(metrics?.display_intensity || metrics?.combined_intensity || 0),
          signal_count: Number(metrics?.signal_count || 0),
        },
      }
    }),
  }
}

function pairFromSignal(signal: WorldSignal): [string, string] | null {
  const meta = (signal.metadata || {}) as Record<string, unknown>
  const metaA = normalizeCountryCode(String(meta.country_a || ''))
  const metaB = normalizeCountryCode(String(meta.country_b || ''))
  if (metaA && metaB && metaA !== metaB) {
    return [metaA, metaB]
  }
  return parseCountryPair(signal.country)
}

function signalsToGeoJSON(
  signals: WorldSignal[],
  palette: SignalPalette,
  centroids: Record<string, CountryCentroid>,
  presentationBySignalType: Record<string, MapPresentationConfig> = {},
  presentationBySourceName: Record<string, MapPresentationConfig> = {},
): GeoFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: signals
      .map((signal) => {
        const metadata = (signal.metadata || {}) as Record<string, unknown>
        const activityType = String(metadata.activity_type || '').trim().toLowerCase()
        let coords: LngLatTuple | null = null
        let geocodeMode = 'native'
        let countryText = signal.country || ''

        if (signal.latitude != null && signal.longitude != null) {
          coords = [Number(signal.longitude), Number(signal.latitude)]
        } else {
          const pair = pairFromSignal(signal)
          if (pair) {
            const a = centroids[pair[0]]
            const b = centroids[pair[1]]
            if (a && b) {
              coords = midpoint(a, b)
              geocodeMode = 'pair_geodesic_midpoint'
              countryText = formatCountryPair(pair[0], pair[1])
            }
          }

          if (!coords) {
            const iso3 = normalizeCountryCode(signal.country)
            if (iso3) {
              const centroid = centroids[iso3]
              if (centroid) {
                coords = [centroid.longitude, centroid.latitude]
                geocodeMode = 'country_centroid'
                countryText = centroid.name
              }
            }
          }
        }

        if (!coords || !Number.isFinite(coords[0]) || !Number.isFinite(coords[1])) {
          return null
        }

        const detectedAt = signal.detected_at ? new Date(signal.detected_at).getTime() : null
        const ageHours = detectedAt ? (Date.now() - detectedAt) / 3_600_000 : 24
        const isFresh = ageHours < 6

        // Serialize metadata as JSON string so MapLibre can store it in feature properties
        const metadataJson = JSON.stringify(signal.metadata || {})

        // Resolve map presentation config from data source
        const pres =
          presentationBySignalType[signal.signal_type?.toLowerCase() || ''] ||
          presentationBySourceName[signal.source?.toLowerCase() || ''] ||
          MAP_PRESENTATION_DEFAULTS

        return {
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: coords,
          },
          properties: {
            signal_id: signal.signal_id,
            signal_type: signal.signal_type,
            severity: signal.severity,
            title: signal.title,
            country: signal.country || '',
            country_name: countryText || formatCountry(signal.country),
            source: signal.source,
            color: palette[signal.signal_type] || pres.map_color || '#64748b',
            geocode_mode: geocodeMode,
            activity_type: activityType,
            age_hours: Number(ageHours.toFixed(1)),
            is_fresh: isFresh,
            detected_at: signal.detected_at || null,
            metadata_json: metadataJson,
            // Map presentation from data source config (data-driven styling)
            map_radius_min: pres.map_radius_min,
            map_radius_max: pres.map_radius_max,
            map_opacity: pres.map_opacity,
            map_blur: pres.map_blur,
            map_stroke_width: pres.map_stroke_width,
            map_glow: pres.map_glow ? 1 : 0,
            related_market_count: Array.isArray(signal.related_market_ids) ? signal.related_market_ids.length : 0,
            related_market_ids: Array.isArray(signal.related_market_ids)
              ? signal.related_market_ids.slice(0, 5).join(', ')
              : '',
            market_relevance_score: Number(signal.market_relevance_score || 0),
          },
        } as PointFeature
      })
      .filter((feature): feature is PointFeature => feature !== null),
  }
}

function convergencesToGeoJSON(zones: ConvergenceZone[]): GeoFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: zones.map((zone) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [zone.longitude, zone.latitude],
      },
      properties: {
        grid_key: zone.grid_key,
        urgency_score: zone.urgency_score,
        signal_count: zone.signal_count,
        signal_types: zone.signal_types.join(', '),
        country: zone.country || '',
        country_name: zone.country ? formatCountry(zone.country) : '',
      },
    })),
  }
}

function hotspotsToGeoJSON(hotspots: WorldRegionHotspot[]): GeoFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: hotspots.map((hotspot) => ({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [hotspot.lon_min, hotspot.lat_min],
            [hotspot.lon_max, hotspot.lat_min],
            [hotspot.lon_max, hotspot.lat_max],
            [hotspot.lon_min, hotspot.lat_max],
            [hotspot.lon_min, hotspot.lat_min],
          ],
        ],
      },
      properties: {
        id: hotspot.id,
        name: hotspot.name,
        lat_min: hotspot.lat_min,
        lat_max: hotspot.lat_max,
        lon_min: hotspot.lon_min,
        lon_max: hotspot.lon_max,
        event_count: hotspot.event_count ?? 0,
        last_detected_at: hotspot.last_detected_at || '',
        activity_types: (hotspot.activity_types || []).join(', '),
      },
    })),
  }
}

function chokepointsToGeoJSON(chokepoints: WorldRegionChokepoint[]): GeoFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: chokepoints.map((chokepoint) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [chokepoint.longitude, chokepoint.latitude],
      },
      properties: {
        id: chokepoint.id,
        name: chokepoint.name,
        risk_score: Number(chokepoint.risk_score || 0),
        nearby_signal_count: Number(chokepoint.nearby_signal_count || 0),
        daily_transit_total: Number(chokepoint.daily_transit_total || 0),
        daily_capacity_estimate: Number(chokepoint.daily_capacity_estimate || 0),
        baseline_vessel_count_total: Number(chokepoint.baseline_vessel_count_total || 0),
        signal_breakdown: JSON.stringify(chokepoint.signal_breakdown || {}),
        source: String(chokepoint.source || ''),
        chokepoint_source: String(chokepoint.chokepoint_source || ''),
        risk_method: String(chokepoint.risk_method || ''),
        daily_metrics_date: chokepoint.daily_metrics_date || '',
        daily_dataset_updated_at: chokepoint.daily_dataset_updated_at || '',
        last_updated: chokepoint.last_updated || '',
      },
    })),
  }
}

function tensionsToGeoJSON(
  tensions: TensionPair[],
  centroids: Record<string, CountryCentroid>
): GeoFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: tensions
      .map((pair) => {
        const normalizedPair = pairFromTension(pair)
        if (!normalizedPair) return null
        const centroidsForPair = pairCentroids(normalizedPair[0], normalizedPair[1], centroids)
        if (!centroidsForPair) return null
        const [left, right] = centroidsForPair
        const score = Number(pair.tension_score || 0)
        return {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: interpolateGreatCircleArc(left, right, 84),
          },
          properties: {
            country_a: left.iso3,
            country_b: right.iso3,
            country_a_name: left.name,
            country_b_name: right.name,
            pair_name: formatCountryPair(left.iso3, right.iso3),
            tension_score: Number(score.toFixed(2)),
            tension_intensity: Number(clamp01(score / 100).toFixed(4)),
            trend: String(pair.trend || 'stable'),
            event_count: Number(pair.event_count || 0),
            top_event_types: (pair.top_event_types || []).join(', '),
            last_updated: pair.last_updated || '',
          },
        } as LineFeature
      })
      .filter((feature): feature is LineFeature => feature !== null),
  }
}

function conflictSignalsToGeoJSON(
  signals: WorldSignal[],
  centroids: Record<string, CountryCentroid>
): GeoFeatureCollection {
  const features: PointFeature[] = []

  for (const signal of signals) {
    if (signal.signal_type !== 'conflict') continue
    let coordinates: LngLatTuple | null = null
    if (signal.latitude != null && signal.longitude != null) {
      coordinates = [Number(signal.longitude), Number(signal.latitude)]
    } else {
      const iso3 = normalizeCountryCode(signal.country)
      if (iso3 && centroids[iso3]) {
        coordinates = [centroids[iso3].longitude, centroids[iso3].latitude]
      }
    }
    if (!coordinates || !Number.isFinite(coordinates[0]) || !Number.isFinite(coordinates[1])) continue

    features.push({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates,
      },
      properties: {
        signal_id: signal.signal_id,
        title: signal.title,
        source: signal.source,
        severity: Number(signal.severity || 0),
        signal_type: signal.signal_type,
        country_name: formatCountry(signal.country),
      },
    })
  }

  return {
    type: 'FeatureCollection',
    features,
  }
}

function createMilitaryIconImage(type: 'plane' | 'vessel', theme: 'dark' | 'light'): ImageData {
  const size = 48
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return new ImageData(size, size)
  }

  const fill = theme === 'light' ? '#0f172a' : '#e2e8f0'
  const stroke = theme === 'light' ? '#f8fafc' : '#020617'

  ctx.clearRect(0, 0, size, size)
  ctx.fillStyle = fill
  ctx.strokeStyle = stroke
  ctx.lineWidth = 2.2
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'

  if (type === 'plane') {
    ctx.beginPath()
    ctx.moveTo(24, 5)
    ctx.lineTo(29, 18)
    ctx.lineTo(42, 21)
    ctx.lineTo(42, 27)
    ctx.lineTo(29, 30)
    ctx.lineTo(24, 43)
    ctx.lineTo(19, 30)
    ctx.lineTo(6, 27)
    ctx.lineTo(6, 21)
    ctx.lineTo(19, 18)
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  } else {
    ctx.beginPath()
    ctx.moveTo(7, 29)
    ctx.lineTo(41, 29)
    ctx.lineTo(37, 36)
    ctx.lineTo(11, 36)
    ctx.closePath()
    ctx.fill()
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(17, 29)
    ctx.lineTo(17, 18)
    ctx.lineTo(28, 18)
    ctx.lineTo(32, 29)
    ctx.closePath()
    ctx.fill()
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(22, 18)
    ctx.lineTo(22, 12)
    ctx.lineTo(25, 12)
    ctx.lineTo(25, 18)
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  }

  return ctx.getImageData(0, 0, size, size)
}

function ensureMilitaryIcons(map: any, theme: 'dark' | 'light') {
  const planeIconId = 'wi-military-plane'
  const vesselIconId = 'wi-military-vessel'
  if (map.hasImage(planeIconId)) {
    map.removeImage(planeIconId)
  }
  if (map.hasImage(vesselIconId)) {
    map.removeImage(vesselIconId)
  }
  map.addImage(planeIconId, createMilitaryIconImage('plane', theme), { pixelRatio: 2 })
  map.addImage(vesselIconId, createMilitaryIconImage('vessel', theme), { pixelRatio: 2 })
}

function addDataLayers(map: any, theme: 'dark' | 'light') {
  const hotspotFillColor = theme === 'light' ? '#2563eb' : '#60a5fa'
  const countryBorderColor = theme === 'light' ? '#94a3b8' : '#475569'
  ensureMilitaryIcons(map, theme)

  map.addSource('countries', {
    type: 'geojson',
    data: emptyFeatureCollection(),
  })
  map.addLayer({
    id: 'countries-fill-intensity',
    type: 'fill',
    source: 'countries',
    paint: {
      'fill-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'display_intensity'], 0],
        0, theme === 'light' ? '#f8fafc' : '#0f172a',
        0.2, theme === 'light' ? '#fde68a' : '#854d0e',
        0.45, theme === 'light' ? '#fb923c' : '#c2410c',
        0.7, theme === 'light' ? '#ef4444' : '#dc2626',
        1, theme === 'light' ? '#b91c1c' : '#7f1d1d',
      ],
      'fill-opacity': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'display_intensity'], 0],
        0, 0.045,
        0.2, 0.135,
        0.5, 0.215,
        1, 0.325,
      ],
    },
  })
  map.addLayer({
    id: 'countries-border-base',
    type: 'line',
    source: 'countries',
    paint: {
      'line-color': countryBorderColor,
      'line-width': 1.0,
      'line-opacity': theme === 'light' ? 0.5 : 0.68,
    },
  })
  map.addLayer({
    id: 'countries-border-tension',
    type: 'line',
    source: 'countries',
    paint: {
      'line-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_intensity'], 0],
        0, theme === 'light' ? '#f1f5f9' : '#1e293b',
        0.25, '#facc15',
        0.5, '#fb923c',
        0.75, '#ef4444',
        1, '#991b1b',
      ],
      'line-width': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_intensity'], 0],
        0, 0.45,
        0.25, 1,
        0.5, 1.8,
        1, 3,
      ],
      'line-opacity': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_intensity'], 0],
        0, 0.18,
        0.1, 0.45,
        0.6, 0.8,
        1, 1,
      ],
    },
  })
  map.addLayer({
    id: 'countries-focus-fill',
    type: 'fill',
    source: 'countries',
    paint: {
      'fill-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'combined_intensity'], 0],
        0.2, theme === 'light' ? '#fef3c7' : '#78350f',
        0.45, theme === 'light' ? '#fb923c' : '#c2410c',
        0.7, theme === 'light' ? '#ef4444' : '#dc2626',
        1, theme === 'light' ? '#991b1b' : '#7f1d1d',
      ],
      'fill-opacity': [
        'case',
        [
          'any',
          ['>=', ['coalesce', ['get', 'combined_intensity'], 0], 0.2],
          ['>=', ['coalesce', ['get', 'signal_count'], 0], 1],
        ],
        [
          'interpolate',
          ['linear'],
          ['coalesce', ['get', 'combined_intensity'], 0],
          0.2, 0.06,
          0.6, 0.16,
          1, 0.24,
        ],
        0,
      ],
    },
  })
  map.addLayer({
    id: 'countries-focus-outline',
    type: 'line',
    source: 'countries',
    paint: {
      'line-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'combined_intensity'], 0],
        0.2, theme === 'light' ? '#f59e0b' : '#f59e0b',
        0.6, '#ef4444',
        1, '#7f1d1d',
      ],
      'line-width': [
        'case',
        [
          'any',
          ['>=', ['coalesce', ['get', 'combined_intensity'], 0], 0.2],
          ['>=', ['coalesce', ['get', 'signal_count'], 0], 1],
        ],
        [
          'interpolate',
          ['linear'],
          ['coalesce', ['get', 'combined_intensity'], 0],
          0.2, 1.1,
          0.6, 2.2,
          1, 3.2,
        ],
        0,
      ],
      'line-opacity': [
        'case',
        [
          'any',
          ['>=', ['coalesce', ['get', 'combined_intensity'], 0], 0.2],
          ['>=', ['coalesce', ['get', 'signal_count'], 0], 1],
        ],
        0.9,
        0,
      ],
      'line-dasharray': [2, 1],
    },
  })

  map.addSource('tension-arcs', {
    type: 'geojson',
    data: emptyFeatureCollection(),
    lineMetrics: true,
  })
  map.addLayer({
    id: 'tension-arcs-glow',
    type: 'line',
    source: 'tension-arcs',
    paint: {
      'line-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_score'], 0],
        0, theme === 'light' ? '#fde68a' : '#854d0e',
        40, '#f59e0b',
        70, '#ef4444',
        100, '#991b1b',
      ],
      'line-width': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_score'], 0],
        0, 1.5,
        40, 3,
        70, 5.5,
        100, 8,
      ],
      'line-opacity': 0.22,
      'line-blur': 1.2,
    },
  })
  map.addLayer({
    id: 'tension-arcs-line',
    type: 'line',
    source: 'tension-arcs',
    paint: {
      'line-color': [
        'match', ['coalesce', ['get', 'trend'], 'stable'],
        'rising', theme === 'light' ? '#ef4444' : '#f87171',
        'falling', theme === 'light' ? '#2563eb' : '#60a5fa',
        /* stable default */ theme === 'light' ? '#ca8a04' : '#facc15',
      ],
      'line-width': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_score'], 0],
        0, 0.8,
        40, 1.6,
        70, 2.4,
        100, 3.2,
      ],
      'line-opacity': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_score'], 0],
        0, 0.35,
        40, 0.55,
        70, 0.78,
        100, 0.95,
      ],
    },
  })

  map.addSource('conflicts', {
    type: 'geojson',
    data: emptyFeatureCollection(),
  })
  map.addLayer({
    id: 'conflicts-heat',
    type: 'heatmap',
    source: 'conflicts',
    maxzoom: 8,
    paint: {
      'heatmap-weight': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'severity'], 0],
        0, 0.1,
        0.5, 0.65,
        1, 1,
      ],
      'heatmap-intensity': [
        'interpolate',
        ['linear'],
        ['zoom'],
        1.5, 0.4,
        5, 0.9,
        8, 1.3,
      ],
      'heatmap-color': [
        'interpolate',
        ['linear'],
        ['heatmap-density'],
        0, 'rgba(15,23,42,0)',
        0.2, theme === 'light' ? 'rgba(254,240,138,0.35)' : 'rgba(234,179,8,0.28)',
        0.5, theme === 'light' ? 'rgba(251,146,60,0.62)' : 'rgba(249,115,22,0.58)',
        0.8, 'rgba(239,68,68,0.78)',
        1, 'rgba(127,29,29,0.9)',
      ],
      'heatmap-radius': [
        'interpolate',
        ['linear'],
        ['zoom'],
        1.5, 18,
        5, 24,
        8, 30,
      ],
      'heatmap-opacity': [
        'interpolate',
        ['linear'],
        ['zoom'],
        1.5, 0.55,
        8, 0.75,
      ],
    },
  })
  map.addLayer({
    id: 'conflicts-dot',
    type: 'circle',
    source: 'conflicts',
    minzoom: 4,
    paint: {
      'circle-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'severity'], 0],
        0, theme === 'light' ? '#fde68a' : '#f59e0b',
        0.4, '#f97316',
        0.7, '#ef4444',
        1, '#7f1d1d',
      ],
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'severity'], 0],
        0, 3,
        0.4, 4.8,
        0.7, 6.2,
        1, 7.8,
      ],
      'circle-opacity': 0.86,
      'circle-stroke-color': theme === 'light' ? '#ffffff' : '#020617',
      'circle-stroke-width': 1,
    },
  })

  map.addSource('hotspots', {
    type: 'geojson',
    data: emptyFeatureCollection(),
  })
  map.addLayer({
    id: 'hotspots-fill',
    type: 'fill',
    source: 'hotspots',
    paint: {
      'fill-color': hotspotFillColor,
      'fill-opacity': 0.08,
    },
  })
  map.addLayer({
    id: 'hotspots-outline',
    type: 'line',
    source: 'hotspots',
    paint: {
      'line-color': hotspotFillColor,
      'line-width': 1.5,
      'line-opacity': 0.65,
      'line-dasharray': [4, 3],
    },
  })

  map.addSource('chokepoints', {
    type: 'geojson',
    data: emptyFeatureCollection(),
  })
  map.addLayer({
    id: 'chokepoints-icon',
    type: 'circle',
    source: 'chokepoints',
    paint: {
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'risk_score'], 0],
        0, 5,
        30, 6.5,
        60, 8,
        100, 10,
      ],
      'circle-color': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'risk_score'], 0],
        0, theme === 'light' ? '#10b981' : '#34d399',
        40, theme === 'light' ? '#f59e0b' : '#f59e0b',
        70, '#ef4444',
        100, '#7f1d1d',
      ],
      'circle-stroke-color': theme === 'light' ? '#064e3b' : '#022c22',
      'circle-stroke-width': 1.5,
      'circle-opacity': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'risk_score'], 0],
        0, 0.85,
        100, 0.95,
      ],
    },
  })

  map.addSource('signals', {
    type: 'geojson',
    data: emptyFeatureCollection(),
  })
  // Glow layer: data-driven from source config (map_glow, map_radius_min/max)
  map.addLayer({
    id: 'signals-glow',
    type: 'circle',
    source: 'signals',
    filter: [
      'all',
      ['!=', ['get', 'signal_type'], 'military'],
      ['!=', ['get', 'signal_type'], 'conflict'],
      ['==', ['get', 'map_glow'], 1],
    ],
    paint: {
      'circle-radius': [
        'interpolate', ['linear'], ['coalesce', ['get', 'severity'], 0],
        0, ['*', ['coalesce', ['get', 'map_radius_min'], 4], 2.5],
        1, ['*', ['coalesce', ['get', 'map_radius_max'], 9], 2.5],
      ],
      'circle-color': ['get', 'color'],
      'circle-opacity': [
        'interpolate', ['linear'], ['coalesce', ['get', 'age_hours'], 24],
        0, 0.28,
        6, 0.22,
        24, 0.12,
      ],
      'circle-blur': 1,
    },
  })
  // Dot layer: all paint props driven by per-feature map_* properties from data source config
  map.addLayer({
    id: 'signals-dot',
    type: 'circle',
    source: 'signals',
    filter: [
      'all',
      ['!=', ['get', 'signal_type'], 'military'],
      ['!=', ['get', 'signal_type'], 'conflict'],
    ],
    paint: {
      'circle-radius': [
        'interpolate', ['linear'], ['coalesce', ['get', 'severity'], 0],
        0, ['coalesce', ['get', 'map_radius_min'], 4],
        1, ['coalesce', ['get', 'map_radius_max'], 9],
      ],
      'circle-color': ['get', 'color'],
      'circle-opacity': [
        'interpolate', ['linear'], ['coalesce', ['get', 'age_hours'], 24],
        0, ['coalesce', ['get', 'map_opacity'], 0.92],
        6, ['*', ['coalesce', ['get', 'map_opacity'], 0.92], 0.92],
        24, ['*', ['coalesce', ['get', 'map_opacity'], 0.92], 0.65],
      ],
      'circle-stroke-color': theme === 'light' ? '#ffffff' : '#020617',
      'circle-stroke-width': ['coalesce', ['get', 'map_stroke_width'], 1],
      'circle-blur': ['coalesce', ['get', 'map_blur'], 0.1],
    },
  })
  map.addLayer({
    id: 'signals-military-flight-icon',
    type: 'symbol',
    source: 'signals',
    filter: [
      'all',
      ['==', ['get', 'signal_type'], 'military'],
      ['==', ['get', 'activity_type'], 'flight'],
    ],
    layout: {
      'icon-image': 'wi-military-plane',
      'icon-size': ['interpolate', ['linear'], ['coalesce', ['get', 'severity'], 0], 0, 0.42, 1, 0.66],
      'icon-allow-overlap': true,
      'icon-ignore-placement': true,
    },
    paint: {
      'icon-opacity': 0.95,
    },
  })
  map.addLayer({
    id: 'signals-military-vessel-icon',
    type: 'symbol',
    source: 'signals',
    filter: [
      'all',
      ['==', ['get', 'signal_type'], 'military'],
      ['==', ['get', 'activity_type'], 'vessel'],
    ],
    layout: {
      'icon-image': 'wi-military-vessel',
      'icon-size': ['interpolate', ['linear'], ['coalesce', ['get', 'severity'], 0], 0, 0.38, 1, 0.62],
      'icon-allow-overlap': true,
      'icon-ignore-placement': true,
    },
    paint: {
      'icon-opacity': 0.95,
    },
  })

  map.addSource('convergences', {
    type: 'geojson',
    data: emptyFeatureCollection(),
  })
  map.addLayer({
    id: 'convergences-fill',
    type: 'circle',
    source: 'convergences',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'urgency_score'], 0, 24, 50, 36, 100, 52],
      'circle-color': theme === 'light' ? '#7c3aed' : '#c084fc',
      'circle-opacity': 0.14,
    },
  })
  map.addLayer({
    id: 'convergences-ring',
    type: 'circle',
    source: 'convergences',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'urgency_score'], 0, 24, 50, 36, 100, 52],
      'circle-color': 'transparent',
      'circle-stroke-color': theme === 'light' ? '#6d28d9' : '#c084fc',
      'circle-stroke-width': 2.5,
      'circle-opacity': 0.85,
    },
  })

  // NOTE: Earthquake and wildfire signals no longer have dedicated layers.
  // Their visual properties (radius, opacity, color, stroke, blur) are all
  // driven by per-feature map_* properties stamped from data source configs.
  // Users control presentation entirely from the DataSourcesManager UI.
}

function updateSourceData(map: any, sourceId: string, data: unknown) {
  const source = map.getSource(sourceId)
  source?.setData(data as any)
}

type PopupDetailRow = {
  label: string
  value: string
}

function splitPopupSegments(value?: string): string[] {
  if (!value) return []
  return value
    .split('·')
    .map((segment) => segment.trim())
    .filter(Boolean)
}

function extractPopupDetails(segments: string[]): PopupDetailRow[] {
  const rows: PopupDetailRow[] = []
  for (const segment of segments) {
    const separatorIndex = segment.indexOf(':')
    if (separatorIndex <= 0 || separatorIndex >= segment.length - 1) continue
    const label = segment.slice(0, separatorIndex).trim()
    const value = segment.slice(separatorIndex + 1).trim()
    if (!label || !value) continue
    rows.push({ label, value })
  }
  return rows
}

function truncateText(value: string, maxLength = 84): string {
  if (value.length <= maxLength) return value
  return `${value.slice(0, maxLength - 1).trimEnd()}...`
}

function formatSignalTypeLabel(value: string): string {
  const normalized = value.replace(/_/g, ' ').trim()
  if (!normalized) return 'Unknown'
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

function summarizeTopCounts(
  counts: Record<string, number>,
  limit = 3,
  labelFormatter?: (value: string) => string
): string {
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, count]) => `${labelFormatter ? labelFormatter(label) : label} ${count}`)
    .join(', ')
}

const LAYER_LABELS: Record<string, string> = {
  countryIntensity: 'Country intensity',
  tensionBorders: 'Tension borders',
  tensionArcs: 'Tension arcs',
  countryBoundaries: 'Country boundaries',
  conflictZones: 'Conflict zones',
  signals: 'Signals',
  convergences: 'Convergences',
  hotspots: 'Hotspots',
  chokepoints: 'Chokepoints',
}

const LAYER_SHORT_LABELS: Record<string, string> = {
  countryIntensity: 'CI',
  tensionBorders: 'TB',
  tensionArcs: 'TA',
  countryBoundaries: 'CB',
  conflictZones: 'CF',
  signals: 'SG',
  convergences: 'CV',
  hotspots: 'HS',
  chokepoints: 'CP',
}

const LAYER_COLORS: Record<string, string> = {
  countryIntensity: '#ef4444',
  tensionBorders: '#f97316',
  tensionArcs: '#f43f5e',
  countryBoundaries: '#38bdf8',
  conflictZones: '#dc2626',
  signals: '#8b5cf6',
  convergences: '#a855f7',
  hotspots: '#3b82f6',
  chokepoints: '#10b981',
}

const EVENT_SOURCE_HEALTH_KEY_BY_SLUG: Record<string, string> = {
  events_acled: 'acled',
  events_gdelt_tensions: 'gdelt_tensions',
  events_military: 'military',
  events_infrastructure: 'infrastructure',
  events_gdelt_news: 'gdelt_news',
  events_usgs: 'usgs',
}

type LayerDockItem = {
  key: string
  label: string
  short: string
  color: string
}

type SourceHealthTone = 'ok' | 'degraded' | 'error' | 'disabled' | 'unknown'

type DataSourceDockItem = {
  id: string
  name: string
  slug: string
  sourceKey: string
  canonicalKey: string
  aliases: string[]
  enabled: boolean
  status: string
  tone: SourceHealthTone
  count: number
  signalCount: number
}

function formatLayerLabelFromKey(key: string): string {
  const raw = String(key)
  const withSpaces = raw.replace(/([a-z])([A-Z])/g, '$1 $2')
  return withSpaces.charAt(0).toUpperCase() + withSpaces.slice(1)
}

function formatShortLayerLabel(label: string): string {
  const letters = label
    .split(/\s+/)
    .map((part) => part.trim().charAt(0).toUpperCase())
    .filter(Boolean)
    .slice(0, 2)
  return letters.join('') || 'LY'
}

function classifySourceHealthTone(enabled: boolean, details: any): SourceHealthTone {
  if (!enabled) return 'disabled'
  if (!details) return 'unknown'
  if (details?.degraded) return 'degraded'
  if (details?.ok === false) {
    const rawError = String(details?.error || details?.last_error || '').toLowerCase()
    if (
      rawError.includes('missing_api_key')
      || rawError.includes('disabled')
      || rawError.includes('429')
      || rawError.includes('403')
      || rawError.includes('rate limited')
    ) {
      return 'degraded'
    }
    return 'error'
  }
  return 'ok'
}

function sourceToneClass(tone: SourceHealthTone): string {
  if (tone === 'ok') return 'text-emerald-400'
  if (tone === 'degraded') return 'text-yellow-400'
  if (tone === 'error') return 'text-red-400'
  if (tone === 'disabled') return 'text-muted-foreground'
  return 'text-slate-400'
}

function sourceToneLabel(tone: SourceHealthTone, count: number): string {
  if (tone === 'ok') return `ok (${count})`
  if (tone === 'degraded') return `degraded (${count})`
  if (tone === 'error') return 'error'
  if (tone === 'disabled') return 'disabled'
  return 'unknown'
}

function resolveEventSourceHealthKey(source: UnifiedDataSource): string | null {
  const slug = String(source.slug || '').trim().toLowerCase()
  if (!slug) return null
  const mapped = EVENT_SOURCE_HEALTH_KEY_BY_SLUG[slug]
  if (mapped) return mapped
  // Events sources: strip "events_" prefix for the health key
  if (slug.startsWith('events_')) {
    const suffix = slug.slice('events_'.length).trim().toLowerCase()
    return suffix || null
  }
  // Story and custom sources: the API now returns health keyed by full slug
  const sourceKey = String(source.source_key || '').trim().toLowerCase()
  if (sourceKey === 'stories' || sourceKey === 'custom') return slug
  return slug
}

function normalizeSourceToken(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^\w]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function MapRightDock({
  selection,
  relatedEvents,
  tab,
  onTabChange,
  expanded,
  onExpandedChange,
  onCloseSelection,
  layerItems,
  toggles,
  onToggle,
  signalCount,
  signalTotal,
  criticalCount,
  geocodedSignalCount,
  convergenceCount,
  hotspotCount,
  byType,
  colors,
  sourceItems,
  sourceVisibility,
  onToggleSource,
  sourceError,
}: {
  selection: FlyoutSelection | null
  relatedEvents: FlyoutRelatedEvent[]
  tab: FlyoutTab
  onTabChange: (next: FlyoutTab) => void
  expanded: boolean
  onExpandedChange: (next: boolean) => void
  onCloseSelection: () => void
  layerItems: LayerDockItem[]
  toggles: LayerToggles
  onToggle: (key: string) => void
  signalCount: number
  signalTotal: number
  criticalCount: number
  geocodedSignalCount: number
  convergenceCount: number
  hotspotCount: number
  byType: Record<string, number>
  colors: SignalPalette
  sourceItems: DataSourceDockItem[]
  sourceVisibility: Record<string, boolean>
  onToggleSource: (canonicalKey: string) => void
  sourceError: string | null
}) {
  const signalLabel = signalTotal > signalCount ? `${signalCount}/${signalTotal}` : `${signalCount}`
  const typeEntries = Object.entries(byType)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
  const bodySegments = splitPopupSegments(selection?.body)
  const detailRows = extractPopupDetails(bodySegments)
  const summaryRows = bodySegments.filter((segment) => !segment.includes(':'))

  const openContext = () => {
    onTabChange('context')
    onExpandedChange(true)
  }
  const openLayers = () => {
    onTabChange('layers')
    onExpandedChange(true)
  }
  const openSources = () => {
    onTabChange('sources')
    onExpandedChange(true)
  }

  return (
    <div
      className={`absolute inset-y-0 right-0 z-20 border-l border-border/70 bg-background/94 backdrop-blur-md shadow-2xl transition-[width] duration-300 pointer-events-auto ${expanded ? 'w-[420px] max-w-[100vw]' : 'w-[58px]'}`}
    >
      <div className="h-full flex">
        <div className="w-[58px] shrink-0 border-r border-border/60 bg-card/50 px-1.5 py-2.5 space-y-2">
          <button
            type="button"
            onClick={() => onExpandedChange(!expanded)}
            className="w-full h-8 rounded-lg border border-border bg-background text-[11px] font-semibold text-muted-foreground hover:text-foreground hover:bg-muted/45 transition-colors"
            title={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? '<' : '>'}
          </button>

          <button
            type="button"
            onClick={openContext}
            className={`relative w-full h-9 rounded-lg border text-[10px] font-semibold tracking-wide transition-colors ${tab === 'context' && expanded ? 'border-blue-500/45 bg-blue-500/15 text-blue-300' : 'border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/45'}`}
            title="Selected context"
          >
            CTX
            {selection ? <span className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-emerald-400" /> : null}
          </button>

          <button
            type="button"
            onClick={openLayers}
            className={`w-full h-9 rounded-lg border text-[10px] font-semibold tracking-wide transition-colors ${tab === 'layers' && expanded ? 'border-orange-500/45 bg-orange-500/15 text-orange-300' : 'border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/45'}`}
            title="Layer controls"
          >
            LYR
          </button>

          <button
            type="button"
            onClick={openSources}
            className={`w-full h-9 rounded-lg border text-[10px] font-semibold tracking-wide transition-colors ${tab === 'sources' && expanded ? 'border-cyan-500/45 bg-cyan-500/15 text-cyan-300' : 'border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/45'}`}
            title="Data sources"
          >
            SRC
          </button>

          <div className="rounded-lg border border-border/70 bg-background/80 px-1 py-1.5 text-center">
            <div className="text-[8px] leading-none text-muted-foreground uppercase">sig</div>
            <div className="mt-1 text-[10px] leading-none font-semibold text-foreground">{signalLabel}</div>
          </div>
        </div>

        {expanded ? (
          <div className="flex-1 min-w-0 h-full flex flex-col">
            <div className="shrink-0 border-b border-border/70 px-4 py-3 bg-gradient-to-r from-card/90 via-card/90 to-muted/55">
              <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                {tab === 'context' ? 'Selected Context' : tab === 'layers' ? 'Layer Control' : 'Data Source Control'}
              </div>
              <div className="mt-1 flex items-center justify-between gap-2">
                <div className="text-sm font-semibold text-foreground">
                  {tab === 'context'
                    ? (selection ? `${selection.category} intelligence` : 'Nothing selected')
                    : tab === 'layers'
                      ? 'Map layers and signal legend'
                      : 'Configured data sources'}
                </div>
                {tab === 'context' && selection ? (
                  <button
                    type="button"
                    onClick={onCloseSelection}
                    className="rounded-md border border-border px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                  >
                    Clear
                  </button>
                ) : null}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {tab === 'context' ? (
                <>
                  {selection ? (
                    <div className="space-y-3">
                      <div className="space-y-2 rounded-xl border border-border/70 bg-gradient-to-br from-card via-card to-muted/30 p-3">
                        <div className="text-[14px] font-semibold text-foreground leading-tight">{selection.title}</div>
                        {selection.subtitle ? (
                          <div className="flex flex-wrap gap-1.5">
                            {splitPopupSegments(selection.subtitle).map((segment, index) => (
                              <span
                                key={`selection-subtitle-${segment}-${index}`}
                                className="inline-flex items-center rounded-full border border-border bg-background/75 px-2 py-0.5 text-[10px] leading-4 text-muted-foreground"
                              >
                                {segment}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        {summaryRows.length > 0 ? (
                          <div className="flex flex-wrap gap-1.5">
                            {summaryRows.map((row, index) => (
                              <span
                                key={`selection-summary-${row}-${index}`}
                                className="inline-flex items-center rounded-md border border-border/70 bg-muted/45 px-2 py-1 text-[10px] leading-4 text-foreground/90"
                              >
                                {row}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        {detailRows.length > 0 ? (
                          <dl className="space-y-1.5 rounded-md border border-border/70 bg-background/70 p-2.5">
                            {detailRows.map((row, index) => (
                              <div key={`selection-detail-${row.label}-${index}`} className="grid grid-cols-[100px_minmax(0,1fr)] items-start gap-2">
                                <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">{row.label}</dt>
                                <dd className="text-[11px] leading-4 text-foreground break-words">{row.value}</dd>
                              </div>
                            ))}
                          </dl>
                        ) : null}
                      </div>

                      <div className="rounded-xl border border-border/70 bg-card/70 p-2.5 space-y-1.5">
                        <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Nearby and Related</div>
                        {relatedEvents.length > 0 ? relatedEvents.map((event) => (
                          <div key={event.id} className="rounded-md border border-border/55 bg-background/70 px-2 py-1.5 space-y-0.5">
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] uppercase tracking-wide text-muted-foreground">{event.kind}</span>
                              <span className="ml-auto text-[10px] font-semibold text-foreground">{Math.round(event.score)}</span>
                            </div>
                            <div className="text-[11px] text-foreground leading-4">{event.title}</div>
                            <div className="text-[10px] text-muted-foreground leading-4">{event.subtitle}</div>
                          </div>
                        )) : (
                          <div className="rounded-md border border-dashed border-border/60 bg-background/65 px-2 py-2 text-[11px] text-muted-foreground">
                            No nearby or related events found for this selection.
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-xl border border-dashed border-border/70 bg-card/35 px-3 py-3 text-[11px] text-muted-foreground">
                      Click a country, signal, hotspot, chokepoint, arc, or convergence to open context here.
                    </div>
                  )}
                </>
              ) : null}

              {tab === 'layers' ? (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-lg border border-border/70 bg-card/70 px-2.5 py-2">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Signals</div>
                      <div className="mt-1 text-[14px] font-semibold text-foreground">{signalLabel}</div>
                    </div>
                    <div className="rounded-lg border border-border/70 bg-card/70 px-2.5 py-2">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Critical</div>
                      <div className={`mt-1 text-[14px] font-semibold ${criticalCount > 0 ? 'text-red-400' : 'text-foreground'}`}>{criticalCount}</div>
                    </div>
                    <div className="rounded-lg border border-border/70 bg-card/70 px-2.5 py-2">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Geocoded</div>
                      <div className="mt-1 text-[14px] font-semibold text-emerald-400">{geocodedSignalCount}</div>
                    </div>
                    <div className="rounded-lg border border-border/70 bg-card/70 px-2.5 py-2">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Convergences</div>
                      <div className="mt-1 text-[14px] font-semibold text-purple-400">{convergenceCount}</div>
                    </div>
                    <div className="rounded-lg border border-border/70 bg-card/70 px-2.5 py-2 col-span-2">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Hotspots</div>
                      <div className="mt-1 text-[14px] font-semibold text-blue-400">{hotspotCount}</div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-border/70 bg-card/70 p-2.5 space-y-1.5">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Layer Matrix</div>
                    {layerItems.map((item) => {
                      const enabled = toggles[item.key]
                      return (
                        <button
                          key={item.key}
                          type="button"
                          onClick={() => onToggle(item.key)}
                          className="w-full rounded-md border border-border/55 bg-background/65 px-2 py-1.5 text-left hover:bg-muted/40 transition-colors flex items-center justify-between gap-2"
                        >
                          <span className="flex items-center gap-2">
                            <span className="inline-flex h-5 w-6 items-center justify-center rounded border border-border bg-background text-[9px] font-semibold tracking-wide text-muted-foreground">{item.short}</span>
                            <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                            <span className="text-[11px] text-foreground">{item.label}</span>
                          </span>
                          <span className={enabled ? 'text-[10px] font-semibold text-emerald-400' : 'text-[10px] text-muted-foreground'}>
                            {enabled ? 'ON' : 'OFF'}
                          </span>
                        </button>
                      )
                    })}
                  </div>

                  <div className="rounded-xl border border-border/70 bg-card/70 p-2.5 space-y-1.5">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Source Layers</div>
                    {sourceItems.length > 0 ? sourceItems.map((source) => {
                      const visible = sourceVisibility[source.canonicalKey] ?? source.enabled
                      return (
                        <button
                          key={`source-layer-${source.id}`}
                          type="button"
                          onClick={() => onToggleSource(source.canonicalKey)}
                          className="w-full rounded-md border border-border/55 bg-background/65 px-2 py-1.5 text-left hover:bg-muted/40 transition-colors flex items-center justify-between gap-2"
                        >
                          <span className="min-w-0 flex items-center gap-2">
                            <span className="inline-flex h-5 min-w-6 items-center justify-center rounded border border-border bg-background px-1 text-[9px] font-semibold tracking-wide text-muted-foreground">
                              {source.sourceKey.slice(0, 3).toUpperCase()}
                            </span>
                            <span className="text-[11px] text-foreground truncate">{source.name}</span>
                          </span>
                          <span className={visible ? 'text-[10px] font-semibold text-emerald-400' : 'text-[10px] text-muted-foreground'}>
                            {visible ? 'ON' : 'OFF'}
                          </span>
                        </button>
                      )
                    }) : (
                      <div className="text-[11px] text-muted-foreground">No configured sources.</div>
                    )}
                  </div>

                  <div className="rounded-xl border border-border/70 bg-card/70 p-2.5 space-y-1.5">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Signal Legend</div>
                    {typeEntries.length > 0 ? typeEntries.map(([type, count]) => (
                      <div key={type} className="flex items-center gap-2 text-[11px]">
                        <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: colors[type] || '#64748b' }} />
                        <span className="text-foreground capitalize">{type.replace(/_/g, ' ')}</span>
                        <span className="ml-auto font-semibold text-foreground">{count}</span>
                      </div>
                    )) : (
                      <div className="text-[11px] text-muted-foreground">No active signals.</div>
                    )}
                  </div>
                </>
              ) : null}

              {tab === 'sources' ? (
                <div className="rounded-xl border border-border/70 bg-card/70 p-2.5 space-y-1.5">
                  <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Data SDK Sources</div>
                  {sourceError ? (
                    <div className="rounded-md border border-red-500/30 bg-red-500/10 px-2 py-1 text-[10px] text-red-400">
                      {sourceError}
                    </div>
                  ) : null}
                  {sourceItems.length > 0 ? sourceItems.map((source) => (
                    <div key={source.id} className="rounded-md border border-border/55 bg-background/70 px-2 py-1.5 space-y-0.5">
                      <div className="flex items-center justify-between gap-2 min-w-0">
                        <div className="text-[11px] text-foreground truncate">{source.name}</div>
                        <div className={`text-[10px] font-semibold shrink-0 ${sourceToneClass(source.tone)}`}>
                          {sourceToneLabel(source.tone, source.count)}
                        </div>
                      </div>
                      <div className="text-[10px] text-muted-foreground truncate flex items-center justify-between gap-2">
                        <span className="truncate">{source.sourceKey} / {source.slug}</span>
                        <span className="shrink-0">{source.signalCount} {source.sourceKey === 'events' ? 'signals' : 'records'}</span>
                      </div>
                      <div className="text-[10px] text-muted-foreground flex items-center justify-between gap-2">
                        <span>{source.enabled ? 'enabled' : 'disabled'} / {source.status}</span>
                        <button
                          type="button"
                          onClick={() => onToggleSource(source.canonicalKey)}
                          className={`rounded border px-1.5 py-0.5 transition-colors ${((sourceVisibility[source.canonicalKey] ?? source.enabled) ? 'border-emerald-500/35 text-emerald-400 bg-emerald-500/10' : 'border-border text-muted-foreground hover:text-foreground hover:bg-muted/40')}`}
                        >
                          {(sourceVisibility[source.canonicalKey] ?? source.enabled) ? 'Shown' : 'Hidden'}
                        </button>
                      </div>
                    </div>
                  )) : (
                    <div className="text-[11px] text-muted-foreground">No configured data sources.</div>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}

function militaryEntityKey(signal: WorldSignal): string {
  if (signal.signal_type !== 'military') return ''
  const meta = (signal.metadata || {}) as Record<string, unknown>
  const activityType = String(meta.activity_type || '').trim().toLowerCase() || 'flight'
  const transponder = String(meta.transponder || '').trim().toLowerCase()
  if (transponder) return `${activityType}:${transponder}`
  const callsign = String(meta.callsign || '').trim().toUpperCase().replace(/\s+/g, '')
  const iso3 = normalizeCountryCode(signal.country) || ''
  if (callsign) return `${activityType}:${callsign}:${iso3}`
  return ''
}

export default function WorldMap({ isConnected = true }: { isConnected?: boolean }) {
  const theme = useAtomValue(themeAtom)

  const { data: unifiedDataSourcesData } = useQuery({
    queryKey: ['unified-data-sources'],
    queryFn: () => getUnifiedDataSources(),
    refetchInterval: 120000,
    staleTime: 60000,
  })

  // Build dynamic palette: start with hardcoded defaults, then overlay
  // map_color values from each data source config so users can control
  // signal colors entirely from the DataSourceSDK level.
  const colors = useMemo(() => {
    const base: SignalPalette = { ...(theme === 'light' ? SIGNAL_COLORS_LIGHT : SIGNAL_COLORS_DARK) }
    const sources = Array.isArray(unifiedDataSourcesData) ? unifiedDataSourcesData : []
    for (const source of sources) {
      const cfg = (source.config || {}) as Record<string, unknown>
      const mapColor = typeof cfg.map_color === 'string' ? cfg.map_color.trim() : ''
      if (!mapColor) continue
      // Extract signal_types from config — each maps to the user-chosen color
      const signalTypes = Array.isArray(cfg.signal_types) ? cfg.signal_types : []
      for (const st of signalTypes) {
        const key = String(st).trim().toLowerCase()
        if (key) base[key] = mapColor
      }
      // Also map the source name itself (e.g. "nasa_firms" → color)
      const sourceNames = Array.isArray(cfg.source_names) ? cfg.source_names : []
      for (const sn of sourceNames) {
        const key = String(sn).trim().toLowerCase()
        if (key) base[key] = mapColor
      }
    }
    return base
  }, [theme, unifiedDataSourcesData])

  // Build lookup: signal_type/source_name → map presentation config from data sources
  const mapPresentation = useMemo(() => {
    const bySignalType: Record<string, MapPresentationConfig> = {}
    const bySourceName: Record<string, MapPresentationConfig> = {}
    const sources = Array.isArray(unifiedDataSourcesData) ? unifiedDataSourcesData : []
    for (const source of sources) {
      if (source.source_key !== 'events') continue
      const cfg = (source.config || {}) as Record<string, unknown>
      const pres: MapPresentationConfig = {
        map_color: String(cfg.map_color || MAP_PRESENTATION_DEFAULTS.map_color),
        map_layer_label: String(cfg.map_layer_label || ''),
        map_layer_short: String(cfg.map_layer_short || ''),
        map_radius_min: Number(cfg.map_radius_min ?? MAP_PRESENTATION_DEFAULTS.map_radius_min),
        map_radius_max: Number(cfg.map_radius_max ?? MAP_PRESENTATION_DEFAULTS.map_radius_max),
        map_opacity: Number(cfg.map_opacity ?? MAP_PRESENTATION_DEFAULTS.map_opacity),
        map_blur: Number(cfg.map_blur ?? MAP_PRESENTATION_DEFAULTS.map_blur),
        map_stroke_width: Number(cfg.map_stroke_width ?? MAP_PRESENTATION_DEFAULTS.map_stroke_width),
        map_glow: cfg.map_glow !== false && cfg.map_glow !== 0,
        map_dedicated_toggle: Boolean(cfg.map_dedicated_toggle),
        slug: source.slug,
      }
      const signalTypes = Array.isArray(cfg.signal_types) ? cfg.signal_types : []
      for (const st of signalTypes) {
        const key = String(st).trim().toLowerCase()
        if (key) bySignalType[key] = pres
      }
      const sourceNames = Array.isArray(cfg.source_names) ? cfg.source_names : []
      for (const sn of sourceNames) {
        const key = String(sn).trim().toLowerCase()
        if (key) bySourceName[key] = pres
      }
    }
    return { bySignalType, bySourceName }
  }, [unifiedDataSourcesData])

  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<any>(null)
  const mapReadyRef = useRef(false)
  const [mapReady, setMapReady] = useState(false)
  const [mapInitError, setMapInitError] = useState<string | null>(null)
  const [layerToggles, setLayerToggles] = useState<LayerToggles>(DEFAULT_LAYER_TOGGLES)
  const [flyoutExpanded, setFlyoutExpanded] = useState(false)
  const [flyoutTab, setFlyoutTab] = useState<FlyoutTab>('context')
  const [flyoutSelection, setFlyoutSelection] = useState<FlyoutSelection | null>(null)
  const [hoverTooltip, setHoverTooltip] = useState<{
    x: number
    y: number
    name: string
    instability: number
    tension: number
  } | null>(null)
  const pollingInterval = isConnected ? false : 180000
  const sourceRefreshInterval = 120000

  const { data: worldSourceStatusData } = useQuery({
    queryKey: ['events-sources'],
    queryFn: getWorldSourceStatus,
    refetchInterval: sourceRefreshInterval,
    staleTime: 60000,
  })

  const { data: signalsData, isLoading: signalsLoading } = useQuery({
    queryKey: ['world-signals-map', { page_size: MAP_SIGNAL_PAGE_SIZE, max: MAP_SIGNAL_MAX }],
    queryFn: async () => {
      const mergedSignals: WorldSignal[] = []
      const seenMilitaryKeys = new Set<string>()
      let lastCollection: string | null = null
      let offset = 0
      let total = 0
      while (mergedSignals.length < MAP_SIGNAL_MAX) {
        const pageLimit = Math.min(MAP_SIGNAL_PAGE_SIZE, MAP_SIGNAL_MAX - mergedSignals.length)
        const page = await getWorldSignals({ limit: pageLimit, offset })
        if (!lastCollection) {
          lastCollection = page.last_collection
        }
        const chunk = Array.isArray(page.signals) ? page.signals : []
        if (!chunk.length) {
          total = Number(page.total || mergedSignals.length)
          break
        }
        for (const row of chunk) {
          const key = militaryEntityKey(row)
          if (key) {
            if (seenMilitaryKeys.has(key)) {
              continue
            }
            seenMilitaryKeys.add(key)
          }
          mergedSignals.push(row)
          if (mergedSignals.length >= MAP_SIGNAL_MAX) {
            break
          }
        }
        total = Number(page.total || mergedSignals.length)

        const nextOffset =
          typeof page.next_offset === 'number'
            ? page.next_offset
            : (offset + chunk.length)
        const hasMore =
          typeof page.has_more === 'boolean'
            ? page.has_more
            : (chunk.length >= pageLimit && nextOffset > offset)
        if (!hasMore || nextOffset <= offset) {
          break
        }
        offset = nextOffset
      }

      return {
        signals: mergedSignals,
        total,
        last_collection: lastCollection,
      }
    },
    refetchInterval: pollingInterval,
    retry: 2,
    retryDelay: (attempt) => Math.min(10000, attempt * 1500),
  })

  const { data: convergenceData, isLoading: convergenceLoading } = useQuery({
    queryKey: ['world-convergences'],
    queryFn: getConvergenceZones,
    refetchInterval: pollingInterval,
    retry: 2,
    retryDelay: (attempt) => Math.min(10000, attempt * 1500),
  })

  const { data: regionsData, isLoading: regionsLoading } = useQuery({
    queryKey: ['world-regions'],
    queryFn: getWorldRegions,
    staleTime: 60 * 1000,
    refetchInterval: pollingInterval,
    retry: 2,
    retryDelay: (attempt) => Math.min(10000, attempt * 1500),
  })

  const { data: tensionsData, isLoading: tensionsLoading } = useQuery({
    queryKey: ['world-tensions', { min_tension: 0, limit: 100 }],
    queryFn: () => getTensionPairs({ min_tension: 0, limit: 100 }),
    refetchInterval: pollingInterval,
    retry: 2,
    retryDelay: (attempt) => Math.min(10000, attempt * 1500),
  })

  const { data: instabilityData, isLoading: instabilityLoading } = useQuery({
    queryKey: ['world-instability', { min_score: 0, limit: 250 }],
    queryFn: () => getInstabilityScores({ min_score: 0, limit: 250 }),
    refetchInterval: pollingInterval,
    retry: 2,
    retryDelay: (attempt) => Math.min(10000, attempt * 1500),
  })

  const { data: countryGeoData, isLoading: countriesLoading } = useQuery({
    queryKey: ['world-country-boundaries'],
    queryFn: async () => {
      const response = await fetch(COUNTRY_BOUNDARY_URL)
      if (!response.ok) {
        throw new Error(`Country boundary fetch failed: ${response.status}`)
      }
      return toCountryBoundaryGeoJSON(await response.json())
    },
    retry: 4,
    retryDelay: (attempt) => Math.min(12000, attempt * 2000),
    staleTime: 24 * 60 * 60 * 1000,
  })

  const [stableSignalsData] = useStickyValue(
    signalsData,
    { signals: [] as WorldSignal[], total: 0, last_collection: null as string | null }
  )
  const [stableConvergenceData] = useStickyValue(
    convergenceData,
    { zones: [] as ConvergenceZone[], total: 0 }
  )
  const [stableRegionsData] = useStickyValue(
    regionsData,
    {
      version: 0,
      updated_at: null as string | null,
      hotspots: [] as WorldRegionHotspot[],
      chokepoints: [] as WorldRegionChokepoint[],
    }
  )
  const [stableTensionsData] = useStickyValue(
    tensionsData,
    { tensions: [] as TensionPair[], total: 0 }
  )
  const [stableInstabilityData] = useStickyValue(
    instabilityData,
    { scores: [] as Array<{
      country: string
      iso3: string
      country_name?: string | null
      score: number
      trend: 'rising' | 'falling' | 'stable'
      change_24h: number | null
      change_7d: number | null
      components: Record<string, number>
      contributing_signals: Array<Record<string, any>>
      last_updated: string | null
    }>, total: 0 }
  )
  const [stableCountryGeoData] = useStickyValue(
    countryGeoData,
    EMPTY_COUNTRY_BOUNDARY_COLLECTION
  )

  const allSignals = stableSignalsData.signals || []
  const convergences = stableConvergenceData.zones || []
  const hotspots = stableRegionsData.hotspots || []
  const chokepoints = stableRegionsData.chokepoints || []
  const tensions = stableTensionsData.tensions || []
  const instabilityScores = stableInstabilityData.scores || []
  const [sourceVisibilityByKey, setSourceVisibilityByKey] = useState<Record<string, boolean>>({})
  const countryCentroids = useMemo(() => buildCountryCentroids(stableCountryGeoData), [stableCountryGeoData])

  const layerDockItems = useMemo<LayerDockItem[]>(() => {
    // Static structural items
    const items: LayerDockItem[] = (Object.keys(STRUCTURAL_LAYER_GROUPS) as string[]).map((key) => {
      const label = LAYER_LABELS[key] || formatLayerLabelFromKey(key)
      return {
        key,
        label,
        short: LAYER_SHORT_LABELS[key] || formatShortLayerLabel(label),
        color: LAYER_COLORS[key] || '#64748b',
      }
    })
    // Dynamic items from data sources with map_dedicated_toggle: true
    const sources = Array.isArray(unifiedDataSourcesData) ? unifiedDataSourcesData : []
    for (const source of sources) {
      if (source.source_key !== 'events') continue
      const cfg = (source.config || {}) as Record<string, unknown>
      if (!cfg.map_dedicated_toggle) continue
      const label = String(cfg.map_layer_label || source.name || source.slug)
      const short = String(cfg.map_layer_short || label.slice(0, 2).toUpperCase())
      const color = String(cfg.map_color || '#64748b')
      items.push({ key: `dsrc_${source.slug}`, label, short, color })
    }
    return items
  }, [unifiedDataSourcesData])

  const sourceSignalCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const signal of allSignals) {
      const key = normalizeSourceToken(signal.source || '')
      if (!key) continue
      counts[key] = (counts[key] || 0) + 1
    }
    return counts
  }, [allSignals])

  const sourceDockItems = useMemo<DataSourceDockItem[]>(() => {
    const sourceHealthMap = worldSourceStatusData?.sources || {}
    const sources = Array.isArray(unifiedDataSourcesData) ? unifiedDataSourcesData : []
    return sources
      .slice()
      .sort((a, b) => {
        if (a.enabled !== b.enabled) return Number(b.enabled) - Number(a.enabled)
        const keyCompare = String(a.source_key || '').localeCompare(String(b.source_key || ''))
        if (keyCompare !== 0) return keyCompare
        return String(a.name || '').localeCompare(String(b.name || ''))
      })
      .map((source) => {
        const healthKey = resolveEventSourceHealthKey(source)
        const healthDetails = healthKey ? sourceHealthMap[healthKey] : null
        const tone = classifySourceHealthTone(Boolean(source.enabled), healthDetails)
        const count = Number(healthDetails?.count || 0)
        const slugToken = normalizeSourceToken(String(source.slug || ''))
        const sourceKeyToken = normalizeSourceToken(String(source.source_key || ''))
        const healthToken = normalizeSourceToken(healthKey || '')
        const aliases = new Set<string>()
        if (slugToken) aliases.add(slugToken)
        if (slugToken.startsWith('events_')) aliases.add(slugToken.slice('events_'.length))
        if (sourceKeyToken && sourceKeyToken !== 'events') aliases.add(sourceKeyToken)
        if (healthToken) aliases.add(healthToken)
        const canonicalKey = healthToken || (slugToken.startsWith('events_') ? slugToken.slice('events_'.length) : slugToken) || source.id
        aliases.add(canonicalKey)
        let signalCount = 0
        for (const alias of aliases) {
          signalCount += Number(sourceSignalCounts[alias] || 0)
        }
        // For non-events sources, use total_records from the health endpoint
        // since they don't produce EventsSignal rows
        if (signalCount === 0 && healthDetails?.total_records) {
          signalCount = Number(healthDetails.total_records)
        }
        return {
          id: source.id,
          name: String(source.name || source.slug || source.id),
          slug: String(source.slug || ''),
          sourceKey: String(source.source_key || 'custom'),
          canonicalKey,
          aliases: [...aliases],
          enabled: Boolean(source.enabled),
          status: String(source.status || 'unknown'),
          tone,
          count,
          signalCount,
        }
      })
  }, [sourceSignalCounts, unifiedDataSourcesData, worldSourceStatusData?.sources])

  useEffect(() => {
    if (!sourceDockItems.length) return
    setSourceVisibilityByKey((prev) => {
      const next: Record<string, boolean> = {}
      for (const source of sourceDockItems) {
        next[source.canonicalKey] = prev[source.canonicalKey] ?? source.enabled
      }
      return next
    })
  }, [sourceDockItems])

  // Initialize dynamic toggle state for data sources with map_dedicated_toggle
  useEffect(() => {
    const sources = Array.isArray(unifiedDataSourcesData) ? unifiedDataSourcesData : []
    const dynamicKeys: string[] = []
    for (const source of sources) {
      if (source.source_key !== 'events') continue
      const cfg = (source.config || {}) as Record<string, unknown>
      if (!cfg.map_dedicated_toggle) continue
      dynamicKeys.push(`dsrc_${source.slug}`)
    }
    if (dynamicKeys.length === 0) return
    setLayerToggles((prev) => {
      const next = { ...prev }
      let changed = false
      for (const key of dynamicKeys) {
        if (!(key in next)) {
          next[key] = true
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [unifiedDataSourcesData])

  const sourceAliasToCanonical = useMemo(() => {
    const out: Record<string, string> = {}
    for (const source of sourceDockItems) {
      out[source.canonicalKey] = source.canonicalKey
      for (const alias of source.aliases) {
        if (!out[alias]) out[alias] = source.canonicalKey
      }
    }
    return out
  }, [sourceDockItems])

  const signals = useMemo(() => {
    if (!sourceDockItems.length) return allSignals
    return allSignals.filter((signal) => {
      const signalKey = normalizeSourceToken(signal.source || '')
      if (!signalKey) return true
      const canonical = sourceAliasToCanonical[signalKey]
      if (!canonical) return true
      const visible = sourceVisibilityByKey[canonical]
      return visible ?? true
    })
  }, [allSignals, sourceAliasToCanonical, sourceDockItems.length, sourceVisibilityByKey])

  const geocodedSignalsGeoJSON = useMemo(
    () => signalsToGeoJSON(
      signals, colors, countryCentroids,
      mapPresentation.bySignalType,
      mapPresentation.bySourceName,
    ),
    [signals, colors, countryCentroids, mapPresentation]
  )
  const geocodedSignalPoints = useMemo(
    () => geocodedSignalsGeoJSON.features.map((feature) => ({
      lon: Number(feature.geometry.coordinates[0]),
      lat: Number(feature.geometry.coordinates[1]),
    })),
    [geocodedSignalsGeoJSON]
  )
  const geocodedSignalCount = geocodedSignalsGeoJSON.features.length

  const signalStatsExtra = useMemo(() => {
    const byType: Record<string, number> = {}
    let criticalCount = 0
    let oldestMs: number | null = null
    for (const signal of signals) {
      byType[signal.signal_type] = (byType[signal.signal_type] || 0) + 1
      if ((signal.severity || 0) >= 0.7) criticalCount++
      if (signal.detected_at) {
        const ms = new Date(signal.detected_at).getTime()
        if (!Number.isNaN(ms)) {
          if (oldestMs == null || ms < oldestMs) oldestMs = ms
        }
      }
    }
    const oldestSignalHours = oldestMs != null ? (Date.now() - oldestMs) / 3_600_000 : null
    return { byType, criticalCount, oldestSignalHours }
  }, [signals])

  const relatedEvents = useMemo<FlyoutRelatedEvent[]>(() => {
    if (!flyoutSelection) return []

    const selectedIso = normalizeCountryCode(flyoutSelection.iso3 || flyoutSelection.countryName || '')
    const hasCenter = isFiniteCoordinatePair(flyoutSelection.lat, flyoutSelection.lon)
    const centerLat = hasCenter ? Number(flyoutSelection.lat) : 0
    const centerLon = hasCenter ? Number(flyoutSelection.lon) : 0
    const rows: FlyoutRelatedEvent[] = []

    for (const signal of signals) {
      const signalIso = normalizeCountryCode(
        signal.country_iso3
        || signal.country_name
        || signal.country
        || ''
      )
      const severity = Number(signal.severity || 0)
      let score = 0

      if (selectedIso && signalIso === selectedIso) {
        score = Math.max(score, 100 + severity * 100)
      }
      if (selectedIso) {
        const pair = pairFromSignal(signal)
        if (pair && (pair[0] === selectedIso || pair[1] === selectedIso)) {
          score = Math.max(score, 90 + severity * 100)
        }
      }
      if (hasCenter && isFiniteCoordinatePair(signal.latitude, signal.longitude)) {
        const distanceKm = haversineDistanceKm(centerLat, centerLon, Number(signal.latitude), Number(signal.longitude))
        if (distanceKm <= 1800) {
          score = Math.max(score, 85 + severity * 100 - Math.min(60, distanceKm / 28))
        }
      }
      if (score <= 0) continue

      const ageHours = signal.detected_at ? (Date.now() - new Date(signal.detected_at).getTime()) / 3_600_000 : null
      const ageLabel = ageHours == null
        ? ''
        : ageHours < 1
          ? '<1h'
          : ageHours < 24
            ? `${Math.round(ageHours)}h`
            : `${Math.round(ageHours / 24)}d`
      rows.push({
        id: `signal:${signal.signal_id}`,
        kind: 'signal',
        title: signal.title || `${formatSignalTypeLabel(signal.signal_type || 'signal')} signal`,
        subtitle: `${formatSignalTypeLabel(signal.signal_type || 'signal')} · ${signal.source || 'unknown'}${ageLabel ? ` · ${ageLabel}` : ''}`,
        score,
      })
    }

    for (const pair of tensions) {
      const isoA = normalizeCountryCode(pair.country_a_iso3 || pair.country_a_name || pair.country_a || '')
      const isoB = normalizeCountryCode(pair.country_b_iso3 || pair.country_b_name || pair.country_b || '')
      if (!isoA || !isoB) continue

      let score = 0
      if (selectedIso && (isoA === selectedIso || isoB === selectedIso)) {
        score = Math.max(score, 95 + Number(pair.tension_score || 0))
      }

      if (hasCenter) {
        const a = countryCentroids[isoA]
        const b = countryCentroids[isoB]
        if (a && b) {
          const midLat = (a.latitude + b.latitude) / 2
          const midLon = normalizeLongitude((a.longitude + b.longitude) / 2)
          const distanceKm = haversineDistanceKm(centerLat, centerLon, midLat, midLon)
          if (distanceKm <= 2400) {
            score = Math.max(score, 70 + Number(pair.tension_score || 0) - Math.min(40, distanceKm / 60))
          }
        }
      }
      if (score <= 0) continue

      rows.push({
        id: `tension:${isoA}:${isoB}`,
        kind: 'tension',
        title: formatCountryPair(isoA, isoB),
        subtitle: `Tension arc · ${Number(pair.tension_score || 0).toFixed(1)} · ${String(pair.trend || 'stable')}`,
        score,
      })
    }

    for (const convergence of convergences) {
      const convergenceIso = normalizeCountryCode(convergence.country || '')
      let score = 0
      if (selectedIso && convergenceIso && convergenceIso === selectedIso) {
        score = Math.max(score, 95 + Number(convergence.urgency_score || 0))
      }
      if (hasCenter && isFiniteCoordinatePair(convergence.latitude, convergence.longitude)) {
        const distanceKm = haversineDistanceKm(centerLat, centerLon, Number(convergence.latitude), Number(convergence.longitude))
        if (distanceKm <= 1500) {
          score = Math.max(score, 75 + Number(convergence.urgency_score || 0) - Math.min(35, distanceKm / 40))
        }
      }
      if (score <= 0) continue

      rows.push({
        id: `convergence:${convergence.grid_key}`,
        kind: 'convergence',
        title: `${formatCountry(convergence.country || convergenceIso || 'Unknown')} convergence`,
        subtitle: `${Number(convergence.signal_count || 0)} signals · urgency ${Math.round(Number(convergence.urgency_score || 0))}`,
        score,
      })
    }

    rows.sort((a, b) => b.score - a.score)
    return rows.slice(0, 8)
  }, [countryCentroids, convergences, flyoutSelection, signals, tensions])

  const signalCountByIso3 = useMemo(() => {
    const out: Record<string, number> = {}
    for (const signal of signals) {
      const pair = pairFromSignal(signal)
      if (pair) {
        out[pair[0]] = (out[pair[0]] || 0) + 1
        out[pair[1]] = (out[pair[1]] || 0) + 1
        continue
      }
      const iso3 = normalizeCountryCode(signal.country)
      if (iso3) {
        out[iso3] = (out[iso3] || 0) + 1
      }
    }
    return out
  }, [signals])

  const tensionScoreByIso3 = useMemo(() => {
    const out: Record<string, number> = {}
    for (const pair of tensions) {
      const isoA = normalizeCountryCode(pair.country_a_iso3 || pair.country_a_name || pair.country_a)
      const isoB = normalizeCountryCode(pair.country_b_iso3 || pair.country_b_name || pair.country_b)
      const score = Number(pair.tension_score || 0)
      if (isoA) out[isoA] = Math.max(out[isoA] || 0, score)
      if (isoB) out[isoB] = Math.max(out[isoB] || 0, score)
    }
    return out
  }, [tensions])

  const instabilityScoreByIso3 = useMemo(() => {
    const out: Record<string, number> = {}
    for (const score of instabilityScores) {
      const iso3 = normalizeCountryCode(score.iso3 || score.country_name || score.country)
      if (!iso3) continue
      out[iso3] = Math.max(out[iso3] || 0, Number(score.score || 0))
    }
    return out
  }, [instabilityScores])

  const countryMetricsByIso3 = useMemo(() => {
    const out: Record<string, CountryMetric> = {}
    const allIso3 = new Set<string>([
      ...Object.keys(signalCountByIso3),
      ...Object.keys(tensionScoreByIso3),
      ...Object.keys(instabilityScoreByIso3),
    ])
    for (const feature of stableCountryGeoData.features || []) {
      const iso3 = normalizeCountryCode(String(feature.id || feature.properties?.id || ''))
      if (iso3) {
        allIso3.add(iso3)
      }
    }

    for (const iso3 of allIso3) {
      const instabilityScore = Number(instabilityScoreByIso3[iso3] || 0)
      const tensionScore = Number(tensionScoreByIso3[iso3] || 0)
      const signalCount = Number(signalCountByIso3[iso3] || 0)
      const instabilityIntensity = clamp01(instabilityScore / 100)
      const tensionIntensity = clamp01(tensionScore / 100)
      const combinedIntensity = Math.max(instabilityIntensity, tensionIntensity)
      // Keep active countries lightly shaded even when source scores are near-zero.
      const displayFloor = signalCount > 0 ? 0.2 : 0.06
      const displayIntensity = Math.max(combinedIntensity, displayFloor)
      out[iso3] = {
        country_name: getCountryName(iso3) || iso3,
        instability_score: Number(instabilityScore.toFixed(2)),
        instability_intensity: Number(instabilityIntensity.toFixed(4)),
        tension_score: Number(tensionScore.toFixed(2)),
        tension_intensity: Number(tensionIntensity.toFixed(4)),
        combined_intensity: Number(combinedIntensity.toFixed(4)),
        display_intensity: Number(displayIntensity.toFixed(4)),
        signal_count: signalCount,
      }
    }
    return out
  }, [signalCountByIso3, tensionScoreByIso3, instabilityScoreByIso3, stableCountryGeoData.features])

  const countryPopupSummaryByIso3 = useMemo(() => {
    const out: Record<string, CountryPopupSummary> = {}

    const ensureCountry = (iso3: string): CountryPopupSummary => {
      if (out[iso3]) return out[iso3]
      const created: CountryPopupSummary = {
        totalSignals: 0,
        criticalSignals: 0,
        riskSignals: 0,
        newsSignals: 0,
        tensionArcCount: 0,
        convergenceCount: 0,
        typeCounts: {},
        sourceCounts: {},
        headlinePreviews: [],
        arcPreviews: [],
      }
      out[iso3] = created
      return created
    }

    for (const signal of signals) {
      const data = signal as unknown as Record<string, unknown>
      const iso3 = normalizeCountryCode(String(
        data.iso3 || data.country_iso3 || data.country_code || data.country_name || data.country || ''
      ))
      if (!iso3) continue

      const entry = ensureCountry(iso3)
      const signalType = String(data.signal_type || 'unknown').trim().toLowerCase()
      const severity = Number(data.severity || 0)
      const source = String(data.source || '').trim()
      const title = String(data.title || '').trim()

      entry.totalSignals += 1
      if (severity >= 0.75) entry.criticalSignals += 1
      if (signalType === 'news') {
        entry.newsSignals += 1
      } else {
        entry.riskSignals += 1
      }
      entry.typeCounts[signalType] = (entry.typeCounts[signalType] || 0) + 1
      if (source) {
        entry.sourceCounts[source] = (entry.sourceCounts[source] || 0) + 1
      }
      if (signalType === 'news' && title && entry.headlinePreviews.length < 3) {
        entry.headlinePreviews.push(truncateText(title, 72))
      }
    }

    for (const convergence of convergences) {
      const data = convergence as unknown as Record<string, unknown>
      const iso3 = normalizeCountryCode(String(
        data.iso3 || data.country_iso3 || data.country_code || data.country_name || data.country || ''
      ))
      if (!iso3) continue
      const entry = ensureCountry(iso3)
      entry.convergenceCount += 1
    }

    for (const pair of tensions) {
      const data = pair as unknown as Record<string, unknown>
      let pairCodes: [string, string] | null = parseCountryPair(String(data.country_pair || data.pair_name || data.pair || ''))
      if (!pairCodes) {
        const left = normalizeCountryCode(String(data.country_a || data.country_a_iso3 || ''))
        const right = normalizeCountryCode(String(data.country_b || data.country_b_iso3 || ''))
        if (left && right) pairCodes = [left, right]
      }
      if (!pairCodes) continue

      const trend = String(data.trend || 'stable')
      const score = Number(data.tension_score || 0)
      const eventCount = Number(data.event_count || 0)
      const pairLabel = truncateText(
        `${pairCodes[0]}-${pairCodes[1]} ${score.toFixed(1)} ${trend}${eventCount > 0 ? ` (${eventCount})` : ''}`,
        58
      )

      for (const iso3 of pairCodes) {
        const entry = ensureCountry(iso3)
        entry.tensionArcCount += 1
        if (entry.arcPreviews.length < 2) {
          entry.arcPreviews.push(pairLabel)
        }
      }
    }

    return out
  }, [convergences, signals, tensions])

  const countriesStyledGeoJSON = useMemo(
    () => withCountryMetrics(stableCountryGeoData, countryMetricsByIso3),
    [stableCountryGeoData, countryMetricsByIso3]
  )

  const tensionArcsGeoJSON = useMemo(
    () => tensionsToGeoJSON(tensions, countryCentroids),
    [tensions, countryCentroids]
  )

  const conflictsGeoJSON = useMemo(
    () => conflictSignalsToGeoJSON(signals, countryCentroids),
    [signals, countryCentroids]
  )

  const openSelection = useCallback((selection: FlyoutSelection) => {
    setFlyoutSelection(selection)
    setFlyoutTab('context')
    setFlyoutExpanded(true)
  }, [])

  const closeSelection = useCallback(() => {
    setFlyoutSelection(null)
    setFlyoutTab('context')
    setFlyoutExpanded(false)
  }, [])

  const toggleSourceVisibility = useCallback((canonicalKey: string) => {
    setSourceVisibilityByKey((prev) => ({ ...prev, [canonicalKey]: !(prev[canonicalKey] ?? true) }))
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const map = new maplibregl.Map({
      container: el,
      style: buildStyle(theme),
      center: [30, 25],
      zoom: 2.2,
      minZoom: 1.5,
      maxZoom: 12,
      attributionControl: { compact: true },
    })

    mapRef.current = map
    mapReadyRef.current = false
    setMapReady(false)
    setMapInitError(null)

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left')

    map.once('load', () => {
      try {
        addDataLayers(map, theme)
        mapReadyRef.current = true
        setMapInitError(null)
        setMapReady(true)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed adding map layers'
        setMapInitError(message)
      }
    })

    map.on('error', (evt: unknown) => {
      const errorEvent = evt as { error?: { message?: string } }
      const message = errorEvent.error?.message || 'Map runtime error'
      if (!mapReadyRef.current) {
        setMapInitError(message)
      }
      console.warn('[WorldMap]', message)
    })

    return () => {
      mapReadyRef.current = false
      setMapReady(false)
      mapRef.current = null
      map.remove()
    }
  }, [theme])

  useEffect(() => {
    const map = mapRef.current
    const el = containerRef.current
    if (!map || !el) return
    const observer = new ResizeObserver(() => map.resize())
    observer.observe(el)
    return () => observer.disconnect()
  }, [mapReady])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'countries', countriesStyledGeoJSON)
  }, [mapReady, countriesStyledGeoJSON])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'signals', geocodedSignalsGeoJSON)
  }, [mapReady, geocodedSignalsGeoJSON])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'convergences', convergencesToGeoJSON(convergences))
  }, [mapReady, convergences])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'hotspots', hotspotsToGeoJSON(hotspots))
  }, [mapReady, hotspots])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'tension-arcs', tensionArcsGeoJSON)
  }, [mapReady, tensionArcsGeoJSON])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'conflicts', conflictsGeoJSON)
  }, [mapReady, conflictsGeoJSON])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    updateSourceData(mapRef.current, 'chokepoints', chokepointsToGeoJSON(chokepoints))
  }, [mapReady, chokepoints])

  useEffect(() => {
    const map = mapRef.current
    if (!mapReady || !map) return
    for (const [toggleKey, layerIds] of Object.entries(STRUCTURAL_LAYER_GROUPS) as Array<[string, readonly string[]]>) {
      const visibility = layerToggles[toggleKey] ? 'visible' : 'none'
      for (const layerId of layerIds) {
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, 'visibility', visibility)
        }
      }
    }
  }, [mapReady, layerToggles])

  type LayerClickEvent = {
    features?: Array<MapGeoJSONFeature & { id?: string | number }>
    lngLat?: {
      lng: number
      lat: number
    }
  }

  const handleSignalClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>

      const signalType = String(props.signal_type || 'unknown')
      const severity = Math.round((Number(props.severity) || 0) * 100)
      const ageHours = Number(props.age_hours || 0)
      const ageLabel = ageHours < 1 ? '<1h ago' : ageHours < 24 ? `${Math.round(ageHours)}h ago` : `${Math.round(ageHours / 24)}d ago`

      let meta: Record<string, unknown> = {}
      try {
        meta = JSON.parse(String(props.metadata_json || '{}'))
      } catch {
        meta = {}
      }

      let metaDetails = ''
      if (signalType === 'conflict') {
        const fatalities = meta.fatalities != null ? `Fatalities: ${meta.fatalities}` : ''
        const eventType = meta.event_type ? String(meta.event_type) : ''
        const subType = meta.sub_event_type ? String(meta.sub_event_type) : ''
        metaDetails = [eventType, subType, fatalities].filter(Boolean).join(' · ')
      } else if (signalType === 'tension') {
        const trend = meta.trend ? `Trend: ${meta.trend}` : ''
        const count = meta.event_count != null ? `${meta.event_count} events` : ''
        metaDetails = [trend, count].filter(Boolean).join(' · ')
      } else if (signalType === 'earthquake') {
        const mag = meta.magnitude != null ? `M${Number(meta.magnitude).toFixed(1)}` : ''
        const depth = meta.depth_km != null ? `${Number(meta.depth_km).toFixed(0)}km depth` : ''
        const tsunami = meta.tsunami ? '⚠ Tsunami warning' : ''
        const alert = meta.alert ? `Alert: ${meta.alert}` : ''
        metaDetails = [mag, depth, tsunami, alert].filter(Boolean).join(' · ')
      } else if (signalType === 'fire') {
        const frp = meta.frp != null ? `FRP: ${Number(meta.frp).toFixed(1)} MW` : ''
        const bright = meta.bright_ti4 != null ? `Brightness: ${Number(meta.bright_ti4).toFixed(1)}K` : ''
        const confidence = meta.confidence ? `Confidence: ${meta.confidence}` : ''
        const daynight = meta.daynight === 'D' ? 'Daytime' : meta.daynight === 'N' ? 'Nighttime' : ''
        metaDetails = [frp, bright, confidence, daynight].filter(Boolean).join(' · ')
      } else if (signalType === 'military') {
        const aircraft = meta.aircraft_type ? String(meta.aircraft_type) : ''
        const actType = meta.activity_type ? String(meta.activity_type) : ''
        const region = meta.region ? String(meta.region) : ''
        metaDetails = [aircraft, actType, region].filter(Boolean).join(' · ')
      } else if (signalType === 'convergence') {
        const types = Array.isArray(meta.signal_types) ? meta.signal_types.join(', ') : ''
        const count = meta.signal_count != null ? `${meta.signal_count} signals` : ''
        metaDetails = [types, count].filter(Boolean).join(' · ')
      } else if (signalType === 'news') {
        const url = meta.url ? String(meta.url).replace(/^https?:\/\//, '').split('/')[0] : ''
        const category = meta.category ? String(meta.category) : ''
        metaDetails = [category, url].filter(Boolean).join(' · ')
      }

      const bodyParts = [
        `${signalType}${props.activity_type ? `/${String(props.activity_type)}` : ''} · ${severity}% severity`,
        metaDetails,
        ageLabel,
      ].filter(Boolean)

      const relatedMarketCount = Number(props.related_market_count || 0)
      const relatedMarketsRaw = String(props.related_market_ids || '').trim()
      const marketRelevance = Number(props.market_relevance_score || 0)
      if (relatedMarketCount > 0) {
        const relatedMarkets = relatedMarketsRaw
          ? relatedMarketsRaw.split(',').map((value) => value.trim()).filter(Boolean)
          : []
        const relatedMarketsLabel = `${relatedMarketCount} related market${relatedMarketCount === 1 ? '' : 's'}`
        const relevanceLabel = marketRelevance > 0 ? `relevance ${Math.round(marketRelevance * 100)}%` : ''
        const previewLabel = relatedMarkets.length > 0 ? `top: ${relatedMarkets.slice(0, 3).join(', ')}` : ''
        bodyParts.push([relatedMarketsLabel, relevanceLabel, previewLabel].filter(Boolean).join(' · '))
      }

      const signalIso3 = normalizeCountryCode(String(
        props.country_iso3
        || props.country_code
        || props.country_name
        || props.country
        || ''
      )) || undefined
      const signalLat = Number(props.latitude)
      const signalLon = Number(props.longitude)
      const hasSignalCoordinates = Number.isFinite(signalLat) && Number.isFinite(signalLon)

      openSelection({
        category: 'Signal',
        title: String(props.title || 'Signal'),
        subtitle: `${props.country_name ? `${String(props.country_name)} · ` : ''}${String(props.source || '')}`,
        body: bodyParts.join(' · '),
        iso3: signalIso3,
        countryName: props.country_name ? String(props.country_name) : undefined,
        lat: hasSignalCoordinates ? signalLat : undefined,
        lon: hasSignalCoordinates ? signalLon : undefined,
        source: String(props.source || ''),
        signalType,
      })
    },
    [openSelection]
  )

  const handleConvergenceClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const convergenceIso3 = normalizeCountryCode(String(
        props.country_iso3
        || props.country_code
        || props.country_name
        || props.country
        || ''
      )) || undefined
      const convergenceLat = Number(props.latitude)
      const convergenceLon = Number(props.longitude)
      const hasConvergenceCoordinates = Number.isFinite(convergenceLat) && Number.isFinite(convergenceLon)
      openSelection({
        category: 'Convergence',
        title: 'Convergence Zone',
        subtitle: `${props.country_name ? `${String(props.country_name)} · ` : ''}${String(props.signal_count || 0)} signals`,
        body: `Urgency: ${Math.round(Number(props.urgency_score) || 0)} · Types: ${String(props.signal_types || 'unknown')}`,
        iso3: convergenceIso3,
        countryName: props.country_name ? String(props.country_name) : undefined,
        lat: hasConvergenceCoordinates ? convergenceLat : undefined,
        lon: hasConvergenceCoordinates ? convergenceLon : undefined,
      })
    },
    [openSelection]
  )

  const handleHotspotClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const latMin = Number(props.lat_min)
      const latMax = Number(props.lat_max)
      const lonMin = Number(props.lon_min)
      const lonMax = Number(props.lon_max)
      const hasBounds =
        Number.isFinite(latMin)
        && Number.isFinite(latMax)
        && Number.isFinite(lonMin)
        && Number.isFinite(lonMax)

      const signalsInZone = hasBounds
        ? geocodedSignalPoints.filter((point) => {
          const lat = Number(point.lat)
          const lon = Number(point.lon)
          return lat >= latMin && lat <= latMax && lon >= lonMin && lon <= lonMax
        }).length
        : 0

      const convergencesInZone = hasBounds
        ? convergences.filter((zone) => (
          zone.latitude >= latMin
          && zone.latitude <= latMax
          && zone.longitude >= lonMin
          && zone.longitude <= lonMax
        )).length
        : 0
      const eventCount = Number(props.event_count || 0)
      const lastDetectedAt = String(props.last_detected_at || '')
      const activityTypes = String(props.activity_types || '')

      const body = hasBounds
        ? `Bounds: ${latMin.toFixed(1)}-${latMax.toFixed(1)} lat, ${lonMin.toFixed(1)}-${lonMax.toFixed(1)} lon · Events: ${eventCount || signalsInZone} · Signals: ${signalsInZone} · Convergences: ${convergencesInZone}${activityTypes ? ` · Types: ${activityTypes}` : ''}${lastDetectedAt ? ` · Last: ${new Date(lastDetectedAt).toLocaleTimeString()}` : ''}`
        : 'No bounding data available for this zone.'
      const zoneLat = hasBounds ? (latMin + latMax) / 2 : event.lngLat?.lat
      const zoneLon = hasBounds ? normalizeLongitude((lonMin + lonMax) / 2) : event.lngLat?.lng

      openSelection({
        category: 'Hotspot',
        title: String(props.name || 'Hotspot'),
        subtitle: 'Military monitoring hotspot',
        body,
        lat: Number.isFinite(Number(zoneLat)) ? Number(zoneLat) : undefined,
        lon: Number.isFinite(Number(zoneLon)) ? Number(zoneLon) : undefined,
      })
    },
    [convergences, geocodedSignalPoints, openSelection]
  )

  const handleChokepointClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const risk = Number(props.risk_score || 0)
      const nearbySignals = Number(props.nearby_signal_count || 0)
      const dailyTransit = Number(props.daily_transit_total || 0)
      const dailyCapacity = Number(props.daily_capacity_estimate || 0)
      const source = String(props.source || '')
      const chokepointSource = String(props.chokepoint_source || '')
      const dailyMetricsDate = String(props.daily_metrics_date || props.daily_dataset_updated_at || '')
      const lastUpdated = String(props.last_updated || '')
      const chokepointLat = Number(props.latitude)
      const chokepointLon = Number(props.longitude)
      const hasChokepointCoordinates = Number.isFinite(chokepointLat) && Number.isFinite(chokepointLon)
      openSelection({
        category: 'Chokepoint',
        title: String(props.name || 'Chokepoint'),
        subtitle: `Global trade chokepoint · Risk ${risk.toFixed(1)}`,
        body: `Nearby signals: ${nearbySignals}${dailyTransit > 0 ? ` · Daily transit: ${dailyTransit}` : ''}${dailyCapacity > 0 ? ` · Capacity: ${dailyCapacity.toLocaleString()}` : ''}${chokepointSource ? ` · Base source: ${chokepointSource}` : ''}${source ? ` · Risk source: ${source}` : ''}${dailyMetricsDate ? ` · Daily feed: ${new Date(dailyMetricsDate).toLocaleDateString()}` : ''}${lastUpdated ? ` · Updated: ${new Date(lastUpdated).toLocaleTimeString()}` : ''}`,
        lat: hasChokepointCoordinates ? chokepointLat : undefined,
        lon: hasChokepointCoordinates ? chokepointLon : undefined,
        source: source || undefined,
      })
    },
    [openSelection]
  )

  const handleCountryHover = useCallback(
    (event: LayerClickEvent & { point?: { x: number; y: number } }) => {
      if (!event.features?.length || !containerRef.current) {
        setHoverTooltip(null)
        return
      }
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const iso3 = normalizeCountryCode(String((feature as any).id || props.id || ''))
      const metrics = iso3 ? countryMetricsByIso3[iso3] : undefined
      if (!metrics && !iso3) {
        setHoverTooltip(null)
        return
      }
      const point = (event as any).point as { x: number; y: number } | undefined
      if (!point) {
        setHoverTooltip(null)
        return
      }
      setHoverTooltip({
        x: point.x + 14,
        y: point.y - 14,
        name: metrics?.country_name || formatCountry(iso3 || ''),
        instability: metrics?.instability_score || 0,
        tension: metrics?.tension_score || 0,
      })
    },
    [countryMetricsByIso3]
  )

  const handleCountryHoverLeave = useCallback(() => {
    setHoverTooltip(null)
  }, [])

  const handleCountryClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const iso3 = normalizeCountryCode(String(feature.id || props.id || ''))
      if (!iso3) return

      const metrics = countryMetricsByIso3[iso3] || {
        country_name: formatCountry(iso3),
        instability_score: 0,
        instability_intensity: 0,
        tension_score: 0,
        tension_intensity: 0,
        combined_intensity: 0,
        display_intensity: 0.06,
        signal_count: 0,
      }
      const countryCenter = countryCentroids[iso3]
      const popupSummary = countryPopupSummaryByIso3[iso3]
      const trackedSignals = popupSummary?.totalSignals ?? metrics.signal_count
      const uniqueSources = popupSummary ? Object.keys(popupSummary.sourceCounts).length : 0
      const summaryChips = popupSummary
        ? [
          popupSummary.riskSignals > 0 ? `Risk ${popupSummary.riskSignals}` : '',
          popupSummary.newsSignals > 0 ? `News ${popupSummary.newsSignals}` : '',
          popupSummary.criticalSignals > 0 ? `Critical ${popupSummary.criticalSignals}` : '',
          popupSummary.tensionArcCount > 0 ? `Arcs ${popupSummary.tensionArcCount}` : '',
          popupSummary.convergenceCount > 0 ? `Convergences ${popupSummary.convergenceCount}` : '',
        ].filter(Boolean)
        : []

      const topSignalMix = popupSummary
        ? summarizeTopCounts(popupSummary.typeCounts, 3, formatSignalTypeLabel)
        : ''
      const topSources = popupSummary
        ? summarizeTopCounts(popupSummary.sourceCounts, 2)
        : ''
      const arcContext = popupSummary?.arcPreviews.length
        ? popupSummary.arcPreviews.join(' | ')
        : ''
      const newsHighlights = popupSummary?.headlinePreviews.length
        ? popupSummary.headlinePreviews.join(' | ')
        : ''

      const bodyParts = [
        ...summaryChips,
        `Instability: ${metrics.instability_score.toFixed(1)}`,
        `Tension: ${metrics.tension_score.toFixed(1)}`,
        `Signals: ${trackedSignals}`,
        uniqueSources > 0 ? `Sources: ${uniqueSources}` : '',
        topSignalMix ? `Signal mix: ${topSignalMix}` : '',
        topSources ? `Top sources: ${topSources}` : '',
        arcContext ? `Bilateral context: ${arcContext}` : '',
        newsHighlights ? `News highlights: ${newsHighlights}` : '',
      ].filter(Boolean)

      openSelection({
        category: 'Country',
        title: metrics.country_name || formatCountry(iso3),
        subtitle: `ISO3 ${iso3} · ${trackedSignals} tracked${uniqueSources > 0 ? ` · ${uniqueSources} sources` : ''}`,
        body: bodyParts.join(' · '),
        iso3,
        countryName: metrics.country_name || formatCountry(iso3),
        lat: countryCenter?.latitude,
        lon: countryCenter?.longitude,
      })
    },
    [countryCentroids, countryMetricsByIso3, countryPopupSummaryByIso3, openSelection]
  )

  const handleTensionArcClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const score = Number(props.tension_score || 0)
      const trend = String(props.trend || 'stable')
      const eventCount = Number(props.event_count || 0)
      const eventTypes = String(props.top_event_types || '')
      const lastUpdated = String(props.last_updated || '')
      const pairCodes = parseCountryPair(String(props.country_pair || props.pair_name || props.pair || ''))
      openSelection({
        category: 'Tension Arc',
        title: String(props.pair_name || 'Tension Arc'),
        subtitle: `Score ${score.toFixed(1)} · ${trend}`,
        body: `Events: ${eventCount}${eventTypes ? ` · Types: ${eventTypes}` : ''}${lastUpdated ? ` · Updated: ${new Date(lastUpdated).toLocaleTimeString()}` : ''}`,
        iso3: pairCodes?.[0] || undefined,
        lat: event.lngLat?.lat,
        lon: event.lngLat?.lng,
      })
    },
    [openSelection]
  )

  const handleConflictClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const conflictIso3 = normalizeCountryCode(String(
        props.country_iso3
        || props.country_code
        || props.country_name
        || props.country
        || ''
      )) || undefined
      openSelection({
        category: 'Conflict',
        title: String(props.title || 'Conflict Signal'),
        subtitle: `${String(props.country_name || 'Unknown')} · ${String(props.source || 'unknown')}`,
        body: `Severity: ${Math.round((Number(props.severity) || 0) * 100)}%`,
        iso3: conflictIso3,
        countryName: props.country_name ? String(props.country_name) : undefined,
        lat: Number.isFinite(Number(props.latitude)) ? Number(props.latitude) : undefined,
        lon: Number.isFinite(Number(props.longitude)) ? Number(props.longitude) : undefined,
        source: String(props.source || ''),
        signalType: 'conflict',
      })
    },
    [openSelection]
  )

  const handleMapBackgroundClick = useCallback((event: LayerClickEvent & { point?: { x: number; y: number } }) => {
    const map = mapRef.current
    if (!map || !event.point) return
    const layers = CLICKABLE_LAYERS.filter((layerId) => Boolean(map.getLayer(layerId)))
    if (!layers.length) {
      closeSelection()
      return
    }
    const features = map.queryRenderedFeatures(event.point, { layers: [...layers] as string[] })
    if (!features?.length) {
      closeSelection()
    }
  }, [closeSelection])

  useEffect(() => {
    const map = mapRef.current
    if (!mapReady || !map) return

    const cursorOn = () => {
      map.getCanvas().style.cursor = 'pointer'
    }
    const cursorOff = () => {
      map.getCanvas().style.cursor = ''
    }

    map.on('click', 'countries-fill-intensity', handleCountryClick)
    map.on('click', 'countries-border-tension', handleCountryClick)
    map.on('click', 'countries-focus-fill', handleCountryClick)
    map.on('click', 'countries-focus-outline', handleCountryClick)
    map.on('click', 'tension-arcs-line', handleTensionArcClick)
    map.on('click', 'conflicts-dot', handleConflictClick)
    map.on('click', 'signals-dot', handleSignalClick)
    map.on('click', 'signals-glow', handleSignalClick)
    map.on('click', 'signals-military-flight-icon', handleSignalClick)
    map.on('click', 'signals-military-vessel-icon', handleSignalClick)
    map.on('click', 'convergences-ring', handleConvergenceClick)
    map.on('click', 'convergences-fill', handleConvergenceClick)
    map.on('click', 'hotspots-fill', handleHotspotClick)
    map.on('click', 'hotspots-outline', handleHotspotClick)
    map.on('click', 'chokepoints-icon', handleChokepointClick)
    map.on('click', handleMapBackgroundClick)

    map.on('mousemove', 'countries-fill-intensity', handleCountryHover)
    map.on('mouseleave', 'countries-fill-intensity', handleCountryHoverLeave)

    for (const layerId of CLICKABLE_LAYERS) {
      map.on('mouseenter', layerId, cursorOn)
      map.on('mouseleave', layerId, cursorOff)
    }

    return () => {
      map.off('click', 'countries-fill-intensity', handleCountryClick)
      map.off('click', 'countries-border-tension', handleCountryClick)
      map.off('click', 'countries-focus-fill', handleCountryClick)
      map.off('click', 'countries-focus-outline', handleCountryClick)
      map.off('click', 'tension-arcs-line', handleTensionArcClick)
      map.off('click', 'conflicts-dot', handleConflictClick)
      map.off('click', 'signals-dot', handleSignalClick)
      map.off('click', 'signals-glow', handleSignalClick)
      map.off('click', 'signals-military-flight-icon', handleSignalClick)
      map.off('click', 'signals-military-vessel-icon', handleSignalClick)
      map.off('click', 'convergences-ring', handleConvergenceClick)
      map.off('click', 'convergences-fill', handleConvergenceClick)
      map.off('click', 'hotspots-fill', handleHotspotClick)
      map.off('click', 'hotspots-outline', handleHotspotClick)
      map.off('click', 'chokepoints-icon', handleChokepointClick)
      map.off('click', handleMapBackgroundClick)
      map.off('mousemove', 'countries-fill-intensity', handleCountryHover)
      map.off('mouseleave', 'countries-fill-intensity', handleCountryHoverLeave)
      for (const layerId of CLICKABLE_LAYERS) {
        map.off('mouseenter', layerId, cursorOn)
        map.off('mouseleave', layerId, cursorOff)
      }
    }
  }, [
    mapReady,
    handleCountryClick,
    handleCountryHover,
    handleCountryHoverLeave,
    handleTensionArcClick,
    handleConflictClick,
    handleSignalClick,
    handleConvergenceClick,
    handleHotspotClick,
    handleChokepointClick,
    handleMapBackgroundClick,
  ])

  const loading =
    signalsLoading
    || convergenceLoading
    || regionsLoading
    || tensionsLoading
    || instabilityLoading
    || countriesLoading
  const coreError = Boolean(mapInitError)

  return (
    <div className="absolute inset-0 bg-background">
      <div ref={containerRef} className="w-full h-full" />

      <MapRightDock
        selection={flyoutSelection}
        relatedEvents={relatedEvents}
        tab={flyoutTab}
        onTabChange={setFlyoutTab}
        expanded={flyoutExpanded}
        onExpandedChange={setFlyoutExpanded}
        onCloseSelection={closeSelection}
        layerItems={layerDockItems}
        toggles={layerToggles}
        onToggle={(key) => {
          setLayerToggles((prev) => ({ ...prev, [key]: !prev[key] }))
          // Dynamic toggles (dsrc_*) also control source visibility
          if (key.startsWith('dsrc_')) {
            const slug = key.slice('dsrc_'.length)
            const item = sourceDockItems.find((s) => s.slug === slug)
            if (item) toggleSourceVisibility(item.canonicalKey)
          }
        }}
        signalCount={signals.length}
        signalTotal={Number(stableSignalsData.total || signals.length)}
        criticalCount={signalStatsExtra.criticalCount}
        geocodedSignalCount={geocodedSignalCount}
        convergenceCount={convergences.length}
        hotspotCount={hotspots.length}
        byType={signalStatsExtra.byType}
        colors={colors}
        sourceItems={sourceDockItems}
        sourceVisibility={sourceVisibilityByKey}
        onToggleSource={toggleSourceVisibility}
        sourceError={worldSourceStatusData?.errors?.[0] || null}
      />

      {coreError ? (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20">
          <div className="px-3 py-2 rounded-md border border-red-500/30 bg-red-500/10 text-xs text-red-500">
            Map data unavailable. Check events status.
          </div>
        </div>
      ) : null}

      {!loading && !coreError && signals.length === 0 ? (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20">
          <div className="px-3 py-2 rounded-md border border-border bg-background/90 text-xs text-muted-foreground">
            No active event signals detected yet.
          </div>
        </div>
      ) : null}

      {hoverTooltip ? (
        <div
          className="pointer-events-none absolute z-20 rounded border border-border bg-background/95 backdrop-blur-sm px-2 py-1 text-[10px] space-y-0.5 shadow-md"
          style={{ left: hoverTooltip.x, top: hoverTooltip.y }}
        >
          <div className="font-semibold text-foreground text-[11px]">{hoverTooltip.name}</div>
          {hoverTooltip.instability > 0 ? (
            <div className="text-muted-foreground font-mono">
              Instability: <span className="text-orange-400">{hoverTooltip.instability.toFixed(1)}</span>
            </div>
          ) : null}
          {hoverTooltip.tension > 0 ? (
            <div className="text-muted-foreground font-mono">
              Tension: <span className="text-red-400">{hoverTooltip.tension.toFixed(1)}</span>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
