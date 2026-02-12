import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAtomValue } from 'jotai'
import { createRoot } from 'react-dom/client'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import {
  getConvergenceZones,
  getInstabilityScores,
  getTensionPairs,
  getWorldRegions,
  getWorldSignals,
  type ConvergenceZone,
  type TensionPair,
  type WorldRegionChokepoint,
  type WorldRegionHotspot,
  type WorldSignal,
} from '../services/worldIntelligenceApi'
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

type LayerToggles = {
  countryIntensity: boolean
  tensionBorders: boolean
  tensionArcs: boolean
  countryBoundaries: boolean
  conflictZones: boolean
  signals: boolean
  convergences: boolean
  hotspots: boolean
  chokepoints: boolean
}

type CountryMetric = {
  country_name: string
  instability_score: number
  instability_intensity: number
  tension_score: number
  tension_intensity: number
  combined_intensity: number
  signal_count: number
}

const SIGNAL_COLORS_DARK: SignalPalette = {
  conflict: '#f87171',
  tension: '#fb923c',
  instability: '#facc15',
  convergence: '#c084fc',
  anomaly: '#22d3ee',
  military: '#60a5fa',
  infrastructure: '#34d399',
}

const SIGNAL_COLORS_LIGHT: SignalPalette = {
  conflict: '#dc2626',
  tension: '#c2410c',
  instability: '#a16207',
  convergence: '#7c3aed',
  anomaly: '#0e7490',
  military: '#2563eb',
  infrastructure: '#15803d',
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
}

const LAYER_GROUPS: Record<keyof LayerToggles, readonly string[]> = {
  countryIntensity: ['countries-fill-intensity'],
  tensionBorders: ['countries-border-tension'],
  tensionArcs: ['tension-arcs-glow', 'tension-arcs-line'],
  countryBoundaries: ['countries-focus-fill', 'countries-focus-outline'],
  conflictZones: ['conflicts-heat', 'conflicts-dot'],
  signals: ['signals-dot', 'signals-glow'],
  convergences: ['convergences-fill', 'convergences-ring'],
  hotspots: ['hotspots-fill', 'hotspots-outline'],
  chokepoints: ['chokepoints-icon'],
}

const COUNTRY_BOUNDARY_URL = `${import.meta.env.BASE_URL}data/world_countries.geojson`
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
  centroids: Record<string, CountryCentroid>
): GeoFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: signals
      .map((signal) => {
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
            color: palette[signal.signal_type] || '#64748b',
            geocode_mode: geocodeMode,
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

function addDataLayers(map: any, theme: 'dark' | 'light') {
  const hotspotFillColor = theme === 'light' ? '#2563eb' : '#60a5fa'
  const countryBorderColor = theme === 'light' ? '#94a3b8' : '#475569'

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
        ['coalesce', ['get', 'combined_intensity'], 0],
        0, theme === 'light' ? '#f8fafc' : '#0f172a',
        0.2, theme === 'light' ? '#fde68a' : '#854d0e',
        0.45, theme === 'light' ? '#fb923c' : '#c2410c',
        0.7, theme === 'light' ? '#ef4444' : '#dc2626',
        1, theme === 'light' ? '#b91c1c' : '#7f1d1d',
      ],
      'fill-opacity': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'combined_intensity'], 0],
        0, 0.02,
        0.2, 0.12,
        0.5, 0.2,
        1, 0.3,
      ],
    },
  })
  map.addLayer({
    id: 'countries-border-base',
    type: 'line',
    source: 'countries',
    paint: {
      'line-color': countryBorderColor,
      'line-width': 0.8,
      'line-opacity': theme === 'light' ? 0.4 : 0.55,
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
          ['>=', ['coalesce', ['get', 'signal_count'], 0], 6],
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
          ['>=', ['coalesce', ['get', 'signal_count'], 0], 6],
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
          ['>=', ['coalesce', ['get', 'signal_count'], 0], 6],
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
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'tension_score'], 0],
        0, theme === 'light' ? '#facc15' : '#f59e0b',
        40, '#f97316',
        70, '#ef4444',
        100, '#991b1b',
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
  map.addLayer({
    id: 'signals-glow',
    type: 'circle',
    source: 'signals',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'severity'], 0, 10, 0.5, 16, 1, 24],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.2,
      'circle-blur': 1,
    },
  })
  map.addLayer({
    id: 'signals-dot',
    type: 'circle',
    source: 'signals',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'severity'], 0, 4, 0.5, 6, 1, 9],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.95,
      'circle-stroke-color': theme === 'light' ? '#ffffff' : '#020617',
      'circle-stroke-width': 1,
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
}

function updateSourceData(map: any, sourceId: string, data: unknown) {
  const source = map.getSource(sourceId)
  source?.setData(data as any)
}

function PopupCard({ title, subtitle, body }: { title: string; subtitle?: string; body?: string }) {
  return (
    <div className="text-[11px] leading-5 max-w-[260px]">
      <div className="font-semibold text-foreground">{title}</div>
      {subtitle ? <div className="text-muted-foreground mt-0.5">{subtitle}</div> : null}
      {body ? <div className="text-foreground/90 mt-1">{body}</div> : null}
    </div>
  )
}

function MapLegend({ colors }: { colors: SignalPalette }) {
  const [collapsed, setCollapsed] = useState(true)

  return (
    <div className="text-[10px] space-y-1 min-w-[176px]">
      <div className="flex items-center justify-between gap-2">
        <div className="font-semibold text-[11px] text-foreground">Legend</div>
        <button
          type="button"
          onClick={() => setCollapsed((prev) => !prev)}
          className="text-[10px] leading-none rounded border border-border px-1.5 py-0.5 text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
        >
          {collapsed ? 'Show' : 'Hide'}
        </button>
      </div>

      {!collapsed ? (
        <>
          {Object.entries(colors).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
              <span className="text-muted-foreground capitalize">{type.replace('_', ' ')}</span>
            </div>
          ))}
          <div className="border-t border-border pt-1 mt-1.5 space-y-1.5">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 shrink-0 border border-red-500/70 bg-red-500/20" />
              <span className="text-muted-foreground">Country intensity</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 shrink-0 bg-orange-500" />
              <span className="text-muted-foreground">Tension border</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 shrink-0 bg-red-500" />
              <span className="text-muted-foreground">Tension arc</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 shrink-0 border border-sky-500/70 bg-sky-500/10" />
              <span className="text-muted-foreground">Country boundary focus</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 shrink-0 rounded-full bg-red-500/80" />
              <span className="text-muted-foreground">Conflict zone heat</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full shrink-0 border-2 border-purple-500 bg-transparent" />
              <span className="text-muted-foreground">Convergence zone</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 shrink-0 border border-blue-500/60 bg-blue-500/10" />
              <span className="text-muted-foreground">Live military hotspot</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 shrink-0 rounded-full bg-emerald-500" />
              <span className="text-muted-foreground">Chokepoint risk</span>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}

function MapStats({
  signalCount,
  geocodedSignalCount,
  convergenceCount,
  hotspotCount,
}: {
  signalCount: number
  geocodedSignalCount: number
  convergenceCount: number
  hotspotCount: number
}) {
  return (
    <div className="absolute bottom-3 right-3 bg-background/90 backdrop-blur-sm border border-border rounded-lg p-2.5 text-[10px] z-10 space-y-1">
      <div className="font-mono">
        <span className="text-muted-foreground">Signals:</span>{' '}
        <span className="text-foreground font-bold">{signalCount}</span>
      </div>
      <div className="font-mono">
        <span className="text-muted-foreground">Geocoded:</span>{' '}
        <span className="text-emerald-500 font-bold">{geocodedSignalCount}</span>
      </div>
      <div className="font-mono">
        <span className="text-muted-foreground">Convergences:</span>{' '}
        <span className="text-purple-500 font-bold">{convergenceCount}</span>
      </div>
      <div className="font-mono">
        <span className="text-muted-foreground">Hotspots:</span>{' '}
        <span className="text-blue-500 font-bold">{hotspotCount}</span>
      </div>
    </div>
  )
}

function LayerControls({
  toggles,
  onToggle,
}: {
  toggles: LayerToggles
  onToggle: (key: keyof LayerToggles) => void
}) {
  const [collapsed, setCollapsed] = useState(true)
  const items: Array<{ key: keyof LayerToggles; label: string }> = [
    { key: 'countryIntensity', label: 'Country intensity' },
    { key: 'tensionBorders', label: 'Tension borders' },
    { key: 'tensionArcs', label: 'Tension arcs' },
    { key: 'countryBoundaries', label: 'Country boundaries' },
    { key: 'conflictZones', label: 'Conflict zones' },
    { key: 'signals', label: 'Signals' },
    { key: 'convergences', label: 'Convergences' },
    { key: 'hotspots', label: 'Hotspots' },
    { key: 'chokepoints', label: 'Chokepoints' },
  ]

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold text-foreground">Map Layers</div>
        <button
          type="button"
          onClick={() => setCollapsed((prev) => !prev)}
          className="text-[10px] leading-none rounded border border-border px-1.5 py-0.5 text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
        >
          {collapsed ? 'Show' : 'Hide'}
        </button>
      </div>
      {!collapsed ? items.map((item) => {
        const enabled = toggles[item.key]
        return (
          <button
            key={item.key}
            type="button"
            onClick={() => onToggle(item.key)}
            className="w-full text-left flex items-center justify-between gap-2 rounded px-1.5 py-1 text-[10px] hover:bg-muted/50 transition-colors"
          >
            <span className="text-muted-foreground">{item.label}</span>
            <span className={enabled ? 'text-emerald-400 font-semibold' : 'text-muted-foreground'}>
              {enabled ? 'ON' : 'OFF'}
            </span>
          </button>
        )
      }) : null}
    </div>
  )
}

function MapControlDock({
  colors,
  toggles,
  onToggle,
}: {
  colors: SignalPalette
  toggles: LayerToggles
  onToggle: (key: keyof LayerToggles) => void
}) {
  return (
    <div className="absolute top-3 right-3 z-10 w-[228px] rounded-lg border border-border bg-background/90 backdrop-blur-sm shadow-sm overflow-hidden">
      <div className="p-2.5">
        <LayerControls toggles={toggles} onToggle={onToggle} />
      </div>
      <div className="border-t border-border" />
      <div className="p-2.5">
        <MapLegend colors={colors} />
      </div>
    </div>
  )
}

function featurePointCoordinates(feature: MapGeoJSONFeature): LngLatTuple | null {
  const geometry = feature.geometry as { type?: string; coordinates?: unknown }
  if (geometry?.type !== 'Point' || !Array.isArray(geometry.coordinates)) return null
  if (geometry.coordinates.length < 2) return null
  const lon = Number(geometry.coordinates[0])
  const lat = Number(geometry.coordinates[1])
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null
  return [lon, lat]
}

function featureLineMidpoint(feature: MapGeoJSONFeature): LngLatTuple | null {
  const geometry = feature.geometry as { type?: string; coordinates?: unknown }
  if (geometry?.type !== 'LineString' || !Array.isArray(geometry.coordinates)) return null
  if (geometry.coordinates.length === 0) return null
  const mid = geometry.coordinates[Math.floor(geometry.coordinates.length / 2)]
  if (!Array.isArray(mid) || mid.length < 2) return null
  const lon = Number(mid[0])
  const lat = Number(mid[1])
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null
  return [normalizeLongitude(lon), lat]
}

export default function WorldMap({ isConnected = true }: { isConnected?: boolean }) {
  const theme = useAtomValue(themeAtom)
  const colors = useMemo(
    () => (theme === 'light' ? SIGNAL_COLORS_LIGHT : SIGNAL_COLORS_DARK),
    [theme]
  )

  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<any>(null)
  const popupRef = useRef<any>(null)
  const mapReadyRef = useRef(false)
  const [mapReady, setMapReady] = useState(false)
  const [mapInitError, setMapInitError] = useState<string | null>(null)
  const [layerToggles, setLayerToggles] = useState<LayerToggles>(DEFAULT_LAYER_TOGGLES)
  const pollingInterval = isConnected ? false : 180000

  const { data: signalsData, isLoading: signalsLoading } = useQuery({
    queryKey: ['world-signals', { limit: 500 }],
    queryFn: () => getWorldSignals({ limit: 500 }),
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

  const signals = stableSignalsData.signals || []
  const convergences = stableConvergenceData.zones || []
  const hotspots = stableRegionsData.hotspots || []
  const chokepoints = stableRegionsData.chokepoints || []
  const tensions = stableTensionsData.tensions || []
  const instabilityScores = stableInstabilityData.scores || []
  const countryCentroids = useMemo(() => buildCountryCentroids(stableCountryGeoData), [stableCountryGeoData])

  const geocodedSignalsGeoJSON = useMemo(
    () => signalsToGeoJSON(signals, colors, countryCentroids),
    [signals, colors, countryCentroids]
  )
  const geocodedSignalPoints = useMemo(
    () => geocodedSignalsGeoJSON.features.map((feature) => ({
      lon: Number(feature.geometry.coordinates[0]),
      lat: Number(feature.geometry.coordinates[1]),
    })),
    [geocodedSignalsGeoJSON]
  )
  const geocodedSignalCount = geocodedSignalsGeoJSON.features.length

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

    for (const iso3 of allIso3) {
      const instabilityScore = Number(instabilityScoreByIso3[iso3] || 0)
      const tensionScore = Number(tensionScoreByIso3[iso3] || 0)
      const instabilityIntensity = clamp01(instabilityScore / 100)
      const tensionIntensity = clamp01(tensionScore / 100)
      out[iso3] = {
        country_name: getCountryName(iso3) || iso3,
        instability_score: Number(instabilityScore.toFixed(2)),
        instability_intensity: Number(instabilityIntensity.toFixed(4)),
        tension_score: Number(tensionScore.toFixed(2)),
        tension_intensity: Number(tensionIntensity.toFixed(4)),
        combined_intensity: Number(Math.max(instabilityIntensity, tensionIntensity).toFixed(4)),
        signal_count: signalCountByIso3[iso3] || 0,
      }
    }
    return out
  }, [signalCountByIso3, tensionScoreByIso3, instabilityScoreByIso3])

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

  const openPopup = useCallback((coords: LngLatTuple, node: ReactNode) => {
    if (!mapRef.current) return
    popupRef.current?.remove()

    const mount = document.createElement('div')
    const root = createRoot(mount)
    root.render(node)

    const popup = new maplibregl.Popup({
      closeButton: true,
      closeOnClick: true,
      maxWidth: '320px',
      className: 'world-map-popup',
    })
      .setLngLat(coords)
      .setDOMContent(mount)
      .addTo(mapRef.current)

    popup.on('close', () => {
      root.unmount()
    })

    popupRef.current = popup
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
      popupRef.current?.remove()
      popupRef.current = null
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
    for (const [toggleKey, layerIds] of Object.entries(LAYER_GROUPS) as Array<[keyof LayerToggles, readonly string[]]>) {
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
      const coords = featurePointCoordinates(feature)
      if (!coords) return
      openPopup(
        coords,
        <PopupCard
          title={String(props.title || 'Signal')}
          subtitle={`${props.country_name ? `${String(props.country_name)} · ` : ''}${String(props.source || '')}`}
          body={`Type: ${String(props.signal_type || 'unknown')} · Severity: ${Math.round((Number(props.severity) || 0) * 100)}% · Geocode: ${String(props.geocode_mode || 'native')}`}
        />
      )
    },
    [openPopup]
  )

  const handleConvergenceClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const coords = featurePointCoordinates(feature)
      if (!coords) return
      openPopup(
        coords,
        <PopupCard
          title="Convergence Zone"
          subtitle={`${props.country_name ? `${String(props.country_name)} · ` : ''}${String(props.signal_count || 0)} signals`}
          body={`Urgency: ${Math.round(Number(props.urgency_score) || 0)} · Types: ${String(props.signal_types || 'unknown')}`}
        />
      )
    },
    [openPopup]
  )

  const handleHotspotClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const coords: LngLatTuple = event.lngLat
        ? [event.lngLat.lng, event.lngLat.lat]
        : [0, 0]
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

      openPopup(
        coords,
        <PopupCard
          title={String(props.name || 'Hotspot')}
          subtitle="Military monitoring hotspot"
          body={body}
        />
      )
    },
    [convergences, geocodedSignalPoints, openPopup]
  )

  const handleChokepointClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const coords = featurePointCoordinates(feature)
      if (!coords) return
      const risk = Number(props.risk_score || 0)
      const nearbySignals = Number(props.nearby_signal_count || 0)
      const dailyTransit = Number(props.daily_transit_total || 0)
      const dailyCapacity = Number(props.daily_capacity_estimate || 0)
      const source = String(props.source || '')
      const chokepointSource = String(props.chokepoint_source || '')
      const dailyMetricsDate = String(props.daily_metrics_date || props.daily_dataset_updated_at || '')
      const lastUpdated = String(props.last_updated || '')
      openPopup(
        coords,
        <PopupCard
          title={String(props.name || 'Chokepoint')}
          subtitle={`Global trade chokepoint · Risk ${risk.toFixed(1)}`}
          body={`Nearby signals: ${nearbySignals}${dailyTransit > 0 ? ` · Daily transit: ${dailyTransit}` : ''}${dailyCapacity > 0 ? ` · Capacity: ${dailyCapacity.toLocaleString()}` : ''}${chokepointSource ? ` · Base source: ${chokepointSource}` : ''}${source ? ` · Risk source: ${source}` : ''}${dailyMetricsDate ? ` · Daily feed: ${new Date(dailyMetricsDate).toLocaleDateString()}` : ''}${lastUpdated ? ` · Updated: ${new Date(lastUpdated).toLocaleTimeString()}` : ''}`}
        />
      )
    },
    [openPopup]
  )

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
        signal_count: 0,
      }
      const center = countryCentroids[iso3]
      const coords: LngLatTuple = event.lngLat
        ? [event.lngLat.lng, event.lngLat.lat]
        : center
          ? [center.longitude, center.latitude]
          : [0, 0]

      openPopup(
        coords,
        <PopupCard
          title={metrics.country_name || formatCountry(iso3)}
          subtitle={`ISO3 ${iso3}`}
          body={`Instability: ${metrics.instability_score.toFixed(1)} · Tension: ${metrics.tension_score.toFixed(1)} · Signals: ${metrics.signal_count}`}
        />
      )
    },
    [countryCentroids, countryMetricsByIso3, openPopup]
  )

  const handleTensionArcClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const coords: LngLatTuple = event.lngLat
        ? [event.lngLat.lng, event.lngLat.lat]
        : (featureLineMidpoint(feature) || [0, 0])
      const score = Number(props.tension_score || 0)
      const trend = String(props.trend || 'stable')
      const eventCount = Number(props.event_count || 0)
      const eventTypes = String(props.top_event_types || '')
      const lastUpdated = String(props.last_updated || '')
      openPopup(
        coords,
        <PopupCard
          title={String(props.pair_name || 'Tension Arc')}
          subtitle={`Score ${score.toFixed(1)} · ${trend}`}
          body={`Events: ${eventCount}${eventTypes ? ` · Types: ${eventTypes}` : ''}${lastUpdated ? ` · Updated: ${new Date(lastUpdated).toLocaleTimeString()}` : ''}`}
        />
      )
    },
    [openPopup]
  )

  const handleConflictClick = useCallback(
    (event: LayerClickEvent) => {
      if (!event.features?.length) return
      const feature = event.features[0]
      const props = (feature.properties || {}) as Record<string, unknown>
      const coords = featurePointCoordinates(feature) || (
        event.lngLat ? [event.lngLat.lng, event.lngLat.lat] : [0, 0]
      )
      openPopup(
        coords,
        <PopupCard
          title={String(props.title || 'Conflict Signal')}
          subtitle={`${String(props.country_name || 'Unknown')} · ${String(props.source || 'unknown')}`}
          body={`Severity: ${Math.round((Number(props.severity) || 0) * 100)}%`}
        />
      )
    },
    [openPopup]
  )

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
    map.on('click', 'convergences-ring', handleConvergenceClick)
    map.on('click', 'convergences-fill', handleConvergenceClick)
    map.on('click', 'hotspots-fill', handleHotspotClick)
    map.on('click', 'hotspots-outline', handleHotspotClick)
    map.on('click', 'chokepoints-icon', handleChokepointClick)

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
      map.off('click', 'convergences-ring', handleConvergenceClick)
      map.off('click', 'convergences-fill', handleConvergenceClick)
      map.off('click', 'hotspots-fill', handleHotspotClick)
      map.off('click', 'hotspots-outline', handleHotspotClick)
      map.off('click', 'chokepoints-icon', handleChokepointClick)
      for (const layerId of CLICKABLE_LAYERS) {
        map.off('mouseenter', layerId, cursorOn)
        map.off('mouseleave', layerId, cursorOff)
      }
    }
  }, [
    mapReady,
    handleCountryClick,
    handleTensionArcClick,
    handleConflictClick,
    handleSignalClick,
    handleConvergenceClick,
    handleHotspotClick,
    handleChokepointClick,
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

      <MapControlDock
        colors={colors}
        toggles={layerToggles}
        onToggle={(key) => {
          setLayerToggles((prev) => ({ ...prev, [key]: !prev[key] }))
        }}
      />
      <MapStats
        signalCount={signals.length}
        geocodedSignalCount={geocodedSignalCount}
        convergenceCount={convergences.length}
        hotspotCount={hotspots.length}
      />

      {coreError ? (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20">
          <div className="px-3 py-2 rounded-md border border-red-500/30 bg-red-500/10 text-xs text-red-500">
            Map data unavailable. Check world intelligence status.
          </div>
        </div>
      ) : null}

      {!loading && !coreError && signals.length === 0 ? (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20">
          <div className="px-3 py-2 rounded-md border border-border bg-background/90 text-xs text-muted-foreground">
            No active world signals detected yet.
          </div>
        </div>
      ) : null}


      <style>{`
        .world-map-popup .maplibregl-popup-content {
          background: hsl(var(--card));
          color: hsl(var(--card-foreground));
          border: 1px solid hsl(var(--border));
          border-radius: 10px;
          box-shadow: 0 10px 28px rgba(0,0,0,0.22);
          padding: 10px 12px;
        }
        .world-map-popup .maplibregl-popup-tip {
          border-top-color: hsl(var(--card));
        }
        .world-map-popup .maplibregl-popup-close-button {
          color: hsl(var(--muted-foreground));
          font-size: 16px;
          padding: 2px 6px;
        }
        .world-map-popup .maplibregl-popup-close-button:hover {
          color: hsl(var(--foreground));
          background: transparent;
        }
      `}</style>
    </div>
  )
}
