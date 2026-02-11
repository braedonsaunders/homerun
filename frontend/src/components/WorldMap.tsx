import { useEffect, useRef, useCallback, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import {
  getWorldSignals,
  getConvergenceZones,
  getMilitaryActivity,
  type WorldSignal,
  type ConvergenceZone,
} from '../services/worldIntelligenceApi'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Inline dark style so the map works without external style JSON fetches.
// Uses OSM raster tiles which are broadly accessible.
const DARK_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  name: 'dark-base',
  sources: {
    'osm-tiles': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
      ],
      tileSize: 256,
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
      maxzoom: 18,
    },
  },
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#0a0e17' },
    },
    {
      id: 'osm-tiles',
      type: 'raster',
      source: 'osm-tiles',
      paint: { 'raster-opacity': 0.85 },
    },
  ],
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
}

// Military hotspot bounding boxes (mirrors backend HOTSPOT_BBOXES)
const HOTSPOT_BBOXES: Record<string, [number, number, number, number]> = {
  black_sea: [41.0, 47.0, 27.5, 42.0],
  eastern_med: [31.0, 37.0, 24.0, 36.0],
  taiwan_strait: [22.0, 26.0, 117.0, 122.0],
  korean_dmz: [37.0, 39.0, 125.0, 130.0],
  persian_gulf: [24.0, 30.0, 48.0, 57.0],
  south_china_sea: [5.0, 22.0, 105.0, 121.0],
  baltics: [54.0, 60.0, 20.0, 28.0],
}

// Infrastructure chokepoints for static markers
const CHOKEPOINTS: Array<{ name: string; lat: number; lon: number }> = [
  { name: 'Suez Canal', lat: 30.46, lon: 32.34 },
  { name: 'Strait of Hormuz', lat: 26.57, lon: 56.25 },
  { name: 'Malacca Strait', lat: 2.5, lon: 101.8 },
  { name: 'Panama Canal', lat: 9.08, lon: -79.68 },
  { name: 'Bosphorus', lat: 41.12, lon: 29.05 },
]

// Signal type -> color mapping (matches panel badges)
const SIGNAL_COLORS: Record<string, string> = {
  conflict: '#f87171',    // red-400
  tension: '#fb923c',     // orange-400
  instability: '#facc15', // yellow-400
  convergence: '#c084fc', // purple-400
  anomaly: '#22d3ee',     // cyan-400
  military: '#60a5fa',    // blue-400
  infrastructure: '#34d399', // emerald-400
}

// ---------------------------------------------------------------------------
// GeoJSON builders
// ---------------------------------------------------------------------------

function signalsToGeoJSON(signals: WorldSignal[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: signals
      .filter((s) => s.latitude != null && s.longitude != null)
      .map((s) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Point' as const,
          coordinates: [s.longitude!, s.latitude!],
        },
        properties: {
          signal_id: s.signal_id,
          signal_type: s.signal_type,
          severity: s.severity,
          title: s.title,
          country: s.country || '',
          source: s.source,
          color: SIGNAL_COLORS[s.signal_type] || '#94a3b8',
        },
      })),
  }
}

function convergencesToGeoJSON(zones: ConvergenceZone[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: zones.map((z) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Point' as const,
        coordinates: [z.longitude, z.latitude],
      },
      properties: {
        grid_key: z.grid_key,
        urgency_score: z.urgency_score,
        signal_count: z.signal_count,
        signal_types: z.signal_types.join(', '),
        country: z.country || '',
      },
    })),
  }
}

function hotspotsToGeoJSON(): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: Object.entries(HOTSPOT_BBOXES).map(([name, [latMin, latMax, lonMin, lonMax]]) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Polygon' as const,
        coordinates: [
          [
            [lonMin, latMin],
            [lonMax, latMin],
            [lonMax, latMax],
            [lonMin, latMax],
            [lonMin, latMin],
          ],
        ],
      },
      properties: {
        name: name.replace(/_/g, ' '),
      },
    })),
  }
}

function chokepointsToGeoJSON(): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: CHOKEPOINTS.map((cp) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Point' as const,
        coordinates: [cp.lon, cp.lat],
      },
      properties: {
        name: cp.name,
      },
    })),
  }
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function MapLegend() {
  return (
    <div className="absolute bottom-3 left-3 bg-background/90 backdrop-blur-sm border border-border rounded-lg p-2.5 text-[10px] space-y-1 z-10 pointer-events-auto">
      <div className="font-semibold text-[11px] text-foreground mb-1.5">Signal Types</div>
      {Object.entries(SIGNAL_COLORS).map(([type, color]) => (
        <div key={type} className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
          <span className="text-muted-foreground capitalize">{type}</span>
        </div>
      ))}
      <div className="border-t border-border pt-1 mt-1.5">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full shrink-0 border-2 border-purple-400 bg-transparent" />
          <span className="text-muted-foreground">Convergence zone</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 shrink-0 border border-blue-400/60 bg-blue-400/10" />
          <span className="text-muted-foreground">Military hotspot</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 shrink-0 rotate-45 bg-emerald-400" />
          <span className="text-muted-foreground">Chokepoint</span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Signal count overlay
// ---------------------------------------------------------------------------

function MapStats({ signalCount, convergenceCount, surgeRegions }: { signalCount: number; convergenceCount: number; surgeRegions: string[] }) {
  return (
    <div className="absolute top-3 right-3 bg-background/90 backdrop-blur-sm border border-border rounded-lg p-2.5 text-[10px] z-10 space-y-1 pointer-events-auto">
      <div className="font-mono">
        <span className="text-muted-foreground">Signals:</span>{' '}
        <span className="text-foreground font-bold">{signalCount}</span>
      </div>
      <div className="font-mono">
        <span className="text-muted-foreground">Convergences:</span>{' '}
        <span className="text-purple-400 font-bold">{convergenceCount}</span>
      </div>
      {surgeRegions.length > 0 && (
        <div className="font-mono">
          <span className="text-red-400 font-bold">Surge:</span>{' '}
          <span className="text-red-400">{surgeRegions.join(', ')}</span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers to add data layers once style is loaded
// ---------------------------------------------------------------------------

function addDataLayers(map: maplibregl.Map) {
  // Military hotspot regions
  map.addSource('hotspots', { type: 'geojson', data: hotspotsToGeoJSON() })
  map.addLayer({
    id: 'hotspots-fill',
    type: 'fill',
    source: 'hotspots',
    paint: { 'fill-color': '#60a5fa', 'fill-opacity': 0.06 },
  })
  map.addLayer({
    id: 'hotspots-outline',
    type: 'line',
    source: 'hotspots',
    paint: { 'line-color': '#60a5fa', 'line-width': 1, 'line-opacity': 0.4, 'line-dasharray': [4, 3] },
  })
  // Chokepoints
  map.addSource('chokepoints', { type: 'geojson', data: chokepointsToGeoJSON() })
  map.addLayer({
    id: 'chokepoints-icon',
    type: 'circle',
    source: 'chokepoints',
    paint: { 'circle-radius': 5, 'circle-color': '#34d399', 'circle-stroke-color': '#064e3b', 'circle-stroke-width': 1.5, 'circle-opacity': 0.8 },
  })
  // Signals (empty initially, updated by data)
  map.addSource('signals', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id: 'signals-glow',
    type: 'circle',
    source: 'signals',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'severity'], 0, 6, 0.5, 10, 1, 18],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.15,
      'circle-blur': 1,
    },
  })
  map.addLayer({
    id: 'signals-dot',
    type: 'circle',
    source: 'signals',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'severity'], 0, 3, 0.5, 5, 1, 8],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.85,
      'circle-stroke-color': '#000000',
      'circle-stroke-width': 0.5,
    },
  })

  // Convergence zones
  map.addSource('convergences', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id: 'convergences-ring',
    type: 'circle',
    source: 'convergences',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'urgency_score'], 0, 20, 50, 30, 100, 45],
      'circle-color': 'transparent',
      'circle-stroke-color': '#c084fc',
      'circle-stroke-width': 2,
      'circle-opacity': 0.7,
    },
  })
  map.addLayer({
    id: 'convergences-fill',
    type: 'circle',
    source: 'convergences',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'urgency_score'], 0, 20, 50, 30, 100, 45],
      'circle-color': '#c084fc',
      'circle-opacity': 0.08,
    },
  })
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function WorldMap() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const popupRef = useRef<maplibregl.Popup | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [mapError, setMapError] = useState<string | null>(null)

  // Fetch data
  const { data: signalsData } = useQuery({
    queryKey: ['world-signals', { limit: 500 }],
    queryFn: () => getWorldSignals({ limit: 500 }),
    refetchInterval: 30000,
  })

  const { data: convergenceData } = useQuery({
    queryKey: ['world-convergences'],
    queryFn: getConvergenceZones,
    refetchInterval: 60000,
  })

  const { data: militaryData } = useQuery({
    queryKey: ['world-military'],
    queryFn: getMilitaryActivity,
    refetchInterval: 60000,
  })

  // Initialize map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const container = containerRef.current
    // Ensure the container has non-zero dimensions before creating the map
    const { offsetWidth, offsetHeight } = container
    if (offsetWidth === 0 || offsetHeight === 0) {
      // Retry on next animation frame when layout has settled
      const raf = requestAnimationFrame(() => {
        if (!mapRef.current && containerRef.current) {
          // Force a re-render to retry
          setMapError(null)
        }
      })
      return () => cancelAnimationFrame(raf)
    }

    let map: maplibregl.Map
    try {
      map = new maplibregl.Map({
        container,
        style: DARK_STYLE,
        center: [30, 25],
        zoom: 2.2,
        minZoom: 1.5,
        maxZoom: 12,
        attributionControl: false,
      })
    } catch (err) {
      setMapError(`Map init failed: ${err instanceof Error ? err.message : String(err)}`)
      return
    }

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left')
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')

    map.on('load', () => {
      addDataLayers(map)
      map.resize()
      mapRef.current = map
      setMapReady(true)
    })

    map.on('error', (e) => {
      // Log tile/style errors but don't crash the map
      console.warn('[WorldMap] MapLibre error:', e.error?.message || e)
    })

    return () => {
      map.remove()
      mapRef.current = null
      setMapReady(false)
    }
  // mapError in deps so the raf retry triggers a new attempt
  }, [mapError])

  // Resize when container dimensions change (e.g. sidebar toggle)
  useEffect(() => {
    if (!mapReady || !mapRef.current || !containerRef.current) return
    const map = mapRef.current
    const ro = new ResizeObserver(() => map.resize())
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [mapReady])

  // Update signal layer when data changes
  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    const source = mapRef.current.getSource('signals') as maplibregl.GeoJSONSource | undefined
    if (!source) return
    source.setData(signalsToGeoJSON(signalsData?.signals || []))
  }, [mapReady, signalsData])

  // Update convergence layer when data changes
  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    const source = mapRef.current.getSource('convergences') as maplibregl.GeoJSONSource | undefined
    if (!source) return
    source.setData(convergencesToGeoJSON(convergenceData?.zones || []))
  }, [mapReady, convergenceData])

  // Popup on click
  const handleSignalClick = useCallback(
    (e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
      if (!mapRef.current || !e.features || e.features.length === 0) return
      const f = e.features[0]
      const coords = (f.geometry as GeoJSON.Point).coordinates.slice() as [number, number]
      const props = f.properties
      popupRef.current?.remove()

      const html = `
        <div style="font-family: ui-monospace, monospace; font-size: 11px; line-height: 1.5; max-width: 260px;">
          <div style="font-weight: 700; margin-bottom: 4px;">${props?.title || 'Signal'}</div>
          <div style="color: #94a3b8;">
            ${props?.country ? `<span>${props.country}</span> · ` : ''}
            <span>${props?.source || ''}</span>
            ${props?.severity != null ? ` · <span style="color: ${props.severity >= 0.7 ? '#f87171' : props.severity >= 0.4 ? '#fb923c' : '#facc15'};">sev ${(props.severity * 100).toFixed(0)}%</span>` : ''}
          </div>
        </div>
      `

      popupRef.current = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '280px', className: 'world-map-popup' })
        .setLngLat(coords)
        .setHTML(html)
        .addTo(mapRef.current!)
    },
    [],
  )

  const handleConvergenceClick = useCallback(
    (e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
      if (!mapRef.current || !e.features || e.features.length === 0) return
      const f = e.features[0]
      const coords = (f.geometry as GeoJSON.Point).coordinates.slice() as [number, number]
      const props = f.properties
      popupRef.current?.remove()

      const html = `
        <div style="font-family: ui-monospace, monospace; font-size: 11px; line-height: 1.5; max-width: 260px;">
          <div style="font-weight: 700; color: #c084fc; margin-bottom: 4px;">Convergence Zone</div>
          <div style="color: #94a3b8;">
            ${props?.country ? `<span>${props.country}</span> · ` : ''}
            <span>${props?.signal_count || 0} signals</span> ·
            <span style="color: ${(props?.urgency_score || 0) >= 70 ? '#f87171' : '#fb923c'};">urgency ${props?.urgency_score || 0}</span>
          </div>
          <div style="color: #94a3b8; margin-top: 2px;">${props?.signal_types || ''}</div>
        </div>
      `

      popupRef.current = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '280px', className: 'world-map-popup' })
        .setLngLat(coords)
        .setHTML(html)
        .addTo(mapRef.current!)
    },
    [],
  )

  // Register click handlers
  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    const map = mapRef.current

    map.on('click', 'signals-dot', handleSignalClick)
    map.on('click', 'convergences-ring', handleConvergenceClick)

    const pointerOn = () => { map.getCanvas().style.cursor = 'pointer' }
    const pointerOff = () => { map.getCanvas().style.cursor = '' }

    map.on('mouseenter', 'signals-dot', pointerOn)
    map.on('mouseleave', 'signals-dot', pointerOff)
    map.on('mouseenter', 'convergences-ring', pointerOn)
    map.on('mouseleave', 'convergences-ring', pointerOff)

    return () => {
      map.off('click', 'signals-dot', handleSignalClick)
      map.off('click', 'convergences-ring', handleConvergenceClick)
      map.off('mouseenter', 'signals-dot', pointerOn)
      map.off('mouseleave', 'signals-dot', pointerOff)
      map.off('mouseenter', 'convergences-ring', pointerOn)
      map.off('mouseleave', 'convergences-ring', pointerOff)
    }
  }, [mapReady, handleSignalClick, handleConvergenceClick])

  const signalCount = signalsData?.signals?.length || 0
  const convergenceCount = convergenceData?.zones?.length || 0
  const surgeRegions: string[] = militaryData?.surge_regions || []

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} className="absolute inset-0" style={{ background: '#0a0e17' }} />
      {mapError && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-20">
          <div className="text-sm text-red-400 font-mono text-center p-4">{mapError}</div>
        </div>
      )}
      <MapLegend />
      <MapStats signalCount={signalCount} convergenceCount={convergenceCount} surgeRegions={surgeRegions} />

      <style>{`
        .world-map-popup .maplibregl-popup-content {
          background: #0f172a;
          border: 1px solid #1e293b;
          border-radius: 8px;
          padding: 10px 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
          color: #e2e8f0;
        }
        .world-map-popup .maplibregl-popup-tip {
          border-top-color: #0f172a;
        }
        .world-map-popup .maplibregl-popup-close-button {
          color: #94a3b8;
          font-size: 16px;
          padding: 2px 6px;
        }
        .world-map-popup .maplibregl-popup-close-button:hover {
          color: #e2e8f0;
          background: transparent;
        }
      `}</style>
    </div>
  )
}
