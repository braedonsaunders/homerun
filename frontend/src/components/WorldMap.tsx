import { useEffect, useRef, useCallback, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import {
  getWorldSignals,
  getConvergenceZones,
  type WorldSignal,
  type ConvergenceZone,
} from '../services/worldIntelligenceApi'

// ---------------------------------------------------------------------------
// Style — fully inline, no external JSON fetch needed
// ---------------------------------------------------------------------------

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
      attribution: '&copy; CARTO &copy; OSM',
      maxzoom: 18,
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#0a0e17' } },
    { id: 'osm-tiles', type: 'raster', source: 'osm-tiles', paint: { 'raster-opacity': 0.85 } },
  ],
}

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const HOTSPOT_BBOXES: Record<string, [number, number, number, number]> = {
  black_sea: [41.0, 47.0, 27.5, 42.0],
  eastern_med: [31.0, 37.0, 24.0, 36.0],
  taiwan_strait: [22.0, 26.0, 117.0, 122.0],
  korean_dmz: [37.0, 39.0, 125.0, 130.0],
  persian_gulf: [24.0, 30.0, 48.0, 57.0],
  south_china_sea: [5.0, 22.0, 105.0, 121.0],
  baltics: [54.0, 60.0, 20.0, 28.0],
}

const CHOKEPOINTS = [
  { name: 'Suez Canal', lat: 30.46, lon: 32.34 },
  { name: 'Strait of Hormuz', lat: 26.57, lon: 56.25 },
  { name: 'Malacca Strait', lat: 2.5, lon: 101.8 },
  { name: 'Panama Canal', lat: 9.08, lon: -79.68 },
  { name: 'Bosphorus', lat: 41.12, lon: 29.05 },
]

const SIGNAL_COLORS: Record<string, string> = {
  conflict: '#f87171',
  tension: '#fb923c',
  instability: '#facc15',
  convergence: '#c084fc',
  anomaly: '#22d3ee',
  military: '#60a5fa',
  infrastructure: '#34d399',
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
        geometry: { type: 'Point' as const, coordinates: [s.longitude!, s.latitude!] },
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
      geometry: { type: 'Point' as const, coordinates: [z.longitude, z.latitude] },
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
        coordinates: [[[lonMin, latMin], [lonMax, latMin], [lonMax, latMax], [lonMin, latMax], [lonMin, latMin]]],
      },
      properties: { name: name.replace(/_/g, ' ') },
    })),
  }
}

function chokepointsToGeoJSON(): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: CHOKEPOINTS.map((cp) => ({
      type: 'Feature' as const,
      geometry: { type: 'Point' as const, coordinates: [cp.lon, cp.lat] },
      properties: { name: cp.name },
    })),
  }
}

// ---------------------------------------------------------------------------
// Add overlay layers (no symbol/text layers — avoids glyph fetches)
// ---------------------------------------------------------------------------

function addDataLayers(map: maplibregl.Map) {
  map.addSource('hotspots', { type: 'geojson', data: hotspotsToGeoJSON() })
  map.addLayer({ id: 'hotspots-fill', type: 'fill', source: 'hotspots', paint: { 'fill-color': '#60a5fa', 'fill-opacity': 0.06 } })
  map.addLayer({ id: 'hotspots-outline', type: 'line', source: 'hotspots', paint: { 'line-color': '#60a5fa', 'line-width': 1, 'line-opacity': 0.4, 'line-dasharray': [4, 3] } })

  map.addSource('chokepoints', { type: 'geojson', data: chokepointsToGeoJSON() })
  map.addLayer({ id: 'chokepoints-icon', type: 'circle', source: 'chokepoints', paint: { 'circle-radius': 5, 'circle-color': '#34d399', 'circle-stroke-color': '#064e3b', 'circle-stroke-width': 1.5, 'circle-opacity': 0.8 } })

  map.addSource('signals', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({ id: 'signals-glow', type: 'circle', source: 'signals', paint: { 'circle-radius': ['interpolate', ['linear'], ['get', 'severity'], 0, 6, 0.5, 10, 1, 18], 'circle-color': ['get', 'color'], 'circle-opacity': 0.15, 'circle-blur': 1 } })
  map.addLayer({ id: 'signals-dot', type: 'circle', source: 'signals', paint: { 'circle-radius': ['interpolate', ['linear'], ['get', 'severity'], 0, 3, 0.5, 5, 1, 8], 'circle-color': ['get', 'color'], 'circle-opacity': 0.85, 'circle-stroke-color': '#000000', 'circle-stroke-width': 0.5 } })

  map.addSource('convergences', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({ id: 'convergences-ring', type: 'circle', source: 'convergences', paint: { 'circle-radius': ['interpolate', ['linear'], ['get', 'urgency_score'], 0, 20, 50, 30, 100, 45], 'circle-color': 'transparent', 'circle-stroke-color': '#c084fc', 'circle-stroke-width': 2, 'circle-opacity': 0.7 } })
  map.addLayer({ id: 'convergences-fill', type: 'circle', source: 'convergences', paint: { 'circle-radius': ['interpolate', ['linear'], ['get', 'urgency_score'], 0, 20, 50, 30, 100, 45], 'circle-color': '#c084fc', 'circle-opacity': 0.08 } })
}

// ---------------------------------------------------------------------------
// Overlays
// ---------------------------------------------------------------------------

function MapLegend() {
  return (
    <div className="absolute bottom-3 left-3 bg-background/90 backdrop-blur-sm border border-border rounded-lg p-2.5 text-[10px] space-y-1 z-10">
      <div className="font-semibold text-[11px] text-foreground mb-1.5">Signal Types</div>
      {Object.entries(SIGNAL_COLORS).map(([type, color]) => (
        <div key={type} className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
          <span className="text-muted-foreground capitalize">{type}</span>
        </div>
      ))}
      <div className="border-t border-border pt-1 mt-1.5 space-y-1">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full shrink-0 border-2 border-purple-400 bg-transparent" />
          <span className="text-muted-foreground">Convergence zone</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 shrink-0 border border-blue-400/60 bg-blue-400/10" />
          <span className="text-muted-foreground">Military hotspot</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 shrink-0 rounded-full bg-emerald-400" />
          <span className="text-muted-foreground">Chokepoint</span>
        </div>
      </div>
    </div>
  )
}

function MapStats({ signalCount, convergenceCount }: { signalCount: number; convergenceCount: number }) {
  return (
    <div className="absolute top-3 right-3 bg-background/90 backdrop-blur-sm border border-border rounded-lg p-2.5 text-[10px] z-10 space-y-1">
      <div className="font-mono">
        <span className="text-muted-foreground">Signals:</span>{' '}
        <span className="text-foreground font-bold">{signalCount}</span>
      </div>
      <div className="font-mono">
        <span className="text-muted-foreground">Convergences:</span>{' '}
        <span className="text-purple-400 font-bold">{convergenceCount}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function WorldMap() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const popupRef = useRef<maplibregl.Popup | null>(null)
  const [mapReady, setMapReady] = useState(false)

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

  // Create map — simple, no guards, just create it
  useEffect(() => {
    const el = containerRef.current
    if (!el || mapRef.current) return

    const map = new maplibregl.Map({
      container: el,
      style: DARK_STYLE,
      center: [30, 25],
      zoom: 2.2,
      minZoom: 1.5,
      maxZoom: 12,
      attributionControl: false,
    })

    mapRef.current = map

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left')

    map.once('load', () => {
      addDataLayers(map)
      setMapReady(true)
    })

    // Resize after first idle to handle late layout
    map.once('idle', () => map.resize())

    map.on('error', (e) => {
      console.warn('[WorldMap]', e.error?.message || e)
    })

    return () => {
      mapRef.current = null
      setMapReady(false)
      map.remove()
    }
  }, [])

  // Keep canvas in sync with container size
  useEffect(() => {
    const el = containerRef.current
    const map = mapRef.current
    if (!el || !map) return
    const ro = new ResizeObserver(() => { map.resize() })
    ro.observe(el)
    return () => ro.disconnect()
  })

  // Push signal data
  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    const src = mapRef.current.getSource('signals') as maplibregl.GeoJSONSource | undefined
    src?.setData(signalsToGeoJSON(signalsData?.signals || []))
  }, [mapReady, signalsData])

  // Push convergence data
  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    const src = mapRef.current.getSource('convergences') as maplibregl.GeoJSONSource | undefined
    src?.setData(convergencesToGeoJSON(convergenceData?.zones || []))
  }, [mapReady, convergenceData])

  // Click handlers
  const handleSignalClick = useCallback((e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
    if (!mapRef.current || !e.features?.length) return
    const props = e.features[0].properties
    const coords = (e.features[0].geometry as GeoJSON.Point).coordinates.slice() as [number, number]
    popupRef.current?.remove()
    popupRef.current = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '280px', className: 'world-map-popup' })
      .setLngLat(coords)
      .setHTML(`<div style="font:11px/1.5 ui-monospace,monospace;max-width:260px"><b>${props?.title||'Signal'}</b><div style="color:#94a3b8">${props?.country?props.country+' · ':''}${props?.source||''}</div></div>`)
      .addTo(mapRef.current)
  }, [])

  const handleConvergenceClick = useCallback((e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
    if (!mapRef.current || !e.features?.length) return
    const props = e.features[0].properties
    const coords = (e.features[0].geometry as GeoJSON.Point).coordinates.slice() as [number, number]
    popupRef.current?.remove()
    popupRef.current = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '280px', className: 'world-map-popup' })
      .setLngLat(coords)
      .setHTML(`<div style="font:11px/1.5 ui-monospace,monospace;max-width:260px"><b style="color:#c084fc">Convergence Zone</b><div style="color:#94a3b8">${props?.country?props.country+' · ':''}${props?.signal_count||0} signals · urgency ${props?.urgency_score||0}</div></div>`)
      .addTo(mapRef.current)
  }, [])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return
    const map = mapRef.current
    map.on('click', 'signals-dot', handleSignalClick)
    map.on('click', 'convergences-ring', handleConvergenceClick)
    const on = () => { map.getCanvas().style.cursor = 'pointer' }
    const off = () => { map.getCanvas().style.cursor = '' }
    map.on('mouseenter', 'signals-dot', on)
    map.on('mouseleave', 'signals-dot', off)
    map.on('mouseenter', 'convergences-ring', on)
    map.on('mouseleave', 'convergences-ring', off)
    return () => {
      map.off('click', 'signals-dot', handleSignalClick)
      map.off('click', 'convergences-ring', handleConvergenceClick)
      map.off('mouseenter', 'signals-dot', on)
      map.off('mouseleave', 'signals-dot', off)
      map.off('mouseenter', 'convergences-ring', on)
      map.off('mouseleave', 'convergences-ring', off)
    }
  }, [mapReady, handleSignalClick, handleConvergenceClick])

  return (
    <div style={{ position: 'absolute', inset: 0, background: '#0a0e17' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <MapLegend />
      <MapStats signalCount={signalsData?.signals?.length || 0} convergenceCount={convergenceData?.zones?.length || 0} />
      <style>{`
        .world-map-popup .maplibregl-popup-content{background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:10px 12px;box-shadow:0 4px 20px rgba(0,0,0,.5);color:#e2e8f0}
        .world-map-popup .maplibregl-popup-tip{border-top-color:#0f172a}
        .world-map-popup .maplibregl-popup-close-button{color:#94a3b8;font-size:16px;padding:2px 6px}
        .world-map-popup .maplibregl-popup-close-button:hover{color:#e2e8f0;background:transparent}
      `}</style>
    </div>
  )
}
