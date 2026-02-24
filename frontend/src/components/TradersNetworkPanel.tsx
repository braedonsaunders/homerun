import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Filter,
  Link2,
  Loader2,
  Network,
  RefreshCw,
  Search,
  Target,
  Users,
  UserPlus,
  UserX,
  Zap,
} from 'lucide-react'
import {
  discoveryApi,
  type TraderGroup,
  type TraderNetworkCohort,
  type TraderNetworkEdge,
  type TraderNetworkNode,
} from '../services/discoveryApi'
import { addWallet, removeWallet } from '../services/api'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'

interface TradersNetworkPanelProps {
  onAnalyzeWallet?: (address: string, username?: string) => void
  onNavigateToWallet?: (address: string) => void
}

type Viewport = {
  x: number
  y: number
  scale: number
}

type PositionedNode = TraderNetworkNode & {
  x: number
  y: number
  r: number
}

type DockTab = 'context' | 'filters' | 'stats'
type FocusMode = 'all' | 'tracked' | 'selected' | 'cohort'

type FilteredGraph = {
  nodes: TraderNetworkNode[]
  edges: TraderNetworkEdge[]
  trackedSeedCount: number
  selectedSeedCount: number
  cohortSeedCount: number
  focusSeedCount: number
  focusMode: FocusMode
  counts: {
    walletsVisible: number
    groupsVisible: number
    clustersVisible: number
    edgesVisible: number
    walletsTotal: number
    edgesTotal: number
  }
}

const MIN_EDGE_SCORE = 0.5
const MAX_EDGE_SCORE = 0.96
const DEFAULT_NODE_LIMIT = 180
const NODE_LIMIT_OPTIONS = [110, 150, 180, 240, 320]
const DEFAULT_EDGE_THRESHOLD = 0.74
const DEFAULT_CONNECTION_CAP = 5
const MIN_CONNECTION_CAP = 2
const MAX_CONNECTION_CAP = 16
const DEFAULT_NEIGHBOR_DEPTH = 1
const MIN_NEIGHBOR_DEPTH = 1
const MAX_NEIGHBOR_DEPTH = 3
const VIEWBOX_WIDTH = 2200
const VIEWBOX_HEIGHT = 1300

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function shortAddress(address: string | undefined): string {
  const value = String(address || '').trim()
  if (value.length < 12) return value
  return `${value.slice(0, 6)}...${value.slice(-4)}`
}

function formatUsd(value: number | null | undefined): string {
  const num = Number(value || 0)
  if (!Number.isFinite(num)) return '$0'
  const abs = Math.abs(num)
  if (abs >= 1_000_000) return `$${(num / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `$${(num / 1_000).toFixed(1)}K`
  return `$${num.toFixed(0)}`
}

function formatPct(value: number | null | undefined): string {
  const num = Number(value || 0)
  if (!Number.isFinite(num)) return '0.0%'
  const ratio = Math.abs(num) <= 1 ? num * 100 : num
  return `${ratio.toFixed(1)}%`
}

function extractErrorMessage(error: unknown, fallback: string): string {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0]
    if (typeof first === 'string' && first.trim()) return first.trim()
  }
  const message = (error as { message?: unknown })?.message
  if (typeof message === 'string' && message.trim()) return message.trim()
  return fallback
}

function hashToSeed(text: string): number {
  let hash = 2166136261
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0) / 4294967295
}

function nodeRadius(node: TraderNetworkNode): number {
  const kind = String(node.kind || 'wallet')
  if (kind === 'wallet') {
    const degree = Number(node.co_trade_degree ?? node.degree ?? 0)
    const composite = Number(node.composite_score ?? 0)
    return clamp(7 + degree * 0.65 + composite * 4.5, 7, 18)
  }
  if (kind === 'group') {
    return clamp(10 + Number(node.linked_wallet_count || 0) * 0.35, 10, 20)
  }
  if (kind === 'cluster') {
    return clamp(9 + Number(node.member_count || 0) * 0.35, 9, 19)
  }
  return 8
}

function initialPosition(node: TraderNetworkNode, index: number, total: number): { x: number; y: number } {
  const ring = Math.max(200, Math.min(430, total * 2.0))
  const angle = (index / Math.max(total, 1)) * Math.PI * 2
  const jitter = (hashToSeed(String(node.id)) - 0.5) * 120
  return {
    x: Math.cos(angle) * (ring + jitter),
    y: Math.sin(angle) * (ring + jitter * 0.75),
  }
}

function scoreOfEdge(edge: TraderNetworkEdge): number {
  return clamp(Number(edge.combined_score ?? edge.weight ?? 0), 0, 1)
}

function pruneEdgesByNodeCap(edges: TraderNetworkEdge[], cap: number): TraderNetworkEdge[] {
  if (cap <= 0) return edges
  const sorted = [...edges].sort((a, b) => scoreOfEdge(b) - scoreOfEdge(a))
  const degree = new Map<string, number>()
  const kept: TraderNetworkEdge[] = []
  for (const edge of sorted) {
    const source = String(edge.source)
    const target = String(edge.target)
    const sourceDegree = degree.get(source) || 0
    const targetDegree = degree.get(target) || 0
    if (sourceDegree >= cap || targetDegree >= cap) continue
    kept.push(edge)
    degree.set(source, sourceDegree + 1)
    degree.set(target, targetDegree + 1)
  }
  return kept
}

function connectedComponents(nodeIds: string[], edges: TraderNetworkEdge[]): string[][] {
  const adjacency = new Map<string, Set<string>>()
  for (const id of nodeIds) adjacency.set(id, new Set())
  for (const edge of edges) {
    const source = String(edge.source)
    const target = String(edge.target)
    if (!adjacency.has(source) || !adjacency.has(target)) continue
    adjacency.get(source)?.add(target)
    adjacency.get(target)?.add(source)
  }

  const visited = new Set<string>()
  const components: string[][] = []
  for (const id of nodeIds) {
    if (visited.has(id)) continue
    const queue = [id]
    const component: string[] = []
    while (queue.length) {
      const current = queue.shift() as string
      if (visited.has(current)) continue
      visited.add(current)
      component.push(current)
      const neighbors = adjacency.get(current)
      if (!neighbors) continue
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) queue.push(neighbor)
      }
    }
    components.push(component)
  }

  components.sort((a, b) => b.length - a.length)
  return components
}

function buildWalletAdjacency(edges: TraderNetworkEdge[]): Map<string, Array<{ id: string; weight: number }>> {
  const adjacency = new Map<string, Array<{ id: string; weight: number }>>()
  for (const edge of edges) {
    if (String(edge.kind) !== 'co_trade') continue
    const source = String(edge.source)
    const target = String(edge.target)
    if (!source || !target || source === target) continue
    const weight = scoreOfEdge(edge)
    if (!adjacency.has(source)) adjacency.set(source, [])
    if (!adjacency.has(target)) adjacency.set(target, [])
    adjacency.get(source)?.push({ id: target, weight })
    adjacency.get(target)?.push({ id: source, weight })
  }
  return adjacency
}

function expandWalletNeighborhood(
  edges: TraderNetworkEdge[],
  seeds: Set<string>,
  depth: number,
): Set<string> {
  if (seeds.size === 0) return new Set()
  const adjacency = buildWalletAdjacency(edges)
  const visited = new Set<string>(seeds)
  let frontier = [...seeds]
  const maxDepth = Math.max(0, Math.floor(depth))

  for (let hop = 0; hop < maxDepth; hop += 1) {
    const next: string[] = []
    for (const nodeId of frontier) {
      const neighbors = adjacency.get(nodeId) || []
      for (const neighbor of neighbors) {
        if (visited.has(neighbor.id)) continue
        visited.add(neighbor.id)
        next.push(neighbor.id)
      }
    }
    if (!next.length) break
    frontier = next
  }

  return visited
}

function detectWalletCommunities(
  walletNodeIds: string[],
  coTradeEdges: TraderNetworkEdge[],
): string[][] {
  if (!walletNodeIds.length) return []

  const walletSet = new Set(walletNodeIds)
  const adjacency = new Map<string, Array<{ id: string; weight: number }>>()
  for (const id of walletNodeIds) adjacency.set(id, [])

  for (const edge of coTradeEdges) {
    const source = String(edge.source)
    const target = String(edge.target)
    if (!walletSet.has(source) || !walletSet.has(target) || source === target) continue
    const weight = scoreOfEdge(edge)
    adjacency.get(source)?.push({ id: target, weight })
    adjacency.get(target)?.push({ id: source, weight })
  }

  const labels = new Map<string, string>()
  for (const id of walletNodeIds) labels.set(id, id)

  const orderedWalletIds = [...walletNodeIds].sort((a, b) => {
    const degreeDelta = (adjacency.get(b)?.length || 0) - (adjacency.get(a)?.length || 0)
    if (degreeDelta !== 0) return degreeDelta
    return a.localeCompare(b)
  })

  for (let iteration = 0; iteration < 12; iteration += 1) {
    let changed = false
    for (const nodeId of orderedWalletIds) {
      const neighbors = adjacency.get(nodeId) || []
      if (!neighbors.length) continue

      const scoreByLabel = new Map<string, number>()
      for (const neighbor of neighbors) {
        const label = labels.get(neighbor.id) || neighbor.id
        scoreByLabel.set(label, (scoreByLabel.get(label) || 0) + neighbor.weight)
      }

      let nextLabel = labels.get(nodeId) || nodeId
      let nextScore = -1
      for (const [candidateLabel, score] of scoreByLabel) {
        if (score > nextScore || (score === nextScore && candidateLabel < nextLabel)) {
          nextLabel = candidateLabel
          nextScore = score
        }
      }

      if (nextLabel !== (labels.get(nodeId) || nodeId)) {
        labels.set(nodeId, nextLabel)
        changed = true
      }
    }
    if (!changed) break
  }

  const groupByLabel = (): Map<string, string[]> => {
    const groups = new Map<string, string[]>()
    for (const id of walletNodeIds) {
      const label = labels.get(id) || id
      if (!groups.has(label)) groups.set(label, [])
      groups.get(label)?.push(id)
    }
    return groups
  }

  const MIN_COMMUNITY_SIZE = 3
  let groups = groupByLabel()

  for (const [label, members] of groups) {
    if (members.length >= MIN_COMMUNITY_SIZE) continue

    const scoreByTargetLabel = new Map<string, number>()
    for (const memberId of members) {
      const neighbors = adjacency.get(memberId) || []
      for (const neighbor of neighbors) {
        const candidateLabel = labels.get(neighbor.id) || neighbor.id
        if (candidateLabel === label) continue
        scoreByTargetLabel.set(
          candidateLabel,
          (scoreByTargetLabel.get(candidateLabel) || 0) + neighbor.weight,
        )
      }
    }

    let targetLabel = ''
    let targetScore = -1
    for (const [candidateLabel, score] of scoreByTargetLabel) {
      const candidateSize = groups.get(candidateLabel)?.length || 0
      if (candidateSize < MIN_COMMUNITY_SIZE && groups.size > 1) continue
      if (score > targetScore || (score === targetScore && candidateLabel < targetLabel)) {
        targetLabel = candidateLabel
        targetScore = score
      }
    }

    if (!targetLabel) continue
    for (const memberId of members) {
      labels.set(memberId, targetLabel)
    }
  }

  groups = groupByLabel()
  return [...groups.values()]
    .map((members) => [...members])
    .sort((a, b) => b.length - a.length)
}

function computeForceLayout(
  nodes: TraderNetworkNode[],
  edges: TraderNetworkEdge[],
): Record<string, { x: number; y: number }> {
  if (!nodes.length) return {}
  const idToIndex = new Map<string, number>()
  const points = nodes.map((node, index) => {
    idToIndex.set(String(node.id), index)
    const pos = initialPosition(node, index, nodes.length)
    return {
      id: String(node.id),
      x: pos.x,
      y: pos.y,
      vx: 0,
      vy: 0,
      mass: clamp(nodeRadius(node), 8, 24),
    }
  })

  const springs = edges
    .map((edge) => {
      const source = idToIndex.get(String(edge.source))
      const target = idToIndex.get(String(edge.target))
      if (source == null || target == null || source === target) return null
      const weight = scoreOfEdge(edge)
      const restLength = edge.kind === 'co_trade'
        ? clamp(340 - weight * 175, 140, 340)
        : edge.kind === 'group_membership'
          ? 205
          : 190
      const spring = edge.kind === 'co_trade' ? 0.015 + weight * 0.03 : 0.033
      return { source, target, restLength, spring }
    })
    .filter((edge): edge is { source: number; target: number; restLength: number; spring: number } => edge != null)

  const iterations = clamp(Math.round(62 + nodes.length * 0.9), 68, 190)
  const repelStrength = 8700
  const centerPull = 0.0022
  const damping = 0.86

  for (let iter = 0; iter < iterations; iter += 1) {
    for (let i = 0; i < points.length; i += 1) {
      const a = points[i]
      for (let j = i + 1; j < points.length; j += 1) {
        const b = points[j]
        const dx = a.x - b.x
        const dy = a.y - b.y
        const distSq = dx * dx + dy * dy + 0.1
        const force = repelStrength / distSq
        const invDist = 1 / Math.sqrt(distSq)
        const fx = dx * invDist * force
        const fy = dy * invDist * force
        a.vx += fx / a.mass
        a.vy += fy / a.mass
        b.vx -= fx / b.mass
        b.vy -= fy / b.mass
      }
    }

    for (const spring of springs) {
      const source = points[spring.source]
      const target = points[spring.target]
      const dx = target.x - source.x
      const dy = target.y - source.y
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.0001
      const stretch = dist - spring.restLength
      const force = stretch * spring.spring
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      source.vx += fx / source.mass
      source.vy += fy / source.mass
      target.vx -= fx / target.mass
      target.vy -= fy / target.mass
    }

    for (const point of points) {
      point.vx += -point.x * centerPull
      point.vy += -point.y * centerPull
      point.vx *= damping
      point.vy *= damping
      point.x += point.vx
      point.y += point.vy
    }
  }

  const out: Record<string, { x: number; y: number }> = {}
  for (const point of points) {
    out[point.id] = { x: point.x, y: point.y }
  }
  return out
}

function computePackedLayout(
  nodes: TraderNetworkNode[],
  edges: TraderNetworkEdge[],
): Record<string, { x: number; y: number }> {
  if (!nodes.length) return {}

  const byId = new Map<string, TraderNetworkNode>()
  for (const node of nodes) byId.set(String(node.id), node)

  const walletNodes = nodes.filter((node) => String(node.kind) === 'wallet')
  const walletNodeIds = walletNodes.map((node) => String(node.id))
  const walletIdSet = new Set(walletNodeIds)
  const coTradeEdges = edges.filter(
    (edge) =>
      String(edge.kind) === 'co_trade'
      && walletIdSet.has(String(edge.source))
      && walletIdSet.has(String(edge.target)),
  )

  let walletCommunities = detectWalletCommunities(walletNodeIds, coTradeEdges)
  if (!walletCommunities.length) {
    walletCommunities = connectedComponents(walletNodeIds, coTradeEdges)
  }
  if (!walletCommunities.length) {
    walletCommunities = walletNodeIds.length ? [walletNodeIds] : []
  }

  const communities: Array<Set<string>> = walletCommunities.map((members) => new Set(members))
  const walletCommunityByNode = new Map<string, number>()
  communities.forEach((community, index) => {
    for (const nodeId of community) walletCommunityByNode.set(nodeId, index)
  })

  const nonWalletNodes = nodes.filter((node) => String(node.kind) !== 'wallet')
  for (const nonWallet of nonWalletNodes) {
    const nonWalletId = String(nonWallet.id)
    const scoreByCommunity = new Map<number, number>()
    for (const edge of edges) {
      const source = String(edge.source)
      const target = String(edge.target)
      if (source !== nonWalletId && target !== nonWalletId) continue
      const otherId = source === nonWalletId ? target : source
      const communityIndex = walletCommunityByNode.get(otherId)
      if (communityIndex == null) continue
      scoreByCommunity.set(
        communityIndex,
        (scoreByCommunity.get(communityIndex) || 0) + scoreOfEdge(edge) + 0.2,
      )
    }

    let assignedIndex = communities.length ? 0 : -1
    let assignedScore = -1
    for (const [communityIndex, score] of scoreByCommunity) {
      const sizeBias = (communities[communityIndex]?.size || 0) * 0.01
      const weighted = score + sizeBias
      if (weighted > assignedScore) {
        assignedScore = weighted
        assignedIndex = communityIndex
      }
    }

    if (assignedIndex === -1) {
      communities.push(new Set<string>([nonWalletId]))
      assignedIndex = communities.length - 1
    } else {
      communities[assignedIndex]?.add(nonWalletId)
    }
  }

  const seenNodes = new Set<string>()
  for (const community of communities) {
    for (const id of community) seenNodes.add(id)
  }
  for (const node of nodes) {
    const nodeId = String(node.id)
    if (seenNodes.has(nodeId)) continue
    communities.push(new Set<string>([nodeId]))
  }

  const orderedCommunities = communities
    .map((nodeIds) => [...nodeIds])
    .filter((nodeIds) => nodeIds.length > 0)
    .sort((a, b) => b.length - a.length)

  const communityLayouts = orderedCommunities.map((nodeIds, index) => {
    const nodeSet = new Set(nodeIds)
    const componentNodes = nodeIds
      .map((id) => byId.get(id))
      .filter((node): node is TraderNetworkNode => Boolean(node))
    const componentEdges = edges.filter((edge) => {
      const source = String(edge.source)
      const target = String(edge.target)
      return nodeSet.has(source) && nodeSet.has(target)
    })

    const local = computeForceLayout(componentNodes, componentEdges)
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const nodeId of nodeIds) {
      const pos = local[nodeId]
      if (!pos) continue
      minX = Math.min(minX, pos.x)
      maxX = Math.max(maxX, pos.x)
      minY = Math.min(minY, pos.y)
      maxY = Math.max(maxY, pos.y)
    }

    const centerX = Number.isFinite(minX) ? (minX + maxX) / 2 : 0
    const centerY = Number.isFinite(minY) ? (minY + maxY) / 2 : 0
    const width = Number.isFinite(minX) ? maxX - minX : 120
    const height = Number.isFinite(minY) ? maxY - minY : 120
    const radius = clamp(
      Math.max(width, height, 130) * 0.58 + Math.sqrt(Math.max(componentNodes.length, 1)) * 16,
      110,
      560,
    )

    return {
      index,
      nodeIds,
      local,
      centerX,
      centerY,
      radius,
    }
  })

  const GOLDEN_ANGLE = 2.399963229728653
  const centers = communityLayouts.map((community, index) => {
    const angle = index * GOLDEN_ANGLE
    const ring = index === 0 ? 0 : 290 + Math.sqrt(index) * 250 + community.radius * 0.35
    return {
      x: Math.cos(angle) * ring,
      y: Math.sin(angle) * ring,
      radius: community.radius,
    }
  })

  for (let iteration = 0; iteration < 120; iteration += 1) {
    for (let i = 0; i < centers.length; i += 1) {
      for (let j = i + 1; j < centers.length; j += 1) {
        const a = centers[i]
        const b = centers[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.001
        const minDist = a.radius + b.radius + 110
        if (dist >= minDist) continue
        const push = (minDist - dist) * 0.52
        const ux = dx / dist
        const uy = dy / dist
        a.x -= ux * push
        a.y -= uy * push
        b.x += ux * push
        b.y += uy * push
      }
    }

    for (const center of centers) {
      center.x *= 0.996
      center.y *= 0.996
    }
  }

  const finalPositions: Record<string, { x: number; y: number }> = {}
  for (const community of communityLayouts) {
    const anchor = centers[community.index] || { x: 0, y: 0 }
    for (const nodeId of community.nodeIds) {
      const base = community.local[nodeId] || { x: 0, y: 0 }
      finalPositions[nodeId] = {
        x: clamp(
          base.x - community.centerX + anchor.x,
          -VIEWBOX_WIDTH * 0.46,
          VIEWBOX_WIDTH * 0.46,
        ),
        y: clamp(
          base.y - community.centerY + anchor.y,
          -VIEWBOX_HEIGHT * 0.46,
          VIEWBOX_HEIGHT * 0.46,
        ),
      }
    }
  }

  if (!Object.keys(finalPositions).length) {
    orderedCommunities.forEach((communityNodeIds, index) => {
      const angle = index * GOLDEN_ANGLE
      const ring = index === 0 ? 0 : 260 + Math.sqrt(index) * 220
      const cx = Math.cos(angle) * ring
      const cy = Math.sin(angle) * ring
      communityNodeIds.forEach((nodeId, nodeIndex) => {
        const pos = initialPosition(byId.get(nodeId) as TraderNetworkNode, nodeIndex, communityNodeIds.length)
        finalPositions[nodeId] = {
          x: clamp(cx + pos.x * 0.36, -VIEWBOX_WIDTH * 0.46, VIEWBOX_WIDTH * 0.46),
          y: clamp(cy + pos.y * 0.36, -VIEWBOX_HEIGHT * 0.46, VIEWBOX_HEIGHT * 0.46),
        }
      })
    })
  }

  return finalPositions
}

function edgePath(source: PositionedNode, target: PositionedNode, curveSeed: string): string {
  const x1 = source.x
  const y1 = source.y
  const x2 = target.x
  const y2 = target.y
  const mx = (x1 + x2) / 2
  const my = (y1 + y2) / 2
  const dx = x2 - x1
  const dy = y2 - y1
  const dist = Math.sqrt(dx * dx + dy * dy) || 1
  const nx = -dy / dist
  const ny = dx / dist
  const curveBias = (hashToSeed(curveSeed) - 0.5) * clamp(dist * 0.18, 6, 34)
  const cx = mx + nx * curveBias
  const cy = my + ny * curveBias
  return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`
}

export default function TradersNetworkPanel({ onAnalyzeWallet, onNavigateToWallet }: TradersNetworkPanelProps) {
  const queryClient = useQueryClient()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const autoFitDoneRef = useRef(false)

  const [nodeLimit, setNodeLimit] = useState(DEFAULT_NODE_LIMIT)
  const [edgeThreshold, setEdgeThreshold] = useState(DEFAULT_EDGE_THRESHOLD)
  const [connectionCap, setConnectionCap] = useState(DEFAULT_CONNECTION_CAP)
  const [showGroups, setShowGroups] = useState(true)
  const [showClusters, setShowClusters] = useState(false)
  const [focusMode, setFocusMode] = useState<FocusMode>('all')
  const [neighborhoodDepth, setNeighborhoodDepth] = useState(DEFAULT_NEIGHBOR_DEPTH)
  const [hideIsolated, setHideIsolated] = useState(true)
  const [search, setSearch] = useState('')

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedGroupId, setSelectedGroupId] = useState('')
  const [selectedCohortId, setSelectedCohortId] = useState<string | null>(null)
  const [dockExpanded, setDockExpanded] = useState(false)
  const [dockTab, setDockTab] = useState<DockTab>('context')
  const [message, setMessage] = useState<string | null>(null)

  const [viewport, setViewport] = useState<Viewport>({ x: 0, y: 0, scale: 1 })
  const [isPanning, setIsPanning] = useState(false)
  const [panOrigin, setPanOrigin] = useState<{ x: number; y: number } | null>(null)
  const [size, setSize] = useState({ width: 1200, height: 740 })

  const { data: graphData, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['traders-network-graph', nodeLimit],
    queryFn: () =>
      discoveryApi.getTradersNetworkGraph({
        limit_wallets: nodeLimit,
        min_pair_score: MIN_EDGE_SCORE,
        include_groups: true,
        include_clusters: true,
      }),
    staleTime: 25_000,
    refetchInterval: 60_000,
  })

  const { data: groups = [] } = useQuery<TraderGroup[]>({
    queryKey: ['trader-groups-network'],
    queryFn: () => discoveryApi.getTraderGroups(false, 200),
    staleTime: 20_000,
    refetchInterval: 60_000,
  })

  const setActionMessage = useCallback((text: string) => {
    setMessage(text)
  }, [])

  useEffect(() => {
    if (!message) return
    const timer = setTimeout(() => setMessage(null), 3400)
    return () => clearTimeout(timer)
  }, [message])

  const invalidateNetworkData = useCallback((text: string) => {
    setActionMessage(text)
    queryClient.invalidateQueries({ queryKey: ['traders-network-graph'] })
    queryClient.invalidateQueries({ queryKey: ['trader-groups-network'] })
    queryClient.invalidateQueries({ queryKey: ['trader-groups'] })
  }, [queryClient, setActionMessage])

  const trackWalletMutation = useMutation({
    mutationFn: ({ address, label }: { address: string; label?: string }) => addWallet(address, label),
    onSuccess: () => invalidateNetworkData('Wallet added to tracking'),
    onError: (error) => setActionMessage(extractErrorMessage(error, 'Failed to track wallet')),
  })

  const untrackWalletMutation = useMutation({
    mutationFn: (address: string) => removeWallet(address),
    onSuccess: () => invalidateNetworkData('Wallet removed from tracking'),
    onError: (error) => setActionMessage(extractErrorMessage(error, 'Failed to untrack wallet')),
  })

  const addToGroupMutation = useMutation({
    mutationFn: ({ groupId, address }: { groupId: string; address: string }) =>
      discoveryApi.addTraderGroupMembers(groupId, {
        wallet_addresses: [address],
        add_to_tracking: true,
        source_label: 'network_graph',
      }),
    onSuccess: () => invalidateNetworkData('Wallet added to group'),
    onError: (error) => setActionMessage(extractErrorMessage(error, 'Failed to add wallet to group')),
  })

  const createGroupMutation = useMutation({
    mutationFn: (payload: { name: string; walletAddresses: string[]; seedAddress: string }) =>
      discoveryApi.createTraderGroup({
        name: payload.name,
        description: `Generated from trader network around ${shortAddress(payload.seedAddress)}`,
        wallet_addresses: payload.walletAddresses,
        source_type: 'manual',
        criteria: {
          source: 'network_graph',
          seed_wallet: payload.seedAddress,
        },
        source_label: 'network_graph',
        auto_track_members: true,
      }),
    onSuccess: (result) =>
      invalidateNetworkData(
        `Group created (${result.group?.member_count ?? 0} members, ${result.tracked_members} tracked)`,
      ),
    onError: (error) => setActionMessage(extractErrorMessage(error, 'Failed to create group')),
  })

  const trackGroupMutation = useMutation({
    mutationFn: (groupId: string) => discoveryApi.trackTraderGroupMembers(groupId),
    onSuccess: (result) => invalidateNetworkData(`Tracking refreshed for ${result.tracked_members} group members`),
    onError: (error) => setActionMessage(extractErrorMessage(error, 'Failed to track group members')),
  })

  const deleteGroupMutation = useMutation({
    mutationFn: (groupId: string) => discoveryApi.deleteTraderGroup(groupId),
    onSuccess: () => {
      setSelectedNodeId(null)
      invalidateNetworkData('Group deleted')
    },
    onError: (error) => setActionMessage(extractErrorMessage(error, 'Failed to delete group')),
  })

  useEffect(() => {
    if (!containerRef.current) return
    const resizeObserver = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect
      if (!rect) return
      setSize({
        width: Math.max(360, rect.width),
        height: Math.max(420, rect.height),
      })
    })
    resizeObserver.observe(containerRef.current)
    return () => resizeObserver.disconnect()
  }, [])

  const rawNodes = graphData?.nodes || []
  const rawEdges = graphData?.edges || []
  const rawCohorts = graphData?.cohorts || []

  const filteredGraph = useMemo<FilteredGraph>(() => {
    const walletNodes = rawNodes.filter((node) => String(node.kind) === 'wallet')
    const walletNodeIds = new Set(walletNodes.map((node) => String(node.id)))
    const walletNodeIdByAddress = new Map<string, string>()
    for (const node of walletNodes) {
      const address = String(node.address || '').trim().toLowerCase()
      if (!address) continue
      walletNodeIdByAddress.set(address, String(node.id))
    }
    const nodeKindById = new Map<string, string>()
    for (const node of rawNodes) {
      nodeKindById.set(String(node.id), String(node.kind))
    }

    const rawCoTradeEdges = rawEdges.filter(
      (edge) =>
        String(edge.kind) === 'co_trade'
        && scoreOfEdge(edge) >= edgeThreshold,
    )
    const cappedCoTradeEdges = pruneEdgesByNodeCap(rawCoTradeEdges, connectionCap)

    const trackedSeedIds = new Set(
      walletNodes
        .filter((node) => Boolean(node.tracked))
        .map((node) => String(node.id)),
    )

    const selectedSeedIds = new Set<string>()
    if (selectedNodeId && nodeKindById.has(selectedNodeId)) {
      const selectedKind = nodeKindById.get(selectedNodeId)
      if (selectedKind === 'wallet') {
        selectedSeedIds.add(selectedNodeId)
      } else {
        for (const edge of rawEdges) {
          const source = String(edge.source)
          const target = String(edge.target)
          if (source === selectedNodeId && nodeKindById.get(target) === 'wallet') {
            selectedSeedIds.add(target)
          }
          if (target === selectedNodeId && nodeKindById.get(source) === 'wallet') {
            selectedSeedIds.add(source)
          }
        }
      }
    }

    const cohortSeedIds = new Set<string>()
    if (selectedCohortId) {
      const selectedCohort = rawCohorts.find((cohort) => String(cohort.id) === selectedCohortId)
      const cohortAddresses = (selectedCohort?.visible_wallet_addresses || selectedCohort?.wallet_addresses || [])
        .map((address) => String(address).trim().toLowerCase())
      for (const address of cohortAddresses) {
        const nodeId = walletNodeIdByAddress.get(address)
        if (nodeId) cohortSeedIds.add(nodeId)
      }
    }

    let focusSeedIds = new Set<string>()
    if (focusMode === 'tracked') {
      focusSeedIds = trackedSeedIds
    } else if (focusMode === 'selected') {
      focusSeedIds = selectedSeedIds
    } else if (focusMode === 'cohort') {
      focusSeedIds = cohortSeedIds
    }

    let allowedWalletIds = new Set(walletNodeIds)
    if (focusMode !== 'all') {
      allowedWalletIds = expandWalletNeighborhood(cappedCoTradeEdges, focusSeedIds, neighborhoodDepth)
    }

    const coTradeEdges = cappedCoTradeEdges.filter(
      (edge) =>
        allowedWalletIds.has(String(edge.source))
        && allowedWalletIds.has(String(edge.target)),
    )

    const clusterEdges = showClusters
      ? rawEdges.filter(
        (edge) =>
          String(edge.kind) === 'cluster_membership'
          && allowedWalletIds.has(String(edge.target)),
      )
      : []

    const groupEdges = showGroups
      ? rawEdges.filter(
        (edge) =>
          String(edge.kind) === 'group_membership'
          && allowedWalletIds.has(String(edge.target)),
      )
      : []

    const visibleNodeIds = new Set<string>(allowedWalletIds)
    for (const edge of [...coTradeEdges, ...clusterEdges, ...groupEdges]) {
      visibleNodeIds.add(String(edge.source))
      visibleNodeIds.add(String(edge.target))
    }

    let visibleNodes = rawNodes.filter((node) => {
      const id = String(node.id)
      if (!visibleNodeIds.has(id)) return false
      if (String(node.kind) === 'group' && !showGroups) return false
      if (String(node.kind) === 'cluster' && !showClusters) return false
      return true
    })

    let visibleEdges = [...coTradeEdges, ...clusterEdges, ...groupEdges].filter((edge) => {
      const source = String(edge.source)
      const target = String(edge.target)
      return visibleNodeIds.has(source) && visibleNodeIds.has(target)
    })

    if (hideIsolated) {
      const degreeByNode = new Map<string, number>()
      for (const edge of visibleEdges) {
        const source = String(edge.source)
        const target = String(edge.target)
        degreeByNode.set(source, (degreeByNode.get(source) || 0) + 1)
        degreeByNode.set(target, (degreeByNode.get(target) || 0) + 1)
      }
      const filteredNodeIds = new Set<string>()
      for (const node of visibleNodes) {
        const id = String(node.id)
        const degree = degreeByNode.get(id) || 0
        if (degree > 0) {
          filteredNodeIds.add(id)
          continue
        }
        if (String(node.kind) === 'wallet' && focusSeedIds.has(id)) {
          filteredNodeIds.add(id)
        }
      }
      visibleNodes = visibleNodes.filter((node) => filteredNodeIds.has(String(node.id)))
      visibleEdges = visibleEdges.filter(
        (edge) =>
          filteredNodeIds.has(String(edge.source))
          && filteredNodeIds.has(String(edge.target)),
      )
    }

    return {
      nodes: visibleNodes,
      edges: visibleEdges,
      trackedSeedCount: trackedSeedIds.size,
      selectedSeedCount: selectedSeedIds.size,
      cohortSeedCount: cohortSeedIds.size,
      focusSeedCount: focusSeedIds.size,
      focusMode,
      counts: {
        walletsVisible: visibleNodes.filter((node) => String(node.kind) === 'wallet').length,
        groupsVisible: visibleNodes.filter((node) => String(node.kind) === 'group').length,
        clustersVisible: visibleNodes.filter((node) => String(node.kind) === 'cluster').length,
        edgesVisible: visibleEdges.length,
        walletsTotal: walletNodes.length,
        edgesTotal: rawCoTradeEdges.length,
      },
    }
  }, [
    connectionCap,
    edgeThreshold,
    focusMode,
    hideIsolated,
    neighborhoodDepth,
    rawEdges,
    rawNodes,
    rawCohorts,
    selectedCohortId,
    selectedNodeId,
    showClusters,
    showGroups,
  ])

  useEffect(() => {
    if (focusMode === 'tracked' && filteredGraph.trackedSeedCount === 0) {
      setActionMessage('Tracked neighborhood needs tracked wallets in the current graph.')
    }
    if (focusMode === 'selected' && filteredGraph.selectedSeedCount === 0) {
      setActionMessage('Select a wallet or entity first, then switch to selected neighborhood focus.')
    }
    if (focusMode === 'cohort' && filteredGraph.cohortSeedCount === 0) {
      setActionMessage('Select a cohort from insights first, then enable cohort focus.')
    }
  }, [
    filteredGraph.cohortSeedCount,
    filteredGraph.selectedSeedCount,
    filteredGraph.trackedSeedCount,
    focusMode,
    setActionMessage,
  ])

  const searchMatches = useMemo(() => {
    const searchTerm = search.trim().toLowerCase()
    if (!searchTerm) return new Set<string>()
    const matches = filteredGraph.nodes
      .filter((node) => {
        const haystack = `${node.label || ''} ${node.address || ''} ${node.username || ''} ${node.group_id || ''}`.toLowerCase()
        return haystack.includes(searchTerm)
      })
      .map((node) => String(node.id))
    return new Set(matches)
  }, [filteredGraph.nodes, search])

  const layoutMap = useMemo(
    () => computePackedLayout(filteredGraph.nodes, filteredGraph.edges),
    [filteredGraph.edges, filteredGraph.nodes],
  )

  const positionedNodes = useMemo(() => {
    return filteredGraph.nodes.map((node) => {
      const pos = layoutMap[String(node.id)] || { x: 0, y: 0 }
      return {
        ...node,
        x: pos.x,
        y: pos.y,
        r: nodeRadius(node),
      }
    })
  }, [filteredGraph.nodes, layoutMap])

  const positionedNodeById = useMemo(() => {
    const map = new Map<string, PositionedNode>()
    for (const node of positionedNodes) {
      map.set(String(node.id), node)
    }
    return map
  }, [positionedNodes])

  const selectedNode = useMemo(
    () => (selectedNodeId ? positionedNodeById.get(selectedNodeId) || null : null),
    [positionedNodeById, selectedNodeId],
  )

  useEffect(() => {
    if (!selectedNodeId) return
    if (!positionedNodeById.has(selectedNodeId)) {
      setSelectedNodeId(null)
    }
  }, [positionedNodeById, selectedNodeId])

  const selectedNeighborhoodIds = useMemo(() => {
    if (!selectedNodeId) return new Set<string>()
    const ids = new Set<string>([selectedNodeId])
    for (const edge of filteredGraph.edges) {
      const source = String(edge.source)
      const target = String(edge.target)
      if (source === selectedNodeId) ids.add(target)
      if (target === selectedNodeId) ids.add(source)
    }
    return ids
  }, [filteredGraph.edges, selectedNodeId])

  const neighborhoodWallets = useMemo(() => {
    if (!selectedNode || String(selectedNode.kind) !== 'wallet') return []
    const selectedId = String(selectedNode.id)
    const candidates: Array<{ address: string; score: number }> = []
    for (const edge of filteredGraph.edges) {
      if (String(edge.kind) !== 'co_trade') continue
      const source = String(edge.source)
      const target = String(edge.target)
      if (source !== selectedId && target !== selectedId) continue
      const neighborId = source === selectedId ? target : source
      const neighbor = positionedNodeById.get(neighborId)
      if (!neighbor || !neighbor.address) continue
      candidates.push({
        address: neighbor.address,
        score: scoreOfEdge(edge),
      })
    }

    candidates.sort((a, b) => b.score - a.score)
    const out: string[] = []
    const seen = new Set<string>()
    for (const candidate of candidates) {
      const address = String(candidate.address).toLowerCase()
      if (!address || seen.has(address)) continue
      seen.add(address)
      out.push(address)
      if (out.length >= 12) break
    }
    return out
  }, [filteredGraph.edges, positionedNodeById, selectedNode])

  const topVisibleCohorts = useMemo(() => {
    const visibleWalletAddressSet = new Set(
      filteredGraph.nodes
        .filter((node) => String(node.kind) === 'wallet')
        .map((node) => String(node.address || '').trim().toLowerCase())
        .filter((address) => Boolean(address)),
    )

    const rows = rawCohorts
      .map((cohort) => {
        const candidateAddresses = (cohort.visible_wallet_addresses || cohort.wallet_addresses || [])
          .map((address) => String(address).trim().toLowerCase())
          .filter((address) => Boolean(address))
        const visibleAddresses = candidateAddresses.filter((address) => visibleWalletAddressSet.has(address))
        return {
          ...cohort,
          visibleAddresses,
          visibleCount: visibleAddresses.length,
        }
      })
      .filter((cohort) => cohort.visibleCount >= 2)
      .sort((a, b) => {
        const scoreDelta = Number(b.avg_combined_score || 0) - Number(a.avg_combined_score || 0)
        if (scoreDelta !== 0) return scoreDelta
        return b.visibleCount - a.visibleCount
      })

    return rows.slice(0, 8)
  }, [filteredGraph.nodes, rawCohorts])

  const handleAnalyzeWallet = useCallback((address: string, username?: string | null) => {
    if (onAnalyzeWallet) {
      onAnalyzeWallet(address, username || undefined)
      return
    }
    onNavigateToWallet?.(address)
  }, [onAnalyzeWallet, onNavigateToWallet])

  const fitToNodes = useCallback((nodes: PositionedNode[]) => {
    if (!nodes.length || size.width <= 0 || size.height <= 0) return
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const node of nodes) {
      minX = Math.min(minX, node.x - node.r)
      maxX = Math.max(maxX, node.x + node.r)
      minY = Math.min(minY, node.y - node.r)
      maxY = Math.max(maxY, node.y + node.r)
    }
    if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return
    const graphWidth = Math.max(120, maxX - minX)
    const graphHeight = Math.max(120, maxY - minY)
    const viewportWidth = Math.max(200, size.width)
    const viewportHeight = Math.max(200, size.height)
    const fitScale = clamp(Math.min((viewportWidth * 0.86) / graphWidth, (viewportHeight * 0.86) / graphHeight), 0.36, 2.4)
    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    setViewport({
      x: -centerX * fitScale,
      y: -centerY * fitScale,
      scale: fitScale,
    })
  }, [size.height, size.width])

  useEffect(() => {
    autoFitDoneRef.current = false
  }, [nodeLimit])

  useEffect(() => {
    if (!positionedNodes.length) return
    fitToNodes(positionedNodes)
  }, [fitToNodes, focusMode, hideIsolated, neighborhoodDepth, positionedNodes, showClusters, showGroups])

  useEffect(() => {
    if (autoFitDoneRef.current) return
    if (!positionedNodes.length) return
    fitToNodes(positionedNodes)
    autoFitDoneRef.current = true
  }, [fitToNodes, positionedNodes])

  const centerOnNode = useCallback((node: PositionedNode | null) => {
    if (!node) return
    setViewport((current) => ({
      ...current,
      x: -node.x * current.scale,
      y: -node.y * current.scale,
    }))
  }, [])

  const focusSearchMatch = useCallback(() => {
    if (!search.trim()) return
    const first = positionedNodes.find((node) => searchMatches.has(String(node.id)))
    if (!first) {
      setActionMessage('No node matches that query in current filters')
      return
    }
    setSelectedNodeId(String(first.id))
    centerOnNode(first)
    setDockTab('context')
    setDockExpanded(true)
  }, [centerOnNode, positionedNodes, search, searchMatches, setActionMessage])

  const onWheel = useCallback((event: ReactWheelEvent<SVGSVGElement>) => {
    event.preventDefault()
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const px = event.clientX - rect.left - size.width / 2
    const py = event.clientY - rect.top - size.height / 2

    setViewport((current) => {
      const scaleFactor = event.deltaY < 0 ? 1.11 : 0.9
      const nextScale = clamp(current.scale * scaleFactor, 0.35, 3.0)
      const k = nextScale / current.scale
      return {
        x: current.x - px * (k - 1),
        y: current.y - py * (k - 1),
        scale: nextScale,
      }
    })
  }, [size.height, size.width])

  const onMouseDown = useCallback((event: ReactMouseEvent<SVGSVGElement>) => {
    if ((event.target as Element)?.closest('[data-node="true"]')) return
    setIsPanning(true)
    setPanOrigin({ x: event.clientX, y: event.clientY })
  }, [])

  const onMouseMove = useCallback((event: ReactMouseEvent<SVGSVGElement>) => {
    if (!isPanning || !panOrigin) return
    const dx = event.clientX - panOrigin.x
    const dy = event.clientY - panOrigin.y
    setPanOrigin({ x: event.clientX, y: event.clientY })
    setViewport((current) => ({ ...current, x: current.x + dx, y: current.y + dy }))
  }, [isPanning, panOrigin])

  const onMouseUp = useCallback(() => {
    setIsPanning(false)
    setPanOrigin(null)
  }, [])

  const createNeighborGroup = useCallback(() => {
    if (!selectedNode || String(selectedNode.kind) !== 'wallet' || !selectedNode.address) return
    const addresses = [String(selectedNode.address).toLowerCase(), ...neighborhoodWallets]
    const deduped = Array.from(new Set(addresses.filter((address) => Boolean(address))))
    if (deduped.length < 2) {
      setActionMessage('Need at least two connected wallets to create a group')
      return
    }
    const name = `Graph Cohort ${shortAddress(selectedNode.address).replace('...', '')}`
    createGroupMutation.mutate({
      name,
      walletAddresses: deduped,
      seedAddress: selectedNode.address,
    })
  }, [createGroupMutation, neighborhoodWallets, selectedNode, setActionMessage])

  const activeMutations =
    trackWalletMutation.isPending
    || untrackWalletMutation.isPending
    || addToGroupMutation.isPending
    || createGroupMutation.isPending
    || trackGroupMutation.isPending
    || deleteGroupMutation.isPending

  const summary = graphData?.summary
  const selectedWalletAddress = selectedNode && selectedNode.kind === 'wallet' ? String(selectedNode.address || '') : ''
  const selectedGroupNode = selectedNode && selectedNode.kind === 'group' ? selectedNode : null
  const selectedFocusReady = filteredGraph.selectedSeedCount > 0

  const activateFocusMode = useCallback((nextMode: FocusMode) => {
    if (nextMode === 'tracked' && filteredGraph.trackedSeedCount === 0) {
      setActionMessage('No tracked wallets are in this graph snapshot yet.')
      return
    }
    if (nextMode === 'cohort' && (!selectedCohortId || filteredGraph.cohortSeedCount === 0)) {
      setActionMessage('Pick a visible cohort in insights before switching to cohort focus.')
      return
    }
    if (nextMode === 'selected' && !selectedFocusReady) {
      setActionMessage('Select a wallet, group, or cluster to focus its neighborhood.')
      return
    }

    setFocusMode(nextMode)
    if (nextMode === 'all') {
      setActionMessage(`Showing full graph (${filteredGraph.counts.walletsTotal} wallets)`)
      return
    }
    if (nextMode === 'tracked') {
      setActionMessage(`Focused tracked neighborhood (${filteredGraph.trackedSeedCount} seeds, depth ${neighborhoodDepth})`)
      return
    }
    if (nextMode === 'cohort') {
      setActionMessage(`Focused cohort neighborhood (${filteredGraph.cohortSeedCount} seeds, depth ${neighborhoodDepth})`)
      return
    }
    setActionMessage(`Focused selected neighborhood (${filteredGraph.selectedSeedCount} seeds, depth ${neighborhoodDepth})`)
  }, [
    filteredGraph.counts.walletsVisible,
    filteredGraph.counts.walletsTotal,
    filteredGraph.cohortSeedCount,
    filteredGraph.focusSeedCount,
    filteredGraph.selectedSeedCount,
    filteredGraph.trackedSeedCount,
    neighborhoodDepth,
    selectedCohortId,
    selectedFocusReady,
    setActionMessage,
  ])

  const focusSelectedNeighborhood = useCallback(() => {
    if (!selectedFocusReady) {
      setActionMessage('Select a wallet, group, or cluster first.')
      return
    }
    setFocusMode('selected')
    setActionMessage(
      `Focused selected neighborhood (${filteredGraph.selectedSeedCount} seeds, depth ${neighborhoodDepth})`,
    )
  }, [filteredGraph.selectedSeedCount, neighborhoodDepth, selectedFocusReady, setActionMessage])

  const focusCohort = useCallback((cohort: TraderNetworkCohort & { visibleAddresses?: string[]; visibleCount?: number }) => {
    setSelectedCohortId(String(cohort.id))
    setSelectedNodeId(null)
    setFocusMode('cohort')
    setDockTab('stats')
    const visibleCount = Number(cohort.visibleCount || cohort.visibleAddresses?.length || 0)
    setActionMessage(`Focused cohort ${String(cohort.id).slice(0, 6)} (${visibleCount} wallets, depth ${neighborhoodDepth})`)
  }, [neighborhoodDepth, setActionMessage])

  const createGroupFromCohort = useCallback((cohort: TraderNetworkCohort & { visibleAddresses?: string[]; visibleCount?: number }) => {
    const addresses = Array.from(new Set((cohort.visibleAddresses || []).map((address) => String(address).toLowerCase())))
    if (addresses.length < 2) {
      setActionMessage('Need at least two visible wallets in cohort to create a group.')
      return
    }
    const seedAddress = addresses[0]
    createGroupMutation.mutate({
      name: `Cohort ${String(cohort.id).slice(0, 6).toUpperCase()}`,
      walletAddresses: addresses,
      seedAddress,
    })
  }, [createGroupMutation, setActionMessage])

  const focusBadgeLabel = useMemo(() => {
    if (focusMode === 'tracked') {
      return `Tracked focus (${filteredGraph.counts.walletsVisible})`
    }
    if (focusMode === 'selected') {
      return `Selected focus (${filteredGraph.counts.walletsVisible})`
    }
    if (focusMode === 'cohort') {
      return `Cohort focus (${filteredGraph.counts.walletsVisible})`
    }
    return 'All graph'
  }, [filteredGraph.counts.walletsVisible, focusMode])

  return (
    <div className="h-full min-h-0 overflow-hidden rounded-lg border border-border/60 bg-card/30 relative">
      <div ref={containerRef} className="absolute inset-0 overflow-hidden bg-background">
        <div
          className="absolute top-3 left-3 z-20 flex min-w-0 flex-nowrap items-start gap-2"
          style={{ right: dockExpanded ? 418 : 66 }}
        >
          <div className="min-w-0 max-w-full flex-[1_1_0] flex flex-wrap items-center gap-2 rounded-lg border border-border/70 bg-background/90 backdrop-blur-md px-3 py-2 shadow-lg">
            <Badge variant="outline" className="h-6 px-2 text-[11px] border-cyan-500/50 text-cyan-700 dark:text-cyan-300 bg-cyan-500/10">
              <Network className="w-3 h-3 mr-1.5" />
              Trader Network
            </Badge>
            <Badge variant="secondary" className="h-5 text-[10px]">{filteredGraph.counts.walletsVisible} / {filteredGraph.counts.walletsTotal} wallets</Badge>
            <Badge variant="secondary" className="h-5 text-[10px]">{filteredGraph.counts.edgesVisible} edges</Badge>
            <Badge
              variant="secondary"
              className={cn(
                'h-5 max-w-[220px] truncate text-[10px]',
                focusMode === 'tracked' && 'border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200',
                focusMode === 'selected' && 'border border-orange-500/40 bg-orange-500/10 text-orange-700 dark:text-orange-200',
                focusMode === 'cohort' && 'border border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-200',
              )}
            >
              {focusBadgeLabel}
            </Badge>
            <div className="h-5 w-px bg-border/70" />
            <div className="flex items-center gap-1 rounded-md border border-border/70 bg-background/75 p-0.5">
              <button
                type="button"
                onClick={() => activateFocusMode('all')}
                className={cn(
                  'h-6 rounded px-2 text-[10px] font-medium transition-colors',
                  focusMode === 'all'
                    ? 'bg-muted text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
                )}
              >
                All
              </button>
              <button
                type="button"
                onClick={() => activateFocusMode('tracked')}
                className={cn(
                  'h-6 rounded px-2 text-[10px] font-medium transition-colors',
                  focusMode === 'tracked'
                    ? 'bg-emerald-500/25 text-emerald-200'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
                )}
              >
                Tracked
              </button>
              <button
                type="button"
                onClick={() => activateFocusMode('selected')}
                className={cn(
                  'h-6 rounded px-2 text-[10px] font-medium transition-colors',
                  focusMode === 'selected'
                    ? 'bg-orange-500/25 text-orange-200'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
                )}
              >
                Selected
              </button>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[10px] gap-1"
              onClick={() => {
                setSelectedNodeId(null)
                fitToNodes(positionedNodes)
              }}
            >
              <Target className="w-3 h-3" />
              Fit
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[10px] gap-1"
              onClick={() => refetch()}
              disabled={isFetching}
            >
              <RefreshCw className={cn('w-3 h-3', isFetching && 'animate-spin')} />
              Refresh
            </Button>
          </div>
          <div
            className={cn(
              'min-w-0 flex items-center gap-2 rounded-lg border border-border/70 bg-background/92 px-3 py-2 backdrop-blur-md shadow-lg',
              dockExpanded
                ? 'max-w-[380px] flex-[0_1_280px]'
                : 'max-w-[560px] flex-[0_1_420px]',
            )}
          >
            <Search className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Find wallet, username, group"
              className="h-7 min-w-0 flex-1 border-0 bg-transparent p-0 text-xs focus-visible:ring-0"
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-6 shrink-0 px-2 text-[10px]"
              onClick={focusSearchMatch}
              disabled={!search.trim()}
            >
              <Zap className="w-3 h-3 mr-1" />
              Focus
            </Button>
          </div>
        </div>

        {message && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 rounded-md border border-cyan-500/35 bg-cyan-500/12 px-3 py-1.5 text-[11px] text-cyan-800 dark:text-cyan-100">
            {message}
          </div>
        )}

        {isLoading && (
          <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center text-sm text-foreground/80">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Building trader network...
          </div>
        )}

        {!isLoading && positionedNodes.length === 0 && (
          <div className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
            <Network className="w-7 h-7 text-muted-foreground/70" />
            <p>
              {focusMode === 'tracked'
                ? 'No tracked-neighborhood nodes for current filters.'
                : focusMode === 'selected'
                  ? 'No selected-neighborhood nodes for current filters.'
                  : 'No nodes for current filters.'}
            </p>
          </div>
        )}

        <svg
          ref={svgRef}
          className={cn('absolute inset-0 w-full h-full', isPanning ? 'cursor-grabbing' : 'cursor-grab')}
          onWheel={onWheel}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onClick={() => setSelectedNodeId(null)}
        >
          <defs>
            <filter id="tradersNodeGlow" x="-300%" y="-300%" width="700%" height="700%">
              <feGaussianBlur stdDeviation="3.1" result="glow" />
              <feMerge>
                <feMergeNode in="glow" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <g transform={`translate(${size.width / 2 + viewport.x} ${size.height / 2 + viewport.y}) scale(${viewport.scale})`}>
            {filteredGraph.edges.map((edge) => {
              const source = positionedNodeById.get(String(edge.source))
              const target = positionedNodeById.get(String(edge.target))
              if (!source || !target) return null

              const kind = String(edge.kind)
              const score = scoreOfEdge(edge)
              const selected = Boolean(selectedNodeId && (selectedNodeId === source.id || selectedNodeId === target.id))
              const contextualFade = selectedNodeId ? (selected ? 1 : 0.08) : 1
              const dx = target.x - source.x
              const dy = target.y - source.y
              const distance = Math.sqrt(dx * dx + dy * dy) || 1
              const distanceFade = clamp(1.2 - distance / 1300, 0.2, 1)

              const baseOpacity = kind === 'co_trade'
                ? clamp(0.12 + score * 0.72, 0.12, 0.86)
                : kind === 'group_membership'
                  ? 0.52
                  : 0.34
              const strokeWidth = kind === 'co_trade'
                ? clamp(0.8 + score * 2.4, 0.8, 3.4)
                : kind === 'group_membership'
                  ? 1.4
                  : 1.1

              return (
                <path
                  key={edge.id}
                  d={edgePath(source, target, edge.id)}
                  fill="none"
                  stroke={
                    kind === 'co_trade'
                      ? '#0ea5e9'
                      : kind === 'group_membership'
                        ? '#0891b2'
                        : '#ca8a04'
                  }
                  strokeWidth={selected ? strokeWidth + 0.7 : strokeWidth}
                  strokeOpacity={baseOpacity * contextualFade * (kind === 'co_trade' ? distanceFade : 1)}
                  strokeDasharray={
                    kind === 'group_membership'
                      ? '4 3'
                      : kind === 'co_trade' && distance > 760
                        ? '2 2'
                        : undefined
                  }
                />
              )
            })}

            {positionedNodes.map((node) => {
              const nodeId = String(node.id)
              const kind = String(node.kind)
              const isSelected = selectedNodeId === nodeId
              const isSearchMatch = searchMatches.has(nodeId)
              const isAdjacent = selectedNeighborhoodIds.has(nodeId)

              const nodeOpacity = selectedNodeId
                ? (isSelected || isAdjacent ? 1 : 0.22)
                : 1

              const fill = kind === 'wallet'
                ? node.tracked
                  ? '#10b981'
                  : node.in_top_pool
                    ? '#38bdf8'
                    : '#64748b'
                : kind === 'group'
                  ? '#06b6d4'
                  : '#f59e0b'

              const stroke = kind === 'wallet'
                ? '#1e293b'
                : kind === 'group'
                  ? '#0e7490'
                  : '#a16207'

              const showLabel =
                isSelected
                || isSearchMatch
                || kind !== 'wallet'
                || (viewport.scale >= 0.74 && Number(node.co_trade_degree ?? node.degree ?? 0) >= (focusMode === 'all' ? 7 : 5))

              return (
                <g
                  key={nodeId}
                  transform={`translate(${node.x} ${node.y})`}
                  onClick={(event) => {
                    event.stopPropagation()
                    setSelectedNodeId(nodeId)
                    setDockTab('context')
                    setDockExpanded(true)
                  }}
                  onDoubleClick={(event) => {
                    event.stopPropagation()
                    centerOnNode(node)
                  }}
                  data-node="true"
                >
                  <circle
                    r={node.r + (isSelected ? 2 : 0)}
                    fill={fill}
                    fillOpacity={(kind === 'wallet' ? 0.8 : 0.88) * nodeOpacity}
                    stroke={stroke}
                    strokeWidth={isSelected ? 2.4 : isSearchMatch ? 1.9 : 1.2}
                    filter="url(#tradersNodeGlow)"
                  />
                  {showLabel && (
                    <text
                      x={node.r + 6}
                      y={4}
                      className="select-none"
                      style={{
                        fontSize: kind === 'wallet' ? 10 : 11,
                        fill: 'hsl(var(--foreground))',
                        textShadow: '0 1px 2px rgba(148, 163, 184, 0.34)',
                        pointerEvents: 'none',
                        fontWeight: kind === 'wallet' ? 500 : 600,
                        opacity: nodeOpacity,
                      }}
                    >
                      {kind === 'wallet' ? (node.username || shortAddress(node.address)) : node.label}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        </svg>

        <div
          className={cn(
            'absolute inset-y-0 right-0 z-30 border-l border-border/70 bg-background/94 backdrop-blur-md shadow-2xl transition-[width] duration-300',
            dockExpanded ? 'w-[410px] max-w-[98vw]' : 'w-[58px]',
          )}
        >
          <div className="h-full flex">
            <div className="w-[58px] shrink-0 border-r border-border/60 bg-card/55 px-1.5 py-2.5 space-y-2">
              <button
                type="button"
                onClick={() => setDockExpanded((current) => !current)}
                className="w-full h-8 rounded-lg border border-border bg-background text-[11px] font-semibold text-muted-foreground hover:text-foreground hover:bg-muted/45 transition-colors"
                title={dockExpanded ? 'Collapse panel' : 'Expand panel'}
              >
                {dockExpanded ? <ChevronRight className="w-3.5 h-3.5 mx-auto" /> : <ChevronLeft className="w-3.5 h-3.5 mx-auto" />}
              </button>

              <button
                type="button"
                onClick={() => {
                  setDockTab('context')
                  setDockExpanded(true)
                }}
                className={cn(
                  'relative w-full h-9 rounded-lg border text-[10px] font-semibold tracking-wide transition-colors',
                  dockTab === 'context' && dockExpanded
                    ? 'border-blue-500/45 bg-blue-500/15 text-blue-300'
                    : 'border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/45',
                )}
                title="Selected context"
              >
                CTX
                {selectedNode ? <span className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-emerald-400" /> : null}
              </button>

              <button
                type="button"
                onClick={() => {
                  setDockTab('filters')
                  setDockExpanded(true)
                }}
                className={cn(
                  'w-full h-9 rounded-lg border text-[10px] font-semibold tracking-wide transition-colors',
                  dockTab === 'filters' && dockExpanded
                    ? 'border-orange-500/45 bg-orange-500/15 text-orange-300'
                    : 'border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/45',
                )}
                title="Graph filters"
              >
                FLT
              </button>

              <button
                type="button"
                onClick={() => {
                  setDockTab('stats')
                  setDockExpanded(true)
                }}
                className={cn(
                  'w-full h-9 rounded-lg border text-[10px] font-semibold tracking-wide transition-colors',
                  dockTab === 'stats' && dockExpanded
                    ? 'border-cyan-500/45 bg-cyan-500/15 text-cyan-300'
                    : 'border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/45',
                )}
                title="Graph stats"
              >
                STS
              </button>

              <div className="rounded-lg border border-border/70 bg-background/80 px-1 py-1.5 text-center">
                <div className="text-[8px] leading-none text-muted-foreground uppercase">nodes</div>
                <div className="mt-1 text-[10px] leading-none font-semibold text-foreground">{filteredGraph.nodes.length}</div>
              </div>
            </div>

            {dockExpanded ? (
              <div className="flex-1 min-w-0 h-full flex flex-col">
                <div className="shrink-0 border-b border-border/70 px-4 py-3 bg-card/92">
                  <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                    {dockTab === 'context' ? 'Selected Context' : dockTab === 'filters' ? 'Filter Controls' : 'Network Metrics'}
                  </div>
                  <div className="mt-1 flex items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-foreground">
                      {dockTab === 'context'
                        ? selectedNode
                          ? `${selectedNode.kind} detail`
                          : 'Select a node'
                        : dockTab === 'filters'
                          ? 'Declutter and focus graph'
                          : 'Live graph health'}
                    </div>
                    {dockTab === 'context' && selectedNode ? (
                      <button
                        type="button"
                        onClick={() => setSelectedNodeId(null)}
                        className="rounded-md border border-border px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                      >
                        Clear
                      </button>
                    ) : null}
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto p-3 space-y-3">
                  {dockTab === 'context' ? (
                    <>
                      {selectedNode ? (
                        <div className="space-y-3">
                          <div className="rounded-lg border border-border bg-muted/40 px-3 py-2">
                            <div className="flex items-center gap-2">
                              <Badge
                                variant="outline"
                                className={cn(
                                  'h-5 text-[10px] uppercase tracking-wide',
                                  selectedNode.kind === 'wallet'
                                    ? 'border-emerald-500/50 text-emerald-300 bg-emerald-500/10'
                                    : selectedNode.kind === 'group'
                                      ? 'border-cyan-500/50 text-cyan-300 bg-cyan-500/10'
                                      : 'border-amber-500/50 text-amber-200 bg-amber-500/10',
                                )}
                              >
                                {selectedNode.kind}
                              </Badge>
                              <p className="text-sm font-semibold truncate">{selectedNode.label}</p>
                            </div>
                            {selectedNode.kind === 'wallet' ? (
                              <div className="mt-2 space-y-1 text-[11px] text-muted-foreground">
                                <p>{selectedNode.username || shortAddress(selectedNode.address)}</p>
                                <div className="flex flex-wrap gap-1.5">
                                  <Badge variant="secondary" className="h-5 text-[10px]">PnL {formatUsd(selectedNode.total_pnl)}</Badge>
                                  <Badge variant="secondary" className="h-5 text-[10px]">WR {formatPct(selectedNode.win_rate)}</Badge>
                                  <Badge variant="secondary" className="h-5 text-[10px]">Deg {selectedNode.co_trade_degree ?? selectedNode.degree ?? 0}</Badge>
                                </div>
                              </div>
                            ) : null}
                            {selectedNode.kind === 'group' ? (
                              <div className="mt-2 space-y-1 text-[11px] text-muted-foreground">
                                <p>{selectedNode.description || 'Tracked trader group'}</p>
                                <div className="flex flex-wrap gap-1.5">
                                  <Badge variant="secondary" className="h-5 text-[10px]">Members {selectedNode.member_count || 0}</Badge>
                                  <Badge variant="secondary" className="h-5 text-[10px]">Linked {selectedNode.linked_wallet_count || 0}</Badge>
                                </div>
                              </div>
                            ) : null}
                          </div>

                          {selectedNode.kind === 'wallet' && selectedWalletAddress ? (
                            <div className="space-y-2">
                              <div className="grid grid-cols-2 gap-2">
                                <Button
                                  size="sm"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={() => handleAnalyzeWallet(selectedWalletAddress, selectedNode.username)}
                                >
                                  <Activity className="w-3.5 h-3.5" />
                                  Analyze
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={() => {
                                    if (selectedNode.tracked) {
                                      untrackWalletMutation.mutate(selectedWalletAddress)
                                    } else {
                                      trackWalletMutation.mutate({
                                        address: selectedWalletAddress,
                                        label: selectedNode.username || selectedNode.label,
                                      })
                                    }
                                  }}
                                  disabled={trackWalletMutation.isPending || untrackWalletMutation.isPending}
                                >
                                  {selectedNode.tracked ? <UserX className="w-3.5 h-3.5" /> : <UserPlus className="w-3.5 h-3.5" />}
                                  {selectedNode.tracked ? 'Untrack' : 'Track'}
                                </Button>

                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={() => window.open(`https://polymarket.com/profile/${selectedWalletAddress}`, '_blank', 'noopener,noreferrer')}
                                >
                                  <ExternalLink className="w-3.5 h-3.5" />
                                  Profile
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={createNeighborGroup}
                                  disabled={createGroupMutation.isPending}
                                >
                                  <Users className="w-3.5 h-3.5" />
                                  Group Neighbors
                                </Button>
                              </div>

                              <div className="grid grid-cols-2 gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={() => centerOnNode(selectedNode)}
                                >
                                  <Target className="w-3.5 h-3.5" />
                                  Center
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={focusSelectedNeighborhood}
                                >
                                  <Zap className="w-3.5 h-3.5" />
                                  Focus Hood
                                </Button>
                              </div>

                              <div className="rounded-md border border-border bg-muted/30 p-2 space-y-2">
                                <p className="text-[11px] text-muted-foreground">Add wallet to existing group</p>
                                <div className="flex gap-2">
                                  <select
                                    value={selectedGroupId}
                                    onChange={(event) => setSelectedGroupId(event.target.value)}
                                    className="h-8 flex-1 rounded-md border border-border bg-background px-2 text-xs"
                                  >
                                    <option value="">Select group</option>
                                    {groups.map((group) => (
                                      <option key={group.id} value={group.id}>
                                        {group.name} ({group.member_count})
                                      </option>
                                    ))}
                                  </select>
                                  <Button
                                    size="sm"
                                    className="h-8 text-xs gap-1.5"
                                    disabled={!selectedGroupId || addToGroupMutation.isPending}
                                    onClick={() => addToGroupMutation.mutate({ groupId: selectedGroupId, address: selectedWalletAddress })}
                                  >
                                    <Link2 className="w-3.5 h-3.5" />
                                    Add
                                  </Button>
                                </div>
                              </div>

                              {neighborhoodWallets.length > 0 ? (
                                <div className="rounded-md border border-cyan-500/25 bg-cyan-500/8 p-2">
                                  <p className="text-[11px] text-cyan-100 mb-1">Top connected neighbors ({neighborhoodWallets.length})</p>
                                  <div className="flex flex-wrap gap-1">
                                    {neighborhoodWallets.map((address) => (
                                      <Badge key={address} variant="outline" className="text-[10px] h-5 border-cyan-500/35 text-cyan-200">
                                        {shortAddress(address)}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          ) : null}

                          {selectedGroupNode ? (
                            <div className="space-y-2">
                              <div className="grid grid-cols-2 gap-2">
                                <Button
                                  size="sm"
                                  className="h-8 text-xs gap-1.5"
                                  onClick={() => {
                                    if (!selectedGroupNode.group_id) return
                                    trackGroupMutation.mutate(String(selectedGroupNode.group_id))
                                  }}
                                  disabled={trackGroupMutation.isPending || !selectedGroupNode.group_id}
                                >
                                  <Users className="w-3.5 h-3.5" />
                                  Track Members
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs gap-1.5 border-red-500/40 text-red-200 hover:text-red-100 hover:bg-red-500/20"
                                  onClick={() => {
                                    if (!selectedGroupNode.group_id) return
                                    deleteGroupMutation.mutate(String(selectedGroupNode.group_id))
                                  }}
                                  disabled={deleteGroupMutation.isPending || !selectedGroupNode.group_id}
                                >
                                  <UserX className="w-3.5 h-3.5" />
                                  Delete Group
                                </Button>
                              </div>
                              <Button
                                size="sm"
                                variant="outline"
                                className="w-full h-8 text-xs gap-1.5"
                                onClick={focusSelectedNeighborhood}
                              >
                                <Zap className="w-3.5 h-3.5" />
                                Focus Group Members
                              </Button>
                            </div>
                          ) : null}
                        </div>
                      ) : (
                        <div className="rounded-lg border border-border bg-muted/30 px-3 py-4 text-[12px] text-muted-foreground">
                          Select a node in the graph. Double-click any node to center it.
                        </div>
                      )}
                    </>
                  ) : null}

                  {dockTab === 'filters' ? (
                    <div className="space-y-3">
                      <div className="rounded-lg border border-border bg-card/70 p-3 space-y-2">
                        <label className="text-[10px] uppercase tracking-wide text-muted-foreground">Search</label>
                        <Input
                          value={search}
                          onChange={(event) => setSearch(event.target.value)}
                          placeholder="Wallet, username, group"
                          className="h-8 text-xs"
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8 text-xs w-full gap-1.5"
                          onClick={focusSearchMatch}
                          disabled={!search.trim()}
                        >
                          <Search className="w-3.5 h-3.5" />
                          Focus First Match
                        </Button>
                      </div>

                      <div className="rounded-lg border border-border bg-card/70 p-3 space-y-3">
                        <div>
                          <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                            <span>Min co-trade edge score</span>
                            <span className="font-mono text-cyan-300">{edgeThreshold.toFixed(2)}</span>
                          </div>
                          <input
                            type="range"
                            min={MIN_EDGE_SCORE}
                            max={MAX_EDGE_SCORE}
                            step={0.01}
                            value={edgeThreshold}
                            onChange={(event) => setEdgeThreshold(Number(event.target.value))}
                            className="w-full"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                            <span>Max edges per wallet</span>
                            <span className="font-mono text-cyan-300">{connectionCap}</span>
                          </div>
                          <input
                            type="range"
                            min={MIN_CONNECTION_CAP}
                            max={MAX_CONNECTION_CAP}
                            step={1}
                            value={connectionCap}
                            onChange={(event) => setConnectionCap(Number(event.target.value))}
                            className="w-full"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                            <span>Neighborhood depth</span>
                            <span className="font-mono text-cyan-300">{neighborhoodDepth}</span>
                          </div>
                          <input
                            type="range"
                            min={MIN_NEIGHBOR_DEPTH}
                            max={MAX_NEIGHBOR_DEPTH}
                            step={1}
                            value={neighborhoodDepth}
                            onChange={(event) => setNeighborhoodDepth(Number(event.target.value))}
                            className="w-full"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                            <span>Node budget</span>
                          </div>
                          <select
                            value={nodeLimit}
                            onChange={(event) => setNodeLimit(Number(event.target.value))}
                            className="h-8 mt-1 w-full rounded-md border border-border bg-background px-2 text-xs"
                          >
                            {NODE_LIMIT_OPTIONS.map((value) => (
                              <option key={value} value={value}>{value}</option>
                            ))}
                          </select>
                        </div>
                      </div>

                      <div className="rounded-lg border border-border bg-card/70 p-3 space-y-3">
                        <div className="flex items-center justify-between">
                          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Focus + visibility</p>
                          <Filter className="w-3.5 h-3.5 text-muted-foreground" />
                        </div>

                        <div className="grid grid-cols-2 gap-1.5">
                          <button
                            type="button"
                            onClick={() => activateFocusMode('all')}
                            className={cn(
                              'h-8 rounded-md border text-[11px] font-medium transition-colors',
                              focusMode === 'all'
                                ? 'border-slate-300/40 bg-slate-100/10 text-slate-100'
                                : 'border-border bg-background/70 text-muted-foreground hover:text-foreground',
                            )}
                          >
                            All
                          </button>
                          <button
                            type="button"
                            onClick={() => activateFocusMode('tracked')}
                            className={cn(
                              'h-8 rounded-md border text-[11px] font-medium transition-colors',
                              focusMode === 'tracked'
                                ? 'border-emerald-500/60 bg-emerald-500/14 text-emerald-200'
                                : 'border-border bg-background/70 text-muted-foreground hover:text-foreground',
                            )}
                          >
                            Tracked
                          </button>
                          <button
                            type="button"
                            onClick={() => activateFocusMode('selected')}
                            className={cn(
                              'h-8 rounded-md border text-[11px] font-medium transition-colors',
                              focusMode === 'selected'
                                ? 'border-orange-500/60 bg-orange-500/14 text-orange-200'
                                : 'border-border bg-background/70 text-muted-foreground hover:text-foreground',
                            )}
                          >
                            Selected
                          </button>
                          <button
                            type="button"
                            onClick={() => activateFocusMode('cohort')}
                            className={cn(
                              'h-8 rounded-md border text-[11px] font-medium transition-colors',
                              focusMode === 'cohort'
                                ? 'border-fuchsia-500/60 bg-fuchsia-500/14 text-fuchsia-200'
                                : 'border-border bg-background/70 text-muted-foreground hover:text-foreground',
                            )}
                          >
                            Cohort
                          </button>
                        </div>

                        <p className="text-[10px] text-muted-foreground">
                          Seeds in view: tracked {filteredGraph.trackedSeedCount}, selected {filteredGraph.selectedSeedCount}, cohort {filteredGraph.cohortSeedCount}
                        </p>

                        <button
                          type="button"
                          onClick={() => setShowGroups((current) => !current)}
                          className={cn(
                            'w-full h-8 rounded-md border px-2 text-xs text-left transition-colors',
                            showGroups
                              ? 'border-cyan-500/60 bg-cyan-500/14 text-cyan-200'
                              : 'border-border bg-background/70 text-muted-foreground',
                          )}
                        >
                          Show groups ({filteredGraph.counts.groupsVisible} visible)
                        </button>

                        <button
                          type="button"
                          onClick={() => setShowClusters((current) => !current)}
                          className={cn(
                            'w-full h-8 rounded-md border px-2 text-xs text-left transition-colors',
                            showClusters
                              ? 'border-amber-500/60 bg-amber-500/14 text-amber-100'
                              : 'border-border bg-background/70 text-muted-foreground',
                          )}
                        >
                          Show clusters ({filteredGraph.counts.clustersVisible} visible)
                        </button>

                        <button
                          type="button"
                          onClick={() => setHideIsolated((current) => !current)}
                          className={cn(
                            'w-full h-8 rounded-md border px-2 text-xs text-left transition-colors',
                            hideIsolated
                              ? 'border-violet-500/60 bg-violet-500/14 text-violet-200'
                              : 'border-border bg-background/70 text-muted-foreground',
                          )}
                        >
                          Hide isolated wallets
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {dockTab === 'stats' ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="rounded-md border border-border bg-muted/35 px-2 py-1.5">
                          <p className="text-[10px] text-muted-foreground">Visible wallets</p>
                          <p className="text-sm font-semibold">{filteredGraph.counts.walletsVisible}</p>
                        </div>
                        <div className="rounded-md border border-border bg-muted/35 px-2 py-1.5">
                          <p className="text-[10px] text-muted-foreground">Visible edges</p>
                          <p className="text-sm font-semibold">{filteredGraph.counts.edgesVisible}</p>
                        </div>
                        <div className="rounded-md border border-border bg-muted/35 px-2 py-1.5">
                          <p className="text-[10px] text-muted-foreground">Total wallets</p>
                          <p className="text-sm font-semibold">{filteredGraph.counts.walletsTotal}</p>
                        </div>
                        <div className="rounded-md border border-border bg-muted/35 px-2 py-1.5">
                          <p className="text-[10px] text-muted-foreground">Raw co-trade edges</p>
                          <p className="text-sm font-semibold">{filteredGraph.counts.edgesTotal}</p>
                        </div>
                      </div>

                      <div className="rounded-md border border-border bg-muted/35 px-3 py-2">
                        <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Focus state</p>
                        <div className="mt-2 text-[11px] text-muted-foreground space-y-1">
                          <p>Mode: <span className="text-foreground/90 font-medium capitalize">{focusMode}</span></p>
                          <p>Seed count: <span className="text-foreground/90 font-medium">{filteredGraph.focusSeedCount}</span></p>
                          <p>Neighborhood depth: <span className="text-foreground/90 font-medium">{neighborhoodDepth}</span></p>
                        </div>
                      </div>

                      <div className="rounded-md border border-border bg-muted/35 px-3 py-2">
                        <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Cohort intelligence</p>
                        {topVisibleCohorts.length > 0 ? (
                          <div className="mt-2 space-y-2">
                            {topVisibleCohorts.map((cohort) => (
                              <div
                                key={cohort.id}
                                className={cn(
                                  'rounded-md border px-2 py-1.5',
                                  selectedCohortId === String(cohort.id)
                                    ? 'border-fuchsia-500/40 bg-fuchsia-500/8'
                                    : 'border-border bg-background/60',
                                )}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <div className="text-[11px] font-medium text-foreground/90">
                                    Cohort {String(cohort.id).slice(0, 6).toUpperCase()}
                                  </div>
                                  <div className="text-[10px] text-muted-foreground">
                                    {cohort.visibleCount} wallets
                                  </div>
                                </div>
                                <div className="mt-1 text-[10px] text-muted-foreground">
                                  Score {Number(cohort.avg_combined_score || 0).toFixed(2)} | Shared markets {Number(cohort.shared_market_count || 0)}
                                </div>
                                <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className="h-7 text-[10px] gap-1"
                                    onClick={() => focusCohort(cohort)}
                                  >
                                    <Zap className="w-3 h-3" />
                                    Focus
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className="h-7 text-[10px] gap-1"
                                    onClick={() => createGroupFromCohort(cohort)}
                                    disabled={createGroupMutation.isPending}
                                  >
                                    <Users className="w-3 h-3" />
                                    Group
                                  </Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="mt-2 text-[11px] text-muted-foreground">No visible cohorts for current filters.</p>
                        )}
                      </div>

                      <div className="rounded-md border border-border bg-muted/35 px-3 py-2">
                        <p className="text-[10px] uppercase tracking-wide text-muted-foreground">System summary</p>
                        <div className="mt-2 text-[11px] text-muted-foreground space-y-1">
                          <p>Components: <span className="text-foreground/90 font-medium">{summary?.components ?? 0}</span></p>
                          <p>Density: <span className="text-foreground/90 font-medium">{(Number(summary?.density || 0) * 100).toFixed(1)}%</span></p>
                          <p>Generated: <span className="text-foreground/90 font-medium">{graphData?.generated_at ? new Date(graphData.generated_at).toLocaleString() : 'n/a'}</span></p>
                        </div>
                      </div>

                      <div className="rounded-md border border-border bg-muted/35 px-3 py-2">
                        <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Node legend</p>
                        <div className="mt-2 space-y-2 text-[11px]">
                          <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />Tracked wallet</div>
                          <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-sky-400" />Pool wallet</div>
                          <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-cyan-500" />Group node</div>
                          <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-amber-500" />Cluster node</div>
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {activeMutations ? (
                    <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1.5 text-[11px] text-emerald-200 flex items-center gap-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Applying network action...
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}
