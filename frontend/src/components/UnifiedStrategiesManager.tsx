import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code2,
  Copy,
  FlaskConical,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings2,
  Trash2,
  X,
  Zap,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import CodeEditor from './CodeEditor'
import StrategyConfigForm from './StrategyConfigForm'
import {
  getUnifiedStrategies,
  createUnifiedStrategy,
  updateUnifiedStrategy,
  deleteUnifiedStrategy,
  validateUnifiedStrategy,
  reloadUnifiedStrategy,
  getUnifiedStrategyTemplate,
  getValidationOverview,
  overrideValidationStrategy,
  clearValidationStrategyOverride,
  UnifiedStrategy,
} from '../services/api'
import StrategyApiDocsFlyout from './StrategyApiDocsFlyout'
import StrategyBacktestFlyout from './StrategyBacktestFlyout'

// ==================== Helpers ====================

function parseJsonObject(value: string): { value?: Record<string, unknown>; error?: string } {
  try {
    const parsed = JSON.parse(value || '{}')
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { error: 'must be a JSON object' }
    }
    return { value: parsed as Record<string, unknown> }
  } catch (error: any) {
    return { error: error?.message || 'invalid JSON' }
  }
}

function uniqueStrings(values: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of values) {
    const value = String(raw || '').trim().toLowerCase()
    if (!value || seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
}

function normalizeSlug(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, '_')
}

function toStrategyClassNameFromSlug(value: string): string {
  const normalizedSlug = normalizeSlug(value)
  const tokens = normalizedSlug
    .split('_')
    .map((token) => token.trim())
    .filter(Boolean)
  const baseRaw = tokens
    .map((token) => `${token.charAt(0).toUpperCase()}${token.slice(1)}`)
    .join('')
    .replace(/[^A-Za-z0-9_]/g, '')
  const rooted = baseRaw ? (baseRaw.match(/^[A-Za-z_]/) ? baseRaw : `S${baseRaw}`) : 'Custom'
  if (rooted.toLowerCase().endsWith('strategy')) return rooted
  return `${rooted}Strategy`
}

function pythonStringLiteral(value: string): string {
  return `"${String(value || '')
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\r/g, '\\r')
    .replace(/\n/g, '\\n')}"`
}

function renderTemplateSource(
  template: string,
  {
    className,
    strategyName,
    strategyDescription,
    sourceKey,
  }: {
    className: string
    strategyName: string
    strategyDescription: string
    sourceKey: string
  }
): string {
  let out = String(template || '')
  out = out.split('__CLASS_NAME__').join(className)
  out = out.split('__STRATEGY_NAME__').join(strategyName)
  out = out.split('__STRATEGY_DESCRIPTION__').join(strategyDescription)
  out = out.split('__SOURCE_KEY__').join(sourceKey)

  out = out.replace(
    /class\s+[A-Za-z_][A-Za-z0-9_]*\s*\(BaseStrategy\)\s*:/,
    `class ${className}(BaseStrategy):`
  )
  out = out.replace(/^\s*name\s*=\s*".*"$/m, `    name = ${pythonStringLiteral(strategyName)}`)
  out = out.replace(/^\s*description\s*=\s*".*"$/m, `    description = ${pythonStringLiteral(strategyDescription)}`)
  out = out.replace(/^\s*source_key\s*=\s*".*"$/m, `    source_key = ${pythonStringLiteral(sourceKey)}`)
  out = out.replace(/^\s*worker_affinity\s*=\s*".*"$/m, `    worker_affinity = ${pythonStringLiteral(sourceKey)}`)
  return out
}

function inferClassName(sourceCode: string): string | null {
  const classPattern = /class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:/gm
  let match: RegExpExecArray | null = classPattern.exec(sourceCode)
  while (match) {
    const className = String(match[1] || '').trim()
    const bases = String(match[2] || '')
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    const isStrategyClass = bases.some(
      (base) =>
        base === 'BaseStrategy' ||
        base.endsWith('.BaseStrategy') ||
        base === 'BaseTraderStrategy' ||
        base.endsWith('.BaseTraderStrategy')
    )
    if (className && isStrategyClass) {
      return className
    }
    match = classPattern.exec(sourceCode)
  }
  return null
}

function parseAliases(csv: string): string[] {
  return uniqueStrings(
    String(csv || '')
      .split(',')
      .map((item) => item.trim())
  )
}

function errorMessage(error: unknown, fallback: string): string {
  const err = error as any
  const detail = err?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (detail && typeof detail === 'object') {
    if (Array.isArray(detail.errors) && detail.errors.length > 0) {
      return detail.errors.join('; ')
    }
    if (typeof detail.message === 'string') return detail.message
  }
  if (typeof err?.message === 'string' && err.message.trim()) return err.message
  return fallback
}

// ==================== Constants ====================

const STATUS_COLORS: Record<string, string> = {
  loaded: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  active: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  error: 'bg-red-500/15 text-red-400 border-red-500/30',
  unloaded: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  draft: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
}

const SOURCE_LABELS: Record<string, string> = {
  scanner: 'Scanner',
  news: 'News',
  crypto: 'Crypto',
  weather: 'Weather',
  traders: 'Traders',
}

// ==================== Capability Detection ====================

interface Capabilities {
  has_detect: boolean
  has_detect_async: boolean
  has_evaluate: boolean
  has_should_exit: boolean
}

function CapabilityBadges({ capabilities }: { capabilities?: Capabilities }) {
  const caps = capabilities || { has_detect: false, has_detect_async: false, has_evaluate: false, has_should_exit: false }
  return (
    <div className="flex items-center gap-1 mt-1">
      {(caps.has_detect || caps.has_detect_async) && (
        <span className="text-[8px] font-semibold uppercase tracking-wider px-1.5 py-0 rounded bg-amber-500/15 text-amber-400 border border-amber-500/25">
          Detect
        </span>
      )}
      {caps.has_evaluate && (
        <span className="text-[8px] font-semibold uppercase tracking-wider px-1.5 py-0 rounded bg-violet-500/15 text-violet-400 border border-violet-500/25">
          Evaluate
        </span>
      )}
      {caps.has_should_exit && (
        <span className="text-[8px] font-semibold uppercase tracking-wider px-1.5 py-0 rounded bg-rose-500/15 text-rose-400 border border-rose-500/25">
          Exit
        </span>
      )}
    </div>
  )
}

// ==================== Templates ====================

interface NewStrategyTemplate {
  key: string
  label: string
  description: string
  template: string
}

const FULL_TEMPLATE_FALLBACK = [
  'from models import Market, Event, Opportunity',
  'from services.strategies.base import BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck',
  '',
  '',
  'class MyCustomStrategy(BaseStrategy):',
  '    name = "My Custom Strategy"',
  '    description = "Describe what this strategy detects and how it trades"',
  '    source_key = "scanner"',
  '    worker_affinity = "scanner"',
  '',
  '    default_config = {',
  '        "min_edge_percent": 2.0,',
  '        "base_size_usd": 25.0,',
  '        "max_size_usd": 200.0,',
  '        "take_profit_pct": 15.0,',
  '        "stop_loss_pct": 8.0,',
  '    }',
  '',
  '    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:',
  '        opportunities: list[Opportunity] = []',
  '        return opportunities',
  '',
  '    def evaluate(self, signal, context):',
  '        params = context.get("params") or {}',
  '        checks = [DecisionCheck(name="default", passed=True, value=params.get("min_edge_percent"), threshold=0.0)]',
  '        return StrategyDecision(decision="selected", reason="Template default gate", checks=checks)',
  '',
  '    def should_exit(self, position, market_state):',
  '        return ExitDecision(action="hold", reason="Template default hold")',
  '',
].join('\n')

const DETECT_TEMPLATE = [
  'from models import Market, Event, Opportunity',
  'from services.strategies.base import BaseStrategy',
  '',
  '',
  'class __CLASS_NAME__(BaseStrategy):',
  '    name = "__STRATEGY_NAME__"',
  '    description = "__STRATEGY_DESCRIPTION__"',
  '    source_key = "__SOURCE_KEY__"',
  '    worker_affinity = "__SOURCE_KEY__"',
  '',
  '    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:',
  '        opportunities: list[Opportunity] = []',
  '        return opportunities',
  '',
].join('\n')

const EVALUATE_TEMPLATE = [
  'from services.strategies.base import BaseStrategy, StrategyDecision',
  '',
  '',
  'class __CLASS_NAME__(BaseStrategy):',
  '    name = "__STRATEGY_NAME__"',
  '    description = "__STRATEGY_DESCRIPTION__"',
  '    source_key = "__SOURCE_KEY__"',
  '    worker_affinity = "__SOURCE_KEY__"',
  '',
  '    def evaluate(self, signal, context):',
  '        return StrategyDecision(decision="skipped", reason="No custom gating logic configured")',
  '',
].join('\n')

const UNIFIED_MINIMAL_TEMPLATE = [
  'from models import Market, Event, Opportunity',
  'from services.strategies.base import BaseStrategy, StrategyDecision, ExitDecision',
  '',
  '',
  'class __CLASS_NAME__(BaseStrategy):',
  '    name = "__STRATEGY_NAME__"',
  '    description = "__STRATEGY_DESCRIPTION__"',
  '    source_key = "__SOURCE_KEY__"',
  '    worker_affinity = "__SOURCE_KEY__"',
  '',
  '    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:',
  '        opportunities: list[Opportunity] = []',
  '        return opportunities',
  '',
  '    def evaluate(self, signal, context):',
  '        return StrategyDecision(decision="selected", reason="Default minimal template decision")',
  '',
  '    def should_exit(self, position, market_state):',
  '        return ExitDecision(action="hold", reason="Default minimal template hold")',
  '',
].join('\n')

const DEFAULT_NEW_TEMPLATE_KEY = 'full_unified'

function normalizeSourceFilter(value: unknown): string | null {
  const raw = String(value || '').trim().toLowerCase()
  if (!raw) return null
  return raw
}

function normalizeStrategySourceFilter(value: unknown): string {
  return normalizeSourceFilter(value) || 'scanner'
}

function normalizeMetricKey(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

function healthStatusClass(status: string): string {
  if (status === 'demoted') {
    return 'border-red-500/30 bg-red-500/10 text-red-300'
  }
  if (status === 'active') {
    return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
  }
  return 'border-border/40 bg-background/40 text-muted-foreground'
}

// ==================== Main Component ====================

export default function UnifiedStrategiesManager({
  initialSourceFilter,
  onSourceFilterApplied,
}: {
  initialSourceFilter?: string | null
  onSourceFilterApplied?: () => void
}) {
  const queryClient = useQueryClient()

  // UI toggles
  const [showSettings, setShowSettings] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [showHealth, setShowHealth] = useState(false)
  const [showRawJson, setShowRawJson] = useState(false)
  const [showApiDocs, setShowApiDocs] = useState(false)
  const [showBacktest, setShowBacktest] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  // Selection
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null)
  const [draftToken, setDraftToken] = useState<string | null>(null)
  const [newStrategyName, setNewStrategyName] = useState('Custom Strategy')
  const [newStrategySlug, setNewStrategySlug] = useState(() => `custom_${Date.now().toString().slice(-6)}`)
  const [newStrategyDescription, setNewStrategyDescription] = useState('')
  const [newStrategySourceKey, setNewStrategySourceKey] = useState('scanner')
  const [newStrategyTemplateKey, setNewStrategyTemplateKey] = useState(DEFAULT_NEW_TEMPLATE_KEY)
  const [newStrategySlugDirty, setNewStrategySlugDirty] = useState(false)
  const [newStrategyError, setNewStrategyError] = useState<string | null>(null)

  // Editor state
  const [editorSlug, setEditorSlug] = useState('')
  const [editorName, setEditorName] = useState('')
  const [editorDescription, setEditorDescription] = useState('')
  const [editorSourceKey, setEditorSourceKey] = useState('scanner')
  const [editorEnabled, setEditorEnabled] = useState(true)
  const [editorCode, setEditorCode] = useState('')
  const [editorConfigJson, setEditorConfigJson] = useState('{}')
  const [editorSchemaJson, setEditorSchemaJson] = useState('{}')
  const [editorAliasesCsv, setEditorAliasesCsv] = useState('')
  const [editorError, setEditorError] = useState<string | null>(null)
  const [validation, setValidation] = useState<{
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
  } | null>(null)

  // ── Queries ──

  const strategiesQuery = useQuery({
    queryKey: ['unified-strategies'],
    queryFn: () => getUnifiedStrategies(),
    staleTime: 15000,
    refetchInterval: 15000,
  })

  const templateQuery = useQuery({
    queryKey: ['unified-strategy-template'],
    queryFn: getUnifiedStrategyTemplate,
    staleTime: Infinity,
  })

  const strategyTrackerQuery = useQuery({
    queryKey: ['validation-overview', 'strategy-tracker'],
    queryFn: getValidationOverview,
    staleTime: 60000,
    refetchInterval: 60000,
  })

  const catalog = strategiesQuery.data || []

  // ── Derived state ──

  const sourceKeys = useMemo(() => {
    const fromCatalog = catalog.map((s) => normalizeStrategySourceFilter(s.source_key))
    const merged = uniqueStrings(fromCatalog)
    return merged
  }, [catalog])

  useEffect(() => {
    const normalizedSource = normalizeSourceFilter(initialSourceFilter)
    if (!normalizedSource) {
      return
    }
    if (normalizedSource !== 'all' && sourceKeys.length === 0) {
      return
    }

    const normalizedSourceKeys = new Set(sourceKeys)
    const target =
      normalizedSource === 'all' || normalizedSourceKeys.has(normalizedSource) ? normalizedSource : 'all'

    setSourceFilter((current) => {
      if (current === target) return current
      return target
    })
    onSourceFilterApplied?.()
  }, [initialSourceFilter, onSourceFilterApplied, sourceKeys])

  const createSourceKeys = useMemo(
    () => uniqueStrings([...Object.keys(SOURCE_LABELS), ...sourceKeys]),
    [sourceKeys]
  )

  const newStrategyClassName = useMemo(() => {
    const base = toStrategyClassNameFromSlug(newStrategySlug)
    const existing = new Set(
      catalog
        .map((strategy) => String(strategy.class_name || '').trim().toLowerCase())
        .filter(Boolean)
    )
    if (!existing.has(base.toLowerCase())) return base
    let index = 2
    let candidate = `${base}${index}`
    while (existing.has(candidate.toLowerCase())) {
      index += 1
      candidate = `${base}${index}`
    }
    return candidate
  }, [newStrategySlug, catalog])

  const newStrategyTemplates = useMemo<NewStrategyTemplate[]>(
    () => [
      {
        key: DEFAULT_NEW_TEMPLATE_KEY,
        label: 'Full Unified',
        description: 'Detect + evaluate + exit scaffold with complete defaults.',
        template: templateQuery.data?.template || FULL_TEMPLATE_FALLBACK,
      },
      {
        key: 'scanner_detect',
        label: 'Detect Only',
        description: 'Scanner-first template with detect() only.',
        template: DETECT_TEMPLATE,
      },
      {
        key: 'execution_gate',
        label: 'Evaluate Only',
        description: 'Execution-gating template with evaluate() only.',
        template: EVALUATE_TEMPLATE,
      },
      {
        key: 'unified_minimal',
        label: 'Unified Minimal',
        description: 'Lean detect + evaluate + should_exit base.',
        template: UNIFIED_MINIMAL_TEMPLATE,
      },
    ],
    [templateQuery.data?.template]
  )

  const selectedNewTemplate = useMemo(() => {
    return newStrategyTemplates.find((template) => template.key === newStrategyTemplateKey) || newStrategyTemplates[0] || null
  }, [newStrategyTemplateKey, newStrategyTemplates])

  const newStrategyPreviewCode = useMemo(() => {
    if (!selectedNewTemplate) return ''
    const title = String(newStrategyName || '').trim() || 'Custom Strategy'
    const description = String(newStrategyDescription || '').trim() || `${title} strategy`
    return renderTemplateSource(selectedNewTemplate.template, {
      className: newStrategyClassName,
      strategyName: title,
      strategyDescription: description,
      sourceKey: normalizeStrategySourceFilter(newStrategySourceKey),
    })
  }, [selectedNewTemplate, newStrategyName, newStrategyDescription, newStrategySourceKey, newStrategyClassName])

  const grouped = useMemo(() => {
    let rows = [...catalog]
    if (sourceFilter !== 'all') {
      rows = rows.filter((s) => normalizeStrategySourceFilter(s.source_key) === sourceFilter)
    }
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase()
      rows = rows.filter(
        (s) =>
          (s.name || '').toLowerCase().includes(q) ||
          (s.slug || '').toLowerCase().includes(q) ||
          normalizeStrategySourceFilter(s.source_key).includes(q) ||
          (s.description || '').toLowerCase().includes(q) ||
          (s.class_name || '').toLowerCase().includes(q)
      )
    }
    // Group by source_key
    const groups: Record<string, UnifiedStrategy[]> = {}
    for (const s of rows) {
      const key = normalizeStrategySourceFilter(s.source_key)
      if (!groups[key]) groups[key] = []
      groups[key].push(s)
    }
    return groups
  }, [catalog, sourceFilter, searchQuery])

  const flatFiltered = useMemo(() => Object.values(grouped).flat(), [grouped])

  const strategyPerformance = useMemo(() => {
    const out: Record<
      string,
      {
        roi: number | null
        winRate: number | null
        sampleCount: number | null
        realizedPnl: number | null
        terminalCount: number | null
      }
    > = {}
    const strategyAccuracy = strategyTrackerQuery.data?.strategy_accuracy
    if (strategyAccuracy && typeof strategyAccuracy === 'object' && !Array.isArray(strategyAccuracy)) {
      for (const [strategyKey, raw] of Object.entries(strategyAccuracy as Record<string, unknown>)) {
        const key = String(strategyKey || '').trim().toLowerCase()
        if (!key) continue
        const row = raw && typeof raw === 'object' && !Array.isArray(raw)
          ? raw as Record<string, unknown>
          : {}
        const winRate = Number(row.true_positive_rate)
        const resolved = Number(row.resolved)
        out[key] = {
          roi: null,
          winRate: Number.isFinite(winRate) ? winRate * 100 : null,
          sampleCount: Number.isFinite(resolved) ? resolved : null,
          realizedPnl: null,
          terminalCount: null,
        }
      }
    }

    const calibrationByStrategy = strategyTrackerQuery.data?.calibration_90d?.by_strategy
    if (calibrationByStrategy && typeof calibrationByStrategy === 'object' && !Array.isArray(calibrationByStrategy)) {
      for (const [strategyKey, raw] of Object.entries(calibrationByStrategy as Record<string, unknown>)) {
        const key = String(strategyKey || '').trim().toLowerCase()
        if (!key) continue
        const row = raw && typeof raw === 'object' && !Array.isArray(raw)
          ? raw as Record<string, unknown>
          : {}
        const actualRoiMean = Number(row.actual_roi_mean)
        const sampleSize = Number(row.sample_size)
        const existing = out[key]
        out[key] = {
          roi: Number.isFinite(actualRoiMean) ? actualRoiMean : existing?.roi ?? null,
          winRate: existing?.winRate ?? null,
          sampleCount: Number.isFinite(sampleSize) ? sampleSize : existing?.sampleCount ?? null,
          realizedPnl: existing?.realizedPnl ?? null,
          terminalCount: existing?.terminalCount ?? null,
        }
      }
    }

    const orchestratorByStrategy = strategyTrackerQuery.data?.trader_orchestrator_execution_30d?.by_strategy
    if (Array.isArray(orchestratorByStrategy)) {
      for (const raw of orchestratorByStrategy) {
        const row = raw && typeof raw === 'object' && !Array.isArray(raw)
          ? raw as Record<string, unknown>
          : {}
        const key = normalizeMetricKey(row.strategy_type)
        if (!key || key === 'unknown') continue

        const realizedPnl = Number(row.realized_pnl_total)
        const terminalCount = Number(row.terminal)
        const existing = out[key]
        out[key] = {
          roi: existing?.roi ?? null,
          winRate: existing?.winRate ?? null,
          sampleCount:
            existing?.sampleCount ??
            (Number.isFinite(terminalCount) ? terminalCount : null),
          realizedPnl: Number.isFinite(realizedPnl) ? realizedPnl : existing?.realizedPnl ?? null,
          terminalCount: Number.isFinite(terminalCount) ? terminalCount : existing?.terminalCount ?? null,
        }
      }
    }

    return out
  }, [strategyTrackerQuery.data])

  const strategyHealthRows = useMemo(() => {
    const rows = [...(strategyTrackerQuery.data?.strategy_health || [])]
    rows.sort((left, right) => {
      const leftDemoted = left.status === 'demoted'
      const rightDemoted = right.status === 'demoted'
      if (leftDemoted !== rightDemoted) return leftDemoted ? -1 : 1
      return right.sample_size - left.sample_size
    })
    return rows
  }, [strategyTrackerQuery.data?.strategy_health])

  const strategyHealthByKey = useMemo(() => {
    const out: Record<string, (typeof strategyHealthRows)[number]> = {}
    for (const row of strategyHealthRows) {
      const key = normalizeMetricKey(row.strategy_type)
      if (!key) continue
      out[key] = row
    }
    return out
  }, [strategyHealthRows])

  const strategyHealthDemotedCount = useMemo(
    () => strategyHealthRows.filter((row) => row.status === 'demoted').length,
    [strategyHealthRows]
  )

  const selectedStrategy = useMemo(
    () => catalog.find((s) => s.id === selectedStrategyId) || null,
    [selectedStrategyId, catalog]
  )

  const selectedStrategyHealth = useMemo(() => {
    if (!selectedStrategy) return null
    const keys = uniqueStrings([
      normalizeMetricKey(selectedStrategy.slug),
      normalizeMetricKey(selectedStrategy.class_name),
      ...(selectedStrategy.aliases || []).map((alias) => normalizeMetricKey(alias)),
    ])
    for (const key of keys) {
      if (strategyHealthByKey[key]) return strategyHealthByKey[key]
    }
    return null
  }, [selectedStrategy, strategyHealthByKey])

  const inferredClassName = useMemo(() => inferClassName(editorCode), [editorCode])

  // ── Auto-select first ──

  useEffect(() => {
    if (selectedStrategyId) return
    if (draftToken) return
    if (catalog.length > 0) {
      setSelectedStrategyId(catalog[0].id)
    }
  }, [selectedStrategyId, catalog, draftToken])

  // ── Sync editor from selection ──

  useEffect(() => {
    if (!selectedStrategyId) return
    const strategy = catalog.find((s) => s.id === selectedStrategyId)
    if (!strategy) return
    setDraftToken(null)
    setEditorSlug(strategy.slug || '')
    setEditorName(strategy.name || '')
    setEditorDescription(strategy.description || '')
    setEditorSourceKey(normalizeStrategySourceFilter(strategy.source_key))
    setEditorEnabled(Boolean(strategy.enabled))
    setEditorCode(strategy.source_code || '')
    setEditorConfigJson(JSON.stringify(strategy.config || {}, null, 2))
    setEditorSchemaJson(JSON.stringify(strategy.config_schema || {}, null, 2))
    setEditorAliasesCsv((strategy.aliases || []).join(', '))
    setEditorError(null)
    setValidation(null)
  }, [selectedStrategyId, catalog])

  useEffect(() => {
    if (newStrategyTemplates.some((item) => item.key === newStrategyTemplateKey)) return
    if (newStrategyTemplates.length > 0) {
      setNewStrategyTemplateKey(newStrategyTemplates[0].key)
    }
  }, [newStrategyTemplates, newStrategyTemplateKey])

  useEffect(() => {
    if (!showCreateModal || typeof document === 'undefined') return
    const previousOverflow = document.body.style.overflow
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowCreateModal(false)
      }
    }
    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [showCreateModal])

  // ── Refresh helper ──

  const refreshCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-strategies'] })
    queryClient.invalidateQueries({ queryKey: ['strategies'] })
    queryClient.invalidateQueries({ queryKey: ['plugins'] })
    queryClient.invalidateQueries({ queryKey: ['trader-strategies-catalog'] })
  }

  // ── Mutations ──

  const saveMutation = useMutation({
    mutationFn: async () => {
      const parsedConfig = parseJsonObject(editorConfigJson || '{}')
      if (!parsedConfig.value) {
        throw new Error(`Config JSON error: ${parsedConfig.error || 'invalid object'}`)
      }
      const parsedSchema = parseJsonObject(editorSchemaJson || '{}')
      if (!parsedSchema.value) {
        throw new Error(`Schema JSON error: ${parsedSchema.error || 'invalid object'}`)
      }

      const payload = {
        slug: normalizeSlug(editorSlug),
        source_key: normalizeStrategySourceFilter(editorSourceKey),
        name: String(editorName || '').trim(),
        description: editorDescription.trim() || undefined,
        source_code: editorCode,
        config: parsedConfig.value,
        config_schema: parsedSchema.value,
        aliases: parseAliases(editorAliasesCsv),
        enabled: editorEnabled,
      }

      if (!payload.slug) throw new Error('Strategy key is required')
      if (!payload.name) throw new Error('Name is required')

      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (selected) {
        return updateUnifiedStrategy(selected.id, {
          ...payload,
          unlock_system: Boolean(selected.is_system),
        })
      }
      return createUnifiedStrategy(payload)
    },
    onSuccess: (strategy) => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedStrategyId(strategy.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Failed to save strategy'))
    },
  })

  const validateMutation = useMutation({
    mutationFn: async () => validateUnifiedStrategy(editorCode),
    onSuccess: (result) => {
      setValidation({
        valid: Boolean(result.valid),
        class_name: result.class_name || null,
        errors: result.errors || [],
        warnings: result.warnings || [],
      })
      setEditorError(null)
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Validation failed'))
    },
  })

  const reloadMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to reload')
      return reloadUnifiedStrategy(selected.id)
    },
    onSuccess: () => {
      setEditorError(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Reload failed'))
    },
  })

  const cloneMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to clone')
      return createUnifiedStrategy({
        slug: `${selected.slug}_clone_${Date.now().toString().slice(-6)}`,
        source_key: normalizeStrategySourceFilter(selected.source_key),
        name: `${selected.name} (Clone)`,
        description: selected.description || undefined,
        source_code: selected.source_code || '',
        config: selected.config || {},
        config_schema: selected.config_schema || {},
        aliases: selected.aliases || [],
        enabled: true,
      })
    },
    onSuccess: (strategy) => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedStrategyId(strategy.id)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Clone failed'))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to delete')
      return deleteUnifiedStrategy(selected.id)
    },
    onSuccess: () => {
      setEditorError(null)
      setDraftToken(null)
      setSelectedStrategyId(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Delete failed'))
    },
  })

  const overrideStrategyMutation = useMutation({
    mutationFn: ({
      strategyType,
      status,
    }: {
      strategyType: string
      status: 'active' | 'demoted'
    }) => overrideValidationStrategy(strategyType, status),
    onSuccess: () => {
      setEditorError(null)
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Failed to update strategy health override'))
    },
  })

  const clearOverrideMutation = useMutation({
    mutationFn: (strategyType: string) => clearValidationStrategyOverride(strategyType),
    onSuccess: () => {
      setEditorError(null)
      queryClient.invalidateQueries({ queryKey: ['validation-overview'] })
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Failed to clear strategy health override'))
    },
  })

  const healthBusy = overrideStrategyMutation.isPending || clearOverrideMutation.isPending

  const busy =
    saveMutation.isPending ||
    validateMutation.isPending ||
    reloadMutation.isPending ||
    cloneMutation.isPending ||
    deleteMutation.isPending ||
    healthBusy

  // ── New draft ──

  const openCreateModal = () => {
    const nonce = Date.now().toString().slice(-6)
    setNewStrategyName('Custom Strategy')
    setNewStrategySlug(`custom_${nonce}`)
    setNewStrategyDescription('')
    setNewStrategySourceKey(normalizeStrategySourceFilter('scanner'))
    setNewStrategyTemplateKey(DEFAULT_NEW_TEMPLATE_KEY)
    setNewStrategySlugDirty(false)
    setNewStrategyError(null)
    setShowCreateModal(true)
  }

  const createDraftFromModal = () => {
    const normalizedSlug = normalizeSlug(newStrategySlug)
    const normalizedSourceKey = normalizeStrategySourceFilter(newStrategySourceKey)
    const trimmedName = String(newStrategyName || '').trim()
    const trimmedDescription = String(newStrategyDescription || '').trim()
    const selectedTemplate = selectedNewTemplate || newStrategyTemplates[0]

    if (!trimmedName) {
      setNewStrategyError('Strategy name is required.')
      return
    }
    if (!normalizedSlug) {
      setNewStrategyError('Strategy key is required.')
      return
    }
    if (!selectedTemplate) {
      setNewStrategyError('Select a template before creating the draft.')
      return
    }

    const renderedSource = renderTemplateSource(selectedTemplate.template, {
      className: newStrategyClassName,
      strategyName: trimmedName,
      strategyDescription: trimmedDescription || `${trimmedName} strategy`,
      sourceKey: normalizedSourceKey,
    })

    setSelectedStrategyId(null)
    setDraftToken(`draft_${Date.now()}`)
    setEditorSlug(normalizedSlug)
    setEditorSourceKey(normalizedSourceKey)
    setEditorName(trimmedName)
    setEditorDescription(trimmedDescription)
    setEditorEnabled(true)
    setEditorCode(renderedSource)
    setEditorConfigJson('{}')
    setEditorSchemaJson('{"param_fields": []}')
    setEditorAliasesCsv('')
    setEditorError(null)
    setValidation(null)
    setShowSettings(true)
    setShowCreateModal(false)
    setNewStrategyError(null)
  }

  // ── Derived display state ──

  const hasSelection = Boolean(selectedStrategy || draftToken)
  const displayStatus = selectedStrategy?.status || 'draft'
  const statusColor = STATUS_COLORS[displayStatus] || STATUS_COLORS.draft

  // Determine flyout variant based on capabilities
  const flyoutVariant: 'opportunity' | 'trader' = useMemo(() => {
    const code = editorCode || ''
    const hasDetect = /def\s+(detect|detect_async)\s*\(/.test(code) || /BaseWeatherStrategy/.test(code)
    const hasEvaluate = /def\s+evaluate\s*\(/.test(code) || /Base(Weather)?Strategy/.test(code)
    if (hasEvaluate && !hasDetect) return 'trader'
    return 'opportunity'
  }, [editorCode])

  // Parse config_schema for StrategyConfigForm
  const configSchemaFields = useMemo(() => {
    try {
      const schema = JSON.parse(editorSchemaJson || '{}')
      return schema?.param_fields || []
    } catch {
      return []
    }
  }, [editorSchemaJson])

  // ==================== Render ====================

  return (
    <div className="h-full min-h-0 flex gap-3">
      {/* ══════════════════ Left sidebar: Strategy list ══════════════════ */}
      <div className="w-[280px] shrink-0 min-h-0 flex flex-col rounded-lg border border-border/70 bg-card/50">
        {/* Header actions */}
        <div className="shrink-0 p-3 space-y-3 border-b border-border/50">
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              className="h-7 gap-1.5 px-2.5 text-[11px] flex-1"
              onClick={openCreateModal}
              disabled={busy}
            >
              <Plus className="w-3 h-3" />
              New Strategy
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 gap-1 px-2 text-[11px]"
              onClick={() => cloneMutation.mutate()}
              disabled={busy || !selectedStrategy}
              title="Clone selected strategy"
            >
              <Copy className="w-3 h-3" />
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px]"
              onClick={() => refreshCatalog()}
              disabled={busy}
              title="Refresh catalog"
            >
              <RefreshCw className={cn('w-3 h-3', strategiesQuery.isFetching && 'animate-spin')} />
            </Button>
          </div>

          {/* Source filter */}
          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="h-7 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources ({catalog.length})</SelectItem>
              {sourceKeys.map((sk) => (
                <SelectItem key={sk} value={sk}>
                  {SOURCE_LABELS[sk] || sk} ({catalog.filter((s) => normalizeStrategySourceFilter(s.source_key) === sk).length})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search strategies..."
              className="h-7 pl-8 pr-7 text-xs"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 hover:bg-muted/50 rounded"
              >
                <X className="w-3 h-3 text-muted-foreground" />
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            <div className="rounded-md border border-border/60 bg-background/35 px-2 py-1.5">
              <p className="text-[9px] uppercase tracking-wide text-muted-foreground">Health Rows</p>
              <p className="mt-0.5 text-[11px] font-mono">{strategyHealthRows.length}</p>
            </div>
            <div className="rounded-md border border-border/60 bg-background/35 px-2 py-1.5">
              <p className="text-[9px] uppercase tracking-wide text-muted-foreground">Demoted</p>
              <p className={cn('mt-0.5 text-[11px] font-mono', strategyHealthDemotedCount > 0 ? 'text-red-300' : 'text-emerald-300')}>
                {strategyHealthDemotedCount}
              </p>
            </div>
          </div>
        </div>

        {/* Strategy list — grouped by source_key */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-1.5 space-y-1">
            {flatFiltered.length === 0 ? (
              <p className="px-3 py-6 text-xs text-muted-foreground text-center">No strategies found.</p>
            ) : (
              Object.entries(grouped).map(([sourceKey, strategies]) => (
                <div key={sourceKey}>
                  {/* Section divider */}
                  {Object.keys(grouped).length > 1 && (
                    <div className="px-2.5 pt-2.5 pb-1 flex items-center gap-1.5">
                      <div className="flex-1 h-px bg-border/50" />
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-medium shrink-0">
                        {SOURCE_LABELS[sourceKey] || sourceKey} ({strategies.length})
                      </p>
                      <div className="flex-1 h-px bg-border/50" />
                    </div>
                  )}
                  {strategies.map((strategy) => {
                    const active = selectedStrategyId === strategy.id
                    const sColor = STATUS_COLORS[strategy.status] || STATUS_COLORS.draft
                    const metricKeys = uniqueStrings([
                      normalizeMetricKey(strategy.slug),
                      normalizeMetricKey(strategy.class_name),
                      ...(strategy.aliases || []).map((alias) => normalizeMetricKey(alias)),
                    ])
                    const tracker = metricKeys
                      .map((metricKey) => strategyPerformance[metricKey])
                      .find((row) => row != null)
                    const health = metricKeys
                      .map((metricKey) => strategyHealthByKey[metricKey])
                      .find((row) => row != null)
                    const roi = tracker?.roi ?? null
                    const winRate = tracker?.winRate ?? null
                    const sampleCount = tracker?.sampleCount ?? null
                    const realizedPnl = tracker?.realizedPnl ?? null
                    const healthAccuracy = Number(health?.directional_accuracy)
                    const healthMae = Number(health?.mae_roi)
                    return (
                      <button
                        key={strategy.id}
                        type="button"
                        onClick={() => {
                          setDraftToken(null)
                          setSelectedStrategyId(strategy.id)
                        }}
                        className={cn(
                          'w-full rounded-md px-2.5 py-2 text-left transition-all duration-150',
                          active ? 'bg-violet-500/10 ring-1 ring-violet-500/30' : 'hover:bg-muted/50'
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-medium truncate" title={strategy.name}>
                            {strategy.name}
                          </p>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {!strategy.enabled && (
                              <span className="w-1.5 h-1.5 rounded-full bg-zinc-500" title="Disabled" />
                            )}
                            <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0 h-4 border', sColor)}>
                              {strategy.status}
                            </Badge>
                            {health && (
                              <Badge variant="outline" className={cn('text-[9px] px-1.5 py-0 h-4 border', healthStatusClass(health.status))}>
                                {health.status}
                              </Badge>
                            )}
                          </div>
                        </div>
                        <p className="text-[10px] font-mono text-muted-foreground mt-1 truncate">
                          {strategy.slug}
                        </p>
                        <div className="mt-1 flex items-center gap-2 text-[9px]">
                          <span className={cn(
                            'font-mono',
                            realizedPnl == null
                              ? 'text-muted-foreground/60'
                              : realizedPnl >= 0
                                ? 'text-emerald-400'
                                : 'text-red-400'
                          )}>
                            P&L {realizedPnl == null ? '--' : `${realizedPnl >= 0 ? '+' : ''}$${realizedPnl.toFixed(2)}`}
                          </span>
                          <span className={cn(
                            'font-mono',
                            roi == null
                              ? 'text-muted-foreground/60'
                              : roi >= 0
                                ? 'text-emerald-400'
                                : 'text-red-400'
                          )}>
                            ROI {roi == null ? '--' : `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`}
                          </span>
                          <span className="text-muted-foreground/75">
                            Win {winRate == null ? '--' : `${winRate.toFixed(0)}%`}
                          </span>
                          <span className="text-muted-foreground/75">
                            N {sampleCount == null ? '--' : Math.round(sampleCount)}
                          </span>
                          {health && (
                            <span className="text-muted-foreground/75">
                              Acc {Number.isFinite(healthAccuracy) ? `${(healthAccuracy * 100).toFixed(1)}%` : '--'}
                            </span>
                          )}
                          {health && (
                            <span className="text-muted-foreground/75">
                              MAE {Number.isFinite(healthMae) ? healthMae.toFixed(2) : '--'}
                            </span>
                          )}
                        </div>
                        {strategy.capabilities && <CapabilityBadges capabilities={strategy.capabilities} />}
                      </button>
                    )
                  })}
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer stats */}
        <div className="shrink-0 px-3 py-2 border-t border-border/50 text-[10px] text-muted-foreground flex justify-between">
          <span>{flatFiltered.length} strategies</span>
          <span>{catalog.filter((s) => s.enabled).length} enabled</span>
        </div>
      </div>

      {/* ══════════════════ Right panel: Editor ══════════════════ */}
      <div className="flex-1 min-w-0 min-h-0 flex flex-col rounded-lg border border-border/70">
        {!hasSelection ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center space-y-2">
              <Code2 className="w-8 h-8 mx-auto opacity-40" />
              <p className="text-sm">Select a strategy or create a new one</p>
            </div>
          </div>
        ) : (
          <>
            {/* ── Editor toolbar ── */}
            <div className="shrink-0 px-4 py-2.5 border-b border-border/50 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="flex items-center gap-2 min-w-0">
                  <Code2 className="w-4 h-4 text-violet-400 shrink-0" />
                  <span className="text-sm font-medium truncate">
                    {editorName || 'Untitled Strategy'}
                  </span>
                </div>
                <Badge variant="outline" className={cn('text-[10px] shrink-0 border', statusColor)}>
                  {displayStatus}
                </Badge>
                {selectedStrategy?.is_system && (
                  <Badge variant="secondary" className="text-[10px] shrink-0">System</Badge>
                )}
                {selectedStrategy && (
                  <span className="text-[10px] font-mono text-muted-foreground shrink-0">
                    v{selectedStrategy.version}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <div className="flex items-center gap-2 mr-2 pr-2 border-r border-border/50">
                  <span className="text-[10px] text-muted-foreground">Enabled</span>
                  <Switch
                    checked={editorEnabled}
                    onCheckedChange={setEditorEnabled}
                    className="scale-75"
                  />
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => setShowBacktest(true)}
                  disabled={!editorCode.trim()}
                >
                  <FlaskConical className="w-3 h-3" />
                  Backtest
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => setShowApiDocs(true)}
                >
                  <BookOpen className="w-3 h-3" />
                  API Docs
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => validateMutation.mutate()}
                  disabled={busy || !editorCode.trim()}
                >
                  {validateMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Check className="w-3 h-3" />
                  )}
                  Validate
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={() => reloadMutation.mutate()}
                  disabled={busy || !selectedStrategy}
                >
                  {reloadMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Zap className="w-3 h-3" />
                  )}
                  Reload
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px] bg-violet-600 hover:bg-violet-500 text-white"
                  onClick={() => saveMutation.mutate()}
                  disabled={
                    busy ||
                    !editorCode.trim() ||
                    !editorName.trim() ||
                    !editorSlug.trim()
                  }
                >
                  {saveMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Save className="w-3 h-3" />
                  )}
                  Save
                </Button>
                {selectedStrategy && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 px-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    onClick={() => {
                      const message = selectedStrategy.is_system
                        ? `Delete system strategy "${selectedStrategy.name}"? It will be tombstoned and will NOT auto-reseed.`
                        : `Delete "${selectedStrategy.name}"? This cannot be undone.`
                      if (window.confirm(message)) deleteMutation.mutate()
                    }}
                    disabled={busy}
                    title="Delete strategy"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                )}
              </div>
            </div>

            {/* ── Validation / Error banners ── */}
            {validation && (
              <div
                className={cn(
                  'shrink-0 px-4 py-2 text-xs flex items-start gap-2 border-b',
                  validation.valid
                    ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
                    : 'bg-red-500/5 border-red-500/20 text-red-400'
                )}
              >
                {validation.valid ? (
                  <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                ) : (
                  <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                )}
                <div className="min-w-0">
                  <p className="font-medium">
                    {validation.valid ? 'Validation passed' : 'Validation failed'}
                    {validation.class_name && (
                      <span className="font-mono ml-2 opacity-70">{validation.class_name}</span>
                    )}
                  </p>
                  {validation.errors.map((err, i) => (
                    <p key={`e-${i}`} className="font-mono text-[11px] mt-0.5">{err}</p>
                  ))}
                  {validation.warnings.map((warn, i) => (
                    <p key={`w-${i}`} className="font-mono text-[11px] mt-0.5 text-amber-400">{warn}</p>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setValidation(null)}
                  className="shrink-0 p-0.5 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {editorError && (
              <div className="shrink-0 px-4 py-2 text-xs flex items-start gap-2 bg-red-500/5 border-b border-red-500/20 text-red-400">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <p className="min-w-0 flex-1">{editorError}</p>
                <button
                  type="button"
                  onClick={() => setEditorError(null)}
                  className="shrink-0 p-0.5 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {/* ── Main editor content ── */}
            <div className="flex-1 min-h-0 flex flex-col">
              {/* Collapsible Settings */}
              <div className="shrink-0 border-b border-border/50">
                <button
                  type="button"
                  onClick={() => setShowSettings((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showSettings ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Settings</span>
                  <span className="ml-auto font-mono text-[10px] opacity-60">
                    {editorSlug || 'no-key'}
                  </span>
                </button>

                {showSettings && (
                  <div className="px-4 pb-3 space-y-3 animate-in fade-in duration-200">
                    <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Strategy Key</Label>
                        <Input
                          value={editorSlug}
                          onChange={(e) => setEditorSlug(normalizeSlug(e.target.value))}
                          className="mt-1 h-8 text-xs font-mono"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Name</Label>
                        <Input
                          value={editorName}
                          onChange={(e) => setEditorName(e.target.value)}
                          className="mt-1 h-8 text-xs"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source</Label>
                        <Select value={editorSourceKey} onValueChange={setEditorSourceKey}>
                          <SelectTrigger className="mt-1 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {sourceKeys.map((sk) => (
                              <SelectItem key={sk} value={sk}>
                                {SOURCE_LABELS[sk] || sk}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Class Name</Label>
                        <Input
                          value={
                            validation?.class_name ||
                            inferredClassName ||
                            selectedStrategy?.class_name ||
                            'auto-detected'
                          }
                          className="mt-1 h-8 text-xs font-mono"
                          disabled
                        />
                      </div>
                    </div>
                    <div className="grid gap-3 grid-cols-1 xl:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Description</Label>
                        <Input
                          value={editorDescription}
                          onChange={(e) => setEditorDescription(e.target.value)}
                          className="mt-1 h-8 text-xs"
                          placeholder="Describe what this strategy does..."
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Aliases (comma separated)</Label>
                        <Input
                          value={editorAliasesCsv}
                          onChange={(e) => setEditorAliasesCsv(e.target.value)}
                          className="mt-1 h-8 text-xs font-mono"
                          placeholder="alias_one, alias_two"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Collapsible Runtime Config section */}
              <div className="shrink-0 border-b border-border/50">
                <button
                  type="button"
                  onClick={() => setShowConfig((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfig ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Runtime Config</span>
                  {configSchemaFields.length > 0 && (
                    <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 ml-1">
                      {configSchemaFields.length} fields
                    </Badge>
                  )}
                </button>
                {showConfig && (
                  <div className="px-3 pb-3 animate-in fade-in duration-200 space-y-3">
                    {/* Dynamic config form when schema has param_fields */}
                    {configSchemaFields.length > 0 && !showRawJson && (
                      <>
                        <StrategyConfigForm
                          schema={{ param_fields: configSchemaFields }}
                          values={(() => {
                            try {
                              return JSON.parse(editorConfigJson || '{}')
                            } catch {
                              return {}
                            }
                          })()}
                          onChange={(vals) => setEditorConfigJson(JSON.stringify(vals, null, 2))}
                        />
                        <button
                          type="button"
                          onClick={() => setShowRawJson(true)}
                          className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                        >
                          Show Raw JSON
                        </button>
                      </>
                    )}
                    {/* Raw JSON editors — shown when no schema or toggled */}
                    {(configSchemaFields.length === 0 || showRawJson) && (
                      <>
                        <div className="grid gap-3 grid-cols-1 lg:grid-cols-2">
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-[11px] font-medium text-muted-foreground">Config</span>
                              <span className="text-[10px] text-muted-foreground font-mono">JSON</span>
                            </div>
                            <CodeEditor
                              value={editorConfigJson}
                              onChange={setEditorConfigJson}
                              language="json"
                              minHeight="140px"
                              placeholder="{}"
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-[11px] font-medium text-muted-foreground">Config Schema</span>
                              <span className="text-[10px] text-muted-foreground font-mono">JSON</span>
                            </div>
                            <CodeEditor
                              value={editorSchemaJson}
                              onChange={setEditorSchemaJson}
                              language="json"
                              minHeight="140px"
                              placeholder='{"param_fields": []}'
                            />
                          </div>
                        </div>
                        {configSchemaFields.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setShowRawJson(false)}
                            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
                          >
                            Show Config Form
                          </button>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>

              {/* Collapsible Health */}
              <div className="shrink-0 border-b border-border/50">
                <button
                  type="button"
                  onClick={() => setShowHealth((prev) => !prev)}
                  className="w-full px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showHealth ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <Settings2 className="w-3 h-3" />
                  <span>Health</span>
                  <span className="ml-auto font-mono text-[10px] opacity-60">
                    {selectedStrategyHealth?.status || 'untracked'}
                  </span>
                </button>
                {showHealth && (
                  <div className="px-4 pb-3 space-y-3 animate-in fade-in duration-200">
                    {selectedStrategyHealth ? (
                      <>
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge variant="outline" className={cn('text-[10px] border', healthStatusClass(selectedStrategyHealth.status))}>
                            {selectedStrategyHealth.status}
                          </Badge>
                          <span className="text-[10px] text-muted-foreground font-mono">
                            N {selectedStrategyHealth.sample_size}
                          </span>
                          <span className="text-[10px] text-muted-foreground">
                            Acc {Number.isFinite(Number(selectedStrategyHealth.directional_accuracy))
                              ? `${(Number(selectedStrategyHealth.directional_accuracy) * 100).toFixed(1)}%`
                              : '--'}
                          </span>
                          <span className="text-[10px] text-muted-foreground">
                            MAE {Number.isFinite(Number(selectedStrategyHealth.mae_roi))
                              ? Number(selectedStrategyHealth.mae_roi).toFixed(2)
                              : '--'}
                          </span>
                          {selectedStrategyHealth.manual_override && (
                            <Badge variant="outline" className="text-[10px] border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
                              Manual override
                            </Badge>
                          )}
                        </div>

                        <div className="flex flex-wrap items-center gap-1">
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-7 px-2 text-[11px]"
                            disabled={healthBusy || selectedStrategyHealth.status === 'active'}
                            onClick={() =>
                              overrideStrategyMutation.mutate({
                                strategyType: selectedStrategyHealth.strategy_type,
                                status: 'active',
                              })
                            }
                          >
                            Activate
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-7 px-2 text-[11px]"
                            disabled={healthBusy || selectedStrategyHealth.status === 'demoted'}
                            onClick={() =>
                              overrideStrategyMutation.mutate({
                                strategyType: selectedStrategyHealth.strategy_type,
                                status: 'demoted',
                              })
                            }
                          >
                            Demote
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-7 px-2 text-[11px]"
                            disabled={healthBusy || !selectedStrategyHealth.manual_override}
                            onClick={() => clearOverrideMutation.mutate(selectedStrategyHealth.strategy_type)}
                          >
                            Clear
                          </Button>
                        </div>

                        {selectedStrategyHealth.last_reason && (
                          <p className="text-[10px] text-muted-foreground">{selectedStrategyHealth.last_reason}</p>
                        )}
                      </>
                    ) : (
                      <p className="text-[10px] text-muted-foreground">
                        No health telemetry exists yet for key <span className="font-mono">{selectedStrategy?.slug || editorSlug || 'unknown'}</span>.
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* Code editor — takes remaining space */}
              <div className="flex-1 min-h-0 flex flex-col">
                <div className="px-4 py-2 flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-3.5 h-3.5 text-violet-400" />
                    <span className="text-xs font-medium">Source Code</span>
                    <span className="text-[10px] text-muted-foreground font-mono">Python</span>
                  </div>
                  {inferredClassName && (
                    <span className="text-[10px] font-mono text-muted-foreground">
                      class {inferredClassName}
                    </span>
                  )}
                </div>
                <div className="flex-1 min-h-0 px-3 pb-2">
                  <CodeEditor
                    value={editorCode}
                    onChange={setEditorCode}
                    language="python"
                    className="h-full"
                    minHeight="100%"
                    placeholder="Write your strategy source code here..."
                  />
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {showCreateModal && (
            <motion.div
              key="create-strategy-modal"
              className="fixed inset-0 z-[140] flex items-center justify-center p-3 sm:p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="absolute inset-0 bg-black/35 dark:bg-black/75 backdrop-blur-[2px] dark:backdrop-blur-[3px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={() => setShowCreateModal(false)}
                aria-hidden
              />
              <motion.div
                role="dialog"
                aria-modal="true"
                aria-label="Create strategy draft"
                className="relative z-10 w-full max-w-5xl rounded-2xl border border-border/70 bg-gradient-to-br from-background via-background to-violet-50/70 dark:border-violet-500/30 dark:from-card dark:via-card dark:to-violet-950/20 shadow-[0_40px_120px_rgba(0,0,0,0.35)] dark:shadow-[0_40px_120px_rgba(0,0,0,0.55)]"
                initial={{ scale: 0.94, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.98, opacity: 0, y: 12 }}
                transition={{ type: 'spring', stiffness: 250, damping: 28, mass: 0.95 }}
              >
                <div className="border-b border-border/60 px-5 py-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold">Create Strategy</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Configure strategy metadata and choose a scaffold. Code editing starts after draft creation.
                      </p>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-7 w-7 p-0"
                      onClick={() => setShowCreateModal(false)}
                    >
                      <X className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>

                <div className="grid gap-4 p-4 lg:grid-cols-[1.2fr_0.8fr]">
                  <div className="space-y-4">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Name</Label>
                        <Input
                          value={newStrategyName}
                          onChange={(event) => {
                            const value = event.target.value
                            setNewStrategyName(value)
                            setNewStrategyError(null)
                            if (!newStrategySlugDirty) {
                              const autoSlug = normalizeSlug(value)
                              setNewStrategySlug(autoSlug || `custom_${Date.now().toString().slice(-6)}`)
                            }
                          }}
                          className="mt-1 h-9 text-xs"
                          placeholder="My Strategy"
                        />
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Strategy Key</Label>
                        <Input
                          value={newStrategySlug}
                          onChange={(event) => {
                            setNewStrategySlugDirty(true)
                            setNewStrategySlug(normalizeSlug(event.target.value))
                            setNewStrategyError(null)
                          }}
                          className="mt-1 h-9 text-xs font-mono"
                          placeholder="my_strategy"
                        />
                      </div>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Source</Label>
                        <Select
                          value={newStrategySourceKey}
                          onValueChange={(value) => {
                            setNewStrategySourceKey(value)
                            setNewStrategyError(null)
                          }}
                        >
                          <SelectTrigger className="mt-1 h-9 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {createSourceKeys.map((sourceKey) => (
                              <SelectItem key={sourceKey} value={sourceKey}>
                                {SOURCE_LABELS[sourceKey] || sourceKey}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-[11px] text-muted-foreground">Class Name (auto from key)</Label>
                        <Input value={newStrategyClassName} className="mt-1 h-9 text-xs font-mono" disabled />
                      </div>
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">Description</Label>
                      <textarea
                        value={newStrategyDescription}
                        onChange={(event) => {
                          setNewStrategyDescription(event.target.value)
                          setNewStrategyError(null)
                        }}
                        className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-xs leading-5 text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        rows={3}
                        placeholder="Describe what this strategy does."
                      />
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">Template</Label>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        {newStrategyTemplates.map((template) => {
                          const active = template.key === (selectedNewTemplate?.key || '')
                          return (
                            <button
                              key={template.key}
                              type="button"
                              onClick={() => {
                                setNewStrategyTemplateKey(template.key)
                                setNewStrategyError(null)
                              }}
                              className={cn(
                                'rounded-lg border px-3 py-2 text-left transition-all',
                                active
                                  ? 'border-violet-500/70 bg-violet-500/10 shadow-[0_0_0_1px_rgba(139,92,246,0.22)]'
                                  : 'border-border/70 bg-card/50 hover:border-border'
                              )}
                            >
                              <p className="text-xs font-semibold">{template.label}</p>
                              <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">{template.description}</p>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  </div>

                  <div className="min-h-0 rounded-xl border border-border/60 bg-muted/30 dark:bg-black/30">
                    <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
                      <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Template Preview</span>
                      <span className="text-[10px] font-mono text-muted-foreground">Read-only</span>
                    </div>
                    <div className="max-h-[420px] overflow-auto px-3 py-2">
                      <pre className="whitespace-pre-wrap text-[11px] leading-5 text-muted-foreground font-mono">{newStrategyPreviewCode}</pre>
                    </div>
                  </div>
                </div>

                {newStrategyError && (
                  <div className="mx-4 mb-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                    {newStrategyError}
                  </div>
                )}

                <div className="flex items-center justify-end gap-2 border-t border-border/60 px-4 py-3">
                  <Button type="button" variant="outline" size="sm" onClick={() => setShowCreateModal(false)}>
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    className="bg-violet-600 hover:bg-violet-500 text-white"
                    onClick={createDraftFromModal}
                    disabled={busy || !newStrategyName.trim() || !normalizeSlug(newStrategySlug)}
                  >
                    Continue to Editor
                  </Button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      {/* ── Flyouts ── */}
      <StrategyApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} variant={flyoutVariant} />
      <StrategyBacktestFlyout
        open={showBacktest}
        onOpenChange={setShowBacktest}
        sourceCode={editorCode}
        slug={editorSlug || '_backtest_preview'}
        config={(() => { try { return JSON.parse(editorConfigJson || '{}') } catch { return {} } })()}
        variant={flyoutVariant}
      />
    </div>
  )
}
