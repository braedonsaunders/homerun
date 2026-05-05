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
  Code2,
  Copy,
  Loader2,
  MoreHorizontal,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings2,
  Sparkles,
  Trash2,
  Wand2,
  X,
  Zap,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover'
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
  getUnifiedStrategyVersions,
  getValidationOverview,
  overrideValidationStrategy,
  restoreUnifiedStrategyVersion,
  clearValidationStrategyOverride,
  generateAIStrategyDraft,
  modifyAIStrategyCode,
  UnifiedStrategy,
  UnifiedStrategyVersion,
  AIStrategyDraftGenerationResponse,
  AIModifyStrategyCodeResponse,
} from '../services/api'
import StrategyApiDocsFlyout from './StrategyApiDocsFlyout'
// Backtesting was moved out of the strategy editor into Strategies →
// Research where it lives alongside the code-evolution autoresearcher.
// The flyout component is no longer mounted from here.

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
  return out
}

function applyStrategyMetadataToSource(
  sourceCode: string,
  {
    strategyName,
    strategyDescription,
    sourceKey,
  }: {
    strategyName: string
    strategyDescription: string
    sourceKey: string
  }
): string {
  let out = String(sourceCode || '')
  out = out.replace(/^\s*name\s*=\s*".*"$/m, `    name = ${pythonStringLiteral(strategyName)}`)
  out = out.replace(/^\s*description\s*=\s*".*"$/m, `    description = ${pythonStringLiteral(strategyDescription)}`)
  out = out.replace(/^\s*source_key\s*=\s*".*"$/m, `    source_key = ${pythonStringLiteral(sourceKey)}`)
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

function CalibrationSparkline({
  trend,
}: {
  trend: Array<{ bucket_start: string; sample_size: number; mae_roi: number; directional_accuracy: number }>
}) {
  if (trend.length < 2) return null
  const w = 600
  const h = 60
  const acc = trend.map((t) => Math.max(0, Math.min(1, t.directional_accuracy || 0)))
  const mae = trend.map((t) => Math.max(0, t.mae_roi || 0))
  const maeMax = Math.max(...mae, 1)
  const stepX = w / Math.max(1, trend.length - 1)
  const accPath = acc.map((v, i) => {
    const x = i * stepX
    const y = h - v * (h - 4) - 2
    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
  const maePath = mae.map((v, i) => {
    const x = i * stepX
    const y = h - (v / maeMax) * (h - 4) - 2
    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
  const lastAcc = acc[acc.length - 1]
  const lastMae = mae[mae.length - 1]
  const firstBucket = trend[0]?.bucket_start?.slice(0, 10) || ''
  const lastBucket = trend[trend.length - 1]?.bucket_start?.slice(0, 10) || ''
  return (
    <div className="rounded-md border border-border/40 bg-card/30 p-2 space-y-1">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-14" preserveAspectRatio="none">
        {/* 50% reference line for accuracy */}
        <line
          x1="0"
          x2={w}
          y1={h / 2}
          y2={h / 2}
          stroke="rgba(148,163,184,0.25)"
          strokeDasharray="2 3"
          strokeWidth="0.5"
        />
        <path d={maePath} stroke="rgb(248 113 113)" strokeWidth="1.2" fill="none" opacity="0.6" />
        <path d={accPath} stroke="rgb(34 211 238)" strokeWidth="1.4" fill="none" />
      </svg>
      <div className="flex items-center justify-between text-[9px] font-mono text-muted-foreground">
        <span>{firstBucket}</span>
        <span className="flex items-center gap-3">
          <span>
            <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-1 align-middle" />
            acc {(lastAcc * 100).toFixed(1)}%
          </span>
          <span>
            <span className="inline-block w-2 h-2 rounded-full bg-red-400 mr-1 align-middle opacity-60" />
            mae {lastMae.toFixed(2)}
          </span>
        </span>
        <span>{lastBucket}</span>
      </div>
    </div>
  )
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
  manual: 'Manual',
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
  onOpenCopilot: _onOpenCopilot,
}: {
  initialSourceFilter?: string | null
  onSourceFilterApplied?: () => void
  onOpenCopilot?: (options?: {
    contextType?: string
    contextId?: string
    label?: string
    prompt?: string
    autoSend?: boolean
  }) => void
}) {
  const queryClient = useQueryClient()

  // UI toggles
  // Editor body subtab — replaces 4 separate collapsible booleans with
  // one horizontal tab strip. Source Code is the default since editing
  // strategy code is the primary task.
  // Autoresearch is its own subview under Strategies → Research now;
  // the editor doesn't need a tab for it anymore.
  type EditorTab = 'code' | 'settings' | 'config' | 'health'
  const [editorTab, setEditorTab] = useState<EditorTab>('code')
  const showSettings = editorTab === 'settings'
  const showConfig = editorTab === 'config'
  const showHealth = editorTab === 'health'
  // Setter shim for the few existing call-sites that toggled the
  // settings collapsible from outside the editor body (e.g. the deep-link
  // in setShowSettings(true) below). Other tab navigation now goes
  // through ``setEditorTab`` directly via the horizontal tab strip.
  const setShowSettings = (next: boolean | ((prev: boolean) => boolean)) => {
    const v = typeof next === 'function' ? next(showSettings) : next
    if (v) setEditorTab('settings')
  }
  const [showRawJson, setShowRawJson] = useState(false)
  const [showApiDocs, setShowApiDocs] = useState(false)
  // Backtest flyout removed; backtesting now lives only in Strategies →
  // Research alongside the code-evolution autoresearcher.
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showModifyModal, setShowModifyModal] = useState(false)

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
  const [newStrategyAiPrompt, setNewStrategyAiPrompt] = useState('')
  const [newStrategyAiDraft, setNewStrategyAiDraft] = useState<AIStrategyDraftGenerationResponse | null>(null)
  const [newStrategyUseAiDraft, setNewStrategyUseAiDraft] = useState(false)
  const [modifyStrategyAiPrompt, setModifyStrategyAiPrompt] = useState('')
  const [modifyStrategyAiDraft, setModifyStrategyAiDraft] = useState<AIModifyStrategyCodeResponse | null>(null)
  const [modifyStrategyAiError, setModifyStrategyAiError] = useState<string | null>(null)

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
  const [selectedVersionForRestore, setSelectedVersionForRestore] = useState<string>('')
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

  const strategyVersionsQuery = useQuery({
    queryKey: ['unified-strategy-versions', selectedStrategyId],
    enabled: Boolean(selectedStrategyId),
    queryFn: () => getUnifiedStrategyVersions(String(selectedStrategyId), { limit: 250 }),
    staleTime: 10000,
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
    if (newStrategyUseAiDraft && newStrategyAiDraft?.source_code) {
      const title = String(newStrategyName || '').trim() || 'Custom Strategy'
      const description = String(newStrategyDescription || '').trim() || `${title} strategy`
      return applyStrategyMetadataToSource(String(newStrategyAiDraft.source_code || ''), {
        strategyName: title,
        strategyDescription: description,
        sourceKey: normalizeStrategySourceFilter(newStrategySourceKey),
      })
    }
    if (!selectedNewTemplate) return ''
    const title = String(newStrategyName || '').trim() || 'Custom Strategy'
    const description = String(newStrategyDescription || '').trim() || `${title} strategy`
    return renderTemplateSource(selectedNewTemplate.template, {
      className: newStrategyClassName,
      strategyName: title,
      strategyDescription: description,
      sourceKey: normalizeStrategySourceFilter(newStrategySourceKey),
    })
  }, [
    selectedNewTemplate,
    newStrategyName,
    newStrategyDescription,
    newStrategySourceKey,
    newStrategyClassName,
    newStrategyUseAiDraft,
    newStrategyAiDraft?.source_code,
  ])

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
    type ModeBucket = {
      total: number
      terminalCount: number | null
      realizedPnl: number | null
    }
    const out: Record<
      string,
      {
        roi: number | null
        winRate: number | null
        sampleCount: number | null
        // Aggregated (live + shadow) — kept for back-compat with older
        // callers / lists. The Health subtab now reads ``modes.live``
        // and ``modes.shadow`` separately so users can tell real
        // venue trades apart from simulated/paper rows.
        realizedPnl: number | null
        terminalCount: number | null
        modes: {
          live: ModeBucket | null
          shadow: ModeBucket | null
          other: ModeBucket | null
        }
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
          modes: { live: null, shadow: null, other: null },
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
          modes: existing?.modes ?? { live: null, shadow: null, other: null },
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

        // Pull the per-mode breakdown so live and shadow rows stay
        // separate. ``modes`` is a dict keyed by mode label ("live",
        // "shadow", possibly "unknown"); each value carries its own
        // terminal/realized_pnl_total counters.
        const modesRaw = row.modes
        const modeBuckets: { live: ModeBucket | null; shadow: ModeBucket | null; other: ModeBucket | null } = {
          live: null,
          shadow: null,
          other: null,
        }
        if (modesRaw && typeof modesRaw === 'object' && !Array.isArray(modesRaw)) {
          for (const [modeKey, modeVal] of Object.entries(modesRaw as Record<string, unknown>)) {
            if (!modeVal || typeof modeVal !== 'object' || Array.isArray(modeVal)) continue
            const modeRow = modeVal as Record<string, unknown>
            const total = Number(modeRow.total)
            const terminal = Number(modeRow.terminal)
            const pnl = Number(modeRow.realized_pnl_total)
            const bucket: ModeBucket = {
              total: Number.isFinite(total) ? total : 0,
              terminalCount: Number.isFinite(terminal) ? terminal : null,
              realizedPnl: Number.isFinite(pnl) ? pnl : null,
            }
            const normalized = String(modeKey || '').trim().toLowerCase()
            if (normalized === 'live') {
              modeBuckets.live = bucket
            } else if (normalized === 'shadow' || normalized === 'simulated' || normalized === 'paper') {
              modeBuckets.shadow = bucket
            } else {
              modeBuckets.other = bucket
            }
          }
        }

        const existing = out[key]
        out[key] = {
          roi: existing?.roi ?? null,
          winRate: existing?.winRate ?? null,
          sampleCount:
            existing?.sampleCount ??
            (Number.isFinite(terminalCount) ? terminalCount : null),
          realizedPnl: Number.isFinite(realizedPnl) ? realizedPnl : existing?.realizedPnl ?? null,
          terminalCount: Number.isFinite(terminalCount) ? terminalCount : existing?.terminalCount ?? null,
          modes: modeBuckets,
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
  const strategyVersions = useMemo<UnifiedStrategyVersion[]>(
    () => (Array.isArray(strategyVersionsQuery.data) ? strategyVersionsQuery.data : []),
    [strategyVersionsQuery.data]
  )
  const latestVersionRow = strategyVersions.length > 0 ? strategyVersions[0] : null
  const selectedRestoreVersion = Number(selectedVersionForRestore || 0)
  const selectedRestoreVersionValid = Number.isFinite(selectedRestoreVersion) && selectedRestoreVersion > 0
  const restoreVersionIsCurrent = Boolean(
    selectedStrategy && selectedRestoreVersionValid && selectedRestoreVersion === Number(selectedStrategy.version || 0)
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

  // Per-strategy performance aggregates pulled from the validation
  // overview (strategy_accuracy + calibration_90d.by_strategy +
  // trader_orchestrator_execution_30d.by_strategy).
  const selectedStrategyPerformance = useMemo(() => {
    if (!selectedStrategy) return null
    const keys = uniqueStrings([
      normalizeMetricKey(selectedStrategy.slug),
      normalizeMetricKey(selectedStrategy.class_name),
      ...(selectedStrategy.aliases || []).map((alias) => normalizeMetricKey(alias)),
    ])
    for (const key of keys) {
      const perf = strategyPerformance[key]
      if (perf) return perf
    }
    return null
  }, [selectedStrategy, strategyPerformance])

  // Raw strategy_accuracy row for the totals (total / resolved /
  // profitable / unprofitable). Lives next to ``strategy_accuracy`` on
  // the validation overview.
  const selectedStrategyAccuracy = useMemo(() => {
    if (!selectedStrategy) return null
    const accuracyMap = strategyTrackerQuery.data?.strategy_accuracy
    if (!accuracyMap || typeof accuracyMap !== 'object' || Array.isArray(accuracyMap)) return null
    const keys = uniqueStrings([
      normalizeMetricKey(selectedStrategy.slug),
      normalizeMetricKey(selectedStrategy.class_name),
      ...(selectedStrategy.aliases || []).map((alias) => normalizeMetricKey(alias)),
    ])
    for (const key of keys) {
      const row = (accuracyMap as Record<string, unknown>)[key]
      if (row && typeof row === 'object' && !Array.isArray(row)) {
        return row as Record<string, unknown>
      }
    }
    return null
  }, [selectedStrategy, strategyTrackerQuery.data?.strategy_accuracy])

  // Calibration rolling history specific to this strategy — used to draw
  // a small accuracy + MAE sparkline so operators can see whether health
  // is trending toward demotion or recovery.
  const selectedStrategyCalibrationTrend = useMemo(() => {
    if (!selectedStrategy) return [] as Array<{ bucket_start: string; sample_size: number; mae_roi: number; directional_accuracy: number; strategy_type?: string }>
    const trend = (strategyTrackerQuery.data as Record<string, unknown> | undefined)?.calibration_trend_90d
    if (!Array.isArray(trend)) return []
    const keys = uniqueStrings([
      normalizeMetricKey(selectedStrategy.slug),
      normalizeMetricKey(selectedStrategy.class_name),
      ...(selectedStrategy.aliases || []).map((alias) => normalizeMetricKey(alias)),
    ])
    return (trend as Array<Record<string, unknown>>)
      .filter((row) => {
        const t = normalizeMetricKey((row as any).strategy_type)
        return t && keys.includes(t)
      })
      .map((row) => ({
        bucket_start: String((row as any).bucket_start || ''),
        sample_size: Number((row as any).sample_size || 0),
        mae_roi: Number((row as any).mae_roi || 0),
        directional_accuracy: Number((row as any).directional_accuracy || 0),
        strategy_type: String((row as any).strategy_type || ''),
      }))
      .sort((a, b) => a.bucket_start.localeCompare(b.bucket_start))
  }, [selectedStrategy, strategyTrackerQuery.data])

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
    setSelectedVersionForRestore(String(strategy.version || ''))
    setEditorError(null)
    setValidation(null)
  }, [selectedStrategyId, catalog])

  useEffect(() => {
    if (!selectedStrategy) return
    if (!selectedVersionForRestore) {
      setSelectedVersionForRestore(String(selectedStrategy.version || ''))
      return
    }
    const hasSelected = strategyVersions.some((versionRow) => String(versionRow.version) === selectedVersionForRestore)
    if (!hasSelected) {
      setSelectedVersionForRestore(String(selectedStrategy.version || ''))
    }
  }, [selectedStrategy, selectedVersionForRestore, strategyVersions])

  useEffect(() => {
    if (newStrategyTemplates.some((item) => item.key === newStrategyTemplateKey)) return
    if (newStrategyTemplates.length > 0) {
      setNewStrategyTemplateKey(newStrategyTemplates[0].key)
    }
  }, [newStrategyTemplates, newStrategyTemplateKey])

  useEffect(() => {
    if ((!showCreateModal && !showModifyModal) || typeof document === 'undefined') return
    const previousOverflow = document.body.style.overflow
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (showModifyModal) {
          setShowModifyModal(false)
        } else {
          setShowCreateModal(false)
        }
      }
    }
    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [showCreateModal, showModifyModal])

  // ── Refresh helper ──

  const refreshCatalog = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-strategies'] })
    queryClient.invalidateQueries({ queryKey: ['unified-strategy-versions'] })
    queryClient.invalidateQueries({ queryKey: ['strategy-experiments'] })
    queryClient.invalidateQueries({ queryKey: ['strategies'] })
    queryClient.invalidateQueries({ queryKey: ['plugins'] })
    queryClient.invalidateQueries({ queryKey: ['trader-strategies-catalog'] })
    queryClient.invalidateQueries({ queryKey: ['trader-config-schema'] })
    queryClient.invalidateQueries({ queryKey: ['trader-sources'] })
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

  const restoreVersionMutation = useMutation({
    mutationFn: async () => {
      const selected = catalog.find((s) => s.id === selectedStrategyId)
      if (!selected) throw new Error('Select a strategy to restore')
      const version = Number(selectedVersionForRestore || 0)
      if (!Number.isFinite(version) || version <= 0) {
        throw new Error('Select a valid version to restore')
      }
      return restoreUnifiedStrategyVersion(selected.id, version, {
        reason: 'manual_restore_from_ui',
      })
    },
    onSuccess: () => {
      setEditorError(null)
      refreshCatalog()
    },
    onError: (error: unknown) => {
      setEditorError(errorMessage(error, 'Failed to restore version'))
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

  const generateDraftMutation = useMutation({
    mutationFn: async () => {
      const prompt = String(newStrategyAiPrompt || '').trim()
      if (!prompt) throw new Error('Describe the strategy you want to generate.')
      return generateAIStrategyDraft({
        description: prompt,
        source_key: normalizeStrategySourceFilter(newStrategySourceKey),
      })
    },
    onSuccess: (draft) => {
      setNewStrategyAiDraft(draft)
      setNewStrategyUseAiDraft(true)
      setNewStrategyName(String(draft.name || 'Custom Strategy').trim() || 'Custom Strategy')
      setNewStrategySlug(normalizeSlug(draft.slug || 'custom_strategy'))
      setNewStrategySlugDirty(true)
      setNewStrategySourceKey(normalizeStrategySourceFilter(draft.source_key || newStrategySourceKey))
      setNewStrategyDescription(String(draft.description || '').trim())
      setNewStrategyError(null)
    },
    onError: (error: unknown) => {
      setNewStrategyError(errorMessage(error, 'AI generation failed'))
    },
  })

  const modifyStrategyMutation = useMutation({
    mutationFn: async () => {
      const instruction = String(modifyStrategyAiPrompt || '').trim()
      if (!instruction) throw new Error('Describe how the AI should modify this strategy.')
      if (!editorCode.trim()) throw new Error('Strategy source code is empty.')

      const parsedConfig = parseJsonObject(editorConfigJson || '{}')
      if (!parsedConfig.value) {
        throw new Error(`Config JSON error: ${parsedConfig.error || 'invalid object'}`)
      }
      const parsedSchema = parseJsonObject(editorSchemaJson || '{}')
      if (!parsedSchema.value) {
        throw new Error(`Schema JSON error: ${parsedSchema.error || 'invalid object'}`)
      }

      const normalizedSlug = normalizeSlug(editorSlug || selectedStrategy?.slug || 'strategy')
      const normalizedName = String(editorName || selectedStrategy?.name || normalizedSlug).trim()
      if (!normalizedSlug) throw new Error('Strategy key is required before AI modification.')
      if (!normalizedName) throw new Error('Strategy name is required before AI modification.')

      return modifyAIStrategyCode({
        instruction,
        strategy_id: selectedStrategy?.id,
        slug: normalizedSlug,
        name: normalizedName,
        source_key: normalizeStrategySourceFilter(editorSourceKey || selectedStrategy?.source_key || 'scanner'),
        description: String(editorDescription || '').trim() || undefined,
        source_code: editorCode,
        config: parsedConfig.value,
        config_schema: parsedSchema.value,
        aliases: parseAliases(editorAliasesCsv),
      })
    },
    onSuccess: (draft) => {
      setModifyStrategyAiDraft(draft)
      setModifyStrategyAiError(null)
    },
    onError: (error: unknown) => {
      setModifyStrategyAiError(errorMessage(error, 'AI modification failed'))
    },
  })

  const healthBusy = overrideStrategyMutation.isPending || clearOverrideMutation.isPending

  const busy =
    generateDraftMutation.isPending ||
    modifyStrategyMutation.isPending ||
    saveMutation.isPending ||
    validateMutation.isPending ||
    reloadMutation.isPending ||
    restoreVersionMutation.isPending ||
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
    setNewStrategyAiPrompt('')
    setNewStrategyAiDraft(null)
    setNewStrategyUseAiDraft(false)
    setShowCreateModal(true)
  }

  const createDraftFromModal = () => {
    const normalizedSlug = normalizeSlug(newStrategySlug)
    const normalizedSourceKey = normalizeStrategySourceFilter(newStrategySourceKey)
    const trimmedName = String(newStrategyName || '').trim()
    const trimmedDescription = String(newStrategyDescription || '').trim()
    const selectedTemplate = selectedNewTemplate || newStrategyTemplates[0]
    const useAiDraft = Boolean(newStrategyUseAiDraft && newStrategyAiDraft)

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

    const aiDraft = newStrategyAiDraft
    const aiSourceCode = String(aiDraft?.source_code || '').trim()
    const nextConfig =
      useAiDraft && aiDraft && aiDraft.config && typeof aiDraft.config === 'object' && !Array.isArray(aiDraft.config)
        ? (aiDraft.config as Record<string, unknown>)
        : {}
    const nextSchema =
      useAiDraft && aiDraft && aiDraft.config_schema && typeof aiDraft.config_schema === 'object' && !Array.isArray(aiDraft.config_schema)
        ? (aiDraft.config_schema as Record<string, unknown>)
        : { param_fields: [] }
    const nextAliases =
      useAiDraft && aiDraft && Array.isArray(aiDraft.aliases)
        ? aiDraft.aliases
            .map((value) => String(value || '').trim())
            .filter(Boolean)
        : []

    const baseSource = useAiDraft && aiSourceCode
      ? aiSourceCode
      : renderTemplateSource(selectedTemplate.template, {
        className: newStrategyClassName,
        strategyName: trimmedName,
        strategyDescription: trimmedDescription || `${trimmedName} strategy`,
        sourceKey: normalizedSourceKey,
      })

    const renderedSource = applyStrategyMetadataToSource(baseSource, {
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
    setEditorConfigJson(JSON.stringify(nextConfig, null, 2))
    setEditorSchemaJson(JSON.stringify(nextSchema, null, 2))
    setEditorAliasesCsv(nextAliases.join(', '))
    setEditorError(null)
    setValidation(null)
    setShowSettings(true)
    setShowCreateModal(false)
    setNewStrategyError(null)
  }

  // ── Derived display state ──

  const openModifyModal = () => {
    setModifyStrategyAiPrompt('')
    setModifyStrategyAiDraft(null)
    setModifyStrategyAiError(null)
    setShowModifyModal(true)
  }

  const applyModifyDraftToEditor = () => {
    const draft = modifyStrategyAiDraft
    if (!draft) return
    setEditorSlug(normalizeSlug(draft.slug || editorSlug))
    setEditorName(String(draft.name || editorName).trim())
    setEditorDescription(String(draft.description || editorDescription).trim())
    setEditorSourceKey(normalizeStrategySourceFilter(draft.source_key || editorSourceKey))
    setEditorCode(String(draft.source_code || editorCode))
    setEditorConfigJson(JSON.stringify(draft.config || {}, null, 2))
    setEditorSchemaJson(JSON.stringify(draft.config_schema || {}, null, 2))
    setEditorAliasesCsv(Array.isArray(draft.aliases) ? draft.aliases.join(', ') : editorAliasesCsv)
    setValidation({
      valid: Boolean(draft.validation?.valid),
      class_name: draft.validation?.class_name || null,
      errors: draft.validation?.errors || [],
      warnings: draft.validation?.warnings || [],
    })
    setEditorError(null)
    setModifyStrategyAiError(null)
    setShowModifyModal(false)
  }

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
            {strategiesQuery.isPending ? (
              <div className="space-y-1">
                {Array.from({ length: 7 }).map((_, index) => (
                  <div
                    key={`strategy-skeleton-${index}`}
                    className="rounded-md border border-border/40 bg-background/35 px-2.5 py-2 animate-pulse"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="h-3 w-28 rounded bg-muted/60" />
                      <div className="h-4 w-12 rounded bg-muted/55" />
                    </div>
                    <div className="mt-2 h-2.5 w-36 rounded bg-muted/55" />
                    <div className="mt-2 flex items-center gap-1.5">
                      <div className="h-2 w-10 rounded bg-muted/50" />
                      <div className="h-2 w-8 rounded bg-muted/45" />
                      <div className="h-2 w-12 rounded bg-muted/50" />
                    </div>
                  </div>
                ))}
              </div>
            ) : flatFiltered.length === 0 ? (
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
                    const sampleCount = tracker?.sampleCount ?? null
                    // Prefer live (real venue) P&L for the pill so the
                    // headline number is unambiguous. Fall back to
                    // shadow when only paper rows exist; in that case
                    // we tag the pill as ``sim`` so the user can tell
                    // it apart from real trades.
                    const liveBucket = tracker?.modes?.live ?? null
                    const shadowBucket = tracker?.modes?.shadow ?? null
                    const pnlBucket = liveBucket && liveBucket.realizedPnl != null
                      ? liveBucket
                      : shadowBucket && shadowBucket.realizedPnl != null
                        ? shadowBucket
                        : null
                    const realizedPnl = pnlBucket?.realizedPnl ?? tracker?.realizedPnl ?? null
                    const realizedPnlMode: 'live' | 'sim' | 'mixed' | null = pnlBucket
                      ? (pnlBucket === liveBucket ? 'live' : 'sim')
                      : tracker?.realizedPnl != null
                        ? 'mixed'
                        : null
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
                          )}
                            title={
                              realizedPnlMode === 'live'
                                ? 'Live trades only — submitted to Polymarket'
                                : realizedPnlMode === 'sim'
                                  ? 'Simulated/shadow trades only — no live rows yet'
                                  : realizedPnlMode === 'mixed'
                                    ? 'Aggregate of live + simulated trades'
                                    : undefined
                            }
                          >
                            P&L {realizedPnl == null ? '--' : `${realizedPnl >= 0 ? '+' : ''}$${realizedPnl.toFixed(2)}`}
                            {realizedPnlMode === 'sim' && (
                              <span className="ml-1 text-[8px] uppercase tracking-wider text-sky-400/80 font-semibold">sim</span>
                            )}
                            {realizedPnlMode === 'mixed' && (
                              <span className="ml-1 text-[8px] uppercase tracking-wider text-muted-foreground/80 font-semibold">mix</span>
                            )}
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
                          {/* "Win %" pill removed — derived from
                              ``OpportunityHistory.was_profitable`` which
                              is unreliable for several strategies (it
                              requires ``actual_payout > total_cost``
                              which doesn't fit every payout shape) and
                              shows 0% for strategies that are actually
                              net-positive. The "Acc" pill below is the
                              meaningful directional-accuracy signal. */}
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
            {/* ── Editor toolbar (reimagined: compact + grouped) ──
                Left: name + a single "meta" pill summarizing version /
                latest / loaded / system, click to expand a popover with
                version restore, enable toggle, and ID details.
                Right: Validate (pre-save gate) · AI Modify · Save
                (primary) · More popover (Research deep-link, API Docs,
                Reload, Delete). Backtesting is no longer reachable from
                the editor — it lives under Strategies → Research. */}
            <div className="shrink-0 px-4 py-2 border-b border-border/50 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 min-w-0">
                <Code2 className="w-4 h-4 text-violet-400 shrink-0" />
                <span className="text-sm font-medium truncate">
                  {editorName || 'Untitled Strategy'}
                </span>
                {/* Single compact meta pill — replaces the row of separate
                    badges. Click to manage version / enabled / system. */}
                {selectedStrategy && (
                  <Popover>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className={cn(
                          'flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[10px] font-mono transition-colors',
                          editorEnabled
                            ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-300 hover:bg-emerald-500/10'
                            : 'border-zinc-500/30 bg-zinc-500/5 text-muted-foreground hover:bg-zinc-500/10',
                        )}
                      >
                        <span className={cn(
                          'inline-block w-1.5 h-1.5 rounded-full',
                          editorEnabled ? 'bg-emerald-400' : 'bg-zinc-500',
                        )} />
                        v{selectedStrategy.version}
                        {latestVersionRow && latestVersionRow.version === selectedStrategy.version && (
                          <span className="opacity-60">· latest</span>
                        )}
                        {selectedStrategy.is_system && (
                          <span className="opacity-60">· system</span>
                        )}
                        <span className="opacity-60">· {displayStatus}</span>
                        <ChevronDown className="w-3 h-3 opacity-60" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent align="start" className="w-72 p-3 space-y-3">
                      <div>
                        <Label className="text-[10px] text-muted-foreground">Status</Label>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className={cn('text-[10px] border', statusColor)}>
                            {displayStatus}
                          </Badge>
                          {selectedStrategy.is_system && (
                            <Badge variant="secondary" className="text-[10px]">System</Badge>
                          )}
                        </div>
                      </div>
                      <div>
                        <Label className="text-[10px] text-muted-foreground">Enabled</Label>
                        <div className="flex items-center gap-2 mt-1">
                          <Switch
                            checked={editorEnabled}
                            onCheckedChange={setEditorEnabled}
                            className="scale-75"
                          />
                          <span className="text-[11px] text-muted-foreground">
                            {editorEnabled ? 'Strategy is active' : 'Disabled — will not run'}
                          </span>
                        </div>
                      </div>
                      <div>
                        <Label className="text-[10px] text-muted-foreground">Version</Label>
                        <div className="flex items-center gap-1.5 mt-1">
                          <Select
                            value={selectedVersionForRestore || String(selectedStrategy.version || '')}
                            onValueChange={setSelectedVersionForRestore}
                            disabled={busy || strategyVersions.length === 0}
                          >
                            <SelectTrigger className="h-7 flex-1 text-[10px] font-mono">
                              <SelectValue placeholder="Select version" />
                            </SelectTrigger>
                            <SelectContent>
                              {strategyVersions.map((versionRow) => (
                                <SelectItem key={versionRow.id} value={String(versionRow.version)} className="font-mono">
                                  {`v${versionRow.version}${
                                    versionRow.is_latest ? ' • latest' : ''
                                  }${
                                    Number(versionRow.version) === Number(selectedStrategy.version || 0) ? ' • current' : ''
                                  }`}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="h-7 gap-1 px-2 text-[10px]"
                            onClick={() => restoreVersionMutation.mutate()}
                            disabled={busy || !selectedRestoreVersionValid || restoreVersionIsCurrent}
                          >
                            {restoreVersionMutation.isPending ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <RefreshCw className="w-3 h-3" />
                            )}
                            Restore
                          </Button>
                        </div>
                      </div>
                      <div className="text-[10px] font-mono text-muted-foreground border-t border-border/30 pt-2">
                        <div>ID: <span className="text-foreground/80">{selectedStrategy.id.slice(0, 12)}</span></div>
                        <div>Slug: <span className="text-foreground/80">{selectedStrategy.slug}</span></div>
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                {/* Validate stays prominent — it's the pre-save gate. */}
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

                {/* Consolidated "More" dropdown — Backtest is intentionally
                    NOT here; backtesting lives under Strategies → Research
                    so the strategy editor doesn't double as a research UI. */}
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 gap-1 px-2 text-[11px]"
                      disabled={!editorCode.trim() && !selectedStrategy}
                    >
                      <MoreHorizontal className="w-3 h-3" />
                      More
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent align="end" className="w-56 p-1">
                    <div className="flex flex-col">
                      <button
                        type="button"
                        className="flex items-center gap-2 px-2 py-1.5 text-[11px] rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-left text-violet-700 dark:text-violet-300"
                        onClick={() => {
                          if (editorSlug) {
                            sessionStorage.setItem('homerun:research:strategy', editorSlug)
                          }
                          sessionStorage.setItem('homerun:research:open', '1')
                          window.dispatchEvent(new CustomEvent('homerun:research:open'))
                        }}
                        disabled={!editorSlug}
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        Open in Research
                        <span className="ml-auto text-[10px] text-muted-foreground">backtest + code evo</span>
                      </button>
                      <button
                        type="button"
                        className="flex items-center gap-2 px-2 py-1.5 text-[11px] rounded hover:bg-muted text-left"
                        onClick={() => setShowApiDocs(true)}
                      >
                        <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
                        API Docs
                      </button>
                      <button
                        type="button"
                        className="flex items-center gap-2 px-2 py-1.5 text-[11px] rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-left"
                        onClick={() => reloadMutation.mutate()}
                        disabled={busy || !selectedStrategy}
                      >
                        {reloadMutation.isPending ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Zap className="w-3.5 h-3.5 text-amber-400" />
                        )}
                        Reload
                      </button>
                    </div>
                  </PopoverContent>
                </Popover>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-[11px]"
                  onClick={openModifyModal}
                  disabled={busy || !editorCode.trim()}
                >
                  {modifyStrategyMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Wand2 className="w-3 h-3" />
                  )}
                  AI Modify
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
              {/* Horizontal subtab strip — replaces the 5 vertical
                  collapsible headers. Source Code is the default. */}
              <div className="shrink-0 flex items-center gap-0.5 border-b border-border/50 px-3 overflow-x-auto">
                {([
                  { key: 'code' as const, label: 'Source Code', icon: Code2, color: 'text-violet-400', sub: 'Python' },
                  { key: 'settings' as const, label: 'Settings', icon: Settings2, color: 'text-cyan-400', sub: editorSlug || 'no-key' },
                  { key: 'config' as const, label: 'Runtime Config', icon: Settings2, color: 'text-blue-400', sub: undefined },
                  { key: 'health' as const, label: 'Health', icon: AlertTriangle, color: 'text-amber-400', sub: selectedStrategyHealth?.status || 'untracked' },
                ]).map((tab) => (
                  <button
                    key={tab.key}
                    type="button"
                    onClick={() => setEditorTab(tab.key)}
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
                      editorTab === tab.key
                        ? 'border-violet-500 text-foreground'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <tab.icon className={cn('w-3 h-3', editorTab === tab.key ? tab.color : '')} />
                    {tab.label}
                    {tab.sub && (
                      <span className="ml-1 text-[9px] font-mono opacity-60">{tab.sub}</span>
                    )}
                  </button>
                ))}
              </div>

              {/* Settings panel — flex-1 + overflow-y-auto when active so
                  long forms scroll inside the editor pane. ``shrink-0``
                  when inactive so the wrapper stays out of the layout. */}
              <div className={cn(showSettings ? 'flex-1 min-h-0 overflow-y-auto' : 'shrink-0')}>
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

              {/* Runtime Config panel — header was the old collapsible
                  button; replaced by the horizontal tab strip above.
                  ``flex-1 min-h-0 overflow-y-auto`` while active so
                  long config forms or raw JSON editors scroll inside
                  the editor pane instead of getting clipped. */}
              <div className={cn(showConfig ? 'flex-1 min-h-0 overflow-y-auto' : 'shrink-0')}>
                {showConfig && (
                  <div className="px-3 pt-3 pb-3 animate-in fade-in duration-200 space-y-3">
                    {/* Explain the override cascade so users know what they're
                        editing here vs in the per-bot Tune tab. */}
                    <div className="rounded-md border border-cyan-500/30 bg-cyan-500/5 px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                      <div className="flex items-center gap-1.5 mb-1">
                        <Settings2 className="w-3 h-3 text-cyan-400" />
                        <span className="text-foreground font-medium text-[11px]">
                          Strategy-level defaults
                        </span>
                      </div>
                      Edits here update <code className="text-cyan-300">Strategy.config</code> — the
                      <span className="text-foreground"> defaults every bot inherits</span> when they run this
                      strategy. The runtime cascade is{' '}
                      <code className="text-foreground">source <code className="text-muted-foreground/80">default_config</code></code>{' '}→{' '}
                      <code className="text-foreground">Runtime Config (here)</code>{' '}→{' '}
                      <code className="text-foreground">Bot Tune <code className="text-muted-foreground/80">strategy_params</code></code>;
                      each layer overrides the previous. Per-bot tweaks live in{' '}
                      <span className="font-medium">Bots → Tune → Parameters</span>.
                    </div>

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

              {/* Health panel — header replaced by horizontal tab strip.
                  ``flex-1 min-h-0 overflow-y-auto`` while active so the
                  status hero, signal volume, trading-outcome cards,
                  calibration trend, and override controls all scroll
                  together inside the editor pane. */}
              <div className={cn(showHealth ? 'flex-1 min-h-0 overflow-y-auto' : 'shrink-0')}>
                {showHealth && (
                  <div className="px-4 py-3 animate-in fade-in duration-200">
                    {selectedStrategyHealth ? (
                      <div className="space-y-3">
                        {/* Status hero — explains what this status means */}
                        <div className={cn(
                          'rounded-lg border p-3 space-y-1',
                          selectedStrategyHealth.status === 'active'
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : selectedStrategyHealth.status === 'demoted'
                              ? 'border-red-500/30 bg-red-500/5'
                              : 'border-amber-500/30 bg-amber-500/5',
                        )}>
                          <div className="flex items-center gap-2">
                            <span className={cn(
                              'inline-block w-2 h-2 rounded-full',
                              selectedStrategyHealth.status === 'active'
                                ? 'bg-emerald-400'
                                : selectedStrategyHealth.status === 'demoted'
                                  ? 'bg-red-400'
                                  : 'bg-amber-400',
                            )} />
                            <span className="text-sm font-semibold capitalize">
                              {selectedStrategyHealth.status}
                            </span>
                            {selectedStrategyHealth.manual_override && (
                              <Badge variant="outline" className="text-[10px] border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
                                Manual override
                              </Badge>
                            )}
                            <span className="ml-auto text-[10px] text-muted-foreground font-mono">
                              {selectedStrategyHealth.strategy_type}
                            </span>
                          </div>
                          <p className="text-[11px] text-muted-foreground leading-snug">
                            {selectedStrategyHealth.status === 'active'
                              ? 'Strategy is participating in trading. Signals are being acted on.'
                              : selectedStrategyHealth.status === 'demoted'
                                ? 'Strategy is parked — signals are still recorded but not acted on. Investigate accuracy / MAE before reactivating.'
                                : 'Strategy hasn\'t accumulated enough data yet to be promoted or demoted.'}
                          </p>
                          {selectedStrategyHealth.last_reason && (
                            <p className="text-[10px] text-muted-foreground italic">
                              <span className="text-muted-foreground/70">Last reason:</span>{' '}
                              {selectedStrategyHealth.last_reason}
                            </p>
                          )}
                        </div>

                        {/* Guardrail metrics — what the auto-demotion engine
                            actually evaluates against. */}
                        <div>
                          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
                            Guardrail signals
                          </p>
                          <div className="grid grid-cols-3 gap-2">
                            <div className="rounded-md border border-border/40 bg-card/30 p-2.5">
                              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Sample size</p>
                              <p className="text-base font-mono font-bold mt-0.5">
                                {Number(selectedStrategyHealth.sample_size || 0).toLocaleString()}
                              </p>
                              <p className="text-[10px] text-muted-foreground/70">opportunities tracked</p>
                            </div>
                            <div className="rounded-md border border-border/40 bg-card/30 p-2.5">
                              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Directional accuracy</p>
                              <p className="text-base font-mono font-bold mt-0.5">
                                {Number.isFinite(Number(selectedStrategyHealth.directional_accuracy))
                                  ? `${(Number(selectedStrategyHealth.directional_accuracy) * 100).toFixed(1)}%`
                                  : '—'}
                              </p>
                              <p className="text-[10px] text-muted-foreground/70">predicted side correct</p>
                            </div>
                            <div className="rounded-md border border-border/40 bg-card/30 p-2.5">
                              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">MAE (ROI)</p>
                              <p className="text-base font-mono font-bold mt-0.5">
                                {Number.isFinite(Number(selectedStrategyHealth.mae_roi))
                                  ? Number(selectedStrategyHealth.mae_roi).toFixed(2)
                                  : '—'}
                              </p>
                              <p className="text-[10px] text-muted-foreground/70">mean abs ROI error</p>
                            </div>
                          </div>
                        </div>

                        {/* Signal-level metrics — counts of opportunities
                            DETECTED by this strategy, regardless of whether
                            any bot traded them. Pulled from
                            ``OpportunityHistory`` via the validation overview.
                            We surface totals only here; the
                            ``was_profitable`` flag from this table is
                            unreliable for several strategies (it requires
                            ``actual_payout > total_cost`` which doesn't fit
                            every strategy's payout shape) so we don't show
                            a "X profitable" or "win rate" line from it. */}
                        {selectedStrategyAccuracy && (
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
                              Signal volume (lifetime)
                            </p>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="rounded-md border border-border/40 bg-card/30 p-2.5">
                                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Detected</p>
                                <p className="text-base font-mono font-bold mt-0.5">
                                  {Number(selectedStrategyAccuracy.total || 0).toLocaleString()}
                                </p>
                                <p className="text-[10px] text-muted-foreground/70">
                                  opportunities the strategy emitted
                                </p>
                              </div>
                              <div className="rounded-md border border-border/40 bg-card/30 p-2.5">
                                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Resolved</p>
                                <p className="text-base font-mono font-bold mt-0.5">
                                  {Number(selectedStrategyAccuracy.resolved || 0).toLocaleString()}
                                </p>
                                <p className="text-[10px] text-muted-foreground/70">
                                  markets settled with an outcome
                                </p>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Position-level metrics — what the orchestrator
                            executed. Split by mode (live = real venue,
                            shadow = paper) when the validation
                            overview returns the per-mode breakdown.
                            Falls back to a single "All trades" card
                            when only the aggregate counters are
                            available (older backend response). */}
                        {selectedStrategyPerformance && (
                          selectedStrategyPerformance.modes.live
                          || selectedStrategyPerformance.modes.shadow
                          || selectedStrategyPerformance.modes.other
                          || selectedStrategyPerformance.realizedPnl != null
                          || selectedStrategyPerformance.terminalCount != null
                        ) && (() => {
                          const hasModeBreakdown = Boolean(
                            selectedStrategyPerformance.modes.live
                            || selectedStrategyPerformance.modes.shadow
                            || selectedStrategyPerformance.modes.other,
                          )
                          type RenderableBucket = {
                            kind: 'live' | 'shadow' | 'other' | 'aggregate'
                            total: number
                            terminalCount: number | null
                            realizedPnl: number | null
                          }
                          const buckets: RenderableBucket[] = []
                          if (hasModeBreakdown) {
                            for (const modeKey of ['live', 'shadow', 'other'] as const) {
                              const bucket = selectedStrategyPerformance.modes[modeKey]
                              if (!bucket || bucket.total <= 0) continue
                              buckets.push({
                                kind: modeKey,
                                total: bucket.total,
                                terminalCount: bucket.terminalCount,
                                realizedPnl: bucket.realizedPnl,
                              })
                            }
                          } else {
                            buckets.push({
                              kind: 'aggregate',
                              total: selectedStrategyPerformance.terminalCount ?? 0,
                              terminalCount: selectedStrategyPerformance.terminalCount,
                              realizedPnl: selectedStrategyPerformance.realizedPnl,
                            })
                          }
                          return (
                            <div>
                              <div className="flex items-center justify-between mb-1">
                                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                                  Trading outcome (30d, all bots)
                                </p>
                                {hasModeBreakdown && (
                                  <p className="text-[9px] text-muted-foreground/60">
                                    Live = real venue · Simulated = paper / shadow
                                  </p>
                                )}
                              </div>
                              <div className="grid grid-cols-2 gap-2">
                                {buckets.map((bucket) => {
                                  const isLive = bucket.kind === 'live'
                                  const isShadow = bucket.kind === 'shadow'
                                  const isAggregate = bucket.kind === 'aggregate'
                                  const label = isLive
                                    ? 'Live trades'
                                    : isShadow
                                      ? 'Simulated trades'
                                      : isAggregate
                                        ? 'All trades'
                                        : 'Other trades'
                                  const sublabel = isLive
                                    ? 'Submitted to Polymarket'
                                    : isShadow
                                      ? 'Paper / shadow rows'
                                      : isAggregate
                                        ? 'Live + simulated combined'
                                        : 'Mode not classified'
                                  const badgeText = isLive
                                    ? 'Live'
                                    : isShadow
                                      ? 'Simulated'
                                      : isAggregate
                                        ? 'All'
                                        : 'Other'
                                  const badgeClasses = isLive
                                    ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30'
                                    : isShadow
                                      ? 'bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/30'
                                      : 'bg-muted/40 text-muted-foreground border-border/40'
                                  const pnlValue = bucket.realizedPnl
                                  const pnlClasses = pnlValue != null && pnlValue > 0
                                    ? 'text-emerald-500 dark:text-emerald-300'
                                    : pnlValue != null && pnlValue < 0
                                      ? 'text-red-500 dark:text-red-300'
                                      : ''
                                  return (
                                    <div
                                      key={bucket.kind}
                                      className="rounded-md border border-border/40 bg-card/30 p-2.5 space-y-1.5"
                                    >
                                      <div className="flex items-center justify-between gap-1.5">
                                        <span
                                          className={cn(
                                            'inline-flex items-center px-1.5 py-0.5 rounded-sm border text-[9px] uppercase tracking-wider font-semibold',
                                            badgeClasses,
                                          )}
                                        >
                                          {badgeText}
                                        </span>
                                        <span className="text-[9px] text-muted-foreground/70">{sublabel}</span>
                                      </div>
                                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
                                      <div className="grid grid-cols-2 gap-1.5">
                                        <div>
                                          <p className="text-[9px] uppercase tracking-wider text-muted-foreground/70">
                                            Closed
                                          </p>
                                          <p className="text-sm font-mono font-bold">
                                            {bucket.terminalCount != null
                                              ? Number(bucket.terminalCount).toLocaleString()
                                              : '—'}
                                          </p>
                                          {bucket.total > 0 && (
                                            <p className="text-[9px] text-muted-foreground/60">
                                              of {Number(bucket.total).toLocaleString()} total
                                            </p>
                                          )}
                                        </div>
                                        <div>
                                          <p className="text-[9px] uppercase tracking-wider text-muted-foreground/70">
                                            P&amp;L
                                          </p>
                                          <p className={cn('text-sm font-mono font-bold', pnlClasses)}>
                                            {pnlValue != null
                                              ? `${pnlValue >= 0 ? '+' : ''}$${pnlValue.toFixed(2)}`
                                              : '—'}
                                          </p>
                                          <p className="text-[9px] text-muted-foreground/60">net of fees</p>
                                        </div>
                                      </div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )
                        })()}

                        {/* Calibration trend sparkline. Empty state is
                            explicit (rather than silently hiding) so the
                            user can tell whether the strategy lacks
                            telemetry or the chart is broken. */}
                        <div>
                          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
                            Calibration trend (90d)
                          </p>
                          {selectedStrategyCalibrationTrend.length >= 2 ? (
                            <CalibrationSparkline trend={selectedStrategyCalibrationTrend} />
                          ) : (
                            <div className="rounded-md border border-border/30 bg-card/20 px-3 py-3 text-[10px] text-muted-foreground">
                              No weekly calibration buckets are available for this strategy yet.
                              The trend appears once the validation engine has accumulated
                              at least two completed 7-day windows of resolved opportunities.
                            </div>
                          )}
                        </div>

                        {/* Real action buttons — proper variants, icons, tooltips */}
                        <div className="rounded-md border border-border/40 bg-background/40 p-2 space-y-1.5">
                          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                            Manual status override
                          </p>
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className={cn(
                                'h-7 gap-1 px-2.5 text-[11px]',
                                selectedStrategyHealth.status === 'active'
                                  ? ''
                                  : 'border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/10',
                              )}
                              disabled={healthBusy || selectedStrategyHealth.status === 'active'}
                              onClick={() =>
                                overrideStrategyMutation.mutate({
                                  strategyType: selectedStrategyHealth.strategy_type,
                                  status: 'active',
                                })
                              }
                              title="Force this strategy to participate in trading regardless of telemetry"
                            >
                              {overrideStrategyMutation.isPending ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                <CheckCircle2 className="w-3 h-3" />
                              )}
                              Activate
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className={cn(
                                'h-7 gap-1 px-2.5 text-[11px]',
                                selectedStrategyHealth.status === 'demoted'
                                  ? ''
                                  : 'border-red-500/30 text-red-300 hover:bg-red-500/10',
                              )}
                              disabled={healthBusy || selectedStrategyHealth.status === 'demoted'}
                              onClick={() =>
                                overrideStrategyMutation.mutate({
                                  strategyType: selectedStrategyHealth.strategy_type,
                                  status: 'demoted',
                                })
                              }
                              title="Park this strategy — signals still recorded, but not acted on"
                            >
                              {overrideStrategyMutation.isPending ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                <AlertTriangle className="w-3 h-3" />
                              )}
                              Demote
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              className="h-7 gap-1 px-2.5 text-[11px]"
                              disabled={healthBusy || !selectedStrategyHealth.manual_override}
                              onClick={() => clearOverrideMutation.mutate(selectedStrategyHealth.strategy_type)}
                              title={selectedStrategyHealth.manual_override
                                ? 'Drop the manual override and let the auto-status engine drive it again'
                                : 'No manual override active'}
                            >
                              {clearOverrideMutation.isPending ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                <X className="w-3 h-3" />
                              )}
                              Clear override
                            </Button>
                          </div>
                          <p className="text-[10px] text-muted-foreground/70">
                            Auto-status uses the metrics above plus configured guardrails. Manual overrides
                            stick until you clear them.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="rounded-lg border border-border/40 bg-card/30 p-6 text-center space-y-2">
                        <AlertTriangle className="w-6 h-6 mx-auto text-muted-foreground/40" />
                        <p className="text-[11px] text-muted-foreground">
                          No health telemetry yet for{' '}
                          <span className="font-mono text-foreground">{selectedStrategy?.slug || editorSlug || 'this strategy'}</span>.
                        </p>
                        <p className="text-[10px] text-muted-foreground/70 max-w-md mx-auto">
                          Telemetry accumulates as opportunities flow through detect → evaluate → exit. Run a
                          backtest in <span className="font-medium">Research</span> or wait for live signals
                          to populate the metrics.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Code editor — takes remaining space when its tab is active. */}
              {editorTab === 'code' && (
                <div className="flex-1 min-h-0 flex flex-col">
                  {inferredClassName && (
                    <div className="px-4 py-1.5 flex items-center justify-end shrink-0">
                      <span className="text-[10px] font-mono text-muted-foreground">
                        class {inferredClassName}
                      </span>
                    </div>
                  )}
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
              )}
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
                    <div className="rounded-lg border border-violet-500/25 bg-violet-500/5 px-3 py-3 space-y-2.5">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-xs font-semibold flex items-center gap-1.5">
                            <Sparkles className="w-3.5 h-3.5 text-violet-700 dark:text-violet-300" />
                            Generate with AI
                          </p>
                          <p className="mt-0.5 text-[10px] text-muted-foreground">
                            Describe the strategy behavior, markets, and risk controls.
                          </p>
                        </div>
                        <Button
                          type="button"
                          size="sm"
                          className="h-7 gap-1.5 text-[11px] bg-violet-600 hover:bg-violet-500 text-white"
                          onClick={() => generateDraftMutation.mutate()}
                          disabled={generateDraftMutation.isPending || !newStrategyAiPrompt.trim()}
                        >
                          {generateDraftMutation.isPending ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                          ) : (
                            <Sparkles className="w-3 h-3" />
                          )}
                          Generate
                        </Button>
                      </div>
                      <textarea
                        value={newStrategyAiPrompt}
                        onChange={(event) => {
                          setNewStrategyAiPrompt(event.target.value)
                          setNewStrategyError(null)
                        }}
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-xs leading-5 text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        rows={3}
                        placeholder="Build a momentum strategy on BTC/ETH event markets with volatility filters, configurable edge threshold, and strict stop-loss logic."
                      />
                      {newStrategyAiDraft && (
                        <div className="rounded-md border border-violet-500/30 bg-violet-500/10 px-2.5 py-2 space-y-1.5">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-[10px] text-violet-700 dark:text-violet-200">
                              Generated by {newStrategyAiDraft.model || 'AI'}{newStrategyAiDraft.used_repair_pass ? ' with repair pass' : ''}
                            </p>
                            <div className="flex items-center gap-1.5">
                              <Badge
                                variant="outline"
                                className={cn(
                                  'h-4 px-1.5 text-[9px] border',
                                  newStrategyAiDraft.validation?.valid
                                    ? 'border-emerald-500/35 text-emerald-300 bg-emerald-500/10'
                                    : 'border-amber-500/35 text-amber-300 bg-amber-500/10'
                                )}
                              >
                                {newStrategyAiDraft.validation?.valid ? 'Validated' : 'Needs review'}
                              </Badge>
                              <div className="flex items-center gap-1">
                                <span className="text-[10px] text-muted-foreground">Use AI Draft</span>
                                <Switch checked={newStrategyUseAiDraft} onCheckedChange={setNewStrategyUseAiDraft} />
                              </div>
                            </div>
                          </div>
                          {!newStrategyAiDraft.validation?.valid && (newStrategyAiDraft.validation?.errors || []).slice(0, 2).map((err, index) => (
                            <p key={`strategy-ai-error-${index}`} className="text-[10px] text-amber-300 font-mono">
                              {err}
                            </p>
                          ))}
                          {(newStrategyAiDraft.validation?.warnings || []).slice(0, 2).map((warning, index) => (
                            <p key={`strategy-ai-warning-${index}`} className="text-[10px] text-muted-foreground font-mono">
                              {warning}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>

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
      {typeof document !== 'undefined' && createPortal(
        <AnimatePresence>
          {showModifyModal && (
            <motion.div
              className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowModifyModal(false)}
            >
              <motion.div
                className="flex max-h-[88vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border/70 bg-card shadow-2xl"
                initial={{ opacity: 0, y: 18, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 18, scale: 0.98 }}
                transition={{ duration: 0.16 }}
                onClick={(event) => event.stopPropagation()}
              >
                <div className="flex items-start justify-between gap-4 border-b border-border/60 px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-violet-500/15 text-violet-700 dark:text-violet-300">
                      <Wand2 className="h-4 w-4" />
                    </span>
                    <div>
                      <h3 className="text-sm font-semibold">Modify Strategy With AI</h3>
                      <p className="text-[11px] text-muted-foreground">
                        Generate a revised source draft, review it, then apply it to the editor before saving.
                      </p>
                    </div>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                    onClick={() => setShowModifyModal(false)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                <div className="grid min-h-0 flex-1 gap-4 overflow-auto p-4 lg:grid-cols-[360px_minmax(0,1fr)]">
                  <div className="space-y-4">
                    <div className="rounded-xl border border-border/60 bg-muted/25 p-3">
                      <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                        Current Strategy
                      </p>
                      <p className="mt-2 text-sm font-semibold">{editorName || selectedStrategy?.name || 'Draft strategy'}</p>
                      <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                        {normalizeSlug(editorSlug || selectedStrategy?.slug || 'strategy')} · {normalizeStrategySourceFilter(editorSourceKey)}
                      </p>
                    </div>

                    <div>
                      <Label className="text-[11px] text-muted-foreground">Modification Instructions</Label>
                      <textarea
                        value={modifyStrategyAiPrompt}
                        onChange={(event) => {
                          setModifyStrategyAiPrompt(event.target.value)
                          setModifyStrategyAiError(null)
                        }}
                        className="mt-1 min-h-[170px] w-full rounded-md border border-input bg-background px-3 py-2 text-xs leading-5 text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        placeholder="Example: tighten the exit logic, reduce risk after two losses, and only select positions when liquidity is above the configured threshold."
                      />
                    </div>

                    <Button
                      type="button"
                      size="sm"
                      className="w-full gap-2 bg-violet-600 text-white hover:bg-violet-500"
                      onClick={() => modifyStrategyMutation.mutate()}
                      disabled={busy || !modifyStrategyAiPrompt.trim() || !editorCode.trim()}
                    >
                      {modifyStrategyMutation.isPending ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Sparkles className="h-3.5 w-3.5" />
                      )}
                      Generate Modified Draft
                    </Button>

                    {modifyStrategyAiError && (
                      <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                        {modifyStrategyAiError}
                      </div>
                    )}

                    {modifyStrategyAiDraft && (
                      <div className="space-y-3 rounded-xl border border-border/60 bg-muted/25 p-3">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                            Draft Result
                          </span>
                          <Badge
                            variant="outline"
                            className={cn(
                              'text-[10px]',
                              modifyStrategyAiDraft.validation.valid
                                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                                : 'border-red-500/30 bg-red-500/10 text-red-300'
                            )}
                          >
                            {modifyStrategyAiDraft.validation.valid ? 'Valid' : 'Needs Fix'}
                          </Badge>
                        </div>
                        <p className="text-xs leading-5 text-muted-foreground">
                          {modifyStrategyAiDraft.change_summary || 'AI generated a modified source draft.'}
                        </p>
                        {modifyStrategyAiDraft.validation.errors.length > 0 && (
                          <div className="space-y-1 rounded-md border border-red-500/20 bg-red-500/10 px-2 py-2 text-[11px] text-red-300">
                            {modifyStrategyAiDraft.validation.errors.slice(0, 4).map((error, index) => (
                              <p key={`${error}-${index}`}>{error}</p>
                            ))}
                          </div>
                        )}
                        {modifyStrategyAiDraft.validation.warnings.length > 0 && (
                          <div className="space-y-1 rounded-md border border-amber-500/20 bg-amber-500/10 px-2 py-2 text-[11px] text-amber-300">
                            {modifyStrategyAiDraft.validation.warnings.slice(0, 4).map((warning, index) => (
                              <p key={`${warning}-${index}`}>{warning}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="min-h-0 rounded-xl border border-border/60 bg-muted/30 dark:bg-black/30">
                    <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
                      <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                        Modified Source Preview
                      </span>
                      <span className="text-[10px] font-mono text-muted-foreground">
                        {modifyStrategyAiDraft ? 'Editable before apply' : 'Generate a draft to preview changes'}
                      </span>
                    </div>
                    <div className="p-3">
                      {modifyStrategyAiDraft ? (
                        <CodeEditor
                          value={modifyStrategyAiDraft.source_code}
                          onChange={(value) =>
                            setModifyStrategyAiDraft((current) =>
                              current ? { ...current, source_code: value } : current
                            )
                          }
                          minHeight="520px"
                        />
                      ) : (
                        <div className="flex min-h-[520px] items-center justify-center rounded-md border border-dashed border-border/70 bg-background/30 px-8 text-center text-xs text-muted-foreground">
                          Describe the modification you want, then generate a revised strategy source draft.
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-end gap-2 border-t border-border/60 px-4 py-3">
                  <Button type="button" variant="outline" size="sm" onClick={() => setShowModifyModal(false)}>
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    className="bg-violet-600 text-white hover:bg-violet-500"
                    onClick={applyModifyDraftToEditor}
                    disabled={!modifyStrategyAiDraft || !modifyStrategyAiDraft.source_code.trim()}
                  >
                    Apply to Editor
                  </Button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      <StrategyApiDocsFlyout open={showApiDocs} onOpenChange={setShowApiDocs} variant={flyoutVariant} />
    </div>
  )
}
