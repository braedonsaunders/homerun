import { useState, useEffect, useRef, type ChangeEvent } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Bot,
  Bell,
  Database,
  RefreshCw,
  Save,
  CheckCircle,
  AlertCircle,
  Eye,
  EyeOff,
  Lock,
  MessageSquare,
  DollarSign,
  Shield,
  ChevronDown,
  Loader2,
  Trash2,
  Play,
  Download,
  Upload,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import { Separator } from './ui/separator'
import { Badge } from './ui/badge'
import {
  getSettings,
  updateSettings,
  testTelegramConnection,
  testLLMConnection,
  testTradingProxy,
  flushDatabaseData,
  getDatabaseMaintenanceStats,
  getLLMModels,
  refreshLLMModels,
  exportSettingsBundle,
  importSettingsBundle,
  runWorkerOnce,
  type DatabaseFlushTarget,
  type LLMModelOption,
  type DiscoverySettings,
  type UILockSettings,
  type SettingsTransferCategory,
  type SettingsExportBundle,
} from '../services/api'

type SettingsSection = 'llm' | 'scanner' | 'notifications' | 'security' | 'vpn' | 'discovery' | 'maintenance' | 'transfer'

const DEFAULT_DISCOVERY_SETTINGS: DiscoverySettings = {
  max_discovered_wallets: 20_000,
  maintenance_enabled: true,
  keep_recent_trade_days: 7,
  keep_new_discoveries_days: 30,
  maintenance_batch: 900,
  stale_analysis_hours: 12,
  analysis_priority_batch_limit: 2500,
  delay_between_markets: 0.25,
  delay_between_wallets: 0.15,
  max_markets_per_run: 100,
  max_wallets_per_market: 50,
  trader_opps_source_filter: 'all',
  trader_opps_min_tier: 'WATCH',
  trader_opps_side_filter: 'all',
  trader_opps_confluence_limit: 50,
  trader_opps_insider_limit: 40,
  trader_opps_insider_min_confidence: 0.62,
  trader_opps_insider_max_age_minutes: 180,
  pool_recompute_mode: 'quality_only',
  pool_target_size: 500,
  pool_min_size: 400,
  pool_max_size: 600,
  pool_active_window_hours: 72,
  pool_inactive_rising_retention_hours: 336,
  pool_selection_score_floor: 0.55,
  pool_max_hourly_replacement_rate: 0.15,
  pool_replacement_score_cutoff: 0.05,
  pool_max_cluster_share: 0.08,
  pool_high_conviction_threshold: 0.72,
  pool_insider_priority_threshold: 0.62,
  pool_min_eligible_trades: 50,
  pool_max_eligible_anomaly: 0.5,
  pool_core_min_win_rate: 0.60,
  pool_core_min_sharpe: 1.0,
  pool_core_min_profit_factor: 1.5,
  pool_rising_min_win_rate: 0.55,
  pool_slo_min_analyzed_pct: 95.0,
  pool_slo_min_profitable_pct: 80.0,
  pool_leaderboard_wallet_trade_sample: 160,
  pool_incremental_wallet_trade_sample: 80,
  pool_full_sweep_interval_seconds: 1800,
  pool_incremental_refresh_interval_seconds: 120,
  pool_activity_reconciliation_interval_seconds: 120,
  pool_recompute_interval_seconds: 60,
}

const DEFAULT_UI_LOCK_SETTINGS: UILockSettings = {
  enabled: false,
  idle_timeout_minutes: 15,
  has_password: false,
}

const SETTINGS_TRANSFER_CATEGORIES: Array<{
  id: SettingsTransferCategory
  label: string
  description: string
}> = [
  { id: 'bot_traders', label: 'Bot Traders', description: 'Configured trader bots and risk/runtime settings' },
  { id: 'strategies', label: 'Strategies', description: 'Strategy definitions, source code, and version snapshots' },
  { id: 'data_sources', label: 'Data Sources', description: 'Data-source definitions and retention/config' },
  { id: 'market_credentials', label: 'Market Credentials', description: 'Polymarket + Kalshi API credentials' },
  { id: 'vpn_configuration', label: 'VPN Configuration', description: 'Trading proxy URL and VPN enforcement settings' },
  { id: 'llm_configuration', label: 'LLM Configuration', description: 'Provider, model, API keys, and spend cap' },
  {
    id: 'telegram_configuration',
    label: 'Telegram Setup',
    description: 'Bot token, chat ID, and notification delivery rules',
  },
]

const getDiscoverySettings = (value: Partial<DiscoverySettings> | null | undefined): DiscoverySettings => {
  if (!value || typeof value !== 'object') {
    return DEFAULT_DISCOVERY_SETTINGS
  }

  return {
    ...DEFAULT_DISCOVERY_SETTINGS,
    ...value,
  }
}

const formatDbBytes = (value: number | null | undefined): string => {
  if (value == null || !Number.isFinite(value)) {
    return 'Unavailable'
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = Math.max(0, value)
  let unitIndex = 0
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024
    unitIndex += 1
  }
  const precision = unitIndex <= 1 ? 0 : 2
  return `${amount.toFixed(precision)} ${units[unitIndex]}`
}

function SecretInput({
  label,
  value,
  placeholder,
  onChange,
  showSecret,
  onToggle,
  description
}: {
  label: string
  value: string
  placeholder: string
  onChange: (value: string) => void
  showSecret: boolean
  onToggle: () => void
  description?: string
}) {
  return (
    <div>
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <div className="relative mt-1">
        <Input
          type={showSecret ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="pr-10 font-mono text-sm"
        />
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="absolute right-0 top-0 h-full px-3"
          onClick={onToggle}
        >
          {showSecret ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
        </Button>
      </div>
      {description && <p className="text-[11px] text-muted-foreground/70 mt-1">{description}</p>}
    </div>
  )
}

export default function SettingsPanel({
  showHeader = true,
}: {
  showHeader?: boolean
}) {
  const [expandedSections, setExpandedSections] = useState<Set<SettingsSection>>(new Set())
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [activeFlushTarget, setActiveFlushTarget] = useState<DatabaseFlushTarget | null>(null)

  // Form state for each section
  const [llmForm, setLlmForm] = useState({
    provider: 'none',
    openai_api_key: '',
    anthropic_api_key: '',
    google_api_key: '',
    xai_api_key: '',
    deepseek_api_key: '',
    ollama_api_key: '',
    ollama_base_url: '',
    lmstudio_api_key: '',
    lmstudio_base_url: '',
    model: '',
    max_monthly_spend: 50.0
  })

  const [availableModels, setAvailableModels] = useState<Record<string, LLMModelOption[]>>({})
  const [isRefreshingModels, setIsRefreshingModels] = useState(false)

  const [notificationsForm, setNotificationsForm] = useState({
    enabled: false,
    telegram_bot_token: '',
    telegram_chat_id: '',
    notify_on_opportunity: true,
    notify_on_trade: true,
    notify_min_roi: 5.0,
    notify_autotrader_orders: false,
    notify_autotrader_closes: true,
    notify_autotrader_issues: true,
    notify_autotrader_timeline: true,
    notify_autotrader_summary_interval_minutes: 60,
    notify_autotrader_summary_per_trader: false,
  })

  const [discoveryForm, setDiscoveryForm] = useState<DiscoverySettings>(DEFAULT_DISCOVERY_SETTINGS)

  const [scannerForm, setScannerForm] = useState({
    scan_interval_seconds: 60,
    min_profit_threshold: 2.5,
    max_markets_to_scan: 0,
    max_events_to_scan: 0,
    market_fetch_page_size: 200,
    market_fetch_order: 'volume',
    min_liquidity: 1000.0,
    max_opportunities_total: 500,
    max_opportunities_per_strategy: 120,
  })

  const [maintenanceForm, setMaintenanceForm] = useState({
    auto_cleanup_enabled: false,
    cleanup_interval_hours: 24,
    cleanup_resolved_trade_days: 30,
    cleanup_trade_signal_emission_days: 21,
    cleanup_trade_signal_update_days: 3,
    cleanup_wallet_activity_rollup_days: 60,
    cleanup_wallet_activity_dedupe_enabled: true,
    llm_usage_retention_days: 30,
    market_cache_hygiene_enabled: true,
    market_cache_hygiene_interval_hours: 6,
    market_cache_retention_days: 120,
    market_cache_reference_lookback_days: 45,
    market_cache_weak_entry_grace_days: 7,
    market_cache_max_entries_per_slug: 3,
  })

  const [vpnForm, setVpnForm] = useState({
    enabled: false,
    proxy_url: '',
    verify_ssl: true,
    timeout: 30,
    require_vpn: true
  })

  const [uiLockForm, setUiLockForm] = useState({
    enabled: DEFAULT_UI_LOCK_SETTINGS.enabled,
    idle_timeout_minutes: DEFAULT_UI_LOCK_SETTINGS.idle_timeout_minutes,
    has_password: DEFAULT_UI_LOCK_SETTINGS.has_password,
    password: '',
    confirm_password: '',
    clear_password: false,
  })

  const transferFileInputRef = useRef<HTMLInputElement | null>(null)
  const [transferCategories, setTransferCategories] = useState<Record<SettingsTransferCategory, boolean>>(() => {
    const initial: Record<SettingsTransferCategory, boolean> = {
      bot_traders: true,
      strategies: true,
      data_sources: true,
      market_credentials: true,
      vpn_configuration: true,
      llm_configuration: true,
      telegram_configuration: true,
    }
    return initial
  })
  const [importBundle, setImportBundle] = useState<SettingsExportBundle | null>(null)
  const [importFileName, setImportFileName] = useState<string>('')

  const queryClient = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const maintenanceStatsQuery = useQuery({
    queryKey: ['maintenance-stats'],
    queryFn: getDatabaseMaintenanceStats,
    enabled: expandedSections.has('maintenance'),
    refetchInterval: expandedSections.has('maintenance') ? 30_000 : false,
  })

  // Sync form state with loaded settings
  useEffect(() => {
    if (settings) {
      setLlmForm({
        provider: settings.llm?.provider || 'none',
        openai_api_key: '',
        anthropic_api_key: '',
        google_api_key: '',
        xai_api_key: '',
        deepseek_api_key: '',
        ollama_api_key: '',
        ollama_base_url: settings.llm?.ollama_base_url || '',
        lmstudio_api_key: '',
        lmstudio_base_url: settings.llm?.lmstudio_base_url || '',
        model: settings.llm?.model || '',
        max_monthly_spend: settings.llm?.max_monthly_spend ?? 50.0
      })

      setNotificationsForm({
        enabled: settings.notifications?.enabled ?? false,
        telegram_bot_token: '',
        telegram_chat_id: settings.notifications?.telegram_chat_id || '',
        notify_on_opportunity: settings.notifications?.notify_on_opportunity ?? true,
        notify_on_trade: settings.notifications?.notify_on_trade ?? true,
        notify_min_roi: settings.notifications?.notify_min_roi ?? 5.0,
        notify_autotrader_orders: settings.notifications?.notify_autotrader_orders ?? false,
        notify_autotrader_closes: settings.notifications?.notify_autotrader_closes ?? true,
        notify_autotrader_issues: settings.notifications?.notify_autotrader_issues ?? true,
        notify_autotrader_timeline: settings.notifications?.notify_autotrader_timeline ?? true,
        notify_autotrader_summary_interval_minutes: settings.notifications?.notify_autotrader_summary_interval_minutes ?? 60,
        notify_autotrader_summary_per_trader: settings.notifications?.notify_autotrader_summary_per_trader ?? false,
      })

      const discoverySettings = getDiscoverySettings(settings.discovery)
      setDiscoveryForm(discoverySettings)

      setScannerForm({
        scan_interval_seconds: settings.scanner?.scan_interval_seconds ?? 60,
        min_profit_threshold: settings.scanner?.min_profit_threshold ?? 2.5,
        max_markets_to_scan: settings.scanner?.max_markets_to_scan ?? 0,
        max_events_to_scan: settings.scanner?.max_events_to_scan ?? 0,
        market_fetch_page_size: settings.scanner?.market_fetch_page_size ?? 200,
        market_fetch_order: settings.scanner?.market_fetch_order ?? 'volume',
        min_liquidity: settings.scanner?.min_liquidity ?? 1000.0,
        max_opportunities_total: settings.scanner?.max_opportunities_total ?? 500,
        max_opportunities_per_strategy: settings.scanner?.max_opportunities_per_strategy ?? 120,
      })

      setMaintenanceForm({
        auto_cleanup_enabled: settings.maintenance?.auto_cleanup_enabled ?? false,
        cleanup_interval_hours: settings.maintenance?.cleanup_interval_hours ?? 24,
        cleanup_resolved_trade_days: settings.maintenance?.cleanup_resolved_trade_days ?? 30,
        cleanup_trade_signal_emission_days: settings.maintenance?.cleanup_trade_signal_emission_days ?? 21,
        cleanup_trade_signal_update_days: settings.maintenance?.cleanup_trade_signal_update_days ?? 3,
        cleanup_wallet_activity_rollup_days: settings.maintenance?.cleanup_wallet_activity_rollup_days ?? 60,
        cleanup_wallet_activity_dedupe_enabled: settings.maintenance?.cleanup_wallet_activity_dedupe_enabled ?? true,
        llm_usage_retention_days: settings.maintenance?.llm_usage_retention_days ?? 30,
        market_cache_hygiene_enabled: settings.maintenance?.market_cache_hygiene_enabled ?? true,
        market_cache_hygiene_interval_hours: settings.maintenance?.market_cache_hygiene_interval_hours ?? 6,
        market_cache_retention_days: settings.maintenance?.market_cache_retention_days ?? 120,
        market_cache_reference_lookback_days: settings.maintenance?.market_cache_reference_lookback_days ?? 45,
        market_cache_weak_entry_grace_days: settings.maintenance?.market_cache_weak_entry_grace_days ?? 7,
        market_cache_max_entries_per_slug: settings.maintenance?.market_cache_max_entries_per_slug ?? 3,
      })

      if (settings.trading_proxy) {
        setVpnForm({
          enabled: settings.trading_proxy?.enabled ?? false,
          proxy_url: '',  // Don't pre-fill masked URL
          verify_ssl: settings.trading_proxy?.verify_ssl ?? true,
          timeout: settings.trading_proxy?.timeout ?? 30,
          require_vpn: settings.trading_proxy?.require_vpn ?? true
        })
      }

      setUiLockForm({
        enabled: settings.ui_lock?.enabled ?? DEFAULT_UI_LOCK_SETTINGS.enabled,
        idle_timeout_minutes: settings.ui_lock?.idle_timeout_minutes ?? DEFAULT_UI_LOCK_SETTINGS.idle_timeout_minutes,
        has_password: settings.ui_lock?.has_password ?? DEFAULT_UI_LOCK_SETTINGS.has_password,
        password: '',
        confirm_password: '',
        clear_password: false,
      })

    }
  }, [settings])

  // Load available models on mount
  useEffect(() => {
    getLLMModels().then(res => {
      if (res.models) setAvailableModels(res.models)
    }).catch(() => {})
  }, [])

  const handleRefreshModels = async () => {
    setIsRefreshingModels(true)
    try {
      const provider = llmForm.provider !== 'none' ? llmForm.provider : undefined
      const res = await refreshLLMModels(provider)
      if (res.models) setAvailableModels(res.models)
    } catch {
      // ignore
    } finally {
      setIsRefreshingModels(false)
    }
  }

  // Get models for the currently selected provider
  const modelsForProvider = availableModels[llmForm.provider] || []

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      queryClient.invalidateQueries({ queryKey: ['ui-lock-status'] })
      queryClient.invalidateQueries({ queryKey: ['ai-usage'] })
      queryClient.invalidateQueries({ queryKey: ['ai-status'] })
      setSaveMessage({ type: 'success', text: 'Settings saved successfully' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save settings' })
      setTimeout(() => setSaveMessage(null), 5000)
    }
  })


  const testTelegramMutation = useMutation({
    mutationFn: testTelegramConnection,
  })

  const testVpnMutation = useMutation({
    mutationFn: testTradingProxy,
  })

  const testLlmMutation = useMutation({
    mutationFn: () => testLLMConnection(),
  })

  const flushDataMutation = useMutation({
    mutationFn: (target: DatabaseFlushTarget) => flushDatabaseData(target),
    onSuccess: (data, target) => {
      queryClient.invalidateQueries()
      const totalCleared = Object.values(data.flushed || {}).reduce((datasetSum, datasetCounts) => {
        return datasetSum + Object.values(datasetCounts || {}).reduce((sum, count) => sum + Number(count || 0), 0)
      }, 0)
      const targetLabel = target === 'all' ? 'all selected datasets' : `${target} dataset`
      setSaveMessage({
        type: 'success',
        text: `Flushed ${targetLabel} (${totalCleared} rows cleared). Live positions/history preserved.`,
      })
      setTimeout(() => setSaveMessage(null), 5000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'Failed to flush database data' })
      setTimeout(() => setSaveMessage(null), 7000)
    },
    onSettled: () => {
      setActiveFlushTarget(null)
    },
  })

  const runOrchestratorOnceMutation = useMutation({
    mutationFn: () => runWorkerOnce('trader_orchestrator'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-status'] })
      setSaveMessage({ type: 'success', text: 'Trader orchestrator one-time run queued' })
      setTimeout(() => setSaveMessage(null), 4000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'Failed to queue trader orchestrator run' })
      setTimeout(() => setSaveMessage(null), 7000)
    },
  })

  const selectedTransferCategories = SETTINGS_TRANSFER_CATEGORIES
    .filter((category) => transferCategories[category.id])
    .map((category) => category.id)

  const exportSettingsMutation = useMutation({
    mutationFn: (categories: SettingsTransferCategory[]) => exportSettingsBundle({ include_categories: categories }),
    onSuccess: (data, categories) => {
      const blob = new Blob([JSON.stringify(data.bundle, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      anchor.href = url
      anchor.download = `homerun-settings-${timestamp}.json`
      anchor.click()
      URL.revokeObjectURL(url)
      setSaveMessage({ type: 'success', text: `Exported ${categories.length} configuration categories` })
      setTimeout(() => setSaveMessage(null), 5000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'Failed to export settings bundle' })
      setTimeout(() => setSaveMessage(null), 7000)
    },
  })

  const importSettingsMutation = useMutation({
    mutationFn: ({ bundle, categories }: { bundle: Record<string, unknown>; categories: SettingsTransferCategory[] }) =>
      importSettingsBundle({ bundle, include_categories: categories }),
    onSuccess: (data) => {
      queryClient.invalidateQueries()
      const importedCount = Array.isArray(data.imported_categories) ? data.imported_categories.length : 0
      setSaveMessage({ type: 'success', text: `Imported ${importedCount} configuration categories` })
      setTimeout(() => setSaveMessage(null), 6000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'Failed to import settings bundle' })
      setTimeout(() => setSaveMessage(null), 8000)
    },
  })

  const setAllTransferCategories = (checked: boolean) => {
    setTransferCategories({
      bot_traders: checked,
      strategies: checked,
      data_sources: checked,
      market_credentials: checked,
      vpn_configuration: checked,
      llm_configuration: checked,
      telegram_configuration: checked,
    })
  }

  const toggleTransferCategory = (category: SettingsTransferCategory) => {
    setTransferCategories((prev) => ({ ...prev, [category]: !prev[category] }))
  }

  const handleTransferFileSelect = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Invalid settings bundle format')
      }
      setImportBundle(parsed as SettingsExportBundle)
      setImportFileName(file.name)
      setSaveMessage({ type: 'success', text: `Loaded import bundle: ${file.name}` })
      setTimeout(() => setSaveMessage(null), 4000)
    } catch {
      setImportBundle(null)
      setImportFileName('')
      setSaveMessage({ type: 'error', text: 'Invalid JSON file. Select a valid settings bundle export.' })
      setTimeout(() => setSaveMessage(null), 7000)
    } finally {
      event.target.value = ''
    }
  }

  const handleExportBundle = () => {
    if (!selectedTransferCategories.length) {
      setSaveMessage({ type: 'error', text: 'Select at least one category to export.' })
      setTimeout(() => setSaveMessage(null), 5000)
      return
    }
    exportSettingsMutation.mutate(selectedTransferCategories)
  }

  const handleImportBundle = () => {
    if (!importBundle) {
      setSaveMessage({ type: 'error', text: 'Select a settings bundle JSON file first.' })
      setTimeout(() => setSaveMessage(null), 5000)
      return
    }
    if (!selectedTransferCategories.length) {
      setSaveMessage({ type: 'error', text: 'Select at least one category to import.' })
      setTimeout(() => setSaveMessage(null), 5000)
      return
    }
    const confirmed = window.confirm(
      'Import selected categories now?\n\nThis overwrites current configuration values in those categories.'
    )
    if (!confirmed) {
      return
    }
    importSettingsMutation.mutate({
      bundle: importBundle as unknown as Record<string, unknown>,
      categories: selectedTransferCategories,
    })
  }

  const handleSaveSection = (section: SettingsSection) => {
    const updates: any = {}

    switch (section) {
      case 'llm':
        updates.llm = {
          provider: llmForm.provider,
          model: llmForm.model || null,
          max_monthly_spend: llmForm.max_monthly_spend,
          ollama_base_url: llmForm.ollama_base_url || null,
          lmstudio_base_url: llmForm.lmstudio_base_url || null,
        }
        if (llmForm.openai_api_key) updates.llm.openai_api_key = llmForm.openai_api_key
        if (llmForm.anthropic_api_key) updates.llm.anthropic_api_key = llmForm.anthropic_api_key
        if (llmForm.google_api_key) updates.llm.google_api_key = llmForm.google_api_key
        if (llmForm.xai_api_key) updates.llm.xai_api_key = llmForm.xai_api_key
        if (llmForm.deepseek_api_key) updates.llm.deepseek_api_key = llmForm.deepseek_api_key
        if (llmForm.ollama_api_key) updates.llm.ollama_api_key = llmForm.ollama_api_key
        if (llmForm.lmstudio_api_key) updates.llm.lmstudio_api_key = llmForm.lmstudio_api_key
        break
      case 'notifications':
        updates.notifications = {
          enabled: notificationsForm.enabled,
          notify_on_opportunity: notificationsForm.notify_on_opportunity,
          notify_on_trade: notificationsForm.notify_on_trade,
          notify_min_roi: notificationsForm.notify_min_roi,
          telegram_chat_id: notificationsForm.telegram_chat_id || null,
          notify_autotrader_orders: notificationsForm.notify_autotrader_orders,
          notify_autotrader_closes: notificationsForm.notify_autotrader_closes,
          notify_autotrader_issues: notificationsForm.notify_autotrader_issues,
          notify_autotrader_timeline: notificationsForm.notify_autotrader_timeline,
          notify_autotrader_summary_interval_minutes: notificationsForm.notify_autotrader_summary_interval_minutes,
          notify_autotrader_summary_per_trader: notificationsForm.notify_autotrader_summary_per_trader,
        }
        if (notificationsForm.telegram_bot_token) {
          updates.notifications.telegram_bot_token = notificationsForm.telegram_bot_token
        }
        break
      case 'security': {
        const normalizedPassword = uiLockForm.password.trim()
        const normalizedConfirm = uiLockForm.confirm_password.trim()
        if (normalizedPassword && normalizedPassword !== normalizedConfirm) {
          setSaveMessage({ type: 'error', text: 'UI lock passwords do not match' })
          setTimeout(() => setSaveMessage(null), 5000)
          return
        }
        if (uiLockForm.enabled && !uiLockForm.has_password && !normalizedPassword) {
          setSaveMessage({ type: 'error', text: 'Set a password before enabling UI lock' })
          setTimeout(() => setSaveMessage(null), 5000)
          return
        }
        if (uiLockForm.enabled && uiLockForm.clear_password && !normalizedPassword) {
          setSaveMessage({ type: 'error', text: 'Cannot clear password while UI lock remains enabled' })
          setTimeout(() => setSaveMessage(null), 5000)
          return
        }
        updates.ui_lock = {
          enabled: uiLockForm.enabled,
          idle_timeout_minutes: uiLockForm.idle_timeout_minutes,
          clear_password: uiLockForm.clear_password,
        } as Partial<UILockSettings>
        if (normalizedPassword) {
          updates.ui_lock.password = normalizedPassword
        }
        break
      }
      case 'discovery':
        updates.discovery = discoveryForm
        break
      case 'scanner':
        updates.scanner = scannerForm
        break
      case 'vpn':
        updates.trading_proxy = {
          enabled: vpnForm.enabled,
          verify_ssl: vpnForm.verify_ssl,
          timeout: vpnForm.timeout,
          require_vpn: vpnForm.require_vpn,
        } as any
        // Only send proxy_url if the user entered a new value
        if (vpnForm.proxy_url) {
          (updates.trading_proxy as any).proxy_url = vpnForm.proxy_url
        }
        break
      case 'maintenance':
        updates.maintenance = maintenanceForm
        break
    }

    saveMutation.mutate(updates)
  }

  const toggleSecret = (key: string) => {
    setShowSecrets(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const handleFlushTarget = (target: DatabaseFlushTarget) => {
    const labelMap: Record<DatabaseFlushTarget, string> = {
      scanner: 'Scanner/Market data',
      weather: 'Weather workflow data',
      news: 'News workflow/feed data',
      trader_orchestrator: 'Trader orchestrator runtime data',
      all: 'ALL non-trading datasets',
    }

    const selectedLabel = labelMap[target]
    const confirmed = window.confirm(
      `Flush ${selectedLabel}?\n\nThis cannot be undone.\n\nProtected data that will NOT be deleted:\n- Live/executed trade history\n- Position ledgers\n- Simulation trade history`
    )
    if (!confirmed) return

    setActiveFlushTarget(target)
    flushDataMutation.mutate(target)
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const toggleSection = (id: SettingsSection) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  // Compute status summaries for each collapsed section
  const getSectionStatus = (id: SettingsSection): string => {
    switch (id) {
      case 'llm':
        return llmForm.provider !== 'none' ? `${llmForm.provider}` : 'Disabled'
      case 'notifications':
        return notificationsForm.enabled ? 'Enabled' : 'Disabled'
      case 'security':
        return uiLockForm.enabled ? `Auto-lock ${uiLockForm.idle_timeout_minutes}m` : 'Disabled'
      case 'scanner':
        return `caps ${scannerForm.max_opportunities_total}/${scannerForm.max_opportunities_per_strategy}`
      case 'discovery':
        return discoveryForm.maintenance_enabled
          ? `${discoveryForm.max_discovered_wallets.toLocaleString()} cap`
          : 'Disabled'
      case 'vpn':
        return vpnForm.enabled ? 'Active' : 'Disabled'
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'Auto-clean on' : 'Manual'
      case 'transfer':
        return `${selectedTransferCategories.length} selected`
      default:
        return ''
    }
  }

  const getStatusColor = (id: SettingsSection): string => {
    switch (id) {
      case 'llm':
        return llmForm.provider !== 'none' ? 'text-purple-400 bg-purple-500/10' : 'text-muted-foreground bg-muted'
      case 'notifications':
        return notificationsForm.enabled ? 'text-blue-400 bg-blue-500/10' : 'text-muted-foreground bg-muted'
      case 'security':
        return uiLockForm.enabled ? 'text-emerald-400 bg-emerald-500/10' : 'text-muted-foreground bg-muted'
      case 'scanner':
        return 'text-amber-400 bg-amber-500/10'
      case 'discovery':
        return discoveryForm.maintenance_enabled
          ? 'text-green-400 bg-green-500/10'
          : 'text-muted-foreground bg-muted'
      case 'vpn':
        return vpnForm.enabled ? 'text-indigo-400 bg-indigo-500/10' : 'text-muted-foreground bg-muted'
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'text-red-400 bg-red-500/10' : 'text-muted-foreground bg-muted'
      case 'transfer':
        return 'text-cyan-400 bg-cyan-500/10'
      default:
        return 'text-muted-foreground bg-muted'
    }
  }

  const sections: { id: SettingsSection; icon: any; label: string; description: string }[] = [
    { id: 'llm', icon: Bot, label: 'AI / LLM Services', description: 'Configure AI providers' },
    { id: 'scanner', icon: Database, label: 'Scanner', description: 'Scan limits, thresholds, and pool caps' },
    { id: 'notifications', icon: Bell, label: 'Notifications', description: 'Telegram alerts' },
    { id: 'security', icon: Lock, label: 'UI Lock', description: 'Local screen lock and inactivity timeout' },
    { id: 'vpn', icon: Shield, label: 'Trading VPN/Proxy', description: 'Route trades through VPN' },
    { id: 'discovery', icon: Database, label: 'Discovery', description: 'Wallet discovery growth and maintenance' },
    { id: 'maintenance', icon: Database, label: 'Database', description: 'Cleanup & maintenance' },
    { id: 'transfer', icon: Upload, label: 'Import / Export', description: 'Migrate trading configuration bundle' },
  ]

  return (
    <div className="space-y-4 relative">
      {/* Header */}
      {showHeader ? (
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold tracking-tight">Settings</h2>
          {settings?.updated_at && (
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Updated {new Date(settings.updated_at).toLocaleString()}
            </span>
          )}
        </div>
      ) : null}

      {/* Floating toast for save messages */}
      {saveMessage && (
        <div className={cn(
          "fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300",
          saveMessage.type === 'success'
            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
            : "bg-red-500/10 text-red-400 border-red-500/20"
        )}>
          {saveMessage.type === 'success' ? (
            <CheckCircle className="w-4 h-4 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 shrink-0" />
          )}
          {saveMessage.text}
        </div>
      )}

      {/* Two-column grid of collapsible sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {sections.map(section => {
          const isExpanded = expandedSections.has(section.id)
          const Icon = section.icon
          const status = getSectionStatus(section.id)
          const statusColor = getStatusColor(section.id)

          return (
            <div
              key={section.id}
              className={cn(
                "bg-card/60 border border-border/40 rounded-xl overflow-hidden transition-all duration-200",
                isExpanded && "lg:col-span-2"
              )}
            >
              {/* Section Header - clickable */}
              <button
                type="button"
                onClick={() => toggleSection(section.id)}
                className="w-full flex items-center gap-3 p-3 hover:bg-muted/40 transition-colors cursor-pointer"
              >
                <div className="shrink-0">
                  <Icon className="w-4 h-4 text-muted-foreground" />
                </div>
                <div className="flex-1 text-left min-w-0">
                  <div className="text-sm font-medium leading-tight">{section.label}</div>
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground truncate">{section.description}</div>
                </div>
                <Badge variant="outline" className={cn("text-[10px] px-2 py-0.5 border-0 shrink-0", statusColor)}>
                  {status}
                </Badge>
                <ChevronDown className={cn(
                  "w-4 h-4 text-muted-foreground shrink-0 transition-transform duration-200",
                  isExpanded && "rotate-180"
                )} />
              </button>

              {/* Section Content - animated */}
              <div
                className={cn(
                  "overflow-hidden transition-all duration-300 ease-in-out",
                  isExpanded ? "max-h-[2000px] opacity-100" : "max-h-0 opacity-0"
                )}
              >
                <div className="p-4 pt-1 border-t border-border/30">

                  {/* LLM Settings */}
                  {section.id === 'llm' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">LLM Provider</Label>
                          <select
                            value={llmForm.provider}
                            onChange={(e) => setLlmForm(p => ({ ...p, provider: e.target.value, model: '' }))}
                            className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm mt-1"
                          >
                            <option value="none">None (Disabled)</option>
                            <option value="openai">OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="google">Google (Gemini)</option>
                            <option value="xai">xAI (Grok)</option>
                            <option value="deepseek">DeepSeek</option>
                            <option value="ollama">Ollama (Local)</option>
                            <option value="lmstudio">LM Studio (Local)</option>
                          </select>
                        </div>

                        {(llmForm.provider === 'openai' || llmForm.provider === 'none') && (
                          <SecretInput
                            label="OpenAI API Key"
                            value={llmForm.openai_api_key}
                            placeholder={settings?.llm.openai_api_key || 'sk-...'}
                            onChange={(v) => setLlmForm(p => ({ ...p, openai_api_key: v }))}
                            showSecret={showSecrets['openai_key']}
                            onToggle={() => toggleSecret('openai_key')}
                          />
                        )}

                        {(llmForm.provider === 'anthropic' || llmForm.provider === 'none') && (
                          <SecretInput
                            label="Anthropic API Key"
                            value={llmForm.anthropic_api_key}
                            placeholder={settings?.llm.anthropic_api_key || 'sk-ant-...'}
                            onChange={(v) => setLlmForm(p => ({ ...p, anthropic_api_key: v }))}
                            showSecret={showSecrets['anthropic_key']}
                            onToggle={() => toggleSecret('anthropic_key')}
                          />
                        )}

                        {(llmForm.provider === 'google' || llmForm.provider === 'none') && (
                          <SecretInput
                            label="Google (Gemini) API Key"
                            value={llmForm.google_api_key}
                            placeholder={settings?.llm.google_api_key || 'AIza...'}
                            onChange={(v) => setLlmForm(p => ({ ...p, google_api_key: v }))}
                            showSecret={showSecrets['google_key']}
                            onToggle={() => toggleSecret('google_key')}
                          />
                        )}

                        {(llmForm.provider === 'xai' || llmForm.provider === 'none') && (
                          <SecretInput
                            label="xAI (Grok) API Key"
                            value={llmForm.xai_api_key}
                            placeholder={settings?.llm.xai_api_key || 'xai-...'}
                            onChange={(v) => setLlmForm(p => ({ ...p, xai_api_key: v }))}
                            showSecret={showSecrets['xai_key']}
                            onToggle={() => toggleSecret('xai_key')}
                          />
                        )}

                        {(llmForm.provider === 'deepseek' || llmForm.provider === 'none') && (
                          <SecretInput
                            label="DeepSeek API Key"
                            value={llmForm.deepseek_api_key}
                            placeholder={settings?.llm.deepseek_api_key || 'sk-...'}
                            onChange={(v) => setLlmForm(p => ({ ...p, deepseek_api_key: v }))}
                            showSecret={showSecrets['deepseek_key']}
                            onToggle={() => toggleSecret('deepseek_key')}
                          />
                        )}

                        {(llmForm.provider === 'ollama' || llmForm.provider === 'none') && (
                          <>
                            <div>
                              <Label className="text-xs text-muted-foreground">Ollama Base URL</Label>
                              <Input
                                type="text"
                                value={llmForm.ollama_base_url}
                                onChange={(e) => setLlmForm(p => ({ ...p, ollama_base_url: e.target.value }))}
                                placeholder="http://localhost:11434"
                                className="mt-1 text-sm font-mono"
                              />
                              <p className="text-[11px] text-muted-foreground/70 mt-1">
                                Uses the OpenAI-compatible endpoint at /v1. Leave blank for default localhost URL.
                              </p>
                            </div>
                            <SecretInput
                              label="Ollama API Key (Optional)"
                              value={llmForm.ollama_api_key}
                              placeholder={settings?.llm.ollama_api_key || 'Optional'}
                              onChange={(v) => setLlmForm(p => ({ ...p, ollama_api_key: v }))}
                              showSecret={showSecrets['ollama_key']}
                              onToggle={() => toggleSecret('ollama_key')}
                            />
                          </>
                        )}

                        {(llmForm.provider === 'lmstudio' || llmForm.provider === 'none') && (
                          <>
                            <div>
                              <Label className="text-xs text-muted-foreground">LM Studio Base URL</Label>
                              <Input
                                type="text"
                                value={llmForm.lmstudio_base_url}
                                onChange={(e) => setLlmForm(p => ({ ...p, lmstudio_base_url: e.target.value }))}
                                placeholder="http://localhost:1234/v1"
                                className="mt-1 text-sm font-mono"
                              />
                              <p className="text-[11px] text-muted-foreground/70 mt-1">
                                OpenAI-compatible server URL. Leave blank for default localhost URL.
                              </p>
                            </div>
                            <SecretInput
                              label="LM Studio API Key (Optional)"
                              value={llmForm.lmstudio_api_key}
                              placeholder={settings?.llm.lmstudio_api_key || 'Optional'}
                              onChange={(v) => setLlmForm(p => ({ ...p, lmstudio_api_key: v }))}
                              showSecret={showSecrets['lmstudio_key']}
                              onToggle={() => toggleSecret('lmstudio_key')}
                            />
                          </>
                        )}

                        <div>
                          <Label className="text-xs text-muted-foreground">Model</Label>
                          <div className="flex gap-2 mt-1">
                            <select
                              value={llmForm.model}
                              onChange={(e) => setLlmForm(p => ({ ...p, model: e.target.value }))}
                              className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm"
                            >
                              <option value="">Select a model...</option>
                              {modelsForProvider.map(m => (
                                <option key={m.id} value={m.id}>{m.name}</option>
                              ))}
                              {llmForm.model && !modelsForProvider.find(m => m.id === llmForm.model) && (
                                <option value={llmForm.model}>{llmForm.model} (current)</option>
                              )}
                            </select>
                            <Button
                              variant="secondary"
                              size="icon"
                              onClick={handleRefreshModels}
                              disabled={isRefreshingModels || llmForm.provider === 'none'}
                              title="Refresh models from provider API"
                            >
                              <RefreshCw className={cn("w-4 h-4", isRefreshingModels && "animate-spin")} />
                            </Button>
                          </div>
                          <p className="text-[11px] text-muted-foreground/70 mt-1">
                            {modelsForProvider.length > 0
                              ? `${modelsForProvider.length} models available`
                              : llmForm.provider !== 'none'
                                ? 'Click refresh to fetch available models from the API'
                                : 'Select a provider first'}
                          </p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div>
                        <Label className="text-xs text-muted-foreground">Monthly Spend Limit (USD)</Label>
                        <div className="flex items-center gap-3 mt-1">
                          <DollarSign className="w-4 h-4 text-muted-foreground" />
                          <Input
                            type="number"
                            min={0}
                            step={5}
                            value={llmForm.max_monthly_spend}
                            onChange={(e) => setLlmForm(p => ({ ...p, max_monthly_spend: parseFloat(e.target.value) || 0 }))}
                            className="w-40 text-sm"
                          />
                        </div>
                        <p className="text-[11px] text-muted-foreground/70 mt-1">
                          LLM requests will be blocked once monthly spend reaches this limit. Set to 0 to disable the limit.
                        </p>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('llm')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => testLlmMutation.mutate()}
                          disabled={testLlmMutation.isPending}
                        >
                          <MessageSquare className="w-3.5 h-3.5 mr-1.5" />
                          Test
                        </Button>
                        {testLlmMutation.data && (
                          <Badge
                            variant={testLlmMutation.data.status === 'success' ? "default" : "outline"}
                            className={cn(
                              "text-xs",
                              testLlmMutation.data.status === 'success'
                                ? "bg-green-500/10 text-green-400"
                                : testLlmMutation.data.status === 'warning'
                                  ? "bg-yellow-500/10 text-yellow-400"
                                  : "bg-red-500/10 text-red-400"
                            )}
                          >
                            {testLlmMutation.data.message}
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Scanner Settings */}
                  {section.id === 'scanner' && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">Scan Interval (seconds)</Label>
                          <Input
                            type="number"
                            value={scannerForm.scan_interval_seconds}
                            onChange={(e) => setScannerForm(p => ({ ...p, scan_interval_seconds: Math.max(10, parseInt(e.target.value) || 60) }))}
                            min={10}
                            max={3600}
                            className="mt-1 text-sm"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Min Profit Threshold (%)</Label>
                          <Input
                            type="number"
                            value={scannerForm.min_profit_threshold}
                            onChange={(e) => setScannerForm(p => ({ ...p, min_profit_threshold: Math.max(0, parseFloat(e.target.value) || 0) }))}
                            min={0}
                            step={0.1}
                            className="mt-1 text-sm"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Min Liquidity (USD)</Label>
                          <Input
                            type="number"
                            value={scannerForm.min_liquidity}
                            onChange={(e) => setScannerForm(p => ({ ...p, min_liquidity: Math.max(0, parseFloat(e.target.value) || 0) }))}
                            min={0}
                            step={100}
                            className="mt-1 text-sm"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Max Markets / Scan</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_markets_to_scan}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_markets_to_scan: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={200000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Set `0` for no cap (full active universe).</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Max Events / Scan</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_events_to_scan}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_events_to_scan: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={200000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Set `0` for no cap (full active universe).</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Market Fetch Page Size</Label>
                          <Input
                            type="number"
                            value={scannerForm.market_fetch_page_size}
                            onChange={(e) => setScannerForm(p => ({ ...p, market_fetch_page_size: Math.max(50, parseInt(e.target.value) || 50) }))}
                            min={50}
                            max={500}
                            className="mt-1 text-sm"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Market Fetch Order</Label>
                          <Input
                            type="text"
                            value={scannerForm.market_fetch_order}
                            onChange={(e) => setScannerForm(p => ({ ...p, market_fetch_order: e.target.value }))}
                            placeholder="volume"
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">`volume`, `updatedAt`, `createdAt`, or empty</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Max Opportunities (Total)</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_opportunities_total}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_opportunities_total: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={50000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Set `0` to disable</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Max Opportunities / Strategy</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_opportunities_per_strategy}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_opportunities_per_strategy: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={10000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Set `0` to disable</p>
                        </div>
                      </div>
                      <Separator className="opacity-30" />
                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('scanner')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Notification Settings */}
                  {section.id === 'notifications' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Enable Notifications</p>
                              <p className="text-xs text-muted-foreground">Receive alerts via Telegram</p>
                            </div>
                            <Switch
                              checked={notificationsForm.enabled}
                              onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <SecretInput
                          label="Telegram Bot Token"
                          value={notificationsForm.telegram_bot_token}
                          placeholder={settings?.notifications.telegram_bot_token || 'Enter bot token'}
                          onChange={(v) => setNotificationsForm(p => ({ ...p, telegram_bot_token: v }))}
                          showSecret={showSecrets['tg_token']}
                          onToggle={() => toggleSecret('tg_token')}
                          description="Get this from @BotFather on Telegram"
                        />

                        <div>
                          <Label className="text-xs text-muted-foreground">Telegram Chat ID</Label>
                          <Input
                            type="text"
                            value={notificationsForm.telegram_chat_id}
                            onChange={(e) => setNotificationsForm(p => ({ ...p, telegram_chat_id: e.target.value }))}
                            placeholder="Your chat ID"
                            className="mt-1 text-sm"
                          />
                        </div>

                        <div className="space-y-2 pt-2">
                          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Alert Types</p>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">Autotrader Timeline</p>
                                <p className="text-xs text-muted-foreground">Periodic timeline summaries while orchestrator is running</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_autotrader_timeline}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_autotrader_timeline: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">Autotrader Issue Alerts</p>
                                <p className="text-xs text-muted-foreground">Immediate alerts for kill switch, preflight failure, order failures, and worker errors</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_autotrader_issues}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_autotrader_issues: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">Autotrader Order Alerts</p>
                                <p className="text-xs text-muted-foreground">Immediate order activity summaries per cycle</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_autotrader_orders}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_autotrader_orders: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">Autotrader Close Alerts</p>
                                <p className="text-xs text-muted-foreground">Immediate alert when a position closes or resolves</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_autotrader_closes}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_autotrader_closes: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">Per-Trader Timeline Breakdown</p>
                                <p className="text-xs text-muted-foreground">Include trader-level lines in timeline summaries</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_autotrader_summary_per_trader}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_autotrader_summary_per_trader: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <div>
                            <Label className="text-xs text-muted-foreground">Autotrader Summary Interval (minutes)</Label>
                            <Input
                              type="number"
                              value={notificationsForm.notify_autotrader_summary_interval_minutes}
                              onChange={(e) => setNotificationsForm(p => ({ ...p, notify_autotrader_summary_interval_minutes: parseInt(e.target.value) || 60 }))}
                              step="5"
                              min="5"
                              max="1440"
                              className="mt-1 text-sm"
                            />
                          </div>

                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2 flex-wrap">
                        <Button size="sm" onClick={() => handleSaveSection('notifications')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => testTelegramMutation.mutate()}
                          disabled={testTelegramMutation.isPending}
                        >
                          <MessageSquare className="w-3.5 h-3.5 mr-1.5" />
                          Test Telegram
                        </Button>
                        {testTelegramMutation.data && (
                          <Badge variant={testTelegramMutation.data.status === 'success' ? "default" : "outline"} className={cn(
                            "text-xs",
                            testTelegramMutation.data.status === 'success' ? "bg-green-500/10 text-green-400" : "bg-yellow-500/10 text-yellow-400"
                          )}>
                            {testTelegramMutation.data.message}
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}

                  {/* UI Lock Settings */}
                  {section.id === 'security' && (
                    <div className="space-y-4">
                      <Card className="bg-muted border-emerald-500/30">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="font-medium text-sm">Enable UI Lock</p>
                            <p className="text-xs text-muted-foreground">
                              Locks the UI after inactivity while backend workers continue running.
                            </p>
                          </div>
                          <Switch
                            checked={uiLockForm.enabled}
                            onCheckedChange={(checked) => setUiLockForm((prev) => ({ ...prev, enabled: checked }))}
                          />
                        </CardContent>
                      </Card>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">Idle Timeout (minutes)</Label>
                          <Input
                            type="number"
                            value={uiLockForm.idle_timeout_minutes}
                            onChange={(e) => {
                              const parsed = parseInt(e.target.value, 10)
                              setUiLockForm((prev) => ({
                                ...prev,
                                idle_timeout_minutes: Number.isFinite(parsed) ? Math.max(1, Math.min(1440, parsed)) : 15,
                              }))
                            }}
                            min={1}
                            max={1440}
                            className="mt-1 text-sm"
                          />
                        </div>
                        <div className="flex items-end">
                          <p className="text-[11px] text-muted-foreground/70">
                            Password configured: {uiLockForm.has_password ? 'Yes' : 'No'}
                          </p>
                        </div>
                      </div>

                      <SecretInput
                        label="Set New Password"
                        value={uiLockForm.password}
                        placeholder={uiLockForm.has_password ? 'Leave blank to keep existing' : 'Enter a password'}
                        onChange={(value) => setUiLockForm((prev) => ({ ...prev, password: value }))}
                        showSecret={showSecrets['ui_lock_password']}
                        onToggle={() => toggleSecret('ui_lock_password')}
                      />

                      <SecretInput
                        label="Confirm Password"
                        value={uiLockForm.confirm_password}
                        placeholder="Re-enter password"
                        onChange={(value) => setUiLockForm((prev) => ({ ...prev, confirm_password: value }))}
                        showSecret={showSecrets['ui_lock_confirm_password']}
                        onToggle={() => toggleSecret('ui_lock_confirm_password')}
                      />

                      <Card className="bg-muted">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="text-sm">Clear Stored Password</p>
                            <p className="text-xs text-muted-foreground">
                              Only use this when UI lock is disabled, or while setting a replacement password now.
                            </p>
                          </div>
                          <Switch
                            checked={uiLockForm.clear_password}
                            onCheckedChange={(checked) => setUiLockForm((prev) => ({ ...prev, clear_password: checked }))}
                          />
                        </CardContent>
                      </Card>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('security')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Discovery Settings */}
                  {section.id === 'discovery' && (
                    <div className="space-y-4">
                      <Card className="bg-muted border-green-500/30">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="font-medium text-sm">Discovery Catalog Maintenance</p>
                            <p className="text-xs text-muted-foreground">
                              Control catalog growth, cleanup cadence, and retention policy
                            </p>
                          </div>
                          <Switch
                            checked={discoveryForm.maintenance_enabled}
                            onCheckedChange={(checked) => setDiscoveryForm(p => ({ ...p, maintenance_enabled: checked }))}
                          />
                        </CardContent>
                      </Card>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">Max Discovered Wallets</Label>
                          <Input
                            type="number"
                            value={discoveryForm.max_discovered_wallets}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, max_discovered_wallets: parseInt(e.target.value) || 20_000 }))}
                            min={10}
                            max={1_000_000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Max rows kept in wallet catalog</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Discovery Maintenance Batch</Label>
                          <Input
                            type="number"
                            value={discoveryForm.maintenance_batch}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, maintenance_batch: parseInt(e.target.value) || 900 }))}
                            min={10}
                            max={5000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Chunk size for remove/insert operations</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Keep Wallets w/ Recent Trades (days)</Label>
                          <Input
                            type="number"
                            value={discoveryForm.keep_recent_trade_days}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, keep_recent_trade_days: parseInt(e.target.value) || 7 }))}
                            min={1}
                            max={365}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Protect wallets that traded recently</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Keep Newly Discovered Wallets (days)</Label>
                          <Input
                            type="number"
                            value={discoveryForm.keep_new_discoveries_days}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, keep_new_discoveries_days: parseInt(e.target.value) || 30 }))}
                            min={1}
                            max={365}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Protect wallets found in this window</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Stale Analysis Threshold (hours)</Label>
                          <Input
                            type="number"
                            value={discoveryForm.stale_analysis_hours}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, stale_analysis_hours: parseInt(e.target.value) || 12 }))}
                            min={1}
                            max={720}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Re-analyze wallets older than this age</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Priority Queue Limit</Label>
                          <Input
                            type="number"
                            value={discoveryForm.analysis_priority_batch_limit}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, analysis_priority_batch_limit: parseInt(e.target.value) || 2500 }))}
                            min={100}
                            max={10_000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">High-priority queue cap for new/stale wallets</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Delay Between Markets (s)</Label>
                          <Input
                            type="number"
                            value={discoveryForm.delay_between_markets}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, delay_between_markets: parseFloat(e.target.value) || 0 }))}
                            min={0}
                            max={10}
                            step={0.05}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Throttling between market scans</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Delay Between Wallet Analysis (s)</Label>
                          <Input
                            type="number"
                            value={discoveryForm.delay_between_wallets}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, delay_between_wallets: parseFloat(e.target.value) || 0 }))}
                            min={0}
                            max={10}
                            step={0.05}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Throttle wallet analysis loop</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Max Markets Per Discovery Run</Label>
                          <Input
                            type="number"
                            value={discoveryForm.max_markets_per_run}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, max_markets_per_run: parseInt(e.target.value) || 100 }))}
                            min={1}
                            max={1_000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">How many active markets to sample</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Max Wallets Per Market</Label>
                          <Input
                            type="number"
                            value={discoveryForm.max_wallets_per_market}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, max_wallets_per_market: parseInt(e.target.value) || 50 }))}
                            min={1}
                            max={500}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Wallets extracted per sampled market</p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('discovery')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* VPN/Proxy Settings */}
                  {section.id === 'vpn' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <Card className="bg-muted border-indigo-500/30">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Enable Trading Proxy</p>
                              <p className="text-xs text-muted-foreground">Route Polymarket/Kalshi trading requests through the proxy below</p>
                            </div>
                            <Switch
                              checked={vpnForm.enabled}
                              onCheckedChange={(checked) => setVpnForm(p => ({ ...p, enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <SecretInput
                          label="Proxy URL"
                          value={vpnForm.proxy_url}
                          placeholder={settings?.trading_proxy?.proxy_url || 'socks5://user:pass@host:port'}
                          onChange={(v) => setVpnForm(p => ({ ...p, proxy_url: v }))}
                          showSecret={showSecrets['proxy_url']}
                          onToggle={() => toggleSecret('proxy_url')}
                          description="Supports socks5://, http://, https:// proxy URLs"
                        />

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Request Timeout (seconds)</Label>
                            <Input
                              type="number"
                              value={vpnForm.timeout}
                              onChange={(e) => setVpnForm(p => ({ ...p, timeout: parseFloat(e.target.value) || 30 }))}
                              min={5}
                              max={120}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="text-sm font-medium">Verify SSL Certificates</p>
                              <p className="text-xs text-muted-foreground">Verify SSL certs for requests through the proxy</p>
                            </div>
                            <Switch
                              checked={vpnForm.verify_ssl}
                              onCheckedChange={(checked) => setVpnForm(p => ({ ...p, verify_ssl: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <Card className="bg-muted border-yellow-500/20">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="text-sm font-medium">Require VPN for Trading</p>
                              <p className="text-xs text-muted-foreground">Block all trades if the VPN proxy is unreachable (recommended)</p>
                            </div>
                            <Switch
                              checked={vpnForm.require_vpn}
                              onCheckedChange={(checked) => setVpnForm(p => ({ ...p, require_vpn: checked }))}
                            />
                          </CardContent>
                        </Card>

                        {/* Info box */}
                        <div className="flex items-start gap-2 p-3 bg-indigo-500/5 border border-indigo-500/20 rounded-lg">
                          <Shield className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
                          <p className="text-xs text-muted-foreground">
                            Only actual trading requests (order placement, cancellation) are routed through the proxy.
                            Market scanning, price feeds, and all other data remain on your direct connection for maximum speed.
                          </p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2 flex-wrap">
                        <Button size="sm" onClick={() => handleSaveSection('vpn')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => testVpnMutation.mutate()}
                          disabled={testVpnMutation.isPending}
                        >
                          <Shield className="w-3.5 h-3.5 mr-1.5" />
                          Test VPN
                        </Button>
                        {testVpnMutation.data && (
                          <Badge variant={testVpnMutation.data.status === 'success' ? "default" : "outline"} className={cn(
                            "text-xs",
                            testVpnMutation.data.status === 'success' ? "bg-green-500/10 text-green-400"
                              : testVpnMutation.data.status === 'warning' ? "bg-yellow-500/10 text-yellow-400"
                              : "bg-red-500/10 text-red-400"
                          )}>
                            {testVpnMutation.data.message}
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Import/Export Settings */}
                  {section.id === 'transfer' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Categories</p>
                          <div className="flex items-center gap-2">
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              className="h-7 px-2 text-[11px]"
                              onClick={() => setAllTransferCategories(true)}
                            >
                              All
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              className="h-7 px-2 text-[11px]"
                              onClick={() => setAllTransferCategories(false)}
                            >
                              None
                            </Button>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {SETTINGS_TRANSFER_CATEGORIES.map((category) => (
                            <Card key={category.id} className="bg-muted border-border/50">
                              <CardContent className="p-3">
                                <div className="flex items-start justify-between gap-2">
                                  <div>
                                    <p className="text-sm font-medium">{category.label}</p>
                                    <p className="text-xs text-muted-foreground">{category.description}</p>
                                  </div>
                                  <Switch
                                    checked={transferCategories[category.id]}
                                    onCheckedChange={() => toggleTransferCategory(category.id)}
                                  />
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>

                        <Card className="bg-amber-500/5 border-amber-500/25">
                          <CardContent className="p-3">
                            <p className="text-xs text-amber-300 font-medium">Sensitive export warning</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Export bundles include plaintext API credentials for selected categories. Store them securely.
                            </p>
                          </CardContent>
                        </Card>

                        <Card className="bg-muted border-border/50">
                          <CardContent className="p-3 space-y-2">
                            <p className="text-xs uppercase tracking-widest text-muted-foreground">Import Bundle</p>
                            <input
                              ref={transferFileInputRef}
                              type="file"
                              accept="application/json,.json"
                              className="hidden"
                              onChange={handleTransferFileSelect}
                            />
                            <div className="flex items-center gap-2 flex-wrap">
                              <Button
                                type="button"
                                variant="secondary"
                                size="sm"
                                onClick={() => transferFileInputRef.current?.click()}
                              >
                                <Upload className="w-3.5 h-3.5 mr-1.5" />
                                Choose JSON
                              </Button>
                              <span className="text-xs text-muted-foreground">
                                {importFileName || 'No file selected'}
                              </span>
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2 flex-wrap">
                        <Button
                          size="sm"
                          onClick={handleExportBundle}
                          disabled={exportSettingsMutation.isPending || !selectedTransferCategories.length}
                        >
                          {exportSettingsMutation.isPending ? (
                            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          ) : (
                            <Download className="w-3.5 h-3.5 mr-1.5" />
                          )}
                          Export Bundle
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={handleImportBundle}
                          disabled={importSettingsMutation.isPending || !importBundle || !selectedTransferCategories.length}
                        >
                          {importSettingsMutation.isPending ? (
                            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          ) : (
                            <Upload className="w-3.5 h-3.5 mr-1.5" />
                          )}
                          Import Bundle
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Maintenance Settings */}
                  {section.id === 'maintenance' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <Card className="bg-muted">
                          <CardContent className="p-3 space-y-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-medium text-sm">Database Metrics</p>
                                <p className="text-xs text-muted-foreground">Current footprint and row volume</p>
                              </div>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs"
                                onClick={() => maintenanceStatsQuery.refetch()}
                                disabled={maintenanceStatsQuery.isFetching}
                              >
                                {maintenanceStatsQuery.isFetching ? (
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                ) : (
                                  <RefreshCw className="w-3.5 h-3.5" />
                                )}
                              </Button>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              <div className="rounded-lg border border-border/50 bg-card/40 px-3 py-2">
                                <p className="text-[11px] uppercase tracking-widest text-muted-foreground">DB Size</p>
                                <p className="text-sm font-semibold">
                                  {formatDbBytes(maintenanceStatsQuery.data?.db_size_bytes)}
                                </p>
                              </div>
                              <div className="rounded-lg border border-border/50 bg-card/40 px-3 py-2">
                                <p className="text-[11px] uppercase tracking-widest text-muted-foreground">Total Rows</p>
                                <p className="text-sm font-semibold">
                                  {maintenanceStatsQuery.data?.total_rows != null
                                    ? maintenanceStatsQuery.data.total_rows.toLocaleString()
                                    : 'Unavailable'}
                                </p>
                              </div>
                            </div>

                            {maintenanceStatsQuery.isError && (
                              <p className="text-[11px] text-red-400">Unable to load database metrics.</p>
                            )}

                            {!maintenanceStatsQuery.isLoading && maintenanceStatsQuery.dataUpdatedAt > 0 && (
                              <p className="text-[10px] text-muted-foreground">
                                Updated {new Date(maintenanceStatsQuery.dataUpdatedAt).toLocaleString()}
                              </p>
                            )}
                          </CardContent>
                        </Card>

                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Auto Cleanup</p>
                              <p className="text-xs text-muted-foreground">Automatically delete old records</p>
                            </div>
                            <Switch
                              checked={maintenanceForm.auto_cleanup_enabled}
                              onCheckedChange={(checked) => setMaintenanceForm(p => ({ ...p, auto_cleanup_enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Cleanup Interval (hours)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_interval_hours}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_interval_hours: parseInt(e.target.value) || 24 }))}
                              min={1}
                              max={168}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Keep Resolved Trades (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_resolved_trade_days}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_resolved_trade_days: parseInt(e.target.value) || 30 }))}
                              min={1}
                              max={365}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">LLM Usage Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.llm_usage_retention_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, llm_usage_retention_days: Number.isNaN(value) ? 30 : value }))
                              }}
                              min={0}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Trade Signal Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_trade_signal_emission_days}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_trade_signal_emission_days: parseInt(e.target.value) || 21 }))}
                              min={1}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Trade Signal Update Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_trade_signal_update_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, cleanup_trade_signal_update_days: Number.isNaN(value) ? 3 : value }))
                              }}
                              min={0}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Wallet Activity Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_wallet_activity_rollup_days}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_wallet_activity_rollup_days: parseInt(e.target.value) || 60 }))}
                              min={45}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Wallet Activity Duplicate Cleanup</p>
                              <p className="text-xs text-muted-foreground">
                                Remove duplicate rollups during scheduled maintenance
                              </p>
                            </div>
                            <Switch
                              checked={maintenanceForm.cleanup_wallet_activity_dedupe_enabled}
                              onCheckedChange={(checked) => setMaintenanceForm(p => ({ ...p, cleanup_wallet_activity_dedupe_enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Market Metadata Hygiene</p>
                              <p className="text-xs text-muted-foreground">
                                Prune stale or poisoned cached market names/slugs
                              </p>
                            </div>
                            <Switch
                              checked={maintenanceForm.market_cache_hygiene_enabled}
                              onCheckedChange={(checked) => setMaintenanceForm(p => ({ ...p, market_cache_hygiene_enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Metadata Hygiene Interval (hours)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.market_cache_hygiene_interval_hours}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, market_cache_hygiene_interval_hours: parseInt(e.target.value) || 6 }))}
                              min={1}
                              max={168}
                              className="mt-1 text-sm"
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Metadata Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.market_cache_retention_days}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, market_cache_retention_days: parseInt(e.target.value) || 120 }))}
                              min={7}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Reference Lookback (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.market_cache_reference_lookback_days}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, market_cache_reference_lookback_days: parseInt(e.target.value) || 45 }))}
                              min={1}
                              max={365}
                              className="mt-1 text-sm"
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Weak Entry Grace (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.market_cache_weak_entry_grace_days}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, market_cache_weak_entry_grace_days: parseInt(e.target.value) || 7 }))}
                              min={1}
                              max={180}
                              className="mt-1 text-sm"
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Entries Per Slug</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.market_cache_max_entries_per_slug}
                              onChange={(e) => setMaintenanceForm(p => ({ ...p, market_cache_max_entries_per_slug: parseInt(e.target.value) || 3 }))}
                              min={1}
                              max={50}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <Card className="bg-red-500/5 border-red-500/20">
                          <CardContent className="p-3 space-y-3">
                            <div>
                              <p className="font-medium text-sm">Manual Data Flush</p>
                              <p className="text-xs text-muted-foreground">
                                Manually clear runtime/cache datasets for scanner, weather, news, and trader orchestrator pipelines.
                              </p>
                              <p className="text-[11px] text-emerald-400/80 mt-1">
                                Protected automatically: live/executed positions and trade history.
                              </p>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                className="justify-start border-red-500/30 hover:bg-red-500/10"
                                onClick={() => handleFlushTarget('scanner')}
                                disabled={flushDataMutation.isPending}
                              >
                                {flushDataMutation.isPending && activeFlushTarget === 'scanner'
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Trash2 className="w-3.5 h-3.5 mr-1.5" />}
                                Flush Scanner/Market
                              </Button>

                              <Button
                                variant="outline"
                                size="sm"
                                className="justify-start border-red-500/30 hover:bg-red-500/10"
                                onClick={() => handleFlushTarget('weather')}
                                disabled={flushDataMutation.isPending}
                              >
                                {flushDataMutation.isPending && activeFlushTarget === 'weather'
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Trash2 className="w-3.5 h-3.5 mr-1.5" />}
                                Flush Weather
                              </Button>

                              <Button
                                variant="outline"
                                size="sm"
                                className="justify-start border-red-500/30 hover:bg-red-500/10"
                                onClick={() => handleFlushTarget('news')}
                                disabled={flushDataMutation.isPending}
                              >
                                {flushDataMutation.isPending && activeFlushTarget === 'news'
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Trash2 className="w-3.5 h-3.5 mr-1.5" />}
                                Flush News
                              </Button>

                              <Button
                                variant="outline"
                                size="sm"
                                className="justify-start border-red-500/30 hover:bg-red-500/10"
                                onClick={() => handleFlushTarget('trader_orchestrator')}
                                disabled={flushDataMutation.isPending}
                              >
                                {flushDataMutation.isPending && activeFlushTarget === 'trader_orchestrator'
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Trash2 className="w-3.5 h-3.5 mr-1.5" />}
                                Flush Trader Orchestrator Runtime
                              </Button>
                            </div>

                            <div className="flex items-center gap-2 flex-wrap">
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-red-500/40 hover:bg-red-500/15"
                                onClick={() => handleFlushTarget('all')}
                                disabled={flushDataMutation.isPending}
                              >
                                {flushDataMutation.isPending && activeFlushTarget === 'all'
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Trash2 className="w-3.5 h-3.5 mr-1.5" />}
                                Flush All Non-Trading Data
                              </Button>

                              <Button
                                variant="secondary"
                                size="sm"
                                onClick={() => runOrchestratorOnceMutation.mutate()}
                                disabled={runOrchestratorOnceMutation.isPending}
                              >
                                {runOrchestratorOnceMutation.isPending
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Play className="w-3.5 h-3.5 mr-1.5" />}
                                Run Trader Orchestrator Once
                              </Button>

                              <p className="text-[10px] text-muted-foreground">
                                Trader orchestrator run is non-destructive and queues one immediate cycle.
                              </p>
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('maintenance')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                      </div>
                    </div>
                  )}

                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
