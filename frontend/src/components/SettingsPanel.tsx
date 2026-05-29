import { useState, useEffect, useRef, type ChangeEvent, type ReactNode } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import LanguageSwitcher from './LanguageSwitcher'
import {
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
  Shield,
  ChevronDown,
  Loader2,
  Trash2,
  Play,
  Download,
  Upload,
  Wifi,
  Search,
  X,
  Tag,
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
  flushDatabaseData,
  getDatabaseMaintenanceStats,
  runVacuumAnalyze,
  runReindexTables,
  runWorkerOnce,
  pauseWorker,
  startWorker,
  type DatabaseFlushTarget,
} from '../services/apiTraders'
import {
  getProviderSettings,
  updateProviderSettings,
  type ProviderSettings,
} from '../services/apiProviders'
import {
  getSettings,
  updateSettings,
  testTelegramConnection,
  testTradingProxy,
  exportSettingsBundle,
  importSettingsBundle,
  getMarketFilterAvailableTags,
  type DiscoverySettings,
  type UILockSettings,
  type SettingsTransferCategory,
  type SettingsExportBundle,
  type MarketFilterAvailableTag,
} from '../services/apiSettings'

type SettingsSection =
  | 'search'
  | 'scanner'
  | 'notifications'
  | 'security'
  | 'vpn'
  | 'rpc'
  | 'network'
  | 'discovery'
  | 'providers'
  | 'maintenance'
  | 'transfer'

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
  labelKey: string
  descriptionKey: string
}> = [
  { id: 'bot_traders', labelKey: 'settings.transfer.botTraders', descriptionKey: 'settings.transfer.botTradersDesc' },
  { id: 'strategies', labelKey: 'settings.transfer.strategies', descriptionKey: 'settings.transfer.strategiesDesc' },
  { id: 'data_sources', labelKey: 'settings.transfer.dataSources', descriptionKey: 'settings.transfer.dataSourcesDesc' },
  { id: 'market_credentials', labelKey: 'settings.transfer.marketCredentials', descriptionKey: 'settings.transfer.marketCredentialsDesc' },
  { id: 'vpn_configuration', labelKey: 'settings.transfer.vpnConfiguration', descriptionKey: 'settings.transfer.vpnConfigurationDesc' },
  { id: 'llm_configuration', labelKey: 'settings.transfer.llmConfiguration', descriptionKey: 'settings.transfer.llmConfigurationDesc' },
  {
    id: 'telegram_configuration',
    labelKey: 'settings.transfer.telegramSetup',
    descriptionKey: 'settings.transfer.telegramSetupDesc',
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
  const { t } = useTranslation()
  const [expandedSections, setExpandedSections] = useState<Set<SettingsSection>>(new Set())
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [activeFlushTarget, setActiveFlushTarget] = useState<DatabaseFlushTarget | null>(null)

  // Form state for each section
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
    skipped_signal_reactivation_cooldown_seconds: 180,
    strict_ws_max_age_ms: 30000,
    market_filter_tags: [] as string[],
    crypto_lane_enabled: true,
  })
  const [cryptoLaneToggling, setCryptoLaneToggling] = useState(false)
  const [cryptoLaneError, setCryptoLaneError] = useState<string | null>(null)

  const [maintenanceForm, setMaintenanceForm] = useState({
    auto_cleanup_enabled: false,
    cleanup_interval_hours: 24,
    cleanup_resolved_trade_days: 30,
    cleanup_trade_signal_emission_days: 21,
    cleanup_trade_signal_update_days: 3,
    cleanup_wallet_activity_rollup_days: 60,
    cleanup_wallet_activity_dedupe_enabled: true,
    llm_usage_retention_days: 30,
    trader_events_firehose_retention_days: 7,
    trader_events_other_retention_days: 90,
    cleanup_wallet_monitor_events_days: 14,
    cleanup_trader_decision_checks_days: 14,
    cleanup_trader_decisions_days: 30,
    cleanup_opportunity_history_days: 30,
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

  const [rpcForm, setRpcForm] = useState({
    rpc_url: '',
    ws_url: '',
  })

  const [uiLockForm, setUiLockForm] = useState({
    enabled: DEFAULT_UI_LOCK_SETTINGS.enabled,
    idle_timeout_minutes: DEFAULT_UI_LOCK_SETTINGS.idle_timeout_minutes,
    has_password: DEFAULT_UI_LOCK_SETTINGS.has_password,
    password: '',
    confirm_password: '',
    clear_password: false,
  })

  const [networkForm, setNetworkForm] = useState({
    allow_network_access: false,
  })

  // Data Providers — separate from /api/settings (uses /api/providers/settings).
  const providerSettingsQuery = useQuery({
    queryKey: ['providers', 'settings'],
    queryFn: getProviderSettings,
    staleTime: 60_000,
  })
  const [providerForm, setProviderForm] = useState({
    polybacktest_api_key: '',
    polybacktest_base_url: '',
    telonex_api_key: '',
    telonex_base_url: '',
    reverse_engineer_max_iterations: '',
    reverse_engineer_target_score: '',
    reverse_engineer_max_cost_usd: '',
    reverse_engineer_max_wallet_trades: '',
  })
  const [showProviderKey, setShowProviderKey] = useState(false)
  const [showTelonexKey, setShowTelonexKey] = useState(false)
  useEffect(() => {
    const s: ProviderSettings | undefined = providerSettingsQuery.data
    if (!s) return
    setProviderForm({
      polybacktest_api_key: s.polybacktest_api_key_set ? '********' : '',
      polybacktest_base_url: s.polybacktest_base_url ?? '',
      telonex_api_key: s.telonex_api_key_set ? '********' : '',
      telonex_base_url: s.telonex_base_url ?? '',
      reverse_engineer_max_iterations: s.reverse_engineer_max_iterations?.toString() ?? '',
      reverse_engineer_target_score: s.reverse_engineer_target_score?.toString() ?? '',
      reverse_engineer_max_cost_usd: s.reverse_engineer_max_cost_usd?.toString() ?? '',
      reverse_engineer_max_wallet_trades: s.reverse_engineer_max_wallet_trades?.toString() ?? '',
    })
  }, [providerSettingsQuery.data])
  const saveProviderSettingsMutation = useMutation({
    mutationFn: () =>
      updateProviderSettings({
        polybacktest_api_key:
          providerForm.polybacktest_api_key === '********'
            ? null
            : providerForm.polybacktest_api_key,
        polybacktest_base_url: providerForm.polybacktest_base_url,
        telonex_api_key:
          providerForm.telonex_api_key === '********'
            ? null
            : providerForm.telonex_api_key,
        telonex_base_url: providerForm.telonex_base_url,
        reverse_engineer_max_iterations: providerForm.reverse_engineer_max_iterations
          ? parseInt(providerForm.reverse_engineer_max_iterations, 10)
          : null,
        reverse_engineer_target_score: providerForm.reverse_engineer_target_score
          ? parseFloat(providerForm.reverse_engineer_target_score)
          : null,
        reverse_engineer_max_cost_usd: providerForm.reverse_engineer_max_cost_usd
          ? parseFloat(providerForm.reverse_engineer_max_cost_usd)
          : null,
        reverse_engineer_max_wallet_trades: providerForm.reverse_engineer_max_wallet_trades
          ? parseInt(providerForm.reverse_engineer_max_wallet_trades, 10)
          : null,
      }),
    onSuccess: () => {
      setSaveMessage({ type: 'success', text: 'Provider settings saved' })
      providerSettingsQuery.refetch()
    },
    onError: (err: Error) => {
      setSaveMessage({ type: 'error', text: err.message || 'Save failed' })
    },
  })

  const [searchForm, setSearchForm] = useState({
    search_polymarket_enabled: true,
    search_kalshi_enabled: false,
    search_max_results: 50,
    serpapi_key: '',
    brave_search_key: '',
  })
  const [showSearchSecrets, setShowSearchSecrets] = useState<Record<string, boolean>>({})

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
        skipped_signal_reactivation_cooldown_seconds:
          settings.scanner?.skipped_signal_reactivation_cooldown_seconds ?? 180,
        strict_ws_max_age_ms: settings.scanner?.strict_ws_max_age_ms ?? 30000,
        market_filter_tags: Array.isArray(settings.scanner?.market_filter_tags)
          ? (settings.scanner?.market_filter_tags ?? [])
          : [],
        crypto_lane_enabled: settings.scanner?.crypto_lane_enabled ?? true,
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
        trader_events_firehose_retention_days: settings.maintenance?.trader_events_firehose_retention_days ?? 7,
        trader_events_other_retention_days: settings.maintenance?.trader_events_other_retention_days ?? 90,
        cleanup_wallet_monitor_events_days: settings.maintenance?.cleanup_wallet_monitor_events_days ?? 14,
        cleanup_trader_decision_checks_days: settings.maintenance?.cleanup_trader_decision_checks_days ?? 14,
        cleanup_trader_decisions_days: settings.maintenance?.cleanup_trader_decisions_days ?? 30,
        cleanup_opportunity_history_days: settings.maintenance?.cleanup_opportunity_history_days ?? 30,
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

      // RPC URLs come back masked from the server; don't pre-fill the
      // input or the user would overwrite their stored secret with a
      // masked preview on save.
      setRpcForm({ rpc_url: '', ws_url: '' })

      setUiLockForm({
        enabled: settings.ui_lock?.enabled ?? DEFAULT_UI_LOCK_SETTINGS.enabled,
        idle_timeout_minutes: settings.ui_lock?.idle_timeout_minutes ?? DEFAULT_UI_LOCK_SETTINGS.idle_timeout_minutes,
        has_password: settings.ui_lock?.has_password ?? DEFAULT_UI_LOCK_SETTINGS.has_password,
        password: '',
        confirm_password: '',
        clear_password: false,
      })

      setNetworkForm({
        allow_network_access: settings.network?.allow_network_access ?? false,
      })

      setSearchForm({
        search_polymarket_enabled: settings.search_filters?.search_polymarket_enabled ?? true,
        search_kalshi_enabled: settings.search_filters?.search_kalshi_enabled ?? false,
        search_max_results: settings.search_filters?.search_max_results ?? 50,
        serpapi_key: '',
        brave_search_key: '',
      })

    }
  }, [settings])


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

  const vacuumMutation = useMutation({
    mutationFn: (full: boolean) => runVacuumAnalyze(full),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['maintenance-stats'] })
      const processed = data?.tables_processed ?? 0
      const secs = data?.total_seconds ?? '?'
      setSaveMessage({
        type: 'success',
        text: `VACUUM ANALYZE completed: ${processed} tables in ${secs}s.`,
      })
      setTimeout(() => setSaveMessage(null), 5000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'VACUUM ANALYZE failed' })
      setTimeout(() => setSaveMessage(null), 7000)
    },
  })

  const reindexMutation = useMutation({
    mutationFn: () => runReindexTables(),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['maintenance-stats'] })
      const processed = data?.tables_processed ?? 0
      const secs = data?.total_seconds ?? '?'
      setSaveMessage({
        type: 'success',
        text: `REINDEX completed: ${processed} tables in ${secs}s.`,
      })
      setTimeout(() => setSaveMessage(null), 5000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'REINDEX failed' })
      setTimeout(() => setSaveMessage(null), 7000)
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

  const handleCryptoLaneToggle = async (next: boolean) => {
    setCryptoLaneError(null)
    setCryptoLaneToggling(true)
    setScannerForm((p) => ({ ...p, crypto_lane_enabled: next }))
    try {
      if (next) {
        await startWorker('crypto')
      } else {
        await pauseWorker('crypto')
      }
      queryClient.invalidateQueries({ queryKey: ['settings'] })
    } catch (err: any) {
      setCryptoLaneError(err?.response?.data?.detail || err?.message || 'Failed to toggle crypto lane')
      setScannerForm((p) => ({ ...p, crypto_lane_enabled: !next }))
    } finally {
      setCryptoLaneToggling(false)
    }
  }

  const handleSaveSection = (section: SettingsSection) => {
    const updates: any = {}

    switch (section) {
      case 'search': {
        const sf: Record<string, unknown> = {
          search_polymarket_enabled: searchForm.search_polymarket_enabled,
          search_kalshi_enabled: searchForm.search_kalshi_enabled,
          search_max_results: searchForm.search_max_results,
        }
        if (searchForm.serpapi_key) sf.serpapi_key = searchForm.serpapi_key
        if (searchForm.brave_search_key) sf.brave_search_key = searchForm.brave_search_key
        updates.search_filters = sf
        break
      }
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
      case 'rpc':
        // Only include the RPC fields the user actually changed —
        // empty input means "leave unchanged" (server keeps stored
        // value). To clear a stored URL the user must explicitly type
        // a single space and save (the server treats empty/whitespace
        // as a clear signal via the trim+empty check).
        updates.blockchain_rpc = {} as any
        if (rpcForm.rpc_url) {
          (updates.blockchain_rpc as any).rpc_url = rpcForm.rpc_url
        }
        if (rpcForm.ws_url) {
          (updates.blockchain_rpc as any).ws_url = rpcForm.ws_url
        }
        break
      case 'maintenance':
        updates.maintenance = maintenanceForm
        break
      case 'network':
        updates.network = networkForm
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
      case 'search': {
        const platforms = [
          searchForm.search_polymarket_enabled && 'Poly',
          searchForm.search_kalshi_enabled && 'Kalshi',
        ].filter(Boolean)
        return platforms.length ? platforms.join(' + ') : t('settings.status.disabled')
      }
      case 'notifications':
        return notificationsForm.enabled ? t('settings.status.enabled') : t('settings.status.disabled')
      case 'security':
        return uiLockForm.enabled
          ? t('settings.status.autoLock', { minutes: uiLockForm.idle_timeout_minutes })
          : t('settings.status.disabled')
      case 'scanner':
        return t('settings.status.scannerCaps', {
          total: scannerForm.max_opportunities_total,
          perStrategy: scannerForm.max_opportunities_per_strategy,
          ws: scannerForm.strict_ws_max_age_ms,
        })
      case 'discovery':
        return discoveryForm.maintenance_enabled
          ? t('settings.status.walletCap', { count: discoveryForm.max_discovered_wallets.toLocaleString() })
          : t('settings.status.disabled')
      case 'vpn':
        return vpnForm.enabled ? t('settings.status.active') : t('settings.status.disabled')
      case 'rpc':
        return settings?.blockchain_rpc?.rpc_url
          ? t('settings.status.rpcConfigured', { defaultValue: 'Configured' })
          : t('settings.status.rpcPublic', { defaultValue: 'Public-tier (rate-limited)' })
      case 'network':
        return networkForm.allow_network_access ? t('settings.status.lanEnabled') : t('settings.status.localhostOnly')
      case 'providers': {
        const s = providerSettingsQuery.data
        if (!s) return '—'
        const configured = [
          s.polybacktest_api_key_set ? 'polybacktest' : null,
          s.telonex_api_key_set ? 'telonex' : null,
        ].filter(Boolean)
        if (configured.length === 0) return t('settings.status.polybacktestNotConfigured')
        if (configured.length === 2) return `${configured.length} configured`
        return configured[0] === 'polybacktest'
          ? t('settings.status.polybacktestConfigured')
          : 'telonex configured'
      }
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? t('settings.status.autoCleanOn') : t('settings.status.manual')
      case 'transfer':
        return t('settings.status.selected', { count: selectedTransferCategories.length })
      default:
        return ''
    }
  }

  const getStatusColor = (id: SettingsSection): string => {
    switch (id) {
      case 'search':
        return (searchForm.search_polymarket_enabled || searchForm.search_kalshi_enabled)
          ? 'text-orange-400 bg-orange-500/10' : 'text-muted-foreground bg-muted'
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
      case 'rpc':
        return settings?.blockchain_rpc?.rpc_url
          ? 'text-emerald-400 bg-emerald-500/10'
          : 'text-yellow-400 bg-yellow-500/10'
      case 'network':
        return networkForm.allow_network_access ? 'text-sky-400 bg-sky-500/10' : 'text-muted-foreground bg-muted'
      case 'providers':
        return (providerSettingsQuery.data?.polybacktest_api_key_set || providerSettingsQuery.data?.telonex_api_key_set)
          ? 'text-violet-400 bg-violet-500/10'
          : 'text-muted-foreground bg-muted'
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'text-red-400 bg-red-500/10' : 'text-muted-foreground bg-muted'
      case 'transfer':
        return 'text-cyan-400 bg-cyan-500/10'
      default:
        return 'text-muted-foreground bg-muted'
    }
  }

  const sections: { id: SettingsSection; icon: any; labelKey: string; descriptionKey: string }[] = [
    { id: 'search', icon: Search, labelKey: 'settings.sections.search', descriptionKey: 'settings.sections.searchDesc' },
    { id: 'scanner', icon: Database, labelKey: 'settings.sections.scanner', descriptionKey: 'settings.sections.scannerDesc' },
    { id: 'notifications', icon: Bell, labelKey: 'settings.sections.notifications', descriptionKey: 'settings.sections.notificationsDesc' },
    { id: 'security', icon: Lock, labelKey: 'settings.sections.security', descriptionKey: 'settings.sections.securityDesc' },
    { id: 'vpn', icon: Shield, labelKey: 'settings.sections.vpn', descriptionKey: 'settings.sections.vpnDesc' },
    { id: 'rpc', icon: Database, labelKey: 'settings.sections.rpc', descriptionKey: 'settings.sections.rpcDesc' },
    { id: 'network', icon: Wifi, labelKey: 'settings.sections.network', descriptionKey: 'settings.sections.networkDesc' },
    { id: 'discovery', icon: Database, labelKey: 'settings.sections.discovery', descriptionKey: 'settings.sections.discoveryDesc' },
    { id: 'providers', icon: Database, labelKey: 'settings.sections.providers', descriptionKey: 'settings.sections.providersDesc' },
    { id: 'maintenance', icon: Database, labelKey: 'settings.sections.maintenance', descriptionKey: 'settings.sections.maintenanceDesc' },
    { id: 'transfer', icon: Upload, labelKey: 'settings.sections.transfer', descriptionKey: 'settings.sections.transferDesc' },
  ]

  return (
    <div className="space-y-4 relative">
      {/* Header */}
      {showHeader ? (
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold tracking-tight">{t('settings.title')}</h2>
          {settings?.updated_at && (
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
              {t('settings.updatedAt')} {new Date(settings.updated_at).toLocaleString()}
            </span>
          )}
        </div>
      ) : null}

      {/* Language switcher */}
      <LanguageSwitcher />

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

      {/* Two-column grid of collapsible sections.
          Headers stay in their original column; expanded panels render full-width below the row. */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {(() => {
          const renderHeader = (section: (typeof sections)[number]) => {
            const isExpanded = expandedSections.has(section.id)
            const Icon = section.icon
            const status = getSectionStatus(section.id)
            const statusColor = getStatusColor(section.id)
            return (
              <div
                key={`header-${section.id}`}
                className={cn(
                  "bg-card/60 border border-border/40 rounded-xl overflow-hidden transition-colors duration-200 self-start",
                  isExpanded && "border-primary/40 ring-1 ring-primary/15 relative z-10"
                )}
              >
                <button
                  type="button"
                  onClick={() => toggleSection(section.id)}
                  className={cn(
                    "w-full flex items-center gap-3 p-3 transition-colors cursor-pointer",
                    isExpanded ? "bg-muted/50" : "hover:bg-muted/40"
                  )}
                >
                  <div className="shrink-0">
                    <Icon className="w-4 h-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1 text-left min-w-0">
                    <div className="text-sm font-medium leading-tight">{t(section.labelKey)}</div>
                    <div className="text-[10px] uppercase tracking-widest text-muted-foreground truncate">{t(section.descriptionKey)}</div>
                  </div>
                  <Badge variant="outline" className={cn("text-[10px] px-2 py-0.5 border-0 shrink-0", statusColor)}>
                    {status}
                  </Badge>
                  <ChevronDown className={cn(
                    "w-4 h-4 text-muted-foreground shrink-0 transition-transform duration-200",
                    isExpanded && "rotate-180"
                  )} />
                </button>
              </div>
            )
          }

          const renderPanel = (section: (typeof sections)[number]) => (
            <div
              key={`panel-${section.id}`}
              className="col-span-full bg-card/60 border border-primary/40 rounded-xl ring-1 ring-primary/15 overflow-hidden animate-in fade-in-0 slide-in-from-top-2 duration-200"
            >
              <div className="p-4">

                  {/* Search Settings */}
                  {section.id === 'search' && (
                    <div className="space-y-4">
                      <Card className="bg-muted border-orange-500/30">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="font-medium text-sm">Polymarket</p>
                            <p className="text-xs text-muted-foreground">
                              Native full-text search via Gamma API
                            </p>
                          </div>
                          <Switch
                            checked={searchForm.search_polymarket_enabled}
                            onCheckedChange={(checked) => setSearchForm(p => ({ ...p, search_polymarket_enabled: checked }))}
                          />
                        </CardContent>
                      </Card>

                      <Card className="bg-muted border-orange-500/30">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="font-medium text-sm">Kalshi</p>
                            <p className="text-xs text-muted-foreground">
                              Client-side filtering (slower, no native search API)
                            </p>
                          </div>
                          <Switch
                            checked={searchForm.search_kalshi_enabled}
                            onCheckedChange={(checked) => setSearchForm(p => ({ ...p, search_kalshi_enabled: checked }))}
                          />
                        </CardContent>
                      </Card>

                      <div>
                        <Label className="text-xs text-muted-foreground">Max Results</Label>
                        <Input
                          type="number"
                          value={searchForm.search_max_results}
                          onChange={(e) => setSearchForm(p => ({ ...p, search_max_results: Math.max(5, Math.min(200, parseInt(e.target.value) || 50)) }))}
                          min={5}
                          max={200}
                          className="h-8 text-xs mt-1 w-32"
                        />
                      </div>

                      <Separator className="opacity-30" />

                      <div className="space-y-3">
                        <div>
                          <p className="font-medium text-sm">Web Search Providers</p>
                          <p className="text-xs text-muted-foreground">
                            Used by AI agents for web search. Falls back in order: SerpAPI → Brave → DuckDuckGo (no key needed).
                          </p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">SerpAPI Key</Label>
                          <div className="relative mt-1">
                            <Input
                              type={showSearchSecrets['serpapi'] ? 'text' : 'password'}
                              value={searchForm.serpapi_key}
                              onChange={(e) => setSearchForm(p => ({ ...p, serpapi_key: e.target.value }))}
                              placeholder={settings?.search_filters?.serpapi_key || 'Not configured'}
                              className="pr-10 font-mono text-sm"
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="absolute right-0 top-0 h-full px-3"
                              onClick={() => setShowSearchSecrets(p => ({ ...p, serpapi: !p.serpapi }))}
                            >
                              {showSearchSecrets['serpapi'] ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            </Button>
                          </div>
                          <p className="text-[11px] text-muted-foreground/70 mt-1">serpapi.com — Google search results</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Brave Search API Key</Label>
                          <div className="relative mt-1">
                            <Input
                              type={showSearchSecrets['brave'] ? 'text' : 'password'}
                              value={searchForm.brave_search_key}
                              onChange={(e) => setSearchForm(p => ({ ...p, brave_search_key: e.target.value }))}
                              placeholder={settings?.search_filters?.brave_search_key || 'Not configured'}
                              className="pr-10 font-mono text-sm"
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="absolute right-0 top-0 h-full px-3"
                              onClick={() => setShowSearchSecrets(p => ({ ...p, brave: !p.brave }))}
                            >
                              {showSearchSecrets['brave'] ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            </Button>
                          </div>
                          <p className="text-[11px] text-muted-foreground/70 mt-1">api.search.brave.com — Brave web search</p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('search')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                      </div>
                    </div>
                  )}


                  {/* Scanner Settings */}
                  {section.id === 'scanner' && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.scanInterval')}</Label>
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
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.minProfit')}</Label>
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
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.minLiquidity')}</Label>
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
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.maxMarkets')}</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_markets_to_scan}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_markets_to_scan: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={200000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.noCapHint')}</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.maxEvents')}</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_events_to_scan}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_events_to_scan: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={200000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.noCapHint')}</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.marketPageSize')}</Label>
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
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.marketOrder')}</Label>
                          <Input
                            type="text"
                            value={scannerForm.market_fetch_order}
                            onChange={(e) => setScannerForm(p => ({ ...p, market_fetch_order: e.target.value }))}
                            placeholder="volume"
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.orderHint')}</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.maxOppsTotal')}</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_opportunities_total}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_opportunities_total: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={50000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.disableHint')}</p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.scanner.maxOppsPerStrategy')}</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_opportunities_per_strategy}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_opportunities_per_strategy: Math.max(0, parseInt(e.target.value) || 0) }))}
                            min={0}
                            max={10000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.disableHint')}</p>
                        </div>
                      </div>
                      <div className="rounded-md border border-border/60 bg-muted/15 p-3 space-y-3">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{t('settings.scanner.runtime')}</p>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div>
                            <Label className="text-xs text-muted-foreground">{t('settings.scanner.skippedCooldown')}</Label>
                            <Input
                              type="number"
                              value={scannerForm.skipped_signal_reactivation_cooldown_seconds}
                              onChange={(e) => setScannerForm((p) => ({
                                ...p,
                                skipped_signal_reactivation_cooldown_seconds: Math.max(0, parseInt(e.target.value) || 0),
                              }))}
                              min={0}
                              max={86400}
                              className="mt-1 text-sm"
                            />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.skippedCooldownDesc')}</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">{t('settings.scanner.wsAge')}</Label>
                            <Input
                              type="number"
                              value={scannerForm.strict_ws_max_age_ms}
                              onChange={(e) => setScannerForm((p) => ({
                                ...p,
                                strict_ws_max_age_ms: Math.max(25, parseInt(e.target.value) || 25),
                              }))}
                              min={25}
                              max={30000}
                              className="mt-1 text-sm"
                            />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.scanner.wsAgeDesc')}</p>
                          </div>
                        </div>
                      </div>
                      <div className="rounded-md border border-border/60 bg-muted/15 p-3 space-y-2">
                        <div className="flex items-start justify-between gap-3">
                          <div className="space-y-1">
                            <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                              Crypto fast-binary lane
                            </p>
                            <p className="text-[11px] text-muted-foreground/80 leading-snug">
                              Disable the crypto fast-binary scanner if you only trade
                              Polymarket general markets. The 4 Binance feeds remain
                              connected; only the per-market payload rebuild stops.
                            </p>
                          </div>
                          <div className="flex items-center gap-2 pt-0.5">
                            <span
                              className={cn(
                                'text-[11px] font-medium tabular-nums',
                                scannerForm.crypto_lane_enabled
                                  ? 'text-emerald-500'
                                  : 'text-muted-foreground'
                              )}
                            >
                              {scannerForm.crypto_lane_enabled ? 'On' : 'Off'}
                            </span>
                            <Switch
                              checked={scannerForm.crypto_lane_enabled}
                              disabled={cryptoLaneToggling}
                              onCheckedChange={(checked) => {
                                void handleCryptoLaneToggle(Boolean(checked))
                              }}
                            />
                          </div>
                        </div>
                        {cryptoLaneError && (
                          <p className="text-[11px] text-destructive">{cryptoLaneError}</p>
                        )}
                      </div>
                      <MarketTagFilterSection
                        selectedTags={scannerForm.market_filter_tags}
                        onChange={(next) =>
                          setScannerForm((p) => ({ ...p, market_filter_tags: next }))
                        }
                      />
                      <Separator className="opacity-30" />
                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('scanner')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.save')}
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
                              <p className="font-medium text-sm">{t('settings.notif.enableTitle')}</p>
                              <p className="text-xs text-muted-foreground">{t('settings.notif.enableDesc')}</p>
                            </div>
                            <Switch
                              checked={notificationsForm.enabled}
                              onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <SecretInput
                          label={t('settings.notif.botToken')}
                          value={notificationsForm.telegram_bot_token}
                          placeholder={settings?.notifications.telegram_bot_token || t('settings.notif.botTokenPlaceholder')}
                          onChange={(v) => setNotificationsForm(p => ({ ...p, telegram_bot_token: v }))}
                          showSecret={showSecrets['tg_token']}
                          onToggle={() => toggleSecret('tg_token')}
                          description={t('settings.notif.botTokenDesc')}
                        />

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.notif.chatId')}</Label>
                          <Input
                            type="text"
                            value={notificationsForm.telegram_chat_id}
                            onChange={(e) => setNotificationsForm(p => ({ ...p, telegram_chat_id: e.target.value }))}
                            placeholder={t('settings.notif.chatIdPlaceholder')}
                            className="mt-1 text-sm"
                          />
                        </div>

                        <div className="space-y-2 pt-2">
                          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">{t('settings.notif.alertTypes')}</p>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">{t('settings.notif.timeline')}</p>
                                <p className="text-xs text-muted-foreground">{t('settings.notif.timelineDesc')}</p>
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
                                <p className="text-sm">{t('settings.notif.issues')}</p>
                                <p className="text-xs text-muted-foreground">{t('settings.notif.issuesDesc')}</p>
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
                                <p className="text-sm">{t('settings.notif.orders')}</p>
                                <p className="text-xs text-muted-foreground">{t('settings.notif.ordersDesc')}</p>
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
                                <p className="text-sm">{t('settings.notif.closes')}</p>
                                <p className="text-xs text-muted-foreground">{t('settings.notif.closesDesc')}</p>
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
                                <p className="text-sm">{t('settings.notif.perTrader')}</p>
                                <p className="text-xs text-muted-foreground">{t('settings.notif.perTraderDesc')}</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_autotrader_summary_per_trader}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_autotrader_summary_per_trader: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <div>
                            <Label className="text-xs text-muted-foreground">{t('settings.notif.summaryInterval')}</Label>
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
                          {t('settings.actions.save')}
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => testTelegramMutation.mutate()}
                          disabled={testTelegramMutation.isPending}
                        >
                          <MessageSquare className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.testTelegram')}
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
                            <p className="font-medium text-sm">{t('settings.uiLock.enableTitle')}</p>
                            <p className="text-xs text-muted-foreground">
                              {t('settings.uiLock.enableDesc')}
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
                          <Label className="text-xs text-muted-foreground">{t('settings.uiLock.idleTimeout')}</Label>
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
                            {t('settings.uiLock.passwordConfigured', { value: uiLockForm.has_password ? t('common.yes') : t('common.no') })}
                          </p>
                        </div>
                      </div>

                      <SecretInput
                        label={t('settings.uiLock.setNewPassword')}
                        value={uiLockForm.password}
                        placeholder={uiLockForm.has_password ? t('settings.uiLock.setNewPlaceholderHas') : t('settings.uiLock.setNewPlaceholder')}
                        onChange={(value) => setUiLockForm((prev) => ({ ...prev, password: value }))}
                        showSecret={showSecrets['ui_lock_password']}
                        onToggle={() => toggleSecret('ui_lock_password')}
                      />

                      <SecretInput
                        label={t('settings.uiLock.confirmPassword')}
                        value={uiLockForm.confirm_password}
                        placeholder={t('settings.uiLock.confirmPlaceholder')}
                        onChange={(value) => setUiLockForm((prev) => ({ ...prev, confirm_password: value }))}
                        showSecret={showSecrets['ui_lock_confirm_password']}
                        onToggle={() => toggleSecret('ui_lock_confirm_password')}
                      />

                      <Card className="bg-muted">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="text-sm">{t('settings.uiLock.clearStored')}</p>
                            <p className="text-xs text-muted-foreground">
                              {t('settings.uiLock.clearStoredDesc')}
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
                          {t('settings.actions.save')}
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
                            <p className="font-medium text-sm">{t('settings.discovery.maintenanceTitle')}</p>
                            <p className="text-xs text-muted-foreground">
                              {t('settings.discovery.maintenanceDesc')}
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
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.maxWallets')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.max_discovered_wallets}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, max_discovered_wallets: parseInt(e.target.value) || 20_000 }))}
                            min={10}
                            max={1_000_000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.maxWalletsDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.batch')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.maintenance_batch}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, maintenance_batch: parseInt(e.target.value) || 900 }))}
                            min={10}
                            max={5000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.batchDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.keepRecent')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.keep_recent_trade_days}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, keep_recent_trade_days: parseInt(e.target.value) || 7 }))}
                            min={1}
                            max={365}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.keepRecentDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.keepNew')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.keep_new_discoveries_days}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, keep_new_discoveries_days: parseInt(e.target.value) || 30 }))}
                            min={1}
                            max={365}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.keepNewDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.staleHours')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.stale_analysis_hours}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, stale_analysis_hours: parseInt(e.target.value) || 12 }))}
                            min={1}
                            max={720}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.staleHoursDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.priorityLimit')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.analysis_priority_batch_limit}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, analysis_priority_batch_limit: parseInt(e.target.value) || 2500 }))}
                            min={100}
                            max={10_000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.priorityLimitDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.delayMarkets')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.delay_between_markets}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, delay_between_markets: parseFloat(e.target.value) || 0 }))}
                            min={0}
                            max={10}
                            step={0.05}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.delayMarketsDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.delayWallets')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.delay_between_wallets}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, delay_between_wallets: parseFloat(e.target.value) || 0 }))}
                            min={0}
                            max={10}
                            step={0.05}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.delayWalletsDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.maxMarketsPerRun')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.max_markets_per_run}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, max_markets_per_run: parseInt(e.target.value) || 100 }))}
                            min={1}
                            max={1_000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.maxMarketsPerRunDesc')}</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">{t('settings.discovery.maxWalletsPerMarket')}</Label>
                          <Input
                            type="number"
                            value={discoveryForm.max_wallets_per_market}
                            onChange={(e) => setDiscoveryForm(p => ({ ...p, max_wallets_per_market: parseInt(e.target.value) || 50 }))}
                            min={1}
                            max={500}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">{t('settings.discovery.maxWalletsPerMarketDesc')}</p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('discovery')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.save')}
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
                              <p className="font-medium text-sm">{t('settings.vpn.enableTitle')}</p>
                              <p className="text-xs text-muted-foreground">{t('settings.vpn.enableDesc')}</p>
                            </div>
                            <Switch
                              checked={vpnForm.enabled}
                              onCheckedChange={(checked) => setVpnForm(p => ({ ...p, enabled: checked }))}
                            />
                          </CardContent>
                        </Card>

                        <SecretInput
                          label={t('settings.vpn.proxyUrl')}
                          value={vpnForm.proxy_url}
                          placeholder={settings?.trading_proxy?.proxy_url || t('settings.vpn.proxyUrlPlaceholder')}
                          onChange={(v) => setVpnForm(p => ({ ...p, proxy_url: v }))}
                          showSecret={showSecrets['proxy_url']}
                          onToggle={() => toggleSecret('proxy_url')}
                          description={t('settings.vpn.proxyUrlDesc')}
                        />

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">{t('settings.vpn.requestTimeout')}</Label>
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
                              <p className="text-sm font-medium">{t('settings.vpn.verifySsl')}</p>
                              <p className="text-xs text-muted-foreground">{t('settings.vpn.verifySslDesc')}</p>
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
                              <p className="text-sm font-medium">{t('settings.vpn.requireVpn')}</p>
                              <p className="text-xs text-muted-foreground">{t('settings.vpn.requireVpnDesc')}</p>
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
                            {t('settings.vpn.infoNote')}
                          </p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2 flex-wrap">
                        <Button size="sm" onClick={() => handleSaveSection('vpn')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.save')}
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => testVpnMutation.mutate()}
                          disabled={testVpnMutation.isPending}
                        >
                          <Shield className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.testVpn')}
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

                  {/* Polygon Blockchain RPC Settings */}
                  {section.id === 'rpc' && (
                    <div className="space-y-4">
                      <Card className="bg-muted border-sky-500/30">
                        <CardContent className="p-3 space-y-1">
                          <p className="font-medium text-sm">Polygon RPC for wallet monitor</p>
                          <p className="text-xs text-muted-foreground">
                            The wallet monitor polls Polygon every block for OrderFilled events.
                            On free public endpoints (publicnode, llamarpc, 1rpc) you'll see
                            <code className="mx-1 px-1 bg-background rounded text-[10px]">RPC failed across all endpoints</code>
                            errors and copy-trade detection drops sub-blocks under load.
                            Configure an authenticated provider (Ankr / Alchemy / QuickNode) here
                            to eliminate the wallet-monitor failure cascade.
                          </p>
                          <p className="text-xs text-muted-foreground">
                            URLs are stored encrypted (Ankr-style URL-path API keys are supported).
                            Empty input leaves the stored value unchanged. Workers pick up new
                            URLs on next restart.
                          </p>
                        </CardContent>
                      </Card>

                      <SecretInput
                        label="Polygon HTTPS RPC URL"
                        value={rpcForm.rpc_url}
                        placeholder={settings?.blockchain_rpc?.rpc_url || 'https://rpc.ankr.com/polygon/<your-api-key>'}
                        onChange={(v) => setRpcForm(p => ({ ...p, rpc_url: v }))}
                        showSecret={showSecrets['polygon_rpc_url']}
                        onToggle={() => toggleSecret('polygon_rpc_url')}
                        description="Used for eth_getLogs polling. Ankr / Alchemy / QuickNode all work."
                      />

                      <SecretInput
                        label="Polygon WebSocket RPC URL (optional)"
                        value={rpcForm.ws_url}
                        placeholder={settings?.blockchain_rpc?.ws_url || 'wss://rpc.ankr.com/polygon/ws/<your-api-key>'}
                        onChange={(v) => setRpcForm(p => ({ ...p, ws_url: v }))}
                        showSecret={showSecrets['polygon_ws_url']}
                        onToggle={() => toggleSecret('polygon_ws_url')}
                        description="Optional. Used to subscribe to new block headers. Falls back to a public WS endpoint if unset."
                      />

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('rpc')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.save')}
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Network Access Settings */}
                  {section.id === 'network' && (
                    <div className="space-y-4">
                      <Card className="bg-muted border-sky-500/30">
                        <CardContent className="flex items-center justify-between p-3">
                          <div>
                            <p className="font-medium text-sm">{t('settings.net.allowTitle')}</p>
                            <p className="text-xs text-muted-foreground">
                              {t('settings.net.allowDesc')}
                            </p>
                          </div>
                          <Switch
                            checked={networkForm.allow_network_access}
                            onCheckedChange={(checked) => setNetworkForm(p => ({ ...p, allow_network_access: checked }))}
                          />
                        </CardContent>
                      </Card>

                      <div className="flex items-start gap-2 p-3 bg-sky-500/5 border border-sky-500/20 rounded-lg">
                        <Wifi className="w-4 h-4 text-sky-400 mt-0.5 shrink-0" />
                        <p className="text-xs text-muted-foreground">
                          When enabled, the dashboard binds to <span className="font-mono">0.0.0.0</span> instead of <span className="font-mono">localhost</span>,
                          making it reachable from any device on your LAN (e.g. phone, tablet, another PC).
                          <span className="text-yellow-400 font-medium"> Requires a restart to take effect.</span>
                        </p>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('network')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          {t('settings.actions.save')}
                        </Button>
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
                                    <p className="text-sm font-medium">{t(category.labelKey)}</p>
                                    <p className="text-xs text-muted-foreground">{t(category.descriptionKey)}</p>
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
                  {section.id === 'providers' && (
                    <div className="space-y-4">
                      <Card className="bg-muted">
                        <CardContent className="p-3 space-y-3">
                          <div>
                            <p className="font-medium text-sm">Polybacktest</p>
                            <p className="text-xs text-muted-foreground">
                              Sub-second Polymarket Up/Down book history + Binance reference prices.
                              Buy a Pro tier at <a href="https://polybacktest.com/dashboard" target="_blank" rel="noreferrer" className="underline">polybacktest.com</a>.
                            </p>
                          </div>
                          <div className="space-y-2">
                            <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">API key</Label>
                            <div className="flex items-center gap-1">
                              <Input
                                type={showProviderKey ? 'text' : 'password'}
                                value={providerForm.polybacktest_api_key}
                                onChange={(e) => setProviderForm((p) => ({ ...p, polybacktest_api_key: e.target.value }))}
                                placeholder={
                                  providerSettingsQuery.data?.polybacktest_api_key_set
                                    ? '(set — leave to keep)'
                                    : 'Paste API key'
                                }
                                className="h-9 font-mono text-xs"
                              />
                              <Button
                                type="button"
                                size="sm"
                                variant="ghost"
                                className="h-9 w-9 p-0"
                                onClick={() => setShowProviderKey((v) => !v)}
                                title={showProviderKey ? 'Hide' : 'Show'}
                              >
                                {showProviderKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                              </Button>
                            </div>
                            <p className="text-[10px] text-muted-foreground">
                              Empty value clears the key. Mask is shown when a key is stored — leave it untouched to keep.
                            </p>
                          </div>
                          <div className="space-y-2">
                            <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Base URL (optional)</Label>
                            <Input
                              value={providerForm.polybacktest_base_url}
                              onChange={(e) => setProviderForm((p) => ({ ...p, polybacktest_base_url: e.target.value }))}
                              placeholder="https://api.polybacktest.com"
                              className="h-9 font-mono text-xs"
                            />
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="bg-muted">
                        <CardContent className="p-3 space-y-3">
                          <div>
                            <p className="font-medium text-sm">Telonex</p>
                            <p className="text-xs text-muted-foreground">
                              Daily Parquet files for Polymarket (trades, quotes, L2 book snapshots,
                              on-chain fills) + Binance reference prices. Get a key at{' '}
                              <a href="https://telonex.io" target="_blank" rel="noreferrer" className="underline">telonex.io</a>.
                              Free tier = 5 total downloads.
                            </p>
                          </div>
                          <div className="space-y-2">
                            <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">API key</Label>
                            <div className="flex items-center gap-1">
                              <Input
                                type={showTelonexKey ? 'text' : 'password'}
                                value={providerForm.telonex_api_key}
                                onChange={(e) => setProviderForm((p) => ({ ...p, telonex_api_key: e.target.value }))}
                                placeholder={
                                  providerSettingsQuery.data?.telonex_api_key_set
                                    ? '(set — leave to keep)'
                                    : 'Paste API key'
                                }
                                className="h-9 font-mono text-xs"
                              />
                              <Button
                                type="button"
                                size="sm"
                                variant="ghost"
                                className="h-9 w-9 p-0"
                                onClick={() => setShowTelonexKey((v) => !v)}
                                title={showTelonexKey ? 'Hide' : 'Show'}
                              >
                                {showTelonexKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                              </Button>
                            </div>
                            <p className="text-[10px] text-muted-foreground">
                              Empty value clears the key. Mask is shown when a key is stored — leave it untouched to keep.
                            </p>
                          </div>
                          <div className="space-y-2">
                            <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Base URL (optional)</Label>
                            <Input
                              value={providerForm.telonex_base_url}
                              onChange={(e) => setProviderForm((p) => ({ ...p, telonex_base_url: e.target.value }))}
                              placeholder="https://api.telonex.io/v1"
                              className="h-9 font-mono text-xs"
                            />
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="bg-muted">
                        <CardContent className="p-3 space-y-3">
                          <div>
                            <p className="font-medium text-sm">Reverse-engineer defaults</p>
                            <p className="text-xs text-muted-foreground">
                              Operator-tunable defaults for new strategy reverse-engineer jobs.
                              Empty fields fall back to the AI default model + service guards (no hidden defaults in code).
                            </p>
                          </div>
                          <p className="text-[10px] text-muted-foreground">
                            Default LLM model is set in <strong>AI → Models</strong> (under
                            "Strategy Reverse-Engineer" — same place every other per-purpose model lives).
                          </p>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <div className="space-y-2">
                              <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Max iterations</Label>
                              <Input
                                value={providerForm.reverse_engineer_max_iterations}
                                onChange={(e) => setProviderForm((p) => ({ ...p, reverse_engineer_max_iterations: e.target.value }))}
                                placeholder="10"
                                className="h-9 text-xs"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Target score (0-1)</Label>
                              <Input
                                value={providerForm.reverse_engineer_target_score}
                                onChange={(e) => setProviderForm((p) => ({ ...p, reverse_engineer_target_score: e.target.value }))}
                                placeholder="0.7"
                                className="h-9 text-xs"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Max cost (USD)</Label>
                              <Input
                                value={providerForm.reverse_engineer_max_cost_usd}
                                onChange={(e) => setProviderForm((p) => ({ ...p, reverse_engineer_max_cost_usd: e.target.value }))}
                                placeholder="(no cap)"
                                className="h-9 text-xs"
                              />
                            </div>
                            <div className="space-y-2 md:col-span-2">
                              <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Max wallet trades pulled</Label>
                              <Input
                                value={providerForm.reverse_engineer_max_wallet_trades}
                                onChange={(e) => setProviderForm((p) => ({ ...p, reverse_engineer_max_wallet_trades: e.target.value }))}
                                placeholder="50000"
                                className="h-9 text-xs"
                              />
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <div className="flex justify-end">
                        <Button
                          type="button"
                          size="sm"
                          onClick={() => saveProviderSettingsMutation.mutate()}
                          disabled={saveProviderSettingsMutation.isPending}
                          className="gap-1.5"
                        >
                          {saveProviderSettingsMutation.isPending ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Save className="h-3.5 w-3.5" />
                          )}
                          Save provider settings
                        </Button>
                      </div>
                    </div>
                  )}

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
                                {maintenanceStatsQuery.isLoading ? (
                                  <div className="h-5 w-20 rounded bg-muted-foreground/20 animate-pulse mt-0.5" />
                                ) : (
                                  <p className="text-sm font-semibold">
                                    {formatDbBytes(maintenanceStatsQuery.data?.db_size_bytes)}
                                  </p>
                                )}
                              </div>
                              <div className="rounded-lg border border-border/50 bg-card/40 px-3 py-2">
                                <p className="text-[11px] uppercase tracking-widest text-muted-foreground">Total Rows</p>
                                {maintenanceStatsQuery.isLoading ? (
                                  <div className="h-5 w-24 rounded bg-muted-foreground/20 animate-pulse mt-0.5" />
                                ) : (
                                  <p className="text-sm font-semibold">
                                    {maintenanceStatsQuery.data?.total_rows != null
                                      ? maintenanceStatsQuery.data.total_rows.toLocaleString()
                                      : 'Unavailable'}
                                  </p>
                                )}
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

                        {/* Dead Tuple / Bloat Stats */}
                        {maintenanceStatsQuery.data?.table_bloat && maintenanceStatsQuery.data.table_bloat.length > 0 && (() => {
                          const bloat = maintenanceStatsQuery.data!.table_bloat!
                          const totalDead = bloat.reduce((s, t) => s + t.dead_tuples, 0)
                          const totalLive = bloat.reduce((s, t) => s + t.live_tuples, 0)
                          const totalIndexBytes = bloat.reduce((s, t) => s + (t.index_bytes ?? 0), 0)
                          const totalTableBytes = bloat.reduce((s, t) => s + (t.table_bytes ?? 0), 0)
                          const bloatedTables = bloat.filter(t => t.dead_tuples > 1000)
                          const indexHeavyTables = bloat.filter(t => t.index_bytes > 0 && t.table_bytes > 0 && t.index_bytes > t.table_bytes * 5)
                          const fmtBytes = (b: number) => b >= 1073741824 ? `${(b / 1073741824).toFixed(1)} GB` : b >= 1048576 ? `${(b / 1048576).toFixed(0)} MB` : `${(b / 1024).toFixed(0)} KB`
                          const anyMutationPending = vacuumMutation.isPending || reindexMutation.isPending
                          return (
                            <Card className={cn('bg-muted', totalDead > 100000 && 'border-amber-500/30')}>
                              <CardContent className="p-3 space-y-2">
                                <div className="flex items-center justify-between">
                                  <div>
                                    <p className="font-medium text-sm">Dead Tuples & Index Bloat</p>
                                    <p className="text-xs text-muted-foreground">
                                      {totalDead.toLocaleString()} dead / {totalLive.toLocaleString()} live
                                      {totalLive > 0 && ` (${(100 * totalDead / totalLive).toFixed(1)}%)`}
                                      {totalIndexBytes > 0 && totalTableBytes > 0 && (
                                        <span className={cn('ml-2', totalIndexBytes > totalTableBytes * 3 && 'text-amber-400')}>
                                          Idx: {fmtBytes(totalIndexBytes)} / Data: {fmtBytes(totalTableBytes)}
                                        </span>
                                      )}
                                    </p>
                                  </div>
                                  <div className="flex items-center gap-1">
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      className="h-7 px-2 text-xs"
                                      onClick={() => vacuumMutation.mutate(false)}
                                      disabled={anyMutationPending}
                                      title="Reclaim dead tuples (non-blocking)"
                                    >
                                      {vacuumMutation.isPending && !vacuumMutation.variables
                                        ? <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                        : <RefreshCw className="w-3 h-3 mr-1" />}
                                      Vacuum
                                    </Button>
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      className="h-7 px-2 text-xs border-amber-500/40 text-amber-400 hover:bg-amber-500/10"
                                      onClick={() => vacuumMutation.mutate(true)}
                                      disabled={anyMutationPending}
                                      title="Rewrites tables to reclaim disk. Takes exclusive lock — avoid during active trading."
                                    >
                                      {vacuumMutation.isPending && vacuumMutation.variables
                                        ? <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                        : <RefreshCw className="w-3 h-3 mr-1" />}
                                      Full
                                    </Button>
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      className="h-7 px-2 text-xs"
                                      onClick={() => reindexMutation.mutate()}
                                      disabled={anyMutationPending}
                                      title="Rebuild bloated indexes to reclaim disk space"
                                    >
                                      {reindexMutation.isPending
                                        ? <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                        : <Database className="w-3 h-3 mr-1" />}
                                      Reindex
                                    </Button>
                                  </div>
                                </div>

                                {bloatedTables.length > 0 && (
                                  <div className="max-h-36 overflow-y-auto text-[11px] space-y-0.5">
                                    {bloatedTables.map(t => {
                                      const indexBloated = t.index_bytes > 0 && t.table_bytes > 0 && t.index_bytes > t.table_bytes * 5
                                      return (
                                        <div key={t.table} className="flex justify-between items-center py-0.5 border-b border-border/20 last:border-0">
                                          <span className="font-mono text-muted-foreground truncate mr-2">{t.table}</span>
                                          <div className="flex items-center gap-2">
                                            {indexBloated && (
                                              <span className="whitespace-nowrap tabular-nums text-amber-400" title={`Index: ${fmtBytes(t.index_bytes)} vs Data: ${fmtBytes(t.table_bytes)}`}>
                                                idx {fmtBytes(t.index_bytes)}
                                              </span>
                                            )}
                                            <span className={cn(
                                              'whitespace-nowrap tabular-nums',
                                              t.dead_pct > 50 ? 'text-red-400' : t.dead_pct > 10 ? 'text-amber-400' : 'text-muted-foreground',
                                            )}>
                                              {t.dead_tuples.toLocaleString()} dead ({t.dead_pct}%)
                                            </span>
                                          </div>
                                        </div>
                                      )
                                    })}
                                  </div>
                                )}

                                {indexHeavyTables.length > 0 && bloatedTables.length === 0 && (
                                  <div className="max-h-36 overflow-y-auto text-[11px] space-y-0.5">
                                    <p className="text-[10px] text-amber-400/80 mb-1">Tables with bloated indexes (index &gt; 5x data):</p>
                                    {indexHeavyTables.map(t => (
                                      <div key={t.table} className="flex justify-between items-center py-0.5 border-b border-border/20 last:border-0">
                                        <span className="font-mono text-muted-foreground truncate mr-2">{t.table}</span>
                                        <span className="whitespace-nowrap tabular-nums text-amber-400">
                                          idx {fmtBytes(t.index_bytes)} / data {fmtBytes(t.table_bytes)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </CardContent>
                            </Card>
                          )
                        })()}

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

                          <div>
                            <Label className="text-xs text-muted-foreground">Trader Events Firehose Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.trader_events_firehose_retention_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, trader_events_firehose_retention_days: Number.isNaN(value) ? 7 : value }))
                              }}
                              min={1}
                              max={365}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Trader Events Other Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.trader_events_other_retention_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, trader_events_other_retention_days: Number.isNaN(value) ? 90 : value }))
                              }}
                              min={1}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Wallet Monitor Events Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_wallet_monitor_events_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, cleanup_wallet_monitor_events_days: Number.isNaN(value) ? 14 : value }))
                              }}
                              min={0}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Decision Checks Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_trader_decision_checks_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, cleanup_trader_decision_checks_days: Number.isNaN(value) ? 14 : value }))
                              }}
                              min={0}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Decisions Retention (days, unreferenced only)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_trader_decisions_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, cleanup_trader_decisions_days: Number.isNaN(value) ? 30 : value }))
                              }}
                              min={0}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Opportunity History Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_opportunity_history_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, cleanup_opportunity_history_days: Number.isNaN(value) ? 30 : value }))
                              }}
                              min={0}
                              max={3650}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Signal Emission Retention (days)</Label>
                            <Input
                              type="number"
                              value={maintenanceForm.cleanup_trade_signal_emission_days}
                              onChange={(e) => {
                                const value = Number.parseInt(e.target.value, 10)
                                setMaintenanceForm(p => ({ ...p, cleanup_trade_signal_emission_days: Number.isNaN(value) ? p.cleanup_trade_signal_emission_days : value }))
                              }}
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
          )

          const nodes: ReactNode[] = []
          for (let i = 0; i < sections.length; i += 2) {
            const left = sections[i]
            const right = sections[i + 1]
            nodes.push(renderHeader(left))
            if (right) nodes.push(renderHeader(right))
            const leftExpanded = expandedSections.has(left.id)
            const rightExpanded = right ? expandedSections.has(right.id) : false
            if (leftExpanded) nodes.push(renderPanel(left))
            if (rightExpanded && right) nodes.push(renderPanel(right))
          }
          return nodes
        })()}
      </div>
    </div>
  )
}

interface MarketTagFilterSectionProps {
  selectedTags: string[]
  onChange: (next: string[]) => void
}

function MarketTagFilterSection({ selectedTags, onChange }: MarketTagFilterSectionProps) {
  const [draft, setDraft] = useState('')

  const availableTagsQuery = useQuery({
    queryKey: ['settings', 'market-filter', 'available-tags'],
    queryFn: getMarketFilterAvailableTags,
    staleTime: 5 * 60 * 1000,
  })

  const normalised = (value: string): string => value.trim().toLowerCase()

  const addTag = (raw: string) => {
    const cleaned = normalised(raw)
    if (!cleaned) return
    if (selectedTags.includes(cleaned)) return
    onChange([...selectedTags, cleaned])
  }

  const removeTag = (tag: string) => {
    onChange(selectedTags.filter((t) => t !== tag))
  }

  const handleSubmitDraft = () => {
    if (!draft.trim()) return
    addTag(draft)
    setDraft('')
  }

  const suggestions: MarketFilterAvailableTag[] = (
    availableTagsQuery.data?.tags ?? []
  ).filter((entry) => !selectedTags.includes(entry.name))

  const datalistId = 'market-filter-tag-suggestions'

  return (
    <div className="rounded-md border border-border/60 bg-muted/15 p-3 space-y-3">
      <div className="flex items-center gap-2">
        <Tag className="w-3.5 h-3.5 text-muted-foreground" />
        <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Market Tag Filter
        </p>
      </div>
      <p className="text-[11px] text-muted-foreground/80 leading-relaxed">
        Limit which markets the scanner ingests. Markets must carry at least one
        matching tag (OR-logic, case-insensitive). Empty list = no filter applied —
        the scanner ingests every Polymarket / Kalshi market as today.
      </p>
      <div className="flex flex-wrap items-center gap-1.5">
        {selectedTags.length === 0 ? (
          <span className="text-[11px] italic text-muted-foreground/70">
            No tags selected — filter inactive.
          </span>
        ) : (
          selectedTags.map((tag) => (
            <Badge
              key={tag}
              variant="secondary"
              className="pl-2 pr-1 gap-1 text-[11px] font-normal"
            >
              {tag}
              <button
                type="button"
                onClick={() => removeTag(tag)}
                className="rounded-full hover:bg-muted-foreground/20 p-0.5 transition-colors"
                aria-label={`Remove ${tag}`}
              >
                <X className="w-3 h-3" />
              </button>
            </Badge>
          ))
        )}
      </div>
      <div className="flex items-center gap-2">
        <Input
          type="text"
          list={datalistId}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              handleSubmitDraft()
            }
          }}
          placeholder="Type or pick a tag, press Enter"
          className="text-sm"
        />
        <datalist id={datalistId}>
          {suggestions.map((entry) => (
            <option key={entry.name} value={entry.name}>
              {`${entry.name} - seen ${entry.occurrences}x in last 24h`}
            </option>
          ))}
        </datalist>
        <Button
          size="sm"
          variant="secondary"
          onClick={handleSubmitDraft}
          disabled={!draft.trim()}
        >
          Add
        </Button>
      </div>
      <div className="text-[10px] text-muted-foreground/70">
        {availableTagsQuery.isLoading ? (
          <span className="inline-flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            Loading tag suggestions...
          </span>
        ) : availableTagsQuery.isError ? (
          <span className="text-destructive/80">
            Failed to load tag suggestions - try again later.
          </span>
        ) : (availableTagsQuery.data?.total ?? 0) === 0 ? (
          <span>
            No tags ingested yet - the list is populated from live Polymarket
            traffic and refreshes within one ingest cycle.
          </span>
        ) : (
          <span>
            {availableTagsQuery.data?.total ?? 0} distinct tag
            {(availableTagsQuery.data?.total ?? 0) === 1 ? '' : 's'} seen in the
            last 24 h.
          </span>
        )}
      </div>
    </div>
  )
}
