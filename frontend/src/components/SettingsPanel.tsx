import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Bot,
  Bell,
  Scan,
  TrendingUp,
  Database,
  RefreshCw,
  Save,
  CheckCircle,
  AlertCircle,
  Eye,
  EyeOff,
  MessageSquare,
  DollarSign,
  Shield,
  ChevronDown,
  Puzzle,
  Plus,
  Trash2,
  Code,
  Play,
  Loader2,
  FileCode,
  CircleDot,
  XCircle,
  RotateCcw,
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
  testTradingProxy,
  flushDatabaseData,
  getLLMModels,
  refreshLLMModels,
  getPlugins,
  createPlugin,
  updatePlugin,
  deletePlugin,
  validatePlugin,
  getPluginTemplate,
  reloadPlugin,
  getPluginDocs,
  runWorkerOnce,
  type DatabaseFlushTarget,
  type LLMModelOption,
} from '../services/api'

type SettingsSection = 'llm' | 'notifications' | 'scanner' | 'trading' | 'vpn' | 'plugins' | 'maintenance'

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

export default function SettingsPanel() {
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
    notify_min_roi: 5.0
  })

  const [scannerForm, setScannerForm] = useState({
    scan_interval_seconds: 60,
    min_profit_threshold: 2.5,
    max_markets_to_scan: 500,
    min_liquidity: 1000
  })

  const [tradingForm, setTradingForm] = useState({
    trading_enabled: false,
    max_trade_size_usd: 100,
    max_daily_trade_volume: 1000,
    max_open_positions: 10,
    max_slippage_percent: 2.0
  })

  const [maintenanceForm, setMaintenanceForm] = useState({
    auto_cleanup_enabled: false,
    cleanup_interval_hours: 24,
    cleanup_resolved_trade_days: 30,
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

  const [pluginForm, setPluginForm] = useState<{
    slug: string
    source_code: string
    config: Record<string, unknown>
  } | null>(null)
  const [editingPluginId, setEditingPluginId] = useState<string | null>(null)
  const [pluginValidation, setPluginValidation] = useState<{
    valid: boolean
    errors: string[]
    warnings: string[]
    class_name: string | null
    strategy_name: string | null
  } | null>(null)
  const [validating, setValidating] = useState(false)
  const [viewingPluginId, setViewingPluginId] = useState<string | null>(null)
  const [showPluginDocs, setShowPluginDocs] = useState(false)

  const queryClient = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const { data: pluginDocs } = useQuery<Record<string, any>>({
    queryKey: ['plugin-docs'],
    queryFn: getPluginDocs,
    staleTime: Infinity, // docs don't change at runtime
  })

  const { data: plugins = [] } = useQuery({
    queryKey: ['plugins'],
    queryFn: getPlugins,
  })


  // Expand plugins when navigating from Search Filters flyout
  useEffect(() => {
    const handler = (e: CustomEvent<SettingsSection>) => {
      if (e.detail === 'plugins') {
        setExpandedSections((prev) => new Set(prev).add('plugins'))
      }
    }
    window.addEventListener('navigate-settings-section', handler as EventListener)
    return () => window.removeEventListener('navigate-settings-section', handler as EventListener)
  }, [])

  // Sync form state with loaded settings
  useEffect(() => {
    if (settings) {
      setLlmForm({
        provider: settings.llm.provider || 'none',
        openai_api_key: '',
        anthropic_api_key: '',
        google_api_key: '',
        xai_api_key: '',
        deepseek_api_key: '',
        ollama_api_key: '',
        ollama_base_url: settings.llm.ollama_base_url || '',
        lmstudio_api_key: '',
        lmstudio_base_url: settings.llm.lmstudio_base_url || '',
        model: settings.llm.model || '',
        max_monthly_spend: settings.llm.max_monthly_spend ?? 50.0
      })

      setNotificationsForm({
        enabled: settings.notifications.enabled,
        telegram_bot_token: '',
        telegram_chat_id: settings.notifications.telegram_chat_id || '',
        notify_on_opportunity: settings.notifications.notify_on_opportunity,
        notify_on_trade: settings.notifications.notify_on_trade,
        notify_min_roi: settings.notifications.notify_min_roi
      })

      setScannerForm({
        scan_interval_seconds: settings.scanner.scan_interval_seconds,
        min_profit_threshold: settings.scanner.min_profit_threshold,
        max_markets_to_scan: settings.scanner.max_markets_to_scan,
        min_liquidity: settings.scanner.min_liquidity
      })

      setTradingForm({
        trading_enabled: settings.trading.trading_enabled,
        max_trade_size_usd: settings.trading.max_trade_size_usd,
        max_daily_trade_volume: settings.trading.max_daily_trade_volume,
        max_open_positions: settings.trading.max_open_positions,
        max_slippage_percent: settings.trading.max_slippage_percent
      })

      setMaintenanceForm({
        auto_cleanup_enabled: settings.maintenance.auto_cleanup_enabled,
        cleanup_interval_hours: settings.maintenance.cleanup_interval_hours,
        cleanup_resolved_trade_days: settings.maintenance.cleanup_resolved_trade_days,
        market_cache_hygiene_enabled: settings.maintenance.market_cache_hygiene_enabled,
        market_cache_hygiene_interval_hours: settings.maintenance.market_cache_hygiene_interval_hours,
        market_cache_retention_days: settings.maintenance.market_cache_retention_days,
        market_cache_reference_lookback_days: settings.maintenance.market_cache_reference_lookback_days,
        market_cache_weak_entry_grace_days: settings.maintenance.market_cache_weak_entry_grace_days,
        market_cache_max_entries_per_slug: settings.maintenance.market_cache_max_entries_per_slug,
      })

      if (settings.trading_proxy) {
        setVpnForm({
          enabled: settings.trading_proxy.enabled,
          proxy_url: '',  // Don't pre-fill masked URL
          verify_ssl: settings.trading_proxy.verify_ssl,
          timeout: settings.trading_proxy.timeout,
          require_vpn: settings.trading_proxy.require_vpn
        })
      }

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

  const runAutoTraderOnceMutation = useMutation({
    mutationFn: () => runWorkerOnce('autotrader'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['autotrader-overview'] })
      queryClient.invalidateQueries({ queryKey: ['autotrader-status'] })
      setSaveMessage({ type: 'success', text: 'Auto-trader one-time run queued' })
      setTimeout(() => setSaveMessage(null), 4000)
    },
    onError: (error: any) => {
      const detail = error?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || error?.message || 'Failed to queue auto-trader run' })
      setTimeout(() => setSaveMessage(null), 7000)
    },
  })

  const createPluginMutation = useMutation({
    mutationFn: createPlugin,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      setPluginForm(null)
      setPluginValidation(null)
      setSaveMessage({ type: 'success', text: 'Plugin created and loaded' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (err: any) => {
      const detail = err?.response?.data?.detail
      const msg = typeof detail === 'object' ? detail.errors?.join('; ') || detail.message : detail || 'Failed to create plugin'
      setSaveMessage({ type: 'error', text: msg })
      setTimeout(() => setSaveMessage(null), 8000)
    },
  })

  const updatePluginMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Parameters<typeof updatePlugin>[1] }) =>
      updatePlugin(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      setPluginForm(null)
      setEditingPluginId(null)
      setPluginValidation(null)
    },
    onError: (err: any) => {
      const detail = err?.response?.data?.detail
      const msg = typeof detail === 'object' ? detail.errors?.join('; ') || detail.message : detail || 'Failed to update plugin'
      setSaveMessage({ type: 'error', text: msg })
      setTimeout(() => setSaveMessage(null), 8000)
    },
  })

  const deletePluginMutation = useMutation({
    mutationFn: deletePlugin,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      setPluginForm(null)
      setEditingPluginId(null)
      setViewingPluginId(null)
    },
    onError: (err: any) => {
      setSaveMessage({ type: 'error', text: err?.response?.data?.detail || 'Failed to delete plugin' })
      setTimeout(() => setSaveMessage(null), 5000)
    },
  })

  const reloadPluginMutation = useMutation({
    mutationFn: reloadPlugin,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      setSaveMessage({ type: 'success', text: 'Plugin reloaded' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (err: any) => {
      const detail = err?.response?.data?.detail
      const msg = typeof detail === 'object' ? detail.error || detail.message : detail || 'Failed to reload plugin'
      setSaveMessage({ type: 'error', text: msg })
      setTimeout(() => setSaveMessage(null), 8000)
    },
  })

  const handleValidatePlugin = async (code: string) => {
    setValidating(true)
    try {
      const result = await validatePlugin(code)
      setPluginValidation(result)
    } catch {
      setPluginValidation({ valid: false, errors: ['Validation request failed'], warnings: [], class_name: null, strategy_name: null })
    } finally {
      setValidating(false)
    }
  }

  const handleLoadTemplate = async () => {
    try {
      const { template } = await getPluginTemplate()
      setPluginForm(prev => prev ? { ...prev, source_code: template } : null)
      setPluginValidation(null)
    } catch {
      setSaveMessage({ type: 'error', text: 'Failed to load template' })
      setTimeout(() => setSaveMessage(null), 3000)
    }
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
          telegram_chat_id: notificationsForm.telegram_chat_id || null
        }
        if (notificationsForm.telegram_bot_token) {
          updates.notifications.telegram_bot_token = notificationsForm.telegram_bot_token
        }
        break
      case 'scanner':
        updates.scanner = scannerForm
        break
      case 'trading':
        updates.trading = tradingForm
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
      autotrader: 'Auto-trader runtime data',
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
      case 'scanner':
        return `${scannerForm.scan_interval_seconds}s interval`
      case 'trading':
        return tradingForm.trading_enabled ? 'Live' : 'Disabled'
      case 'vpn':
        return vpnForm.enabled ? 'Active' : 'Disabled'
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'Auto-clean on' : 'Manual'
      case 'plugins':
        return `${plugins.length} plugin${plugins.length !== 1 ? 's' : ''}`
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
      case 'scanner':
        return 'text-cyan-400 bg-cyan-500/10'
      case 'trading':
        return tradingForm.trading_enabled ? 'text-yellow-400 bg-yellow-500/10' : 'text-muted-foreground bg-muted'
      case 'vpn':
        return vpnForm.enabled ? 'text-indigo-400 bg-indigo-500/10' : 'text-muted-foreground bg-muted'
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'text-red-400 bg-red-500/10' : 'text-muted-foreground bg-muted'
      case 'plugins':
        return plugins.length > 0 ? 'text-violet-400 bg-violet-500/10' : 'text-muted-foreground bg-muted'
      default:
        return 'text-muted-foreground bg-muted'
    }
  }

  const sections: { id: SettingsSection; icon: any; label: string; description: string }[] = [
    { id: 'llm', icon: Bot, label: 'AI / LLM Services', description: 'Configure AI providers' },
    { id: 'notifications', icon: Bell, label: 'Notifications', description: 'Telegram alerts' },
    { id: 'scanner', icon: Scan, label: 'Scanner', description: 'Market scanning settings' },
    { id: 'trading', icon: TrendingUp, label: 'Trading Safety', description: 'Trading limits & safety' },
    { id: 'vpn', icon: Shield, label: 'Trading VPN/Proxy', description: 'Route trades through VPN' },
    { id: 'plugins', icon: Puzzle, label: 'Strategy Plugins', description: 'Custom strategy code' },
    { id: 'maintenance', icon: Database, label: 'Database', description: 'Cleanup & maintenance' },
  ]

  return (
    <div className="space-y-4 relative">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold tracking-tight">Settings</h2>
        {settings?.updated_at && (
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Updated {new Date(settings.updated_at).toLocaleString()}
          </span>
        )}
      </div>

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
                                <p className="text-sm">New Opportunities</p>
                                <p className="text-xs text-muted-foreground">Alert when new opportunities are found</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_on_opportunity}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_on_opportunity: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <Card className="bg-muted">
                            <CardContent className="flex items-center justify-between p-3">
                              <div>
                                <p className="text-sm">Trade Executions</p>
                                <p className="text-xs text-muted-foreground">Alert when trades are executed</p>
                              </div>
                              <Switch
                                checked={notificationsForm.notify_on_trade}
                                onCheckedChange={(checked) => setNotificationsForm(p => ({ ...p, notify_on_trade: checked }))}
                              />
                            </CardContent>
                          </Card>

                          <div>
                            <Label className="text-xs text-muted-foreground">Minimum ROI for Alerts (%)</Label>
                            <Input
                              type="number"
                              value={notificationsForm.notify_min_roi}
                              onChange={(e) => setNotificationsForm(p => ({ ...p, notify_min_roi: parseFloat(e.target.value) || 0 }))}
                              step="0.5"
                              min="0"
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

                  {/* Scanner Settings */}
                  {section.id === 'scanner' && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">Scan Interval (seconds)</Label>
                          <Input
                            type="number"
                            value={scannerForm.scan_interval_seconds}
                            onChange={(e) => setScannerForm(p => ({ ...p, scan_interval_seconds: parseInt(e.target.value) || 60 }))}
                            min={10}
                            max={3600}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">How often to scan for opportunities</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Min Profit Threshold (%)</Label>
                          <Input
                            type="number"
                            value={scannerForm.min_profit_threshold}
                            onChange={(e) => setScannerForm(p => ({ ...p, min_profit_threshold: parseFloat(e.target.value) || 0 }))}
                            step="0.5"
                            min={0}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Minimum ROI to report as opportunity</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Max Markets to Scan</Label>
                          <Input
                            type="number"
                            value={scannerForm.max_markets_to_scan}
                            onChange={(e) => setScannerForm(p => ({ ...p, max_markets_to_scan: parseInt(e.target.value) || 100 }))}
                            min={10}
                            max={5000}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Limit on markets per scan</p>
                        </div>

                        <div>
                          <Label className="text-xs text-muted-foreground">Min Liquidity ($)</Label>
                          <Input
                            type="number"
                            value={scannerForm.min_liquidity}
                            onChange={(e) => setScannerForm(p => ({ ...p, min_liquidity: parseFloat(e.target.value) || 0 }))}
                            min={0}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Minimum market liquidity</p>
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

                  {/* Trading Settings */}
                  {section.id === 'trading' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <Card className="bg-muted border-yellow-500/30">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Enable Live Trading</p>
                              <p className="text-xs text-muted-foreground">Allow real money trades on Polymarket</p>
                            </div>
                            <Switch
                              checked={tradingForm.trading_enabled}
                              onCheckedChange={(checked) => {
                                if (!tradingForm.trading_enabled || confirm('Are you sure you want to enable live trading?')) {
                                  setTradingForm(p => ({ ...p, trading_enabled: checked }))
                                }
                              }}
                            />
                          </CardContent>
                        </Card>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Trade Size ($)</Label>
                            <Input
                              type="number"
                              value={tradingForm.max_trade_size_usd}
                              onChange={(e) => setTradingForm(p => ({ ...p, max_trade_size_usd: parseFloat(e.target.value) || 0 }))}
                              min={1}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Max Daily Volume ($)</Label>
                            <Input
                              type="number"
                              value={tradingForm.max_daily_trade_volume}
                              onChange={(e) => setTradingForm(p => ({ ...p, max_daily_trade_volume: parseFloat(e.target.value) || 0 }))}
                              min={10}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Max Open Positions</Label>
                            <Input
                              type="number"
                              value={tradingForm.max_open_positions}
                              onChange={(e) => setTradingForm(p => ({ ...p, max_open_positions: parseInt(e.target.value) || 1 }))}
                              min={1}
                              max={100}
                              className="mt-1 text-sm"
                            />
                          </div>

                          <div>
                            <Label className="text-xs text-muted-foreground">Max Slippage (%)</Label>
                            <Input
                              type="number"
                              value={tradingForm.max_slippage_percent}
                              onChange={(e) => setTradingForm(p => ({ ...p, max_slippage_percent: parseFloat(e.target.value) || 0 }))}
                              step="0.1"
                              min={0.1}
                              max={10}
                              className="mt-1 text-sm"
                            />
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('trading')} disabled={saveMutation.isPending}>
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

                  {/* Plugins Settings */}
                  {section.id === 'plugins' && (
                    <div className="space-y-4">
                      <p className="text-xs text-muted-foreground">
                        Write custom strategy plugins — full Python strategy files with their own detection logic. Plugins run alongside built-in strategies during each scan cycle.
                      </p>

                      {/* API Reference Toggle */}
                      <div>
                        <button
                          onClick={() => setShowPluginDocs(!showPluginDocs)}
                          className="flex items-center gap-1.5 text-[11px] text-violet-400 hover:text-violet-300 transition-colors"
                        >
                          <FileCode className="w-3.5 h-3.5" />
                          {showPluginDocs ? 'Hide' : 'Show'} Plugin API Reference
                          <ChevronDown className={cn("w-3 h-3 transition-transform", showPluginDocs && "rotate-180")} />
                        </button>

                        {showPluginDocs && pluginDocs && (
                          <div className="mt-3 space-y-3">
                            {/* Overview */}
                            <Card className="bg-black/30 border-violet-500/10">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-violet-400">How Plugins Work</p>
                                <p className="text-[10px] text-muted-foreground leading-relaxed">
                                  {pluginDocs.overview?.description}
                                </p>
                              </CardContent>
                            </Card>

                            {/* Class Structure */}
                            <Card className="bg-black/30 border-border/20">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-foreground">Class Structure</p>
                                <p className="text-[10px] text-muted-foreground mb-1">{pluginDocs.class_structure?.description}</p>
                                <div className="space-y-1.5">
                                  <p className="text-[10px] font-medium text-emerald-400/80">Required attributes:</p>
                                  {pluginDocs.class_structure?.required_attributes && Object.entries(pluginDocs.class_structure.required_attributes).map(([key, desc]) => (
                                    <div key={key} className="flex gap-2 text-[10px]">
                                      <code className="text-violet-400 font-mono shrink-0">{key}</code>
                                      <span className="text-muted-foreground">{desc as string}</span>
                                    </div>
                                  ))}
                                  <p className="text-[10px] font-medium text-yellow-400/80 mt-2">Available on self:</p>
                                  {pluginDocs.class_structure?.inherited_attributes && Object.entries(pluginDocs.class_structure.inherited_attributes).map(([key, desc]) => (
                                    <div key={key} className="flex gap-2 text-[10px]">
                                      <code className="text-violet-400 font-mono shrink-0">{key}</code>
                                      <span className="text-muted-foreground">{desc as string}</span>
                                    </div>
                                  ))}
                                </div>
                              </CardContent>
                            </Card>

                            {/* detect() Method */}
                            <Card className="bg-black/30 border-border/20">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-foreground">detect() — Your Main Method</p>
                                <pre className="text-[9px] font-mono text-violet-300 bg-black/40 p-2 rounded overflow-x-auto">
                                  {pluginDocs.detect_method?.signature}
                                </pre>
                                <p className="text-[10px] text-muted-foreground">{pluginDocs.detect_method?.description}</p>

                                {/* Parameters */}
                                {pluginDocs.detect_method?.parameters && Object.entries(pluginDocs.detect_method.parameters).map(([param, info]: [string, any]) => (
                                  <div key={param} className="mt-2">
                                    <div className="flex items-center gap-1.5">
                                      <code className="text-[10px] font-mono text-emerald-400">{param}</code>
                                      <span className="text-[9px] text-muted-foreground/60">{info.type}</span>
                                    </div>
                                    <p className="text-[10px] text-muted-foreground ml-2">{info.description}</p>
                                    {info.structure && (
                                      <pre className="text-[9px] font-mono text-muted-foreground/80 bg-black/30 p-1.5 rounded ml-2 mt-1">{info.structure}</pre>
                                    )}
                                    {info.usage && (
                                      <p className="text-[9px] text-muted-foreground/60 ml-2 mt-0.5 italic">{info.usage}</p>
                                    )}
                                    {info.fields && (
                                      <div className="ml-2 mt-1 space-y-0.5">
                                        {Object.entries(info.fields).map(([field, desc]) => (
                                          <div key={field} className="flex gap-1.5 text-[9px]">
                                            <code className="text-violet-400/70 font-mono shrink-0">.{field}</code>
                                            <span className="text-muted-foreground/70">{desc as string}</span>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </CardContent>
                            </Card>

                            {/* create_opportunity() */}
                            <Card className="bg-black/30 border-border/20">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-foreground">self.create_opportunity() — Build Opportunities</p>
                                <p className="text-[10px] text-muted-foreground">{pluginDocs.create_opportunity_method?.description}</p>
                                <div className="space-y-1 mt-1">
                                  {pluginDocs.create_opportunity_method?.parameters && Object.entries(pluginDocs.create_opportunity_method.parameters).map(([key, desc]) => (
                                    <div key={key} className="flex gap-2 text-[10px]">
                                      <code className="text-violet-400 font-mono shrink-0">{key}</code>
                                      <span className="text-muted-foreground">{desc as string}</span>
                                    </div>
                                  ))}
                                </div>
                                <div className="mt-2">
                                  <p className="text-[10px] font-medium text-yellow-400/80">Hard filters (auto-applied):</p>
                                  <ul className="mt-1 space-y-0.5">
                                    {pluginDocs.create_opportunity_method?.hard_filters_applied?.map((f: string, i: number) => (
                                      <li key={i} className="text-[9px] text-muted-foreground/70 flex gap-1">
                                        <span className="text-muted-foreground/40">•</span> {f}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              </CardContent>
                            </Card>

                            {/* Config System */}
                            <Card className="bg-black/30 border-border/20">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-foreground">Config System</p>
                                <p className="text-[10px] text-muted-foreground">{pluginDocs.config_system?.description}</p>
                                <pre className="text-[9px] font-mono text-muted-foreground/80 bg-black/40 p-2 rounded overflow-x-auto whitespace-pre">{pluginDocs.config_system?.example}</pre>
                              </CardContent>
                            </Card>

                            {/* Common Patterns */}
                            <Card className="bg-black/30 border-border/20">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-foreground">Common Code Patterns</p>
                                {pluginDocs.common_patterns && Object.entries(pluginDocs.common_patterns).map(([key, code]) => (
                                  <div key={key} className="mt-1">
                                    <p className="text-[10px] text-emerald-400/80 font-medium">{key.replace(/_/g, ' ')}</p>
                                    <pre className="text-[9px] font-mono text-muted-foreground/80 bg-black/40 p-2 rounded overflow-x-auto whitespace-pre mt-0.5">{code as string}</pre>
                                  </div>
                                ))}
                              </CardContent>
                            </Card>

                            {/* Allowed & Blocked Imports */}
                            <Card className="bg-black/30 border-border/20">
                              <CardContent className="p-3 space-y-2">
                                <p className="text-[11px] font-medium text-foreground">Allowed Imports</p>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                                  {pluginDocs.allowed_imports?.map((imp: any, i: number) => (
                                    <div key={i} className="flex gap-1.5 text-[9px]">
                                      <code className="text-violet-400 font-mono shrink-0">{imp.module}</code>
                                      <span className="text-muted-foreground/60 truncate">{imp.items}</span>
                                    </div>
                                  ))}
                                </div>
                                <p className="text-[10px] font-medium text-red-400/80 mt-2">Blocked (security):</p>
                                <ul className="space-y-0.5">
                                  {pluginDocs.blocked_imports?.map((b: string, i: number) => (
                                    <li key={i} className="text-[9px] text-muted-foreground/60 flex gap-1">
                                      <span className="text-red-400/40">✕</span> {b}
                                    </li>
                                  ))}
                                </ul>
                              </CardContent>
                            </Card>
                          </div>
                        )}
                      </div>

                      {/* Plugin Editor (create or edit) */}
                      {pluginForm ? (
                        <Card className="bg-muted/50 border-violet-500/20">
                          <CardContent className="p-4 space-y-3">
                            {!editingPluginId && (
                              <div>
                                <Label className="text-xs text-muted-foreground">Plugin Slug</Label>
                                <Input
                                  value={pluginForm.slug}
                                  onChange={(e) => setPluginForm(p => p ? { ...p, slug: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_') } : null)}
                                  placeholder="e.g. whale_follower"
                                  className="mt-1 text-sm font-mono"
                                />
                                <p className="text-[10px] text-muted-foreground/60 mt-0.5">Unique identifier: lowercase letters, numbers, underscores</p>
                              </div>
                            )}
                            <div>
                              <div className="flex items-center justify-between mb-1">
                                <Label className="text-xs text-muted-foreground">Strategy Code</Label>
                                <div className="flex gap-1.5">
                                  {!editingPluginId && (
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="h-6 px-2 text-[10px] text-violet-400 hover:text-violet-300"
                                      onClick={handleLoadTemplate}
                                    >
                                      <FileCode className="w-3 h-3 mr-1" />
                                      Load Template
                                    </Button>
                                  )}
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-6 px-2 text-[10px]"
                                    onClick={() => handleValidatePlugin(pluginForm.source_code)}
                                    disabled={validating || !pluginForm.source_code.trim()}
                                  >
                                    {validating ? (
                                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                    ) : (
                                      <Play className="w-3 h-3 mr-1" />
                                    )}
                                    Validate
                                  </Button>
                                </div>
                              </div>
                              <textarea
                                value={pluginForm.source_code}
                                onChange={(e) => {
                                  setPluginForm(p => p ? { ...p, source_code: e.target.value } : null)
                                  setPluginValidation(null) // Clear validation on edit
                                }}
                                placeholder="Paste your strategy code here, or click Load Template..."
                                className="w-full h-80 bg-black/40 border border-border/40 rounded-lg p-3 text-xs font-mono text-foreground placeholder:text-muted-foreground/40 resize-y focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                                spellCheck={false}
                              />
                              {/* Validation Results */}
                              {pluginValidation && (
                                <div className={cn(
                                  "mt-2 p-2.5 rounded-lg border text-xs",
                                  pluginValidation.valid
                                    ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-400"
                                    : "bg-red-500/5 border-red-500/20 text-red-400"
                                )}>
                                  {pluginValidation.valid ? (
                                    <div className="space-y-1">
                                      <div className="flex items-center gap-1.5 font-medium">
                                        <CheckCircle className="w-3.5 h-3.5" />
                                        Valid — class "{pluginValidation.class_name}"
                                        {pluginValidation.strategy_name && ` (${pluginValidation.strategy_name})`}
                                      </div>
                                      {pluginValidation.warnings.length > 0 && (
                                        <div className="text-yellow-400/80 mt-1">
                                          {pluginValidation.warnings.map((w, i) => (
                                            <p key={i} className="text-[10px]">Warning: {w}</p>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  ) : (
                                    <div className="space-y-1">
                                      <div className="flex items-center gap-1.5 font-medium">
                                        <XCircle className="w-3.5 h-3.5" />
                                        Validation Failed
                                      </div>
                                      {pluginValidation.errors.map((e, i) => (
                                        <p key={i} className="text-[10px] mt-0.5">{e}</p>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                            <div className="flex gap-2 pt-2">
                              <Button
                                size="sm"
                                onClick={() => {
                                  if (editingPluginId) {
                                    updatePluginMutation.mutate({
                                      id: editingPluginId,
                                      updates: {
                                        source_code: pluginForm.source_code,
                                        config: pluginForm.config,
                                      },
                                    })
                                  } else {
                                    createPluginMutation.mutate({
                                      slug: pluginForm.slug,
                                      source_code: pluginForm.source_code,
                                      config: pluginForm.config,
                                      enabled: true,
                                    })
                                  }
                                }}
                                disabled={
                                  (!editingPluginId && !pluginForm.slug.trim()) ||
                                  !pluginForm.source_code.trim() ||
                                  createPluginMutation.isPending ||
                                  updatePluginMutation.isPending
                                }
                              >
                                {(createPluginMutation.isPending || updatePluginMutation.isPending) ? (
                                  <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                ) : (
                                  <Save className="w-3.5 h-3.5 mr-1.5" />
                                )}
                                {editingPluginId ? 'Save & Reload' : 'Create Plugin'}
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => {
                                  setPluginForm(null)
                                  setEditingPluginId(null)
                                  setPluginValidation(null)
                                }}
                              >
                                Cancel
                              </Button>
                            </div>
                          </CardContent>
                        </Card>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setPluginForm({ slug: '', source_code: '', config: {} })}
                          className="gap-1.5"
                        >
                          <Plus className="w-3.5 h-3.5" />
                          New Plugin
                        </Button>
                      )}

                      {/* Plugin List */}
                      {plugins.length > 0 && (
                        <div className="space-y-2">
                          <Label className="text-xs text-muted-foreground">Your Plugins</Label>
                          {plugins.map((plugin) => (
                            <div key={plugin.id} className="space-y-0">
                              <div
                                className="flex items-center justify-between gap-3 p-3 rounded-lg bg-muted/30 border border-border/40 cursor-pointer hover:bg-muted/50 transition-colors"
                                onClick={() => setViewingPluginId(viewingPluginId === plugin.id ? null : plugin.id)}
                              >
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-2">
                                    <p className="font-medium text-sm truncate">{plugin.name}</p>
                                    <span className="text-[10px] font-mono text-muted-foreground/60">{plugin.slug}</span>
                                    {/* Status indicator */}
                                    {plugin.status === 'loaded' && (
                                      <Badge className="text-[9px] bg-emerald-500/15 text-emerald-400 border-emerald-500/30 px-1.5">
                                        <CircleDot className="w-2.5 h-2.5 mr-0.5" />
                                        Loaded
                                      </Badge>
                                    )}
                                    {plugin.status === 'error' && (
                                      <Badge className="text-[9px] bg-red-500/15 text-red-400 border-red-500/30 px-1.5">
                                        <XCircle className="w-2.5 h-2.5 mr-0.5" />
                                        Error
                                      </Badge>
                                    )}
                                    {plugin.status === 'unloaded' && (
                                      <Badge variant="outline" className="text-[9px] text-muted-foreground border-muted px-1.5">
                                        Unloaded
                                      </Badge>
                                    )}
                                    {!plugin.enabled && (
                                      <Badge variant="outline" className="text-[9px] text-muted-foreground border-muted px-1.5">
                                        Disabled
                                      </Badge>
                                    )}
                                  </div>
                                  {plugin.description && (
                                    <p className="text-[11px] text-muted-foreground truncate">{plugin.description}</p>
                                  )}
                                  {/* Runtime stats */}
                                  {plugin.runtime && (
                                    <p className="text-[10px] text-muted-foreground/70 mt-0.5">
                                      v{plugin.version} · {plugin.runtime.run_count} runs · {plugin.runtime.total_opportunities} opportunities found
                                      {plugin.runtime.error_count > 0 && (
                                        <span className="text-red-400/70"> · {plugin.runtime.error_count} errors</span>
                                      )}
                                    </p>
                                  )}
                                  {!plugin.runtime && (
                                    <p className="text-[10px] text-muted-foreground/50 mt-0.5">
                                      v{plugin.version} · {plugin.class_name || 'unknown class'}
                                    </p>
                                  )}
                                </div>
                                <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                                  <Switch
                                    checked={plugin.enabled}
                                    onCheckedChange={(checked) => {
                                      updatePluginMutation.mutate({
                                        id: plugin.id,
                                        updates: { enabled: checked },
                                      })
                                    }}
                                    className="scale-90"
                                  />
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7"
                                    title="Reload plugin"
                                    onClick={() => reloadPluginMutation.mutate(plugin.id)}
                                    disabled={!plugin.enabled || reloadPluginMutation.isPending}
                                  >
                                    <RotateCcw className="w-3.5 h-3.5" />
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7"
                                    title="Edit code"
                                    onClick={() => {
                                      setEditingPluginId(plugin.id)
                                      setPluginForm({
                                        slug: plugin.slug,
                                        source_code: plugin.source_code,
                                        config: plugin.config || {},
                                      })
                                      setPluginValidation(null)
                                    }}
                                  >
                                    <Code className="w-3.5 h-3.5" />
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                                    onClick={() => {
                                      if (window.confirm(`Delete plugin "${plugin.name}"? This will remove its code and unload it.`)) {
                                        deletePluginMutation.mutate(plugin.id)
                                      }
                                    }}
                                    disabled={deletePluginMutation.isPending}
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                  </Button>
                                </div>
                              </div>
                              {/* Expanded detail view */}
                              {viewingPluginId === plugin.id && (
                                <div className="ml-3 mr-3 mb-2 p-3 bg-black/30 border border-border/20 rounded-b-lg space-y-2">
                                  {plugin.error_message && (
                                    <div className="p-2 bg-red-500/5 border border-red-500/20 rounded text-[11px] text-red-400">
                                      <p className="font-medium mb-0.5">Load Error:</p>
                                      <pre className="whitespace-pre-wrap font-mono text-[10px] text-red-400/80">{plugin.error_message}</pre>
                                    </div>
                                  )}
                                  <div>
                                    <p className="text-[10px] text-muted-foreground/60 mb-1">Source code preview:</p>
                                    <pre className="text-[10px] font-mono text-muted-foreground bg-black/40 p-2 rounded border border-border/20 max-h-32 overflow-auto whitespace-pre-wrap">
                                      {plugin.source_code.slice(0, 600)}{plugin.source_code.length > 600 ? '\n...' : ''}
                                    </pre>
                                  </div>
                                  {plugin.runtime?.last_error && (
                                    <div className="text-[10px] text-yellow-400/70">
                                      Last runtime error: {plugin.runtime.last_error}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Maintenance Settings */}
                  {section.id === 'maintenance' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
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

                        <div className="grid grid-cols-2 gap-3">
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
                        </div>

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
                                Manually clear runtime/cache datasets for scanner, weather, news, and auto-trader pipelines.
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
                                onClick={() => handleFlushTarget('autotrader')}
                                disabled={flushDataMutation.isPending}
                              >
                                {flushDataMutation.isPending && activeFlushTarget === 'autotrader'
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Trash2 className="w-3.5 h-3.5 mr-1.5" />}
                                Flush Auto-Trader Runtime
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
                                onClick={() => runAutoTraderOnceMutation.mutate()}
                                disabled={runAutoTraderOnceMutation.isPending}
                              >
                                {runAutoTraderOnceMutation.isPending
                                  ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                  : <Play className="w-3.5 h-3.5 mr-1.5" />}
                                Run Auto-Trader Once
                              </Button>

                              <p className="text-[10px] text-muted-foreground">
                                Auto-trader run is non-destructive and queues one immediate cycle.
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
