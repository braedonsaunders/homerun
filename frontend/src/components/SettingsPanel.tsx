import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Key,
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
  Zap,
  MessageSquare,
  Activity,
  DollarSign,
  Brain,
  Shield,
  BarChart3,
  ChevronDown,
  SlidersHorizontal,
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
  testPolymarketConnection,
  testTelegramConnection,
  testTradingProxy,
  getLLMModels,
  refreshLLMModels,
  getAutoTraderStatus,
  updateAutoTraderConfig,
  type LLMModelOption,
  type AutoTraderConfig,
} from '../services/api'

type SettingsSection = 'polymarket' | 'kalshi' | 'llm' | 'notifications' | 'scanner' | 'trading' | 'vpn' | 'autotrader' | 'maintenance' | 'search_filters'

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

const ALL_STRATEGIES = [
  { key: 'basic', label: 'Basic Arb' },
  { key: 'negrisk', label: 'NegRisk' },
  { key: 'mutually_exclusive', label: 'Mutually Exclusive' },
  { key: 'contradiction', label: 'Contradiction' },
  { key: 'must_happen', label: 'Must-Happen' },
  { key: 'cross_platform', label: 'Cross-Platform Oracle' },
  { key: 'bayesian_cascade', label: 'Bayesian Cascade' },
  { key: 'liquidity_vacuum', label: 'Liquidity Vacuum' },
  { key: 'entropy_arb', label: 'Entropy Arbitrage' },
  { key: 'event_driven', label: 'Event-Driven' },
  { key: 'temporal_decay', label: 'Temporal Decay' },
  { key: 'correlation_arb', label: 'Correlation Arb' },
  { key: 'market_making', label: 'Market Making' },
  { key: 'stat_arb', label: 'Statistical Arb' },
]

export default function SettingsPanel() {
  const [expandedSections, setExpandedSections] = useState<Set<SettingsSection>>(new Set())
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Form state for each section
  const [polymarketForm, setPolymarketForm] = useState({
    api_key: '',
    api_secret: '',
    api_passphrase: '',
    private_key: ''
  })

  const [kalshiForm, setKalshiForm] = useState({
    email: '',
    password: '',
    api_key: ''
  })

  const [llmForm, setLlmForm] = useState({
    provider: 'none',
    openai_api_key: '',
    anthropic_api_key: '',
    google_api_key: '',
    xai_api_key: '',
    deepseek_api_key: '',
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
    cleanup_resolved_trade_days: 30
  })

  const [vpnForm, setVpnForm] = useState({
    enabled: false,
    proxy_url: '',
    verify_ssl: true,
    timeout: 30,
    require_vpn: true
  })

  const [searchFiltersForm, setSearchFiltersForm] = useState({
    min_liquidity_hard: 200,
    min_position_size: 25,
    min_absolute_profit: 5,
    min_annualized_roi: 10,
    max_resolution_months: 18,
    max_plausible_roi: 30,
    max_trade_legs: 8,
    negrisk_min_total_yes: 0.95,
    negrisk_warn_total_yes: 0.97,
    negrisk_election_min_total_yes: 0.97,
    negrisk_max_resolution_spread_days: 7,
    settlement_lag_max_days_to_resolution: 14,
    settlement_lag_near_zero: 0.05,
    settlement_lag_near_one: 0.95,
    settlement_lag_min_sum_deviation: 0.03,
    risk_very_short_days: 2,
    risk_short_days: 7,
    risk_long_lockup_days: 180,
    risk_extended_lockup_days: 90,
    risk_low_liquidity: 1000,
    risk_moderate_liquidity: 5000,
    risk_complex_legs: 5,
    risk_multiple_legs: 3,
    btc_eth_pure_arb_max_combined: 0.98,
    btc_eth_dump_hedge_drop_pct: 0.05,
    btc_eth_thin_liquidity_usd: 500,
    miracle_min_no_price: 0.90,
    miracle_max_no_price: 0.995,
    miracle_min_impossibility_score: 0.70,
  })

  const [autotraderAiForm, setAutotraderAiForm] = useState({
    llm_verify_trades: false,
    llm_verify_strategies: '',
    auto_ai_scoring: false,
    enabled_strategies: [] as string[],
  })
  const [autotraderAiDirty, setAutotraderAiDirty] = useState(false)

  const queryClient = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const { data: autoTraderStatus } = useQuery({
    queryKey: ['auto-trader-status'],
    queryFn: getAutoTraderStatus,
    refetchInterval: 10000,
  })

  const autoTraderConfigMutation = useMutation({
    mutationFn: (updates: Partial<AutoTraderConfig>) => updateAutoTraderConfig(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      setAutotraderAiDirty(false)
      setSaveMessage({ type: 'success', text: 'Auto-Trader AI settings saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save auto-trader settings' })
      setTimeout(() => setSaveMessage(null), 5000)
    }
  })

  // Sync auto-trader AI form from server config
  useEffect(() => {
    if (autoTraderStatus?.config && !autotraderAiDirty) {
      const cfg = autoTraderStatus.config
      setAutotraderAiForm({
        llm_verify_trades: cfg.llm_verify_trades ?? false,
        llm_verify_strategies: (cfg.llm_verify_strategies ?? []).join(', '),
        auto_ai_scoring: cfg.auto_ai_scoring ?? false,
        enabled_strategies: cfg.enabled_strategies ?? [],
      })
    }
  }, [autoTraderStatus?.config, autotraderAiDirty])

  // Sync form state with loaded settings
  useEffect(() => {
    if (settings) {
      // Only update if form hasn't been modified (checking for empty values)
      setPolymarketForm(prev => ({
        api_key: prev.api_key || '',
        api_secret: prev.api_secret || '',
        api_passphrase: prev.api_passphrase || '',
        private_key: prev.private_key || ''
      }))

      setLlmForm({
        provider: settings.llm.provider || 'none',
        openai_api_key: '',
        anthropic_api_key: '',
        google_api_key: '',
        xai_api_key: '',
        deepseek_api_key: '',
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
        cleanup_resolved_trade_days: settings.maintenance.cleanup_resolved_trade_days
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

      if (settings.search_filters) {
        setSearchFiltersForm(settings.search_filters)
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
      const res = await refreshLLMModels()
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


  const testPolymarketMutation = useMutation({
    mutationFn: testPolymarketConnection,
  })

  const testTelegramMutation = useMutation({
    mutationFn: testTelegramConnection,
  })

  const testVpnMutation = useMutation({
    mutationFn: testTradingProxy,
  })

  const handleSaveSection = (section: SettingsSection) => {
    const updates: any = {}

    switch (section) {
      case 'polymarket':
        // Only include non-empty values
        updates.polymarket = {}
        if (polymarketForm.api_key) updates.polymarket.api_key = polymarketForm.api_key
        if (polymarketForm.api_secret) updates.polymarket.api_secret = polymarketForm.api_secret
        if (polymarketForm.api_passphrase) updates.polymarket.api_passphrase = polymarketForm.api_passphrase
        if (polymarketForm.private_key) updates.polymarket.private_key = polymarketForm.private_key
        break
      case 'kalshi':
        updates.kalshi = {}
        if (kalshiForm.email) updates.kalshi.email = kalshiForm.email
        if (kalshiForm.password) updates.kalshi.password = kalshiForm.password
        if (kalshiForm.api_key) updates.kalshi.api_key = kalshiForm.api_key
        break
      case 'llm':
        updates.llm = {
          provider: llmForm.provider,
          model: llmForm.model || null,
          max_monthly_spend: llmForm.max_monthly_spend
        }
        if (llmForm.openai_api_key) updates.llm.openai_api_key = llmForm.openai_api_key
        if (llmForm.anthropic_api_key) updates.llm.anthropic_api_key = llmForm.anthropic_api_key
        if (llmForm.google_api_key) updates.llm.google_api_key = llmForm.google_api_key
        if (llmForm.xai_api_key) updates.llm.xai_api_key = llmForm.xai_api_key
        if (llmForm.deepseek_api_key) updates.llm.deepseek_api_key = llmForm.deepseek_api_key
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
      case 'search_filters':
        updates.search_filters = searchFiltersForm
        break
      case 'autotrader': {
        const strategies = autotraderAiForm.llm_verify_strategies
          .split(',')
          .map(s => s.trim())
          .filter(s => s.length > 0)
        autoTraderConfigMutation.mutate({
          llm_verify_trades: autotraderAiForm.llm_verify_trades,
          llm_verify_strategies: strategies,
          auto_ai_scoring: autotraderAiForm.auto_ai_scoring,
          enabled_strategies: autotraderAiForm.enabled_strategies,
        })
        return
      }
    }

    saveMutation.mutate(updates)
  }

  const toggleSecret = (key: string) => {
    setShowSecrets(prev => ({ ...prev, [key]: !prev[key] }))
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
      case 'polymarket': {
        const keysSet = [
          settings?.polymarket?.api_key,
          settings?.polymarket?.api_secret,
          settings?.polymarket?.api_passphrase,
          settings?.polymarket?.private_key,
        ].filter(Boolean).length
        return keysSet > 0 ? `${keysSet} key${keysSet !== 1 ? 's' : ''} set` : 'Not configured'
      }
      case 'kalshi': {
        const hasEmail = !!settings?.kalshi?.email
        const hasKey = !!settings?.kalshi?.api_key
        if (hasEmail || hasKey) return 'Configured'
        return 'Not configured'
      }
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
      case 'autotrader': {
        const count = autotraderAiForm.enabled_strategies.length
        return count > 0 ? `${count} strateg${count !== 1 ? 'ies' : 'y'}` : 'No strategies'
      }
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'Auto-clean on' : 'Manual'
      case 'search_filters':
        return `${searchFiltersForm.max_trade_legs} legs, $${searchFiltersForm.min_liquidity_hard} min liq`
      default:
        return ''
    }
  }

  const getStatusColor = (id: SettingsSection): string => {
    switch (id) {
      case 'polymarket': {
        const keysSet = [
          settings?.polymarket?.api_key,
          settings?.polymarket?.api_secret,
          settings?.polymarket?.api_passphrase,
          settings?.polymarket?.private_key,
        ].filter(Boolean).length
        return keysSet >= 3 ? 'text-emerald-400 bg-emerald-500/10' : keysSet > 0 ? 'text-yellow-400 bg-yellow-500/10' : 'text-muted-foreground bg-muted'
      }
      case 'kalshi': {
        const configured = !!settings?.kalshi?.email || !!settings?.kalshi?.api_key
        return configured ? 'text-emerald-400 bg-emerald-500/10' : 'text-muted-foreground bg-muted'
      }
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
      case 'autotrader':
        return autotraderAiForm.enabled_strategies.length > 0 ? 'text-emerald-400 bg-emerald-500/10' : 'text-muted-foreground bg-muted'
      case 'maintenance':
        return maintenanceForm.auto_cleanup_enabled ? 'text-red-400 bg-red-500/10' : 'text-muted-foreground bg-muted'
      case 'search_filters':
        return 'text-orange-400 bg-orange-500/10'
      default:
        return 'text-muted-foreground bg-muted'
    }
  }

  const sections: { id: SettingsSection; icon: any; label: string; description: string }[] = [
    { id: 'polymarket', icon: Key, label: 'Polymarket Account', description: 'API credentials for trading' },
    { id: 'kalshi', icon: BarChart3, label: 'Kalshi Account', description: 'Kalshi exchange credentials' },
    { id: 'llm', icon: Bot, label: 'AI / LLM Services', description: 'Configure AI providers' },
    { id: 'notifications', icon: Bell, label: 'Notifications', description: 'Telegram alerts' },
    { id: 'scanner', icon: Scan, label: 'Scanner', description: 'Market scanning settings' },
    { id: 'search_filters', icon: SlidersHorizontal, label: 'Search Filters', description: 'Opportunity detection thresholds' },
    { id: 'trading', icon: TrendingUp, label: 'Trading Safety', description: 'Trading limits & safety' },
    { id: 'vpn', icon: Shield, label: 'Trading VPN/Proxy', description: 'Route trades through VPN' },
    { id: 'autotrader', icon: Brain, label: 'Auto Trader AI', description: 'LLM verification & scoring' },
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

                  {/* Polymarket Settings */}
                  {section.id === 'polymarket' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <SecretInput
                          label="API Key"
                          value={polymarketForm.api_key}
                          placeholder={settings?.polymarket.api_key || 'Enter API key'}
                          onChange={(v) => setPolymarketForm(p => ({ ...p, api_key: v }))}
                          showSecret={showSecrets['pm_key']}
                          onToggle={() => toggleSecret('pm_key')}
                        />

                        <SecretInput
                          label="API Secret"
                          value={polymarketForm.api_secret}
                          placeholder={settings?.polymarket.api_secret || 'Enter API secret'}
                          onChange={(v) => setPolymarketForm(p => ({ ...p, api_secret: v }))}
                          showSecret={showSecrets['pm_secret']}
                          onToggle={() => toggleSecret('pm_secret')}
                        />

                        <SecretInput
                          label="API Passphrase"
                          value={polymarketForm.api_passphrase}
                          placeholder={settings?.polymarket.api_passphrase || 'Enter API passphrase'}
                          onChange={(v) => setPolymarketForm(p => ({ ...p, api_passphrase: v }))}
                          showSecret={showSecrets['pm_pass']}
                          onToggle={() => toggleSecret('pm_pass')}
                        />

                        <SecretInput
                          label="Private Key"
                          value={polymarketForm.private_key}
                          placeholder={settings?.polymarket.private_key || 'Enter wallet private key'}
                          onChange={(v) => setPolymarketForm(p => ({ ...p, private_key: v }))}
                          showSecret={showSecrets['pm_pk']}
                          onToggle={() => toggleSecret('pm_pk')}
                          description="Your wallet private key for signing transactions"
                        />
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2 flex-wrap">
                        <Button size="sm" onClick={() => handleSaveSection('polymarket')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => testPolymarketMutation.mutate()}
                          disabled={testPolymarketMutation.isPending}
                        >
                          <Zap className="w-3.5 h-3.5 mr-1.5" />
                          Test Connection
                        </Button>
                        {testPolymarketMutation.data && (
                          <Badge variant={testPolymarketMutation.data.status === 'success' ? "default" : "outline"} className={cn(
                            "text-xs",
                            testPolymarketMutation.data.status === 'success' ? "bg-green-500/10 text-green-400" : "bg-yellow-500/10 text-yellow-400"
                          )}>
                            {testPolymarketMutation.data.message}
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Kalshi Settings */}
                  {section.id === 'kalshi' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        <div>
                          <Label className="text-xs text-muted-foreground">Kalshi Email</Label>
                          <Input
                            type="email"
                            value={kalshiForm.email}
                            onChange={(e) => setKalshiForm(p => ({ ...p, email: e.target.value }))}
                            placeholder={settings?.kalshi?.email || 'Enter Kalshi account email'}
                            className="mt-1 text-sm"
                          />
                          <p className="text-[11px] text-muted-foreground/70 mt-1">Your Kalshi account email address</p>
                        </div>

                        <SecretInput
                          label="Kalshi Password"
                          value={kalshiForm.password}
                          placeholder={settings?.kalshi?.password || 'Enter Kalshi password'}
                          onChange={(v) => setKalshiForm(p => ({ ...p, password: v }))}
                          showSecret={showSecrets['kalshi_pass']}
                          onToggle={() => toggleSecret('kalshi_pass')}
                          description="Used for email/password authentication to Kalshi API"
                        />

                        <Separator className="opacity-30" />

                        <SecretInput
                          label="API Key (Alternative)"
                          value={kalshiForm.api_key}
                          placeholder={settings?.kalshi?.api_key || 'Enter Kalshi API key'}
                          onChange={(v) => setKalshiForm(p => ({ ...p, api_key: v }))}
                          showSecret={showSecrets['kalshi_key']}
                          onToggle={() => toggleSecret('kalshi_key')}
                          description="Alternative to email/password. If set, API key is preferred for authentication."
                        />
                      </div>

                      <div className="flex items-start gap-2 p-3 bg-indigo-500/5 border border-indigo-500/20 rounded-lg">
                        <Activity className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
                        <p className="text-xs text-muted-foreground">
                          Kalshi credentials enable cross-platform arbitrage trading. The scanner will automatically detect price differences between Polymarket and Kalshi for the same events. You can use either email/password or an API key for authentication.
                        </p>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('kalshi')} disabled={saveMutation.isPending}>
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          Save
                        </Button>
                      </div>
                    </div>
                  )}

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
                                <p className="text-xs text-muted-foreground">Alert when new arbitrage opportunities are found</p>
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

                  {/* Search Filter Settings */}
                  {section.id === 'search_filters' && (
                    <div className="space-y-4">
                      {/* Hard Rejection Filters */}
                      <div>
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Hard Rejection Filters</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Liquidity Hard ($)</Label>
                            <Input type="number" value={searchFiltersForm.min_liquidity_hard} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_liquidity_hard: parseFloat(e.target.value) || 0 }))} min={0} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Hard reject below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Position Size ($)</Label>
                            <Input type="number" value={searchFiltersForm.min_position_size} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_position_size: parseFloat(e.target.value) || 0 }))} min={0} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if max position below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Absolute Profit ($)</Label>
                            <Input type="number" value={searchFiltersForm.min_absolute_profit} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_absolute_profit: parseFloat(e.target.value) || 0 }))} min={0} step="0.5" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if net profit below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Annualized ROI (%)</Label>
                            <Input type="number" value={searchFiltersForm.min_annualized_roi} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_annualized_roi: parseFloat(e.target.value) || 0 }))} min={0} step="1" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if annualized ROI below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Resolution (months)</Label>
                            <Input type="number" value={searchFiltersForm.max_resolution_months} onChange={(e) => setSearchFiltersForm(p => ({ ...p, max_resolution_months: parseInt(e.target.value) || 18 }))} min={1} max={120} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if too far out</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Plausible ROI (%)</Label>
                            <Input type="number" value={searchFiltersForm.max_plausible_roi} onChange={(e) => setSearchFiltersForm(p => ({ ...p, max_plausible_roi: parseFloat(e.target.value) || 30 }))} min={1} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Above this = false positive</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Trade Legs</Label>
                            <Input type="number" value={searchFiltersForm.max_trade_legs} onChange={(e) => setSearchFiltersForm(p => ({ ...p, max_trade_legs: parseInt(e.target.value) || 8 }))} min={2} max={20} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Max legs in multi-leg trade</p>
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      {/* NegRisk Exhaustivity */}
                      <div>
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">NegRisk Exhaustivity</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Total YES</Label>
                            <Input type="number" value={searchFiltersForm.negrisk_min_total_yes} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_min_total_yes: parseFloat(e.target.value) || 0.95 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Hard reject below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Warn Total YES</Label>
                            <Input type="number" value={searchFiltersForm.negrisk_warn_total_yes} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_warn_total_yes: parseFloat(e.target.value) || 0.97 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Warn below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Election Min Total YES</Label>
                            <Input type="number" value={searchFiltersForm.negrisk_election_min_total_yes} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_election_min_total_yes: parseFloat(e.target.value) || 0.97 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Stricter for elections</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Resolution Spread (days)</Label>
                            <Input type="number" value={searchFiltersForm.negrisk_max_resolution_spread_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_max_resolution_spread_days: parseInt(e.target.value) || 7 }))} min={0} max={365} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Max date spread in bundle</p>
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      {/* Settlement Lag */}
                      <div>
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Settlement Lag</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Max Days to Resolution</Label>
                            <Input type="number" value={searchFiltersForm.settlement_lag_max_days_to_resolution} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_max_days_to_resolution: parseInt(e.target.value) || 14 }))} min={0} max={365} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Detection window</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Near-Zero Threshold</Label>
                            <Input type="number" value={searchFiltersForm.settlement_lag_near_zero} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_near_zero: parseFloat(e.target.value) || 0.05 }))} min={0.001} max={0.5} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Below = resolved NO</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Near-One Threshold</Label>
                            <Input type="number" value={searchFiltersForm.settlement_lag_near_one} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_near_one: parseFloat(e.target.value) || 0.95 }))} min={0.5} max={0.999} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Above = resolved YES</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Sum Deviation</Label>
                            <Input type="number" value={searchFiltersForm.settlement_lag_min_sum_deviation} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_min_sum_deviation: parseFloat(e.target.value) || 0.03 }))} min={0.001} max={0.5} step="0.005" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Min deviation from 1.0</p>
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      {/* Risk Scoring */}
                      <div>
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Risk Scoring Thresholds</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Very Short Days</Label>
                            <Input type="number" value={searchFiltersForm.risk_very_short_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_very_short_days: parseInt(e.target.value) || 2 }))} min={0} max={30} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">High risk if &lt; this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Short Days</Label>
                            <Input type="number" value={searchFiltersForm.risk_short_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_short_days: parseInt(e.target.value) || 7 }))} min={1} max={60} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Moderate risk if &lt; this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Long Lockup Days</Label>
                            <Input type="number" value={searchFiltersForm.risk_long_lockup_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_long_lockup_days: parseInt(e.target.value) || 180 }))} min={30} max={3650} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">High risk if &gt; this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Extended Lockup Days</Label>
                            <Input type="number" value={searchFiltersForm.risk_extended_lockup_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_extended_lockup_days: parseInt(e.target.value) || 90 }))} min={14} max={1825} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Moderate risk if &gt; this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Low Liquidity ($)</Label>
                            <Input type="number" value={searchFiltersForm.risk_low_liquidity} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_low_liquidity: parseFloat(e.target.value) || 1000 }))} min={0} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">High risk below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Moderate Liquidity ($)</Label>
                            <Input type="number" value={searchFiltersForm.risk_moderate_liquidity} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_moderate_liquidity: parseFloat(e.target.value) || 5000 }))} min={0} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Moderate risk below this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Complex Legs Threshold</Label>
                            <Input type="number" value={searchFiltersForm.risk_complex_legs} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_complex_legs: parseInt(e.target.value) || 5 }))} min={2} max={20} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Above = complex trade risk</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Multiple Legs Threshold</Label>
                            <Input type="number" value={searchFiltersForm.risk_multiple_legs} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_multiple_legs: parseInt(e.target.value) || 3 }))} min={2} max={20} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Above = multi-position risk</p>
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      {/* Strategy-Specific */}
                      <div>
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">BTC/ETH High-Frequency</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Pure Arb Max Combined</Label>
                            <Input type="number" value={searchFiltersForm.btc_eth_pure_arb_max_combined} onChange={(e) => setSearchFiltersForm(p => ({ ...p, btc_eth_pure_arb_max_combined: parseFloat(e.target.value) || 0.98 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Pure arb when YES+NO &lt; this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Dump-Hedge Drop %</Label>
                            <Input type="number" value={searchFiltersForm.btc_eth_dump_hedge_drop_pct} onChange={(e) => setSearchFiltersForm(p => ({ ...p, btc_eth_dump_hedge_drop_pct: parseFloat(e.target.value) || 0.05 }))} min={0.01} max={0.5} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Min drop to trigger hedge</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Thin Liquidity ($)</Label>
                            <Input type="number" value={searchFiltersForm.btc_eth_thin_liquidity_usd} onChange={(e) => setSearchFiltersForm(p => ({ ...p, btc_eth_thin_liquidity_usd: parseFloat(e.target.value) || 500 }))} min={0} className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Below = thin order book</p>
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div>
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Miracle Strategy</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                          <div>
                            <Label className="text-xs text-muted-foreground">Min NO Price</Label>
                            <Input type="number" value={searchFiltersForm.miracle_min_no_price} onChange={(e) => setSearchFiltersForm(p => ({ ...p, miracle_min_no_price: parseFloat(e.target.value) || 0.90 }))} min={0.5} max={0.999} step="0.01" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Only consider NO &ge; this</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Max NO Price</Label>
                            <Input type="number" value={searchFiltersForm.miracle_max_no_price} onChange={(e) => setSearchFiltersForm(p => ({ ...p, miracle_max_no_price: parseFloat(e.target.value) || 0.995 }))} min={0.9} max={1} step="0.005" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Skip if NO already at this+</p>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Min Impossibility Score</Label>
                            <Input type="number" value={searchFiltersForm.miracle_min_impossibility_score} onChange={(e) => setSearchFiltersForm(p => ({ ...p, miracle_min_impossibility_score: parseFloat(e.target.value) || 0.70 }))} min={0} max={1} step="0.05" className="mt-1 text-sm" />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">Min confidence event is impossible</p>
                          </div>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2">
                        <Button size="sm" onClick={() => handleSaveSection('search_filters')} disabled={saveMutation.isPending}>
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

                  {/* Auto Trader AI Settings */}
                  {section.id === 'autotrader' && (
                    <div className="space-y-4">
                      <div className="space-y-3">
                        {/* LLM Verify Before Trading */}
                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">LLM Verify Before Trading</p>
                              <p className="text-xs text-muted-foreground">Consult AI before executing each auto-trade. Trades scored as &quot;skip&quot; or &quot;strong_skip&quot; are blocked.</p>
                            </div>
                            <Switch
                              checked={autotraderAiForm.llm_verify_trades}
                              onCheckedChange={(checked) => {
                                setAutotraderAiForm(p => ({ ...p, llm_verify_trades: checked }))
                                setAutotraderAiDirty(true)
                              }}
                            />
                          </CardContent>
                        </Card>

                        {/* Strategies to LLM-Verify */}
                        {autotraderAiForm.llm_verify_trades && (
                          <div>
                            <Label className="text-xs text-muted-foreground">Strategies to LLM-Verify</Label>
                            <Input
                              type="text"
                              value={autotraderAiForm.llm_verify_strategies}
                              onChange={(e) => {
                                setAutotraderAiForm(p => ({ ...p, llm_verify_strategies: e.target.value }))
                                setAutotraderAiDirty(true)
                              }}
                              placeholder="e.g. basic, negrisk, miracle (empty = verify all)"
                              className="mt-1 text-sm"
                            />
                            <p className="text-[11px] text-muted-foreground/70 mt-1">
                              Comma-separated list of strategy types to verify. Leave empty to verify all strategies.
                            </p>
                          </div>
                        )}

                        <Separator className="opacity-30" />

                        {/* Auto AI Scoring */}
                        <Card className="bg-muted">
                          <CardContent className="flex items-center justify-between p-3">
                            <div>
                              <p className="font-medium text-sm">Auto AI Scoring</p>
                              <p className="text-xs text-muted-foreground">Automatically AI-score all opportunities after each scan. Manual analysis per opportunity is always available regardless of this setting.</p>
                            </div>
                            <Switch
                              checked={autotraderAiForm.auto_ai_scoring}
                              onCheckedChange={(checked) => {
                                setAutotraderAiForm(p => ({ ...p, auto_ai_scoring: checked }))
                                setAutotraderAiDirty(true)
                              }}
                            />
                          </CardContent>
                        </Card>

                        {/* Info Note */}
                        <div className="flex items-start gap-2 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                          <Activity className="w-4 h-4 text-blue-400 mt-0.5 shrink-0" />
                          <p className="text-xs text-muted-foreground">
                            When LLM Verify is enabled, the auto-trader will consult AI before executing trades. Disable for faster execution.
                            Additional auto-trader settings (position sizing, risk management, spread exits, AI resolution gate) are available under Trading &gt; Auto Trader &gt; Settings.
                          </p>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      {/* Enabled Strategies */}
                      <div className="space-y-3">
                        <div>
                          <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Enabled Strategies</h4>
                          <p className="text-xs text-muted-foreground">Select which strategies the auto trader should use</p>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {ALL_STRATEGIES.map(s => {
                            const enabled = autotraderAiForm.enabled_strategies.includes(s.key)
                            return (
                              <button
                                key={s.key}
                                type="button"
                                onClick={() => {
                                  setAutotraderAiForm(prev => ({
                                    ...prev,
                                    enabled_strategies: enabled
                                      ? prev.enabled_strategies.filter(k => k !== s.key)
                                      : [...prev.enabled_strategies, s.key]
                                  }))
                                  setAutotraderAiDirty(true)
                                }}
                                className={cn(
                                  "px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors",
                                  enabled
                                    ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                                    : "bg-muted text-muted-foreground border-border hover:border-emerald-500/20"
                                )}
                              >
                                {s.label}
                              </button>
                            )
                          })}
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => {
                              setAutotraderAiForm(p => ({ ...p, enabled_strategies: ALL_STRATEGIES.map(s => s.key) }))
                              setAutotraderAiDirty(true)
                            }}
                          >
                            Select All
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => {
                              setAutotraderAiForm(p => ({ ...p, enabled_strategies: [] }))
                              setAutotraderAiDirty(true)
                            }}
                          >
                            Clear All
                          </Button>
                        </div>
                      </div>

                      <Separator className="opacity-30" />

                      <div className="flex items-center gap-2 flex-wrap">
                        <Button
                          size="sm"
                          onClick={() => handleSaveSection('autotrader')}
                          disabled={autoTraderConfigMutation.isPending || !autotraderAiDirty}
                        >
                          <Save className="w-3.5 h-3.5 mr-1.5" />
                          {autoTraderConfigMutation.isPending ? 'Saving...' : 'Save'}
                        </Button>
                        {autotraderAiDirty && (
                          <Badge variant="outline" className="text-[10px] text-yellow-400 border-yellow-500/30 bg-yellow-500/10">
                            Unsaved changes
                          </Badge>
                        )}
                      </div>
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
