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
  getLLMModels,
  refreshLLMModels,
  getAutoTraderStatus,
  updateAutoTraderConfig,
  type LLMModelOption
} from '../services/api'

type SettingsSection = 'polymarket' | 'llm' | 'notifications' | 'scanner' | 'trading' | 'autotrader' | 'maintenance'

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
          className="pr-10 font-mono"
        />
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="absolute right-0 top-0 h-full px-3"
          onClick={onToggle}
        >
          {showSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </Button>
      </div>
      {description && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
    </div>
  )
}

export default function SettingsPanel() {
  const [activeSection, setActiveSection] = useState<SettingsSection>('polymarket')
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Form state for each section
  const [polymarketForm, setPolymarketForm] = useState({
    api_key: '',
    api_secret: '',
    api_passphrase: '',
    private_key: ''
  })

  const [llmForm, setLlmForm] = useState({
    provider: 'none',
    openai_api_key: '',
    anthropic_api_key: '',
    google_api_key: '',
    xai_api_key: '',
    deepseek_api_key: '',
    model: ''
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

  const [autoTraderForm, setAutoTraderForm] = useState({
    min_roi_percent: 2.5,
    max_risk_score: 0.5,
    base_position_size_usd: 10,
    max_position_size_usd: 100,
    max_daily_trades: 50,
    max_daily_loss_usd: 100,
    paper_account_capital: 10000
  })

  const queryClient = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const { data: autoTraderStatus } = useQuery({
    queryKey: ['auto-trader-status'],
    queryFn: getAutoTraderStatus,
  })

  // Sync auto trader form with loaded config
  useEffect(() => {
    if (autoTraderStatus?.config) {
      const c = autoTraderStatus.config
      setAutoTraderForm({
        min_roi_percent: c.min_roi_percent ?? 2.5,
        max_risk_score: c.max_risk_score ?? 0.5,
        base_position_size_usd: c.base_position_size_usd ?? 10,
        max_position_size_usd: c.max_position_size_usd ?? 100,
        max_daily_trades: c.max_daily_trades ?? 50,
        max_daily_loss_usd: c.max_daily_loss_usd ?? 100,
        paper_account_capital: c.paper_account_capital ?? 10000
      })
    }
  }, [autoTraderStatus])

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
        model: settings.llm.model || ''
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
      setSaveMessage({ type: 'success', text: 'Settings saved successfully' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save settings' })
      setTimeout(() => setSaveMessage(null), 5000)
    }
  })

  const autoTraderConfigMutation = useMutation({
    mutationFn: updateAutoTraderConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trader-status'] })
      setSaveMessage({ type: 'success', text: 'Auto trader config saved successfully' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save auto trader config' })
      setTimeout(() => setSaveMessage(null), 5000)
    }
  })

  const testPolymarketMutation = useMutation({
    mutationFn: testPolymarketConnection,
  })

  const testTelegramMutation = useMutation({
    mutationFn: testTelegramConnection,
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
      case 'llm':
        updates.llm = {
          provider: llmForm.provider,
          model: llmForm.model || null
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
      case 'maintenance':
        updates.maintenance = maintenanceForm
        break
      case 'autotrader':
        autoTraderConfigMutation.mutate(autoTraderForm)
        return
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

  const sections: { id: SettingsSection; icon: any; label: string; description: string }[] = [
    { id: 'polymarket', icon: Key, label: 'Polymarket Account', description: 'API credentials for trading' },
    { id: 'llm', icon: Bot, label: 'AI / LLM Services', description: 'Configure AI providers' },
    { id: 'notifications', icon: Bell, label: 'Notifications', description: 'Telegram alerts' },
    { id: 'scanner', icon: Scan, label: 'Scanner', description: 'Market scanning settings' },
    { id: 'trading', icon: TrendingUp, label: 'Trading Safety', description: 'Trading limits & safety' },
    { id: 'autotrader', icon: Activity, label: 'Auto Trader', description: 'Auto trading configuration' },
    { id: 'maintenance', icon: Database, label: 'Database', description: 'Cleanup & maintenance' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Settings</h2>
          <p className="text-sm text-muted-foreground">
            Configure application settings and integrations
          </p>
        </div>
        {settings?.updated_at && (
          <div className="text-xs text-muted-foreground">
            Last updated: {new Date(settings.updated_at).toLocaleString()}
          </div>
        )}
      </div>

      {/* Save Message */}
      {saveMessage && (
        <div className={cn(
          "flex items-center gap-2 p-3 rounded-lg text-sm",
          saveMessage.type === 'success' ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
        )}>
          {saveMessage.type === 'success' ? (
            <CheckCircle className="w-4 h-4" />
          ) : (
            <AlertCircle className="w-4 h-4" />
          )}
          {saveMessage.text}
        </div>
      )}

      <div className="flex gap-6">
        {/* Sidebar Navigation */}
        <div className="w-64 space-y-1">
          {sections.map(section => (
            <Button
              key={section.id}
              variant={activeSection === section.id ? "default" : "ghost"}
              className={cn(
                "w-full justify-start gap-3 h-auto py-3",
                activeSection === section.id && "bg-primary/10 text-primary"
              )}
              onClick={() => setActiveSection(section.id)}
            >
              <section.icon className="w-5 h-5" />
              <div className="text-left">
                <div className="font-medium text-sm">{section.label}</div>
                <div className="text-xs text-muted-foreground">{section.description}</div>
              </div>
            </Button>
          ))}
        </div>

        {/* Main Content */}
        <Card className="flex-1">
          <CardContent className="p-6">
            {/* Polymarket Settings */}
            {activeSection === 'polymarket' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-green-500/10 rounded-lg">
                    <Key className="w-5 h-5 text-green-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Polymarket Account</h3>
                    <p className="text-sm text-muted-foreground">Configure your Polymarket API credentials for live trading</p>
                  </div>
                </div>

                <div className="space-y-4">
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

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('polymarket')} disabled={saveMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save Polymarket Settings
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => testPolymarketMutation.mutate()}
                    disabled={testPolymarketMutation.isPending}
                  >
                    <Zap className="w-4 h-4 mr-2" />
                    Test Connection
                  </Button>
                  {testPolymarketMutation.data && (
                    <Badge variant={testPolymarketMutation.data.status === 'success' ? "default" : "outline"} className={cn(
                      "text-sm",
                      testPolymarketMutation.data.status === 'success' ? "bg-green-500/10 text-green-400" : "bg-yellow-500/10 text-yellow-400"
                    )}>
                      {testPolymarketMutation.data.message}
                    </Badge>
                  )}
                </div>
              </div>
            )}

            {/* LLM Settings */}
            {activeSection === 'llm' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-purple-500/10 rounded-lg">
                    <Bot className="w-5 h-5 text-purple-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">AI / LLM Services</h3>
                    <p className="text-sm text-muted-foreground">Configure AI providers for analysis and insights</p>
                  </div>
                </div>

                <div className="space-y-4">
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
                    <p className="text-xs text-muted-foreground mt-1">
                      {modelsForProvider.length > 0
                        ? `${modelsForProvider.length} models available`
                        : llmForm.provider !== 'none'
                          ? 'Click refresh to fetch available models from the API'
                          : 'Select a provider first'}
                    </p>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('llm')} disabled={saveMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save LLM Settings
                  </Button>
                </div>
              </div>
            )}

            {/* Notification Settings */}
            {activeSection === 'notifications' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <Bell className="w-5 h-5 text-blue-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Notifications</h3>
                    <p className="text-sm text-muted-foreground">Configure Telegram alerts for opportunities and trades</p>
                  </div>
                </div>

                <div className="space-y-4">
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
                      className="mt-1"
                    />
                  </div>

                  <div className="space-y-3 pt-4">
                    <p className="text-sm font-medium">Alert Types</p>

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
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('notifications')} disabled={saveMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save Notification Settings
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => testTelegramMutation.mutate()}
                    disabled={testTelegramMutation.isPending}
                  >
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Test Telegram
                  </Button>
                  {testTelegramMutation.data && (
                    <Badge variant={testTelegramMutation.data.status === 'success' ? "default" : "outline"} className={cn(
                      "text-sm",
                      testTelegramMutation.data.status === 'success' ? "bg-green-500/10 text-green-400" : "bg-yellow-500/10 text-yellow-400"
                    )}>
                      {testTelegramMutation.data.message}
                    </Badge>
                  )}
                </div>
              </div>
            )}

            {/* Scanner Settings */}
            {activeSection === 'scanner' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-cyan-500/10 rounded-lg">
                    <Scan className="w-5 h-5 text-cyan-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Scanner Settings</h3>
                    <p className="text-sm text-muted-foreground">Configure market scanning behavior</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Scan Interval (seconds)</Label>
                    <Input
                      type="number"
                      value={scannerForm.scan_interval_seconds}
                      onChange={(e) => setScannerForm(p => ({ ...p, scan_interval_seconds: parseInt(e.target.value) || 60 }))}
                      min={10}
                      max={3600}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">How often to scan for opportunities</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Min Profit Threshold (%)</Label>
                    <Input
                      type="number"
                      value={scannerForm.min_profit_threshold}
                      onChange={(e) => setScannerForm(p => ({ ...p, min_profit_threshold: parseFloat(e.target.value) || 0 }))}
                      step="0.5"
                      min={0}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Minimum ROI to report as opportunity</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Max Markets to Scan</Label>
                    <Input
                      type="number"
                      value={scannerForm.max_markets_to_scan}
                      onChange={(e) => setScannerForm(p => ({ ...p, max_markets_to_scan: parseInt(e.target.value) || 100 }))}
                      min={10}
                      max={5000}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Limit on markets per scan</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Min Liquidity ($)</Label>
                    <Input
                      type="number"
                      value={scannerForm.min_liquidity}
                      onChange={(e) => setScannerForm(p => ({ ...p, min_liquidity: parseFloat(e.target.value) || 0 }))}
                      min={0}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Minimum market liquidity</p>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('scanner')} disabled={saveMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save Scanner Settings
                  </Button>
                </div>
              </div>
            )}

            {/* Trading Settings */}
            {activeSection === 'trading' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-yellow-500/10 rounded-lg">
                    <TrendingUp className="w-5 h-5 text-yellow-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Trading Safety</h3>
                    <p className="text-sm text-muted-foreground">Configure trading limits and safety controls</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <Card className="bg-muted border-yellow-500/30">
                    <CardContent className="flex items-center justify-between p-4">
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

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-xs text-muted-foreground">Max Trade Size ($)</Label>
                      <Input
                        type="number"
                        value={tradingForm.max_trade_size_usd}
                        onChange={(e) => setTradingForm(p => ({ ...p, max_trade_size_usd: parseFloat(e.target.value) || 0 }))}
                        min={1}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-xs text-muted-foreground">Max Daily Volume ($)</Label>
                      <Input
                        type="number"
                        value={tradingForm.max_daily_trade_volume}
                        onChange={(e) => setTradingForm(p => ({ ...p, max_daily_trade_volume: parseFloat(e.target.value) || 0 }))}
                        min={10}
                        className="mt-1"
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
                        className="mt-1"
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
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('trading')} disabled={saveMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save Trading Settings
                  </Button>
                </div>
              </div>
            )}

            {/* Auto Trader Config */}
            {activeSection === 'autotrader' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-emerald-500/10 rounded-lg">
                    <Activity className="w-5 h-5 text-emerald-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Auto Trader Configuration</h3>
                    <p className="text-sm text-muted-foreground">Configure autonomous trading parameters and limits</p>
                  </div>
                </div>

                {autoTraderStatus?.config && (
                  <Card className="bg-muted">
                    <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Mode</p>
                        <p className="font-mono text-sm font-medium">{autoTraderStatus.config.mode?.toUpperCase()}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Status</p>
                        <Badge variant={autoTraderStatus.running ? "default" : "secondary"} className={cn(
                          "font-mono text-sm font-medium",
                          autoTraderStatus.running ? "bg-green-500/10 text-green-400" : "bg-muted text-muted-foreground"
                        )}>
                          {autoTraderStatus.running ? 'Running' : 'Stopped'}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Strategies</p>
                        <p className="font-mono text-xs">{autoTraderStatus.config.enabled_strategies?.join(', ') || 'All'}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Circuit Breaker</p>
                        <p className="font-mono text-sm">{autoTraderStatus.config.circuit_breaker_losses} losses</p>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Min ROI %</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.min_roi_percent}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, min_roi_percent: parseFloat(e.target.value) || 0 }))}
                      step="0.5"
                      min={0}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Minimum return to execute a trade</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Max Risk Score</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.max_risk_score}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, max_risk_score: parseFloat(e.target.value) || 0 }))}
                      step="0.1"
                      min={0}
                      max={1}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">0 = lowest risk, 1 = highest</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Base Position Size ($)</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.base_position_size_usd}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, base_position_size_usd: parseFloat(e.target.value) || 1 }))}
                      min={1}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Default trade size</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Max Position Size ($)</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.max_position_size_usd}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, max_position_size_usd: parseFloat(e.target.value) || 1 }))}
                      min={1}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Maximum single trade size</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Max Daily Trades</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.max_daily_trades}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, max_daily_trades: parseInt(e.target.value) || 1 }))}
                      min={1}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Daily trade count limit</p>
                  </div>

                  <div>
                    <Label className="text-xs text-muted-foreground">Max Daily Loss ($)</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.max_daily_loss_usd}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, max_daily_loss_usd: parseFloat(e.target.value) || 0 }))}
                      min={0}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Stop trading after this daily loss</p>
                  </div>

                  <div className="col-span-2">
                    <Label className="text-xs text-muted-foreground">Paper Account Capital ($)</Label>
                    <Input
                      type="number"
                      value={autoTraderForm.paper_account_capital}
                      onChange={(e) => setAutoTraderForm(p => ({ ...p, paper_account_capital: parseFloat(e.target.value) || 100 }))}
                      min={100}
                      className="mt-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Starting capital for paper trading simulation</p>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('autotrader')} disabled={autoTraderConfigMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save Auto Trader Config
                  </Button>
                </div>
              </div>
            )}

            {/* Maintenance Settings */}
            {activeSection === 'maintenance' && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-red-500/10 rounded-lg">
                    <Database className="w-5 h-5 text-red-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Database Maintenance</h3>
                    <p className="text-sm text-muted-foreground">Configure automatic cleanup of old data</p>
                  </div>
                </div>

                <div className="space-y-4">
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

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-xs text-muted-foreground">Cleanup Interval (hours)</Label>
                      <Input
                        type="number"
                        value={maintenanceForm.cleanup_interval_hours}
                        onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_interval_hours: parseInt(e.target.value) || 24 }))}
                        min={1}
                        max={168}
                        className="mt-1"
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
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-3">
                  <Button onClick={() => handleSaveSection('maintenance')} disabled={saveMutation.isPending}>
                    <Save className="w-4 h-4 mr-2" />
                    Save Maintenance Settings
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
