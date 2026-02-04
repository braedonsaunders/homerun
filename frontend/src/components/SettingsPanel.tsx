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
  MessageSquare
} from 'lucide-react'
import clsx from 'clsx'
import {
  getSettings,
  updateSettings,
  testPolymarketConnection,
  testTelegramConnection
} from '../services/api'

type SettingsSection = 'polymarket' | 'llm' | 'notifications' | 'scanner' | 'trading' | 'maintenance'

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
    model: ''
  })

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

  const queryClient = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

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
    }

    saveMutation.mutate(updates)
  }

  const toggleSecret = (key: string) => {
    setShowSecrets(prev => ({ ...prev, [key]: !prev[key] }))
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-500" />
      </div>
    )
  }

  const sections: { id: SettingsSection; icon: any; label: string; description: string }[] = [
    { id: 'polymarket', icon: Key, label: 'Polymarket Account', description: 'API credentials for trading' },
    { id: 'llm', icon: Bot, label: 'AI / LLM Services', description: 'Configure AI providers' },
    { id: 'notifications', icon: Bell, label: 'Notifications', description: 'Telegram alerts' },
    { id: 'scanner', icon: Scan, label: 'Scanner', description: 'Market scanning settings' },
    { id: 'trading', icon: TrendingUp, label: 'Trading Safety', description: 'Trading limits & safety' },
    { id: 'maintenance', icon: Database, label: 'Database', description: 'Cleanup & maintenance' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Settings</h2>
          <p className="text-sm text-gray-500">
            Configure application settings and integrations
          </p>
        </div>
        {settings?.updated_at && (
          <div className="text-xs text-gray-500">
            Last updated: {new Date(settings.updated_at).toLocaleString()}
          </div>
        )}
      </div>

      {/* Save Message */}
      {saveMessage && (
        <div className={clsx(
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
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={clsx(
                "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors",
                activeSection === section.id
                  ? "bg-green-500/10 text-green-400 border border-green-500/30"
                  : "bg-[#141414] text-gray-400 hover:text-white border border-gray-800 hover:border-gray-700"
              )}
            >
              <section.icon className="w-5 h-5" />
              <div>
                <div className="font-medium text-sm">{section.label}</div>
                <div className="text-xs text-gray-500">{section.description}</div>
              </div>
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="flex-1 bg-[#141414] border border-gray-800 rounded-lg p-6">
          {/* Polymarket Settings */}
          {activeSection === 'polymarket' && (
            <div className="space-y-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-green-500/10 rounded-lg">
                  <Key className="w-5 h-5 text-green-500" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Polymarket Account</h3>
                  <p className="text-sm text-gray-500">Configure your Polymarket API credentials for live trading</p>
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

              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => handleSaveSection('polymarket')}
                  disabled={saveMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
                >
                  <Save className="w-4 h-4" />
                  Save Polymarket Settings
                </button>
                <button
                  onClick={() => testPolymarketMutation.mutate()}
                  disabled={testPolymarketMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
                >
                  <Zap className="w-4 h-4" />
                  Test Connection
                </button>
                {testPolymarketMutation.data && (
                  <span className={clsx(
                    "text-sm",
                    testPolymarketMutation.data.status === 'success' ? "text-green-400" : "text-yellow-400"
                  )}>
                    {testPolymarketMutation.data.message}
                  </span>
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
                  <p className="text-sm text-gray-500">Configure AI providers for analysis and insights</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">LLM Provider</label>
                  <select
                    value={llmForm.provider}
                    onChange={(e) => setLlmForm(p => ({ ...p, provider: e.target.value }))}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="none">None (Disabled)</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
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

                <div>
                  <label className="block text-xs text-gray-500 mb-1">Model</label>
                  <input
                    type="text"
                    value={llmForm.model}
                    onChange={(e) => setLlmForm(p => ({ ...p, model: e.target.value }))}
                    placeholder="e.g., gpt-4, claude-3-opus"
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => handleSaveSection('llm')}
                  disabled={saveMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
                >
                  <Save className="w-4 h-4" />
                  Save LLM Settings
                </button>
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
                  <p className="text-sm text-gray-500">Configure Telegram alerts for opportunities and trades</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
                  <div>
                    <p className="font-medium text-sm">Enable Notifications</p>
                    <p className="text-xs text-gray-500">Receive alerts via Telegram</p>
                  </div>
                  <button
                    onClick={() => setNotificationsForm(p => ({ ...p, enabled: !p.enabled }))}
                    className={clsx(
                      "w-12 h-6 rounded-full transition-colors relative",
                      notificationsForm.enabled ? "bg-green-500" : "bg-gray-600"
                    )}
                  >
                    <div className={clsx(
                      "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform",
                      notificationsForm.enabled ? "translate-x-6" : "translate-x-0.5"
                    )} />
                  </button>
                </div>

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
                  <label className="block text-xs text-gray-500 mb-1">Telegram Chat ID</label>
                  <input
                    type="text"
                    value={notificationsForm.telegram_chat_id}
                    onChange={(e) => setNotificationsForm(p => ({ ...p, telegram_chat_id: e.target.value }))}
                    placeholder="Your chat ID"
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                </div>

                <div className="space-y-3 pt-4">
                  <p className="text-sm font-medium">Alert Types</p>

                  <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
                    <div>
                      <p className="text-sm">New Opportunities</p>
                      <p className="text-xs text-gray-500">Alert when new arbitrage opportunities are found</p>
                    </div>
                    <button
                      onClick={() => setNotificationsForm(p => ({ ...p, notify_on_opportunity: !p.notify_on_opportunity }))}
                      className={clsx(
                        "w-12 h-6 rounded-full transition-colors relative",
                        notificationsForm.notify_on_opportunity ? "bg-green-500" : "bg-gray-600"
                      )}
                    >
                      <div className={clsx(
                        "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform",
                        notificationsForm.notify_on_opportunity ? "translate-x-6" : "translate-x-0.5"
                      )} />
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
                    <div>
                      <p className="text-sm">Trade Executions</p>
                      <p className="text-xs text-gray-500">Alert when trades are executed</p>
                    </div>
                    <button
                      onClick={() => setNotificationsForm(p => ({ ...p, notify_on_trade: !p.notify_on_trade }))}
                      className={clsx(
                        "w-12 h-6 rounded-full transition-colors relative",
                        notificationsForm.notify_on_trade ? "bg-green-500" : "bg-gray-600"
                      )}
                    >
                      <div className={clsx(
                        "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform",
                        notificationsForm.notify_on_trade ? "translate-x-6" : "translate-x-0.5"
                      )} />
                    </button>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Minimum ROI for Alerts (%)</label>
                    <input
                      type="number"
                      value={notificationsForm.notify_min_roi}
                      onChange={(e) => setNotificationsForm(p => ({ ...p, notify_min_roi: parseFloat(e.target.value) || 0 }))}
                      step="0.5"
                      min="0"
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => handleSaveSection('notifications')}
                  disabled={saveMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
                >
                  <Save className="w-4 h-4" />
                  Save Notification Settings
                </button>
                <button
                  onClick={() => testTelegramMutation.mutate()}
                  disabled={testTelegramMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
                >
                  <MessageSquare className="w-4 h-4" />
                  Test Telegram
                </button>
                {testTelegramMutation.data && (
                  <span className={clsx(
                    "text-sm",
                    testTelegramMutation.data.status === 'success' ? "text-green-400" : "text-yellow-400"
                  )}>
                    {testTelegramMutation.data.message}
                  </span>
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
                  <p className="text-sm text-gray-500">Configure market scanning behavior</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Scan Interval (seconds)</label>
                  <input
                    type="number"
                    value={scannerForm.scan_interval_seconds}
                    onChange={(e) => setScannerForm(p => ({ ...p, scan_interval_seconds: parseInt(e.target.value) || 60 }))}
                    min={10}
                    max={3600}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                  <p className="text-xs text-gray-600 mt-1">How often to scan for opportunities</p>
                </div>

                <div>
                  <label className="block text-xs text-gray-500 mb-1">Min Profit Threshold (%)</label>
                  <input
                    type="number"
                    value={scannerForm.min_profit_threshold}
                    onChange={(e) => setScannerForm(p => ({ ...p, min_profit_threshold: parseFloat(e.target.value) || 0 }))}
                    step="0.5"
                    min={0}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                  <p className="text-xs text-gray-600 mt-1">Minimum ROI to report as opportunity</p>
                </div>

                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max Markets to Scan</label>
                  <input
                    type="number"
                    value={scannerForm.max_markets_to_scan}
                    onChange={(e) => setScannerForm(p => ({ ...p, max_markets_to_scan: parseInt(e.target.value) || 100 }))}
                    min={10}
                    max={5000}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                  <p className="text-xs text-gray-600 mt-1">Limit on markets per scan</p>
                </div>

                <div>
                  <label className="block text-xs text-gray-500 mb-1">Min Liquidity ($)</label>
                  <input
                    type="number"
                    value={scannerForm.min_liquidity}
                    onChange={(e) => setScannerForm(p => ({ ...p, min_liquidity: parseFloat(e.target.value) || 0 }))}
                    min={0}
                    className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                  <p className="text-xs text-gray-600 mt-1">Minimum market liquidity</p>
                </div>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => handleSaveSection('scanner')}
                  disabled={saveMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
                >
                  <Save className="w-4 h-4" />
                  Save Scanner Settings
                </button>
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
                  <p className="text-sm text-gray-500">Configure trading limits and safety controls</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-lg border border-yellow-500/30">
                  <div>
                    <p className="font-medium text-sm">Enable Live Trading</p>
                    <p className="text-xs text-gray-500">Allow real money trades on Polymarket</p>
                  </div>
                  <button
                    onClick={() => {
                      if (!tradingForm.trading_enabled || confirm('Are you sure you want to enable live trading?')) {
                        setTradingForm(p => ({ ...p, trading_enabled: !p.trading_enabled }))
                      }
                    }}
                    className={clsx(
                      "w-12 h-6 rounded-full transition-colors relative",
                      tradingForm.trading_enabled ? "bg-yellow-500" : "bg-gray-600"
                    )}
                  >
                    <div className={clsx(
                      "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform",
                      tradingForm.trading_enabled ? "translate-x-6" : "translate-x-0.5"
                    )} />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Max Trade Size ($)</label>
                    <input
                      type="number"
                      value={tradingForm.max_trade_size_usd}
                      onChange={(e) => setTradingForm(p => ({ ...p, max_trade_size_usd: parseFloat(e.target.value) || 0 }))}
                      min={1}
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Max Daily Volume ($)</label>
                    <input
                      type="number"
                      value={tradingForm.max_daily_trade_volume}
                      onChange={(e) => setTradingForm(p => ({ ...p, max_daily_trade_volume: parseFloat(e.target.value) || 0 }))}
                      min={10}
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Max Open Positions</label>
                    <input
                      type="number"
                      value={tradingForm.max_open_positions}
                      onChange={(e) => setTradingForm(p => ({ ...p, max_open_positions: parseInt(e.target.value) || 1 }))}
                      min={1}
                      max={100}
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Max Slippage (%)</label>
                    <input
                      type="number"
                      value={tradingForm.max_slippage_percent}
                      onChange={(e) => setTradingForm(p => ({ ...p, max_slippage_percent: parseFloat(e.target.value) || 0 }))}
                      step="0.1"
                      min={0.1}
                      max={10}
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => handleSaveSection('trading')}
                  disabled={saveMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
                >
                  <Save className="w-4 h-4" />
                  Save Trading Settings
                </button>
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
                  <p className="text-sm text-gray-500">Configure automatic cleanup of old data</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
                  <div>
                    <p className="font-medium text-sm">Auto Cleanup</p>
                    <p className="text-xs text-gray-500">Automatically delete old records</p>
                  </div>
                  <button
                    onClick={() => setMaintenanceForm(p => ({ ...p, auto_cleanup_enabled: !p.auto_cleanup_enabled }))}
                    className={clsx(
                      "w-12 h-6 rounded-full transition-colors relative",
                      maintenanceForm.auto_cleanup_enabled ? "bg-green-500" : "bg-gray-600"
                    )}
                  >
                    <div className={clsx(
                      "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform",
                      maintenanceForm.auto_cleanup_enabled ? "translate-x-6" : "translate-x-0.5"
                    )} />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Cleanup Interval (hours)</label>
                    <input
                      type="number"
                      value={maintenanceForm.cleanup_interval_hours}
                      onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_interval_hours: parseInt(e.target.value) || 24 }))}
                      min={1}
                      max={168}
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Keep Resolved Trades (days)</label>
                    <input
                      type="number"
                      value={maintenanceForm.cleanup_resolved_trade_days}
                      onChange={(e) => setMaintenanceForm(p => ({ ...p, cleanup_resolved_trade_days: parseInt(e.target.value) || 30 }))}
                      min={1}
                      max={365}
                      className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => handleSaveSection('maintenance')}
                  disabled={saveMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg text-sm font-medium"
                >
                  <Save className="w-4 h-4" />
                  Save Maintenance Settings
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
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
      <label className="block text-xs text-gray-500 mb-1">{label}</label>
      <div className="relative">
        <input
          type={showSecret ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-2 pr-10 text-sm font-mono"
        />
        <button
          type="button"
          onClick={onToggle}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-500 hover:text-white"
        >
          {showSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </button>
      </div>
      {description && <p className="text-xs text-gray-600 mt-1">{description}</p>}
    </div>
  )
}
