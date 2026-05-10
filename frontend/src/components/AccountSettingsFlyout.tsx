import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Key,
  Save,
  Eye,
  EyeOff,
  Zap,
  Activity,
  BarChart3,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Separator } from './ui/separator'
import { Badge } from './ui/badge'
import {
  getSettings,
  updateSettings,
  testKalshiConnection,
  testPolymarketConnection,
} from '../services/api'

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

export default function AccountSettingsFlyout({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { t } = useTranslation()
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

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

  const [oracleForm, setOracleForm] = useState({
    chainlink_direct_api_key: '',
    chainlink_direct_user_secret: '',
  })

  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  useEffect(() => {
    if (settings) {
      setPolymarketForm(prev => ({
        api_key: prev.api_key || '',
        api_secret: prev.api_secret || '',
        api_passphrase: prev.api_passphrase || '',
        private_key: prev.private_key || ''
      }))
    }
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setSaveMessage({ type: 'success', text: t('accountSettingsFlyout.toastSaved') })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || t('accountSettingsFlyout.toastFailed') })
      setTimeout(() => setSaveMessage(null), 5000)
    }
  })

  const testPolymarketMutation = useMutation({
    mutationFn: testPolymarketConnection,
  })

  const testKalshiMutation = useMutation({
    mutationFn: testKalshiConnection,
  })

  const handleSavePolymarket = () => {
    const updates: any = { polymarket: {} }
    if (polymarketForm.api_key) updates.polymarket.api_key = polymarketForm.api_key
    if (polymarketForm.api_secret) updates.polymarket.api_secret = polymarketForm.api_secret
    if (polymarketForm.api_passphrase) updates.polymarket.api_passphrase = polymarketForm.api_passphrase
    if (polymarketForm.private_key) updates.polymarket.private_key = polymarketForm.private_key
    saveMutation.mutate(updates)
  }

  const handleSaveKalshi = () => {
    const updates: any = { kalshi: {} }
    if (kalshiForm.email) updates.kalshi.email = kalshiForm.email
    if (kalshiForm.password) updates.kalshi.password = kalshiForm.password
    if (kalshiForm.api_key) updates.kalshi.api_key = kalshiForm.api_key
    saveMutation.mutate(updates)
  }

  const handleSaveOracle = () => {
    const updates: any = { oracle: {} }
    if (oracleForm.chainlink_direct_api_key) {
      updates.oracle.chainlink_direct_api_key = oracleForm.chainlink_direct_api_key
    }
    if (oracleForm.chainlink_direct_user_secret) {
      updates.oracle.chainlink_direct_user_secret = oracleForm.chainlink_direct_user_secret
    }
    saveMutation.mutate(updates)
  }

  const toggleSecret = (key: string) => {
    setShowSecrets(prev => ({ ...prev, [key]: !prev[key] }))
  }

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-background/80 z-40 transition-opacity"
        onClick={onClose}
      />
      {/* Drawer */}
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-2xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 bg-background border-b border-border/40">
          <div className="flex items-center gap-2">
            <Key className="w-4 h-4 text-green-500" />
            <h3 className="text-sm font-semibold">{t('accountSettingsFlyout.title')}</h3>
          </div>
          <Button
            variant="ghost"
            onClick={onClose}
            className="text-xs h-auto px-2.5 py-1 hover:bg-card"
          >
            <X className="w-3.5 h-3.5 mr-1" />
            {t('accountSettingsFlyout.close')}
          </Button>
        </div>

        {/* Floating toast */}
        {saveMessage && (
          <div className={cn(
            "fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300",
            saveMessage.type === 'success'
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
              : "bg-red-500/10 text-red-400 border-red-500/20"
          )}>
            {saveMessage.text}
          </div>
        )}

        {/* Content */}
        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Polymarket Account */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 md:col-span-2">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <Key className="w-3.5 h-3.5 text-emerald-500" />
              {t('accountSettingsFlyout.polymarketAccount')}
              <Badge variant="outline" className={cn(
                "ml-auto text-[10px] px-2 py-0.5 border-0 shrink-0",
                (() => {
                  const keysSet = [
                    settings?.polymarket?.api_key,
                    settings?.polymarket?.api_secret,
                    settings?.polymarket?.api_passphrase,
                    settings?.polymarket?.private_key,
                  ].filter(Boolean).length
                  return keysSet >= 3 ? 'text-emerald-400 bg-emerald-500/10' : keysSet > 0 ? 'text-yellow-400 bg-yellow-500/10' : 'text-muted-foreground bg-muted'
                })()
              )}>
                {(() => {
                  const keysSet = [
                    settings?.polymarket?.api_key,
                    settings?.polymarket?.api_secret,
                    settings?.polymarket?.api_passphrase,
                    settings?.polymarket?.private_key,
                  ].filter(Boolean).length
                  return keysSet > 0 ? t('accountSettingsFlyout.keysSet', { n: keysSet }) : t('accountSettingsFlyout.notConfigured')
                })()}
              </Badge>
            </h4>
            <div className="space-y-3">
              <SecretInput
                label={t('accountSettingsFlyout.apiKey')}
                value={polymarketForm.api_key}
                placeholder={settings?.polymarket.api_key || t('accountSettingsFlyout.enterApiKey')}
                onChange={(v) => setPolymarketForm(p => ({ ...p, api_key: v }))}
                showSecret={showSecrets['pm_key']}
                onToggle={() => toggleSecret('pm_key')}
              />
              <SecretInput
                label={t('accountSettingsFlyout.apiSecret')}
                value={polymarketForm.api_secret}
                placeholder={settings?.polymarket.api_secret || t('accountSettingsFlyout.enterApiSecret')}
                onChange={(v) => setPolymarketForm(p => ({ ...p, api_secret: v }))}
                showSecret={showSecrets['pm_secret']}
                onToggle={() => toggleSecret('pm_secret')}
              />
              <SecretInput
                label={t('accountSettingsFlyout.apiPassphrase')}
                value={polymarketForm.api_passphrase}
                placeholder={settings?.polymarket.api_passphrase || t('accountSettingsFlyout.enterApiPassphrase')}
                onChange={(v) => setPolymarketForm(p => ({ ...p, api_passphrase: v }))}
                showSecret={showSecrets['pm_pass']}
                onToggle={() => toggleSecret('pm_pass')}
              />
              <SecretInput
                label={t('accountSettingsFlyout.privateKey')}
                value={polymarketForm.private_key}
                placeholder={settings?.polymarket.private_key || t('accountSettingsFlyout.enterPrivateKey')}
                onChange={(v) => setPolymarketForm(p => ({ ...p, private_key: v }))}
                showSecret={showSecrets['pm_pk']}
                onToggle={() => toggleSecret('pm_pk')}
                description={t('accountSettingsFlyout.privateKeyDescription')}
              />

              <Separator className="opacity-30" />

              <div className="flex items-center gap-2 flex-wrap">
                <Button size="sm" onClick={handleSavePolymarket} disabled={saveMutation.isPending}>
                  <Save className="w-3.5 h-3.5 mr-1.5" />
                  {t('accountSettingsFlyout.save')}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => testPolymarketMutation.mutate()}
                  disabled={testPolymarketMutation.isPending}
                >
                  <Zap className="w-3.5 h-3.5 mr-1.5" />
                  {t('accountSettingsFlyout.testConnection')}
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
          </Card>

          {/* Kalshi Account */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 md:col-span-2">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <BarChart3 className="w-3.5 h-3.5 text-indigo-500" />
              {t('accountSettingsFlyout.kalshiAccount')}
              <Badge variant="outline" className={cn(
                "ml-auto text-[10px] px-2 py-0.5 border-0 shrink-0",
                (!!settings?.kalshi?.email || !!settings?.kalshi?.api_key) ? 'text-emerald-400 bg-emerald-500/10' : 'text-muted-foreground bg-muted'
              )}>
                {(!!settings?.kalshi?.email || !!settings?.kalshi?.api_key) ? t('accountSettingsFlyout.configured') : t('accountSettingsFlyout.notConfigured')}
              </Badge>
            </h4>
            <div className="space-y-3">
              <div>
                <Label className="text-xs text-muted-foreground">{t('accountSettingsFlyout.kalshiEmail')}</Label>
                <Input
                  type="email"
                  value={kalshiForm.email}
                  onChange={(e) => setKalshiForm(p => ({ ...p, email: e.target.value }))}
                  placeholder={settings?.kalshi?.email || t('accountSettingsFlyout.enterKalshiEmail')}
                  className="mt-1 text-sm"
                />
                <p className="text-[11px] text-muted-foreground/70 mt-1">{t('accountSettingsFlyout.kalshiEmailHelp')}</p>
              </div>

              <SecretInput
                label={t('accountSettingsFlyout.kalshiPassword')}
                value={kalshiForm.password}
                placeholder={settings?.kalshi?.password || t('accountSettingsFlyout.enterKalshiPassword')}
                onChange={(v) => setKalshiForm(p => ({ ...p, password: v }))}
                showSecret={showSecrets['kalshi_pass']}
                onToggle={() => toggleSecret('kalshi_pass')}
                description={t('accountSettingsFlyout.kalshiPasswordHelp')}
              />

              <Separator className="opacity-30" />

              <SecretInput
                label={t('accountSettingsFlyout.apiKeyAlternative')}
                value={kalshiForm.api_key}
                placeholder={settings?.kalshi?.api_key || t('accountSettingsFlyout.enterKalshiApiKey')}
                onChange={(v) => setKalshiForm(p => ({ ...p, api_key: v }))}
                showSecret={showSecrets['kalshi_key']}
                onToggle={() => toggleSecret('kalshi_key')}
                description={t('accountSettingsFlyout.kalshiApiKeyHelp')}
              />

              <div className="flex items-start gap-2 p-3 bg-indigo-500/5 border border-indigo-500/20 rounded-lg">
                <Activity className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
                <p className="text-xs text-muted-foreground">
                  {t('accountSettingsFlyout.kalshiInfo')}
                </p>
              </div>

              <Separator className="opacity-30" />

              <div className="flex items-center gap-2">
                <Button size="sm" onClick={handleSaveKalshi} disabled={saveMutation.isPending}>
                  <Save className="w-3.5 h-3.5 mr-1.5" />
                  {t('accountSettingsFlyout.save')}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => testKalshiMutation.mutate()}
                  disabled={testKalshiMutation.isPending}
                >
                  <Zap className="w-3.5 h-3.5 mr-1.5" />
                  {t('accountSettingsFlyout.testConnection')}
                </Button>
                {testKalshiMutation.data && (
                  <Badge variant={testKalshiMutation.data.status === 'success' ? "default" : "outline"} className={cn(
                    "text-xs",
                    testKalshiMutation.data.status === 'success' ? "bg-green-500/10 text-green-400" : "bg-yellow-500/10 text-yellow-400"
                  )}>
                    {testKalshiMutation.data.message}
                  </Badge>
                )}
              </div>
            </div>
          </Card>

          {/* Oracle (Chainlink Data Streams) Account */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 md:col-span-2">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <Activity className="w-3.5 h-3.5 text-amber-500" />
              {t('accountSettingsFlyout.oracleTitle')}
              <Badge variant="outline" className={cn(
                "ml-auto text-[10px] px-2 py-0.5 border-0 shrink-0",
                (settings?.oracle?.chainlink_direct_api_key && settings?.oracle?.chainlink_direct_user_secret)
                  ? 'text-emerald-400 bg-emerald-500/10'
                  : (settings?.oracle?.chainlink_direct_api_key || settings?.oracle?.chainlink_direct_user_secret)
                  ? 'text-yellow-400 bg-yellow-500/10'
                  : 'text-muted-foreground bg-muted'
              )}>
                {(settings?.oracle?.chainlink_direct_api_key && settings?.oracle?.chainlink_direct_user_secret)
                  ? t('accountSettingsFlyout.configured')
                  : (settings?.oracle?.chainlink_direct_api_key || settings?.oracle?.chainlink_direct_user_secret)
                  ? t('accountSettingsFlyout.partial')
                  : t('accountSettingsFlyout.notConfigured')}
              </Badge>
            </h4>
            <div className="space-y-3">
              <SecretInput
                label={t('accountSettingsFlyout.chainlinkApiKey')}
                value={oracleForm.chainlink_direct_api_key}
                placeholder={settings?.oracle?.chainlink_direct_api_key || t('accountSettingsFlyout.enterChainlinkApiKey')}
                onChange={(v) => setOracleForm(p => ({ ...p, chainlink_direct_api_key: v }))}
                showSecret={showSecrets['cl_key']}
                onToggle={() => toggleSecret('cl_key')}
                description={t('accountSettingsFlyout.chainlinkApiKeyHelp')}
              />
              <SecretInput
                label={t('accountSettingsFlyout.chainlinkUserSecret')}
                value={oracleForm.chainlink_direct_user_secret}
                placeholder={settings?.oracle?.chainlink_direct_user_secret || t('accountSettingsFlyout.enterChainlinkUserSecret')}
                onChange={(v) => setOracleForm(p => ({ ...p, chainlink_direct_user_secret: v }))}
                showSecret={showSecrets['cl_secret']}
                onToggle={() => toggleSecret('cl_secret')}
                description={t('accountSettingsFlyout.chainlinkUserSecretHelp')}
              />

              <div className="flex items-start gap-2 p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
                <Activity className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                <p className="text-xs text-muted-foreground">
                  {t('accountSettingsFlyout.oracleInfo')}
                </p>
              </div>

              <Separator className="opacity-30" />

              <div className="flex items-center gap-2">
                <Button size="sm" onClick={handleSaveOracle} disabled={saveMutation.isPending}>
                  <Save className="w-3.5 h-3.5 mr-1.5" />
                  {t('accountSettingsFlyout.save')}
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}
