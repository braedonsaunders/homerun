import { useState, useEffect } from 'react'
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
      setSaveMessage({ type: 'success', text: 'Account settings saved' })
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
            <h3 className="text-sm font-semibold">Account Settings</h3>
          </div>
          <Button
            variant="ghost"
            onClick={onClose}
            className="text-xs h-auto px-2.5 py-1 hover:bg-card"
          >
            <X className="w-3.5 h-3.5 mr-1" />
            Close
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
              Polymarket Account
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
                  return keysSet > 0 ? `${keysSet} key${keysSet !== 1 ? 's' : ''} set` : 'Not configured'
                })()}
              </Badge>
            </h4>
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

              <Separator className="opacity-30" />

              <div className="flex items-center gap-2 flex-wrap">
                <Button size="sm" onClick={handleSavePolymarket} disabled={saveMutation.isPending}>
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
          </Card>

          {/* Kalshi Account */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3 md:col-span-2">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <BarChart3 className="w-3.5 h-3.5 text-indigo-500" />
              Kalshi Account
              <Badge variant="outline" className={cn(
                "ml-auto text-[10px] px-2 py-0.5 border-0 shrink-0",
                (!!settings?.kalshi?.email || !!settings?.kalshi?.api_key) ? 'text-emerald-400 bg-emerald-500/10' : 'text-muted-foreground bg-muted'
              )}>
                {(!!settings?.kalshi?.email || !!settings?.kalshi?.api_key) ? 'Configured' : 'Not configured'}
              </Badge>
            </h4>
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

              <div className="flex items-start gap-2 p-3 bg-indigo-500/5 border border-indigo-500/20 rounded-lg">
                <Activity className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
                <p className="text-xs text-muted-foreground">
                  Kalshi credentials enable cross-platform arbitrage trading. The scanner will automatically detect price differences between Polymarket and Kalshi for the same events. You can use either email/password or an API key for authentication.
                </p>
              </div>

              <Separator className="opacity-30" />

              <div className="flex items-center gap-2">
                <Button size="sm" onClick={handleSaveKalshi} disabled={saveMutation.isPending}>
                  <Save className="w-3.5 h-3.5 mr-1.5" />
                  Save
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}
