import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertCircle,
  CheckCircle,
  Globe,
  Save,
  Settings2,
  Shield,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import {
  getSettings,
  updateSettings,
  type WorldIntelligenceSettings,
} from '../services/api'

const DEFAULTS: WorldIntelligenceSettings = {
  enabled: true,
  interval_seconds: 300,
  emit_trade_signals: false,
  acled_api_key: '',
  acled_email: '',
  opensky_username: '',
  opensky_password: '',
  aisstream_api_key: '',
  cloudflare_radar_token: '',
  ais_enabled: true,
  ais_sample_seconds: 10,
  ais_max_messages: 250,
  airplanes_live_enabled: true,
  airplanes_live_timeout_seconds: 20,
  airplanes_live_max_records: 1500,
  military_dedupe_radius_km: 45,
  military_enabled: true,
  country_reference_sync_enabled: true,
  country_reference_sync_hours: 24,
  country_reference_request_timeout_seconds: 20,
  ucdp_sync_enabled: true,
  ucdp_sync_hours: 24,
  ucdp_lookback_years: 8,
  ucdp_max_pages: 100,
  ucdp_request_timeout_seconds: 25,
  mid_sync_enabled: true,
  mid_sync_hours: 168,
  mid_request_timeout_seconds: 20,
  trade_dependency_sync_enabled: true,
  trade_dependency_sync_hours: 24,
  trade_dependency_request_timeout_seconds: 20,
  trade_dependency_wb_per_page: 5000,
  trade_dependency_wb_max_pages: 50,
  trade_dependency_base_divisor: 120,
  trade_dependency_min_factor: 0.5,
  trade_dependency_max_factor: 1.5,
  chokepoints_enabled: true,
  chokepoints_refresh_seconds: 1800,
  chokepoints_request_timeout_seconds: 20,
  chokepoints_max_daily_rows: 500,
  chokepoints_db_sync_enabled: true,
  chokepoints_db_sync_hours: 6,
  convergence_min_types: 2,
  anomaly_threshold: 1.8,
  anomaly_min_baseline_points: 3,
  instability_signal_min: 15,
  instability_critical: 60,
  tension_critical: 70,
  gdelt_query_delay_seconds: 5,
  gdelt_max_concurrency: 1,
  gdelt_news_enabled: true,
  gdelt_news_timespan_hours: 6,
  gdelt_news_max_records: 40,
  gdelt_news_request_timeout_seconds: 20,
  gdelt_news_cache_seconds: 300,
  gdelt_news_query_delay_seconds: 5,
  gdelt_news_sync_enabled: true,
  gdelt_news_sync_hours: 1,
  acled_rate_limit_per_min: 5,
  acled_auth_rate_limit_per_min: 12,
  acled_cb_max_failures: 8,
  acled_cb_cooldown_seconds: 180,
  opensky_cb_max_failures: 6,
  opensky_cb_cooldown_seconds: 120,
  usgs_enabled: true,
  usgs_min_magnitude: 4.5,
}

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  min,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  step?: number
  min?: number
}) {
  return (
    <div>
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <Input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="h-7 mt-1 text-xs"
      />
    </div>
  )
}

function SecretField({
  label,
  value,
  onChange,
  visible,
  onToggle,
  placeholder,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  visible: boolean
  onToggle: () => void
  placeholder?: string
}) {
  return (
    <div>
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <div className="relative mt-1">
        <Input
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="h-7 pr-16 text-xs font-mono"
        />
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="absolute right-1 top-1 h-5 px-2 text-[10px]"
          onClick={onToggle}
        >
          {visible ? 'Hide' : 'Show'}
        </Button>
      </div>
    </div>
  )
}

function ToggleRow({
  label,
  checked,
  onCheckedChange,
}: {
  label: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border/40 px-2 py-1.5">
      <span className="text-xs">{label}</span>
      <Switch checked={checked} onCheckedChange={onCheckedChange} className="scale-75" />
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="bg-card/40 border-border/40 rounded-xl p-3 space-y-2.5">
      <h4 className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground">{title}</h4>
      {children}
    </Card>
  )
}

export default function WorldIntelligenceSettingsFlyout({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const [form, setForm] = useState<WorldIntelligenceSettings>(DEFAULTS)
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
    enabled: isOpen,
  })

  useEffect(() => {
    if (!settings?.world_intelligence) return
    setForm({
      ...DEFAULTS,
      ...settings.world_intelligence,
      acled_api_key: '',
      opensky_password: '',
      aisstream_api_key: '',
      cloudflare_radar_token: '',
    })
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-status'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-summary'] })
      queryClient.invalidateQueries({ queryKey: ['world-intelligence-sources'] })
      setSaveMessage({ type: 'success', text: 'World intelligence settings saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error?.message || 'Failed to save settings' })
      setTimeout(() => setSaveMessage(null), 5000)
    },
  })

  const set = <K extends keyof WorldIntelligenceSettings>(key: K, value: WorldIntelligenceSettings[K]) =>
    setForm((prev) => ({ ...prev, [key]: value }))

  const handleSave = () => {
    const payload: Partial<WorldIntelligenceSettings> = { ...form }
    if (!form.acled_api_key) delete payload.acled_api_key
    if (!form.opensky_password) delete payload.opensky_password
    if (!form.aisstream_api_key) delete payload.aisstream_api_key
    if (!form.cloudflare_radar_token) delete payload.cloudflare_radar_token
    saveMutation.mutate({ world_intelligence: payload })
  }

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 bg-background/80 z-40 transition-opacity" onClick={onClose} />
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-3xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-blue-500" />
            <h3 className="text-sm font-semibold">World Intelligence Settings</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending} className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white">
              <Save className="w-3 h-3" /> {saveMutation.isPending ? 'Saving...' : 'Save'}
            </Button>
            <Button variant="ghost" onClick={onClose} className="text-xs h-auto px-2.5 py-1 hover:bg-card">
              <X className="w-3.5 h-3.5 mr-1" />
              Close
            </Button>
          </div>
        </div>

        {saveMessage && (
          <div
            className={cn(
              'fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300',
              saveMessage.type === 'success'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-red-500/10 text-red-400 border-red-500/20'
            )}
          >
            {saveMessage.type === 'success' ? (
              <CheckCircle className="w-4 h-4 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 shrink-0" />
            )}
            {saveMessage.text}
          </div>
        )}

        <div className="p-3 space-y-3 pb-6">
          <Section title="Pipeline">
            <ToggleRow label="Enable World Intelligence" checked={form.enabled} onCheckedChange={(v) => set('enabled', v)} />
            <ToggleRow label="Emit Trade Signals to Orchestrator" checked={form.emit_trade_signals} onCheckedChange={(v) => set('emit_trade_signals', v)} />
            <NumberField label="Collection Interval (seconds)" value={form.interval_seconds} min={30} onChange={(v) => set('interval_seconds', Math.max(30, Math.round(v)))} />
          </Section>

          <Section title="Credentials">
            <SecretField
              label="ACLED API Key"
              value={form.acled_api_key || ''}
              onChange={(v) => set('acled_api_key', v)}
              visible={!!showSecrets.acled_api_key}
              onToggle={() => setShowSecrets((p) => ({ ...p, acled_api_key: !p.acled_api_key }))}
            />
            <div>
              <Label className="text-[11px] text-muted-foreground">ACLED Email</Label>
              <Input
                value={form.acled_email || ''}
                onChange={(e) => set('acled_email', e.target.value)}
                className="h-7 mt-1 text-xs"
              />
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">OpenSky Username</Label>
              <Input
                value={form.opensky_username || ''}
                onChange={(e) => set('opensky_username', e.target.value)}
                className="h-7 mt-1 text-xs"
              />
            </div>
            <SecretField
              label="OpenSky Password"
              value={form.opensky_password || ''}
              onChange={(v) => set('opensky_password', v)}
              visible={!!showSecrets.opensky_password}
              onToggle={() => setShowSecrets((p) => ({ ...p, opensky_password: !p.opensky_password }))}
            />
            <SecretField
              label="AISStream API Key"
              value={form.aisstream_api_key || ''}
              onChange={(v) => set('aisstream_api_key', v)}
              visible={!!showSecrets.aisstream_api_key}
              onToggle={() => setShowSecrets((p) => ({ ...p, aisstream_api_key: !p.aisstream_api_key }))}
            />
            <SecretField
              label="Cloudflare Radar Token"
              value={form.cloudflare_radar_token || ''}
              onChange={(v) => set('cloudflare_radar_token', v)}
              visible={!!showSecrets.cloudflare_radar_token}
              onToggle={() => setShowSecrets((p) => ({ ...p, cloudflare_radar_token: !p.cloudflare_radar_token }))}
            />
          </Section>

          <Section title="Source Toggles">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <ToggleRow label="Military" checked={form.military_enabled} onCheckedChange={(v) => set('military_enabled', v)} />
              <ToggleRow label="USGS" checked={form.usgs_enabled} onCheckedChange={(v) => set('usgs_enabled', v)} />
              <ToggleRow label="AIS Stream" checked={form.ais_enabled} onCheckedChange={(v) => set('ais_enabled', v)} />
              <ToggleRow label="Airplanes.live" checked={form.airplanes_live_enabled} onCheckedChange={(v) => set('airplanes_live_enabled', v)} />
              <ToggleRow label="Chokepoints" checked={form.chokepoints_enabled} onCheckedChange={(v) => set('chokepoints_enabled', v)} />
              <ToggleRow label="GDELT News" checked={form.gdelt_news_enabled} onCheckedChange={(v) => set('gdelt_news_enabled', v)} />
            </div>
          </Section>

          <Section title="Sync Sources">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <ToggleRow label="Country Reference Sync" checked={form.country_reference_sync_enabled} onCheckedChange={(v) => set('country_reference_sync_enabled', v)} />
              <ToggleRow label="UCDP Sync" checked={form.ucdp_sync_enabled} onCheckedChange={(v) => set('ucdp_sync_enabled', v)} />
              <ToggleRow label="MID Sync" checked={form.mid_sync_enabled} onCheckedChange={(v) => set('mid_sync_enabled', v)} />
              <ToggleRow label="Trade Dependency Sync" checked={form.trade_dependency_sync_enabled} onCheckedChange={(v) => set('trade_dependency_sync_enabled', v)} />
              <ToggleRow label="Chokepoints DB Sync" checked={form.chokepoints_db_sync_enabled} onCheckedChange={(v) => set('chokepoints_db_sync_enabled', v)} />
              <ToggleRow label="GDELT News Sync" checked={form.gdelt_news_sync_enabled} onCheckedChange={(v) => set('gdelt_news_sync_enabled', v)} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <NumberField label="Country Sync (h)" value={form.country_reference_sync_hours} min={1} onChange={(v) => set('country_reference_sync_hours', Math.max(1, Math.round(v)))} />
              <NumberField label="UCDP Sync (h)" value={form.ucdp_sync_hours} min={1} onChange={(v) => set('ucdp_sync_hours', Math.max(1, Math.round(v)))} />
              <NumberField label="MID Sync (h)" value={form.mid_sync_hours} min={1} onChange={(v) => set('mid_sync_hours', Math.max(1, Math.round(v)))} />
              <NumberField label="Trade Sync (h)" value={form.trade_dependency_sync_hours} min={1} onChange={(v) => set('trade_dependency_sync_hours', Math.max(1, Math.round(v)))} />
              <NumberField label="Chokepoints DB (h)" value={form.chokepoints_db_sync_hours} min={1} onChange={(v) => set('chokepoints_db_sync_hours', Math.max(1, Math.round(v)))} />
              <NumberField label="GDELT Sync (h)" value={form.gdelt_news_sync_hours} min={1} onChange={(v) => set('gdelt_news_sync_hours', Math.max(1, Math.round(v)))} />
            </div>
          </Section>

          <Section title="Collection Controls">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <NumberField label="AIS Sample (s)" value={form.ais_sample_seconds} min={3} onChange={(v) => set('ais_sample_seconds', Math.max(3, Math.round(v)))} />
              <NumberField label="AIS Max Messages" value={form.ais_max_messages} min={10} onChange={(v) => set('ais_max_messages', Math.max(10, Math.round(v)))} />
              <NumberField label="Airplanes Timeout (s)" value={form.airplanes_live_timeout_seconds} min={5} onChange={(v) => set('airplanes_live_timeout_seconds', Math.max(5, v))} />
              <NumberField label="Airplanes Max Rows" value={form.airplanes_live_max_records} min={50} onChange={(v) => set('airplanes_live_max_records', Math.max(50, Math.round(v)))} />
              <NumberField label="Military Dedupe Radius (km)" value={form.military_dedupe_radius_km} min={1} onChange={(v) => set('military_dedupe_radius_km', Math.max(1, v))} />
              <NumberField label="USGS Min Magnitude" value={form.usgs_min_magnitude} step={0.1} min={0} onChange={(v) => set('usgs_min_magnitude', Math.max(0, v))} />
              <NumberField label="Chokepoints Refresh (s)" value={form.chokepoints_refresh_seconds} min={60} onChange={(v) => set('chokepoints_refresh_seconds', Math.max(60, Math.round(v)))} />
              <NumberField label="Chokepoints Max Daily Rows" value={form.chokepoints_max_daily_rows} min={100} onChange={(v) => set('chokepoints_max_daily_rows', Math.max(100, Math.round(v)))} />
              <NumberField label="GDELT News Max Records" value={form.gdelt_news_max_records} min={10} onChange={(v) => set('gdelt_news_max_records', Math.max(10, Math.round(v)))} />
            </div>
          </Section>

          <Section title="Thresholds And Limits">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <NumberField label="Convergence Min Types" value={form.convergence_min_types} min={2} onChange={(v) => set('convergence_min_types', Math.max(2, Math.round(v)))} />
              <NumberField label="Anomaly Threshold (z)" value={form.anomaly_threshold} step={0.1} min={0.1} onChange={(v) => set('anomaly_threshold', Math.max(0.1, v))} />
              <NumberField label="Anomaly Baseline Min Points" value={form.anomaly_min_baseline_points} min={2} onChange={(v) => set('anomaly_min_baseline_points', Math.max(2, Math.round(v)))} />
              <NumberField label="Instability Signal Min" value={form.instability_signal_min} step={0.5} min={0} onChange={(v) => set('instability_signal_min', Math.max(0, v))} />
              <NumberField label="Instability Critical" value={form.instability_critical} step={0.5} min={0} onChange={(v) => set('instability_critical', Math.max(0, v))} />
              <NumberField label="Tension Critical" value={form.tension_critical} step={0.5} min={0} onChange={(v) => set('tension_critical', Math.max(0, v))} />
              <NumberField label="GDELT Query Delay (s)" value={form.gdelt_query_delay_seconds} step={0.1} min={0} onChange={(v) => set('gdelt_query_delay_seconds', Math.max(0, v))} />
              <NumberField label="GDELT Max Concurrency" value={form.gdelt_max_concurrency} min={1} onChange={(v) => set('gdelt_max_concurrency', Math.max(1, Math.round(v)))} />
              <NumberField label="GDELT Timespan (h)" value={form.gdelt_news_timespan_hours} min={1} onChange={(v) => set('gdelt_news_timespan_hours', Math.max(1, Math.round(v)))} />
              <NumberField label="ACLED Rate Limit / Min" value={form.acled_rate_limit_per_min} min={1} onChange={(v) => set('acled_rate_limit_per_min', Math.max(1, Math.round(v)))} />
              <NumberField label="ACLED Auth Rate / Min" value={form.acled_auth_rate_limit_per_min} min={1} onChange={(v) => set('acled_auth_rate_limit_per_min', Math.max(1, Math.round(v)))} />
              <NumberField label="ACLED CB Max Failures" value={form.acled_cb_max_failures} min={1} onChange={(v) => set('acled_cb_max_failures', Math.max(1, Math.round(v)))} />
              <NumberField label="ACLED CB Cooldown (s)" value={form.acled_cb_cooldown_seconds} min={30} onChange={(v) => set('acled_cb_cooldown_seconds', Math.max(30, v))} />
              <NumberField label="OpenSky CB Max Failures" value={form.opensky_cb_max_failures} min={1} onChange={(v) => set('opensky_cb_max_failures', Math.max(1, Math.round(v)))} />
              <NumberField label="OpenSky CB Cooldown (s)" value={form.opensky_cb_cooldown_seconds} min={30} onChange={(v) => set('opensky_cb_cooldown_seconds', Math.max(30, v))} />
            </div>
          </Section>

          <Section title="Request Timeouts">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <NumberField label="Country Ref Timeout (s)" value={form.country_reference_request_timeout_seconds} min={1} onChange={(v) => set('country_reference_request_timeout_seconds', Math.max(1, v))} />
              <NumberField label="UCDP Timeout (s)" value={form.ucdp_request_timeout_seconds} min={1} onChange={(v) => set('ucdp_request_timeout_seconds', Math.max(1, v))} />
              <NumberField label="MID Timeout (s)" value={form.mid_request_timeout_seconds} min={1} onChange={(v) => set('mid_request_timeout_seconds', Math.max(1, v))} />
              <NumberField label="Trade Timeout (s)" value={form.trade_dependency_request_timeout_seconds} min={1} onChange={(v) => set('trade_dependency_request_timeout_seconds', Math.max(1, v))} />
              <NumberField label="Chokepoints Timeout (s)" value={form.chokepoints_request_timeout_seconds} min={1} onChange={(v) => set('chokepoints_request_timeout_seconds', Math.max(1, v))} />
              <NumberField label="GDELT Timeout (s)" value={form.gdelt_news_request_timeout_seconds} min={1} onChange={(v) => set('gdelt_news_request_timeout_seconds', Math.max(1, v))} />
              <NumberField label="GDELT Cache (s)" value={form.gdelt_news_cache_seconds} min={0} onChange={(v) => set('gdelt_news_cache_seconds', Math.max(0, Math.round(v)))} />
              <NumberField label="GDELT News Delay (s)" value={form.gdelt_news_query_delay_seconds} min={0} step={0.1} onChange={(v) => set('gdelt_news_query_delay_seconds', Math.max(0, v))} />
            </div>
          </Section>

          <Section title="Reference Source Fetch Limits">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <NumberField label="UCDP Lookback (years)" value={form.ucdp_lookback_years} min={1} onChange={(v) => set('ucdp_lookback_years', Math.max(1, Math.round(v)))} />
              <NumberField label="UCDP Max Pages" value={form.ucdp_max_pages} min={1} onChange={(v) => set('ucdp_max_pages', Math.max(1, Math.round(v)))} />
              <NumberField label="Trade WB Per Page" value={form.trade_dependency_wb_per_page} min={100} onChange={(v) => set('trade_dependency_wb_per_page', Math.max(100, Math.round(v)))} />
              <NumberField label="Trade WB Max Pages" value={form.trade_dependency_wb_max_pages} min={1} onChange={(v) => set('trade_dependency_wb_max_pages', Math.max(1, Math.round(v)))} />
              <NumberField label="Trade Base Divisor" value={form.trade_dependency_base_divisor} step={1} min={1} onChange={(v) => set('trade_dependency_base_divisor', Math.max(1, v))} />
              <NumberField label="Trade Min Factor" value={form.trade_dependency_min_factor} step={0.1} min={0} onChange={(v) => set('trade_dependency_min_factor', Math.max(0, v))} />
              <NumberField label="Trade Max Factor" value={form.trade_dependency_max_factor} step={0.1} min={0} onChange={(v) => set('trade_dependency_max_factor', Math.max(0, v))} />
            </div>
          </Section>

          <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-3 py-2 text-[11px] text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <Globe className="w-3 h-3 text-blue-400" />
              Runtime note: some source clients snapshot settings at process start.
            </div>
            <div className="flex items-center gap-1.5 mt-1">
              <Shield className="w-3 h-3 text-blue-400" />
              If behavior does not update immediately, restart the world intelligence worker.
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
