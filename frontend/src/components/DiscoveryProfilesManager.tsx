import { useState, useEffect, useMemo, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { cn } from '../lib/utils'
import {
  getDiscoveryProfiles,
  updateDiscoveryProfile,
  validateDiscoveryProfile,
  reloadDiscoveryProfile,
  getDiscoveryProfileVersions,
  restoreDiscoveryProfileVersion,
  getSettings,
  updateSettings,
  type DiscoveryProfile,
  type DiscoveryProfileVersion,
  type DiscoveryProfileValidationResult,
} from '../services/api'
import CodeEditor from './CodeEditor'
import StrategyConfigSections from './StrategyConfigSections'
import DiscoveryProfileDocsFlyout from './DiscoveryProfileDocsFlyout'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Sheet, SheetContent, SheetHeader, SheetTitle } from './ui/sheet'
import {
  AlertTriangle,
  BookOpen,
  Check,
  CheckCircle2,
  Clock,
  Code2,
  History,
  Loader2,
  RefreshCw,
  Save,
  Settings2,
  Sliders,
} from 'lucide-react'

// ==================== Constants ====================

const TABS = [
  { key: 'discovery_scoring', labelKey: 'discoveryProfilesManager.tabDiscoveryScoring', icon: Code2 },
  { key: 'pool_selection', labelKey: 'discoveryProfilesManager.tabPoolSelection', icon: Sliders },
  { key: 'signal_settings', labelKey: 'discoveryProfilesManager.tabSignalSettings', icon: Settings2 },
] as const

type TabKey = (typeof TABS)[number]['key']

// ==================== Helpers ====================

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function NumericField({
  label,
  help,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string
  help: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <div>
      <Label className="text-[11px] text-muted-foreground leading-tight">{label}</Label>
      <Input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        className="mt-0.5 text-xs h-7"
      />
      <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">{help}</p>
    </div>
  )
}

// ==================== Signal Settings Tab ====================

function SignalSettingsTab() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const [form, setForm] = useState({
    confluence_limit: 50,
    insider_limit: 40,
    insider_min_confidence: 0.62,
    insider_max_age_minutes: 180,
  })
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    if (!settings?.discovery) return
    const d = settings.discovery
    setForm({
      confluence_limit: d.trader_opps_confluence_limit ?? 50,
      insider_limit: d.trader_opps_insider_limit ?? 40,
      insider_min_confidence: d.trader_opps_insider_min_confidence ?? 0.62,
      insider_max_age_minutes: d.trader_opps_insider_max_age_minutes ?? 180,
    })
    setDirty(false)
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: (f: typeof form) =>
      updateSettings({
        discovery: {
          trader_opps_confluence_limit: Math.round(clamp(f.confluence_limit, 1, 200)),
          trader_opps_insider_limit: Math.round(clamp(f.insider_limit, 1, 500)),
          trader_opps_insider_min_confidence: clamp(f.insider_min_confidence, 0, 1),
          trader_opps_insider_max_age_minutes: Math.round(clamp(f.insider_max_age_minutes, 1, 1440)),
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setDirty(false)
    },
  })

  const update = useCallback(
    (patch: Partial<typeof form>) => {
      setForm((prev) => ({ ...prev, ...patch }))
      setDirty(true)
    },
    [],
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground gap-2 text-xs">
        <Loader2 className="w-4 h-4 animate-spin" />
        {t('discoveryProfilesManager.loadingSettings')}
      </div>
    )
  }

  return (
    <div className="space-y-4 p-4">
      {/* Signal display limits */}
      <div className="space-y-3">
        <div className="flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-orange-400" />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">{t('discoveryProfilesManager.signalDisplaySettings')}</h4>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <NumericField
            label={t('discoveryProfilesManager.confluenceFetchLimitLabel')}
            help={t('discoveryProfilesManager.confluenceFetchLimitHelp')}
            value={form.confluence_limit}
            onChange={(v) => update({ confluence_limit: v })}
            min={1}
            max={200}
            step={1}
          />
          <NumericField
            label={t('discoveryProfilesManager.individualTradeLimitLabel')}
            help={t('discoveryProfilesManager.individualTradeLimitHelp')}
            value={form.insider_limit}
            onChange={(v) => update({ insider_limit: v })}
            min={1}
            max={500}
            step={1}
          />
          <NumericField
            label={t('discoveryProfilesManager.individualTradeMinConfidenceLabel')}
            help={t('discoveryProfilesManager.individualTradeMinConfidenceHelp')}
            value={form.insider_min_confidence}
            onChange={(v) => update({ insider_min_confidence: v })}
            min={0}
            max={1}
            step={0.01}
          />
          <NumericField
            label={t('discoveryProfilesManager.individualTradeMaxAgeLabel')}
            help={t('discoveryProfilesManager.individualTradeMaxAgeHelp')}
            value={form.insider_max_age_minutes}
            onChange={(v) => update({ insider_max_age_minutes: v })}
            min={1}
            max={1440}
            step={1}
          />
        </div>
      </div>

      {/* Save button */}
      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center gap-2">
          {dirty && (
            <Badge variant="outline" className="text-[9px] border-amber-500/30 text-amber-400 bg-amber-500/5">
              {t('discoveryProfilesManager.unsavedChanges')}
            </Badge>
          )}
          {saveMutation.isSuccess && (
            <span className="text-[10px] text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" /> {t('discoveryProfilesManager.saved')}
            </span>
          )}
          {saveMutation.isError && (
            <span className="text-[10px] text-red-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> {t('discoveryProfilesManager.saveFailed')}
            </span>
          )}
        </div>
        <Button
          size="sm"
          onClick={() => saveMutation.mutate(form)}
          disabled={!dirty || saveMutation.isPending}
          className="gap-1.5 text-[10px] h-7 px-3 bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-40"
        >
          {saveMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
          {saveMutation.isPending ? t('discoveryProfilesManager.saving') : t('discoveryProfilesManager.saveSettings')}
        </Button>
      </div>

      {/* Strategy Config Sections */}
      <div className="space-y-2 pt-2 border-t border-border/30">
        <div className="flex items-center gap-1.5 pt-2">
          <Code2 className="w-3.5 h-3.5 text-violet-400" />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">{t('discoveryProfilesManager.strategyConfigurations')}</h4>
        </div>
        <StrategyConfigSections sourceKey="discovery" />
      </div>
    </div>
  )
}

// ==================== Version History Sheet ====================

function VersionHistorySheet({
  open,
  onOpenChange,
  profileId,
  profileName,
  currentVersion,
  onRestore,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  profileId: string
  profileName: string
  currentVersion: number
  onRestore: (version: DiscoveryProfileVersion) => void
}) {
  const { t } = useTranslation()
  const { data: versions, isLoading } = useQuery({
    queryKey: ['discovery-profile-versions', profileId],
    queryFn: () => getDiscoveryProfileVersions(profileId),
    enabled: open && !!profileId,
  })

  const [restoringVersion, setRestoringVersion] = useState<number | null>(null)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[480px] sm:max-w-[480px] p-0 border-l border-border/50">
        <SheetHeader className="px-4 py-3 border-b border-border/30">
          <div className="flex items-center gap-2">
            <History className="w-4 h-4 text-blue-400" />
            <SheetTitle className="text-sm">{t('discoveryProfilesManager.versionHistoryTitle', { name: profileName })}</SheetTitle>
          </div>
        </SheetHeader>

        <div className="overflow-y-auto h-[calc(100vh-60px)]">
          {isLoading ? (
            <div className="flex items-center justify-center py-20 text-muted-foreground gap-2 text-xs">
              <Loader2 className="w-4 h-4 animate-spin" />
              {t('discoveryProfilesManager.loadingVersions')}
            </div>
          ) : !versions?.length ? (
            <div className="flex items-center justify-center py-20 text-muted-foreground text-xs">
              {t('discoveryProfilesManager.noVersionHistory')}
            </div>
          ) : (
            <div className="divide-y divide-border/20">
              {versions.map((v) => (
                <div
                  key={v.id}
                  className={cn(
                    'px-4 py-3 hover:bg-card/50 transition-colors',
                    v.version === currentVersion && 'bg-blue-500/5 border-l-2 border-l-blue-500',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-medium">v{v.version}</span>
                      {v.is_latest && (
                        <Badge variant="outline" className="text-[9px] border-blue-500/30 text-blue-400 bg-blue-500/5 py-0">
                          {t('discoveryProfilesManager.badgeLatest')}
                        </Badge>
                      )}
                      {v.version === currentVersion && (
                        <Badge variant="outline" className="text-[9px] border-emerald-500/30 text-emerald-400 bg-emerald-500/5 py-0">
                          {t('discoveryProfilesManager.badgeCurrent')}
                        </Badge>
                      )}
                    </div>
                    {v.version !== currentVersion && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setRestoringVersion(v.version)
                          onRestore(v)
                        }}
                        disabled={restoringVersion === v.version}
                        className="text-[10px] h-6 px-2"
                      >
                        {restoringVersion === v.version ? (
                          <Loader2 className="w-3 h-3 animate-spin mr-1" />
                        ) : (
                          <RefreshCw className="w-3 h-3 mr-1" />
                        )}
                        {t('discoveryProfilesManager.restore')}
                      </Button>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-[10px] text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {v.created_at ? new Date(v.created_at).toLocaleString() : t('discoveryProfilesManager.unknownDate')}
                    </span>
                    {v.class_name && (
                      <span className="font-mono text-cyan-400/70">{v.class_name}</span>
                    )}
                  </div>
                  {v.reason && (
                    <p className="text-[10px] text-muted-foreground/60 mt-1">{v.reason}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}

// ==================== Profile Editor Tab ====================

function ProfileEditorTab({
  profile,
  onSaved,
}: {
  profile: DiscoveryProfile
  onSaved: () => void
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [editorCode, setEditorCode] = useState(profile.source_code)
  const [validation, setValidation] = useState<DiscoveryProfileValidationResult | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [docsOpen, setDocsOpen] = useState(false)

  // Sync editor when profile changes (e.g. after restore)
  useEffect(() => {
    setEditorCode(profile.source_code)
    setValidation(null)
  }, [profile.source_code, profile.id])

  const isDirty = editorCode !== profile.source_code

  const validateMutation = useMutation({
    mutationFn: (code: string) => validateDiscoveryProfile(code),
    onSuccess: (result) => setValidation(result),
  })

  const saveMutation = useMutation({
    mutationFn: (code: string) =>
      updateDiscoveryProfile(profile.id, { source_code: code }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-profiles'] })
      onSaved()
    },
  })

  const reloadMutation = useMutation({
    mutationFn: () => reloadDiscoveryProfile(profile.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-profiles'] })
    },
  })

  const restoreMutation = useMutation({
    mutationFn: (v: DiscoveryProfileVersion) =>
      restoreDiscoveryProfileVersion(profile.id, v.version, t('discoveryProfilesManager.restoreReason', { version: v.version })),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-profiles'] })
      queryClient.invalidateQueries({ queryKey: ['discovery-profile-versions', profile.id] })
      setHistoryOpen(false)
    },
  })

  const handleSave = useCallback(() => {
    saveMutation.mutate(editorCode)
  }, [editorCode, saveMutation])

  const handleValidate = useCallback(() => {
    validateMutation.mutate(editorCode)
  }, [editorCode, validateMutation])

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/30 bg-card/20 shrink-0">
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold">{profile.name}</span>
              {isDirty && (
                <Badge variant="outline" className="text-[9px] border-amber-500/30 text-amber-400 bg-amber-500/5 py-0">
                  {t('discoveryProfilesManager.badgeUnsaved')}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                {profile.status === 'loaded' ? (
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                ) : profile.status === 'error' ? (
                  <AlertTriangle className="w-3 h-3 text-red-400" />
                ) : (
                  <Clock className="w-3 h-3 text-muted-foreground" />
                )}
                {profile.status === 'loaded'
                  ? t('discoveryProfilesManager.statusLoaded')
                  : profile.status === 'error'
                    ? t('discoveryProfilesManager.statusError')
                    : profile.status}
              </span>
              <span className="text-[10px] text-muted-foreground">
                v{profile.version}
              </span>
              {profile.class_name && (
                <span className="text-[10px] font-mono text-cyan-400/70">
                  {profile.class_name}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="sm"
            onClick={handleValidate}
            disabled={validateMutation.isPending}
            className="text-[10px] h-6 px-2 gap-1"
          >
            {validateMutation.isPending ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <Check className="w-3 h-3" />
            )}
            {t('discoveryProfilesManager.validate')}
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!isDirty || saveMutation.isPending}
            className="text-[10px] h-6 px-2 gap-1 bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-40"
          >
            {saveMutation.isPending ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <Save className="w-3 h-3" />
            )}
            {t('discoveryProfilesManager.save')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => reloadMutation.mutate()}
            disabled={reloadMutation.isPending}
            className="text-[10px] h-6 px-2 gap-1"
            title={t('discoveryProfilesManager.tooltipHotReload')}
          >
            {reloadMutation.isPending ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <RefreshCw className="w-3 h-3" />
            )}
            {t('discoveryProfilesManager.reload')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setHistoryOpen(true)}
            className="text-[10px] h-6 px-2 gap-1"
          >
            <History className="w-3 h-3" />
            {t('discoveryProfilesManager.history')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setDocsOpen(true)}
            className="text-[10px] h-6 px-2 gap-1"
          >
            <BookOpen className="w-3 h-3" />
            {t('discoveryProfilesManager.docs')}
          </Button>
        </div>
      </div>

      {/* Error banner */}
      {profile.status === 'error' && profile.error_message && (
        <div className="px-3 py-2 bg-red-500/10 border-b border-red-500/20 text-[11px] text-red-400 flex items-start gap-2 shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>{profile.error_message}</span>
        </div>
      )}

      {/* Save success/error banner */}
      {saveMutation.isSuccess && (
        <div className="px-3 py-1.5 bg-emerald-500/10 border-b border-emerald-500/20 text-[10px] text-emerald-400 flex items-center gap-1.5 shrink-0">
          <CheckCircle2 className="w-3 h-3" /> {t('discoveryProfilesManager.savedSuccessfully')}
        </div>
      )}
      {saveMutation.isError && (
        <div className="px-3 py-1.5 bg-red-500/10 border-b border-red-500/20 text-[10px] text-red-400 flex items-center gap-1.5 shrink-0">
          <AlertTriangle className="w-3 h-3" /> {t('discoveryProfilesManager.saveFailedWith', {
            message: (saveMutation.error as Error)?.message || t('discoveryProfilesManager.unknownError'),
          })}
        </div>
      )}

      {/* Code editor */}
      <div className="flex-1 min-h-0">
        <CodeEditor
          value={editorCode}
          onChange={setEditorCode}
          language="python"
          minHeight="100%"
          className="h-full"
        />
      </div>

      {/* Validation banner */}
      {validation && (
        <div
          className={cn(
            'px-3 py-2 border-t text-[11px] flex items-center gap-3 shrink-0',
            validation.valid
              ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
              : 'bg-red-500/5 border-red-500/20 text-red-400',
          )}
        >
          <span className="flex items-center gap-1.5">
            {validation.valid ? (
              <CheckCircle2 className="w-3.5 h-3.5" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5" />
            )}
            {validation.valid ? t('discoveryProfilesManager.valid') : t('discoveryProfilesManager.invalid')}
          </span>
          {validation.class_name && (
            <span className="text-muted-foreground">
              {t('discoveryProfilesManager.classLabel')} <span className="font-mono text-cyan-400">{validation.class_name}</span>
            </span>
          )}
          {validation.capabilities && (
            <span className="flex items-center gap-2 text-muted-foreground">
              <span>
                score_wallet{' '}
                {validation.capabilities.has_score_wallet ? (
                  <CheckCircle2 className="w-3 h-3 inline text-emerald-400" />
                ) : (
                  <AlertTriangle className="w-3 h-3 inline text-muted-foreground/50" />
                )}
              </span>
              <span>
                select_pool{' '}
                {validation.capabilities.has_select_pool ? (
                  <CheckCircle2 className="w-3 h-3 inline text-emerald-400" />
                ) : (
                  <AlertTriangle className="w-3 h-3 inline text-muted-foreground/50" />
                )}
              </span>
            </span>
          )}
          {validation.errors.length > 0 && (
            <span className="text-red-400">{validation.errors.join('; ')}</span>
          )}
          {validation.warnings.length > 0 && (
            <span className="text-amber-400">{validation.warnings.join('; ')}</span>
          )}
        </div>
      )}

      {/* Version History Sheet */}
      <VersionHistorySheet
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        profileId={profile.id}
        profileName={profile.name}
        currentVersion={profile.version}
        onRestore={(v) => restoreMutation.mutate(v)}
      />

      {/* Docs Flyout */}
      <DiscoveryProfileDocsFlyout open={docsOpen} onOpenChange={setDocsOpen} />
    </div>
  )
}

// ==================== Main Component ====================

export default function DiscoveryProfilesManager() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<TabKey>('discovery_scoring')

  const {
    data: profiles,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ['discovery-profiles'],
    queryFn: () => getDiscoveryProfiles(),
  })

  const profileMap = useMemo(() => {
    if (!profiles) return {} as Record<string, DiscoveryProfile>
    const map: Record<string, DiscoveryProfile> = {}
    for (const p of profiles) {
      if (p.slug === 'discovery_scoring' || p.slug === 'pool_selection') {
        map[p.slug] = p
      }
    }
    return map
  }, [profiles])

  const activeProfile =
    activeTab === 'discovery_scoring'
      ? profileMap['discovery_scoring']
      : activeTab === 'pool_selection'
        ? profileMap['pool_selection']
        : null

  const handleSaved = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['discovery-profiles'] })
  }, [queryClient])

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex items-center gap-1 px-3 py-2 border-b border-border/30">
          {TABS.map((tab) => (
            <div key={tab.key} className="h-7 w-32 rounded bg-muted/30 animate-pulse" />
          ))}
        </div>
        <div className="flex-1 flex items-center justify-center text-muted-foreground gap-2 text-xs">
          <Loader2 className="w-4 h-4 animate-spin" />
          {t('discoveryProfilesManager.loadingProfiles')}
        </div>
      </div>
    )
  }

  // Error state
  if (isError) {
    return (
      <div className="h-full flex items-center justify-center text-red-400 gap-2 text-xs">
        <AlertTriangle className="w-4 h-4" />
        {t('discoveryProfilesManager.failedToLoad')}
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Tab buttons */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-border/30 shrink-0">
        {TABS.map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.key
          const profile =
            tab.key === 'signal_settings'
              ? null
              : profileMap[tab.key]
          const hasError = profile?.status === 'error'

          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-colors',
                isActive
                  ? 'bg-card text-foreground shadow-sm border border-border/40'
                  : 'text-muted-foreground hover:text-foreground hover:bg-card/50',
              )}
            >
              <Icon className={cn('w-3.5 h-3.5', hasError && 'text-red-400')} />
              {t(tab.labelKey)}
              {hasError && <AlertTriangle className="w-3 h-3 text-red-400" />}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0">
        {activeTab === 'signal_settings' ? (
          <div className="h-full overflow-y-auto">
            <SignalSettingsTab />
          </div>
        ) : activeProfile ? (
          <ProfileEditorTab
            key={activeProfile.id}
            profile={activeProfile}
            onSaved={handleSaved}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground text-xs gap-2">
            <AlertTriangle className="w-4 h-4" />
            {t('discoveryProfilesManager.profileNotFound', { name: activeTab })}
          </div>
        )}
      </div>
    </div>
  )
}
