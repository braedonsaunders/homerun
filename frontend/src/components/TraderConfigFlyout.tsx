import { memo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAtom, useAtomValue } from 'jotai'
import { useTranslation } from 'react-i18next'
import { AlertTriangle, Clock3, Sparkles, Zap } from 'lucide-react'
import { getWorkersStatus, type Trader, type TraderLatencyClass, type WorkerStatus } from '../services/apiTraders'
import { draftDescriptionAtom, draftNameAtom, draftTradingScheduleAtom } from '../store/atoms'
import { AtomInput } from './AtomInput'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { Switch } from './ui/switch'
import {
  FlyoutSection,
  TRADING_SCHEDULE_DAYS,
  TRADING_SCHEDULE_DAY_LABEL_KEYS,
  TRADING_SCHEDULE_WEEKDAYS,
  TRADING_SCHEDULE_WEEKENDS,
  type StrategyCatalogOption,
  type StrategyOptionDetail,
  type TradingScheduleDay,
  type TradingScheduleDraft,
  normalizeStrategyKey,
  normalizeTradingScheduleDays,
  normalizeTradingScheduleDraft,
  normalizeVersionList,
} from './tradingPanelFlyoutShared'

type DeleteAction = 'block' | 'disable' | 'force_delete' | 'transfer_delete'

type Mutation<TInput = void> = {
  isPending: boolean
  mutate: (input: TInput) => void
}

export type TraderConfigFlyoutProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  mode: 'create' | 'edit'
  busy: boolean
  saveError: string | null

  // Bot Profile
  // draftName / draftDescription are NOT props — the inputs subscribe to
  // jotai atoms (draftNameAtom / draftDescriptionAtom) directly so typing
  // doesn't cascade into a parent re-render.
  draftMode: 'shadow' | 'live'
  draftLatencyClass: TraderLatencyClass
  setDraftLatencyClass: (value: TraderLatencyClass) => void

  // Copy Settings
  draftCopyFromMode: 'shadow' | 'live'
  setDraftCopyFromMode: (value: 'shadow' | 'live') => void
  draftCopyFromTraderId: string
  copySourceTraders: Trader[]
  applyCreateCopyFromSelection: (value: string) => void

  // Strategy
  draftStrategyKey: string
  setDraftStrategy: (value: string) => void
  setDraftStrategyVersionFromValue: (value: string) => void
  allStrategyOptions: StrategyCatalogOption[]
  draftStrategyOption: StrategyCatalogOption | null
  effectiveDraftSourceKey: string
  effectiveDraftStrategyDetail: StrategyOptionDetail | null
  effectiveDraftStrategyVersion: number | null

  // Trading Schedule lives in draftTradingScheduleAtom — no props needed.

  // Delete (edit mode only)
  selectedTrader: Trader | null
  selectedTraderDeleteExposureSummary: string
  selectedTraderHasLiveDeleteExposure: boolean
  selectedTraderHasAnyDeleteExposure: boolean
  selectedTraderOpenLivePositions: number
  selectedTraderOpenShadowPositions: number
  selectedTraderOpenLiveOrders: number
  selectedTraderOpenShadowOrders: number
  traders: Trader[]
  deleteAction: DeleteAction
  setDeleteAction: (value: DeleteAction) => void
  deleteForceConfirm: boolean
  setDeleteForceConfirm: (value: boolean) => void
  deleteTransferTargetId: string | null
  setDeleteTransferTargetId: (value: string | null) => void
  deleteTraderMutation: Mutation<{
    traderId: string
    action: DeleteAction
    transferToTraderId?: string
  }>

  // Save / Create
  createTraderMutation: Mutation<void>
  saveTraderMutation: Mutation<string>
}

function TraderConfigFlyoutImpl(props: TraderConfigFlyoutProps) {
  const {
    open,
    onOpenChange,
    mode,
    busy,
    saveError,
    draftMode,
    draftLatencyClass,
    setDraftLatencyClass,
    draftCopyFromMode,
    setDraftCopyFromMode,
    draftCopyFromTraderId,
    copySourceTraders,
    applyCreateCopyFromSelection,
    draftStrategyKey,
    setDraftStrategy,
    setDraftStrategyVersionFromValue,
    allStrategyOptions,
    draftStrategyOption,
    effectiveDraftSourceKey,
    effectiveDraftStrategyDetail,
    effectiveDraftStrategyVersion,
    selectedTrader,
    selectedTraderDeleteExposureSummary,
    selectedTraderHasLiveDeleteExposure,
    selectedTraderHasAnyDeleteExposure,
    selectedTraderOpenLivePositions,
    selectedTraderOpenShadowPositions,
    selectedTraderOpenLiveOrders,
    selectedTraderOpenShadowOrders,
    traders,
    deleteAction,
    setDeleteAction,
    deleteForceConfirm,
    setDeleteForceConfirm,
    deleteTransferTargetId,
    setDeleteTransferTargetId,
    deleteTraderMutation,
    createTraderMutation,
    saveTraderMutation,
  } = props

  const { t } = useTranslation()
  const detail = effectiveDraftStrategyDetail
  const sourceKey = effectiveDraftSourceKey
  const sourceLabel = draftStrategyOption?.sourceLabel || sourceKey.toUpperCase()
  // Off-by-default subsystems a strategy can depend on. If the selected
  // strategy's source maps to a disabled subsystem, warn the user to enable it
  // (mirrors Settings > Maintenance > Background Subsystems).
  const workersStatusQuery = useQuery({
    queryKey: ['workers-status'],
    queryFn: getWorkersStatus,
    enabled: open,
    staleTime: 15_000,
  })
  const sourceDependency = ({
    news: { worker: 'news', label: 'News ingestion' },
    weather: { worker: 'weather', label: 'Weather ingestion' },
    traders: { worker: 'discovery', label: 'Wallet discovery' },
  } as Record<string, { worker: string; label: string }>)[(sourceKey || '').toLowerCase()]
  const sourceDependencyEnabled = (() => {
    if (!sourceDependency) return true
    const w = workersStatusQuery.data?.workers?.find((x: WorkerStatus) => x.worker_name === sourceDependency.worker)
    const ctrl = w?.control as Record<string, any> | undefined
    return ctrl?.is_enabled ?? true
  })()
  const showSubsystemDisabledWarning = Boolean(sourceDependency) && !sourceDependencyEnabled
  const latestVersion = detail?.latestVersion ?? detail?.version ?? null
  const selectedVersion = effectiveDraftStrategyVersion
  const selectedVersionToken = selectedVersion == null ? 'latest' : `v${selectedVersion}`
  const availableVersions = (() => {
    const rows = normalizeVersionList(detail?.versions || [])
    if (latestVersion != null && !rows.includes(latestVersion)) rows.unshift(latestVersion)
    if (selectedVersion != null && !rows.includes(selectedVersion)) rows.unshift(selectedVersion)
    return rows
  })()
  // Subscribe here (only the flyout re-renders on draftName changes) so the
  // Save button's "name is required" check stays reactive.
  const draftNameValue = useAtomValue(draftNameAtom)

  // Schedule editing is fully owned by the flyout via the schedule atom;
  // typing in time/date pickers, toggling days, etc. only re-renders the
  // flyout — never bubbles into TradingPanel.
  const [tradingScheduleDraft, setTradingSchedule] = useAtom(draftTradingScheduleAtom)
  const setDraftTradingSchedule = (
    patch:
      | Partial<TradingScheduleDraft>
      | ((current: TradingScheduleDraft) => Partial<TradingScheduleDraft>)
  ) => {
    const current = tradingScheduleDraft
    const resolvedPatch = typeof patch === 'function' ? patch(current) : patch
    setTradingSchedule(normalizeTradingScheduleDraft({
      enabled: resolvedPatch.enabled ?? current.enabled,
      days: resolvedPatch.days ?? current.days,
      start_time: resolvedPatch.startTimeUtc ?? current.startTimeUtc,
      end_time: resolvedPatch.endTimeUtc ?? current.endTimeUtc,
      start_date: resolvedPatch.startDateUtc ?? current.startDateUtc,
      end_date: resolvedPatch.endDateUtc ?? current.endDateUtc,
      end_at: resolvedPatch.endAtUtc ?? current.endAtUtc,
    }))
  }
  const toggleTradingScheduleDay = (day: TradingScheduleDay) => {
    setDraftTradingSchedule((current) => {
      const exists = current.days.includes(day)
      const nextDays = exists
        ? current.days.filter((item) => item !== day)
        : [...current.days, day]
      return { days: normalizeTradingScheduleDays(nextDays) }
    })
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-3xl p-0">
        <div className="h-full min-h-0 flex flex-col">
          <div className="border-b border-border px-4 py-3">
            <SheetHeader className="space-y-1 text-left">
              <SheetTitle className="text-base">
                {mode === 'create' ? t('traderConfigFlyout.titleCreate') : t('traderConfigFlyout.titleEdit')}
              </SheetTitle>
              <SheetDescription>
                {mode === 'create'
                  ? t('traderConfigFlyout.descriptionCreate')
                  : t('traderConfigFlyout.descriptionEdit')}
              </SheetDescription>
            </SheetHeader>
          </div>

          <ScrollArea className="flex-1 min-h-0 px-4 py-3">
            <div className="space-y-3 pb-2">
              <FlyoutSection
                title={t('traderConfigFlyout.botProfileTitle')}
                icon={Sparkles}
                subtitle={t('traderConfigFlyout.botProfileSubtitle')}
              >
                <div className="rounded-md border border-border/60 bg-muted/15 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-[11px] uppercase tracking-wider text-muted-foreground">{t('traderConfigFlyout.botMode')}</p>
                    <Badge className="h-5 px-1.5 text-[10px]" variant={draftMode === 'live' ? 'destructive' : 'outline'}>
                      {draftMode.toUpperCase()}
                    </Badge>
                  </div>
                  <p className="mt-1 text-[10px] text-muted-foreground/75">
                    {draftMode === 'live'
                      ? t('traderConfigFlyout.botModeScopeLive')
                      : t('traderConfigFlyout.botModeScopeSandbox')}
                  </p>
                </div>

                <div>
                  <Label>{t('traderConfigFlyout.name')}</Label>
                  <AtomInput atom={draftNameAtom} className="mt-1" />
                </div>

                <div>
                  <Label>{t('traderConfigFlyout.description')}</Label>
                  <AtomInput atom={draftDescriptionAtom} className="mt-1" />
                </div>

                <div>
                  <Label>{t('traderConfigFlyout.latencyClass')}</Label>
                  <Select
                    value={draftLatencyClass}
                    onValueChange={(value: string) => setDraftLatencyClass((value === 'fast' || value === 'slow') ? (value as TraderLatencyClass) : 'normal')}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fast">{t('traderConfigFlyout.latencyFast')}</SelectItem>
                      <SelectItem value="normal">{t('traderConfigFlyout.latencyNormal')}</SelectItem>
                      <SelectItem value="slow">{t('traderConfigFlyout.latencySlow')}</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="mt-1 text-[10px] text-muted-foreground/75 leading-tight">
                    {t('traderConfigFlyout.latencyHelp')}
                  </p>
                </div>

                {mode === 'create' ? (
                  <div>
                    <Label>{t('traderConfigFlyout.copyFromLabel')}</Label>
                    <div className="mt-1 flex items-center gap-1">
                      <Button
                        type="button"
                        size="sm"
                        variant={draftCopyFromMode === 'shadow' ? 'default' : 'outline'}
                        className="h-6 px-2 text-[10px]"
                        onClick={() => setDraftCopyFromMode('shadow')}
                      >
                        {t('traderConfigFlyout.sandboxBots')}
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant={draftCopyFromMode === 'live' ? 'default' : 'outline'}
                        className="h-6 px-2 text-[10px]"
                        onClick={() => setDraftCopyFromMode('live')}
                      >
                        {t('traderConfigFlyout.liveBots')}
                      </Button>
                    </div>
                    <Select
                      value={draftCopyFromTraderId || '__none__'}
                      onValueChange={applyCreateCopyFromSelection}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder={t('traderConfigFlyout.startFromScratch')} />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none__">{t('traderConfigFlyout.startFromScratch')}</SelectItem>
                        {copySourceTraders.map((trader) => (
                          <SelectItem key={trader.id} value={trader.id}>
                            {trader.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {copySourceTraders.length === 0 ? (
                      <p className="mt-1 text-[10px] text-muted-foreground/75 leading-tight">
                        {draftCopyFromMode === 'live'
                          ? t('traderConfigFlyout.noBotsAvailableLive')
                          : t('traderConfigFlyout.noBotsAvailableSandbox')}
                      </p>
                    ) : null}
                    <p className="mt-1 text-[10px] text-muted-foreground/75 leading-tight">
                      {t('traderConfigFlyout.copyFromHelp')}
                    </p>
                  </div>
                ) : null}
              </FlyoutSection>

              <FlyoutSection
                title={t('traderConfigFlyout.strategyTitle')}
                icon={Zap}
                subtitle={t('traderConfigFlyout.strategySubtitle')}
              >
                <div>
                  <Label>{t('traderConfigFlyout.strategy')}</Label>
                  <Select
                    value={normalizeStrategyKey(draftStrategyKey)}
                    onValueChange={setDraftStrategy}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder={t('traderConfigFlyout.chooseStrategy')} />
                    </SelectTrigger>
                    <SelectContent>
                      {allStrategyOptions.map((option) => (
                        <SelectItem key={option.key} value={option.key}>
                          <span>{option.label}</span>
                          <span className="ml-2 text-muted-foreground/70 text-[10px]">
                            {option.sourceLabel}
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-muted-foreground/80">
                    <span>{t('traderConfigFlyout.sourceLabel')}</span>
                    <Badge
                      variant="outline"
                      className="h-4 px-1.5 text-[9px] font-mono border-emerald-500/30 text-emerald-300 bg-emerald-500/10"
                    >
                      {sourceLabel || 'AUTO'}
                    </Badge>
                    <span className="text-muted-foreground/60">{t('traderConfigFlyout.autoDerived')}</span>
                  </div>
                  {showSubsystemDisabledWarning && sourceDependency && (
                    <div className="mt-2 flex items-start gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1.5">
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-amber-500" />
                      <p className="text-[10px] leading-snug text-amber-300">
                        {sourceDependency.label} is off, so this strategy won't receive signals. Enable it in
                        Settings → Maintenance → Background Subsystems.
                      </p>
                    </div>
                  )}
                </div>

                {detail ? (
                  <div className="mt-3">
                    <Label>{t('traderConfigFlyout.version')}</Label>
                    <div className="mt-1 flex min-w-0 items-center gap-1.5">
                      <Badge
                        variant="outline"
                        className="h-5 min-w-0 flex-1 truncate px-1.5 text-[10px] font-mono border-emerald-500/30 text-emerald-300 bg-emerald-500/10"
                      >
                        {latestVersion != null ? t('traderConfigFlyout.latestVersionWithNum', { n: latestVersion }) : t('traderConfigFlyout.latest')}
                      </Badge>
                      <Select
                        value={selectedVersionToken}
                        onValueChange={setDraftStrategyVersionFromValue}
                      >
                        <SelectTrigger className="h-7 w-[160px] shrink-0 text-[10px] font-mono">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="latest">
                            {latestVersion != null ? t('traderConfigFlyout.latestWithVersion', { n: latestVersion }) : t('traderConfigFlyout.latestLower')}
                          </SelectItem>
                          {availableVersions.map((version) => (
                            <SelectItem key={`v${version}`} value={`v${version}`}>
                              {`v${version}`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                ) : null}
              </FlyoutSection>

              <FlyoutSection
                title={t('traderConfigFlyout.tradingScheduleTitle')}
                icon={Clock3}
                iconClassName="text-cyan-500"
                count={tradingScheduleDraft.enabled ? t('traderConfigFlyout.scheduleStateActive') : t('traderConfigFlyout.scheduleStateAlwaysOn')}
                subtitle={t('traderConfigFlyout.tradingScheduleSubtitle')}
              >
                <div className="rounded-md border border-border p-3 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">{t('traderConfigFlyout.enableScheduleGate')}</p>
                      <p className="text-[10px] text-muted-foreground">
                        {t('traderConfigFlyout.scheduleGateOffHelp')}
                      </p>
                    </div>
                    <Switch
                      checked={tradingScheduleDraft.enabled}
                      onCheckedChange={(checked) => setDraftTradingSchedule({ enabled: checked })}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-1.5 md:grid-cols-6">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px]"
                      onClick={() => setDraftTradingSchedule({ enabled: false })}
                    >
                      {t('traderConfigFlyout.alwaysOn')}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px]"
                      onClick={() =>
                        setDraftTradingSchedule({
                          enabled: true,
                          days: [...TRADING_SCHEDULE_DAYS],
                          startTimeUtc: '00:00',
                          endTimeUtc: '23:59',
                        })
                      }
                    >
                      24/7
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px]"
                      onClick={() =>
                        setDraftTradingSchedule({
                          enabled: true,
                          days: [...TRADING_SCHEDULE_WEEKDAYS],
                          startTimeUtc: '00:00',
                          endTimeUtc: '23:59',
                        })
                      }
                    >
                      {t('traderConfigFlyout.weekdays')}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px]"
                      onClick={() =>
                        setDraftTradingSchedule({
                          enabled: true,
                          days: [...TRADING_SCHEDULE_WEEKENDS],
                          startTimeUtc: '00:00',
                          endTimeUtc: '23:59',
                        })
                      }
                    >
                      {t('traderConfigFlyout.weekends')}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px]"
                      onClick={() =>
                        setDraftTradingSchedule({
                          enabled: true,
                          startTimeUtc: '00:00',
                          endTimeUtc: '23:59',
                        })
                      }
                    >
                      {t('traderConfigFlyout.resetWindow')}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px]"
                      onClick={() =>
                        setDraftTradingSchedule({
                          startDateUtc: '',
                          endDateUtc: '',
                          endAtUtc: '',
                        })
                      }
                    >
                      {t('traderConfigFlyout.clearBounds')}
                    </Button>
                  </div>

                  <div>
                    <Label className="text-[11px] text-muted-foreground">{t('traderConfigFlyout.daysUtc')}</Label>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {TRADING_SCHEDULE_DAYS.map((day) => {
                        const selected = tradingScheduleDraft.days.includes(day)
                        return (
                          <Button
                            key={day}
                            type="button"
                            size="sm"
                            variant={selected ? 'default' : 'outline'}
                            className="h-6 px-2 text-[11px]"
                            onClick={() => toggleTradingScheduleDay(day)}
                            disabled={!tradingScheduleDraft.enabled}
                          >
                            {t(TRADING_SCHEDULE_DAY_LABEL_KEYS[day])}
                          </Button>
                        )
                      })}
                    </div>
                  </div>

                  <div className="grid gap-2 md:grid-cols-2">
                    <div>
                      <Label className="text-[11px] text-muted-foreground">{t('traderConfigFlyout.startTimeUtc')}</Label>
                      <Input
                        type="time"
                        value={tradingScheduleDraft.startTimeUtc}
                        onChange={(event) => setDraftTradingSchedule({ startTimeUtc: event.target.value })}
                        className="mt-1 h-8 text-xs font-mono"
                        disabled={!tradingScheduleDraft.enabled}
                      />
                    </div>
                    <div>
                      <Label className="text-[11px] text-muted-foreground">{t('traderConfigFlyout.endTimeUtc')}</Label>
                      <Input
                        type="time"
                        value={tradingScheduleDraft.endTimeUtc}
                        onChange={(event) => setDraftTradingSchedule({ endTimeUtc: event.target.value })}
                        className="mt-1 h-8 text-xs font-mono"
                        disabled={!tradingScheduleDraft.enabled}
                      />
                    </div>
                    <div>
                      <Label className="text-[11px] text-muted-foreground">{t('traderConfigFlyout.startDateUtc')}</Label>
                      <Input
                        type="date"
                        value={tradingScheduleDraft.startDateUtc}
                        onChange={(event) => setDraftTradingSchedule({ startDateUtc: event.target.value })}
                        className="mt-1 h-8 text-xs font-mono"
                        disabled={!tradingScheduleDraft.enabled}
                      />
                    </div>
                    <div>
                      <Label className="text-[11px] text-muted-foreground">{t('traderConfigFlyout.endDateUtc')}</Label>
                      <Input
                        type="date"
                        value={tradingScheduleDraft.endDateUtc}
                        onChange={(event) => setDraftTradingSchedule({ endDateUtc: event.target.value })}
                        className="mt-1 h-8 text-xs font-mono"
                        disabled={!tradingScheduleDraft.enabled}
                      />
                    </div>
                  </div>

                  <div>
                    <Label className="text-[11px] text-muted-foreground">{t('traderConfigFlyout.hardEndTimestamp')}</Label>
                    <Input
                      value={tradingScheduleDraft.endAtUtc}
                      onChange={(event) => setDraftTradingSchedule({ endAtUtc: event.target.value })}
                      placeholder="2026-02-28T23:59:59Z"
                      className="mt-1 h-8 text-xs font-mono"
                      disabled={!tradingScheduleDraft.enabled}
                    />
                  </div>
                </div>
              </FlyoutSection>

              {mode === 'edit' && selectedTrader ? (
                <FlyoutSection
                  title={t('traderConfigFlyout.deleteDisableTitle')}
                  icon={AlertTriangle}
                  iconClassName="text-red-500"
                  tone="danger"
                  count={selectedTraderDeleteExposureSummary || t('traderConfigFlyout.noOpenExposure')}
                  defaultOpen={false}
                >
                  <p className="text-xs text-muted-foreground">
                    {t('traderConfigFlyout.openLivePositions')}: {selectedTraderOpenLivePositions} • {t('traderConfigFlyout.openShadowPositions')}: {selectedTraderOpenShadowPositions}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {t('traderConfigFlyout.openLiveOrders')}: {selectedTraderOpenLiveOrders} • {t('traderConfigFlyout.openShadowOrders')}: {selectedTraderOpenShadowOrders}
                  </p>
                  <Select
                    value={deleteAction}
                    onValueChange={(value) => {
                      setDeleteAction(value as DeleteAction)
                      setDeleteForceConfirm(false)
                      if (value !== 'transfer_delete') setDeleteTransferTargetId(null)
                    }}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="disable">{t('traderConfigFlyout.actionDisable')}</SelectItem>
                      <SelectItem value="transfer_delete">{t('traderConfigFlyout.actionTransferDelete')}</SelectItem>
                      <SelectItem value="block">{t('traderConfigFlyout.actionBlock')}</SelectItem>
                      <SelectItem value="force_delete">{t('traderConfigFlyout.actionForceDelete')}</SelectItem>
                    </SelectContent>
                  </Select>
                  {deleteAction === 'transfer_delete' ? (
                    <div className="space-y-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3">
                      <p className="text-xs font-medium text-amber-700 dark:text-amber-100">
                        {t('traderConfigFlyout.transferTradesTitle')}
                      </p>
                      <p className="text-[11px] text-amber-700/90 dark:text-amber-100/90">
                        {t('traderConfigFlyout.transferTradesHelp', { name: selectedTrader.name })}
                      </p>
                      <Select
                        value={deleteTransferTargetId || ''}
                        onValueChange={(value) => setDeleteTransferTargetId(value || null)}
                      >
                        <SelectTrigger className="h-8">
                          <SelectValue placeholder={t('traderConfigFlyout.selectTargetBot')} />
                        </SelectTrigger>
                        <SelectContent>
                          {traders
                            .filter((t) => t.id !== selectedTrader.id)
                            .map((t) => (
                              <SelectItem key={t.id} value={t.id}>
                                {t.name}
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ) : null}
                  {deleteAction === 'force_delete' ? (
                    <div className="space-y-2 rounded-md border border-red-500/40 bg-red-500/10 p-3">
                      <p className="text-xs font-medium text-red-700 dark:text-red-100">
                        {selectedTraderHasLiveDeleteExposure ? t('traderConfigFlyout.forceDeleteWithLive') : t('traderConfigFlyout.confirmForceDelete')}
                      </p>
                      <p className="text-[11px] text-red-700/90 dark:text-red-100/90">
                        {selectedTraderHasLiveDeleteExposure
                          ? t('traderConfigFlyout.forceDeleteLiveBody', { name: selectedTrader.name })
                          : t('traderConfigFlyout.forceDeleteBody', { name: selectedTrader.name })}
                      </p>
                      {selectedTraderHasAnyDeleteExposure ? (
                        <p className="text-[11px] text-red-700/90 dark:text-red-100/90">
                          {t('traderConfigFlyout.currentExposure', { exposure: selectedTraderDeleteExposureSummary })}
                        </p>
                      ) : null}
                      <div className="flex items-center justify-between gap-3 rounded-md border border-red-500/30 bg-background/70 px-3 py-2">
                        <div className="space-y-0.5">
                          <Label className="text-xs text-foreground">{t('traderConfigFlyout.confirmPermanentDeletion')}</Label>
                          <p className="text-[11px] text-muted-foreground">
                            {selectedTraderHasLiveDeleteExposure
                              ? t('traderConfigFlyout.understandOrphanLive')
                              : t('traderConfigFlyout.understandPermanentDelete')}
                          </p>
                        </div>
                        <Switch checked={deleteForceConfirm} onCheckedChange={setDeleteForceConfirm} />
                      </div>
                    </div>
                  ) : null}
                  <Button
                    variant="destructive"
                    className="h-8 text-xs"
                    disabled={
                      deleteTraderMutation.isPending ||
                      (deleteAction === 'force_delete' && !deleteForceConfirm) ||
                      (deleteAction === 'transfer_delete' && !deleteTransferTargetId)
                    }
                    onClick={() => deleteTraderMutation.mutate({
                      traderId: selectedTrader.id,
                      action: deleteAction,
                      ...(deleteAction === 'transfer_delete' && deleteTransferTargetId
                        ? { transferToTraderId: deleteTransferTargetId }
                        : {}),
                    })}
                  >
                    {deleteTraderMutation.isPending
                      ? t('traderConfigFlyout.processing')
                      : deleteAction === 'disable'
                        ? t('traderConfigFlyout.disableBot')
                        : deleteAction === 'transfer_delete'
                          ? t('traderConfigFlyout.deleteAndTransfer')
                          : deleteAction === 'force_delete'
                            ? t('traderConfigFlyout.forceDeleteBot')
                            : t('traderConfigFlyout.deleteBot')}
                  </Button>
                </FlyoutSection>
              ) : null}
            </div>
          </ScrollArea>

          <div className="border-t border-border px-4 py-3 flex flex-wrap items-center justify-end gap-2">
            {saveError ? (
              <div className="mr-auto text-xs text-red-500 max-w-[65%] break-words leading-tight" title={saveError}>
                {saveError}
              </div>
            ) : null}
            <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
              {t('traderConfigFlyout.close')}
            </Button>
            <Button
              onClick={() => {
                if (mode === 'create') {
                  createTraderMutation.mutate()
                  return
                }
                if (selectedTrader) {
                  saveTraderMutation.mutate(selectedTrader.id)
                }
              }}
              disabled={
                busy ||
                !draftNameValue.trim() ||
                (mode === 'create' && !draftCopyFromTraderId && !effectiveDraftSourceKey)
              }
            >
              {mode === 'create' ? t('traderConfigFlyout.createBot') : t('traderConfigFlyout.saveBot')}
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}

export const TraderConfigFlyout = memo(TraderConfigFlyoutImpl)
