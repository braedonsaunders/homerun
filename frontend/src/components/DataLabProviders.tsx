/**
 * Data Lab → Providers tab.
 *
 * Lets the operator browse external data providers (currently only
 * polybacktest.com), search their market catalog, kick off historical
 * imports, watch the import jobs make progress, and review the
 * imported datasets that the Backtest Studio can now consume.
 *
 * State machine:
 *   1. Pick provider (defaults to first configured).
 *   2. Pick coin → search → checkbox-pick markets.
 *   3. Set time window + click Import → enqueues a ProviderImportJob.
 *   4. Active jobs panel polls every 2s while any are running.
 *   5. Imported datasets panel lists the catalog with delete + use-in-
 *      backtest hooks.
 *
 * No hardcoded settings — the API key + base URL come from
 * Settings → Providers (per the no-hidden-defaults policy).
 */
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  CircleAlert,
  CheckCircle2,
  Database,
  Download,
  ExternalLink,
  Loader2,
  Search,
  Server,
  Trash2,
  X,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { ScrollArea } from './ui/scroll-area'
import { cn } from '../lib/utils'
import {
  cancelImportJob,
  deleteProviderDataset,
  getParquetRoot,
  getProviderSettings,
  importPolybacktest,
  listImportJobs,
  listParquetDatasets,
  listPolybacktestMarkets,
  listProviderDatasets,
  listProviders,
  rescanParquetRoot,
  updateProviderSettings,
  type ImportJob,
  type ImportJobStatus,
  type ParquetDataset,
  type ParquetRescanReport,
  type PolybacktestMarket,
  type ProviderDataset,
  type ProviderInfo,
  type ProviderSettings,
} from '../services/apiProviders'
import {
  listRecordingSessions,
  runRecorderBackfill,
  type BackfillResult,
  type BackfillScope,
} from '../services/apiDataset'


const COINS = ['btc', 'eth', 'sol'] as const

const TIME_PRESETS: Array<{ label: string; hours: number }> = [
  { label: '24 h', hours: 24 },
  { label: '3 d', hours: 24 * 3 },
  { label: '7 d', hours: 24 * 7 },
  { label: '30 d', hours: 24 * 30 },
]


// Synthetic provider key for the built-in Polymarket REST backfill — it's
// not part of /api/providers (which only exposes configurable third-party
// vendors like polybacktest), but we surface it as a sub-tab so the
// operator finds historical-gap-filling tools next to other importers.
const POLYMARKET_TAB_KEY = '__polymarket__'

// Synthetic provider key for the parquet bring-your-own-data sub-tab.
// Operator drops parquet files into HOMERUN_PARQUET_ROOT; the
// auto-discovery scanner upserts them into provider_datasets and the
// backtester's resolver picks them up automatically.
const PARQUET_TAB_KEY = '__parquet__'

export default function DataLabProviders() {
  const { t } = useTranslation()
  // ── Providers list ───────────────────────────────────────────────
  const providersQuery = useQuery({
    queryKey: ['providers', 'list'],
    queryFn: listProviders,
    staleTime: 60_000,
  })
  const providers: ProviderInfo[] = providersQuery.data ?? []
  const [activeProvider, setActiveProvider] = useState<string | null>(null)
  // Auto-select the first provider when the list arrives.
  useEffect(() => {
    if (activeProvider == null && providers.length > 0) {
      setActiveProvider(providers[0].key)
    }
  }, [providers, activeProvider])
  const selected = providers.find((p) => p.key === activeProvider) ?? null
  const isPolymarketTab = activeProvider === POLYMARKET_TAB_KEY
  const isParquetTab = activeProvider === PARQUET_TAB_KEY

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-violet-400" />
            <span className="text-sm font-semibold">{t('dataLabProviders.title')}</span>
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            {t('dataLabProviders.subtitle')}
          </p>
        </div>
      </div>

      {providersQuery.isLoading ? (
        <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {t('dataLabProviders.loadingProviders')}
        </div>
      ) : null}

      {/* Provider sub-tabs — one pill per integrated provider.  Polymarket
          is a built-in sub-tab (it's the source for the REST backfill
          synthesizer); Polybacktest etc. come from /api/providers. */}
      <div className="flex items-center gap-1 border-b border-border/30">
        {providers.map((p) => (
          <button
            key={p.key}
            type="button"
            onClick={() => setActiveProvider(p.key)}
            className={cn(
              '-mb-px flex items-center gap-1.5 border-b-2 px-3 py-1.5 text-[11px] font-medium transition-colors',
              activeProvider === p.key
                ? 'border-violet-500 text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
          >
            <Server className="h-3 w-3" />
            {p.label}
            <ProviderHealthDot provider={p} />
          </button>
        ))}
        <button
          type="button"
          onClick={() => setActiveProvider(POLYMARKET_TAB_KEY)}
          className={cn(
            '-mb-px flex items-center gap-1.5 border-b-2 px-3 py-1.5 text-[11px] font-medium transition-colors',
            isPolymarketTab
              ? 'border-violet-500 text-foreground'
              : 'border-transparent text-muted-foreground hover:text-foreground',
          )}
        >
          <Download className="h-3 w-3 rotate-180" />
          {t('dataLabProviders.polymarketLabel')}
        </button>
        <button
          type="button"
          onClick={() => setActiveProvider(PARQUET_TAB_KEY)}
          className={cn(
            '-mb-px flex items-center gap-1.5 border-b-2 px-3 py-1.5 text-[11px] font-medium transition-colors',
            isParquetTab
              ? 'border-violet-500 text-foreground'
              : 'border-transparent text-muted-foreground hover:text-foreground',
          )}
        >
          <Server className="h-3 w-3" />
          Parquet
        </button>
      </div>

      {isPolymarketTab ? (
        <PolymarketSection />
      ) : isParquetTab ? (
        <ParquetSection />
      ) : selected?.key === 'polybacktest' ? (
        <PolybacktestSection provider={selected} />
      ) : selected ? (
        <div className="rounded-md border border-border/40 bg-card/40 p-4 text-[11px] text-muted-foreground">
          {t('dataLabProviders.notImplemented', { label: selected.label })}
        </div>
      ) : null}
    </div>
  )
}


/** Tiny health pulse beside the provider tab label. */
function ProviderHealthDot({ provider }: { provider: ProviderInfo }) {
  const tone = !provider.configured
    ? 'bg-amber-500/70'
    : provider.health.ok === false
    ? 'bg-rose-500/70'
    : 'bg-emerald-500/70'
  return <span className={cn('h-1.5 w-1.5 rounded-full', tone)} />
}


function ProviderHealthBadge({ provider }: { provider: ProviderInfo }) {
  const { t } = useTranslation()
  if (!provider.configured) {
    return (
      <Badge variant="outline" className="gap-1 border-amber-500/40 text-amber-700 dark:text-amber-300">
        <CircleAlert className="h-3 w-3" />
        {t('dataLabProviders.needsApiKey')}
      </Badge>
    )
  }
  if (provider.health.ok === false) {
    return (
      <Badge variant="outline" className="gap-1 border-rose-500/40 text-rose-700 dark:text-rose-300">
        <AlertTriangle className="h-3 w-3" />
        {t('dataLabProviders.unreachable')}
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="gap-1 border-emerald-500/40 text-emerald-700 dark:text-emerald-300">
      <CheckCircle2 className="h-3 w-3" />
      {t('dataLabProviders.healthy')}
    </Badge>
  )
}


function PolybacktestSection({ provider }: { provider: ProviderInfo }) {
  const { t } = useTranslation()
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {/* Provider summary strip — health, links, coin support. */}
      <div className="rounded-md border border-border/40 bg-card/40 p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold">{provider.label}</span>
              <ProviderHealthBadge provider={provider} />
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">{provider.description}</p>
            <div className="mt-1 flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
              <a
                href={provider.homepage}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 hover:text-foreground"
              >
                {t('dataLabProviders.homepage')} <ExternalLink className="h-3 w-3" />
              </a>
              <a
                href={provider.docs_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 hover:text-foreground"
              >
                {t('dataLabProviders.apiDocs')} <ExternalLink className="h-3 w-3" />
              </a>
              <span>{t('dataLabProviders.coins', { list: provider.supported_coins.join(', ') })}</span>
            </div>
          </div>
        </div>
        {!provider.configured ? (
          <div className="mt-2 rounded-sm border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] text-amber-700 dark:text-amber-200" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.addApiKeyHint') }} />
        ) : null}
      </div>

      <ProviderSettingsCard providerKey="polybacktest" />

      {provider.configured ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <PolybacktestImportPanel />
          <PolybacktestActiveJobsPanel />
        </div>
      ) : null}

      <PolybacktestDatasetsPanel />
    </div>
  )
}


/**
 * Settings card — covers the polybacktest API key + numeric reverse-
 * engineer defaults (max iterations, target score, cost cap).
 *
 * The default LLM *model* for the reverse-engineer agent lives in
 * AI → Models (under "Strategy Reverse-Engineer") so it sits next to
 * every other per-purpose model override.
 */
function ProviderSettingsCard({ providerKey: _providerKey }: { providerKey: string }) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ['providers', 'settings'],
    queryFn: getProviderSettings,
    staleTime: 60_000,
  })
  const settings: ProviderSettings | null = settingsQuery.data ?? null

  const [apiKey, setApiKey] = useState<string>('')
  const [showKey, setShowKey] = useState<boolean>(false)
  const [baseUrl, setBaseUrl] = useState<string>('')
  const [maxIter, setMaxIter] = useState<string>('')
  const [targetScore, setTargetScore] = useState<string>('')
  const [maxCost, setMaxCost] = useState<string>('')
  const [maxTrades, setMaxTrades] = useState<string>('')

  // Hydrate the form from the server snapshot once.
  useEffect(() => {
    if (!settings) return
    setApiKey(settings.polybacktest_api_key_set ? '********' : '')
    setBaseUrl(settings.polybacktest_base_url ?? '')
    setMaxIter(settings.reverse_engineer_max_iterations?.toString() ?? '')
    setTargetScore(settings.reverse_engineer_target_score?.toString() ?? '')
    setMaxCost(settings.reverse_engineer_max_cost_usd?.toString() ?? '')
    setMaxTrades(settings.reverse_engineer_max_wallet_trades?.toString() ?? '')
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: () =>
      updateProviderSettings({
        polybacktest_api_key: apiKey === '********' ? null : apiKey,
        polybacktest_base_url: baseUrl,
        reverse_engineer_max_iterations: maxIter ? parseInt(maxIter, 10) : null,
        reverse_engineer_target_score: targetScore ? parseFloat(targetScore) : null,
        reverse_engineer_max_cost_usd: maxCost ? parseFloat(maxCost) : null,
        reverse_engineer_max_wallet_trades: maxTrades ? parseInt(maxTrades, 10) : null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] })
    },
  })

  return (
    <details className="rounded-md border border-border/40 bg-card/40 p-3">
      <summary className="cursor-pointer text-xs font-semibold flex items-center gap-1.5">
        <Server className="h-3.5 w-3.5 text-violet-400" />
        {t('dataLabProviders.providerSettings')}
        <span className="ml-auto text-[10px] font-normal text-muted-foreground">
          {settings?.polybacktest_api_key_set ? t('dataLabProviders.configured') : t('dataLabProviders.notConfigured')}
        </span>
      </summary>
      <div className="mt-3 space-y-3">
        {/* API key */}
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">
            {t('dataLabProviders.polybacktestApiKey')}
          </Label>
          <div className="flex items-center gap-1">
            <Input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={settings?.polybacktest_api_key_set ? t('dataLabProviders.apiKeyPlaceholderSet') : t('dataLabProviders.apiKeyPlaceholder')}
              className="h-8 font-mono text-xs"
            />
            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0"
              onClick={() => setShowKey((v) => !v)}
              title={showKey ? t('dataLabProviders.hideKey') : t('dataLabProviders.showKey')}
            >
              {showKey ? '🙈' : '👁'}
            </Button>
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.apiKeyHelp', { interpolation: { escapeValue: false } }).replace('<a>', '<a href="https://polybacktest.com/dashboard" target="_blank" rel="noreferrer" class="underline">') }} />
        </div>

        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.baseUrl')}</Label>
          <Input
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder={t('dataLabProviders.baseUrlPlaceholder')}
            className="h-8 font-mono text-xs"
          />
        </div>

        <div className="border-t border-border/30 pt-3">
          <div className="text-[11px] font-semibold mb-1">{t('dataLabProviders.reverseEngineerDefaults')}</div>
          <p className="mb-2 text-[10px] text-muted-foreground" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.reverseEngineerDefaultsHelp') }} />
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.maxIterations')}</Label>
              <Input
                value={maxIter}
                onChange={(e) => setMaxIter(e.target.value)}
                placeholder="10"
                className="h-8 text-xs"
              />
            </div>
            <div>
              <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.targetScore')}</Label>
              <Input
                value={targetScore}
                onChange={(e) => setTargetScore(e.target.value)}
                placeholder="0.7"
                className="h-8 text-xs"
              />
            </div>
            <div>
              <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.maxCostUsd')}</Label>
              <Input
                value={maxCost}
                onChange={(e) => setMaxCost(e.target.value)}
                placeholder={t('dataLabProviders.maxCostPlaceholder')}
                className="h-8 text-xs"
              />
            </div>
            <div className="col-span-2">
              <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.maxWalletTrades')}</Label>
              <Input
                value={maxTrades}
                onChange={(e) => setMaxTrades(e.target.value)}
                placeholder="50000"
                className="h-8 text-xs"
              />
            </div>
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.fallbackHelp') }} />
        </div>

        <div className="flex items-center justify-end gap-2">
          {saveMutation.isError ? (
            <span className="text-[10px] text-rose-700 dark:text-rose-300">
              {(saveMutation.error as Error)?.message || t('dataLabProviders.saveFailed')}
            </span>
          ) : null}
          {saveMutation.isSuccess ? (
            <span className="text-[10px] text-emerald-700 dark:text-emerald-300">{t('dataLabProviders.saved')}</span>
          ) : null}
          <Button
            size="sm"
            className="h-7 text-[11px]"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending}
          >
            {saveMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : t('dataLabProviders.save')}
          </Button>
        </div>
      </div>
    </details>
  )
}


// ─── Import panel: pick markets + window, kick off job ──────────────

type MarketTypeFilter = 'all' | '5m' | '15m' | '1h' | '4h' | '24h'
type ResolvedFilter = 'all' | 'resolved' | 'open'

function PolybacktestImportPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [coin, setCoin] = useState<(typeof COINS)[number]>('btc')
  const [search, setSearch] = useState('')
  const [appliedSearch, setAppliedSearch] = useState('')
  const [marketType, setMarketType] = useState<MarketTypeFilter>('all')
  const [resolvedFilter, setResolvedFilter] = useState<ResolvedFilter>('all')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [hours, setHours] = useState<number>(24 * 7)

  const marketsQuery = useQuery({
    queryKey: [
      'polybacktest',
      'markets',
      coin,
      appliedSearch,
      marketType,
      resolvedFilter,
    ],
    queryFn: () =>
      listPolybacktestMarkets({
        coin,
        search: appliedSearch || undefined,
        market_type: marketType === 'all' ? undefined : marketType,
        resolved:
          resolvedFilter === 'all'
            ? undefined
            : resolvedFilter === 'resolved',
        limit: 100,
      }),
    staleTime: 60_000,
  })
  const markets: PolybacktestMarket[] = marketsQuery.data?.markets ?? []

  const importMutation = useMutation({
    mutationFn: () => {
      // Use each market's ACTUAL window when available — that gives
      // us the full 5m/15m/1h slice the operator selected, not an
      // arbitrary "last N days" overlay.  For markets that haven't
      // closed yet, fall back to the operator's chosen lookback.
      const selectedMarkets = markets.filter((m) => selected.has(m.market_id))
      let start: Date
      let end: Date
      if (selectedMarkets.length > 0 && selectedMarkets.every((m) => m.start_time && m.end_time)) {
        const starts = selectedMarkets.map((m) => new Date(m.start_time!).getTime())
        const ends = selectedMarkets.map((m) => new Date(m.end_time!).getTime())
        start = new Date(Math.min(...starts))
        end = new Date(Math.max(...ends))
      } else {
        end = new Date()
        start = new Date(end.getTime() - hours * 3600 * 1000)
      }
      return importPolybacktest({
        coin,
        market_ids: Array.from(selected),
        start: start.toISOString(),
        end: end.toISOString(),
      })
    },
    onSuccess: () => {
      setSelected(new Set())
      queryClient.invalidateQueries({ queryKey: ['providers', 'import-jobs'] })
    },
  })

  return (
    <div className="flex min-h-0 flex-col rounded-md border border-border/40 bg-card/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs font-semibold">{t('dataLabProviders.importHistorical')}</div>
        <Badge variant="outline" className="text-[10px]">{t('dataLabProviders.selectedCount', { n: selected.size })}</Badge>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2">
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.coin')}</Label>
          <Select value={coin} onValueChange={(v) => setCoin(v as (typeof COINS)[number])}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {COINS.map((c) => (
                <SelectItem key={c} value={c} className="text-xs">{c.toUpperCase()}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.horizon')}</Label>
          <Select value={marketType} onValueChange={(v) => setMarketType(v as MarketTypeFilter)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="text-xs">{t('dataLabProviders.allHorizons')}</SelectItem>
              {(['5m', '15m', '1h', '4h', '24h'] as const).map((mt) => (
                <SelectItem key={mt} value={mt} className="text-xs">{mt}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">{t('dataLabProviders.status')}</Label>
          <Select value={resolvedFilter} onValueChange={(v) => setResolvedFilter(v as ResolvedFilter)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="text-xs">{t('dataLabProviders.all')}</SelectItem>
              <SelectItem value="resolved" className="text-xs">{t('dataLabProviders.resolvedOnly')}</SelectItem>
              <SelectItem value="open" className="text-xs">{t('dataLabProviders.openOnly')}</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">
            {t('dataLabProviders.fallbackWindow')}
          </Label>
          <Select value={String(hours)} onValueChange={(v) => setHours(Number(v))}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {TIME_PRESETS.map((p) => (
                <SelectItem key={p.hours} value={String(p.hours)} className="text-xs">
                  {t('dataLabProviders.lastRange', { label: p.label })}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="mt-2 flex items-center gap-1">
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') setAppliedSearch(search)
          }}
          placeholder={t('dataLabProviders.searchMarkets')}
          className="h-8 text-xs"
        />
        <Button
          size="sm"
          variant="outline"
          className="h-8 gap-1 text-[10px]"
          onClick={() => setAppliedSearch(search)}
          disabled={marketsQuery.isFetching}
        >
          <Search className="h-3 w-3" /> {t('dataLabProviders.search')}
        </Button>
      </div>

      <ScrollArea className="mt-2 h-56 rounded-sm border border-border/30 bg-background/40">
        {marketsQuery.isLoading ? (
          <div className="flex h-full items-center justify-center text-[11px] text-muted-foreground">
            <Loader2 className="mr-2 h-3 w-3 animate-spin" /> {t('dataLabProviders.loading')}
          </div>
        ) : marketsQuery.isError ? (
          <div className="p-3 text-[11px] text-rose-700 dark:text-rose-300">
            {String((marketsQuery.error as Error)?.message || t('dataLabProviders.failedToLoad'))}
          </div>
        ) : markets.length === 0 ? (
          <div className="p-3 text-[11px] text-muted-foreground">{t('dataLabProviders.noMarketsFound')}</div>
        ) : (
          <div className="divide-y divide-border/20">
            {markets.map((m) => {
              const isSel = selected.has(m.market_id)
              return (
                <button
                  key={m.market_id}
                  type="button"
                  onClick={() => {
                    setSelected((prev) => {
                      const next = new Set(prev)
                      if (next.has(m.market_id)) next.delete(m.market_id)
                      else next.add(m.market_id)
                      return next
                    })
                  }}
                  className={cn(
                    'block w-full px-2 py-1.5 text-left text-[11px] transition-colors hover:bg-card/40',
                    isSel && 'bg-violet-500/10',
                  )}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={isSel}
                      readOnly
                      className="h-3 w-3 accent-violet-500"
                    />
                    <span className="flex-1 truncate font-medium">{m.title}</span>
                    {m.winner ? (
                      <Badge
                        variant="outline"
                        className={cn(
                          'text-[9px]',
                          m.winner === 'Up'
                            ? 'border-emerald-500/40 text-emerald-700 dark:text-emerald-300'
                            : 'border-rose-500/40 text-rose-700 dark:text-rose-300',
                        )}
                      >
                        {m.winner.toUpperCase()}
                      </Badge>
                    ) : null}
                  </div>
                  <div className="ml-5 flex items-center gap-2 text-[10px] text-muted-foreground">
                    <span className="font-mono">{m.market_id}</span>
                    {m.market_type ? <span>· {m.market_type}</span> : null}
                    {m.final_volume != null ? (
                      <span>· vol ${m.final_volume.toLocaleString()}</span>
                    ) : null}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </ScrollArea>

      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-[10px] text-muted-foreground">
          {t('dataLabProviders.depthDescription')}
        </span>
        <Button
          size="sm"
          className="h-8 gap-1.5 text-[11px]"
          disabled={selected.size === 0 || importMutation.isPending}
          onClick={() => importMutation.mutate()}
        >
          {importMutation.isPending ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Download className="h-3 w-3" />
          )}
          {selected.size === 1 ? t('dataLabProviders.importNMarkets', { n: selected.size }) : t('dataLabProviders.importNMarketsPlural', { n: selected.size })}
        </Button>
      </div>

      {importMutation.isError ? (
        <div className="mt-2 rounded-sm border border-rose-500/30 bg-rose-500/5 p-2 text-[10px] text-rose-700 dark:text-rose-300">
          {String((importMutation.error as Error)?.message || t('dataLabProviders.importFailed'))}
        </div>
      ) : null}
      {importMutation.isSuccess ? (
        <div className="mt-2 rounded-sm border border-emerald-500/30 bg-emerald-500/5 p-2 text-[10px] text-emerald-700 dark:text-emerald-300" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.jobQueued', { id: importMutation.data.id }) }} />
      ) : null}
    </div>
  )
}


// ─── Active import jobs panel (auto-polling) ─────────────────────────

function PolybacktestActiveJobsPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const jobsQuery = useQuery({
    queryKey: ['providers', 'import-jobs'],
    queryFn: () => listImportJobs({ limit: 20 }),
    refetchInterval: (q) => {
      const data = q.state.data as ImportJob[] | undefined
      const anyActive = (data ?? []).some(
        (j) => j.status === 'queued' || j.status === 'running',
      )
      return anyActive ? 2_000 : 30_000
    },
  })
  const jobs: ImportJob[] = jobsQuery.data ?? []

  const cancelMutation = useMutation({
    mutationFn: (id: string) => cancelImportJob(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers', 'import-jobs'] }),
  })

  return (
    <div className="flex min-h-0 flex-col rounded-md border border-border/40 bg-card/40 p-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold">{t('dataLabProviders.activeImportJobs')}</div>
        <Badge variant="outline" className="text-[10px]">{jobs.length}</Badge>
      </div>
      <ScrollArea className="mt-2 max-h-72">
        {jobs.length === 0 ? (
          <div className="px-1 py-4 text-center text-[11px] text-muted-foreground">
            {t('dataLabProviders.noImportsYet')}
          </div>
        ) : (
          <div className="space-y-1.5">
            {jobs.map((job) => (
              <ImportJobRow
                key={job.id}
                job={job}
                onCancel={() => cancelMutation.mutate(job.id)}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}


function statusColor(status: ImportJobStatus): string {
  switch (status) {
    case 'completed':
      return 'border-emerald-500/40 text-emerald-700 dark:text-emerald-300'
    case 'failed':
      return 'border-rose-500/40 text-rose-700 dark:text-rose-300'
    case 'cancelled':
      return 'border-zinc-500/40 text-zinc-300'
    case 'running':
      return 'border-blue-500/40 text-blue-700 dark:text-blue-300'
    default:
      return 'border-amber-500/40 text-amber-700 dark:text-amber-300'
  }
}


function ImportJobRow({ job, onCancel }: { job: ImportJob; onCancel: () => void }) {
  const { t } = useTranslation()
  const payload = job.payload as { coin?: string; market_ids?: string[] } | null
  const coin = payload?.coin ?? '?'
  const marketCount = payload?.market_ids?.length ?? 0
  const pct = Math.max(0, Math.min(1, job.progress)) * 100
  const isActive = job.status === 'queued' || job.status === 'running'

  return (
    <div className="rounded-sm border border-border/30 bg-background/40 p-2">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-[11px]">
            <Badge variant="outline" className={cn('text-[9px]', statusColor(job.status))}>
              {job.status}
            </Badge>
            <span className="font-mono text-[10px] text-muted-foreground">{job.id}</span>
          </div>
          <div className="mt-0.5 truncate text-[11px]">
            {marketCount === 1 ? t('dataLabProviders.marketCount', { coin: coin.toUpperCase(), n: marketCount }) : t('dataLabProviders.marketCountPlural', { coin: coin.toUpperCase(), n: marketCount })}
          </div>
          <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
            {job.message || job.error || t('dataLabProviders.snapshotsInsertedShort', { n: job.snapshots_inserted.toLocaleString() })}
          </div>
        </div>
        {isActive ? (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-rose-700 dark:text-rose-300 hover:bg-rose-500/10"
            onClick={onCancel}
            title={t('dataLabProviders.cancel')}
          >
            <X className="h-3 w-3" />
          </Button>
        ) : null}
      </div>
      {isActive ? (
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-card/40">
          <div
            className="h-full rounded-full bg-violet-500/60 transition-all"
            style={{ width: `${pct.toFixed(1)}%` }}
          />
        </div>
      ) : null}
      {job.snapshots_inserted > 0 || job.api_calls > 0 ? (
        <div className="mt-1 flex flex-wrap gap-2 text-[9px] text-muted-foreground">
          <span>{t('dataLabProviders.apiCalls', { n: job.api_calls.toLocaleString() })}</span>
          <span>{t('dataLabProviders.snapshotsInserted', { n: job.snapshots_inserted.toLocaleString() })}</span>
          <span>{t('dataLabProviders.tradesFetched', { n: job.trades_fetched.toLocaleString() })}</span>
          {job.bytes_downloaded ? (
            <span>{t('dataLabProviders.kbDownloaded', { n: (job.bytes_downloaded / 1024).toFixed(0) })}</span>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}


// ─── Imported datasets panel ─────────────────────────────────────────

function PolybacktestDatasetsPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const datasetsQuery = useQuery({
    queryKey: ['providers', 'datasets'],
    queryFn: () => listProviderDatasets({ limit: 200 }),
    refetchInterval: 30_000,
  })
  const rows: ProviderDataset[] = datasetsQuery.data ?? []
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteProviderDataset(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers', 'datasets'] }),
  })

  return (
    <div className="flex min-h-0 flex-col rounded-md border border-border/40 bg-card/40 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-1.5 text-xs font-semibold">
            <Database className="h-3.5 w-3.5 text-violet-400" />
            {t('dataLabProviders.importedDatasets')}
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.importedDatasetsHint') }} />
        </div>
        <Badge variant="outline" className="text-[10px]">{rows.length}</Badge>
      </div>

      <ScrollArea className="max-h-80">
        {rows.length === 0 ? (
          <div className="py-4 text-center text-[11px] text-muted-foreground">
            {t('dataLabProviders.noDatasetsYet')}
          </div>
        ) : (
          <table className="w-full text-[11px]">
            <thead className="text-[10px] uppercase text-muted-foreground">
              <tr className="border-b border-border/30">
                <th className="px-2 py-1.5 text-left">{t('dataLabProviders.colProvider')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLabProviders.colCoin')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLabProviders.colMarket')}</th>
                <th className="px-2 py-1.5 text-right">{t('dataLabProviders.colSnapshots')}</th>
                <th className="px-2 py-1.5 text-right">{t('dataLabProviders.colTrades')}</th>
                <th className="px-2 py-1.5 text-left">{t('dataLabProviders.colWindow')}</th>
                <th className="px-2 py-1.5"></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-b border-border/20 hover:bg-card/30">
                  <td className="px-2 py-1.5 font-mono text-[10px]">{row.provider}</td>
                  <td className="px-2 py-1.5 font-mono">{row.coin ?? '—'}</td>
                  <td className="px-2 py-1.5">
                    <div className="truncate font-medium">{row.title || row.external_slug || row.external_id}</div>
                    <div className="truncate font-mono text-[9px] text-muted-foreground">
                      {row.external_id}
                    </div>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono">{row.snapshot_count.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right font-mono">{row.trade_count.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-[10px] text-muted-foreground">
                    {row.start_ts ? new Date(row.start_ts).toLocaleDateString() : '—'} →{' '}
                    {row.end_ts ? new Date(row.end_ts).toLocaleDateString() : '—'}
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 text-rose-700 dark:text-rose-300 hover:bg-rose-500/10"
                      onClick={() => {
                        if (confirm(t('dataLabProviders.confirmDeleteDataset', { title: row.title || row.external_id, n: row.snapshot_count.toLocaleString() }))) {
                          deleteMutation.mutate(row.id)
                        }
                      }}
                      title={t('dataLabProviders.delete')}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </ScrollArea>
    </div>
  )
}


// ─── Polymarket REST backfill ────────────────────────────────────────
//
// The "Polymarket" sub-tab synthesizes book snapshots from Polymarket's
// /prices-history endpoint to fill historical gaps that the live WS
// recorder didn't reach.  Rows land in the same MarketMicrostructureSnapshot
// table the recorder writes to (tagged synthetic = true so the
// backtester can downweight).

const BACKFILL_INTERVAL_KEYS: { value: string; labelKey: string }[] = [
  { value: '1m', labelKey: 'interval1m' },
  { value: '1h', labelKey: 'interval1h' },
  { value: '6h', labelKey: 'interval6h' },
  { value: '1d', labelKey: 'interval1d' },
  { value: 'max', labelKey: 'intervalMax' },
]

const BACKFILL_SCOPE_KEYS: { value: BackfillScope; labelKey: string; hintKey: string }[] = [
  { value: 'token', labelKey: 'scopeToken', hintKey: 'scopeTokenHint' },
  { value: 'strategy', labelKey: 'scopeStrategy', hintKey: 'scopeStrategyHint' },
  { value: 'session', labelKey: 'scopeSession', hintKey: 'scopeSessionHint' },
  { value: 'catalog_top_liquid', labelKey: 'scopeCatalog', hintKey: 'scopeCatalogHint' },
]

function PolymarketBackfillFlyout({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation()
  const [scope, setScope] = useState<BackfillScope>('strategy')
  const [tokenText, setTokenText] = useState('')
  const [strategySlug, setStrategySlug] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [startInput, setStartInput] = useState('')
  const [endInput, setEndInput] = useState('')
  const [lookbackDays, setLookbackDays] = useState('14')
  const [interval, setInterval] = useState('1h')
  const [syntheticSpreadBps, setSyntheticSpreadBps] = useState('50')
  const [catalogMaxTokens, setCatalogMaxTokens] = useState('500')
  const [catalogMinLiquidity, setCatalogMinLiquidity] = useState('100')
  const [maxTokens, setMaxTokens] = useState('1000')
  const [result, setResult] = useState<BackfillResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Sessions list for the picker
  const sessionsQuery = useQuery({
    queryKey: ['data-lab', 'recording-sessions-for-backfill'],
    queryFn: () => listRecordingSessions(undefined, 50),
    enabled: open && scope === 'session',
  })

  const queryClient = useQueryClient()
  const backfillMutation = useMutation({
    mutationFn: runRecorderBackfill,
    onSuccess: (data) => {
      setResult(data)
      setError(null)
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'storage'] })
      queryClient.invalidateQueries({ queryKey: ['data-lab', 'query'] })
    },
    onError: (err) => setError((err as Error).message || t('dataLabProviders.errBackfillFailed')),
  })

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const submit = () => {
    setError(null)
    setResult(null)

    let start: string | undefined
    let end: string | undefined
    if (startInput && endInput) {
      start = new Date(startInput).toISOString()
      end = new Date(endInput).toISOString()
    } else {
      const days = parseInt(lookbackDays, 10) || 14
      const e = new Date()
      const s = new Date(e.getTime() - days * 24 * 3600 * 1000)
      start = s.toISOString()
      end = e.toISOString()
    }

    const payload: any = {
      scope,
      start,
      end,
      interval,
      synthetic_spread_bps: parseFloat(syntheticSpreadBps) || 50,
      catalog_max_tokens: parseInt(catalogMaxTokens, 10) || 500,
      catalog_min_liquidity_usd: parseFloat(catalogMinLiquidity) || 100,
      max_tokens: parseInt(maxTokens, 10) || 1000,
      concurrency: 5,
    }
    if (scope === 'token') {
      const tokens = tokenText.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean)
      if (tokens.length === 0) { setError(t('dataLabProviders.errProvideToken')); return }
      payload.target_values = tokens
    } else if (scope === 'strategy') {
      if (!strategySlug.trim()) { setError(t('dataLabProviders.errProvideStrategy')); return }
      payload.strategy_slug = strategySlug.trim()
    } else if (scope === 'session') {
      if (!sessionId) { setError(t('dataLabProviders.errPickSession')); return }
      payload.session_id = sessionId
    }
    backfillMutation.mutate(payload)
  }

  return (
    <>
      <div
        className={cn(
          'fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-opacity',
          open ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        onClick={onClose}
      />
      <div
        className={cn(
          'fixed inset-y-0 right-0 z-50 flex w-[520px] max-w-[95vw] flex-col border-l border-border/60 bg-background/95 shadow-2xl backdrop-blur transition-transform duration-200',
          open ? 'translate-x-0' : 'translate-x-full',
        )}
      >
        <div className="flex items-center justify-between border-b border-border/40 px-4 py-3">
          <div className="flex items-center gap-2">
            <Download className="h-4 w-4 text-violet-700 dark:text-violet-300 rotate-180" />
            <div>
              <div className="text-sm font-semibold leading-tight">{t('dataLabProviders.backfillTitle')}</div>
              <div className="text-[10px] text-muted-foreground leading-tight">
                {t('dataLabProviders.backfillSub')}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-sm p-1 text-muted-foreground hover:bg-muted/40 hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-4 p-4">
            {/* Caveat banner */}
            <div className="rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-[11px] text-amber-700 dark:text-amber-300">
              <div className="font-medium">{t('dataLabProviders.syntheticDataCaveat')}</div>
              <div className="mt-1 text-amber-700/90 dark:text-amber-300/90" dangerouslySetInnerHTML={{ __html: t('dataLabProviders.syntheticDataCaveatBody') }} />
            </div>

            {/* Scope */}
            <div className="space-y-2 rounded-md border border-border/40 bg-card/30 p-3">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('dataLabProviders.scope')}</div>
              <div className="grid grid-cols-2 gap-1.5">
                {BACKFILL_SCOPE_KEYS.map((o) => {
                  const active = scope === o.value
                  return (
                    <button
                      key={o.value}
                      onClick={() => setScope(o.value)}
                      className={cn(
                        'rounded-sm border px-2 py-1.5 text-left transition-colors',
                        active
                          ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300'
                          : 'border-border/40 text-muted-foreground hover:text-foreground',
                      )}
                    >
                      <div className="text-[11px] font-medium">{t(`dataLabProviders.${o.labelKey}`)}</div>
                      <div className="text-[9px] text-muted-foreground/80">{t(`dataLabProviders.${o.hintKey}`)}</div>
                    </button>
                  )
                })}
              </div>

              {scope === 'token' ? (
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.tokenIdsLabel')}</Label>
                  <textarea
                    value={tokenText}
                    onChange={(e) => setTokenText(e.target.value)}
                    placeholder={t('dataLabProviders.tokenIdsPlaceholder')}
                    className="min-h-[80px] w-full rounded-sm border border-border/40 bg-background/60 px-2 py-1.5 font-mono text-[11px]"
                  />
                </div>
              ) : null}

              {scope === 'strategy' ? (
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.strategySlug')}</Label>
                  <Input
                    value={strategySlug}
                    onChange={(e) => setStrategySlug(e.target.value)}
                    placeholder={t('dataLabProviders.strategySlugPlaceholder')}
                    className="h-8 text-[12px]"
                  />
                  <div className="text-[10px] text-muted-foreground">
                    {t('dataLabProviders.strategySlugHint')}
                  </div>
                </div>
              ) : null}

              {scope === 'session' ? (
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.recordingSession')}</Label>
                  <select
                    value={sessionId}
                    onChange={(e) => setSessionId(e.target.value)}
                    className="h-8 w-full rounded-md border border-input bg-background px-3 text-[11px]"
                  >
                    <option value="">{t('dataLabProviders.pickSession')}</option>
                    {(sessionsQuery.data ?? []).map((s) => (
                      <option key={s.id} value={s.id}>
                        {t('dataLabProviders.sessionOptionLabel', { name: s.name, status: s.status, n: s.target_token_ids.length })}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}

              {scope === 'catalog_top_liquid' ? (
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-[10px]">{t('dataLabProviders.capTokens')}</Label>
                    <Input
                      type="number"
                      min={10}
                      max={5000}
                      value={catalogMaxTokens}
                      onChange={(e) => setCatalogMaxTokens(e.target.value)}
                      className="h-8 text-[12px]"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[10px]">{t('dataLabProviders.minLiquidityUsd')}</Label>
                    <Input
                      type="number"
                      min={0}
                      value={catalogMinLiquidity}
                      onChange={(e) => setCatalogMinLiquidity(e.target.value)}
                      className="h-8 text-[12px]"
                    />
                  </div>
                </div>
              ) : null}
            </div>

            {/* Window + cadence */}
            <div className="space-y-2 rounded-md border border-border/40 bg-card/30 p-3">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {t('dataLabProviders.windowCadence')}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.lookbackDays')}</Label>
                  <Input
                    type="number"
                    min={1}
                    max={180}
                    value={lookbackDays}
                    onChange={(e) => setLookbackDays(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.intervalFidelity')}</Label>
                  <select
                    value={interval}
                    onChange={(e) => setInterval(e.target.value)}
                    className="h-8 w-full rounded-md border border-input bg-background px-3 text-[11px]"
                  >
                    {BACKFILL_INTERVAL_KEYS.map((i) => (
                      <option key={i.value} value={i.value}>{t(`dataLabProviders.${i.labelKey}`)}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.startOverridesLookback')}</Label>
                  <Input
                    type="datetime-local"
                    value={startInput}
                    onChange={(e) => setStartInput(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.endLabel')}</Label>
                  <Input
                    type="datetime-local"
                    value={endInput}
                    onChange={(e) => setEndInput(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
              </div>
            </div>

            {/* Synth + caps */}
            <div className="space-y-2 rounded-md border border-border/40 bg-card/30 p-3">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {t('dataLabProviders.synthCaps')}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.syntheticSpreadBps')}</Label>
                  <Input
                    type="number"
                    min={1}
                    max={1000}
                    value={syntheticSpreadBps}
                    onChange={(e) => setSyntheticSpreadBps(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">{t('dataLabProviders.maxTokensCap')}</Label>
                  <Input
                    type="number"
                    min={10}
                    max={10000}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
              </div>
            </div>

            {error ? (
              <div className="rounded-sm bg-rose-500/10 px-3 py-2 text-[12px] text-rose-700 dark:text-rose-300">{error}</div>
            ) : null}

            {result ? (
              <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3 text-[11px]">
                <div className="font-medium text-emerald-700 dark:text-emerald-300">{t('dataLabProviders.backfillComplete')}</div>
                <div className="mt-1 grid grid-cols-2 gap-1 text-emerald-700 dark:text-emerald-300">
                  <div>{t('dataLabProviders.backfillJob')} <span className="font-mono">{result.job_id}</span></div>
                  <div>{t('dataLabProviders.backfillDuration', { n: result.duration_seconds.toFixed(1) })}</div>
                  <div>{t('dataLabProviders.backfillTokensTargeted')} <strong>{result.target_token_count.toLocaleString()}</strong></div>
                  <div>{t('dataLabProviders.backfillTokensWithData', { n: result.tokens_with_data.toLocaleString() })}</div>
                  <div>{t('dataLabProviders.backfillRowsInserted')} <strong>{result.rows_inserted_total.toLocaleString()}</strong></div>
                  <div>{t('dataLabProviders.backfillPointsFetched', { n: result.points_fetched_total.toLocaleString() })}</div>
                  <div>{t('dataLabProviders.backfillExistingSkipped', { n: result.skipped_existing_total.toLocaleString() })}</div>
                  <div>{t('dataLabProviders.backfillErrorsCount', { n: result.tokens_with_errors })}</div>
                </div>
                {result.tokens_with_errors > 0 ? (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-rose-700 dark:text-rose-300">
                      {t('dataLabProviders.backfillFailedTokens', { n: result.tokens_with_errors })}
                    </summary>
                    <div className="mt-1 max-h-[140px] space-y-0.5 overflow-y-auto">
                      {result.per_token.filter((tk) => tk.error).slice(0, 50).map((tk) => (
                        <div key={tk.token_id} className="font-mono text-[10px] text-rose-700 dark:text-rose-300/90">
                          {tk.token_id.slice(0, 14)} — {tk.error}
                        </div>
                      ))}
                    </div>
                  </details>
                ) : null}
              </div>
            ) : null}
          </div>
        </ScrollArea>

        <div className="flex items-center justify-between gap-2 border-t border-border/40 px-4 py-3">
          <span className="text-[10px] text-muted-foreground">
            {t('dataLabProviders.backfillIdempotent')}
          </span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-8 text-[11px]" onClick={onClose}>
              {t('dataLabProviders.close')}
            </Button>
            <Button
              size="sm"
              className="h-8 gap-1 text-[11px]"
              onClick={submit}
              disabled={backfillMutation.isPending}
            >
              {backfillMutation.isPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Download className="h-3 w-3 rotate-180" />
              )}
              {t('dataLabProviders.runBackfill')}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}

function PolymarketSection() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="rounded-md border border-border/40 bg-card/40 p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold">{t('dataLabProviders.polymarketLabel')}</span>
              <Badge variant="outline" className="gap-1 border-emerald-500/40 text-emerald-700 dark:text-emerald-300">
                <CheckCircle2 className="h-3 w-3" />
                {t('dataLabProviders.polymarketBuiltIn')}
              </Badge>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {t('dataLabProviders.polymarketDescription')}
            </p>
          </div>
        </div>
      </div>

      <div className="rounded-md border border-border/40 bg-card/30">
        <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
          <div className="flex items-center gap-2">
            <Download className="h-3.5 w-3.5 rotate-180 text-violet-700 dark:text-violet-300" />
            <span className="text-xs font-semibold">{t('dataLabProviders.restBackfill')}</span>
            <span className="text-[10px] text-muted-foreground">
              {t('dataLabProviders.restBackfillSub')}
            </span>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 text-[10px]"
            onClick={() => setOpen(true)}
          >
            <Download className="h-3 w-3 rotate-180" />
            {t('dataLabProviders.newBackfill')}
          </Button>
        </div>
        <div className="px-3 py-2 text-[10px] text-muted-foreground">
          {t('dataLabProviders.restBackfillBody')}
        </div>
      </div>
      <PolymarketBackfillFlyout open={open} onClose={() => setOpen(false)} />
    </div>
  )
}


// ─── Parquet sub-tab ───────────────────────────────────────────────────
//
// Local-single-user shop: there's no upload UI.  The operator copies
// parquet files into ``HOMERUN_PARQUET_ROOT`` (the path is shown in the
// header card so they know where) using whatever tool — Explorer,
// scp, rsync, a download script — and hits Rescan.  The backtester's
// source resolver picks up parquet-covered tokens automatically on
// the next run.
//
// Files must follow the layout in services/external_data/parquet_schema.py:
//   {root}/{provider}/{coin}/{startISO}__{endISO}/{kind}__{token_id}.parquet
//
// The auto-discovery scanner also runs once every 60s when a backtest
// kicks off, so a file dropped just before pressing Run is picked up
// without an explicit Rescan press.

function ParquetSection() {
  const queryClient = useQueryClient()
  const rootQuery = useQuery({
    queryKey: ['providers', 'parquet', 'root'],
    queryFn: getParquetRoot,
    staleTime: 5 * 60_000,
  })
  const datasetsQuery = useQuery({
    queryKey: ['providers', 'parquet', 'datasets'],
    queryFn: listParquetDatasets,
    staleTime: 30_000,
  })
  const rescanMutation = useMutation({
    mutationFn: rescanParquetRoot,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers', 'parquet', 'datasets'] })
    },
  })
  const datasets = datasetsQuery.data ?? []
  const lastReport: ParquetRescanReport | undefined = rescanMutation.data

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* Storage root — operator copies files into this directory. */}
      <div className="rounded-md border border-border/40 bg-card/40 p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <Database className="h-3.5 w-3.5 text-violet-400" />
              <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                Parquet storage root
              </span>
            </div>
            <p className="mt-1 break-all font-mono text-[11px] text-foreground">
              {rootQuery.isLoading ? 'loading…' : rootQuery.data?.root ?? '—'}
            </p>
            <p className="mt-1 text-[10px] text-muted-foreground">
              Drop parquet files here following the layout{' '}
              <code className="text-[10px]">
                {'{provider}/{coin}/{startISO}__{endISO}/{kind}__{token_id}.parquet'}
              </code>
              .  Override via the <code>HOMERUN_PARQUET_ROOT</code> env var.
              {rootQuery.data && !rootQuery.data.exists && (
                <span className="ml-1 text-amber-400">
                  Directory does not exist yet — it will be created on first import.
                </span>
              )}
            </p>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1 text-[10px]"
            disabled={rescanMutation.isPending}
            onClick={() => rescanMutation.mutate()}
          >
            {rescanMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Search className="h-3 w-3" />
            )}
            Rescan
          </Button>
        </div>
        {lastReport && (
          <div className="mt-2 rounded border border-border/30 bg-muted/20 p-2 text-[10px] text-muted-foreground">
            Last rescan: {lastReport.groups_seen} group(s) found in{' '}
            {lastReport.elapsed_ms.toFixed(0)} ms.
            {lastReport.results.some((r) => r.error) && (
              <span className="ml-1 text-amber-400">
                {lastReport.results.filter((r) => r.error).length} group(s) errored.
              </span>
            )}
          </div>
        )}
      </div>

      {/* Catalog table */}
      <div className="flex-1 min-h-0 rounded-md border border-border/40 bg-card/40">
        <div className="border-b border-border/30 px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Discovered datasets ({datasets.length})
        </div>
        {datasetsQuery.isLoading ? (
          <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading…
          </div>
        ) : datasets.length === 0 ? (
          <div className="px-3 py-6 text-center text-[11px] text-muted-foreground">
            No parquet datasets discovered yet.  Drop files into the storage
            root above and hit Rescan.
          </div>
        ) : (
          <ScrollArea className="h-full">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-card/95 backdrop-blur">
                <tr className="border-b border-border/30 text-left text-[10px] uppercase tracking-wide text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Provider</th>
                  <th className="px-3 py-2 font-medium">Coin</th>
                  <th className="px-3 py-2 font-medium">Window</th>
                  <th className="px-3 py-2 text-right font-medium">Tokens</th>
                  <th className="px-3 py-2 text-right font-medium">Snapshots</th>
                  <th className="px-3 py-2 text-right font-medium">Trades</th>
                  <th className="px-3 py-2 font-medium">Last imported</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d: ParquetDataset) => (
                  <tr
                    key={d.id}
                    className="border-b border-border/20 hover:bg-muted/20"
                  >
                    <td className="px-3 py-1.5 font-mono">{d.provider}</td>
                    <td className="px-3 py-1.5 font-mono">{d.coin ?? '—'}</td>
                    <td className="px-3 py-1.5 text-muted-foreground">
                      {d.start_ts && d.end_ts
                        ? `${d.start_ts.slice(0, 10)} → ${d.end_ts.slice(0, 10)}`
                        : '—'}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums">
                      {d.token_count.toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums">
                      {d.snapshot_count.toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums">
                      {d.trade_count.toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 text-[10px] text-muted-foreground">
                      {d.last_imported_at
                        ? d.last_imported_at.replace('T', ' ').slice(0, 19)
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollArea>
        )}
      </div>
    </div>
  )
}
