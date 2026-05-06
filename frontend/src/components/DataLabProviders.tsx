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
  getProviderSettings,
  importPolybacktest,
  listImportJobs,
  listPolybacktestMarkets,
  listProviderDatasets,
  listProviders,
  updateProviderSettings,
  type ImportJob,
  type ImportJobStatus,
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

export default function DataLabProviders() {
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

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-violet-400" />
            <span className="text-sm font-semibold">External data providers</span>
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            Pull historical market data on demand from third-party vendors into
            your Data Lab. Imports land in the same microstructure tables
            backtests already read from.
          </p>
        </div>
      </div>

      {providersQuery.isLoading ? (
        <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading providers…
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
          Polymarket
        </button>
      </div>

      {isPolymarketTab ? (
        <PolymarketSection />
      ) : selected?.key === 'polybacktest' ? (
        <PolybacktestSection provider={selected} />
      ) : selected ? (
        <div className="rounded-md border border-border/40 bg-card/40 p-4 text-[11px] text-muted-foreground">
          {selected.label} integration not yet implemented in this build.
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
  if (!provider.configured) {
    return (
      <Badge variant="outline" className="gap-1 border-amber-500/40 text-amber-700 dark:text-amber-300">
        <CircleAlert className="h-3 w-3" />
        Needs API key
      </Badge>
    )
  }
  if (provider.health.ok === false) {
    return (
      <Badge variant="outline" className="gap-1 border-rose-500/40 text-rose-700 dark:text-rose-300">
        <AlertTriangle className="h-3 w-3" />
        Unreachable
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="gap-1 border-emerald-500/40 text-emerald-700 dark:text-emerald-300">
      <CheckCircle2 className="h-3 w-3" />
      Healthy
    </Badge>
  )
}


function PolybacktestSection({ provider }: { provider: ProviderInfo }) {
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
                Homepage <ExternalLink className="h-3 w-3" />
              </a>
              <a
                href={provider.docs_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 hover:text-foreground"
              >
                API docs <ExternalLink className="h-3 w-3" />
              </a>
              <span>Coins: {provider.supported_coins.join(', ')}</span>
            </div>
          </div>
        </div>
        {!provider.configured ? (
          <div className="mt-2 rounded-sm border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] text-amber-700 dark:text-amber-200">
            Add your polybacktest API key in <strong>Settings → Data Providers</strong> to enable import.
          </div>
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
        Provider settings
        <span className="ml-auto text-[10px] font-normal text-muted-foreground">
          {settings?.polybacktest_api_key_set ? 'configured' : 'not configured'}
        </span>
      </summary>
      <div className="mt-3 space-y-3">
        {/* API key */}
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">
            Polybacktest API key
          </Label>
          <div className="flex items-center gap-1">
            <Input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={settings?.polybacktest_api_key_set ? '(set — leave to keep)' : 'Paste API key'}
              className="h-8 font-mono text-xs"
            />
            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0"
              onClick={() => setShowKey((v) => !v)}
              title={showKey ? 'Hide' : 'Show'}
            >
              {showKey ? '🙈' : '👁'}
            </Button>
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            Get a key at <a href="https://polybacktest.com/dashboard" target="_blank" rel="noreferrer" className="underline">polybacktest.com/dashboard</a>.
            Empty value clears the key.
          </p>
        </div>

        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">Base URL (optional)</Label>
          <Input
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.polybacktest.com"
            className="h-8 font-mono text-xs"
          />
        </div>

        <div className="border-t border-border/30 pt-3">
          <div className="text-[11px] font-semibold mb-1">Reverse-engineer defaults</div>
          <p className="mb-2 text-[10px] text-muted-foreground">
            Default LLM model is set in <strong>AI → Models</strong> (under
            "Strategy Reverse-Engineer").
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label className="text-[10px] uppercase text-muted-foreground">Max iterations</Label>
              <Input
                value={maxIter}
                onChange={(e) => setMaxIter(e.target.value)}
                placeholder="10"
                className="h-8 text-xs"
              />
            </div>
            <div>
              <Label className="text-[10px] uppercase text-muted-foreground">Target score</Label>
              <Input
                value={targetScore}
                onChange={(e) => setTargetScore(e.target.value)}
                placeholder="0.7"
                className="h-8 text-xs"
              />
            </div>
            <div>
              <Label className="text-[10px] uppercase text-muted-foreground">Max cost (USD)</Label>
              <Input
                value={maxCost}
                onChange={(e) => setMaxCost(e.target.value)}
                placeholder="(no cap)"
                className="h-8 text-xs"
              />
            </div>
            <div className="col-span-2">
              <Label className="text-[10px] uppercase text-muted-foreground">Max wallet trades pulled</Label>
              <Input
                value={maxTrades}
                onChange={(e) => setMaxTrades(e.target.value)}
                placeholder="50000"
                className="h-8 text-xs"
              />
            </div>
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground">
            Empty fields fall back to <code className="font-mono">ai_default_model</code> + the
            service-level guards. No defaults baked into code.
          </p>
        </div>

        <div className="flex items-center justify-end gap-2">
          {saveMutation.isError ? (
            <span className="text-[10px] text-rose-700 dark:text-rose-300">
              {(saveMutation.error as Error)?.message || 'Save failed'}
            </span>
          ) : null}
          {saveMutation.isSuccess ? (
            <span className="text-[10px] text-emerald-700 dark:text-emerald-300">Saved</span>
          ) : null}
          <Button
            size="sm"
            className="h-7 text-[11px]"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending}
          >
            {saveMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Save'}
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
        <div className="text-xs font-semibold">Import historical data</div>
        <Badge variant="outline" className="text-[10px]">{selected.size} selected</Badge>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2">
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">Coin</Label>
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
          <Label className="text-[10px] uppercase text-muted-foreground">Horizon</Label>
          <Select value={marketType} onValueChange={(v) => setMarketType(v as MarketTypeFilter)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="text-xs">All horizons</SelectItem>
              {(['5m', '15m', '1h', '4h', '24h'] as const).map((mt) => (
                <SelectItem key={mt} value={mt} className="text-xs">{mt}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">Status</Label>
          <Select value={resolvedFilter} onValueChange={(v) => setResolvedFilter(v as ResolvedFilter)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="text-xs">All</SelectItem>
              <SelectItem value="resolved" className="text-xs">Resolved only</SelectItem>
              <SelectItem value="open" className="text-xs">Open only</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-[10px] uppercase text-muted-foreground">
            Fallback window (open markets only)
          </Label>
          <Select value={String(hours)} onValueChange={(v) => setHours(Number(v))}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {TIME_PRESETS.map((p) => (
                <SelectItem key={p.hours} value={String(p.hours)} className="text-xs">
                  Last {p.label}
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
          placeholder="Search markets…"
          className="h-8 text-xs"
        />
        <Button
          size="sm"
          variant="outline"
          className="h-8 gap-1 text-[10px]"
          onClick={() => setAppliedSearch(search)}
          disabled={marketsQuery.isFetching}
        >
          <Search className="h-3 w-3" /> Search
        </Button>
      </div>

      <ScrollArea className="mt-2 h-56 rounded-sm border border-border/30 bg-background/40">
        {marketsQuery.isLoading ? (
          <div className="flex h-full items-center justify-center text-[11px] text-muted-foreground">
            <Loader2 className="mr-2 h-3 w-3 animate-spin" /> Loading…
          </div>
        ) : marketsQuery.isError ? (
          <div className="p-3 text-[11px] text-rose-700 dark:text-rose-300">
            {String((marketsQuery.error as Error)?.message || 'Failed to load')}
          </div>
        ) : markets.length === 0 ? (
          <div className="p-3 text-[11px] text-muted-foreground">No markets found.</div>
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
          Always pulls full L2 depth (15 levels per side · UP + DOWN)
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
          Import {selected.size} market{selected.size !== 1 ? 's' : ''}
        </Button>
      </div>

      {importMutation.isError ? (
        <div className="mt-2 rounded-sm border border-rose-500/30 bg-rose-500/5 p-2 text-[10px] text-rose-700 dark:text-rose-300">
          {String((importMutation.error as Error)?.message || 'Import failed')}
        </div>
      ) : null}
      {importMutation.isSuccess ? (
        <div className="mt-2 rounded-sm border border-emerald-500/30 bg-emerald-500/5 p-2 text-[10px] text-emerald-700 dark:text-emerald-300">
          Job <span className="font-mono">{importMutation.data.id}</span> queued — watch progress in the Active jobs panel.
        </div>
      ) : null}
    </div>
  )
}


// ─── Active import jobs panel (auto-polling) ─────────────────────────

function PolybacktestActiveJobsPanel() {
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
        <div className="text-xs font-semibold">Active import jobs</div>
        <Badge variant="outline" className="text-[10px]">{jobs.length}</Badge>
      </div>
      <ScrollArea className="mt-2 max-h-72">
        {jobs.length === 0 ? (
          <div className="px-1 py-4 text-center text-[11px] text-muted-foreground">
            No imports yet — pick markets on the left to start.
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
            {coin.toUpperCase()} · {marketCount} market{marketCount !== 1 ? 's' : ''}
          </div>
          <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
            {job.message || job.error || `${job.snapshots_inserted.toLocaleString()} snapshots`}
          </div>
        </div>
        {isActive ? (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-rose-700 dark:text-rose-300 hover:bg-rose-500/10"
            onClick={onCancel}
            title="Cancel"
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
          <span>API calls: {job.api_calls.toLocaleString()}</span>
          <span>Snapshots inserted: {job.snapshots_inserted.toLocaleString()}</span>
          <span>Trades fetched: {job.trades_fetched.toLocaleString()}</span>
          {job.bytes_downloaded ? (
            <span>{(job.bytes_downloaded / 1024).toFixed(0)} KB</span>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}


// ─── Imported datasets panel ─────────────────────────────────────────

function PolybacktestDatasetsPanel() {
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
            Imported datasets
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            Available in the Backtest Studio dataset picker. Snapshot rows live
            in <span className="font-mono">market_microstructure_snapshots</span>.
          </p>
        </div>
        <Badge variant="outline" className="text-[10px]">{rows.length}</Badge>
      </div>

      <ScrollArea className="max-h-80">
        {rows.length === 0 ? (
          <div className="py-4 text-center text-[11px] text-muted-foreground">
            No datasets yet.
          </div>
        ) : (
          <table className="w-full text-[11px]">
            <thead className="text-[10px] uppercase text-muted-foreground">
              <tr className="border-b border-border/30">
                <th className="px-2 py-1.5 text-left">Provider</th>
                <th className="px-2 py-1.5 text-left">Coin</th>
                <th className="px-2 py-1.5 text-left">Market</th>
                <th className="px-2 py-1.5 text-right">Snapshots</th>
                <th className="px-2 py-1.5 text-right">Trades</th>
                <th className="px-2 py-1.5 text-left">Window</th>
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
                        if (confirm(`Delete dataset "${row.title || row.external_id}" and ${row.snapshot_count.toLocaleString()} snapshots?`)) {
                          deleteMutation.mutate(row.id)
                        }
                      }}
                      title="Delete"
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

const BACKFILL_INTERVALS: { value: string; label: string }[] = [
  { value: '1m', label: '1 minute' },
  { value: '1h', label: '1 hour' },
  { value: '6h', label: '6 hours' },
  { value: '1d', label: '1 day' },
  { value: 'max', label: 'max (full history)' },
]

const BACKFILL_SCOPE_OPTIONS: { value: BackfillScope; label: string; hint: string }[] = [
  { value: 'token', label: 'Specific tokens', hint: 'paste clob_token_ids' },
  { value: 'strategy', label: 'Strategy', hint: 'all tokens this strategy fired on' },
  { value: 'session', label: 'Recording session', hint: 'a session\'s target tokens' },
  { value: 'catalog_top_liquid', label: 'Top liquid catalog', hint: 'top N most-liquid markets' },
]

function PolymarketBackfillFlyout({ open, onClose }: { open: boolean; onClose: () => void }) {
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
    onError: (err) => setError((err as Error).message || 'backfill failed'),
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
      if (tokens.length === 0) { setError('Provide at least one token'); return }
      payload.target_values = tokens
    } else if (scope === 'strategy') {
      if (!strategySlug.trim()) { setError('Provide a strategy slug'); return }
      payload.strategy_slug = strategySlug.trim()
    } else if (scope === 'session') {
      if (!sessionId) { setError('Pick a session'); return }
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
              <div className="text-sm font-semibold leading-tight">REST backfill</div>
              <div className="text-[10px] text-muted-foreground leading-tight">
                Synthesize book snapshots from Polymarket /prices-history (mid only).
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
              <div className="font-medium">Synthetic data caveat</div>
              <div className="mt-1 text-amber-700/90 dark:text-amber-300/90">
                REST gives mid prices only — best_bid/best_ask are reconstructed by centering on
                the mid with a configurable spread.  Sizes are zero (no depth).  Each row is
                tagged <code className="font-mono">payload_json.synthetic = true</code> so the
                backtester / Cox PH trainer can filter or downweight.
              </div>
            </div>

            {/* Scope */}
            <div className="space-y-2 rounded-md border border-border/40 bg-card/30 p-3">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Scope</div>
              <div className="grid grid-cols-2 gap-1.5">
                {BACKFILL_SCOPE_OPTIONS.map((o) => {
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
                      <div className="text-[11px] font-medium">{o.label}</div>
                      <div className="text-[9px] text-muted-foreground/80">{o.hint}</div>
                    </button>
                  )
                })}
              </div>

              {scope === 'token' ? (
                <div className="space-y-1">
                  <Label className="text-[10px]">Token IDs (one per line / comma-separated)</Label>
                  <textarea
                    value={tokenText}
                    onChange={(e) => setTokenText(e.target.value)}
                    placeholder="0x..."
                    className="min-h-[80px] w-full rounded-sm border border-border/40 bg-background/60 px-2 py-1.5 font-mono text-[11px]"
                  />
                </div>
              ) : null}

              {scope === 'strategy' ? (
                <div className="space-y-1">
                  <Label className="text-[10px]">Strategy slug</Label>
                  <Input
                    value={strategySlug}
                    onChange={(e) => setStrategySlug(e.target.value)}
                    placeholder="tail_end_carry"
                    className="h-8 text-[12px]"
                  />
                  <div className="text-[10px] text-muted-foreground">
                    Pulls every distinct token from this strategy's OpportunityHistory rows in
                    the time window.
                  </div>
                </div>
              ) : null}

              {scope === 'session' ? (
                <div className="space-y-1">
                  <Label className="text-[10px]">Recording session</Label>
                  <select
                    value={sessionId}
                    onChange={(e) => setSessionId(e.target.value)}
                    className="h-8 w-full rounded-md border border-input bg-background px-3 text-[11px]"
                  >
                    <option value="">— pick a session —</option>
                    {(sessionsQuery.data ?? []).map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name} · {s.status} · {s.target_token_ids.length} tokens
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}

              {scope === 'catalog_top_liquid' ? (
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-[10px]">Cap (tokens)</Label>
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
                    <Label className="text-[10px]">Min liquidity ($)</Label>
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
                Window + cadence
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px]">Lookback (days, blank = use range)</Label>
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
                  <Label className="text-[10px]">Interval (fidelity)</Label>
                  <select
                    value={interval}
                    onChange={(e) => setInterval(e.target.value)}
                    className="h-8 w-full rounded-md border border-input bg-background px-3 text-[11px]"
                  >
                    {BACKFILL_INTERVALS.map((i) => (
                      <option key={i.value} value={i.value}>{i.label}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Start (overrides lookback)</Label>
                  <Input
                    type="datetime-local"
                    value={startInput}
                    onChange={(e) => setStartInput(e.target.value)}
                    className="h-8 text-[12px]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">End</Label>
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
                Synth + caps
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px]">Synthetic spread (bps)</Label>
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
                  <Label className="text-[10px]">Max tokens (cap)</Label>
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
                <div className="font-medium text-emerald-700 dark:text-emerald-300">Backfill complete</div>
                <div className="mt-1 grid grid-cols-2 gap-1 text-emerald-700 dark:text-emerald-300">
                  <div>job: <span className="font-mono">{result.job_id}</span></div>
                  <div>duration: {result.duration_seconds.toFixed(1)}s</div>
                  <div>tokens targeted: <strong>{result.target_token_count.toLocaleString()}</strong></div>
                  <div>tokens with data: {result.tokens_with_data.toLocaleString()}</div>
                  <div>rows inserted: <strong>{result.rows_inserted_total.toLocaleString()}</strong></div>
                  <div>points fetched: {result.points_fetched_total.toLocaleString()}</div>
                  <div>existing skipped: {result.skipped_existing_total.toLocaleString()}</div>
                  <div>errors: {result.tokens_with_errors}</div>
                </div>
                {result.tokens_with_errors > 0 ? (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-rose-700 dark:text-rose-300">
                      Failed tokens ({result.tokens_with_errors})
                    </summary>
                    <div className="mt-1 max-h-[140px] space-y-0.5 overflow-y-auto">
                      {result.per_token.filter((t) => t.error).slice(0, 50).map((t) => (
                        <div key={t.token_id} className="font-mono text-[10px] text-rose-700 dark:text-rose-300/90">
                          {t.token_id.slice(0, 14)} — {t.error}
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
            Idempotent — already-recorded seconds are skipped.
          </span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-8 text-[11px]" onClick={onClose}>
              Close
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
              Run backfill
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}

function PolymarketSection() {
  const [open, setOpen] = useState(false)
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="rounded-md border border-border/40 bg-card/40 p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold">Polymarket</span>
              <Badge variant="outline" className="gap-1 border-emerald-500/40 text-emerald-700 dark:text-emerald-300">
                <CheckCircle2 className="h-3 w-3" />
                Built-in
              </Badge>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Synthesize historical book snapshots from Polymarket's /prices-history
              endpoint to fill gaps the live WebSocket recorder didn't reach.  Inserts
              into the same MarketMicrostructureSnapshot table the recorder writes to,
              tagged synthetic so the backtester can downweight.
            </p>
          </div>
        </div>
      </div>

      <div className="rounded-md border border-border/40 bg-card/30">
        <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
          <div className="flex items-center gap-2">
            <Download className="h-3.5 w-3.5 rotate-180 text-violet-700 dark:text-violet-300" />
            <span className="text-xs font-semibold">REST backfill</span>
            <span className="text-[10px] text-muted-foreground">
              fill historical gaps via Polymarket /prices-history
            </span>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 text-[10px]"
            onClick={() => setOpen(true)}
          >
            <Download className="h-3 w-3 rotate-180" />
            New backfill
          </Button>
        </div>
        <div className="px-3 py-2 text-[10px] text-muted-foreground">
          Use when WS coverage doesn't reach back far enough.  Pick a scope (token /
          strategy / recording session / top-liquid catalog), a window, and a fidelity —
          the service synthesizes book snapshots centered on each mid price and inserts
          them into the same MarketMicrostructureSnapshot table the live recorder writes
          to.  Synthetic rows carry a flag the backtester can read.
        </div>
      </div>
      <PolymarketBackfillFlyout open={open} onClose={() => setOpen(false)} />
    </div>
  )
}
