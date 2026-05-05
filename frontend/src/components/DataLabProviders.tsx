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


const COINS = ['btc', 'eth', 'sol'] as const

const TIME_PRESETS: Array<{ label: string; hours: number }> = [
  { label: '24 h', hours: 24 },
  { label: '3 d', hours: 24 * 3 },
  { label: '7 d', hours: 24 * 7 },
  { label: '30 d', hours: 24 * 30 },
]


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

      {/* Provider sub-tabs — one pill per integrated provider.
          Today only Polybacktest is wired; the layout stays the same
          when we add Kaiko, Tardis, etc. */}
      {providers.length > 0 ? (
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
        </div>
      ) : null}

      {selected?.key === 'polybacktest' ? (
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
