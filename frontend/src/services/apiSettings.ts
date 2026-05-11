import { api, unwrapApiData } from './apiClient'

// ==================== SETTINGS ====================

export interface PolymarketSettings {
  api_key: string | null
  api_secret: string | null
  api_passphrase: string | null
  private_key: string | null
}

export interface KalshiSettings {
  email: string | null
  password: string | null
  api_key: string | null
}

export interface OracleSettings {
  // Chainlink Data Streams direct REST polling. Used as a parallel oracle
  // source alongside the RTDS-relayed Chainlink prices — when both sides
  // are configured the source-comparison panel surfaces relay-vs-direct
  // delta. Free-tier creds available at https://chain.link/data-streams.
  chainlink_direct_api_key: string | null
  chainlink_direct_user_secret: string | null
}

export interface LLMSettings {
  provider: string
  openai_api_key: string | null
  anthropic_api_key: string | null
  google_api_key: string | null
  xai_api_key: string | null
  deepseek_api_key: string | null
  ollama_api_key: string | null
  ollama_base_url: string | null
  lmstudio_api_key: string | null
  lmstudio_base_url: string | null
  openrouter_api_key: string | null
  openrouter_base_url: string | null
  nvidia_api_key: string | null
  nvidia_base_url: string | null
  model: string | null
  max_monthly_spend: number | null
  model_assignments: Record<string, string>
  enabled_features: Record<string, boolean>
}

export interface NotificationSettings {
  enabled: boolean
  telegram_bot_token: string | null
  telegram_chat_id: string | null
  notify_on_opportunity: boolean
  notify_on_trade: boolean
  notify_min_roi: number
  notify_autotrader_orders: boolean
  notify_autotrader_closes: boolean
  notify_autotrader_issues: boolean
  notify_autotrader_timeline: boolean
  notify_autotrader_summary_interval_minutes: number
  notify_autotrader_summary_per_trader: boolean
}

export interface ScannerSettings {
  scan_interval_seconds: number
  min_profit_threshold: number
  max_markets_to_scan: number
  max_events_to_scan: number
  market_fetch_page_size: number
  market_fetch_order: string
  min_liquidity: number
  max_opportunities_total: number
  max_opportunities_per_strategy: number
  skipped_signal_reactivation_cooldown_seconds: number
  strict_ws_max_age_ms: number
  market_filter_tags: string[]
  crypto_lane_enabled: boolean
}

export interface MarketFilterAvailableTag {
  name: string
  last_seen: string | null
  occurrences: number
}

export interface MarketFilterAvailableTagsResponse {
  tags: MarketFilterAvailableTag[]
  total: number
}

export interface DiscoverySettings {
  max_discovered_wallets: number
  maintenance_enabled: boolean
  keep_recent_trade_days: number
  keep_new_discoveries_days: number
  maintenance_batch: number
  stale_analysis_hours: number
  analysis_priority_batch_limit: number
  delay_between_markets: number
  delay_between_wallets: number
  max_markets_per_run: number
  max_wallets_per_market: number
  trader_opps_source_filter: 'all' | 'tracked' | 'pool'
  trader_opps_min_tier: 'WATCH' | 'HIGH' | 'EXTREME'
  trader_opps_side_filter: 'all' | 'buy' | 'sell'
  trader_opps_confluence_limit: number
  trader_opps_insider_limit: number
  trader_opps_insider_min_confidence: number
  trader_opps_insider_max_age_minutes: number
  pool_recompute_mode: 'quality_only' | 'balanced'
  pool_target_size: number
  pool_min_size: number
  pool_max_size: number
  pool_active_window_hours: number
  pool_inactive_rising_retention_hours: number
  pool_selection_score_floor: number
  pool_max_hourly_replacement_rate: number
  pool_replacement_score_cutoff: number
  pool_max_cluster_share: number
  pool_high_conviction_threshold: number
  pool_insider_priority_threshold: number
  pool_min_eligible_trades: number
  pool_max_eligible_anomaly: number
  pool_core_min_win_rate: number
  pool_core_min_sharpe: number
  pool_core_min_profit_factor: number
  pool_rising_min_win_rate: number
  pool_slo_min_analyzed_pct: number
  pool_slo_min_profitable_pct: number
  pool_leaderboard_wallet_trade_sample: number
  pool_incremental_wallet_trade_sample: number
  pool_full_sweep_interval_seconds: number
  pool_incremental_refresh_interval_seconds: number
  pool_activity_reconciliation_interval_seconds: number
  pool_recompute_interval_seconds: number
}

export interface LiveExecutionSettingsConfig {
  max_trade_size_usd: number
  max_daily_trade_volume: number
  max_slippage_percent: number
  min_account_balance_usd: number
}

export interface MaintenanceSettings {
  auto_cleanup_enabled: boolean
  cleanup_interval_hours: number
  cleanup_resolved_trade_days: number
  cleanup_trade_signal_emission_days: number
  cleanup_trade_signal_update_days: number
  cleanup_wallet_activity_rollup_days: number
  cleanup_wallet_activity_dedupe_enabled: boolean
  llm_usage_retention_days: number
  trader_events_firehose_retention_days: number
  trader_events_other_retention_days: number
  market_cache_hygiene_enabled: boolean
  market_cache_hygiene_interval_hours: number
  market_cache_retention_days: number
  market_cache_reference_lookback_days: number
  market_cache_weak_entry_grace_days: number
  market_cache_max_entries_per_slug: number
}

export interface TradingProxySettings {
  enabled: boolean
  proxy_url: string | null
  verify_ssl: boolean
  timeout: number
  require_vpn: boolean
}

export interface BlockchainRpcSettings {
  // Both URLs are masked previews on GET; on PUT, send the full URL or
  // empty string to clear. ``null``/undefined leaves the field
  // unchanged on the server.
  rpc_url: string | null
  ws_url: string | null
}

export interface UILockSettings {
  enabled: boolean
  idle_timeout_minutes: number
  has_password: boolean
  password?: string | null
  clear_password?: boolean
}

export interface EventsSettings {
  enabled: boolean
  interval_seconds: number
}

export interface SearchFilterSettings {
  // Search platform toggles & limits
  search_polymarket_enabled: boolean
  search_kalshi_enabled: boolean
  search_max_results: number
  // Web search provider API keys (masked)
  serpapi_key: string | null
  brave_search_key: string | null
  // Hard rejection filters
  min_liquidity_hard: number
  min_position_size: number
  min_absolute_profit: number
  min_annualized_roi: number
  max_resolution_months: number
  max_plausible_roi: number
  max_trade_legs: number
  min_liquidity_per_leg: number
  // Risk scoring
  risk_very_short_days: number
  risk_short_days: number
  risk_long_lockup_days: number
  risk_extended_lockup_days: number
  risk_low_liquidity: number
  risk_moderate_liquidity: number
  risk_complex_legs: number
  risk_multiple_legs: number
  // BTC/ETH high-frequency
  btc_eth_hf_enabled: boolean
  btc_eth_hf_maker_mode: boolean
  btc_eth_hf_series_btc_15m: string
  btc_eth_hf_series_eth_15m: string
  btc_eth_hf_series_sol_15m: string
  btc_eth_hf_series_xrp_15m: string
  btc_eth_hf_series_btc_5m: string
  btc_eth_hf_series_eth_5m: string
  btc_eth_hf_series_sol_5m: string
  btc_eth_hf_series_xrp_5m: string
  btc_eth_hf_series_btc_1h: string
  btc_eth_hf_series_eth_1h: string
  btc_eth_hf_series_sol_1h: string
  btc_eth_hf_series_xrp_1h: string
  btc_eth_hf_series_btc_4h: string
  btc_eth_hf_series_eth_4h: string
  btc_eth_hf_series_sol_4h: string
  btc_eth_hf_series_xrp_4h: string
  btc_eth_pure_arb_max_combined: number
  btc_eth_dump_hedge_drop_pct: number
  btc_eth_thin_liquidity_usd: number
}

export interface NetworkSettings {
  allow_network_access: boolean
}

export interface AllSettings {
  polymarket: PolymarketSettings
  kalshi: KalshiSettings
  oracle: OracleSettings
  llm: LLMSettings
  notifications: NotificationSettings
  scanner: ScannerSettings
  live_execution: LiveExecutionSettingsConfig
  maintenance: MaintenanceSettings
  discovery: DiscoverySettings
  trading_proxy: TradingProxySettings
  blockchain_rpc: BlockchainRpcSettings
  ui_lock: UILockSettings
  network: NetworkSettings
  events: EventsSettings
  search_filters: SearchFilterSettings
  updated_at: string | null
}

export interface UpdateSettingsRequest {
  polymarket?: Partial<PolymarketSettings>
  kalshi?: Partial<KalshiSettings>
  oracle?: Partial<OracleSettings>
  llm?: Partial<LLMSettings>
  notifications?: Partial<NotificationSettings>
  scanner?: Partial<ScannerSettings>
  live_execution?: Partial<LiveExecutionSettingsConfig>
  maintenance?: Partial<MaintenanceSettings>
  discovery?: Partial<DiscoverySettings>
  trading_proxy?: Partial<TradingProxySettings>
  blockchain_rpc?: Partial<BlockchainRpcSettings>
  ui_lock?: Partial<UILockSettings>
  network?: Partial<NetworkSettings>
  events?: Partial<EventsSettings>
  search_filters?: Partial<SearchFilterSettings>
}

export type SettingsTransferCategory =
  | 'bot_traders'
  | 'strategies'
  | 'data_sources'
  | 'market_credentials'
  | 'vpn_configuration'
  | 'llm_configuration'
  | 'telegram_configuration'

export interface BotTraderLiveWalletStateBundle {
  runtime_states?: Array<Record<string, unknown>>
  orders?: Array<Record<string, unknown>>
  positions?: Array<Record<string, unknown>>
}

export interface BotTraderTradeStateBundle {
  orders: Array<Record<string, unknown>>
  live_wallet_state?: BotTraderLiveWalletStateBundle
}

export interface BotTradersBundle {
  traders: Array<Record<string, unknown>>
  orchestrator?: Record<string, unknown> | null
  trade_state?: BotTraderTradeStateBundle
}

export interface SettingsExportBundle {
  schema_version: number
  exported_at: string
  categories: SettingsTransferCategory[]
  bot_traders?: BotTradersBundle
  strategies?: Array<Record<string, unknown>>
  data_sources?: Array<Record<string, unknown>>
  market_credentials?: Record<string, unknown>
  vpn_configuration?: Record<string, unknown>
  llm_configuration?: Record<string, unknown>
  telegram_configuration?: Record<string, unknown>
}

export interface ExportSettingsRequest {
  include_categories?: SettingsTransferCategory[]
}

export interface ExportSettingsResponse {
  status: string
  bundle: SettingsExportBundle
  counts: Record<string, number>
}

export interface ImportSettingsRequest {
  bundle: Record<string, unknown>
  include_categories?: SettingsTransferCategory[]
}

export interface ImportSettingsResponse {
  status: string
  imported_categories: SettingsTransferCategory[]
  results: Record<string, Record<string, number>>
  imported_at: string
}

export const getSettings = async (): Promise<AllSettings> => {
  const { data } = await api.get('/settings')
  return unwrapApiData(data)
}

export const updateSettings = async (settings: UpdateSettingsRequest): Promise<{ status: string; message: string; updated_at: string }> => {
  const { data } = await api.put('/settings', settings)
  return unwrapApiData(data)
}

export const exportSettingsBundle = async (request: ExportSettingsRequest): Promise<ExportSettingsResponse> => {
  const { data } = await api.post('/settings/export', request)
  return unwrapApiData(data)
}

export const importSettingsBundle = async (request: ImportSettingsRequest): Promise<ImportSettingsResponse> => {
  const { data } = await api.post('/settings/import', request)
  return unwrapApiData(data)
}

export interface UILockStatus {
  enabled: boolean
  locked: boolean
  idle_timeout_minutes: number
  has_password: boolean
}

export const getUILockStatus = async (): Promise<UILockStatus> => {
  const { data } = await api.get('/ui-lock/status')
  return unwrapApiData(data)
}

export const unlockUILock = async (password: string): Promise<{ status: string; message: string; locked: boolean }> => {
  const { data } = await api.post('/ui-lock/unlock', { password })
  return unwrapApiData(data)
}

export const lockUILock = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/ui-lock/lock')
  return unwrapApiData(data)
}

export const sendUILockActivity = async (): Promise<{ status: string }> => {
  const { data } = await api.post('/ui-lock/activity')
  return unwrapApiData(data)
}

export const updatePolymarketSettings = async (settings: Partial<PolymarketSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/polymarket', settings)
  return unwrapApiData(data)
}

export const updateLLMSettings = async (settings: Partial<LLMSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/llm', settings)
  return unwrapApiData(data)
}

export const updateNotificationSettings = async (settings: Partial<NotificationSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/notifications', settings)
  return unwrapApiData(data)
}

export const updateScannerSettings = async (settings: Partial<ScannerSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/scanner', settings)
  return unwrapApiData(data)
}

export const getMarketFilterAvailableTags = async (): Promise<MarketFilterAvailableTagsResponse> => {
  const { data } = await api.get('/settings/market-filter/available-tags')
  return unwrapApiData(data)
}

export const updateLiveExecutionSettings = async (
  settings: Partial<LiveExecutionSettingsConfig>
): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/live-execution', settings)
  return unwrapApiData(data)
}

export const updateMaintenanceSettings = async (settings: Partial<MaintenanceSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/maintenance', settings)
  return unwrapApiData(data)
}

export const getDiscoverySettings = async (): Promise<DiscoverySettings> => {
  const { data } = await api.get('/settings/discovery')
  return unwrapApiData(data)
}

export const updateDiscoverySettings = async (settings: Partial<DiscoverySettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/discovery', settings)
  return unwrapApiData(data)
}

export interface LLMModelOption {
  id: string
  name: string
}

export interface LLMModelsResponse {
  models: Record<string, LLMModelOption[]>
}

export interface LLMTestResponse {
  status: string
  message: string
  provider?: string
  model_count?: number
}

export interface RefreshModelsResponse {
  status: string
  message: string
  models: Record<string, LLMModelOption[]>
}

export const getLLMModels = async (provider?: string): Promise<LLMModelsResponse> => {
  const { data } = await api.get('/settings/llm/models', { params: provider ? { provider } : undefined })
  return unwrapApiData(data)
}

export const refreshLLMModels = async (provider?: string): Promise<RefreshModelsResponse> => {
  const { data } = await api.post('/settings/llm/models/refresh', null, { params: provider ? { provider } : undefined })
  return unwrapApiData(data)
}

export const testLLMConnection = async (provider?: string): Promise<LLMTestResponse> => {
  const { data } = await api.post('/settings/test/llm', null, { params: provider ? { provider } : undefined })
  return unwrapApiData(data)
}

export const testPolymarketConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/polymarket')
  return unwrapApiData(data)
}

export const testKalshiConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/kalshi')
  return unwrapApiData(data)
}

export const testTelegramConnection = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/settings/test/telegram')
  return unwrapApiData(data)
}

export const testTradingProxy = async (): Promise<{ status: string; message: string; proxy_enabled?: boolean; proxy_ip?: string; direct_ip?: string; vpn_active?: boolean }> => {
  const { data } = await api.post('/settings/test/trading-proxy')
  return unwrapApiData(data)
}

// ==================== VALIDATION / BACKTESTING ====================

export interface ValidationSummaryMetric {
  sample_size?: number
  expected_roi_mean?: number | null
  actual_roi_mean?: number | null
  mae_roi?: number | null
  rmse_roi?: number | null
  directional_accuracy?: number | null
  optimism_bias_roi?: number | null
}

export interface ValidationOverview {
  current_params: Record<string, unknown>
  active_parameter_set: Record<string, unknown> | null
  parameter_spec_count: number
  parameter_set_count: number
  latest_optimization: Record<string, unknown> | null
  opportunity_stats: Record<string, unknown>
  strategy_accuracy: Record<string, unknown>
  roi_30d: Record<string, unknown>
  decay_30d: Record<string, unknown>
  calibration_90d: {
    window_days: number
    sample_size: number
    overall: ValidationSummaryMetric
    by_strategy: Record<string, ValidationSummaryMetric>
  }
  calibration_trend_90d: Array<{
    bucket_start: string
    sample_size: number
    mae_roi: number
    directional_accuracy: number
  }>
  strategy_health: ValidationStrategyHealth[]
  guardrail_config: ValidationGuardrailConfig
  trader_orchestrator_execution_30d?: {
    window_days: number
    sample_size: number
    executed_or_open: number
    failed: number
    failure_rate: number
    notional_total_usd: number
    realized_pnl_total: number
    by_source: Array<Record<string, unknown>>
    by_strategy: Array<Record<string, unknown>>
  }
  events_resolver_7d?: {
    window_days: number
    signals_sampled: number
    candidates: number
    tradable: number
    tradable_rate: number
    by_signal_type: Array<Record<string, unknown>>
  }
  jobs: ValidationJob[]
}

export interface ValidationJob {
  id: string
  job_type: 'backtest' | 'optimize' | 'execution_simulation' | 'live_truth_monitor' | string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string
  payload?: Record<string, unknown>
  result?: Record<string, unknown>
  error?: string | null
  progress?: number
  message?: string | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
}

export interface LiveTruthMonitorJobRequest {
  trader_id?: string
  trader_name?: string
  duration_seconds?: number
  poll_seconds?: number
  run_llm_analysis?: boolean
  llm_model?: string
  include_strategy_source?: boolean
  max_alerts_for_llm?: number
  enable_provider_checks?: boolean
}

export interface LiveTruthMonitorReportPayload {
  path: string
  line_count: number
  alert_count: number
  heartbeat_count: number
  transition_count: number
  alerts_by_rule: Record<string, number>
  alert_samples: Array<Record<string, unknown>>
}

export interface LiveTruthMonitorRawResponse {
  job_id: string
  job_status: ValidationJob['status']
  payload: Record<string, unknown>
  monitor: {
    summary: Record<string, unknown>
    summary_path: string
    report_path: string
    stdout_events: Array<Record<string, unknown>>
    report: LiveTruthMonitorReportPayload
  }
  llm_analysis: Record<string, unknown>
}

export type LiveTruthMonitorArtifact = 'summary_json' | 'report_jsonl' | 'llm_analysis_json' | 'bundle_json'

export interface ValidationJobEnqueueResponse {
  status: string
  job_id: string
}

export interface ValidationGuardrailConfig {
  enabled: boolean
  min_samples: number
  min_directional_accuracy: number
  max_mae_roi: number
  lookback_days: number
  auto_promote: boolean
}

export interface ValidationStrategyHealth {
  strategy_type: string
  status: 'active' | 'demoted' | string
  sample_size: number
  directional_accuracy?: number | null
  mae_roi?: number | null
  rmse_roi?: number | null
  optimism_bias_roi?: number | null
  last_reason?: string | null
  manual_override?: boolean
  manual_override_note?: string | null
  demoted_at?: string | null
  restored_at?: string | null
  updated_at?: string | null
}

export const getValidationOverview = async (): Promise<ValidationOverview> => {
  const { data } = await api.get('/validation/overview')
  return unwrapApiData(data)
}

export const enqueueLiveTruthMonitorJob = async (
  payload: LiveTruthMonitorJobRequest
): Promise<ValidationJobEnqueueResponse> => {
  const { data } = await api.post('/validation/jobs/live-truth-monitor', payload)
  return unwrapApiData(data)
}

export const getValidationJob = async (jobId: string): Promise<ValidationJob> => {
  const { data } = await api.get(`/validation/jobs/${jobId}`)
  return unwrapApiData(data)
}

export const getLiveTruthMonitorRaw = async (
  jobId: string,
  params?: { max_alerts?: number }
): Promise<LiveTruthMonitorRawResponse> => {
  const { data } = await api.get(`/validation/jobs/${jobId}/live-truth-monitor/raw`, { params })
  return unwrapApiData(data)
}

function resolveExportFilename(contentDisposition: unknown, fallback: string): string {
  const header = String(contentDisposition || '').trim()
  if (!header) return fallback
  const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i)
  if (utf8Match && utf8Match[1]) {
    try {
      return decodeURIComponent(utf8Match[1].trim().replace(/^"|"$/g, '')) || fallback
    } catch {
      return utf8Match[1].trim().replace(/^"|"$/g, '') || fallback
    }
  }
  const plainMatch = header.match(/filename=([^;]+)/i)
  if (!plainMatch || !plainMatch[1]) return fallback
  return plainMatch[1].trim().replace(/^"|"$/g, '') || fallback
}

export const exportLiveTruthMonitorArtifact = async (
  jobId: string,
  artifact: LiveTruthMonitorArtifact
): Promise<{ filename: string; mediaType: string; blob: Blob }> => {
  const response = await api.get(`/validation/jobs/${jobId}/live-truth-monitor/export`, {
    params: { artifact },
    responseType: 'blob',
  })
  const mediaType = String(response.headers['content-type'] || 'application/octet-stream')
  const fallback = `live_truth_monitor_${jobId.slice(0, 8)}_${artifact}.json`
  const filename = resolveExportFilename(response.headers['content-disposition'], fallback)
  return {
    filename,
    mediaType,
    blob: response.data as Blob,
  }
}

export const overrideValidationStrategy = async (
  strategyType: string,
  status: 'active' | 'demoted',
  note?: string
): Promise<Record<string, unknown>> => {
  const { data } = await api.post(`/validation/strategy-health/${strategyType}/override`, null, {
    params: { status, note }
  })
  return unwrapApiData(data)
}

export const clearValidationStrategyOverride = async (strategyType: string): Promise<Record<string, unknown>> => {
  const { data } = await api.delete(`/validation/strategy-health/${strategyType}/override`)
  return unwrapApiData(data)
}

// ── Strategy health (validation guardrail) ──────────────────────────────

export interface StrategyHealthRow {
  strategy_type: string
  status: 'active' | 'demoted' | 'untracked' | string
  manual_override?: boolean
  manual_override_note?: string | null
  sample_size?: number
  directional_accuracy?: number | null
  mae_roi?: number | null
  last_reason?: string | null
  updated_at?: string | null
}

export const getValidationStrategyHealth = async (): Promise<StrategyHealthRow[]> => {
  const { data } = await api.get('/validation/strategy-health')
  const payload = unwrapApiData(data) as Record<string, unknown> | undefined
  const rows = (payload?.strategy_health || []) as StrategyHealthRow[]
  return rows
}

export interface GuardrailConfig {
  enabled: boolean
  min_samples: number
  min_directional_accuracy: number
  max_mae_roi: number
  lookback_days: number
  auto_promote: boolean
}

export const getValidationGuardrailConfig = async (): Promise<GuardrailConfig> => {
  const { data } = await api.get('/validation/guardrails/config')
  return unwrapApiData(data)
}

export const updateValidationGuardrailConfig = async (
  patch: Partial<GuardrailConfig>,
): Promise<GuardrailConfig> => {
  const { data } = await api.put('/validation/guardrails/config', patch)
  return unwrapApiData(data)
}

export interface GuardrailEvalResult {
  enabled?: boolean
  updated?: number
  demoted?: string[]
  restored?: string[]
}

export const runValidationGuardrails = async (): Promise<GuardrailEvalResult> => {
  const { data } = await api.post('/validation/guardrails/evaluate')
  return unwrapApiData(data)
}
