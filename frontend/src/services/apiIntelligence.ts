import { api, getStrategyManagerItems, unwrapApiData, unwrapStrategyManagerPayload } from './apiClient'
import type { Opportunity } from './apiCore'
import type { KalshiSettings } from './apiSettings'
import type { Trader } from './apiTraders'

// ==================== AI INTELLIGENCE ====================

// AI endpoints that invoke LLM calls need a longer timeout than the default 15s
const AI_TIMEOUT = { timeout: 120_000 }

export const getAIStatus = () => api.get('/ai/status')
export const analyzeResolution = (data: any) => api.post('/ai/resolution/analyze', data, AI_TIMEOUT)
export const getResolutionAnalysis = (marketId: string) => api.get(`/ai/resolution/${marketId}`)
export const judgeOpportunity = (data: any) => api.post('/ai/judge/opportunity', data, AI_TIMEOUT)
export const judgeOpportunitiesBulk = (data?: { opportunity_ids?: string[]; force?: boolean }) =>
  api.post('/ai/judge/opportunities/bulk', data || {}, AI_TIMEOUT)
export const getJudgmentHistory = (params?: any) => api.get('/ai/judge/history', { params })
export const getAgreementStats = () => api.get('/ai/judge/agreement-stats')
export const analyzeMarket = (data: any) => api.post('/ai/market/analyze', data, AI_TIMEOUT)
export const analyzeNewsSentiment = (data: any) => api.post('/ai/news/sentiment', data, AI_TIMEOUT)

// ==================== NEWS INTELLIGENCE ====================

export interface NewsArticle {
  article_id: string
  title: string
  source: string
  feed_source: string
  url: string
  published: string | null
  category: string
  summary: string
  has_embedding: boolean
  fetched_at: string
}

export interface NewsFeedStatus {
  article_count: number
  sources: Record<string, number>
  running: boolean
}

export const getNewsFeedStatus = async (): Promise<NewsFeedStatus> => {
  const { data } = await api.get('/news/feed/status')
  return unwrapApiData(data)
}

export const triggerNewsFetch = async (): Promise<{ new_articles: number; total_articles: number; articles: Array<{ title: string; source: string; feed_source: string; url: string; published: string | null; category: string }> }> => {
  const { data } = await api.post('/news/feed/fetch')
  return unwrapApiData(data)
}

export const getNewsArticles = async (params?: {
  max_age_hours?: number
  source?: string
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; has_more: boolean; articles: NewsArticle[] }> => {
  const { data } = await api.get('/news/feed/articles', { params })
  return unwrapApiData(data)
}

export const searchNewsArticles = async (params: {
  q: string
  max_age_hours?: number
  limit?: number
}): Promise<{ query: string; total: number; articles: NewsArticle[] }> => {
  const { data } = await api.get('/news/feed/search', { params })
  return unwrapApiData(data)
}

export const clearNewsArticles = async (): Promise<{ cleared: number }> => {
  const { data } = await api.delete('/news/feed/clear')
  return unwrapApiData(data)
}

export const listSkills = () => api.get('/ai/skills')
export const executeSkill = (data: any) => api.post('/ai/skills/execute', data, AI_TIMEOUT)
export const getResearchSessions = (params?: any) => api.get('/ai/sessions', { params })
export const getResearchSession = (sessionId: string) => api.get(`/ai/sessions/${sessionId}`)
export const getAIUsage = () => api.get('/ai/usage')

// ==================== AI DEEP INTEGRATION ====================

export interface MarketSearchResult {
  market_id: string
  question: string
  yes_price: number | null
  no_price: number | null
  liquidity: number | null
  event_title: string | null
  category: string | null
}

export const searchMarkets = async (q: string, limit = 10): Promise<{ results: MarketSearchResult[]; total: number }> => {
  const { data } = await api.get('/ai/markets/search', { params: { q, limit } })
  return unwrapApiData(data)
}

export interface OpportunityAISummary {
  opportunity_id: string
  judgment: {
    overall_score: number
    profit_viability: number
    resolution_safety: number
    execution_feasibility: number
    market_efficiency: number
    recommendation: string
    reasoning: string
  } | null
  resolution_analyses: Array<{
    market_id: string
    clarity_score: number
    risk_score: number
    confidence: number
    recommendation: string
    summary: string
    ambiguities: string[]
    edge_cases: string[]
  }>
}

export const getOpportunityAISummary = async (opportunityId: string): Promise<OpportunityAISummary> => {
  const { data } = await api.get(`/ai/opportunity/${opportunityId}/summary`)
  return unwrapApiData(data)
}

export interface AIChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AIChatResponse {
  session_id: string
  response: string
  model: string
  tokens_used: Record<string, number>
  actions_applied?: Array<{
    type: string
    strategy_id?: string
    data_source_id?: string
    slug?: string
    version?: number
    status?: string
  }>
  action_errors?: Array<{
    index?: number
    type?: string
    error: string
  }>
}

export const sendAIChat = async (params: {
  message: string
  session_id?: string
  context_type?: string
  context_id?: string
  history?: AIChatMessage[]
  allow_actions?: boolean
}): Promise<AIChatResponse> => {
  const { data } = await api.post('/ai/chat', params, AI_TIMEOUT)
  return unwrapApiData(data)
}

export interface AIStrategyDraftGenerationResponse {
  name: string
  slug: string
  source_key: string
  description: string
  source_code: string
  class_name: string
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  aliases: string[]
  validation: {
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
    capabilities: Record<string, boolean>
    inferred_type: string
  }
  used_repair_pass: boolean
  model: string
  tokens_used: Record<string, number>
}

export const generateAIStrategyDraft = async (params: {
  description: string
  source_key?: string
  model?: string
}): Promise<AIStrategyDraftGenerationResponse> => {
  const { data } = await api.post('/ai/generate/strategy-draft', params, AI_TIMEOUT)
  return unwrapApiData(data)
}

export interface AIModifyStrategyCodeResponse extends AIStrategyDraftGenerationResponse {
  change_summary: string
}

export const modifyAIStrategyCode = async (params: {
  instruction: string
  strategy_id?: string
  slug: string
  name: string
  source_key: string
  description?: string
  source_code: string
  config?: Record<string, unknown>
  config_schema?: Record<string, unknown>
  aliases?: string[]
  model?: string
}): Promise<AIModifyStrategyCodeResponse> => {
  const { data } = await api.post('/ai/modify/strategy-code', params, AI_TIMEOUT)
  return unwrapApiData(data)
}

export interface AIDataSourceDraftGenerationResponse {
  name: string
  slug: string
  source_key: string
  source_kind: string
  description: string
  source_code: string
  class_name: string
  retention: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  validation: {
    valid: boolean
    class_name: string | null
    errors: string[]
    warnings: string[]
    capabilities: Record<string, boolean>
  }
  used_repair_pass: boolean
  model: string
  tokens_used: Record<string, number>
}

export const generateAIDataSourceDraft = async (params: {
  description: string
  source_key?: string
  source_kind?: string
  model?: string
}): Promise<AIDataSourceDraftGenerationResponse> => {
  const { data } = await api.post('/ai/generate/data-source-draft', params, AI_TIMEOUT)
  return unwrapApiData(data)
}

export interface AIChatSession {
  session_id: string
  context_type: string | null
  context_id: string | null
  title: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AIChatSessionDetail extends AIChatSession {
  messages: Array<{
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    created_at: string | null
  }>
}

export const listAIChatSessions = async (params?: {
  context_type?: string
  context_id?: string
  limit?: number
}): Promise<{ sessions: AIChatSession[]; total: number }> => {
  const { data } = await api.get('/ai/chat/sessions', { params })
  return unwrapApiData(data)
}

export const getAIChatSession = async (sessionId: string): Promise<AIChatSessionDetail> => {
  const { data } = await api.get(`/ai/chat/sessions/${sessionId}`)
  return unwrapApiData(data)
}

export const archiveAIChatSession = async (sessionId: string): Promise<{ status: string; session_id: string }> => {
  const { data } = await api.delete(`/ai/chat/sessions/${sessionId}`)
  return unwrapApiData(data)
}

// ==================== KALSHI ACCOUNT ====================

export interface KalshiAccountStatus {
  platform: string
  authenticated: boolean
  member_id: string | null
  email: string | null
  balance: {
    balance: number
    payout: number
    available: number
    reserved: number
    currency: string
  } | null
  positions_count: number
}

export interface KalshiPosition {
  token_id: string
  market_id: string
  event_slug?: string
  market_question: string
  outcome: string
  size: number
  average_cost: number
  current_price: number
  unrealized_pnl: number
  platform: string
}

export const getKalshiStatus = async (): Promise<KalshiAccountStatus> => {
  const { data } = await api.get('/kalshi/status')
  return unwrapApiData(data)
}

export const loginKalshi = async (params: {
  email?: string
  password?: string
  api_key?: string
}): Promise<{ status: string; message: string; authenticated: boolean; member_id?: string }> => {
  const { data } = await api.post('/kalshi/login', params)
  return unwrapApiData(data)
}

export const logoutKalshi = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/kalshi/logout')
  return unwrapApiData(data)
}

export const getKalshiBalance = async (): Promise<{ balance: number; available: number; reserved: number; currency: string }> => {
  const { data } = await api.get('/kalshi/balance')
  return unwrapApiData(data)
}

export const getKalshiPositions = async (): Promise<KalshiPosition[]> => {
  const { data } = await api.get('/kalshi/positions')
  return unwrapApiData(data)
}

export const updateKalshiSettings = async (settings: Partial<KalshiSettings>): Promise<{ status: string; message: string }> => {
  const { data } = await api.put('/settings/kalshi', settings)
  return unwrapApiData(data)
}

// ==================== NEWS WORKFLOW (Independent Pipeline) ====================

export interface NewsWorkflowFinding {
  id: string
  article_id: string
  market_id: string
  article_title: string
  article_source: string
  article_url: string
  signal_key?: string | null
  strategy_sdk?: string | null
  cache_key?: string | null
  market_question: string
  market_price: number
  model_probability: number
  edge_percent: number
  direction: string
  confidence: number
  retrieval_score: number
  semantic_score: number
  keyword_score: number
  event_score: number
  rerank_score: number
  event_graph: Record<string, unknown>
  evidence: Record<string, unknown>
  reasoning: string
  actionable: boolean
  consumed_by_orchestrator: boolean
  price_history?: Array<Record<string, unknown> | unknown[]>
  outcome_labels?: string[]
  outcome_prices?: number[]
  market_token_ids?: string[]
  yes_price?: number | null
  no_price?: number | null
  current_yes_price?: number | null
  current_no_price?: number | null
  market_platform?: string | null
  market_slug?: string | null
  market_event_slug?: string | null
  market_event_ticker?: string | null
  market_url?: string | null
  polymarket_url?: string | null
  kalshi_url?: string | null
  supporting_articles?: NewsSupportingArticle[]
  supporting_article_count?: number
  created_at: string
}

export interface NewsSupportingArticle {
  article_id: string
  title: string
  url: string
  source: string
  published?: string | null
  fetched_at?: string | null
}

export interface NewsWorkflowStatus {
  running: boolean
  enabled: boolean
  paused: boolean
  interval_seconds: number
  last_scan: string | null
  next_scan: string | null
  current_activity: string | null
  last_error: string | null
  degraded_mode: boolean
  budget_remaining: number | null
  pending_intents: number
  requested_scan_at: string | null
  stats: Record<string, unknown>
}

export interface NewsWorkflowSettings {
  enabled: boolean
  auto_run: boolean
  scan_interval_seconds: number
  top_k: number
  rerank_top_n: number
  similarity_threshold: number
  keyword_weight: number
  semantic_weight: number
  event_weight: number
  require_verifier: boolean
  market_min_liquidity: number
  market_max_days_to_resolution: number
  min_keyword_signal: number
  min_semantic_signal: number
  min_edge_percent: number
  min_confidence: number
  require_second_source: boolean
  orchestrator_enabled: boolean
  orchestrator_min_edge: number
  orchestrator_max_age_minutes: number
  model: string | null
  article_max_age_hours: number
  cycle_spend_cap_usd: number
  hourly_spend_cap_usd: number
  cycle_llm_call_cap: number
  cache_ttl_minutes: number
  max_edge_evals_per_article: number
}

type NewsWorkflowSettingsUpdate = Omit<NewsWorkflowSettings, 'article_max_age_hours'>

export const getNewsWorkflowStatus = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.get('/news-workflow/status')
  return unwrapApiData(data)
}

export const runNewsWorkflow = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/news-workflow/run')
  return unwrapApiData(data)
}

export const startNewsWorkflow = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/start')
  return unwrapApiData(data)
}

export const pauseNewsWorkflow = async (): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/pause')
  return unwrapApiData(data)
}

export const setNewsWorkflowInterval = async (intervalSeconds: number): Promise<NewsWorkflowStatus> => {
  const { data } = await api.post('/news-workflow/interval', null, {
    params: { interval_seconds: intervalSeconds },
  })
  return unwrapApiData(data)
}

export const getNewsWorkflowSettings = async (): Promise<NewsWorkflowSettings> => {
  const { data } = await api.get('/news-workflow/settings')
  return unwrapApiData(data)
}

export const updateNewsWorkflowSettings = async (
  settings: Partial<NewsWorkflowSettingsUpdate>
): Promise<{ status: string; settings: NewsWorkflowSettings }> => {
  const { data } = await api.put('/news-workflow/settings', settings)
  return unwrapApiData(data)
}

export const getNewsWorkflowFindings = async (params?: {
  min_edge?: number
  actionable_only?: boolean
  include_debug_rejections?: boolean
  max_age_hours?: number
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; findings: NewsWorkflowFinding[] }> => {
  const { data } = await api.get('/news-workflow/findings', { params })
  return unwrapApiData(data)
}

// ==================== WEATHER WORKFLOW (Independent Pipeline) ====================

export interface WeatherWorkflowStatus {
  running: boolean
  enabled: boolean
  interval_seconds: number
  last_scan: string | null
  opportunities_count: number
  current_activity: string | null
  stats: Record<string, unknown>
  pending_intents: number
  paused: boolean
  requested_scan_at: string | null
}

export interface WeatherWorkflowSettings {
  enabled: boolean
  auto_run: boolean
  scan_interval_seconds: number
  entry_max_price: number
  take_profit_price: number
  stop_loss_pct: number
  min_edge_percent: number
  min_confidence: number
  min_model_agreement: number
  min_liquidity: number
  max_markets_per_scan: number
  default_size_usd: number
  max_size_usd: number
  model: string | null
  temperature_unit: 'F' | 'C'
}

export interface WeatherTradeIntent {
  id: string
  market_id: string
  market_question: string
  direction: string
  entry_price: number | null
  take_profit_price: number | null
  stop_loss_pct: number | null
  model_probability: number | null
  edge_percent: number | null
  confidence: number | null
  model_agreement: number | null
  suggested_size_usd: number | null
  metadata: Record<string, unknown> | null
  status: string
  created_at: string | null
  consumed_at: string | null
}

export interface WeatherWorkflowPerformance {
  lookback_days: number
  trades_total: number
  trades_resolved: number
  wins: number
  losses: number
  win_rate: number
  total_pnl: number
  intents_total: number
  pending_intents: number
  executed_intents: number
}

export interface WeatherOpportunityDateBucket {
  date: string
  count: number
}

export interface WeatherWorkflowOpportunityIdsResponse {
  total: number
  offset: number
  limit: number
  ids: string[]
}

export const getWeatherWorkflowStatus = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.get('/weather-workflow/status')
  return unwrapApiData(data)
}

export const runWeatherWorkflow = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.post('/weather-workflow/run')
  return unwrapApiData(data)
}

export const startWeatherWorkflow = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/start')
  return unwrapApiData(data)
}

export const pauseWeatherWorkflow = async (): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/pause')
  return unwrapApiData(data)
}

export const setWeatherWorkflowInterval = async (
  intervalSeconds: number
): Promise<WeatherWorkflowStatus> => {
  const { data } = await api.post('/weather-workflow/interval', null, {
    params: { interval_seconds: intervalSeconds }
  })
  return unwrapApiData(data)
}

export const getWeatherWorkflowOpportunities = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  target_date?: string
  include_report_only?: boolean
  limit?: number
  offset?: number
}): Promise<{ total: number; offset: number; limit: number; opportunities: Opportunity[] }> => {
  const { data } = await api.get('/weather-workflow/opportunities', { params })
  return unwrapApiData(data)
}

export const getWeatherWorkflowOpportunityDates = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  include_report_only?: boolean
}): Promise<{ total_dates: number; dates: WeatherOpportunityDateBucket[] }> => {
  const { data } = await api.get('/weather-workflow/opportunity-dates', { params })
  return unwrapApiData(data)
}

export const getWeatherWorkflowOpportunityIds = async (params?: {
  min_edge?: number
  direction?: string
  max_entry?: number
  location?: string
  target_date?: string
  include_report_only?: boolean
  limit?: number
  offset?: number
}): Promise<WeatherWorkflowOpportunityIdsResponse> => {
  const { data } = await api.get('/weather-workflow/opportunity-ids', { params })
  return unwrapApiData(data)
}

export const getWeatherWorkflowIntents = async (params?: {
  status_filter?: string
  limit?: number
}): Promise<{ total: number; intents: WeatherTradeIntent[] }> => {
  const { data } = await api.get('/weather-workflow/intents', { params })
  return unwrapApiData(data)
}

export const skipWeatherWorkflowIntent = async (intentId: string): Promise<{ status: string; intent_id: string }> => {
  const { data } = await api.post(`/weather-workflow/intents/${intentId}/skip`)
  return unwrapApiData(data)
}

export const getWeatherWorkflowSettings = async (): Promise<WeatherWorkflowSettings> => {
  const { data } = await api.get('/weather-workflow/settings')
  return unwrapApiData(data)
}

export const updateWeatherWorkflowSettings = async (
  settings: Partial<WeatherWorkflowSettings>
): Promise<{ status: string; settings: WeatherWorkflowSettings }> => {
  const { data } = await api.put('/weather-workflow/settings', settings)
  return unwrapApiData(data)
}

export const getWeatherWorkflowPerformance = async (
  lookbackDays = 90
): Promise<WeatherWorkflowPerformance> => {
  const { data } = await api.get('/weather-workflow/performance', {
    params: { lookback_days: lookbackDays }
  })
  return unwrapApiData(data)
}

export default api

// ==================== UNIFIED STRATEGY API ====================

export interface UnifiedStrategy {
  id: string
  slug: string
  source_key: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  is_system: boolean
  enabled: boolean
  status: string
  error_message: string | null
  version: number
  config: Record<string, unknown>
  config_schema: Record<string, unknown> | null
  aliases: string[]
  sort_order: number
  created_at: string | null
  updated_at: string | null
  capabilities: {
    has_detect: boolean
    has_detect_async: boolean
    has_evaluate: boolean
    has_should_exit: boolean
  }
  strategy_type: string  // 'detect' | 'execute' | 'unified'
  runtime: Record<string, any> | null
}

export const getUnifiedStrategies = async (params?: {
  type?: string
  source_key?: string
  enabled?: boolean
}): Promise<UnifiedStrategy[]> => {
  const { data } = await api.get('/strategy-manager', { params })
  return getStrategyManagerItems(data)
}

export const getUnifiedStrategy = async (id: string): Promise<UnifiedStrategy> => {
  const { data } = await api.get(`/strategy-manager/${id}`)
  return unwrapStrategyManagerPayload(data)
}

export const createUnifiedStrategy = async (payload: {
  slug: string
  source_key?: string
  name?: string
  description?: string
  source_code: string
  class_name?: string
  config?: Record<string, unknown>
  config_schema?: Record<string, unknown>
  aliases?: string[]
  enabled?: boolean
}): Promise<UnifiedStrategy> => {
  const { data } = await api.post('/strategy-manager', payload)
  return unwrapApiData(data)
}

export const updateUnifiedStrategy = async (
  id: string,
  payload: Partial<{
    slug: string
    source_key: string
    name: string
    description: string
    source_code: string
    class_name: string
    config: Record<string, unknown>
    config_schema: Record<string, unknown>
    aliases: string[]
    enabled: boolean
    unlock_system: boolean
  }>
): Promise<UnifiedStrategy> => {
  const { data } = await api.put(`/strategy-manager/${id}`, payload)
  return unwrapApiData(data)
}

export const deleteUnifiedStrategy = async (id: string): Promise<void> => {
  await api.delete(`/strategy-manager/${id}`)
}

export const validateUnifiedStrategy = async (source_code: string, class_name?: string): Promise<{
  valid: boolean
  inferred_type: string
  capabilities: Record<string, boolean>
  class_name: string | null
  errors: string[]
  warnings: string[]
}> => {
  const { data } = await api.post('/strategy-manager/validate', { source_code, class_name })
  return unwrapApiData(data)
}

export const reloadUnifiedStrategy = async (id: string): Promise<{
  status: string
  message?: string
  runtime?: Record<string, any>
}> => {
  const { data } = await api.post(`/strategy-manager/${id}/reload`)
  return unwrapApiData(data)
}

export const getUnifiedStrategyTemplate = async (): Promise<{
  template: string
  instructions: string
  available_imports: string[]
}> => {
  const { data } = await api.get('/strategy-manager/template')
  return unwrapApiData(data)
}

export const getUnifiedStrategyDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/strategy-manager/docs')
  const payload = unwrapStrategyManagerPayload(data)
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return {}
  }
  return payload
}

export interface UnifiedStrategyVersion {
  id: string
  strategy_id: string
  strategy_slug: string
  source_key: string
  version: number
  is_latest: boolean
  name: string
  description: string | null
  source_code?: string
  class_name: string | null
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  aliases: string[]
  enabled: boolean
  is_system: boolean
  sort_order: number
  parent_version: number | null
  created_by: string | null
  reason: string | null
  created_at: string | null
}

export const getUnifiedStrategyVersions = async (
  strategyId: string,
  params?: { limit?: number; include_source?: boolean }
): Promise<UnifiedStrategyVersion[]> => {
  const { data } = await api.get(`/strategy-manager/${strategyId}/versions`, { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

export const getUnifiedStrategyVersion = async (
  strategyId: string,
  version: number | string
): Promise<UnifiedStrategyVersion> => {
  const { data } = await api.get(`/strategy-manager/${strategyId}/versions/${version}`)
  return unwrapApiData(data)
}

export const restoreUnifiedStrategyVersion = async (
  strategyId: string,
  version: number | string,
  payload?: { reason?: string; created_by?: string }
): Promise<{
  status: string
  strategy: UnifiedStrategy
  restored_snapshot: UnifiedStrategyVersion
  source_snapshot: UnifiedStrategyVersion
}> => {
  const { data } = await api.post(`/strategy-manager/${strategyId}/versions/${version}/restore`, payload || {})
  return unwrapApiData(data)
}

export interface StrategyExperiment {
  id: string
  name: string
  source_key: string
  strategy_key: string
  control_version: number
  candidate_version: number
  candidate_allocation_pct: number
  scope: Record<string, unknown>
  status: 'active' | 'paused' | 'completed' | 'archived' | string
  created_by: string | null
  notes: string | null
  metadata: Record<string, unknown>
  promoted_version: number | null
  ended_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface StrategyExperimentAssignment {
  id: string
  experiment_id: string
  trader_id: string | null
  signal_id: string | null
  decision_id: string | null
  order_id: string | null
  source_key: string
  strategy_key: string
  strategy_version: number
  assignment_group: string
  payload: Record<string, unknown>
  created_at: string | null
  updated_at: string | null
}

export const listStrategyExperiments = async (params?: {
  source_key?: string
  strategy_key?: string
  status?: string
  limit?: number
}): Promise<StrategyExperiment[]> => {
  const { data } = await api.get('/strategy-manager/experiments', { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

export const createStrategyExperiment = async (payload: {
  name: string
  source_key: string
  strategy_key: string
  control_version: number
  candidate_version: number
  candidate_allocation_pct?: number
  scope?: Record<string, unknown>
  notes?: string
  created_by?: string
}): Promise<StrategyExperiment> => {
  const { data } = await api.post('/strategy-manager/experiments', payload)
  return unwrapApiData(data)
}

export const setStrategyExperimentStatus = async (
  experimentId: string,
  status: string
): Promise<StrategyExperiment> => {
  const { data } = await api.post(`/strategy-manager/experiments/${experimentId}/status`, { status })
  return unwrapApiData(data)
}

export const promoteStrategyExperiment = async (
  experimentId: string,
  payload?: { promoted_version?: number; notes?: string }
): Promise<StrategyExperiment> => {
  const { data } = await api.post(`/strategy-manager/experiments/${experimentId}/promote`, payload || {})
  return unwrapApiData(data)
}

export const listStrategyExperimentAssignments = async (
  experimentId: string,
  params?: { limit?: number }
): Promise<StrategyExperimentAssignment[]> => {
  const { data } = await api.get(`/strategy-manager/experiments/${experimentId}/assignments`, { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (payload && typeof payload === 'object' && Array.isArray((payload as any).items)) {
    return (payload as any).items
  }
  return []
}

export interface TraderTuneAgentRequest {
  prompt: string
  model?: string
  max_iterations?: number
  monitor_job_id?: string
}

export interface TraderTuneAgentResponse {
  session_id: string
  answer: string
  parsed: Record<string, unknown> | null
  applied_param_updates: Array<{
    source_key: string
    strategy_key: string
    changed_keys: string[]
    params_patch: Record<string, unknown>
    reason: string
  }>
  applied_param_update_count: number
  updated_trader: Trader | null
  raw: Record<string, unknown>
}

export const runTraderTuneIteration = async (
  traderId: string,
  payload: TraderTuneAgentRequest
): Promise<TraderTuneAgentResponse> => {
  const { data } = await api.post(`/traders/${traderId}/tune/iterate`, payload, { timeout: 240_000 })
  return unwrapApiData(data)
}

// ==================== UNIFIED DATA SOURCE API ====================

export interface UnifiedDataSource {
  id: string
  slug: string
  source_key: string
  source_kind: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  is_system: boolean
  enabled: boolean
  status: string
  error_message: string | null
  version: number
  retention: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown> | null
  sort_order: number
  created_at: string | null
  updated_at: string | null
  capabilities: {
    has_fetch: boolean
    has_fetch_async: boolean
    has_transform: boolean
  }
  runtime: Record<string, any> | null
  record_count: number | null
}

export interface UnifiedDataSourceRun {
  id: string
  status: string
  fetched_count: number
  transformed_count: number
  upserted_count: number
  skipped_count: number
  error_message: string | null
  metadata: Record<string, unknown>
  started_at: string | null
  completed_at: string | null
  duration_ms: number | null
}

export interface UnifiedDataSourceRecord {
  id: string
  external_id: string | null
  title: string | null
  summary: string | null
  category: string | null
  source: string | null
  url: string | null
  geotagged: boolean
  country_iso3: string | null
  latitude: number | null
  longitude: number | null
  observed_at: string | null
  ingested_at: string | null
  payload: Record<string, unknown>
  transformed: Record<string, unknown>
  tags: string[]
}

export interface UnifiedDataSourceTemplatePreset {
  id: string
  slug_prefix?: string
  name: string
  description: string
  source_key: string
  source_kind: string
  source_code: string
  retention?: {
    max_records?: number
    max_age_days?: number
  }
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  is_system_seed?: boolean
}

export interface UnifiedDataSourceTemplateResponse {
  template: string
  default_preset?: string
  presets?: UnifiedDataSourceTemplatePreset[]
  instructions: string
  available_imports: string[]
}

export const getUnifiedDataSources = async (params?: {
  source_key?: string
  enabled?: boolean
  include_code?: boolean
}): Promise<UnifiedDataSource[]> => {
  const { data } = await api.get('/data-sources', { params })
  const payload = unwrapApiData(data)
  if (Array.isArray(payload)) return payload
  if (Array.isArray(payload?.items)) return payload.items
  return []
}

export const getUnifiedDataSource = async (id: string): Promise<UnifiedDataSource> => {
  const { data } = await api.get(`/data-sources/${id}`)
  return unwrapApiData(data)
}

export const createUnifiedDataSource = async (payload: {
  slug: string
  source_key?: string
  source_kind?: string
  name?: string
  description?: string
  source_code: string
  retention?: {
    max_records?: number
    max_age_days?: number
  }
  config?: Record<string, unknown>
  config_schema?: Record<string, unknown>
  enabled?: boolean
}): Promise<UnifiedDataSource> => {
  const { data } = await api.post('/data-sources', payload)
  return unwrapApiData(data)
}

export const updateUnifiedDataSource = async (
  id: string,
  payload: Partial<{
    slug: string
    source_key: string
    source_kind: string
    name: string
    description: string
    source_code: string
    retention: {
      max_records?: number
      max_age_days?: number
    }
    config: Record<string, unknown>
    config_schema: Record<string, unknown>
    enabled: boolean
    unlock_system: boolean
  }>
): Promise<UnifiedDataSource> => {
  const { data } = await api.put(`/data-sources/${id}`, payload)
  return unwrapApiData(data)
}

export const deleteUnifiedDataSource = async (id: string): Promise<void> => {
  await api.delete(`/data-sources/${id}`)
}

export const validateUnifiedDataSource = async (source_code: string, class_name?: string): Promise<{
  valid: boolean
  class_name: string | null
  source_name: string | null
  source_description: string | null
  capabilities: Record<string, boolean>
  errors: string[]
  warnings: string[]
}> => {
  const { data } = await api.post('/data-sources/validate', { source_code, class_name })
  return unwrapApiData(data)
}

export const reloadUnifiedDataSource = async (id: string): Promise<{
  status: string
  message?: string
  runtime?: Record<string, any> | null
}> => {
  const { data } = await api.post(`/data-sources/${id}/reload`)
  return unwrapApiData(data)
}

export const runUnifiedDataSource = async (
  id: string,
  payload?: { max_records?: number }
): Promise<{
  run_id: string
  source_slug: string
  status: string
  fetched_count: number
  transformed_count: number
  upserted_count: number
  skipped_count: number
  error_message: string | null
  duration_ms: number | null
}> => {
  const { data } = await api.post(`/data-sources/${id}/run`, payload || {})
  return unwrapApiData(data)
}

export const getUnifiedDataSourceRuns = async (
  id: string,
  params?: { limit?: number }
): Promise<{
  source_id: string
  source_slug: string
  runs: UnifiedDataSourceRun[]
}> => {
  const { data } = await api.get(`/data-sources/${id}/runs`, { params })
  return unwrapApiData(data)
}

export const getUnifiedDataSourceRecords = async (
  id: string,
  params?: { limit?: number; offset?: number; geotagged?: boolean }
): Promise<{
  source_id: string
  source_slug: string
  total: number
  offset: number
  limit: number
  records: UnifiedDataSourceRecord[]
}> => {
  const { data } = await api.get(`/data-sources/${id}/records`, { params })
  return unwrapApiData(data)
}

export const previewUnifiedDataSource = async (
  id: string,
  payload?: { max_records?: number }
): Promise<{
  source_id: string
  source_slug: string
  total_fetched: number
  duration_ms: number
  records: UnifiedDataSourceRecord[]
}> => {
  const { data } = await api.post(`/data-sources/${id}/preview`, payload || {})
  return unwrapApiData(data)
}

export const getUnifiedDataSourceTemplate = async (): Promise<UnifiedDataSourceTemplateResponse> => {
  const { data } = await api.get('/data-sources/template')
  return unwrapApiData(data)
}

export const getUnifiedDataSourceDocs = async (): Promise<Record<string, any>> => {
  const { data } = await api.get('/data-sources/docs')
  return unwrapApiData(data)
}

// --- Crypto Filter Diagnostics ---

export interface CryptoFilterRejection {
  market: string
  asset: string
  timeframe: string
  gate: 'oracle_move' | 'repriced'
  oracle_move_pct: number
  threshold_pct?: number
  side?: string
  price?: number
  max_price?: number
}

export interface CryptoFilterDiagnostics {
  scanned_at: string
  markets_scanned: number
  signals_emitted: number
  strategy_key?: string | null
  primary_strategy_key?: string | null
  rejections: CryptoFilterRejection[] | Record<string, number>
  message?: string
  thresholds_percent?: Record<string, number>
  summary: {
    oracle_move?: number
    repriced?: number
    max_oracle_move_pct?: number
    strategies_loaded?: number
    strategies_reporting_diagnostics?: number
    strategies_with_signals?: number
    total_signals_emitted?: number
    [key: string]: unknown
  }
  strategies?: Record<string, CryptoFilterDiagnostics>
  dispatch_summary?: {
    strategies_loaded?: number
    strategies_reporting_diagnostics?: number
    strategies_missing_diagnostics?: string[]
    opportunities_by_strategy?: Record<string, number>
    rejection_counts_by_strategy?: Record<string, Record<string, number>>
  }
}

export const getCryptoFilterDiagnostics = async (): Promise<CryptoFilterDiagnostics> => {
  const { data } = await api.get('/crypto/filter-diagnostics')
  return data
}

// ==================== AI AGENTS ====================

export interface UserAgent {
  id: string
  name: string
  description: string
  system_prompt: string
  tools: string[]
  model: string | null
  temperature: number
  max_iterations: number
  is_builtin: boolean
  created_at?: string
  updated_at?: string
}

export async function listAgents(): Promise<{ agents: UserAgent[] }> {
  const response = await api.get('/ai/agents')
  return response.data
}

export async function getAgent(agentId: string): Promise<UserAgent> {
  const response = await api.get(`/ai/agents/${agentId}`)
  return response.data
}

export async function createAgent(data: Omit<UserAgent, 'id' | 'is_builtin' | 'created_at' | 'updated_at'>): Promise<UserAgent> {
  const response = await api.post('/ai/agents', data)
  return response.data
}

export async function updateAgent(agentId: string, data: Partial<UserAgent>): Promise<UserAgent> {
  const response = await api.put(`/ai/agents/${agentId}`, data)
  return response.data
}

export async function deleteAgent(agentId: string): Promise<void> {
  await api.delete(`/ai/agents/${agentId}`)
}

export async function listAvailableTools(): Promise<{ tools: { name: string; description: string; category: string; parameters?: Record<string, unknown> }[] }> {
  const response = await api.get('/ai/agents/meta/available-tools')
  return response.data
}

// ==================== AI STREAMING CHAT ====================

export interface ChatStreamEvent {
  event: 'session' | 'token' | 'thinking' | 'tool_start' | 'tool_end' | 'tool_error' | 'done' | 'error'
    | 'experiment_start' | 'iteration_start' | 'agent_progress' | 'proposal' | 'decision'
  data: Record<string, unknown>
}

export function streamAIChat(
  params: {
    message: string
    session_id?: string
    context_type?: string
    context_id?: string
    model?: string
  },
  onToken: (text: string) => void,
  onDone: (data: { session_id: string; model_used: string }) => void,
  onError: (error: string) => void,
  signal?: AbortSignal,
  onEvent?: (event: ChatStreamEvent) => void,
): void {
  fetch('/api/ai/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  }).then(response => {
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) { onError('No response body'); return }

    let buffer = ''
    let currentEventType = ''
    const read = (): void => {
      reader.read().then(({ done, value }) => {
        if (done) return
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim()
            continue
          }
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              const eventType = currentEventType || 'token'
              currentEventType = ''

              if (onEvent) {
                onEvent({ event: eventType as ChatStreamEvent['event'], data })
              }

              if (eventType === 'token' && data.text !== undefined) onToken(data.text)
              if (eventType === 'done' && data.session_id !== undefined) onDone(data as { session_id: string; model_used: string })
              if (eventType === 'error' && data.error) onError(data.error as string)
            } catch { /* skip malformed SSE lines */ }
          }
        }
        read()
      }).catch(err => {
        if (err.name !== 'AbortError') onError(err.message)
      })
    }
    read()
  }).catch(err => {
    if (err.name !== 'AbortError') onError(err.message)
  })
}

// ==================== LLM USAGE LOG ====================

export interface LLMUsageLogEntry {
  id: string
  provider: string
  model: string
  input_tokens: number
  output_tokens: number
  cost_usd: number
  purpose: string | null
  session_id: string | null
  requested_at: string
  latency_ms: number | null
  success: boolean
  error: string | null
}

export const listLLMUsageLog = async (params: {
  limit?: number
  offset?: number
  purpose?: string
  provider?: string
  success?: boolean
}): Promise<{ entries: LLMUsageLogEntry[]; total: number }> => {
  const searchParams = new URLSearchParams()
  if (params.limit) searchParams.set('limit', String(params.limit))
  if (params.offset) searchParams.set('offset', String(params.offset))
  if (params.purpose) searchParams.set('purpose', params.purpose)
  if (params.provider) searchParams.set('provider', params.provider)
  if (params.success !== undefined) searchParams.set('success', String(params.success))
  const { data } = await api.get(`/ai/usage-log?${searchParams.toString()}`)
  return data
}

// ==================== AI CHAT SESSION MANAGEMENT ====================

export async function renameChatSession(sessionId: string, title: string): Promise<AIChatSession> {
  const response = await api.patch(`/ai/chat/sessions/${sessionId}`, { title })
  return response.data
}

// ==================== DISCOVERY PROFILES ====================

export interface DiscoveryProfile {
  id: string
  slug: string
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  is_system: boolean
  enabled: boolean
  is_active: boolean
  status: string
  error_message: string | null
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  profile_kind: string
  version: number
  sort_order: number
  capabilities: { has_score_wallet: boolean; has_select_pool: boolean } | null
  runtime: {
    slug: string
    class_name: string
    source_hash: string
    loaded_at: string
    run_count: number
    error_count: number
    last_run: string | null
    last_error: string | null
  } | null
  created_at: string | null
  updated_at: string | null
}

export interface DiscoveryProfileVersion {
  id: string
  profile_id: string
  profile_slug: string
  version: number
  is_latest: boolean
  name: string
  description: string | null
  source_code: string
  class_name: string | null
  config: Record<string, unknown>
  config_schema: Record<string, unknown>
  profile_kind: string
  enabled: boolean
  is_system: boolean
  sort_order: number
  parent_version: number | null
  created_by: string | null
  reason: string | null
  created_at: string | null
}

export interface DiscoveryProfileValidationResult {
  valid: boolean
  class_name: string | null
  errors: string[]
  warnings: string[]
  capabilities: { has_score_wallet: boolean; has_select_pool: boolean } | null
}

export interface DiscoveryProfileTemplateResponse {
  template: string
  available_imports: Record<string, unknown>
}

export const getDiscoveryProfiles = async (params?: {
  enabled?: boolean
  profile_kind?: string
}): Promise<DiscoveryProfile[]> => {
  const searchParams = new URLSearchParams()
  if (params?.enabled !== undefined) searchParams.set('enabled', String(params.enabled))
  if (params?.profile_kind) searchParams.set('profile_kind', params.profile_kind)
  const query = searchParams.toString()
  const { data } = await api.get(`/discovery-profiles${query ? `?${query}` : ''}`)
  return data
}

export const getDiscoveryProfile = async (id: string): Promise<DiscoveryProfile> => {
  const { data } = await api.get(`/discovery-profiles/${id}`)
  return data
}

export const createDiscoveryProfile = async (payload: {
  slug: string
  name?: string
  description?: string
  source_code?: string
  profile_kind?: string
  config?: Record<string, unknown>
  config_schema?: Record<string, unknown>
  enabled?: boolean
}): Promise<DiscoveryProfile> => {
  const { data } = await api.post('/discovery-profiles', payload)
  return data
}

export const updateDiscoveryProfile = async (
  id: string,
  payload: Partial<{
    name: string
    description: string
    source_code: string
    config: Record<string, unknown>
    config_schema: Record<string, unknown>
    enabled: boolean
    profile_kind: string
  }>
): Promise<DiscoveryProfile> => {
  const { data } = await api.put(`/discovery-profiles/${id}`, payload)
  return data
}

export const deleteDiscoveryProfile = async (id: string): Promise<void> => {
  await api.delete(`/discovery-profiles/${id}`)
}

export const validateDiscoveryProfile = async (
  sourceCode: string,
  className?: string
): Promise<DiscoveryProfileValidationResult> => {
  const { data } = await api.post('/discovery-profiles/validate', {
    source_code: sourceCode,
    class_name: className,
  })
  return data
}

export const reloadDiscoveryProfile = async (id: string): Promise<DiscoveryProfile> => {
  const { data } = await api.post(`/discovery-profiles/${id}/reload`)
  return data
}

export const getDiscoveryProfileTemplate = async (): Promise<DiscoveryProfileTemplateResponse> => {
  const { data } = await api.get('/discovery-profiles/template')
  return data
}

export const getDiscoveryProfileDocs = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.get('/discovery-profiles/docs')
  return data
}

export const getDiscoveryProfileVersions = async (id: string): Promise<DiscoveryProfileVersion[]> => {
  const { data } = await api.get(`/discovery-profiles/${id}/versions`)
  return data
}

export const restoreDiscoveryProfileVersion = async (
  id: string,
  version: number,
  reason?: string
): Promise<DiscoveryProfile> => {
  const { data } = await api.post(`/discovery-profiles/${id}/versions/${version}/restore`, { reason })
  return data
}

export const setActiveDiscoveryProfile = async (profileId: string): Promise<{ active_profile_id: string }> => {
  const { data } = await api.put('/discovery-profiles/active', { profile_id: profileId })
  return data
}

// ==================== CORTEX AGENT ====================

export interface CortexMemoryEntry {
  id: string
  category: string
  content: string
  importance: number
  access_count: number
  context: Record<string, unknown> | null
  expired: boolean
  created_at: string | null
  updated_at: string | null
}

export interface CortexSettings {
  enabled: boolean
  model: string | null
  interval_seconds: number
  max_iterations: number
  temperature: number
  mandate: string | null
  memory_limit: number
  write_actions_enabled: boolean
  notify_telegram: boolean
}

export interface CortexStatus {
  enabled: boolean
  write_actions_enabled: boolean
  interval_seconds: number
  memory_count: number
  session_id: string
}

export interface CortexHistoryMessage {
  id: string
  session_id: string
  role: string
  content: string
  model_used: string | null
  created_at: string | null
}

export async function getCortexStatus(): Promise<CortexStatus> {
  const { data } = await api.get('/cortex/status')
  return data
}

export async function getCortexHistory(limit?: number): Promise<{ session_id: string; messages: CortexHistoryMessage[] }> {
  const { data } = await api.get('/cortex/history', { params: { limit } })
  return data
}

export async function clearCortexHistory(): Promise<{ session_id: string }> {
  const { data } = await api.delete('/cortex/history')
  return data
}

export function streamCortexCycle(
  onEvent: (event: ChatStreamEvent) => void,
  onDone: () => void,
  onError: (error: string) => void,
  signal?: AbortSignal,
): void {
  fetch('/api/cortex/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
  }).then(response => {
    if (!response.ok) {
      response.text().then(text => onError(text || `HTTP ${response.status}`))
      return
    }
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) { onError('No response body'); return }

    let buffer = ''
    const read = (): void => {
      reader.read().then(({ done, value }) => {
        if (done) { onDone(); return }
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6))
              onEvent({ event: parsed.type, data: parsed.data || {} })
              if (parsed.type === 'done' || parsed.type === 'error') {
                onDone()
                return
              }
            } catch { /* skip malformed */ }
          }
        }
        read()
      }).catch(err => {
        if (err.name !== 'AbortError') onError(String(err))
      })
    }
    read()
  }).catch(err => {
    if (err.name !== 'AbortError') onError(String(err))
  })
}

export async function listCortexMemories(params?: {
  limit?: number
  offset?: number
  category?: string
  include_expired?: boolean
}): Promise<{ memories: CortexMemoryEntry[]; total: number }> {
  const { data } = await api.get('/cortex/memory', { params })
  return data
}

export async function createCortexMemory(memory: {
  content: string
  category: string
  importance?: number
  context?: Record<string, unknown>
}): Promise<CortexMemoryEntry> {
  const { data } = await api.post('/cortex/memory', memory)
  return data
}

export async function updateCortexMemory(id: string, updates: {
  content?: string
  category?: string
  importance?: number
  expired?: boolean
}): Promise<CortexMemoryEntry> {
  const { data } = await api.put(`/cortex/memory/${id}`, updates)
  return data
}

export async function deleteCortexMemory(id: string): Promise<void> {
  await api.delete(`/cortex/memory/${id}`)
}

export async function getCortexSettings(): Promise<CortexSettings> {
  const { data } = await api.get('/cortex/settings')
  return data
}

export async function updateCortexSettings(settings: Partial<CortexSettings>): Promise<{ status: string; updated: string[] }> {
  const { data } = await api.put('/cortex/settings', settings)
  return data
}

// ---------------------------------------------------------------------------
// Autoresearch
// ---------------------------------------------------------------------------

export interface AutoresearchSettings {
  model: string | null
  max_iterations: number
  interval_seconds: number
  temperature: number
  mandate: string | null
  auto_apply: boolean
  walk_forward_windows: number
  train_ratio: number
  mode: 'params' | 'code'
}

export interface AutoresearchExperimentStatus {
  experiment_id: string | null
  trader_id: string
  name: string | null
  status: 'running' | 'paused' | 'completed' | 'failed' | 'idle'
  iteration_count: number
  best_score: number
  baseline_score: number
  best_params?: Record<string, unknown>
  kept_count: number
  reverted_count: number
  started_at: string | null
  finished_at?: string | null
  settings?: Record<string, unknown>
  mode?: 'params' | 'code'
  strategy_id?: string | null
  best_version?: number | null
}

export interface AutoresearchIteration {
  id: string
  iteration_number: number
  baseline_score: number
  new_score: number
  score_delta: number
  decision: 'kept' | 'reverted'
  reasoning: string
  changed_params: Record<string, unknown> | null
  backtest_result: Record<string, unknown> | null
  source_diff?: string | null
  validation_result?: Record<string, unknown> | null
  source_diff_lines?: number
  duration_seconds: number
  tokens_used: number
  created_at: string
}

export async function getAutoresearchStatus(traderId: string): Promise<AutoresearchExperimentStatus> {
  const { data } = await api.get(`/autoresearch/status/${traderId}`)
  return data
}

export async function getAutoresearchHistory(traderId: string, limit = 50): Promise<{ iterations: AutoresearchIteration[] }> {
  const { data } = await api.get(`/autoresearch/history/${traderId}`, { params: { limit } })
  return data
}

export async function listAutoresearchExperiments(traderId: string): Promise<{ experiments: AutoresearchExperimentStatus[] }> {
  const { data } = await api.get(`/autoresearch/experiments/${traderId}`)
  return data
}

export async function getAutoresearchSettings(): Promise<AutoresearchSettings> {
  const { data } = await api.get('/autoresearch/settings')
  return data
}

export async function updateAutoresearchSettings(settings: Partial<AutoresearchSettings>): Promise<AutoresearchSettings> {
  const { data } = await api.put('/autoresearch/settings', settings)
  return data
}

export async function stopAutoresearchExperiment(traderId: string): Promise<{ stopped: boolean; experiment_id: string | null }> {
  const { data } = await api.post(`/autoresearch/stop/${traderId}`)
  return data
}

// ── Strategy-scoped autoresearch (code evolution against backtest data plane) ──

export async function getStrategyAutoresearchStatus(strategyId: string): Promise<AutoresearchExperimentStatus> {
  const { data } = await api.get(`/autoresearch/strategy/${strategyId}/status`)
  return data
}

export async function getStrategyAutoresearchHistory(
  strategyId: string,
  limit = 50,
): Promise<{ iterations: AutoresearchIteration[] }> {
  const { data } = await api.get(`/autoresearch/strategy/${strategyId}/history`, { params: { limit } })
  return data
}

export async function stopStrategyAutoresearchExperiment(
  strategyId: string,
): Promise<{ stopped: boolean; experiment_id: string | null }> {
  const { data } = await api.post(`/autoresearch/strategy/${strategyId}/stop`)
  return data
}

export function streamStrategyAutoresearchExperiment(
  strategyId: string,
  onEvent: (event: ChatStreamEvent) => void,
  onDone: () => void,
  onError: (error: string) => void,
  signal?: AbortSignal,
  body?: { model?: string; max_iterations?: number; mandate?: string },
): void {
  fetch(`/api/autoresearch/strategy/${strategyId}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  }).then(response => {
    if (!response.ok) {
      response.text().then(text => onError(text || `HTTP ${response.status}`))
      return
    }
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) { onError('No response body'); return }

    let buffer = ''
    const read = (): void => {
      reader.read().then(({ done, value }) => {
        if (done) { onDone(); return }
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6))
              onEvent({ event: parsed.type, data: parsed.data || {} })
              if (parsed.type === 'done' || parsed.type === 'error') {
                onDone()
                return
              }
            } catch { /* skip malformed */ }
          }
        }
        read()
      }).catch(err => {
        if (err.name !== 'AbortError') onError(String(err))
      })
    }
    read()
  }).catch(err => {
    if (err.name !== 'AbortError') onError(String(err))
  })
}

export async function createAbExperimentFromAutoresearch(experimentId: string): Promise<{ ab_experiment_id: string; control_version: number; candidate_version: number; strategy_slug: string }> {
  const { data } = await api.post(`/autoresearch/create-ab-experiment/${experimentId}`)
  return data
}

// ── Strategy-scoped PARAMS iteration (the missing diagonal — runs the
// LLM-driven loop over the strategy's declared param_schema against the
// unified backtest engine, distinct from the code-evolution loop above). ──

export async function getStrategyParamsAutoresearchStatus(
  strategyId: string,
): Promise<AutoresearchExperimentStatus> {
  const { data } = await api.get(`/autoresearch/strategy/${strategyId}/params/status`)
  return data
}

export async function getStrategyParamsAutoresearchHistory(
  strategyId: string,
  limit = 50,
): Promise<{ iterations: AutoresearchIteration[] }> {
  const { data } = await api.get(`/autoresearch/strategy/${strategyId}/params/history`, { params: { limit } })
  return data
}

export async function stopStrategyParamsAutoresearchExperiment(
  strategyId: string,
): Promise<{ stopped: boolean; experiment_id: string | null }> {
  const { data } = await api.post(`/autoresearch/strategy/${strategyId}/params/stop`)
  return data
}

export interface StrategyParamsStartBody {
  model?: string
  max_iterations?: number
  mandate?: string
  auto_apply?: boolean
  // Stop conditions — exit early once best_score >= target_score, or
  // after max_no_improvement consecutive non-improving iterations.
  target_score?: number
  max_no_improvement?: number
}

export function streamStrategyParamsAutoresearchExperiment(
  strategyId: string,
  onEvent: (event: ChatStreamEvent) => void,
  onDone: () => void,
  onError: (error: string) => void,
  signal?: AbortSignal,
  body?: StrategyParamsStartBody,
): void {
  fetch(`/api/autoresearch/strategy/${strategyId}/params/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  }).then(response => {
    if (!response.ok) {
      response.text().then(text => onError(text || `HTTP ${response.status}`))
      return
    }
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) { onError('No response body'); return }

    let buffer = ''
    const read = (): void => {
      reader.read().then(({ done, value }) => {
        if (done) { onDone(); return }
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6))
              onEvent({ event: parsed.type, data: parsed.data || {} })
              if (parsed.type === 'done' || parsed.type === 'error') {
                onDone()
                return
              }
            } catch { /* skip malformed */ }
          }
        }
        read()
      }).catch(err => {
        if (err.name !== 'AbortError') onError(String(err))
      })
    }
    read()
  }).catch(err => {
    if (err.name !== 'AbortError') onError(String(err))
  })
}

export function streamAutoresearchExperiment(
  traderId: string,
  onEvent: (event: ChatStreamEvent) => void,
  onDone: () => void,
  onError: (error: string) => void,
  signal?: AbortSignal,
  body?: { mode?: string; strategy_id?: string },
): void {
  fetch(`/api/autoresearch/stream/${traderId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  }).then(response => {
    if (!response.ok) {
      response.text().then(text => onError(text || `HTTP ${response.status}`))
      return
    }
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) { onError('No response body'); return }

    let buffer = ''
    const read = (): void => {
      reader.read().then(({ done, value }) => {
        if (done) { onDone(); return }
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6))
              onEvent({ event: parsed.type, data: parsed.data || {} })
              if (parsed.type === 'done' || parsed.type === 'error') {
                onDone()
                return
              }
            } catch { /* skip malformed */ }
          }
        }
        read()
      }).catch(err => {
        if (err.name !== 'AbortError') onError(String(err))
      })
    }
    read()
  }).catch(err => {
    if (err.name !== 'AbortError') onError(String(err))
  })
}
