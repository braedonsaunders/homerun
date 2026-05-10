import { useState, useRef, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Brain,
  Play,
  Square,
  Clock,
  Zap,
  BookOpen,
  Settings2,
  Trash2,
  Plus,
  Pencil,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'
import { Switch } from '../ui/switch'
import { MessageContent } from './ChatRendering'
import {
  getCortexStatus,
  getCortexHistory,
  clearCortexHistory,
  getCortexSettings,
  updateCortexSettings,
  listCortexMemories,
  createCortexMemory,
  updateCortexMemory,
  deleteCortexMemory,
  streamCortexCycle,
  type CortexSettings,
  type CortexHistoryMessage,
  type ChatStreamEvent,
  getLLMModels,
} from '../../services/api'

type Panel = 'activity' | 'memory' | 'settings'

export default function CortexView() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [panel, setPanel] = useState<Panel>('activity')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamPhase, setStreamPhase] = useState<'working' | 'answering'>('working')
  const [streamContent, setStreamContent] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['cortex-status'],
    queryFn: getCortexStatus,
    refetchInterval: 15000,
  })

  const toggleMutation = useMutation({
    mutationFn: (updates: Partial<CortexSettings>) => updateCortexSettings(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cortex-status'] })
      queryClient.invalidateQueries({ queryKey: ['cortex-settings'] })
    },
  })

  const handleTriggerRun = useCallback(() => {
    setIsStreaming(true)
    setStreamPhase('working')
    setStreamContent('')
    setPanel('activity')
    const controller = new AbortController()
    abortRef.current = controller

    const answerChunks: string[] = []
    let toolName = ''

    streamCortexCycle(
      (event: ChatStreamEvent) => {
        switch (event.event) {
          case 'thinking':
            break
          case 'tool_start':
            toolName = String(event.data?.tool || '').replace(/_/g, ' ')
            setStreamContent(toolName)
            break
          case 'tool_end':
          case 'tool_error':
            break
          case 'token':
            setStreamPhase('answering')
            answerChunks.push(String(event.data.text || ''))
            setStreamContent(answerChunks.join(''))
            break
          case 'done': {
            setStreamPhase('answering')
            const result = event.data.result as Record<string, unknown> | undefined
            if (result?.answer) {
              setStreamContent(String(result.answer))
            } else if (answerChunks.length > 0) {
              setStreamContent(answerChunks.join(''))
            }
            break
          }
          case 'error':
            setStreamPhase('answering')
            setStreamContent(`**${t('aiCortexView.errorBold')}** ${event.data.error || t('aiCortexView.unknownError')}`)
            break
        }
      },
      () => {
        setIsStreaming(false)
        abortRef.current = null
        queryClient.invalidateQueries({ queryKey: ['cortex-status'] })
        queryClient.invalidateQueries({ queryKey: ['cortex-history'] })
      },
      (error) => {
        setIsStreaming(false)
        abortRef.current = null
        setStreamContent(`**${t('aiCortexView.errorBold')}** ${error}`)
      },
      controller.signal,
    )
  }, [queryClient, t])

  const handleStopRun = useCallback(() => {
    abortRef.current?.abort()
    setIsStreaming(false)
  }, [])

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Top Controls: Enable + Write Actions + Trigger */}
      <div className="flex items-center gap-4 px-4 py-3 rounded-lg bg-muted/50 border border-border shrink-0">
        <div className="flex items-center gap-3">
          <div className={cn(
            'w-2 h-2 rounded-full',
            statusLoading ? 'bg-muted-foreground animate-pulse' :
            status?.enabled ? 'bg-green-500' : 'bg-muted-foreground'
          )} />
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch
              checked={status?.enabled ?? false}
              onCheckedChange={(v) => toggleMutation.mutate({ enabled: v })}
              disabled={toggleMutation.isPending}
            />
            <span className="text-sm font-medium">{t('aiCortexView.enabled')}</span>
          </label>
        </div>

        <div className="w-px h-5 bg-border" />

        <label className="flex items-center gap-2 cursor-pointer">
          <Switch
            checked={status?.write_actions_enabled ?? false}
            onCheckedChange={(v) => toggleMutation.mutate({ write_actions_enabled: v })}
            disabled={toggleMutation.isPending}
          />
          <span className="text-sm">{t('aiCortexView.writeActions')}</span>
        </label>

        <div className="flex-1" />

        {status && (
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {t('aiCortexView.memoriesCount', { count: status.memory_count })}
          </span>
        )}

        <div className="w-px h-5 bg-border" />

        {isStreaming ? (
          <Button size="sm" variant="destructive" onClick={handleStopRun}>
            <Square className="w-3.5 h-3.5 mr-1.5" /> {t('aiCortexView.stop')}
          </Button>
        ) : (
          <Button
            size="sm"
            onClick={handleTriggerRun}
            disabled={!status?.enabled}
          >
            <Play className="w-3.5 h-3.5 mr-1.5" /> {t('aiCortexView.runNow')}
          </Button>
        )}
      </div>

      {/* Panel Tabs */}
      <div className="flex gap-1 border-b border-border shrink-0">
        {([
          { key: 'activity' as Panel, label: t('aiCortexView.tabs.activity'), icon: Zap },
          { key: 'memory' as Panel, label: t('aiCortexView.tabs.memory', { count: status?.memory_count ?? 0 }), icon: BookOpen },
          { key: 'settings' as Panel, label: t('aiCortexView.tabs.settings'), icon: Settings2 },
        ]).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setPanel(key)}
            className={cn(
              'flex items-center gap-1.5 px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
              panel === key
                ? 'border-primary text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30'
            )}
          >
            <Icon className="w-3.5 h-3.5" /> {label}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {panel === 'activity' && (
          <ActivityPanel
            isStreaming={isStreaming}
            streamPhase={streamPhase}
            streamContent={streamContent}
          />
        )}
        {panel === 'memory' && <MemoryPanel />}
        {panel === 'settings' && <SettingsPanel />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Activity Panel — continuous chat log
// ---------------------------------------------------------------------------

function ActivityPanel({
  isStreaming,
  streamPhase,
  streamContent,
}: {
  isStreaming: boolean
  streamPhase: 'working' | 'answering'
  streamContent: string
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const scrollRef = useRef<HTMLDivElement>(null)

  const { data: historyData, isLoading } = useQuery({
    queryKey: ['cortex-history'],
    queryFn: () => getCortexHistory(200),
    refetchInterval: isStreaming ? false : 30000,
  })

  const clearMutation = useMutation({
    mutationFn: clearCortexHistory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cortex-history'] })
      queryClient.invalidateQueries({ queryKey: ['cortex-status'] })
    },
  })

  const messages = historyData?.messages ?? []

  const assistantMessages = messages.filter(m => m.role === 'assistant')
  // Newest first
  const reversed = [...assistantMessages].reverse()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        {t('aiCortexView.loadingActivity')}
      </div>
    )
  }

  if (assistantMessages.length === 0 && !isStreaming) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <Brain className="w-10 h-10 mb-3 opacity-20" />
        <p className="text-sm">{t('aiCortexView.noActivityYet')}</p>
      </div>
    )
  }

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto p-3 space-y-3">
      {/* Clear history */}
      {assistantMessages.length > 0 && !isStreaming && (
        <div className="flex justify-end">
          <Button
            size="sm"
            variant="ghost"
            className="text-xs text-muted-foreground hover:text-red-400"
            onClick={() => clearMutation.mutate()}
            disabled={clearMutation.isPending}
          >
            <Trash2 className="w-3 h-3 mr-1" /> {t('aiCortexView.clearHistory')}
          </Button>
        </div>
      )}

      {/* Live streaming card */}
      {isStreaming && (
        <div className="rounded-lg border border-purple-500/30 bg-muted/20 overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-border/20 bg-purple-500/5">
            <Brain className="w-3.5 h-3.5 text-purple-400 animate-pulse" />
            <span className="text-xs font-medium text-purple-400">{t('aiCortexView.runningNow')}</span>
          </div>
          <div className="p-3">
            {streamPhase === 'answering' && streamContent ? (
              <MessageContent content={streamContent} isStreaming={true} />
            ) : (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>{streamContent || t('aiCortexView.startingCycle')}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Historical messages — newest first, bordered cards */}
      {reversed.map((msg) => (
        <CortexRunCard key={msg.id} message={msg} />
      ))}
    </div>
  )
}

function CortexRunCard({ message }: { message: CortexHistoryMessage }) {
  const timestamp = message.created_at ? new Date(message.created_at) : null
  const dateStr = timestamp ? timestamp.toLocaleDateString() : ''
  const timeStr = timestamp ? timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''

  return (
    <div className="rounded-lg border border-border/40 bg-muted/10 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border/20 bg-muted/30">
        <Clock className="w-3 h-3 text-muted-foreground/60" />
        <span className="text-[11px] text-muted-foreground">{dateStr} {timeStr}</span>
        {message.model_used && (
          <Badge variant="outline" className="text-[9px] ml-auto">{message.model_used}</Badge>
        )}
      </div>
      <div className="p-3">
        <MessageContent content={message.content} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Memory Panel (unchanged)
// ---------------------------------------------------------------------------

function MemoryPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [showExpired, setShowExpired] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editImportance, setEditImportance] = useState(0.5)
  const [newContent, setNewContent] = useState('')
  const [newCategory, setNewCategory] = useState('observation')
  const [showAdd, setShowAdd] = useState(false)

  const { data } = useQuery({
    queryKey: ['cortex-memory', categoryFilter, showExpired],
    queryFn: () => listCortexMemories({
      limit: 100,
      category: categoryFilter || undefined,
      include_expired: showExpired,
    }),
    refetchInterval: 30000,
  })

  const invalidateMemory = () => {
    queryClient.invalidateQueries({ queryKey: ['cortex-memory'] })
    queryClient.invalidateQueries({ queryKey: ['cortex-status'] })
  }

  const deleteMutation = useMutation({
    mutationFn: deleteCortexMemory,
    onSuccess: invalidateMemory,
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Parameters<typeof updateCortexMemory>[1] }) =>
      updateCortexMemory(id, updates),
    onSuccess: () => {
      invalidateMemory()
      setEditingId(null)
    },
  })

  const createMutation = useMutation({
    mutationFn: createCortexMemory,
    onSuccess: () => {
      invalidateMemory()
      setNewContent('')
      setShowAdd(false)
    },
  })

  const memories = data?.memories ?? []
  const categories = ['observation', 'lesson', 'rule', 'preference']
  const categoryColors: Record<string, string> = {
    observation: 'bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-300 border-blue-200 dark:border-blue-500/30',
    lesson: 'bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-300 border-green-200 dark:border-green-500/30',
    rule: 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-300 border-amber-200 dark:border-amber-500/30',
    preference: 'bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-300 border-purple-200 dark:border-purple-500/30',
  }

  return (
    <div className="space-y-3 h-full overflow-y-auto">
      {/* Filters */}
      <div className="flex items-center gap-2">
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          className="px-2 py-1 rounded bg-muted border border-border text-sm"
        >
          <option value="">{t('aiCortexView.memory.allCategories')}</option>
          {categories.map(c => <option key={c} value={c}>{t(`aiCortexView.memory.categories.${c}`)}</option>)}
        </select>
        <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
          <Switch
            checked={showExpired}
            onCheckedChange={setShowExpired}
            className="scale-75"
          />
          {t('aiCortexView.memory.showExpired')}
        </label>
        <div className="flex-1" />
        <Button size="sm" variant="outline" onClick={() => setShowAdd(!showAdd)}>
          <Plus className="w-3.5 h-3.5 mr-1" /> {t('aiCortexView.memory.addMemory')}
        </Button>
      </div>

      {/* Add Form */}
      {showAdd && (
        <Card>
          <CardContent className="p-3 space-y-2">
            <textarea
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
              placeholder={t('aiCortexView.memory.contentPlaceholder')}
              className="w-full px-3 py-2 rounded bg-muted border border-border text-sm min-h-[60px]"
            />
            <div className="flex gap-2 items-center">
              <select
                value={newCategory}
                onChange={(e) => setNewCategory(e.target.value)}
                className="px-2 py-1 rounded bg-muted border border-border text-sm"
              >
                {categories.map(c => <option key={c} value={c}>{t(`aiCortexView.memory.categories.${c}`)}</option>)}
              </select>
              <Button
                size="sm"
                onClick={() => createMutation.mutate({ content: newContent, category: newCategory })}
                disabled={!newContent.trim() || createMutation.isPending}
              >
                {t('aiCortexView.memory.save')}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)}>{t('aiCortexView.memory.cancel')}</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Memory List */}
      <div className="space-y-1.5">
        {memories.map((m) => (
          <div
            key={m.id}
            className={cn(
              'px-3 py-2 rounded border text-sm',
              m.expired ? 'opacity-40 bg-muted/30 border-border/50' : 'bg-muted/50 border-border'
            )}
          >
            {editingId === m.id ? (
              <div className="space-y-2">
                <textarea
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  className="w-full px-2 py-1 rounded bg-muted border border-border text-sm min-h-[50px]"
                />
                <div className="flex gap-2 items-center">
                  <label className="text-xs text-muted-foreground">{t('aiCortexView.memory.importanceLabel')}</label>
                  <Input
                    type="number"
                    min={0} max={1} step={0.1}
                    value={editImportance}
                    onChange={(e) => setEditImportance(Number(e.target.value))}
                    className="w-20 h-7 text-xs"
                  />
                  <Button size="sm" onClick={() => updateMutation.mutate({
                    id: m.id,
                    updates: { content: editContent, importance: editImportance },
                  })}>
                    {t('aiCortexView.memory.save')}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}>{t('aiCortexView.memory.cancel')}</Button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className={cn('text-[10px] shrink-0 mt-0.5', categoryColors[m.category])}>
                    {t(`aiCortexView.memory.categories.${m.category}`, { defaultValue: m.category })}
                  </Badge>
                  <span className="flex-1">{m.content}</span>
                  <div className="flex items-center gap-1 shrink-0">
                    <span className="text-[10px] text-muted-foreground">{m.importance.toFixed(1)}</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0"
                      onClick={() => { setEditingId(m.id); setEditContent(m.content); setEditImportance(m.importance) }}
                    >
                      <Pencil className="w-3 h-3" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
                      onClick={() => deleteMutation.mutate(m.id)}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
                <div className="text-[10px] text-muted-foreground mt-1">
                  {t('aiCortexView.memory.accessedTimes', { count: m.access_count })} | {m.updated_at && new Date(m.updated_at).toLocaleDateString()}
                </div>
              </>
            )}
          </div>
        ))}
        {memories.length === 0 && (
          <div className="text-sm text-muted-foreground text-center py-8">
            {t('aiCortexView.memory.empty')}
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Settings Panel (unchanged)
// ---------------------------------------------------------------------------

function SettingsPanel() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const { data: settings, isLoading } = useQuery({
    queryKey: ['cortex-settings'],
    queryFn: getCortexSettings,
  })

  const { data: modelsData } = useQuery({
    queryKey: ['llm-models'],
    queryFn: () => getLLMModels(),
  })

  const [form, setForm] = useState<Partial<CortexSettings>>({})
  const [dirty, setDirty] = useState(false)

  const currentSettings = { ...settings, ...form }

  const updateField = <K extends keyof CortexSettings>(key: K, value: CortexSettings[K]) => {
    setForm(prev => ({ ...prev, [key]: value }))
    setDirty(true)
  }

  const saveMutation = useMutation({
    mutationFn: () => updateCortexSettings(form),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cortex-settings'] })
      queryClient.invalidateQueries({ queryKey: ['cortex-status'] })
      setForm({})
      setDirty(false)
    },
  })

  if (isLoading) return <div className="text-sm text-muted-foreground">{t('aiCortexView.settings.loading')}</div>

  const allModels: { id: string; name: string }[] = []
  if (modelsData?.models) {
    for (const providerModels of Object.values(modelsData.models)) {
      for (const m of providerModels as Array<{ id: string; name: string }>) {
        allModels.push(m)
      }
    }
  }

  return (
    <div className="space-y-4 max-w-2xl overflow-y-auto h-full">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-muted-foreground" /> {t('aiCortexView.settings.configuration')}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">{t('aiCortexView.settings.model')}</label>
              <select
                value={currentSettings.model ?? ''}
                onChange={(e) => updateField('model', e.target.value || null)}
                className="w-full px-2 py-1.5 rounded bg-muted border border-border text-sm"
              >
                <option value="">{t('aiCortexView.settings.defaultModel')}</option>
                {allModels.map(m => <option key={m.id} value={m.id}>{m.name || m.id}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">{t('aiCortexView.settings.runIntervalSeconds')}</label>
              <Input
                type="number"
                min={30} max={3600}
                value={currentSettings.interval_seconds ?? 300}
                onChange={(e) => updateField('interval_seconds', Number(e.target.value))}
                className="h-8 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">{t('aiCortexView.settings.maxIterations')}</label>
              <Input
                type="number"
                min={1} max={50}
                value={currentSettings.max_iterations ?? 15}
                onChange={(e) => updateField('max_iterations', Number(e.target.value))}
                className="h-8 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">{t('aiCortexView.settings.temperature')}</label>
              <Input
                type="number"
                min={0} max={2} step={0.05}
                value={currentSettings.temperature ?? 0.1}
                onChange={(e) => updateField('temperature', Number(e.target.value))}
                className="h-8 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">{t('aiCortexView.settings.memoryLimit')}</label>
              <Input
                type="number"
                min={1} max={100}
                value={currentSettings.memory_limit ?? 20}
                onChange={(e) => updateField('memory_limit', Number(e.target.value))}
                className="h-8 text-sm"
              />
            </div>
          </div>

          <label className="flex items-center gap-2 pt-2 cursor-pointer">
            <Switch
              checked={currentSettings.notify_telegram ?? false}
              onCheckedChange={(v) => updateField('notify_telegram', v)}
            />
            <div>
              <span className="text-sm font-medium">{t('aiCortexView.settings.telegramNotifications')}</span>
              <span className="text-xs text-muted-foreground ml-2">{t('aiCortexView.settings.onWriteActions')}</span>
            </div>
          </label>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Brain className="w-4 h-4 text-muted-foreground" /> {t('aiCortexView.settings.mandateTitle')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <textarea
            value={currentSettings.mandate ?? ''}
            onChange={(e) => updateField('mandate', e.target.value || null)}
            placeholder={t('aiCortexView.settings.mandatePlaceholder')}
            className="w-full px-3 py-2 rounded bg-muted border border-border text-sm min-h-[120px] font-mono"
          />
        </CardContent>
      </Card>

      {dirty && (
        <div className="flex items-center gap-3">
          <Button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
            {t('aiCortexView.settings.saveSettings')}
          </Button>
          <Button variant="ghost" onClick={() => { setForm({}); setDirty(false) }}>{t('aiCortexView.settings.discard')}</Button>
        </div>
      )}
    </div>
  )
}
