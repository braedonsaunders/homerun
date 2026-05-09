import { useState, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  Bot,
  Trash2,
  Save,
  RefreshCw,
  Play,
  Shield,
  Wrench,
  ChevronRight,
  X,
  Cpu,
  Sliders,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import {
  listAgents,
  createAgent,
  updateAgent,
  deleteAgent,
  listAvailableTools,
  streamAIChat,
  type UserAgent,
} from '../../services/api'

interface AgentFormState {
  name: string
  description: string
  system_prompt: string
  tools: string[]
  model: string | null
  temperature: number
  max_iterations: number
}

const EMPTY_FORM: AgentFormState = {
  name: '',
  description: '',
  system_prompt: '',
  tools: [],
  model: null,
  temperature: 0.7,
  max_iterations: 5,
}

const MODEL_OPTIONS: Array<{ value: string; label: string; labelKey?: string }> = [
  { value: '', label: 'Default', labelKey: 'ai.agentsView.modelDefault' },
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
  { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
  { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' },
  { value: 'deepseek-chat', label: 'DeepSeek Chat' },
]

function agentToForm(agent: UserAgent): AgentFormState {
  return {
    name: agent.name,
    description: agent.description,
    system_prompt: agent.system_prompt,
    tools: agent.tools,
    model: agent.model,
    temperature: agent.temperature,
    max_iterations: agent.max_iterations,
  }
}

export default function AIAgentsView() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [form, setForm] = useState<AgentFormState>(EMPTY_FORM)
  const [testQuery, setTestQuery] = useState('')
  const [testResult, setTestResult] = useState('')
  const [testRunning, setTestRunning] = useState(false)
  const testAbortRef = useRef<AbortController | null>(null)

  const { data: agentsData, isLoading } = useQuery({
    queryKey: ['ai-agents'],
    queryFn: listAgents,
  })

  const { data: toolsData } = useQuery({
    queryKey: ['ai-available-tools'],
    queryFn: listAvailableTools,
  })

  const agents = agentsData?.agents ?? []
  const availableTools = toolsData?.tools ?? []
  const selectedAgent = agents.find(a => a.id === selectedAgentId)

  const saveMutation = useMutation({
    mutationFn: async () => {
      if (isCreating) {
        return createAgent({
          name: form.name,
          description: form.description,
          system_prompt: form.system_prompt,
          tools: form.tools,
          model: form.model,
          temperature: form.temperature,
          max_iterations: form.max_iterations,
        })
      }
      if (!selectedAgentId) throw new Error('No agent selected')
      return updateAgent(selectedAgentId, form)
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['ai-agents'] })
      if (isCreating) {
        setIsCreating(false)
        setSelectedAgentId(result.id)
      }
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async () => {
      if (!selectedAgentId) throw new Error('No agent selected')
      return deleteAgent(selectedAgentId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ai-agents'] })
      setSelectedAgentId(null)
      setForm(EMPTY_FORM)
    },
  })

  const handleSelectAgent = (agent: UserAgent) => {
    setIsCreating(false)
    setSelectedAgentId(agent.id)
    setForm(agentToForm(agent))
    setTestResult('')
    setTestQuery('')
  }

  const handleCreate = () => {
    setIsCreating(true)
    setSelectedAgentId(null)
    setForm(EMPTY_FORM)
    setTestResult('')
    setTestQuery('')
  }

  const toggleTool = (toolName: string) => {
    setForm(prev => ({
      ...prev,
      tools: prev.tools.includes(toolName)
        ? prev.tools.filter(t => t !== toolName)
        : [...prev.tools, toolName],
    }))
  }

  const handleTest = () => {
    if (!testQuery.trim()) return
    setTestRunning(true)
    setTestResult('')

    const abort = new AbortController()
    testAbortRef.current = abort

    let accumulated = ''
    streamAIChat(
      { message: testQuery, model: form.model || undefined },
      (text) => {
        accumulated += text
        setTestResult(accumulated)
      },
      () => {
        setTestRunning(false)
      },
      (error) => {
        setTestResult(prev => prev + `\n\n[Error: ${error}]`)
        setTestRunning(false)
      },
      abort.signal,
    )
  }

  const cancelTest = () => {
    testAbortRef.current?.abort()
    setTestRunning(false)
  }

  const showEditor = isCreating || selectedAgentId

  return (
    <div className="flex gap-4 h-full">
      {/* Agent List */}
      <div className="w-72 max-w-72 shrink-0 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground">{t('ai.agentsView.title')}</h3>
          <Button
            size="sm"
            onClick={handleCreate}
            className="h-7 gap-1.5 text-xs bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30"
          >
            <Plus className="w-3 h-3" />
            {t('ai.agentsView.create')}
          </Button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
          </div>
        ) : agents.length === 0 ? (
          <div className="text-center py-8">
            <Bot className="w-8 h-8 text-muted-foreground/20 mx-auto mb-2" />
            <p className="text-xs text-muted-foreground/50">{t('ai.agentsView.noneConfigured')}</p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
            <div className="space-y-1.5 pr-1">
              {agents.map(agent => (
                <button
                  key={agent.id}
                  onClick={() => handleSelectAgent(agent)}
                  className={cn(
                    'w-full min-w-0 text-left p-3 rounded-lg border transition-colors overflow-hidden',
                    selectedAgentId === agent.id
                      ? 'bg-cyan-500/10 border-cyan-500/30'
                      : 'bg-background/30 border-border/55 hover:border-border/80'
                  )}
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <div className={cn(
                      'w-7 h-7 rounded-md flex items-center justify-center shrink-0',
                      selectedAgentId === agent.id ? 'bg-cyan-500/15' : 'bg-muted/40'
                    )}>
                      <Bot className={cn(
                        'w-3.5 h-3.5',
                        selectedAgentId === agent.id ? 'text-cyan-400' : 'text-muted-foreground'
                      )} />
                    </div>
                    <div className="flex-1 min-w-0 overflow-hidden">
                      <p className="text-xs font-medium truncate">{agent.name}</p>
                      <p className="text-[10px] text-muted-foreground truncate">{agent.description || t('ai.agentsView.noDescription')}</p>
                    </div>
                    <ChevronRight className="w-3 h-3 text-muted-foreground shrink-0" />
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 mt-2 overflow-hidden">
                    {agent.model && (
                      <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-purple-500/20 bg-purple-500/10 text-purple-400 font-mono truncate max-w-[120px]">
                        {agent.model}
                      </Badge>
                    )}
                    <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-border/40 bg-muted/30">
                      {t('ai.agentsView.toolsCount', { count: agent.tools.length })}
                    </Badge>
                    {agent.is_builtin && (
                      <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-amber-500/20 bg-amber-500/10 text-amber-400">
                        {t('ai.agentsView.builtin')}
                      </Badge>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Agent Editor */}
      {showEditor ? (
        <div className="flex-1 overflow-y-auto space-y-4">
          <Card className="overflow-hidden border-border/60 bg-card/80">
            <div className="h-0.5 bg-cyan-400" />
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                <Bot className="h-4 w-4 text-cyan-400" />
                {isCreating ? t('ai.agentsView.createAgent') : t('ai.agentsView.editAgent', { name: selectedAgent?.name ?? '' })}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.agentsView.nameRequired')}</label>
                  <Input
                    value={form.name}
                    onChange={e => setForm(prev => ({ ...prev, name: e.target.value }))}
                    placeholder={t('ai.agentsView.namePlaceholder')}
                    className="bg-muted/60 rounded-lg focus-visible:ring-cyan-500"
                    disabled={selectedAgent?.is_builtin}
                  />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.agentsView.description')}</label>
                  <Input
                    value={form.description}
                    onChange={e => setForm(prev => ({ ...prev, description: e.target.value }))}
                    placeholder={t('ai.agentsView.descriptionPlaceholder')}
                    className="bg-muted/60 rounded-lg focus-visible:ring-cyan-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.agentsView.systemPrompt')}</label>
                <textarea
                  value={form.system_prompt}
                  onChange={e => setForm(prev => ({ ...prev, system_prompt: e.target.value }))}
                  placeholder={t('ai.agentsView.systemPromptPlaceholder')}
                  rows={6}
                  className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-cyan-500 resize-y"
                />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
                    <Cpu className="w-3 h-3 inline mr-1" />
                    {t('ai.agentsView.model')}
                  </label>
                  <select
                    value={form.model || ''}
                    onChange={e => setForm(prev => ({ ...prev, model: e.target.value || null }))}
                    className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-sm h-9"
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.labelKey ? t(opt.labelKey) : opt.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
                    <Sliders className="w-3 h-3 inline mr-1" />
                    {t('ai.agentsView.temperature', { value: form.temperature.toFixed(1) })}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={form.temperature}
                    onChange={e => setForm(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                    className="w-full accent-cyan-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.agentsView.maxIterations')}</label>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={form.max_iterations}
                    onChange={e => setForm(prev => ({ ...prev, max_iterations: parseInt(e.target.value) || 5 }))}
                    className="bg-muted/60 rounded-lg focus-visible:ring-cyan-500"
                  />
                </div>
              </div>

              {/* Tools */}
              <div>
                <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-2">
                  <Wrench className="w-3 h-3 inline mr-1" />
                  {t('ai.agentsView.toolsLabel', { count: form.tools.length })}
                </label>
                <div className="max-h-64 overflow-y-auto rounded-lg border border-border/40 p-2 space-y-2">
                  {(() => {
                    const grouped: Record<string, typeof availableTools> = {}
                    for (const tool of availableTools) {
                      const cat = (tool as any).category || 'general'
                      if (!grouped[cat]) grouped[cat] = []
                      grouped[cat].push(tool)
                    }
                    return Object.entries(grouped).map(([cat, tools]) => (
                      <div key={cat}>
                        <p className="text-[9px] uppercase tracking-wider text-muted-foreground/60 mb-1 px-1">{cat.replace('_', ' ')}</p>
                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-1">
                          {tools.map(tool => {
                            const isSelected = form.tools.includes(tool.name)
                            return (
                              <button
                                key={tool.name}
                                onClick={() => toggleTool(tool.name)}
                                className={cn(
                                  'text-left p-1.5 rounded-md border transition-colors text-xs',
                                  isSelected
                                    ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                                    : 'bg-background/30 border-border/40 text-muted-foreground hover:border-border/80'
                                )}
                              >
                                <p className="font-medium truncate text-[11px]">{tool.name}</p>
                                <p className="text-[9px] text-muted-foreground truncate mt-0.5">{tool.description}</p>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    ))
                  })()}
                  {availableTools.length === 0 && (
                    <p className="text-xs text-muted-foreground col-span-full py-2 text-center">{t('ai.agentsView.noToolsAvailable')}</p>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 pt-2 border-t border-border/40">
                <Button
                  onClick={() => saveMutation.mutate()}
                  disabled={!form.name.trim() || saveMutation.isPending}
                  className={cn(
                    'h-9 gap-1.5 text-sm',
                    saveMutation.isPending
                      ? 'bg-muted text-muted-foreground cursor-not-allowed'
                      : 'bg-cyan-500 hover:bg-cyan-600 text-white'
                  )}
                >
                  {saveMutation.isPending ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                  {isCreating ? t('ai.agentsView.createAgent') : t('ai.agentsView.saveChanges')}
                </Button>
                {selectedAgentId && !selectedAgent?.is_builtin && (
                  <Button
                    variant="outline"
                    onClick={() => deleteMutation.mutate()}
                    disabled={deleteMutation.isPending}
                    className="h-9 gap-1.5 text-sm text-red-400 border-red-500/30 hover:bg-red-500/10"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    {t('ai.agentsView.delete')}
                  </Button>
                )}
                {saveMutation.error && (
                  <p className="text-xs text-red-400 ml-2">{(saveMutation.error as Error).message}</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Test Section */}
          <Card className="overflow-hidden border-border/60 bg-card/80">
            <div className="h-0.5 bg-emerald-400" />
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                <Play className="h-4 w-4 text-emerald-400" />
                {t('ai.agentsView.testAgent')}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input
                  value={testQuery}
                  onChange={e => setTestQuery(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !testRunning) handleTest() }}
                  placeholder={t('ai.agentsView.testPlaceholder')}
                  className="bg-muted/60 rounded-lg focus-visible:ring-emerald-500"
                />
                {testRunning ? (
                  <Button
                    onClick={cancelTest}
                    variant="outline"
                    className="h-9 gap-1.5 text-sm text-red-400 border-red-500/30 hover:bg-red-500/10 shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                    {t('ai.agentsView.stop')}
                  </Button>
                ) : (
                  <Button
                    onClick={handleTest}
                    disabled={!testQuery.trim()}
                    className={cn(
                      'h-9 gap-1.5 text-sm shrink-0',
                      !testQuery.trim()
                        ? 'bg-muted text-muted-foreground cursor-not-allowed'
                        : 'bg-emerald-500 hover:bg-emerald-600 text-white'
                    )}
                  >
                    <Play className="w-3.5 h-3.5" />
                    {t('ai.agentsView.run')}
                  </Button>
                )}
              </div>

              {testResult && (
                <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-4">
                  <p className="text-[10px] uppercase tracking-wide text-emerald-400 mb-2">{t('ai.agentsView.response')}</p>
                  <div className="text-sm text-foreground/90 whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto">
                    {testResult}
                  </div>
                  {testRunning && (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-400 mt-2" />
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Shield className="w-12 h-12 text-muted-foreground/20 mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">{t('ai.agentsView.selectOrCreate')}</p>
          </div>
        </div>
      )}
    </div>
  )
}
