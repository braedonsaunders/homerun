import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  Plus,
  Wrench,
  Trash2,
  Save,
  RefreshCw,
  ChevronRight,
  Code,
  Shield,
  Search,
  Newspaper,
  BookOpen,
  BarChart3,
  Zap,
  Globe,
  Settings,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'
import {
  listAvailableTools,
} from '../../services/api'

// Until backend CRUD is ready, we use the existing available-tools endpoint
// and provide a management UI for viewing/toggling tools

interface ToolItem {
  name: string
  description: string
  tool_type?: string
  is_builtin?: boolean
  enabled?: boolean
  parameters_schema?: Record<string, unknown> | null
  implementation?: string | null
  id?: string
  category?: string
}

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  market_data: <BarChart3 className="w-4 h-4" />,
  portfolio: <BookOpen className="w-4 h-4" />,
  trading: <Zap className="w-4 h-4" />,
  wallets: <Search className="w-4 h-4" />,
  strategies: <Shield className="w-4 h-4" />,
  news: <Newspaper className="w-4 h-4" />,
  analytics: <BarChart3 className="w-4 h-4" />,
  web: <Globe className="w-4 h-4" />,
  system: <Settings className="w-4 h-4" />,
  general: <Wrench className="w-4 h-4" />,
}

const TOOL_ICONS: Record<string, React.ReactNode> = {
  get_market_details: <BarChart3 className="w-4 h-4" />,
  search_news: <Newspaper className="w-4 h-4" />,
  analyze_resolution: <Shield className="w-4 h-4" />,
  check_orderbook: <BookOpen className="w-4 h-4" />,
  find_related_markets: <Search className="w-4 h-4" />,
}

interface ToolFormState {
  name: string
  description: string
  tool_type: string
  parameters_schema: string
  implementation: string
}

const EMPTY_FORM: ToolFormState = {
  name: '',
  description: '',
  tool_type: 'function',
  parameters_schema: '{}',
  implementation: '',
}

export default function AIToolsView() {
  const { t } = useTranslation()
  const [selectedToolName, setSelectedToolName] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [form, setForm] = useState<ToolFormState>(EMPTY_FORM)
  const [searchFilter, setSearchFilter] = useState('')

  const { data: toolsData, isLoading } = useQuery({
    queryKey: ['ai-available-tools'],
    queryFn: listAvailableTools,
  })

  const tools: ToolItem[] = (toolsData?.tools ?? []).map((t: any) => ({
    name: t.name,
    description: t.description,
    tool_type: t.tool_type || 'function',
    is_builtin: t.is_builtin ?? true,
    enabled: t.enabled ?? true,
    id: t.id || t.name,
    category: t.category || 'general',
  }))

  const filteredTools = searchFilter
    ? tools.filter(t =>
        t.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
        t.description?.toLowerCase().includes(searchFilter.toLowerCase())
      )
    : tools

  const selectedTool = tools.find(t => t.name === selectedToolName)

  const handleSelectTool = (tool: ToolItem) => {
    setIsCreating(false)
    setSelectedToolName(tool.name)
    setForm({
      name: tool.name,
      description: tool.description || '',
      tool_type: tool.tool_type || 'function',
      parameters_schema: tool.parameters_schema ? JSON.stringify(tool.parameters_schema, null, 2) : '{}',
      implementation: tool.implementation || '',
    })
  }

  const handleCreate = () => {
    setIsCreating(true)
    setSelectedToolName(null)
    setForm(EMPTY_FORM)
  }

  const showEditor = isCreating || selectedToolName

  return (
    <div className="flex gap-4 h-full">
      {/* Tool List */}
      <div className="w-72 max-w-72 shrink-0 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground">{t('ai.toolsView.title')}</h3>
          <Button
            size="sm"
            onClick={handleCreate}
            className="h-7 gap-1.5 text-xs bg-violet-500/20 text-violet-700 dark:text-violet-300 border border-violet-500/30 hover:bg-violet-500/30"
          >
            <Plus className="w-3 h-3" />
            {t('ai.toolsView.create')}
          </Button>
        </div>

        <div className="mb-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
            <Input
              value={searchFilter}
              onChange={e => setSearchFilter(e.target.value)}
              placeholder={t('ai.toolsView.filterPlaceholder')}
              className="h-8 pl-8 text-xs bg-muted/60 rounded-lg"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-5 h-5 animate-spin text-violet-400" />
          </div>
        ) : filteredTools.length === 0 ? (
          <div className="text-center py-8">
            <Wrench className="w-8 h-8 text-muted-foreground/20 mx-auto mb-2" />
            <p className="text-xs text-muted-foreground/50">
              {searchFilter ? t('ai.toolsView.noMatch') : t('ai.toolsView.noneConfigured')}
            </p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
            <div className="space-y-3 pr-1">
              {(() => {
                const grouped: Record<string, ToolItem[]> = {}
                for (const tool of filteredTools) {
                  const cat = tool.category || 'general'
                  if (!grouped[cat]) grouped[cat] = []
                  grouped[cat].push(tool)
                }
                return Object.entries(grouped).map(([cat, catTools]) => (
                  <div key={cat}>
                    <div className="flex items-center gap-1.5 mb-1.5 px-1">
                      <span className="text-muted-foreground/60">{CATEGORY_ICONS[cat] || <Wrench className="w-3.5 h-3.5" />}</span>
                      <p className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-medium">{cat.replace('_', ' ')} ({catTools.length})</p>
                    </div>
                    <div className="space-y-1">
              {catTools.map(tool => (
                <button
                  key={tool.name}
                  onClick={() => handleSelectTool(tool)}
                  className={cn(
                    'w-full min-w-0 text-left p-2.5 rounded-lg border transition-colors overflow-hidden',
                    selectedToolName === tool.name
                      ? 'bg-violet-500/10 border-violet-500/30'
                      : 'bg-background/30 border-border/55 hover:border-border/80'
                  )}
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <div className={cn(
                      'w-7 h-7 rounded-md flex items-center justify-center shrink-0',
                      selectedToolName === tool.name ? 'bg-violet-500/15' : 'bg-muted/40'
                    )}>
                      <span className={cn(
                        selectedToolName === tool.name ? 'text-violet-400' : 'text-muted-foreground'
                      )}>
                        {TOOL_ICONS[tool.name] || CATEGORY_ICONS[tool.category || 'general'] || <Wrench className="w-3.5 h-3.5" />}
                      </span>
                    </div>
                    <div className="flex-1 min-w-0 overflow-hidden">
                      <p className="text-xs font-medium truncate font-mono">{tool.name}</p>
                      <p className="text-[10px] text-muted-foreground truncate">{tool.description || t('ai.agentsView.noDescription')}</p>
                    </div>
                    <ChevronRight className="w-3 h-3 text-muted-foreground shrink-0" />
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 mt-2">
                    <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-border/40 bg-muted/30">
                      {tool.tool_type || 'function'}
                    </Badge>
                    {tool.is_builtin && (
                      <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-amber-500/20 bg-amber-500/10 text-amber-400">
                        {t('ai.toolsView.builtin')}
                      </Badge>
                    )}
                    {tool.enabled === false && (
                      <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-red-500/20 bg-red-500/10 text-red-400">
                        {t('ai.toolsView.disabled')}
                      </Badge>
                    )}
                  </div>
                </button>
              ))}
                    </div>
                  </div>
                ))
              })()}
            </div>
          </div>
        )}
      </div>

      {/* Tool Editor */}
      {showEditor ? (
        <div className="flex-1 overflow-y-auto space-y-4">
          <Card className="overflow-hidden border-border/60 bg-card/80">
            <div className="h-0.5 bg-violet-400" />
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                <Wrench className="h-4 w-4 text-violet-400" />
                {isCreating ? t('ai.toolsView.createTool') : t('ai.toolsView.toolHeader', { name: selectedTool?.name ?? '' })}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.toolsView.nameRequired')}</label>
                  <Input
                    value={form.name}
                    onChange={e => setForm(prev => ({ ...prev, name: e.target.value }))}
                    placeholder={t('ai.toolsView.namePlaceholder')}
                    className="bg-muted/60 rounded-lg focus-visible:ring-violet-500 font-mono text-sm"
                    disabled={!isCreating && selectedTool?.is_builtin}
                  />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.toolsView.type')}</label>
                  <select
                    value={form.tool_type}
                    onChange={e => setForm(prev => ({ ...prev, tool_type: e.target.value }))}
                    className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-sm h-9"
                    disabled={!isCreating && selectedTool?.is_builtin}
                  >
                    <option value="function">{t('ai.toolsView.typeFunction')}</option>
                    <option value="api">{t('ai.toolsView.typeApi')}</option>
                    <option value="composite">{t('ai.toolsView.typeComposite')}</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">{t('ai.toolsView.description')}</label>
                <textarea
                  value={form.description}
                  onChange={e => setForm(prev => ({ ...prev, description: e.target.value }))}
                  placeholder={t('ai.toolsView.descriptionPlaceholder')}
                  rows={3}
                  className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500 resize-y"
                  disabled={!isCreating && selectedTool?.is_builtin}
                />
              </div>

              <div>
                <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
                  <Code className="w-3 h-3 inline mr-1" />
                  {t('ai.toolsView.paramsSchema')}
                </label>
                <textarea
                  value={form.parameters_schema}
                  onChange={e => setForm(prev => ({ ...prev, parameters_schema: e.target.value }))}
                  placeholder='{"type": "object", "properties": { ... }}'
                  rows={6}
                  className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-violet-500 resize-y"
                  disabled={!isCreating && selectedTool?.is_builtin}
                />
              </div>

              {!selectedTool?.is_builtin && (
                <div>
                  <label className="block text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
                    <Code className="w-3 h-3 inline mr-1" />
                    {t('ai.toolsView.implementation')}
                  </label>
                  <textarea
                    value={form.implementation}
                    onChange={e => setForm(prev => ({ ...prev, implementation: e.target.value }))}
                    placeholder={t('ai.toolsView.implPlaceholder')}
                    rows={8}
                    className="w-full bg-muted/60 border border-border rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-violet-500 resize-y"
                  />
                </div>
              )}

              {/* Actions */}
              {(!selectedTool?.is_builtin || isCreating) && (
                <div className="flex items-center gap-2 pt-2 border-t border-border/40">
                  <Button
                    disabled={!form.name.trim()}
                    className={cn(
                      'h-9 gap-1.5 text-sm',
                      !form.name.trim()
                        ? 'bg-muted text-muted-foreground cursor-not-allowed'
                        : 'bg-violet-500 hover:bg-violet-600 text-white'
                    )}
                  >
                    <Save className="w-3.5 h-3.5" />
                    {isCreating ? t('ai.toolsView.createBtn') : t('ai.toolsView.saveChanges')}
                  </Button>
                  {selectedToolName && !selectedTool?.is_builtin && (
                    <Button
                      variant="outline"
                      className="h-9 gap-1.5 text-sm text-red-400 border-red-500/30 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      {t('ai.toolsView.delete')}
                    </Button>
                  )}
                </div>
              )}

              {selectedTool?.is_builtin && !isCreating && (
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
                  <p className="text-xs text-amber-300/90">
                    <Shield className="w-3 h-3 inline mr-1.5" />
                    {t('ai.toolsView.builtinNotice')}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Wrench className="w-12 h-12 text-muted-foreground/20 mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">{t('ai.toolsView.selectOrCreate')}</p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              {t('ai.toolsView.toolsDescription')}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
