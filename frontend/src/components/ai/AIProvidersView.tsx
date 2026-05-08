import { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Eye,
  EyeOff,
  RefreshCw,
  DollarSign,
  Cpu,
  Cloud,
  Sparkles,
  Bot,
  Zap,
  Globe,
  Server,
  ChevronDown,
  ExternalLink,
  Check,
  ChevronsUpDown,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { Card, CardContent } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Separator } from '../ui/separator'
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '../ui/command'
import {
  getSettings,
  updateSettings,
  getLLMModels,
  refreshLLMModels,
  testLLMConnection,
  type LLMModelOption,
} from '../../services/api'

// ---------------------------------------------------------------------------
// Types & constants
// ---------------------------------------------------------------------------

interface ProviderConfig {
  id: string
  name: string
  icon: React.ElementType
  keyField: string
  keyPlaceholder: string
  hasBaseUrl?: boolean
  baseUrlField?: string
  baseUrlDefault?: string
  isLocal?: boolean
  description: string
}

const PROVIDERS: ProviderConfig[] = [
  {
    id: 'openai',
    name: 'OpenAI',
    icon: Sparkles,
    keyField: 'openai_api_key',
    keyPlaceholder: 'sk-...',
    description: 'GPT-4o, GPT-4o-mini, o1, o3',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    icon: Bot,
    keyField: 'anthropic_api_key',
    keyPlaceholder: 'sk-ant-...',
    description: 'Claude Opus, Sonnet, Haiku',
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    icon: ExternalLink,
    keyField: 'openrouter_api_key',
    keyPlaceholder: 'sk-or-...',
    description: '500+ models, unified API',
  },
  {
    id: 'google',
    name: 'Google Gemini',
    icon: Globe,
    keyField: 'google_api_key',
    keyPlaceholder: 'AIza...',
    description: 'Gemini 2.0, Gemini Flash',
  },
  {
    id: 'xai',
    name: 'xAI Grok',
    icon: Zap,
    keyField: 'xai_api_key',
    keyPlaceholder: 'xai-...',
    description: 'Grok-2, Grok-3',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    icon: Cloud,
    keyField: 'deepseek_api_key',
    keyPlaceholder: 'sk-...',
    description: 'DeepSeek V3, R1',
  },
  {
    id: 'ollama',
    name: 'Ollama',
    icon: Server,
    keyField: 'ollama_api_key',
    keyPlaceholder: 'Optional',
    hasBaseUrl: true,
    baseUrlField: 'ollama_base_url',
    baseUrlDefault: 'http://localhost:11434',
    isLocal: true,
    description: 'Local models via Ollama',
  },
  {
    id: 'lmstudio',
    name: 'LM Studio',
    icon: Cpu,
    keyField: 'lmstudio_api_key',
    keyPlaceholder: 'Optional',
    hasBaseUrl: true,
    baseUrlField: 'lmstudio_base_url',
    baseUrlDefault: 'http://localhost:1234/v1',
    isLocal: true,
    description: 'Local models via LM Studio',
  },
  {
    id: 'nvidia',
    name: 'NVIDIA NIM',
    icon: Sparkles,
    keyField: 'nvidia_api_key',
    keyPlaceholder: 'nvapi-...',
    hasBaseUrl: true,
    baseUrlField: 'nvidia_base_url',
    baseUrlDefault: 'https://integrate.api.nvidia.com/v1',
    description: 'Llama 3.x, Nemotron, Mixtral via build.nvidia.com',
  },
]

type TestStatus = { status: 'idle' | 'testing' | 'success' | 'error'; message?: string }

// ---------------------------------------------------------------------------
// Searchable Model Combobox
// ---------------------------------------------------------------------------

function ModelCombobox({
  models,
  value,
  onChange,
  disabled,
  placeholder = 'Select a model...',
}: {
  models: LLMModelOption[]
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  placeholder?: string
}) {
  const [open, setOpen] = useState(false)
  const selected = models.find(m => m.id === value)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className="w-full justify-between bg-muted/60 border-border text-sm h-9 font-normal"
        >
          <span className="truncate">
            {selected ? selected.name : value || placeholder}
          </span>
          <ChevronsUpDown className="ml-2 h-3.5 w-3.5 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[350px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search models..." className="h-9" />
          <CommandList>
            <CommandEmpty>No models found.</CommandEmpty>
            <CommandGroup className="max-h-[300px] overflow-auto">
              {models.map(m => (
                <CommandItem
                  key={m.id}
                  value={m.name}
                  onSelect={() => {
                    onChange(m.id)
                    setOpen(false)
                  }}
                >
                  <Check className={cn('mr-2 h-3.5 w-3.5', value === m.id ? 'opacity-100' : 'opacity-0')} />
                  <span className="truncate">{m.name}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

// ---------------------------------------------------------------------------
// Provider Row (expandable)
// ---------------------------------------------------------------------------

function ProviderRow({
  provider,
  isPrimary,
  isExpanded,
  onToggleExpand,
  connectionStatus,
  testStatus,
  apiKey,
  baseUrl,
  showSecret,
  onKeyChange,
  onBaseUrlChange,
  onToggleSecret,
  onTest,
  onSave,
  isSaving,
}: {
  provider: ProviderConfig
  isPrimary: boolean
  isExpanded: boolean
  onToggleExpand: () => void
  connectionStatus: 'connected' | 'not_configured' | 'error'
  testStatus?: TestStatus
  apiKey: string
  baseUrl: string
  showSecret: boolean
  onKeyChange: (v: string) => void
  onBaseUrlChange: (v: string) => void
  onToggleSecret: () => void
  onTest: () => void
  onSave: () => void
  isSaving: boolean
}) {
  const Icon = provider.icon

  return (
    <div
      className={cn(
        'border border-border/40 rounded-lg transition-colors',
        isPrimary && 'border-purple-500/40 bg-purple-500/5',
        isExpanded && 'bg-muted/20',
      )}
    >
      {/* Summary row */}
      <button
        onClick={onToggleExpand}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-muted/30 rounded-lg transition-colors"
      >
        <Icon className={cn('w-4 h-4 shrink-0', isPrimary ? 'text-purple-400' : 'text-muted-foreground')} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">{provider.name}</span>
            {isPrimary && (
              <Badge variant="outline" className="text-[9px] py-0 bg-purple-500/10 text-purple-400 border-purple-500/30">
                Primary
              </Badge>
            )}
            {provider.isLocal && (
              <Badge variant="outline" className="text-[9px] py-0 text-muted-foreground border-border/50">
                Local
              </Badge>
            )}
          </div>
          <p className="text-[11px] text-muted-foreground">{provider.description}</p>
        </div>
        <StatusDot status={connectionStatus} />
        <ChevronDown className={cn('w-4 h-4 text-muted-foreground transition-transform', isExpanded && 'rotate-180')} />
      </button>

      {/* Expanded config */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-1 space-y-3 border-t border-border/20">
          {/* Base URL */}
          {provider.hasBaseUrl && provider.baseUrlField && (
            <div>
              <Label className="text-xs text-muted-foreground">Base URL</Label>
              <Input
                type="text"
                value={baseUrl}
                onChange={e => onBaseUrlChange(e.target.value)}
                placeholder={provider.baseUrlDefault}
                className="mt-1 text-xs font-mono h-8"
              />
            </div>
          )}

          {/* API Key */}
          <div>
            <Label className="text-xs text-muted-foreground">
              {provider.isLocal ? 'API Key (Optional)' : 'API Key'}
            </Label>
            <div className="relative mt-1">
              <Input
                type={showSecret ? 'text' : 'password'}
                value={apiKey}
                onChange={e => onKeyChange(e.target.value)}
                placeholder={provider.keyPlaceholder}
                className="pr-10 font-mono text-xs h-8"
              />
              <button
                type="button"
                className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-foreground"
                onClick={onToggleSecret}
              >
                {showSecret ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
              </button>
            </div>
          </div>

          {/* Test message */}
          {testStatus?.message && (
            <p className={cn('text-[11px]', testStatus.status === 'success' ? 'text-green-400' : 'text-red-400')}>
              {testStatus.message}
            </p>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={onTest}
              disabled={testStatus?.status === 'testing'}
              className="text-xs h-7"
            >
              {testStatus?.status === 'testing' ? (
                <RefreshCw className="w-3 h-3 mr-1.5 animate-spin" />
              ) : (
                <Zap className="w-3 h-3 mr-1.5" />
              )}
              Test
            </Button>
            <Button size="sm" onClick={onSave} disabled={isSaving} className="text-xs h-7">
              Save
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusDot({ status }: { status: 'connected' | 'not_configured' | 'error' }) {
  return (
    <div
      className={cn(
        'w-2 h-2 rounded-full shrink-0',
        status === 'connected' && 'bg-emerald-400',
        status === 'error' && 'bg-red-400',
        status === 'not_configured' && 'bg-muted-foreground/30',
      )}
      title={status === 'connected' ? 'Connected' : status === 'error' ? 'Error' : 'Not configured'}
    />
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function AIProvidersView() {
  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const [availableModels, setAvailableModels] = useState<Record<string, LLMModelOption[]>>({})
  const [isRefreshingModels, setIsRefreshingModels] = useState(false)

  // Form state
  const [primaryProvider, setPrimaryProvider] = useState('none')
  const [primaryModel, setPrimaryModel] = useState('')
  const [maxMonthlySpend, setMaxMonthlySpend] = useState(50)
  const [keys, setKeys] = useState<Record<string, string>>({})
  const [dirtyKeys, setDirtyKeys] = useState<Record<string, boolean>>({})
  const [baseUrls, setBaseUrls] = useState<Record<string, string>>({})
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [testStatuses, setTestStatuses] = useState<Record<string, TestStatus>>({})
  const [expandedProvider, setExpandedProvider] = useState<string | null>(null)
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Sync from settings
  useEffect(() => {
    if (!settings) return
    const llm = settings.llm
    setPrimaryProvider(llm?.provider || 'none')
    setPrimaryModel(llm?.model || '')
    setMaxMonthlySpend(llm?.max_monthly_spend ?? 50)
    setBaseUrls({
      ollama_base_url: llm?.ollama_base_url || '',
      lmstudio_base_url: llm?.lmstudio_base_url || '',
      nvidia_base_url: llm?.nvidia_base_url || '',
    })
    // Populate keys from server masked values (e.g. "sk-p***") so
    // the input shows that a key is configured. Only overwrite keys
    // that the user hasn't touched this session (i.e. still empty).
    setKeys(prev => {
      const next = { ...prev }
      for (const p of PROVIDERS) {
        const serverVal = (llm as any)?.[p.keyField] as string | null | undefined
        // Only backfill from server if user hasn't typed anything locally
        if (!next[p.keyField] && serverVal) {
          next[p.keyField] = serverVal
        }
      }
      return next
    })
  }, [settings])

  // Load models
  useEffect(() => {
    getLLMModels()
      .then(res => { if (res.models) setAvailableModels(res.models) })
      .catch(() => {})
  }, [])

  const handleRefreshModels = useCallback(async () => {
    setIsRefreshingModels(true)
    try {
      const provider = primaryProvider !== 'none' ? primaryProvider : undefined
      const res = await refreshLLMModels(provider)
      if (res.models) setAvailableModels(res.models)
    } catch {
      // ignore
    } finally {
      setIsRefreshingModels(false)
    }
  }, [primaryProvider])

  // All models for the primary provider
  const modelsForProvider = useMemo(() => {
    return availableModels[primaryProvider] || []
  }, [availableModels, primaryProvider])



  // Mutations
  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      queryClient.invalidateQueries({ queryKey: ['ai-status'] })
      queryClient.invalidateQueries({ queryKey: ['ai-usage'] })
      setSaveMessage({ type: 'success', text: 'Settings saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save' })
      setTimeout(() => setSaveMessage(null), 5000)
    },
  })

  const getConnectionStatus = (provider: ProviderConfig): 'connected' | 'not_configured' | 'error' => {
    const test = testStatuses[provider.id]
    if (test?.status === 'success') return 'connected'
    if (test?.status === 'error') return 'error'
    const serverKey = settings?.llm?.[provider.keyField as keyof typeof settings.llm]
    if (serverKey) return 'connected'
    return 'not_configured'
  }

  const handleTestProvider = async (providerId: string) => {
    setTestStatuses(p => ({ ...p, [providerId]: { status: 'testing' } }))
    try {
      const res = await testLLMConnection(providerId)
      setTestStatuses(p => ({
        ...p,
        [providerId]: {
          status: res.status === 'success' ? 'success' : 'error',
          message: res.message,
        },
      }))
    } catch (err: any) {
      setTestStatuses(p => ({
        ...p,
        [providerId]: { status: 'error', message: err.message || 'Connection failed' },
      }))
    }
  }

  const handleSaveProvider = (provider: ProviderConfig) => {
    const llmUpdate: Record<string, unknown> = {}
    // Only send the key if the user actually typed a new value
    if (dirtyKeys[provider.keyField]) {
      const keyVal = keys[provider.keyField]
      if (keyVal) llmUpdate[provider.keyField] = keyVal
    }
    if (provider.hasBaseUrl && provider.baseUrlField) {
      llmUpdate[provider.baseUrlField] = baseUrls[provider.baseUrlField] || null
    }
    if (Object.keys(llmUpdate).length === 0) {
      setSaveMessage({ type: 'error', text: 'No changes to save' })
      setTimeout(() => setSaveMessage(null), 3000)
      return
    }
    saveMutation.mutate({ llm: llmUpdate as any }, {
      onSuccess: () => {
        // Clear dirty flag so the masked value from server is accepted
        setDirtyKeys(p => ({ ...p, [provider.keyField]: false }))
        // Clear local key so the masked server value gets backfilled
        setKeys(p => {
          const next = { ...p }
          delete next[provider.keyField]
          return next
        })
      },
    })
  }

  const handleSaveGlobal = () => {
    saveMutation.mutate({
      llm: {
        provider: primaryProvider,
        model: primaryModel || '',
        max_monthly_spend: maxMonthlySpend,
      },
    })
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold tracking-tight">Providers</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Configure LLM providers and select your primary model.
          </p>
        </div>
        {saveMessage && (
          <Badge
            variant="outline"
            className={cn(
              'text-xs',
              saveMessage.type === 'success'
                ? 'bg-green-500/10 text-green-400 border-green-500/30'
                : 'bg-red-500/10 text-red-400 border-red-500/30',
            )}
          >
            {saveMessage.text}
          </Badge>
        )}
      </div>

      {/* Primary config */}
      <Card className="border-border/40 bg-card/50">
        <CardContent className="p-4 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Provider selector with search */}
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Primary Provider</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    className="w-full justify-between bg-muted/60 border-border text-sm h-9 font-normal"
                  >
                    {primaryProvider === 'none'
                      ? 'None (Disabled)'
                      : PROVIDERS.find(p => p.id === primaryProvider)?.name || primaryProvider}
                    <ChevronsUpDown className="ml-2 h-3.5 w-3.5 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[250px] p-0" align="start">
                  <Command>
                    <CommandInput placeholder="Search providers..." className="h-9" />
                    <CommandList>
                      <CommandEmpty>No provider found.</CommandEmpty>
                      <CommandGroup>
                        <CommandItem
                          value="None Disabled"
                          onSelect={() => { setPrimaryProvider('none'); setPrimaryModel('') }}
                        >
                          <Check className={cn('mr-2 h-3.5 w-3.5', primaryProvider === 'none' ? 'opacity-100' : 'opacity-0')} />
                          None (Disabled)
                        </CommandItem>
                        {PROVIDERS.map(p => (
                          <CommandItem
                            key={p.id}
                            value={p.name}
                            onSelect={() => { setPrimaryProvider(p.id); setPrimaryModel('') }}
                          >
                            <Check className={cn('mr-2 h-3.5 w-3.5', primaryProvider === p.id ? 'opacity-100' : 'opacity-0')} />
                            <p.icon className="mr-2 h-3.5 w-3.5 text-muted-foreground" />
                            {p.name}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>

            {/* Model selector with search */}
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Model</Label>
              <div className="flex gap-1.5">
                <div className="flex-1">
                  <ModelCombobox
                    models={modelsForProvider}
                    value={primaryModel}
                    onChange={setPrimaryModel}
                    disabled={primaryProvider === 'none'}
                    placeholder={
                      primaryProvider === 'none'
                        ? 'Select provider first'
                        : modelsForProvider.length === 0
                          ? 'Click refresh to fetch'
                          : 'Search models...'
                    }
                  />
                </div>
                <Button
                  variant="secondary"
                  size="icon"
                  onClick={handleRefreshModels}
                  disabled={isRefreshingModels || primaryProvider === 'none'}
                  title="Refresh models"
                  className="h-9 w-9"
                >
                  <RefreshCw className={cn('w-3.5 h-3.5', isRefreshingModels && 'animate-spin')} />
                </Button>
              </div>
              <p className="text-[11px] text-muted-foreground/70 mt-1">
                {modelsForProvider.length > 0 ? `${modelsForProvider.length} models` : ''}
              </p>
            </div>

            {/* Spend limit */}
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Monthly Spend Limit</Label>
              <div className="flex items-center gap-2">
                <DollarSign className="w-4 h-4 text-muted-foreground shrink-0" />
                <Input
                  type="number"
                  min={0}
                  step={5}
                  value={maxMonthlySpend}
                  onChange={e => setMaxMonthlySpend(parseFloat(e.target.value) || 0)}
                  className="text-sm h-9"
                />
              </div>
              <p className="text-[11px] text-muted-foreground/70 mt-1">0 = unlimited</p>
            </div>
          </div>

          <div className="flex justify-end">
            <Button size="sm" onClick={handleSaveGlobal} disabled={saveMutation.isPending} className="h-8">
              Save Primary Settings
            </Button>
          </div>
        </CardContent>
      </Card>

      <Separator className="opacity-30" />

      {/* Provider list — compact expandable rows */}
      <div className="space-y-2">
        {PROVIDERS.map(provider => (
          <ProviderRow
            key={provider.id}
            provider={provider}
            isPrimary={primaryProvider === provider.id}
            isExpanded={expandedProvider === provider.id}
            onToggleExpand={() => setExpandedProvider(expandedProvider === provider.id ? null : provider.id)}
            connectionStatus={getConnectionStatus(provider)}
            testStatus={testStatuses[provider.id]}
            apiKey={keys[provider.keyField] || ''}
            baseUrl={baseUrls[provider.baseUrlField || ''] || ''}
            showSecret={!!showSecrets[provider.id]}
            onKeyChange={v => {
              setKeys(p => ({ ...p, [provider.keyField]: v }))
              setDirtyKeys(p => ({ ...p, [provider.keyField]: true }))
            }}
            onBaseUrlChange={v => setBaseUrls(p => ({ ...p, [provider.baseUrlField!]: v }))}
            onToggleSecret={() => setShowSecrets(p => ({ ...p, [provider.id]: !p[provider.id] }))}
            onTest={() => handleTestProvider(provider.id)}
            onSave={() => handleSaveProvider(provider)}
            isSaving={saveMutation.isPending}
          />
        ))}
      </div>
    </div>
  )
}
