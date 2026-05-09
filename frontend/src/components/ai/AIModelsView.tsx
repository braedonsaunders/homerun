import { useState, useEffect, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { RefreshCw, Save, Cpu, CheckCircle, Check, ChevronsUpDown } from 'lucide-react'
import { cn } from '../../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Switch } from '../ui/switch'
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
  type AllSettings,
  type LLMModelOption,
} from '../../services/api'

interface PurposeConfig {
  key: string
  labelKey: string
  descKey: string
}

const PURPOSES: PurposeConfig[] = [
  { key: 'chat', labelKey: 'ai.modelsView.purpose.chat', descKey: 'ai.modelsView.purpose.chatDesc' },
  { key: 'news_analysis', labelKey: 'ai.modelsView.purpose.newsAnalysis', descKey: 'ai.modelsView.purpose.newsAnalysisDesc' },
  { key: 'resolution_analysis', labelKey: 'ai.modelsView.purpose.resolutionAnalysis', descKey: 'ai.modelsView.purpose.resolutionAnalysisDesc' },
  { key: 'opportunity_judgment', labelKey: 'ai.modelsView.purpose.opportunityJudgment', descKey: 'ai.modelsView.purpose.opportunityJudgmentDesc' },
  { key: 'market_analysis', labelKey: 'ai.modelsView.purpose.marketAnalysis', descKey: 'ai.modelsView.purpose.marketAnalysisDesc' },
  { key: 'agent_execution', labelKey: 'ai.modelsView.purpose.agentExecution', descKey: 'ai.modelsView.purpose.agentExecutionDesc' },
  { key: 'strategy_intelligence', labelKey: 'ai.modelsView.purpose.strategyIntelligence', descKey: 'ai.modelsView.purpose.strategyIntelligenceDesc' },
  { key: 'strategy_reverse_engineer', labelKey: 'ai.modelsView.purpose.strategyReverseEngineer', descKey: 'ai.modelsView.purpose.strategyReverseEngineerDesc' },
]

function ModelCombobox({
  models,
  value,
  onChange,
}: {
  models: LLMModelOption[]
  value: string
  onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const selected = value === '__default__' ? null : models.find(m => m.id === value)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between bg-muted/60 border-border text-xs h-8 font-normal"
        >
          <span className="truncate text-left">
            {selected ? selected.name : value === '__default__' ? t('ai.modelsView.modelDefault') : value || t('ai.modelsView.modelDefault')}
          </span>
          <ChevronsUpDown className="ml-2 h-3 w-3 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0" align="start">
        <Command>
          <CommandInput placeholder={t('ai.providersView.searchModels')} className="h-8 text-xs" />
          <CommandList>
            <CommandEmpty>{t('ai.providersView.noModels')}</CommandEmpty>
            <CommandGroup className="max-h-[250px] overflow-auto">
              <CommandItem
                value="Default primary model"
                onSelect={() => { onChange('__default__'); setOpen(false) }}
              >
                <Check className={cn('mr-2 h-3 w-3', value === '__default__' ? 'opacity-100' : 'opacity-0')} />
                <span className="text-muted-foreground">{t('ai.modelsView.modelDefault')}</span>
              </CommandItem>
              {models.map(m => (
                <CommandItem
                  key={m.id}
                  value={m.name}
                  onSelect={() => { onChange(m.id); setOpen(false) }}
                >
                  <Check className={cn('mr-2 h-3 w-3', value === m.id ? 'opacity-100' : 'opacity-0')} />
                  <span className="truncate text-xs">{m.name}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

export default function AIModelsView() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const [modelAssignments, setModelAssignments] = useState<Record<string, string>>({})
  const [enabledFeatures, setEnabledFeatures] = useState<Record<string, boolean>>({})
  const [dirty, setDirty] = useState(false)

  const { data: settings, isLoading: settingsLoading } = useQuery<AllSettings>({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const { data: modelsData, isLoading: modelsLoading } = useQuery({
    queryKey: ['llm-models'],
    queryFn: () => getLLMModels(),
  })

  const refreshMutation = useMutation({
    mutationFn: () => refreshLLMModels(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-models'] })
    },
  })

  const saveMutation = useMutation({
    mutationFn: () =>
      updateSettings({
        llm: {
          model_assignments: modelAssignments,
          enabled_features: enabledFeatures,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setDirty(false)
    },
  })

  // Sync local state from settings
  useEffect(() => {
    if (settings?.llm) {
      setModelAssignments(settings.llm.model_assignments ?? {})
      setEnabledFeatures(settings.llm.enabled_features ?? {})
    }
  }, [settings])

  // Flatten all models from all providers into one list
  const allModels = useMemo(() => {
    const models: LLMModelOption[] = []
    if (modelsData?.models) {
      for (const providerModels of Object.values(modelsData.models)) {
        for (const m of providerModels) {
          if (!models.some(x => x.id === m.id)) {
            models.push(m)
          }
        }
      }
    }
    models.sort((a, b) => a.name.localeCompare(b.name))
    return models
  }, [modelsData])

  const handleModelChange = (purposeKey: string, modelId: string) => {
    setModelAssignments(prev => {
      const next = { ...prev }
      if (modelId === '__default__') {
        delete next[purposeKey]
      } else {
        next[purposeKey] = modelId
      }
      return next
    })
    setDirty(true)
  }

  const handleToggle = (purposeKey: string, enabled: boolean) => {
    setEnabledFeatures(prev => ({ ...prev, [purposeKey]: enabled }))
    setDirty(true)
  }

  const isLoading = settingsLoading || modelsLoading

  return (
    <div className="space-y-4">
      <Card className="overflow-hidden border-border/60 bg-card/80">
        <div className="h-0.5 bg-violet-400" />
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Cpu className="h-4 w-4 text-violet-400" />
              {t('ai.modelsView.title')}
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => refreshMutation.mutate()}
                disabled={refreshMutation.isPending}
                className="h-7 gap-1.5 text-xs"
              >
                <RefreshCw className={cn('h-3 w-3', refreshMutation.isPending && 'animate-spin')} />
                {t('ai.modelsView.refresh')}
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {t('ai.modelsView.description')}
          </p>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 animate-spin text-violet-400" />
            </div>
          ) : (
            <div className="space-y-1">
              {/* Header */}
              <div className="grid grid-cols-[1fr_200px_60px] gap-3 px-3 py-2 text-[10px] uppercase tracking-wide text-muted-foreground border-b border-border/40">
                <span>{t('ai.modelsView.colFeature')}</span>
                <span>{t('ai.modelsView.colModel')}</span>
                <span className="text-center">{t('ai.modelsView.colEnabled')}</span>
              </div>

              {/* Rows */}
              {PURPOSES.map(purpose => {
                const currentModel = modelAssignments[purpose.key] ?? ''
                const isEnabled = enabledFeatures[purpose.key] !== false

                return (
                  <div
                    key={purpose.key}
                    className={cn(
                      'grid grid-cols-[1fr_200px_60px] gap-3 items-center px-3 py-2.5 rounded-lg transition-colors',
                      isEnabled
                        ? 'hover:bg-muted/40'
                        : 'opacity-50 hover:bg-muted/20'
                    )}
                  >
                    {/* Label + description */}
                    <div>
                      <p className="text-sm font-medium">{t(purpose.labelKey)}</p>
                      <p className="text-[11px] text-muted-foreground">{t(purpose.descKey)}</p>
                    </div>

                    {/* Model dropdown with search */}
                    <ModelCombobox
                      models={allModels}
                      value={currentModel || '__default__'}
                      onChange={(val) => handleModelChange(purpose.key, val)}
                    />

                    {/* Toggle */}
                    <div className="flex justify-center">
                      <Switch
                        checked={isEnabled}
                        onCheckedChange={(val) => handleToggle(purpose.key, val)}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Active assignments summary */}
          {Object.keys(modelAssignments).length > 0 && (
            <div className="mt-4 pt-3 border-t border-border/40">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">{t('ai.modelsView.activeOverrides')}</p>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(modelAssignments).map(([key, model]) => {
                  const purpose = PURPOSES.find(p => p.key === key)
                  return (
                    <Badge key={key} variant="outline" className="text-[10px] bg-violet-500/10 text-violet-700 dark:text-violet-300 border-violet-500/20">
                      {purpose ? t(purpose.labelKey) : key}: {model}
                    </Badge>
                  )
                })}
              </div>
            </div>
          )}

          {/* Save button */}
          <div className="mt-4 flex items-center gap-3">
            <Button
              onClick={() => saveMutation.mutate()}
              disabled={!dirty || saveMutation.isPending}
              className={cn(
                'gap-2 px-6 py-2 rounded-xl text-sm font-medium transition-colors',
                dirty
                  ? 'bg-violet-500 hover:bg-violet-600 text-white'
                  : 'bg-muted text-muted-foreground cursor-not-allowed'
              )}
            >
              {saveMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              {t('ai.modelsView.saveAssignments')}
            </Button>
            {saveMutation.isSuccess && (
              <span className="flex items-center gap-1.5 text-xs text-emerald-400">
                <CheckCircle className="w-3.5 h-3.5" />
                {t('ai.modelsView.saved')}
              </span>
            )}
            {saveMutation.isError && (
              <span className="text-xs text-red-400">
                {(saveMutation.error as Error).message}
              </span>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
