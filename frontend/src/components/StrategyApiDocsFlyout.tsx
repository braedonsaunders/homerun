import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Code2,
  Zap,
  Package,
  Copy,
  Check,
  Settings2,
  LogOut,
  Play,
  Rocket,
  Shield,
  ListChecks,
  Radio,
  Filter,
  ShieldAlert,
  Sliders,
  LayoutGrid,
  Database,
  Users,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { cn } from '../lib/utils'
import { getTraderStrategyDocs } from '../services/api'

// ==================== CODE BLOCK ====================

function CodeBlock({ code, className }: { code: string; className?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className={cn('relative group', className)}>
      <pre className="bg-[#1e1e2e] border border-border/30 rounded-md p-3 text-[11px] leading-relaxed font-mono text-gray-300 overflow-x-auto whitespace-pre">
        {code}
      </pre>
      <button
        className="absolute top-1.5 right-1.5 p-1 rounded bg-white/5 hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={() => {
          navigator.clipboard.writeText(code)
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        }}
      >
        {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-gray-400" />}
      </button>
    </div>
  )
}

// ==================== COLLAPSIBLE SECTION ====================

function Section({
  title,
  icon: Icon,
  iconColor,
  defaultOpen = false,
  children,
}: {
  title: string
  icon: React.ComponentType<{ className?: string }>
  iconColor?: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-border/30 rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium hover:bg-card/50 transition-colors"
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown className="w-3 h-3 text-muted-foreground" /> : <ChevronRight className="w-3 h-3 text-muted-foreground" />}
        <Icon className={cn('w-3.5 h-3.5', iconColor || 'text-muted-foreground')} />
        {title}
      </button>
      {open && <div className="px-3 pb-3 space-y-2 border-t border-border/20">{children}</div>}
    </div>
  )
}

// ==================== FIELD TABLE ====================

function FieldTable({ fields }: { fields: Record<string, string | Record<string, any>> }) {
  return (
    <div className="space-y-0.5">
      {Object.entries(fields).map(([key, desc]) => (
        <div key={key} className="flex gap-2 text-[11px] py-0.5">
          <code className="text-amber-400 font-mono shrink-0">{key}</code>
          <span className="text-muted-foreground">
            {typeof desc === 'string' ? desc : (desc as Record<string, any>)?.description as string || JSON.stringify(desc)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ==================== PHASE CARD ====================

function PhaseCard({ phase }: { phase: Record<string, string> }) {
  const { t } = useTranslation()
  return (
    <div className="border border-border/20 rounded-md p-2 space-y-1">
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="text-[9px] h-4 font-semibold">{phase.phase}</Badge>
        <span className="text-[10px] text-muted-foreground">{phase.purpose}</span>
      </div>
      <code className="text-[10px] text-cyan-400 font-mono block">{phase.method}</code>
      {phase.async_method && (
        <code className="text-[10px] text-cyan-400/70 font-mono block">{phase.async_method}</code>
      )}
      <div className="text-[10px] text-muted-foreground">
        <span className="text-amber-400/80">{t('strategyApiDocsFlyout.caller')}:</span> {phase.caller}
      </div>
      <div className="text-[10px] text-muted-foreground">
        <span className="text-emerald-400/80">{t('strategyApiDocsFlyout.default')}:</span> {phase.default_behavior}
      </div>
    </div>
  )
}

// ==================== UNIFIED DOCS ====================

function UnifiedDocs({ docs }: { docs: Record<string, any> }) {
  const { t } = useTranslation()
  const overview = docs.overview as Record<string, any> | undefined
  const baseStrategy = docs.base_strategy as Record<string, any> | undefined
  const detectPhase = docs.detect_phase as Record<string, any> | undefined
  const evaluatePhase = docs.evaluate_phase as Record<string, any> | undefined
  const exitPhase = docs.exit_phase as Record<string, any> | undefined
  const advancedExits = docs.advanced_exits as Record<string, any> | undefined
  const composableEvaluate = docs.composable_evaluate as Record<string, any> | undefined
  const eventSubscriptions = docs.event_subscriptions as Record<string, any> | undefined
  const qualityFilter = docs.quality_filter as Record<string, any> | undefined
  const platformHooks = docs.platform_hooks as Record<string, any> | undefined
  const configSchema = docs.config_schema as Record<string, any> | undefined
  const strategySdk = docs.strategy_sdk as Record<string, any> | undefined
  const imports = docs.imports as Record<string, any> | undefined
  const dataSourceSdk = docs.data_source_sdk as Record<string, any> | undefined
  const traderDataSdk = docs.trader_data_sdk as Record<string, any> | undefined
  const examples = docs.examples as Record<string, Record<string, string>> | undefined
  const backtesting = docs.backtesting as Record<string, any> | undefined
  const validation = docs.validation as Record<string, any> | undefined
  const endpoints = docs.endpoints as Record<string, Record<string, string>> | undefined
  const quickStart = docs.quick_start as string[] | undefined

  return (
    <div className="space-y-2">
      {/* Overview */}
      {overview && (
        <div className="text-xs text-muted-foreground px-1 pb-1">
          {overview.summary as string}
        </div>
      )}

      {/* Quick Start */}
      {quickStart && (
        <Section title={t('strategyApiDocsFlyout.sections.quickStart')} icon={Rocket} iconColor="text-emerald-400" defaultOpen>
          <ol className="space-y-1 pt-2">
            {quickStart.map((step, i) => (
              <li key={i} className="text-[11px] text-muted-foreground flex items-start gap-2">
                <span className="text-emerald-400 font-mono shrink-0 w-4 text-right">{i + 1}.</span>
                <span>{step.replace(/^\d+\.\s*/, '')}</span>
              </li>
            ))}
          </ol>
        </Section>
      )}

      {/* Three Phase Lifecycle */}
      {overview?.three_phase_lifecycle && (
        <Section title={t('strategyApiDocsFlyout.sections.threePhaseLifecycle')} icon={Zap} iconColor="text-amber-400" defaultOpen>
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">
              {(overview.three_phase_lifecycle as Record<string, any>).description as string}
            </p>
            {((overview.three_phase_lifecycle as Record<string, any>).phases as Record<string, string>[])?.map((phase) => (
              <PhaseCard key={phase.phase} phase={phase} />
            ))}
          </div>
        </Section>
      )}

      {/* BaseStrategy Interface */}
      {baseStrategy && (
        <Section title={t('strategyApiDocsFlyout.sections.baseStrategyInterface')} icon={Code2} iconColor="text-cyan-400">
          <div className="space-y-3 pt-2">
            <code className="text-[11px] text-cyan-400 font-mono block">
              {baseStrategy.import as string}
            </code>

            {baseStrategy.class_attributes && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.classAttributes')}</div>
                <FieldTable fields={baseStrategy.class_attributes as Record<string, Record<string, any>>} />
              </div>
            )}

            {baseStrategy.built_in_properties && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.builtInProperties')}</div>
                <FieldTable fields={baseStrategy.built_in_properties as Record<string, string>} />
              </div>
            )}

            {baseStrategy.helper_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.helperMethods')}</div>
                {Object.entries(baseStrategy.helper_methods as Record<string, Record<string, any>>).map(([name, info]) => (
                  <div key={name} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-1">
                    <code className="text-[10px] text-emerald-400 font-mono">{name}</code>
                    <p className="text-[10px] text-muted-foreground">{info.description as string}</p>
                    {info.signature && (
                      <code className="text-[9px] text-muted-foreground/70 font-mono block break-all">{info.signature as string}</code>
                    )}
                    {info.config_params && (
                      <FieldTable fields={info.config_params as Record<string, string>} />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </Section>
      )}

      {/* DETECT Phase */}
      {detectPhase && (
        <Section title={t('strategyApiDocsFlyout.sections.detectPhase')} icon={Zap} iconColor="text-emerald-400">
          <div className="space-y-2 pt-2">
            {detectPhase.methods && (
              <div className="space-y-1">
                {Object.entries(detectPhase.methods as Record<string, Record<string, string>>).map(([key, info]) => (
                  <div key={key} className="border border-border/20 rounded-md p-2 space-y-0.5">
                    <code className="text-[10px] text-cyan-400 font-mono block">{info.signature}</code>
                    <p className="text-[10px] text-muted-foreground">{info.when_to_use}</p>
                    {info.note && <p className="text-[10px] text-amber-400/80 italic">{info.note}</p>}
                  </div>
                ))}
              </div>
            )}

            {detectPhase.parameters && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.parameters')}</div>
                {Object.entries(detectPhase.parameters as Record<string, Record<string, string>>).map(([name, param]) => (
                  <div key={name} className="space-y-0.5 mb-1">
                    <div className="flex items-center gap-2">
                      <code className="text-[11px] font-mono text-cyan-400">{name}</code>
                      <Badge variant="outline" className="text-[9px] h-4">{param.type}</Badge>
                    </div>
                    <p className="text-[10px] text-muted-foreground">{param.description}</p>
                    {param.useful_fields && (
                      <code className="text-[9px] text-muted-foreground/70 font-mono block">{param.useful_fields}</code>
                    )}
                    {param.structure && (
                      <code className="text-[9px] text-muted-foreground/70 font-mono block">{param.structure}</code>
                    )}
                  </div>
                ))}
              </div>
            )}

            {detectPhase.return_value && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.returns')}</div>
                <p className="text-[10px] text-muted-foreground">
                  {(detectPhase.return_value as Record<string, string>).tip}
                </p>
                <p className="text-[10px] text-amber-400/80 mt-1">
                  {(detectPhase.return_value as Record<string, string>).strategy_context}
                </p>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* EVALUATE Phase */}
      {evaluatePhase && (
        <Section title={t('strategyApiDocsFlyout.sections.evaluatePhase')} icon={Play} iconColor="text-blue-400">
          <div className="space-y-2 pt-2">
            <code className="text-[10px] text-cyan-400 font-mono block">{evaluatePhase.method as string}</code>
            <p className="text-[11px] text-muted-foreground">{evaluatePhase.when_called as string}</p>

            {evaluatePhase.signal_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.signalObject')}
                </div>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(evaluatePhase.signal_object as Record<string, any>).description as string}
                </p>
                <FieldTable fields={(evaluatePhase.signal_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {evaluatePhase.context_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.contextObject')}
                </div>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(evaluatePhase.context_object as Record<string, any>).description as string}
                </p>
                <FieldTable fields={(evaluatePhase.context_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {evaluatePhase.return_value && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  StrategyDecision
                </div>
                <code className="text-[9px] text-muted-foreground/70 font-mono block mb-1">
                  {String((evaluatePhase.return_value as any).constructor ?? '')}
                </code>
                {(evaluatePhase.return_value as Record<string, Record<string, string>>).decision_values && (
                  <FieldTable fields={(evaluatePhase.return_value as Record<string, Record<string, string>>).decision_values} />
                )}
                {(evaluatePhase.return_value as Record<string, Record<string, any>>).checks_field && (
                  <div className="mt-1">
                    <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.decisionCheck')}</div>
                    <code className="text-[9px] text-muted-foreground/70 font-mono block">
                      {String(((evaluatePhase.return_value as any).checks_field?.constructor) ?? '')}
                    </code>
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      {((evaluatePhase.return_value as Record<string, Record<string, string>>).checks_field).purpose}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Advanced Exits (laddered/chunked) */}
      {advancedExits && (
        <Section title={t('strategyApiDocsFlyout.sections.advancedExitExecution')} icon={LayoutGrid} iconColor="text-fuchsia-400">
          <div className="space-y-3 pt-2">
            {advancedExits.summary && (
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {advancedExits.summary as string}
              </p>
            )}

            {advancedExits.imports && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.imports')}
                </div>
                <CodeBlock code={advancedExits.imports as string} />
              </div>
            )}

            {advancedExits.how_to_attach && (
              <div className="space-y-2">
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                  {t('strategyApiDocsFlyout.labels.howToAttachPolicy')}
                </div>
                <p className="text-[10px] text-muted-foreground">
                  {(advancedExits.how_to_attach as Record<string, string>).description}
                </p>
                <div>
                  <div className="text-[10px] text-emerald-400/80 mb-0.5">
                    {t('strategyApiDocsFlyout.labels.classAttributePerTrigger')}
                  </div>
                  <CodeBlock
                    code={(advancedExits.how_to_attach as Record<string, string>).class_attribute_example}
                  />
                </div>
                <div>
                  <div className="text-[10px] text-emerald-400/80 mb-0.5">
                    {t('strategyApiDocsFlyout.labels.perDecisionOverride')}
                  </div>
                  <CodeBlock
                    code={(advancedExits.how_to_attach as Record<string, string>).per_decision_override_example}
                  />
                </div>
                <p className="text-[10px] text-amber-400/80">
                  {(advancedExits.how_to_attach as Record<string, string>).trigger_keys}
                </p>
              </div>
            )}

            {advancedExits.exit_policy_fields && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.exitPolicyFields')}
                </div>
                <FieldTable
                  fields={advancedExits.exit_policy_fields as Record<string, string>}
                />
              </div>
            )}

            {advancedExits.sdk_helpers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.strategySdkHelpers')}
                </div>
                <FieldTable
                  fields={advancedExits.sdk_helpers as Record<string, string>}
                />
              </div>
            )}

            {advancedExits.child_order_lifecycle && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.childOrderLifecycle')}
                </div>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(advancedExits.child_order_lifecycle as Record<string, string>).summary}
                </p>
                <FieldTable
                  fields={
                    (advancedExits.child_order_lifecycle as Record<string, Record<string, string>>).states
                  }
                />
              </div>
            )}

            {advancedExits.polymarket_notes && (
              <div className="border border-amber-400/20 bg-amber-400/5 rounded-md p-2">
                <div className="text-[10px] font-medium text-amber-400/90 uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.polymarketConstraints')}
                </div>
                <p className="text-[10px] text-muted-foreground">
                  {advancedExits.polymarket_notes as string}
                </p>
              </div>
            )}

            {advancedExits.tip && (
              <div className="border border-emerald-400/20 bg-emerald-400/5 rounded-md p-2">
                <div className="text-[10px] font-medium text-emerald-400/90 uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.suggestedStartingPoint')}
                </div>
                <p className="text-[10px] text-muted-foreground">
                  {advancedExits.tip as string}
                </p>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* EXIT Phase */}
      {exitPhase && (
        <Section title={t('strategyApiDocsFlyout.sections.exitPhase')} icon={LogOut} iconColor="text-red-400">
          <div className="space-y-2 pt-2">
            <code className="text-[10px] text-cyan-400 font-mono block">{exitPhase.method as string}</code>
            <p className="text-[11px] text-muted-foreground">{exitPhase.when_called as string}</p>

            {exitPhase.position_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.positionObject')}
                </div>
                <FieldTable fields={(exitPhase.position_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {exitPhase.market_state_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.marketState')}
                </div>
                <FieldTable fields={(exitPhase.market_state_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {exitPhase.return_value && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {t('strategyApiDocsFlyout.labels.exitDecision')}
                </div>
                <code className="text-[9px] text-muted-foreground/70 font-mono block mb-1">
                  {String((exitPhase.return_value as any).constructor ?? '')}
                </code>
                {(exitPhase.return_value as Record<string, Record<string, string>>).action_values && (
                  <FieldTable fields={(exitPhase.return_value as Record<string, Record<string, string>>).action_values} />
                )}
                <p className="text-[10px] text-amber-400/80 mt-1">
                  {(exitPhase.return_value as Record<string, string>).tip}
                </p>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Composable Evaluate Pipeline */}
      {composableEvaluate && (
        <Section title={t('strategyApiDocsFlyout.sections.composableEvaluatePipeline')} icon={Sliders} iconColor="text-violet-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{composableEvaluate.description as string}</p>

            {composableEvaluate.scoring_weights && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.scoringWeights')}</div>
                <code className="text-[9px] text-cyan-400/70 font-mono block mb-1">
                  {(composableEvaluate.scoring_weights as Record<string, string>).import}
                </code>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(composableEvaluate.scoring_weights as Record<string, string>).formula}
                </p>
                {(composableEvaluate.scoring_weights as Record<string, Record<string, string>>).fields && (
                  <FieldTable fields={(composableEvaluate.scoring_weights as Record<string, Record<string, string>>).fields} />
                )}
              </div>
            )}

            {composableEvaluate.sizing_config && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.sizingConfig')}</div>
                <code className="text-[9px] text-cyan-400/70 font-mono block mb-1">
                  {(composableEvaluate.sizing_config as Record<string, string>).import}
                </code>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(composableEvaluate.sizing_config as Record<string, string>).formula}
                </p>
                {(composableEvaluate.sizing_config as Record<string, Record<string, string>>).fields && (
                  <FieldTable fields={(composableEvaluate.sizing_config as Record<string, Record<string, string>>).fields} />
                )}
              </div>
            )}

            {composableEvaluate.custom_checks_override && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.customChecksOverride')}</div>
                <code className="text-[10px] text-cyan-400 font-mono block mb-1">
                  {(composableEvaluate.custom_checks_override as Record<string, string>).signature}
                </code>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(composableEvaluate.custom_checks_override as Record<string, string>).description}
                </p>
                {(composableEvaluate.custom_checks_override as Record<string, string>).example && (
                  <CodeBlock code={(composableEvaluate.custom_checks_override as Record<string, string>).example} />
                )}
              </div>
            )}

            {composableEvaluate.how_to_opt_in && (
              <p className="text-[10px] text-amber-400/80 mt-1">{composableEvaluate.how_to_opt_in as string}</p>
            )}
          </div>
        </Section>
      )}

      {/* Event Subscriptions */}
      {eventSubscriptions && (
        <Section title={t('strategyApiDocsFlyout.sections.eventSubscriptions')} icon={Radio} iconColor="text-pink-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{eventSubscriptions.description as string}</p>
            <p className="text-[10px] text-amber-400/80">{eventSubscriptions.how_to_subscribe as string}</p>

            {eventSubscriptions.data_event_types && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.eventTypes')}</div>
                {Object.entries(eventSubscriptions.data_event_types as Record<string, Record<string, string>>).map(([key, info]) => (
                  <div key={key} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-0.5">
                    <code className="text-[10px] text-cyan-400 font-mono">{key}</code>
                    <p className="text-[10px] text-muted-foreground">{info.description}</p>
                    <p className="text-[9px] text-muted-foreground/60 font-mono">{info.payload_fields}</p>
                  </div>
                ))}
              </div>
            )}

            {eventSubscriptions.data_event_structure && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.dataEventStructure')}</div>
                <code className="text-[9px] text-cyan-400/70 font-mono block mb-1">
                  {(eventSubscriptions.data_event_structure as Record<string, string>).import}
                </code>
                {(eventSubscriptions.data_event_structure as Record<string, Record<string, string>>).fields && (
                  <FieldTable fields={(eventSubscriptions.data_event_structure as Record<string, Record<string, string>>).fields} />
                )}
              </div>
            )}

            {eventSubscriptions.on_event_method && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.onEventMethod')}</div>
                <code className="text-[10px] text-cyan-400 font-mono block">
                  {(eventSubscriptions.on_event_method as Record<string, string>).signature}
                </code>
                <p className="text-[10px] text-muted-foreground mt-0.5">
                  {(eventSubscriptions.on_event_method as Record<string, string>).description}
                </p>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Quality Filters */}
      {qualityFilter && (
        <Section title={t('strategyApiDocsFlyout.sections.qualityFilterPipeline')} icon={Filter} iconColor="text-orange-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{qualityFilter.description as string}</p>
            <code className="text-[9px] text-cyan-400/70 font-mono block">{qualityFilter.import as string}</code>

            {qualityFilter.quality_report && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.qualityReport')}</div>
                <FieldTable fields={(qualityFilter.quality_report as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {qualityFilter.filter_result && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.filterResult')}</div>
                <FieldTable fields={(qualityFilter.filter_result as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {qualityFilter.filters_applied && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.filtersApplied')}</div>
                <div className="space-y-0.5">
                  {(qualityFilter.filters_applied as string[]).map((filter, i) => (
                    <div key={i} className="flex items-start gap-1.5 text-[10px]">
                      <span className="text-emerald-400 shrink-0 font-mono">{i + 1}.</span>
                      <span className="text-muted-foreground">{filter}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Platform Hooks */}
      {platformHooks && (
        <Section title={t('strategyApiDocsFlyout.sections.platformHooks')} icon={ShieldAlert} iconColor="text-red-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{platformHooks.description as string}</p>

            {platformHooks.on_blocked && (
              <div className="border border-border/20 rounded-md p-2 space-y-1">
                <code className="text-[10px] text-cyan-400 font-mono block">
                  {(platformHooks.on_blocked as Record<string, string>).signature}
                </code>
                <p className="text-[10px] text-muted-foreground">
                  {(platformHooks.on_blocked as Record<string, string>).description}
                </p>
                {(platformHooks.on_blocked as Record<string, string[]>).called_when && (
                  <div className="mt-1">
                    <div className="text-[9px] font-medium text-muted-foreground/80 mb-0.5">{t('strategyApiDocsFlyout.labels.calledWhen')}</div>
                    {(platformHooks.on_blocked as Record<string, string[]>).called_when.map((reason, i) => (
                      <div key={i} className="flex items-start gap-1 text-[9px] text-muted-foreground/70">
                        <span className="text-red-400/60 shrink-0">-</span>
                        <span>{reason}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {platformHooks.on_size_capped && (
              <div className="border border-border/20 rounded-md p-2 space-y-1">
                <code className="text-[10px] text-cyan-400 font-mono block">
                  {(platformHooks.on_size_capped as Record<string, string>).signature}
                </code>
                <p className="text-[10px] text-muted-foreground">
                  {(platformHooks.on_size_capped as Record<string, string>).description}
                </p>
                {(platformHooks.on_size_capped as Record<string, string[]>).called_when && (
                  <div className="mt-1">
                    <div className="text-[9px] font-medium text-muted-foreground/80 mb-0.5">{t('strategyApiDocsFlyout.labels.calledWhen')}</div>
                    {(platformHooks.on_size_capped as Record<string, string[]>).called_when.map((reason, i) => (
                      <div key={i} className="flex items-start gap-1 text-[9px] text-muted-foreground/70">
                        <span className="text-amber-400/60 shrink-0">-</span>
                        <span>{reason}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Config Schema */}
      {configSchema && (
        <Section title={t('strategyApiDocsFlyout.sections.configSchema')} icon={Settings2} iconColor="text-blue-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{configSchema.description as string}</p>
            {configSchema.format && (
              <CodeBlock code={JSON.stringify(configSchema.format, null, 2)} />
            )}
            {configSchema.field_types && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.fieldTypes')}</div>
                <FieldTable fields={configSchema.field_types as Record<string, string>} />
              </div>
            )}
            <p className="text-[10px] text-muted-foreground">{configSchema.how_it_works as string}</p>
          </div>
        </Section>
      )}

      {/* StrategySDK */}
      {strategySdk && (
        <Section title={t('strategyApiDocsFlyout.sections.strategySdkReference')} icon={Code2} iconColor="text-indigo-400">
          <div className="space-y-3 pt-2">
            {strategySdk.summary && (
              <p className="text-[11px] text-muted-foreground">{strategySdk.summary as string}</p>
            )}

            {Array.isArray(strategySdk.business_logic_contract) && strategySdk.business_logic_contract.length > 0 && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.businessLogicContract')}</div>
                <ol className="space-y-0.5">
                  {(strategySdk.business_logic_contract as string[]).map((item, i) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <span className="text-amber-400 shrink-0">{i + 1}.</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {strategySdk.signal_routing_controls && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.signalRoutingControls')}</div>
                <FieldTable fields={strategySdk.signal_routing_controls as Record<string, string>} />
              </div>
            )}

            {strategySdk.configuration_helpers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.configurationHelpers')}</div>
                <FieldTable fields={strategySdk.configuration_helpers as Record<string, string>} />
              </div>
            )}

            {strategySdk.validation_helpers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.validationHelpers')}</div>
                <FieldTable fields={strategySdk.validation_helpers as Record<string, string>} />
              </div>
            )}

            {strategySdk.market_and_execution_helpers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.marketExecutionHelpers')}</div>
                <FieldTable fields={strategySdk.market_and_execution_helpers as Record<string, string>} />
              </div>
            )}

            {strategySdk.llm_and_news_helpers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.llmNewsHelpers')}</div>
                <FieldTable fields={strategySdk.llm_and_news_helpers as Record<string, string>} />
              </div>
            )}

            {strategySdk.trader_data_helpers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.traderDataHelpers')}</div>
                <FieldTable fields={strategySdk.trader_data_helpers as Record<string, string>} />
              </div>
            )}

            {strategySdk.crypto_highfreq_scope_defaults && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.cryptoHfScopeDefaults')}</div>
                <CodeBlock code={JSON.stringify(strategySdk.crypto_highfreq_scope_defaults, null, 2)} />
              </div>
            )}

            {strategySdk.crypto_highfreq_scope_schema && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.cryptoHfScopeSchema')}</div>
                <CodeBlock code={JSON.stringify(strategySdk.crypto_highfreq_scope_schema, null, 2)} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Opportunity Tab Routing */}
      <Section title={t('strategyApiDocsFlyout.sections.opportunityTabRouting')} icon={LayoutGrid} iconColor="text-violet-400">
        <div className="space-y-3 pt-2">
          <p className="text-[11px] text-muted-foreground">
            <span dangerouslySetInnerHTML={{ __html: t('strategyApiDocsFlyout.opportunityRouting.intro') }} />
          </p>

          <div>
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.opportunityRouting.settingSourceKey')}</div>
            <CodeBlock code={`from services.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    strategy_type = "my_strategy"
    name = "My Strategy"
    description = "..."
    source_key = "scanner"   # controls which tab this appears under`} />
          </div>

          <div>
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.opportunityRouting.builtInSourceKeys')}</div>
            <div className="space-y-0.5">
              {([
                ['scanner', 'green', t('strategyApiDocsFlyout.opportunityRouting.scannerDesc')],
                ['news', 'amber', t('strategyApiDocsFlyout.opportunityRouting.newsDesc')],
                ['weather', 'cyan', t('strategyApiDocsFlyout.opportunityRouting.weatherDesc')],
                ['crypto', 'orange', t('strategyApiDocsFlyout.opportunityRouting.cryptoDesc')],
                ['traders', 'orange', t('strategyApiDocsFlyout.opportunityRouting.tradersDesc')],
                ['manual', 'violet', t('strategyApiDocsFlyout.opportunityRouting.manualDesc')],
                ['events', 'blue', t('strategyApiDocsFlyout.opportunityRouting.eventsDesc')],
              ] as const).map(([key, color, desc]) => (
                <div key={key} className="flex gap-2 text-[11px] py-0.5">
                  <code className={`font-mono shrink-0 text-${color}-400`}>{key}</code>
                  <span className="text-muted-foreground">{desc}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="border border-violet-500/20 rounded-md p-2 bg-violet-500/5">
            <div className="text-[10px] font-medium text-violet-400 mb-1">{t('strategyApiDocsFlyout.opportunityRouting.unknownSourceKeyTitle')}</div>
            <p className="text-[10px] text-muted-foreground">
              <span dangerouslySetInnerHTML={{ __html: t('strategyApiDocsFlyout.opportunityRouting.unknownSourceKeyDesc') }} />
            </p>
          </div>

          <div>
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.default')}</div>
            <p className="text-[10px] text-muted-foreground">
              <span dangerouslySetInnerHTML={{ __html: t('strategyApiDocsFlyout.opportunityRouting.defaultDesc') }} />
            </p>
          </div>
        </div>
      </Section>

      {/* Imports */}
      {imports && (
        <Section title={t('strategyApiDocsFlyout.sections.availableImports')} icon={Package} iconColor="text-emerald-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{imports.description as string}</p>

            {imports.app_modules && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.appModules')}</div>
                <FieldTable fields={imports.app_modules as Record<string, string>} />
              </div>
            )}

            {imports.standard_library && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.standardLibrary')}</div>
                <div className="flex flex-wrap gap-1">
                  {(imports.standard_library as string[]).map((mod) => (
                    <code key={mod} className="text-[9px] text-cyan-400/80 font-mono bg-cyan-400/5 px-1.5 py-0.5 rounded">{mod}</code>
                  ))}
                </div>
              </div>
            )}

            {imports.third_party && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.thirdParty')}</div>
                <FieldTable fields={imports.third_party as Record<string, string>} />
              </div>
            )}

            {imports.blocked && (
              <div>
                <div className="text-[10px] font-medium text-red-400/80 uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.blockedSecurity')}</div>
                <FieldTable fields={(imports.blocked as Record<string, string>)} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Data Source SDK */}
      {dataSourceSdk && (
        <Section title={t('strategyApiDocsFlyout.sections.dataSourceSdk')} icon={Database} iconColor="text-cyan-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{dataSourceSdk.description as string}</p>

            {dataSourceSdk.imports && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.imports')}</div>
                <FieldTable fields={dataSourceSdk.imports as Record<string, string>} />
              </div>
            )}

            {dataSourceSdk.when_to_use && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.whenToUse')}</div>
                <FieldTable fields={dataSourceSdk.when_to_use as Record<string, string>} />
              </div>
            )}

            {dataSourceSdk.read_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.readMethods')}</div>
                {Object.entries(dataSourceSdk.read_methods as Record<string, Record<string, string>>).map(([method, info]) => (
                  <div key={method} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-1">
                    <code className="text-[10px] text-emerald-400 font-mono">{method}</code>
                    <code className="text-[9px] text-cyan-400/70 font-mono block break-all">{info.signature}</code>
                    <p className="text-[10px] text-muted-foreground">{info.description}</p>
                  </div>
                ))}
              </div>
            )}

            {dataSourceSdk.management_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.managementMethods')}</div>
                {Object.entries(dataSourceSdk.management_methods as Record<string, Record<string, string>>).map(([method, info]) => (
                  <div key={method} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-1">
                    <code className="text-[10px] text-emerald-400 font-mono">{method}</code>
                    <code className="text-[9px] text-cyan-400/70 font-mono block break-all">{info.signature}</code>
                    <p className="text-[10px] text-muted-foreground">{info.description}</p>
                  </div>
                ))}
              </div>
            )}

            {dataSourceSdk.strategy_sdk_wrappers && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.strategySdkWrappers')}</div>
                <FieldTable fields={dataSourceSdk.strategy_sdk_wrappers as Record<string, string>} />
              </div>
            )}

            {dataSourceSdk.examples && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.examples')}</div>
                {Object.entries(dataSourceSdk.examples as Record<string, string>).map(([key, sourceCode]) => (
                  <div key={key} className="mb-1.5">
                    <div className="text-[10px] font-medium text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</div>
                    <CodeBlock code={sourceCode} />
                  </div>
                ))}
              </div>
            )}

            {Array.isArray(dataSourceSdk.guidance) && dataSourceSdk.guidance.length > 0 && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.guidance')}</div>
                <ol className="space-y-0.5">
                  {(dataSourceSdk.guidance as string[]).map((item, i) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <span className="text-amber-400 shrink-0">{i + 1}.</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Trader Data SDK */}
      {traderDataSdk && (
        <Section title={t('strategyApiDocsFlyout.sections.traderDataSdk')} icon={Users} iconColor="text-orange-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{traderDataSdk.description as string}</p>

            {traderDataSdk.imports && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.imports')}</div>
                <FieldTable fields={traderDataSdk.imports as Record<string, string>} />
              </div>
            )}

            {traderDataSdk.datasets && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.datasets')}</div>
                <FieldTable fields={traderDataSdk.datasets as Record<string, string>} />
              </div>
            )}

            {traderDataSdk.strategy_sdk_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.strategySdkMethods')}</div>
                <FieldTable fields={traderDataSdk.strategy_sdk_methods as Record<string, string>} />
              </div>
            )}

            {traderDataSdk.examples && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.examples')}</div>
                {Object.entries(traderDataSdk.examples as Record<string, string>).map(([key, sourceCode]) => (
                  <div key={key} className="mb-1.5">
                    <div className="text-[10px] font-medium text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</div>
                    <CodeBlock code={sourceCode} />
                  </div>
                ))}
              </div>
            )}

            {Array.isArray(traderDataSdk.guidance) && traderDataSdk.guidance.length > 0 && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.guidance')}</div>
                <ol className="space-y-0.5">
                  {(traderDataSdk.guidance as string[]).map((item, i) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <span className="text-amber-400 shrink-0">{i + 1}.</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Examples */}
      {examples && (
        <Section title={t('strategyApiDocsFlyout.sections.completeExamples')} icon={BookOpen} iconColor="text-orange-400">
          <div className="space-y-3 pt-2">
            {Object.entries(examples).map(([key, example]) => (
              <div key={key}>
                <div className="text-[10px] font-medium text-muted-foreground mb-1">
                  {example.description}
                </div>
                <CodeBlock code={example.source_code} />
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Backtesting */}
      {backtesting && (
        <Section title={t('strategyApiDocsFlyout.sections.backtesting')} icon={Play} iconColor="text-purple-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{backtesting.description as string}</p>

            {backtesting.modes && Object.entries(backtesting.modes as Record<string, Record<string, string>>).map(([mode, info]) => (
              <div key={mode} className="border border-border/20 rounded-md p-2 space-y-0.5">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-[9px] h-4 font-semibold uppercase">{mode}</Badge>
                  <code className="text-[9px] text-cyan-400/70 font-mono">{info.endpoint}</code>
                </div>
                <p className="text-[10px] text-muted-foreground">{info.what_it_does}</p>
                <p className="text-[10px] text-emerald-400/80">{info.returns}</p>
              </div>
            ))}

            {backtesting.request_body && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.requestBody')}</div>
                <FieldTable fields={backtesting.request_body as Record<string, string>} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Validation */}
      {validation && (
        <Section title={t('strategyApiDocsFlyout.sections.validation')} icon={Shield} iconColor="text-yellow-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{validation.description as string}</p>

            {validation.checks_performed && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.checksPerformed')}</div>
                <ol className="space-y-0.5">
                  {(validation.checks_performed as string[]).map((check, i) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <span className="text-amber-400 shrink-0">{check.match(/^\d+/)?.[0] || i + 1}.</span>
                      <span>{check.replace(/^\d+\.\s*/, '')}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {validation.response && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">{t('strategyApiDocsFlyout.labels.response')}</div>
                <FieldTable fields={validation.response as Record<string, string | Record<string, any>>} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* API Endpoints */}
      {endpoints && (
        <Section title={t('strategyApiDocsFlyout.sections.apiEndpoints')} icon={ListChecks} iconColor="text-teal-400">
          <div className="space-y-3 pt-2">
            {Object.entries(endpoints).map(([group, groupEndpoints]) => (
              <div key={group}>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  {group.replace(/_/g, ' ')}
                </div>
                <div className="space-y-0.5">
                  {Object.entries(groupEndpoints).map(([endpoint, desc]) => (
                    <div key={endpoint} className="flex gap-2 text-[11px] py-0.5">
                      <code className="text-cyan-400 font-mono shrink-0 text-[10px]">{endpoint}</code>
                      <span className="text-muted-foreground">{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}

// ==================== MAIN FLYOUT (Sheet-based) ====================

export default function StrategyApiDocsFlyout({
  open,
  onOpenChange,
  variant: _variant,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  variant?: 'opportunity' | 'trader'
}) {
  const { t } = useTranslation()
  const docsQuery = useQuery({
    queryKey: ['strategy-docs'],
    queryFn: getTraderStrategyDocs,
    staleTime: Infinity,
    enabled: open,
  })

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-xl p-0">
        <div className="h-full min-h-0 flex flex-col">
          <div className="border-b border-border px-4 py-3">
            <SheetHeader className="space-y-1 text-left">
              <SheetTitle className="text-base flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                {t('strategyApiDocsFlyout.title')}
                <Badge variant="outline" className="text-[9px] h-4 font-normal">v2.0</Badge>
              </SheetTitle>
              <SheetDescription>
                {t('strategyApiDocsFlyout.headerDescription')}
              </SheetDescription>
            </SheetHeader>
          </div>

          <ScrollArea className="flex-1 min-h-0 px-4 py-3">
            {docsQuery.isLoading && (
              <div className="text-xs text-muted-foreground text-center py-8">{t('strategyApiDocsFlyout.loadingApiReference')}</div>
            )}
            {docsQuery.error && (
              <div className="text-xs text-red-400 text-center py-8">{t('strategyApiDocsFlyout.failedToLoadDocs')}</div>
            )}
            {docsQuery.data && <UnifiedDocs docs={docsQuery.data} />}
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  )
}
