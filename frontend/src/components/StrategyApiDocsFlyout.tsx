import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Code2,
  Zap,
  Package,
  AlertTriangle,
  Copy,
  Check,
  Settings2,
  LogOut,
  Play,
  Rocket,
  Shield,
  ListChecks,
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

function FieldTable({ fields }: { fields: Record<string, string | Record<string, unknown>> }) {
  return (
    <div className="space-y-0.5">
      {Object.entries(fields).map(([key, desc]) => (
        <div key={key} className="flex gap-2 text-[11px] py-0.5">
          <code className="text-amber-400 font-mono shrink-0">{key}</code>
          <span className="text-muted-foreground">
            {typeof desc === 'string' ? desc : (desc as Record<string, unknown>)?.description as string || JSON.stringify(desc)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ==================== PHASE CARD ====================

function PhaseCard({ phase }: { phase: Record<string, string> }) {
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
        <span className="text-amber-400/80">Caller:</span> {phase.caller}
      </div>
      <div className="text-[10px] text-muted-foreground">
        <span className="text-emerald-400/80">Default:</span> {phase.default_behavior}
      </div>
    </div>
  )
}

// ==================== UNIFIED DOCS ====================

function UnifiedDocs({ docs }: { docs: Record<string, unknown> }) {
  const overview = docs.overview as Record<string, unknown> | undefined
  const baseStrategy = docs.base_strategy as Record<string, unknown> | undefined
  const detectPhase = docs.detect_phase as Record<string, unknown> | undefined
  const evaluatePhase = docs.evaluate_phase as Record<string, unknown> | undefined
  const exitPhase = docs.exit_phase as Record<string, unknown> | undefined
  const configSchema = docs.config_schema as Record<string, unknown> | undefined
  const imports = docs.imports as Record<string, unknown> | undefined
  const examples = docs.examples as Record<string, Record<string, string>> | undefined
  const backtesting = docs.backtesting as Record<string, unknown> | undefined
  const validation = docs.validation as Record<string, unknown> | undefined
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
        <Section title="Quick Start" icon={Rocket} iconColor="text-emerald-400" defaultOpen>
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
        <Section title="Three-Phase Lifecycle" icon={Zap} iconColor="text-amber-400" defaultOpen>
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">
              {(overview.three_phase_lifecycle as Record<string, unknown>).description as string}
            </p>
            {((overview.three_phase_lifecycle as Record<string, unknown>).phases as Record<string, string>[])?.map((phase) => (
              <PhaseCard key={phase.phase} phase={phase} />
            ))}
          </div>
        </Section>
      )}

      {/* BaseStrategy Interface */}
      {baseStrategy && (
        <Section title="BaseStrategy Interface" icon={Code2} iconColor="text-cyan-400">
          <div className="space-y-3 pt-2">
            <code className="text-[11px] text-cyan-400 font-mono block">
              {baseStrategy.import as string}
            </code>

            {baseStrategy.class_attributes && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Class Attributes</div>
                <FieldTable fields={baseStrategy.class_attributes as Record<string, Record<string, unknown>>} />
              </div>
            )}

            {baseStrategy.built_in_properties && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Built-in Properties</div>
                <FieldTable fields={baseStrategy.built_in_properties as Record<string, string>} />
              </div>
            )}

            {baseStrategy.helper_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Helper Methods</div>
                {Object.entries(baseStrategy.helper_methods as Record<string, Record<string, unknown>>).map(([name, info]) => (
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
        <Section title="DETECT Phase" icon={Zap} iconColor="text-emerald-400">
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
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Parameters</div>
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
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Returns</div>
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
        <Section title="EVALUATE Phase" icon={Play} iconColor="text-blue-400">
          <div className="space-y-2 pt-2">
            <code className="text-[10px] text-cyan-400 font-mono block">{evaluatePhase.method as string}</code>
            <p className="text-[11px] text-muted-foreground">{evaluatePhase.when_called as string}</p>

            {evaluatePhase.signal_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  Signal Object
                </div>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(evaluatePhase.signal_object as Record<string, unknown>).description as string}
                </p>
                <FieldTable fields={(evaluatePhase.signal_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {evaluatePhase.context_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  Context Object
                </div>
                <p className="text-[10px] text-muted-foreground mb-1">
                  {(evaluatePhase.context_object as Record<string, unknown>).description as string}
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
                  {(evaluatePhase.return_value as Record<string, string>).constructor}
                </code>
                {(evaluatePhase.return_value as Record<string, Record<string, string>>).decision_values && (
                  <FieldTable fields={(evaluatePhase.return_value as Record<string, Record<string, string>>).decision_values} />
                )}
                {(evaluatePhase.return_value as Record<string, Record<string, unknown>>).checks_field && (
                  <div className="mt-1">
                    <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">DecisionCheck</div>
                    <code className="text-[9px] text-muted-foreground/70 font-mono block">
                      {((evaluatePhase.return_value as Record<string, Record<string, string>>).checks_field).constructor}
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

      {/* EXIT Phase */}
      {exitPhase && (
        <Section title="EXIT Phase" icon={LogOut} iconColor="text-red-400">
          <div className="space-y-2 pt-2">
            <code className="text-[10px] text-cyan-400 font-mono block">{exitPhase.method as string}</code>
            <p className="text-[11px] text-muted-foreground">{exitPhase.when_called as string}</p>

            {exitPhase.position_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  Position Object
                </div>
                <FieldTable fields={(exitPhase.position_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {exitPhase.market_state_object && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  Market State
                </div>
                <FieldTable fields={(exitPhase.market_state_object as Record<string, Record<string, string>>).fields} />
              </div>
            )}

            {exitPhase.return_value && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                  ExitDecision
                </div>
                <code className="text-[9px] text-muted-foreground/70 font-mono block mb-1">
                  {(exitPhase.return_value as Record<string, string>).constructor}
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

      {/* Config Schema */}
      {configSchema && (
        <Section title="Config Schema" icon={Settings2} iconColor="text-blue-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{configSchema.description as string}</p>
            {configSchema.format && (
              <CodeBlock code={JSON.stringify(configSchema.format, null, 2)} />
            )}
            {configSchema.field_types && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Field Types</div>
                <FieldTable fields={configSchema.field_types as Record<string, string>} />
              </div>
            )}
            <p className="text-[10px] text-muted-foreground">{configSchema.how_it_works as string}</p>
          </div>
        </Section>
      )}

      {/* Imports */}
      {imports && (
        <Section title="Available Imports" icon={Package} iconColor="text-emerald-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{imports.description as string}</p>

            {imports.app_modules && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">App Modules</div>
                <FieldTable fields={imports.app_modules as Record<string, string>} />
              </div>
            )}

            {imports.standard_library && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Standard Library</div>
                <div className="flex flex-wrap gap-1">
                  {(imports.standard_library as string[]).map((mod) => (
                    <code key={mod} className="text-[9px] text-cyan-400/80 font-mono bg-cyan-400/5 px-1.5 py-0.5 rounded">{mod}</code>
                  ))}
                </div>
              </div>
            )}

            {imports.third_party && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Third Party</div>
                <FieldTable fields={imports.third_party as Record<string, string>} />
              </div>
            )}

            {imports.blocked && (
              <div>
                <div className="text-[10px] font-medium text-red-400/80 uppercase tracking-wider mb-1">Blocked (Security)</div>
                <FieldTable fields={(imports.blocked as Record<string, string>)} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Examples */}
      {examples && (
        <Section title="Complete Examples" icon={BookOpen} iconColor="text-orange-400">
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
        <Section title="Backtesting" icon={Play} iconColor="text-purple-400">
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
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Request Body</div>
                <FieldTable fields={backtesting.request_body as Record<string, string>} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Validation */}
      {validation && (
        <Section title="Validation" icon={Shield} iconColor="text-yellow-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{validation.description as string}</p>

            {validation.checks_performed && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Checks Performed</div>
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
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Response</div>
                <FieldTable fields={validation.response as Record<string, string | Record<string, unknown>>} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* API Endpoints */}
      {endpoints && (
        <Section title="API Endpoints" icon={ListChecks} iconColor="text-teal-400">
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
                Strategy Developer Reference
                <Badge variant="outline" className="text-[9px] h-4 font-normal">v2.0</Badge>
              </SheetTitle>
              <SheetDescription>
                Three-phase lifecycle: DETECT → EVALUATE → EXIT. Covers all strategy types.
              </SheetDescription>
            </SheetHeader>
          </div>

          <ScrollArea className="flex-1 min-h-0 px-4 py-3">
            {docsQuery.isLoading && (
              <div className="text-xs text-muted-foreground text-center py-8">Loading API reference...</div>
            )}
            {docsQuery.error && (
              <div className="text-xs text-red-400 text-center py-8">Failed to load docs</div>
            )}
            {docsQuery.data && <UnifiedDocs docs={docsQuery.data} />}
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  )
}
