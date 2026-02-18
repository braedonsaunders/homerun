import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Code2,
  Copy,
  Check,
  Rocket,
  Database,
  ListChecks,
  Package,
  Shield,
  Play,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { cn } from '../lib/utils'
import { getUnifiedDataSourceDocs } from '../services/api'

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

function StageCard({ stage }: { stage: Record<string, string> }) {
  return (
    <div className="border border-border/20 rounded-md p-2 space-y-1">
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="text-[9px] h-4 font-semibold">{stage.stage}</Badge>
        <span className="text-[10px] text-muted-foreground">{stage.purpose}</span>
      </div>
      <code className="text-[10px] text-cyan-400 font-mono block">{stage.method}</code>
      <p className="text-[10px] text-muted-foreground">
        <span className="text-emerald-400/80">Output:</span> {stage.output}
      </p>
    </div>
  )
}

function DataSourceDocs({ docs }: { docs: Record<string, any> }) {
  const overview = docs.overview as Record<string, any> | undefined
  const quickStart = docs.quick_start as string[] | undefined
  const base = docs.base_data_source as Record<string, any> | undefined
  const recordContract = docs.record_contract as Record<string, any> | undefined
  const runner = docs.runner_and_storage as Record<string, any> | undefined
  const sdk = docs.sdk_reference as Record<string, any> | undefined
  const imports = docs.imports as Record<string, any> | undefined
  const validation = docs.validation as Record<string, any> | undefined
  const endpoints = docs.api_endpoints as Record<string, Record<string, string> | string> | undefined
  const examples = docs.examples as Record<string, Record<string, string>> | undefined

  return (
    <div className="space-y-2">
      {overview?.summary && (
        <div className="text-xs text-muted-foreground px-1 pb-1">
          {overview.summary as string}
        </div>
      )}

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

      {overview?.concepts && (
        <Section title="Core Concepts" icon={Database} iconColor="text-blue-400">
          <div className="pt-2">
            <FieldTable fields={overview.concepts as Record<string, string>} />
          </div>
        </Section>
      )}

      {overview?.three_stage_lifecycle && (
        <Section title="Three-Stage Lifecycle" icon={Play} iconColor="text-amber-400" defaultOpen>
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">
              {(overview.three_stage_lifecycle as Record<string, any>).description as string}
            </p>
            {((overview.three_stage_lifecycle as Record<string, any>).stages as Record<string, string>[])?.map((stage) => (
              <StageCard key={stage.stage} stage={stage} />
            ))}
          </div>
        </Section>
      )}

      {base && (
        <Section title="BaseDataSource Interface" icon={Code2} iconColor="text-cyan-400">
          <div className="space-y-3 pt-2">
            {base.import && (
              <code className="text-[11px] text-cyan-400 font-mono block">
                {base.import as string}
              </code>
            )}

            {base.class_attributes && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Class Attributes</div>
                <FieldTable fields={base.class_attributes as Record<string, Record<string, any>>} />
              </div>
            )}

            {base.methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Lifecycle Methods</div>
                {Object.entries(base.methods as Record<string, Record<string, any>>).map(([name, info]) => (
                  <div key={name} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-1">
                    <div className="flex items-center gap-2">
                      <code className="text-[10px] text-emerald-400 font-mono">{name}</code>
                      <Badge variant="outline" className="text-[9px] h-4 font-semibold">
                        {info.required ? 'Required' : 'Optional'}
                      </Badge>
                    </div>
                    {info.signature && (
                      <code className="text-[9px] text-cyan-400/70 font-mono block break-all">
                        {info.signature as string}
                      </code>
                    )}
                    <p className="text-[10px] text-muted-foreground">{info.description as string}</p>
                  </div>
                ))}
              </div>
            )}

            {base.method_selection && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Method Selection</div>
                {(base.method_selection as Record<string, any>).note && (
                  <p className="text-[10px] text-muted-foreground">{(base.method_selection as Record<string, any>).note as string}</p>
                )}
                {(base.method_selection as Record<string, any>).runner_behavior && (
                  <p className="text-[10px] text-amber-400/80 mt-1">{(base.method_selection as Record<string, any>).runner_behavior as string}</p>
                )}
              </div>
            )}
          </div>
        </Section>
      )}

      {recordContract && (
        <Section title="Record Contract" icon={ListChecks} iconColor="text-violet-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{recordContract.description as string}</p>
            {recordContract.fields && (
              <FieldTable fields={recordContract.fields as Record<string, string>} />
            )}
            {recordContract.normalization_rules && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Normalization Rules</div>
                <ol className="space-y-0.5">
                  {(recordContract.normalization_rules as string[]).map((rule, i) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <span className="text-amber-400 shrink-0">{i + 1}.</span>
                      <span>{rule}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </Section>
      )}

      {runner && (
        <Section title="Runner & Storage" icon={Database} iconColor="text-teal-400">
          <div className="space-y-3 pt-2">
            {runner.run_status_values && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Run Status Values</div>
                <FieldTable fields={runner.run_status_values as Record<string, string>} />
              </div>
            )}
            {runner.run_metrics && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Run Metrics</div>
                <FieldTable fields={runner.run_metrics as Record<string, string>} />
              </div>
            )}
            {runner.storage_tables && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Storage Tables</div>
                <FieldTable fields={runner.storage_tables as Record<string, string>} />
              </div>
            )}
          </div>
        </Section>
      )}

      {sdk && (
        <Section title="Data Source SDK" icon={Code2} iconColor="text-orange-400">
          <div className="space-y-3 pt-2">
            {sdk.import && (
              <code className="text-[10px] text-cyan-400 font-mono block">{sdk.import as string}</code>
            )}

            {sdk.read_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Read Methods</div>
                {Object.entries(sdk.read_methods as Record<string, Record<string, string>>).map(([method, info]) => (
                  <div key={method} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-1">
                    <code className="text-[10px] text-emerald-400 font-mono">{method}</code>
                    <code className="text-[9px] text-cyan-400/70 font-mono block break-all">{info.signature}</code>
                    <p className="text-[10px] text-muted-foreground">{info.description}</p>
                  </div>
                ))}
              </div>
            )}

            {sdk.management_methods && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Management Methods</div>
                {Object.entries(sdk.management_methods as Record<string, Record<string, string>>).map(([method, info]) => (
                  <div key={method} className="border border-border/20 rounded-md p-2 mb-1.5 space-y-1">
                    <code className="text-[10px] text-emerald-400 font-mono">{method}</code>
                    <code className="text-[9px] text-cyan-400/70 font-mono block break-all">{info.signature}</code>
                    <p className="text-[10px] text-muted-foreground">{info.description}</p>
                  </div>
                ))}
              </div>
            )}

            {sdk.strategy_sdk_bridge && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">StrategySDK Bridge</div>
                {(sdk.strategy_sdk_bridge as Record<string, any>).description && (
                  <p className="text-[10px] text-muted-foreground mb-1">
                    {(sdk.strategy_sdk_bridge as Record<string, any>).description as string}
                  </p>
                )}
                {(sdk.strategy_sdk_bridge as Record<string, any>).methods && (
                  <FieldTable fields={(sdk.strategy_sdk_bridge as Record<string, any>).methods as Record<string, string>} />
                )}
              </div>
            )}
          </div>
        </Section>
      )}

      {imports && (
        <Section title="Available Imports" icon={Package} iconColor="text-emerald-400">
          <div className="space-y-3 pt-2">
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
          </div>
        </Section>
      )}

      {validation && (
        <Section title="Validation" icon={Shield} iconColor="text-yellow-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{validation.description as string}</p>
            {validation.endpoint && (
              <code className="text-[10px] text-cyan-400 font-mono block">{validation.endpoint as string}</code>
            )}
            {validation.checks_performed && (
              <ol className="space-y-0.5">
                {(validation.checks_performed as string[]).map((check, i) => (
                  <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                    <span className="text-amber-400 shrink-0">{check.match(/^\d+/)?.[0] || i + 1}.</span>
                    <span>{check.replace(/^\d+\.\s*/, '')}</span>
                  </li>
                ))}
              </ol>
            )}
            {validation.response && (
              <FieldTable fields={validation.response as Record<string, string | Record<string, any>>} />
            )}
          </div>
        </Section>
      )}

      {endpoints && (
        <Section title="API Endpoints" icon={ListChecks} iconColor="text-teal-400">
          <div className="space-y-3 pt-2">
            {Object.entries(endpoints).map(([group, groupEndpoints]) => {
              const endpointMap = typeof groupEndpoints === 'string' ? { [group]: groupEndpoints } : groupEndpoints
              const label = typeof groupEndpoints === 'string' ? 'endpoints' : group.replace(/_/g, ' ')
              return (
                <div key={group}>
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                    {label}
                  </div>
                  <div className="space-y-0.5">
                    {Object.entries(endpointMap).map(([endpoint, desc]) => (
                      <div key={endpoint} className="flex gap-2 text-[11px] py-0.5">
                        <code className="text-cyan-400 font-mono shrink-0 text-[10px]">{endpoint}</code>
                        <span className="text-muted-foreground">{String(desc)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </Section>
      )}

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
    </div>
  )
}

export default function DataSourceApiDocsFlyout({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const docsQuery = useQuery({
    queryKey: ['data-source-docs'],
    queryFn: getUnifiedDataSourceDocs,
    staleTime: 60000,
    refetchOnMount: 'always',
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
                Data Source Developer Reference
                <Badge variant="outline" className="text-[9px] h-4 font-normal">v2.0</Badge>
              </SheetTitle>
              <SheetDescription>
                {'Source lifecycle: FETCH -> TRANSFORM -> UPSERT. Covers ingestion + SDK integration.'}
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
            {docsQuery.data && <DataSourceDocs docs={docsQuery.data} />}
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  )
}
