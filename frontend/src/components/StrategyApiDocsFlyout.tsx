import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Code2,
  Brain,
  Activity,
  DollarSign,
  Filter,
  Zap,
  Package,
  AlertTriangle,
  Copy,
  Check,
  Settings2,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { cn } from '../lib/utils'
import { getPluginDocs, getTraderStrategyDocs } from '../services/api'

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

function FieldTable({ fields }: { fields: Record<string, string> }) {
  return (
    <div className="space-y-0.5">
      {Object.entries(fields).map(([key, desc]) => (
        <div key={key} className="flex gap-2 text-[11px] py-0.5">
          <code className="text-amber-400 font-mono shrink-0">{key}</code>
          <span className="text-muted-foreground">{desc}</span>
        </div>
      ))}
    </div>
  )
}

// ==================== IMPORT LIST ====================

function ImportList({ imports }: { imports: Array<{ module: string; items: string }> }) {
  return (
    <div className="space-y-1">
      {imports.map((imp) => (
        <div key={imp.module} className="flex gap-2 text-[11px] py-0.5">
          <code className="text-cyan-400 font-mono shrink-0">{imp.module}</code>
          <span className="text-muted-foreground">{imp.items}</span>
        </div>
      ))}
    </div>
  )
}

// ==================== OPPORTUNITY DOCS ====================

function OpportunityDocs({ docs }: { docs: Record<string, any> }) {
  return (
    <div className="space-y-2">
      {/* Overview */}
      <div className="text-xs text-muted-foreground px-1 pb-1">
        {docs.overview?.description}
      </div>

      {/* Class Structure */}
      <Section title="Class Structure" icon={Code2} iconColor="text-amber-400" defaultOpen>
        <div className="space-y-2 pt-2">
          <p className="text-[11px] text-muted-foreground">{docs.class_structure?.description}</p>
          {docs.class_structure?.required_attributes && (
            <div>
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Required Attributes</div>
              <FieldTable fields={docs.class_structure.required_attributes} />
            </div>
          )}
          {docs.class_structure?.inherited_attributes && (
            <div>
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Inherited (from BaseStrategy)</div>
              <FieldTable fields={docs.class_structure.inherited_attributes} />
            </div>
          )}
        </div>
      </Section>

      {/* detect() Method */}
      <Section title="detect() Method" icon={Zap} iconColor="text-emerald-400" defaultOpen>
        <div className="space-y-2 pt-2">
          <CodeBlock code={docs.detect_method?.signature || ''} />
          <p className="text-[11px] text-muted-foreground">{docs.detect_method?.description}</p>

          {docs.detect_method?.parameters && Object.entries(docs.detect_method.parameters).map(([name, param]: [string, any]) => (
            <div key={name} className="space-y-1">
              <div className="flex items-center gap-2">
                <code className="text-[11px] font-mono text-cyan-400">{name}</code>
                <Badge variant="outline" className="text-[9px] h-4">{param.type}</Badge>
              </div>
              <p className="text-[11px] text-muted-foreground">{param.description}</p>
              {param.fields && <FieldTable fields={param.fields} />}
              {param.structure && <code className="text-[10px] text-muted-foreground font-mono">{param.structure}</code>}
              {param.usage && <p className="text-[10px] text-muted-foreground italic">{param.usage}</p>}
            </div>
          ))}
        </div>
      </Section>

      {/* create_opportunity() */}
      {docs.create_opportunity_method && (
        <Section title="create_opportunity()" icon={Package} iconColor="text-purple-400">
          <div className="space-y-2 pt-2">
            <CodeBlock code={docs.create_opportunity_method.signature || ''} />
            <p className="text-[11px] text-muted-foreground">{docs.create_opportunity_method.description}</p>
            {docs.create_opportunity_method.parameters && (
              <FieldTable fields={docs.create_opportunity_method.parameters} />
            )}
            {docs.create_opportunity_method.hard_filters_applied && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1 mt-2">Hard Filters Applied</div>
                <ul className="space-y-0.5">
                  {docs.create_opportunity_method.hard_filters_applied.map((f: string, i: number) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <span className="text-amber-400 mt-0.5">-</span> {f}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* AI / LLM Integration */}
      {docs.ai_integration && (
        <Section title="AI / LLM Integration" icon={Brain} iconColor="text-violet-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.ai_integration.description}</p>
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Basic Chat</div>
            <CodeBlock code={docs.ai_integration.usage || ''} />
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Structured JSON Output</div>
            <CodeBlock code={docs.ai_integration.structured_output || ''} />
            {docs.ai_integration.available_classes && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Available Classes</div>
                <FieldTable fields={docs.ai_integration.available_classes} />
              </div>
            )}
            {docs.ai_integration.notes && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Notes</div>
                <ul className="space-y-0.5">
                  {docs.ai_integration.notes.map((n: string, i: number) => (
                    <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                      <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" /> {n}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Strategy SDK */}
      {docs.strategy_sdk && (
        <Section title="Strategy SDK" icon={Package} iconColor="text-cyan-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.strategy_sdk.description}</p>
            {docs.strategy_sdk.methods && (
              <div className="space-y-0.5">
                {Object.entries(docs.strategy_sdk.methods).map(([method, desc]) => (
                  <div key={method} className="flex gap-2 text-[11px] py-0.5">
                    <code className="text-emerald-400 font-mono shrink-0 text-[10px]">{method}</code>
                    <span className="text-muted-foreground">{desc as string}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Config System */}
      {docs.config_system && (
        <Section title="Config System" icon={Settings2} iconColor="text-blue-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.config_system.description}</p>
            <CodeBlock code={docs.config_system.example || ''} />
          </div>
        </Section>
      )}

      {/* Cookbook */}
      {docs.cookbook && (
        <Section title="Cookbook & Recipes" icon={BookOpen} iconColor="text-orange-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.cookbook.description}</p>
            {docs.cookbook.recipes && Object.entries(docs.cookbook.recipes).map(([name, code]) => (
              <div key={name}>
                <div className="text-[10px] font-medium text-muted-foreground mb-1">
                  {name.replace(/_/g, ' ')}
                </div>
                <CodeBlock code={code as string} />
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Common Patterns */}
      {docs.common_patterns && (
        <Section title="Common Patterns" icon={Code2} iconColor="text-teal-400">
          <div className="space-y-3 pt-2">
            {Object.entries(docs.common_patterns).map(([name, code]) => (
              <div key={name}>
                <div className="text-[10px] font-medium text-muted-foreground mb-1">
                  {name.replace(/_/g, ' ')}
                </div>
                <CodeBlock code={code as string} />
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Allowed Imports */}
      {docs.allowed_imports && (
        <Section title="Allowed Imports" icon={Package} iconColor="text-emerald-400">
          <div className="pt-2">
            <ImportList imports={docs.allowed_imports} />
          </div>
        </Section>
      )}

      {/* Blocked Imports */}
      {docs.blocked_imports && (
        <Section title="Blocked Imports" icon={AlertTriangle} iconColor="text-red-400">
          <ul className="space-y-0.5 pt-2">
            {docs.blocked_imports.map((b: string, i: number) => (
              <li key={i} className="text-[10px] text-red-400/80 flex items-start gap-1">
                <span className="text-red-500">x</span> {b}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Risk Scoring */}
      {docs.risk_scoring && (
        <Section title="Risk Scoring" icon={Activity} iconColor="text-yellow-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.risk_scoring.description}</p>
            {docs.risk_scoring.risk_factors_considered && (
              <ul className="space-y-0.5">
                {docs.risk_scoring.risk_factors_considered.map((f: string, i: number) => (
                  <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                    <span className="text-yellow-400">-</span> {f}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Section>
      )}
    </div>
  )
}

// ==================== TRADER DOCS ====================

function TraderDocs({ docs }: { docs: Record<string, any> }) {
  return (
    <div className="space-y-2">
      {/* Overview */}
      <div className="text-xs text-muted-foreground px-1 pb-1">
        {docs.overview?.description}
      </div>

      {/* Class Structure */}
      <Section title="Class Structure" icon={Code2} iconColor="text-cyan-400" defaultOpen>
        <div className="space-y-2 pt-2">
          {docs.class_structure?.imports && (
            <CodeBlock code={docs.class_structure.imports} />
          )}
          {docs.class_structure?.required_attributes && (
            <div>
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Required Attributes</div>
              <FieldTable fields={docs.class_structure.required_attributes} />
            </div>
          )}
        </div>
      </Section>

      {/* evaluate() Method */}
      <Section title="evaluate() Method" icon={Zap} iconColor="text-emerald-400" defaultOpen>
        <div className="space-y-2 pt-2">
          <CodeBlock code={docs.evaluate_method?.signature || ''} />
          <p className="text-[11px] text-muted-foreground">{docs.evaluate_method?.description}</p>

          {docs.evaluate_method?.parameters && Object.entries(docs.evaluate_method.parameters).map(([name, param]: [string, any]) => (
            <div key={name} className="space-y-1">
              <div className="flex items-center gap-2">
                <code className="text-[11px] font-mono text-cyan-400">{name}</code>
                {param.type && <Badge variant="outline" className="text-[9px] h-4">{param.type}</Badge>}
              </div>
              {typeof param === 'string' && <p className="text-[11px] text-muted-foreground">{param}</p>}
              {param.fields && <FieldTable fields={param.fields} />}
            </div>
          ))}

          {docs.evaluate_method?.returns && (
            <div className="space-y-1">
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Returns</div>
              {docs.evaluate_method.returns.fields && <FieldTable fields={docs.evaluate_method.returns.fields} />}
            </div>
          )}
        </div>
      </Section>

      {/* Decision Checks */}
      {docs.decision_checks && (
        <Section title="Decision Checks" icon={Filter} iconColor="text-purple-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.decision_checks.description}</p>
            <CodeBlock code={docs.decision_checks.signature || ''} />
            {docs.decision_checks.example && <CodeBlock code={docs.decision_checks.example} />}
          </div>
        </Section>
      )}

      {/* Sizing */}
      {docs.sizing && (
        <Section title="Trade Sizing" icon={DollarSign} iconColor="text-green-400">
          <div className="space-y-2 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.sizing.description}</p>
            {docs.sizing.example && <CodeBlock code={docs.sizing.example} />}
          </div>
        </Section>
      )}

      {/* Cookbook */}
      {docs.cookbook && (
        <Section title="Cookbook & Recipes" icon={BookOpen} iconColor="text-orange-400">
          <div className="space-y-3 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.cookbook.description}</p>
            {docs.cookbook.recipes && Object.entries(docs.cookbook.recipes).map(([name, code]) => (
              <div key={name}>
                <div className="text-[10px] font-medium text-muted-foreground mb-1">
                  {name.replace(/_/g, ' ')}
                </div>
                <CodeBlock code={code as string} />
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Allowed Imports */}
      {docs.allowed_imports && (
        <Section title="Allowed Imports" icon={Package} iconColor="text-emerald-400">
          <div className="pt-2">
            <ImportList imports={docs.allowed_imports} />
          </div>
        </Section>
      )}

      {/* Blocked Imports */}
      {docs.blocked_imports && (
        <Section title="Blocked Imports" icon={AlertTriangle} iconColor="text-red-400">
          <ul className="space-y-0.5 pt-2">
            {docs.blocked_imports.map((b: string, i: number) => (
              <li key={i} className="text-[10px] text-red-400/80 flex items-start gap-1">
                <span className="text-red-500">x</span> {b}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Safety */}
      {docs.safety && (
        <Section title="Safety & Validation" icon={AlertTriangle} iconColor="text-yellow-400">
          <div className="space-y-1 pt-2">
            <p className="text-[11px] text-muted-foreground">{docs.safety.description}</p>
            {docs.safety.notes && (
              <ul className="space-y-0.5">
                {docs.safety.notes.map((n: string, i: number) => (
                  <li key={i} className="text-[10px] text-muted-foreground flex items-start gap-1">
                    <span className="text-yellow-400">-</span> {n}
                  </li>
                ))}
              </ul>
            )}
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
  variant,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  variant: 'opportunity' | 'trader'
}) {
  const docsQuery = useQuery({
    queryKey: [variant === 'opportunity' ? 'plugin-docs' : 'trader-strategy-docs'],
    queryFn: variant === 'opportunity' ? getPluginDocs : getTraderStrategyDocs,
    staleTime: Infinity,
    enabled: open,
  })

  const title = variant === 'opportunity' ? 'Opportunity Strategy API' : 'Trader Strategy API'
  const description = variant === 'opportunity'
    ? 'Available classes, methods, imports, and patterns for opportunity detection strategies.'
    : 'Available classes, methods, imports, and patterns for trader evaluation strategies.'

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-xl p-0">
        <div className="h-full min-h-0 flex flex-col">
          <div className="border-b border-border px-4 py-3">
            <SheetHeader className="space-y-1 text-left">
              <SheetTitle className="text-base flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                {title}
                <Badge variant="outline" className="text-[9px] h-4 font-normal">Reference</Badge>
              </SheetTitle>
              <SheetDescription>{description}</SheetDescription>
            </SheetHeader>
          </div>

          <ScrollArea className="flex-1 min-h-0 px-4 py-3">
            {docsQuery.isLoading && (
              <div className="text-xs text-muted-foreground text-center py-8">Loading API reference...</div>
            )}
            {docsQuery.error && (
              <div className="text-xs text-red-400 text-center py-8">Failed to load docs</div>
            )}
            {docsQuery.data && (
              variant === 'opportunity'
                ? <OpportunityDocs docs={docsQuery.data} />
                : <TraderDocs docs={docsQuery.data} />
            )}
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  )
}
