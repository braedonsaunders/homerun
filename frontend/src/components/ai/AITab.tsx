import { lazy, Suspense } from 'react'
import { useAtom } from 'jotai'
import { useTranslation } from 'react-i18next'
import { aiTabSubtabAtom } from '../../store/atoms'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs'
import { RefreshCw } from 'lucide-react'
import { cn } from '../../lib/utils'
import AIChatView from './AIChatView'
import AIAgentsView from './AIAgentsView'
import AIToolsView from './AIToolsView'

// Lazy load heavier subtabs for code splitting
const AIProvidersView = lazy(() => import('./AIProvidersView'))
const AIModelsView = lazy(() => import('./AIModelsView'))
const AIActivityView = lazy(() => import('./AIActivityView'))
function SubtabFallback() {
  return (
    <div className="flex items-center justify-center py-16">
      <RefreshCw className="w-5 h-5 animate-spin text-purple-400" />
    </div>
  )
}

const VALID_SUBTABS = new Set(['chat', 'agents', 'tools', 'providers', 'models', 'activity'])

export default function AITab() {
  const { t } = useTranslation()
  const [subtab, setSubtab] = useAtom(aiTabSubtabAtom)

  // Guard against stale localStorage values (e.g. removed 'system' tab)
  const activeSubtab = VALID_SUBTABS.has(subtab) ? subtab : 'chat'
  if (activeSubtab !== subtab) {
    // Fix the stored value
    setTimeout(() => setSubtab('chat'), 0)
  }

  return (
    <div className="flex flex-col h-full">
      <Tabs value={activeSubtab} onValueChange={(v) => setSubtab(v as typeof subtab)} className="flex flex-col h-full">
        <div className="border-b border-white/10 px-4">
          <TabsList className="bg-transparent gap-1">
            <TabsTrigger value="chat" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-300">{t('ai.chat')}</TabsTrigger>
            <TabsTrigger value="agents" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300">{t('ai.agents')}</TabsTrigger>
            <TabsTrigger value="tools" className="data-[state=active]:bg-violet-500/20 data-[state=active]:text-violet-700 dark:text-violet-300">{t('ai.tools')}</TabsTrigger>
            <TabsTrigger value="providers" className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-300">{t('ai.providers')}</TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-300">{t('ai.models')}</TabsTrigger>
            <TabsTrigger value="activity" className="data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-300">{t('ai.activity')}</TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="chat" forceMount className={cn('flex-1 overflow-hidden mt-0', activeSubtab !== 'chat' && 'hidden')}><AIChatView /></TabsContent>
        <TabsContent value="agents" className="flex-1 overflow-auto mt-0 p-4"><AIAgentsView /></TabsContent>
        <TabsContent value="tools" className="flex-1 overflow-auto mt-0 p-4"><AIToolsView /></TabsContent>
        <TabsContent value="providers" className="flex-1 overflow-auto mt-0 p-4">
          <Suspense fallback={<SubtabFallback />}><AIProvidersView /></Suspense>
        </TabsContent>
        <TabsContent value="models" className="flex-1 overflow-auto mt-0 p-4">
          <Suspense fallback={<SubtabFallback />}><AIModelsView /></Suspense>
        </TabsContent>
        <TabsContent value="activity" className="flex-1 overflow-auto mt-0 p-4">
          <Suspense fallback={<SubtabFallback />}><AIActivityView /></Suspense>
        </TabsContent>
      </Tabs>
    </div>
  )
}
