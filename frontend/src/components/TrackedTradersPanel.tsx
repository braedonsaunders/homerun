import { useState } from 'react'
import { FolderTree, Wallet } from 'lucide-react'
import WalletTracker from './WalletTracker'
import RecentTradesPanel from './RecentTradesPanel'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

interface TrackedTradersPanelProps {
  onAnalyzeWallet?: (address: string, username?: string) => void
  onNavigateToWallet?: (address: string) => void
}

type TrackedView = 'wallets' | 'groups'

export default function TrackedTradersPanel({ onAnalyzeWallet, onNavigateToWallet }: TrackedTradersPanelProps) {
  const [activeView, setActiveView] = useState<TrackedView>('wallets')

  const navigateWallet = (address: string, username?: string | null) => {
    if (onNavigateToWallet) {
      onNavigateToWallet(address)
      return
    }
    onAnalyzeWallet?.(address, username || undefined)
  }

  return (
    <div className="space-y-3">
      <Tabs value={activeView} onValueChange={(value) => setActiveView(value as TrackedView)}>
        <TabsList className="flex h-auto w-full justify-start gap-2 bg-transparent p-0">
          <TabsTrigger
            value="wallets"
            className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-300 data-[state=active]:border data-[state=active]:border-emerald-500/40 data-[state=active]:shadow-none"
          >
            <Wallet className="w-4 h-4" />
            Wallets
          </TabsTrigger>
          <TabsTrigger
            value="groups"
            className="gap-2 rounded-lg bg-muted text-muted-foreground hover:text-foreground data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-300 data-[state=active]:border data-[state=active]:border-amber-500/40 data-[state=active]:shadow-none"
          >
            <FolderTree className="w-4 h-4" />
            Groups
          </TabsTrigger>
        </TabsList>

        <TabsContent value="wallets" className="mt-4">
          <WalletTracker
            section="tracked"
            showManagementPanel={false}
            onAnalyzeWallet={onAnalyzeWallet}
            onNavigateToWallet={onNavigateToWallet}
          />
        </TabsContent>

        <TabsContent value="groups" className="mt-4">
          <RecentTradesPanel
            mode="management"
            onNavigateToWallet={(address) => navigateWallet(address)}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}
