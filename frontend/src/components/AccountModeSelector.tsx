import { useAtom } from 'jotai'
import { Shield, Zap } from 'lucide-react'
import { cn } from '../lib/utils'
import { accountModeAtom } from '../store/atoms'
import type { AccountMode } from '../store/atoms'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'

export default function AccountModeSelector() {
  const [mode, setMode] = useAtom(accountModeAtom)

  const toggle = (newMode: AccountMode) => {
    if (newMode === 'live' && mode !== 'live') {
      setMode('live')
    } else {
      setMode(newMode)
    }
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex items-center h-7 rounded-lg border border-border/50 bg-card/60 overflow-hidden">
          <button
            onClick={() => toggle('sandbox')}
            className={cn(
              "flex items-center gap-1.5 px-2.5 h-full text-[10px] font-semibold uppercase tracking-wide transition-all",
              mode === 'sandbox'
                ? "bg-amber-500/15 text-amber-400 border-r border-amber-500/20"
                : "text-muted-foreground hover:text-foreground border-r border-border/50"
            )}
          >
            <Shield className="w-3 h-3" />
            Sandbox
          </button>
          <button
            onClick={() => toggle('live')}
            className={cn(
              "flex items-center gap-1.5 px-2.5 h-full text-[10px] font-semibold uppercase tracking-wide transition-all",
              mode === 'live'
                ? "bg-green-500/15 text-green-400"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Zap className="w-3 h-3" />
            Live
          </button>
        </div>
      </TooltipTrigger>
      <TooltipContent>
        {mode === 'sandbox' ? 'Sandbox mode — simulated trading' : 'Live mode — real money trading'}
      </TooltipContent>
    </Tooltip>
  )
}

export function SandboxIndicator() {
  const [mode] = useAtom(accountModeAtom)

  if (mode !== 'sandbox') return null

  return (
    <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[10px] font-semibold uppercase tracking-wide">
      <Shield className="w-3 h-3" />
      Sandbox
    </div>
  )
}
