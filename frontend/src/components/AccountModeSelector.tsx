import { useState, useRef, useEffect } from 'react'
import { useAtom } from 'jotai'
import { useQuery } from '@tanstack/react-query'
import { Shield, Zap, ChevronDown, DollarSign, BarChart3, Check } from 'lucide-react'
import { cn } from '../lib/utils'
import { accountModeAtom, selectedAccountIdAtom } from '../store/atoms'
import { getSimulationAccounts } from '../services/api'

export default function AccountModeSelector() {
  const [, setMode] = useAtom(accountModeAtom)
  const [selectedAccountId, setSelectedAccountId] = useAtom(selectedAccountIdAtom)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const { data: sandboxAccounts = [], isFetched: sandboxAccountsFetched } = useQuery({
    queryKey: ['simulation-accounts'],
    queryFn: getSimulationAccounts,
  })

  // Auto-select first sandbox account if none selected and accounts loaded
  useEffect(() => {
    if (!selectedAccountId && sandboxAccounts.length > 0) {
      setSelectedAccountId(sandboxAccounts[0].id)
      setMode('sandbox')
    }
  }, [sandboxAccounts, selectedAccountId, setSelectedAccountId, setMode])

  // If a stale sandbox id is persisted, clear it once accounts are fetched.
  useEffect(() => {
    if (!sandboxAccountsFetched || !selectedAccountId) return
    if (selectedAccountId.startsWith('live:')) return

    const exists = sandboxAccounts.some((account) => account.id === selectedAccountId)
    if (exists) return

    if (sandboxAccounts.length > 0) {
      setSelectedAccountId(sandboxAccounts[0].id)
      setMode('sandbox')
      return
    }

    setSelectedAccountId(null)
  }, [sandboxAccountsFetched, sandboxAccounts, selectedAccountId, setSelectedAccountId, setMode])

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const selectedSandbox = sandboxAccounts.find(a => a.id === selectedAccountId)
  const isLive = selectedAccountId?.startsWith('live:')
  const livePlatform = isLive ? selectedAccountId?.replace('live:', '') : null

  // Display label for the selected account
  const getLabel = () => {
    if (isLive) {
      return livePlatform === 'polymarket' ? 'Polymarket' : 'Kalshi'
    }
    if (selectedSandbox) {
      return selectedSandbox.name
    }
    return 'Select Account'
  }

  const selectAccount = (id: string) => {
    setSelectedAccountId(id)
    if (id.startsWith('live:')) {
      setMode('live')
    } else {
      setMode('sandbox')
    }
    setOpen(false)
  }

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 h-7 px-2.5 rounded-lg border text-[11px] font-semibold transition-all",
          isLive
            ? "bg-green-500/15 text-green-400 border-green-500/30 hover:bg-green-500/20"
            : "bg-amber-500/15 text-amber-400 border-amber-500/30 hover:bg-amber-500/20"
        )}
      >
        {isLive ? <Zap className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
        <span className="max-w-[120px] truncate">{getLabel()}</span>
        <ChevronDown className={cn("w-3 h-3 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute top-full mt-1 left-0 z-[100] w-72 bg-popover border border-border rounded-lg shadow-xl shadow-black/30 overflow-hidden">
          {/* Sandbox Accounts */}
          <div className="px-3 py-2 border-b border-border">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-amber-400">
              <Shield className="w-3 h-3" />
              Sandbox Accounts
            </div>
          </div>
          <div className="max-h-48 overflow-y-auto">
            {sandboxAccounts.length === 0 ? (
              <div className="px-3 py-3 text-xs text-muted-foreground">
                No sandbox accounts yet
              </div>
            ) : (
              sandboxAccounts.map(acc => (
                <button
                  key={acc.id}
                  onClick={() => selectAccount(acc.id)}
                  className={cn(
                    "w-full text-left px-3 py-2 hover:bg-accent transition-colors flex items-center gap-3",
                    selectedAccountId === acc.id && "bg-accent"
                  )}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-foreground truncate">{acc.name}</div>
                    <div className="text-xs text-muted-foreground font-mono">
                      ${(acc.current_capital ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  {selectedAccountId === acc.id && (
                    <Check className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                  )}
                </button>
              ))
            )}
          </div>

          {/* Live Accounts */}
          <div className="px-3 py-2 border-y border-border">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-green-400">
              <Zap className="w-3 h-3" />
              Live Accounts
            </div>
          </div>
          <button
            onClick={() => selectAccount('live:polymarket')}
            className={cn(
              "w-full text-left px-3 py-2 hover:bg-accent transition-colors flex items-center gap-3",
              selectedAccountId === 'live:polymarket' && "bg-accent"
            )}
          >
            <div className="w-7 h-7 bg-blue-500/20 rounded-md flex items-center justify-center shrink-0">
              <DollarSign className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm text-foreground">Polymarket</div>
              <div className="text-xs text-muted-foreground">Live USDC trading</div>
            </div>
            {selectedAccountId === 'live:polymarket' && (
              <Check className="w-3.5 h-3.5 text-green-400 shrink-0" />
            )}
          </button>
          <button
            onClick={() => selectAccount('live:kalshi')}
            className={cn(
              "w-full text-left px-3 py-2 hover:bg-accent transition-colors flex items-center gap-3",
              selectedAccountId === 'live:kalshi' && "bg-accent"
            )}
          >
            <div className="w-7 h-7 bg-indigo-500/20 rounded-md flex items-center justify-center shrink-0">
              <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm text-foreground">Kalshi</div>
              <div className="text-xs text-muted-foreground">Live USD trading</div>
            </div>
            {selectedAccountId === 'live:kalshi' && (
              <Check className="w-3.5 h-3.5 text-green-400 shrink-0" />
            )}
          </button>
        </div>
      )}
    </div>
  )
}
