import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, Loader2, PlusCircle } from 'lucide-react'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { getTraders, type Trader } from '../services/api'
import { addWalletToTraderBot, type AddWalletToTraderBotResult, type AddWalletToTraderBotTarget } from '../lib/traderBotActions'

interface AddWalletToBotDialogProps {
  open: boolean
  walletAddress: string | null
  walletLabel?: string | null
  onOpenChange: (open: boolean) => void
  onAdded?: (result: AddWalletToTraderBotResult) => void
}

function shortAddress(address: string): string {
  if (address.length <= 12) return address
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

function errorMessage(error: unknown): string {
  const detail = (error as any)?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  const message = (error as any)?.message
  if (typeof message === 'string' && message.trim()) return message.trim()
  return 'Failed to add wallet to bot'
}

function resolveDefaultBotName(walletAddress: string, walletLabel?: string | null): string {
  const label = String(walletLabel || '').trim()
  if (label) return `${label} Copy Bot`
  return `${shortAddress(walletAddress)} Copy Bot`
}

function resolveTraderCaption(trader: Trader): string {
  const sourceCount = Array.isArray(trader.source_configs) ? trader.source_configs.length : 0
  const mode = trader.mode === 'live' ? 'live' : 'paper'
  return `${mode} | ${sourceCount} source${sourceCount === 1 ? '' : 's'}`
}

export default function AddWalletToBotDialog({
  open,
  walletAddress,
  walletLabel,
  onOpenChange,
  onAdded,
}: AddWalletToBotDialogProps) {
  const queryClient = useQueryClient()
  const [target, setTarget] = useState<AddWalletToTraderBotTarget>('new')
  const [newTraderName, setNewTraderName] = useState('')
  const [newTraderMode, setNewTraderMode] = useState<'paper' | 'live'>('paper')
  const [existingTraderId, setExistingTraderId] = useState('')

  const tradersQuery = useQuery({
    queryKey: ['traders-list', 'wallet-picker'],
    queryFn: () => getTraders(),
    enabled: open,
    staleTime: 30_000,
  })
  const traders = tradersQuery.data || []

  const existingTraderOptions = useMemo(
    () =>
      traders.map((trader) => ({
        id: trader.id,
        name: trader.name,
        caption: resolveTraderCaption(trader),
      })),
    [traders],
  )

  useEffect(() => {
    if (!open || !walletAddress) return
    setTarget('new')
    setNewTraderMode('paper')
    setNewTraderName(resolveDefaultBotName(walletAddress, walletLabel))
    setExistingTraderId('')
  }, [open, walletAddress, walletLabel])

  useEffect(() => {
    if (!open || target !== 'existing') return
    if (existingTraderId) return
    if (existingTraderOptions.length === 0) return
    setExistingTraderId(existingTraderOptions[0].id)
  }, [existingTraderId, existingTraderOptions, open, target])

  const addWalletMutation = useMutation({
    mutationFn: async () => {
      if (!walletAddress) {
        throw new Error('Wallet address is required')
      }
      return addWalletToTraderBot({
        walletAddress,
        walletLabel: walletLabel || undefined,
        target,
        newTraderName: target === 'new' ? newTraderName : undefined,
        existingTraderIdOrName: target === 'existing' ? existingTraderId : undefined,
        mode: target === 'new' ? newTraderMode : undefined,
        tradersSnapshot: traders,
      })
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['traders-list'] })
      queryClient.invalidateQueries({ queryKey: ['trader-orchestrator-overview'] })
      queryClient.invalidateQueries({ queryKey: ['traders-overview'] })
      queryClient.invalidateQueries({ queryKey: ['opportunities', 'traders'] })
      queryClient.invalidateQueries({ queryKey: ['traders-scope-pool-members'] })
      onAdded?.(result)
      onOpenChange(false)
    },
  })

  const canSubmit = Boolean(
    walletAddress
    && (target === 'new' || existingTraderId),
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base">
            <Bot className="h-4 w-4 text-sky-400" />
            Add Wallet To Bot
          </DialogTitle>
          <DialogDescription>
            Configure copy-trade routing for this wallet.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="rounded-md border border-border bg-muted/35 px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Wallet</p>
            <p className="mt-0.5 font-mono text-sm">{walletAddress || '--'}</p>
            {walletLabel && (
              <p className="mt-0.5 text-[11px] text-muted-foreground">{walletLabel}</p>
            )}
          </div>

          <div className="space-y-1">
            <Label>Target</Label>
            <Select value={target} onValueChange={(value) => setTarget(value as AddWalletToTraderBotTarget)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="new">Create new bot</SelectItem>
                <SelectItem value="existing">Add to existing bot</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {target === 'new' ? (
            <div className="space-y-3">
              <div className="space-y-1">
                <Label>Bot Name</Label>
                <Input
                  value={newTraderName}
                  onChange={(event) => setNewTraderName(event.target.value)}
                  placeholder="Wallet Copy Bot"
                />
              </div>
              <div className="space-y-1">
                <Label>Mode</Label>
                <Select value={newTraderMode} onValueChange={(value) => setNewTraderMode(value === 'live' ? 'live' : 'paper')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="paper">Paper</SelectItem>
                    <SelectItem value="live">Live</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <Label>Existing Bot</Label>
              <Select value={existingTraderId} onValueChange={setExistingTraderId} disabled={existingTraderOptions.length === 0 || tradersQuery.isLoading}>
                <SelectTrigger>
                  <SelectValue placeholder={tradersQuery.isLoading ? 'Loading bots...' : 'Select bot'} />
                </SelectTrigger>
                <SelectContent>
                  {existingTraderOptions.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.name} ({option.caption})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {!tradersQuery.isLoading && existingTraderOptions.length === 0 && (
                <p className="text-xs text-muted-foreground">No existing bots found. Select "Create new bot".</p>
              )}
            </div>
          )}

          {addWalletMutation.isError && (
            <p className="text-xs text-rose-400">{errorMessage(addWalletMutation.error)}</p>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={addWalletMutation.isPending}>
            Cancel
          </Button>
          <Button
            onClick={() => addWalletMutation.mutate()}
            disabled={!canSubmit || addWalletMutation.isPending}
            className="gap-2"
          >
            {addWalletMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlusCircle className="h-4 w-4" />}
            {target === 'new' ? 'Create Bot + Add Wallet' : 'Add Wallet'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
