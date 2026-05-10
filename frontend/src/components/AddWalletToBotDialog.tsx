import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
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

function errorMessage(error: unknown, t: TFunction): string {
  const detail = (error as any)?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  const message = (error as any)?.message
  if (typeof message === 'string' && message.trim()) return message.trim()
  return t('addWalletToBotDialog.failedToAdd')
}

function resolveDefaultBotName(walletAddress: string, walletLabel: string | null | undefined, t: TFunction): string {
  const label = String(walletLabel || '').trim()
  const suffix = t('addWalletToBotDialog.copyBotSuffix')
  if (label) return `${label} ${suffix}`
  return `${shortAddress(walletAddress)} ${suffix}`
}

function resolveTraderCaption(trader: Trader, t: TFunction): string {
  const sourceCount = Array.isArray(trader.source_configs) ? trader.source_configs.length : 0
  const mode = trader.mode === 'live'
    ? t('addWalletToBotDialog.modeLive')
    : t('addWalletToBotDialog.modePaper')
  const sourcesLabel = sourceCount === 1
    ? t('addWalletToBotDialog.sourceSingular')
    : t('addWalletToBotDialog.sourcePlural')
  return `${mode} | ${sourceCount} ${sourcesLabel}`
}

export default function AddWalletToBotDialog({
  open,
  walletAddress,
  walletLabel,
  onOpenChange,
  onAdded,
}: AddWalletToBotDialogProps) {
  const { t } = useTranslation()
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
        caption: resolveTraderCaption(trader, t),
      })),
    [traders, t],
  )

  useEffect(() => {
    if (!open || !walletAddress) return
    setTarget('new')
    setNewTraderMode('paper')
    setNewTraderName(resolveDefaultBotName(walletAddress, walletLabel, t))
    setExistingTraderId('')
  }, [open, walletAddress, walletLabel, t])

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
            {t('addWalletToBotDialog.title')}
          </DialogTitle>
          <DialogDescription>
            {t('addWalletToBotDialog.description')}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="rounded-md border border-border bg-muted/35 px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{t('addWalletToBotDialog.wallet')}</p>
            <p className="mt-0.5 font-mono text-sm">{walletAddress || '--'}</p>
            {walletLabel && (
              <p className="mt-0.5 text-[11px] text-muted-foreground">{walletLabel}</p>
            )}
          </div>

          <div className="space-y-1">
            <Label>{t('addWalletToBotDialog.target')}</Label>
            <Select value={target} onValueChange={(value) => setTarget(value as AddWalletToTraderBotTarget)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="new">{t('addWalletToBotDialog.createNewBot')}</SelectItem>
                <SelectItem value="existing">{t('addWalletToBotDialog.addToExistingBot')}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {target === 'new' ? (
            <div className="space-y-3">
              <div className="space-y-1">
                <Label>{t('addWalletToBotDialog.botName')}</Label>
                <Input
                  value={newTraderName}
                  onChange={(event) => setNewTraderName(event.target.value)}
                  placeholder={t('addWalletToBotDialog.botNamePlaceholder')}
                />
              </div>
              <div className="space-y-1">
                <Label>{t('addWalletToBotDialog.mode')}</Label>
                <Select value={newTraderMode} onValueChange={(value) => setNewTraderMode(value === 'live' ? 'live' : 'paper')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="paper">{t('addWalletToBotDialog.paper')}</SelectItem>
                    <SelectItem value="live">{t('addWalletToBotDialog.live')}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <Label>{t('addWalletToBotDialog.existingBot')}</Label>
              <Select value={existingTraderId} onValueChange={setExistingTraderId} disabled={existingTraderOptions.length === 0 || tradersQuery.isLoading}>
                <SelectTrigger>
                  <SelectValue placeholder={tradersQuery.isLoading ? t('addWalletToBotDialog.loadingBots') : t('addWalletToBotDialog.selectBot')} />
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
                <p className="text-xs text-muted-foreground">{t('addWalletToBotDialog.noExistingBots')}</p>
              )}
            </div>
          )}

          {addWalletMutation.isError && (
            <p className="text-xs text-rose-400">{errorMessage(addWalletMutation.error, t)}</p>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={addWalletMutation.isPending}>
            {t('common.cancel')}
          </Button>
          <Button
            onClick={() => addWalletMutation.mutate()}
            disabled={!canSubmit || addWalletMutation.isPending}
            className="gap-2"
          >
            {addWalletMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlusCircle className="h-4 w-4" />}
            {target === 'new' ? t('addWalletToBotDialog.createBotAndAddWallet') : t('addWalletToBotDialog.addWallet')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
