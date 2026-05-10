import { useState, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Trans, useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'
import {
  ShoppingCart,
  Check,
  AlertTriangle,
  Loader2,
  Bot,
  DollarSign,
  Zap,
  Target,
  Shield,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { getTraders, traderManualBuy, Opportunity, Trader } from '../services/api'
import { Button } from './ui/button'
import type { UnifiedTraderSignal } from './TraderSignalViews'

// ─── Types ──────────────────────────────────────────────

interface BuyPosition {
  token_id: string
  side: string
  price: number
  market_id: string
  market_question: string
  outcome: string
}

interface BuyConfig {
  title: string
  positions: BuyPosition[]
  defaultSizeUsd: number
  opportunityId?: string
  suggestedSizeUsd?: number | null
  direction?: 'BUY' | 'SELL' | null
  confidence?: number | null
  roi?: number | null
  risk?: number | null
}

interface BuyButtonProps {
  opportunity?: Opportunity
  traderSignal?: UnifiedTraderSignal
  className?: string
  variant?: 'full' | 'compact' | 'inline'
}

// ─── Size Presets ───────────────────────────────────────

const SIZE_PRESETS = [5, 10, 25, 50, 100, 250]

// ─── Helpers ────────────────────────────────────────────

function buildConfigFromOpportunity(opp: Opportunity): BuyConfig {
  return {
    title: opp.title,
    positions: opp.positions_to_take.map(pos => ({
      token_id: pos.token_id || '',
      side: pos.action || 'BUY',
      price: pos.price,
      market_id: pos.market_id || '',
      market_question: pos.market || '',
      outcome: pos.outcome || '',
    })),
    defaultSizeUsd: opp.total_cost > 0 ? opp.total_cost : 10,
    opportunityId: opp.id,
    direction: opp.positions_to_take[0]?.action === 'SELL' ? 'SELL' : 'BUY',
    confidence: opp.confidence ?? null,
    roi: opp.roi_percent ?? null,
    risk: opp.risk_score ?? null,
  }
}

function resolveTraderSignalOutcomeLabel(sig: UnifiedTraderSignal): string {
  const yesLabel = String(sig.yes_label || sig.outcome_labels?.[0] || 'Yes').trim() || 'Yes'
  const noLabel = String(sig.no_label || sig.outcome_labels?.[1] || 'No').trim() || 'No'
  if (sig.selected_outcome_label) return String(sig.selected_outcome_label).trim() || yesLabel
  if (sig.direction === 'SELL') return noLabel
  if (sig.direction === 'BUY') return yesLabel
  return String(sig.outcome || yesLabel).trim() || yesLabel
}

function buildConfigFromTraderSignal(sig: UnifiedTraderSignal): BuyConfig {
  const price = sig.direction === 'SELL'
    ? (sig.current_no_price ?? sig.no_price ?? 0.5)
    : (sig.current_yes_price ?? sig.yes_price ?? 0.5)
  const outcome = resolveTraderSignalOutcomeLabel(sig)

  return {
    title: sig.market_question,
    positions: [{
      token_id: '',
      side: 'BUY',
      price,
      market_id: sig.market_id || '',
      market_question: sig.market_question || '',
      outcome,
    }],
    defaultSizeUsd: sig.suggested_size_usd ?? 10,
    suggestedSizeUsd: sig.suggested_size_usd,
    direction: 'BUY',
    confidence: sig.confidence ?? null,
    roi: sig.edge_percent ?? null,
    risk: null,
  }
}

function formatConfidencePercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${(value > 1 ? value : value * 100).toFixed(0)}%`
}

function formatUsd(value: number): string {
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}k`
  return `$${value.toFixed(value < 1 ? 2 : 0)}`
}

// ─── Component ──────────────────────────────────────────

export default function BuyButton({ opportunity, traderSignal, className, variant = 'full' }: BuyButtonProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [result, setResult] = useState<{ success: boolean; message: string; traderId?: string } | null>(null)
  const [sizeUsd, setSizeUsd] = useState<number | null>(null)
  const [customSize, setCustomSize] = useState('')
  const [selectedTraderId, setSelectedTraderId] = useState<string | null>(null)
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market')
  const [limitPrice, setLimitPrice] = useState('')
  const [confirmStep, setConfirmStep] = useState(false)
  const queryClient = useQueryClient()

  const config = useMemo<BuyConfig | null>(() => {
    if (opportunity) return buildConfigFromOpportunity(opportunity)
    if (traderSignal) return buildConfigFromTraderSignal(traderSignal)
    return null
  }, [opportunity, traderSignal])

  const { data: traders = [] } = useQuery({
    queryKey: ['traders-for-buy'],
    queryFn: () => getTraders(),
    staleTime: 30_000,
    enabled: open,
  })

  const enabledTraders = traders.filter((t: Trader) => t.is_enabled)
  const selectedTrader = enabledTraders.find((t: Trader) => t.id === selectedTraderId) || null

  const effectiveSize = sizeUsd ?? (config?.defaultSizeUsd || 10)
  const effectivePrice = orderType === 'limit' && limitPrice
    ? parseFloat(limitPrice)
    : (config?.positions[0]?.price || 0.5)
  const estimatedShares = effectivePrice > 0 ? effectiveSize / effectivePrice : 0
  const estimatedPayout = estimatedShares * 1.0
  const estimatedProfit = estimatedPayout - effectiveSize

  const mutation = useMutation({
    mutationFn: (traderId: string) => {
      if (!config) throw new Error('No configuration')
      const positions = config.positions.map(pos => ({
        ...pos,
        price: orderType === 'limit' && limitPrice ? parseFloat(limitPrice) : pos.price,
      }))
      return traderManualBuy(traderId, {
        positions,
        size_usd: effectiveSize,
        opportunity_id: config.opportunityId,
      })
    },
    onSuccess: (data) => {
      const success = data.status !== 'partial_failure'
      setResult({ success, message: data.message, traderId: selectedTraderId || undefined })
      setConfirmStep(false)
      queryClient.invalidateQueries({ queryKey: ['trader-orders'] })
      queryClient.invalidateQueries({ queryKey: ['trading-positions'] })
    },
    onError: (error: any) => {
      setResult({
        success: false,
        message: error?.response?.data?.detail || error?.message || t('buyButton.buyFailed'),
      })
      setConfirmStep(false)
    },
  })

  function resetModal() {
    setSizeUsd(null)
    setCustomSize('')
    setSelectedTraderId(null)
    setOrderType('market')
    setLimitPrice('')
    setConfirmStep(false)
  }

  function handleOpen(e: React.MouseEvent) {
    e.stopPropagation()
    resetModal()
    setOpen(true)
  }

  function handleExecute() {
    if (!selectedTraderId) return
    if (selectedTrader?.mode === 'live' && !confirmStep) {
      setConfirmStep(true)
      return
    }
    mutation.mutate(selectedTraderId)
  }

  const hasPositions = config && config.positions.length > 0
  const isReady = hasPositions && selectedTraderId && effectiveSize > 0

  return (
    <div className={cn('relative', className)}>
      {variant === 'inline' ? (
        <button
          onClick={handleOpen}
          disabled={!hasPositions || mutation.isPending}
          className={cn(
            'inline-flex items-center gap-1 text-xs font-medium transition-colors',
            'text-emerald-400 hover:text-emerald-300',
            (!hasPositions || mutation.isPending) && 'opacity-40 pointer-events-none',
          )}
        >
          {mutation.isPending ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <ShoppingCart className="w-3 h-3" />
          )}
          {t('buyButton.buy')}
        </button>
      ) : (
        <Button
          onClick={handleOpen}
          disabled={!hasPositions || mutation.isPending}
          size="sm"
          className={cn(
            variant === 'compact' ? 'h-7 px-2.5 text-[11px]' : 'w-full h-8 text-xs',
            'font-medium',
            'bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600',
            'shadow-lg shadow-green-500/20 text-white',
          )}
        >
          {mutation.isPending ? (
            <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
          ) : (
            <ShoppingCart className="w-3 h-3 mr-1.5" />
          )}
          {t('buyButton.buy')}
        </Button>
      )}

      {/* ─── Order Modal ─── */}
      {open && typeof document !== 'undefined' && createPortal(
        <div className="fixed inset-0 z-[200]" onClick={(e) => { e.stopPropagation() }}>
          <motion.div
            className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => { setOpen(false); resetModal() }}
          />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <motion.div
              className="relative z-10 w-full max-w-md max-h-[calc(100vh-2rem)] bg-background border border-border rounded-xl shadow-2xl shadow-black/40 overflow-hidden flex flex-col"
              initial={{ scale: 0.95, opacity: 0, y: 16 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.8 }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="px-5 pt-5 pb-3 flex-shrink-0">
                <div className="flex items-center gap-2.5 mb-1">
                  <div className={cn(
                    'w-8 h-8 rounded-lg flex items-center justify-center',
                    config?.direction === 'SELL'
                      ? 'bg-red-500/15 text-red-400'
                      : 'bg-emerald-500/15 text-emerald-400'
                  )}>
                    {config?.direction === 'SELL'
                      ? <ArrowDownRight className="w-4 h-4" />
                      : <ArrowUpRight className="w-4 h-4" />
                    }
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2 className="text-sm font-semibold text-foreground">{t('buyButton.placeOrder')}</h2>
                    <p className="text-xs text-muted-foreground truncate">{config?.title}</p>
                  </div>
                  <button
                    onClick={() => { setOpen(false); resetModal() }}
                    className="text-muted-foreground/60 hover:text-foreground/80 transition-colors p-1"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Signal Badges */}
                <div className="flex items-center gap-1.5 mt-2.5">
                  {config?.positions.map((pos, i) => (
                    <span
                      key={i}
                      className={cn(
                        'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium',
                        pos.side === 'SELL' || pos.outcome?.toLowerCase() === 'no'
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      )}
                    >
                      {pos.outcome || pos.side} @ {pos.price.toFixed(2)}c
                    </span>
                  ))}
                  {config?.confidence != null && (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      <Target className="w-3 h-3" />
                      {formatConfidencePercent(config.confidence)}
                    </span>
                  )}
                </div>
              </div>

              <div className="h-px bg-border/60" />

              {/* Body */}
              <div className="px-5 py-4 space-y-4 overflow-y-auto flex-1 min-h-0">

                {/* ─── Order Type ─── */}
                <div>
                  <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-2 block">
                    {t('buyButton.orderType')}
                  </label>
                  <div className="grid grid-cols-2 gap-1.5 p-1 bg-muted/40 rounded-lg">
                    <button
                      onClick={() => setOrderType('market')}
                      className={cn(
                        'flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium transition-all',
                        orderType === 'market'
                          ? 'bg-background text-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground/80'
                      )}
                    >
                      <Zap className="w-3.5 h-3.5" />
                      {t('buyButton.market')}
                    </button>
                    <button
                      onClick={() => setOrderType('limit')}
                      className={cn(
                        'flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium transition-all',
                        orderType === 'limit'
                          ? 'bg-background text-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground/80'
                      )}
                    >
                      <Target className="w-3.5 h-3.5" />
                      {t('buyButton.limit')}
                    </button>
                  </div>

                  {orderType === 'limit' && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="mt-2.5 relative">
                        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">{t('buyButton.price')}</span>
                        <input
                          type="number"
                          step="0.01"
                          min="0.01"
                          max="0.99"
                          placeholder={config?.positions[0]?.price.toFixed(2)}
                          value={limitPrice}
                          onChange={(e) => setLimitPrice(e.target.value)}
                          className="w-full h-9 pl-14 pr-3 rounded-lg border border-border bg-muted/30 text-sm font-data text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/40 focus:border-emerald-500/40"
                        />
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-muted-foreground">c</span>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* ─── Position Size ─── */}
                <div>
                  <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <DollarSign className="w-3 h-3" />
                    {t('buyButton.positionSize')}
                    {config?.suggestedSizeUsd != null && (
                      <span className="text-emerald-400/70 font-normal normal-case tracking-normal">
                        {t('buyButton.suggested', { amount: formatUsd(config.suggestedSizeUsd) })}
                      </span>
                    )}
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    {SIZE_PRESETS.map((preset) => (
                      <button
                        key={preset}
                        onClick={() => { setSizeUsd(preset); setCustomSize('') }}
                        className={cn(
                          'px-3 py-1.5 rounded-lg text-xs font-medium transition-all border',
                          sizeUsd === preset
                            ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 shadow-sm shadow-emerald-500/10'
                            : 'bg-muted/40 text-muted-foreground border-transparent hover:bg-muted/70 hover:text-foreground/80'
                        )}
                      >
                        ${preset}
                      </button>
                    ))}
                  </div>
                  <div className="mt-2 relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">$</span>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      placeholder={t('buyButton.customAmount')}
                      value={customSize}
                      onChange={(e) => {
                        setCustomSize(e.target.value)
                        const val = parseFloat(e.target.value)
                        if (val > 0) setSizeUsd(val)
                        else setSizeUsd(null)
                      }}
                      className="w-full h-9 pl-7 pr-3 rounded-lg border border-border bg-muted/30 text-sm font-data text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/40 focus:border-emerald-500/40"
                    />
                  </div>
                </div>

                {/* ─── Bot Selection ─── */}
                <div>
                  <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Bot className="w-3 h-3" />
                    {t('buyButton.executeVia')}
                  </label>
                  {enabledTraders.length === 0 ? (
                    <div className="flex items-center gap-2 px-3 py-3 rounded-lg bg-muted/30 border border-border/50">
                      <Bot className="w-4 h-4 text-muted-foreground/40" />
                      <span className="text-xs text-muted-foreground">{t('buyButton.noEnabledBots')}</span>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {enabledTraders.map((trader: Trader) => (
                        <button
                          key={trader.id}
                          onClick={() => setSelectedTraderId(trader.id === selectedTraderId ? null : trader.id)}
                          className={cn(
                            'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all border',
                            selectedTraderId === trader.id
                              ? 'bg-emerald-500/10 border-emerald-500/30 shadow-sm'
                              : 'bg-muted/20 border-transparent hover:bg-muted/40'
                          )}
                        >
                          <div className={cn(
                            'w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all flex-shrink-0',
                            selectedTraderId === trader.id
                              ? 'border-emerald-500 bg-emerald-500'
                              : 'border-border/60'
                          )}>
                            {selectedTraderId === trader.id && (
                              <Check className="w-3 h-3 text-white" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0 text-left">
                            <p className="text-sm font-medium text-foreground truncate">{trader.name}</p>
                          </div>
                          <span className={cn(
                            'inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-medium',
                            trader.mode === 'live'
                              ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                              : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          )}>
                            {trader.mode === 'live' ? t('buyButton.modeLive') : t('buyButton.modeShadow')}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* ─── Trade Summary ─── */}
                {isReady && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="rounded-lg bg-muted/30 border border-border/50 p-3 space-y-2">
                      <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">{t('buyButton.tradeSummary')}</p>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t('buyButton.cost')}</span>
                          <span className="font-data text-foreground">{formatUsd(effectiveSize)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t('buyButton.entry')}</span>
                          <span className="font-data text-foreground">{effectivePrice.toFixed(2)}c</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t('buyButton.shares')}</span>
                          <span className="font-data text-foreground">~{estimatedShares.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t('buyButton.ifYes')}</span>
                          <span className="font-data text-emerald-400">+{formatUsd(estimatedProfit)}</span>
                        </div>
                      </div>
                      {config?.roi != null && (
                        <div className="flex items-center gap-1.5 pt-1 border-t border-border/30">
                          <TrendingUp className="w-3 h-3 text-emerald-400/70" />
                          <span className="text-[11px] text-muted-foreground">{t('buyButton.roiEdge')}</span>
                          <span className="text-[11px] font-data text-emerald-400 ml-auto">{config.roi.toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Footer */}
              <div className="px-5 py-4 border-t border-border/60 bg-muted/20 flex-shrink-0 space-y-3">
                {/* Result Banner (shows after execution) */}
                {result && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn(
                      'flex items-start gap-2 px-3 py-2.5 rounded-lg border',
                      result.success
                        ? 'bg-green-500/10 border-green-500/20'
                        : 'bg-red-500/10 border-red-500/20'
                    )}
                  >
                    {result.success
                      ? <Check className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      : <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />}
                    <div className="flex-1 min-w-0">
                      <p className={cn('text-xs font-medium', result.success ? 'text-green-400' : 'text-red-400')}>
                        {result.success ? t('buyButton.orderPlaced') : t('buyButton.orderFailed')}
                      </p>
                      <p className={cn('text-[11px] mt-0.5', result.success ? 'text-green-400/70' : 'text-red-400/70')}>
                        {result.message}
                      </p>
                    </div>
                  </motion.div>
                )}

                {result ? (
                  <div className="flex gap-2">
                    {result.traderId && (
                      <Button
                        onClick={() => {
                          setOpen(false)
                          setResult(null)
                          resetModal()
                          window.location.hash = `#/traders/${result.traderId}`
                        }}
                        size="sm"
                        variant="outline"
                        className="flex-1 h-9 text-xs"
                      >
                        <Bot className="w-3.5 h-3.5 mr-1.5" />
                        {t('buyButton.goToTrader')}
                      </Button>
                    )}
                    <Button
                      onClick={() => { setOpen(false); setResult(null); resetModal() }}
                      size="sm"
                      className={cn(
                        'flex-1 h-9 text-xs',
                        result.success
                          ? 'bg-green-500/20 hover:bg-green-500/30 text-green-400 border border-green-500/30'
                          : 'bg-muted/40 hover:bg-muted/60 text-foreground border border-border/50'
                      )}
                    >
                      {result.success ? t('buyButton.done') : t('buyButton.close')}
                    </Button>
                  </div>
                ) : confirmStep && selectedTrader?.mode === 'live' ? (
                  <div className="space-y-3">
                    <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg bg-red-500/10 border border-red-500/20">
                      <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-xs font-medium text-red-400">{t('buyButton.liveOrderConfirmation')}</p>
                        <p className="text-[11px] text-red-400/70 mt-0.5">
                          <Trans i18nKey="buyButton.realMoneyWarning" t={t} values={{ amount: formatUsd(effectiveSize), name: selectedTrader.name }} components={{ strong: <strong /> }} />
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => setConfirmStep(false)}
                        size="sm"
                        variant="outline"
                        className="flex-1 h-9 text-xs"
                      >
                        {t('buyButton.cancel')}
                      </Button>
                      <Button
                        onClick={() => mutation.mutate(selectedTraderId!)}
                        disabled={mutation.isPending}
                        size="sm"
                        className="flex-1 h-9 text-xs bg-red-500 hover:bg-red-600 text-white"
                      >
                        {mutation.isPending ? (
                          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                        ) : (
                          <Shield className="w-3.5 h-3.5 mr-1.5" />
                        )}
                        {t('buyButton.confirmLiveOrder')}
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    onClick={handleExecute}
                    disabled={!isReady || mutation.isPending}
                    size="sm"
                    className={cn(
                      'w-full h-10 text-sm font-medium',
                      'bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600',
                      'shadow-lg shadow-green-500/25 text-white',
                      'disabled:opacity-40 disabled:shadow-none',
                    )}
                  >
                    {mutation.isPending ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <ShoppingCart className="w-4 h-4 mr-2" />
                    )}
                    {!selectedTraderId ? t('buyButton.selectABot') : t('buyButton.buyAmount', { amount: formatUsd(effectiveSize) })}
                  </Button>
                )}
              </div>
            </motion.div>
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}
