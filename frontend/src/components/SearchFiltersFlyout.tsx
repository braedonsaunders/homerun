import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import {
  getSettings,
  updateSettings,
} from '../services/api'

export default function SearchFiltersFlyout({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const [searchFiltersForm, setSearchFiltersForm] = useState({
    min_liquidity_hard: 200,
    min_position_size: 25,
    min_absolute_profit: 5,
    min_annualized_roi: 10,
    max_resolution_months: 18,
    max_plausible_roi: 30,
    max_trade_legs: 8,
    negrisk_min_total_yes: 0.95,
    negrisk_warn_total_yes: 0.97,
    negrisk_election_min_total_yes: 0.97,
    negrisk_max_resolution_spread_days: 7,
    settlement_lag_max_days_to_resolution: 14,
    settlement_lag_near_zero: 0.05,
    settlement_lag_near_one: 0.95,
    settlement_lag_min_sum_deviation: 0.03,
    risk_very_short_days: 2,
    risk_short_days: 7,
    risk_long_lockup_days: 180,
    risk_extended_lockup_days: 90,
    risk_low_liquidity: 1000,
    risk_moderate_liquidity: 5000,
    risk_complex_legs: 5,
    risk_multiple_legs: 3,
    btc_eth_pure_arb_max_combined: 0.98,
    btc_eth_dump_hedge_drop_pct: 0.05,
    btc_eth_thin_liquidity_usd: 500,
    miracle_min_no_price: 0.90,
    miracle_max_no_price: 0.995,
    miracle_min_impossibility_score: 0.70,
  })

  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  useEffect(() => {
    if (settings?.search_filters) {
      setSearchFiltersForm(settings.search_filters)
    }
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setSaveMessage({ type: 'success', text: 'Search filters saved' })
      setTimeout(() => setSaveMessage(null), 3000)
    },
    onError: (error: any) => {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to save search filters' })
      setTimeout(() => setSaveMessage(null), 5000)
    }
  })

  const handleSave = () => {
    saveMutation.mutate({ search_filters: searchFiltersForm })
  }

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/40 z-40 transition-opacity"
        onClick={onClose}
      />
      {/* Drawer */}
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-2xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 bg-background border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-orange-500" />
            <h3 className="text-sm font-semibold">Search Filters</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending} className="gap-1 text-[10px] h-auto px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white">
              <Save className="w-3 h-3" /> {saveMutation.isPending ? 'Saving...' : 'Save'}
            </Button>
            <Button
              variant="ghost"
              onClick={onClose}
              className="text-xs h-auto px-2.5 py-1 hover:bg-card"
            >
              <X className="w-3.5 h-3.5 mr-1" />
              Close
            </Button>
          </div>
        </div>

        {/* Floating toast */}
        {saveMessage && (
          <div className={cn(
            "fixed top-4 right-4 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm shadow-lg border backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300",
            saveMessage.type === 'success'
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
              : "bg-red-500/10 text-red-400 border-red-500/20"
          )}>
            {saveMessage.type === 'success' ? (
              <CheckCircle className="w-4 h-4 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 shrink-0" />
            )}
            {saveMessage.text}
          </div>
        )}

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Hard Rejection Filters */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <SlidersHorizontal className="w-3.5 h-3.5 text-red-500" />
              Hard Rejection Filters
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Min Liquidity Hard ($)</Label>
                <Input type="number" value={searchFiltersForm.min_liquidity_hard} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_liquidity_hard: parseFloat(e.target.value) || 0 }))} min={0} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Hard reject below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Min Position Size ($)</Label>
                <Input type="number" value={searchFiltersForm.min_position_size} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_position_size: parseFloat(e.target.value) || 0 }))} min={0} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if max position below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Min Absolute Profit ($)</Label>
                <Input type="number" value={searchFiltersForm.min_absolute_profit} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_absolute_profit: parseFloat(e.target.value) || 0 }))} min={0} step="0.5" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if net profit below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Min Annualized ROI (%)</Label>
                <Input type="number" value={searchFiltersForm.min_annualized_roi} onChange={(e) => setSearchFiltersForm(p => ({ ...p, min_annualized_roi: parseFloat(e.target.value) || 0 }))} min={0} step="1" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if annualized ROI below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Max Resolution (months)</Label>
                <Input type="number" value={searchFiltersForm.max_resolution_months} onChange={(e) => setSearchFiltersForm(p => ({ ...p, max_resolution_months: parseInt(e.target.value) || 18 }))} min={1} max={120} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Reject if too far out</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Max Plausible ROI (%)</Label>
                <Input type="number" value={searchFiltersForm.max_plausible_roi} onChange={(e) => setSearchFiltersForm(p => ({ ...p, max_plausible_roi: parseFloat(e.target.value) || 30 }))} min={1} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Above this = false positive</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Max Trade Legs</Label>
                <Input type="number" value={searchFiltersForm.max_trade_legs} onChange={(e) => setSearchFiltersForm(p => ({ ...p, max_trade_legs: parseInt(e.target.value) || 8 }))} min={2} max={20} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Max legs in multi-leg trade</p>
              </div>
            </div>
          </Card>

          {/* NegRisk Exhaustivity */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-500" />
              NegRisk Exhaustivity
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Min Total YES</Label>
                <Input type="number" value={searchFiltersForm.negrisk_min_total_yes} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_min_total_yes: parseFloat(e.target.value) || 0.95 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Hard reject below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Warn Total YES</Label>
                <Input type="number" value={searchFiltersForm.negrisk_warn_total_yes} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_warn_total_yes: parseFloat(e.target.value) || 0.97 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Warn below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Election Min Total YES</Label>
                <Input type="number" value={searchFiltersForm.negrisk_election_min_total_yes} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_election_min_total_yes: parseFloat(e.target.value) || 0.97 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Stricter for elections</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Max Resolution Spread (days)</Label>
                <Input type="number" value={searchFiltersForm.negrisk_max_resolution_spread_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, negrisk_max_resolution_spread_days: parseInt(e.target.value) || 7 }))} min={0} max={365} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Max date spread in bundle</p>
              </div>
            </div>
          </Card>

          {/* Settlement Lag */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <SlidersHorizontal className="w-3.5 h-3.5 text-purple-500" />
              Settlement Lag
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Max Days to Resolution</Label>
                <Input type="number" value={searchFiltersForm.settlement_lag_max_days_to_resolution} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_max_days_to_resolution: parseInt(e.target.value) || 14 }))} min={0} max={365} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Detection window</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Near-Zero Threshold</Label>
                <Input type="number" value={searchFiltersForm.settlement_lag_near_zero} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_near_zero: parseFloat(e.target.value) || 0.05 }))} min={0.001} max={0.5} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Below = resolved NO</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Near-One Threshold</Label>
                <Input type="number" value={searchFiltersForm.settlement_lag_near_one} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_near_one: parseFloat(e.target.value) || 0.95 }))} min={0.5} max={0.999} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Above = resolved YES</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Min Sum Deviation</Label>
                <Input type="number" value={searchFiltersForm.settlement_lag_min_sum_deviation} onChange={(e) => setSearchFiltersForm(p => ({ ...p, settlement_lag_min_sum_deviation: parseFloat(e.target.value) || 0.03 }))} min={0.001} max={0.5} step="0.005" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Min deviation from 1.0</p>
              </div>
            </div>
          </Card>

          {/* Risk Scoring */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <SlidersHorizontal className="w-3.5 h-3.5 text-orange-500" />
              Risk Scoring Thresholds
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Very Short Days</Label>
                <Input type="number" value={searchFiltersForm.risk_very_short_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_very_short_days: parseInt(e.target.value) || 2 }))} min={0} max={30} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">High risk if &lt; this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Short Days</Label>
                <Input type="number" value={searchFiltersForm.risk_short_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_short_days: parseInt(e.target.value) || 7 }))} min={1} max={60} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Moderate risk if &lt; this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Long Lockup Days</Label>
                <Input type="number" value={searchFiltersForm.risk_long_lockup_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_long_lockup_days: parseInt(e.target.value) || 180 }))} min={30} max={3650} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">High risk if &gt; this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Extended Lockup Days</Label>
                <Input type="number" value={searchFiltersForm.risk_extended_lockup_days} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_extended_lockup_days: parseInt(e.target.value) || 90 }))} min={14} max={1825} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Moderate risk if &gt; this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Low Liquidity ($)</Label>
                <Input type="number" value={searchFiltersForm.risk_low_liquidity} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_low_liquidity: parseFloat(e.target.value) || 1000 }))} min={0} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">High risk below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Moderate Liquidity ($)</Label>
                <Input type="number" value={searchFiltersForm.risk_moderate_liquidity} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_moderate_liquidity: parseFloat(e.target.value) || 5000 }))} min={0} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Moderate risk below this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Complex Legs Threshold</Label>
                <Input type="number" value={searchFiltersForm.risk_complex_legs} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_complex_legs: parseInt(e.target.value) || 5 }))} min={2} max={20} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Above = complex trade risk</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Multiple Legs Threshold</Label>
                <Input type="number" value={searchFiltersForm.risk_multiple_legs} onChange={(e) => setSearchFiltersForm(p => ({ ...p, risk_multiple_legs: parseInt(e.target.value) || 3 }))} min={2} max={20} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Above = multi-position risk</p>
              </div>
            </div>
          </Card>

          {/* BTC/ETH High-Frequency */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <SlidersHorizontal className="w-3.5 h-3.5 text-yellow-500" />
              BTC/ETH High-Frequency
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Pure Arb Max Combined</Label>
                <Input type="number" value={searchFiltersForm.btc_eth_pure_arb_max_combined} onChange={(e) => setSearchFiltersForm(p => ({ ...p, btc_eth_pure_arb_max_combined: parseFloat(e.target.value) || 0.98 }))} min={0.5} max={1} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Pure arb when YES+NO &lt; this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Dump-Hedge Drop %</Label>
                <Input type="number" value={searchFiltersForm.btc_eth_dump_hedge_drop_pct} onChange={(e) => setSearchFiltersForm(p => ({ ...p, btc_eth_dump_hedge_drop_pct: parseFloat(e.target.value) || 0.05 }))} min={0.01} max={0.5} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Min drop to trigger hedge</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Thin Liquidity ($)</Label>
                <Input type="number" value={searchFiltersForm.btc_eth_thin_liquidity_usd} onChange={(e) => setSearchFiltersForm(p => ({ ...p, btc_eth_thin_liquidity_usd: parseFloat(e.target.value) || 500 }))} min={0} className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Below = thin order book</p>
              </div>
            </div>
          </Card>

          {/* Miracle Strategy */}
          <Card className="bg-card/40 border-border/40 rounded-xl shadow-none p-3">
            <h4 className="text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 mb-3">
              <SlidersHorizontal className="w-3.5 h-3.5 text-emerald-500" />
              Miracle Strategy
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Min NO Price</Label>
                <Input type="number" value={searchFiltersForm.miracle_min_no_price} onChange={(e) => setSearchFiltersForm(p => ({ ...p, miracle_min_no_price: parseFloat(e.target.value) || 0.90 }))} min={0.5} max={0.999} step="0.01" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Only consider NO &ge; this</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Max NO Price</Label>
                <Input type="number" value={searchFiltersForm.miracle_max_no_price} onChange={(e) => setSearchFiltersForm(p => ({ ...p, miracle_max_no_price: parseFloat(e.target.value) || 0.995 }))} min={0.9} max={1} step="0.005" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Skip if NO already at this+</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Min Impossibility Score</Label>
                <Input type="number" value={searchFiltersForm.miracle_min_impossibility_score} onChange={(e) => setSearchFiltersForm(p => ({ ...p, miracle_min_impossibility_score: parseFloat(e.target.value) || 0.70 }))} min={0} max={1} step="0.05" className="mt-1 text-sm" />
                <p className="text-[11px] text-muted-foreground/70 mt-1">Min confidence event is impossible</p>
              </div>
            </div>
          </Card>

          {/* Bottom Save */}
          <div className="flex items-center gap-2 pt-2 pb-4">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending}>
              <Save className="w-3.5 h-3.5 mr-1.5" />
              {saveMutation.isPending ? 'Saving...' : 'Save All Filters'}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}
