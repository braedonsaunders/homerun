import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SlidersHorizontal,
  Save,
  X,
  CheckCircle,
  AlertCircle,
  Shield,
  Zap,
  TrendingUp,
  BarChart3,
  GitBranch,
  Clock,
  Globe,
  Brain,
  Droplets,
  Activity,
  Target,
  LineChart,
  BarChart,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Switch } from './ui/switch'
import {
  getSettings,
  updateSettings,
} from '../services/api'

// ==================== TYPES ====================

const DEFAULTS = {
  // Global
  min_liquidity_hard: 200,
  min_position_size: 25,
  min_absolute_profit: 5,
  min_annualized_roi: 10,
  max_resolution_months: 18,
  max_plausible_roi: 30,
  max_trade_legs: 8,
  // Risk scoring
  risk_very_short_days: 2,
  risk_short_days: 7,
  risk_long_lockup_days: 180,
  risk_extended_lockup_days: 90,
  risk_low_liquidity: 1000,
  risk_moderate_liquidity: 5000,
  risk_complex_legs: 5,
  risk_multiple_legs: 3,
  // NegRisk
  negrisk_min_total_yes: 0.95,
  negrisk_warn_total_yes: 0.97,
  negrisk_election_min_total_yes: 0.97,
  negrisk_max_resolution_spread_days: 7,
  // Settlement lag
  settlement_lag_max_days_to_resolution: 14,
  settlement_lag_near_zero: 0.05,
  settlement_lag_near_one: 0.95,
  settlement_lag_min_sum_deviation: 0.03,
  // BTC/ETH
  btc_eth_hf_enabled: true,
  btc_eth_pure_arb_max_combined: 0.98,
  btc_eth_dump_hedge_drop_pct: 0.05,
  btc_eth_thin_liquidity_usd: 500,
  // Miracle
  miracle_min_no_price: 0.90,
  miracle_max_no_price: 0.995,
  miracle_min_impossibility_score: 0.70,
  // Cross-platform
  cross_platform_enabled: true,
  // Combinatorial
  combinatorial_min_confidence: 0.75,
  combinatorial_high_confidence: 0.90,
  // Bayesian cascade
  bayesian_cascade_enabled: true,
  bayesian_min_edge_percent: 5.0,
  bayesian_propagation_depth: 3,
  // Liquidity vacuum
  liquidity_vacuum_enabled: true,
  liquidity_vacuum_min_imbalance_ratio: 5.0,
  liquidity_vacuum_min_depth_usd: 100.0,
  // Entropy arb
  entropy_arb_enabled: true,
  entropy_arb_min_deviation: 0.25,
  // Event-driven
  event_driven_enabled: true,
  // Temporal decay
  temporal_decay_enabled: true,
  // Correlation arb
  correlation_arb_enabled: true,
  correlation_arb_min_correlation: 0.7,
  correlation_arb_min_divergence: 0.05,
  // Market making
  market_making_enabled: true,
  market_making_spread_bps: 100.0,
  market_making_max_inventory_usd: 500.0,
  // Stat arb
  stat_arb_enabled: true,
  stat_arb_min_edge: 0.05,
}

// ==================== HELPER COMPONENTS ====================

function NumericField({
  label,
  help,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string
  help: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
}) {
  return (
    <div className={cn(disabled && 'opacity-40 pointer-events-none')}>
      <Label className="text-[11px] text-muted-foreground leading-tight">{label}</Label>
      <Input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="mt-0.5 text-xs h-7"
      />
      <p className="text-[10px] text-muted-foreground/60 mt-0.5 leading-tight">{help}</p>
    </div>
  )
}

function StrategyToggle({
  label,
  enabled,
  onToggle,
  icon: Icon,
  color,
  badge,
}: {
  label: string
  enabled: boolean
  onToggle: (v: boolean) => void
  icon: any
  color: string
  badge?: string
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-1.5">
        <Icon className={cn('w-3.5 h-3.5', color)} />
        <span className="text-xs font-medium">{label}</span>
        {badge && (
          <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
            {badge}
          </span>
        )}
      </div>
      <Switch
        checked={enabled}
        onCheckedChange={onToggle}
        className="scale-75"
      />
    </div>
  )
}

function CollapsibleSection({
  title,
  icon: Icon,
  color,
  children,
  defaultOpen = false,
  count,
}: {
  title: string
  icon: any
  color: string
  children: React.ReactNode
  defaultOpen?: boolean
  count?: number
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <Card className="bg-card/40 border-border/40 rounded-xl shadow-none overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <Icon className={cn('w-3.5 h-3.5', color)} />
          <h4 className="text-[10px] uppercase tracking-widest font-semibold">{title}</h4>
          {count !== undefined && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/60 text-muted-foreground">
              {count}
            </span>
          )}
        </div>
        {open ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />}
      </button>
      {open && <div className="px-3 pb-3 space-y-3">{children}</div>}
    </Card>
  )
}

// ==================== MAIN COMPONENT ====================

export default function SearchFiltersFlyout({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [form, setForm] = useState(DEFAULTS)

  const queryClient = useQueryClient()

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  useEffect(() => {
    if (settings?.search_filters) {
      setForm((prev) => ({ ...prev, ...settings.search_filters }))
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
    saveMutation.mutate({ search_filters: form })
  }

  const set = <K extends keyof typeof DEFAULTS>(key: K, val: (typeof DEFAULTS)[K]) =>
    setForm((p) => ({ ...p, [key]: val }))

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-background/80 z-40 transition-opacity"
        onClick={onClose}
      />
      {/* Drawer */}
      <div className="fixed top-0 right-0 bottom-0 w-full max-w-3xl z-50 bg-background border-l border-border/40 shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-2.5 bg-background/95 backdrop-blur-sm border-b border-border/40">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-orange-500" />
            <h3 className="text-sm font-semibold">Opportunity Filters</h3>
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">18 strategies</span>
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
        <div className="p-3 space-y-2">

          {/* ============================================================ */}
          {/* SECTION 1: GLOBAL FILTERS */}
          {/* ============================================================ */}
          <CollapsibleSection title="Global Rejection Filters" icon={Shield} color="text-red-500" defaultOpen={true} count={7}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Hard rejection thresholds applied to every opportunity regardless of strategy.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <NumericField label="Min Liquidity ($)" help="Hard reject below this" value={form.min_liquidity_hard} onChange={(v) => set('min_liquidity_hard', v)} min={0} />
              <NumericField label="Min Position Size ($)" help="Reject if max position below" value={form.min_position_size} onChange={(v) => set('min_position_size', v)} min={0} />
              <NumericField label="Min Absolute Profit ($)" help="Reject if net profit below" value={form.min_absolute_profit} onChange={(v) => set('min_absolute_profit', v)} min={0} step={0.5} />
              <NumericField label="Min Annualized ROI (%)" help="Reject if ROI below" value={form.min_annualized_roi} onChange={(v) => set('min_annualized_roi', v)} min={0} step={1} />
              <NumericField label="Max Resolution (months)" help="Reject if too far out" value={form.max_resolution_months} onChange={(v) => set('max_resolution_months', v)} min={1} max={120} />
              <NumericField label="Max Plausible ROI (%)" help="Above = false positive" value={form.max_plausible_roi} onChange={(v) => set('max_plausible_roi', v)} min={1} />
              <NumericField label="Max Trade Legs" help="Max legs in multi-leg trade" value={form.max_trade_legs} onChange={(v) => set('max_trade_legs', v)} min={2} max={20} />
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 2: RISK SCORING */}
          {/* ============================================================ */}
          <CollapsibleSection title="Risk Scoring Thresholds" icon={BarChart3} color="text-orange-500" count={8}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Cross-cutting risk assessment thresholds that affect the risk score of every opportunity.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <NumericField label="Very Short Days" help="High risk if < this" value={form.risk_very_short_days} onChange={(v) => set('risk_very_short_days', v)} min={0} max={30} />
              <NumericField label="Short Days" help="Moderate risk if < this" value={form.risk_short_days} onChange={(v) => set('risk_short_days', v)} min={1} max={60} />
              <NumericField label="Long Lockup Days" help="High risk if > this" value={form.risk_long_lockup_days} onChange={(v) => set('risk_long_lockup_days', v)} min={30} max={3650} />
              <NumericField label="Extended Lockup Days" help="Moderate risk if > this" value={form.risk_extended_lockup_days} onChange={(v) => set('risk_extended_lockup_days', v)} min={14} max={1825} />
              <NumericField label="Low Liquidity ($)" help="High risk below this" value={form.risk_low_liquidity} onChange={(v) => set('risk_low_liquidity', v)} min={0} />
              <NumericField label="Moderate Liquidity ($)" help="Moderate risk below this" value={form.risk_moderate_liquidity} onChange={(v) => set('risk_moderate_liquidity', v)} min={0} />
              <NumericField label="Complex Legs" help="Above = complex trade risk" value={form.risk_complex_legs} onChange={(v) => set('risk_complex_legs', v)} min={2} max={20} />
              <NumericField label="Multiple Legs" help="Above = multi-position risk" value={form.risk_multiple_legs} onChange={(v) => set('risk_multiple_legs', v)} min={2} max={20} />
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 3: STRUCTURAL ARBITRAGE (Methods 1-5) */}
          {/* ============================================================ */}
          <CollapsibleSection title="Structural Arbitrage" icon={GitBranch} color="text-cyan-500" count={5}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Risk-free or near-risk-free strategies exploiting structural market mispricings.
            </p>

            {/* Method 1: Basic */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-emerald-500/20 flex items-center justify-center text-[8px] font-bold text-emerald-400">1</div>
                <span className="text-[11px] font-medium">Basic Arbitrage</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400">risk-free</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Buy YES + NO on same binary market when total cost &lt; $1.00. Uses global filters only.
              </p>
            </div>

            {/* Method 2: Mutually Exclusive */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-yellow-500/20 flex items-center justify-center text-[8px] font-bold text-yellow-400">2</div>
                <span className="text-[11px] font-medium">Mutually Exclusive</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-yellow-500/10 text-yellow-400">verify</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Two markets with opposite outcomes; buy YES on both for &lt; $1. Uses NegRisk thresholds below.
              </p>
            </div>

            {/* Method 3: Contradiction */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-yellow-500/20 flex items-center justify-center text-[8px] font-bold text-yellow-400">3</div>
                <span className="text-[11px] font-medium">Contradiction</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-yellow-500/10 text-yellow-400">verify</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Two markets saying opposite things (before/after, win/lose). Uses global filters only.
              </p>
            </div>

            {/* Method 4: NegRisk / One-of-Many */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-emerald-500/20 flex items-center justify-center text-[8px] font-bold text-emerald-400">4</div>
                <span className="text-[11px] font-medium">NegRisk / One-of-Many</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400">risk-free</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Buy YES on all outcomes in exhaustive events where exactly one must win.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                <NumericField label="Min Total YES" help="Hard reject below" value={form.negrisk_min_total_yes} onChange={(v) => set('negrisk_min_total_yes', v)} min={0.5} max={1} step={0.01} />
                <NumericField label="Warn Total YES" help="Warn below this" value={form.negrisk_warn_total_yes} onChange={(v) => set('negrisk_warn_total_yes', v)} min={0.5} max={1} step={0.01} />
                <NumericField label="Election Min YES" help="Stricter for elections" value={form.negrisk_election_min_total_yes} onChange={(v) => set('negrisk_election_min_total_yes', v)} min={0.5} max={1} step={0.01} />
                <NumericField label="Max Date Spread (days)" help="Max resolution spread" value={form.negrisk_max_resolution_spread_days} onChange={(v) => set('negrisk_max_resolution_spread_days', v)} min={0} max={365} />
              </div>
            </div>

            {/* Method 5: Must-Happen */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-yellow-500/20 flex items-center justify-center text-[8px] font-bold text-yellow-400">5</div>
                <span className="text-[11px] font-medium">Must-Happen</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-yellow-500/10 text-yellow-400">verify</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Buy YES on all outcomes when one must happen. Uses same NegRisk thresholds above.
              </p>
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 4: SETTLEMENT & TIMING (Methods 8, 15) */}
          {/* ============================================================ */}
          <CollapsibleSection title="Settlement & Timing" icon={Clock} color="text-purple-500" count={2}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Strategies exploiting time-related mispricings and settlement delays.
            </p>

            {/* Method 8: Settlement Lag */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-emerald-500/20 flex items-center justify-center text-[8px] font-bold text-emerald-400">8</div>
                <span className="text-[11px] font-medium">Settlement Lag</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400">risk-free</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Exploit delayed price updates after outcome has been determined but prices haven't locked.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                <NumericField label="Max Days to Resolution" help="Detection window" value={form.settlement_lag_max_days_to_resolution} onChange={(v) => set('settlement_lag_max_days_to_resolution', v)} min={0} max={365} />
                <NumericField label="Near-Zero Threshold" help="Below = resolved NO" value={form.settlement_lag_near_zero} onChange={(v) => set('settlement_lag_near_zero', v)} min={0.001} max={0.5} step={0.01} />
                <NumericField label="Near-One Threshold" help="Above = resolved YES" value={form.settlement_lag_near_one} onChange={(v) => set('settlement_lag_near_one', v)} min={0.5} max={0.999} step={0.01} />
                <NumericField label="Min Sum Deviation" help="Min deviation from 1.0" value={form.settlement_lag_min_sum_deviation} onChange={(v) => set('settlement_lag_min_sum_deviation', v)} min={0.001} max={0.5} step={0.005} />
              </div>
            </div>

            {/* Method 15: Temporal Decay */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Temporal Decay"
                enabled={form.temporal_decay_enabled}
                onToggle={(v) => set('temporal_decay_enabled', v)}
                icon={Clock}
                color="text-purple-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Time-decay mispricing in deadline markets. Like options theta — prices should decay as deadlines pass without event occurring.
              </p>
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 5: CRYPTO HIGH-FREQUENCY (Method 9) */}
          {/* ============================================================ */}
          <CollapsibleSection title="Crypto High-Frequency" icon={Zap} color="text-yellow-500" count={1}>
            {/* Method 9: BTC/ETH HighFreq */}
            <div className="space-y-2">
              <StrategyToggle
                label="BTC/ETH High-Frequency"
                enabled={form.btc_eth_hf_enabled}
                onToggle={(v) => set('btc_eth_hf_enabled', v)}
                icon={Zap}
                color="text-yellow-400"
                badge="hybrid"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Dynamic sub-strategy selection for crypto 15-min/1-hr binary markets (BTC, ETH, SOL, XRP). Includes pure arb, dump-hedge, and pre-placed limits.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
                <NumericField label="Pure Arb Max Combined" help="Pure arb when YES+NO < this" value={form.btc_eth_pure_arb_max_combined} onChange={(v) => set('btc_eth_pure_arb_max_combined', v)} min={0.5} max={1} step={0.01} disabled={!form.btc_eth_hf_enabled} />
                <NumericField label="Dump-Hedge Drop %" help="Min drop to trigger hedge" value={form.btc_eth_dump_hedge_drop_pct} onChange={(v) => set('btc_eth_dump_hedge_drop_pct', v)} min={0.01} max={0.5} step={0.01} disabled={!form.btc_eth_hf_enabled} />
                <NumericField label="Thin Liquidity ($)" help="Below = thin order book" value={form.btc_eth_thin_liquidity_usd} onChange={(v) => set('btc_eth_thin_liquidity_usd', v)} min={0} disabled={!form.btc_eth_hf_enabled} />
              </div>
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 6: MIRACLE STRATEGY (Method 6) */}
          {/* ============================================================ */}
          <CollapsibleSection title="Miracle Strategy" icon={Target} color="text-emerald-500" count={1}>
            {/* Method 6: Miracle */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-red-500/20 flex items-center justify-center text-[8px] font-bold text-red-400">6</div>
                <span className="text-[11px] font-medium">Miracle Market Scanner</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-red-500/10 text-red-400">edge-based</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Bet NO on impossible/absurd events with high NO prices. Uses weighted keyword matching and impossibility scoring.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
                <NumericField label="Min NO Price" help="Only consider NO >= this" value={form.miracle_min_no_price} onChange={(v) => set('miracle_min_no_price', v)} min={0.5} max={0.999} step={0.01} />
                <NumericField label="Max NO Price" help="Skip if NO at this+" value={form.miracle_max_no_price} onChange={(v) => set('miracle_max_no_price', v)} min={0.9} max={1} step={0.005} />
                <NumericField label="Min Impossibility Score" help="Min confidence impossible" value={form.miracle_min_impossibility_score} onChange={(v) => set('miracle_min_impossibility_score', v)} min={0} max={1} step={0.05} />
              </div>
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 7: CROSS-MARKET ARBITRAGE (Methods 7, 10, 11, 14) */}
          {/* ============================================================ */}
          <CollapsibleSection title="Cross-Market Arbitrage" icon={Globe} color="text-blue-500" count={4}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Strategies that exploit price differences across related or dependent markets.
            </p>

            {/* Method 7: Combinatorial */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-full bg-yellow-500/20 flex items-center justify-center text-[8px] font-bold text-yellow-400">7</div>
                <span className="text-[11px] font-medium">Combinatorial Arbitrage</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-yellow-500/10 text-yellow-400">LLM-dependent</span>
              </div>
              <p className="text-[10px] text-muted-foreground/60">
                Integer programming on multi-dependent markets. Uses constraint solver + LLM for dependency detection.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min LLM Confidence" help="Min confidence for trades" value={form.combinatorial_min_confidence} onChange={(v) => set('combinatorial_min_confidence', v)} min={0} max={1} step={0.05} />
                <NumericField label="High Confidence" help="High confidence threshold" value={form.combinatorial_high_confidence} onChange={(v) => set('combinatorial_high_confidence', v)} min={0} max={1} step={0.05} />
              </div>
            </div>

            {/* Method 10: Cross-Platform */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Cross-Platform Arbitrage"
                enabled={form.cross_platform_enabled}
                onToggle={(v) => set('cross_platform_enabled', v)}
                icon={Globe}
                color="text-blue-400"
                badge="risk-free"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Fuzzy-match same events across Polymarket and Kalshi, exploit price divergence. 9% combined fee floor.
              </p>
            </div>

            {/* Method 11: Bayesian Cascade */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Bayesian Cascade"
                enabled={form.bayesian_cascade_enabled}
                onToggle={(v) => set('bayesian_cascade_enabled', v)}
                icon={Brain}
                color="text-violet-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                DAG probability graph propagation — find markets not yet adjusted to related market moves.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min Edge (%)" help="Min expected-vs-actual diff" value={form.bayesian_min_edge_percent} onChange={(v) => set('bayesian_min_edge_percent', v)} min={0} max={100} step={0.5} disabled={!form.bayesian_cascade_enabled} />
                <NumericField label="Propagation Depth" help="Max hops in graph" value={form.bayesian_propagation_depth} onChange={(v) => set('bayesian_propagation_depth', v)} min={1} max={10} disabled={!form.bayesian_cascade_enabled} />
              </div>
            </div>

            {/* Method 14: Event-Driven */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Event-Driven Arbitrage"
                enabled={form.event_driven_enabled}
                onToggle={(v) => set('event_driven_enabled', v)}
                icon={Activity}
                color="text-orange-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Find lagging markets after catalyst moves. Exploits information propagation delays between related markets.
              </p>
            </div>
          </CollapsibleSection>

          {/* ============================================================ */}
          {/* SECTION 8: QUANTITATIVE STRATEGIES (Methods 12, 13, 16, 17, 18) */}
          {/* ============================================================ */}
          <CollapsibleSection title="Quantitative Strategies" icon={TrendingUp} color="text-pink-500" count={5}>
            <p className="text-[10px] text-muted-foreground/60 -mt-1">
              Data-driven strategies using statistical analysis, entropy, correlations, and market microstructure.
            </p>

            {/* Method 12: Liquidity Vacuum */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Liquidity Vacuum"
                enabled={form.liquidity_vacuum_enabled}
                onToggle={(v) => set('liquidity_vacuum_enabled', v)}
                icon={Droplets}
                color="text-sky-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Order book imbalance exploitation — buy the thin side before mean reversion.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min Imbalance Ratio" help="Min bid/ask depth ratio" value={form.liquidity_vacuum_min_imbalance_ratio} onChange={(v) => set('liquidity_vacuum_min_imbalance_ratio', v)} min={1} max={100} step={0.5} disabled={!form.liquidity_vacuum_enabled} />
                <NumericField label="Min Depth ($)" help="Min order book depth" value={form.liquidity_vacuum_min_depth_usd} onChange={(v) => set('liquidity_vacuum_min_depth_usd', v)} min={0} disabled={!form.liquidity_vacuum_enabled} />
              </div>
            </div>

            {/* Method 13: Entropy Arb */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Entropy Arbitrage"
                enabled={form.entropy_arb_enabled}
                onToggle={(v) => set('entropy_arb_enabled', v)}
                icon={BarChart}
                color="text-fuchsia-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Information-theoretic mispricing using Shannon entropy. Detects markets with anomalously high entropy near resolution.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min Deviation" help="Min entropy deviation from expected" value={form.entropy_arb_min_deviation} onChange={(v) => set('entropy_arb_min_deviation', v)} min={0} max={2} step={0.05} disabled={!form.entropy_arb_enabled} />
              </div>
            </div>

            {/* Method 16: Correlation Arb */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Correlation Arbitrage"
                enabled={form.correlation_arb_enabled}
                onToggle={(v) => set('correlation_arb_enabled', v)}
                icon={LineChart}
                color="text-teal-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Mean-reversion on correlated pair spreads. Detects historically correlated markets that have diverged.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min Correlation" help="Min correlation coefficient" value={form.correlation_arb_min_correlation} onChange={(v) => set('correlation_arb_min_correlation', v)} min={0} max={1} step={0.05} disabled={!form.correlation_arb_enabled} />
                <NumericField label="Min Divergence" help="Min price divergence" value={form.correlation_arb_min_divergence} onChange={(v) => set('correlation_arb_min_divergence', v)} min={0} max={1} step={0.01} disabled={!form.correlation_arb_enabled} />
              </div>
            </div>

            {/* Method 17: Market Making */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Market Making"
                enabled={form.market_making_enabled}
                onToggle={(v) => set('market_making_enabled', v)}
                icon={BarChart3}
                color="text-amber-400"
                badge="spread-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Provide liquidity on both sides to earn bid-ask spread. Makers pay 0% fees on Polymarket.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min Spread (bps)" help="Min bid-ask spread in basis points" value={form.market_making_spread_bps} onChange={(v) => set('market_making_spread_bps', v)} min={10} max={1000} step={10} disabled={!form.market_making_enabled} />
                <NumericField label="Max Inventory ($)" help="Max inventory per market" value={form.market_making_max_inventory_usd} onChange={(v) => set('market_making_max_inventory_usd', v)} min={0} disabled={!form.market_making_enabled} />
              </div>
            </div>

            {/* Method 18: Statistical Arb */}
            <div className="rounded-lg bg-muted/20 p-2.5 space-y-2">
              <StrategyToggle
                label="Statistical Arbitrage"
                enabled={form.stat_arb_enabled}
                onToggle={(v) => set('stat_arb_enabled', v)}
                icon={TrendingUp}
                color="text-rose-400"
                badge="edge-based"
              />
              <p className="text-[10px] text-muted-foreground/60">
                Ensemble weak signals (anchoring, base rates, consensus, momentum, volume-price) into composite fair probability vs market price.
              </p>
              <div className="grid grid-cols-2 gap-2.5">
                <NumericField label="Min Edge" help="Min fair-value edge to trade" value={form.stat_arb_min_edge} onChange={(v) => set('stat_arb_min_edge', v)} min={0} max={1} step={0.01} disabled={!form.stat_arb_enabled} />
              </div>
            </div>
          </CollapsibleSection>

          {/* Bottom Save */}
          <div className="flex items-center gap-2 pt-1 pb-4">
            <Button size="sm" onClick={handleSave} disabled={saveMutation.isPending} className="gap-1.5">
              <Save className="w-3.5 h-3.5" />
              {saveMutation.isPending ? 'Saving...' : 'Save All Filters'}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}
