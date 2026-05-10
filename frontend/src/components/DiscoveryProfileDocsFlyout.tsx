import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Code2,
  Zap,
  Copy,
  Check,
  Rocket,
  Shield,
  Database,
  Users,
  Package,
  Target,
  Layers,
} from 'lucide-react'
import { ScrollArea } from './ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { cn } from '../lib/utils'

// ==================== TYPES ====================

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
}

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

function FieldTable({ fields }: { fields: [string, string][] }) {
  return (
    <div className="space-y-0.5">
      {fields.map(([key, desc]) => (
        <div key={key} className="flex gap-2 text-[11px] py-0.5">
          <code className="text-amber-400 font-mono shrink-0">{key}</code>
          <span className="text-muted-foreground">{desc}</span>
        </div>
      ))}
    </div>
  )
}

// ==================== MAIN COMPONENT ====================

export default function DiscoveryProfileDocsFlyout({ open, onOpenChange }: Props) {
  const { t } = useTranslation()
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[640px] sm:max-w-[640px] p-0 border-l border-border/50">
        <SheetHeader className="px-4 py-3 border-b border-border/30">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-purple-400" />
            <SheetTitle className="text-sm">{t('discoveryProfileDocsFlyout.title')}</SheetTitle>
          </div>
          <SheetDescription className="text-[11px] text-muted-foreground">
            {t('discoveryProfileDocsFlyout.headerDescription')}
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-80px)]">
          <div className="p-4 space-y-2">

            {/* ==================== 1. QUICK START ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.quickStart')} icon={Rocket} iconColor="text-emerald-400" defaultOpen>
              <p className="text-[11px] text-muted-foreground pt-1">
                {t('discoveryProfileDocsFlyout.quickStart.intro')}
              </p>

              <div className="space-y-2 pt-1">
                <div className="flex items-start gap-2">
                  <span className="text-[10px] font-bold text-emerald-400 mt-0.5 shrink-0">1.</span>
                  <div className="space-y-1">
                    <p className="text-[11px] text-foreground font-medium">{t('discoveryProfileDocsFlyout.quickStart.step1Title')}</p>
                    <CodeBlock code={`from services.discovery_profile_sdk import BaseDiscoveryProfile

class MyProfile(BaseDiscoveryProfile):
    name = "My Custom Profile"
    description = "Scores wallets with custom logic"
    default_config = {"min_trades": 10, "boost_recent": True}`} />
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <span className="text-[10px] font-bold text-emerald-400 mt-0.5 shrink-0">2.</span>
                  <div className="space-y-1">
                    <p className="text-[11px] text-foreground font-medium">{t('discoveryProfileDocsFlyout.quickStart.step2Title')}</p>
                    <CodeBlock code={`def score_wallet(self, wallet, trades, rolling):
    base = self._calculate_rank_score(wallet, trades)
    # Boost wallets with recent high win rate
    if rolling.get("rolling_win_rate", 0) > 0.65:
        base["rank_score"] = min(1.0, base["rank_score"] * 1.2)
    return base`} />
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <span className="text-[10px] font-bold text-emerald-400 mt-0.5 shrink-0">3.</span>
                  <div className="space-y-1">
                    <p className="text-[11px] text-foreground font-medium">{t('discoveryProfileDocsFlyout.quickStart.step3Title')}</p>
                    <p className="text-[11px] text-muted-foreground">
                      {t('discoveryProfileDocsFlyout.quickStart.step3Desc')}
                    </p>
                  </div>
                </div>
              </div>
            </Section>

            {/* ==================== 2. BASE INTERFACE ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.baseInterface')} icon={Code2} iconColor="text-cyan-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                {t('discoveryProfileDocsFlyout.baseInterface.intro')}
              </p>

              <CodeBlock code={`class BaseDiscoveryProfile:
    name: str = "Unnamed Profile"
    description: str = ""
    default_config: dict = {}`} className="mt-1" />

              <div className="space-y-3 pt-2">
                <div>
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.labels.classAttributes')}</p>
                  <FieldTable fields={[
                    ['name', t('discoveryProfileDocsFlyout.baseInterface.attrName')],
                    ['description', t('discoveryProfileDocsFlyout.baseInterface.attrDescription')],
                    ['default_config', t('discoveryProfileDocsFlyout.baseInterface.attrDefaultConfig')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">score_wallet(wallet, trades, rolling) → dict</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.baseInterface.scoreWalletDesc')}
                  </p>
                  <CodeBlock code={`# Return type:
{
    "rank_score": float,       # 0.0-1.0, overall composite score
    "quality_score": float,    # 0.0-1.0, trade quality metric
    "insider_score": float,    # 0.0-1.0, insider detection score
    "tags": list[str],         # e.g. ["sniper", "whale", "consistent"]
    "recommendation": str      # "strong_follow" | "follow" | "watch" | "avoid"
}`} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">select_pool(scored_wallets, current_pool) → dict</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.baseInterface.selectPoolDesc')}
                  </p>
                  <CodeBlock code={`# scored_wallets: list of (wallet, scores) tuples
# current_pool: set of addresses currently in pool
#
# Return type:
{
    "members": list[str],    # addresses to include in new pool
    "removals": list[str],   # addresses explicitly removed
    "pool_size": int         # final pool size
}`} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">configure(config) → None</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.baseInterface.configureDesc')}
                  </p>
                  <CodeBlock code={`def configure(self, config):
    # self.config is already merged with default_config
    # Override to validate or transform config values
    if self.config.get("min_trades", 0) < 1:
        self.config["min_trades"] = 1`} />
                </div>
              </div>
            </Section>

            {/* ==================== 3. WALLET DATA CONTRACT ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.walletDataContract')} icon={Database} iconColor="text-amber-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.walletData.intro') }} />
              </p>

              <div className="space-y-3 pt-1">
                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.walletData.identityHeading')}</p>
                  <FieldTable fields={[
                    ['address', t('discoveryProfileDocsFlyout.walletData.fieldAddress')],
                    ['username', t('discoveryProfileDocsFlyout.walletData.fieldUsername')],
                    ['cluster_id', t('discoveryProfileDocsFlyout.walletData.fieldClusterId')],
                    ['tags', t('discoveryProfileDocsFlyout.walletData.fieldTags')],
                    ['is_bot', t('discoveryProfileDocsFlyout.walletData.fieldIsBot')],
                  ]} />
                </div>

                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.walletData.performanceHeading')}</p>
                  <FieldTable fields={[
                    ['total_trades', t('discoveryProfileDocsFlyout.walletData.fieldTotalTrades')],
                    ['wins', t('discoveryProfileDocsFlyout.walletData.fieldWins')],
                    ['losses', t('discoveryProfileDocsFlyout.walletData.fieldLosses')],
                    ['win_rate', t('discoveryProfileDocsFlyout.walletData.fieldWinRate')],
                    ['total_pnl', t('discoveryProfileDocsFlyout.walletData.fieldTotalPnl')],
                    ['realized_pnl', t('discoveryProfileDocsFlyout.walletData.fieldRealizedPnl')],
                    ['unrealized_pnl', t('discoveryProfileDocsFlyout.walletData.fieldUnrealizedPnl')],
                    ['avg_roi', t('discoveryProfileDocsFlyout.walletData.fieldAvgRoi')],
                  ]} />
                </div>

                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.walletData.riskMetricsHeading')}</p>
                  <FieldTable fields={[
                    ['sharpe_ratio', t('discoveryProfileDocsFlyout.walletData.fieldSharpeRatio')],
                    ['sortino_ratio', t('discoveryProfileDocsFlyout.walletData.fieldSortinoRatio')],
                    ['max_drawdown', t('discoveryProfileDocsFlyout.walletData.fieldMaxDrawdown')],
                    ['profit_factor', t('discoveryProfileDocsFlyout.walletData.fieldProfitFactor')],
                    ['calmar_ratio', t('discoveryProfileDocsFlyout.walletData.fieldCalmarRatio')],
                  ]} />
                </div>

                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.walletData.rollingWindowHeading')}</p>
                  <FieldTable fields={[
                    ['rolling_pnl', t('discoveryProfileDocsFlyout.walletData.fieldRollingPnl')],
                    ['rolling_roi', t('discoveryProfileDocsFlyout.walletData.fieldRollingRoi')],
                    ['rolling_win_rate', t('discoveryProfileDocsFlyout.walletData.fieldRollingWinRate')],
                    ['rolling_trade_count', t('discoveryProfileDocsFlyout.walletData.fieldRollingTradeCount')],
                    ['trades_1h', t('discoveryProfileDocsFlyout.walletData.fieldTrades1h')],
                    ['trades_24h', t('discoveryProfileDocsFlyout.walletData.fieldTrades24h')],
                    ['last_trade_at', t('discoveryProfileDocsFlyout.walletData.fieldLastTradeAt')],
                  ]} />
                </div>

                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.walletData.detectionHeading')}</p>
                  <FieldTable fields={[
                    ['anomaly_score', t('discoveryProfileDocsFlyout.walletData.fieldAnomalyScore')],
                    ['insider_score', t('discoveryProfileDocsFlyout.walletData.fieldInsiderScore')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.walletData.tradesListFormatTitle')}</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.walletData.tradesListFormatDesc') }} />
                  </p>
                  <CodeBlock code={`{
    "size": float,          # Position size in USD
    "price": float,         # Entry price
    "side": str,            # "long" | "short"
    "market": str,          # Market identifier (e.g. "SOL-PERP")
    "outcome": str,         # "win" | "loss" | "open"
    "timestamp": str        # ISO 8601 timestamp
}`} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.walletData.rollingDictFormatTitle')}</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.walletData.rollingDictFormatDesc') }} />
                  </p>
                  <CodeBlock code={`{
    "rolling_pnl": float,          # PnL over the window period
    "rolling_roi": float,          # ROI over the window period
    "rolling_win_rate": float,     # Win rate over the window period
    "rolling_trade_count": int,    # Trade count in the window
    "rolling_sharpe": float,       # Sharpe ratio over the window
    "rolling_max_drawdown": float  # Max drawdown over the window
}`} />
                </div>
              </div>
            </Section>

            {/* ==================== 4. SCORING COMPONENTS ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.scoringComponents')} icon={Target} iconColor="text-orange-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.scoringComponents.intro') }} />
              </p>

              <div className="mt-2 border border-border/20 rounded-md overflow-hidden">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="border-b border-border/20 bg-card/30">
                      <th className="text-left px-2 py-1.5 font-medium text-muted-foreground">{t('discoveryProfileDocsFlyout.scoringComponents.colComponent')}</th>
                      <th className="text-right px-2 py-1.5 font-medium text-muted-foreground">{t('discoveryProfileDocsFlyout.scoringComponents.colWeight')}</th>
                      <th className="text-left px-2 py-1.5 font-medium text-muted-foreground">{t('discoveryProfileDocsFlyout.scoringComponents.colDescription')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      [t('discoveryProfileDocsFlyout.scoringComponents.timingSkillName'), '20.0%', t('discoveryProfileDocsFlyout.scoringComponents.timingSkillDesc')],
                      [t('discoveryProfileDocsFlyout.scoringComponents.sharpeRatioName'), '19.5%', t('discoveryProfileDocsFlyout.scoringComponents.sharpeRatioDesc')],
                      [t('discoveryProfileDocsFlyout.scoringComponents.profitFactorName'), '16.25%', t('discoveryProfileDocsFlyout.scoringComponents.profitFactorDesc')],
                      [t('discoveryProfileDocsFlyout.scoringComponents.executionQualityName'), '15.0%', t('discoveryProfileDocsFlyout.scoringComponents.executionQualityDesc')],
                      [t('discoveryProfileDocsFlyout.scoringComponents.winRateName'), '13.0%', t('discoveryProfileDocsFlyout.scoringComponents.winRateDesc')],
                      [t('discoveryProfileDocsFlyout.scoringComponents.pnlName'), '9.75%', t('discoveryProfileDocsFlyout.scoringComponents.pnlDesc')],
                      [t('discoveryProfileDocsFlyout.scoringComponents.consistencyName'), '6.5%', t('discoveryProfileDocsFlyout.scoringComponents.consistencyDesc')],
                    ].map(([name, weight, desc]) => (
                      <tr key={name} className="border-b border-border/10 last:border-0">
                        <td className="px-2 py-1.5">
                          <code className="text-cyan-400 font-mono">{name}</code>
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono text-emerald-400">{weight}</td>
                        <td className="px-2 py-1.5 text-muted-foreground">{desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <CodeBlock code={`# Default rank_score calculation (simplified)
rank_score = (
    timing_skill    * 0.200 +
    sharpe_norm     * 0.195 +
    profit_factor   * 0.1625 +
    execution_qual  * 0.150 +
    win_rate_score  * 0.130 +
    pnl_norm        * 0.0975 +
    consistency     * 0.065
)`} className="mt-2" />
            </Section>

            {/* ==================== 5. POOL SELECTION ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.poolSelection')} icon={Users} iconColor="text-blue-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.poolSelection.intro') }} />
              </p>

              <div className="space-y-3 pt-2">
                <div>
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.poolSelection.eligibilityGatesTitle')}</p>
                  <p className="text-[11px] text-muted-foreground mb-1">{t('discoveryProfileDocsFlyout.poolSelection.eligibilityGatesDesc')}</p>
                  <FieldTable fields={[
                    ['min_trades', t('discoveryProfileDocsFlyout.poolSelection.fieldMinTrades')],
                    ['min_win_rate', t('discoveryProfileDocsFlyout.poolSelection.fieldMinWinRate')],
                    ['min_sharpe', t('discoveryProfileDocsFlyout.poolSelection.fieldMinSharpe')],
                    ['min_profit_factor', t('discoveryProfileDocsFlyout.poolSelection.fieldMinProfitFactor')],
                    ['max_anomaly', t('discoveryProfileDocsFlyout.poolSelection.fieldMaxAnomaly')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.poolSelection.tierSystemTitle')}</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.poolSelection.tierSystemDesc')}
                  </p>
                  <FieldTable fields={[
                    ['core', t('discoveryProfileDocsFlyout.poolSelection.fieldCoreTier')],
                    ['rising', t('discoveryProfileDocsFlyout.poolSelection.fieldRisingTier')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.poolSelection.churnGuardTitle')}</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.poolSelection.churnGuardDesc')}
                  </p>
                  <FieldTable fields={[
                    ['max_hourly_replacement_rate', t('discoveryProfileDocsFlyout.poolSelection.fieldMaxHourlyReplacementRate')],
                    ['replacement_score_cutoff', t('discoveryProfileDocsFlyout.poolSelection.fieldReplacementScoreCutoff')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.poolSelection.clusterDiversityTitle')}</p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.poolSelection.clusterDiversityDesc')}
                  </p>
                  <FieldTable fields={[
                    ['max_cluster_share', t('discoveryProfileDocsFlyout.poolSelection.fieldMaxClusterShare')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.poolSelection.targetPoolSizeTitle')}</p>
                  <p className="text-[11px] text-muted-foreground">
                    <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.poolSelection.targetPoolSizeDesc') }} />
                  </p>
                </div>
              </div>
            </Section>

            {/* ==================== 6. INSIDER DETECTION ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.insiderDetection')} icon={Shield} iconColor="text-red-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.insiderDetection.intro') }} />
              </p>

              <div className="mt-2 border border-border/20 rounded-md overflow-hidden">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="border-b border-border/20 bg-card/30">
                      <th className="text-left px-2 py-1.5 font-medium text-muted-foreground">{t('discoveryProfileDocsFlyout.insiderDetection.colSignal')}</th>
                      <th className="text-right px-2 py-1.5 font-medium text-muted-foreground">{t('discoveryProfileDocsFlyout.insiderDetection.colWeight')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalEarlyEntry'), '0.18'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalPreAnnouncement'), '0.16'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalAbnormalWinRate'), '0.14'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalSizeConcentration'), '0.12'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalCorrelatedTiming'), '0.10'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalUnusualPnl'), '0.08'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalLowLatencyClustering'), '0.07'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalCrossWalletFlows'), '0.06'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalSocialCorrelation'), '0.05'],
                      [t('discoveryProfileDocsFlyout.insiderDetection.signalAnomalousVolume'), '0.04'],
                    ].map(([signal, weight]) => (
                      <tr key={signal} className="border-b border-border/10 last:border-0">
                        <td className="px-2 py-1.5 text-muted-foreground">{signal}</td>
                        <td className="px-2 py-1.5 text-right font-mono text-red-400">{weight}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-2">
                <p className="text-[11px] font-medium text-foreground mb-1">{t('discoveryProfileDocsFlyout.insiderDetection.classificationThresholdsTitle')}</p>
                <FieldTable fields={[
                  ['insider_score >= 0.8', t('discoveryProfileDocsFlyout.insiderDetection.thresholdConfirmed')],
                  ['insider_score >= 0.6', t('discoveryProfileDocsFlyout.insiderDetection.thresholdLikely')],
                  ['insider_score >= 0.4', t('discoveryProfileDocsFlyout.insiderDetection.thresholdSuspected')],
                  ['insider_score < 0.4', t('discoveryProfileDocsFlyout.insiderDetection.thresholdClean')],
                ]} />
              </div>
            </Section>

            {/* ==================== 7. HELPER METHODS ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.helperMethods')} icon={Zap} iconColor="text-yellow-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                <span dangerouslySetInnerHTML={{ __html: t('discoveryProfileDocsFlyout.helperMethods.intro') }} />
              </p>

              <div className="space-y-3 pt-2">
                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.helperMethods.tradeAnalysisHeading')}</p>
                  <FieldTable fields={[
                    ['_calculate_trade_stats(trades)', t('discoveryProfileDocsFlyout.helperMethods.helperCalcTradeStats')],
                    ['_calculate_risk_adjusted_metrics(trades)', t('discoveryProfileDocsFlyout.helperMethods.helperCalcRiskAdjusted')],
                    ['_calculate_rolling_windows(trades, windows)', t('discoveryProfileDocsFlyout.helperMethods.helperCalcRollingWindows')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.helperMethods.scoringHeading')}</p>
                  <FieldTable fields={[
                    ['_calculate_rank_score(wallet, trades)', t('discoveryProfileDocsFlyout.helperMethods.helperCalcRankScore')],
                    ['_compute_timing_skill(trades)', t('discoveryProfileDocsFlyout.helperMethods.helperTimingSkill')],
                    ['_compute_execution_quality(trades)', t('discoveryProfileDocsFlyout.helperMethods.helperExecutionQuality')],
                    ['_score_quality(wallet, trades)', t('discoveryProfileDocsFlyout.helperMethods.helperScoreQuality')],
                    ['_score_activity(wallet)', t('discoveryProfileDocsFlyout.helperMethods.helperScoreActivity')],
                    ['_score_stability(wallet, trades)', t('discoveryProfileDocsFlyout.helperMethods.helperScoreStability')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.helperMethods.classificationHeading')}</p>
                  <FieldTable fields={[
                    ['_classify_wallet(wallet, scores)', t('discoveryProfileDocsFlyout.helperMethods.helperClassifyWallet')],
                    ['_detect_strategies(trades)', t('discoveryProfileDocsFlyout.helperMethods.helperDetectStrategies')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.helperMethods.poolSelectionHeading')}</p>
                  <FieldTable fields={[
                    ['_default_pool_selection(scored_wallets, current_pool)', t('discoveryProfileDocsFlyout.helperMethods.helperDefaultPoolSelection')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.helperMethods.insiderDetectionHeading')}</p>
                  <FieldTable fields={[
                    ['_compute_insider_score(wallet, trades)', t('discoveryProfileDocsFlyout.helperMethods.helperComputeInsider')],
                    ['_classify_insider(insider_score)', t('discoveryProfileDocsFlyout.helperMethods.helperClassifyInsider')],
                  ]} />
                </div>

                <div className="border-t border-border/20 pt-2">
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('discoveryProfileDocsFlyout.helperMethods.utilitiesHeading')}</p>
                  <FieldTable fields={[
                    ['_std_dev(values)', t('discoveryProfileDocsFlyout.helperMethods.helperStdDev')],
                    ['_parse_timestamp(ts)', t('discoveryProfileDocsFlyout.helperMethods.helperParseTimestamp')],
                    ['_to_float(value, default=0.0)', t('discoveryProfileDocsFlyout.helperMethods.helperToFloat')],
                    ['_confidence_adjusted_win_rate(wins, losses)', t('discoveryProfileDocsFlyout.helperMethods.helperConfidenceWinRate')],
                  ]} />
                </div>
              </div>
            </Section>

            {/* ==================== 8. AVAILABLE IMPORTS ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.availableImports')} icon={Package} iconColor="text-violet-400">
              <p className="text-[11px] text-muted-foreground pt-1">
                {t('discoveryProfileDocsFlyout.imports.intro')}
              </p>

              <CodeBlock code={`# SDK
from services.discovery_profile_sdk import BaseDiscoveryProfile

# Standard library
import math
import statistics
import json
import re
import hashlib
import time
import datetime
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

# Third-party (pre-installed)
import numpy as np`} className="mt-1" />

              <p className="text-[11px] text-muted-foreground mt-2">
                {t('discoveryProfileDocsFlyout.imports.sandboxNote')}
              </p>
            </Section>

            {/* ==================== 9. CODE EXAMPLES ==================== */}
            <Section title={t('discoveryProfileDocsFlyout.sections.codeExamples')} icon={Layers} iconColor="text-pink-400">
              <div className="space-y-4 pt-1">

                <div>
                  <p className="text-[11px] font-medium text-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.examples.example1Title')}
                  </p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.examples.example1Desc')}
                  </p>
                  <CodeBlock code={`from services.discovery_profile_sdk import BaseDiscoveryProfile

class RecentActivityProfile(BaseDiscoveryProfile):
    name = "Recent Activity Boost"
    description = "Favors wallets with strong recent performance"
    default_config = {
        "rolling_boost": 1.3,
        "rolling_win_threshold": 0.60,
        "rolling_trade_min": 5,
        "min_trades": 10,
    }

    def score_wallet(self, wallet, trades, rolling):
        base = self._calculate_rank_score(wallet, trades)
        quality = self._score_quality(wallet, trades)
        insider = self._compute_insider_score(wallet, trades)
        tags_info = self._classify_wallet(wallet, base)

        rolling_count = rolling.get("rolling_trade_count", 0)
        rolling_wr = rolling.get("rolling_win_rate", 0)

        if (rolling_count >= self.config["rolling_trade_min"]
                and rolling_wr >= self.config["rolling_win_threshold"]):
            boost = self.config["rolling_boost"]
            base["rank_score"] = min(1.0, base["rank_score"] * boost)
            tags_info["tags"].append("hot_streak")

        return {
            "rank_score": base["rank_score"],
            "quality_score": quality,
            "insider_score": insider,
            "tags": tags_info["tags"],
            "recommendation": tags_info["recommendation"],
        }`} />
                </div>

                <div className="border-t border-border/20 pt-3">
                  <p className="text-[11px] font-medium text-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.examples.example2Title')}
                  </p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.examples.example2Desc')}
                  </p>
                  <CodeBlock code={`from services.discovery_profile_sdk import BaseDiscoveryProfile

class StrictPoolProfile(BaseDiscoveryProfile):
    name = "Strict Pool"
    description = "High-bar pool with tight eligibility"
    default_config = {
        "pool_size": 20,
        "min_trades": 50,
        "min_win_rate": 0.60,
        "min_sharpe": 1.5,
        "min_profit_factor": 1.8,
        "max_anomaly": 0.4,
        "max_cluster_share": 0.15,
        "max_hourly_replacement_rate": 0.10,
        "replacement_score_cutoff": 0.08,
    }

    def select_pool(self, scored_wallets, current_pool):
        # Use default selection but with our strict config
        result = self._default_pool_selection(
            scored_wallets, current_pool
        )

        # Extra filter: remove any wallet with insider suspicion
        clean_members = [
            addr for addr, scores in scored_wallets
            if addr in result["members"]
            and scores.get("insider_score", 0) < 0.35
        ]

        return {
            "members": clean_members,
            "removals": [
                a for a in result["members"]
                if a not in clean_members
            ],
            "pool_size": len(clean_members),
        }`} />
                </div>

                <div className="border-t border-border/20 pt-3">
                  <p className="text-[11px] font-medium text-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.examples.example3Title')}
                  </p>
                  <p className="text-[11px] text-muted-foreground mb-1">
                    {t('discoveryProfileDocsFlyout.examples.example3Desc')}
                  </p>
                  <CodeBlock code={`from services.discovery_profile_sdk import BaseDiscoveryProfile

class InsiderWeightedProfile(BaseDiscoveryProfile):
    name = "Insider-Weighted"
    description = "Deprioritizes wallets with insider signals"
    default_config = {
        "insider_penalty_factor": 0.5,
        "insider_hard_cutoff": 0.7,
        "min_trades": 15,
    }

    def score_wallet(self, wallet, trades, rolling):
        base = self._calculate_rank_score(wallet, trades)
        quality = self._score_quality(wallet, trades)
        insider = self._compute_insider_score(wallet, trades)
        classification = self._classify_insider(insider)
        tags_info = self._classify_wallet(wallet, base)

        rank = base["rank_score"]

        # Hard cutoff for high insider scores
        if insider >= self.config["insider_hard_cutoff"]:
            rank = 0.0
            tags_info["tags"].append("insider_blocked")
            tags_info["recommendation"] = "avoid"
        elif insider >= 0.4:
            # Proportional penalty
            penalty = insider * self.config["insider_penalty_factor"]
            rank = max(0.0, rank - penalty)
            tags_info["tags"].append(f"insider_{classification}")

        return {
            "rank_score": rank,
            "quality_score": quality,
            "insider_score": insider,
            "tags": tags_info["tags"],
            "recommendation": tags_info["recommendation"],
        }`} />
                </div>
              </div>
            </Section>

          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}
