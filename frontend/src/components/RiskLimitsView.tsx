import { memo } from 'react'
import { useTranslation } from 'react-i18next'
import { useAtom } from 'jotai'
import { AlertTriangle, Loader2 } from 'lucide-react'
import type { Trader } from '../services/apiTraders'
import { draftRiskValuesAtom } from '../store/atoms'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import StrategyConfigForm from './StrategyConfigForm'

type RiskLimitsViewProps = {
  selectedTrader: Trader | null
  riskFormSchema: { param_fields: any[] }
  riskDraftDirty: boolean
  setRiskDraftDirty: (value: boolean) => void
  riskSaveError: string | null
  saveRiskLimitsMutation: { isPending: boolean; mutate: () => void }
  // Re-load the trader's persisted risk_limits into the atom (Discard).
  onDiscard: () => void
  // Whether the flyout is currently open — when it is we don't flag dirty
  // (the flyout has its own save flow and we don't want to double-mark).
  flyoutOpen: boolean
}

function RiskLimitsViewImpl({
  selectedTrader,
  riskFormSchema,
  riskDraftDirty,
  setRiskDraftDirty,
  riskSaveError,
  saveRiskLimitsMutation,
  onDiscard,
  flyoutOpen,
}: RiskLimitsViewProps) {
  const { t } = useTranslation()
  const [riskValues, setRiskValues] = useAtom(draftRiskValuesAtom)

  if (!selectedTrader) {
    return (
      <div className="h-full min-h-0 overflow-auto px-1">
        <div className="h-full min-h-0 rounded-md border border-border/50 bg-muted/10 p-2">
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-700 dark:text-amber-100">
            {t('riskLimitsView.selectBotPrompt')}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 overflow-auto px-1">
      <div className="h-full min-h-0 rounded-md border border-border/50 bg-muted/10 p-2">
        <div className="flex h-full min-h-0 flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 text-rose-500" />
              <p className="text-[11px] font-medium">{t('riskLimitsView.title')}</p>
              <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono">
                {t('riskLimitsView.fieldsCount', { count: riskFormSchema.param_fields.length })}
              </Badge>
              {riskDraftDirty ? (
                <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-amber-500">
                  {t('riskLimitsView.unsaved')}
                </span>
              ) : null}
            </div>
            <div className="flex items-center gap-1.5">
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-6 px-2 text-[10px]"
                onClick={() => {
                  onDiscard()
                  setRiskDraftDirty(false)
                }}
                disabled={!riskDraftDirty}
              >
                {t('riskLimitsView.discard')}
              </Button>
              <Button
                type="button"
                size="sm"
                className="h-6 px-2 text-[10px]"
                onClick={() => saveRiskLimitsMutation.mutate()}
                disabled={saveRiskLimitsMutation.isPending || !riskDraftDirty}
              >
                {saveRiskLimitsMutation.isPending ? (
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                ) : null}
                {t('riskLimitsView.saveRiskLimits')}
              </Button>
            </div>
          </div>
          <p className="text-[10px] text-muted-foreground/80">
            {t('riskLimitsView.renderedFromSchema')}
          </p>
          {riskSaveError ? (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-2 py-1 text-[10px] text-red-500">
              {riskSaveError}
            </div>
          ) : null}
          <div className="flex-1 min-h-0 overflow-auto">
            {riskFormSchema.param_fields.length > 0 ? (
              <StrategyConfigForm
                schema={riskFormSchema}
                values={riskValues}
                onChange={(values) => {
                  setRiskValues(values)
                  if (!flyoutOpen) setRiskDraftDirty(true)
                }}
              />
            ) : (
              <p className="text-[10px] text-muted-foreground/80">{t('riskLimitsView.noRiskSchema')}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export const RiskLimitsView = memo(RiskLimitsViewImpl)
