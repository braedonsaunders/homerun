import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'

interface TwitterSourceFormProps {
  config: Record<string, any>
  onChange: (config: Record<string, any>) => void
}

export default function TwitterSourceForm({ config, onChange }: TwitterSourceFormProps) {
  const { t } = useTranslation()
  const pollIntervalOptions = useMemo(() => [
    { value: '5', label: t('twitterSourceForm.pollOption5min') },
    { value: '15', label: t('twitterSourceForm.pollOption15min') },
    { value: '30', label: t('twitterSourceForm.pollOption30min') },
    { value: '60', label: t('twitterSourceForm.pollOption1hour') },
    { value: '240', label: t('twitterSourceForm.pollOption4hours') },
    { value: '1440', label: t('twitterSourceForm.pollOption24hours') },
  ], [t])

  const update = (key: string, value: unknown) => {
    onChange({ ...config, [key]: value })
  }

  const handles = String(config.handles || '')
  const keywords = String(config.keywords || '')
  const bearerToken = String(config.bearer_token || '')
  const nitterInstance = String(config.nitter_instance || 'nitter.privacydev.net')
  const pollInterval = String(config.poll_interval_minutes || '15')
  const limit = String(config.limit || '50')

  return (
    <div className="space-y-3">
      <div>
        <Label className="text-[11px] text-muted-foreground">
          {t('twitterSourceForm.accounts')} <span className="text-muted-foreground/60 font-normal">{t('twitterSourceForm.accountsHint')}</span>
        </Label>
        <textarea
          value={handles}
          onChange={(e) => update('handles', e.target.value)}
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-xs font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          rows={2}
          placeholder="@polyaborist, @KalshiWatch, @ElectionBettingOdds, @PolymarketWhale"
          spellCheck={false}
        />
        <p className="text-[10px] text-muted-foreground mt-1">{t('twitterSourceForm.accountsHelp')}</p>
      </div>

      <div>
        <Label className="text-[11px] text-muted-foreground">
          {t('twitterSourceForm.keywords')} <span className="text-muted-foreground/60 font-normal">{t('twitterSourceForm.keywordsHint')}</span>
        </Label>
        <textarea
          value={keywords}
          onChange={(e) => update('keywords', e.target.value)}
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-xs font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          rows={2}
          placeholder="polymarket, kalshi, prediction market, election odds"
          spellCheck={false}
        />
        <p className="text-[10px] text-muted-foreground mt-1">{t('twitterSourceForm.keywordsHelp')}</p>
      </div>

      <div>
        <Label className="text-[11px] text-muted-foreground">
          {t('twitterSourceForm.bearerToken')} <span className="text-muted-foreground/60 font-normal">{t('twitterSourceForm.optional')}</span>
        </Label>
        <Input
          type="password"
          value={bearerToken}
          onChange={(e) => update('bearer_token', e.target.value)}
          className="mt-1 h-8 text-xs font-mono"
          placeholder="AAAA..."
          autoComplete="off"
        />
        <p className="text-[10px] text-muted-foreground mt-1">
          {t('twitterSourceForm.bearerTokenHelp')}
        </p>
      </div>

      <div>
        <Label className="text-[11px] text-muted-foreground">{t('twitterSourceForm.nitterInstance')}</Label>
        <Input
          value={nitterInstance}
          onChange={(e) => update('nitter_instance', e.target.value)}
          className="mt-1 h-8 text-xs font-mono"
          placeholder="nitter.privacydev.net"
        />
        <p className="text-[10px] text-muted-foreground mt-1">{t('twitterSourceForm.nitterInstanceHelp')}</p>
      </div>

      <div className="grid gap-3 grid-cols-2">
        <div>
          <Label className="text-[11px] text-muted-foreground">{t('twitterSourceForm.pollInterval')}</Label>
          <Select value={pollInterval} onValueChange={(val) => update('poll_interval_minutes', parseInt(val, 10))}>
            <SelectTrigger className="mt-1 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pollIntervalOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label className="text-[11px] text-muted-foreground">{t('twitterSourceForm.maxTweets')}</Label>
          <Input
            type="number"
            value={limit}
            onChange={(e) => update('limit', parseInt(e.target.value, 10) || 50)}
            className="mt-1 h-8 text-xs font-mono"
            min={1}
            max={200}
            placeholder="50"
          />
        </div>
      </div>
    </div>
  )
}
