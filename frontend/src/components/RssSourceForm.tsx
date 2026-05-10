import { useTranslation } from 'react-i18next'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'

interface RssSourceFormProps {
  config: Record<string, any>
  onChange: (config: Record<string, any>) => void
}

export default function RssSourceForm({ config, onChange }: RssSourceFormProps) {
  const { t } = useTranslation()

  const POLL_INTERVAL_OPTIONS = [
    { value: '5', label: t('rssSourceForm.poll5m') },
    { value: '15', label: t('rssSourceForm.poll15m') },
    { value: '30', label: t('rssSourceForm.poll30m') },
    { value: '60', label: t('rssSourceForm.poll1h') },
    { value: '240', label: t('rssSourceForm.poll4h') },
    { value: '1440', label: t('rssSourceForm.poll24h') },
  ]

  const update = (key: string, value: unknown) => {
    onChange({ ...config, [key]: value })
  }

  const feedUrl = String(config.url || '')
  const pollInterval = String(config.poll_interval_minutes || '15')
  const categoryFilter = String(config.category_filter || '')

  return (
    <div className="space-y-3">
      <div>
        <Label className="text-[11px] text-muted-foreground">
          {t('rssSourceForm.feedUrl')} <span className="text-red-400">*</span>
        </Label>
        <Input
          type="url"
          value={feedUrl}
          onChange={(e) => update('url', e.target.value)}
          className="mt-1 h-8 text-xs font-mono"
          placeholder="https://example.com/feed.xml"
          required
        />
        {feedUrl && !/^https?:\/\/.+/.test(feedUrl) && (
          <p className="text-[10px] text-red-400 mt-1">{t('rssSourceForm.urlInvalid')}</p>
        )}
      </div>

      <div className="grid gap-3 grid-cols-2">
        <div>
          <Label className="text-[11px] text-muted-foreground">{t('rssSourceForm.pollInterval')}</Label>
          <Select value={pollInterval} onValueChange={(val) => update('poll_interval_minutes', parseInt(val, 10))}>
            <SelectTrigger className="mt-1 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {POLL_INTERVAL_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label className="text-[11px] text-muted-foreground">{t('rssSourceForm.categoryFilter')}</Label>
          <Input
            value={categoryFilter}
            onChange={(e) => update('category_filter', e.target.value || undefined)}
            className="mt-1 h-8 text-xs"
            placeholder={t('rssSourceForm.categoryPlaceholder')}
          />
        </div>
      </div>
    </div>
  )
}
