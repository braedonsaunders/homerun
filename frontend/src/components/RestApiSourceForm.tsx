import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'

interface RestApiSourceFormProps {
  config: Record<string, any>
  onChange: (config: Record<string, any>) => void
}

export default function RestApiSourceForm({ config, onChange }: RestApiSourceFormProps) {
  const { t } = useTranslation()
  const pollIntervalOptions = useMemo(() => [
    { value: '5', label: t('restApiSourceForm.pollOption5min') },
    { value: '15', label: t('restApiSourceForm.pollOption15min') },
    { value: '30', label: t('restApiSourceForm.pollOption30min') },
    { value: '60', label: t('restApiSourceForm.pollOption1hour') },
    { value: '240', label: t('restApiSourceForm.pollOption4hours') },
    { value: '1440', label: t('restApiSourceForm.pollOption24hours') },
  ], [t])

  const update = (key: string, value: unknown) => {
    onChange({ ...config, [key]: value })
  }

  const apiUrl = String(config.url || '')
  const jsonPath = String(config.json_path || '')
  const pollInterval = String(config.poll_interval_minutes || '15')

  const headersRaw = config.headers || {}
  const headersText = useMemo(() => {
    try {
      return typeof headersRaw === 'string' ? headersRaw : JSON.stringify(headersRaw, null, 2)
    } catch {
      return '{}'
    }
  }, [headersRaw])

  const headersError = useMemo(() => {
    try {
      const parsed = JSON.parse(headersText)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return t('restApiSourceForm.mustBeJsonObject')
      }
      return null
    } catch {
      return t('restApiSourceForm.invalidJson')
    }
  }, [headersText, t])

  return (
    <div className="space-y-3">
      <div>
        <Label className="text-[11px] text-muted-foreground">
          {t('restApiSourceForm.apiUrl')} <span className="text-red-400">*</span>
        </Label>
        <Input
          type="url"
          value={apiUrl}
          onChange={(e) => update('url', e.target.value)}
          className="mt-1 h-8 text-xs font-mono"
          placeholder="https://api.example.com/v1/data"
          required
        />
        {apiUrl && !/^https?:\/\/.+/.test(apiUrl) && (
          <p className="text-[10px] text-red-400 mt-1">{t('restApiSourceForm.mustBeValidUrl')}</p>
        )}
      </div>

      <div>
        <Label className="text-[11px] text-muted-foreground">{t('restApiSourceForm.headers')}</Label>
        <textarea
          value={headersText}
          onChange={(e) => {
            const text = e.target.value
            try {
              const parsed = JSON.parse(text)
              if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                update('headers', parsed)
              } else {
                update('headers', text)
              }
            } catch {
              update('headers', text)
            }
          }}
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-xs font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          rows={4}
          placeholder='{"Authorization": "Bearer ...", "Accept": "application/json"}'
          spellCheck={false}
        />
        {headersError && (
          <p className="text-[10px] text-red-400 mt-1">{headersError}</p>
        )}
      </div>

      <div className="grid gap-3 grid-cols-2">
        <div>
          <Label className="text-[11px] text-muted-foreground">{t('restApiSourceForm.jsonPathExpression')}</Label>
          <Input
            value={jsonPath}
            onChange={(e) => update('json_path', e.target.value)}
            className="mt-1 h-8 text-xs font-mono"
            placeholder="$.items[*]"
          />
          <p className="text-[10px] text-muted-foreground mt-1">{t('restApiSourceForm.jsonPathHelp')}</p>
        </div>

        <div>
          <Label className="text-[11px] text-muted-foreground">{t('restApiSourceForm.pollInterval')}</Label>
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
      </div>
    </div>
  )
}
