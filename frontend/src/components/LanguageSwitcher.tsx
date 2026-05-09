import { useTranslation } from 'react-i18next'
import { Globe } from 'lucide-react'
import { SUPPORTED_LANGUAGES } from '../i18n'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'

export default function LanguageSwitcher() {
  const { t, i18n } = useTranslation()
  const current = (i18n.resolvedLanguage || i18n.language || 'en').split('-')[0]

  const handleChange = (code: string) => {
    i18n.changeLanguage(code)
  }

  return (
    <div className="bg-card/60 border border-border/40 rounded-xl p-4 flex items-center justify-between gap-4">
      <div className="flex items-start gap-3 min-w-0">
        <div className="w-9 h-9 rounded-lg bg-muted/60 flex items-center justify-center shrink-0">
          <Globe className="w-4 h-4 text-muted-foreground" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold">{t('settings.language')}</div>
          <div className="text-xs text-muted-foreground truncate">
            {t('settings.languageDescription')}
          </div>
        </div>
      </div>
      <div className="shrink-0">
        <Select value={current} onValueChange={handleChange}>
          <SelectTrigger className="w-[180px]" data-testid="language-switcher-trigger">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SUPPORTED_LANGUAGES.map((lang) => (
              <SelectItem key={lang.code} value={lang.code}>
                {lang.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
