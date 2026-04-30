import { useAtom, useAtomValue, useSetAtom } from 'jotai'
import { Sun, Moon, Monitor } from 'lucide-react'
import { cn } from '../lib/utils'
import { systemThemeAtom, themeAtom, themePreferenceAtom, ThemePreference } from '../store/atoms'
import { useEffect } from 'react'
import { Button } from './ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'

const PREFERENCE_CYCLE: ThemePreference[] = ['system', 'light', 'dark']

const PREFERENCE_LABEL: Record<ThemePreference, string> = {
  system: 'Auto',
  light: 'Light',
  dark: 'Dark',
}

export default function ThemeToggle({ className = '' }: { className?: string }) {
  const theme = useAtomValue(themeAtom)
  const setSystemTheme = useSetAtom(systemThemeAtom)
  const [themePreference, setThemePreference] = useAtom(themePreferenceAtom)

  const currentIndex = PREFERENCE_CYCLE.indexOf(themePreference)
  const nextPreference = PREFERENCE_CYCLE[(currentIndex + 1) % PREFERENCE_CYCLE.length]

  const cycleTheme = () => {
    setThemePreference(nextPreference)
  }

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const updateSystemTheme = (matches: boolean) => {
      setSystemTheme(matches ? 'dark' : 'light')
    }

    updateSystemTheme(mediaQuery.matches)

    const handleChange = (event: MediaQueryListEvent) => {
      updateSystemTheme(event.matches)
    }

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', handleChange)
      return () => mediaQuery.removeEventListener('change', handleChange)
    }

    mediaQuery.addListener(handleChange)
    return () => mediaQuery.removeListener(handleChange)
  }, [setSystemTheme])

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('theme-light', theme === 'light')
    root.classList.toggle('theme-dark', theme === 'dark')
    root.classList.toggle('dark', theme === 'dark')
    root.style.colorScheme = theme
  }, [theme])

  const Icon = themePreference === 'system' ? Monitor : themePreference === 'dark' ? Moon : Sun

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          onClick={cycleTheme}
          aria-label={`Theme: ${PREFERENCE_LABEL[themePreference]}. Click to switch to ${PREFERENCE_LABEL[nextPreference]}.`}
          className={cn('h-8 px-2', className)}
        >
          <Icon className="w-3.5 h-3.5" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        {themePreference === 'system'
          ? `Auto · ${theme} (follows system). Click for ${PREFERENCE_LABEL[nextPreference]}.`
          : `${PREFERENCE_LABEL[themePreference]} (manual). Click for ${PREFERENCE_LABEL[nextPreference]}.`}
      </TooltipContent>
    </Tooltip>
  )
}
