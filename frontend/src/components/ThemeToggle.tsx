import { useAtom } from 'jotai'
import { Sun, Moon } from 'lucide-react'
import clsx from 'clsx'
import { themeAtom, Theme } from '../store/atoms'
import { useEffect } from 'react'

export default function ThemeToggle({ className = '' }: { className?: string }) {
  const [theme, setTheme] = useAtom(themeAtom)

  const toggleTheme = () => {
    setTheme((prev: Theme) => prev === 'dark' ? 'light' : 'dark')
  }

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'light') {
      root.classList.add('theme-light')
      root.classList.remove('theme-dark')
    } else {
      root.classList.add('theme-dark')
      root.classList.remove('theme-light')
    }
  }, [theme])

  return (
    <button
      onClick={toggleTheme}
      className={clsx(
        'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all',
        theme === 'dark'
          ? 'bg-gray-800 text-gray-400 hover:text-yellow-400 hover:bg-gray-700'
          : 'bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20',
        className
      )}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? (
        <Sun className="w-3.5 h-3.5" />
      ) : (
        <Moon className="w-3.5 h-3.5" />
      )}
    </button>
  )
}
