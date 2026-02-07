import { atom } from 'jotai'

// Theme
export type Theme = 'dark' | 'light'
export const themeAtom = atom<Theme>('dark')

// Derive theme class for applying to body
export const themeClassAtom = atom((get) => {
  return get(themeAtom) === 'light' ? 'theme-light' : 'theme-dark'
})

// UI State
export const shortcutsHelpOpenAtom = atom(false)
export const copilotOpenAtom = atom(false)
export const commandBarOpenAtom = atom(false)

// Scanner data freshness
export const lastScanTimeAtom = atom<string | null>(null)
export const lastWebSocketMessageTimeAtom = atom<string | null>(null)

// Data simulation
export const simulationEnabledAtom = atom(true)
