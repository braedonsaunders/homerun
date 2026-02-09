import { atom } from 'jotai'
import { atomWithStorage } from 'jotai/utils'

// Theme
export type Theme = 'dark' | 'light'
export const themeAtom = atomWithStorage<Theme>('theme', 'dark')

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

// Global account mode (sandbox vs live)
export type AccountMode = 'sandbox' | 'live'
export const accountModeAtom = atomWithStorage<AccountMode>('accountMode', 'sandbox')

// Global selected account
// For sandbox: the simulation account ID (e.g. "sim_abc123")
// For live: "live:polymarket" or "live:kalshi"
export const selectedAccountIdAtom = atomWithStorage<string | null>('selectedAccountId', null)

// Derived: is the selected account a live account?
export const isLiveAccountAtom = atom((get) => {
  const id = get(selectedAccountIdAtom)
  return id?.startsWith('live:') ?? false
})

// Derived: live platform from selected account
export const selectedLivePlatformAtom = atom((get) => {
  const id = get(selectedAccountIdAtom)
  if (!id?.startsWith('live:')) return null
  return id.replace('live:', '') as 'polymarket' | 'kalshi'
})
