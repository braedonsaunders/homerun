// Polyfill crypto.randomUUID for non-secure contexts (HTTP) where
// the native API is unavailable.  OpenUI's useTransformedKeys hook
// relies on it for chart CSS-variable names.
if (typeof crypto !== 'undefined' && typeof crypto.randomUUID !== 'function') {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(crypto as any).randomUUID = (): string => {
    // RFC-4122 v4 UUID via getRandomValues (available in all modern browsers)
    const bytes = new Uint8Array(16)
    crypto.getRandomValues(bytes)
    bytes[6] = (bytes[6] & 0x0f) | 0x40 // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80 // variant 1
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
  }
}

import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider as JotaiProvider } from 'jotai'
import App from './App'
import './i18n'
// OpenUI CSS must load BEFORE index.css so our .theme-light/.theme-dark
// overrides in index.css take precedence over OpenUI's @media(prefers-color-scheme).
import '@openuidev/react-ui/defaults.css'
import '@openuidev/react-ui/components.css'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchInterval: 120000, // Fallback poll every 120s (WS pushes are primary)
      refetchOnWindowFocus: false,
      staleTime: 30000, // Data stays fresh for 30s (WS updates override)
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <JotaiProvider>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </JotaiProvider>
  </React.StrictMode>,
)
