import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider as JotaiProvider } from 'jotai'
import App from './App'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 120000, // Fallback poll every 120s (WS pushes are primary)
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
