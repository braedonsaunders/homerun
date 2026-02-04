import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            // Suppress EPIPE and ECONNRESET errors which occur when
            // WebSocket connections close - these are expected behavior
            if (err.message.includes('EPIPE') || err.message.includes('ECONNRESET')) {
              return
            }
            console.error('WebSocket proxy error:', err.message)
          })
        },
      },
    },
  },
})
