import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/agentic-serve/',
  define: {
    __BUILD_HASH__: JSON.stringify(Date.now().toString(36)),
  },
})
