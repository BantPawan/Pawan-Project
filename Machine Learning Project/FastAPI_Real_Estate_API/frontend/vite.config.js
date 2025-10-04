import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { fileURLToPath } from 'url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  plugins: [react()],
  root: './',
  build: {
    rollupOptions: {
      input: resolve(__dirname, 'index.html')  // Now points to root index.html
    },
    outDir: 'dist'
  },
  publicDir: 'public',  // Still serves public/ assets (e.g., favicon) to dist/
  base: '/',  // Ensures /assets/ paths
  server: {
    port: 3000
  }
})
