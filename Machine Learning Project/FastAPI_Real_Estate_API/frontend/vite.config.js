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
      input: resolve(__dirname, 'index.html')
    },
    outDir: 'dist'
  },
  publicDir: 'public',
  server: {
    port: 3000
  }
})
