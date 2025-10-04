#!/bin/bash
set -euo pipefail

echo "🚀 Starting build process..."

# Show current directory for debug
echo "Working dir: $(pwd)"
echo "Listing repo root:"
ls -la

# Build frontend
echo "📦 Installing frontend dependencies..."
cd frontend
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi

echo "🔨 Building React frontend..."
npm run build

# Move back to root
cd ..

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create backend static directory
echo "📁 Setting up static files..."
mkdir -p backend/static

# Copy built files: index.html to root of static/, assets/ to static/assets/
cp frontend/dist/index.html backend/static/index.html
cp -r frontend/dist/assets backend/static/assets 2>/dev/null || echo "No assets dir (fine for first build)"
# Copy any public/ leftovers (e.g., favicon) if needed
if [ -d "frontend/dist/*.ico" ] || [ -d "frontend/public" ]; then
  cp -r frontend/public/* backend/static/ 2>/dev/null || true
fi

# Debug: List static contents
echo "📋 Backend static contents:"
ls -la backend/static/
ls -la backend/static/assets/ 2>/dev/null || echo "No assets yet"

echo "🎉 Build completed successfully!"
