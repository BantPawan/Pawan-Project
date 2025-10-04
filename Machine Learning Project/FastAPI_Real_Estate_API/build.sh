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

# Move back to FastAPI folder root
cd ..

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create backend static directory and copy frontend build artifacts
echo "📁 Setting up static files..."
mkdir -p backend/static

# typical build outputs: frontend/dist or frontend/build
if [ -d frontend/dist ]; then
  cp -r frontend/dist/* backend/static/
elif [ -d frontend/build ]; then
  cp -r frontend/build/* backend/static/
else
  echo "❌ Frontend build output not found (expected frontend/dist or frontend/build)"
  exit 1
fi

echo "🎉 Build completed successfully!"
