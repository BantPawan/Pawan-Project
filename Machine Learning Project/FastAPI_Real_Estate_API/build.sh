#!/bin/bash
set -euo pipefail

echo "🚀 Starting build process..."

# Show current directory for debug
echo "Working dir: $(pwd)"
echo "Listing:"
ls -la

# Build frontend
echo "📦 Building React frontend..."
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "🔨 Building React app..."
npm run build

# Move back to root
cd ..

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create backend static directory
echo "📁 Setting up static files..."
mkdir -p backend/static

# Copy ALL built files from frontend/dist to backend/static
echo "Copying built files..."
cp -r frontend/dist/* backend/static/ 2>/dev/null || echo "Some files not copied, continuing..."

# Debug: List static contents
echo "📋 Backend static contents:"
ls -la backend/static/ || echo "Static directory might be empty"

echo "🎉 Build completed successfully!"
