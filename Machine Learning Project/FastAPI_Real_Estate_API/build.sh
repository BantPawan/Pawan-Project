#!/bin/bash
set -e

echo "🚀 Starting build process..."
echo "Current directory: $(pwd)"

# Build frontend
echo "📦 Building React frontend..."
cd frontend

# Install dependencies
npm install

echo "🔨 Building React app..."
npm run build

# Return to root
cd ..

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create and clean static directory
echo "📁 Setting up static files..."
rm -rf backend/static
mkdir -p backend/static

# Copy ALL files from frontend/dist to backend/static
echo "📋 Copying built files..."
cp -r frontend/dist/* backend/static/

echo "🎉 Build completed successfully!"
