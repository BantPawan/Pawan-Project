#!/bin/bash

echo "🚀 Starting build process..."

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd "Machine Learning Project/FastAPI_Real_Estate_API/frontend"
npm install

# Build frontend
echo "🔨 Building React frontend..."
npm run build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Frontend build successful"
else
    echo "❌ Frontend build failed"
    exit 1
fi

# Go back to project root
cd ../..

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
cd "Machine Learning Project/FastAPI_Real_Estate_API/backend"
pip install -r requirements.txt

# Create static directory for frontend files
echo "📁 Setting up static files..."
mkdir -p static
cp -r ../frontend/dist/* static/

echo "🎉 Build completed successfully!"
