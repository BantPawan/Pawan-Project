FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy backend requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Move built files to backend static directory
WORKDIR /app
RUN mkdir -p backend/static
RUN cp -r frontend/dist/* backend/static/

# Expose port
EXPOSE 8000

# Start the application
WORKDIR /app/backend
CMD ["python", "app.py"]
