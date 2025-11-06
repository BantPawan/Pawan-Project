FROM ollama/ollama:latest

# Install curl + wget
RUN apt-get update && apt-get install -y curl wget && rm -rf /var/lib/apt/lists/*

# Copy Modelfile FIRST
COPY Modelfile /root/.ollama/Modelfile

# Start Ollama server in background
RUN ollama serve & \
    # Wait for server
    until wget -q --spider http://localhost:11434/api/tags 2>/dev/null || curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do \
        echo "Waiting for Ollama..."; sleep 2; \
    done && \
    # Pull model
    ollama pull llama3.2:1b && \
    # Create custom model
    ollama create paper-analyzer -f /root/.ollama/Modelfile

# Copy app
COPY . /app

EXPOSE 11434
CMD ["serve"]
