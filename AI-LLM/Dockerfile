FROM ollama/ollama:latest

# Start Ollama server in background
RUN ollama serve & \
    # Wait for server to be ready
    until curl -s http://localhost:11434/api/tags > /dev/null; do sleep 1; done && \
    # Now pull the model
    ollama pull llama3.2:1b && \
    # Copy and create custom model
    mkdir -p /root/.ollama && \
    cp /app/Modelfile /root/.ollama/Modelfile && \
    ollama create paper-analyzer -f /root/.ollama/Modelfile

# Copy app files (if needed later)
COPY . /app

EXPOSE 11434
CMD ["serve"]
