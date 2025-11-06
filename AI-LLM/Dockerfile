FROM ollama/ollama:latest

# Pull base model first (ensures it's available)
RUN ollama pull llama3.2:1b

# Copy Modelfile and create at runtime (avoids build-time issues)
COPY Modelfile /root/.ollama/Modelfile
RUN ollama create paper-analyzer -f /root/.ollama/Modelfile

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:11434/api/tags || exit 1

EXPOSE 11434
CMD ["serve"]
