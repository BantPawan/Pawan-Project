FROM ollama/ollama:latest

# Copy and create model
COPY Modelfile /root/.ollama/
RUN ollama create paper-analyzer -f /root/.ollama/Modelfile

EXPOSE 11434
CMD ["serve"]
