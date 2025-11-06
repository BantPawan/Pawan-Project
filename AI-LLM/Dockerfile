FROM ollama/ollama:latest

COPY Modelfile /Modelfile
RUN ollama create paper-analyzer -f /Modelfile

EXPOSE 11434
CMD ["serve"]
