# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Build tools needed to compile llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Local model ---------------------------------------------------------
# Small (1.5B, 4-bit) instruct model — fits comfortably in the 4GB RAM /
# 2 vCPU grading box alongside the agent process.
# Swap this URL for any other small GGUF model if you prefer.
RUN mkdir -p /app/models && \
    curl -L -o /app/models/local-model.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

COPY agent.py .

# The harness mounts /input and expects /output to exist
RUN mkdir -p /input /output

CMD ["python", "agent.py"]
