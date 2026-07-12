# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Build tools needed to compile llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Use prebuilt CPU wheels for llama-cpp-python instead of compiling from
# source (compiling can take 15-25 minutes; the prebuilt wheel installs
# in seconds). Falls back to source build automatically if no matching
# wheel is found for this Python/platform combo.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# --- Local model ---------------------------------------------------------
# Small (1.5B, 4-bit) instruct model — fits comfortably in the 4GB RAM /
# 2 vCPU grading box alongside the agent process.
# Swap this URL for any other small GGUF model if you prefer.
RUN mkdir -p /app/models && \
    curl -L --retry 3 --retry-delay 5 --fail --show-error \
    -o /app/models/local-model.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

COPY agent.py .

# The harness mounts /input and expects /output to exist
RUN mkdir -p /input /output

CMD ["python", "agent.py"]
