# ModelServer Dockerfile
# Hosts Qwen2.5-VL-7B-Instruct and MineCLIP models
# Supports GPU (CUDA) with fallback to CPU

FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

WORKDIR /app

# Copy requirements first for layer caching
COPY ModelServer/requirements.txt /app/ModelServer/requirements.txt

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
RUN pip install --no-cache-dir -r /app/ModelServer/requirements.txt

# Copy MineCLIP module
COPY MineCLIP /app/MineCLIP

# Copy ModelServer code
COPY ModelServer /app/ModelServer

# Set Python path to include app root and MineCLIP
ENV PYTHONPATH=/app:/app/MineCLIP

# Environment variables for model caching
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run server with uvicorn
CMD ["python", "-m", "uvicorn", "ModelServer.server:app", "--host", "0.0.0.0", "--port", "8080"]
