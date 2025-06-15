# Multi-stage Docker build for AI News Dashboard MLOps
# Stage 1: Base image with system dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    redis-server \
    postgresql-client \
    mongodb-clients \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Docker CLI for container orchestration
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli

# Install kubectl for Kubernetes management
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm for Kubernetes package management
RUN curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | tee /usr/share/keyrings/helm.gpg > /dev/null \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | tee /etc/apt/sources.list.d/helm-stable-debian.list \
    && apt-get update \
    && apt-get install -y helm

# Create application user
RUN useradd --create-home --shell /bin/bash mlops
USER mlops
WORKDIR /home/mlops

# Stage 2: Python environment setup
FROM base AS python-env

# Create virtual environment
RUN python3.10 -m venv /home/mlops/venv
ENV PATH="/home/mlops/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with GPU support
RUN pip install tensorflow[and-cuda]==2.13.0

# Copy requirements and install Python dependencies
COPY --chown=mlops:mlops mlops/requirements.txt /home/mlops/
RUN pip install -r requirements.txt

# Install additional ML libraries
RUN pip install \
    accelerate \
    bitsandbytes \
    deepspeed \
    fairscale \
    flash-attn \
    xformers

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Stage 3: Application setup
FROM python-env AS app

# Copy application code
COPY --chown=mlops:mlops . /home/mlops/app/
WORKDIR /home/mlops/app

# Install Node.js dependencies for frontend
RUN cd /home/mlops/app && npm install

# Build Next.js application
RUN cd /home/mlops/app && npm run build

# Create necessary directories
RUN mkdir -p /home/mlops/app/logs \
    /home/mlops/app/models \
    /home/mlops/app/data \
    /home/mlops/app/cache \
    /home/mlops/app/monitoring

# Set permissions
RUN chmod +x /home/mlops/app/mlops/scripts/*.sh || true

# Stage 4: Production image
FROM app AS production

# Copy configuration files
COPY --chown=mlops:mlops mlops/config/ /home/mlops/app/config/
COPY --chown=mlops:mlops mlops/docker/entrypoint.sh /home/mlops/entrypoint.sh
RUN chmod +x /home/mlops/entrypoint.sh

# Expose ports
EXPOSE 3000 8000 8080 8081 9090 9091 6379 5432 27017

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/home/mlops/entrypoint.sh"]
CMD ["all"]

# Stage 5: Development image
FROM app AS development

# Install development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook \
    pre-commit \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov

# Install VS Code server for remote development
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Expose additional development ports
EXPOSE 8888 8080 8081

# Development entrypoint
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 6: Training image
FROM python-env AS training

# Copy training scripts and data
COPY --chown=mlops:mlops mlops/training/ /home/mlops/training/
COPY --chown=mlops:mlops mlops/data/ /home/mlops/data/
WORKDIR /home/mlops/training

# Install additional training dependencies
RUN pip install \
    wandb \
    mlflow \
    optuna \
    ray[tune] \
    pytorch-lightning \
    transformers[torch] \
    datasets

# Set up MLflow tracking
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV WANDB_MODE=offline

# Training entrypoint
CMD ["python", "train_summarization.py"]

# Stage 7: Inference image
FROM python-env AS inference

# Copy inference code
COPY --chown=mlops:mlops mlops/inference/ /home/mlops/inference/
COPY --chown=mlops:mlops mlops/models/ /home/mlops/models/
WORKDIR /home/mlops/inference

# Install inference-specific dependencies
RUN pip install \
    fastapi \
    uvicorn \
    gunicorn \
    celery \
    redis \
    prometheus-client

# Optimize for inference
RUN pip install \
    onnxruntime-gpu \
    tensorrt \
    torch-tensorrt

# Expose inference port
EXPOSE 8000

# Inference entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 8: Monitoring image
FROM python-env AS monitoring

# Copy monitoring code
COPY --chown=mlops:mlops mlops/monitoring/ /home/mlops/monitoring/
WORKDIR /home/mlops/monitoring

# Install monitoring dependencies
RUN pip install \
    prometheus-client \
    grafana-api \
    elasticsearch \
    kibana \
    jaeger-client \
    opentelemetry-api \
    opentelemetry-sdk

# Expose monitoring ports
EXPOSE 9090 3000 5601 16686

# Monitoring entrypoint
CMD ["python", "metrics_collector.py"]

# Stage 9: Edge deployment image
FROM python-env AS edge

# Copy edge computing code
COPY --chown=mlops:mlops mlops/edge/ /home/mlops/edge/
WORKDIR /home/mlops/edge

# Install edge-specific dependencies
RUN pip install \
    tensorflow-lite \
    onnxruntime \
    openvino \
    coremltools

# Optimize for edge deployment
RUN pip install \
    quantization \
    pruning \
    distillation

# Edge entrypoint
CMD ["python", "edge_deployment.py"]

# Stage 10: Blockchain image
FROM python-env AS blockchain

# Copy blockchain code
COPY --chown=mlops:mlops mlops/blockchain/ /home/mlops/blockchain/
WORKDIR /home/mlops/blockchain

# Install blockchain dependencies
RUN pip install \
    web3 \
    cryptography \
    hashlib \
    ecdsa \
    pycryptodome

# Blockchain entrypoint
CMD ["python", "news_verification.py"]