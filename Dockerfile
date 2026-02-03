# BigTooth - FLUX.2 Klein Stylization RunPod Endpoint
# Transforms photos into Pixar/Disney/Anime/Ghibli styles
# Model: black-forest-labs/FLUX.2-klein-4B (Apache 2.0 license)
# Requires diffusers dev version for Flux2KleinPipeline support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install dependencies
# IMPORTANT: diffusers must be installed from git (dev version) for FLUX.2 Klein support
# The model requires Flux2KleinPipeline which is only in diffusers >= 0.37.0.dev0
# PyTorch 2.5+ required for enable_gqa in scaled_dot_product_attention
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.1.99 \
    pillow>=10.0.0 \
    requests>=2.31.0 \
    runpod>=1.6.0 \
    mediapipe>=0.10.0

# Copy handler
COPY handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "/app/handler.py"]
