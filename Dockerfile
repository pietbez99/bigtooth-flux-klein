# BigTooth - FLUX.2 Klein Stylization RunPod Endpoint
# Transforms photos into Pixar/Disney/Anime/Ghibli styles
# Model: black-forest-labs/FLUX.2-klein-4B (Apache 2.0 license)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install diffusers and other dependencies
# Using latest diffusers for FLUX.2 Klein support
RUN pip3 install --no-cache-dir \
    diffusers>=0.31.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    pillow>=10.0.0 \
    requests>=2.31.0 \
    runpod>=1.6.0

# Create app directory
WORKDIR /app

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build (optional - increases image size but faster cold starts)
# Uncomment to pre-download FLUX.2 Klein 4B (~24GB download):
# RUN python3 -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-4B', torch_dtype='bfloat16')"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
