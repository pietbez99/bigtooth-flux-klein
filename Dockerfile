# BigTooth - FLUX.2 Klein Stylization RunPod Endpoint
# Transforms photos into Pixar/Disney/Anime/Ghibli styles
# Model: black-forest-labs/FLUX.2-klein-4B (Apache 2.0 license)

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    diffusers>=0.31.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.1.99 \
    pillow>=10.0.0 \
    requests>=2.31.0 \
    runpod>=1.6.0

# Copy handler
COPY handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "/app/handler.py"]
