"""
BigTooth - Flux 2 Klein Image Stylization Handler for RunPod Serverless

This handler transforms photos into various cartoon styles using FLUX.2 Klein 4B.
Supports: Pixar, Disney, Anime, Ghibli styles via prompt-based image editing.

Model: black-forest-labs/FLUX.2-klein-4B (Apache 2.0 license)
"""

import runpod
import torch
import requests
import base64
import io
from PIL import Image
from diffusers import FluxPriorReduxPipeline, FluxPipeline

# Global model references (loaded once at startup)
pipe = None
pipe_prior = None

def load_model():
    """Load FLUX.2 Klein 4B model into GPU memory."""
    global pipe, pipe_prior

    if pipe is not None:
        return pipe, pipe_prior

    print("Loading FLUX.2 Klein 4B model...")

    # Load the main pipeline for image-to-image editing
    # FLUX.2 Klein supports unified generation and editing
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    # Enable memory optimizations
    pipe.enable_attention_slicing()

    # Try to load the prior redux for image conditioning (if available)
    try:
        pipe_prior = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16
        )
        pipe_prior = pipe_prior.to("cuda")
        print("Prior redux pipeline loaded for image conditioning")
    except Exception as e:
        print(f"Prior redux not available, using standard img2img: {e}")
        pipe_prior = None

    print("Model loaded successfully!")
    return pipe, pipe_prior

# Style prompt templates - matching Wavespeed format
STYLE_PROMPTS = {
    "pixar": "Turn this image into pixar style. crop to point of interest",
    "disney": "Turn this image into 2D disney style. crop to point of interest",
    "anime": "Turn this image into anime style. crop to point of interest",
    "ghibli": "Turn this image into ghibli style. crop to point of interest",
    "cartoon": "Turn this image into cartoon style. crop to point of interest"
}

def download_image(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(job):
    """
    RunPod handler function for image stylization.

    Input:
    {
        "input": {
            "image_url": "https://...",  # URL of image to transform
            "style": "pixar",             # Style: pixar, disney, anime, ghibli, cartoon
            "strength": 0.75,             # How much to transform (0.0-1.0, default 0.75)
            "num_inference_steps": 28,    # Steps (default 28 for Klein)
            "guidance_scale": 3.5,        # Guidance scale (default 3.5)
            "seed": -1                    # Random seed (-1 for random)
        }
    }

    Output:
    {
        "image_base64": "...",           # Base64 encoded result image
        "style_used": "pixar",
        "success": true
    }
    """
    try:
        job_input = job["input"]

        # Extract parameters
        image_url = job_input.get("image_url")
        style = job_input.get("style", "pixar").lower()
        strength = job_input.get("strength", 0.75)
        num_inference_steps = job_input.get("num_inference_steps", 28)
        guidance_scale = job_input.get("guidance_scale", 3.5)
        seed = job_input.get("seed", -1)

        if not image_url:
            return {"error": "image_url is required", "success": False}

        # Load model (cached after first call)
        model, prior = load_model()

        # Download source image
        print(f"Downloading image from: {image_url[:50]}...")
        source_image = download_image(image_url)

        # Resize to 512x512 to match Wavespeed output
        source_image = source_image.resize((512, 512), Image.Resampling.LANCZOS)

        # Get style prompt
        prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["pixar"])

        # Set random seed if specified
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            # Use random seed
            import random
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"Generating {style} style image with prompt: {prompt}")

        # Generate styled image using img2img approach
        result = model(
            prompt=prompt,
            image=source_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=512,
            width=512
        ).images[0]

        # Convert to base64
        result_base64 = image_to_base64(result)

        print(f"Successfully generated {style} style image")

        return {
            "image_base64": result_base64,
            "style_used": style,
            "seed_used": seed,
            "success": True
        }

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "success": False
        }

# Load model at startup (keeps it in memory for fast inference)
print("Initializing FLUX.2 Klein stylization endpoint...")
load_model()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
