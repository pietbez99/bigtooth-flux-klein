"""
BigTooth - Flux 2 Klein Image Stylization Handler for RunPod Serverless

This handler transforms photos into various cartoon styles using FLUX.2 Klein 4B.
Supports: Pixar, Disney, Anime, Ghibli styles via image-to-image editing.

Model: black-forest-labs/FLUX.2-klein-4B (Apache 2.0 license)
Requires: diffusers from git (dev version with Flux2KleinPipeline support)
"""

import runpod
import torch
import requests
import base64
import io
from PIL import Image, ImageOps

# Global model reference (loaded once at startup)
pipe = None

def load_model():
    """Load FLUX.2 Klein 4B model into GPU memory."""
    global pipe

    if pipe is not None:
        return pipe

    print("Loading FLUX.2 Klein 4B model...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Import diffusers (must be dev version with Flux2KleinPipeline)
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")

    # FLUX.2 Klein uses a custom pipeline class (Flux2KleinPipeline)
    # DiffusionPipeline.from_pretrained with trust_remote_code should auto-detect it
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True  # Required for custom pipeline/components
    )
    pipe = pipe.to("cuda")

    # Enable memory optimizations if available
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()

    print(f"Model loaded successfully! Pipeline type: {type(pipe)}")

    # Log available parameters
    import inspect
    sig = inspect.signature(pipe.__call__)
    print(f"Available pipeline parameters: {list(sig.parameters.keys())}")

    return pipe

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
    image = Image.open(io.BytesIO(response.content))
    # Apply EXIF orientation to fix rotated images (common from phone cameras)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Handle data URI format (data:image/jpeg;base64,...)
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    # Apply EXIF orientation to fix rotated images (common from phone cameras)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")

def handler(job):
    """
    RunPod handler function for image stylization.

    Input:
    {
        "input": {
            "image_url": "https://...",   # URL of image to transform (OR use image_base64)
            "image_base64": "...",        # Base64 encoded image (alternative to URL)
            "style": "pixar",             # Style: pixar, disney, anime, ghibli, cartoon
            "num_inference_steps": 4,     # Steps (default 4 for Klein - it's fast)
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
        image_base64_input = job_input.get("image_base64")
        style = job_input.get("style", "pixar").lower()
        num_inference_steps = job_input.get("num_inference_steps", 4)
        guidance_scale = job_input.get("guidance_scale", 3.5)
        seed = job_input.get("seed", -1)

        if not image_url and not image_base64_input:
            return {"error": "image_url or image_base64 is required", "success": False}

        # Load model (cached after first call)
        model = load_model()

        # Get source image from URL or base64
        if image_base64_input:
            print("Loading image from base64 input...")
            source_image = base64_to_image(image_base64_input)
        else:
            print(f"Downloading image from: {image_url[:80]}...")
            source_image = download_image(image_url)

        original_size = source_image.size
        print(f"Original image size: {original_size}")

        # Resize to 512x512 to match Wavespeed output
        source_image = source_image.resize((512, 512), Image.Resampling.LANCZOS)

        # Get style prompt
        prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["pixar"])

        # Set random seed
        import random
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"Generating {style} style image with prompt: {prompt}")

        # Try image-to-image with input_images parameter (Flux2KleinPipeline style)
        # The pipeline accepts input_images as a list of PIL Images for reference
        try:
            result = model(
                prompt=prompt,
                input_images=[source_image],  # Pass as reference image
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=512,
                width=512
            ).images[0]
            print("Used img2img with input_images parameter")
        except TypeError as e:
            # Fallback: try with 'image' parameter (standard img2img)
            print(f"input_images not supported, trying 'image' parameter: {e}")
            try:
                result = model(
                    prompt=prompt,
                    image=source_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=512,
                    width=512
                ).images[0]
                print("Used img2img with image parameter")
            except TypeError as e2:
                # Final fallback: text-to-image only
                print(f"image parameter not supported, using text-to-image only: {e2}")
                result = model(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=512,
                    width=512
                ).images[0]
                print("Used text-to-image (no input image)")

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

# Start RunPod serverless handler
print("Initializing FLUX.2 Klein stylization endpoint...")
runpod.serverless.start({"handler": handler})
