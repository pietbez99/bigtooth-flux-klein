"""
BigTooth - Flux 2 Klein Image Stylization Handler for RunPod Serverless

This handler transforms photos into various cartoon styles using FLUX.2 Klein 4B.
Supports: Pixar, Disney, Anime, Ghibli styles via image-to-image editing.

Features:
- Automatic face detection using MediaPipe (GPU accelerated)
- Face cropping with margin for better stylization
- Safe prompts that focus on face when detected

Model: black-forest-labs/FLUX.2-klein-4B (Apache 2.0 license)
Requires: diffusers from git (dev version with Flux2KleinPipeline support)
"""

import runpod
import torch
import requests
import base64
import io
from PIL import Image, ImageOps

# Global model references (loaded once at startup)
pipe = None
face_detector = None

def load_face_detector():
    """Load MediaPipe face detector."""
    global face_detector

    if face_detector is not None:
        return face_detector

    print("Loading MediaPipe face detector...")
    import mediapipe as mp

    # Use MediaPipe Face Detection (fast, GPU accelerated)
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,  # 1 = full range model (better for various distances)
        min_detection_confidence=0.5
    )

    print("Face detector loaded successfully!")
    return face_detector

def detect_and_crop_face(image: Image.Image, margin_percent: float = 0.3) -> tuple:
    """
    Detect face in image and crop to face with margin.

    Args:
        image: PIL Image
        margin_percent: How much margin to add around face (0.3 = 30%)

    Returns:
        tuple: (cropped_image, face_detected)
        - If face found: returns cropped image and True
        - If no face: returns original image and False
    """
    import numpy as np

    detector = load_face_detector()

    # Convert PIL to RGB numpy array for MediaPipe
    img_array = np.array(image)

    # Detect faces
    results = detector.process(img_array)

    if not results.detections or len(results.detections) == 0:
        print("No face detected - using original image")
        return image, False

    # Get the first (most confident) face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    img_width, img_height = image.size

    # Convert relative coordinates to absolute pixels
    x = int(bbox.xmin * img_width)
    y = int(bbox.ymin * img_height)
    w = int(bbox.width * img_width)
    h = int(bbox.height * img_height)

    # Add margin around face
    margin_x = int(w * margin_percent)
    margin_y = int(h * margin_percent)

    # Calculate crop box with margin, clamped to image bounds
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(img_width, x + w + margin_x)
    bottom = min(img_height, y + h + margin_y)

    # Make it square (take the larger dimension)
    crop_width = right - left
    crop_height = bottom - top

    if crop_width > crop_height:
        # Expand height
        diff = crop_width - crop_height
        top = max(0, top - diff // 2)
        bottom = min(img_height, bottom + diff // 2)
    elif crop_height > crop_width:
        # Expand width
        diff = crop_height - crop_width
        left = max(0, left - diff // 2)
        right = min(img_width, right + diff // 2)

    # Crop to face
    cropped = image.crop((left, top, right, bottom))

    print(f"Face detected at ({x}, {y}, {w}, {h}) - cropped to ({left}, {top}, {right}, {bottom})")

    return cropped, True

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

# Style prompts - different versions for face vs no-face images
# Face prompts focus on the face and avoid body details
STYLE_PROMPTS_FACE = {
    "pixar": "Transform this portrait into Pixar 3D animation style, focus on the face, friendly cartoon character",
    "disney": "Transform this portrait into Disney 2D animation style, focus on the face, friendly cartoon character",
    "anime": "Transform this portrait into anime style, focus on the face, friendly character",
    "ghibli": "Transform this portrait into Studio Ghibli watercolor style, focus on the face, gentle character",
    "cartoon": "Transform this portrait into cartoon style, focus on the face, friendly character"
}

# Generic prompts for when no face is detected
STYLE_PROMPTS_GENERIC = {
    "pixar": "Transform this image into Pixar 3D animation style",
    "disney": "Transform this image into Disney 2D animation style",
    "anime": "Transform this image into anime style",
    "ghibli": "Transform this image into Studio Ghibli watercolor style",
    "cartoon": "Transform this image into cartoon style"
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
            "prompt": "...",              # Custom prompt (optional - overrides style)
            "style": "pixar",             # Style: pixar, disney, anime, ghibli, cartoon (fallback if no prompt)
            "num_inference_steps": 4,     # Steps (default 4 for Klein - it's fast)
            "guidance_scale": 3.5,        # Guidance scale (default 3.5)
            "seed": -1,                   # Random seed (-1 for random)
            "skip_face_detection": false  # Skip face detection (default false)
        }
    }

    Output:
    {
        "image_base64": "...",           # Base64 encoded result image
        "style_used": "pixar",
        "prompt_used": "...",            # The actual prompt that was used
        "face_detected": true,           # Whether a face was detected and cropped
        "success": true
    }
    """
    try:
        job_input = job["input"]

        # Extract parameters
        image_url = job_input.get("image_url")
        image_base64_input = job_input.get("image_base64")
        custom_prompt = job_input.get("prompt")  # Custom prompt takes priority
        style = job_input.get("style", "pixar").lower()
        num_inference_steps = job_input.get("num_inference_steps", 4)
        guidance_scale = job_input.get("guidance_scale", 3.5)
        seed = job_input.get("seed", -1)
        skip_face_detection = job_input.get("skip_face_detection", False)

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

        # Face detection and cropping
        face_detected = False
        if not skip_face_detection:
            print("Running face detection...")
            source_image, face_detected = detect_and_crop_face(source_image, margin_percent=0.4)
            if face_detected:
                print(f"Face cropped, new size: {source_image.size}")

        # Resize to 512x512 for FLUX
        source_image = source_image.resize((512, 512), Image.Resampling.LANCZOS)

        # Get prompt - custom prompt takes priority, then face/generic style prompts
        if custom_prompt:
            prompt = custom_prompt
            print(f"Using custom prompt: {prompt[:80]}...")
        elif face_detected:
            prompt = STYLE_PROMPTS_FACE.get(style, STYLE_PROMPTS_FACE["pixar"])
            print(f"Using face-focused {style} prompt: {prompt}")
        else:
            prompt = STYLE_PROMPTS_GENERIC.get(style, STYLE_PROMPTS_GENERIC["pixar"])
            print(f"Using generic {style} prompt: {prompt}")

        # Set random seed
        import random
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"Generating {style} style image (face_detected={face_detected})")

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

        print(f"Successfully generated {style} style image (face_detected={face_detected})")

        return {
            "image_base64": result_base64,
            "style_used": style,
            "prompt_used": prompt,
            "seed_used": seed,
            "face_detected": face_detected,
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
print("Initializing FLUX.2 Klein stylization endpoint with face detection...")
runpod.serverless.start({"handler": handler})
