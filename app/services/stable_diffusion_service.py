"""Local Stable Diffusion inpainting service with optional IP-Adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageFilter, ImageOps

from app.config import RESULT_DIR, get_config


class StableDiffusionError(RuntimeError):
    """Raised when local Stable Diffusion generation cannot run."""


@dataclass(frozen=True)
class StableDiffusionResult:
    result_image_path: Path
    base_image_path: Path
    mask_image_path: Path
    prompt: str
    negative_prompt: str


_PIPELINE = None
_IP_ADAPTER_LOADED = False


def _fit_image(image: Image.Image, size: tuple[int, int], background: tuple[int, int, int] = (245, 245, 245)) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _largest_component(mask_array):
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_array

    largest = max(contours, key=cv2.contourArea)
    output = np.zeros_like(mask_array)
    cv2.drawContours(output, [largest], -1, 255, thickness=cv2.FILLED)
    return output


def _segment_foreground_grabcut(image: Image.Image):
    import cv2
    import numpy as np

    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    margin_x = max(4, round(width * 0.04))
    margin_y = max(4, round(height * 0.04))
    rect = (margin_x, margin_y, width - margin_x * 2, height - margin_y * 2)

    mask = np.zeros((height, width), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(rgb, mask, rect, bg_model, fg_model, 4, cv2.GC_INIT_WITH_RECT)
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = np.ones((9, 9), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component(foreground)


def _bbox_from_mask(mask_array) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.where(mask_array > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _make_inpaint_mask(base_image: Image.Image) -> Image.Image:
    """Create a dog-aware torso mask from foreground segmentation and bbox proportions."""

    import cv2
    import numpy as np

    dog_mask = _segment_foreground_grabcut(base_image)
    bbox = _bbox_from_mask(dog_mask)
    if bbox is None:
        width, height = base_image.size
        fallback = np.zeros((height, width), dtype=np.uint8)
        fallback[round(height * 0.28) : round(height * 0.74), round(width * 0.14) : round(width * 0.82)] = 255
        return Image.fromarray(fallback, mode="L").filter(ImageFilter.GaussianBlur(radius=12))

    x1, y1, x2, y2 = bbox
    dog_width = max(1, x2 - x1)
    dog_height = max(1, y2 - y1)

    torso = np.zeros_like(dog_mask)
    torso_top = y1 + round(dog_height * 0.26)
    torso_bottom = y1 + round(dog_height * 0.72)
    torso_left = x1 + round(dog_width * 0.08)
    torso_right = x1 + round(dog_width * 0.88)
    torso[torso_top:torso_bottom, torso_left:torso_right] = 255

    # Add extra shoulder/chest coverage on the dog's front side, which is where harness straps sit.
    front_left = x1 + round(dog_width * 0.42)
    front_right = x1 + round(dog_width * 0.94)
    chest_top = y1 + round(dog_height * 0.24)
    chest_bottom = y1 + round(dog_height * 0.62)
    torso[chest_top:chest_bottom, front_left:front_right] = 255

    # Keep face/top head and lower legs protected.
    protected = np.zeros_like(dog_mask)
    protected[y1 : y1 + round(dog_height * 0.24), :] = 255
    protected[y1 + round(dog_height * 0.74) : y2, :] = 255

    torso = cv2.bitwise_and(torso, dog_mask)
    torso = cv2.bitwise_and(torso, cv2.bitwise_not(protected))

    kernel = np.ones((17, 17), dtype=np.uint8)
    torso = cv2.dilate(torso, kernel, iterations=1)
    torso = cv2.bitwise_and(torso, cv2.dilate(dog_mask, np.ones((11, 11), dtype=np.uint8), iterations=1))
    return Image.fromarray(torso, mode="L").filter(ImageFilter.GaussianBlur(radius=max(8, round(min(base_image.size) * 0.018))))


def _load_product_reference(harness_image_path: Path, image_size: int) -> Image.Image:
    with Image.open(harness_image_path) as harness_source:
        return _fit_image(harness_source, (image_size, image_size), background=(255, 255, 255))


def _load_pipeline():
    global _PIPELINE, _IP_ADAPTER_LOADED
    if _PIPELINE is not None:
        return _PIPELINE

    config = get_config()
    Path(config.sd_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", config.sd_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(config.sd_cache_dir) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(config.sd_cache_dir) / "transformers"))

    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise StableDiffusionError("Install local generation packages first: pip install -r requirements.txt") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config.sd_model_id,
            torch_dtype=dtype,
            cache_dir=config.sd_cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline = pipeline.to(device)
        if config.sd_use_ip_adapter:
            pipeline.load_ip_adapter(
                config.sd_ip_adapter_repo,
                subfolder=config.sd_ip_adapter_subfolder,
                weight_name=config.sd_ip_adapter_weight,
                cache_dir=config.sd_cache_dir,
            )
            pipeline.set_ip_adapter_scale(config.sd_ip_adapter_scale)
            _IP_ADAPTER_LOADED = True
    except Exception as exc:
        raise StableDiffusionError(f"Could not load Stable Diffusion model `{config.sd_model_id}`: {exc}") from exc

    _PIPELINE = pipeline
    return _PIPELINE


def generate_dog_harness_image(dog_image_path: Path, harness_image_path: Path, prompt: str) -> StableDiffusionResult:
    """Inpaint only the dog's torso while using the harness as product reference."""

    config = get_config()
    image_size = max(512, config.sd_image_size)
    image_size = image_size - (image_size % 8)
    user_prompt = (prompt or config.sd_prompt).strip() or "a dog is wearing the harness"
    generation_prompt = (
        "same dog as the input photo, same breed, same face, same ears, same eyes, same black tan and white fur, "
        "same body shape and same standing pose, preserve the original background, "
        "only edit the neck, chest, shoulders and torso so the dog is wearing the harness, "
        f"{user_prompt}, the harness should match the product reference image: olive green fabric body, black straps, "
        "black buckles, black handle, chest harness, realistic fit, natural fur occlusion, realistic shadows"
    )

    with Image.open(dog_image_path) as dog_source:
        base_image = _fit_image(dog_source, (image_size, image_size))

    mask_image = _make_inpaint_mask(base_image)
    product_reference = _load_product_reference(harness_image_path, image_size)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base_path = RESULT_DIR / f"inpaint_base_{uuid4().hex}.png"
    mask_path = RESULT_DIR / f"inpaint_mask_{uuid4().hex}.png"
    base_image.save(base_path, format="PNG")
    mask_image.save(mask_path, format="PNG")

    pipeline = _load_pipeline()

    try:
        import torch

        generator = None
        if config.sd_seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(config.sd_seed)

        kwargs = {
            "prompt": generation_prompt,
            "negative_prompt": config.sd_negative_prompt,
            "image": base_image,
            "mask_image": mask_image,
            "strength": config.sd_strength,
            "guidance_scale": config.sd_guidance_scale,
            "num_inference_steps": config.sd_steps,
            "generator": generator,
        }
        if _IP_ADAPTER_LOADED:
            kwargs["ip_adapter_image"] = product_reference

        output = pipeline(**kwargs).images[0]
    except Exception as exc:
        raise StableDiffusionError(f"Stable Diffusion generation failed: {exc}") from exc

    result_path = RESULT_DIR / f"dog_wearing_harness_{uuid4().hex}.png"
    output.save(result_path, format="PNG")
    return StableDiffusionResult(
        result_image_path=result_path,
        base_image_path=base_path,
        mask_image_path=mask_path,
        prompt=generation_prompt,
        negative_prompt=config.sd_negative_prompt,
    )
