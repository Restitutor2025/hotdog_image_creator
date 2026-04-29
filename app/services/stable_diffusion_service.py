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
    mask_overlay_path: Path
    product_reference_path: Path
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


def _center_of_mask_band(mask_array, y_top: int, y_bottom: int) -> tuple[int, int] | None:
    import numpy as np

    band = mask_array[max(0, y_top) : min(mask_array.shape[0], y_bottom), :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean() + max(0, y_top))


def _make_inpaint_mask(base_image: Image.Image) -> Image.Image:
    """Create narrow harness strap regions from dog foreground geometry."""

    import cv2
    import numpy as np

    dog_mask = _segment_foreground_grabcut(base_image)
    bbox = _bbox_from_mask(dog_mask)
    if bbox is None:
        width, height = base_image.size
        fallback = np.zeros((height, width), dtype=np.uint8)
        cv2.line(
            fallback,
            (round(width * 0.42), round(height * 0.28)),
            (round(width * 0.42), round(height * 0.58)),
            255,
            max(18, round(height * 0.035)),
        )
        cv2.line(
            fallback,
            (round(width * 0.22), round(height * 0.42)),
            (round(width * 0.68), round(height * 0.42)),
            255,
            max(18, round(height * 0.035)),
        )
        return Image.fromarray(fallback, mode="L").filter(ImageFilter.GaussianBlur(radius=12))

    x1, y1, x2, y2 = bbox
    dog_width = max(1, x2 - x1)
    dog_height = max(1, y2 - y1)

    body_center = _center_of_mask_band(
        dog_mask,
        y1 + round(dog_height * 0.36),
        y1 + round(dog_height * 0.66),
    )
    head_center = _center_of_mask_band(
        dog_mask,
        y1 + round(dog_height * 0.08),
        y1 + round(dog_height * 0.34),
    )
    body_x = body_center[0] if body_center else x1 + dog_width // 2
    head_x = head_center[0] if head_center else body_x
    front_sign = 1 if head_x >= body_x else -1

    front_x = x1 + round(dog_width * (0.70 if front_sign > 0 else 0.30))
    rear_x = x1 + round(dog_width * (0.30 if front_sign > 0 else 0.70))
    mid_x = x1 + round(dog_width * 0.50)
    neck_y = y1 + round(dog_height * 0.29)
    shoulder_y = y1 + round(dog_height * 0.38)
    chest_y = y1 + round(dog_height * 0.50)
    side_y = y1 + round(dog_height * 0.56)
    strap_thickness = max(14, round(dog_height * 0.035))

    harness_mask = np.zeros_like(dog_mask)

    # Neck ring, upper shoulder strap, upper chest strap, and side/body strap only.
    cv2.ellipse(
        harness_mask,
        (front_x, neck_y),
        (max(18, round(dog_width * 0.11)), max(10, round(dog_height * 0.035))),
        0,
        0,
        360,
        210,
        thickness=max(10, round(strap_thickness * 0.75)),
    )
    cv2.line(harness_mask, (front_x, shoulder_y), (rear_x, shoulder_y + round(dog_height * 0.04)), 255, strap_thickness)
    cv2.line(harness_mask, (front_x, shoulder_y), (front_x, chest_y + round(dog_height * 0.08)), 255, strap_thickness)
    cv2.line(harness_mask, (front_x, chest_y), (mid_x - front_sign * round(dog_width * 0.10), side_y), 255, strap_thickness)
    cv2.line(
        harness_mask,
        (mid_x - front_sign * round(dog_width * 0.08), side_y),
        (rear_x, y1 + round(dog_height * 0.54)),
        255,
        max(10, round(strap_thickness * 0.8)),
    )

    # Keep face, ears, eyes, mouth, lower legs, paws, tail, and most fur protected.
    protected = np.zeros_like(dog_mask)
    protected[y1 : y1 + round(dog_height * 0.25), :] = 255
    protected[y1 + round(dog_height * 0.66) : y2, :] = 255
    protected[:, x1 : x1 + round(dog_width * 0.08)] = 255
    protected[:, x1 + round(dog_width * 0.92) : x2] = 255

    dog_area = cv2.dilate(dog_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    harness_mask = cv2.bitwise_and(harness_mask, dog_area)
    harness_mask = cv2.bitwise_and(harness_mask, cv2.bitwise_not(protected))

    kernel = np.ones((7, 7), dtype=np.uint8)
    harness_mask = cv2.dilate(harness_mask, kernel, iterations=1)
    return Image.fromarray(harness_mask, mode="L").filter(
        ImageFilter.GaussianBlur(radius=max(4, round(min(base_image.size) * 0.008)))
    )


def _make_mask_overlay(base_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = mask_image.convert("L").point(lambda value: round(value * 0.55))
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay)


def _crop_product_to_object(product_image: Image.Image, padding_ratio: float = 0.08) -> Image.Image:
    import numpy as np

    rgba = product_image.convert("RGBA")
    array = np.array(rgba)
    rgb = array[:, :, :3].astype(np.int16)
    alpha = array[:, :, 3]

    not_transparent = alpha > 12
    distance_from_white = np.abs(255 - rgb).max(axis=2)
    not_white_background = distance_from_white > 18
    object_mask = not_transparent & not_white_background

    ys, xs = np.where(object_mask)
    if len(xs) == 0 or len(ys) == 0:
        return rgba

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad_x = round((x2 - x1) * padding_ratio)
    pad_y = round((y2 - y1) * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(rgba.width, x2 + pad_x)
    y2 = min(rgba.height, y2 + pad_y)
    return rgba.crop((x1, y1, x2, y2))


def _load_product_reference(harness_image_path: Path, image_size: int) -> Image.Image:
    with Image.open(harness_image_path) as harness_source:
        cropped = _crop_product_to_object(harness_source)
        reference_size = round(image_size * 0.86)
        fitted = ImageOps.contain(cropped, (reference_size, reference_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        x = (image_size - fitted.width) // 2
        y = (image_size - fitted.height) // 2
        if fitted.mode == "RGBA":
            canvas.paste(fitted, (x, y), fitted)
        else:
            canvas.paste(fitted.convert("RGB"), (x, y))
        return canvas


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
        "same dog as the input photo, same breed, same face, same ears, same eyes, same fur color and fur pattern, "
        "same body shape and same standing pose, preserve the original background, "
        "only edit the narrow harness contact areas around the neck, shoulders, upper chest, and side straps, "
        f"{user_prompt}, the harness must match the uploaded product reference image exactly, including its color, "
        "fabric, straps, buckles, handle, outline, and overall design, realistic fit, natural fur occlusion, realistic shadows"
    )

    with Image.open(dog_image_path) as dog_source:
        base_image = _fit_image(dog_source, (image_size, image_size))

    mask_image = _make_inpaint_mask(base_image)
    product_reference = _load_product_reference(harness_image_path, image_size)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base_path = RESULT_DIR / f"inpaint_base_{uuid4().hex}.png"
    mask_path = RESULT_DIR / f"inpaint_mask_{uuid4().hex}.png"
    overlay_path = RESULT_DIR / f"inpaint_mask_overlay_{uuid4().hex}.png"
    product_reference_path = RESULT_DIR / f"product_reference_{uuid4().hex}.png"
    base_image.save(base_path, format="PNG")
    mask_image.save(mask_path, format="PNG")
    _make_mask_overlay(base_image, mask_image).save(overlay_path, format="PNG")
    product_reference.save(product_reference_path, format="PNG")

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
        mask_overlay_path=overlay_path,
        product_reference_path=product_reference_path,
        prompt=generation_prompt,
        negative_prompt=config.sd_negative_prompt,
    )
