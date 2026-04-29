"""Mask helpers reserved for local image-editing workflows."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFilter

from app.config import UPLOAD_DIR


def create_rough_torso_mask(dog_image_path: Path) -> Image.Image:
    """Create a soft central body mask for future inpainting workflows."""

    with Image.open(dog_image_path) as source_image:
        width, height = source_image.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    left = round(width * 0.22)
    top = round(height * 0.22)
    right = round(width * 0.78)
    bottom = round(height * 0.78)
    draw.ellipse((left, top, right, bottom), fill=210)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(12, round(min(width, height) * 0.025))))


def save_rough_torso_mask(dog_image_path: Path) -> Path:
    """Save the placeholder mask under static/uploads and return its path."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = UPLOAD_DIR / f"rough_torso_mask_{uuid4().hex}.png"
    create_rough_torso_mask(dog_image_path).save(mask_path, format="PNG")
    return mask_path
