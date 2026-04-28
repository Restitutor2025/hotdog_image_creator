"""Preview API routes for uploading images and creating a local composition.

Process overview:
1. Receive a dog photo and product image through FastAPI UploadFile fields.
2. Optionally receive placement JSON as a multipart form field.
3. Save raw uploads under static/uploads for debugging and repeatability.
4. Remove the product background with the local rembg package.
5. Composite the unchanged product pixels onto the dog photo with PIL.
6. Return the generated result path, image dimensions, and fixed prompt text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.background_remove_service import remove_product_background
from app.services.composite_service import composite_product_on_dog
from app.services.prompt_service import load_pet_product_preview_prompt


router = APIRouter(prefix="/preview", tags=["preview"])

BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _safe_extension(filename: str | None, content_type: str | None) -> str:
    """Choose a safe image extension without trusting arbitrary upload names."""

    extension = Path(filename or "").suffix.lower()
    if extension in ALLOWED_EXTENSIONS:
        return extension

    # Fall back to the content type when the filename is missing or has no suffix.
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/webp":
        return ".webp"

    return ".png"


async def _save_upload(upload_file: UploadFile, prefix: str) -> Path:
    """Save an uploaded image to static/uploads and return its local path."""

    if upload_file.content_type and not upload_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{prefix} must be an image file")

    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{prefix} file is empty")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    extension = _safe_extension(upload_file.filename, upload_file.content_type)
    destination = UPLOAD_DIR / f"{prefix}_{uuid4().hex}{extension}"
    destination.write_bytes(data)
    return destination


def _parse_placement(placement: str | None) -> dict[str, float]:
    """Parse optional placement JSON and validate the supported numeric fields."""

    if not placement:
        return {}

    try:
        raw_data: Any = json.loads(placement)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="placement must be valid JSON") from exc

    if not isinstance(raw_data, dict):
        raise HTTPException(status_code=400, detail="placement must be a JSON object")

    parsed: dict[str, float] = {}
    for key in ("x", "y", "scale", "opacity"):
        if key not in raw_data or raw_data[key] is None:
            continue

        try:
            parsed[key] = float(raw_data[key])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"placement.{key} must be a number") from exc

    if "scale" in parsed and parsed["scale"] <= 0:
        raise HTTPException(status_code=400, detail="placement.scale must be greater than 0")

    if "opacity" in parsed and not 0 <= parsed["opacity"] <= 1:
        raise HTTPException(status_code=400, detail="placement.opacity must be between 0 and 1")

    return parsed


@router.post("/composite")
async def create_pet_product_preview(
    dog_image: UploadFile = File(...),
    product_image: UploadFile = File(...),
    placement: str | None = Form(default=None),
) -> dict[str, int | str]:
    """Create a pet shopping preview by compositing the product onto the dog photo."""

    placement_data = _parse_placement(placement)
    dog_path = await _save_upload(dog_image, "dog")
    product_path = await _save_upload(product_image, "product")

    try:
        product_without_background = remove_product_background(product_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result_path, width, height = composite_product_on_dog(
        dog_path=dog_path,
        product_image=product_without_background,
        placement=placement_data,
        result_dir=RESULT_DIR,
    )

    fixed_prompt = load_pet_product_preview_prompt()
    public_result_path = f"/static/results/{result_path.name}"

    return {
        "result_image_path": public_result_path,
        "width": width,
        "height": height,
        "fixed_prompt": fixed_prompt,
    }


# Example multipart field: placement={"x":120,"y":180,"scale":0.8,"opacity":0.95}
