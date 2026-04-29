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
from PIL import Image, ImageDraw

from app.services.background_remove_service import remove_product_background
from app.services.composite_service import composite_product_on_dog
from app.services.dog_keypoint_service import estimate_dog_keypoints
from app.services.dog_segmentation_service import DogSegmentationError, segment_dog
from app.services.inpainting_refine_service import refine_harness_layer
from app.services.occlusion_service import build_occlusion_mask, composite_with_occlusion
from app.services.product_warp_service import warp_product_to_dog
from app.services.prompt_service import load_pet_product_preview_prompt


router = APIRouter(prefix="/preview", tags=["preview"])

BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
DEBUG_DIR = STATIC_DIR / "debug"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise HTTPException(status_code=400, detail=f"placement.{field_name} must be a boolean")


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


def _parse_placement(placement: str | None) -> dict[str, Any]:
    """Parse optional placement JSON and validate supported fitting fields."""

    if not placement:
        return {}

    try:
        raw_data: Any = json.loads(placement)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="placement must be valid JSON") from exc

    if not isinstance(raw_data, dict):
        raise HTTPException(status_code=400, detail="placement must be a JSON object")

    parsed: dict[str, Any] = {}
    for key in ("x", "y", "scale", "opacity", "warp_strength", "occlusion_strength"):
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

    for key in ("warp_strength", "occlusion_strength"):
        if key in parsed and not 0 <= parsed[key] <= 1:
            raise HTTPException(status_code=400, detail=f"placement.{key} must be between 0 and 1")

    if "mode" in raw_data and raw_data["mode"] is not None:
        mode = str(raw_data["mode"])
        if mode not in {"overlay", "fit_harness"}:
            raise HTTPException(status_code=400, detail='placement.mode must be "overlay" or "fit_harness"')
        parsed["mode"] = mode

    for key in ("auto_fit", "debug"):
        if key in raw_data and raw_data[key] is not None:
            parsed[key] = _parse_bool(raw_data[key], key)

    return parsed


def _public_static_path(path: Path) -> str:
    return f"/static/{path.relative_to(STATIC_DIR).as_posix()}"


def _save_result_image(image: Image.Image) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_DIR / f"pet_product_preview_{uuid4().hex}.png"
    image.save(result_path, format="PNG")
    return result_path


def _save_debug_image(run_id: str, name: str, image: Image.Image) -> str:
    debug_dir = DEBUG_DIR / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{name}.png"
    image.save(path, format="PNG")
    return _public_static_path(path)


def _draw_keypoints(base_image: Image.Image, keypoints: dict[str, list[int]]) -> Image.Image:
    visualization = base_image.convert("RGBA")
    draw = ImageDraw.Draw(visualization)
    for name, point in keypoints.items():
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 80, 20, 230), outline=(255, 255, 255, 255), width=2)
        draw.text((x + 7, y - 7), name, fill=(255, 255, 255, 255))
    return visualization


def _overlay_response(
    dog_path: Path,
    product_without_background: Image.Image,
    placement_data: dict[str, Any],
    fixed_prompt: str,
) -> dict[str, Any]:
    overlay_placement = {
        key: float(placement_data[key])
        for key in ("x", "y", "scale", "opacity")
        if key in placement_data
    }
    result_path, width, height = composite_product_on_dog(
        dog_path=dog_path,
        product_image=product_without_background,
        placement=overlay_placement,
        result_dir=RESULT_DIR,
    )

    return {
        "result_image_path": _public_static_path(result_path),
        "width": width,
        "height": height,
        "mode": "overlay",
        "keypoints": None,
        "debug_paths": None,
        "fixed_prompt": fixed_prompt,
    }


def _fit_harness_response(
    dog_path: Path,
    product_without_background: Image.Image,
    placement_data: dict[str, Any],
    fixed_prompt: str,
) -> dict[str, Any]:
    run_id = uuid4().hex
    debug_paths: dict[str, str] | None = {} if bool(placement_data.get("debug", False)) else None

    with Image.open(dog_path) as source_dog:
        base_image = source_dog.convert("RGBA")

    segmentation = segment_dog(dog_path)
    keypoints = estimate_dog_keypoints(segmentation.dog_mask, segmentation.dog_bbox)
    warped_product = warp_product_to_dog(
        product_image=product_without_background,
        canvas_size=base_image.size,
        keypoints=keypoints,
        placement=placement_data,
    )
    refined_product = refine_harness_layer(base_image, warped_product)
    occlusion_mask = build_occlusion_mask(
        dog_mask=segmentation.dog_mask,
        keypoints=keypoints,
        harness_layer=refined_product,
        occlusion_strength=float(placement_data.get("occlusion_strength", 0.45)),
    )
    final_image = composite_with_occlusion(base_image, refined_product, occlusion_mask)
    result_path = _save_result_image(final_image)

    if debug_paths is not None:
        debug_paths["dog_mask_image"] = _save_debug_image(run_id, "dog_mask", segmentation.dog_mask)
        debug_paths["keypoints_visualization"] = _save_debug_image(run_id, "keypoints", _draw_keypoints(base_image, keypoints))
        debug_paths["warped_product_image"] = _save_debug_image(run_id, "warped_product", refined_product)
        debug_paths["occlusion_mask"] = _save_debug_image(run_id, "occlusion_mask", occlusion_mask)
        debug_paths["final_result"] = _save_debug_image(run_id, "final_result", final_image)

    return {
        "result_image_path": _public_static_path(result_path),
        "width": base_image.width,
        "height": base_image.height,
        "mode": "fit_harness",
        "keypoints": keypoints,
        "debug_paths": debug_paths,
        "fixed_prompt": fixed_prompt,
    }


@router.post("/composite")
async def create_pet_product_preview(
    dog_image: UploadFile = File(...),
    product_image: UploadFile = File(...),
    placement: str | None = Form(default=None),
) -> dict[str, Any]:
    """Create a pet shopping preview by compositing the product onto the dog photo."""

    placement_data = _parse_placement(placement)
    dog_path = await _save_upload(dog_image, "dog")
    product_path = await _save_upload(product_image, "product")

    try:
        product_without_background = remove_product_background(product_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    fixed_prompt = load_pet_product_preview_prompt()
    mode = placement_data.get("mode", "overlay")

    if mode == "overlay":
        return _overlay_response(dog_path, product_without_background, placement_data, fixed_prompt)

    try:
        return _fit_harness_response(dog_path, product_without_background, placement_data, fixed_prompt)
    except DogSegmentationError:
        if bool(placement_data.get("auto_fit", True)):
            return _overlay_response(dog_path, product_without_background, placement_data, fixed_prompt)
        raise HTTPException(status_code=422, detail="fit_harness could not estimate a dog mask. Try overlay mode or a clearer dog photo.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"fit_harness failed: {exc}") from exc


# Example multipart field: placement={"mode":"fit_harness","auto_fit":true,"warp_strength":0.75,"debug":true}
