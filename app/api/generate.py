"""Primary local image generation route for dog clothes and harness previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.config import STATIC_DIR, UPLOAD_DIR
from app.services.generation_prompt_service import build_generation_prompt, load_negative_prompt
from app.services.image_generation_service import generate_dog_outfit_preview
from app.services.mask_service import save_rough_torso_mask
from app.services.model_backend_service import BACKEND_UNAVAILABLE_MESSAGE, GenerationBackendUnavailableError


router = APIRouter(prefix="/preview", tags=["generation"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_PRODUCT_TYPES = {"harness", "clothes"}


def _error_response(status_code: int, stage: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "failed",
            "stage": stage,
            "message": message,
        },
    )


def _safe_extension(filename: str | None, content_type: str | None) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension in ALLOWED_EXTENSIONS:
        return extension
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/webp":
        return ".webp"
    return ".png"


async def _save_upload(upload_file: UploadFile, prefix: str) -> Path:
    if upload_file.content_type and not upload_file.content_type.startswith("image/"):
        raise ValueError(f"{prefix} must be an image file")

    data = await upload_file.read()
    if not data:
        raise ValueError(f"{prefix} file is empty")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    extension = _safe_extension(upload_file.filename, upload_file.content_type)
    destination = UPLOAD_DIR / f"{prefix}_{uuid4().hex}{extension}"
    destination.write_bytes(data)
    return destination


def _public_static_path(path: Path) -> str:
    return f"/static/{path.relative_to(STATIC_DIR).as_posix()}"


@router.post("/generate", response_model=None)
async def generate_pet_product_preview(
    dog_image: UploadFile = File(...),
    product_image: UploadFile = File(...),
    product_type: str = Form(...),
    prompt_mode: str = Form(default="wearing_preview"),
    use_comfyui: bool = Form(default=True),
) -> Any:
    """Generate a realistic local dog outfit preview through a model backend."""

    normalized_product_type = product_type.strip().lower()
    if normalized_product_type not in SUPPORTED_PRODUCT_TYPES:
        return _error_response(400, "validation", "product_type must be harness or clothes.")

    try:
        dog_path = await _save_upload(dog_image, "dog")
        product_path = await _save_upload(product_image, "product")
    except ValueError as exc:
        return _error_response(400, "validation", str(exc))

    generation_prompt = build_generation_prompt(normalized_product_type, prompt_mode)
    negative_prompt = load_negative_prompt()
    rough_mask_path = save_rough_torso_mask(dog_path)

    try:
        generation_result = generate_dog_outfit_preview(
            dog_image_path=dog_path,
            product_image_path=product_path,
            mask_image_path=rough_mask_path,
            product_type=normalized_product_type,
            generation_prompt=generation_prompt,
            negative_prompt=negative_prompt,
            use_comfyui=use_comfyui,
        )
    except GenerationBackendUnavailableError:
        return _error_response(501, "model_backend", BACKEND_UNAVAILABLE_MESSAGE)

    return {
        "status": "success",
        "result_image_path": _public_static_path(generation_result.result_image_path),
        "product_type": normalized_product_type,
        "generation_prompt": generation_prompt,
        "negative_prompt": negative_prompt,
        "backend": generation_result.backend,
        "message": generation_result.message,
    }
