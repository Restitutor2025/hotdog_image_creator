"""High-level orchestration for local dog outfit generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import RESULT_DIR, get_config
from app.services.model_backend_service import GenerationBackendRequest, select_generation_backend


@dataclass(frozen=True)
class ImageGenerationResult:
    result_image_path: Path
    backend: str
    message: str


def generate_dog_outfit_preview(
    dog_image_path: Path,
    product_image_path: Path,
    mask_image_path: Path | None,
    product_type: str,
    generation_prompt: str,
    negative_prompt: str,
    use_comfyui: bool,
) -> ImageGenerationResult:
    """Generate a dog wearing a product through the configured local backend."""

    config = get_config()
    backend = select_generation_backend(config, use_comfyui=use_comfyui)
    request = GenerationBackendRequest(
        dog_image_path=dog_image_path,
        product_image_path=product_image_path,
        mask_image_path=mask_image_path,
        product_type=product_type,
        generation_prompt=generation_prompt,
        negative_prompt=negative_prompt,
        result_dir=RESULT_DIR,
    )
    result_path = backend.generate(request)
    return ImageGenerationResult(
        result_image_path=result_path,
        backend=backend.name,
        message="Generated a local image-editing preview with the configured backend.",
    )
