"""Pluggable image generation backend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig
from app.services.comfyui_client_service import ComfyUIClient, ComfyUIClientError


BACKEND_UNAVAILABLE_MESSAGE = (
    "No local image generation backend is configured or ComfyUI is not running. "
    "Start ComfyUI and configure COMFYUI_BASE_URL."
)


class GenerationBackendUnavailableError(RuntimeError):
    """Raised when no usable local generation backend is available."""


@dataclass(frozen=True)
class GenerationBackendRequest:
    dog_image_path: Path
    product_image_path: Path
    mask_image_path: Path | None
    product_type: str
    generation_prompt: str
    negative_prompt: str
    result_dir: Path


class ImageGenerationBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, request: GenerationBackendRequest) -> Path:
        """Generate or edit an image and return the saved output path."""


class ComfyUIGenerationBackend(ImageGenerationBackend):
    name = "comfyui"

    def __init__(self, config: AppConfig) -> None:
        self.client = ComfyUIClient(
            base_url=config.comfyui_base_url,
            workflow_path=config.comfyui_workflow_path,
            timeout_seconds=config.comfyui_timeout_seconds,
            poll_interval_seconds=config.comfyui_poll_interval_seconds,
            prompt_timeout_seconds=config.comfyui_prompt_timeout_seconds,
        )

    def generate(self, request: GenerationBackendRequest) -> Path:
        try:
            return self.client.generate_image(
                dog_image_path=request.dog_image_path,
                product_image_path=request.product_image_path,
                mask_image_path=request.mask_image_path,
                product_type=request.product_type,
                generation_prompt=request.generation_prompt,
                negative_prompt=request.negative_prompt,
                result_dir=request.result_dir,
            )
        except ComfyUIClientError as exc:
            raise GenerationBackendUnavailableError(BACKEND_UNAVAILABLE_MESSAGE) from exc


class LocalStubGenerationBackend(ImageGenerationBackend):
    name = "local_stub"

    def generate(self, request: GenerationBackendRequest) -> Path:
        raise GenerationBackendUnavailableError(BACKEND_UNAVAILABLE_MESSAGE)


def select_generation_backend(config: AppConfig, use_comfyui: bool) -> ImageGenerationBackend:
    """Return the requested backend without importing heavyweight model stacks."""

    if use_comfyui and config.image_backend == "comfyui":
        return ComfyUIGenerationBackend(config)
    return LocalStubGenerationBackend()
