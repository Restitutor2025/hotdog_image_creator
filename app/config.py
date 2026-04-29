"""Runtime configuration for the local Stable Diffusion preview backend."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
LOCAL_DIR = BASE_DIR / ".local"
MODEL_CACHE_DIR = LOCAL_DIR / "huggingface"


class AppConfig(BaseModel):
    """Environment-backed Stable Diffusion settings."""

    sd_model_id: str = Field(default_factory=lambda: os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-inpainting"))
    sd_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "SD_PROMPT",
            "a dog is wearing the same olive green and black harness from the product reference image",
        )
    )
    sd_negative_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "SD_NEGATIVE_PROMPT",
            "different dog, different breed, changed face, distorted dog face, changed fur color, changed dog identity, "
            "pasted sticker, flat overlay, floating harness, different harness, black-only harness, collar only, leash, "
            "product beside dog, extra dog, duplicate dog, bad anatomy, missing legs, extra legs, blurry, low quality",
        )
    )
    sd_image_size: int = Field(default_factory=lambda: int(os.getenv("SD_IMAGE_SIZE", "768")))
    sd_strength: float = Field(default_factory=lambda: float(os.getenv("SD_STRENGTH", "0.72")))
    sd_guidance_scale: float = Field(default_factory=lambda: float(os.getenv("SD_GUIDANCE_SCALE", "7.0")))
    sd_steps: int = Field(default_factory=lambda: int(os.getenv("SD_STEPS", "35")))
    sd_seed: int | None = Field(default_factory=lambda: int(os.getenv("SD_SEED")) if os.getenv("SD_SEED") else None)
    sd_cache_dir: str = Field(default_factory=lambda: os.getenv("SD_CACHE_DIR", str(MODEL_CACHE_DIR)))
    sd_use_ip_adapter: bool = Field(
        default_factory=lambda: os.getenv("SD_USE_IP_ADAPTER", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    sd_ip_adapter_repo: str = Field(default_factory=lambda: os.getenv("SD_IP_ADAPTER_REPO", "h94/IP-Adapter"))
    sd_ip_adapter_subfolder: str = Field(default_factory=lambda: os.getenv("SD_IP_ADAPTER_SUBFOLDER", "models"))
    sd_ip_adapter_weight: str = Field(default_factory=lambda: os.getenv("SD_IP_ADAPTER_WEIGHT", "ip-adapter_sd15.bin"))
    sd_ip_adapter_scale: float = Field(default_factory=lambda: float(os.getenv("SD_IP_ADAPTER_SCALE", "0.75")))


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()
