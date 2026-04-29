"""Runtime configuration for the local pet preview backend."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
DEBUG_DIR = STATIC_DIR / "debug"


class AppConfig(BaseModel):
    """Environment-backed settings without contacting model backends at import time."""

    comfyui_base_url: str = Field(
        default_factory=lambda: os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
    )
    image_backend: str = Field(default_factory=lambda: os.getenv("IMAGE_BACKEND", "comfyui").lower())
    comfyui_workflow_path: str | None = Field(default_factory=lambda: os.getenv("COMFYUI_WORKFLOW_PATH"))
    comfyui_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("COMFYUI_TIMEOUT_SECONDS", "10")))
    comfyui_poll_interval_seconds: float = Field(
        default_factory=lambda: float(os.getenv("COMFYUI_POLL_INTERVAL_SECONDS", "1.5"))
    )
    comfyui_prompt_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("COMFYUI_PROMPT_TIMEOUT_SECONDS", "600"))
    )


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()
