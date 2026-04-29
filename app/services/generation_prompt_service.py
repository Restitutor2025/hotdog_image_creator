"""Prompt loading for local dog outfit image generation."""

from __future__ import annotations

from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
GENERATION_PROMPT_PATH = PROMPT_DIR / "dog_outfit_generation_prompt.txt"
NEGATIVE_PROMPT_PATH = PROMPT_DIR / "dog_outfit_negative_prompt.txt"


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_generation_prompt(product_type: str, prompt_mode: str = "wearing_preview") -> str:
    """Build the fixed generation prompt with request metadata."""

    base_prompt = _read_prompt(GENERATION_PROMPT_PATH)
    product_label = "dog harness" if product_type == "harness" else "dog clothes"
    return f"{base_prompt}\nProduct type: {product_label}.\nPrompt mode: {prompt_mode}."


def load_negative_prompt() -> str:
    """Load the fixed negative prompt used by generation backends."""

    return _read_prompt(NEGATIVE_PROMPT_PATH)
