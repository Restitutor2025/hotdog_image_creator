"""Prompt loader service for the fixed pet product preview prompt.

Process overview:
1. Resolve the prompt file inside app/prompts.
2. Read the exact prompt text from disk.
3. Return that text so API responses can include it for debugging or logging.
"""

from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "pet_product_preview.txt"


def load_pet_product_preview_prompt() -> str:
    """Load the fixed prompt used to describe the allowed preview composition."""

    # Drop only the final file newline so JSON responses match the prompt text block exactly.
    return PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")


# Example: include load_pet_product_preview_prompt() in the /preview/composite response.
