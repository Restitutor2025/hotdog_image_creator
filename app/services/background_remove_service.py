"""Background removal service for product uploads.

Process overview:
1. Load the product image from disk with PIL.
2. Convert it to RGBA so transparency is available.
3. Use rembg locally to remove the background.
4. Return a transparent RGBA PIL image for the compositing service.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[2]
NUMBA_CACHE_DIR = BASE_DIR / ".cache" / "numba"


def remove_product_background(product_path: Path) -> Image.Image:
    """Remove the background from a product image using local rembg inference."""

    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError("rembg is required. Run `pip install -r requirements.txt` before using this endpoint.") from exc

    with Image.open(product_path) as source_image:
        product_rgba = source_image.convert("RGBA")

    # rembg may return a PIL image or bytes depending on package version/input type.
    removed_background = remove(product_rgba)
    if isinstance(removed_background, Image.Image):
        return removed_background.convert("RGBA")

    return Image.open(BytesIO(removed_background)).convert("RGBA")


# Example: pass the returned RGBA image directly into composite_product_on_dog().
