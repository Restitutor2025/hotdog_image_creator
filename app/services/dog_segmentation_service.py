"""Local dog segmentation helpers for the experimental harness fitting mode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[2]
NUMBA_CACHE_DIR = BASE_DIR / ".cache" / "numba"


class DogSegmentationError(RuntimeError):
    """Raised when a useful dog mask cannot be estimated."""


@dataclass(frozen=True)
class DogSegmentationResult:
    dog_mask: Image.Image
    dog_bbox: list[int]


def _mask_bbox(mask_array: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask_array > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _clean_mask(mask_array: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned

    largest = max(contours, key=cv2.contourArea)
    largest_only = np.zeros_like(cleaned)
    cv2.drawContours(largest_only, [largest], -1, 255, thickness=cv2.FILLED)
    return largest_only


def _segment_with_rembg(dog_image: Image.Image) -> np.ndarray:
    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

    try:
        from rembg import remove
    except ImportError as exc:
        raise DogSegmentationError("rembg is required for dog segmentation. Run `pip install -r requirements.txt`.") from exc

    removed_background = remove(dog_image.convert("RGBA"))
    if isinstance(removed_background, Image.Image):
        segmented = removed_background.convert("RGBA")
    else:
        segmented = Image.open(BytesIO(removed_background)).convert("RGBA")

    alpha = np.array(segmented.getchannel("A"), dtype=np.uint8)
    _, binary = cv2.threshold(alpha, 24, 255, cv2.THRESH_BINARY)
    return binary


def _segment_with_grabcut(dog_image: Image.Image) -> np.ndarray:
    rgb = np.array(dog_image.convert("RGB"))
    height, width = rgb.shape[:2]

    margin_x = max(2, round(width * 0.08))
    margin_y = max(2, round(height * 0.08))
    rect = (margin_x, margin_y, max(1, width - margin_x * 2), max(1, height - margin_y * 2))

    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return foreground


def segment_dog(dog_path: Path) -> DogSegmentationResult:
    """Segment the dog from the background and return a binary mask plus bbox."""

    with Image.open(dog_path) as source_image:
        dog_image = source_image.convert("RGB")

    try:
        mask_array = _segment_with_rembg(dog_image)
    except Exception:
        mask_array = _segment_with_grabcut(dog_image)

    cleaned = _clean_mask(mask_array)
    bbox = _mask_bbox(cleaned)
    image_area = dog_image.width * dog_image.height
    mask_area = int(np.count_nonzero(cleaned))

    if bbox is None or mask_area < max(64, image_area * 0.015):
        raise DogSegmentationError("Could not estimate a reliable dog mask for fit_harness mode.")

    return DogSegmentationResult(dog_mask=Image.fromarray(cleaned, mode="L"), dog_bbox=bbox)
