"""Soft dog-over-harness occlusion masks for the fitting MVP."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.services.dog_keypoint_service import Keypoints


def _ellipse(draw: ImageDraw.ImageDraw, center: list[int], radius_x: int, radius_y: int, fill: int) -> None:
    x, y = center
    draw.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y), fill=fill)


def build_occlusion_mask(
    dog_mask: Image.Image,
    keypoints: Keypoints,
    harness_layer: Image.Image,
    occlusion_strength: float,
) -> Image.Image:
    """Create a soft mask where original dog pixels should sit over the harness."""

    strength = float(np.clip(occlusion_strength, 0.0, 1.0))
    if strength <= 0:
        return Image.new("L", dog_mask.size, 0)

    dog = np.array(dog_mask.convert("L"), dtype=np.uint8)
    alpha = np.array(harness_layer.getchannel("A"), dtype=np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    edge = cv2.subtract(cv2.dilate(dog, kernel, iterations=1), cv2.erode(dog, kernel, iterations=1))
    edge = cv2.bitwise_and(edge, alpha)

    mask = Image.fromarray(edge, mode="L")
    draw = ImageDraw.Draw(mask)

    shoulder_width = max(8, abs(keypoints["shoulder_right"][0] - keypoints["shoulder_left"][0]))
    torso_width = max(8, abs(keypoints["torso_right"][0] - keypoints["torso_left"][0]))
    radius_x = max(8, round(shoulder_width * 0.18))
    radius_y = max(8, round(shoulder_width * 0.11))

    _ellipse(draw, keypoints["neck_center"], max(10, round(shoulder_width * 0.22)), radius_y, 190)
    _ellipse(draw, keypoints["shoulder_left"], radius_x, radius_y, 170)
    _ellipse(draw, keypoints["shoulder_right"], radius_x, radius_y, 170)
    _ellipse(draw, keypoints["chest_center"], max(10, round(torso_width * 0.16)), max(10, round(torso_width * 0.13)), 145)

    mask_array = np.array(mask, dtype=np.uint8)
    mask_array = cv2.bitwise_and(mask_array, dog)
    mask_array = cv2.bitwise_and(mask_array, alpha)
    mask_array = np.clip(mask_array.astype(np.float32) * strength, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_array, mode="L").filter(ImageFilter.GaussianBlur(radius=8))


def composite_with_occlusion(base_image: Image.Image, harness_layer: Image.Image, occlusion_mask: Image.Image) -> Image.Image:
    """Composite dog, warped harness, then selected dog pixels restored on top."""

    base_rgba = base_image.convert("RGBA")
    fitted = Image.alpha_composite(base_rgba, harness_layer.convert("RGBA"))
    dog_top = base_rgba.copy()
    dog_top.putalpha(occlusion_mask.convert("L"))
    return Image.alpha_composite(fitted, dog_top)
