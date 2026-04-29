"""Lightweight local refinement hooks for fitted harness layers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter


def _contact_shadow_from_alpha(alpha: Image.Image) -> Image.Image:
    shadow_alpha = alpha.point(lambda value: round(value * 0.22)).filter(ImageFilter.GaussianBlur(radius=10))
    shadow = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    return shadow


def refine_harness_layer(base_image: Image.Image, harness_layer: Image.Image) -> Image.Image:
    """Feather edges, add contact shadow, and gently match local brightness."""

    base_rgb = np.array(base_image.convert("RGB"), dtype=np.float32)
    harness_rgba = np.array(harness_layer.convert("RGBA"), dtype=np.float32)
    alpha = harness_rgba[:, :, 3].astype(np.uint8)

    if not np.any(alpha):
        return harness_layer.convert("RGBA")

    alpha_soft = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.7, sigmaY=0.7)
    edge = cv2.subtract(cv2.dilate(alpha, np.ones((7, 7), np.uint8)), cv2.erode(alpha, np.ones((7, 7), np.uint8)))
    product_pixels = harness_rgba[:, :, :3][alpha > 24]
    base_pixels = base_rgb[edge > 0]

    if len(product_pixels) and len(base_pixels):
        product_luma = float(np.mean(product_pixels))
        base_luma = float(np.mean(base_pixels))
        gain = float(np.clip((base_luma / max(product_luma, 1.0)) * 0.18 + 0.82, 0.88, 1.10))
        harness_rgba[:, :, :3] = np.clip(harness_rgba[:, :, :3] * gain, 0, 255)

    harness_rgba[:, :, 3] = alpha_soft
    refined_harness = Image.fromarray(harness_rgba.astype(np.uint8), mode="RGBA")
    shadow = _contact_shadow_from_alpha(refined_harness.getchannel("A"))

    # TODO: add an optional local diffusion/inpainting backend here later, without
    # downloading large models automatically or changing the default CPU path.
    return Image.alpha_composite(shadow, refined_harness)
