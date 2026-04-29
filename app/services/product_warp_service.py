"""Product warping for experimental dog harness fitting."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.services.dog_keypoint_service import Keypoints


def _apply_opacity(layer: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1:
        return layer
    adjusted = layer.copy()
    adjusted.putalpha(adjusted.getchannel("A").point(lambda value: round(value * opacity)))
    return adjusted


def _destination_quad(keypoints: Keypoints, warp_strength: float) -> np.ndarray:
    shoulder_left = np.array(keypoints["shoulder_left"], dtype=np.float32)
    shoulder_right = np.array(keypoints["shoulder_right"], dtype=np.float32)
    torso_left = np.array(keypoints["torso_left"], dtype=np.float32)
    torso_right = np.array(keypoints["torso_right"], dtype=np.float32)
    chest = np.array(keypoints["chest_center"], dtype=np.float32)
    neck = np.array(keypoints["neck_center"], dtype=np.float32)

    top_left = shoulder_left * 0.82 + neck * 0.18
    top_right = shoulder_right * 0.82 + neck * 0.18
    bottom_left = torso_left * 0.72 + chest * 0.28
    bottom_right = torso_right * 0.72 + chest * 0.28
    fitted = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    center = fitted.mean(axis=0)
    rectangle_width = max(1.0, np.linalg.norm(shoulder_right - shoulder_left) * 1.12)
    rectangle_height = max(1.0, np.linalg.norm((torso_left + torso_right) / 2 - neck) * 1.05)
    rectangle = np.array(
        [
            [center[0] - rectangle_width / 2, center[1] - rectangle_height / 2],
            [center[0] + rectangle_width / 2, center[1] - rectangle_height / 2],
            [center[0] + rectangle_width / 2, center[1] + rectangle_height / 2],
            [center[0] - rectangle_width / 2, center[1] + rectangle_height / 2],
        ],
        dtype=np.float32,
    )

    strength = float(np.clip(warp_strength, 0.0, 1.0))
    return rectangle * (1.0 - strength) + fitted * strength


def warp_product_to_dog(
    product_image: Image.Image,
    canvas_size: tuple[int, int],
    keypoints: Keypoints,
    placement: dict[str, object],
) -> Image.Image:
    """Perspective-warp a transparent product into the estimated torso region."""

    canvas_width, canvas_height = canvas_size
    product_rgba = product_image.convert("RGBA")
    product_array = np.array(product_rgba)

    source_quad = np.array(
        [
            [0, 0],
            [product_rgba.width - 1, 0],
            [product_rgba.width - 1, product_rgba.height - 1],
            [0, product_rgba.height - 1],
        ],
        dtype=np.float32,
    )
    destination_quad = _destination_quad(keypoints, float(placement.get("warp_strength", 0.75)))

    if not bool(placement.get("auto_fit", True)):
        scale = float(placement.get("scale", 1.0))
        width = max(1, round(product_rgba.width * scale))
        height = max(1, round(product_rgba.height * scale))
        x = float(placement.get("x", (canvas_width - width) / 2))
        y = float(placement.get("y", (canvas_height - height) / 2))
        manual_quad = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)
        destination_quad = manual_quad * 0.55 + destination_quad * 0.45

    transform = cv2.getPerspectiveTransform(source_quad, destination_quad)
    warped = cv2.warpPerspective(
        product_array,
        transform,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    layer = Image.fromarray(warped, mode="RGBA")
    return _apply_opacity(layer, float(placement.get("opacity", 1.0)))
