"""Heuristic dog body landmarks estimated from a segmentation mask."""

from __future__ import annotations

import numpy as np
from PIL import Image


Keypoints = dict[str, list[int]]


def _point_at_mask_row(mask: np.ndarray, y: int, fallback_x: int) -> tuple[int, int, int]:
    height = mask.shape[0]
    y = int(np.clip(y, 0, height - 1))
    nearby_rows = mask[max(0, y - 3) : min(height, y + 4), :]
    xs = np.where(nearby_rows > 0)[1]
    if len(xs) == 0:
        return fallback_x, fallback_x, y
    return int(xs.min()), int(xs.max()), y


def estimate_dog_keypoints(dog_mask: Image.Image, dog_bbox: list[int]) -> Keypoints:
    """Estimate coarse JSON-safe body points from the dog mask geometry."""

    mask = np.array(dog_mask.convert("L"), dtype=np.uint8)
    x1, y1, x2, y2 = dog_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        body_center = [int(x1 + width * 0.5), int(y1 + height * 0.5)]
    else:
        body_center = [int(round(xs.mean())), int(round(ys.mean()))]

    center_x = body_center[0]
    neck_y = int(round(y1 + height * 0.20))
    shoulder_y = int(round(y1 + height * 0.34))
    chest_y = int(round(y1 + height * 0.46))
    torso_y = int(round(y1 + height * 0.60))
    back_y = int(round(y1 + height * 0.26))

    shoulder_left_x, shoulder_right_x, shoulder_y = _point_at_mask_row(mask, shoulder_y, center_x)
    torso_left_x, torso_right_x, torso_y = _point_at_mask_row(mask, torso_y, center_x)
    neck_left_x, neck_right_x, neck_y = _point_at_mask_row(mask, neck_y, center_x)
    back_left_x, back_right_x, back_y = _point_at_mask_row(mask, back_y, center_x)
    chest_left_x, chest_right_x, chest_y = _point_at_mask_row(mask, chest_y, center_x)

    shoulder_inset = max(2, round((shoulder_right_x - shoulder_left_x) * 0.18))
    torso_inset = max(2, round((torso_right_x - torso_left_x) * 0.18))

    return {
        "body_center": body_center,
        "neck_center": [int(round((neck_left_x + neck_right_x) / 2)), neck_y],
        "chest_center": [int(round((chest_left_x + chest_right_x) / 2)), chest_y],
        "shoulder_left": [shoulder_left_x + shoulder_inset, shoulder_y],
        "shoulder_right": [shoulder_right_x - shoulder_inset, shoulder_y],
        "back_center": [int(round((back_left_x + back_right_x) / 2)), back_y],
        "torso_left": [torso_left_x + torso_inset, torso_y],
        "torso_right": [torso_right_x - torso_inset, torso_y],
    }
