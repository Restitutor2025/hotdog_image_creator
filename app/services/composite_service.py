"""Image compositing service for the local pet product preview MVP.

Process overview:
1. Load the dog photo as the unchanged base image.
2. Keep the product angle, aspect ratio, shape, color, logo, and texture intact.
3. Resize the product only when placement.scale is explicitly provided.
4. Apply optional opacity by adjusting alpha only.
5. Add a simple blurred contact shadow below the product.
6. Alpha-composite the product onto the dog photo and save the result.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageFilter


def _resize_only_when_requested(product_image: Image.Image, placement: dict[str, float]) -> Image.Image:
    """Scale the product only when the request explicitly includes placement.scale."""

    if "scale" not in placement:
        return product_image

    scale = placement["scale"]
    width = max(1, round(product_image.width * scale))
    height = max(1, round(product_image.height * scale))
    return product_image.resize((width, height), Image.Resampling.LANCZOS)


def _apply_opacity(product_image: Image.Image, opacity: float) -> Image.Image:
    """Apply alpha blending without changing product RGB pixels."""

    if opacity >= 1:
        return product_image

    product_with_opacity = product_image.copy()
    alpha = product_with_opacity.getchannel("A")
    alpha = alpha.point(lambda value: round(value * opacity))
    product_with_opacity.putalpha(alpha)
    return product_with_opacity


def _make_contact_shadow(product_image: Image.Image) -> Image.Image:
    """Create a soft shadow from the product alpha mask without altering the product."""

    alpha = product_image.getchannel("A")
    shadow_strength = 0.32
    shadow_alpha = alpha.point(lambda value: round(value * shadow_strength))
    blur_radius = max(6, round(min(product_image.width, product_image.height) * 0.045))
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    shadow = Image.new("RGBA", product_image.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    return shadow


def _alpha_composite_clipped(base_image: Image.Image, overlay: Image.Image, x: int, y: int) -> None:
    """Composite an overlay onto the base image while allowing off-canvas placement."""

    left = max(0, x)
    top = max(0, y)
    right = min(base_image.width, x + overlay.width)
    bottom = min(base_image.height, y + overlay.height)

    if right <= left or bottom <= top:
        return

    crop_left = left - x
    crop_top = top - y
    crop_right = crop_left + (right - left)
    crop_bottom = crop_top + (bottom - top)
    visible_overlay = overlay.crop((crop_left, crop_top, crop_right, crop_bottom))
    base_image.alpha_composite(visible_overlay, (left, top))


def _resolve_position(base_image: Image.Image, product_image: Image.Image, placement: dict[str, float]) -> tuple[int, int]:
    """Use placement x/y when provided, otherwise center the product on the dog photo."""

    default_x = (base_image.width - product_image.width) / 2
    default_y = (base_image.height - product_image.height) / 2
    x = placement.get("x", default_x)
    y = placement.get("y", default_y)
    return round(x), round(y)


def composite_product_on_dog(
    dog_path: Path,
    product_image: Image.Image,
    placement: dict[str, float],
    result_dir: Path,
) -> tuple[Path, int, int]:
    """Composite a transparent product image onto a dog photo and save the result."""

    with Image.open(dog_path) as source_dog:
        base_image = source_dog.convert("RGBA")

    product_rgba = product_image.convert("RGBA")
    product_rgba = _resize_only_when_requested(product_rgba, placement)
    product_rgba = _apply_opacity(product_rgba, placement.get("opacity", 1.0))

    x, y = _resolve_position(base_image, product_rgba, placement)
    shadow = _make_contact_shadow(product_rgba)
    shadow_offset = max(4, round(min(product_rgba.width, product_rgba.height) * 0.035))

    # The shadow is placed first, so the original product pixels remain visible on top.
    _alpha_composite_clipped(base_image, shadow, x + shadow_offset, y + shadow_offset)
    _alpha_composite_clipped(base_image, product_rgba, x, y)

    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"pet_product_preview_{uuid4().hex}.png"
    base_image.save(result_path, format="PNG")
    return result_path, base_image.width, base_image.height


# Example: omit placement.scale to preserve the product image's original pixel size.

