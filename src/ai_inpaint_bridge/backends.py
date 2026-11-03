"""Image backends that rewrite the masked area of a region."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from PIL import Image, ImageOps

from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.pixelize import (
    Color,
    opaque_colors,
    pick_key_color,
    to_sprite_pixels,
)

MASK_THRESHOLD = 127
DIFFUSION_SIZE = 512
LATENT_MULTIPLE = 8
NEUTRAL_GREY = (128, 128, 128)


class InpaintGenerator(
    Protocol
):  # pylint: disable=too-few-public-methods  # a protocol describing one callable capability
    """Anything able to inpaint an upscaled RGB image with a mask."""

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Return settings.batch_size RGB images of the same size as the input."""


def binarize_mask(mask: Image.Image) -> Image.Image:
    """Return a mode "1"-like L mask: 255 inside the selection, 0 outside."""
    return mask.convert("L").point(lambda value: 255 if value > MASK_THRESHOLD else 0)


def _check_sizes(source: Image.Image, mask: Image.Image) -> None:
    if source.size != mask.size:
        raise ValueError(
            f"source size {source.size} differs from mask size {mask.size}"
        )


def invert_masked(source: Image.Image, mask: Image.Image) -> Image.Image:
    """Invert the colors of the masked pixels, keep everything else untouched."""
    _check_sizes(source, mask)
    rgba = source.convert("RGBA")
    inverted = ImageOps.invert(rgba.convert("RGB")).convert("RGBA")
    inverted.putalpha(rgba.getchannel("A"))
    return Image.composite(inverted, rgba, binarize_mask(mask))


def scale_factor(size: tuple[int, int], target: int = DIFFUSION_SIZE) -> int:
    """Integer factor bringing the longest side close to the diffusion size."""
    return max(1, round(target / max(size)))


def _pad(image: Image.Image, fill: int | tuple[int, int, int]) -> Image.Image:
    width = -(-image.width // LATENT_MULTIPLE) * LATENT_MULTIPLE
    height = -(-image.height // LATENT_MULTIPLE) * LATENT_MULTIPLE
    padded = Image.new(image.mode, (width, height), fill)
    padded.paste(image, (0, 0))
    return padded


def upscale_for_diffusion(
    source: Image.Image,
    mask: Image.Image,
    factor: int,
    background: Color = NEUTRAL_GREY,
) -> tuple[Image.Image, Image.Image]:
    """Flatten on the background, upscale without smoothing, pad to the latent grid."""
    rgba = source.convert("RGBA")
    flat = Image.new("RGB", rgba.size, background)
    flat.paste(rgba, (0, 0), rgba)
    size = (source.width * factor, source.height * factor)
    big_image = flat.resize(size, Image.Resampling.NEAREST)
    big_mask = binarize_mask(mask).resize(size, Image.Resampling.NEAREST)
    return _pad(big_image, background), _pad(big_mask, 0)


def has_transparency(image: Image.Image) -> bool:
    """True when at least one pixel is not fully opaque."""
    lowest_alpha = image.convert("RGBA").getchannel("A").getextrema()[0]
    return isinstance(lowest_alpha, int) and lowest_alpha < 255


def diffusion_inpaint(
    source: Image.Image,
    mask: Image.Image,
    settings: InpaintSettings,
    generator: InpaintGenerator,
    palette: Sequence[Color] | None = None,
) -> list[Image.Image]:
    """Rewrite the masked pixels with the generator, keep the others untouched.

    One result is returned per generated image (settings.batch_size).

    Transparent areas are sent on a key color that is turned back into
    transparency, then every pixel is snapped to the palette (or to an
    adaptive one when no palette is given).
    """
    _check_sizes(source, mask)
    rgba = source.convert("RGBA")
    factor = scale_factor(source.size)
    key, key_name = pick_key_color(opaque_colors(rgba))
    transparent_area = has_transparency(rgba)
    if transparent_area:
        settings = replace(settings, background=key_name)
    big_image, big_mask = upscale_for_diffusion(rgba, mask, factor, key)
    results = generator.generate(big_image, big_mask, settings)
    if len(results) != settings.batch_size:
        raise ValueError(
            f"expected {settings.batch_size} generated images, got {len(results)}"
        )
    selection = binarize_mask(mask)
    key_out = key if transparent_area else None
    return [
        Image.composite(
            to_sprite_pixels(result, source.size, factor, palette, key_out),
            rgba,
            selection,
        )
        for result in results
    ]
