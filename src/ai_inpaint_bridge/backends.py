"""Image backends that rewrite the masked area of a region."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from PIL import Image, ImageFilter, ImageOps

from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.pixelize import (
    MEASURED_KEY_TOLERANCE,
    Color,
    band_values,
    border_color,
    pixel_tuples,
    opaque_colors,
    pick_key_color,
    to_sprite_pixels,
)

MASK_THRESHOLD = 127
DIFFUSION_SIZE = 512
LATENT_MULTIPLE = 8
TEXT_MULTIPLE = 16
NEUTRAL_GREY = (128, 128, 128)


class InpaintGenerator(
    Protocol
):  # pylint: disable=too-few-public-methods  # a protocol describing one callable capability
    """Anything able to inpaint an upscaled RGB image with a mask."""

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Return settings.batch_size RGB images of the same size as the input."""


class TextGenerator(
    Protocol
):  # pylint: disable=too-few-public-methods  # a protocol describing one callable capability
    """Anything able to draw RGB images from the prompt alone."""

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """Return settings.batch_size RGB images of the given size."""


class ReferenceGenerator(
    Protocol
):  # pylint: disable=too-few-public-methods  # a protocol describing one callable capability
    """Anything able to draw RGB images from the prompt and a reference image."""

    def generate_reference(
        self, reference: Image.Image, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """Return settings.batch_size RGB images of the given size."""


class RedrawGenerator(
    Protocol
):  # pylint: disable=too-few-public-methods  # a protocol describing one callable capability
    """Anything able to redraw an image, keeping its layout."""

    def generate_redraw(
        self, image: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Return settings.batch_size RGB images of the size of the image."""


class GuidedGenerator(
    Protocol
):  # pylint: disable=too-few-public-methods  # a protocol describing one callable capability
    """Anything able to redraw a guide image with the look of a reference image."""

    def generate_guided(
        self,
        reference: Image.Image,
        guide: Image.Image,
        settings: InpaintSettings,
    ) -> list[Image.Image]:
        """Return settings.batch_size RGB images of the size of the guide."""


def binarize_mask(mask: Image.Image) -> Image.Image:
    """Return a mode "1"-like L mask: 255 inside the selection, 0 outside."""
    return mask.convert("L").point(lambda value: 255 if value > MASK_THRESHOLD else 0)


def grow_mask(mask: Image.Image, pixels: int) -> Image.Image:
    """Binarized mask dilated by the given number of pixels (square neighbourhood)."""
    binary = binarize_mask(mask)
    if pixels <= 0:
        return binary
    return binary.filter(ImageFilter.MaxFilter(2 * pixels + 1))


def changed_fraction(
    original: Image.Image, result: Image.Image, mask: Image.Image
) -> float:
    """Share of the selected pixels whose RGBA value differs from the original."""
    selected = [value > MASK_THRESHOLD for value in band_values(binarize_mask(mask))]
    before = pixel_tuples(original.convert("RGBA"))
    after = pixel_tuples(result.convert("RGBA"))
    changed = [
        old != new and (old[3] > 0 or new[3] > 0)
        for old, new, keep in zip(before, after, selected)
        if keep
    ]
    return sum(changed) / len(changed) if changed else 0.0


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

    One result is returned per generated image (settings.batch_size). The
    mask sent to the generator is grown by settings.mask_grow pixels so that
    edges drawn along it fall outside the selection, which alone is kept.

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
    big_image, big_mask = upscale_for_diffusion(
        rgba, grow_mask(mask, settings.mask_grow), factor, key
    )
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


def diffusion_generate(
    size: tuple[int, int],
    settings: InpaintSettings,
    generator: TextGenerator,
    palette: Sequence[Color] | None = None,
    transparent: bool = True,
) -> list[Image.Image]:
    """Draw settings.batch_size sprites of the given size from the prompt alone.

    The image is generated at an integer multiple of the size, rounded up to
    the latent grid, then reduced by majority vote. A transparent sprite is
    asked on a key color named in the prompt; the model only approximates it
    (klein draws "magenta" as a pink), so the background that becomes
    transparent is the color measured on the border of each image.
    """
    factor = scale_factor(size)
    width = -(-size[0] * factor // TEXT_MULTIPLE) * TEXT_MULTIPLE
    height = -(-size[1] * factor // TEXT_MULTIPLE) * TEXT_MULTIPLE
    if transparent:
        key_name = pick_key_color(list(palette or []))[1]
        settings = replace(settings, background=key_name, background_hint=True)
    results = generator.generate_text(settings, width, height)
    if len(results) != settings.batch_size:
        raise ValueError(
            f"expected {settings.batch_size} generated images, got {len(results)}"
        )
    return [
        to_sprite_pixels(
            result,
            size,
            factor,
            palette,
            border_color(result) if transparent else None,
            tolerance=MEASURED_KEY_TOLERANCE,
        )
        for result in results
    ]
