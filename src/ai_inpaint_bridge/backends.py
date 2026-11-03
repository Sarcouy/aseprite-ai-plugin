"""Image backends that rewrite the masked area of a region."""

from typing import Protocol

from PIL import Image, ImageOps

from ai_inpaint_bridge.comfyui import InpaintSettings

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
    ) -> Image.Image:
        """Return an RGB image of the same size as the input."""


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
    source: Image.Image, mask: Image.Image, factor: int
) -> tuple[Image.Image, Image.Image]:
    """Flatten on grey, upscale without smoothing and pad to the latent grid."""
    rgba = source.convert("RGBA")
    flat = Image.new("RGB", rgba.size, NEUTRAL_GREY)
    flat.paste(rgba, (0, 0), rgba)
    size = (source.width * factor, source.height * factor)
    big_image = flat.resize(size, Image.Resampling.NEAREST)
    big_mask = binarize_mask(mask).resize(size, Image.Resampling.NEAREST)
    return _pad(big_image, NEUTRAL_GREY), _pad(big_mask, 0)


def downscale_from_diffusion(
    result: Image.Image, size: tuple[int, int], factor: int
) -> Image.Image:
    """Drop the padding and come back to the sprite resolution."""
    cropped = result.crop((0, 0, size[0] * factor, size[1] * factor))
    return cropped.resize(size, Image.Resampling.NEAREST)


def diffusion_inpaint(
    source: Image.Image,
    mask: Image.Image,
    settings: InpaintSettings,
    generator: InpaintGenerator,
) -> Image.Image:
    """Rewrite the masked pixels with the generator, keep the others untouched."""
    _check_sizes(source, mask)
    factor = scale_factor(source.size)
    big_image, big_mask = upscale_for_diffusion(source, mask, factor)
    result = generator.generate(big_image, big_mask, settings)
    small = downscale_from_diffusion(result, source.size, factor).convert("RGBA")
    return Image.composite(small, source.convert("RGBA"), binarize_mask(mask))
