"""Tests of the diffusion pre and post processing with a fake generator."""

from PIL import Image

from ai_inpaint_bridge.backends import (
    LATENT_MULTIPLE,
    diffusion_inpaint,
    scale_factor,
    upscale_for_diffusion,
)
from ai_inpaint_bridge.comfyui import InpaintSettings

BLUE = (10, 20, 200, 255)
GREEN = (0, 255, 0)
SETTINGS = InpaintSettings(prompt="a sword", seed=1)


class SolidGenerator:  # pylint: disable=too-few-public-methods  # test double of the single-method InpaintGenerator protocol
    """Fills the whole image with one color and records what it received."""

    def __init__(self) -> None:
        self.received: list[tuple[Image.Image, Image.Image]] = []

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> Image.Image:
        """Return a green image of the input size."""
        assert settings is SETTINGS
        self.received.append((image, mask))
        return Image.new("RGB", image.size, GREEN)


def make_inputs() -> tuple[Image.Image, Image.Image]:
    """Return a blue 10x6 region with a 2x2 selection at (4, 2)."""
    source = Image.new("RGBA", (10, 6), BLUE)
    mask = Image.new("L", (10, 6), 0)
    mask.paste(255, (4, 2, 6, 4))
    return source, mask


def test_scale_factor_targets_the_diffusion_size() -> None:
    """Small regions are enlarged, large ones are left alone."""
    assert scale_factor((32, 16)) == 16
    assert scale_factor((10, 6)) == 51
    assert scale_factor((2000, 100)) == 1


def test_upscale_pads_to_the_latent_grid() -> None:
    """Both images are enlarged without smoothing and padded to a multiple of 8."""
    source, mask = make_inputs()
    image, big_mask = upscale_for_diffusion(source, mask, 5)

    assert image.size == big_mask.size == (56, 32)
    assert image.width % LATENT_MULTIPLE == 0 and image.height % LATENT_MULTIPLE == 0
    assert image.getpixel((0, 0)) == BLUE[:3]
    assert big_mask.getpixel((4 * 5, 2 * 5)) == 255
    assert big_mask.getpixel((4 * 5 - 1, 2 * 5)) == 0
    assert big_mask.getpixel((55, 31)) == 0


def test_diffusion_inpaint_only_rewrites_the_selection() -> None:
    """Generated pixels land inside the mask, the rest of the region is kept."""
    source, mask = make_inputs()
    generator = SolidGenerator()

    result = diffusion_inpaint(source, mask, SETTINGS, generator)

    assert len(generator.received) == 1
    assert result.size == source.size
    assert result.getpixel((4, 2)) == (*GREEN, 255)
    assert result.getpixel((5, 3)) == (*GREEN, 255)
    assert result.getpixel((3, 2)) == BLUE
    assert result.getpixel((6, 3)) == BLUE
