"""Tests of the diffusion pre and post processing with a fake generator."""

from dataclasses import replace

import pytest
from PIL import Image

from ai_inpaint_bridge.backends import (
    LATENT_MULTIPLE,
    changed_fraction,
    diffusion_generate,
    diffusion_inpaint,
    grow_mask,
    scale_factor,
    upscale_for_diffusion,
)
from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.pixelize import pick_key_color

BLUE = (10, 20, 200, 255)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
PINK = (235, 60, 200)
SETTINGS = InpaintSettings(prompt="a sword", seed=1)


class SolidGenerator:  # pylint: disable=too-few-public-methods  # test double of the single-method InpaintGenerator protocol
    """Fills the whole image with one color and records what it received."""

    def __init__(self, color: tuple[int, int, int] = GREEN) -> None:
        self.color = color
        self.received: list[tuple[Image.Image, Image.Image, InpaintSettings]] = []

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Return batch_size images of the input size filled with the configured color."""
        self.received.append((image, mask, settings))
        return [Image.new("RGB", image.size, self.color)] * settings.batch_size


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

    [result] = diffusion_inpaint(source, mask, SETTINGS, generator)

    assert len(generator.received) == 1
    assert generator.received[0][2] == SETTINGS
    assert result.size == source.size
    assert result.getpixel((4, 2)) == (*GREEN, 255)
    assert result.getpixel((5, 3)) == (*GREEN, 255)
    assert result.getpixel((3, 2)) == BLUE
    assert result.getpixel((6, 3)) == BLUE


def test_transparent_region_is_sent_on_a_key_color_and_keyed_back() -> None:
    """A key-colored result over a transparent area stays transparent."""
    source = Image.new("RGBA", (10, 6), (0, 0, 0, 0))
    source.paste(BLUE, (0, 0, 2, 6))
    mask = Image.new("L", (10, 6), 0)
    mask.paste(255, (4, 2, 6, 4))
    key, key_name = pick_key_color([BLUE[:3]])
    generator = SolidGenerator(key)

    [result] = diffusion_inpaint(source, mask, SETTINGS, generator)

    sent_image, _, sent_settings = generator.received[0]
    assert sent_image.getpixel((sent_image.width - 1, 0)) == key
    assert sent_settings.background == key_name
    pixel = result.getpixel((4, 2))
    assert isinstance(pixel, tuple) and pixel[3] == 0


def test_opaque_region_never_turns_transparent() -> None:
    """Without transparency in the region, key-like colors are kept as colors."""
    source, mask = make_inputs()

    [result] = diffusion_inpaint(source, mask, SETTINGS, SolidGenerator(MAGENTA))

    pixel = result.getpixel((4, 2))
    assert isinstance(pixel, tuple) and pixel[3] == 255


def test_one_result_per_generated_image() -> None:
    """Every image of the batch becomes its own result, in order."""
    source, mask = make_inputs()

    results = diffusion_inpaint(
        source, mask, replace(SETTINGS, batch_size=3), SolidGenerator()
    )

    assert len(results) == 3
    assert all(result.getpixel((4, 2)) == (*GREEN, 255) for result in results)


class SingleImageGenerator:  # pylint: disable=too-few-public-methods  # test double of the single-method InpaintGenerator protocol
    """Ignores the requested batch size and always returns one image."""

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Return one green image of the input size."""
        del mask, settings
        return [Image.new("RGB", image.size, GREEN)]


def test_missing_images_are_reported() -> None:
    """A generator returning fewer images than requested is an error."""
    source, mask = make_inputs()

    with pytest.raises(ValueError, match="expected 2"):
        diffusion_inpaint(
            source, mask, replace(SETTINGS, batch_size=2), SingleImageGenerator()
        )


def test_grow_mask_dilates_by_whole_pixels() -> None:
    """A 2x2 selection grown by 1 becomes 4x4, 0 keeps it unchanged."""
    _, mask = make_inputs()

    assert grow_mask(mask, 0).getbbox() == (4, 2, 6, 4)
    assert grow_mask(mask, 1).getbbox() == (3, 1, 7, 5)


def test_grown_mask_is_sent_but_only_the_selection_is_kept() -> None:
    """The generator sees a larger hole, the result changes only selected pixels."""
    source, mask = make_inputs()
    generator = SolidGenerator()

    [result] = diffusion_inpaint(
        source, mask, replace(SETTINGS, mask_grow=1), generator
    )

    sent_mask = generator.received[0][1]
    factor = sent_mask.width // 10
    assert sent_mask.getpixel((3 * factor, 1 * factor)) == 255
    assert result.getpixel((3, 1)) == BLUE
    assert result.getpixel((4, 2)) == (*GREEN, 255)


def test_changed_fraction_counts_only_selected_pixels() -> None:
    """Untouched selections score 0, a fully redrawn one scores 1."""
    source, mask = make_inputs()
    [redrawn] = diffusion_inpaint(source, mask, SETTINGS, SolidGenerator())

    assert changed_fraction(source, source, mask) == 0.0
    assert changed_fraction(source, redrawn, mask) == 1.0


def test_transparent_pixels_count_as_unchanged_whatever_their_rgb() -> None:
    """Two fully transparent pixels are equal even with different hidden colors."""
    original = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    result = Image.new("RGBA", (2, 1), (255, 0, 255, 0))
    mask = Image.new("L", (2, 1), 255)

    assert changed_fraction(original, result, mask) == 0.0


def test_approximate_key_over_a_transparent_area_is_keyed_back() -> None:
    """Klein's pink for a requested magenta still turns back into transparency."""
    source = Image.new("RGBA", (10, 6), (0, 0, 0, 0))
    mask = Image.new("L", (10, 6), 255)

    [result] = diffusion_inpaint(source, mask, SETTINGS, SolidGenerator(PINK))

    assert result.getchannel("A").getbbox() is None


class OrangeOnGolden:  # pylint: disable=too-few-public-methods  # test double of the single-method TextGenerator protocol
    """An orange square on the golden yellow klein draws for a yellow key."""

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """batch_size images with the subject in their middle half."""
        image = Image.new("RGB", (width, height), (254, 219, 61))
        image.paste(
            (255, 143, 58), (width // 4, height // 4, width * 3 // 4, height * 3 // 4)
        )
        return [image] * settings.batch_size


def test_generated_subject_close_to_the_drawn_key_is_kept() -> None:
    """The measured background is keyed out, not a subject of a neighbouring hue."""
    palette = [(200, 10, 30), (55, 245, 225), (20, 40, 220)]

    [sprite] = diffusion_generate((8, 8), SETTINGS, OrangeOnGolden(), palette)

    assert sprite.getchannel("A").getbbox() == (2, 2, 6, 6)
    assert sprite.getpixel((4, 4)) == (200, 10, 30, 255)
