"""Tests of the conversion from diffusion output to sprite pixels."""

from PIL import Image

from ai_inpaint_bridge.pixelize import (
    KEY_CANDIDATES,
    opaque_colors,
    palette_from_strip,
    pick_key_color,
    to_sprite_pixels,
)

MAGENTA = KEY_CANDIDATES[0][0]
RED = (220, 30, 30)
DARK_RED = (120, 10, 10)
WHITE = (255, 255, 255)


def test_palette_strip_keeps_order_and_drops_transparent_entries() -> None:
    """Aseprite's transparent entry and duplicates are skipped."""
    strip = Image.new("RGBA", (4, 1))
    for x, pixel in enumerate(
        ((0, 0, 0, 0), (*WHITE, 255), (*RED, 255), (*WHITE, 255))
    ):
        strip.putpixel((x, 0), pixel)

    assert palette_from_strip(strip) == [WHITE, RED]


def test_opaque_colors_ignores_transparent_pixels() -> None:
    """Fully transparent pixels do not count as region colors."""
    image = Image.new("RGBA", (2, 1), (*RED, 255))
    image.putpixel((1, 0), (*WHITE, 0))

    assert opaque_colors(image) == [RED]


def test_key_color_avoids_the_region_colors() -> None:
    """Magenta by default, otherwise the candidate farthest from the region colors."""
    assert pick_key_color([])[0] == MAGENTA
    assert pick_key_color([RED]) == ((0, 255, 255), "cyan")
    assert pick_key_color([(250, 5, 250)])[0] != MAGENTA


def test_each_block_takes_its_majority_color() -> None:
    """A 3x3 block with 5 red and 4 white pixels becomes red."""
    big = Image.new("RGB", (6, 3), WHITE)
    for xy in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0)):
        big.putpixel(xy, RED)

    small = to_sprite_pixels(big, (2, 1), 3, palette=[RED, WHITE])

    assert small.getpixel((0, 0)) == (*RED, 255)
    assert small.getpixel((1, 0)) == (*WHITE, 255)


def test_colors_snap_to_the_palette() -> None:
    """A near-red pixel is replaced by the closest palette entry."""
    big = Image.new("RGB", (2, 2), (130, 20, 15))

    small = to_sprite_pixels(big, (1, 1), 2, palette=[RED, DARK_RED, WHITE])

    assert small.getpixel((0, 0)) == (*DARK_RED, 255)


def test_key_colored_blocks_become_transparent() -> None:
    """Blocks dominated by a color close to the key vote for transparency."""
    big = Image.new("RGB", (4, 2), (240, 20, 235))
    big.paste(RED, (2, 0, 4, 2))

    small = to_sprite_pixels(big, (2, 1), 2, palette=[RED], key=MAGENTA)

    pixel = small.getpixel((0, 0))
    assert isinstance(pixel, tuple) and pixel[3] == 0
    assert small.getpixel((1, 0)) == (*RED, 255)


def test_without_palette_colors_are_reduced() -> None:
    """The adaptive palette keeps clearly different colors apart."""
    big = Image.new("RGB", (4, 2), RED)
    big.paste(WHITE, (2, 0, 4, 2))

    small = to_sprite_pixels(big, (2, 1), 2)

    assert small.getpixel((0, 0)) == (*RED, 255)
    assert small.getpixel((1, 0)) == (*WHITE, 255)
