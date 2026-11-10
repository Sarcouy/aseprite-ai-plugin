"""Tests of the reference grid: enlargement, registration and reduction."""

from PIL import Image

from ai_inpaint_bridge.pixelize import Color, opaque_colors
from ai_inpaint_bridge.reference import (
    GENERATION_SIZE,
    reference_grid,
    registration,
    sprite_on_grid,
)

KEY: Color = (0, 255, 0)
DRAWN_KEY: Color = (20, 240, 30)


def knight() -> Image.Image:
    """A 32 x 32 sprite of two colors, 16 x 28 opaque pixels."""
    sprite = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    sprite.paste((40, 40, 200, 255), (8, 2, 24, 30))
    sprite.paste((230, 200, 60, 255), (12, 4, 20, 12))
    sprite.putpixel((8, 2), (0, 0, 0, 0))
    return sprite


def test_grid_enlarges_the_trimmed_character_to_fill_the_reference() -> None:
    """The longest side, 28 pixels, gets the largest factor fitting 512."""
    grid = reference_grid(knight())

    assert grid.factor == 18
    assert grid.offset == ((512 - 16 * 18) // 2 - 8 * 18, (512 - 28 * 18) // 2 - 36)
    assert grid.size == (32, 32)


def test_mask_and_placed_sprite_share_the_grid() -> None:
    """The opaque pixels are where the enlarged sprite is drawn."""
    sprite = knight()
    grid = reference_grid(sprite)

    placed = grid.place(sprite, KEY)
    mask = grid.mask(sprite)

    assert placed.size == mask.size == (GENERATION_SIZE, GENERATION_SIZE)
    assert mask.getbbox() == (112, 4, 400, 508)
    assert placed.getpixel((112, 4)) == KEY
    assert placed.getpixel((112 + 18, 4)) == (40, 40, 200)
    assert mask.getpixel((112 + 18, 4)) == 255


def drawn(sprite: Image.Image, shift: tuple[int, int]) -> Image.Image:
    """The reference as the model would redraw it, moved and on another green."""
    grid = reference_grid(sprite)
    moved = Image.new("RGB", (GENERATION_SIZE, GENERATION_SIZE), DRAWN_KEY)
    rgba = grid.place(sprite, KEY).convert("RGBA")
    rgba.putalpha(grid.mask(sprite))
    moved.paste(rgba, shift, rgba)
    return moved


def test_registration_moves_the_drawing_back_onto_the_grid() -> None:
    """A drawing off the grid by a third of a pixel and more is moved back."""
    sprite = knight()
    grid = reference_grid(sprite)

    assert registration(drawn(sprite, (18, -6)), DRAWN_KEY, grid, sprite) == (-18, 6)
    assert registration(drawn(sprite, (0, 0)), DRAWN_KEY, grid, sprite) == (0, 0)


def test_registered_drawing_reduces_to_the_sprite() -> None:
    """Back on the grid, each block votes for the pixel it was drawn from."""
    sprite = knight()
    grid = reference_grid(sprite)
    image = drawn(sprite, (24, 12))

    shift = registration(image, DRAWN_KEY, grid, sprite)
    frame = sprite_on_grid(image, grid, DRAWN_KEY, shift, opaque_colors(sprite))

    assert list(frame.get_flattened_data()) == list(sprite.get_flattened_data())
