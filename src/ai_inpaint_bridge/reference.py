"""Reference image of a sprite for klein edit mode, and the way back to its pixels."""

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageChops

from ai_inpaint_bridge.pixelize import (
    MEASURED_KEY_TOLERANCE,
    Color,
    foreground_mask,
    to_sprite_pixels,
)

GENERATION_SIZE = 512
STEPS_PER_PIXEL = 3
SEARCH_PIXELS = (2, 1)


@dataclass(frozen=True)
class ReferenceGrid:
    """Where the sprite pixels lie in its reference image.

    Each sprite pixel is a factor x factor block, pixel (0, 0) of the sprite
    starts at offset, which may be outside the reference.
    """

    factor: int
    offset: tuple[int, int]
    size: tuple[int, int]

    def place(self, sprite: Image.Image, key: Color) -> Image.Image:
        """The sprite enlarged on the grid, on the key color."""
        big = sprite.convert("RGBA").resize(
            (sprite.width * self.factor, sprite.height * self.factor),
            Image.Resampling.NEAREST,
        )
        canvas = Image.new("RGB", (GENERATION_SIZE, GENERATION_SIZE), key)
        canvas.paste(big, self.offset, big)
        return canvas

    def mask(self, sprite: Image.Image) -> Image.Image:
        """L mask of the opaque sprite pixels, enlarged on the grid."""
        alpha = (
            sprite.convert("RGBA")
            .getchannel("A")
            .point(lambda value: 255 if value else 0)
        )
        big = alpha.resize(
            (sprite.width * self.factor, sprite.height * self.factor),
            Image.Resampling.NEAREST,
        )
        canvas = Image.new("L", (GENERATION_SIZE, GENERATION_SIZE), 0)
        canvas.paste(big, self.offset)
        return canvas


def reference_grid(sprite: Image.Image) -> ReferenceGrid:
    """The grid enlarging the character by the largest integer factor that fits.

    The character, trimmed to its opaque pixels, is centered: drawn at the
    scale of a frame it is too small for the model to keep its look.
    """
    rgba = sprite.convert("RGBA")
    box = rgba.getchannel("A").getbbox() or (0, 0, *rgba.size)
    width, height = box[2] - box[0], box[3] - box[1]
    factor = max(1, GENERATION_SIZE // max(width, height))
    left = (GENERATION_SIZE - width * factor) // 2 - box[0] * factor
    top = (GENERATION_SIZE - height * factor) // 2 - box[1] * factor
    return ReferenceGrid(factor, (left, top), rgba.size)


def reference_image(sprite: Image.Image, key: Color) -> Image.Image:
    """The character trimmed, enlarged without smoothing and centered on the key."""
    return reference_grid(sprite).place(sprite, key)


def _coarse(mask: Image.Image, step: int) -> Image.Image:
    return mask.resize((mask.width // step, mask.height // step), Image.Resampling.BOX)


def _mismatch(drawn: Image.Image, target: Image.Image, shift: tuple[int, int]) -> int:
    moved = Image.new("L", drawn.size, 0)
    moved.paste(drawn, shift)
    histogram = ImageChops.difference(moved, target).histogram()
    return sum(value * count for value, count in enumerate(histogram))


def registration(
    image: Image.Image, key: Color, grid: ReferenceGrid, sprite: Image.Image
) -> tuple[int, int]:
    """Shift moving the character drawn on the key onto the sprite, by thirds of a block.

    The model redraws the character a little off the reference grid: without
    this shift its pixels would be cut across and the frame would jump. The
    silhouettes are compared at a coarse scale, within SEARCH_PIXELS sprite
    pixels, the smallest shift winning ties.
    """
    step = max(1, grid.factor // STEPS_PER_PIXEL)
    drawn = _coarse(foreground_mask(image, key, MEASURED_KEY_TOLERANCE), step)
    wanted = _coarse(grid.mask(sprite), step)
    reach_x = SEARCH_PIXELS[0] * STEPS_PER_PIXEL
    reach_y = SEARCH_PIXELS[1] * STEPS_PER_PIXEL
    shifts = [
        (x, y)
        for y in range(-reach_y, reach_y + 1)
        for x in range(-reach_x, reach_x + 1)
    ]
    best = min(
        shifts,
        key=lambda shift: (
            _mismatch(drawn, wanted, shift),
            abs(shift[0]) + abs(shift[1]),
        ),
    )
    return best[0] * step, best[1] * step


def sprite_on_grid(
    image: Image.Image,
    grid: ReferenceGrid,
    key: Color,
    shift: tuple[int, int],
    palette: Sequence[Color] | None,
) -> Image.Image:
    """The drawn image, moved by shift, reduced block by block to the sprite size."""
    width, height = grid.size
    canvas = Image.new("RGB", (width * grid.factor, height * grid.factor), key)
    canvas.paste(
        image.convert("RGB"), (shift[0] - grid.offset[0], shift[1] - grid.offset[1])
    )
    return to_sprite_pixels(
        canvas, grid.size, grid.factor, palette, key, tolerance=MEASURED_KEY_TOLERANCE
    )
