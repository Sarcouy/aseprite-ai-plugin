"""Turn a diffusion output back into sprite pixels: key out, quantize, vote."""

from collections import Counter
from collections.abc import Sequence

from PIL import Image

Color = tuple[int, int, int]

TRANSPARENT = -1
KEY_CANDIDATES: tuple[tuple[Color, str], ...] = (
    ((255, 0, 255), "magenta"),
    ((0, 255, 0), "bright green"),
    ((0, 255, 255), "cyan"),
    ((255, 255, 0), "yellow"),
)
KEY_TOLERANCE = 110
ADAPTIVE_COLORS = 16
MIN_PALETTE_COLORS = 2
PALETTE_SIZE = 256


def _distance2(first: Color, second: Color) -> int:
    return sum((a - b) ** 2 for a, b in zip(first, second))


def pixel_tuples(image: Image.Image) -> list[tuple[int, ...]]:
    """Pixels of a multi-band image as integer tuples, row by row."""
    return [
        tuple(int(c) for c in pixel)
        for pixel in image.get_flattened_data()
        if isinstance(pixel, tuple)
    ]


def band_values(image: Image.Image) -> list[int]:
    """Pixels of a single-band image as integers, row by row."""
    return [
        int(value)
        for value in image.get_flattened_data()
        if isinstance(value, (int, float))
    ]


def palette_from_strip(image: Image.Image) -> list[Color]:
    """Colors of a palette strip, in order, without duplicates or transparent entries."""
    colors: dict[Color, None] = {}
    for red, green, blue, alpha in pixel_tuples(image.convert("RGBA")):
        if alpha > 0:
            colors.setdefault((red, green, blue), None)
    return list(colors)


def opaque_colors(image: Image.Image) -> list[Color]:
    """Distinct colors of the pixels that are not fully transparent."""
    return sorted(
        {(r, g, b) for r, g, b, a in pixel_tuples(image.convert("RGBA")) if a > 0}
    )


def pick_key_color(colors: Sequence[Color]) -> tuple[Color, str]:
    """Return the key candidate farthest from every color of the region."""
    if not colors:
        return KEY_CANDIDATES[0]
    return max(
        KEY_CANDIDATES,
        key=lambda candidate: min(_distance2(candidate[0], c) for c in colors),
    )


def _palette_image(palette: Sequence[Color]) -> Image.Image:
    used = list(palette[:PALETTE_SIZE])
    padding = [used[0]] * (PALETTE_SIZE - len(used))
    image = Image.new("P", (1, 1))
    image.putpalette([channel for color in used + padding for channel in color])
    return image


def _quantize(
    image: Image.Image, palette: Sequence[Color] | None
) -> tuple[list[int], list[Color]]:
    if palette:
        colors = list(palette[:PALETTE_SIZE])
        quantized = image.quantize(
            palette=_palette_image(colors), dither=Image.Dither.NONE
        )
    else:
        quantized = image.quantize(colors=ADAPTIVE_COLORS, dither=Image.Dither.NONE)
        flat = quantized.getpalette() or []
        colors = [
            (flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat) - 2, 3)
        ]
    indices = [index if index < len(colors) else 0 for index in band_values(quantized)]
    return indices, colors


def _vote_blocks(labels: list[int], size: tuple[int, int], factor: int) -> list[int]:
    width = size[0] * factor
    winners: list[int] = []
    for block_y in range(size[1]):
        for block_x in range(size[0]):
            origin = block_y * factor * width + block_x * factor
            votes = Counter(
                labels[origin + dy * width + dx]
                for dy in range(factor)
                for dx in range(factor)
            )
            winners.append(votes.most_common(1)[0][0])
    return winners


def to_sprite_pixels(
    result: Image.Image,
    size: tuple[int, int],
    factor: int,
    palette: Sequence[Color] | None = None,
    key: Color | None = None,
) -> Image.Image:
    """Reduce each factor x factor block to its most frequent palette color.

    Pixels close to the key color vote for transparency.
    """
    cropped = result.convert("RGB").crop((0, 0, size[0] * factor, size[1] * factor))
    labels, colors = _quantize(cropped, palette)
    if key is not None:
        tolerance = KEY_TOLERANCE**2
        labels = [
            TRANSPARENT if _distance2((p[0], p[1], p[2]), key) < tolerance else label
            for p, label in zip(pixel_tuples(cropped), labels)
        ]
    small = Image.new("RGBA", size, (0, 0, 0, 0))
    for position, label in enumerate(_vote_blocks(labels, size, factor)):
        if label != TRANSPARENT:
            xy = (position % size[0], position // size[0])
            small.putpixel(xy, (*colors[label], 255))
    return small
