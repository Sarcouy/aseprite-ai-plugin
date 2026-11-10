"""Autotile: the 16 wang tiles of the corners between two materials."""

import argparse
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from PIL import Image

from ai_inpaint_bridge.backends import RedrawGenerator, TextGenerator
from ai_inpaint_bridge.comfyui import ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B_TILE
from ai_inpaint_bridge.pixelize import Color, to_sprite_pixels
from ai_inpaint_bridge.tasks.base import Task, TaskContext
from ai_inpaint_bridge.tasks.tile import (
    GENERATION_SIZE,
    PLACEHOLDER_COLORS,
    add_arguments as add_tile_arguments,
    tile_prompt,
)

Corners = tuple[int, int, int, int]

GRID = 4
SHEET_SIZE = 512
REDRAW_STRENGTH = 0.5
WOBBLE = 0.08
CORNER_GRID: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 0),
    (1, 1, 0, 1),
    (0, 1, 0, 0),
    (1, 0, 0, 0),
)
TEMPLATE = (
    "seamless tileable pixel art top-down terrain tileset texture, {second} patches "
    "on {first}, natural irregular border between {first} and {second}, flat even "
    "lighting, large chunky pixels, simple 16-bit RPG game tile"
)


class SheetGenerator(TextGenerator, RedrawGenerator, Protocol):
    """Draws the material textures, then redraws the sheet guide."""


def sheet_prompt(first: str, second: str) -> str:
    """Both materials placed in the template of a transition sheet."""
    return TEMPLATE.format(first=first.strip(), second=second.strip())


def tile_corners(column: int, row: int) -> Corners:
    """Materials at the NW, NE, SW and SE corners of a tile of the sheet.

    CORNER_GRID repeats in both directions and holds each of the 16 corner
    combinations exactly once: every tile of the sheet is a different wang
    tile, and the sheet itself repeats without seams.
    """
    grid = CORNER_GRID
    right, below = (column + 1) % GRID, (row + 1) % GRID
    return (
        grid[row][column],
        grid[row][right],
        grid[below][column],
        grid[below][right],
    )


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def second_share(corners: Corners, u: float, v: float) -> float:
    """How much of the second material is at (u, v) of a tile, both in [0, 1].

    The corners are blended smoothly; the wobble, the same in every tile,
    bends the border without changing it along the tile edges.
    """
    su, sv = _smooth(u), _smooth(v)
    top = corners[0] * (1 - su) + corners[1] * su
    bottom = corners[2] * (1 - su) + corners[3] * su
    wobble = math.sin(2 * math.pi * (2 * u + v)) * math.sin(2 * math.pi * (3 * v - u))
    return top * (1 - sv) + bottom * sv + WOBBLE * wobble


def sheet_mask(size: int) -> Image.Image:
    """White where the sheet of the given side shows the second material."""
    cell = size // GRID
    mask = Image.new("L", (size, size))
    for y in range(size):
        for x in range(size):
            share = second_share(
                tile_corners(x // cell, y // cell),
                (x % cell + 0.5) / cell,
                (y % cell + 0.5) / cell,
            )
            if share > 0.5:
                mask.putpixel((x, y), 255)
    return mask


def repeated(texture: Image.Image, size: int) -> Image.Image:
    """The texture shrunk to one tile of the sheet, repeated over the sheet."""
    cell = size // GRID
    small = texture.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (size, size))
    for row in range(GRID):
        for column in range(GRID):
            sheet.paste(small, (column * cell, row * cell))
    return sheet


def sheet_guide(first: Image.Image, second: Image.Image, size: int) -> Image.Image:
    """Both textures cut along the corners of every tile, for the model to redraw."""
    return Image.composite(
        repeated(second, size), repeated(first, size), sheet_mask(size)
    )


def reduce_sheet(
    image: Image.Image, tile: int, palette: Sequence[Color] | None
) -> Image.Image:
    """The drawing averaged down to GRID x GRID tiles, then snapped to the palette."""
    side = GRID * tile
    small = image.convert("RGB").resize((side, side), Image.Resampling.BOX)
    return to_sprite_pixels(small, (side, side), 1, palette)


def generate_sheets(
    tile: int,
    settings: InpaintSettings,
    second: str,
    generator: SheetGenerator,
    palette: Sequence[Color] | None,
) -> list[Image.Image]:
    """settings.batch_size sheets between settings.prompt and the second material."""
    textures = [
        generator.generate_text(
            replace(settings, prompt=tile_prompt(material), batch_size=1),
            GENERATION_SIZE,
            GENERATION_SIZE,
        )[0]
        for material in (settings.prompt, second)
    ]
    asked = replace(
        settings,
        prompt=sheet_prompt(settings.prompt, second),
        denoise=REDRAW_STRENGTH,
    )
    images = generator.generate_redraw(
        sheet_guide(textures[0], textures[1], SHEET_SIZE), asked
    )
    if len(images) != settings.batch_size:
        raise ValueError(
            f"expected {settings.batch_size} generated images, got {len(images)}"
        )
    return [reduce_sheet(image, tile, palette) for image in images]


def placeholder(tile: int, palette: Sequence[Color] | None) -> Image.Image:
    """The sheet mask in the first two palette colors."""
    colors = list(palette[:2]) if palette and len(palette) >= 2 else PLACEHOLDER_COLORS
    side = GRID * tile
    first = Image.new("RGBA", (side, side), (*colors[0], 255))
    second = Image.new("RGBA", (side, side), (*colors[1], 255))
    return Image.composite(second, first, sheet_mask(side))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Side of a tile and second material."""
    add_tile_arguments(parser)
    parser.add_argument(
        "--second",
        default="",
        help="material of the patches drawn on the --prompt material",
    )


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """One sheet of GRID x GRID tiles per variant, its colors in the palette."""
    if not context.settings.prompt.strip() or not args.second.strip():
        raise ValueError("describe both materials of the autotile")
    if context.backend == "comfyui":
        client = ComfyUIClient(context.comfyui_url, model=context.model)
        return generate_sheets(
            args.size, context.settings, args.second, client, context.palette
        )
    return [placeholder(args.size, context.palette)] * context.settings.batch_size


TASK = Task(
    name="autotile",
    description="Draw the 16 wang tiles of the corners between two materials.",
    models=(KLEIN_4B_TILE,),
    add_arguments=add_arguments,
    run=run,
)
