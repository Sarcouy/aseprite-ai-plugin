"""Tile: a square of a material that repeats without visible seams."""

import argparse
from collections.abc import Sequence
from dataclasses import replace

from PIL import Image

from ai_inpaint_bridge.backends import TextGenerator
from ai_inpaint_bridge.comfyui import ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B_TILE
from ai_inpaint_bridge.pixelize import Color, to_sprite_pixels
from ai_inpaint_bridge.tasks.base import Task, TaskContext
from ai_inpaint_bridge.tasks.character import parse_size

GENERATION_SIZE = 256
TILE_SIZES = (16, 32)
TEMPLATE = (
    "seamless tileable pixel art texture of {material}, top-down, flat even "
    "lighting, repeating pattern, no objects, large chunky pixels, simple 16-bit "
    "RPG game tile, a few big shapes"
)
PLACEHOLDER_COLORS: tuple[Color, Color] = ((90, 150, 60), (60, 110, 40))


def tile_prompt(material: str) -> str:
    """The material placed in the template of a seamless texture."""
    return TEMPLATE.format(material=material.strip())


def parse_tile_size(text: str) -> int:
    """Side of the tile, given as N or NxN, among TILE_SIZES."""
    width, height = parse_size(text)
    if width != height or width not in TILE_SIZES:
        sizes = " or ".join(str(size) for size in TILE_SIZES)
        raise argparse.ArgumentTypeError(f"a tile is a square of {sizes} pixels")
    return width


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Side of the tile."""
    parser.add_argument(
        "--size",
        type=parse_tile_size,
        default=32,
        help="side of the square tile in pixels, N or NxN",
    )


def reduce_tile(
    image: Image.Image, size: int, palette: Sequence[Color] | None
) -> Image.Image:
    """The drawing averaged down to size x size pixels, then snapped to the palette.

    Averaging keeps the thin dark lines of the drawing (the joints between
    slabs) as darker pixels, where a vote per block drops them.
    """
    small = image.convert("RGB").resize((size, size), Image.Resampling.BOX)
    return to_sprite_pixels(small, (size, size), 1, palette)


def generate_tiles(
    size: int,
    settings: InpaintSettings,
    generator: TextGenerator,
    palette: Sequence[Color] | None,
) -> list[Image.Image]:
    """settings.batch_size tiles of the material described by settings.prompt."""
    asked = replace(settings, prompt=tile_prompt(settings.prompt))
    images = generator.generate_text(asked, GENERATION_SIZE, GENERATION_SIZE)
    if len(images) != settings.batch_size:
        raise ValueError(
            f"expected {settings.batch_size} generated images, got {len(images)}"
        )
    return [reduce_tile(image, size, palette) for image in images]


def placeholder(size: int, palette: Sequence[Color] | None) -> Image.Image:
    """A checkerboard of 4 squares, which repeats without seams."""
    colors = list(palette[:2]) if palette and len(palette) >= 2 else PLACEHOLDER_COLORS
    tile = Image.new("RGBA", (size, size), (*colors[0], 255))
    half = size // 2
    tile.paste((*colors[1], 255), (half, 0, size, half))
    tile.paste((*colors[1], 255), (0, half, half, size))
    return tile


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """One tile per variant, its colors in the palette."""
    if not context.settings.prompt.strip():
        raise ValueError("describe the material of the tile")
    if context.backend == "comfyui":
        client = ComfyUIClient(context.comfyui_url, model=context.model)
        return generate_tiles(args.size, context.settings, client, context.palette)
    return [placeholder(args.size, context.palette)] * context.settings.batch_size


TASK = Task(
    name="tile",
    description="Draw a square tile of a material that repeats without seams.",
    models=(KLEIN_4B_TILE,),
    add_arguments=add_arguments,
    run=run,
)
