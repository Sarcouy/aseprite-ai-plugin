"""Character: draw a game sprite from a description, on a transparent background."""

import argparse
import re
from collections.abc import Sequence
from dataclasses import replace

from PIL import Image, ImageDraw

from ai_inpaint_bridge.backends import TextGenerator
from ai_inpaint_bridge.comfyui import ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B
from ai_inpaint_bridge.pixelize import (
    Color,
    border_color,
    fit_to_sprite,
    foreground_box,
    pick_key_color,
)
from ai_inpaint_bridge.tasks.base import Task, TaskContext

MAX_SIZE = 64
PLACEHOLDER_COLOR: Color = (200, 10, 30)
GENERATION_SIZE = 512
GRID = 2
CHARACTER_TEMPLATE = (
    "a 2 by 2 grid of four separate top-down RPG overworld game sprites of {}, "
    "three-quarter view from above, full body, small and simple, evenly spaced"
)


def character_prompt(description: str) -> str:
    """Prompt sent to the model for a character described by the user."""
    return CHARACTER_TEMPLATE.format(description.strip())


def grid_cells(image: Image.Image) -> list[Image.Image]:
    """The GRID x GRID cells of an image, row by row."""
    width, height = image.width // GRID, image.height // GRID
    return [
        image.crop((x * width, y * height, (x + 1) * width, (y + 1) * height))
        for y in range(GRID)
        for x in range(GRID)
    ]


def _inside_area(cell: Image.Image) -> int:
    box = foreground_box(cell, border_color(cell))
    if box is None:
        return 0
    left, top, right, bottom = box
    if min(left, top) == 0 or right == cell.width or bottom == cell.height:
        return 0
    return (right - left) * (bottom - top)


def best_cell(image: Image.Image) -> Image.Image:
    """The cell holding the largest whole sprite, the image itself when none does.

    A sprite touching its cell edge is cut, or the model drew no grid.
    """
    cells = grid_cells(image)
    areas = [_inside_area(cell) for cell in cells]
    best = max(range(len(cells)), key=lambda index: areas[index])
    return cells[best] if areas[best] else image


def parse_size(text: str) -> tuple[int, int]:
    """Read "32" as 32x32, "32x48" as 32 wide and 48 high."""
    match = re.fullmatch(r"(\d+)(?:x(\d+))?", text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"{text} is not N or WxH")
    width = int(match.group(1))
    height = int(match.group(2) or width)
    if not (0 < width <= MAX_SIZE and 0 < height <= MAX_SIZE):
        raise argparse.ArgumentTypeError(f"{text} is not within 1..{MAX_SIZE}")
    return width, height


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Sprite size."""
    parser.add_argument(
        "--size", type=parse_size, default=(32, 32), help="N or WxH, in pixels"
    )


def placeholder(size: tuple[int, int], color: Color) -> Image.Image:
    """A box of half the size, centered on a transparent sprite."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    width, height = size
    box = (width // 4, height // 4, width - width // 4 - 1, height - height // 4 - 1)
    ImageDraw.Draw(image).rectangle(box, fill=(*color, 255))
    return image


def generate_characters(
    size: tuple[int, int],
    settings: InpaintSettings,
    generator: TextGenerator,
    palette: Sequence[Color] | None,
) -> list[Image.Image]:
    """Draw settings.batch_size characters described by settings.prompt.

    Each image is a grid of sprites on the key color, of which the best cell
    is cropped around its sprite: drawn small, the sprites are simpler and
    keep their shape once reduced, and the variants come from distinct images.
    """
    character = replace(
        settings,
        prompt=character_prompt(settings.prompt),
        background=pick_key_color(list(palette or []))[1],
        background_hint=True,
    )
    images = generator.generate_text(character, GENERATION_SIZE, GENERATION_SIZE)
    if len(images) != settings.batch_size:
        raise ValueError(
            f"expected {settings.batch_size} generated images, got {len(images)}"
        )
    return [fit_to_sprite(best_cell(image), size, palette) for image in images]


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """One transparent sprite per variant, its colors in the palette."""
    if not context.settings.prompt.strip():
        raise ValueError("describe the character to draw")
    if context.backend == "comfyui":
        client = ComfyUIClient(context.comfyui_url, model=context.model)
        return generate_characters(args.size, context.settings, client, context.palette)
    color = context.palette[0] if context.palette else PLACEHOLDER_COLOR
    return [placeholder(args.size, color)] * context.settings.batch_size


TASK = Task(
    name="character",
    description="Draw a character sprite from a description.",
    models=(KLEIN_4B,),
    add_arguments=add_arguments,
    run=run,
)
