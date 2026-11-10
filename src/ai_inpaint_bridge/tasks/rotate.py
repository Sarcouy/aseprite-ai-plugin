"""Rotations: the views of an existing character in 4 directions, drawn by klein."""

import argparse
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps

from ai_inpaint_bridge.backends import ReferenceGenerator
from ai_inpaint_bridge.comfyui import ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B
from ai_inpaint_bridge.pixelize import (
    MEASURED_KEY_TOLERANCE,
    border_color,
    foreground_box,
    opaque_colors,
    pick_key_color,
)
from ai_inpaint_bridge.reference import (
    GENERATION_SIZE,
    ReferenceGrid,
    reference_grid,
    sprite_on_grid,
)
from ai_inpaint_bridge.tasks.animation import strip
from ai_inpaint_bridge.tasks.base import Task, TaskContext

SECONDS_PER_VIEW = 300
TURN = (
    "Turn the character in the image to a {view}. Keep exactly the same character "
    "design, outfit, colors, proportions, size and pixel art style. Full body, "
    "centered."
)
SIDE_VIEW = "side profile view, facing right"
BACK_VIEW = (
    "back view, facing away from the viewer: the face is not visible, we see the "
    "back of the head and the back of the body, with anything worn on the back "
    "covering it"
)


def view_prompt(view: str, description: str) -> str:
    """The turn instruction for this view, followed by the user's details."""
    instruction = TURN.format(view=view)
    detail = description.strip()
    return f"{instruction} {detail}" if detail else instruction


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The character to turn."""
    parser.add_argument(
        "--reference", type=Path, required=True, help="PNG of the character"
    )


def _reduce(
    image: Image.Image, grid: ReferenceGrid, sprite: Image.Image
) -> Image.Image:
    key = border_color(image)
    drawn = foreground_box(image, key, MEASURED_KEY_TOLERANCE)
    if drawn is None:
        return Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    box = sprite.getchannel("A").getbbox() or (0, 0, *sprite.size)
    center = grid.offset[0] + (box[0] + box[2]) * grid.factor // 2
    feet = grid.offset[1] + box[3] * grid.factor
    shift = (center - (drawn[0] + drawn[2]) // 2, feet - drawn[3])
    return sprite_on_grid(image, grid, key, shift, opaque_colors(sprite))


def draw_views(
    sprite: Image.Image, settings: InpaintSettings, generator: ReferenceGenerator
) -> list[tuple[Image.Image, Image.Image]]:
    """settings.batch_size pairs of the side view facing right and the back view.

    Each view is drawn from the prompt, with the character as reference.
    Its feet are put on those of the character and its middle on the middle
    of the character, then it is reduced on the grid of the character, to
    its colors.
    """
    rgba = sprite.convert("RGBA")
    key, key_name = pick_key_color(opaque_colors(rgba))
    grid = reference_grid(rgba)
    reference = grid.place(rgba, key)
    views: list[list[Image.Image]] = []
    for view in (SIDE_VIEW, BACK_VIEW):
        asked = replace(
            settings,
            prompt=view_prompt(view, settings.prompt),
            background=key_name,
            background_hint=True,
        )
        images = generator.generate_reference(
            reference, asked, GENERATION_SIZE, GENERATION_SIZE
        )
        if len(images) != settings.batch_size:
            raise ValueError(
                f"expected {settings.batch_size} generated images, got {len(images)}"
            )
        views.append([_reduce(image, grid, rgba) for image in images])
    return list(zip(views[0], views[1]))


def directions(
    front: Image.Image, right: Image.Image, back: Image.Image
) -> Image.Image:
    """The views down, left, right and up, the left one mirroring the right one."""
    return strip([front, ImageOps.mirror(right), right, back], range(4))


def rotate(
    sprite: Image.Image, settings: InpaintSettings, generator: ReferenceGenerator
) -> list[Image.Image]:
    """One strip of the 4 directions per variant, the character as drawn first."""
    front = sprite.convert("RGBA")
    return [
        directions(front, right, back)
        for right, back in draw_views(front, settings, generator)
    ]


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """One strip per variant: the character facing down, left, right and up."""
    with Image.open(args.reference) as opened:
        sprite = opened.convert("RGBA")
    if sprite.getchannel("A").getbbox() is None:
        raise ValueError("the character to turn is empty")
    if context.backend == "comfyui":
        timeout = SECONDS_PER_VIEW * (context.settings.batch_size + 1)
        client = ComfyUIClient(
            context.comfyui_url, timeout=timeout, model=context.model
        )
        return rotate(sprite, context.settings, client)
    return [directions(sprite, sprite, sprite)] * context.settings.batch_size


TASK = Task(
    name="rotate",
    description="Turn an existing character to 4 directions, drawn by klein.",
    models=(KLEIN_4B,),
    add_arguments=add_arguments,
    run=run,
)
