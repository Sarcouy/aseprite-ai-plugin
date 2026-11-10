"""Animation: frames of an existing character, each pose redrawn by klein from a guide."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageChops

from ai_inpaint_bridge.backends import GuidedGenerator
from ai_inpaint_bridge.comfyui import ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B
from ai_inpaint_bridge.pixelize import (
    Color,
    border_color,
    opaque_colors,
    pick_key_color,
)
from ai_inpaint_bridge.reference import (
    ReferenceGrid,
    reference_grid,
    registration,
    sprite_on_grid,
)
from ai_inpaint_bridge.tasks.base import Task, TaskContext

SECONDS_PER_POSE = 300
GUIDE_STRENGTH = 0.5
WAIST = 0.65
KEEP_LOOK = (
    " Keep every piece of clothing and equipment and every color exactly as in "
    "the reference; the arms stay down at the sides."
)


@dataclass(frozen=True)
class Pose:
    """Instruction to klein, and the rows the body above the waist is raised by."""

    instruction: str
    lift: int


@dataclass(frozen=True)
class Animation:
    """Poses and the order of the frames.

    Frame 0 of loop is the character as drawn, frame i the pose i.
    """

    poses: tuple[Pose, ...]
    loop: tuple[int, ...]


ANIMATIONS: dict[str, Animation] = {
    "idle": Animation(
        poses=(
            Pose(
                "Redraw this pixel art character cleanly, breathing out: head "
                "and shoulders slightly lowered." + KEEP_LOOK,
                lift=-1,
            ),
            Pose(
                "Redraw this pixel art character cleanly, breathing in: head "
                "and shoulders slightly raised." + KEEP_LOOK,
                lift=1,
            ),
        ),
        loop=(0, 1, 0, 2),
    ),
}


def guide(sprite: Image.Image, lift: int) -> Image.Image:
    """The sprite with the rows above its waist raised by lift pixels.

    Lowered, the upper body overlaps the waist; raised, the row above the
    waist is repeated to close the gap. klein redraws this rough pose.
    """
    rgba = sprite.convert("RGBA")
    box = rgba.getchannel("A").getbbox()
    if box is None or lift == 0:
        return rgba.copy()
    waist = box[1] + round((box[3] - box[1]) * WAIST)
    upper = rgba.crop((0, 0, rgba.width, waist))
    posed = rgba.copy()
    posed.paste((0, 0, 0, 0), (0, 0, rgba.width, waist))
    posed.paste(upper, (0, -lift))
    seam = rgba.crop((0, waist - 1, rgba.width, waist))
    for row in range(waist - lift, waist):
        posed.paste(seam, (0, row))
    return posed


def _union(images: Sequence[Image.Image]) -> Image.Image:
    alpha = images[0].getchannel("A")
    for image in images[1:]:
        alpha = ImageChops.lighter(alpha, image.getchannel("A"))
    union = images[0].copy()
    union.putalpha(alpha)
    return union


def pose_prompt(pose: Pose, description: str) -> str:
    """The pose instruction, followed by the user's details when given."""
    detail = description.strip()
    return f"{pose.instruction} {detail}" if detail else pose.instruction


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The character to animate and the animation."""
    parser.add_argument(
        "--reference", type=Path, required=True, help="PNG of the character"
    )
    parser.add_argument("--animation", choices=sorted(ANIMATIONS), default="idle")


def strip(frames: Sequence[Image.Image], loop: Sequence[int]) -> Image.Image:
    """The frames side by side in loop order, as one transparent image."""
    width, height = frames[0].size
    image = Image.new("RGBA", (width * len(loop), height), (0, 0, 0, 0))
    for position, frame in enumerate(loop):
        image.paste(frames[frame], (position * width, 0))
    return image


def _reduce(
    image: Image.Image,
    grid: ReferenceGrid,
    posed: Image.Image,
    colors: Sequence[Color],
) -> Image.Image:
    key = border_color(image)
    shift = registration(image, key, grid, posed)
    return sprite_on_grid(image, grid, key, shift, colors)


def draw_poses(
    sprite: Image.Image,
    animation: Animation,
    settings: InpaintSettings,
    generator: GuidedGenerator,
) -> list[list[Image.Image]]:
    """settings.batch_size drawings of each pose, at the size of the sprite.

    Each pose starts from its guide, noised to GUIDE_STRENGTH, with the
    character as drawn for reference: asked for a pose alone, klein redraws
    the details of the character every frame. Each drawing is reduced on the
    grid of the guide, so that the character keeps its scale and the pose its
    place, and to the colors of the character, so that the poses match the
    frames it is drawn in.
    """
    rgba = sprite.convert("RGBA")
    guides = [guide(rgba, pose.lift) for pose in animation.poses]
    colors = opaque_colors(rgba)
    key, key_name = pick_key_color(colors)
    grid = reference_grid(_union([rgba, *guides]))
    reference = grid.place(rgba, key)
    poses: list[list[Image.Image]] = []
    for pose, posed in zip(animation.poses, guides):
        asked = replace(
            settings,
            prompt=pose_prompt(pose, settings.prompt),
            denoise=GUIDE_STRENGTH,
            background=key_name,
            background_hint=True,
        )
        images = generator.generate_guided(reference, grid.place(posed, key), asked)
        if len(images) != settings.batch_size:
            raise ValueError(
                f"expected {settings.batch_size} generated images, got {len(images)}"
            )
        poses.append([_reduce(image, grid, posed, colors) for image in images])
    return poses


def animate(
    sprite: Image.Image,
    animation: Animation,
    settings: InpaintSettings,
    generator: GuidedGenerator,
) -> list[Image.Image]:
    """One strip of frames per variant, the character as drawn in frame 0."""
    poses = draw_poses(sprite, animation, settings, generator)
    original = sprite.convert("RGBA")
    return [
        strip([original, *(drawings[variant] for drawings in poses)], animation.loop)
        for variant in range(settings.batch_size)
    ]


def placeholder_poses(sprite: Image.Image, animation: Animation) -> list[Image.Image]:
    """The sprite, then the guide of every pose, as an offline stand-in."""
    rgba = sprite.convert("RGBA")
    return [rgba, *(guide(rgba, pose.lift) for pose in animation.poses)]


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """One strip per variant: the frames of the animation, in loop order."""
    animation = ANIMATIONS[args.animation]
    with Image.open(args.reference) as opened:
        sprite = opened.convert("RGBA")
    if sprite.getchannel("A").getbbox() is None:
        raise ValueError("the character to animate is empty")
    if context.backend == "comfyui":
        timeout = SECONDS_PER_POSE * (context.settings.batch_size + 1)
        client = ComfyUIClient(
            context.comfyui_url, timeout=timeout, model=context.model
        )
        return animate(sprite, animation, context.settings, client)
    frames = placeholder_poses(sprite, animation)
    return [strip(frames, animation.loop)] * context.settings.batch_size


TASK = Task(
    name="animation",
    description="Animate an existing character, klein redrawing a guide per pose.",
    models=(KLEIN_4B,),
    add_arguments=add_arguments,
    run=run,
)
