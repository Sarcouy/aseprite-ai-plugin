"""Inpainting: rewrite the selected pixels of a sprite region."""

import argparse
from dataclasses import replace
from pathlib import Path

from PIL import Image

from ai_inpaint_bridge.backends import diffusion_inpaint, invert_masked
from ai_inpaint_bridge.comfyui import SD15_INPAINT, ComfyUIClient
from ai_inpaint_bridge.klein import KLEIN_4B
from ai_inpaint_bridge.tasks.base import Task, TaskContext


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Region, mask and change strength."""
    parser.add_argument("--input", type=Path, required=True, help="region PNG")
    parser.add_argument(
        "--mask", type=Path, required=True, help="mask PNG, white = rewrite"
    )
    parser.add_argument(
        "--denoise", type=float, default=0.8, help="0 keeps the pixels, 1 redraws"
    )


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """Rewrite the masked pixels of the region, one image per variant."""
    settings = replace(context.settings, denoise=args.denoise)
    with Image.open(args.input) as source, Image.open(args.mask) as mask:
        if context.backend == "comfyui":
            client = ComfyUIClient(context.comfyui_url, model=context.model)
            return diffusion_inpaint(source, mask, settings, client, context.palette)
        return [invert_masked(source, mask)] * settings.batch_size


TASK = Task(
    name="inpaint",
    description="Rewrite the masked area of a sprite region.",
    models=(KLEIN_4B, SD15_INPAINT),
    add_arguments=add_arguments,
    run=run,
)
