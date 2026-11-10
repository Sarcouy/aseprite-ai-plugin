"""Walk sheet: four directions of a walking character, sliced into frames."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

from PIL import Image

from ai_inpaint_bridge.backends import ReferenceGenerator, TextGenerator
from ai_inpaint_bridge.comfyui import ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B_BASE_WALK
from ai_inpaint_bridge.pixelize import (
    MEASURED_KEY_TOLERANCE,
    Color,
    border_color,
    foreground_mask,
    opaque_colors,
    pick_key_color,
    to_sprite_pixels,
)
from ai_inpaint_bridge.reference import reference_image
from ai_inpaint_bridge.tasks.base import Task, TaskContext
from ai_inpaint_bridge.tasks.character import PLACEHOLDER_COLOR, placeholder

FRAME = 32
GRID = 4
GENERATION_SIZE = 512
FACTOR = GENERATION_SIZE // (GRID * FRAME)
MARGIN = 1
MIN_RUN = 4
SECONDS_PER_VARIANT = 300
WALK_LAYOUT = (
    "The spritesheet is a 4 by 4 grid of four rows of frames - first row is 3 "
    "walking frames facing down and 1 frame both arms raised, second row is 3 "
    "walking frames facing left and 1 frame jumping left, third row is 3 walking "
    "frames facing right and 1 frame jumping right, fourth row is 3 walking frames "
    "back view facing up and 1 frame lying on floor."
)
WALK_TEMPLATE = "A pixel art spritesheet of {}. " + WALK_LAYOUT
EDIT_TEMPLATE = "Create a pixel art spritesheet of the character in the image{}. "

Band = tuple[int, int]
Box = tuple[int, int, int, int]


def walk_prompt(description: str) -> str:
    """Prompt sent to the model, worded as on the card of the svntax LoRA."""
    return WALK_TEMPLATE.format(description.strip().rstrip("."))


def edit_prompt(description: str) -> str:
    """Prompt sent with a reference image, worded as on the card of the LoRA."""
    detail = description.strip().rstrip(".")
    return EDIT_TEMPLATE.format(f", {detail}" if detail else "") + WALK_LAYOUT


def foreground_runs(counts: Sequence[int]) -> list[Band]:
    """Runs of consecutive non-zero counts, the shortest ones dropped as noise."""
    runs: list[Band] = []
    start = None
    for index, count in enumerate([*counts, 0]):
        if count and start is None:
            start = index
        elif not count and start is not None:
            if index - start >= MIN_RUN:
                runs.append((start, index))
            start = None
    return runs


def group_bands(runs: Sequence[Band], count: int) -> list[Band] | None:
    """Merge the runs across their narrowest gaps until count bands remain.

    None when there are fewer runs than bands.
    """
    bands = list(runs)
    if len(bands) < count:
        return None
    while len(bands) > count:
        gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
        narrowest = gaps.index(min(gaps))
        bands[narrowest : narrowest + 2] = [
            (bands[narrowest][0], bands[narrowest + 1][1])
        ]
    return bands


def band_limits(bands: Sequence[Band], length: int) -> list[Band]:
    """Areas of the bands, split at the middle of the gaps between them."""
    cuts = [0]
    cuts += [(bands[i][1] + bands[i + 1][0]) // 2 for i in range(len(bands) - 1)]
    cuts.append(length)
    return [(cuts[i], cuts[i + 1]) for i in range(len(bands))]


def _quarters(length: int) -> list[Band]:
    return [(i * length // GRID, (i + 1) * length // GRID) for i in range(GRID)]


def _projection(mask: Image.Image, axis: int) -> list[int]:
    """Foreground pixels of each column (axis 0) or row (axis 1) of an L mask."""
    lines = mask.transpose(Image.Transpose.TRANSPOSE) if axis == 0 else mask
    width = lines.width
    data = lines.tobytes()
    return [
        width - data[start : start + width].count(0)
        for start in range(0, len(data), width)
    ]


def _split(mask: Image.Image, axis: int) -> list[Band]:
    length = mask.size[axis]
    bands = group_bands(foreground_runs(_projection(mask, axis)), GRID)
    return band_limits(bands, length) if bands else _quarters(length)


def locate_cells(mask: Image.Image) -> list[list[Box]]:
    """GRID x GRID areas of the sheet, row by row, each holding one sprite.

    Rows are split at the background bands between them, then the columns of
    each row; a row or column that cannot be told apart falls back to equal
    quarters.
    """
    cells: list[list[Box]] = []
    for top, bottom in _split(mask, 1):
        row = mask.crop((0, top, mask.width, bottom))
        cells.append([(left, top, right, bottom) for left, right in _split(row, 0)])
    return cells


@dataclass(frozen=True)
class Placement:
    """Where the frames of one row are cut: the column centers and the baseline.

    cells are the areas of the sheet holding each sprite, sprites their
    bounding boxes, None for an empty cell.
    """

    cells: Sequence[Box]
    sprites: Sequence[Box | None]
    centers: list[float]
    baseline: int

    def factor(self) -> int:
        """Smallest reduction factor at which every sprite fits its frame."""
        return max(
            _needed_factor(box, center, self.baseline)
            for box, center in zip(self.sprites, self.centers)
        )


def _sprite_box(mask: Image.Image, cell: Box) -> Box | None:
    box = mask.crop(cell).getbbox()
    if box is None:
        return None
    return (box[0] + cell[0], box[1] + cell[1], box[2] + cell[0], box[3] + cell[1])


def place_row(cells: Sequence[Box], sprites: Sequence[Box | None]) -> Placement:
    """Column centers on the regular step of the grid, common baseline.

    The centers are shifted by the median offset of the sprites from their
    cell: unlike centering each sprite on itself, this keeps the body still
    when a swinging arm or cape widens one frame.
    """
    step = (cells[-1][2] - cells[0][0]) / len(cells)
    offsets = [
        (box[0] + box[2]) / 2 - (cells[0][0] + step * (column + 0.5))
        for column, box in enumerate(sprites)
        if box is not None
    ]
    shift = median(offsets) if offsets else 0.0
    centers = [
        cells[0][0] + step * (column + 0.5) + shift for column in range(len(cells))
    ]
    bottoms = [box[3] for box in sprites if box is not None]
    baseline = max(bottoms) if bottoms else cells[0][3]
    return Placement(cells, sprites, centers, baseline)


def _window(center: float, baseline: int, factor: int) -> Box:
    size = FRAME * factor
    left = round((center - size / 2) / factor) * factor
    bottom = -(-(baseline + MARGIN * factor) // factor) * factor
    return (left, bottom - size, left + size, bottom)


def _needed_factor(box: Box | None, center: float, baseline: int) -> int:
    """Smallest factor at which the sprite fits its frame, FACTOR at least."""
    if box is None:
        return FACTOR
    inner = FRAME - 2 * MARGIN
    width = 2 * max(center - box[0], box[2] - center)
    return max(FACTOR, -(-int(width) // inner), -(-(baseline - box[1]) // inner))


def _frame(image: Image.Image, cell: Box, window: Box, key: Color) -> Image.Image:
    """The window of the sheet on the key color, its neighbours' pixels removed."""
    canvas = Image.new("RGB", (window[2] - window[0], window[3] - window[1]), key)
    left, top = max(cell[0], window[0]), max(cell[1], window[1])
    right, bottom = min(cell[2], window[2]), min(cell[3], window[3])
    if left < right and top < bottom:
        canvas.paste(
            image.crop((left, top, right, bottom)),
            (left - window[0], top - window[1]),
        )
    return canvas


def _cut(
    image: Image.Image, placement: Placement, column: int, factor: int, key: Color
) -> Image.Image:
    """Frame of the given column, at factor times its final size, on the key."""
    window = _window(placement.centers[column], placement.baseline, factor)
    return _frame(image, placement.cells[column], window, key)


def slice_sheet(
    image: Image.Image, palette: Sequence[Color] | None = None
) -> Image.Image:
    """Reduce a generated sheet to GRID x GRID frames of FRAME pixels.

    Every frame is reduced by the same factor, FACTOR unless a sprite would
    not fit, so that the character keeps its size from frame to frame. The
    background, measured on the border, becomes transparent.
    """
    rgb = image.convert("RGB")
    key = border_color(rgb)
    mask = foreground_mask(rgb, key, MEASURED_KEY_TOLERANCE)
    rows = [
        place_row(row, [_sprite_box(mask, cell) for cell in row])
        for row in locate_cells(mask)
    ]
    factor = max(row.factor() for row in rows)
    sheet = Image.new("RGBA", (GRID * FRAME, GRID * FRAME), (0, 0, 0, 0))
    for row_index, row in enumerate(rows):
        for column in range(len(row.cells)):
            canvas = _cut(rgb, row, column, factor, key)
            frame = to_sprite_pixels(
                canvas,
                (FRAME, FRAME),
                factor,
                palette,
                key,
                tolerance=MEASURED_KEY_TOLERANCE,
            )
            sheet.paste(frame, (column * FRAME, row_index * FRAME))
    return sheet


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Reference sprite of the character, the frame size is set by the LoRA."""
    parser.add_argument(
        "--reference",
        type=Path,
        help="PNG of the character to animate, the description is then optional",
    )


def _sheets(
    images: Sequence[Image.Image], count: int, palette: Sequence[Color] | None
) -> list[Image.Image]:
    if len(images) != count:
        raise ValueError(f"expected {count} generated images, got {len(images)}")
    return [slice_sheet(image, palette) for image in images]


def generate_walk_sheets(
    settings: InpaintSettings,
    generator: TextGenerator,
    palette: Sequence[Color] | None,
) -> list[Image.Image]:
    """Draw settings.batch_size walk sheets of the character in settings.prompt.

    The sheet is asked on a key color: without it the model draws on white,
    and the light highlights of the character would become transparent.
    """
    walk = replace(
        settings,
        prompt=walk_prompt(settings.prompt),
        background=pick_key_color(list(palette or []))[1],
        background_hint=True,
    )
    images = generator.generate_text(walk, GENERATION_SIZE, GENERATION_SIZE)
    return _sheets(images, settings.batch_size, palette)


def generate_walk_sheets_from(
    sprite: Image.Image,
    settings: InpaintSettings,
    generator: ReferenceGenerator,
    palette: Sequence[Color] | None,
) -> list[Image.Image]:
    """Draw settings.batch_size walk sheets of the character drawn in sprite.

    settings.prompt, optional, adds to the description of the character.
    """
    key, key_name = pick_key_color(opaque_colors(sprite) or list(palette or []))
    walk = replace(
        settings,
        prompt=edit_prompt(settings.prompt),
        background=key_name,
        background_hint=True,
    )
    reference = reference_image(sprite, key)
    images = generator.generate_reference(
        reference, walk, GENERATION_SIZE, GENERATION_SIZE
    )
    return _sheets(images, settings.batch_size, palette)


def placeholder_sheet(color: Color) -> Image.Image:
    """A box per frame, shifted by one pixel per column so that it moves."""
    sheet = Image.new("RGBA", (GRID * FRAME, GRID * FRAME), (0, 0, 0, 0))
    frame = placeholder((FRAME, FRAME), color)
    for row in range(GRID):
        for column in range(GRID):
            sheet.paste(frame, (column * FRAME + column, row * FRAME))
    return sheet


def run(args: argparse.Namespace, context: TaskContext) -> list[Image.Image]:
    """One sheet of GRID x GRID frames per variant, transparent, in the palette."""
    if args.reference is None and not context.settings.prompt.strip():
        raise ValueError("describe the character to animate")
    if context.backend == "comfyui":
        timeout = SECONDS_PER_VARIANT * (context.settings.batch_size + 1)
        client = ComfyUIClient(
            context.comfyui_url, timeout=timeout, model=context.model
        )
        if args.reference is None:
            return generate_walk_sheets(context.settings, client, context.palette)
        with Image.open(args.reference) as sprite:
            return generate_walk_sheets_from(
                sprite, context.settings, client, context.palette
            )
    color = context.palette[0] if context.palette else PLACEHOLDER_COLOR
    return [placeholder_sheet(color)] * context.settings.batch_size


TASK = Task(
    name="walk-sheet",
    description="Draw the four-direction walk sheet of a character.",
    models=(KLEIN_4B_BASE_WALK,),
    add_arguments=add_arguments,
    run=run,
)
