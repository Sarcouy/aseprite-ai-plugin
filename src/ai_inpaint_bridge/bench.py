"""Quality bench: fixed cases, prompts and seeds rendered into one comparison sheet.

Inpaint cases rewrite a selection of a drawn scene; sprite cases draw from the
prompt alone.
"""

import argparse
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from ai_inpaint_bridge.backends import (
    InpaintGenerator,
    TextGenerator,
    changed_fraction,
    diffusion_generate,
    diffusion_inpaint,
)
from ai_inpaint_bridge.comfyui import (
    DEFAULT_URL,
    ComfyUIClient,
    InpaintSettings,
    ModelProfile,
)
from ai_inpaint_bridge.klein import KLEIN_4B_BASE
from ai_inpaint_bridge.pixelize import Color
from ai_inpaint_bridge.tasks import TASKS

BENCH_MODELS: tuple[ModelProfile, ...] = (*TASKS["inpaint"].models, KLEIN_4B_BASE)

DB32_PATH = Path(
    "/usr/local/share/aseprite/data/extensions/dawnbringer-palettes/db32.gpl"
)
CONTEXT_MARGIN = 8
CELL = 160
GAP = 6
LABEL_WIDTH = 190
LABEL_CHARS = 30
HEADER = 24
SHEET_BACKGROUND = (40, 40, 48)
LABEL_COLOR = (230, 230, 230)
CHECKER = ((200, 200, 200), (160, 160, 160))
CHECKER_SIZE = 10

Box = tuple[int, int, int, int]
Painter = Callable[[ImageDraw.ImageDraw, Sequence[Color]], None]


@dataclass(frozen=True)
class BenchCase:
    """One scene to inpaint: canvas, selection box (left, top, right, bottom), prompt."""

    name: str
    prompt: str
    size: tuple[int, int]
    selection: Box
    paint: Painter
    denoise: float = 1.0


@dataclass(frozen=True)
class SpriteCase:
    """One sprite drawn from the prompt alone, at its final size.

    A tile is opaque and shown repeated 3x3 on the sheet to reveal its seams.
    """

    name: str
    prompt: str
    size: tuple[int, int]
    transparent: bool = True
    tile: bool = False


Case = BenchCase | SpriteCase


@dataclass(frozen=True)
class BenchRow:
    """The variants of one case, with the region and mask sent for inpaint cases."""

    case: Case
    variants: list[Image.Image]
    original: Image.Image | None = None
    mask: Image.Image | None = None


class BenchGenerator(InpaintGenerator, TextGenerator, Protocol):
    """A generator for both kinds of cases."""


def load_gpl(path: Path) -> list[Color]:
    """Colors of a GIMP palette file, in order."""
    colors: list[Color] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3 and all(field.isdigit() for field in fields[:3]):
            colors.append((int(fields[0]), int(fields[1]), int(fields[2])))
    return colors


def _ground(top: int) -> Painter:
    def paint(draw: ImageDraw.ImageDraw, palette: Sequence[Color]) -> None:
        width = draw.im.size[0]
        draw.rectangle((0, top, width - 1, top + 1), fill=palette[12])
        draw.rectangle((0, top + 2, width - 1, 999), fill=palette[4])

    return paint


def _sky_and_ground(top: int) -> Painter:
    def paint(draw: ImageDraw.ImageDraw, palette: Sequence[Color]) -> None:
        width = draw.im.size[0]
        draw.rectangle((0, 0, width - 1, top - 1), fill=palette[18])
        _ground(top)(draw, palette)

    return paint


def _nothing(draw: ImageDraw.ImageDraw, palette: Sequence[Color]) -> None:
    del draw, palette


def _rock(draw: ImageDraw.ImageDraw, palette: Sequence[Color]) -> None:
    draw.ellipse((8, 12, 24, 26), fill=palette[23], outline=palette[1])
    draw.ellipse((11, 14, 17, 18), fill=palette[22])


BENCH_CASES: tuple[BenchCase, ...] = (
    BenchCase("potion", "red potion bottle", (32, 32), (10, 8, 22, 24), _ground(24)),
    BenchCase(
        "chest",
        "wooden treasure chest",
        (32, 32),
        (8, 12, 24, 26),
        _sky_and_ground(24),
    ),
    BenchCase("slime", "green slime monster", (24, 24), (4, 6, 20, 20), _nothing),
    BenchCase("tree", "pine tree", (48, 48), (14, 6, 34, 40), _sky_and_ground(40)),
    BenchCase("sword", "steel sword, vertical", (32, 32), (13, 4, 19, 28), _nothing),
    BenchCase("rock", "mossy rock", (32, 32), (6, 10, 26, 28), _rock, denoise=0.6),
    BenchCase("rock-cut", "mossy rock", (32, 32), (16, 10, 28, 28), _rock, denoise=0.8),
)

KNIGHT = "top-down view, a knight character with a blue cape"
WALK = (
    "A pixel art spritesheet of a knight character with a blue cape. The spritesheet "
    "is a 4 by 4 grid of four rows of frames: the first row walks facing down, the "
    "second row walks facing left, the third row walks facing right, the fourth row "
    "walks facing up."
)
SPRITE_CASES: tuple[SpriteCase, ...] = (
    SpriteCase("knight-16", KNIGHT, (16, 16)),
    SpriteCase("knight-32", KNIGHT, (32, 32)),
    SpriteCase(
        "grass-tile",
        "top-down view, seamless grass ground texture filling the whole image",
        (16, 16),
        transparent=False,
        tile=True,
    ),
    SpriteCase(
        "wall-tile",
        "top-down view, seamless grey stone brick wall texture filling the whole image",
        (16, 16),
        transparent=False,
        tile=True,
    ),
    SpriteCase("knight-walk", WALK, (128, 128)),
)
ALL_CASES: tuple[Case, ...] = (*BENCH_CASES, *SPRITE_CASES)


def render_case(
    case: BenchCase, palette: Sequence[Color]
) -> tuple[Image.Image, Image.Image]:
    """Return the region around the selection and its mask, as the extension sends them."""
    canvas = Image.new("RGBA", case.size, (0, 0, 0, 0))
    case.paint(ImageDraw.Draw(canvas), palette)
    left, top, right, bottom = case.selection
    region = (
        max(0, left - CONTEXT_MARGIN),
        max(0, top - CONTEXT_MARGIN),
        min(case.size[0], right + CONTEXT_MARGIN),
        min(case.size[1], bottom + CONTEXT_MARGIN),
    )
    mask = Image.new("L", case.size, 0)
    mask.paste(255, case.selection)
    return canvas.crop(region), mask.crop(region)


def run_case(
    case: Case,
    settings: InpaintSettings,
    generator: BenchGenerator,
    palette: Sequence[Color],
    denoise: float | None = None,
) -> BenchRow:
    """Generate the variants of one case.

    denoise replaces the change strength of inpaint cases when given.
    """
    if isinstance(case, SpriteCase):
        sprite_settings = replace(settings, prompt=case.prompt)
        variants = diffusion_generate(
            case.size, sprite_settings, generator, palette, case.transparent
        )
        return BenchRow(case, variants)
    source, mask = render_case(case, palette)
    strength = case.denoise if denoise is None else denoise
    case_settings = replace(settings, prompt=case.prompt, denoise=strength)
    variants = diffusion_inpaint(source, mask, case_settings, generator, palette)
    return BenchRow(case, variants, source, mask)


def run_bench(
    cases: Sequence[Case],
    settings: InpaintSettings,
    generator: BenchGenerator,
    palette: Sequence[Color],
    denoise: float | None = None,
) -> list[BenchRow]:
    """Run every case with the same settings (and seed)."""
    return [run_case(case, settings, generator, palette, denoise) for case in cases]


def tiled_preview(tile: Image.Image, repeat: int = 3) -> Image.Image:
    """The tile repeated repeat x repeat times."""
    preview = Image.new(tile.mode, (tile.width * repeat, tile.height * repeat))
    for row in range(repeat):
        for column in range(repeat):
            preview.paste(tile, (column * tile.width, row * tile.height))
    return preview


def _checkered(image: Image.Image, zoom: int) -> Image.Image:
    big = image.convert("RGBA").resize(
        (image.width * zoom, image.height * zoom), Image.Resampling.NEAREST
    )
    board = Image.new("RGBA", big.size)
    draw = ImageDraw.Draw(board)
    for y in range(0, big.height, CHECKER_SIZE):
        for x in range(0, big.width, CHECKER_SIZE):
            color = CHECKER[(x // CHECKER_SIZE + y // CHECKER_SIZE) % 2]
            draw.rectangle(
                (x, y, x + CHECKER_SIZE - 1, y + CHECKER_SIZE - 1), fill=color
            )
    board.alpha_composite(big)
    return board


def _paste_row(sheet: Image.Image, row: BenchRow, top: int) -> None:
    prompt = "\n".join(textwrap.wrap(row.case.prompt, LABEL_CHARS))
    ImageDraw.Draw(sheet).text(
        (GAP, top), f"{row.case.name}\n{prompt}", fill=LABEL_COLOR
    )
    variants = row.variants
    if isinstance(row.case, SpriteCase) and row.case.tile:
        variants = [tiled_preview(variant) for variant in variants]
    shown = row.original if row.original is not None else variants[0]
    zoom = max(1, CELL // max(shown.size))
    for column, image in enumerate([row.original, *variants]):
        if image is None:
            continue
        tile = _checkered(image, zoom)
        left = LABEL_WIDTH + GAP + column * (CELL + GAP)
        sheet.paste(
            tile.convert("RGB"),
            (left + (CELL - tile.width) // 2, top + (CELL - tile.height) // 2),
        )
        if column > 0 and row.original is not None and row.mask is not None:
            changed = changed_fraction(row.original, image, row.mask)
            ImageDraw.Draw(sheet).text(
                (left + 2, top + 2), f"{changed:.0%}", fill=(255, 64, 64)
            )


def render_sheet(rows: Sequence[BenchRow], title: str) -> Image.Image:
    """One line per case: label, original region, then every variant."""
    columns = 1 + max(len(row.variants) for row in rows)
    width = LABEL_WIDTH + columns * (CELL + GAP) + GAP
    height = HEADER + len(rows) * (CELL + GAP) + GAP
    sheet = Image.new("RGB", (width, height), SHEET_BACKGROUND)
    ImageDraw.Draw(sheet).text((GAP, GAP), title, fill=LABEL_COLOR)
    for index, row in enumerate(rows):
        _paste_row(sheet, row, HEADER + GAP + index * (CELL + GAP))
    return sheet


def _strength(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not in (0, 1]")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-inpaint-bench", description="Render the quality comparison sheet."
    )
    parser.add_argument("--output", type=Path, required=True, help="sheet PNG")
    parser.add_argument("--palette", type=Path, default=DB32_PATH, help="GIMP palette")
    parser.add_argument(
        "--model",
        choices=[profile.name for profile in BENCH_MODELS],
        default=BENCH_MODELS[0].name,
        help="base model, its settings are used unless overridden below",
    )
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument(
        "--denoise",
        type=_strength,
        help="change strength of every inpaint case, in (0, 1], their own when omitted",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--sampler")
    parser.add_argument("--scheduler")
    parser.add_argument("--mask-grow", type=int)
    parser.add_argument(
        "--lora", help='LoRA file, "" for none, the model one when omitted'
    )
    parser.add_argument("--lora-strength", type=float)
    parser.add_argument(
        "--background-hint",
        action="store_true",
        default=None,
        help="name the key color in the prompt",
    )
    parser.add_argument(
        "--inpaint-model",
        help='diffusion_models UNet of SD 1.5, "" to use the checkpoint one',
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=[case.name for case in ALL_CASES],
        help="case names, all when omitted",
    )
    parser.add_argument("--comfyui-url", default=DEFAULT_URL)
    return parser.parse_args(argv)


OVERRIDABLE = (
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "mask_grow",
    "lora",
    "lora_strength",
    "background_hint",
    "inpaint_model",
)


def bench_settings(args: argparse.Namespace, model: ModelProfile) -> InpaintSettings:
    """Settings of the model, with the options given on the command line."""
    overrides = {
        name: getattr(args, name)
        for name in OVERRIDABLE
        if getattr(args, name) is not None
    }
    return replace(
        model.settings,
        seed=args.seed,
        batch_size=args.variants,
        **overrides,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run every case on ComfyUI and save the sheet."""
    args = _parse_args(argv)
    model = next(profile for profile in BENCH_MODELS if profile.name == args.model)
    settings = bench_settings(args, model)
    cases = [c for c in ALL_CASES if not args.cases or c.name in args.cases]
    start = time.monotonic()
    client = ComfyUIClient(args.comfyui_url, model=model)
    rows = run_bench(cases, settings, client, load_gpl(args.palette), args.denoise)
    title = (
        f"{model.name}  seed {settings.seed}  steps {settings.steps}  "
        f"cfg {settings.cfg}  {settings.sampler}/{settings.scheduler}  "
        f"unet {settings.inpaint_model or '-'}  "
        f"lora {settings.lora or '-'} {settings.lora_strength}  hint {settings.background_hint}  "
        f"grow {settings.mask_grow}  denoise {args.denoise or 'per case'}  "
        f"{time.monotonic() - start:.0f} s"
    )
    render_sheet(rows, title).save(args.output)
    print(f"{args.output}: {title}")
    return 0
