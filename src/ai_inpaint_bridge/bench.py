"""Quality bench: fixed scenes, prompts and seeds rendered into one comparison sheet."""

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw

from ai_inpaint_bridge.backends import (
    InpaintGenerator,
    changed_fraction,
    diffusion_inpaint,
)
from ai_inpaint_bridge.comfyui import DEFAULT_URL, ComfyUIClient, InpaintSettings
from ai_inpaint_bridge.pixelize import Color

DB32_PATH = Path(
    "/usr/local/share/aseprite/data/extensions/dawnbringer-palettes/db32.gpl"
)
CONTEXT_MARGIN = 8
CELL = 160
GAP = 6
LABEL_WIDTH = 190
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
class BenchRow:
    """The region sent for one case and the variants it produced."""

    case: BenchCase
    original: Image.Image
    mask: Image.Image
    variants: list[Image.Image]


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


def run_bench(
    cases: Sequence[BenchCase],
    settings: InpaintSettings,
    generator: InpaintGenerator,
    palette: Sequence[Color],
) -> list[BenchRow]:
    """Inpaint every case with the same settings (and seed)."""
    rows: list[BenchRow] = []
    for case in cases:
        source, mask = render_case(case, palette)
        case_settings = replace(settings, prompt=case.prompt, denoise=case.denoise)
        variants = diffusion_inpaint(source, mask, case_settings, generator, palette)
        rows.append(BenchRow(case, source, mask, variants))
    return rows


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
    ImageDraw.Draw(sheet).text(
        (GAP, top), f"{row.case.name}\n{row.case.prompt}", fill=LABEL_COLOR
    )
    zoom = max(1, CELL // max(row.original.size))
    for column, image in enumerate([row.original, *row.variants]):
        tile = _checkered(image, zoom)
        left = LABEL_WIDTH + GAP + column * (CELL + GAP)
        sheet.paste(
            tile.convert("RGB"),
            (left + (CELL - tile.width) // 2, top + (CELL - tile.height) // 2),
        )
        if column > 0:
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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = InpaintSettings(prompt="", seed=0)
    parser = argparse.ArgumentParser(
        prog="ai-inpaint-bench", description="Render the quality comparison sheet."
    )
    parser.add_argument("--output", type=Path, required=True, help="sheet PNG")
    parser.add_argument("--palette", type=Path, default=DB32_PATH, help="GIMP palette")
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--cfg", type=float, default=defaults.cfg)
    parser.add_argument("--sampler", default=defaults.sampler)
    parser.add_argument("--scheduler", default=defaults.scheduler)
    parser.add_argument("--mask-grow", type=int, default=defaults.mask_grow)
    parser.add_argument("--lora-strength", type=float, default=defaults.lora_strength)
    parser.add_argument(
        "--background-hint",
        action="store_true",
        help="name the key color in the prompt",
    )
    parser.add_argument(
        "--inpaint-model",
        default=defaults.inpaint_model,
        help='diffusion_models UNet, "" to use the checkpoint one',
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=[case.name for case in BENCH_CASES],
        help="case names, all when omitted",
    )
    parser.add_argument("--comfyui-url", default=DEFAULT_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every case on ComfyUI and save the sheet."""
    args = _parse_args(argv)
    settings = InpaintSettings(
        prompt="",
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        batch_size=args.variants,
        inpaint_model=args.inpaint_model,
        lora_strength=args.lora_strength,
        background_hint=args.background_hint,
        mask_grow=args.mask_grow,
    )
    cases = [c for c in BENCH_CASES if not args.cases or c.name in args.cases]
    start = time.monotonic()
    rows = run_bench(
        cases, settings, ComfyUIClient(args.comfyui_url), load_gpl(args.palette)
    )
    title = (
        f"seed {settings.seed}  steps {settings.steps}  cfg {settings.cfg}  "
        f"{settings.sampler}/{settings.scheduler}  "
        f"unet {settings.inpaint_model or 'checkpoint'}  "
        f"lora {settings.lora_strength}  hint {settings.background_hint}  "
        f"grow {settings.mask_grow}  "
        f"{time.monotonic() - start:.0f} s"
    )
    render_sheet(rows, title).save(args.output)
    print(f"{args.output}: {title}")
    return 0
