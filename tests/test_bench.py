"""Tests of the quality bench with a fake generator."""

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from ai_inpaint_bridge.bench import (
    ALL_CASES,
    BENCH_CASES,
    CELL,
    DB32_PATH,
    SPRITE_CASES,
    SpriteCase,
    _parse_args,
    load_gpl,
    render_case,
    render_sheet,
    run_bench,
    tiled_preview,
)
from ai_inpaint_bridge.comfyui import InpaintSettings

PALETTE = [(i * 8, 255 - i * 8, i * 4) for i in range(32)]
DRAWN_KEYS = {
    "magenta": (216, 21, 107),
    "green": (0, 255, 0),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
}


class FlatGenerator:
    """Returns batch_size images filled with the first palette color.

    Text to image results get a background drawn as klein draws the key color
    (a pink for "magenta"), around a flat square.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.text_sizes: list[tuple[int, int]] = []

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Record the prompt, return flat images of the input size."""
        del mask
        self.prompts.append(settings.prompt)
        return [Image.new("RGB", image.size, PALETTE[0])] * settings.batch_size

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """Record the prompt and size, return a flat square on the key color."""
        self.prompts.append(settings.prompt)
        self.text_sizes.append((width, height))
        key = DRAWN_KEYS.get(settings.background, PALETTE[1])
        image = Image.new("RGB", (width, height), key)
        image.paste(
            PALETTE[0], (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
        )
        return [image] * settings.batch_size


def test_load_gpl_reads_colors_in_order(tmp_path: Path) -> None:
    """Header and comment lines are skipped, names are ignored."""
    path = tmp_path / "p.gpl"
    path.write_text("GIMP Palette\n#\n  0   0   0\tBlack\n255 128  1\tOrange\n")

    assert load_gpl(path) == [(0, 0, 0), (255, 128, 1)]


def test_db32_has_32_colors() -> None:
    """The bench palette is the one installed with Aseprite."""
    if DB32_PATH.exists():
        assert len(load_gpl(DB32_PATH)) == 32


def test_render_case_crops_the_selection_with_context() -> None:
    """The region is the selection plus the margin, clamped to the canvas."""
    case = BENCH_CASES[0]
    source, mask = render_case(case, PALETTE)

    assert source.size == mask.size
    assert mask.getbbox() is not None
    left, top, right, bottom = case.selection
    assert source.width >= right - left and source.height >= bottom - top


def test_sprite_case_starts_from_the_stored_sprite() -> None:
    """The knight cases paste the bench sprite before cropping the selection."""
    case = next(case for case in BENCH_CASES if case.name == "knight-cape")
    source, mask = render_case(case, PALETTE)

    assert source.size == (32, 32)
    assert source.getchannel("A").getbbox() is not None
    assert mask.getbbox() == (8, 8, 25, 30)


def test_bench_runs_every_case_and_renders_one_line_each() -> None:
    """Each case gets its prompt and the sheet grows with the cases."""
    generator = FlatGenerator()
    settings = replace(InpaintSettings("", 1), batch_size=2)

    rows = run_bench(ALL_CASES, settings, generator, PALETTE)
    sheet = render_sheet(rows, "title")

    assert generator.prompts == [case.prompt for case in ALL_CASES]
    assert all(len(row.variants) == 2 for row in rows)
    assert sheet.height > len(ALL_CASES) * CELL


def test_sprite_cases_are_drawn_at_their_size_on_a_transparent_background() -> None:
    """Sprites are generated large, reduced to their size, the key becomes transparent."""
    generator = FlatGenerator()
    settings = replace(InpaintSettings("", 1), batch_size=1)

    cases = [case for case in SPRITE_CASES if case.transparent]
    rows = run_bench(cases, settings, generator, PALETTE)

    assert generator.text_sizes == [(512, 512)] * len(cases)
    for row, case in zip(rows, cases):
        sprite = row.variants[0]
        assert row.original is None and row.mask is None
        assert sprite.size == case.size
        assert sprite.getpixel((0, 0)) == (0, 0, 0, 0)
        center = (case.size[0] // 2, case.size[1] // 2)
        assert sprite.getpixel(center) == (*PALETTE[0], 255)


def test_opaque_sprite_keeps_its_background() -> None:
    """Without transparency nothing is keyed out."""
    generator = FlatGenerator()
    settings = replace(InpaintSettings("", 1, background="magenta"), batch_size=1)
    case = SpriteCase("tile", "grass", (16, 16), transparent=False)

    sprite = run_bench([case], settings, generator, PALETTE)[0].variants[0]

    alpha = sprite.getchannel("A").getextrema()
    assert alpha == (255, 255)


def test_every_case_runs_by_default() -> None:
    """Without --cases the sheet holds inpaint and sprite cases."""
    args = _parse_args(["--output", "sheet.png"])

    assert args.cases is None
    assert {case.name for case in ALL_CASES} >= {"potion", "knight-16", "knight-32"}


class StrengthRecorder(FlatGenerator):
    """Also records the change strength of inpaint calls."""

    def __init__(self) -> None:
        super().__init__()
        self.strengths: list[float] = []

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Record the strength, then behave as the flat generator."""
        self.strengths.append(settings.denoise)
        return super().generate(image, mask, settings)


def test_denoise_option_overrides_every_inpaint_case() -> None:
    """--denoise forces the strength, otherwise each case keeps its own."""
    settings = replace(InpaintSettings("", 1), batch_size=1)
    own, forced = StrengthRecorder(), StrengthRecorder()

    run_bench(BENCH_CASES, settings, own, PALETTE)
    run_bench(BENCH_CASES, settings, forced, PALETTE, denoise=0.5)

    assert own.strengths == [case.denoise for case in BENCH_CASES]
    assert forced.strengths == [0.5] * len(BENCH_CASES)


@pytest.mark.parametrize("value", ["0", "1.5", "-1"])
def test_denoise_option_is_bounded(value: str) -> None:
    """A zero or above-one strength is refused before any generation."""
    with pytest.raises(SystemExit):
        _parse_args(["--output", "sheet.png", "--denoise", value])


def test_tile_preview_repeats_the_tile_three_times() -> None:
    """The seams of a tile show on its 3x3 repetition."""
    tile = Image.new("RGBA", (16, 16), (*PALETTE[3], 255))
    tile.putpixel((0, 0), (*PALETTE[5], 255))

    preview = tiled_preview(tile)

    assert preview.size == (48, 48)
    assert preview.getpixel((16, 16)) == (*PALETTE[5], 255)
    assert preview.getpixel((47, 47)) == (*PALETTE[3], 255)


def test_sheet_mixes_inpaint_sprite_tile_and_walk_rows() -> None:
    """Every kind of case renders in the same sheet, one row each."""
    settings = replace(InpaintSettings("", 1), batch_size=1)
    rows = run_bench(SPRITE_CASES, settings, FlatGenerator(), PALETTE)

    sheet = render_sheet(rows, "title")

    kinds = {case.name for case in SPRITE_CASES if case.tile}
    assert kinds == {"grass-tile", "wall-tile"}
    assert any(case.size == (128, 128) for case in SPRITE_CASES)
    assert sheet.height >= len(SPRITE_CASES) * CELL
    assert all(
        row.variants[0].size == row.case.size for row in rows if row.original is None
    )
