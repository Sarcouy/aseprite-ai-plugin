"""Tests of the quality bench with a fake generator."""

from dataclasses import replace
from pathlib import Path

from PIL import Image

from ai_inpaint_bridge.bench import (
    BENCH_CASES,
    CELL,
    DB32_PATH,
    load_gpl,
    render_case,
    render_sheet,
    run_bench,
)
from ai_inpaint_bridge.comfyui import InpaintSettings

PALETTE = [(i * 8, 255 - i * 8, i * 4) for i in range(32)]


class FlatGenerator:  # pylint: disable=too-few-public-methods  # test double of the single-method InpaintGenerator protocol
    """Returns batch_size images filled with the first palette color."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Record the prompt, return flat images of the input size."""
        del mask
        self.prompts.append(settings.prompt)
        return [Image.new("RGB", image.size, PALETTE[0])] * settings.batch_size


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


def test_bench_runs_every_case_and_renders_one_line_each() -> None:
    """Each case gets its prompt and the sheet grows with the cases."""
    generator = FlatGenerator()
    settings = replace(InpaintSettings("", 1), batch_size=2)

    rows = run_bench(BENCH_CASES, settings, generator, PALETTE)
    sheet = render_sheet(rows, "title")

    assert generator.prompts == [case.prompt for case in BENCH_CASES]
    assert all(len(row.variants) == 2 for row in rows)
    assert sheet.height > len(BENCH_CASES) * CELL
