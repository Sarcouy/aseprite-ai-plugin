"""Tests of the autotile task: corner sheet, guide, redraw graph and placeholder."""

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from ai_inpaint_bridge.cli import main, parse_args
from ai_inpaint_bridge.comfyui import (
    OUTPUT_NODE,
    ComfyUIClient,
    ComfyUIError,
    InpaintSettings,
)
from ai_inpaint_bridge.klein import KLEIN_4B, KLEIN_4B_TILE, klein_sigmas
from ai_inpaint_bridge.pixelize import pixel_tuples
from ai_inpaint_bridge.tasks import TASKS
from ai_inpaint_bridge.tasks.autotile import (
    GRID,
    REDRAW_STRENGTH,
    SHEET_SIZE,
    generate_sheets,
    second_share,
    sheet_mask,
    sheet_prompt,
    tile_corners,
)
from ai_inpaint_bridge.tasks.tile import GENERATION_SIZE, tile_prompt

SETTINGS = replace(KLEIN_4B_TILE.settings, prompt="grass", seed=1, batch_size=2)
TILES = [(column, row) for row in range(GRID) for column in range(GRID)]
SAMPLES = [index / 10 for index in range(11)]


def test_sheet_holds_every_corner_combination_once() -> None:
    """The 16 tiles are the 16 wang tiles."""
    assert len({tile_corners(column, row) for column, row in TILES}) == 16


def test_neighbors_in_the_sheet_share_their_corners() -> None:
    """The sheet repeats: each tile meets tiles with the same corners on its edges."""
    for column, row in TILES:
        corners = tile_corners(column, row)
        right = tile_corners((column + 1) % GRID, row)
        below = tile_corners(column, (row + 1) % GRID)
        assert (corners[1], corners[3]) == (right[0], right[2])
        assert (corners[2], corners[3]) == (below[0], below[1])


def test_border_meets_on_every_edge_of_matching_tiles() -> None:
    """Along an edge the border depends only on the two corners of that edge."""
    for a in TILES:
        for b in TILES:
            left, right = tile_corners(*a), tile_corners(*b)
            if (left[1], left[3]) == (right[0], right[2]):
                for v in SAMPLES:
                    assert second_share(left, 1.0, v) == pytest.approx(
                        second_share(right, 0.0, v)
                    )
            if (left[2], left[3]) == (right[0], right[1]):
                for u in SAMPLES:
                    assert second_share(left, u, 1.0) == pytest.approx(
                        second_share(right, u, 0.0)
                    )


def test_plain_tiles_hold_a_single_material() -> None:
    """A tile with the same material at its 4 corners shows only that material."""
    mask = sheet_mask(GRID * 8)
    for column, row in TILES:
        corners = tile_corners(column, row)
        if len(set(corners)) == 1:
            box = (column * 8, row * 8, column * 8 + 8, row * 8 + 8)
            assert mask.crop(box).getextrema() == (255 * corners[0],) * 2


def test_redraw_graph_starts_from_the_noised_image() -> None:
    """The uploaded image is encoded, then sampled rolled from the strength level."""
    asked = replace(SETTINGS, denoise=0.5)
    build = KLEIN_4B_TILE.build_redraw
    assert build is not None

    workflow = build("guide.png", asked, SHEET_SIZE, SHEET_SIZE)

    assert workflow["5"]["inputs"]["image"] == "guide.png"
    assert workflow["k0"]["inputs"]["latent_image"] == ["9", 0]
    assert "7" not in workflow
    first = float(workflow["s0"]["inputs"]["sigmas"].split(", ")[0])
    assert first == pytest.approx(klein_sigmas(asked, SHEET_SIZE, SHEET_SIZE)[0])
    assert workflow[OUTPUT_NODE]["inputs"]["images"] == ["54", 0]


def test_models_without_redraw_graph_refuse() -> None:
    """Only the seamless klein profile redraws an image."""
    client = ComfyUIClient("http://127.0.0.1:1", model=KLEIN_4B)
    with pytest.raises(ComfyUIError):
        client.generate_redraw(Image.new("RGB", (256, 256)), SETTINGS)


class Painter:
    """Flat textures, then the guide given back for every variant."""

    def __init__(self) -> None:
        self.texts: list[tuple[InpaintSettings, int, int]] = []
        self.redraws: list[tuple[Image.Image, InpaintSettings]] = []

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """A green texture first, a brown one next."""
        self.texts.append((settings, width, height))
        color = (60, 160, 60) if len(self.texts) == 1 else (150, 100, 50)
        return [Image.new("RGB", (width, height), color)]

    def generate_redraw(
        self, image: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """The guide as it is."""
        self.redraws.append((image, settings))
        return [image] * settings.batch_size


def test_sheets_are_redrawn_from_both_textures() -> None:
    """Each material is drawn as a tile, the cut guide redrawn and reduced."""
    painter = Painter()

    sheets = generate_sheets(16, SETTINGS, "dirt", painter, None)

    assert [sheet.size for sheet in sheets] == [(GRID * 16, GRID * 16)] * 2
    assert [settings.prompt for settings, _, _ in painter.texts] == [
        tile_prompt("grass"),
        tile_prompt("dirt"),
    ]
    assert painter.texts[0][1:] == (GENERATION_SIZE, GENERATION_SIZE)
    guide, asked = painter.redraws[0]
    assert guide.size == (SHEET_SIZE, SHEET_SIZE)
    assert asked.prompt == sheet_prompt("grass", "dirt")
    assert asked.denoise == REDRAW_STRENGTH
    assert guide.getpixel((0, 0)) == (150, 100, 50)


def test_invert_backend_writes_the_corner_sheet(tmp_path: Path) -> None:
    """Offline runs give the sheet mask in the first two palette colors."""
    palette = Image.new("RGBA", (2, 1), (200, 10, 30, 255))
    palette.putpixel((1, 0), (20, 40, 220, 255))
    palette.save(tmp_path / "palette.png")
    argv = ["--task", "autotile", "--output", str(tmp_path / "out.png")]
    argv += ["--size", "16", "--prompt", "grass", "--second", "dirt"]

    assert TASKS["autotile"].models == (KLEIN_4B_TILE,)
    assert parse_args(argv).args.second == "dirt"
    assert main([*argv, "--palette", str(tmp_path / "palette.png")]) == 0

    with Image.open(tmp_path / "out-1.png") as output:
        assert output.size == (GRID * 16, GRID * 16)
        colors = set(pixel_tuples(output.convert("RGBA")))
        assert colors == {(200, 10, 30, 255), (20, 40, 220, 255)}


def test_both_materials_are_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the second material there is no transition to draw."""
    argv = ["--task", "autotile", "--output", str(tmp_path / "out.png")]

    assert main([*argv, "--prompt", "grass"]) == 1
    assert "both materials" in capsys.readouterr().err
