"""Tests of the tile task: seamless graph, reduction and placeholder."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ai_inpaint_bridge.cli import main, parse_args
from ai_inpaint_bridge.comfyui import OUTPUT_NODE, InpaintSettings
from ai_inpaint_bridge.klein import (
    KLEIN_4B_TILE,
    LATENT_UNIT,
    WRAP_CELLS,
    build_klein_tile_workflow,
)
from ai_inpaint_bridge.pixelize import pixel_tuples
from ai_inpaint_bridge.tasks import TASKS
from ai_inpaint_bridge.tasks.tile import (
    GENERATION_SIZE,
    generate_tiles,
    reduce_tile,
    tile_prompt,
)

Grid = list[list[int]]
SETTINGS = replace(KLEIN_4B_TILE.settings, prompt="grass", seed=1, batch_size=2)


def _paste(target: Grid, piece: Grid, x: int, y: int) -> Grid:
    pasted = [row[:] for row in target]
    for row_index, row in enumerate(piece[: len(target) - y]):
        pasted[y + row_index][x : x + len(row)] = row[: len(target[0]) - x]
    return pasted


def evaluate(
    workflow: dict[str, Any], link: list[Any], sources: dict[str, Grid]
) -> Grid:
    """Value of a latent node, cells as numbers, sources standing for the samplers."""
    if link[0] in sources:
        return sources[link[0]]
    inputs = workflow[link[0]]["inputs"]
    kind = workflow[link[0]]["class_type"]
    if kind == "EmptyFlux2LatentImage":
        return [[0] * (inputs["width"] // 16) for _ in range(inputs["height"] // 16)]
    x, y = inputs["x"] // LATENT_UNIT, inputs["y"] // LATENT_UNIT
    if kind == "LatentCrop":
        grid = evaluate(workflow, inputs["samples"], sources)
        width, height = inputs["width"] // LATENT_UNIT, inputs["height"] // LATENT_UNIT
        return [row[x : x + width] for row in grid[y : y + height]]
    assert kind == "LatentComposite"
    target = evaluate(workflow, inputs["samples_to"], sources)
    return _paste(target, evaluate(workflow, inputs["samples_from"], sources), x, y)


def numbered(cells: int) -> Grid:
    """A latent whose cells are all different."""
    return [[y * cells + x for x in range(cells)] for y in range(cells)]


def test_task_draws_with_the_seamless_graph() -> None:
    """The tile task draws with klein through its seamless text graph."""
    assert TASKS["tile"].models == (KLEIN_4B_TILE,)
    assert KLEIN_4B_TILE.build_text is build_klein_tile_workflow


def test_each_step_starts_from_the_previous_one_rolled_by_half() -> None:
    """One sampler per step, the noise added once, the latent rolled in between."""
    workflow = build_klein_tile_workflow(SETTINGS, 256, 256)
    samplers = [f"k{step}" for step in range(SETTINGS.steps)]
    cells = 256 // 16

    assert [workflow[name]["inputs"]["noise"] for name in samplers] == [
        ["17", 0],
        *[["40", 0]] * (SETTINGS.steps - 1),
    ]
    sigmas = [
        workflow[f"s{step}"]["inputs"]["sigmas"] for step in range(SETTINGS.steps)
    ]
    bounds = [tuple(float(value) for value in pair.split(", ")) for pair in sigmas]
    assert bounds[0][0] == pytest.approx(1.0) and bounds[-1][1] == 0.0
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
    rolled = evaluate(
        workflow, workflow["k1"]["inputs"]["latent_image"], {"k0": numbered(cells)}
    )
    half = cells // 2
    assert rolled == [
        [numbered(cells)[(y - half) % cells][(x - half) % cells] for x in range(cells)]
        for y in range(cells)
    ]


def test_latent_is_decoded_surrounded_by_its_opposite_edges() -> None:
    """The decoded latent repeats the tile around it, the margin is cropped back."""
    workflow = build_klein_tile_workflow(SETTINGS, 256, 256)
    cells = 256 // 16
    last = f"k{SETTINGS.steps - 1}"

    wrapped = evaluate(
        workflow, workflow["10"]["inputs"]["samples"], {last: numbered(cells)}
    )

    side = cells + 2 * WRAP_CELLS
    assert wrapped == [
        [
            numbered(cells)[(y - WRAP_CELLS) % cells][(x - WRAP_CELLS) % cells]
            for x in range(side)
        ]
        for y in range(side)
    ]
    crop = workflow["54"]["inputs"]
    assert (crop["x"], crop["y"], crop["width"]) == (WRAP_CELLS * 16,) * 2 + (256,)
    assert workflow[OUTPUT_NODE]["inputs"]["images"] == ["54", 0]


def test_tile_must_be_a_square_of_256_pixels_multiple() -> None:
    """Rolled halves narrower than the margin cannot be cropped by ComfyUI."""
    with pytest.raises(ValueError, match="256"):
        build_klein_tile_workflow(SETTINGS, 128, 128)
    with pytest.raises(ValueError, match="256"):
        build_klein_tile_workflow(SETTINGS, 256, 512)


def joints() -> Image.Image:
    """White slabs of 64 pixels with 4-pixel black joints, as klein draws them."""
    image = Image.new("RGB", (GENERATION_SIZE, GENERATION_SIZE), (250, 250, 250))
    for x in range(0, GENERATION_SIZE, 64):
        image.paste((0, 0, 0), (x, 0, x + 4, GENERATION_SIZE))
    return image


def test_averaging_keeps_the_thin_joints() -> None:
    """A joint thinner than a block stays, darker, in the reduced tile."""
    palette = [(250, 250, 250), (120, 120, 120), (0, 0, 0)]

    tile = reduce_tile(joints(), 32, palette)

    columns = {x: tile.getpixel((x, 5)) for x in range(32)}
    assert columns[0] != (250, 250, 250, 255)
    assert columns[8] != (250, 250, 250, 255)
    assert columns[4] == (250, 250, 250, 255)
    assert {pixel[:3] for pixel in pixel_tuples(tile)} <= set(palette)


class Texture:  # pylint: disable=too-few-public-methods  # stands for the one-method text generator
    """Draws the slab joints for every variant."""

    def __init__(self) -> None:
        self.calls: list[tuple[InpaintSettings, int, int]] = []

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """batch_size drawings of joints."""
        self.calls.append((settings, width, height))
        return [joints()] * settings.batch_size


def test_tiles_are_drawn_from_the_material_and_reduced() -> None:
    """The material goes in the template, each drawing gives a tile of the size."""
    generator = Texture()

    tiles = generate_tiles(16, SETTINGS, generator, None)

    assert [tile.size for tile in tiles] == [(16, 16)] * 2
    settings, width, height = generator.calls[0]
    assert settings.prompt == tile_prompt("grass")
    assert (width, height) == (GENERATION_SIZE, GENERATION_SIZE)


def test_invert_backend_writes_a_seamless_placeholder(tmp_path: Path) -> None:
    """Offline runs give a checkerboard of the first two palette colors."""
    palette = Image.new("RGBA", (2, 1), (200, 10, 30, 255))
    palette.putpixel((1, 0), (20, 40, 220, 255))
    palette.save(tmp_path / "palette.png")
    argv = ["--task", "tile", "--output", str(tmp_path / "out.png"), "--size", "16"]
    argv += ["--prompt", "grass", "--palette", str(tmp_path / "palette.png")]

    assert parse_args(argv).context.model is KLEIN_4B_TILE
    assert main([*argv, "--variants", "2"]) == 0

    with Image.open(tmp_path / "out-2.png") as output:
        assert output.size == (16, 16)
        colors = set(pixel_tuples(output.convert("RGBA")))
        assert colors == {(200, 10, 30, 255), (20, 40, 220, 255)}


def test_size_is_a_square_of_16_or_32() -> None:
    """The extension sends NxN, a single side is accepted too."""
    argv = ["--task", "tile", "--output", "out.png", "--prompt", "sand"]

    assert parse_args([*argv, "--size", "16x16"]).args.size == 16
    assert parse_args([*argv, "--size", "32"]).args.size == 32
    for wrong in ("24", "16x32"):
        with pytest.raises(SystemExit):
            parse_args([*argv, "--size", wrong])


def test_missing_material_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a material there is nothing to draw."""
    argv = ["--task", "tile", "--output", str(tmp_path / "out.png")]

    assert main(argv) == 1
    assert "material" in capsys.readouterr().err
