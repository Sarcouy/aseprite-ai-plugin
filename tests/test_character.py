"""Tests of the character task: arguments, prompt, placeholder and generation."""

import argparse
from pathlib import Path

import pytest
from PIL import Image

from ai_inpaint_bridge.cli import main, parse_args
from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B
from ai_inpaint_bridge.pixelize import (
    Color,
    fit_to_sprite,
    foreground_box,
    pixel_tuples,
)
from ai_inpaint_bridge.tasks import TASKS
from ai_inpaint_bridge.tasks.character import (
    best_cell,
    character_prompt,
    generate_characters,
    grid_cells,
    parse_size,
)

PALETTE: list[Color] = [(20, 40, 220), (240, 200, 40), (30, 30, 30)]


def test_task_is_registered_with_klein() -> None:
    """The character task draws with the distilled klein only."""
    task = TASKS["character"]

    assert task.models == (KLEIN_4B,)


def test_size_accepts_square_and_rectangle() -> None:
    """N is a square, WxH a rectangle, both bounded."""
    assert parse_size("16") == (16, 16)
    assert parse_size("32x48") == (32, 48)
    for wrong in ("0", "65", "16x", "big"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size(wrong)


def test_parser_needs_no_input_image(tmp_path: Path) -> None:
    """Only the output, the prompt and the size are given."""
    request = parse_args(
        [
            "--task",
            "character",
            "--output",
            str(tmp_path / "out.png"),
            "--prompt",
            "a cat",
            "--size",
            "16",
        ]
    )

    assert request.task.name == "character"
    assert request.args.size == (16, 16)
    assert request.context.model is KLEIN_4B


def test_prompt_asks_a_grid_of_any_description() -> None:
    """The view and the grid are asked around the user's description."""
    prompt = character_prompt(" an orange cat ")

    assert prompt.startswith("a 2 by 2 grid of four separate top-down RPG")
    assert "sprites of an orange cat, three-quarter view" in prompt


def write_strip(path: Path) -> None:
    """Save PALETTE as the strip exported by the extension."""
    strip = Image.new("RGBA", (len(PALETTE), 1))
    for index, color in enumerate(PALETTE):
        strip.putpixel((index, 0), (*color, 255))
    strip.save(path)


def test_invert_backend_writes_a_transparent_placeholder(tmp_path: Path) -> None:
    """Offline runs give a palette box centered on transparency, one per variant."""
    write_strip(tmp_path / "palette.png")
    argv = ["--task", "character", "--output", str(tmp_path / "out.png")]
    argv += ["--prompt", "a knight", "--size", "8x12", "--variants", "2"]
    argv += ["--palette", str(tmp_path / "palette.png")]

    assert main(argv) == 0

    for index in (1, 2):
        with Image.open(tmp_path / f"out-{index}.png") as output:
            assert output.size == (8, 12)
            assert output.getpixel((0, 0)) == (0, 0, 0, 0)
            assert output.getpixel((4, 6)) == (*PALETTE[0], 255)


def test_empty_description_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a description there is nothing to draw."""
    argv = ["--task", "character", "--output", str(tmp_path / "out.png")]

    assert main([*argv, "--prompt", "  "]) == 1
    assert "describe the character" in capsys.readouterr().err


KEY = (235, 60, 200)
INK = (22, 41, 210)


def grid_image(boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    """A 512 image on a pink close to magenta, as klein draws the key color."""
    image = Image.new("RGB", (512, 512), KEY)
    for box in boxes:
        image.paste(INK, box)
    return image


class GridOnKey:  # pylint: disable=too-few-public-methods  # stands for the one-method text generator
    """Draws a grid of four sprites, the bottom right one the largest."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """One grid per variant."""
        self.calls.append((settings.prompt, width, height))
        boxes = [(40, 40, 80, 120), (300, 40, 340, 120), (40, 300, 80, 380)]
        return [grid_image([*boxes, (300, 300, 380, 460)])] * settings.batch_size


def test_fit_fills_the_sprite_with_the_subject() -> None:
    """A small subject is cropped so that it spans the sprite minus its margin."""
    image = grid_image([(200, 100, 240, 260)])

    sprite = fit_to_sprite(image, (16, 16), PALETTE)

    assert sprite.getchannel("A").getbbox() == (6, 1, 10, 15)
    assert sprite.getpixel((7, 8)) == (*PALETTE[0], 255)


def test_best_cell_is_the_largest_whole_sprite() -> None:
    """Cells whose sprite touches the edge are cut; without a grid the image is kept."""
    image = grid_image([(40, 40, 80, 120), (300, 300, 380, 460)])

    assert best_cell(image).getbbox() == grid_cells(image)[3].getbbox()
    assert foreground_box(best_cell(image), KEY) == (44, 44, 124, 204)
    single = grid_image([(150, 150, 400, 400)])
    assert best_cell(single) is single


def test_characters_are_transparent_and_in_the_palette() -> None:
    """One 512 grid per variant; transparent background, colors of the palette."""
    generator = GridOnKey()
    settings = InpaintSettings(prompt="a knight", seed=1, batch_size=2)

    sprites = generate_characters((16, 16), settings, generator, PALETTE)

    assert generator.calls == [(character_prompt("a knight"), 512, 512)]
    assert len(sprites) == 2
    assert sprites[0].getchannel("A").getbbox() == (5, 1, 11, 15)
    pixels = pixel_tuples(sprites[0])
    assert pixels[0] == (0, 0, 0, 0)
    assert {pixel[:3] for pixel in pixels if pixel[3] > 0} == {PALETTE[0]}
