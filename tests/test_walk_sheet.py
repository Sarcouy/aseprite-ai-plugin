"""Tests of the walk-sheet task: prompt, grid detection, slicing and placeholder."""

from pathlib import Path

import pytest
from PIL import Image

from ai_inpaint_bridge.cli import main, parse_args
from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B_BASE_WALK, SVNTAX_LORA
from ai_inpaint_bridge.pixelize import Color, foreground_mask, pixel_tuples
from ai_inpaint_bridge.tasks import TASKS
from ai_inpaint_bridge.tasks.walk_sheet import (
    FRAME,
    GRID,
    foreground_runs,
    edit_prompt,
    generate_walk_sheets,
    generate_walk_sheets_from,
    group_bands,
    locate_cells,
    reference_image,
    slice_sheet,
    walk_prompt,
)

PALETTE: list[Color] = [(20, 40, 220), (240, 200, 40), (30, 30, 30)]
KEY = (235, 60, 200)
INK = (22, 41, 210)
CAPE = (238, 198, 44)

Box = tuple[int, int, int, int]


def test_task_draws_with_klein_base_and_the_walk_lora() -> None:
    """The sheet needs klein base and the svntax LoRA it was trained on."""
    task = TASKS["walk-sheet"]

    assert task.models == (KLEIN_4B_BASE_WALK,)
    assert task.model().settings.lora == SVNTAX_LORA
    assert task.model().settings.steps == 20


def test_prompt_follows_the_card_layout() -> None:
    """Description first, then the 3 walking frames and the pose of each row."""
    prompt = walk_prompt(" a witch with a purple hat. ")

    assert prompt.startswith("A pixel art spritesheet of a witch with a purple hat. ")
    assert "first row is 3 walking frames facing down" in prompt
    assert prompt.endswith("1 frame lying on floor.")


def test_runs_skip_noise_and_bands_merge_across_the_narrowest_gaps() -> None:
    """Short runs are noise; a sprite split in two runs is merged back."""
    counts = [0, 5, 0, 3, 3, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 4, 4, 4, 0, 0, 0]
    counts += [1, 1, 1, 1, 1, 0]

    runs = foreground_runs(counts)

    assert runs == [(3, 7), (9, 13), (14, 18), (21, 26)]
    assert group_bands(runs, 3) == [(3, 7), (9, 18), (21, 26)]
    assert group_bands(runs, 5) is None


def sheet_image(boxes: list[tuple[Box, Color]]) -> Image.Image:
    """A 512 sheet on a pink close to magenta, as klein draws the key color."""
    image = Image.new("RGB", (512, 512), KEY)
    for box, color in boxes:
        image.paste(color, box)
    return image


def body(row: int, column: int, shift: int = 0) -> Box:
    """A 40 x 80 body in its 128 cell, feet at 112, on the 4 pixel grid."""
    left = column * 128 + 44 + shift
    return (left, row * 128 + 32, left + 40, row * 128 + 112)


def test_cells_follow_the_gaps_between_shifted_sprites() -> None:
    """A grid drawn off the quarters is still split between its sprites."""
    boxes = [(body(r, c, 24), INK) for r in range(GRID) for c in range(GRID)]

    cells = locate_cells(foreground_mask(sheet_image(boxes), KEY))

    assert [cell[0] for cell in cells[0]] == [0, 152, 280, 408]
    assert [cell[2] for cell in cells[0]] == [152, 280, 408, 512]


def frame_box(sheet: Image.Image, row: int, column: int) -> Box | None:
    """Bounding box of the opaque pixels of one frame."""
    frame = sheet.crop(
        (column * FRAME, row * FRAME, (column + 1) * FRAME, (row + 1) * FRAME)
    )
    return frame.getchannel("A").getbbox()


def test_frames_keep_the_body_still_and_on_the_baseline() -> None:
    """A cape widening one frame moves neither its body nor the other frames."""
    boxes = [(body(r, c), INK) for r in range(GRID) for c in range(GRID)]
    cape_left = body(0, 1)[0] - 16
    boxes.append(((cape_left, 64, cape_left + 16, 112), CAPE))

    sheet = slice_sheet(sheet_image(boxes), PALETTE)

    assert sheet.size == (GRID * FRAME, GRID * FRAME)
    for row in range(GRID):
        for column in range(GRID):
            expected = (11, 11, 21, 31) if (row, column) != (0, 1) else (7, 11, 21, 31)
            assert frame_box(sheet, row, column) == expected
    assert sheet.getpixel((32 + 12, 20)) == (*PALETTE[0], 255)
    assert sheet.getpixel((32 + 8, 20)) == (*PALETTE[1], 255)
    assert sheet.getpixel((0, 0)) == (0, 0, 0, 0)


def test_large_sprites_are_reduced_more_to_fit() -> None:
    """A sprite taller than a frame at the default factor raises the factor."""
    boxes = [(body(r, c), INK) for r in range(GRID) for c in range(GRID)]
    boxes.append(((44, 0, 84, 124), INK))

    sheet = slice_sheet(sheet_image(boxes), PALETTE)

    top_left = frame_box(sheet, 0, 0)
    assert top_left is not None
    assert top_left[1] >= 1 and top_left[3] == 31


def test_a_sheet_without_grid_is_cut_in_quarters() -> None:
    """Without four rows of sprites the quarters are used."""
    sheet = slice_sheet(sheet_image([((100, 100, 400, 400), INK)]), PALETTE)

    assert sheet.size == (GRID * FRAME, GRID * FRAME)
    assert frame_box(sheet, 0, 0) is not None


class SheetOnKey:  # pylint: disable=too-few-public-methods  # stands for the one-method text generator
    """Draws a regular sheet of bodies."""

    def __init__(self) -> None:
        self.calls: list[InpaintSettings] = []

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """One sheet per variant, at 512."""
        assert (width, height) == (512, 512)
        self.calls.append(settings)
        boxes = [(body(r, c), INK) for r in range(GRID) for c in range(GRID)]
        return [sheet_image(boxes)] * settings.batch_size


def test_sheets_are_asked_on_the_key_color() -> None:
    """The card prompt is sent with the key color, keyed on the border."""
    generator = SheetOnKey()
    settings = InpaintSettings(prompt="a knight", seed=1, batch_size=2)

    sheets = generate_walk_sheets(settings, generator, PALETTE)

    assert generator.calls[0].prompt == walk_prompt("a knight")
    assert generator.calls[0].background_hint
    assert generator.calls[0].background == "magenta"
    assert len(sheets) == 2
    colors = {pixel for pixel in pixel_tuples(sheets[0]) if pixel[3] > 0}
    assert colors == {(*PALETTE[0], 255)}


def test_invert_backend_writes_a_moving_placeholder_sheet(tmp_path: Path) -> None:
    """Offline runs give one transparent 128 sheet per variant."""
    argv = ["--task", "walk-sheet", "--output", str(tmp_path / "out.png")]
    argv += ["--prompt", "a knight", "--variants", "2"]

    assert parse_args(argv).context.model is KLEIN_4B_BASE_WALK
    assert main(argv) == 0

    for index in (1, 2):
        with Image.open(tmp_path / f"out-{index}.png") as output:
            assert output.size == (GRID * FRAME, GRID * FRAME)
            assert output.getpixel((0, 0)) == (0, 0, 0, 0)
            first = frame_box(output.convert("RGBA"), 0, 0)
            second = frame_box(output.convert("RGBA"), 0, 1)
            assert first is not None and second is not None
            assert second[0] == first[0] + 1


def test_empty_description_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a description there is nothing to animate."""
    argv = ["--task", "walk-sheet", "--output", str(tmp_path / "out.png")]

    assert main([*argv, "--prompt", " "]) == 1
    assert "describe the character" in capsys.readouterr().err


def test_edit_prompt_describes_the_image_and_the_layout() -> None:
    """The description is optional with a reference image."""
    assert edit_prompt("").startswith(
        "Create a pixel art spritesheet of the character in the image. The"
    )
    assert edit_prompt("with a red plume.").startswith(
        "Create a pixel art spritesheet of the character in the image, with a red"
        " plume. The spritesheet is a 4 by 4 grid"
    )


def test_reference_is_trimmed_enlarged_and_centered_on_the_key() -> None:
    """The sprite pixels are enlarged to fill the generation size."""
    sprite = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    sprite.paste((*INK, 255), (10, 4, 20, 28))

    reference = reference_image(sprite, KEY)

    assert reference.size == (512, 512)
    assert reference.getpixel((0, 0)) == KEY
    assert foreground_mask(reference, KEY).getbbox() == (151, 4, 361, 508)


class ReferenceSheet(
    SheetOnKey
):  # pylint: disable=too-few-public-methods  # stands for the one-method reference generator
    """Records the reference image and draws the regular sheet."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[Image.Image] = []

    def generate_reference(
        self,
        reference: Image.Image,
        settings: InpaintSettings,
        width: int,
        height: int,
    ) -> list[Image.Image]:
        """One sheet per variant, at 512."""
        self.references.append(reference)
        return self.generate_text(settings, width, height)


def test_sheets_from_a_sprite_send_it_as_reference() -> None:
    """The key color avoids the sprite colors and is named in the prompt."""
    sprite = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    sprite.paste((255, 0, 200, 255), (4, 2, 12, 15))
    generator = ReferenceSheet()
    settings = InpaintSettings(prompt="", seed=1, batch_size=1)

    sheets = generate_walk_sheets_from(sprite, settings, generator, PALETTE)

    assert generator.calls[0].prompt == edit_prompt("")
    assert generator.calls[0].background == "bright green"
    assert generator.references[0].getpixel((0, 0)) == (0, 255, 0)
    assert len(sheets) == 1


def test_invert_backend_accepts_a_reference_without_description(
    tmp_path: Path,
) -> None:
    """Animating an existing sprite needs no description."""
    Image.new("RGBA", (8, 8), (*INK, 255)).save(tmp_path / "ref.png")
    argv = ["--task", "walk-sheet", "--output", str(tmp_path / "out.png")]
    argv += ["--reference", str(tmp_path / "ref.png")]

    assert main(argv) == 0
    assert (tmp_path / "out-1.png").is_file()
