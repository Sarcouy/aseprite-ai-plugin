"""Tests of the rotate task: views asked to klein, their order and the placeholder."""

from pathlib import Path

import pytest
from PIL import Image, ImageOps

from ai_inpaint_bridge.cli import main, parse_args
from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B
from ai_inpaint_bridge.pixelize import pixel_tuples
from ai_inpaint_bridge.reference import GENERATION_SIZE
from ai_inpaint_bridge.tasks import TASKS
from ai_inpaint_bridge.tasks.rotate import (
    BACK_VIEW,
    SIDE_VIEW,
    TURN,
    rotate,
    view_prompt,
)

DRAWN_KEY = (10, 250, 20)


def character() -> Image.Image:
    """A 16 x 24 asymmetric sprite of two colors on a transparent background."""
    sprite = Image.new("RGBA", (16, 24), (0, 0, 0, 0))
    sprite.paste((40, 40, 200, 255), (4, 2, 12, 22))
    sprite.paste((230, 200, 60, 255), (5, 3, 8, 8))
    return sprite


def data(image: Image.Image) -> list[tuple[int, ...]]:
    """The RGBA pixels of an image."""
    return pixel_tuples(image.convert("RGBA"))


def frames_of(strip: Image.Image, width: int) -> list[Image.Image]:
    """The frames of a strip, left to right."""
    return [
        strip.crop((index * width, 0, (index + 1) * width, strip.height))
        for index in range(strip.width // width)
    ]


class Turn:  # pylint: disable=too-few-public-methods  # stands for the one-method reference generator
    """Redraws the reference off its place, on another green, empty when asked."""

    def __init__(self, empty_back: bool = False) -> None:
        self.calls: list[InpaintSettings] = []
        self.empty_back = empty_back

    def generate_reference(
        self, reference: Image.Image, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """The reference moved by 6 pixels, or the key alone for an empty back."""
        assert reference.size == (width, height) == (GENERATION_SIZE, GENERATION_SIZE)
        self.calls.append(settings)
        image = Image.new("RGB", reference.size, DRAWN_KEY)
        if not (self.empty_back and len(self.calls) == 2):
            key = reference.getpixel((0, 0))
            mask = Image.new("L", reference.size, 0)
            mask.putdata(
                [0 if pixel == key else 255 for pixel in reference.get_flattened_data()]
            )
            image.paste(reference, (6, -6), mask)
        return [image] * settings.batch_size


def test_task_draws_from_a_reference_with_distilled_klein() -> None:
    """The views are drawn by klein distilled from the character."""
    assert TASKS["rotate"].models == (KLEIN_4B,)
    assert KLEIN_4B.build_reference is not None


def test_view_prompt_appends_the_details() -> None:
    """The user's description adds to the turn instruction."""
    instruction = TURN.format(view=SIDE_VIEW)

    assert view_prompt(SIDE_VIEW, "  ") == instruction
    assert view_prompt(SIDE_VIEW, " a sword ") == f"{instruction} a sword"


def test_directions_are_down_left_right_up() -> None:
    """The character, the mirrored side view, the side view, then the back view."""
    sprite = character()
    generator = Turn()
    settings = InpaintSettings(prompt="a knight", seed=3, batch_size=2)

    strips = rotate(sprite, settings, generator)

    assert len(strips) == 2
    frames = frames_of(strips[0], sprite.width)
    assert len(frames) == 4 and strips[0].height == sprite.height
    assert data(frames[0]) == data(sprite)
    assert data(frames[2]) == data(sprite)
    assert data(frames[1]) == data(ImageOps.mirror(sprite))
    assert frames[3].getchannel("A").getbbox() == (4, 2, 12, 22)
    assert [call.prompt for call in generator.calls] == [
        view_prompt(SIDE_VIEW, "a knight"),
        view_prompt(BACK_VIEW, "a knight"),
    ]
    assert generator.calls[0].background == "bright green"
    assert generator.calls[0].background_hint


def test_view_drawn_empty_gives_an_empty_frame() -> None:
    """A back view without any character stays transparent."""
    sprite = character()
    settings = InpaintSettings(prompt="", seed=1)

    [strip] = rotate(sprite, settings, Turn(empty_back=True))

    assert frames_of(strip, sprite.width)[3].getchannel("A").getbbox() is None


def test_invert_backend_writes_a_strip_per_variant(tmp_path: Path) -> None:
    """Offline runs give the character in every direction, mirrored to the left."""
    character().save(tmp_path / "ref.png")
    output = tmp_path / "out.png"
    argv = ["--task", "rotate", "--reference", str(tmp_path / "ref.png")]

    assert parse_args([*argv, "--output", str(output)]).context.model is KLEIN_4B
    assert main([*argv, "--output", str(output), "--variants", "2"]) == 0

    strips = [Image.open(tmp_path / f"out-{index}.png") for index in (1, 2)]
    for strip in strips:
        frames = frames_of(strip.convert("RGBA"), 16)
        assert len(frames) == 4 and strip.height == 24
        assert data(frames[1]) == data(ImageOps.mirror(character()))
        assert data(frames[3]) == data(character())
        strip.close()


def test_empty_character_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transparent reference has nothing to turn."""
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(tmp_path / "ref.png")
    argv = ["--task", "rotate", "--output", str(tmp_path / "out.png")]

    assert main([*argv, "--reference", str(tmp_path / "ref.png")]) == 1
    assert "empty" in capsys.readouterr().err
