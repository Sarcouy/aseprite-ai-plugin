"""Tests of the animation task: guides, pose prompts, frame order and placeholder."""

from pathlib import Path

import pytest
from PIL import Image

from ai_inpaint_bridge.cli import main, parse_args
from ai_inpaint_bridge.comfyui import InpaintSettings
from ai_inpaint_bridge.klein import KLEIN_4B, build_klein_guided_workflow
from ai_inpaint_bridge.pixelize import pixel_tuples
from ai_inpaint_bridge.reference import GENERATION_SIZE
from ai_inpaint_bridge.tasks import TASKS
from ai_inpaint_bridge.tasks.animation import (
    ANIMATIONS,
    GUIDE_STRENGTH,
    animate,
    guide,
    pose_prompt,
)

DRAWN_KEY = (10, 250, 20)


def character() -> Image.Image:
    """A 16 x 24 sprite of two colors on a transparent background."""
    sprite = Image.new("RGBA", (16, 24), (0, 0, 0, 0))
    sprite.paste((40, 40, 200, 255), (4, 2, 12, 22))
    sprite.paste((230, 200, 60, 255), (5, 3, 11, 8))
    return sprite


def test_task_redraws_guides_with_distilled_klein() -> None:
    """Each pose is a guide redrawn by klein, through its guided graph."""
    task = TASKS["animation"]

    assert task.models == (KLEIN_4B,)
    assert KLEIN_4B.build_guided is build_klein_guided_workflow
    assert ANIMATIONS["idle"].loop == (0, 1, 0, 2)
    assert [pose.lift for pose in ANIMATIONS["idle"].poses] == [-1, 1]


def test_guide_moves_the_upper_body_only() -> None:
    """Lowered or raised by a row, the body above the waist moves, the legs stay."""
    sprite = character()

    lowered, raised = guide(sprite, -1), guide(sprite, 1)

    assert lowered.getchannel("A").getbbox() == (4, 3, 12, 22)
    assert raised.getchannel("A").getbbox() == (4, 1, 12, 22)
    assert lowered.getpixel((6, 8)) == (230, 200, 60, 255)
    assert raised.getpixel((6, 2)) == (230, 200, 60, 255)
    legs = (0, 16, 16, 24)
    for posed in (lowered, raised):
        assert list(posed.crop(legs).get_flattened_data()) == list(
            sprite.crop(legs).get_flattened_data()
        )
    assert guide(sprite, 0).tobytes() == sprite.tobytes()


def test_pose_prompt_appends_the_details() -> None:
    """The user's description adds to the instruction of each pose."""
    pose = ANIMATIONS["idle"].poses[0]

    assert pose_prompt(pose, "  ") == pose.instruction
    assert pose_prompt(pose, " keep the sword ") == f"{pose.instruction} keep the sword"


class Redraw:  # pylint: disable=too-few-public-methods  # stands for the one-method guided generator
    """Redraws the guide off the grid, inverted for the second pose."""

    def __init__(self) -> None:
        self.calls: list[InpaintSettings] = []
        self.references: list[Image.Image] = []

    def generate_guided(
        self, reference: Image.Image, posed: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """The guide on another green, moved by 6 pixels."""
        assert reference.size == posed.size == (GENERATION_SIZE, GENERATION_SIZE)
        self.calls.append(settings)
        self.references.append(reference)
        key = posed.getpixel((0, 0))
        image = Image.new("RGB", posed.size, DRAWN_KEY)
        drawing = posed.copy()
        if len(self.calls) == 2:
            drawing = drawing.point(lambda value: 255 - value)
        mask = Image.new("L", posed.size, 0)
        mask.putdata(
            [0 if pixel == key else 255 for pixel in posed.get_flattened_data()]
        )
        image.paste(drawing, (6, -6), mask)
        return [image] * settings.batch_size


def frames_of(strip: Image.Image, width: int) -> list[Image.Image]:
    """The frames of a strip, left to right."""
    return [
        strip.crop((index * width, 0, (index + 1) * width, strip.height))
        for index in range(strip.width // width)
    ]


def test_idle_keeps_the_character_between_its_two_poses() -> None:
    """Frames are the character, pose 1, the character, pose 2, at its size."""
    sprite = character()
    generator = Redraw()
    settings = InpaintSettings(prompt="a knight", seed=3, batch_size=2)

    strips = animate(sprite, ANIMATIONS["idle"], settings, generator)

    assert len(strips) == 2
    frames = frames_of(strips[0], sprite.width)
    assert len(frames) == 4 and strips[0].height == sprite.height
    original = list(sprite.get_flattened_data())
    assert list(frames[0].get_flattened_data()) == original
    assert list(frames[1].get_flattened_data()) == list(
        guide(sprite, -1).get_flattened_data()
    )
    assert list(frames[2].get_flattened_data()) == original
    assert frames[3].getchannel("A").getbbox() == (4, 1, 12, 22)
    assert generator.calls[0].prompt == pose_prompt(
        ANIMATIONS["idle"].poses[0], "a knight"
    )
    assert generator.calls[0].denoise == GUIDE_STRENGTH
    assert generator.calls[0].background == "bright green"
    assert generator.calls[0].background_hint
    assert generator.references[0].getpixel((0, 0)) == (0, 255, 0)


def test_invert_backend_writes_a_strip_per_variant(tmp_path: Path) -> None:
    """Offline runs give the character and the guides of its poses, in loop order."""
    character().save(tmp_path / "ref.png")
    argv = ["--task", "animation", "--output", str(tmp_path / "out.png")]
    argv += ["--reference", str(tmp_path / "ref.png"), "--variants", "2"]

    assert parse_args(argv).context.model is KLEIN_4B
    assert main(argv) == 0

    for index in (1, 2):
        with Image.open(tmp_path / f"out-{index}.png") as output:
            assert output.size == (16 * 4, 24)
            frames = frames_of(output.convert("RGBA"), 16)
            boxes = [frame.getchannel("A").getbbox() for frame in frames]
            assert boxes == [
                (4, 2, 12, 22),
                (4, 3, 12, 22),
                (4, 2, 12, 22),
                (4, 1, 12, 22),
            ]


def test_empty_character_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transparent reference has nothing to animate."""
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(tmp_path / "ref.png")
    argv = ["--task", "animation", "--output", str(tmp_path / "out.png")]

    assert main([*argv, "--reference", str(tmp_path / "ref.png")]) == 1
    assert "empty" in capsys.readouterr().err


def test_poses_keep_the_colors_of_the_character(tmp_path: Path) -> None:
    """A sprite palette without the character colors does not recolor the poses."""
    character().save(tmp_path / "ref.png")
    palette = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
    palette.putpixel((1, 0), (0, 128, 0, 255))
    palette.save(tmp_path / "palette.png")
    argv = ["--task", "animation", "--output", str(tmp_path / "out.png")]
    argv += ["--reference", str(tmp_path / "ref.png")]

    assert main([*argv, "--palette", str(tmp_path / "palette.png")]) == 0

    with Image.open(tmp_path / "out-1.png") as output:
        colors = {pixel for pixel in pixel_tuples(output) if pixel[3]}
    assert colors == {(40, 40, 200, 255), (230, 200, 60, 255)}
