"""Tests of the bridge command line with the invert backend."""

from pathlib import Path

from PIL import Image

import pytest

from ai_inpaint_bridge.cli import (
    URL_VARIABLE,
    load_palette,
    main,
    parse_args,
    variant_path,
)
from ai_inpaint_bridge.comfyui import DEFAULT_URL
from ai_inpaint_bridge.tasks import TASKS

RED = (200, 10, 30, 255)
INVERTED_RED = (55, 245, 225, 255)


def test_single_color_palette_is_ignored(tmp_path: Path) -> None:
    """An all-black palette falls back to the adaptive palette."""
    Image.new("RGBA", (256, 1), (0, 0, 0, 255)).save(tmp_path / "black.png")
    strip = Image.new("RGBA", (2, 1), RED)
    strip.putpixel((1, 0), INVERTED_RED)
    strip.save(tmp_path / "two.png")

    assert load_palette(tmp_path / "black.png") is None
    assert load_palette(tmp_path / "two.png") == [RED[:3], INVERTED_RED[:3]]
    assert load_palette(None) is None


def write_inputs(tmp_path: Path, mask_size: tuple[int, int] = (4, 4)) -> list[str]:
    """Write a red 4x4 region and its mask, return the bridge arguments."""
    source = Image.new("RGBA", (4, 4), RED)
    source.putpixel((3, 3), (0, 0, 0, 0))
    mask = Image.new("RGBA", mask_size, (0, 0, 0, 0))
    for xy in ((1, 1), (2, 1), (1, 2), (3, 3)):
        if xy[0] < mask_size[0] and xy[1] < mask_size[1]:
            mask.putpixel(xy, (255, 255, 255, 255))
    source.save(tmp_path / "input.png")
    mask.save(tmp_path / "mask.png")
    return [
        "--input",
        str(tmp_path / "input.png"),
        "--mask",
        str(tmp_path / "mask.png"),
        "--output",
        str(tmp_path / "output.png"),
    ]


def test_variant_path_numbers_from_one() -> None:
    """Variants are written next to the requested output name."""
    assert variant_path(Path("/tmp/run-output.png"), 1) == Path("/tmp/run-output-1.png")
    assert variant_path(Path("/tmp/run-output.png"), 8) == Path("/tmp/run-output-8.png")


def test_inverts_only_masked_pixels(tmp_path: Path) -> None:
    """Pixels outside the mask keep their color."""
    assert main(write_inputs(tmp_path)) == 0

    with Image.open(tmp_path / "output-1.png") as output:
        assert output.size == (4, 4)
        assert output.getpixel((1, 1)) == INVERTED_RED
        assert output.getpixel((2, 1)) == INVERTED_RED
        assert output.getpixel((0, 0)) == RED
        assert output.getpixel((2, 2)) == RED


def test_keeps_transparency_of_masked_pixels(tmp_path: Path) -> None:
    """A transparent masked pixel stays transparent."""
    assert main(write_inputs(tmp_path)) == 0

    with Image.open(tmp_path / "output-1.png") as output:
        pixel = output.getpixel((3, 3))
        assert isinstance(pixel, tuple)
        assert pixel[3] == 0


def test_fails_when_mask_size_differs(tmp_path: Path) -> None:
    """A mismatched mask is reported and no output is written."""
    assert main(write_inputs(tmp_path, mask_size=(3, 3))) == 1
    assert not list(tmp_path.glob("output*.png"))


def test_writes_one_file_per_variant(tmp_path: Path) -> None:
    """--variants N gives output-1.png to output-N.png and nothing else."""
    assert main([*write_inputs(tmp_path), "--variants", "3"]) == 0

    written = sorted(path.name for path in tmp_path.glob("output*.png"))
    assert written == ["output-1.png", "output-2.png", "output-3.png"]


@pytest.mark.parametrize("variants", ["0", "9"])
def test_rejects_variants_out_of_range(tmp_path: Path, variants: str) -> None:
    """The variant count is bounded to keep Aseprite's wait reasonable."""
    with pytest.raises(SystemExit):
        main([*write_inputs(tmp_path), "--variants", variants])


def test_inpaint_is_the_default_task(tmp_path: Path) -> None:
    """Without --task the bridge inpaints, as the extension expects."""
    request = parse_args(write_inputs(tmp_path))

    assert request.task is TASKS["inpaint"]
    assert main([*write_inputs(tmp_path), "--task", "inpaint"]) == 0
    with Image.open(tmp_path / "output-1.png") as output:
        assert output.getpixel((1, 1)) == INVERTED_RED


def test_rejects_unknown_task(tmp_path: Path) -> None:
    """Only registered tasks can be run."""
    with pytest.raises(SystemExit):
        main([*write_inputs(tmp_path), "--task", "teleport"])


def test_task_arguments_are_required(tmp_path: Path) -> None:
    """The inpaint task needs its region and mask."""
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "output.png")])


def test_klein_is_the_default_model(tmp_path: Path) -> None:
    """SD 1.5 inpainting is only used when asked for."""
    assert parse_args(write_inputs(tmp_path)).context.model.name == "klein-4b"
    request = parse_args([*write_inputs(tmp_path), "--model", "sd15-inpaint"])
    assert request.context.model.name == "sd15-inpaint"
    with pytest.raises(SystemExit):
        parse_args([*write_inputs(tmp_path), "--model", "sdxl"])


def test_server_url_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """COMFYUI_URL points the extension at a remote server, --comfyui-url wins."""
    argv = write_inputs(tmp_path)
    monkeypatch.delenv(URL_VARIABLE, raising=False)
    assert parse_args(argv).context.comfyui_url == DEFAULT_URL

    monkeypatch.setenv(URL_VARIABLE, "http://gpu-box:8188")
    assert parse_args(argv).context.comfyui_url == "http://gpu-box:8188"
    explicit = [*argv, "--comfyui-url", "http://other:8188"]
    assert parse_args(explicit).context.comfyui_url == "http://other:8188"
