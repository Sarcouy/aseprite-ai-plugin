"""Tests of the bridge command line with the invert backend."""

from pathlib import Path

from PIL import Image

from ai_inpaint_bridge.cli import load_palette, main

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


def test_inverts_only_masked_pixels(tmp_path: Path) -> None:
    """Pixels outside the mask keep their color."""
    assert main(write_inputs(tmp_path)) == 0

    with Image.open(tmp_path / "output.png") as output:
        assert output.size == (4, 4)
        assert output.getpixel((1, 1)) == INVERTED_RED
        assert output.getpixel((2, 1)) == INVERTED_RED
        assert output.getpixel((0, 0)) == RED
        assert output.getpixel((2, 2)) == RED


def test_keeps_transparency_of_masked_pixels(tmp_path: Path) -> None:
    """A transparent masked pixel stays transparent."""
    assert main(write_inputs(tmp_path)) == 0

    with Image.open(tmp_path / "output.png") as output:
        pixel = output.getpixel((3, 3))
        assert isinstance(pixel, tuple)
        assert pixel[3] == 0


def test_fails_when_mask_size_differs(tmp_path: Path) -> None:
    """A mismatched mask is reported and no output is written."""
    assert main(write_inputs(tmp_path, mask_size=(3, 3))) == 1
    assert not (tmp_path / "output.png").exists()
