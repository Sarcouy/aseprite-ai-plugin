"""Command line entry point called by the Aseprite extension."""

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from ai_inpaint_bridge.backends import diffusion_inpaint, invert_masked
from ai_inpaint_bridge.comfyui import (
    DEFAULT_URL,
    ComfyUIClient,
    ComfyUIError,
    InpaintSettings,
)
from ai_inpaint_bridge.pixelize import MIN_PALETTE_COLORS, Color, palette_from_strip

BACKENDS = ("invert", "comfyui")
MAX_SEED = 2**32 - 1
MAX_VARIANTS = 8


@dataclass(frozen=True)
class BridgeRequest:
    """Files and settings exchanged with Aseprite for one inpainting run."""

    source: Path
    mask: Path
    output: Path
    backend: str
    settings: InpaintSettings
    comfyui_url: str
    palette: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> BridgeRequest:
    """Parse the command line into a bridge request."""
    parser = argparse.ArgumentParser(
        prog="ai-inpaint-bridge",
        description="Rewrite the masked area of a sprite region.",
    )
    parser.add_argument("--input", type=Path, required=True, help="region PNG")
    parser.add_argument(
        "--mask", type=Path, required=True, help="mask PNG, white = rewrite"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="result PNG name, variant i is written as <stem>-<i><suffix>",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="invert")
    parser.add_argument("--prompt", default="", help="what to draw in the selection")
    parser.add_argument("--seed", type=int, help="random when omitted")
    parser.add_argument(
        "--denoise", type=float, default=0.8, help="0 keeps the pixels, 1 redraws"
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        choices=range(1, MAX_VARIANTS + 1),
        metavar=f"1..{MAX_VARIANTS}",
        help="number of results generated in one pass",
    )
    parser.add_argument("--comfyui-url", default=DEFAULT_URL)
    parser.add_argument(
        "--palette", type=Path, help="PNG strip of the colors to snap to"
    )
    args = parser.parse_args(argv)
    seed: int = args.seed if args.seed is not None else random.randint(0, MAX_SEED)
    return BridgeRequest(
        source=args.input,
        mask=args.mask,
        output=args.output,
        backend=args.backend,
        settings=InpaintSettings(
            prompt=args.prompt,
            seed=seed,
            denoise=args.denoise,
            batch_size=args.variants,
        ),
        comfyui_url=args.comfyui_url,
        palette=args.palette,
    )


def load_palette(path: Path | None) -> list[Color] | None:
    """Read the palette strip exported by Aseprite.

    A palette with a single color (e.g. an uninitialised all-black one) is
    ignored so that an adaptive palette is used instead.
    """
    if path is None:
        return None
    with Image.open(path) as strip:
        palette = palette_from_strip(strip)
    return palette if len(palette) >= MIN_PALETTE_COLORS else None


def variant_path(output: Path, index: int) -> Path:
    """Path of the variant numbered from 1: out.png gives out-1.png, out-2.png..."""
    return output.with_name(f"{output.stem}-{index}{output.suffix}")


def run(request: BridgeRequest) -> None:
    """Compute the result images and write them next to the inputs."""
    palette = load_palette(request.palette)
    with Image.open(request.source) as source, Image.open(request.mask) as mask:
        if request.backend == "comfyui":
            client = ComfyUIClient(request.comfyui_url)
            results = diffusion_inpaint(source, mask, request.settings, client, palette)
        else:
            results = [invert_masked(source, mask)] * request.settings.batch_size
    for index, result in enumerate(results, start=1):
        result.save(variant_path(request.output, index))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge and report errors on stderr for the Aseprite dialog."""
    request = parse_args(argv)
    try:
        run(request)
    except requests.ConnectionError:
        print(
            f"ai-inpaint-bridge: ComfyUI is not reachable at {request.comfyui_url}, "
            "start it with scripts/start-comfyui.sh",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, ComfyUIError) as error:
        print(f"ai-inpaint-bridge: {error}", file=sys.stderr)
        return 1
    return 0
