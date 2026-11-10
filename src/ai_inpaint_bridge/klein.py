"""Graphs of FLUX.2 klein 4B (distilled and base): inpainting and text to image."""

import math
from dataclasses import replace
from typing import Any

from ai_inpaint_bridge.comfyui import (
    OUTPUT_NODE,
    InpaintSettings,
    ModelProfile,
    Workflow,
    load_image_nodes,
)

KLEIN_UNET = "flux-2-klein-4b-fp8.safetensors"
KLEIN_BASE_UNET = "flux-2-klein-base-4b-fp8.safetensors"
KLEIN_TEXT_ENCODER = "qwen_3_4b_fp8_mixed.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"
PROMPT_PREFIX = "pixel art"
KLEIN_SETTINGS = InpaintSettings(
    prompt="",
    seed=0,
    steps=4,
    cfg=1.0,
    sampler="euler",
    negative_prompt="",
    checkpoint=KLEIN_UNET,
    lora="",
    inpaint_model="",
)


def klein_prompt(settings: InpaintSettings) -> str:
    """Pixel art prefix, user prompt and background hint, skipping empty parts."""
    parts = [PROMPT_PREFIX, settings.prompt]
    if settings.background and settings.background_hint:
        parts.append(f"plain {settings.background} background")
    return ", ".join(part for part in parts if part)


def _flux2_shift(seq_len: int, steps: int) -> float:
    """exp(mu) of the FLUX.2 schedule, mu as in ComfyUI's Flux2Scheduler."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if seq_len > 4300:
        return math.exp(a2 * seq_len + b2)
    mu_200 = a2 * seq_len + b2
    mu_10 = a1 * seq_len + b1
    slope = (mu_200 - mu_10) / 190.0
    return math.exp(mu_200 + slope * (steps - 200.0))


def klein_sigmas(settings: InpaintSettings, width: int, height: int) -> list[float]:
    """Noise levels of the settings.steps sampled steps, for an image of that size.

    The FLUX.2 schedule is followed from the time settings.denoise down to 0:
    a change strength of 1 gives the full schedule, a lower one starts from a
    partly noised image. Every strength samples the same number of steps.
    """
    if settings.denoise <= 0:
        raise ValueError("the change strength must be above 0 with FLUX.2 klein")
    start = min(settings.denoise, 1.0)
    shift = _flux2_shift(round(width * height / 256), settings.steps)
    times = [start * (1 - step / settings.steps) for step in range(settings.steps)]
    return [shift / (shift + 1 / time - 1) for time in times] + [0.0]


Link = list[Any]


def _klein_nodes(
    settings: InpaintSettings,
    positive: Link,
    latent: Link,
    sigmas: dict[str, Any],
) -> Workflow:
    """Loaders, prompt, sampler and decoder shared by both graphs.

    sigmas is node 15, which gives the noise levels. settings.checkpoint is the
    diffusion model file. With a CFG of 1 the negative is the zeroed prompt,
    otherwise the empty prompt. settings.lora, when given, is applied to the
    model only.
    """
    model: Link = ["1", 0]
    extra: Workflow = {}
    if settings.lora:
        extra["23"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model,
                "lora_name": settings.lora,
                "strength_model": settings.lora_strength,
            },
        }
        model = ["23", 0]
    if settings.cfg == 1:
        negative: dict[str, Any] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": positive},
        }
    else:
        negative = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": settings.negative_prompt},
        }
    return {
        **extra,
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": settings.checkpoint, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": KLEIN_TEXT_ENCODER,
                "type": "flux2",
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": klein_prompt(settings)},
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["19", 0], "vae": ["3", 0]},
        },
        OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
        "13": negative,
        "14": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": settings.sampler},
        },
        "15": sigmas,
        "17": {"class_type": "RandomNoise", "inputs": {"noise_seed": settings.seed}},
        "18": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": model,
                "positive": positive,
                "negative": ["13", 0],
                "cfg": settings.cfg,
            },
        },
        "19": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["17", 0],
                "guider": ["18", 0],
                "sampler": ["14", 0],
                "sigmas": ["15", 0],
                "latent_image": latent,
            },
        },
    }


def build_klein_inpaint_workflow(
    image_name: str, mask_name: str, settings: InpaintSettings, size: tuple[int, int]
) -> Workflow:
    """Return the API-format graph: masked sampling with the image as reference.

    The encoded image is both the starting latent (with a noise mask) and a
    reference latent added to the prompt conditioning. size is the uploaded
    image size, which sets the schedule.
    """
    sigmas = ", ".join(f"{sigma:.10f}" for sigma in klein_sigmas(settings, *size))
    return {
        **_klein_nodes(
            settings,
            positive=["20", 0],
            latent=["9", 0],
            sigmas={"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}},
        ),
        **load_image_nodes(image_name, mask_name),
        "7": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
        },
        "8": {
            "class_type": "RepeatLatentBatch",
            "inputs": {"samples": ["7", 0], "amount": settings.batch_size},
        },
        "9": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["8", 0], "mask": ["6", 0]},
        },
        "20": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["4", 0], "latent": ["7", 0]},
        },
    }


def build_klein_text_workflow(
    settings: InpaintSettings, width: int, height: int
) -> Workflow:
    """Return the API-format graph: text to image from an empty latent."""
    return {
        **_klein_nodes(
            settings,
            positive=["4", 0],
            latent=["7", 0],
            sigmas={
                "class_type": "Flux2Scheduler",
                "inputs": {"steps": settings.steps, "width": width, "height": height},
            },
        ),
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": settings.batch_size,
            },
        },
    }


def build_klein_reference_workflow(
    image_name: str, settings: InpaintSettings, width: int, height: int
) -> Workflow:
    """Return the API-format graph: text to image with the uploaded image as reference.

    The encoded image is added as a reference latent to the prompt only: on
    the negative prompt too, the guidance would cancel it out and the
    character would lose its look.
    """
    return {
        **_klein_nodes(
            settings,
            positive=["20", 0],
            latent=["7", 0],
            sigmas={
                "class_type": "Flux2Scheduler",
                "inputs": {"steps": settings.steps, "width": width, "height": height},
            },
        ),
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": settings.batch_size,
            },
        },
        "9": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
        },
        "20": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["4", 0], "latent": ["9", 0]},
        },
    }


def build_klein_guided_workflow(
    image_name: str, guide_name: str, settings: InpaintSettings, width: int, height: int
) -> Workflow:
    """Return the API-format graph: the guide redrawn with the image as reference.

    Sampling starts from the encoded guide, noised to settings.denoise: the
    guide sets the pose, the reference latent on the prompt keeps the look.
    """
    sigmas = ", ".join(
        f"{sigma:.10f}" for sigma in klein_sigmas(settings, width, height)
    )
    return {
        **_klein_nodes(
            settings,
            positive=["20", 0],
            latent=["8", 0],
            sigmas={"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}},
        ),
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "LoadImage", "inputs": {"image": guide_name}},
        "7": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["6", 0], "vae": ["3", 0]},
        },
        "8": {
            "class_type": "RepeatLatentBatch",
            "inputs": {"samples": ["7", 0], "amount": settings.batch_size},
        },
        "9": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
        },
        "20": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["4", 0], "latent": ["9", 0]},
        },
    }


LATENT_SCALE = 16
LATENT_UNIT = 8
WRAP_CELLS = 8


def _latent_crop(
    workflow: Workflow, node: str, samples: Link, box: tuple[int, int, int, int]
) -> Link:
    """Crop box (x, y, width, height) of a latent, in latent cells."""
    x, y, width, height = box
    workflow[node] = {
        "class_type": "LatentCrop",
        "inputs": {
            "samples": samples,
            "x": x * LATENT_UNIT,
            "y": y * LATENT_UNIT,
            "width": width * LATENT_UNIT,
            "height": height * LATENT_UNIT,
        },
    }
    return [node, 0]


def _latent_paste(
    workflow: Workflow, node: str, target: Link, samples: Link, at: tuple[int, int]
) -> Link:
    """samples pasted on target at (x, y), in latent cells."""
    workflow[node] = {
        "class_type": "LatentComposite",
        "inputs": {
            "samples_to": target,
            "samples_from": samples,
            "x": at[0] * LATENT_UNIT,
            "y": at[1] * LATENT_UNIT,
            "feather": 0,
        },
    }
    return [node, 0]


def _half_roll(workflow: Workflow, prefix: str, samples: Link, cells: int) -> Link:
    """The latent rolled by half its size on both axes, its corners swapped."""
    half = cells // 2
    rolled = samples
    for index, (x, y) in enumerate(((0, 0), (half, 0), (0, half), (half, half))):
        piece = _latent_crop(workflow, f"{prefix}c{index}", samples, (x, y, half, half))
        rolled = _latent_paste(
            workflow,
            f"{prefix}p{index}",
            rolled,
            piece,
            ((x + half) % cells, (y + half) % cells),
        )
    return rolled


def _empty_square(cells: int, batch_size: int) -> dict[str, Any]:
    side = cells * LATENT_SCALE
    return {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": side, "height": side, "batch_size": batch_size},
    }


def _wrapped(workflow: Workflow, samples: Link, cells: int, batch_size: int) -> Link:
    """The latent surrounded by WRAP_CELLS of its opposite edges, as if repeated.

    The latent is pasted at the 4 corners of the inner square, so that its
    first rows and columns fill the right and bottom margins; its last ones,
    cropped, fill the left and top margins.
    """
    workflow["50"] = _empty_square(cells + 2 * WRAP_CELLS, batch_size)
    edge = cells - WRAP_CELLS
    near, far = WRAP_CELLS, WRAP_CELLS + cells
    crops = {
        "right": ((edge, 0, WRAP_CELLS, cells), ((0, near), (0, far))),
        "bottom": ((0, edge, cells, WRAP_CELLS), ((near, 0), (far, 0))),
        "corner": ((edge, edge, WRAP_CELLS, WRAP_CELLS), ((0, 0),)),
    }
    pieces = [
        (samples, at) for at in ((near, near), (far, near), (near, far), (far, far))
    ]
    for index, (box, places) in enumerate(crops.values()):
        piece = _latent_crop(workflow, f"5{index + 1}", samples, box)
        pieces.extend((piece, at) for at in places)
    wrapped: Link = ["50", 0]
    for index, (piece, at) in enumerate(pieces):
        wrapped = _latent_paste(workflow, f"6{index}", wrapped, piece, at)
    return wrapped


def _check_tile_size(width: int, height: int) -> int:
    """Latent cells of the side of a square of 256 x n pixels."""
    cells = width // LATENT_SCALE
    if width != height or cells % (2 * WRAP_CELLS):
        raise ValueError(
            f"a tile is drawn on a square of 256 x n pixels, not {width}x{height}"
        )
    return cells


def _seamless(
    workflow: Workflow,
    settings: InpaintSettings,
    start: Link,
    sigmas: list[float],
    width: int,
) -> Workflow:
    """Sample the start latent step by step, rolled by half between two steps.

    The latent is decoded surrounded by its opposite edges and cropped back,
    so that neither the model nor the decoder sees an edge.
    """
    cells = width // LATENT_SCALE
    del workflow["15"], workflow["19"]
    workflow["40"] = {"class_type": "DisableNoise", "inputs": {}}
    latent = start
    for step in range(settings.steps):
        if step:
            latent = _half_roll(workflow, f"r{step}", latent, cells)
        workflow[f"s{step}"] = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": f"{sigmas[step]:.10f}, {sigmas[step + 1]:.10f}"},
        }
        workflow[f"k{step}"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["17", 0] if step == 0 else ["40", 0],
                "guider": ["18", 0],
                "sampler": ["14", 0],
                "sigmas": [f"s{step}", 0],
                "latent_image": latent,
            },
        }
        latent = [f"k{step}", 0]
    workflow["10"]["inputs"]["samples"] = _wrapped(
        workflow, latent, cells, settings.batch_size
    )
    margin = WRAP_CELLS * LATENT_SCALE
    workflow["54"] = {
        "class_type": "ImageCrop",
        "inputs": {
            "image": ["10", 0],
            "width": width,
            "height": width,
            "x": margin,
            "y": margin,
        },
    }
    workflow[OUTPUT_NODE]["inputs"]["images"] = ["54", 0]
    return workflow


def build_klein_tile_workflow(
    settings: InpaintSettings, width: int, height: int
) -> Workflow:
    """Return the API-format graph: text to an image that repeats without seams.

    Each step is sampled alone; between two steps the latent is rolled by half
    its size, so that its edges meet in the middle, where the model paints
    across them. The square side must be a multiple of 256 pixels: the rolled
    halves are at least WRAP_CELLS latent cells wide.
    """
    _check_tile_size(width, height)
    sigmas = klein_sigmas(replace(settings, denoise=1.0), width, height)
    workflow = build_klein_text_workflow(settings, width, height)
    return _seamless(workflow, settings, ["7", 0], sigmas, width)


def build_klein_tile_redraw_workflow(
    image_name: str, settings: InpaintSettings, width: int, height: int
) -> Workflow:
    """Return the API-format graph: an image redrawn so that it repeats without seams.

    Sampling starts from the encoded image noised to settings.denoise, then
    goes on as in build_klein_tile_workflow.
    """
    _check_tile_size(width, height)
    sigmas = klein_sigmas(settings, width, height)
    workflow = build_klein_text_workflow(settings, width, height)
    del workflow["7"]
    workflow["5"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    workflow["8"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
    }
    workflow["9"] = {
        "class_type": "RepeatLatentBatch",
        "inputs": {"samples": ["8", 0], "amount": settings.batch_size},
    }
    return _seamless(workflow, settings, ["9", 0], sigmas, width)


KLEIN_4B = ModelProfile(
    name="klein-4b",
    settings=KLEIN_SETTINGS,
    build_inpaint=build_klein_inpaint_workflow,
    build_text=build_klein_text_workflow,
    build_reference=build_klein_reference_workflow,
    build_guided=build_klein_guided_workflow,
)

KLEIN_4B_BASE = ModelProfile(
    name="klein-4b-base",
    settings=replace(KLEIN_SETTINGS, checkpoint=KLEIN_BASE_UNET, steps=20, cfg=5.0),
    build_inpaint=build_klein_inpaint_workflow,
    build_text=build_klein_text_workflow,
    build_reference=build_klein_reference_workflow,
)

SVNTAX_LORA = "klein4b-base-pixel-4walk-svntax.safetensors"
KLEIN_4B_BASE_WALK = replace(
    KLEIN_4B_BASE,
    name="klein-4b-base-svntax",
    settings=replace(KLEIN_4B_BASE.settings, lora=SVNTAX_LORA),
)

KLEIN_4B_TILE = replace(
    KLEIN_4B,
    name="klein-4b-tile",
    build_text=build_klein_tile_workflow,
    build_redraw=build_klein_tile_redraw_workflow,
)
