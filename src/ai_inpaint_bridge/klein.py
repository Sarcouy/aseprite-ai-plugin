"""Graphs of FLUX.2 klein 4B (distilled and base): inpainting and text to image."""

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


def total_steps(settings: InpaintSettings) -> int:
    """Length of the schedule whose last settings.steps steps are sampled.

    Skipping the first steps of a longer schedule starts from a partly noised
    image, which is how a change strength below 1 is applied.
    """
    if settings.denoise <= 0:
        raise ValueError("the change strength must be above 0 with FLUX.2 klein")
    return max(settings.steps, round(settings.steps / settings.denoise))


Link = list[Any]


def _klein_nodes(
    settings: InpaintSettings,
    positive: Link,
    latent: Link,
    schedule: dict[str, Any],
    sigmas: Link,
) -> Workflow:
    """Loaders, prompt, sampler and decoder shared by both graphs.

    settings.checkpoint is the diffusion model file. With a CFG of 1 the
    negative is the zeroed prompt, otherwise the empty prompt. settings.lora,
    when given, is applied to the model only.
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
        "15": {"class_type": "Flux2Scheduler", "inputs": schedule},
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
                "sigmas": sigmas,
                "latent_image": latent,
            },
        },
    }


def build_klein_inpaint_workflow(
    image_name: str, mask_name: str, settings: InpaintSettings
) -> Workflow:
    """Return the API-format graph: masked sampling with the image as reference.

    The encoded image is both the starting latent (with a noise mask) and a
    reference latent added to the prompt conditioning.
    """
    total = total_steps(settings)
    return {
        **_klein_nodes(
            settings,
            positive=["20", 0],
            latent=["9", 0],
            schedule={"steps": total, "width": ["22", 0], "height": ["22", 1]},
            sigmas=["16", 1],
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
        "16": {
            "class_type": "SplitSigmas",
            "inputs": {"sigmas": ["15", 0], "step": total - settings.steps},
        },
        "20": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["4", 0], "latent": ["7", 0]},
        },
        "22": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
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
            schedule={"steps": settings.steps, "width": width, "height": height},
            sigmas=["15", 0],
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


KLEIN_4B = ModelProfile(
    name="klein-4b",
    settings=KLEIN_SETTINGS,
    build_inpaint=build_klein_inpaint_workflow,
    build_text=build_klein_text_workflow,
)

KLEIN_4B_BASE = ModelProfile(
    name="klein-4b-base",
    settings=replace(KLEIN_SETTINGS, checkpoint=KLEIN_BASE_UNET, steps=20, cfg=5.0),
    build_inpaint=build_klein_inpaint_workflow,
    build_text=build_klein_text_workflow,
)
