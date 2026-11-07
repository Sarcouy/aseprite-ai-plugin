"""Inpainting graph for FLUX.2 klein 4B distilled."""

from ai_inpaint_bridge.comfyui import (
    OUTPUT_NODE,
    InpaintSettings,
    Workflow,
    load_image_nodes,
)

KLEIN_UNET = "flux-2-klein-4b-fp8.safetensors"
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
    checkpoint="",
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


def build_klein_inpaint_workflow(
    image_name: str, mask_name: str, settings: InpaintSettings
) -> Workflow:
    """Return the API-format graph: masked sampling with the image as reference.

    The encoded image is both the starting latent (with a noise mask) and a
    reference latent added to the prompt conditioning.
    """
    total = total_steps(settings)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": KLEIN_UNET, "weight_dtype": "default"},
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
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["19", 0], "vae": ["3", 0]},
        },
        OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
        "13": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["20", 0]},
        },
        "14": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": settings.sampler},
        },
        "15": {
            "class_type": "Flux2Scheduler",
            "inputs": {"steps": total, "width": ["22", 0], "height": ["22", 1]},
        },
        "16": {
            "class_type": "SplitSigmas",
            "inputs": {"sigmas": ["15", 0], "step": total - settings.steps},
        },
        "17": {"class_type": "RandomNoise", "inputs": {"noise_seed": settings.seed}},
        "18": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["20", 0],
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
                "sigmas": ["16", 1],
                "latent_image": ["9", 0],
            },
        },
        "20": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["4", 0], "latent": ["7", 0]},
        },
        "22": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
    }
