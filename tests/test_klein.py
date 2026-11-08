"""Tests of the FLUX.2 klein inpainting graph."""

from dataclasses import replace

import pytest

from ai_inpaint_bridge.comfyui import OUTPUT_NODE
from ai_inpaint_bridge.klein import (
    KLEIN_4B_BASE,
    KLEIN_BASE_UNET,
    KLEIN_UNET,
    KLEIN_SETTINGS,
    build_klein_inpaint_workflow,
    build_klein_text_workflow,
    klein_prompt,
    klein_sigmas,
)


def test_prompt_has_no_sd15_lora_trigger() -> None:
    """klein gets a plain pixel art prefix, no LoRA trigger word."""
    settings = replace(KLEIN_SETTINGS, prompt="a red potion")

    assert klein_prompt(settings) == "pixel art, a red potion"
    assert klein_prompt(KLEIN_SETTINGS) == "pixel art"


def test_background_hint_is_optional() -> None:
    """The key color is only named when the hint is enabled."""
    settings = replace(KLEIN_SETTINGS, prompt="a chest", background="magenta")

    assert klein_prompt(settings) == "pixel art, a chest"
    hinted = replace(settings, background_hint=True)
    assert klein_prompt(hinted) == "pixel art, a chest, plain magenta background"


@pytest.mark.parametrize(
    ("steps", "size", "expected"),
    [
        (4, (512, 512), [1.0, 0.958085, 0.883982, 0.717497, 0.0]),
        (4, (416, 512), [1.0, 0.957427, 0.882302, 0.714185, 0.0]),
        (4, (1024, 1024), [1.0, 0.967384, 0.908144, 0.7672, 0.0]),
        (
            20,
            (512, 512),
            [1.0, 0.992315, 0.983914, 0.974691, 0.964519, 0.953245, 0.940679]
            + [0.926586, 0.910668, 0.892546, 0.871731, 0.847571, 0.819192]
            + [0.785382, 0.744416, 0.693755, 0.629496, 0.545312, 0.430239]
            + [0.263454, 0.0],
        ),
    ],
)
def test_full_strength_matches_comfyui_flux2_schedule(
    steps: int, size: tuple[int, int], expected: list[float]
) -> None:
    """At strength 1 the schedule is the one of ComfyUI's Flux2Scheduler."""
    settings = replace(KLEIN_SETTINGS, steps=steps, denoise=1.0)

    assert klein_sigmas(settings, *size) == pytest.approx(expected, abs=1e-6)


def test_lower_strength_starts_later_with_the_same_step_count() -> None:
    """Each strength samples 4 steps, from a lower noise level when it is lower."""
    starts = []
    for denoise in (0.6, 0.8, 0.9, 1.0):
        sigmas = klein_sigmas(replace(KLEIN_SETTINGS, denoise=denoise), 512, 512)
        assert len(sigmas) == 5
        assert sigmas == sorted(sigmas, reverse=True)
        assert sigmas[-1] == 0.0
        starts.append(sigmas[0])

    assert starts == sorted(starts)
    assert len(set(starts)) == 4
    assert starts[1] == pytest.approx(0.968, abs=1e-3)


@pytest.mark.parametrize("denoise", [0.0, -0.2])
def test_rejects_zero_strength(denoise: float) -> None:
    """A zero change strength would not change anything."""
    with pytest.raises(ValueError, match="above 0"):
        klein_sigmas(replace(KLEIN_SETTINGS, denoise=denoise), 512, 512)


def test_workflow_uses_the_image_as_reference() -> None:
    """The encoded image conditions the prompt and seeds the masked latent."""
    settings = replace(KLEIN_SETTINGS, prompt="a gem", seed=42, denoise=0.8)
    settings = replace(settings, batch_size=4)

    workflow = build_klein_inpaint_workflow("in.png", "mask.png", settings, (512, 512))

    assert workflow["5"]["inputs"]["image"] == "in.png"
    assert workflow["6"]["inputs"] == {"image": "mask.png", "channel": "red"}
    assert workflow["4"]["inputs"]["text"] == "pixel art, a gem"
    assert workflow["20"]["inputs"] == {"conditioning": ["4", 0], "latent": ["7", 0]}
    assert workflow["8"]["inputs"] == {"samples": ["7", 0], "amount": 4}
    assert workflow["17"]["inputs"]["noise_seed"] == 42
    assert workflow["18"]["inputs"]["cfg"] == 1.0
    assert workflow["15"]["class_type"] == "ManualSigmas"
    sigmas = [float(x) for x in workflow["15"]["inputs"]["sigmas"].split(", ")]
    assert sigmas == pytest.approx(klein_sigmas(settings, 512, 512), abs=1e-9)
    assert workflow["19"]["inputs"]["sigmas"] == ["15", 0]
    assert workflow[OUTPUT_NODE]["inputs"]["images"] == ["10", 0]


def test_text_workflow_starts_from_an_empty_latent() -> None:
    """Text to image: full 4 step schedule at the requested size, no image node."""
    settings = replace(KLEIN_SETTINGS, prompt="a knight", seed=7, batch_size=4)

    workflow = build_klein_text_workflow(settings, 512, 256)

    assert workflow["7"] == {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": 512, "height": 256, "batch_size": 4},
    }
    assert workflow["15"]["inputs"] == {"steps": 4, "width": 512, "height": 256}
    assert workflow["19"]["inputs"]["sigmas"] == ["15", 0]
    assert workflow["18"]["inputs"]["positive"] == ["4", 0]
    assert workflow["4"]["inputs"]["text"] == "pixel art, a knight"
    assert not {"5", "6", "20", "22"} & set(workflow)


def test_lora_and_base_model_reach_the_graph() -> None:
    """klein base samples with a real negative; a LoRA patches the model only."""
    settings = replace(
        KLEIN_4B_BASE.settings, lora="walk.safetensors", lora_strength=0.8
    )

    workflow = build_klein_text_workflow(settings, 512, 512)

    assert workflow["1"]["inputs"]["unet_name"] == KLEIN_BASE_UNET
    assert workflow["23"]["inputs"] == {
        "model": ["1", 0],
        "lora_name": "walk.safetensors",
        "strength_model": 0.8,
    }
    assert workflow["18"]["inputs"]["model"] == ["23", 0]
    assert workflow["18"]["inputs"]["cfg"] == 5.0
    assert workflow["13"]["class_type"] == "CLIPTextEncode"
    assert workflow["15"]["inputs"]["steps"] == 20


def test_distilled_model_has_no_lora_by_default() -> None:
    """The default graph keeps the zeroed negative and the bare model."""
    workflow = build_klein_text_workflow(KLEIN_SETTINGS, 512, 512)

    assert "23" not in workflow
    assert workflow["1"]["inputs"]["unet_name"] == KLEIN_UNET
    assert workflow["13"]["class_type"] == "ConditioningZeroOut"
