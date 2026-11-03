"""Tests of the ComfyUI workflow builder."""

from ai_inpaint_bridge.comfyui import (
    LORA_TRIGGER,
    OUTPUT_NODE,
    REPEAT_NODE,
    InpaintSettings,
    build_inpaint_workflow,
)


def test_workflow_wires_the_uploaded_images_and_settings() -> None:
    """Uploaded names, seed, denoise and prompt reach the right nodes."""
    settings = InpaintSettings(prompt="a red potion", seed=42, denoise=0.6)

    workflow = build_inpaint_workflow("in.png", "mask.png", settings)

    assert workflow["5"]["inputs"]["image"] == "in.png"
    assert workflow["6"]["inputs"] == {"image": "mask.png", "channel": "red"}
    assert workflow["9"]["inputs"]["seed"] == 42
    assert workflow["9"]["inputs"]["denoise"] == 0.6
    assert workflow["3"]["inputs"]["text"] == f"{LORA_TRIGGER}, a red potion"
    assert workflow[OUTPUT_NODE]["class_type"] == "PreviewImage"


def test_every_link_points_to_an_existing_node() -> None:
    """No input references a node id missing from the graph."""
    workflow = build_inpaint_workflow("in.png", "mask.png", InpaintSettings("", 0))

    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in workflow


def test_background_hint_is_appended_for_transparent_areas() -> None:
    """The key color name is added so the model keeps a plain background."""
    settings = InpaintSettings("a chest", 0, background="magenta")

    workflow = build_inpaint_workflow("in.png", "mask.png", settings)

    assert workflow["3"]["inputs"]["text"].endswith("a chest, plain magenta background")


def test_empty_prompt_keeps_only_the_trigger() -> None:
    """The LoRA trigger alone is sent when the user gives no prompt."""
    workflow = build_inpaint_workflow("in.png", "mask.png", InpaintSettings("", 0))

    assert workflow["3"]["inputs"]["text"] == LORA_TRIGGER


def test_batch_size_repeats_the_encoded_latent() -> None:
    """The image is encoded once and repeated before the noise mask is set."""
    workflow = build_inpaint_workflow(
        "in.png", "mask.png", InpaintSettings("", 0, batch_size=4)
    )

    assert workflow[REPEAT_NODE]["class_type"] == "RepeatLatentBatch"
    assert workflow[REPEAT_NODE]["inputs"] == {"samples": ["7", 0], "amount": 4}
    assert workflow["8"]["inputs"]["samples"] == [REPEAT_NODE, 0]
