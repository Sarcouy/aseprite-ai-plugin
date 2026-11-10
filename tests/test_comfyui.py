"""Tests of the ComfyUI workflow builder and client."""

from typing import cast

import pytest
import requests
from PIL import Image

from ai_inpaint_bridge.comfyui import (
    LORA_TRIGGER,
    ComfyUIClient,
    ComfyUIError,
    OUTPUT_NODE,
    INPAINT_UNET_NODE,
    REPEAT_NODE,
    InpaintSettings,
    build_inpaint_workflow,
    build_sd15_text_workflow,
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


@pytest.mark.parametrize("inpaint_model", ["", "inpaint.safetensors"])
def test_every_link_points_to_an_existing_node(inpaint_model: str) -> None:
    """No input references a node id missing from the graph."""
    settings = InpaintSettings("", 0, inpaint_model=inpaint_model)
    workflow = build_inpaint_workflow("in.png", "mask.png", settings)

    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in workflow


def test_background_hint_is_appended_for_transparent_areas() -> None:
    """The key color name is added so the model keeps a plain background."""
    settings = InpaintSettings("a chest", 0, background="magenta", background_hint=True)

    workflow = build_inpaint_workflow("in.png", "mask.png", settings)

    assert workflow["3"]["inputs"]["text"].endswith("a chest, plain magenta background")


def test_empty_prompt_keeps_only_the_trigger() -> None:
    """The LoRA trigger alone is sent when the user gives no prompt."""
    workflow = build_inpaint_workflow("in.png", "mask.png", InpaintSettings("", 0))

    assert workflow["3"]["inputs"]["text"] == LORA_TRIGGER


def test_batch_size_repeats_the_encoded_latent() -> None:
    """With the checkpoint UNet, the latent is repeated before the noise mask is set."""
    workflow = build_inpaint_workflow(
        "in.png", "mask.png", InpaintSettings("", 0, batch_size=4, inpaint_model="")
    )

    assert workflow[REPEAT_NODE]["class_type"] == "RepeatLatentBatch"
    assert workflow[REPEAT_NODE]["inputs"] == {"samples": ["7", 0], "amount": 4}
    assert workflow["8"]["inputs"]["samples"] == [REPEAT_NODE, 0]


def test_inpaint_model_conditions_on_the_masked_image() -> None:
    """The inpainting UNet carries the LoRA and samples the conditioned latent."""
    settings = InpaintSettings("", 0, inpaint_model="inpaint.safetensors")

    workflow = build_inpaint_workflow("in.png", "mask.png", settings)

    assert workflow[INPAINT_UNET_NODE]["inputs"]["unet_name"] == "inpaint.safetensors"
    assert workflow["2"]["inputs"]["model"] == [INPAINT_UNET_NODE, 0]
    assert workflow["7"]["class_type"] == "InpaintModelConditioning"
    assert workflow[REPEAT_NODE]["inputs"]["samples"] == ["7", 2]
    sampler = workflow["9"]["inputs"]
    assert (sampler["positive"], sampler["negative"]) == (["7", 0], ["7", 1])
    assert "8" not in workflow


def test_background_hint_can_be_disabled() -> None:
    """Without the hint the key color is not named in the prompt."""
    settings = InpaintSettings(
        "a chest", 0, background="magenta", background_hint=False
    )

    workflow = build_inpaint_workflow("in.png", "mask.png", settings)

    assert workflow["3"]["inputs"]["text"] == f"{LORA_TRIGGER}, a chest"


class FakeSession:  # pylint: disable=too-few-public-methods  # stands in for the one requests.Session method the cancel path uses
    """Records the POST requests, as a requests.Session would send them."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, object]] = []

    def post(self, url: str, json: object = None, **kwargs: object) -> None:
        """Record the URL and JSON body."""
        del kwargs
        self.posts.append((url, json))


def test_interrupted_wait_cancels_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping the bridge removes its prompt from ComfyUI."""
    session = FakeSession()
    client = ComfyUIClient("http://comfy", session=cast(requests.Session, session))
    monkeypatch.setattr(client, "_upload", lambda image, name: name)
    monkeypatch.setattr(client, "_queue", lambda workflow: "p1")

    def interrupted(prompt_id: str) -> dict[str, object]:
        raise KeyboardInterrupt(prompt_id)

    monkeypatch.setattr(client, "_wait", interrupted)
    image = Image.new("RGB", (8, 8))

    with pytest.raises(KeyboardInterrupt):
        client.generate(image, image, InpaintSettings(prompt="", seed=1))

    assert session.posts == [
        ("http://comfy/queue", {"delete": ["p1"]}),
        ("http://comfy/interrupt", {"prompt_id": "p1"}),
    ]


def test_sd15_text_workflow_uses_the_checkpoint() -> None:
    """Text to image draws from an empty latent with the checkpoint's UNet."""
    settings = InpaintSettings("a knight", 3, batch_size=2)

    workflow = build_sd15_text_workflow(settings, 512, 256)

    assert workflow["7"]["inputs"] == {"width": 512, "height": 256, "batch_size": 2}
    assert workflow["2"]["inputs"]["model"] == ["1", 0]
    assert workflow["9"]["inputs"]["latent_image"] == ["7", 0]
    assert workflow["9"]["inputs"]["denoise"] == 1.0
    assert workflow["3"]["inputs"]["text"] == f"{LORA_TRIGGER}, a knight"
    assert INPAINT_UNET_NODE not in workflow and "5" not in workflow


def test_trigger_word_only_comes_with_the_default_lora() -> None:
    """Another LoRA, or none, does not get the PixelArtRedmond trigger."""
    other = InpaintSettings("a chest", 1, lora="other.safetensors")

    workflow = build_inpaint_workflow("in.png", "mask.png", other)

    assert workflow["3"]["inputs"]["text"] == "a chest"


def test_model_without_guided_graph_refuses_a_guide() -> None:
    """SD 1.5 has no graph redrawing a guide: nothing is uploaded."""
    client = ComfyUIClient(session=cast(requests.Session, FakeSession()))
    image = Image.new("RGB", (8, 8))

    with pytest.raises(ComfyUIError, match="cannot redraw a guide"):
        client.generate_guided(image, image, InpaintSettings("", 0))
