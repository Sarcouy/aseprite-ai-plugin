"""Tests of the task registry and of the workflow builder given to ComfyUI."""

import pytest
from PIL import Image

from ai_inpaint_bridge.comfyui import (
    ComfyUIClient,
    InpaintSettings,
    ModelProfile,
    Workflow,
    build_inpaint_workflow,
)
from ai_inpaint_bridge.klein import build_klein_inpaint_workflow
from ai_inpaint_bridge.tasks import DEFAULT_TASK, TASKS


def test_inpaint_task_defaults_to_klein() -> None:
    """klein is the default model, SD 1.5 inpainting stays selectable."""
    task = TASKS[DEFAULT_TASK]

    assert task.name == "inpaint"
    assert task.model().build_inpaint is build_klein_inpaint_workflow
    sd15 = task.model("sd15-inpaint")
    assert sd15.build_inpaint is build_inpaint_workflow
    assert sd15.settings == InpaintSettings(prompt="", seed=0)
    with pytest.raises(ValueError):
        task.model("sdxl")


def test_client_queues_the_graphs_of_its_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inpainting and text to image both use the client's model."""
    queued: list[Workflow] = []

    def build(image_name: str, mask_name: str, settings: InpaintSettings) -> Workflow:
        del image_name, mask_name
        return {"1": {"class_type": "Inpaint", "inputs": {"seed": settings.seed}}}

    def build_text(settings: InpaintSettings, width: int, height: int) -> Workflow:
        del settings
        return {"1": {"class_type": "Text", "inputs": {"size": [width, height]}}}

    def queue(workflow: Workflow) -> str:
        queued.append(workflow)
        return "id"

    model = ModelProfile("probe", InpaintSettings("", 0), build, build_text)
    client = ComfyUIClient(model=model)
    monkeypatch.setattr(client, "_upload", lambda image, name: name)
    monkeypatch.setattr(client, "_queue", queue)
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {})
    monkeypatch.setattr(client, "_download_all", lambda entry: [])
    image = Image.new("RGB", (8, 8))

    client.generate(image, image, InpaintSettings(prompt="", seed=5))
    client.generate_text(InpaintSettings(prompt="", seed=5), 64, 32)

    assert queued == [
        {"1": {"class_type": "Inpaint", "inputs": {"seed": 5}}},
        {"1": {"class_type": "Text", "inputs": {"size": [64, 32]}}},
    ]
