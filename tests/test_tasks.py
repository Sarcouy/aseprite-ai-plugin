"""Tests of the task registry and of the workflow builder given to ComfyUI."""

import pytest
from PIL import Image

from ai_inpaint_bridge.comfyui import (
    ComfyUIClient,
    InpaintSettings,
    Workflow,
    build_inpaint_workflow,
)
from ai_inpaint_bridge.tasks import DEFAULT_TASK, TASKS


def test_inpaint_task_uses_sd15_inpainting() -> None:
    """The default task keeps the SD 1.5 inpainting graph and settings."""
    task = TASKS[DEFAULT_TASK]

    assert task.name == "inpaint"
    assert task.model.build_workflow is build_inpaint_workflow
    assert task.model.settings == InpaintSettings(prompt="", seed=0)


def test_client_queues_the_injected_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client sends the graph built by the task's model."""
    queued: list[Workflow] = []

    def build(image_name: str, mask_name: str, settings: InpaintSettings) -> Workflow:
        del image_name, mask_name
        return {"1": {"class_type": "Probe", "inputs": {"seed": settings.seed}}}

    def queue(workflow: Workflow) -> str:
        queued.append(workflow)
        return "id"

    client = ComfyUIClient(build_workflow=build)
    monkeypatch.setattr(client, "_upload", lambda image, name: name)
    monkeypatch.setattr(client, "_queue", queue)
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {})
    monkeypatch.setattr(client, "_download_all", lambda entry: [])
    image = Image.new("RGB", (8, 8))

    client.generate(image, image, InpaintSettings(prompt="", seed=5))

    assert queued == [{"1": {"class_type": "Probe", "inputs": {"seed": 5}}}]
