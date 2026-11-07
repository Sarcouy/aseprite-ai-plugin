"""What a bridge task declares: its arguments, its base model and how it runs."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image

from ai_inpaint_bridge.comfyui import InpaintSettings, WorkflowBuilder
from ai_inpaint_bridge.pixelize import Color


@dataclass(frozen=True)
class ModelProfile:
    """Base model of a task: its ComfyUI graph and its default settings."""

    name: str
    build_workflow: WorkflowBuilder
    settings: InpaintSettings


@dataclass(frozen=True)
class TaskContext:
    """Options shared by every task, resolved from the command line."""

    backend: str
    model: ModelProfile
    settings: InpaintSettings
    comfyui_url: str
    palette: Sequence[Color] | None


@dataclass(frozen=True)
class Task:
    """One command of the extension, run by the bridge with --task <name>.

    The first model is the default one, the others are chosen with --model.
    """

    name: str
    description: str
    models: tuple[ModelProfile, ...]
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace, TaskContext], list[Image.Image]]

    def model(self, name: str | None = None) -> ModelProfile:
        """The model with this name, the default one when no name is given."""
        if name is None:
            return self.models[0]
        for profile in self.models:
            if profile.name == name:
                return profile
        raise ValueError(f"task {self.name} has no model {name}")
