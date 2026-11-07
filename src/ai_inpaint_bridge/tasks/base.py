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
    """One command of the extension, run by the bridge with --task <name>."""

    name: str
    description: str
    model: ModelProfile
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace, TaskContext], list[Image.Image]]
