"""Generation through the HTTP API of a local ComfyUI server."""

import io
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import requests
from PIL import Image

DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"
DEFAULT_LORA = "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
LORA_TRIGGER = "pixel art, PixArFK"
DEFAULT_NEGATIVE = (
    "blurry, smooth, gradient, photo, realistic, 3d render, text, watermark"
)
OUTPUT_NODE = "11"
REPEAT_NODE = "12"
INPAINT_UNET_NODE = "13"
DEFAULT_INPAINT_MODEL = "sd-v1-5-inpainting-unet-fp16.safetensors"

Workflow = dict[str, dict[str, Any]]


class ComfyUIError(Exception):
    """ComfyUI rejected the workflow or failed to run it."""


@dataclass(frozen=True)
class InpaintSettings:  # pylint: disable=too-many-instance-attributes  # one field per workflow parameter, grouping them would only add indirection
    """Generation parameters sent to the inpainting workflow."""

    prompt: str
    seed: int
    denoise: float = 0.8
    steps: int = 20
    cfg: float = 10.0
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    negative_prompt: str = DEFAULT_NEGATIVE
    checkpoint: str = DEFAULT_CHECKPOINT
    lora: str = DEFAULT_LORA
    lora_strength: float = 1.0
    background: str = ""
    background_hint: bool = False
    batch_size: int = 1
    inpaint_model: str = DEFAULT_INPAINT_MODEL
    mask_grow: int = 2


WorkflowBuilder = Callable[[str, str, InpaintSettings, tuple[int, int]], Workflow]
TextWorkflowBuilder = Callable[[InpaintSettings, int, int], Workflow]
ReferenceWorkflowBuilder = Callable[[str, InpaintSettings, int, int], Workflow]
GuidedWorkflowBuilder = Callable[[str, str, InpaintSettings, int, int], Workflow]


def load_image_nodes(image_name: str, mask_name: str) -> Workflow:
    """Nodes 5 (uploaded image) and 6 (its mask, white = rewrite)."""
    return {
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {
            "class_type": "LoadImageMask",
            "inputs": {"image": mask_name, "channel": "red"},
        },
    }


def positive_prompt(settings: InpaintSettings) -> str:
    """Trigger of the default LoRA, user prompt and background hint, skipping empty parts."""
    trigger = LORA_TRIGGER if settings.lora == DEFAULT_LORA else ""
    parts = [trigger, settings.prompt]
    if settings.background and settings.background_hint:
        parts.append(f"plain {settings.background} background")
    return ", ".join(part for part in parts if part)


def build_inpaint_workflow(
    image_name: str,
    mask_name: str,
    settings: InpaintSettings,
    size: tuple[int, int] | None = None,
) -> Workflow:
    """Return the API-format graph: masked img2img on an SD checkpoint with a LoRA.

    With an inpaint_model, its UNet replaces the checkpoint's one (CLIP and VAE
    are shared with SD 1.5) and also receives the masked image as conditioning.
    The image size is not needed.
    """
    del size
    positive = positive_prompt(settings)
    workflow: Workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": settings.checkpoint},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": settings.lora,
                "strength_model": settings.lora_strength,
                "strength_clip": settings.lora_strength,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["2", 1]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": settings.negative_prompt, "clip": ["2", 1]},
        },
        **load_image_nodes(image_name, mask_name),
        "7": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["1", 2]},
        },
        REPEAT_NODE: {
            "class_type": "RepeatLatentBatch",
            "inputs": {"samples": ["7", 0], "amount": settings.batch_size},
        },
        "8": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": [REPEAT_NODE, 0], "mask": ["6", 0]},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["8", 0],
                "seed": settings.seed,
                "steps": settings.steps,
                "cfg": settings.cfg,
                "sampler_name": settings.sampler,
                "scheduler": settings.scheduler,
                "denoise": settings.denoise,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
        },
        OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
    }
    if settings.inpaint_model:
        _use_inpaint_model(workflow, settings.inpaint_model)
    return workflow


def _use_inpaint_model(workflow: Workflow, unet_name: str) -> None:
    workflow[INPAINT_UNET_NODE] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
    }
    workflow["2"]["inputs"]["model"] = [INPAINT_UNET_NODE, 0]
    workflow["7"] = {
        "class_type": "InpaintModelConditioning",
        "inputs": {
            "positive": ["3", 0],
            "negative": ["4", 0],
            "vae": ["1", 2],
            "pixels": ["5", 0],
            "mask": ["6", 0],
            "noise_mask": True,
        },
    }
    workflow[REPEAT_NODE]["inputs"]["samples"] = ["7", 2]
    del workflow["8"]
    sampler = workflow["9"]["inputs"]
    sampler["positive"] = ["7", 0]
    sampler["negative"] = ["7", 1]
    sampler["latent_image"] = [REPEAT_NODE, 0]


def build_sd15_text_workflow(
    settings: InpaintSettings, width: int, height: int
) -> Workflow:
    """Return the API-format graph: text to image on the SD checkpoint with the LoRA.

    The inpainting UNet is not used: it expects a masked image.
    """
    workflow = build_inpaint_workflow("", "", replace(settings, inpaint_model=""))
    for node in ("5", "6", "7", "8", REPEAT_NODE):
        del workflow[node]
    workflow["7"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": settings.batch_size,
        },
    }
    sampler = workflow["9"]["inputs"]
    sampler["latent_image"] = ["7", 0]
    sampler["denoise"] = 1.0
    return workflow


@dataclass(frozen=True)
class ModelProfile:
    """Base model: its ComfyUI graphs and its default settings.

    build_inpaint rewrites the masked area of an uploaded image of the given
    size, build_text draws from the prompt alone at the given size and
    build_reference, when the model has one, draws at the given size from
    the prompt and an uploaded reference image. build_guided, when the model
    has one, redraws an uploaded guide at the given size, with the uploaded
    reference image for the look.
    """

    name: str
    settings: InpaintSettings
    build_inpaint: WorkflowBuilder
    build_text: TextWorkflowBuilder
    build_reference: ReferenceWorkflowBuilder | None = None
    build_guided: GuidedWorkflowBuilder | None = None


SD15_INPAINT = ModelProfile(
    name="sd15-inpaint",
    settings=InpaintSettings(prompt="", seed=0),
    build_inpaint=build_inpaint_workflow,
    build_text=build_sd15_text_workflow,
)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ComfyUIClient:
    """Minimal client for the upload, prompt, history, view and cancel endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        session: requests.Session | None = None,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
        *,
        model: ModelProfile = SD15_INPAINT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._session = session or requests.Session()
        self._timeout = timeout
        self._poll_interval = poll_interval

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> list[Image.Image]:
        """Inpaint the masked area, return the decoded images in batch order."""
        image_name = self._upload(image, "aseprite-ai-inpaint-input.png")
        mask_name = self._upload(mask, "aseprite-ai-inpaint-mask.png")
        workflow = self._model.build_inpaint(
            image_name, mask_name, settings, image.size
        )
        return self._run(workflow)

    def generate_text(
        self, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """Draw from the prompt alone, return the decoded images in batch order."""
        return self._run(self._model.build_text(settings, width, height))

    def generate_reference(
        self, reference: Image.Image, settings: InpaintSettings, width: int, height: int
    ) -> list[Image.Image]:
        """Draw from the prompt and the reference image, in batch order."""
        build = self._model.build_reference
        if build is None:
            raise ComfyUIError(f"{self._model.name} cannot draw from a reference image")
        name = self._upload(reference, "aseprite-ai-inpaint-reference.png")
        return self._run(build(name, settings, width, height))

    def generate_guided(
        self,
        reference: Image.Image,
        guide: Image.Image,
        settings: InpaintSettings,
    ) -> list[Image.Image]:
        """Redraw the guide, with the reference for the look, in batch order."""
        build = self._model.build_guided
        if build is None:
            raise ComfyUIError(f"{self._model.name} cannot redraw a guide")
        name = self._upload(reference, "aseprite-ai-inpaint-reference.png")
        guide_name = self._upload(guide, "aseprite-ai-inpaint-guide.png")
        return self._run(build(name, guide_name, settings, *guide.size))

    def _run(self, workflow: Workflow) -> list[Image.Image]:
        prompt_id = self._queue(workflow)
        try:
            entry = self._wait(prompt_id)
        except BaseException:
            self.cancel(prompt_id)
            raise
        return self._download_all(entry)

    def cancel(self, prompt_id: str) -> None:
        """Drop the prompt from the queue, or stop it if it is running."""
        try:
            self._session.post(
                f"{self._base_url}/queue", json={"delete": [prompt_id]}, timeout=5
            )
            self._session.post(
                f"{self._base_url}/interrupt", json={"prompt_id": prompt_id}, timeout=5
            )
        except requests.RequestException:
            pass

    def _upload(self, image: Image.Image, name: str) -> str:
        response = self._session.post(
            f"{self._base_url}/upload/image",
            files={"image": (name, _png_bytes(image), "image/png")},
            data={"overwrite": "true", "type": "input"},
            timeout=30,
        )
        response.raise_for_status()
        return str(response.json()["name"])

    def _queue(self, workflow: Workflow) -> str:
        response = self._session.post(
            f"{self._base_url}/prompt", json={"prompt": workflow}, timeout=30
        )
        if response.status_code != 200:
            raise ComfyUIError(f"workflow rejected: {response.text}")
        return str(response.json()["prompt_id"])

    def _wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            response = self._session.get(
                f"{self._base_url}/history/{prompt_id}", timeout=30
            )
            response.raise_for_status()
            entry: dict[str, Any] | None = response.json().get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError(f"generation failed: {status.get('messages')}")
                if status.get("completed"):
                    return entry
            time.sleep(self._poll_interval)
        raise ComfyUIError(f"no result after {self._timeout:.0f} s")

    def _download_all(self, entry: dict[str, Any]) -> list[Image.Image]:
        refs: list[dict[str, str]] = (
            entry.get("outputs", {}).get(OUTPUT_NODE, {}).get("images", [])
        )
        if not refs:
            raise ComfyUIError("the workflow produced no image")
        return [self._download(ref) for ref in refs]

    def _download(self, ref: dict[str, str]) -> Image.Image:
        response = self._session.get(
            f"{self._base_url}/view",
            params={
                "filename": ref["filename"],
                "subfolder": ref.get("subfolder", ""),
                "type": ref.get("type", "temp"),
            },
            timeout=30,
        )
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as result:
            return result.convert("RGB")
