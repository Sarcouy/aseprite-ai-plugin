"""Inpainting through the HTTP API of a local ComfyUI server."""

import io
import time
from dataclasses import dataclass
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
    cfg: float = 7.0
    negative_prompt: str = DEFAULT_NEGATIVE
    checkpoint: str = DEFAULT_CHECKPOINT
    lora: str = DEFAULT_LORA
    lora_strength: float = 1.0
    background: str = ""


def positive_prompt(settings: InpaintSettings) -> str:
    """LoRA trigger, user prompt and background hint, skipping empty parts."""
    parts = [LORA_TRIGGER, settings.prompt]
    if settings.background:
        parts.append(f"plain {settings.background} background")
    return ", ".join(part for part in parts if part)


def build_inpaint_workflow(
    image_name: str, mask_name: str, settings: InpaintSettings
) -> Workflow:
    """Return the API-format graph: masked img2img on an SD checkpoint with a LoRA."""
    positive = positive_prompt(settings)
    return {
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
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {
            "class_type": "LoadImageMask",
            "inputs": {"image": mask_name, "channel": "red"},
        },
        "7": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["7", 0], "mask": ["6", 0]},
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
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": settings.denoise,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
        },
        OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
    }


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ComfyUIClient:  # pylint: disable=too-few-public-methods  # implements the single-method InpaintGenerator protocol
    """Minimal client for the upload, prompt, history and view endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        session: requests.Session | None = None,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._poll_interval = poll_interval

    def generate(
        self, image: Image.Image, mask: Image.Image, settings: InpaintSettings
    ) -> Image.Image:
        """Run the inpainting workflow and return the decoded image."""
        image_name = self._upload(image, "aseprite-ai-inpaint-input.png")
        mask_name = self._upload(mask, "aseprite-ai-inpaint-mask.png")
        prompt_id = self._queue(build_inpaint_workflow(image_name, mask_name, settings))
        return self._download(self._wait(prompt_id))

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

    def _download(self, entry: dict[str, Any]) -> Image.Image:
        images = entry.get("outputs", {}).get(OUTPUT_NODE, {}).get("images", [])
        if not images:
            raise ComfyUIError("the workflow produced no image")
        ref = images[0]
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
