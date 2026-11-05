from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .apimart import image_to_data_uri, redacted_image_payload
from .image_task_journal import run_persisted_image_task
from .json_utils import write_json
from .style_models import StyleReferencePlan
from .style_rendering import render_mock_background, render_mock_style_reference


DEFAULT_IMAGE_MODEL = "gpt-image-2-official"
NANO_BANANA_MODELS = {"gemini-3.1-flash-image-preview", "gemini-3.1-flash-image-preview-official"}


class ImageTaskClient(Protocol):
    image_call_count: int

    def submit_image_task(self, payload: dict[str, Any]) -> str: ...

    def poll_task(self, task_id: str) -> dict[str, Any]: ...

    def download(self, url: str, output_path: Path) -> None: ...


def generate_style_reference_image(
    *,
    prompt: str,
    prompt_path: Path,
    control_path: Path,
    mask_path: Path,
    background_path: Path,
    plan: StyleReferencePlan,
    output_dir: Path,
    image_model: str = DEFAULT_IMAGE_MODEL,
    size: str = "16:9",
    resolution: str = "1k",
    quality: str = "medium",
    mock_images: bool = False,
    client: ImageTaskClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "style_reference_raw.png"
    request_path = output_dir / "style_reference.request.json"
    task_path = output_dir / "style_reference.task.json"
    validate_style_image_inputs(control_path, mask_path, background_path, plan.board_size)
    family = image_model_family(image_model)
    payload = build_style_image_payload(
        prompt=prompt,
        control_path=control_path,
        mask_path=mask_path,
        background_path=background_path,
        image_model=image_model,
        size=size,
        resolution=resolution,
        quality=quality,
    )
    transport = {
        "gpt_image_masked_edit": "base64 canonical background then base64 control image, plus the original base64 alpha mask",
        "nano_banana_reference": "base64 canonical background then base64 chroma-canvas reference image; no undocumented mask field",
        "qwen_text_to_image": "text-only; Qwen rejects base64 image_urls and no public reference URL was supplied",
    }[family]
    write_json(
        request_path,
        {
            "mock": mock_images,
            "model_family": family,
            "payload": redacted_image_payload(payload),
            "prompt_character_count": len(prompt),
            "prompt_path": str(prompt_path),
            "control_path": str(control_path) if family != "qwen_text_to_image" else None,
            "mask_path": str(mask_path) if family == "gpt_image_masked_edit" else None,
            "background_path": str(background_path) if family != "qwen_text_to_image" else None,
            "transport": transport,
            "previous_generated_image_used": family != "qwen_text_to_image",
            "incremental_reference_role": "canonical_background_style_only",
            "data_uris_redacted": True,
        },
    )

    if mock_images:
        render_mock_style_reference(raw_path, plan)
        write_json(task_path, {"status": "mock", "external_image_calls": 0})
        return _result(
            status="mock",
            image_model=image_model,
            raw_path=raw_path,
            request_path=request_path,
            task_path=task_path,
            image_url=None,
        )

    if client is None:
        raise RuntimeError("style-reference image generation needs an APIMART client unless --mock-images is used")

    generated = run_persisted_image_task(client=client, payload=payload, task_path=task_path, output_path=raw_path)
    image_url = generated["image_url"]
    return _result(
        status="generated",
        image_model=image_model,
        raw_path=raw_path,
        request_path=request_path,
        task_path=task_path,
        image_url=image_url,
    )


def generate_background_image(
    *,
    prompt: str,
    prompt_path: Path,
    output_dir: Path,
    expected_size: tuple[int, int],
    image_model: str = DEFAULT_IMAGE_MODEL,
    size: str = "16:9",
    resolution: str = "1k",
    quality: str = "medium",
    mock_images: bool = False,
    client: ImageTaskClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "background_raw.png"
    request_path = output_dir / "background.request.json"
    task_path = output_dir / "background.task.json"
    family = image_model_family(image_model)
    payload = build_background_image_payload(
        prompt=prompt,
        image_model=image_model,
        size=size,
        resolution=resolution,
        quality=quality,
    )
    write_json(
        request_path,
        {
            "mock": mock_images,
            "model_family": family,
            "payload": redacted_image_payload(payload),
            "prompt_character_count": len(prompt),
            "prompt_path": str(prompt_path),
            "transport": "independent text-to-image generation",
            "previous_generated_image_used": False,
            "data_uris_redacted": True,
        },
    )

    if mock_images:
        render_mock_background(raw_path, expected_size)
        write_json(task_path, {"status": "mock", "external_image_calls": 0})
        return _result(
            status="mock",
            image_model=image_model,
            raw_path=raw_path,
            request_path=request_path,
            task_path=task_path,
            image_url=None,
        )

    if client is None:
        raise RuntimeError("background image generation needs an APIMART client unless --mock-images is used")

    generated = run_persisted_image_task(client=client, payload=payload, task_path=task_path, output_path=raw_path)
    image_url = generated["image_url"]
    return _result(
        status="generated",
        image_model=image_model,
        raw_path=raw_path,
        request_path=request_path,
        task_path=task_path,
        image_url=image_url,
    )


def validate_style_image_inputs(
    control_path: Path,
    mask_path: Path,
    background_path: Path,
    expected_size: tuple[int, int],
) -> None:
    with Image.open(control_path) as control:
        if control.mode != "RGB" or control.size != expected_size:
            raise ValueError(
                f"style control image must be RGB {expected_size}, got {control.mode} {control.size}"
            )
        control_size = control.size
    with Image.open(mask_path) as mask:
        if mask.mode != "RGBA" or mask.size != expected_size:
            raise ValueError(
                f"style generation mask must be RGBA {expected_size}, got {mask.mode} {mask.size}"
            )
        if mask.size != control_size:
            raise ValueError("style control image and generation mask must have identical dimensions")
        alpha_values = {value for _, value in mask.getchannel("A").getcolors(maxcolors=3) or []}
        if alpha_values != {0, 255}:
            raise ValueError(
                f"style generation mask alpha values must be exactly {{0, 255}}, got {sorted(alpha_values)}"
            )
    with Image.open(background_path) as background:
        if background.mode != "RGB" or background.size != expected_size:
            raise ValueError(
                f"canonical background must be RGB {expected_size}, got {background.mode} {background.size}"
            )


def _result(
    *,
    status: str,
    image_model: str,
    raw_path: Path,
    request_path: Path,
    task_path: Path,
    image_url: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "model": image_model,
        "raw_image": str(raw_path),
        "request": str(request_path),
        "task": str(task_path),
        "image_url": image_url,
    }


def image_model_family(image_model: str) -> str:
    if image_model.startswith("gpt-image-"):
        return "gpt_image_masked_edit"
    if image_model in NANO_BANANA_MODELS:
        return "nano_banana_reference"
    if image_model.startswith("qwen-image-2.0"):
        return "qwen_text_to_image"
    raise ValueError(f"unsupported style-reference image model: {image_model}")


def build_style_image_payload(
    *,
    prompt: str,
    control_path: Path,
    mask_path: Path,
    background_path: Path,
    image_model: str,
    size: str,
    resolution: str,
    quality: str,
) -> dict[str, Any]:
    family = image_model_family(image_model)
    payload: dict[str, Any] = {
        "model": image_model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    if family == "gpt_image_masked_edit":
        payload.update(
            {
                "resolution": resolution,
                "quality": quality,
                "output_format": "png",
                "image_urls": [
                    image_to_data_uri(background_path),
                    image_to_data_uri(control_path),
                ],
                "mask_url": image_to_data_uri(mask_path),
            }
        )
        return payload

    normalized_resolution = resolution.upper()
    if family == "nano_banana_reference":
        if normalized_resolution not in {"0.5K", "1K", "2K", "4K"}:
            raise ValueError(f"Nano Banana resolution must be 0.5K, 1K, 2K, or 4K: {resolution}")
        payload.update(
            {
                "resolution": normalized_resolution,
                "image_urls": [
                    image_to_data_uri(background_path),
                    image_to_data_uri(control_path),
                ],
            }
        )
        return payload

    if len(prompt) > 800:
        raise ValueError(f"qwen-image-2.0 prompt exceeds its 800-character limit: {len(prompt)}")
    if normalized_resolution not in {"1K", "2K"}:
        raise ValueError(f"Qwen Image 2.0 resolution must be 1K or 2K: {resolution}")
    payload.update(
        {
            "resolution": normalized_resolution,
            "negative_prompt": (
                "readable text, letters, numbers, pseudo-text, captions, wordmarks, watermarks, uniform rounded-square "
                "icon tiles, repeated card frames, connected assets, non-chroma background"
            ),
        }
    )
    return payload


def build_background_image_payload(
    *,
    prompt: str,
    image_model: str,
    size: str,
    resolution: str,
    quality: str,
) -> dict[str, Any]:
    family = image_model_family(image_model)
    payload: dict[str, Any] = {
        "model": image_model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    if family == "gpt_image_masked_edit":
        payload.update(
            {
                "resolution": resolution,
                "quality": quality,
                "output_format": "png",
            }
        )
        return payload

    normalized_resolution = resolution.upper()
    if family == "nano_banana_reference":
        if normalized_resolution not in {"0.5K", "1K", "2K", "4K"}:
            raise ValueError(f"Nano Banana resolution must be 0.5K, 1K, 2K, or 4K: {resolution}")
        payload["resolution"] = normalized_resolution
        return payload

    if len(prompt) > 800:
        raise ValueError(f"qwen-image-2.0 prompt exceeds its 800-character limit: {len(prompt)}")
    if normalized_resolution not in {"1K", "2K"}:
        raise ValueError(f"Qwen Image 2.0 resolution must be 1K or 2K: {resolution}")
    payload.update(
        {
            "resolution": normalized_resolution,
            "negative_prompt": (
                "readable text, letters, numbers, labels, diagrams, charts, data, teaching entities, "
                "fixed cards, panels, placeholders, logos, watermarks"
            ),
        }
    )
    return payload
