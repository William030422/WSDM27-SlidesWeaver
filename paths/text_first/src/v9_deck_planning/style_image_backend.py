from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from .apimart import redacted_image_payload
from .image_task_journal import run_persisted_image_task
from .json_utils import write_json

CANVAS = (1536, 864)
DEFAULT_IMAGE_MODEL = "gpt-image-2-official"
NANO_BANANA_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image-preview-official",
}


class ImageTaskClient(Protocol):
    image_call_count: int

    def submit_image_task(self, payload: dict[str, Any]) -> str: ...

    def poll_task(self, task_id: str) -> dict[str, Any]: ...

    def download(self, url: str, output_path: Path) -> None: ...


def generate_background_image(
    *,
    prompt: str,
    prompt_path: Path,
    output_dir: Path,
    expected_size: tuple[int, int],
    image_model: str = DEFAULT_IMAGE_MODEL,
    size: str = "1536x864",
    resolution: str = "1k",
    quality: str = "medium",
    mock_images: bool = False,
    client: ImageTaskClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "background_raw.png"
    request_path = output_dir / "background.request.json"
    task_path = output_dir / "background.task.json"
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
            "model_family": image_model_family(image_model),
            "payload": redacted_image_payload(payload),
            "prompt_character_count": len(prompt),
            "prompt_path": str(prompt_path),
            "transport": "independent text-to-image generation",
            "data_uris_redacted": True,
        },
    )
    if mock_images:
        render_mock_background(raw_path, expected_size)
        write_json(task_path, {"status": "mock", "external_image_calls": 0})
        return _result("mock", image_model, raw_path, request_path, task_path, None)
    if client is None:
        raise RuntimeError(
            "background image generation needs an APIMART client unless --mock-images is used"
        )
    generated = run_persisted_image_task(client=client, payload=payload, task_path=task_path,
                                         output_path=raw_path, max_attempts=2)
    image_url = generated["image_url"]
    return _result(
        "generated", image_model, raw_path, request_path, task_path, image_url
    )


def build_background_image_payload(
    *, prompt: str, image_model: str, size: str, resolution: str, quality: str
) -> dict[str, Any]:
    family = image_model_family(image_model)
    payload: dict[str, Any] = {
        "model": image_model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    normalized_resolution = resolution.upper()
    if family == "gpt_image":
        payload.update(
            {"resolution": resolution, "quality": quality, "output_format": "png"}
        )
    elif family == "nano_banana":
        if normalized_resolution not in {"0.5K", "1K", "2K", "4K"}:
            raise ValueError(
                f"Nano Banana resolution must be 0.5K, 1K, 2K, or 4K: {resolution}"
            )
        payload["resolution"] = normalized_resolution
    else:
        if len(prompt) > 800:
            raise ValueError(
                f"qwen-image-2.0 prompt exceeds its 800-character limit: {len(prompt)}"
            )
        if normalized_resolution not in {"1K", "2K"}:
            raise ValueError(f"Qwen Image resolution must be 1K or 2K: {resolution}")
        payload.update(
            {
                "resolution": normalized_resolution,
                "negative_prompt": "readable text, cards, panels, placeholders, logos, watermarks",
            }
        )
    return payload


def image_model_family(image_model: str) -> str:
    if image_model.startswith("gpt-image-"):
        return "gpt_image"
    if image_model in NANO_BANANA_MODELS:
        return "nano_banana"
    if image_model.startswith("qwen-image-2.0"):
        return "qwen"
    raise ValueError(f"unsupported background image model: {image_model}")


def normalize_background(source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        if image.size != CANVAS:
            from PIL import ImageOps

            # The canonical canvas remains authoritative after a valid image response.
            rgb = ImageOps.fit(rgb, CANVAS, method=Image.Resampling.LANCZOS)
        rgb.save(output_path)


def render_mock_background(path: Path, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (226, 221, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, round(size[0] * 0.035), size[1]), fill=(66, 83, 67))
    draw.rectangle(
        (round(size[0] * 0.91), round(size[1] * 0.92), size[0], size[1]),
        fill=(151, 105, 46),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _result(
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
