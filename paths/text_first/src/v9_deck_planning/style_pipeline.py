from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .apimart import DEFAULT_API_ROOT, ApiMartClient
from .json_utils import read_json, write_json, write_text
from .models import DeckPlan, UserInput
from .style_image_backend import (
    CANVAS,
    DEFAULT_IMAGE_MODEL,
    ImageTaskClient,
    generate_background_image,
    image_model_family,
    normalize_background,
)


def main(*, project_root: Path) -> None:
    args = parse_args(project_root)
    output_dir = args.output_dir or project_root / "outputs" / f"style_background_{timestamp()}"
    manifest = run_style_asset_generation(
        project_root=project_root,
        source_dir=args.deck_plan_dir,
        output_dir=output_dir,
        image_model=args.image_model,
        api_root=args.api_root,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        poll_initial_delay_seconds=args.poll_initial_delay_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        poll_timeout_seconds=args.poll_timeout_seconds,
        size=args.size,
        resolution=args.resolution,
        quality=args.quality,
        mock_images=args.mock_images,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "succeeded":
        raise SystemExit(1)


def run_style_asset_generation(
    *,
    project_root: Path,
    source_dir: Path,
    output_dir: Path,
    image_model: str = DEFAULT_IMAGE_MODEL,
    api_root: str = DEFAULT_API_ROOT,
    api_key_env: str = "APIMART_API_KEY",
    timeout_seconds: float = 240.0,
    poll_initial_delay_seconds: float = 10.0,
    poll_interval_seconds: float = 5.0,
    poll_timeout_seconds: float = 1200.0,
    size: str = "1536x864",
    resolution: str = "1k",
    quality: str = "medium",
    mock_images: bool = False,
    client: ImageTaskClient | None = None,
) -> dict[str, Any]:
    """Generate only the deck-wide background and style metadata.

    Per-slide visuals and raster holders are generated together after the final
    HTML layout exists. Step 2 deliberately does not build or segment an asset
    atlas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    background_prompt_path = output_dir / "background_prompt.txt"
    canonical_background_path = output_dir / "canonical_background.png"
    design_system_path = output_dir / "design_system.json"
    started_at = now_iso()
    image_call_count = 0
    background_result: dict[str, Any] | None = None
    background_validation: dict[str, Any] | None = None
    status = "failed"
    error: str | None = None
    warnings: list[str] = []

    try:
        model_family = image_model_family(image_model)
        user_input, deck_plan = load_step_one_artifacts(source_dir)
        write_json(output_dir / "input.json", user_input)
        write_json(output_dir / "deck_plan.json", deck_plan)
        background_prompt = build_background_prompt(user_input, deck_plan)
        write_text(background_prompt_path, background_prompt)

        active_client = client
        if not mock_images and active_client is None:
            active_client = ApiMartClient(
                api_root=api_root,
                api_key_env=api_key_env,
                env_path=project_root / ".env",
                timeout_seconds=timeout_seconds,
                poll_initial_delay_seconds=poll_initial_delay_seconds,
                poll_interval_seconds=poll_interval_seconds,
                poll_timeout_seconds=poll_timeout_seconds,
            )
        calls_before = active_client.image_call_count if active_client is not None else 0
        try:
            background_result = generate_background_image(
                prompt=background_prompt,
                prompt_path=background_prompt_path,
                output_dir=output_dir,
                expected_size=CANVAS,
                image_model=image_model,
                size=size,
                resolution=resolution,
                quality=quality,
                mock_images=mock_images,
                client=active_client,
            )
        finally:
            if active_client is not None:
                image_call_count = active_client.image_call_count - calls_before

        raw_path = Path(background_result["raw_image"])
        with Image.open(raw_path) as raw:
            background_validation = {
                "raw_size": [raw.width, raw.height],
                "raw_mode": raw.mode,
                "canonical_size": list(CANVAS),
                "canonical_mode": "RGB",
                "operation": "validated_and_copied_without_resize_or_crop",
            }
            if raw.size != CANVAS:
                background_validation["operation"] = "normalized_to_canonical_canvas"
                warnings.append(f"background dimensions {raw.size} were normalized to {CANVAS}")
        normalize_background(raw_path, canonical_background_path)
        write_json(
            design_system_path,
            {
                "schema_version": "2.0",
                "design_system_id": "deck_style_01",
                "source_style": user_input.style,
                "style_prompt": user_input.style,
                "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
                "renderer_capabilities": [
                    "direct_text",
                    "generated_raster_holder",
                    "independent_visual_regions",
                    "editable_text_overlay",
                ],
                "generation_references": [
                    {
                        "reference_id": "background_main",
                        "path": "canonical_background.png",
                        "role": "deck_wide_canonical_background",
                    }
                ],
            },
        )
        status = "succeeded"
    except Exception as exc:
        model_family = None
        error = str(exc)

    manifest = {
        "stage": "style_background_generation",
        "status": status,
        "quality_status": "degraded" if warnings else "mock" if status == "succeeded" and mock_images else "strict_pass",
        "warnings": warnings,
        "error": error,
        "source_deck_plan_dir": str(source_dir),
        "image_model": image_model,
        "image_model_family": model_family,
        "model_call_count": 0,
        "image_call_count": image_call_count,
        "total_external_call_count": image_call_count,
        "mock_images": mock_images,
        "started_at": started_at,
        "finished_at": now_iso(),
        "output_dir": str(output_dir),
        "background_validation": background_validation,
        "generation_order": ["canonical_background"],
        "asset_atlas_generated": False,
        "segmentation_executed": False,
        "image_calls": {
            "canonical_background": _image_call_record(
                background_result, background_prompt_path, canonical_background_path
            )
        },
        "artifacts": {
            "input": _existing(output_dir / "input.json"),
            "deck_plan": _existing(output_dir / "deck_plan.json"),
            "background_prompt": _existing(background_prompt_path),
            "background_raw": (
                background_result.get("raw_image") if background_result else None
            ),
            "canonical_background": _existing(canonical_background_path),
            "design_system": _existing(design_system_path),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def load_step_one_artifacts(source_dir: Path) -> tuple[UserInput, DeckPlan]:
    required = {
        "manifest": source_dir / "run_manifest.json",
        "input": source_dir / "input.json",
        "deck_plan": source_dir / "deck_plan.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"step 1 artifact is missing: {path}")
    source_manifest = read_json(required["manifest"])
    if source_manifest.get("stage") != "deck_level_planning" or source_manifest.get("status") != "succeeded":
        raise ValueError("style generation requires a succeeded deck_level_planning manifest")
    user_input = UserInput.model_validate(read_json(required["input"]))
    deck_plan = DeckPlan.model_validate(read_json(required["deck_plan"]))
    if deck_plan.style != user_input.style:
        raise ValueError("step 1 deck plan did not preserve the exact input style")
    return user_input, deck_plan


def build_background_prompt(user_input: UserInput, deck_plan: DeckPlan) -> str:
    if deck_plan.style != user_input.style:
        raise ValueError("deck plan style must match the source input style")
    return f"""Create one neat, reusable 16:9 background for an entire presentation deck.

Match only the visual language of this style -- palette, materials, line work, lighting, and surface treatment:
{user_input.style}

Deck topic, as a weak atmospheric hint only:
{deck_plan.topic}

This is a background, not an illustration. Keep at least 90 percent of the full canvas as one broad, contiguous,
low-detail negative-space field suitable for later text and teaching visuals. All decoration combined must occupy
no more than 10 percent of the canvas. Keep it at the thin outer perimeter, leaving quiet space across the whole canvas.
Use at most two small abstract edge accents. Do not draw characters, people, rooms, neighborhoods, gadgets, doors,
vehicles, scenes, or other recognizable objects. Any decoration must remain sparse, peripheral, and subordinate.
Keep the composition clean, orderly, quiet, and spacious, with restrained decoration and broad low-detail areas.
Keep the background largely free of semantic information. Do not depict topic-specific objects, teaching entities,
diagrams, charts, data, or semantic relationships.
Do not include text, labels, slide-specific content, fixed cards, panels, placeholders, logos, or watermarks.

Return only the background image.
"""


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: generate the canonical deck background without an asset atlas."
    )
    parser.add_argument("--deck-plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--api-root", default=DEFAULT_API_ROOT)
    parser.add_argument("--api-key-env", default="APIMART_API_KEY")
    parser.add_argument("--size", default="1536x864")
    parser.add_argument("--resolution", default="1k")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--poll-initial-delay-seconds", type=float, default=10.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--mock-images", action="store_true")
    args = parser.parse_args()
    if not args.deck_plan_dir.is_absolute():
        args.deck_plan_dir = project_root / args.deck_plan_dir
    if args.output_dir is not None and not args.output_dir.is_absolute():
        args.output_dir = project_root / args.output_dir
    return args


def _image_call_record(
    result: dict[str, Any] | None, prompt_path: Path, output_path: Path
) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.get("status"),
        "prompt": str(prompt_path),
        "request": result.get("request"),
        "task": result.get("task"),
        "output": _existing(output_path),
    }


def _existing(path: Path | None) -> str | None:
    return str(path) if path is not None and path.exists() else None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
