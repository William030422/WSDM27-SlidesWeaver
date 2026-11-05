from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .apimart import DEFAULT_API_ROOT, ApiMartClient
from .json_utils import read_json, write_json, write_text
from .models import DeckPlan, UserInput
from .green_background_removal import extract_assets
from .segmented_asset_naming import name_segmented_assets
from .style_assets import (
    CANVAS,
    build_asset_manifest,
    build_design_system,
    normalize_background,
    normalize_style_reference,
    render_segmented_reference,
)
from .style_image_backend import (
    DEFAULT_IMAGE_MODEL,
    ImageTaskClient,
    generate_background_image,
    generate_style_reference_image,
    image_model_family,
)
from .style_reference import (
    build_background_prompt,
    build_style_reference_plan,
    build_style_reference_prompt,
)
from .style_models import StyleReferencePlan
from .style_rendering import render_style_reference_controls


POST_GENERATION_ORDER = [
    "segment_assets_and_write_contact_sheet",
    "remove_enclosed_background_from_all_assets",
    "compose_style_reference_from_postprocessed_assets",
]


def main(*, project_root: Path) -> None:
    args = parse_args(project_root)
    output_dir = args.output_dir or project_root / "outputs" / f"style_assets_{timestamp()}"
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
    manifest_path = output_dir / "run_manifest.json"
    print(json.dumps(summary(manifest, manifest_path), ensure_ascii=False, indent=2))
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
    size: str = "16:9",
    resolution: str = "1k",
    quality: str = "medium",
    mock_images: bool = False,
    client: ImageTaskClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    plan_path = output_dir / "style_reference_plan.json"
    prompt_path = output_dir / "style_reference_prompt.txt"
    background_prompt_path = output_dir / "background_prompt.txt"
    canonical_background_path = output_dir / "canonical_background.png"
    working_path = output_dir / "style_reference_working.png"
    asset_manifest_path = output_dir / "asset_manifest.json"
    design_system_path = output_dir / "design_system.json"
    segmentation_dir = output_dir / "segmentation"
    segmentation_manifest_path = segmentation_dir / "manifest.json"
    assets_dir = output_dir / "assets"
    naming_manifest_path = assets_dir / "manifest.json"
    started_at = now_iso()
    image_call_count = 0
    error: str | None = None
    status = "failed"
    quality_status = "not_evaluated"
    image_result: dict[str, Any] | None = None
    background_result: dict[str, Any] | None = None
    board_validation: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    background_validation: dict[str, Any] | None = None
    control_paths: dict[str, Path] = {}
    model_family: str | None = None
    postprocessing_warning: str | None = None

    try:
        model_family = image_model_family(image_model)
        user_input, deck_plan = load_step_one_artifacts(source_dir)
        write_json(output_dir / "input.json", user_input)
        write_json(output_dir / "deck_plan.json", deck_plan)

        plan = build_style_reference_plan(user_input, deck_plan)
        if plan.board_size != CANVAS:
            raise ValueError(f"style-reference board must use native canvas {CANVAS}")
        write_json(plan_path, plan)
        prompt = build_style_reference_prompt(user_input, deck_plan, plan, image_model=image_model)
        write_text(prompt_path, prompt)
        background_prompt = build_background_prompt(user_input, deck_plan)
        write_text(background_prompt_path, background_prompt)
        control_paths = render_style_reference_controls(plan, output_dir)

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

            background_raw_path = Path(background_result["raw_image"])
            with Image.open(background_raw_path) as background_raw:
                background_validation = {
                    "raw_size": [background_raw.width, background_raw.height],
                    "raw_mode": background_raw.mode,
                    "canonical_size": list(CANVAS),
                    "canonical_mode": "RGB",
                    "operation": "validated_and_copied_without_resize_or_crop",
                }
            normalize_background(background_raw_path, canonical_background_path)

            image_result = generate_style_reference_image(
                prompt=prompt,
                prompt_path=prompt_path,
                control_path=control_paths["control"],
                mask_path=control_paths["mask"],
                background_path=canonical_background_path,
                plan=plan,
                output_dir=output_dir,
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

        try:
            processed = process_generated_style_assets(
                raw_path=Path(image_result["raw_image"]),
                output_dir=output_dir,
                plan=plan,
            )
            normalization = processed["normalization"]
            segmentation = processed["segmentation"]
            naming = processed["naming"]
            asset_manifest = processed["asset_manifest"]
            board_validation = _build_board_validation(plan, segmentation, naming, asset_manifest)
            quality_status = "strict_pass" if board_validation["complete"] else "degraded"
        except Exception as exc:
            postprocessing_warning = str(exc)
            shutil.copyfile(canonical_background_path, output_dir / "style_reference.png")
            asset_manifest = {
                "schema_version": "1.1",
                "stage": "style_asset_manifest",
                "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
                "assets": [],
            }
            write_json(asset_manifest_path, asset_manifest)
            write_json(design_system_path, build_design_system(plan, asset_manifest))
            normalization = {
                "operation": "fallback_to_canonical_background",
                "error": postprocessing_warning,
            }
            board_validation = {
                "passed": True,
                "complete": False,
                "matching_status": "unavailable",
                "policy": "Style assets are optional; continue with direct text and native shapes.",
            }
            quality_status = "degraded"
        status = "succeeded"
    except Exception as exc:
        error = str(exc)

    manifest = {
        "stage": "style_asset_generation",
        "status": status,
        "build_status": status,
        "quality_status": quality_status,
        "error": error,
        "warnings": [postprocessing_warning] if postprocessing_warning else [],
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
        "board_validation": board_validation,
        "normalization": normalization,
        "background_validation": background_validation,
        "generation_order": ["canonical_background", "style_asset_atlas"],
        "post_generation_order": POST_GENERATION_ORDER,
        "style_asset_increment": {
            "added_reference": "canonical_background.png",
            "original_control_and_mask_logic_preserved": True,
        },
        "image_calls": {
            "style_asset_atlas": _image_call_record(
                image_result,
                prompt_path,
                output_dir / "style_reference.png",
            ),
            "canonical_background": _image_call_record(
                background_result,
                background_prompt_path,
                canonical_background_path,
            ),
        },
        "raster_contract": {
            "canonical_size": list(CANVAS),
            "normalization": "validate_and_copy_without_resize_or_crop",
            "control": _image_metadata(control_paths.get("control")),
            "mask": _image_metadata(control_paths.get("mask"), include_alpha=True),
            "guide": _image_metadata(control_paths.get("guide")),
            "raw_image": _image_metadata(Path(image_result["raw_image"])) if image_result else None,
            "working_atlas": _image_metadata(working_path),
            "style_reference": _image_metadata(output_dir / "style_reference.png"),
            "background_raw": _image_metadata(
                Path(background_result["raw_image"]) if background_result else None
            ),
            "canonical_background": _image_metadata(canonical_background_path),
        },
        "artifacts": {
            "source_manifest": str(source_dir / "run_manifest.json"),
            "input": _existing(output_dir / "input.json"),
            "deck_plan": _existing(output_dir / "deck_plan.json"),
            "style_reference_plan": _existing(plan_path),
            "style_reference_prompt": _existing(prompt_path),
            "background_prompt": _existing(background_prompt_path),
            "control": _existing(control_paths.get("control")),
            "mask": _existing(control_paths.get("mask")),
            "guide": _existing(control_paths.get("guide")),
            "raw_image": image_result.get("raw_image") if image_result else None,
            "working_atlas": _existing(working_path),
            "style_reference": _existing(output_dir / "style_reference.png"),
            "background_raw": (
                background_result.get("raw_image") if background_result else _existing(output_dir / "background_raw.png")
            ),
            "canonical_background": _existing(canonical_background_path),
            "request": image_result.get("request") if image_result else _existing(output_dir / "style_reference.request.json"),
            "task": image_result.get("task") if image_result else _existing(output_dir / "style_reference.task.json"),
            "background_request": (
                background_result.get("request")
                if background_result
                else _existing(output_dir / "background.request.json")
            ),
            "background_task": (
                background_result.get("task")
                if background_result
                else _existing(output_dir / "background.task.json")
            ),
            "segmentation_manifest": _existing(segmentation_manifest_path),
            "segmentation_background_mask": _existing(segmentation_dir / "background_mask.png"),
            "segmentation_foreground_mask": _existing(segmentation_dir / "foreground_mask.png"),
            "segmentation_contact_sheet": _existing(segmentation_dir / "contact_sheet.png"),
            "naming_manifest": _existing(naming_manifest_path),
            "asset_manifest": _existing(asset_manifest_path),
            "design_system": _existing(design_system_path),
            "assets_dir": str(assets_dir) if assets_dir.exists() else None,
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def reprocess_generated_style_assets(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    plan_path = output_dir / "style_reference_plan.json"
    raw_path = output_dir / "style_reference_raw.png"
    for path in (manifest_path, plan_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"Step 2 reprocessing input is missing: {path}")

    plan = StyleReferencePlan.model_validate(read_json(plan_path))
    processed = process_generated_style_assets(
        raw_path=raw_path,
        output_dir=output_dir,
        plan=plan,
    )
    board_validation = _build_board_validation(
        plan,
        processed["segmentation"],
        processed["naming"],
        processed["asset_manifest"],
    )

    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "succeeded",
            "build_status": "succeeded",
            "quality_status": "strict_pass" if board_validation["complete"] else "degraded",
            "error": None,
            "finished_at": now_iso(),
            "board_validation": board_validation,
            "normalization": processed["normalization"],
            "post_generation_order": POST_GENERATION_ORDER,
            "reprocessing": {
                "source_raw_image": str(raw_path.resolve()),
                "external_call_count": 0,
                "finished_at": now_iso(),
            },
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def process_generated_style_assets(
    *,
    raw_path: Path,
    output_dir: Path,
    plan: StyleReferencePlan,
) -> dict[str, Any]:
    working_path = output_dir / "style_reference_working.png"
    reference_path = output_dir / "style_reference.png"
    plan_path = output_dir / "style_reference_plan.json"
    segmentation_dir = output_dir / "segmentation"
    assets_dir = output_dir / "assets"
    asset_manifest_path = output_dir / "asset_manifest.json"
    design_system_path = output_dir / "design_system.json"

    if not plan_path.is_file():
        raise FileNotFoundError(f"style reference plan is missing: {plan_path}")

    with Image.open(raw_path) as raw_image:
        normalization = {
            "raw_size": [raw_image.width, raw_image.height],
            "raw_mode": raw_image.mode,
            "working_size": list(CANVAS),
            "working_mode": "RGB",
            "operation": "validated_and_copied_without_resize_or_crop",
        }

    _clear_generated_asset_outputs(
        segmentation_dir=segmentation_dir,
        assets_dir=assets_dir,
        derived_files=(reference_path, asset_manifest_path, design_system_path),
    )
    normalize_style_reference(raw_path, working_path)

    # The contact sheet records the segmentation result before enclosed
    # background removal mutates the canonical files in assets/.
    segmentation = extract_assets(
        working_path,
        segmentation_dir,
        assets_dir=assets_dir,
        slots=[slot.model_dump(mode="json") for slot in plan.slots if slot.enabled],
    )
    naming = name_segmented_assets(
        segmentation_dir / "manifest.json",
        plan_path,
        assets_dir,
    )
    render_segmented_reference(segmentation, segmentation_dir, reference_path)

    with Image.open(reference_path) as reference:
        if reference.mode != "RGB" or reference.size != CANVAS:
            raise ValueError(
                f"canonical style_reference.png must be RGB {CANVAS}, got {reference.mode} {reference.size}"
            )

    matched_slot_ids = {
        str(record["asset_id"])
        for record in naming["assets"]
        if record["status"] == "matched"
    }
    matched_plan = plan.model_copy(
        update={"slots": [slot for slot in plan.slots if slot.slot_id in matched_slot_ids]}
    )
    asset_manifest = build_asset_manifest(naming, matched_plan, assets_dir)
    design_system = build_design_system(plan, asset_manifest)
    write_json(asset_manifest_path, asset_manifest)
    write_json(design_system_path, design_system)
    return {
        "normalization": normalization,
        "segmentation": segmentation,
        "naming": naming,
        "asset_manifest": asset_manifest,
        "design_system": design_system,
    }


def _build_board_validation(
    plan: StyleReferencePlan,
    segmentation: dict[str, Any],
    naming: dict[str, Any],
    asset_manifest: dict[str, Any],
) -> dict[str, Any]:
    available = {asset["asset_id"] for asset in asset_manifest["assets"] if asset["available"]}
    complete = all(not slot.required or slot.slot_id in available for slot in plan.slots)
    return {
        "passed": True,
        "complete": complete,
        "matching_status": "complete" if complete else "partial",
        "validation_mode": "partial_global_slot_matching",
        "segmentation_algorithm": segmentation["algorithm"],
        "naming_algorithm": naming["algorithm"],
        "island_count": naming["island_count"],
        "fragment_count": naming["fragment_count"],
        "matched_count": naming["matched_count"],
        "unmatched_island_count": naming["unmatched_island_count"],
        "unmatched_fragment_count": naming["unmatched_fragment_count"],
        "missing_slot_count": naming["missing_slot_count"],
        "policy": "Matched assets remain available; missing or misaligned slots are recorded without stopping the pipeline.",
    }


def _clear_generated_asset_outputs(
    *,
    segmentation_dir: Path,
    assets_dir: Path,
    derived_files: tuple[Path, ...],
) -> None:
    for directory in (segmentation_dir, assets_dir):
        if directory.exists():
            shutil.rmtree(directory)
    for path in derived_files:
        path.unlink(missing_ok=True)


def load_step_one_artifacts(source_dir: Path) -> tuple[UserInput, DeckPlan]:
    manifest_path = source_dir / "run_manifest.json"
    input_path = source_dir / "input.json"
    deck_plan_path = source_dir / "deck_plan.json"
    for path in (manifest_path, input_path, deck_plan_path):
        if not path.is_file():
            raise FileNotFoundError(f"step 1 artifact is missing: {path}")
    source_manifest = read_json(manifest_path)
    if source_manifest.get("stage") != "deck_level_planning" or source_manifest.get("status") != "succeeded":
        raise ValueError("style generation requires a succeeded deck_level_planning manifest")
    user_input = UserInput.model_validate(read_json(input_path))
    deck_plan = DeckPlan.model_validate(read_json(deck_plan_path))
    if deck_plan.style != user_input.style:
        raise ValueError("step 1 deck plan did not preserve the exact input style")
    return user_input, deck_plan


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V9 step 2: generate a reusable style-reference atlas and canonical background."
    )
    parser.add_argument("--deck-plan-dir", type=Path, required=True, help="Succeeded V9 step-1 output directory.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--api-root", default=DEFAULT_API_ROOT)
    parser.add_argument("--api-key-env", default="APIMART_API_KEY")
    parser.add_argument("--size", default="16:9")
    parser.add_argument("--resolution", default="1k")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--poll-initial-delay-seconds", type=float, default=10.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--mock-images",
        action="store_true",
        help="Render a deterministic local atlas and background without API calls.",
    )
    args = parser.parse_args()
    if not args.deck_plan_dir.is_absolute():
        args.deck_plan_dir = project_root / args.deck_plan_dir
    if args.output_dir is not None and not args.output_dir.is_absolute():
        args.output_dir = project_root / args.output_dir
    return args


def summary(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") or {}
    return {
        "ok": manifest.get("status") == "succeeded",
        "status": manifest.get("status"),
        "quality_status": manifest.get("quality_status"),
        "image_call_count": manifest.get("image_call_count"),
        "output_dir": manifest.get("output_dir"),
        "style_reference": artifacts.get("style_reference"),
        "canonical_background": artifacts.get("canonical_background"),
        "asset_manifest": artifacts.get("asset_manifest"),
        "design_system": artifacts.get("design_system"),
        "manifest": str(manifest_path),
        "error": manifest.get("error"),
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _existing(path: Path | None) -> str | None:
    return str(path) if path is not None and path.exists() else None


def _image_metadata(
    path: Path | None,
    *,
    include_alpha: bool = False,
) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        metadata: dict[str, Any] = {
            "path": str(path),
            "mode": image.mode,
            "size": list(image.size),
        }
        if include_alpha and "A" in image.getbands():
            metadata["alpha_unique_values"] = sorted(
                value for _, value in image.getchannel("A").getcolors(maxcolors=3) or []
            )
        return metadata


def _image_call_record(
    result: dict[str, Any] | None,
    prompt_path: Path,
    canonical_path: Path,
) -> dict[str, Any]:
    return {
        "status": result.get("status") if result else None,
        "model": result.get("model") if result else None,
        "prompt": _existing(prompt_path),
        "request": result.get("request") if result else None,
        "task": result.get("task") if result else None,
        "raw_image": result.get("raw_image") if result else None,
        "canonical_image": _existing(canonical_path),
    }
