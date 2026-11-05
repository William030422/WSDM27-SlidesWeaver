#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from integration.config import load_config
from integration.page_pipeline import run_page_pipeline
from integration.page_runtime import RunLock, atomic_write_json
from integration.shared_checkpoint import checkpoint_shared_stage
from integration.diffusion import run_reused_diffusion
from integration.expression_logic import run_expression_logic_planning
from integration.final_background import (
    run_local_visual_block_backgrounds,
)
from integration.html_compile import run_html_to_coordinates
from integration.html_planning import run_html_planning
from integration.text_assembly import run_reused_text_assembly
from integration.text_generation import run_text_generation
from v9_deck_planning.json_utils import read_json, write_json
from v9_deck_planning.models import UserInput
from v9_deck_planning.pipeline import run_deck_planning
from v9_deck_planning.style_pipeline import run_style_asset_generation

STAGES = (
    "step1",
    "text_generation",
    "step2",
    "expression_logic",
    "diffusion",
    "html_planning",
    "html_compile",
    "background",
    "assembly",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the expression-logic HTML-planning presentation pipeline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--content")
    parser.add_argument("--style")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mock-api", action="store_true")
    parser.add_argument(
        "--mock-images",
        action="store_true",
        help="skip all image-generation API calls while keeping chat/VLM calls live",
    )
    parser.add_argument("--mock-diffusion", action="store_true")
    parser.add_argument("--page-workers", type=int, default=5,
                        help="persistent page owners (default: 5; use 1 for serial execution)")
    parser.add_argument("--gpu-devices", help="comma-separated GPU indices or UUIDs; defaults to visible devices")
    parser.add_argument(
        "--slide-limit",
        type=int,
        help="run downstream slide-specific stages for only the first N deck slides",
    )
    parser.add_argument("--start-at", choices=STAGES, default="step1")
    parser.add_argument("--stop-after", choices=STAGES, default="assembly")
    args = parser.parse_args()
    if args.content and not args.style:
        parser.error("--style is required with --content")
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        parser.error("--start-at must not come after --stop-after")
    if args.start_at != "step1" and args.output_dir is None:
        parser.error("--output-dir is required when resuming with --start-at")
    if args.slide_limit is not None and args.slide_limit < 1:
        parser.error("--slide-limit must be at least 1")
    if args.page_workers < 1:
        parser.error("--page-workers must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / _timestamp()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    with RunLock(output_dir):
        try:
            return _run(args, output_dir)
        except BaseException as exc:
            # A failed rerun must not leave an earlier successful status visible.
            previous = None
            try:
                previous = read_json(output_dir / "run_manifest.json")
            except (OSError, ValueError):
                pass
            atomic_write_json(output_dir / "run_manifest.json", {
                "pipeline": "expression_logic_html_planning", "status": "failed",
                "quality_status": "degraded", "error_type": type(exc).__name__,
                "failed_phase": getattr(exc, "phase", None), "finished_at": _now_iso(),
                "previous_manifest": previous,
            })
            raise


def _run(args: argparse.Namespace, output_dir: Path) -> int:
    config = load_config(PROJECT_ROOT)
    _validate_native_contract(config)
    user_input = _load_user_input(args)
    stages: list[dict] = []
    started_at = _now_iso()

    step1_dir = output_dir / "step1"
    if _should_run(args, "step1"):
        step1 = checkpoint_shared_stage(
            run_dir=output_dir, stage="step1", source_inputs=[],
            settings={"user_input": user_input.model_dump(mode="json"), "model": config.chat_model, "api_root": config.api_root, "max_tokens": config.chat_max_tokens, "mock_api": args.mock_api},
            fn=lambda: run_deck_planning(
                project_root=PROJECT_ROOT,
                user_input=user_input,
                output_dir=step1_dir,
                llm_model=config.chat_model,
                api_root=config.api_root,
                api_key_env=config.api_key_env,
                timeout_seconds=config.timeout_seconds,
                max_tokens=config.chat_max_tokens,
                mock_llm=args.mock_api,
            ),
        )
        _require_success(step1, "Step 1")
    else:
        step1 = _load_existing_stage(step1_dir, "Step 1")
    stages.append(step1)
    if args.stop_after == "step1":
        return _finish(output_dir, config, args, stages, started_at)

    text_generation_dir = output_dir / "text_generation"
    if _should_run(args, "text_generation"):
        text_generation = checkpoint_shared_stage(
            run_dir=output_dir, stage="text_generation", source_inputs=[step1_dir / "input.json", step1_dir / "deck_plan.json"],
            settings={"model": config.chat_model, "api_root": config.api_root, "max_tokens": config.chat_max_tokens, "mock_api": args.mock_api},
            fn=lambda: run_text_generation(
                project_root=PROJECT_ROOT,
                source_dir=step1_dir,
                output_dir=text_generation_dir,
                config=config,
                mock_llm=args.mock_api,
            ),
        )
        _require_success(text_generation, "Text Generation")
    else:
        text_generation = _load_existing_stage(text_generation_dir, "Text Generation")
    stages.append(text_generation)
    if args.stop_after == "text_generation":
        return _finish(output_dir, config, args, stages, started_at)

    step2_dir = output_dir / "step2"
    if _should_run(args, "step2"):
        step2 = checkpoint_shared_stage(
            run_dir=output_dir, stage="step2", source_inputs=[step1_dir / "input.json", step1_dir / "deck_plan.json"],
            settings={"model": config.image_model, "api_root": config.api_root, "size": config.image_size.as_text(), "resolution": config.image_resolution, "quality": config.image_quality, "mock_images": args.mock_api or args.mock_images},
            fn=lambda: run_style_asset_generation(
                project_root=PROJECT_ROOT,
                source_dir=step1_dir,
                output_dir=step2_dir,
                image_model=config.image_model,
                api_root=config.api_root,
                api_key_env=config.api_key_env,
                timeout_seconds=config.timeout_seconds,
                poll_initial_delay_seconds=config.poll_initial_delay_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
                poll_timeout_seconds=config.poll_timeout_seconds,
                size=config.image_size.as_text(),
                resolution=config.image_resolution,
                quality=config.image_quality,
                mock_images=args.mock_api or args.mock_images,
            ),
        )
        _require_success(step2, "Step 2")
    else:
        step2 = _load_existing_stage(step2_dir, "Step 2")
    stages.append(step2)
    if args.stop_after == "step2":
        return _finish(output_dir, config, args, stages, started_at)

    deck_plan = read_json(step2_dir / "deck_plan.json")
    slide_ids = [item["slide_id"] for item in deck_plan["slides"]]
    if args.slide_limit is not None:
        slide_ids = slide_ids[: args.slide_limit]
    expression_logic_dir = output_dir / "expression_logic"
    if _should_run(args, "expression_logic"):
        expression_logic = checkpoint_shared_stage(
            run_dir=output_dir, stage="expression_logic", source_inputs=[text_generation_dir / "parsed_text_plan.json", text_generation_dir / "run_manifest.json"],
            settings={"model": config.chat_model, "api_root": config.api_root, "max_tokens": config.chat_max_tokens, "mock_api": args.mock_api, "slide_ids": slide_ids},
            fn=lambda: run_expression_logic_planning(
                project_root=PROJECT_ROOT,
                text_generation_dir=text_generation_dir,
                output_dir=expression_logic_dir,
                slide_ids=slide_ids,
                config=config,
                mock_llm=args.mock_api,
            ),
        )
        _require_success(expression_logic, "Expression logic")
    else:
        expression_logic = _load_existing_stage(
            expression_logic_dir, "Expression logic"
        )
    stages.append(expression_logic)
    if args.stop_after == "expression_logic":
        return _finish(output_dir, config, args, stages, started_at)
    stages.extend(run_page_pipeline(
        output_dir=output_dir, slide_ids=slide_ids, config=config,
        start_at=args.start_at, stop_after=args.stop_after, page_workers=args.page_workers,
        gpu_devices=args.gpu_devices.split(",") if args.gpu_devices is not None else None,
        mock_api=args.mock_api, mock_images=args.mock_images,
        mock_diffusion=args.mock_diffusion, run_lock=False,
    ))
    return _finish(output_dir, config, args, stages, started_at)


def _load_user_input(args: argparse.Namespace) -> UserInput:
    if args.input:
        path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
        payload = read_json(path)
        return UserInput.model_validate(
            {
                "content": payload.get("content", payload.get("user_request")),
                "style": payload.get("style", payload.get("style_description")),
            }
        )
    return UserInput(content=args.content, style=args.style)


def _validate_native_contract(config) -> None:
    if config.image_size.as_list() != [1536, 864]:
        raise ValueError("HTML/background stages require APIMART_IMAGE_SIZE=1536x864")


def _require_success(manifest: dict, label: str) -> None:
    if manifest.get("status") != "succeeded":
        raise RuntimeError(f"{label} failed: {manifest.get('error')}")


def _should_run(args: argparse.Namespace, stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(args.start_at)


def _load_existing_stage(output_dir: Path, label: str) -> dict:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label} resume manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded":
        raise RuntimeError(f"{label} cannot be resumed: {manifest.get('error')}")
    return {**manifest, "resumed": True, "output_dir": str(output_dir.resolve())}


def _finish(output_dir: Path, config, args, stages: list[dict], started_at: str) -> int:
    failed_slides = sorted(
        {
            int(slide_number)
            for stage in stages
            for slide_number in stage.get("failed_slides", [])
        }
    )
    quality_stages = [
        item for item in stages
        if item.get("quality_status", "strict_pass") not in {"strict_pass", "mock", "skipped"}
        or item.get("warnings")
    ]
    manifest = {
        "pipeline": "expression_logic_html_planning",
        "status": "partial" if failed_slides else "succeeded",
        "quality_status": "degraded" if quality_stages or failed_slides else "strict_pass",
        "quality_policy": "local_repair_or_retain_output_with_warnings",
        "warnings": [
            {"stage": item.get("stage"), "quality_status": item.get("quality_status"), "details": item.get("warnings", [])}
            for item in quality_stages
        ],
        "started_at": started_at,
        "finished_at": _now_iso(),
        "stop_after": args.stop_after,
        "start_at": args.start_at,
        "mock_api": args.mock_api,
        "mock_images": args.mock_images,
        "mock_diffusion": args.mock_api or args.mock_diffusion,
        "slide_limit": args.slide_limit,
        "page_workers": args.page_workers,
        "gpu_devices": args.gpu_devices,
        "failed_slides": failed_slides,
        "configuration": {
            "chat_model": config.chat_model,
            "image_model": config.image_model,
            "image_size": config.image_size.as_list(),
            "diffusion_model": str(config.diffusion_model_path),
            "diffusion_size": config.diffusion_size.as_list(),
            "deck_font": config.deck_font,
        },
        "stages": [
            {
                "stage": item.get("stage"),
                "status": item.get("status", "succeeded"),
                "quality_status": item.get("quality_status"),
                "output_dir": item.get("output_dir"),
            }
            for item in stages
        ],
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
