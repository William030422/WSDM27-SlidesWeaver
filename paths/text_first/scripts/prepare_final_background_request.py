#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.config import load_config  # noqa: E402
from integration.final_background import (  # noqa: E402
    PROMPT_SCHEMA_VERSION,
    build_asset_board_image_payload,
    build_asset_board_plan,
    build_generation_objects,
    compile_generation_prompt,
    plan_asset_board_slots,
    render_asset_board_controls,
    select_key_color,
)
from v9_deck_planning.apimart import redacted_image_payload  # noqa: E402
from v9_deck_planning.json_utils import read_json, write_json, write_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one Final Background request without creating an API client."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slide-id", default="slide_001")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    slide_id = args.slide_id
    config = load_config(ROOT)
    canonical_background = run_dir / "step2/canonical_background.png"
    design_system = read_json(run_dir / "step2/design_system.json")
    layout = read_json(
        run_dir / "html_compile/slides" / slide_id / "html_layout.json"
    )
    visual_plan = read_json(
        run_dir
        / "visual_prompt_planning/slides"
        / slide_id
        / "visual_prompt_plan.json"
    )

    prepared_dir = run_dir / "final_background" / slide_id / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    style_prompt = str(design_system.get("style_prompt") or "").strip()
    key_color = select_key_color(style_prompt)
    objects = plan_asset_board_slots(build_generation_objects(layout, visual_plan))
    objects_path = prepared_dir / "generation_objects.json"
    write_json(objects_path, objects)
    asset_plan_path = prepared_dir / "asset_board_plan.json"
    write_json(asset_plan_path, build_asset_board_plan(objects, key_color))
    control_path = prepared_dir / "chroma_control_canvas.png"
    mask_path = prepared_dir / "generation_mask.png"
    render_asset_board_controls(objects, key_color, control_path, mask_path)
    prompt = compile_generation_prompt(objects, style_prompt, key_color)
    prompt_path = prepared_dir / "prompt.txt"
    write_text(prompt_path, prompt)
    payload = build_asset_board_image_payload(
        model=config.image_model,
        prompt=prompt,
        background_path=canonical_background,
        control_path=control_path,
        mask_path=mask_path,
        config=config,
    )
    request_preview_path = prepared_dir / "request_preview.json"
    write_json(
        request_preview_path,
        {
            "status": "prepared_not_submitted",
            "image_call_budget": 1,
            "reference_image_order": [
                "canonical_background",
                "chroma_control_canvas",
            ],
            "payload": redacted_image_payload(payload),
        },
    )
    digest = hashlib.sha256()
    for path in (
        canonical_background,
        control_path,
        mask_path,
        prompt_path,
        objects_path,
        asset_plan_path,
    ):
        digest.update(path.read_bytes())
    manifest = {
        "stage": "final_background_request_preparation",
        "status": "prepared_not_submitted",
        "slide_id": slide_id,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "model": config.image_model,
        "client_initialized": False,
        "image_call_count": 0,
        "accepted_task_count": 0,
        "bundle_sha256": digest.hexdigest(),
        "objects": str(objects_path.resolve()),
        "prompt": str(prompt_path.resolve()),
        "canonical_background": str(canonical_background.resolve()),
        "key_color": key_color,
        "asset_board_plan": str(asset_plan_path.resolve()),
        "chroma_control_canvas": str(control_path.resolve()),
        "generation_mask": str(mask_path.resolve()),
        "request_preview": str(request_preview_path.resolve()),
    }
    write_json(run_dir / "final_background/preflight_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
