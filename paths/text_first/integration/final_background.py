from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from v9_deck_planning.apimart import (
    ApiMartClient,
    ApiMartImageTaskFailure,
    extract_image_url,
    image_to_data_uri,
    redacted_image_payload,
)
from v9_deck_planning.green_background_removal import extract_assets
from v9_deck_planning.json_utils import read_json, write_json, write_text
from v9_deck_planning.segmented_asset_naming import name_segmented_assets

from .config import PipelineConfig
from .holder_safe_rect import detect_text_safe_rect

CANVAS = (1536, 864)
PROMPT_SCHEMA_VERSION = "9.0-semantic-visual-groups"
BOARD_MARGIN = 64
SLOT_GUTTER = 64
SLOT_RENDER_INSET = 14
MAX_COMPONENT_SLOT_DISTANCE = 1.25


def run_local_visual_block_backgrounds(
    *,
    project_root: Path,
    html_compile_dir: Path,
    step2_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_images: bool,
) -> dict[str, Any]:
    return run_background_set(
        project_root=project_root,
        html_compile_dir=html_compile_dir,
        step2_dir=step2_dir,
        output_dir=output_dir,
        slide_ids=slide_ids,
        config=config,
        mock_images=mock_images,
        resume_missing=True,
    )


def run_background_set(
    *,
    project_root: Path,
    html_compile_dir: Path,
    step2_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_images: bool,
    resume_missing: bool,
    client_factory: Callable[[], Any] | None = None,
    only_slide_id: str | None = None,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    del project_root
    canonical_background = step2_dir / "canonical_background.png"
    if not canonical_background.is_file():
        raise FileNotFoundError(
            f"canonical background is missing: {canonical_background}"
        )
    design_system = read_json(step2_dir / "design_system.json")
    style_prompt = str(design_system.get("style_prompt") or "").strip()
    key_color = select_key_color(style_prompt)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = None
    results = []
    image_call_count = 0

    if only_slide_id is not None and only_slide_id not in slide_ids:
        raise ValueError(f"Unknown slide: {only_slide_id}")
    for slide_number, slide_id in enumerate(slide_ids, 1):
        if only_slide_id is not None and slide_id != only_slide_id:
            continue
        slide_dir = output_dir / slide_id
        manifest_path = slide_dir / "run_manifest.json"
        prepared_dir = slide_dir / "prepared"
        live_dir = slide_dir / "live"
        layers_dir = slide_dir / "layers"
        segmentation_dir = slide_dir / "segmentation"
        extracted_assets_dir = slide_dir / "extracted_assets"
        for directory in (prepared_dir, live_dir, layers_dir):
            directory.mkdir(parents=True, exist_ok=True)

        layout = read_json(html_compile_dir / "slides" / slide_id / "html_layout.json")
        objects = plan_asset_board_slots(build_generation_objects(layout))
        objects_path = prepared_dir / "generation_objects.json"
        write_json(objects_path, objects)
        asset_plan_path = prepared_dir / "asset_board_plan.json"
        write_json(asset_plan_path, build_asset_board_plan(objects, key_color))
        control_path = prepared_dir / "chroma_control_canvas.png"
        mask_path = prepared_dir / "generation_mask.png"
        structure_path = prepared_dir / "visual_structure_reference.png"
        render_asset_board_controls(objects, key_color, control_path, mask_path)
        has_structure_reference = render_visual_structure_reference_board(
            objects, structure_path
        )
        prompt = compile_generation_prompt(objects, style_prompt, key_color)
        prompt_path = prepared_dir / "prompt.txt"
        write_text(prompt_path, prompt)
        image_requested = bool(objects)
        payload = (
            build_asset_board_image_payload(
                model=config.image_model,
                prompt=prompt,
                background_path=canonical_background,
                control_path=control_path,
                mask_path=mask_path,
                structure_path=structure_path if has_structure_reference else None,
                config=config,
            )
            if image_requested
            else None
        )
        if payload is not None:
            write_json(
                prepared_dir / "request_preview.json",
                {
                    "image_call_budget": 2,
                    "second_call_condition": "only when the primary board omits a registered slot",
                    "reference_image_order": [
                        "canonical_background",
                        "chroma_control_canvas",
                        *(
                            ["visual_structure_reference"]
                            if has_structure_reference
                            else []
                        ),
                    ],
                    "payload": redacted_image_payload(payload),
                },
            )
        bundle_paths = [
            canonical_background,
            control_path,
            mask_path,
            prompt_path,
            objects_path,
            asset_plan_path,
            *([structure_path] if has_structure_reference else []),
        ]
        bundle_sha256 = hashlib.sha256((
            _prepared_digest(bundle_paths) + json.dumps({
                "model": config.image_model, "size": config.image_size.as_list(),
                "resolution": config.image_resolution, "quality": config.image_quality,
                "schema": PROMPT_SCHEMA_VERSION,
            }, sort_keys=True)
        ).encode()).hexdigest()
        if resume_missing and _completed_slide_is_reusable(
            manifest_path, expected_bundle_sha256=bundle_sha256
        ):
            results.append(read_json(manifest_path))
            continue

        generated_path = live_dir / "generated.png"
        failure_placeholder: dict[str, Any] | None = None
        if not image_requested:
            Image.new("RGB", CANVAS, _hex_to_rgb(key_color)).save(generated_path)
            task_id = None
            slide_image_calls = 0
        elif mock_images:
            render_mock_asset_board(objects, key_color, generated_path)
            write_json(
                live_dir / "result.json",
                {
                    "status": "succeeded",
                    "mock": True,
                    "bundle_sha256": bundle_sha256,
                    "image_call_count": 0,
                    "output": str(generated_path.resolve()),
                },
            )
            task_id = None
            slide_image_calls = 0
        else:
            if client is None:
                client = (
                    client_factory()
                    if client_factory
                    else ApiMartClient(
                        api_root=config.api_root,
                        api_key_env=config.api_key_env,
                        env_path=config.env_path,
                        timeout_seconds=config.timeout_seconds,
                        poll_initial_delay_seconds=config.poll_initial_delay_seconds,
                        poll_interval_seconds=config.poll_interval_seconds,
                        poll_timeout_seconds=config.poll_timeout_seconds,
                    )
                )
            assert payload is not None
            try:
                generation = _generate_or_resume(
                    client=client,
                    payload=payload,
                    bundle_sha256=bundle_sha256,
                    live_dir=live_dir,
                )
                task_id = generation["task_id"]
                slide_image_calls = generation["image_calls_during_run"]
            except ApiMartImageTaskFailure as exc:
                task_id = exc.task_id
                slide_image_calls = exc.image_calls_during_run
                failure_placeholder = {
                    "slide_number": slide_number,
                    "reason": "image_task_failed_after_one_resubmission",
                    "task_id": task_id,
                    "error": exc.error,
                }
                render_failure_placeholder(generated_path, slide_number)
                write_json(
                    live_dir / "result.json",
                    {
                        "status": "failed_placeholder",
                        "bundle_sha256": bundle_sha256,
                        "task_id": task_id,
                        "failure": failure_placeholder,
                        "output": str(generated_path.resolve()),
                    },
                )
            image_call_count += slide_image_calls

        if failure_placeholder is not None:
            layer_manifest = empty_layer_manifest(generated_path)
            layer_manifest["failure_placeholder"] = failure_placeholder
            recovery_task_id = None
        elif image_requested:
            layer_manifest = extract_chroma_layers(
                generated_path=generated_path,
                canonical_background=canonical_background,
                objects=objects,
                asset_plan_path=asset_plan_path,
                segmentation_dir=segmentation_dir,
                extracted_assets_dir=extracted_assets_dir,
                output_dir=layers_dir,
                allow_missing=True,
            )
            missing_ids = layer_manifest.pop("missing_object_ids")
            recovery_task_id = None
            if missing_ids and not mock_images:
                assert client is not None
                try:
                    recovery = _recover_missing_objects(
                        client=client,
                        model=config.image_model,
                        config=config,
                        canonical_background=canonical_background,
                        objects=[
                            item for item in objects if item["object_id"] in missing_ids
                        ],
                        key_color=key_color,
                        style_prompt=style_prompt,
                        output_dir=slide_dir / "recovery",
                    )
                except ApiMartImageTaskFailure as exc:
                    recovery_task_id = exc.task_id
                    slide_image_calls += exc.image_calls_during_run
                    image_call_count += exc.image_calls_during_run
                    failure_placeholder = {
                        "slide_number": slide_number,
                        "reason": "missing_object_recovery_failed_after_one_resubmission",
                        "task_id": recovery_task_id,
                        "missing_object_ids": missing_ids,
                        "error": exc.error,
                    }
                    render_failure_placeholder(generated_path, slide_number)
                    layer_manifest = empty_layer_manifest(generated_path)
                    layer_manifest["failure_placeholder"] = failure_placeholder
                    write_json(
                        slide_dir / "recovery" / "live" / "result.json",
                        {
                            "status": "failed_placeholder",
                            "failure": failure_placeholder,
                            "output": str(generated_path.resolve()),
                        },
                    )
                else:
                    layer_manifest["layers"].extend(
                        recovery["layer_manifest"]["layers"]
                    )
                    layer_manifest.setdefault("quality_warnings", []).extend(
                        recovery["layer_manifest"].get("quality_warnings", [])
                    )
                    layer_manifest["recovery"] = recovery["record"]
                    recovery_task_id = recovery["record"]["task_id"]
                    recovery_image_calls = recovery["record"]["image_calls_during_run"]
                    slide_image_calls += recovery_image_calls
                    image_call_count += recovery_image_calls
            if missing_ids and failure_placeholder is None:
                _append_missing_slot_fallbacks(
                    layer_manifest,
                    [item for item in objects if item["object_id"] in missing_ids],
                    generated_path,
                    layers_dir,
                )
        else:
            layer_manifest = empty_layer_manifest(canonical_background)
            recovery_task_id = None
        layer_manifest.setdefault("quality_warnings", []).extend(
            {"object_id": item["object_id"], "reason": item["quality_warning"]}
            for item in objects if item.get("quality_warning")
        )
        layer_manifest_path = slide_dir / "layer_manifest.json"
        write_json(layer_manifest_path, layer_manifest)
        background_path = slide_dir / "background.png"
        shutil.copyfile(
            generated_path if failure_placeholder is not None else canonical_background,
            background_path,
        )
        quality_status = (
            "failed_placeholder"
            if failure_placeholder is not None
            else "degraded" if layer_manifest.get("quality_warnings")
            else ("mock" if mock_images else "strict_pass")
        )
        result = {
            "slide_id": slide_id,
            "status": "failed_placeholder"
            if failure_placeholder is not None
            else "succeeded",
            "quality_status": quality_status,
            "model": config.image_model,
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "bundle_sha256": bundle_sha256,
            "task_id": task_id,
            "image_requested": image_requested,
            "image_calls_during_run": slide_image_calls,
            "accepted_task_count": (
                0
                if failure_placeholder is not None
                else int(task_id is not None) + int(recovery_task_id is not None)
            ),
            "recovery_task_id": recovery_task_id,
            "background": str(background_path.resolve()),
            "generated_canvas": str(generated_path.resolve()),
            "layer_manifest": str(layer_manifest_path.resolve()),
            "visual_count": sum(item["kind"] == "visual" for item in objects),
            "visual_generation_group_count": sum(
                item["kind"] == "visual" for item in objects
            ),
            "holder_count": sum(item["kind"] == "holder" for item in objects),
            "key_color": key_color,
            "control_canvas": str(control_path.resolve()),
            "generation_mask": str(mask_path.resolve()),
            "asset_board_plan": str(asset_plan_path.resolve()),
            "prompt": str(prompt_path.resolve()),
            "failure_placeholder": failure_placeholder,
            "quality_warnings": layer_manifest.get("quality_warnings", []),
        }
        write_json(manifest_path, result)
        results.append(result)

    failed_slides = [
        int(item["failure_placeholder"]["slide_number"])
        for item in results
        if item.get("failure_placeholder")
    ]
    manifest = {
        "stage": "chroma_asset_board_visual_and_holder_generation",
        "status": "succeeded",
        "quality_status": (
            "degraded" if failed_slides or any(item.get("quality_warnings") for item in results)
            else ("mock" if mock_images else "strict_pass")
        ),
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "model": config.image_model,
        "mock_images": mock_images,
        "image_call_granularity": "one primary asset-board task per slide plus one missing-slot recovery when needed",
        "image_call_count": image_call_count,
        "accepted_task_count": sum(item["accepted_task_count"] for item in results),
        "slide_count": len(results),
        "failed_slides": failed_slides,
        "background_reference_index": 0,
        "extraction_algorithm": "clean_branch_global_lab_dual_threshold_slot_aware_chroma_removal",
        "visual_placement": "one_native_resolution_asset_per_html_visual_region",
        "output_dir": str(output_dir.resolve()),
        "slides": results,
    }
    if write_stage_manifest:
        write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_generation_objects(layout: dict[str, Any]) -> list[dict[str, Any]]:
    objects = []
    text_by_id = {item["region_id"]: item for item in layout.get("text_regions", [])}
    group_by_region = {}
    for group in layout.get("visual_generation_groups") or []:
        region_ids = list(
            group.get("source_region_ids") or group.get("region_ids") or []
        )
        if len(region_ids) == 1:
            group_by_region[region_ids[0]] = group
    for visual in layout.get("visual_regions") or []:
        region_id = visual["region_id"]
        group = group_by_region.get(region_id, {})
        objects.append(
            {
                "object_id": f"visual_group_{region_id}",
                "kind": "visual",
                "bbox_px": visual["bbox_px"],
                "prompt": _visual_generation_prompt(
                    group.get("semantic_description")
                    or visual.get("visual_brief")
                    or ""
                ),
                "source_region_ids": [region_id],
                "relation_type": "single_visual",
                "structure_reference": None,
            }
        )
    for holder in layout.get("holders", []):
        text = text_by_id.get(holder["text_region_id"], {})
        planned_text_bbox = text.get("text_bbox_px") or text.get("bbox_px")
        required = _holder_required_rect(holder["bbox_px"], planned_text_bbox)
        objects.append(
            {
                "object_id": holder["holder_id"],
                "kind": "holder",
                "bbox_px": holder["bbox_px"],
                "text_region_id": holder["text_region_id"],
                "required_text_bbox_px": planned_text_bbox,
                "required_text_rect": required,
                "prompt": (
                    "Generate one creative blank presentation text surface that feels compatible with the "
                    "overall deck style. Choose an appropriate shape, edge, subtle texture, and restrained "
                    "decoration on the OUTER FRAME ONLY while keeping the entire reserved text rectangle "
                    "calm, opaque and undecorated. Preserve a rich themed frame; do not substitute a plain box. "
                    "No characters, portraits, scenes, badges, icons, diagonal bands or other intrusions "
                    "inside the text interior, even if characters are requested for visual assets. Do not copy "
                    "or miniaturize the canonical background composition. Ignore any visual flow structure "
                    "reference; it applies only to visual assets. No readable text or pseudo-text."
                ),
            }
        )
    return objects


def _holder_required_rect(holder_bbox: list[float], text_bbox: list[float] | None) -> dict[str, float]:
    if not text_bbox:
        return {"x": 0.10, "y": 0.10, "width": 0.80, "height": 0.80}
    x, y, width, height = [float(value) for value in holder_bbox]
    tx, ty, tw, th = [float(value) for value in text_bbox]
    return {"x": (tx - x) / max(width, 1), "y": (ty - y) / max(height, 1),
            "width": tw / max(width, 1), "height": th / max(height, 1)}


def _holder_compatibility(item: dict[str, Any], safe: dict[str, float] | None) -> dict[str, Any]:
    required = item.get("required_text_rect") or _holder_required_rect(item["bbox_px"], None)
    covered = bool(safe and required["x"] >= safe["x"] - 1e-6
                   and required["y"] >= safe["y"] - 1e-6
                   and required["x"] + required["width"] <= safe["x"] + safe["width"] + 1e-6
                   and required["y"] + required["height"] <= safe["y"] + safe["height"] + 1e-6)
    return {"status": "compatible" if covered else "requires_quiet_backing",
            "planned_text_bbox_px": item.get("required_text_bbox_px"),
            "required_text_rect": required, "detected_safe_rect": safe,
            "planned_rect_covered": covered, "keep_decoration": True}


def plan_asset_board_slots(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not objects:
        return []
    available_width = CANVAS[0] - 2 * BOARD_MARGIN
    available_height = CANVAS[1] - 2 * BOARD_MARGIN
    placements: dict[str, list[int]] | None = None
    scale = 1.0
    while scale >= 0.2:
        packed = _pack_shelves(objects, scale, available_width, available_height)
        if packed is not None:
            placements = packed
            break
        scale = round(scale - 0.025, 3)
    if placements is None:
        # Generation-board packing is independent of final slide geometry.
        # A dense but bounded board is preferable to aborting valid slides.
        columns = max(1, math.ceil(math.sqrt(len(objects) * available_width / available_height)))
        rows = math.ceil(len(objects) / columns)
        cell_width = max(1, available_width // columns)
        cell_height = max(1, available_height // rows)
        gutter = min(SLOT_GUTTER, max(0, min(cell_width, cell_height) // 5))
        placements = {}
        for index, item in enumerate(objects):
            target = item["bbox_px"]
            fit_scale = min((cell_width - gutter) / max(1, target[2]), (cell_height - gutter) / max(1, target[3]))
            width, height = max(1, round(target[2] * fit_scale)), max(1, round(target[3] * fit_scale))
            placements[item["object_id"]] = [
                BOARD_MARGIN + index % columns * cell_width + (cell_width - width) // 2,
                BOARD_MARGIN + index // columns * cell_height + (cell_height - height) // 2,
                width, height,
            ]
        scale = 0.0

    planned = []
    for item in objects:
        slot = placements[item["object_id"]]
        planned.append(
            {
                **item,
                "slot_bbox_px": slot,
                "slot_bbox": _normalized_bbox(slot),
                "board_scale": scale,
                **({"quality_warning": "dense_generation_board_fallback"} if scale == 0.0 else {}),
            }
        )
    return planned


def _visual_generation_prompt(brief: str) -> str:
    cleaned = re.sub(
        r"^(?:empty\s+)?(?:neutral\s+)?(?:abstract\s+)?visual\s+placeholder\s+for\s+",
        "",
        str(brief).strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^neutral\s+abstract\s+", "", cleaned, flags=re.IGNORECASE)
    return (
        "Generate one integrated, actual content-filled, text-free visual depicting: "
        f"{cleaned} Preserve the important scene, progression, cycle, branching, convergence, or shared "
        "environment described here as one coherent composition. Do not generate a blank frame, "
        "placeholder, or text holder."
    )


def _pack_shelves(
    objects: list[dict[str, Any]],
    scale: float,
    available_width: int,
    available_height: int,
) -> dict[str, list[int]] | None:
    sized = [
        (
            item,
            max(24, round(item["bbox_px"][2] * scale)),
            max(24, round(item["bbox_px"][3] * scale)),
        )
        for item in objects
    ]
    if any(
        width > available_width or height > available_height
        for _, width, height in sized
    ):
        return None
    sized.sort(key=lambda value: (-value[2], -value[1], value[0]["object_id"]))
    rows: list[dict[str, int]] = []
    placements: dict[str, list[int]] = {}
    for item, width, height in sized:
        row = next(
            (
                candidate
                for candidate in rows
                if height <= candidate["height"]
                and candidate["next_x"] + width <= BOARD_MARGIN + available_width
            ),
            None,
        )
        if row is None:
            y = BOARD_MARGIN + sum(candidate["height"] for candidate in rows)
            if rows:
                y += SLOT_GUTTER * len(rows)
            if y + height > BOARD_MARGIN + available_height:
                return None
            row = {"y": y, "height": height, "next_x": BOARD_MARGIN}
            rows.append(row)
        x = row["next_x"]
        placements[item["object_id"]] = [x, row["y"], width, height]
        row["next_x"] = x + width + SLOT_GUTTER
    return placements


def build_asset_board_plan(
    objects: list[dict[str, Any]], key_color: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "board_size": list(CANVAS),
        "key_color": key_color,
        "slots": [
            {
                "slot_id": item["object_id"],
                "kind": item["kind"],
                "bbox": item["slot_bbox"],
                "bbox_px": item["slot_bbox_px"],
                "target_bbox_px": item["bbox_px"],
                "enabled": True,
            }
            for item in objects
        ],
    }


def render_asset_board_controls(
    objects: list[dict[str, Any]],
    key_color: str,
    control_path: Path,
    mask_path: Path,
) -> None:
    key = _hex_to_rgb(key_color)
    control = Image.new("RGB", CANVAS, key)
    mask = Image.new("RGBA", CANVAS, (*key, 255))
    mask_draw = ImageDraw.Draw(mask, "RGBA")
    for item in objects:
        mask_draw.rectangle(_bbox_xyxy(item["slot_bbox_px"]), fill=(*key, 0))
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control.save(control_path)
    mask.save(mask_path)


def render_visual_structure_reference_board(
    objects: list[dict[str, Any]], output_path: Path
) -> bool:
    references = [
        item
        for item in objects
        if item["kind"] == "visual"
        and item.get("structure_reference")
        and Path(str(item["structure_reference"])).is_file()
    ]
    if not references:
        return False
    board = Image.new("RGB", CANVAS, "white")
    for item in references:
        x, y, width, height = item["slot_bbox_px"]
        with Image.open(item["structure_reference"]).convert("RGB") as source:
            fitted_bbox = _fit_bbox_for_aspect(source.size, [x, y, width, height])
            resized = source.resize(fitted_bbox[2:], Image.Resampling.LANCZOS)
            board.paste(resized, fitted_bbox[:2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)
    return True


def compile_generation_prompt(
    objects: list[dict[str, Any]], style_prompt: str, key_color: str
) -> str:
    lines = []
    for item in objects:
        x, y, width, height = item["slot_bbox_px"]
        lines.append(
            f"- {item['object_id']} ({item['kind']}) in x={x}-{x + width}, y={y}-{y + height}: "
            f"{item['prompt']}"
        )
        if item["kind"] == "holder":
            lines.append(
                "  Reserved text rectangle, normalized to this holder's complete outer edges: "
                + json.dumps(item.get("required_text_rect") or _holder_required_rect(item["bbox_px"], None),
                             separators=(",", ":"))
                + ". Keep the WHOLE rectangle quiet and opaque; decorations stay outside it. "
                  "These coordinates are control instructions, never readable text or guide marks."
            )
    return """Generate one reusable chroma-key asset board, not a finished slide.

Reference image 1 is the canonical deck background. Use it only as broad evidence of the deck's overall style and
palette. Do not reproduce its layout or make holders look like miniature copies of it. Reference image 2 is the flat
chroma control board. When reference image 3 is present, it is a faint visual-structure guide board embedded only in
visual slots. Use it to understand flow, branching, convergence, cycles, and relative composition, then redesign the
guide as a coherent final scene. Holders must ignore reference image 3.

Keep every unused pixel perfectly flat {key_color}. Paint exactly one isolated asset inside every listed slot.
Do not let any asset blend into or visually merge with the {key_color} contrast-color background.
The slots have been sized from the HTML plan and moved apart only to create wide chroma gutters. Stay inside each slot,
keep every part of an asset close together, and never connect, bridge, shadow, or bleed across a gutter. Do not use
{key_color} inside an asset. Avoid transparency, motion blur, glow, soft bleeding, and cast shadows on the chroma field.

Every visual slot corresponds to exactly one complete HTML visual region and one downstream asset. Generate one coherent,
content-rich composition inside that single region; depict any sequence, comparison, cycle, or branching internally.
Never fragment it into disconnected mini-assets or connect it to another visual, a holder, the background, or chroma field.
Each holder is a separate creative blank text surface with an opaque quiet interior. Visual slots must contain actual
semantic imagery, never an empty frame or placeholder. Do not connect holders to visuals.

Slots:
{objects}

Deck style:
{style}

No readable text, pseudo-text, letters, numbers, labels, captions, logos, or watermarks. Do not add extra assets,
background scenes, paper, frames around the full board, or decoration outside the slots.
""".format(
        key_color=key_color,
        objects="\n".join(lines) or "- none",
        style=style_prompt,
    )


def build_asset_board_image_payload(
    *,
    model: str,
    prompt: str,
    background_path: Path,
    control_path: Path,
    mask_path: Path,
    structure_path: Path | None = None,
    config: PipelineConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": config.image_size.as_text(),
        "resolution": config.image_resolution,
        "n": 1,
        "image_urls": [
            image_to_data_uri(background_path),
            image_to_data_uri(control_path),
            *([image_to_data_uri(structure_path)] if structure_path else []),
        ],
    }
    if model.startswith("gpt-image-"):
        payload.update(
            {
                "quality": config.image_quality,
                "output_format": "png",
                "mask_url": image_to_data_uri(mask_path),
            }
        )
    elif model == "doubao-seedream-5-0-pro":
        payload.update({"output_format": "png", "watermark": False})
    elif model not in {
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview-official",
    }:
        raise ValueError(f"unsupported final asset-board image model: {model}")
    return payload


# Compatibility alias for callers that prepared the old full-canvas request directly.
build_full_canvas_image_payload = build_asset_board_image_payload


def render_mock_asset_board(
    objects: list[dict[str, Any]], key_color: str, output_path: Path
) -> None:
    image = Image.new("RGB", CANVAS, _hex_to_rgb(key_color))
    draw = ImageDraw.Draw(image)
    for index, item in enumerate(objects):
        x, y, width, height = item["slot_bbox_px"]
        inset = min(SLOT_RENDER_INSET, max(4, min(width, height) // 10))
        box = (x + inset, y + inset, x + width - inset - 1, y + height - inset - 1)
        if item["kind"] == "holder":
            radius = max(8, min(width, height) // 10)
            draw.rounded_rectangle(
                box,
                radius=radius,
                fill=(239, 232, 207),
                outline=(42, 55, 48),
                width=7,
            )
            draw.line(
                (box[0] + 18, box[1] + 12, box[2] - 18, box[1] + 12),
                fill=(164, 117, 47),
                width=4,
            )
        else:
            fill = (61, 106, 82) if index % 2 == 0 else (70, 92, 133)
            draw.rounded_rectangle(
                box, radius=14, fill=fill, outline=(28, 38, 34), width=7
            )
            center_y = (box[1] + box[3]) // 2
            for step in range(4):
                cx = box[0] + (step + 1) * (box[2] - box[0]) // 5
                size = max(9, min(width, height) // 14)
                draw.ellipse(
                    (cx - size, center_y - size, cx + size, center_y + size),
                    fill=(224, 180, 66),
                    outline=(28, 38, 34),
                    width=3,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def extract_chroma_layers(
    *,
    generated_path: Path,
    canonical_background: Path,
    objects: list[dict[str, Any]],
    asset_plan_path: Path,
    segmentation_dir: Path,
    extracted_assets_dir: Path,
    output_dir: Path,
    allow_missing: bool = False,
) -> dict[str, Any]:
    for directory in (segmentation_dir, extracted_assets_dir, output_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    quality_warnings = []
    registered_path = generated_path
    with Image.open(generated_path) as downloaded:
        source_board_size = list(downloaded.size)
        if downloaded.size != CANVAS:
            # The slot contract is expressed in CANVAS pixels. Keep the original
            # download and register a separate copy before any pixel assignment.
            registered_path = segmentation_dir / "registered_canvas.png"
            downloaded.convert("RGB").resize(CANVAS, Image.Resampling.LANCZOS).save(registered_path)
            quality_warnings.append({
                "reason": "generated_board_size_normalized",
                "source_size": source_board_size, "registered_size": list(CANVAS),
                "original_download": str(generated_path.resolve()),
            })
    slots = [
        {"slot_id": item["object_id"], "bbox": item["slot_bbox"]} for item in objects
    ]
    component_dir = segmentation_dir / "components"
    segmentation = extract_assets(
        registered_path,
        segmentation_dir,
        assets_dir=component_dir,
        slots=slots,
    )
    naming = name_segmented_assets(
        segmentation_dir / "manifest.json",
        asset_plan_path,
        extracted_assets_dir,
    )
    segmentation = read_json(segmentation_dir / "manifest.json")
    assignments = _assign_components_to_slots(segmentation["assets"], objects)
    layers = []
    for item in objects:
        records = assignments[item["object_id"]]
        if not records:
            continue
        try:
            asset, content_bbox = _compose_component_records(records, segmentation_dir)
        except ValueError as exc:
            # Fully transparent matte is a local quality failure, not an API
            # failure. Keep other assets and let this slot recover below.
            quality_warnings.append({"object_id": item["object_id"], "reason": str(exc)})
            continue
        if item["kind"] == "holder":
            placed_bbox = item["bbox_px"]
            placement = "fit_exact_html_holder_bbox"
        else:
            placed_bbox = _fit_bbox_for_aspect(asset.size, item["bbox_px"])
            placement = "aspect_fit_inside_html_visual_region"
        native_pixel_size = list(asset.size)
        path = output_dir / f"{item['object_id']}.png"
        asset.save(path)
        text_safe_rect = (
            detect_text_safe_rect(path) if item["kind"] == "holder" else None
        )
        compatibility = _holder_compatibility(item, text_safe_rect) if item["kind"] == "holder" else None
        if compatibility and not compatibility["planned_rect_covered"]:
            quality_warnings.append({"object_id": item["object_id"],
                "reason": "holder_interior_does_not_cover_planned_text", "repair": "quiet_backing_keep_decorated_frame"})
        layers.append(
            {
                "layer_id": item["object_id"],
                "kind": item["kind"],
                "bbox_px": placed_bbox,
                "target_bbox_px": item["bbox_px"],
                "slot_bbox_px": item["slot_bbox_px"],
                "source_content_bbox_px": list(content_bbox),
                "native_pixel_size": native_pixel_size,
                "embedded_pixel_size": native_pixel_size,
                "display_pixel_size": placed_bbox[2:],
                "resolution_preserved": True,
                "placement": placement,
                "path": str(path.resolve()),
                "source_region_ids": item.get("source_region_ids", []),
                "text_region_id": item.get("text_region_id"),
                "text_safe_rect": text_safe_rect,
                "holder_compatibility": compatibility,
                "holder_fallback": bool(
                    item["kind"] == "holder" and text_safe_rect is None
                ),
                "source_component_ids": [record["asset_id"] for record in records],
            }
        )
    manifest = {
        "schema_version": "2.0",
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "base_background": str(canonical_background.resolve()),
        "registered_canvas": str(registered_path.resolve()),
        "source_board_size": source_board_size,
        "extraction": "clean_branch_slot_aware_chroma_removal",
        "segmentation_algorithm": segmentation["algorithm"],
        "naming_algorithm": naming["algorithm"],
        "segmentation_manifest": str((segmentation_dir / "manifest.json").resolve()),
        "naming_manifest": str((extracted_assets_dir / "manifest.json").resolve()),
        "missing_object_ids": [item["object_id"] for item in objects if item["object_id"] not in {layer["layer_id"] for layer in layers}],
        "quality_warnings": quality_warnings,
        "layers": layers,
    }
    if not allow_missing:
        _append_missing_slot_fallbacks(manifest, objects, registered_path, output_dir)
    return manifest


def _append_missing_slot_fallbacks(
    manifest: dict[str, Any], objects: list[dict[str, Any]], generated_path: Path, output_dir: Path
) -> None:
    """Keep registered text/slots usable after the existing recovery budget.

    Only the object's own source slot may supply a degraded crop. Empty holder
    slots publish native-holder fallback metadata; unrelated components are not
    borrowed and the canonical background is kept.
    """
    existing = {layer["layer_id"] for layer in manifest["layers"]}
    if all(item["object_id"] in existing for item in objects):
        return
    warnings = manifest.setdefault("quality_warnings", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(manifest.get("registered_canvas") or generated_path)).convert("RGB") as board:
        array = np.asarray(board)
        band = np.concatenate((array[:12].reshape(-1, 3), array[-12:].reshape(-1, 3), array[:, :12].reshape(-1, 3), array[:, -12:].reshape(-1, 3)))
        key = np.median(band, axis=0).astype(np.uint8)
        key_lab = cv2.cvtColor(key.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(3)
        for item in objects:
            object_id = item["object_id"]
            if object_id in existing:
                continue
            x, y, width, height = item["slot_bbox_px"]
            crop = board.crop((x, y, x + max(1, width), y + max(1, height))).convert("RGBA")
            rgba = np.array(crop)
            distance = np.linalg.norm(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32) - key_lab, axis=2)
            rgba[:, :, 3] = (distance >= 35.0).astype(np.uint8) * 255
            crop = Image.fromarray(rgba)
            content_bbox = crop.getchannel("A").getbbox()
            warning = {"object_id": object_id, "reason": "missing_reliable_slot_asset", "fallback": "native_holder" if item["kind"] == "holder" else "registered_slot_crop" if content_bbox else "canonical_background_only"}
            warnings.append(warning)
            if item["kind"] != "holder" and content_bbox is None:
                continue
            if item["kind"] == "holder":
                # A native editable surface is safer than a partial holder.
                crop = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
            elif content_bbox is not None:
                crop = crop.crop(content_bbox)
            path = output_dir / f"{object_id}_fallback.png"
            crop.save(path)
            placed = item["bbox_px"] if item["kind"] == "holder" else _fit_bbox_for_aspect(crop.size, item["bbox_px"])
            manifest["layers"].append({
                "layer_id": object_id, "kind": item["kind"], "bbox_px": placed,
                "target_bbox_px": item["bbox_px"], "slot_bbox_px": item["slot_bbox_px"],
                "path": str(path.resolve()), "native_pixel_size": list(crop.size),
                "embedded_pixel_size": list(crop.size), "display_pixel_size": placed[2:],
                "resolution_preserved": True, "placement": "local_quality_fallback",
                "source_region_ids": item.get("source_region_ids", []), "text_region_id": item.get("text_region_id"),
                "text_safe_rect": None, "holder_fallback": item["kind"] == "holder",
                "source_component_ids": [], "quality_warning": warning,
            })
            existing.add(object_id)
    manifest["missing_object_ids"] = [item["object_id"] for item in objects if item["object_id"] not in existing]


def _recover_missing_objects(
    *,
    client: Any,
    model: str,
    config: PipelineConfig,
    canonical_background: Path,
    objects: list[dict[str, Any]],
    key_color: str,
    style_prompt: str,
    output_dir: Path,
) -> dict[str, Any]:
    prepared_dir = output_dir / "prepared"
    live_dir = output_dir / "live"
    for directory in (prepared_dir, live_dir):
        directory.mkdir(parents=True, exist_ok=True)
    planned = _plan_recovery_slots(objects)
    objects_path = prepared_dir / "generation_objects.json"
    asset_plan_path = prepared_dir / "asset_board_plan.json"
    control_path = prepared_dir / "chroma_control_canvas.png"
    mask_path = prepared_dir / "generation_mask.png"
    structure_path = prepared_dir / "visual_structure_reference.png"
    prompt_path = prepared_dir / "prompt.txt"
    write_json(objects_path, planned)
    write_json(asset_plan_path, build_asset_board_plan(planned, key_color))
    render_asset_board_controls(planned, key_color, control_path, mask_path)
    has_structure_reference = render_visual_structure_reference_board(
        planned, structure_path
    )
    prompt = compile_generation_prompt(planned, style_prompt, key_color)
    write_text(prompt_path, prompt)
    payload = build_asset_board_image_payload(
        model=model,
        prompt=prompt,
        background_path=canonical_background,
        control_path=control_path,
        mask_path=mask_path,
        structure_path=structure_path if has_structure_reference else None,
        config=config,
    )
    bundle_sha256 = _prepared_digest(
        [
            canonical_background,
            control_path,
            mask_path,
            prompt_path,
            objects_path,
            asset_plan_path,
            *([structure_path] if has_structure_reference else []),
        ]
    )
    generation = _generate_or_resume(
        client=client,
        payload=payload,
        bundle_sha256=bundle_sha256,
        live_dir=live_dir,
    )
    layer_manifest = extract_chroma_layers(
        generated_path=live_dir / "generated.png",
        canonical_background=canonical_background,
        objects=planned,
        asset_plan_path=asset_plan_path,
        segmentation_dir=output_dir / "segmentation",
        extracted_assets_dir=output_dir / "extracted_assets",
        output_dir=output_dir / "layers",
    )
    layer_manifest.pop("missing_object_ids", None)
    return {
        "record": {
            "recovered_object_ids": [layer["layer_id"] for layer in layer_manifest["layers"] if not layer.get("quality_warning")],
            "quality_warnings": layer_manifest.get("quality_warnings", []),
            "task_id": generation["task_id"],
            "image_calls_during_run": generation["image_calls_during_run"],
            "bundle_sha256": bundle_sha256,
        },
        "layer_manifest": layer_manifest,
    }


def _plan_recovery_slots(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not objects:
        return []
    height = 480.0
    raw_widths = [height * item["bbox_px"][2] / item["bbox_px"][3] for item in objects]
    available_width = CANVAS[0] - 2 * BOARD_MARGIN - SLOT_GUTTER * (len(objects) - 1)
    scale = min(1.0, available_width / sum(raw_widths))
    widths = [max(120, round(width * scale)) for width in raw_widths]
    slot_height = max(120, round(height * scale))
    total_width = sum(widths) + SLOT_GUTTER * (len(objects) - 1)
    if total_width > CANVAS[0] - 2 * BOARD_MARGIN:
        return plan_asset_board_slots(objects)
    x = round((CANVAS[0] - total_width) / 2)
    y = round((CANVAS[1] - slot_height) / 2)
    planned = []
    for item, width in zip(objects, widths, strict=True):
        slot = [x, y, width, slot_height]
        planned.append(
            {
                **item,
                "slot_bbox_px": slot,
                "slot_bbox": _normalized_bbox(slot),
                "board_scale": round(slot_height / item["bbox_px"][3], 6),
            }
        )
        x += width + SLOT_GUTTER
    return planned


def _assign_components_to_slots(
    records: list[dict[str, Any]], objects: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    assignments = {item["object_id"]: [] for item in objects}
    if not objects:
        return assignments
    claimed_records: set[str] = set()
    claimed_objects: set[str] = set()
    overlap_pairs = []
    for record in records:
        component = _xyxy_to_bbox(record["component_bbox"])
        for item in objects:
            overlap = _intersection_area(component, item["slot_bbox_px"])
            if overlap:
                overlap_pairs.append((overlap, record, item))
    for _overlap, record, item in sorted(
        overlap_pairs, key=lambda value: value[0], reverse=True
    ):
        record_id = record["asset_id"]
        object_id = item["object_id"]
        if record_id in claimed_records or object_id in claimed_objects:
            continue
        assignments[object_id].append(record)
        claimed_records.add(record_id)
        claimed_objects.add(object_id)

    unmatched = [
        record for record in records if record["asset_id"] not in claimed_records
    ]
    # Detached parts of an occupied slot stay with that object; an unrelated
    # missing slot must not consume them just to make the board look complete.
    for record in list(unmatched):
        component = _xyxy_to_bbox(record["component_bbox"])
        owner = max(objects, key=lambda item: _intersection_area(component, item["slot_bbox_px"]))
        if _intersection_area(component, owner["slot_bbox_px"]) > 0 and owner["object_id"] in claimed_objects:
            assignments[owner["object_id"]].append(record)
            claimed_records.add(record["asset_id"])
            unmatched.remove(record)
    while unmatched and len(claimed_objects) < len(objects):
        missing = [item for item in objects if item["object_id"] not in claimed_objects]
        candidates = [
            (
                _normalized_center_distance(*record["centroid"], item["slot_bbox_px"]),
                record,
                item,
            )
            for record in unmatched
            for item in missing
            if _normalized_center_distance(*record["centroid"], item["slot_bbox_px"])
            <= MAX_COMPONENT_SLOT_DISTANCE
        ]
        if not candidates:
            break
        distance, record, owner = min(candidates, key=lambda value: value[0])
        assignments[owner["object_id"]].append(record)
        claimed_records.add(record["asset_id"])
        claimed_objects.add(owner["object_id"])
        unmatched.remove(record)

    for record in unmatched:
        cx, cy = record["centroid"]
        owner = min(
            objects,
            key=lambda item: _normalized_center_distance(cx, cy, item["slot_bbox_px"]),
        )
        distance = _normalized_center_distance(cx, cy, owner["slot_bbox_px"])
        if distance > MAX_COMPONENT_SLOT_DISTANCE:
            continue
        assignments[owner["object_id"]].append(record)
    return assignments


def _compose_component_records(
    records: list[dict[str, Any]], segmentation_dir: Path
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    board = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    for record in records:
        path = (segmentation_dir / record["path"]).resolve()
        with Image.open(path).convert("RGBA") as component:
            x, y, _, _ = record["crop_bbox"]
            board.alpha_composite(component, (x, y))
    content_bbox = board.getchannel("A").getbbox()
    if content_bbox is None:
        raise ValueError("extracted chroma asset is fully transparent")
    return board.crop(content_bbox), content_bbox


def empty_layer_manifest(canonical_background: Path) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "base_background": str(canonical_background.resolve()),
        "extraction": "not_required",
        "layers": [],
    }


def render_failure_placeholder(output_path: Path, slide_number: int) -> None:
    del slide_number
    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    red = (204, 30, 45)
    draw.line((520, 250, 1016, 614), fill=red, width=42)
    draw.line((1016, 250, 520, 614), fill=red, width=42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def select_key_color(style: str) -> str:
    lowered = style.lower()
    if re.search(r"\b(?:green|lime|emerald|chartreuse|minecraft)\b", lowered):
        if re.search(r"\b(?:magenta|fuchsia|pink|purple)\b", lowered):
            return "#0000ff"
        return "#ff00ff"
    return "#00ff00"


def _save_task(path: Path, record: dict[str, Any]) -> None:
    from .page_runtime import atomic_write_json
    atomic_write_json(path, record)


def _retryable_image_failure(error: Any) -> bool:
    # Terminal moderation/policy refusals are not repaired by repeating a POST.
    def text(value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(f"{key} {text(item)}" for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return " ".join(text(item) for item in value)
        return str(value)
    message = " ".join(text(error).casefold().split())
    if any(marker in message for marker in (
        "content_review", "content_policy", "safety_reject", "moderation", "policy_violation",
        "content safety review",
    )):
        return False
    return re.search(
        r"\bdid not pass (?:the |a |our )?(?:content )?safety review\b"
        r"|\bbilled for (?:the )?input only\b", message,
    ) is None


def _generate_or_resume(
    *, client: Any, payload: dict[str, Any], bundle_sha256: str, live_dir: Path
) -> dict[str, Any]:
    generated_path = live_dir / "generated.png"
    result_path = live_dir / "result.json"
    task_path = live_dir / "task.json"
    if result_path.is_file() and generated_path.is_file():
        result = read_json(result_path)
        if result.get("status") == "succeeded" and result.get("bundle_sha256") == bundle_sha256:
            return {"task_id": result.get("task_id"), "image_calls_during_run": 0}
    record = read_json(task_path) if task_path.is_file() else {}
    if record.get("bundle_sha256") != bundle_sha256:
        record = {}
    task_id = record.get("task_id")
    attempts = int(record.get("submission_attempt") or (1 if task_id else 0))
    calls = 0
    if not task_id and record.get("status") in {"submitting", "submission_unknown"}:
        raise RuntimeError("Previous image submission outcome is unknown; reconcile it before submitting again")
    if record.get("status") == "failed":
        if attempts >= 2 or not _retryable_image_failure(record.get("error")):
            raise ApiMartImageTaskFailure(str(task_id), record.get("error"))
        task_id = None
    while True:
        if not task_id:
            attempts += 1
            record = {"status": "submitting", "task_id": None,
                      "bundle_sha256": bundle_sha256, "submission_attempt": attempts}
            _save_task(task_path, record)
            try:
                task_id = client.submit_image_task(payload)
                calls += 1
            except Exception as exc:
                record.update(status="submission_unknown", last_error_type=type(exc).__name__)
                _save_task(task_path, record)
                raise
            record.update(status="submitted", task_id=task_id)
            _save_task(task_path, record)
        try:
            task = (record["result"] if record.get("status") in {"download_pending", "completed"}
                    and record.get("result") else client.poll_task(task_id))
        except ApiMartImageTaskFailure as exc:
            record.update(status="failed", error=exc.error)
            _save_task(task_path.with_name(f"task.attempt_{attempts:02d}.failed.json"), record)
            _save_task(task_path, record)
            if attempts < 2 and _retryable_image_failure(exc.error):
                task_id = None
                continue
            exc.image_calls_during_run = calls
            raise
        except Exception as exc:
            record.update(status="poll_pending", last_error_type=type(exc).__name__)
            _save_task(task_path, record)
            raise
        record.update(status="download_pending", result=task)
        _save_task(task_path, record)
        partial = live_dir / "generated.downloading.png"
        client.download(extract_image_url(task), partial)
        partial.replace(generated_path)
        record.update(status="completed")
        _save_task(task_path, record)
        _save_task(result_path, {
            "status": "succeeded", "completed_at": datetime.now(timezone.utc).isoformat(),
            "bundle_sha256": bundle_sha256, "task_id": task_id,
            "output": str(generated_path.resolve()),
        })
        return {"task_id": task_id, "image_calls_during_run": calls}


def _prepared_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _completed_slide_is_reusable(
    manifest_path: Path, *, expected_bundle_sha256: str
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        if (manifest.get("status") != "succeeded"
                or manifest.get("prompt_schema_version") != PROMPT_SCHEMA_VERSION
                or manifest.get("bundle_sha256") != expected_bundle_sha256):
            return False
        layers = read_json(Path(manifest["layer_manifest"]))
        paths = [Path(manifest["background"]), Path(layers["base_background"])]
        paths.extend(Path(layer["path"]) for layer in layers["layers"] if layer.get("path"))
        for path in paths:
            with Image.open(path) as image:
                image.verify()
        return True
    except (OSError, ValueError, KeyError, TypeError):
        # Interrupted derived reports/layers can be rebuilt from the existing
        # generated board and task journal without another image submission.
        return False


def _normalized_bbox(bbox: list[int]) -> list[float]:
    x, y, width, height = bbox
    return [
        round(x / CANVAS[0], 6),
        round(y / CANVAS[1], 6),
        round((x + width) / CANVAS[0], 6),
        round((y + height) / CANVAS[1], 6),
    ]


def _bbox_xyxy(bbox: list[int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return x, y, x + width - 1, y + height - 1


def _xyxy_to_bbox(bbox: list[int]) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [x0, y0, x1 - x0, y1 - y0]


def _intersection_area(left: list[int], right: list[int]) -> int:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    width = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    height = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    return width * height


def _fit_bbox_for_aspect(
    asset_size: tuple[int, int], target_bbox: list[int]
) -> list[int]:
    asset_width, asset_height = asset_size
    x, y, width, height = target_bbox
    scale = min(width / asset_width, height / asset_height)
    fitted_width = max(1, round(asset_width * scale))
    fitted_height = max(1, round(asset_height * scale))
    return [
        round(x + (width - fitted_width) / 2),
        round(y + (height - fitted_height) / 2),
        fitted_width,
        fitted_height,
    ]


def _normalized_center_distance(cx: float, cy: float, bbox: list[int]) -> float:
    x, y, width, height = bbox
    return math.hypot((cx - x - width / 2) / width, (cy - y - height / 2) / height)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
