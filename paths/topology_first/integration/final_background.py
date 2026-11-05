from __future__ import annotations

import colorsys
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

from v9_deck_planning.apimart import (
    ApiMartClient,
    ApiMartImageTaskFailure,
    extract_image_url,
    image_to_data_uri,
    redacted_image_payload,
)
from v9_deck_planning.json_utils import read_json, write_json, write_text

from .config import PipelineConfig


CANVAS = (1536, 864)
FRAME_WIDTH = 6
COLOR_ALPHA = 0.80
COLOR_FEATHER_PX = 2
PROMPT_SCHEMA_VERSION = "2.0-spatial-registration"

BACKGROUND_VARIANTS = (
    {
        "slug": "gpt_soft_contrast",
        "model": "gpt-image-2-official",
        "control_mode": "contrast_frames",
        "prompt_mode": "soft",
        "require_distinct": False,
    },
    {
        "slug": "nano_color_control",
        "model": "gemini-3-pro-image-preview",
        "control_mode": "color_fields",
        "prompt_mode": "strict",
        "require_distinct": False,
    },
    {
        "slug": "doubao_soft_contrast_distinct",
        "model": "doubao-seedream-5-0-pro",
        "control_mode": "contrast_frames",
        "prompt_mode": "soft",
        "require_distinct": True,
    },
)


def run_soft_boundary_final_backgrounds(
    *,
    project_root: Path,
    region_content_dir: Path,
    step2_dir: Path,
    topology_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_images: bool,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    variant = _strategy_for_model(config.image_model)
    return run_background_set(
        project_root=project_root,
        region_content_dir=region_content_dir,
        step2_dir=step2_dir,
        topology_dir=topology_dir,
        output_dir=output_dir,
        slide_ids=slide_ids,
        config=config,
        variant=variant,
        mock_images=mock_images,
        resume_missing=True,
        write_stage_manifest=write_stage_manifest,
    )


def run_background_matrix(
    *,
    project_root: Path,
    region_content_dir: Path,
    step2_dir: Path,
    topology_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_images: bool = False,
    resume_missing: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(BACKGROUND_VARIANTS)) as executor:
        futures = {
            executor.submit(
                run_background_set,
                project_root=project_root,
                region_content_dir=region_content_dir,
                step2_dir=step2_dir,
                topology_dir=topology_dir,
                output_dir=output_dir / variant["slug"],
                slide_ids=slide_ids,
                config=config,
                variant=variant,
                mock_images=mock_images,
                resume_missing=resume_missing,
            ): variant
            for variant in BACKGROUND_VARIANTS
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["variant"])
    manifest = {
        "stage": "three_model_region_aware_background_matrix",
        "status": "succeeded",
        "mock_images": mock_images,
        "slide_count_per_variant": len(slide_ids),
        "variant_count": len(results),
        "total_image_call_count": sum(item["image_call_count"] for item in results),
        "output_dir": str(output_dir.resolve()),
        "variants": results,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def run_background_set(
    *,
    project_root: Path,
    region_content_dir: Path,
    step2_dir: Path,
    topology_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    variant: dict[str, Any],
    mock_images: bool,
    resume_missing: bool,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    canonical_background = step2_dir / "canonical_background.png"
    design_system = read_json(step2_dir / "design_system.json")
    client = None
    if not mock_images:
        client = ApiMartClient(
            api_root=config.api_root,
            api_key_env=config.api_key_env,
            env_path=project_root / ".env",
            timeout_seconds=config.timeout_seconds,
            poll_initial_delay_seconds=config.poll_initial_delay_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
            poll_timeout_seconds=config.poll_timeout_seconds,
        )
    results = []
    image_call_count = 0
    warnings = []
    for slide_id in slide_ids:
        slide_dir = output_dir / slide_id
        slide_manifest_path = slide_dir / "run_manifest.json"
        # Construct the actual request before considering reuse. A schema
        # version alone cannot distinguish changed content, style or controls.
        plan = read_json(
            region_content_dir / "slides" / slide_id / "region_content_plan.json"
        )
        topology = read_json(
            topology_dir / "calibrated_topology" / slide_id / "calibrated_topology.json"
        )
        layout = read_json(
            topology_dir / "postprocessed" / "layouts" / f"{slide_id}.json"
        )
        bindings = bind_visual_briefs(plan, layout, topology)
        write_json(slide_dir / "component_briefs.json", bindings)
        region_coordinates = compile_region_coordinate_map(bindings)
        region_coordinates_path = slide_dir / "region_coordinates.json"
        write_json(region_coordinates_path, region_coordinates)
        reference_path = slide_dir / "control_reference.png"
        render_control_reference(
            canonical_background,
            bindings,
            reference_path,
            variant["control_mode"],
        )
        prompt = compile_generation_prompt(
            bindings,
            design_system["style_prompt"],
            prompt_mode=variant["prompt_mode"],
            control_mode=variant["control_mode"],
            require_distinct=variant["require_distinct"],
        ) if bindings else "No visual regions were planned; preserve the canonical background and editable copy."
        prompt_path = slide_dir / "prompt.txt"
        write_text(prompt_path, prompt)
        raw_output_path = slide_dir / "background_raw.png"
        output_path = slide_dir / "background.png"
        request_path = slide_dir / "request.json"
        payload = build_image_payload(
            model=variant["model"],
            prompt=prompt,
            reference_path=reference_path,
        )
        request_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        write_json(
            request_path,
            {
                "mock": mock_images,
                "variant": variant,
                "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                "planned_text_sent_to_image_model": False,
                "generation_skipped_reason": "no_visual_regions" if not bindings else None,
                "source_experiment": str(
                    (project_root / "reused" / "final_background" / "run_matrix.py").resolve()
                ),
                "payload": redacted_image_payload(payload),
            },
        )
        task_id = None
        failed_visual = None
        if not bindings:
            failed_visual = {"task_id": None, "category": "no_visual_regions",
                             "error": "No planned visual regions; required scene imagery is unavailable",
                             "fallback": "canonical_background_only"}
            warnings.append({"slide_id": slide_id, "failed_visual": failed_visual,
                             "warning": "missing_planned_visual"})
            shutil.copyfile(canonical_background, raw_output_path)
        elif mock_images:
            shutil.copyfile(canonical_background, raw_output_path)
        else:
            assert client is not None
            try:
                generation = _generate_or_resume(
                    client=client, payload=payload, request_sha256=request_sha256,
                    slide_dir=slide_dir, raw_output_path=raw_output_path,
                    resume=resume_missing,
                )
                task_id = generation["task_id"]
                image_call_count += generation["image_calls_during_run"]
            except ApiMartImageTaskFailure as exc:
                task_id = exc.task_id
                image_call_count += exc.image_calls_during_run
                failed_visual = {"task_id": task_id, "category": exc.category,
                                 "error": exc.error, "fallback": "canonical_background_only"}
                warnings.append({"slide_id": slide_id, "failed_visual": failed_visual})
                # Keep this deck's own healthy background and editable text.
                # This is explicitly a failed visual, never a claimed image.
                shutil.copyfile(canonical_background, raw_output_path)
        normalization = normalize_background(raw_output_path, output_path)
        result = {
            "slide_id": slide_id,
            "status": "succeeded",
            "quality_status": "degraded" if failed_visual else "strict_pass",
            "failed_visual": failed_visual,
            "model": variant["model"],
            "variant": variant["slug"],
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "task_id": task_id,
            "background": str(output_path.resolve()),
            "raw_background": str(raw_output_path.resolve()),
            "reference": str(reference_path.resolve()),
            "prompt": str(prompt_path.resolve()),
            "component_briefs": str((slide_dir / "component_briefs.json").resolve()),
            "region_coordinates": str(region_coordinates_path.resolve()),
            "normalization": normalization,
        }
        write_json(slide_manifest_path, result)
        results.append(result)
    manifest = {
        "stage": "region_aware_background_generation",
        "status": "succeeded",
        "quality_status": "degraded" if warnings else "strict_pass",
        "warnings": warnings,
        "variant": variant["slug"],
        "model": variant["model"],
        "control_mode": variant["control_mode"],
        "prompt_mode": variant["prompt_mode"],
        "require_distinct": variant["require_distinct"],
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "mock_images": mock_images,
        "image_call_count": image_call_count,
        "slide_count": len(results),
        "output_dir": str(output_dir.resolve()),
        "slides": results,
    }
    if write_stage_manifest:
        write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _save_task(path: Path, record: dict[str, Any]) -> None:
    """Replace the existing task record atomically, including pending states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, record)
    temporary.replace(path)


def _generate_or_resume(
    *, client: Any, payload: dict[str, Any], request_sha256: str,
    slide_dir: Path, raw_output_path: Path, resume: bool = True,
) -> dict[str, Any]:
    task_path = slide_dir / "task.json"
    record = read_json(task_path) if resume and task_path.is_file() else {}
    if resume and not record:
        # The per-page completed manifest is a second durable record of the
        # paid task. Losing a journal must not cause another image submission.
        manifest_path = slide_dir / "run_manifest.json"
        previous = read_json(manifest_path) if manifest_path.is_file() else {}
        if previous.get("request_sha256") == request_sha256 and previous.get("task_id"):
            failed = previous.get("failed_visual")
            record = {"status": "failed" if failed else "completed", "task_id": previous["task_id"],
                      "request_sha256": request_sha256, "submission_attempt": 2,
                      "error": (failed or {}).get("error")}
            _save_task(task_path, record)
    if record.get("request_sha256") != request_sha256:
        record = {}
    if record.get("status") == "completed" and raw_output_path.is_file():
        try:
            with Image.open(raw_output_path) as image:
                image.verify()
            return {"task_id": record["task_id"], "image_calls_during_run": 0}
        except (OSError, ValueError):
            pass  # Re-download the completed task; never pay for a new image.
    calls = 0
    attempts = int(record.get("submission_attempt", 0))
    history = list(record.get("attempt_history", []))
    task_id = record.get("task_id")
    if not task_id and record.get("status") in {"submitting", "submission_unknown"}:
        raise RuntimeError("Previous image submission outcome is unknown; reconcile it before submitting again")
    if record.get("status") == "failed":
        error = ApiMartImageTaskFailure(str(task_id), record.get("error"))
        if not error.retryable or attempts >= 2:
            raise error
        task_id = None
    while True:
        if not task_id:
            # A POST timeout has an unknown remote outcome. Do not automatically
            # repeat it here: the caller must reconcile that transport failure.
            attempts += 1
            record = {"status": "submitting", "task_id": None,
                      "request_sha256": request_sha256,
                      "submission_attempt": attempts, "attempt_history": history}
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
            task = client.poll_task(task_id)
        except ApiMartImageTaskFailure as exc:
            history.append({"task_id": task_id, "error": exc.error,
                            "category": exc.category, "submission_attempt": attempts})
            record.update(status="failed", error=exc.error, error_category=exc.category,
                          attempt_history=history)
            _save_task(task_path, record)
            if exc.retryable and attempts < 2:
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
        download_path = slide_dir / ".download"
        client.download(extract_image_url(task), download_path)
        with Image.open(download_path) as image:
            image.convert("RGB").save(raw_output_path, format="PNG")
        download_path.unlink(missing_ok=True)
        record.update(status="completed")
        _save_task(task_path, record)
        return {"task_id": task_id, "image_calls_during_run": calls}


def bind_visual_briefs(
    region_plan: dict[str, Any],
    layout: dict[str, Any],
    topology: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if topology is not None:
        from .region_content import build_generation_units

        boxes = {
            item["unit_id"]: item["bbox_px"]
            for item in build_generation_units(topology, layout)
        }
    else:
        boxes = {
            item["region_id"]: item["postprocessed_bbox_px"]
            for item in layout.get("entities", [])
        }
        boxes.update(
            {
                item["unit_id"]: item["bbox_px"]
                for item in layout.get("generation_units", [])
            }
        )
    bindings = []
    for item in region_plan["visual_regions"]:
        region_id = item["region_id"]
        if region_id not in boxes:
            raise ValueError(f"visual brief references unknown generation unit: {region_id}")
        bbox = boxes[region_id]
        bindings.append(
            {
                "component_id": region_id,
                "position": describe_position(bbox),
                "bbox_px": bbox,
                "visual_brief": item["semantic_brief"],
            }
        )
    bindings.sort(
        key=lambda item: (
            item["bbox_px"][1],
            item["bbox_px"][0],
            item["component_id"],
        )
    )
    for index, item in enumerate(bindings):
        item["region_label"] = _region_label(index)
    return bindings


def render_control_reference(
    background_path: Path,
    bindings: list[dict[str, Any]],
    output_path: Path,
    control_mode: str,
) -> None:
    with Image.open(background_path).convert("RGB") as background:
        if background.size != CANVAS:
            raise ValueError(f"background must be {CANVAS}, got {background.size}")
        reference = background.copy()
        source = background.copy()
    if control_mode == "contrast_frames":
        draw = ImageDraw.Draw(reference)
        for item in bindings:
            bbox = item["bbox_px"]
            average = tuple(
                round(value) for value in ImageStat.Stat(source.crop(_bbox_xyxy(bbox))).mean
            )
            draw.rectangle(
                _bbox_xyxy(bbox),
                outline=choose_contrast_color(average),
                width=FRAME_WIDTH,
            )
    elif control_mode == "color_fields":
        for item in bindings:
            bbox = item["bbox_px"]
            average = tuple(
                round(value) for value in ImageStat.Stat(source.crop(_bbox_xyxy(bbox))).mean
            )
            color = choose_contrast_color(average)
            mask = _color_field_mask(bbox)
            reference = Image.composite(Image.new("RGB", CANVAS, color), reference, mask)
    else:
        raise ValueError(f"unsupported control mode: {control_mode}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.save(output_path)


def compile_generation_prompt(
    bindings: list[dict[str, Any]],
    style: str,
    *,
    prompt_mode: str,
    control_mode: str,
    require_distinct: bool,
) -> str:
    if not bindings:
        raise ValueError("at least one visual binding is required")
    mapping = "\n".join(
        f"REGION {_region_label(index)}: {item['visual_brief'].strip()}"
        for index, item in enumerate(bindings)
    )
    region_coordinates = compile_region_coordinate_json(bindings)
    if control_mode == "contrast_frames":
        marker = f"{len(bindings)} hollow contrast-color rectangles"
    elif control_mode == "color_fields":
        marker = f"{len(bindings)} 80%-opaque color fields with 2px feathered edges"
    else:
        raise ValueError(f"unsupported control mode: {control_mode}")
    if prompt_mode not in {"strict", "soft"}:
        raise ValueError(f"unsupported prompt mode: {prompt_mode}")
    contrast = (
        "Make every requested concept concrete, clearly visible, and separated from the surrounding "
        "background with strong local contrast."
        if require_distinct
        else "Use enough local contrast for every requested concept to remain readable without hard edges."
    )
    style_commands = extract_core_style_commands(style)
    return f"""Edit the provided single 16:9 slide image.

SPATIAL REGISTRATION IS THE HIGHEST-PRIORITY REQUIREMENT.
Preserving the supplied layout is more important than visual richness, storytelling, or cinematic composition.

The {marker} are fixed and independent placement regions. They are not suggestions.
Do not move, reorder, merge, duplicate, resize, or reinterpret any region.
Do not redesign the slide into a storyboard, comic strip, collage, or full-canvas illustration.

REGION COORDINATES (JSON; values are [x,y,width,height] pixels):
{region_coordinates}

REGION SEMANTICS:
{mapping}

For every visual cluster:
- Place its visual centroid inside the central 40% of its assigned rectangle.
- Keep at least 95% of the requested subject inside that rectangle.
- Only soft glow, faint shadow, dust, and atmospheric feathering may extend outside it by a very small margin.
- Do not let a character, object, arrow, symbol, or meaningful event enter another region.
- Generate exactly one semantic concept per region; no swaps, merges, duplicates, or omissions.
- {contrast}

PRESERVE:
- Preserve the exact canvas dimensions, aspect ratio, and all unmarked background areas.
- Change only the interiors of the assigned regions and the guide marks.
- Remove all guide rectangles, frames, or tint cleanly after generating the visuals.
- Keep the existing background illumination, texture, frame, and decorations unchanged.
- Do not crop, zoom, shift, or recompose the slide.

STYLE:
{style_commands}
Use {len(bindings)} isolated visual clusters, not comic panels.
Do not introduce extra scenes, characters, narrative beats, or unrelated objects beyond the region semantics.

No panels, borders, cards, UI, charts, readable text, labels, letters, numbers, logos, or watermarks."""


def compile_region_coordinate_map(bindings: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the small coordinate payload embedded in the image prompt."""
    return {
        "canvas": list(CANVAS),
        "regions": {
            _region_label(index): list(item["bbox_px"])
            for index, item in enumerate(bindings)
        },
    }


def compile_region_coordinate_json(bindings: list[dict[str, Any]]) -> str:
    return json.dumps(
        compile_region_coordinate_map(bindings),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def extract_core_style_commands(style: str) -> str:
    """Preserve identity and palette, including attributes following a colon."""
    normalized = re.sub(r"\s+", " ", str(style or "").strip())
    if not normalized:
        return "Use a clean, coherent visual language matching the supplied reference."
    return normalized


def build_image_payload(
    *,
    model: str,
    prompt: str,
    reference_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "image_urls": [image_to_data_uri(reference_path)],
    }
    if model == "gpt-image-2-official":
        payload.update(
            {
                "size": "1536x864",
                "resolution": "1k",
                "quality": "high",
                "output_format": "png",
            }
        )
    elif model == "gemini-3-pro-image-preview":
        payload.update({"size": "16:9", "resolution": "1K"})
    elif model == "doubao-seedream-5-0-pro":
        payload.update(
            {
                "size": "16:9",
                "resolution": "1K",
                "output_format": "png",
                "watermark": False,
            }
        )
    else:
        raise ValueError(f"unsupported background model: {model}")
    return payload


def normalize_background(source_path: Path, output_path: Path) -> dict[str, Any]:
    with Image.open(source_path).convert("RGB") as source:
        source_size = source.size
        if source_size == CANVAS:
            source.save(output_path, format="PNG")
            operation = "copied_without_resize"
        else:
            source.resize(CANVAS, Image.Resampling.LANCZOS).save(output_path, format="PNG")
            operation = "resized_to_canonical_canvas"
    return {
        "source_size": list(source_size),
        "canonical_size": list(CANVAS),
        "operation": operation,
    }


def choose_contrast_color(average: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = (value / 255 for value in average)
    hue, _, saturation = colorsys.rgb_to_hls(red, green, blue)
    hue = (hue + 0.5) % 1.0
    saturation = max(saturation, 0.65)
    candidates = [
        tuple(round(value * 255) for value in colorsys.hls_to_rgb(hue, lightness, saturation))
        for lightness in (0.22, 0.78)
    ]
    return max(candidates, key=lambda color: _contrast_ratio(average, color))


def describe_position(bbox: list[int]) -> str:
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2
    horizontal = "left" if center_x < 512 else "right" if center_x > 1024 else "center"
    vertical = "upper" if center_y < 288 else "lower" if center_y > 576 else "middle"
    return f"{vertical}-{horizontal}"


def _region_label(index: int) -> str:
    if index < 0:
        raise ValueError("region index must be non-negative")
    label = ""
    while True:
        index, remainder = divmod(index, 26)
        label = chr(ord("A") + remainder) + label
        if index == 0:
            return label
        index -= 1


def _strategy_for_model(model: str) -> dict[str, Any]:
    for variant in BACKGROUND_VARIANTS:
        if variant["model"] == model:
            return variant
    raise ValueError(f"no tested background strategy for model: {model}")


def _bbox_xyxy(bbox: list[int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return x, y, x + width - 1, y + height - 1


def _color_field_mask(bbox: list[int]) -> Image.Image:
    bounds = _bbox_xyxy(bbox)
    peak_alpha = round(255 * COLOR_ALPHA)
    mask = Image.new("L", CANVAS, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bounds, fill=peak_alpha)
    left, top, right, bottom = bounds
    for offset in range(COLOR_FEATHER_PX):
        inset = (left + offset, top + offset, right - offset, bottom - offset)
        if inset[0] > inset[2] or inset[1] > inset[3]:
            break
        edge_alpha = round(peak_alpha * (offset + 1) / (COLOR_FEATHER_PX + 1))
        draw.rectangle(inset, outline=edge_alpha, width=1)
    return mask


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255
        channels.append(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    high, low = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)
