from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from v9_deck_planning.json_utils import read_json, write_json
from v9_deck_planning.models import DeckPlan

from .config import PipelineConfig
from .html_planning import CANVAS, _committed_copy_text, prepare_html_layout
from .text_generation import TEXT_FIELD_PATTERN, parse_text_block, parse_text_plan


RECOMMENDED_REGISTERED_COVERAGE = 0.45
EQUIVALENT_SIZE_TOLERANCE_PX = 1
SLIDE_HEIGHT_IN = 7.5
POINTS_PER_CANVAS_PIXEL = SLIDE_HEIGHT_IN * 72.0 / CANVAS[1]


def run_html_to_coordinates(
    *,
    html_planning_dir: Path,
    step2_dir: Path,
    text_generation_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    only_slide_id: str | None = None,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    """Compile the final HTML layout directly into the downstream slide contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        if config.image_size.as_list() != list(CANVAS):
            warnings.append(
                {
                    "stage": "configuration",
                    "warning": f"configured image canvas differs from HTML canvas {CANVAS}",
                }
            )
        deck_plan = DeckPlan.model_validate(read_json(step2_dir / "deck_plan.json"))
        if [slide.slide_id for slide in deck_plan.slides][: len(slide_ids)] != slide_ids:
            raise ValueError("HTML compile slide IDs do not match the deck plan")
        text_plan = parse_text_plan(
            json.dumps(read_json(text_generation_dir / "text_plan.json"), ensure_ascii=False),
            deck_plan,
        )
        text_by_slide = {slide["slide_id"]: slide for slide in text_plan["slides"]}
        canonical_background = step2_dir / "canonical_background.png"
        if not canonical_background.is_file():
            raise FileNotFoundError(
                f"HTML compile requires the canonical background: {canonical_background}"
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": CANVAS[0], "height": CANVAS[1]},
                device_scale_factor=1,
            )
            context.route("http://**/*", lambda route: route.abort())
            context.route("https://**/*", lambda route: route.abort())
            page = context.new_page()
            if only_slide_id is not None and only_slide_id not in slide_ids:
                raise ValueError(f"Unknown slide: {only_slide_id}")
            for slide_id in slide_ids:
                if only_slide_id is not None and slide_id != only_slide_id:
                    continue
                source_html = html_planning_dir / "slides" / slide_id / "layout.html"
                if not source_html.is_file():
                    raise FileNotFoundError(f"HTML compile requires planned HTML: {source_html}")
                result = _compile_slide(
                    page=page,
                    slide_id=slide_id,
                    html_path=source_html,
                    source_text=text_by_slide[slide_id],
                    visual_groups_path=(
                        html_planning_dir
                        / "slides"
                        / slide_id
                        / "visual_generation_groups.json"
                    ),
                    output_dir=output_dir / "slides" / slide_id,
                    deck_font=config.deck_font,
                    semantic_context={"deck_style": deck_plan.style, "deck_goal": deck_plan.deck_goal,
                                      "slide_purpose": next(slide.rough_message for slide in deck_plan.slides if slide.slide_id == slide_id)},
                )
                records.append(result)
                warnings.extend(
                    {
                        "slide_id": slide_id,
                        "stage": "html_compile",
                        "warning": warning,
                    }
                    for warning in result["warnings"]
                )
            context.close()
            browser.close()

        if write_stage_manifest:
            write_json(
                output_dir / "html_layout.json",
                {
                    "schema_version": "2.0",
                    "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
                    "slides": [
                        {"slide_id": item["slide_id"], "html_layout": item["html_layout"]}
                        for item in records
                    ],
                },
            )
        manifest = {
            "stage": "final_html_layout_compile",
            "status": "succeeded",
            "quality_status": "degraded" if warnings else "strict_pass",
            "browser": "playwright_chromium",
            "network_enabled": False,
            "canvas": list(CANVAS),
            "model_call_count": 0,
            "topology_generated": False,
            "calibration_executed": False,
            "slide_count": len(records),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "output_dir": str(output_dir.resolve()),
            "warnings": warnings,
            "slides": records,
        }
    except Exception as exc:
        manifest = {
            "stage": "final_html_layout_compile",
            "status": "failed",
            "error": str(exc),
            "model_call_count": 0,
            "slide_count": len(records),
            "output_dir": str(output_dir.resolve()),
            "slides": records,
        }
    if write_stage_manifest:
        write_json(output_dir / "run_manifest.json", manifest)
    if manifest["status"] != "succeeded":
        raise RuntimeError(f"HTML compile failed: {manifest['error']}")
    return manifest


def _compile_slide(
    *,
    page: Any,
    slide_id: str,
    html_path: Path,
    source_text: dict[str, Any],
    visual_groups_path: Path,
    output_dir: Path,
    deck_font: str = "Noto Sans",
    semantic_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = prepare_html_layout(
        html_path, deck_font=deck_font, source_text=source_text, slide_id=slide_id, page=page
    )
    page.goto(html_path.resolve().as_uri(), wait_until="load")
    page.evaluate("document.fonts.ready")
    slides = page.locator("section.slide")
    slide_count = slides.count()
    if slide_count != 1:
        warnings.append(f"expected one section.slide, found {slide_count}")
    slide = slides.first if slide_count else page.locator("body")
    if slide.get_attribute("data-slide-id") != slide_id:
        warnings.append("layout root has a missing or mismatched data-slide-id")
    payload = slide.evaluate(_BROWSER_EXTRACTION_SCRIPT)
    if [round(value) for value in payload["slide_size"]] != list(CANVAS):
        warnings.append(f"rendered size is {payload['slide_size']} instead of {list(CANVAS)}")
    regions, region_warnings = _validate_regions(payload, source_text)
    warnings.extend(region_warnings)
    modules, module_warnings = _validate_modules(payload.get("modules", []), regions)
    warnings.extend(module_warnings)
    visual_regions = [item for item in regions if item["role"] == "visual"]
    text_regions = [item for item in regions if item["content_key"] is not None]
    holders = _deduplicate_holders(text_regions)
    visual_generation_groups, grouping_warnings = _compile_visual_generation_groups(
        visual_regions=visual_regions,
        visual_groups_path=visual_groups_path,
    )
    warnings.extend(grouping_warnings)
    connectors, connector_warnings = _compile_connectors(regions)
    warnings.extend(connector_warnings)
    page.reload(wait_until="load")
    page.evaluate("document.fonts.ready")
    slide = page.locator("section.slide").first
    layout = {
        "schema_version": "2.0",
        "slide_id": slide_id,
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "source_html": str(html_path.resolve()),
        "layout_authority": "browser_getBoundingClientRect",
        "text_regions": text_regions,
        "modules": modules,
        "holders": holders,
        "visual_regions": visual_regions,
        "visual_generation_groups": visual_generation_groups,
        "connectors": connectors,
        "semantic_context": semantic_context or {},
        "regions": regions,
        "reading_order": [item["region_id"] for item in text_regions],
        "registered_coverage": round(
            sum(item["bbox_px"][2] * item["bbox_px"][3] for item in regions)
            / (CANVAS[0] * CANVAS[1]),
            4,
        ),
        "text_sha256": _text_hash(source_text),
        "display_text_sha256": _display_text_hash(text_regions),
        "geometry_repairs": payload.get("geometry_repairs", []),
    }
    layout_path = output_dir / "html_layout.json"
    write_json(layout_path, layout)
    preview_path = output_dir / "layout_preview.png"
    slide.screenshot(path=str(preview_path))
    page.add_style_tag(
        content=(
            "[data-content-key],[data-content-key] *{color:transparent!important;"
            "text-shadow:none!important;-webkit-text-fill-color:transparent!important}"
        )
    )
    control_path = output_dir / "background_control.png"
    slide.screenshot(path=str(control_path))
    return {
        "slide_id": slide_id,
        "status": "succeeded",
        "quality_status": "degraded" if warnings else "strict_pass",
        "warnings": warnings,
        "html_layout": str(layout_path.resolve()),
        "layout_preview": str(preview_path.resolve()),
        "background_control": str(control_path.resolve()),
        "region_count": len(regions),
        "text_region_count": len(text_regions),
        "holder_count": len(holders),
        "visual_region_count": len(visual_regions),
        "visual_generation_group_count": len(visual_generation_groups),
        "registered_coverage": layout["registered_coverage"],
        "module_count": len(modules),
    }


def _validate_regions(
    payload: dict[str, Any], source_text: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if payload.get("extra_text"):
        warnings.append("contains unregistered visible text")
    expected = _expected_content(source_text)
    seen_content: set[str] = set()
    seen_ids: set[str] = set()
    regions: list[dict[str, Any]] = []
    raw_regions = list(payload.get("regions", []))
    present = {raw.get("content_key") for raw in raw_regions}
    for key in expected:
        if key in present:
            continue
        role, text = expected[key]
        fallback = _find_recovery_text_bbox(raw_regions)
        raw_regions.append({
            "region_id": f"recovered_{key}", "content_key": key, "role": role,
            "bbox_px": fallback, "inner_text": text, "text_surface": "plain",
            "font_size_px": 32 if role == "title" else 24,
        })
        warnings.append(f"restored missing frozen content key: {key}")
    for index, raw in enumerate(raw_regions, start=1):
        region_id = str(raw.get("region_id") or f"region_{index:03d}").strip()
        if region_id in seen_ids:
            original_id = region_id
            region_id = f"{region_id}__recovered_{index}"
            while region_id in seen_ids:
                region_id += "_"
            warnings.append(f"renamed duplicate data-region-id {original_id} to {region_id}")
        seen_ids.add(region_id)
        content_key = raw.get("content_key")
        role = str(raw.get("role") or "").strip()
        if content_key in expected and content_key not in seen_content:
            expected_role, expected_text = expected[content_key]
            role = expected_role
            seen_content.add(content_key)
        elif role == "visual" and content_key is None:
            expected_text = ""
        else:
            warnings.append(f"ignored unknown or duplicate content key on {region_id}")
            continue
        bbox, bbox_warning = _normalize_bbox(raw.get("bbox_px", []))
        if bbox_warning:
            warnings.append(f"{region_id} {bbox_warning}")
        if role == "visual":
            visual_brief = str(raw.get("visual_brief") or "").strip()
            if not visual_brief:
                visual_brief = "Create one detailed instructive text-free visual."
                warnings.append(f"{region_id} lacks data-visual-brief")
            presentation = None
            holder = None
            text_bbox = None
            overflow = False
            display_text = None
            parsed_display = None
            typography = None
        else:
            display_text = str(raw.get("inner_text") or "").strip()
            if not display_text:
                display_text = expected_text
                raw = {**raw, "rich_paragraphs": [], "list_items": []}
                warnings.append(f"restored empty display text for {content_key}")
            elif _committed_copy_text(display_text) != _committed_copy_text(expected_text):
                display_text = expected_text
                raw = {**raw, "rich_paragraphs": [], "list_items": []}
                warnings.append(f"restored committed wording for {content_key}")
            overflow = bool(raw.get("overflow"))
            text_bbox, text_bbox_warning = _normalize_bbox(raw.get("text_content_bbox_px") or bbox)
            if text_bbox_warning:
                warnings.append(f"{region_id} content box {text_bbox_warning}")
            holder = _normalize_holder(raw.get("holder"), region_id)
            explicit_surface = str(raw.get("text_surface") or "").strip()
            presentation = (
                "generated_holder_text"
                if explicit_surface == "holder" or holder is not None
                else "direct_text"
            )
            if presentation == "generated_holder_text" and holder is None:
                presentation = "direct_text"
                warnings.append(
                    f"{content_key} requests a holder but has no explicit data-holder-id; used direct text"
                )
            visual_brief = None
            if overflow:
                warnings.append(f"{content_key} overflows its HTML-planned text box")
            parsed_display = _parse_display_content(content_key, display_text, raw)
            typography = _normalize_typography(raw)
        regions.append(
            {
                "region_id": region_id,
                "role": role,
                "content_key": content_key,
                "module_id": raw.get("module_id"),
                "bbox_px": bbox,
                "raw_bbox_px": raw.get("original_bbox_px") or raw.get("bbox_px"),
                "text_bbox_px": text_bbox,
                "presentation_mode": presentation,
                "text_alignment": typography["text_alignment"] if typography else None,
                "holder": holder,
                "visual_brief": visual_brief,
                "source_text": expected_text if role != "visual" else None,
                "display_text": display_text,
                "copy_changed": bool(role != "visual" and display_text != expected_text),
                "display_content": parsed_display,
                "typography": typography,
                "text_box_insets_px": raw.get("text_box_insets_px", [0, 0, 0, 0]),
                "vertical_alignment": raw.get("vertical_alignment", "top"),
                "equivalent_group": raw.get("equivalent_group"),
                "generation_group": raw.get("generation_group"),
                "overflow_detected": overflow,
                "text_ink_bbox_px": raw.get("text_ink_bbox_px"),
                "z_index": raw.get("z_index"),
                "connect_to": str(raw.get("connect_to") or "") if role == "visual" else "",
                "connector_color": raw.get("connector_color"),
                "connector_width_pt": raw.get("connector_width_pt"),
                "connector_arrow": raw.get("connector_arrow", True),
            }
        )
    warnings.extend(_validate_no_overlaps(regions))
    coverage = sum(item["bbox_px"][2] * item["bbox_px"][3] for item in regions) / (
        CANVAS[0] * CANVAS[1]
    )
    if coverage < RECOMMENDED_REGISTERED_COVERAGE:
        warnings.append(
            f"registered coverage {coverage:.3f} is below the recommended {RECOMMENDED_REGISTERED_COVERAGE}"
        )
    return regions, warnings


def _segment_crosses_box(start: list[float], end: list[float], box: list[float]) -> bool:
    """Test actual line/rectangle intersection, not broad union-box overlap."""
    x, y, width, height = box
    dx, dy = end[0] - start[0], end[1] - start[1]
    low, high = 0.0, 1.0
    for direction, distance in ((-dx, start[0] - x), (dx, x + width - start[0]),
                                (-dy, start[1] - y), (dy, y + height - start[1])):
        if abs(direction) < 1e-9:
            if distance < 0:
                return False
            continue
        ratio = distance / direction
        if direction < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return False
    return True


def _compile_connectors(regions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Export only explicit HTML relationships as small native connector records.

    No keyword inference: three visual masses are not automatically the four
    stages named in a paragraph. An unmapped relationship stays a warning.
    """
    visuals = {item["region_id"]: item for item in regions if item["role"] == "visual"}
    text_boxes = [item["text_bbox_px"] for item in regions if item.get("content_key") and item.get("text_bbox_px")]
    connectors, warnings, seen = [], [], set()
    for source_id, source in visuals.items():
        for target_id in re.split(r"[\s,]+", str(source.get("connect_to") or "").strip()):
            if not target_id:
                continue
            pair = (source_id, target_id)
            if pair in seen:
                continue
            seen.add(pair)
            if target_id == source_id or target_id not in visuals:
                warnings.append(f"ignored unmapped connector {source_id}->{target_id}")
                continue
            a, b = source["bbox_px"], visuals[target_id]["bbox_px"]
            ac = [a[0] + a[2] / 2, a[1] + a[3] / 2]
            bc = [b[0] + b[2] / 2, b[1] + b[3] / 2]
            if abs(bc[0] - ac[0]) >= abs(bc[1] - ac[1]):
                sign = 1 if bc[0] > ac[0] else -1
                start, end = [ac[0] + sign * a[2] / 2, ac[1]], [bc[0] - sign * b[2] / 2, bc[1]]
            else:
                sign = 1 if bc[1] > ac[1] else -1
                start, end = [ac[0], ac[1] + sign * a[3] / 2], [bc[0], bc[1] - sign * b[3] / 2]
            if math.dist(start, end) < 8 or any(_segment_crosses_box(start, end, box) for box in text_boxes):
                warnings.append(f"connector {source_id}->{target_id} has insufficient text-free clearance")
                continue
            color = str(source.get("connector_color") or "5B728A").lstrip("#").upper()
            if not re.fullmatch(r"[0-9A-F]{6}", color):
                color = "5B728A"
            try:
                width = float(source.get("connector_width_pt") or 2)
                width = max(0.5, min(3.0, width)) if math.isfinite(width) else 2.0
            except (TypeError, ValueError):
                width = 2.0
            connectors.append({"from_region_id": source_id, "to_region_id": target_id,
                               "from_px": [round(v, 3) for v in start], "to_px": [round(v, 3) for v in end],
                               "color": color, "width_pt": width,
                               "arrow": source.get("connector_arrow") is not False})
    return connectors, warnings


def _parse_display_content(
    content_key: str, display_text: str, raw: dict[str, Any]
) -> dict[str, Any]:
    rich_paragraphs = _normalize_rich_paragraphs(raw.get("rich_paragraphs"))
    if rich_paragraphs:
        return {
            "format": "rich_text",
            "text": display_text,
            "bullet_items": [],
            "paragraphs": rich_paragraphs,
        }
    list_items = [
        str(value).strip()
        for value in raw.get("list_items", [])
        if str(value).strip()
    ]
    if list_items:
        lead = str(raw.get("list_lead") or "").strip()
        return {
            "format": str(raw.get("list_format") or "unordered_list"),
            "text": lead,
            "bullet_items": ([lead] if lead else []) + list_items,
        }
    parsed = parse_text_block(content_key, display_text)
    if parsed["format"] == "paragraph":
        return {"format": "paragraph", "text": parsed["text"], "bullet_items": []}
    lead = str(parsed.get("lead") or "").strip()
    items = [str(value).strip() for value in parsed.get("items", []) if str(value).strip()]
    return {
        "format": parsed["format"],
        "text": lead,
        "bullet_items": ([lead] if lead else []) + items,
    }


def _normalize_rich_paragraphs(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    paragraphs = []
    for paragraph in raw:
        if not isinstance(paragraph, dict):
            continue
        kind = str(paragraph.get("kind") or "body").strip().lower()
        if kind not in {"lead", "body", "bullet"}:
            kind = "body"
        runs = []
        for run in paragraph.get("runs") or []:
            if not isinstance(run, dict):
                continue
            text = str(run.get("text") or "")
            if not text.strip():
                continue
            try:
                font_scale = float(run.get("font_scale") or 1.0)
            except (TypeError, ValueError):
                font_scale = 1.0
            runs.append(
                {
                    "text": text,
                    "bold": bool(run.get("bold", False)),
                    "italic": bool(run.get("italic", False)),
                    "font_scale": round(max(0.8, min(1.3, font_scale)), 3),
                }
            )
        if runs:
            item = {"kind": kind, "runs": runs}
            box = paragraph.get("bbox_px")
            if (isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                    and box[2] > 0 and box[3] > 0):
                item["bbox_px"] = [round(float(v), 4) for v in box]
                leading = paragraph.get("line_height_px")
                if isinstance(leading, (int, float)) and math.isfinite(leading) and leading > 0:
                    item["line_height_px"] = round(float(leading), 4)
                if paragraph.get("line_count"):
                    item["line_count"] = max(1, int(paragraph["line_count"]))
                if paragraph.get("bullet_indent_px"):
                    item["bullet_indent_px"] = float(paragraph["bullet_indent_px"])
                item["text_alignment"] = str(paragraph.get("text_alignment") or "left")
            paragraphs.append(item)
    return paragraphs


def _normalize_typography(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        font_size_px = max(1.0, float(raw.get("font_size_px") or 16))
    except (TypeError, ValueError):
        font_size_px = 16.0
    try:
        font_weight_value = int(float(raw.get("font_weight") or 400))
    except (TypeError, ValueError):
        font_weight_value = 700 if str(raw.get("font_weight")).lower() == "bold" else 400
    try:
        line_height_px = float(raw.get("line_height_px") or 0)
    except (TypeError, ValueError):
        line_height_px = 0
    line_height_ratio = line_height_px / font_size_px if line_height_px > 0 else 1.08
    alignment = str(raw.get("text_alignment") or "left").lower()
    if alignment not in {"left", "center", "right", "justify"}:
        alignment = "left"
    result = {
        "font_size_px": round(font_size_px, 3),
        "font_size_pt": round(font_size_px * POINTS_PER_CANVAS_PIXEL, 3),
        "font_family": str(raw.get("font_family") or "Noto Sans"),
        "font_weight": "bold" if font_weight_value >= 600 else "regular",
        "font_style": "italic" if str(raw.get("font_style") or "").lower() == "italic" else "normal",
        "line_height_ratio": round(max(0.9, min(1.6, line_height_ratio)), 3),
        "text_alignment": alignment,
        "color": _css_color_to_hex(str(raw.get("text_color") or "")),
    }
    if raw.get("text_line_count") and line_height_px > 0:
        result["line_count"] = max(1, int(raw["text_line_count"]))
        result["line_height_px"] = round(line_height_px, 4)
    return result


def _css_color_to_hex(value: str) -> str:
    channels = re.findall(r"\d+(?:\.\d+)?", value)
    if len(channels) < 3:
        return "111111"
    rgb = [max(0, min(255, round(float(channel)))) for channel in channels[:3]]
    return "".join(f"{channel:02X}" for channel in rgb)


def _validate_modules(
    raw_modules: list[dict[str, Any]], regions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    modules = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_modules, start=1):
        module_id = str(raw.get("module_id") or f"module_{index:02d}").strip()
        if module_id in seen:
            warnings.append(f"duplicate data-module-id: {module_id}")
            continue
        seen.add(module_id)
        bbox, bbox_warning = _normalize_bbox(raw.get("bbox_px", []))
        if bbox_warning:
            warnings.append(f"{module_id} {bbox_warning}")
        modules.append(
            {
                "module_id": module_id,
                "bbox_px": bbox,
                "parent_module_id": raw.get("parent_module_id"),
                "region_ids": list(raw.get("region_ids") or []),
            }
        )
    if not 2 <= len(modules) <= 6:
        warnings.append(f"expected 2-6 flat HTML modules, found {len(modules)}")
    nested = [item["module_id"] for item in modules if item.get("parent_module_id")]
    if nested:
        warnings.append(f"nested HTML modules are not allowed: {sorted(nested)}")
    by_id = {item["module_id"]: item for item in modules}
    for region in regions:
        module_id = str(region.get("module_id") or "").strip()
        if module_id not in by_id:
            warnings.append(f"{region['region_id']} is not inside a registered module")
            continue
        if not _contains(by_id[module_id]["bbox_px"], region["bbox_px"]):
            warnings.append(f"{region['region_id']} extends outside module {module_id}")
    for index, left in enumerate(modules):
        for right in modules[index + 1 :]:
            if _intersection_area(left["bbox_px"], right["bbox_px"]) > 1:
                warnings.append(f"modules overlap: {left['module_id']} and {right['module_id']}")
    coverage = sum(item["bbox_px"][2] * item["bbox_px"][3] for item in modules) / (
        CANVAS[0] * CANVAS[1]
    )
    if modules and coverage < 0.62:
        warnings.append(f"module coverage {coverage:.3f} is below the recommended 0.620")
    return modules, warnings


def _normalize_holder(
    raw: dict[str, Any] | None, region_id: str
) -> dict[str, Any] | None:
    if not raw:
        return None
    bbox, _ = _normalize_bbox(raw.get("bbox_px", []))
    return {
        "holder_id": str(raw.get("holder_id") or f"holder_{region_id}"),
        "bbox_px": bbox,
        "text_region_id": region_id,
        "source": str(raw.get("source") or "html_visible_container"),
        "geometry_authority": "explicit_html_holder",
    }


def _deduplicate_holders(text_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    holders: dict[str, dict[str, Any]] = {}
    for region in text_regions:
        holder = region.get("holder")
        if holder is not None:
            holders[holder["holder_id"]] = holder
    return sorted(holders.values(), key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))


def _compile_visual_generation_groups(
    *,
    visual_regions: list[dict[str, Any]],
    visual_groups_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    visual_by_id = {item["region_id"]: item for item in visual_regions}
    proposed = []
    if visual_groups_path.is_file():
        try:
            payload = read_json(visual_groups_path)
            proposed = payload.get("groups", []) if isinstance(payload, dict) else []
        except (ValueError, TypeError) as exc:
            warnings.append(f"visual grouping sidecar could not be parsed; used HTML visual briefs: {exc}")
    proposed_by_region: dict[str, dict[str, Any]] = {}
    if isinstance(proposed, list):
        for raw in proposed:
            if not isinstance(raw, dict):
                continue
            region_ids = [str(value).strip() for value in raw.get("region_ids", [])]
            if len(region_ids) == 1 and region_ids[0] in visual_by_id:
                proposed_by_region[region_ids[0]] = raw
            elif region_ids:
                warnings.append(
                    "ignored a multi-region visual group; HTML visual regions map 1:1 to assets"
                )
    elif visual_regions:
        warnings.append("visual generation grouping sidecar was malformed; used HTML visual briefs")
    compiled = []
    for visual in visual_regions:
        region_id = visual["region_id"]
        raw = proposed_by_region.get(region_id, {})
        compiled.append(
            {
                "group_id": f"visual_group_{region_id}",
                "source_region_ids": [region_id],
                "bbox_px": visual["bbox_px"],
                "relation_type": "single_visual",
                "semantic_description": str(raw.get("semantic_description") or "").strip()
                or visual["visual_brief"],
                "use_structure_reference": False,
                "structure_reference": None,
            }
        )
    return compiled, warnings


def _normalize_bbox(values: list[Any]) -> tuple[list[int], str | None]:
    try:
        x, y, width, height = [float(value) for value in values[:4]]
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError
    except (TypeError, ValueError):
        return [48, 48, 256, 128], "had an invalid bbox; used a visible recovery box"
    resolved_width = max(1, min(CANVAS[0], round(width)))
    resolved_height = max(1, min(CANVAS[1], round(height)))
    left = max(0, min(CANVAS[0] - resolved_width, round(x)))
    top = max(0, min(CANVAS[1] - resolved_height, round(y)))
    bbox = [left, top, resolved_width, resolved_height]
    original = [round(x), round(y), round(width), round(height)]
    return bbox, None if bbox == original else "bbox was normalized to the canvas"


def _find_recovery_text_bbox(raw_regions: list[dict[str, Any]]) -> list[int]:
    """Place only a missing text block, preferring an unoccupied existing gap."""
    occupied = [_normalize_bbox(raw.get("bbox_px", []))[0] for raw in raw_regions]
    width, height = 640, 120
    candidates = [
        [x, y, width, height]
        for y in range(48, CANVAS[1] - height + 1, 24)
        for x in (48, CANVAS[0] - width - 48)
    ]
    return min(candidates, key=lambda bbox: sum(_intersection_area(bbox, other) for other in occupied))


def _expected_content(source_text: dict[str, Any]) -> dict[str, tuple[str, str]]:
    expected = {"title": ("title", source_text["title"])}
    if source_text["subtitle"] is not None:
        expected["subtitle"] = ("subtitle", source_text["subtitle"])
    expected.update(
        {
            key: ("body", value)
            for key, value in source_text.items()
            if TEXT_FIELD_PATTERN.fullmatch(key)
        }
    )
    return expected


def _validate_no_overlaps(regions: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            if _intersection_area(left["bbox_px"], right["bbox_px"]) > 1:
                warnings.append(f"regions overlap: {left['region_id']} and {right['region_id']}")
    return warnings


def _intersection_area(left: list[int], right: list[int]) -> int:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[0] + left[2], right[0] + right[2])
    y1 = min(left[1] + left[3], right[1] + right[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _contains(outer: list[int], inner: list[int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[0] + outer[2] >= inner[0] + inner[2]
        and outer[1] + outer[3] >= inner[1] + inner[3]
    )


def _text_hash(source_text: dict[str, Any]) -> str:
    values = [source_text["title"]]
    if source_text["subtitle"] is not None:
        values.append(source_text["subtitle"])
    values.extend(
        value for key, value in source_text.items() if TEXT_FIELD_PATTERN.fullmatch(key)
    )
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def _display_text_hash(text_regions: list[dict[str, Any]]) -> str:
    values = [str(region.get("display_text") or "") for region in text_regions]
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


_BROWSER_EXTRACTION_SCRIPT = r"""el => {
  const slideRect = el.getBoundingClientRect();
  const registeredText = node => [...node.querySelectorAll('[data-content-key]')];
  const registeredVisual = node => node.querySelector('[data-role="visual"]');
  const explicitHolder = node => {
    let current = node;
    while (current && current !== el) {
      const registeredCount = registeredText(current).length +
        (current.matches('[data-content-key]') ? 1 : 0);
      if (current.hasAttribute('data-holder-id') &&
          registeredCount === 1 && !registeredVisual(current)) return current;
      current = current.parentElement;
    }
    return null;
  };
  const richParagraphs = node => {
    const baseSize = parseFloat(getComputedStyle(node).fontSize) || 16;
    // A registered block may contain a grid/flex list. Its DOM paragraphs are
    // independent native cells; flattening them changes both layout and fit.
    let parts;
    if (node.dataset.textReworked === 'true') {
      parts = [...node.querySelectorAll(':scope > [data-text-part]')];
    } else if (node.matches('ul,ol') || node.querySelector('ul,ol')) {
      parts = node.matches('ul,ol') ? [...node.querySelectorAll(':scope > li')] :
        [...node.childNodes].flatMap(child => {
          if (!child.textContent.trim()) return [];
          return child.nodeType === Node.ELEMENT_NODE && child.matches('ul,ol') ?
            [...child.querySelectorAll(':scope > li')] : [child];
        });
    } else return [];
    return parts.map(part => {
      const walker = document.createTreeWalker(part, NodeFilter.SHOW_TEXT);
      const runs = [];
      const textNodes = [];
      if (part.nodeType === Node.TEXT_NODE) textNodes.push(part);
      else while (walker.nextNode()) textNodes.push(walker.currentNode);
      for (const textNode of textNodes) {
        if (textNode.parentElement?.closest('[data-generated-bullet="true"]')) continue;
        const text = String(textNode.nodeValue || '').replace(/\s+/g, ' ');
        if (!text.trim()) continue;
        const parent = textNode.parentElement || part;
        const runStyle = getComputedStyle(parent);
        const weight = parseFloat(runStyle.fontWeight) || (runStyle.fontWeight === 'bold' ? 700 : 400);
        const runSize = parseFloat(runStyle.fontSize) || baseSize;
        runs.push({
          text,
          bold: weight >= 600 || Boolean(parent.closest('strong,b')),
          italic: runStyle.fontStyle === 'italic' || Boolean(parent.closest('em,i')),
          font_scale: runSize / baseSize
        });
      }
      const element = part.nodeType === Node.ELEMENT_NODE ? part : part.parentElement;
      const range = document.createRange(); range.selectNodeContents(part);
      const rect = part.nodeType === Node.ELEMENT_NODE ? part.getBoundingClientRect() : range.getBoundingClientRect();
      const style = getComputedStyle(element);
      const fontSize = parseFloat(style.fontSize) || baseSize;
      const cssHeight = parseFloat(style.height);
      const usedHeight = style.boxSizing === 'border-box' ? cssHeight : cssHeight +
        ['paddingTop','paddingBottom','borderTopWidth','borderBottomWidth'].reduce((sum,key)=>sum+(parseFloat(style[key])||0),0);
      const sy = part.nodeType === Node.ELEMENT_NODE && usedHeight > 0 ? rect.height / usedHeight : 1;
      const lineHeight = parseFloat(style.lineHeight);
      const kind=element.dataset.textPart || (element.matches('li') ? 'bullet' : 'body');
      const indent=kind==='bullet' ? (parseFloat(style.paddingLeft)||fontSize)*sy : 0;
      const outsideMarker=element.matches('li') && style.listStylePosition!=='inside' ? indent : 0;
      const lineTops=[];
      for(const textNode of textNodes){const r=document.createRange();r.selectNodeContents(textNode);
        for(const box of r.getClientRects())if(box.width>0)lineTops.push(box.top);}
      let lineCount=0,last=-Infinity;
      for(const top of lineTops.sort((a,b)=>a-b)){if(top-last>fontSize*sy*.5){lineCount++;last=top;}}
      return {kind, runs,
        bbox_px: [rect.left-slideRect.left-outsideMarker, rect.top-slideRect.top, rect.width+outsideMarker, rect.height],
        line_height_px: (Number.isFinite(lineHeight) ? lineHeight : fontSize*1.2)*sy,
        line_count:Math.max(1,lineCount),bullet_indent_px:indent,
        text_alignment: style.textAlign};
    }).filter(paragraph => paragraph.runs.length);
  };
  const textLineCount = (node, fontSize) => {
    const tops=[],walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT);
    while(walker.nextNode()){
      if(!walker.currentNode.textContent.trim())continue;
      const r=document.createRange();r.selectNodeContents(walker.currentNode);
      for(const box of r.getClientRects())if(box.width>0)tops.push(box.top);
    }
    let count=0,last=-Infinity;
    for(const top of tops.sort((a,b)=>a-b)){if(top-last>fontSize*.5){count++;last=top;}}
    return Math.max(1,count);
  };
  const regions = [...el.querySelectorAll('[data-region-id]')].map(node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    // offsetHeight/Width round to integers. Treating a 64.5px untransformed
    // title as scaled by64.5/65 silently changes its exported shared font.
    const usedSize = (axis) => {
      const css = parseFloat(axis === 'x' ? style.width : style.height);
      if (!Number.isFinite(css) || css <= 0) return axis === 'x' ? node.offsetWidth : node.offsetHeight;
      if (style.boxSizing === 'border-box') return css;
      const sides = axis === 'x' ? ['Left','Right'] : ['Top','Bottom'];
      return css+sides.reduce((sum,side)=>sum+(parseFloat(style[`padding${side}`])||0)+(parseFloat(style[`border${side}Width`])||0),0);
    };
    const unscaledWidth=usedSize('x'), unscaledHeight=usedSize('y');
    const scaleX = unscaledWidth > 0 ? rect.width / unscaledWidth : 1;
    const scaleY = unscaledHeight > 0 ? rect.height / unscaledHeight : 1;
    const insetTop = ((parseFloat(style.borderTopWidth) || 0) + (parseFloat(style.paddingTop) || 0)) * scaleY;
    const insetRight = ((parseFloat(style.borderRightWidth) || 0) + (parseFloat(style.paddingRight) || 0)) * scaleX;
    const insetBottom = ((parseFloat(style.borderBottomWidth) || 0) + (parseFloat(style.paddingBottom) || 0)) * scaleY;
    const insetLeft = ((parseFloat(style.borderLeftWidth) || 0) + (parseFloat(style.paddingLeft) || 0)) * scaleX;
    let verticalAlignment = 'top';
    if (style.display.includes('flex')) {
      const alignment = style.flexDirection.startsWith('column') ? style.justifyContent : style.alignItems;
      if (alignment === 'center') verticalAlignment = 'mid';
      else if (alignment === 'flex-end' || alignment === 'end') verticalAlignment = 'bottom';
    } else if (style.display === 'table-cell') {
      if (style.verticalAlign === 'middle') verticalAlignment = 'mid';
      else if (style.verticalAlign === 'bottom') verticalAlignment = 'bottom';
    }
    const range = document.createRange();
    range.selectNodeContents(node);
    const inkRect = range.getBoundingClientRect();
    const holder = node.dataset.contentKey ? explicitHolder(node) : null;
    const holderRect = holder ? holder.getBoundingClientRect() : null;
    const listNodes = node.matches('ul,ol') ? [node] : [...node.querySelectorAll('ul,ol')];
    const listItems = listNodes.flatMap(list =>
      [...list.querySelectorAll(':scope > li')].map(item => (item.innerText ?? item.textContent ?? '').trim())
    ).filter(Boolean);
    let listLead = '';
    if (!node.matches('ul,ol') && listNodes.length) {
      const clone = node.cloneNode(true);
      clone.querySelectorAll('ul,ol').forEach(list => list.remove());
      listLead = (clone.textContent || '').replace(/\s+/g, ' ').trim();
    }
    const module = node.closest('[data-module-id]');
    const parsedLineHeight = parseFloat(style.lineHeight);
    return {
      region_id: node.dataset.regionId || '',
      role: node.dataset.role || '',
      content_key: node.dataset.contentKey || null,
      text_surface: node.dataset.textSurface || null,
      visual_brief: node.dataset.visualBrief || null,
      connect_to: node.dataset.connectTo || '',
      connector_color: node.dataset.connectorColor || null,
      connector_width_pt: node.dataset.connectorWidth || null,
      connector_arrow: node.dataset.connectorArrow !== 'false',
      equivalent_group: node.dataset.equivalentGroup || null,
      generation_group: node.dataset.generationGroup || null,
      module_id: module?.dataset.moduleId || null,
      bbox_px: [rect.left-slideRect.left, rect.top-slideRect.top, rect.width, rect.height],
      original_bbox_px: (() => {try {return JSON.parse(node.dataset.qualityOriginalBbox || 'null');} catch {return null;}})(),
      text_content_bbox_px: [rect.left-slideRect.left+insetLeft, rect.top-slideRect.top+insetTop,
        Math.max(1,rect.width-insetLeft-insetRight),Math.max(1,rect.height-insetTop-insetBottom)],
      text_box_insets_px: [insetTop,insetRight,insetBottom,insetLeft],
      vertical_alignment: verticalAlignment,
      text_ink_bbox_px: [inkRect.left-slideRect.left, inkRect.top-slideRect.top, inkRect.width, inkRect.height],
      text_content: node.textContent.trim(),
      inner_text: (node.innerText ?? node.textContent ?? '').trim(),
      list_items: listItems,
      list_lead: listLead,
      list_format: listNodes[0]?.tagName === 'OL' ? 'ordered_list' : 'unordered_list',
      rich_paragraphs: richParagraphs(node),
      font_size_px: parseFloat(style.fontSize) * scaleY,
      font_family: style.fontFamily.split(',')[0].trim().replace(/^["']|["']$/g, ''),
      font_weight: style.fontWeight,
      font_style: style.fontStyle,
      line_height_px: Number.isFinite(parsedLineHeight) ? parsedLineHeight * scaleY : null,
      text_line_count: textLineCount(node,parseFloat(style.fontSize)*scaleY),
      text_alignment: style.textAlign,
      text_color: style.color,
      client_width: node.clientWidth,
      client_height: node.clientHeight,
      scroll_width: node.scrollWidth,
      scroll_height: node.scrollHeight,
      overflow: node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1,
      display: style.display,
      visibility: style.visibility,
      z_index: style.zIndex,
      holder: holderRect ? {
        holder_id: holder.dataset.holderId || `holder_${node.dataset.regionId || node.dataset.contentKey}`,
        bbox_px: [holderRect.left-slideRect.left, holderRect.top-slideRect.top, holderRect.width, holderRect.height],
        source: 'explicit_html_holder'
      } : null
    };
  });
  const extraText = [...el.querySelectorAll('*')]
    .filter(node => node.children.length === 0 && (node.innerText ?? node.textContent ?? '').trim())
    .filter(node => !node.closest('[data-content-key]'))
    .map(node => (node.innerText ?? node.textContent ?? '').trim());
  const modules = [...el.querySelectorAll('[data-module-id]')].map(node => {
    const rect = node.getBoundingClientRect();
    return {
      module_id: node.dataset.moduleId || '',
      parent_module_id: node.parentElement?.closest('[data-module-id]')?.dataset.moduleId || null,
      bbox_px: [rect.left-slideRect.left, rect.top-slideRect.top, rect.width, rect.height],
      region_ids: [...node.querySelectorAll('[data-region-id]')]
        .filter(region => region.closest('[data-module-id]') === node)
        .map(region => region.dataset.regionId || '').filter(Boolean)
    };
  });
  const geometryRepairs=[...el.querySelectorAll('[data-quality-geometry-repair="true"]')].map(node => ({
    unit_id:node.dataset.moduleId || node.dataset.regionId,
    original_bbox_px:(() => {try{return JSON.parse(node.dataset.qualityOriginalBbox || 'null');}catch{return null;}})(),
    bbox_px:(() => {const r=node.getBoundingClientRect();return [r.left-slideRect.left,r.top-slideRect.top,r.width,r.height];})()
  }));
  return {slide_size:[slideRect.width,slideRect.height],regions,modules,extra_text:extraText,geometry_repairs:geometryRepairs};
}"""
