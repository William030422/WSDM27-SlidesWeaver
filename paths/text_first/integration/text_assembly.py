from __future__ import annotations

import shutil
import subprocess
import math
import tempfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

from v9_deck_planning.json_utils import read_json, write_json

from .config import PipelineConfig
from .contrast import apply_background_aware_text_colors, resolved_text_bbox_px
from .text_generation import ASCII_ARROW_REPLACEMENTS
from .render_quality import inspect_pdf_text
from .decorative_surface import prepare_decorative_surface, reframe_decorative_surface

ROOT = Path(__file__).resolve().parents[1]


def run_reused_text_assembly(
    *,
    html_compile_dir: Path,
    text_generation_dir: Path,
    final_background_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_vlm: bool = False,
    prepared_pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble exact HTML geometry, generated raster layers, and editable text."""
    del mock_vlm
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = prepared_pages if prepared_pages is not None else [
        prepare_text_assembly_slide(
            html_compile_dir=html_compile_dir, text_generation_dir=text_generation_dir,
            final_background_dir=final_background_dir, output_dir=output_dir,
            slide_id=slide_id, config=config,
        ) for slide_id in slide_ids
    ]
    if [page["slide_id"] for page in pages] != slide_ids:
        raise ValueError("Committed assembly plans must preserve selected slide order")
    compilation_slides = [page["compilation"] for page in pages]
    expected_raster_objects = [page["expected_raster"] for page in pages]
    contrast_slides = [page["contrast"] for page in pages]
    failed_slides = [number for page in pages for number in page["failed_slides"]]
    quality_warnings = [warning for page in pages for warning in page["warnings"]]

    compilation_path = output_dir / "deck_compilation.json"
    write_json(
        compilation_path, {"schema_version": "2.0", "slides": compilation_slides}
    )
    pptx_path = output_dir / "deck.pptx"
    text_fit_path = output_dir / "text_fit_report.json"
    contrast_path = output_dir / "contrast_report.json"
    write_json(
        contrast_path,
        {
            "schema_version": "1.0",
            "status": "strict_pass" if all(item["status"] == "strict_pass" for item in contrast_slides) else "degraded",
            "sampling_scope": "final composited background including local contrast checks",
            "slides": contrast_slides,
        },
    )
    _compile_deck(compilation_path, pptx_path, text_fit_path)
    raster_editability_path = output_dir / "raster_editability_report.json"
    raster_editability = _validate_editable_raster_objects(
        pptx_path, expected_raster_objects
    )
    write_json(raster_editability_path, raster_editability)
    if raster_editability["status"] != "strict_pass":
        quality_warnings.append({"raster_editability": raster_editability})
    pdf_path = output_dir / "deck.pdf"
    rendered_dir = output_dir / "rendered_slides"
    render_report = _render_and_validate(
        pptx_path,
        pdf_path,
        rendered_dir,
        len(slide_ids),
    )
    rendered_text = inspect_pdf_text(pdf_path, {
        record['slide_id']: read_json(Path(record['plan'])) for record in compilation_slides
    })
    render_report['rendered_text'] = rendered_text
    render_report['warnings'].extend(rendered_text['warnings'])
    if rendered_text['warnings']:
        render_report['status'] = 'degraded'
    write_json(output_dir / "pptx_render_check.json", render_report)
    quality_warnings.extend(render_report.get("warnings", []))
    _copy_rendered_previews(rendered_dir, output_dir, slide_ids)
    montage_path = output_dir / "deck_montage.png"
    _create_montage(rendered_dir, montage_path)
    text_fit = read_json(text_fit_path)
    unresolved = {
        slide["slide_id"]: slide["overflow_region_ids"]
        for slide in text_fit["slides"]
        if slide["overflow_region_ids"]
    }
    geometry_report = _validate_final_geometry(compilation_slides, text_fit)
    geometry_path = output_dir / "geometry_quality_report.json"
    write_json(geometry_path, geometry_report)
    quality_warnings.extend(geometry_report["issues"])
    if unresolved:
        quality_warnings.append({"unresolved_text_overflow": unresolved})
    manifest = {
        "stage": "final_html_geometry_assembly",
        "status": "succeeded",
        "quality_status": "degraded" if quality_warnings or failed_slides else "strict_pass",
        "warnings": quality_warnings,
        "slide_count": len(slide_ids),
        "failed_slides": failed_slides,
        "deck_font": config.deck_font,
        "output_dir": str(output_dir.resolve()),
        "topology_used": False,
        "critic_rounds": {"status": "removed", "reason": "final HTML owns geometry"},
        "slides": compilation_slides,
        "artifacts": {
            "pptx": str(pptx_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "rendered_slides": str(rendered_dir.resolve()),
            "montage": str(montage_path.resolve()),
            "text_fit_report": str(text_fit_path.resolve()),
            "contrast_report": str(contrast_path.resolve()),
            "raster_editability_report": str(raster_editability_path.resolve()),
            "geometry_quality_report": str(geometry_path.resolve()),
            "pptx_render_check": str((output_dir / "pptx_render_check.json").resolve()),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def prepare_text_assembly_slide(
    *, html_compile_dir: Path, text_generation_dir: Path, final_background_dir: Path,
    output_dir: Path, slide_id: str, config: PipelineConfig,
) -> dict[str, Any]:
    """Finalize one page's native plan without compiling or reviewing the deck."""
    parsed_text = read_json(text_generation_dir / "parsed_text_plan.json")
    parsed_by_slide = {item["slide_id"]: item for item in parsed_text["slides"]}
    compilation_slides, expected_raster_objects, contrast_slides, failed_slides = [], [], [], []
    quality_warnings: list[dict[str, Any]] = []
    layout = read_json(html_compile_dir / "slides" / slide_id / "html_layout.json")
    layer_manifest = read_json(
        final_background_dir / slide_id / "layer_manifest.json"
    )
    quality_warnings.extend(
        {"slide_id": slide_id, "stage": "raster_layers", "detail": warning}
        for warning in layer_manifest.get("quality_warnings", [])
    )
    failure_placeholder = layer_manifest.get("failure_placeholder")
    content = _content_by_key(parsed_by_slide[slide_id])
    holders_by_region = {
        layer["text_region_id"]: layer
        for layer in layer_manifest["layers"]
        if layer.get("kind") == "holder" and layer.get("text_region_id")
    }
    fallback_holder_ids = {
        layer["layer_id"]
        for layer in holders_by_region.values()
        if not Path(str(layer.get('path') or '')).is_file()
    }
    elements = []
    for region in layout["text_regions"]:
        key = region["content_key"]
        if key not in content:
            quality_warnings.append({"slide_id": slide_id, "unknown_content_key": key})
            continue
        item = region.get("display_content") or content[key]
        holder_layer = holders_by_region.get(region["region_id"])
        holder_binding = None
        native_holder = None
        if holder_layer is not None:
            if holder_layer["layer_id"] not in fallback_holder_ids:
                holder_binding = {
                    "layer_id": holder_layer["layer_id"],
                    "holder_bbox_px": holder_layer["bbox_px"],
                    "text_safe_rect": holder_layer.get("text_safe_rect") or {'x': 0, 'y': 0, 'width': 1, 'height': 1},
                    "asset_path": holder_layer.get('path'),
                    "repair_interior": (not _usable_text_safe_rect(holder_layer.get('text_safe_rect'))
                                        or (holder_layer.get('holder_compatibility') or {}).get('status') == 'requires_quiet_backing'),
                }
            else:
                native_holder = {"bbox_px": holder_layer["bbox_px"]}
        elements.append(
            {
                "region_id": region["region_id"],
                "content_key": key,
                "role": region["role"],
                "bbox_px": region["text_bbox_px"],
                "holder_binding": holder_binding,
                "native_holder": native_holder,
                "presentation_mode": region["presentation_mode"],
                "text_alignment": region.get("text_alignment", "left"),
                "vertical_alignment": region.get("vertical_alignment") or (region.get("typography") or {}).get("vertical_alignment", "top"),
                "text": _normalize_font_compatible_text(item["text"]),
                "bullet_items": [
                    _normalize_font_compatible_text(value)
                    for value in item.get("bullet_items", [])
                ],
                "rich_text": _normalize_rich_text(item.get("paragraphs")),
                "source_text": region.get("source_text") or content[key]["text"],
                "display_text": region.get("display_text") or item["text"],
                "copy_changed": bool(region.get("copy_changed", False)),
                "font": _font_from_html(region, config),
                "expected_line_count": (region.get("typography") or {}).get("line_count", 1),
            }
        )
    if failure_placeholder is not None:
        slide_number = int(failure_placeholder["slide_number"])
        failed_slides.append(slide_number)
        elements = [
            {
                "region_id": "generation_failure",
                "content_key": "generation_failure",
                "role": "body",
                "bbox_px": [430, 748, 676, 54],
                "holder_binding": None,
                "native_holder": None,
                "presentation_mode": "direct_text",
                "text_alignment": "center",
                "text": f"Slide {slide_number} generation failed",
                "bullet_items": [],
                "rich_text": [],
                "source_text": "",
                "display_text": f"Slide {slide_number} generation failed",
                "copy_changed": False,
                "font": {
                    "face": config.deck_font,
                    "size_pt": 16,
                    "minimum_pt": 16,
                    "weight": "regular",
                    "color": "464646",
                },
            }
        ]
    plan = {
        "schema_version": "3.0",
        "slide_id": slide_id,
        "canvas_px": layout["canvas"],
        "background_path": layer_manifest["base_background"],
        "raster_layers": [
            layer
            for layer in layer_manifest["layers"]
            if layer.get("layer_id") not in fallback_holder_ids
        ],
        "groups": []
        if failure_placeholder is not None
        else _equivalent_groups(layout["text_regions"]),
        "elements": elements,
        "layout_authority": "final_html_layout",
        "topology_used": False,
        "holder_asset_selection_used": False,
        "failure_placeholder": failure_placeholder,
        "connectors": layout.get('connectors', []),
    }
    slide_dir = output_dir / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    for element in elements:
        _resolve_element_text_geometry(element, quality_warnings, slide_id, output_dir=slide_dir)
        holder = element.get('holder_binding') or {}
        if holder.get('quiet_frame_path'):
            for layer in plan['raster_layers']:
                if layer['layer_id'] == element['holder_binding']['layer_id']:
                    layer['path'] = element['holder_binding']['quiet_frame_path']
    contrast_report = apply_background_aware_text_colors(plan)
    contrast_slides.append(contrast_report)
    if contrast_report["status"] != "strict_pass":
        quality_warnings.append({"slide_id": slide_id, "text_contrast": contrast_report["status"]})
    plan_path = slide_dir / "composition_plan.json"
    write_json(plan_path, plan)
    compilation_slides.append(
        {"slide_id": slide_id, "plan": str(plan_path.resolve())}
    )
    expected_raster_objects.append(
        {
            "slide_id": slide_id,
            "object_names": [f"background:{slide_id}"]
            + [
                f"raster:{layer['layer_id']}"
                for layer in layer_manifest["layers"]
                if layer.get("layer_id") not in fallback_holder_ids
            ],
        }
    )

    return {"slide_id": slide_id, "compilation": compilation_slides[0],
            "expected_raster": expected_raster_objects[0], "contrast": contrast_slides[0],
            "failed_slides": failed_slides, "warnings": quality_warnings}


def _intersection_bbox(first: list[float], second: list[float]) -> list[float] | None:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return [left, top, right - left, bottom - top] if right > left and bottom > top else None


def _resolve_element_text_geometry(element: dict[str, Any], warnings: list[dict[str, Any]], slide_id: str, *, output_dir: Path | None = None) -> None:
    planned = [float(value) for value in element["bbox_px"]]
    safe = resolved_text_bbox_px(element)
    if element.get("holder_binding") or element.get("native_holder"):
        intersection = _intersection_bbox(planned, safe)
        # Keep the decorative raster frame. Its generated interior must not
        # reduce the committed HTML copy area; repair only the quiet backing.
        if (element.get('holder_binding') and (element['holder_binding'].get('repair_interior')
                or intersection is None or any(abs(a - b) > .5 for a, b in zip(intersection, planned)))):
            holder = element['holder_binding']
            outer = holder['holder_bbox_px']
            contained = _intersection_bbox(planned, outer)
            if contained is not None and all(abs(a - b) <= .5 for a, b in zip(contained, planned)):
                color = _holder_interior_color(element, holder)
                element['text_backing'] = {'bbox_px': planned, 'fill_color': color, 'overlay': True}
                if output_dir is not None and Path(str(holder.get('asset_path') or '')).is_file():
                    # Keep half of each existing rim when copy is close to an
                    # edge. Clipping a fixed8px pad to the outer box erased the
                    # border and made nine-slice reframing degenerate.
                    left = planned[0] - min(8, max(0, planned[0] - outer[0]) / 2)
                    top = planned[1] - min(8, max(0, planned[1] - outer[1]) / 2)
                    right = planned[0] + planned[2] + min(8, max(0, outer[0] + outer[2] - planned[0] - planned[2]) / 2)
                    bottom = planned[1] + planned[3] + min(8, max(0, outer[1] + outer[3] - planned[1] - planned[3]) / 2)
                    quiet = {'x': (left - outer[0]) / outer[2], 'y': (top - outer[1]) / outer[3],
                             'width': (right - left) / outer[2], 'height': (bottom - top) / outer[3]}
                    try:
                        surface = prepare_decorative_surface(Path(holder['asset_path']),
                            output_dir / f"{element['region_id']}_frame.png", quiet)
                        safe_rect = surface['text_safe_rect']
                        actual_safe = [outer[0] + outer[2] * safe_rect['x'], outer[1] + outer[3] * safe_rect['y'],
                                       outer[2] * safe_rect['width'], outer[3] * safe_rect['height']]
                        covered = _intersection_bbox(planned, actual_safe)
                        if covered is not None and all(abs(a - b) <= .5 for a, b in zip(covered, planned)):
                            holder['quiet_frame_path'] = surface['frame_path']
                            element['text_backing'] = {'bbox_px': list(outer), 'fill_color': surface['fill_color'],
                                                       'normalized_polygon': surface['normalized_polygon']}
                        else:
                            surface = reframe_decorative_surface(Path(holder['asset_path']),
                                output_dir / f"{element['region_id']}_frame.png", holder['text_safe_rect'], quiet)
                            holder['quiet_frame_path'] = surface['frame_path']
                            element['text_backing'] = {'bbox_px': list(outer), 'fill_color': surface['fill_color'],
                                                       'normalized_polygon': surface['normalized_polygon']}
                            element['holder_interior_repair'] = 'decorative_frame_reframed_to_reserved_text'
                    except ValueError as exc:
                        warnings.append({'slide_id': slide_id, 'region_id': element['region_id'],
                                         'decorative_interior_warning': str(exc)})
                element['resolved_text_bbox_px'] = planned
                element.setdefault('holder_interior_repair', 'decorative_frame_with_quiet_backing')
                return
        if intersection is None:
            # Retain the HTML text box rather than moving copy to unrelated artwork.
            warnings.append({"slide_id": slide_id, "region_id": element["region_id"], "holder_safe_rect_disjoint": True})
            element["holder_binding"] = None
            element["native_holder"] = {"bbox_px": planned}
            intersection = planned
        element["resolved_text_bbox_px"] = intersection
    else:
        element["resolved_text_bbox_px"] = planned


def _holder_interior_color(element: dict[str, Any], holder: dict[str, Any]) -> str:
    # The decorative frame stays raster; use a calm fill behind editable copy.
    # The final contrast solver sees the same backing as the PPTX renderer.
    asset_path = Path(str(holder.get('asset_path') or ''))
    if asset_path.is_file():
        with Image.open(asset_path).convert('RGBA') as image:
            safe = holder['text_safe_rect'] if _usable_text_safe_rect(holder['text_safe_rect']) else {'x': .1, 'y': .1, 'width': .8, 'height': .8}
            crop = image.crop((int(image.width * safe['x']), int(image.height * safe['y']),
                               int(image.width * (safe['x'] + safe['width'])),
                               int(image.height * (safe['y'] + safe['height']))))
            sample = crop.convert('RGB').resize((32, 32)).quantize(colors=6)
            dominant = max(sample.getcolors(), key=lambda value: value[0])[1]
            rgb = sample.getpalette()[dominant * 3:dominant * 3 + 3]
            return ''.join(f'{value:02X}' for value in rgb)
    planned = str((element.get('font') or {}).get('color') or '111111').lstrip('#')
    try:
        brightness = sum(int(planned[i:i+2], 16) for i in (0, 2, 4)) / 3
    except ValueError:
        brightness = 0
    return '14182B' if brightness > 160 else 'FAF5DD'


def _validate_final_geometry(slides: list[dict[str, Any]], text_fit: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking checks of resolved output geometry; no layout replanning."""
    issues = []
    fit_by_slide = {slide["slide_id"]: slide for slide in text_fit.get("slides", [])}
    for record in slides:
        plan = read_json(Path(record["plan"]))
        width, height = plan["canvas_px"]["width"], plan["canvas_px"]["height"]
        resolved = {
            item["region_id"]: item["resolved_text_bbox_px"]
            for item in fit_by_slide.get(plan["slide_id"], {}).get("elements", [])
        }
        text_boxes = [(element["region_id"], resolved.get(element["region_id"], element["bbox_px"])) for element in plan["elements"]]
        all_boxes = text_boxes + [(layer["layer_id"], layer["bbox_px"]) for layer in plan.get("raster_layers", [])]
        for identity, box in all_boxes:
            if not all(math.isfinite(float(value)) for value in box) or box[2] <= 0 or box[3] <= 0:
                issues.append({"slide_id": plan["slide_id"], "region_id": identity, "issue": "invalid_final_bbox"})
            elif box[0] < -0.5 or box[1] < -0.5 or box[0] + box[2] > width + 0.5 or box[1] + box[3] > height + 0.5:
                issues.append({"slide_id": plan["slide_id"], "region_id": identity, "issue": "out_of_canvas", "bbox_px": box})
        for index, (identity, box) in enumerate(text_boxes):
            for other_id, other in text_boxes[index + 1:]:
                overlap = _intersection_bbox(box, other)
                if overlap is not None and overlap[2] * overlap[3] > 1:
                    issues.append({"slide_id": plan["slide_id"], "region_ids": [identity, other_id], "issue": "text_box_overlap", "overlap_area_px2": overlap[2] * overlap[3]})
    return {"schema_version": "1.0", "status": "degraded" if issues else "strict_pass", "issues": issues, "blocking": False}


def _content_by_key(slide: dict[str, Any]) -> dict[str, dict[str, Any]]:
    content = {
        "title": {"text": slide["title"], "bullet_items": []},
    }
    if slide.get("subtitle") is not None:
        content["subtitle"] = {"text": slide["subtitle"], "bullet_items": []}
    for block in slide.get("text_blocks", []):
        if block["format"] == "paragraph":
            content[block["text_id"]] = {"text": block["text"], "bullet_items": []}
        else:
            lead = str(block.get("lead") or "").strip()
            items = [
                str(item).strip()
                for item in block.get("items", [])
                if str(item).strip()
            ]
            content[block["text_id"]] = {
                "text": lead,
                "bullet_items": ([lead] if lead else []) + items,
            }
    return content


def _validate_editable_raster_objects(
    pptx_path: Path, expected_slides: list[dict[str, Any]]
) -> dict[str, Any]:
    namespaces = {
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }
    slides = []
    warnings = []
    with ZipFile(pptx_path) as archive:
        for index, expected in enumerate(expected_slides, start=1):
            root = ElementTree.fromstring(archive.read(f"ppt/slides/slide{index}.xml"))
            object_names = [
                item.attrib.get("name", "")
                for item in root.findall(".//p:pic/p:nvPicPr/p:cNvPr", namespaces)
            ]
            required = expected["object_names"]
            if len(required) != len(set(required)):
                warnings.append({"slide_id": expected["slide_id"], "issue": "duplicate_raster_layer_ids"})
            if object_names != required:
                warnings.append({"slide_id": expected["slide_id"], "issue": "raster_object_contract_mismatch", "expected": required, "found": object_names})
            slides.append(
                {
                    "slide_id": expected["slide_id"],
                    "expected_object_count": len(required),
                    "picture_object_count": len(object_names),
                    "object_names": object_names,
                }
            )
    return {
        "schema_version": "1.0",
        "status": "degraded" if warnings else "strict_pass",
        "warnings": warnings,
        "contract": "background_and_generated_rasters_are_separate_selectable_pptx_objects",
        "slides": slides,
    }


def _font_for_role(role: str, config: PipelineConfig) -> dict[str, Any]:
    if role == "title":
        return {
            "face": config.deck_font,
            "size_pt": 34,
            "minimum_pt": 25,
            "weight": "bold",
        }
    if role == "subtitle":
        return {
            "face": config.deck_font,
            "size_pt": 22,
            "minimum_pt": 18,
            "weight": "regular",
        }
    return {
        "face": config.deck_font,
        "size_pt": 15,
        "minimum_pt": 14,
        "weight": "regular",
    }


def _font_from_html(region: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    typography = region.get("typography")
    if not isinstance(typography, dict):
        return _font_for_role(region["role"], config)
    try:
        size_pt = float(typography.get("font_size_pt") or 0)
        if not math.isfinite(size_pt) or size_pt <= 0:
            raise ValueError("invalid HTML font size")
    except (TypeError, ValueError):
        return _font_for_role(region["role"], config)
    floor_by_role = {"title": 18.0, "subtitle": 12.0, "body": 12.0}
    minimum_pt = min(
        size_pt,
        max(floor_by_role.get(region["role"], 9.0), size_pt * 0.72),
    )
    result = {
        "face": str(typography.get("font_family") or config.deck_font),
        "size_pt": round(size_pt, 2),
        "minimum_pt": round(minimum_pt, 2),
        "weight": "bold" if typography.get("font_weight") == "bold" else "regular",
        "italic": typography.get("font_style") == "italic",
        "line_height_ratio": float(typography.get("line_height_ratio") or 1.08),
        "color": str(typography.get("color") or "111111"),
    }
    if typography.get('line_count') and typography.get('line_height_px'):
        result['line_count'] = int(typography['line_count'])
        result['line_height_pt'] = float(typography['line_height_px']) * 540 / 864
    return result


def _usable_text_safe_rect(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        x, y = float(value["x"]), float(value["y"])
        width, height = float(value["width"]), float(value["height"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        x >= 0
        and y >= 0
        and width >= 0.2
        and height >= 0.2
        and x + width <= 1.001
        and y + height <= 1.001
    )


def _equivalent_groups(text_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    singletons = []
    for region in text_regions:
        group = region.get("equivalent_group")
        if group:
            grouped.setdefault(group, []).append(region["region_id"])
        else:
            singletons.append(region["region_id"])
    groups = [
        {"group_id": key, "member_region_ids": value} for key, value in grouped.items()
    ]
    groups.extend(
        {"group_id": f"singleton:{region_id}", "member_region_ids": [region_id]}
        for region_id in singletons
    )
    return groups


def _normalize_font_compatible_text(value: str) -> str:
    normalized = value
    for source, replacement in ASCII_ARROW_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def _normalize_rich_text(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    paragraphs = []
    for paragraph in value:
        if not isinstance(paragraph, dict):
            continue
        runs = []
        for run in paragraph.get("runs") or []:
            if not isinstance(run, dict):
                continue
            text = _normalize_font_compatible_text(str(run.get("text") or ""))
            if not text.strip():
                continue
            runs.append(
                {
                    "text": text,
                    "bold": bool(run.get("bold", False)),
                    "italic": bool(run.get("italic", False)),
                    "font_scale": float(run.get("font_scale") or 1.0),
                }
            )
        if runs:
            item = {"kind": str(paragraph.get("kind") or "body"), "runs": runs}
            box = paragraph.get("bbox_px")
            if isinstance(box, list) and len(box) == 4 and box[2] > 0 and box[3] > 0:
                item["bbox_px"] = list(box)
                if paragraph.get("line_height_px"):
                    item["line_height_px"] = float(paragraph["line_height_px"])
                if paragraph.get("line_count"):
                    item["line_count"] = int(paragraph["line_count"])
                if paragraph.get("bullet_indent_px"):
                    item["bullet_indent_px"] = float(paragraph["bullet_indent_px"])
                item["text_alignment"] = str(paragraph.get("text_alignment") or "left")
            paragraphs.append(item)
    return paragraphs


def _compile_deck(compilation_path: Path, output_path: Path, audit_path: Path) -> None:
    result = subprocess.run(
        [
            "node",
            str((ROOT / "compile_deck.mjs").resolve()),
            str(compilation_path.resolve()),
            str(output_path.resolve()),
            str(audit_path.resolve()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            "PPTX compilation failed: "
            + (result.stderr.strip() or result.stdout.strip())[:2000]
        )


def _render_and_validate(
    pptx_path: Path,
    pdf_path: Path,
    rendered_dir: Path,
    slide_count: int,
) -> dict[str, Any]:
    if shutil.which("libreoffice") is None or shutil.which("pdftoppm") is None:
        raise RuntimeError("LibreOffice and pdftoppm are required")
    pdf_path.unlink(missing_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)
    for stale in rendered_dir.glob("slide-*.png"):
        stale.unlink()
    with tempfile.TemporaryDirectory(prefix="rl4ppt-office-") as profile:
        result = subprocess.run(
            [
            "libreoffice",
            f"-env:UserInstallation={Path(profile).as_uri()}",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_path.parent),
            str(pptx_path),
            ],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
    if result.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError("LibreOffice failed to render the PPTX")
    result = subprocess.run(
        ["pdftoppm", "-png", "-r", "120", str(pdf_path), str(rendered_dir / "slide")],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("pdftoppm failed to render the PDF")
    rendered = sorted(rendered_dir.glob("slide-*.png"))
    warnings = []
    if len(rendered) != slide_count:
        warnings.append({"issue": "rendered_slide_count_mismatch", "expected": slide_count, "actual": len(rendered)})
    for path in rendered:
        with Image.open(path).convert("RGB") as image:
            if not any(low != high for low, high in image.getextrema()):
                warnings.append({"issue": "blank_rendered_slide", "path": str(path)})
    return {"status": "degraded" if warnings else "strict_pass", "warnings": warnings, "blocking": False, "pdf": str(pdf_path.resolve())}


def _copy_rendered_previews(
    rendered_dir: Path, output_dir: Path, slide_ids: list[str]
) -> None:
    for slide_id, rendered_path in zip(
        slide_ids, sorted(rendered_dir.glob("slide-*.png"))
    ):
        target = output_dir / slide_id / "preview.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered_path, target)


def _create_montage(rendered_dir: Path, output_path: Path) -> None:
    paths = sorted(rendered_dir.glob("slide-*.png"))
    gap, width, height, label = 12, 384, 216, 22
    montage = Image.new(
        "RGB", (gap + len(paths) * (width + gap), height + label + 2 * gap), "white"
    )
    draw = ImageDraw.Draw(montage)
    for index, path in enumerate(paths):
        image = ImageOps.contain(Image.open(path).convert("RGB"), (width, height))
        x = gap + index * (width + gap) + (width - image.width) // 2
        y = gap + (height - image.height) // 2
        montage.paste(image, (x, y))
        draw.text(
            (gap + index * (width + gap), gap + height + 3),
            str(index + 1),
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)
