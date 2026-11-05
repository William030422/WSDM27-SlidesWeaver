from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from .style_models import StyleAssetSlot, StyleReferencePlan


CANVAS = (1536, 864)
TEXT_HOLDER_CATEGORIES = {"title_bar", "subtitle_bar", "raster_holder"}

_SLOT_ROLES: dict[str, tuple[list[str], list[str]]] = {
    "title_bar_primary": (["title", "section"], ["title_background", "section_marker"]),
    "subtitle_bar_primary": (["subtitle", "section"], ["subtitle_background", "section_marker"]),
    "holder_wide": (["explanation", "comparison"], ["short_explanation", "comparison"]),
    "holder_compact": (["fact", "definition"], ["short_explanation", "definition"]),
    "holder_circle": (["keyword", "focus"], ["focal_annotation", "emphasis"]),
    "holder_speech": (["dialogue", "friendly"], ["annotation", "short_explanation", "dialogue"]),
    "holder_callout": (["annotation", "pointer"], ["annotation", "short_explanation"]),
    "holder_pill": (["takeaway", "stage"], ["takeaway", "stage_label"]),
    "badge_emphasis": (["important", "emphasis"], ["emphasis", "section_marker"]),
    "badge_evidence": (["evidence", "verification"], ["evidence", "verification"]),
    "badge_warning": (["warning", "risk"], ["warning", "misconception"]),
    "badge_milestone": (["milestone", "progress"], ["milestone", "emphasis"]),
    "badge_category": (["category", "section"], ["category", "section_marker"]),
    "badge_takeaway": (["takeaway", "key_point"], ["emphasis", "takeaway"]),
    "icon_concept": (["concept", "idea"], ["concept", "annotation"]),
    "icon_process": (["process", "transition"], ["process", "annotation"]),
    "icon_evidence": (["evidence", "measurement"], ["evidence", "annotation"]),
    "icon_relationship": (["relationship", "comparison"], ["relationship", "annotation"]),
    "icon_warning": (["warning", "constraint"], ["warning", "annotation"]),
    "icon_takeaway": (["takeaway", "action"], ["takeaway", "annotation"]),
    "icon_model": (["model", "simulation"], ["model", "annotation"]),
    "icon_environment": (["environment", "system"], ["environment", "annotation"]),
    "icon_planning": (["planning", "decision"], ["planning", "annotation"]),
}


def normalize_style_reference(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path).convert("RGB") as source:
        if source.size != CANVAS:
            source = source.resize(CANVAS, Image.Resampling.LANCZOS)
        source.save(output_path, format="PNG")


def normalize_background(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path).convert("RGB") as source:
        if source.size != CANVAS:
            source = source.resize(CANVAS, Image.Resampling.LANCZOS)
        source.save(output_path, format="PNG")


def render_segmented_reference(
    segmentation_manifest: dict[str, object],
    segmentation_dir: Path,
    output_path: Path,
) -> None:
    width, height = cast(list[int], segmentation_manifest["image_size"])
    reference = Image.new("RGBA", (int(width), int(height)), (246, 247, 245, 255))
    for record in cast(list[dict[str, Any]], segmentation_manifest["assets"]):
        asset = Image.open(segmentation_dir / record["path"]).convert("RGBA")
        x0, y0, _, _ = record["crop_bbox"]
        reference.alpha_composite(asset, (int(x0), int(y0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.convert("RGB").save(output_path)


def build_asset_manifest(
    naming_manifest: dict[str, object],
    plan: StyleReferencePlan,
    assets_dir: Path,
) -> dict[str, Any]:
    matches = {
        str(record["asset_id"]): record
        for record in cast(list[dict[str, Any]], naming_manifest["assets"])
    }
    records: list[dict[str, Any]] = []
    for slot in plan.slots:
        match = matches[slot.slot_id]
        asset_path = assets_dir / str(match["path"])
        with Image.open(asset_path) as image:
            native_size = [image.width, image.height]
        metadata = _slot_metadata(slot)
        available = match["status"] == "matched" and asset_path.is_file()
        record = {
            "asset_id": slot.slot_id,
            "category": slot.category,
            "shape": slot.shape,
            "path": (Path("assets") / str(match["path"])).as_posix(),
            "available": available,
            "quality_status": match["status"],
            "semantic_tags": metadata["semantic_tags"],
            "allowed_roles": metadata["allowed_roles"],
            "aspect_ratio": round(native_size[0] / native_size[1], 4),
            "native_pixel_size": native_size,
            "visual_weight": metadata["visual_weight"],
            "scalable": True,
            "preferred_size": metadata["preferred_size"],
            "matching_distance": match.get("matching_distance"),
            "source_fragment_ids": match.get("source_fragment_ids", []),
        }
        if slot.category in TEXT_HOLDER_CATEGORIES:
            text_safe_rect = detect_text_safe_rect(asset_path) if available else None
            record["editable_text_available"] = text_safe_rect is not None
            record["text_safe_rect_status"] = "detected" if text_safe_rect else "unavailable"
            if text_safe_rect:
                record["text_safe_rect"] = text_safe_rect
        records.append(record)
    return {
        "schema_version": "1.1",
        "stage": "style_asset_manifest",
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "assets": records,
    }


def build_design_system(plan: StyleReferencePlan, asset_manifest: dict[str, Any]) -> dict[str, Any]:
    available_categories = {
        "raster_holder" if asset["category"] in {"title_bar", "subtitle_bar"} else asset["category"]
        for asset in asset_manifest["assets"]
        if asset["available"]
    }
    capability_order = ["raster_holder", "identity_asset", "badge", "icon"]
    return {
        "schema_version": "1.0",
        "design_system_id": "deck_style_01",
        "source_style": plan.source_style,
        "style_prompt": plan.source_style,
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "key_color": plan.key_color,
        "avatar_policy": plan.avatar_policy,
        "renderer_capabilities": [
            "direct_text",
            "translucent_card",
            "editable_shape",
            "editable_connector",
        ],
        "asset_capabilities": [
            category for category in capability_order if category in available_categories
        ],
        "generation_references": [
            {
                "reference_id": "style_main",
                "path": "style_reference.png",
                "role": "global_style_and_identity",
            },
            {
                "reference_id": "background_main",
                "path": "canonical_background.png",
                "role": "deck_wide_canonical_background",
            },
        ],
        "asset_manifest_path": "asset_manifest.json",
    }


def _slot_metadata(slot: StyleAssetSlot) -> dict[str, Any]:
    if slot.category == "identity_asset":
        tags, roles = ["identity", "character", "thematic_object"], ["agent_identity", "identity"]
        preferred = _size(6, 18, 6, 16)
        weight = "medium"
    elif slot.category == "badge":
        tags, roles = _SLOT_ROLES[slot.slot_id]
        preferred = _size(3, 6, 3, 6)
        weight = "low"
    elif slot.category == "icon":
        tags, roles = _SLOT_ROLES[slot.slot_id]
        preferred = _size(2, 5, 2, 5)
        weight = "low"
    else:
        tags, roles = _SLOT_ROLES[slot.slot_id]
        weight = "medium"
        if slot.category == "title_bar":
            preferred = _size(16, 58, 3, 8)
        elif slot.category == "subtitle_bar":
            preferred = _size(12, 40, 3, 6)
        elif slot.shape == "circle":
            preferred = _size(4, 10, 4, 10)
        elif slot.shape == "pill":
            preferred = _size(8, 24, 3, 7)
        elif slot.slot_id == "holder_compact":
            preferred = _size(6, 16, 4, 12)
        elif slot.slot_id == "holder_wide":
            preferred = _size(10, 26, 4, 12)
        else:
            preferred = _size(8, 20, 4, 10)
    result: dict[str, Any] = {
        "semantic_tags": tags,
        "allowed_roles": roles,
        "visual_weight": weight,
        "preferred_size": preferred,
    }
    return result


def detect_text_safe_rect(asset_path: Path) -> dict[str, float] | None:
    with Image.open(asset_path).convert("RGBA") as image:
        rgba = np.asarray(image)
    alpha = rgba[:, :, 3]
    if alpha.size == 0 or int(alpha.max()) == 0:
        return None

    holder_mask = (alpha >= max(1, round(float(alpha.max()) * 0.5))).astype(np.uint8)
    holder_mask = _largest_component(holder_mask)
    if not holder_mask.any():
        return None

    opaque_rgb = rgba[:, :, :3][holder_mask.astype(bool)]
    quantized = np.clip(opaque_rgb // 24, 0, 10).astype(np.int32)
    keys = quantized[:, 0] * 121 + quantized[:, 1] * 11 + quantized[:, 2]
    image_lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    cleanup_radius = max(1, round(min(holder_mask.shape) * 0.015))
    kernel_size = cleanup_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    counts = np.bincount(keys, minlength=1331)
    candidate_keys = np.argsort(counts)[-8:][::-1]
    best: tuple[int, tuple[int, int, int, int], np.ndarray] | None = None
    for candidate_key in candidate_keys:
        if counts[candidate_key] == 0:
            continue
        fill_rgb = np.median(opaque_rgb[keys == candidate_key], axis=0).astype(np.uint8)
        fill_lab = cv2.cvtColor(fill_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3)
        color_distance = np.linalg.norm(image_lab - fill_lab, axis=2)
        interior_mask = ((holder_mask > 0) & (color_distance <= 24.0)).astype(np.uint8)
        safe_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_CLOSE, kernel)
        # Closing may connect candidate interiors, but cannot authorize text over
        # a real transparent hole, border detail, or contrasting decoration.
        safe_mask &= interior_mask
        safe_mask = _largest_component(safe_mask)
        rectangle = _largest_binary_rectangle(safe_mask)
        if rectangle is None:
            continue
        area = rectangle[2] * rectangle[3]
        if best is None or area > best[0]:
            best = (area, rectangle, safe_mask)
    if best is None:
        return None

    _, rectangle, safe_mask = best
    x, y, width, height = rectangle
    x += 2
    y += 2
    width -= 4
    height -= 4
    if width <= 0 or height <= 0 or not safe_mask[y : y + height, x : x + width].all():
        return None

    image_height, image_width = holder_mask.shape
    return {
        "x": round(x / image_width, 6),
        "y": round(y / image_height, 6),
        "width": round(width / image_width, 6),
        "height": round(height / image_height, 6),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def _largest_binary_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    heights = [0] * mask.shape[1]
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            heights[x] = heights[x] + 1 if value else 0
        stack: list[tuple[int, int]] = []
        for x in range(mask.shape[1] + 1):
            current = heights[x] if x < mask.shape[1] else 0
            start = x
            while stack and stack[-1][1] > current:
                start, height = stack.pop()
                width = x - start
                area = width * height
                if area > best_area:
                    best_area = area
                    best = (start, y - height + 1, width, height)
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def _size(min_width: int, max_width: int, min_height: int, max_height: int) -> dict[str, int]:
    return {
        "min_width": min_width,
        "max_width": max_width,
        "min_height": min_height,
        "max_height": max_height,
    }
