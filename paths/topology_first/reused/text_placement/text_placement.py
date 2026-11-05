#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


VLM_MODEL: str | None = None
VLM_ENV_PATH = Path(
    __file__
).resolve().parents[2] / ".env"
V9_SRC = Path(
    __file__
).resolve().parents[2] / "src"

TEXT_CANONICAL_ROLES = {
    "title": "title",
    "supporting_text": "subtitle",
    "standalone_text": "subtitle",
    "subtitle": "subtitle",
    "body": "body",
    "caption": "caption",
    "label": "caption",
    "annotation": "caption",
    "internal_label": "caption",
}

MIN_NATIVE_SHAPE_OPACITY = 0.90
MAX_HOLDER_ASPECT_RATIO_CHANGE = 0.20
GEOMETRY_MAX_TRANSLATION_RATIO = 0.02
GEOMETRY_MAX_SIZE_CHANGE = 0.05
DEFAULT_TITLE_BACKGROUND_COMPLEXITY_THRESHOLD = 30.0


class QuotaExhaustedError(RuntimeError):
    pass


def configured_vlm_model(*, required: bool = False) -> str | None:
    if VLM_MODEL:
        return VLM_MODEL
    model = os.environ.get("APIMART_CHAT_MODEL", "").strip()
    if not model and VLM_ENV_PATH.is_file():
        for raw_line in VLM_ENV_PATH.read_text(encoding="utf-8").splitlines():
            if raw_line.strip().startswith("APIMART_CHAT_MODEL="):
                model = raw_line.split("=", 1)[1].strip().strip("'\"")
                break
    if required and not model:
        raise ValueError("APIMART_CHAT_MODEL is not configured")
    return model or None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x, y, width, height = bbox
    return x + width / 2, y + height / 2


def _relative_luminance(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def _contrast_ratio(first_rgb: tuple[int, int, int], second_rgb: tuple[int, int, int]) -> float:
    values = np.asarray([first_rgb, second_rgb], dtype=np.uint8)
    luminance = _relative_luminance(values)
    lighter = float(max(luminance))
    darker = float(min(luminance))
    return (lighter + 0.05) / (darker + 0.05)


def _dominant_rgb(pixels: np.ndarray) -> tuple[int, int, int]:
    flat = pixels.reshape((-1, 3))
    quantized = np.clip(flat // 32, 0, 7).astype(np.int32)
    keys = quantized[:, 0] * 64 + quantized[:, 1] * 8 + quantized[:, 2]
    dominant_key = int(np.bincount(keys, minlength=512).argmax())
    selected = flat[keys == dominant_key]
    if not len(selected):
        selected = flat
    median = np.median(selected, axis=0)
    return tuple(int(round(value)) for value in median)


def _color_entropy(lab: np.ndarray, bins: int) -> float | None:
    if bins <= 1:
        return None
    histogram, _, _ = np.histogram2d(
        lab[..., 1].ravel(),
        lab[..., 2].ravel(),
        bins=bins,
        range=((-128, 127), (-128, 127)),
    )
    probabilities = histogram.ravel() / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def analyze_region(
    image_rgb: np.ndarray,
    bbox: list[float],
    *,
    entropy_bins: int = 0,
) -> dict[str, Any]:
    image_height, image_width = image_rgb.shape[:2]
    x, y, width, height = bbox
    left = max(0, min(image_width - 1, int(math.floor(x))))
    top = max(0, min(image_height - 1, int(math.floor(y))))
    right = max(left + 1, min(image_width, int(math.ceil(x + width))))
    bottom = max(top + 1, min(image_height, int(math.ceil(y + height))))
    crop = image_rgb[top:bottom, left:right]
    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    color_complexity = float(
        math.sqrt(float(np.var(lab[..., 1])) + float(np.var(lab[..., 2])))
    )
    luminance_std = float(np.std(lab[..., 0]))
    local_luminance = lab[..., 0]
    local_detail_p90 = float(
        np.percentile(
            np.abs(
                local_luminance
                - cv2.GaussianBlur(local_luminance, (0, 0), sigmaX=1.2)
            ),
            90,
        )
    )
    grayscale = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edge_density = float(np.mean(cv2.Canny(grayscale, 60, 140) > 0))
    median_lab = np.median(lab.reshape((-1, 3)), axis=0)
    color_distance = np.linalg.norm(lab - median_lab, axis=2)
    color_outlier_ratio = float(np.mean(color_distance > 12.0))

    # Low-frequency gradients stay cheap; texture and object boundaries dominate.
    complexity_components = (
        min(color_complexity / 14.0, 1.0),
        min(luminance_std / 22.0, 1.0),
        min(local_detail_p90 / 8.0, 1.0),
        min(edge_density / 0.12, 1.0),
        min(color_outlier_ratio / 0.35, 1.0),
    )
    background_complexity_score = 100.0 * sum(
        weight * value
        for weight, value in zip(
            (0.20, 0.22, 0.23, 0.23, 0.12),
            complexity_components,
            strict=True,
        )
    )
    dominant = _dominant_rgb(crop)

    luminance = _relative_luminance(crop)
    white_contrast = (1.0 + 0.05) / (luminance + 0.05)
    black_contrast = (luminance + 0.05) / 0.05
    white_p10 = float(np.percentile(white_contrast, 10))
    black_p10 = float(np.percentile(black_contrast, 10))
    text_color = "FFFFFF" if white_p10 >= black_p10 else "000000"
    direct_contrast_p10 = max(white_p10, black_p10)

    return {
        "color_complexity": round(color_complexity, 4),
        "color_entropy": (
            round(value, 4)
            if (value := _color_entropy(lab, entropy_bins)) is not None
            else None
        ),
        "luminance_std": round(luminance_std, 4),
        "local_detail_p90": round(local_detail_p90, 4),
        "edge_density": round(edge_density, 4),
        "color_outlier_ratio": round(color_outlier_ratio, 4),
        "background_complexity_score": round(background_complexity_score, 4),
        "dominant_rgb": list(dominant),
        "direct_text_color": text_color,
        "direct_contrast_p10": round(direct_contrast_p10, 4),
    }


def _mix_rgb(
    first: tuple[int, int, int], second: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    return tuple(
        int(round((1 - amount) * left + amount * right))
        for left, right in zip(first, second, strict=True)
    )


def native_shape_style_for_group(
    metrics_items: list[dict[str, Any]],
    opacity: float,
    theme_rgb: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    opacity = max(MIN_NATIVE_SHAPE_OPACITY, min(1.0, float(opacity)))
    local_colors = [tuple(metrics["dominant_rgb"]) for metrics in metrics_items]
    base_color = theme_rgb or local_colors[0]
    candidates = [
        base_color,
        _mix_rgb(base_color, (0, 0, 0), 0.28),
        _mix_rgb(base_color, (0, 0, 0), 0.48),
        _mix_rgb(base_color, (255, 255, 255), 0.32),
    ]
    best: tuple[float, tuple[int, int, int], str] | None = None
    for fill_rgb in candidates:
        for text_rgb, text_color in (((255, 255, 255), "FFFFFF"), ((0, 0, 0), "000000")):
            contrasts = [
                _contrast_ratio(_mix_rgb(local_color, fill_rgb, opacity), text_rgb)
                for local_color in local_colors
            ]
            score = min(contrasts) - 0.002 * sum(
                abs(left - right)
                for left, right in zip(fill_rgb, base_color, strict=True)
            )
            candidate = (score, fill_rgb, text_color)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    fill_rgb = best[1]
    text_color = best[2]
    text_rgb = (255, 255, 255) if text_color == "FFFFFF" else (0, 0, 0)
    contrast = min(
        _contrast_ratio(_mix_rgb(local_color, fill_rgb, opacity), text_rgb)
        for local_color in local_colors
    )
    return {
        "mode": "native_shape_text",
        "shape": {
            "kind": "round_rect",
            "fill_rgb": list(fill_rgb),
            "fill_opacity": opacity,
            "line": None,
            "corner_radius_px": 8,
        },
        "text_color": text_color,
        "expected_contrast": round(contrast, 4),
    }


def native_shape_style(
    metrics: dict[str, Any],
    opacity: float,
    theme_rgb: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    return native_shape_style_for_group([metrics], opacity, theme_rgb)


def choose_algorithmic_presentation(
    metrics: dict[str, Any],
    role: str,
    policy: dict[str, Any],
    theme_rgb: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    minimum_contrast = 3.0 if role == "title" else 4.5
    direct = (
        metrics["color_complexity"] <= policy["color_complexity_threshold"]
        and metrics["luminance_std"] <= policy["luminance_std_threshold"]
        and metrics["direct_contrast_p10"] >= minimum_contrast
    )
    if direct:
        return {
            "mode": "direct_text",
            "shape": None,
            "text_color": metrics["direct_text_color"],
            "expected_contrast": metrics["direct_contrast_p10"],
        }
    return native_shape_style(
        metrics,
        float(policy["native_shape_opacity"]),
        theme_rgb,
    )


def _component_region_map(topology: dict[str, Any]) -> dict[str, list[str]]:
    return {
        component["component_id"]: list(component.get("source_region_ids", []))
        for component in topology.get("components", [])
    }


def extract_text_slots(
    topology: dict[str, Any], layout: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layout_by_region = {
        item["region_id"]: list(item["postprocessed_bbox_px"])
        for item in layout.get("entities", [])
    }
    entities = {
        item["region_id"]: item
        for item in topology.get("entities", [])
        if item.get("canonical_role") in TEXT_CANONICAL_ROLES
    }
    component_regions = _component_region_map(topology)

    ordered_region_ids: list[str] = []
    for item_id in topology.get("reading_order", []):
        if item_id in entities:
            ordered_region_ids.append(item_id)
        for region_id in component_regions.get(item_id, []):
            if region_id in entities:
                ordered_region_ids.append(region_id)
    remaining = sorted(
        set(entities) - set(ordered_region_ids),
        key=lambda region_id: (
            layout_by_region.get(region_id, [0, 0])[1],
            layout_by_region.get(region_id, [0, 0])[0],
        ),
    )
    ordered_region_ids.extend(remaining)

    slots: list[dict[str, Any]] = []
    for order, region_id in enumerate(ordered_region_ids):
        entity = entities[region_id]
        bbox = layout_by_region.get(region_id)
        if not bbox:
            continue
        canonical_role = entity["canonical_role"]
        slots.append(
            {
                "region_id": region_id,
                "component_id": entity.get("parent_component_id"),
                "role": TEXT_CANONICAL_ROLES[canonical_role],
                "canonical_role": canonical_role,
                "bbox_px": bbox,
                "anchor_center_px": list(bbox_center(bbox)),
                "order": order,
            }
        )

    slot_by_region = {slot["region_id"]: slot for slot in slots}
    groups: list[dict[str, Any]] = []
    grouped_regions: set[str] = set()
    for topology_group in topology.get("groups", []):
        peer_relation = topology_group.get("peer_relation")
        if peer_relation not in {"equivalent", "sequential", "comparative"}:
            continue
        member_regions = [
            region_id
            for child_id in topology_group.get("children", [])
            for region_id in component_regions.get(child_id, [])
            if region_id in slot_by_region
        ]
        role_buckets: dict[str, list[str]] = {}
        for region_id in member_regions:
            role_buckets.setdefault(
                slot_by_region[region_id]["canonical_role"], []
            ).append(region_id)
        for canonical_role, role_regions in role_buckets.items():
            if len(role_regions) < 2:
                continue
            policy = topology_group.get("calibration_policy") or {}
            group_id = f"{topology_group['group_id']}:{canonical_role}"
            groups.append(
                {
                    "group_id": group_id,
                    "member_region_ids": role_regions,
                    "arrangement": topology_group.get("arrangement", "freeform"),
                    "peer_relation": peer_relation,
                    "equal_size": policy.get("size_policy") == "equal_size",
                    "even_spacing": policy.get("spacing") == "even",
                    "preserve_alignment": policy.get("alignment") not in {None, "preserve"}
                    or topology_group.get("arrangement") in {"row", "column", "grid"},
                }
            )
            grouped_regions.update(role_regions)
    for slot in slots:
        if slot["region_id"] not in grouped_regions:
            groups.append(
                {
                    "group_id": f"singleton:{slot['region_id']}",
                    "member_region_ids": [slot["region_id"]],
                    "arrangement": "freeform",
                    "peer_relation": "none",
                    "equal_size": False,
                    "even_spacing": False,
                    "preserve_alignment": False,
                }
            )
    return slots, groups


def _role_score(content_role: str, slot_role: str) -> int:
    normalized = TEXT_CANONICAL_ROLES.get(content_role, content_role)
    if normalized == slot_role:
        return 120
    compatible = {
        ("body", "subtitle"),
        ("subtitle", "body"),
        ("caption", "body"),
        ("body", "caption"),
    }
    return 45 if (normalized, slot_role) in compatible else -400


def _binding_score(
    content: dict[str, Any], slot: dict[str, Any], content_order: int, slot_order: int
) -> int:
    score = _role_score(str(content.get("role", "body")), slot["role"])
    target_region = content.get("target_region_id")
    if target_region:
        score += 1200 if target_region == slot["region_id"] else -1200
    target_component = content.get("target_component_id")
    if target_component:
        score += 700 if target_component == slot["component_id"] else -700
    expected_order = int(content.get("order", content_order))
    score += max(0, 50 - 8 * abs(expected_order - slot_order))
    return score


def bind_content(
    slots: list[dict[str, Any]], content_items: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates = sorted(
        (
            _binding_score(content, slot, content_index, slot_index),
            content_index,
            slot_index,
        )
        for content_index, content in enumerate(content_items)
        for slot_index, slot in enumerate(slots)
    )
    assigned_content: set[int] = set()
    assigned_slots: set[int] = set()
    bindings: list[dict[str, Any]] = []
    for score, content_index, slot_index in reversed(candidates):
        if score <= 0 or content_index in assigned_content or slot_index in assigned_slots:
            continue
        assigned_content.add(content_index)
        assigned_slots.add(slot_index)
        bindings.append(
            {
                "region_id": slots[slot_index]["region_id"],
                "content_id": content_items[content_index]["content_id"],
                "score": score,
            }
        )
    return {
        "bindings": sorted(bindings, key=lambda item: next(slot["order"] for slot in slots if slot["region_id"] == item["region_id"])),
        "unplaced_content_ids": [
            item["content_id"]
            for index, item in enumerate(content_items)
            if index not in assigned_content
        ],
        "empty_region_ids": [
            slot["region_id"]
            for index, slot in enumerate(slots)
            if index not in assigned_slots
        ],
    }


def _semantic_asset_candidates(
    content: dict[str, Any], asset_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    requested_roles = set(content.get("semantic_asset_roles", []))
    if not requested_roles:
        return []
    candidates: list[dict[str, Any]] = []
    for asset in asset_manifest.get("assets", []):
        if asset.get("category") not in {"raster_holder", "icon"}:
            continue
        text_safe_rect = _validated_text_safe_rect(asset.get("text_safe_rect"))
        if asset.get("category") == "raster_holder" and text_safe_rect is None:
            continue
        allowed_roles = set(asset.get("allowed_roles", []))
        if requested_roles & allowed_roles and asset.get("available", False):
            candidates.append(
                {
                    "asset_id": asset["asset_id"],
                    "category": asset["category"],
                    "allowed_roles": sorted(allowed_roles),
                    "text_safe_rect": text_safe_rect,
                }
            )
    return candidates


def _available_style_assets(
    asset_manifest: dict[str, Any], manifest_path: Path
) -> list[dict[str, Any]]:
    if str(V9_SRC) not in sys.path:
        sys.path.insert(0, str(V9_SRC))
    from v9_deck_planning.style_assets import detect_text_safe_rect

    allowed_categories = {
        "title_bar",
        "subtitle_bar",
        "raster_holder",
        "badge",
        "icon",
    }
    assets: list[dict[str, Any]] = []
    for asset in asset_manifest.get("assets", []):
        if not asset.get("available") or asset.get("category") not in allowed_categories:
            continue
        asset_path = (manifest_path.parent / str(asset["path"])).resolve()
        if not asset_path.is_file():
            continue
        is_text_holder = asset.get("category") in {
            "title_bar",
            "subtitle_bar",
            "raster_holder",
        }
        text_safe_rect = (
            _validated_text_safe_rect(detect_text_safe_rect(asset_path))
            if is_text_holder
            else None
        )
        if is_text_holder and text_safe_rect is None:
            continue
        assets.append(
            {
                "asset_id": asset["asset_id"],
                "category": asset["category"],
                "path": str(asset_path),
                "allowed_roles": list(asset.get("allowed_roles", [])),
                "semantic_tags": list(asset.get("semantic_tags", [])),
                "text_safe_rect": text_safe_rect,
                "aspect_ratio": float(asset.get("aspect_ratio", 0)),
                "native_pixel_size": list(asset.get("native_pixel_size", [])),
            }
        )
    return assets


def _title_background_requires_holder(
    metrics: dict[str, Any], policy: dict[str, Any]
) -> bool:
    try:
        return float(metrics["background_complexity_score"]) > float(
            policy.get(
                "title_background_complexity_threshold",
                DEFAULT_TITLE_BACKGROUND_COMPLEXITY_THRESHOLD,
            )
        )
    except (KeyError, TypeError, ValueError):
        return True


def _select_holder_asset(
    role: str,
    topology_bbox_px: list[float],
    available_assets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    holders = [
        asset
        for asset in available_assets
        if asset["category"] in {"title_bar", "subtitle_bar", "raster_holder"}
        and asset.get("text_safe_rect")
        and float(asset.get("aspect_ratio", 0)) > 0
    ]
    if not holders:
        return None

    target_aspect = float(topology_bbox_px[2]) / float(topology_bbox_px[3])

    def aspect_error(asset: dict[str, Any]) -> float:
        return abs(math.log(float(asset["aspect_ratio"]) / target_aspect))

    if role == "title":
        title_holders = [asset for asset in holders if asset["category"] == "title_bar"]
        return min(title_holders, key=aspect_error) if title_holders else None

    regular_holders = [asset for asset in holders if asset["category"] == "raster_holder"]
    if regular_holders:
        closest_regular = min(regular_holders, key=aspect_error)
        regular_aspect = float(closest_regular["aspect_ratio"])
        if abs(regular_aspect / target_aspect - 1.0) <= 0.35:
            return closest_regular
    return min(holders, key=aspect_error)


def _holder_presentation(
    asset: dict[str, Any],
    topology_bbox_px: list[float],
    background_rgb: np.ndarray,
) -> tuple[dict[str, Any], list[float]]:
    native_aspect = float(asset["aspect_ratio"])
    holder_bbox = _contain_bbox_for_aspect(topology_bbox_px, native_aspect)
    holder_metrics = _analyze_holder_surface(
        background_rgb,
        asset["path"],
        holder_bbox,
        asset["text_safe_rect"],
    )
    return (
        {
            "mode": "raster_holder_text",
            "shape": None,
            "asset_id": asset["asset_id"],
            "asset_category": asset["category"],
            "asset_path": asset["path"],
            "text_safe_rect": asset["text_safe_rect"],
            "native_aspect_ratio": native_aspect,
            "holder_family": asset["asset_id"],
            "text_color": holder_metrics["direct_text_color"],
            "expected_contrast": holder_metrics["direct_contrast_p10"],
            "contrast_source": "holder_composited_surface",
        },
        holder_bbox,
    )


def _analyze_holder_surface(
    background_rgb: np.ndarray,
    asset_path: str | Path,
    holder_bbox_px: list[float],
    text_safe_rect: dict[str, float],
) -> dict[str, Any]:
    """Measure the pixels visible behind holder text, including holder alpha."""
    if background_rgb.ndim != 3 or background_rgb.shape[2] != 3:
        raise ValueError("background_rgb must be an HxWx3 RGB array")
    canvas = Image.fromarray(background_rgb.astype(np.uint8), mode="RGB").convert(
        "RGBA"
    )
    x, y, width, height = (float(value) for value in holder_bbox_px)
    raw_left = int(round(x))
    raw_top = int(round(y))
    raw_width = max(1, int(round(width)))
    raw_height = max(1, int(round(height)))
    source = Image.open(Path(asset_path))
    try:
        holder = source.convert("RGBA").resize(
            (raw_width, raw_height), Image.Resampling.LANCZOS
        )
    finally:
        source.close()
    try:
        left = max(0, raw_left)
        top = max(0, raw_top)
        right = min(canvas.width, raw_left + raw_width)
        bottom = min(canvas.height, raw_top + raw_height)
        if left >= right or top >= bottom:
            raise ValueError("raster holder is outside the background canvas")
        crop = holder.crop(
            (left - raw_left, top - raw_top, right - raw_left, bottom - raw_top)
        )
        try:
            canvas.alpha_composite(crop, (left, top))
        finally:
            crop.close()
    finally:
        holder.close()

    rendered = canvas.convert("RGB")
    composed = np.asarray(rendered, dtype=np.uint8).copy()
    rendered.close()
    canvas.close()
    safe = _validated_text_safe_rect(text_safe_rect)
    if safe is None:
        raise ValueError("raster holder is missing a valid text_safe_rect")
    text_bbox = derive_holder_text_bbox_px(holder_bbox_px, safe)
    return analyze_region(composed, text_bbox)


def _analyze_native_shape_surface(
    background_rgb: np.ndarray, element: dict[str, Any]
) -> dict[str, Any]:
    presentation = element["presentation"]
    shape = presentation.get("shape") or {}
    fill_rgb = tuple(int(value) for value in shape.get("fill_rgb", (255, 255, 255)))
    opacity = max(
        MIN_NATIVE_SHAPE_OPACITY,
        min(1.0, float(shape.get("fill_opacity", 1.0))),
    )
    base = Image.fromarray(background_rgb.astype(np.uint8), mode="RGB").convert(
        "RGBA"
    )
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y, width, height = (float(value) for value in element["bbox_px"])
    draw.rounded_rectangle(
        (round(x), round(y), round(x + width), round(y + height)),
        radius=int(shape.get("corner_radius_px", 8)),
        fill=fill_rgb + (round(255 * opacity),),
    )
    base.alpha_composite(layer)
    layer.close()
    rendered = base.convert("RGB")
    composed = np.asarray(rendered, dtype=np.uint8).copy()
    rendered.close()
    base.close()
    return analyze_region(composed, element_text_rect_px(element))


def apply_text_contrast_solver(plan: dict[str, Any]) -> dict[str, Any]:
    """Re-solve every text box against the surface visible beneath its text."""
    with Image.open(plan["background_path"]) as source:
        background_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    for element in plan.get("elements", []):
        mode = element.get("presentation", {}).get("mode")
        if element.get("text_backing"):
            color = element["text_backing"]["fill_color"]
            rgb = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
            metrics = analyze_region(np.full((16, 16, 3), rgb, dtype=np.uint8), [0, 0, 16, 16])
        elif mode == "raster_holder_text":
            presentation = element["presentation"]
            metrics = _analyze_holder_surface(
                background_rgb,
                presentation["asset_path"],
                element["bbox_px"],
                presentation["text_safe_rect"],
            )
        elif mode == "native_shape_text":
            metrics = _analyze_native_shape_surface(background_rgb, element)
        else:
            metrics = analyze_region(
                background_rgb,
                element_text_rect_px(element),
            )
        presentation = element["presentation"]
        presentation["text_color"] = metrics["direct_text_color"]
        presentation["expected_contrast"] = metrics["direct_contrast_p10"]
        presentation["contrast_source"] = (
            "holder_composited_surface"
            if mode == "raster_holder_text"
            else "final_text_surface"
        )
        element["contrast_metrics"] = {
            "direct_text_color": metrics["direct_text_color"],
            "direct_contrast_p10": metrics["direct_contrast_p10"],
        }
    return plan


def _normalize_equal_group_sizes(
    elements: list[dict[str, Any]], groups: list[dict[str, Any]]
) -> None:
    element_by_region = {element["region_id"]: element for element in elements}
    for group in groups:
        members = [element_by_region[item] for item in group["member_region_ids"] if item in element_by_region]
        if len(members) < 2:
            continue
        if group["equal_size"]:
            holder_aspects = {
                round(float(member["presentation"].get("native_aspect_ratio", 0)), 6)
                for member in members
                if member["presentation"]["mode"] == "raster_holder_text"
            }
            if holder_aspects and (
                len(holder_aspects) != 1
                or any(
                    member["presentation"]["mode"] != "raster_holder_text"
                    for member in members
                )
            ):
                group["equal_size"] = False
                group["size_consistency"] = "approximately_consistent"
                continue
            width = sum(member["bbox_px"][2] for member in members) / len(members)
            height = sum(member["bbox_px"][3] for member in members) / len(members)
            for member in members:
                center_x, center_y = member["anchor_center_px"]
                member["bbox_px"] = [center_x - width / 2, center_y - height / 2, width, height]
        common_font = min(member["font"]["size_pt"] for member in members)
        for member in members:
            member["font"]["size_pt"] = common_font


def _harmonize_group_presentations(
    elements: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    opacity: float,
    theme_rgb: tuple[int, int, int],
) -> None:
    element_by_region = {element["region_id"]: element for element in elements}
    for group in groups:
        members = [
            element_by_region[item]
            for item in group["member_region_ids"]
            if item in element_by_region
        ]
        if len(members) < 2:
            continue
        if any(
            member["presentation"]["mode"] == "raster_holder_text"
            for member in members
        ):
            continue
        if any(member["presentation"]["mode"] == "native_shape_text" for member in members):
            shared_style = native_shape_style_for_group(
                [member["metrics"] for member in members],
                opacity,
                theme_rgb,
            )
            for member in members:
                member["presentation"] = copy.deepcopy(shared_style)


def build_plan(case_path: Path) -> dict[str, Any]:
    case = load_json(case_path)
    bundled_case = Path(__file__).with_name("case_slide_004.json").resolve()
    if case_path.resolve() == bundled_case:
        for key in ("background_path", "topology_path", "layout_path", "asset_manifest_path"):
            case[key] = str((bundled_case.parent / case[key]).resolve())
    mode_policy = copy.deepcopy(case["mode_policy"])
    mode_policy["native_shape_opacity"] = max(
        MIN_NATIVE_SHAPE_OPACITY,
        float(mode_policy["native_shape_opacity"]),
    )
    topology = load_json(Path(case["topology_path"]))
    layout = load_json(Path(case["layout_path"]))
    asset_manifest_path = Path(case["asset_manifest_path"])
    asset_manifest = load_json(asset_manifest_path)
    available_assets = _available_style_assets(asset_manifest, asset_manifest_path)
    background_path = Path(case["background_path"])
    image_rgb = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    theme_rgb = _dominant_rgb(image_rgb)
    slots, groups = extract_text_slots(topology, layout)
    binding = bind_content(slots, case["content"])
    content_by_id = {item["content_id"]: item for item in case["content"]}
    content_by_region = {
        item["region_id"]: content_by_id[item["content_id"]]
        for item in binding["bindings"]
    }
    region_to_group = {
        region_id: group["group_id"]
        for group in groups
        for region_id in group["member_region_ids"]
    }

    elements: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    for slot in slots:
        topology_bbox = [float(value) for value in slot["bbox_px"]]
        content = content_by_region.get(slot["region_id"])
        if content is None:
            fallbacks.append(
                {
                    "region_id": slot["region_id"],
                    "action": "omit_unbound_optional_slot",
                }
            )
            continue
        try:
            metrics = analyze_region(
                image_rgb,
                slot["bbox_px"],
                entropy_bins=int(case["mode_policy"].get("entropy_bins", 0)),
            )
            presentation = choose_algorithmic_presentation(
                metrics,
                slot["role"],
                mode_policy,
                theme_rgb,
            )
        except Exception as exc:
            metrics = {
                "color_complexity": None,
                "color_entropy": None,
                "luminance_std": None,
                "local_detail_p90": None,
                "edge_density": None,
                "color_outlier_ratio": None,
                "background_complexity_score": None,
                "dominant_rgb": [32, 32, 32],
                "direct_text_color": "FFFFFF",
                "direct_contrast_p10": None,
            }
            presentation = native_shape_style(
                metrics, float(mode_policy["native_shape_opacity"])
            )
            fallbacks.append(
                {
                    "region_id": slot["region_id"],
                    "action": "conservative_native_shape",
                    "reason": str(exc),
                }
            )
        semantic_candidates = _semantic_asset_candidates(content, asset_manifest)
        title_requires_holder = (
            slot["role"] == "title"
            and _title_background_requires_holder(metrics, mode_policy)
        )
        if slot["role"] == "title" and not title_requires_holder:
            presentation = {
                "mode": "direct_text",
                "shape": None,
                "text_color": metrics["direct_text_color"],
                "expected_contrast": metrics["direct_contrast_p10"],
            }
        use_holder = slot["role"] != "title" or title_requires_holder
        holder_asset = (
            _select_holder_asset(slot["role"], topology_bbox, available_assets)
            if use_holder
            else None
        )
        if holder_asset is not None:
            presentation, holder_bbox = _holder_presentation(
                holder_asset,
                topology_bbox,
                image_rgb,
            )
            slot = {
                **slot,
                "bbox_px": holder_bbox,
                "anchor_center_px": list(bbox_center(topology_bbox)),
            }
        elif use_holder and slot["role"] == "title":
            fallbacks.append(
                {
                    "region_id": slot["region_id"],
                    "action": "native_shape_title_fallback",
                    "reason": "non_solid_title_has_no_title_holder",
                }
            )
        elements.append(
            {
                **slot,
                "topology_bbox_px": topology_bbox,
                "content_id": content["content_id"],
                "text": content["text"],
                "bullet_items": list(content.get("bullet_items", [])),
                "constraint_group_id": region_to_group[slot["region_id"]],
                "font": copy.deepcopy(case["font_roles"][slot["role"]]),
                "metrics": metrics,
                "presentation": presentation,
                "semantic_asset_candidates": semantic_candidates,
                "needs_terra_asset_review": bool(semantic_candidates),
            }
        )
    _normalize_equal_group_sizes(elements, groups)
    _harmonize_group_presentations(
        elements,
        groups,
        opacity=float(mode_policy["native_shape_opacity"]),
        theme_rgb=theme_rgb,
    )
    for element in elements:
        _refresh_holder_binding(element)
        _refresh_content_unit(element)
    apply_text_contrast_solver(
        {"background_path": str(background_path), "elements": elements}
    )
    status = "ok" if not binding["unplaced_content_ids"] and not binding["empty_region_ids"] and not fallbacks else "degraded"
    return {
        "schema_version": "2.0",
        "slide_id": case["slide_id"],
        "status": status,
        "canvas_px": {"width": image_rgb.shape[1], "height": image_rgb.shape[0]},
        "background_path": str(background_path),
        "mode_policy": mode_policy,
        "text_presentation_policy": copy.deepcopy(
            case.get("text_presentation_policy", {"resolved": "light"})
        ),
        "overflow_owner": "content_unit_fit",
        "content_units_frozen": False,
        "font_roles": case["font_roles"],
        "theme_rgb": list(theme_rgb),
        "binding": binding,
        "groups": groups,
        "elements": elements,
        "available_assets": available_assets,
        "fallbacks": fallbacks,
        "vlm_model": configured_vlm_model(),
        "vlm_call_needed": any(element["needs_terra_asset_review"] for element in elements),
    }


def _validate_group_invariants(plan: dict[str, Any]) -> str | None:
    element_by_region = {item["region_id"]: item for item in plan["elements"]}
    for group in plan["groups"]:
        members = [element_by_region[item] for item in group["member_region_ids"] if item in element_by_region]
        if len(members) < 2:
            continue
        if group["equal_size"]:
            sizes = {(round(item["bbox_px"][2], 6), round(item["bbox_px"][3], 6)) for item in members}
            if len(sizes) != 1:
                return f"{group['group_id']}:unequal_size"
        if len({item["font"]["size_pt"] for item in members}) != 1:
            return f"{group['group_id']}:unequal_font"
        arrangement = group["arrangement"]
        if group["even_spacing"] and len(members) > 2 and arrangement in {"row", "column"}:
            axis = 0 if arrangement == "row" else 1
            positions = [bbox_center(item["bbox_px"])[axis] for item in members]
            gaps = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
            if max(gaps) - min(gaps) > 1.0:
                return f"{group['group_id']}:uneven_spacing"
    width = plan["canvas_px"]["width"]
    height = plan["canvas_px"]["height"]
    for element in plan["elements"]:
        x, y, element_width, element_height = element["bbox_px"]
        if x < 0 or y < 0 or x + element_width > width or y + element_height > height:
            return f"{element['region_id']}:out_of_bounds"
        if element["presentation"]["mode"] == "raster_holder_text":
            safe = _validated_text_safe_rect(
                element["presentation"].get("text_safe_rect")
            )
            if safe is None:
                return f"{element['region_id']}:missing_text_safe_rect"
            expected = {
                "asset_id": element["presentation"]["asset_id"],
                "holder_bbox_px": [float(value) for value in element["bbox_px"]],
                "text_bbox_px": _holder_text_bbox(element, safe),
            }
            if element.get("holder_binding") != expected:
                return f"{element['region_id']}:stale_holder_binding"
        unit = element.get("content_unit")
        if not isinstance(unit, dict):
            return f"{element['region_id']}:missing_content_unit"
        if unit.get("outer_bbox_px") != [float(value) for value in element["bbox_px"]]:
            return f"{element['region_id']}:stale_content_unit_bbox"
        holder_binding = element.get("holder_binding") or {}
        if unit.get("holder_asset_id") != holder_binding.get("asset_id"):
            return f"{element['region_id']}:stale_content_unit_holder"
        decoration = element.get("decoration") or {}
        if unit.get("decoration_asset_id") != decoration.get("asset_id"):
            return f"{element['region_id']}:stale_content_unit_decoration"
        if unit.get("binding_frozen"):
            if unit.get("frozen_holder_asset_id") != unit.get("holder_asset_id"):
                return f"{element['region_id']}:frozen_holder_binding_changed"
            if unit.get("frozen_decoration_asset_id") != unit.get("decoration_asset_id"):
                return f"{element['region_id']}:frozen_decoration_binding_changed"
    return None


def _move_group_from_member(
    members: list[dict[str, Any]],
    target_region_id: str,
    arrangement: str,
    dx: float,
    dy: float,
) -> None:
    centers = {item["region_id"]: bbox_center(item["bbox_px"]) for item in members}
    target_center = centers[target_region_id]
    centroid_x = sum(center[0] for center in centers.values()) / len(centers)
    centroid_y = sum(center[1] for center in centers.values()) / len(centers)

    def transform_axis(axis: int, delta: float, symmetric: bool) -> dict[str, float]:
        centroid = centroid_x if axis == 0 else centroid_y
        target = target_center[axis]
        if not symmetric or abs(target - centroid) < 1e-6:
            return {region_id: center[axis] + delta for region_id, center in centers.items()}
        scale = (target + delta - centroid) / (target - centroid)
        return {
            region_id: centroid + (center[axis] - centroid) * scale
            for region_id, center in centers.items()
        }

    x_positions = transform_axis(0, dx, arrangement in {"row", "grid"})
    y_positions = transform_axis(1, dy, arrangement in {"column", "grid"})
    for member in members:
        region_id = member["region_id"]
        width, height = member["bbox_px"][2:]
        member["bbox_px"][0] = x_positions[region_id] - width / 2
        member["bbox_px"][1] = y_positions[region_id] - height / 2


def apply_group_delta_non_failing(
    plan: dict[str, Any], proposal: dict[str, Any], target_group_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        if str(proposal.get("target_group_id", "")) != target_group_id:
            return original, {"status": "ignored", "reason": "wrong_group_target"}
        group = next(item for item in working["groups"] if item["group_id"] == target_group_id)
        element_by_region = {item["region_id"]: item for item in working["elements"]}
        members = [
            element_by_region[region_id]
            for region_id in group["member_region_ids"]
            if region_id in element_by_region
        ]
        if len(members) < 2:
            return original, {"status": "ignored", "reason": "group_has_fewer_than_two_units"}
        op = str(proposal.get("op", ""))
        if op == "set_consistency_rule":
            attribute = str(proposal["attribute"])
            value = str(proposal["value"])
            if attribute not in {"size", "alignment", "spacing", "visual_weight"}:
                return original, {"status": "ignored", "reason": "invalid_consistency_attribute"}
            if value not in {
                "strictly_consistent",
                "approximately_consistent",
                "intentionally_variable",
            }:
                return original, {"status": "ignored", "reason": "invalid_consistency_value"}
            group.setdefault("consistency_rules", {})[attribute] = value
        elif op == "normalize_alignment":
            axis = str(proposal.get("axis", "x"))
            if axis not in {"x", "y"}:
                return original, {"status": "ignored", "reason": "invalid_alignment_axis"}
            coordinate = 0 if axis == "x" else 1
            target = sum(bbox_center(item["bbox_px"])[coordinate] for item in members) / len(members)
            limit = GEOMETRY_MAX_TRANSLATION_RATIO * working["canvas_px"][
                "width" if axis == "x" else "height"
            ]
            deltas = [target - bbox_center(item["bbox_px"])[coordinate] for item in members]
            if any(abs(delta) > limit for delta in deltas):
                return original, {"status": "ignored", "reason": "alignment_exceeds_trust_region"}
            for member, delta in zip(members, deltas, strict=True):
                member["bbox_px"][coordinate] += delta
        elif op == "normalize_spacing":
            axis = 0 if group["arrangement"] == "row" else 1 if group["arrangement"] == "column" else None
            if axis is None:
                return original, {"status": "ignored", "reason": "group_has_no_linear_spacing_axis"}
            positions = [bbox_center(member["bbox_px"])[axis] for member in members]
            first, last = positions[0], positions[-1]
            if len(members) == 2:
                return original, {"status": "ignored", "reason": "two_member_spacing_already_defined"}
            targets = [first + index * (last - first) / (len(members) - 1) for index in range(len(members))]
            limit = GEOMETRY_MAX_TRANSLATION_RATIO * working["canvas_px"][
                "width" if axis == 0 else "height"
            ]
            if any(abs(target - position) > limit for target, position in zip(targets, positions, strict=True)):
                return original, {"status": "ignored", "reason": "spacing_exceeds_trust_region"}
            for member, target in zip(members, targets, strict=True):
                member["bbox_px"][axis] += target - bbox_center(member["bbox_px"])[axis]
        else:
            return original, {"status": "ignored", "reason": f"unsupported_group_op:{op}"}
        for member in members:
            _refresh_holder_binding(member)
            _refresh_content_unit(member)
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {"status": "applied", "group_id": target_group_id, "op": op}
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_group_delta:{exc}"}


def apply_unit_delta_non_failing(
    plan: dict[str, Any], proposal: dict[str, Any], target_region_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        if str(proposal.get("region_id", "")) != target_region_id:
            return original, {"status": "ignored", "reason": "wrong_unit_target"}
        element = next(item for item in working["elements"] if item["region_id"] == target_region_id)
        outer_before = list(element["bbox_px"])
        holder_before = (element.get("holder_binding") or {}).get("asset_id")
        op = str(proposal.get("op", ""))
        if op == "set_padding":
            padding = {
                key: float(proposal.get(key, 0))
                for key in ("left", "right", "top", "bottom")
            }
            if any(value < 0 or value > 0.08 for value in padding.values()):
                return original, {"status": "ignored", "reason": "padding_out_of_range"}
            if padding["left"] + padding["right"] >= 0.5 or padding["top"] + padding["bottom"] >= 0.5:
                return original, {"status": "ignored", "reason": "padding_collapses_text_area"}
            element["internal_padding"] = padding
        elif op == "move_decoration":
            decoration = element.get("decoration")
            if not decoration:
                return original, {"status": "ignored", "reason": "unit_has_no_decoration"}
            dx = float(proposal.get("dx_px", 0))
            dy = float(proposal.get("dy_px", 0))
            if abs(dx) > element["bbox_px"][2] * 0.05 or abs(dy) > element["bbox_px"][3] * 0.05:
                return original, {"status": "ignored", "reason": "decoration_move_exceeds_trust_region"}
            decoration["offset_px"] = [dx, dy]
        elif op == "set_crop_focal_point":
            x = float(proposal["x"])
            y = float(proposal["y"])
            if not (0.4 <= x <= 0.6 and 0.4 <= y <= 0.6):
                return original, {"status": "ignored", "reason": "crop_focal_point_out_of_range"}
            element["crop_focal_point"] = [x, y]
        else:
            return original, {"status": "ignored", "reason": f"unsupported_unit_op:{op}"}
        _refresh_holder_binding(element)
        _refresh_content_unit(element)
        if element["bbox_px"] != outer_before:
            return original, {"status": "ignored", "reason": "unit_outer_bbox_changed"}
        if (element.get("holder_binding") or {}).get("asset_id") != holder_before:
            return original, {"status": "ignored", "reason": "unit_holder_binding_changed"}
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {"status": "applied", "region_id": target_region_id, "op": op}
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_unit_delta:{exc}"}


def apply_style_delta_non_failing(
    plan: dict[str, Any], proposal: dict[str, Any], required_overlay: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        op = str(proposal.get("op", ""))
        elements = {item["region_id"]: item for item in working["elements"]}
        accent_id = f"accent_{len(working.get('accents', [])) + 1:03d}"
        if op == "add_outline":
            region_id = str(proposal.get("region_id", ""))
            if region_id not in elements:
                return original, {"status": "ignored", "reason": "unknown_outline_target"}
            bbox = elements[region_id]["bbox_px"]
            accent = {
                "accent_id": accent_id,
                "kind": "outline",
                "target_region_ids": [region_id],
                "bbox_px": list(bbox),
                "color": str(proposal.get("color", "4A5568")),
                "width_pt": min(2.0, max(0.5, float(proposal.get("width_pt", 1.0)))),
            }
        elif op == "add_connector":
            from_region_id = str(proposal.get("from_region_id", ""))
            to_region_id = str(proposal.get("to_region_id", ""))
            if from_region_id not in elements or to_region_id not in elements or from_region_id == to_region_id:
                return original, {"status": "ignored", "reason": "invalid_connector_targets"}
            accent = {
                "accent_id": accent_id,
                "kind": "connector",
                "target_region_ids": [from_region_id, to_region_id],
                "from_px": list(bbox_center(elements[from_region_id]["bbox_px"])),
                "to_px": list(bbox_center(elements[to_region_id]["bbox_px"])),
                "color": str(proposal.get("color", "4A5568")),
                "width_pt": min(2.0, max(0.5, float(proposal.get("width_pt", 1.0)))),
            }
        else:
            return original, {"status": "ignored", "reason": f"unsupported_style_op:{op}"}
        allowed_targets = set(required_overlay.get("target_region_ids", []))
        if allowed_targets and not set(accent["target_region_ids"]).issubset(allowed_targets):
            return original, {"status": "ignored", "reason": "accent_exceeds_required_overlay_target"}
        working.setdefault("accents", []).append(accent)
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {"status": "applied", "accent_id": accent_id, "op": op}
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_style_delta:{exc}"}


def apply_patch_non_failing(
    plan: dict[str, Any], patch: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        element_by_region = {item["region_id"]: item for item in working["elements"]}
        target_region_id = patch["region_id"]
        target = element_by_region[target_region_id]
        group = next(
            item
            for item in working["groups"]
            if target_region_id in item["member_region_ids"]
        )
        members = [
            element_by_region[item]
            for item in group["member_region_ids"]
            if item in element_by_region
        ]
        op = patch["op"]
        if op == "move":
            dx = float(patch.get("dx_px", 0))
            dy = float(patch.get("dy_px", 0))
            if (
                abs(dx) > GEOMETRY_MAX_TRANSLATION_RATIO * working["canvas_px"]["width"]
                or abs(dy)
                > GEOMETRY_MAX_TRANSLATION_RATIO * working["canvas_px"]["height"]
            ):
                return original, {"status": "ignored", "reason": "translation_exceeds_trust_region"}
            _move_group_from_member(
                members,
                target_region_id,
                group["arrangement"],
                dx,
                dy,
            )
        elif op == "resize":
            target_width = float(target["bbox_px"][2])
            target_height = float(target["bbox_px"][3])
            width = float(patch.get("width_px", target_width + float(patch.get("dw_px", 0))))
            height = float(
                patch.get("height_px", target_height + float(patch.get("dh_px", 0)))
            )
            if width <= 0 or height <= 0:
                return original, {"status": "ignored", "reason": "invalid_resize_dimensions"}
            width_scale = width / target_width
            height_scale = height / target_height
            if (
                abs(width_scale - 1.0) > GEOMETRY_MAX_SIZE_CHANGE + 1e-6
                or abs(height_scale - 1.0) > GEOMETRY_MAX_SIZE_CHANGE + 1e-6
            ):
                return original, {"status": "ignored", "reason": "resize_exceeds_trust_region"}
            holder_present = any(
                member["presentation"]["mode"] == "raster_holder_text"
                for member in members
            )
            if holder_present and abs(width_scale - height_scale) > 0.002:
                return original, {"status": "ignored", "reason": "holder_resize_must_be_proportional"}
            for member in members:
                center_x, center_y = bbox_center(member["bbox_px"])
                if holder_present:
                    member_width = member["bbox_px"][2] * width_scale
                    member_height = member["bbox_px"][3] * width_scale
                else:
                    member_width = width
                    member_height = height
                member["bbox_px"] = [
                    center_x - member_width / 2,
                    center_y - member_height / 2,
                    member_width,
                    member_height,
                ]
        else:
            return original, {"status": "ignored", "reason": f"unsupported_op:{op}"}
        for member in members:
            _refresh_holder_binding(member)
            _refresh_content_unit(member)
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {
            "status": "applied",
            "group_id": group["group_id"],
            "source_region_id": target_region_id,
            "op": op,
        }
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_patch:{exc}"}


def apply_rewrite_non_failing(
    plan: dict[str, Any], rewrite: dict[str, Any], requested_region_ids: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        if working.get("content_units_frozen"):
            return original, {"status": "ignored", "reason": "content_unit_content_frozen"}
        region_id = str(rewrite["region_id"])
        if region_id not in requested_region_ids:
            return original, {"status": "ignored", "reason": "region_not_requested"}
        element = next(item for item in working["elements"] if item["region_id"] == region_id)
        old_text = str(element["text"])
        new_text = str(rewrite["text"]).strip()
        lines = [line.strip() for line in new_text.splitlines() if line.strip()]
        if not lines or len(lines) > 3 or "•" in new_text:
            return original, {"status": "ignored", "reason": "invalid_text_structure"}
        if len(new_text) > len(old_text) + max(20, round(len(old_text) * 0.2)):
            return original, {"status": "ignored", "reason": "rewrite_increases_density"}
        protected = set(re.findall(r"\b(?:\d+(?:\.\d+)?%?|[A-Z]{2,})\b", old_text))
        missing = sorted(token for token in protected if token not in new_text)
        if missing:
            return original, {
                "status": "ignored",
                "reason": f"protected_tokens_missing:{','.join(missing)}",
            }
        element["text"] = "\n".join(lines)
        element["bullet_items"] = (
            lines
            if len(lines) > 1 and (element.get("bullet_items") or element["role"] == "body")
            else []
        )
        return working, {"status": "applied", "region_id": region_id, "op": "rewrite"}
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_rewrite:{exc}"}


def apply_asset_decision_non_failing(
    plan: dict[str, Any], decision: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        region_id = str(decision["region_id"])
        element = next(item for item in working["elements"] if item["region_id"] == region_id)
        action = str(decision.get("decision", "keep_algorithmic"))
        if action == "keep_algorithmic":
            return working, {"status": "applied", "region_id": region_id, "op": action}
        if working.get("content_units_frozen"):
            return original, {"status": "ignored", "reason": "content_unit_binding_frozen"}
        asset_id = str(decision["asset_id"])
        asset = next(
            item for item in working.get("available_assets", []) if item["asset_id"] == asset_id
        )
        expected_category = {
            "use_raster_holder": "raster_holder",
            "add_badge": "badge",
            "add_icon": "icon",
        }.get(action)
        if expected_category is None or asset["category"] != expected_category:
            return original, {"status": "ignored", "reason": "asset_category_mismatch"}
        if action == "use_raster_holder":
            text_safe_rect = _validated_text_safe_rect(asset.get("text_safe_rect"))
            if text_safe_rect is None:
                return original, {"status": "ignored", "reason": "missing_text_safe_rect"}
            native_aspect = float(asset.get("aspect_ratio", 0))
            if native_aspect <= 0:
                return original, {"status": "ignored", "reason": "invalid_holder_aspect_ratio"}
            group = next(
                item for item in working["groups"] if region_id in item["member_region_ids"]
            )
            target_ids = (
                group["member_region_ids"] if group.get("equal_size") else [region_id]
            )
            target_elements = [
                item for item in working["elements"] if item["region_id"] in target_ids
            ]
            with Image.open(working["background_path"]) as source:
                image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
            for target_element in target_elements:
                target_element["presentation"], target_element["bbox_px"] = (
                    _holder_presentation(
                        {
                            **asset,
                            "text_safe_rect": text_safe_rect,
                            "aspect_ratio": native_aspect,
                        },
                        target_element["bbox_px"],
                        image_rgb,
                    )
                )
                _refresh_holder_binding(target_element)
                _refresh_content_unit(target_element)
        else:
            element["decoration"] = {
                "asset_id": asset_id,
                "asset_path": asset["path"],
                "category": asset["category"],
                "placement": "left",
            }
            _refresh_content_unit(element)
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {
            "status": "applied",
            "region_id": region_id,
            "op": action,
            "asset_id": asset_id,
        }
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_asset_decision:{exc}"}


def _intersection_area(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def apply_addition_non_failing(
    plan: dict[str, Any], addition: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        region_id = str(addition["region_id"])
        if not region_id.startswith("annotation_new_"):
            return original, {"status": "ignored", "reason": "invalid_region_id"}
        if any(item["region_id"] == region_id for item in working["elements"]):
            return original, {"status": "ignored", "reason": "duplicate_region_id"}
        role = str(addition.get("role", "caption"))
        if role not in {"body", "caption", "subtitle"}:
            return original, {"status": "ignored", "reason": "invalid_role"}
        text = str(addition["text"]).strip()
        if not text or len(text) > 160 or len(text.splitlines()) > 3:
            return original, {"status": "ignored", "reason": "invalid_text"}
        bbox = [float(value) for value in addition["bbox_px"]]
        if len(bbox) != 4 or bbox[2] < 80 or bbox[3] < 32:
            return original, {"status": "ignored", "reason": "invalid_bbox"}
        width = working["canvas_px"]["width"]
        height = working["canvas_px"]["height"]
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > width or bbox[1] + bbox[3] > height:
            return original, {"status": "ignored", "reason": "addition_out_of_bounds"}
        if any(
            _intersection_area(bbox, item["bbox_px"]) > bbox[2] * bbox[3] * 0.02
            for item in working["elements"]
        ):
            return original, {"status": "ignored", "reason": "addition_overlaps_text"}
        image_rgb = np.asarray(
            Image.open(working["background_path"]).convert("RGB"), dtype=np.uint8
        )
        metrics = analyze_region(
            image_rgb,
            bbox,
            entropy_bins=int(working["mode_policy"].get("entropy_bins", 0)),
        )
        presentation = choose_algorithmic_presentation(
            metrics,
            role,
            working["mode_policy"],
            tuple(working["theme_rgb"]),
        )
        element = {
            "region_id": region_id,
            "component_id": None,
            "role": role,
            "bbox_px": bbox,
            "anchor_center_px": list(bbox_center(bbox)),
            "order": len(working["elements"]),
            "content_id": f"content_{region_id}",
            "text": text,
            "bullet_items": (
                [line.strip() for line in text.splitlines() if line.strip()]
                if role == "body" and len(text.splitlines()) > 1
                else []
            ),
            "constraint_group_id": f"singleton:{region_id}",
            "font": copy.deepcopy(working["font_roles"][role]),
            "metrics": metrics,
            "presentation": presentation,
            "semantic_asset_candidates": [],
            "needs_terra_asset_review": False,
        }
        _refresh_content_unit(element)
        working["elements"].append(element)
        working["groups"].append(
            {
                "group_id": f"singleton:{region_id}",
                "member_region_ids": [region_id],
                "arrangement": "freeform",
                "peer_relation": "none",
                "equal_size": False,
                "even_spacing": False,
                "preserve_alignment": False,
            }
        )
        asset_id = addition.get("asset_id")
        outcome: dict[str, Any] = {"status": "applied", "region_id": region_id, "op": "add_text"}
        if asset_id:
            asset = next(
                item for item in working.get("available_assets", []) if item["asset_id"] == asset_id
            )
            action = "use_raster_holder" if asset["category"] == "raster_holder" else (
                "add_badge" if asset["category"] == "badge" else "add_icon"
            )
            working, asset_outcome = apply_asset_decision_non_failing(
                working,
                {"region_id": region_id, "decision": action, "asset_id": asset_id},
            )
            outcome["asset"] = asset_outcome
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, outcome
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_addition:{exc}"}


def apply_role_update_non_failing(
    plan: dict[str, Any], update: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        role = str(update["role"])
        members = [item for item in working["elements"] if item["role"] == role]
        if not members:
            return original, {"status": "ignored", "reason": "role_not_present"}
        op = str(update["op"])
        if op == "set_font_size":
            size = float(update["size_pt"])
            if any(size < item["font"]["minimum_pt"] or size > item["font"]["maximum_pt"] for item in members):
                return original, {"status": "ignored", "reason": "font_size_out_of_range"}
            for item in members:
                item["font"]["size_pt"] = size
        elif op == "set_shape_opacity":
            opacity = float(update["opacity"])
            if not MIN_NATIVE_SHAPE_OPACITY <= opacity <= 1.0:
                return original, {"status": "ignored", "reason": "opacity_out_of_range"}
            for item in members:
                if item["presentation"]["mode"] == "native_shape_text":
                    item["presentation"]["shape"]["fill_opacity"] = opacity
        else:
            return original, {"status": "ignored", "reason": f"unsupported_op:{op}"}
        violation = _validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {"status": "applied", "role": role, "op": op}
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_role_update:{exc}"}


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        phrase in message
        for phrase in (
            "out of quota",
            "insufficient_quota",
            "credits exhausted",
            "billing limit",
            "quota limit reached",
            "token quota is not enough",
            "pre_consume_token_quota_failed",
            "http 402",
        )
    )


def call_vlm_json(
    *,
    prompt: str,
    image_path: Path,
    client: Any | None = None,
    max_tokens: int = 1800,
) -> tuple[dict[str, Any] | None, str | None]:
    if client is None:
        if str(V9_SRC) not in sys.path:
            sys.path.insert(0, str(V9_SRC))
        from v9_deck_planning.apimart import ApiMartClient

        client = ApiMartClient(env_path=VLM_ENV_PATH)
    try:
        raw = client.vlm_json(
            model=configured_vlm_model(required=True),
            prompt=prompt,
            image_path=image_path,
            max_tokens=max_tokens,
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None, "vlm_non_object_response"
        return payload, None
    except Exception as exc:
        if is_quota_error(exc):
            raise QuotaExhaustedError(str(exc)) from exc
        return None, f"vlm_unavailable:{exc}"


def call_terra_json(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Compatibility wrapper for the original reused helper name."""
    return call_vlm_json(**kwargs)


def _font_path(weight: str) -> Path:
    filename = "DejaVuSans-Bold.ttf" if weight in {"bold", "semibold"} else "DejaVuSans.ttf"
    return Path("/usr/share/fonts/truetype/dejavu") / filename


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    maximum_width: float,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= maximum_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


def _validated_text_safe_rect(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        safe = {key: float(value[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None
    if (
        safe["x"] < 0
        or safe["y"] < 0
        or safe["width"] <= 0
        or safe["height"] <= 0
        or safe["x"] + safe["width"] > 1
        or safe["y"] + safe["height"] > 1
    ):
        return None
    return safe


def derive_holder_text_bbox_px(
    holder_bbox_px: list[float], text_safe_rect: dict[str, float]
) -> list[float]:
    x, y, width, height = holder_bbox_px
    return [
        x + width * text_safe_rect["x"],
        y + height * text_safe_rect["y"],
        width * text_safe_rect["width"],
        height * text_safe_rect["height"],
    ]


def _contain_bbox_for_aspect(
    target_bbox_px: list[float], aspect_ratio: float
) -> list[float]:
    x, y, width, height = (float(value) for value in target_bbox_px)
    if width / height > aspect_ratio:
        fitted_height = height
        fitted_width = height * aspect_ratio
    else:
        fitted_width = width
        fitted_height = width / aspect_ratio
    return [
        x + (width - fitted_width) / 2,
        y + (height - fitted_height) / 2,
        fitted_width,
        fitted_height,
    ]


def _holder_text_bbox(element: dict[str, Any], safe: dict[str, float]) -> list[float]:
    backing = element.get("text_backing", {})
    if backing.get("mode") == "themed_quiet_interior":
        backing["bbox_px"] = [float(v) for v in element["bbox_px"]]
        return derive_holder_text_bbox_px(element["bbox_px"], backing["text_safe_rect"])
    return _apply_internal_padding(
        derive_holder_text_bbox_px(element["bbox_px"], safe), element.get("internal_padding")
    )


def restore_quiet_interior(element: dict[str, Any], output_dir: Path | None = None) -> bool:
    """Keep the generated rim over an editable, matching quiet center."""
    from integration.decorative_surface import prepare_decorative_surface
    presentation = element.get("presentation") or {}
    asset = Path(presentation.get("asset_path") or "")
    if presentation.get("mode") != "raster_holder_text" or not asset.is_file():
        return False
    if element.get("text_backing"):
        _refresh_holder_binding(element)
        return True
    if output_dir is None:
        return False
    surface = prepare_decorative_surface(
        asset, output_dir / f"{element['region_id']}__{presentation['asset_id']}.png",
        {"x": .06, "y": .12, "width": .88, "height": .76},
    )
    presentation["quiet_frame_path"] = surface["frame_path"]
    element["text_backing"] = {
        "mode": "themed_quiet_interior", "fill_color": surface["fill_color"],
        "bbox_px": list(element["bbox_px"]), "normalized_polygon": surface["normalized_polygon"],
        "text_safe_rect": surface["text_safe_rect"],
    }
    if _validated_text_safe_rect(presentation.get("text_safe_rect")) is None:
        presentation["text_safe_rect"] = surface["text_safe_rect"]
    _refresh_holder_binding(element)
    _refresh_content_unit(element)
    return True


def _refresh_holder_binding(element: dict[str, Any]) -> None:
    if element["presentation"]["mode"] != "raster_holder_text":
        element.pop("holder_binding", None)
        return
    safe = _validated_text_safe_rect(element["presentation"].get("text_safe_rect"))
    if safe is None:
        element["presentation"] = native_shape_style(
            element.get("metrics") or {"dominant_rgb": [247, 247, 244]},
            MIN_NATIVE_SHAPE_OPACITY,
        )
        element.pop("holder_binding", None)
        element.setdefault("quality_warnings", []).append("invalid_holder_safe_rect_native_fallback")
        return
    holder_bbox = [float(value) for value in element["bbox_px"]]
    text_bbox = _holder_text_bbox(element, safe)
    element["holder_binding"] = {
        "asset_id": element["presentation"]["asset_id"],
        "holder_bbox_px": holder_bbox,
        "text_bbox_px": text_bbox,
    }


def _refresh_content_unit(element: dict[str, Any]) -> None:
    existing = element.get("content_unit") or {}
    holder = element.get("holder_binding") or {}
    decoration = element.get("decoration") or {}
    unit = {
        "unit_id": existing.get("unit_id", f"unit_{element['region_id']}"),
        "text_region_id": element["region_id"],
        "outer_bbox_px": [float(value) for value in element["bbox_px"]],
        "holder_asset_id": holder.get("asset_id"),
        "decoration_asset_id": decoration.get("asset_id"),
        "binding_frozen": bool(existing.get("binding_frozen", False)),
        "preserve_relative_order": True,
        "max_holder_aspect_ratio_change": MAX_HOLDER_ASPECT_RATIO_CHANGE,
    }
    if unit["binding_frozen"]:
        unit["frozen_holder_asset_id"] = existing.get(
            "frozen_holder_asset_id", unit["holder_asset_id"]
        )
        unit["frozen_decoration_asset_id"] = existing.get(
            "frozen_decoration_asset_id", unit["decoration_asset_id"]
        )
    element["content_unit"] = unit


def freeze_content_units(plan: dict[str, Any]) -> dict[str, Any]:
    working = copy.deepcopy(plan)
    working["content_units_frozen"] = True
    for element in working["elements"]:
        _refresh_holder_binding(element)
        _refresh_content_unit(element)
        unit = element["content_unit"]
        unit["binding_frozen"] = True
        unit["frozen_holder_asset_id"] = unit["holder_asset_id"]
        unit["frozen_decoration_asset_id"] = unit["decoration_asset_id"]
    violation = _validate_group_invariants(working)
    if violation:
        working.setdefault("quality_warnings", []).append({
            "stage": "freeze_content_units", "invariant": violation,
        })
    return working


def element_text_rect_px(element: dict[str, Any]) -> list[float]:
    x, y, width, height = element["bbox_px"]
    mode = element["presentation"]["mode"]
    if mode == "native_shape_text":
        x += width * 0.06
        y += height * 0.12
        width *= 0.88
        height *= 0.76
    elif mode == "raster_holder_text":
        holder_binding = element.get("holder_binding") or {}
        if not holder_binding.get("text_bbox_px"):
            raise ValueError("raster holder is missing a valid text_safe_rect")
        x, y, width, height = holder_binding["text_bbox_px"]
    if mode != "raster_holder_text":
        x, y, width, height = _apply_internal_padding(
            [x, y, width, height], element.get("internal_padding")
        )
    if element.get("decoration"):
        decoration_width = min(height * 0.82, width * 0.22)
        gap = min(12.0, width * 0.035)
        x += decoration_width + gap
        width -= decoration_width + gap
    return [x, y, max(1.0, width), max(1.0, height)]


def _apply_internal_padding(
    bbox: list[float], padding: dict[str, Any] | None
) -> list[float]:
    if not padding:
        return [float(value) for value in bbox]
    x, y, width, height = (float(value) for value in bbox)
    left = float(padding.get("left", 0))
    right = float(padding.get("right", 0))
    top = float(padding.get("top", 0))
    bottom = float(padding.get("bottom", 0))
    return [
        x + width * left,
        y + height * top,
        max(1.0, width * (1 - left - right)),
        max(1.0, height * (1 - top - bottom)),
    ]


def render_preview(plan: dict[str, Any], output_path: Path) -> None:
    image = Image.open(plan["background_path"]).convert("RGBA")
    shape_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shape_draw = ImageDraw.Draw(shape_layer)
    for element in plan["elements"]:
        x, y, width, height = element["bbox_px"]
        mode = element["presentation"]["mode"]
        if mode == "native_shape_text":
            shape = element["presentation"]["shape"]
            fill = tuple(shape["fill_rgb"]) + (
                round(255 * max(MIN_NATIVE_SHAPE_OPACITY, shape["fill_opacity"])),
            )
            shape_draw.rounded_rectangle(
                (round(x), round(y), round(x + width), round(y + height)),
                radius=shape["corner_radius_px"],
                fill=fill,
            )
        elif mode == "raster_holder_text":
            holder = Image.open(element["presentation"]["asset_path"]).convert("RGBA")
            holder = holder.resize((max(1, round(width)), max(1, round(height))), Image.Resampling.LANCZOS)
            shape_layer.alpha_composite(holder, (round(x), round(y)))
        decoration = element.get("decoration")
        if decoration:
            decoration_width = min(height * 0.82, width * 0.22)
            asset = Image.open(decoration["asset_path"]).convert("RGBA")
            asset.thumbnail(
                (max(1, round(decoration_width)), max(1, round(height * 0.82))),
                Image.Resampling.LANCZOS,
            )
            shape_layer.alpha_composite(
                asset,
                (round(x), round(y + (height - asset.height) / 2)),
            )
    image = Image.alpha_composite(image, shape_layer)
    draw = ImageDraw.Draw(image)
    for element in plan["elements"]:
        x, y, width, height = element_text_rect_px(element)
        bullet_items = [
            str(item).strip() for item in element.get("bullet_items", []) if str(item).strip()
        ]
        display_text = (
            "\n".join(f"• {item}" for item in bullet_items)
            if bullet_items
            else element["text"]
        )
        font_size_px = max(10, round(element["font"]["size_pt"] * 1.6))
        font = ImageFont.truetype(
            str(_font_path(element["font"]["weight"])), font_size_px
        )
        available_width = width
        available_height = height
        lines = _wrapped_lines(draw, display_text, font, available_width)
        line_height = draw.textbbox((0, 0), "Ag", font=font)[3] * 1.15
        while line_height * len(lines) > available_height and font_size_px > 10:
            font_size_px -= 1
            font = ImageFont.truetype(
                str(_font_path(element["font"]["weight"])), font_size_px
            )
            lines = _wrapped_lines(draw, display_text, font, available_width)
            line_height = draw.textbbox((0, 0), "Ag", font=font)[3] * 1.15
        block_height = line_height * len(lines)
        text_y = y + max(0, (available_height - block_height) / 2)
        for line in lines:
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            text_x = (
                x
                if bullet_items or element["role"] in {"title", "subtitle", "body"}
                else x + (width - line_width) / 2
            )
            draw.text(
                (round(text_x), round(text_y)),
                line,
                font=font,
                fill=f"#{element['presentation']['text_color']}",
            )
            text_y += line_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def write_report(plan: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"slide: {plan['slide_id']}",
        f"status: {plan['status']}",
        f"vlm model: {plan['vlm_model']}",
        f"vlm semantic asset review needed: {plan['vlm_call_needed']}",
        "",
    ]
    for element in plan["elements"]:
        metrics = element["metrics"]
        lines.append(
            " | ".join(
                (
                    element["region_id"],
                    element["content_id"],
                    f"group={element['constraint_group_id']}",
                    f"V_C={metrics['color_complexity']}",
                    f"H_C={metrics['color_entropy']}",
                    f"L_std={metrics['luminance_std']}",
                    f"B_score={metrics.get('background_complexity_score')}",
                    f"mode={element['presentation']['mode']}",
                    f"semantic_candidates={len(element['semantic_asset_candidates'])}",
                )
            )
        )
    if plan["fallbacks"]:
        lines.extend(("", "fallbacks:", *[str(item) for item in plan["fallbacks"]]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic topology-bound text placement plan."
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path(__file__).with_name("case_slide_004.json"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(__file__).with_name("outputs") / "slide_004_mode_preview_final.png",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).with_name("outputs") / "slide_004_report_final.txt",
    )
    parser.add_argument("--emit-plan", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args.case.resolve())
    render_preview(plan, args.preview.resolve())
    write_report(plan, args.report.resolve())
    if args.emit_plan:
        args.emit_plan.parent.mkdir(parents=True, exist_ok=True)
        args.emit_plan.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"status={plan['status']}")
    print(f"preview={args.preview.resolve()}")
    print(f"report={args.report.resolve()}")
    print(f"vlm_call_needed={plan['vlm_call_needed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
