from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


BORDER_WIDTH = 12
MIN_COMPONENT_AREA = 64
MIN_COMPONENT_AREA_FRACTION = 0.0004
EDGE_RADIUS = 5
CROP_PADDING = 4
ALPHA_INSET_PIXELS = 1
FOREGROUND_DISTANCE_THRESHOLD = 35.0
CORE_DISTANCE_THRESHOLDS = (50.0, 70.0, 90.0, 110.0)
CORE_AREA_FRACTION = 0.03
KEY_CHROMA_EPSILON = 1.0
KEY_HUE_ALIGNMENT_FLOOR = 0.75


def extract_assets(
    input_path: Path,
    output_dir: Path,
    *,
    assets_dir: Path | None = None,
    slots: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read input image: {input_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    height, width = rgb.shape[:2]

    border_width = min(BORDER_WIDTH, max(1, min(height, width) // 4))
    border_lab = _border_pixels(lab, border_width)
    border_rgb = _border_pixels(rgb, border_width)
    background_lab = np.median(border_lab, axis=0)
    background_rgb = np.median(border_rgb, axis=0)

    distance = np.linalg.norm(lab - background_lab, axis=2)
    border_distance = np.linalg.norm(border_lab - background_lab, axis=1)
    seed_threshold = max(4.0, min(12.0, float(np.percentile(border_distance, 95)) + 2.0))
    grow_threshold = max(
        seed_threshold + 12.0,
        min(35.0, float(np.percentile(border_distance, 99.5)) + 8.0),
    )

    foreground_threshold = max(FOREGROUND_DISTANCE_THRESHOLD, grow_threshold)
    background_mask = distance < foreground_threshold
    coarse_foreground = ~background_mask
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        coarse_foreground.astype(np.uint8), connectivity=8
    )

    minimum_component_area = max(
        MIN_COMPONENT_AREA,
        round(width * height * MIN_COMPONENT_AREA_FRACTION),
    )
    components: list[np.ndarray] = []
    split_fragments: list[dict[str, object]] = []
    slot_masks = _slot_masks(slots or [], width, height)
    retained_source_component_count = 0
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        if area < minimum_component_area or component_width < 4 or component_height < 4:
            continue
        retained_source_component_count += 1
        component_mask = labels == label
        split = _split_component_by_confident_cores(
            component_mask,
            distance,
            slot_masks,
            minimum_component_area,
        )
        if split is None:
            components.append(component_mask)
            continue
        threshold, pieces, owner_slot_ids = split
        components.extend(pieces)
        split_fragments.append(
            {
                "source_component_bbox": list(_mask_bbox(component_mask)),
                "threshold": threshold,
                "owner_slot_ids": owner_slot_ids,
                "piece_areas": [int(np.count_nonzero(piece)) for piece in pieces],
            }
        )
    components.sort(key=lambda mask: (_mask_bbox(mask)[1], _mask_bbox(mask)[0]))

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_assets_dir = assets_dir or output_dir / "assets"
    canonical_assets_dir.mkdir(parents=True, exist_ok=True)

    filtered_foreground = np.zeros((height, width), dtype=np.uint8)
    asset_records: list[dict[str, object]] = []
    for index, component_mask in enumerate(components, start=1):
        filtered_foreground[component_mask] = 255
        rgba, crop_box = _refine_component(
            rgb,
            distance,
            component_mask,
            background_rgb,
            seed_threshold,
            foreground_threshold,
        )
        filename = f"asset_{index:03d}.png"
        asset_path = canonical_assets_dir / filename
        Image.fromarray(rgba, mode="RGBA").save(asset_path)
        relative_asset_path = Path(os.path.relpath(asset_path, output_dir))
        x, y, x1, y1 = _mask_bbox(component_mask)
        area = int(np.count_nonzero(component_mask))
        y_values, x_values = np.nonzero(component_mask)
        centroid = [float(x_values.mean()), float(y_values.mean())]
        asset_records.append(
            {
                "asset_id": f"asset_{index:03d}",
                "path": relative_asset_path.as_posix(),
                "component_bbox": [x, y, x1, y1],
                "crop_bbox": list(crop_box),
                "area": area,
                "centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
                "opaque_pixel_count": int(np.count_nonzero(rgba[:, :, 3] >= 128)),
            }
        )

    cv2.imwrite(str(output_dir / "background_mask.png"), background_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / "foreground_mask.png"), filtered_foreground)
    _write_contact_sheet(canonical_assets_dir, asset_records, output_dir / "contact_sheet.png")

    manifest: dict[str, object] = {
        "input": str(input_path.resolve()),
        "image_size": [width, height],
        "algorithm": "global_lab_dual_threshold_slot_aware_chroma_removal",
        "despill_algorithm": "border_sampled_key_chroma_projection",
        "background_rgb": [round(float(value), 3) for value in background_rgb],
        "background_lab": [round(float(value), 3) for value in background_lab],
        "border_width": border_width,
        "alpha_inset_pixels": ALPHA_INSET_PIXELS,
        "seed_threshold": round(seed_threshold, 3),
        "grow_threshold": round(grow_threshold, 3),
        "foreground_threshold": round(foreground_threshold, 3),
        "core_thresholds": list(CORE_DISTANCE_THRESHOLDS),
        "minimum_component_area": minimum_component_area,
        "background_pixel_fraction": round(float(background_mask.mean()), 6),
        "discarded_component_count": (component_count - 1) - retained_source_component_count,
        "asset_count": len(asset_records),
        "split_fragment_count": len(split_fragments),
        "split_fragments": split_fragments,
        "assets_dir": str(canonical_assets_dir.resolve()),
        "assets": asset_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _border_pixels(image: np.ndarray, width: int) -> np.ndarray:
    return np.concatenate(
        (
            image[:width].reshape(-1, image.shape[2]),
            image[-width:].reshape(-1, image.shape[2]),
            image[width:-width, :width].reshape(-1, image.shape[2]),
            image[width:-width, -width:].reshape(-1, image.shape[2]),
        ),
        axis=0,
    )


def _slot_masks(
    slots: list[dict[str, object]], width: int, height: int
) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = []
    for slot in slots:
        bbox = slot.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in bbox)
        mask = np.zeros((height, width), dtype=bool)
        mask[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = True
        masks.append((str(slot.get("slot_id", len(masks))), mask))
    return masks


def _split_component_by_confident_cores(
    component_mask: np.ndarray,
    distance: np.ndarray,
    slot_masks: list[tuple[str, np.ndarray]],
    minimum_component_area: int,
) -> tuple[float, list[np.ndarray], list[str]] | None:
    if len(slot_masks) < 2:
        return None

    component_area = int(np.count_nonzero(component_mask))
    minimum_core_area = max(minimum_component_area, round(component_area * CORE_AREA_FRACTION))
    minimum_owner_overlap = max(64, round(minimum_component_area * 0.25))
    for threshold in CORE_DISTANCE_THRESHOLDS:
        core_mask = component_mask & (distance >= threshold)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            core_mask.astype(np.uint8), connectivity=8
        )
        core_labels = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_core_area
        ]
        if len(core_labels) < 2:
            continue

        owner_ids: list[int] = []
        for label in core_labels:
            core = labels == label
            overlaps = [
                int(np.count_nonzero(core & slot_mask)) for _, slot_mask in slot_masks
            ]
            owner_id = int(np.argmax(overlaps))
            if overlaps[owner_id] < minimum_owner_overlap:
                break
            owner_ids.append(owner_id)
        if len(owner_ids) != len(core_labels) or len(set(owner_ids)) != len(owner_ids):
            continue

        seed_image = np.ones(component_mask.shape, dtype=np.uint8)
        for label in core_labels:
            seed_image[labels == label] = 0
        _, nearest_labels = cv2.distanceTransformWithLabels(
            seed_image,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        pieces: list[np.ndarray] = []
        nearest_owner_labels: set[int] = set()
        for label in core_labels:
            y, x = np.argwhere(labels == label)[0]
            nearest_owner_label = int(nearest_labels[y, x])
            if nearest_owner_label in nearest_owner_labels:
                pieces = []
                break
            nearest_owner_labels.add(nearest_owner_label)
            piece = component_mask & (nearest_labels == nearest_owner_label)
            if int(np.count_nonzero(piece)) < minimum_component_area:
                pieces = []
                break
            pieces.append(piece)
        if len(pieces) == len(core_labels):
            return threshold, pieces, [slot_masks[owner_id][0] for owner_id in owner_ids]
    return None


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y_values, x_values = np.nonzero(mask)
    return (
        int(x_values.min()),
        int(y_values.min()),
        int(x_values.max()) + 1,
        int(y_values.max()) + 1,
    )


def _refine_component(
    rgb: np.ndarray,
    distance: np.ndarray,
    component_mask: np.ndarray,
    background_rgb: np.ndarray,
    seed_threshold: float,
    grow_threshold: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    y_values, x_values = np.nonzero(component_mask)
    margin = EDGE_RADIUS + CROP_PADDING
    x0 = max(0, int(x_values.min()) - margin)
    y0 = max(0, int(y_values.min()) - margin)
    x1 = min(rgb.shape[1], int(x_values.max()) + margin + 1)
    y1 = min(rgb.shape[0], int(y_values.max()) + margin + 1)

    local_mask = component_mask[y0:y1, x0:x1].astype(np.uint8)
    local_distance = distance[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (EDGE_RADIUS * 2 + 1, EDGE_RADIUS * 2 + 1)
    )
    inner = cv2.erode(local_mask, kernel).astype(bool)
    outer = cv2.dilate(local_mask, kernel).astype(bool)
    geometric_alpha = cv2.GaussianBlur(
        local_mask.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0
    )
    color_alpha = np.clip(
        (local_distance - seed_threshold) / max(1e-6, grow_threshold - seed_threshold),
        0.0,
        1.0,
    )
    observed = rgb[y0:y1, x0:x1].astype(np.float32)
    dominance_alpha = 1.0 - _key_spill_strength(observed, background_rgb)
    edge_color_alpha = np.minimum(color_alpha, dominance_alpha)

    alpha = np.zeros(local_mask.shape, dtype=np.float32)
    alpha[inner] = 1.0
    inside_edge = outer & ~inner & local_mask.astype(bool)
    outside_edge = outer & ~local_mask.astype(bool)
    alpha[inside_edge] = np.minimum(geometric_alpha[inside_edge], edge_color_alpha[inside_edge])
    alpha[outside_edge] = np.minimum(geometric_alpha[outside_edge], edge_color_alpha[outside_edge])

    refined_rgb = observed.copy()
    partial = (alpha > 0.01) & (alpha < 0.999)
    safe_alpha = np.maximum(alpha[:, :, None], 0.05)
    unmixed = (observed - (1.0 - alpha[:, :, None]) * background_rgb) / safe_alpha
    refined_rgb[partial] = np.clip(unmixed[partial], 0.0, 255.0)

    refined_rgb = _despill_key_color(refined_rgb, background_rgb, partial)
    alpha = cv2.erode(
        alpha,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=ALPHA_INSET_PIXELS,
    )

    rgba = np.dstack((np.rint(refined_rgb).astype(np.uint8), np.rint(alpha * 255).astype(np.uint8)))
    rgba[rgba[:, :, 3] < 8, 3] = 0
    rgba[rgba[:, :, 3] == 0, :3] = 0
    return rgba, (x0, y0, x1, y1)


def _key_spill_strength(pixels: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    """Estimate contamination using the realized key chroma, including magenta."""
    key = np.asarray(background_rgb, dtype=np.float32)
    key_chroma = key - float(key.mean())
    key_norm = float(np.linalg.norm(key_chroma))
    if key_norm < KEY_CHROMA_EPSILON:
        return np.zeros(pixels.shape[:2], dtype=np.float32)
    observed = np.asarray(pixels, dtype=np.float32)
    pixel_chroma = observed - observed.mean(axis=2, keepdims=True)
    pixel_norm = np.linalg.norm(pixel_chroma, axis=2)
    projection = np.einsum("...c,c->...", pixel_chroma, key_chroma)
    cosine = np.divide(
        projection, pixel_norm * key_norm,
        out=np.zeros_like(projection, dtype=np.float32),
        where=pixel_norm > KEY_CHROMA_EPSILON,
    )
    hue_alignment = np.clip(
        (cosine - KEY_HUE_ALIGNMENT_FLOOR) / (1.0 - KEY_HUE_ALIGNMENT_FLOOR), 0.0, 1.0
    )
    relative_chroma = np.clip(projection / (key_norm * key_norm), 0.0, 1.0)
    return (hue_alignment * relative_chroma).astype(np.float32)


def _despill_key_color(
    pixels: np.ndarray, background_rgb: np.ndarray, partial_alpha: np.ndarray
) -> np.ndarray:
    """Remove sampled key spill only on the existing partially opaque edge."""
    refined = np.asarray(pixels, dtype=np.float32).copy()
    strength = np.where(partial_alpha, _key_spill_strength(refined, background_rgb), 0.0)
    key = np.asarray(background_rgb, dtype=np.float32)
    refined -= strength[:, :, None] * (key - float(key.min()))
    return np.clip(refined, 0.0, 255.0)


def _write_contact_sheet(
    assets_dir: Path, assets: list[dict[str, object]], output_path: Path
) -> None:
    columns = 4
    cell_width, cell_height = 300, 210
    rows = max(1, (len(assets) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(assets):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        checker = Image.new("RGB", (cell_width - 20, cell_height - 40), (238, 238, 238))
        checker_draw = ImageDraw.Draw(checker)
        for cy in range(0, checker.height, 16):
            for cx in range(0, checker.width, 16):
                if (cx // 16 + cy // 16) % 2:
                    checker_draw.rectangle((cx, cy, cx + 15, cy + 15), fill=(210, 210, 210))
        asset = Image.open(assets_dir / Path(str(record["path"])).name).convert("RGBA")
        asset.thumbnail((checker.width - 16, checker.height - 16), Image.Resampling.LANCZOS)
        paste_x = (checker.width - asset.width) // 2
        paste_y = (checker.height - asset.height) // 2
        checker.paste(asset, (paste_x, paste_y), asset)
        sheet.paste(checker, (x + 10, y + 24))
        draw.text((x + 10, y + 6), str(record["asset_id"]), fill=(20, 20, 20))
    sheet.save(output_path)
