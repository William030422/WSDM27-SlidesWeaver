from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DIFF_THRESHOLD = 20
MIN_COMPONENT_AREA = 256
MORPH_CLOSE_SIZE = 7
MORPH_OPEN_SIZE = 3
EDGE_GUARD = 12


def decompose_background(
    *,
    canonical_path: Path,
    final_path: Path,
    region_coordinates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract coarse, movable foreground clusters from a generated background.

    The original Step 2 background remains the base layer.  This function only
    writes transparent crops of pixels that differ materially in the final
    inpainted image, using the already-planned visual regions as cluster seeds.
    """
    with Image.open(canonical_path).convert("RGB") as canonical_image:
        canonical = np.asarray(canonical_image, dtype=np.int16)
    with Image.open(final_path).convert("RGB") as final_image:
        final = np.asarray(final_image, dtype=np.int16)
    quality_warnings = []
    if canonical.shape != final.shape:
        quality_warnings.append("final_background_resized_to_canonical_for_decomposition")
        final = np.asarray(Image.fromarray(final.astype(np.uint8)).resize(
            (canonical.shape[1], canonical.shape[0]), Image.Resampling.LANCZOS
        ), dtype=np.int16)

    height, width = canonical.shape[:2]
    diff = np.max(np.abs(final - canonical), axis=2).astype(np.uint8)
    raw_mask = diff >= DIFF_THRESHOLD
    raw_mask[:EDGE_GUARD] = False
    raw_mask[-EDGE_GUARD:] = False
    raw_mask[:, :EDGE_GUARD] = False
    raw_mask[:, -EDGE_GUARD:] = False

    regions = _read_regions(region_coordinates_path, width, height)
    if not regions:
        regions = [{"label": "global", "bbox_px": [0, 0, width, height]}]
    owner_masks = _assign_region_masks(raw_mask, regions, width, height)

    output_dir.mkdir(parents=True, exist_ok=True)
    modules_dir = output_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff, mode="L").save(output_dir / "rgb_difference.png")
    Image.fromarray((raw_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "difference_mask.png"
    )

    modules: list[dict[str, Any]] = []
    reconstructed = canonical.copy()
    for index, (region, candidate) in enumerate(
        zip(regions, owner_masks, strict=True), start=1
    ):
        mask, component_count = _clean_mask(candidate)
        if int(np.count_nonzero(mask)) < MIN_COMPONENT_AREA:
            continue
        bbox = _mask_bbox(mask)
        x, y, x1, y1 = bbox
        crop = final[y:y1, x:x1].astype(np.uint8)
        alpha = (mask[y:y1, x:x1].astype(np.uint8) * 255)
        rgba = np.dstack((crop, alpha))
        asset_path = modules_dir / f"module_{index:03d}.png"
        Image.fromarray(rgba, mode="RGBA").save(asset_path)
        reconstructed[mask] = final[mask]
        modules.append(
            {
                "module_id": f"module_{index:03d}",
                "region_label": region["label"],
                "asset_path": str(asset_path.resolve()),
                "bbox_px": [x, y, x1 - x, y1 - y],
                "pixel_area": int(np.count_nonzero(mask)),
                "component_count": component_count,
                "diff_threshold": DIFF_THRESHOLD,
            }
        )

    # Preserve omitted fine arrows and edge details as one local residual layer.
    # Existing region clusters and the primary deck remain unchanged.
    residual = np.max(np.abs(reconstructed - final), axis=2) >= DIFF_THRESHOLD
    residual_area = int(np.count_nonzero(residual))
    if residual_area:
        x, y, x1, y1 = _mask_bbox(residual)
        rgba = np.dstack((final[y:y1, x:x1].astype(np.uint8),
                          residual[y:y1, x:x1].astype(np.uint8) * 255))
        asset_path = modules_dir / "module_residual_details.png"
        Image.fromarray(rgba).save(asset_path)
        modules.append({
            "module_id": "module_residual_details", "region_label": "residual_details",
            "asset_path": str(asset_path.resolve()), "bbox_px": [x, y, x1 - x, y1 - y],
            "pixel_area": residual_area, "component_count": 1, "diff_threshold": DIFF_THRESHOLD,
        })
        reconstructed[residual] = final[residual]
        quality_warnings.append("fine_or_unassigned_details_preserved_in_residual_layer")
    reconstruction_error = np.abs(reconstructed - final)

    result = {
        "status": "succeeded",
        "quality_status": "repaired" if quality_warnings else "strict_pass",
        "warnings": quality_warnings,
        "algorithm": "rgb_difference_region_seeded_cluster",
        "canonical_background": str(canonical_path.resolve()),
        "final_background": str(final_path.resolve()),
        "region_coordinates": str(region_coordinates_path.resolve()),
        "canvas_px": {"width": width, "height": height},
        "diff_threshold": DIFF_THRESHOLD,
        "raw_changed_pixel_fraction": round(float(raw_mask.mean()), 6),
        "module_count": len(modules),
        "residual_detail_pixel_count": residual_area,
        "reconstruction_mean_absolute_error": round(float(reconstruction_error.mean()), 6),
        "reconstruction_max_channel_error": int(reconstruction_error.max()),
        "modules": modules,
    }
    (output_dir / "decomposition.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _read_regions(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_regions = payload.get("regions", {})
    regions: list[dict[str, Any]] = []
    if isinstance(raw_regions, dict):
        for label, raw_bbox in raw_regions.items():
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            try:
                x, y, w, h = [int(round(float(value))) for value in raw_bbox]
            except (TypeError, ValueError):
                continue
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            w = max(1, min(width - x, w))
            h = max(1, min(height - y, h))
            regions.append({"label": str(label), "bbox_px": [x, y, w, h]})
    return regions


def _assign_region_masks(
    raw_mask: np.ndarray,
    regions: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[np.ndarray]:
    centers = np.asarray(
        [
            [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]
            for bbox in (region["bbox_px"] for region in regions)
        ],
        dtype=np.float32,
    )
    yy, xx = np.indices((height, width), dtype=np.float32)
    distance = (xx[..., None] - centers[:, 0]) ** 2 + (
        yy[..., None] - centers[:, 1]
    ) ** 2
    nearest = np.argmin(distance, axis=2)
    allowed = np.zeros_like(raw_mask, dtype=bool)
    for region in regions:
        x, y, w, h = region["bbox_px"]
        # Generated subjects commonly extend past the guide rectangle.  Keep a
        # generous local window, then use nearest-center ownership to separate
        # adjacent columns without hard-coding a slide layout.
        pad_x = max(48, round(w * 1.20))
        pad_y = max(32, round(h * 0.30))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        allowed[y0:y1, x0:x1] = True
    masks = [raw_mask & allowed & (nearest == index) for index in range(len(regions))]
    return masks


def _clean_mask(candidate: np.ndarray) -> tuple[np.ndarray, int]:
    mask = candidate.astype(np.uint8)
    close_kernel = np.ones((MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE), dtype=np.uint8)
    open_kernel = np.ones((MORPH_OPEN_SIZE, MORPH_OPEN_SIZE), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area_limit = max(MIN_COMPONENT_AREA, int(mask.size * 0.0002))
    retained = np.zeros_like(mask)
    component_count = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < area_limit:
            continue
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        thinness = max(component_width, component_height) / max(
            1, min(component_width, component_height)
        )
        # Long, thin bands are usually the generated frame or residual border
        # drift rather than a semantic object.  Keep thin arrows when they are
        # attached to a larger component, but drop isolated bands.
        if thinness > 20 and min(component_width, component_height) < 40:
            continue
        retained[labels == label] = 1
        component_count += 1
    if not component_count:
        return np.zeros_like(candidate, dtype=bool), 0
    # Preserve a small antialiased fringe around silhouettes in the crop.
    retained = cv2.dilate(retained, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return retained.astype(bool), component_count


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("cannot compute a bounding box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
