from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_MAX_DISTANCE = 0.10
HOLE_STRICT_THRESHOLD = 10.0
HOLE_LOOSE_THRESHOLD = 24.0
HOLE_SOFT_THRESHOLD = 30.0
MIN_HOLE_AREA = 9
MICRO_HOLE_MAX_CHROMA_DISTANCE = 0.05


def name_segmented_assets(
    segmented_manifest_path: Path,
    reference_plan_path: Path,
    output_dir: Path,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> dict[str, object]:
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    segmented = json.loads(segmented_manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_plan_path.read_text(encoding="utf-8"))
    image_width, image_height = segmented["image_size"]
    fragments = segmented["assets"]
    slots = [slot for slot in reference["slots"] if slot.get("enabled", True)]
    source_rgb = None
    background_lab = None
    background_rgb = None
    if segmented.get("input") and segmented.get("background_lab") and segmented.get("background_rgb"):
        source_rgb = np.asarray(Image.open(segmented["input"]).convert("RGB"))
        background_lab = np.asarray(segmented["background_lab"], dtype=np.float32)
        background_rgb = np.asarray(segmented["background_rgb"], dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    slot_centers = np.asarray([_bbox_center(slot["bbox"]) for slot in slots], dtype=np.float32)

    processed_fragments: list[dict[str, object]] = []
    for fragment in fragments:
        source_path = segmented_manifest_path.parent / fragment["path"]
        with Image.open(source_path).convert("RGBA") as source:
            rgba = np.asarray(source).copy()
        x0, y0, _, _ = fragment["crop_bbox"]
        recovered_hole_pixels = 0
        if source_rgb is not None and background_lab is not None and background_rgb is not None:
            rgba, recovered_hole_pixels = _recover_internal_key_holes(
                source_rgb[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]],
                rgba,
                background_lab,
                background_rgb,
            )

        destination_path = output_dir / Path(fragment["path"]).name
        Image.fromarray(rgba, mode="RGBA").save(destination_path)
        fragment["path"] = Path("..") / output_dir.name / destination_path.name
        fragment["path"] = Path(fragment["path"]).as_posix()
        fragment["opaque_pixel_count"] = int(np.count_nonzero(rgba[:, :, 3] >= 128))
        fragment["recovered_inner_hole_pixels"] = recovered_hole_pixels

        center = np.asarray(
            _bbox_center(fragment["component_bbox"], image_width, image_height),
            dtype=np.float32,
        )
        processed_fragments.append(
            {
                "fragment": fragment,
                "center": center,
                "recovered_inner_hole_pixels": recovered_hole_pixels,
            }
        )

    segmented["postprocessing_algorithm"] = "enclosed_key_hole_recovery"
    segmented["postprocessed_asset_count"] = len(processed_fragments)
    segmented["recovered_inner_hole_pixels"] = sum(
        int(item["recovered_inner_hole_pixels"]) for item in processed_fragments
    )
    segmented_manifest_path.write_text(
        json.dumps(segmented, indent=2) + "\n", encoding="utf-8"
    )

    assignment_by_slot = _minimum_cost_assignment(slot_centers, processed_fragments)

    accepted_fragments: set[str] = set()
    asset_records: list[dict[str, object]] = []
    missing_slots: list[dict[str, object]] = []
    for slot_index, (slot, reference_center) in enumerate(zip(slots, slot_centers)):
        slot_id = slot["slot_id"]
        fragment_index = assignment_by_slot.get(slot_index)
        if fragment_index is None:
            source_fragment_ids: list[str] = []
            component_bbox = None
            crop_bbox = _pixel_bbox(slot["bbox"], image_width, image_height)
            center = reference_center
            distance = float("inf")
            path = None
            recovered_hole_pixels = 0
        else:
            item = processed_fragments[fragment_index]
            fragment = item["fragment"]
            source_fragment_ids = [str(fragment["asset_id"])]
            component_bbox = tuple(int(value) for value in fragment["component_bbox"])
            crop_bbox = tuple(int(value) for value in fragment["crop_bbox"])
            center = np.asarray(
                _bbox_center(component_bbox, image_width, image_height), dtype=np.float32
            )
            distance = float(np.linalg.norm(center - reference_center))
            path = Path(str(fragment["path"])).name
            recovered_hole_pixels = int(item["recovered_inner_hole_pixels"])

        record: dict[str, object] = {
            "asset_id": slot_id,
            "path": path,
            "source_fragment_ids": source_fragment_ids,
            "fragment_count": len(source_fragment_ids),
            "recovered_inner_hole_pixels": recovered_hole_pixels,
            "component_bbox": list(component_bbox) if component_bbox else None,
            "crop_bbox": list(crop_bbox),
            "normalized_center": _rounded_pair(center),
            "reference_slot_id": slot_id,
            "reference_center": _rounded_pair(reference_center),
            "matching_distance": round(distance, 6) if np.isfinite(distance) else None,
        }
        if fragment_index is not None and distance < max_distance:
            record["status"] = "matched"
            accepted_fragments.update(source_fragment_ids)
        else:
            record["status"] = "missing_slot" if component_bbox is None else "misaligned_slot"
            missing_slots.append(
                {
                    "slot_id": slot_id,
                    "normalized_center": _rounded_pair(reference_center),
                    "status": record["status"],
                }
            )
        asset_records.append(record)

    unmatched_fragments = [
        fragment["asset_id"]
        for fragment in fragments
        if fragment["asset_id"] not in accepted_fragments
    ]
    manifest: dict[str, object] = {
        "segmented_manifest": str(segmented_manifest_path.resolve()),
        "reference_plan": str(reference_plan_path.resolve()),
        "image_size": segmented["image_size"],
        "reference_board_size": reference["board_size"],
        "segmentation_algorithm": segmented["algorithm"],
        "algorithm": "global_minimum_cost_slot_matching",
        "max_distance": max_distance,
        "island_count": len(fragments),
        "fragment_count": len(fragments),
        "reference_slot_count": len(slots),
        "matched_count": len(slots) - len(missing_slots),
        "unmatched_island_count": len(unmatched_fragments),
        "unmatched_fragment_count": len(unmatched_fragments),
        "missing_slot_count": len(missing_slots),
        "unmatched_islands": unmatched_fragments,
        "unmatched_fragments": unmatched_fragments,
        "split_fragment_count": int(segmented.get("split_fragment_count", 0)),
        "split_fragments": segmented.get("split_fragments", []),
        "postprocessing_algorithm": "enclosed_key_hole_recovery",
        "postprocessed_asset_count": len(processed_fragments),
        "recovered_inner_hole_pixels": sum(
            int(record["recovered_inner_hole_pixels"]) for record in asset_records
        ),
        "missing_slots": missing_slots,
        "assets": asset_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _bbox_center(
    bbox: list[float], width: float = 1.0, height: float = 1.0
) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / (2 * width), (y0 + y1) / (2 * height)


def _minimum_cost_assignment(
    slot_centers: np.ndarray,
    processed_fragments: list[dict[str, object]],
) -> dict[int, int]:
    if len(slot_centers) == 0 or not processed_fragments:
        return {}
    fragment_centers = np.asarray(
        [item["center"] for item in processed_fragments], dtype=np.float64
    )
    costs = np.linalg.norm(
        slot_centers.astype(np.float64)[:, None, :] - fragment_centers[None, :, :],
        axis=2,
    )
    return {row: column for row, column in _linear_sum_assignment(costs)}


def _linear_sum_assignment(costs: np.ndarray) -> list[tuple[int, int]]:
    rows, columns = costs.shape
    if rows == 0 or columns == 0:
        return []

    size = max(rows, columns)
    padding_cost = max(2.0, float(costs.max()) + 1.0)
    padded = np.full((size, size), padding_cost, dtype=np.float64)
    padded[:rows, :columns] = costs

    row_potential = np.zeros(size + 1, dtype=np.float64)
    column_potential = np.zeros(size + 1, dtype=np.float64)
    matched_row = np.zeros(size + 1, dtype=np.int32)
    previous_column = np.zeros(size + 1, dtype=np.int32)
    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = int(matched_row[column])
            delta = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced_cost = (
                    padded[current_row - 1, candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced_cost < minimum[candidate]:
                    minimum[candidate] = reduced_cost
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = int(previous_column[column])
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignments: list[tuple[int, int]] = []
    for column in range(1, size + 1):
        row = int(matched_row[column])
        if 1 <= row <= rows and column <= columns:
            assignments.append((row - 1, column - 1))
    return assignments


def _pixel_bbox(
    bbox: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)


def _recover_internal_key_holes(
    source_rgb: np.ndarray,
    rgba: np.ndarray,
    background_lab: np.ndarray,
    background_rgb: np.ndarray,
) -> tuple[np.ndarray, int]:
    lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - background_lab, axis=2)
    strict_key = distance < HOLE_STRICT_THRESHOLD
    loose_key = distance < HOLE_LOOSE_THRESHOLD
    seeds = np.zeros(loose_key.shape, dtype=bool)
    seeds[:2, :] = strict_key[:2, :]
    seeds[-2:, :] = strict_key[-2:, :]
    seeds[:, :2] |= strict_key[:, :2]
    seeds[:, -2:] |= strict_key[:, -2:]

    _, key_labels = cv2.connectedComponents(loose_key.astype(np.uint8), connectivity=8)
    external_labels = np.unique(key_labels[seeds])
    external_labels = external_labels[external_labels != 0]
    external_key = np.isin(key_labels, external_labels)

    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    candidates = loose_key & ~external_key & (alpha > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8
    )
    strong_wall = (alpha > 0.8) & ~loose_key
    edge_map = cv2.Canny(source_rgb, 80, 160) > 0
    accepted = np.zeros(candidates.shape, dtype=bool)
    accepted_micro = np.zeros(candidates.shape, dtype=bool)
    accepted_large = np.zeros(candidates.shape, dtype=bool)
    foreground_area = max(1, int(np.count_nonzero(alpha > 0)))
    micro_hole_max_area = min(256, max(32, round(alpha.size * 0.003)))
    background_chroma = background_rgb / max(float(background_rgb.sum()), 1e-6)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_HOLE_AREA or area > 0.7 * foreground_area:
            continue
        component = labels == label
        if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
            continue
        neighborhood = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
        purity = float(np.count_nonzero(component & strict_key)) / area
        surround = float(np.count_nonzero(neighborhood & strong_wall)) / max(
            1, int(np.count_nonzero(neighborhood))
        )
        edge_density = float(np.count_nonzero(component & edge_map)) / area
        strict_accept = (
            purity >= 0.7
            and distance[component].mean() <= 18
            and surround >= 0.45
            and edge_density <= 0.12
        )
        hole_rgb = source_rgb[component].mean(axis=0)
        hole_chroma = hole_rgb / max(float(hole_rgb.sum()), 1e-6)
        micro_accept = (
            area <= micro_hole_max_area
            and surround >= 0.95
            and edge_density <= 0.06
            and np.linalg.norm(hole_chroma - background_chroma)
            <= MICRO_HOLE_MAX_CHROMA_DISTANCE
        )
        if strict_accept or micro_accept:
            accepted |= component
            if micro_accept:
                accepted_micro |= component
            if area > micro_hole_max_area:
                accepted_large |= component

    if not accepted.any():
        return rgba, 0

    transparent = (accepted & strict_key) | accepted_micro
    transparent |= cv2.dilate(accepted_large.astype(np.uint8), kernel).astype(bool)
    soft = cv2.dilate(transparent.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(bool)
    soft &= distance < HOLE_SOFT_THRESHOLD
    t = np.clip(
        (distance - HOLE_STRICT_THRESHOLD)
        / (HOLE_SOFT_THRESHOLD - HOLE_STRICT_THRESHOLD),
        0.0,
        1.0,
    )
    key_alpha = t * t * (3.0 - 2.0 * t)
    alpha[transparent] = 0.0
    alpha[soft] = np.minimum(alpha[soft], key_alpha[soft])
    refined = rgba.copy()
    refined[:, :, 3] = np.rint(alpha * 255).astype(np.uint8)
    refined[refined[:, :, 3] == 0, :3] = 0
    return refined, int(np.count_nonzero(accepted))


def _rounded_pair(value: np.ndarray) -> list[float]:
    return [round(float(item), 6) for item in value]
