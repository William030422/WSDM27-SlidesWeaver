from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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
    best: tuple[int, tuple[int, int, int, int], np.ndarray] | None = None
    for candidate_key in np.argsort(counts)[-8:][::-1]:
        if counts[candidate_key] == 0:
            continue
        fill_rgb = np.median(opaque_rgb[keys == candidate_key], axis=0).astype(np.uint8)
        fill_lab = cv2.cvtColor(fill_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3)
        color_distance = np.linalg.norm(image_lab - fill_lab, axis=2)
        interior_mask = ((holder_mask > 0) & (color_distance <= 24.0)).astype(np.uint8)
        safe_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_CLOSE, kernel)
        safe_mask = _largest_component(safe_mask)
        # Closing may help identify a surface, but its filled holes are not
        # usable pixels. Keep the final rectangle on the original opaque fill.
        safe_mask &= interior_mask
        safe_mask &= (alpha >= 224).astype(np.uint8)
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
