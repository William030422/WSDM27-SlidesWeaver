"""Restore a quiet holder interior while retaining its original painted rim."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image



def prepare_decorative_surface(asset_path: Path, output_path: Path,
                               normalized_quiet_rect: dict[str, float]) -> dict:
    with Image.open(asset_path).convert('RGBA') as image:
        rgba = np.array(image)
    height, width = rgba.shape[:2]
    foreground = (rgba[:, :, 3] > 20).astype(np.uint8)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 3)
    # Preserve a visible painted rim, including the holder's curved silhouette.
    inside = distance > max(3., min(width, height) * .07)
    q = normalized_quiet_rect
    x, y = round(q['x'] * width), round(q['y'] * height)
    right, bottom = round((q['x'] + q['width']) * width), round((q['y'] + q['height']) * height)
    allowed = np.zeros_like(foreground)
    allowed[max(0, y):min(height, bottom), max(0, x):min(width, right)] = 1
    contours, _ = cv2.findContours((inside & (allowed > 0)).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('holder has no restorable quiet interior')
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(foreground)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    # Median color comes from the original existing interior, not a generic card.
    center = rgba[int(height * .3):max(int(height * .7), int(height * .3) + 1),
                  int(width * .3):max(int(width * .7), int(width * .3) + 1)]
    opaque = center[center[:, :, 3] > 220, :3]
    rgb = np.median(opaque, axis=0) if len(opaque) else np.array([247, 247, 244])
    rectangle = _largest_binary_rectangle(mask.astype(bool))
    if rectangle is None:
        raise ValueError('holder quiet interior has no positive rectangle')
    a, b, w, h = rectangle
    rgba[mask > 0, 3] = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(output_path)
    polygon = cv2.approxPolyDP(contour, max(.5, min(width, height) * .003), True)[:, 0, :]
    return {
        'frame_path': str(output_path.resolve()),
        'fill_color': ''.join(f'{int(v):02X}' for v in rgb),
        'normalized_polygon': [[float(px) / width, float(py) / height] for px, py in polygon],
        'text_safe_rect': {'x': a / width, 'y': b / height, 'width': w / width, 'height': h / height},
    }


def reframe_decorative_surface(asset_path: Path, output_path: Path,
                              source_safe_rect: dict[str, float],
                              target_quiet_rect: dict[str, float]) -> dict:
    """Nine-slice the original ornament around an exact reserved text area.

    Outer image bounds stay fixed. Each original safe-area edge maps to the
    matching target edge, so none of the ornamental border is covered by fill.
    """
    with Image.open(asset_path).convert('RGBA') as image:
        rgba = np.array(image)
    height, width = rgba.shape[:2]
    for rect in (source_safe_rect, target_quiet_rect):
        if not (0 < rect['x'] < rect['x'] + rect['width'] < 1
                and 0 < rect['y'] < rect['y'] + rect['height'] < 1):
            raise ValueError('reframing needs positive interior and nonzero border on every side')
    source = source_safe_rect
    target = target_quiet_rect
    sx = [0., source['x'] * width, (source['x'] + source['width']) * width, float(width - 1)]
    sy = [0., source['y'] * height, (source['y'] + source['height']) * height, float(height - 1)]
    tx = [0., target['x'] * width, (target['x'] + target['width']) * width, float(width - 1)]
    ty = [0., target['y'] * height, (target['y'] + target['height']) * height, float(height - 1)]
    if min(np.diff(tx)) <= 0 or min(np.diff(ty)) <= 0 or min(np.diff(sx)) <= 0 or min(np.diff(sy)) <= 0:
        raise ValueError('holder is too small for the requested border widths')
    map_x, map_y = np.meshgrid(np.interp(np.arange(width), tx, sx),
                              np.interp(np.arange(height), ty, sy))
    reframed = cv2.remap(rgba, map_x.astype(np.float32), map_y.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # The generated center is intentionally replaced; the four edge/corner
    # slices remain, proportionally compressed into their new reserved bands.
    left, right = int(np.ceil(tx[1])), int(np.floor(tx[2]))
    top, bottom = int(np.ceil(ty[1])), int(np.floor(ty[2]))
    reframed[top:bottom + 1, left:right + 1, 3] = 0
    a, b = int(np.ceil(sx[1])), int(np.ceil(sy[1]))
    c, d = int(np.floor(sx[2])), int(np.floor(sy[2]))
    pixels = rgba[b:d + 1, a:c + 1]
    opaque = pixels[pixels[:, :, 3] > 220, :3]
    rgb = np.median(opaque, axis=0) if len(opaque) else np.array([247, 247, 244])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reframed).save(output_path)
    x, y, w, h = (target[k] for k in ('x', 'y', 'width', 'height'))
    return {'frame_path': str(output_path.resolve()),
            'fill_color': ''.join(f'{int(v):02X}' for v in rgb),
            'normalized_polygon': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            'text_safe_rect': dict(target)}


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

