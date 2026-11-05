from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .style_models import StyleAssetSlot, StyleReferencePlan


def render_style_reference_controls(plan: StyleReferencePlan, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "style_reference_control.png"
    mask_path = output_dir / "style_reference_mask.png"
    guide_path = output_dir / "style_reference_guide.png"
    key = hex_to_rgb(plan.key_color)

    Image.new("RGB", plan.board_size, key).save(base_path)

    mask = Image.new("RGBA", plan.board_size, (*key, 255))
    mask_draw = ImageDraw.Draw(mask, "RGBA")
    for slot in plan.slots:
        if slot.enabled:
            _draw_slot_shape(mask_draw, slot, plan.board_size, fill=(*key, 0), inset=6)
    mask.save(mask_path)

    guide = Image.new("RGB", plan.board_size, (248, 249, 250))
    guide_draw = ImageDraw.Draw(guide)
    font = ImageFont.load_default()
    colors = {
        "title_bar": (42, 84, 132),
        "subtitle_bar": (87, 140, 132),
        "raster_holder": (225, 151, 66),
        "identity_asset": (188, 100, 126),
        "badge": (88, 151, 102),
        "icon": (107, 104, 161),
    }
    for slot in plan.slots:
        box = normalized_bbox_to_pixels(slot.bbox, plan.board_size)
        fill = colors[slot.category] if slot.enabled else (218, 221, 225)
        _draw_slot_shape(guide_draw, slot, plan.board_size, fill=fill, outline=(40, 44, 48), inset=2)
        label = slot.slot_id if slot.enabled else f"{slot.slot_id} (disabled)"
        text_color = (255, 255, 255) if slot.enabled else (75, 78, 82)
        guide_draw.text((box[0] + 8, box[1] + 7), label, fill=text_color, font=font)
    guide.save(guide_path)
    return {"control": base_path, "mask": mask_path, "guide": guide_path}


def render_mock_style_reference(path: Path, plan: StyleReferencePlan) -> None:
    key = hex_to_rgb(plan.key_color)
    image = Image.new("RGB", plan.board_size, key)
    draw = ImageDraw.Draw(image)
    palette = [(33, 50, 67), (39, 139, 137), (237, 101, 84), (244, 196, 74), (238, 242, 239)]

    identity_index = 0
    for index, slot in enumerate(slot for slot in plan.slots if slot.enabled):
        box = _inset_box(normalized_bbox_to_pixels(slot.bbox, plan.board_size), 12)
        accent = palette[index % 4]
        light = palette[4]
        if slot.category in {"title_bar", "subtitle_bar"}:
            radius = max(10, (box[3] - box[1]) // 3)
            draw.rounded_rectangle(box, radius=radius, fill=accent, outline=light, width=4)
        elif slot.category == "raster_holder":
            _draw_mock_holder(draw, box, slot.shape, accent, light)
        elif slot.category == "identity_asset":
            if plan.avatar_policy == "included":
                _draw_mock_avatar(draw, box, accent, pose=identity_index)
            else:
                _draw_mock_thematic_asset(draw, box, accent, light, identity_index)
            identity_index += 1
        elif slot.category == "badge":
            _draw_mock_badge(draw, box, accent, light, index)
        else:
            _draw_mock_icon(draw, box, accent, light, index)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_mock_background(path: Path, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (22, 28, 38))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, round(size[1] * 0.72), size[0], size[1]), fill=(31, 43, 53))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def normalized_bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = bbox
    return (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _draw_slot_shape(
    draw: ImageDraw.ImageDraw,
    slot: StyleAssetSlot,
    size: tuple[int, int],
    *,
    fill: Any,
    outline: Any | None = None,
    inset: int,
) -> None:
    box = _inset_box(normalized_bbox_to_pixels(slot.bbox, size), inset)
    radius = max(8, min(box[2] - box[0], box[3] - box[1]) // 5)
    if slot.shape == "freeform":
        draw.rectangle(box, fill=fill, outline=outline, width=2)
    elif slot.shape == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=3)
    elif slot.shape == "pill":
        draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=outline, width=3)
    elif slot.shape in {"speech_bubble", "callout"}:
        tail_width = max(14, (box[2] - box[0]) // 8)
        body = (box[0], box[1], box[2], box[3] - tail_width)
        draw.rounded_rectangle(body, radius=radius, fill=fill, outline=outline, width=3)
        if slot.shape == "speech_bubble":
            tail = [
                (box[0] + tail_width, body[3] - 2),
                (box[0] + tail_width * 2, body[3] - 2),
                (box[0] + tail_width, box[3]),
            ]
        else:
            tail = [
                (box[2] - tail_width * 2, body[3] - 2),
                (box[2] - tail_width, body[3] - 2),
                (box[2] - tail_width // 2, box[3]),
            ]
        draw.polygon(tail, fill=fill)
    else:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)


def _draw_mock_holder(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    shape: str,
    accent: tuple[int, int, int],
    light: tuple[int, int, int],
) -> None:
    width = max(5, (box[3] - box[1]) // 18)
    if shape == "circle":
        draw.ellipse(box, fill=light, outline=accent, width=width)
    elif shape == "pill":
        draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=light, outline=accent, width=width)
    else:
        radius = max(12, (box[3] - box[1]) // 5)
        tail = max(16, (box[3] - box[1]) // 6) if shape in {"speech_bubble", "callout"} else 0
        body = (box[0], box[1], box[2], box[3] - tail)
        draw.rounded_rectangle(body, radius=radius, fill=light, outline=accent, width=width)
        if tail:
            if shape == "speech_bubble":
                points = [(box[0] + 30, body[3] - 3), (box[0] + 75, body[3] - 3), (box[0] + 38, box[3])]
            else:
                points = [(box[2] - 80, body[3] - 3), (box[2] - 35, body[3] - 3), (box[2] - 45, box[3])]
            draw.polygon(points, fill=light)
            draw.line(points + [points[0]], fill=accent, width=width)


def _draw_mock_avatar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    pose: int,
) -> None:
    center_x = (box[0] + box[2]) // 2
    head_radius = max(18, (box[2] - box[0]) // 8)
    head_y = box[1] + head_radius + 12
    light = (238, 242, 239)
    head = (center_x - head_radius, head_y - head_radius, center_x + head_radius, head_y + head_radius)
    draw.ellipse(head, fill=light, outline=accent, width=5)
    body = (center_x - head_radius - 8, head_y + head_radius - 2, center_x + head_radius + 8, box[3] - 24)
    draw.rounded_rectangle(body, radius=head_radius, fill=accent, outline=light, width=4)
    eye_y = head_y - 4
    draw.ellipse((center_x - 14, eye_y, center_x - 7, eye_y + 8), fill=accent)
    draw.ellipse((center_x + 7, eye_y, center_x + 14, eye_y + 8), fill=accent)
    arm_y = body[1] + 24
    if pose == 0:
        draw.line((body[0], arm_y, body[0] - 28, arm_y + 42), fill=accent, width=10)
        draw.line((body[2], arm_y, body[2] + 28, arm_y + 42), fill=accent, width=10)
    elif pose == 1:
        draw.line((body[0], arm_y, body[0] - 38, arm_y - 22), fill=accent, width=10)
        draw.line((body[2], arm_y, body[2] + 38, arm_y + 34), fill=accent, width=10)
    else:
        draw.line((body[0], arm_y, body[0] - 34, arm_y + 30), fill=accent, width=10)
        draw.line((body[2], arm_y, center_x + 18, head_y + 16), fill=accent, width=10)


def _draw_mock_thematic_asset(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    light: tuple[int, int, int],
    variant: int,
) -> None:
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    radius = max(24, min(box[2] - box[0], box[3] - box[1]) // 4)
    outer = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    draw.ellipse(outer, fill=light, outline=accent, width=7)
    if variant % 3 == 0:
        draw.arc(
            (center_x - radius - 24, center_y - radius // 2, center_x + radius + 24, center_y + radius // 2),
            15,
            345,
            fill=accent,
            width=8,
        )
        draw.ellipse((center_x - 12, center_y - 12, center_x + 12, center_y + 12), fill=accent)
    elif variant % 3 == 1:
        draw.line(
            (center_x - radius, center_y + radius // 2, center_x, center_y - radius, center_x + radius, center_y + radius // 2),
            fill=accent,
            width=9,
            joint="curve",
        )
        draw.ellipse((center_x - 12, center_y - radius - 12, center_x + 12, center_y - radius + 12), fill=accent)
    else:
        body = (center_x - radius, center_y - radius // 2, center_x + radius, center_y + radius // 2)
        draw.rounded_rectangle(body, radius=18, outline=accent, width=8)
        draw.line((center_x - radius - 24, center_y, center_x + radius + 24, center_y), fill=accent, width=7)


def _draw_mock_badge(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    light: tuple[int, int, int],
    variant: int,
) -> None:
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    half_width = max(18, (box[2] - box[0]) // 3)
    half_height = max(18, (box[3] - box[1]) // 3)
    kind = variant % 6
    if kind == 0:
        draw.polygon(_star_points(center_x, center_y, half_width, half_width // 2, 8), fill=accent)
    elif kind == 1:
        draw.polygon(
            [
                (center_x - half_width, center_y - half_height),
                (center_x + half_width, center_y - half_height),
                (center_x + half_width * 3 // 4, center_y + half_height // 2),
                (center_x, center_y + half_height),
                (center_x - half_width * 3 // 4, center_y + half_height // 2),
            ],
            fill=accent,
        )
    elif kind == 2:
        draw.polygon(
            [
                (center_x - half_width, center_y - half_height // 2),
                (center_x + half_width, center_y - half_height // 2),
                (center_x + half_width * 2 // 3, center_y),
                (center_x + half_width, center_y + half_height // 2),
                (center_x - half_width, center_y + half_height // 2),
                (center_x - half_width * 2 // 3, center_y),
            ],
            fill=accent,
        )
    elif kind == 3:
        draw.regular_polygon((center_x, center_y, half_width), n_sides=6, fill=accent)
    elif kind == 4:
        draw.ellipse(
            (center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height),
            fill=accent,
        )
        draw.ellipse(
            (center_x - half_width // 2, center_y - half_height // 2, center_x + half_width // 2, center_y + half_height // 2),
            fill=light,
        )
    else:
        draw.polygon(
            [
                (center_x - half_width, center_y - half_height),
                (center_x + half_width, center_y - half_height * 2 // 3),
                (center_x + half_width * 2 // 3, center_y + half_height),
                (center_x, center_y + half_height * 2 // 3),
                (center_x - half_width * 2 // 3, center_y + half_height),
            ],
            fill=accent,
        )


def _draw_mock_icon(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    light: tuple[int, int, int],
    variant: int,
) -> None:
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    radius = max(12, min(box[2] - box[0], box[3] - box[1]) // 5)
    if variant % 5 == 0:
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=accent,
            width=6,
        )
        draw.line((center_x - radius - 10, center_y, center_x + radius + 10, center_y), fill=accent, width=5)
    elif variant % 5 == 1:
        draw.line(
            (center_x - radius, center_y + radius, center_x, center_y - radius, center_x + radius, center_y + radius),
            fill=accent,
            width=6,
            joint="curve",
        )
    elif variant % 5 == 2:
        inner = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        draw.rounded_rectangle(inner, radius=6, outline=accent, width=6)
        draw.line((center_x - radius - 10, center_y, center_x + radius + 10, center_y), fill=accent, width=5)
    elif variant % 5 == 3:
        draw.arc(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            25,
            315,
            fill=accent,
            width=6,
        )
        draw.polygon(
            [(center_x + radius - 5, center_y - 8), (center_x + radius + 10, center_y), (center_x + radius - 4, center_y + 8)],
            fill=accent,
        )
    else:
        draw.line((center_x - radius, center_y + radius, center_x, center_y - radius), fill=accent, width=6)
        draw.line((center_x, center_y - radius, center_x + radius, center_y + radius), fill=accent, width=6)
        draw.ellipse((center_x - 7, center_y - radius - 7, center_x + 7, center_y - radius + 7), fill=accent)


def _star_points(center_x: int, center_y: int, outer_radius: int, inner_radius: int, points: int) -> list[tuple[int, int]]:
    coordinates = []
    for index in range(points * 2):
        angle = -math.pi / 2 + index * math.pi / points
        radius = outer_radius if index % 2 == 0 else inner_radius
        coordinates.append((round(center_x + math.cos(angle) * radius), round(center_y + math.sin(angle) * radius)))
    return coordinates


def _inset_box(box: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    return (box[0] + amount, box[1] + amount, box[2] - amount, box[3] - amount)
