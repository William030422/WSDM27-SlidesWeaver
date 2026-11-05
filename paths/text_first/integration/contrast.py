from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


TEXT_COLOR_REVERSE_HUE_OFFSETS = (0.0, -0.035, 0.035, -0.07, 0.07)
TEXT_SAFE_INSET_PX = 3


def apply_background_aware_text_colors(plan: dict[str, Any]) -> dict[str, Any]:
    composite = compose_slide_background(plan)
    reports = []
    try:
        for element in plan.get("elements", []):
            bbox = resolved_text_bbox_px(element)
            planned_hex = normalize_hex_color(
                (element.get("font") or {}).get("color"), "111111"
            )
            background_rgb = representative_background_rgb(composite, bbox)
            final_hex, report = choose_background_aware_font_color(
                planned_hex, background_rgb, element
            )
            samples = _local_background_samples(composite, bbox)
            local_ratio = _local_contrast(final_hex, samples)
            report["local_contrast_ratio"] = round(local_ratio, 2)
            report["local_sampling_quantile"] = 0.10
            # A single representative color cannot describe a bright/dark
            # boundary. An editable text outline keeps the registered geometry.
            outline = None
            if local_ratio < report["threshold"]:
                candidates = [final_hex, planned_hex, "000000", "FFFFFF"]
                best_color = max(candidates, key=lambda color: _local_contrast(color, samples))
                local_ratio = _local_contrast(best_color, samples)
                final_hex = best_color
                if local_ratio < report["threshold"]:
                    outline_color = "FFFFFF" if relative_luminance(hex_to_rgb(final_hex)) < 0.5 else "000000"
                    outline = {"color": outline_color, "size": 0.6}
                    effective = _local_contrast(final_hex, samples, outline_color)
                    report["strategy"] = "local_contrast_native_outline"
                    report["effective_local_contrast_ratio"] = round(effective, 2)
                else:
                    report["strategy"] = "local_contrast_color"
                report["final_font_color"] = final_hex
                report["final_contrast_ratio"] = round(contrast_ratio(hex_to_rgb(final_hex), background_rgb), 2)
                report["local_contrast_ratio"] = round(local_ratio, 2)
            element.setdefault("font", {})["planned_color"] = planned_hex
            element["font"]["color"] = final_hex
            if outline is not None:
                element["font"]["outline"] = outline
            else:
                element["font"].pop("outline", None)
            report["local_readability_status"] = (
                "outlined" if outline is not None else
                "pass" if local_ratio >= report["threshold"] else "unresolved"
            )
            element["font"]["contrast"] = report
            reports.append(
                {
                    "slide_id": plan.get("slide_id"),
                    "region_id": element.get("region_id"),
                    "content_key": element.get("content_key"),
                    "role": element.get("role"),
                    "bbox_px": [round(float(value), 3) for value in bbox],
                    **report,
                }
            )
    finally:
        composite.close()
    return {
        "slide_id": plan.get("slide_id"),
        "status": (
            "strict_pass"
            if all(
                item["local_contrast_ratio"] >= item["threshold"]
                for item in reports
            )
            else "degraded"
        ),
        "blocks": reports,
    }


def _local_background_samples(
    image: Image.Image, bbox_px: list[float]
) -> list[tuple[int, int, int]]:
    x, y, width, height = _integer_bbox(bbox_px, image.size)
    with image.crop((x, y, x + width, y + height)) as crop:
        with crop.resize((min(48, width), min(48, height)), Image.Resampling.NEAREST) as sample:
            return list(sample.convert("RGB").get_flattened_data())


def _local_contrast(
    color: str, samples: list[tuple[int, int, int]], outline_color: str | None = None
) -> float:
    foreground = hex_to_rgb(color)
    outline = hex_to_rgb(outline_color) if outline_color is not None else None
    values = sorted(
        max(contrast_ratio(foreground, sample), contrast_ratio(outline, sample))
        if outline is not None else contrast_ratio(foreground, sample)
        for sample in samples
    )
    return values[min(len(values) - 1, int(len(values) * 0.10))] if values else 1.0


def compose_slide_background(plan: dict[str, Any]) -> Image.Image:
    canvas = plan.get("canvas_px") or {"width": 1536, "height": 864}
    size = (int(canvas["width"]), int(canvas["height"]))
    base = Image.open(Path(plan["background_path"])).convert("RGBA")
    if base.size != size:
        base = base.resize(size, Image.Resampling.LANCZOS)
    composite = base.copy()
    base.close()
    _draw_text_backings(composite, plan.get('elements', []), overlay=False)
    for layer in plan.get("raster_layers", []):
        path = Path(str(layer.get("path") or ""))
        if not path.is_file():
            continue
        x, y, width, height = _integer_bbox(layer.get("bbox_px"), size)
        source = Image.open(path).convert("RGBA")
        try:
            if source.size != (width, height):
                source = source.resize((width, height), Image.Resampling.LANCZOS)
            composite.alpha_composite(source, (x, y))
        finally:
            source.close()

    _draw_text_backings(composite, plan.get('elements', []), overlay=True)
    draw = ImageDraw.Draw(composite, "RGBA")
    for element in plan.get("elements", []):
        native = element.get("native_holder")
        if not isinstance(native, dict):
            continue
        x, y, width, height = _integer_bbox(native.get("bbox_px"), size)
        draw.rounded_rectangle(
            (x, y, x + width, y + height),
            radius=min(12, height // 5),
            fill=(247, 247, 244, 250),
            outline=(102, 102, 102, 255),
            width=1,
        )
    return composite.convert("RGB")


def _draw_text_backings(composite: Image.Image, elements: list[dict[str, Any]], *, overlay: bool) -> None:
    draw = ImageDraw.Draw(composite, 'RGBA')
    size = composite.size
    for element in elements:
        backing = element.get('text_backing')
        if isinstance(backing, dict) and bool(backing.get('overlay')) == overlay:
            x, y, width, height = _integer_bbox(backing['bbox_px'], size)
            polygon = backing.get('normalized_polygon')
            if polygon:
                draw.polygon([(x + a * width, y + b * height) for a, b in polygon],
                             fill=hex_to_rgb(backing['fill_color']) + (255,))
            else:
                draw.rounded_rectangle((x, y, x + width, y + height), radius=min(8, height // 8),
                                       fill=hex_to_rgb(backing['fill_color']) + (255,))


def resolved_text_bbox_px(element: dict[str, Any]) -> list[float]:
    resolved = element.get("resolved_text_bbox_px")
    if isinstance(resolved, list) and len(resolved) == 4:
        return [float(value) for value in resolved]
    holder = element.get("holder_binding")
    if isinstance(holder, dict) and isinstance(holder.get("text_safe_rect"), dict):
        base = [float(value) for value in holder["holder_bbox_px"]]
        safe = holder["text_safe_rect"]
        return [
            base[0] + base[2] * float(safe["x"]) + TEXT_SAFE_INSET_PX,
            base[1] + base[3] * float(safe["y"]) + TEXT_SAFE_INSET_PX,
            max(
                1.0,
                base[2] * float(safe["width"]) - TEXT_SAFE_INSET_PX * 2,
            ),
            max(
                1.0,
                base[3] * float(safe["height"]) - TEXT_SAFE_INSET_PX * 2,
            ),
        ]
    native = element.get("native_holder")
    if isinstance(native, dict):
        base = [float(value) for value in native["bbox_px"]]
        return [
            base[0] + base[2] * 0.07,
            base[1] + base[3] * 0.13,
            max(1.0, base[2] * 0.86),
            max(1.0, base[3] * 0.74),
        ]
    return [float(value) for value in element.get("bbox_px", [0, 0, 1, 1])]


def representative_background_rgb(
    image: Image.Image, bbox_px: Any
) -> tuple[int, int, int]:
    try:
        x, y, width, height = [float(value) for value in bbox_px]
    except (TypeError, ValueError):
        x, y, width, height = 0.0, 0.0, float(image.width), float(image.height)
    x0 = max(0, min(image.width - 1, int(round(x))))
    y0 = max(0, min(image.height - 1, int(round(y))))
    x1 = max(x0 + 1, min(image.width, int(round(x + width))))
    y1 = max(y0 + 1, min(image.height, int(round(y + height))))

    crop_width = x1 - x0
    crop_height = y1 - y0
    inset_x = int(crop_width * 0.12)
    inset_y = int(crop_height * 0.12)
    if crop_width - 2 * inset_x >= 4 and crop_height - 2 * inset_y >= 4:
        x0 += inset_x
        x1 -= inset_x
        y0 += inset_y
        y1 -= inset_y

    crop = image.crop((x0, y0, x1, y1))
    try:
        sample_width = max(1, min(48, crop.width))
        sample_height = max(1, min(48, crop.height))
        if crop.size != (sample_width, sample_height):
            crop = crop.resize((sample_width, sample_height))
        pixels = list(crop.get_flattened_data())
    finally:
        crop.close()
    if not pixels:
        return (255, 255, 255)
    channels = list(zip(*pixels))
    return tuple(
        int(sorted(channel)[len(channel) // 2]) for channel in channels
    )  # type: ignore[return-value]


def choose_background_aware_font_color(
    planned_hex: str,
    background_rgb: tuple[int, int, int],
    element: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    planned_rgb = hex_to_rgb(planned_hex)
    planned_ratio = contrast_ratio(planned_rgb, background_rgb)
    threshold = contrast_threshold_for_element(element)
    if planned_ratio >= threshold:
        return planned_hex, {
            "strategy": "planned_color_kept",
            "background_rgb": list(background_rgb),
            "planned_font_color": planned_hex,
            "final_font_color": planned_hex,
            "planned_contrast_ratio": round(planned_ratio, 2),
            "final_contrast_ratio": round(planned_ratio, 2),
            "threshold": threshold,
        }

    candidates = [
        (rgb_to_hex(rgb), rgb, contrast_ratio(rgb, background_rgb))
        for rgb in reverse_hue_text_candidates(background_rgb, planned_rgb)
    ]
    passing = [item for item in candidates if item[2] >= threshold]
    if passing:
        final_hex, _final_rgb, final_ratio = max(passing, key=lambda item: item[2])
        strategy = "background_reverse_or_adjacent_hue"
    else:
        final_hex, _final_rgb, final_ratio = max(
            (
                (rgb_to_hex(rgb), rgb, contrast_ratio(rgb, background_rgb))
                for rgb in [(255, 255, 255), (0, 0, 0)]
            ),
            key=lambda item: item[2],
        )
        strategy = "neutral_contrast_fallback"
    return final_hex, {
        "strategy": strategy,
        "background_rgb": list(background_rgb),
        "planned_font_color": planned_hex,
        "final_font_color": final_hex,
        "planned_contrast_ratio": round(planned_ratio, 2),
        "final_contrast_ratio": round(final_ratio, 2),
        "threshold": threshold,
        "reverse_hue_candidates": [
            {"color": candidate_hex, "contrast_ratio": round(candidate_ratio, 2)}
            for candidate_hex, _candidate_rgb, candidate_ratio in candidates[:10]
        ],
    }


def reverse_hue_text_candidates(
    background_rgb: tuple[int, int, int], planned_rgb: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    bg_r, bg_g, bg_b = [value / 255.0 for value in background_rgb]
    bg_h, bg_l, bg_s = colorsys.rgb_to_hls(bg_r, bg_g, bg_b)
    planned_h, _planned_l, planned_s = colorsys.rgb_to_hls(
        *(value / 255.0 for value in planned_rgb)
    )
    base_hue = (bg_h + 0.5) % 1.0
    saturation = max(
        0.55,
        min(0.95, bg_s * 1.18 if bg_s >= 0.12 else max(planned_s, 0.60)),
    )
    lightness_values = (
        (0.92, 0.84, 0.76, 0.68)
        if bg_l < 0.52
        else (0.10, 0.16, 0.23, 0.30)
    )
    candidates = []
    for offset in TEXT_COLOR_REVERSE_HUE_OFFSETS:
        hue = (base_hue + offset) % 1.0
        for lightness in lightness_values:
            candidates.append(hls_to_rgb255(hue, lightness, saturation))
    if bg_s < 0.12:
        for lightness in lightness_values:
            candidates.append(
                hls_to_rgb255(planned_h, lightness, max(planned_s, 0.35))
            )
    return list(dict.fromkeys(candidates))


def contrast_threshold_for_element(element: dict[str, Any]) -> float:
    role = str(element.get("role") or "body").lower()
    if role == "title":
        return 3.4
    if role == "subtitle":
        return 4.0
    return 4.5


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    def linearize(channel: int) -> float:
        value = channel / 255.0
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(
    foreground: tuple[int, int, int], background: tuple[int, int, int]
) -> float:
    foreground_luminance = relative_luminance(foreground)
    background_luminance = relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def normalize_hex_color(raw: Any, fallback: str) -> str:
    value = str(raw or "").strip().lstrip("#")
    if len(value) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", value):
        value = "".join(character * 2 for character in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        value = fallback
    return value.upper()


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def hex_to_rgb(raw: str) -> tuple[int, int, int]:
    value = normalize_hex_color(raw, "252B35")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def hls_to_rgb255(
    hue: float, lightness: float, saturation: float
) -> tuple[int, int, int]:
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (round(red * 255), round(green * 255), round(blue * 255))


def _integer_bbox(raw: Any, size: tuple[int, int]) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = [float(value) for value in raw]
    except (TypeError, ValueError):
        return (0, 0, 1, 1)
    left = max(0, min(size[0] - 1, int(round(x))))
    top = max(0, min(size[1] - 1, int(round(y))))
    resolved_width = max(1, min(size[0] - left, int(round(width))))
    resolved_height = max(1, min(size[1] - top, int(round(height))))
    return left, top, resolved_width, resolved_height
