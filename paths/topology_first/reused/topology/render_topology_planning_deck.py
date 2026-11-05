from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin

from extract_layout_topology_with_vlm import (
    GRID,
    ROOT,
    TopologyResult,
    derive_topology_boxes,
)


DEFAULT_INPUT_DIR = ROOT / "gpt_5_6_terra_layout_identity_v3"
CANVAS = (1536, 864)
CELL = 24


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def grid_box(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (x * CELL, y * CELL, (x + width) * CELL - 1, (y + height) * CELL - 1)


def center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, width, height = bbox
    return ((2 * x + width) * CELL // 2, (2 * y + height) * CELL // 2)


def clipped(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    label_font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=label_font)
    draw.rectangle(
        (box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2),
        fill=(255, 255, 255),
        outline=(218, 222, 228),
        width=1,
    )
    draw.text((x, y), text, fill=color, font=label_font)


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int = 3,
    dash: int = 10,
    gap: int = 7,
) -> None:
    left, top, right, bottom = box
    for start in range(left, right, dash + gap):
        draw.line((start, top, min(start + dash, right), top), fill=color, width=width)
        draw.line((start, bottom, min(start + dash, right), bottom), fill=color, width=width)
    for start in range(top, bottom, dash + gap):
        draw.line((left, start, left, min(start + dash, bottom)), fill=color, width=width)
        draw.line((right, start, right, min(start + dash, bottom)), fill=color, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    draw.line((start, end), fill=color, width=3)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    base = (end[0] - ux * 13, end[1] - uy * 13)
    draw.polygon(
        [
            tip,
            (base[0] + px * 6, base[1] + py * 6),
            (base[0] - px * 6, base[1] - py * 6),
        ],
        fill=color,
    )


def render_slide(slide_dir: Path, output_path: Path) -> dict:
    regions_payload = json.loads((slide_dir / "regions.json").read_text(encoding="utf-8"))
    regions = regions_payload["regions"]
    result = TopologyResult.model_validate_json((slide_dir / "topology.json").read_text(encoding="utf-8"))
    region_boxes = {item["region_id"]: tuple(item["bbox"]) for item in regions}
    component_boxes, group_boxes = derive_topology_boxes(result, regions)
    all_boxes = {**region_boxes, **component_boxes, **group_boxes}

    image = Image.new("RGB", CANVAS, (247, 248, 250))
    draw = ImageDraw.Draw(image)
    for x in range(0, CANVAS[0], CELL):
        draw.line((x, 0, x, CANVAS[1]), fill=(235, 237, 240), width=1)
    for y in range(0, CANVAS[1], CELL):
        draw.line((0, y, CANVAS[0], y), fill=(235, 237, 240), width=1)

    for group in result.groups:
        box = grid_box(group_boxes[group.group_id])
        color = (112, 78, 164) if group.scope == "top_level" else (166, 91, 145)
        dashed_rectangle(draw, box, color)
        label(draw, (box[0] + 5, box[1] + 5), clipped(f"{group.group_id}  {group.role}", 52), color, font(13, True))

    role_colors = {
        "title": ((214, 226, 247), (36, 99, 186)),
        "text": ((251, 231, 208), (185, 99, 27)),
        "visual": ((215, 238, 220), (31, 132, 70)),
    }
    for region in regions:
        box = grid_box(tuple(region["bbox"]))
        fill, outline = role_colors[region["predicted_type"]]
        draw.rectangle(box, fill=fill, outline=outline, width=2)
        label(draw, (box[0] + 4, box[1] + 4), region["region_id"], outline, font(12))

    for relation in result.relations:
        if relation.source_id in all_boxes and relation.target_id in all_boxes:
            arrow(
                draw,
                center(all_boxes[relation.source_id]),
                center(all_boxes[relation.target_id]),
                (194, 48, 119),
            )

    component_colors = {
        "title": (28, 77, 151),
        "primary_visual": (14, 112, 65),
        "supporting_visual": (49, 137, 82),
        "composite_diagram": (10, 119, 139),
        "standalone_text": (184, 91, 22),
        "supporting_text": (205, 121, 36),
        "annotation": (171, 60, 128),
        "decorative": (102, 105, 112),
        "mixed": (100, 69, 151),
    }
    components = {item.component_id: item for item in result.components}
    for component in result.components:
        box = grid_box(component_boxes[component.component_id])
        color = component_colors[component.component_role]
        draw.rectangle(box, outline=color, width=5)
        label(
            draw,
            (box[0] + 5, min(box[1] + 22, box[3] - 18)),
            f"{component.component_id}  {component.component_role}",
            color,
            font(13, True),
        )

    for index, item_id in enumerate(result.reading_order, start=1):
        if item_id not in all_boxes:
            continue
        x, y = center(all_boxes[item_id])
        radius = 13
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(24, 28, 36), outline="white", width=2)
        number = str(index)
        number_font = font(12, True)
        number_box = draw.textbbox((0, 0), number, font=number_font)
        draw.text(
            (x - (number_box[2] - number_box[0]) / 2, y - (number_box[3] - number_box[1]) / 2 - 1),
            number,
            fill="white",
            font=number_font,
        )

    formula = clipped(result.global_topology.layout_formula, 115)
    meta_box = (18, 14, 1518, 52)
    draw.rectangle(meta_box, fill=(255, 255, 255), outline=(207, 211, 217), width=1)
    draw.text((30, 24), slide_dir.name, fill=(25, 29, 37), font=font(15, True))
    draw.text((155, 24), formula, fill=(65, 70, 80), font=font(14))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "slide_id": slide_dir.name,
        "output": str(output_path.resolve()),
        "composition_family": result.global_topology.composition_family,
        "layout_formula": result.global_topology.layout_formula,
        "region_count": len(regions),
        "component_count": len(components),
        "group_count": len(result.groups),
        "relation_count": len(result.relations),
    }


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumb = (512, 288)
    columns = 2
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (thumb[0] * columns, thumb[1] * rows), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize(thumb)
        sheet.paste(tile, ((index % columns) * thumb[0], (index // columns) * thumb[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render topology as an OCR-free planning deck.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "planning_deck").resolve()
    slide_dirs = sorted(path for path in input_dir.glob("slide_*"))
    if not slide_dirs:
        raise SystemExit(f"No slide directories found in {input_dir}")

    results = []
    output_paths = []
    for slide_dir in slide_dirs:
        output_path = output_dir / f"{slide_dir.name}.png"
        results.append(render_slide(slide_dir, output_path))
        output_paths.append(output_path)

    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(output_paths, contact_sheet)
    pages = [Image.open(path).convert("RGB") for path in output_paths]
    pdf_path = output_dir / "planning_deck.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=144)
    for page in pages:
        page.close()

    manifest = {
        "stage": "topology_planning_deck_render",
        "source": str(input_dir),
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "grid": {"width": GRID[0], "height": GRID[1]},
        "slide_count": len(results),
        "slides": results,
        "contact_sheet": str(contact_sheet),
        "pdf": str(pdf_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
