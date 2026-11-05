from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, JpegImagePlugin

from extract_calibrated_topology_with_vlm import (
    DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_DIR,
    CalibratedTopologyResult,
)
from extract_layout_topology_with_vlm import GRID, ROOT, union_bbox
from render_topology_planning_deck import (
    CANVAS,
    CELL,
    clipped,
    dashed_rectangle,
    font,
    grid_box,
    label,
    make_contact_sheet,
)


DEFAULT_OUTPUT_DIR = ROOT / "gpt_5_6_terra_layout_identity_v3" / "planning_deck_v2"
GridBBox = tuple[int, int, int, int]


def center(bbox: GridBBox) -> tuple[int, int]:
    x, y, width, height = bbox
    return ((2 * x + width) * CELL // 2, (2 * y + height) * CELL // 2)


def derive_boxes(
    result: CalibratedTopologyResult,
) -> tuple[dict[str, GridBBox], dict[str, GridBBox], dict[str, GridBBox]]:
    entity_boxes = {item.region_id: item.calibrated_bbox for item in result.entities}
    component_boxes = {
        component.component_id: union_bbox(
            [entity_boxes[region_id] for region_id in component.source_region_ids]
        )
        for component in result.components
    }
    group_map = {group.group_id: group for group in result.groups}
    group_boxes: dict[str, GridBBox] = {}
    visiting: set[str] = set()

    def group_box(group_id: str) -> GridBBox:
        if group_id in group_boxes:
            return group_boxes[group_id]
        if group_id in visiting:
            raise ValueError(f"group hierarchy contains a cycle at {group_id}")
        visiting.add(group_id)
        boxes = [
            group_box(child) if child in group_map else component_boxes[child]
            for child in group_map[group_id].children
        ]
        group_boxes[group_id] = union_bbox(boxes)
        visiting.remove(group_id)
        return group_boxes[group_id]

    for group_id in group_map:
        group_box(group_id)
    return entity_boxes, component_boxes, group_boxes


def render_slide(slide_dir: Path, output_path: Path) -> dict:
    regions_payload = json.loads((slide_dir / "regions.json").read_text(encoding="utf-8"))
    observed_boxes = {
        item["region_id"]: tuple(item["bbox"]) for item in regions_payload["regions"]
    }
    result = CalibratedTopologyResult.model_validate_json(
        (slide_dir / "calibrated_topology.json").read_text(encoding="utf-8")
    )
    report = json.loads((slide_dir / "calibration_report.json").read_text(encoding="utf-8"))
    entity_boxes, component_boxes, group_boxes = derive_boxes(result)
    all_boxes = {**entity_boxes, **component_boxes, **group_boxes}

    image = Image.new("RGB", CANVAS, (247, 248, 250))
    draw = ImageDraw.Draw(image)
    for x in range(0, CANVAS[0], CELL):
        draw.line((x, 0, x, CANVAS[1]), fill=(235, 237, 240), width=1)
    for y in range(0, CANVAS[1], CELL):
        draw.line((0, y, CANVAS[0], y), fill=(235, 237, 240), width=1)

    for region_id, observed_box in observed_boxes.items():
        pixel_box = grid_box(observed_box)
        dashed_rectangle(draw, pixel_box, (163, 168, 176), width=2, dash=7, gap=6)
        calibrated_box = entity_boxes[region_id]
        if calibrated_box != observed_box:
            draw.line(
                (center(observed_box), center(calibrated_box)),
                fill=(177, 181, 188),
                width=2,
            )

    for group in result.groups:
        box = grid_box(group_boxes[group.group_id])
        color = (112, 78, 164) if group.scope == "top_level" else (166, 91, 145)
        dashed_rectangle(draw, box, color)
        policy = group.calibration_policy
        group_text = f"{group.group_id}  {group.role}  [{policy.size_policy}, {policy.spacing}]"
        label(draw, (box[0] + 5, box[1] + 5), clipped(group_text, 68), color, font(13, True))

    role_colors = {
        "title": ((214, 226, 247), (36, 99, 186)),
        "standalone_text": ((251, 231, 208), (185, 99, 27)),
        "supporting_text": ((251, 231, 208), (185, 99, 27)),
        "caption": ((255, 238, 219), (193, 105, 33)),
        "label": ((255, 238, 219), (193, 105, 33)),
        "internal_label": ((255, 238, 219), (193, 105, 33)),
        "visual": ((215, 238, 220), (31, 132, 70)),
        "decorative": ((229, 231, 235), (104, 108, 116)),
    }
    entities = {item.region_id: item for item in result.entities}
    for entity in result.entities:
        box = grid_box(entity.calibrated_bbox)
        fill, outline = role_colors[entity.canonical_role]
        draw.rectangle(box, fill=fill, outline=outline, width=3)
        label(draw, (box[0] + 4, box[1] + 4), entity.region_id, outline, font(12))

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
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(24, 28, 36),
            outline="white",
            width=2,
        )
        number = str(index)
        number_font = font(12, True)
        number_box = draw.textbbox((0, 0), number, font=number_font)
        draw.text(
            (
                x - (number_box[2] - number_box[0]) / 2,
                y - (number_box[3] - number_box[1]) / 2 - 1,
            ),
            number,
            fill="white",
            font=number_font,
        )

    meta_box = (18, 14, 1518, 52)
    draw.rectangle(meta_box, fill=(255, 255, 255), outline=(207, 211, 217), width=1)
    draw.text((30, 24), f"{slide_dir.name}  calibrated", fill=(25, 29, 37), font=font(15, True))
    draw.text(
        (225, 24),
        "minimal topology and calibrated region geometry",
        fill=(65, 70, 80),
        font=font(14),
    )
    draw.rectangle((1190, 818, 1518, 850), fill=(255, 255, 255), outline=(207, 211, 217))
    dashed_rectangle(draw, (1203, 829, 1240, 841), (163, 168, 176), width=2, dash=5, gap=4)
    draw.text((1249, 826), "observed", fill=(90, 94, 102), font=font(12))
    draw.rectangle((1328, 829, 1365, 841), fill=(215, 238, 220), outline=(31, 132, 70), width=2)
    draw.text((1374, 826), "calibrated", fill=(54, 60, 70), font=font(12))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "slide_id": slide_dir.name,
        "output": str(output_path.resolve()),
        "entity_count": len(entities),
        "component_count": len(result.components),
        "group_count": len(result.groups),
        "changed_region_count": report["changed_region_count"],
        "overlap_count": len(report["overlaps"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render jointly calibrated topology planning slides.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    slide_dirs = sorted(
        path for path in input_dir.glob("slide_*") if (path / "calibrated_topology.json").exists()
    )
    if not slide_dirs:
        raise SystemExit(f"No calibrated slide directories found in {input_dir}")

    results = []
    output_paths = []
    for slide_dir in slide_dirs:
        output_path = output_dir / f"{slide_dir.name}.png"
        results.append(render_slide(slide_dir, output_path))
        output_paths.append(output_path)

    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(output_paths, contact_sheet)
    pages = [Image.open(path).convert("RGB") for path in output_paths]
    pdf_path = output_dir / "planning_deck_v2.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=144)
    for page in pages:
        page.close()

    manifest = {
        "stage": "jointly_calibrated_topology_planning_deck_render",
        "source": str(input_dir),
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "grid": {"width": GRID[0], "height": GRID[1]},
        "slide_count": len(results),
        "slides": results,
        "contact_sheet": str(contact_sheet),
        "pdf": str(pdf_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
