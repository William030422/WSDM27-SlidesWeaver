from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, JpegImagePlugin

from extract_calibrated_topology_with_vlm import (
    DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_DIR,
    CalibratedTopologyResult,
)
from extract_layout_topology_with_vlm import ROOT
from render_topology_planning_deck import (
    CANVAS,
    CELL,
    clipped,
    dashed_rectangle,
    font,
    label,
    make_contact_sheet,
)


DEFAULT_OUTPUT_DIR = ROOT / "gpt_5_6_terra_layout_identity_v3" / "planning_deck_v2_postprocessed"
OUTER_MARGIN_RATIO = 0.03
MAX_HEADER_CONTENT_GAP_RATIO = 0.08
PixelBBox = tuple[int, int, int, int]


def union_bbox(boxes: list[PixelBBox]) -> PixelBBox:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def draw_box(bbox: PixelBBox) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (x, y, x + width - 1, y + height - 1)


def center(bbox: PixelBBox) -> tuple[int, int]:
    x, y, width, height = bbox
    return (x + width // 2, y + height // 2)


def grid_to_pixels(bbox: tuple[int, int, int, int]) -> PixelBBox:
    x, y, width, height = bbox
    return (x * CELL, y * CELL, width * CELL, height * CELL)


def descendant_component_ids(
    node_id: str,
    group_children: dict[str, list[str]],
    component_ids: set[str],
) -> set[str]:
    if node_id in component_ids:
        return {node_id}
    descendants: set[str] = set()
    for child in group_children[node_id]:
        descendants.update(descendant_component_ids(child, group_children, component_ids))
    return descendants


def identify_header_and_main_regions(
    result: CalibratedTopologyResult,
) -> tuple[set[str], set[str], str]:
    components = {item.component_id: item for item in result.components}
    component_ids = set(components)
    group_children = {item.group_id: item.children for item in result.groups}
    title_components = {
        item.component_id for item in result.components if item.component_role == "title"
    }
    candidates: list[tuple[int, str, set[str]]] = []
    for group in result.groups:
        descendants = descendant_component_ids(group.group_id, group_children, component_ids)
        role = group.role.lower().replace("_", " ")
        if title_components & descendants and ("header" in role or "title" in role):
            region_ids = {
                region_id
                for component_id in descendants
                for region_id in components[component_id].source_region_ids
            }
            candidates.append((len(region_ids), group.group_id, region_ids))

    if candidates:
        _, header_node_id, header_region_ids = min(candidates, key=lambda item: item[0])
    else:
        header_node_id = "+".join(sorted(title_components))
        header_region_ids = {
            region_id
            for component_id in title_components
            for region_id in components[component_id].source_region_ids
        }
    all_region_ids = {item.region_id for item in result.entities}
    main_region_ids = all_region_ids - header_region_ids
    if not header_region_ids or not main_region_ids:
        raise ValueError(f"{result.slide_id} requires non-empty header and main blocks")
    return header_region_ids, main_region_ids, header_node_id


def map_boxes_to_target(
    boxes: dict[str, PixelBBox],
    target: PixelBBox,
) -> dict[str, PixelBBox]:
    source = union_bbox(list(boxes.values()))
    scale_x = target[2] / source[2]
    scale_y = target[3] / source[3]
    mapped: dict[str, PixelBBox] = {}
    for region_id, box in boxes.items():
        left = round(target[0] + (box[0] - source[0]) * scale_x)
        top = round(target[1] + (box[1] - source[1]) * scale_y)
        right = round(target[0] + (box[0] + box[2] - source[0]) * scale_x)
        bottom = round(target[1] + (box[1] + box[3] - source[1]) * scale_y)
        mapped[region_id] = (left, top, max(1, right - left), max(1, bottom - top))
    return mapped


def compile_group_policies(
    result: CalibratedTopologyResult,
    boxes: dict[str, PixelBBox],
) -> dict[str, PixelBBox]:
    """Apply calibrated peer policies before the global margin pass."""
    output = dict(boxes)
    components = {item.component_id: item for item in result.components}
    groups = {item.group_id: item for item in result.groups}
    entities = {item.region_id: item for item in result.entities}

    def region_ids(node_id: str) -> list[str]:
        if node_id in components:
            return list(components[node_id].source_region_ids)
        regions: list[str] = []
        for child in groups[node_id].children:
            regions.extend(region_ids(child))
        return regions

    def node_bbox(node_id: str) -> PixelBBox:
        return union_bbox([output[region_id] for region_id in region_ids(node_id)])

    def move_node(node_id: str, target: PixelBBox) -> None:
        source = node_bbox(node_id)
        for region_id in region_ids(node_id):
            output[region_id] = _map_box(output[region_id], source, target)

    visited: set[str] = set()

    def compile_group(group_id: str) -> None:
        if group_id in visited:
            return
        group = groups[group_id]
        for child in group.children:
            if child in groups:
                compile_group(child)
        visited.add(group_id)
        if group.arrangement not in {"row", "column"} or len(group.children) < 2:
            return
        policy = group.calibration_policy
        if (
            policy.size_policy in {"preserve", "weighted"}
            and policy.alignment == "preserve"
            and policy.spacing == "preserve"
        ):
            return
        current = [node_bbox(child) for child in group.children]
        targets = _peer_targets(
            current,
            arrangement=group.arrangement,
            size_policy=policy.size_policy,
            alignment=policy.alignment,
            spacing=policy.spacing,
        )
        for child, target in zip(group.children, targets):
            move_node(child, target)
        if group.arrangement == "row" and policy.size_policy in {"equal_size", "equal_height"}:
            caption_ids = []
            for child in group.children:
                if child not in components:
                    caption_ids = []
                    break
                captions = [
                    region_id
                    for region_id in components[child].source_region_ids
                    if entities[region_id].canonical_role == "caption"
                ]
                if len(captions) != 1:
                    caption_ids = []
                    break
                caption_ids.append(captions[0])
            if caption_ids:
                caption_height = max(output[region_id][3] for region_id in caption_ids)
                for child, region_id in zip(group.children, caption_ids):
                    component_box = node_bbox(child)
                    height = min(caption_height, component_box[3] - 1)
                    output[region_id] = (
                        component_box[0],
                        component_box[1] + component_box[3] - height,
                        component_box[2],
                        height,
                    )

    child_groups = {
        child
        for group in result.groups
        for child in group.children
        if child in groups
    }
    roots = [group_id for group_id in groups if group_id not in child_groups]
    for group_id in roots:
        compile_group(group_id)
    for group_id in groups:
        compile_group(group_id)
    return output


def _map_box(box: PixelBBox, source: PixelBBox, target: PixelBBox) -> PixelBBox:
    scale_x = target[2] / source[2]
    scale_y = target[3] / source[3]
    left = round(target[0] + (box[0] - source[0]) * scale_x)
    top = round(target[1] + (box[1] - source[1]) * scale_y)
    right = round(target[0] + (box[0] + box[2] - source[0]) * scale_x)
    bottom = round(target[1] + (box[1] + box[3] - source[1]) * scale_y)
    return (left, top, max(1, right - left), max(1, bottom - top))


def _peer_targets(
    boxes: list[PixelBBox],
    *,
    arrangement: str,
    size_policy: str,
    alignment: str,
    spacing: str,
) -> list[PixelBBox]:
    outer = union_bbox(boxes)
    horizontal = arrangement == "row"
    primary_sizes = [box[2] if horizontal else box[3] for box in boxes]
    cross_sizes = [box[3] if horizontal else box[2] for box in boxes]
    equal_primary = (
        size_policy in {"equal_width", "equal_size"}
        if horizontal
        else size_policy in {"equal_height", "equal_size"}
    )
    equal_cross = (
        size_policy in {"equal_height", "equal_size"}
        if horizontal
        else size_policy in {"equal_width", "equal_size"}
    )
    outer_primary = outer[2] if horizontal else outer[3]
    starts = [box[0] if horizontal else box[1] for box in boxes]
    ends = [start + size for start, size in zip(starts, primary_sizes)]
    positive_gaps = [
        starts[index + 1] - ends[index]
        for index in range(len(boxes) - 1)
        if starts[index + 1] - ends[index] > 0
    ]

    target_primary_sizes = list(primary_sizes)
    if equal_primary:
        gap = (
            round(statistics.median(positive_gaps))
            if spacing == "even" and positive_gaps
            else 0
        )
        available = outer_primary - gap * (len(boxes) - 1)
        if available < len(boxes):
            gap = 0
            available = outer_primary
        target_primary_sizes = [max(1, available // len(boxes))] * len(boxes)
    target_cross_sizes = list(cross_sizes)
    if equal_cross:
        target_cross_sizes = [max(cross_sizes)] * len(boxes)

    if spacing == "even":
        free = outer_primary - sum(target_primary_sizes)
        gap = max(0, round(free / (len(boxes) - 1)))
        primary_positions = [outer[0] if horizontal else outer[1]]
        for size in target_primary_sizes[:-1]:
            primary_positions.append(primary_positions[-1] + size + gap)
    else:
        primary_positions = starts

    targets: list[PixelBBox] = []
    for index, box in enumerate(boxes):
        cross_size = target_cross_sizes[index]
        if alignment == "center":
            cross_position = (
                outer[1] + (outer[3] - cross_size) // 2
                if horizontal
                else outer[0] + (outer[2] - cross_size) // 2
            )
        elif alignment in {"bottom", "baseline"}:
            cross_position = (
                outer[1] + outer[3] - cross_size
                if horizontal
                else outer[0] + outer[2] - cross_size
            )
        elif alignment in {"top", "grid"}:
            cross_position = outer[1] if horizontal else outer[0]
        else:
            cross_position = box[1] if horizontal else box[0]
        if horizontal:
            targets.append(
                (
                    primary_positions[index],
                    cross_position,
                    target_primary_sizes[index],
                    cross_size,
                )
            )
        else:
            targets.append(
                (
                    cross_position,
                    primary_positions[index],
                    cross_size,
                    target_primary_sizes[index],
                )
            )
    return targets


def cap_header_content_gap(
    boxes: dict[str, PixelBBox],
    header_region_ids: set[str],
    main_region_ids: set[str],
    max_gap: int,
) -> tuple[dict[str, PixelBBox], bool, int, int]:
    header_box = union_bbox([boxes[region_id] for region_id in header_region_ids])
    main_box = union_bbox([boxes[region_id] for region_id in main_region_ids])
    gap_before = main_box[1] - (header_box[1] + header_box[3])
    if gap_before <= max_gap:
        return boxes, False, gap_before, gap_before

    target_main_top = header_box[1] + header_box[3] + max_gap
    main_bottom = main_box[1] + main_box[3]
    scale_y = (main_bottom - target_main_top) / main_box[3]
    adjusted = dict(boxes)
    for region_id in main_region_ids:
        box = boxes[region_id]
        top = round(target_main_top + (box[1] - main_box[1]) * scale_y)
        bottom = round(target_main_top + (box[1] + box[3] - main_box[1]) * scale_y)
        adjusted[region_id] = (box[0], top, box[2], max(1, bottom - top))
    final_main_box = union_bbox([adjusted[region_id] for region_id in main_region_ids])
    gap_after = final_main_box[1] - (header_box[1] + header_box[3])
    return adjusted, True, gap_before, gap_after


def postprocess_layout(result: CalibratedTopologyResult) -> dict:
    observed_boxes = {
        entity.region_id: grid_to_pixels(entity.calibrated_bbox) for entity in result.entities
    }
    input_boxes = compile_group_policies(result, observed_boxes)
    header_region_ids, main_region_ids, header_node_id = identify_header_and_main_regions(result)
    margin_x = round(CANVAS[0] * OUTER_MARGIN_RATIO)
    margin_y = round(CANVAS[1] * OUTER_MARGIN_RATIO)
    target_outer = (
        margin_x,
        margin_y,
        CANVAS[0] - 2 * margin_x,
        CANVAS[1] - 2 * margin_y,
    )
    globally_resized = map_boxes_to_target(input_boxes, target_outer)
    max_gap = round(CANVAS[1] * MAX_HEADER_CONTENT_GAP_RATIO)
    output_boxes, gap_adjusted, gap_before, gap_after = cap_header_content_gap(
        globally_resized,
        header_region_ids,
        main_region_ids,
        max_gap,
    )
    outer_after = union_bbox(list(output_boxes.values()))
    if outer_after != target_outer:
        raise ValueError(
            f"{result.slide_id} outer bbox is {outer_after}, expected exact target {target_outer}"
        )
    if gap_after > max_gap:
        raise ValueError(f"{result.slide_id} final header-content gap exceeds {max_gap}px")

    return {
        "schema_version": "1.0",
        "slide_id": result.slide_id,
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "rules": {
            "max_header_content_gap_ratio": MAX_HEADER_CONTENT_GAP_RATIO,
            "outer_margin_ratio": OUTER_MARGIN_RATIO,
            "gap_behavior": "reduce_only_when_above_limit",
        },
        "header_node_id": header_node_id,
        "header_region_ids": sorted(header_region_ids),
        "main_region_ids": sorted(main_region_ids),
        "measurements": {
            "outer_bbox_before_px": union_bbox(list(observed_boxes.values())),
            "group_policy_bbox_px": union_bbox(list(input_boxes.values())),
            "outer_bbox_after_px": outer_after,
            "target_outer_bbox_px": target_outer,
            "max_header_content_gap_px": max_gap,
            "header_content_gap_after_outer_resize_px": gap_before,
            "header_content_gap_final_px": gap_after,
            "gap_adjusted": gap_adjusted,
        },
        "entities": [
            {
                "region_id": region_id,
                "input_calibrated_bbox_px": observed_boxes[region_id],
                "policy_compiled_bbox_px": input_boxes[region_id],
                "postprocessed_bbox_px": output_boxes[region_id],
            }
            for region_id in sorted(output_boxes)
        ],
    }


def derive_boxes(
    result: CalibratedTopologyResult,
    entity_boxes: dict[str, PixelBBox],
) -> tuple[dict[str, PixelBBox], dict[str, PixelBBox]]:
    component_boxes = {
        component.component_id: union_bbox(
            [entity_boxes[region_id] for region_id in component.source_region_ids]
        )
        for component in result.components
    }
    group_map = {group.group_id: group for group in result.groups}
    group_boxes: dict[str, PixelBBox] = {}
    visiting: set[str] = set()

    def group_box(group_id: str) -> PixelBBox:
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
    return component_boxes, group_boxes


def render_slide(slide_dir: Path, output_dir: Path) -> dict:
    result = CalibratedTopologyResult.model_validate_json(
        (slide_dir / "calibrated_topology.json").read_text(encoding="utf-8")
    )
    layout = postprocess_layout(result)
    input_boxes = {
        item["region_id"]: tuple(item["input_calibrated_bbox_px"])
        for item in layout["entities"]
    }
    entity_boxes = {
        item["region_id"]: tuple(item["postprocessed_bbox_px"])
        for item in layout["entities"]
    }
    component_boxes, group_boxes = derive_boxes(result, entity_boxes)
    layout["components"] = [
        {
            "component_id": component.component_id,
            "component_role": component.component_role,
            "source_region_ids": component.source_region_ids,
            "bbox_px": component_boxes[component.component_id],
        }
        for component in result.components
    ]
    layout["groups"] = [
        {
            "group_id": group.group_id,
            "role": group.role,
            "arrangement": group.arrangement,
            "children": group.children,
            "peer_relation": group.peer_relation,
            "calibration_policy": group.calibration_policy.model_dump(mode="json"),
            "bbox_px": group_boxes[group.group_id],
        }
        for group in result.groups
    ]
    layout_path = output_dir / "layouts" / f"{result.slide_id}.json"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    all_boxes = {**entity_boxes, **component_boxes, **group_boxes}

    image = Image.new("RGB", CANVAS, (247, 248, 250))
    draw = ImageDraw.Draw(image)
    for x in range(0, CANVAS[0], CELL):
        draw.line((x, 0, x, CANVAS[1]), fill=(235, 237, 240), width=1)
    for y in range(0, CANVAS[1], CELL):
        draw.line((0, y, CANVAS[0], y), fill=(235, 237, 240), width=1)

    for region_id, input_box in input_boxes.items():
        dashed_rectangle(draw, draw_box(input_box), (163, 168, 176), width=2, dash=7, gap=6)
        if input_box != entity_boxes[region_id]:
            draw.line(
                (center(input_box), center(entity_boxes[region_id])),
                fill=(177, 181, 188),
                width=2,
            )

    for group in result.groups:
        box = draw_box(group_boxes[group.group_id])
        color = (112, 78, 164) if group.scope == "top_level" else (166, 91, 145)
        dashed_rectangle(draw, box, color)
        label(
            draw,
            (box[0] + 5, box[1] + 5),
            clipped(f"{group.group_id}  {group.role}", 58),
            color,
            font(13, True),
        )

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
    for entity in result.entities:
        box = draw_box(entity_boxes[entity.region_id])
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
        box = draw_box(component_boxes[component.component_id])
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

    measurements = layout["measurements"]
    meta_box = (18, 4, 1518, 24)
    draw.rectangle(meta_box, fill=(255, 255, 255), outline=(207, 211, 217), width=1)
    draw.text((28, 7), f"{result.slide_id}  postprocessed", fill=(25, 29, 37), font=font(12, True))
    draw.text(
        (225, 7),
        (
            f"outer margin=3% | header-content gap="
            f"{measurements['header_content_gap_final_px']}px / "
            f"{measurements['max_header_content_gap_px']}px max"
        ),
        fill=(65, 70, 80),
        font=font(11),
    )
    draw.rectangle((1145, 832, 1518, 858), fill=(255, 255, 255), outline=(207, 211, 217))
    dashed_rectangle(draw, (1158, 840, 1195, 852), (163, 168, 176), width=2, dash=5, gap=4)
    draw.text((1204, 838), "pre-rule", fill=(90, 94, 102), font=font(11))
    draw.rectangle((1300, 840, 1337, 852), fill=(215, 238, 220), outline=(31, 132, 70), width=2)
    draw.text((1346, 838), "postprocessed", fill=(54, 60, 70), font=font(11))

    output_path = output_dir / f"{result.slide_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "slide_id": result.slide_id,
        "output": str(output_path.resolve()),
        "layout": str(layout_path.resolve()),
        "header_node_id": layout["header_node_id"],
        **measurements,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply deterministic deck margin postprocessing.")
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

    results = [render_slide(slide_dir, output_dir) for slide_dir in slide_dirs]
    output_paths = [Path(item["output"]) for item in results]
    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(output_paths, contact_sheet)
    pages = [Image.open(path).convert("RGB") for path in output_paths]
    pdf_path = output_dir / "planning_deck_v2_postprocessed.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=144)
    for page in pages:
        page.close()

    manifest = {
        "stage": "deterministic_planning_margin_postprocess",
        "source": str(input_dir),
        "llm_call_count": 0,
        "rules": {
            "max_header_content_gap_ratio": MAX_HEADER_CONTENT_GAP_RATIO,
            "outer_margin_ratio": OUTER_MARGIN_RATIO,
        },
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "slide_count": len(results),
        "slides": results,
        "contact_sheet": str(contact_sheet.resolve()),
        "pdf": str(pdf_path.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
