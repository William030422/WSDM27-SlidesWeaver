from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parent
V9_ROOT = ROOT.parents[1]
REFERENCE_DIR = ROOT / "slides"
V2_DIR = ROOT / "gpt_5_6_terra_layout_identity_v2"
DEFAULT_OUTPUT_DIR = ROOT / "gpt_5_6_terra_layout_identity_v3"
MODEL = "gpt-5.6-terra"
MAX_TOKENS = 8192
GRID = (64, 36)
NATIVE_CANVAS = (1024, 576)
OVERLAY_CANVAS = (1536, 864)

sys.path.insert(0, str(V9_ROOT / "src"))

from v9_deck_planning.apimart import (
    ApiMartClient,
    _extract_text_response,
    _extract_usage,
    image_to_data_uri,
)
from v9_deck_planning.json_utils import extract_json as extract_model_object
from v9_deck_planning.json_utils import read_json, write_json, write_text


BASE_PROMPT = """You are a presentation-slide topology parser.

You are given:

1. A low-step diffusion-generated ghost slide.
2. An overlay image showing already detected regions and their IDs.
3. A list of detected regions containing region ID, bounding box,
   predicted type, and confidence.

The region detections are immutable observations.
Do not create new regions, redetect bounding boxes, resize regions,
or output new coordinates.

The visible text may be pseudo-text. Do not perform OCR, transcribe,
guess, or infer its linguistic meaning. Treat text only as typographic geometry.

Your task is to recover the hierarchical visual topology of the slide.

Infer:

1. Which detected regions form a single component.
2. Which components are top-level slide components.
3. Which regions are internal children of a larger diagram or component.
4. Which components form nested row, column, split, grid, overlay,
   or freeform groups.
5. Which components are equivalent peers, sequential stages,
   comparative items, or hierarchical elements.
6. The dominant component, supporting components, and reading order.
7. Whether detected text functions geometrically as:
   title, standalone_text, supporting_text, caption, label, or internal_label.
8. Which geometric differences are likely intentional hierarchy and which
   are likely diffusion noise.

Important distinctions:

- Do not promote every detected label into a top-level component.
- When several labels or anchors share a connector, baseline, container,
  or continuous graphical structure, treat them as internal children of one
  composite diagram.
- A long text block below a visual is not automatically a caption.
  If it shares the visual's column boundaries and forms a substantial
  explanatory region, classify it as supporting_text paired with the visual.
- Multiple independent images may form repeated visual units, but repeated
  stages inside one connected diagram must remain inside one diagram component.
- Connectors are not standalone content elements, but they are evidence for
  sequence, grouping, and attachment.
- Detected roles may be corrected. Preserve the source region ID and return
  both detector_role and canonical_role when they differ.
- Do not generate final text boxes, final wording, typography, carrier choices,
  or precise PPT coordinates.

Return one JSON object with:

{
  "global_topology": {
    "composition_family": "column | split | grid | centered | sequence | mixed",
    "layout_formula": "concise recursive formula",
    "reading_direction": "string",
    "confidence": 0.0
  },
  "layout_tree": {
    "node_id": "root",
    "node_type": "group",
    "arrangement": "column | row | split | grid | overlay | freeform",
    "children": []
  },
  "groups": [
    {
      "group_id": "g1",
      "role": "string",
      "scope": "top_level | nested | internal",
      "arrangement": "column | row | split | grid | overlay | freeform",
      "children": [],
      "peer_relation": "none | equivalent | sequential | comparative | hierarchical",
      "reading_direction": "string",
      "shared_rules": [],
      "confidence": 0.0
    }
  ],
  "components": [
    {
      "component_id": "c1",
      "component_role": "title | primary_visual | supporting_visual | composite_diagram | standalone_text | supporting_text | annotation | decorative | mixed",
      "source_region_ids": [],
      "local_pattern": "string",
      "importance": "dominant | primary | secondary | minor",
      "confidence": 0.0
    }
  ],
  "relations": [
    {
      "source_id": "string",
      "target_id": "string",
      "relation": "paired_with | explains | supports | sequence_next | compares_with | attached_to | contains",
      "evidence": [],
      "confidence": 0.0
    }
  ],
  "region_disposition": [
    {
      "region_id": "string",
      "detector_role": "string",
      "canonical_role": "string",
      "status": "keep | internal_child | merge_into_parent | ignore_as_noise",
      "parent_id": "string or null"
    }
  ],
  "reading_order": [],
  "ambiguities": []
}
"""


Arrangement = Literal["column", "row", "split", "grid", "overlay", "freeform"]
GridBBox = tuple[int, int, int, int]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GlobalTopology(StrictModel):
    composition_family: Literal["column", "split", "grid", "centered", "sequence", "mixed"]
    layout_formula: str
    reading_direction: str
    confidence: float = Field(ge=0.0, le=1.0)


class LayoutTree(StrictModel):
    node_id: Literal["root"] = "root"
    node_type: Literal["group"] = "group"
    arrangement: Arrangement
    children: list[str | dict[str, Any]]


class TopologyGroup(StrictModel):
    group_id: str = Field(pattern=r"^g[a-z0-9_]*$")
    role: str
    scope: Literal["top_level", "nested", "internal"]
    arrangement: Arrangement
    children: list[str] = Field(min_length=1)
    peer_relation: Literal["none", "equivalent", "sequential", "comparative", "hierarchical"]
    reading_direction: str
    shared_rules: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class TopologyComponent(StrictModel):
    component_id: str = Field(pattern=r"^c[a-z0-9_]*$")
    component_role: Literal[
        "title",
        "primary_visual",
        "supporting_visual",
        "composite_diagram",
        "standalone_text",
        "supporting_text",
        "annotation",
        "decorative",
        "mixed",
    ]
    source_region_ids: list[str] = Field(min_length=1)
    local_pattern: str
    importance: Literal["dominant", "primary", "secondary", "minor"]
    confidence: float = Field(ge=0.0, le=1.0)


class TopologyRelation(StrictModel):
    source_id: str
    target_id: str
    relation: Literal[
        "paired_with",
        "explains",
        "supports",
        "sequence_next",
        "compares_with",
        "attached_to",
        "contains",
    ]
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class RegionDisposition(StrictModel):
    region_id: str
    detector_role: str
    canonical_role: str
    status: Literal["keep", "internal_child", "merge_into_parent", "ignore_as_noise"]
    parent_id: str | None


class TopologyResult(StrictModel):
    global_topology: GlobalTopology
    layout_tree: LayoutTree
    groups: list[TopologyGroup]
    components: list[TopologyComponent] = Field(min_length=1)
    relations: list[TopologyRelation]
    region_disposition: list[RegionDisposition]
    reading_order: list[str] = Field(min_length=1)
    ambiguities: list[str | dict[str, Any]]

    @model_validator(mode="after")
    def validate_ids_and_tree(self) -> "TopologyResult":
        group_ids = [item.group_id for item in self.groups]
        component_ids = [item.component_id for item in self.components]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("group IDs must be unique")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component IDs must be unique")
        hierarchy_nodes = set(group_ids) | set(component_ids)
        valid_relation_ids = hierarchy_nodes | {item.region_id for item in self.region_disposition}
        for relation in self.relations:
            if relation.source_id not in valid_relation_ids or relation.target_id not in valid_relation_ids:
                raise ValueError("relation references an unknown ID")
        if len(self.reading_order) != len(set(self.reading_order)):
            raise ValueError("reading_order IDs must be unique")
        if not set(self.reading_order).issubset(valid_relation_ids):
            raise ValueError("reading_order references an unknown ID")
        return self


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_bbox(boxes: list[GridBBox]) -> GridBBox:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def validate_no_output_coordinates(value: Any, path: str = "root") -> None:
    forbidden = {"bbox", "bounding_box", "coordinates", "x", "y", "width", "height"}
    if isinstance(value, dict):
        found = forbidden & set(value)
        if found:
            raise ValueError(f"topology output contains forbidden coordinate fields at {path}: {sorted(found)}")
        for key, child in value.items():
            validate_no_output_coordinates(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_no_output_coordinates(child, f"{path}.{index}")


def load_regions(slide_id: str) -> tuple[Path, list[dict]]:
    source_path = V2_DIR / slide_id / "layout_identity.json"
    source = read_json(source_path)
    regions = [
        {
            "region_id": item["element_id"],
            "bbox": item["bbox"],
            "predicted_type": _kind_for_role(item["role"]),
            "detector_role": item["role"],
            "confidence": item["confidence"],
        }
        for item in source["elements"]
    ]
    return source_path, regions


def _kind_for_role(role: str) -> str:
    if role == "title":
        return "title"
    if role == "visual":
        return "visual"
    return "text"


def validate_against_regions(result: TopologyResult, regions: list[dict]) -> None:
    region_ids = {item["region_id"] for item in regions}
    detector_roles = {item["region_id"]: item["detector_role"] for item in regions}
    dispositions = {item.region_id: item for item in result.region_disposition}
    if set(dispositions) != region_ids or len(dispositions) != len(result.region_disposition):
        raise ValueError("region_disposition must contain every immutable region exactly once")
    for region_id, item in dispositions.items():
        if item.detector_role != detector_roles[region_id]:
            raise ValueError(f"{region_id} detector_role does not preserve the immutable observation")

    covered = [region_id for component in result.components for region_id in component.source_region_ids]
    if len(covered) != len(set(covered)):
        raise ValueError("immutable regions cannot belong to multiple components")
    if not set(covered).issubset(region_ids):
        raise ValueError("component references an unknown immutable region")
    expected_covered = {
        region_id
        for region_id, item in dispositions.items()
        if item.status != "ignore_as_noise"
    }
    if set(covered) != expected_covered:
        raise ValueError("every non-noise region must belong to exactly one component")

    valid_parents = {item.group_id for item in result.groups} | {
        item.component_id for item in result.components
    }
    for item in dispositions.values():
        if item.status in {"internal_child", "merge_into_parent"} and item.parent_id is None:
            raise ValueError(f"{item.region_id} status {item.status} requires parent_id")
        if item.status == "ignore_as_noise" and item.parent_id is not None:
            raise ValueError(f"{item.region_id} ignored noise must have null parent_id")
        if item.parent_id is not None and item.parent_id not in valid_parents:
            raise ValueError(f"{item.region_id} references unknown parent_id")

    valid_children = valid_parents | region_ids
    group_children = [child for group in result.groups for child in group.children]
    unknown_group_children = set(group_children) - valid_children
    if unknown_group_children:
        raise ValueError(f"groups reference unknown children: {sorted(unknown_group_children)}")

    def tree_child_ids(children: list[str | dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for child in children:
            if isinstance(child, str):
                ids.append(child)
                continue
            node_id = child.get("node_id")
            if not isinstance(node_id, str):
                raise ValueError("nested layout_tree child requires a string node_id")
            ids.append(node_id)
            nested_children = child.get("children", [])
            if not isinstance(nested_children, list):
                raise ValueError(f"nested layout_tree node {node_id} children must be a list")
            ids.extend(tree_child_ids(nested_children))
        return ids

    unknown_tree_children = set(tree_child_ids(result.layout_tree.children)) - valid_children
    if unknown_tree_children:
        raise ValueError(f"layout_tree references unknown children: {sorted(unknown_tree_children)}")


def call_topology_vlm(
    client: ApiMartClient,
    *,
    reference_path: Path,
    region_overlay_path: Path,
    regions: list[dict],
) -> str:
    user_content = [
        {"type": "input_text", "text": BASE_PROMPT},
        {"type": "input_text", "text": "Input 1: low-step diffusion-generated ghost slide."},
        {"type": "input_image", "image_url": image_to_data_uri(reference_path)},
        {"type": "input_text", "text": "Input 2: immutable detected-region overlay with region IDs."},
        {"type": "input_image", "image_url": image_to_data_uri(region_overlay_path)},
        {"type": "input_text", "text": "Input 3: immutable detected-region list."},
        {"type": "input_text", "text": json.dumps(regions, indent=2)},
    ]
    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Return one valid JSON object with no markdown fences "
                            "and no prose outside JSON."
                        ),
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    client.call_count += 1
    try:
        response = client._post_json("/responses", payload)
    except RuntimeError as exc:
        if "convert_request_failed" not in str(exc) and "not implemented" not in str(exc):
            raise
        chat_content = []
        for item in user_content:
            if item["type"] == "input_text":
                chat_content.append({"type": "text", "text": item["text"]})
            else:
                chat_content.append(
                    {"type": "image_url", "image_url": {"url": item["image_url"]}}
                )
        response = client._post_json(
            "/chat/completions",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return one valid JSON object with no markdown fences "
                            "and no prose outside JSON."
                        ),
                    },
                    {"role": "user", "content": chat_content},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": MAX_TOKENS,
                "stream": False,
            },
        )
    client.last_usage = _extract_usage(response)
    text = _extract_text_response(response)
    if not text.strip():
        raise RuntimeError(f"APIMART topology call returned empty content: {response}")
    return text


def normalize_payload(payload: dict, regions: list[dict]) -> tuple[dict, dict]:
    normalized = copy.deepcopy(payload)
    changes: list[dict] = []

    def replace(container: dict, key: str, aliases: dict[str, str], path: str) -> None:
        value = container.get(key)
        replacement = aliases.get(value)
        if replacement is None:
            return
        container[key] = replacement
        changes.append({"json_path": path, "from": value, "to": replacement})

    topology = normalized.get("global_topology", {})
    replace(
        topology,
        "composition_family",
        {"row": "column", "freeform": "mixed", "overlay": "mixed"},
        "global_topology.composition_family",
    )
    replace(
        normalized.get("layout_tree", {}),
        "arrangement",
        {"centered": "freeform", "sequence": "row"},
        "layout_tree.arrangement",
    )
    for index, group in enumerate(normalized.get("groups", [])):
        replace(
            group,
            "arrangement",
            {"centered": "freeform", "sequence": "row"},
            f"groups.{index}.arrangement",
        )
        replace(
            group,
            "peer_relation",
            {"parallel": "equivalent", "comparison": "comparative", "sequence": "sequential"},
            f"groups.{index}.peer_relation",
        )
    for index, component in enumerate(normalized.get("components", [])):
        replace(
            component,
            "component_role",
            {
                "title_block": "title",
                "visual": "supporting_visual",
                "diagram": "composite_diagram",
                "text": "standalone_text",
            },
            f"components.{index}.component_role",
        )
    for index, relation in enumerate(normalized.get("relations", [])):
        replace(
            relation,
            "relation",
            {"paired": "paired_with", "next": "sequence_next", "compares": "compares_with"},
            f"relations.{index}.relation",
        )
    for index, disposition in enumerate(normalized.get("region_disposition", [])):
        replace(
            disposition,
            "status",
            {"internal": "internal_child", "merge": "merge_into_parent", "ignore": "ignore_as_noise"},
            f"region_disposition.{index}.status",
        )
    region_types = {item["region_id"]: item["predicted_type"] for item in regions}
    for index, disposition in enumerate(normalized.get("region_disposition", [])):
        region_id = disposition.get("region_id")
        canonical_role = disposition.get("canonical_role")
        if region_types.get(region_id) == "visual" and canonical_role != "visual":
            disposition["canonical_role"] = "visual"
            changes.append(
                {
                    "json_path": f"region_disposition.{index}.canonical_role",
                    "from": canonical_role,
                    "to": "visual",
                }
            )
    return normalized, {"schema_version": "3.0", "changed": bool(changes), "changes": changes}


def font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
    label_font: ImageFont.ImageFont,
) -> None:
    x, y = position
    box = draw.textbbox((x, y), text, font=label_font)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(255, 255, 255, 235))
    draw.text((x, y), text, fill=color, font=label_font)


def render_region_overlay(reference_path: Path, regions: list[dict], output_path: Path) -> None:
    with Image.open(reference_path) as reference:
        base = reference.convert("RGBA").resize(NATIVE_CANVAS)
    layer = Image.new("RGBA", NATIVE_CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    label_font = font(13)
    colors = {
        "title": (36, 99, 186, 230),
        "text": (221, 126, 43, 230),
        "visual": (32, 143, 76, 230),
    }
    fills = {
        "title": (36, 99, 186, 40),
        "text": (221, 126, 43, 40),
        "visual": (32, 143, 76, 40),
    }
    for region in regions:
        x, y, width, height = region["bbox"]
        box = (x * 16, y * 16, (x + width) * 16 - 1, (y + height) * 16 - 1)
        kind = region["predicted_type"]
        draw.rectangle(box, fill=fills[kind], outline=colors[kind], width=3)
        draw_label(draw, (box[0] + 3, box[1] + 3), region["region_id"], colors[kind], label_font)
    Image.alpha_composite(base, layer).convert("RGB").save(output_path)


def derive_topology_boxes(result: TopologyResult, regions: list[dict]) -> tuple[dict[str, GridBBox], dict[str, GridBBox]]:
    region_boxes = {item["region_id"]: tuple(item["bbox"]) for item in regions}
    component_boxes = {
        item.component_id: union_bbox([region_boxes[region_id] for region_id in item.source_region_ids])
        for item in result.components
    }
    group_map = {item.group_id: item for item in result.groups}
    group_boxes: dict[str, GridBBox] = {}
    visiting: set[str] = set()

    def group_box(group_id: str) -> GridBBox:
        if group_id in group_boxes:
            return group_boxes[group_id]
        if group_id in visiting:
            raise ValueError(f"group hierarchy contains a cycle at {group_id}")
        visiting.add(group_id)
        boxes = []
        for child in group_map[group_id].children:
            if child in group_map:
                boxes.append(group_box(child))
            elif child in component_boxes:
                boxes.append(component_boxes[child])
            else:
                boxes.append(region_boxes[child])
        group_boxes[group_id] = union_bbox(boxes)
        visiting.remove(group_id)
        return group_boxes[group_id]

    for group_id in group_map:
        group_box(group_id)
    return component_boxes, group_boxes


def grid_to_overlay_bbox(bbox: GridBBox) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (x * 24, y * 24, (x + width) * 24 - 1, (y + height) * 24 - 1)


def bbox_center(bbox: GridBBox) -> tuple[int, int]:
    x, y, width, height = bbox
    return ((2 * x + width) * 12, (2 * y + height) * 12)


def render_topology_overlay(
    reference_path: Path,
    regions: list[dict],
    result: TopologyResult,
    output_path: Path,
) -> None:
    with Image.open(reference_path) as reference:
        base = reference.convert("RGBA").resize(OVERLAY_CANVAS)
    layer = Image.new("RGBA", OVERLAY_CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    label_font = font(14)
    region_boxes = {item["region_id"]: tuple(item["bbox"]) for item in regions}
    component_boxes, group_boxes = derive_topology_boxes(result, regions)

    for region_id, bbox in region_boxes.items():
        box = grid_to_overlay_bbox(bbox)
        draw.rectangle(box, outline=(120, 120, 120, 150), width=2)
        draw_label(draw, (box[0] + 2, box[1] + 2), region_id, (80, 80, 80, 220), label_font)

    component_colors = {
        "title": (36, 99, 186, 220),
        "primary_visual": (28, 135, 74, 220),
        "supporting_visual": (75, 160, 100, 220),
        "composite_diagram": (16, 125, 145, 220),
        "standalone_text": (221, 126, 43, 220),
        "supporting_text": (229, 153, 73, 220),
        "annotation": (192, 84, 145, 220),
        "decorative": (120, 120, 120, 180),
        "mixed": (118, 82, 166, 220),
    }
    for component in result.components:
        box = grid_to_overlay_bbox(component_boxes[component.component_id])
        color = component_colors[component.component_role]
        draw.rectangle(box, outline=color, width=4)
        draw_label(
            draw,
            (box[0] + 3, box[1] + 20),
            f"{component.component_id} | {component.component_role}",
            color,
            label_font,
        )
    for group in result.groups:
        box = grid_to_overlay_bbox(group_boxes[group.group_id])
        color = (114, 75, 165, 235) if group.scope == "top_level" else (176, 83, 142, 220)
        draw.rectangle(box, outline=color, width=3)
        draw_label(draw, (box[0] + 3, box[1] + 38), f"{group.group_id} | {group.role}", color, label_font)

    all_boxes = {**region_boxes, **component_boxes, **group_boxes}
    for relation in result.relations:
        if relation.source_id in all_boxes and relation.target_id in all_boxes:
            draw.line(
                (bbox_center(all_boxes[relation.source_id]), bbox_center(all_boxes[relation.target_id])),
                fill=(205, 50, 135, 210),
                width=3,
            )
    Image.alpha_composite(base, layer).convert("RGB").save(output_path)


def extract_slide(slide_id: str, output_dir: Path, reuse_raw: bool) -> dict:
    slide_dir = output_dir / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    reference_path = REFERENCE_DIR / f"{slide_id}.png"
    v2_path, regions = load_regions(slide_id)
    regions_path = slide_dir / "regions.json"
    write_json(
        regions_path,
        {
            "schema_version": "1.0",
            "slide_id": slide_id,
            "coordinate_system": {"width": 64, "height": 36},
            "regions": regions,
        },
    )
    region_overlay_path = slide_dir / "immutable_region_overlay.png"
    render_region_overlay(reference_path, regions, region_overlay_path)
    prompt = BASE_PROMPT
    current_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_path = slide_dir / "topology_prompt.txt"
    write_text(prompt_path, prompt)
    input_manifest_path = slide_dir / "model_inputs.json"
    write_json(
        input_manifest_path,
        {
            "slide_id": slide_id,
            "inputs": [
                {
                    "input_number": 1,
                    "kind": "image",
                    "role": "ghost_slide",
                    "path": str(reference_path.resolve()),
                    "sha256": sha256(reference_path),
                },
                {
                    "input_number": 2,
                    "kind": "image",
                    "role": "immutable_region_overlay",
                    "path": str(region_overlay_path.resolve()),
                    "sha256": sha256(region_overlay_path),
                },
                {
                    "input_number": 3,
                    "kind": "json",
                    "role": "immutable_detected_regions",
                    "path": str(regions_path.resolve()),
                    "sha256": sha256(regions_path),
                },
            ],
        },
    )

    raw_path = slide_dir / "vlm_raw_response.json"
    previous_manifest_path = slide_dir / "run_manifest.json"
    previous_manifest = read_json(previous_manifest_path) if previous_manifest_path.exists() else {}
    client = ApiMartClient(env_path=V9_ROOT / ".env")
    started_at = now_iso()
    result = None
    topology_overlay_path = None
    normalization_path = None
    usage = None
    reused_raw = False
    try:
        if reuse_raw:
            if not raw_path.exists():
                raise FileNotFoundError(f"cannot reuse missing raw response: {raw_path}")
            raw = raw_path.read_text(encoding="utf-8")
            usage = previous_manifest.get("usage")
            reused_raw = True
        else:
            raw = call_topology_vlm(
                client,
                reference_path=reference_path,
                region_overlay_path=region_overlay_path,
                regions=regions,
            )
            write_text(raw_path, raw)
            usage = client.last_usage
        payload = extract_model_object(raw)
        validate_no_output_coordinates(payload)
        normalized, normalization_report = normalize_payload(payload, regions)
        normalization_path = slide_dir / "normalization_report.json"
        write_json(normalization_path, normalization_report)
        result = TopologyResult.model_validate(normalized)
        validate_against_regions(result, regions)
        write_json(slide_dir / "topology.json", result)
        topology_overlay_path = slide_dir / "topology_overlay.png"
        render_topology_overlay(reference_path, regions, result, topology_overlay_path)
        status = "succeeded"
        error = None
    except Exception as exc:
        status = "failed"
        error = str(exc)

    previous_raw_prompt_sha256 = (
        previous_manifest.get("inputs", {}).get("raw_response_prompt_sha256")
        or previous_manifest.get("inputs", {}).get("prompt_sha256")
        or current_prompt_sha256
    )
    manifest = {
        "slide_id": slide_id,
        "status": status,
        "error": error,
        "model": MODEL,
        "model_call_count": client.call_count,
        "raw_response_model_call_count": (
            previous_manifest.get("raw_response_model_call_count")
            or (1 if raw_path.exists() and usage else client.call_count)
        ),
        "reused_raw_response": reused_raw,
        "usage": usage,
        "started_at": started_at,
        "finished_at": now_iso(),
        "inputs": {
            "reference": str(reference_path.resolve()),
            "reference_sha256": sha256(reference_path),
            "v2_layout_identity": str(v2_path.resolve()),
            "v2_layout_identity_sha256": sha256(v2_path),
            "regions": str(regions_path.resolve()),
            "regions_sha256": sha256(regions_path),
            "input_manifest": str(input_manifest_path.resolve()),
            "input_manifest_sha256": sha256(input_manifest_path),
            "raw_response_prompt_sha256": previous_raw_prompt_sha256 if reused_raw else current_prompt_sha256,
            "validation_prompt_sha256": current_prompt_sha256,
            "prompt_changed_since_raw": reused_raw and previous_raw_prompt_sha256 != current_prompt_sha256,
        },
        "composition_family": result.global_topology.composition_family if result else None,
        "layout_formula": result.global_topology.layout_formula if result else None,
        "group_count": len(result.groups) if result else 0,
        "component_count": len(result.components) if result else 0,
        "relation_count": len(result.relations) if result else 0,
        "artifacts": {
            "regions": str(regions_path.resolve()),
            "region_overlay": str(region_overlay_path.resolve()),
            "input_manifest": str(input_manifest_path.resolve()),
            "prompt": str(prompt_path.resolve()),
            "raw_response": str(raw_path.resolve()) if raw_path.exists() else None,
            "normalization_report": str(normalization_path.resolve()) if normalization_path else None,
            "topology": str((slide_dir / "topology.json").resolve())
            if (slide_dir / "topology.json").exists()
            else None,
            "topology_overlay": str(topology_overlay_path.resolve()) if topology_overlay_path else None,
            "manifest": str((slide_dir / "run_manifest.json").resolve()),
        },
    }
    write_json(slide_dir / "run_manifest.json", manifest)
    return manifest


def make_contact_sheet(slides: list[dict], output_dir: Path) -> Path:
    thumb_width = 512
    thumb_height = 288
    label_height = 32
    columns = min(2, len(slides))
    rows = math.ceil(len(slides) / columns)
    canvas = Image.new(
        "RGB", (thumb_width * columns, (thumb_height + label_height) * rows), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, slide in enumerate(slides):
        overlay_path = slide["artifacts"].get("topology_overlay")
        if overlay_path:
            image = Image.open(overlay_path).convert("RGB").resize((thumb_width, thumb_height))
        else:
            image = Image.new("RGB", (thumb_width, thumb_height), (245, 230, 230))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 12, y + 10), f'{slide["slide_id"]}: {slide["status"]}', fill="black")
    output_path = output_dir / "topology_contact_sheet.png"
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse slide topology from immutable V2 regions.")
    parser.add_argument(
        "--slide-id",
        action="append",
        choices=[f"slide_{index:03d}" for index in range(1, 7)],
        help="Parse only this slide. Repeat to select multiple slides.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--reuse-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slide_ids = args.slide_id or [f"slide_{index:03d}" for index in range(1, 7)]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    workers = min(max(args.max_workers, 1), len(slide_ids))
    if workers == 1:
        slides = [extract_slide(slide_id, output_dir, args.reuse_raw) for slide_id in slide_ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            slides = list(
                executor.map(
                    lambda item: extract_slide(item, output_dir, args.reuse_raw),
                    slide_ids,
                )
            )
    contact_sheet = make_contact_sheet(slides, output_dir)
    failed = [slide for slide in slides if slide["status"] == "failed"]
    manifest = {
        "stage": "immutable_region_slide_topology_parsing_v3",
        "status": "failed" if failed else "succeeded",
        "error": f"{len(failed)} topology call(s) failed" if failed else None,
        "model": MODEL,
        "schema_version": "3.0",
        "slide_count": len(slides),
        "model_call_count": sum(slide["model_call_count"] for slide in slides),
        "raw_response_model_call_count": sum(
            slide["raw_response_model_call_count"] for slide in slides
        ),
        "reused_raw_count": sum(slide["reused_raw_response"] for slide in slides),
        "total_tokens": sum((slide.get("usage") or {}).get("total_tokens", 0) or 0 for slide in slides),
        "parallel_execution": workers > 1,
        "max_workers": workers,
        "started_at": started_at,
        "finished_at": now_iso(),
        "contact_sheet": str(contact_sheet.resolve()),
        "slides": slides,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
