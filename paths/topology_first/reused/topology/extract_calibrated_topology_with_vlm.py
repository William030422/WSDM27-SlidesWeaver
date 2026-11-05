from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from extract_layout_topology_with_vlm import (
    GRID,
    MAX_TOKENS,
    MODEL,
    REFERENCE_DIR,
    ROOT,
    V2_DIR,
    V9_ROOT,
    load_regions,
    now_iso,
    render_region_overlay,
    sha256,
    union_bbox,
)
from v9_deck_planning.apimart import (
    ApiMartClient,
    _extract_text_response,
    _extract_usage,
    image_to_data_uri,
)
from v9_deck_planning.json_utils import extract_json as extract_model_object
from v9_deck_planning.json_utils import read_json, write_json, write_text


DEFAULT_OUTPUT_DIR = ROOT / "gpt_5_6_terra_layout_identity_v3" / "calibrated_topology_v2"
GridBBox = tuple[int, int, int, int]
Arrangement = Literal["column", "row", "split", "grid", "overlay", "freeform"]


CALIBRATION_PROMPT = """You are a presentation-slide topology and geometric calibration parser.

You are given three separate inputs:

1. A low-step diffusion-generated ghost slide.
2. An overlay image showing detected regions and their immutable IDs.
3. A JSON object containing detected regions with immutable IDs and observed bounding boxes.

The visible text may be pseudo-text. Do not perform OCR, transcribe, guess, or infer
its linguistic meaning. Treat text only as typographic geometry.

Perform topology recovery and geometric calibration jointly in this single response.

Identity constraints:

- Region IDs and region count are immutable. Do not create, delete, merge, or split regions.
- Each input region must appear exactly once in entities and exactly once in one component.
- canonical_role may correct a text region's geometric role. Visual and title detector roles are immutable.
- Connectors are topology evidence, not entities.

Calibration task:

- Treat observed bounding boxes as noisy starting observations, not fixed final geometry.
- Return one calibrated 64 by 36 integer-grid bounding box for every region.
- Preserve the broad composition, margins, hierarchy, and reading direction.
- Correct likely diffusion noise by aligning edges and baselines, regularizing repeated sizes,
  making repeated gaps even, and keeping paired text attached to its visual.
- Repeated equivalent, comparative, or sequential stage units should normally use equal
  container sizes, equal visual-anchor sizes, aligned captions, and even spacing.
- For calibration, create one component per calibratable peer unit. A visual and its directly
  attached caption or label should be one component. Do not merge all repeated stages into
  one component merely because connectors make them one composite diagram; put the stage
  components inside one sequential group instead.
- In a repeated sequential row, express sequence through child order, not accidental
  size drift. Default to equal-size stage visuals. Preserve unequal sizes only when at least
  two independent non-size cues establish explicit hierarchy. Size difference alone is
  diffusion noise. Record any preserved exception in the group's calibration rationale.
- Keep title and supporting text as separate components when they occupy separate bands.
- Calibrated boxes must stay within the canvas, have positive size, and avoid unintended overlap.
- Do not generate pixel coordinates, final wording, typography, visual content, confidence,
  explanations, summaries, relations, reading order, parent IDs, or fields owned by the orchestrator.

Return only one JSON object matching this minimal shape and enum vocabulary:

{
  "groups": [
    {
      "group_id": "g1",
      "role": "string",
      "arrangement": "column | row | split | grid | overlay | freeform",
      "children": ["group_or_component_id"],
      "peer_relation": "none | equivalent | sequential | comparative | hierarchical",
      "calibration_policy": {
        "size_policy": "preserve | equal_width | equal_height | equal_size | weighted",
        "alignment": "preserve | top | center | bottom | baseline | grid",
        "spacing": "preserve | even"
      }
    }
  ],
  "components": [
    {
      "component_id": "c1",
      "component_role": "title | primary_visual | supporting_visual | composite_diagram | standalone_text | supporting_text | annotation | decorative | mixed",
      "source_region_ids": ["region_id"]
    }
  ],
  "entities": [
    {
      "region_id": "source region ID",
      "canonical_role": "title | standalone_text | supporting_text | caption | label | internal_label | visual | decorative",
      "calibrated_bbox": [0, 0, 1, 1]
    }
  ]
}
"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationPolicy(StrictModel):
    model_config = ConfigDict(extra="ignore")

    size_policy: Literal["preserve", "equal_width", "equal_height", "equal_size", "weighted"]
    alignment: Literal["preserve", "top", "center", "bottom", "baseline", "grid"]
    spacing: Literal["preserve", "even"]


class CalibratedGroup(StrictModel):
    group_id: str = Field(pattern=r"^g[a-z0-9_]*$")
    role: str
    scope: Literal["top_level", "nested", "internal"]
    arrangement: Arrangement
    children: list[str] = Field(min_length=1)
    peer_relation: Literal["none", "equivalent", "sequential", "comparative", "hierarchical"]
    calibration_policy: CalibrationPolicy


class CalibratedComponent(StrictModel):
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


class CalibratedEntity(StrictModel):
    region_id: str
    canonical_role: Literal[
        "title",
        "standalone_text",
        "supporting_text",
        "caption",
        "label",
        "internal_label",
        "visual",
        "decorative",
    ]
    parent_component_id: str
    calibrated_bbox: GridBBox


class CalibrationGroupResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    group_id: str = Field(pattern=r"^g[a-z0-9_]*$")
    role: str
    arrangement: Arrangement
    children: list[str] = Field(min_length=1)
    peer_relation: Literal["none", "equivalent", "sequential", "comparative", "hierarchical"]
    calibration_policy: CalibrationPolicy


class CalibrationComponentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


class CalibrationEntityResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    region_id: str
    canonical_role: Literal[
        "title",
        "standalone_text",
        "supporting_text",
        "caption",
        "label",
        "internal_label",
        "visual",
        "decorative",
    ]
    calibrated_bbox: GridBBox


class CalibrationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    groups: list[CalibrationGroupResponse]
    components: list[CalibrationComponentResponse] = Field(min_length=1)
    entities: list[CalibrationEntityResponse] = Field(min_length=1)


class CalibratedTopologyResult(StrictModel):
    slide_id: str = Field(pattern=r"^slide_[0-9]{3}$")
    groups: list[CalibratedGroup]
    components: list[CalibratedComponent] = Field(min_length=1)
    entities: list[CalibratedEntity] = Field(min_length=1)
    reading_order: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "CalibratedTopologyResult":
        for label, values in (
            ("group", [item.group_id for item in self.groups]),
            ("component", [item.component_id for item in self.components]),
            ("entity", [item.region_id for item in self.entities]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be unique")
        return self


def validate_bbox(bbox: GridBBox, label: str) -> None:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has non-positive size")
    if x < 0 or y < 0 or x + width > GRID[0] or y + height > GRID[1]:
        raise ValueError(f"{label} is outside the 64x36 canvas")


def normalize_payload(
    payload: dict,
    regions: list[dict],
    slide_id: str | None = None,
) -> tuple[dict, dict]:
    normalized = copy.deepcopy(payload)
    changes: list[dict] = []
    fallbacks: list[dict] = []

    def replace(container: dict, key: str, aliases: dict[str, str], path: str) -> None:
        value = container.get(key)
        if value not in aliases:
            return
        container[key] = aliases[value]
        changes.append({"json_path": path, "from": value, "to": aliases[value]})

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
        policy = group.get("calibration_policy", {})
        replace(policy, "size_policy", {"uniform": "equal_size"}, f"groups.{index}.calibration_policy.size_policy")
        replace(policy, "alignment", {"middle": "center"}, f"groups.{index}.calibration_policy.alignment")
        replace(policy, "spacing", {"equal": "even"}, f"groups.{index}.calibration_policy.spacing")
        for key, default in (
            ("size_policy", "preserve"),
            ("alignment", "preserve"),
            ("spacing", "preserve"),
        ):
            if key not in policy:
                policy[key] = default
                fallbacks.append(
                    {"json_path": f"groups.{index}.calibration_policy.{key}", "to": default}
                )
    region_types = {item["region_id"]: item["predicted_type"] for item in regions}
    detector_roles = {item["region_id"]: item["detector_role"] for item in regions}
    for index, entity in enumerate(normalized.get("entities", [])):
        region_id = entity.get("region_id")
        if region_types.get(region_id) == "visual" and entity.get("canonical_role") != "visual":
            previous = entity.get("canonical_role")
            entity["canonical_role"] = "visual"
            changes.append(
                {
                    "json_path": f"entities.{index}.canonical_role",
                    "from": previous,
                    "to": "visual",
                }
            )
        if detector_roles.get(region_id) == "title" and entity.get("canonical_role") != "title":
            previous = entity.get("canonical_role")
            entity["canonical_role"] = "title"
            changes.append(
                {
                    "json_path": f"entities.{index}.canonical_role",
                    "from": previous,
                    "to": "title",
                }
            )
        bbox = _normalized_bbox(entity.get("calibrated_bbox"))
        if bbox is not None and bbox != entity.get("calibrated_bbox"):
            changes.append(
                {
                    "json_path": f"entities.{index}.calibrated_bbox",
                    "from": entity.get("calibrated_bbox"),
                    "to": bbox,
                }
            )
            entity["calibrated_bbox"] = bbox

    response = CalibrationResponse.model_validate(normalized)
    component_parent = {
        region_id: component.component_id
        for component in response.components
        for region_id in component.source_region_ids
    }
    nested_group_ids = {
        child
        for group in response.groups
        for child in group.children
        if child.startswith("g")
    }
    entities = [
        {
            "region_id": entity.region_id,
            "canonical_role": entity.canonical_role,
            "parent_component_id": component_parent.get(entity.region_id, ""),
            "calibrated_bbox": list(entity.calibrated_bbox),
        }
        for entity in response.entities
    ]
    entities.sort(key=lambda item: (item["calibrated_bbox"][1], item["calibrated_bbox"][0]))
    expanded = {
        "slide_id": slide_id or normalized.get("slide_id") or "slide_000",
        "groups": [
            {
                **group.model_dump(mode="json"),
                "scope": "nested" if group.group_id in nested_group_ids else "top_level",
            }
            for group in response.groups
        ],
        "components": [item.model_dump(mode="json") for item in response.components],
        "entities": entities,
        "reading_order": [item["region_id"] for item in entities],
    }
    retained = {"slide_id", "groups", "components", "entities"}
    dropped = sorted(set(normalized) - retained)
    return expanded, {
        "schema_version": "5.0",
        "changed": bool(changes or fallbacks or dropped),
        "changes": changes,
        "dropped_top_level_fields": dropped,
        "fallbacks_applied": fallbacks,
        "validation_errors": [],
    }


def _normalized_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, width, height = [round(float(item)) for item in value]
    except (TypeError, ValueError):
        return None
    x = min(max(x, 0), GRID[0] - 1)
    y = min(max(y, 0), GRID[1] - 1)
    width = min(max(width, 1), GRID[0] - x)
    height = min(max(height, 1), GRID[1] - y)
    return [x, y, width, height]


def validate_against_regions(result: CalibratedTopologyResult, regions: list[dict]) -> None:
    region_ids = {item["region_id"] for item in regions}
    entity_ids = {item.region_id for item in result.entities}
    if entity_ids != region_ids or len(result.entities) != len(regions):
        raise ValueError("entities must contain every immutable region exactly once")
    for entity in result.entities:
        validate_bbox(entity.calibrated_bbox, entity.region_id)

    component_ids = {item.component_id for item in result.components}
    group_ids = {item.group_id for item in result.groups}
    covered = [region_id for component in result.components for region_id in component.source_region_ids]
    if len(covered) != len(set(covered)) or set(covered) != region_ids:
        raise ValueError("components must cover every region exactly once")
    parent_by_region = {
        region_id: component.component_id
        for component in result.components
        for region_id in component.source_region_ids
    }
    for entity in result.entities:
        if entity.parent_component_id != parent_by_region[entity.region_id]:
            raise ValueError(f"{entity.region_id} parent_component_id conflicts with component membership")

    hierarchy_ids = component_ids | group_ids
    for group in result.groups:
        unknown = set(group.children) - hierarchy_ids
        if unknown:
            raise ValueError(f"{group.group_id} references unknown children: {sorted(unknown)}")
    grouped_components = [
        child
        for group in result.groups
        for child in group.children
        if child in component_ids
    ]
    if set(grouped_components) != component_ids:
        raise ValueError("groups must cover every component")
    title_components = {
        item.component_id for item in result.components if item.component_role == "title"
    }
    if not any(title_components & set(group.children) for group in result.groups):
        raise ValueError("a group must contain the title component")
    if len(result.reading_order) != len(set(result.reading_order)):
        raise ValueError("reading_order IDs must be unique")
    if set(result.reading_order) != region_ids:
        raise ValueError("reading_order must cover every region")


def intersection_area(first: GridBBox, second: GridBBox) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return max(0, right - left) * max(0, bottom - top)


def build_calibration_report(result: CalibratedTopologyResult, regions: list[dict]) -> dict:
    observed = {item["region_id"]: tuple(item["bbox"]) for item in regions}
    calibrated = {item.region_id: item.calibrated_bbox for item in result.entities}
    component_boxes = {
        component.component_id: union_bbox(
            [calibrated[region_id] for region_id in component.source_region_ids]
        )
        for component in result.components
    }
    changes = []
    for region_id in sorted(observed):
        before = observed[region_id]
        after = calibrated[region_id]
        changes.append(
            {
                "region_id": region_id,
                "observed_bbox": before,
                "calibrated_bbox": after,
                "changed": before != after,
                "delta": [after[index] - before[index] for index in range(4)],
            }
        )

    overlaps = []
    entity_ids = sorted(calibrated)
    for index, first_id in enumerate(entity_ids):
        for second_id in entity_ids[index + 1 :]:
            area = intersection_area(calibrated[first_id], calibrated[second_id])
            if area:
                overlaps.append(
                    {"first_region_id": first_id, "second_region_id": second_id, "area": area}
                )

    peer_geometry = []
    for group in result.groups:
        member_ids = [child for child in group.children if child in component_boxes]
        boxes = [component_boxes[member_id] for member_id in member_ids]
        if len(boxes) < 2:
            continue
        ordered = sorted(zip(member_ids, boxes), key=lambda item: (item[1][0], item[1][1]))
        gaps = [
            ordered[index + 1][1][0] - (ordered[index][1][0] + ordered[index][1][2])
            for index in range(len(ordered) - 1)
        ] if group.arrangement == "row" else []
        peer_geometry.append(
            {
                "group_id": group.group_id,
                "member_component_ids": member_ids,
                "member_bboxes": boxes,
                "widths": [box[2] for box in boxes],
                "heights": [box[3] for box in boxes],
                "horizontal_gaps": gaps,
                "calibration_policy": group.calibration_policy.model_dump(mode="json"),
            }
        )
    return {
        "schema_version": "1.0",
        "slide_id": result.slide_id,
        "changed_region_count": sum(item["changed"] for item in changes),
        "region_count": len(changes),
        "changes": changes,
        "overlaps": overlaps,
        "peer_geometry": peer_geometry,
    }


def deterministic_calibration(slide_id: str, regions: list[dict]) -> CalibratedTopologyResult:
    component_ids: set[str] = set()
    components = []
    entities = []
    for index, region in enumerate(regions, start=1):
        base = "c_" + "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(region["region_id"]).lower()
        ).strip("_")
        component_id = base or f"c_region_{index:03d}"
        suffix = 2
        while component_id in component_ids:
            component_id = f"{base}_{suffix}"
            suffix += 1
        component_ids.add(component_id)
        canonical_role = _canonical_role_from_region(region)
        components.append(
            {
                "component_id": component_id,
                "component_role": _component_role(canonical_role),
                "source_region_ids": [region["region_id"]],
            }
        )
        entities.append(
            {
                "region_id": region["region_id"],
                "canonical_role": canonical_role,
                "parent_component_id": component_id,
                "calibrated_bbox": region["bbox"],
            }
        )
    title_components = [
        item["component_id"] for item in components if item["component_role"] == "title"
    ]
    content_components = [
        item["component_id"] for item in components if item["component_role"] != "title"
    ]
    groups = [
        {
            "group_id": "g_header",
            "role": "header",
            "scope": "top_level",
            "arrangement": "column",
            "children": title_components,
            "peer_relation": "none",
            "calibration_policy": {
                "size_policy": "preserve",
                "alignment": "preserve",
                "spacing": "preserve",
            },
        },
        {
            "group_id": "g_content",
            "role": "main content",
            "scope": "top_level",
            "arrangement": "row" if len(content_components) > 1 else "column",
            "children": content_components,
            "peer_relation": "none",
            "calibration_policy": {
                "size_policy": "preserve",
                "alignment": "preserve",
                "spacing": "preserve",
            },
        },
    ]
    entities.sort(key=lambda item: (item["calibrated_bbox"][1], item["calibrated_bbox"][0]))
    result = CalibratedTopologyResult.model_validate(
        {
            "slide_id": slide_id,
            "groups": groups,
            "components": components,
            "entities": entities,
            "reading_order": [item["region_id"] for item in entities],
        }
    )
    validate_against_regions(result, regions)
    return result


def _canonical_role_from_region(region: dict) -> str:
    detector_role = region.get("detector_role")
    if detector_role == "title":
        return "title"
    if region.get("predicted_type") == "visual":
        return "visual"
    if detector_role in {"caption", "label"}:
        return detector_role
    return "standalone_text"


def _component_role(canonical_role: str) -> str:
    if canonical_role == "title":
        return "title"
    if canonical_role == "visual":
        return "primary_visual"
    if canonical_role == "standalone_text":
        return "standalone_text"
    return "annotation"


def call_joint_vlm(
    client: ApiMartClient,
    *,
    reference_path: Path,
    region_overlay_path: Path,
    regions_payload: dict,
) -> str:
    user_content = [
        {"type": "input_text", "text": CALIBRATION_PROMPT},
        {"type": "input_text", "text": "Input 1: low-step diffusion-generated ghost slide."},
        {"type": "input_image", "image_url": image_to_data_uri(reference_path)},
        {"type": "input_text", "text": "Input 2: detected-region overlay with immutable IDs."},
        {"type": "input_image", "image_url": image_to_data_uri(region_overlay_path)},
        {"type": "input_text", "text": "Input 3: slide ID and detected-region observations."},
        {"type": "input_text", "text": json.dumps(regions_payload, indent=2)},
    ]
    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Return one valid JSON object with no markdown fences or outside prose.",
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    client.call_count += 1
    response = client._post_json("/responses", payload)
    client.last_usage = _extract_usage(response)
    text = _extract_text_response(response)
    if not text.strip():
        raise RuntimeError(f"APIMART calibration call returned empty content: {response}")
    return text


def extract_slide(slide_id: str, output_dir: Path, reuse_raw: bool) -> dict:
    slide_dir = output_dir / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(slide_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    reference_path = REFERENCE_DIR / f"{slide_id}.png"
    v2_path, regions = load_regions(slide_id)
    regions_payload = {
        "slide_id": slide_id,
        "coordinate_system": {"width": GRID[0], "height": GRID[1]},
        "regions": regions,
    }
    regions_path = slide_dir / "regions.json"
    write_json(regions_path, regions_payload)
    overlay_path = slide_dir / "immutable_region_overlay.png"
    render_region_overlay(reference_path, regions, overlay_path)
    prompt_path = slide_dir / "calibration_prompt.txt"
    write_text(prompt_path, CALIBRATION_PROMPT)
    write_text(attempt_dir / prompt_path.name, CALIBRATION_PROMPT)
    prompt_sha256 = hashlib.sha256(CALIBRATION_PROMPT.encode("utf-8")).hexdigest()
    input_manifest_path = slide_dir / "model_inputs.json"
    write_json(
        input_manifest_path,
        {
            "slide_id": slide_id,
            "inputs": [
                {"input_number": 1, "kind": "image", "role": "ghost_slide", "path": str(reference_path.resolve()), "sha256": sha256(reference_path)},
                {"input_number": 2, "kind": "image", "role": "immutable_region_overlay", "path": str(overlay_path.resolve()), "sha256": sha256(overlay_path)},
                {"input_number": 3, "kind": "json", "role": "detected_region_observations", "path": str(regions_path.resolve()), "sha256": sha256(regions_path)},
            ],
        },
    )

    raw_path = slide_dir / "vlm_raw_response.json"
    previous_manifest_path = slide_dir / "run_manifest.json"
    previous_manifest = read_json(previous_manifest_path) if previous_manifest_path.exists() else {}
    client = ApiMartClient(env_path=V9_ROOT / ".env")
    started_at = now_iso()
    result: CalibratedTopologyResult | None = None
    report_path = None
    normalization_path = None
    usage = None
    fallback_error = None
    api_error = None
    reused_raw = False
    try:
        if reuse_raw and raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8")
            usage = previous_manifest.get("usage")
            reused_raw = True
        elif reuse_raw:
            raw = "{}"
            fallback_error = f"missing reusable raw response: {raw_path}"
        else:
            raw = call_joint_vlm(
                client,
                reference_path=reference_path,
                region_overlay_path=overlay_path,
                regions_payload=regions_payload,
            )
            write_text(raw_path, raw)
            usage = client.last_usage
    except Exception as exc:
        api_error = str(exc)
        raw = ""

    if api_error is None:
        if raw_path.exists():
            write_text(attempt_dir / raw_path.name, raw_path.read_text(encoding="utf-8"))
        try:
            normalized, normalization_report = normalize_payload(
                extract_model_object(raw), regions, slide_id
            )
            result = CalibratedTopologyResult.model_validate(normalized)
            validate_against_regions(result, regions)
        except Exception as exc:
            fallback_error = fallback_error or str(exc)
            result = deterministic_calibration(slide_id, regions)
            normalization_report = {
                "schema_version": "5.0",
                "changed": True,
                "changes": [],
                "dropped_top_level_fields": [],
                "fallbacks_applied": [
                    {"reason": "invalid calibration response", "strategy": "one_component_per_region"}
                ],
                "validation_errors": [str(exc)],
            }
        normalization_path = slide_dir / "normalization_report.json"
        write_json(normalization_path, normalization_report)
        write_json(attempt_dir / normalization_path.name, normalization_report)
        write_json(slide_dir / "calibrated_topology.json", result)
        write_json(attempt_dir / "calibrated_topology.json", result)
        report_path = slide_dir / "calibration_report.json"
        write_json(report_path, build_calibration_report(result, regions))
        status = "succeeded"
        error = None
    else:
        status = "failed"
        error = api_error

    manifest = {
        "slide_id": slide_id,
        "status": status,
        "quality_status": "fallback" if fallback_error else "strict_pass",
        "error": error,
        "warnings": [fallback_error] if fallback_error else [],
        "model": MODEL,
        "model_call_count": client.call_count,
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
            "prompt_sha256": prompt_sha256,
        },
        "group_count": len(result.groups) if result else 0,
        "component_count": len(result.components) if result else 0,
        "entity_count": len(result.entities) if result else 0,
        "artifacts": {
            "attempt": str(attempt_dir.resolve()),
            "prompt": str(prompt_path.resolve()),
            "raw_response": str(raw_path.resolve()) if raw_path.exists() else None,
            "normalization_report": str(normalization_path.resolve()) if normalization_path else None,
            "calibrated_topology": str((slide_dir / "calibrated_topology.json").resolve()) if result else None,
            "calibration_report": str(report_path.resolve()) if report_path else None,
            "manifest": str((slide_dir / "run_manifest.json").resolve()),
        },
    }
    write_json(slide_dir / "run_manifest.json", manifest)
    write_json(attempt_dir / "run_manifest.json", manifest)
    return manifest


def _next_attempt_dir(slide_dir: Path) -> Path:
    attempts = slide_dir / "attempts"
    indices = []
    for path in attempts.glob("attempt_*"):
        try:
            indices.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return attempts / f"attempt_{max(indices, default=0) + 1:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly recover topology and calibrate entity geometry.")
    parser.add_argument(
        "--slide-id",
        action="append",
        choices=[f"slide_{index:03d}" for index in range(1, 7)],
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
                    lambda slide_id: extract_slide(slide_id, output_dir, args.reuse_raw),
                    slide_ids,
                )
            )
    failed = [slide for slide in slides if slide["status"] == "failed"]
    manifest = {
        "stage": "joint_topology_and_geometry_calibration_v2",
        "status": "failed" if failed else "succeeded",
        "error": f"{len(failed)} slide(s) failed" if failed else None,
        "model": MODEL,
        "schema_version": "4.0",
        "slide_count": len(slides),
        "model_call_count": sum(slide["model_call_count"] for slide in slides),
        "total_tokens": sum((slide.get("usage") or {}).get("total_tokens", 0) or 0 for slide in slides),
        "parallel_execution": workers > 1,
        "max_workers": workers,
        "started_at": started_at,
        "finished_at": now_iso(),
        "slides": slides,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
