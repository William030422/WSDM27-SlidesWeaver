from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import extract_json, read_json, write_json, write_text
from v9_deck_planning.planner import infer_text_presentation_policy

from .config import PipelineConfig


TEXT_ROLES = {
    "title",
    "standalone_text",
    "supporting_text",
    "caption",
    "label",
    "internal_label",
}
MAJOR_EXPLANATORY_ROLES = {"standalone_text", "supporting_text"}
HOLDER_ROLE_TAGS = {"short_explanation", "comparison", "definition", "takeaway"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisualRegionBrief(StrictModel):
    region_id: str
    semantic_brief: str

    @field_validator("region_id", "semantic_brief")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("visual region fields must be non-empty")
        return value


class TextRegionContent(StrictModel):
    region_id: str
    role: Literal[
        "title",
        "standalone_text",
        "supporting_text",
        "caption",
        "label",
        "internal_label",
    ]
    lines: list[str] = Field(min_length=1)

    @field_validator("region_id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text region ID must be non-empty")
        return value

    @field_validator("lines")
    @classmethod
    def non_empty_lines(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("text lines must be non-empty")
        return normalized


class RegionContentPlan(StrictModel):
    slide_id: str = Field(pattern=r"^slide_[0-9]{3}$")
    visual_regions: list[VisualRegionBrief]
    text_regions: list[TextRegionContent]


class RegionContentClient(Protocol):
    def vlm_json(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
    ) -> str: ...


def run_region_content_planning(
    *,
    project_root: Path,
    step2_dir: Path | None = None,
    slide_spec_dir: Path,
    topology_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_llm: bool,
    reuse_raw: bool = False,
    client_factory: Callable[[], RegionContentClient] | None = None,
    text_policy: dict[str, Any] | None = None,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    step2_dir = step2_dir or slide_spec_dir.parent / "step2"
    deck_plan = read_json(step2_dir / "deck_plan.json")
    rough_by_id = {item["slide_id"]: item for item in deck_plan.get("slides", [])}
    asset_manifest_path = step2_dir / "asset_manifest.json"
    asset_manifest = read_json(asset_manifest_path) if asset_manifest_path.is_file() else {"assets": []}
    raw_inventories: dict[str, list[dict[str, Any]]] = {}
    slide_inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for slide_id in slide_ids:
        slide_spec = read_json(slide_spec_dir / "slides" / slide_id / "slide_spec.json")
        # Keep the original per-slide brief alongside the concise bridge. The
        # bridge is an image-planning aid, not a replacement for requirements.
        slide_spec = {**slide_spec,
                      "rough_message": rough_by_id.get(slide_id, {}).get("rough_message", slide_spec["message"]),
                      "style": deck_plan.get("style", "")}
        topology = read_json(
            topology_dir / "calibrated_topology" / slide_id / "calibrated_topology.json"
        )
        layout = read_json(topology_dir / "postprocessed" / "layouts" / f"{slide_id}.json")
        overview = ensure_explanatory_overview(topology, layout, slide_spec)
        if overview is not None:
            write_json(topology_dir / "calibrated_topology" / slide_id / "calibrated_topology.json", topology)
            write_json(topology_dir / "postprocessed" / "layouts" / f"{slide_id}.json", layout)
        inventory = build_region_inventory(topology, layout)
        semantic_topology_path = (
            topology_dir / "semantic_topology" / slide_id / "semantic_topology.json"
        )
        if semantic_topology_path.is_file():
            inventory = attach_semantic_context(
                inventory, read_json(semantic_topology_path)
            )
        raw_inventories[slide_id] = inventory
        slide_inputs[slide_id] = (slide_spec, topology, layout)
    policy = text_policy if text_policy is not None else build_text_presentation_policy(
        deck_plan=deck_plan,
        asset_manifest=asset_manifest,
        asset_manifest_path=asset_manifest_path,
        inventories=raw_inventories,
    )
    if write_stage_manifest:
        write_json(output_dir / "text_presentation_policy.json", policy)
    client = None
    if not mock_llm:
        client = client_factory() if client_factory else ApiMartClient(
            api_root=config.api_root,
            api_key_env=config.api_key_env,
            env_path=project_root / ".env",
            timeout_seconds=config.timeout_seconds,
            poll_initial_delay_seconds=config.poll_initial_delay_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
            poll_timeout_seconds=config.poll_timeout_seconds,
        )

    records = []
    warnings = []
    model_call_count = 0
    for slide_id in slide_ids:
        slide_dir = output_dir / "slides" / slide_id
        slide_spec, _topology, _layout = slide_inputs[slide_id]
        inventory = apply_text_presentation_policy(raw_inventories[slide_id], policy)
        write_json(slide_dir / "region_inventory.json", inventory)
        layout_image_path = topology_dir / "postprocessed" / f"{slide_id}.png"
        if not layout_image_path.is_file():
            raise FileNotFoundError(f"postprocessed layout image is missing: {layout_image_path}")
        prompt = region_content_prompt(
            slide_spec=slide_spec,
            inventory=inventory,
        )
        prompt_path = slide_dir / "prompt.txt"
        raw_path = slide_dir / "raw_response.json"
        plan_path = slide_dir / "region_content_plan.json"
        write_text(prompt_path, prompt)
        if mock_llm:
            mock_plan = mock_region_content_plan(slide_spec, inventory)
            raw = mock_plan.model_dump_json(indent=2)
            write_text(raw_path, raw)
        elif reuse_raw and raw_path.is_file():
            raw = raw_path.read_text(encoding="utf-8")
        else:
            if client is None:
                raise RuntimeError("region-content planning requires an API client")
            from .api_journal import journal_response
            tokens = min(config.vlm_max_tokens, 4096)
            response = journal_response(path=slide_dir / "api_response.json",
                request={"method": "vlm_json", "api_root": config.api_root, "model": config.chat_model,
                         "prompt": prompt, "max_tokens": tokens}, image_paths=[layout_image_path],
                invoke=lambda: client.vlm_json(model=config.chat_model, prompt=prompt,
                                               image_path=layout_image_path, max_tokens=tokens))
            raw = response["raw"]
            model_call_count += int(not response["reused"])
            write_text(raw_path, raw)
        try:
            plan, normalization = parse_region_content(raw, slide_id, inventory)
            validate_region_content_plan(plan, slide_id, inventory)
        except Exception as exc:
            plan = mock_region_content_plan(slide_spec, inventory)
            normalization = {
                "changed": True,
                "changes": [],
                "validation_errors": [str(exc)],
                "fallbacks_applied": [
                    {"strategy": "inventory_complete_content", "slide_id": slide_id}
                ],
            }
            warnings.append({"slide_id": slide_id, "error": str(exc)})
        write_json(slide_dir / "normalization_report.json", normalization)
        write_json(plan_path, plan)
        records.append(
            {
                "slide_id": slide_id,
                "visual_generation_unit_count": len(plan.visual_regions),
                "text_region_count": len(plan.text_regions),
                "region_content_plan": str(plan_path.resolve()),
                "region_inventory": str((slide_dir / "region_inventory.json").resolve()),
                "layout_image": str(layout_image_path.resolve()),
                "prompt": str(prompt_path.resolve()),
                "raw_response": str(raw_path.resolve()),
            }
        )

    manifest = {
        "stage": "per_generation_unit_semantic_and_text_planning",
        "status": "succeeded",
        "quality_status": "fallback" if warnings else "strict_pass",
        "warnings": warnings,
        "model": config.chat_model,
        "model_call_count": model_call_count,
        "reused_raw": reuse_raw,
        "mock_llm": mock_llm,
        "text_presentation_policy": policy,
        "slide_count": len(records),
        "output_dir": str(output_dir.resolve()),
        "slides": records,
    }
    if write_stage_manifest:
        write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def ensure_explanatory_overview(
    topology: dict[str, Any], layout: dict[str, Any], slide_spec: dict[str, Any]
) -> dict[str, Any] | None:
    """Reserve one source-owned overview when an explanatory slide has only a title.

    This is a pre-commit inventory correction, not a caption for every picture.
    Explicit covers/dividers stay covers; without a declared type, a message
    essentially repeating the title is the conservative cover heuristic.
    """
    explicit_type = str(slide_spec.get("slide_type") or slide_spec.get("type") or slide_spec.get("role") or "").casefold()
    if explicit_type in {"cover", "title", "title_slide", "section_divider", "divider"}:
        return None
    entities = topology.get("entities", [])
    if any(e.get("canonical_role") in TEXT_ROLES - {"title"} for e in entities):
        return None
    message = str(slide_spec.get("rough_message") or slide_spec.get("message") or "").strip()
    title_words = set(re.findall(r"\w+", str(slide_spec.get("title", "")).casefold()))
    message_words = re.findall(r"\w+", message.casefold())
    if not message_words or (len(message_words) <= len(title_words) + 2
                             and len(set(message_words) - title_words) <= 2):
        return None
    boxes = {e["region_id"]: e for e in layout.get("entities", [])}
    titles = [boxes[e["region_id"]]["postprocessed_bbox_px"] for e in entities
              if e.get("canonical_role") == "title" and e["region_id"] in boxes]
    visuals = [e for e in entities if e.get("canonical_role") == "visual" and e["region_id"] in boxes]
    if not titles or not visuals:
        return None
    canvas = layout.get("canvas", {"width": 1536, "height": 864})
    cw, ch = float(canvas["width"]), float(canvas["height"])
    cell_x, cell_y = cw / 64, ch / 36
    margin = round(cw * .03)
    y = math.ceil((max(b[1] + b[3] for b in titles) + cell_y / 2) / cell_y) * cell_y
    overview_bbox = [margin, y, cw - margin * 2, cell_y * 5]
    top = overview_bbox[1] + overview_bbox[3] + cell_y
    old_top = min(boxes[e["region_id"]]["postprocessed_bbox_px"][1] for e in visuals)
    bottom = max(sum(boxes[e["region_id"]]["postprocessed_bbox_px"][i] for i in (1,3)) for e in visuals)
    if bottom - top < cell_y * 4:
        return None  # No safe inventory reservation; existing quality reports retain this limitation.
    scale = min(1.0, (bottom - top) / max(1, bottom - old_top))

    def grid(bbox):
        x, by, w, h = bbox
        gx, gy = max(0, math.floor(x / cell_x)), max(0, math.floor(by / cell_y))
        return [gx, gy, min(64-gx, max(1, math.ceil(w/cell_x))), min(36-gy, max(1, math.ceil(h/cell_y)))]

    for entity in visuals:
        item = boxes[entity["region_id"]]
        bx, by, bw, bh = item["postprocessed_bbox_px"]
        new_box = [bx, top + (by - old_top) * scale, bw, bh * scale]
        item["postprocessed_bbox_px"] = new_box
        entity["calibrated_bbox"] = grid(new_box)
    rid, cid = "b_semantic_overview", "c_semantic_overview"
    entities.append({"region_id": rid, "canonical_role": "supporting_text", "parent_component_id": cid,
                     "calibrated_bbox": grid(overview_bbox), "required_text": message,
                     "text_provenance": "authoritative_slide_message"})
    topology.setdefault("components", []).append({"component_id": cid, "component_role": "standalone_text", "source_region_ids": [rid]})
    for group in topology.get("groups", []):
        if group.get("group_id") == "g_header":
            group.setdefault("children", []).append(cid)
    order = topology.setdefault("reading_order", [e["region_id"] for e in entities if e["region_id"] != rid])
    title_index = next((i for i, value in enumerate(order) if any(e["region_id"] == value and e.get("canonical_role") == "title" for e in entities)), -1)
    order.insert(title_index + 1, rid)
    layout.setdefault("entities", []).append({"region_id": rid, "input_calibrated_bbox_px": overview_bbox,
        "policy_compiled_bbox_px": overview_bbox, "postprocessed_bbox_px": overview_bbox})
    layout.setdefault("header_region_ids", []).append(rid)
    result = {"region_id": rid, "bbox_px": overview_bbox, "required_text": message,
              "text_provenance": "authoritative_slide_message"}
    topology["semantic_overview"] = result
    return result


def build_region_inventory(
    topology: dict[str, Any],
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    canvas = layout.get("canvas", {"width": 1536, "height": 864})
    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    boxes = {
        item["region_id"]: item["postprocessed_bbox_px"]
        for item in layout["entities"]
    }
    inventory = []
    for unit in build_generation_units(topology, layout):
        bbox = unit["bbox_px"]
        width = float(bbox[2])
        height = float(bbox[3])
        inventory.append(
            {
                "region_id": unit["unit_id"],
                "role": "visual",
                "bbox_px": bbox,
                "position": describe_position(bbox, canvas_width, canvas_height),
                "parent_component_id": unit.get("component_id"),
                "parent_group_id": unit.get("group_id"),
                "generation_unit": True,
                "source_region_ids": unit["source_region_ids"],
                "relative_size": {
                    "width": round(width / canvas_width, 4),
                    "height": round(height / canvas_height, 4),
                    "area": round((width * height) / (canvas_width * canvas_height), 4),
                },
                "aspect_ratio": round(width / height, 4),
                "canvas_px": [round(canvas_width), round(canvas_height)],
            }
        )

    for entity in topology["entities"]:
        region_id = entity["region_id"]
        role = entity["canonical_role"]
        if region_id not in boxes or role in {"visual", "decorative"}:
            continue
        bbox = boxes[region_id]
        width = float(bbox[2])
        height = float(bbox[3])
        record = {
            "region_id": region_id,
            "role": role,
            "bbox_px": bbox,
            "position": describe_position(bbox, canvas_width, canvas_height),
            "parent_component_id": entity["parent_component_id"],
            "relative_size": {
                "width": round(width / canvas_width, 4),
                "height": round(height / canvas_height, 4),
                "area": round((width * height) / (canvas_width * canvas_height), 4),
            },
            "aspect_ratio": round(width / height, 4),
            "canvas_px": [round(canvas_width), round(canvas_height)],
        }
        if entity.get("required_text"):
            record.update(required_text=entity["required_text"],
                          text_provenance=entity.get("text_provenance", "semantic_annotation"))
        inventory.append(record)
    inventory.sort(key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))
    return inventory


def attach_semantic_context(
    inventory: list[dict[str, Any]], semantic_topology: dict[str, Any]
) -> list[dict[str, Any]]:
    blocks = {
        item["block_id"]: item for item in semantic_topology.get("blocks", [])
    }
    connections: dict[str, list[dict[str, Any]]] = {
        block_id: [] for block_id in blocks
    }
    for item in semantic_topology.get("connections", []):
        source = item.get("source_block_id")
        target = item.get("target_block_id")
        if source in blocks and target in blocks:
            connections[source].append(
                {
                    "direction": "outgoing",
                    "relation": item.get("relation"),
                    "other_block_id": target,
                    "other_semantic_role": blocks[target].get("semantic_role"),
                }
            )
            connections[target].append(
                {
                    "direction": "incoming",
                    "relation": item.get("relation"),
                    "other_block_id": source,
                    "other_semantic_role": blocks[source].get("semantic_role"),
                }
            )
    by_region_id: dict[str, dict[str, Any]] = {}
    for block_id, block in blocks.items():
        context = {
            "semantic_block_id": block_id,
            "semantic_block_role": block.get("role"),
            "semantic_modality": block.get("modality"),
            "semantic_role": block.get("semantic_role"),
            "semantic_source_region_ids": block.get("source_region_ids", []),
            "semantic_connections": connections.get(block_id, []),
        }
        region_ids = {
            block_id,
            _semantic_component_id(block_id),
            f"{block_id}_visual",
            f"{block_id}_text_top",
            f"{block_id}_text_bottom",
        }
        for region_id in region_ids:
            by_region_id[region_id] = context
    result = [{**item, **by_region_id.get(item["region_id"], {})} for item in inventory]
    for item in result:
        if (item.get("role") == "label" and item.get("semantic_modality") == "visual_only"
                and item.get("semantic_block_role") == "semantic_stage"
                and item.get("semantic_connections")):
            item["required_text"] = str(item["semantic_role"]).strip()
            item["text_provenance"] = "semantic_stage_annotation"
    return result


def _semantic_component_id(block_id: str) -> str:
    suffix = block_id.removeprefix("b_").removeprefix("b") or "block"
    return f"c_{suffix}"


def build_generation_units(
    topology: dict[str, Any],
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collapse visual entities into the smallest semantic raster units.

    Components are the default unit. An overlay group is one shared scene, so
    its visual descendants are emitted as a single group unit. Legacy fixtures
    without calibrated groups/components keep their visual entities as units.
    """
    boxes = {
        item["region_id"]: item["postprocessed_bbox_px"]
        for item in layout.get("entities", [])
        if "postprocessed_bbox_px" in item
    }
    all_entities = {
        item["region_id"]: item
        for item in topology.get("entities", [])
        if item["region_id"] in boxes
    }
    entities = {
        item["region_id"]: item
        for item in topology.get("entities", [])
        if item.get("canonical_role") == "visual" and item["region_id"] in boxes
    }
    components = {
        item["component_id"]: item for item in topology.get("components", [])
    }
    groups = {item["group_id"]: item for item in topology.get("groups", [])}
    if not entities or not components:
        return [
            {
                "unit_id": region_id,
                "source_region_ids": [region_id],
                "bbox_px": boxes[region_id],
            }
            for region_id in sorted(
                entities, key=lambda value: (boxes[value][1], boxes[value][0])
            )
        ]

    parent_group: dict[str, str] = {}
    for group_id, group in groups.items():
        for child in group.get("children", []):
            if child in groups:
                parent_group[child] = group_id
            elif child in components:
                parent_group[child] = group_id

    def overlay_ancestor(node_id: str) -> str | None:
        parent = parent_group.get(node_id)
        while parent is not None:
            if groups[parent].get("arrangement") == "overlay":
                return parent
            parent = parent_group.get(parent)
        return None

    def component_visuals(component_id: str) -> list[str]:
        return [
            region_id
            for region_id in components[component_id].get("source_region_ids", [])
            if region_id in entities
        ]

    def descendants(node_id: str) -> list[str]:
        if node_id in components:
            return component_visuals(node_id)
        regions: list[str] = []
        for child in groups[node_id].get("children", []):
            regions.extend(descendants(child))
        return regions

    units: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        if group.get("arrangement") != "overlay" or overlay_ancestor(group_id):
            continue
        source_region_ids = descendants(group_id)
        if source_region_ids:
            units.append(
                {
                    "unit_id": group_id,
                    "group_id": group_id,
                    "source_region_ids": source_region_ids,
                    "bbox_px": _union_bboxes([boxes[item] for item in source_region_ids]),
                }
            )

    for component_id in components:
        source_region_ids = component_visuals(component_id)
        if not source_region_ids or overlay_ancestor(component_id):
            continue
        visual_bbox = _union_bboxes([boxes[item] for item in source_region_ids])
        component_region_ids = [
            region_id
            for region_id in components[component_id].get("source_region_ids", [])
            if region_id in boxes
        ]
        captions = [
            region_id
            for region_id in component_region_ids
            if all_entities[region_id].get("canonical_role") == "caption"
        ]
        if len(captions) == 1:
            component_bbox = _union_bboxes([boxes[item] for item in component_region_ids])
            caption_top = boxes[captions[0]][1]
            if caption_top > component_bbox[1]:
                visual_bbox = [
                    component_bbox[0],
                    component_bbox[1],
                    component_bbox[2],
                    caption_top - component_bbox[1],
                ]
        units.append(
            {
                "unit_id": component_id,
                "component_id": component_id,
                "source_region_ids": source_region_ids,
                "bbox_px": visual_bbox,
            }
        )

    if not units:
        return [
            {
                "unit_id": region_id,
                "source_region_ids": [region_id],
                "bbox_px": boxes[region_id],
            }
            for region_id in sorted(
                entities, key=lambda value: (boxes[value][1], boxes[value][0])
            )
        ]
    return sorted(units, key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))


def _union_bboxes(boxes: list[list[int]]) -> list[int]:
    left = min(item[0] for item in boxes)
    top = min(item[1] for item in boxes)
    right = max(item[0] + item[2] for item in boxes)
    bottom = max(item[1] + item[3] for item in boxes)
    return [left, top, right - left, bottom - top]


def region_content_prompt(
    *,
    slide_spec: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> str:
    inventory_json = json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))
    return f"""Use the attached layout and this structured region inventory to write editable slide content.

Slide title: {slide_spec['title']}
Core message: {slide_spec['message']}
Original per-slide teaching requirements (authoritative, do not lose conditions or steps): {slide_spec.get('rough_message', slide_spec['message'])}
Requested deck style and character identities: {slide_spec.get('style', '')}
Rough visual intent, used only as soft inspiration: {slide_spec.get('visual_intent', '')}
Region inventory: {inventory_json}

For text regions, use the role, relative size, aspect ratio, effective_text_bbox_px, and capacity_hint as density signals,
not hard word limits. Write the shortest clear explanation likely to fit at the normal role font size. Keep labels and
captions to one short point; larger supporting regions may use several short `lines` only when needed. Preserve the
supplied title with at most minimal shortening. Prefer concrete observation, mechanism, and consequence. Do not add
bullet glyphs or numbering.
When the inventory supplies required_text, use that editable stage label. Do not omit intermediate steps because
their source image was visual-only. Keep sequential stage labels unnumbered; do not create numbering gaps.
Preserve all original per-slide requirements (including delay as well as uncertainty); paraphrase concisely without
dropping a condition. Keep human pilots human and machines machines, with the requested character appearance.
Never add a prohibition on requested character identities or replace the requested palette with a neutral/white scene.

For visual generation units, give one concrete semantic brief that supports the message. A unit may be a component or
an overlay group containing several internal visual entities; describe the shared scene or composition as one whole.
Use `semantic_role` and `semantic_connections` when present so the brief preserves the extracted stage meaning and its
relationship to peer stages. Do not split a unit into its child visuals. Use every supplied generation unit ID and role exactly once.
Return only:
{{"slide_id":"{slide_spec['slide_id']}","visual_regions":[{{"region_id":"...","semantic_brief":"..."}}],"text_regions":[{{"region_id":"...","role":"...","lines":["..."]}}]}}
"""


def build_text_presentation_policy(
    *,
    deck_plan: dict[str, Any],
    asset_manifest: dict[str, Any],
    asset_manifest_path: Path,
    inventories: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    requested = deck_plan.get("text_presentation_policy")
    if requested not in {"light", "styled"}:
        requested = infer_text_presentation_policy(str(deck_plan.get("style", "")))
    holders = []
    for asset in asset_manifest.get("assets", []):
        safe = _valid_text_safe_rect(asset.get("text_safe_rect"))
        asset_path = asset_manifest_path.parent / str(asset.get("path", ""))
        if (
            asset.get("available")
            and asset.get("category") == "raster_holder"
            and safe is not None
            and asset_path.is_file()
            and float(asset.get("aspect_ratio", 0)) > 0
        ):
            holders.append(
                {
                    "asset_id": asset["asset_id"],
                    "aspect_ratio": float(asset["aspect_ratio"]),
                    "text_safe_rect": safe,
                    "allowed_roles": list(asset.get("allowed_roles", [])),
                }
            )

    role_families: dict[str, str] = {}
    fallbacks: list[dict[str, Any]] = []
    if requested == "styled":
        for role in sorted(MAJOR_EXPLANATORY_ROLES):
            regions = [
                item
                for inventory in inventories.values()
                for item in inventory
                if item["role"] == role
            ]
            candidates = [
                holder
                for holder in holders
                if not holder["allowed_roles"]
                or HOLDER_ROLE_TAGS.intersection(holder["allowed_roles"])
            ]
            if regions and candidates:
                selected = max(
                    candidates,
                    key=lambda holder: sum(
                        _holder_effective_area_fraction(item["bbox_px"], holder)
                        for item in regions
                    )
                    / len(regions),
                )
                role_families[role] = selected["asset_id"]
            elif regions:
                fallbacks.append(
                    {
                        "role": role,
                        "reason": "no_compatible_holder_with_safe_area",
                        "fallback": "direct_text",
                    }
                )
    resolved = "styled" if requested == "styled" and role_families else "light"
    if requested == "styled" and not role_families:
        fallbacks.append(
            {
                "scope": "deck",
                "reason": "styled_policy_has_no_usable_major_text_holder",
                "fallback": "light",
            }
        )
    return {
        "schema_version": "1.0",
        "requested": requested,
        "resolved": resolved,
        "major_explanatory_roles": sorted(MAJOR_EXPLANATORY_ROLES),
        "holder_family_by_role": role_families,
        "available_holder_ids": [holder["asset_id"] for holder in holders],
        "holders": holders,
        "fallbacks": fallbacks,
    }


def apply_text_presentation_policy(
    inventory: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    holders = {holder["asset_id"]: holder for holder in policy.get("holders", [])}
    family_by_role = policy.get("holder_family_by_role", {})
    enriched: list[dict[str, Any]] = []
    for source in inventory:
        item = dict(source)
        if item["role"] not in TEXT_ROLES:
            enriched.append(item)
            continue
        asset_id = family_by_role.get(item["role"])
        holder = holders.get(asset_id)
        if policy.get("resolved") == "styled" and holder is not None:
            holder_bbox = _contain_bbox(item["bbox_px"], holder["aspect_ratio"])
            effective_bbox = _safe_bbox(holder_bbox, holder["text_safe_rect"])
            item["presentation_mode"] = "raster_holder_text"
            item["holder_binding"] = {
                "asset_id": asset_id,
                "holder_family": asset_id,
                "holder_bbox_px": holder_bbox,
                "text_safe_rect": holder["text_safe_rect"],
                "aspect_ratio_change": 0.0,
            }
        else:
            effective_bbox = [float(value) for value in item["bbox_px"]]
            item["presentation_mode"] = "direct_text"
            item["holder_binding"] = None
        item["effective_text_bbox_px"] = effective_bbox
        item["capacity_hint"] = _capacity_hint(effective_bbox, item["canvas_px"])
        enriched.append(item)
    return enriched


def _valid_text_safe_rect(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        safe = {key: float(value[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None
    if (
        safe["x"] < 0
        or safe["y"] < 0
        or safe["width"] <= 0
        or safe["height"] <= 0
        or safe["x"] + safe["width"] > 1
        or safe["y"] + safe["height"] > 1
    ):
        return None
    return safe


def _contain_bbox(bbox: list[float], aspect_ratio: float) -> list[float]:
    x, y, width, height = (float(value) for value in bbox)
    if width / height > aspect_ratio:
        holder_height = height
        holder_width = height * aspect_ratio
    else:
        holder_width = width
        holder_height = width / aspect_ratio
    return [
        round(x + (width - holder_width) / 2, 3),
        round(y + (height - holder_height) / 2, 3),
        round(holder_width, 3),
        round(holder_height, 3),
    ]


def _safe_bbox(holder_bbox: list[float], safe: dict[str, float]) -> list[float]:
    x, y, width, height = holder_bbox
    return [
        round(x + width * safe["x"], 3),
        round(y + height * safe["y"], 3),
        round(width * safe["width"], 3),
        round(height * safe["height"], 3),
    ]


def _holder_effective_area_fraction(
    bbox: list[float], holder: dict[str, Any]
) -> float:
    outer = _contain_bbox(bbox, holder["aspect_ratio"])
    safe = holder["text_safe_rect"]
    effective_area = outer[2] * outer[3] * safe["width"] * safe["height"]
    height_retention = (outer[3] * safe["height"]) / max(1.0, float(bbox[3]))
    return (
        effective_area / max(1.0, float(bbox[2]) * float(bbox[3]))
    ) * height_retention


def _capacity_hint(effective_bbox: list[float], canvas_px: list[int]) -> str:
    area_ratio = (effective_bbox[2] * effective_bbox[3]) / (
        float(canvas_px[0]) * float(canvas_px[1])
    )
    if area_ratio < 0.035:
        return "compact"
    if area_ratio >= 0.10:
        return "roomy"
    return "standard"


def validate_region_content_plan(
    plan: RegionContentPlan,
    slide_id: str,
    inventory: list[dict[str, Any]],
) -> None:
    if plan.slide_id != slide_id:
        raise ValueError(f"region plan ID mismatch: expected {slide_id}, got {plan.slide_id}")
    expected_visuals = {
        item["region_id"] for item in inventory if item["role"] == "visual"
    }
    expected_text = {
        item["region_id"]: item for item in inventory if item["role"] in TEXT_ROLES
    }
    actual_visuals = [item.region_id for item in plan.visual_regions]
    actual_text = [item.region_id for item in plan.text_regions]
    if len(actual_visuals) != len(set(actual_visuals)) or set(actual_visuals) != expected_visuals:
        raise ValueError("visual briefs must cover every generation unit exactly once")
    if len(actual_text) != len(set(actual_text)) or set(actual_text) != set(expected_text):
        raise ValueError("text content must cover every text region exactly once")
    for text in plan.text_regions:
        expected = expected_text[text.region_id]
        if text.role != expected["role"]:
            raise ValueError(f"{text.region_id} role must be {expected['role']}")


def parse_region_content(
    raw: str,
    slide_id: str,
    inventory: list[dict[str, Any]],
) -> tuple[RegionContentPlan, dict[str, Any]]:
    payload = extract_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("region-content response must be one JSON object")
    changes = []
    raw_slide_id = payload.get("slide_id")
    if raw_slide_id != slide_id:
        payload["slide_id"] = slide_id
        changes.append(
            {
                "json_path": "slide_id",
                "from": raw_slide_id,
                "to": slide_id,
                "reason": "orchestration-owned slide ID",
            }
        )
    expected_roles = {
        item["region_id"]: item["role"]
        for item in inventory
        if item["role"] in TEXT_ROLES
    }
    expected_visual_ids = {
        item["region_id"] for item in inventory if item["role"] == "visual"
    }
    raw_visuals = payload.get("visual_regions", [])
    for index, item in enumerate(raw_visuals):
        raw_region_id = item.get("region_id")
        if raw_region_id in expected_visual_ids or not isinstance(raw_region_id, str):
            continue
        matches = [
            region_id
            for region_id in expected_visual_ids
            if raw_region_id.endswith(region_id)
        ]
        if len(matches) == 1:
            item["region_id"] = matches[0]
            changes.append(
                {
                    "json_path": f"visual_regions.{index}.region_id",
                    "from": raw_region_id,
                    "to": matches[0],
                    "reason": "mapped a group-frame alias to its unique generation unit",
                }
            )
    filtered_visuals = [
        item for item in raw_visuals if item.get("region_id") in expected_visual_ids
    ]
    if filtered_visuals != raw_visuals:
        payload["visual_regions"] = filtered_visuals
        changes.append(
            {
                "json_path": "visual_regions",
                "dropped_region_ids": [
                    item.get("region_id")
                    for item in raw_visuals
                    if item.get("region_id") not in expected_visual_ids
                ],
                "reason": "removed summaries that are not generation units",
            }
        )
    raw_text = payload.get("text_regions", [])
    for index, item in enumerate(raw_text):
        raw_region_id = item.get("region_id")
        if raw_region_id in expected_roles or not isinstance(raw_region_id, str):
            continue
        matches = [
            region_id for region_id in expected_roles if raw_region_id.endswith(region_id)
        ]
        if len(matches) == 1:
            item["region_id"] = matches[0]
            changes.append(
                {
                    "json_path": f"text_regions.{index}.region_id",
                    "from": raw_region_id,
                    "to": matches[0],
                    "reason": "mapped a group-frame alias to its unique text region",
                }
            )
    filtered_text = [item for item in raw_text if item.get("region_id") in expected_roles]
    if filtered_text != raw_text:
        payload["text_regions"] = filtered_text
        changes.append(
            {
                "json_path": "text_regions",
                "dropped_region_ids": [
                    item.get("region_id")
                    for item in raw_text
                    if item.get("region_id") not in expected_roles
                ],
                "reason": "removed group or component summaries that are not text regions",
            }
        )
    for index, item in enumerate(payload.get("text_regions", [])):
        region_id = item.get("region_id")
        expected_role = expected_roles.get(region_id)
        if expected_role is not None and item.get("role") != expected_role:
            raw_role = item.get("role")
            item["role"] = expected_role
            changes.append(
                {
                    "json_path": f"text_regions.{index}.role",
                    "from": raw_role,
                    "to": expected_role,
                    "reason": "topology-owned region role",
                }
            )
    # Synthetic stage annotations are renderer-owned copy before commitment,
    # not optional model prose. Fill/restore them without another model call.
    for item in inventory:
        if not item.get("required_text"):
            continue
        existing = next((entry for entry in payload.get("text_regions", [])
                         if entry.get("region_id") == item["region_id"]), None)
        if existing is None:
            payload.setdefault("text_regions", []).append({"region_id": item["region_id"],
                "role": item["role"], "lines": [item["required_text"]]})
        else:
            existing["lines"] = [item["required_text"]]
        changes.append({"json_path": f"text_regions.{item['region_id']}",
                        "reason": "restored semantic stage annotation before copy commitment"})
    stage_labels = {item["region_id"] for item in inventory
                    if item.get("role") == "label" and item.get("semantic_block_role") == "semantic_stage"
                    and item.get("semantic_connections")}
    for entry in payload.get("text_regions", []):
        if entry.get("region_id") in stage_labels:
            entry["lines"] = [re.sub(r"^\s*\d+[.)]\s+", "", line) for line in entry["lines"]]
    return RegionContentPlan.model_validate(payload), {
        "changed": bool(changes),
        "changes": changes,
    }


def mock_region_content_plan(
    slide_spec: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> RegionContentPlan:
    visuals = [
        VisualRegionBrief(
            region_id=item["region_id"],
            semantic_brief=(
                f"{slide_spec.get('visual_intent', slide_spec['message'])}. "
                f"Connect it to: {slide_spec['message']}"
            ),
        )
        for item in inventory
        if item["role"] == "visual"
    ]
    non_title_regions = [
        item
        for item in inventory
        if item["role"] in TEXT_ROLES and item["role"] != "title"
    ]
    message_words = slide_spec["message"].split()
    chunk_size = max(1, math.ceil(len(message_words) / max(1, len(non_title_regions))))
    message_chunks = [
        " ".join(message_words[index : index + chunk_size])
        for index in range(0, len(message_words), chunk_size)
    ]
    text_regions = []
    non_title_index = 0
    for item in inventory:
        if item["role"] not in TEXT_ROLES:
            continue
        if item.get("required_text"):
            source = item["required_text"]
        elif item["role"] == "title":
            source = slide_spec["title"]
        else:
            source = message_chunks[min(non_title_index, len(message_chunks) - 1)]
            non_title_index += 1
            source = _mock_capacity_copy(
                source,
                role=item["role"],
                capacity_hint=item.get("capacity_hint", "compact"),
            )
        text_regions.append(
            TextRegionContent(region_id=item["region_id"], role=item["role"], lines=[source])
        )
    return RegionContentPlan(
        slide_id=slide_spec["slide_id"],
        visual_regions=visuals,
        text_regions=text_regions,
    )


def _mock_capacity_copy(text: str, *, role: str, capacity_hint: str) -> str:
    limits = {
        "caption": {"compact": 5, "standard": 7, "roomy": 10},
        "label": {"compact": 4, "standard": 6, "roomy": 8},
        "internal_label": {"compact": 4, "standard": 6, "roomy": 8},
        "supporting_text": {"compact": 2, "standard": 8, "roomy": 14},
        "standalone_text": {"compact": 2, "standard": 8, "roomy": 14},
    }
    words = text.split()
    if role in MAJOR_EXPLANATORY_ROLES and capacity_hint == "compact":
        words = [
            word
            for word in words
            if word.casefold().strip(" ,;:-") not in {"a", "an", "the", "by", "from"}
        ] or words
    limit = limits.get(role, limits["caption"]).get(capacity_hint, 7)
    compact = " ".join(words[:limit]).strip(" ,;:-")
    return re.sub(r"\s+", " ", compact) or "Key point"


def describe_position(
    bbox: list[int], canvas_width: float = 1536, canvas_height: float = 864
) -> str:
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2
    horizontal = (
        "left"
        if center_x < canvas_width / 3
        else "right"
        if center_x > 2 * canvas_width / 3
        else "center"
    )
    vertical = (
        "upper"
        if center_y < canvas_height / 3
        else "lower"
        if center_y > 2 * canvas_height / 3
        else "middle"
    )
    return f"{vertical}-{horizontal}"
