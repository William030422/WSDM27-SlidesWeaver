from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from v9_deck_planning.apimart import (
    ApiMartClient,
    _extract_text_response,
    _extract_usage,
    image_to_data_uri,
)
from v9_deck_planning.json_utils import extract_json, read_json, write_json, write_text


GRID = (64, 36)
CANVAS = (1536, 864)
CELL = 24
TEXT_SOURCE_ROLES = {"standalone_text", "label", "caption"}
TEXT_MODALITY_ROLES = TEXT_SOURCE_ROLES | {"title"}
ALLOWED_ROLES = {
    "title",
    "body_text",
    "semantic_visual",
    "semantic_stage",
    "decorative",
}
ALLOWED_RELATIONS = {
    "sequence_next",
    "causes",
    "contrasts",
    "supports",
    "loops_to",
}


SEMANTIC_TOPOLOGY_PROMPT = """You are calibrating fine detected regions into a MID-LEVEL semantic topology.

Inputs:
1. The original diffusion slide.
2. The previous fine-region overlay with source IDs.
3. The previous extraction JSON on a 64x36 grid.

The goal is neither object-level fragmentation nor one giant slide-level block.
Return independently meaningful stages/scenes plus explicit connections between
them.

Granularity rules:
1. Keep the true title and major explanatory text as standalone text blocks.
2. Absorb small pseudo-text, numbers, badges, captions, icons, and local props
   into the stage/scene they explain. Never return those as standalone text.
3. Merge a character with nearby local cues, objects, sun, or props when they
   express one local idea.
4. Do NOT merge all nodes merely because arrows or one path connect them.
   Connectors usually mean separate semantic blocks plus a connection.
5. Split a construction when it has distinct local centers, successive stages,
   repeated snapshots, conceptual poles, or visibly separable scenes.
6. A three-part progression should normally remain three content blocks. A
   two-pole explanation joined by an arrow/path should normally remain two
   blocks plus a connection. A trajectory of repeated episode snapshots should
   retain several meaningful stage blocks rather than one giant trajectory.
7. A block should be one independently meaningful content unit. Text-only
   stages are valid and must not imply a new visual region.
8. Use `semantic_stage` for a node in a progression and `semantic_visual` for an
   independent scene/concept. Connections carry sequence, causality, contrast,
   support, or loop structure.
9. Each source region ID must appear exactly once in one block or in
   `ignored_region_ids`. Source IDs are provenance; several IDs may merge into
   one block.
10. bbox is [x,y,width,height] on the 64x36 grid. It may expand to cover the
    complete local semantic stage, but should not swallow neighboring stages.
11. Do not transcribe visible wording. Keep `semantic_role` short.
12. Prefer 2-5 content blocks below the header when the visual structure
    supports them. Do not force a fixed number against the image.

Return one compact JSON object with no prose:

{
  "slide_id": "SLIDE_ID",
  "blocks": [
    {
      "block_id": "b_title",
      "role": "title | body_text | semantic_visual | semantic_stage | decorative",
      "semantic_role": "short structural role",
      "source_region_ids": ["source_region_id"],
      "bbox": [0,0,1,1],
      "confidence": 0.0
    }
  ],
  "connections": [
    {
      "source_block_id": "block_id",
      "target_block_id": "block_id",
      "relation": "sequence_next | causes | contrasts | supports | loops_to"
    }
  ],
  "ignored_region_ids": [],
  "reading_order": ["block_id"]
}
"""


def extract_semantic_topology(
    *,
    slide_id: str,
    skeleton_path: Path,
    identity_path: Path,
    identity_overlay_path: Path,
    output_dir: Path,
    client: ApiMartClient | None,
    model: str,
    max_tokens: int,
    reuse_raw: bool,
    offline: bool,
) -> dict[str, Any]:
    slide_dir = output_dir / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = slide_dir / "semantic_topology.json"
    manifest_path = slide_dir / "run_manifest.json"
    import hashlib
    digest = hashlib.sha256(skeleton_path.read_bytes() + identity_path.read_bytes())
    digest.update((model + str(max_tokens) + str(offline) + SEMANTIC_TOPOLOGY_PROMPT).encode("utf-8"))
    input_fingerprint = digest.hexdigest()
    if semantic_path.is_file() and manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("status") == "succeeded" and previous.get("input_fingerprint") == input_fingerprint:
            return {**previous, "resumed": True, "model_call_count": 0}

    identity = read_json(identity_path)
    source_ids = {item["element_id"] for item in identity["elements"]}
    source_roles = {
        item["element_id"]: item["role"] for item in identity["elements"]
    }
    prompt = SEMANTIC_TOPOLOGY_PROMPT.replace("SLIDE_ID", slide_id)
    prompt_path = slide_dir / "prompt.txt"
    raw_path = slide_dir / "vlm_raw_response.json"
    normalization_path = slide_dir / "normalization_report.json"
    overlay_path = slide_dir / "semantic_topology_overlay.png"
    write_text(prompt_path, prompt)

    model_call_count = 0
    usage = None
    warning = None
    raw = None
    try:
        if offline:
            semantic = fallback_semantic_topology(identity, slide_id)
            normalization = {
                "changed": True,
                "validation_errors": [],
                "fallbacks_applied": [{"reason": "offline semantic topology"}],
            }
        else:
            if reuse_raw and raw_path.is_file():
                raw = raw_path.read_text(encoding="utf-8")
            else:
                if client is None:
                    raise RuntimeError("semantic topology extraction requires an API client")
                from .api_journal import journal_response
                response = journal_response(path=slide_dir / "api_response.json",
                    request={"method": "semantic_vlm", "api_root": client.api_root, "model": model,
                             "prompt": prompt, "identity": identity, "max_tokens": max_tokens},
                    image_paths=[skeleton_path, identity_overlay_path],
                    invoke=lambda: _call_semantic_vlm(client=client, model=model, prompt=prompt,
                        skeleton_path=skeleton_path, identity_overlay_path=identity_overlay_path,
                        identity=identity, max_tokens=max_tokens))
                raw, usage = response["raw"], response["usage"]
                model_call_count = int(not response["reused"])
                write_text(raw_path, raw)
            semantic = normalize_semantic_topology(
                extract_json(raw),
                slide_id=slide_id,
                source_ids=source_ids,
                source_roles=source_roles,
            )
            normalization = {
                "changed": False,
                "validation_errors": [],
                "fallbacks_applied": [],
            }
    except Exception as exc:
        warning = str(exc)
        semantic = fallback_semantic_topology(identity, slide_id)
        normalization = {
            "changed": True,
            "validation_errors": [warning],
            "fallbacks_applied": [{"reason": "semantic topology validation fallback"}],
        }

    write_json(semantic_path, semantic)
    write_json(normalization_path, normalization)
    render_semantic_overlay(semantic, skeleton_path, overlay_path)
    content_blocks = [
        item
        for item in semantic["blocks"]
        if item["role"] not in {"title", "body_text", "decorative"}
    ]
    result = {
        "input_fingerprint": input_fingerprint,
        "slide_id": slide_id,
        "status": "succeeded",
        "quality_status": "fallback" if warning or offline else "strict_pass",
        "warning": warning,
        "model": model,
        "model_call_count": model_call_count,
        "usage": usage,
        "source_region_count": len(source_ids),
        "semantic_block_count": len(semantic["blocks"]),
        "content_block_count": len(content_blocks),
        "connection_count": len(semantic["connections"]),
        "semantic_topology": str(semantic_path.resolve()),
        "semantic_overlay": str(overlay_path.resolve()),
        "prompt": str(prompt_path.resolve()),
        "raw_response": str(raw_path.resolve()) if raw_path.is_file() else None,
        "normalization_report": str(normalization_path.resolve()),
    }
    write_json(manifest_path, result)
    return result


def _call_semantic_vlm(
    *,
    client: ApiMartClient,
    model: str,
    prompt: str,
    skeleton_path: Path,
    identity_overlay_path: Path,
    identity: dict[str, Any],
    max_tokens: int,
) -> tuple[str, dict[str, Any] | None]:
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "Input 1: original diffusion skeleton."},
        {"type": "input_image", "image_url": image_to_data_uri(skeleton_path)},
        {"type": "input_text", "text": "Input 2: previous extraction overlay."},
        {"type": "input_image", "image_url": image_to_data_uri(identity_overlay_path)},
        {"type": "input_text", "text": "Input 3: previous extraction JSON."},
        {"type": "input_text", "text": json.dumps(identity, ensure_ascii=False, indent=2)},
    ]
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Return one valid JSON object with no markdown fences or prose.",
                    }
                ],
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }
    client.call_count += 1
    response = client._post_json("/responses", payload)
    raw = _extract_text_response(response)
    if not raw.strip():
        raise RuntimeError("semantic topology extraction returned empty content")
    return raw, _extract_usage(response)


def normalize_semantic_topology(
    payload: Any,
    *,
    slide_id: str,
    source_ids: set[str],
    source_roles: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("semantic topology response is not an object")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) < 2:
        raise ValueError("semantic topology must contain at least two blocks")
    blocks = []
    block_ids: set[str] = set()
    covered: list[str] = []
    for index, raw in enumerate(raw_blocks, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"block {index} is not an object")
        block_id = str(raw.get("block_id") or f"b_{index:03d}").strip().lower()
        if not block_id.startswith("b"):
            block_id = f"b_{block_id}"
        if block_id in block_ids:
            raise ValueError(f"duplicate block ID: {block_id}")
        block_ids.add(block_id)
        role = str(raw.get("role", "semantic_stage")).strip()
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported semantic role: {role}")
        sources = [str(value) for value in raw.get("source_region_ids", [])]
        if not sources:
            raise ValueError(f"block {block_id} has no source regions")
        covered.extend(sources)
        blocks.append(
            {
                "block_id": block_id,
                "role": role,
                "semantic_role": str(raw.get("semantic_role", "semantic block")).strip(),
                "source_region_ids": sources,
                "modality": block_modality(sources, source_roles),
                "bbox": normalize_bbox(raw.get("bbox")),
                "confidence": min(1.0, max(0.0, float(raw.get("confidence", 0.5)))),
            }
        )
    ignored = [str(value) for value in payload.get("ignored_region_ids", [])]
    accounted = covered + ignored
    if len(accounted) != len(set(accounted)):
        raise ValueError("a source region ID was accounted for more than once")
    if set(accounted) != source_ids:
        raise ValueError(
            f"source coverage mismatch: missing={sorted(source_ids - set(accounted))}, "
            f"unknown={sorted(set(accounted) - source_ids)}"
        )

    title_blocks = [item for item in blocks if item["role"] == "title"]
    if not title_blocks:
        min(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))["role"] = "title"
    elif len(title_blocks) > 1:
        keep = min(title_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        for item in title_blocks:
            if item is not keep:
                item["role"] = "body_text"

    connections = []
    for raw in payload.get("connections", []):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_block_id", ""))
        target = str(raw.get("target_block_id", ""))
        relation = str(raw.get("relation", "supports"))
        if source in block_ids and target in block_ids and relation in ALLOWED_RELATIONS:
            connections.append(
                {
                    "source_block_id": source,
                    "target_block_id": target,
                    "relation": relation,
                }
            )
    reading_order = [
        str(value) for value in payload.get("reading_order", []) if value in block_ids
    ]
    if set(reading_order) != block_ids:
        reading_order = [
            item["block_id"]
            for item in sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        ]
    return {
        "slide_id": slide_id,
        "blocks": blocks,
        "connections": connections,
        "ignored_region_ids": ignored,
        "reading_order": reading_order,
    }


def fallback_semantic_topology(identity: dict[str, Any], slide_id: str) -> dict[str, Any]:
    elements = sorted(identity["elements"], key=lambda item: (item["bbox"][1], item["bbox"][0]))
    source_roles = {item["element_id"]: item["role"] for item in elements}
    title = next((item for item in elements if item["role"] == "title"), elements[0])
    used = {title["element_id"]}
    blocks = [
        {
            "block_id": "b_title",
            "role": "title",
            "semantic_role": "slide title",
            "source_region_ids": [title["element_id"]],
            "bbox": list(title["bbox"]),
            "confidence": title.get("confidence", 0.5),
        }
    ]
    visual_top = min(
        (item["bbox"][1] for item in elements if item["role"] == "visual"),
        default=GRID[1],
    )
    intro = [
        item
        for item in elements
        if item["element_id"] not in used
        and item["role"] in TEXT_SOURCE_ROLES
        and item["bbox"][1] < visual_top
    ]
    if intro:
        used.update(item["element_id"] for item in intro)
        blocks.append(
            {
                "block_id": "b_intro",
                "role": "body_text",
                "semantic_role": "introductory explanation",
                "source_region_ids": [item["element_id"] for item in intro],
                "bbox": union_bbox([item["bbox"] for item in intro]),
                "confidence": 0.25,
            }
        )
    remaining = [item for item in elements if item["element_id"] not in used]
    group_count = min(5, max(1, len([item for item in remaining if item["role"] == "visual"])))
    ordered = sorted(remaining, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    groups = [[] for _ in range(group_count)]
    for index, item in enumerate(ordered):
        groups[min(group_count - 1, index * group_count // max(1, len(ordered)))].append(item)
    for index, group in enumerate((item for item in groups if item), start=1):
        blocks.append(
            {
                "block_id": f"b_stage_{index:02d}",
                "role": "semantic_stage" if len(groups) > 1 else "semantic_visual",
                "semantic_role": f"content stage {index}",
                "source_region_ids": [item["element_id"] for item in group],
                "bbox": union_bbox([item["bbox"] for item in group]),
                "confidence": 0.2,
            }
        )
    content_ids = [item["block_id"] for item in blocks if item["role"].startswith("semantic_")]
    connections = [
        {
            "source_block_id": source,
            "target_block_id": target,
            "relation": "sequence_next",
        }
        for source, target in zip(content_ids, content_ids[1:])
    ]
    for block in blocks:
        block["modality"] = block_modality(block["source_region_ids"], source_roles)
    return {
        "slide_id": slide_id,
        "blocks": blocks,
        "connections": connections,
        "ignored_region_ids": [],
        "reading_order": [item["block_id"] for item in blocks],
    }


def block_modality(
    source_region_ids: list[str], source_roles: dict[str, str]
) -> str:
    has_visual = any(source_roles.get(region_id) == "visual" for region_id in source_region_ids)
    has_text = any(
        source_roles.get(region_id) in TEXT_MODALITY_ROLES
        for region_id in source_region_ids
    )
    if has_visual and has_text:
        return "mixed"
    if has_visual:
        return "visual_only"
    return "text_only"


def regularize_semantic_topology(
    *, semantic: dict[str, Any], identity: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    slide_id = semantic["slide_id"]
    source_elements = {item["element_id"]: item for item in identity["elements"]}
    source_roles = {
        region_id: item["role"] for region_id, item in source_elements.items()
    }
    blocks = [
        {
            **item,
            "modality": block_modality(item["source_region_ids"], source_roles),
        }
        for item in semantic["blocks"]
    ]
    blocks_by_id = {item["block_id"]: item for item in blocks}
    title = next(item for item in blocks if item["role"] == "title")
    non_header_blocks = [
        item for item in blocks if item["role"] not in {"title", "decorative"}
    ]
    body = next(
        (
            item
            for item in non_header_blocks
            if item["role"] == "body_text" and len(non_header_blocks) > 1
        ),
        None,
    )
    content_blocks = [
        item
        for item in non_header_blocks
        if item is not body
    ]
    if not content_blocks:
        raise ValueError(f"{slide_id} has no semantic content blocks")

    def source_bbox(block: dict[str, Any]) -> list[int]:
        return union_bbox(
            [source_elements[region_id]["bbox"] for region_id in block["source_region_ids"]]
        )

    title_height = min(6, max(3, title["bbox"][3]))
    title_bbox = [3, 2, 58, title_height]
    body_bbox = None
    if body is not None:
        body_height = min(6, max(3, body["bbox"][3]))
        body_bbox = [3, title_bbox[1] + title_bbox[3] + 1, 58, body_height]

    fallback_order = [
        block_id
        for block_id in semantic["reading_order"]
        if block_id in {item["block_id"] for item in content_blocks}
    ]
    content_order = topological_content_order(
        [item["block_id"] for item in content_blocks],
        semantic["connections"],
        fallback_order,
    )
    ordered_content = [blocks_by_id[block_id] for block_id in content_order]
    layout_family = infer_layout_family(
        ordered_content, source_elements=source_elements
    )

    entities = []
    components = []
    header_components = []
    content_components = []
    regularized_blocks = []

    def add_header(block: dict[str, Any], bbox: list[int], canonical_role: str) -> None:
        block_id = block["block_id"]
        cid = component_id(block_id)
        entities.append(
            {
                "region_id": block_id,
                "canonical_role": canonical_role,
                "parent_component_id": cid,
                "calibrated_bbox": bbox,
            }
        )
        components.append(
            {
                "component_id": cid,
                "component_role": "title" if canonical_role == "title" else "supporting_text",
                "source_region_ids": [block_id],
            }
        )
        header_components.append(cid)
        regularized_blocks.append(
            {
                **block,
                "container_bbox": bbox,
                "visual_bbox": None,
                "text_slots": [
                    {
                        "position": "standalone",
                        "bbox": bbox,
                        "source_region_ids": block["source_region_ids"],
                    }
                ],
            }
        )

    add_header(title, title_bbox, "title")
    if body is not None and body_bbox is not None:
        add_header(body, body_bbox, "supporting_text")

    for block_id in content_order:
        block = blocks_by_id[block_id]
        container = source_bbox(block)
        modality = block["modality"]
        cid = component_id(block_id)
        if modality == "text_only":
            text_sources = [
                source_elements[region_id]
                for region_id in block["source_region_ids"]
                if source_elements[region_id]["role"] in TEXT_MODALITY_ROLES
            ]
            text_bbox = union_bbox(
                [item["bbox"] for item in text_sources]
                or [source_elements[region_id]["bbox"] for region_id in block["source_region_ids"]]
            )
            canonical_role = (
                "standalone_text"
                if any(item["role"] == "standalone_text" for item in text_sources)
                else "supporting_text"
            )
            entities.append(
                {
                    "region_id": block_id,
                    "canonical_role": canonical_role,
                    "parent_component_id": cid,
                    "calibrated_bbox": text_bbox,
                }
            )
            components.append(
                {
                    "component_id": cid,
                    "component_role": "standalone_text",
                    "source_region_ids": [block_id],
                }
            )
            content_components.append(cid)
            regularized_blocks.append(
                {
                    **block,
                    "container_bbox": container,
                    "visual_bbox": None,
                    "text_slots": [
                        {
                            "position": "standalone",
                            "bbox": text_bbox,
                            "source_region_ids": block["source_region_ids"],
                        }
                    ],
                }
            )
            continue

        zones = classify_text_sources(block=block, source_elements=source_elements)
        visual_source_ids = [
            region_id
            for region_id in block["source_region_ids"]
            if source_elements[region_id]["role"] == "visual"
        ]
        if not visual_source_ids:
            raise ValueError(f"{block_id} cannot generate a visual without visual provenance")
        x, y, width, height = container
        # A required stage in a progression needs an editable name even when
        # diffusion happened to draw only a picture for that stage. Reserve the
        # label inside this existing component before its geometry is frozen.
        semantic_label = str(block.get("semantic_role", "")).strip()
        stage_ids = {node["block_id"] for node in semantic["blocks"]
                     if node.get("role") == "semantic_stage"}
        progression = [edge for edge in semantic.get("connections", [])
                       if edge.get("relation") in {"sequence_next", "causes", "loops_to"}
                       and edge["source_block_id"] in stage_ids
                       and edge["target_block_id"] in stage_ids]
        connected = (len(stage_ids) >= 3 and len(progression) >= 2
                     and any(block_id in (edge["source_block_id"], edge["target_block_id"])
                             for edge in progression))
        annotation = (
            modality == "visual_only" and block.get("role") == "semantic_stage"
            and connected and semantic_label
            and not re.fullmatch(r"(?:(?:content|first|second|third)\s+)?stage(?:\s+\d+)?", semantic_label, re.I)
        )
        if annotation and height >= 6:
            zones["top"] = [{
                "source_region_id": visual_source_ids[0],
                "source_role": "semantic_annotation",
                "source_bbox": [block["bbox"][0], block["bbox"][1], block["bbox"][2], 3],
                "relative_center_y": 0.0,
            }]
        top_height = (
            desired_slot_height(
                source_boxes=[item["source_bbox"] for item in zones["top"]],
                source_block_bbox=block["bbox"],
                target_height=height,
            )
            if zones["top"]
            else None
        )
        if annotation and zones["top"]:
            top_height = min(4 if width < 14 else 3, max(2, height // 3))
        bottom_height = (
            desired_slot_height(
                source_boxes=[item["source_bbox"] for item in zones["bottom"]],
                source_block_bbox=block["bbox"],
                target_height=height,
            )
            if zones["bottom"]
            else None
        )
        top_height, bottom_height = fit_slot_heights(
            target_height=height,
            top_height=top_height,
            bottom_height=bottom_height,
        )
        active_slots = sum(value is not None for value in (top_height, bottom_height))
        # Small observed blocks still need positive integer text and visual
        # regions. Expand only this soft scaffold when it cannot contain them.
        if height < active_slots + 1:
            height = active_slots + 1
            y = min(y, GRID[1] - height)
            container = [x, y, width, height]
        gap = 1 if height - sum(value or 0 for value in (top_height, bottom_height)) > active_slots else 0
        slots_by_position = {
            "top": [x, y, width, top_height] if top_height is not None else None,
            "bottom": (
                [x, y + height - bottom_height, width, bottom_height]
                if bottom_height is not None
                else None
            ),
        }
        visual_top = y + (top_height + gap if top_height is not None else 0)
        visual_bottom = y + height - (
            bottom_height + gap if bottom_height is not None else 0
        )
        visual_bbox = [x, visual_top, width, max(1, visual_bottom - visual_top)]
        visual_id = f"{block_id}_visual"
        component_entities = [visual_id]
        entities.append(
            {
                "region_id": visual_id,
                "canonical_role": "visual",
                "parent_component_id": cid,
                "calibrated_bbox": visual_bbox,
            }
        )
        slots = []
        for position, role in (("top", "label"), ("bottom", "caption")):
            slot = slots_by_position[position]
            if slot is None:
                continue
            text_id = f"{block_id}_text_{position}"
            component_entities.append(text_id)
            entities.append(
                {
                    "region_id": text_id,
                    "canonical_role": role,
                    "parent_component_id": cid,
                    "calibrated_bbox": slot,
                }
            )
            slots.append(
                {
                    "position": position,
                    "bbox": slot,
                    "source_region_ids": [
                        item["source_region_id"] for item in zones[position]
                    ],
                    "source_observations": zones[position],
                }
            )
        components.append(
            {
                "component_id": cid,
                "component_role": "mixed" if modality == "mixed" else "primary_visual",
                "source_region_ids": component_entities,
            }
        )
        content_components.append(cid)
        regularized_blocks.append(
            {
                **block,
                "container_bbox": container,
                "visual_bbox": visual_bbox,
                "text_slots": slots,
                **({"semantic_annotation": {"text": semantic_label,
                     "source": "semantic_stage", "position": "top"}} if annotation else {}),
            }
        )

    groups = [
        {
            "group_id": "g_root",
            "role": "slide_layout",
            "scope": "top_level",
            "arrangement": "column",
            "children": ["g_header", "g_content"],
            "peer_relation": "none",
            "calibration_policy": {
                "size_policy": "preserve",
                "alignment": "preserve",
                "spacing": "preserve",
            },
        },
        {
            "group_id": "g_header",
            "role": "header",
            "scope": "nested",
            "arrangement": "column",
            "children": header_components,
            "peer_relation": "none",
            "calibration_policy": {
                "size_policy": "preserve",
                "alignment": "preserve",
                "spacing": "preserve",
            },
        },
        {
            "group_id": "g_content",
            "role": "semantic_content",
            "scope": "nested",
            "arrangement": layout_family,
            "children": content_components,
            "peer_relation": "sequential" if semantic["connections"] else "none",
            "calibration_policy": {
                "size_policy": "equal_size" if layout_family == "row" else "preserve",
                "alignment": "top" if layout_family == "row" else "preserve",
                "spacing": "even" if layout_family == "row" else "preserve",
            },
        },
    ]
    entities.sort(key=lambda item: (item["calibrated_bbox"][1], item["calibrated_bbox"][0]))
    compiled = {
        "slide_id": slide_id,
        "groups": groups,
        "components": components,
        "entities": entities,
        "reading_order": [item["region_id"] for item in entities],
    }
    content_regularized = [
        item
        for item in regularized_blocks
        if item["block_id"] in set(content_order)
    ]
    report = {
        "slide_id": slide_id,
        "layout_family": layout_family,
        "content_order": content_order,
        "content_block_count": len(content_regularized),
        "target_gap_grid": None,
        "overlap_count": sum(
            _bbox_intersection_area(left["calibrated_bbox"], right["calibrated_bbox"]) > 0
            for index, left in enumerate(entities)
            for right in entities[index + 1:]
        ),
        "top_text_slot_count": sum(
            any(slot["position"] == "top" for slot in item["text_slots"])
            for item in content_regularized
        ),
        "bottom_text_slot_count": sum(
            any(slot["position"] == "bottom" for slot in item["text_slots"])
            for item in content_regularized
        ),
        "dual_text_slot_block_count": sum(
            {slot["position"] for slot in item["text_slots"]} == {"top", "bottom"}
            for item in content_regularized
        ),
        "blocks": regularized_blocks,
        "connections": semantic["connections"],
    }
    return compiled, report


def classify_text_sources(
    *, block: dict[str, Any], source_elements: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    block_y = block["bbox"][1]
    block_height = block["bbox"][3]
    zones: dict[str, list[dict[str, Any]]] = {"top": [], "bottom": []}
    for source_id in block["source_region_ids"]:
        source = source_elements[source_id]
        if source["role"] not in TEXT_SOURCE_ROLES:
            continue
        x, y, width, height = source["bbox"]
        relative_center_y = (y + height / 2 - block_y) / max(1, block_height)
        position = "top" if relative_center_y <= 0.5 else "bottom"
        zones[position].append(
            {
                "source_region_id": source_id,
                "source_role": source["role"],
                "source_bbox": [x, y, width, height],
                "relative_center_y": round(relative_center_y, 4),
            }
        )
    return zones


def infer_layout_family(
    blocks: list[dict[str, Any]],
    *,
    source_elements: dict[str, dict[str, Any]],
) -> str:
    if len(blocks) < 2 or any(block["modality"] == "text_only" for block in blocks):
        return "freeform"
    boxes = [
        union_bbox(
            [source_elements[source_id]["bbox"] for source_id in block["source_region_ids"]]
        )
        for block in blocks
    ]
    if any(
        left[0] + left[2] > right[0]
        for left, right in zip(boxes, boxes[1:])
    ):
        return "freeform"
    overlap = min(box[1] + box[3] for box in boxes) - max(box[1] for box in boxes)
    if overlap <= 0 or overlap / min(box[3] for box in boxes) < 0.5:
        return "freeform"
    return "row"


def desired_slot_height(
    *, source_boxes: list[list[int]], source_block_bbox: list[int], target_height: int
) -> int:
    relative_height = union_bbox(source_boxes)[3] / max(1, source_block_bbox[3])
    return min(6, max(4, math.ceil(target_height * relative_height * 1.5)))


def fit_slot_heights(
    *, target_height: int, top_height: int | None, bottom_height: int | None
) -> tuple[int | None, int | None]:
    active = sum(value is not None for value in (top_height, bottom_height))
    if not active:
        return None, None
    target_height = max(target_height, active + 1)
    desired_available = target_height - max(5, round(target_height * 0.4)) - active
    # Preserve the usual three-cell text minimum only where the container can
    # actually support it; reserve at least one cell for the visual throughout.
    available = min(target_height - 1, max(active * 3, desired_available))
    minimum = 3 if available >= active * 3 else 1
    heights = [max(minimum, value) for value in (top_height, bottom_height) if value is not None]
    while sum(heights) > available:
        index = max(range(len(heights)), key=lambda item: heights[item])
        if heights[index] <= minimum:
            break
        heights[index] -= 1
    values = iter(heights)
    return (
        next(values) if top_height is not None else None,
        next(values) if bottom_height is not None else None,
    )


def _bbox_intersection_area(left: list[int], right: list[int]) -> float:
    return max(0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0])) * max(
        0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1])
    )


def component_id(block_id: str) -> str:
    suffix = block_id.removeprefix("b_").removeprefix("b") or "block"
    return f"c_{suffix}"


def topological_content_order(
    content_ids: list[str], connections: list[dict[str, Any]], fallback: list[str]
) -> list[str]:
    content = set(content_ids)
    edges = [
        (item["source_block_id"], item["target_block_id"])
        for item in connections
        if item["source_block_id"] in content
        and item["target_block_id"] in content
        and item["relation"] in {"sequence_next", "causes", "supports"}
    ]
    incoming = {block_id: 0 for block_id in content_ids}
    outgoing = {block_id: [] for block_id in content_ids}
    for source, target in edges:
        if target not in outgoing[source]:
            outgoing[source].append(target)
            incoming[target] += 1
    queue = [item for item in fallback if item in content and incoming[item] == 0]
    ordered = []
    while queue:
        block_id = queue.pop(0)
        if block_id in ordered:
            continue
        ordered.append(block_id)
        for target in outgoing[block_id]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return ordered if len(ordered) == len(content_ids) else [item for item in fallback if item in content]


def union_bbox(boxes: list[list[int]]) -> list[int]:
    left = min(item[0] for item in boxes)
    top = min(item[1] for item in boxes)
    right = max(item[0] + item[2] for item in boxes)
    bottom = max(item[1] + item[3] for item in boxes)
    return [left, top, right - left, bottom - top]


def normalize_bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"invalid bbox: {value!r}")
    x, y, width, height = [round(float(item)) for item in value]
    x = min(max(x, 0), GRID[0] - 1)
    y = min(max(y, 0), GRID[1] - 1)
    width = min(max(width, 1), GRID[0] - x)
    height = min(max(height, 1), GRID[1] - y)
    return [x, y, width, height]


def render_semantic_overlay(
    semantic: dict[str, Any], skeleton_path: Path, output_path: Path
) -> None:
    with Image.open(skeleton_path) as source:
        image = source.convert("RGBA").resize(CANVAS)
    layer = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    font = ImageFont.load_default()
    colors = {
        "title": (31, 91, 185, 230),
        "body_text": (218, 119, 30, 230),
        "semantic_visual": (20, 143, 74, 230),
        "semantic_stage": (156, 50, 175, 230),
        "decorative": (100, 105, 115, 230),
    }
    for block in semantic["blocks"]:
        x, y, width, height = block["bbox"]
        box = (x * CELL, y * CELL, (x + width) * CELL - 1, (y + height) * CELL - 1)
        color = colors[block["role"]]
        draw.rectangle(box, fill=(*color[:3], 35), outline=color, width=5)
        draw.text((box[0] + 5, box[1] + 5), f"{block['block_id']} | {block['semantic_role']}", fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, layer).convert("RGB").save(output_path)
