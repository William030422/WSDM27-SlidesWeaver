from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
DEFAULT_OUTPUT_DIR = ROOT / "gpt_5_6_terra_layout_identity_v2"
MODEL = "gpt-5.6-terra"
MAX_TOKENS = 4096
GRID = (64, 36)
CANVAS = (1536, 864)
CELL = 24

sys.path.insert(0, str(V9_ROOT / "src"))

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import extract_json as extract_model_object
from v9_deck_planning.json_utils import read_json, write_json, write_text


GridBBox = tuple[int, int, int, int]
RegionRole = Literal["title", "standalone_text", "caption", "label", "visual"]
ROLE_ALIASES = {
    "body": "standalone_text",
    "body_text": "standalone_text",
    "subtitle": "standalone_text",
    "supporting_text": "standalone_text",
    "text": "standalone_text",
    "image": "visual",
    "illustration": "visual",
    "photo": "visual",
    "diagram": "visual",
    "chart": "visual",
}


class RegionElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    role: RegionRole
    bbox: GridBBox
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_geometry(self) -> "RegionElement":
        validate_bbox(self.bbox, self.element_id)
        return self


class RegionDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slide_id: str = Field(pattern=r"^slide_[0-9]{3}$")
    elements: list[RegionElement] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_elements(self) -> "RegionDetectionResult":
        ids = [item.element_id for item in self.elements]
        if len(ids) != len(set(ids)):
            raise ValueError("element IDs must be unique")
        if not any(item.role == "title" for item in self.elements):
            raise ValueError("at least one title region is required")
        if not any(item.role != "title" for item in self.elements):
            raise ValueError("at least one non-title region is required")
        return self


def validate_bbox(bbox: GridBBox, label: str) -> None:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has a non-positive bbox")
    if x < 0 or y < 0 or x + width > GRID[0] or y + height > GRID[1]:
        raise ValueError(f"{label} is outside the 64x36 canvas")


def kind_for_role(role: str) -> str:
    if role == "title":
        return "title"
    if role == "visual":
        return "visual"
    return "text"


def extraction_prompt(slide_id: str) -> str:
    return f"""Detect only visible layout regions on this presentation slide.

Slide ID: {slide_id}
Coordinate system: a 64 by 36 integer grid over the full 16:9 image.

Rules:
1. Return every visible title, standalone text block, caption or label, and visual once.
2. bbox is [x, y, width, height] with positive integers fully inside 64x36.
3. role must be exactly one of: title, standalone_text, caption, label, visual.
4. Use spatial appearance only. Do not OCR, transcribe, or infer readable wording.
5. Ignore arrows and connector lines as regions.
6. Return JSON only. Do not add metadata, hierarchy, relations, repeated patterns, notes, or derived fields.

{{
  "slide_id": "{slide_id}",
  "elements": [
    {{
      "element_id": "title_001",
      "role": "title",
      "bbox": [4, 2, 56, 5],
      "confidence": 0.98
    }}
  ]
}}
"""


def normalize_payload(
    payload: Any,
    slide_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_slide_id = slide_id or (
        payload.get("slide_id") if isinstance(payload, dict) else None
    ) or "slide_000"
    changes: list[dict[str, Any]] = []
    unknown_values: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    fallbacks: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        payload = {}
        validation_errors.append("response was not a JSON object")

    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        raw_elements = []
        validation_errors.append("elements was missing or not an array")

    normalized_elements: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(raw_elements):
        if not isinstance(raw, dict):
            unknown_values.append({"json_path": f"elements.{index}", "value": raw})
            continue
        role = _normalize_role(raw.get("role"), raw)
        if role is None:
            unknown_values.append(
                {"json_path": f"elements.{index}.role", "value": raw.get("role")}
            )
            continue
        bbox = _normalize_bbox(raw.get("bbox"))
        if bbox is None:
            validation_errors.append(f"elements.{index}.bbox was invalid")
            continue
        element_id = _normalize_id(raw.get("element_id"), role, index + 1)
        element_id = _deduplicate_id(element_id, used_ids)
        used_ids.add(element_id)
        confidence = _normalize_confidence(raw.get("confidence"))
        normalized = {
            "element_id": element_id,
            "role": role,
            "bbox": bbox,
            "confidence": confidence,
        }
        normalized_elements.append(normalized)
        projected = {key: raw.get(key) for key in normalized}
        if projected != normalized or set(raw) != set(normalized):
            changes.append(
                {
                    "json_path": f"elements.{index}",
                    "reason": "projected model output to the consumed region schema",
                }
            )

    _ensure_minimum_regions(normalized_elements, fallbacks)
    normalized = {"slide_id": expected_slide_id, "elements": normalized_elements}
    dropped = sorted(set(payload) - {"slide_id", "elements"})
    if payload.get("slide_id") != expected_slide_id:
        changes.append(
            {
                "json_path": "slide_id",
                "from": payload.get("slide_id"),
                "to": expected_slide_id,
                "reason": "orchestration-owned slide ID",
            }
        )
    report = {
        "schema_version": "3.0",
        "changed": bool(changes or dropped or fallbacks or validation_errors or unknown_values),
        "changes": changes,
        "dropped_top_level_fields": dropped,
        "unknown_values": unknown_values,
        "validation_errors": validation_errors,
        "fallbacks_applied": fallbacks,
    }
    return normalized, report


def _normalize_role(value: Any, raw: dict[str, Any]) -> RegionRole | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = ROLE_ALIASES.get(normalized, normalized)
    if normalized in {"title", "standalone_text", "caption", "label", "visual"}:
        return normalized  # type: ignore[return-value]
    kind = str(raw.get("kind", "")).strip().lower()
    element_id = str(raw.get("element_id", "")).lower()
    if kind == "title" or element_id.startswith("title"):
        return "title"
    if kind == "visual" or element_id.startswith(("visual", "image", "illustration")):
        return "visual"
    if kind == "text":
        return "standalone_text"
    return None


def _normalize_bbox(value: Any) -> list[int] | None:
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


def _normalize_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _normalize_id(value: Any, role: str, index: int) -> str:
    candidate = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not candidate or not candidate[0].isalpha():
        candidate = f"{role}_{index:03d}"
    return candidate


def _deduplicate_id(candidate: str, used_ids: set[str]) -> str:
    if candidate not in used_ids:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def _ensure_minimum_regions(
    elements: list[dict[str, Any]],
    fallbacks: list[dict[str, Any]],
) -> None:
    if not elements:
        elements.extend(_default_elements())
        fallbacks.append({"reason": "no usable detected regions", "added": ["title_001", "visual_001"]})
        return
    if not any(item["role"] == "title" for item in elements):
        text_candidates = [item for item in elements if item["role"] != "visual"]
        if text_candidates:
            candidate = min(text_candidates, key=lambda item: (item["bbox"][1], item["bbox"][0]))
            previous = candidate["role"]
            candidate["role"] = "title"
            fallbacks.append(
                {"reason": "missing title region", "element_id": candidate["element_id"], "from": previous, "to": "title"}
            )
        else:
            title = _default_elements()[0]
            title["element_id"] = _deduplicate_id(title["element_id"], {item["element_id"] for item in elements})
            elements.insert(0, title)
            fallbacks.append({"reason": "missing title region", "added": title["element_id"]})
    if not any(item["role"] != "title" for item in elements):
        visual = _default_elements()[1]
        visual["element_id"] = _deduplicate_id(visual["element_id"], {item["element_id"] for item in elements})
        elements.append(visual)
        fallbacks.append({"reason": "missing non-title region", "added": visual["element_id"]})


def _default_elements() -> list[dict[str, Any]]:
    return [
        {"element_id": "title_001", "role": "title", "bbox": [3, 2, 58, 5], "confidence": 0.25},
        {"element_id": "visual_001", "role": "visual", "bbox": [3, 9, 58, 24], "confidence": 0.2},
    ]


def bbox_pixels(bbox: GridBBox) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (x * CELL, y * CELL, (x + width) * CELL - 1, (y + height) * CELL - 1)


def render_overlay(reference_path: Path, result: RegionDetectionResult, output_path: Path) -> None:
    with Image.open(reference_path) as reference:
        base = reference.convert("RGBA").resize(CANVAS)
    overlay = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    label_font = ImageFont.load_default()
    colors = {
        "title": (36, 99, 186, 220),
        "text": (221, 126, 43, 220),
        "visual": (32, 143, 76, 220),
    }
    fills = {
        "title": (36, 99, 186, 38),
        "text": (221, 126, 43, 38),
        "visual": (32, 143, 76, 38),
    }
    for element in result.elements:
        kind = kind_for_role(element.role)
        box = bbox_pixels(element.bbox)
        draw.rectangle(box, fill=fills[kind], outline=colors[kind], width=4)
        draw.text(
            (box[0] + 4, box[1] + 4),
            f"{element.element_id} | {element.role}",
            fill=colors[kind],
            font=label_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path)


def extract_slide(slide_id: str, output_dir: Path, reuse_raw: bool) -> dict[str, Any]:
    slide_dir = output_dir / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(slide_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    reference_path = REFERENCE_DIR / f"{slide_id}.png"
    prompt = extraction_prompt(slide_id)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_path = slide_dir / "layout_identity_prompt.txt"
    write_text(prompt_path, prompt)
    write_text(attempt_dir / prompt_path.name, prompt)

    raw_path = slide_dir / "vlm_raw_response.json"
    previous_manifest_path = slide_dir / "run_manifest.json"
    previous_manifest = read_json(previous_manifest_path) if previous_manifest_path.exists() else {}
    client = ApiMartClient(env_path=V9_ROOT / ".env")
    started_at = now_iso()
    result: RegionDetectionResult | None = None
    normalization_path: Path | None = None
    overlay_path: Path | None = None
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
            from integration.api_journal import journal_response
            response = journal_response(path=slide_dir / "api_response.json",
                request={"method": "vlm_json", "api_root": client.api_root, "model": MODEL,
                         "prompt": prompt, "max_tokens": MAX_TOKENS}, image_paths=[reference_path],
                invoke=lambda: (client.vlm_json(model=MODEL, prompt=prompt, image_path=reference_path,
                                                max_tokens=MAX_TOKENS), client.last_usage))
            raw, usage = response["raw"], response["usage"]
            reused_raw = response["reused"]
            write_text(raw_path, raw)
    except Exception as exc:
        api_error = str(exc)
        raw = ""

    if api_error is None:
        if raw_path.exists():
            write_text(attempt_dir / raw_path.name, raw_path.read_text(encoding="utf-8"))
        try:
            payload = extract_model_object(raw)
            normalized, normalization_report = normalize_payload(payload, slide_id)
            result = RegionDetectionResult.model_validate(normalized)
        except Exception as exc:
            fallback_error = fallback_error or str(exc)
            normalized, normalization_report = normalize_payload({}, slide_id)
            normalization_report["validation_errors"].append(str(exc))
            result = RegionDetectionResult.model_validate(normalized)
        normalization_path = slide_dir / "normalization_report.json"
        write_json(normalization_path, normalization_report)
        write_json(attempt_dir / normalization_path.name, normalization_report)
        identity_path = slide_dir / "layout_identity.json"
        write_json(identity_path, result)
        write_json(attempt_dir / identity_path.name, result)
        overlay_path = slide_dir / "layout_identity_overlay.png"
        render_overlay(reference_path, result, overlay_path)
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
        "max_tokens": MAX_TOKENS,
        "model_call_count": client.call_count,
        "raw_response_model_call_count": client.call_count or previous_manifest.get("raw_response_model_call_count", 0),
        "reused_raw_response": reused_raw,
        "usage": usage,
        "started_at": started_at,
        "finished_at": now_iso(),
        "inputs": {
            "reference": str(reference_path.resolve()),
            "reference_sha256": sha256(reference_path),
            "prompt_sha256": prompt_sha256,
        },
        "element_count": len(result.elements) if result else 0,
        "artifacts": {
            "attempt": str(attempt_dir.resolve()),
            "prompt": str(prompt_path.resolve()),
            "raw_response": str(raw_path.resolve()) if raw_path.exists() else None,
            "normalization_report": str(normalization_path.resolve()) if normalization_path else None,
            "layout_identity": str((slide_dir / "layout_identity.json").resolve()) if result else None,
            "overlay": str(overlay_path.resolve()) if overlay_path else None,
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_contact_sheet(slides: list[dict[str, Any]], output_dir: Path) -> Path:
    thumb_width, thumb_height, label_height = 512, 288, 32
    columns = min(2, len(slides))
    rows = math.ceil(len(slides) / columns)
    canvas = Image.new("RGB", (thumb_width * columns, (thumb_height + label_height) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, slide in enumerate(slides):
        overlay_path = slide["artifacts"].get("overlay")
        if overlay_path:
            with Image.open(overlay_path) as source:
                image = source.convert("RGB").resize((thumb_width, thumb_height))
        else:
            image = Image.new("RGB", (thumb_width, thumb_height), (245, 230, 230))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 12, y + 10), f'{slide["slide_id"]}: {slide["status"]}', fill="black")
    output_path = output_dir / "layout_identity_contact_sheet.png"
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OCR-free slide regions.")
    parser.add_argument("--slide-id", action="append", choices=[f"slide_{index:03d}" for index in range(1, 31)])
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
            slides = list(executor.map(lambda item: extract_slide(item, output_dir, args.reuse_raw), slide_ids))
    contact_sheet = make_contact_sheet(slides, output_dir)
    failed = [slide for slide in slides if slide["status"] == "failed"]
    manifest = {
        "stage": "ocr_free_region_detection",
        "status": "failed" if failed else "succeeded",
        "error": f"{len(failed)} API call(s) failed" if failed else None,
        "model": MODEL,
        "slide_count": len(slides),
        "model_call_count": sum(slide["model_call_count"] for slide in slides),
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
