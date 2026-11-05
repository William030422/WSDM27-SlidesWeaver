from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import (
    call_json_with_retries,
    extract_json,
    read_json,
    write_json,
    write_text,
)
from v9_deck_planning.models import DeckPlan, UserInput

from .config import PipelineConfig


TEXT_FIELD_PATTERN = re.compile(r"^text_([0-9]{2})$")
UNORDERED_ITEM_PATTERN = re.compile(r"^\s*[\u2022\u25cf\u25aa\u25e6\u2023\u2043*+-]\s+(.+?)\s*$")
ORDERED_ITEM_PATTERN = re.compile(
    r"^\s*(?:\(([0-9]{1,2})\)\s*|([0-9]{1,2})[.)]\s+|"
    r"([0-9]{1,2})[\u3001:]\s*)(.+?)\s*$"
)
MAX_TEXT_BLOCKS = 5
ASCII_ARROW_REPLACEMENTS = {
    "→": "->",
    "⟶": "->",
    "⇒": "=>",
    "←": "<-",
    "⟵": "<-",
    "⇐": "<=",
    "↔": "<->",
    "⟷": "<->",
    "⇔": "<=>",
}


class TextGenerationClient(Protocol):
    last_usage: dict[str, Any] | None

    def chat_json(self, *, model: str, prompt: str, max_tokens: int) -> str: ...


def text_generation_template(deck_plan: DeckPlan) -> str:
    del deck_plan
    return "\n".join(
        [
            "slides[]",
            "slides[].slide_id",
            "slides[].title",
            "slides[].subtitle",
            *(f"slides[].text_{index:02d}" for index in range(1, MAX_TEXT_BLOCKS + 1)),
        ]
    )


def text_generation_prompt(user_input: UserInput, deck_plan: DeckPlan) -> str:
    template = text_generation_template(deck_plan)
    planning_context = {
        "topic": deck_plan.topic,
        "audience": deck_plan.audience,
        "deck_goal": deck_plan.deck_goal,
        "slides": [
            {
                "slide_id": slide.slide_id,
                "working_title": slide.working_title,
                "purpose": slide.rough_message,
            }
            for slide in deck_plan.slides
        ],
    }
    return f"""Write the final editable text for the complete presentation deck.

Original user content:
{user_input.content}

Requested style:
{user_input.style}

Cross-planning result:
{json.dumps(planning_context, ensure_ascii=False, indent=2)}

Requirements:
- Return one JSON object and no prose outside it.
- Return every slide exactly once and preserve the supplied slide IDs and order.
- Preserve every specifically requested process stage, comparison alternative, checklist item and required distinction.
  Prioritize those requirements over optional restatements, repeated caveats, or decorative takeaway sentences.
- For ordinary English teaching slides, prefer about 45-65 body words per slide when the requirements permit;
  concise copy must remain understandable at presentation size. This is a density guide, not permission to omit a fact.
- Each slide must contain only slide_id, title, subtitle, and sequential text_NN fields.
- title must be one non-empty final title.
- subtitle must be one non-empty string or null. Use null when the slide does not need a subtitle.
- Choose the number of text blocks independently for each slide. Use between 1 and 5 blocks. Prefer
  the fewest blocks that communicate the slide purpose clearly; do not target the same count on
  every slide.
- Every slide must contain text_01. Add text_02 through text_05 only for content that should be
  placed independently. Never return text_06 or higher.
- Number text fields continuously from text_01 with two digits. Do not use a text array or a single
  combined body field.
- Choose the best format for each text block. A text_NN value may be a prose paragraph, an unordered
  list, or an ordered list.
- An itemized block is one text block, not one block per item. Keep its optional lead sentence and
  all list items in the same JSON string, separated by newline characters. Do not split list items
  across text_01, text_02, and so on.
- Use clear list markers so the format can be parsed later. For an unordered list, use `• item` on
  each item line. For an ordered list, use `1. item`, `2. item`, and so on.
- Keep distinct independently placed ideas in distinct text_NN fields, but keep a coherent itemized
  explanation together in one field.
- Write final copy only. Do not add purposes, visual ideas, layouts, coordinates, notes, metadata,
  or any other fields.
- Use ASCII arrows such as `->`, `<-`, and `<->`; do not use Unicode arrow symbols.

Valid single-block formats include:
- Prose: `A model-free agent improves directly from experience.`
- Unordered itemize: `The loop has four parts:\n• state\n• action\n• reward\n• next state`
- Ordered itemize: `Repeat these steps:\n1. Observe\n2. Act\n3. Update`

Use this field structure. It lists field names only; populate them according to the requirements
above and include sequential text_02 through text_05 only when needed:

{template}
"""


def parse_text_plan(raw_response: str, deck_plan: DeckPlan) -> dict[str, Any]:
    payload = extract_json(raw_response)
    if not isinstance(payload, dict) or set(payload) != {"slides"}:
        raise ValueError("text generation must return one object containing only slides")
    raw_slides = payload["slides"]
    if not isinstance(raw_slides, list):
        raise ValueError("slides must be an array")
    if len(raw_slides) != len(deck_plan.slides):
        raise ValueError(
            f"expected {len(deck_plan.slides)} slides, got {len(raw_slides)}"
        )

    slides = []
    for index, (raw_slide, planned_slide) in enumerate(
        zip(raw_slides, deck_plan.slides, strict=True), start=1
    ):
        slides.append(_normalize_slide(raw_slide, planned_slide.slide_id, index))
    return {"slides": slides}


def build_parsed_text_plan(text_plan: dict[str, Any]) -> dict[str, Any]:
    slides = []
    for slide in text_plan["slides"]:
        text_blocks = [
            parse_text_block(key, value)
            for key, value in slide.items()
            if TEXT_FIELD_PATTERN.fullmatch(key)
        ]
        slides.append(
            {
                "slide_id": slide["slide_id"],
                "title": slide["title"],
                "subtitle": slide["subtitle"],
                "text_blocks": text_blocks,
            }
        )
    return {"slides": slides}


def parse_text_block(text_id: str, value: str) -> dict[str, Any]:
    text = _normalize_multiline_text(value)
    lines = text.splitlines()
    marked = []
    for index, line in enumerate(lines):
        unordered = UNORDERED_ITEM_PATTERN.fullmatch(line)
        if unordered:
            marked.append((index, "unordered_list", None, unordered.group(1)))
            continue
        ordered = ORDERED_ITEM_PATTERN.fullmatch(line)
        if ordered:
            ordinal = int(ordered.group(1) or ordered.group(2) or ordered.group(3))
            marked.append((index, "ordered_list", ordinal, ordered.group(4)))

    if len(marked) < 2:
        return {"text_id": text_id, "format": "paragraph", "text": text}
    first_index, list_format, _, _ = marked[0]
    if any(item[1] != list_format for item in marked):
        return {"text_id": text_id, "format": "paragraph", "text": text}

    lead = "\n".join(line.strip() for line in lines[:first_index] if line.strip()) or None
    items = []
    ordinals = []
    for line in lines[first_index:]:
        marker = _match_list_item(line, list_format)
        if marker is not None:
            ordinal, item_text = marker
            ordinals.append(ordinal)
            items.append(item_text)
        elif line.strip() and items:
            items[-1] = f"{items[-1]} {line.strip()}"

    if len(items) < 2:
        return {"text_id": text_id, "format": "paragraph", "text": text}
    parsed = {
        "text_id": text_id,
        "format": list_format,
        "lead": lead,
        "items": items,
    }
    if list_format == "ordered_list":
        expected = list(range(ordinals[0], ordinals[0] + len(ordinals)))
        if ordinals != expected:
            return {"text_id": text_id, "format": "paragraph", "text": text}
        parsed["start"] = ordinals[0]
    return parsed


def mock_text_plan(deck_plan: DeckPlan) -> dict[str, Any]:
    return {
        "slides": [
            {
                "slide_id": slide.slide_id,
                "title": slide.working_title,
                "subtitle": None,
                "text_01": slide.rough_message,
            }
            for slide in deck_plan.slides
        ]
    }


def run_text_generation(
    *,
    project_root: Path,
    source_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    mock_llm: bool,
    client: TextGenerationClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source_dir / "run_manifest.json"
    input_path = source_dir / "input.json"
    deck_plan_path = source_dir / "deck_plan.json"
    prompt_path = output_dir / "prompt.txt"
    raw_response_path = output_dir / "raw_response.json"
    text_plan_path = output_dir / "text_plan.json"
    parsed_text_plan_path = output_dir / "parsed_text_plan.json"
    manifest_path = output_dir / "run_manifest.json"

    started_at = _now_iso()
    model_call_count = 0
    usage: list[dict[str, Any]] = []
    warnings: list[str] = []
    error = None
    status = "failed"
    slide_count = 0
    try:
        for path in (source_manifest_path, input_path, deck_plan_path):
            if not path.is_file():
                raise FileNotFoundError(f"text generation requires Step 1 artifact: {path}")
        source_manifest = read_json(source_manifest_path)
        if (
            source_manifest.get("stage") != "deck_level_planning"
            or source_manifest.get("status") != "succeeded"
        ):
            raise ValueError("text generation requires succeeded Step 1 planning")
        user_input = UserInput.model_validate(read_json(input_path))
        deck_plan = DeckPlan.model_validate(read_json(deck_plan_path))
        if deck_plan.style != user_input.style:
            raise ValueError("Step 1 deck plan does not preserve the input style")
        prompt = text_generation_prompt(user_input, deck_plan)
        write_text(prompt_path, prompt)

        if mock_llm:
            raw_response = json.dumps(mock_text_plan(deck_plan), ensure_ascii=False)
            write_text(raw_response_path, raw_response)
            text_plan = parse_text_plan(raw_response, deck_plan)
        else:
            active_client = client or ApiMartClient(
                api_root=config.api_root,
                api_key_env=config.api_key_env,
                env_path=project_root / ".env",
                timeout_seconds=config.timeout_seconds,
            )

            def call(active_prompt: str) -> str:
                nonlocal model_call_count
                model_call_count += 1
                return active_client.chat_json(
                    model=config.chat_model,
                    prompt=active_prompt,
                    max_tokens=config.chat_max_tokens,
                )

            def record_usage(attempt: int, _: str) -> None:
                last_usage = getattr(active_client, "last_usage", None)
                if last_usage is not None:
                    usage.append({"attempt": attempt, **last_usage})

            def recover_invalid_copy(exc: Exception) -> dict[str, Any]:
                # No final wording is committed yet: retain the established deck intent.
                warnings.append(f"text response validation exhausted; retained planned copy: {exc}")
                return recover_text_plan(raw_response_path.read_text(encoding="utf-8"), deck_plan)

            text_plan, raw_response = call_json_with_retries(
                prompt=prompt,
                call=call,
                parse=lambda raw: parse_text_plan(raw, deck_plan),
                response_path=raw_response_path,
                on_response=record_usage,
                on_validation_failure=recover_invalid_copy,
            )
        write_json(text_plan_path, text_plan)
        write_json(parsed_text_plan_path, build_parsed_text_plan(text_plan))
        slide_count = len(text_plan["slides"])
        status = "succeeded"
    except Exception as exc:
        error = str(exc)

    manifest = {
        "stage": "deck_level_text_generation",
        "status": status,
        "error": error,
        "quality_status": ("degraded" if warnings else "strict_pass") if status == "succeeded" else "failed",
        "warnings": warnings,
        "model": config.chat_model,
        "model_call_count": model_call_count,
        "mock_llm": mock_llm,
        "usage": usage,
        "slide_count": slide_count,
        "schema": {
            "fixed_fields": ["slide_id", "title", "subtitle"],
            "body_field_pattern": "text_NN",
            "minimum_body_fields": 1,
            "maximum_body_fields": MAX_TEXT_BLOCKS,
            "parsed_formats": ["paragraph", "unordered_list", "ordered_list"],
        },
        "started_at": started_at,
        "finished_at": _now_iso(),
        "source_step1_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "artifacts": {
            "prompt": str(prompt_path.resolve()) if prompt_path.is_file() else None,
            "raw_response": (
                str(raw_response_path.resolve()) if raw_response_path.is_file() else None
            ),
            "text_plan": str(text_plan_path.resolve()) if text_plan_path.is_file() else None,
            "parsed_text_plan": (
                str(parsed_text_plan_path.resolve())
                if parsed_text_plan_path.is_file()
                else None
            ),
            "manifest": str(manifest_path.resolve()),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def recover_text_plan(raw_response: str, deck_plan: DeckPlan) -> dict[str, Any]:
    """Repair malformed fields only; retain every usable generated slide and block."""
    fallback = mock_text_plan(deck_plan)
    try:
        payload = extract_json(raw_response)
    except Exception:
        return fallback
    candidates = payload.get("slides", []) if isinstance(payload, dict) else []
    if not isinstance(candidates, list):
        return fallback
    by_id = {item["slide_id"]: item for item in candidates if isinstance(item, dict) and isinstance(item.get("slide_id"), str)}
    known_ids = {item["slide_id"] for item in fallback["slides"]}
    for index, default in enumerate(fallback["slides"]):
        candidate = by_id.get(default["slide_id"])
        if candidate is None and index < len(candidates) and isinstance(candidates[index], dict):
            positional = candidates[index]
            positional_id = positional.get("slide_id")
            if not isinstance(positional_id, str) or positional_id not in known_ids:
                candidate = positional
        if not isinstance(candidate, dict):
            continue
        title = candidate.get("title")
        if isinstance(title, str) and title.strip():
            default["title"] = _normalize_multiline_text(title)
        subtitle = candidate.get("subtitle")
        default["subtitle"] = _normalize_multiline_text(subtitle) if isinstance(subtitle, str) and subtitle.strip() else None
        body = [value for key, value in sorted(candidate.items()) if TEXT_FIELD_PATTERN.fullmatch(str(key)) and isinstance(value, str) and value.strip()]
        if body:
            default.pop("text_01", None)
            for block_index, value in enumerate(body[:MAX_TEXT_BLOCKS], start=1):
                default[f"text_{block_index:02d}"] = _normalize_multiline_text(value)
            if len(body) > MAX_TEXT_BLOCKS:
                default[f"text_{MAX_TEXT_BLOCKS:02d}"] += "\n" + "\n".join(body[MAX_TEXT_BLOCKS:])
    return fallback


def _normalize_slide(raw_slide: Any, expected_slide_id: str, index: int) -> dict[str, Any]:
    if not isinstance(raw_slide, dict):
        raise ValueError(f"slides[{index - 1}] must be an object")
    fixed_fields = {"slide_id", "title", "subtitle"}
    unknown_fields = sorted(
        key
        for key in raw_slide
        if key not in fixed_fields and not TEXT_FIELD_PATTERN.fullmatch(key)
    )
    if unknown_fields:
        raise ValueError(f"{expected_slide_id} contains unknown fields: {unknown_fields}")
    missing_fields = sorted(fixed_fields - set(raw_slide))
    if missing_fields:
        raise ValueError(f"{expected_slide_id} is missing fields: {missing_fields}")
    if raw_slide["slide_id"] != expected_slide_id:
        raise ValueError(
            f"slide {index} ID must be {expected_slide_id}, got {raw_slide['slide_id']!r}"
        )

    title = _non_empty_string(raw_slide["title"], f"{expected_slide_id}.title")
    raw_subtitle = raw_slide["subtitle"]
    if raw_subtitle is None:
        subtitle = None
    elif isinstance(raw_subtitle, str):
        subtitle = raw_subtitle.strip() or None
    else:
        raise ValueError(f"{expected_slide_id}.subtitle must be a string or null")

    text_fields = []
    for key in raw_slide:
        match = TEXT_FIELD_PATTERN.fullmatch(key)
        if match:
            text_fields.append((int(match.group(1)), key))
    text_fields.sort()
    if not text_fields:
        raise ValueError(f"{expected_slide_id} must contain text_01")
    if len(text_fields) > MAX_TEXT_BLOCKS:
        raise ValueError(
            f"{expected_slide_id} has {len(text_fields)} text blocks; "
            f"the maximum is {MAX_TEXT_BLOCKS}"
        )
    expected_numbers = list(range(1, len(text_fields) + 1))
    actual_numbers = [number for number, _ in text_fields]
    if actual_numbers != expected_numbers:
        expected_keys = [f"text_{number:02d}" for number in expected_numbers]
        actual_keys = [key for _, key in text_fields]
        raise ValueError(
            f"{expected_slide_id} text fields must be continuous: "
            f"expected {expected_keys}, got {actual_keys}"
        )

    normalized = {
        "slide_id": expected_slide_id,
        "title": title,
        "subtitle": subtitle,
    }
    for _, key in text_fields:
        normalized[key] = _normalize_multiline_text(
            _non_empty_string(raw_slide[key], f"{expected_slide_id}.{key}")
        )
    return normalized


def _match_list_item(line: str, list_format: str) -> tuple[int | None, str] | None:
    if list_format == "unordered_list":
        match = UNORDERED_ITEM_PATTERN.fullmatch(line)
        return (None, match.group(1)) if match else None
    match = ORDERED_ITEM_PATTERN.fullmatch(line)
    if match is None:
        return None
    ordinal = int(match.group(1) or match.group(2) or match.group(3))
    return ordinal, match.group(4)


def _normalize_multiline_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    for source, replacement in ASCII_ARROW_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    lines = normalized.split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
