from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel


class JsonExtractionError(RuntimeError):
    pass


MAX_JSON_CALL_ATTEMPTS = 3
_ParsedJson = TypeVar("_ParsedJson")


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for start in [index for index, char in enumerate(cleaned) if char in "[{"]:
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
            return payload
        except json.JSONDecodeError as exc:
            last_error = exc
    raise JsonExtractionError(f"Could not extract JSON: {last_error or cleaned[:160]}")


def call_json_with_retries(
    *,
    prompt: str,
    call: Callable[[str], str],
    parse: Callable[[str], _ParsedJson],
    response_path: Path,
    on_response: Callable[[int, str], None] | None = None,
    on_validation_failure: Callable[[Exception], _ParsedJson] | None = None,
) -> tuple[_ParsedJson, str]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_JSON_CALL_ATTEMPTS + 1):
        active_prompt = (
            prompt
            if last_error is None
            else _json_repair_prompt(prompt, last_error)
        )
        raw = call(active_prompt)
        write_text(response_path, raw)
        write_text(
            response_path.with_name(
                f"{response_path.stem}_attempt_{attempt:02d}{response_path.suffix}"
            ),
            raw,
        )
        if on_response is not None:
            on_response(attempt, raw)
        try:
            return parse(raw), raw
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    if on_validation_failure is not None:
        return on_validation_failure(last_error), raw
    raise last_error


def _json_repair_prompt(prompt: str, error: Exception) -> str:
    return f"""{prompt}

Retry instruction:
The previous response was rejected by downstream JSON validation:
{type(error).__name__}: {error}

Return a complete corrected JSON object from scratch. Include every required field and item.
Return JSON only, with no explanation or markdown fences.
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.model_dump(mode="json", by_alias=True) if isinstance(data, BaseModel) else data
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
