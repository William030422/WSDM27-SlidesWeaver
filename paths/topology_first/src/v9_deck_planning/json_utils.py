from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class JsonExtractionError(RuntimeError):
    pass


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.model_dump(mode="json", by_alias=True) if isinstance(data, BaseModel) else data
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
