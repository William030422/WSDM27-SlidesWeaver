"""Persist completed paid responses before downstream parsing or rendering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .page_runtime import atomic_write_json


def journal_response(*, path: Path, request: dict[str, Any], invoke: Callable[[], Any],
                     image_paths: list[Path] | None = None) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    for image_path in image_paths or []:
        digest.update(Path(image_path).read_bytes())
    fingerprint = digest.hexdigest()
    previous = json.loads(path.read_text()) if path.is_file() else {}
    if previous.get("fingerprint") == fingerprint:
        if previous.get("status") == "completed":
            return {**previous["response"], "reused": True}
        if previous.get("status") in {"submitting", "submission_unknown"}:
            raise RuntimeError("Previous model submission outcome is unknown; reconcile before submitting again")
    atomic_write_json(path, {"status": "submitting", "fingerprint": fingerprint})
    try:
        result = invoke()
    except BaseException as exc:
        atomic_write_json(path, {"status": "submission_unknown", "fingerprint": fingerprint,
                                 "error_type": type(exc).__name__})
        raise
    raw, usage = result if isinstance(result, tuple) else (result, None)
    response = {"raw": raw, "usage": usage}
    atomic_write_json(path, {"status": "completed", "fingerprint": fingerprint, "response": response})
    return {**response, "reused": False}
