"""Persist shared image submissions before polling or local postprocessing."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .apimart import ApiMartImageTaskFailure, extract_image_url


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _retryable(error: Any) -> bool:
    code = str(error.get("code", "")) if isinstance(error, dict) else ""
    return code in {"server_error", "internal_error", "upstream_error", "temporarily_unavailable"}


def run_persisted_image_task(*, client: Any, payload: dict[str, Any], task_path: Path,
                             output_path: Path, max_attempts: int = 1) -> dict[str, Any]:
    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    record = json.loads(task_path.read_text()) if task_path.is_file() else {}
    if record.get("request_sha256") != request_hash:
        record = {}
    task_id = record.get("task_id")
    attempts = int(record.get("submission_attempt", 0))
    calls = 0
    if record.get("status") == "completed" and _valid_image(output_path):
        return {"task_id": task_id, "image_url": extract_image_url(record["result"]),
                "image_calls_during_run": 0}
    if not task_id and record.get("status") in {"submitting", "submission_unknown"}:
        raise RuntimeError("Shared image submission outcome is unknown; reconcile before submitting again")
    if record.get("status") == "failed":
        if not _retryable(record.get("error")) or attempts >= max_attempts:
            raise ApiMartImageTaskFailure(str(task_id), record.get("error"))
        task_id = None
    while True:
        if not task_id:
            attempts += 1
            record = {"status": "submitting", "task_id": None, "request_sha256": request_hash,
                      "submission_attempt": attempts}
            _save(task_path, record)
            try:
                task_id = client.submit_image_task(payload)
                calls += 1
            except BaseException as exc:
                record.update(status="submission_unknown", error_type=type(exc).__name__)
                _save(task_path, record)
                raise
            record.update(status="submitted", task_id=task_id)
            _save(task_path, record)
        try:
            if record.get("status") in {"download_pending", "completed"} and record.get("result"):
                task = record["result"]
            else:
                task = client.poll_task(task_id)
        except ApiMartImageTaskFailure as exc:
            record.update(status="failed", error=exc.error)
            _save(task_path, record)
            if _retryable(exc.error) and attempts < max_attempts:
                task_id = None
                continue
            raise
        except BaseException as exc:
            record.update(status="poll_pending", error_type=type(exc).__name__)
            _save(task_path, record)
            raise
        record.update(status="download_pending", result=task)
        _save(task_path, record)
        image_url = extract_image_url(task)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".png", dir=output_path.parent)
        os.close(descriptor)
        try:
            client.download(image_url, Path(temporary))
            if not _valid_image(Path(temporary)):
                raise RuntimeError("Shared image download is not a complete image")
            os.replace(temporary, output_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        record["status"] = "completed"
        _save(task_path, record)
        return {"task_id": task_id, "image_url": image_url, "image_calls_during_run": calls}
