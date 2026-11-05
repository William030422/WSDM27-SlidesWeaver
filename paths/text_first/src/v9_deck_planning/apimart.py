from __future__ import annotations

import base64
import errno
import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_API_ROOT = "https://api.apimart.ai/v1"
IDEMPOTENT_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_RETRYABLE_OS_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.ETIMEDOUT,
}
_Result = TypeVar("_Result")


class ApiMartImageTaskFailure(RuntimeError):
    def __init__(self, task_id: str, error: Any) -> None:
        super().__init__(f"APIMART image task {task_id} failed: {error}")
        self.task_id = task_id
        self.error = error
        self.image_calls_during_run = 0


class ApiMartClient:
    def __init__(
        self,
        *,
        api_root: str = DEFAULT_API_ROOT,
        api_key_env: str = "APIMART_API_KEY",
        env_path: str | Path = ".env",
        timeout_seconds: float = 240.0,
        poll_initial_delay_seconds: float = 10.0,
        poll_interval_seconds: float = 5.0,
        poll_timeout_seconds: float = 1200.0,
        idempotent_retry_delays_seconds: tuple[float, ...] = (
            IDEMPOTENT_RETRY_DELAYS_SECONDS
        ),
    ) -> None:
        self.api_root = api_root.rstrip("/")
        self.api_key_env = api_key_env
        self.env_path = Path(env_path)
        self.timeout_seconds = timeout_seconds
        self.poll_initial_delay_seconds = poll_initial_delay_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        self.idempotent_retry_delays_seconds = tuple(
            float(delay) for delay in idempotent_retry_delays_seconds
        )
        if any(delay < 0 for delay in self.idempotent_retry_delays_seconds):
            raise ValueError("idempotent retry delays must be non-negative")
        self.call_count = 0
        self.image_call_count = 0
        self.last_usage: dict[str, Any] | None = None
        self._env_loaded = False

    def chat_json(self, *, model: str, prompt: str, max_tokens: int = 4096) -> str:
        self.call_count += 1
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise deck-planning JSON API. Return one valid JSON object, "
                        "with no markdown fences and no prose outside JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._post_json("/chat/completions", payload)
        self.last_usage = _extract_usage(data)
        text = _extract_text_response(data)
        if not text.strip():
            raise RuntimeError(f"APIMART chat returned empty content: {data}")
        return text

    def vlm_json(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int = 4096,
    ) -> str:
        self.call_count += 1
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a precise visual critic JSON API. Return one valid JSON object, "
                                "with no markdown fences and no prose outside JSON."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_to_data_uri(image_path),
                        },
                    ],
                },
            ],
            "text": {"format": {"type": "json_object"}},
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            data = self._post_json("/responses", payload)
        except RuntimeError as exc:
            if "convert_request_failed" not in str(
                exc
            ) and "not implemented" not in str(exc):
                raise
            data = self._post_json(
                "/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a precise visual critic JSON API. Return one valid JSON object, "
                                "with no markdown fences and no prose outside JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_to_data_uri(image_path)},
                                },
                            ],
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        self.last_usage = _extract_usage(data)
        text = _extract_text_response(data)
        if not text.strip():
            raise RuntimeError(f"APIMART VLM returned empty content: {data}")
        return text

    def submit_image_task(self, payload: dict[str, Any]) -> str:
        self.image_call_count += 1
        data = self._post_json("/images/generations", payload)
        task_id = _extract_task_id(data)
        if not task_id:
            raise RuntimeError(f"APIMART image response missing task_id: {data}")
        return task_id

    def poll_task(self, task_id: str) -> dict[str, Any]:
        if self.poll_initial_delay_seconds > 0:
            time.sleep(self.poll_initial_delay_seconds)
        deadline = time.monotonic() + self.poll_timeout_seconds
        last_status = "submitted"
        while time.monotonic() < deadline:
            data = _unwrap(
                self._retry_idempotent(
                    lambda: self._get_json(f"/tasks/{task_id}"),
                    operation=f"poll APIMART task {task_id}",
                )
            )
            if not isinstance(data, dict):
                raise TypeError(f"Unexpected task response for {task_id}: {data}")
            last_status = str(data.get("status") or "unknown")
            if last_status == "completed":
                return data
            if last_status == "failed":
                raise ApiMartImageTaskFailure(task_id, data.get("error") or data)
            time.sleep(self.poll_interval_seconds)
        raise RuntimeError(
            f"Timed out waiting for APIMART image task {task_id}; last status={last_status}"
        )

    def download(self, url: str, output_path: Path) -> None:
        request = urllib.request.Request(url, method="GET")
        content = self._retry_idempotent(
            lambda: self._download_bytes(request),
            operation="download APIMART image",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)

    def _download_bytes(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Download HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Download failed: {exc}") from exc

    def _retry_idempotent(
        self,
        operation_fn: Callable[[], _Result],
        *,
        operation: str,
    ) -> _Result:
        attempts = len(self.idempotent_retry_delays_seconds) + 1
        for attempt in range(attempts):
            try:
                return operation_fn()
            except Exception as exc:
                if attempt >= len(
                    self.idempotent_retry_delays_seconds
                ) or not _is_retryable_transport_error(exc):
                    raise
                time.sleep(self.idempotent_retry_delays_seconds[attempt])
        raise RuntimeError(f"{operation} exhausted retries")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.api_root}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._request_json(request)

    def _get_json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.api_root}{path}",
            headers={"Authorization": f"Bearer {self._api_key()}"},
            method="GET",
        )
        return self._request_json(request)

    def _request_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"APIMART HTTP {exc.code}: {body[:1200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"APIMART request failed: {exc}") from exc

    def _api_key(self) -> str:
        if not self._env_loaded:
            load_env_file(self.env_path)
            self._env_loaded = True
        for key in (self.api_key_env, "APIMART_API_KEY", "OPENAI_API_KEY", "API_KEY"):
            value = os.environ.get(key)
            if value:
                return value
        raise RuntimeError(
            f"API key not found. Set {self.api_key_env} in the environment or {self.env_path}."
        )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def image_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def redacted_image_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    if "image_urls" in redacted:
        redacted["image_urls"] = [
            f"<data-uri-or-url:{len(str(item))} chars>"
            for item in redacted["image_urls"]
        ]
    if "mask_url" in redacted:
        redacted["mask_url"] = f"<mask:{len(str(redacted['mask_url']))} chars>"
    return redacted


def extract_image_url(task: dict[str, Any]) -> str:
    try:
        value = task["result"]["images"][0]["url"]
        if isinstance(value, list):
            value = value[0]
        url = str(value).strip()
        if not url:
            raise ValueError("empty URL")
        return url
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not extract image URL from task response: {task}"
        ) from exc


def _is_retryable_transport_error(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, urllib.error.HTTPError):
            if current.code in RETRYABLE_HTTP_STATUS_CODES:
                return True
        elif isinstance(current, socket.gaierror):
            if current.errno == socket.EAI_AGAIN:
                return True
        elif (
            isinstance(
                current,
                (
                    TimeoutError,
                    socket.timeout,
                    ConnectionAbortedError,
                    ConnectionResetError,
                ),
            )
            or isinstance(current, OSError)
            and current.errno in _RETRYABLE_OS_ERRNOS
        ):
            return True
        nested = [current.__cause__, current.__context__]
        if isinstance(current, urllib.error.URLError) and isinstance(
            current.reason, BaseException
        ):
            nested.append(current.reason)
        pending.extend(item for item in nested if item is not None)
    return False


def _unwrap(response_json: Any) -> Any:
    if isinstance(response_json, dict) and "data" in response_json:
        return response_json["data"]
    return response_json


def _extract_task_id(response_json: dict[str, Any]) -> str:
    payload = _unwrap(response_json)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if isinstance(payload, dict):
        return str(payload.get("task_id") or "").strip()
    return ""


def _extract_usage(response_json: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in (response_json, response_json.get("data")):
        if isinstance(candidate, dict) and isinstance(candidate.get("usage"), dict):
            return candidate["usage"]
    return None


def _extract_text_response(response_json: dict[str, Any]) -> str:
    candidates: list[Any] = [response_json, response_json.get("data")]
    for data in candidates:
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = (
                choices[0].get("message") if isinstance(choices[0], dict) else None
            )
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        str(item.get("text") or item.get("content") or "")
                        for item in content
                        if isinstance(item, dict)
                    ]
                    if any(parts):
                        return "\n".join(part for part in parts if part)
        output = data.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                content = item.get("content") if isinstance(item, dict) else None
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict):
                        parts.append(
                            str(block.get("text") or block.get("content") or "")
                        )
            if any(parts):
                return "\n".join(part for part in parts if part)
    raise RuntimeError(f"Unexpected APIMART text response shape: {response_json}")
