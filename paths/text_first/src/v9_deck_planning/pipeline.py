from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .apimart import DEFAULT_API_ROOT, ApiMartClient
from .json_utils import call_json_with_retries, read_json, write_json, write_text
from .models import UserInput
from .planner import deck_planning_prompt, mock_deck_plan, mock_raw_response, parse_deck_plan


DEFAULT_LLM_MODEL = "claude-sonnet-5"


class ChatJsonClient(Protocol):
    call_count: int
    last_usage: dict[str, Any] | None

    def chat_json(self, *, model: str, prompt: str, max_tokens: int = 4096) -> str: ...


def main(*, project_root: Path) -> None:
    args = parse_args(project_root)
    output_dir = args.output_dir or project_root / "outputs" / f"deck_plan_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_input = prepare_user_input(args)
    except Exception as exc:
        manifest = failure_manifest(output_dir=output_dir, model=args.llm_model, error=str(exc), model_call_count=0)
        manifest_path = output_dir / "run_manifest.json"
        write_json(manifest_path, manifest)
        print(json.dumps(summary(manifest, manifest_path), ensure_ascii=False, indent=2))
        raise SystemExit(1) from None

    manifest = run_deck_planning(
        project_root=project_root,
        user_input=user_input,
        output_dir=output_dir,
        llm_model=args.llm_model,
        api_root=args.api_root,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        mock_llm=args.mock_llm,
    )
    manifest_path = output_dir / "run_manifest.json"
    print(json.dumps(summary(manifest, manifest_path), ensure_ascii=False, indent=2))
    if manifest["status"] != "succeeded":
        raise SystemExit(1)


def run_deck_planning(
    *,
    project_root: Path,
    user_input: UserInput,
    output_dir: Path,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_root: str = DEFAULT_API_ROOT,
    api_key_env: str = "APIMART_API_KEY",
    timeout_seconds: float = 240.0,
    max_tokens: int = 4096,
    mock_llm: bool = False,
    client: ChatJsonClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.json"
    prompt_path = output_dir / "deck_plan_prompt.txt"
    raw_response_path = output_dir / "deck_plan_raw_response.json"
    deck_plan_path = output_dir / "deck_plan.json"
    manifest_path = output_dir / "run_manifest.json"

    write_json(input_path, user_input)
    prompt = deck_planning_prompt(user_input)
    write_text(prompt_path, prompt)

    started_at = now_iso()
    model_call_count = 0
    usage: list[dict[str, Any]] = []
    error: str | None = None
    warnings: list[str] = []
    fallback_used = False
    status = "failed"
    try:
        if mock_llm:
            raw_response = mock_raw_response(user_input)
            write_text(raw_response_path, raw_response)
            deck_plan = parse_deck_plan(raw_response, user_input)
        else:
            active_client = client or ApiMartClient(
                api_root=api_root,
                api_key_env=api_key_env,
                env_path=project_root / ".env",
                timeout_seconds=timeout_seconds,
            )

            def call(active_prompt: str) -> str:
                nonlocal model_call_count
                model_call_count += 1
                return active_client.chat_json(
                    model=llm_model,
                    prompt=active_prompt,
                    max_tokens=max_tokens,
                )

            def record_usage(attempt: int, _: str) -> None:
                if active_client.last_usage is not None:
                    usage.append({"attempt": attempt, **active_client.last_usage})

            try:
                deck_plan, raw_response = call_json_with_retries(
                    prompt=prompt,
                    call=call,
                    parse=lambda raw: parse_deck_plan(raw, user_input),
                    response_path=raw_response_path,
                    on_response=record_usage,
                )
            except Exception as exc:
                deck_plan = mock_deck_plan(user_input)
                fallback_used = True
                warnings.append(
                    f"model response fallback after {model_call_count} attempts: {exc}"
                )
        write_json(deck_plan_path, deck_plan)
        status = "succeeded"
    except Exception as exc:
        error = str(exc)

    manifest = {
        "stage": "deck_level_planning",
        "status": status,
        "error": error,
        "quality_status": "fallback" if fallback_used else "strict_pass",
        "warnings": warnings,
        "model": llm_model,
        "model_call_count": model_call_count,
        "mock_llm": mock_llm,
        "usage": usage,
        "started_at": started_at,
        "finished_at": now_iso(),
        "output_dir": str(output_dir),
        "artifacts": {
            "input": str(input_path),
            "prompt": str(prompt_path),
            "raw_response": str(raw_response_path) if raw_response_path.exists() else None,
            "deck_plan": str(deck_plan_path) if deck_plan_path.exists() else None,
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V9 step 1: create a rough deck-level plan with one LLM call.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="JSON file with {content, style}.")
    input_group.add_argument("--content", help="User intention/content. Requires --style.")
    parser.add_argument("--style", help="Requested visual style when using --content.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api-root", default=DEFAULT_API_ROOT)
    parser.add_argument("--api-key-env", default="APIMART_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mock-llm", action="store_true", help="Use a deterministic local response for testing.")
    args = parser.parse_args()

    if args.content and not args.style:
        parser.error("--style is required when using --content")
    if args.input is not None and not args.input.is_absolute():
        args.input = project_root / args.input
    if args.output_dir is not None and not args.output_dir.is_absolute():
        args.output_dir = project_root / args.output_dir
    return args


def prepare_user_input(args: argparse.Namespace) -> UserInput:
    data = read_json(args.input) if args.input else {"content": args.content, "style": args.style}
    return UserInput.model_validate(normalize_user_input_payload(data))


def normalize_user_input_payload(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    if "content" in data or "style" in data:
        return data
    if "user_request" in data or "style_description" in data:
        return {
            "content": data.get("user_request"),
            "style": data.get("style_description"),
        }
    return data


def failure_manifest(*, output_dir: Path, model: str, error: str, model_call_count: int) -> dict[str, Any]:
    return {
        "stage": "deck_level_planning",
        "status": "failed",
        "error": error,
        "model": model,
        "model_call_count": model_call_count,
        "mock_llm": False,
        "usage": None,
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "output_dir": str(output_dir),
        "artifacts": {"manifest": str(output_dir / "run_manifest.json")},
    }


def summary(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") or {}
    return {
        "ok": manifest.get("status") == "succeeded",
        "status": manifest.get("status"),
        "model_call_count": manifest.get("model_call_count"),
        "output_dir": manifest.get("output_dir"),
        "deck_plan": artifacts.get("deck_plan"),
        "manifest": str(manifest_path),
        "error": manifest.get("error"),
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
