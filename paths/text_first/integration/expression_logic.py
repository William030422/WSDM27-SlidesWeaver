from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import (
    call_json_with_retries,
    extract_json,
    read_json,
    write_json,
    write_text,
)

from .config import PipelineConfig


class ExpressionLogicClient(Protocol):
    last_usage: dict[str, Any] | None

    def chat_json(self, *, model: str, prompt: str, max_tokens: int) -> str: ...


class SlideExpressionLogic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slide_id: str
    expression_logic: str

    @field_validator("expression_logic")
    @classmethod
    def concise_non_empty_logic(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("expression logic must be non-empty")
        return value


class ExpressionLogicPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slides: list[SlideExpressionLogic] = Field(min_length=1, max_length=30)


def build_expression_logic_prompt(text_trees: list[dict[str, Any]]) -> str:
    return f"""Describe one simple visual logic for every presentation slide below.

Visual logic is one short, natural-language composition idea. It should only give
the diffusion model a rough relationship between visuals and text. For example:
"Pair each idea with a visual and let the pairs form a circle" or "Pair each idea
with a visual and let the pairs progress in a clean left-to-right rhythm."

Requirements:
- Base every decision only on the supplied text trees.
- Return exactly one entry per slide, preserving slide IDs and order.
- Write exactly one compact sentence per slide, preferably no more than 30 words.
- Describe only the rough visual relationship, grouping, or progression.
- For an explicitly requested loop or ordered process, retain every named stage and its direction.
  A comparison needs distinguishable alternatives; a human-machine explanation must keep human and machine roles distinct.
  Do not replace missing stages with generic decorations or imply relationships unsupported by the text.
- Refer to ideas, explanations, labels, and visuals naturally. Never refer to
  internal identifiers such as text_01, text02, region IDs, or holder IDs.
- Do not prescribe where every individual text block goes. Leave local layout,
  spacing, and visual interpretation open to the diffusion model.
- Be content-driven, creative, and diverse across the deck. Avoid repeating one
  template on consecutive slides.
- Do not write HTML, CSS, pixel coordinates, colors, typography, decoration, image
  style, or final image-generation prompts.
- Do not rewrite, summarize, or quote the slide copy.

Return only one JSON object in this exact shape:
{{"slides":[{{"slide_id":"slide_001","expression_logic":"..."}}]}}

Text trees:
{json.dumps(text_trees, ensure_ascii=False, separators=(",", ":"))}
"""


def parse_expression_logic_plan(
    raw_response: str, slide_ids: list[str]
) -> ExpressionLogicPlan:
    plan = ExpressionLogicPlan.model_validate(extract_json(raw_response))
    actual_ids = [slide.slide_id for slide in plan.slides]
    if actual_ids != slide_ids:
        raise ValueError(
            f"expression logic slide IDs must be {slide_ids}, got {actual_ids}"
        )
    return plan


def mock_expression_logic_plan(
    text_trees: list[dict[str, Any]], slide_ids: list[str]
) -> ExpressionLogicPlan:
    patterns = (
        "Pair each idea with a visual and let the pairs progress in a clean rhythm.",
        "Arrange matched idea-and-visual pairs around one shared center.",
        "Use one visual sequence, pairing each stage with its explanation.",
        "Use one central visual with supporting idea-and-visual pairs radiating around it.",
    )
    slides = []
    for index, (text_tree, slide_id) in enumerate(
        zip(text_trees, slide_ids, strict=True)
    ):
        text_blocks = text_tree.get("text_blocks", [])
        if len(text_blocks) == 1:
            single_patterns = (
                "Pair one concise explanation with one dominant visual scene.",
                "Let one central visual carry the idea with a single supporting explanation.",
                "Use one visual progression paired with one concise takeaway.",
                "Wrap one explanatory visual around a single central idea.",
            )
            logic = single_patterns[index % len(single_patterns)]
        else:
            logic = patterns[index % len(patterns)]
        slides.append({"slide_id": slide_id, "expression_logic": logic})
    return ExpressionLogicPlan.model_validate({"slides": slides})


def normalize_expression_logic(
    value: str, *, fallback: str
) -> tuple[str, str | None]:
    compact = " ".join(str(value).split())
    internal_notation = re.search(
        r"\b(?:text|region|holder)[_\s-]*\d+\b|\b\d+(?:\.\d+)?\s*px\b|\b(?:html|css)\b",
        compact,
        flags=re.IGNORECASE,
    )
    if not compact:
        return fallback, "empty visual logic was replaced with a fallback"
    if internal_notation:
        return fallback, "internal layout notation was replaced with a natural-language fallback"
    if len(compact.split()) > 40 or len(compact) > 320:
        return fallback, "overly detailed visual logic was replaced with a concise fallback"
    return compact, None


def run_expression_logic_planning(
    *,
    project_root: Path,
    text_generation_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_llm: bool,
    client: ExpressionLogicClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.txt"
    raw_response_path = output_dir / "raw_response.json"
    plan_path = output_dir / "expression_logic_plan.json"
    manifest_path = output_dir / "run_manifest.json"
    started_at = _now_iso()
    model_call_count = 0
    usage: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        text_manifest_path = text_generation_dir / "run_manifest.json"
        parsed_text_path = text_generation_dir / "parsed_text_plan.json"
        for path in (text_manifest_path, parsed_text_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"expression logic requires Text Generation artifact: {path}"
                )
        text_manifest = read_json(text_manifest_path)
        if (
            text_manifest.get("stage") != "deck_level_text_generation"
            or text_manifest.get("status") != "succeeded"
        ):
            raise ValueError("expression logic requires succeeded Text Generation")
        parsed_text = read_json(parsed_text_path)
        text_trees = parsed_text.get("slides")
        if not isinstance(text_trees, list):
            raise ValueError("parsed text plan must contain a slides array")
        actual_ids = [tree.get("slide_id") for tree in text_trees]
        if actual_ids[: len(slide_ids)] != slide_ids:
            raise ValueError(
                f"expression logic text tree IDs must begin with {slide_ids}, got {actual_ids}"
            )
        text_trees = text_trees[: len(slide_ids)]

        prompt = build_expression_logic_prompt(text_trees)
        write_text(prompt_path, prompt)
        if mock_llm:
            plan = mock_expression_logic_plan(text_trees, slide_ids)
            raw_response = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)
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
                    max_tokens=min(config.chat_max_tokens, 4096),
                )

            def record_usage(attempt: int, _: str) -> None:
                last_usage = getattr(active_client, "last_usage", None)
                if last_usage is not None:
                    usage.append({"attempt": attempt, **last_usage})

            try:
                plan, raw_response = call_json_with_retries(
                    prompt=prompt,
                    call=call,
                    parse=lambda raw: parse_expression_logic_plan(raw, slide_ids),
                    response_path=raw_response_path,
                    on_response=record_usage,
                )
            except Exception as exc:
                plan = mock_expression_logic_plan(text_trees, slide_ids)
                warnings.append(
                    "model response could not be parsed after "
                    f"{model_call_count} attempts; used concise fallbacks: {exc}"
                )
        fallback_plan = mock_expression_logic_plan(text_trees, slide_ids)
        normalized_slides = []
        for slide, fallback_slide in zip(
            plan.slides, fallback_plan.slides, strict=True
        ):
            logic, warning = normalize_expression_logic(
                slide.expression_logic,
                fallback=fallback_slide.expression_logic,
            )
            normalized_slides.append(
                {"slide_id": slide.slide_id, "expression_logic": logic}
            )
            if warning:
                warnings.append(f"{slide.slide_id}: {warning}")
        plan = ExpressionLogicPlan.model_validate({"slides": normalized_slides})
        if mock_llm:
            write_text(raw_response_path, raw_response)
        write_json(plan_path, plan.model_dump(mode="json"))

        for slide in plan.slides:
            slide_dir = output_dir / "slides" / slide.slide_id
            logic_path = slide_dir / "expression_logic.json"
            write_json(logic_path, slide.model_dump(mode="json"))
            records.append(
                {
                    "slide_id": slide.slide_id,
                    "expression_logic": str(logic_path.resolve()),
                    "source_text_tree": str(parsed_text_path.resolve()),
                }
            )
        manifest = {
            "stage": "deck_level_expression_logic_planning",
            "status": "succeeded",
            "quality_status": "mock" if mock_llm else ("fallback" if warnings else "strict_pass"),
            "warnings": warnings,
            "model": None if mock_llm else config.chat_model,
            "model_call_count": model_call_count,
            "call_granularity": "one_deck_level_call_for_cross_slide_diversity",
            "mock_llm": mock_llm,
            "usage": usage,
            "slide_count": len(records),
            "source_text_generation_dir": str(text_generation_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "slides": records,
            "artifacts": {
                "prompt": str(prompt_path.resolve()),
                "raw_response": str(raw_response_path.resolve()),
                "expression_logic_plan": str(plan_path.resolve()),
                "manifest": str(manifest_path.resolve()),
            },
            "started_at": started_at,
            "finished_at": _now_iso(),
        }
    except Exception as exc:
        manifest = {
            "stage": "deck_level_expression_logic_planning",
            "status": "failed",
            "error": str(exc),
            "model": None if mock_llm else config.chat_model,
            "model_call_count": model_call_count,
            "mock_llm": mock_llm,
            "slide_count": len(records),
            "source_text_generation_dir": str(text_generation_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "slides": records,
            "started_at": started_at,
            "finished_at": _now_iso(),
        }
    write_json(manifest_path, manifest)
    return manifest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
