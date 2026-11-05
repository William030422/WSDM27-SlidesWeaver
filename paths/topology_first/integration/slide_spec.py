from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import extract_json, read_json, write_json, write_text
from v9_deck_planning.models import DeckPlan, RoughSlidePlan, UserInput

from .config import PipelineConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlideContentDraft(StrictModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    message: str
    visual_intent: str | None = None

    @field_validator("title", "message")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("slide content text fields must be non-empty")
        return value

    @field_validator("visual_intent")
    @classmethod
    def optional_non_empty_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("visual intent must be non-empty when provided")
        return value


class SlideSpec(StrictModel):
    slide_id: str
    title: str
    message: str
    visual_intent: str

    @field_validator("title", "message", "visual_intent")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("slide specification text fields must be non-empty")
        return value


class SlideSpecClient(Protocol):
    def chat_json(self, *, model: str, prompt: str, max_tokens: int) -> str: ...


def run_slide_spec_bridge(
    *,
    project_root: Path,
    step2_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    mock_llm: bool,
    client_factory: Callable[[], SlideSpecClient] | None = None,
    slide_ids: list[str] | None = None,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    try:
        deck_plan = _load_step2_sources(step2_dir, config)
        client = None
        if not mock_llm:
            client = client_factory() if client_factory else ApiMartClient(
                api_root=config.api_root,
                api_key_env=config.api_key_env,
                env_path=project_root / ".env",
                timeout_seconds=config.timeout_seconds,
                poll_initial_delay_seconds=config.poll_initial_delay_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
                poll_timeout_seconds=config.poll_timeout_seconds,
            )

        records = []
        warnings = []
        for slide in deck_plan.slides:
            if slide_ids is not None and slide.slide_id not in slide_ids:
                continue
            slide_dir = output_dir / "slides" / slide.slide_id
            prompt = slide_spec_prompt(deck_plan=deck_plan, slide=slide)
            prompt_path = slide_dir / "prompt.txt"
            raw_path = slide_dir / "raw_response.json"
            spec_path = slide_dir / "slide_spec.json"
            write_text(prompt_path, prompt)
            if mock_llm:
                draft = mock_slide_content(slide)
                raw = draft.model_dump_json(indent=2)
            else:
                if client is None:
                    raise RuntimeError("slide-spec planning requires an API client")
                from .api_journal import journal_response
                tokens = min(config.chat_max_tokens, 2048)
                response = journal_response(path=slide_dir / "api_response.json",
                    request={"method": "chat_json", "api_root": config.api_root, "model": config.chat_model,
                             "prompt": prompt, "max_tokens": tokens},
                    invoke=lambda: client.chat_json(model=config.chat_model, prompt=prompt, max_tokens=tokens))
                raw = response["raw"]
                try:
                    draft = SlideContentDraft.model_validate(extract_json(raw))
                except Exception as exc:
                    draft = mock_slide_content(slide)
                    warnings.append(
                        {"slide_id": slide.slide_id, "fallback": "rough_slide_plan", "error": str(exc)}
                    )
            spec = finalize_slide_spec(slide.slide_id, draft, slide.rough_message)
            write_text(raw_path, raw)
            write_json(spec_path, spec)
            records.append(
                {
                    "slide_id": slide.slide_id,
                    "slide_spec": str(spec_path.resolve()),
                    "prompt": str(prompt_path.resolve()),
                    "raw_response": str(raw_path.resolve()),
                }
            )

        manifest = {
            "stage": "lightweight_per_slide_semantic_bridge",
            "status": "succeeded",
            "source_step2_dir": str(step2_dir.resolve()),
            "model": config.chat_model,
            "model_call_count": 0 if mock_llm else len(records),
            "mock_llm": mock_llm,
            "slide_count": len(records),
            "quality_status": "fallback" if warnings else "strict_pass",
            "warnings": warnings,
            "schema_fields": ["slide_id", "title", "message", "visual_intent"],
            "output_dir": str(output_dir.resolve()),
            "slides": records,
        }
    except Exception as exc:
        manifest = {
            "stage": "lightweight_per_slide_semantic_bridge",
            "status": "failed",
            "error": str(exc),
            "source_step2_dir": str(step2_dir.resolve()),
            "model": config.chat_model,
            "model_call_count": 0,
            "mock_llm": mock_llm,
            "slide_count": 0,
            "output_dir": str(output_dir.resolve()),
            "slides": [],
        }
    if write_stage_manifest:
        write_json(manifest_path, manifest)
    if manifest["status"] != "succeeded":
        raise RuntimeError(f"slide-spec bridge failed: {manifest['error']}")
    return manifest


def slide_spec_prompt(
    *,
    deck_plan: DeckPlan,
    slide: RoughSlidePlan,
) -> str:
    return f"""Create a concise semantic and visual idea for one presentation slide.

Deck topic: {deck_plan.topic}
Audience: {deck_plan.audience}
Deck goal: {deck_plan.deck_goal}
Requested deck style and character identities: {deck_plan.style}
Slide ID: {slide.slide_id}
Title direction: {slide.working_title}
Message direction: {slide.rough_message}

Return exactly one JSON object with only these three top-level fields:
- title: one concise slide title
- message: one concise sentence stating the teaching point
- visual_intent: one short, open-ended idea for how the teaching point could be expressed visually

Do not return the slide ID or repeat title/message in another structure; the orchestrator owns IDs.
The visual_intent is only rough inspiration. It may use a metaphor, scene, process, comparison, character, object,
or any other visual language that fits the topic. Keep it high-level and leave the actual composition to diffusion.
Do not specify coordinates, masks, exact asset placement, or typography.
Keep every teaching requirement in the message direction, including conditions such as delay, uncertainty,
and evaluation. Do not replace named human characters with machines or invent a ban on the requested identities.
"""


def mock_slide_content(slide: RoughSlidePlan) -> SlideContentDraft:
    return SlideContentDraft(
        title=slide.working_title,
        message=slide.rough_message,
        visual_intent=f"Find a memorable visual way to explain: {slide.rough_message}",
    )


def finalize_slide_spec(
    slide_id: str,
    draft: SlideContentDraft,
    rough_message: str,
) -> SlideSpec:
    return SlideSpec(
        slide_id=slide_id,
        title=draft.title,
        # The model may polish a title/visual idea, but the teaching brief is
        # authoritative. Re-summarizing it here used to lose checklist items.
        message=rough_message.strip() or draft.message,
        visual_intent=(
            draft.visual_intent
            or f"Find a memorable visual way to explain: {rough_message}"
        ),
    )


def _load_step2_sources(
    step2_dir: Path,
    config: PipelineConfig,
) -> DeckPlan:
    required = {
        "manifest": step2_dir / "run_manifest.json",
        "input": step2_dir / "input.json",
        "deck_plan": step2_dir / "deck_plan.json",
        "background": step2_dir / "canonical_background.png",
        "design_system": step2_dir / "design_system.json",
        "asset_manifest": step2_dir / "asset_manifest.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"required Step 2 artifact is missing: {path}")
    manifest = read_json(required["manifest"])
    if manifest.get("stage") != "style_asset_generation" or manifest.get("status") != "succeeded":
        raise ValueError("slide-spec bridge requires a succeeded Step 2 manifest")
    user_input = UserInput.model_validate(read_json(required["input"]))
    deck_plan = DeckPlan.model_validate(read_json(required["deck_plan"]))
    design_system = read_json(required["design_system"])
    asset_manifest = read_json(required["asset_manifest"])
    if design_system.get("style_prompt") != user_input.style:
        raise ValueError("Step 2 design system does not preserve the input style")
    if not isinstance(asset_manifest.get("assets"), list):
        raise ValueError("Step 2 asset manifest is invalid")
    with Image.open(required["background"]) as background:
        expected = (config.image_size.width, config.image_size.height)
        if background.mode != "RGB" or background.size != expected:
            raise ValueError(f"Step 2 background must be RGB {expected}")
    return deck_plan
