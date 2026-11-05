from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .json_utils import extract_json
from .models import DeckPlan, RoughSlidePlan, UserInput


class RoughSlideDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    working_title: str
    rough_message: str

    @field_validator("working_title", "rough_message")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rough slide text fields must be non-empty")
        return value


class DeckPlanDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str
    audience: str
    deck_goal: str
    text_presentation_policy: Literal["light", "styled"] = "light"
    slides: list[RoughSlideDraft] = Field(min_length=1, max_length=30)

    @field_validator("topic", "audience", "deck_goal")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("deck plan text fields must be non-empty")
        return value


def deck_planning_prompt(user_input: UserInput) -> str:
    schema = {
        "topic": "short deck topic",
        "audience": "target audience",
        "deck_goal": "what the complete deck should teach or communicate",
        "text_presentation_policy": "light or styled",
        "slides": [
            {
                "working_title": "descriptive working title, not polished final copy",
                "rough_message": "what the audience should understand after this slide",
            }
        ],
    }
    return f"""Create only the deck-level plan for a presentation.

User intention and content:
{user_input.content}

Requested visual style:
{user_input.style}

This is the first planning step. Decide what each slide should tell roughly. A later per-slide stage will decide how
to express that teaching point visually.

Requirements:
- Decide the slide count from the user's request. Honor an explicit slide count when one is provided.
- Use only as many slides as the content needs; avoid filler, repeated messages, and premature detail.
- Give every slide one distinct rough_message that states what the audience should understand after that slide.
- Keep a coherent narrative arc across the sequence.
- Set text_presentation_policy to styled only when the requested visual style clearly calls for prominent themed
  surfaces around explanatory text; otherwise use light for mostly bare editable text.
- Do not return slide IDs, counts, style, stage names, or dependency metadata; the orchestrator owns them.
- Do not plan illustrations, visual metaphors, images, icons, small assets, charts, or image prompts yet.
- Do not plan layout, geometry, bounding boxes, typography, holders, colors, or animation.
- Do not write polished body copy, speaker notes, or detailed per-slide explanations.
- Return only one JSON object matching this schema:

{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def parse_deck_plan(raw_response: str, user_input: UserInput) -> DeckPlan:
    payload = extract_json(raw_response)
    draft = DeckPlanDraft.model_validate(payload)
    return DeckPlan(
        topic=draft.topic,
        audience=draft.audience,
        deck_goal=draft.deck_goal,
        style=user_input.style,
        text_presentation_policy=draft.text_presentation_policy,
        slides=[
            RoughSlidePlan(
                slide_id=f"slide_{index:03d}",
                working_title=slide.working_title,
                rough_message=slide.rough_message,
            )
            for index, slide in enumerate(draft.slides, start=1)
        ],
    )


def mock_deck_plan(user_input: UserInput) -> DeckPlan:
    explicit_slide_count = _requested_slide_count(user_input.content)
    slide_count = explicit_slide_count or 3
    if slide_count == 1:
        slides = [
            RoughSlidePlan(
                slide_id="slide_001",
                working_title="Core message",
                rough_message=user_input.content,
            )
        ]
    elif explicit_slide_count is None or slide_count == 3:
        slides = [
            RoughSlidePlan(
                slide_id="slide_001",
                working_title="Context",
                rough_message=f"Introduce the context and motivation for: {user_input.content}",
            ),
            RoughSlidePlan(
                slide_id="slide_002",
                working_title="Core explanation",
                rough_message=f"Develop the main explanation required by: {user_input.content}",
            ),
            RoughSlidePlan(
                slide_id="slide_003",
                working_title="Takeaway",
                rough_message="Connect the explanation back to the user's intended takeaway.",
            ),
        ]
    else:
        slides = []
        for index in range(1, slide_count + 1):
            if index == 1:
                working_title = "Context"
            elif index == slide_count:
                working_title = "Takeaway"
            else:
                working_title = f"Core development {index - 1}"
            slides.append(
                RoughSlidePlan(
                    slide_id=f"slide_{index:03d}",
                    working_title=working_title,
                    rough_message=f"Advance part {index} of the requested deck: {user_input.content}",
                )
            )
    return DeckPlan(
        topic=_short_topic(user_input.content),
        audience="the audience implied by the user request",
        deck_goal=user_input.content,
        style=user_input.style,
        text_presentation_policy=infer_text_presentation_policy(user_input.style),
        slides=slides,
    )


def mock_raw_response(user_input: UserInput) -> str:
    plan = mock_deck_plan(user_input)
    return json.dumps(
        {
            "topic": plan.topic,
            "audience": plan.audience,
            "deck_goal": plan.deck_goal,
            "text_presentation_policy": plan.text_presentation_policy,
            "slides": [
                {
                    "working_title": slide.working_title,
                    "rough_message": slide.rough_message,
                }
                for slide in plan.slides
            ],
        },
        ensure_ascii=False,
    )


def _short_topic(content: str) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= 100 else compact[:97].rstrip() + "..."


def infer_text_presentation_policy(style: str) -> Literal["light", "styled"]:
    """Deterministic stand-in for the deck-level LLM decision in mock mode."""
    styled_cues = re.compile(
        r"\b(?:ornate|themed|comic|manga|anime|game|fantasy|scrapbook|sticker|"
        r"neon|cyberpunk|steampunk|hand[- ]drawn|franchise|character)\b",
        flags=re.IGNORECASE,
    )
    light_cues = re.compile(
        r"\b(?:minimal|minimalist|clean|plain|editorial|corporate|academic|"
        r"white[- ]space|typographic)\b",
        flags=re.IGNORECASE,
    )
    if light_cues.search(style) and not styled_cues.search(style):
        return "light"
    return "styled" if styled_cues.search(style) else "light"


def _requested_slide_count(content: str) -> int | None:
    match = re.search(r"\b([1-9]|[12][0-9]|30)\s*[- ]?slides?\b", content, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r"\bone\s+slides?\b", content, flags=re.IGNORECASE):
        return 1
    return None
