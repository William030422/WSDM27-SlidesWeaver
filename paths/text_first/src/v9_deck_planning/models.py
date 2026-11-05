from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class UserInput(StrictModel):
    content: str
    style: str

    @field_validator("content", "style")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content and style must be non-empty strings")
        return value


class RoughSlidePlan(StrictModel):
    slide_id: str = Field(pattern=r"^slide_[0-9]{3}$")
    working_title: str
    rough_message: str

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("working_title", "rough_message")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rough slide text fields must be non-empty")
        return value


class DeckPlan(StrictModel):
    topic: str
    audience: str
    deck_goal: str
    style: str
    text_presentation_policy: Literal["light", "styled"] = "light"
    slides: list[RoughSlidePlan] = Field(min_length=1, max_length=30)

    # Older artifacts contain orchestration and diagnostic fields that are no
    # longer part of the model-generated schema.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("topic", "audience", "deck_goal", "style")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("deck plan text fields must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_slide_sequence(self) -> "DeckPlan":
        expected_ids = [f"slide_{index:03d}" for index in range(1, len(self.slides) + 1)]
        actual_ids = [slide.slide_id for slide in self.slides]
        if actual_ids != expected_ids:
            raise ValueError(f"slide IDs must be sequential: expected {expected_ids}, got {actual_ids}")
        return self
