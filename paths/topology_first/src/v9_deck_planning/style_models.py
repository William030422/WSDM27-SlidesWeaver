from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel


AssetCategory = Literal["title_bar", "subtitle_bar", "raster_holder", "identity_asset", "badge", "icon"]
AssetShape = Literal["bar", "rounded_rect", "circle", "speech_bubble", "callout", "pill", "freeform"]


class StyleAssetSlot(StrictModel):
    slot_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: AssetCategory
    shape: AssetShape
    bbox: tuple[float, float, float, float]
    enabled: bool = True
    required: bool = True
    generation_instruction: str
    reuse_hint: str

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = value
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"bbox must be normalized and non-empty: {value}")
        return value

    @field_validator("generation_instruction", "reuse_hint")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("slot instructions must be non-empty")
        return value


class StyleReferencePlan(StrictModel):
    stage: Literal["style_asset_generation"] = "style_asset_generation"
    source_style: str
    board_size: tuple[int, int] = (1536, 864)
    key_color: str = Field(pattern=r"^#[0-9a-f]{6}$")
    boundary_tolerance_px: int = Field(default=16, ge=0, le=32)
    avatar_policy: Literal["included", "fallback"]
    avatar_policy_reason: str
    slots: list[StyleAssetSlot] = Field(min_length=1)

    @field_validator("source_style", "avatar_policy_reason")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("style-reference text fields must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_slots(self) -> "StyleReferencePlan":
        ids = [slot.slot_id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("style asset slot IDs must be unique")

        enabled = [slot for slot in self.slots if slot.enabled]
        required_categories = {"title_bar", "subtitle_bar", "raster_holder", "identity_asset", "badge", "icon"}
        categories = {slot.category for slot in enabled}
        missing = sorted(required_categories - categories)
        if missing:
            raise ValueError(f"style reference is missing required enabled categories: {missing}")

        identity_slots = [slot for slot in enabled if slot.category == "identity_asset"]
        if len(identity_slots) != 3:
            raise ValueError("style reference requires exactly three enabled identity_asset slots")

        for index, left in enumerate(enabled):
            for right in enabled[index + 1 :]:
                if _overlap(left.bbox, right.bbox):
                    raise ValueError(f"enabled style asset slots overlap: {left.slot_id}, {right.slot_id}")
        return self


def _overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])
