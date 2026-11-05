from __future__ import annotations

import re

from .models import DeckPlan, UserInput
from .style_models import AssetCategory, AssetShape, StyleAssetSlot, StyleReferencePlan


SlotSpec = tuple[
    str,
    AssetCategory,
    AssetShape,
    tuple[float, float, float, float],
    str,
    str,
]


_MAIN_SLOTS: tuple[SlotSpec, ...] = (
    (
        "title_bar_primary",
        "title_bar",
        "bar",
        (0.025, 0.035, 0.700, 0.180),
        "One substantial tall blank title bar with a calm writable interior and style-defining edge treatment.",
        "Resize proportionally behind an editable slide title.",
    ),
    (
        "subtitle_bar_primary",
        "subtitle_bar",
        "bar",
        (0.060, 0.205, 0.620, 0.305),
        "One substantial blank subtitle bar, visibly shorter and quieter than the title bar but not a thin rule.",
        "Resize proportionally behind an editable subtitle or section label.",
    ),
    (
        "holder_wide",
        "raster_holder",
        "rounded_rect",
        (0.025, 0.340, 0.350, 0.515),
        "A wide empty holder with generous undecorated interior and restrained decoration at the perimeter.",
        "Candidate for a short editable explanation or comparison row.",
    ),
    (
        "holder_compact",
        "raster_holder",
        "rounded_rect",
        (0.372, 0.340, 0.545, 0.515),
        "A compact empty card holder with a different silhouette from the wide holder.",
        "Candidate for a compact editable fact, definition, or note.",
    ),
    (
        "holder_circle",
        "raster_holder",
        "circle",
        (0.572, 0.325, 0.684, 0.525),
        "A circular empty holder with a large calm center and decoration confined to the rim.",
        "Candidate for a short editable keyword, number, or focal annotation.",
    ),
    (
        "holder_speech",
        "raster_holder",
        "speech_bubble",
        (0.025, 0.555, 0.231, 0.740),
        "An empty speech-bubble holder with a clear tail and a broad writable center.",
        "Candidate for editable dialogue or a concise claim.",
    ),
    (
        "holder_callout",
        "raster_holder",
        "callout",
        (0.256, 0.555, 0.462, 0.740),
        "An empty callout holder with a distinct asymmetric pointer and calm writable center.",
        "Candidate for an editable annotation tied to a visual entity.",
    ),
    (
        "holder_pill",
        "raster_holder",
        "pill",
        (0.487, 0.590, 0.700, 0.705),
        "A long empty pill holder with minimal edge ornament and high internal contrast.",
        "Candidate for an editable takeaway, stage label, or short caption.",
    ),
)


_BADGES = (
    ("badge_emphasis", "emphasis, rank, or importance"),
    ("badge_evidence", "evidence, verification, or authority"),
    ("badge_warning", "warning, risk, or misconception"),
    ("badge_milestone", "milestone, achievement, or progress"),
    ("badge_category", "category, faction, or section identity"),
    ("badge_takeaway", "takeaway, conclusion, or recommended action"),
)


_ICONS = (
    ("icon_concept", "a core concept or idea"),
    ("icon_process", "process, transition, or iteration"),
    ("icon_evidence", "evidence, observation, or measurement"),
    ("icon_relationship", "relationship, connection, or comparison"),
    ("icon_warning", "caution, misconception, or constraint"),
    ("icon_takeaway", "conclusion, direction, or action"),
    ("icon_model", "a model, simulation, or internal representation"),
    ("icon_environment", "an environment, world, or system"),
    ("icon_planning", "planning, search, or decision making"),
)


def build_style_reference_plan(user_input: UserInput, deck_plan: DeckPlan) -> StyleReferencePlan:
    if deck_plan.style != user_input.style:
        raise ValueError("deck plan style must match the source input style")
    include_avatar, avatar_reason = avatar_policy(user_input)
    slots = [*(_slot(*spec) for spec in _MAIN_SLOTS)]
    slots.extend(_identity_slots(include_avatar))
    slots.extend(_small_asset_slots())
    return StyleReferencePlan(
        source_style=user_input.style,
        key_color=select_key_color(user_input.style),
        avatar_policy="included" if include_avatar else "fallback",
        avatar_policy_reason=avatar_reason,
        slots=slots,
    )


def build_style_reference_prompt(
    user_input: UserInput,
    deck_plan: DeckPlan,
    plan: StyleReferencePlan,
    *,
    image_model: str | None = None,
) -> str:
    if plan.source_style != user_input.style:
        raise ValueError("style reference plan must preserve the exact input style")
    if image_model and image_model.startswith("qwen-image-2.0"):
        return _build_qwen_prompt(user_input, plan)
    if image_model and image_model.startswith("gemini-3.1-flash-image-preview"):
        return _build_nano_prompt(user_input, plan)
    return _build_full_prompt(user_input, deck_plan, plan)


def build_background_prompt(user_input: UserInput, deck_plan: DeckPlan) -> str:
    if deck_plan.style != user_input.style:
        raise ValueError("deck plan style must match the source input style")
    return f"""Create one neat, reusable 16:9 background for an entire presentation deck.

Match only the visual language of this style -- palette, materials, line work, lighting, and surface treatment:
{user_input.style}

Deck topic, as a weak atmospheric hint only:
{deck_plan.topic}

This is a background, not an illustration. Keep at least 90 percent of the full canvas as one broad, contiguous,
low-detail negative-space field suitable for later text and teaching visuals. All decoration combined must occupy
no more than 10 percent of the canvas. Keep it at the thin outer perimeter, leaving quiet space across the whole canvas.
Use at most two small abstract edge accents. Do not draw characters, people, rooms, neighborhoods, gadgets, doors,
vehicles, scenes, or other recognizable objects. Any decoration must remain sparse, peripheral, and subordinate.
Keep the composition clean, orderly, quiet, and spacious, with restrained decoration and broad low-detail areas.
Keep the background largely free of semantic information. Do not depict topic-specific objects, teaching entities,
diagrams, charts, data, or semantic relationships.
Do not include text, labels, slide-specific content, fixed cards, panels, placeholders, logos, or watermarks.

Return only the background image.
"""


def _build_full_prompt(user_input: UserInput, deck_plan: DeckPlan, plan: StyleReferencePlan) -> str:
    slot_lines = []
    for slot in plan.slots:
        x0, y0, x1, y1 = slot.bbox
        slot_lines.append(
            f"- {slot.slot_id} [{slot.category}/{slot.shape}] at x={x0:.3f}-{x1:.3f}, "
            f"y={y0:.3f}-{y1:.3f}: {slot.generation_instruction}"
        )

    return f"""Generate one 16:9 reusable style-asset reference atlas for a presentation deck.

This is a style board and asset sheet, not a finished slide. Later image-generation calls will use it as the exact
deck-wide visual reference, and its bounded regions will be cropped into optional reusable assets.

User request, verbatim:
{user_input.content}

Shared style direction, verbatim:
{user_input.style}

Do not blur, genericize, sanitize, substitute, or omit an explicitly requested named style, universe, or avatar.

Deck context, used only to keep the reusable motifs relevant:
- topic: {deck_plan.topic}
- audience: {deck_plan.audience}
- deck goal: {deck_plan.deck_goal}

The source canvas is perfectly flat {plan.key_color}. Paint one complete isolated asset inside every listed region and
keep a visible {plan.key_color} gutter between assets. Stay inside the stated ranges.

Normalized slot map (0.0-1.0 coordinates on a {plan.board_size[0]}x{plan.board_size[1]} board):
{chr(10).join(slot_lines)}

Identity policy:
{_identity_instruction(user_input, plan)}

Global rules:
- Use one coherent palette, material language, line quality, lighting, edge treatment, and rendering style.
- Make title/subtitle bars substantial blank surfaces, not thin sticks or divider lines.
- Make every holder blank with a broad quiet interior; keep ornament and texture at the perimeter.
- Badge and icon regions are open canvases: use varied freeform silhouettes, not uniform circles, shields, square tiles,
  rounded-square cards, or repeated enclosing frames unless the requested style itself requires one.
- Keep every badge/icon isolated on the chroma background so it can be extracted independently.
- Keep every part of one asset close together and add a continuous contrasting outer keyline.
- Do not use {plan.key_color} inside an asset.
- Do not add shadows, glow, transparency, motion blur, or soft color bleeding.

Strictly forbidden:
- readable text, letters, numbers, pseudo-text, captions, labels, equations, readable wordmarks, or watermarks
- final slide content, charts, UI screenshots, or a full presentation layout
- extra assets outside the regions, background scenes, or decoration bridging the chroma-key gutters
"""


def _build_qwen_prompt(user_input: UserInput, plan: StyleReferencePlan) -> str:
    identity = _compact_identity(plan)
    prompt = (
        f"NO TEXT. Unused pixels are flat {plan.key_color}; no paper or headings. "
        f"Request, never print: {user_input.content} "
        f"Style, never print: {user_input.style} "
        "Keep named style/avatar exact. Do not blur or omit. "
        "Left 70%: thick blank title/subtitle bars + 6 empty holders. "
        f"Bottom: 3 identity cells with {identity}. "
        "Right: 6 freeform badges + 9 freeform icons, isolated, no uniform tiles/cards. "
        "Clean gutters; contrasting outlines; no key color inside assets. "
        "No words, letters, numbers, labels, wordmarks, or watermarks."
    )
    if len(prompt) > 800:
        raise ValueError(
            f"qwen-image-2.0 prompt would be {len(prompt)} characters; preserving the exact request/style exceeds 800"
        )
    return prompt


def _build_nano_prompt(user_input: UserInput, plan: StyleReferencePlan) -> str:
    return f"""IMAGE ONLY. Never render headings, slot names, labels, captions, letters, numbers, pseudo-text, or watermarks.

Keep every unused pixel as the perfectly flat solid chroma color {plan.key_color}. Do not create white paper, guide
lines, a background board, or text in the gutters.

Apply this exact user direction semantically, but never typeset it:
{user_input.content}

Apply this exact style semantically, but never typeset it:
{user_input.style}

Do not blur, genericize, sanitize, substitute, or omit any named style, universe, or avatar cue.

Create one coherent 16:9 reusable presentation asset sheet:
- left 70 percent: one thick blank title bar, one thick blank subtitle bar, and six varied empty text holders
- bottom left: three identity assets with {_compact_identity(plan)}
- right 25 percent: six varied freeform badges and nine varied freeform icons

Keep all assets isolated with generous {plan.key_color} gutters. Badge/icon silhouettes must be varied and freeform,
without uniform circles, shields, square tiles, rounded cards, enclosing frames, or repeated containers. Holders and bars
must have broad calm blank interiors. Add a continuous contrasting outer keyline, never use {plan.key_color} inside an
asset, and avoid shadows, glow, transparency, motion blur, or soft color bleeding. Use one consistent rendering style.
"""


def _compact_identity(plan: StyleReferencePlan) -> str:
    if plan.avatar_policy == "included":
        return "the same explicitly requested recognizable avatar in 3 consistent front, explaining, and reacting poses"
    return "3 coherent thematic hero objects; do not invent a character"


def avatar_policy(user_input: UserInput) -> tuple[bool, str]:
    text = f"{user_input.content}\n{user_input.style}".lower()
    subject = r"avatars?|mascots?|characters?|people|persons?|humans?|teachers?|students?|robots?|presenters?|hosts?|guides?"
    if re.search(rf"\b(?:no|without|avoid|exclude)\s+(?:any\s+)?(?:{subject})\b", text):
        return False, "The request explicitly excludes character-like assets; identity slots use thematic objects."
    if re.search(rf"\b(?:{subject})\b", text):
        return True, "The request explicitly names a character-like subject or recurring avatar role."

    franchise_cues = (
        r"star\s+wars|doraemon|doreamon|sith|jedi|moff|imperial|holocron|marvel|dc\s+comics|"
        r"pokemon|pok[eé]mon|mario|ghibli|disney|pixar|minecraft|lord\s+of\s+the\s+rings|harry\s+potter"
    )
    if re.search(franchise_cues, text):
        return True, "The request explicitly names a character-driven franchise, universe, or identity aesthetic."
    return False, "No explicit avatar, character, or character-driven franchise cue was found."


def select_key_color(style: str) -> str:
    lowered = style.lower()
    if re.search(r"\b(?:green|lime|emerald|chartreuse|minecraft)\b", lowered):
        if re.search(r"\b(?:magenta|fuchsia|pink|purple)\b", lowered):
            return "#0000ff"
        return "#ff00ff"
    return "#00ff00"


def _identity_instruction(user_input: UserInput, plan: StyleReferencePlan) -> str:
    if plan.avatar_policy == "included":
        return (
            "The user explicitly requested a named character/franchise or avatar-bearing identity. Use that exact "
            f"direction from the verbatim request/style ({user_input.content} / {user_input.style}). Fill all three "
            "identity regions with the same recognizable requested avatar or directly requested representative identity "
            "in front, explaining, and reacting poses. Keep face, costume, proportions, materials, and colors consistent."
        )
    return (
        "No avatar is explicitly requested. Use the three reserved identity regions for coherent thematic hero objects "
        "or motifs instead, without inventing a person, mascot, face, animal, or character."
    )


def _slot(
    slot_id: str,
    category: AssetCategory,
    shape: AssetShape,
    bbox: tuple[float, float, float, float],
    generation_instruction: str,
    reuse_hint: str,
) -> StyleAssetSlot:
    return StyleAssetSlot(
        slot_id=slot_id,
        category=category,
        shape=shape,
        bbox=bbox,
        generation_instruction=generation_instruction,
        reuse_hint=reuse_hint,
    )


def _identity_slots(include_avatar: bool) -> list[StyleAssetSlot]:
    boxes = (
        (0.030, 0.770, 0.220, 0.970),
        (0.255, 0.770, 0.445, 0.970),
        (0.480, 0.770, 0.670, 0.970),
    )
    if include_avatar:
        instructions = (
            "The requested recurring avatar or representative identity in a canonical front-facing pose.",
            "The exact same requested avatar or identity in an explaining or pointing pose.",
            "The exact same requested avatar or identity in a thoughtful or reacting pose.",
        )
    else:
        instructions = (
            "A primary isolated thematic hero object demonstrating the requested style at a larger scale.",
            "A second isolated thematic object with a distinct silhouette but identical rendering language.",
            "A simpler isolated thematic accent or connective motif; do not invent a character.",
        )
    names = ("identity_primary", "identity_variant_02", "identity_variant_03")
    hints = (
        "Canonical identity reference for later selection.",
        "Secondary identity pose or thematic object for later selection.",
        "Reaction pose or thematic accent for later selection.",
    )
    return [
        _slot(name, "identity_asset", "freeform", box, instruction, hint)
        for name, box, instruction, hint in zip(names, boxes, instructions, hints)
    ]


def _small_asset_slots() -> list[StyleAssetSlot]:
    columns = ((0.750, 0.815), (0.830, 0.895), (0.910, 0.975))
    rows = ((0.030, 0.170), (0.220, 0.360), (0.410, 0.550), (0.600, 0.740), (0.790, 0.930))
    specs = [
        (name, "badge", meaning, "Optional freeform badge; select only when its semantics fit.")
        for name, meaning in _BADGES
    ]
    specs.extend(
        (name, "icon", meaning, "Optional freeform icon; select only when its semantics fit.")
        for name, meaning in _ICONS
    )
    slots = []
    for index, (name, category, meaning, reuse_hint) in enumerate(specs):
        column = columns[index % 3]
        row = rows[index // 3]
        instruction = (
            f"One isolated unlabeled freeform {category} suggesting {meaning}; vary its silhouette and do not enclose "
            "it in a standard tile or card."
        )
        slots.append(_slot(name, category, "freeform", (column[0], row[0], column[1], row[1]), instruction, reuse_hint))
    return slots
