from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = """Create a polished, visually rich and visually creative 16:9 presentation slide composition.

Use "{title}" as the only readable heading.

The composition should help the audience understand this idea without quoting it: {purpose}

Roughly organize the imagery and text-shaped negative space this way without quoting it: {visual_logic}

Use only the title, purpose, and visual logic as open-ended semantic guidance.
Explore the overall visual mass, rhythm, hierarchy, negative space, and
explanatory flow. Keep the visual logic rough rather than turning it into a
rigid grid. Do not render instruction wording, field labels, body copy, captions,
or other readable text beyond the heading. No watermark.
"""


def build_prompt(slide_spec: dict) -> str:
    return PROMPT_TEMPLATE.format(
        title=slide_spec["title"],
        purpose=slide_spec["purpose"],
        visual_logic=slide_spec["visual_logic"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Z-Image-Turbo prompts from lightweight slide inputs."
    )
    parser.add_argument("--slide-spec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_paths = sorted(args.slide_spec_dir.glob("slides/slide_*/slide_spec.json"))
    if not source_paths:
        raise FileNotFoundError("no slide_spec.json artifacts found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slides = []
    for source_path in source_paths:
        slide_spec = json.loads(source_path.read_text(encoding="utf-8"))
        slide_id = slide_spec["slide_id"]
        prompt_path = args.output_dir / f"{slide_id}.txt"
        prompt_path.write_text(build_prompt(slide_spec), encoding="utf-8")
        slides.append(
            {
                "slide_id": slide_id,
                "slide_spec": str(source_path.resolve()),
                "prompt": str(prompt_path.resolve()),
            }
        )

    manifest = {
        "stage": "slide_spec_to_presentation_prompt",
        "llm_calls": 0,
        "template": PROMPT_TEMPLATE,
        "included_fields": ["title", "purpose", "visual_logic"],
        "slide_count": len(slides),
        "slides": slides,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
