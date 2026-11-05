from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = """Create a polished and visually creative 16:9 presentation slide.

Title: "{title}"

The slide communicates that {message}

Rough visual intent, offered as open-ended inspiration: "{visual_intent}"

Use the title, message, and visual intent as semantic guidance, not as a fixed
composition. Interpret them freely in a way that fits the topic and requested
style. Choose any visual grammar, composition, and explanatory devices that
communicate the idea clearly. No watermark.
"""


def build_prompt(slide_spec: dict) -> str:
    return PROMPT_TEMPLATE.format(
        title=slide_spec["title"],
        message=slide_spec["message"],
        visual_intent=slide_spec.get(
            "visual_intent",
            "Invent a memorable visual interpretation of the teaching point.",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Z-Image-Turbo prompts from lightweight slide_spec.json files."
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
        "included_fields": ["title", "message", "visual_intent"],
        "visual_guidance": "free-form visual intent used as soft inspiration",
        "slide_count": len(slides),
        "slides": slides,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
