"""Checkpoints for the four existing deck-wide planning operations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .page_runtime import fingerprint_paths, run_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "step1": ("src/v9_deck_planning/pipeline.py", "src/v9_deck_planning/planner.py", "src/v9_deck_planning/models.py"),
    "text_generation": ("integration/text_generation.py", "src/v9_deck_planning/models.py"),
    "step2": ("src/v9_deck_planning/style_pipeline.py", "src/v9_deck_planning/style_image_backend.py",
              "src/v9_deck_planning/image_task_journal.py", "src/v9_deck_planning/models.py"),
    "expression_logic": ("integration/expression_logic.py",),
}
REQUIRED = {
    "step1": ("input.json", "deck_plan.json", "run_manifest.json"),
    "text_generation": ("text_plan.json", "parsed_text_plan.json", "run_manifest.json"),
    "step2": ("input.json", "deck_plan.json", "canonical_background.png", "design_system.json", "run_manifest.json"),
    "expression_logic": ("expression_logic_plan.json", "run_manifest.json"),
}


def checkpoint_shared_stage(*, run_dir: Path, stage: str, source_inputs: list[Path],
                            settings: dict[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Reuse completed matching planning outputs on an ordinary CLI restart.

    Only explicit public settings and selected artifact/code files enter the
    fingerprint. Configuration files and credentials are never inspected.
    """
    directory = run_dir / stage
    def outputs(result: dict[str, Any]) -> list[Path]:
        paths = [directory / name for name in REQUIRED[stage]]
        paths.extend(Path(value) for value in result.get("artifacts", {}).values()
                     if isinstance(value, str) and value and Path(value).is_file())
        if stage == "expression_logic":
            paths.extend(Path(item["expression_logic"]) for item in result.get("slides", []))
        return list(dict.fromkeys(path.resolve() for path in paths))
    def execute() -> dict[str, Any]:
        result = fn()
        if result.get("status") != "succeeded":
            raise RuntimeError(f"Shared stage {stage} did not complete successfully")
        return result
    return run_checkpoint(output_dir=run_dir / "shared_planning", step=stage,
        fingerprint=fingerprint_paths(source_inputs + [ROOT / name for name in SOURCES[stage]]
                                      + [Path(__file__), ROOT / "src/v9_deck_planning/json_utils.py"],
                                      extra={"stage": stage, "settings": settings}),
        required_outputs=outputs, fn=execute)
