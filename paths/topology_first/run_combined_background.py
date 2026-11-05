#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.text_assembly import build_combination_background_deck  # noqa: E402
from v9_deck_planning.json_utils import read_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild only the local combination-background PPTX from existing artifacts."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    step2_dir = output_dir / "step2"
    final_background_dir = output_dir / "final_background"
    assembly_dir = output_dir / "assembly"
    deck_plan = read_json(step2_dir / "deck_plan.json")
    slide_ids = [slide["slide_id"] for slide in deck_plan["slides"]]
    plans = {
        slide_id: read_json(assembly_dir / slide_id / "composition_plan.json")
        for slide_id in slide_ids
    }
    result = build_combination_background_deck(
        output_dir=assembly_dir,
        step2_dir=step2_dir,
        final_background_dir=final_background_dir,
        slide_ids=slide_ids,
        plans_by_slide=plans,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
