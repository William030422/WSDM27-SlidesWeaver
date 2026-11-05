#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.config import load_config  # noqa: E402
from integration.text_assembly import run_reused_text_assembly  # noqa: E402
from v9_deck_planning.json_utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 4 text placement on one or more existing background sets."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--background-set",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each background set.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(ROOT)
    slide_ids = sorted(
        path.parent.name
        for path in (source_run / "region_content" / "slides").glob(
            "slide_*/region_content_plan.json"
        )
    )
    if not slide_ids:
        raise ValueError(f"no region content found under {source_run}")
    background_sets = [_parse_background_set(value) for value in args.background_set]
    manifests = []
    for name, background_dir in background_sets:
        missing = [
            slide_id
            for slide_id in slide_ids
            if not (background_dir / slide_id / "background.png").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{name} is missing backgrounds for: {', '.join(missing)}")
        deck_output = output_dir / name
        manifest = run_reused_text_assembly(
            region_content_dir=source_run / "region_content",
            step2_dir=source_run / "step2",
            topology_dir=source_run / "topology",
            final_background_dir=background_dir,
            output_dir=deck_output,
            slide_ids=slide_ids,
            config=config,
        )
        manifests.append(
            {
                "name": name,
                "background_dir": str(background_dir),
                "manifest": str((deck_output / "run_manifest.json").resolve()),
                "pptx": manifest["artifacts"]["pptx"],
            }
        )
    summary = {
        "stage": "step4_text_placement_matrix",
        "status": "succeeded",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "vlm_model": config.chat_model,
        "slide_ids": slide_ids,
        "decks": manifests,
    }
    write_json(output_dir / "run_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_background_set(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--background-set must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or "/" in name:
        raise ValueError(f"invalid background set name: {name!r}")
    return name, Path(raw_path).expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
