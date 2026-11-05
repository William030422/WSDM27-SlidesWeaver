#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.config import load_config  # noqa: E402
from integration.text_assembly import run_content_unit_binding  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and preview text-holder content units from an existing run."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--live-vlm",
        action="store_true",
        help="run the registration gate with the configured VLM API",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    slide_ids = sorted(
        path.parent.name
        for path in (source_run / "region_content" / "slides").glob(
            "slide_*/region_content_plan.json"
        )
    )
    if not slide_ids:
        raise ValueError(f"no region content found under {source_run}")
    manifest = run_content_unit_binding(
        region_content_dir=source_run / "region_content",
        step2_dir=source_run / "step2",
        topology_dir=source_run / "topology",
        final_background_dir=source_run / "final_background",
        output_dir=output_dir,
        slide_ids=slide_ids,
        config=load_config(ROOT),
        mock_vlm=not args.live_vlm,
        preview_only=not args.live_vlm,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
