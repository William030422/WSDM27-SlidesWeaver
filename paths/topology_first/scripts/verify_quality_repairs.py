#!/usr/bin/env python3
"""Run quality regressions and/or a complete mock deck without loading .env."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def offline_values() -> dict[str, str]:
    return {
        "APIMART_API_ROOT": "https://example.invalid/v1", "APIMART_CHAT_MODEL": "offline-test-model",
        "APIMART_IMAGE_MODEL": "gpt-image-2-official", "APIMART_IMAGE_SIZE": "1536x864",
        "APIMART_IMAGE_RESOLUTION": "1K", "APIMART_IMAGE_QUALITY": "medium",
        "APIMART_IMAGE_MAX_ATTEMPTS": "2", "APIMART_CHAT_MAX_TOKENS": "4096",
        "APIMART_VLM_MAX_TOKENS": "4096", "APIMART_TIMEOUT_SECONDS": "60",
        "APIMART_POLL_INITIAL_DELAY_SECONDS": "0", "APIMART_POLL_INTERVAL_SECONDS": "0",
        "APIMART_POLL_TIMEOUT_SECONDS": "60",
        "DIFFUSION_MODEL_PATH": str(Path.home() / "models" / "Tongyi-MAI_Z-Image-Turbo"),
        "DIFFUSION_IMAGE_SIZE": "1024x576", "DIFFUSION_STEPS": "5",
        "DIFFUSION_GUIDANCE_SCALE": "0", "DIFFUSION_SEED": "42",
        "PPT_TITLE_FONT": "Noto Sans", "PPT_BODY_FONT": "Noto Sans",
        "PPT_CJK_FONT": "Droid Sans Fallback",
    }


@contextmanager
def rich_mock_identity():
    """Supply observed regions to the normal offline topology/content stages."""
    source_dir = ROOT / "reused" / "topology"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    identity = importlib.import_module("extract_layout_identity_geometry_with_vlm")

    def extract_fixture(slide_id: str, output_dir: Path, reuse_raw: bool) -> dict:
        del reuse_raw
        slide_dir = output_dir / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "slide_id": slide_id,
            "elements": [
                {"element_id": "title_001", "role": "title", "bbox": [3, 2, 58, 4], "confidence": 1.0},
                {"element_id": "body_001", "role": "standalone_text", "bbox": [3, 7, 58, 6], "confidence": 1.0},
                {"element_id": "visual_001", "role": "visual", "bbox": [5, 16, 54, 12], "confidence": 1.0},
                {"element_id": "caption_001", "role": "caption", "bbox": [5, 29, 54, 5], "confidence": 1.0},
            ],
        }
        result = identity.RegionDetectionResult.model_validate(payload)
        identity.write_json(slide_dir / "layout_identity.json", result)
        identity.render_overlay(identity.REFERENCE_DIR / f"{slide_id}.png", result,
                                slide_dir / "layout_identity_overlay.png")
        manifest = {"slide_id": slide_id, "status": "succeeded", "quality_status": "mock",
                    "model_call_count": 0, "fixture": "title_body_visual_caption"}
        identity.write_json(slide_dir / "run_manifest.json", manifest)
        return manifest

    with patch.object(identity, "extract_slide", side_effect=extract_fixture):
        yield


def verify_rich_deck(output_dir: Path) -> None:
    """Assert actual PPTX editability and copy coverage, not only file existence."""
    assembly = output_dir / "assembly"
    compilation = json.loads((assembly / "deck_compilation.json").read_text())
    namespaces = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                  "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    normalize = lambda value: re.sub(r"\s+", " ", value).strip()
    records = []
    with ZipFile(assembly / "deck.pptx") as archive:
        for index, record in enumerate(compilation["slides"], start=1):
            slide_id = record["slide_id"]
            plan = json.loads(Path(record["plan"]).read_text())
            source = json.loads((output_dir / "region_content" / "slides" / slide_id / "region_content_plan.json").read_text())
            xml = ElementTree.fromstring(archive.read(f"ppt/slides/slide{index}.xml"))
            native_objects = [shape for shape in xml.findall(".//p:sp", namespaces)
                              if any((node.text or "").strip() for node in shape.findall(".//a:t", namespaces))]
            native_text = normalize(" ".join(node.text or "" for node in xml.findall(".//a:t", namespaces)))
            missing = [line for region in source["text_regions"] for line in region["lines"]
                       if normalize(line) not in native_text]
            holder_count = sum(item["presentation"]["mode"] == "raster_holder_text" for item in plan["elements"])
            if len(native_objects) < 2 or missing or holder_count < 1:
                raise AssertionError({"slide_id": slide_id, "native_objects": len(native_objects),
                                      "missing_copy": missing, "raster_holder_count": holder_count})
            records.append({"slide_id": slide_id, "native_text_object_count": len(native_objects),
                            "required_text_region_count": len(source["text_regions"]),
                            "required_lines_preserved": True, "raster_holder_count": holder_count,
                            "roles": [item["role"] for item in plan["elements"]]})
    report = {"status": "passed", "slide_count": len(records), "slides": records,
              "model_calls_added_by_fixture": 0}
    (output_dir / "rich_mock_coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rich_mock_coverage": report}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--rich-mock-fixture", action="store_true")
    parser.add_argument("--inject-quality-finding", action="store_true")
    parser.add_argument("--style", default="clean academic with clear captions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "engineering_quality_fix_mock")
    args = parser.parse_args()
    if not args.tests and not args.pipeline:
        parser.error("choose --tests and/or --pipeline")
    if os.environ.get("CONDA_DEFAULT_ENV") != "rl4ppt-final":
        parser.error("activate rl4ppt-final before running verification")
    import integration.config as configuration
    with patch.object(configuration, "load_env_file", return_value=offline_values()):
        if args.tests:
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful():
                return 1
        if args.pipeline:
            import run_pipeline
            from contextlib import nullcontext
            import integration.text_assembly as assembly
            load_placement = assembly._load_text_placement
            def with_quality_finding():
                placement = load_placement()
                original_check = placement._validate_group_invariants
                placement._validate_group_invariants = lambda plan: original_check(plan) or "injected_quality_finding"
                return placement
            with patch.object(sys, "argv", [
                "run_pipeline.py", "--content",
                "Create 2 slides explaining model-free reinforcement learning and the explore/exploit tradeoff.",
                "--style", args.style, "--output-dir", str(args.output_dir),
                "--mock-api", "--mock-diffusion",
            ]), (patch.object(assembly, "_load_text_placement", side_effect=with_quality_finding)
                 if args.inject_quality_finding else nullcontext()), (
                     rich_mock_identity() if args.rich_mock_fixture else nullcontext()
                 ):
                code = run_pipeline.main()
            if code == 0 and args.rich_mock_fixture:
                verify_rich_deck(args.output_dir.resolve())
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
