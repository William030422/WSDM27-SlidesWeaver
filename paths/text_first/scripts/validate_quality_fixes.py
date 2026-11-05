#!/usr/bin/env python3
"""Offline regression/E2E validation with public fixture configuration only."""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

PUBLIC_CONFIGURATION = {
    "APIMART_API_ROOT": "https://offline.invalid/v1",
    "APIMART_CHAT_MODEL": "gpt-5.4",
    "APIMART_IMAGE_MODEL": "gpt-image-2",
    "APIMART_IMAGE_SIZE": "1536x864",
    "APIMART_IMAGE_RESOLUTION": "1k",
    "APIMART_IMAGE_QUALITY": "medium",
    "APIMART_IMAGE_MAX_ATTEMPTS": "2",
    "APIMART_CHAT_MAX_TOKENS": "16000",
    "APIMART_VLM_MAX_TOKENS": "16000",
    "APIMART_TIMEOUT_SECONDS": "240",
    "APIMART_POLL_INITIAL_DELAY_SECONDS": "0",
    "APIMART_POLL_INTERVAL_SECONDS": "0",
    "APIMART_POLL_TIMEOUT_SECONDS": "1200",
}


def run_mock(output_dir: Path, inject_quality_findings: bool = False) -> dict:
    import run_pipeline
    from integration.config import load_config
    from integration.html_text_rework import canonical_committed_tokens
    from integration.text_generation import TEXT_FIELD_PATTERN
    from PIL import Image

    with patch("integration.config.load_env_file", return_value=PUBLIC_CONFIGURATION):
        config = load_config(ROOT)
    args = argparse.Namespace(
        input=None,
        content="Explain feedback control in exactly 3 slides for beginning engineering students.",
        style="Clean technical illustration",
        output_dir=output_dir.resolve(),
        mock_api=True, mock_images=False, mock_diffusion=True,
        slide_limit=None, start_at="step1", stop_after="assembly",
    )
    compiler = run_pipeline.run_html_to_coordinates

    def compile_with_findings(**kwargs):
        if inject_quality_findings:
            for slide_id in kwargs["slide_ids"]:
                path = kwargs["html_planning_dir"] / "slides" / slide_id / "layout.html"
                document = path.read_text(encoding="utf-8")
                # Deliberately displace the mock visual's entire module before commitment.
                document = document.replace('style="left:768px;top:174px;width:712px;height:646px"', 'style="left:1600px;top:174px;width:712px;height:646px"', 1)
                path.write_text(document, encoding="utf-8")
        result = compiler(**kwargs)
        if inject_quality_findings:
            for slide_id in kwargs["slide_ids"]:
                path = kwargs["output_dir"] / "slides" / slide_id / "html_layout.json"
                layout = json.loads(path.read_text())
                # A malformed holder support must use native fallback, not stop export.
                for region in layout["text_regions"]:
                    region.setdefault("typography", {})["color"] = "FFFFFF"
                body = next((item for item in layout["text_regions"] if item["role"] == "body"), None)
                if body is not None and not layout.get("holders"):
                    layout["holders"] = [{"holder_id": "quality_fixture_holder", "text_region_id": body["region_id"], "bbox_px": body["bbox_px"]}]
                path.write_text(json.dumps(layout), encoding="utf-8")
        return result

    def empty_asset_board(objects, key_color, output_path):
        Image.new("RGB", (1536, 864), key_color).save(output_path)

    with ExitStack() as stack:
        stack.enter_context(patch.object(run_pipeline, "load_config", return_value=config))
        stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
        stack.enter_context(patch.object(run_pipeline, "run_html_to_coordinates", side_effect=compile_with_findings))
        if inject_quality_findings:
            stack.enter_context(patch("integration.final_background.detect_text_safe_rect", return_value=None))
            stack.enter_context(patch("integration.final_background.render_mock_asset_board", side_effect=empty_asset_board))
        exit_code = run_pipeline.main()
    if exit_code != 0:
        raise RuntimeError(f"offline pipeline exited {exit_code}")
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    required = [output_dir / "assembly" / name for name in ("deck.pptx", "deck.pdf", "pptx_render_check.json")]
    if not all(path.is_file() for path in required):
        raise AssertionError(f"missing required final artifacts: {required}")
    if inject_quality_findings and (manifest["status"] != "succeeded" or manifest["quality_status"] != "degraded"):
        raise AssertionError("quality findings must retain successful output with honest diagnostics")
    committed = json.loads((output_dir / "text_generation" / "text_plan.json").read_text())["slides"]
    with ZipFile(output_dir / "assembly" / "deck.pptx") as archive:
        for index, slide in enumerate(committed, start=1):
            root = ElementTree.fromstring(archive.read(f"ppt/slides/slide{index}.xml"))
            actual = " ".join(node.text or "" for node in root.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"))
            fields = [slide["title"]] + ([slide["subtitle"]] if slide.get("subtitle") else [])
            fields.extend(value for key, value in slide.items() if TEXT_FIELD_PATTERN.fullmatch(key))
            if canonical_committed_tokens(actual) != canonical_committed_tokens("\n".join(fields)):
                raise AssertionError(f"slide {index} lost or changed committed editable text")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--mock-output", type=Path)
    parser.add_argument("--inject-quality-findings", action="store_true")
    args = parser.parse_args()
    if args.tests:
        with patch("integration.config.load_env_file", return_value=PUBLIC_CONFIGURATION):
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                return 1
    if args.mock_output:
        run_mock(args.mock_output.resolve(), args.inject_quality_findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
