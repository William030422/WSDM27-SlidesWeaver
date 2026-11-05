from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from integration.topology import _resume_succeeded, _reuse_raw_if_available


ROOT = Path(__file__).resolve().parents[1]


class TopologyNormalizationTests(unittest.TestCase):
    def test_projects_legacy_response_and_ignores_unused_invalid_enums(self) -> None:
        source = ROOT / "reused/topology/extract_layout_identity_geometry_with_vlm.py"
        spec = importlib.util.spec_from_file_location("layout_identity_normalization", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {
            "slide_id": "slide_002",
            "layout_identity": {
                "reading_direction": "top_to_bottom_then_left_to_right",
            },
            "elements": [
                {
                    "element_id": "title_001",
                    "kind": "title",
                    "role": "title",
                    "bbox": [4, 2, 56, 5],
                    "line_count_estimate": 2,
                    "visual_form": "not_applicable",
                    "visual_weight": "dominant",
                    "confidence": 0.9,
                },
                {
                    "element_id": "visual_001",
                    "kind": "visual",
                    "role": "visual",
                    "bbox": [4, 10, 56, 22],
                    "visual_form": "illustration",
                    "visual_weight": "supporting",
                    "confidence": 0.8,
                },
            ],
            "units": [],
            "bands": [],
            "relations": [],
            "repeated_patterns": [
                {"direction": "mixed", "spacing": "symmetric corner placement"}
            ],
        }

        normalized, report = module.normalize_payload(payload, "slide_002")

        self.assertEqual(set(normalized), {"slide_id", "elements"})
        self.assertEqual(normalized["slide_id"], "slide_002")
        self.assertEqual(
            set(normalized["elements"][0]),
            {"element_id", "role", "bbox", "confidence"},
        )
        self.assertIn("repeated_patterns", report["dropped_top_level_fields"])
        self.assertTrue(report["changed"])

    def test_empty_response_uses_minimum_non_blocking_regions(self) -> None:
        source = ROOT / "reused/topology/extract_layout_identity_geometry_with_vlm.py"
        spec = importlib.util.spec_from_file_location("layout_identity_fallback", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        normalized, report = module.normalize_payload({}, "slide_003")

        self.assertEqual(
            [item["role"] for item in normalized["elements"]], ["title", "visual"]
        )
        self.assertTrue(report["fallbacks_applied"])

    def test_reuses_only_raw_responses_that_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-topology-resume-") as temporary:
            output_dir = Path(temporary)
            self.assertFalse(_reuse_raw_if_available(output_dir, "slide_001", True))
            raw = output_dir / "slide_001/vlm_raw_response.json"
            raw.parent.mkdir(parents=True)
            raw.write_text("{}", encoding="utf-8")
            self.assertTrue(_reuse_raw_if_available(output_dir, "slide_001", True))
            self.assertFalse(_reuse_raw_if_available(output_dir, "slide_001", False))

    def test_resumes_only_current_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-topology-success-") as temporary:
            output_dir = Path(temporary)
            slide_dir = output_dir / "slide_001"
            slide_dir.mkdir(parents=True)
            (slide_dir / "layout_identity.json").write_text("{}", encoding="utf-8")
            (slide_dir / "run_manifest.json").write_text(
                '{"slide_id":"slide_001","status":"succeeded","model_call_count":1}',
                encoding="utf-8",
            )

            resumed = _resume_succeeded(
                output_dir, "slide_001", "layout_identity.json"
            )

            self.assertIsNotNone(resumed)
            assert resumed is not None
            self.assertTrue(resumed["resumed"])
            self.assertEqual(resumed["model_call_count"], 0)
